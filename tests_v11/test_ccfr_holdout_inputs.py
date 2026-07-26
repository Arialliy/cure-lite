from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from cure_lite.decoder import project_occupancy_to_feature_grid
from cure_lite.ccfr_holdout_inputs import (
    ADJACENT_CELL_FAMILY,
    CLEAN_PAIR_COUNT,
    CLEAN_SLOT_COUNT,
    COMPONENT_NULL_PAIR_COUNT,
    COMPONENT_NULL_SLOT_COUNT,
    DESIGN_SEED,
    FACTUAL_BATCH_SIZE,
    FACTUAL_EXPOSURES_PER_STATE,
    FACTUAL_POPULATION_SIZE,
    FEATURE_CHANNELS,
    FEATURE_HEIGHT,
    FEATURE_STRIDE,
    FEATURE_WIDTH,
    GROUP_COUNTS,
    MULTICOUNT_2TO1_FAMILY,
    MULTICOUNT_3TO2_FAMILY,
    ONE_PIXEL_PHASE_PATTERN,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    SAME_CELL_FAMILY,
    THREE_PIXEL_PHASE_PATTERN,
    TOTAL_PAIR_SLOTS,
    UPDATE_COUNT,
    build_ccfr_holdout_factual_batches,
    build_ccfr_holdout_factual_population,
    build_ccfr_holdout_outcome_batch,
    build_ccfr_holdout_pair_specs,
    build_ccfr_holdout_schedule,
    build_ccfr_holdout_strata,
    catalog_fingerprint,
    catalog_manifest,
    factual_indices_for_update,
    factual_population_manifest,
    factual_schedule_fingerprint,
    holdout_fingerprint,
    holdout_manifest,
    schedule_fingerprint,
)
from cure_lite.experiment.conservative_toy_inputs import (
    CONSERVATIVE_TOY_CASES,
    build_conservative_toy_case,
)
from cure_lite.experiment.peco_exposure_confirmation import (
    build_confirmation_factual_population,
    build_confirmation_outcome_batch,
    build_confirmation_pair_specs,
    schedule_fingerprint as old_v10_schedule_fingerprint,
)
from cure_lite.paired_types import tensor_content_fingerprint


ROOT = Path(__file__).resolve().parents[1]
DESIGN_RECEIPT = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "coverage_conditioned_feature_release_v11"
    / "exposure_holdout_design_receipt.json"
)

EXPECTED_HOLDOUT_FINGERPRINT = (
    "3b81cc8cfd4d156ff6b711b1f8163dbb7ee2395d0d7934c425be33bee96ef1e2"
)


def _load_receipt() -> dict[str, object]:
    payload = json.loads(DESIGN_RECEIPT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _outcome_row_fingerprints(outcome: object) -> set[str]:
    pair_batch = outcome.pair_batch
    tensors = (
        pair_batch.feature,
        pair_batch.occupancy_plus,
        pair_batch.occupancy_minus,
        pair_batch.label_increment,
        pair_batch.image_valid_mask,
        outcome.completion_plus,
        outcome.completion_minus,
        outcome.gt_union,
        outcome.intervention_footprint,
    )
    return {
        tensor_content_fingerprint(tensor[index : index + 1])
        for tensor in tensors
        for index in range(tensor.shape[0])
    }


def _branch_row_fingerprints(branches: dict[str, object]) -> set[str]:
    values: set[str] = set()
    for batch in branches.values():
        for tensor in (
            batch.feature,
            batch.occupancy,
            batch.target,
            batch.valid_mask,
        ):
            values.update(
                tensor_content_fingerprint(tensor[index : index + 1])
                for index in range(tensor.shape[0])
            )
    return values


def test_constants_and_population_match_the_frozen_receipt() -> None:
    receipt = _load_receipt()
    contract = receipt["input_contract"]
    population_receipt = receipt["population"]
    schedule_receipt = receipt["schedule"]
    factual_receipt = receipt["factual_population"]

    assert DESIGN_SEED == receipt["design_seed"] == int("edad8c9c", 16)
    assert (
        FEATURE_CHANNELS,
        FEATURE_HEIGHT,
        FEATURE_WIDTH,
        FEATURE_STRIDE,
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH,
    ) == (
        contract["feature_channels"],
        contract["feature_height"],
        contract["feature_width"],
        contract["feature_stride"],
        contract["output_height"],
        contract["output_width"],
    )
    assert CLEAN_PAIR_COUNT == population_receipt["clean_pair_count"] == 206
    assert (
        COMPONENT_NULL_PAIR_COUNT
        == population_receipt["component_null_pair_count"]
        == 16
    )
    assert GROUP_COUNTS == population_receipt["groups"]
    assert UPDATE_COUNT == schedule_receipt["updates"] == 400
    assert TOTAL_PAIR_SLOTS == schedule_receipt["total_pair_slots"] == 800
    assert CLEAN_SLOT_COUNT == schedule_receipt["clean_slots"] == 739
    assert (
        COMPONENT_NULL_SLOT_COUNT
        == schedule_receipt["component_null_slots"]
        == 61
    )
    assert FACTUAL_POPULATION_SIZE == factual_receipt["factual_miss_states"]
    assert FACTUAL_POPULATION_SIZE == factual_receipt["factual_no_miss_states"]
    assert FACTUAL_BATCH_SIZE == factual_receipt["batch_size_per_branch"]
    assert FACTUAL_EXPOSURES_PER_STATE == factual_receipt[
        "exposures_per_state"
    ]

    specs = build_ccfr_holdout_pair_specs()
    assert len(specs) == 222
    assert Counter(spec.group_id for spec in specs) == Counter(GROUP_COUNTS)
    assert len({spec.pair_id for spec in specs}) == len(specs)
    assert len({spec.sample_id for spec in specs}) == len(specs)
    assert all(1 <= value <= 3 for spec in specs for value in spec.component_cell)
    assert all(1 <= value <= 3 for spec in specs for value in spec.response_cell)
    assert all(
        1 <= value <= 3
        for spec in specs
        for cell in spec.fixed_occupancy_cells
        for value in cell
    )
    assert Counter(
        spec.exposure_count
        for spec in specs
        if spec.pair_kind == "clean_positive"
    ) == Counter({3: 85, 4: 121})
    assert Counter(
        spec.exposure_count
        for spec in specs
        if spec.pair_kind == "component_null"
    ) == Counter({3: 3, 4: 13})


def test_sha256_integer_lattice_and_signal_channel_contract() -> None:
    specs = build_ccfr_holdout_pair_specs()
    selected = (
        specs[0],
        next(spec for spec in specs if spec.pair_kind == "component_null"),
    )
    outcome = build_ccfr_holdout_outcome_batch(selected)
    feature = outcome.pair_batch.feature

    for row_index, spec in enumerate(selected):
        cell = (
            spec.response_cell
            if spec.pair_kind == "clean_positive"
            else spec.component_cell
        )
        signal_channels = (
            (0, 1)
            if spec.pair_kind == "clean_positive"
            else (2, 3)
        )
        for channel in signal_channels:
            key = (
                f"{DESIGN_SEED}|{spec.group_id}|{spec.within_group_index}|"
                f"{channel}|{cell[0]}|{cell[1]}"
            )
            unsigned = int.from_bytes(
                hashlib.sha256(key.encode("utf-8")).digest()[:2],
                byteorder="big",
                signed=False,
            )
            expected = float((unsigned % 257) - 128) / 64.0
            assert feature[row_index, channel, cell[0], cell[1]].item() == expected

        nonzero_channels = set(
            torch.nonzero(
                feature[row_index].abs().flatten(1).sum(dim=1),
                as_tuple=False,
            )
            .flatten()
            .tolist()
        )
        assert nonzero_channels == set(signal_channels)


def test_all_pair_geometry_and_four_strata_satisfy_the_receipt() -> None:
    specs = build_ccfr_holdout_pair_specs()
    outcome = build_ccfr_holdout_outcome_batch(specs)
    batch = outcome.pair_batch
    strata = build_ccfr_holdout_strata(outcome)

    assert batch.feature.shape == (222, 8, 5, 5)
    assert batch.occupancy_plus.shape == (222, 1, 20, 20)
    assert batch.occupancy_minus.shape == (222, 1, 20, 20)
    assert outcome.completion_plus.shape == (222, 1, 20, 20)
    assert torch.all(strata.H.flatten(1).any(dim=1))
    assert torch.all(strata.G_near.flatten(1).any(dim=1))
    assert torch.all(strata.G_norm_tail.flatten(1).any(dim=1))
    assert torch.equal(
        strata.D | strata.H | strata.G_near | strata.G_norm_tail,
        batch.image_valid_mask,
    )

    projected_plus = project_occupancy_to_feature_grid(
        batch.occupancy_plus,
        (5, 5),
    )
    projected_minus = project_occupancy_to_feature_grid(
        batch.occupancy_minus,
        (5, 5),
    )
    assert torch.all(
        (projected_plus ^ projected_minus).flatten(1).any(dim=1)
    )
    kernel = torch.ones(1, 1, 3, 3)
    count_plus = F.conv2d(projected_plus.float(), kernel, padding=1)
    count_minus = F.conv2d(projected_minus.float(), kernel, padding=1)

    for index, spec in enumerate(specs):
        component_row, component_column = spec.component_cell
        response_row, response_column = spec.response_cell
        if spec.geometry_family == ADJACENT_CELL_FAMILY:
            assert (
                abs(component_row - response_row)
                + abs(component_column - response_column)
                == 1
            )
        else:
            assert spec.response_cell == spec.component_cell

        response = strata.D[index, 0]
        response_pixels = {
            (int(row), int(column))
            for row, column in torch.nonzero(response, as_tuple=False).tolist()
        }
        if spec.pair_kind == "component_null":
            assert not response_pixels
            continue

        phases = (
            ONE_PIXEL_PHASE_PATTERN
            if spec.response_pixel_count == 1
            else THREE_PIXEL_PHASE_PATTERN
        )
        expected_pixels = {
            (
                FEATURE_STRIDE * response_row + phase_row,
                FEATURE_STRIDE * response_column + phase_column,
            )
            for phase_row, phase_column in phases
        }
        assert response_pixels == expected_pixels
        assert all(
            row // FEATURE_STRIDE == response_row
            and column // FEATURE_STRIDE == response_column
            for row, column in response_pixels
        )

        plus = int(
            count_plus[index, 0, response_row, response_column].item()
        )
        minus = int(
            count_minus[index, 0, response_row, response_column].item()
        )
        if spec.geometry_family in {SAME_CELL_FAMILY, ADJACENT_CELL_FAMILY}:
            assert (plus, minus) == (1, 0)
        elif spec.geometry_family == MULTICOUNT_2TO1_FAMILY:
            assert (plus, minus) == (2, 1)
        elif spec.geometry_family == MULTICOUNT_3TO2_FAMILY:
            assert (plus, minus) == (3, 2)
        else:
            raise AssertionError("unexpected clean geometry family")


def test_new_schedule_has_exact_exposures_and_no_within_update_reuse() -> None:
    specs = build_ccfr_holdout_pair_specs()
    updates = build_ccfr_holdout_schedule(specs)
    by_index = {spec.population_index: spec for spec in specs}

    assert len(updates) == UPDATE_COUNT
    assert sum(len(update.population_indices) for update in updates) == 800
    assert all(len(set(update.pair_ids)) == 2 for update in updates)
    assert all(len(set(update.sample_ids)) == 2 for update in updates)
    observed = Counter(
        index for update in updates for index in update.population_indices
    )
    assert observed == Counter(
        {
            spec.population_index: spec.exposure_count
            for spec in specs
        }
    )
    assert set(observed.values()) == {3, 4}
    clean_slots = sum(
        by_index[index].pair_kind == "clean_positive"
        for update in updates
        for index in update.population_indices
    )
    assert clean_slots == 739
    assert 800 - clean_slots == 61

    assert schedule_fingerprint(specs) == schedule_fingerprint(
        build_ccfr_holdout_pair_specs()
    )
    assert schedule_fingerprint(specs) != old_v10_schedule_fingerprint()


def test_factual_population_and_rotation_give_every_state_100_exposures() -> None:
    populations = build_ccfr_holdout_factual_population()
    assert set(populations) == {"factual_miss", "factual_no_miss"}
    for branch, batch in populations.items():
        batch.validate(expected_branch=branch)
        assert batch.feature.shape == (16, 8, 5, 5)
        assert batch.occupancy.shape == (16, 1, 20, 20)
        assert batch.target.shape == (16, 1, 20, 20)
    assert set(
        torch.nonzero(
            populations["factual_miss"].feature.abs().sum(dim=(0, 2, 3)),
            as_tuple=False,
        )
        .flatten()
        .tolist()
    ) == {0, 1}
    assert set(
        torch.nonzero(
            populations["factual_no_miss"].feature.abs().sum(dim=(0, 2, 3)),
            as_tuple=False,
        )
        .flatten()
        .tolist()
    ) == {4, 5}

    assert factual_indices_for_update(0) == (0, 1, 2, 3)
    assert factual_indices_for_update(1) == (1, 2, 3, 4)
    assert factual_indices_for_update(15) == (15, 0, 1, 2)
    exposures = Counter(
        index
        for update_index in range(UPDATE_COUNT)
        for index in factual_indices_for_update(update_index)
    )
    assert exposures == Counter(
        {index: FACTUAL_EXPOSURES_PER_STATE for index in range(16)}
    )
    batches = build_ccfr_holdout_factual_batches(update_index=399)
    for branch, batch in batches.items():
        batch.validate(expected_branch=branch)
        assert batch.feature.shape[0] == FACTUAL_BATCH_SIZE
    assert factual_schedule_fingerprint() != (
        __import__(
            "cure_lite.experiment.peco_exposure_confirmation",
            fromlist=["factual_schedule_fingerprint"],
        ).factual_schedule_fingerprint()
    )


def test_no_tensor_is_reused_from_old_six_case_or_v10_populations() -> None:
    new_fingerprints = {
        fingerprint
        for row in catalog_manifest()
        for fingerprint in row["tensor_fingerprints"].values()
    }
    new_fingerprints.update(
        fingerprint
        for row in factual_population_manifest()
        for fingerprint in row["tensor_fingerprints"].values()
    )

    old_fingerprints: set[str] = set()
    for family_id, _case_id, pixels in CONSERVATIVE_TOY_CASES:
        outcome, factual = build_conservative_toy_case(family_id, pixels)
        old_fingerprints.update(_outcome_row_fingerprints(outcome))
        old_fingerprints.update(_branch_row_fingerprints(factual))

    old_specs = build_confirmation_pair_specs()
    for index in range(0, len(old_specs), 2):
        outcome = build_confirmation_outcome_batch(
            old_specs[index : index + 2]
        )
        old_fingerprints.update(_outcome_row_fingerprints(outcome))
    old_fingerprints.update(
        _branch_row_fingerprints(build_confirmation_factual_population())
    )

    assert new_fingerprints
    assert old_fingerprints
    assert new_fingerprints.isdisjoint(old_fingerprints)


def test_top_level_fingerprint_is_exact_and_reproducible() -> None:
    first_specs = build_ccfr_holdout_pair_specs()
    second_specs = build_ccfr_holdout_pair_specs()
    assert first_specs == second_specs
    assert catalog_fingerprint(first_specs) == catalog_fingerprint(second_specs)
    assert holdout_manifest() == holdout_manifest()
    assert holdout_fingerprint() == EXPECTED_HOLDOUT_FINGERPRINT
    assert holdout_fingerprint() == holdout_fingerprint()
