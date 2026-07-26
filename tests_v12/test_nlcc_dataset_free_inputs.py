from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from cure_lite.nlcc_dataset_free_inputs import (
    ANCHOR_NULL,
    ANCHOR_POSITIVE,
    COMPLETION_WITNESS_CHANNELS,
    FACTUAL_BATCH_SIZE,
    FACTUAL_POPULATION_SIZE,
    FEATURE_CHANNELS,
    FEATURE_HEIGHT,
    FEATURE_STRIDE,
    FEATURE_WIDTH,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    build_strata,
    catalog_manifest,
    factual_indices_for_update,
    factual_population_manifest,
    reachability_audit,
)
from cure_lite.nlcc_development_inputs import (
    DEVELOPMENT_PROFILE,
    build_nlcc_development_factual_batches,
    build_nlcc_development_factual_population,
    build_nlcc_development_outcome_batch,
    build_nlcc_development_pair_specs,
    build_nlcc_development_schedule,
    nlcc_development_fingerprint,
    nlcc_development_manifest,
    nlcc_development_reachability_audit,
)
from cure_lite.nlcc_holdout_inputs import (
    HOLDOUT_PROFILE,
    build_nlcc_holdout_factual_population,
    build_nlcc_holdout_outcome_batch,
    build_nlcc_holdout_pair_specs,
    build_nlcc_holdout_schedule,
    nlcc_holdout_fingerprint,
    nlcc_holdout_manifest,
    nlcc_holdout_reachability_audit,
)


def _by_match(specs):
    values = {}
    for spec in specs:
        values.setdefault(spec.match_id, {})[spec.anchor_role] = spec
    return values


def _frozen_lattice_value(
    seed,
    stage,
    group,
    within_group_index,
    match_id,
    signal_role,
    channel,
    row,
    column,
):
    digest = hashlib.sha256(
        (
            f"{seed}|{stage}|{group}|{within_group_index}|{match_id}|"
            f"{signal_role}|{channel}|{row}|{column}"
        ).encode("utf-8")
    ).digest()
    unsigned = int.from_bytes(digest[:2], byteorder="big", signed=False)
    magnitude = float(32 + unsigned % 97) / 64.0
    return magnitude if digest[2] & 1 else -magnitude


def test_full_populations_use_exact_nine_field_lattice_and_zero_elsewhere() -> None:
    for profile, specs in (
        (
            DEVELOPMENT_PROFILE,
            build_nlcc_development_pair_specs(),
        ),
        (
            HOLDOUT_PROFILE,
            build_nlcc_holdout_pair_specs(),
        ),
    ):
        materialized = (
            build_nlcc_development_outcome_batch(specs)
            if profile is DEVELOPMENT_PROFILE
            else build_nlcc_holdout_outcome_batch(specs)
        )
        for index, spec in enumerate(specs):
            expected_feature = torch.zeros(
                FEATURE_CHANNELS,
                FEATURE_HEIGHT,
                FEATURE_WIDTH,
                dtype=torch.float32,
            )
            primary_channels = (
                (0, 1) if spec.pair_kind == "clean_positive" else (2, 3)
            )
            primary_cell = (
                spec.response_cell
                if spec.pair_kind == "clean_positive"
                else spec.component_cell
            )
            for channel in primary_channels:
                expected_feature[channel, primary_cell[0], primary_cell[1]] = (
                    _frozen_lattice_value(
                        profile.design_seed,
                        "pair",
                        spec.group_id,
                        spec.within_group_dyad_index,
                        spec.match_id,
                        "primary",
                        channel,
                        primary_cell[0],
                        primary_cell[1],
                    )
                )
            if spec.anchor_role == ANCHOR_POSITIVE:
                for channel in COMPLETION_WITNESS_CHANNELS:
                    expected_feature[
                        channel,
                        spec.anchor_cell[0],
                        spec.anchor_cell[1],
                    ] = _frozen_lattice_value(
                        profile.design_seed,
                        "pair",
                        spec.group_id,
                        spec.within_group_dyad_index,
                        spec.match_id,
                        "anchor_witness",
                        channel,
                        spec.anchor_cell[0],
                        spec.anchor_cell[1],
                    )
            assert torch.equal(
                materialized.pair_batch.feature[index],
                expected_feature,
            )
            nonzero = expected_feature[expected_feature != 0.0]
            assert torch.all(nonzero.abs() >= 0.5)
            assert torch.all(nonzero.abs() <= 2.0)
            assert torch.all(nonzero.mul(64.0) == nonzero.mul(64.0).round())

        factual = (
            build_nlcc_development_factual_population()
            if profile is DEVELOPMENT_PROFILE
            else build_nlcc_holdout_factual_population()
        )
        for branch, batch in factual.items():
            channels = (0, 1) if branch == "factual_miss" else (4, 5)
            for state_index in range(FACTUAL_POPULATION_SIZE):
                cell = (
                    2 + state_index % 3,
                    2 + ((state_index // 3 + 1) % 3),
                )
                match_id = hashlib.sha256(
                    (
                        f"nlcc-v12|{profile.profile_id}|{profile.design_seed}|"
                        f"factual|{branch}|{state_index:03d}"
                    ).encode("utf-8")
                ).hexdigest()
                expected_feature = torch.zeros(
                    FEATURE_CHANNELS,
                    FEATURE_HEIGHT,
                    FEATURE_WIDTH,
                    dtype=torch.float32,
                )
                for channel in channels:
                    expected_feature[channel, cell[0], cell[1]] = (
                        _frozen_lattice_value(
                            profile.design_seed,
                            "factual",
                            branch,
                            state_index,
                            match_id,
                            branch,
                            channel,
                            cell[0],
                            cell[1],
                        )
                    )
                assert torch.equal(batch.feature[state_index], expected_feature)


def test_profiles_freeze_exact_population_and_exposure_contracts() -> None:
    assert DEVELOPMENT_PROFILE.design_seed == 2550254881
    assert DEVELOPMENT_PROFILE.update_count == 320
    assert DEVELOPMENT_PROFILE.group_dyad_counts == (2,) * 8
    assert DEVELOPMENT_PROFILE.dyad_count == 16
    assert DEVELOPMENT_PROFILE.row_count == 32
    assert DEVELOPMENT_PROFILE.factual_exposures_per_state == 80

    development = build_nlcc_development_pair_specs()
    assert Counter(spec.anchor_role for spec in development) == Counter(
        {ANCHOR_POSITIVE: 16, ANCHOR_NULL: 16}
    )
    clean_dyad_exposures = [
        values[ANCHOR_POSITIVE].exposure_count
        for values in _by_match(development).values()
        if values[ANCHOR_POSITIVE].pair_kind == "clean_positive"
    ]
    component_dyad_exposures = [
        values[ANCHOR_POSITIVE].exposure_count
        for values in _by_match(development).values()
        if values[ANCHOR_POSITIVE].pair_kind == "component_null"
    ]
    assert Counter(clean_dyad_exposures) == Counter({25: 8, 24: 4})
    assert Counter(component_dyad_exposures) == Counter({6: 4})

    assert HOLDOUT_PROFILE.design_seed == 1788878112
    assert HOLDOUT_PROFILE.update_count == 400
    assert HOLDOUT_PROFILE.group_dyad_counts == (
        18,
        17,
        17,
        17,
        17,
        17,
        4,
        4,
    )
    assert HOLDOUT_PROFILE.dyad_count == 111
    assert HOLDOUT_PROFILE.row_count == 222
    assert HOLDOUT_PROFILE.factual_exposures_per_state == 100
    holdout = build_nlcc_holdout_pair_specs()
    assert Counter(spec.group_id for spec in holdout) == Counter(
        {
            "clean_same_cell_1px": 36,
            "clean_same_cell_3px": 34,
            "clean_adjacent_cell_1px": 34,
            "clean_adjacent_cell_3px": 34,
            "clean_multicount_2to1": 34,
            "clean_multicount_3to2": 34,
            "component_null_block": 8,
            "component_null_sparse": 8,
        }
    )
    clean = [spec for spec in holdout if spec.pair_kind == "clean_positive"]
    component = [spec for spec in holdout if spec.pair_kind == "component_null"]
    assert Counter(spec.exposure_count for spec in clean) == Counter(
        {3: 84, 4: 122}
    )
    assert Counter(spec.exposure_count for spec in component) == Counter(
        {3: 4, 4: 12}
    )
    assert sum(spec.exposure_count for spec in clean) == 740
    assert sum(spec.exposure_count for spec in component) == 60


def test_matched_twins_have_exactly_the_frozen_input_and_target_difference() -> None:
    specs = build_nlcc_holdout_pair_specs()
    for roles in _by_match(specs).values():
        assert set(roles) == {ANCHOR_POSITIVE, ANCHOR_NULL}
        positive = roles[ANCHOR_POSITIVE]
        null = roles[ANCHOR_NULL]
        assert positive.sample_id == null.sample_id
        assert positive.exposure_count == null.exposure_count
        positive_outcome = build_nlcc_holdout_outcome_batch((positive,))
        null_outcome = build_nlcc_holdout_outcome_batch((null,))

        feature_diff = (
            positive_outcome.pair_batch.feature
            != null_outcome.pair_batch.feature
        )
        expected_feature_diff = torch.zeros_like(feature_diff)
        for channel in COMPLETION_WITNESS_CHANNELS:
            expected_feature_diff[
                0,
                channel,
                positive.anchor_cell[0],
                positive.anchor_cell[1],
            ] = True
        assert torch.equal(feature_diff, expected_feature_diff)
        witness = positive_outcome.pair_batch.feature[
            0,
            list(COMPLETION_WITNESS_CHANNELS),
            positive.anchor_cell[0],
            positive.anchor_cell[1],
        ]
        assert torch.all(witness != 0.0)

        assert torch.equal(
            positive_outcome.pair_batch.occupancy_plus,
            null_outcome.pair_batch.occupancy_plus,
        )
        assert torch.equal(
            positive_outcome.pair_batch.occupancy_minus,
            null_outcome.pair_batch.occupancy_minus,
        )
        assert torch.equal(
            positive_outcome.pair_batch.label_increment,
            null_outcome.pair_batch.label_increment,
        )
        assert torch.equal(
            positive_outcome.intervention_footprint,
            null_outcome.intervention_footprint,
        )
        anchor_row = FEATURE_STRIDE * positive.anchor_cell[0] + 1
        anchor_column = FEATURE_STRIDE * positive.anchor_cell[1] + 1
        expected_anchor = torch.zeros_like(positive_outcome.completion_plus)
        expected_anchor[0, 0, anchor_row, anchor_column] = True
        for field in ("completion_plus", "completion_minus", "gt_union"):
            assert torch.equal(
                getattr(positive_outcome, field)
                ^ getattr(null_outcome, field),
                expected_anchor,
            )


def test_all_geometries_have_exact_count_transitions_and_D_H_G_partition() -> None:
    specs = build_nlcc_holdout_pair_specs()
    outcome = build_nlcc_holdout_outcome_batch(specs)
    batch = outcome.pair_batch
    strata = build_strata(outcome)
    assert batch.feature.shape == (
        222,
        FEATURE_CHANNELS,
        FEATURE_HEIGHT,
        FEATURE_WIDTH,
    )
    assert batch.occupancy_plus.shape == (
        222,
        1,
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH,
    )
    assert torch.equal(
        strata.D | strata.H | strata.G_near | strata.G_norm_tail,
        batch.image_valid_mask,
    )
    assert torch.all(strata.H.flatten(1).any(dim=1))
    assert torch.all(strata.G_near.flatten(1).any(dim=1))
    assert torch.all(strata.G_norm_tail.flatten(1).any(dim=1))

    projected_plus = F.max_pool2d(
        batch.occupancy_plus.float(),
        kernel_size=FEATURE_STRIDE,
        stride=FEATURE_STRIDE,
    )
    projected_minus = F.max_pool2d(
        batch.occupancy_minus.float(),
        kernel_size=FEATURE_STRIDE,
        stride=FEATURE_STRIDE,
    )
    kernel = torch.ones(1, 1, 3, 3)
    count_plus = F.conv2d(projected_plus, kernel, padding=1)
    count_minus = F.conv2d(projected_minus, kernel, padding=1)
    for index, spec in enumerate(specs):
        if spec.pair_kind == "component_null":
            assert not strata.D[index].any()
            continue
        row, column = spec.response_cell
        observed = (
            int(count_plus[index, 0, row, column]),
            int(count_minus[index, 0, row, column]),
        )
        if spec.geometry_family in {"same_cell", "adjacent_cell"}:
            assert observed == (1, 0)
        elif spec.geometry_family == "multicount_2to1":
            assert observed == (2, 1)
        else:
            assert observed == (3, 2)


def test_schedules_are_role_balanced_twin_equal_and_never_pair_one_match() -> None:
    for profile, specs, updates in (
        (
            DEVELOPMENT_PROFILE,
            build_nlcc_development_pair_specs(),
            build_nlcc_development_schedule(),
        ),
        (
            HOLDOUT_PROFILE,
            build_nlcc_holdout_pair_specs(),
            build_nlcc_holdout_schedule(),
        ),
    ):
        assert len(updates) == profile.update_count
        assert all(set(update.anchor_roles) == set((ANCHOR_POSITIVE, ANCHOR_NULL))
                   for update in updates)
        assert all(len(set(update.match_ids)) == 2 for update in updates)
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


def test_factual_populations_keep_sixteen_state_rotation_and_4_4_batches() -> None:
    for profile, population in (
        (
            DEVELOPMENT_PROFILE,
            build_nlcc_development_factual_population(),
        ),
        (HOLDOUT_PROFILE, build_nlcc_holdout_factual_population()),
    ):
        assert set(population) == {"factual_miss", "factual_no_miss"}
        for branch, batch in population.items():
            batch.validate(expected_branch=branch)
            assert batch.feature.shape == (
                FACTUAL_POPULATION_SIZE,
                FEATURE_CHANNELS,
                FEATURE_HEIGHT,
                FEATURE_WIDTH,
            )
        exposures = Counter(
            index
            for update_index in range(profile.update_count)
            for index in factual_indices_for_update(profile, update_index)
        )
        assert exposures == Counter(
            {
                index: profile.factual_exposures_per_state
                for index in range(FACTUAL_POPULATION_SIZE)
            }
        )
    batches = build_nlcc_development_factual_batches(update_index=319)
    assert all(
        batch.feature.shape[0] == FACTUAL_BATCH_SIZE
        for batch in batches.values()
    )


def test_all_six_reachability_counts_and_all_twin_counts_are_exact_zero() -> None:
    for audit in (
        nlcc_development_reachability_audit(),
        nlcc_holdout_reachability_audit(),
    ):
        assert audit["all_pass"] is True
        assert set(audit["required_zero_counts"]) == {
            "unwitnessed_completion_count",
            "completion_changed_support_overlap_count",
            "opposite_label_identical_input_count",
            "opposite_label_local_signature_conflict_count",
            "clean_D_without_feature_witness_count",
            "clean_D_without_count_difference_count",
        }
        assert set(audit["required_zero_counts"].values()) == {0}
        assert set(audit["twin_integrity_counts"].values()) == {0}
        assert set(audit["input_integrity_counts"]) == {
            "same_match_within_update_count",
            "same_sample_within_update_count",
            "same_pair_within_update_count",
            "nonfinite_input_count",
            "twin_D_H_G_partition_difference_count",
        }
        assert set(audit["input_integrity_counts"].values()) == {0}


def test_manifests_are_exact_replays_and_profiles_share_no_input_tensors() -> None:
    development_specs = build_nlcc_development_pair_specs()
    holdout_specs = build_nlcc_holdout_pair_specs()
    development_catalog = catalog_manifest(
        DEVELOPMENT_PROFILE,
        development_specs,
    )
    holdout_catalog = catalog_manifest(HOLDOUT_PROFILE, holdout_specs)
    development_factual = factual_population_manifest(DEVELOPMENT_PROFILE)
    holdout_factual = factual_population_manifest(HOLDOUT_PROFILE)

    development_features = {
        row["tensor_fingerprints"]["feature"]
        for row in (*development_catalog, *development_factual)
    }
    holdout_features = {
        row["tensor_fingerprints"]["feature"]
        for row in (*holdout_catalog, *holdout_factual)
    }
    development_rows = {
        tuple(sorted(row["tensor_fingerprints"].items()))
        for row in (*development_catalog, *development_factual)
    }
    holdout_rows = {
        tuple(sorted(row["tensor_fingerprints"].items()))
        for row in (*holdout_catalog, *holdout_factual)
    }
    assert {
        spec.pair_id for spec in development_specs
    }.isdisjoint({spec.pair_id for spec in holdout_specs})
    assert {
        spec.match_id for spec in development_specs
    }.isdisjoint({spec.match_id for spec in holdout_specs})
    assert {
        spec.sample_id for spec in development_specs
    }.isdisjoint({spec.sample_id for spec in holdout_specs})
    assert development_features.isdisjoint(holdout_features)
    assert development_rows.isdisjoint(holdout_rows)
    development_manifest = nlcc_development_manifest()
    holdout_manifest = nlcc_holdout_manifest()
    assert development_manifest == nlcc_development_manifest()
    assert holdout_manifest == nlcc_holdout_manifest()
    assert json.loads(
        json.dumps(
            development_manifest,
            sort_keys=True,
            separators=(",", ":"),
        )
    ) == development_manifest
    assert json.loads(
        json.dumps(
            holdout_manifest,
            sort_keys=True,
            separators=(",", ":"),
        )
    ) == holdout_manifest
    assert nlcc_development_fingerprint() == nlcc_development_fingerprint()
    assert nlcc_holdout_fingerprint() == nlcc_holdout_fingerprint()
    assert nlcc_development_fingerprint() != nlcc_holdout_fingerprint()


def test_input_modules_keep_the_frozen_static_import_boundary() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    expected_relative_imports = {
        "cure_lite/nlcc_dataset_free_inputs.py": {
            "cache.schema",
            "decoder",
            "paired_outcome_types",
            "paired_types",
            "train.step",
        },
        "cure_lite/nlcc_development_inputs.py": {
            "nlcc_dataset_free_inputs",
        },
        "cure_lite/nlcc_holdout_inputs.py": {
            "nlcc_dataset_free_inputs",
        },
    }
    forbidden_fragments = {
        "datasets",
        "dataloader",
        "detector",
        "optimizer",
        "runner",
        "result",
        "artifact",
        "null_anchored_local_count_crossing_decoder",
    }
    for relative_path, expected in expected_relative_imports.items():
        source = (repo_root / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative_path)
        relative_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 1
        }
        absolute_from_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 0
        }
        absolute_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert relative_modules == expected
        assert absolute_modules | absolute_from_modules <= {
            "__future__",
            "collections",
            "dataclasses",
            "hashlib",
            "typing",
            "torch",
            "torch.nn",
        }
        assert all(
            fragment not in module
            for module in (
                *relative_modules,
                *absolute_modules,
                *absolute_from_modules,
            )
            for fragment in forbidden_fragments
        )
