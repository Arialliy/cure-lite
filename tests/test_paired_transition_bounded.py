from __future__ import annotations

import hashlib

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.config import DecoderConfig, LossConfig
from cure_lite.experiment.paired_bounded_learnability import (
    BoundedMicroPopulation,
    build_bounded_micro_schedule,
)
from cure_lite.experiment.paired_transition_bounded import (
    PAIRED_TRANSITION_BOUNDED_SCHEMA,
    execute_paired_transition_bounded,
)
from cure_lite.experiment.paired_transition_inputs import (
    PairedTransitionInputMaterializer,
)
from cure_lite.paired_types import PairExample, tensor_content_fingerprint
from cure_lite.train.pools import StateExample
from cure_lite.types import BranchSupervision


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _pair(
    *,
    index: int,
    kind: str,
) -> PairExample:
    height = width = 4
    feature = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
    d_locations = ((1, 1), (2, 2))
    anchor_locations = ((2, 0), (1, 3))
    d_row, d_column = d_locations[index % 2]
    anchor_row, anchor_column = anchor_locations[index % 2]
    feature[0, 0, d_row // 2, d_column // 2] = 4.0
    feature[0, 1, anchor_row // 2, anchor_column // 2] = 4.0
    feature[0, 2] = 0.05 * (index + 1)

    plus = torch.zeros((1, height, width), dtype=torch.bool)
    minus = torch.zeros_like(plus)
    removed = torch.zeros_like(plus)
    completion_plus = torch.zeros_like(plus)
    completion_minus = torch.zeros_like(plus)
    increment = torch.zeros((1, height, width), dtype=torch.float32)
    clean_increment = torch.zeros_like(plus)
    visible = kind != "identity_null"
    if visible:
        plus[0, d_row, d_column] = True
        removed.copy_(plus)
    if kind == "clean_positive":
        completion_plus[0, anchor_row, anchor_column] = True
        completion_minus.copy_(completion_plus)
        completion_minus[0, d_row, d_column] = True
        increment[0, d_row, d_column] = 1.0
        clean_increment[0, d_row, d_column] = True

    projected_plus = tensor_content_fingerprint(plus)
    projected_minus = (
        projected_plus
        if kind == "identity_null"
        else tensor_content_fingerprint(minus)
    )
    return PairExample(
        pair_id=_sha(f"{kind}-pair-{index}"),
        pair_kind=kind,
        sample_id=f"{kind}-source-{index:02d}",
        group_id=f"{kind}-group-{index:02d}",
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
        removed_component=removed,
        image_valid_mask=torch.ones_like(plus),
        completion_plus=completion_plus,
        completion_minus=completion_minus,
        label_increment=increment,
        clean_increment=clean_increment,
        evaluation_gt_id=index + 1 if kind == "clean_positive" else None,
        native_gt_id=index + 1 if kind == "clean_positive" else None,
        pred_id=index + 1 if kind != "identity_null" else None,
        feature_fingerprint=tensor_content_fingerprint(feature),
        before_match_fingerprint=_sha(f"{kind}-before-{index}"),
        after_match_fingerprint=_sha(f"{kind}-after-{index}"),
        projected_occupancy_plus_fingerprint=projected_plus,
        projected_occupancy_minus_fingerprint=projected_minus,
        projection_visible=visible,
        geometry_safe_bijective_lineage=(
            True if kind == "clean_positive" else None
        ),
        selected_gt_is_only_new_unmatched=(
            True if kind == "clean_positive" else None
        ),
        other_match_identities_unchanged=(
            True if kind == "clean_positive" else None
        ),
        preexisting_unmatched_gt_noninterference=(
            True if kind == "clean_positive" else None
        ),
    )


def _state(
    *,
    index: int,
    branch: str,
    clean_pair: PairExample,
) -> StateExample:
    occupancy = torch.zeros((1, 4, 4), dtype=torch.bool)
    target = torch.zeros((1, 4, 4), dtype=torch.float32)
    positive_ids: tuple[int, ...] = ()
    reachable_ids: tuple[int, ...] = ()
    if branch == "factual_miss":
        target.copy_(
            (
                clean_pair.completion_plus
                | clean_pair.label_increment.to(dtype=torch.bool)
            ).to(dtype=torch.float32)
        )
        positive_ids = (index + 1,)
        reachable_ids = positive_ids
        feature = clean_pair.feature.detach().clone()
    else:
        feature = torch.zeros_like(clean_pair.feature)
    return StateExample(
        sample_id=f"{branch}-source-{index:02d}",
        feature=feature,
        supervision=BranchSupervision(
            occupancy=occupancy,
            target=target,
            valid_mask=torch.ones_like(occupancy),
            branch=branch,
            positive_gt_ids=positive_ids,
            reachable_gt_ids=reachable_ids,
        ),
    )


def _validated_population() -> BoundedMicroPopulation:
    clean = tuple(_pair(index=index, kind="clean_positive") for index in range(16))
    component = tuple(
        _pair(index=index, kind="component_null") for index in range(16)
    )
    identity = tuple(
        _pair(index=index, kind="identity_null") for index in range(16)
    )
    factual_miss = tuple(
        _state(
            index=index,
            branch="factual_miss",
            clean_pair=clean[index],
        )
        for index in range(16)
    )
    factual_no_miss = tuple(
        _state(
            index=index,
            branch="factual_no_miss",
            clean_pair=clean[index],
        )
        for index in range(16)
    )
    miss_ids = tuple(_sha(f"factual-miss-anchor-{index}") for index in range(16))
    no_miss_ids = tuple(
        _sha(f"factual-no-miss-anchor-{index}") for index in range(16)
    )

    population = object.__new__(BoundedMicroPopulation)
    for name, value in {
        "seed": 1701,
        "pair_catalog_fingerprint": "a" * 64,
        "prepared_catalog_fingerprint": "b" * 64,
        "clean_pairs": clean,
        "factual_miss": factual_miss,
        "factual_no_miss": factual_no_miss,
        "component_null": component,
        "identity_null": identity,
        "factual_miss_ids": miss_ids,
        "factual_no_miss_ids": no_miss_ids,
    }.items():
        object.__setattr__(population, name, value)
    object.__setattr__(
        population,
        "population_fingerprint",
        stable_fingerprint(population.canonical_payload()),
    )
    population.__post_init__()
    return population


def _validated_materializer(
    population: BoundedMicroPopulation,
) -> PairedTransitionInputMaterializer:
    pair_by_id = {pair.pair_id: pair for pair in population.clean_pairs}
    source_ids = tuple(sorted(pair.sample_id for pair in population.clean_pairs))
    gt_union_by_sample = {
        pair.sample_id: (
            pair.completion_plus
            | pair.label_increment.to(dtype=torch.bool)
        )
        for pair in population.clean_pairs
    }
    materializer = object.__new__(PairedTransitionInputMaterializer)
    for name, value in {
        "dataset": "apto-bounded-toy",
        "pair_catalog_fingerprint": population.pair_catalog_fingerprint,
        # The reused v1 micro-population and the APTO materializer seal the
        # prepared catalog under different versioned schemas.
        "prepared_catalog_fingerprint": "c" * 64,
        "prepared_source_ids": source_ids,
        "pair_by_id": pair_by_id,
        "gt_union_by_sample": gt_union_by_sample,
        "feature_shape": (1, 3, 2, 2),
        "evaluation_shape": (1, 4, 4),
    }.items():
        object.__setattr__(materializer, name, value)
    object.__setattr__(
        materializer,
        "materializer_fingerprint",
        stable_fingerprint(materializer._canonical_payload()),
    )
    materializer.__post_init__()
    return materializer


def _budget(updates: int = 8) -> dict[str, object]:
    return {
        "seed": 1701,
        "optimizer_updates": updates,
        "steps_per_epoch": 8,
        "factual_miss_states_per_update": 4,
        "factual_no_miss_states_per_update": 4,
        "clean_pairs_per_update": 2,
        "learning_rate": 2.0e-3,
        "weight_decay": 0.0,
    }


def _execute() -> dict[str, object]:
    population = _validated_population()
    budget = _budget()
    schedule = build_bounded_micro_schedule(population, budget)
    materializer = _validated_materializer(population)
    return execute_paired_transition_bounded(
        population,
        schedule,
        materializer,
        DecoderConfig(feature_channels=3),
        LossConfig(),
        budget,
        device="cpu",
    )


def test_bounded_apto_executes_exact_real_type_budget_and_null_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verify_calls = 0
    original_verify = PairedTransitionInputMaterializer.verify_unchanged

    def counted_verify(self: PairedTransitionInputMaterializer) -> None:
        nonlocal verify_calls
        verify_calls += 1
        original_verify(self)

    monkeypatch.setattr(
        PairedTransitionInputMaterializer,
        "verify_unchanged",
        counted_verify,
    )
    result = _execute()

    assert verify_calls == 2
    assert result["schema_version"] == PAIRED_TRANSITION_BOUNDED_SCHEMA
    assert result["execution_status"] == "completed"
    assert result["structural_execution_pass"] is True
    assert all(result["structural_checks"].values())
    assert result["optimizer_updates_completed"] == 8
    assert result["execution_ledger"] == {
        "backward_calls": 8,
        "optimizer_steps": 8,
        "expected_backward_calls": 8,
        "expected_optimizer_steps": 8,
    }
    assert result["forward_budget"]["initial_evaluation"] == {
        "calls": 3,
        "state_evaluations": 96,
    }
    assert result["forward_budget"]["training"] == {
        "calls": 24,
        "state_evaluations": 96,
    }
    assert result["forward_budget"]["total"] == {
        "calls": 30,
        "state_evaluations": 288,
    }
    assert result["initial"]["clean"]["clean_pair_count"] == 16
    assert result["final"]["clean"]["clean_pair_count"] == 16
    assert len(result["initial"]["clean"]["per_pair"]) == 16
    assert len(result["final"]["clean"]["per_pair"]) == 16
    assert set(result["final"]["clean"]["apto"]) == {
        "total_loss",
        "plus_anchor_loss",
        "transition_loss",
    }

    assert {
        row["count"] for row in result["exposure"]["clean_pairs"]
    } == {1}
    assert result["exposure"]["component_null_optimizer_exposure"] == 0
    assert result["exposure"]["identity_null_optimizer_exposure"] == 0
    for snapshot in ("initial", "final"):
        for name in ("component_null", "identity_null"):
            nulls = result[snapshot]["nulls"][name]
            assert nulls["pair_count"] == 16
            assert nulls["diagnostic_only"] is True
            assert nulls["autograd_enabled"] is False
            assert nulls["optimizer_exposure_count"] == 0

    interpretation = result["interpretation"]
    assert interpretation["not_detection_performance_evidence"] is True
    assert interpretation["does_not_establish_Pd_or_FA"] is True
    assert interpretation["D_V_accessed"] is False
    assert interpretation["D_T_accessed"] is False
    assert interpretation["base_or_backbone_updated"] is False
    assert result["computational_gates"]["not_detection_performance"] is True


def test_bounded_apto_exact_replay_and_result_fingerprint() -> None:
    first = _execute()
    second = _execute()

    assert first == second
    fingerprint = first.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(first)
    assert fingerprint == second["result_fingerprint"]
    assert first["parameters"]["initial_decoder_fingerprint"] != (
        first["parameters"]["final_decoder_fingerprint"]
    )
    assert first["gradients"]["nonfinite_updates"] == 0
    assert first["gradients"]["zero_norm_updates"] == 0


def test_bounded_apto_rejects_schedule_budget_drift_before_training() -> None:
    population = _validated_population()
    materializer = _validated_materializer(population)
    schedule = build_bounded_micro_schedule(population, _budget())
    changed = _budget()
    changed["optimizer_updates"] = 16

    with pytest.raises(ValueError, match="budget and bounded schedule disagree"):
        execute_paired_transition_bounded(
            population,
            schedule,
            materializer,
            DecoderConfig(feature_channels=3),
            LossConfig(),
            changed,
            device="cpu",
        )
