from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.config import DecoderConfig, LossConfig
from cure_lite.experiment.paired_outcome_bounded import (
    COMPUTATIONAL_THRESHOLDS,
    OUTCOME_BOUNDED_ANCHOR_POPULATION_SCHEMA,
    OUTCOME_FACTUAL_ANCHOR_SCHEDULE_SCHEMA,
    PAIRED_OUTCOME_BOUNDED_SCHEMA,
    OutcomeBoundedAnchorPopulation,
    build_outcome_factual_anchor_schedule,
    execute_paired_outcome_bounded,
)
from cure_lite.experiment.paired_outcome_inputs import (
    PairedOutcomeInputMaterializer,
)
from cure_lite.experiment.paired_outcome_schedule import (
    build_outcome_pair_schedule,
)
from cure_lite.paired_types import PairCatalog
from tests.test_paired_transition_bounded import (
    _pair,
    _validated_population,
)


def _full_catalog() -> PairCatalog:
    clean = tuple(
        sorted(
            (
                _pair(index=index, kind="clean_positive")
                for index in range(206)
            ),
            key=lambda pair: (
                pair.sample_id,
                int(pair.evaluation_gt_id),
                int(pair.pred_id),
                pair.pair_id,
            ),
        )
    )
    component = tuple(
        sorted(
            (
                _pair(index=index, kind="component_null")
                for index in range(16)
            ),
            key=lambda pair: (
                pair.sample_id,
                -1,
                int(pair.pred_id),
                pair.pair_id,
            ),
        )
    )
    unsealed = PairCatalog(
        dataset="oc-apto-bounded-toy",
        split="D_R",
        paired_protocol_fingerprint="1" * 64,
        geometry_catalog_fingerprint="2" * 64,
        source_catalog_fingerprint="3" * 64,
        manifest_fingerprint="4" * 64,
        clean_positive=clean,
        component_null=component,
        identity_null=(),
        exclusions=(),
        catalog_fingerprint="",
    )
    return replace(
        unsealed,
        catalog_fingerprint=stable_fingerprint(
            unsealed.canonical_payload()
        ),
    )


def _population(catalog: PairCatalog) -> OutcomeBoundedAnchorPopulation:
    source = _validated_population()
    population = object.__new__(OutcomeBoundedAnchorPopulation)
    for name in (
        "prepared_catalog_fingerprint",
        "factual_miss",
        "factual_no_miss",
        "identity_null",
    ):
        object.__setattr__(population, name, getattr(source, name))
    for branch in ("factual_miss", "factual_no_miss"):
        object.__setattr__(
            population,
            f"{branch}_ids",
            tuple(
                stable_fingerprint(
                    {
                        "schema_version": (
                            "cure-lite-bounded-anchor-id-v1"
                        ),
                        "branch": branch,
                        "sample_id": example.sample_id,
                        "positive_gt_ids": list(
                            example.supervision.positive_gt_ids
                        ),
                    }
                )
                for example in getattr(source, branch)
            ),
        )
    object.__setattr__(population, "seed", 42)
    object.__setattr__(
        population,
        "pair_catalog_fingerprint",
        catalog.catalog_fingerprint,
    )
    object.__setattr__(
        population,
        "population_fingerprint",
        stable_fingerprint(population.canonical_payload()),
    )
    population.__post_init__()
    return population


def _materializer(
    catalog: PairCatalog,
) -> PairedOutcomeInputMaterializer:
    pairs = (*catalog.clean_positive, *catalog.component_null)
    pair_by_id = {pair.pair_id: pair for pair in pairs}
    source_ids = tuple(sorted(pair.sample_id for pair in pairs))
    gt_union_by_sample: dict[str, torch.Tensor] = {}
    for pair in pairs:
        if pair.pair_kind == "clean_positive":
            gt_union = pair.completion_minus.detach().clone()
        else:
            gt_union = torch.zeros_like(pair.image_valid_mask)
            gt_union[0, 0, 0] = True
        gt_union_by_sample[pair.sample_id] = gt_union

    materializer = object.__new__(PairedOutcomeInputMaterializer)
    for name, value in {
        "dataset": catalog.dataset,
        "pair_catalog_fingerprint": catalog.catalog_fingerprint,
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


def _budget() -> dict[str, object]:
    return {
        "seed": 42,
        "optimizer_updates": 400,
        "steps_per_epoch": 40,
        "factual_miss_states_per_update": 4,
        "factual_no_miss_states_per_update": 4,
        "outcome_pairs_per_update": 2,
        "learning_rate": 2.0e-3,
        "weight_decay": 0.0,
    }


def _inputs() -> tuple[
    OutcomeBoundedAnchorPopulation,
    object,
    object,
    PairedOutcomeInputMaterializer,
]:
    catalog = _full_catalog()
    population = _population(catalog)
    factual_schedule = build_outcome_factual_anchor_schedule(
        population,
        optimizer_updates=400,
        steps_per_epoch=40,
    )
    outcome_schedule = build_outcome_pair_schedule(
        catalog,
        seed=42,
        optimizer_updates=400,
        steps_per_epoch=40,
    )
    return (
        population,
        factual_schedule,
        outcome_schedule,
        _materializer(catalog),
    )


def test_bounded_oc_apto_executes_full_population_and_exact_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()
    verify_calls = 0
    original_verify = PairedOutcomeInputMaterializer.verify_unchanged

    def counted_verify(self: PairedOutcomeInputMaterializer) -> None:
        nonlocal verify_calls
        verify_calls += 1
        original_verify(self)

    monkeypatch.setattr(
        PairedOutcomeInputMaterializer,
        "verify_unchanged",
        counted_verify,
    )
    result = execute_paired_outcome_bounded(
        population,
        factual_schedule,
        schedule,
        materializer,
        DecoderConfig(feature_channels=3),
        LossConfig(),
        _budget(),
        device="cpu",
        evaluation_chunk_size=64,
    )

    assert verify_calls == 2
    assert result["schema_version"] == PAIRED_OUTCOME_BOUNDED_SCHEMA
    assert result["execution_status"] == "completed"
    assert result["optimizer_updates_completed"] == 400
    assert result["structural_execution_pass"] is True
    assert all(result["structural_checks"].values())
    assert result["execution_ledger"] == {
        "backward_calls": 400,
        "optimizer_steps": 400,
        "expected_backward_calls": 400,
        "expected_optimizer_steps": 400,
    }
    assert result["forward_budget"]["initial_evaluation"] == {
        "calls": 7,
        "state_evaluations": 508,
    }
    assert result["forward_budget"]["training"] == {
        "calls": 1200,
        "state_evaluations": 4800,
    }
    assert result["forward_budget"]["total"] == {
        "calls": 1214,
        "state_evaluations": 5816,
    }

    for snapshot_name in ("initial", "final"):
        snapshot = result[snapshot_name]
        outcome = snapshot["outcome_population"]
        assert outcome["pair_count"] == 222
        assert outcome["clean_positive_count"] == 206
        assert outcome["component_null_count"] == 16
        assert len(outcome["per_pair"]) == 222
        assert outcome["strata_pixel_counts"]["clean_positive"]["D"] > 0
        assert outcome["strata_pixel_counts"]["component_null"]["D"] == 0
        assert outcome["strata_pixel_counts"]["component_null"]["H"] > 0
        assert outcome["strata_pixel_counts"]["component_null"]["G"] > 0
        assert (
            snapshot["factual_anchors"]["factual_miss"]["state_count"]
            == 16
        )
        assert (
            snapshot["factual_anchors"]["factual_no_miss"]["state_count"]
            == 16
        )
        assert snapshot["identity_null"]["pair_count"] == 16
        assert snapshot["identity_null"]["autograd_enabled"] is False

    assert {
        row["count"] for row in result["exposure"]["outcome_pairs"]
    } == {3, 4}
    assert len(result["exposure"]["outcome_pairs"]) == 222
    assert result["exposure"]["identity_null_optimizer_exposure"] == 0
    assert set(result["computational_gates"]["thresholds"]) == set(
        COMPUTATIONAL_THRESHOLDS
    )
    assert len(result["computational_gates"]["checks"]) == 11

    final_pairs = result["final"]["outcome_population"]["per_pair"]
    component_h_maxima = [
        row["H_max_abs_delta"]
        for row in final_pairs
        if row["pair_kind"] == "component_null"
    ]
    assert (
        result["final"]["outcome_population"]["component_null"][
            "footprint_global_max_abs_delta"
        ]
        == max(component_h_maxima)
    )

    interpretation = result["interpretation"]
    assert interpretation["not_detection_performance_evidence"] is True
    assert interpretation["D_V_accessed"] is False
    assert interpretation["D_T_accessed"] is False
    assert interpretation["base_or_backbone_updated"] is False

    payload = dict(result)
    fingerprint = payload.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(payload)
    assert result["parameters"]["initial_decoder_fingerprint"] != (
        result["parameters"]["final_decoder_fingerprint"]
    )
    assert result["gradients"]["nonfinite_updates"] == 0
    assert result["gradients"]["zero_norm_updates"] == 0

    population_receipt = population.canonical_receipt()
    assert population_receipt["schema_version"] == (
        OUTCOME_BOUNDED_ANCHOR_POPULATION_SCHEMA
    )
    assert "clean_pairs" not in population_receipt
    assert "component_null" not in population_receipt
    factual_receipt = factual_schedule.canonical_receipt()
    assert factual_receipt["schema_version"] == (
        OUTCOME_FACTUAL_ANCHOR_SCHEDULE_SCHEMA
    )
    assert "pair_indices" not in factual_receipt
    assert set(factual_receipt) >= {
        "factual_miss_indices",
        "factual_no_miss_indices",
        "factual_miss_counts",
        "factual_no_miss_counts",
    }


def test_bounded_oc_apto_rejects_budget_drift_before_verification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    population, factual_schedule, schedule, materializer = _inputs()

    def fail_verify(self: PairedOutcomeInputMaterializer) -> None:
        del self
        raise AssertionError("verification must not run after budget rejection")

    monkeypatch.setattr(
        PairedOutcomeInputMaterializer,
        "verify_unchanged",
        fail_verify,
    )
    changed = _budget()
    changed["optimizer_updates"] = 399
    with pytest.raises(ValueError, match="fixes 400 optimizer updates"):
        execute_paired_outcome_bounded(
            population,
            factual_schedule,
            schedule,
            materializer,
            DecoderConfig(feature_channels=3),
            LossConfig(),
            changed,
            device="cpu",
        )
