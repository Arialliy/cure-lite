from __future__ import annotations

from math import isfinite

import pytest
import torch

from cure_lite.experiment.p0_protocol import (
    P0OverlapConfig,
    P0SeparabilityConfig,
)
from cure_lite.experiment.synthetic_state_failure_attribution import (
    COMMON_BLOCKS,
    CommonStateRecord,
    SameSourceExpectation,
    SharedGroupExpectation,
    build_common_state_record,
    build_legal_occupancy_ledger,
    exact_same_source_subset,
    run_block_coverage_mmd,
    run_block_only_group_oof,
    run_composite_group_oof,
    run_exact_same_source_sensitivity,
    shared_manifest_group_subset,
)


def _overlap(*, k: int = 2) -> P0OverlapConfig:
    return P0OverlapConfig(
        factual_population="reachable-factual-misses",
        legal_population="decoder-visible-legal-targets",
        group_key="manifest.group_id",
        exclude_same_group_neighbors=True,
        handcrafted_descriptor_fields=("unused-by-attribution-core",),
        probability_clip=1e-6,
        ring_inner_radius=1,
        ring_outer_radius=3,
        joint_feature_components=1,
        joint_feature_residual=(
            "legal-subspace-reconstruction-l2-per-sqrt-dimension-v1"
        ),
        joint_occupancy_representation=(
            "raw-local-patch-plus-global-fraction-v1"
        ),
        joint_occupancy_patch_radius=2,
        knn_k=k,
        legal_reference_quantile=0.95,
        coverage_minimum=0.9,
        robust_scale_rule="median-mad-maxdev-constant-floor-v1",
        quantile_rule="sorted-higher-v1",
    )


def _separability(*, folds: int = 3) -> P0SeparabilityConfig:
    return P0SeparabilityConfig(
        folds=folds,
        classifier="class-balanced-l2-logistic-irls-v1",
        classifier_l2=1.0,
        classifier_max_iterations=100,
        classifier_tolerance=1e-10,
        auc_maximum=0.7,
        auc_gate_rule="group-balanced-oof-point-estimate-v1",
        bootstrap_replicates=16,
        bootstrap_seed=1729,
        bootstrap_interval=(0.025, 0.975),
        bootstrap_interpretation=(
            "conditional-group-bootstrap-of-fixed-oof-scores-v1"
        ),
        mmd="group-u-multiscale-rbf-matched-legal-null-v1",
        mmd_group_overlap_policy="remove-overlap-from-legal-reference-v1",
        mmd_observed_summary_quantile=0.5,
        mmd_kernel_scales=(0.5, 1.0, 2.0),
        mmd_bandwidth_rule=(
            "legal-exclusive-source-disjoint-positive-distance-median-v1"
        ),
        mmd_reference_replicates=16,
        mmd_reference_seed=2718,
        mmd_reference_quantile=0.95,
        require_mmd_within_legal_reference=True,
    )


def _explicit_record(
    *,
    sample_id: str,
    group_id: str,
    role: str,
    ordinal: int,
    signal: float,
) -> CommonStateRecord:
    pred_id = None if role == "factual" else ordinal + 1
    gt_id = ordinal + 1
    def vector(length: int, offset: float) -> torch.Tensor:
        return torch.tensor(
            tuple(
                signal
                + offset
                + 0.01 * ordinal
                + 0.001 * dimension
                for dimension in range(length)
            ),
            dtype=torch.float64,
        )

    return CommonStateRecord(
        identity=(sample_id, gt_id, pred_id),
        sample_id=sample_id,
        group_id=group_id,
        role=role,
        G_full=vector(6, 0.0),
        W=vector(4, 0.1),
        P=vector(7, 0.2),
        F_local=vector(4, 0.3),
        F_background_global=vector(4, 0.4),
        O=vector(29, 0.5),
    )


def test_common_blocks_use_full_gt_while_w_is_isolated() -> None:
    target = torch.zeros((16, 16), dtype=torch.bool)
    target[7:9, 7:9] = True
    labels = target.to(torch.int64)
    valid = torch.ones_like(target)
    occupancy = torch.zeros_like(target)
    occupancy[2:4, 2:4] = True
    probability = torch.linspace(0.01, 0.99, 16 * 16).reshape(16, 16)
    feature = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)
    supervision_small = torch.zeros_like(target)
    supervision_small[7, 7] = True

    first = build_common_state_record(
        sample_id="sample",
        group_id="group",
        role="factual",
        gt_id=1,
        pred_id=None,
        target_mask=target,
        supervision_mask=supervision_small,
        conditioning_occupancy=occupancy,
        probability=probability,
        feature=feature,
        gt_labels=labels,
        valid_mask=valid,
        overlap=_overlap(),
    )
    second = build_common_state_record(
        sample_id="sample",
        group_id="group",
        role="factual",
        gt_id=2,
        pred_id=None,
        target_mask=target,
        supervision_mask=target,
        conditioning_occupancy=occupancy,
        probability=probability,
        feature=feature,
        gt_labels=labels,
        valid_mask=valid,
        overlap=_overlap(),
    )

    assert COMMON_BLOCKS == (
        "G_full",
        "W",
        "P",
        "F_local",
        "F_background_global",
        "O",
    )
    for block in ("G_full", "P", "F_local", "F_background_global", "O"):
        assert torch.equal(first.block(block), second.block(block))
    assert not torch.equal(first.W, second.W)
    assert first.G_full.shape == (6,)
    assert first.W.shape == (4,)
    assert first.P.shape == (7,)
    assert first.F_local.shape == (10,)
    assert first.F_background_global.shape == (12,)


def test_common_record_enforces_the_frozen_block_dimensions() -> None:
    with pytest.raises(ValueError, match="G_full"):
        CommonStateRecord(
            identity=("sample", 1, None),
            sample_id="sample",
            group_id="group",
            role="factual",
            G_full=torch.zeros(5, dtype=torch.float64),
            W=torch.zeros(4, dtype=torch.float64),
            P=torch.zeros(7, dtype=torch.float64),
            F_local=torch.zeros(4, dtype=torch.float64),
            F_background_global=torch.zeros(4, dtype=torch.float64),
            O=torch.zeros(29, dtype=torch.float64),
        )

    with pytest.raises(ValueError, match="share one channel count"):
        CommonStateRecord(
            identity=("sample", 1, None),
            sample_id="sample",
            group_id="group",
            role="factual",
            G_full=torch.zeros(6, dtype=torch.float64),
            W=torch.zeros(4, dtype=torch.float64),
            P=torch.zeros(7, dtype=torch.float64),
            F_local=torch.zeros(4, dtype=torch.float64),
            F_background_global=torch.zeros(8, dtype=torch.float64),
            O=torch.zeros(29, dtype=torch.float64),
        )


def test_legal_pre_post_ledger_is_not_a_classifier_record() -> None:
    target = torch.zeros((16, 16), dtype=torch.bool)
    target[7:9, 7:9] = True
    ring = torch.zeros_like(target)
    ring[4:12, 4:12] = True
    ring &= ~target
    before = torch.zeros_like(target)
    before[7:9, 7:9] = True
    before[1:3, 1:3] = True
    after = before.clone()
    after[7:9, 7:9] = False
    pred_labels = torch.zeros((16, 16), dtype=torch.int64)
    pred_labels[7:9, 7:9] = 1
    pred_labels[1:3, 1:3] = 2
    feature = torch.ones((1, 2, 4, 4), dtype=torch.float32)
    ledger = build_legal_occupancy_ledger(
        identity=("sample", 1, 1),
        group_id="group",
        target_mask=target,
        ring_mask=ring,
        before_occupancy=before,
        after_occupancy=after,
        pred_labels=pred_labels,
        source_feature=feature,
        synthetic_feature=feature,
        supervision_mask=target,
        valid_mask=torch.ones_like(target),
        feature_size=(4, 4),
    )

    assert ledger.removed_target_fraction == 1.0
    assert ledger.projected_changed_cells >= 1
    assert ledger.deletion_equals_frozen_pred_component is True
    assert ledger.source_feature_fingerprint == ledger.synthetic_feature_fingerprint
    assert ledger.canonical_payload()["classifier_eligible"] is False
    with pytest.raises(TypeError, match="CommonStateRecord"):
        run_block_only_group_oof(
            (ledger,),  # type: ignore[arg-type]
            block="G_full",
            separability=_separability(),
        )


def test_block_oof_is_deterministic_group_disjoint_and_non_authorizing() -> None:
    records = []
    for index in range(12):
        records.extend(
            (
                _explicit_record(
                    sample_id=f"s{index}",
                    group_id=f"g{index}",
                    role="factual",
                    ordinal=0,
                    signal=1.0 + index / 100.0,
                ),
                _explicit_record(
                    sample_id=f"s{index}",
                    group_id=f"g{index}",
                    role="legal",
                    ordinal=1,
                    signal=-1.0 + index / 100.0,
                ),
            )
        )
    config = _separability()
    first = run_block_only_group_oof(
        records,
        block="G_full",
        separability=config,
        feature_components=1,
    )
    second = run_block_only_group_oof(
        tuple(reversed(records)),
        block="G_full",
        separability=config,
        feature_components=1,
    )

    assert first == second
    assert first["estimands"]["group_balanced_oof_auc"] > 0.95
    assert 0.0 <= first["auc_bootstrap"]["lower"] <= 1.0
    assert 0.0 <= first["auc_bootstrap"]["upper"] <= 1.0
    assert isfinite(
        first["estimands"]["group_balanced_cross_fitted_log_loss"]
    )
    for fold in first["folds"]:
        assert not set(fold["train_groups"]) & set(fold["test_groups"])
    assert all(value is False for value in first["authority"].values())
    assert first["not_an_independent_causal_effect"] is True
    composite = run_composite_group_oof(
        records,
        blocks=("F_local", "F_background_global", "O"),
        separability=config,
        feature_components=1,
    )
    assert composite["blocks"] == [
        "F_local",
        "F_background_global",
        "O",
    ]
    assert set(composite["block_definitions"]) == {
        "F_local",
        "F_background_global",
        "O",
    }


def test_exact_same_source_subset_and_centered_sensitivity_use_frozen_counts() -> None:
    records = []
    for index in range(14):
        records.extend(
            (
                _explicit_record(
                    sample_id=f"both{index}",
                    group_id=f"g{index}",
                    role="factual",
                    ordinal=0,
                    signal=2.0 + index,
                ),
                _explicit_record(
                    sample_id=f"both{index}",
                    group_id=f"g{index}",
                    role="legal",
                    ordinal=1,
                    signal=-2.0 + index,
                ),
            )
        )
        if index < 4:
            records.append(
                _explicit_record(
                    sample_id=f"both{index}",
                    group_id=f"g{index}",
                    role="factual",
                    ordinal=2,
                    signal=2.2 + index,
                )
            )
        if index < 7:
            records.append(
                _explicit_record(
                    sample_id=f"both{index}",
                    group_id=f"g{index}",
                    role="legal",
                    ordinal=3,
                    signal=-2.2 + index,
                )
            )
    records.extend(
        (
            _explicit_record(
                sample_id="factual-only",
                group_id="gf",
                role="factual",
                ordinal=0,
                signal=3.0,
            ),
            _explicit_record(
                sample_id="legal-only",
                group_id="gl",
                role="legal",
                ordinal=0,
                signal=-3.0,
            ),
        )
    )
    expectation = SameSourceExpectation(
        sources=14,
        factual_targets=18,
        legal_targets=21,
    )
    subset = exact_same_source_subset(records, expectation=expectation)

    assert len({item.sample_id for item in subset}) == 14
    assert sum(item.role == "factual" for item in subset) == 18
    assert sum(item.role == "legal" for item in subset) == 21
    assert {item.sample_id for item in subset}.isdisjoint(
        {"factual-only", "legal-only"}
    )
    result = run_exact_same_source_sensitivity(
        records,
        blocks=("G_full",),
        separability=_separability(folds=5),
        expectation=expectation,
        feature_components=1,
    )
    assert result["population"] == {
        "sources": 14,
        "factual_targets": 18,
        "legal_targets": 21,
    }
    assert result["source_centering"] == "label-blind-within-sample-mean-v1"
    assert all(value is False for value in result["authority"].values())
    shared = shared_manifest_group_subset(
        records,
        expectation=SharedGroupExpectation(
            groups=14,
            factual_targets=18,
            legal_targets=21,
        ),
    )
    assert len(shared) == 39


def test_block_coverage_mmd_wrapper_is_deterministic_and_non_formal() -> None:
    records = [
        _explicit_record(
            sample_id=f"f{index}",
            group_id=f"fg{index}",
            role="factual",
            ordinal=0,
            signal=20.0 + index,
        )
        for index in range(4)
    ]
    records.extend(
        _explicit_record(
            sample_id=f"l{index}",
            group_id=f"lg{index}",
            role="legal",
            ordinal=0,
            signal=float(index) + 0.1 * index * index,
        )
        for index in range(10)
    )
    first = run_block_coverage_mmd(
        records,
        block="G_full",
        overlap=_overlap(k=2),
        separability=_separability(),
        feature_components=1,
    )
    second = run_block_coverage_mmd(
        tuple(reversed(records)),
        block="G_full",
        overlap=_overlap(k=2),
        separability=_separability(),
        feature_components=1,
    )

    assert first == second
    assert first["coverage"]["factual_total"] == 4
    assert first["mmd"]["groups"]["factual"] == 4
    assert first["mmd"]["groups"]["legal_exclusive"] == 10
    assert first["formal_p0_gate"] is False
    assert "pass" not in first["coverage"]
    assert "descriptive_threshold_crossed" in first["coverage"]
    assert "pass" not in first["mmd"]
    assert "observed_within_legal_reference_q95" in first["mmd"]
    assert all(value is False for value in first["authority"].values())
