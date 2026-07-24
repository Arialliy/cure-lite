from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cure_lite.experiment import exposure_audit
from cure_lite.experiment.exposure_audit import (
    _count_stats,
    _gini,
)
from cure_lite.experiment.p0_geometry import _resize_mask, _scaled_centroid
from cure_lite.experiment.p0_protocol import load_p0_config
from cure_lite.experiment.p0_support import (
    _TargetRecord,
    _fit_feature_projector,
    _group_mmd_u,
    _group_kth_distance,
    _higher_quantile,
    _mmd_receipt,
    _project_feature,
    _robust_scale,
    _robust_scale_fit,
    _weighted_auc,
)
from cure_lite.train.pools import (
    BranchPools,
    StateExample,
    _draw,
)
from cure_lite.types import BranchSupervision


_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "protocols"
    / "IRSTD-1K"
    / "p0_v1"
    / "p0_config.json"
)


def _example(sample_id: str, branch: str) -> StateExample:
    target = torch.zeros((1, 4, 4), dtype=torch.float32)
    positive_ids: tuple[int, ...] = ()
    reachable_ids: tuple[int, ...] = ()
    if branch in {"factual_miss", "synthetic"}:
        target[0, 0, 0] = 1.0
        positive_ids = (1,)
        if branch == "factual_miss":
            reachable_ids = (1,)
    return StateExample(
        sample_id=sample_id,
        feature=torch.zeros((1, 2, 2, 2), dtype=torch.float32),
        supervision=BranchSupervision(
            occupancy=torch.zeros((1, 4, 4), dtype=torch.bool),
            target=target,
            valid_mask=torch.ones((1, 4, 4), dtype=torch.bool),
            branch=branch,
            positive_gt_ids=positive_ids,
            reachable_gt_ids=reachable_ids,
        ),
    )


def test_exposure_replay_can_consume_the_frozen_training_draw() -> None:
    pools = BranchPools(
        factual_miss=tuple(
            _example(f"f-{index}", "factual_miss") for index in range(3)
        ),
        synthetic=tuple(
            _example(f"s-{index}", "synthetic") for index in range(5)
        ),
    )
    first = _draw(
        pools.synthetic,
        4,
        branch="synthetic",
        epoch=7,
        step=3,
        global_seed=42,
    )
    second = _draw(
        pools.synthetic,
        4,
        branch="synthetic",
        epoch=7,
        step=3,
        global_seed=42,
    )
    assert first == second
    assert all(item in pools.synthetic for item in first)


def test_nearest_resize_can_remove_a_native_pixel() -> None:
    mask = torch.zeros((4, 4), dtype=torch.bool)
    mask[0, 0] = True
    assert not torch.any(_resize_mask(mask, (2, 2)))


def test_pixel_center_centroid_scaling_is_explicit() -> None:
    assert _scaled_centroid((0.0, 0.0), (4, 4), (2, 2)) == pytest.approx(
        (-0.25, -0.25)
    )
    assert _scaled_centroid((1.5, 1.5), (4, 4), (2, 2)) == pytest.approx(
        (0.5, 0.5)
    )


def test_group_kth_distance_excludes_the_query_group() -> None:
    references = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float64)
    distance = _group_kth_distance(
        torch.tensor([0.0], dtype=torch.float64),
        "g0",
        references,
        ("g0", "g1", "g2", "g3"),
        2,
    )
    assert distance == pytest.approx(2.0)


def test_higher_quantile_uses_frozen_order_statistic() -> None:
    assert _higher_quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0
    assert _higher_quantile([1.0, 2.0, 3.0, 4.0], 0.95) == 4.0


def test_weighted_auc_handles_ties_and_group_weights() -> None:
    assert _weighted_auc(
        [0.9, 0.5, 0.5, 0.1],
        [1, 1, 0, 0],
        [0.25, 0.25, 0.25, 0.25],
    ) == pytest.approx(0.875)


def test_robust_scale_preserves_zero_mad_and_constant_dimensions() -> None:
    legal = torch.tensor(
        [[0.0, 2.0], [0.0, 2.0], [1.0, 2.0]],
        dtype=torch.float64,
    )
    factual = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    median, scale, constant, maxdev = _robust_scale_fit(legal)
    projected = _robust_scale(factual, median, scale)
    assert maxdev.tolist() == [True, False]
    assert constant.tolist() == [False, True]
    assert projected[0, 0] == pytest.approx(2.0)
    assert projected[0, 1] > 1e10


def test_feature_projection_keeps_legal_subspace_residual() -> None:
    legal = torch.tensor(
        [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        dtype=torch.float64,
    )
    projector = _fit_feature_projector(legal, 1)
    factual = _project_feature(
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        projector,
    )
    assert factual.shape == (1, 2)
    assert factual[0, -1] > 1e10


def test_group_mmd_u_excludes_group_diagonals_and_keeps_negative_value() -> None:
    kernel = torch.tensor(
        [
            [1.0, 0.2, 0.5, 0.5],
            [0.2, 1.0, 0.5, 0.5],
            [0.5, 0.5, 1.0, 0.2],
            [0.5, 0.5, 0.2, 1.0],
        ],
        dtype=torch.float64,
    )
    assert _group_mmd_u(kernel, [0, 1], [2, 3]) == pytest.approx(-0.6)


def _record(identity: tuple[str, int, int | None], group: str, role: str):
    return _TargetRecord(
        identity=identity,
        sample_id=identity[0],
        group_id=group,
        role=role,
        hand=torch.zeros(1, dtype=torch.float64),
        joint_feature_raw=torch.zeros(1, dtype=torch.float64),
        joint_occupancy_raw=torch.zeros(1, dtype=torch.float64),
    )


def test_mmd_uses_matched_source_disjoint_group_sizes() -> None:
    config = load_p0_config(_CONFIG).separability
    factual = (
        _record(("f0", 1, None), "f0", "factual"),
        _record(("f1", 1, None), "f1", "factual"),
    )
    legal = (
        _record(("same", 1, 1), "f0", "legal"),
        _record(("l0", 1, 1), "l0", "legal"),
        _record(("l1", 1, 1), "l1", "legal"),
        _record(("l2", 1, 1), "l2", "legal"),
        _record(("l3", 1, 1), "l3", "legal"),
    )
    receipt = _mmd_receipt(
        factual,
        legal,
        torch.tensor([[4.0], [5.0]], dtype=torch.float64),
        torch.tensor([[9.0], [0.0], [1.0], [2.0], [3.0]], dtype=torch.float64),
        space="test",
        config=config,
    )
    assert receipt["groups"] == {
        "factual": 2,
        "legal_all": 5,
        "overlap_removed_from_legal": 1,
        "legal_exclusive": 4,
        "observed_left": 2,
        "matched_null_left": 2,
        "shared_right": 2,
    }
    assert (
        receipt["observed_factual_vs_matched_legal"]["replicates"]
        == config.mmd_reference_replicates
    )


def test_p0_d_does_not_construct_candidate_when_upstream_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_p0_config(_CONFIG)
    identity = ("legal", 1, 1)
    monkeypatch.setattr(
        exposure_audit,
        "_legal_catalog",
        lambda catalog: ((identity,), {}, {identity: "legal"}),
    )
    monkeypatch.setattr(
        exposure_audit,
        "_historical_sequence",
        lambda *args, **kwargs: {"variant": kwargs["variant"]},
    )

    def forbidden(**kwargs):
        raise AssertionError("candidate proposal must remain unreachable")

    monkeypatch.setattr(exposure_audit, "_build_candidate_proposal", forbidden)
    catalog = SimpleNamespace(
        entries=(),
        miss_alignment_fingerprint="a" * 64,
    )
    manifest = SimpleNamespace(records_for=lambda split: ())
    result = exposure_audit.build_p0_d_exposure(
        catalog,
        manifest,
        config.overlap,
        config.exposure,
        upstream_pass=False,
        factual_hand=torch.empty((0, 1), dtype=torch.float64),
        legal_hand=torch.empty((0, 1), dtype=torch.float64),
    )
    assert result["candidate_s"]["status"] == "not_evaluated"
    assert result["p0_d_pass"] is None


def test_exposure_statistics_include_zero_support_and_top_k() -> None:
    stats = _count_stats({("a", 1, 1): 4, ("b", 1, 1): 0, ("c", 1, 1): 2})
    assert stats["total_exposures"] == 6
    assert stats["unique_exposed"] == 2
    assert stats["zero_exposure"] == 1
    assert stats["maximum_share"] == pytest.approx(4 / 6)
    assert stats["ess"] == pytest.approx(36 / 20)
    assert stats["top1"]["share"] == pytest.approx(4 / 6)
    assert _gini([0, 2, 4]) == pytest.approx(4 / 9)
