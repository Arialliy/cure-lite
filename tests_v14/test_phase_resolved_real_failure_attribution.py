from __future__ import annotations

import json
import math

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.experiment import (
    phase_resolved_real_failure_attribution as attribution,
)
from cure_lite.experiment.phase_resolved_real_failure_attribution import (
    PFCR_REAL_FAILURE_ATTRIBUTION_SCHEMA,
    PFCRAttributionState,
    load_published_pfcr_real_d_r_failure_attribution,
    local_occupancy_support_row,
    publish_pfcr_real_d_r_failure_attribution,
    residual_score_row,
    run_pfcr_real_d_r_failure_attribution,
    spearman_correlation,
)
from cure_lite.train.pools import StateExample
from cure_lite.types import BranchSupervision


def _state(
    *,
    occupancy_pixels: tuple[tuple[int, int], ...] = (),
    target_pixels: tuple[tuple[int, int], ...] = ((16, 16),),
    branch: str = "factual_miss",
) -> PFCRAttributionState:
    occupancy = torch.zeros((1, 20, 20), dtype=torch.bool)
    for y, x in occupancy_pixels:
        occupancy[0, y, x] = True
    target = torch.zeros((1, 20, 20), dtype=torch.float32)
    for y, x in target_pixels:
        target[0, y, x] = 1.0
    valid = torch.ones((1, 20, 20), dtype=torch.bool)
    valid &= ~occupancy
    positive_ids = () if branch == "factual_no_miss" else (1,)
    supervision = BranchSupervision(
        occupancy=occupancy,
        target=target,
        valid_mask=valid,
        branch=branch,
        positive_gt_ids=positive_ids,
        reachable_gt_ids=positive_ids,
    )
    example = StateExample(
        sample_id="sample",
        feature=torch.zeros((1, 2, 5, 5), dtype=torch.float32),
        supervision=supervision,
    )
    identity = (
        ("sample", None, None)
        if branch == "factual_no_miss"
        else ("sample", 1, None)
    )
    return PFCRAttributionState(
        branch=branch,
        identity=identity,
        example=example,
    )


def test_local_coverage_participation_distinguishes_near_and_far() -> None:
    far = _state(occupancy_pixels=((0, 0),))
    near = _state(occupancy_pixels=((12, 12),))

    far_row = local_occupancy_support_row(
        far,
        feature_shape=(5, 5),
    )
    near_row = local_occupancy_support_row(
        near,
        feature_shape=(5, 5),
    )

    assert far_row["positive_feature_cell_count"] == 1
    assert far_row["active_positive_feature_cell_count"] == 0
    assert far_row["any_local_occupancy_support"] is False
    assert near_row["active_positive_feature_cell_count"] == 1
    assert near_row["any_local_occupancy_support"] is True
    assert (
        near_row["all_positive_cells_local_occupancy_support"] is True
    )


def test_local_coverage_requires_a_positive_state() -> None:
    state = _state(
        target_pixels=(),
        branch="factual_no_miss",
    )
    with pytest.raises(ValueError, match="positive state"):
        local_occupancy_support_row(
            state,
            feature_shape=(5, 5),
        )


def test_residual_score_row_reports_endpoint_ordering() -> None:
    state = _state(target_pixels=((16, 16), (16, 17)))
    probability = torch.full((1, 1, 20, 20), 0.1)
    probability[0, 0, 16, 16] = 0.8
    probability[0, 0, 16, 17] = 0.6
    probability[0, 0, 0, 1] = 0.9

    row = residual_score_row(probability, state)

    assert row["target_min"] == pytest.approx(0.6)
    assert row["target_mean"] == pytest.approx(0.7)
    assert row["target_max"] == pytest.approx(0.8)
    assert row["background_max"] == pytest.approx(0.9)
    assert row["margin_target_min_vs_background_max"] == pytest.approx(
        -0.3
    )
    assert math.isfinite(float(row["background_q999"]))


def test_residual_score_row_supports_no_miss_background() -> None:
    state = _state(
        target_pixels=(),
        branch="factual_no_miss",
    )
    probability = torch.full((1, 1, 20, 20), 0.02)
    row = residual_score_row(probability, state)
    assert set(row) == {
        "branch",
        "identity",
        "background_max",
        "background_q999",
    }


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1.0, 2.0, 3.0], [10.0, 20.0, 30.0], 1.0),
        ([1.0, 2.0, 3.0], [30.0, 20.0, 10.0], -1.0),
        ([1.0, 1.0, 2.0], [5.0, 5.0, 8.0], 1.0),
    ],
)
def test_spearman_correlation(
    left: list[float],
    right: list[float],
    expected: float,
) -> None:
    assert spearman_correlation(left, right) == pytest.approx(expected)


def test_spearman_constant_input_is_explicitly_undefined() -> None:
    assert spearman_correlation(
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0],
    ) is None


def test_public_runner_rejects_non_catalog_before_data_access() -> None:
    with pytest.raises(TypeError, match="cache"):
        run_pfcr_real_d_r_failure_attribution(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            (),
            device_by_seed={},
        )


def _residual_row(
    branch: str,
    identity: list[object],
    *,
    background_max: float,
    background_q999: float,
    target_min: float | None = None,
    target_mean: float | None = None,
    target_max: float | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "branch": branch,
        "identity": identity,
        "background_max": background_max,
        "background_q999": background_q999,
    }
    if target_min is not None:
        assert target_mean is not None and target_max is not None
        row.update(
            {
                "target_min": target_min,
                "target_mean": target_mean,
                "target_max": target_max,
                "margin_target_min_vs_background_max": (
                    target_min - background_max
                ),
                "margin_target_max_vs_background_q999": (
                    target_max - background_q999
                ),
            }
        )
    return row


def _canonical_result() -> dict[str, object]:
    local_rows = [
        {
            "branch": "factual_miss",
            "identity": ["a", 1, None],
            "positive_pixel_count": 1,
            "positive_feature_cell_count": 1,
            "active_positive_feature_cell_count": 0,
            "active_positive_feature_cell_fraction": 0.0,
            "any_local_occupancy_support": False,
            "all_positive_cells_local_occupancy_support": False,
        },
        {
            "branch": "factual_miss",
            "identity": ["b", 1, None],
            "positive_pixel_count": 2,
            "positive_feature_cell_count": 2,
            "active_positive_feature_cell_count": 1,
            "active_positive_feature_cell_fraction": 0.5,
            "any_local_occupancy_support": True,
            "all_positive_cells_local_occupancy_support": False,
        },
        {
            "branch": "synthetic",
            "identity": ["c", 1, 1],
            "positive_pixel_count": 1,
            "positive_feature_cell_count": 1,
            "active_positive_feature_cell_count": 1,
            "active_positive_feature_cell_fraction": 1.0,
            "any_local_occupancy_support": True,
            "all_positive_cells_local_occupancy_support": True,
        },
    ]
    identities = (
        ("factual_miss", ["a", 1, None]),
        ("factual_miss", ["b", 1, None]),
        ("factual_no_miss", ["d", None, None]),
        ("synthetic", ["c", 1, 1]),
    )
    seed_results = []
    for offset, seed in enumerate((42, 43)):
        rows = [
            _residual_row(
                branch,
                identity,
                background_max=0.2,
                background_q999=0.1,
                target_min=(0.3 + 0.1 * index + 0.01 * offset)
                if branch != "factual_no_miss"
                else None,
                target_mean=(0.4 + 0.1 * index + 0.01 * offset)
                if branch != "factual_no_miss"
                else None,
                target_max=(0.5 + 0.1 * index + 0.01 * offset)
                if branch != "factual_no_miss"
                else None,
            )
            for index, (branch, identity) in enumerate(identities)
        ]
        grouped = {
            branch: tuple(
                row for row in rows if row["branch"] == branch
            )
            for branch in ("factual_miss", "factual_no_miss", "synthetic")
        }
        seed_results.append(
            {
                "seed": seed,
                "attempt_complete_fingerprint": (
                    ("a" if seed == 42 else "b") * 64
                ),
                "decoder_artifact_fingerprint": (
                    ("c" if seed == 42 else "d") * 64
                ),
                "decoder_state_fingerprint": (
                    ("e" if seed == 42 else "f") * 64
                ),
                "rows": rows,
                "branch_summaries": {
                    branch: attribution._branch_summary(grouped[branch])
                    for branch in grouped
                },
            }
        )
    core: dict[str, object] = {
        "schema_version": PFCR_REAL_FAILURE_ATTRIBUTION_SCHEMA,
        "model": "CURE-Lite",
        "method": "PFCR",
        "runtime_split": "D_R",
        "cache_contract_fingerprint": "1" * 64,
        "state_catalog_fingerprint": "2" * 64,
        "lineage_allowlist_fingerprint": "3" * 64,
        "population": {
            "factual_miss_targets": 2,
            "factual_no_miss_states": 1,
            "synthetic_targets": 1,
        },
        "local_occupancy_support": {
            "rows": local_rows,
            "summaries": attribution._local_support_summaries(
                local_rows
            ),
            "mathematical_boundary": "test-only boundary",
        },
        "seed_results": seed_results,
        "seed_consistency": attribution._seed_consistency(
            seed_results[0],
            seed_results[1],
        ),
        "execution_binding": {
            "seed_order": [42, 43],
            "device_by_seed": {"42": "cpu", "43": "cpu"},
            "device_name_by_seed": {"42": "cpu", "43": "cpu"},
            "torch_version": "test",
            "cuda_version": None,
            "deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        "implementation_binding": {
            "cure_lite/test.py": "4" * 64,
        },
        "interpretation_boundary": {
            "diagnostic_only": True,
            "causal_failure_attribution_established": False,
            "replacement_equation_authorized": False,
            "new_formal_training_authorized": False,
            "D_V_read": False,
            "D_T_read": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
        },
    }
    return {
        **core,
        "result_fingerprint": stable_fingerprint(core),
    }


def test_publish_and_strict_load_are_create_only(tmp_path) -> None:
    output = tmp_path / "attribution"
    published = publish_pfcr_real_d_r_failure_attribution(
        output,
        _canonical_result(),
    )
    loaded = load_published_pfcr_real_d_r_failure_attribution(output)

    assert published.result_fingerprint == loaded.result_fingerprint
    assert published.complete_fingerprint == loaded.complete_fingerprint
    published.verify_unchanged()
    loaded.verify_unchanged()
    assert set(path.name for path in output.iterdir()) == {
        "result.json",
        "COMPLETE.json",
    }
    assert loaded.result["runtime_split"] == "D_R"
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        publish_pfcr_real_d_r_failure_attribution(
            output,
            _canonical_result(),
        )


def test_strict_loader_rejects_result_tampering(tmp_path) -> None:
    output = tmp_path / "attribution"
    publish_pfcr_real_d_r_failure_attribution(
        output,
        _canonical_result(),
    )
    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["runtime_split"] = "D_V"
    result_path.write_text(
        json.dumps(result, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fingerprint/identity"):
        load_published_pfcr_real_d_r_failure_attribution(output)


def test_strict_loader_rejects_inventory_expansion(tmp_path) -> None:
    output = tmp_path / "attribution"
    publish_pfcr_real_d_r_failure_attribution(
        output,
        _canonical_result(),
    )
    (output / "extra.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inventory"):
        load_published_pfcr_real_d_r_failure_attribution(output)


def test_publisher_rejects_incomplete_self_fingerprinted_result(
    tmp_path,
) -> None:
    core = {
        "schema_version": PFCR_REAL_FAILURE_ATTRIBUTION_SCHEMA,
        "runtime_split": "D_R",
    }
    incomplete = {
        **core,
        "result_fingerprint": stable_fingerprint(core),
    }
    with pytest.raises(ValueError, match="fields"):
        publish_pfcr_real_d_r_failure_attribution(
            tmp_path / "incomplete",
            incomplete,
        )


def test_loaded_result_detects_in_memory_mutation(tmp_path) -> None:
    output = tmp_path / "attribution"
    loaded = publish_pfcr_real_d_r_failure_attribution(
        output,
        _canonical_result(),
    )
    assert isinstance(loaded.result, dict)
    loaded.result["runtime_split"] = "D_V"

    with pytest.raises(ValueError, match="fingerprint/identity"):
        loaded.verify_unchanged()


def test_residual_score_rejects_out_of_range_probability() -> None:
    state = _state()
    probability = torch.zeros((1, 1, 20, 20))
    probability[0, 0, 0, 0] = 1.1

    with pytest.raises(ValueError, match=r"\[0,1\]"):
        residual_score_row(probability, state)
