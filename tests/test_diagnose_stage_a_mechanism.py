from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from cure_lite.config import MatchConfig
from cure_lite.instances import instances_from_binary_mask
from cure_lite.matching import match_components
from tools import diagnose_stage_a_mechanism as diagnostic


def _call_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def test_cli_is_selected_point_diagnosis_only() -> None:
    source = Path(diagnostic.__file__).resolve()
    calls = _call_names(source)
    assert {
        "load_stage_a_run",
        "evaluate_candidate_ledger",
        "calibrate_paired_gate2",
        "evaluate_paired_gate2",
        "run_paired_gate2_training",
        "save_completed_decoder_run",
    }.isdisjoint(calls)
    text = source.read_text(encoding="utf-8")
    assert "D_T input" in text
    parser = diagnostic.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert {"--manifest", "--reference-base-run", "--run", "--output"} <= options
    assert {
        "--epochs",
        "--resume",
        "--threshold",
        "--device",
        "--d-t",
    }.isdisjoint(options)


def test_output_must_be_create_only_and_outside_stage_runs(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    output = tmp_path / "diagnostic.json"
    assert diagnostic._prepare_output(output, (stage,)) == output

    existing = tmp_path / "existing.json"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        diagnostic._prepare_output(existing, (stage,))
    assert existing.read_text(encoding="utf-8") == "keep"

    with pytest.raises(ValueError, match="outside"):
        diagnostic._prepare_output(stage / "new.json", (stage,))


def test_output_rejects_a_symbolic_link_parent(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    actual = tmp_path / "actual"
    actual.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        diagnostic._prepare_output(linked / "result.json", (stage,))


def _target_mask(
    shape: tuple[int, int],
    origins: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    result = torch.zeros(shape, dtype=torch.bool)
    for y, x in origins:
        result[y : y + 2, x : x + 2] = True
    return result


def _outcome(
    sample_id: str,
    residual: torch.Tensor,
    residual_mask: torch.Tensor,
    occupancy: torch.Tensor,
    gt: object,
) -> diagnostic._MethodImageOutcome:
    prediction = occupancy | residual_mask
    pred = instances_from_binary_mask(prediction, connectivity=8, min_area=1)
    return diagnostic._MethodImageOutcome(
        sample_id=sample_id,
        residual_probability=residual,
        residual_mask=residual_mask,
        prediction=prediction,
        pred_instances=pred,
        match=match_components(pred, gt, MatchConfig()),
    )


def test_target_partition_uses_final_matching_and_inclusive_thresholds() -> None:
    shape = (24, 24)
    origins = ((1, 1), (1, 8), (8, 1), (8, 8), (17, 17))
    gt_mask = _target_mask(shape, origins)
    gt = instances_from_binary_mask(gt_mask, connectivity=8, min_area=1)
    occupancy = _target_mask(shape, (origins[4],))
    anchor = instances_from_binary_mask(occupancy, connectivity=8, min_area=1)
    base = torch.zeros(shape, dtype=torch.float32)
    base[occupancy] = 1.0
    feature = torch.arange(2 * 6 * 6, dtype=torch.float32).reshape(1, 2, 6, 6)
    prepared = diagnostic._PreparedDVRow(
        sample_id="sample",
        base_probability=base,
        feature=feature,
        occupancy=occupancy,
        anchor_instances=anchor,
        gt_instances=gt,
        anchor_match=match_components(anchor, gt, MatchConfig()),
    )

    f_residual = torch.zeros(shape, dtype=torch.float32)
    u_residual = torch.zeros(shape, dtype=torch.float32)
    for index in (0, 2):
        y, x = origins[index]
        f_residual[y : y + 2, x : x + 2] = 0.78
    for index in (1, 2):
        y, x = origins[index]
        u_residual[y : y + 2, x : x + 2] = 1.0
    f_mask = (f_residual >= 0.78) & ~occupancy
    u_mask = (u_residual >= 1.0) & ~occupancy

    payload = diagnostic._build_d_v_partition(
        (prepared,),
        (_outcome("sample", f_residual, f_mask, occupancy, gt),),
        (_outcome("sample", u_residual, u_mask, occupancy, gt),),
        f_threshold=0.78,
        u_threshold=1.0,
    )
    rows = {
        row["gt_id"]: row
        for row in payload["targets"]
        if row["anchor_miss"]
    }
    assert {gt_id: row["category"] for gt_id, row in rows.items()} == {
        1: "f_only",
        2: "u_only",
        3: "both",
        4: "neither",
    }
    assert payload["anchor_miss_targets"] == {
        "target_scope": "anchor_misses",
        "total_targets": 4,
        "counts": {
            "f_only": 1,
            "u_only": 1,
            "both": 1,
            "neither": 1,
        },
        "fractions": {
            "f_only": 0.25,
            "u_only": 0.25,
            "both": 0.25,
            "neither": 0.25,
        },
        "f_matched_targets": 2,
        "u_matched_targets": 2,
    }
    assert payload["all_targets"]["total_targets"] == 5
    assert payload["all_targets"]["counts"]["both"] == 2
    assert rows[3]["f_residual"]["active_pixels_in_gt"] == 4
    assert rows[3]["u_residual"]["active_pixels_in_gt"] == 4


def test_feature_descriptor_uses_area_weights_and_preserves_channels() -> None:
    feature = torch.tensor(
        [
            [
                [[1.0, 3.0], [5.0, 7.0]],
                [[2.0, 4.0], [6.0, 8.0]],
            ]
        ]
    )
    mask = torch.zeros((4, 4), dtype=torch.bool)
    mask[:2, :2] = True
    descriptor = diagnostic._feature_descriptor(feature, mask)
    assert descriptor["feature_cells"] == 1
    assert descriptor["feature_embedding"] == pytest.approx([1.0, 2.0])
    assert len(descriptor["feature_channel_weighted_std"]) == 2
    assert descriptor["feature_weight_sum"] == pytest.approx(1.0)


def _diagnostic_run(seed: int, categories: tuple[str, ...]) -> dict[str, object]:
    return {
        "seed": seed,
        "d_v_partition": {
            "targets": [
                {
                    "sample_id": "sample",
                    "gt_id": index + 1,
                    "anchor_miss": True,
                    "category": category,
                }
                for index, category in enumerate(categories)
            ]
        },
    }


def test_cross_seed_transition_table_is_exact_and_stable() -> None:
    payload = diagnostic._cross_seed_payload(
        (
            _diagnostic_run(43, ("both", "u_only")),
            _diagnostic_run(42, ("f_only", "u_only")),
        )
    )
    assert payload["ordered_seeds"] == [42, 43]
    assert payload["shared_target_count"] == 2
    assert payload["transition_counts"] == {
        "f_only -> both": 1,
        "u_only -> u_only": 1,
    }


def test_cross_seed_rejects_different_target_membership() -> None:
    second = _diagnostic_run(43, ("both",))
    second["d_v_partition"]["targets"][0]["sample_id"] = "other"
    with pytest.raises(RuntimeError, match="identities differ"):
        diagnostic._cross_seed_payload(
            (_diagnostic_run(42, ("f_only",)), second)
        )


def test_diagnostic_tool_is_outside_the_versioned_method_source_tree() -> None:
    from cure_lite.experiment.stage_a_runner import _SOURCE_ROOT, _source_tree_digest

    diagnostic_source = Path(diagnostic.__file__).resolve()
    assert _SOURCE_ROOT.resolve() not in diagnostic_source.parents
    current_digest = _source_tree_digest()
    assert len(current_digest) == 64
    assert set(current_digest) <= set("0123456789abcdef")
