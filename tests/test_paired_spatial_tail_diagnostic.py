from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.decoder import CURELiteDecoder, project_occupancy_to_feature_grid
from cure_lite.experiment.paired_spatial_tail_diagnostic import (
    SPATIAL_TAIL_POPULATION_SCHEMA,
    evaluate_spatial_tail_populations,
    summarize_pair_spatial_tail,
    validate_spatial_tail_specification,
)
from cure_lite.paired_types import (
    PairExample,
    stack_pair_examples,
    tensor_content_fingerprint,
)
from tools import run_paired_spatial_tail_diagnostic as runner


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "paired_spatial_tail_diagnostic_v1"
    / "config.json"
)


def _specification() -> dict[str, object]:
    return {
        "absolute_delta_thresholds": [0.1, 0.25, 0.5],
        "quantiles": [0.5, 0.9, 0.99, 0.999],
        "deleted_component_neighborhood_radii_px": [1, 2, 4, 8],
        "connected_component_connectivity": 8,
        "projected_cell_output_mapping": (
            "nearest-output-support-of-xor-projected-endpoints-v1"
        ),
        "thresholds_are_descriptive_not_gates": True,
    }


def _pair(
    kind: str,
    *,
    digit: str,
    target_pixel: tuple[int, int] = (2, 2),
) -> PairExample:
    feature = torch.ones((1, 2, 2, 2), dtype=torch.float32)
    valid = torch.ones((1, 8, 8), dtype=torch.bool)
    empty = torch.zeros_like(valid)
    plus = empty.clone()
    minus = empty.clone()
    increment = empty.clone()
    if kind in {"clean_positive", "component_null"}:
        plus[0, target_pixel[0], target_pixel[1]] = True
    if kind == "clean_positive":
        increment[0, target_pixel[0], target_pixel[1]] = True
    removed = plus & ~minus
    projected_plus = project_occupancy_to_feature_grid(
        plus.unsqueeze(0),
        (2, 2),
    )
    projected_minus = project_occupancy_to_feature_grid(
        minus.unsqueeze(0),
        (2, 2),
    )
    visible = not torch.equal(projected_plus, projected_minus)
    clean = increment.clone()
    return PairExample(
        pair_id=digit * 64,
        pair_kind=kind,
        sample_id=f"sample-{digit}",
        group_id=f"group-{digit}",
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
        removed_component=removed,
        image_valid_mask=valid,
        completion_plus=empty,
        completion_minus=increment,
        label_increment=increment.to(torch.float32),
        clean_increment=clean,
        evaluation_gt_id=1 if kind == "clean_positive" else None,
        native_gt_id=1 if kind == "clean_positive" else None,
        pred_id=1 if kind != "identity_null" else None,
        feature_fingerprint=tensor_content_fingerprint(feature),
        before_match_fingerprint="a" * 64,
        after_match_fingerprint=(
            "b" * 64 if kind != "identity_null" else "a" * 64
        ),
        projected_occupancy_plus_fingerprint=(
            tensor_content_fingerprint(projected_plus)
        ),
        projected_occupancy_minus_fingerprint=(
            tensor_content_fingerprint(projected_minus)
        ),
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


def test_frozen_config_fingerprint_and_execution_boundary() -> None:
    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    unsigned = dict(config)
    fingerprint = unsigned.pop("config_fingerprint")

    assert fingerprint == runner.SPATIAL_CONFIG_FINGERPRINT
    assert stable_fingerprint(unsigned) == fingerprint
    assert runner._load_config(_CONFIG) == config
    assert config["decision_semantics"]["retroactive_bounded_gate_added"] is False
    assert config["decision_semantics"]["authorizes_formal_800"] is False
    assert config["execution_policy"]["allow_D_V"] is False
    assert config["execution_policy"]["allow_D_T"] is False
    assert config["replay_contract"]["optimizer_updates"] == 400
    assert config["replay_contract"]["checkpoint_recovery"] is False


def test_spatial_summary_preserves_sparse_signed_tail_and_locality() -> None:
    pair = _pair("component_null", digit="1")
    batch = stack_pair_examples((pair,), device="cpu")
    delta = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    delta[0, 0, 2, 2] = 0.9
    delta[0, 0, 2, 3] = 0.2
    delta[0, 0, 7, 7] = -0.6

    result = summarize_pair_spatial_tail(
        delta,
        batch,
        (pair,),
        _specification(),
    )

    assert result["schema_version"] == SPATIAL_TAIL_POPULATION_SCHEMA
    assert result["pair_count"] == 1
    assert result["pair_kind"] == "component_null"
    row = result["rows"][0]
    assert row["absolute_max_delta"] == pytest.approx(0.9)
    assert row["signed_delta_at_absolute_argmax"] == pytest.approx(0.9)
    assert row["signed_min_delta"] == pytest.approx(-0.6)
    assert row["signed_max_delta"] == pytest.approx(0.9)
    assert row["signed_min_argmin_yx"] == [7, 7]
    assert row["signed_max_argmax_yx"] == [2, 2]
    assert row["absolute_argmax_yx"] == [2, 2]
    assert row["removed_component_pixel_count"] == 1
    assert row["label_increment_pixel_count"] == 0
    assert row["rms_delta"] == pytest.approx(
        ((0.9**2 + 0.2**2 + 0.6**2) / 64) ** 0.5
    )
    assert row["absolute_argmax_distance_to_support"][
        "deleted_component"
    ] == {"euclidean_px": 0.0, "chebyshev_px": 0.0}

    low = row["thresholds"]["0.100"]
    assert low["absolute_pixel_count"] == 3
    assert low["positive_pixel_count"] == 2
    assert low["negative_pixel_count"] == 1
    assert low["connected_components_8"]["count"] == 2
    assert low["connected_components_8"]["areas_px_descending"] == [2, 1]
    assert low["support_overlap"]["deleted_component"][
        "intersection_pixel_count"
    ] == 1
    high = row["thresholds"]["0.500"]
    assert high["absolute_pixel_count"] == 2
    assert high["connected_components_8"]["count"] == 2


def test_identity_null_reports_exact_zero_and_empty_support() -> None:
    pair = _pair("identity_null", digit="2")
    batch = stack_pair_examples((pair,), device="cpu")
    delta = torch.zeros((1, 1, 8, 8), dtype=torch.float32)

    result = summarize_pair_spatial_tail(
        delta,
        batch,
        (pair,),
        _specification(),
    )

    row = result["rows"][0]
    assert result["maximum_abs_delta"] == 0.0
    assert row["removed_component_pixel_count"] == 0
    assert row["projected_changed_feature_cell_count"] == 0
    assert row["absolute_argmax_yx"] == [0, 0]
    assert row["absolute_argmax_distance_to_support"][
        "deleted_component"
    ] is None
    for threshold in ("0.100", "0.250", "0.500"):
        assert row["thresholds"][threshold]["absolute_pixel_count"] == 0
        assert (
            row["thresholds"][threshold]["connected_components_8"]["count"]
            == 0
        )


def test_population_evaluator_accepts_clean_and_both_null_kinds() -> None:
    population = SimpleNamespace(
        clean_pairs=(_pair("clean_positive", digit="4"),),
        component_null=(_pair("component_null", digit="5"),),
        identity_null=(_pair("identity_null", digit="6"),),
    )
    result = evaluate_spatial_tail_populations(
        CURELiteDecoder(feature_channels=2),
        population,
        _specification(),
        device="cpu",
    )

    assert set(result) == {
        "clean_positive",
        "component_null",
        "identity_null",
    }
    assert all(value["pair_count"] == 1 for value in result.values())
    assert result["identity_null"]["maximum_abs_delta"] == 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("absolute_delta_thresholds", [0.1, 0.1, 0.5]),
        ("quantiles", [0.5, 1.0]),
        ("deleted_component_neighborhood_radii_px", [1, 0]),
        ("connected_component_connectivity", 4),
        ("thresholds_are_descriptive_not_gates", False),
    ),
)
def test_spatial_specification_rejects_protocol_drift(
    field: str,
    value: object,
) -> None:
    specification = _specification()
    specification[field] = value
    with pytest.raises((TypeError, ValueError)):
        validate_spatial_tail_specification(specification)


def test_decision_never_changes_bounded_status_or_authorizes_formal() -> None:
    result = {
        "exact_bounded_replay_verified": True,
        "receipt_fingerprint": "a" * 64,
    }
    decision = runner._decision(
        result,
        failure=None,
        evidence_receipt_fingerprint="a" * 64,
    )

    assert decision["status"] == "SPATIAL_TAIL_DIAGNOSTIC_COMPLETE"
    assert decision["descriptive_only"] is True
    assert decision["retroactive_bounded_gate_added"] is False
    assert decision["bounded_decision_changed"] is False
    assert decision["authorizes_formal_800"] is False
    assert decision["authorizes_D_V_or_D_T"] is False
    assert decision["not_performance_evidence"] is True


def test_cli_is_create_only_and_requires_explicit_device_output(
    tmp_path: Path,
) -> None:
    parsed = runner.parse_args(
        [
            "--config",
            str(_CONFIG),
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "new"),
        ]
    )
    assert parsed.device == "cpu"
    assert runner._prepare_output(parsed.output) == parsed.output

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        runner._prepare_output(existing)


def test_cli_returns_nonzero_after_sealing_execution_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = SimpleNamespace()
    monkeypatch.setattr(runner, "parse_args", lambda argv: arguments)
    monkeypatch.setattr(
        runner,
        "run",
        lambda args: {
            "status": "SPATIAL_TAIL_DIAGNOSTIC_EXECUTION_ERROR",
            "output": "/sealed/failure",
        },
    )

    with pytest.raises(SystemExit) as error:
        runner.main([])

    assert error.value.code == 2
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "SPATIAL_TAIL_DIAGNOSTIC_EXECUTION_ERROR"


def test_diagnostic_implementation_does_not_write_checkpoint_files() -> None:
    source = (
        _ROOT
        / "tools"
        / "run_paired_spatial_tail_diagnostic.py"
    ).read_text(encoding="utf-8")
    module = (
        _ROOT
        / "cure_lite"
        / "experiment"
        / "paired_spatial_tail_diagnostic.py"
    ).read_text(encoding="utf-8")

    assert "torch.save(" not in source
    assert "torch.save(" not in module
    assert "load_state_dict(" not in source
    assert "load_state_dict(" not in module
    assert "authorizes_formal_800\": False" in source


def test_loader_verifies_sealed_failure_boundary(tmp_path: Path) -> None:
    config = json.loads(_CONFIG.read_text(encoding="utf-8"))
    root = tmp_path / "sealed"
    receipts = root / "receipts"
    receipts.mkdir(parents=True)
    config_binding = runner._fingerprinted(
        {
            "schema_version": runner.SPATIAL_CONFIG_BINDING_SCHEMA,
            "config": config,
            "config_file_sha256": runner.SPATIAL_CONFIG_FILE_SHA256,
        }
    )
    failure = {
        "schema_version": (
            "cure-lite-paired-spatial-tail-diagnostic-failure-v1"
        ),
        "exception_type": "RuntimeError",
        "message": "synthetic test failure",
        "exact_bounded_replay_verified": False,
    }
    evidence = runner._fingerprinted(failure)
    decision = runner._decision(
        None,
        failure=failure,
        evidence_receipt_fingerprint=evidence["receipt_fingerprint"],
    )
    runner._write_new_json(
        receipts / "config_binding.json",
        config_binding,
    )
    runner._write_new_json(receipts / "failure.json", evidence)
    runner._write_new_json(receipts / "decision.json", decision)
    artifact_files = runner._artifact_hashes(root)
    complete = runner._fingerprinted(
        {
            "schema_version": runner.SPATIAL_RUN_SCHEMA,
            "execution_status": "complete",
            "decision": decision["status"],
            "exact_bounded_replay_verified": False,
            "spatial_tail_report_complete": False,
            "descriptive_only": True,
            "retroactive_bounded_gate_added": False,
            "bounded_decision_changed": False,
            "authorizes_formal_800": False,
            "split": "D_R",
            "config_fingerprint": runner.SPATIAL_CONFIG_FINGERPRINT,
            "config_binding_fingerprint": config_binding[
                "receipt_fingerprint"
            ],
            "micro_population_fingerprint": "1" * 64,
            "schedule_fingerprint": "2" * 64,
            "evidence_kind": "failure",
            "evidence_receipt_fingerprint": evidence[
                "receipt_fingerprint"
            ],
            "decision_fingerprint": decision["receipt_fingerprint"],
            "artifact_files": artifact_files,
            "artifact_file_count": len(artifact_files),
            "formal_training_performed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_evaluation_performed": False,
            "checkpoint_recovery_performed": False,
            "checkpoint_persisted": False,
        },
        field="complete_fingerprint",
    )
    runner._write_new_json(root / "COMPLETE.json", complete)

    published = runner.load_spatial_tail_diagnostic_artifact(root)

    assert (
        published.status
        == "SPATIAL_TAIL_DIAGNOSTIC_EXECUTION_ERROR"
    )
    assert published.exact_bounded_replay_verified is False
    published.verify_unchanged()
