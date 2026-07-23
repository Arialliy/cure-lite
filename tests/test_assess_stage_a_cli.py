from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import assess_stage_a as cli


_ABORTED_FX_V1_SHA256 = {
    "protocol_freeze.json": (
        "998c1e51c1e69c6ea9ed0bd0635c8e90"
        "f6956b47d455a988fa27f74ee5f9e912"
    ),
    "stage_a_config.json": (
        "b904de307f209fec148cd0c69da49459"
        "cf03daa1ecfa73467ff1ec3b795447d1"
    ),
    "stage_a_decision_rule.json": (
        "3f9ed6d8b6a33541dfdc4399f4d8a752"
        "5139d6fdc2ded796977da9154b0dd4e0"
    ),
}

_FAILED_FX_V2_SHA256 = {
    "failure_record.json": (
        "65627439f4e5d475d3d14e02c4c02cba"
        "3a11874f9a7c59d734e22520eb03362d"
    ),
    "protocol_freeze.json": (
        "e4f61c0d3706f438e4dc44b50ac8041b"
        "e3033db13a962289dd6853812bfd318e"
    ),
    "stage_a_config.json": (
        "b904de307f209fec148cd0c69da49459"
        "cf03daa1ecfa73467ff1ec3b795447d1"
    ),
    "stage_a_decision_rule.json": (
        "3f9ed6d8b6a33541dfdc4399f4d8a752"
        "5139d6fdc2ded796977da9154b0dd4e0"
    ),
}


def _metrics(
    pd: float,
    *,
    miou: float = 0.5,
    niou: float = 0.5,
    budget_violation: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        pd=pd,
        miou=miou,
        niou=niou,
        pixel_fa=0.0,
        fp_components_per_mp=0.0,
        raw_background_fa=0.0,
        retention=1.0,
        budget_violation=budget_violation,
    )


def test_parser_has_no_d_t_or_detector_specific_option() -> None:
    options = {
        option
        for action in cli.build_parser()._actions
        for option in action.option_strings
    }
    assert "--stage-run" in options
    assert "--reference-base-run" in options
    assert "--decision-rule" in options
    assert "--protocol-freeze" in options
    assert not any("d-t" in option or "d_t" in option for option in options)
    assert not any("mshnet" in option.lower() for option in options)


def test_assessment_payload_uses_strict_pd_rule_and_records_thresholds() -> None:
    results = SimpleNamespace(
        anchor=_metrics(0.60),
        base_at_budget=_metrics(0.65),
        factual_only=_metrics(0.70),
        factual_exposure_matched=_metrics(0.705),
        uniform_legal=_metrics(0.71, miou=0.49, niou=0.49),
    )
    completed = SimpleNamespace(
        results=results,
        config=SimpleNamespace(training=SimpleNamespace(global_seed=42)),
        complete_fingerprint="1" * 64,
        support_summary=SimpleNamespace(
            canonical_payload=lambda: {"source_images": 160}
        ),
        efficiency=SimpleNamespace(
            canonical_payload=lambda: {
                "deployed_method": "U",
                "efficiency_is_scientific_gate_metric": False,
            }
        ),
        anchor=SimpleNamespace(selected_threshold=0.5),
        calibration=SimpleNamespace(
            base_at_budget=SimpleNamespace(
                protocol=SimpleNamespace(selected_threshold=0.4)
            ),
            factual_only=SimpleNamespace(
                protocol=SimpleNamespace(selected_threshold=0.8)
            ),
            factual_exposure_matched=SimpleNamespace(
                protocol=SimpleNamespace(selected_threshold=0.75)
            ),
            uniform_legal=SimpleNamespace(
                protocol=SimpleNamespace(selected_threshold=0.7)
            ),
        ),
    )
    manifest = SimpleNamespace(dataset="IRSTD-1K", fingerprint="2" * 64)
    payload = cli._assessment_payload(
        completed,
        manifest,
        decision_rule_sha256="3" * 64,
        protocol_freeze_sha256="4" * 64,
        stage_config_sha256="5" * 64,
    )

    assert payload["verified_full_replay"] is True
    assert payload["unused_split"] == "D_T"
    assert payload["efficiency"]["deployed_method"] == "U"
    assert payload["efficiency"]["efficiency_is_scientific_gate_metric"] is False
    assert payload["development_mechanism_screen"]["mechanism_signal"] is True
    assert payload["conclusion"] == "positive_pd_signal_with_secondary_iou_tradeoff"
    assert payload["selected_thresholds"] == {
        "A": 0.5,
        "Base@B": 0.4,
        "F": 0.8,
        "F×": 0.75,
        "U": 0.7,
    }
    assert set(payload["methods"]["U"]) == {
        "pd",
        "miou",
        "niou",
        "pixel_fa",
        "fp_components_per_mp",
        "raw_background_fa",
        "retention",
        "budget_violation",
    }


def test_current_protocol_freeze_binds_frozen_inputs() -> None:
    protocol = cli._ROOT / "protocols" / "IRSTD-1K" / "stage_a_seed42"
    freeze = json.loads((protocol / "protocol_freeze.json").read_text("utf-8"))
    with pytest.raises(ValueError, match="unsupported.*schema"):
        cli.validate_protocol_freeze(
            freeze,
            manifest_path=protocol / "manifest.json",
            stage_config_path=protocol / "stage_a_config.json",
            decision_rule_path=protocol / "stage_a_decision_rule.json",
            d_r_index_path=(
                cli._ROOT
                / "runs/irstd1k_stage_a_seed42/reference_base_cache_v1/D_R/index.json"
            ),
            d_v_index_path=(
                cli._ROOT
                / "runs/irstd1k_stage_a_seed42/reference_base_cache_v1/D_V/index.json"
            ),
            stage_run_path=(
                cli._ROOT / "runs/irstd1k_stage_a_seed42/cure_lite_stage_a_v1"
            ),
        )


def test_v2_protocol_freeze_binds_run_and_assessment_tools() -> None:
    protocol = cli._ROOT / "protocols" / "IRSTD-1K" / "stage_a_seed42"
    freeze = json.loads((protocol / "protocol_freeze.json").read_text("utf-8"))
    freeze.update(
        {
            "schema_version": cli.FREEZE_SCHEMA,
            "method_source_tree_digest": cli._source_tree_digest(),
            "run_tool_sha256": cli._sha256(cli._ROOT / "tools" / "run_stage_a.py"),
            "assessment_tool_sha256": cli._sha256(Path(cli.__file__)),
        }
    )
    arguments = {
        "manifest_path": protocol / "manifest.json",
        "stage_config_path": protocol / "stage_a_config.json",
        "decision_rule_path": protocol / "stage_a_decision_rule.json",
        "d_r_index_path": (
            cli._ROOT
            / "runs/irstd1k_stage_a_seed42/reference_base_cache_v1/D_R/index.json"
        ),
        "d_v_index_path": (
            cli._ROOT
            / "runs/irstd1k_stage_a_seed42/reference_base_cache_v1/D_V/index.json"
        ),
        "stage_run_path": (
            cli._ROOT / "runs/irstd1k_stage_a_seed42/cure_lite_stage_a_v1"
        ),
    }
    cli.validate_protocol_freeze(freeze, **arguments)
    freeze["run_tool_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="run_tool_sha256"):
        cli.validate_protocol_freeze(freeze, **arguments)


def test_completed_v01_protocol_freeze_rejects_current_v02_source() -> None:
    manifest_protocol = (
        cli._ROOT / "protocols" / "IRSTD-1K" / "stage_a_seed42"
    )
    stage_protocol = (
        cli._ROOT / "protocols" / "IRSTD-1K" / "stage_a_seed42_fx_v3"
    )
    freeze = json.loads(
        (stage_protocol / "protocol_freeze.json").read_text("utf-8")
    )
    cache = (
        cli._ROOT
        / "runs/irstd1k_stage_a_seed42/reference_base_cache_fx_v2"
    )
    stage_run = (
        cli._ROOT / "runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3"
    )
    with pytest.raises(RuntimeError, match="method source differs"):
        cli.validate_protocol_freeze(
            freeze,
            manifest_path=manifest_protocol / "manifest.json",
            stage_config_path=stage_protocol / "stage_a_config.json",
            decision_rule_path=stage_protocol / "stage_a_decision_rule.json",
            d_r_index_path=cache / "D_R" / "index.json",
            d_v_index_path=cache / "D_V" / "index.json",
            stage_run_path=stage_run,
        )


def test_seed43_v01_freeze_rejects_v02_source_and_remains_path_isolated() -> None:
    manifest_protocol = (
        cli._ROOT / "protocols" / "IRSTD-1K" / "stage_a_seed42"
    )
    seed42_protocol = (
        cli._ROOT / "protocols" / "IRSTD-1K" / "stage_a_seed42_fx_v3"
    )
    seed43_protocol = (
        cli._ROOT / "protocols" / "IRSTD-1K" / "stage_a_seed42_fx_v3_s43"
    )
    seed42_freeze = json.loads(
        (seed42_protocol / "protocol_freeze.json").read_text("utf-8")
    )
    seed43_freeze = json.loads(
        (seed43_protocol / "protocol_freeze.json").read_text("utf-8")
    )
    cache = (
        cli._ROOT
        / "runs/irstd1k_stage_a_seed42/reference_base_cache_fx_v2"
    )
    seed43_stage_run = (
        cli._ROOT
        / "runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3_s43"
    )
    with pytest.raises(RuntimeError, match="method source differs"):
        cli.validate_protocol_freeze(
            seed43_freeze,
            manifest_path=manifest_protocol / "manifest.json",
            stage_config_path=seed43_protocol / "stage_a_config.json",
            decision_rule_path=seed43_protocol / "stage_a_decision_rule.json",
            d_r_index_path=cache / "D_R" / "index.json",
            d_v_index_path=cache / "D_V" / "index.json",
            stage_run_path=seed43_stage_run,
        )
    assert seed42_freeze["stage_a_output"] != seed43_freeze["stage_a_output"]
    assert (
        seed42_freeze["assessment_output"]
        != seed43_freeze["assessment_output"]
    )
    assert (
        seed42_freeze["stage_a_service_invocation_id"]
        != seed43_freeze["stage_a_service_invocation_id"]
    )


def test_aborted_fx_v1_protocol_is_immutable_history() -> None:
    protocol = (
        cli._ROOT / "protocols" / "IRSTD-1K" / "stage_a_seed42_fx_v1"
    )
    assert {
        name: hashlib.sha256((protocol / name).read_bytes()).hexdigest()
        for name in _ABORTED_FX_V1_SHA256
    } == _ABORTED_FX_V1_SHA256


def test_failed_fx_v2_protocol_is_immutable_history() -> None:
    protocol = (
        cli._ROOT / "protocols" / "IRSTD-1K" / "stage_a_seed42_fx_v2"
    )
    assert {
        name: hashlib.sha256((protocol / name).read_bytes()).hexdigest()
        for name in _FAILED_FX_V2_SHA256
    } == _FAILED_FX_V2_SHA256
