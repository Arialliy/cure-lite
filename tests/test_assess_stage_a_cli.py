from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tools import assess_stage_a as cli


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
        gross_rmr=0.0,
        net_rmr=0.0,
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
    assert "--decision-rule" in options
    assert "--protocol-freeze" in options
    assert not any("d-t" in option or "d_t" in option for option in options)
    assert not any("mshnet" in option.lower() for option in options)


def test_assessment_payload_uses_strict_pd_rule_and_records_thresholds() -> None:
    results = SimpleNamespace(
        anchor=_metrics(0.60),
        base_at_budget=_metrics(0.65),
        factual_only=_metrics(0.70),
        uniform_legal=_metrics(0.71, miou=0.49, niou=0.49),
    )
    completed = SimpleNamespace(
        results=results,
        config=SimpleNamespace(training=SimpleNamespace(global_seed=42)),
        complete_fingerprint="1" * 64,
        support_summary=SimpleNamespace(
            canonical_payload=lambda: {"source_images": 160}
        ),
        anchor=SimpleNamespace(selected_threshold=0.5),
        calibration=SimpleNamespace(
            base_at_budget=SimpleNamespace(
                protocol=SimpleNamespace(selected_threshold=0.4)
            ),
            factual_only=SimpleNamespace(
                protocol=SimpleNamespace(selected_threshold=0.8)
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
    assert payload["development_mechanism_screen"]["mechanism_signal"] is True
    assert payload["conclusion"] == "positive_pd_signal_with_secondary_iou_tradeoff"
    assert payload["selected_thresholds"] == {
        "A": 0.5,
        "Base@B": 0.4,
        "F": 0.8,
        "U": 0.7,
    }


def test_current_protocol_freeze_binds_frozen_inputs() -> None:
    protocol = cli._ROOT / "protocols" / "IRSTD-1K" / "stage_a_seed42"
    freeze = json.loads((protocol / "protocol_freeze.json").read_text("utf-8"))
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
