from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.assess_stage_a import _validate_decision_rule, stage_a_decision_rule_payload
from tools.run_stage_a import _development_mechanism_screen, load_stage_a_config


_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = _ROOT / "protocols" / "IRSTD-1K" / "stage_a_seed42"
_FX_PROTOCOL = _ROOT / "protocols" / "IRSTD-1K" / "stage_a_seed42_fx_v1"


def _metrics(
    *,
    pd: float,
    miou: float = 0.5,
    niou: float = 0.5,
    budget_violation: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        pd=pd,
        miou=miou,
        niou=niou,
        budget_violation=budget_violation,
    )


def test_decision_rule_is_bound_to_the_active_stage_a_config() -> None:
    rule = json.loads(
        (_PROTOCOL / "stage_a_decision_rule.json").read_text(encoding="utf-8")
    )
    config_bytes = (_PROTOCOL / "stage_a_config.json").read_bytes()

    assert rule["schema_version"] == "cure-lite-stage-a-decision-rule-v1"
    assert rule["evaluation_split"] == "D_V"
    assert rule["independent_generalization_claim"] is False
    assert rule["primary_metric"] == "total_object_level_pd"
    assert rule["strict_improvement_required"] is True
    assert rule["positive_signal_requires_all"] == [
        "Pd(U) > Pd(Base@B)",
        "Pd(U) > Pd(F)",
    ]
    assert rule["stage_a_config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    with pytest.raises(ValueError, match="differs from the supported rule"):
        _validate_decision_rule(
            rule,
            dataset="IRSTD-1K",
            seed=42,
            stage_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        )


def test_cli_screen_implements_the_predeclared_strict_pd_rule() -> None:
    positive = SimpleNamespace(
        anchor=_metrics(pd=0.60),
        base_at_budget=_metrics(pd=0.65),
        factual_only=_metrics(pd=0.70),
        factual_exposure_matched=_metrics(pd=0.705),
        uniform_legal=_metrics(pd=0.71, miou=0.49, niou=0.49),
    )
    screen = _development_mechanism_screen(positive)
    assert screen["mechanism_signal"] is True
    assert screen["secondary_iou_non_degradation"] is False

    tied_factual = SimpleNamespace(
        anchor=_metrics(pd=0.60),
        base_at_budget=_metrics(pd=0.65),
        factual_only=_metrics(pd=0.71),
        factual_exposure_matched=_metrics(pd=0.70),
        uniform_legal=_metrics(pd=0.71),
    )
    assert _development_mechanism_screen(tied_factual)["mechanism_signal"] is False

    tied_exposure = SimpleNamespace(
        anchor=_metrics(pd=0.60),
        base_at_budget=_metrics(pd=0.65),
        factual_only=_metrics(pd=0.69),
        factual_exposure_matched=_metrics(pd=0.71),
        uniform_legal=_metrics(pd=0.71),
    )
    assert _development_mechanism_screen(tied_exposure)["mechanism_signal"] is False

    violating_u = SimpleNamespace(
        anchor=_metrics(pd=0.60),
        base_at_budget=_metrics(pd=0.65),
        factual_only=_metrics(pd=0.70),
        factual_exposure_matched=_metrics(pd=0.705),
        uniform_legal=_metrics(pd=0.71, budget_violation=True),
    )
    assert _development_mechanism_screen(violating_u)["mechanism_signal"] is False

    violating_exposure = SimpleNamespace(
        anchor=_metrics(pd=0.60),
        base_at_budget=_metrics(pd=0.65),
        factual_only=_metrics(pd=0.69),
        factual_exposure_matched=_metrics(pd=0.70, budget_violation=True),
        uniform_legal=_metrics(pd=0.71),
    )
    assert (
        _development_mechanism_screen(violating_exposure)["mechanism_signal"]
        is False
    )


def test_next_decision_data_requires_u_to_beat_exposure_matched_control() -> None:
    payload = stage_a_decision_rule_payload(
        dataset="IRSTD-1K",
        seed=42,
        stage_config_sha256="a" * 64,
    )
    assert payload["schema_version"] == "cure-lite-stage-a-decision-rule-v2"
    assert payload["method_order"] == ["A", "Base@B", "F", "F×", "U"]
    assert payload["comparators"] == ["Base@B", "F", "F×"]
    assert payload["positive_signal_requires_all"][-1] == "Pd(U) > Pd(F×)"


def test_five_way_protocol_config_and_decision_rule_are_bound() -> None:
    config_path = _FX_PROTOCOL / "stage_a_config.json"
    config = load_stage_a_config(config_path)
    assert config.canonical_payload()["method_contract"]["method_order"] == [
        "A",
        "Base@B",
        "F",
        "F×",
        "U",
    ]
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    rule = json.loads(
        (_FX_PROTOCOL / "stage_a_decision_rule.json").read_text(encoding="utf-8")
    )
    assert rule == stage_a_decision_rule_payload(
        dataset="IRSTD-1K",
        seed=42,
        stage_config_sha256=config_sha256,
    )
