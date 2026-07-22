from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from tools.run_stage_a import _development_mechanism_screen


_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = _ROOT / "protocols" / "IRSTD-1K" / "stage_a_seed42"


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


def test_cli_screen_implements_the_predeclared_strict_pd_rule() -> None:
    positive = SimpleNamespace(
        anchor=_metrics(pd=0.60),
        base_at_budget=_metrics(pd=0.65),
        factual_only=_metrics(pd=0.70),
        uniform_legal=_metrics(pd=0.71, miou=0.49, niou=0.49),
    )
    screen = _development_mechanism_screen(positive)
    assert screen["mechanism_signal"] is True
    assert screen["secondary_iou_non_degradation"] is False

    tied_factual = SimpleNamespace(
        anchor=_metrics(pd=0.60),
        base_at_budget=_metrics(pd=0.65),
        factual_only=_metrics(pd=0.71),
        uniform_legal=_metrics(pd=0.71),
    )
    assert _development_mechanism_screen(tied_factual)["mechanism_signal"] is False

    violating_u = SimpleNamespace(
        anchor=_metrics(pd=0.60),
        base_at_budget=_metrics(pd=0.65),
        factual_only=_metrics(pd=0.70),
        uniform_legal=_metrics(pd=0.71, budget_violation=True),
    )
    assert _development_mechanism_screen(violating_u)["mechanism_signal"] is False
