from __future__ import annotations

import pytest

from cure_lite.experiment import coverage_state_paet_formal_decision as module
from cure_lite.experiment.coverage_state_paet_formal_decision import (
    assess_paet_formal_d_v_result,
)
from cure_lite.experiment.coverage_state_paet_formal_evaluation import (
    PAETFormalDVEvaluationResult,
)


def test_only_sealed_evaluation_result_can_authorize_dt() -> None:
    with pytest.raises(TypeError, match="PAETFormalDVEvaluationResult"):
        assess_paet_formal_d_v_result(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        PAETFormalDVEvaluationResult()  # type: ignore[call-arg]


def test_metrics_helper_is_private_and_never_authorizes_dt() -> None:
    assert "assess_paet_formal_d_v_gate" not in module.__all__
    assert hasattr(module, "_assess_paet_formal_d_v_metrics")


def test_decision_module_exports_only_provenance_checked_gate() -> None:
    assert module.__all__[-1] == "assess_paet_formal_d_v_result"
