from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest

from cure_lite.metrics import AggregateEvaluation
from tools import verify_cure_lite_v23_pacre_vc_formal_d_v_receipt as verifier


def _metrics(
    recovered: int,
    *,
    miou: float,
    niou: float,
    pixel_fa: float = 1.0e-6,
    budget_violation: bool = False,
) -> AggregateEvaluation:
    true_targets = 147 + recovered
    return AggregateEvaluation(
        pd=true_targets / 170,
        rmr=recovered / 23,
        gross_rmr=recovered / 23,
        net_rmr=recovered / 23,
        retention=1.0,
        reachable_rmr=recovered / 23,
        oracle_upper_bound=1.0,
        overlap_supported_rmr=recovered / 23,
        pixel_fa=pixel_fa,
        raw_background_fa=1.0e-6,
        fp_components_per_mp=1.0,
        miou=miou,
        niou=niou,
        images=120,
        recovered_anchor_misses=recovered,
        net_recovered_anchor_misses=recovered,
        total_anchor_misses=23,
        retained_anchor_covered=147,
        total_anchor_covered=147,
        recovered_reachable_anchor_misses=recovered,
        total_reachable_anchor_misses=23,
        budget_violation=budget_violation,
    )


def _evaluation_result(cure_recovered: int) -> dict[str, object]:
    base_a = _metrics(3, miou=0.60, niou=0.55)
    base_b = base_a
    rejected_base = _metrics(
        3,
        miou=0.60,
        niou=0.55,
        pixel_fa=2.0e-4,
        budget_violation=True,
    )
    cure = _metrics(cure_recovered, miou=0.61, niou=0.56)
    cure_true = 147 + cure_recovered
    checks = dict(
        sorted(
            {
                "CURE_true_targets_strictly_above_best_valid_Base": (
                    cure_true > 150
                ),
                "CURE_recovered_anchor_misses_strictly_above_best_valid_Base": (
                    cure_recovered > 3
                ),
                "CURE_mIoU_not_below_best_valid_Base": True,
                "CURE_nIoU_not_below_best_valid_Base": True,
                "CURE_retention_equal_1": True,
                "CURE_pixel_Fa_le_1e-4": True,
                "CURE_raw_background_Fa_le_1e-4": True,
                "CURE_false_positive_components_per_megapixel_le_100": True,
                "CURE_budget_violation_false": True,
                "D_T_payload_accessed_false": True,
            }.items()
        )
    )
    failed = [name for name, passed in checks.items() if not passed]
    passed = not failed
    ledger_entries = [
        {
            "threshold": threshold,
            "aggregate_evaluation": asdict(
                base_a if threshold == 0.72 else rejected_base
            ),
            "budget_accepted": threshold == 0.72,
        }
        for threshold in verifier.PACRE_VC_FORMAL_BASE_THRESHOLD_GRID
    ]
    ledger_body = {
        "schema_version": (
            "cure-lite-v23-pacre-vc-base-at-b-51-ledger-v1"
        ),
        "method": "Base@B",
        "mode": "base",
        "anchor_threshold": 0.72,
        "candidate_count": 51,
        "entries": ledger_entries,
    }
    return {
        "schema_version": verifier.PACRE_VC_FORMAL_DV_RESULT_SCHEMA,
        "method": verifier.PACRE_VC_FORMAL_METHOD,
        "seed": 42,
        "batch_size": 8,
        "runtime_split": "D_V",
        "D_V_adaptive": True,
        "D_V_payload_accessed": True,
        "D_T_payload_accessed": False,
        "output_contract": {
            "rule": verifier.PACRE_VC_FIXED_OUTPUT_RULE,
            "field_threshold": 0.0,
            "zero_tie_policy": verifier.PACRE_VC_ZERO_TIE_POLICY,
            "hard_union": True,
            "sigmoid_applied": False,
            "PACRE_threshold_search_performed": False,
            "exact_zero_field_pixels": 0,
            "negative_field_pixels": 1,
            "completion_pixels": 1,
        },
        "Base@B_selection": {
            "policy": verifier.PACRE_VC_BASE_AT_B_SELECTION_POLICY,
            "base_threshold_search_performed": True,
            "candidate_threshold_grid": list(
                verifier.PACRE_VC_FORMAL_BASE_THRESHOLD_GRID
            ),
            "candidate_count": 51,
            "candidate_ledger": {
                **ledger_body,
                "ledger_fingerprint": verifier.stable_fingerprint(
                    ledger_body
                ),
            },
            "selected_threshold": 0.72,
            "stage_a_config_sha256": (
                verifier.PACRE_VC_FORMAL_STAGE_A_CONFIG_SHA256
            ),
            "budget": {
                "pixel_fa_budget": 1.0e-4,
                "component_fa_per_mp_budget": 100.0,
                "raw_background_fa_budget": 1.0e-4,
                "minimum_retention": 0.99,
            },
        },
        "operating_points": {
            name: {
                "aggregate_evaluation": asdict(metrics),
                "summary": verifier._summary(metrics),
            }
            for name, metrics in (
                ("Base@A", base_a),
                ("Base@B", base_b),
                ("Base@A+CURE", cure),
            )
        },
        "development_gate": {
            "comparison": "best_valid_Base",
            "valid_base_names": ["Base@A", "Base@B"],
            "requirements": {
                "true_targets": (
                    "strictly_greater_than_best_valid_Base"
                ),
                "recovered_anchor_misses": (
                    "strictly_greater_than_best_valid_Base"
                ),
                "mIoU": "not_below_best_valid_Base",
                "nIoU": "not_below_best_valid_Base",
                "retention": 1.0,
                "maximum_pixel_Fa": 1.0e-4,
                "maximum_raw_background_Fa": 1.0e-4,
                "maximum_false_positive_components_per_megapixel": (
                    100.0
                ),
                "budget_violation": False,
            },
            "best_valid_Base": {
                "true_targets": 150,
                "recovered_anchor_misses": 3,
                "mIoU": 0.60,
                "nIoU": 0.55,
            },
            "CURE_margins": {
                "true_targets": cure_true - 150,
                "recovered_anchor_misses": cure_recovered - 3,
            },
            "checks": checks,
            "failed_checks": failed,
            "gate_passed": passed,
            "status": (
                "PACRE_V23_FORMAL_D_V_GATE_PASS"
                if passed
                else "PACRE_V23_FORMAL_D_V_GATE_FAIL"
            ),
        },
        "bindings": {
            name: "a" * 64
            for name in (
                "base_samples_fingerprint",
                "cure_samples_fingerprint",
                "model_binding_fingerprint",
                "artifact_fingerprint",
                "model_state_fingerprint",
                "comparison_protocol_fingerprint",
                "manifest_fingerprint",
                "base_state_fingerprint",
                "D_V_base_index_fingerprint",
                "D_V_image_fingerprint",
                "D_V_GT_fingerprint",
            )
        },
        "eligible_for_D_T_confirmation": passed,
        "authorizes_D_T": False,
        "final_model_success_established": False,
    }


def test_independent_verifier_recomputes_plus_one_as_pass() -> None:
    verified = verifier._recompute_gate(_evaluation_result(4))
    assert verified["gate_passed"] is True
    assert verified["CURE_margins"] == {
        "true_targets": 1,
        "recovered_anchor_misses": 1,
    }


def test_independent_verifier_recomputes_plus_zero_as_fail() -> None:
    verified = verifier._recompute_gate(_evaluation_result(3))
    assert verified["gate_passed"] is False
    assert set(verified["failed_checks"]) == {
        "CURE_true_targets_strictly_above_best_valid_Base",
        "CURE_recovered_anchor_misses_strictly_above_best_valid_Base",
    }


def _refingerprint_ledger(result: dict[str, object]) -> None:
    ledger = result["Base@B_selection"]["candidate_ledger"]
    body = dict(ledger)
    body.pop("ledger_fingerprint")
    ledger["ledger_fingerprint"] = verifier.stable_fingerprint(body)


def test_independent_verifier_rejects_tampered_candidate() -> None:
    result = deepcopy(_evaluation_result(4))
    row = result["Base@B_selection"]["candidate_ledger"]["entries"][0]
    row["budget_accepted"] = True
    _refingerprint_ledger(result)
    with pytest.raises(ValueError, match="candidate budget flag"):
        verifier._recompute_gate(result)


def test_independent_verifier_rejects_only_37_candidates() -> None:
    result = deepcopy(_evaluation_result(4))
    ledger = result["Base@B_selection"]["candidate_ledger"]
    ledger["entries"] = ledger["entries"][:37]
    ledger["candidate_count"] = 37
    _refingerprint_ledger(result)
    with pytest.raises(ValueError, match="51-point ledger contract"):
        verifier._recompute_gate(result)


def test_independent_verifier_rejects_wrong_persisted_selection() -> None:
    result = deepcopy(_evaluation_result(4))
    result["Base@B_selection"]["selected_threshold"] = 1.0
    with pytest.raises(RuntimeError, match="independent 51-point reselection"):
        verifier._recompute_gate(result)


def test_terminal_verifier_rejects_surviving_staging_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / verifier.RUN_ID
    staging = tmp_path / f".{verifier.RUN_ID}.incomplete"
    output.mkdir()
    staging.mkdir()
    monkeypatch.setattr(verifier, "OUTPUT_PATH", output)
    monkeypatch.setattr(verifier, "STAGING_PATH", staging)
    with pytest.raises(RuntimeError, match="not canonical"):
        verifier.verify_terminal(output)


def test_terminal_verifier_has_no_d_v_payload_loader_or_d_t_route() -> None:
    source = Path(verifier.__file__).read_text(encoding="utf-8")
    assert "load_d_v_cache_bundle" not in source
    assert "ManifestImageDataset" not in source
    assert "load_and_validate_manifest" not in source
    assert "D_V_payload_reopened_by_verifier" in source
    assert '"authorizes_D_T": False' in source
