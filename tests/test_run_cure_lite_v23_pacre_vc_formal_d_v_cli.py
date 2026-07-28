from __future__ import annotations

from dataclasses import asdict
import ctypes
import json
import os
from pathlib import Path

import pytest

from cure_lite.metrics import AggregateEvaluation
from cure_lite_v23.protocol import read_strict_json
from tools import run_cure_lite_v23_pacre_vc_formal_d_v as runner
from tools import verify_cure_lite_v23_pacre_vc_formal_d_v_receipt as verifier


_DIGEST = "a" * 64


class _FakeArtifact:
    artifact_json = "{}"

    def verify_unchanged(self) -> None:
        return None


class _FakeVerifiedTerminal:
    def __init__(self) -> None:
        self.artifact = _FakeArtifact()

    def verify_unchanged(self) -> None:
        self.artifact.verify_unchanged()


def _fake_formal() -> runner._FormalTerminal:
    return runner._FormalTerminal(
        verified_terminal=(  # type: ignore[arg-type]
            _FakeVerifiedTerminal()
        ),
        payload={
            "complete_fingerprint": _DIGEST,
            "artifact_fingerprint": "b" * 64,
            "source_closure_fingerprint": "c" * 64,
        },
    )


def _metrics(
    recovered: int,
    *,
    miou: float,
    niou: float,
    pixel_fa: float = 1.0e-6,
    raw_background_fa: float = 1.0e-6,
    fp_components_per_mp: float = 1.0,
    retained: int = 147,
    budget_violation: bool = False,
) -> AggregateEvaluation:
    true_targets = retained + recovered
    net_recovered = true_targets - 147
    return AggregateEvaluation(
        pd=true_targets / 170,
        rmr=recovered / 23,
        gross_rmr=recovered / 23,
        net_rmr=net_recovered / 23,
        retention=retained / 147,
        reachable_rmr=recovered / 23,
        oracle_upper_bound=1.0,
        overlap_supported_rmr=recovered / 23,
        pixel_fa=pixel_fa,
        raw_background_fa=raw_background_fa,
        fp_components_per_mp=fp_components_per_mp,
        miou=miou,
        niou=niou,
        images=120,
        recovered_anchor_misses=recovered,
        net_recovered_anchor_misses=net_recovered,
        total_anchor_misses=23,
        retained_anchor_covered=retained,
        total_anchor_covered=147,
        recovered_reachable_anchor_misses=recovered,
        total_reachable_anchor_misses=23,
        budget_violation=budget_violation,
    )


def _result(
    *,
    cure_recovered: int,
    cure_miou: float = 0.61,
    cure_niou: float = 0.56,
    cure_pixel_fa: float = 1.0e-6,
) -> dict[str, object]:
    base_a = _metrics(3, miou=0.60, niou=0.55)
    base_b = base_a
    rejected_base = _metrics(
        3,
        miou=0.60,
        niou=0.55,
        pixel_fa=2.0e-4,
        budget_violation=True,
    )
    cure = _metrics(
        cure_recovered,
        miou=cure_miou,
        niou=cure_niou,
        pixel_fa=cure_pixel_fa,
    )
    operating = {
        name: {
            "aggregate_evaluation": asdict(metrics),
            "summary": verifier._summary(metrics),
        }
        for name, metrics in (
            ("Base@A", base_a),
            ("Base@B", base_b),
            ("Base@A+CURE", cure),
        )
    }
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
    ledger = {
        **ledger_body,
        "ledger_fingerprint": verifier.stable_fingerprint(ledger_body),
    }
    cure_true = cure.retained_anchor_covered + cure.recovered_anchor_misses
    margins = {
        "true_targets": cure_true - 150,
        "recovered_anchor_misses": cure.recovered_anchor_misses - 3,
    }
    checks = dict(
        sorted(
            {
                "CURE_true_targets_strictly_above_best_valid_Base": (
                    cure_true > 150
                ),
                "CURE_recovered_anchor_misses_strictly_above_best_valid_Base": (
                    cure.recovered_anchor_misses > 3
                ),
                "CURE_mIoU_not_below_best_valid_Base": (
                    cure.miou >= 0.60
                ),
                "CURE_nIoU_not_below_best_valid_Base": (
                    cure.niou >= 0.55
                ),
                "CURE_retention_equal_1": cure.retention == 1.0,
                "CURE_pixel_Fa_le_1e-4": cure.pixel_fa <= 1.0e-4,
                "CURE_raw_background_Fa_le_1e-4": (
                    cure.raw_background_fa <= 1.0e-4
                ),
                "CURE_false_positive_components_per_megapixel_le_100": (
                    cure.fp_components_per_mp <= 100.0
                ),
                "CURE_budget_violation_false": (
                    cure.budget_violation is False
                ),
                "D_T_payload_accessed_false": True,
            }.items()
        )
    )
    failed = [name for name, passed in checks.items() if not passed]
    passed = not failed
    gate_status = (
        "PACRE_V23_FORMAL_D_V_GATE_PASS"
        if passed
        else "PACRE_V23_FORMAL_D_V_GATE_FAIL"
    )
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
            "exact_zero_field_pixels": 7,
            "negative_field_pixels": 11,
            "completion_pixels": 9,
        },
        "Base@B_selection": {
            "policy": verifier.PACRE_VC_BASE_AT_B_SELECTION_POLICY,
            "base_threshold_search_performed": True,
            "candidate_threshold_grid": list(
                verifier.PACRE_VC_FORMAL_BASE_THRESHOLD_GRID
            ),
            "candidate_count": 51,
            "candidate_ledger": ledger,
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
        "operating_points": operating,
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
            "CURE_margins": margins,
            "checks": checks,
            "failed_checks": failed,
            "gate_passed": passed,
            "status": gate_status,
        },
        "bindings": {
            name: _DIGEST
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


def test_relative_gate_accepts_a_strict_plus_one_without_plus_two() -> None:
    recomputed = verifier._recompute_gate(_result(cure_recovered=4))
    assert recomputed["gate_passed"] is True
    assert recomputed["CURE_margins"] == {
        "true_targets": 1,
        "recovered_anchor_misses": 1,
    }
    plan = runner._fixed_plan()
    assert plan["development_gate"]["minimum_fixed_uplift_margin"] is None
    assert plan["development_gate"]["plus_one_is_sufficient"] is True


def test_relative_gate_rejects_no_strict_improvement() -> None:
    recomputed = verifier._recompute_gate(_result(cure_recovered=3))
    assert recomputed["gate_passed"] is False
    assert set(recomputed["failed_checks"]) == {
        "CURE_true_targets_strictly_above_best_valid_Base",
        "CURE_recovered_anchor_misses_strictly_above_best_valid_Base",
    }


def test_relative_gate_preserves_safety_constraints() -> None:
    recomputed = verifier._recompute_gate(
        _result(cure_recovered=4, cure_pixel_fa=2.0e-4)
    )
    assert recomputed["gate_passed"] is False
    assert "CURE_pixel_Fa_le_1e-4" in recomputed["failed_checks"]


def test_recompute_rejects_changed_frozen_budget_and_d_t_access() -> None:
    changed_budget = _result(cure_recovered=4)
    changed_budget["Base@B_selection"]["budget"][
        "pixel_fa_budget"
    ] = 2.0e-4
    with pytest.raises(ValueError, match="fixed evaluation contract"):
        verifier._recompute_gate(changed_budget)

    changed_split = _result(cure_recovered=4)
    changed_split["D_T_payload_accessed"] = True
    with pytest.raises(ValueError, match="fixed evaluation contract"):
        verifier._recompute_gate(changed_split)


def test_runner_contract_requires_batch_8_and_all_51_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SyntheticResult:
        gate_passed = True

        def __init__(self, payload: dict[str, object]) -> None:
            self.payload = payload

        def verify_unchanged(self) -> None:
            return None

        def canonical_payload(self) -> dict[str, object]:
            return self.payload

    monkeypatch.setattr(
        runner,
        "PACREVCFormalDVEvaluationResult",
        _SyntheticResult,
    )
    valid = _result(cure_recovered=4)
    assert runner._result_contract(_SyntheticResult(valid)) == valid

    wrong_batch = json.loads(json.dumps(valid))
    wrong_batch["batch_size"] = 4
    with pytest.raises(RuntimeError, match="result contract"):
        runner._result_contract(_SyntheticResult(wrong_batch))

    incomplete = json.loads(json.dumps(valid))
    ledger = incomplete["Base@B_selection"]["candidate_ledger"]
    ledger["entries"] = ledger["entries"][:37]
    ledger["candidate_count"] = 37
    body = dict(ledger)
    body.pop("ledger_fingerprint")
    ledger["ledger_fingerprint"] = verifier.stable_fingerprint(body)
    with pytest.raises(RuntimeError, match="result contract"):
        runner._result_contract(_SyntheticResult(incomplete))


def test_validate_create_only_never_calls_d_v_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "_require_atomic_rename_noreplace", lambda: None)
    monkeypatch.setattr(runner, "_ensure_outputs_absent", lambda: None)
    monkeypatch.setattr(runner, "_verify_fixed_metadata", lambda: None)
    monkeypatch.setattr(
        runner,
        "_require_runtime",
        lambda: {"runtime_environment_fingerprint": _DIGEST},
    )
    monkeypatch.setattr(runner, "_load_formal_terminal", _fake_formal)
    monkeypatch.setattr(
        runner,
        "load_d_v_cache_bundle",
        lambda *args, **kwargs: pytest.fail("validate opened D_V"),
    )
    monkeypatch.setattr(
        runner,
        "load_frozen_comparison_protocol",
        lambda *args, **kwargs: pytest.fail("validate loaded D_V protocol"),
    )
    result = runner.validate_create_only()
    assert result["D_V_payload_accessed"] is False
    assert result["D_T_payload_accessed"] is False
    assert result["evaluation_performed"] is False


def test_run_claims_before_d_v_and_failure_is_not_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / runner.RUN_ID
    staging = tmp_path / f".{runner.RUN_ID}.incomplete"
    monkeypatch.setattr(runner, "OUTPUT_PATH", output)
    monkeypatch.setattr(runner, "STAGING_PATH", staging)
    monkeypatch.setattr(runner, "_require_atomic_rename_noreplace", lambda: None)
    monkeypatch.setattr(runner, "_verify_fixed_metadata", lambda: None)
    monkeypatch.setattr(
        runner,
        "_require_runtime",
        lambda: {"runtime_environment_fingerprint": _DIGEST},
    )
    monkeypatch.setattr(runner, "_load_formal_terminal", _fake_formal)

    def fail_after_claim() -> None:
        claim = read_strict_json(staging / runner.CLAIM_FILE)
        assert claim["status"] == "claimed_before_D_V_materialization"
        assert claim["D_V_payload_accessed"] is False
        assert claim["D_T_payload_accessed"] is False
        raise RuntimeError("synthetic pre-materialization stop")

    monkeypatch.setattr(runner, "_load_fixed_d_v_inputs", fail_after_claim)
    with pytest.raises(RuntimeError, match="synthetic"):
        runner.run_once()
    assert not output.exists()
    assert staging.is_dir()
    failure = read_strict_json(staging / runner.FAILURE_FILE)
    assert failure["output_reusable"] is False
    assert failure["retry_allowed"] is False
    # Once the strict loader is entered, failure reporting is conservative:
    # a partial loader failure may already have opened a D_V asset.
    assert failure["D_V_payload_accessed"] is True
    assert failure["D_T_payload_accessed"] is False


def test_cli_exposes_no_threshold_split_model_or_output_override() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--run-once", "--threshold", "0.1"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--run-once", "--D_T"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--run-once", "--output", "/tmp/elsewhere"])
    assert runner.parse_args(["--validate-create-only"]).validate_create_only
    assert runner.parse_args(["--run-once"]).run_once


def test_atomic_rename_noreplace_uses_linux_argument_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class _RenameAt2:
        argtypes: object = None
        restype: object = None

        def __call__(self, *args: object) -> int:
            calls.append(args)
            return 0

    renameat2 = _RenameAt2()

    class _LibC:
        pass

    libc = _LibC()
    libc.renameat2 = renameat2  # type: ignore[attr-defined]
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: libc)
    source = tmp_path / ".attempt.incomplete"
    target = tmp_path / "published"
    runner._atomic_rename_noreplace(source, target)
    assert calls == [
        (
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(target),
            1,
        )
    ]


def test_formal_terminal_can_be_reverified_repeatedly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _fake_formal()
    calls = 0

    def reload_same() -> runner._FormalTerminal:
        nonlocal calls
        calls += 1
        return _fake_formal()

    monkeypatch.setattr(runner, "_load_formal_terminal", reload_same)
    original.verify_unchanged()
    original.verify_unchanged()
    assert calls == 2


def test_verifier_does_not_import_the_d_v_runner() -> None:
    source = Path(verifier.__file__).read_text(encoding="utf-8")
    assert (
        "from tools.run_cure_lite_v23_pacre_vc_formal_d_v"
        not in source
    )
    assert "load_d_v_cache_bundle" not in source
    assert "ManifestImageDataset" not in source
    assert "D_V_payload_reopened_by_verifier" in source
