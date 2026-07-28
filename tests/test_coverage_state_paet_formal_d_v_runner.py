from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.experiment import (
    coverage_state_paet_formal_d_v_runner as module,
)


def _fingerprinted(
    payload: dict[str, object],
    *,
    field: str,
) -> dict[str, object]:
    return {**payload, field: stable_fingerprint(payload)}


class _FakeAttempt:
    def __init__(self) -> None:
        self.complete_fingerprint = "a" * 64
        self.formal_training_result_fingerprint = "b" * 64
        self.authorization_fingerprint = "c" * 64
        self.structural_result_fingerprint = "d" * 64
        self.source_receipt_fingerprint = "e" * 64
        self.source_closure_manifest_sha256 = "f" * 64
        self.source_closure_archive_sha256 = "0" * 64
        self.source_closure_content_fingerprint = "1" * 64
        self.source_closure_file_count = 241
        self.post_formal_structural_retention_passed = True
        self.artifact = SimpleNamespace()
        self.verification_count = 0

    def verify_unchanged(self) -> None:
        self.verification_count += 1


def _evidence() -> module._EvaluationEvidence:
    binding = {"schema_version": "binding", "value": 1}
    samples = {"schema_version": "samples", "value": 2}
    evaluation = {"schema_version": "evaluation", "value": 3}
    binding_fp = stable_fingerprint(binding)
    sample_fp = stable_fingerprint(samples)
    evaluation_fp = stable_fingerprint(evaluation)
    decision = _fingerprinted(
        {
            "schema_version": "decision",
            "seed": 42,
            "runtime_split": "D_V",
            "D_T_accessed": False,
            "gate_passed": True,
            "authorizes_D_T": True,
            "bindings": {
                "artifact_binding_fingerprint": binding_fp,
                "evaluation_result_fingerprint": evaluation_fp,
            },
        },
        field="decision_fingerprint",
    )
    return module._EvaluationEvidence(
        artifact_binding_payload=binding,
        artifact_binding_fingerprint=binding_fp,
        sample_payload=samples,
        sample_fingerprint=sample_fp,
        evaluation_payload=evaluation,
        evaluation_result_fingerprint=evaluation_fp,
        decision=decision,
    )


def _patch_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Path, Path]:
    output = tmp_path / module.PAET_FORMAL_DV_RUN_ID
    staging = tmp_path / f".{module.PAET_FORMAL_DV_RUN_ID}.incomplete"
    monkeypatch.setattr(module, "PAET_FORMAL_DV_OUTPUT_PATH", output)
    monkeypatch.setattr(module, "PAET_FORMAL_DV_STAGING_PATH", staging)
    monkeypatch.setattr(
        module,
        "_require_atomic_rename_noreplace",
        lambda: None,
    )
    monkeypatch.setattr(
        module,
        "_atomic_rename_noreplace",
        lambda source, target: os.rename(source, target),
    )
    return output, staging


def _patch_fake_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> _FakeAttempt:
    attempt = _FakeAttempt()
    monkeypatch.setattr(
        module,
        "LoadedCoverageStatePAETFormalAttempt",
        _FakeAttempt,
    )
    monkeypatch.setattr(
        module,
        "load_coverage_state_paet_formal_attempt",
        lambda: attempt,
    )
    monkeypatch.setattr(
        module,
        "_reverify_formal_attempt",
        lambda received: received.verify_unchanged(),
    )
    return attempt


def test_create_only_is_deterministic_and_does_not_open_d_v(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output, staging = _patch_paths(monkeypatch, tmp_path)

    def forbidden() -> None:
        raise AssertionError("create-only opened D_V")

    monkeypatch.setattr(module, "_load_fixed_d_v_inputs", forbidden)
    left = module.validate_paet_formal_d_v_create_only()
    right = module.validate_paet_formal_d_v_create_only()
    assert left == right
    assert left["D_V_accessed"] is False
    assert left["D_T_accessed"] is False
    assert left["output_created"] is False
    assert not output.exists()
    assert not staging.exists()


def test_simulated_success_claims_before_evaluation_and_publishes_only_three(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output, staging = _patch_paths(monkeypatch, tmp_path)
    attempt = _patch_fake_attempt(monkeypatch)
    evidence = _evidence()

    def execute(received: _FakeAttempt) -> module._EvaluationEvidence:
        assert received is attempt
        assert staging.is_dir()
        assert {path.name for path in staging.iterdir()} == {
            module._CLAIM_NAME
        }
        claim = json.loads(
            (staging / module._CLAIM_NAME).read_text(encoding="utf-8")
        )
        assert claim["D_V_accessed"] is False
        assert claim["D_T_accessed"] is False
        return evidence

    monkeypatch.setattr(module, "_execute_fixed_evaluation", execute)
    summary = module.run_paet_formal_d_v_once()
    assert not staging.exists()
    assert output.is_dir()
    assert {path.name for path in output.iterdir()} == {
        "receipt.json",
        "decision.json",
        "COMPLETE.json",
    }
    assert summary["status"] == "complete"
    assert summary["gate_passed"] is True
    assert summary["authorizes_D_T"] is True
    assert summary["D_T_accessed"] is False

    receipt = json.loads(
        (output / "receipt.json").read_text(encoding="utf-8")
    )
    complete = json.loads(
        (output / "COMPLETE.json").read_text(encoding="utf-8")
    )
    assert receipt["formal_attempt"]["complete_fingerprint"] == "a" * 64
    assert (
        receipt["formal_attempt"][
            "source_closure_content_fingerprint"
        ]
        == "1" * 64
    )
    assert (
        receipt["evaluation_result_fingerprint"]
        == evidence.evaluation_result_fingerprint
    )
    assert receipt["model_training_performed"] is False
    assert receipt["model_state_update_performed"] is False
    assert receipt["PAET_threshold_search_performed"] is False
    assert receipt["D_T_accessed"] is False
    assert complete["D_T_accessed"] is False
    assert attempt.verification_count >= 4


def test_failed_attempt_stays_nonreusable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output, staging = _patch_paths(monkeypatch, tmp_path)
    _patch_fake_attempt(monkeypatch)

    def fail(_: _FakeAttempt) -> module._EvaluationEvidence:
        assert staging.is_dir()
        raise RuntimeError("synthetic evaluation failure")

    monkeypatch.setattr(module, "_execute_fixed_evaluation", fail)
    with pytest.raises(RuntimeError, match="synthetic evaluation failure"):
        module.run_paet_formal_d_v_once()
    assert not output.exists()
    assert staging.is_dir()
    assert (staging / module._CLAIM_NAME).is_file()
    with pytest.raises(FileExistsError, match="not reusable"):
        module.run_paet_formal_d_v_once()


def test_published_tampering_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output, _ = _patch_paths(monkeypatch, tmp_path)
    _patch_fake_attempt(monkeypatch)
    monkeypatch.setattr(
        module,
        "_execute_fixed_evaluation",
        lambda _: _evidence(),
    )
    module.run_paet_formal_d_v_once()
    decision = json.loads(
        (output / "decision.json").read_text(encoding="utf-8")
    )
    decision["gate_passed"] = False
    (output / "decision.json").write_text(
        json.dumps(decision),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fingerprint changed"):
        module._validate_published_output(output)


def test_fixed_plan_contains_no_method_selection_surface() -> None:
    plan = module._fixed_plan_payload()
    assert plan["seed"] == 42
    assert plan["runtime_split"] == "D_V"
    assert plan["fixed_output"] == {
        "rule": (
            "occupancy=(base_probability>=0.72);"
            "completion=(field<0)&~occupancy;"
            "final=occupancy|completion"
        ),
        "zero_tie_policy": "field_equal_zero_is_not_completion",
        "sigmoid_applied": False,
        "PAET_threshold_search_performed": False,
    }
    assert plan["Base@B"]["candidate_count"] == 51
    assert plan["execution_policy"]["model_training_performed"] is False
    assert plan["execution_policy"]["D_T_accessed"] is False
