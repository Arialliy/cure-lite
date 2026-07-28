from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

import cure_lite.experiment.coverage_state_paet_formal_artifacts as artifacts
from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
    CoverageStatePhaseAlignedEvidenceTransportConfig,
)
from cure_lite.experiment.coverage_state_training import (
    COVERAGE_STATE_MATCHED_RESULT_SCHEMA,
    COVERAGE_STATE_TRAINING_RESULT_SCHEMA,
    CoverageStateMatchedTrainingConfig,
    coverage_state_model_contract_payload,
    coverage_state_model_fingerprint,
)
from cure_lite.train.coverage_state_fused_step import (
    CSLF_PMOPE_POLICY,
    CoverageStatePairObjective,
)


def _digest(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _fairness() -> dict[str, object]:
    return {
        "candidate_model": "PAET-BFA",
        "single_candidate_only": True,
        "same_initial_state": True,
        "same_schedule": True,
        "same_endpoints": True,
        "same_model": True,
        "same_optimizer": True,
        "same_device_cache": True,
        "same_compute_budget": True,
        "same_natural_branches": True,
        "historical_controls_retrained": False,
        "historical_v20_objective_reused": True,
        "allowed_difference_from_sealed_v20": (
            "predeclared_field_equation_only"
        ),
        "same_model_class": True,
        "same_model_config": True,
        "same_parameter_count": True,
        "same_parameter_shapes": True,
    }


def _epoch_logs() -> list[dict[str, object]]:
    return [
        {
            "epoch": epoch,
            "completed_updates": (epoch + 1) * 40,
            "objective": CoverageStatePairObjective.PMOPE_JOINT.value,
            "selection_sequence_fingerprint": _digest(
                f"selection-{epoch}"
            ),
            "mean_factual_miss/loss": 0.1,
            "mean_factual_no_miss/loss": 0.2,
            "mean_pair/loss": 0.3,
            "mean_total": 0.6,
            "mean_gradient_l2_norm": 0.4,
        }
        for epoch in range(800)
    ]


class _FakeTraining:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def canonical_payload(self) -> dict[str, object]:
        return self._payload


class _FakeFormalResult:
    def __init__(
        self,
        model: CURELitePhaseAlignedEvidenceTransportLevelSet,
    ) -> None:
        final = coverage_state_model_fingerprint(model)
        initial = artifacts._expected_initial_model_fingerprint(
            artifacts._expected_model_config()
        )
        schedule = (
            artifacts.COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT
        )
        cache = (
            artifacts.COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        )
        logs = _epoch_logs()
        objective = {
            "schema_version": COVERAGE_STATE_TRAINING_RESULT_SCHEMA,
            "objective": CoverageStatePairObjective.PMOPE_JOINT.value,
            "objective_policy": CSLF_PMOPE_POLICY,
            "seed": 42,
            "epochs": 800,
            "steps_per_epoch": 40,
            "completed_updates": 32_000,
            "schedule_fingerprint": schedule,
            "cache_fingerprint": cache,
            "execution_device": "cuda:0",
            "device_cache_fingerprint": "4" * 64,
            "device_cache_resident_bytes": 123,
            "optimizer_config_fingerprint": "5" * 64,
            "initial_model_fingerprint": initial,
            "final_model_fingerprint": final,
            "epoch_logs": logs,
            "first_nonzero_gradient_update": {
                "joint_hidden_bias": 1,
                "joint_state_weight": 1,
                "scalar_energy_weight": 0,
            },
            "compute": {
                "forward_calls": 32_000,
                "backward_calls": 32_000,
                "optimizer_steps": 32_000,
                "logical_state_evaluations": 384_000,
                "finite_state_audits": 32_001,
            },
        }
        training = {
            "schema_version": COVERAGE_STATE_MATCHED_RESULT_SCHEMA,
            "config": CoverageStateMatchedTrainingConfig(
                seed=42
            ).canonical_payload(),
            "common_initial_model_fingerprint": initial,
            "schedule_fingerprint": schedule,
            "cache_fingerprint": cache,
            "objectives": [objective],
            "objective_suite": [
                CoverageStatePairObjective.PMOPE_JOINT.value
            ],
            "fairness": _fairness(),
            "model_contract": coverage_state_model_contract_payload(
                model
            ),
        }
        authorization_fingerprint = "6" * 64
        formal = {
            "schema_version": (
                artifacts.COVERAGE_STATE_PAET_FORMAL_RESULT_SCHEMA
            ),
            "run_id": artifacts.COVERAGE_STATE_PAET_FORMAL_RUN_ID,
            "runtime_splits": ["D_R"],
            "authorization_fingerprint": authorization_fingerprint,
            "structural_advancement_passed": True,
            "generic_population_gate_passed": False,
            "bounded_evidence_interpretation": (
                "structural_advancement_only_not_performance"
            ),
            "training": training,
            "training_invocations": 1,
            "checks": {
                name: True for name in artifacts._FORMAL_CHECK_NAMES
            },
            "failed_checks": [],
            "training_complete": True,
            "field_threshold_hex": 0.0.hex(),
            "threshold_search_performed": False,
            "training_contract": {
                "from_scratch": True,
                "process_local_single_attempt_claim": True,
                "cross_process_output_claim_required": True,
                "resume_allowed": False,
                "automatic_retry_allowed": False,
                "checkpoint_policy": "final_model_only",
                "intermediate_checkpoint_saved": False,
                "optimizer_state_saved": False,
            },
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
        }
        self.final_model = model
        self.training = _FakeTraining(training)
        self.authorization = SimpleNamespace(
            authorization_fingerprint=authorization_fingerprint
        )
        self.training_complete = True
        self._payload = formal

    def verify_unchanged(self) -> None:
        return None

    def canonical_payload(self) -> dict[str, object]:
        return self._payload

    @property
    def result_fingerprint(self) -> str:
        return stable_fingerprint(self._payload)


@pytest.fixture
def fake_result(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        artifacts,
        "CoverageStatePAETFormal800RunResult",
        _FakeFormalResult,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = CURELitePhaseAlignedEvidenceTransportLevelSet(
            artifacts._expected_model_config()
        )
    with torch.no_grad():
        model.scalar_energy_weight.fill_(0.125)
    return _FakeFormalResult(model)


@pytest.fixture
def saved_artifact(
    tmp_path: Path,
    fake_result: _FakeFormalResult,
) -> tuple[Path, _FakeFormalResult, str]:
    target = tmp_path / "formal"
    fingerprint = artifacts.save_coverage_state_paet_formal_artifact(
        target,
        fake_result,
    )
    return target, fake_result, fingerprint


def _read_receipt(path: Path) -> dict[str, object]:
    return json.loads((path / "receipt.json").read_text(encoding="utf-8"))


def _write_receipt(path: Path, receipt: dict[str, object]) -> None:
    (path / "receipt.json").write_text(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _reseal_receipt(receipt: dict[str, object]) -> None:
    receipt.pop("artifact_fingerprint", None)
    receipt["artifact_fingerprint"] = stable_fingerprint(receipt)


def test_round_trip_is_final_only_and_strictly_bound(
    saved_artifact,
):
    target, result, fingerprint = saved_artifact
    loaded = artifacts.load_coverage_state_paet_formal_artifact(
        target,
        expected_authorization_fingerprint=(
            result.authorization.authorization_fingerprint
        ),
        expected_result_fingerprint=result.result_fingerprint,
    )

    assert {path.name for path in target.iterdir()} == {
        "model.safetensors",
        "formal_result.json",
        "training.json",
        "epoch_log.json",
        "receipt.json",
    }
    assert not any("optim" in path.name for path in target.iterdir())
    assert not any("checkpoint" in path.name for path in target.iterdir())
    assert loaded.artifact_fingerprint == fingerprint
    assert loaded.formal_result_fingerprint == result.result_fingerprint
    assert loaded.training_model_fingerprint == (
        coverage_state_model_fingerprint(result.final_model)
    )
    assert loaded.module_state_fingerprint != (
        loaded.training_model_fingerprint
    )
    assert len(loaded.epoch_logs) == 800
    assert loaded.model_config.expected_parameter_count == 64_064
    assert not loaded.model.training
    assert not any(
        parameter.requires_grad
        for parameter in loaded.model.parameters()
    )
    loaded.verify_unchanged()


def test_save_refuses_overwrite_and_wrong_model_contract(
    saved_artifact,
    fake_result: _FakeFormalResult,
):
    target, _, _ = saved_artifact
    with pytest.raises(FileExistsError, match="overwrite"):
        artifacts.save_coverage_state_paet_formal_artifact(
            target,
            fake_result,
        )

    wrong = CURELitePhaseAlignedEvidenceTransportLevelSet(
        CoverageStatePhaseAlignedEvidenceTransportConfig(
            feature_channels=64,
            feature_stride=4,
            width=16,
        )
    )
    with pytest.raises(ValueError, match="non-formal|not Formal800"):
        artifacts.save_coverage_state_paet_formal_artifact(
            target.parent / "wrong",
            _FakeFormalResult(wrong),
        )


def test_expected_fingerprints_are_enforced(saved_artifact):
    target, _, _ = saved_artifact
    with pytest.raises(ValueError, match="authorization differs"):
        artifacts.load_coverage_state_paet_formal_artifact(
            target,
            expected_authorization_fingerprint="7" * 64,
        )
    with pytest.raises(ValueError, match="result differs"):
        artifacts.load_coverage_state_paet_formal_artifact(
            target,
            expected_result_fingerprint="8" * 64,
        )


@pytest.mark.parametrize(
    "member",
    ["model.safetensors", "formal_result.json", "training.json", "epoch_log.json"],
)
def test_member_hash_tampering_is_rejected(
    saved_artifact,
    tmp_path: Path,
    member: str,
):
    target, _, _ = saved_artifact
    changed = tmp_path / f"changed-{member.replace('.', '-')}"
    shutil.copytree(target, changed)
    with (changed / member).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        artifacts.load_coverage_state_paet_formal_artifact(changed)


def test_duplicate_key_and_nonfinite_json_are_rejected(
    saved_artifact,
    tmp_path: Path,
):
    target, _, _ = saved_artifact
    duplicate = tmp_path / "duplicate"
    shutil.copytree(target, duplicate)
    receipt_path = duplicate / "receipt.json"
    text = receipt_path.read_text(encoding="utf-8")
    receipt_path.write_text(
        '{"schema_version":"duplicate",' + text[1:],
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        artifacts.load_coverage_state_paet_formal_artifact(duplicate)

    nonfinite = tmp_path / "nonfinite"
    shutil.copytree(target, nonfinite)
    receipt_path = nonfinite / "receipt.json"
    text = receipt_path.read_text(encoding="utf-8")
    receipt_path.write_text(
        text.replace('"parameter_count":64064', '"parameter_count":NaN'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite"):
        artifacts.load_coverage_state_paet_formal_artifact(nonfinite)


def test_coherently_resealed_config_and_state_tampering_are_rejected(
    saved_artifact,
    tmp_path: Path,
):
    target, _, _ = saved_artifact
    config_changed = tmp_path / "config-changed"
    shutil.copytree(target, config_changed)
    receipt = _read_receipt(config_changed)
    receipt["model_config"]["width"] = 16
    _reseal_receipt(receipt)
    _write_receipt(config_changed, receipt)
    with pytest.raises(ValueError, match="frozen Formal800 config"):
        artifacts.load_coverage_state_paet_formal_artifact(config_changed)

    state_changed = tmp_path / "state-changed"
    shutil.copytree(target, state_changed)
    from safetensors.torch import load_file, save_file

    weights = load_file(
        str(state_changed / "model.safetensors"),
        device="cpu",
    )
    weights["scalar_energy_weight"] = (
        weights["scalar_energy_weight"] + 0.25
    )
    save_file(weights, str(state_changed / "replacement.safetensors"))
    (state_changed / "replacement.safetensors").replace(
        state_changed / "model.safetensors"
    )
    receipt = _read_receipt(state_changed)
    receipt["weights_sha256"] = file_sha256(
        state_changed / "model.safetensors"
    )
    _reseal_receipt(receipt)
    _write_receipt(state_changed, receipt)
    with pytest.raises(ValueError, match="model fingerprint mismatch"):
        artifacts.load_coverage_state_paet_formal_artifact(state_changed)


def test_symlinks_inventory_and_parent_traversal_are_rejected(
    saved_artifact,
    tmp_path: Path,
):
    target, _, _ = saved_artifact
    directory_link = tmp_path / "directory-link"
    directory_link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        artifacts.load_coverage_state_paet_formal_artifact(directory_link)

    member_link = tmp_path / "member-link"
    shutil.copytree(target, member_link)
    weights = member_link / "model.safetensors"
    actual = member_link / "actual.safetensors"
    weights.rename(actual)
    weights.symlink_to(actual.name)
    with pytest.raises(ValueError, match="inventory"):
        artifacts.load_coverage_state_paet_formal_artifact(member_link)

    unexpected = tmp_path / "unexpected"
    shutil.copytree(target, unexpected)
    (unexpected / "optimizer.pt").write_bytes(b"forbidden")
    with pytest.raises(ValueError, match="inventory"):
        artifacts.load_coverage_state_paet_formal_artifact(unexpected)

    traversal = target.parent / "unused" / ".." / target.name
    with pytest.raises(ValueError, match="parent traversal"):
        artifacts.load_coverage_state_paet_formal_artifact(traversal)


def test_loaded_memory_and_disk_mutation_are_detected(
    saved_artifact,
):
    target, _, _ = saved_artifact
    loaded = artifacts.load_coverage_state_paet_formal_artifact(target)
    with torch.no_grad():
        loaded.model.scalar_energy_weight.add_(1.0)
    with pytest.raises(RuntimeError, match="changed in memory"):
        loaded.verify_unchanged()

    loaded = artifacts.load_coverage_state_paet_formal_artifact(target)
    with (target / "epoch_log.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(RuntimeError, match="changed on disk"):
        loaded.verify_unchanged()
