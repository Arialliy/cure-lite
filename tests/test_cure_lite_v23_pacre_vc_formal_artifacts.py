from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import cure_lite_v23.formal_artifacts as artifacts
from cure_lite.cache.schema import canonical_json, stable_fingerprint
from cure_lite.experiment.coverage_state_training import (
    CoverageStateTrainingResult,
    coverage_state_model_fingerprint,
    coverage_state_optimizer_config_fingerprint,
)
from cure_lite.train.coverage_state_fused_step import (
    CoverageStatePairObjective,
    coverage_state_pair_objective_policy,
)
from cure_lite_v23.factory import build_pacre_vc_training_model
from cure_lite_v23.formal_artifacts import (
    LoadedPACREVCFormalArtifact,
    load_pacre_vc_formal_final_model,
    save_pacre_vc_formal_final_model,
)
from cure_lite_v23.formal_training import (
    PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT,
    PACRE_VC_FORMAL_INITIAL_MODEL_FINGERPRINT,
    PACRE_VC_FORMAL_SCHEDULE_FINGERPRINT,
)
from cure_lite_v23.pacre_vc import (
    CoverageStatePACREVerifierCorrectedConfig,
)
from cure_lite_v23.training import PACRE_PMOPE_TRAINING_CONFIG


class _FakeFormalRunResult:
    """Test-only stand-in for serialization tests after the public type gate."""

    def __init__(
        self,
        model: torch.nn.Module,
        training_result: CoverageStateTrainingResult,
    ) -> None:
        self.model = model
        self.final_model = model
        self.training_result = training_result
        self.authorization = SimpleNamespace(
            authorization_fingerprint="f" * 64,
            source_closure_fingerprint="1" * 64,
        )
        self.source_closure_fingerprint_after = "1" * 64
        self.training_complete = True
        self.result_fingerprint = "e" * 64

    def verify_unchanged(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _patch_exact_formal_result_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        artifacts,
        "CoverageStatePACREVCFormal800RunResult",
        _FakeFormalRunResult,
    )


def _formal_result(model: torch.nn.Module) -> CoverageStateTrainingResult:
    objective = CoverageStatePairObjective.PMOPE_JOINT.value
    fingerprint = coverage_state_model_fingerprint(model)
    config = PACRE_PMOPE_TRAINING_CONFIG
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    return CoverageStateTrainingResult(
        objective=objective,
        objective_policy=coverage_state_pair_objective_policy(objective),
        seed=42,
        epochs=800,
        steps_per_epoch=40,
        completed_updates=32000,
        schedule_fingerprint=PACRE_VC_FORMAL_SCHEDULE_FINGERPRINT,
        cache_fingerprint=PACRE_VC_FORMAL_FULL_CACHE_FINGERPRINT,
        execution_device="cuda:0",
        device_cache_fingerprint="c" * 64,
        device_cache_resident_bytes=1,
        optimizer_config_fingerprint=(
            coverage_state_optimizer_config_fingerprint(model, optimizer)
        ),
        initial_model_fingerprint=(
            PACRE_VC_FORMAL_INITIAL_MODEL_FINGERPRINT
        ),
        final_model_fingerprint=fingerprint,
        epoch_logs=tuple({"epoch": epoch} for epoch in range(800)),
        first_nonzero_gradient_update=(
            ("joint_hidden_bias", 1),
            ("joint_state_weight", 1),
            ("scalar_energy_weight", 0),
        ),
        forward_calls=32000,
        backward_calls=32000,
        optimizer_steps=32000,
        logical_state_evaluations=384000,
        finite_state_audits=32001,
    )


def _model() -> torch.nn.Module:
    config = CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    model = build_pacre_vc_training_model(config)
    with torch.no_grad():
        model.scalar_energy_weight.add_(0.125)
    return model


def test_formal_final_model_is_final_only_and_roundtrips(
    tmp_path: Path,
) -> None:
    model = _model()
    result = _formal_result(model)
    directory = tmp_path / "final_model"

    receipt = save_pacre_vc_formal_final_model(
        directory,
        formal_result=_FakeFormalRunResult(model, result),
    )
    loaded = load_pacre_vc_formal_final_model(directory, receipt)

    assert type(loaded) is LoadedPACREVCFormalArtifact
    assert loaded.receipt == receipt
    assert coverage_state_model_fingerprint(loaded.model) == (
        coverage_state_model_fingerprint(model)
    )
    assert {path.name for path in directory.iterdir()} == {
        "model.safetensors",
        "artifact.json",
    }
    assert receipt["optimizer_state_saved"] is False
    assert receipt["intermediate_checkpoint_saved"] is False
    assert receipt["formal_training_ledger"]["epochs"] == 800
    assert (
        receipt["formal_training_ledger"]["completed_updates"] == 32000
    )
    loaded.verify_unchanged()


def test_formal_loader_rejects_extra_or_tampered_members(
    tmp_path: Path,
) -> None:
    model = _model()
    directory = tmp_path / "final_model"
    receipt = save_pacre_vc_formal_final_model(
        directory,
        formal_result=_FakeFormalRunResult(
            model,
            _formal_result(model),
        ),
    )

    (directory / "optimizer.pt").write_bytes(b"forbidden")
    with pytest.raises(RuntimeError, match="inventory"):
        load_pacre_vc_formal_final_model(directory, receipt)
    (directory / "optimizer.pt").unlink()

    changed = dict(receipt)
    changed["parameter_count"] = 1
    with pytest.raises(ValueError, match="fingerprint"):
        load_pacre_vc_formal_final_model(directory, changed)

    extended = dict(receipt)
    extended["unexpected_extension"] = True
    extended.pop("artifact_fingerprint")
    extended["artifact_fingerprint"] = stable_fingerprint(extended)
    (directory / "artifact.json").write_text(
        canonical_json(extended) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fields changed"):
        load_pacre_vc_formal_final_model(directory, extended)


def test_formal_save_is_create_only(tmp_path: Path) -> None:
    model = _model()
    directory = tmp_path / "final_model"
    result = _formal_result(model)
    save_pacre_vc_formal_final_model(
        directory,
        formal_result=_FakeFormalRunResult(model, result),
    )
    with pytest.raises(FileExistsError):
        save_pacre_vc_formal_final_model(
            directory,
            formal_result=_FakeFormalRunResult(model, result),
        )


def test_formal_save_rejects_nonformal_schedule_or_initial_state(
    tmp_path: Path,
) -> None:
    model = _model()
    result = _formal_result(model)
    forged = CoverageStateTrainingResult(
        **{
            **result.__dict__,
            "schedule_fingerprint": "a" * 64,
            "initial_model_fingerprint": "b" * 64,
        }
    )
    with pytest.raises(ValueError, match="exact completed Formal800"):
        save_pacre_vc_formal_final_model(
            tmp_path / "rejected",
            formal_result=_FakeFormalRunResult(model, forged),
        )


def test_formal_save_rejects_a_generic_caller_assembled_ledger(
    tmp_path: Path,
) -> None:
    model = _model()
    with pytest.raises(TypeError, match="engine-issued"):
        save_pacre_vc_formal_final_model(
            tmp_path / "forged",
            formal_result=_formal_result(model),  # type: ignore[arg-type]
        )
