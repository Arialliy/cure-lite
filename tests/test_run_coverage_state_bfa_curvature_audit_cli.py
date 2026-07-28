from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cure_lite.coverage_state_binary_flip_antisymmetric import (
    CURELiteBinaryFlipAntisymmetricLevelSet,
    CoverageStateBinaryFlipAntisymmetricConfig,
)
from cure_lite.frozen_base import module_state_fingerprint
from tools import run_coverage_state_bfa_curvature_audit as cli


def test_frozen_authority_constants_are_exact_v20_r2() -> None:
    assert cli.SOURCE_RUN_ID == (
        "cure_lite_bfa_cmif_v20_pmope_bounded_400_r2"
    )
    assert cli.RUN_ID == (
        "cure_lite_bfa_cmif_v20_curvature_audit_r1"
    )
    assert cli.FROZEN_DEVICE == "cuda:0"
    assert cli.FROZEN_SEED == 42
    assert cli.FROZEN_FEATURE_CHANNELS == 64
    assert cli.FROZEN_FEATURE_STRIDE == 4
    assert cli.FROZEN_MODEL_WIDTH == 32
    assert cli.FROZEN_PARAMETER_COUNT == 64064
    for value in (
        cli.SOURCE_COMPLETE_SHA256,
        cli.SOURCE_COMPLETE_FINGERPRINT,
        cli.SOURCE_ZERO_SHA256,
        cli.SOURCE_ZERO_RECEIPT_FINGERPRINT,
        cli.SOURCE_ZERO_DIAGNOSTIC_FINGERPRINT,
        cli.SOURCE_CHECKPOINT_SHA256,
        cli.SOURCE_CHECKPOINT_RECEIPT_SHA256,
        cli.SOURCE_CHECKPOINT_RECEIPT_FINGERPRINT,
        cli.SOURCE_MODULE_STATE_FINGERPRINT,
        cli.SOURCE_INPUT_RECEIPT_SHA256,
        cli.SOURCE_INPUT_RECEIPT_FINGERPRINT,
        cli.SOURCE_POPULATION_FINGERPRINT,
        cli.SOURCE_CACHE_FINGERPRINT,
    ):
        assert cli._is_sha256(value)


def test_repository_v20_r2_bindings_are_exact_and_read_only() -> None:
    before = {
        relative: cli.file_sha256(cli.SOURCE_RUN_PATH / relative)
        for relative in (
            cli.SOURCE_COMPLETE_RELATIVE,
            cli.SOURCE_ZERO_RELATIVE,
            cli.SOURCE_CHECKPOINT_RELATIVE,
            cli.SOURCE_CHECKPOINT_RECEIPT_RELATIVE,
            cli.SOURCE_INPUT_RECEIPT_RELATIVE,
        )
    }
    binding = cli._verify_frozen_v20_bindings()
    after = {
        relative: cli.file_sha256(cli.SOURCE_RUN_PATH / relative)
        for relative in before
    }

    assert before == after
    assert binding["complete"]["complete_fingerprint"] == (
        cli.SOURCE_COMPLETE_FINGERPRINT
    )
    assert binding["zero"]["receipt_fingerprint"] == (
        cli.SOURCE_ZERO_RECEIPT_FINGERPRINT
    )
    assert binding["checkpoint_receipt"]["receipt_fingerprint"] == (
        cli.SOURCE_CHECKPOINT_RECEIPT_FINGERPRINT
    )
    assert binding["candidate_diagnostic"]["checkpoint_fingerprint"] == (
        cli.SOURCE_MODULE_STATE_FINGERPRINT
    )


def test_validate_bindings_does_not_enter_dr_or_claim_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "OUTPUT_PATH", tmp_path / "not-claimed")
    monkeypatch.setattr(
        cli,
        "_verify_frozen_v20_bindings",
        lambda: {
            "complete": {
                "complete_fingerprint": cli.SOURCE_COMPLETE_FINGERPRINT
            }
        },
    )
    monkeypatch.setattr(
        cli,
        "_implementation_binding",
        lambda: {"audit.py": "1" * 64},
    )
    monkeypatch.setattr(
        cli,
        "_build_frozen_population",
        lambda: pytest.fail("validate must not build D_R"),
    )
    monkeypatch.setattr(
        cli,
        "_load_exact_bfa_model",
        lambda *args, **kwargs: pytest.fail(
            "validate must not load checkpoint tensors"
        ),
    )
    monkeypatch.setattr(
        cli,
        "_claim_output",
        lambda *args, **kwargs: pytest.fail(
            "validate must not claim output"
        ),
    )

    result = cli.validate_bindings()

    assert result["status"] == "bindings_valid"
    assert result["runtime_splits"] == []
    assert result["D_R_tensor_payload_accessed"] is False
    assert result["training_performed"] is False
    assert result["backward_performed"] is False
    assert result["optimizer_constructed"] is False
    assert result["D_V_accessed"] is False
    assert result["D_T_accessed"] is False
    assert result["output_claimed"] is False
    assert not cli.OUTPUT_PATH.exists()


def test_exact_loader_uses_strict_tensor_state_without_gradients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = CoverageStateBinaryFlipAntisymmetricConfig(
        feature_channels=cli.FROZEN_FEATURE_CHANNELS,
        feature_stride=cli.FROZEN_FEATURE_STRIDE,
        width=cli.FROZEN_MODEL_WIDTH,
    )
    source = CURELiteBinaryFlipAntisymmetricLevelSet(config).eval()
    from safetensors.torch import save_file

    checkpoint = tmp_path / "bfa.safetensors"
    save_file(source.state_dict(), str(checkpoint))
    expected_state = module_state_fingerprint(source)
    monkeypatch.setattr(cli, "FROZEN_DEVICE", "cpu")
    monkeypatch.setattr(
        cli,
        "SOURCE_CHECKPOINT_SHA256",
        cli.file_sha256(checkpoint),
    )
    monkeypatch.setattr(
        cli,
        "SOURCE_MODULE_STATE_FINGERPRINT",
        expected_state,
    )

    loaded = cli._load_exact_bfa_model(checkpoint, device="cpu")

    assert type(loaded) is CURELiteBinaryFlipAntisymmetricLevelSet
    assert loaded.training is False
    assert module_state_fingerprint(loaded) == expected_state
    assert all(not value.requires_grad for value in loaded.parameters())
    assert all(value.device.type == "cpu" for value in loaded.state_dict().values())


def test_run_audit_prefers_requested_adapter_and_passes_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cure_lite.experiment import (
        coverage_state_bfa_curvature_audit as audit,
    )

    received: dict[str, object] = {}

    def preferred(
        model: object,
        population: object,
        baseline: object,
        *,
        device: str,
    ) -> dict[str, object]:
        received.update(
            model=model,
            population=population,
            baseline=baseline,
            device=device,
        )
        return {"decision": "test"}

    monkeypatch.setattr(
        audit,
        "run_bfa_curvature_audit",
        preferred,
        raising=False,
    )
    model = object()
    real_inputs = object()
    population = object()
    baseline = {"frozen": True}

    result = cli._run_audit(
        model,  # type: ignore[arg-type]
        real_inputs,
        population,
        baseline,
        device="cuda:0",
    )

    assert result == {"decision": "test"}
    assert received == {
        "model": model,
        "population": population,
        "baseline": baseline,
        "device": "cuda:0",
    }


def test_run_audit_adapts_to_strict_frozen_r2_core_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cure_lite.experiment import (
        coverage_state_bfa_curvature_audit as audit,
    )

    monkeypatch.delattr(
        audit,
        "run_bfa_curvature_audit",
        raising=False,
    )
    baseline_path = cli.SOURCE_RUN_PATH / cli.SOURCE_ZERO_RELATIVE
    baseline = json.loads(
        baseline_path.read_text(encoding="utf-8")
    )["candidate_diagnostic"]
    received: dict[str, object] = {}

    def strict(*args: object, **kwargs: object) -> dict[str, object]:
        received["args"] = args
        received["kwargs"] = kwargs
        return {"decision": "REJECTED"}

    monkeypatch.setattr(
        audit,
        "audit_frozen_coverage_state_bfa_v20_r2_curvature_checkpoint",
        strict,
    )
    model = object()
    real_inputs = object()
    population = object()

    result = cli._run_audit(
        model,  # type: ignore[arg-type]
        real_inputs,
        population,
        baseline,
        device="cuda:0",
    )

    assert result == {"decision": "REJECTED"}
    assert received["args"] == (model, real_inputs, population)
    assert received["kwargs"] == {
        "device": "cuda:0",
        "complete_fingerprint": cli.SOURCE_COMPLETE_FINGERPRINT,
        "zero_receipt_fingerprint": (
            cli.SOURCE_ZERO_RECEIPT_FINGERPRINT
        ),
        "diagnostic_fingerprint": (
            cli.SOURCE_ZERO_DIAGNOSTIC_FINGERPRINT
        ),
        "checkpoint_file_sha256": cli.SOURCE_CHECKPOINT_SHA256,
    }


def test_run_once_writes_one_read_only_receipt_and_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "curvature"
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)
    monkeypatch.setattr(
        cli,
        "_verify_frozen_v20_bindings",
        lambda: {
            "checkpoint_path": tmp_path / "unused.safetensors",
            "candidate_diagnostic": {
                "checkpoint_fingerprint": (
                    cli.SOURCE_MODULE_STATE_FINGERPRINT
                )
            },
        },
    )
    monkeypatch.setattr(
        cli,
        "_implementation_binding",
        lambda: {"audit.py": "2" * 64},
    )
    real_inputs = object()
    population = SimpleNamespace(
        population_fingerprint=cli.SOURCE_POPULATION_FINGERPRINT,
        cache=SimpleNamespace(
            cache_fingerprint=cli.SOURCE_CACHE_FINGERPRINT
        ),
    )
    monkeypatch.setattr(
        cli,
        "_build_frozen_population",
        lambda: (real_inputs, population),
    )
    model = torch.nn.Linear(1, 1).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    monkeypatch.setattr(
        cli,
        "_load_exact_bfa_model",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        cli,
        "module_state_fingerprint",
        lambda value: cli.SOURCE_MODULE_STATE_FINGERPRINT,
    )
    calls = {"count": 0}

    def run_audit(*args: object, **kwargs: object) -> dict[str, object]:
        calls["count"] += 1
        assert torch.is_inference_mode_enabled()
        assert args == (
            model,
            real_inputs,
            population,
            {
                "checkpoint_fingerprint": (
                    cli.SOURCE_MODULE_STATE_FINGERPRINT
                )
            },
        )
        assert kwargs == {"device": "cuda:0"}
        return {
            "decision": "REJECTED",
            "execution": {
                "training_performed": False,
                "backward_performed": False,
                "optimizer_constructed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            },
        }

    monkeypatch.setattr(cli, "_run_audit", run_audit)

    complete = cli.run_once()

    assert calls["count"] == 1
    assert complete["status"] == "complete"
    assert complete["decision"] == "REJECTED"
    assert complete["runtime_splits"] == ["D_R"]
    assert complete["training_performed"] is False
    assert complete["backward_performed"] is False
    assert complete["optimizer_constructed"] is False
    assert complete["optimizer_step_performed"] is False
    assert complete["D_V_accessed"] is False
    assert complete["D_T_accessed"] is False
    assert complete["checkpoint_written"] is False
    assert complete["artifact_file_count"] == 1
    assert not (output / cli._INCOMPLETE).exists()
    receipt_path = output / "receipts/curvature_audit.json"
    assert receipt_path.is_file()
    assert (output / "COMPLETE.json").is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["execution"]["audit_invocations"] == 1
    assert receipt["execution"]["training_performed"] is False
    assert receipt["execution"]["D_V_accessed"] is False
    assert receipt["execution"]["D_T_accessed"] is False
    assert receipt["source_run"]["read_only"] is True
    assert set(output.iterdir()) == {
        output / "receipts",
        output / "COMPLETE.json",
    }

    with pytest.raises(FileExistsError, match="single-use"):
        cli.run_once()


def test_cli_requires_exactly_one_mode() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args([])
    with pytest.raises(SystemExit):
        cli.parse_args(["--validate-bindings", "--run-once"])
    assert cli.parse_args(["--validate-bindings"]).validate_bindings
    assert cli.parse_args(["--run-once"]).run_once
