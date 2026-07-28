from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite_v22.pacre import (
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CoverageStatePACREConfig,
)
from tools import run_cure_lite_v22_pacre_bounded_400 as cli


def _dataset_free_receipt() -> dict[str, object]:
    body = {
        "schema_version": "generated-test",
        "candidate": "PACRE-v22",
        "gate_passed": True,
        "parameter_count": cli.FROZEN_PARAMETER_COUNT,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "training_performed": False,
    }
    return {
        **body,
        "receipt_fingerprint": stable_fingerprint(body),
    }


def _runtime_receipt() -> dict[str, object]:
    """Return the pre-claim runtime envelope, which must not touch CUDA."""

    return {
        "device": cli.FROZEN_DEVICE,
        "CUDA_VISIBLE_DEVICES": cli.FROZEN_VISIBLE_GPU,
        "CUBLAS_WORKSPACE_CONFIG": (
            cli.FROZEN_CUBLAS_WORKSPACE_CONFIG
        ),
        "temperature_wrapper_repo_path": (
            cli.TEMPERATURE_WRAPPER_REPO_PATH
        ),
        "temperature_wrapper_file_sha256": (
            cli.TEMPERATURE_WRAPPER_FILE_SHA256
        ),
        "pause_temperature_c": cli.FROZEN_PAUSE_TEMPERATURE_C,
        "resume_temperature_c": cli.FROZEN_RESUME_TEMPERATURE_C,
    }


def _verified_runtime_receipt() -> dict[str, object]:
    return {
        **_runtime_receipt(),
        "visible_device_count": 1,
        "visible_device_index": 0,
        "device_name": "generated CUDA device",
        "device_total_memory_bytes": 1024,
        "cuda_runtime_verified_after_output_claim": True,
    }


def _valid_attempt() -> dict[str, object]:
    return cli._fingerprinted(
        {
            "schema_version": cli.ATTEMPT_SCHEMA,
            "run_id": cli.RUN_ID,
            "output_repo_path": cli.OUTPUT_REPO_PATH,
            "config_fingerprint": "a" * 64,
            "runtime": _runtime_receipt(),
            "candidate": "PACRE-v22",
            "objective": cli.PACRE_PMOPE_OBJECTIVE,
            "budget": {
                "seed": cli.FROZEN_SEED,
                "epochs": cli.FROZEN_EPOCHS,
                "steps_per_epoch": cli.FROZEN_STEPS_PER_EPOCH,
                "updates": cli.FROZEN_UPDATES,
            },
            "process_identity": cli.pacre_bounded_process_identity(),
            "dataset_free_receipt_fingerprint": "b" * 64,
            "dataset_free_invocations_before_claim": 1,
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "formal_800_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )


def _generated_run_graph(
    *,
    dr_passed: bool,
) -> tuple[
    SimpleNamespace,
    SimpleNamespace,
    SimpleNamespace,
    SimpleNamespace,
]:
    source_binding = SimpleNamespace(
        canonical_payload=lambda: {"split": "D_R"},
    )
    scalar_cache = object()
    real_inputs = SimpleNamespace(
        scalar_cache=scalar_cache,
        source_binding=source_binding,
        build_fingerprint="1" * 64,
        canonical_payload=lambda: {
            "split": "D_R",
            "build_fingerprint": "1" * 64,
        },
    )
    bounded_cache_fingerprint = "2" * 64
    bounded_cache = SimpleNamespace(
        cache_fingerprint=bounded_cache_fingerprint
    )
    population_payload = {
        "seed": 42,
        "bounded_cache_fingerprint": bounded_cache_fingerprint,
    }
    population_fingerprint = stable_fingerprint(population_payload)
    population = SimpleNamespace(
        cache=bounded_cache,
        population_fingerprint=population_fingerprint,
        canonical_payload=lambda: dict(population_payload),
    )
    schedule_payload = {
        "seed": 42,
        "epochs": 10,
        "steps_per_epoch": 40,
        "updates": 400,
    }
    schedule_fingerprint = stable_fingerprint(schedule_payload)
    schedule = SimpleNamespace(
        canonical_payload=lambda: dict(schedule_payload),
    )
    preflight_payload = {
        "training_authorized": True,
        "population_fingerprint": population_fingerprint,
        "bounded_cache_fingerprint": bounded_cache_fingerprint,
        "schedule_fingerprint": schedule_fingerprint,
    }
    preflight_fingerprint = stable_fingerprint(preflight_payload)
    preflight = SimpleNamespace(
        population=population,
        schedule=schedule,
        preflight_fingerprint=preflight_fingerprint,
        training_authorized=True,
        canonical_payload=lambda: dict(preflight_payload),
    )
    dataset_free_fingerprint = str(
        _dataset_free_receipt()["receipt_fingerprint"]
    )
    dr_payload = {
        "gate_passed": dr_passed,
        "decision": (
            cli.PACRE_DR_PASS_DECISION
            if dr_passed
            else cli.PACRE_DR_FAIL_DECISION
        ),
        "dataset_free_receipt_fingerprint": (
            dataset_free_fingerprint
        ),
        "real_inputs_fingerprint": real_inputs.build_fingerprint,
        "population_fingerprint": population_fingerprint,
        "cache_fingerprint": bounded_cache_fingerprint,
    }
    dr_fingerprint = stable_fingerprint(dr_payload)
    dr_gate = SimpleNamespace(
        gate_passed=dr_passed,
        decision=(
            cli.PACRE_DR_PASS_DECISION
            if dr_passed
            else cli.PACRE_DR_FAIL_DECISION
        ),
        receipt_fingerprint=dr_fingerprint,
        failed_checks=() if dr_passed else ("generated_failure",),
        canonical_payload=lambda: dict(dr_payload),
    )
    return real_inputs, population, preflight, dr_gate


def _patch_common_run_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    dr_passed: bool,
) -> tuple[Path, dict[str, int], SimpleNamespace]:
    generated_repo_path = f"runs/generated/{cli.RUN_ID}"
    output = tmp_path / generated_repo_path
    monkeypatch.setattr(cli, "_ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)
    monkeypatch.setattr(
        cli,
        "OUTPUT_REPO_PATH",
        generated_repo_path,
    )
    implementation = (("generated.py", "a" * 64),)
    monkeypatch.setattr(
        cli,
        "_implementation_binding",
        lambda: implementation,
    )
    monkeypatch.setattr(cli, "_verify_frozen_sources", lambda: {})
    monkeypatch.setattr(
        cli,
        "_verify_runtime_contract",
        _runtime_receipt,
    )

    counts = {
        "dataset_free": 0,
        "output_claim": 0,
        "cuda_runtime": 0,
        "real_inputs": 0,
        "population": 0,
        "preflight": 0,
        "dr_gate": 0,
        "authorization": 0,
        "runner": 0,
    }
    dataset_free = _dataset_free_receipt()
    real_inputs, population, preflight, dr_gate = (
        _generated_run_graph(dr_passed=dr_passed)
    )

    def output_claim_once() -> SimpleNamespace:
        counts["output_claim"] += 1
        assert output.is_dir()
        assert (output / ".incomplete").is_file()
        assert (output / "attempt.json").is_file()
        attempt = json.loads(
            (output / "attempt.json").read_text(encoding="utf-8")
        )
        claim_payload = {
            "schema_version": (
                "cure-lite-v22-pacre-bounded-output-claim-v1"
            ),
            "run_id": cli.RUN_ID,
            "output_repo_path": cli.OUTPUT_REPO_PATH,
            "config_fingerprint": attempt["config_fingerprint"],
            "runtime": attempt["runtime"],
            "dataset_free_receipt_fingerprint": (
                attempt["dataset_free_receipt_fingerprint"]
            ),
            "attempt_receipt_fingerprint": (
                attempt["receipt_fingerprint"]
            ),
            "process_identity": attempt["process_identity"],
            "exclusive_directory_claimed": True,
            "incomplete_marker_present": True,
            "complete_marker_absent": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
        }
        return SimpleNamespace(
            claim_fingerprint=stable_fingerprint(claim_payload),
            canonical_payload=lambda: dict(claim_payload),
        )

    def cuda_runtime_after_claim(
        envelope: object,
    ) -> dict[str, object]:
        counts["cuda_runtime"] += 1
        assert envelope == _runtime_receipt()
        assert counts["output_claim"] == 1
        assert output.is_dir()
        assert (output / ".incomplete").is_file()
        assert (output / "attempt.json").is_file()
        return _verified_runtime_receipt()

    def dataset_free_once() -> dict[str, object]:
        counts["dataset_free"] += 1
        return dataset_free

    def real_inputs_once(**kwargs: object) -> SimpleNamespace:
        assert kwargs == {}
        counts["real_inputs"] += 1
        return real_inputs

    def population_once(
        cache: object,
        *,
        seed: int,
    ) -> SimpleNamespace:
        assert cache is real_inputs.scalar_cache
        assert seed == 42
        counts["population"] += 1
        return population

    def preflight_once(value: object) -> SimpleNamespace:
        assert value is population
        counts["preflight"] += 1
        return preflight

    def dr_once(**kwargs: object) -> SimpleNamespace:
        assert kwargs == {
            "dataset_free_receipt": dataset_free,
            "real_inputs": real_inputs,
            "bounded_population": population,
            "device": "cuda:0",
        }
        counts["dr_gate"] += 1
        return dr_gate

    monkeypatch.setattr(
        cli,
        "run_pacre_dataset_free_gate",
        dataset_free_once,
    )
    monkeypatch.setattr(
        cli,
        "load_pacre_bounded_output_claim",
        output_claim_once,
    )
    monkeypatch.setattr(
        cli,
        "_verify_cuda_runtime_contract",
        cuda_runtime_after_claim,
    )
    monkeypatch.setattr(
        cli,
        "build_coverage_state_real_dr_inputs",
        real_inputs_once,
    )
    monkeypatch.setattr(
        cli,
        "build_coverage_state_bounded_population",
        population_once,
    )
    monkeypatch.setattr(
        cli,
        "prepare_coverage_state_bounded_preflight",
        preflight_once,
    )
    monkeypatch.setattr(cli, "run_pacre_dr_gate", dr_once)
    return output, counts, preflight


def test_fixed_protocol_and_cli_surface() -> None:
    assert cli.RUN_ID == (
        "cure_lite_pacre_v22_pmope_bounded_400_seed42_r1"
    )
    assert cli.FROZEN_DEVICE == "cuda:0"
    assert cli.FROZEN_VISIBLE_GPU == "0"
    assert cli.FROZEN_CUBLAS_WORKSPACE_CONFIG == ":4096:8"
    assert cli.FROZEN_SEED == 42
    assert cli.FROZEN_EPOCHS == 10
    assert cli.FROZEN_STEPS_PER_EPOCH == 40
    assert cli.FROZEN_UPDATES == 400
    assert cli.FROZEN_FEATURE_CHANNELS == 64
    assert cli.FROZEN_FEATURE_STRIDE == 4
    assert cli.FROZEN_MODEL_WIDTH == 32
    assert cli.FROZEN_PARAMETER_COUNT == 64064
    assert cli.FROZEN_PAUSE_TEMPERATURE_C == 82
    assert cli.FROZEN_RESUME_TEMPERATURE_C == 75

    config = cli._static_config_payload(
        source_paths={},
        implementation=(("generated.py", "a" * 64),),
        dataset_free_receipt_fingerprint="b" * 64,
    )
    assert config["model"]["candidate"] == "PACRE-v22"
    assert config["model"]["input_interface"] == ["F_b", "O"]
    assert config["model"]["single_completion_field"] is True
    assert config["model"]["additional_heads"] == 0
    assert config["model"]["additional_branches"] == 0
    assert config["model"]["parameter_count"] == 64064
    assert config["budget"] == {
        "seed": 42,
        "epochs": 10,
        "steps_per_epoch": 40,
        "updates": 400,
        "objectives": 1,
    }
    assert config["execution"]["resume_allowed"] is False
    assert config["execution"]["automatic_retry_allowed"] is False

    assert cli.parse_args(
        ("--validate-create-only",)
    ).validate_create_only
    assert cli.parse_args(("--run-once",)).run_once
    for arguments in (
        (),
        ("--run-once", "--seed", "43"),
        ("--run-once", "--epochs", "11"),
        ("--run-once", "--gpu", "1"),
        ("--run-once", "--output", "/tmp/forbidden"),
        ("--run-once", "--resume"),
        ("--run-once", "--retry"),
        ("--validate-create-only", "--run-once"),
    ):
        with pytest.raises(SystemExit):
            cli.parse_args(arguments)


def test_claim_validates_fixed_attempt_before_creating_output(
    tmp_path: Path,
) -> None:
    valid = _valid_attempt()
    assert cli._validate_attempt_receipt(valid) == (
        valid["receipt_fingerprint"]
    )

    wrong_schema = dict(valid)
    wrong_schema["schema_version"] = "wrong-attempt-schema"
    wrong_schema_body = dict(wrong_schema)
    wrong_schema_body.pop("receipt_fingerprint")
    wrong_schema["receipt_fingerprint"] = stable_fingerprint(
        wrong_schema_body
    )
    wrong_fingerprint = {**valid, "receipt_fingerprint": "f" * 64}
    extra_field = {**valid, "unexpected": True}

    for index, invalid in enumerate(
        (wrong_schema, wrong_fingerprint, extra_field)
    ):
        output = tmp_path / f"invalid-{index}"
        with pytest.raises(
            ValueError,
            match="attempt",
        ):
            cli._claim_output(output, attempt=invalid)
        assert not output.exists()


def test_validate_create_only_never_enters_real_path_or_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)
    monkeypatch.setattr(cli, "_verify_frozen_sources", lambda: {})
    monkeypatch.setattr(
        cli,
        "_implementation_binding",
        lambda: (("generated.py", "a" * 64),),
    )
    calls = {"dataset_free": 0}

    def dataset_free_once() -> dict[str, object]:
        calls["dataset_free"] += 1
        return _dataset_free_receipt()

    monkeypatch.setattr(
        cli,
        "run_pacre_dataset_free_gate",
        dataset_free_once,
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("create-only entered a real run path")

    for name in (
        "_claim_output",
        "_verify_runtime_contract",
        "_verify_cuda_runtime_contract",
        "load_pacre_bounded_output_claim",
        "build_coverage_state_real_dr_inputs",
        "build_coverage_state_bounded_population",
        "prepare_coverage_state_bounded_preflight",
        "run_pacre_dr_gate",
        "prepare_pacre_bounded_run_authorization",
        "run_pacre_pmope_bounded_400",
    ):
        monkeypatch.setattr(cli, name, forbidden)

    receipt = cli.validate_create_only()
    assert calls == {"dataset_free": 1}
    assert not output.exists()
    assert receipt["static_contract_valid"] is True
    assert receipt["dataset_free_invocations"] == 1
    assert receipt["output_claimed"] is False
    assert receipt["D_R_cached_tensor_payload_accessed"] is False
    assert receipt["D_R_gate_performed"] is False
    assert receipt["training_performed"] is False
    assert receipt["checkpoint_written"] is False


def test_existing_output_stops_before_every_prerequisite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "already-claimed"
    output.mkdir()
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("a prerequisite ran after output existed")

    for name in (
        "_verify_frozen_sources",
        "_implementation_binding",
        "_verify_runtime_contract",
        "_verify_cuda_runtime_contract",
        "load_pacre_bounded_output_claim",
        "run_pacre_dataset_free_gate",
        "build_coverage_state_real_dr_inputs",
        "run_pacre_dr_gate",
        "run_pacre_pmope_bounded_400",
    ):
        monkeypatch.setattr(cli, name, forbidden)
    with pytest.raises(FileExistsError, match="already exists"):
        cli.run_once()


def test_dr_gate_failure_is_complete_and_never_trains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, counts, preflight = _patch_common_run_prerequisites(
        monkeypatch,
        tmp_path,
        dr_passed=False,
    )
    model_config = object()
    monkeypatch.setattr(
        cli,
        "_expected_model_config",
        lambda value: (
            model_config
            if value is preflight
            else pytest.fail("wrong preflight")
        ),
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("D_R failure reached authorization/training")

    monkeypatch.setattr(
        cli,
        "prepare_pacre_bounded_run_authorization",
        forbidden,
    )
    monkeypatch.setattr(
        cli,
        "run_pacre_pmope_bounded_400",
        forbidden,
    )
    result = cli.run_once()

    assert counts == {
        "dataset_free": 1,
        "output_claim": 1,
        "cuda_runtime": 1,
        "real_inputs": 1,
        "population": 1,
        "preflight": 1,
        "dr_gate": 1,
        "authorization": 0,
        "runner": 0,
    }
    assert result["decision"] == "PACRE_V22_D_R_GATE_FAIL"
    assert result["bounded_gate_passed"] is False
    assert (output / "COMPLETE.json").is_file()
    assert not (output / ".incomplete").exists()
    assert not (output / "FAILURE.json").exists()
    assert list((output / "checkpoints").iterdir()) == []
    complete = json.loads(
        (output / "COMPLETE.json").read_text(encoding="utf-8")
    )
    assert complete["artifact_file_count"] == 7
    assert complete["D_R_gate_passed"] is False
    assert complete["authorization_created"] is False
    assert complete["bounded_training_performed"] is False
    assert complete["bounded_runner_invocations"] == 0
    assert complete["checkpoint_count"] == 0
    assert complete["formal_800_authorized"] is False


@pytest.mark.parametrize(
    "replacement",
    ("schema", "self_fingerprint", "config_link", "decision_link"),
)
def test_replaced_attempt_or_decision_cannot_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    output, _, preflight = _patch_common_run_prerequisites(
        monkeypatch,
        tmp_path,
        dr_passed=False,
    )
    monkeypatch.setattr(
        cli,
        "_expected_model_config",
        lambda value: (
            object()
            if value is preflight
            else pytest.fail("wrong preflight")
        ),
    )
    monkeypatch.setattr(
        cli,
        "prepare_pacre_bounded_run_authorization",
        lambda *args, **kwargs: pytest.fail(
            "D_R failure reached authorization"
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_pacre_pmope_bounded_400",
        lambda *args, **kwargs: pytest.fail(
            "D_R failure reached training"
        ),
    )
    original_complete = cli._complete_run

    def replace_before_complete(**kwargs: object) -> object:
        if replacement in {"schema", "self_fingerprint", "config_link"}:
            path = output / "attempt.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if replacement == "schema":
                payload["schema_version"] = "replacement-schema"
                body = dict(payload)
                body.pop("receipt_fingerprint")
                payload["receipt_fingerprint"] = stable_fingerprint(body)
            elif replacement == "self_fingerprint":
                payload["receipt_fingerprint"] = "f" * 64
            else:
                payload["config_fingerprint"] = "f" * 64
                body = dict(payload)
                body.pop("receipt_fingerprint")
                payload["receipt_fingerprint"] = stable_fingerprint(body)
        else:
            path = output / "receipts" / "decision.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["attempt_receipt_fingerprint"] = "f" * 64
            body = dict(payload)
            body.pop("receipt_fingerprint")
            payload["receipt_fingerprint"] = stable_fingerprint(body)
        path.write_bytes(cli._json_bytes(payload))
        return original_complete(**kwargs)

    monkeypatch.setattr(cli, "_complete_run", replace_before_complete)
    with pytest.raises(
        (ValueError, RuntimeError),
        match="attempt|association|persisted artifact",
    ):
        cli.run_once()
    assert (output / ".incomplete").is_file()
    assert not (output / "COMPLETE.json").exists()
    assert (output / "FAILURE.json").is_file()


def _write_syntactic_terminal_population(
    output: Path,
    *,
    pass_path: bool,
) -> tuple[dict[str, bytes], dict[str, object]]:
    """Write a schema-valid population for byte-integrity failure tests."""

    expected_names = (
        cli._PASS_TERMINAL_FILES
        if pass_path
        else cli._COMMON_TERMINAL_FILES
    )
    (output / "receipts").mkdir(parents=True)
    (output / "checkpoints").mkdir()
    (output / ".incomplete").write_bytes(b"")
    expected: dict[str, bytes] = {}
    decision: dict[str, object] | None = None
    for relative in sorted(expected_names):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == cli._CHECKPOINT_FILE:
            raw = b"generated-safetensors"
        elif relative == "attempt.json":
            raw = cli._json_bytes(_valid_attempt())
        else:
            schema, fingerprint_field = (
                cli._TERMINAL_JSON_CONTRACTS[relative]
            )
            payload = cli._fingerprinted(
                {
                    "schema_version": schema,
                    "run_id": cli.RUN_ID,
                    "status": "generated",
                },
                field=fingerprint_field,
            )
            raw = cli._json_bytes(payload)
            if relative == "receipts/decision.json":
                decision = payload
        path.write_bytes(raw)
        expected[relative] = raw
    assert decision is not None
    return expected, decision


@pytest.mark.parametrize("pass_path", (False, True))
def test_every_terminal_artifact_replacement_prevents_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pass_path: bool,
) -> None:
    expected_names = sorted(
        cli._PASS_TERMINAL_FILES
        if pass_path
        else cli._COMMON_TERMINAL_FILES
    )
    for index, relative in enumerate(expected_names):
        output = tmp_path / f"{int(pass_path)}-{index}"
        monkeypatch.setattr(cli, "OUTPUT_PATH", output)
        implementation = (("generated.py", "a" * 64),)
        monkeypatch.setattr(
            cli,
            "_implementation_binding",
            lambda: implementation,
        )
        monkeypatch.setattr(
            cli,
            "_RUN_IMPLEMENTATION_BINDING",
            implementation,
        )
        expected, decision = _write_syntactic_terminal_population(
            output,
            pass_path=pass_path,
        )
        (output / relative).write_bytes(b"replaced")
        with pytest.raises(
            RuntimeError,
            match="persisted artifact changed",
        ):
            cli._complete_run(
                decision=decision,
                expected_artifact_count=len(expected_names),
                fields={},
                expected_attempt_receipt_fingerprint="a" * 64,
                expected_config_fingerprint="b" * 64,
                expected_authorization_fingerprint=(
                    "c" * 64 if pass_path else None
                ),
                expected_result_fingerprint=(
                    "d" * 64 if pass_path else None
                ),
                expected_artifact_bytes=expected,
            )
        assert not (output / "COMPLETE.json").exists()
        assert (output / ".incomplete").is_file()


@pytest.mark.parametrize(
    "mutation",
    (
        "missing",
        "extra",
        "nested_incomplete_marker",
        "nested_complete_marker",
    ),
)
def test_missing_or_extra_terminal_file_prevents_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    output = tmp_path / mutation
    monkeypatch.setattr(cli, "OUTPUT_PATH", output)
    implementation = (("generated.py", "a" * 64),)
    monkeypatch.setattr(
        cli,
        "_implementation_binding",
        lambda: implementation,
    )
    monkeypatch.setattr(
        cli,
        "_RUN_IMPLEMENTATION_BINDING",
        implementation,
    )
    expected, decision = _write_syntactic_terminal_population(
        output,
        pass_path=False,
    )
    if mutation == "missing":
        (output / "receipts" / "preflight.json").unlink()
    elif mutation == "extra":
        (output / "receipts" / "unexpected.json").write_bytes(b"extra")
    elif mutation == "nested_incomplete_marker":
        (output / "receipts" / ".incomplete").write_bytes(b"extra")
    else:
        (output / "receipts" / "COMPLETE.json").write_bytes(b"extra")

    with pytest.raises(RuntimeError, match="artifact names differ"):
        cli._complete_run(
            decision=decision,
            expected_artifact_count=len(cli._COMMON_TERMINAL_FILES),
            fields={},
            expected_attempt_receipt_fingerprint="a" * 64,
            expected_config_fingerprint="b" * 64,
            expected_authorization_fingerprint=None,
            expected_result_fingerprint=None,
            expected_artifact_bytes=expected,
        )
    assert not (output / "COMPLETE.json").exists()
    assert (output / ".incomplete").is_file()


def test_atomic_complete_rename_failure_preserves_incomplete_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _, preflight = _patch_common_run_prerequisites(
        monkeypatch,
        tmp_path,
        dr_passed=False,
    )
    monkeypatch.setattr(
        cli,
        "_expected_model_config",
        lambda value: (
            object()
            if value is preflight
            else pytest.fail("wrong preflight")
        ),
    )
    monkeypatch.setattr(
        cli,
        "prepare_pacre_bounded_run_authorization",
        lambda *args, **kwargs: pytest.fail(
            "D_R failure reached authorization"
        ),
    )
    monkeypatch.setattr(
        cli,
        "run_pacre_pmope_bounded_400",
        lambda *args, **kwargs: pytest.fail(
            "D_R failure reached training"
        ),
    )

    def fail_atomic_rename(source: object, target: object) -> None:
        assert Path(source) == output / ".incomplete"
        assert Path(target) == output / "COMPLETE.json"
        raise OSError("generated atomic rename failure")

    monkeypatch.setattr(cli.os, "rename", fail_atomic_rename)
    with pytest.raises(OSError, match="atomic rename failure"):
        cli.run_once()
    assert not (output / "COMPLETE.json").exists()
    assert (output / ".incomplete").is_file()
    assert (output / "FAILURE.json").is_file()


def test_pass_path_executes_each_stage_once_and_roundtrips_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, counts, preflight = _patch_common_run_prerequisites(
        monkeypatch,
        tmp_path,
        dr_passed=True,
    )
    model_config = CoverageStatePACREConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )
    monkeypatch.setattr(
        cli,
        "_expected_model_config",
        lambda value: (
            model_config
            if value is preflight
            else pytest.fail("wrong preflight")
        ),
    )
    authorization_holder: dict[str, SimpleNamespace] = {}

    def authorize_once(*args: object, **kwargs: object) -> object:
        counts["authorization"] += 1
        assert args[0] is preflight
        assert args[1] == _dataset_free_receipt()
        assert args[3].build_fingerprint == "1" * 64
        assert args[4] is model_config
        assert kwargs["run_id"] == cli.RUN_ID
        assert set(kwargs) == {"output_claim", "run_id"}
        output_claim = kwargs["output_claim"]
        claim_payload = output_claim.canonical_payload()
        authorization_payload = {
            "schema_version": "generated-authorization",
            "run_id": cli.RUN_ID,
            "attempt_fingerprint": "e" * 64,
            "training_authorized": True,
            "output_claim": claim_payload,
            "output_claim_fingerprint": (
                output_claim.claim_fingerprint
            ),
            "dataset_free_receipt_fingerprint": (
                _dataset_free_receipt()["receipt_fingerprint"]
            ),
            "D_R_gate_receipt_fingerprint": (
                args[2].receipt_fingerprint
            ),
            "preflight_fingerprint": (
                preflight.preflight_fingerprint
            ),
        }
        authorization = SimpleNamespace(
            run_id=cli.RUN_ID,
            prerequisites_passed=True,
            available=True,
            attempt_fingerprint="e" * 64,
            authorization_fingerprint=stable_fingerprint(
                authorization_payload
            ),
            canonical_payload=lambda: dict(authorization_payload),
        )
        authorization_holder["value"] = authorization
        return authorization

    monkeypatch.setattr(
        cli,
        "prepare_pacre_bounded_run_authorization",
        authorize_once,
    )
    model = (
        CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet(
            model_config
        )
    )
    with torch.no_grad():
        model.scalar_energy_weight.fill_(0.25)

    training_result_payload = {
        "seed": 42,
        "epochs": 10,
        "steps_per_epoch": 40,
        "completed_updates": 400,
    }
    training_result = SimpleNamespace(
        seed=42,
        epochs=10,
        steps_per_epoch=40,
        completed_updates=400,
        forward_calls=400,
        backward_calls=400,
        optimizer_steps=400,
        objective=cli.PACRE_PMOPE_OBJECTIVE,
        result_fingerprint=stable_fingerprint(training_result_payload),
        canonical_payload=lambda: dict(training_result_payload),
    )
    training_receipt_payload = {
        "candidate": "PACRE-v22",
        "completed_updates": 400,
    }
    training_receipt = SimpleNamespace(
        receipt_fingerprint=stable_fingerprint(
            training_receipt_payload
        ),
        canonical_payload=lambda: dict(training_receipt_payload),
    )
    bundle = SimpleNamespace(
        model=model,
        training_result=training_result,
        receipt=training_receipt,
        bundle_fingerprint="9" * 64,
    )
    diagnostic_payload = {
        "split": "D_R",
        "threshold": 0.0,
        "checkpoint_fingerprint": cli.module_state_fingerprint(model),
    }
    diagnostic = SimpleNamespace(
        result_fingerprint=stable_fingerprint(diagnostic_payload),
        canonical_payload=lambda: dict(diagnostic_payload),
    )
    pacre_decision_payload = {
        "run_id": cli.RUN_ID,
        "bounded_gate_passed": True,
    }
    decision = SimpleNamespace(
        decision_fingerprint=stable_fingerprint(
            pacre_decision_payload
        ),
        canonical_payload=lambda: dict(pacre_decision_payload),
    )
    result_holder: dict[str, SimpleNamespace] = {}

    def run_once(
        value: object,
        config: object,
        **kwargs: object,
    ) -> object:
        counts["runner"] += 1
        authorization = authorization_holder["value"]
        assert value is authorization
        assert config is model_config
        assert kwargs == {
            "run_id": cli.RUN_ID,
            "device": "cuda:0",
        }
        bounded_payload = {
            "schema_version": "generated-result",
            "run_id": cli.RUN_ID,
            "authorization_fingerprint": (
                authorization.authorization_fingerprint
            ),
            "training": {
                "receipt": training_receipt_payload,
                "result": training_result_payload,
            },
            "diagnostic": diagnostic_payload,
            "decision": pacre_decision_payload,
            "bounded_gate_passed": True,
        }
        bounded_result = SimpleNamespace(
            run_id=cli.RUN_ID,
            training_invocations=1,
            zero_level_evaluation_invocations=1,
            decision_invocations=1,
            training=bundle,
            diagnostic=diagnostic,
            decision=decision,
            result_fingerprint=stable_fingerprint(bounded_payload),
            bounded_gate_passed=True,
            failed_checks=(),
            formal800_eligible=True,
            canonical_payload=lambda: dict(bounded_payload),
        )
        result_holder["value"] = bounded_result
        return bounded_result

    monkeypatch.setattr(
        cli,
        "run_pacre_pmope_bounded_400",
        run_once,
    )
    result = cli.run_once()
    authorization = authorization_holder["value"]
    bounded_result = result_holder["value"]

    assert counts == {
        "dataset_free": 1,
        "output_claim": 1,
        "cuda_runtime": 1,
        "real_inputs": 1,
        "population": 1,
        "preflight": 1,
        "dr_gate": 1,
        "authorization": 1,
        "runner": 1,
    }
    assert result["bounded_gate_passed"] is True
    assert result["formal800_eligible"] is True
    assert not (output / ".incomplete").exists()
    assert not (output / "FAILURE.json").exists()
    complete = json.loads(
        (output / "COMPLETE.json").read_text(encoding="utf-8")
    )
    assert complete["artifact_file_count"] == 13
    assert complete["bounded_runner_invocations"] == 1
    assert complete["checkpoint_count"] == 1
    attempt = json.loads(
        (output / "attempt.json").read_text(encoding="utf-8")
    )
    assert complete["attempt_receipt_fingerprint"] == (
        attempt["receipt_fingerprint"]
    )
    assert complete["authorization_attempt_fingerprint"] == (
        authorization.attempt_fingerprint
    )
    assert complete["authorization_fingerprint"] == (
        authorization.authorization_fingerprint
    )
    assert complete["result_fingerprint"] == (
        bounded_result.result_fingerprint
    )

    checkpoint_path = (
        output
        / "checkpoints"
        / f"{cli.PACRE_PMOPE_OBJECTIVE}.safetensors"
    )
    checkpoint_receipt_path = (
        output
        / "checkpoints"
        / f"{cli.PACRE_PMOPE_OBJECTIVE}.checkpoint.json"
    )
    assert checkpoint_path.is_file()
    assert checkpoint_receipt_path.is_file()
    from safetensors.torch import load_file

    loaded = load_file(str(checkpoint_path), device="cpu")
    expected = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
    }
    assert set(loaded) == set(expected)
    assert all(
        torch.equal(loaded[name], expected[name]) for name in expected
    )
    checkpoint_receipt = json.loads(
        checkpoint_receipt_path.read_text(encoding="utf-8")
    )
    assert checkpoint_receipt["model_class"] == (
        "CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet"
    )
    assert checkpoint_receipt["weights_only_roundtrip_verified"] is True
    assert checkpoint_receipt["tensor_only_state_dict"] is True


def test_checkpoint_rejects_wrong_model_and_existing_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "_ROOT", tmp_path)
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    with pytest.raises(TypeError, match="exact PACRE"):
        cli._write_checkpoint_new(
            directory,
            model=object(),  # type: ignore[arg-type]
        )

    model = (
        CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet(
            CoverageStatePACREConfig(
                feature_channels=2,
                feature_stride=2,
                width=4,
            )
        )
    )
    cli._write_checkpoint_new(directory, model=model)
    with pytest.raises(FileExistsError):
        cli._write_checkpoint_new(directory, model=model)
