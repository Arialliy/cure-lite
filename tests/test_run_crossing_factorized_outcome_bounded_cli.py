from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.config import LossConfig
from cure_lite.crossing_factorized_config import (
    CrossingFactorizedDecoderConfig,
)
from tests.test_crossing_factorized_outcome_bounded import (
    _crossing_budget,
    _inputs,
)
from tools import run_crossing_factorized_outcome_bounded as runner


def _frozen_config_path() -> Path:
    return (runner._ROOT / runner.CONFIG_REPO_PATH).resolve()


def _args(output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config=_frozen_config_path(),
        device=runner.FROZEN_DEVICE,
        output=output,
    )


def test_frozen_static_chain_and_unsigned_runtime_validate_without_D_R() -> None:
    config = runner._load_config(_frozen_config_path())
    runner._load_proposal(config)
    runner._load_toy_closure(config)
    runner._load_implementation_proposal(config)
    runner._load_frozen_dry_evidence()
    unsigned = runner._implementation_binding()
    runner._verify_implementation_files(unsigned)

    assert "receipt_fingerprint" not in unsigned
    assert len(unsigned["v4_runtime_files"]) == 45
    assert len(unsigned["v7_runtime_files"]) == 5
    assert len(unsigned["all_runtime_files"]) == 50


def test_signed_runtime_is_verified_then_compared_as_unsigned() -> None:
    unsigned = runner._implementation_binding()
    signed = runner._fingerprinted(unsigned)

    assert (
        runner._verified_unsigned_receipt(
            signed,
            name="test signed runtime",
        )
        == unsigned
    )
    assert stable_fingerprint(unsigned) == signed["receipt_fingerprint"]

    corrupted = dict(signed)
    corrupted["receipt_fingerprint"] = "0" * 64
    with pytest.raises(RuntimeError, match="fingerprint"):
        runner._verified_unsigned_receipt(
            corrupted,
            name="corrupt signed runtime",
        )


def test_authorization_requires_post_signing_closure_test_evidence(
    tmp_path: Path,
) -> None:
    closure, closure_path, _, _, _ = _fake_closure_and_authorization(
        tmp_path
    )
    implementation = runner._implementation_binding()
    receipt = runner._fingerprinted(
        {
            "schema_version": runner.AUTHORIZATION_SCHEMA,
            "method_id": runner.CROSSING_FACTORIZED_OUTCOME_METHOD_ID,
            "split": "D_R",
            "phase_status": (
                "FROZEN_SINGLE_REAL_D_R_RUN_AUTHORIZATION"
            ),
            "decision": (
                "CR_LVEC_V7_ONE_REAL_D_R_BOUNDED_RUN_AUTHORIZED"
            ),
            "authorization": {
                "real_D_R_bounded_execution": True,
                "exact_run_count": 1,
                "device": runner.FROZEN_DEVICE,
                "output_repo_path": runner.OUTPUT_REPO_PATH,
                "resume_allowed": False,
                "automatic_retry_allowed": False,
                "D_V_access_allowed": False,
                "D_T_access_allowed": False,
                "formal_800_allowed": False,
            },
            "bounded_config_binding": {
                "repo_path": runner.CONFIG_REPO_PATH,
                "file_sha256": runner.CONFIG_FILE_SHA256,
                "config_fingerprint": runner.CONFIG_FINGERPRINT,
            },
            "implementation_closure_binding": {
                "repo_path": runner.IMPLEMENTATION_CLOSURE_REPO_PATH,
                "file_sha256": file_sha256(closure_path),
                "receipt_fingerprint": closure[
                    "receipt_fingerprint"
                ],
            },
            "runtime_implementation_binding": {
                "implementation_fingerprint": stable_fingerprint(
                    implementation
                ),
                "all_runtime_files": implementation[
                    "all_runtime_files"
                ],
            },
            "execution_control_binding": {
                "gpu_index": 0,
                "pause_temperature_celsius": 82,
                "resume_temperature_celsius": 75,
                "wrapper_repo_path": (
                    runner.TEMPERATURE_WRAPPER_REPO_PATH
                ),
                "wrapper_file_sha256": (
                    runner.TEMPERATURE_WRAPPER_FILE_SHA256
                ),
                "wrapped_command": runner._expected_temperature_command(),
            },
            "post_signing_closure_test_evidence": {
                "evidence_stage": "post_signing",
                "closure_receipt_present_during_execution": True,
                "repo_path": runner._CLOSURE_STATIC_TEST_REPO_PATH,
                "file_sha256": file_sha256(
                    runner._ROOT
                    / runner._CLOSURE_STATIC_TEST_REPO_PATH
                ),
                "command": (
                    runner._expected_post_signing_closure_test_command()
                ),
                "exit_code": 0,
                "passed_count": 1,
                "failed_count": 0,
                "skipped_count": 0,
                "deselected_count": 0,
                "selected_count": 1,
                "collected_count": 1,
                "closure_repo_path": (
                    runner.IMPLEMENTATION_CLOSURE_REPO_PATH
                ),
                "closure_file_sha256": file_sha256(closure_path),
                "closure_receipt_fingerprint": closure[
                    "receipt_fingerprint"
                ],
                "D_R_payload_accessed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            },
        }
    )

    runner._validate_authorization_payload(
        receipt,
        closure=closure,
        closure_path=closure_path,
        implementation_unsigned=implementation,
    )

    invalid = dict(receipt)
    invalid.pop("receipt_fingerprint")
    invalid.pop("post_signing_closure_test_evidence")
    invalid = runner._fingerprinted(invalid)
    with pytest.raises(RuntimeError, match="authorization"):
        runner._validate_authorization_payload(
            invalid,
            closure=closure,
            closure_path=closure_path,
            implementation_unsigned=implementation,
        )


@pytest.mark.parametrize(
    "closure_error",
    (
        FileNotFoundError("missing implementation closure"),
        RuntimeError("implementation closure hash mismatch"),
    ),
)
def test_closure_failure_precedes_D_R_loader_and_output_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    closure_error: Exception,
) -> None:
    output = tmp_path / "v7-r1"
    loader_calls = 0
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        runner,
        "_validate_output_target",
        lambda _path: output,
    )

    def reject_closure(*_args: object, **_kwargs: object):
        raise closure_error

    def forbidden_loader(*_args: object, **_kwargs: object):
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("D_R loader must remain unreachable")

    monkeypatch.setattr(
        runner,
        "_load_implementation_closure",
        reject_closure,
    )
    monkeypatch.setattr(
        runner,
        "_load_frozen_real_inputs",
        forbidden_loader,
    )

    with pytest.raises(type(closure_error), match=str(closure_error)):
        runner.run(_args(output))
    assert loader_calls == 0
    assert not output.exists()


def test_device_requires_frozen_wrapper_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES=0"):
        runner._validate_device("cuda:0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES=0"):
        runner._validate_device("cuda:0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert runner._validate_device("cuda:0") == "cuda:0"
    with pytest.raises(ValueError, match="cuda:0"):
        runner._validate_device("cuda:1")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _fake_closure_and_authorization(
    tmp_path: Path,
) -> tuple[
    dict[str, object],
    Path,
    dict[str, object],
    dict[str, object],
    Path,
]:
    unsigned = runner._implementation_binding()
    signed = runner._fingerprinted(unsigned)
    closure = runner._fingerprinted(
        {
            "schema_version": runner.IMPLEMENTATION_CLOSURE_SCHEMA,
            "method_id": runner.CROSSING_FACTORIZED_OUTCOME_METHOD_ID,
            "phase_status": "FROZEN_BOUNDED_IMPLEMENTATION_PASS",
            "decision": "CR_LVEC_V7_BOUNDED_IMPLEMENTATION_GATE_PASS",
            "runtime_implementation_binding": signed,
        }
    )
    closure_path = tmp_path / "closure.json"
    _write_json(closure_path, closure)
    authorization = runner._fingerprinted(
        {
            "schema_version": runner.AUTHORIZATION_SCHEMA,
            "method_id": runner.CROSSING_FACTORIZED_OUTCOME_METHOD_ID,
            "split": "D_R",
            "phase_status": "FROZEN_SINGLE_REAL_D_R_RUN_AUTHORIZATION",
            "decision": "CR_LVEC_V7_ONE_REAL_D_R_BOUNDED_RUN_AUTHORIZED",
        }
    )
    authorization_path = tmp_path / "authorization.json"
    _write_json(authorization_path, authorization)
    return closure, closure_path, signed, authorization, authorization_path


def test_authorization_failure_precedes_D_R_loader_and_output_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "v7-r1"
    closure, closure_path, signed, _, _ = (
        _fake_closure_and_authorization(tmp_path)
    )
    loader_calls = 0
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        runner,
        "_validate_output_target",
        lambda _path: output,
    )
    monkeypatch.setattr(
        runner,
        "_load_implementation_closure",
        lambda *_args, **_kwargs: (closure, closure_path, signed),
    )

    def reject_authorization(*_args: object, **_kwargs: object):
        raise RuntimeError("authorization fingerprint mismatch")

    def forbidden_loader(*_args: object, **_kwargs: object):
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("D_R loader must remain unreachable")

    monkeypatch.setattr(runner, "_load_authorization", reject_authorization)
    monkeypatch.setattr(
        runner,
        "_load_frozen_real_inputs",
        forbidden_loader,
    )

    with pytest.raises(RuntimeError, match="authorization fingerprint"):
        runner.run(_args(output))
    assert loader_calls == 0
    assert not output.exists()


class _ReceiptObject:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def canonical_receipt(self) -> dict[str, object]:
        return deepcopy(self._payload)


class _UnchangedBundle:
    def __init__(self) -> None:
        self.verify_calls = 0

    def verify_unchanged(self) -> None:
        self.verify_calls += 1


def _fake_real_inputs() -> runner._FrozenRealInputs:
    source_path = (
        runner._ROOT
        / "protocols/IRSTD-1K/paired_bounded_learnability_v1/config.json"
    ).resolve()
    population = _ReceiptObject(
        {
            "schema_version": "test-population-v1",
            "population_fingerprint": "a" * 64,
        }
    )
    factual = _ReceiptObject(
        {
            "schema_version": "test-factual-schedule-v1",
            "schedule_fingerprint": "b" * 64,
        }
    )
    materializer = _ReceiptObject(
        {
            "schema_version": "test-materializer-v1",
            "materializer_fingerprint": "c" * 64,
        }
    )
    schedule = _ReceiptObject(
        {
            "schema_version": "test-outcome-schedule-v1",
            "schedule_fingerprint": "d" * 64,
        }
    )
    return runner._FrozenRealInputs(
        source_config={"config_fingerprint": "e" * 64},
        source_config_path=source_path,
        bundle=_UnchangedBundle(),
        immutable={},
        population=population,
        factual_schedule=factual,
        materializer=materializer,
        outcome_schedule=schedule,
        pair_catalog_fingerprint=runner.PAIR_CATALOG_FINGERPRINT,
        prepared_catalog_fingerprint=runner.PREPARED_CATALOG_FINGERPRINT,
    )


def _core_result(*, model_pass: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": (
            runner.CROSSING_FACTORIZED_OUTCOME_BOUNDED_SCHEMA
        ),
        "method_id": runner.CROSSING_FACTORIZED_OUTCOME_METHOD_ID,
        "execution_status": "completed",
        "device": runner.FROZEN_DEVICE,
        "decision": (
            "CR_LVEC_BOUNDED_MODEL_CODE_GATE_PASS"
            if model_pass
            else "CR_LVEC_BOUNDED_MODEL_CODE_GATE_FAIL"
        ),
        "structural_execution_pass": True,
        "computational_model_code_gate_pass": model_pass,
        "population_fingerprint": runner.ANCHOR_POPULATION_FINGERPRINT,
        "factual_schedule_fingerprint": (
            runner.FACTUAL_SCHEDULE_FINGERPRINT
        ),
        "materializer_fingerprint": runner.MATERIALIZER_FINGERPRINT,
        "outcome_schedule_fingerprint": runner.OUTCOME_SCHEDULE_FINGERPRINT,
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


def _install_authorized_publication_mocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    core_result: dict[str, object] | Exception,
) -> tuple[Path, list[Path], list[dict[str, object]]]:
    output = tmp_path / "v7-r1"
    closure, closure_path, signed, authorization, authorization_path = (
        _fake_closure_and_authorization(tmp_path)
    )
    loader_observations: list[Path] = []
    verified_cores: list[dict[str, object]] = []
    fake_inputs = _fake_real_inputs()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        runner,
        "_validate_output_target",
        lambda _path: output
        if not output.exists()
        else (_ for _ in ()).throw(FileExistsError("already consumed")),
    )
    monkeypatch.setattr(
        runner,
        "_load_implementation_closure",
        lambda *_args, **_kwargs: (closure, closure_path, signed),
    )
    monkeypatch.setattr(
        runner,
        "_load_authorization",
        lambda *_args, **_kwargs: (authorization, authorization_path),
    )

    def load_after_claim(_config: object) -> runner._FrozenRealInputs:
        loader_observations.append(output)
        assert output.is_dir()
        assert (output / runner._INCOMPLETE).is_file()
        assert (output / "receipts/run_claim.json").is_file()
        assert (output / "receipts/authorization_binding.json").is_file()
        return fake_inputs

    def execute(*_args: object, **_kwargs: object) -> dict[str, object]:
        if isinstance(core_result, Exception):
            raise core_result
        return deepcopy(core_result)

    def verify_core(value: object) -> None:
        assert isinstance(value, dict)
        assert "receipt_fingerprint" not in value
        unsigned = dict(value)
        fingerprint = unsigned.pop("result_fingerprint")
        assert fingerprint == stable_fingerprint(unsigned)
        assert value == core_result
        verified_cores.append(deepcopy(value))

    monkeypatch.setattr(
        runner,
        "_load_frozen_real_inputs",
        load_after_claim,
    )
    monkeypatch.setattr(
        runner,
        "_verify_published_input_receipts",
        lambda _payloads: {
            "population_fingerprint": runner.ANCHOR_POPULATION_FINGERPRINT,
            "factual_schedule_fingerprint": (
                runner.FACTUAL_SCHEDULE_FINGERPRINT
            ),
            "materializer_fingerprint": runner.MATERIALIZER_FINGERPRINT,
            "outcome_schedule_fingerprint": (
                runner.OUTCOME_SCHEDULE_FINGERPRINT
            ),
            "all_pair_inputs_fingerprint": (
                runner.ALL_PAIR_INPUTS_FINGERPRINT
            ),
            "gt_union_population_fingerprint": (
                runner.GT_UNION_POPULATION_FINGERPRINT
            ),
            "outcome_sequence_fingerprint": (
                runner.OUTCOME_SEQUENCE_FINGERPRINT
            ),
        },
    )
    monkeypatch.setattr(
        runner,
        "execute_crossing_factorized_outcome_bounded",
        execute,
    )
    monkeypatch.setattr(runner, "_verify_core_result", verify_core)
    return output, loader_observations, verified_cores


@pytest.mark.parametrize(
    ("model_pass", "expected"),
    (
        (True, "BOUNDED_MODEL_CODE_GATE_PASS"),
        (False, "BOUNDED_MODEL_CODE_GATE_FAIL"),
    ),
)
def test_completed_pass_and_nonpass_publish_and_strictly_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_pass: bool,
    expected: str,
) -> None:
    core = _core_result(model_pass=model_pass)
    output, loader_observations, verified_cores = (
        _install_authorized_publication_mocks(
            monkeypatch,
            tmp_path,
            core_result=core,
        )
    )

    published = runner.run(_args(output))

    assert published["decision"] == expected
    assert len(loader_observations) == 1
    assert len(verified_cores) == 2
    assert all("receipt_fingerprint" not in row for row in verified_cores)
    assert not (output / runner._INCOMPLETE).exists()
    assert (output / "receipts/result.json").is_file()
    assert not (output / "receipts/failure.json").exists()
    assert (
        runner.load_crossing_factorized_outcome_bounded_artifact(
            output
        ).decision
        == expected
    )
    with pytest.raises(FileExistsError, match="already consumed"):
        runner.run(_args(output))


def test_execution_error_consumes_run_and_publishes_reloadable_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output, loader_observations, verified_cores = (
        _install_authorized_publication_mocks(
            monkeypatch,
            tmp_path,
            core_result=RuntimeError("synthetic execution failure"),
        )
    )

    published = runner.run(_args(output))

    assert published["decision"] == "STRUCTURAL_EXECUTION_ERROR"
    assert len(loader_observations) == 1
    assert verified_cores == []
    assert not (output / runner._INCOMPLETE).exists()
    assert (output / "receipts/failure.json").is_file()
    assert not (output / "receipts/result.json").exists()
    failure = json.loads(
        (output / "receipts/failure.json").read_text(encoding="utf-8")
    )
    assert failure["phase"] == "BOUNDED_EXECUTION"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["message"] == "synthetic execution failure"
    assert failure["post_attempt_verification_passed"] is True
    assert (
        runner.load_crossing_factorized_outcome_bounded_artifact(
            output
        ).decision
        == "STRUCTURAL_EXECUTION_ERROR"
    )
    with pytest.raises(FileExistsError, match="already consumed"):
        runner.run(_args(output))


def test_D_R_reconstruction_error_is_one_claimed_reloadable_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output, _, _ = _install_authorized_publication_mocks(
        monkeypatch,
        tmp_path,
        core_result=_core_result(model_pass=True),
    )
    loader_calls = 0

    def fail_after_claim(_config: object) -> runner._FrozenRealInputs:
        nonlocal loader_calls
        loader_calls += 1
        assert output.is_dir()
        assert (output / runner._INCOMPLETE).is_file()
        assert (output / "receipts/run_claim.json").is_file()
        assert (output / "receipts/authorization_binding.json").is_file()
        raise RuntimeError("synthetic D_R reconstruction failure")

    monkeypatch.setattr(
        runner,
        "_load_frozen_real_inputs",
        fail_after_claim,
    )

    published = runner.run(_args(output))

    assert published["decision"] == "STRUCTURAL_EXECUTION_ERROR"
    assert loader_calls == 1
    assert not (output / runner._INCOMPLETE).exists()
    receipt_names = {
        path.name for path in (output / "receipts").iterdir()
    }
    assert receipt_names == (
        runner._PRE_RUN_RECEIPTS | {"decision.json", "failure.json"}
    )
    failure = json.loads(
        (output / "receipts/failure.json").read_text(encoding="utf-8")
    )
    assert failure["phase"] == "D_R_RECONSTRUCTION"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["message"] == "synthetic D_R reconstruction failure"
    assert failure["real_D_R_run_claim_consumed"] is True
    assert failure["post_attempt_verification_passed"] is True
    assert (
        runner.load_crossing_factorized_outcome_bounded_artifact(
            output
        ).decision
        == "STRUCTURAL_EXECUTION_ERROR"
    )
    with pytest.raises(FileExistsError, match="already consumed"):
        runner.run(_args(output))
    assert loader_calls == 1


def test_strict_loader_rejects_resigned_failure_decision_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output, _, _ = _install_authorized_publication_mocks(
        monkeypatch,
        tmp_path,
        core_result=RuntimeError("synthetic execution failure"),
    )
    runner.run(_args(output))

    decision_path = output / "receipts/decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision.pop("receipt_fingerprint")
    assert isinstance(decision["failure"], dict)
    decision["failure"]["message"] = "resigned mismatch"
    decision = runner._fingerprinted(decision)
    decision_path.write_text(
        json.dumps(
            decision,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    complete_path = output / "COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete.pop("complete_fingerprint")
    complete["decision_fingerprint"] = decision["receipt_fingerprint"]
    complete["artifact_files"] = runner._artifact_hashes(output)
    complete = runner._fingerprinted(
        complete,
        field="complete_fingerprint",
    )
    complete_path.write_text(
        json.dumps(
            complete,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="decision binding"):
        runner.load_crossing_factorized_outcome_bounded_artifact(output)


def _resign_core(payload: dict[str, object]) -> dict[str, object]:
    output = deepcopy(payload)
    output.pop("result_fingerprint", None)
    output["result_fingerprint"] = stable_fingerprint(output)
    return output


def test_full_400_structural_nonpass_is_a_result_not_execution_error() -> None:
    population, factual, schedule, materializer = _inputs()
    result = runner.execute_crossing_factorized_outcome_bounded(
        population,
        factual,
        schedule,
        materializer,
        CrossingFactorizedDecoderConfig(3, 2),
        LossConfig(),
        _crossing_budget(),
        device="cpu",
        evaluation_chunk_size=32,
    )
    assert result["structural_execution_pass"] is True
    assert result["computational_gates"]["all_pass"] is False
    assert result["computational_model_code_gate_pass"] is False

    result["device"] = runner.FROZEN_DEVICE
    result["population_fingerprint"] = (
        runner.ANCHOR_POPULATION_FINGERPRINT
    )
    result["factual_schedule_fingerprint"] = (
        runner.FACTUAL_SCHEDULE_FINGERPRINT
    )
    result["materializer_fingerprint"] = runner.MATERIALIZER_FINGERPRINT
    result["outcome_schedule_fingerprint"] = (
        runner.OUTCOME_SCHEDULE_FINGERPRINT
    )
    result["decoder_config"] = vars(
        CrossingFactorizedDecoderConfig(64, 4)
    )
    result["parameters"]["trainable_parameter_count"] = 4385
    result["parameters"]["expected_parameter_count"] = 4385
    full_nonpass = _resign_core(result)
    runner._verify_core_result(full_nonpass)

    structural_nonpass = deepcopy(full_nonpass)
    initial_fingerprint = structural_nonpass["parameters"][
        "initial_decoder_fingerprint"
    ]
    structural_nonpass["parameters"][
        "final_decoder_fingerprint"
    ] = initial_fingerprint
    structural_nonpass["structural_checks"][
        "decoder_parameters_changed"
    ] = False
    structural_nonpass["structural_execution_pass"] = False
    structural_nonpass["computational_model_code_gate_pass"] = False
    structural_nonpass["decision"] = "CR_LVEC_STRUCTURAL_EXECUTION_FAIL"
    structural_nonpass["interpretation"][
        "eligible_for_frozen_review"
    ] = False
    structural_nonpass = _resign_core(structural_nonpass)

    runner._verify_core_result(structural_nonpass)

    invalid_zero_stop = deepcopy(full_nonpass)
    operator = invalid_zero_stop["pretraining_structural_audit"][
        "operator_contract"
    ]
    failed_name = runner.CROSSING_OPERATOR_STRUCTURAL_CHECKS[0]
    operator["checks"][failed_name] = False
    operator["all_pass"] = False
    invalid_zero_stop["pretraining_structural_audit"]["checks"][
        failed_name
    ] = False
    invalid_zero_stop["pretraining_structural_audit"][
        "all_pass"
    ] = False
    invalid_zero_stop["structural_checks"] = deepcopy(
        invalid_zero_stop["pretraining_structural_audit"]["checks"]
    )
    invalid_zero_stop["structural_execution_pass"] = False
    invalid_zero_stop["computational_model_code_gate_pass"] = False
    invalid_zero_stop["decision"] = "CR_LVEC_STRUCTURAL_EXECUTION_FAIL"
    invalid_zero_stop["interpretation"][
        "eligible_for_frozen_review"
    ] = False
    invalid_zero_stop = _resign_core(invalid_zero_stop)
    with pytest.raises(RuntimeError, match="zero-update|stop rule"):
        runner._verify_core_result(invalid_zero_stop)
