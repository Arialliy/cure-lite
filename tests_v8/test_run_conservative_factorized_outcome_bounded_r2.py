from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Mapping

import pytest

from cure_lite.cache.schema import stable_fingerprint
from tests.test_paired_outcome_bounded import _inputs
from tools import run_conservative_factorized_outcome_bounded_r2 as runner


class _ReceiptObject:
    def __init__(self, payload: Mapping[str, object]) -> None:
        self.payload = dict(payload)

    def canonical_receipt(self) -> dict[str, object]:
        return deepcopy(self.payload)


class _Bundle:
    def __init__(self) -> None:
        self.verify_calls = 0

    def verify_unchanged(self) -> None:
        self.verify_calls += 1


def _args(output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config=runner.ROOT / runner.CONFIG_REPO_PATH,
        device=runner.FROZEN_DEVICE,
        output=output,
    )


def _fake_real_inputs() -> runner.v1._FrozenRealInputs:
    population, factual_schedule, schedule, materializer = _inputs()
    source_path = runner.ROOT / runner.V1_CONFIG_REPO_PATH
    source_config = json.loads(source_path.read_text(encoding="utf-8"))
    return runner.v1._FrozenRealInputs(
        source_config=source_config,
        source_config_path=source_path,
        bundle=_Bundle(),
        immutable={},
        population=population,
        factual_schedule=factual_schedule,
        materializer=materializer,
        outcome_schedule=schedule,
        pair_catalog_fingerprint=runner.v1.PAIR_CATALOG_FINGERPRINT,
        prepared_catalog_fingerprint=(
            runner.v1.PREPARED_CATALOG_FINGERPRINT
        ),
    )


def _core_result(*, model_pass: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "structural_execution_pass": True,
        "computational_model_code_gate_pass": model_pass,
        "decision": (
            "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_PASS"
            if model_pass
            else "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_FAIL"
        ),
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


def _fake_static_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[dict[str, object], Path, dict[str, object], Path]:
    implementation = runner._implementation_binding()
    closure = runner._fingerprinted(
        {
            "schema_version": runner.CLOSURE_SCHEMA,
            "method_id": runner.METHOD_ID,
            "correction_id": runner.CORRECTION_ID,
            "decision": (
                "CC_SEA_V8_R2_VERIFIER_CORRECTION_IMPLEMENTATION_PASS"
            ),
            "runtime_implementation_binding": runner._fingerprinted(
                implementation
            ),
            "test_evidence": {
                "all_required_tests_passed": True,
                "direct_core_to_real_verifier_passed": True,
                "r1_strict_loader_regression_passed": True,
            },
            "boundary": {
                "D_R_payload_accessed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
                "r2_execution_authorized": False,
                "formal_800_authorized": False,
            },
        }
    )
    closure_path = tmp_path / "closure.json"
    closure_path.write_text(
        json.dumps(closure, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    authorization = runner._fingerprinted(
        {
            "schema_version": runner.AUTHORIZATION_SCHEMA,
            "method_id": runner.METHOD_ID,
            "correction_id": runner.CORRECTION_ID,
            "decision": "CC_SEA_V8_R2_ONE_CORRECTIVE_D_R_RUN_AUTHORIZED",
            "config_fingerprint": runner.CONFIG_FINGERPRINT,
            "implementation_fingerprint": stable_fingerprint(
                implementation
            ),
            "closure_fingerprint": closure["receipt_fingerprint"],
            "closure_file_sha256": runner.file_sha256(closure_path),
            "authorization": {
                "exact_r2_run_count": 1,
                "output_repo_path": runner.OUTPUT_REPO_PATH,
                "device": runner.FROZEN_DEVICE,
                "create_only": True,
                "resume_allowed": False,
                "automatic_retry_allowed": False,
                "D_V_access_allowed": False,
                "D_T_access_allowed": False,
                "formal_800_allowed": False,
            },
            "boundary": {
                "r1_remains_consumed_and_immutable": True,
                "r2_is_not_an_additional_seed": True,
                "r2_pass_or_nonpass_must_be_frozen": True,
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
            },
        }
    )
    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text(
        json.dumps(authorization, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner,
        "_load_closure",
        lambda _implementation: (closure, closure_path),
    )
    monkeypatch.setattr(
        runner,
        "_load_authorization",
        lambda *_args, **_kwargs: (
            authorization,
            authorization_path,
        ),
    )
    return closure, closure_path, authorization, authorization_path


def test_missing_closure_precedes_output_claim_and_real_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "must-not-exist"
    loader_calls = 0

    def guarded_loader(
        _config: Mapping[str, object],
    ) -> runner.v1._FrozenRealInputs:
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("real loader must not run")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(runner.v1, "_load_frozen_real_inputs", guarded_loader)
    monkeypatch.setattr(runner, "_validate_output_target", lambda _p: output)
    monkeypatch.setattr(
        runner,
        "_load_closure",
        lambda _implementation: (_ for _ in ()).throw(
            FileNotFoundError("mock r2 closure is missing")
        ),
    )
    with pytest.raises(FileNotFoundError):
        runner.run(_args(output))
    assert loader_calls == 0
    assert not output.exists()


@pytest.mark.parametrize(
    ("model_pass", "expected"),
    (
        (True, "CC_SEA_V8_R2_BOUNDED_MODEL_CODE_GATE_PASS"),
        (False, "CC_SEA_V8_R2_BOUNDED_MODEL_CODE_GATE_FAIL"),
    ),
)
def test_versioned_result_publication_is_create_only_and_reloadable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_pass: bool,
    expected: str,
) -> None:
    output = tmp_path / ("pass" if model_pass else "nonpass")
    _fake_static_chain(monkeypatch, tmp_path)
    fake_inputs = _fake_real_inputs()
    core = _core_result(model_pass=model_pass)
    verified: list[dict[str, object]] = []

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(runner, "_validate_output_target", lambda _p: output)
    monkeypatch.setattr(
        runner.v1,
        "_load_frozen_real_inputs",
        lambda _config: fake_inputs,
    )
    monkeypatch.setattr(
        runner.v1,
        "execute_conservative_factorized_outcome_bounded",
        lambda *_args, **_kwargs: deepcopy(core),
    )

    def verify(value: Mapping[str, object]) -> None:
        unsigned = dict(value)
        observed = unsigned.pop("result_fingerprint")
        assert observed == stable_fingerprint(unsigned)
        verified.append(dict(value))

    monkeypatch.setattr(
        runner,
        "verify_conservative_factorized_core_result",
        verify,
    )

    published = runner.run(_args(output))
    assert published["decision"] == expected
    assert published["bounded_model_code_gate_pass"] is model_pass
    assert published["r2_run_claim_consumed"] is True
    assert published["r1_remains_immutable"] is True
    assert len(verified) >= 2
    assert not (output / runner.INCOMPLETE).exists()
    assert (output / "receipts" / "result.json").is_file()

    strict = runner.load_correction_bounded_artifact(output)
    assert strict.decision == expected
    with pytest.raises(FileExistsError):
        runner.run(_args(output))


def test_execution_error_is_frozen_not_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "failure"
    _fake_static_chain(monkeypatch, tmp_path)
    fake_inputs = _fake_real_inputs()
    execute_calls = 0

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal execute_calls
        execute_calls += 1
        raise RuntimeError("mock executor error")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(runner, "_validate_output_target", lambda _p: output)
    monkeypatch.setattr(
        runner.v1,
        "_load_frozen_real_inputs",
        lambda _config: fake_inputs,
    )
    monkeypatch.setattr(
        runner.v1,
        "execute_conservative_factorized_outcome_bounded",
        fail,
    )

    published = runner.run(_args(output))
    assert execute_calls == 1
    assert published["decision"] == "CC_SEA_V8_R2_BOUNDED_EXECUTION_ERROR"
    assert (output / "receipts" / "failure.json").is_file()
    with pytest.raises(FileExistsError):
        runner.run(_args(output))
    assert execute_calls == 1
