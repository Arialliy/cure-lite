from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Mapping

import pytest

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from tools import run_conservative_factorized_outcome_bounded as runner


_ROOT = Path(__file__).resolve().parents[1]


def _args(output: Path, *, device: str = runner.FROZEN_DEVICE) -> argparse.Namespace:
    return argparse.Namespace(
        config=(_ROOT / runner.CONFIG_REPO_PATH).resolve(),
        device=device,
        output=output,
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
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


@pytest.fixture
def loader_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int]:
    """Install a fail-fast replacement before any runner invocation."""

    state = {"calls": 0}

    def forbidden_real_loader(*_args: object, **_kwargs: object) -> object:
        state["calls"] += 1
        raise AssertionError("the real D_R loader must remain unreachable")

    monkeypatch.setattr(
        runner,
        "_load_frozen_real_inputs",
        forbidden_real_loader,
    )
    return state


def _fake_closure_and_authorization(
    tmp_path: Path,
) -> tuple[
    dict[str, object],
    Path,
    dict[str, object],
    dict[str, object],
    Path,
]:
    implementation_unsigned = runner._implementation_binding()
    runtime_signed = runner._fingerprinted(implementation_unsigned)
    closure = runner._fingerprinted(
        {
            "schema_version": runner.IMPLEMENTATION_CLOSURE_SCHEMA,
            "method_id": runner.CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID,
            "phase_status": "FROZEN_BOUNDED_IMPLEMENTATION_PASS",
            "decision": "CC_SEA_V8_BOUNDED_IMPLEMENTATION_GATE_PASS",
            "runtime_implementation_binding": runtime_signed,
            "boundary": {
                "D_R_payload_accessed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
                "real_D_R_bounded_execution_authorized": False,
            },
        }
    )
    closure_path = tmp_path / "mock-implementation-closure.json"
    _write_json(closure_path, closure)

    authorization = runner._fingerprinted(
        {
            "schema_version": runner.AUTHORIZATION_SCHEMA,
            "method_id": runner.CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID,
            "split": "D_R",
            "phase_status": "FROZEN_SINGLE_REAL_D_R_RUN_AUTHORIZATION",
            "decision": "CC_SEA_V8_ONE_REAL_D_R_BOUNDED_RUN_AUTHORIZED",
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
        }
    )
    authorization_path = tmp_path / "mock-run-authorization.json"
    _write_json(authorization_path, authorization)
    return (
        closure,
        closure_path,
        runtime_signed,
        authorization,
        authorization_path,
    )


@pytest.mark.parametrize(
    ("failure_stage", "exception", "device"),
    (
        (
            "closure",
            FileNotFoundError("mock implementation closure is missing"),
            runner.FROZEN_DEVICE,
        ),
        (
            "authorization",
            FileNotFoundError("mock run authorization is missing"),
            runner.FROZEN_DEVICE,
        ),
        (
            "hash",
            RuntimeError("mock frozen protocol hash mismatch"),
            runner.FROZEN_DEVICE,
        ),
        (
            "runtime",
            RuntimeError("mock runtime binding mismatch"),
            runner.FROZEN_DEVICE,
        ),
        (
            "device",
            ValueError("CC-SEA v8 bounded execution fixes --device at cuda:0"),
            "cuda:1",
        ),
        (
            "output",
            FileExistsError("mock output target was already consumed"),
            runner.FROZEN_DEVICE,
        ),
    ),
)
def test_static_failure_precedes_loader_and_output_claim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loader_guard: dict[str, int],
    failure_stage: str,
    exception: Exception,
    device: str,
) -> None:
    output = tmp_path / "cc-sea-v8-r1"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        runner,
        "_validate_output_target",
        (
            (lambda _path: (_ for _ in ()).throw(exception))
            if failure_stage == "output"
            else (lambda _path: output)
        ),
    )

    if failure_stage == "hash":
        monkeypatch.setattr(
            runner,
            "_load_frozen_dry_evidence",
            lambda: (_ for _ in ()).throw(exception),
        )
    elif failure_stage == "runtime":
        monkeypatch.setattr(
            runner,
            "_verify_implementation_files",
            lambda _binding: (_ for _ in ()).throw(exception),
        )
    elif failure_stage in {"closure", "authorization"}:
        if failure_stage == "authorization":
            (
                closure,
                closure_path,
                runtime_signed,
                _,
                _,
            ) = _fake_closure_and_authorization(tmp_path)
            monkeypatch.setattr(
                runner,
                "_load_implementation_closure",
                lambda *_args, **_kwargs: (
                    closure,
                    closure_path,
                    runtime_signed,
                ),
            )
        else:
            monkeypatch.setattr(
                runner,
                "_load_implementation_closure",
                lambda *_args, **_kwargs: (
                    _ for _ in ()
                ).throw(exception),
            )
        if failure_stage == "authorization":
            monkeypatch.setattr(
                runner,
                "_load_authorization",
                lambda *_args, **_kwargs: (
                    _ for _ in ()
                ).throw(exception),
            )

    expected_type = type(exception)
    with pytest.raises(expected_type, match=str(exception)):
        runner.run(_args(output, device=device))
    assert loader_guard["calls"] == 0
    assert not output.exists()


class _ReceiptObject:
    def __init__(self, payload: Mapping[str, object]) -> None:
        self._payload = dict(payload)

    def canonical_receipt(self) -> dict[str, object]:
        return deepcopy(self._payload)


class _UnchangedBundle:
    def __init__(self) -> None:
        self.verify_calls = 0

    def verify_unchanged(self) -> None:
        self.verify_calls += 1


def _fake_real_inputs() -> runner._FrozenRealInputs:
    return runner._FrozenRealInputs(
        source_config={"config_fingerprint": runner.CONFIG_FINGERPRINT},
        source_config_path=(_ROOT / runner.CONFIG_REPO_PATH).resolve(),
        bundle=_UnchangedBundle(),
        immutable={},
        population=_ReceiptObject(
            {
                "schema_version": "mock-anchor-population-v1",
                "pair_catalog_fingerprint": (
                    runner.PAIR_CATALOG_FINGERPRINT
                ),
                "prepared_catalog_fingerprint": (
                    runner.PREPARED_CATALOG_FINGERPRINT
                ),
                "population_fingerprint": (
                    runner.ANCHOR_POPULATION_FINGERPRINT
                ),
            }
        ),
        factual_schedule=_ReceiptObject(
            {
                "schema_version": "mock-factual-schedule-v1",
                "population_fingerprint": (
                    runner.ANCHOR_POPULATION_FINGERPRINT
                ),
                "schedule_fingerprint": (
                    runner.FACTUAL_SCHEDULE_FINGERPRINT
                ),
            }
        ),
        materializer=_ReceiptObject(
            {
                "schema_version": "mock-outcome-inputs-v1",
                "pair_catalog_fingerprint": (
                    runner.PAIR_CATALOG_FINGERPRINT
                ),
                "prepared_catalog_fingerprint": (
                    runner.PREPARED_CATALOG_FINGERPRINT
                ),
                "materializer_fingerprint": (
                    runner.MATERIALIZER_FINGERPRINT
                ),
                "all_outcome_pair_input_fingerprint": (
                    runner.ALL_PAIR_INPUTS_FINGERPRINT
                ),
                "gt_union_population_fingerprint": (
                    runner.GT_UNION_POPULATION_FINGERPRINT
                ),
            }
        ),
        outcome_schedule=_ReceiptObject(
            {
                "schema_version": "mock-outcome-schedule-v1",
                "catalog_fingerprint": runner.PAIR_CATALOG_FINGERPRINT,
                "schedule_fingerprint": (
                    runner.OUTCOME_SCHEDULE_FINGERPRINT
                ),
                "sequence_fingerprint": (
                    runner.OUTCOME_SEQUENCE_FINGERPRINT
                ),
            }
        ),
        pair_catalog_fingerprint=runner.PAIR_CATALOG_FINGERPRINT,
        prepared_catalog_fingerprint=runner.PREPARED_CATALOG_FINGERPRINT,
    )


def _core_result(*, model_pass: bool) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": (
            runner.CONSERVATIVE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA
        ),
        "method_id": runner.CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID,
        "execution_status": "completed",
        "device": runner.FROZEN_DEVICE,
        "decision": (
            "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_PASS"
            if model_pass
            else "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_FAIL"
        ),
        "structural_execution_pass": True,
        "computational_model_code_gate_pass": model_pass,
        "population_fingerprint": runner.ANCHOR_POPULATION_FINGERPRINT,
        "factual_schedule_fingerprint": (
            runner.FACTUAL_SCHEDULE_FINGERPRINT
        ),
        "materializer_fingerprint": runner.MATERIALIZER_FINGERPRINT,
        "outcome_schedule_fingerprint": (
            runner.OUTCOME_SCHEDULE_FINGERPRINT
        ),
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


def _expected_input_fingerprints() -> dict[str, str]:
    return {
        "population_fingerprint": runner.ANCHOR_POPULATION_FINGERPRINT,
        "factual_schedule_fingerprint": (
            runner.FACTUAL_SCHEDULE_FINGERPRINT
        ),
        "materializer_fingerprint": runner.MATERIALIZER_FINGERPRINT,
        "outcome_schedule_fingerprint": (
            runner.OUTCOME_SCHEDULE_FINGERPRINT
        ),
        "all_pair_inputs_fingerprint": runner.ALL_PAIR_INPUTS_FINGERPRINT,
        "gt_union_population_fingerprint": (
            runner.GT_UNION_POPULATION_FINGERPRINT
        ),
        "outcome_sequence_fingerprint": (
            runner.OUTCOME_SEQUENCE_FINGERPRINT
        ),
    }


def _install_authorized_publication_mocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loader_guard: dict[str, int],
    *,
    core_result: Mapping[str, object] | Exception,
) -> tuple[Path, list[dict[str, object]]]:
    output = tmp_path / "cc-sea-v8-r1"
    (
        closure,
        closure_path,
        runtime_signed,
        authorization,
        authorization_path,
    ) = _fake_closure_and_authorization(tmp_path)
    fake_inputs = _fake_real_inputs()
    verified_core_results: list[dict[str, object]] = []

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    def validate_output(_path: Path) -> Path:
        if output.exists():
            raise FileExistsError("mock CC-SEA v8 run already consumed")
        return output

    monkeypatch.setattr(runner, "_validate_output_target", validate_output)
    monkeypatch.setattr(
        runner,
        "_load_implementation_closure",
        lambda *_args, **_kwargs: (
            closure,
            closure_path,
            runtime_signed,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_load_authorization",
        lambda *_args, **_kwargs: (
            authorization,
            authorization_path,
        ),
    )

    def fake_loader(_config: Mapping[str, object]) -> runner._FrozenRealInputs:
        loader_guard["calls"] += 1
        assert output.is_dir()
        assert (output / runner._INCOMPLETE).is_file()
        assert (output / "receipts" / "run_claim.json").is_file()
        assert (
            output / "receipts" / "authorization_binding.json"
        ).is_file()
        return fake_inputs

    def fake_execute(*_args: object, **_kwargs: object) -> dict[str, object]:
        if isinstance(core_result, Exception):
            raise core_result
        return deepcopy(dict(core_result))

    def verify_core(value: Mapping[str, object]) -> None:
        unsigned = dict(value)
        result_fingerprint = unsigned.pop("result_fingerprint", None)
        assert result_fingerprint == stable_fingerprint(unsigned)
        assert value.get("structural_execution_pass") is True
        assert isinstance(
            value.get("computational_model_code_gate_pass"),
            bool,
        )
        verified_core_results.append(deepcopy(dict(value)))

    monkeypatch.setattr(runner, "_load_frozen_real_inputs", fake_loader)
    monkeypatch.setattr(
        runner,
        "execute_conservative_factorized_outcome_bounded",
        fake_execute,
    )
    monkeypatch.setattr(runner, "_verify_core_result", verify_core)
    monkeypatch.setattr(
        runner,
        "_verify_published_input_receipts",
        lambda _payloads: _expected_input_fingerprints(),
    )
    return output, verified_core_results


@pytest.mark.parametrize(
    ("model_pass", "expected_decision"),
    (
        (True, "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_PASS"),
        (False, "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_FAIL"),
    ),
)
def test_complete_pass_and_nonpass_publish_reload_and_consume_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loader_guard: dict[str, int],
    model_pass: bool,
    expected_decision: str,
) -> None:
    output, verified = _install_authorized_publication_mocks(
        monkeypatch,
        tmp_path,
        loader_guard,
        core_result=_core_result(model_pass=model_pass),
    )

    published = runner.run(_args(output))

    assert published["decision"] == expected_decision
    assert published["structural_execution_pass"] is True
    assert published["bounded_model_code_gate_pass"] is model_pass
    assert published["real_D_R_run_claim_consumed"] is True
    assert published["not_detection_performance_evidence"] is True
    assert loader_guard["calls"] == 1
    assert len(verified) >= 2
    assert not (output / runner._INCOMPLETE).exists()
    assert (output / "receipts" / "result.json").is_file()
    assert not (output / "receipts" / "failure.json").exists()

    strict = (
        runner.load_conservative_factorized_outcome_bounded_artifact(
            output
        )
    )
    assert strict.decision == expected_decision
    assert strict.structural_execution_pass is True
    assert strict.bounded_model_code_gate_pass is model_pass

    with pytest.raises(FileExistsError, match="already consumed"):
        runner.run(_args(output))
    assert loader_guard["calls"] == 1


def test_execution_error_is_create_only_reloadable_and_consumes_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loader_guard: dict[str, int],
) -> None:
    output, verified = _install_authorized_publication_mocks(
        monkeypatch,
        tmp_path,
        loader_guard,
        core_result=RuntimeError("mock bounded execution failure"),
    )

    published = runner.run(_args(output))

    assert published["decision"] == "CC_SEA_V8_BOUNDED_EXECUTION_ERROR"
    assert published["structural_execution_pass"] is False
    assert published["bounded_model_code_gate_pass"] is False
    assert loader_guard["calls"] == 1
    assert verified == []
    assert not (output / runner._INCOMPLETE).exists()
    assert (output / "receipts" / "failure.json").is_file()
    assert not (output / "receipts" / "result.json").exists()
    failure = json.loads(
        (output / "receipts" / "failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["phase"] == "BOUNDED_EXECUTION"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["message"] == "mock bounded execution failure"
    assert failure["real_D_R_run_claim_consumed"] is True

    strict = (
        runner.load_conservative_factorized_outcome_bounded_artifact(
            output
        )
    )
    assert strict.decision == "CC_SEA_V8_BOUNDED_EXECUTION_ERROR"
    with pytest.raises(FileExistsError, match="already consumed"):
        runner.run(_args(output))
    assert loader_guard["calls"] == 1


def test_reconstruction_error_is_create_only_reloadable_and_consumes_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    loader_guard: dict[str, int],
) -> None:
    output, _ = _install_authorized_publication_mocks(
        monkeypatch,
        tmp_path,
        loader_guard,
        core_result=_core_result(model_pass=True),
    )

    def failing_fake_loader(
        _config: Mapping[str, object],
    ) -> runner._FrozenRealInputs:
        loader_guard["calls"] += 1
        assert output.is_dir()
        assert (output / runner._INCOMPLETE).is_file()
        assert (output / "receipts" / "run_claim.json").is_file()
        raise RuntimeError("mock D_R reconstruction failure")

    monkeypatch.setattr(
        runner,
        "_load_frozen_real_inputs",
        failing_fake_loader,
    )

    published = runner.run(_args(output))

    assert published["decision"] == "CC_SEA_V8_BOUNDED_EXECUTION_ERROR"
    assert loader_guard["calls"] == 1
    assert not (output / runner._INCOMPLETE).exists()
    names = {
        path.name for path in (output / "receipts").iterdir()
    }
    assert names == (
        runner._PRE_RUN_RECEIPTS | {"decision.json", "failure.json"}
    )
    failure = json.loads(
        (output / "receipts" / "failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert failure["phase"] == "D_R_RECONSTRUCTION"
    assert failure["exception_type"] == "RuntimeError"
    assert failure["message"] == "mock D_R reconstruction failure"
    assert failure["real_D_R_run_claim_consumed"] is True

    strict = (
        runner.load_conservative_factorized_outcome_bounded_artifact(
            output
        )
    )
    assert strict.decision == "CC_SEA_V8_BOUNDED_EXECUTION_ERROR"
    with pytest.raises(FileExistsError, match="already consumed"):
        runner.run(_args(output))
    assert loader_guard["calls"] == 1

