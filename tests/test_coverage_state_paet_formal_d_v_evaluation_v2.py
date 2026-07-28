from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite_eval_v2 import evaluation_source_closure as closure
from cure_lite_eval_v2 import formal800_schema_erratum as erratum
from cure_lite_eval_v2 import formal_d_v_runner_v2 as runner


def test_append_only_erratum_binds_frozen_formal_and_original_closure() -> None:
    receipt = erratum.verify_formal800_schema_erratum()

    assert receipt["verified"] is True
    assert receipt["authorized_alias_count"] == 2
    assert receipt["formal_complete_sha256"] == (
        "e5a75a171d455bb31d8194b65981cdd618f8a26f49fa5a224334f9f14d182f1a"
    )
    assert receipt["original_source_closure_manifest_sha256"] == (
        "1afb838456525f136cfe73a5b52debdd93dc939ed421c743530da69621f190f5"
    )


def test_original_loader_passes_with_only_two_exact_in_memory_aliases() -> None:
    before = (
        runner._attempt.PAET_FORMAL_ATTEMPT_SCHEMA,
        runner._attempt.PAET_FORMAL_STARTED_SCHEMA,
    )
    loaded = runner._load_original_attempt_with_aliases()

    assert type(loaded) is runner._attempt.LoadedCoverageStatePAETFormalAttempt
    assert loaded.complete_fingerprint == (
        "116fbe67c5f9ec74cc2317ee9dc283845b4da9f5699cd07537b72917338bb74c"
    )
    assert (
        runner._attempt.PAET_FORMAL_ATTEMPT_SCHEMA,
        runner._attempt.PAET_FORMAL_STARTED_SCHEMA,
    ) == before


def test_alias_context_rejects_an_unlisted_consumer_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner._attempt,
        "PAET_FORMAL_ATTEMPT_SCHEMA",
        "unexpected-consumer-schema",
    )

    with pytest.raises(RuntimeError, match="consumer constants changed"):
        with runner._exact_formal800_schema_aliases():
            raise AssertionError("must not enter context")


def test_evaluation_closure_scope_is_separate_from_original_package() -> None:
    paths = closure.evaluation_source_closure_paths()

    assert len(paths) == 7
    assert not any(path.startswith("cure_lite/") for path in paths)
    assert "cure_lite_eval_v2/formal_d_v_runner_v2.py" in paths
    assert (
        closure.EVALUATION_SOURCE_CLOSURE_MANIFEST_REPO_PATH
        not in paths
    )


def _fake_closure_bindings() -> tuple[
    dict[str, object],
    dict[str, object],
]:
    return (
        {
            "manifest_repo_path": "old.json",
            "manifest_sha256": "1" * 64,
            "archive_repo_path": "old.tar",
            "archive_sha256": "2" * 64,
            "content_fingerprint": "3" * 64,
            "file_count": 223,
        },
        {
            "repo_path": "erratum.json",
            "sha256": "4" * 64,
            "erratum_fingerprint": "5" * 64,
        },
    )


def test_evaluation_closure_builder_and_validator_are_create_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    for repo_path in closure._SOURCE_PATHS:
        source = repository / repo_path
        target = tmp_path / repo_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    monkeypatch.setattr(
        closure,
        "_closure_bindings",
        lambda repository_root=None: _fake_closure_bindings(),
    )

    first = closure.build_evaluation_source_closure(tmp_path)
    second = closure.verify_evaluation_source_closure(tmp_path)

    assert first == second
    assert first["sealed"] is True
    assert first["file_count"] == 7
    with pytest.raises(FileExistsError):
        closure.build_evaluation_source_closure(tmp_path)

    changed = tmp_path / "cure_lite_eval_v2/__init__.py"
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(
        closure.EvaluationSourceClosureError,
        match="live evaluation source changed",
    ):
        closure.verify_evaluation_source_closure(tmp_path)


def test_validate_create_only_delegates_without_d_v_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_chain = {
        "erratum": {"erratum_fingerprint": "a" * 64},
        "evaluation_closure": {
            "content_fingerprint": "b" * 64,
        },
    }
    fake_loaded = SimpleNamespace(
        complete_fingerprint="c" * 64,
        artifact=SimpleNamespace(artifact_fingerprint="d" * 64),
        source_closure_manifest_sha256="e" * 64,
        source_closure_archive_sha256="f" * 64,
        source_closure_content_fingerprint="0" * 64,
    )
    calls = {"original_create_only": 0}

    def original_create_only() -> dict[str, object]:
        calls["original_create_only"] += 1
        return {
            "create_only": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "output_created": False,
        }

    monkeypatch.setattr(runner, "_validate_closure_chain", lambda: fake_chain)
    monkeypatch.setattr(
        runner,
        "_load_original_attempt_with_aliases",
        lambda: fake_loaded,
    )
    monkeypatch.setattr(
        runner._d_v,
        "validate_paet_formal_d_v_create_only",
        original_create_only,
    )

    result = runner.validate_paet_formal_d_v_create_only_v2()

    assert calls == {"original_create_only": 1}
    assert result["D_V_accessed"] is False
    assert result["D_T_accessed"] is False
    assert result["output_created"] is False
    assert result["Formal800_attempt_loaded"] is True
    body = dict(result)
    fingerprint = body.pop("validation_fingerprint")
    assert fingerprint == stable_fingerprint(body)


def test_external_receipt_binds_three_immutable_d_v_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "runs/fixed_d_v"
    output.mkdir(parents=True)
    payloads = {
        "COMPLETE.json": {
            "complete_fingerprint": "1" * 64,
        },
        "receipt.json": {
            "receipt_fingerprint": "2" * 64,
            "evaluation_result_fingerprint": "3" * 64,
        },
        "decision.json": {
            "decision_fingerprint": "4" * 64,
        },
    }
    for name, payload in payloads.items():
        (output / name).write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
    monkeypatch.setattr(runner, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        runner._d_v,
        "PAET_FORMAL_DV_OUTPUT_PATH",
        output,
    )
    monkeypatch.setattr(
        runner._d_v,
        "_validate_published_output",
        lambda path: {
            "gate_passed": True,
            "authorizes_D_T": True,
        },
    )
    chain = {
        "erratum": {
            "formal_complete_sha256": "5" * 64,
            "formal_complete_fingerprint": "6" * 64,
            "original_source_closure_manifest_sha256": "7" * 64,
            "original_source_closure_archive_sha256": "8" * 64,
            "original_source_closure_content_fingerprint": "9" * 64,
            "original_source_closure_file_count": 223,
            "erratum_repo_path": "protocols/erratum.json",
            "erratum_sha256": "a" * 64,
            "erratum_fingerprint": "b" * 64,
            "authorized_alias_count": 2,
        },
        "evaluation_closure": {
            "manifest_repo_path": "artifacts/new.json",
            "manifest_sha256": "c" * 64,
            "archive_repo_path": "artifacts/new.tar",
            "archive_sha256": "d" * 64,
            "content_fingerprint": "e" * 64,
            "file_count": 7,
        },
    }

    binding = runner._build_external_evidence_binding(chain)

    assert set(binding["D_V"]["files"]) == set(runner._D_V_MEMBER_NAMES)
    for name, row in binding["D_V"]["files"].items():
        assert row["sha256"] == file_sha256(output / name)
    assert binding["recovery_scope"] == {
        "in_memory_schema_alias_count": 2,
        "artifact_bytes_modified": False,
        "original_D_V_output_modified": False,
        "model_retrained": False,
        "model_state_updated": False,
        "metric_or_gate_modified": False,
    }
    body = dict(binding)
    fingerprint = body.pop("binding_fingerprint")
    assert fingerprint == stable_fingerprint(body)


def test_run_wrapper_patches_only_during_original_runner_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, str]] = []

    def original_run() -> dict[str, object]:
        observed.append(
            (
                runner._attempt.PAET_FORMAL_ATTEMPT_SCHEMA,
                runner._attempt.PAET_FORMAL_STARTED_SCHEMA,
            )
        )
        return {"status": "complete"}

    monkeypatch.setattr(
        runner,
        "_validate_closure_chain",
        lambda: {"erratum": {}, "evaluation_closure": {}},
    )
    monkeypatch.setattr(runner._d_v, "run_paet_formal_d_v_once", original_run)
    monkeypatch.setattr(
        runner,
        "_build_external_evidence_binding",
        lambda chain: {
            "schema_version": runner.EVIDENCE_BINDING_SCHEMA,
            "binding_fingerprint": "f" * 64,
        },
    )
    monkeypatch.setattr(runner, "_publish_external_binding", lambda payload: None)
    monkeypatch.setattr(
        runner,
        "_verify_external_binding",
        lambda payload: {
            "repo_path": "binding.json",
            "sha256": "0" * 64,
            "binding_fingerprint": "f" * 64,
        },
    )
    monkeypatch.setattr(
        runner,
        "EVIDENCE_BINDING_PATH",
        tmp_path / "binding.json",
    )
    monkeypatch.setattr(
        runner,
        "EVIDENCE_BINDING_STAGING_PATH",
        tmp_path / ".binding.json.incomplete",
    )

    result = runner.run_paet_formal_d_v_once_v2()

    assert observed == [
        (erratum.ATTEMPT_PRODUCER_SCHEMA, erratum.STARTED_PRODUCER_SCHEMA)
    ]
    assert runner._attempt.PAET_FORMAL_ATTEMPT_SCHEMA == (
        erratum.ATTEMPT_CONSUMER_SCHEMA
    )
    assert runner._attempt.PAET_FORMAL_STARTED_SCHEMA == (
        erratum.STARTED_CONSUMER_SCHEMA
    )
    assert result["model_retrained"] is False
    assert result["artifact_bytes_modified"] is False
