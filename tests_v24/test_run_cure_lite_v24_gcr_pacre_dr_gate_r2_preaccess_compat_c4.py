from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = (
    REPOSITORY
    / "tools/run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c4.py"
)
FROZEN_SOURCE = (
    REPOSITORY / "tools/run_cure_lite_v24_gcr_pacre_dr_gate_r2.py"
)
EXPECTED_FROZEN_SHA256 = (
    "5cbfd073d7df8f4257079c71e6f05110d31b383c48abe6e0e9127ee154785495"
)
EXPECTED_SUPERVISOR_MODULE = (
    "tools.cure_lite_v24_runtime_supervisor_preaccess_compat_c4"
)
EXPECTED_R2_IDENTITY = {
    "GCR_PACRE_DR_GATE_SCHEMA": (
        "cure-lite-v24-gcr-pacre-real-dr-structural-gate-r2-v1"
    ),
    "GCR_PACRE_DR_RUN_ID": (
        "gcr_pacre_v24_D_R_zero_update_structural_r2"
    ),
    "GCR_PACRE_DR_PREACCESS_SCHEMA": (
        "cure-lite-v24-D_R-structural-r2-authorization-v1"
    ),
    "GCR_PACRE_DR_PREACCESS_STAGE_ID": (
        "gcr_pacre_v24_D_R_structural_r2"
    ),
    "GCR_PACRE_DR_PREACCESS_STATUS": (
        "GCR_PACRE_V24_D_R_STRUCTURAL_R2_AUTHORIZED"
    ),
    "GCR_PACRE_DR_ACCESS_AUDIT_SCHEMA": (
        "cure-lite-v24-split-access-audit-v1"
    ),
    "GCR_PACRE_DR_RUN_START_SCHEMA": (
        "cure-lite-v24-D_R-persistent-run-start-r2-v1"
    ),
    "GCR_PACRE_DR_RUN_START_PARENT": (
        "runs/irstd1k_stage_a_seed42/"
        "gcr_pacre_v24_D_R_structural_attempt_r2"
    ),
    "GCR_PACRE_DR_ACCESS_AUDIT_PATH": (
        "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_access_audit.json"
    ),
    "GCR_PACRE_DR_PREACCESS_AUTHORIZATION_PATH": (
        "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_authorization.json"
    ),
    "GCR_PACRE_DR_RECEIPT_PATH": (
        "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_receipt.json"
    ),
}


@pytest.fixture(scope="module")
def adapter():
    name = "run_cure_lite_v24_gcr_pacre_dr_gate_r2_compat_c4_tested"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)
        sys.modules.pop(
            "tools._run_cure_lite_v24_gcr_pacre_dr_gate_r2_"
            "materialized_for_preaccess_compat_c4",
            None,
        )


def _fingerprint(value: dict[str, object]) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def test_frozen_hash_and_transformation_are_exactly_one_import(adapter) -> None:
    frozen = FROZEN_SOURCE.read_bytes()
    assert hashlib.sha256(frozen).hexdigest() == EXPECTED_FROZEN_SHA256
    assert adapter.FROZEN_ADAPTER_PATH == FROZEN_SOURCE
    assert adapter.FROZEN_ADAPTER_SHA256 == EXPECTED_FROZEN_SHA256
    assert frozen.count(adapter._FROZEN_VERIFIER_IMPORT) == 1
    assert adapter._COMPAT_VERIFIER_IMPORT not in frozen

    transformed = adapter._materialized_compat_source()
    expected = frozen.replace(
        adapter._FROZEN_VERIFIER_IMPORT,
        adapter._COMPAT_VERIFIER_IMPORT,
    )
    assert transformed == expected
    assert transformed.count(adapter._FROZEN_VERIFIER_IMPORT) == 0
    assert transformed.count(adapter._COMPAT_VERIFIER_IMPORT) == 1
    assert transformed.replace(
        adapter._COMPAT_VERIFIER_IMPORT,
        adapter._FROZEN_VERIFIER_IMPORT,
    ) == frozen

    tree = ast.parse(transformed, filename=str(SOURCE))
    verifier_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and any(
            alias.name == "verify_child_runtime_attestation"
            for alias in node.names
        )
    ]
    assert len(verifier_imports) == 1
    assert verifier_imports[0].module == EXPECTED_SUPERVISOR_MODULE


def test_materialized_import_and_compatibility_identity_are_exact(adapter) -> None:
    assert Path(adapter.COMPAT_ADAPTER_PATH) == SOURCE
    assert adapter.COMPAT_SUPERVISOR_MODULE == EXPECTED_SUPERVISOR_MODULE
    assert Path(adapter.legacy.__file__).resolve() == SOURCE
    assert adapter.legacy.__package__ == "tools"
    assert adapter.legacy._R2_IDENTITY == EXPECTED_R2_IDENTITY

    identity = adapter.verify_compatibility_identity()
    assert identity == {
        "scientific_attempt_ordinal": 2,
        "runtime_compatibility_generation": "c4",
        "adapter_path": str(SOURCE),
        "adapter_repo_path": str(SOURCE.relative_to(REPOSITORY)),
        "attestation_verifier_module": EXPECTED_SUPERVISOR_MODULE,
        "frozen_adapter_path": str(FROZEN_SOURCE),
        "frozen_adapter_file_sha256": EXPECTED_FROZEN_SHA256,
        "transformed_import_count": 1,
        "scientific_identity_changed": False,
        "scientific_receipt_paths_changed": False,
        "scientific_run_marker_paths_changed": False,
        "automatic_retry_allowed": False,
        "resume_allowed": False,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }


def test_scientific_identity_paths_and_decision_source_are_unchanged(
    adapter,
) -> None:
    legacy = adapter.legacy
    assert legacy._R2_IDENTITY == EXPECTED_R2_IDENTITY
    assert legacy.R2_EXECUTION_IDENTITY_SCHEMA == (
        "cure-lite-v24-D_R-structural-r2-execution-identity-v1"
    )
    assert legacy.EXPECTED_R1_SCIENTIFIC_PATH_COUNT == 103
    assert legacy.EXPECTED_R1_SOURCE_CLOSURE_FINGERPRINT == (
        "28d26759a68785e9c99917fcfa8b36430c7f6e5463282d66eeab5c711e425e9f"
    )
    assert legacy._ALLOWED_R2_MODES == frozenset(
        {"preaccess-create", "preaccess-verify", "real"}
    )
    assert EXPECTED_R2_IDENTITY["GCR_PACRE_DR_RECEIPT_PATH"].endswith(
        "D_R_structural_attempt_r2_receipt.json"
    )
    assert EXPECTED_R2_IDENTITY["GCR_PACRE_DR_RUN_START_PARENT"] == (
        "runs/irstd1k_stage_a_seed42/"
        "gcr_pacre_v24_D_R_structural_attempt_r2"
    )

    # Byte equality after reversing the sole verifier import proves that no
    # model, gate, split, threshold, decision, receipt, or marker code moved.
    transformed = adapter._materialized_compat_source()
    reconstructed = transformed.replace(
        adapter._COMPAT_VERIFIER_IMPORT,
        adapter._FROZEN_VERIFIER_IMPORT,
    )
    assert reconstructed == FROZEN_SOURCE.read_bytes()


def test_runtime_attestation_dispatches_only_to_c4_supervisor(
    adapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    supervisor = ModuleType(EXPECTED_SUPERVISOR_MODULE)

    def verify(attestation: str, authorization: str) -> dict[str, object]:
        calls.append((attestation, authorization))
        return {
            "runtime_attestation_valid": True,
            "D_R_payload_accessed": False,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }

    supervisor.verify_child_runtime_attestation = verify
    monkeypatch.setitem(sys.modules, EXPECTED_SUPERVISOR_MODULE, supervisor)
    monkeypatch.setenv(
        adapter.legacy._RUNTIME_ATTESTATION_ENV,
        "/metadata/c4-child-runtime-attestation.json",
    )

    result = adapter._verify_runtime_launch(
        "/metadata/c4-runtime-launch-authorization.json"
    )
    assert calls == [
        (
            "/metadata/c4-child-runtime-attestation.json",
            "/metadata/c4-runtime-launch-authorization.json",
        )
    ]
    assert result["runtime_attestation_valid"] is True
    assert result["D_R_payload_accessed"] is False
    assert result["D_V_payload_accessed"] is False
    assert result["D_T_payload_accessed"] is False


def test_frozen_hash_drift_fails_closed(
    adapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter, "FROZEN_ADAPTER_SHA256", "0" * 64)
    with pytest.raises(PermissionError, match="frozen r2 adapter source changed"):
        adapter._verify_frozen_adapter_source()
    with pytest.raises(PermissionError, match="frozen r2 adapter source changed"):
        adapter._materialized_compat_source()
    with pytest.raises(PermissionError, match="frozen r2 adapter source changed"):
        adapter.verify_compatibility_identity()


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("_FROZEN_VERIFIER_IMPORT", b"import "),
        ("_COMPAT_VERIFIER_IMPORT", b"import os"),
    ),
)
def test_nonunique_or_preexisting_projection_fails_closed(
    adapter,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: bytes,
) -> None:
    monkeypatch.setattr(adapter, field, replacement)
    with pytest.raises(
        PermissionError,
        match="attestation-verifier import changed|projection is not unique",
    ):
        adapter._materialized_compat_source()


def test_direct_cli_identity_summary_is_metadata_only_and_still_r2() -> None:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("CURE_LITE_V24_RUNTIME_")
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            "/usr/bin/python3.12",
            "-I",
            "-S",
            "-B",
            "-u",
            str(SOURCE),
            "--r2-execution-identity-summary",
        ],
        cwd=REPOSITORY,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.endswith("\n")
    assert completed.stdout.count("\n") == 1
    payload = json.loads(completed.stdout)
    fingerprint = payload.pop("execution_identity_fingerprint")
    assert fingerprint == _fingerprint(payload)
    assert payload["schema_version"] == (
        "cure-lite-v24-D_R-structural-r2-execution-identity-v1"
    )
    assert payload["adapter_repo_path"] == str(SOURCE.relative_to(REPOSITORY))
    assert payload["attempt_ordinal"] == 2
    assert payload["frozen_scientific_path_count"] == 103
    assert payload["frozen_scientific_source_closure_fingerprint"] == (
        "28d26759a68785e9c99917fcfa8b36430c7f6e5463282d66eeab5c711e425e9f"
    )
    assert payload["numerical_or_scientific_change_authorized"] is False
    assert payload["D_R_payload_authorized_by_adapter"] is False
    assert payload["D_V_payload_authorized"] is False
    assert payload["D_T_payload_authorized"] is False
    assert payload["training_authorized"] is False
    assert payload["optimizer_steps_authorized"] == 0
    assert payload["parameter_updates_authorized"] == 0
    assert payload["automatic_retry_allowed"] is False
    assert payload["resume_allowed"] is False


def test_empty_or_unknown_entry_never_reaches_gate(adapter) -> None:
    with pytest.raises(
        PermissionError,
        match="permits only real/preaccess modes",
    ):
        adapter.main([])
    with pytest.raises(
        PermissionError,
        match="permits only real/preaccess modes",
    ):
        adapter.main(["unknown-mode"])
