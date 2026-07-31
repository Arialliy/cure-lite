from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cure_lite_v24 import dr_gate
from tools import run_cure_lite_v24_gcr_pacre_dr_gate_r2 as adapter


REPOSITORY = Path(__file__).resolve().parents[1]
ADAPTER = (
    REPOSITORY / "tools/run_cure_lite_v24_gcr_pacre_dr_gate_r2.py"
)
ACTUAL_PYTHON = Path("/usr/bin/python3.12")


def _run_summary() -> dict[str, object]:
    completed = subprocess.run(
        [
            str(ACTUAL_PYTHON),
            "-I",
            "-S",
            "-B",
            str(ADAPTER),
            "--r2-execution-identity-summary",
        ],
        cwd=REPOSITORY,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.stderr == ""
    return json.loads(completed.stdout)


def _path_state(path: Path) -> tuple[object, ...]:
    if not path.exists() and not path.is_symlink():
        return ("absent",)
    metadata = path.lstat()
    content = path.read_bytes() if path.is_file() and not path.is_symlink() else b""
    return (
        "present",
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_nlink,
        content,
    )


def test_r2_summary_is_data_free_and_uses_distinct_identity() -> None:
    summary = _run_summary()

    assert summary["schema_version"] == (
        "cure-lite-v24-D_R-structural-r2-execution-identity-v1"
    )
    assert summary["attempt_ordinal"] == 2
    assert summary["prior_attempt_count"] == 1
    assert summary["prior_attempt_status"] == (
        "OBSERVABILITY_LOST_NO_AUTHENTICATED_DECISION"
    )
    assert summary["numerical_or_scientific_change_authorized"] is False
    assert summary["D_R_payload_authorized_by_adapter"] is False
    assert summary["D_V_payload_authorized"] is False
    assert summary["D_T_payload_authorized"] is False
    assert summary["training_authorized"] is False
    assert summary["optimizer_steps_authorized"] == 0
    assert summary["parameter_updates_authorized"] == 0
    assert summary["automatic_retry_allowed"] is False
    assert summary["resume_allowed"] is False
    assert summary["frozen_scientific_path_count"] == 103
    assert summary[
        "frozen_scientific_source_closure_fingerprint"
    ] == "28d26759a68785e9c99917fcfa8b36430c7f6e5463282d66eeab5c711e425e9f"

    transition = summary["identity_transition"]
    assert isinstance(transition, dict)
    for name, values in transition.items():
        assert isinstance(values, dict)
        if name == "GCR_PACRE_DR_ACCESS_AUDIT_SCHEMA":
            assert values["r1"] == values["r2"] == (
                "cure-lite-v24-split-access-audit-v1"
            )
        else:
            assert values["r1"] != values["r2"]
    assert transition["GCR_PACRE_DR_RUN_ID"]["r2"] == (
        "gcr_pacre_v24_D_R_zero_update_structural_r2"
    )
    assert transition["GCR_PACRE_DR_RECEIPT_PATH"]["r2"].endswith(
        "/D_R_structural_attempt_r2_receipt.json"
    )
    assert transition["GCR_PACRE_DR_RUN_START_PARENT"]["r2"].endswith(
        "/gcr_pacre_v24_D_R_structural_attempt_r2"
    )


def test_summary_does_not_create_r2_protocol_artifacts() -> None:
    relative_paths = (
        "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_access_audit.json",
        "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_authorization.json",
        "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2/"
        "D_R_structural_attempt_r2_receipt.json",
    )
    paths = tuple(REPOSITORY / value for value in relative_paths)
    before = {path: _path_state(path) for path in paths}

    _run_summary()

    assert {path: _path_state(path) for path in paths} == before


def test_r2_metadata_only_preaccess_builder_reuses_supported_audit_schema(
) -> None:
    fixed_paths = tuple(
        REPOSITORY / adapter._R2_IDENTITY[name]
        for name in (
            "GCR_PACRE_DR_ACCESS_AUDIT_PATH",
            "GCR_PACRE_DR_PREACCESS_AUTHORIZATION_PATH",
            "GCR_PACRE_DR_RECEIPT_PATH",
        )
    )
    before = {path: _path_state(path) for path in fixed_paths}
    original = {
        name: getattr(dr_gate, name)
        for name in adapter._R1_EXPECTED
    }
    try:
        summary = adapter.configure_r2_execution_identity()
        access, authorization = (
            dr_gate.build_gcr_pacre_dr_preaccess_artifacts()
        )
        assert access["schema_version"] == (
            "cure-lite-v24-split-access-audit-v1"
        )
        assert access["stage_id"] == (
            "gcr_pacre_v24_D_R_structural_r2"
        )
        access_body = {
            key: value for key, value in access.items()
            if key != "receipt_fingerprint"
        }
        assert access["receipt_fingerprint"] == (
            adapter.stable_fingerprint(access_body)
        )
        assert authorization["schema_version"] == (
            "cure-lite-v24-D_R-structural-r2-authorization-v1"
        )
        assert authorization["stage_id"] == access["stage_id"]
        assert authorization["run_id"] == (
            "gcr_pacre_v24_D_R_zero_update_structural_r2"
        )
        assert authorization["access_audit_receipt_fingerprint"] == (
            access["receipt_fingerprint"]
        )
        authorization_body = {
            key: value for key, value in authorization.items()
            if key != "authorization_fingerprint"
        }
        assert authorization["authorization_fingerprint"] == (
            adapter.stable_fingerprint(authorization_body)
        )
        assert authorization["D_R_payload_authorized"] is True
        assert authorization["D_V_payload_authorized"] is False
        assert authorization["D_T_payload_authorized"] is False
        assert authorization["training_authorized"] is False
        transition = summary["identity_transition"]
        assert transition["GCR_PACRE_DR_ACCESS_AUDIT_SCHEMA"][
            "r1"
        ] == transition["GCR_PACRE_DR_ACCESS_AUDIT_SCHEMA"]["r2"]
    finally:
        for name, value in original.items():
            setattr(dr_gate, name, value)
    assert {path: _path_state(path) for path in fixed_paths} == before


def test_adapter_is_outside_frozen_r1_scientific_closure() -> None:
    relative = str(ADAPTER.relative_to(REPOSITORY))

    assert len(dr_gate.GCR_PACRE_DR_IMPLEMENTATION_PATHS) == 103
    assert relative not in dr_gate.GCR_PACRE_DR_IMPLEMENTATION_PATHS
    assert dr_gate.GCR_PACRE_DR_RUN_ID == (
        "gcr_pacre_v24_D_R_zero_update_structural_r1"
    )
    assert dr_gate.GCR_PACRE_DR_RECEIPT_PATH.endswith(
        "/D_R_structural_receipt.json"
    )
    assert dr_gate.GCR_PACRE_DR_RUN_START_PARENT == (
        "runs/irstd1k_stage_a_seed42"
    )


def test_r2_preaccess_summary_projects_marker_without_creating_run_root(
) -> None:
    run_root = (
        REPOSITORY
        / adapter._R2_IDENTITY["GCR_PACRE_DR_RUN_START_PARENT"]
    )
    assert _path_state(run_root) == ("absent",)

    marker = adapter._project_r2_run_start_marker_path("a" * 64)

    assert marker == run_root / (
        "gcr_pacre_v24_D_R_structural_run_start_"
        f"{'a' * 64}.json"
    )
    assert _path_state(run_root) == ("absent",)
    with pytest.raises(PermissionError, match="private preaccess token"):
        adapter._project_live_r2_preaccess_run_start_marker_path(object())
    assert _path_state(run_root) == ("absent",)


@pytest.mark.parametrize(
    "fingerprint",
    (True, 1, 1.0, None, "a" * 63, "A" * 64, "g" * 64),
)
def test_r2_marker_projection_rejects_nonexact_fingerprint(
    fingerprint: object,
) -> None:
    with pytest.raises(ValueError, match="exact SHA-256"):
        adapter._project_r2_run_start_marker_path(fingerprint)


def test_adapter_refuses_transition_after_token_issuance() -> None:
    code = f"""
import sys
sys.path.insert(0, {str(REPOSITORY)!r})
from cure_lite_v24 import dr_gate
from tools import run_cure_lite_v24_gcr_pacre_dr_gate_r2 as adapter
dr_gate._ISSUED_REAL_PREACCESS_TOKENS[1] = ("already-issued",)
try:
    adapter.configure_r2_execution_identity()
except RuntimeError as error:
    assert "after token issuance" in str(error)
else:
    raise AssertionError("identity transition unexpectedly succeeded")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr


def test_adapter_refuses_scientific_closure_drift() -> None:
    code = f"""
import sys
sys.path.insert(0, {str(REPOSITORY)!r})
from cure_lite_v24 import dr_gate
from tools import run_cure_lite_v24_gcr_pacre_dr_gate_r2 as adapter
original = dr_gate._implementation_binding()
dr_gate._implementation_binding = lambda: (
    (original[0][0], "0" * 64),
    *original[1:],
)
try:
    adapter.configure_r2_execution_identity()
except RuntimeError as error:
    assert "scientific source closure changed" in str(error)
else:
    raise AssertionError("drifted closure unexpectedly succeeded")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr


def test_adapter_rejects_generated_mode() -> None:
    completed = subprocess.run(
        [
            str(ACTUAL_PYTHON), "-I", "-S", "-B", str(ADAPTER),
            "generated",
        ],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode != 0
    assert "permits only real/preaccess modes" in completed.stderr


def test_runtime_launch_authorization_is_adapter_only_and_exact(
    tmp_path: Path,
) -> None:
    authorization = tmp_path / "runtime-launch-authorization.json"
    delegated, observed = adapter._split_runtime_launch_authorization(
        [
            "real",
            "--execute-real-dr",
            "--runtime-launch-authorization",
            str(authorization),
            "--device",
            "cuda:0",
        ]
    )

    assert delegated == [
        "real",
        "--execute-real-dr",
        "--device",
        "cuda:0",
    ]
    assert observed == str(authorization)

    for malformed in (
        ["real", "--runtime-launch-authorization"],
        [
            "real",
            "--runtime-launch-authorization",
            "relative.json",
        ],
        [
            "real",
            f"--runtime-launch-authorization={authorization}",
        ],
        [
            "real",
            "--runtime-launch-authorization",
            str(authorization),
            "--runtime-launch-authorization",
            str(authorization),
        ],
    ):
        with pytest.raises(PermissionError):
            adapter._split_runtime_launch_authorization(malformed)


def test_real_mode_requires_supervisor_runtime_authorization() -> None:
    completed = subprocess.run(
        [
            str(ACTUAL_PYTHON),
            "-I",
            "-S",
            "-B",
            str(ADAPTER),
            "real",
            "--execute-real-dr",
        ],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode != 0
    assert (
        "real mode requires runtime launch authorization"
        in completed.stderr
    )


def test_root_python_no_site_runtime_import_smoke_is_payload_free() -> None:
    completed = subprocess.run(
        [
            str(ACTUAL_PYTHON),
            "-I",
            "-S",
            "-B",
            str(ADAPTER),
            "--runtime-import-smoke",
        ],
        cwd=REPOSITORY,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    body = dict(payload)
    fingerprint = body.pop("receipt_fingerprint")
    assert fingerprint == adapter.stable_fingerprint(body)
    assert payload["python_executable"] == str(ACTUAL_PYTHON)
    assert payload["isolated"] is True
    assert payload["no_site"] is True
    assert payload["dont_write_bytecode"] is True
    assert payload["runtime_dependency_site_binding"]["path"] == (
        "/home/md0/ly/MSHNet/.venv/lib/python3.12/site-packages"
    )
    assert payload["runtime_dependency_site_binding"] == (
        adapter.EXPECTED_RUNTIME_DEPENDENCY_SITE_BINDING
    )
    authority = payload[
        "runtime_dependency_site_authority_binding"
    ]
    assert authority["path"].startswith("/proc/self/fd/")
    assert (
        authority["device"],
        authority["inode"],
        authority["owner_uid"],
        authority["mode"],
    ) == (
        payload["runtime_dependency_site_binding"]["device"],
        payload["runtime_dependency_site_binding"]["inode"],
        payload["runtime_dependency_site_binding"]["owner_uid"],
        payload["runtime_dependency_site_binding"]["mode"],
    )
    assert payload["runtime_dependency_import_mode"] == (
        "retained-procfd-no-site-processing"
    )
    assert payload["pth_files_processed"] is False
    assert all(
        value.startswith(authority["path"] + "/")
        for value in payload["module_origins"].values()
    )
    assert payload["torch_cuda_initialized"] is False
    assert isinstance(
        payload["site_module_imported_after_dependencies"], bool
    )
    assert payload["sitecustomize_imported"] is False
    assert payload["usercustomize_imported"] is False
    assert payload["gpu_accessed"] is False
    assert payload["D_R_payload_accessed"] is False
    assert payload["D_V_payload_accessed"] is False
    assert payload["D_T_payload_accessed"] is False


def test_site_import_remains_on_bound_generation_after_path_replacement(
    tmp_path: Path,
) -> None:
    site_path = (tmp_path / "site-packages").resolve()
    site_path.mkdir(mode=0o755)
    os.chmod(site_path, 0o755)
    module_name = "generated_bound_site_generation"
    (site_path / f"{module_name}.py").write_text(
        "GENERATION = 'bound-old' \n",
        encoding="utf-8",
    )
    marker = tmp_path / "pth-executed"
    (site_path / "generated.pth").write_text(
        (
            "import pathlib; "
            f"pathlib.Path({str(marker)!r}).write_text('unexpected')\n"
        ),
        encoding="utf-8",
    )
    (site_path / "sitecustomize.py").write_text(
        "raise RuntimeError('sitecustomize unexpectedly executed')\n",
        encoding="utf-8",
    )
    metadata = site_path.stat()
    expected = {
        "path": str(site_path),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "owner_uid": metadata.st_uid,
        "mode": metadata.st_mode & 0o777,
    }
    descriptor, observed, authority = (
        adapter._open_runtime_dependency_site_authority(
            site_path,
            expected,
        )
    )
    displaced = tmp_path / "bound-site-generation"
    site_path.rename(displaced)
    site_path.mkdir(mode=0o755)
    os.chmod(site_path, 0o755)
    (site_path / f"{module_name}.py").write_text(
        "GENERATION = 'replacement-new'\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(authority["path"]))
    importlib.invalidate_caches()
    try:
        loaded = importlib.import_module(module_name)
        origin = adapter._validate_bound_runtime_module_origin(
            str(loaded.__file__),
            site_descriptor=descriptor,
            authority_path=str(authority["path"]),
            site_device=int(authority["device"]),
            site_inode=int(authority["inode"]),
        )
        assert loaded.GENERATION == "bound-old"
        assert origin.startswith(str(authority["path"]) + os.sep)
        assert observed == expected
        assert (
            site_path.stat().st_dev,
            site_path.stat().st_ino,
        ) != (observed["device"], observed["inode"])
        assert marker.exists() is False
    finally:
        sys.modules.pop(module_name, None)
        sys.path.remove(str(authority["path"]))
        importlib.invalidate_caches()
        os.close(descriptor)


def test_direct_adapter_rejects_mutable_venv_interpreter() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(ADAPTER),
            "--r2-execution-identity-summary",
        ],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode != 0
    assert "exact root Python -I -S runtime" in completed.stderr


def test_runtime_attestation_helper_is_mandatory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools import cure_lite_v24_runtime_supervisor as supervisor

    attestation = tmp_path / "runtime-attestation.json"
    authorization = tmp_path / "runtime-launch-authorization.json"
    observed: list[tuple[object, ...]] = []

    def verify(
        attestation_path: str | Path,
        authorization_path: str | Path,
    ) -> dict[str, object]:
        observed.append((attestation_path, authorization_path))
        return {"runtime_attestation_valid": True}

    monkeypatch.setenv(
        "CURE_LITE_V24_RUNTIME_ATTESTATION_PATH",
        str(attestation),
    )
    monkeypatch.setattr(
        supervisor,
        "verify_child_runtime_attestation",
        verify,
    )

    assert adapter._verify_runtime_launch(str(authorization)) == {
        "runtime_attestation_valid": True
    }
    assert observed == [(str(attestation), str(authorization))]

    monkeypatch.delenv(
        "CURE_LITE_V24_RUNTIME_ATTESTATION_PATH"
    )
    with pytest.raises(PermissionError, match="attestation path"):
        adapter._verify_runtime_launch(str(authorization))


def test_preaccess_rejects_reserved_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CURE_LITE_V24_RUNTIME_ATTESTATION_PATH",
        "/tmp/forged-attestation.json",
    )

    with pytest.raises(PermissionError, match="outside real mode"):
        adapter._reject_reserved_runtime_environment()


def test_delegated_cli_inherits_every_r2_path_and_function_globals() -> None:
    code = f"""
import sys
sys.path.insert(0, {str(REPOSITORY)!r})
from tools import run_cure_lite_v24_gcr_pacre_dr_gate_r2 as adapter
summary = adapter.configure_r2_execution_identity()
from tools import run_cure_lite_v24_gcr_pacre_dr_gate as delegated
expected = {{
    name: transition["r2"]
    for name, transition in summary["identity_transition"].items()
}}
for name in (
    "GCR_PACRE_DR_ACCESS_AUDIT_PATH",
    "GCR_PACRE_DR_PREACCESS_AUTHORIZATION_PATH",
    "GCR_PACRE_DR_RECEIPT_PATH",
):
    assert getattr(delegated, name) == expected[name]
gate_globals = delegated.run_gcr_pacre_dr_gate.__globals__
for name in (
    "GCR_PACRE_DR_GATE_SCHEMA",
    "GCR_PACRE_DR_RUN_ID",
    "GCR_PACRE_DR_PREACCESS_SCHEMA",
    "GCR_PACRE_DR_PREACCESS_STAGE_ID",
    "GCR_PACRE_DR_PREACCESS_STATUS",
    "GCR_PACRE_DR_ACCESS_AUDIT_SCHEMA",
    "GCR_PACRE_DR_RUN_START_SCHEMA",
):
    assert gate_globals[name] == expected[name]
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr


def test_adapter_refuses_legacy_cli_preimport_and_second_transition() -> None:
    code = f"""
import sys
sys.path.insert(0, {str(REPOSITORY)!r})
from tools import run_cure_lite_v24_gcr_pacre_dr_gate
from tools import run_cure_lite_v24_gcr_pacre_dr_gate_r2 as adapter
try:
    adapter.configure_r2_execution_identity()
except RuntimeError as error:
    assert "imported before" in str(error)
else:
    raise AssertionError("legacy pre-import unexpectedly succeeded")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr

    second_code = f"""
import sys
sys.path.insert(0, {str(REPOSITORY)!r})
from tools import run_cure_lite_v24_gcr_pacre_dr_gate_r2 as adapter
adapter.configure_r2_execution_identity()
try:
    adapter.configure_r2_execution_identity()
except RuntimeError as error:
    assert "identity constants changed" in str(error)
else:
    raise AssertionError("second transition unexpectedly succeeded")
"""
    second = subprocess.run(
        [sys.executable, "-I", "-c", second_code],
        cwd=REPOSITORY,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert second.returncode == 0, second.stderr


def test_private_r2_evidence_is_sealed_and_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (
        tmp_path
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
    )
    root.mkdir(parents=True, mode=0o700)
    os.chmod(root, 0o700)
    path = root / "evidence.json"
    body = {"schema_version": "test-r2-evidence-v1", "payload": False}
    payload = {
        **body,
        "receipt_fingerprint": adapter.stable_fingerprint(body),
    }
    path.write_text(adapter.canonical_json(payload) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)
    monkeypatch.setattr(adapter, "REPOSITORY", tmp_path)

    observed = adapter._verify_or_seal_evidence(
        path,
        fingerprint_field="receipt_fingerprint",
        seal=True,
    )

    assert observed == payload
    assert path.stat().st_mode & 0o777 == 0o444
    assert adapter._verify_or_seal_evidence(
        path,
        fingerprint_field="receipt_fingerprint",
        seal=False,
    ) == payload


def test_r2_evidence_rejects_nonprivate_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = (
        tmp_path
        / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
    )
    root.mkdir(parents=True, mode=0o755)
    os.chmod(root, 0o755)
    path = root / "evidence.json"
    body = {"schema_version": "test-r2-evidence-v1"}
    path.write_text(
        adapter.canonical_json(
            {
                **body,
                "receipt_fingerprint": adapter.stable_fingerprint(body),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(adapter, "REPOSITORY", tmp_path)

    with pytest.raises(PermissionError, match="mode 0700"):
        adapter._verify_or_seal_evidence(
            path,
            fingerprint_field="receipt_fingerprint",
            seal=True,
        )
