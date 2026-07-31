from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import socket
import stat
from types import SimpleNamespace

import pytest

from tools import cure_lite_v24_runtime_environment as environment


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "cure_lite_v24_runtime_environment.py"
)
GPU_UUID = "GPU-12cdabd0-7910-8f4a-e4d7-e3c7867d1296"
OTHER_GPU_UUID = "GPU-48b3f9d5-25d4-2398-5483-ee6bd406b655"


def _fingerprint(label: str) -> str:
    return environment.stable_fingerprint({"label": label})


def _lease_body() -> dict[str, object]:
    return {
        "schema_version": environment.GPU_LEASE_SCHEMA,
        "created_at_utc": "2026-07-30T00:00:00Z",
        "boot_id": "12345678-1234-1234-1234-123456789abc",
        "gpu_uuid": GPU_UUID,
        "runtime_spec_fingerprint": _fingerprint("runtime-spec"),
        "attempt_id": "generated-test-attempt",
        "authorization_fingerprint": _fingerprint("authorization"),
        "planned_attempt_commit_fingerprint": _fingerprint("attempt-commit"),
        "committer_pid": 123,
        "committer_starttime": 456,
    }


def _gpu_release_receipt_path(
    tmp_path: Path,
    name: str = "released.json",
) -> Path:
    parent = tmp_path / "gpu-release-receipts"
    parent.mkdir(mode=0o700, exist_ok=True)
    return (parent / name).resolve()


def _process(
    *,
    pid: int = 321,
    starttime: int = 12345,
    uid: int | None = None,
    cgroup: str = (
        "/user.slice/user-1000.slice/user@1000.service/"
        "app.slice/allowed.service/child"
    ),
) -> environment.ProcessIdentity:
    return environment.ProcessIdentity(
        pid=pid,
        starttime_ticks=starttime,
        uid=os.getuid() if uid is None else uid,
        cgroup_path=cgroup,
        argv=("/usr/bin/generated", "--no-payload"),
    )


def _manager_identity(
    uid: int | None = None,
) -> environment.ProcessIdentity:
    selected_uid = os.getuid() if uid is None else uid
    return environment.ProcessIdentity(
        pid=123,
        starttime_ticks=456,
        uid=selected_uid,
        cgroup_path=(
            f"/user.slice/user-{selected_uid}.slice/"
            f"user@{selected_uid}.service/init.scope"
        ),
        argv=("/usr/lib/systemd/systemd", "--user"),
    )



def _endpoint(uid: int) -> dict[str, object]:
    runtime = f"/run/user/{uid}"
    return {
        "uid": uid,
        "runtime_directory": runtime,
        "runtime_directory_device": 11,
        "runtime_directory_inode": 12,
        "bus_path": f"{runtime}/bus",
        "bus_device": 11,
        "bus_inode": 13,
    }



def test_module_is_stdlib_only_and_cli_is_audit_only() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", 1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])
    assert "torch" not in imported_roots
    assert "cure_lite" not in imported_roots
    assert "cure_lite_v24" not in imported_roots

    parser = environment.build_parser()
    assert parser.parse_args(
        [
            "audit-only",
            "--output",
            "/tmp/generated-environment-receipt.json",
            "--selected-gpu-index",
            "0",
        ]
    ).command == "audit-only"
    with pytest.raises(SystemExit):
        parser.parse_args(["cleanup"])


def test_fixed_manager_environment_and_command_allowlist_do_not_inherit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="running\n", stderr="")

    monkeypatch.setattr(environment.subprocess, "run", fake_run)
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:path=/tmp/hostile")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    result = environment.run_read_only_command(
        (
            environment.SYSTEMCTL_PATH,
            "--user",
            "is-system-running",
            "--no-pager",
        ),
        uid=1234,
    )
    assert result.returncode == 0
    assert captured["shell"] is False
    assert captured["env"] == environment.fixed_user_manager_environment(1234)
    assert "CUDA_VISIBLE_DEVICES" not in captured["env"]
    assert captured["env"]["DBUS_SESSION_BUS_ADDRESS"] == (
        "unix:path=/run/user/1234/bus"
    )

    with pytest.raises(PermissionError, match="audit-only"):
        environment.run_read_only_command(
            (environment.SYSTEMCTL_PATH, "--user", "stop", "x.service")
        )
    with pytest.raises(PermissionError, match="exact metadata"):
        environment.run_read_only_command(
            (environment.NVIDIA_SMI_PATH, "-pm", "1")
        )


def test_user_manager_endpoint_requires_private_owned_socket(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / str(os.getuid())
    runtime.mkdir(mode=0o700)
    bus = runtime / "bus"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(bus))
        receipt = environment.validate_user_manager_endpoint(
            run_user_root=tmp_path,
        )
        assert receipt["uid"] == os.getuid()
        assert receipt["bus_path"] == str(bus)
        assert receipt["runtime_directory"] == str(runtime)
        assert receipt["runtime_directory_device"] == runtime.stat().st_dev
        assert receipt["runtime_directory_inode"] == runtime.stat().st_ino
        os.chmod(runtime, 0o755)
        with pytest.raises(PermissionError, match="0700"):
            environment.validate_user_manager_endpoint(
                run_user_root=tmp_path,
            )
    finally:
        listener.close()


def test_nvidia_smi_parsers_are_strict_and_preserve_physical_identity() -> None:
    devices = environment.parse_gpu_inventory(
        "0, GPU-12cdabd0-7910-8f4a-e4d7-e3c7867d1296, "
        "00000000:02:00.0, Default, [N/A], 580.126.09\n"
        "1, GPU-48b3f9d5-25d4-2398-5483-ee6bd406b655, "
        "00000000:82:00.0, Exclusive_Process, Enabled, 580.126.09\n"
    )
    assert [row.index for row in devices] == [0, 1]
    assert devices[0].uuid == GPU_UUID
    assert devices[0].pci_bus_id == "00000000:02:00.0"
    assert devices[0].mig_mode is None
    assert devices[0].minor_number is None
    assert devices[0].mps_state == "unknown"

    apps = environment.parse_gpu_processes(
        f"321, {GPU_UUID}, python, 1193\n"
    )
    assert apps == (
        environment.GPUProcess(
            pid=321,
            gpu_uuid=GPU_UUID,
            process_name="python",
            used_gpu_memory_mib=1193,
        ),
    )
    assert environment.parse_gpu_processes("") == ()
    with pytest.raises(ValueError, match="row width"):
        environment.parse_gpu_inventory("0, too, few\n")
    with pytest.raises(ValueError, match="duplicate"):
        environment.parse_gpu_processes(
            f"321, {GPU_UUID}, python, 1\n"
            f"321, {GPU_UUID}, python, 2\n"
        )


def test_proc_identity_read_detects_starttime_and_unified_cgroup(
    tmp_path: Path,
) -> None:
    pid_root = tmp_path / "321"
    pid_root.mkdir()
    fields = ["S"] + ["0"] * 18 + ["12345"]
    (pid_root / "stat").write_text(
        "321 (generated worker) " + " ".join(fields) + "\n",
        encoding="ascii",
    )
    (pid_root / "status").write_text(
        f"Name:\tgenerated\nUid:\t{os.getuid()}\t{os.getuid()}\t"
        f"{os.getuid()}\t{os.getuid()}\n",
        encoding="ascii",
    )
    (pid_root / "cgroup").write_text(
        "0::/user.slice/user-1000.slice/test.service/child\n",
        encoding="utf-8",
    )
    (pid_root / "cmdline").write_bytes(
        b"/usr/bin/generated\x00--audit-only\x00"
    )
    identity = environment.read_process_identity(321, proc_root=tmp_path)
    assert identity.starttime_ticks == 12345
    assert identity.cgroup_path.endswith("/test.service/child")
    assert identity.argv == ("/usr/bin/generated", "--audit-only")


def test_boot_id_and_user_manager_identity_are_bound_from_procfs(
    tmp_path: Path,
) -> None:
    boot_source = tmp_path / "boot_id"
    boot_source.write_text(
        "12345678-1234-1234-1234-123456789abc\n",
        encoding="ascii",
    )
    assert environment.read_boot_id(source=boot_source).endswith("9abc")

    uid = os.getuid()
    manager = _manager_identity(uid)
    members = (
        tmp_path
        / "user.slice"
        / f"user-{uid}.slice"
        / f"user@{uid}.service"
        / "init.scope"
        / "cgroup.procs"
    )
    members.parent.mkdir(parents=True)
    members.write_text("123\n124\n", encoding="ascii")
    helper = environment.ProcessIdentity(
        pid=124,
        starttime_ticks=999,
        uid=uid,
        cgroup_path=manager.cgroup_path,
        argv=("(sd-pam)",),
    )
    identities = {123: manager, 124: helper}
    calls: list[int] = []

    def reader(pid: int) -> environment.ProcessIdentity:
        calls.append(pid)
        return identities[pid]

    resolved = environment.read_user_manager_identity(
        uid,
        cgroup_root=tmp_path,
        process_reader=reader,
    )
    assert resolved == manager
    assert calls == [123, 124, 123]
def test_cgroup_mapping_and_gpu_double_snapshot_fail_closed() -> None:
    device = environment.GPUDevice(
        index=0,
        uuid=GPU_UUID,
        pci_bus_id="00000000:02:00.0",
        compute_mode="Default",
        mig_mode=None,
        driver_version="580.126.09",
        minor_number=0,
        mps_state="not_observed",
    )
    app = environment.GPUProcess(
        pid=321,
        gpu_uuid=GPU_UUID,
        process_name="python",
        used_gpu_memory_mib=10,
    )
    process = _process()
    groups = {
        "allowed.service": (
            "/user.slice/user-1000.slice/user@1000.service/"
            "app.slice/allowed.service"
        )
    }
    assert environment.map_process_to_user_unit(
        process,
        groups,
        expected_uid=os.getuid(),
    ) == "allowed.service"
    passed = environment.verify_gpu_double_snapshot(
        devices=(device,),
        first_apps=(app,),
        first_processes=(process,),
        second_processes=(process,),
        second_apps=(app,),
        selected_gpu_uuid=GPU_UUID,
        expected_uid=os.getuid(),
        allowed_unit_ids=("allowed.service",),
        unit_control_groups=groups,
    )
    assert passed["passed"] is True
    assert passed["blockers"] == []

    changed = _process(starttime=12346)
    failed = environment.verify_gpu_double_snapshot(
        devices=(device,),
        first_apps=(app,),
        first_processes=(process,),
        second_processes=(changed,),
        second_apps=(app,),
        selected_gpu_uuid=GPU_UUID,
        expected_uid=os.getuid(),
        allowed_unit_ids=("allowed.service",),
        unit_control_groups=groups,
    )
    assert failed["passed"] is False
    assert "pid_identity_changed:321" in failed["blockers"]

    empty = environment.verify_gpu_double_snapshot(
        devices=(device,),
        first_apps=(),
        first_processes=(),
        second_processes=(),
        second_apps=(),
        selected_gpu_uuid=GPU_UUID,
        expected_uid=os.getuid(),
        allowed_unit_ids=(),
        unit_control_groups={},
    )
    assert empty["passed"] is True


def test_nonselected_gpu_consumers_are_observed_unless_global_strict() -> None:
    devices = (
        environment.GPUDevice(
            index=0,
            uuid=GPU_UUID,
            pci_bus_id="00000000:02:00.0",
            compute_mode="Default",
            mig_mode=None,
            driver_version="580.126.09",
            minor_number=0,
            mps_state="not_observed",
        ),
        environment.GPUDevice(
            index=1,
            uuid=OTHER_GPU_UUID,
            pci_bus_id="00000000:82:00.0",
            compute_mode="Default",
            mig_mode=None,
            driver_version="580.126.09",
            minor_number=1,
            mps_state="not_observed",
        ),
    )
    other_app = environment.GPUProcess(
        pid=999,
        gpu_uuid=OTHER_GPU_UUID,
        process_name="unrelated",
        used_gpu_memory_mib=1,
    )
    arguments = {
        "devices": devices,
        "first_apps": (other_app,),
        "first_processes": (),
        "second_processes": (),
        "second_apps": (other_app,),
        "selected_gpu_uuid": GPU_UUID,
        "expected_uid": os.getuid(),
        "allowed_unit_ids": (),
        "unit_control_groups": {},
    }
    scoped = environment.verify_gpu_double_snapshot(**arguments)
    assert scoped["passed"] is True
    assert scoped["blockers"] == []
    assert "missing_proc_identity:999" in scoped["observations"]

    strict = environment.verify_gpu_double_snapshot(
        **arguments,
        strict_all_gpu_consumers=True,
    )
    assert strict["passed"] is False
    assert "missing_proc_identity:999" in strict["blockers"]


def test_collector_orders_gpu_proc_gpu_proc_and_binds_manager_endpoint() -> None:
    events: list[str] = []
    app_text = f"321, {GPU_UUID}, python, 10\n"

    def runner(argv) -> environment.CommandResult:
        materialized = tuple(argv)
        if "is-system-running" in materialized:
            stdout = "degraded\n"
        elif "--failed" in materialized:
            stdout = (
                "unrelated.service loaded failed failed generated\n"
            )
        elif materialized == environment.GPU_QUERY_ARGV:
            stdout = (
                f"0, {GPU_UUID}, 00000000:02:00.0, Default, "
                "[N/A], 580.126.09\n"
            )
        elif materialized == environment.GPU_APPS_QUERY_ARGV:
            events.append("gpu-apps")
            stdout = app_text
        elif "list-units" in materialized:
            events.append("list-units")
            stdout = "allowed.service loaded active running generated\n"
        elif "show" in materialized:
            if "LoadState" in materialized:
                stdout = (
                    "Id=allowed.service\n"
                    "LoadState=loaded\n"
                    "ActiveState=activating\n"
                    "SubState=auto-restart\n"
                    "UnitFileState=enabled\n"
                    "Restart=on-failure\n"
                    "RestartUSec=1000000\n"
                    "NRestarts=10\n"
                    "ControlGroup=\n"
                    "FragmentPath=/tmp/generated.service\n"
                    "DropInPaths=\n"
                    "TriggeredBy=\n"
                    "Triggers=\n"
                    "WantedBy=\n"
                    "RequiredBy=\n"
                    "PartOf=\n"
                )
            else:
                stdout = (
                    "Id=allowed.service\n"
                    "ControlGroup=/user.slice/user-1000.slice/"
                    "allowed.service\n"
                )
        else:
            raise AssertionError(materialized)
        return environment.CommandResult(
            argv=materialized,
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    identity = _process(
        cgroup="/user.slice/user-1000.slice/allowed.service/child"
    )

    def process_reader(pid: int) -> environment.ProcessIdentity:
        assert pid == 321
        events.append("proc")
        return identity

    inventory = environment.collect_environment_inventory(
        selected_gpu_index=0,
        allowed_unit_ids=("allowed.service",),
        command_runner=runner,
        process_reader=process_reader,
        endpoint_validator=_endpoint,
        boot_id_reader=lambda: "12345678-1234-1234-1234-123456789abc",
        manager_identity_reader=_manager_identity,
        driver_metadata_binder=lambda devices: tuple(
            environment.GPUDevice(
                **{
                    **environment.asdict(row),
                    "minor_number": row.index,
                }
            )
            for row in devices
        ),
        mps_detector=lambda: "not_observed",
        allowed_failed_unit_ids=("unrelated.service",),
    )
    assert inventory["passed"] is True
    assert events == [
        "gpu-apps",
        "proc",
        "list-units",
        "gpu-apps",
        "proc",
    ]
    assert inventory["manager"]["endpoint"]["uid"] == os.getuid()
    assert inventory["manager"]["state"] == "degraded"
    assert inventory["manager"]["unexpected_failed_unit_ids"] == []
    assert inventory["boot_id"].endswith("9abc")
    assert inventory["manager"]["identity"]["pid"] == 123
    assert inventory["manager"]["identity"]["starttime_ticks"] == 456
    assert inventory["manager"]["identity"]["control_group"].endswith("init.scope")

    blocked = environment.collect_environment_inventory(
        selected_gpu_index=0,
        allowed_unit_ids=("allowed.service",),
        conflict_unit_ids=("allowed.service",),
        command_runner=runner,
        process_reader=process_reader,
        endpoint_validator=_endpoint,
        boot_id_reader=lambda: "12345678-1234-1234-1234-123456789abc",
        manager_identity_reader=_manager_identity,
        driver_metadata_binder=lambda devices: tuple(
            environment.GPUDevice(
                **{
                    **environment.asdict(row),
                    "minor_number": row.index,
                }
            )
            for row in devices
        ),
        mps_detector=lambda: "not_observed",
        allowed_failed_unit_ids=("unrelated.service",),
    )
    assert blocked["passed"] is False
    assert (
        "scoped_blocker_unit_not_quiescent:allowed.service"
        in blocked["blockers"]
    )


def test_gpu_driver_minor_binding_and_unknown_mps_are_fail_closed(
    tmp_path: Path,
) -> None:
    device = environment.parse_gpu_inventory(
        f"0, {GPU_UUID}, 00000000:02:00.0, Default, [N/A], 580.126.09\n"
    )[0]
    information = tmp_path / "0000:02:00.0" / "information"
    information.parent.mkdir()
    information.write_text(
        f"GPU UUID: {GPU_UUID}\nDevice Minor: 7\n",
        encoding="utf-8",
    )
    bound = environment.bind_gpu_driver_metadata(
        (device,),
        driver_root=tmp_path,
    )
    assert bound[0].minor_number == 7

    snapshot = environment.verify_gpu_double_snapshot(
        devices=bound,
        first_apps=(),
        first_processes=(),
        second_processes=(),
        second_apps=(),
        selected_gpu_uuid=GPU_UUID,
        expected_uid=os.getuid(),
        allowed_unit_ids=(),
        unit_control_groups={},
    )
    assert snapshot["passed"] is False
    assert "selected_gpu_mps_state_unknown" in snapshot["blockers"]


def test_create_once_receipt_is_immutable_and_refuses_replay(
    tmp_path: Path,
) -> None:
    target = (tmp_path / "receipt.json").resolve()
    payload = environment.write_create_once_receipt(
        target,
        {
            "schema_version": "generated-receipt-v1",
            "passed": True,
        },
    )
    assert target.stat().st_mode & 0o777 == 0o444
    assert target.stat().st_nlink == 1
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert payload["receipt_fingerprint"] == environment.stable_fingerprint(
        {
            "schema_version": "generated-receipt-v1",
            "passed": True,
        }
    )
    with pytest.raises(FileExistsError):
        environment.write_create_once_receipt(
            target,
            {"schema_version": "generated-receipt-v1", "passed": False},
        )

    source = tmp_path / "source"
    source.write_text("do-not-replace", encoding="utf-8")
    alias = tmp_path / "alias.json"
    os.symlink(source, alias)
    with pytest.raises(FileExistsError):
        environment.write_create_once_receipt(
            alias,
            {"schema_version": "generated-receipt-v1"},
        )


def test_receipt_selfcheck_rejects_owner_uid_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actual_uid = os.getuid()
    getuid_results = iter((actual_uid, actual_uid + 1))
    monkeypatch.setattr(
        environment.os, "getuid", lambda: next(getuid_results)
    )
    target = (tmp_path / "owner-drift.json").resolve()
    with pytest.raises(RuntimeError, match="self-verification"):
        environment.write_create_once_receipt(
            target,
            {"schema_version": "generated-receipt-v1"},
        )
    assert target.exists()


def test_create_once_receipt_parent_generation_swap_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "receipt-parent"
    parent.mkdir(mode=0o700)
    replacement = tmp_path / "receipt-parent-replacement"
    replacement.mkdir(mode=0o700)
    displaced = tmp_path / "receipt-parent-displaced"
    target = (parent / "receipt.json").resolve()
    parent_identity = (parent.stat().st_dev, parent.stat().st_ino)
    real_fsync = os.fsync
    swapped = False

    def swap_after_parent_fsync(descriptor: int) -> None:
        nonlocal swapped
        real_fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not swapped
            and stat.S_ISDIR(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino) == parent_identity
        ):
            parent.rename(displaced)
            replacement.rename(parent)
            swapped = True

    monkeypatch.setattr(environment.os, "fsync", swap_after_parent_fsync)
    with pytest.raises(PermissionError, match="generation changed"):
        environment.write_create_once_receipt(
            target,
            {"schema_version": "generated-receipt-v1"},
        )
    assert swapped is True
    assert not target.exists()
    assert (displaced / target.name).is_file()


def test_load_receipt_rejects_parent_generation_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "receipt-parent"
    parent.mkdir(mode=0o700)
    replacement = tmp_path / "receipt-parent-replacement"
    replacement.mkdir(mode=0o700)
    displaced = tmp_path / "receipt-parent-displaced"
    target = (parent / "receipt.json").resolve()
    environment.write_create_once_receipt(
        target,
        {"schema_version": "generated-receipt-v1"},
    )
    original_inode = target.stat().st_ino
    real_read = os.read
    swapped = False

    def swap_during_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        metadata = os.fstat(descriptor)
        if (
            not swapped
            and stat.S_ISREG(metadata.st_mode)
            and metadata.st_ino == original_inode
        ):
            parent.rename(displaced)
            replacement.rename(parent)
            swapped = True
        return real_read(descriptor, count)

    monkeypatch.setattr(environment.os, "read", swap_during_read)
    with pytest.raises(PermissionError, match="generation changed"):
        environment.load_sealed_receipt_with_evidence(target)
    assert swapped is True
    assert not target.exists()
    assert (displaced / target.name).is_file()


def test_load_receipt_rejects_same_bytes_replacement_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (tmp_path / "receipt.json").resolve()
    environment.write_create_once_receipt(
        target,
        {"schema_version": "generated-receipt-v1", "passed": True},
    )
    encoded = target.read_bytes()
    original_inode = target.stat().st_ino
    held_descriptor = os.open(target, os.O_RDONLY)
    real_read = os.read
    replaced = False

    def replace_during_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        metadata = os.fstat(descriptor)
        if (
            not replaced
            and descriptor != held_descriptor
            and metadata.st_ino == original_inode
        ):
            target.unlink()
            replacement_descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o444,
            )
            try:
                assert os.write(replacement_descriptor, encoded) == len(
                    encoded
                )
                os.fsync(replacement_descriptor)
            finally:
                os.close(replacement_descriptor)
            replaced = True
        return real_read(descriptor, count)

    monkeypatch.setattr(environment.os, "read", replace_during_read)
    try:
        with pytest.raises(
            PermissionError,
            match="changed while being read",
        ):
            environment.load_sealed_receipt_with_evidence(target)
    finally:
        os.close(held_descriptor)
    assert replaced is True
    assert target.read_bytes() == encoded
    assert target.stat().st_ino != original_inode


def test_sealed_receipt_evidence_rejects_bool_identity(
    tmp_path: Path,
) -> None:
    target = (tmp_path / "receipt.json").resolve()
    environment.write_create_once_receipt(
        target,
        {"schema_version": "generated-receipt-v1"},
    )
    _, evidence = environment.load_sealed_receipt_with_evidence(target)
    invalid = dict(evidence)
    invalid["inode"] = True
    with pytest.raises(ValueError, match="evidence is malformed"):
        environment.verify_sealed_receipt_evidence(target, invalid)


def test_create_once_writer_preserves_primary_and_closes_both_fds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (tmp_path / "receipt.json").resolve()
    real_close = os.close
    close_kinds: list[str] = []

    def close_leaf_with_secondary_error(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        kind = "directory" if stat.S_ISDIR(metadata.st_mode) else "leaf"
        close_kinds.append(kind)
        real_close(descriptor)
        if kind == "leaf":
            raise OSError("generated secondary leaf close failure")

    def fail_guard(
        receipt_descriptor: int,
        receipt_parent_descriptor: int,
        receipt_metadata: os.stat_result,
        receipt_parent_metadata: os.stat_result,
    ) -> None:
        assert os.fstat(receipt_descriptor).st_ino == receipt_metadata.st_ino
        assert (
            os.fstat(receipt_parent_descriptor).st_ino
            == receipt_parent_metadata.st_ino
        )
        raise PermissionError("generated primary guard failure")

    monkeypatch.setattr(
        environment.os,
        "close",
        close_leaf_with_secondary_error,
    )
    with pytest.raises(
        PermissionError,
        match="generated primary guard failure",
    ):
        environment.write_create_once_receipt(
            target,
            {"schema_version": "generated-receipt-v1"},
            while_open_guard=fail_guard,
        )
    assert close_kinds[-2:] == ["leaf", "directory"]
    assert target.is_file()


def test_create_once_writer_reports_first_close_error_after_both_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = (tmp_path / "receipt.json").resolve()
    real_close = os.close
    close_kinds: list[str] = []
    leaf_close_attempted = False

    def close_then_fail(descriptor: int) -> None:
        nonlocal leaf_close_attempted
        metadata = os.fstat(descriptor)
        kind = "directory" if stat.S_ISDIR(metadata.st_mode) else "leaf"
        close_kinds.append(kind)
        real_close(descriptor)
        if kind == "leaf":
            leaf_close_attempted = True
            raise OSError("generated first leaf close failure")
        if leaf_close_attempted:
            raise OSError("generated second parent close failure")

    monkeypatch.setattr(environment.os, "close", close_then_fail)
    with pytest.raises(
        OSError,
        match="generated first leaf close failure",
    ):
        environment.write_create_once_receipt(
            target,
            {"schema_version": "generated-receipt-v1"},
        )
    assert close_kinds[-2:] == ["leaf", "directory"]
    assert target.is_file()


def test_gpu_lease_is_o_excl_locked_and_released_via_tombstone(
    tmp_path: Path,
) -> None:
    lease = (tmp_path / "gpu-uuid.lease").resolve()
    handle = environment.acquire_gpu_lease(lease, _lease_body())
    lease_descriptor = handle.descriptor
    parent_descriptor = handle.parent_descriptor
    parent_metadata = lease.parent.stat()
    assert (handle.parent_device, handle.parent_inode) == (
        parent_metadata.st_dev,
        parent_metadata.st_ino,
    )
    assert lease.stat().st_mode & 0o777 == 0o600
    assert handle.payload["lease_fingerprint"] == (
        environment.stable_fingerprint(_lease_body())
    )
    assert "planned_attempt_commit_fingerprint" in handle.payload
    with pytest.raises(FileExistsError):
        environment.acquire_gpu_lease(lease, _lease_body())

    second_descriptor = os.open(lease, os.O_RDONLY)
    try:
        with pytest.raises(BlockingIOError):
            environment.fcntl.flock(
                second_descriptor,
                environment.fcntl.LOCK_EX | environment.fcntl.LOCK_NB,
            )
    finally:
        os.close(second_descriptor)

    tombstone = (tmp_path / "gpu-uuid.released.lease").resolve()
    receipt_path = _gpu_release_receipt_path(
        tmp_path,
        "lease-release.json",
    )
    receipt = environment.release_gpu_lease_to_tombstone(
        handle,
        tombstone_path=tombstone,
        release_receipt_path=receipt_path,
        release_kind="committed_terminal",
        attempt_consumed=True,
        evidence_fingerprint=_fingerprint("terminal"),
    )
    assert handle.closed is True
    assert not lease.exists()
    assert tombstone.is_file()
    assert tombstone.stat().st_mode & 0o777 == 0o444
    assert receipt["active_lease_absent"] is True
    assert receipt["tombstone_file_sha256"] == environment.file_sha256(
        tombstone
    )
    assert (receipt["tombstone_device"], receipt["tombstone_inode"]) == (
        tombstone.stat().st_dev,
        tombstone.stat().st_ino,
    )
    assert (
        receipt["lease_parent_device"],
        receipt["lease_parent_inode"],
    ) == (parent_metadata.st_dev, parent_metadata.st_ino)
    assert environment.validate_gpu_lease_release_receipt(receipt) == receipt
    assert receipt_path.stat().st_mode & 0o777 == 0o444
    with pytest.raises(OSError):
        os.fstat(lease_descriptor)
    with pytest.raises(OSError):
        os.fstat(parent_descriptor)
    with pytest.raises(RuntimeError, match="already closed"):
        environment.release_gpu_lease_to_tombstone(
            handle,
            tombstone_path=tmp_path / "again.lease",
            release_receipt_path=_gpu_release_receipt_path(
                tmp_path,
                "again.json",
            ),
            release_kind="committed_terminal",
            attempt_consumed=True,
            evidence_fingerprint=_fingerprint("terminal"),
        )


def test_gpu_lease_release_requires_distinct_receipt_directory(
    tmp_path: Path,
) -> None:
    lease = (tmp_path / "gpu-uuid.lease").resolve()
    handle = environment.acquire_gpu_lease(lease, _lease_body())
    with pytest.raises(ValueError, match="different directory"):
        environment.release_gpu_lease_to_tombstone(
            handle,
            tombstone_path=tmp_path / "released.lease",
            release_receipt_path=tmp_path / "released.json",
            release_kind="uncommitted_forensic",
            attempt_consumed=False,
            evidence_fingerprint=_fingerprint("no-commit"),
        )
    assert handle.closed is False
    assert lease.is_file()
    assert not (tmp_path / "released.lease").exists()
    handle.close_without_release()


def test_gpu_lease_acquire_parent_generation_swap_preserves_partial_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "lease-parent"
    parent.mkdir(mode=0o700)
    replacement = tmp_path / "lease-parent-replacement"
    replacement.mkdir(mode=0o700)
    displaced = tmp_path / "lease-parent-displaced"
    lease = (parent / "gpu-uuid.lease").resolve()
    parent_identity = (parent.stat().st_dev, parent.stat().st_ino)
    real_fsync = os.fsync
    swapped = False

    def swap_after_parent_fsync(descriptor: int) -> None:
        nonlocal swapped
        real_fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not swapped
            and stat.S_ISDIR(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino) == parent_identity
        ):
            parent.rename(displaced)
            replacement.rename(parent)
            swapped = True

    monkeypatch.setattr(environment.os, "fsync", swap_after_parent_fsync)
    with pytest.raises(PermissionError, match="generation changed"):
        environment.acquire_gpu_lease(lease, _lease_body())
    assert swapped is True
    assert not lease.exists()
    assert (displaced / lease.name).is_file()


def test_gpu_lease_acquire_preserves_primary_and_closes_both_fds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = (tmp_path / "gpu-uuid.lease").resolve()
    real_verify = environment._verify_parent_path_generation
    verification_count = 0

    def fail_final_parent_verification(*args: object, **kwargs: object):
        nonlocal verification_count
        verification_count += 1
        if verification_count == 2:
            raise PermissionError("generated acquire primary failure")
        return real_verify(*args, **kwargs)

    real_close = os.close
    close_kinds: list[str] = []

    def close_lease_with_secondary_error(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        kind = "directory" if stat.S_ISDIR(metadata.st_mode) else "lease"
        close_kinds.append(kind)
        real_close(descriptor)
        if kind == "lease":
            raise OSError("generated secondary lease close failure")

    monkeypatch.setattr(
        environment,
        "_verify_parent_path_generation",
        fail_final_parent_verification,
    )
    monkeypatch.setattr(
        environment.os,
        "close",
        close_lease_with_secondary_error,
    )
    with pytest.raises(
        PermissionError,
        match="generated acquire primary failure",
    ):
        environment.acquire_gpu_lease(lease, _lease_body())
    assert close_kinds[-2:] == ["lease", "directory"]
    assert lease.is_file()


def test_gpu_lease_tombstone_hash_uses_still_open_lease_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = (tmp_path / "gpu-uuid.lease").resolve()
    tombstone = (tmp_path / "released.lease").resolve()
    handle = environment.acquire_gpu_lease(lease, _lease_body())

    def reject_path_hash(path: str | Path) -> str:
        raise AssertionError(f"unexpected post-close path hash:{path}")

    monkeypatch.setattr(environment, "file_sha256", reject_path_hash)
    receipt = environment.release_gpu_lease_to_tombstone(
        handle,
        tombstone_path=tombstone,
        release_receipt_path=_gpu_release_receipt_path(tmp_path),
        release_kind="uncommitted_forensic",
        attempt_consumed=False,
        evidence_fingerprint=_fingerprint("no-commit"),
    )
    assert receipt["tombstone_file_sha256"] == hashlib.sha256(
        tombstone.read_bytes()
    ).hexdigest()


def test_gpu_lease_release_cross_checks_both_open_parent_generations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease_parent = tmp_path / "lease-parent"
    lease_parent.mkdir(mode=0o700)
    lease = (lease_parent / "gpu-uuid.lease").resolve()
    tombstone = (lease_parent / "released.lease").resolve()
    receipt_path = _gpu_release_receipt_path(tmp_path)
    handle = environment.acquire_gpu_lease(lease, _lease_body())
    real_fsync = os.fsync
    fsynced_identities: set[tuple[int, int, int]] = set()

    def record_fsync(descriptor: int) -> None:
        real_fsync(descriptor)
        metadata = os.fstat(descriptor)
        fsynced_identities.add(
            (descriptor, metadata.st_dev, metadata.st_ino)
        )

    real_writer = environment.write_create_once_receipt
    guard_observed = False

    def inspect_writer(*args: object, **kwargs: object):
        nonlocal guard_observed
        original_guard = kwargs["while_open_guard"]

        def inspect_guard(
            receipt_descriptor: int,
            receipt_parent_descriptor: int,
            receipt_metadata: os.stat_result,
            receipt_parent_metadata: os.stat_result,
        ) -> None:
            nonlocal guard_observed
            assert (
                receipt_descriptor,
                receipt_metadata.st_dev,
                receipt_metadata.st_ino,
            ) in fsynced_identities
            assert (
                receipt_parent_descriptor,
                receipt_parent_metadata.st_dev,
                receipt_parent_metadata.st_ino,
            ) in fsynced_identities
            assert os.fstat(handle.descriptor).st_ino == handle.inode
            assert (
                os.fstat(handle.parent_descriptor).st_ino
                == handle.parent_inode
            )
            assert (
                receipt_parent_metadata.st_dev,
                receipt_parent_metadata.st_ino,
            ) != (handle.parent_device, handle.parent_inode)
            guard_observed = True
            original_guard(
                receipt_descriptor,
                receipt_parent_descriptor,
                receipt_metadata,
                receipt_parent_metadata,
            )

        kwargs["while_open_guard"] = inspect_guard
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(environment.os, "fsync", record_fsync)
    monkeypatch.setattr(
        environment,
        "write_create_once_receipt",
        inspect_writer,
    )
    environment.release_gpu_lease_to_tombstone(
        handle,
        tombstone_path=tombstone,
        release_receipt_path=receipt_path,
        release_kind="uncommitted_forensic",
        attempt_consumed=False,
        evidence_fingerprint=_fingerprint("no-commit"),
    )
    assert guard_observed is True
    assert handle.closed is True


def test_gpu_lease_post_close_named_verification_rejects_inode_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease_parent = tmp_path / "lease-parent"
    lease_parent.mkdir(mode=0o700)
    lease = (lease_parent / "gpu-uuid.lease").resolve()
    tombstone = (lease_parent / "released.lease").resolve()
    receipt_path = _gpu_release_receipt_path(tmp_path)
    handle = environment.acquire_gpu_lease(lease, _lease_body())
    original_inode = handle.inode
    real_post_close_verify = (
        environment._verify_gpu_lease_tombstone_after_close
    )
    hook_observed = False

    def replace_then_verify(**kwargs: object) -> None:
        nonlocal hook_observed
        assert handle.closed is True
        encoded = tombstone.read_bytes()
        held_descriptor = os.open(tombstone, os.O_RDONLY)
        try:
            tombstone.unlink()
            replacement_descriptor = os.open(
                tombstone,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o444,
            )
            try:
                assert os.write(replacement_descriptor, encoded) == len(
                    encoded
                )
                os.fsync(replacement_descriptor)
            finally:
                os.close(replacement_descriptor)
            assert tombstone.stat().st_ino != original_inode
            hook_observed = True
            real_post_close_verify(**kwargs)
        finally:
            os.close(held_descriptor)

    monkeypatch.setattr(
        environment,
        "_verify_gpu_lease_tombstone_after_close",
        replace_then_verify,
    )
    with pytest.raises(
        PermissionError,
        match="post-close tombstone closure changed",
    ):
        environment.release_gpu_lease_to_tombstone(
            handle,
            tombstone_path=tombstone,
            release_receipt_path=receipt_path,
            release_kind="uncommitted_forensic",
            attempt_consumed=False,
            evidence_fingerprint=_fingerprint("no-commit"),
        )
    assert hook_observed is True
    assert handle.closed is True
    assert receipt_path.is_file()


def test_gpu_lease_release_parent_swap_is_fail_closed_on_saved_dirfd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "lease-parent"
    parent.mkdir(mode=0o700)
    replacement = tmp_path / "lease-parent-replacement"
    replacement.mkdir(mode=0o700)
    displaced = tmp_path / "lease-parent-displaced"
    lease = (parent / "gpu-uuid.lease").resolve()
    tombstone = (parent / "gpu-uuid.released.lease").resolve()
    handle = environment.acquire_gpu_lease(lease, _lease_body())
    real_rename_noreplace = environment._rename_noreplace
    swapped = False

    def rename_then_swap(*args: object) -> None:
        nonlocal swapped
        real_rename_noreplace(*args)
        parent.rename(displaced)
        replacement.rename(parent)
        swapped = True

    monkeypatch.setattr(
        environment,
        "_rename_noreplace",
        rename_then_swap,
    )
    with pytest.raises(PermissionError, match="generation changed"):
        environment.release_gpu_lease_to_tombstone(
            handle,
            tombstone_path=tombstone,
            release_receipt_path=_gpu_release_receipt_path(tmp_path),
            release_kind="uncommitted_forensic",
            attempt_consumed=False,
            evidence_fingerprint=_fingerprint("no-commit"),
        )
    assert swapped is True
    assert handle.closed is False
    assert not tombstone.exists()
    assert (displaced / tombstone.name).is_file()
    assert not (displaced / lease.name).exists()
    handle.close_without_release()


def test_gpu_lease_stays_locked_when_receipt_parent_generation_swaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease_parent = tmp_path / "lease-parent"
    lease_parent.mkdir(mode=0o700)
    receipt_parent = tmp_path / "receipt-parent"
    receipt_parent.mkdir(mode=0o700)
    receipt_replacement = tmp_path / "receipt-parent-replacement"
    receipt_replacement.mkdir(mode=0o700)
    receipt_displaced = tmp_path / "receipt-parent-displaced"
    lease = (lease_parent / "gpu-uuid.lease").resolve()
    tombstone = (lease_parent / "released.lease").resolve()
    receipt_path = (receipt_parent / "released.json").resolve()
    handle = environment.acquire_gpu_lease(lease, _lease_body())
    receipt_parent_identity = (
        receipt_parent.stat().st_dev,
        receipt_parent.stat().st_ino,
    )
    real_fsync = os.fsync
    swapped = False

    def swap_after_receipt_parent_fsync(descriptor: int) -> None:
        nonlocal swapped
        real_fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not swapped
            and stat.S_ISDIR(metadata.st_mode)
            and (metadata.st_dev, metadata.st_ino)
            == receipt_parent_identity
        ):
            receipt_parent.rename(receipt_displaced)
            receipt_replacement.rename(receipt_parent)
            swapped = True

    monkeypatch.setattr(
        environment.os,
        "fsync",
        swap_after_receipt_parent_fsync,
    )
    with pytest.raises(PermissionError, match="generation changed"):
        environment.release_gpu_lease_to_tombstone(
            handle,
            tombstone_path=tombstone,
            release_receipt_path=receipt_path,
            release_kind="uncommitted_forensic",
            attempt_consumed=False,
            evidence_fingerprint=_fingerprint("no-commit"),
        )
    assert swapped is True
    assert handle.closed is False
    assert tombstone.is_file()
    assert not receipt_path.exists()
    assert (receipt_displaced / receipt_path.name).is_file()
    competing_descriptor = os.open(tombstone, os.O_RDONLY)
    try:
        with pytest.raises(BlockingIOError):
            environment.fcntl.flock(
                competing_descriptor,
                environment.fcntl.LOCK_EX | environment.fcntl.LOCK_NB,
            )
    finally:
        os.close(competing_descriptor)
    handle.close_without_release()


def test_gpu_lease_close_attempts_both_fds_after_unlock_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = (tmp_path / "gpu-uuid.lease").resolve()
    handle = environment.acquire_gpu_lease(lease, _lease_body())
    lease_descriptor = handle.descriptor
    parent_descriptor = handle.parent_descriptor
    real_close = os.close
    closed: list[int] = []

    def fail_unlock(descriptor: int, operation: int) -> None:
        raise OSError("generated unlock failure")

    def record_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(environment.fcntl, "flock", fail_unlock)
    monkeypatch.setattr(environment.os, "close", record_close)
    with pytest.raises(OSError, match="generated unlock failure"):
        handle.close_without_release()
    assert closed == [lease_descriptor, parent_descriptor]
    assert handle.closed is True
    handle.close_without_release()
    assert closed == [lease_descriptor, parent_descriptor]


@pytest.mark.parametrize("failing_descriptor_name", ["lease", "parent"])
def test_gpu_lease_close_attempts_both_fds_after_close_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_descriptor_name: str,
) -> None:
    lease = (tmp_path / "gpu-uuid.lease").resolve()
    handle = environment.acquire_gpu_lease(lease, _lease_body())
    lease_descriptor = handle.descriptor
    parent_descriptor = handle.parent_descriptor
    failing_descriptor = {
        "lease": lease_descriptor,
        "parent": parent_descriptor,
    }[failing_descriptor_name]
    real_close = os.close
    closed: list[int] = []

    def close_then_report_error(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)
        if descriptor == failing_descriptor:
            raise OSError(f"generated {failing_descriptor_name} close failure")

    monkeypatch.setattr(
        environment.os,
        "close",
        close_then_report_error,
    )
    with pytest.raises(
        OSError,
        match=f"generated {failing_descriptor_name} close failure",
    ):
        handle.close_without_release()
    assert closed == [lease_descriptor, parent_descriptor]
    assert handle.closed is True
    handle.close_without_release()
    assert closed == [lease_descriptor, parent_descriptor]


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("tombstone_device", True),
        ("tombstone_inode", 0),
        ("lease_parent_device", "1"),
        ("lease_parent_inode", -1),
    ],
)
def test_gpu_lease_release_receipt_rejects_identity_types(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    lease = (tmp_path / "gpu-uuid.lease").resolve()
    handle = environment.acquire_gpu_lease(lease, _lease_body())
    receipt = environment.release_gpu_lease_to_tombstone(
        handle,
        tombstone_path=tmp_path / "released.lease",
        release_receipt_path=_gpu_release_receipt_path(tmp_path),
        release_kind="uncommitted_forensic",
        attempt_consumed=False,
        evidence_fingerprint=_fingerprint("no-commit"),
    )
    mutated = dict(receipt)
    mutated[field] = invalid
    body = dict(mutated)
    body.pop("receipt_fingerprint")
    mutated["receipt_fingerprint"] = environment.stable_fingerprint(body)
    with pytest.raises(ValueError, match="body is malformed"):
        environment.validate_gpu_lease_release_receipt(mutated)


def test_gpu_lease_release_rejects_hardlink_identity_drift(
    tmp_path: Path,
) -> None:
    lease = (tmp_path / "gpu-uuid.lease").resolve()
    handle = environment.acquire_gpu_lease(lease, _lease_body())
    alias = tmp_path / "lease-hardlink"
    os.link(lease, alias)
    with pytest.raises(RuntimeError, match="identity changed"):
        environment.release_gpu_lease_to_tombstone(
            handle,
            tombstone_path=tmp_path / "released.lease",
            release_receipt_path=_gpu_release_receipt_path(tmp_path),
            release_kind="uncommitted_forensic",
            attempt_consumed=False,
            evidence_fingerprint=_fingerprint("no-commit"),
        )
    assert lease.exists()
    assert handle.closed is False
    alias.unlink()
    environment.release_gpu_lease_to_tombstone(
        handle,
        tombstone_path=tmp_path / "released.lease",
        release_receipt_path=_gpu_release_receipt_path(tmp_path),
        release_kind="uncommitted_forensic",
        attempt_consumed=False,
        evidence_fingerprint=_fingerprint("no-commit"),
    )


def test_audit_only_cli_writes_failure_or_pass_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = {
        "schema_version": environment.ENVIRONMENT_INVENTORY_SCHEMA,
        "boot_id": "12345678-1234-1234-1234-123456789abc",
        "manager": {
            "endpoint": _endpoint(os.getuid()),
            "identity": {
                "pid": 123,
                "starttime_ticks": 456,
                "uid": os.getuid(),
                "control_group": _manager_identity().cgroup_path,
            },
        },
        "passed": True,
        "inventory_fingerprint": _fingerprint("inventory"),
    }
    monkeypatch.setattr(
        environment,
        "collect_environment_inventory",
        lambda **kwargs: inventory,
    )
    output = (tmp_path / "audit.json").resolve()
    assert environment.main(
        [
            "audit-only",
            "--output",
            str(output),
            "--selected-gpu-index",
            "0",
        ]
    ) == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["passed"] is True
    assert receipt["inventory"] == inventory
    assert receipt["environment_binding"]["boot_id"].endswith("9abc")
    assert receipt["environment_binding"]["runtime_directory_device"] == 11
    assert receipt["environment_binding"]["runtime_directory_inode"] == 12
    assert receipt["environment_binding"]["manager_identity"]["pid"] == 123
    assert receipt["D_R_payload_accessed"] is False
    assert receipt["D_V_payload_accessed"] is False
    assert receipt["D_T_payload_accessed"] is False
    with pytest.raises(FileExistsError):
        environment.main(
            [
                "audit-only",
                "--output",
                str(output),
                "--selected-gpu-index",
                "0",
            ]
        )


STABILITY_TARGET = "generated-target.service"
STABILITY_CONFLICT = "generated-gpu0-conflict.service"
STABILITY_FAILED = ("generated-gpu2.service", "generated-gpu3.service")


def _stability_inventory(
    *,
    target_nrestarts: str = "0",
    allowed_unit_ids: tuple[str, ...] = (),
    selected_consumer: bool = False,
    target_triggered_by: str = "",
    postcleanup: bool = False,
    conflict_unit: str = STABILITY_CONFLICT,
    conflict_nrestarts: str = "18425",
    postcleanup_unit_file_state: str = "masked-runtime",
) -> dict[str, object]:
    uid = os.getuid()
    endpoint = _endpoint(uid)
    identity = _manager_identity(uid)

    def shadow(unit: str) -> dict[str, str]:
        conflict = unit == conflict_unit
        return {
            "Id": unit,
            "LoadState": "loaded",
            "ActiveState": "inactive" if postcleanup or not conflict else "activating",
            "SubState": "dead" if postcleanup or not conflict else "auto-restart",
            "UnitFileState": (
                postcleanup_unit_file_state
                if postcleanup and conflict
                else ("enabled" if conflict else "static")
            ),
            "Restart": "on-failure" if conflict else "no",
            "RestartUSec": "30s" if conflict else "0",
            "NRestarts": conflict_nrestarts if conflict else target_nrestarts,
            "ControlGroup": "",
            "FragmentPath": f"/tmp/{unit}",
            "DropInPaths": "",
            "TriggeredBy": "" if conflict else target_triggered_by,
            "Triggers": "",
            "WantedBy": (
                "default.target"
                if (
                    conflict
                    and conflict_unit == environment.GPU0_CONFLICT_UNIT
                )
                else ""
            ),
            "RequiredBy": "",
            "PartOf": "",
        }

    shadows = {
        STABILITY_TARGET: shadow(STABILITY_TARGET),
        conflict_unit: shadow(conflict_unit),
    }
    device = environment.GPUDevice(
        index=0,
        uuid=GPU_UUID,
        pci_bus_id="00000000:02:00.0",
        compute_mode="Default",
        mig_mode=None,
        driver_version="580.126.09",
        minor_number=0,
        mps_state="not_observed",
    )
    apps = (
        [{
            "pid": 777,
            "gpu_uuid": GPU_UUID,
            "process_name": "allowed-before-stability",
            "used_gpu_memory_mib": 1,
        }]
        if selected_consumer else []
    )
    mappings = (
        [{
            "pid": 777,
            "starttime_ticks": 12345,
            "uid": uid,
            "gpu_uuid": GPU_UUID,
            "cgroup_path": (
                f"/user.slice/user-{uid}.slice/user@{uid}.service/"
                f"app.slice/{conflict_unit}/child"
            ),
            "unit_id": conflict_unit,
        }]
        if selected_consumer else []
    )
    gpu_body: dict[str, object] = {
        "schema_version": environment.GPU_DOUBLE_SNAPSHOT_SCHEMA,
        "selected_gpu_uuid": GPU_UUID,
        "expected_uid": uid,
        "allowed_unit_ids": list(allowed_unit_ids),
        "strict_all_gpu_consumers": False,
        "devices": [environment.asdict(device)],
        "first_apps": apps,
        "second_apps": apps,
        "process_unit_mapping": mappings,
        "observations": [],
        "blockers": [],
        "passed": True,
    }
    gpu = {
        **gpu_body,
        "snapshot_fingerprint": environment.stable_fingerprint(gpu_body),
    }
    body: dict[str, object] = {
        "schema_version": environment.ENVIRONMENT_INVENTORY_SCHEMA,
        "created_at_utc": "2026-07-30T00:00:00Z",
        "uid": uid,
        "boot_id": "12345678-1234-1234-1234-123456789abc",
        "manager": {
            "state": "degraded",
            "allowed_states": ["running", "degraded"],
            "returncode": 1,
            "failed_units": list(STABILITY_FAILED),
            "allowed_failed_unit_ids": list(STABILITY_FAILED),
            "unexpected_failed_unit_ids": [],
            "scoped_failed_unit_ids": [],
            "identity": {
                "pid": identity.pid,
                "starttime_ticks": identity.starttime_ticks,
                "uid": uid,
                "control_group": identity.cgroup_path,
            },
            "endpoint": endpoint,
        },
        "unit_scope": {
            "target_unit_id": STABILITY_TARGET,
            "conflict_unit_ids": [conflict_unit],
            "dependency_unit_ids": [],
            "require_target_ready": False,
            "shadows": shadows,
        },
        "gpu_snapshot": gpu,
        "blockers": (
            [] if postcleanup else
            [f"scoped_blocker_unit_not_quiescent:{conflict_unit}"]
        ),
        "passed": postcleanup,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return {
        **body,
        "inventory_fingerprint": environment.stable_fingerprint(body),
    }


def _write_stability_roots(
    tmp_path: Path,
    inventory: dict[str, object],
    *,
    conflict_unit: str = STABILITY_CONFLICT,
    cleanup_mode: str = environment.NORMAL_CLEANUP_MODE,
    activation_guard: dict[str, object] | None = None,
    partial_lineage: dict[str, object] | None = None,
    after_unit_file_state: str | None = None,
) -> tuple[Path, Path]:
    endpoint = inventory["manager"]["endpoint"]
    identity = inventory["manager"]["identity"]
    pre_body: dict[str, object] = {
        "schema_version": environment.ENVIRONMENT_RECEIPT_SCHEMA,
        "created_at_utc": "2026-07-30T00:00:01Z",
        "command": "audit-only",
        "environment_binding": {
            "inventory_fingerprint": inventory["inventory_fingerprint"],
            "boot_id": inventory["boot_id"],
            "runtime_directory": endpoint["runtime_directory"],
            "runtime_directory_device": endpoint["runtime_directory_device"],
            "runtime_directory_inode": endpoint["runtime_directory_inode"],
            "manager_identity": identity,
        },
        "inventory": inventory,
        "passed": inventory["passed"],
        "error_type": None,
        "error_message": None,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    pre_path = (tmp_path / "precleanup.json").resolve()
    environment.write_create_once_receipt(pre_path, pre_body)
    post_conflict = dict(
        inventory["unit_scope"]["shadows"][conflict_unit]
    )
    unit_file_state = after_unit_file_state or (
        "enabled"
        if cleanup_mode == environment.RECOVERY_CLEANUP_MODE
        else "masked-runtime"
    )
    post_conflict.update(
        {
            "UnitFileState": unit_file_state,
            "ActiveState": "inactive",
            "SubState": "dead",
        }
    )
    if activation_guard is None:
        activation_guard = {
            "mode": environment.NORMAL_GUARD_MODE,
            "unit_name": conflict_unit,
            "observed_unit_file_state": "masked-runtime",
        }
    cleanup_body: dict[str, object] = {
        "schema_version": environment.CLEANUP_RECEIPT_SCHEMA,
        "created_at_utc": "2026-07-30T00:00:02Z",
        "intent_fingerprint": _fingerprint("cleanup-intent"),
        "action_receipt_fingerprints": (
            [_fingerprint("recovery-action")]
            if cleanup_mode == environment.RECOVERY_CLEANUP_MODE
            else [_fingerprint("mask-action"), _fingerprint("stop-action")]
        ),
        "boot_id": inventory["boot_id"],
        "manager_generation": {
            "boot_id": inventory["boot_id"],
            "identity": identity,
            "endpoint": endpoint,
        },
        "after": {
            conflict_unit: post_conflict,
        },
        "cleanup_mode": cleanup_mode,
        "activation_guard": activation_guard,
        "partial_lineage": partial_lineage,
        "passed": True,
        "payload_authority": "none",
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    cleanup_path = (tmp_path / "cleanup.json").resolve()
    environment.write_create_once_receipt(
        cleanup_path,
        cleanup_body,
        fingerprint_field="cleanup_receipt_fingerprint",
    )
    return pre_path, cleanup_path


def _write_partial_lineage(tmp_path: Path) -> dict[str, object]:
    fingerprint_fields = {
        "plan": "plan_fingerprint",
        "original_authorization": "authorization_fingerprint",
        "original_intent": "intent_fingerprint",
        "original_terminal_failure": "terminal_failure_fingerprint",
        "recovery_authorization": "recovery_authorization_fingerprint",
        "recovery_intent": "recovery_intent_fingerprint",
        "recovery_action_receipt": "recovery_action_receipt_fingerprint",
    }
    roots: dict[str, object] = {}
    for name, fingerprint_field in fingerprint_fields.items():
        path = (tmp_path / f"{name}.json").resolve()
        payload = environment.write_create_once_receipt(
            path,
            {
                "schema_version": f"generated-{name}-v1",
                "label": name,
            },
            fingerprint_field=fingerprint_field,
        )
        roots[name] = {
            "path": str(path),
            "file_sha256": environment.file_sha256(path),
            "fingerprint_field": fingerprint_field,
            "fingerprint": payload[fingerprint_field],
        }
    return {
        **roots,
        "legacy_runtime_mask_may_remain_false_reconciled": True,
        "original_stop_dispatched": False,
    }


def _guard_observation(
    guard: dict[str, object],
) -> dict[str, object]:
    return {**guard, "file_type": "symlink"}


def _stability_contract(
    inventory: dict[str, object],
) -> environment.EnvironmentAuditContract:
    return environment.environment_audit_contract_from_inventory(
        inventory,
        selected_gpu_index=0,
        target_unit_id=STABILITY_TARGET,
        conflict_unit_ids=(STABILITY_CONFLICT,),
        dependency_unit_ids=(),
        allowed_failed_unit_ids=STABILITY_FAILED,
        cleanup_mode=environment.NORMAL_CLEANUP_MODE,
        quiescence_mode=environment.NORMAL_QUIESCENCE_MODE,
        cleanup_nrestarts_baseline=(
            (
                STABILITY_CONFLICT,
                inventory["unit_scope"]["shadows"][STABILITY_CONFLICT][
                    "NRestarts"
                ],
            ),
        ),
        activation_guard={
            "mode": environment.NORMAL_GUARD_MODE,
            "unit_name": STABILITY_CONFLICT,
            "observed_unit_file_state": "masked-runtime",
        },
        allowed_unit_ids=tuple(inventory["gpu_snapshot"]["allowed_unit_ids"]),
    )



def _write_stability_policy(
    tmp_path: Path,
    pre_path: Path,
    cleanup_path: Path,
    inventory: dict[str, object],
    *,
    sample_count: int = 2,
    interval_seconds: float = 30.0,
) -> tuple[Path, dict[str, object]]:
    _, evidence = environment.load_sealed_receipt_with_evidence(pre_path)
    _, cleanup_evidence = environment.load_sealed_receipt_with_evidence(
        cleanup_path,
        fingerprint_field="cleanup_receipt_fingerprint",
    )
    policy = environment.build_environment_policy(
        _stability_contract(inventory),
        precleanup_root_binding={
            **evidence,
            "inventory_fingerprint": inventory["inventory_fingerprint"],
        },
        cleanup_root_binding=cleanup_evidence,
        toolchain_binding=environment.current_runtime_toolchain_binding(),
        minimum_sample_count=sample_count,
        sample_interval_seconds=interval_seconds,
    )
    policy_path = (tmp_path / "policy.json").resolve()
    environment.write_environment_policy(policy_path, policy)
    return policy_path, policy


def _write_policy_from_contract(
    tmp_path: Path,
    contract: environment.EnvironmentAuditContract,
    roots: dict[str, object],
    *,
    name: str,
) -> tuple[Path, dict[str, object]]:
    policy = environment.build_environment_policy(
        contract,
        precleanup_root_binding=roots["precleanup_inventory_receipt"],
        cleanup_root_binding=roots["cleanup_receipt"],
        toolchain_binding=environment.current_runtime_toolchain_binding(),
        minimum_sample_count=2,
        sample_interval_seconds=30.0,
    )
    path = (tmp_path / name).resolve()
    environment.write_environment_policy(path, policy)
    return path, policy


def _recovery_policy_audit_fixture(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    Path,
    Path,
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    conflict = environment.GPU0_CONFLICT_UNIT
    pre_inventory = _stability_inventory(conflict_unit=conflict)
    lineage = _write_partial_lineage(tmp_path)
    guard = {
        "mode": environment.RECOVERY_GUARD_MODE,
        "unit_name": conflict,
        "path": f"/run/user/{os.getuid()}/systemd/user/{conflict}",
        "target": "/dev/null",
        "owner_uid": os.getuid(),
        "device": 501,
        "inode": 502,
        "observed_unit_file_state": "enabled",
    }
    pre_path, cleanup_path = _write_stability_roots(
        tmp_path,
        pre_inventory,
        conflict_unit=conflict,
        cleanup_mode=environment.RECOVERY_CLEANUP_MODE,
        activation_guard=guard,
        partial_lineage=lineage,
    )
    contract, roots = environment.prepare_environment_stability_contract(
        pre_path,
        cleanup_path,
        selected_gpu_index=0,
        target_unit_id=STABILITY_TARGET,
        conflict_unit_ids=(conflict,),
        dependency_unit_ids=(),
        allowed_failed_unit_ids=STABILITY_FAILED,
        activation_guard_reader=_guard_observation,
    )
    policy_path, policy = _write_policy_from_contract(
        tmp_path,
        contract,
        roots,
        name="policy-bound-recovery.json",
    )
    post_inventory = _stability_inventory(
        conflict_unit=conflict,
        postcleanup=True,
        postcleanup_unit_file_state="enabled",
    )
    sampled_result = environment.run_environment_stability_gate(
        pre_path,
        cleanup_path,
        selected_gpu_index=0,
        target_unit_id=STABILITY_TARGET,
        conflict_unit_ids=(conflict,),
        dependency_unit_ids=(),
        allowed_failed_unit_ids=STABILITY_FAILED,
        sample_count=policy["sampling"]["minimum_sample_count"],
        sample_interval_seconds=policy["sampling"]["sample_interval_seconds"],
        policy_path=policy_path,
        inventory_collector=lambda **kwargs: post_inventory,
        activation_guard_reader=_guard_observation,
        sleeper=lambda seconds: None,
        monotonic_clock=iter((600.0, 630.0)).__next__,
    )
    assert sampled_result["passed"] is True
    stability_path = (tmp_path / "sealed-stability.json").resolve()
    stability_body = dict(sampled_result)
    stability_body.pop("stability_receipt_fingerprint")
    environment.write_create_once_receipt(
        stability_path,
        stability_body,
        fingerprint_field="stability_receipt_fingerprint",
    )
    return (
        pre_path,
        cleanup_path,
        policy_path,
        stability_path,
        policy,
        post_inventory,
        sampled_result,
    )


def _policy_bound_audit_argv(
    output: Path,
    pre_path: Path,
    cleanup_path: Path,
    policy_path: Path,
    stability_path: Path,
    *,
    target_unit: str = STABILITY_TARGET,
) -> list[str]:
    argv = [
        "audit-only",
        "--output",
        str(output),
        "--selected-gpu-index",
        "0",
        "--target-unit",
        target_unit,
        "--conflict-unit",
        environment.GPU0_CONFLICT_UNIT,
    ]
    for unit in STABILITY_FAILED:
        argv.extend(("--allow-failed-unit", unit))
    argv.extend(
        (
            "--policy",
            str(policy_path),
            "--precleanup-inventory-receipt",
            str(pre_path),
            "--cleanup-receipt",
            str(cleanup_path),
            "--stability-receipt",
            str(stability_path),
        )
    )
    return argv


def _reseal_sampled_test_evidence(
    value: dict[str, object],
    *,
    sample_indexes: tuple[int, ...],
    reseal_gpu_snapshot: bool = True,
) -> None:
    for index in sample_indexes:
        sample = value["samples"][index]
        inventory = sample["inventory"]
        if reseal_gpu_snapshot:
            gpu = inventory["gpu_snapshot"]
            gpu_body = dict(gpu)
            gpu_body.pop("snapshot_fingerprint")
            gpu["snapshot_fingerprint"] = environment.stable_fingerprint(
                gpu_body
            )
        inventory_body = dict(inventory)
        inventory_body.pop("inventory_fingerprint")
        inventory["inventory_fingerprint"] = environment.stable_fingerprint(
            inventory_body
        )
        sample_body = dict(sample)
        sample_body.pop("single_audit_fingerprint")
        sample["single_audit_fingerprint"] = environment.stable_fingerprint(
            sample_body
        )
    stability_body = dict(value)
    stability_body.pop("stability_receipt_fingerprint")
    value["stability_receipt_fingerprint"] = (
        environment.stable_fingerprint(stability_body)
    )


def test_audit_only_policy_bound_recovery_contract_passes_and_uses_last_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        pre_path,
        cleanup_path,
        policy_path,
        stability_path,
        policy,
        post_inventory,
        sampled_result,
    ) = _recovery_policy_audit_fixture(tmp_path)
    root_verify_calls: list[tuple[str, str]] = []
    toolchain_rechecks: list[dict[str, object]] = []
    single_audit_validations: list[str] = []
    original_root_verifier = environment.verify_sealed_receipt_evidence
    original_toolchain_reader = environment.current_runtime_toolchain_binding
    original_single_validator = environment.validate_environment_single_audit

    def root_verifier(path, evidence, *, fingerprint_field="receipt_fingerprint"):
        root_verify_calls.append((str(Path(path).absolute()), fingerprint_field))
        return original_root_verifier(
            path,
            evidence,
            fingerprint_field=fingerprint_field,
        )

    def toolchain_reader() -> dict[str, object]:
        value = original_toolchain_reader()
        toolchain_rechecks.append(value)
        return value

    def single_validator(value, *, contract):
        single_audit_validations.append(value["single_audit_fingerprint"])
        return original_single_validator(value, contract=contract)

    monkeypatch.setattr(
        environment,
        "run_environment_stability_gate",
        lambda *args, **kwargs: pytest.fail(
            "policy-bound audit must consume the sealed stability receipt"
        ),
    )
    monkeypatch.setattr(
        environment,
        "verify_sealed_receipt_evidence",
        root_verifier,
    )
    monkeypatch.setattr(
        environment,
        "current_runtime_toolchain_binding",
        toolchain_reader,
    )
    monkeypatch.setattr(
        environment,
        "validate_environment_single_audit",
        single_validator,
    )
    monkeypatch.setattr(
        environment,
        "collect_environment_inventory",
        lambda **kwargs: pytest.fail(
            "normal audit inventory collector must not be used"
        ),
    )
    output = (tmp_path / "policy-bound-postcleanup.json").resolve()
    assert environment.main(
        _policy_bound_audit_argv(
            output,
            pre_path,
            cleanup_path,
            policy_path,
            stability_path,
        )
    ) == 0
    receipt = environment.load_sealed_receipt(output)
    assert receipt["schema_version"] == environment.ENVIRONMENT_RECEIPT_SCHEMA
    assert receipt["command"] == "audit-only"
    assert receipt["passed"] is True
    assert receipt["error_type"] is None
    assert receipt["error_message"] is None
    assert receipt["inventory"] == post_inventory
    assert receipt["inventory"]["passed"] is True
    assert receipt["inventory"]["blockers"] == []
    assert receipt["environment_binding"]["inventory_fingerprint"] == (
        post_inventory["inventory_fingerprint"]
    )
    assert all(
        receipt[field] is False
        for field in (
            "D_R_payload_accessed",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
        )
    )
    assert single_audit_validations == [
        sample["single_audit_fingerprint"]
        for sample in sampled_result["samples"]
    ] + [
        sample["single_audit_fingerprint"]
        for sample in sampled_result["samples"]
    ] + [sampled_result["samples"][-1]["single_audit_fingerprint"]]
    assert root_verify_calls == [
        (str(pre_path), "receipt_fingerprint"),
        (str(cleanup_path), "cleanup_receipt_fingerprint"),
        (str(policy_path), "policy_fingerprint"),
        (str(stability_path), "stability_receipt_fingerprint"),
        (str(pre_path), "receipt_fingerprint"),
        (str(cleanup_path), "cleanup_receipt_fingerprint"),
        (str(policy_path), "policy_fingerprint"),
        (str(stability_path), "stability_receipt_fingerprint"),
    ]
    assert len(toolchain_rechecks) == 5
    assert all(value == policy["toolchain"] for value in toolchain_rechecks)
    with pytest.raises(FileExistsError):
        environment.main(
            _policy_bound_audit_argv(
                output,
                pre_path,
                cleanup_path,
                policy_path,
                stability_path,
            )
        )
    assert environment.load_sealed_receipt(output) == receipt


@pytest.mark.parametrize(
    "tamper_kind",
    ("nested_inventory", "root_evidence"),
)
def test_audit_only_rejects_sampled_data_not_cross_bound_to_policy_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
) -> None:
    (
        pre_path,
        cleanup_path,
        policy_path,
        _stability_path,
        _policy,
        _post_inventory,
        sampled_result,
    ) = _recovery_policy_audit_fixture(tmp_path)
    tampered = json.loads(json.dumps(sampled_result))
    if tamper_kind == "nested_inventory":
        last_sample = tampered["samples"][-1]
        last_sample["inventory"]["inventory_fingerprint"] = "0" * 64
        sample_body = dict(last_sample)
        sample_body.pop("single_audit_fingerprint")
        last_sample["single_audit_fingerprint"] = (
            environment.stable_fingerprint(sample_body)
        )
    else:
        tampered["root_evidence"]["precleanup_inventory_receipt"][
            "inventory_fingerprint"
        ] = "f" * 64
    stability_body = dict(tampered)
    stability_body.pop("stability_receipt_fingerprint")
    tampered["stability_receipt_fingerprint"] = (
        environment.stable_fingerprint(stability_body)
    )
    if tamper_kind == "root_evidence":
        # Root evidence is semantically checked against the separately sealed
        # policy by policy-bound audit-only, not by the generic receipt schema.
        assert environment.validate_environment_stability_receipt(tampered)
    else:
        with pytest.raises(
            PermissionError,
            match="sample semantic replay invalid",
        ):
            environment.validate_environment_stability_receipt(tampered)
    tampered_path = (
        tmp_path / f"tampered-{tamper_kind}-stability.json"
    ).resolve()
    tampered_body = dict(tampered)
    tampered_body.pop("stability_receipt_fingerprint")
    environment.write_create_once_receipt(
        tampered_path,
        tampered_body,
        fingerprint_field="stability_receipt_fingerprint",
    )
    monkeypatch.setattr(
        environment,
        "run_environment_stability_gate",
        lambda *args, **kwargs: pytest.fail(
            "policy-bound audit must not rerun stability"
        ),
    )
    output = (tmp_path / f"cross-bound-{tamper_kind}.json").resolve()
    assert environment.main(
        _policy_bound_audit_argv(
            output,
            pre_path,
            cleanup_path,
            policy_path,
            tampered_path,
        )
    ) == 1
    receipt = environment.load_sealed_receipt(output)
    assert receipt["passed"] is False
    assert receipt["inventory"] is None
    assert receipt["environment_binding"] is None
    assert receipt["error_type"] == "PermissionError"
    assert all(
        receipt[field] is False
        for field in (
            "D_R_payload_accessed",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
        )
    )


def test_stability_validator_rejects_resealed_failed_unit_semantic_drift(
    tmp_path: Path,
) -> None:
    (
        _pre_path,
        _cleanup_path,
        _policy_path,
        _stability_path,
        _policy,
        _post_inventory,
        sampled_result,
    ) = _recovery_policy_audit_fixture(tmp_path)
    tampered = json.loads(json.dumps(sampled_result))
    last_sample = tampered["samples"][-1]
    inventory = last_sample["inventory"]
    inventory["manager"]["failed_units"] = [STABILITY_FAILED[0]]
    inventory_body = dict(inventory)
    inventory_body.pop("inventory_fingerprint")
    inventory["inventory_fingerprint"] = environment.stable_fingerprint(
        inventory_body
    )
    sample_body = dict(last_sample)
    sample_body.pop("single_audit_fingerprint")
    last_sample["single_audit_fingerprint"] = environment.stable_fingerprint(
        sample_body
    )
    stability_body = dict(tampered)
    stability_body.pop("stability_receipt_fingerprint")
    tampered["stability_receipt_fingerprint"] = (
        environment.stable_fingerprint(stability_body)
    )
    with pytest.raises(
        PermissionError,
        match="sample semantic replay (?:invalid|changed)",
    ):
        environment.validate_environment_stability_receipt(tampered)


def test_stability_validator_rejects_resealed_cross_sample_nrestarts_drift(
    tmp_path: Path,
) -> None:
    (
        _pre_path,
        _cleanup_path,
        _policy_path,
        _stability_path,
        _policy,
        _post_inventory,
        sampled_result,
    ) = _recovery_policy_audit_fixture(tmp_path)
    tampered = json.loads(json.dumps(sampled_result))
    contract = environment._contract_from_sampled_environment_receipt(
        tampered
    )
    inventory = json.loads(
        json.dumps(tampered["samples"][-1]["inventory"])
    )
    inventory["unit_scope"]["shadows"][
        environment.GPU0_CONFLICT_UNIT
    ]["NRestarts"] = "18426"
    inventory_body = dict(inventory)
    inventory_body.pop("inventory_fingerprint")
    inventory["inventory_fingerprint"] = environment.stable_fingerprint(
        inventory_body
    )
    drifted_sample = environment.audit_environment_once(
        contract,
        inventory_collector=lambda **kwargs: inventory,
        activation_guard_reader=_guard_observation,
    )
    drifted_body = dict(drifted_sample)
    drifted_body.pop("single_audit_fingerprint")
    drifted_body["created_at_utc"] = tampered["samples"][-1][
        "created_at_utc"
    ]
    tampered["samples"][-1] = {
        **drifted_body,
        "single_audit_fingerprint": environment.stable_fingerprint(
            drifted_body
        ),
    }
    stability_body = dict(tampered)
    stability_body.pop("stability_receipt_fingerprint")
    tampered["stability_receipt_fingerprint"] = (
        environment.stable_fingerprint(stability_body)
    )
    with pytest.raises(
        PermissionError,
        match="receipt semantic replay changed",
    ):
        environment.validate_environment_stability_receipt(tampered)


def test_stability_rejects_endpoint_inode_bool_equal_to_integer_one(
    tmp_path: Path,
) -> None:
    (
        _pre_path,
        _cleanup_path,
        _policy_path,
        _stability_path,
        _policy,
        _post_inventory,
        sampled_result,
    ) = _recovery_policy_audit_fixture(tmp_path)
    baseline = json.loads(json.dumps(sampled_result))
    baseline["contract"]["runtime_directory_inode"] = 1
    for sample in baseline["samples"]:
        sample["contract"]["runtime_directory_inode"] = 1
        sample["inventory"]["manager"]["endpoint"][
            "runtime_directory_inode"
        ] = 1
    _reseal_sampled_test_evidence(
        baseline,
        sample_indexes=tuple(range(len(baseline["samples"]))),
    )
    assert environment.validate_environment_stability_receipt(baseline)

    tampered = json.loads(json.dumps(baseline))
    tampered["samples"][-1]["inventory"]["manager"]["endpoint"][
        "runtime_directory_inode"
    ] = True
    _reseal_sampled_test_evidence(
        tampered,
        sample_indexes=(len(tampered["samples"]) - 1,),
    )
    with pytest.raises(
        PermissionError,
        match="sample semantic replay invalid",
    ):
        environment.validate_environment_stability_receipt(tampered)


@pytest.mark.parametrize(
    ("field_path", "invalid_value"),
    [
        (("gpu_snapshot", "devices", 0, "index"), False),
        (("gpu_snapshot", "devices", 0, "minor_number"), False),
        (("gpu_snapshot", "devices", 0, "minor_number"), 0.0),
        (("manager", "identity", "uid"), float(os.getuid())),
        (("manager", "identity", "pid"), 123.0),
        (("unit_scope", "require_target_ready"), 0),
        (
            (
                "unit_scope",
                "shadows",
                environment.GPU0_CONFLICT_UNIT,
                "NRestarts",
            ),
            18425,
        ),
    ],
)
def test_stability_rejects_resealed_inventory_identity_type_coercions(
    tmp_path: Path,
    field_path: tuple[object, ...],
    invalid_value: object,
) -> None:
    (
        _pre_path,
        _cleanup_path,
        _policy_path,
        _stability_path,
        _policy,
        _post_inventory,
        sampled_result,
    ) = _recovery_policy_audit_fixture(tmp_path)
    tampered = json.loads(json.dumps(sampled_result))
    target: object = tampered["samples"][-1]["inventory"]
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = invalid_value
    _reseal_sampled_test_evidence(
        tampered,
        sample_indexes=(len(tampered["samples"]) - 1,),
    )
    with pytest.raises(PermissionError):
        environment.validate_environment_stability_receipt(tampered)


@pytest.mark.parametrize(
    ("mutation", "invalid_value"),
    [
        ("manager_state", "offline"),
        ("manager_returncode", 0),
        ("gpu_observations", [1]),
        ("gpu_observations", ["z", "z", "a"]),
    ],
)
def test_stability_rejects_resealed_manager_and_gpu_semantic_drift(
    tmp_path: Path,
    mutation: str,
    invalid_value: object,
) -> None:
    (
        _pre_path,
        _cleanup_path,
        _policy_path,
        _stability_path,
        _policy,
        _post_inventory,
        sampled_result,
    ) = _recovery_policy_audit_fixture(tmp_path)
    tampered = json.loads(json.dumps(sampled_result))
    inventory = tampered["samples"][-1]["inventory"]
    if mutation == "manager_state":
        inventory["manager"]["state"] = invalid_value
    elif mutation == "manager_returncode":
        inventory["manager"]["returncode"] = invalid_value
    else:
        inventory["gpu_snapshot"]["observations"] = invalid_value
    _reseal_sampled_test_evidence(
        tampered,
        sample_indexes=(len(tampered["samples"]) - 1,),
    )
    with pytest.raises(
        PermissionError,
        match="sample semantic replay invalid",
    ):
        environment.validate_environment_stability_receipt(tampered)


@pytest.mark.parametrize(
    "mutation",
    (
        "unpropagated_mps_blocker",
        "unpropagated_gpu_failure",
        "mapping_without_app",
        "restart_usec_above_contract",
    ),
)
def test_stability_rejects_resealed_inventory_semantic_propagation_gaps(
    tmp_path: Path,
    mutation: str,
) -> None:
    (
        _pre_path,
        _cleanup_path,
        _policy_path,
        _stability_path,
        _policy,
        _post_inventory,
        sampled_result,
    ) = _recovery_policy_audit_fixture(tmp_path)
    tampered = json.loads(json.dumps(sampled_result))
    inventory = tampered["samples"][-1]["inventory"]
    gpu = inventory["gpu_snapshot"]
    if mutation == "unpropagated_mps_blocker":
        gpu["devices"][0]["mps_state"] = "enabled_observed"
    elif mutation == "unpropagated_gpu_failure":
        gpu["blockers"] = ["generated-gpu-failure"]
        gpu["passed"] = False
    elif mutation == "mapping_without_app":
        gpu["process_unit_mapping"] = [{
            "pid": 777,
            "starttime_ticks": 12345,
            "uid": os.getuid(),
            "gpu_uuid": GPU_UUID,
            "cgroup_path": "/generated",
            "unit_id": environment.GPU0_CONFLICT_UNIT,
        }]
    else:
        inventory["unit_scope"]["shadows"][
            environment.GPU0_CONFLICT_UNIT
        ]["RestartUSec"] = "9999s"
    _reseal_sampled_test_evidence(
        tampered,
        sample_indexes=(len(tampered["samples"]) - 1,),
    )
    with pytest.raises(
        PermissionError,
        match="sample semantic replay (?:invalid|changed)",
    ):
        environment.validate_environment_stability_receipt(tampered)


def test_stability_rejects_resealed_nested_gpu_snapshot_fingerprint(
    tmp_path: Path,
) -> None:
    (
        _pre_path,
        _cleanup_path,
        _policy_path,
        _stability_path,
        _policy,
        _post_inventory,
        sampled_result,
    ) = _recovery_policy_audit_fixture(tmp_path)
    tampered = json.loads(json.dumps(sampled_result))
    tampered["samples"][-1]["inventory"]["gpu_snapshot"][
        "snapshot_fingerprint"
    ] = "0" * 64
    _reseal_sampled_test_evidence(
        tampered,
        sample_indexes=(len(tampered["samples"]) - 1,),
        reseal_gpu_snapshot=False,
    )
    with pytest.raises(
        PermissionError,
        match="sample semantic replay invalid",
    ):
        environment.validate_environment_stability_receipt(tampered)


@pytest.mark.parametrize(
    "extra_location",
    (
        "inventory",
        "manager",
        "endpoint",
        "unit_scope",
        "shadow",
        "rogue_shadow",
        "gpu_snapshot",
        "mapping",
    ),
)
def test_stability_rejects_resealed_inventory_schema_extensions(
    tmp_path: Path,
    extra_location: str,
) -> None:
    (
        _pre_path,
        _cleanup_path,
        _policy_path,
        _stability_path,
        _policy,
        _post_inventory,
        sampled_result,
    ) = _recovery_policy_audit_fixture(tmp_path)
    tampered = json.loads(json.dumps(sampled_result))
    inventory = tampered["samples"][-1]["inventory"]
    if extra_location == "inventory":
        inventory["unexpected"] = 1
    elif extra_location == "manager":
        inventory["manager"]["unexpected"] = 1
    elif extra_location == "endpoint":
        inventory["manager"]["endpoint"]["unexpected"] = 1
    elif extra_location == "unit_scope":
        inventory["unit_scope"]["unexpected"] = 1
    elif extra_location == "shadow":
        inventory["unit_scope"]["shadows"][
            environment.GPU0_CONFLICT_UNIT
        ]["unexpected"] = "1"
    elif extra_location == "rogue_shadow":
        rogue = json.loads(json.dumps(
            inventory["unit_scope"]["shadows"][
                environment.GPU0_CONFLICT_UNIT
            ]
        ))
        rogue["Id"] = "rogue.service"
        inventory["unit_scope"]["shadows"]["rogue.service"] = rogue
    elif extra_location == "gpu_snapshot":
        inventory["gpu_snapshot"]["unexpected"] = 1
    else:
        inventory["gpu_snapshot"]["process_unit_mapping"] = [{
            "pid": 777,
            "starttime_ticks": 12345,
            "uid": os.getuid(),
            "gpu_uuid": GPU_UUID,
            "cgroup_path": "/generated",
            "unit_id": environment.GPU0_CONFLICT_UNIT,
            "unexpected": 1,
        }]
    _reseal_sampled_test_evidence(
        tampered,
        sample_indexes=(len(tampered["samples"]) - 1,),
    )
    with pytest.raises(
        PermissionError,
        match="sample semantic replay invalid",
    ):
        environment.validate_environment_stability_receipt(tampered)


def test_inventory_identity_type_validator_accepts_gpu_memory_na_and_checks_mapping(
) -> None:
    inventory = _stability_inventory(
        allowed_unit_ids=(STABILITY_CONFLICT,),
        selected_consumer=True,
        postcleanup=True,
    )
    gpu = inventory["gpu_snapshot"]
    for app in gpu["first_apps"] + gpu["second_apps"]:
        app["used_gpu_memory_mib"] = None
    gpu_body = dict(gpu)
    gpu_body.pop("snapshot_fingerprint")
    gpu["snapshot_fingerprint"] = environment.stable_fingerprint(gpu_body)
    inventory_body = dict(inventory)
    inventory_body.pop("inventory_fingerprint")
    inventory["inventory_fingerprint"] = environment.stable_fingerprint(
        inventory_body
    )
    environment._validate_environment_inventory_identity_types(inventory)

    mapping = gpu["process_unit_mapping"][0]
    mapping["starttime_ticks"] = True
    mapping["uid"] = float(os.getuid())
    gpu_body = dict(gpu)
    gpu_body.pop("snapshot_fingerprint")
    gpu["snapshot_fingerprint"] = environment.stable_fingerprint(gpu_body)
    inventory_body = dict(inventory)
    inventory_body.pop("inventory_fingerprint")
    inventory["inventory_fingerprint"] = environment.stable_fingerprint(
        inventory_body
    )
    with pytest.raises(
        ValueError,
        match="GPU mapping identity",
    ):
        environment._validate_environment_inventory_identity_types(inventory)


@pytest.mark.parametrize(
    "present",
    [
        ("policy",),
        ("precleanup",),
        ("cleanup",),
        ("stability",),
        ("policy", "precleanup"),
        ("policy", "cleanup"),
        ("policy", "stability"),
        ("precleanup", "cleanup"),
        ("precleanup", "stability"),
        ("cleanup", "stability"),
        ("policy", "precleanup", "cleanup"),
        ("policy", "precleanup", "stability"),
        ("policy", "cleanup", "stability"),
        ("precleanup", "cleanup", "stability"),
    ],
)
def test_audit_only_partial_policy_roots_fail_closed_without_normal_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    present: tuple[str, ...],
) -> None:
    monkeypatch.setattr(
        environment,
        "collect_environment_inventory",
        lambda **kwargs: pytest.fail(
            "partial policy arguments must not fall back to normal audit"
        ),
    )
    output = (tmp_path / f"partial-{'-'.join(present)}.json").resolve()
    paths = {
        "policy": tmp_path / "policy.json",
        "precleanup": tmp_path / "precleanup.json",
        "cleanup": tmp_path / "cleanup.json",
        "stability": tmp_path / "stability.json",
    }
    options = {
        "policy": "--policy",
        "precleanup": "--precleanup-inventory-receipt",
        "cleanup": "--cleanup-receipt",
        "stability": "--stability-receipt",
    }
    argv = [
        "audit-only",
        "--output",
        str(output),
        "--selected-gpu-index",
        "0",
    ]
    for name in present:
        argv.extend((options[name], str(paths[name])))
    assert environment.main(argv) == 1
    receipt = environment.load_sealed_receipt(output)
    assert receipt["passed"] is False
    assert receipt["inventory"] is None
    assert receipt["environment_binding"] is None
    assert receipt["error_type"] == "ValueError"
    assert "requires policy" in receipt["error_message"]
    assert all(
        receipt[field] is False
        for field in (
            "D_R_payload_accessed",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
        )
    )


@pytest.mark.parametrize(
    "drift_kind",
    ("policy", "root", "toolchain", "scope"),
)
def test_audit_only_policy_root_toolchain_or_scope_drift_writes_fail_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift_kind: str,
) -> None:
    (
        pre_path,
        cleanup_path,
        policy_path,
        stability_path,
        _policy,
        _post_inventory,
        _sampled_result,
    ) = _recovery_policy_audit_fixture(tmp_path)
    monkeypatch.setattr(
        environment,
        "run_environment_stability_gate",
        lambda *args, **kwargs: pytest.fail(
            "policy-bound audit must not rerun stability"
        ),
    )
    selected_pre_path = pre_path
    selected_policy_path = policy_path
    selected_target = STABILITY_TARGET
    if drift_kind == "policy":
        policy = environment.load_sealed_receipt(
            policy_path,
            fingerprint_field="policy_fingerprint",
        )
        policy_body = dict(policy)
        policy_body.pop("policy_fingerprint")
        policy_body["candidate"] = "drifted-candidate"
        selected_policy_path = (tmp_path / "drifted-policy.json").resolve()
        environment.write_create_once_receipt(
            selected_policy_path,
            policy_body,
            fingerprint_field="policy_fingerprint",
        )
    elif drift_kind == "root":
        precleanup = environment.load_sealed_receipt(pre_path)
        precleanup_body = dict(precleanup)
        precleanup_body.pop("receipt_fingerprint")
        selected_pre_path = (tmp_path / "drifted-precleanup.json").resolve()
        environment.write_create_once_receipt(
            selected_pre_path,
            precleanup_body,
        )
    elif drift_kind == "toolchain":
        toolchain = environment.current_runtime_toolchain_binding()
        drifted_toolchain = json.loads(json.dumps(toolchain))
        drifted_toolchain["runtime_environment"]["file_sha256"] = "0" * 64
        monkeypatch.setattr(
            environment,
            "current_runtime_toolchain_binding",
            lambda: drifted_toolchain,
        )
    else:
        selected_target = "generated-scope-drift.service"

    output = (tmp_path / f"{drift_kind}-drift-audit.json").resolve()
    assert environment.main(
        _policy_bound_audit_argv(
            output,
            selected_pre_path,
            cleanup_path,
            selected_policy_path,
            stability_path,
            target_unit=selected_target,
        )
    ) == 1
    receipt = environment.load_sealed_receipt(output)
    assert receipt["schema_version"] == environment.ENVIRONMENT_RECEIPT_SCHEMA
    assert receipt["command"] == "audit-only"
    assert receipt["passed"] is False
    assert receipt["inventory"] is None
    assert receipt["environment_binding"] is None
    assert isinstance(receipt["error_type"], str)
    assert receipt["error_type"]
    assert isinstance(receipt["error_message"], str)
    assert receipt["error_message"]
    assert all(
        receipt[field] is False
        for field in (
            "D_R_payload_accessed",
            "D_V_payload_accessed",
            "D_T_payload_accessed",
        )
    )


def test_audit_only_without_policy_keeps_single_normal_inventory_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = {
        "schema_version": environment.ENVIRONMENT_INVENTORY_SCHEMA,
        "boot_id": "12345678-1234-1234-1234-123456789abc",
        "manager": {
            "endpoint": _endpoint(os.getuid()),
            "identity": {
                "pid": 123,
                "starttime_ticks": 456,
                "uid": os.getuid(),
                "control_group": _manager_identity().cgroup_path,
            },
        },
        "passed": True,
        "inventory_fingerprint": _fingerprint("ordinary-inventory"),
    }
    calls: list[dict[str, object]] = []

    def normal_collector(**kwargs) -> dict[str, object]:
        calls.append(kwargs)
        return inventory

    monkeypatch.setattr(
        environment,
        "collect_environment_inventory",
        normal_collector,
    )
    monkeypatch.setattr(
        environment,
        "run_environment_stability_gate",
        lambda *args, **kwargs: pytest.fail(
            "ordinary audit must not enter the policy-bound stability gate"
        ),
    )
    output = (tmp_path / "ordinary-audit.json").resolve()
    assert environment.main(
        [
            "audit-only",
            "--output",
            str(output),
            "--selected-gpu-index",
            "0",
        ]
    ) == 0
    receipt = environment.load_sealed_receipt(output)
    assert receipt["inventory"] == inventory
    assert receipt["passed"] is True
    assert len(calls) == 1


def test_environment_policy_is_closed_sealed_and_restart_window_bound(
    tmp_path: Path,
) -> None:
    inventory = _stability_inventory()
    contract = _stability_contract(inventory)
    pre_path, cleanup_path = _write_stability_roots(tmp_path, inventory)
    path, policy = _write_stability_policy(
        tmp_path, pre_path, cleanup_path, inventory
    )
    assert environment.validate_environment_policy(policy) == policy
    assert policy["payload_authority"] == "none"
    assert policy["selected_gpu"]["index"] == 0
    assert policy["unit_scope"]["conflict_unit_ids"] == [STABILITY_CONFLICT]
    assert path.stat().st_mode & 0o777 == 0o444
    assert environment.load_sealed_receipt(
        path, fingerprint_field="policy_fingerprint"
    ) == policy
    mutated = dict(policy)
    mutated["extra"] = True
    with pytest.raises(ValueError, match="closed schema"):
        environment.validate_environment_policy(mutated)
    assert policy["sampling"]["maximum_restart_usec"] == 30_000_000
    assert policy["sampling"]["maximum_trigger_usec"] == 0
    assert policy["sampling"]["required_stability_window_usec"] == 30_000_000
    with pytest.raises(ValueError, match="invariants"):
        environment.build_environment_policy(
            contract,
            precleanup_root_binding=policy["precleanup_root"],
            cleanup_root_binding=policy["cleanup_root"],
            toolchain_binding=policy["toolchain"],
            minimum_sample_count=2,
            sample_interval_seconds=29.0,
        )


def test_stability_gate_uses_actual_thirty_second_monotonic_window(
    tmp_path: Path,
) -> None:
    pre_inventory = _stability_inventory()
    post_inventory = _stability_inventory(postcleanup=True)
    pre_path, cleanup_path = _write_stability_roots(tmp_path, pre_inventory)
    policy_path, policy = _write_stability_policy(
        tmp_path, pre_path, cleanup_path, pre_inventory
    )
    calls: list[dict[str, object]] = []

    def collector(**kwargs) -> dict[str, object]:
        calls.append(kwargs)
        assert kwargs["conflict_unit_ids"] == (STABILITY_CONFLICT,)
        return post_inventory

    sleeper_calls: list[float] = []
    result = environment.run_environment_stability_gate(
        pre_path,
        cleanup_path,
        selected_gpu_index=0,
        target_unit_id=STABILITY_TARGET,
        conflict_unit_ids=(STABILITY_CONFLICT,),
        dependency_unit_ids=(),
        allowed_failed_unit_ids=STABILITY_FAILED,
        sample_count=2,
        sample_interval_seconds=30.0,
        policy_path=policy_path,
        inventory_collector=collector,
        sleeper=sleeper_calls.append,
        monotonic_clock=iter((100.0, 130.0)).__next__,
    )
    assert result["passed"] is True
    assert result["sample_count"] == 2
    assert result["minimum_window_seconds"] == 30.0
    assert result["observed_window_seconds"] == 30.0
    assert len(result["samples"]) == 2
    assert len(calls) == 2
    assert sleeper_calls == [30.0]
    assert result["root_evidence"]["policy"]["policy_fingerprint"] == (
        policy["policy_fingerprint"]
    )
    assert all(sample["passed"] for sample in result["samples"])


def test_recovery_mode_binds_guard_lineage_baseline_and_passes_exactly(
    tmp_path: Path,
) -> None:
    conflict = environment.GPU0_CONFLICT_UNIT
    pre_inventory = _stability_inventory(conflict_unit=conflict)
    lineage = _write_partial_lineage(tmp_path)
    guard = {
        "mode": environment.RECOVERY_GUARD_MODE,
        "unit_name": conflict,
        "path": f"/run/user/{os.getuid()}/systemd/user/{conflict}",
        "target": "/dev/null",
        "owner_uid": os.getuid(),
        "device": 101,
        "inode": 202,
        "observed_unit_file_state": "enabled",
    }
    pre_path, cleanup_path = _write_stability_roots(
        tmp_path,
        pre_inventory,
        conflict_unit=conflict,
        cleanup_mode=environment.RECOVERY_CLEANUP_MODE,
        activation_guard=guard,
        partial_lineage=lineage,
    )
    contract, roots = environment.prepare_environment_stability_contract(
        pre_path,
        cleanup_path,
        selected_gpu_index=0,
        target_unit_id=STABILITY_TARGET,
        conflict_unit_ids=(conflict,),
        dependency_unit_ids=(),
        allowed_failed_unit_ids=STABILITY_FAILED,
        activation_guard_reader=_guard_observation,
    )
    assert contract.cleanup_mode == environment.RECOVERY_CLEANUP_MODE
    assert contract.quiescence_mode == environment.RECOVERY_QUIESCENCE_MODE
    assert contract.cleanup_nrestarts_baseline == ((conflict, "18425"),)
    assert contract.activation_guard == guard
    policy_path, policy = _write_policy_from_contract(
        tmp_path,
        contract,
        roots,
        name="recovery-policy.json",
    )
    assert policy["cleanup_root"]["path"] == str(cleanup_path)
    assert policy["postcleanup_contract"]["activation_guard"] == guard

    post_inventory = _stability_inventory(
        conflict_unit=conflict,
        postcleanup=True,
        postcleanup_unit_file_state="enabled",
    )
    result = environment.run_environment_stability_gate(
        pre_path,
        cleanup_path,
        selected_gpu_index=0,
        target_unit_id=STABILITY_TARGET,
        conflict_unit_ids=(conflict,),
        dependency_unit_ids=(),
        allowed_failed_unit_ids=STABILITY_FAILED,
        sample_count=2,
        sample_interval_seconds=30.0,
        policy_path=policy_path,
        inventory_collector=lambda **kwargs: post_inventory,
        activation_guard_reader=_guard_observation,
        sleeper=lambda seconds: None,
        monotonic_clock=iter((200.0, 230.0)).__next__,
    )
    assert result["passed"] is True
    for sample in result["samples"]:
        projection = sample["state_projection"][conflict]
        assert sample["passed"] is True
        assert projection["NRestarts"] == "18425"
        assert (
            projection["cleanup_nrestarts_baseline"] == "18425"
        )
        assert projection["quiescence_mode"] == (
            environment.RECOVERY_QUIESCENCE_MODE
        )
        assert projection["activation_guard"] == guard
        assert projection["activation_guard_observation"] == {
            **guard,
            "file_type": "symlink",
        }


def test_enabled_conflict_is_never_accepted_by_normal_cleanup_mode(
    tmp_path: Path,
) -> None:
    inventory = _stability_inventory()
    pre_path, cleanup_path = _write_stability_roots(
        tmp_path,
        inventory,
        after_unit_file_state="enabled",
    )
    with pytest.raises(
        PermissionError,
        match="cleanup conflict unit is not quiescent",
    ):
        environment.prepare_environment_stability_contract(
            pre_path,
            cleanup_path,
            selected_gpu_index=0,
            target_unit_id=STABILITY_TARGET,
            conflict_unit_ids=(STABILITY_CONFLICT,),
            dependency_unit_ids=(),
            allowed_failed_unit_ids=STABILITY_FAILED,
        )


def test_recovery_guard_or_cleanup_nrestarts_drift_fails_closed(
    tmp_path: Path,
) -> None:
    conflict = environment.GPU0_CONFLICT_UNIT
    pre_inventory = _stability_inventory(conflict_unit=conflict)
    lineage = _write_partial_lineage(tmp_path)
    guard = {
        "mode": environment.RECOVERY_GUARD_MODE,
        "unit_name": conflict,
        "path": f"/run/user/{os.getuid()}/systemd/user/{conflict}",
        "target": "/dev/null",
        "owner_uid": os.getuid(),
        "device": 303,
        "inode": 404,
        "observed_unit_file_state": "enabled",
    }
    pre_path, cleanup_path = _write_stability_roots(
        tmp_path,
        pre_inventory,
        conflict_unit=conflict,
        cleanup_mode=environment.RECOVERY_CLEANUP_MODE,
        activation_guard=guard,
        partial_lineage=lineage,
    )
    contract, roots = environment.prepare_environment_stability_contract(
        pre_path,
        cleanup_path,
        selected_gpu_index=0,
        target_unit_id=STABILITY_TARGET,
        conflict_unit_ids=(conflict,),
        dependency_unit_ids=(),
        allowed_failed_unit_ids=STABILITY_FAILED,
        activation_guard_reader=_guard_observation,
    )
    policy_path, _ = _write_policy_from_contract(
        tmp_path,
        contract,
        roots,
        name="recovery-drift-policy.json",
    )
    drifted_inventory = _stability_inventory(
        conflict_unit=conflict,
        conflict_nrestarts="18426",
        postcleanup=True,
        postcleanup_unit_file_state="enabled",
    )
    result = environment.run_environment_stability_gate(
        pre_path,
        cleanup_path,
        selected_gpu_index=0,
        target_unit_id=STABILITY_TARGET,
        conflict_unit_ids=(conflict,),
        dependency_unit_ids=(),
        allowed_failed_unit_ids=STABILITY_FAILED,
        sample_count=2,
        sample_interval_seconds=30.0,
        policy_path=policy_path,
        inventory_collector=lambda **kwargs: drifted_inventory,
        activation_guard_reader=_guard_observation,
        sleeper=lambda seconds: None,
        monotonic_clock=iter((300.0, 330.0)).__next__,
    )
    assert result["passed"] is False
    assert all(
        f"nrestarts_changed_from_cleanup:{conflict}"
        in sample["blockers"]
        for sample in result["samples"]
    )

    exact_observation = _guard_observation(guard)
    drifted_observation = {
        **exact_observation,
        "inode": int(guard["inode"]) + 1,
    }
    observations = iter(
        (
            exact_observation,
            exact_observation,
            drifted_observation,
            exact_observation,
        )
    )
    baseline_inventory = _stability_inventory(
        conflict_unit=conflict,
        postcleanup=True,
        postcleanup_unit_file_state="enabled",
    )
    with pytest.raises(
        PermissionError,
        match="sample semantic replay (?:invalid|changed)",
    ):
        environment.run_environment_stability_gate(
            pre_path,
            cleanup_path,
            selected_gpu_index=0,
            target_unit_id=STABILITY_TARGET,
            conflict_unit_ids=(conflict,),
            dependency_unit_ids=(),
            allowed_failed_unit_ids=STABILITY_FAILED,
            sample_count=2,
            sample_interval_seconds=30.0,
            policy_path=policy_path,
            inventory_collector=lambda **kwargs: baseline_inventory,
            activation_guard_reader=lambda expected: next(observations),
            sleeper=lambda seconds: None,
            monotonic_clock=iter((350.0, 380.0)).__next__,
        )

    def drifted_guard(
        expected: dict[str, object],
    ) -> dict[str, object]:
        return {
            **expected,
            "inode": int(expected["inode"]) + 1,
            "file_type": "symlink",
        }

    with pytest.raises(
        PermissionError,
        match="guard changed before policy sampling",
    ):
        environment.run_environment_stability_gate(
            pre_path,
            cleanup_path,
            selected_gpu_index=0,
            target_unit_id=STABILITY_TARGET,
            conflict_unit_ids=(conflict,),
            dependency_unit_ids=(),
            allowed_failed_unit_ids=STABILITY_FAILED,
            sample_count=2,
            sample_interval_seconds=30.0,
            policy_path=policy_path,
            inventory_collector=lambda **kwargs: drifted_inventory,
            activation_guard_reader=drifted_guard,
            sleeper=lambda seconds: None,
            monotonic_clock=iter((400.0, 430.0)).__next__,
        )


def test_recovery_symlink_reader_uses_exact_lstat_readlink_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "guard.service"
    path.symlink_to("/dev/null")
    metadata = path.lstat()
    guard = {
        "mode": environment.RECOVERY_GUARD_MODE,
        "unit_name": environment.GPU0_CONFLICT_UNIT,
        "path": str(path),
        "target": "/dev/null",
        "owner_uid": os.getuid(),
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "observed_unit_file_state": "enabled",
    }
    assert environment.inspect_recovery_activation_guard(guard) == {
        **guard,
        "file_type": "symlink",
    }
    changed = dict(guard)
    changed["inode"] = metadata.st_ino + 1
    with pytest.raises(PermissionError, match="identity changed"):
        environment.inspect_recovery_activation_guard(changed)


def test_stability_gate_detects_nrestarts_and_activation_closure_drift(
    tmp_path: Path,
) -> None:
    pre_inventory = _stability_inventory()
    baseline = _stability_inventory(postcleanup=True)
    drifted = _stability_inventory(
        target_nrestarts="1",
        target_triggered_by="unexpected.timer",
        postcleanup=True,
    )
    pre_path, cleanup_path = _write_stability_roots(tmp_path, pre_inventory)
    policy_path, _ = _write_stability_policy(
        tmp_path, pre_path, cleanup_path, pre_inventory
    )
    inventories = iter((baseline, drifted))
    result = environment.run_environment_stability_gate(
        pre_path,
        cleanup_path,
        selected_gpu_index=0,
        target_unit_id=STABILITY_TARGET,
        conflict_unit_ids=(STABILITY_CONFLICT,),
        dependency_unit_ids=(),
        allowed_failed_unit_ids=STABILITY_FAILED,
        sample_count=2,
        sample_interval_seconds=30.0,
        policy_path=policy_path,
        inventory_collector=lambda **kwargs: next(inventories),
        sleeper=lambda seconds: None,
        monotonic_clock=iter((1.0, 31.0)).__next__,
    )
    assert result["passed"] is False
    assert f"nrestarts_drift:{STABILITY_TARGET}:1" in result["blockers"]
    assert (
        f"activation_closure_drift:{STABILITY_TARGET}:1"
        in result["blockers"]
    )
    assert result["D_R_payload_accessed"] is False
    assert result["D_V_payload_accessed"] is False
    assert result["D_T_payload_accessed"] is False



def test_stability_requires_selected_gpu_zero_consumers_even_if_preallowed(
    tmp_path: Path,
) -> None:
    pre_inventory = _stability_inventory(
        allowed_unit_ids=(STABILITY_CONFLICT,),
        selected_consumer=True,
    )
    post_inventory = _stability_inventory(
        allowed_unit_ids=(STABILITY_CONFLICT,),
        selected_consumer=True,
        postcleanup=True,
    )
    pre_path, cleanup_path = _write_stability_roots(tmp_path, pre_inventory)
    policy_path, _ = _write_stability_policy(
        tmp_path, pre_path, cleanup_path, pre_inventory
    )
    result = environment.run_environment_stability_gate(
        pre_path,
        cleanup_path,
        selected_gpu_index=0,
        target_unit_id=STABILITY_TARGET,
        conflict_unit_ids=(STABILITY_CONFLICT,),
        dependency_unit_ids=(),
        allowed_failed_unit_ids=STABILITY_FAILED,
        allowed_unit_ids=(STABILITY_CONFLICT,),
        sample_count=2,
        sample_interval_seconds=30.0,
        policy_path=policy_path,
        inventory_collector=lambda **kwargs: post_inventory,
        sleeper=lambda seconds: None,
        monotonic_clock=iter((5.0, 35.0)).__next__,
    )
    assert all(sample["passed"] is False for sample in result["samples"])
    assert all(
        "selected_gpu_not_empty:777" in sample["blockers"]
        for sample in result["samples"]
    )
    assert result["passed"] is False
    assert "selected_gpu_not_empty:777:0" in result["blockers"]
    assert "selected_gpu_not_empty:777:1" in result["blockers"]


def test_create_policy_cli_seals_policy_from_precleanup_receipt(
    tmp_path: Path,
) -> None:
    inventory = _stability_inventory()
    pre_path, cleanup_path = _write_stability_roots(tmp_path, inventory)
    output = (tmp_path / "created-policy.json").resolve()
    assert environment.main(
        [
            "create-policy",
            "--output", str(output),
            "--precleanup-inventory-receipt", str(pre_path),
            "--cleanup-receipt", str(cleanup_path),
            "--selected-gpu-index", "0",
            "--target-unit", STABILITY_TARGET,
            "--conflict-unit", STABILITY_CONFLICT,
            "--allow-failed-unit", STABILITY_FAILED[0],
            "--allow-failed-unit", STABILITY_FAILED[1],
            "--sample-count", "2",
            "--interval-seconds", "30",
        ]
    ) == 0
    policy = environment.load_sealed_receipt(
        output,
        fingerprint_field="policy_fingerprint",
    )
    assert environment.validate_environment_policy(policy) == policy
    assert policy["unit_scope"]["target_unit_id"] == STABILITY_TARGET
    assert policy["unit_scope"]["conflict_unit_ids"] == [STABILITY_CONFLICT]
    assert policy["unit_scope"]["allowed_failed_unit_ids"] == list(
        STABILITY_FAILED
    )
    assert policy["strict_all_gpu_consumers"] is False

    assert policy["sampling"]["maximum_restart_usec"] == 30_000_000
    precleanup = environment.load_sealed_receipt(pre_path)
    assert precleanup["passed"] is False
    assert precleanup["inventory"]["blockers"] == [
        f"scoped_blocker_unit_not_quiescent:{STABILITY_CONFLICT}"
    ]
    assert precleanup["inventory"]["unit_scope"]["shadows"][STABILITY_CONFLICT]["RestartUSec"] == "30s"
    cleanup = environment.load_sealed_receipt(
        cleanup_path, fingerprint_field="cleanup_receipt_fingerprint"
    )
    assert all(cleanup[field] is False for field in (
        "D_R_payload_accessed", "D_V_payload_accessed", "D_T_payload_accessed"
    ))
    assert cleanup["after"][STABILITY_CONFLICT]["UnitFileState"] == "masked-runtime"
    assert cleanup["after"][STABILITY_CONFLICT]["ActiveState"] == "inactive"
    assert policy["precleanup_root"]["path"] == str(pre_path)
    assert policy["precleanup_root"]["inventory_fingerprint"] == (
        inventory["inventory_fingerprint"]
    )
    assert set(policy["toolchain"]) == {
        "runtime_environment", "python", "systemctl", "nvidia_smi"
    }


@pytest.mark.parametrize("interval", ["nan", "inf"])
def test_stability_cli_exception_still_writes_sealed_fail_receipt(
    tmp_path: Path,
    interval: str,
) -> None:
    output = (tmp_path / f"stability-fail-{interval}.json").resolve()
    assert environment.main(
        [
            "stability-gate",
            "--output", str(output),
            "--precleanup-inventory-receipt", str(tmp_path / "pre.json"),
            "--cleanup-receipt", str(tmp_path / "cleanup.json"),
            "--policy", str(tmp_path / "policy.json"),
            "--selected-gpu-index", "0",
            "--target-unit", STABILITY_TARGET,
            "--conflict-unit", STABILITY_CONFLICT,
            "--sample-count", "2",
            "--interval-seconds", interval,
        ]
    ) == 1
    receipt = environment.load_sealed_receipt(
        output,
        fingerprint_field="stability_receipt_fingerprint",
    )
    assert receipt["passed"] is False
    assert receipt["sample_interval_seconds"] == interval
    assert receipt["payload_authority"] == "none"
    assert receipt["blockers"] == ["stability_gate_exception"]
    assert receipt["error_type"] == "ValueError"
    assert receipt["D_R_payload_accessed"] is False

def test_restart_window_parser_and_unclosed_trigger_fail_closed() -> None:
    assert environment.parse_systemd_duration_usec("30s") == 30_000_000
    assert environment.parse_systemd_duration_usec("1min 0.5s") == 60_500_000
    with pytest.raises(ValueError, match="unknown token"):
        environment.parse_systemd_duration_usec("infinity")
    inventory = _stability_inventory(target_triggered_by="rogue.timer")
    with pytest.raises(PermissionError, match="trigger period is not closed"):
        _stability_contract(inventory)


def test_stability_gate_rejects_missing_policy_before_reading_roots() -> None:
    with pytest.raises(PermissionError, match="policy is required"):
        environment.run_environment_stability_gate(
            "/does/not/exist-pre.json",
            "/does/not/exist-cleanup.json",
            selected_gpu_index=0,
            target_unit_id=STABILITY_TARGET,
            conflict_unit_ids=(STABILITY_CONFLICT,),
            dependency_unit_ids=(),
            allowed_failed_unit_ids=STABILITY_FAILED,
            sample_count=2,
            sample_interval_seconds=30.0,
            policy_path=None,
        )


def test_noop_sleeper_cannot_claim_thirty_second_window(tmp_path: Path) -> None:
    pre_inventory = _stability_inventory()
    post_inventory = _stability_inventory(postcleanup=True)
    pre_path, cleanup_path = _write_stability_roots(tmp_path, pre_inventory)
    policy_path, _ = _write_stability_policy(
        tmp_path, pre_path, cleanup_path, pre_inventory
    )
    result = environment.run_environment_stability_gate(
        pre_path,
        cleanup_path,
        selected_gpu_index=0,
        target_unit_id=STABILITY_TARGET,
        conflict_unit_ids=(STABILITY_CONFLICT,),
        dependency_unit_ids=(),
        allowed_failed_unit_ids=STABILITY_FAILED,
        sample_count=2,
        sample_interval_seconds=30.0,
        policy_path=policy_path,
        inventory_collector=lambda **kwargs: post_inventory,
        sleeper=lambda seconds: None,
        monotonic_clock=iter((10.0, 10.0)).__next__,
    )
    assert result["passed"] is False
    assert result["observed_window_seconds"] == 0.0
    assert "observed_stability_window_too_short" in result["blockers"]
    assert environment.validate_environment_stability_receipt(result)
    mutated = dict(result)
    mutated["extra"] = True
    mutated_body = dict(mutated)
    mutated_body.pop("stability_receipt_fingerprint")
    mutated["stability_receipt_fingerprint"] = environment.stable_fingerprint(
        mutated_body
    )
    with pytest.raises(ValueError, match="closed schema"):
        environment.validate_environment_stability_receipt(mutated)


def test_failed_unit_set_is_frozen_and_disappearance_fails(tmp_path: Path) -> None:
    pre_inventory = _stability_inventory()
    post_inventory = _stability_inventory(postcleanup=True)
    post_body = dict(post_inventory)
    post_body.pop("inventory_fingerprint")
    post_body["manager"] = dict(post_body["manager"])
    post_body["manager"]["failed_units"] = [STABILITY_FAILED[0]]
    post_inventory = {
        **post_body,
        "inventory_fingerprint": environment.stable_fingerprint(post_body),
    }
    pre_path, cleanup_path = _write_stability_roots(tmp_path, pre_inventory)
    policy_path, _ = _write_stability_policy(
        tmp_path, pre_path, cleanup_path, pre_inventory
    )
    result = environment.run_environment_stability_gate(
        pre_path,
        cleanup_path,
        selected_gpu_index=0,
        target_unit_id=STABILITY_TARGET,
        conflict_unit_ids=(STABILITY_CONFLICT,),
        dependency_unit_ids=(),
        allowed_failed_unit_ids=STABILITY_FAILED,
        sample_count=2,
        sample_interval_seconds=30.0,
        policy_path=policy_path,
        inventory_collector=lambda **kwargs: post_inventory,
        sleeper=lambda seconds: None,
        monotonic_clock=iter((20.0, 50.0)).__next__,
    )
    assert result["passed"] is False
    assert all(
        "failed_unit_set_changed" in sample["blockers"]
        for sample in result["samples"]
    )
    assert policy_path.exists()


@pytest.mark.parametrize("invalid_value", [None, "missing"])
def test_cleanup_payload_flags_must_be_explicit_false(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    pre_inventory = _stability_inventory()
    pre_path, cleanup_path = _write_stability_roots(tmp_path, pre_inventory)
    cleanup = environment.load_sealed_receipt(
        cleanup_path,
        fingerprint_field="cleanup_receipt_fingerprint",
    )
    cleanup_body = dict(cleanup)
    cleanup_body.pop("cleanup_receipt_fingerprint")
    if invalid_value == "missing":
        cleanup_body.pop("D_R_payload_accessed")
    else:
        cleanup_body["D_R_payload_accessed"] = invalid_value
    invalid_path = (tmp_path / "invalid-cleanup.json").resolve()
    environment.write_create_once_receipt(
        invalid_path,
        cleanup_body,
        fingerprint_field="cleanup_receipt_fingerprint",
    )
    with pytest.raises(PermissionError, match="accessed payload"):
        environment.prepare_environment_stability_contract(
            pre_path,
            invalid_path,
            selected_gpu_index=0,
            target_unit_id=STABILITY_TARGET,
            conflict_unit_ids=(STABILITY_CONFLICT,),
            dependency_unit_ids=(),
            allowed_failed_unit_ids=STABILITY_FAILED,
        )


def test_sealed_root_evidence_rejects_path_inode_swap(tmp_path: Path) -> None:
    path = (tmp_path / "root.json").resolve()
    body = {
        "schema_version": "generated-root-v1",
        "value": 1,
    }
    environment.write_create_once_receipt(path, body)
    _, evidence = environment.load_sealed_receipt_with_evidence(path)
    path.unlink()
    environment.write_create_once_receipt(path, body)
    with pytest.raises(PermissionError, match="root changed"):
        environment.verify_sealed_receipt_evidence(path, evidence)
