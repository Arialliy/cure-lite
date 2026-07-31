from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from tools import cure_lite_v24_actual_unit_realization as realization


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "deploy/systemd/cure-lite-v24-gcr-pacre-dr-r2.service.template"
PYTHON = Path("/usr/bin/python3.12")
SUPERVISOR = ROOT / "tools/cure_lite_v24_runtime_supervisor.py"


def test_actual_realizer_freezes_safe_system_unit_path_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "canonical-system-unit-path"
    target.mkdir(mode=0o755)
    alias = tmp_path / "xdg-system-unit-path"
    alias.symlink_to(target, target_is_directory=True)
    identity = realization._path_row(alias)
    assert identity["path_is_symlink"] is True
    assert identity["link_target"] == str(target)
    assert identity["resolved_path"] == str(target)
    assert identity["owner_uid"] == os.getuid()

    target.chmod(0o777)
    with pytest.raises(PermissionError, match="not trusted"):
        realization._path_row(alias)


def _manager() -> dict[str, object]:
    uid = os.getuid()
    runtime = f"/run/user/{uid}"
    return {
        "boot_id": "12345678-1234-1234-1234-123456789abc",
        "identity": {
            "pid": 412, "starttime_ticks": 991, "uid": uid,
            "control_group": (
                f"/user.slice/user-{uid}.slice/user@{uid}.service/init.scope"
            ),
        },
        "endpoint": {
            "uid": uid, "runtime_directory": runtime,
            "runtime_directory_device": 11, "runtime_directory_inode": 12,
            "bus_path": f"{runtime}/bus", "bus_device": 13,
            "bus_inode": 14,
        },
    }


def _shadow_text(values: dict[str, str]) -> str:
    return "".join(f"{name}={values[name]}\n" for name in realization._SHADOW_PROPERTIES)


def _not_found_shadow() -> dict[str, str]:
    values = {name: "" for name in realization._SHADOW_PROPERTIES}
    values.update({
        "Id": realization.ACTUAL_UNIT, "LoadState": "not-found",
        "ActiveState": "inactive", "SubState": "dead",
    })
    return values


def test_not_found_shadow_normalizes_only_omitted_empty_exec_properties(
) -> None:
    expected = _not_found_shadow()
    omitted_exec = "".join(
        f"{name}={expected[name]}\n"
        for name in realization._SHADOW_PROPERTIES
        if name not in realization._EXEC_MODES
    )

    assert realization._parse_show(omitted_exec) == expected

    loaded = omitted_exec.replace(
        "LoadState=not-found\n",
        "LoadState=loaded\n",
    )
    with pytest.raises(ValueError, match="property set is not exact"):
        realization._parse_show(loaded)
    missing_nonexec = omitted_exec.replace("FragmentPath=\n", "")
    with pytest.raises(ValueError, match="property set is not exact"):
        realization._parse_show(missing_nonexec)
    with pytest.raises(ValueError, match="property set is not exact"):
        realization._parse_show(omitted_exec + "Unexpected=\n")

    nonempty_exec = dict(expected)
    nonempty_exec["ExecStart"] = "{ path=/bin/false ; argv[]=/bin/false ; }"
    with pytest.raises(PermissionError, match="shadow already exists"):
        realization._require_no_shadow(nonempty_exec)


class FakeRunner:
    def __init__(
        self, unit_dir: Path, alternate_dir: Path, generator_late: Path,
    ) -> None:
        self.unit_dir = unit_dir
        self.alternate_dir = alternate_dir
        self.generator_late = generator_late
        self.commands: list[tuple[str, ...]] = []
        self.reloaded = False
        self.daemon_failure = False
        self.exec_drift = False
        self.watchdog_usec_override: str | None = None
        self.analyze_calls = 0
        self.rotate_on_reload = True
        self.reload_generator_mode: int | None = None
        self.rotate_on_analyze_call: int | None = None
        self.rotation_count = 0
        self.replace_fragment_on_shadow = False

    def rotate_generator_late(self) -> None:
        self.rotation_count += 1
        old = self.generator_late.with_name(
            f"generator.late.before-{self.rotation_count}"
        )
        self.generator_late.rename(old)
        self.generator_late.mkdir(mode=0o700)

    def __call__(self, argv: list[str], **kwargs: object) -> SimpleNamespace:
        assert kwargs["shell"] is False
        assert kwargs["check"] is False
        command = tuple(argv)
        self.commands.append(command)
        if command[0] == realization.SYSTEMD_PATH:
            return SimpleNamespace(returncode=0, stdout=f"{self.unit_dir}\n", stderr="")
        if command[0] == realization.SYSTEMD_ANALYZE:
            self.analyze_calls += 1
            if self.rotate_on_analyze_call == self.analyze_calls:
                self.rotate_generator_late()
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    f"{self.alternate_dir}\n{self.unit_dir}\n"
                    f"{self.generator_late}\n"
                ),
                stderr="",
            )
        if command == (realization.SYSTEMCTL, "--user", "daemon-reload"):
            if self.daemon_failure:
                return SimpleNamespace(returncode=1, stdout="", stderr="mock failure")
            if self.rotate_on_reload:
                self.rotate_generator_late()
            if self.reload_generator_mode is not None:
                self.generator_late.chmod(self.reload_generator_mode)
            self.reloaded = True
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        assert command == realization._shadow_argv()
        if not self.reloaded:
            values = _not_found_shadow()
        else:
            fragment = self.unit_dir / realization.ACTUAL_UNIT
            values = realization._expected_static_shadow(fragment)
            values["WatchdogUSec"] = (
                self.watchdog_usec_override
                if self.watchdog_usec_override is not None
                else "infinity"
            )
            expected_exec = realization._expected_exec(
                python_path=PYTHON, supervisor_path=SUPERVISOR,
                runtime_spec_path=self.runtime_spec,
            )
            for directive, exact_argv in expected_exec.items():
                argv_value = list(exact_argv)
                if self.exec_drift and directive == "ExecStart":
                    argv_value[4] = "verify-runtime-spec"
                values[directive] = (
                    f"{{ path={argv_value[0]} ; argv[]={' '.join(argv_value)} ; "
                    "ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; "
                    "pid=0 ; code=(null) ; status=0/0 }}"
                )
            if self.replace_fragment_on_shadow:
                raw = fragment.read_bytes()
                fragment.rename(fragment.with_name(f"{fragment.name}.before-race"))
                fragment.write_bytes(raw)
                fragment.chmod(0o600)
                self.replace_fragment_on_shadow = False
        return SimpleNamespace(returncode=0, stdout=_shadow_text(values), stderr="")


def _workspace(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    evidence = tmp_path / "evidence"
    unit_dir = tmp_path / "runtime-user-units"
    alternate = tmp_path / "alternate-user-units"
    generator_late = tmp_path / "generator.late"
    spec_dir = tmp_path / "future-spec"
    for path in (evidence, unit_dir, generator_late, spec_dir):
        path.mkdir()
        path.chmod(0o700)
    alternate.mkdir()
    alternate.chmod(0o755)
    return (
        evidence, unit_dir, alternate, generator_late,
        spec_dir / "runtime-spec.json",
    )


def _authorize(
    evidence: Path, runtime_spec: Path, runner: FakeRunner,
    *, validity_seconds: int = 300,
) -> tuple[Path, dict[str, object]]:
    authorization = evidence / "authorization.json"
    runner.runtime_spec = runtime_spec
    payload = realization.create_authorization(
        authorization, template_path=TEMPLATE, python_path=PYTHON,
        supervisor_path=SUPERVISOR, runtime_spec_path=runtime_spec,
        authorization_basis=realization.AUTHORIZATION_BASIS,
        instruction_id=realization.INSTRUCTION_ID,
        validity_seconds=validity_seconds, runner=runner,
        manager_reader=lambda: deepcopy(_manager()),
    )
    return authorization, payload


def _reseal_authorization(
    path: Path, payload: dict[str, object],
) -> dict[str, object]:
    body = deepcopy(payload)
    body.pop("authorization_fingerprint")
    return realization.write_create_once_json(
        path, body, fingerprint_field="authorization_fingerprint",
    )


@pytest.mark.parametrize(
    "drift_kind",
    [
        "binding-false-to-zero",
        "binding-int-to-bool",
        "binding-int-to-float",
        "runtime-parent-true-to-one",
        "manager-int-to-float",
    ],
)
def test_resealed_coercive_authorization_drift_fails_before_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift_kind: str,
) -> None:
    evidence, unit_dir, alternate, generator_late, runtime_spec = _workspace(
        tmp_path
    )
    runner = FakeRunner(unit_dir, alternate, generator_late)
    _, authorized = _authorize(evidence, runtime_spec, runner)
    drifted = deepcopy(authorized)
    if drift_kind == "binding-false-to-zero":
        drifted["executable_bindings"]["python"]["path_is_symlink"] = 0
    elif drift_kind == "binding-int-to-bool":
        drifted["executable_bindings"]["python"]["owner_uid"] = False
    elif drift_kind == "binding-int-to-float":
        binding = drifted["template_binding"]
        binding["inode"] = float(binding["inode"])
    elif drift_kind == "runtime-parent-true-to-one":
        drifted["runtime_spec_binding"]["runtime_spec_parent_identity"][
            "exists"
        ] = 1
    else:
        manager_identity = drifted["manager_generation"]["identity"]
        manager_identity["pid"] = float(manager_identity["pid"])
    resealed = evidence / f"authorization-{drift_kind}.json"
    _reseal_authorization(resealed, drifted)

    monkeypatch.setattr(
        realization,
        "_install_fragment",
        lambda *_args, **_kwargs: pytest.fail(
            "fragment install reached after coercive authorization drift"
        ),
    )
    with pytest.raises(PermissionError):
        realization.realize_actual_unit(
            resealed,
            receipt_path=evidence / f"receipt-{drift_kind}.json",
            terminal_path=evidence / f"terminal-{drift_kind}.json",
            runner=runner,
            manager_reader=lambda: deepcopy(_manager()),
        )
    assert not (unit_dir / realization.ACTUAL_UNIT).exists()
    assert (
        realization.SYSTEMCTL, "--user", "daemon-reload"
    ) not in runner.commands


def test_daemon_reload_policy_transition_is_exact_by_default_and_strict(
    tmp_path: Path,
) -> None:
    evidence, unit_dir, alternate, generator_late, runtime_spec = _workspace(
        tmp_path
    )
    del evidence, alternate, runtime_spec
    uid = os.getuid()
    generator_row = {
        "path": str(generator_late), "exists": True, "device": 7,
        "inode": 101, "owner_uid": uid, "mode": 0o700,
    }
    before = {
        "runtime_directory": str(unit_dir),
        "ordered_unit_paths": [
            {"path": str(unit_dir), "exists": False},
            generator_row,
        ],
        "unchanged": {"nested": [False, 3, "exact"]},
    }
    assert realization._validate_daemon_reload_path_policy_transition(
        before, deepcopy(before), runtime_directory=unit_dir,
        authorized_uid=uid,
    ) == before

    after = deepcopy(before)
    after["ordered_unit_paths"][1]["inode"] = 202
    accepted = realization._validate_daemon_reload_path_policy_transition(
        before, after, runtime_directory=unit_dir, authorized_uid=uid,
    )
    assert accepted == after

    for field, malformed in (("device", True), ("mode", 0o10000)):
        malformed_before = deepcopy(before)
        malformed_after = deepcopy(after)
        malformed_before["ordered_unit_paths"][1][field] = malformed
        malformed_after["ordered_unit_paths"][1][field] = malformed
        with pytest.raises(
            PermissionError, match="generator.late identity is invalid"
        ):
            realization._validate_daemon_reload_path_policy_transition(
                malformed_before, malformed_after,
                runtime_directory=unit_dir, authorized_uid=uid,
            )

    invalid_after_rows = []
    for field, value in (
        ("exists", False),
        ("owner_uid", uid + 1),
        ("mode", 0o722),
        ("inode", True),
        ("device", 8),
    ):
        invalid = deepcopy(after)
        invalid["ordered_unit_paths"][1][field] = value
        invalid_after_rows.append(invalid)
    extra_key = deepcopy(after)
    extra_key["ordered_unit_paths"][1]["unexpected"] = "not-exact"
    invalid_after_rows.append(extra_key)
    other_path = deepcopy(after)
    other_path["ordered_unit_paths"][0]["exists"] = True
    invalid_after_rows.append(other_path)
    reordered = deepcopy(after)
    reordered["ordered_unit_paths"].reverse()
    invalid_after_rows.append(reordered)
    for invalid in invalid_after_rows:
        with pytest.raises(PermissionError):
            realization._validate_daemon_reload_path_policy_transition(
                before, invalid, runtime_directory=unit_dir,
                authorized_uid=uid,
            )


def test_authorize_and_realize_is_create_only_static_and_payload_blind(
    tmp_path: Path,
) -> None:
    evidence, unit_dir, alternate, generator_late, runtime_spec = _workspace(
        tmp_path
    )
    runner = FakeRunner(unit_dir, alternate, generator_late)
    authorization, authorized = _authorize(evidence, runtime_spec, runner)
    assert not runtime_spec.exists()
    assert authorized["runtime_spec_binding"]["kind"] == (
        "future-absent-runtime-spec-v2"
    )
    assert authorized["expected_static_shadow"]["WatchdogUSec"] == "disabled"
    assert authorized["unit_path_policy"]["systemd_path_argv"] == [
        realization.SYSTEMD_PATH, "--suffix=systemd/user", "user-runtime"
    ]
    assert authorized["unit_path_policy"]["runtime_directory_priority"] == 1
    python_binding = authorized["executable_bindings"]["python"]
    assert python_binding["path"] == "/usr/bin/python3.12"
    assert python_binding["resolved_path"] == "/usr/bin/python3.12"
    assert python_binding["path_is_symlink"] is False
    assert python_binding["owner_uid"] == 0
    assert python_binding["mode"] & 0o022 == 0
    receipt_path = evidence / "receipt.json"
    terminal_path = evidence / "terminal.json"
    receipt = realization.realize_actual_unit(
        authorization, receipt_path=receipt_path, terminal_path=terminal_path,
        runner=runner, manager_reader=lambda: deepcopy(_manager()),
    )
    fragment = unit_dir / realization.ACTUAL_UNIT
    assert fragment.exists() and stat.S_IMODE(fragment.stat().st_mode) == 0o600
    assert fragment.stat().st_uid == os.getuid() and fragment.stat().st_nlink == 1
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
    assert not terminal_path.exists() and not runtime_spec.exists()
    assert receipt["passed"] is True and receipt["static"] is True
    assert receipt["enabled"] is False and receipt["started"] is False
    assert receipt["removed"] is False
    assert receipt["completed_actions"] == realization._ACTIONS
    assert receipt["executable_bindings"]["python"] == python_binding
    for directive in realization._EXEC_MODES:
        assert receipt["full_static_shadow"][
            f"normalized_{directive}"
        ]["argv"][1:5] == ["-I", "-S", "-B", "-u"]
    before_rows = authorized["unit_path_policy"]["ordered_unit_paths"]
    after_rows = receipt["unit_path_policy"]["ordered_unit_paths"]
    generator_index = next(
        index for index, row in enumerate(before_rows)
        if row["path"] == str(generator_late)
    )
    assert before_rows[generator_index]["inode"] != (
        after_rows[generator_index]["inode"]
    )
    restored = deepcopy(receipt["unit_path_policy"])
    restored["ordered_unit_paths"][generator_index]["inode"] = (
        before_rows[generator_index]["inode"]
    )
    assert restored == authorized["unit_path_policy"]
    assert all(receipt[name] is False for name in (
        "D_R_payload_accessed", "D_V_payload_accessed", "D_T_payload_accessed"
    ))
    joined = " ".join(" ".join(command) for command in runner.commands)
    assert " enable " not in f" {joined} "
    assert " start " not in f" {joined} "
    assert " remove " not in f" {joined} "
    assert runner.commands.count(
        (realization.SYSTEMCTL, "--user", "daemon-reload")
    ) == 1


def test_pre_reload_generator_rotation_is_rejected_before_daemon(
    tmp_path: Path,
) -> None:
    evidence, unit_dir, alternate, generator_late, runtime_spec = _workspace(
        tmp_path
    )
    runner = FakeRunner(unit_dir, alternate, generator_late)
    authorization, _ = _authorize(evidence, runtime_spec, runner)
    runner.rotate_on_analyze_call = 3
    with pytest.raises(PermissionError, match="installed user unit search path"):
        realization.realize_actual_unit(
            authorization,
            receipt_path=evidence / "receipt.json",
            terminal_path=evidence / "terminal.json",
            runner=runner,
            manager_reader=lambda: deepcopy(_manager()),
        )
    assert runner.rotation_count == 1
    assert (
        realization.SYSTEMCTL, "--user", "daemon-reload"
    ) not in runner.commands
    terminal = realization.load_sealed_json(
        evidence / "terminal.json", "terminal_fingerprint"
    )
    assert terminal["completed_actions"] == [realization._ACTIONS[0]]
    assert terminal["daemon_reload_attempted"] is False


def test_second_generator_rotation_is_rejected_at_precommit(
    tmp_path: Path,
) -> None:
    evidence, unit_dir, alternate, generator_late, runtime_spec = _workspace(
        tmp_path
    )
    runner = FakeRunner(unit_dir, alternate, generator_late)
    authorization, _ = _authorize(evidence, runtime_spec, runner)
    runner.rotate_on_analyze_call = 6
    with pytest.raises(PermissionError, match="installed user unit search path"):
        realization.realize_actual_unit(
            authorization,
            receipt_path=evidence / "receipt.json",
            terminal_path=evidence / "terminal.json",
            runner=runner,
            manager_reader=lambda: deepcopy(_manager()),
        )
    assert runner.rotation_count == 2
    assert not (evidence / "receipt.json").exists()
    terminal = realization.load_sealed_json(
        evidence / "terminal.json", "terminal_fingerprint"
    )
    assert terminal["completed_actions"] == realization._ACTIONS
    assert terminal["daemon_reload_attempted"] is True
    assert terminal["start_attempted"] is False


def test_unsafe_reload_rotation_is_rejected_at_immediate_capture(
    tmp_path: Path,
) -> None:
    evidence, unit_dir, alternate, generator_late, runtime_spec = _workspace(
        tmp_path
    )
    runner = FakeRunner(unit_dir, alternate, generator_late)
    authorization, _ = _authorize(evidence, runtime_spec, runner)
    shadow_calls_before = runner.commands.count(realization._shadow_argv())
    runner.reload_generator_mode = 0o722
    manager_reads = 0

    def manager_reader() -> dict[str, object]:
        nonlocal manager_reads
        manager_reads += 1
        if manager_reads > 2:
            pytest.fail("manager read occurred after an invalid reload transition")
        return deepcopy(_manager())

    with pytest.raises(
        PermissionError, match="generator.late identity is invalid"
    ):
        realization.realize_actual_unit(
            authorization,
            receipt_path=evidence / "receipt.json",
            terminal_path=evidence / "terminal.json",
            runner=runner,
            manager_reader=manager_reader,
        )
    terminal = realization.load_sealed_json(
        evidence / "terminal.json", "terminal_fingerprint"
    )
    assert terminal["completed_actions"] == [realization._ACTIONS[0]]
    assert terminal["daemon_reload_attempted"] is True
    assert manager_reads == 2
    assert runner.commands.count(realization._shadow_argv()) == (
        shadow_calls_before + 1
    )


def test_watchdog_disabled_normalization_rejects_enabled_interval(
    tmp_path: Path,
) -> None:
    evidence, unit_dir, alternate, generator_late, runtime_spec = _workspace(
        tmp_path
    )
    runner = FakeRunner(unit_dir, alternate, generator_late)
    authorization, authorized = _authorize(evidence, runtime_spec, runner)
    assert authorized["expected_static_shadow"]["WatchdogUSec"] == "disabled"
    runner.watchdog_usec_override = "1s"
    with pytest.raises(
        PermissionError, match="shadow is not exact static"
    ):
        realization.realize_actual_unit(
            authorization,
            receipt_path=evidence / "receipt.json",
            terminal_path=evidence / "terminal.json",
            runner=runner,
            manager_reader=lambda: deepcopy(_manager()),
        )
    terminal = realization.load_sealed_json(
        evidence / "terminal.json", "terminal_fingerprint"
    )
    assert terminal["passed"] is False
    assert terminal["start_attempted"] is False


def test_authorization_rejects_any_search_path_shadow(tmp_path: Path) -> None:
    evidence, unit_dir, alternate, generator_late, runtime_spec = _workspace(
        tmp_path
    )
    (alternate / realization.ACTUAL_UNIT).write_text("shadow\n", encoding="utf-8")
    runner = FakeRunner(unit_dir, alternate, generator_late)
    with pytest.raises(PermissionError, match="shadows"):
        _authorize(evidence, runtime_spec, runner)
    assert not (unit_dir / realization.ACTUAL_UNIT).exists()
    assert all(command[0] != realization.SYSTEMCTL for command in runner.commands)


def test_future_runtime_spec_must_stay_absent(tmp_path: Path) -> None:
    evidence, unit_dir, alternate, generator_late, runtime_spec = _workspace(
        tmp_path
    )
    runner = FakeRunner(unit_dir, alternate, generator_late)
    authorization, _ = _authorize(evidence, runtime_spec, runner)
    runtime_spec.write_text("not authorized yet\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="future runtime spec"):
        realization.realize_actual_unit(
            authorization, receipt_path=evidence / "receipt.json",
            terminal_path=evidence / "terminal.json", runner=runner,
            manager_reader=lambda: deepcopy(_manager()),
        )
    assert not (unit_dir / realization.ACTUAL_UNIT).exists()
    terminal = realization.load_sealed_json(
        evidence / "terminal.json", "terminal_fingerprint"
    )
    assert terminal["passed"] is False
    assert terminal["fragment_may_remain"] is False


def test_daemon_reload_failure_writes_terminal_and_never_removes(
    tmp_path: Path,
) -> None:
    evidence, unit_dir, alternate, generator_late, runtime_spec = _workspace(
        tmp_path
    )
    runner = FakeRunner(unit_dir, alternate, generator_late)
    authorization, _ = _authorize(evidence, runtime_spec, runner)
    runner.daemon_failure = True
    with pytest.raises(RuntimeError, match="daemon-reload"):
        realization.realize_actual_unit(
            authorization, receipt_path=evidence / "receipt.json",
            terminal_path=evidence / "terminal.json", runner=runner,
            manager_reader=lambda: deepcopy(_manager()),
        )
    fragment = unit_dir / realization.ACTUAL_UNIT
    assert fragment.exists()
    terminal = realization.load_sealed_json(
        evidence / "terminal.json", "terminal_fingerprint"
    )
    assert terminal["fragment_may_remain"] is True
    assert terminal["automatic_removal_performed"] is False
    assert terminal["remove_attempted"] is False
    assert stat.S_IMODE((evidence / "terminal.json").stat().st_mode) == 0o444


def test_exact_supervisor_argv_drift_is_terminal(tmp_path: Path) -> None:
    evidence, unit_dir, alternate, generator_late, runtime_spec = _workspace(
        tmp_path
    )
    runner = FakeRunner(unit_dir, alternate, generator_late)
    authorization, _ = _authorize(evidence, runtime_spec, runner)
    runner.exec_drift = True
    with pytest.raises(PermissionError, match="ExecStart argv"):
        realization.realize_actual_unit(
            authorization, receipt_path=evidence / "receipt.json",
            terminal_path=evidence / "terminal.json", runner=runner,
            manager_reader=lambda: deepcopy(_manager()),
        )
    terminal = realization.load_sealed_json(
        evidence / "terminal.json", "terminal_fingerprint"
    )
    assert terminal["passed"] is False
    assert terminal["enable_attempted"] is False
    assert terminal["start_attempted"] is False


def test_wrong_instruction_or_overlong_authorization_is_rejected(
    tmp_path: Path,
) -> None:
    evidence, unit_dir, alternate, generator_late, runtime_spec = _workspace(
        tmp_path
    )
    runner = FakeRunner(unit_dir, alternate, generator_late)
    runner.runtime_spec = runtime_spec
    common = dict(
        template_path=TEMPLATE, python_path=PYTHON,
        supervisor_path=SUPERVISOR, runtime_spec_path=runtime_spec,
        authorization_basis=realization.AUTHORIZATION_BASIS, runner=runner,
        manager_reader=lambda: deepcopy(_manager()),
    )
    with pytest.raises(PermissionError, match="instruction"):
        realization.create_authorization(
            evidence / "wrong.json", instruction_id="wrong", **common
        )
    with pytest.raises(ValueError, match="validity"):
        realization.create_authorization(
            evidence / "long.json", instruction_id=realization.INSTRUCTION_ID,
            validity_seconds=301, **common,
        )


def test_template_and_command_allowlists_reject_expansion() -> None:
    expanded = TEMPLATE.read_text(encoding="utf-8").replace(
        "Type=exec\n", "Type=exec\nUser=nobody\n"
    )
    with pytest.raises(PermissionError, match="unauthorized directive"):
        realization.render_fragment(
            expanded, python_path=PYTHON, supervisor_path=SUPERVISOR,
            runtime_spec_path=Path("/absolute/future-spec.json"),
        )
    with pytest.raises(ValueError, match="allowlist"):
        realization._run(
            (realization.SYSTEMCTL, "--user", "start", realization.ACTUAL_UNIT),
            runner=lambda *_args, **_kwargs: pytest.fail("runner was called"),
        )


def test_expired_authorization_fails_before_fragment_and_writes_terminal(
    tmp_path: Path,
) -> None:
    evidence, unit_dir, alternate, generator_late, runtime_spec = _workspace(
        tmp_path
    )
    runner = FakeRunner(unit_dir, alternate, generator_late)
    _, payload = _authorize(evidence, runtime_spec, runner)
    expired_body = dict(payload)
    expired_body.pop("authorization_fingerprint")
    issued = datetime.now(timezone.utc) - timedelta(minutes=10)
    expired_body["issued_at_utc"] = issued.isoformat().replace("+00:00", "Z")
    expired_body["expires_at_utc"] = (
        issued + timedelta(minutes=5)
    ).isoformat().replace("+00:00", "Z")
    expired = evidence / "expired.json"
    realization.write_create_once_json(
        expired, expired_body, fingerprint_field="authorization_fingerprint"
    )
    with pytest.raises(PermissionError, match="stale"):
        realization.realize_actual_unit(
            expired, receipt_path=evidence / "receipt.json",
            terminal_path=evidence / "terminal.json", runner=runner,
            manager_reader=lambda: deepcopy(_manager()),
        )
    assert not (unit_dir / realization.ACTUAL_UNIT).exists()
    terminal = realization.load_sealed_json(
        evidence / "terminal.json", "terminal_fingerprint"
    )
    assert terminal["fragment_may_remain"] is False
    assert terminal["daemon_reload_attempted"] is False


def test_stable_file_read_rejects_parent_generation_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "sealed-parent"
    parent.mkdir(mode=0o700)
    target = parent / "input.json"
    target.write_bytes(b"same bytes\n")
    original_read = realization._read_all_fd
    rotated = False

    def read_then_rotate(descriptor: int) -> bytes:
        nonlocal rotated
        raw = original_read(descriptor)
        if not rotated:
            rotated = True
            parent.rename(tmp_path / "sealed-parent-before")
            parent.mkdir(mode=0o700)
            (parent / target.name).write_bytes(raw)
        return raw

    monkeypatch.setattr(realization, "_read_all_fd", read_then_rotate)
    with pytest.raises(PermissionError, match="parent generation changed"):
        realization._stable_read_file(target)


def test_stable_file_read_rejects_same_byte_leaf_inode_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "bound-source.py"
    target.write_bytes(b"same bytes\n")
    original_read = realization._read_all_fd
    replaced = False

    def read_then_replace(descriptor: int) -> bytes:
        nonlocal replaced
        raw = original_read(descriptor)
        if not replaced:
            replaced = True
            target.rename(target.with_name(f"{target.name}.before-race"))
            target.write_bytes(raw)
        return raw

    monkeypatch.setattr(realization, "_read_all_fd", read_then_replace)
    with pytest.raises(PermissionError, match="identity changed"):
        realization._stable_read_file(target)


def test_authorization_rejects_same_byte_template_inode_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence, unit_dir, alternate, generator_late, runtime_spec = _workspace(
        tmp_path
    )
    template = tmp_path / "actual.service.template"
    template.write_text(
        TEMPLATE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    runner = FakeRunner(unit_dir, alternate, generator_late)
    runner.runtime_spec = runtime_spec
    original = realization._read_file_binding
    replaced = False

    def bind_then_replace(
        path: Path, *, allow_symlink: bool = False,
    ) -> tuple[bytes, dict[str, object]]:
        nonlocal replaced
        raw, binding = original(path, allow_symlink=allow_symlink)
        if path == template and not replaced:
            replaced = True
            template.rename(template.with_name(f"{template.name}.before-race"))
            template.write_bytes(raw)
        return raw, binding

    monkeypatch.setattr(realization, "_read_file_binding", bind_then_replace)
    with pytest.raises(PermissionError, match="template changed"):
        realization.create_authorization(
            evidence / "authorization.json",
            template_path=template, python_path=PYTHON,
            supervisor_path=SUPERVISOR, runtime_spec_path=runtime_spec,
            authorization_basis=realization.AUTHORIZATION_BASIS,
            instruction_id=realization.INSTRUCTION_ID,
            runner=runner, manager_reader=lambda: deepcopy(_manager()),
        )
    assert not (evidence / "authorization.json").exists()


def test_fragment_same_bytes_replacement_inode_race_is_terminal(
    tmp_path: Path,
) -> None:
    evidence, unit_dir, alternate, generator_late, runtime_spec = _workspace(
        tmp_path
    )
    runner = FakeRunner(unit_dir, alternate, generator_late)
    authorization, _ = _authorize(evidence, runtime_spec, runner)
    runner.replace_fragment_on_shadow = True
    with pytest.raises(PermissionError, match="fragment identity drifted"):
        realization.realize_actual_unit(
            authorization, receipt_path=evidence / "receipt.json",
            terminal_path=evidence / "terminal.json", runner=runner,
            manager_reader=lambda: deepcopy(_manager()),
        )
    terminal = realization.load_sealed_json(
        evidence / "terminal.json", "terminal_fingerprint",
    )
    assert terminal["passed"] is False
    assert not (evidence / "receipt.json").exists()
