from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

import pytest

from tools import cure_lite_v24_realize_systemd_unit as realizer
from tools import cure_lite_v24_user_systemd_integration as integration
from tests_v24.test_gcr_pacre_v24_user_systemd_integration import (
    DUMMY_CHILD,
    FakeRunner,
    PYTHON,
    REALIZER,
    SCENARIO,
    SUPERVISOR,
    TEMPLATE,
    _completed,
    _manager,
)


_LATE_KEYS = {"path", "exists", "device", "inode", "owner_uid", "mode"}


def _policy(tmp_path: Path) -> dict[str, object]:
    runtime = (tmp_path / "run/systemd/user").resolve()
    late = runtime.parent / "generator.late"
    other = (tmp_path / "config/systemd/user").resolve()
    uid = os.getuid()
    return {
        "systemd_path_argv": ["/usr/bin/systemd-path"],
        "systemd_path_stdout": f"{runtime}\n",
        "systemd_analyze_argv": ["/usr/bin/systemd-analyze"],
        "systemd_analyze_stdout": f"{other}\n{late}\n{runtime}\n",
        "ordered_unit_paths": [
            {
                "path": str(other), "exists": True, "device": 10,
                "inode": 100, "owner_uid": uid, "mode": 0o755,
            },
            {
                "path": str(late), "exists": True, "device": 11,
                "inode": 101, "owner_uid": uid, "mode": 0o755,
            },
            {
                "path": str(runtime), "exists": True, "device": 12,
                "inode": 102, "owner_uid": uid, "mode": 0o700,
            },
        ],
        "runtime_directory": str(runtime),
        "runtime_directory_priority": 2,
        "runtime_directory_identity": {
            "path": str(runtime), "exists": True, "device": 12,
            "inode": 102, "owner_uid": uid, "mode": 0o700,
        },
    }


def _late_rotation(
    policy: dict[str, object], *, inode: object = 201,
) -> dict[str, object]:
    rotated = deepcopy(policy)
    rotated["ordered_unit_paths"][1]["inode"] = inode
    return rotated


def _allow(before: object, after: object) -> None:
    integration._validate_unit_path_policy_transition(
        before,
        after,
        authorized_uid=os.getuid(),
        allow_generator_late_inode_rotation=True,
    )


@pytest.mark.parametrize(
    ("section", "field", "drift"),
    [
        ("identity", "uid", float(os.getuid())),
        ("endpoint", "uid", float(os.getuid())),
        ("identity", "pid", 111.0),
        ("identity", "starttime_ticks", True),
        ("endpoint", "runtime_device", 11.0),
        ("endpoint", "runtime_inode", True),
        ("endpoint", "bus_device", 13.0),
        ("endpoint", "bus_inode", True),
    ],
)
def test_manager_generation_rejects_numeric_type_coercion(
    section: str, field: str, drift: object,
) -> None:
    manager = deepcopy(_manager())
    manager[section][field] = drift
    with pytest.raises(PermissionError, match="manager generation"):
        integration._validate_manager_generation(manager)


def test_policy_transition_is_exact_by_default_and_allows_only_late_inode(
    tmp_path: Path,
) -> None:
    before = _policy(tmp_path)
    integration._validate_unit_path_policy_transition(
        before, deepcopy(before), authorized_uid=os.getuid(),
    )
    with pytest.raises(PermissionError, match="policy changed"):
        integration._validate_unit_path_policy_transition(
            before, _late_rotation(before), authorized_uid=os.getuid(),
        )
    _allow(before, _late_rotation(before))

    type_drift = deepcopy(before)
    type_drift["runtime_directory_priority"] = 2.0
    with pytest.raises(PermissionError):
        integration._validate_unit_path_policy_transition(
            before, type_drift, authorized_uid=os.getuid(),
        )


@pytest.mark.parametrize("side", ["old", "new"])
@pytest.mark.parametrize("bad_inode", [True, 0, -1, "201"])
def test_policy_transition_rejects_malformed_old_or_new_inode(
    tmp_path: Path, side: str, bad_inode: object,
) -> None:
    before = _policy(tmp_path)
    after = _late_rotation(before)
    target = before if side == "old" else after
    target["ordered_unit_paths"][1]["inode"] = bad_inode
    with pytest.raises(PermissionError, match="identity is unsafe"):
        _allow(before, after)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device", 99),
        ("owner_uid", os.getuid() + 1),
        ("mode", 0o700),
    ],
)
def test_policy_transition_rejects_device_uid_or_mode_change(
    tmp_path: Path, field: str, value: int,
) -> None:
    before = _policy(tmp_path)
    after = _late_rotation(before)
    after["ordered_unit_paths"][1][field] = value
    with pytest.raises(PermissionError):
        _allow(before, after)


def test_policy_transition_rejects_unsafe_mode_even_when_unchanged(
    tmp_path: Path,
) -> None:
    before = _policy(tmp_path)
    before["ordered_unit_paths"][1]["mode"] = 0o777
    after = _late_rotation(before)
    with pytest.raises(PermissionError, match="identity is unsafe"):
        _allow(before, after)


@pytest.mark.parametrize("schema_change", ["extra", "missing", "absent"])
def test_policy_transition_rejects_nonexact_late_schema(
    tmp_path: Path, schema_change: str,
) -> None:
    before = _policy(tmp_path)
    after = _late_rotation(before)
    if schema_change == "extra":
        after["ordered_unit_paths"][1]["path_is_symlink"] = False
    elif schema_change == "missing":
        del after["ordered_unit_paths"][1]["device"]
    else:
        before["ordered_unit_paths"][1]["exists"] = False
        after["ordered_unit_paths"][1]["exists"] = False
    assert set(before["ordered_unit_paths"][1]) == _LATE_KEYS
    with pytest.raises(PermissionError, match="schema is not exact"):
        _allow(before, after)


def test_policy_transition_rejects_wrong_sibling_duplicate_and_reorder(
    tmp_path: Path,
) -> None:
    before = _policy(tmp_path)

    wrong_before = deepcopy(before)
    wrong_after = deepcopy(before)
    for policy in (wrong_before, wrong_after):
        policy["ordered_unit_paths"][1]["path"] = str(
            Path(policy["runtime_directory"]).parent / "generator.early"
        )
    wrong_after["ordered_unit_paths"][1]["inode"] = 201
    with pytest.raises(PermissionError, match="only one"):
        _allow(wrong_before, wrong_after)

    duplicate_before = deepcopy(before)
    duplicate_before["ordered_unit_paths"].insert(
        2, deepcopy(duplicate_before["ordered_unit_paths"][1])
    )
    duplicate_after = deepcopy(duplicate_before)
    duplicate_after["ordered_unit_paths"][1]["inode"] = 201
    with pytest.raises(PermissionError, match="only one"):
        _allow(duplicate_before, duplicate_after)

    reordered = _late_rotation(before)
    reordered["ordered_unit_paths"][0], reordered["ordered_unit_paths"][1] = (
        reordered["ordered_unit_paths"][1],
        reordered["ordered_unit_paths"][0],
    )
    with pytest.raises(PermissionError, match="only one"):
        _allow(before, reordered)


def test_policy_transition_rejects_other_path_and_combined_differences(
    tmp_path: Path,
) -> None:
    before = _policy(tmp_path)
    other_inode = deepcopy(before)
    other_inode["ordered_unit_paths"][0]["inode"] = 999
    with pytest.raises(PermissionError, match="only one"):
        _allow(before, other_inode)

    combined = _late_rotation(before)
    combined["systemd_analyze_stdout"] += "/unexpected\n"
    with pytest.raises(PermissionError, match="accompanied another"):
        _allow(before, combined)


class _PhasedPolicyRunner(FakeRunner):
    def __init__(self, unit_dir: Path, late: Path, other: Path) -> None:
        super().__init__(unit_dir, other)
        self.late = late
        self.other = other
        self.analyze_count = 0
        self.rotations: dict[int, Path] = {}
        self.reappearances: dict[int, Path] = {}
        self.not_found_overrides: dict[str, str] = {}
        self.reload_count = 0
        self._retired: list[Path] = []

    def _rotate(self, path: Path) -> None:
        previous_inode = path.stat().st_ino
        retired = path.with_name(f"{path.name}.retired-{len(self._retired)}")
        path.rename(retired)
        path.mkdir(mode=0o755)
        path.chmod(0o755)
        assert path.stat().st_ino != previous_inode
        self._retired.append(retired)

    def __call__(self, argv):
        command = tuple(argv)
        if command[0] == realizer.SYSTEMD_ANALYZE:
            self.calls.append(command)
            self.analyze_count += 1
            rotate = self.rotations.get(self.analyze_count)
            if rotate is not None:
                self._rotate(rotate)
            reappear = self.reappearances.get(self.analyze_count)
            if reappear is not None:
                assert self.authorization is not None
                candidate = (
                    reappear / str(self.authorization["identity"]["unit_name"])
                )
                candidate.write_text("reappeared shadow\n", encoding="utf-8")
                candidate.chmod(0o600)
            return _completed(
                argv,
                stdout=f"{self.other}\n{self.late}\n{self.unit_dir}\n",
            )
        if command[:3] == (
            realizer.SYSTEMCTL_PATH, "--user", "daemon-reload"
        ):
            self.reload_count += 1
        completed = super().__call__(argv)
        if (
            command[:3] == (realizer.SYSTEMCTL_PATH, "--user", "show")
            and self.reload_count >= 2
            and not any(
                value.startswith("--property=Type") for value in command
            )
            and self.not_found_overrides
        ):
            values = dict(
                row.split("=", 1)
                for row in completed.stdout.splitlines() if "=" in row
            )
            values.update(self.not_found_overrides)
            return _completed(
                argv,
                stdout="".join(
                    f"{key}={value}\n" for key, value in values.items()
                ),
            )
        return completed


def _late_workspace(
    tmp_path: Path,
) -> tuple[Path, _PhasedPolicyRunner, dict[str, object], Path]:
    unit_dir = tmp_path / "run/systemd/user"
    late = tmp_path / "run/systemd/generator.late"
    other = tmp_path / "config/systemd/user"
    unit_dir.mkdir(parents=True)
    late.mkdir()
    other.mkdir(parents=True)
    unit_dir.chmod(0o700)
    late.chmod(0o755)
    other.chmod(0o755)
    runner = _PhasedPolicyRunner(
        unit_dir.resolve(), late.resolve(), other.resolve()
    )
    scenario_root = (tmp_path / "scenario").resolve()
    authorization = integration.create_production_authorization(
        scenario_root,
        scenario_id=SCENARIO,
        template_path=TEMPLATE.resolve(),
        python_path=PYTHON,
        supervisor_path=SUPERVISOR.resolve(),
        realizer_path=REALIZER.resolve(),
        dummy_child_path=DUMMY_CHILD.resolve(),
        instruction_id=integration.INSTRUCTION_ID,
        authorization_basis=integration.AUTHORIZATION_BASIS,
        runner=runner,
        manager_reader=lambda: deepcopy(_manager()),
    )
    runner.authorization = authorization
    fragment = (
        unit_dir.resolve() / str(authorization["identity"]["unit_name"])
    )
    return scenario_root, runner, authorization, fragment


def _run(
    scenario_root: Path,
    runner: _PhasedPolicyRunner,
    *,
    manager_reader=None,
) -> dict[str, object]:
    return integration.run_authorized_integration(
        scenario_root / "control/authorization.json",
        execute=True,
        runner=runner,
        manager_reader=(
            manager_reader
            if manager_reader is not None
            else lambda: deepcopy(_manager())
        ),
        timeout_seconds=0.2,
    )


def test_flow_accepts_only_post_reload_and_post_remove_late_rotations(
    tmp_path: Path,
) -> None:
    scenario_root, runner, _authorization, fragment = _late_workspace(tmp_path)
    runner.rotations = {4: runner.late, 8: runner.late}
    result = _run(scenario_root, runner)
    assert result["receipt"]["passed"] is True
    assert result["removal_state"]["passed"] is True
    assert runner.analyze_count == 9
    assert not fragment.exists()


def test_flow_rejects_pre_reload_rotation_before_daemon_reload(
    tmp_path: Path,
) -> None:
    scenario_root, runner, _authorization, fragment = _late_workspace(tmp_path)
    runner.rotations = {3: runner.late}
    with pytest.raises(PermissionError, match="policy changed"):
        _run(scenario_root, runner)
    assert fragment.exists()
    assert not any(call[:3] == (
        realizer.SYSTEMCTL_PATH, "--user", "daemon-reload"
    ) for call in runner.calls)


def test_flow_rejects_second_terminal_rotation_without_unlink(
    tmp_path: Path,
) -> None:
    scenario_root, runner, _authorization, fragment = _late_workspace(tmp_path)
    runner.rotations = {4: runner.late, 6: runner.late}
    with pytest.raises(PermissionError, match="policy changed"):
        _run(scenario_root, runner)
    assert fragment.exists()
    assert runner.analyze_count == 6


def test_flow_rejects_rotation_while_loading_removal_without_unlink(
    tmp_path: Path,
) -> None:
    scenario_root, runner, _authorization, fragment = _late_workspace(tmp_path)
    runner.rotations = {4: runner.late, 7: runner.late}
    with pytest.raises(PermissionError, match="policy changed"):
        _run(scenario_root, runner)
    assert fragment.exists()
    assert runner.analyze_count == 7


def test_flow_rejects_post_remove_non_late_difference(
    tmp_path: Path,
) -> None:
    scenario_root, runner, _authorization, fragment = _late_workspace(tmp_path)
    runner.rotations = {4: runner.late, 8: runner.other}
    with pytest.raises(PermissionError, match="only one"):
        _run(scenario_root, runner)
    assert not fragment.exists()
    assert runner.analyze_count == 8


def test_flow_rejects_second_post_remove_late_rotation(
    tmp_path: Path,
) -> None:
    scenario_root, runner, _authorization, fragment = _late_workspace(tmp_path)
    runner.rotations = {4: runner.late, 8: runner.late, 9: runner.late}
    with pytest.raises(PermissionError, match="policy changed"):
        _run(scenario_root, runner)
    assert not fragment.exists()
    assert runner.analyze_count == 9


@pytest.mark.parametrize("location", ["runtime", "shadow"])
def test_flow_rejects_fragment_or_shadow_reappearance(
    tmp_path: Path, location: str,
) -> None:
    scenario_root, runner, _authorization, fragment = _late_workspace(tmp_path)
    runner.rotations = {4: runner.late, 8: runner.late}
    directory = runner.unit_dir if location == "runtime" else runner.other
    runner.reappearances = {9: directory}
    with pytest.raises(PermissionError, match="shadowed"):
        _run(scenario_root, runner)
    candidate = directory / fragment.name
    assert candidate.exists()


def test_flow_rejects_manager_generation_change_after_not_found(
    tmp_path: Path,
) -> None:
    scenario_root, runner, _authorization, fragment = _late_workspace(tmp_path)
    calls = 0

    def drifting_manager() -> dict[str, object]:
        nonlocal calls
        calls += 1
        value = deepcopy(_manager())
        if calls >= 5:
            value["identity"]["starttime_ticks"] = 999
        return value

    with pytest.raises(PermissionError, match="changed after removal"):
        _run(scenario_root, runner, manager_reader=drifting_manager)
    assert calls == 5
    assert not fragment.exists()


def test_flow_hard_rejects_fragment_reappearance_after_final_freeze(
    tmp_path: Path,
) -> None:
    scenario_root, runner, _authorization, fragment = _late_workspace(tmp_path)
    calls = 0

    def reappearing_fragment_manager() -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 5:
            fragment.write_text("late fragment reappearance\n", encoding="utf-8")
            fragment.chmod(0o600)
        return deepcopy(_manager())

    with pytest.raises(PermissionError, match="fragment reappeared"):
        _run(
            scenario_root,
            runner,
            manager_reader=reappearing_fragment_manager,
        )
    assert calls == 5
    assert fragment.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("DropInPaths", "/tmp/reappeared.conf"),
        ("Transient", "yes"),
        ("Restart", "always"),
        ("NRestarts", "1"),
        ("NeedDaemonReload", "yes"),
    ],
)
def test_flow_rejects_not_found_omitted_field_drift(
    tmp_path: Path, field: str, value: str,
) -> None:
    scenario_root, runner, _authorization, fragment = _late_workspace(tmp_path)
    runner.not_found_overrides = {field: value}
    with pytest.raises(PermissionError, match="not exact not-found"):
        _run(scenario_root, runner)
    assert not fragment.exists()
    assert runner.analyze_count == 8
