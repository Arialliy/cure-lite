from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import stat
import subprocess
from types import SimpleNamespace
from typing import Callable

import pytest

from tests_v24 import (
    test_cure_lite_v24_actual_unit_realization as realization_fixtures,
)
from tools import cure_lite_v24_actual_runtime_release as release
from tools import cure_lite_v24_actual_unit_realization as legacy
from tools import cure_lite_v24_actual_unit_realization_recovery as recovery


ROOT = Path(__file__).resolve().parents[1]
OLD_REALIZER_SHA256 = (
    "0d66bc4007366588ed1393b21092cc57d58e0f7fca084f7266a00e6818703fd9"
)
OLD_SUPERVISOR_SHA256 = (
    "b955ba8ffe869d324cc9319f8031180989746053d7ceec5e50bd12eb19faeeed"
)


class SuccessExitStatusRecoveryRunner:
    """Expose the exact live representation that stopped the old realizer."""

    def __init__(
        self,
        inner: realization_fixtures.FakeRunner,
        *,
        on_live_shadow: Callable[[int], None] | None = None,
    ) -> None:
        self.inner = inner
        self.on_live_shadow = on_live_shadow
        self.live_shadow_calls = 0

    def __call__(
        self,
        argv: list[str],
        **kwargs: object,
    ) -> SimpleNamespace:
        result = self.inner(argv, **kwargs)
        if (
            tuple(argv) == legacy._shadow_argv()
            and self.inner.reloaded
        ):
            self.live_shadow_calls += 1
            if self.on_live_shadow is not None:
                self.on_live_shadow(self.live_shadow_calls)
            old = "SuccessExitStatus=0 0\n"
            assert old in result.stdout
            result.stdout = result.stdout.replace(
                old,
                "SuccessExitStatus=0\n",
                1,
            )
        return result


class FailedRealization:
    def __init__(
        self,
        *,
        evidence: Path,
        runtime_spec: Path,
        runtime_launch_authorization: Path,
        original_authorization: Path,
        terminal: Path,
        normal_receipt: Path,
        recovery_authorization: Path,
        recovery_receipt: Path,
        runner: SuccessExitStatusRecoveryRunner,
    ) -> None:
        self.evidence = evidence
        self.runtime_spec = runtime_spec
        self.runtime_launch_authorization = runtime_launch_authorization
        self.original_authorization = original_authorization
        self.terminal = terminal
        self.normal_receipt = normal_receipt
        self.recovery_authorization = recovery_authorization
        self.recovery_receipt = recovery_receipt
        self.runner = runner


def _manager() -> dict[str, object]:
    return deepcopy(realization_fixtures._manager())


def _patch_fixed_paths(
    monkeypatch: pytest.MonkeyPatch,
    case: FailedRealization,
) -> None:
    fixed = {
        "EVIDENCE_ROOT": case.evidence,
        "ORIGINAL_AUTHORIZATION_PATH": case.original_authorization,
        "FAILURE_TERMINAL_PATH": case.terminal,
        "NORMAL_RECEIPT_PATH": case.normal_receipt,
        "RECOVERY_AUTHORIZATION_PATH": case.recovery_authorization,
        "RECOVERY_RECEIPT_PATH": case.recovery_receipt,
        "RUNTIME_SPEC_PATH": case.runtime_spec,
        "RUNTIME_LAUNCH_AUTHORIZATION_PATH": (
            case.runtime_launch_authorization
        ),
    }
    for name, path in fixed.items():
        monkeypatch.setattr(recovery, name, path)
    monkeypatch.setattr(release, "RUNTIME_SPEC_PATH", case.runtime_spec)
    monkeypatch.setattr(
        release,
        "RUNTIME_LAUNCH_AUTHORIZATION_PATH",
        case.runtime_launch_authorization,
    )


def _failed_realization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> FailedRealization:
    (
        evidence,
        unit_directory,
        alternate_directory,
        generator_late,
        runtime_spec,
    ) = realization_fixtures._workspace(tmp_path)
    inner = realization_fixtures.FakeRunner(
        unit_directory,
        alternate_directory,
        generator_late,
    )
    original_authorization, _ = realization_fixtures._authorize(
        evidence,
        runtime_spec,
        inner,
    )
    runner = SuccessExitStatusRecoveryRunner(inner)
    normal_receipt = evidence / "normal-realization-receipt.json"
    terminal = evidence / "failed-realization-terminal.json"
    with pytest.raises(
        PermissionError,
        match="installed actual unit shadow is not exact static",
    ):
        legacy.realize_actual_unit(
            original_authorization,
            receipt_path=normal_receipt,
            terminal_path=terminal,
            runner=runner,
            manager_reader=_manager,
        )
    case = FailedRealization(
        evidence=evidence,
        runtime_spec=runtime_spec,
        runtime_launch_authorization=(
            evidence / "runtime-launch-authorization.json"
        ),
        original_authorization=original_authorization,
        terminal=terminal,
        normal_receipt=normal_receipt,
        recovery_authorization=(
            evidence / "realization-recovery-authorization.json"
        ),
        recovery_receipt=(
            evidence / "realization-recovery-receipt.json"
        ),
        runner=runner,
    )
    _patch_fixed_paths(monkeypatch, case)
    return case


def _seal_recovery(case: FailedRealization) -> tuple[
    dict[str, object],
    dict[str, object],
]:
    authorization = recovery.create_recovery_authorization(
        case.recovery_authorization,
        original_authorization_path=case.original_authorization,
        failure_terminal_path=case.terminal,
        normal_receipt_path=case.normal_receipt,
        runtime_spec_path=case.runtime_spec,
        runtime_launch_authorization_path=(
            case.runtime_launch_authorization
        ),
        runner=case.runner,
        manager_reader=_manager,
    )
    receipt = recovery.seal_recovery_receipt(
        case.recovery_authorization,
        receipt_path=case.recovery_receipt,
        runner=case.runner,
        manager_reader=_manager,
    )
    return authorization, receipt


def _shadow_reader(
    case: FailedRealization,
) -> dict[str, str]:
    return legacy.query_shadow(runner=case.runner)


def _policy_reader(
    case: FailedRealization,
    fragment: Path,
) -> dict[str, object]:
    return legacy._observe_unit_path_policy(
        runner=case.runner,
        allowed_fragment=fragment.absolute(),
    )


def _validate_release(case: FailedRealization) -> dict[str, object]:
    return release._validate_realization_chain(
        authorization_path=case.recovery_authorization,
        receipt_path=case.recovery_receipt,
        shadow_reader=lambda: _shadow_reader(case),
        manager_reader=_manager,
        unit_path_policy_reader=lambda fragment: _policy_reader(
            case,
            fragment,
        ),
    )


def _replace_sealed(
    path: Path,
    payload: dict[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    body = deepcopy(payload)
    body.pop(fingerprint_field, None)
    path.chmod(0o600)
    path.unlink()
    return legacy.write_create_once_json(
        path,
        body,
        fingerprint_field=fingerprint_field,
    )


def test_direct_isolated_cli_imports_from_repository_root() -> None:
    result = subprocess.run(
        [
            str(realization_fixtures.PYTHON),
            "-I",
            "-S",
            "-B",
            "-u",
            str(Path(recovery.__file__).resolve()),
            "--help",
        ],
        cwd=ROOT,
        shell=False,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "authorize-recovery" in result.stdout
    assert "seal-receipt" in result.stdout


def test_sealed_old_realizer_and_supervisor_sources_are_unchanged() -> None:
    assert legacy.file_sha256(Path(legacy.__file__).resolve()) == (
        OLD_REALIZER_SHA256
    )
    assert legacy.file_sha256(
        ROOT / "tools/cure_lite_v24_runtime_supervisor.py",
    ) == OLD_SUPERVISOR_SHA256


def test_exact_unique_delta_closes_without_any_recovery_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _failed_realization(tmp_path, monkeypatch)
    terminal_before = legacy._file_binding(case.terminal)
    recovery_command_start = len(case.runner.inner.commands)

    authorization, receipt = _seal_recovery(case)
    chain = _validate_release(case)

    assert authorization["compatibility_exception"] == {
        "property": "SuccessExitStatus",
        "classification": (
            "systemd_v255_explicit_zero_default_zero_deduplication"
        ),
        "authorized_expected_value": "0 0",
        "live_observed_value": "0",
        "rendered_directive_values": ["0"],
        "semantic_success_exit_statuses": [0],
        "all_other_static_properties_exact": True,
        "projected_legacy_shadow_fingerprint": (
            authorization["compatibility_exception"][
                "projected_legacy_shadow_fingerprint"
            ]
        ),
        "live_validated_shadow_fingerprint": (
            authorization["compatibility_exception"][
                "live_validated_shadow_fingerprint"
            ]
        ),
    }
    assert receipt["normal_realization_passed"] is False
    assert receipt["recovery_acceptance_passed"] is True
    assert receipt["historical_failure_preserved"] is True
    assert receipt["passed"] is True
    assert set(receipt["recovery_authorization_root"]) == (
        recovery._EVIDENCE_ROOT_KEYS
    )
    assert receipt["recovery_authorization_root"]["path"] == str(
        case.recovery_authorization,
    )
    assert receipt["full_static_shadow"]["SuccessExitStatus"] == "0"
    assert receipt["completed_actions"] == recovery.RECOVERY_ACTIONS
    assert stat.S_IMODE(case.recovery_authorization.stat().st_mode) == 0o444
    assert stat.S_IMODE(case.recovery_receipt.stat().st_mode) == 0o444
    assert legacy._file_binding(case.terminal) == terminal_before
    assert not case.normal_receipt.exists()
    assert not case.runtime_spec.exists()
    assert not case.runtime_launch_authorization.exists()

    recovery_commands = case.runner.inner.commands[recovery_command_start:]
    forbidden = {
        "daemon-reload",
        "enable",
        "start",
        "stop",
        "disable",
        "reset-failed",
        "remove",
    }
    assert all(
        not (
            command
            and command[0] == legacy.SYSTEMCTL
            and any(token in forbidden for token in command[1:])
        )
        for command in recovery_commands
    )
    assert case.runner.inner.commands.count(
        (legacy.SYSTEMCTL, "--user", "daemon-reload"),
    ) == 1

    assert set(chain) == {
        "authorization",
        "receipt",
        "authorization_identity",
        "receipt_identity",
        "live_unit_path_policy",
        "live_shadow",
        "validated_shadow",
    }
    assert chain["authorization"]["schema_version"] == (
        legacy.AUTHORIZATION_SCHEMA
    )
    assert chain["receipt"]["schema_version"] == recovery.RECOVERY_RECEIPT_SCHEMA
    assert chain["authorization_identity"]["file_sha256"] == (
        legacy.file_sha256(case.recovery_authorization)
    )
    assert chain["receipt_identity"]["file_sha256"] == (
        legacy.file_sha256(case.recovery_receipt)
    )
    assert chain["live_shadow"]["SuccessExitStatus"] == "0"
    assert release._realizer_python_source_fields(chain)["python_path"] == (
        str(realization_fixtures.PYTHON)
    )
    immutable = release._immutable_shadow(chain)
    assert set(immutable) == set(
        release.supervisor._SYSTEMD_IMMUTABLE_SHADOW_KEYS
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Restart", "on-failure"),
        ("SuccessExitStatus", "0 1"),
        ("NRestarts", "1"),
    ],
)
def test_any_shadow_delta_beyond_the_one_exception_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    case = _failed_realization(tmp_path, monkeypatch)
    _seal_recovery(case)

    def drifted_shadow() -> dict[str, str]:
        shadow = _shadow_reader(case)
        shadow[field] = value
        return shadow

    with pytest.raises(PermissionError):
        release._validate_realization_chain(
            authorization_path=case.recovery_authorization,
            receipt_path=case.recovery_receipt,
            shadow_reader=drifted_shadow,
            manager_reader=_manager,
            unit_path_policy_reader=lambda fragment: _policy_reader(
                case,
                fragment,
            ),
        )


@pytest.mark.parametrize(
    "drift",
    [
        "historical-failure-boolean",
        "recovery-authorization-root",
        "terminal-root",
        "source-binding",
        "sealed-shadow",
    ],
)
def test_resealed_recovery_receipt_drift_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    case = _failed_realization(tmp_path, monkeypatch)
    _, receipt = _seal_recovery(case)
    changed = deepcopy(receipt)
    if drift == "historical-failure-boolean":
        changed["historical_failure_preserved"] = False
    elif drift == "recovery-authorization-root":
        changed["recovery_authorization_root"]["ctime_ns"] += 1
    elif drift == "terminal-root":
        changed["failure_terminal"]["inode"] += 1
    elif drift == "source-binding":
        changed["source_bindings"]["release_consumer"][
            "file_sha256"
        ] = "0" * 64
    else:
        changed["full_static_shadow"]["Restart"] = "on-failure"
    _replace_sealed(
        case.recovery_receipt,
        changed,
        fingerprint_field="receipt_fingerprint",
    )

    with pytest.raises(PermissionError):
        _validate_release(case)


def test_historical_failure_terminal_drift_is_rejected_before_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _failed_realization(tmp_path, monkeypatch)
    terminal = legacy.load_sealed_json(
        case.terminal,
        "terminal_fingerprint",
    )
    terminal["start_attempted"] = True
    _replace_sealed(
        case.terminal,
        terminal,
        fingerprint_field="terminal_fingerprint",
    )

    with pytest.raises(
        PermissionError,
        match="historical realization failure is not exact",
    ):
        recovery.create_recovery_authorization(
            case.recovery_authorization,
            original_authorization_path=case.original_authorization,
            failure_terminal_path=case.terminal,
            normal_receipt_path=case.normal_receipt,
            runtime_spec_path=case.runtime_spec,
            runtime_launch_authorization_path=(
                case.runtime_launch_authorization
            ),
            runner=case.runner,
            manager_reader=_manager,
        )
    assert not case.recovery_authorization.exists()


def test_preexisting_fixed_recovery_receipt_blocks_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _failed_realization(tmp_path, monkeypatch)
    case.recovery_receipt.write_text("stale\n", encoding="utf-8")
    case.recovery_receipt.chmod(0o444)

    with pytest.raises(
        PermissionError,
        match="recovery receipt must remain absent",
    ):
        recovery.create_recovery_authorization(
            case.recovery_authorization,
            original_authorization_path=case.original_authorization,
            failure_terminal_path=case.terminal,
            normal_receipt_path=case.normal_receipt,
            runtime_spec_path=case.runtime_spec,
            runtime_launch_authorization_path=(
                case.runtime_launch_authorization
            ),
            runner=case.runner,
            manager_reader=_manager,
        )
    assert not case.recovery_authorization.exists()


@pytest.mark.parametrize(
    "predecessor",
    ["recovery-authorization", "failure-terminal"],
)
def test_seal_rejects_predecessor_replacement_during_second_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    predecessor: str,
) -> None:
    case = _failed_realization(tmp_path, monkeypatch)
    authorization = recovery.create_recovery_authorization(
        case.recovery_authorization,
        original_authorization_path=case.original_authorization,
        failure_terminal_path=case.terminal,
        normal_receipt_path=case.normal_receipt,
        runtime_spec_path=case.runtime_spec,
        runtime_launch_authorization_path=(
            case.runtime_launch_authorization
        ),
        runner=case.runner,
        manager_reader=_manager,
    )
    terminal = legacy.load_sealed_json(
        case.terminal,
        "terminal_fingerprint",
    )
    baseline = case.runner.live_shadow_calls

    def replace_predecessor(call: int) -> None:
        if call != baseline + 2:
            return
        if predecessor == "recovery-authorization":
            _replace_sealed(
                case.recovery_authorization,
                authorization,
                fingerprint_field="authorization_fingerprint",
            )
        else:
            _replace_sealed(
                case.terminal,
                terminal,
                fingerprint_field="terminal_fingerprint",
            )

    case.runner.on_live_shadow = replace_predecessor
    with pytest.raises(PermissionError):
        recovery.seal_recovery_receipt(
            case.recovery_authorization,
            receipt_path=case.recovery_receipt,
            runner=case.runner,
            manager_reader=_manager,
        )
    assert not case.recovery_receipt.exists()


def test_final_validator_rejects_receipt_replacement_on_second_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _failed_realization(tmp_path, monkeypatch)
    _, receipt = _seal_recovery(case)
    baseline = case.runner.live_shadow_calls

    def replace_receipt(call: int) -> None:
        if call == baseline + 2:
            _replace_sealed(
                case.recovery_receipt,
                receipt,
                fingerprint_field="receipt_fingerprint",
            )

    case.runner.on_live_shadow = replace_receipt
    with pytest.raises(
        PermissionError,
        match="recovery receipt changed during validation",
    ):
        _validate_release(case)


@pytest.mark.parametrize("chronology", ["zero-duration", "expired"])
def test_invalid_recovery_authorization_lifetime_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chronology: str,
) -> None:
    case = _failed_realization(tmp_path, monkeypatch)
    authorization = recovery.create_recovery_authorization(
        case.recovery_authorization,
        original_authorization_path=case.original_authorization,
        failure_terminal_path=case.terminal,
        normal_receipt_path=case.normal_receipt,
        runtime_spec_path=case.runtime_spec,
        runtime_launch_authorization_path=(
            case.runtime_launch_authorization
        ),
        runner=case.runner,
        manager_reader=_manager,
    )
    changed = deepcopy(authorization)
    if chronology == "zero-duration":
        instant = changed["created_at_utc"]
        changed["issued_at_utc"] = instant
        changed["expires_at_utc"] = instant
    else:
        now = datetime.now(timezone.utc)
        changed["issued_at_utc"] = (
            now - timedelta(seconds=10)
        ).isoformat().replace("+00:00", "Z")
        changed["created_at_utc"] = changed["issued_at_utc"]
        changed["expires_at_utc"] = (
            now - timedelta(seconds=5)
        ).isoformat().replace("+00:00", "Z")
    _replace_sealed(
        case.recovery_authorization,
        changed,
        fingerprint_field="authorization_fingerprint",
    )

    with pytest.raises(
        PermissionError,
        match="recovery authorization chronology changed",
    ):
        recovery.seal_recovery_receipt(
            case.recovery_authorization,
            receipt_path=case.recovery_receipt,
            runner=case.runner,
            manager_reader=_manager,
        )
    assert not case.recovery_receipt.exists()


def test_predecessor_replacement_is_rejected_by_exact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _failed_realization(tmp_path, monkeypatch)
    _seal_recovery(case)
    original = legacy.load_sealed_json(
        case.original_authorization,
        "authorization_fingerprint",
    )
    _replace_sealed(
        case.original_authorization,
        original,
        fingerprint_field="authorization_fingerprint",
    )

    with pytest.raises(PermissionError, match="root was replaced"):
        _validate_release(case)


def test_same_bytes_recovery_authorization_replacement_after_seal_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _failed_realization(tmp_path, monkeypatch)
    authorization, receipt = _seal_recovery(case)
    sealed_root = receipt["recovery_authorization_root"]
    assert sealed_root["parent_device"] == (
        case.recovery_authorization.parent.stat().st_dev
    )
    assert sealed_root["parent_inode"] == (
        case.recovery_authorization.parent.stat().st_ino
    )

    _replace_sealed(
        case.recovery_authorization,
        authorization,
        fingerprint_field="authorization_fingerprint",
    )
    replaced_generation = recovery._evidence_generation(
        case.recovery_authorization,
    )
    assert any(
        replaced_generation[field] != sealed_root[field]
        for field in ("inode", "mtime_ns", "ctime_ns")
    )
    assert replaced_generation["parent_device"] == (
        sealed_root["parent_device"]
    )
    assert replaced_generation["parent_inode"] == (
        sealed_root["parent_inode"]
    )

    with pytest.raises(PermissionError, match="root was replaced"):
        _validate_release(case)


def test_off_path_recovery_authorization_root_is_rejected_before_root_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _failed_realization(tmp_path, monkeypatch)
    authorization, receipt = _seal_recovery(case)
    decoy = case.evidence / "decoy-recovery-authorization.json"
    decoy_body = deepcopy(authorization)
    decoy_body.pop("authorization_fingerprint")
    legacy.write_create_once_json(
        decoy,
        decoy_body,
        fingerprint_field="authorization_fingerprint",
    )
    changed = deepcopy(receipt)
    changed["recovery_authorization_root"]["path"] = str(decoy)
    _replace_sealed(
        case.recovery_receipt,
        changed,
        fingerprint_field="receipt_fingerprint",
    )
    cached_authorization = recovery._validate_recovery_authorization(
        case.recovery_authorization,
        require_fresh=False,
    )
    monkeypatch.setattr(
        recovery,
        "_validate_recovery_authorization",
        lambda *_args, **_kwargs: cached_authorization,
    )
    root_calls = 0

    def forbidden_root_read(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        nonlocal root_calls
        root_calls += 1
        pytest.fail("off-path root was read before fixed-path preflight")

    monkeypatch.setattr(recovery, "_validate_root", forbidden_root_read)
    with pytest.raises(
        PermissionError,
        match="root differs from the fixed path",
    ):
        recovery.validate_release_recovery_chain(
            recovery_authorization_path=case.recovery_authorization,
            recovery_receipt_path=case.recovery_receipt,
            shadow_reader=lambda: pytest.fail(
                "live shadow reached after off-path root",
            ),
            manager_reader=lambda: pytest.fail(
                "manager reached after off-path root",
            ),
            unit_path_policy_reader=lambda _fragment: pytest.fail(
                "unit path reached after off-path root",
            ),
        )
    assert root_calls == 0


def test_fixed_recovery_paths_and_absent_outputs_are_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _failed_realization(tmp_path, monkeypatch)
    _seal_recovery(case)
    with pytest.raises(PermissionError, match="fixed recovery path"):
        recovery.validate_release_recovery_chain(
            recovery_authorization_path=(
                case.evidence / "alternate-recovery-authorization.json"
            ),
            recovery_receipt_path=case.recovery_receipt,
            shadow_reader=lambda: _shadow_reader(case),
            manager_reader=_manager,
            unit_path_policy_reader=lambda fragment: _policy_reader(
                case,
                fragment,
            ),
        )

    case.runtime_spec.write_text("unauthorized\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="runtime spec must remain absent"):
        _validate_release(case)


def test_second_live_observation_drift_rejects_authorization_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _failed_realization(tmp_path, monkeypatch)
    inner = case.runner.inner
    inner.rotate_on_analyze_call = inner.analyze_calls + 3

    with pytest.raises(
        PermissionError,
        match="recovery live observation changed before commit",
    ):
        recovery.create_recovery_authorization(
            case.recovery_authorization,
            original_authorization_path=case.original_authorization,
            failure_terminal_path=case.terminal,
            normal_receipt_path=case.normal_receipt,
            runtime_spec_path=case.runtime_spec,
            runtime_launch_authorization_path=(
                case.runtime_launch_authorization
            ),
            runner=case.runner,
            manager_reader=_manager,
        )
    assert not case.recovery_authorization.exists()


def test_absence_and_source_are_rechecked_after_second_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _failed_realization(tmp_path, monkeypatch)
    original_runner = case.runner

    def introduce_absent_output(call: int) -> None:
        if call == 2:
            case.runtime_spec.write_text("raced\n", encoding="utf-8")

    case.runner = SuccessExitStatusRecoveryRunner(
        original_runner.inner,
        on_live_shadow=introduce_absent_output,
    )
    with pytest.raises(PermissionError, match="runtime spec must remain absent"):
        recovery.create_recovery_authorization(
            case.recovery_authorization,
            original_authorization_path=case.original_authorization,
            failure_terminal_path=case.terminal,
            normal_receipt_path=case.normal_receipt,
            runtime_spec_path=case.runtime_spec,
            runtime_launch_authorization_path=(
                case.runtime_launch_authorization
            ),
            runner=case.runner,
            manager_reader=_manager,
        )
    assert not case.recovery_authorization.exists()

    case.runtime_spec.unlink()
    decoy = case.evidence / "decoy-release-consumer.py"
    decoy.write_text("raise SystemExit(1)\n", encoding="utf-8")

    def rotate_source_path(call: int) -> None:
        if call == 2:
            monkeypatch.setattr(recovery, "RELEASE_CONSUMER_PATH", decoy)

    case.runner = SuccessExitStatusRecoveryRunner(
        original_runner.inner,
        on_live_shadow=rotate_source_path,
    )
    with pytest.raises(PermissionError, match="release_consumer path changed"):
        recovery.create_recovery_authorization(
            case.recovery_authorization,
            original_authorization_path=case.original_authorization,
            failure_terminal_path=case.terminal,
            normal_receipt_path=case.normal_receipt,
            runtime_spec_path=case.runtime_spec,
            runtime_launch_authorization_path=(
                case.runtime_launch_authorization
            ),
            runner=case.runner,
            manager_reader=_manager,
        )
    assert not case.recovery_authorization.exists()
