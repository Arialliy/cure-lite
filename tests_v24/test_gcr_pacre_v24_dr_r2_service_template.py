from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "deploy/systemd/"
    "cure-lite-v24-gcr-pacre-dr-r2.service.template"
)


def _directives(text: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "[")):
            continue
        key, separator, value = line.partition("=")
        assert separator == "="
        result.setdefault(key, []).append(value)
    return result


def test_r2_service_template_is_static_non_installable_and_no_restart() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    directives = _directives(text)

    assert "[Unit]" in text
    assert "[Service]" in text
    assert "[Install]" not in text
    assert "WantedBy=" not in text
    assert "RequiredBy=" not in text
    assert directives["Type"] == ["exec"]
    assert directives["ExitType"] == ["main"]
    assert directives["UMask"] == ["0077"]
    assert directives["Restart"] == ["no"]
    assert directives["RestartSec"] == ["0"]
    assert directives["StartLimitIntervalSec"] == ["infinity"]
    assert directives["StartLimitBurst"] == ["1"]
    assert directives["KillMode"] == ["mixed"]
    assert directives["KillSignal"] == ["SIGTERM"]
    assert directives["SendSIGKILL"] == ["yes"]
    assert directives["WatchdogSec"] == ["0"]
    assert directives["RemainAfterExit"] == ["no"]
    assert directives["StandardInput"] == ["null"]
    assert directives["TimeoutStartSec"] == ["5min"]
    assert directives["OOMPolicy"] == ["kill"]
    assert directives["SuccessExitStatus"] == ["0"]


def test_r2_service_template_invokes_only_absolute_python_supervisor_argv() -> None:
    directives = _directives(TEMPLATE.read_text(encoding="utf-8"))
    expected_modes = {
        "ExecCondition": "claim-materialization",
        "ExecStartPre": "verify-runtime-spec",
        "ExecStart": "run-once",
        "ExecStopPost": "record-systemd-exit",
    }
    for directive, mode in expected_modes.items():
        values = directives[directive]
        assert len(values) == 1
        argv = values[0].split()
        assert argv[0] == "/home/md0/ly/MSHNet/.venv/bin/python"
        assert argv[1:3] == ["-I", "-u"]
        assert argv[3] == (
            "/home/md0/ly/cure_lite/"
            "tools/cure_lite_v24_runtime_supervisor.py"
        )
        assert argv[4] == mode
        assert argv[5] == "--spec"
        assert argv[6] == (
            "/home/md0/ly/cure_lite/protocols/IRSTD-1K/"
            "gcr_pacre_v24/"
            "D_R_structural_attempt_r2_runtime_spec.json"
        )
        assert len(argv) == 7
        assert all(
            forbidden not in values[0]
            for forbidden in (
                "/bin/bash",
                "/bin/sh",
                "systemd-run",
                "Restart=on-failure",
                "run_cure_lite_v24_gcr_pacre_dr_gate.py",
            )
        )
