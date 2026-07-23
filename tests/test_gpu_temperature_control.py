from __future__ import annotations

import sys

import pytest

from tools import run_with_gpu_temperature_control as control


@pytest.mark.parametrize(
    ("paused", "temperature", "expected"),
    [
        (False, 81, control.ACTION_HOLD),
        (False, 82, control.ACTION_PAUSE),
        (False, 90, control.ACTION_PAUSE),
        (True, 76, control.ACTION_HOLD),
        (True, 75, control.ACTION_CONTINUE),
        (True, 60, control.ACTION_CONTINUE),
        (False, None, control.ACTION_PAUSE),
        (True, None, control.ACTION_HOLD),
    ],
)
def test_temperature_action_has_hysteresis(
    paused: bool,
    temperature: int | None,
    expected: str,
) -> None:
    assert (
        control.temperature_action(
            paused=paused,
            temperature=temperature,
            pause_temp=82,
            resume_temp=75,
        )
        == expected
    )


def test_temperature_control_cli_keeps_command_after_separator() -> None:
    args = control.parse_args(
        [
            "--gpu",
            "0",
            "--pause-temp",
            "82",
            "--resume-temp",
            "75",
            "--",
            "python",
            "worker.py",
            "--device",
            "cuda:0",
        ]
    )
    assert args.gpu == 0
    assert args.pause_temp == 82
    assert args.resume_temp == 75
    assert args.command == ["python", "worker.py", "--device", "cuda:0"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--gpu", "0", "--pause-temp", "75", "--resume-temp", "75", "--", "x"],
        ["--gpu", "0", "--pause-temp", "70", "--resume-temp", "75", "--", "x"],
        ["--gpu", "0", "--pause-temp", "151", "--resume-temp", "75", "--", "x"],
        ["--gpu", "0", "--pause-temp", "82", "--resume-temp", "-1", "--", "x"],
        ["--gpu", "0", "--"],
    ],
)
def test_temperature_control_cli_rejects_invalid_contract(
    argv: list[str],
) -> None:
    with pytest.raises(ValueError):
        control.parse_args(argv)


def test_controlled_command_preserves_nonzero_exit_code() -> None:
    assert (
        control.run_controlled(
            [sys.executable, "-c", "raise SystemExit(42)"],
            gpu_index=0,
            pause_temp=82,
            resume_temp=75,
            poll_seconds=0.001,
            log_every_polls=0,
            temperature_reader=lambda _gpu: 70,
        )
        == 42
    )


def test_signal_return_code_is_normalized() -> None:
    assert control._normalized_return_code(-15) == 143
