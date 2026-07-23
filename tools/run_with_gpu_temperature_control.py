#!/usr/bin/env python3
"""Run one command with hysteretic GPU temperature pause/continue control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import signal
import subprocess
import sys
import time
from typing import Callable, Sequence


ACTION_HOLD = "hold"
ACTION_PAUSE = "pause"
ACTION_CONTINUE = "continue"


def _nonnegative_integer(value: str) -> int:
    try:
        resolved = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if resolved < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return resolved


def _positive_float(value: str) -> float:
    try:
        resolved = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not resolved > 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=_nonnegative_integer, required=True)
    parser.add_argument("--pause-temp", type=int, default=82)
    parser.add_argument("--resume-temp", type=int, default=75)
    parser.add_argument("--poll-seconds", type=_positive_float, default=1.0)
    parser.add_argument(
        "--log-every-polls",
        type=_nonnegative_integer,
        default=60,
        help="emit a temperature sample every N polls; zero disables samples",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command to run, conventionally preceded by --",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    _validate_temperature_contract(args.pause_temp, args.resume_temp)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("a command is required after --")
    args.command = command
    return args


def temperature_action(
    *,
    paused: bool,
    temperature: int | None,
    pause_temp: int,
    resume_temp: int,
) -> str:
    """Return the next state transition under a hysteretic policy.

    A missing reading is handled conservatively: active work pauses, while
    already-paused work remains paused until a valid cool reading is observed.
    """

    _validate_temperature_contract(pause_temp, resume_temp)
    if temperature is None:
        return ACTION_HOLD if paused else ACTION_PAUSE
    if paused:
        return ACTION_CONTINUE if temperature <= resume_temp else ACTION_HOLD
    return ACTION_PAUSE if temperature >= pause_temp else ACTION_HOLD


def _validate_temperature_contract(pause_temp: int, resume_temp: int) -> None:
    if not 0 <= resume_temp < pause_temp <= 150:
        raise ValueError(
            "temperature contract must satisfy "
            "0 <= resume_temp < pause_temp <= 150"
        )


def _read_gpu_temperature(gpu_index: int) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_index),
            "--query-gpu=temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=2.0,
    )
    readings = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(readings) != 1:
        raise RuntimeError("nvidia-smi returned an unexpected temperature payload")
    try:
        temperature = int(readings[0])
    except ValueError as error:
        raise RuntimeError("nvidia-smi temperature is not an integer") from error
    if temperature < 0 or temperature > 150:
        raise RuntimeError("nvidia-smi temperature is outside the valid range")
    return temperature


def _event(name: str, **fields: object) -> None:
    payload = {
        "event": name,
        "time_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **fields,
    }
    print(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False),
        file=sys.stderr,
        flush=True,
    )


def _signal_process_group(process: subprocess.Popen[bytes], value: signal.Signals) -> bool:
    try:
        os.killpg(process.pid, value)
    except ProcessLookupError:
        return False
    return True


def _normalized_return_code(return_code: int) -> int:
    return return_code if return_code >= 0 else 128 + (-return_code)


def run_controlled(
    command: Sequence[str],
    *,
    gpu_index: int,
    pause_temp: int,
    resume_temp: int,
    poll_seconds: float,
    log_every_polls: int,
    temperature_reader: Callable[[int], int] = _read_gpu_temperature,
) -> int:
    if not command:
        raise ValueError("command must not be empty")
    _validate_temperature_contract(pause_temp, resume_temp)
    if poll_seconds <= 0.0:
        raise ValueError("poll_seconds must be positive")
    if log_every_polls < 0:
        raise ValueError("log_every_polls must be nonnegative")

    waiting_for_cool_temperature = False
    while True:
        try:
            initial_temperature = temperature_reader(gpu_index)
        except Exception as error:
            waiting_for_cool_temperature = True
            _event(
                "gpu_temperature_probe_failed_before_launch",
                gpu_index=gpu_index,
                error_type=type(error).__name__,
            )
            time.sleep(poll_seconds)
            continue
        if waiting_for_cool_temperature:
            if initial_temperature <= resume_temp:
                break
        elif initial_temperature < pause_temp:
            break
        else:
            waiting_for_cool_temperature = True
        _event(
            "gpu_temperature_wait_before_launch",
            gpu_index=gpu_index,
            temperature_c=initial_temperature,
            pause_temp_c=pause_temp,
            resume_temp_c=resume_temp,
        )
        time.sleep(poll_seconds)

    _event(
        "gpu_temperature_control_start",
        gpu_index=gpu_index,
        initial_temperature_c=initial_temperature,
        pause_temp_c=pause_temp,
        resume_temp_c=resume_temp,
        poll_seconds=poll_seconds,
        executable=command[0],
        argument_count=len(command) - 1,
    )
    child_environment = os.environ.copy()
    child_environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    process = subprocess.Popen(
        list(command),
        start_new_session=True,
        env=child_environment,
    )
    paused = False
    polls = 0
    requested_signal: list[int | None] = [None]
    termination_deadline: float | None = None

    def request_stop(signum: int, _frame: object) -> None:
        requested_signal[0] = signum

    previous_handlers = {
        value: signal.getsignal(value) for value in (signal.SIGINT, signal.SIGTERM)
    }
    for value in previous_handlers:
        signal.signal(value, request_stop)

    try:
        while process.poll() is None:
            if requested_signal[0] is not None:
                forwarded = signal.Signals(requested_signal[0])
                if termination_deadline is not None:
                    _signal_process_group(process, signal.SIGKILL)
                    requested_signal[0] = None
                    continue
                if paused:
                    _signal_process_group(process, signal.SIGCONT)
                    paused = False
                _signal_process_group(process, forwarded)
                _event(
                    "gpu_temperature_control_forward_signal",
                    gpu_index=gpu_index,
                    signal=int(forwarded),
                )
                requested_signal[0] = None
                termination_deadline = time.monotonic() + 30.0

            if termination_deadline is not None:
                if time.monotonic() >= termination_deadline:
                    _signal_process_group(process, signal.SIGKILL)
                    _event(
                        "gpu_temperature_control_force_stop",
                        gpu_index=gpu_index,
                    )
                time.sleep(min(poll_seconds, 0.25))
                continue

            temperature: int | None
            probe_error: Exception | None = None
            try:
                temperature = temperature_reader(gpu_index)
            except Exception as error:
                temperature = None
                probe_error = error

            action = temperature_action(
                paused=paused,
                temperature=temperature,
                pause_temp=pause_temp,
                resume_temp=resume_temp,
            )
            if action == ACTION_PAUSE:
                if _signal_process_group(process, signal.SIGSTOP):
                    paused = True
                    _event(
                        "gpu_work_paused",
                        gpu_index=gpu_index,
                        temperature_c=temperature,
                        reason=(
                            "temperature_probe_failed"
                            if temperature is None
                            else "pause_temperature_reached"
                        ),
                        error_type=(
                            type(probe_error).__name__
                            if probe_error is not None
                            else None
                        ),
                    )
            elif action == ACTION_CONTINUE:
                if _signal_process_group(process, signal.SIGCONT):
                    paused = False
                    _event(
                        "gpu_work_continued",
                        gpu_index=gpu_index,
                        temperature_c=temperature,
                        reason="resume_temperature_reached",
                    )

            polls += 1
            if (
                log_every_polls > 0
                and polls % log_every_polls == 0
            ):
                _event(
                    "gpu_temperature_sample",
                    gpu_index=gpu_index,
                    temperature_c=temperature,
                    paused=paused,
                    probe_ok=temperature is not None,
                )
            time.sleep(poll_seconds)
    finally:
        for value, previous in previous_handlers.items():
            signal.signal(value, previous)
        if paused:
            _signal_process_group(process, signal.SIGCONT)

    return_code = process.wait()
    normalized_return_code = _normalized_return_code(return_code)
    _event(
        "gpu_temperature_control_end",
        gpu_index=gpu_index,
        raw_return_code=return_code,
        return_code=normalized_return_code,
    )
    return normalized_return_code


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    raise SystemExit(
        run_controlled(
            args.command,
            gpu_index=args.gpu,
            pause_temp=args.pause_temp,
            resume_temp=args.resume_temp,
            poll_seconds=args.poll_seconds,
            log_every_polls=args.log_every_polls,
        )
    )


if __name__ == "__main__":
    main()
