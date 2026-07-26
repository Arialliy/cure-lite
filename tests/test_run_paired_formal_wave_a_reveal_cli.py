from __future__ import annotations

import json

import pytest

from tools import run_paired_formal_wave_a_reveal as cli


def test_cli_exposes_only_required_config() -> None:
    parser = cli.build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert option_strings == {"-h", "--help", "--config"}
    for forbidden in (
        "--output",
        "--seed",
        "--method",
        "--D_V",
        "--D_T",
        "--resume",
        "--overwrite",
        "--threshold",
    ):
        assert forbidden not in option_strings
    with pytest.raises(SystemExit):
        cli.parse_args([])


def test_cli_prints_only_final_status_after_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = {
        "status": "FORMAL_WAVE_PASS",
        "output": "/tmp/final",
        "decision_fingerprint": "a" * 64,
        "complete_fingerprint": "b" * 64,
    }
    monkeypatch.setattr(
        cli,
        "run_wave_a_reveal",
        lambda path: type(
            "Published",
            (),
            {"success_summary": lambda self: summary},
        )(),
    )
    cli.main(["--config", "synthetic.json"])
    assert json.loads(capsys.readouterr().out) == summary


def test_cli_failure_prints_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(path):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(cli, "run_wave_a_reveal", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        cli.main(["--config", "synthetic.json"])
    assert capsys.readouterr().out == ""
