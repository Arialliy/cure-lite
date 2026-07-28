from __future__ import annotations

import json

import pytest

from tools import run_coverage_state_paet_formal_d_v_evaluation_v3 as cli


def test_cli_exposes_only_fixed_modes() -> None:
    parser = cli.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert options == {
        "-h",
        "--help",
        "--validate-create-only",
        "--run-once",
        "--finalize-evidence-binding",
    }
    for forbidden in (
        "--path",
        "--output",
        "--seed",
        "--threshold",
        "--field-threshold",
        "--batch-size",
        "--split",
        "--D_T",
        "--resume",
        "--retry",
    ):
        with pytest.raises(SystemExit):
            cli.parse_args(["--run-once", forbidden, "x"])


def test_cli_requires_exactly_one_mode() -> None:
    with pytest.raises(SystemExit):
        cli.parse_args([])
    with pytest.raises(SystemExit):
        cli.parse_args(["--validate-create-only", "--run-once"])


def test_validate_mode_never_calls_run_or_finalize(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {
        "create_only": True,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    monkeypatch.setattr(
        cli,
        "validate_paet_formal_d_v_create_only_v3",
        lambda: expected,
    )

    def forbidden() -> dict[str, object]:
        raise AssertionError("validate mode executed or finalized D_V")

    monkeypatch.setattr(cli, "run_paet_formal_d_v_once_v3", forbidden)
    monkeypatch.setattr(
        cli,
        "finalize_paet_formal_d_v_evidence_binding_v3",
        forbidden,
    )
    cli.main(["--validate-create-only"])
    assert json.loads(capsys.readouterr().out) == expected


def test_run_mode_calls_fixed_no_argument_entry(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {"status": "complete", "D_T_accessed": False}
    monkeypatch.setattr(
        cli,
        "run_paet_formal_d_v_once_v3",
        lambda: expected,
    )
    cli.main(["--run-once"])
    assert json.loads(capsys.readouterr().out) == expected
