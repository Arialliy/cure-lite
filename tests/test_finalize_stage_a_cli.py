from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import finalize_stage_a as cli


def _minimal_incomplete_tree(root: Path) -> None:
    for directory in (
        root / "d_r",
        root / "d_v",
        root / "receipts",
        root / "decoders" / "factual_only",
        root / "decoders" / "factual_exposure_matched",
        root / "decoders" / "uniform_legal",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (root / ".incomplete").write_bytes(b"")
    for name in ("config", "anchor", "support"):
        (root / "receipts" / f"{name}.json").write_text(
            "{}\n",
            encoding="utf-8",
        )


def test_parser_is_post_training_only_and_has_no_official_test_input() -> None:
    options = {
        option
        for action in cli.build_parser()._actions
        for option in action.option_strings
    }
    assert {
        "--manifest",
        "--d-r-base-index",
        "--d-v-base-index",
        "--reference-base-run",
        "--config",
        "--decision-rule",
        "--protocol-freeze",
        "--stage-run",
        "--calibration-workers",
    }.issubset(options)
    assert not any("d-t" in option or "d_t" in option for option in options)
    assert not any("mshnet" in option.lower() for option in options)


def test_tool_source_contains_no_training_or_decoder_writer_call() -> None:
    source = Path(cli.__file__).read_text(encoding="utf-8")
    assert "run_paired_gate2_training" not in source
    assert "save_completed_decoder_run" not in source


def test_parallel_capacity_rejects_low_soft_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.resource, "getrlimit", lambda _: (1024, 1048576))
    with pytest.raises(RuntimeError, match="RLIMIT_NOFILE"):
        cli._require_execution_capacity(24)
    assert cli._require_execution_capacity(1) == (1024, 1048576)


def test_parallel_capacity_accepts_service_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli.resource, "getrlimit", lambda _: (65536, 1048576))
    assert cli._require_execution_capacity(24) == (65536, 1048576)


def test_stage_tree_guard_accepts_only_expected_layout(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    _minimal_incomplete_tree(root)
    assert cli._guard_stage_tree(root) == root / ".incomplete"

    (root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unexpected top-level"):
        cli._guard_stage_tree(root)


def test_stage_tree_guard_requires_empty_regular_marker(tmp_path: Path) -> None:
    root = tmp_path / "stage"
    _minimal_incomplete_tree(root)
    (root / ".incomplete").write_text("not-empty", encoding="utf-8")
    with pytest.raises(RuntimeError, match="must be empty"):
        cli._guard_stage_tree(root)


def test_stage_tree_guard_allows_complete_only_while_marker_remains(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage"
    _minimal_incomplete_tree(root)
    (root / "COMPLETE.json").write_text("{}\n", encoding="utf-8")
    assert cli._guard_stage_tree(root) == root / ".incomplete"
    (root / ".incomplete").unlink()
    with pytest.raises(RuntimeError, match="requires .incomplete"):
        cli._guard_stage_tree(root)


def test_scientific_input_fingerprint_ignores_only_completion_outputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stage"
    _minimal_incomplete_tree(root)
    (root / "d_r" / "input.bin").write_bytes(b"fixed-input")
    before = cli._scientific_input_inventory(root)

    for name in ("calibration", "results", "efficiency", "finalization"):
        (root / "receipts" / f"{name}.json").write_text(
            json.dumps({"name": name}),
            encoding="utf-8",
        )
    (root / "COMPLETE.json").write_text("{}\n", encoding="utf-8")
    assert cli._scientific_input_inventory(root) == before

    (root / "d_r" / "input.bin").write_bytes(b"changed-input")
    assert cli._scientific_input_inventory(root) != before


def test_write_or_require_same_is_create_only_and_exact(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    payload = {"schema_version": "test-v1", "value": 1}
    cli._write_or_require_same(path, payload)
    original = path.read_bytes()
    cli._write_or_require_same(path, payload)
    assert path.read_bytes() == original

    with pytest.raises(RuntimeError, match="differs"):
        cli._write_or_require_same(path, {**payload, "value": 2})
    assert path.read_bytes() == original


def test_finalization_payload_records_zero_new_optimizer_updates(
    tmp_path: Path,
) -> None:
    files = {}
    for name in ("manifest", "config", "rule", "freeze"):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        files[name] = path

    def artifact(variant: str) -> SimpleNamespace:
        return SimpleNamespace(
            config=SimpleNamespace(
                variant=variant,
                global_seed=42,
                trained_epochs=800,
                steps_per_epoch=40,
                initial_decoder_fingerprint="1" * 64,
            ),
            artifact_fingerprint="2" * 64,
            decoder_state_fingerprint="3" * 64,
            receipt_sha256="4" * 64,
            weights_sha256="5" * 64,
            train_log_sha256="6" * 64,
            train_log_fingerprint="7" * 64,
        )

    bundle = SimpleNamespace(
        base_index_fingerprint="8" * 64,
        base_index_sha256="9" * 64,
        state_index_fingerprint="a" * 64,
        state_index_sha256="b" * 64,
    )
    d_v_bundle = SimpleNamespace(
        base_index_fingerprint="c" * 64,
        base_index_sha256="d" * 64,
    )
    payload = cli._finalization_payload(
        root=tmp_path,
        manifest_path=files["manifest"],
        config_path=files["config"],
        decision_rule_path=files["rule"],
        freeze_path=files["freeze"],
        source_digest="e" * 64,
        config=SimpleNamespace(
            training=SimpleNamespace(
                global_seed=42,
                epochs=800,
                steps_per_epoch=40,
            )
        ),
        d_r_bundle=bundle,
        d_v_bundle=d_v_bundle,
        artifacts={
            "F": artifact("factual_only"),
            "F×": artifact("factual_exposure_matched"),
            "U": artifact("uniform_legal"),
        },
        verified_base_payload={"base_fingerprint": "f" * 64},
        calibration_workers=24,
        nofile_soft=65536,
        nofile_hard=1048576,
        scientific_input_fingerprint="0" * 64,
    )
    assert payload["training_entrypoint_called"] is False
    assert payload["optimizer_updates_during_this_operation"] == 0
    assert payload["trained_epochs"] == 800
    assert payload["decoders"]["F"]["optimizer_updates"] == 32000
    assert payload["decoders"]["F×"]["optimizer_updates"] == 32000
    assert payload["decoders"]["U"]["optimizer_updates"] == 32000
    assert payload["calibration_workers"] == 24
    assert payload["execution_file_limit"]["soft"] == 65536
