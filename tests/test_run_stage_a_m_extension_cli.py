from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cure_lite.data import PreprocessConfig
from cure_lite.experiment.training_pipeline import FixedEpochTrainingLog
from tools import run_stage_a_m_extension as cli


def _digest(character: str) -> str:
    return character * 64


def _metrics(pd: float) -> dict[str, object]:
    return {
        "pd": pd,
        "miou": 0.5,
        "niou": 0.45,
        "pixel_fa": 0.001,
        "fp_components_per_mp": 2.0,
        "raw_background_fa": 0.002,
        "retention": 1.0,
        "budget_violation": False,
    }


def _recovery(recovered: int) -> dict[str, object]:
    return {
        "rmr": recovered / 10,
        "gross_rmr": recovered / 10,
        "net_rmr": recovered / 10,
        "reachable_rmr": recovered / 8,
        "oracle_upper_bound": 0.8,
        "overlap_supported_rmr": recovered / 8,
        "recovered_anchor_misses": recovered,
        "net_recovered_anchor_misses": recovered,
        "total_anchor_misses": 10,
        "retained_anchor_covered": 20,
        "total_anchor_covered": 20,
        "recovered_reachable_anchor_misses": recovered,
        "total_reachable_anchor_misses": 8,
    }


def test_parser_has_only_create_only_m_extension_inputs() -> None:
    parser = cli.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert {
        "--reference-stage-a",
        "--manifest",
        "--output",
        "--device",
        "--calibration-workers",
    } <= options
    forbidden = ("resume", "recovery", "restore", "reuse", "checkpoint")
    assert not any(
        fragment in option.lower()
        for option in options
        for fragment in forbidden
    )
    assert not any("d-t" in option or "d_t" in option for option in options)
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--reference-stage-a",
                "old",
                "--manifest",
                "manifest.json",
                "--output",
                "new",
                "--device",
                "cuda:0",
                "--resume",
            ]
        )


def test_reference_training_horizon_is_fixed_to_800(tmp_path: Path) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    config = receipts / "config.json"
    config.write_text(
        json.dumps({"run_config": {"training": {"epochs": 800}}}),
        encoding="utf-8",
    )
    cli._require_frozen_training_horizon(tmp_path)
    config.write_text(
        json.dumps({"run_config": {"training": {"epochs": 799}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="800-epoch"):
        cli._require_frozen_training_horizon(tmp_path)


def test_progress_callbacks_are_bounded_json(capsys) -> None:
    def log(epoch: int) -> FixedEpochTrainingLog:
        return FixedEpochTrainingLog(
            epoch=epoch,
            pool_sizes=(
                ("factual_miss", 32),
                ("factual_no_miss", 128),
                ("synthetic", 32),
            ),
            metrics=(("loss", 0.25),),
        )

    for epoch in (0, 1, 8, 9, 10, 798, 799):
        cli._training_progress(log(epoch))
    for done in range(1, 42):
        cli._calibration_progress(done, 41)

    events = [
        json.loads(line)
        for line in capsys.readouterr().err.strip().splitlines()
    ]
    training = [
        event
        for event in events
        if event["event"] == "m_training_epoch_progress"
    ]
    assert [event["epoch"] for event in training] == [1, 10, 800]
    assert all(event["epochs"] == 800 for event in training)
    assert training[0]["pool_sizes"]["synthetic"] == 32
    calibration = [
        event
        for event in events
        if event["event"] == "calibration_candidate_progress"
    ]
    assert calibration[0] == {
        "event": "calibration_candidate_progress",
        "completed": 1,
        "total": 41,
    }
    assert calibration[-1]["completed"] == 41
    assert len(calibration) <= 22


def test_main_uses_historical_cache_contract_and_prints_full_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    reference = tmp_path / "reference-stage-a"
    (reference / "receipts").mkdir(parents=True)
    (reference / "receipts" / "config.json").write_text(
        json.dumps({"run_config": {"training": {"epochs": 800}}}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "m-extension"
    preprocess = PreprocessConfig(
        height=96,
        width=80,
        color_mode="L",
        mean=(0.25,),
        std=(0.5,),
    )
    manifest = SimpleNamespace(
        dataset="IRSTD-1K",
        fingerprint=_digest("a"),
    )
    contract = SimpleNamespace(preprocessing=preprocess)
    events: list[object] = []

    def fake_manifest(path: Path) -> object:
        events.append(("manifest", path))
        return manifest

    def fake_contract(d_r: Path, d_v: Path) -> object:
        events.append(("contract", d_r, d_v))
        assert d_r == reference / "d_r" / "base_cache" / "index.json"
        assert d_v == reference / "d_v" / "base_cache" / "index.json"
        return contract

    def fake_dataset(
        received_manifest: object,
        split: str,
        received_preprocess: object,
        *,
        manifest_path: Path,
    ) -> object:
        events.append(("dataset", split))
        assert received_manifest is manifest
        assert received_preprocess is preprocess
        assert manifest_path == tmp_path / "manifest.json"
        return SimpleNamespace(split=split)

    fingerprints = {
        "source_tree_digest": _digest("b"),
        "reference_results_fingerprint": _digest("c"),
        "reference_calibration_receipt_fingerprint": _digest("d"),
        "config_fingerprint": _digest("e"),
        "reference_fingerprint": _digest("f"),
        "alignment_receipt_fingerprint": _digest("1"),
        "m_calibration_receipt_fingerprint": _digest("2"),
        "manifest_fingerprint": _digest("3"),
        "preprocessing_fingerprint": _digest("4"),
        "base_fingerprint": _digest("5"),
        "base_state_fingerprint": _digest("6"),
    }
    result_fingerprint = _digest("7")
    complete_fingerprint = _digest("8")
    gate_fingerprint = _digest("9")
    methods = {
        method: _metrics(0.60 + index * 0.02)
        for index, method in enumerate(cli.METHOD_ORDER)
    }
    recovery = {
        "U@historical": _recovery(4),
        "M": _recovery(6),
    }

    def fake_run(
        received_reference: Path,
        d_r_dataset: object,
        d_v_dataset: object,
        received_output: Path,
        *,
        device: str,
        calibration_workers: int,
        calibration_progress: object,
        training_progress: object,
    ) -> object:
        events.append("run")
        assert received_reference == reference
        assert d_r_dataset.split == "D_R"
        assert d_v_dataset.split == "D_V"
        assert received_output == output
        assert device == "cuda:2"
        assert calibration_workers == 3
        assert calibration_progress is cli._calibration_progress
        assert training_progress is cli._training_progress
        received_output.mkdir()
        receipts = received_output / "receipts"
        receipts.mkdir()
        (receipts / "results.json").write_text(
            json.dumps(
                {
                    "method_order": list(cli.METHOD_ORDER),
                    "methods": methods,
                    "reference_results_fingerprint": _digest("c"),
                    "recovery_diagnostics": recovery,
                    "results_fingerprint": result_fingerprint,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (receipts / "gate.json").write_text(
            json.dumps(
                {
                    "results_fingerprint": result_fingerprint,
                    "mechanism_signal": True,
                    "gate_fingerprint": gate_fingerprint,
                }
            ),
            encoding="utf-8",
        )
        (received_output / "COMPLETE.json").write_text(
            json.dumps(
                {
                    **fingerprints,
                    "results_fingerprint": result_fingerprint,
                    "complete_fingerprint": complete_fingerprint,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            root=received_output.resolve(),
            reference=SimpleNamespace(
                snapshot_fingerprint=_digest("0"),
                complete_fingerprint=_digest("a"),
                source_tree_digest=_digest("b"),
            ),
            m_artifact=SimpleNamespace(
                artifact_fingerprint=_digest("c"),
                decoder_state_fingerprint=_digest("d"),
            ),
            m_calibration=SimpleNamespace(
                receipt_fingerprint=_digest("e"),
            ),
            alignment_catalog_fingerprint=_digest("f"),
            results_fingerprint=result_fingerprint,
            complete_fingerprint=complete_fingerprint,
            mechanism_signal=True,
        )

    monkeypatch.setattr(cli, "load_and_validate_manifest", fake_manifest)
    monkeypatch.setattr(cli, "load_base_cache_pair_contract", fake_contract)
    monkeypatch.setattr(cli, "ManifestImageDataset", fake_dataset)
    monkeypatch.setattr(cli, "run_stage_a_m_extension", fake_run)

    cli.main(
        [
            "--reference-stage-a",
            str(reference),
            "--manifest",
            str(manifest_path),
            "--output",
            str(output),
            "--device",
            "cuda:2",
            "--calibration-workers",
            "3",
        ]
    )

    assert events == [
        ("manifest", manifest_path),
        (
            "contract",
            reference / "d_r" / "base_cache" / "index.json",
            reference / "d_v" / "base_cache" / "index.json",
        ),
        ("dataset", "D_R"),
        ("dataset", "D_V"),
        "run",
    ]
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema_version"] == cli.SUMMARY_SCHEMA
    assert summary["evaluation_split"] == "D_V"
    assert summary["independent_generalization_result"] is False
    assert summary["method_order"] == ["A", "Base@B", "F", "F×", "U", "M"]
    assert list(summary["methods"]) == summary["method_order"]
    assert summary["methods"]["M"]["pd"] == pytest.approx(0.70)
    assert summary["recovery_diagnostics"]["U@historical"][
        "recovered_anchor_misses"
    ] == 4
    assert summary["recovery_diagnostics"]["M"][
        "recovered_anchor_misses"
    ] == 6
    assert summary["development_gate"]["mechanism_signal"] is True
    assert summary["fingerprints"]["complete"] == complete_fingerprint
    assert summary["fingerprints"]["results"] == result_fingerprint
    assert summary["fingerprints"]["gate"] == gate_fingerprint
