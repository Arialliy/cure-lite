from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cure_lite.data import PreprocessConfig
from tools import run_stage_a as cli


def _metrics(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "pd": 0.75,
        "miou": 0.5,
        "niou": 0.45,
        "pixel_fa": 0.001,
        "fp_components_per_mp": 2.0,
        "raw_background_fa": 0.002,
        "retention": 1.0,
        "budget_violation": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_parser_is_model_independent_and_has_no_d_t_option() -> None:
    parser = cli.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert {
        "--manifest",
        "--d-r-base-index",
        "--d-v-base-index",
        "--reference-base-run",
        "--config",
        "--output",
    } <= options
    forbidden = ("mshnet", "checkpoint", "provenance", "resume", "reuse")
    assert not any(
        fragment in option.lower()
        for option in options
        for fragment in forbidden
    )
    assert not any("d-t" in option or "d_t" in option for option in options)
    source = Path(cli.__file__).read_text(encoding="utf-8").lower()
    assert "mshnet" not in source


def test_main_consumes_only_generic_d_r_d_v_base_caches(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    config_payload = {"canonical": "stage-a"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_payload), encoding="utf-8")
    output = tmp_path / "stage-a"
    d_r_index = tmp_path / "cache" / "d_r" / "index.json"
    d_v_index = tmp_path / "cache" / "d_v" / "index.json"
    reference_base_run = tmp_path / "reference-base-run"
    expected_verified_identity = object()
    preprocess = PreprocessConfig(
        height=96,
        width=80,
        color_mode="L",
        mean=(0.25,),
        std=(0.5,),
    )
    manifest = SimpleNamespace(
        dataset="IRSTD-1K",
        fingerprint="1" * 64,
    )
    contract = SimpleNamespace(
        preprocessing=preprocess,
        d_r_index_path=d_r_index,
        d_v_index_path=d_v_index,
    )
    events: list[object] = []

    def fake_load_manifest(path: Path) -> object:
        events.append(("manifest", path))
        return manifest

    def fake_pair(received_d_r: Path, received_d_v: Path) -> object:
        events.append(("cache-contract", received_d_r, received_d_v))
        assert received_d_r == d_r_index
        assert received_d_v == d_v_index
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
        return SimpleNamespace(split=split, preprocess=received_preprocess)

    class FakeConfig:
        def canonical_payload(self) -> dict[str, object]:
            return config_payload

    class FakeStageARunConfig:
        @classmethod
        def from_mapping(cls, payload: dict[str, object]) -> FakeConfig:
            events.append("config")
            assert payload == config_payload
            return FakeConfig()

    def fake_load_verified_base_identity(path: Path) -> object:
        events.append(("base-identity", path))
        assert path == reference_base_run
        return expected_verified_identity

    def fake_run(
        received_d_r: Path,
        received_d_v: Path,
        d_r_dataset: object,
        d_v_dataset: object,
        config: object,
        output_dir: Path,
        *,
        verified_base_identity: object,
        calibration_workers: int,
        calibration_progress: object,
    ) -> object:
        events.append("run")
        assert received_d_r == d_r_index
        assert received_d_v == d_v_index
        assert d_r_dataset.split == "D_R"
        assert d_v_dataset.split == "D_V"
        assert config.canonical_payload() == config_payload
        assert output_dir == output
        assert verified_base_identity is expected_verified_identity
        assert calibration_workers == cli.DEFAULT_CALIBRATION_WORKERS
        assert calibration_progress is cli._calibration_progress
        return SimpleNamespace(
            results=SimpleNamespace(
                anchor=_metrics(pd=0.6),
                base_at_budget=_metrics(pd=0.65),
                factual_only=_metrics(pd=0.7),
                factual_exposure_matched=_metrics(pd=0.72),
                uniform_legal=_metrics(pd=0.75),
            ),
            support_summary=SimpleNamespace(
                canonical_payload=lambda: {
                    "source_images": 160,
                    "decoder_visible_legal_candidates": 25,
                }
            ),
            efficiency=SimpleNamespace(
                canonical_payload=lambda: {
                    "deployed_method": "U",
                    "efficiency_is_scientific_gate_metric": False,
                }
            ),
            complete_fingerprint="5" * 64,
        )

    monkeypatch.setattr(cli, "load_and_validate_manifest", fake_load_manifest)
    monkeypatch.setattr(cli, "load_base_cache_pair_contract", fake_pair)
    monkeypatch.setattr(cli, "ManifestImageDataset", fake_dataset)
    monkeypatch.setattr(cli, "StageARunConfig", FakeStageARunConfig)
    monkeypatch.setattr(
        cli,
        "load_verified_reference_base_run_identity",
        fake_load_verified_base_identity,
    )
    monkeypatch.setattr(cli, "run_stage_a_from_base_caches", fake_run)

    cli.main(
        [
            "--manifest",
            str(manifest_path),
            "--d-r-base-index",
            str(d_r_index),
            "--d-v-base-index",
            str(d_v_index),
            "--reference-base-run",
            str(reference_base_run),
            "--config",
            str(config_path),
            "--output",
            str(output),
        ]
    )

    assert events == [
        ("manifest", manifest_path),
        ("cache-contract", d_r_index, d_v_index),
        ("dataset", "D_R"),
        ("dataset", "D_V"),
        "config",
        ("base-identity", reference_base_run),
        "run",
    ]
    summary = json.loads(capsys.readouterr().out)
    assert summary["schema_version"] == cli.SUMMARY_SCHEMA
    assert summary["evaluation_split"] == "D_V"
    assert summary["independent_generalization_result"] is False
    assert summary["training_support"]["source_images"] == 160
    assert summary["efficiency"] == {
        "deployed_method": "U",
        "efficiency_is_scientific_gate_metric": False,
    }
    screen = summary["development_mechanism_screen"]
    assert screen["primary_metric"] == "total_pd"
    assert screen["u_minus_base_at_budget_pd"] == pytest.approx(0.1)
    assert screen["u_minus_factual_only_pd"] == pytest.approx(0.05)
    assert screen["u_minus_factual_exposure_matched_pd"] == pytest.approx(0.03)
    assert screen["mechanism_signal"] is True
    assert screen["not_an_independent_generalization_claim"] is True
    assert summary["method_order"] == ["A", "Base@B", "F", "F×", "U"]
    assert list(summary["methods"]) == ["A", "Base@B", "F", "F×", "U"]
    assert summary["methods"]["F"]["pd"] == 0.7
    assert summary["methods"]["F×"]["pd"] == 0.72
    assert set(summary["methods"]["U"]) == {
        "pd",
        "miou",
        "niou",
        "pixel_fa",
        "fp_components_per_mp",
        "raw_background_fa",
        "retention",
        "budget_violation",
    }
    assert not output.exists()


def test_existing_output_is_rejected_before_cache_access(tmp_path: Path) -> None:
    output = tmp_path / "stage-a"
    output.mkdir()
    try:
        cli.main(
            [
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--d-r-base-index",
                str(tmp_path / "d-r-index.json"),
                "--d-v-base-index",
                str(tmp_path / "d-v-index.json"),
                "--reference-base-run",
                str(tmp_path / "reference-base-run"),
                "--config",
                str(tmp_path / "config.json"),
                "--output",
                str(output),
            ]
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("an existing Stage-A output must be rejected")
