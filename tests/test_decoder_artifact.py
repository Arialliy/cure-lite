from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from cure_lite.config import (
    DecoderConfig,
    InterventionConfig,
    MatchConfig,
    OccupancyConfig,
)
from cure_lite.decoder import CURELiteDecoder
from cure_lite.experiment.artifacts import (
    DecoderRunConfig,
    _save_decoder_artifact,
    decoder_state_fingerprint,
    load_decoder_artifact,
)


def _config() -> DecoderRunConfig:
    return DecoderRunConfig(
        variant="uniform_legal",
        manifest_fingerprint="1" * 64,
        manifest_file_sha256="2" * 64,
        preprocessing_fingerprint="b" * 64,
        base_fingerprint="3" * 64,
        state_fingerprint="4" * 64,
        gt_fingerprint="5" * 64,
        base_index_fingerprint="6" * 64,
        base_index_sha256="7" * 64,
        state_index_fingerprint="8" * 64,
        state_index_sha256="9" * 64,
        initial_decoder_fingerprint="a" * 64,
        occupancy_config=OccupancyConfig(),
        match_config=MatchConfig(),
        intervention_config=InterventionConfig(),
        global_seed=7,
        trained_epochs=2,
        steps_per_epoch=3,
        decoder_config=DecoderConfig(feature_channels=3),
    )


def _logs() -> tuple[dict[str, object], ...]:
    return (
        {
            "epoch": 0,
            "pool_sizes": {
                "factual_miss": 2,
                "factual_no_miss": 1,
                "synthetic": 2,
            },
            "metrics": {"steps": 3, "total": 1.0},
        },
        {
            "epoch": 1,
            "pool_sizes": {
                "factual_miss": 2,
                "factual_no_miss": 1,
                "synthetic": 2,
            },
            "metrics": {"steps": 3, "total": 0.5},
        },
    )


def test_decoder_artifact_round_trip_binds_weights_config_and_logs(tmp_path) -> None:
    torch.manual_seed(5)
    decoder = CURELiteDecoder(feature_channels=3)
    before = {name: tensor.detach().clone() for name, tensor in decoder.state_dict().items()}
    directory = tmp_path / "decoder"
    fingerprint = _save_decoder_artifact(
        directory,
        decoder,
        _config(),
        _logs(),
    )
    loaded = load_decoder_artifact(directory, expected_config=_config())
    assert loaded.artifact_fingerprint == fingerprint
    assert loaded.config == _config()
    assert tuple(log["epoch"] for log in loaded.epoch_logs) == (0, 1)
    assert all(
        torch.equal(value, loaded.decoder.state_dict()[name])
        for name, value in before.items()
    )
    assert not loaded.decoder.training
    assert all(not parameter.requires_grad for parameter in loaded.decoder.parameters())
    loaded.verify_unchanged()

    with torch.no_grad():
        next(loaded.decoder.parameters()).add_(1.0)
    with pytest.raises(RuntimeError, match="loaded decoder changed"):
        loaded.verify_unchanged()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _save_decoder_artifact(
            directory,
            decoder,
            _config(),
            _logs(),
        )


def test_decoder_artifact_rejects_tampered_weights_and_receipt(tmp_path) -> None:
    decoder = CURELiteDecoder(feature_channels=3)
    weights_directory = tmp_path / "weights"
    _save_decoder_artifact(
        weights_directory,
        decoder,
        _config(),
        _logs(),
    )
    with (weights_directory / "decoder.safetensors").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="weights SHA256 mismatch"):
        load_decoder_artifact(weights_directory)

    receipt_directory = tmp_path / "receipt"
    _save_decoder_artifact(
        receipt_directory,
        decoder,
        _config(),
        _logs(),
    )
    receipt_path = receipt_directory / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["run_config"]["global_seed"] = 99
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact fingerprint mismatch"):
        load_decoder_artifact(receipt_directory)


def test_loaded_decoder_artifact_rechecks_persisted_files(tmp_path) -> None:
    directory = tmp_path / "decoder"
    _save_decoder_artifact(
        directory,
        CURELiteDecoder(feature_channels=3),
        _config(),
        _logs(),
    )
    loaded = load_decoder_artifact(directory)
    log_path = directory / "train_log.json"
    log_path.write_bytes(log_path.read_bytes() + b" ")

    with pytest.raises(RuntimeError, match="training log changed on disk"):
        loaded.verify_unchanged()


def test_decoder_artifact_loader_rejects_symlink_entrypoint(tmp_path) -> None:
    directory = tmp_path / "decoder"
    _save_decoder_artifact(
        directory,
        CURELiteDecoder(feature_channels=3),
        _config(),
        _logs(),
    )
    link = tmp_path / "decoder-link"
    link.symlink_to(directory, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        load_decoder_artifact(link)


def test_loaded_artifact_rejects_replace_replay_and_topology_mutation(tmp_path) -> None:
    directory = tmp_path / "decoder"
    _save_decoder_artifact(
        directory,
        CURELiteDecoder(feature_channels=3),
        _config(),
        _logs(),
    )
    loaded = load_decoder_artifact(directory)

    with pytest.raises(TypeError, match="source objects were replaced"):
        replace(
            loaded,
            config=replace(loaded.config, variant="factual_only"),
        )

    replacement_decoder = CURELiteDecoder(feature_channels=3)
    replacement_decoder.requires_grad_(False).eval()
    with pytest.raises(TypeError, match="source objects were replaced"):
        replace(
            loaded,
            decoder=replacement_decoder,
            decoder_state_fingerprint=decoder_state_fingerprint(
                replacement_decoder
            ),
        )

    loaded.decoder.project[2] = torch.nn.ReLU()
    with pytest.raises(RuntimeError, match="module topology changed"):
        loaded.verify_unchanged()


def test_decoder_artifact_rejects_mismatched_contracts(tmp_path) -> None:
    with pytest.raises(ValueError, match="variant"):
        DecoderRunConfig(
            variant="score_hard",
            manifest_fingerprint="1" * 64,
            manifest_file_sha256="2" * 64,
            preprocessing_fingerprint="b" * 64,
            base_fingerprint="3" * 64,
            state_fingerprint="4" * 64,
            gt_fingerprint="5" * 64,
            base_index_fingerprint="6" * 64,
            base_index_sha256="7" * 64,
            state_index_fingerprint="8" * 64,
            state_index_sha256="9" * 64,
            initial_decoder_fingerprint="a" * 64,
            occupancy_config=OccupancyConfig(),
            match_config=MatchConfig(),
            intervention_config=InterventionConfig(),
            global_seed=7,
            trained_epochs=1,
            steps_per_epoch=1,
            decoder_config=DecoderConfig(feature_channels=3),
        )

    decoder = CURELiteDecoder(feature_channels=4)
    with pytest.raises(ValueError, match="channels"):
        _save_decoder_artifact(
            tmp_path / "bad",
            decoder,
            _config(),
            _logs(),
        )


def test_decoder_artifact_rejects_logs_that_disagree_with_fixed_horizon(
    tmp_path,
) -> None:
    decoder = CURELiteDecoder(feature_channels=3)
    logs = list(_logs())
    logs[1] = dict(logs[1])
    logs[1]["metrics"] = {"steps": 2, "total": 0.5}
    with pytest.raises(ValueError, match="steps_per_epoch"):
        _save_decoder_artifact(tmp_path / "bad-steps", decoder, _config(), logs)
