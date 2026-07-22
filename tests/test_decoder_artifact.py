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


def _epoch_metrics(
    total: float,
    *,
    synthetic_active: float = 1.0,
    synthetic_states: float = 4.0,
) -> dict[str, float | int]:
    result: dict[str, float | int] = {"steps": 3, "total": total}
    for branch, active, states in (
        ("factual_miss", 1.0, 4.0),
        ("factual_no_miss", 1.0, 4.0),
        ("synthetic", synthetic_active, synthetic_states),
    ):
        result[f"{branch}/active"] = active
        result[f"{branch}/active_min"] = active
        result[f"{branch}/active_max"] = active
        result[f"{branch}/states"] = states
        result[f"{branch}/states_min"] = states
        result[f"{branch}/states_max"] = states
    return result


def _logs() -> tuple[dict[str, object], ...]:

    return (
        {
            "epoch": 0,
            "pool_sizes": {
                "factual_miss": 2,
                "factual_no_miss": 1,
                "synthetic": 2,
            },
            "metrics": _epoch_metrics(1.0),
        },
        {
            "epoch": 1,
            "pool_sizes": {
                "factual_miss": 2,
                "factual_no_miss": 1,
                "synthetic": 2,
            },
            "metrics": _epoch_metrics(0.5),
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


def test_exposure_matched_artifact_binds_factual_replacement_loss_slot(
    tmp_path,
) -> None:
    config = replace(
        _config(),
        variant="factual_exposure_matched",
        trained_epochs=1,
        steps_per_epoch=3,
        synthetic_batch=4,
    )
    logs = (
        {
            "epoch": 0,
            "pool_sizes": {
                "factual_miss": 2,
                "factual_no_miss": 1,
                "synthetic": 0,
            },
            "metrics": _epoch_metrics(1.0),
        },
    )
    directory = tmp_path / "fx"
    _save_decoder_artifact(
        directory,
        CURELiteDecoder(feature_channels=3),
        config,
        logs,
    )
    loaded = load_decoder_artifact(directory, expected_config=config)
    assert loaded.config.variant_contract["matched_reference_variant"] == (
        "uniform_legal"
    )
    assert loaded.config.variant_contract["deletion_intervention_used"] is False

    bad_logs = [dict(logs[0])]
    bad_logs[0]["metrics"] = dict(logs[0]["metrics"])
    bad_logs[0]["metrics"]["synthetic/states"] = 3.0
    with pytest.raises(ValueError, match="synthetic_batch exposure"):
        _save_decoder_artifact(
            tmp_path / "bad-fx",
            CURELiteDecoder(feature_channels=3),
            config,
            bad_logs,
        )


def test_decoder_artifact_enforces_variant_exposure_contracts(tmp_path) -> None:
    missing_u_pool = json.loads(json.dumps(_logs()))
    missing_u_pool[0]["pool_sizes"]["synthetic"] = 0
    with pytest.raises(ValueError, match="non-empty deletion-synthetic pool"):
        _save_decoder_artifact(
            tmp_path / "bad-u-pool",
            CURELiteDecoder(feature_channels=3),
            _config(),
            missing_u_pool,
        )

    wrong_u_exposure = json.loads(json.dumps(_logs()))
    wrong_u_exposure[0]["metrics"]["synthetic/states"] = 3.0
    with pytest.raises(ValueError, match="U third loss slot exposure"):
        _save_decoder_artifact(
            tmp_path / "bad-u-exposure",
            CURELiteDecoder(feature_channels=3),
            _config(),
            wrong_u_exposure,
        )

    factual_config = replace(_config(), variant="factual_only")
    factual_logs = json.loads(json.dumps(_logs()))
    for row in factual_logs:
        row["pool_sizes"]["synthetic"] = 0
        for quantity in ("active", "states"):
            for suffix in ("", "_min", "_max"):
                row["metrics"][f"synthetic/{quantity}{suffix}"] = 0.0
    _save_decoder_artifact(
        tmp_path / "valid-f",
        CURELiteDecoder(feature_channels=3),
        factual_config,
        factual_logs,
    )
    factual_logs[0]["metrics"]["synthetic/active"] = 1.0
    with pytest.raises(ValueError, match="third loss slot inactive"):
        _save_decoder_artifact(
            tmp_path / "bad-f-exposure",
            CURELiteDecoder(feature_channels=3),
            factual_config,
            factual_logs,
        )
