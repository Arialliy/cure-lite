from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.config import (
    DecoderConfig,
    InterventionConfig,
    LossConfig,
    MatchConfig,
    OccupancyConfig,
)
from cure_lite.decoder import CURELiteDecoder
from cure_lite.experiment.artifacts import decoder_state_fingerprint
from cure_lite.experiment.paired_artifacts import (
    PAIRED_DECODER_ARTIFACT_SCHEMA,
    PAIRED_METHODS,
    PairedDecoderRunConfig,
    PairedExecutionLedger,
    load_paired_decoder_artifact,
    method_objective_contract,
    save_paired_decoder_artifact,
)


def _config(method: str = "paired_difference") -> PairedDecoderRunConfig:
    return PairedDecoderRunConfig(
        method=method,
        seed=42,
        manifest_fingerprint="1" * 64,
        manifest_file_sha256="2" * 64,
        preprocessing_fingerprint="3" * 64,
        base_fingerprint="4" * 64,
        state_fingerprint="5" * 64,
        gt_fingerprint="6" * 64,
        base_index_fingerprint="7" * 64,
        base_index_sha256="8" * 64,
        state_index_fingerprint="9" * 64,
        state_index_sha256="a" * 64,
        formal_protocol_fingerprint="b" * 64,
        paired_objective_fingerprint="c" * 64,
        pair_catalog_fingerprint="d" * 64,
        paired_schedule_fingerprint="e" * 64,
        formal_schedule_fingerprint="f" * 64,
        runtime_input_fingerprint="9" * 64,
        control_preflight_fingerprint="0" * 64,
        control_provider_fingerprint=None,
        method_contract_fingerprint=stable_fingerprint(
            method_objective_contract(method)
        ),
        initial_decoder_fingerprint="2" * 64,
        occupancy_config=OccupancyConfig(),
        match_config=MatchConfig(),
        intervention_config=InterventionConfig(),
        decoder_config=DecoderConfig(feature_channels=3),
        absolute_loss_config=LossConfig(),
    )


def _logs() -> list[dict[str, object]]:
    return [
        {
            "epoch": epoch,
            "steps": 40,
            "metrics": {
                "mean_total_loss": 1.0 / (epoch + 1),
                "mean_factual_miss_loss": 0.3,
                "mean_factual_no_miss_loss": 0.2,
                "mean_paired_or_control_loss": 0.5,
                "minimum_total_loss": 0.1,
                "maximum_total_loss": 1.0,
            },
        }
        for epoch in range(800)
    ]


def _ledger(
    decoder: CURELiteDecoder,
    config: PairedDecoderRunConfig,
) -> PairedExecutionLedger:
    return PairedExecutionLedger(
        method=config.method,
        seed=config.seed,
        formal_schedule_fingerprint=config.formal_schedule_fingerprint,
        runtime_input_fingerprint=config.runtime_input_fingerprint,
        control_provider_fingerprint=config.control_provider_fingerprint,
        pair_exposure_fingerprint="3" * 64,
        factual_miss_exposure_fingerprint="4" * 64,
        factual_no_miss_exposure_fingerprint="5" * 64,
        initial_decoder_fingerprint=config.initial_decoder_fingerprint,
        final_decoder_fingerprint=decoder_state_fingerprint(decoder),
        trainable_parameter_count=sum(
            parameter.numel() for parameter in decoder.parameters()
        ),
        minimum_gradient_l2_norm=0.1,
        maximum_gradient_l2_norm=2.0,
    )


def test_config_is_separate_from_old_synthetic_artifact_semantics() -> None:
    config = _config()
    payload = config.canonical_payload()
    assert payload["schema_version"] == PAIRED_DECODER_ARTIFACT_SCHEMA
    assert payload["method"] == "paired_difference"
    assert payload["per_update"]["decoder_states"] == 12
    assert payload["per_update"]["decoder_forward_calls"] == 3
    assert payload["checkpoint_resume"] is False
    assert "variant_contract" not in payload
    assert PairedDecoderRunConfig.from_mapping(payload) == config
    assert payload["method_objective"] == method_objective_contract(
        "paired_difference"
    )

    for method in PAIRED_METHODS:
        method_config = replace(
            config,
            method=method,
            control_provider_fingerprint=(
                None if method == "paired_difference" else "4" * 64
            ),
            method_contract_fingerprint=stable_fingerprint(
                method_objective_contract(method)
            ),
        )
        assert method_config.method == method
        assert method_config.canonical_payload()["method_objective"] == (
            method_objective_contract(method)
        )
    assert method_objective_contract("independent_endpoint")[
        "coupled_difference_value"
    ] is False
    assert method_objective_contract("after_only")["gradient_path"] == (
        "minus_only"
    )
    assert method_objective_contract("plus_detach")[
        "both_endpoints_receive_paired_gradient"
    ] is False
    with pytest.raises(ValueError, match="method"):
        replace(config, method="uniform_legal")
    with pytest.raises(ValueError, match="control provider"):
        replace(config, control_provider_fingerprint="4" * 64)
    with pytest.raises(ValueError, match="control_provider_fingerprint"):
        replace(
            config,
            method="after_only",
            control_provider_fingerprint=None,
        )
    with pytest.raises(ValueError, match="method_contract_fingerprint"):
        replace(config, method_contract_fingerprint="0" * 64)
    with pytest.raises(ValueError, match="epochs"):
        replace(config, epochs=799)
    decoder = CURELiteDecoder(feature_channels=3)
    with pytest.raises(ValueError, match="minimum_gradient_l2_norm"):
        replace(_ledger(decoder, config), minimum_gradient_l2_norm=0.0)


def test_paired_artifact_round_trip_and_create_only(tmp_path) -> None:
    torch.manual_seed(7)
    decoder = CURELiteDecoder(feature_channels=3)
    config = replace(
        _config(),
        initial_decoder_fingerprint=decoder_state_fingerprint(decoder),
    )
    with torch.no_grad():
        next(decoder.parameters()).add_(0.01)
    ledger = _ledger(decoder, config)
    target = tmp_path / "paired"
    fingerprint = save_paired_decoder_artifact(
        target,
        decoder,
        config,
        _logs(),
        ledger,
    )

    loaded = load_paired_decoder_artifact(target, expected_config=config)
    assert loaded.artifact_fingerprint == fingerprint
    assert loaded.execution_ledger == ledger
    assert len(loaded.epoch_logs) == 800
    assert not loaded.decoder.training
    assert all(
        not parameter.requires_grad
        for parameter in loaded.decoder.parameters()
    )
    loaded.verify_unchanged()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_paired_decoder_artifact(
            target,
            decoder,
            config,
            _logs(),
            ledger,
        )


def test_loader_rejects_incomplete_and_tampered_artifacts(tmp_path) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / ".incomplete").write_text("running\n", encoding="utf-8")
    with pytest.raises(ValueError, match="complete and canonical"):
        load_paired_decoder_artifact(incomplete)

    decoder = CURELiteDecoder(feature_channels=3)
    config = replace(
        _config(),
        initial_decoder_fingerprint=decoder_state_fingerprint(decoder),
    )
    with torch.no_grad():
        next(decoder.parameters()).add_(0.01)
    target = tmp_path / "tampered"
    save_paired_decoder_artifact(
        target,
        decoder,
        config,
        _logs(),
        _ledger(decoder, config),
    )
    ledger_path = target / "execution_ledger.json"
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["optimizer_updates"] = 31_999
    ledger_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        load_paired_decoder_artifact(target)


def test_ledger_rejects_shortened_compute_or_recovery_semantics() -> None:
    decoder = CURELiteDecoder(feature_channels=3)
    config = replace(
        _config(),
        initial_decoder_fingerprint=decoder_state_fingerprint(decoder),
    )
    with torch.no_grad():
        next(decoder.parameters()).add_(0.01)
    ledger = _ledger(decoder, config)
    with pytest.raises(ValueError, match="optimizer_updates"):
        replace(ledger, optimizer_updates=31_999)
    with pytest.raises(ValueError, match="finite gradients"):
        replace(ledger, all_gradients_finite=False)

    payload = config.canonical_payload()
    payload["checkpoint_resume"] = True
    with pytest.raises(ValueError, match="recovery"):
        PairedDecoderRunConfig.from_mapping(payload)
