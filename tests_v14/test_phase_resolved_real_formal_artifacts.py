from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.experiment.phase_resolved_real_artifacts import (
    PFCRRealDecoderRunConfig,
    load_pfcr_real_decoder_artifact,
    save_pfcr_real_decoder_artifact,
)
from cure_lite.experiment.phase_resolved_real_formal_runner import (
    load_pfcr_development_authorization,
    load_pfcr_real_formal_attempt,
    load_pfcr_real_preflight_authorization,
)
from cure_lite.experiment.phase_resolved_real_training import (
    PFCRRealFormalExecutionLedger,
    PFCRRealFormalTrainingConfig,
    pfcr_model_state_fingerprint,
)
from cure_lite.phase_resolved_relation_decoder import (
    CURELitePhaseResolvedRelationDecoder,
    PhaseResolvedRelationDecoderConfig,
)


ROOT = Path(__file__).resolve().parents[1]


def _digest(label: str) -> str:
    return stable_fingerprint({"label": label})


def _epoch_logs() -> list[dict[str, object]]:
    return [
        {
            "epoch": epoch,
            "steps": 40,
            "optimizer_updates_completed": (epoch + 1) * 40,
            "decoder_forward_calls": 40,
            "decoder_state_evaluations": 480,
            "metrics": {
                "mean_total_loss": 6.0,
                "mean_factual_miss_loss": 1.0,
                "mean_factual_no_miss_loss": 2.0,
                "mean_synthetic_loss": 3.0,
                "minimum_total_loss": 5.0,
                "maximum_total_loss": 7.0,
                "minimum_gradient_l2_norm": 0.1,
                "maximum_gradient_l2_norm": 0.2,
            },
        }
        for epoch in range(800)
    ]


def _finished_objects():
    decoder = CURELitePhaseResolvedRelationDecoder(
        PhaseResolvedRelationDecoderConfig(
            feature_channels=4,
            feature_stride=2,
            relation_dim=8,
        )
    )
    initial = pfcr_model_state_fingerprint(decoder)
    with torch.no_grad():
        decoder.baseline_raw.add_(0.125)
    final = pfcr_model_state_fingerprint(decoder)
    training = PFCRRealFormalTrainingConfig(seed=42)
    config = PFCRRealDecoderRunConfig(
        seed=42,
        cache_contract_fingerprint=_digest("cache"),
        state_catalog_fingerprint=_digest("catalog"),
        lineage_allowlist_fingerprint=_digest("allowlist"),
        formal_schedule_fingerprint=_digest("schedule"),
        preflight_result_fingerprint=_digest("preflight"),
        initial_model_fingerprint=initial,
        decoder_config=decoder.config,
        training_config=training,
    )
    ledger = PFCRRealFormalExecutionLedger(
        seed=42,
        cache_contract_fingerprint=config.cache_contract_fingerprint,
        state_catalog_fingerprint=config.state_catalog_fingerprint,
        lineage_allowlist_fingerprint=(
            config.lineage_allowlist_fingerprint
        ),
        formal_schedule_fingerprint=config.formal_schedule_fingerprint,
        initial_model_fingerprint=initial,
        final_model_fingerprint=final,
        optimizer_state_fingerprint=_digest("adam"),
        trainable_parameter_count=decoder.config.expected_parameter_count,
        minimum_gradient_l2_norm=0.1,
        maximum_gradient_l2_norm=0.2,
    )
    return decoder, config, ledger


def test_pfcr_formal_artifact_roundtrip_is_frozen_and_complete(
    tmp_path: Path,
) -> None:
    decoder, config, ledger = _finished_objects()
    target = tmp_path / "artifact"

    fingerprint = save_pfcr_real_decoder_artifact(
        target,
        decoder,
        config,
        _epoch_logs(),
        ledger,
    )
    loaded = load_pfcr_real_decoder_artifact(
        target,
        expected_config=config,
    )

    assert loaded.artifact_fingerprint == fingerprint
    assert len(loaded.epoch_logs) == 800
    assert loaded.execution_ledger.optimizer_updates == 32_000
    assert loaded.execution_ledger.minimum_adam_step == 32_000
    assert loaded.decoder.training is False
    assert all(
        not parameter.requires_grad
        for parameter in loaded.decoder.parameters()
    )
    loaded.verify_unchanged()


def test_pfcr_artifact_detects_in_memory_log_mutation(
    tmp_path: Path,
) -> None:
    decoder, config, ledger = _finished_objects()
    target = tmp_path / "artifact"
    save_pfcr_real_decoder_artifact(
        target,
        decoder,
        config,
        _epoch_logs(),
        ledger,
    )
    loaded = load_pfcr_real_decoder_artifact(target)
    loaded.epoch_logs[0]["steps"] = 39

    with pytest.raises(RuntimeError, match="changed in memory"):
        loaded.verify_unchanged()


def test_pfcr_artifact_refuses_short_log_and_existing_target(
    tmp_path: Path,
) -> None:
    decoder, config, ledger = _finished_objects()
    target = tmp_path / "artifact"

    with pytest.raises(ValueError, match="800 epochs"):
        save_pfcr_real_decoder_artifact(
            target,
            decoder,
            config,
            _epoch_logs()[:-1],
            ledger,
        )
    target.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_pfcr_real_decoder_artifact(
            target,
            decoder,
            config,
            _epoch_logs(),
            ledger,
        )


@pytest.mark.parametrize("seed", (42, 43))
def test_real_preflight_authorization_is_strict(seed: int) -> None:
    root = (
        ROOT
        / "runs/irstd1k_stage_a_seed42"
        / f"cure_lite_pfcr_real_preflight_v1_s{seed}_r1"
    )
    if not root.is_dir():
        pytest.skip("local PFCR real preflight is unavailable")

    loaded = load_pfcr_real_preflight_authorization(
        root,
        expected_seed=seed,
    )

    assert loaded.seed == seed
    assert len(loaded.result_fingerprint) == 64
    assert len(loaded.complete_fingerprint) == 64


def test_current_bounded_development_seeds_are_authorized() -> None:
    protocol = (
        ROOT
        / "protocols/IRSTD-1K"
        / "phase_resolved_feature_coverage_relation_v2"
    )
    seed42 = (
        protocol / "development_bounded_v3_seed42_r1.json"
    )
    seed43 = (
        protocol / "development_bounded_v3_seed43_r1.json"
    )
    if not seed42.is_file() or not seed43.is_file():
        pytest.skip("current PFCR Development results are unavailable")

    loaded = load_pfcr_development_authorization(
        seed42,
        seed43,
    )

    assert set(loaded.file_sha256_by_seed) == {42, 43}
    assert set(loaded.result_fingerprint_by_seed) == {42, 43}
    assert len(loaded.authorization_fingerprint) == 64


def test_incomplete_formal_attempt_cannot_be_loaded(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    (attempt / "STARTED.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete"):
        load_pfcr_real_formal_attempt(attempt)
