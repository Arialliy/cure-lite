from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from cure_lite.config import DecoderConfig
from cure_lite.config import MatchConfig
from cure_lite.calibration import FalseAlarmBudget
from cure_lite.data import ManifestImageDataset, PreprocessConfig
from cure_lite.experiment.artifacts import load_decoder_artifact
from cure_lite.experiment.cache_pipeline import (
    LoadedDRCacheBundle,
    cache_d_r_states,
    cache_manifest_split,
    load_d_r_cache_bundle,
    load_d_v_cache_bundle,
)
from cure_lite.experiment.formal_evaluation import (
    build_loaded_d_v_method_run,
    calibrate_paired_gate2,
    evaluate_formal_base_threshold,
    evaluate_paired_gate2,
    select_formal_base_threshold,
    select_formal_residual_threshold,
)
from cure_lite.experiment.formal_anchor import (
    build_loaded_d_v_base_run,
    evaluate_frozen_anchor,
    select_frozen_anchor,
)
from cure_lite.experiment.evaluation_pipeline import (
    calibration_samples_fingerprint,
)
from cure_lite.experiment.formal_training import (
    PairedGate2TrainingConfig,
    prepare_gate2_training,
    run_paired_gate2_training,
    save_completed_decoder_run,
    summarize_gate2_training_support,
)
from cure_lite.splits import SplitManifest, SplitRecord
from cure_lite.toy import ToyFrozenBaseAdapter


def _write_scene(
    root: Path,
    sample_id: str,
    target_values: tuple[int, int],
) -> None:
    image = np.zeros((32, 32), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    for value, (top, left) in zip(
        target_values,
        ((6, 6), (22, 22)),
        strict=True,
    ):
        image[top : top + 3, left : left + 3] = value
        mask[top : top + 3, left : left + 3] = 255
    Image.fromarray(image, mode="L").save(root / "images" / f"{sample_id}.png")
    Image.fromarray(mask, mode="L").save(root / "masks" / f"{sample_id}.png")


def _d_r_dataset(tmp_path: Path) -> tuple[ManifestImageDataset, str]:
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    scenes = {
        "db": (255, 255),
        "dr-covered": (255, 255),
        # 89/255 < 0.5, hence p=0.1+0.8*x remains below tau_o=0.5.
        "dr-miss": (89, 255),
        "dv": (89, 255),
        "dt": (89, 255),
    }
    for sample_id, values in scenes.items():
        _write_scene(tmp_path, sample_id, values)
    records = (
        SplitRecord("db", "D_B", "g-db", "images/db.png", "masks/db.png"),
        SplitRecord(
            "dr-miss",
            "D_R",
            "g-dr-miss",
            "images/dr-miss.png",
            "masks/dr-miss.png",
        ),
        SplitRecord(
            "dr-covered",
            "D_R",
            "g-dr-covered",
            "images/dr-covered.png",
            "masks/dr-covered.png",
        ),
        SplitRecord("dv", "D_V", "g-dv", "images/dv.png", "masks/dv.png"),
        SplitRecord("dt", "D_T", "g-dt", "images/dt.png", "masks/dt.png"),
    )
    manifest = SplitManifest(dataset="formal-training-toy", records=records)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.canonical_payload(), sort_keys=True),
        encoding="utf-8",
    )
    loaded = SplitManifest.load(manifest_path)
    dataset = ManifestImageDataset(
        loaded,
        "D_R",
        PreprocessConfig(
            height=32,
            width=32,
            color_mode="L",
            mean=(0.0,),
            std=(1.0,),
        ),
        manifest_path=manifest_path,
    )
    return dataset, str(manifest_path)


def test_prepared_bundle_is_fully_verified_only_at_semantic_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, _ = _d_r_dataset(tmp_path)
    adapter = ToyFrozenBaseAdapter()
    base_root = tmp_path / "base"
    cache_manifest_split(adapter, dataset, "D_R", base_root)
    state_root = tmp_path / "state"
    cache_d_r_states(
        base_root / "index.json",
        dataset,
        state_root,
        expected_base_fingerprint=adapter.fingerprint,
    )
    bundle = load_d_r_cache_bundle(
        state_root / "index.json",
        dataset,
        expected_base_fingerprint=adapter.fingerprint,
    )

    calls = 0
    original = LoadedDRCacheBundle.verify_unchanged

    def counted(current: LoadedDRCacheBundle) -> None:
        nonlocal calls
        calls += 1
        original(current)

    monkeypatch.setattr(LoadedDRCacheBundle, "verify_unchanged", counted)
    prepared = prepare_gate2_training(bundle)
    assert calls == 2  # before and after one-time semantic preparation

    summarize_gate2_training_support(bundle, prepared=prepared)
    assert calls == 2  # support is already part of the sealed catalog

    run_paired_gate2_training(
        bundle,
        PairedGate2TrainingConfig(
            decoder_config=DecoderConfig(feature_channels=3),
            optimizer="sgd",
            learning_rate=1e-3,
            epochs=1,
            steps_per_epoch=1,
            global_seed=17,
        ),
        prepared=prepared,
    )
    assert calls == 4  # once immediately before training and once after F/Fx/U


def test_formal_paired_training_binds_bundle_initialization_and_artifacts(
    tmp_path: Path,
) -> None:
    dataset, manifest_path = _d_r_dataset(tmp_path)
    adapter = ToyFrozenBaseAdapter()
    base_root = tmp_path / "base"
    cache_manifest_split(adapter, dataset, "D_R", base_root)

    # tau_o is selected from D_V base-only outputs before any D_R state is
    # constructed and before either residual decoder exists.
    manifest = SplitManifest.load(manifest_path)
    d_v_dataset = ManifestImageDataset(
        manifest,
        "D_V",
        dataset.preprocess,
        manifest_path=manifest_path,
    )
    d_v_cache_root = tmp_path / "d-v-base"
    cache_manifest_split(adapter, d_v_dataset, "D_V", d_v_cache_root)
    d_v_bundle = load_d_v_cache_bundle(
        d_v_cache_root / "index.json",
        d_v_dataset,
        expected_base_fingerprint=adapter.fingerprint,
    )
    base_run = build_loaded_d_v_base_run(d_v_bundle)
    anchor = select_frozen_anchor(base_run, [0.5, 0.7], MatchConfig())
    assert anchor.selected_threshold == 0.7  # equal global mIoU -> higher tau_o
    assert evaluate_frozen_anchor(base_run, anchor) == anchor.selected_metrics
    assert anchor.canonical_payload()["selection_rule"] == (
        "max_global_miou_tie_higher_threshold"
    )
    assert {
        "rmr",
        "gross_rmr",
        "net_rmr",
        "reachable_rmr",
        "oracle_upper_bound",
        "overlap_supported_rmr",
    }.isdisjoint(anchor.canonical_payload()["selected_metrics"])
    with pytest.raises(TypeError, match="receipt fields were replaced"):
        replace(anchor, candidate_threshold_grid=(0.5,))
    budget = FalseAlarmBudget(
        pixel_fa_budget=1.0,
        component_fa_per_mp_budget=float("inf"),
        raw_background_fa_budget=1.0,
        minimum_retention=0.0,
    )
    # Base@B is fully selectable and replayable before any decoder training or
    # artifact exists.  Its receipt binds only the base-run cache provenance.
    independent_base = select_formal_base_threshold(
        base_run,
        [0.3, 0.5],
        anchor.occupancy_config,
        anchor.match_config,
        budget,
    )
    assert evaluate_formal_base_threshold(base_run, independent_base) == (
        independent_base.protocol.selected_metrics
    )
    for decoder_field in (
        "decoder_artifact_fingerprint",
        "decoder_receipt_sha256",
        "decoder_state_fingerprint",
        "decoder_variant",
        "global_seed",
    ):
        assert not hasattr(independent_base, decoder_field)
    with pytest.raises(TypeError, match="Base@B threshold receipt fields"):
        replace(independent_base, base_index_sha256="0" * 64)

    state_root = tmp_path / "state"
    cache_d_r_states(
        base_root / "index.json",
        dataset,
        state_root,
        expected_base_fingerprint=adapter.fingerprint,
        occupancy_config=anchor.occupancy_config,
        match_config=anchor.match_config,
    )
    bundle = load_d_r_cache_bundle(
        state_root / "index.json",
        dataset,
        expected_base_fingerprint=adapter.fingerprint,
    )

    result = run_paired_gate2_training(
        bundle,
        PairedGate2TrainingConfig(
            decoder_config=DecoderConfig(feature_channels=3),
            optimizer="sgd",
            learning_rate=1e-3,
            epochs=1,
            steps_per_epoch=1,
            global_seed=17,
        ),
    )

    assert result.factual_only.config.initial_decoder_fingerprint == (
        result.factual_exposure_matched.config.initial_decoder_fingerprint
    ) == (
        result.uniform_legal.config.initial_decoder_fingerprint
    )
    assert result.factual_only.config.base_index_sha256 == bundle.base_index_sha256
    assert result.uniform_legal.config.state_index_sha256 == bundle.state_index_sha256
    assert dict(result.factual_only.training_log.epoch_logs[0].pool_sizes) == {
        "factual_miss": 1,
        "factual_no_miss": 1,
        "synthetic": 0,
    }
    exposure_log = result.factual_exposure_matched.training_log.epoch_logs[0]
    assert dict(exposure_log.pool_sizes) == {
        "factual_miss": 1,
        "factual_no_miss": 1,
        "synthetic": 0,
    }
    assert dict(exposure_log.metrics)["synthetic/active"] == 1.0
    assert dict(exposure_log.metrics)["synthetic/states"] == 1.0
    assert result.factual_exposure_matched.config.variant_contract == {
        "third_loss_slot_source": "independent_factual_positive_replacement",
        "third_loss_slot_batch": 1,
        "third_loss_slot_coefficient": "training_config.lambda_synthetic",
        "matched_reference_variant": "uniform_legal",
        "deletion_intervention_used": False,
    }
    assert dict(result.uniform_legal.training_log.epoch_logs[0].pool_sizes)[
        "synthetic"
    ] >= 1
    with pytest.raises(TypeError, match="completed decoder run fields were replaced"):
        replace(
            result.uniform_legal,
            config=replace(result.uniform_legal.config, variant="factual_only"),
        )

    factual_fingerprint = save_completed_decoder_run(
        str(tmp_path / "factual-artifact"), result.factual_only
    )
    uniform_fingerprint = save_completed_decoder_run(
        str(tmp_path / "uniform-artifact"), result.uniform_legal
    )
    exposure_fingerprint = save_completed_decoder_run(
        str(tmp_path / "exposure-artifact"),
        result.factual_exposure_matched,
    )
    loaded_factual = load_decoder_artifact(
        tmp_path / "factual-artifact",
        expected_config=result.factual_only.config,
    )
    loaded_uniform = load_decoder_artifact(
        tmp_path / "uniform-artifact",
        expected_config=result.uniform_legal.config,
    )
    loaded_exposure = load_decoder_artifact(
        tmp_path / "exposure-artifact",
        expected_config=result.factual_exposure_matched.config,
    )
    assert loaded_factual.artifact_fingerprint == factual_fingerprint
    assert loaded_uniform.artifact_fingerprint == uniform_fingerprint
    assert loaded_exposure.artifact_fingerprint == exposure_fingerprint

    with pytest.raises(TypeError, match="bound fields were replaced"):
        replace(d_v_bundle, base_fingerprint="0" * 64)
    factual_d_v_run = build_loaded_d_v_method_run(d_v_bundle, loaded_factual)
    exposure_d_v_run = build_loaded_d_v_method_run(d_v_bundle, loaded_exposure)
    uniform_d_v_run = build_loaded_d_v_method_run(d_v_bundle, loaded_uniform)
    factual_sample = factual_d_v_run.residual_samples[0]
    forged_samples = (
        replace(
            factual_sample,
            residual_probability=torch.ones_like(
                factual_sample.residual_probability
            ),
        ),
    )
    with pytest.raises(TypeError, match="source objects were replaced"):
        replace(
            factual_d_v_run,
            residual_samples=forged_samples,
            residual_samples_fingerprint=calibration_samples_fingerprint(
                forged_samples
            ),
        )
    mismatched_anchor = select_frozen_anchor(base_run, [0.5], MatchConfig())
    with pytest.raises(
        RuntimeError,
        match="decoder occupancy config differs from frozen tau_o",
    ):
        calibrate_paired_gate2(
            base_run,
            factual_d_v_run,
            exposure_d_v_run,
            uniform_d_v_run,
            anchor=mismatched_anchor,
            residual_thresholds=[0.5, 0.9],
            base_thresholds=[0.3, 0.5],
            budget=budget,
        )
    calibration = calibrate_paired_gate2(
        base_run,
        factual_d_v_run,
        exposure_d_v_run,
        uniform_d_v_run,
        anchor=anchor,
        residual_thresholds=[0.5, 0.9],
        base_thresholds=[0.3, 0.5],
        budget=budget,
    )
    legacy_base = select_formal_base_threshold(
        base_run,
        [0.3, 0.5],
        anchor.occupancy_config,
        anchor.match_config,
        budget,
    )
    legacy_factual = select_formal_residual_threshold(
        factual_d_v_run,
        [0.5, 0.9],
        budget,
    )
    legacy_exposure = select_formal_residual_threshold(
        exposure_d_v_run,
        [0.5, 0.9],
        budget,
    )
    legacy_uniform = select_formal_residual_threshold(
        uniform_d_v_run,
        [0.5, 0.9],
        budget,
    )
    assert calibration.base_at_budget.protocol == legacy_base.protocol
    assert calibration.base_at_budget == independent_base
    assert calibration.factual_only.protocol == legacy_factual.protocol
    assert (
        calibration.factual_exposure_matched.protocol
        == legacy_exposure.protocol
    )
    assert calibration.uniform_legal.protocol == legacy_uniform.protocol
    metrics = evaluate_paired_gate2(
        base_run,
        factual_d_v_run,
        exposure_d_v_run,
        uniform_d_v_run,
        calibration,
    )
    assert metrics.anchor == calibration.anchor.selected_metrics
    assert metrics.base_at_budget == (
        calibration.base_at_budget.protocol.selected_metrics
    )
    assert metrics.factual_only == calibration.factual_only.protocol.selected_metrics
    assert metrics.factual_exposure_matched == (
        calibration.factual_exposure_matched.protocol.selected_metrics
    )
    assert metrics.uniform_legal == calibration.uniform_legal.protocol.selected_metrics
    assert calibration.anchor.selected_threshold == 0.7
    assert calibration.factual_only.protocol.candidate_threshold_grid == (
        calibration.factual_exposure_matched.protocol.candidate_threshold_grid
    ) == (
        calibration.uniform_legal.protocol.candidate_threshold_grid
    )
    assert (
        calibration.uniform_legal.decoder_artifact_fingerprint
        == uniform_fingerprint
    )
    assert (
        calibration.uniform_legal.d_v_gt_fingerprint
        == d_v_bundle.d_v_gt_fingerprint
    )

    with torch.no_grad():
        next(result.uniform_legal.decoder.parameters()).add_(1.0)
    with pytest.raises(RuntimeError, match="decoder changed"):
        result.verify_unchanged()
