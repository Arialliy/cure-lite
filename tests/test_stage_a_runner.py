from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

import cure_lite.experiment.stage_a_runner as stage_a_runner_module
from cure_lite.base_identity import (
    VerifiedBaseRunIdentity,
    _bind_verified_base_run_identity,
)
from cure_lite.calibration import FalseAlarmBudget
from cure_lite.cache.schema import file_sha256
from cure_lite.config import DecoderConfig
from cure_lite.data import ManifestImageDataset, PreprocessConfig
from cure_lite.experiment.cache_pipeline import cache_manifest_split
from cure_lite.experiment.deployment import build_calibrated_cure_lite_model
from cure_lite.experiment.formal_training import PairedGate2TrainingConfig
from cure_lite.experiment.seed_registry import build_seed_registry_from_stage_a_run
from cure_lite.experiment.stage_a_runner import (
    StageARunConfig,
    load_stage_a_run,
    run_stage_a,
    run_stage_a_from_base_caches,
)
from cure_lite.experiment.training_pipeline import TrainingSupportRequirements
from cure_lite.frozen_base import frozen_base_state_fingerprint
from cure_lite.splits import SplitManifest, SplitRecord
from cure_lite.stage_a import BaseRunIdentity, STAGE_A_METHOD_ORDER
from cure_lite.toy import ToyFrozenBaseAdapter


_UNUSED_D_T_ID = "unused-dt-not-accessed"
_UNUSED_D_T_IMAGE = "unused-assets/dt-image.png"
_UNUSED_D_T_MASK = "unused-assets/dt-mask.png"


def _test_digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _verified_toy_base_identity(
    adapter: ToyFrozenBaseAdapter,
    *,
    label: str = "toy",
) -> VerifiedBaseRunIdentity:
    identity = BaseRunIdentity(
        producer_schema="cure-lite-test-toy-base-v1",
        base_fingerprint=adapter.fingerprint,
        base_state_fingerprint=frozen_base_state_fingerprint(adapter),
        training_run_fingerprint=_test_digest(f"{label}-training-run"),
        completion_receipt_sha256=_test_digest(f"{label}-completion"),
        checkpoint_sha256=_test_digest(f"{label}-checkpoint"),
        selection_fingerprint=_test_digest(f"{label}-selection"),
        source_fingerprint=_test_digest("toy-source"),
    )

    def verify_source() -> None:
        if adapter.fingerprint != identity.base_fingerprint:
            raise RuntimeError("toy Base identity source changed")
        if frozen_base_state_fingerprint(adapter) != identity.base_state_fingerprint:
            raise RuntimeError("toy Base state source changed")

    return _bind_verified_base_run_identity(identity, verify_source)


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


def _development_datasets(
    tmp_path: Path,
) -> tuple[ManifestImageDataset, ManifestImageDataset]:
    (tmp_path / "images").mkdir(parents=True)
    (tmp_path / "masks").mkdir()
    for sample_id, values in {
        "db": (255, 255),
        "dr-covered": (255, 255),
        # 89/255 maps to a base probability below the frozen tau_o=0.5.
        "dr-miss": (89, 255),
        "dv": (89, 255),
    }.items():
        _write_scene(tmp_path, sample_id, values)

    manifest = SplitManifest(
        dataset="formal-stage-a-toy",
        records=(
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
            # These paths deliberately do not exist.  Stage A must never touch
            # official-test assets even though membership is fixed in manifest.
            SplitRecord(
                _UNUSED_D_T_ID,
                "D_T",
                "g-dt-unused",
                _UNUSED_D_T_IMAGE,
                _UNUSED_D_T_MASK,
            ),
        ),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.canonical_payload(), sort_keys=True),
        encoding="utf-8",
    )
    loaded = SplitManifest.load(manifest_path)
    preprocess = PreprocessConfig(
        height=32,
        width=32,
        color_mode="L",
        mean=(0.0,),
        std=(1.0,),
    )
    return (
        ManifestImageDataset(
            loaded,
            "D_R",
            preprocess,
            manifest_path=manifest_path,
        ),
        ManifestImageDataset(
            loaded,
            "D_V",
            preprocess,
            manifest_path=manifest_path,
        ),
    )


def _stage_a_config(*, feature_channels: int = 3) -> StageARunConfig:
    return StageARunConfig(
        training=PairedGate2TrainingConfig(
            decoder_config=DecoderConfig(feature_channels=feature_channels),
            optimizer="sgd",
            learning_rate=1e-3,
            epochs=1,
            steps_per_epoch=1,
            global_seed=17,
        ),
        anchor_thresholds=(0.5,),
        base_thresholds=(0.3, 0.5),
        residual_thresholds=(0.5, 0.9),
        budget=FalseAlarmBudget(
            pixel_fa_budget=1.0,
            component_fa_per_mp_budget=float("inf"),
            raw_background_fa_budget=1.0,
            minimum_retention=0.0,
        ),
        device="cpu",
    )


def _all_json_text(root: Path) -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.json"))
    )


def _nested_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(
            *(_nested_mapping_keys(item) for item in value.values())
        )
    if isinstance(value, list):
        return set().union(*(_nested_mapping_keys(item) for item in value))
    return set()


def test_stage_a_config_rejects_devices_outside_efficiency_contract() -> None:
    with pytest.raises(ValueError, match="only CPU or CUDA"):
        replace(_stage_a_config(), device="mps")


def test_stage_a_device_preflight_skips_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_empty(*args: object, **kwargs: object) -> None:
        raise AssertionError("CPU preflight must not allocate a probe")

    def unexpected_synchronize(*args: object, **kwargs: object) -> None:
        raise AssertionError("CPU preflight must not synchronize CUDA")

    monkeypatch.setattr(stage_a_runner_module.torch, "empty", unexpected_empty)
    monkeypatch.setattr(
        stage_a_runner_module.torch.cuda,
        "synchronize",
        unexpected_synchronize,
    )
    stage_a_runner_module._preflight_stage_a_device("cpu")


def test_stage_a_device_preflight_executes_exact_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []

    class Probe:
        def fill_(self, value: float) -> "Probe":
            calls.append(("fill", value))
            return self

    def fake_empty(
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Probe:
        calls.append(("empty", (shape, dtype, device)))
        return Probe()

    def fake_synchronize(device: torch.device) -> None:
        calls.append(("synchronize", device))

    monkeypatch.setattr(stage_a_runner_module.torch, "empty", fake_empty)
    monkeypatch.setattr(
        stage_a_runner_module.torch.cuda,
        "synchronize",
        fake_synchronize,
    )

    stage_a_runner_module._preflight_stage_a_device("cuda:2")

    assert calls == [
        ("empty", ((1,), torch.float32, torch.device("cuda:2"))),
        ("fill", 0.0),
        ("synchronize", torch.device("cuda:2")),
    ]


def test_stage_a_device_preflight_preserves_cuda_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = RuntimeError("CUDA runtime unavailable")

    def failed_empty(*args: object, **kwargs: object) -> None:
        raise cause

    monkeypatch.setattr(stage_a_runner_module.torch, "empty", failed_empty)

    with pytest.raises(
        RuntimeError,
        match="Stage-A CUDA preflight failed for cuda:0 before output creation",
    ) as captured:
        stage_a_runner_module._preflight_stage_a_device("cuda:0")

    assert captured.value.__cause__ is cause


def test_stage_a_device_preflight_fails_before_direct_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d_r_dataset, d_v_dataset = _development_datasets(tmp_path / "data")
    adapter = ToyFrozenBaseAdapter()
    verified_identity = _verified_toy_base_identity(adapter, label="preflight")
    output_parent = tmp_path / "direct-output-parent"
    output = output_parent / "stage-a"

    def failed_preflight(device: str) -> None:
        assert device == "cuda:0"
        raise RuntimeError("preflight failure")

    monkeypatch.setattr(
        stage_a_runner_module,
        "_preflight_stage_a_device",
        failed_preflight,
    )

    with pytest.raises(RuntimeError, match="preflight failure"):
        run_stage_a(
            adapter,
            d_r_dataset,
            d_v_dataset,
            replace(_stage_a_config(), device="cuda:0"),
            output,
            verified_base_identity=verified_identity,
        )

    assert not output_parent.exists()


def test_stage_a_device_preflight_fails_before_cache_output_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d_r_dataset, d_v_dataset = _development_datasets(tmp_path / "data")
    adapter = ToyFrozenBaseAdapter()
    verified_identity = _verified_toy_base_identity(
        adapter,
        label="cache-preflight",
    )
    output_parent = tmp_path / "cache-output-parent"
    output = output_parent / "stage-a"

    def failed_preflight(device: str) -> None:
        assert device == "cuda:0"
        raise RuntimeError("preflight failure")

    monkeypatch.setattr(
        stage_a_runner_module,
        "_preflight_stage_a_device",
        failed_preflight,
    )

    with pytest.raises(RuntimeError, match="preflight failure"):
        run_stage_a_from_base_caches(
            tmp_path / "unused-d-r-index.json",
            tmp_path / "unused-d-v-index.json",
            d_r_dataset,
            d_v_dataset,
            replace(_stage_a_config(), device="cuda:0"),
            output,
            verified_base_identity=verified_identity,
        )

    assert not output_parent.exists()


def test_existing_stage_a_output_precedes_cuda_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d_r_dataset, d_v_dataset = _development_datasets(tmp_path / "data")
    adapter = ToyFrozenBaseAdapter()
    verified_identity = _verified_toy_base_identity(
        adapter,
        label="existing-output",
    )
    output = tmp_path / "existing"
    output.mkdir()
    calls = 0

    def tracked_preflight(device: str) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(
        stage_a_runner_module,
        "_preflight_stage_a_device",
        tracked_preflight,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_stage_a(
            adapter,
            d_r_dataset,
            d_v_dataset,
            replace(_stage_a_config(), device="cuda:0"),
            output,
            verified_base_identity=verified_identity,
        )

    assert calls == 0


def test_stage_a_single_entry_is_replayable_and_never_reads_d_t(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    d_r_dataset, d_v_dataset = _development_datasets(tmp_path / "data")
    adapter = ToyFrozenBaseAdapter()
    verified_toy_identity = _verified_toy_base_identity(adapter)
    output = tmp_path / "stage-a-run"
    full_replay_calls = 0
    original_full_replay = stage_a_runner_module._load_verified_state

    def tracked_full_replay(*args: object, **kwargs: object) -> object:
        nonlocal full_replay_calls
        full_replay_calls += 1
        return original_full_replay(*args, **kwargs)

    monkeypatch.setattr(
        stage_a_runner_module,
        "_load_verified_state",
        tracked_full_replay,
    )

    completed = run_stage_a(
        adapter,
        d_r_dataset,
        d_v_dataset,
        _stage_a_config(),
        output,
        verified_base_identity=verified_toy_identity,
        calibration_workers=2,
    )
    assert full_replay_calls == 0

    assert (output / "COMPLETE.json").is_file()
    assert not (output / ".incomplete").exists()
    for receipt_name in (
        "config",
        "anchor",
        "support",
        "calibration",
        "results",
        "efficiency",
    ):
        assert (output / "receipts" / f"{receipt_name}.json").is_file()
    for variant in (
        "factual_only",
        "factual_exposure_matched",
        "uniform_legal",
    ):
        assert (output / "decoders" / variant / "receipt.json").is_file()
    assert {path.name for path in (output / "decoders").iterdir()} == {
        "factual_only",
        "factual_exposure_matched",
        "uniform_legal",
    }

    completed.verify_unchanged()
    assert full_replay_calls == 1
    completed.verify_published_receipts()
    reloaded = load_stage_a_run(
        output,
        d_r_dataset,
        d_v_dataset,
        verified_base_identity=verified_toy_identity,
    )
    reloaded.verify_unchanged()
    assert reloaded.results == completed.results
    assert reloaded.calibration == completed.calibration
    assert reloaded.support_summary == completed.support_summary
    assert reloaded.efficiency == completed.efficiency
    assert completed.support_summary.factual_miss_images >= 1
    assert completed.support_summary.factual_no_miss_images >= 1
    assert completed.support_summary.synthetic_images >= 1
    assert completed.efficiency.parameter_count > 0
    assert completed.efficiency.conv2d_macs > 0
    assert completed.efficiency.conv2d_flops == 2 * completed.efficiency.conv2d_macs
    assert completed.efficiency.binding.decoder_artifact_fingerprint == (
        completed.uniform_artifact.artifact_fingerprint
    )
    efficiency_payload = completed.efficiency.canonical_payload()
    assert efficiency_payload["deployed_method"] == "U"
    assert efficiency_payload["static_evidence"]["decoder_variant"] == (
        "uniform_legal"
    )
    assert efficiency_payload["static_evidence"]["input_contract"][
        "reads_image_or_gt_content"
    ] is False

    seed_registry = build_seed_registry_from_stage_a_run(
        completed,
        verified_toy_identity,
    )
    common = seed_registry["common_config"]
    assert tuple(seed_registry["protocols"]) == STAGE_A_METHOD_ORDER
    assert seed_registry["protocols"]["A"]["decoder_artifact_fingerprint"] is None
    assert (
        seed_registry["protocols"]["Base@B"]["decoder_artifact_fingerprint"]
        is None
    )
    assert common["d_v_base_cache_index_fingerprint"] == (
        completed.d_v_bundle.base_index_fingerprint
    )
    assert common["d_r_state_cache_index_sha256"] == (
        completed.d_r_bundle.state_index_sha256
    )
    assert common["stage_a_complete_fingerprint"] == completed.complete_fingerprint
    assert common["efficiency_device_type"] == completed.efficiency.device_type
    assert common["efficiency_warmup"] == completed.efficiency.warmup
    assert common["efficiency_repetitions"] == completed.efficiency.repetitions
    assert common["efficiency_static_fingerprint"] == (
        completed.efficiency.static_fingerprint
    )
    assert common["efficiency_receipt_fingerprint"] == (
        completed.efficiency.receipt_fingerprint
    )

    calibrated = build_calibrated_cure_lite_model(
        completed,
        adapter,
        method="U",
    )
    calibrated.verify_unchanged()
    calibrated_output = calibrated(d_v_dataset[0].image.unsqueeze(0))
    assert calibrated_output.final_mask.shape == (1, 1, 32, 32)
    assert calibrated.residual_threshold == (
        completed.calibration.uniform_legal.protocol.selected_threshold
    )
    assert calibrated.base_fingerprint == adapter.fingerprint
    assert calibrated.receipt.base_state_fingerprint == (
        completed.d_v_bundle.base_state_fingerprint
    )
    assert calibrated.receipt.base_state_fingerprint == (
        frozen_base_state_fingerprint(adapter)
    )
    assert calibrated.receipt.decoder_variant == "uniform_legal"
    assert calibrated.receipt.base_probability_shape == (1, 32, 32)
    assert calibrated.receipt.base_feature_shape == (3, 32, 32)
    assert not hasattr(calibrated, "_stage_a_run")
    assert not hasattr(calibrated, "_artifact")
    assert not hasattr(calibrated, "_preprocessing")
    assert calibrated._core.decoder is not completed.uniform_artifact.decoder

    calibrated_f = build_calibrated_cure_lite_model(completed, adapter, method="F")
    calibrated_fx = build_calibrated_cure_lite_model(
        completed,
        adapter,
        method="F×",
    )
    calibrated_u_second = build_calibrated_cure_lite_model(
        completed,
        adapter,
        method="U",
    )
    assert calibrated_f.receipt.decoder_variant == "factual_only"
    assert calibrated_fx.receipt.decoder_variant == "factual_exposure_matched"
    assert calibrated._core.decoder is not calibrated_u_second._core.decoder
    assert calibrated.decoder_artifact_fingerprint == (
        calibrated_u_second.decoder_artifact_fingerprint
    )

    wrong_base = ToyFrozenBaseAdapter()
    wrong_base._fingerprint = "0" * 64
    with pytest.raises(RuntimeError, match="Base identity"):
        build_calibrated_cure_lite_model(completed, wrong_base, method="U")
    wrong_state = ToyFrozenBaseAdapter()
    assert wrong_state.fingerprint == adapter.fingerprint
    with torch.no_grad():
        next(wrong_state.base.parameters()).data.add_(1.0)
    with pytest.raises(RuntimeError, match="state differs from Stage-A caches"):
        build_calibrated_cure_lite_model(completed, wrong_state, method="U")
    with pytest.raises(ValueError, match="grid/channels"):
        calibrated(torch.zeros(1, 1, 16, 16, dtype=torch.float32))
    with pytest.raises(TypeError, match="float32"):
        calibrated(torch.zeros(1, 1, 32, 32, dtype=torch.float64))
    with pytest.raises(RuntimeError, match="move the frozen Base"):
        calibrated.to("cpu")
    with pytest.raises(RuntimeError, match="inference-only"):
        calibrated.requires_grad_(True)
    calibrated._core.train(True)
    with pytest.raises(RuntimeError, match="left inference mode"):
        calibrated(d_v_dataset[0].image.unsqueeze(0))
    calibrated._core.train(False)
    with pytest.raises(RuntimeError, match="inference-only"):
        calibrated.train()
    with pytest.raises(TypeError):
        calibrated(  # type: ignore[call-arg]
            d_v_dataset[0].image.unsqueeze(0),
            residual_threshold=0.5,
        )
    with torch.no_grad():
        next(calibrated._core.decoder.parameters()).data.add_(1.0)
    with pytest.raises(RuntimeError, match="decoder differs from artifact"):
        calibrated(d_v_dataset[0].image.unsqueeze(0))
    data_mutated_base = ToyFrozenBaseAdapter()
    data_mutated_model = build_calibrated_cure_lite_model(
        completed,
        data_mutated_base,
        method="U",
    )
    with torch.no_grad():
        next(data_mutated_base.base.parameters()).data.add_(1.0)
    with pytest.raises(RuntimeError, match="Base state changed"):
        data_mutated_model.verify_unchanged()
    signature_mutated_base = ToyFrozenBaseAdapter()
    signature_mutated_model = build_calibrated_cure_lite_model(
        completed,
        signature_mutated_base,
        method="U",
    )
    with torch.no_grad():
        next(signature_mutated_base.base.parameters()).add_(1.0)
    with pytest.raises(RuntimeError, match="Base tensors changed"):
        signature_mutated_model(d_v_dataset[0].image.unsqueeze(0))

    results_payload = json.loads(
        (output / "receipts" / "results.json").read_text(encoding="utf-8")
    )
    calibration_payload = json.loads(
        (output / "receipts" / "calibration.json").read_text(encoding="utf-8")
    )
    complete_payload = json.loads(
        (output / "COMPLETE.json").read_text(encoding="utf-8")
    )
    assert set(results_payload["methods"]) == {"A", "Base@B", "F", "F×", "U"}
    assert set(calibration_payload["methods"]) == {
        "A",
        "Base@B",
        "F",
        "F×",
        "U",
    }
    assert complete_payload["schema_version"] == "cure-lite-stage-a-run-v7"
    assert complete_payload["base_state_fingerprint"] == (
        completed.d_v_bundle.base_state_fingerprint
    )
    assert complete_payload["base_run_identity"] == (
        verified_toy_identity.identity.to_registry_dict()
    )
    assert completed.base_run_identity is verified_toy_identity.identity
    anchor_receipt = calibration_payload["methods"]["A"]
    assert {
        "decoder_artifact_fingerprint",
        "decoder_receipt_sha256",
        "decoder_state_fingerprint",
        "decoder_variant",
        "global_seed",
    }.isdisjoint(_nested_mapping_keys(anchor_receipt))
    base_receipt = calibration_payload["methods"]["Base@B"]
    assert base_receipt["schema_version"] == (
        "cure-lite-stage-a-base-at-budget-receipt-v2"
    )
    assert base_receipt["method"] == "Base@B"
    assert base_receipt["d_v_base_run_fingerprint"] == (
        completed.calibration.base_at_budget.d_v_base_run_fingerprint
    )
    assert {
        "decoder_artifact_fingerprint",
        "decoder_receipt_sha256",
        "decoder_state_fingerprint",
        "decoder_variant",
        "global_seed",
    }.isdisjoint(_nested_mapping_keys(base_receipt))
    assert "decoder_artifact_fingerprint" in calibration_payload["methods"]["F"]
    forbidden_metric_fields = {
        "rmr",
        "gross_rmr",
        "net_rmr",
        "reachable_rmr",
        "oracle_upper_bound",
        "overlap_supported_rmr",
    }
    assert forbidden_metric_fields.isdisjoint(
        _nested_mapping_keys(results_payload)
    )
    assert forbidden_metric_fields.isdisjoint(
        _nested_mapping_keys(calibration_payload)
    )
    assert complete_payload["method_order"] == ["A", "Base@B", "F", "F×", "U"]
    assert complete_payload["efficiency_receipt_fingerprint"] == (
        completed.efficiency.receipt_fingerprint
    )

    persisted_json = _all_json_text(output)
    assert _UNUSED_D_T_ID not in persisted_json
    assert _UNUSED_D_T_IMAGE not in persisted_json
    assert _UNUSED_D_T_MASK not in persisted_json

    complete_bytes = (output / "COMPLETE.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_stage_a(
            adapter,
            d_r_dataset,
            d_v_dataset,
            _stage_a_config(),
            output,
            verified_base_identity=verified_toy_identity,
        )
    assert (output / "COMPLETE.json").read_bytes() == complete_bytes

    # Base@B is independently bound inside the calibration receipt.  Any edit
    # is rejected by the completed-run inventory before replay can use it.
    calibration_path = output / "receipts" / "calibration.json"
    calibration_bytes = calibration_path.read_bytes()
    tampered_calibration = json.loads(calibration_bytes)
    tampered_calibration["methods"]["Base@B"]["base_fingerprint"] = "0" * 64
    calibration_path.write_text(
        json.dumps(tampered_calibration, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifact file inventory changed"):
        load_stage_a_run(
            output,
            d_r_dataset,
            d_v_dataset,
            verified_base_identity=verified_toy_identity,
        )


def test_stage_a_rejects_insufficient_real_support_before_decoder_training(
    tmp_path: Path,
) -> None:
    d_r_dataset, d_v_dataset = _development_datasets(tmp_path / "data")
    adapter = ToyFrozenBaseAdapter()
    verified_base_identity = _verified_toy_base_identity(
        adapter, label="insufficient"
    )
    output = tmp_path / "insufficient-support"
    config = replace(
        _stage_a_config(),
        support_requirements=TrainingSupportRequirements(
            minimum_factual_miss_images=100,
        ),
    )

    with pytest.raises(RuntimeError, match="factual_miss_images"):
        run_stage_a(
            adapter,
            d_r_dataset,
            d_v_dataset,
            config,
            output,
            verified_base_identity=verified_base_identity,
        )

    assert (output / "receipts" / "anchor.json").is_file()
    assert not (output / "decoders").exists()
    assert not (output / "COMPLETE.json").exists()


def test_stage_a_failure_never_publishes_complete_marker(tmp_path: Path) -> None:
    d_r_dataset, d_v_dataset = _development_datasets(tmp_path / "data")
    adapter = ToyFrozenBaseAdapter()
    verified_base_identity = _verified_toy_base_identity(adapter, label="failure")
    output = tmp_path / "failed-stage-a-run"

    with pytest.raises((RuntimeError, ValueError)):
        run_stage_a(
            adapter,
            d_r_dataset,
            d_v_dataset,
            _stage_a_config(feature_channels=4),
            output,
            verified_base_identity=verified_base_identity,
        )

    assert not (output / "COMPLETE.json").exists()
    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        load_stage_a_run(
            output,
            d_r_dataset,
            d_v_dataset,
            verified_base_identity=verified_base_identity,
        )


def test_stage_a_from_generic_base_caches_is_self_contained(
    tmp_path: Path,
) -> None:
    d_r_dataset, d_v_dataset = _development_datasets(tmp_path / "data")
    adapter = ToyFrozenBaseAdapter()
    verified_base_identity = _verified_toy_base_identity(adapter, label="cache")
    source = tmp_path / "generic-base-cache"
    d_r_source = source / "d_r"
    d_v_source = source / "d_v"
    cache_manifest_split(adapter, d_r_dataset, "D_R", d_r_source)
    cache_manifest_split(adapter, d_v_dataset, "D_V", d_v_source)
    output = tmp_path / "stage-a-from-cache"

    with pytest.raises(TypeError, match="registered Base-run loader"):
        run_stage_a_from_base_caches(
            d_r_source / "index.json",
            d_v_source / "index.json",
            d_r_dataset,
            d_v_dataset,
            _stage_a_config(),
            tmp_path / "forged-identity",
            verified_base_identity=object(),  # type: ignore[arg-type]
        )
    assert not (tmp_path / "forged-identity").exists()

    wrong_state_adapter = ToyFrozenBaseAdapter()
    with torch.no_grad():
        next(wrong_state_adapter.base.parameters()).add_(1.0)
    wrong_state_identity = _verified_toy_base_identity(
        wrong_state_adapter,
        label="wrong-state",
    )
    with pytest.raises(RuntimeError, match="verified Base state differs"):
        run_stage_a_from_base_caches(
            d_r_source / "index.json",
            d_v_source / "index.json",
            d_r_dataset,
            d_v_dataset,
            _stage_a_config(),
            tmp_path / "wrong-state",
            verified_base_identity=wrong_state_identity,
        )
    assert not (tmp_path / "wrong-state").exists()

    completed = run_stage_a_from_base_caches(
        d_r_source / "index.json",
        d_v_source / "index.json",
        d_r_dataset,
        d_v_dataset,
        _stage_a_config(),
        output,
        verified_base_identity=verified_base_identity,
    )

    assert (output / "COMPLETE.json").is_file()
    assert file_sha256(output / "d_r" / "base_cache" / "index.json") == (
        file_sha256(d_r_source / "index.json")
    )
    assert file_sha256(output / "d_v" / "base_cache" / "index.json") == (
        file_sha256(d_v_source / "index.json")
    )
    completed.verify_unchanged()
    assert _UNUSED_D_T_ID not in _all_json_text(output)

    alternate_identity = _verified_toy_base_identity(adapter, label="alternate-run")
    with pytest.raises(RuntimeError, match="binds another verified Base run"):
        load_stage_a_run(
            output,
            d_r_dataset,
            d_v_dataset,
            verified_base_identity=alternate_identity,
        )

    for path in source.rglob("*"):
        if path.is_file():
            path.unlink()
    for path in sorted(source.rglob("*"), reverse=True):
        if path.is_dir():
            path.rmdir()
    source.rmdir()
    completed.verify_unchanged()
