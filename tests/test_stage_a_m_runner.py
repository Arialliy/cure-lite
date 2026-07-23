from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
import pytest

import cure_lite.experiment.formal_training as formal_training_module
import cure_lite.experiment.stage_a_m_runner as m_runner_module
from cure_lite.base_identity import (
    VerifiedBaseRunIdentity,
    _bind_verified_base_run_identity,
)
from cure_lite.cache.schema import file_sha256
from cure_lite.calibration import FalseAlarmBudget
from cure_lite.config import DecoderConfig
from cure_lite.data import ManifestImageDataset, PreprocessConfig
from cure_lite.experiment.artifacts import DECODER_ARTIFACT_SCHEMA_V3
from cure_lite.experiment.formal_training import PairedGate2TrainingConfig
from cure_lite.experiment.stage_a_m_runner import (
    STAGE_A_M_METHOD_ORDER,
    run_stage_a_m_extension,
)
from cure_lite.experiment.stage_a_runner import (
    StageARunConfig,
    run_stage_a,
)
from cure_lite.frozen_base import frozen_base_state_fingerprint
from cure_lite.metrics import formal_stage_a_metrics_payload
from cure_lite.splits import SplitManifest, SplitRecord
from cure_lite.stage_a import BaseRunIdentity, STAGE_A_METHOD_ORDER
from cure_lite.toy import ToyFrozenBaseAdapter


def _test_digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _verified_toy_base_identity(
    adapter: ToyFrozenBaseAdapter,
) -> VerifiedBaseRunIdentity:
    identity = BaseRunIdentity(
        producer_schema="cure-lite-test-toy-base-v1",
        base_fingerprint=adapter.fingerprint,
        base_state_fingerprint=frozen_base_state_fingerprint(adapter),
        training_run_fingerprint=_test_digest("m-runner-training-run"),
        completion_receipt_sha256=_test_digest("m-runner-completion"),
        checkpoint_sha256=_test_digest("m-runner-checkpoint"),
        selection_fingerprint=_test_digest("m-runner-selection"),
        source_fingerprint=_test_digest("m-runner-source"),
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
    root: Path,
) -> tuple[ManifestImageDataset, ManifestImageDataset]:
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    for sample_id, values in {
        "db": (255, 255),
        "dr-covered": (255, 255),
        "dr-miss": (89, 255),
        "dv": (89, 255),
    }.items():
        _write_scene(root, sample_id, values)

    manifest = SplitManifest(
        dataset="formal-stage-a-m-toy",
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
            SplitRecord(
                "unused-dt",
                "D_T",
                "g-dt",
                "unused/dt-image.png",
                "unused/dt-mask.png",
            ),
        ),
    )
    manifest_path = root / "manifest.json"
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


def _stage_a_config() -> StageARunConfig:
    return StageARunConfig(
        training=PairedGate2TrainingConfig(
            decoder_config=DecoderConfig(feature_channels=3),
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


def _file_inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.fixture(scope="module")
def toy_reference(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[
    tuple[Path, ManifestImageDataset, ManifestImageDataset]
]:
    root = tmp_path_factory.mktemp("stage-a-m-runner")
    d_r_dataset, d_v_dataset = _development_datasets(root / "data")
    adapter = ToyFrozenBaseAdapter()
    reference = root / "reference"
    run_stage_a(
        adapter,
        d_r_dataset,
        d_v_dataset,
        _stage_a_config(),
        reference,
        verified_base_identity=_verified_toy_base_identity(adapter),
    )
    complete = _read_json(reference / "COMPLETE.json")
    assert complete["schema_version"] == "cure-lite-stage-a-run-v7"
    yield reference, d_r_dataset, d_v_dataset


def test_m_extension_reuses_v01_and_publishes_only_m(
    toy_reference: tuple[
        Path,
        ManifestImageDataset,
        ManifestImageDataset,
    ],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, d_r_dataset, d_v_dataset = toy_reference
    reference_before = _file_inventory(reference)
    historical_results_bytes = (
        reference / "receipts" / "results.json"
    ).read_bytes()
    historical_results = json.loads(historical_results_bytes)
    historical_calibration = _read_json(
        reference / "receipts" / "calibration.json"
    )
    historical_u_threshold = historical_calibration["methods"]["U"][
        "protocol"
    ]["selected_threshold"]

    trained_variants: list[str] = []
    original_train = formal_training_module.run_fixed_training

    def tracked_train(*args: object, **kwargs: object) -> object:
        variant = kwargs.get("variant")
        assert isinstance(variant, str)
        trained_variants.append(variant)
        return original_train(*args, **kwargs)

    selected_methods: list[tuple[str, tuple[float, ...]]] = []
    original_select = (
        m_runner_module.select_formal_residual_threshold_from_ledger
    )

    def tracked_select(
        run: object,
        thresholds: object,
        budget: object,
        **kwargs: object,
    ) -> object:
        method_label = kwargs.get("method_label")
        assert isinstance(method_label, str)
        frozen_thresholds = tuple(thresholds)  # type: ignore[arg-type]
        selected_methods.append((method_label, frozen_thresholds))
        return original_select(run, frozen_thresholds, budget, **kwargs)

    fixed_points: list[tuple[str, float | None, str]] = []
    original_fixed = m_runner_module.evaluate_formal_residual_fixed_point

    def tracked_fixed(
        run: object,
        threshold: float | None,
        *,
        method_label: str = "fixed",
    ) -> object:
        variant = run.artifact.config.variant  # type: ignore[union-attr]
        fixed_points.append((variant, threshold, method_label))
        return original_fixed(run, threshold, method_label=method_label)

    monkeypatch.setattr(
        formal_training_module,
        "run_fixed_training",
        tracked_train,
    )
    monkeypatch.setattr(
        m_runner_module,
        "select_formal_residual_threshold_from_ledger",
        tracked_select,
    )
    monkeypatch.setattr(
        m_runner_module,
        "evaluate_formal_residual_fixed_point",
        tracked_fixed,
    )

    output = tmp_path / "m-extension"
    published = run_stage_a_m_extension(
        reference,
        d_r_dataset,
        d_v_dataset,
        output,
        device="cpu",
    )

    assert trained_variants == ["miss_aligned_legal"]
    assert selected_methods == [("M", (0.5, 0.9))]
    assert fixed_points == [
        ("uniform_legal", historical_u_threshold, "U-historical-fixed")
    ]
    assert _file_inventory(reference) == reference_before
    assert (
        reference / "receipts" / "results.json"
    ).read_bytes() == historical_results_bytes

    results = _read_json(output / "receipts" / "results.json")
    assert results["method_order"] == list(STAGE_A_M_METHOD_ORDER)
    for method in STAGE_A_METHOD_ORDER:
        assert results["methods"][method] == historical_results["methods"][method]
        assert json.dumps(
            results["methods"][method],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") == json.dumps(
            historical_results["methods"][method],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    assert results["reference_results_fingerprint"] == (
        historical_results["results_fingerprint"]
    )
    assert formal_stage_a_metrics_payload(published.u_fixed_metrics) == (
        historical_results["methods"]["U"]
    )
    assert results["recovery_diagnostics"]["U@historical"][
        "recovered_anchor_misses"
    ] == published.u_fixed_metrics.recovered_anchor_misses

    assert published.m_artifact.config.schema_version == (
        DECODER_ARTIFACT_SCHEMA_V3
    )
    assert published.m_artifact.config.variant == "miss_aligned_legal"
    assert published.m_artifact.config.alignment_catalog_fingerprint == (
        published.alignment_catalog_fingerprint
    )
    alignment = _read_json(output / "receipts" / "alignment.json")
    assert alignment["catalog_fingerprint"] == (
        published.alignment_catalog_fingerprint
    )
    assert alignment["choices"]

    complete = _read_json(output / "COMPLETE.json")
    assert complete["status"] == "complete"
    assert complete["method_order"] == list(STAGE_A_M_METHOD_ORDER)
    assert complete["alignment_catalog_fingerprint"] == (
        published.alignment_catalog_fingerprint
    )
    assert complete["m_decoder_artifact_fingerprint"] == (
        published.m_artifact.artifact_fingerprint
    )
    assert not (output / ".incomplete").exists()
    directories, files = m_runner_module._tree_inventory(output)
    assert complete["artifact_directories"] == directories
    assert complete["artifact_files"] == files
    published.verify_unchanged()


def test_m_extension_rejects_destinations_before_any_work(
    toy_reference: tuple[
        Path,
        ManifestImageDataset,
        ManifestImageDataset,
    ],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, d_r_dataset, d_v_dataset = toy_reference
    reference_before = _file_inventory(reference)
    preflight_calls = 0

    def unexpected_preflight(device: str) -> None:
        nonlocal preflight_calls
        preflight_calls += 1
        raise AssertionError(f"unexpected device preflight for {device}")

    monkeypatch.setattr(
        m_runner_module,
        "_preflight_stage_a_device",
        unexpected_preflight,
    )

    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_stage_a_m_extension(
            reference,
            d_r_dataset,
            d_v_dataset,
            existing,
            device="cpu",
        )
    assert marker.read_text(encoding="utf-8") == "unchanged"

    nested = reference / "m-extension-must-not-exist"
    with pytest.raises(ValueError, match="outside the historical Stage-A tree"):
        run_stage_a_m_extension(
            reference,
            d_r_dataset,
            d_v_dataset,
            nested,
            device="cpu",
        )
    assert not nested.exists()
    assert preflight_calls == 0
    assert _file_inventory(reference) == reference_before


def test_m_extension_failure_keeps_create_only_incomplete_output(
    toy_reference: tuple[
        Path,
        ManifestImageDataset,
        ManifestImageDataset,
    ],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, d_r_dataset, d_v_dataset = toy_reference
    reference_before = _file_inventory(reference)

    def injected_failure(*args: object, **kwargs: object) -> object:
        raise RuntimeError("injected M training failure")

    monkeypatch.setattr(
        m_runner_module,
        "run_miss_aligned_gate2_extension",
        injected_failure,
    )
    output = tmp_path / "failed-m-extension"
    with pytest.raises(RuntimeError, match="injected M training failure"):
        run_stage_a_m_extension(
            reference,
            d_r_dataset,
            d_v_dataset,
            output,
            device="cpu",
        )

    assert output.is_dir()
    assert (output / ".incomplete").is_file()
    assert not (output / "COMPLETE.json").exists()
    assert (output / "receipts" / "config.json").is_file()
    assert (output / "receipts" / "reference.json").is_file()
    assert (output / "receipts" / "alignment.json").is_file()
    assert _file_inventory(reference) == reference_before


def test_late_publication_failure_keeps_incomplete_marker(
    toy_reference: tuple[
        Path,
        ManifestImageDataset,
        ManifestImageDataset,
    ],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, d_r_dataset, d_v_dataset = toy_reference
    reference_before = _file_inventory(reference)
    original_write = m_runner_module._write_new_json

    def fail_complete(path: Path, payload: object) -> None:
        if path.name == "COMPLETE.json":
            raise RuntimeError("injected COMPLETE publication failure")
        original_write(path, payload)

    monkeypatch.setattr(m_runner_module, "_write_new_json", fail_complete)
    output = tmp_path / "late-failed-m-extension"
    with pytest.raises(
        RuntimeError,
        match="injected COMPLETE publication failure",
    ):
        run_stage_a_m_extension(
            reference,
            d_r_dataset,
            d_v_dataset,
            output,
            device="cpu",
        )

    assert (output / ".incomplete").is_file()
    assert not (output / "COMPLETE.json").exists()
    assert (output / "receipts" / "results.json").is_file()
    assert (output / "receipts" / "gate.json").is_file()
    assert _file_inventory(reference) == reference_before
