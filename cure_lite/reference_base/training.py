"""Deterministic D_B-only training for the Stage-A reference detector."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from time import perf_counter
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..cache.schema import file_sha256, stable_fingerprint
from ..data import ManifestImageDataset
from ..splits import SplitManifest
from .config import ReferenceBaseTrainingConfig
from .model import ReferenceBaseNetwork
from .partition import DBPartition, build_d_b_partition


REFERENCE_BASE_SELECTION_SCHEMA = "cure-lite-reference-base-selection-v1"
REFERENCE_BASE_RUN_SCHEMA = "cure-lite-reference-base-run-v1"
REFERENCE_BASE_CHECKPOINT_SCHEMA = "cure-lite-reference-base-checkpoint-v1"
REFERENCE_BASE_FINGERPRINT_SCHEMA = "cure-lite-reference-base-fingerprint-v1"


@dataclass(frozen=True)
class LoadedReferenceBaseRun:
    root: Path
    config: ReferenceBaseTrainingConfig
    partition: DBPartition
    model: ReferenceBaseNetwork
    base_fingerprint: str
    best_epoch: int
    best_select_miou: float
    best_select_loss: float
    checkpoint_sha256: str
    manifest_fingerprint: str
    manifest_file_sha256: str


@dataclass(frozen=True)
class _TensorStore:
    sample_ids: tuple[str, ...]
    images: Tensor
    masks: Tensor

    def __post_init__(self) -> None:
        if not self.sample_ids:
            raise ValueError("reference-base tensor store must not be empty")
        if self.images.ndim != 4 or self.masks.ndim != 4:
            raise ValueError("reference-base tensors must be NCHW")
        if self.images.shape[0] != len(self.sample_ids):
            raise ValueError("image count differs from sample IDs")
        if self.masks.shape != (
            len(self.sample_ids),
            1,
            self.images.shape[-2],
            self.images.shape[-1],
        ):
            raise ValueError("mask and image shapes differ")
        if self.images.dtype != torch.float32 or self.masks.dtype != torch.float32:
            raise TypeError("reference-base stores must use float32")


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_new_json(path: Path, payload: object) -> None:
    encoded = _json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{name} contains non-finite number {value}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _source_digest() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256(b"cure-lite-reference-base-source-v1")
    paths = tuple(
        sorted(
            path
            for path in root.glob("*.py")
            if path.is_file() and not path.is_symlink()
        )
    )
    if not paths:
        raise RuntimeError("reference-base source package is empty")
    for path in paths:
        relative = path.name.encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _set_reproducible_state(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _resolve_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("reference-base device is invalid") from error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("reference-base training supports only CPU or CUDA")
    return device


def _load_store(
    dataset: ManifestImageDataset,
    sample_ids: tuple[str, ...],
) -> _TensorStore:
    if dataset.split != "D_B":
        raise ValueError("reference-base training may load only D_B")
    index_by_id = {
        record.sample_id: index for index, record in enumerate(dataset.records)
    }
    if not set(sample_ids) <= set(index_by_id):
        raise ValueError("D_B partition contains a sample absent from its dataset")
    images: list[Tensor] = []
    masks: list[Tensor] = []
    for sample_id in sample_ids:
        sample = dataset[index_by_id[sample_id]]
        if sample.sample_id != sample_id or sample.split != "D_B":
            raise RuntimeError("loaded D_B sample identity differs from the partition")
        images.append(sample.image.to(torch.float32))
        masks.append(sample.gt_mask.to(torch.float32))
    return _TensorStore(
        sample_ids=sample_ids,
        images=torch.stack(images).contiguous(),
        masks=torch.stack(masks).contiguous(),
    )


def _augmentation_code(seed: int, epoch: int, sample_id: str) -> int:
    fingerprint = stable_fingerprint(
        {
            "schema_version": "cure-lite-reference-base-augmentation-v1",
            "seed": seed,
            "epoch": epoch,
            "sample_id": sample_id,
        }
    )
    return int(fingerprint[:16], 16) % 8


def _augment_pair(image: Tensor, mask: Tensor, code: int) -> tuple[Tensor, Tensor]:
    rotations = code % 4
    if rotations:
        image = torch.rot90(image, rotations, dims=(-2, -1))
        mask = torch.rot90(mask, rotations, dims=(-2, -1))
    if code >= 4:
        image = torch.flip(image, dims=(-1,))
        mask = torch.flip(mask, dims=(-1,))
    return image.contiguous(), mask.contiguous()


def reference_base_loss(
    logits: Tensor,
    target: Tensor,
    config: ReferenceBaseTrainingConfig,
) -> dict[str, Tensor]:
    if logits.shape != target.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError("reference-base logits and target must be matching [B,1,H,W]")
    if logits.dtype != torch.float32 or target.dtype != torch.float32:
        raise TypeError("reference-base loss requires float32 tensors")
    positive_weight = torch.tensor(
        config.positive_weight,
        device=logits.device,
        dtype=logits.dtype,
    )
    bce = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=positive_weight,
    )
    probability = torch.sigmoid(logits)
    dimensions = (1, 2, 3)
    intersection = torch.sum(probability * target, dim=dimensions)
    union = torch.sum(probability + target - probability * target, dim=dimensions)
    soft_iou = torch.mean(1.0 - (intersection + 1.0) / (union + 1.0))
    total = config.bce_weight * bce + config.soft_iou_weight * soft_iou
    return {"total": total, "bce": bce, "soft_iou": soft_iou}


def _epoch_order(count: int, seed: int, epoch: int) -> tuple[int, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch * 1_000_003)
    return tuple(int(value) for value in torch.randperm(count, generator=generator))


def _train_epoch(
    model: ReferenceBaseNetwork,
    store: _TensorStore,
    optimizer: torch.optim.Optimizer,
    config: ReferenceBaseTrainingConfig,
    *,
    epoch: int,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    sums = {"total": 0.0, "bce": 0.0, "soft_iou": 0.0}
    seen = 0
    order = _epoch_order(len(store.sample_ids), config.training_seed, epoch)
    for start in range(0, len(order), config.batch_size):
        indices = order[start : start + config.batch_size]
        augmented_images: list[Tensor] = []
        augmented_masks: list[Tensor] = []
        for index in indices:
            image, mask = _augment_pair(
                store.images[index],
                store.masks[index],
                _augmentation_code(
                    config.training_seed,
                    epoch,
                    store.sample_ids[index],
                ),
            )
            augmented_images.append(image)
            augmented_masks.append(mask)
        images = torch.stack(augmented_images).to(device=device, dtype=torch.float32)
        targets = torch.stack(augmented_masks).to(device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        losses = reference_base_loss(logits, targets, config)
        losses["total"].backward()
        optimizer.step()
        batch = len(indices)
        seen += batch
        for name in sums:
            sums[name] += float(losses[name].detach().cpu()) * batch
    if seen != len(store.sample_ids):
        raise RuntimeError("reference-base epoch did not visit every D_B-fit sample")
    return {name: value / seen for name, value in sums.items()}


@torch.no_grad()
def _evaluate(
    model: ReferenceBaseNetwork,
    store: _TensorStore,
    config: ReferenceBaseTrainingConfig,
    *,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sums = {"total": 0.0, "bce": 0.0, "soft_iou": 0.0}
    intersection = 0
    union = 0
    seen = 0
    for start in range(0, len(store.sample_ids), config.batch_size):
        stop = min(start + config.batch_size, len(store.sample_ids))
        images = store.images[start:stop].to(device=device, dtype=torch.float32)
        targets = store.masks[start:stop].to(device=device, dtype=torch.float32)
        logits = model(images)
        losses = reference_base_loss(logits, targets, config)
        prediction = torch.sigmoid(logits) >= 0.5
        truth = targets.to(torch.bool)
        intersection += int(torch.count_nonzero(prediction & truth))
        union += int(torch.count_nonzero(prediction | truth))
        batch = stop - start
        seen += batch
        for name in sums:
            sums[name] += float(losses[name].detach().cpu()) * batch
    return {
        "loss": sums["total"] / seen,
        "bce": sums["bce"] / seen,
        "soft_iou_loss": sums["soft_iou"] / seen,
        "global_miou": intersection / union if union else 1.0,
    }


def _save_checkpoint(
    model: ReferenceBaseNetwork,
    path: Path,
    *,
    config_fingerprint: str,
) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("safetensors is required for reference-base weights") from error
    state = {
        name: tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
        for name, tensor in model.state_dict().items()
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        save_file(
            state,
            str(temporary),
            metadata={
                "schema_version": REFERENCE_BASE_CHECKPOINT_SCHEMA,
                "config_fingerprint": config_fingerprint,
            },
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_metric(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _partition_from_payload(payload: Mapping[str, object]) -> DBPartition:
    expected = {
        "schema_version",
        "manifest_fingerprint",
        "selection_fraction",
        "selection_seed",
        "fit_sample_ids",
        "select_sample_ids",
        "fit_group_ids",
        "select_group_ids",
        "partition_fingerprint",
    }
    if set(payload) != expected or payload["schema_version"] != "cure-lite-reference-base-d-b-partition-v1":
        raise ValueError("reference-base partition fields are not canonical")
    fingerprint_payload = dict(payload)
    fingerprint = fingerprint_payload.pop("partition_fingerprint")
    if stable_fingerprint(fingerprint_payload) != fingerprint:
        raise ValueError("reference-base partition fingerprint mismatch")
    list_fields = (
        "fit_sample_ids",
        "select_sample_ids",
        "fit_group_ids",
        "select_group_ids",
    )
    if any(
        not isinstance(payload[name], list)
        or any(not isinstance(item, str) or not item for item in payload[name])
        for name in list_fields
    ):
        raise ValueError("reference-base partition memberships are invalid")
    return DBPartition(
        fit_sample_ids=tuple(payload["fit_sample_ids"]),  # type: ignore[arg-type]
        select_sample_ids=tuple(payload["select_sample_ids"]),  # type: ignore[arg-type]
        fit_group_ids=tuple(payload["fit_group_ids"]),  # type: ignore[arg-type]
        select_group_ids=tuple(payload["select_group_ids"]),  # type: ignore[arg-type]
        manifest_fingerprint=payload["manifest_fingerprint"],  # type: ignore[arg-type]
        selection_fraction=payload["selection_fraction"],  # type: ignore[arg-type]
        selection_seed=payload["selection_seed"],  # type: ignore[arg-type]
        fingerprint=fingerprint,  # type: ignore[arg-type]
    )


def train_reference_base(
    manifest: SplitManifest,
    manifest_path: str | Path,
    config: ReferenceBaseTrainingConfig,
    output_dir: str | Path,
) -> LoadedReferenceBaseRun:
    """Train 800 epochs by default using only the D_B fit/select partition."""

    if not isinstance(manifest, SplitManifest):
        raise TypeError("manifest must be SplitManifest")
    if not isinstance(config, ReferenceBaseTrainingConfig):
        raise TypeError("config must be ReferenceBaseTrainingConfig")
    if manifest.dataset != config.dataset:
        raise ValueError("reference-base config dataset differs from the manifest")
    source_manifest = Path(manifest_path).expanduser().resolve(strict=True)
    if not source_manifest.is_file() or source_manifest.is_symlink():
        raise ValueError("manifest_path must be a regular file")
    manifest_file_digest = file_sha256(source_manifest)
    if manifest.fingerprint != SplitManifest.load(source_manifest).fingerprint:
        raise ValueError("manifest object differs from manifest_path")
    requested = Path(output_dir).expanduser()
    if requested.exists() or requested.is_symlink():
        raise FileExistsError(f"reference-base output already exists: {requested}")
    root = requested.resolve(strict=False)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)

    source_digest = _source_digest()
    config_payload = config.canonical_payload()
    config_fingerprint = stable_fingerprint(config_payload)
    config_record = {
        **config_payload,
        "config_fingerprint": config_fingerprint,
        "manifest_fingerprint": manifest.fingerprint,
        "manifest_file_sha256": manifest_file_digest,
        "reference_source_sha256": source_digest,
    }
    _write_new_json(root / "config.json", config_record)

    partition = build_d_b_partition(
        manifest,
        selection_fraction=config.selection_fraction,
        seed=config.selection_seed,
    )
    _write_new_json(root / "d_b_partition.json", partition.canonical_payload())
    dataset = ManifestImageDataset(
        manifest,
        "D_B",
        config.preprocess,
        manifest_path=source_manifest,
    )
    fit_store = _load_store(dataset, partition.fit_sample_ids)
    select_store = _load_store(dataset, partition.select_sample_ids)
    if file_sha256(source_manifest) != manifest_file_digest:
        raise RuntimeError("manifest file changed while D_B was loaded")

    _set_reproducible_state(config.training_seed)
    device = _resolve_device(config.device)
    model = ReferenceBaseNetwork(config.model).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=0.0,
    )
    metrics_path = root / "metrics.jsonl"
    metrics_path.open("x", encoding="utf-8").close()
    checkpoint_path = root / "model.safetensors"
    best_key: tuple[float, float, int] | None = None
    best_epoch = 0
    best_select: dict[str, float] | None = None

    for epoch in range(1, config.epochs + 1):
        started = perf_counter()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = _train_epoch(
            model,
            fit_store,
            optimizer,
            config,
            epoch=epoch,
            device=device,
        )
        select_metrics = _evaluate(
            model,
            select_store,
            config,
            device=device,
        )
        scheduler.step()
        candidate_key = (
            select_metrics["global_miou"],
            -select_metrics["loss"],
            -epoch,
        )
        selected = best_key is None or candidate_key > best_key
        if selected:
            best_key = candidate_key
            best_epoch = epoch
            best_select = dict(select_metrics)
            _save_checkpoint(
                model,
                checkpoint_path,
                config_fingerprint=config_fingerprint,
            )
        _append_metric(
            metrics_path,
            {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": train_metrics["total"],
                "train_bce": train_metrics["bce"],
                "train_soft_iou_loss": train_metrics["soft_iou"],
                "select_loss": select_metrics["loss"],
                "select_bce": select_metrics["bce"],
                "select_soft_iou_loss": select_metrics["soft_iou_loss"],
                "select_global_miou": select_metrics["global_miou"],
                "selected_checkpoint": selected,
                "elapsed_seconds": perf_counter() - started,
            },
        )

    if best_select is None or best_epoch < 1 or not checkpoint_path.is_file():
        raise RuntimeError("reference-base training did not select a checkpoint")
    if file_sha256(source_manifest) != manifest_file_digest:
        raise RuntimeError("manifest file changed during reference-base training")
    if _source_digest() != source_digest:
        raise RuntimeError("reference-base Python sources changed during training")
    checkpoint_digest = file_sha256(checkpoint_path)
    preprocessing_fingerprint = stable_fingerprint(
        config.preprocess.fingerprint_payload()
    )
    base_fingerprint_payload = {
        "schema_version": REFERENCE_BASE_FINGERPRINT_SCHEMA,
        "checkpoint_sha256": checkpoint_digest,
        "config_fingerprint": config_fingerprint,
        "partition_fingerprint": partition.fingerprint,
        "manifest_fingerprint": manifest.fingerprint,
        "manifest_file_sha256": manifest_file_digest,
        "preprocessing_fingerprint": preprocessing_fingerprint,
        "feature_selector": "encoder-quarter-resolution",
        "feature_channels": config.model.feature_channels,
    }
    base_fingerprint = stable_fingerprint(base_fingerprint_payload)
    selection_payload: dict[str, object] = {
        "schema_version": REFERENCE_BASE_SELECTION_SCHEMA,
        "best_epoch": best_epoch,
        "best_select_global_miou": best_select["global_miou"],
        "best_select_loss": best_select["loss"],
        "checkpoint_sha256": checkpoint_digest,
        "config_fingerprint": config_fingerprint,
        "partition_fingerprint": partition.fingerprint,
        "base_fingerprint_payload": base_fingerprint_payload,
        "base_fingerprint": base_fingerprint,
    }
    selection_payload["selection_fingerprint"] = stable_fingerprint(
        selection_payload
    )
    _write_new_json(root / "selection.json", selection_payload)

    artifact_hashes = {
        "config.json": file_sha256(root / "config.json"),
        "d_b_partition.json": file_sha256(root / "d_b_partition.json"),
        "metrics.jsonl": file_sha256(metrics_path),
        "model.safetensors": checkpoint_digest,
        "selection.json": file_sha256(root / "selection.json"),
    }
    complete_payload: dict[str, object] = {
        "schema_version": REFERENCE_BASE_RUN_SCHEMA,
        "dataset": manifest.dataset,
        "epochs_completed": config.epochs,
        "manifest_fingerprint": manifest.fingerprint,
        "manifest_file_sha256": manifest_file_digest,
        "config_fingerprint": config_fingerprint,
        "partition_fingerprint": partition.fingerprint,
        "reference_source_sha256": source_digest,
        "base_fingerprint": base_fingerprint,
        "artifacts": artifact_hashes,
    }
    complete_payload["run_fingerprint"] = stable_fingerprint(complete_payload)
    _write_new_json(root / "COMPLETE.json", complete_payload)
    return load_reference_base_run(root, device=device)


def load_reference_base_run(
    run_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedReferenceBaseRun:
    """Verify and load a completed project-owned reference-base run."""

    root = Path(run_dir).expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("reference-base run must be a regular directory")
    complete = _strict_json(root / "COMPLETE.json", name="reference-base COMPLETE")
    expected_complete = {
        "schema_version",
        "dataset",
        "epochs_completed",
        "manifest_fingerprint",
        "manifest_file_sha256",
        "config_fingerprint",
        "partition_fingerprint",
        "reference_source_sha256",
        "base_fingerprint",
        "artifacts",
        "run_fingerprint",
    }
    if set(complete) != expected_complete or complete["schema_version"] != REFERENCE_BASE_RUN_SCHEMA:
        raise ValueError("reference-base COMPLETE fields are not canonical")
    fingerprint_payload = dict(complete)
    run_fingerprint = fingerprint_payload.pop("run_fingerprint")
    if stable_fingerprint(fingerprint_payload) != run_fingerprint:
        raise ValueError("reference-base run fingerprint mismatch")
    artifacts = complete["artifacts"]
    expected_artifacts = {
        "config.json",
        "d_b_partition.json",
        "metrics.jsonl",
        "model.safetensors",
        "selection.json",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        raise ValueError("reference-base artifact table is not canonical")
    for relative, expected_digest in artifacts.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing reference-base artifact {relative}")
        if file_sha256(path) != expected_digest:
            raise ValueError(f"reference-base artifact hash mismatch: {relative}")

    config_record = _strict_json(root / "config.json", name="reference-base config")
    extra_config_fields = {
        "config_fingerprint",
        "manifest_fingerprint",
        "manifest_file_sha256",
        "reference_source_sha256",
    }
    config_payload = {
        key: value for key, value in config_record.items() if key not in extra_config_fields
    }
    if set(config_record) != set(config_payload) | extra_config_fields:
        raise ValueError("reference-base config record fields are invalid")
    config = ReferenceBaseTrainingConfig.from_mapping(config_payload)
    config_fingerprint = stable_fingerprint(config_payload)
    if config_record["config_fingerprint"] != config_fingerprint:
        raise ValueError("reference-base config fingerprint mismatch")
    partition_payload = _strict_json(
        root / "d_b_partition.json",
        name="reference-base D_B partition",
    )
    partition = _partition_from_payload(partition_payload)
    selection = _strict_json(root / "selection.json", name="reference-base selection")
    expected_selection = {
        "schema_version",
        "best_epoch",
        "best_select_global_miou",
        "best_select_loss",
        "checkpoint_sha256",
        "config_fingerprint",
        "partition_fingerprint",
        "base_fingerprint_payload",
        "base_fingerprint",
        "selection_fingerprint",
    }
    if set(selection) != expected_selection or selection["schema_version"] != REFERENCE_BASE_SELECTION_SCHEMA:
        raise ValueError("reference-base selection fields are not canonical")
    selection_fingerprint_payload = dict(selection)
    selection_fingerprint = selection_fingerprint_payload.pop("selection_fingerprint")
    if stable_fingerprint(selection_fingerprint_payload) != selection_fingerprint:
        raise ValueError("reference-base selection fingerprint mismatch")
    base_payload = selection["base_fingerprint_payload"]
    if not isinstance(base_payload, Mapping):
        raise ValueError("reference-base fingerprint payload is invalid")
    if stable_fingerprint(base_payload) != selection["base_fingerprint"]:
        raise ValueError("reference-base fingerprint differs from its payload")
    if (
        complete["dataset"] != config.dataset
        or complete["epochs_completed"] != config.epochs
        or complete["config_fingerprint"] != config_fingerprint
        or complete["partition_fingerprint"] != partition.fingerprint
        or complete["base_fingerprint"] != selection["base_fingerprint"]
        or selection["config_fingerprint"] != config_fingerprint
        or selection["partition_fingerprint"] != partition.fingerprint
        or selection["checkpoint_sha256"] != artifacts["model.safetensors"]
    ):
        raise ValueError("reference-base completion records disagree")

    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("safetensors is required for reference-base weights") from error
    checkpoint = root / "model.safetensors"
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata != {
        "schema_version": REFERENCE_BASE_CHECKPOINT_SCHEMA,
        "config_fingerprint": config_fingerprint,
    }:
        raise ValueError("reference-base checkpoint metadata mismatch")
    state = load_file(str(checkpoint), device="cpu")
    model = ReferenceBaseNetwork(config.model)
    expected_keys = set(model.state_dict())
    if set(state) != expected_keys:
        raise ValueError("reference-base checkpoint parameter keys mismatch")
    if any(
        tensor.dtype != torch.float32 or not torch.isfinite(tensor).all()
        for tensor in state.values()
    ):
        raise ValueError("reference-base checkpoint tensors are invalid")
    model.load_state_dict(state, strict=True)
    resolved_device = _resolve_device(str(device))
    model.to(device=resolved_device, dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    best_epoch = selection["best_epoch"]
    best_miou = selection["best_select_global_miou"]
    best_loss = selection["best_select_loss"]
    if (
        isinstance(best_epoch, bool)
        or not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= config.epochs
        or isinstance(best_miou, bool)
        or not isinstance(best_miou, (int, float))
        or not math.isfinite(float(best_miou))
        or not 0.0 <= float(best_miou) <= 1.0
        or isinstance(best_loss, bool)
        or not isinstance(best_loss, (int, float))
        or not math.isfinite(float(best_loss))
    ):
        raise ValueError("reference-base selected metrics are invalid")
    return LoadedReferenceBaseRun(
        root=root,
        config=config,
        partition=partition,
        model=model,
        base_fingerprint=selection["base_fingerprint"],  # type: ignore[arg-type]
        best_epoch=best_epoch,
        best_select_miou=float(best_miou),
        best_select_loss=float(best_loss),
        checkpoint_sha256=selection["checkpoint_sha256"],  # type: ignore[arg-type]
        manifest_fingerprint=complete["manifest_fingerprint"],  # type: ignore[arg-type]
        manifest_file_sha256=complete["manifest_file_sha256"],  # type: ignore[arg-type]
    )


__all__ = [
    "LoadedReferenceBaseRun",
    "REFERENCE_BASE_RUN_SCHEMA",
    "load_reference_base_run",
    "reference_base_loss",
    "train_reference_base",
]
