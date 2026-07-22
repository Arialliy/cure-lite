"""One formal Stage-A execution and replay entry for CURE-Lite.

The runner closes the development-only experiment in one fixed order:

``D_V base cache -> decoder-free anchor -> D_R base/state cache -> paired F/U
training -> immutable decoder artifacts -> A/Base@B/F/U calibration/evaluation``.

Only exact :class:`~cure_lite.data.ManifestImageDataset` views for ``D_R`` and
``D_V`` are accepted.  There is intentionally no ``D_T`` argument or access
path.  A run is loadable only after the final ``COMPLETE.json`` receipt has
been atomically published; an interrupted directory is not a completed run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping

import torch

from ..cache.schema import file_sha256, stable_fingerprint
from ..calibration import FalseAlarmBudget
from ..config import (
    DecoderConfig,
    InterventionConfig,
    LossConfig,
    MatchConfig,
    TrainingConfig,
    config_to_dict,
)
from ..data import ManifestImageDataset
from ..frozen_base import FrozenBaseAdapter
from ..metrics import AggregateEvaluation
from .artifacts import LoadedDecoderArtifact, load_decoder_artifact
from .cache_pipeline import (
    LoadedDRCacheBundle,
    LoadedDVCacheBundle,
    cache_d_r_states,
    cache_manifest_split,
    load_base_cache_pair_contract,
    load_d_r_cache_bundle,
    load_d_v_cache_bundle,
    materialize_base_cache_bundle,
)
from .formal_anchor import (
    FrozenAnchorReceipt,
    LoadedDVBaseRun,
    build_loaded_d_v_base_run,
    evaluate_frozen_anchor,
    select_frozen_anchor,
)
from .formal_evaluation import (
    FormalDVThresholdReceipt,
    Gate2DVResults,
    LoadedDVMethodRun,
    PairedGate2Calibration,
    build_loaded_d_v_method_run,
    calibrate_paired_gate2,
    evaluate_paired_gate2,
)
from .formal_training import (
    PairedGate2TrainingConfig,
    run_paired_gate2_training,
    save_completed_decoder_run,
    summarize_gate2_training_support,
)
from .training_pipeline import TrainingSupportRequirements, TrainingSupportSummary


STAGE_A_RUN_SCHEMA = "cure-lite-stage-a-run-v2"
_CONFIG_SCHEMA = "cure-lite-stage-a-config-receipt-v1"
_ANCHOR_SCHEMA = "cure-lite-stage-a-anchor-receipt-v1"
_SUPPORT_SCHEMA = "cure-lite-stage-a-support-receipt-v1"
_CALIBRATION_SCHEMA = "cure-lite-stage-a-calibration-receipt-v1"
_RESULTS_SCHEMA = "cure-lite-stage-a-results-receipt-v1"
_METHOD_ORDER = ("A", "Base@B", "F", "U")
_INCOMPLETE_NAME = ".incomplete"
_COMPLETE_NAME = "COMPLETE.json"
_SOURCE_ROOT = Path(__file__).resolve().parents[1]
_NON_METHOD_SOURCE_ROOTS = {"adapters", "reference_base", "toy"}
_NON_METHOD_SOURCE_FILES = {"provenance.py"}


def _canonical_threshold_grid(
    values: Iterable[float], *, name: str
) -> tuple[float, ...]:
    resolved: list[float] = []
    try:
        iterator = iter(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of thresholds") from error
    for value in iterator:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} values must be real numbers, not bool")
        threshold = float(value)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{name} values must be finite and in [0,1]")
        resolved.append(threshold)
    grid = tuple(sorted(set(resolved)))
    if not grid:
        raise ValueError(f"{name} must not be empty")
    return grid


def _canonical_digest(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return normalized


def _budget_payload(budget: FalseAlarmBudget) -> dict[str, float | None]:
    def finite_or_none(value: float) -> float | None:
        return None if math.isinf(value) else float(value)

    return {
        "pixel_fa_budget": float(budget.pixel_fa_budget),
        "component_fa_per_mp_budget": finite_or_none(
            budget.component_fa_per_mp_budget
        ),
        "raw_background_fa_budget": finite_or_none(
            budget.raw_background_fa_budget
        ),
        "minimum_retention": float(budget.minimum_retention),
    }


def _budget_from_mapping(value: object) -> FalseAlarmBudget:
    if not isinstance(value, Mapping) or set(value) != {
        "pixel_fa_budget",
        "component_fa_per_mp_budget",
        "raw_background_fa_budget",
        "minimum_retention",
    }:
        raise ValueError("Stage-A budget fields are not canonical")

    if any(
        isinstance(value[name], bool)
        for name in (
            "pixel_fa_budget",
            "component_fa_per_mp_budget",
            "raw_background_fa_budget",
            "minimum_retention",
        )
    ):
        raise TypeError("Stage-A budget fields may not be bool")

    def number_or_infinity(item: object) -> float:
        return float("inf") if item is None else float(item)

    return FalseAlarmBudget(
        pixel_fa_budget=float(value["pixel_fa_budget"]),
        component_fa_per_mp_budget=number_or_infinity(
            value["component_fa_per_mp_budget"]
        ),
        raw_background_fa_budget=number_or_infinity(
            value["raw_background_fa_budget"]
        ),
        minimum_retention=float(value["minimum_retention"]),
    )


def _training_payload(config: PairedGate2TrainingConfig) -> dict[str, object]:
    return {
        "decoder_config": config_to_dict(config.decoder_config),
        "loss_config": config_to_dict(config.loss_config),
        "training_config": config_to_dict(config.training_config),
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "epochs": config.epochs,
        "steps_per_epoch": config.steps_per_epoch,
        "factual_miss_batch": config.factual_miss_batch,
        "factual_no_miss_batch": config.factual_no_miss_batch,
        "synthetic_batch": config.synthetic_batch,
        "global_seed": config.global_seed,
    }


def _training_from_mapping(value: object) -> PairedGate2TrainingConfig:
    expected = {
        "decoder_config",
        "loss_config",
        "training_config",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "epochs",
        "steps_per_epoch",
        "factual_miss_batch",
        "factual_no_miss_batch",
        "synthetic_batch",
        "global_seed",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Stage-A training fields are not canonical")
    decoder = value["decoder_config"]
    loss = value["loss_config"]
    training = value["training_config"]
    if not all(isinstance(item, Mapping) for item in (decoder, loss, training)):
        raise TypeError("Stage-A nested training configs must be mappings")
    result = PairedGate2TrainingConfig(
        decoder_config=DecoderConfig(**dict(decoder)),
        loss_config=LossConfig(**dict(loss)),
        training_config=TrainingConfig(**dict(training)),
        optimizer=value["optimizer"],  # type: ignore[arg-type]
        learning_rate=value["learning_rate"],  # type: ignore[arg-type]
        weight_decay=value["weight_decay"],  # type: ignore[arg-type]
        epochs=value["epochs"],  # type: ignore[arg-type]
        steps_per_epoch=value["steps_per_epoch"],  # type: ignore[arg-type]
        factual_miss_batch=value["factual_miss_batch"],  # type: ignore[arg-type]
        factual_no_miss_batch=value["factual_no_miss_batch"],  # type: ignore[arg-type]
        synthetic_batch=value["synthetic_batch"],  # type: ignore[arg-type]
        global_seed=value["global_seed"],  # type: ignore[arg-type]
    )
    if _training_payload(result) != dict(value):
        raise ValueError("Stage-A training payload is not canonical")
    return result


def _support_requirements_from_mapping(
    value: object,
) -> TrainingSupportRequirements:
    expected = {
        "minimum_factual_miss_images",
        "minimum_factual_no_miss_images",
        "minimum_synthetic_images",
        "minimum_reachable_miss_targets",
        "minimum_visible_legal_candidates",
        "minimum_visible_legal_fraction",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("Stage-A support requirement fields are not canonical")
    result = TrainingSupportRequirements(**dict(value))
    if result.canonical_payload() != dict(value):
        raise ValueError("Stage-A support requirements are not canonical")
    return result


@dataclass(frozen=True)
class StageARunConfig:
    """Every free choice permitted by the formal Stage-A runner."""

    training: PairedGate2TrainingConfig
    anchor_thresholds: tuple[float, ...]
    base_thresholds: tuple[float, ...]
    residual_thresholds: tuple[float, ...]
    budget: FalseAlarmBudget
    support_requirements: TrainingSupportRequirements = TrainingSupportRequirements()
    match_config: MatchConfig = MatchConfig()
    intervention_config: InterventionConfig = InterventionConfig()
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.training, PairedGate2TrainingConfig):
            raise TypeError("training must be PairedGate2TrainingConfig")
        if not isinstance(self.budget, FalseAlarmBudget):
            raise TypeError("budget must be FalseAlarmBudget")
        if not isinstance(self.support_requirements, TrainingSupportRequirements):
            raise TypeError("support_requirements must be TrainingSupportRequirements")
        if not isinstance(self.match_config, MatchConfig):
            raise TypeError("match_config must be MatchConfig")
        if not isinstance(self.intervention_config, InterventionConfig):
            raise TypeError("intervention_config must be InterventionConfig")
        for name in (
            "anchor_thresholds",
            "base_thresholds",
            "residual_thresholds",
        ):
            object.__setattr__(
                self,
                name,
                _canonical_threshold_grid(getattr(self, name), name=name),
            )
        try:
            device = torch.device(self.device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("device is not a valid torch device") from error
        if device.type == "meta":
            raise ValueError("Stage-A cannot execute on the meta device")
        object.__setattr__(self, "device", str(device))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": STAGE_A_RUN_SCHEMA,
            "training": _training_payload(self.training),
            "anchor_thresholds": list(self.anchor_thresholds),
            "base_thresholds": list(self.base_thresholds),
            "residual_thresholds": list(self.residual_thresholds),
            "budget": _budget_payload(self.budget),
            "support_requirements": self.support_requirements.canonical_payload(),
            "match_config": config_to_dict(self.match_config),
            "intervention_config": config_to_dict(self.intervention_config),
            "device": self.device,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "StageARunConfig":
        expected = {
            "schema_version",
            "training",
            "anchor_thresholds",
            "base_thresholds",
            "residual_thresholds",
            "budget",
            "support_requirements",
            "match_config",
            "intervention_config",
            "device",
        }
        if set(value) != expected or value["schema_version"] != STAGE_A_RUN_SCHEMA:
            raise ValueError("Stage-A run config fields or schema are not canonical")
        match = value["match_config"]
        intervention = value["intervention_config"]
        if not isinstance(match, Mapping) or not isinstance(intervention, Mapping):
            raise TypeError("Stage-A mechanism configs must be mappings")
        result = cls(
            training=_training_from_mapping(value["training"]),
            anchor_thresholds=value["anchor_thresholds"],  # type: ignore[arg-type]
            base_thresholds=value["base_thresholds"],  # type: ignore[arg-type]
            residual_thresholds=value["residual_thresholds"],  # type: ignore[arg-type]
            budget=_budget_from_mapping(value["budget"]),
            support_requirements=_support_requirements_from_mapping(
                value["support_requirements"]
            ),
            match_config=MatchConfig(**dict(match)),
            intervention_config=InterventionConfig(**dict(intervention)),
            device=value["device"],  # type: ignore[arg-type]
        )
        if result.canonical_payload() != dict(value):
            raise ValueError("Stage-A run config payload is not canonical")
        return result


def _source_tree_digest() -> str:
    """Hash the model-independent method sources used by Stage-A."""

    digest = hashlib.sha256(b"cure-lite-python-source-tree-v1")
    paths = tuple(
        sorted(
            path
            for path in _SOURCE_ROOT.rglob("*.py")
            if "__pycache__" not in path.parts
            and path.relative_to(_SOURCE_ROOT).parts[0]
            not in _NON_METHOD_SOURCE_ROOTS
            and path.relative_to(_SOURCE_ROOT).as_posix()
            not in _NON_METHOD_SOURCE_FILES
        )
    )
    if not paths:
        raise RuntimeError("CURE-Lite Python source tree is empty")
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("CURE-Lite Python sources must be regular files")
        relative = path.relative_to(_SOURCE_ROOT).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_new_json(path: Path, payload: object) -> None:
    """Atomically create one JSON file without any replacement path."""

    encoded = _json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite Stage-A receipt {path}") from error
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
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

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_nonfinite,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read strict {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _config_receipt(config: StageARunConfig, source_digest: str) -> dict[str, object]:
    run_config = config.canonical_payload()
    return {
        "schema_version": _CONFIG_SCHEMA,
        "method": "CURE-Lite",
        "stage": "Stage-A",
        "runtime_splits": ["D_R", "D_V"],
        "unused_split": "D_T",
        "source_tree_digest": _canonical_digest(
            source_digest, name="source_tree_digest"
        ),
        "run_config": run_config,
        "run_config_fingerprint": stable_fingerprint(run_config),
    }


def _metrics_payload(metrics: AggregateEvaluation) -> dict[str, object]:
    if not isinstance(metrics, AggregateEvaluation):
        raise TypeError("Stage-A metrics must be AggregateEvaluation")
    return asdict(metrics)


def _protocol_payload(receipt: FormalDVThresholdReceipt) -> dict[str, object]:
    protocol = receipt.protocol
    return {
        "schema_version": "cure-lite-stage-a-formal-threshold-receipt-v1",
        "mode": receipt.mode,
        "protocol": {
            "variant": protocol.variant,
            "manifest_fingerprint": protocol.manifest_fingerprint,
            "ordered_d_v_sample_ids": list(protocol.ordered_d_v_sample_ids),
            "sample_tensor_fingerprint": protocol.sample_tensor_fingerprint,
            "candidate_threshold_grid": list(protocol.candidate_threshold_grid),
            "occupancy_config": config_to_dict(protocol.occupancy_config),
            "match_config": config_to_dict(protocol.match_config),
            "budget": _budget_payload(protocol.budget),
            "selected_threshold": protocol.selected_threshold,
            "selected_metrics": _metrics_payload(protocol.selected_metrics),
            "receipt_fingerprint": protocol.receipt_fingerprint,
        },
        "d_v_run_fingerprint": receipt.d_v_run_fingerprint,
        "manifest_file_sha256": receipt.manifest_file_sha256,
        "base_index_fingerprint": receipt.base_index_fingerprint,
        "base_index_sha256": receipt.base_index_sha256,
        "d_v_image_fingerprint": receipt.d_v_image_fingerprint,
        "d_v_gt_fingerprint": receipt.d_v_gt_fingerprint,
        "preprocessing_fingerprint": receipt.preprocessing_fingerprint,
        "base_fingerprint": receipt.base_fingerprint,
        "decoder_artifact_fingerprint": receipt.decoder_artifact_fingerprint,
        "decoder_receipt_sha256": receipt.decoder_receipt_sha256,
        "decoder_state_fingerprint": receipt.decoder_state_fingerprint,
        "decoder_variant": receipt.decoder_variant,
        "global_seed": receipt.global_seed,
        "receipt_fingerprint": receipt.receipt_fingerprint,
    }


def _anchor_receipt_payload(anchor: FrozenAnchorReceipt) -> dict[str, object]:
    return {
        "schema_version": _ANCHOR_SCHEMA,
        "receipt": anchor.canonical_payload(),
        "receipt_fingerprint": anchor.receipt_fingerprint,
    }


def _require_anchor_within_budget(
    metrics: AggregateEvaluation,
    budget: FalseAlarmBudget,
) -> None:
    if not budget.accepts(metrics):
        raise ValueError(
            "fixed D_V anchor is outside the Stage-A total-FA/retention "
            "constraints before decoder training: "
            f"pixel_fa={metrics.pixel_fa}, "
            f"raw_background_fa={metrics.raw_background_fa}, "
            f"fp_components_per_mp={metrics.fp_components_per_mp}, "
            f"retention={metrics.retention}"
        )


def _support_receipt_payload(
    summary: TrainingSupportSummary,
    requirements: TrainingSupportRequirements,
) -> dict[str, object]:
    requirements.require(summary)
    core: dict[str, object] = {
        "schema_version": _SUPPORT_SCHEMA,
        "split": "D_R",
        "summary": summary.canonical_payload(),
        "requirements": requirements.canonical_payload(),
        "requirements_met": True,
    }
    return {**core, "support_fingerprint": stable_fingerprint(core)}


def _calibration_receipt_payload(
    calibration: PairedGate2Calibration,
) -> dict[str, object]:
    return {
        "schema_version": _CALIBRATION_SCHEMA,
        "method_order": list(_METHOD_ORDER),
        "methods": {
            "A": calibration.anchor.canonical_payload(),
            "Base@B": _protocol_payload(calibration.base_at_budget),
            "F": _protocol_payload(calibration.factual_only),
            "U": _protocol_payload(calibration.uniform_legal),
        },
        "common_training_fingerprint": calibration.common_training_fingerprint,
        "receipt_fingerprint": calibration.receipt_fingerprint,
    }


def _results_receipt_payload(
    results: Gate2DVResults,
    calibration: PairedGate2Calibration,
) -> dict[str, object]:
    methods = {
        "A": _metrics_payload(results.anchor),
        "Base@B": _metrics_payload(results.base_at_budget),
        "F": _metrics_payload(results.factual_only),
        "U": _metrics_payload(results.uniform_legal),
    }
    core: dict[str, object] = {
        "schema_version": _RESULTS_SCHEMA,
        "method_order": list(_METHOD_ORDER),
        "methods": methods,
        "calibration_receipt_fingerprint": calibration.receipt_fingerprint,
    }
    return {**core, "results_fingerprint": stable_fingerprint(core)}


@dataclass(frozen=True)
class _StageAState:
    config: StageARunConfig
    d_r_bundle: LoadedDRCacheBundle
    d_v_bundle: LoadedDVCacheBundle
    d_v_base_run: LoadedDVBaseRun
    factual_artifact: LoadedDecoderArtifact
    uniform_artifact: LoadedDecoderArtifact
    factual_d_v_run: LoadedDVMethodRun
    uniform_d_v_run: LoadedDVMethodRun
    anchor: FrozenAnchorReceipt
    support_summary: TrainingSupportSummary
    calibration: PairedGate2Calibration
    results: Gate2DVResults


def _tree_inventory(root: Path) -> tuple[list[str], dict[str, str]]:
    directories: list[str] = []
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == _COMPLETE_NAME:
            continue
        if path.is_symlink():
            raise ValueError(f"Stage-A artifact tree contains symlink {relative!r}")
        if path.is_dir():
            directories.append(relative)
        elif path.is_file():
            files[relative] = file_sha256(path)
        else:
            raise ValueError(f"Stage-A artifact tree contains special file {relative!r}")
    return directories, files


def _complete_receipt(
    state: _StageAState,
    *,
    source_digest: str,
    artifact_directories: list[str],
    artifact_files: dict[str, str],
) -> dict[str, object]:
    config_payload = state.config.canonical_payload()
    results_payload = _results_receipt_payload(state.results, state.calibration)
    core: dict[str, object] = {
        "schema_version": STAGE_A_RUN_SCHEMA,
        "status": "complete",
        "method": "CURE-Lite",
        "stage": "Stage-A",
        "method_order": list(_METHOD_ORDER),
        "runtime_splits": ["D_R", "D_V"],
        "unused_split": "D_T",
        "source_tree_digest": source_digest,
        "run_config_fingerprint": stable_fingerprint(config_payload),
        "anchor_receipt_fingerprint": state.anchor.receipt_fingerprint,
        "support_receipt_fingerprint": _support_receipt_payload(
            state.support_summary,
            state.config.support_requirements,
        )["support_fingerprint"],
        "calibration_receipt_fingerprint": state.calibration.receipt_fingerprint,
        "results_fingerprint": results_payload["results_fingerprint"],
        "dataset": state.d_v_base_run.access.manifest.dataset,
        "manifest_fingerprint": state.d_v_bundle.split_manifest_fingerprint,
        "manifest_file_sha256": state.d_v_bundle.split_manifest_file_sha256,
        "preprocessing_fingerprint": state.d_v_bundle.preprocessing_fingerprint,
        "base_fingerprint": state.d_v_bundle.base_fingerprint,
        "d_r_base_index_fingerprint": state.d_r_bundle.base_index_fingerprint,
        "d_r_state_index_fingerprint": state.d_r_bundle.state_index_fingerprint,
        "d_v_base_index_fingerprint": state.d_v_bundle.base_index_fingerprint,
        "factual_decoder_artifact_fingerprint": (
            state.factual_artifact.artifact_fingerprint
        ),
        "uniform_decoder_artifact_fingerprint": (
            state.uniform_artifact.artifact_fingerprint
        ),
        "artifact_directories": artifact_directories,
        "artifact_files": artifact_files,
    }
    return {**core, "complete_fingerprint": stable_fingerprint(core)}


def _check_dataset_pair(
    d_r_dataset: ManifestImageDataset,
    d_v_dataset: ManifestImageDataset,
) -> None:
    if not isinstance(d_r_dataset, ManifestImageDataset):
        raise TypeError("d_r_dataset must be ManifestImageDataset")
    if not isinstance(d_v_dataset, ManifestImageDataset):
        raise TypeError("d_v_dataset must be ManifestImageDataset")
    if d_r_dataset.split != "D_R" or d_v_dataset.split != "D_V":
        raise ValueError("Stage-A requires exact D_R and D_V dataset views")
    if d_r_dataset.transform is not None or d_v_dataset.transform is not None:
        raise ValueError("Stage-A forbids unbound dataset transforms")
    if d_r_dataset.manifest.fingerprint != d_v_dataset.manifest.fingerprint:
        raise ValueError("D_R and D_V must come from one split manifest")
    if d_r_dataset.manifest.dataset != d_v_dataset.manifest.dataset:
        raise ValueError("D_R and D_V dataset identities differ")
    if d_r_dataset.preprocess != d_v_dataset.preprocess:
        raise ValueError("D_R and D_V preprocessing contracts differ")
    manifest_paths: list[Path] = []
    for name, dataset in (("D_R", d_r_dataset), ("D_V", d_v_dataset)):
        path = dataset.manifest_path
        if not isinstance(path, Path):
            raise ValueError(f"{name} requires an explicit manifest_path")
        if path.is_symlink():
            raise ValueError(f"{name} manifest_path may not be a symlink")
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"{name} manifest_path is unavailable") from error
        if resolved != path or not resolved.is_file():
            raise ValueError(f"{name} manifest_path must be one exact regular file")
        manifest_paths.append(resolved)
    if manifest_paths[0] != manifest_paths[1]:
        raise ValueError("D_R and D_V must bind the same manifest file")
    if file_sha256(manifest_paths[0]) != file_sha256(manifest_paths[1]):
        raise ValueError("D_R and D_V manifest file bytes differ")


def _verify_artifact_training_binding(
    artifact: LoadedDecoderArtifact,
    *,
    expected_variant: str,
    bundle: LoadedDRCacheBundle,
    config: StageARunConfig,
) -> None:
    artifact.verify_unchanged()
    run = artifact.config
    if run.variant != expected_variant:
        raise RuntimeError(f"{expected_variant} decoder artifact has wrong variant")
    bundle_bindings = {
        "manifest_fingerprint": bundle.split_manifest_fingerprint,
        "manifest_file_sha256": bundle.split_manifest_file_sha256,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
        "base_fingerprint": bundle.base_fingerprint,
        "state_fingerprint": bundle.state_fingerprint,
        "gt_fingerprint": bundle.gt_fingerprint,
        "base_index_fingerprint": bundle.base_index_fingerprint,
        "base_index_sha256": bundle.base_index_sha256,
        "state_index_fingerprint": bundle.state_index_fingerprint,
        "state_index_sha256": bundle.state_index_sha256,
    }
    if any(getattr(run, name) != expected for name, expected in bundle_bindings.items()):
        raise RuntimeError("decoder artifact differs from the verified D_R cache bundle")
    if (
        run.occupancy_config != bundle.occupancy_config
        or run.match_config != bundle.match_config
        or run.intervention_config != bundle.intervention_config
    ):
        raise RuntimeError("decoder artifact mechanism configs differ from D_R")
    training = config.training
    expected_training = {
        "global_seed": training.global_seed,
        "trained_epochs": training.epochs,
        "steps_per_epoch": training.steps_per_epoch,
        "decoder_config": training.decoder_config,
        "loss_config": training.loss_config,
        "training_config": training.training_config,
        "optimizer": training.optimizer,
        "learning_rate": training.learning_rate,
        "weight_decay": training.weight_decay,
        "factual_miss_batch": training.factual_miss_batch,
        "factual_no_miss_batch": training.factual_no_miss_batch,
        "synthetic_batch": training.synthetic_batch,
    }
    if any(getattr(run, name) != expected for name, expected in expected_training.items()):
        raise RuntimeError("decoder artifact differs from Stage-A training config")


def _build_downstream_state(
    *,
    config: StageARunConfig,
    d_r_bundle: LoadedDRCacheBundle,
    d_v_bundle: LoadedDVCacheBundle,
    factual_artifact: LoadedDecoderArtifact,
    uniform_artifact: LoadedDecoderArtifact,
) -> _StageAState:
    d_v_base_run = build_loaded_d_v_base_run(d_v_bundle)
    anchor = select_frozen_anchor(
        d_v_base_run,
        config.anchor_thresholds,
        config.match_config,
    )
    anchor_metrics = evaluate_frozen_anchor(d_v_base_run, anchor)
    _require_anchor_within_budget(anchor_metrics, config.budget)
    if d_r_bundle.occupancy_config != anchor.occupancy_config:
        raise RuntimeError("D_R state cache does not use the frozen D_V anchor")
    if d_r_bundle.match_config != config.match_config:
        raise RuntimeError("D_R state cache matching config differs from Stage-A")
    if d_r_bundle.intervention_config != config.intervention_config:
        raise RuntimeError("D_R intervention config differs from Stage-A")
    support_summary = summarize_gate2_training_support(d_r_bundle)
    config.support_requirements.require(support_summary)
    _verify_artifact_training_binding(
        factual_artifact,
        expected_variant="factual_only",
        bundle=d_r_bundle,
        config=config,
    )
    _verify_artifact_training_binding(
        uniform_artifact,
        expected_variant="uniform_legal",
        bundle=d_r_bundle,
        config=config,
    )
    if (
        factual_artifact.config.initial_decoder_fingerprint
        != uniform_artifact.config.initial_decoder_fingerprint
    ):
        raise RuntimeError("paired decoder artifacts do not share initialization")
    factual_run = build_loaded_d_v_method_run(d_v_bundle, factual_artifact)
    uniform_run = build_loaded_d_v_method_run(d_v_bundle, uniform_artifact)
    calibration = calibrate_paired_gate2(
        factual_run,
        uniform_run,
        anchor=anchor,
        residual_thresholds=config.residual_thresholds,
        base_thresholds=config.base_thresholds,
        budget=config.budget,
    )
    results = evaluate_paired_gate2(factual_run, uniform_run, calibration)
    return _StageAState(
        config=config,
        d_r_bundle=d_r_bundle,
        d_v_bundle=d_v_bundle,
        d_v_base_run=d_v_base_run,
        factual_artifact=factual_artifact,
        uniform_artifact=uniform_artifact,
        factual_d_v_run=factual_run,
        uniform_d_v_run=uniform_run,
        anchor=anchor,
        support_summary=support_summary,
        calibration=calibration,
        results=results,
    )


def _receipt_paths(root: Path) -> dict[str, Path]:
    receipts = root / "receipts"
    return {
        "config": receipts / "config.json",
        "anchor": receipts / "anchor.json",
        "support": receipts / "support.json",
        "calibration": receipts / "calibration.json",
        "results": receipts / "results.json",
    }


def run_stage_a(
    adapter: FrozenBaseAdapter,
    d_r_dataset: ManifestImageDataset,
    d_v_dataset: ManifestImageDataset,
    config: StageARunConfig,
    output_dir: str | Path,
) -> "LoadedStageARun":
    """Execute and publish one complete development-only CURE-Lite Stage-A run."""

    if not isinstance(adapter, FrozenBaseAdapter):
        raise TypeError("adapter must be FrozenBaseAdapter")
    if not isinstance(config, StageARunConfig):
        raise TypeError("config must be StageARunConfig")
    _check_dataset_pair(d_r_dataset, d_v_dataset)
    if adapter.feature_channels != config.training.decoder_config.feature_channels:
        raise ValueError("adapter feature channels differ from decoder config")
    base_fingerprint = _canonical_digest(
        adapter.fingerprint, name="adapter.fingerprint"
    )
    source_digest = _source_tree_digest()
    requested = Path(output_dir).expanduser()
    if requested.is_symlink() or requested.exists():
        raise FileExistsError(f"refusing to overwrite Stage-A run {requested}")
    root = requested.resolve(strict=False)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    incomplete = root / _INCOMPLETE_NAME
    incomplete.open("xb").close()

    paths = _receipt_paths(root)
    _write_new_json(paths["config"], _config_receipt(config, source_digest))

    # Anchor selection is deliberately decoder-free and precedes D_R state
    # construction, so the selected tau_o is the one cached and trained.
    d_v_cache_root = root / "d_v" / "base_cache"
    cache_manifest_split(adapter, d_v_dataset, "D_V", d_v_cache_root)
    d_v_bundle = load_d_v_cache_bundle(
        d_v_cache_root / "index.json",
        d_v_dataset,
        expected_base_fingerprint=base_fingerprint,
    )
    d_v_base_run = build_loaded_d_v_base_run(d_v_bundle)
    anchor = select_frozen_anchor(
        d_v_base_run,
        config.anchor_thresholds,
        config.match_config,
    )
    anchor_metrics = evaluate_frozen_anchor(d_v_base_run, anchor)
    _require_anchor_within_budget(anchor_metrics, config.budget)
    _write_new_json(paths["anchor"], _anchor_receipt_payload(anchor))

    d_r_cache_root = root / "d_r" / "base_cache"
    cache_manifest_split(adapter, d_r_dataset, "D_R", d_r_cache_root)
    d_r_state_root = root / "d_r" / "state_cache"
    cache_d_r_states(
        d_r_cache_root / "index.json",
        d_r_dataset,
        d_r_state_root,
        expected_base_fingerprint=base_fingerprint,
        occupancy_config=anchor.occupancy_config,
        match_config=config.match_config,
        intervention_config=config.intervention_config,
    )
    d_r_bundle = load_d_r_cache_bundle(
        d_r_state_root / "index.json",
        d_r_dataset,
        expected_base_fingerprint=base_fingerprint,
    )
    support_summary = summarize_gate2_training_support(d_r_bundle)
    config.support_requirements.require(support_summary)
    _write_new_json(
        paths["support"],
        _support_receipt_payload(
            support_summary,
            config.support_requirements,
        ),
    )
    paired = run_paired_gate2_training(
        d_r_bundle,
        config.training,
        device=config.device,
    )
    factual_directory = root / "decoders" / "factual_only"
    uniform_directory = root / "decoders" / "uniform_legal"
    save_completed_decoder_run(factual_directory, paired.factual_only)
    save_completed_decoder_run(uniform_directory, paired.uniform_legal)
    factual_artifact = load_decoder_artifact(
        factual_directory, expected_config=paired.factual_only.config
    )
    uniform_artifact = load_decoder_artifact(
        uniform_directory, expected_config=paired.uniform_legal.config
    )
    state = _build_downstream_state(
        config=config,
        d_r_bundle=d_r_bundle,
        d_v_bundle=d_v_bundle,
        factual_artifact=factual_artifact,
        uniform_artifact=uniform_artifact,
    )
    if state.anchor.canonical_payload() != anchor.canonical_payload():
        raise RuntimeError("frozen anchor changed before Stage-A publication")
    _write_new_json(
        paths["calibration"], _calibration_receipt_payload(state.calibration)
    )
    _write_new_json(paths["results"], _results_receipt_payload(
        state.results, state.calibration
    ))
    if adapter.fingerprint != base_fingerprint:
        raise RuntimeError("adapter fingerprint changed during Stage-A")
    if _source_tree_digest() != source_digest:
        raise RuntimeError("CURE-Lite Python sources changed during Stage-A")

    incomplete.unlink()
    directories, files = _tree_inventory(root)
    complete = _complete_receipt(
        state,
        source_digest=source_digest,
        artifact_directories=directories,
        artifact_files=files,
    )
    _write_new_json(root / _COMPLETE_NAME, complete)
    return load_stage_a_run(
        root,
        d_r_dataset,
        d_v_dataset,
        expected_base_fingerprint=base_fingerprint,
    )


def run_stage_a_from_base_caches(
    d_r_base_index: str | Path,
    d_v_base_index: str | Path,
    d_r_dataset: ManifestImageDataset,
    d_v_dataset: ManifestImageDataset,
    config: StageARunConfig,
    output_dir: str | Path,
) -> "LoadedStageARun":
    """Execute Stage-A from the generic probability/feature cache contract.

    The base producer is deliberately outside this API.  The two input bundles
    are fully checked and copied into the new run before anchor selection,
    state construction, decoder training, calibration, or evaluation begins.
    """

    if not isinstance(config, StageARunConfig):
        raise TypeError("config must be StageARunConfig")
    _check_dataset_pair(d_r_dataset, d_v_dataset)
    contract = load_base_cache_pair_contract(
        d_r_base_index,
        d_v_base_index,
    )
    manifest = d_r_dataset.manifest
    manifest_paths = (
        Path(d_r_dataset.manifest_path).resolve(strict=True),
        Path(d_v_dataset.manifest_path).resolve(strict=True),
    )
    expected_memberships = {
        "D_R": tuple(
            sorted(record.sample_id for record in manifest.records_for("D_R"))
        ),
        "D_V": tuple(
            sorted(record.sample_id for record in manifest.records_for("D_V"))
        ),
    }
    if (
        contract.dataset != manifest.dataset
        or contract.split_manifest_fingerprint != manifest.fingerprint
        or contract.split_manifest_file_sha256 != file_sha256(manifest_paths[0])
        or contract.d_r_sample_ids != expected_memberships["D_R"]
        or contract.d_v_sample_ids != expected_memberships["D_V"]
    ):
        raise ValueError(
            "generic base caches do not match the exact D_R/D_V manifest views"
        )
    if (
        d_r_dataset.preprocess != contract.preprocessing
        or d_v_dataset.preprocess != contract.preprocessing
    ):
        raise ValueError("datasets do not use the base-cache preprocessing")
    if (
        contract.feature_channels
        != config.training.decoder_config.feature_channels
    ):
        raise ValueError("base-cache feature channels differ from decoder config")

    base_fingerprint = _canonical_digest(
        contract.base_fingerprint,
        name="base cache fingerprint",
    )
    source_digest = _source_tree_digest()
    requested = Path(output_dir).expanduser()
    if requested.is_symlink() or requested.exists():
        raise FileExistsError(f"refusing to overwrite Stage-A run {requested}")
    root = requested.resolve(strict=False)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    incomplete = root / _INCOMPLETE_NAME
    incomplete.open("xb").close()

    paths = _receipt_paths(root)
    _write_new_json(paths["config"], _config_receipt(config, source_digest))
    d_v_cache_root = root / "d_v" / "base_cache"
    materialize_base_cache_bundle(
        contract.d_v_index_path,
        d_v_cache_root,
        expected_split="D_V",
        expected_base_fingerprint=base_fingerprint,
    )
    d_r_cache_root = root / "d_r" / "base_cache"
    materialize_base_cache_bundle(
        contract.d_r_index_path,
        d_r_cache_root,
        expected_split="D_R",
        expected_base_fingerprint=base_fingerprint,
    )

    d_v_bundle = load_d_v_cache_bundle(
        d_v_cache_root / "index.json",
        d_v_dataset,
        expected_base_fingerprint=base_fingerprint,
    )
    d_v_base_run = build_loaded_d_v_base_run(d_v_bundle)
    anchor = select_frozen_anchor(
        d_v_base_run,
        config.anchor_thresholds,
        config.match_config,
    )
    anchor_metrics = evaluate_frozen_anchor(d_v_base_run, anchor)
    _require_anchor_within_budget(anchor_metrics, config.budget)
    _write_new_json(paths["anchor"], _anchor_receipt_payload(anchor))

    d_r_state_root = root / "d_r" / "state_cache"
    cache_d_r_states(
        d_r_cache_root / "index.json",
        d_r_dataset,
        d_r_state_root,
        expected_base_fingerprint=base_fingerprint,
        occupancy_config=anchor.occupancy_config,
        match_config=config.match_config,
        intervention_config=config.intervention_config,
    )
    d_r_bundle = load_d_r_cache_bundle(
        d_r_state_root / "index.json",
        d_r_dataset,
        expected_base_fingerprint=base_fingerprint,
    )
    support_summary = summarize_gate2_training_support(d_r_bundle)
    config.support_requirements.require(support_summary)
    _write_new_json(
        paths["support"],
        _support_receipt_payload(
            support_summary,
            config.support_requirements,
        ),
    )
    paired = run_paired_gate2_training(
        d_r_bundle,
        config.training,
        device=config.device,
    )
    factual_directory = root / "decoders" / "factual_only"
    uniform_directory = root / "decoders" / "uniform_legal"
    save_completed_decoder_run(factual_directory, paired.factual_only)
    save_completed_decoder_run(uniform_directory, paired.uniform_legal)
    factual_artifact = load_decoder_artifact(
        factual_directory,
        expected_config=paired.factual_only.config,
    )
    uniform_artifact = load_decoder_artifact(
        uniform_directory,
        expected_config=paired.uniform_legal.config,
    )
    state = _build_downstream_state(
        config=config,
        d_r_bundle=d_r_bundle,
        d_v_bundle=d_v_bundle,
        factual_artifact=factual_artifact,
        uniform_artifact=uniform_artifact,
    )
    if state.anchor.canonical_payload() != anchor.canonical_payload():
        raise RuntimeError("frozen anchor changed before Stage-A publication")
    _write_new_json(
        paths["calibration"],
        _calibration_receipt_payload(state.calibration),
    )
    _write_new_json(
        paths["results"],
        _results_receipt_payload(state.results, state.calibration),
    )
    if _source_tree_digest() != source_digest:
        raise RuntimeError("CURE-Lite Python sources changed during Stage-A")

    incomplete.unlink()
    directories, files = _tree_inventory(root)
    complete = _complete_receipt(
        state,
        source_digest=source_digest,
        artifact_directories=directories,
        artifact_files=files,
    )
    _write_new_json(root / _COMPLETE_NAME, complete)
    return load_stage_a_run(
        root,
        d_r_dataset,
        d_v_dataset,
        expected_base_fingerprint=base_fingerprint,
    )


def _load_verified_state(
    root: Path,
    d_r_dataset: ManifestImageDataset,
    d_v_dataset: ManifestImageDataset,
    *,
    expected_base_fingerprint: str,
) -> tuple[_StageAState, str]:
    _check_dataset_pair(d_r_dataset, d_v_dataset)
    base_fingerprint = _canonical_digest(
        expected_base_fingerprint, name="expected_base_fingerprint"
    )
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Stage-A root must be a regular non-symlink directory")
    if (root / _INCOMPLETE_NAME).exists() or (root / _INCOMPLETE_NAME).is_symlink():
        raise RuntimeError("Stage-A run is incomplete and cannot be loaded")
    complete_path = root / _COMPLETE_NAME
    complete = _strict_json(complete_path, name="Stage-A COMPLETE receipt")
    expected_complete_keys = {
        "schema_version",
        "status",
        "method",
        "stage",
        "method_order",
        "runtime_splits",
        "unused_split",
        "source_tree_digest",
        "run_config_fingerprint",
        "anchor_receipt_fingerprint",
        "support_receipt_fingerprint",
        "calibration_receipt_fingerprint",
        "results_fingerprint",
        "dataset",
        "manifest_fingerprint",
        "manifest_file_sha256",
        "preprocessing_fingerprint",
        "base_fingerprint",
        "d_r_base_index_fingerprint",
        "d_r_state_index_fingerprint",
        "d_v_base_index_fingerprint",
        "factual_decoder_artifact_fingerprint",
        "uniform_decoder_artifact_fingerprint",
        "artifact_directories",
        "artifact_files",
        "complete_fingerprint",
    }
    if set(complete) != expected_complete_keys:
        raise ValueError("Stage-A COMPLETE fields are not canonical")
    if (
        complete["schema_version"] != STAGE_A_RUN_SCHEMA
        or complete["status"] != "complete"
        or complete["method"] != "CURE-Lite"
        or complete["stage"] != "Stage-A"
        or complete["method_order"] != list(_METHOD_ORDER)
        or complete["runtime_splits"] != ["D_R", "D_V"]
        or complete["unused_split"] != "D_T"
    ):
        raise ValueError("Stage-A COMPLETE protocol fields are invalid")
    complete_core = dict(complete)
    complete_fingerprint = _canonical_digest(
        complete_core.pop("complete_fingerprint"), name="complete_fingerprint"
    )
    if stable_fingerprint(complete_core) != complete_fingerprint:
        raise ValueError("Stage-A COMPLETE fingerprint mismatch")
    directories, files = _tree_inventory(root)
    if complete["artifact_directories"] != directories:
        raise ValueError("Stage-A artifact directory inventory changed")
    if complete["artifact_files"] != files:
        raise ValueError("Stage-A artifact file inventory changed")
    source_digest = _source_tree_digest()
    if complete["source_tree_digest"] != source_digest:
        raise RuntimeError("CURE-Lite Python source tree differs from this run")

    paths = _receipt_paths(root)
    config_receipt = _strict_json(paths["config"], name="Stage-A config receipt")
    if set(config_receipt) != {
        "schema_version",
        "method",
        "stage",
        "runtime_splits",
        "unused_split",
        "source_tree_digest",
        "run_config",
        "run_config_fingerprint",
    }:
        raise ValueError("Stage-A config receipt fields are not canonical")
    raw_config = config_receipt["run_config"]
    if not isinstance(raw_config, Mapping):
        raise TypeError("Stage-A run_config must be a mapping")
    config = StageARunConfig.from_mapping(raw_config)
    if config_receipt != _config_receipt(config, source_digest):
        raise ValueError("Stage-A config receipt does not reproduce")

    d_v_bundle = load_d_v_cache_bundle(
        root / "d_v" / "base_cache" / "index.json",
        d_v_dataset,
        expected_base_fingerprint=base_fingerprint,
    )
    d_r_bundle = load_d_r_cache_bundle(
        root / "d_r" / "state_cache" / "index.json",
        d_r_dataset,
        expected_base_fingerprint=base_fingerprint,
    )
    factual_artifact = load_decoder_artifact(root / "decoders" / "factual_only")
    uniform_artifact = load_decoder_artifact(root / "decoders" / "uniform_legal")
    state = _build_downstream_state(
        config=config,
        d_r_bundle=d_r_bundle,
        d_v_bundle=d_v_bundle,
        factual_artifact=factual_artifact,
        uniform_artifact=uniform_artifact,
    )
    expected_receipts = {
        "anchor": _anchor_receipt_payload(state.anchor),
        "support": _support_receipt_payload(
            state.support_summary,
            state.config.support_requirements,
        ),
        "calibration": _calibration_receipt_payload(state.calibration),
        "results": _results_receipt_payload(state.results, state.calibration),
    }
    for name, expected in expected_receipts.items():
        if _strict_json(paths[name], name=f"Stage-A {name} receipt") != expected:
            raise RuntimeError(f"Stage-A {name} receipt does not reproduce")
    expected_complete = _complete_receipt(
        state,
        source_digest=source_digest,
        artifact_directories=directories,
        artifact_files=files,
    )
    if complete != expected_complete:
        raise RuntimeError("Stage-A COMPLETE receipt does not reproduce")
    return state, complete_fingerprint


@dataclass(frozen=True, slots=True)
class _LoadedStageABinding:
    root: Path
    state: _StageAState
    complete_fingerprint: str
    d_r_dataset: ManifestImageDataset
    d_v_dataset: ManifestImageDataset
    expected_base_fingerprint: str


@dataclass(frozen=True)
class LoadedStageARun:
    """A fully replayed Stage-A result backed by immutable receipts."""

    root: Path
    config: StageARunConfig
    d_r_bundle: LoadedDRCacheBundle
    d_v_bundle: LoadedDVCacheBundle
    factual_artifact: LoadedDecoderArtifact
    uniform_artifact: LoadedDecoderArtifact
    anchor: FrozenAnchorReceipt
    support_summary: TrainingSupportSummary
    calibration: PairedGate2Calibration
    results: Gate2DVResults
    complete_fingerprint: str
    _verification_token: object

    def _verify_binding(self) -> _LoadedStageABinding:
        binding = self._verification_token
        if type(binding) is not _LoadedStageABinding:
            raise TypeError("LoadedStageARun must come from load_stage_a_run")
        state = binding.state
        if (
            binding.root != self.root
            or state.config is not self.config
            or state.d_r_bundle is not self.d_r_bundle
            or state.d_v_bundle is not self.d_v_bundle
            or state.factual_artifact is not self.factual_artifact
            or state.uniform_artifact is not self.uniform_artifact
            or state.anchor is not self.anchor
            or state.support_summary is not self.support_summary
            or state.calibration is not self.calibration
            or state.results is not self.results
            or binding.complete_fingerprint != self.complete_fingerprint
        ):
            raise TypeError("loaded Stage-A fields were replaced")
        return binding

    def __post_init__(self) -> None:
        if not self.root.is_absolute():
            raise ValueError("loaded Stage-A root must be absolute")
        _canonical_digest(
            self.complete_fingerprint, name="complete_fingerprint"
        )
        self._verify_binding()

    def verify_unchanged(self) -> None:
        """Replay every cache, artifact, selection, metric, and root receipt."""

        binding = self._verify_binding()
        state, fingerprint = _load_verified_state(
            self.root,
            binding.d_r_dataset,
            binding.d_v_dataset,
            expected_base_fingerprint=binding.expected_base_fingerprint,
        )
        if fingerprint != self.complete_fingerprint:
            raise RuntimeError("Stage-A COMPLETE fingerprint changed")
        if (
            state.config != self.config
            or state.anchor.canonical_payload() != self.anchor.canonical_payload()
            or state.support_summary.canonical_payload()
            != self.support_summary.canonical_payload()
            or _calibration_receipt_payload(state.calibration)
            != _calibration_receipt_payload(self.calibration)
            or _results_receipt_payload(state.results, state.calibration)
            != _results_receipt_payload(self.results, self.calibration)
        ):
            raise RuntimeError("replayed Stage-A result differs from loaded result")

    def verify(self) -> None:
        """Alias for :meth:`verify_unchanged`."""

        self.verify_unchanged()


def load_stage_a_run(
    output_dir: str | Path,
    d_r_dataset: ManifestImageDataset,
    d_v_dataset: ManifestImageDataset,
    *,
    expected_base_fingerprint: str,
) -> LoadedStageARun:
    """Strictly reload and fully replay a completed Stage-A run."""

    requested = Path(output_dir).expanduser()
    if requested.is_symlink():
        raise ValueError("Stage-A root may not be addressed through a symlink")
    root = requested.resolve(strict=True)
    state, complete_fingerprint = _load_verified_state(
        root,
        d_r_dataset,
        d_v_dataset,
        expected_base_fingerprint=expected_base_fingerprint,
    )
    binding = _LoadedStageABinding(
        root=root,
        state=state,
        complete_fingerprint=complete_fingerprint,
        d_r_dataset=d_r_dataset,
        d_v_dataset=d_v_dataset,
        expected_base_fingerprint=_canonical_digest(
            expected_base_fingerprint, name="expected_base_fingerprint"
        ),
    )
    return LoadedStageARun(
        root=root,
        config=state.config,
        d_r_bundle=state.d_r_bundle,
        d_v_bundle=state.d_v_bundle,
        factual_artifact=state.factual_artifact,
        uniform_artifact=state.uniform_artifact,
        anchor=state.anchor,
        support_summary=state.support_summary,
        calibration=state.calibration,
        results=state.results,
        complete_fingerprint=complete_fingerprint,
        _verification_token=binding,
    )


__all__ = [
    "LoadedStageARun",
    "StageARunConfig",
    "load_stage_a_run",
    "run_stage_a",
    "run_stage_a_from_base_caches",
]
