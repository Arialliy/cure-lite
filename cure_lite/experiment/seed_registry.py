"""Build a Stage-A seed registry from one already loaded completed run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..base_identity import VerifiedBaseRunIdentity
from ..cache.schema import file_sha256, stable_fingerprint
from ..calibration import THRESHOLD_SELECTION_RULE
from ..config import config_to_dict
from ..data import PreprocessConfig
from ..stage_a import (
    METHOD_VERSION,
    SEED_REGISTRY_SCHEMA_VERSION,
    STAGE_A_EVALUATION_MODES,
    STAGE_A_VARIANTS,
    threshold_grid_fingerprint,
    validate_seed_registry_mapping,
)
from .stage_a_runner import LoadedStageARun


def _strict_json_object(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("Stage-A base cache index is unavailable")
    raw = path.read_bytes()
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("Stage-A base cache index changed during registry build")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"base cache index repeats key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"base cache index contains non-finite number {value}")

    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise TypeError("Stage-A base cache index must contain an object")
    return value


def _preprocessing_from_run(run: LoadedStageARun) -> dict[str, object]:
    bundle = run.d_v_bundle
    index = _strict_json_object(
        bundle.base_index_path,
        expected_sha256=bundle.base_index_sha256,
    )
    preprocessing = PreprocessConfig.from_fingerprint_payload(
        index.get("preprocessing")
    )
    payload = preprocessing.fingerprint_payload()
    if stable_fingerprint(payload) != bundle.preprocessing_fingerprint:
        raise RuntimeError("Stage-A preprocessing differs from D_V cache")
    if run.d_r_bundle.preprocessing_fingerprint != bundle.preprocessing_fingerprint:
        raise RuntimeError("D_R and D_V preprocessing fingerprints differ")
    return payload


def _finite_or_none(value: float) -> float | None:
    return None if math.isinf(value) else float(value)


def _protocols(run: LoadedStageARun) -> dict[str, dict[str, object]]:
    calibration = run.calibration
    receipts = {
        "Base@B": calibration.base_at_budget,
        "F": calibration.factual_only,
        "F×": calibration.factual_exposure_matched,
        "U": calibration.uniform_legal,
    }
    result: dict[str, dict[str, object]] = {
        "A": {
            "evaluation_mode": STAGE_A_EVALUATION_MODES["A"],
            "decoder_variant": None,
            "protocol_fingerprint": run.anchor.receipt_fingerprint,
            "decoder_artifact_fingerprint": None,
            "selected_threshold": float(run.anchor.selected_threshold),
        }
    }
    for method_id, receipt in receipts.items():
        variant = STAGE_A_VARIANTS[method_id]
        artifact_fingerprint = (
            None
            if variant is None
            else receipt.decoder_artifact_fingerprint  # type: ignore[union-attr]
        )
        result[method_id] = {
            "evaluation_mode": STAGE_A_EVALUATION_MODES[method_id],
            "decoder_variant": variant,
            "protocol_fingerprint": receipt.protocol.receipt_fingerprint,
            "decoder_artifact_fingerprint": artifact_fingerprint,
            "selected_threshold": receipt.protocol.selected_threshold,
        }
    return result


def build_seed_registry_from_stage_a_run(
    run: LoadedStageARun,
    verified_base_identity: VerifiedBaseRunIdentity,
) -> dict[str, Any]:
    """Construct and validate registry v6 without repeating D_V computation."""

    if not isinstance(run, LoadedStageARun):
        raise TypeError("run must be a LoadedStageARun")
    if not isinstance(verified_base_identity, VerifiedBaseRunIdentity):
        raise TypeError(
            "verified_base_identity must come from a registered Base-run loader"
        )
    run.verify_published_receipts()
    run.d_r_bundle.verify_unchanged()
    run.d_v_bundle.verify_unchanged()
    verified_base_identity.verify_unchanged()

    d_r = run.d_r_bundle
    d_v = run.d_v_bundle
    identity = verified_base_identity.identity
    if identity != run.base_run_identity:
        raise RuntimeError("verified Base run differs from Stage-A completion")
    if d_r.base_fingerprint != d_v.base_fingerprint:
        raise RuntimeError("D_R and D_V Base fingerprints differ")
    if d_r.base_state_fingerprint != d_v.base_state_fingerprint:
        raise RuntimeError("D_R and D_V Base state fingerprints differ")
    if identity.base_fingerprint != d_v.base_fingerprint:
        raise RuntimeError("verified Base run differs from Stage-A Base")
    if identity.base_state_fingerprint != d_v.base_state_fingerprint:
        raise RuntimeError("verified Base state differs from Stage-A Base state")
    if d_r.split_manifest_fingerprint != d_v.split_manifest_fingerprint:
        raise RuntimeError("D_R and D_V manifest fingerprints differ")

    config = run.config
    training = config.training
    anchor_grid = list(config.anchor_thresholds)
    residual_grid = list(config.residual_thresholds)
    base_grid = list(config.base_thresholds)
    budget = config.budget
    efficiency = run.efficiency
    if efficiency.binding.decoder_artifact_fingerprint != (
        run.uniform_artifact.artifact_fingerprint
    ):
        raise RuntimeError("efficiency receipt differs from the deployed U artifact")
    if efficiency.binding.base_index_fingerprint != d_v.base_index_fingerprint:
        raise RuntimeError("efficiency receipt differs from the D_V cache")

    common: dict[str, object] = {
        "manifest_fingerprint": d_v.split_manifest_fingerprint,
        "base_fingerprint": d_v.base_fingerprint,
        "base_state_fingerprint": d_v.base_state_fingerprint,
        "base_run_identity": identity.to_registry_dict(),
        "stage_a_complete_fingerprint": run.complete_fingerprint,
        "d_v_image_fingerprint": d_v.d_v_image_fingerprint,
        "d_v_gt_fingerprint": d_v.d_v_gt_fingerprint,
        "d_v_base_cache_index_fingerprint": d_v.base_index_fingerprint,
        "d_v_base_cache_index_sha256": d_v.base_index_sha256,
        "d_r_base_cache_index_fingerprint": d_r.base_index_fingerprint,
        "d_r_base_cache_index_sha256": d_r.base_index_sha256,
        "d_r_state_cache_index_fingerprint": d_r.state_index_fingerprint,
        "d_r_state_cache_index_sha256": d_r.state_index_sha256,
        "anchor_protocol_sha256": run.anchor.receipt_fingerprint,
        "state_fingerprint": d_r.state_fingerprint,
        "tau_o": float(run.anchor.selected_threshold),
        "tau_B": float(run.calibration.base_at_budget.protocol.selected_threshold),
        "pixel_fa_budget": float(budget.pixel_fa_budget),
        "component_fa_per_mp_budget": _finite_or_none(
            budget.component_fa_per_mp_budget
        ),
        "raw_background_fa_budget": _finite_or_none(
            budget.raw_background_fa_budget
        ),
        "minimum_retention": float(budget.minimum_retention),
        "null_residual_candidate": True,
        "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
        "anchor_threshold_grid": anchor_grid,
        "anchor_threshold_grid_fingerprint": threshold_grid_fingerprint(anchor_grid),
        "residual_threshold_grid": residual_grid,
        "residual_threshold_grid_fingerprint": threshold_grid_fingerprint(
            residual_grid
        ),
        "base_threshold_grid": base_grid,
        "base_threshold_grid_fingerprint": threshold_grid_fingerprint(base_grid),
        "occupancy_config": config_to_dict(run.anchor.occupancy_config),
        "matching_config": config_to_dict(config.match_config),
        "intervention_config": config_to_dict(config.intervention_config),
        "preprocessing": _preprocessing_from_run(run),
        "decoder_config": config_to_dict(training.decoder_config),
        "loss_config": config_to_dict(training.loss_config),
        "training_config": config_to_dict(training.training_config),
        "optimization_config": {
            "optimizer": training.optimizer,
            "learning_rate": training.learning_rate,
            "weight_decay": training.weight_decay,
        },
        "branch_batch_sizes": {
            "factual_miss": training.factual_miss_batch,
            "factual_no_miss": training.factual_no_miss_batch,
            "synthetic": training.synthetic_batch,
        },
        "data_augmentation": "none_frozen_base_cache",
        "fixed_stopping_rule": {
            "epochs": training.epochs,
            "steps_per_epoch": training.steps_per_epoch,
        },
        "global_seed": training.global_seed,
        "steps_per_epoch": training.steps_per_epoch,
        "trained_epochs": training.epochs,
        "efficiency_device_type": efficiency.device_type,
        "efficiency_warmup": efficiency.warmup,
        "efficiency_repetitions": efficiency.repetitions,
        "efficiency_static_fingerprint": efficiency.static_fingerprint,
        "efficiency_receipt_fingerprint": efficiency.receipt_fingerprint,
    }
    payload: dict[str, Any] = {
        "schema_version": SEED_REGISTRY_SCHEMA_VERSION,
        "artifact_type": "stage_a_frozen_registry",
        "method_version": METHOD_VERSION,
        "stage": "Stage A",
        "split": "D_V",
        "thresholds_frozen": True,
        "common_config": common,
        "protocols": _protocols(run),
    }
    validate_seed_registry_mapping(payload)
    return payload


__all__ = ["build_seed_registry_from_stage_a_run"]
