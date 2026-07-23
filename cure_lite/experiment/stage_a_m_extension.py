"""Read-only loader for completed historical CURE-Lite v0.1 Stage-A runs.

This module deliberately does not call :func:`load_stage_a_run`.  A historical
run records the exact source digest that created it, while the current source
tree may legitimately contain the later M extension.  The loader therefore
checks the recorded digest across the historical receipts but never compares it
with the current checkout.

No cache replay, decoder training, calibration, or output publication is
performed here.  The returned object is a sealed reference snapshot whose
persisted bytes can be checked again with :meth:`verify_unchanged`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from ..cache.schema import file_sha256, stable_fingerprint
from ..calibration import THRESHOLD_SELECTION_RULE
from ..metrics import FORMAL_STAGE_A_METRIC_FIELDS
from ..stage_a import BASE_RUN_IDENTITY_FIELDS, STAGE_A_METHOD_ORDER
from .artifacts import (
    DECODER_ARTIFACT_SCHEMA_V2,
    LoadedDecoderArtifact,
    load_decoder_artifact,
)
from .efficiency_evidence import (
    StageAEfficiencyReceipt,
    replay_static_efficiency,
)
from .stage_a_runner import (
    STAGE_A_CONFIG_SCHEMA,
    STAGE_A_RUN_SCHEMA,
    StageARunConfig,
    _strict_json,
    _tree_inventory,
)
from .training_pipeline import (
    TrainingSupportRequirements,
    TrainingSupportSummary,
)


REFERENCE_SNAPSHOT_SCHEMA = "cure-lite-stage-a-reference-snapshot-v1"
_CONFIG_RECEIPT_SCHEMA = "cure-lite-stage-a-config-receipt-v1"
_ANCHOR_RECEIPT_SCHEMA = "cure-lite-stage-a-anchor-receipt-v2"
_ANCHOR_SCHEMA = "cure-lite-frozen-anchor-receipt-v2"
_SUPPORT_SCHEMA = "cure-lite-stage-a-support-receipt-v1"
_CALIBRATION_SCHEMA = "cure-lite-stage-a-calibration-receipt-v4"
_RESULTS_SCHEMA = "cure-lite-stage-a-results-receipt-v3"
_METHOD_ORDER = tuple(STAGE_A_METHOD_ORDER)
_INCOMPLETE_NAME = ".incomplete"
_COMPLETE_NAME = "COMPLETE.json"
_SHA256_CHARS = frozenset("0123456789abcdef")

_COMPLETE_FIELDS = frozenset(
    {
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
        "efficiency_receipt_fingerprint",
        "dataset",
        "manifest_fingerprint",
        "manifest_file_sha256",
        "preprocessing_fingerprint",
        "base_fingerprint",
        "base_state_fingerprint",
        "base_run_identity",
        "d_r_base_index_fingerprint",
        "d_r_state_index_fingerprint",
        "d_v_base_index_fingerprint",
        "factual_decoder_artifact_fingerprint",
        "factual_exposure_matched_decoder_artifact_fingerprint",
        "uniform_decoder_artifact_fingerprint",
        "artifact_directories",
        "artifact_files",
        "complete_fingerprint",
    }
)
_CONFIG_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "method",
        "stage",
        "runtime_splits",
        "unused_split",
        "source_tree_digest",
        "run_config",
        "run_config_fingerprint",
    }
)
_ANCHOR_FIELDS = frozenset(
    {
        "schema_version",
        "selection_rule",
        "candidate_threshold_grid",
        "occupancy_config",
        "match_config",
        "selected_threshold",
        "selected_metrics",
        "d_v_run_fingerprint",
        "ordered_d_v_sample_ids",
        "base_samples_fingerprint",
        "manifest_fingerprint",
        "manifest_file_sha256",
        "base_index_fingerprint",
        "base_index_sha256",
        "d_v_image_fingerprint",
        "d_v_gt_fingerprint",
        "preprocessing_fingerprint",
        "base_fingerprint",
    }
)
_PROTOCOL_FIELDS = frozenset(
    {
        "variant",
        "selection_rule",
        "manifest_fingerprint",
        "ordered_d_v_sample_ids",
        "sample_tensor_fingerprint",
        "candidate_threshold_grid",
        "occupancy_config",
        "match_config",
        "budget",
        "selected_threshold",
        "selected_metrics",
        "receipt_fingerprint",
    }
)
_BASE_METHOD_FIELDS = frozenset(
    {
        "schema_version",
        "method",
        "protocol",
        "d_v_base_run_fingerprint",
        "manifest_file_sha256",
        "base_index_fingerprint",
        "base_index_sha256",
        "d_v_image_fingerprint",
        "d_v_gt_fingerprint",
        "preprocessing_fingerprint",
        "base_fingerprint",
        "receipt_fingerprint",
    }
)
_RESIDUAL_METHOD_FIELDS = frozenset(
    {
        "schema_version",
        "mode",
        "protocol",
        "d_v_run_fingerprint",
        "manifest_file_sha256",
        "base_index_fingerprint",
        "base_index_sha256",
        "d_v_image_fingerprint",
        "d_v_gt_fingerprint",
        "preprocessing_fingerprint",
        "base_fingerprint",
        "decoder_artifact_fingerprint",
        "decoder_receipt_sha256",
        "decoder_state_fingerprint",
        "decoder_variant",
        "global_seed",
        "receipt_fingerprint",
    }
)
_COMMON_D_V_PROVENANCE = (
    "manifest_file_sha256",
    "base_index_fingerprint",
    "base_index_sha256",
    "d_v_image_fingerprint",
    "d_v_gt_fingerprint",
    "preprocessing_fingerprint",
    "base_fingerprint",
)
_DECODER_BY_METHOD = {
    "F": "factual_only",
    "F×": "factual_exposure_matched",
    "U": "uniform_legal",
}
_DECODER_COMPLETE_FIELDS = {
    "F": "factual_decoder_artifact_fingerprint",
    "F×": "factual_exposure_matched_decoder_artifact_fingerprint",
    "U": "uniform_decoder_artifact_fingerprint",
}
_BASE_INDEX_FIELDS = frozenset(
    {
        "schema_version",
        "base_cache_schema",
        "dataset",
        "split",
        "sample_count",
        "split_manifest_fingerprint",
        "split_manifest_file_sha256",
        "base_fingerprint",
        "base_state_fingerprint",
        "preprocessing",
        "preprocessing_fingerprint",
        "records",
        "index_fingerprint",
    }
)
_STATE_INDEX_FIELDS = frozenset(
    {
        "schema_version",
        "state_cache_schema",
        "dataset",
        "split",
        "sample_count",
        "split_manifest_fingerprint",
        "split_manifest_file_sha256",
        "base_fingerprint",
        "base_state_fingerprint",
        "base_index",
        "preprocessing",
        "preprocessing_fingerprint",
        "occupancy_config",
        "matching_config",
        "intervention_config",
        "gt_fingerprint",
        "state_fingerprint",
        "records",
        "index_fingerprint",
    }
)


def _exact_mapping(
    value: object,
    expected: frozenset[str] | set[str],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(f"{name} fields are not canonical")
    return dict(value)


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA256 string")
    if len(value) != 64 or any(character not in _SHA256_CHARS for character in value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _threshold_grid(value: object, *, name: str, allow_empty: bool = False) -> list[float]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    result = [_finite_number(item, name=f"{name} item") for item in value]
    if not result and not allow_empty:
        raise ValueError(f"{name} must not be empty")
    if result != sorted(set(result)) or any(item < 0.0 or item > 1.0 for item in result):
        raise ValueError(f"{name} must be sorted, unique, and in [0,1]")
    return result


def _metrics(value: object, *, name: str) -> dict[str, Any]:
    payload = _exact_mapping(
        value,
        set(FORMAL_STAGE_A_METRIC_FIELDS),
        name=name,
    )
    for field in FORMAL_STAGE_A_METRIC_FIELDS:
        item = payload[field]
        if field == "budget_violation":
            if not isinstance(item, bool):
                raise TypeError(f"{name}.{field} must be bool")
        else:
            _finite_number(item, name=f"{name}.{field}")
    return payload


def _ordered_sample_ids(value: object, *, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must be a non-empty unique string list")
    return value


def _budget(value: object, *, name: str) -> dict[str, Any]:
    payload = _exact_mapping(
        value,
        {
            "pixel_fa_budget",
            "component_fa_per_mp_budget",
            "raw_background_fa_budget",
            "minimum_retention",
        },
        name=name,
    )
    for key, item in payload.items():
        if item is not None:
            _finite_number(item, name=f"{name}.{key}")
    return payload


def _validate_complete_protocol(complete: Mapping[str, Any]) -> None:
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
    if not isinstance(complete["dataset"], str) or not complete["dataset"]:
        raise ValueError("Stage-A COMPLETE dataset must be non-empty")
    for name in (
        "source_tree_digest",
        "run_config_fingerprint",
        "anchor_receipt_fingerprint",
        "support_receipt_fingerprint",
        "calibration_receipt_fingerprint",
        "results_fingerprint",
        "efficiency_receipt_fingerprint",
        "manifest_fingerprint",
        "manifest_file_sha256",
        "preprocessing_fingerprint",
        "base_fingerprint",
        "base_state_fingerprint",
        "d_r_base_index_fingerprint",
        "d_r_state_index_fingerprint",
        "d_v_base_index_fingerprint",
        "factual_decoder_artifact_fingerprint",
        "factual_exposure_matched_decoder_artifact_fingerprint",
        "uniform_decoder_artifact_fingerprint",
    ):
        _digest(complete[name], name=f"COMPLETE.{name}")
    identity = _exact_mapping(
        complete["base_run_identity"],
        set(BASE_RUN_IDENTITY_FIELDS),
        name="COMPLETE.base_run_identity",
    )
    if not isinstance(identity["producer_schema"], str) or not identity["producer_schema"]:
        raise ValueError("COMPLETE base producer schema must be non-empty")
    for name in BASE_RUN_IDENTITY_FIELDS[1:]:
        _digest(identity[name], name=f"COMPLETE.base_run_identity.{name}")
    if (
        identity["base_fingerprint"] != complete["base_fingerprint"]
        or identity["base_state_fingerprint"] != complete["base_state_fingerprint"]
    ):
        raise RuntimeError("Stage-A COMPLETE Base identity fields disagree")


def _load_config_receipt(
    root: Path,
    complete: Mapping[str, Any],
) -> StageARunConfig:
    receipt = _exact_mapping(
        _strict_json(root / "receipts" / "config.json", name="Stage-A config receipt"),
        _CONFIG_RECEIPT_FIELDS,
        name="Stage-A config receipt",
    )
    if (
        receipt["schema_version"] != _CONFIG_RECEIPT_SCHEMA
        or receipt["method"] != "CURE-Lite"
        or receipt["stage"] != "Stage-A"
        or receipt["runtime_splits"] != ["D_R", "D_V"]
        or receipt["unused_split"] != "D_T"
    ):
        raise ValueError("Stage-A config receipt protocol fields are invalid")
    source_digest = _digest(
        receipt["source_tree_digest"],
        name="config source_tree_digest",
    )
    if source_digest != complete["source_tree_digest"]:
        raise RuntimeError("config and COMPLETE source digests differ")
    raw_config = receipt["run_config"]
    if not isinstance(raw_config, Mapping):
        raise TypeError("Stage-A run_config must be a mapping")
    config = StageARunConfig.from_mapping(raw_config)
    fingerprint = _digest(
        receipt["run_config_fingerprint"],
        name="run_config_fingerprint",
    )
    if stable_fingerprint(raw_config) != fingerprint:
        raise ValueError("Stage-A run config fingerprint mismatch")
    if fingerprint != complete["run_config_fingerprint"]:
        raise RuntimeError("config and COMPLETE run fingerprints differ")
    return config


def _validate_anchor(
    root: Path,
    complete: Mapping[str, Any],
    config: StageARunConfig,
) -> tuple[dict[str, Any], str]:
    wrapper = _exact_mapping(
        _strict_json(root / "receipts" / "anchor.json", name="Stage-A anchor receipt"),
        {"schema_version", "receipt", "receipt_fingerprint"},
        name="Stage-A anchor receipt",
    )
    if wrapper["schema_version"] != _ANCHOR_RECEIPT_SCHEMA:
        raise ValueError("Stage-A anchor receipt schema is invalid")
    anchor = _exact_mapping(wrapper["receipt"], _ANCHOR_FIELDS, name="frozen anchor")
    if anchor["schema_version"] != _ANCHOR_SCHEMA:
        raise ValueError("frozen anchor schema is invalid")
    if anchor["selection_rule"] != "max_global_miou_tie_higher_threshold":
        raise ValueError("frozen anchor selection rule is invalid")
    grid = _threshold_grid(
        anchor["candidate_threshold_grid"],
        name="anchor threshold grid",
    )
    if grid != list(config.anchor_thresholds):
        raise RuntimeError("anchor threshold grid differs from Stage-A config")
    if not isinstance(anchor["occupancy_config"], Mapping):
        raise TypeError("anchor occupancy config must be a mapping")
    selected = _finite_number(
        anchor["selected_threshold"],
        name="anchor selected threshold",
    )
    if selected != anchor["occupancy_config"].get("threshold"):
        raise RuntimeError("anchor selected threshold and occupancy config differ")
    if anchor["match_config"] != config.canonical_payload()["match_config"]:
        raise RuntimeError("anchor matching config differs from Stage-A config")
    _metrics(anchor["selected_metrics"], name="anchor selected metrics")
    _ordered_sample_ids(
        anchor["ordered_d_v_sample_ids"],
        name="anchor ordered D_V sample IDs",
    )
    for name in (
        "d_v_run_fingerprint",
        "base_samples_fingerprint",
        "manifest_fingerprint",
        "manifest_file_sha256",
        "base_index_fingerprint",
        "base_index_sha256",
        "d_v_image_fingerprint",
        "d_v_gt_fingerprint",
        "preprocessing_fingerprint",
        "base_fingerprint",
    ):
        _digest(anchor[name], name=f"anchor.{name}")
    expected = {
        "manifest_fingerprint": complete["manifest_fingerprint"],
        "manifest_file_sha256": complete["manifest_file_sha256"],
        "base_index_fingerprint": complete["d_v_base_index_fingerprint"],
        "preprocessing_fingerprint": complete["preprocessing_fingerprint"],
        "base_fingerprint": complete["base_fingerprint"],
    }
    if any(anchor[name] != value for name, value in expected.items()):
        raise RuntimeError("anchor provenance differs from COMPLETE")
    fingerprint = _digest(
        wrapper["receipt_fingerprint"],
        name="anchor receipt fingerprint",
    )
    if stable_fingerprint(anchor) != fingerprint:
        raise ValueError("anchor receipt fingerprint mismatch")
    if fingerprint != complete["anchor_receipt_fingerprint"]:
        raise RuntimeError("anchor and COMPLETE fingerprints differ")
    return anchor, fingerprint


def _validate_protocol(
    value: object,
    *,
    name: str,
    expected_variant: str,
) -> tuple[dict[str, Any], str]:
    protocol = _exact_mapping(value, _PROTOCOL_FIELDS, name=name)
    if protocol["variant"] != expected_variant:
        raise ValueError(f"{name} has the wrong variant")
    if protocol["selection_rule"] != THRESHOLD_SELECTION_RULE:
        raise ValueError(f"{name} has the wrong selection rule")
    for field in ("manifest_fingerprint", "sample_tensor_fingerprint"):
        _digest(protocol[field], name=f"{name}.{field}")
    _ordered_sample_ids(
        protocol["ordered_d_v_sample_ids"],
        name=f"{name}.ordered_d_v_sample_ids",
    )
    grid = _threshold_grid(
        protocol["candidate_threshold_grid"],
        name=f"{name}.candidate_threshold_grid",
        allow_empty=expected_variant == "residual",
    )
    if not isinstance(protocol["occupancy_config"], Mapping):
        raise TypeError(f"{name}.occupancy_config must be a mapping")
    if not isinstance(protocol["match_config"], Mapping):
        raise TypeError(f"{name}.match_config must be a mapping")
    _budget(protocol["budget"], name=f"{name}.budget")
    _metrics(protocol["selected_metrics"], name=f"{name}.selected_metrics")
    threshold = protocol["selected_threshold"]
    if threshold is None:
        if expected_variant != "residual":
            raise ValueError(f"{name} requires a numeric selected threshold")
    else:
        selected = _finite_number(threshold, name=f"{name}.selected_threshold")
        if not 0.0 <= selected <= 1.0:
            raise ValueError(f"{name}.selected_threshold must be in [0,1]")
        permitted = set(grid)
        if expected_variant == "base_at_budget":
            occupancy_threshold = _finite_number(
                protocol["occupancy_config"].get("threshold"),
                name=f"{name}.occupancy threshold",
            )
            permitted.add(occupancy_threshold)
            if selected > occupancy_threshold:
                raise ValueError(f"{name} selected threshold exceeds the anchor")
        if selected not in permitted:
            raise ValueError(f"{name} selected threshold is absent from its grid")
    fingerprint = _digest(
        protocol["receipt_fingerprint"],
        name=f"{name}.receipt_fingerprint",
    )
    core = dict(protocol)
    core.pop("receipt_fingerprint")
    expected_fingerprint = stable_fingerprint(
        {
            "schema_version": "cure-lite-bound-d-v-threshold-protocol-v3",
            **core,
        }
    )
    if fingerprint != expected_fingerprint:
        raise ValueError(f"{name} fingerprint mismatch")
    return protocol, fingerprint


def _validate_base_method(value: object) -> tuple[dict[str, Any], str]:
    method = _exact_mapping(value, _BASE_METHOD_FIELDS, name="Base@B receipt")
    if (
        method["schema_version"] != "cure-lite-stage-a-base-at-budget-receipt-v2"
        or method["method"] != "Base@B"
    ):
        raise ValueError("Base@B receipt protocol fields are invalid")
    protocol, protocol_fingerprint = _validate_protocol(
        method["protocol"],
        name="Base@B protocol",
        expected_variant="base_at_budget",
    )
    for field in (
        "d_v_base_run_fingerprint",
        *_COMMON_D_V_PROVENANCE,
    ):
        _digest(method[field], name=f"Base@B.{field}")
    fingerprint = _digest(
        method["receipt_fingerprint"],
        name="Base@B receipt_fingerprint",
    )
    expected = stable_fingerprint(
        {
            "schema_version": "cure-lite-formal-d-v-base-threshold-receipt-v1",
            "method": "Base@B",
            "threshold_protocol_fingerprint": protocol_fingerprint,
            "d_v_base_run_fingerprint": method["d_v_base_run_fingerprint"],
            **{field: method[field] for field in _COMMON_D_V_PROVENANCE},
        }
    )
    if fingerprint != expected:
        raise ValueError("Base@B receipt fingerprint mismatch")
    if protocol["sample_tensor_fingerprint"] == "":
        raise ValueError("Base@B sample tensor fingerprint is empty")
    return method, fingerprint


def _validate_residual_method(
    value: object,
    *,
    method_name: str,
    expected_variant: str,
    artifact: LoadedDecoderArtifact,
) -> tuple[dict[str, Any], str]:
    method = _exact_mapping(
        value,
        _RESIDUAL_METHOD_FIELDS,
        name=f"{method_name} receipt",
    )
    if (
        method["schema_version"] != "cure-lite-stage-a-formal-threshold-receipt-v3"
        or method["mode"] != "residual"
        or method["decoder_variant"] != expected_variant
    ):
        raise ValueError(f"{method_name} receipt protocol fields are invalid")
    protocol, protocol_fingerprint = _validate_protocol(
        method["protocol"],
        name=f"{method_name} protocol",
        expected_variant="residual",
    )
    for field in (
        "d_v_run_fingerprint",
        *_COMMON_D_V_PROVENANCE,
        "decoder_artifact_fingerprint",
        "decoder_receipt_sha256",
        "decoder_state_fingerprint",
    ):
        _digest(method[field], name=f"{method_name}.{field}")
    if isinstance(method["global_seed"], bool) or not isinstance(
        method["global_seed"], int
    ):
        raise TypeError(f"{method_name}.global_seed must be an integer")
    expected_artifact = {
        "decoder_artifact_fingerprint": artifact.artifact_fingerprint,
        "decoder_receipt_sha256": artifact.receipt_sha256,
        "decoder_state_fingerprint": artifact.decoder_state_fingerprint,
        "decoder_variant": artifact.config.variant,
        "global_seed": artifact.config.global_seed,
    }
    if any(method[field] != expected for field, expected in expected_artifact.items()):
        raise RuntimeError(f"{method_name} calibration does not bind its decoder")
    fingerprint = _digest(
        method["receipt_fingerprint"],
        name=f"{method_name}.receipt_fingerprint",
    )
    expected = stable_fingerprint(
        {
            "schema_version": "cure-lite-formal-d-v-threshold-receipt-v1",
            "mode": "residual",
            "threshold_protocol_fingerprint": protocol_fingerprint,
            "d_v_run_fingerprint": method["d_v_run_fingerprint"],
            **{field: method[field] for field in _COMMON_D_V_PROVENANCE},
            "decoder_artifact_fingerprint": method[
                "decoder_artifact_fingerprint"
            ],
            "decoder_receipt_sha256": method["decoder_receipt_sha256"],
            "decoder_state_fingerprint": method["decoder_state_fingerprint"],
            "decoder_variant": method["decoder_variant"],
            "global_seed": method["global_seed"],
        }
    )
    if fingerprint != expected:
        raise ValueError(f"{method_name} receipt fingerprint mismatch")
    return method, fingerprint


def _common_decoder_payload(
    artifacts: Mapping[str, LoadedDecoderArtifact],
) -> tuple[dict[str, Any], str]:
    common: dict[str, Any] | None = None
    initial: str | None = None
    for method, expected_variant in _DECODER_BY_METHOD.items():
        artifact = artifacts[method]
        if artifact.config.schema_version != DECODER_ARTIFACT_SCHEMA_V2:
            raise ValueError("historical Stage-A reference requires decoder artifact v2")
        if artifact.config.variant != expected_variant:
            raise ValueError(f"{method} decoder has the wrong variant")
        payload = artifact.config.canonical_payload()
        payload_variant = payload.pop("variant")
        payload.pop("variant_contract")
        if payload_variant != expected_variant:
            raise ValueError(f"{method} decoder payload has the wrong variant")
        if common is None:
            common = payload
            initial = artifact.config.initial_decoder_fingerprint
        elif payload != common:
            raise RuntimeError("F/F×/U decoder configs differ outside variant")
        if artifact.config.initial_decoder_fingerprint != initial:
            raise RuntimeError("F/F×/U decoders do not share one initialization")
    assert common is not None and initial is not None
    return common, initial


def _load_index(
    path: Path,
    *,
    expected_fields: frozenset[str],
    name: str,
) -> tuple[dict[str, Any], str, str]:
    payload = _exact_mapping(
        _strict_json(path, name=name),
        expected_fields,
        name=name,
    )
    fingerprint = _digest(
        payload["index_fingerprint"],
        name=f"{name}.index_fingerprint",
    )
    core = dict(payload)
    core.pop("index_fingerprint")
    if stable_fingerprint(core) != fingerprint:
        raise ValueError(f"{name} fingerprint mismatch")
    return payload, fingerprint, file_sha256(path)


def _validate_cache_index_bindings(
    root: Path,
    complete: Mapping[str, Any],
    anchor: Mapping[str, Any],
    reference_artifact: LoadedDecoderArtifact,
) -> None:
    d_r_base, d_r_base_fingerprint, d_r_base_sha = _load_index(
        root / "d_r" / "base_cache" / "index.json",
        expected_fields=_BASE_INDEX_FIELDS,
        name="D_R base index",
    )
    d_r_state, d_r_state_fingerprint, d_r_state_sha = _load_index(
        root / "d_r" / "state_cache" / "index.json",
        expected_fields=_STATE_INDEX_FIELDS,
        name="D_R state index",
    )
    d_v_base, d_v_base_fingerprint, d_v_base_sha = _load_index(
        root / "d_v" / "base_cache" / "index.json",
        expected_fields=_BASE_INDEX_FIELDS,
        name="D_V base index",
    )
    if (
        d_r_base["schema_version"] != "cure-lite-manifest-base-cache-index-v2"
        or d_v_base["schema_version"] != "cure-lite-manifest-base-cache-index-v2"
        or d_r_base["base_cache_schema"] != "cure-lite-base-cache-v2"
        or d_v_base["base_cache_schema"] != "cure-lite-base-cache-v2"
        or d_r_state["schema_version"]
        != "cure-lite-manifest-state-cache-index-v2"
        or d_r_state["state_cache_schema"] != "cure-lite-state-cache-v3"
        or d_r_base["split"] != "D_R"
        or d_r_state["split"] != "D_R"
        or d_v_base["split"] != "D_V"
    ):
        raise ValueError("historical cache index schemas/splits are invalid")
    if any(
        index["dataset"] != complete["dataset"]
        or index["split_manifest_fingerprint"] != complete["manifest_fingerprint"]
        or index["split_manifest_file_sha256"]
        != complete["manifest_file_sha256"]
        or index["preprocessing_fingerprint"]
        != complete["preprocessing_fingerprint"]
        or index["base_fingerprint"] != complete["base_fingerprint"]
        or index["base_state_fingerprint"] != complete["base_state_fingerprint"]
        for index in (d_r_base, d_r_state, d_v_base)
    ):
        raise RuntimeError("cache index provenance differs from COMPLETE")
    if (
        d_r_base_fingerprint != complete["d_r_base_index_fingerprint"]
        or d_r_state_fingerprint != complete["d_r_state_index_fingerprint"]
        or d_v_base_fingerprint != complete["d_v_base_index_fingerprint"]
    ):
        raise RuntimeError("cache index and COMPLETE fingerprints differ")
    run = reference_artifact.config
    if (
        run.base_index_fingerprint != d_r_base_fingerprint
        or run.base_index_sha256 != d_r_base_sha
        or run.state_index_fingerprint != d_r_state_fingerprint
        or run.state_index_sha256 != d_r_state_sha
        or run.state_fingerprint != d_r_state["state_fingerprint"]
        or run.gt_fingerprint != d_r_state["gt_fingerprint"]
    ):
        raise RuntimeError("decoder config differs from the historical D_R indexes")
    base_binding = _exact_mapping(
        d_r_state["base_index"],
        {"path", "sha256", "index_fingerprint"},
        name="D_R state base-index binding",
    )
    if (
        _digest(base_binding["sha256"], name="D_R bound base-index SHA")
        != d_r_base_sha
        or _digest(
            base_binding["index_fingerprint"],
            name="D_R bound base-index fingerprint",
        )
        != d_r_base_fingerprint
    ):
        raise RuntimeError("D_R state index binds another base index")
    if (
        d_r_state["occupancy_config"] != dict(anchor["occupancy_config"])
        or d_r_state["matching_config"]
        != reference_artifact.config.canonical_payload()["matching_config"]
        or d_r_state["intervention_config"]
        != reference_artifact.config.canonical_payload()["intervention_config"]
    ):
        raise RuntimeError("D_R state index mechanism config differs from decoder")
    if (
        anchor["base_index_fingerprint"] != d_v_base_fingerprint
        or anchor["base_index_sha256"] != d_v_base_sha
    ):
        raise RuntimeError("anchor differs from the historical D_V base index")


def _validate_decoder_stage_binding(
    artifacts: Mapping[str, LoadedDecoderArtifact],
    complete: Mapping[str, Any],
    config: StageARunConfig,
    anchor: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    common, initial = _common_decoder_payload(artifacts)
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
    reference = artifacts["F"].config
    if any(getattr(reference, name) != value for name, value in expected_training.items()):
        raise RuntimeError("decoder training config differs from Stage-A config")
    expected_source = {
        "manifest_fingerprint": complete["manifest_fingerprint"],
        "manifest_file_sha256": complete["manifest_file_sha256"],
        "preprocessing_fingerprint": complete["preprocessing_fingerprint"],
        "base_fingerprint": complete["base_fingerprint"],
        "base_index_fingerprint": complete["d_r_base_index_fingerprint"],
        "state_index_fingerprint": complete["d_r_state_index_fingerprint"],
    }
    if any(getattr(reference, name) != value for name, value in expected_source.items()):
        raise RuntimeError("decoder source bindings differ from COMPLETE")
    if (
        reference.match_config != config.match_config
        or reference.intervention_config != config.intervention_config
        or reference.occupancy_config.threshold != anchor["selected_threshold"]
        or reference.canonical_payload()["occupancy_config"]
        != dict(anchor["occupancy_config"])
    ):
        raise RuntimeError("decoder mechanism config differs from Stage-A receipts")
    for method, complete_field in _DECODER_COMPLETE_FIELDS.items():
        if artifacts[method].artifact_fingerprint != complete[complete_field]:
            raise RuntimeError(f"{method} decoder and COMPLETE fingerprints differ")
    return common, initial


def _validate_support(
    root: Path,
    complete: Mapping[str, Any],
    config: StageARunConfig,
) -> TrainingSupportSummary:
    receipt = _exact_mapping(
        _strict_json(root / "receipts" / "support.json", name="Stage-A support receipt"),
        {
            "schema_version",
            "split",
            "summary",
            "requirements",
            "requirements_met",
            "support_fingerprint",
        },
        name="Stage-A support receipt",
    )
    if (
        receipt["schema_version"] != _SUPPORT_SCHEMA
        or receipt["split"] != "D_R"
        or receipt["requirements_met"] is not True
    ):
        raise ValueError("Stage-A support receipt protocol fields are invalid")
    summary_payload = _exact_mapping(
        receipt["summary"],
        {
            "source_images",
            "factual_miss_images",
            "factual_no_miss_images",
            "factual_unreachable_images",
            "real_miss_targets",
            "reachable_miss_targets",
            "legal_candidates",
            "decoder_visible_legal_candidates",
            "synthetic_images",
            "visible_legal_fraction",
        },
        name="Stage-A support summary",
    )
    summary = TrainingSupportSummary(
        **{
            name: summary_payload[name]
            for name in (
                "source_images",
                "factual_miss_images",
                "factual_no_miss_images",
                "factual_unreachable_images",
                "real_miss_targets",
                "reachable_miss_targets",
                "legal_candidates",
                "decoder_visible_legal_candidates",
                "synthetic_images",
            )
        }
    )
    if summary.canonical_payload() != summary_payload:
        raise ValueError("Stage-A support summary is not canonical")
    requirements = receipt["requirements"]
    if requirements != config.support_requirements.canonical_payload():
        raise RuntimeError("support requirements differ from Stage-A config")
    if not isinstance(config.support_requirements, TrainingSupportRequirements):
        raise TypeError("Stage-A support requirements have an invalid type")
    config.support_requirements.require(summary)
    fingerprint = _digest(
        receipt["support_fingerprint"],
        name="support_fingerprint",
    )
    core = dict(receipt)
    core.pop("support_fingerprint")
    if stable_fingerprint(core) != fingerprint:
        raise ValueError("Stage-A support fingerprint mismatch")
    if fingerprint != complete["support_receipt_fingerprint"]:
        raise RuntimeError("support and COMPLETE fingerprints differ")
    return summary


def _validate_calibration(
    root: Path,
    complete: Mapping[str, Any],
    config: StageARunConfig,
    anchor: Mapping[str, Any],
    anchor_fingerprint: str,
    artifacts: Mapping[str, LoadedDecoderArtifact],
    common_decoder_payload: Mapping[str, Any],
    initial_decoder_fingerprint: str,
) -> tuple[dict[str, dict[str, Any]], str, str]:
    receipt = _exact_mapping(
        _strict_json(
            root / "receipts" / "calibration.json",
            name="Stage-A calibration receipt",
        ),
        {
            "schema_version",
            "method_order",
            "methods",
            "common_training_fingerprint",
            "receipt_fingerprint",
        },
        name="Stage-A calibration receipt",
    )
    if (
        receipt["schema_version"] != _CALIBRATION_SCHEMA
        or receipt["method_order"] != list(_METHOD_ORDER)
    ):
        raise ValueError("Stage-A calibration protocol fields are invalid")
    methods = _exact_mapping(
        receipt["methods"],
        set(_METHOD_ORDER),
        name="Stage-A calibration methods",
    )
    if methods["A"] != anchor:
        raise RuntimeError("calibration anchor differs from anchor receipt")
    base, base_fingerprint = _validate_base_method(methods["Base@B"])
    parsed: dict[str, dict[str, Any]] = {"A": dict(anchor), "Base@B": base}
    method_fingerprints: dict[str, str] = {"Base@B": base_fingerprint}
    for method_name, expected_variant in _DECODER_BY_METHOD.items():
        parsed_method, fingerprint = _validate_residual_method(
            methods[method_name],
            method_name=method_name,
            expected_variant=expected_variant,
            artifact=artifacts[method_name],
        )
        parsed[method_name] = parsed_method
        method_fingerprints[method_name] = fingerprint

    reference_protocol = parsed["Base@B"]["protocol"]
    for method_name in ("F", "F×", "U"):
        protocol = parsed[method_name]["protocol"]
        for field in (
            "manifest_fingerprint",
            "ordered_d_v_sample_ids",
            "occupancy_config",
            "match_config",
            "budget",
        ):
            if protocol[field] != reference_protocol[field]:
                raise RuntimeError("Base@B/F/F×/U do not share one D_V protocol")
        if protocol["candidate_threshold_grid"] != list(config.residual_thresholds):
            raise RuntimeError("residual threshold grid differs from Stage-A config")
    if reference_protocol["candidate_threshold_grid"] != list(config.base_thresholds):
        raise RuntimeError("Base@B threshold grid differs from Stage-A config")
    if (
        reference_protocol["manifest_fingerprint"] != complete["manifest_fingerprint"]
        or reference_protocol["ordered_d_v_sample_ids"]
        != anchor["ordered_d_v_sample_ids"]
        or reference_protocol["sample_tensor_fingerprint"]
        != anchor["base_samples_fingerprint"]
        or reference_protocol["occupancy_config"] != anchor["occupancy_config"]
        or reference_protocol["match_config"]
        != config.canonical_payload()["match_config"]
        or reference_protocol["budget"] != config.canonical_payload()["budget"]
    ):
        raise RuntimeError("calibration protocol differs from config/anchor/COMPLETE")
    for method_name in ("Base@B", "F", "F×", "U"):
        method = parsed[method_name]
        if any(
            method[field] != anchor[field]
            for field in _COMMON_D_V_PROVENANCE
        ):
            raise RuntimeError("calibration D_V provenance differs from anchor")

    common_fingerprint = _digest(
        receipt["common_training_fingerprint"],
        name="common_training_fingerprint",
    )
    expected_common = stable_fingerprint(
        {
            "schema_version": "cure-lite-paired-gate2-training-contract-v2",
            "common_run_config": dict(common_decoder_payload),
            "initial_decoder_fingerprint": initial_decoder_fingerprint,
            "d_v_base_samples_fingerprint": anchor["base_samples_fingerprint"],
        }
    )
    if common_fingerprint != expected_common:
        raise ValueError("common training fingerprint mismatch")

    fingerprint = _digest(
        receipt["receipt_fingerprint"],
        name="calibration receipt_fingerprint",
    )
    expected_fingerprint = stable_fingerprint(
        {
            "schema_version": "cure-lite-paired-gate2-calibration-v4",
            "anchor": anchor_fingerprint,
            "base_at_budget": method_fingerprints["Base@B"],
            "factual_only": method_fingerprints["F"],
            "factual_exposure_matched": method_fingerprints["F×"],
            "uniform_legal": method_fingerprints["U"],
            "common_training_fingerprint": common_fingerprint,
        }
    )
    if fingerprint != expected_fingerprint:
        raise ValueError("Stage-A calibration fingerprint mismatch")
    if fingerprint != complete["calibration_receipt_fingerprint"]:
        raise RuntimeError("calibration and COMPLETE fingerprints differ")
    return parsed, common_fingerprint, fingerprint


def _validate_results(
    root: Path,
    complete: Mapping[str, Any],
    calibration_methods: Mapping[str, Mapping[str, Any]],
    calibration_fingerprint: str,
) -> str:
    receipt = _exact_mapping(
        _strict_json(root / "receipts" / "results.json", name="Stage-A results receipt"),
        {
            "schema_version",
            "method_order",
            "methods",
            "calibration_receipt_fingerprint",
            "results_fingerprint",
        },
        name="Stage-A results receipt",
    )
    if (
        receipt["schema_version"] != _RESULTS_SCHEMA
        or receipt["method_order"] != list(_METHOD_ORDER)
    ):
        raise ValueError("Stage-A results protocol fields are invalid")
    methods = _exact_mapping(
        receipt["methods"],
        set(_METHOD_ORDER),
        name="Stage-A result methods",
    )
    for method_name in _METHOD_ORDER:
        result_metrics = _metrics(methods[method_name], name=f"results.{method_name}")
        if method_name == "A":
            selected_metrics = calibration_methods["A"]["selected_metrics"]
        else:
            selected_metrics = calibration_methods[method_name]["protocol"][
                "selected_metrics"
            ]
        if result_metrics != selected_metrics:
            raise RuntimeError(
                f"results.{method_name} differs from calibrated selected metrics"
            )
    bound_calibration = _digest(
        receipt["calibration_receipt_fingerprint"],
        name="results calibration_receipt_fingerprint",
    )
    if bound_calibration != calibration_fingerprint:
        raise RuntimeError("results bind another calibration receipt")
    fingerprint = _digest(
        receipt["results_fingerprint"],
        name="results_fingerprint",
    )
    core = dict(receipt)
    core.pop("results_fingerprint")
    if stable_fingerprint(core) != fingerprint:
        raise ValueError("Stage-A results fingerprint mismatch")
    if fingerprint != complete["results_fingerprint"]:
        raise RuntimeError("results and COMPLETE fingerprints differ")
    return fingerprint


def _validate_efficiency(
    root: Path,
    complete: Mapping[str, Any],
    uniform_artifact: LoadedDecoderArtifact,
) -> StageAEfficiencyReceipt:
    raw = _strict_json(
        root / "receipts" / "efficiency.json",
        name="Stage-A efficiency receipt",
    )
    receipt = StageAEfficiencyReceipt.from_mapping(raw)
    if receipt.receipt_fingerprint != complete["efficiency_receipt_fingerprint"]:
        raise RuntimeError("efficiency and COMPLETE fingerprints differ")
    binding = receipt.binding
    expected = {
        "decoder_artifact_fingerprint": uniform_artifact.artifact_fingerprint,
        "decoder_state_fingerprint": uniform_artifact.decoder_state_fingerprint,
        "decoder_receipt_sha256": uniform_artifact.receipt_sha256,
        "base_index_fingerprint": complete["d_v_base_index_fingerprint"],
        "preprocessing_fingerprint": complete["preprocessing_fingerprint"],
    }
    if any(getattr(binding, name) != value for name, value in expected.items()):
        raise RuntimeError("efficiency receipt does not bind the historical U snapshot")
    replay_static_efficiency(receipt, uniform_artifact.decoder)
    return receipt


@dataclass(frozen=True, slots=True)
class _ReferenceState:
    root: Path
    config: StageARunConfig
    support_summary: TrainingSupportSummary
    efficiency: StageAEfficiencyReceipt
    factual_artifact: LoadedDecoderArtifact
    factual_exposure_matched_artifact: LoadedDecoderArtifact
    uniform_artifact: LoadedDecoderArtifact
    source_tree_digest: str
    complete_fingerprint: str
    run_config_fingerprint: str
    anchor_receipt_fingerprint: str
    support_receipt_fingerprint: str
    calibration_receipt_fingerprint: str
    results_fingerprint: str
    efficiency_receipt_fingerprint: str
    common_training_fingerprint: str
    manifest_fingerprint: str
    manifest_file_sha256: str
    preprocessing_fingerprint: str
    base_fingerprint: str
    base_state_fingerprint: str
    d_r_base_index_fingerprint: str
    d_r_state_index_fingerprint: str
    d_v_base_index_fingerprint: str
    artifact_inventory_fingerprint: str
    snapshot_fingerprint: str

    def signature(self) -> tuple[object, ...]:
        return (
            self.config,
            self.support_summary,
            self.efficiency.canonical_payload(),
            self.factual_artifact.artifact_fingerprint,
            self.factual_exposure_matched_artifact.artifact_fingerprint,
            self.uniform_artifact.artifact_fingerprint,
            self.source_tree_digest,
            self.complete_fingerprint,
            self.run_config_fingerprint,
            self.anchor_receipt_fingerprint,
            self.support_receipt_fingerprint,
            self.calibration_receipt_fingerprint,
            self.results_fingerprint,
            self.efficiency_receipt_fingerprint,
            self.common_training_fingerprint,
            self.manifest_fingerprint,
            self.manifest_file_sha256,
            self.preprocessing_fingerprint,
            self.base_fingerprint,
            self.base_state_fingerprint,
            self.d_r_base_index_fingerprint,
            self.d_r_state_index_fingerprint,
            self.d_v_base_index_fingerprint,
            self.artifact_inventory_fingerprint,
            self.snapshot_fingerprint,
        )


def _load_reference_state(root: Path) -> _ReferenceState:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Stage-A reference root must be a regular non-symlink directory")
    incomplete = root / _INCOMPLETE_NAME
    if incomplete.exists() or incomplete.is_symlink():
        raise RuntimeError("Stage-A reference is incomplete")
    complete = _exact_mapping(
        _strict_json(root / _COMPLETE_NAME, name="Stage-A COMPLETE receipt"),
        _COMPLETE_FIELDS,
        name="Stage-A COMPLETE receipt",
    )
    _validate_complete_protocol(complete)
    complete_core = dict(complete)
    complete_fingerprint = _digest(
        complete_core.pop("complete_fingerprint"),
        name="complete_fingerprint",
    )
    if stable_fingerprint(complete_core) != complete_fingerprint:
        raise ValueError("Stage-A COMPLETE fingerprint mismatch")

    directories, files = _tree_inventory(root)
    if complete["artifact_directories"] != directories:
        raise ValueError("Stage-A artifact directory inventory changed")
    if complete["artifact_files"] != files:
        raise ValueError("Stage-A artifact file inventory changed")
    inventory_fingerprint = stable_fingerprint(
        {"directories": directories, "files": files}
    )

    config = _load_config_receipt(root, complete)
    anchor, anchor_fingerprint = _validate_anchor(root, complete, config)
    support_summary = _validate_support(root, complete, config)

    artifacts = {
        "F": load_decoder_artifact(root / "decoders" / "factual_only"),
        "F×": load_decoder_artifact(
            root / "decoders" / "factual_exposure_matched"
        ),
        "U": load_decoder_artifact(root / "decoders" / "uniform_legal"),
    }
    common_payload, initial_fingerprint = _validate_decoder_stage_binding(
        artifacts,
        complete,
        config,
        anchor,
    )
    _validate_cache_index_bindings(
        root,
        complete,
        anchor,
        artifacts["F"],
    )
    calibration_methods, common_training_fingerprint, calibration_fingerprint = (
        _validate_calibration(
            root,
            complete,
            config,
            anchor,
            anchor_fingerprint,
            artifacts,
            common_payload,
            initial_fingerprint,
        )
    )
    results_fingerprint = _validate_results(
        root,
        complete,
        calibration_methods,
        calibration_fingerprint,
    )
    efficiency = _validate_efficiency(root, complete, artifacts["U"])

    snapshot_core = {
        "schema_version": REFERENCE_SNAPSHOT_SCHEMA,
        "complete_fingerprint": complete_fingerprint,
        "source_tree_digest": complete["source_tree_digest"],
        "run_config_fingerprint": complete["run_config_fingerprint"],
        "anchor_receipt_fingerprint": anchor_fingerprint,
        "support_receipt_fingerprint": complete["support_receipt_fingerprint"],
        "calibration_receipt_fingerprint": calibration_fingerprint,
        "results_fingerprint": results_fingerprint,
        "efficiency_receipt_fingerprint": efficiency.receipt_fingerprint,
        "common_training_fingerprint": common_training_fingerprint,
        "decoder_artifact_fingerprints": {
            method: artifacts[method].artifact_fingerprint
            for method in ("F", "F×", "U")
        },
        "artifact_inventory_fingerprint": inventory_fingerprint,
    }
    snapshot_fingerprint = stable_fingerprint(snapshot_core)
    return _ReferenceState(
        root=root,
        config=config,
        support_summary=support_summary,
        efficiency=efficiency,
        factual_artifact=artifacts["F"],
        factual_exposure_matched_artifact=artifacts["F×"],
        uniform_artifact=artifacts["U"],
        source_tree_digest=complete["source_tree_digest"],
        complete_fingerprint=complete_fingerprint,
        run_config_fingerprint=complete["run_config_fingerprint"],
        anchor_receipt_fingerprint=anchor_fingerprint,
        support_receipt_fingerprint=complete["support_receipt_fingerprint"],
        calibration_receipt_fingerprint=calibration_fingerprint,
        results_fingerprint=results_fingerprint,
        efficiency_receipt_fingerprint=efficiency.receipt_fingerprint,
        common_training_fingerprint=common_training_fingerprint,
        manifest_fingerprint=complete["manifest_fingerprint"],
        manifest_file_sha256=complete["manifest_file_sha256"],
        preprocessing_fingerprint=complete["preprocessing_fingerprint"],
        base_fingerprint=complete["base_fingerprint"],
        base_state_fingerprint=complete["base_state_fingerprint"],
        d_r_base_index_fingerprint=complete["d_r_base_index_fingerprint"],
        d_r_state_index_fingerprint=complete["d_r_state_index_fingerprint"],
        d_v_base_index_fingerprint=complete["d_v_base_index_fingerprint"],
        artifact_inventory_fingerprint=inventory_fingerprint,
        snapshot_fingerprint=snapshot_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class _ReferenceSeal:
    state: _ReferenceState


@dataclass(frozen=True, slots=True)
class StageAReferenceSnapshot:
    """Sealed, read-only reference to one completed historical v0.1 run."""

    root: Path
    config: StageARunConfig
    support_summary: TrainingSupportSummary
    efficiency: StageAEfficiencyReceipt
    factual_artifact: LoadedDecoderArtifact
    factual_exposure_matched_artifact: LoadedDecoderArtifact
    uniform_artifact: LoadedDecoderArtifact
    source_tree_digest: str
    complete_fingerprint: str
    common_training_fingerprint: str
    artifact_inventory_fingerprint: str
    snapshot_fingerprint: str
    _verification_token: object

    def _verify_binding(self) -> _ReferenceState:
        seal = self._verification_token
        if type(seal) is not _ReferenceSeal:
            raise TypeError(
                "StageAReferenceSnapshot must come from "
                "load_stage_a_reference_snapshot"
            )
        state = seal.state
        if (
            self.root != state.root
            or self.config is not state.config
            or self.support_summary is not state.support_summary
            or self.efficiency is not state.efficiency
            or self.factual_artifact is not state.factual_artifact
            or self.factual_exposure_matched_artifact
            is not state.factual_exposure_matched_artifact
            or self.uniform_artifact is not state.uniform_artifact
            or self.source_tree_digest != state.source_tree_digest
            or self.complete_fingerprint != state.complete_fingerprint
            or self.common_training_fingerprint
            != state.common_training_fingerprint
            or self.artifact_inventory_fingerprint
            != state.artifact_inventory_fingerprint
            or self.snapshot_fingerprint != state.snapshot_fingerprint
        ):
            raise TypeError("Stage-A reference snapshot fields were replaced")
        return state

    def __post_init__(self) -> None:
        if not self.root.is_absolute():
            raise ValueError("Stage-A reference root must be absolute")
        for name in (
            "source_tree_digest",
            "complete_fingerprint",
            "common_training_fingerprint",
            "artifact_inventory_fingerprint",
            "snapshot_fingerprint",
        ):
            _digest(getattr(self, name), name=name)
        self._verify_binding()

    def verify_unchanged(self) -> None:
        """Recheck every persisted byte and all historical cross-bindings."""

        state = self._verify_binding()
        self.factual_artifact.verify_unchanged()
        self.factual_exposure_matched_artifact.verify_unchanged()
        self.uniform_artifact.verify_unchanged()
        reloaded = _load_reference_state(self.root)
        if reloaded.signature() != state.signature():
            raise RuntimeError("historical Stage-A reference snapshot changed")

    def verify(self) -> None:
        """Alias for :meth:`verify_unchanged`."""

        self.verify_unchanged()


def load_stage_a_reference_snapshot(
    output_dir: str | Path,
) -> StageAReferenceSnapshot:
    """Strictly load a completed v0.1 Stage-A run as a historical snapshot.

    The requested root itself may not be a symlink.  The recorded historical
    source digest is checked across receipts but is intentionally not compared
    with the current CURE-Lite source tree.
    """

    requested = Path(output_dir).expanduser()
    if requested.is_symlink():
        raise ValueError("Stage-A reference root may not be addressed through a symlink")
    root = requested.resolve(strict=True)
    state = _load_reference_state(root)
    seal = _ReferenceSeal(state=state)
    return StageAReferenceSnapshot(
        root=state.root,
        config=state.config,
        support_summary=state.support_summary,
        efficiency=state.efficiency,
        factual_artifact=state.factual_artifact,
        factual_exposure_matched_artifact=(
            state.factual_exposure_matched_artifact
        ),
        uniform_artifact=state.uniform_artifact,
        source_tree_digest=state.source_tree_digest,
        complete_fingerprint=state.complete_fingerprint,
        common_training_fingerprint=state.common_training_fingerprint,
        artifact_inventory_fingerprint=state.artifact_inventory_fingerprint,
        snapshot_fingerprint=state.snapshot_fingerprint,
        _verification_token=seal,
    )


__all__ = [
    "REFERENCE_SNAPSHOT_SCHEMA",
    "StageAReferenceSnapshot",
    "load_stage_a_reference_snapshot",
]
