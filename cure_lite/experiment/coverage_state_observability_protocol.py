"""Frozen protocol for the D_R-only CSLF observability gate."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

from ..cache.schema import stable_fingerprint
from ..coverage_state_level_set import (
    CSLF_FEATURE_POLICY,
    CSLF_NORMALIZATION_EPSILON,
    CSLF_TARGET_POLICY,
)
from ..coverage_state_observability import (
    COVERAGE_STATE_FEATURE_RADIUS,
    COVERAGE_STATE_OBSERVABILITY_SCHEMA,
)
from ..coverage_state_raw_catalog import (
    COVERAGE_STATE_NATURAL_FOCUS_POLICY,
    COVERAGE_STATE_RAW_CATALOG_SCHEMA,
    COVERAGE_STATE_SCENE_TARGET_POLICY,
)


COVERAGE_STATE_OBSERVABILITY_CONFIG_SCHEMA = (
    "cure-lite-coverage-state-observability-config-v1"
)


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return dict(value)


def _exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} must contain exactly {sorted(expected)}, "
            f"got {sorted(value)}"
        )


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, *, name: str) -> str:
    text = _text(value, name=name)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 fingerprint")
    return text


@dataclass(frozen=True)
class CoverageStateObservabilityInputBinding:
    manifest_file_sha256: str
    state_index_sha256: str
    state_index_fingerprint: str
    base_fingerprint: str
    base_state_fingerprint: str
    state_fingerprint: str
    gt_fingerprint: str
    geometry_protocol_config_file_sha256: str
    geometry_protocol_config_fingerprint: str
    geometry_catalog_receipt_file_sha256: str
    geometry_catalog_fingerprint: str

    @classmethod
    def from_mapping(
        cls,
        value: object,
    ) -> "CoverageStateObservabilityInputBinding":
        payload = _mapping(value, name="input_binding")
        expected = {
            "manifest_file_sha256",
            "state_index_sha256",
            "state_index_fingerprint",
            "base_fingerprint",
            "base_state_fingerprint",
            "state_fingerprint",
            "gt_fingerprint",
            "geometry_protocol_config_file_sha256",
            "geometry_protocol_config_fingerprint",
            "geometry_catalog_receipt_file_sha256",
            "geometry_catalog_fingerprint",
        }
        _exact_keys(payload, expected, name="input_binding")
        normalized = {
            name: _sha256(value, name=f"input_binding.{name}")
            for name, value in payload.items()
        }
        return cls(**normalized)


@dataclass(frozen=True)
class CoverageStateObservabilityProtocol:
    schema_version: str
    protocol_id: str
    dataset: str
    split: str
    input_binding: CoverageStateObservabilityInputBinding
    model_contract: dict[str, object]
    gate: dict[str, object]
    execution_policy: dict[str, object]

    @classmethod
    def from_mapping(
        cls,
        value: object,
    ) -> "CoverageStateObservabilityProtocol":
        payload = _mapping(value, name="protocol")
        _exact_keys(
            payload,
            {
                "schema_version",
                "protocol_id",
                "dataset",
                "split",
                "input_binding",
                "model_contract",
                "gate",
                "execution_policy",
            },
            name="protocol",
        )
        if payload["schema_version"] != (
            COVERAGE_STATE_OBSERVABILITY_CONFIG_SCHEMA
        ):
            raise ValueError("observability config schema changed")
        protocol_id = _text(payload["protocol_id"], name="protocol_id")
        dataset = _text(payload["dataset"], name="dataset")
        if payload["split"] != "D_R":
            raise ValueError("observability protocol permits only D_R")
        contract = _mapping(payload["model_contract"], name="model_contract")
        expected_contract = {
            "raw_catalog_schema": COVERAGE_STATE_RAW_CATALOG_SCHEMA,
            "observability_schema": COVERAGE_STATE_OBSERVABILITY_SCHEMA,
            "scene_target_policy": COVERAGE_STATE_SCENE_TARGET_POLICY,
            "natural_focus_policy": COVERAGE_STATE_NATURAL_FOCUS_POLICY,
            "feature_policy": CSLF_FEATURE_POLICY,
            "target_field_policy": CSLF_TARGET_POLICY,
            "normalization_epsilon_hex": (
                CSLF_NORMALIZATION_EPSILON.hex()
            ),
            "field_valid_policy": "image_valid_domain_v1",
            "natural_loss_valid_policy": (
                "writable_background_plus_focus_target_v1"
            ),
            "pair_valid_policy": "image_valid_domain_v1",
            "feature_receptive_radius": COVERAGE_STATE_FEATURE_RADIUS,
            "representations": ["scalar_max", "phase_preserving"],
            "duplicate_conflict_domain": (
                "actual_input_group_common_field_valid_domain_v1"
            ),
        }
        if contract != expected_contract:
            raise ValueError("observability model_contract differs from frozen policy")
        gate = _mapping(payload["gate"], name="gate")
        expected_gate = {
            "identity_null_nonidentical_count": 0,
            "phase_duplicate_input_target_conflicts": 0,
            "target_response_outside_phase_rf_pixels": 0,
            "pp_trigger": [
                "scalar_duplicate_input_target_conflicts > 0",
                "target_response_outside_scalar_rf_pixels > 0",
            ],
            "scalar_authorization": [
                "scalar_duplicate_input_target_conflicts == 0",
                "target_response_outside_scalar_rf_pixels == 0",
            ],
            "zero_response_scalar_hidden_pair_triggers_pp": False,
        }
        if gate != expected_gate:
            raise ValueError("observability gate differs from frozen decision")
        execution = _mapping(
            payload["execution_policy"],
            name="execution_policy",
        )
        expected_execution = {
            "allowed_runtime_splits": ["D_R"],
            "create_only_output": True,
            "allow_training": False,
            "allow_calibration": False,
            "allow_inference": False,
            "allow_d_v_evaluation": False,
            "allow_d_t_evaluation": False,
            "allow_backbone_integration": False,
        }
        if execution != expected_execution:
            raise ValueError("observability execution policy changed")
        return cls(
            schema_version=COVERAGE_STATE_OBSERVABILITY_CONFIG_SCHEMA,
            protocol_id=protocol_id,
            dataset=dataset,
            split="D_R",
            input_binding=(
                CoverageStateObservabilityInputBinding.from_mapping(
                    payload["input_binding"]
                )
            ),
            model_contract=contract,
            gate=gate,
            execution_policy=execution,
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "dataset": self.dataset,
            "split": self.split,
            "input_binding": {
                name: getattr(self.input_binding, name)
                for name in self.input_binding.__dataclass_fields__
            },
            "model_contract": dict(self.model_contract),
            "gate": dict(self.gate),
            "execution_policy": dict(self.execution_policy),
        }

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def load_coverage_state_observability_protocol(
    path: str | Path,
) -> CoverageStateObservabilityProtocol:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError("observability config may not be a symbolic link")
    resolved = source.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("observability config must be a regular file")

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"observability config contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(
                    f"observability config contains non-finite value {value}"
                )
            ),
        )
    return CoverageStateObservabilityProtocol.from_mapping(payload)


__all__ = [
    "COVERAGE_STATE_OBSERVABILITY_CONFIG_SCHEMA",
    "CoverageStateObservabilityInputBinding",
    "CoverageStateObservabilityProtocol",
    "load_coverage_state_observability_protocol",
]
