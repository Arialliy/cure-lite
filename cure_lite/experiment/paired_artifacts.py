"""Strict create-only artifacts for formal paired CURE-Lite decoders.

The historical decoder artifact schemas describe the old independent
synthetic branch.  A paired-difference run has different training semantics,
so it receives a separate schema instead of being disguised as an old
``uniform_legal`` variant.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import torch

from ..cache.schema import canonical_json, file_sha256, stable_fingerprint
from ..config import (
    DecoderConfig,
    InterventionConfig,
    LossConfig,
    MatchConfig,
    OccupancyConfig,
    config_to_dict,
)
from ..decoder import CURELiteDecoder
from ..train.paired_control_step import CONTROL_KINDS
from .artifacts import decoder_state_fingerprint


PAIRED_DECODER_ARTIFACT_SCHEMA = "cure-lite-paired-decoder-artifact-v1"
PAIRED_EXECUTION_LEDGER_SCHEMA = "cure-lite-paired-execution-ledger-v1"
PAIRED_METHODS = ("paired_difference", *CONTROL_KINDS)
_WEIGHTS_NAME = "decoder.safetensors"
_LOG_NAME = "train_log.json"
_LEDGER_NAME = "execution_ledger.json"
_RECEIPT_NAME = "receipt.json"
_HEX = frozenset("0123456789abcdef")
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "run_config",
        "weights_file",
        "weights_sha256",
        "decoder_state_fingerprint",
        "train_log_file",
        "train_log_sha256",
        "execution_ledger_file",
        "execution_ledger_sha256",
        "artifact_fingerprint",
    }
)


def method_objective_contract(method: str) -> dict[str, object]:
    """Return the exact third-branch objective semantics for one method."""

    if method not in PAIRED_METHODS:
        raise ValueError(f"method must be one of {PAIRED_METHODS}")
    if method == "independent_endpoint":
        return {
            "family": "independent_absolute_endpoint_erm",
            "loss_id": (
                "geometry_matched_independent_endpoint_absolute_erm_v1"
            ),
            "input_transform": "none",
            "gradient_path": "plus_and_minus_independent",
            "coupled_difference_value": False,
            "both_endpoints_receive_paired_gradient": False,
        }
    if method == "after_only":
        return {
            "family": "after_only_absolute_synthetic",
            "loss_id": "after_only_absolute_synthetic_v1",
            "input_transform": "none",
            "gradient_path": "minus_only",
            "coupled_difference_value": False,
            "both_endpoints_receive_paired_gradient": False,
        }
    input_transforms = {
        "paired_difference": "none",
        "zero_feature": "zero_feature",
        "coordinate_basis": "fixed_dct_coordinate_basis",
        "feature_only": "zero_occupancy_both_endpoints",
        "target_permutation": "source_disjoint_target_permutation",
        "plus_detach": "none",
        "minus_detach": "none",
    }
    gradient_paths = {
        "paired_difference": "plus_and_minus",
        "zero_feature": "plus_and_minus",
        "coordinate_basis": "plus_and_minus",
        "feature_only": "plus_and_minus",
        "target_permutation": "plus_and_minus",
        "plus_detach": "minus_only_plus_value_detached",
        "minus_detach": "plus_only_minus_value_detached",
    }
    return {
        "family": "coupled_paired_difference",
        "loss_id": "balanced_pre_mask_score_difference_regression_v1",
        "input_transform": input_transforms[method],
        "gradient_path": gradient_paths[method],
        "coupled_difference_value": True,
        "both_endpoints_receive_paired_gradient": method
        not in {"plus_detach", "minus_detach"},
    }


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(
    value: object,
    *,
    name: str,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result) or result < 0.0 or (positive and result == 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


@dataclass(frozen=True)
class PairedDecoderRunConfig:
    """Complete frozen identity of one proposed/control formal run."""

    method: str
    seed: int
    manifest_fingerprint: str
    manifest_file_sha256: str
    preprocessing_fingerprint: str
    base_fingerprint: str
    state_fingerprint: str
    gt_fingerprint: str
    base_index_fingerprint: str
    base_index_sha256: str
    state_index_fingerprint: str
    state_index_sha256: str
    formal_protocol_fingerprint: str
    paired_objective_fingerprint: str
    pair_catalog_fingerprint: str
    paired_schedule_fingerprint: str
    formal_schedule_fingerprint: str
    runtime_input_fingerprint: str
    control_preflight_fingerprint: str
    control_provider_fingerprint: str | None
    method_contract_fingerprint: str
    initial_decoder_fingerprint: str
    occupancy_config: OccupancyConfig
    match_config: MatchConfig
    intervention_config: InterventionConfig
    decoder_config: DecoderConfig
    absolute_loss_config: LossConfig = LossConfig()
    paired_loss_id: str = "balanced_pre_mask_score_difference_regression_v1"
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 800
    steps_per_epoch: int = 40
    factual_miss_batch: int = 4
    factual_no_miss_batch: int = 4
    pair_batch: int = 2
    objective_coefficients: tuple[float, float, float] = (1.0, 1.0, 1.0)
    schema_version: str = PAIRED_DECODER_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PAIRED_DECODER_ARTIFACT_SCHEMA:
            raise ValueError("unsupported paired decoder artifact schema")
        if self.method not in PAIRED_METHODS:
            raise ValueError(f"method must be one of {PAIRED_METHODS}")
        if self.seed not in (42, 43):
            raise ValueError("formal paired seed must be 42 or 43")
        for name in (
            "manifest_fingerprint",
            "manifest_file_sha256",
            "preprocessing_fingerprint",
            "base_fingerprint",
            "state_fingerprint",
            "gt_fingerprint",
            "base_index_fingerprint",
            "base_index_sha256",
            "state_index_fingerprint",
            "state_index_sha256",
            "formal_protocol_fingerprint",
            "paired_objective_fingerprint",
            "pair_catalog_fingerprint",
            "paired_schedule_fingerprint",
            "formal_schedule_fingerprint",
            "runtime_input_fingerprint",
            "control_preflight_fingerprint",
            "method_contract_fingerprint",
            "initial_decoder_fingerprint",
        ):
            _digest(getattr(self, name), name=name)
        if self.method == "paired_difference":
            if self.control_provider_fingerprint is not None:
                raise ValueError(
                    "paired_difference cannot bind a control provider"
                )
        else:
            _digest(
                self.control_provider_fingerprint,
                name="control_provider_fingerprint",
            )
        expected_method_contract = stable_fingerprint(
            method_objective_contract(self.method)
        )
        if self.method_contract_fingerprint != expected_method_contract:
            raise ValueError(
                "method_contract_fingerprint differs from the method objective"
            )
        if not isinstance(self.occupancy_config, OccupancyConfig):
            raise TypeError("occupancy_config must be OccupancyConfig")
        if not isinstance(self.match_config, MatchConfig):
            raise TypeError("match_config must be MatchConfig")
        if not isinstance(self.intervention_config, InterventionConfig):
            raise TypeError("intervention_config must be InterventionConfig")
        if not isinstance(self.decoder_config, DecoderConfig):
            raise TypeError("decoder_config must be DecoderConfig")
        if not isinstance(self.absolute_loss_config, LossConfig):
            raise TypeError("absolute_loss_config must be LossConfig")
        if self.paired_loss_id != (
            "balanced_pre_mask_score_difference_regression_v1"
        ):
            raise ValueError("paired_loss_id differs from the frozen objective")
        if self.optimizer != "adam":
            raise ValueError("formal paired optimizer must remain adam")
        object.__setattr__(
            self,
            "learning_rate",
            _finite_nonnegative(
                self.learning_rate,
                name="learning_rate",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "weight_decay",
            _finite_nonnegative(self.weight_decay, name="weight_decay"),
        )
        exact_integers = {
            "epochs": 800,
            "steps_per_epoch": 40,
            "factual_miss_batch": 4,
            "factual_no_miss_batch": 4,
            "pair_batch": 2,
        }
        for name, expected in exact_integers.items():
            value = _positive_int(getattr(self, name), name=name)
            if value != expected:
                raise ValueError(f"{name} must remain {expected}")
        normalized_coefficients = tuple(
            _finite_nonnegative(value, name="objective coefficient")
            for value in self.objective_coefficients
        )
        if normalized_coefficients != (1.0, 1.0, 1.0):
            raise ValueError("formal objective coefficients must remain 1:1:1")
        object.__setattr__(
            self,
            "objective_coefficients",
            normalized_coefficients,
        )

    @property
    def global_seed(self) -> int:
        """Compatibility alias used by the frozen single-method evaluator."""

        return self.seed

    @property
    def variant(self) -> str:
        """Compatibility alias; never maps paired methods to old variants."""

        return self.method

    @property
    def optimizer_updates(self) -> int:
        return self.epochs * self.steps_per_epoch

    @property
    def decoder_states_per_update(self) -> int:
        return (
            self.factual_miss_batch
            + self.factual_no_miss_batch
            + 2 * self.pair_batch
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "seed": self.seed,
            "manifest_fingerprint": self.manifest_fingerprint,
            "manifest_file_sha256": self.manifest_file_sha256,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "base_fingerprint": self.base_fingerprint,
            "state_fingerprint": self.state_fingerprint,
            "gt_fingerprint": self.gt_fingerprint,
            "base_index_fingerprint": self.base_index_fingerprint,
            "base_index_sha256": self.base_index_sha256,
            "state_index_fingerprint": self.state_index_fingerprint,
            "state_index_sha256": self.state_index_sha256,
            "formal_protocol_fingerprint": self.formal_protocol_fingerprint,
            "paired_objective_fingerprint": self.paired_objective_fingerprint,
            "pair_catalog_fingerprint": self.pair_catalog_fingerprint,
            "paired_schedule_fingerprint": self.paired_schedule_fingerprint,
            "formal_schedule_fingerprint": self.formal_schedule_fingerprint,
            "runtime_input_fingerprint": self.runtime_input_fingerprint,
            "control_preflight_fingerprint": (
                self.control_preflight_fingerprint
            ),
            "control_provider_fingerprint": (
                self.control_provider_fingerprint
            ),
            "method_contract_fingerprint": self.method_contract_fingerprint,
            "initial_decoder_fingerprint": self.initial_decoder_fingerprint,
            "occupancy_config": config_to_dict(self.occupancy_config),
            "matching_config": config_to_dict(self.match_config),
            "intervention_config": config_to_dict(self.intervention_config),
            "decoder_config": config_to_dict(self.decoder_config),
            "absolute_loss_config": config_to_dict(
                self.absolute_loss_config
            ),
            "paired_criterion": {
                "id": self.paired_loss_id,
                "difference_domain": "raw_pre_hard_mask_score",
                "positive_zero_stratum_weights": [0.5, 0.5],
                "used_by_third_branch": self.method
                not in {"independent_endpoint", "after_only"},
            },
            "method_objective": method_objective_contract(self.method),
            "optimization": {
                "optimizer": self.optimizer,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
            },
            "horizon": {
                "epochs": self.epochs,
                "steps_per_epoch": self.steps_per_epoch,
                "optimizer_updates": self.optimizer_updates,
            },
            "per_update": {
                "factual_miss_states": self.factual_miss_batch,
                "factual_no_miss_states": self.factual_no_miss_batch,
                "pairs": self.pair_batch,
                "paired_endpoint_states": 2 * self.pair_batch,
                "decoder_states": self.decoder_states_per_update,
                "decoder_forward_calls": 3,
                "backward_calls": 1,
                "optimizer_steps": 1,
            },
            "objective_coefficients": {
                "factual_miss": self.objective_coefficients[0],
                "factual_no_miss": self.objective_coefficients[1],
                "paired_or_control": self.objective_coefficients[2],
            },
            "frozen_base_cache_only": True,
            "checkpoint_resume": False,
            "intermediate_checkpoint_written": False,
            "training_split": "D_R",
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "PairedDecoderRunConfig":
        if not isinstance(value, Mapping):
            raise TypeError("paired run config must be a mapping")
        required = {
            "schema_version",
            "method",
            "seed",
            "manifest_fingerprint",
            "manifest_file_sha256",
            "preprocessing_fingerprint",
            "base_fingerprint",
            "state_fingerprint",
            "gt_fingerprint",
            "base_index_fingerprint",
            "base_index_sha256",
            "state_index_fingerprint",
            "state_index_sha256",
            "formal_protocol_fingerprint",
            "paired_objective_fingerprint",
            "pair_catalog_fingerprint",
            "paired_schedule_fingerprint",
            "formal_schedule_fingerprint",
            "runtime_input_fingerprint",
            "control_preflight_fingerprint",
            "control_provider_fingerprint",
            "method_contract_fingerprint",
            "initial_decoder_fingerprint",
            "occupancy_config",
            "matching_config",
            "intervention_config",
            "decoder_config",
            "absolute_loss_config",
            "paired_criterion",
            "method_objective",
            "optimization",
            "horizon",
            "per_update",
            "objective_coefficients",
            "frozen_base_cache_only",
            "checkpoint_resume",
            "intermediate_checkpoint_written",
            "training_split",
        }
        if set(value) != required:
            raise ValueError("paired run config fields are not canonical")
        mappings = {
            name: value[name]
            for name in (
                "occupancy_config",
                "matching_config",
                "intervention_config",
                "decoder_config",
                "absolute_loss_config",
                "paired_criterion",
                "method_objective",
                "optimization",
                "horizon",
                "per_update",
                "objective_coefficients",
            )
        }
        if any(not isinstance(item, Mapping) for item in mappings.values()):
            raise TypeError("paired run config sections must be mappings")
        paired_criterion = mappings["paired_criterion"]
        method_objective = mappings["method_objective"]
        optimization = mappings["optimization"]
        horizon = mappings["horizon"]
        per_update = mappings["per_update"]
        coefficients = mappings["objective_coefficients"]
        if paired_criterion != {
            "id": "balanced_pre_mask_score_difference_regression_v1",
            "difference_domain": "raw_pre_hard_mask_score",
            "positive_zero_stratum_weights": [0.5, 0.5],
            "used_by_third_branch": value["method"]
            not in {"independent_endpoint", "after_only"},
        }:
            raise ValueError("paired criterion payload differs from the freeze")
        if method_objective != method_objective_contract(value["method"]):
            raise ValueError("method objective payload differs from the freeze")
        if value["frozen_base_cache_only"] is not True:
            raise ValueError("formal paired run must use only frozen Base cache")
        if (
            value["checkpoint_resume"] is not False
            or value["intermediate_checkpoint_written"] is not False
        ):
            raise ValueError("paired artifact cannot describe recovery/checkpoints")
        if value["training_split"] != "D_R":
            raise ValueError("formal paired training split must be D_R")
        config = cls(
            schema_version=value["schema_version"],
            method=value["method"],
            seed=value["seed"],
            manifest_fingerprint=value["manifest_fingerprint"],
            manifest_file_sha256=value["manifest_file_sha256"],
            preprocessing_fingerprint=value["preprocessing_fingerprint"],
            base_fingerprint=value["base_fingerprint"],
            state_fingerprint=value["state_fingerprint"],
            gt_fingerprint=value["gt_fingerprint"],
            base_index_fingerprint=value["base_index_fingerprint"],
            base_index_sha256=value["base_index_sha256"],
            state_index_fingerprint=value["state_index_fingerprint"],
            state_index_sha256=value["state_index_sha256"],
            formal_protocol_fingerprint=value["formal_protocol_fingerprint"],
            paired_objective_fingerprint=value["paired_objective_fingerprint"],
            pair_catalog_fingerprint=value["pair_catalog_fingerprint"],
            paired_schedule_fingerprint=value["paired_schedule_fingerprint"],
            formal_schedule_fingerprint=value["formal_schedule_fingerprint"],
            runtime_input_fingerprint=value["runtime_input_fingerprint"],
            control_preflight_fingerprint=value[
                "control_preflight_fingerprint"
            ],
            control_provider_fingerprint=value[
                "control_provider_fingerprint"
            ],
            method_contract_fingerprint=value["method_contract_fingerprint"],
            initial_decoder_fingerprint=value["initial_decoder_fingerprint"],
            occupancy_config=OccupancyConfig(
                **dict(mappings["occupancy_config"])
            ),
            match_config=MatchConfig(**dict(mappings["matching_config"])),
            intervention_config=InterventionConfig(
                **dict(mappings["intervention_config"])
            ),
            decoder_config=DecoderConfig(**dict(mappings["decoder_config"])),
            absolute_loss_config=LossConfig(
                **dict(mappings["absolute_loss_config"])
            ),
            paired_loss_id=paired_criterion["id"],
            optimizer=optimization["optimizer"],
            learning_rate=optimization["learning_rate"],
            weight_decay=optimization["weight_decay"],
            epochs=horizon["epochs"],
            steps_per_epoch=horizon["steps_per_epoch"],
            factual_miss_batch=per_update["factual_miss_states"],
            factual_no_miss_batch=per_update["factual_no_miss_states"],
            pair_batch=per_update["pairs"],
            objective_coefficients=(
                coefficients["factual_miss"],
                coefficients["factual_no_miss"],
                coefficients["paired_or_control"],
            ),
        )
        if config.canonical_payload() != dict(value):
            raise ValueError("paired run config payload is not canonical")
        return config


@dataclass(frozen=True)
class PairedExecutionLedger:
    """Exact completed-compute proof for one 32,000-update run."""

    method: str
    seed: int
    formal_schedule_fingerprint: str
    runtime_input_fingerprint: str
    control_provider_fingerprint: str | None
    pair_exposure_fingerprint: str
    factual_miss_exposure_fingerprint: str
    factual_no_miss_exposure_fingerprint: str
    initial_decoder_fingerprint: str
    final_decoder_fingerprint: str
    trainable_parameter_count: int
    optimizer_updates: int = 32_000
    completed_epochs: int = 800
    steps_per_epoch: int = 40
    factual_forward_calls: int = 64_000
    pair_endpoint_forward_calls: int = 32_000
    decoder_forward_calls: int = 96_000
    decoder_state_evaluations: int = 384_000
    backward_calls: int = 32_000
    optimizer_steps: int = 32_000
    all_gradients_finite: bool = True
    parameters_changed: bool = True
    minimum_gradient_l2_norm: float = 0.0
    maximum_gradient_l2_norm: float = 0.0
    schema_version: str = PAIRED_EXECUTION_LEDGER_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PAIRED_EXECUTION_LEDGER_SCHEMA:
            raise ValueError("unsupported paired execution ledger schema")
        if self.method not in PAIRED_METHODS:
            raise ValueError("execution ledger method is invalid")
        if self.seed not in (42, 43):
            raise ValueError("execution ledger seed must be 42 or 43")
        for name in (
            "formal_schedule_fingerprint",
            "runtime_input_fingerprint",
            "pair_exposure_fingerprint",
            "factual_miss_exposure_fingerprint",
            "factual_no_miss_exposure_fingerprint",
            "initial_decoder_fingerprint",
            "final_decoder_fingerprint",
        ):
            _digest(getattr(self, name), name=name)
        if self.method == "paired_difference":
            if self.control_provider_fingerprint is not None:
                raise ValueError(
                    "paired_difference execution cannot bind a control provider"
                )
        else:
            _digest(
                self.control_provider_fingerprint,
                name="control_provider_fingerprint",
            )
        exact = {
            "optimizer_updates": 32_000,
            "completed_epochs": 800,
            "steps_per_epoch": 40,
            "factual_forward_calls": 64_000,
            "pair_endpoint_forward_calls": 32_000,
            "decoder_forward_calls": 96_000,
            "decoder_state_evaluations": 384_000,
            "backward_calls": 32_000,
            "optimizer_steps": 32_000,
        }
        for name, expected in exact.items():
            if _positive_int(getattr(self, name), name=name) != expected:
                raise ValueError(f"{name} must equal {expected}")
        _positive_int(
            self.trainable_parameter_count,
            name="trainable_parameter_count",
        )
        if self.all_gradients_finite is not True:
            raise ValueError("formal execution requires all finite gradients")
        if self.parameters_changed is not True:
            raise ValueError("formal execution requires changed parameters")
        object.__setattr__(
            self,
            "minimum_gradient_l2_norm",
            _finite_nonnegative(
                self.minimum_gradient_l2_norm,
                name="minimum_gradient_l2_norm",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "maximum_gradient_l2_norm",
            _finite_nonnegative(
                self.maximum_gradient_l2_norm,
                name="maximum_gradient_l2_norm",
                positive=True,
            ),
        )
        if self.maximum_gradient_l2_norm < self.minimum_gradient_l2_norm:
            raise ValueError("gradient norm bounds are inconsistent")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "seed": self.seed,
            "formal_schedule_fingerprint": self.formal_schedule_fingerprint,
            "runtime_input_fingerprint": self.runtime_input_fingerprint,
            "control_provider_fingerprint": (
                self.control_provider_fingerprint
            ),
            "pair_exposure_fingerprint": self.pair_exposure_fingerprint,
            "factual_miss_exposure_fingerprint": (
                self.factual_miss_exposure_fingerprint
            ),
            "factual_no_miss_exposure_fingerprint": (
                self.factual_no_miss_exposure_fingerprint
            ),
            "initial_decoder_fingerprint": self.initial_decoder_fingerprint,
            "final_decoder_fingerprint": self.final_decoder_fingerprint,
            "trainable_parameter_count": self.trainable_parameter_count,
            "optimizer_updates": self.optimizer_updates,
            "completed_epochs": self.completed_epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "factual_forward_calls": self.factual_forward_calls,
            "pair_endpoint_forward_calls": self.pair_endpoint_forward_calls,
            "decoder_forward_calls": self.decoder_forward_calls,
            "decoder_state_evaluations": self.decoder_state_evaluations,
            "backward_calls": self.backward_calls,
            "optimizer_steps": self.optimizer_steps,
            "all_gradients_finite": self.all_gradients_finite,
            "parameters_changed": self.parameters_changed,
            "minimum_gradient_l2_norm": self.minimum_gradient_l2_norm,
            "maximum_gradient_l2_norm": self.maximum_gradient_l2_norm,
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "PairedExecutionLedger":
        if not isinstance(value, Mapping):
            raise TypeError("paired execution ledger must be a mapping")
        fields = tuple(cls.__dataclass_fields__)
        if set(value) != set(fields):
            raise ValueError("paired execution ledger fields are not canonical")
        ledger = cls(**{field: value[field] for field in fields})
        if ledger.canonical_payload() != dict(value):
            raise ValueError("paired execution ledger is not canonical")
        return ledger


def _normalized_logs(
    logs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, object], ...]:
    if len(logs) != 800:
        raise ValueError("formal paired train log must contain 800 epochs")
    normalized = json.loads(canonical_json(tuple(logs)))
    if not isinstance(normalized, list):
        raise TypeError("formal paired train log must be a list")
    output: list[dict[str, object]] = []
    required_metrics = {
        "mean_total_loss",
        "mean_factual_miss_loss",
        "mean_factual_no_miss_loss",
        "mean_paired_or_control_loss",
        "minimum_total_loss",
        "maximum_total_loss",
    }
    for epoch, row in enumerate(normalized):
        if not isinstance(row, dict) or set(row) != {
            "epoch",
            "steps",
            "metrics",
        }:
            raise ValueError("formal paired epoch log fields are not canonical")
        if row["epoch"] != epoch or row["steps"] != 40:
            raise ValueError("formal paired epoch numbering/horizon changed")
        metrics = row["metrics"]
        if not isinstance(metrics, dict) or set(metrics) != required_metrics:
            raise ValueError("formal paired epoch metrics are not canonical")
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"epoch metric {name} must be numeric")
            if not isfinite(float(value)):
                raise ValueError(f"epoch metric {name} must be finite")
        if metrics["minimum_total_loss"] > metrics["maximum_total_loss"]:
            raise ValueError("epoch total-loss bounds are inconsistent")
        output.append(row)
    return tuple(output)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _decoder_tensors(
    decoder: CURELiteDecoder,
) -> dict[str, torch.Tensor]:
    if not isinstance(decoder, CURELiteDecoder):
        raise TypeError("decoder must be CURELiteDecoder")
    result: dict[str, torch.Tensor] = {}
    for name, tensor in decoder.state_dict().items():
        value = tensor.detach().to("cpu").contiguous()
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"decoder tensor {name!r} must be finite floating point")
        result[name] = value
    if not result:
        raise ValueError("decoder state is empty")
    return result


@dataclass(frozen=True, slots=True)
class _LoadedPairedArtifactSeal:
    decoder: CURELiteDecoder
    config: PairedDecoderRunConfig
    epoch_logs: tuple[Mapping[str, object], ...]
    execution_ledger: PairedExecutionLedger
    source_directory: Path
    decoder_state_fingerprint: str
    artifact_fingerprint: str
    receipt_sha256: str
    weights_sha256: str
    train_log_sha256: str
    execution_ledger_sha256: str


@dataclass(frozen=True)
class LoadedPairedDecoderArtifact:
    """A strictly verified formal paired decoder artifact."""

    decoder: CURELiteDecoder
    config: PairedDecoderRunConfig
    epoch_logs: tuple[Mapping[str, object], ...]
    execution_ledger: PairedExecutionLedger
    decoder_state_fingerprint: str
    artifact_fingerprint: str
    receipt_sha256: str
    source_directory: Path
    weights_sha256: str
    train_log_sha256: str
    execution_ledger_sha256: str
    _verification_token: object

    def _verify_seal(self) -> None:
        seal = self._verification_token
        if type(seal) is not _LoadedPairedArtifactSeal:
            raise TypeError("paired artifact must come from the strict loader")
        if (
            seal.decoder is not self.decoder
            or seal.config is not self.config
            or seal.epoch_logs is not self.epoch_logs
            or seal.execution_ledger is not self.execution_ledger
        ):
            raise TypeError("loaded paired artifact objects were replaced")
        for name in (
            "source_directory",
            "decoder_state_fingerprint",
            "artifact_fingerprint",
            "receipt_sha256",
            "weights_sha256",
            "train_log_sha256",
            "execution_ledger_sha256",
        ):
            if getattr(seal, name) != getattr(self, name):
                raise TypeError("loaded paired artifact binding was replaced")

    def __post_init__(self) -> None:
        self._verify_seal()
        self.verify_unchanged()

    def verify_unchanged(self) -> None:
        self._verify_seal()
        if type(self.decoder) is not CURELiteDecoder:
            raise RuntimeError("loaded paired decoder type changed")
        if any(module.training for module in self.decoder.modules()):
            raise RuntimeError("loaded paired decoder is not in evaluation mode")
        if any(parameter.requires_grad for parameter in self.decoder.parameters()):
            raise RuntimeError("loaded paired decoder parameters are not frozen")
        if (
            self.decoder.config != self.config.decoder_config
            or decoder_state_fingerprint(self.decoder)
            != self.decoder_state_fingerprint
        ):
            raise RuntimeError("loaded paired decoder changed in memory")
        source = self.source_directory
        if source.is_symlink() or source.resolve(strict=True) != source:
            raise RuntimeError("paired artifact directory identity changed")
        members = {path.name: path for path in source.iterdir()}
        expected = {
            _WEIGHTS_NAME,
            _LOG_NAME,
            _LEDGER_NAME,
            _RECEIPT_NAME,
        }
        if set(members) != expected or any(
            path.is_symlink() or not path.is_file() for path in members.values()
        ):
            raise RuntimeError("paired artifact file inventory changed")
        expected_hashes = {
            _WEIGHTS_NAME: self.weights_sha256,
            _LOG_NAME: self.train_log_sha256,
            _LEDGER_NAME: self.execution_ledger_sha256,
            _RECEIPT_NAME: self.receipt_sha256,
        }
        if any(
            file_sha256(members[name]) != digest
            for name, digest in expected_hashes.items()
        ):
            raise RuntimeError("paired artifact member changed on disk")
        receipt = _load_json(members[_RECEIPT_NAME], name="paired receipt")
        if set(receipt) != _RECEIPT_FIELDS:
            raise RuntimeError("paired artifact receipt fields changed")
        receipt_core = dict(receipt)
        receipt_core.pop("artifact_fingerprint")
        if (
            stable_fingerprint(receipt_core) != self.artifact_fingerprint
            or receipt["run_config"] != self.config.canonical_payload()
        ):
            raise RuntimeError("paired artifact receipt binding changed")
        if tuple(_load_json(members[_LOG_NAME], name="paired train log")) != (
            self.epoch_logs
        ):
            raise RuntimeError("paired artifact train log changed")
        if PairedExecutionLedger.from_mapping(
            _load_json(members[_LEDGER_NAME], name="paired execution ledger")
        ) != self.execution_ledger:
            raise RuntimeError("paired artifact execution ledger changed")


def save_paired_decoder_artifact(
    directory: str | Path,
    decoder: CURELiteDecoder,
    config: PairedDecoderRunConfig,
    epoch_logs: Sequence[Mapping[str, Any]],
    execution_ledger: PairedExecutionLedger,
) -> str:
    """Publish only one completely finished 32,000-update artifact."""

    if not isinstance(config, PairedDecoderRunConfig):
        raise TypeError("config must be PairedDecoderRunConfig")
    if not isinstance(execution_ledger, PairedExecutionLedger):
        raise TypeError("execution_ledger must be PairedExecutionLedger")
    if decoder.config != config.decoder_config:
        raise ValueError("decoder topology differs from paired run config")
    final_fingerprint = decoder_state_fingerprint(decoder)
    if (
        execution_ledger.method != config.method
        or execution_ledger.seed != config.seed
        or execution_ledger.formal_schedule_fingerprint
        != config.formal_schedule_fingerprint
        or execution_ledger.runtime_input_fingerprint
        != config.runtime_input_fingerprint
        or execution_ledger.control_provider_fingerprint
        != config.control_provider_fingerprint
        or execution_ledger.initial_decoder_fingerprint
        != config.initial_decoder_fingerprint
        or execution_ledger.final_decoder_fingerprint != final_fingerprint
    ):
        raise ValueError("execution ledger and paired run config/decoder differ")
    logs = _normalized_logs(epoch_logs)
    target = Path(directory).expanduser().resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite paired artifact {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        from safetensors.torch import save_file

        weights = staging / _WEIGHTS_NAME
        train_log = staging / _LOG_NAME
        ledger_path = staging / _LEDGER_NAME
        receipt_path = staging / _RECEIPT_NAME
        save_file(_decoder_tensors(decoder), str(weights))
        train_log.write_bytes(_json_bytes(list(logs)))
        ledger_path.write_bytes(_json_bytes(execution_ledger.canonical_payload()))
        receipt_core = {
            "schema_version": PAIRED_DECODER_ARTIFACT_SCHEMA,
            "artifact_type": "cure_lite_paired_decoder",
            "run_config": config.canonical_payload(),
            "weights_file": _WEIGHTS_NAME,
            "weights_sha256": file_sha256(weights),
            "decoder_state_fingerprint": final_fingerprint,
            "train_log_file": _LOG_NAME,
            "train_log_sha256": file_sha256(train_log),
            "execution_ledger_file": _LEDGER_NAME,
            "execution_ledger_sha256": file_sha256(ledger_path),
        }
        receipt = {
            **receipt_core,
            "artifact_fingerprint": stable_fingerprint(receipt_core),
        }
        receipt_path.write_bytes(_json_bytes(receipt))
        os.rename(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return load_paired_decoder_artifact(target).artifact_fingerprint


def _load_json(path: Path, *, name: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error


def load_paired_decoder_artifact(
    directory: str | Path,
    *,
    expected_config: PairedDecoderRunConfig | None = None,
) -> LoadedPairedDecoderArtifact:
    """Load a complete paired artifact; incomplete directories are rejected."""

    requested = Path(directory).expanduser()
    if requested.is_symlink():
        raise ValueError("paired artifact cannot be addressed through a symlink")
    source = requested.resolve(strict=True)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("paired artifact must be a regular directory")
    members = {path.name: path for path in source.iterdir()}
    expected = {_WEIGHTS_NAME, _LOG_NAME, _LEDGER_NAME, _RECEIPT_NAME}
    if set(members) != expected or any(
        path.is_symlink() or not path.is_file() for path in members.values()
    ):
        raise ValueError("paired artifact file set is not complete and canonical")
    receipt = _load_json(members[_RECEIPT_NAME], name="paired receipt")
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("paired receipt fields are not canonical")
    if (
        receipt["schema_version"] != PAIRED_DECODER_ARTIFACT_SCHEMA
        or receipt["artifact_type"] != "cure_lite_paired_decoder"
        or receipt["weights_file"] != _WEIGHTS_NAME
        or receipt["train_log_file"] != _LOG_NAME
        or receipt["execution_ledger_file"] != _LEDGER_NAME
    ):
        raise ValueError("paired receipt identity is invalid")
    hashes = {
        _WEIGHTS_NAME: _digest(
            receipt["weights_sha256"],
            name="weights_sha256",
        ),
        _LOG_NAME: _digest(
            receipt["train_log_sha256"],
            name="train_log_sha256",
        ),
        _LEDGER_NAME: _digest(
            receipt["execution_ledger_sha256"],
            name="execution_ledger_sha256",
        ),
    }
    if any(
        file_sha256(members[name]) != digest for name, digest in hashes.items()
    ):
        raise ValueError("paired artifact member SHA256 mismatch")
    artifact_fingerprint = _digest(
        receipt["artifact_fingerprint"],
        name="artifact_fingerprint",
    )
    receipt_core = dict(receipt)
    receipt_core.pop("artifact_fingerprint")
    if stable_fingerprint(receipt_core) != artifact_fingerprint:
        raise ValueError("paired artifact fingerprint mismatch")
    config = PairedDecoderRunConfig.from_mapping(receipt["run_config"])
    if expected_config is not None and config != expected_config:
        raise ValueError("paired artifact run config differs from expected")
    logs_raw = _load_json(members[_LOG_NAME], name="paired train log")
    if not isinstance(logs_raw, list):
        raise TypeError("paired train log must be a list")
    logs = _normalized_logs(logs_raw)
    ledger = PairedExecutionLedger.from_mapping(
        _load_json(members[_LEDGER_NAME], name="paired execution ledger")
    )
    if (
        ledger.method != config.method
        or ledger.seed != config.seed
        or ledger.formal_schedule_fingerprint
        != config.formal_schedule_fingerprint
        or ledger.runtime_input_fingerprint
        != config.runtime_input_fingerprint
        or ledger.control_provider_fingerprint
        != config.control_provider_fingerprint
        or ledger.initial_decoder_fingerprint
        != config.initial_decoder_fingerprint
    ):
        raise ValueError("paired execution ledger differs from run config")
    try:
        from safetensors.torch import load_file

        tensors = load_file(str(members[_WEIGHTS_NAME]), device="cpu")
    except Exception as error:
        raise ValueError("paired decoder weights are not valid safetensors") from error
    with torch.random.fork_rng(devices=[]):
        decoder = CURELiteDecoder(config.decoder_config)
    expected_names = set(decoder.state_dict())
    if set(tensors) != expected_names:
        raise ValueError("paired decoder weight names differ from topology")
    decoder.load_state_dict(tensors, strict=True)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    state_fingerprint = decoder_state_fingerprint(decoder)
    if (
        state_fingerprint
        != _digest(
            receipt["decoder_state_fingerprint"],
            name="decoder_state_fingerprint",
        )
        or state_fingerprint != ledger.final_decoder_fingerprint
    ):
        raise ValueError("paired decoder state fingerprint mismatch")
    receipt_sha256 = file_sha256(members[_RECEIPT_NAME])
    seal = _LoadedPairedArtifactSeal(
        decoder=decoder,
        config=config,
        epoch_logs=logs,
        execution_ledger=ledger,
        source_directory=source,
        decoder_state_fingerprint=state_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
        receipt_sha256=receipt_sha256,
        weights_sha256=hashes[_WEIGHTS_NAME],
        train_log_sha256=hashes[_LOG_NAME],
        execution_ledger_sha256=hashes[_LEDGER_NAME],
    )
    return LoadedPairedDecoderArtifact(
        decoder=decoder,
        config=config,
        epoch_logs=logs,
        execution_ledger=ledger,
        decoder_state_fingerprint=state_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
        receipt_sha256=receipt_sha256,
        source_directory=source,
        weights_sha256=hashes[_WEIGHTS_NAME],
        train_log_sha256=hashes[_LOG_NAME],
        execution_ledger_sha256=hashes[_LEDGER_NAME],
        _verification_token=seal,
    )


__all__ = [
    "PAIRED_DECODER_ARTIFACT_SCHEMA",
    "PAIRED_EXECUTION_LEDGER_SCHEMA",
    "PAIRED_METHODS",
    "LoadedPairedDecoderArtifact",
    "PairedDecoderRunConfig",
    "PairedExecutionLedger",
    "load_paired_decoder_artifact",
    "method_objective_contract",
    "save_paired_decoder_artifact",
]
