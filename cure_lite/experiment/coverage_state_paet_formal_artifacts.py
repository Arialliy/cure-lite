"""Strict final-only artifact for a completed v21 PAET Formal800 run.

This module persists exactly one trained PAET-BFA completion field after the
frozen 800 x 40 D_R training ledger has completed.  It deliberately contains
no training, calibration, D_V/D_T access, optimizer serialization, resume
state, or intermediate-checkpoint path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isclose, isfinite
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import torch

from ..cache.schema import file_sha256, stable_fingerprint
from ..coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
    CoverageStatePhaseAlignedEvidenceTransportConfig,
)
from .coverage_state_paet_dataset_free import (
    COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_PAET_FORMAL_WIDTH,
)
from .coverage_state_paet_formal_training import (
    COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT,
    COVERAGE_STATE_PAET_FORMAL_RESULT_SCHEMA,
    COVERAGE_STATE_PAET_FORMAL_RUN_ID,
    COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT,
    COVERAGE_STATE_PAET_FORMAL_SEED,
    COVERAGE_STATE_PAET_FORMAL_UPDATES,
    CoverageStatePAETFormal800RunResult,
)
from .coverage_state_training import (
    COVERAGE_STATE_MATCHED_RESULT_SCHEMA,
    COVERAGE_STATE_TRAINING_RESULT_SCHEMA,
    CoverageStateMatchedTrainingConfig,
    coverage_state_model_contract_payload,
    coverage_state_model_fingerprint,
)
from ..frozen_base import module_state_fingerprint
from ..train.coverage_state_fused_step import (
    CSLF_PMOPE_POLICY,
    CoverageStatePairObjective,
)


COVERAGE_STATE_PAET_FORMAL_ARTIFACT_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal800-final-artifact-v1"
)
_ARTIFACT_TYPE = "cure_lite_paet_bfa_v21_formal800_final_model"
_WEIGHTS_NAME = "model.safetensors"
_FORMAL_RESULT_NAME = "formal_result.json"
_TRAINING_NAME = "training.json"
_EPOCH_LOG_NAME = "epoch_log.json"
_RECEIPT_NAME = "receipt.json"
_MEMBER_NAMES = frozenset(
    {
        _WEIGHTS_NAME,
        _FORMAL_RESULT_NAME,
        _TRAINING_NAME,
        _EPOCH_LOG_NAME,
        _RECEIPT_NAME,
    }
)
_HEX = frozenset("0123456789abcdef")
_STATE_NAMES = frozenset(
    {
        "joint_state_weight",
        "joint_hidden_bias",
        "scalar_energy_weight",
    }
)
_FORMAL_CHECK_NAMES = frozenset(
    {
        "authorization_consumed_exactly_once",
        "full_D_R_cache_and_formal_schedule",
        "fixed_seed42_800x40_compute_budget",
        "singleton_paet_bfa_pmope",
        "from_scratch_initial_state",
        "single_final_model_output",
        "fixed_level_set_decode_contract",
        "D_R_only_training_inputs",
        "status_semantics_kept_separate",
    }
)
_FORMAL_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "runtime_splits",
        "authorization_fingerprint",
        "structural_advancement_passed",
        "generic_population_gate_passed",
        "bounded_evidence_interpretation",
        "training",
        "training_invocations",
        "checks",
        "failed_checks",
        "training_complete",
        "field_threshold_hex",
        "threshold_search_performed",
        "training_contract",
        "D_V_accessed",
        "D_T_accessed",
        "calibration_performed",
        "inference_performed",
        "performance_evaluation_performed",
        "performance_claim_supported",
        "full_CURE_authorized",
        "cross_backbone_authorized",
    }
)
_TRAINING_FIELDS = frozenset(
    {
        "schema_version",
        "config",
        "common_initial_model_fingerprint",
        "schedule_fingerprint",
        "cache_fingerprint",
        "objectives",
        "objective_suite",
        "fairness",
        "model_contract",
    }
)
_OBJECTIVE_FIELDS = frozenset(
    {
        "schema_version",
        "objective",
        "objective_policy",
        "seed",
        "epochs",
        "steps_per_epoch",
        "completed_updates",
        "schedule_fingerprint",
        "cache_fingerprint",
        "execution_device",
        "device_cache_fingerprint",
        "device_cache_resident_bytes",
        "optimizer_config_fingerprint",
        "initial_model_fingerprint",
        "final_model_fingerprint",
        "epoch_logs",
        "first_nonzero_gradient_update",
        "compute",
    }
)
_EPOCH_FIELDS = frozenset(
    {
        "epoch",
        "completed_updates",
        "objective",
        "selection_sequence_fingerprint",
        "mean_factual_miss/loss",
        "mean_factual_no_miss/loss",
        "mean_pair/loss",
        "mean_total",
        "mean_gradient_l2_norm",
    }
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "run_id",
        "model_config",
        "parameter_count",
        "weights_file",
        "weights_sha256",
        "training_model_fingerprint",
        "module_state_fingerprint",
        "formal_result_file",
        "formal_result_sha256",
        "formal_result_fingerprint",
        "training_file",
        "training_sha256",
        "training_fingerprint",
        "epoch_log_file",
        "epoch_log_sha256",
        "epoch_log_fingerprint",
        "authorization_fingerprint",
        "checkpoint_policy",
        "optimizer_state_saved",
        "intermediate_checkpoint_saved",
        "runtime_splits",
        "D_V_accessed",
        "D_T_accessed",
        "artifact_fingerprint",
    }
)


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _exact_dict(
    value: object,
    *,
    fields: frozenset[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields are not canonical")
    return value


def _expected_model_config(
) -> CoverageStatePhaseAlignedEvidenceTransportConfig:
    config = CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS,
        feature_stride=COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE,
        width=COVERAGE_STATE_PAET_FORMAL_WIDTH,
    )
    if (
        config.expected_parameter_count
        != COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
    ):
        raise RuntimeError("frozen PAET Formal800 parameter count changed")
    return config


def _model_config_payload(
    config: CoverageStatePhaseAlignedEvidenceTransportConfig,
) -> dict[str, object]:
    if type(config) is not CoverageStatePhaseAlignedEvidenceTransportConfig:
        raise TypeError("artifact requires the exact PAET-BFA config type")
    if config != _expected_model_config():
        raise ValueError("artifact rejects a non-formal PAET-BFA config")
    return asdict(config)


def _expected_initial_model_fingerprint(
    config: CoverageStatePhaseAlignedEvidenceTransportConfig,
) -> str:
    if config != _expected_model_config():
        raise ValueError("initial-state check requires the formal config")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(
            COVERAGE_STATE_PAET_FORMAL_SEED
        )
        model = CURELitePhaseAlignedEvidenceTransportLevelSet(config)
        return coverage_state_model_fingerprint(model)


def _config_from_payload(
    value: object,
) -> CoverageStatePhaseAlignedEvidenceTransportConfig:
    if not isinstance(value, dict):
        raise ValueError("model_config must be a mapping")
    expected_fields = frozenset(
        CoverageStatePhaseAlignedEvidenceTransportConfig
        .__dataclass_fields__
    )
    if set(value) != expected_fields:
        raise ValueError("model_config fields are not canonical")
    try:
        config = CoverageStatePhaseAlignedEvidenceTransportConfig(
            **value
        )
    except (TypeError, ValueError) as error:
        raise ValueError("model_config violates the PAET contract") from error
    if config != _expected_model_config() or asdict(config) != value:
        raise ValueError("model_config is not the frozen Formal800 config")
    return config


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json(path: Path, *, name: str) -> Any:
    if path.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    source = path.resolve(strict=True)
    if source != path or not source.is_file() or source.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")

    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"{name} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{name} contains non-finite number {value}")

    try:
        with source.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_nonfinite,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not strict JSON") from error


def _safe_absolute_path(
    value: str | Path,
    *,
    must_exist: bool,
) -> Path:
    raw = Path(value).expanduser()
    if ".." in raw.parts:
        raise ValueError("artifact path may not contain parent traversal")
    absolute = Path(os.path.abspath(os.fspath(raw)))
    if must_exist:
        resolved = absolute.resolve(strict=True)
        if resolved != absolute:
            raise ValueError("artifact path may not traverse a symlink")
        return resolved
    parent = absolute.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.resolve(strict=True) != parent:
        raise ValueError("artifact parent may not traverse a symlink")
    return absolute


def _model_tensors(
    model: CURELitePhaseAlignedEvidenceTransportLevelSet,
) -> dict[str, torch.Tensor]:
    if type(model) is not CURELitePhaseAlignedEvidenceTransportLevelSet:
        raise TypeError("artifact model must be the exact PAET-BFA class")
    if model.config != _expected_model_config():
        raise ValueError("artifact model config is not Formal800")
    tensors: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        tensor = value.detach().to(device="cpu").contiguous()
        if (
            name not in _STATE_NAMES
            or tensor.dtype != torch.float32
            or not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(
                f"PAET model tensor {name!r} violates its contract"
            )
        tensors[name] = tensor
    if set(tensors) != _STATE_NAMES:
        raise ValueError("PAET model state inventory changed")
    return tensors


def _normalized_epoch_logs(
    value: object,
) -> tuple[dict[str, object], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 800:
        raise ValueError("Formal800 epoch log must contain 800 rows")
    normalized = json.loads(
        json.dumps(
            list(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    output: list[dict[str, object]] = []
    objective = CoverageStatePairObjective.PMOPE_JOINT.value
    numeric_fields = _EPOCH_FIELDS - {
        "epoch",
        "completed_updates",
        "objective",
        "selection_sequence_fingerprint",
    }
    for epoch, raw_row in enumerate(normalized):
        row = _exact_dict(
            raw_row,
            fields=_EPOCH_FIELDS,
            name=f"epoch_log[{epoch}]",
        )
        if (
            row["epoch"] != epoch
            or row["completed_updates"] != (epoch + 1) * 40
            or row["objective"] != objective
        ):
            raise ValueError("Formal800 epoch execution accounting changed")
        _digest(
            row["selection_sequence_fingerprint"],
            name="selection_sequence_fingerprint",
        )
        for name in numeric_fields:
            number = row[name]
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not isfinite(float(number))
            ):
                raise ValueError(
                    f"Formal800 epoch metric {name!r} must be finite"
                )
        if float(row["mean_gradient_l2_norm"]) < 0.0:
            raise ValueError("Formal800 gradient norm must be nonnegative")
        branch_sum = (
            float(row["mean_factual_miss/loss"])
            + float(row["mean_factual_no_miss/loss"])
            + float(row["mean_pair/loss"])
        )
        if not isclose(
            float(row["mean_total"]),
            branch_sum,
            rel_tol=2.0e-6,
            abs_tol=2.0e-7,
        ):
            raise ValueError(
                "Formal800 mean total differs from the three branches"
            )
        output.append(row)
    return tuple(output)


def _validate_training_payload(
    value: object,
    *,
    model: CURELitePhaseAlignedEvidenceTransportLevelSet | None,
) -> tuple[dict[str, Any], tuple[dict[str, object], ...]]:
    training = _exact_dict(
        value,
        fields=_TRAINING_FIELDS,
        name="Formal800 training payload",
    )
    expected_optimizer = CoverageStateMatchedTrainingConfig(
        seed=COVERAGE_STATE_PAET_FORMAL_SEED
    ).canonical_payload()
    objective_name = CoverageStatePairObjective.PMOPE_JOINT.value
    if (
        training["schema_version"] != COVERAGE_STATE_MATCHED_RESULT_SCHEMA
        or training["config"] != expected_optimizer
        or training["objective_suite"] != [objective_name]
    ):
        raise ValueError("Formal800 matched-training identity changed")
    for name in (
        "common_initial_model_fingerprint",
        "schedule_fingerprint",
        "cache_fingerprint",
    ):
        _digest(training[name], name=name)
    objectives = training["objectives"]
    if not isinstance(objectives, list) or len(objectives) != 1:
        raise ValueError("Formal800 must contain one objective result")
    row = _exact_dict(
        objectives[0],
        fields=_OBJECTIVE_FIELDS,
        name="Formal800 objective result",
    )
    compute = row["compute"]
    if not isinstance(compute, dict) or compute != {
        "forward_calls": COVERAGE_STATE_PAET_FORMAL_UPDATES,
        "backward_calls": COVERAGE_STATE_PAET_FORMAL_UPDATES,
        "optimizer_steps": COVERAGE_STATE_PAET_FORMAL_UPDATES,
        "logical_state_evaluations": (
            12 * COVERAGE_STATE_PAET_FORMAL_UPDATES
        ),
        "finite_state_audits": COVERAGE_STATE_PAET_FORMAL_UPDATES + 1,
    }:
        raise ValueError("Formal800 compute ledger changed")
    if (
        row["schema_version"] != COVERAGE_STATE_TRAINING_RESULT_SCHEMA
        or row["objective"] != objective_name
        or row["objective_policy"] != CSLF_PMOPE_POLICY
        or row["seed"] != COVERAGE_STATE_PAET_FORMAL_SEED
        or row["epochs"] != 800
        or row["steps_per_epoch"] != 40
        or row["completed_updates"]
        != COVERAGE_STATE_PAET_FORMAL_UPDATES
        or row["schedule_fingerprint"]
        != training["schedule_fingerprint"]
        or row["schedule_fingerprint"]
        != COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT
        or row["cache_fingerprint"] != training["cache_fingerprint"]
        or row["cache_fingerprint"]
        != COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        or row["execution_device"] != "cuda:0"
        or isinstance(row["device_cache_resident_bytes"], bool)
        or not isinstance(row["device_cache_resident_bytes"], int)
        or row["device_cache_resident_bytes"] < 1
        or row["initial_model_fingerprint"]
        != training["common_initial_model_fingerprint"]
        or row["initial_model_fingerprint"]
        != _expected_initial_model_fingerprint(
            _expected_model_config()
        )
    ):
        raise ValueError("Formal800 objective ledger changed")
    for name in (
        "device_cache_fingerprint",
        "optimizer_config_fingerprint",
        "initial_model_fingerprint",
        "final_model_fingerprint",
    ):
        _digest(row[name], name=name)
    first_nonzero = row["first_nonzero_gradient_update"]
    if (
        not isinstance(first_nonzero, dict)
        or set(first_nonzero) != _STATE_NAMES
        or any(
            isinstance(update, bool)
            or not isinstance(update, int)
            or not 0 <= update < COVERAGE_STATE_PAET_FORMAL_UPDATES
            for update in first_nonzero.values()
        )
    ):
        raise ValueError("Formal800 gradient-coverage ledger changed")
    epoch_logs = _normalized_epoch_logs(row["epoch_logs"])
    fairness = training["fairness"]
    required_fairness = {
        "candidate_model": "PAET-BFA",
        "single_candidate_only": True,
        "same_initial_state": True,
        "same_schedule": True,
        "same_endpoints": True,
        "same_model": True,
        "same_optimizer": True,
        "same_device_cache": True,
        "same_compute_budget": True,
        "same_natural_branches": True,
        "historical_controls_retrained": False,
        "historical_v20_objective_reused": True,
        "allowed_difference_from_sealed_v20": (
            "predeclared_field_equation_only"
        ),
        "same_model_class": True,
        "same_model_config": True,
        "same_parameter_count": True,
        "same_parameter_shapes": True,
    }
    if fairness != required_fairness:
        raise ValueError("Formal800 fairness contract changed")
    contract = training["model_contract"]
    if (
        not isinstance(contract, dict)
        or contract.get("parameter_count")
        != COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
    ):
        raise ValueError("Formal800 model contract changed")
    if model is not None:
        if contract != coverage_state_model_contract_payload(model):
            raise ValueError(
                "Formal800 training/model structural binding changed"
            )
        if (
            coverage_state_model_fingerprint(model)
            != row["final_model_fingerprint"]
        ):
            raise ValueError(
                "Formal800 training/model state binding changed"
            )
    return training, epoch_logs


def _validate_formal_result_payload(
    value: object,
    *,
    training: Mapping[str, Any],
    authorization_fingerprint: str,
) -> dict[str, Any]:
    result = _exact_dict(
        value,
        fields=_FORMAL_RESULT_FIELDS,
        name="Formal800 result payload",
    )
    checks = result["checks"]
    training_contract = result["training_contract"]
    if (
        result["schema_version"]
        != COVERAGE_STATE_PAET_FORMAL_RESULT_SCHEMA
        or result["run_id"] != COVERAGE_STATE_PAET_FORMAL_RUN_ID
        or result["runtime_splits"] != ["D_R"]
        or result["authorization_fingerprint"]
        != authorization_fingerprint
        or result["structural_advancement_passed"] is not True
        or result["generic_population_gate_passed"] is not False
        or result["bounded_evidence_interpretation"]
        != "structural_advancement_only_not_performance"
        or result["training"] != training
        or result["training_invocations"] != 1
        or not isinstance(checks, dict)
        or set(checks) != _FORMAL_CHECK_NAMES
        or not all(value is True for value in checks.values())
        or result["failed_checks"] != []
        or result["training_complete"] is not True
        or result["field_threshold_hex"] != 0.0.hex()
        or result["threshold_search_performed"] is not False
        or training_contract
        != {
            "from_scratch": True,
            "process_local_single_attempt_claim": True,
            "cross_process_output_claim_required": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "checkpoint_policy": "final_model_only",
            "intermediate_checkpoint_saved": False,
            "optimizer_state_saved": False,
        }
        or result["D_V_accessed"] is not False
        or result["D_T_accessed"] is not False
        or result["calibration_performed"] is not False
        or result["inference_performed"] is not False
        or result["performance_evaluation_performed"] is not False
        or result["performance_claim_supported"] is not False
        or result["full_CURE_authorized"] is not False
        or result["cross_backbone_authorized"] is not False
    ):
        raise ValueError("Formal800 result contract changed")
    return result


@dataclass(frozen=True, slots=True)
class _LoadedCoverageStatePAETFormalArtifactSeal:
    model: CURELitePhaseAlignedEvidenceTransportLevelSet
    model_config: CoverageStatePhaseAlignedEvidenceTransportConfig
    formal_result_payload: dict[str, Any]
    training_payload: dict[str, Any]
    epoch_logs: tuple[dict[str, object], ...]
    source_directory: Path
    training_model_fingerprint: str
    module_state_fingerprint: str
    formal_result_fingerprint: str
    training_fingerprint: str
    epoch_log_fingerprint: str
    authorization_fingerprint: str
    artifact_fingerprint: str
    receipt_sha256: str
    weights_sha256: str
    formal_result_sha256: str
    training_sha256: str
    epoch_log_sha256: str


@dataclass(frozen=True)
class LoadedCoverageStatePAETFormalArtifact:
    """A loaded PAET Formal800 model whose memory and files stay bound."""

    model: CURELitePhaseAlignedEvidenceTransportLevelSet
    model_config: CoverageStatePhaseAlignedEvidenceTransportConfig
    formal_result_payload: dict[str, Any]
    training_payload: dict[str, Any]
    epoch_logs: tuple[dict[str, object], ...]
    training_model_fingerprint: str
    module_state_fingerprint: str
    formal_result_fingerprint: str
    training_fingerprint: str
    epoch_log_fingerprint: str
    authorization_fingerprint: str
    artifact_fingerprint: str
    source_directory: Path
    receipt_sha256: str
    weights_sha256: str
    formal_result_sha256: str
    training_sha256: str
    epoch_log_sha256: str
    _verification_token: object

    def _seal(self) -> _LoadedCoverageStatePAETFormalArtifactSeal:
        seal = self._verification_token
        if type(seal) is not _LoadedCoverageStatePAETFormalArtifactSeal:
            raise TypeError(
                "PAET Formal800 artifact must come from its strict loader"
            )
        if (
            seal.model is not self.model
            or seal.model_config is not self.model_config
            or seal.formal_result_payload
            is not self.formal_result_payload
            or seal.training_payload is not self.training_payload
            or seal.epoch_logs is not self.epoch_logs
        ):
            raise TypeError("loaded PAET artifact objects were replaced")
        return seal

    def __post_init__(self) -> None:
        self.verify_unchanged()

    def verify_unchanged(self) -> None:
        seal = self._seal()
        for name in (
            "source_directory",
            "training_model_fingerprint",
            "module_state_fingerprint",
            "formal_result_fingerprint",
            "training_fingerprint",
            "epoch_log_fingerprint",
            "authorization_fingerprint",
            "artifact_fingerprint",
            "receipt_sha256",
            "weights_sha256",
            "formal_result_sha256",
            "training_sha256",
            "epoch_log_sha256",
        ):
            if getattr(self, name) != getattr(seal, name):
                raise RuntimeError("loaded PAET artifact binding changed")
        if (
            type(self.model)
            is not CURELitePhaseAlignedEvidenceTransportLevelSet
            or self.model.config != self.model_config
            or self.model_config != _expected_model_config()
            or any(module.training for module in self.model.modules())
            or any(
                parameter.requires_grad
                for parameter in self.model.parameters()
            )
            or coverage_state_model_fingerprint(self.model)
            != self.training_model_fingerprint
            or module_state_fingerprint(self.model)
            != self.module_state_fingerprint
            or stable_fingerprint(self.formal_result_payload)
            != self.formal_result_fingerprint
            or stable_fingerprint(self.training_payload)
            != self.training_fingerprint
            or stable_fingerprint(self.epoch_logs)
            != self.epoch_log_fingerprint
        ):
            raise RuntimeError("loaded PAET artifact changed in memory")
        source = self.source_directory
        if (
            source.is_symlink()
            or source.resolve(strict=True) != source
            or not source.is_dir()
        ):
            raise RuntimeError("PAET artifact directory identity changed")
        members = {path.name: path for path in source.iterdir()}
        if set(members) != _MEMBER_NAMES or any(
            path.is_symlink() or not path.is_file()
            for path in members.values()
        ):
            raise RuntimeError("PAET artifact file inventory changed")
        hashes = {
            _WEIGHTS_NAME: self.weights_sha256,
            _FORMAL_RESULT_NAME: self.formal_result_sha256,
            _TRAINING_NAME: self.training_sha256,
            _EPOCH_LOG_NAME: self.epoch_log_sha256,
            _RECEIPT_NAME: self.receipt_sha256,
        }
        if any(
            file_sha256(members[name]) != digest
            for name, digest in hashes.items()
        ):
            raise RuntimeError("PAET artifact member changed on disk")


def save_coverage_state_paet_formal_artifact(
    directory: str | Path,
    result: CoverageStatePAETFormal800RunResult,
) -> str:
    """Atomically publish only the final model of one completed Formal800."""

    if not isinstance(result, CoverageStatePAETFormal800RunResult):
        raise TypeError(
            "result must be CoverageStatePAETFormal800RunResult"
        )
    result.verify_unchanged()
    if not result.training_complete:
        raise ValueError("Formal800 training is not complete")
    model = result.final_model
    model_config = _model_config_payload(model.config)
    if (
        sum(parameter.numel() for parameter in model.parameters())
        != COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
    ):
        raise ValueError("Formal800 model parameter count changed")
    formal_result = result.canonical_payload()
    training = result.training.canonical_payload()
    authorization_fingerprint = _digest(
        result.authorization.authorization_fingerprint,
        name="authorization_fingerprint",
    )
    if (
        formal_result.get("training") != training
        or stable_fingerprint(formal_result) != result.result_fingerprint
    ):
        raise ValueError("Formal800 result canonical payload changed")
    training, epoch_logs = _validate_training_payload(
        training,
        model=model,
    )
    formal_result = _validate_formal_result_payload(
        formal_result,
        training=training,
        authorization_fingerprint=authorization_fingerprint,
    )
    target = _safe_absolute_path(directory, must_exist=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite PAET Formal800 artifact {target}"
        )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            dir=target.parent,
        )
    )
    try:
        from safetensors.torch import save_file

        weights_path = staging / _WEIGHTS_NAME
        formal_result_path = staging / _FORMAL_RESULT_NAME
        training_path = staging / _TRAINING_NAME
        epoch_log_path = staging / _EPOCH_LOG_NAME
        receipt_path = staging / _RECEIPT_NAME
        save_file(_model_tensors(model), str(weights_path))
        formal_result_path.write_bytes(_json_bytes(formal_result))
        training_path.write_bytes(_json_bytes(training))
        epoch_log_path.write_bytes(_json_bytes(list(epoch_logs)))
        receipt_core = {
            "schema_version": (
                COVERAGE_STATE_PAET_FORMAL_ARTIFACT_SCHEMA
            ),
            "artifact_type": _ARTIFACT_TYPE,
            "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
            "model_config": model_config,
            "parameter_count": (
                COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
            ),
            "weights_file": _WEIGHTS_NAME,
            "weights_sha256": file_sha256(weights_path),
            "training_model_fingerprint": (
                coverage_state_model_fingerprint(model)
            ),
            "module_state_fingerprint": module_state_fingerprint(model),
            "formal_result_file": _FORMAL_RESULT_NAME,
            "formal_result_sha256": file_sha256(formal_result_path),
            "formal_result_fingerprint": stable_fingerprint(
                formal_result
            ),
            "training_file": _TRAINING_NAME,
            "training_sha256": file_sha256(training_path),
            "training_fingerprint": stable_fingerprint(training),
            "epoch_log_file": _EPOCH_LOG_NAME,
            "epoch_log_sha256": file_sha256(epoch_log_path),
            "epoch_log_fingerprint": stable_fingerprint(epoch_logs),
            "authorization_fingerprint": authorization_fingerprint,
            "checkpoint_policy": "final_model_only",
            "optimizer_state_saved": False,
            "intermediate_checkpoint_saved": False,
            "runtime_splits": ["D_R"],
            "D_V_accessed": False,
            "D_T_accessed": False,
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
    return load_coverage_state_paet_formal_artifact(
        target,
        expected_authorization_fingerprint=authorization_fingerprint,
        expected_result_fingerprint=result.result_fingerprint,
    ).artifact_fingerprint


def load_coverage_state_paet_formal_artifact(
    directory: str | Path,
    *,
    expected_authorization_fingerprint: str | None = None,
    expected_result_fingerprint: str | None = None,
) -> LoadedCoverageStatePAETFormalArtifact:
    """Load only a complete, canonical, final-only PAET Formal800 model."""

    source = _safe_absolute_path(directory, must_exist=True)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("PAET artifact path must be a non-symlink directory")
    members = {path.name: path for path in source.iterdir()}
    if set(members) != _MEMBER_NAMES or any(
        path.is_symlink() or not path.is_file()
        for path in members.values()
    ):
        raise ValueError("PAET artifact file inventory is not canonical")
    receipt = _exact_dict(
        _strict_json(
            members[_RECEIPT_NAME],
            name="PAET artifact receipt",
        ),
        fields=_RECEIPT_FIELDS,
        name="PAET artifact receipt",
    )
    if (
        receipt["schema_version"]
        != COVERAGE_STATE_PAET_FORMAL_ARTIFACT_SCHEMA
        or receipt["artifact_type"] != _ARTIFACT_TYPE
        or receipt["run_id"] != COVERAGE_STATE_PAET_FORMAL_RUN_ID
        or receipt["weights_file"] != _WEIGHTS_NAME
        or receipt["formal_result_file"] != _FORMAL_RESULT_NAME
        or receipt["training_file"] != _TRAINING_NAME
        or receipt["epoch_log_file"] != _EPOCH_LOG_NAME
        or receipt["parameter_count"]
        != COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
        or receipt["checkpoint_policy"] != "final_model_only"
        or receipt["optimizer_state_saved"] is not False
        or receipt["intermediate_checkpoint_saved"] is not False
        or receipt["runtime_splits"] != ["D_R"]
        or receipt["D_V_accessed"] is not False
        or receipt["D_T_accessed"] is not False
    ):
        raise ValueError("PAET artifact receipt identity changed")
    model_config = _config_from_payload(receipt["model_config"])
    hashes = {
        _WEIGHTS_NAME: _digest(
            receipt["weights_sha256"],
            name="weights_sha256",
        ),
        _FORMAL_RESULT_NAME: _digest(
            receipt["formal_result_sha256"],
            name="formal_result_sha256",
        ),
        _TRAINING_NAME: _digest(
            receipt["training_sha256"],
            name="training_sha256",
        ),
        _EPOCH_LOG_NAME: _digest(
            receipt["epoch_log_sha256"],
            name="epoch_log_sha256",
        ),
    }
    if any(
        file_sha256(members[name]) != digest
        for name, digest in hashes.items()
    ):
        raise ValueError("PAET artifact member SHA256 mismatch")
    receipt_core = dict(receipt)
    artifact_fingerprint = _digest(
        receipt_core.pop("artifact_fingerprint"),
        name="artifact_fingerprint",
    )
    if stable_fingerprint(receipt_core) != artifact_fingerprint:
        raise ValueError("PAET artifact fingerprint mismatch")
    authorization_fingerprint = _digest(
        receipt["authorization_fingerprint"],
        name="authorization_fingerprint",
    )
    formal_result_fingerprint = _digest(
        receipt["formal_result_fingerprint"],
        name="formal_result_fingerprint",
    )
    training_fingerprint = _digest(
        receipt["training_fingerprint"],
        name="training_fingerprint",
    )
    epoch_log_fingerprint = _digest(
        receipt["epoch_log_fingerprint"],
        name="epoch_log_fingerprint",
    )
    if expected_authorization_fingerprint is not None and (
        _digest(
            expected_authorization_fingerprint,
            name="expected_authorization_fingerprint",
        )
        != authorization_fingerprint
    ):
        raise ValueError("PAET artifact authorization differs from expected")
    if expected_result_fingerprint is not None and (
        _digest(
            expected_result_fingerprint,
            name="expected_result_fingerprint",
        )
        != formal_result_fingerprint
    ):
        raise ValueError("PAET artifact result differs from expected")
    formal_result_raw = _strict_json(
        members[_FORMAL_RESULT_NAME],
        name="PAET formal result",
    )
    training_raw = _strict_json(
        members[_TRAINING_NAME],
        name="PAET training payload",
    )
    epoch_log_raw = _strict_json(
        members[_EPOCH_LOG_NAME],
        name="PAET epoch log",
    )
    try:
        from safetensors.torch import load_file

        tensors = load_file(
            str(members[_WEIGHTS_NAME]),
            device="cpu",
        )
    except Exception as error:
        raise ValueError(
            "PAET model weights are not valid safetensors"
        ) from error
    model = CURELitePhaseAlignedEvidenceTransportLevelSet(model_config)
    expected_state = model.state_dict()
    if set(tensors) != set(expected_state) or set(tensors) != _STATE_NAMES:
        raise ValueError("PAET model weight inventory changed")
    for name, expected_tensor in expected_state.items():
        actual = tensors[name]
        if (
            actual.shape != expected_tensor.shape
            or actual.dtype != expected_tensor.dtype
            or actual.dtype != torch.float32
            or not bool(torch.isfinite(actual).all())
        ):
            raise ValueError(
                f"PAET model tensor {name!r} violates its contract"
            )
    model.load_state_dict(tensors, strict=True)
    model.eval()
    model.requires_grad_(False)
    training_model_fingerprint = coverage_state_model_fingerprint(
        model
    )
    loaded_module_state_fingerprint = module_state_fingerprint(model)
    if (
        training_model_fingerprint
        != _digest(
            receipt["training_model_fingerprint"],
            name="training_model_fingerprint",
        )
        or loaded_module_state_fingerprint
        != _digest(
            receipt["module_state_fingerprint"],
            name="module_state_fingerprint",
        )
    ):
        raise ValueError("PAET loaded model fingerprint mismatch")
    training, epoch_logs = _validate_training_payload(
        training_raw,
        model=model,
    )
    if (
        epoch_logs != _normalized_epoch_logs(epoch_log_raw)
        or stable_fingerprint(training) != training_fingerprint
        or stable_fingerprint(epoch_logs) != epoch_log_fingerprint
    ):
        raise ValueError("PAET training or epoch-log binding changed")
    formal_result = _validate_formal_result_payload(
        formal_result_raw,
        training=training,
        authorization_fingerprint=authorization_fingerprint,
    )
    if stable_fingerprint(formal_result) != formal_result_fingerprint:
        raise ValueError("PAET formal-result fingerprint mismatch")
    receipt_sha256 = file_sha256(members[_RECEIPT_NAME])
    seal = _LoadedCoverageStatePAETFormalArtifactSeal(
        model=model,
        model_config=model_config,
        formal_result_payload=formal_result,
        training_payload=training,
        epoch_logs=epoch_logs,
        source_directory=source,
        training_model_fingerprint=training_model_fingerprint,
        module_state_fingerprint=loaded_module_state_fingerprint,
        formal_result_fingerprint=formal_result_fingerprint,
        training_fingerprint=training_fingerprint,
        epoch_log_fingerprint=epoch_log_fingerprint,
        authorization_fingerprint=authorization_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
        receipt_sha256=receipt_sha256,
        weights_sha256=hashes[_WEIGHTS_NAME],
        formal_result_sha256=hashes[_FORMAL_RESULT_NAME],
        training_sha256=hashes[_TRAINING_NAME],
        epoch_log_sha256=hashes[_EPOCH_LOG_NAME],
    )
    return LoadedCoverageStatePAETFormalArtifact(
        model=model,
        model_config=model_config,
        formal_result_payload=formal_result,
        training_payload=training,
        epoch_logs=epoch_logs,
        training_model_fingerprint=training_model_fingerprint,
        module_state_fingerprint=loaded_module_state_fingerprint,
        formal_result_fingerprint=formal_result_fingerprint,
        training_fingerprint=training_fingerprint,
        epoch_log_fingerprint=epoch_log_fingerprint,
        authorization_fingerprint=authorization_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
        source_directory=source,
        receipt_sha256=receipt_sha256,
        weights_sha256=hashes[_WEIGHTS_NAME],
        formal_result_sha256=hashes[_FORMAL_RESULT_NAME],
        training_sha256=hashes[_TRAINING_NAME],
        epoch_log_sha256=hashes[_EPOCH_LOG_NAME],
        _verification_token=seal,
    )


__all__ = [
    "COVERAGE_STATE_PAET_FORMAL_ARTIFACT_SCHEMA",
    "LoadedCoverageStatePAETFormalArtifact",
    "load_coverage_state_paet_formal_artifact",
    "save_coverage_state_paet_formal_artifact",
]
