"""Pure in-memory D_V evaluation for the PACRE-VC v23 Formal800 model.

The evaluator has one fixed PACRE operating point::

    occupancy = base_probability >= 0.72
    completion = (pacre_field < 0) & ~occupancy
    final = occupancy | completion

An exact zero field value is therefore not completion.  PACRE has no
threshold search; the only searched operating point is the historical
``Base@B`` control over the frozen 51-point base-probability grid.

This module deliberately contains no result writer and no D_T entry point.
Importing it does not load D_V.  Runtime evaluation accepts only the exact
v23 model, strict D_V cache bundle, frozen comparison protocol, and a
module-issued binding to a strictly loaded v23 Formal800 artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field as dataclass_field, fields, replace
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path
from typing import Final, Sequence

import torch
from torch import Tensor

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.calibration import CalibrationSample, FalseAlarmBudget
from cure_lite.calibration_ledger import (
    CalibrationCandidateLedger,
    CandidateEvaluation,
    PreparedCalibrationContext,
    prepare_calibration_context,
)
from cure_lite.config import MatchConfig, OccupancyConfig
from cure_lite.experiment.cache_pipeline import LoadedDVCacheBundle
from cure_lite.experiment.evaluation_pipeline import (
    calibration_samples_fingerprint,
)
from cure_lite.experiment.paired_formal_evaluation import (
    FORMAL_DV_ANCHOR_COVERED,
    FORMAL_DV_ANCHOR_MISSES,
    FORMAL_DV_IMAGES,
    FORMAL_DV_TOTAL_TARGETS,
    FrozenComparisonProtocol,
)
from cure_lite.frozen_base import module_state_fingerprint
from cure_lite.instances import instances_from_binary_mask
from cure_lite.metrics import (
    AggregateEvaluation,
    aggregate_evaluations,
    evaluate_binary_prediction_from_instances,
)

from .formal_artifacts import (
    LoadedPACREVCFormalArtifact,
    VerifiedPACREVCFormalTerminal,
    formal_model_config_payload,
    verify_formal_training_ledger,
)
from .pacre_vc import CURELitePACREVerifierCorrectedLevelSet


PACRE_VC_FORMAL_MODEL_BINDING_SCHEMA: Final = (
    "cure-lite-v23-pacre-vc-formal800-model-binding-v1"
)
PACRE_VC_FORMAL_DV_RESULT_SCHEMA: Final = (
    "cure-lite-v23-pacre-vc-formal-d-v-result-v1"
)
PACRE_VC_FORMAL_METHOD: Final = "PACRE-VC-v23"
PACRE_VC_FORMAL_SEED: Final = 42
PACRE_VC_FORMAL_EPOCHS: Final = 800
PACRE_VC_FORMAL_STEPS_PER_EPOCH: Final = 40
PACRE_VC_FORMAL_UPDATES: Final = (
    PACRE_VC_FORMAL_EPOCHS * PACRE_VC_FORMAL_STEPS_PER_EPOCH
)
PACRE_VC_FORMAL_BATCH_SIZE: Final = 8
PACRE_VC_FORMAL_BASE_THRESHOLD: Final = 0.72
PACRE_VC_FORMAL_BASE_THRESHOLD_GRID: Final = tuple(
    index / 50 for index in range(51)
)
PACRE_VC_FORMAL_STAGE_A_CONFIG_SHA256: Final = (
    "6eecdc10f87a043cafb945db40d0b767b5f0a2ccb64963c1043160f165ce9d6c"
)
PACRE_VC_FIXED_OUTPUT_RULE: Final = (
    "occupancy=(base_probability>=0.72);"
    "completion=(pacre_field<0)&~occupancy;"
    "final=occupancy|completion"
)
PACRE_VC_ZERO_TIE_POLICY: Final = (
    "pacre_field_equal_zero_is_not_completion"
)
PACRE_VC_BASE_AT_B_SELECTION_POLICY: Final = (
    "base_probability_only_existing_51_point_grid_under_frozen_budget"
)
PACRE_VC_MAXIMUM_PIXEL_FA: Final = 1.0e-4
PACRE_VC_MAXIMUM_RAW_BACKGROUND_FA: Final = 1.0e-4
PACRE_VC_MAXIMUM_FP_COMPONENTS_PER_MP: Final = 100.0

_HEX = frozenset("0123456789abcdef")
_AGGREGATE_FIELDS = tuple(
    aggregate_field.name for aggregate_field in fields(AggregateEvaluation)
)
_FORMAL_OCCUPANCY_CONFIG = OccupancyConfig(
    threshold=PACRE_VC_FORMAL_BASE_THRESHOLD,
    connectivity=8,
    min_component_area=1,
)
_FORMAL_MATCH_CONFIG = MatchConfig(
    max_distance=3.0,
    distance_quantization=1_000_000,
    iou_quantization=1_000_000,
)
_FORMAL_BUDGET = FalseAlarmBudget(
    pixel_fa_budget=PACRE_VC_MAXIMUM_PIXEL_FA,
    component_fa_per_mp_budget=(
        PACRE_VC_MAXIMUM_FP_COMPONENTS_PER_MP
    ),
    raw_background_fa_budget=(
        PACRE_VC_MAXIMUM_RAW_BACKGROUND_FA
    ),
    minimum_retention=0.99,
)


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _base_grid(values: Sequence[float]) -> tuple[float, ...]:
    resolved: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("Base@B threshold grid values must be real")
        threshold = float(value)
        if not isfinite(threshold):
            raise ValueError("Base@B threshold grid must be finite")
        resolved.append(threshold)
    grid = tuple(resolved)
    if grid != PACRE_VC_FORMAL_BASE_THRESHOLD_GRID:
        raise ValueError(
            "Base@B must use the existing ordered 51-point base grid"
        )
    return grid


def _stage_a_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "protocols/IRSTD-1K/stage_a_seed42_fx_v3/stage_a_config.json"
    )


def _verified_stage_a_base_grid() -> tuple[float, ...]:
    """Verify the frozen metadata source of the Base@B grid.

    This reads protocol metadata only.  It does not open D_V images, masks,
    tensors, or cache payloads.
    """

    path = _stage_a_config_path()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("frozen stage_a_config.json is unavailable")
    source = path.read_bytes()
    if sha256(source).hexdigest() != PACRE_VC_FORMAL_STAGE_A_CONFIG_SHA256:
        raise RuntimeError("frozen stage_a_config.json bytes changed")
    try:
        payload = json.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("frozen stage_a_config.json is invalid") from error
    if not isinstance(payload, dict):
        raise RuntimeError("frozen stage_a_config.json must be an object")
    return _base_grid(payload.get("base_thresholds", ()))


def fixed_pacre_vc_completion(
    field: Tensor,
    occupancy: Tensor,
) -> Tensor:
    """Decode PACRE at the sole level-set threshold, exactly zero."""

    if (
        not isinstance(field, Tensor)
        or not field.is_floating_point()
        or field.ndim != 4
        or field.shape[0] < 1
        or field.shape[1] != 1
        or min(field.shape[-2:]) < 1
    ):
        raise TypeError("field must be floating [B,1,H,W]")
    if (
        not isinstance(occupancy, Tensor)
        or occupancy.dtype != torch.bool
        or tuple(occupancy.shape) != tuple(field.shape)
    ):
        raise TypeError(
            "occupancy must be bool with the exact field shape"
        )
    if field.device != occupancy.device:
        raise ValueError("field and occupancy must share one device")
    if not bool(torch.isfinite(field).all()):
        raise ValueError("field must contain only finite values")
    return ((field < 0.0) & ~occupancy).contiguous()


def pacre_vc_formal_model_config_fingerprint(
    model: CURELitePACREVerifierCorrectedLevelSet,
) -> str:
    """Bind the exact v23 FQCN, config, and parameter count."""

    if type(model) is not CURELitePACREVerifierCorrectedLevelSet:
        raise TypeError("model must be the exact PACRE-VC v23 class")
    return stable_fingerprint(
        {
            "model_fqcn": (
                "cure_lite_v23.pacre_vc."
                "CURELitePACREVerifierCorrectedLevelSet"
            ),
            "config_fqcn": (
                f"{type(model.config).__module__}."
                f"{type(model.config).__qualname__}"
            ),
            "config": asdict(model.config),
            "expected_parameter_count": (
                model.config.expected_parameter_count
            ),
        }
    )


def _validate_protocol(
    comparison_protocol: FrozenComparisonProtocol,
) -> None:
    if type(comparison_protocol) is not FrozenComparisonProtocol:
        raise TypeError(
            "comparison_protocol must be exact FrozenComparisonProtocol"
        )
    if comparison_protocol.occupancy_config != _FORMAL_OCCUPANCY_CONFIG:
        raise ValueError("PACRE formal Base@A must remain 0.72/CC8")
    if comparison_protocol.match_config != _FORMAL_MATCH_CONFIG:
        raise ValueError("PACRE formal matching protocol changed")
    if comparison_protocol.budget != _FORMAL_BUDGET:
        raise ValueError("PACRE formal false-addition budget changed")
    _base_grid(comparison_protocol.residual_thresholds)


def _validate_runtime_sources(
    artifact: LoadedPACREVCFormalArtifact,
    bundle: LoadedDVCacheBundle,
    comparison_protocol: FrozenComparisonProtocol,
) -> None:
    if type(artifact) is not LoadedPACREVCFormalArtifact:
        raise TypeError(
            "artifact must be exact LoadedPACREVCFormalArtifact"
        )
    if type(bundle) is not LoadedDVCacheBundle:
        raise TypeError("bundle must be exact LoadedDVCacheBundle")
    if type(comparison_protocol) is not FrozenComparisonProtocol:
        raise TypeError(
            "comparison_protocol must be exact FrozenComparisonProtocol"
        )
    artifact.verify_unchanged()
    bundle.verify_unchanged()
    _validate_protocol(comparison_protocol)
    comparison_protocol.verify_bundle(bundle)


@dataclass(frozen=True, slots=True)
class _PACREVCFormalModelBindingSeal:
    issuer: object
    terminal: VerifiedPACREVCFormalTerminal
    bundle: LoadedDVCacheBundle
    comparison_protocol: FrozenComparisonProtocol


_PACRE_VC_FORMAL_MODEL_BINDING_ISSUER = object()


@dataclass(frozen=True)
class PACREVCFormalModelBinding:
    """Unforgeable binding from one strict v23 artifact to one D_V protocol."""

    artifact_fingerprint: str
    formal_result_fingerprint: str
    training_result_fingerprint: str
    authorization_fingerprint: str
    source_closure_fingerprint: str
    model_state_fingerprint: str
    model_config_fingerprint: str
    comparison_protocol_fingerprint: str
    manifest_fingerprint: str
    manifest_file_sha256: str
    preprocessing_fingerprint: str
    base_fingerprint: str
    base_state_fingerprint: str
    d_v_base_index_fingerprint: str
    d_v_image_fingerprint: str
    d_v_gt_fingerprint: str
    seed: int = PACRE_VC_FORMAL_SEED
    epochs: int = PACRE_VC_FORMAL_EPOCHS
    steps_per_epoch: int = PACRE_VC_FORMAL_STEPS_PER_EPOCH
    completed_updates: int = PACRE_VC_FORMAL_UPDATES
    trained_from_scratch: bool = True
    resumed: bool = False
    runtime_splits: tuple[str, ...] = ("D_R",)
    d_v_payload_accessed_during_training: bool = False
    d_t_payload_accessed_during_training: bool = False
    schema_version: str = PACRE_VC_FORMAL_MODEL_BINDING_SCHEMA
    _seal: _PACREVCFormalModelBindingSeal | None = dataclass_field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        seal = self._seal
        if (
            type(seal) is not _PACREVCFormalModelBindingSeal
            or seal.issuer is not _PACRE_VC_FORMAL_MODEL_BINDING_ISSUER
        ):
            raise PermissionError(
                "PACRE-VC formal binding must come from its strict factory"
            )
        if self.schema_version != PACRE_VC_FORMAL_MODEL_BINDING_SCHEMA:
            raise ValueError("unsupported PACRE-VC formal binding schema")
        if (
            self.seed != PACRE_VC_FORMAL_SEED
            or self.epochs != PACRE_VC_FORMAL_EPOCHS
            or self.steps_per_epoch
            != PACRE_VC_FORMAL_STEPS_PER_EPOCH
            or self.completed_updates != PACRE_VC_FORMAL_UPDATES
            or self.trained_from_scratch is not True
            or self.resumed is not False
            or self.runtime_splits != ("D_R",)
            or self.d_v_payload_accessed_during_training is not False
            or self.d_t_payload_accessed_during_training is not False
        ):
            raise ValueError(
                "PACRE-VC D_V requires the seed-42, from-zero, "
                "800x40 Formal800 artifact"
            )
        for name in (
            "artifact_fingerprint",
            "formal_result_fingerprint",
            "training_result_fingerprint",
            "authorization_fingerprint",
            "source_closure_fingerprint",
            "model_state_fingerprint",
            "model_config_fingerprint",
            "comparison_protocol_fingerprint",
            "manifest_fingerprint",
            "manifest_file_sha256",
            "preprocessing_fingerprint",
            "base_fingerprint",
            "base_state_fingerprint",
            "d_v_base_index_fingerprint",
            "d_v_image_fingerprint",
            "d_v_gt_fingerprint",
        ):
            _digest(getattr(self, name), name=name)
        self.verify_unchanged()

    def _sealed_sources(self) -> _PACREVCFormalModelBindingSeal:
        seal = self._seal
        if (
            type(seal) is not _PACREVCFormalModelBindingSeal
            or seal.issuer is not _PACRE_VC_FORMAL_MODEL_BINDING_ISSUER
        ):
            raise PermissionError("PACRE-VC formal binding is unsealed")
        return seal

    def verify_unchanged(self) -> None:
        seal = self._sealed_sources()
        terminal = seal.terminal
        terminal.verify_unchanged()
        artifact = terminal.artifact
        bundle = seal.bundle
        protocol = seal.comparison_protocol
        _validate_runtime_sources(artifact, bundle, protocol)
        receipt = artifact.receipt
        model = artifact.model
        if (
            type(model) is not CURELitePACREVerifierCorrectedLevelSet
            or self.artifact_fingerprint
            != artifact.artifact_fingerprint
            or self.formal_result_fingerprint
            != receipt.get("formal_result_fingerprint")
            or self.training_result_fingerprint
            != receipt.get("training_result_fingerprint")
            or self.authorization_fingerprint
            != receipt.get("authorization_fingerprint")
            or self.source_closure_fingerprint
            != receipt.get("source_closure_fingerprint")
            or receipt.get("final_checkpoint_only") is not True
            or receipt.get("optimizer_state_saved") is not False
            or receipt.get("intermediate_checkpoint_saved") is not False
            or receipt.get("D_V_payload_accessed") is not False
            or receipt.get("D_T_payload_accessed") is not False
            or receipt.get("performance_evaluation_performed") is not False
            or not isinstance(
                receipt.get("formal_training_ledger"),
                dict,
            )
            or self.model_state_fingerprint
            != module_state_fingerprint(model)
            or self.model_state_fingerprint
            != receipt.get("module_state_fingerprint")
            or self.model_config_fingerprint
            != pacre_vc_formal_model_config_fingerprint(model)
            or receipt.get("model_config_fingerprint")
            != stable_fingerprint(formal_model_config_payload(model.config))
            or self.comparison_protocol_fingerprint
            != protocol.comparison_protocol_fingerprint
            or self.manifest_fingerprint
            != bundle.split_manifest_fingerprint
            or self.manifest_file_sha256
            != bundle.split_manifest_file_sha256
            or self.preprocessing_fingerprint
            != bundle.preprocessing_fingerprint
            or self.base_fingerprint != bundle.base_fingerprint
            or self.base_state_fingerprint
            != bundle.base_state_fingerprint
            or self.d_v_base_index_fingerprint
            != bundle.base_index_fingerprint
            or self.d_v_image_fingerprint
            != bundle.d_v_image_fingerprint
            or self.d_v_gt_fingerprint != bundle.d_v_gt_fingerprint
        ):
            raise RuntimeError(
                "PACRE-VC formal artifact/D_V binding changed"
            )
        verify_formal_training_ledger(
            receipt["formal_training_ledger"],
            model=model,
            training_result_fingerprint=self.training_result_fingerprint,
        )

    def verify_inputs(
        self,
        model: CURELitePACREVerifierCorrectedLevelSet,
        bundle: LoadedDVCacheBundle,
        comparison_protocol: FrozenComparisonProtocol,
    ) -> None:
        self.verify_unchanged()
        seal = self._sealed_sources()
        if (
            type(model) is not CURELitePACREVerifierCorrectedLevelSet
            or type(bundle) is not LoadedDVCacheBundle
            or type(comparison_protocol) is not FrozenComparisonProtocol
        ):
            raise TypeError(
                "PACRE-VC evaluation inputs must have exact formal types"
            )
        if (
            model is not seal.terminal.artifact.model
            or bundle is not seal.bundle
            or comparison_protocol is not seal.comparison_protocol
        ):
            raise RuntimeError(
                "PACRE-VC formal binding rejects substituted inputs"
            )

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "method": PACRE_VC_FORMAL_METHOD,
            "formal_budget": {
                "seed": self.seed,
                "epochs": self.epochs,
                "steps_per_epoch": self.steps_per_epoch,
                "completed_updates": self.completed_updates,
                "trained_from_scratch": self.trained_from_scratch,
                "resumed": self.resumed,
                "runtime_splits": list(self.runtime_splits),
                "D_V_payload_accessed": (
                    self.d_v_payload_accessed_during_training
                ),
                "D_T_payload_accessed": (
                    self.d_t_payload_accessed_during_training
                ),
            },
            "formal_artifact": {
                "artifact_fingerprint": self.artifact_fingerprint,
                "formal_result_fingerprint": (
                    self.formal_result_fingerprint
                ),
                "training_result_fingerprint": (
                    self.training_result_fingerprint
                ),
                "authorization_fingerprint": (
                    self.authorization_fingerprint
                ),
                "source_closure_fingerprint": (
                    self.source_closure_fingerprint
                ),
                "model_state_fingerprint": (
                    self.model_state_fingerprint
                ),
                "model_config_fingerprint": (
                    self.model_config_fingerprint
                ),
            },
            "D_V_protocol": {
                "comparison_protocol_fingerprint": (
                    self.comparison_protocol_fingerprint
                ),
                "manifest_fingerprint": self.manifest_fingerprint,
                "manifest_file_sha256": self.manifest_file_sha256,
                "preprocessing_fingerprint": (
                    self.preprocessing_fingerprint
                ),
                "base_fingerprint": self.base_fingerprint,
                "base_state_fingerprint": self.base_state_fingerprint,
                "base_index_fingerprint": (
                    self.d_v_base_index_fingerprint
                ),
                "image_fingerprint": self.d_v_image_fingerprint,
                "GT_fingerprint": self.d_v_gt_fingerprint,
            },
        }

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return self._canonical_payload()

    @property
    def binding_fingerprint(self) -> str:
        self.verify_unchanged()
        return stable_fingerprint(self._canonical_payload())


def bind_pacre_vc_formal_model(
    terminal: VerifiedPACREVCFormalTerminal,
    comparison_protocol: FrozenComparisonProtocol,
    bundle: LoadedDVCacheBundle,
) -> PACREVCFormalModelBinding:
    """Bind one strict final-only v23 artifact to one exact D_V bundle."""

    if type(terminal) is not VerifiedPACREVCFormalTerminal:
        raise TypeError(
            "formal model binding requires an exact verifier-issued "
            "VerifiedPACREVCFormalTerminal"
        )
    terminal.verify_unchanged()
    artifact = terminal.artifact
    _validate_runtime_sources(artifact, bundle, comparison_protocol)
    receipt = artifact.receipt
    for name in (
        "artifact_fingerprint",
        "formal_result_fingerprint",
        "training_result_fingerprint",
        "authorization_fingerprint",
        "source_closure_fingerprint",
        "module_state_fingerprint",
        "model_config_fingerprint",
    ):
        _digest(receipt.get(name), name=name)
    model = artifact.model
    if (
        type(model) is not CURELitePACREVerifierCorrectedLevelSet
        or receipt.get("final_checkpoint_only") is not True
        or receipt.get("optimizer_state_saved") is not False
        or receipt.get("intermediate_checkpoint_saved") is not False
        or receipt.get("D_V_payload_accessed") is not False
        or receipt.get("D_T_payload_accessed") is not False
        or receipt.get("performance_evaluation_performed") is not False
        or not isinstance(receipt.get("formal_training_ledger"), dict)
        or receipt.get("module_state_fingerprint")
        != module_state_fingerprint(model)
        or receipt.get("model_config_fingerprint")
        != stable_fingerprint(formal_model_config_payload(model.config))
    ):
        raise PermissionError(
            "loaded artifact is not the completed final-only v23 model"
        )
    verify_formal_training_ledger(
        receipt["formal_training_ledger"],
        model=model,
        training_result_fingerprint=str(
            receipt["training_result_fingerprint"]
        ),
    )
    seal = _PACREVCFormalModelBindingSeal(
        issuer=_PACRE_VC_FORMAL_MODEL_BINDING_ISSUER,
        terminal=terminal,
        bundle=bundle,
        comparison_protocol=comparison_protocol,
    )
    return PACREVCFormalModelBinding(
        artifact_fingerprint=artifact.artifact_fingerprint,
        formal_result_fingerprint=str(
            receipt["formal_result_fingerprint"]
        ),
        training_result_fingerprint=str(
            receipt["training_result_fingerprint"]
        ),
        authorization_fingerprint=str(
            receipt["authorization_fingerprint"]
        ),
        source_closure_fingerprint=str(
            receipt["source_closure_fingerprint"]
        ),
        model_state_fingerprint=module_state_fingerprint(model),
        model_config_fingerprint=(
            pacre_vc_formal_model_config_fingerprint(model)
        ),
        comparison_protocol_fingerprint=(
            comparison_protocol.comparison_protocol_fingerprint
        ),
        manifest_fingerprint=bundle.split_manifest_fingerprint,
        manifest_file_sha256=bundle.split_manifest_file_sha256,
        preprocessing_fingerprint=bundle.preprocessing_fingerprint,
        base_fingerprint=bundle.base_fingerprint,
        base_state_fingerprint=bundle.base_state_fingerprint,
        d_v_base_index_fingerprint=bundle.base_index_fingerprint,
        d_v_image_fingerprint=bundle.d_v_image_fingerprint,
        d_v_gt_fingerprint=bundle.d_v_gt_fingerprint,
        _seal=seal,
    )


@dataclass(frozen=True)
class _PACREVCFixedDVSamples:
    base_samples: tuple[CalibrationSample, ...]
    cure_samples: tuple[CalibrationSample, ...]
    ordered_sample_ids: tuple[str, ...]
    base_samples_fingerprint: str
    cure_samples_fingerprint: str
    exact_zero_field_pixels: int
    negative_field_pixels: int
    completion_pixels: int

    def __post_init__(self) -> None:
        if (
            len(self.base_samples) != FORMAL_DV_IMAGES
            or len(self.cure_samples) != FORMAL_DV_IMAGES
            or len(self.ordered_sample_ids) != FORMAL_DV_IMAGES
            or tuple(
                sample.sample_id for sample in self.base_samples
            )
            != self.ordered_sample_ids
            or tuple(
                sample.sample_id for sample in self.cure_samples
            )
            != self.ordered_sample_ids
        ):
            raise ValueError(
                "PACRE-VC formal D_V samples must preserve 120 aligned rows"
            )
        if (
            len(set(self.ordered_sample_ids)) != FORMAL_DV_IMAGES
            or tuple(sorted(self.ordered_sample_ids))
            != self.ordered_sample_ids
        ):
            raise ValueError(
                "PACRE-VC formal D_V sample IDs must be sorted and unique"
            )
        for name in (
            "base_samples_fingerprint",
            "cure_samples_fingerprint",
        ):
            _digest(getattr(self, name), name=name)
        if (
            calibration_samples_fingerprint(self.base_samples)
            != self.base_samples_fingerprint
            or calibration_samples_fingerprint(self.cure_samples)
            != self.cure_samples_fingerprint
        ):
            raise RuntimeError("PACRE-VC D_V sample tensors changed")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in (
                self.exact_zero_field_pixels,
                self.negative_field_pixels,
                self.completion_pixels,
            )
        ):
            raise ValueError(
                "PACRE-VC field pixel counts must be nonnegative integers"
            )


def _build_fixed_d_v_samples(
    model: CURELitePACREVerifierCorrectedLevelSet,
    bundle: LoadedDVCacheBundle,
    comparison_protocol: FrozenComparisonProtocol,
    model_binding: PACREVCFormalModelBinding,
    *,
    batch_size: int,
) -> _PACREVCFixedDVSamples:
    model_binding.verify_inputs(model, bundle, comparison_protocol)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size != PACRE_VC_FORMAL_BATCH_SIZE
    ):
        raise ValueError("PACRE-VC formal D_V fixes batch_size=8")
    if len(bundle.rows) != FORMAL_DV_IMAGES:
        raise ValueError("PACRE-VC formal D_V requires exactly 120 rows")
    feature_channels = {
        int(row.base_output.feature.shape[1]) for row in bundle.rows
    }
    if feature_channels != {model.config.feature_channels}:
        raise RuntimeError(
            "D_V feature channels differ from the PACRE-VC artifact"
        )
    parameters = tuple(model.parameters())
    if not parameters:
        raise RuntimeError("PACRE-VC model unexpectedly has no parameters")
    devices = {parameter.device for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    if len(devices) != 1 or dtypes != {torch.float32}:
        raise RuntimeError(
            "PACRE-VC D_V evaluation requires one-device FP32 parameters"
        )
    device = next(iter(devices))

    base_samples: list[CalibrationSample] = []
    cure_samples: list[CalibrationSample] = []
    exact_zero_field_pixels = 0
    negative_field_pixels = 0
    completion_pixels = 0
    initial_state = module_state_fingerprint(model)
    was_training = model.training
    try:
        model.eval()
        for start in range(0, len(bundle.rows), batch_size):
            rows = bundle.rows[start : start + batch_size]
            features = torch.cat(
                [row.base_output.feature for row in rows],
                dim=0,
            ).to(device=device, dtype=torch.float32)
            base_probabilities = torch.cat(
                [row.base_output.probability for row in rows],
                dim=0,
            ).detach().to(device="cpu", dtype=torch.float32)
            occupancies = (
                base_probabilities >= PACRE_VC_FORMAL_BASE_THRESHOLD
            ).to(device=device)
            with torch.no_grad():
                field = model(features, occupancies)
                if (
                    tuple(field.shape) != tuple(occupancies.shape)
                    or field.dtype != torch.float32
                    or not bool(torch.isfinite(field).all())
                ):
                    raise RuntimeError(
                        "PACRE-VC field violates the formal output contract"
                    )
                completion = fixed_pacre_vc_completion(
                    field,
                    occupancies,
                )
            exact_zero_field_pixels += int(
                torch.count_nonzero(field == 0.0).item()
            )
            negative_field_pixels += int(
                torch.count_nonzero(field < 0.0).item()
            )
            completion_pixels += int(
                torch.count_nonzero(completion).item()
            )
            completion_cpu = completion.to(device="cpu")
            for index, row in enumerate(rows):
                base = row.base_output.probability.detach().to(
                    device="cpu",
                    dtype=torch.float32,
                )
                gt = row.gt_mask.detach().to(device="cpu")
                zero = torch.zeros_like(base)
                fixed_completion = completion_cpu[
                    index : index + 1
                ].to(dtype=torch.float32)
                base_sample = CalibrationSample(
                    row.sample_id,
                    base,
                    zero,
                    gt,
                )
                cure_sample = CalibrationSample(
                    row.sample_id,
                    base,
                    fixed_completion,
                    gt,
                )
                normalized_base, normalized_zero, normalized_gt = (
                    base_sample.normalized()
                )
                cure_base, normalized_completion, cure_gt = (
                    cure_sample.normalized()
                )
                if (
                    not torch.equal(normalized_base, cure_base)
                    or not torch.equal(normalized_gt, cure_gt)
                    or bool(
                        torch.any(
                            (normalized_completion != 0.0)
                            & (normalized_completion != 1.0)
                        )
                    )
                ):
                    raise RuntimeError(
                        "PACRE-VC fixed sample normalization changed"
                    )
                base_samples.append(
                    CalibrationSample(
                        row.sample_id,
                        normalized_base,
                        normalized_zero,
                        normalized_gt,
                    )
                )
                cure_samples.append(
                    CalibrationSample(
                        row.sample_id,
                        cure_base,
                        normalized_completion,
                        cure_gt,
                    )
                )
    finally:
        model.train(was_training)

    if module_state_fingerprint(model) != initial_state:
        raise RuntimeError("PACRE-VC model changed during D_V evaluation")
    model_binding.verify_inputs(model, bundle, comparison_protocol)
    base_tuple = tuple(base_samples)
    cure_tuple = tuple(cure_samples)
    return _PACREVCFixedDVSamples(
        base_samples=base_tuple,
        cure_samples=cure_tuple,
        ordered_sample_ids=tuple(row.sample_id for row in bundle.rows),
        base_samples_fingerprint=calibration_samples_fingerprint(
            base_tuple
        ),
        cure_samples_fingerprint=calibration_samples_fingerprint(
            cure_tuple
        ),
        exact_zero_field_pixels=exact_zero_field_pixels,
        negative_field_pixels=negative_field_pixels,
        completion_pixels=completion_pixels,
    )


@dataclass(frozen=True)
class _PACREVCFixedOperatingPoints:
    base_at_a: AggregateEvaluation
    base_at_b: AggregateEvaluation
    base_at_a_plus_cure: AggregateEvaluation
    base_at_b_selected_threshold: float
    base_candidate_ledger: CalibrationCandidateLedger


def _evaluate_full_base_candidate_ledger(
    context: PreparedCalibrationContext,
    *,
    grid: tuple[float, ...],
    budget: FalseAlarmBudget,
) -> CalibrationCandidateLedger:
    """Evaluate the declared Base@B grid without the common anchor cutoff.

    The shared calibration ledger intentionally drops base thresholds above
    the fixed Base@A anchor because its usual role is false-addition
    calibration.  PACRE-VC's frozen comparator contract is different: it
    declares all 51 values in ``{0, .02, ..., 1}``.  Evaluate that complete
    grid locally so the persisted evidence and the computation agree.
    """

    if type(context) is not PreparedCalibrationContext:
        raise TypeError("Base@B requires a prepared calibration context")
    anchor_threshold = PACRE_VC_FORMAL_BASE_THRESHOLD
    anchor_metrics = replace(
        context.anchor_metrics,
        budget_violation=not budget.accepts(context.anchor_metrics),
    )
    if not budget.accepts(anchor_metrics):
        raise ValueError(
            "the frozen budget cannot reject the Base@A anchor"
        )
    entries: list[CandidateEvaluation] = []
    for threshold in grid:
        if threshold == anchor_threshold:
            metrics = anchor_metrics
        else:
            evaluations = []
            for row in context.rows:
                prediction = row.base_probability >= threshold
                pred_instances = instances_from_binary_mask(
                    prediction,
                    connectivity=8,
                    min_area=1,
                )
                evaluations.append(
                    evaluate_binary_prediction_from_instances(
                        prediction,
                        row.gt_mask,
                        pred_instances,
                        row.gt_instances,
                        context.match_config,
                        anchor_miss_ids=row.anchor_miss_ids,
                        reachable_anchor_miss_ids=(
                            row.reachable_anchor_miss_ids
                        ),
                        residual_mask=None,
                    )
                )
            raw_metrics = aggregate_evaluations(evaluations)
            metrics = replace(
                raw_metrics,
                budget_violation=not budget.accepts(raw_metrics),
            )
        entries.append(
            CandidateEvaluation(
                method="Base@B",
                mode="base",
                threshold=threshold,
                metrics=metrics,
            )
        )
    ledger = CalibrationCandidateLedger(
        base_method="Base@B",
        anchor_threshold=anchor_threshold,
        anchor_metrics=anchor_metrics,
        entries=tuple(entries),
    )
    if (
        ledger.methods != ("Base@B",)
        or not grid
        or len(ledger.entries) != len(grid)
        or tuple(entry.threshold for entry in ledger.entries) != grid
        or any(entry.mode != "base" for entry in ledger.entries)
    ):
        raise RuntimeError("Base@B did not evaluate the exact 51-point grid")
    return ledger


def _evaluate_fixed_operating_points(
    base_samples: Sequence[CalibrationSample],
    cure_samples: Sequence[CalibrationSample],
    *,
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
    base_threshold_grid: Sequence[float],
    budget: FalseAlarmBudget,
) -> _PACREVCFixedOperatingPoints:
    """Search Base@B only and evaluate PACRE by a fixed hard union."""

    if occupancy_config != _FORMAL_OCCUPANCY_CONFIG:
        raise ValueError("PACRE-VC formal Base@A must remain 0.72/CC8")
    if match_config != _FORMAL_MATCH_CONFIG:
        raise ValueError("PACRE-VC formal matching protocol changed")
    if budget != _FORMAL_BUDGET:
        raise ValueError("PACRE-VC formal budget changed")
    grid = _base_grid(base_threshold_grid)
    base_tuple = tuple(base_samples)
    cure_tuple = tuple(cure_samples)
    if (
        not base_tuple
        or len(base_tuple) != len(cure_tuple)
        or tuple(sample.sample_id for sample in base_tuple)
        != tuple(sample.sample_id for sample in cure_tuple)
    ):
        raise ValueError(
            "base and PACRE-VC samples must be non-empty/aligned"
        )
    context = prepare_calibration_context(
        base_tuple,
        occupancy_config,
        match_config,
    )
    ledger = _evaluate_full_base_candidate_ledger(
        context,
        grid=grid,
        budget=budget,
    )
    selection = ledger.select("Base@B", budget)
    if (
        not selection.feasible
        or selection.threshold is None
        or selection.metrics is None
    ):
        raise RuntimeError("frozen Base@B selection is infeasible")

    cure_by_id = {sample.sample_id: sample for sample in cure_tuple}
    evaluations = []
    for row in context.rows:
        _, completion_probability, cure_gt = cure_by_id[
            row.sample_id
        ].normalized()
        if (
            not torch.equal(cure_gt, row.gt_mask)
            or bool(
                torch.any(
                    (completion_probability != 0.0)
                    & (completion_probability != 1.0)
                )
            )
        ):
            raise ValueError(
                "PACRE-VC completion must share GT and be exactly binary"
            )
        completion = completion_probability.to(torch.bool)
        if bool(torch.any(completion & row.occupancy)):
            raise ValueError(
                "PACRE-VC completion overlaps Base@A occupancy"
            )
        final = row.occupancy | completion
        pred_instances = instances_from_binary_mask(
            final,
            connectivity=8,
            min_area=1,
        )
        evaluations.append(
            evaluate_binary_prediction_from_instances(
                final,
                row.gt_mask,
                pred_instances,
                row.gt_instances,
                context.match_config,
                anchor_miss_ids=row.anchor_miss_ids,
                reachable_anchor_miss_ids=(
                    row.reachable_anchor_miss_ids
                ),
                residual_mask=completion,
            )
        )

    def with_budget(
        metrics: AggregateEvaluation,
    ) -> AggregateEvaluation:
        return replace(
            metrics,
            budget_violation=not budget.accepts(metrics),
        )

    return _PACREVCFixedOperatingPoints(
        base_at_a=ledger.anchor_metrics,
        base_at_b=selection.metrics,
        base_at_a_plus_cure=with_budget(
            aggregate_evaluations(evaluations)
        ),
        base_at_b_selected_threshold=float(selection.threshold),
        base_candidate_ledger=ledger,
    )


def _aggregate_payload(
    metrics: AggregateEvaluation,
) -> dict[str, object]:
    if not isinstance(metrics, AggregateEvaluation):
        raise TypeError("metrics must be AggregateEvaluation")
    return {
        name: getattr(metrics, name) for name in _AGGREGATE_FIELDS
    }


def _summary_payload(
    metrics: AggregateEvaluation,
) -> dict[str, object]:
    return {
        "true_targets": (
            metrics.retained_anchor_covered
            + metrics.recovered_anchor_misses
        ),
        "Pd": metrics.pd,
        "mIoU": metrics.miou,
        "nIoU": metrics.niou,
        "pixel_Fa": metrics.pixel_fa,
        "raw_background_Fa": metrics.raw_background_fa,
        "false_positive_components_per_megapixel": (
            metrics.fp_components_per_mp
        ),
        "recovered_anchor_misses": (
            metrics.recovered_anchor_misses
        ),
        "retained_anchor_covered": (
            metrics.retained_anchor_covered
        ),
        "total_anchor_misses": metrics.total_anchor_misses,
        "total_anchor_covered": metrics.total_anchor_covered,
        "retention": metrics.retention,
        "budget_violation": metrics.budget_violation,
    }


def _validate_metrics(
    metrics: AggregateEvaluation,
    *,
    name: str,
) -> tuple[int, int]:
    if not isinstance(metrics, AggregateEvaluation):
        raise TypeError(f"{name} must be AggregateEvaluation")
    if (
        metrics.images != FORMAL_DV_IMAGES
        or metrics.total_anchor_misses != FORMAL_DV_ANCHOR_MISSES
        or metrics.total_anchor_covered != FORMAL_DV_ANCHOR_COVERED
        or (
            metrics.total_anchor_misses
            + metrics.total_anchor_covered
        )
        != FORMAL_DV_TOTAL_TARGETS
    ):
        raise ValueError(
            f"{name} must bind the frozen 120/170/23/147 population"
        )
    if not (
        0
        <= metrics.recovered_anchor_misses
        <= metrics.total_anchor_misses
        and 0
        <= metrics.retained_anchor_covered
        <= metrics.total_anchor_covered
    ):
        raise ValueError(f"{name} target counts are inconsistent")
    true_targets = (
        metrics.retained_anchor_covered
        + metrics.recovered_anchor_misses
    )
    if not isclose(
        metrics.pd,
        true_targets / FORMAL_DV_TOTAL_TARGETS,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{name} Pd is inconsistent with target counts")
    if not isclose(
        metrics.retention,
        metrics.retained_anchor_covered
        / FORMAL_DV_ANCHOR_COVERED,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            f"{name} retention is inconsistent with covered counts"
        )
    unit_values = (
        metrics.pd,
        metrics.rmr,
        metrics.gross_rmr,
        metrics.retention,
        metrics.reachable_rmr,
        metrics.oracle_upper_bound,
        metrics.overlap_supported_rmr,
        metrics.miou,
        metrics.niou,
    )
    if any(
        not isfinite(value) or not 0.0 <= value <= 1.0
        for value in unit_values
    ):
        raise ValueError(f"{name} unit metrics must be finite in [0,1]")
    if any(
        not isfinite(value) or value < 0.0
        for value in (
            metrics.pixel_fa,
            metrics.raw_background_fa,
            metrics.fp_components_per_mp,
        )
    ):
        raise ValueError(
            f"{name} false-addition metrics must be finite/nonnegative"
        )
    if not isinstance(metrics.budget_violation, bool):
        raise TypeError(f"{name} budget_violation must be bool")
    return true_targets, metrics.recovered_anchor_misses


def _base_candidate_ledger_payload(
    ledger: CalibrationCandidateLedger,
    *,
    budget: FalseAlarmBudget,
) -> dict[str, object]:
    """Return independently reselectable evidence for all 51 Base@B rows."""

    if type(ledger) is not CalibrationCandidateLedger:
        raise TypeError(
            "Base@B candidate ledger must have the exact common type"
        )
    if (
        ledger.base_method != "Base@B"
        or ledger.anchor_threshold != PACRE_VC_FORMAL_BASE_THRESHOLD
        or ledger.methods != ("Base@B",)
        or len(ledger.entries)
        != len(PACRE_VC_FORMAL_BASE_THRESHOLD_GRID)
        or tuple(entry.threshold for entry in ledger.entries)
        != PACRE_VC_FORMAL_BASE_THRESHOLD_GRID
        or any(entry.mode != "base" for entry in ledger.entries)
    ):
        raise ValueError("Base@B ledger is not the exact 51-point grid")
    entries: list[dict[str, object]] = []
    for entry in ledger.entries:
        _validate_metrics(
            entry.metrics,
            name=f"Base@B[{entry.threshold}]",
        )
        accepted = budget.accepts(entry.metrics)
        if entry.metrics.budget_violation is accepted:
            raise ValueError(
                "Base@B candidate budget flag differs from its metrics"
            )
        entries.append(
            {
                "threshold": entry.threshold,
                "aggregate_evaluation": _aggregate_payload(
                    entry.metrics
                ),
                "budget_accepted": accepted,
            }
        )
    body: dict[str, object] = {
        "schema_version": (
            "cure-lite-v23-pacre-vc-base-at-b-51-ledger-v1"
        ),
        "method": "Base@B",
        "mode": "base",
        "anchor_threshold": ledger.anchor_threshold,
        "candidate_count": len(entries),
        "entries": entries,
    }
    return {
        **body,
        "ledger_fingerprint": stable_fingerprint(body),
    }


@dataclass(frozen=True)
class _PACREVCGateAssessment:
    valid_base_names: tuple[str, ...]
    best_base_true_targets: int
    best_base_recovered_anchor_misses: int
    best_base_miou: float
    best_base_niou: float
    true_target_margin: int
    recovered_anchor_miss_margin: int
    checks: tuple[tuple[str, bool], ...]

    @property
    def gate_passed(self) -> bool:
        return bool(self.checks) and all(
            value for _, value in self.checks
        )


def _assess_gate(
    *,
    base_at_a: AggregateEvaluation,
    base_at_b: AggregateEvaluation,
    base_at_a_plus_cure: AggregateEvaluation,
    budget: FalseAlarmBudget,
) -> _PACREVCGateAssessment:
    base_a_true, base_a_recovered = _validate_metrics(
        base_at_a,
        name="Base@A",
    )
    base_b_true, base_b_recovered = _validate_metrics(
        base_at_b,
        name="Base@B",
    )
    cure_true, cure_recovered = _validate_metrics(
        base_at_a_plus_cure,
        name="Base@A+CURE",
    )
    base_rows = (
        ("Base@A", base_at_a, base_a_true, base_a_recovered),
        ("Base@B", base_at_b, base_b_true, base_b_recovered),
    )
    valid = tuple(
        row
        for row in base_rows
        if row[1].budget_violation is False
        and budget.accepts(row[1])
    )
    if not valid:
        raise ValueError(
            "no valid Base comparator remains under the frozen budget"
        )
    best_base_true = max(row[2] for row in valid)
    best_base_recovered = max(row[3] for row in valid)
    best_base_miou = max(row[1].miou for row in valid)
    best_base_niou = max(row[1].niou for row in valid)
    true_target_margin = cure_true - best_base_true
    recovered_margin = cure_recovered - best_base_recovered
    checks = tuple(
        sorted(
            {
                "CURE_true_targets_strictly_above_best_valid_Base": (
                    cure_true > best_base_true
                ),
                "CURE_recovered_anchor_misses_strictly_above_best_valid_Base": (
                    cure_recovered > best_base_recovered
                ),
                "CURE_mIoU_not_below_best_valid_Base": (
                    base_at_a_plus_cure.miou >= best_base_miou
                ),
                "CURE_nIoU_not_below_best_valid_Base": (
                    base_at_a_plus_cure.niou >= best_base_niou
                ),
                "CURE_retention_equal_1": isclose(
                    base_at_a_plus_cure.retention,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ),
                "CURE_pixel_Fa_le_1e-4": (
                    base_at_a_plus_cure.pixel_fa
                    <= PACRE_VC_MAXIMUM_PIXEL_FA
                ),
                "CURE_raw_background_Fa_le_1e-4": (
                    base_at_a_plus_cure.raw_background_fa
                    <= PACRE_VC_MAXIMUM_RAW_BACKGROUND_FA
                ),
                "CURE_false_positive_components_per_megapixel_le_100": (
                    base_at_a_plus_cure.fp_components_per_mp
                    <= PACRE_VC_MAXIMUM_FP_COMPONENTS_PER_MP
                ),
                "CURE_budget_violation_false": (
                    base_at_a_plus_cure.budget_violation is False
                ),
                "D_T_payload_accessed_false": True,
            }.items()
        )
    )
    return _PACREVCGateAssessment(
        valid_base_names=tuple(row[0] for row in valid),
        best_base_true_targets=best_base_true,
        best_base_recovered_anchor_misses=best_base_recovered,
        best_base_miou=best_base_miou,
        best_base_niou=best_base_niou,
        true_target_margin=true_target_margin,
        recovered_anchor_miss_margin=recovered_margin,
        checks=checks,
    )


@dataclass(frozen=True, slots=True)
class _PACREVCFormalDVEvaluationResultSeal:
    issuer: object
    binding: PACREVCFormalModelBinding
    samples: _PACREVCFixedDVSamples
    canonical_fingerprint: str = ""


_PACRE_VC_FORMAL_DV_RESULT_ISSUER = object()


@dataclass(frozen=True)
class PACREVCFormalDVEvaluationResult:
    """Sealed Base@A/Base@B/Base@A+CURE summaries and aggregates."""

    base_at_a: AggregateEvaluation
    base_at_b: AggregateEvaluation
    base_at_a_plus_cure: AggregateEvaluation
    base_at_b_selected_threshold: float
    base_threshold_grid: tuple[float, ...]
    base_candidate_ledger: CalibrationCandidateLedger
    budget: FalseAlarmBudget
    valid_base_names: tuple[str, ...]
    best_base_true_targets: int
    best_base_recovered_anchor_misses: int
    best_base_miou: float
    best_base_niou: float
    true_target_margin: int
    recovered_anchor_miss_margin: int
    checks: tuple[tuple[str, bool], ...]
    exact_zero_field_pixels: int
    negative_field_pixels: int
    completion_pixels: int
    base_samples_fingerprint: str
    cure_samples_fingerprint: str
    model_binding_fingerprint: str
    artifact_fingerprint: str
    model_state_fingerprint: str
    comparison_protocol_fingerprint: str
    manifest_fingerprint: str
    base_state_fingerprint: str
    d_v_base_index_fingerprint: str
    d_v_image_fingerprint: str
    d_v_gt_fingerprint: str
    batch_size: int = PACRE_VC_FORMAL_BATCH_SIZE
    seed: int = PACRE_VC_FORMAL_SEED
    _seal: _PACREVCFormalDVEvaluationResultSeal | None = (
        dataclass_field(default=None, repr=False, compare=False)
    )

    def __post_init__(self) -> None:
        seal = self._seal
        if (
            type(seal) is not _PACREVCFormalDVEvaluationResultSeal
            or seal.issuer is not _PACRE_VC_FORMAL_DV_RESULT_ISSUER
        ):
            raise PermissionError(
                "PACRE-VC formal D_V result must come from its evaluator"
            )
        if (
            self.seed != PACRE_VC_FORMAL_SEED
            or self.batch_size != PACRE_VC_FORMAL_BATCH_SIZE
        ):
            raise ValueError(
                "PACRE-VC formal D_V fixes seed 42 and batch_size 8"
            )
        if self.base_threshold_grid != PACRE_VC_FORMAL_BASE_THRESHOLD_GRID:
            raise ValueError("PACRE-VC Base@B grid changed")
        if (
            isinstance(self.base_at_b_selected_threshold, bool)
            or not isinstance(
                self.base_at_b_selected_threshold,
                (int, float),
            )
            or not isfinite(float(self.base_at_b_selected_threshold))
            or float(self.base_at_b_selected_threshold)
            not in self.base_threshold_grid
        ):
            raise ValueError("Base@B selected threshold is invalid")
        if self.budget != _FORMAL_BUDGET:
            raise ValueError("PACRE-VC formal D_V budget changed")
        ledger_payload = _base_candidate_ledger_payload(
            self.base_candidate_ledger,
            budget=self.budget,
        )
        selection = self.base_candidate_ledger.select(
            "Base@B",
            self.budget,
        )
        if (
            ledger_payload["candidate_count"]
            != len(PACRE_VC_FORMAL_BASE_THRESHOLD_GRID)
            or not selection.feasible
            or selection.threshold != self.base_at_b_selected_threshold
            or selection.metrics != self.base_at_b
            or self.base_candidate_ledger.anchor_metrics != self.base_at_a
        ):
            raise RuntimeError(
                "PACRE-VC Base@B selection differs from its 51-point ledger"
            )
        assessment = _assess_gate(
            base_at_a=self.base_at_a,
            base_at_b=self.base_at_b,
            base_at_a_plus_cure=self.base_at_a_plus_cure,
            budget=self.budget,
        )
        if (
            self.valid_base_names != assessment.valid_base_names
            or self.best_base_true_targets
            != assessment.best_base_true_targets
            or self.best_base_recovered_anchor_misses
            != assessment.best_base_recovered_anchor_misses
            or self.best_base_miou != assessment.best_base_miou
            or self.best_base_niou != assessment.best_base_niou
            or self.true_target_margin != assessment.true_target_margin
            or self.recovered_anchor_miss_margin
            != assessment.recovered_anchor_miss_margin
            or self.checks != assessment.checks
        ):
            raise RuntimeError(
                "PACRE-VC formal D_V gate assessment changed"
            )
        for name in (
            "base_samples_fingerprint",
            "cure_samples_fingerprint",
            "model_binding_fingerprint",
            "artifact_fingerprint",
            "model_state_fingerprint",
            "comparison_protocol_fingerprint",
            "manifest_fingerprint",
            "base_state_fingerprint",
            "d_v_base_index_fingerprint",
            "d_v_image_fingerprint",
            "d_v_gt_fingerprint",
        ):
            _digest(getattr(self, name), name=name)
        self.verify_unchanged()

    @property
    def gate_passed(self) -> bool:
        return bool(self.checks) and all(
            value for _, value in self.checks
        )

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(
            name for name, passed in self.checks if not passed
        )

    def verify_unchanged(self) -> None:
        seal = self._seal
        if (
            type(seal) is not _PACREVCFormalDVEvaluationResultSeal
            or seal.issuer is not _PACRE_VC_FORMAL_DV_RESULT_ISSUER
        ):
            raise PermissionError("PACRE-VC formal D_V result is unsealed")
        seal.binding.verify_unchanged()
        if (
            self.model_binding_fingerprint
            != seal.binding.binding_fingerprint
            or self.base_samples_fingerprint
            != seal.samples.base_samples_fingerprint
            or self.cure_samples_fingerprint
            != seal.samples.cure_samples_fingerprint
            or self.exact_zero_field_pixels
            != seal.samples.exact_zero_field_pixels
            or self.negative_field_pixels
            != seal.samples.negative_field_pixels
            or self.completion_pixels
            != seal.samples.completion_pixels
            or (
                seal.canonical_fingerprint
                and stable_fingerprint(self._canonical_payload())
                != seal.canonical_fingerprint
            )
        ):
            raise RuntimeError(
                "PACRE-VC formal D_V result changed after issuance"
            )

    def _canonical_payload(self) -> dict[str, object]:
        status = (
            "PACRE_V23_FORMAL_D_V_GATE_PASS"
            if self.gate_passed
            else "PACRE_V23_FORMAL_D_V_GATE_FAIL"
        )
        base_candidate_ledger = _base_candidate_ledger_payload(
            self.base_candidate_ledger,
            budget=self.budget,
        )
        return {
            "schema_version": PACRE_VC_FORMAL_DV_RESULT_SCHEMA,
            "method": PACRE_VC_FORMAL_METHOD,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "runtime_split": "D_V",
            "D_V_adaptive": True,
            "D_V_payload_accessed": True,
            "D_T_payload_accessed": False,
            "output_contract": {
                "rule": PACRE_VC_FIXED_OUTPUT_RULE,
                "field_threshold": 0.0,
                "zero_tie_policy": PACRE_VC_ZERO_TIE_POLICY,
                "hard_union": True,
                "sigmoid_applied": False,
                "PACRE_threshold_search_performed": False,
                "exact_zero_field_pixels": (
                    self.exact_zero_field_pixels
                ),
                "negative_field_pixels": (
                    self.negative_field_pixels
                ),
                "completion_pixels": self.completion_pixels,
            },
            "Base@B_selection": {
                "policy": PACRE_VC_BASE_AT_B_SELECTION_POLICY,
                "base_threshold_search_performed": True,
                "candidate_threshold_grid": list(
                    self.base_threshold_grid
                ),
                "candidate_count": len(self.base_threshold_grid),
                "candidate_ledger": base_candidate_ledger,
                "selected_threshold": (
                    self.base_at_b_selected_threshold
                ),
                "stage_a_config_sha256": (
                    PACRE_VC_FORMAL_STAGE_A_CONFIG_SHA256
                ),
                "budget": {
                    "pixel_fa_budget": self.budget.pixel_fa_budget,
                    "component_fa_per_mp_budget": (
                        self.budget.component_fa_per_mp_budget
                    ),
                    "raw_background_fa_budget": (
                        self.budget.raw_background_fa_budget
                    ),
                    "minimum_retention": (
                        self.budget.minimum_retention
                    ),
                },
            },
            "operating_points": {
                "Base@A": {
                    "aggregate_evaluation": _aggregate_payload(
                        self.base_at_a
                    ),
                    "summary": _summary_payload(self.base_at_a),
                },
                "Base@B": {
                    "aggregate_evaluation": _aggregate_payload(
                        self.base_at_b
                    ),
                    "summary": _summary_payload(self.base_at_b),
                },
                "Base@A+CURE": {
                    "aggregate_evaluation": _aggregate_payload(
                        self.base_at_a_plus_cure
                    ),
                    "summary": _summary_payload(
                        self.base_at_a_plus_cure
                    ),
                },
            },
            "development_gate": {
                "comparison": "best_valid_Base",
                "valid_base_names": list(self.valid_base_names),
                "requirements": {
                    "true_targets": (
                        "strictly_greater_than_best_valid_Base"
                    ),
                    "recovered_anchor_misses": (
                        "strictly_greater_than_best_valid_Base"
                    ),
                    "mIoU": "not_below_best_valid_Base",
                    "nIoU": "not_below_best_valid_Base",
                    "retention": 1.0,
                    "maximum_pixel_Fa": (
                        PACRE_VC_MAXIMUM_PIXEL_FA
                    ),
                    "maximum_raw_background_Fa": (
                        PACRE_VC_MAXIMUM_RAW_BACKGROUND_FA
                    ),
                    "maximum_false_positive_components_per_megapixel": (
                        PACRE_VC_MAXIMUM_FP_COMPONENTS_PER_MP
                    ),
                    "budget_violation": False,
                },
                "best_valid_Base": {
                    "true_targets": self.best_base_true_targets,
                    "recovered_anchor_misses": (
                        self.best_base_recovered_anchor_misses
                    ),
                    "mIoU": self.best_base_miou,
                    "nIoU": self.best_base_niou,
                },
                "CURE_margins": {
                    "true_targets": self.true_target_margin,
                    "recovered_anchor_misses": (
                        self.recovered_anchor_miss_margin
                    ),
                },
                "checks": dict(self.checks),
                "failed_checks": list(self.failed_checks),
                "gate_passed": self.gate_passed,
                "status": status,
            },
            "bindings": {
                "base_samples_fingerprint": (
                    self.base_samples_fingerprint
                ),
                "cure_samples_fingerprint": (
                    self.cure_samples_fingerprint
                ),
                "model_binding_fingerprint": (
                    self.model_binding_fingerprint
                ),
                "artifact_fingerprint": self.artifact_fingerprint,
                "model_state_fingerprint": (
                    self.model_state_fingerprint
                ),
                "comparison_protocol_fingerprint": (
                    self.comparison_protocol_fingerprint
                ),
                "manifest_fingerprint": self.manifest_fingerprint,
                "base_state_fingerprint": (
                    self.base_state_fingerprint
                ),
                "D_V_base_index_fingerprint": (
                    self.d_v_base_index_fingerprint
                ),
                "D_V_image_fingerprint": (
                    self.d_v_image_fingerprint
                ),
                "D_V_GT_fingerprint": self.d_v_gt_fingerprint,
            },
            "eligible_for_D_T_confirmation": self.gate_passed,
            "authorizes_D_T": False,
            "final_model_success_established": False,
        }

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return self._canonical_payload()

    @property
    def result_fingerprint(self) -> str:
        self.verify_unchanged()
        return stable_fingerprint(self._canonical_payload())


def evaluate_pacre_vc_formal_d_v(
    model: CURELitePACREVerifierCorrectedLevelSet,
    bundle: LoadedDVCacheBundle,
    comparison_protocol: FrozenComparisonProtocol,
    model_binding: PACREVCFormalModelBinding,
    *,
    batch_size: int = 8,
) -> PACREVCFormalDVEvaluationResult:
    """Evaluate the exact bound v23 model on one strict D_V cache bundle."""

    if type(model) is not CURELitePACREVerifierCorrectedLevelSet:
        raise TypeError(
            "model must be exact CURELitePACREVerifierCorrectedLevelSet"
        )
    if type(bundle) is not LoadedDVCacheBundle:
        raise TypeError("bundle must be exact LoadedDVCacheBundle")
    if type(comparison_protocol) is not FrozenComparisonProtocol:
        raise TypeError(
            "comparison_protocol must be exact FrozenComparisonProtocol"
        )
    if type(model_binding) is not PACREVCFormalModelBinding:
        raise TypeError(
            "model_binding must be exact PACREVCFormalModelBinding"
        )
    model_binding.verify_inputs(model, bundle, comparison_protocol)
    grid = _verified_stage_a_base_grid()
    samples = _build_fixed_d_v_samples(
        model,
        bundle,
        comparison_protocol,
        model_binding,
        batch_size=batch_size,
    )
    points = _evaluate_fixed_operating_points(
        samples.base_samples,
        samples.cure_samples,
        occupancy_config=comparison_protocol.occupancy_config,
        match_config=comparison_protocol.match_config,
        base_threshold_grid=grid,
        budget=comparison_protocol.budget,
    )
    if points.base_candidate_ledger.methods != ("Base@B",):
        raise RuntimeError(
            "PACRE-VC formal evaluation searched a non-Base method"
        )
    assessment = _assess_gate(
        base_at_a=points.base_at_a,
        base_at_b=points.base_at_b,
        base_at_a_plus_cure=points.base_at_a_plus_cure,
        budget=comparison_protocol.budget,
    )
    seal = _PACREVCFormalDVEvaluationResultSeal(
        issuer=_PACRE_VC_FORMAL_DV_RESULT_ISSUER,
        binding=model_binding,
        samples=samples,
    )
    result = PACREVCFormalDVEvaluationResult(
        base_at_a=points.base_at_a,
        base_at_b=points.base_at_b,
        base_at_a_plus_cure=points.base_at_a_plus_cure,
        base_at_b_selected_threshold=(
            points.base_at_b_selected_threshold
        ),
        base_threshold_grid=grid,
        base_candidate_ledger=points.base_candidate_ledger,
        budget=comparison_protocol.budget,
        valid_base_names=assessment.valid_base_names,
        best_base_true_targets=(
            assessment.best_base_true_targets
        ),
        best_base_recovered_anchor_misses=(
            assessment.best_base_recovered_anchor_misses
        ),
        best_base_miou=assessment.best_base_miou,
        best_base_niou=assessment.best_base_niou,
        true_target_margin=assessment.true_target_margin,
        recovered_anchor_miss_margin=(
            assessment.recovered_anchor_miss_margin
        ),
        checks=assessment.checks,
        exact_zero_field_pixels=samples.exact_zero_field_pixels,
        negative_field_pixels=samples.negative_field_pixels,
        completion_pixels=samples.completion_pixels,
        base_samples_fingerprint=(
            samples.base_samples_fingerprint
        ),
        cure_samples_fingerprint=(
            samples.cure_samples_fingerprint
        ),
        model_binding_fingerprint=model_binding.binding_fingerprint,
        artifact_fingerprint=model_binding.artifact_fingerprint,
        model_state_fingerprint=(
            model_binding.model_state_fingerprint
        ),
        comparison_protocol_fingerprint=(
            model_binding.comparison_protocol_fingerprint
        ),
        manifest_fingerprint=model_binding.manifest_fingerprint,
        base_state_fingerprint=model_binding.base_state_fingerprint,
        d_v_base_index_fingerprint=(
            model_binding.d_v_base_index_fingerprint
        ),
        d_v_image_fingerprint=(
            model_binding.d_v_image_fingerprint
        ),
        d_v_gt_fingerprint=model_binding.d_v_gt_fingerprint,
        batch_size=batch_size,
        _seal=seal,
    )
    object.__setattr__(
        seal,
        "canonical_fingerprint",
        stable_fingerprint(result._canonical_payload()),
    )
    result.verify_unchanged()
    return result


__all__ = [
    "PACRE_VC_BASE_AT_B_SELECTION_POLICY",
    "PACRE_VC_FIXED_OUTPUT_RULE",
    "PACRE_VC_FORMAL_BASE_THRESHOLD",
    "PACRE_VC_FORMAL_BASE_THRESHOLD_GRID",
    "PACRE_VC_FORMAL_BATCH_SIZE",
    "PACRE_VC_FORMAL_DV_RESULT_SCHEMA",
    "PACRE_VC_FORMAL_EPOCHS",
    "PACRE_VC_FORMAL_METHOD",
    "PACRE_VC_FORMAL_MODEL_BINDING_SCHEMA",
    "PACRE_VC_FORMAL_SEED",
    "PACRE_VC_FORMAL_STAGE_A_CONFIG_SHA256",
    "PACRE_VC_FORMAL_STEPS_PER_EPOCH",
    "PACRE_VC_FORMAL_UPDATES",
    "PACRE_VC_MAXIMUM_FP_COMPONENTS_PER_MP",
    "PACRE_VC_MAXIMUM_PIXEL_FA",
    "PACRE_VC_MAXIMUM_RAW_BACKGROUND_FA",
    "PACRE_VC_ZERO_TIE_POLICY",
    "PACREVCFormalDVEvaluationResult",
    "PACREVCFormalModelBinding",
    "bind_pacre_vc_formal_model",
    "evaluate_pacre_vc_formal_d_v",
    "fixed_pacre_vc_completion",
    "pacre_vc_formal_model_config_fingerprint",
]
