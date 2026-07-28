"""Fixed-output D_V evaluation for the v21 PAET-BFA CURE-Lite field.

PAET-BFA is a binary level-set method, not a probabilistic residual head.  Its
formal output is therefore decoded exactly once:

``O = (p_b >= 0.72)``
``completion = (phi < 0) & ~O``
``final = O | completion``

In particular, ``phi == 0`` is *not* completion.  This module never applies a
sigmoid to ``phi`` and exposes no PAET threshold grid.  The only selected
operating point is the historical ``Base@B`` control, evaluated over the
existing 51-point base-probability grid with the existing false-addition
budget.  All object, false-addition, and IoU quantities delegate to the shared
calibration-ledger/metric implementation.

The strict cache entry point accepts only a verified
:class:`LoadedDVCacheBundle`, an exact PAET model, an immutable formal-artifact
identity binding, and the frozen common comparison protocol.  It deliberately
has no D_T entry point.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from ..calibration import CalibrationSample, FalseAlarmBudget
from ..calibration_ledger import (
    CalibrationCandidateLedger,
    prepare_calibration_context,
    evaluate_candidate_ledger,
)
from ..config import MatchConfig, OccupancyConfig
from ..coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
)
from ..frozen_base import module_state_fingerprint
from ..instances import instances_from_binary_mask
from ..metrics import (
    AggregateEvaluation,
    aggregate_evaluations,
    evaluate_binary_prediction_from_instances,
)
from .cache_pipeline import LoadedDVCacheBundle
from .coverage_state_paet_formal_artifacts import (
    LoadedCoverageStatePAETFormalArtifact,
)
from .coverage_state_paet_formal_attempt import (
    LoadedCoverageStatePAETFormalAttempt,
)
from .evaluation_pipeline import calibration_samples_fingerprint
from .paired_formal_evaluation import (
    FORMAL_DV_ANCHOR_COVERED,
    FORMAL_DV_ANCHOR_MISSES,
    FORMAL_DV_IMAGES,
    FORMAL_DV_TOTAL_TARGETS,
    FrozenComparisonProtocol,
)


PAET_FORMAL_DV_SAMPLE_SCHEMA = (
    "cure-lite-paet-bfa-v21-fixed-d-v-samples-v1"
)
PAET_FORMAL_DV_RESULT_SCHEMA = (
    "cure-lite-paet-bfa-v21-fixed-d-v-result-v1"
)
PAET_FORMAL_ARTIFACT_BINDING_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal-artifact-binding-v1"
)
PAET_FORMAL_METHOD = "PAET-BFA-v21"
PAET_FORMAL_SEED = 42
PAET_FORMAL_EPOCHS = 800
PAET_FORMAL_STEPS_PER_EPOCH = 40
PAET_FORMAL_UPDATES = PAET_FORMAL_EPOCHS * PAET_FORMAL_STEPS_PER_EPOCH
PAET_FORMAL_BASE_THRESHOLD = 0.72
PAET_FORMAL_BASE_THRESHOLD_GRID = tuple(
    index / 50 for index in range(51)
)
# SHA256 of
# protocols/IRSTD-1K/stage_a_seed42_fx_v3/stage_a_config.json, already bound
# by paired_formal_preflight_v1.  Its ``base_thresholds`` field is the source
# of the 51-point Base@B grid; it is distinct from a historical run receipt's
# own copied ``config.json`` SHA.
PAET_FORMAL_STAGE_A_CONFIG_SHA256 = (
    "6eecdc10f87a043cafb945db40d0b767b5f0a2ccb64963c1043160f165ce9d6c"
)
PAET_FIXED_OUTPUT_RULE = (
    "occupancy=(base_probability>=0.72);"
    "completion=(field<0)&~occupancy;"
    "final=occupancy|completion"
)
PAET_ZERO_TIE_POLICY = "field_equal_zero_is_not_completion"
PAET_BASE_AT_B_SELECTION_POLICY = (
    "base_probability_only_51_point_grid_existing_budget"
)
_HEX = frozenset("0123456789abcdef")
_AGGREGATE_FIELDS = tuple(field.name for field in fields(AggregateEvaluation))


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _aggregate_payload(metrics: AggregateEvaluation) -> dict[str, object]:
    if not isinstance(metrics, AggregateEvaluation):
        raise TypeError("metrics must be AggregateEvaluation")
    return {
        name: getattr(metrics, name)
        for name in _AGGREGATE_FIELDS
    }


def _base_grid(values: Sequence[float]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("Base@B grid values must be real numbers")
        resolved = float(value)
        if not isfinite(resolved):
            raise ValueError("Base@B grid values must be finite")
        result.append(resolved)
    grid = tuple(result)
    if grid != PAET_FORMAL_BASE_THRESHOLD_GRID:
        raise ValueError(
            "Base@B must use the existing ordered 51-point base grid"
        )
    return grid


def _summary_payload(metrics: AggregateEvaluation) -> dict[str, object]:
    true_targets = (
        metrics.retained_anchor_covered
        + metrics.recovered_anchor_misses
    )
    return {
        "true_targets": true_targets,
        "Pd": metrics.pd,
        "mIoU": metrics.miou,
        "nIoU": metrics.niou,
        "pixel_Fa": metrics.pixel_fa,
        "raw_background_Fa": metrics.raw_background_fa,
        "false_positive_components_per_megapixel": (
            metrics.fp_components_per_mp
        ),
        "recovered_anchor_misses": metrics.recovered_anchor_misses,
        "retained_anchor_covered": metrics.retained_anchor_covered,
        "total_anchor_misses": metrics.total_anchor_misses,
        "total_anchor_covered": metrics.total_anchor_covered,
        "retention": metrics.retention,
        "budget_violation": metrics.budget_violation,
    }


def fixed_paet_completion(field: Tensor, occupancy: Tensor) -> Tensor:
    """Decode the sole PAET operating point without a learned/search threshold.

    ``torch.lt`` is intentional.  Replacing it with
    ``sigmoid(field) >= 0.5`` would reverse the negative-level-set convention
    and would also classify an exact zero tie as positive.
    """

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
        raise TypeError("occupancy must be bool with the exact field shape")
    if field.device != occupancy.device:
        raise ValueError("field and occupancy must share one device")
    if not bool(torch.isfinite(field).all()):
        raise ValueError("field must contain only finite values")
    return ((field < 0) & ~occupancy).contiguous()


def paet_formal_model_config_fingerprint(
    model: CURELitePhaseAlignedEvidenceTransportLevelSet,
) -> str:
    """Return the class/config/count identity used by the formal binding."""

    if type(model) is not CURELitePhaseAlignedEvidenceTransportLevelSet:
        raise TypeError("model must be the exact PAET-BFA class")
    return stable_fingerprint(
        {
            "config": asdict(model.config),
            "model_class": (
                "CURELitePhaseAlignedEvidenceTransportLevelSet"
            ),
            "expected_parameter_count": (
                model.config.expected_parameter_count
            ),
        }
    )


def _stage_a_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "protocols/IRSTD-1K/stage_a_seed42_fx_v3/stage_a_config.json"
    )


def _verified_stage_a_base_grid() -> tuple[float, ...]:
    """Re-hash and parse the frozen Stage-A source of the Base@B grid."""

    path = _stage_a_config_path()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("frozen stage_a_config.json is unavailable")
    source = path.read_bytes()
    if sha256(source).hexdigest() != PAET_FORMAL_STAGE_A_CONFIG_SHA256:
        raise RuntimeError("frozen stage_a_config.json bytes changed")
    try:
        payload = json.loads(source.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("frozen stage_a_config.json is invalid") from error
    if not isinstance(payload, dict):
        raise RuntimeError("frozen stage_a_config.json must be an object")
    return _base_grid(payload.get("base_thresholds", ()))


@dataclass(frozen=True, slots=True)
class _PAETFormalArtifactBindingSeal:
    issuer: object
    attempt: LoadedCoverageStatePAETFormalAttempt
    comparison_protocol: FrozenComparisonProtocol
    bundle: LoadedDVCacheBundle


_PAET_FORMAL_ARTIFACT_BINDING_ISSUER = object()


@dataclass(frozen=True)
class PAETFormalArtifactBinding:
    """Immutable bridge from a strictly loaded formal artifact to D_V.

    The artifact loader remains responsible for validating persisted bytes.
    This binding prevents the evaluator from accepting a bounded-400 model,
    a continuation checkpoint, or a model whose exact state/config differs
    from that loaded artifact.
    """

    seed: int
    epochs: int
    steps_per_epoch: int
    completed_updates: int
    trained_from_scratch: bool
    resumed: bool
    runtime_splits: tuple[str, ...]
    artifact_fingerprint: str
    artifact_receipt_sha256: str
    model_state_fingerprint: str
    model_config_fingerprint: str
    formal_training_protocol_fingerprint: str
    formal_schedule_fingerprint: str
    formal_training_result_fingerprint: str
    source_closure_fingerprint: str
    source_closure_manifest_sha256: str
    source_closure_archive_sha256: str
    source_closure_file_count: int
    structural_source_receipt_fingerprint: str
    formal_attempt_complete_fingerprint: str
    manifest_fingerprint: str
    manifest_file_sha256: str
    preprocessing_fingerprint: str
    base_fingerprint: str
    base_state_fingerprint: str
    stage_a_config_sha256: str
    comparison_protocol_fingerprint: str
    d_v_accessed_during_training: bool = False
    d_t_accessed_during_training: bool = False
    schema_version: str = PAET_FORMAL_ARTIFACT_BINDING_SCHEMA
    _seal: _PAETFormalArtifactBindingSeal | None = None

    def __post_init__(self) -> None:
        seal = self._seal
        if (
            type(seal) is not _PAETFormalArtifactBindingSeal
            or seal.issuer is not _PAET_FORMAL_ARTIFACT_BINDING_ISSUER
        ):
            raise PermissionError(
                "PAET formal artifact binding must come from its strict factory"
            )
        if self.schema_version != PAET_FORMAL_ARTIFACT_BINDING_SCHEMA:
            raise ValueError("unsupported PAET formal-artifact binding schema")
        if (
            self.seed != PAET_FORMAL_SEED
            or self.epochs != PAET_FORMAL_EPOCHS
            or self.steps_per_epoch != PAET_FORMAL_STEPS_PER_EPOCH
            or self.completed_updates != PAET_FORMAL_UPDATES
            or self.trained_from_scratch is not True
            or self.resumed is not False
            or self.runtime_splits != ("D_R",)
            or self.d_v_accessed_during_training is not False
            or self.d_t_accessed_during_training is not False
        ):
            raise ValueError(
                "PAET D_V evaluation requires the completed, from-scratch "
                "seed-42 800x40 D_R formal artifact"
            )
        for name in (
            "artifact_fingerprint",
            "artifact_receipt_sha256",
            "model_state_fingerprint",
            "model_config_fingerprint",
            "formal_training_protocol_fingerprint",
            "formal_schedule_fingerprint",
            "formal_training_result_fingerprint",
            "source_closure_fingerprint",
            "source_closure_manifest_sha256",
            "source_closure_archive_sha256",
            "structural_source_receipt_fingerprint",
            "formal_attempt_complete_fingerprint",
            "manifest_fingerprint",
            "manifest_file_sha256",
            "preprocessing_fingerprint",
            "base_fingerprint",
            "base_state_fingerprint",
            "stage_a_config_sha256",
            "comparison_protocol_fingerprint",
        ):
            _digest(getattr(self, name), name=name)
        if self.stage_a_config_sha256 != PAET_FORMAL_STAGE_A_CONFIG_SHA256:
            raise ValueError(
                "PAET Base@B grid must bind the frozen seed-42 "
                "stage_a_config.json bytes"
            )
        if (
            isinstance(self.source_closure_file_count, bool)
            or not isinstance(self.source_closure_file_count, int)
            or self.source_closure_file_count < 1
        ):
            raise ValueError(
                "PAET formal source-closure file count must be positive"
            )
        self.verify_unchanged()

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "formal_budget": {
                "epochs": self.epochs,
                "steps_per_epoch": self.steps_per_epoch,
                "completed_updates": self.completed_updates,
            },
            "trained_from_scratch": self.trained_from_scratch,
            "resumed": self.resumed,
            "runtime_splits": list(self.runtime_splits),
            "artifact_fingerprint": self.artifact_fingerprint,
            "artifact_receipt_sha256": self.artifact_receipt_sha256,
            "model_state_fingerprint": self.model_state_fingerprint,
            "model_config_fingerprint": self.model_config_fingerprint,
            "formal_training_protocol_fingerprint": (
                self.formal_training_protocol_fingerprint
            ),
            "formal_schedule_fingerprint": self.formal_schedule_fingerprint,
            "formal_training_result_fingerprint": (
                self.formal_training_result_fingerprint
            ),
            "source_closure_fingerprint": self.source_closure_fingerprint,
            "source_closure_manifest_sha256": (
                self.source_closure_manifest_sha256
            ),
            "source_closure_archive_sha256": (
                self.source_closure_archive_sha256
            ),
            "source_closure_file_count": self.source_closure_file_count,
            "structural_source_receipt_fingerprint": (
                self.structural_source_receipt_fingerprint
            ),
            "formal_attempt_complete_fingerprint": (
                self.formal_attempt_complete_fingerprint
            ),
            "manifest_fingerprint": self.manifest_fingerprint,
            "manifest_file_sha256": self.manifest_file_sha256,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "base_fingerprint": self.base_fingerprint,
            "base_state_fingerprint": self.base_state_fingerprint,
            "stage_a_config_sha256": self.stage_a_config_sha256,
            "comparison_protocol_fingerprint": (
                self.comparison_protocol_fingerprint
            ),
            "D_V_accessed_during_training": (
                self.d_v_accessed_during_training
            ),
            "D_T_accessed_during_training": (
                self.d_t_accessed_during_training
            ),
        }

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return self._canonical_payload()

    @property
    def binding_fingerprint(self) -> str:
        self.verify_unchanged()
        return stable_fingerprint(self._canonical_payload())

    def verify_model(
        self,
        model: CURELitePhaseAlignedEvidenceTransportLevelSet,
    ) -> None:
        if type(model) is not CURELitePhaseAlignedEvidenceTransportLevelSet:
            raise TypeError(
                "formal D_V evaluation accepts only the exact PAET-BFA model"
            )
        if module_state_fingerprint(model) != self.model_state_fingerprint:
            raise RuntimeError(
                "PAET model state differs from the formal artifact binding"
            )
        if (
            paet_formal_model_config_fingerprint(model)
            != self.model_config_fingerprint
        ):
            raise RuntimeError(
                "PAET model config differs from the formal artifact binding"
            )

    def _sealed_inputs(
        self,
    ) -> _PAETFormalArtifactBindingSeal:
        seal = self._seal
        if (
            type(seal) is not _PAETFormalArtifactBindingSeal
            or seal.issuer is not _PAET_FORMAL_ARTIFACT_BINDING_ISSUER
        ):
            raise PermissionError("PAET formal artifact binding is unsealed")
        return seal

    def verify_unchanged(self) -> None:
        """Revalidate every strict source before D_V work or result use."""

        seal = self._sealed_inputs()
        attempt = seal.attempt
        artifact = attempt.artifact
        protocol = seal.comparison_protocol
        bundle = seal.bundle
        if (
            type(attempt) is not LoadedCoverageStatePAETFormalAttempt
            or type(artifact) is not LoadedCoverageStatePAETFormalArtifact
            or type(protocol) is not FrozenComparisonProtocol
            or type(bundle) is not LoadedDVCacheBundle
        ):
            raise TypeError("PAET formal binding sources must be strict loaded objects")
        attempt.verify_unchanged()
        artifact.verify_unchanged()
        bundle.verify_unchanged()
        protocol.verify_bundle(bundle)
        stage_grid = _verified_stage_a_base_grid()
        if (
            stage_grid != PAET_FORMAL_BASE_THRESHOLD_GRID
            or not attempt.post_formal_structural_retention_passed
            or artifact.module_state_fingerprint
            != attempt.artifact.module_state_fingerprint
            or artifact.formal_result_fingerprint
            != attempt.formal_training_result_fingerprint
            or artifact.authorization_fingerprint
            != attempt.authorization_fingerprint
            or self.artifact_fingerprint != artifact.artifact_fingerprint
            or self.artifact_receipt_sha256 != artifact.receipt_sha256
            or self.model_state_fingerprint != artifact.module_state_fingerprint
            or self.model_config_fingerprint
            != paet_formal_model_config_fingerprint(artifact.model)
            or self.formal_training_protocol_fingerprint
            != artifact.training_fingerprint
            or self.formal_schedule_fingerprint
            != artifact.training_payload["schedule_fingerprint"]
            or self.formal_training_result_fingerprint
            != artifact.formal_result_fingerprint
            or self.source_closure_fingerprint
            != attempt.source_closure_content_fingerprint
            or self.source_closure_manifest_sha256
            != attempt.source_closure_manifest_sha256
            or self.source_closure_archive_sha256
            != attempt.source_closure_archive_sha256
            or self.source_closure_file_count
            != attempt.source_closure_file_count
            or self.structural_source_receipt_fingerprint
            != attempt.source_receipt_fingerprint
            or self.formal_attempt_complete_fingerprint
            != attempt.complete_fingerprint
            or self.manifest_fingerprint != protocol.manifest_fingerprint
            or self.manifest_file_sha256 != protocol.manifest_file_sha256
            or self.preprocessing_fingerprint != protocol.preprocessing_fingerprint
            or self.base_fingerprint != protocol.base_fingerprint
            or self.base_state_fingerprint != bundle.base_state_fingerprint
            or self.stage_a_config_sha256
            != PAET_FORMAL_STAGE_A_CONFIG_SHA256
            or self.comparison_protocol_fingerprint
            != protocol.comparison_protocol_fingerprint
        ):
            raise RuntimeError("PAET formal artifact binding sources differ")

    def verify_cache_and_protocol(
        self,
        bundle: LoadedDVCacheBundle,
        comparison_protocol: FrozenComparisonProtocol,
    ) -> None:
        self.verify_unchanged()
        seal = self._sealed_inputs()
        if bundle is not seal.bundle or comparison_protocol is not seal.comparison_protocol:
            raise RuntimeError("PAET formal binding rejects substituted D_V inputs")
        comparison_protocol.verify_bundle(bundle)
        expected = {
            "manifest_fingerprint": bundle.split_manifest_fingerprint,
            "manifest_file_sha256": bundle.split_manifest_file_sha256,
            "preprocessing_fingerprint": (
                bundle.preprocessing_fingerprint
            ),
            "base_fingerprint": bundle.base_fingerprint,
            "base_state_fingerprint": bundle.base_state_fingerprint,
            "comparison_protocol_fingerprint": (
                comparison_protocol.comparison_protocol_fingerprint
            ),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise RuntimeError(
                    f"PAET formal artifact/cache/protocol mismatch at {name}"
                )


def bind_paet_formal_artifact(
    attempt: LoadedCoverageStatePAETFormalAttempt,
    comparison_protocol: FrozenComparisonProtocol,
    bundle: LoadedDVCacheBundle,
) -> PAETFormalArtifactBinding:
    """Issue the only D_V binding accepted by PAET evaluation.

    This factory does not evaluate D_V.  It only joins already loaded and
    sealed sources after independently re-checking the on-disk Stage-A config.
    """

    if type(attempt) is not LoadedCoverageStatePAETFormalAttempt:
        raise TypeError("attempt must be strictly LoadedCoverageStatePAETFormalAttempt")
    if type(comparison_protocol) is not FrozenComparisonProtocol:
        raise TypeError("comparison_protocol must be strictly FrozenComparisonProtocol")
    if type(bundle) is not LoadedDVCacheBundle:
        raise TypeError("bundle must be strictly LoadedDVCacheBundle")
    attempt.verify_unchanged()
    artifact = attempt.artifact
    artifact.verify_unchanged()
    comparison_protocol.verify_bundle(bundle)
    if _verified_stage_a_base_grid() != PAET_FORMAL_BASE_THRESHOLD_GRID:
        raise RuntimeError("frozen Stage-A Base@B grid changed")
    if not attempt.post_formal_structural_retention_passed:
        raise PermissionError("post-Formal800 structural retention did not pass")
    if (
        artifact.module_state_fingerprint != attempt.artifact.module_state_fingerprint
        or artifact.formal_result_fingerprint
        != attempt.formal_training_result_fingerprint
        or artifact.authorization_fingerprint
        != attempt.authorization_fingerprint
    ):
        raise RuntimeError("loaded artifact and structural receipt do not bind one model")
    return PAETFormalArtifactBinding(
        seed=PAET_FORMAL_SEED,
        epochs=PAET_FORMAL_EPOCHS,
        steps_per_epoch=PAET_FORMAL_STEPS_PER_EPOCH,
        completed_updates=PAET_FORMAL_UPDATES,
        trained_from_scratch=True,
        resumed=False,
        runtime_splits=("D_R",),
        artifact_fingerprint=artifact.artifact_fingerprint,
        artifact_receipt_sha256=artifact.receipt_sha256,
        model_state_fingerprint=artifact.module_state_fingerprint,
        model_config_fingerprint=paet_formal_model_config_fingerprint(artifact.model),
        formal_training_protocol_fingerprint=artifact.training_fingerprint,
        formal_schedule_fingerprint=artifact.training_payload["schedule_fingerprint"],
        formal_training_result_fingerprint=artifact.formal_result_fingerprint,
        source_closure_fingerprint=(
            attempt.source_closure_content_fingerprint
        ),
        source_closure_manifest_sha256=(
            attempt.source_closure_manifest_sha256
        ),
        source_closure_archive_sha256=(
            attempt.source_closure_archive_sha256
        ),
        source_closure_file_count=attempt.source_closure_file_count,
        structural_source_receipt_fingerprint=(
            attempt.source_receipt_fingerprint
        ),
        formal_attempt_complete_fingerprint=(
            attempt.complete_fingerprint
        ),
        manifest_fingerprint=comparison_protocol.manifest_fingerprint,
        manifest_file_sha256=comparison_protocol.manifest_file_sha256,
        preprocessing_fingerprint=comparison_protocol.preprocessing_fingerprint,
        base_fingerprint=comparison_protocol.base_fingerprint,
        base_state_fingerprint=bundle.base_state_fingerprint,
        stage_a_config_sha256=PAET_FORMAL_STAGE_A_CONFIG_SHA256,
        comparison_protocol_fingerprint=(
            comparison_protocol.comparison_protocol_fingerprint
        ),
        _seal=_PAETFormalArtifactBindingSeal(
            issuer=_PAET_FORMAL_ARTIFACT_BINDING_ISSUER,
            attempt=attempt,
            comparison_protocol=comparison_protocol,
            bundle=bundle,
        ),
    )


@dataclass(frozen=True, slots=True)
class _PAETFixedDVSamplesSeal:
    issuer: object
    binding: PAETFormalArtifactBinding


_PAET_FIXED_DV_SAMPLES_ISSUER = object()


@dataclass(frozen=True)
class PAETFixedDVSamples:
    """Exact fixed-completion samples emitted from a verified D_V cache."""

    base_samples: tuple[CalibrationSample, ...]
    cure_samples: tuple[CalibrationSample, ...]
    ordered_sample_ids: tuple[str, ...]
    base_samples_fingerprint: str
    cure_samples_fingerprint: str
    exact_zero_field_pixels: int
    negative_field_pixels: int
    completion_pixels: int
    artifact_binding_fingerprint: str
    comparison_protocol_fingerprint: str
    d_v_base_index_fingerprint: str
    d_v_image_fingerprint: str
    d_v_gt_fingerprint: str
    _seal: _PAETFixedDVSamplesSeal | None = None

    def __post_init__(self) -> None:
        seal = self._seal
        if (
            type(seal) is not _PAETFixedDVSamplesSeal
            or seal.issuer is not _PAET_FIXED_DV_SAMPLES_ISSUER
        ):
            raise PermissionError(
                "PAET fixed D_V samples must come from the sealed evaluator"
            )
        if (
            not isinstance(self.base_samples, tuple)
            or not self.base_samples
            or not isinstance(self.cure_samples, tuple)
            or len(self.base_samples) != len(self.cure_samples)
            or len(self.base_samples) != len(self.ordered_sample_ids)
        ):
            raise ValueError("PAET D_V sample tuples must be non-empty/aligned")
        if tuple(
            sample.sample_id for sample in self.base_samples
        ) != self.ordered_sample_ids or tuple(
            sample.sample_id for sample in self.cure_samples
        ) != self.ordered_sample_ids:
            raise ValueError("PAET D_V sample IDs/order changed")
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
            raise ValueError("PAET field/completion counts must be nonnegative")
        for name in (
            "base_samples_fingerprint",
            "cure_samples_fingerprint",
            "artifact_binding_fingerprint",
            "comparison_protocol_fingerprint",
            "d_v_base_index_fingerprint",
            "d_v_image_fingerprint",
            "d_v_gt_fingerprint",
        ):
            _digest(getattr(self, name), name=name)
        if (
            calibration_samples_fingerprint(self.base_samples)
            != self.base_samples_fingerprint
            or calibration_samples_fingerprint(self.cure_samples)
            != self.cure_samples_fingerprint
        ):
            raise RuntimeError("PAET D_V sample tensors changed")
        for base_sample, cure_sample in zip(
            self.base_samples,
            self.cure_samples,
            strict=True,
        ):
            base, residual, gt = base_sample.normalized()
            cure_base, completion, cure_gt = cure_sample.normalized()
            if (
                not torch.equal(base, cure_base)
                or not torch.equal(gt, cure_gt)
                or bool(torch.any(residual != 0))
                or bool(
                    torch.any(
                        (completion != 0) & (completion != 1)
                    )
                )
            ):
                raise RuntimeError(
                    "PAET fixed D_V sample contract changed"
                )
        self.verify_unchanged()

    def verify_unchanged(self) -> None:
        seal = self._seal
        if (
            type(seal) is not _PAETFixedDVSamplesSeal
            or seal.issuer is not _PAET_FIXED_DV_SAMPLES_ISSUER
        ):
            raise PermissionError("PAET fixed D_V samples are unsealed")
        binding = seal.binding
        binding.verify_unchanged()
        source = binding._sealed_inputs()
        if (
            self.artifact_binding_fingerprint != binding.binding_fingerprint
            or self.comparison_protocol_fingerprint
            != source.comparison_protocol.comparison_protocol_fingerprint
            or self.d_v_base_index_fingerprint != source.bundle.base_index_fingerprint
            or self.d_v_image_fingerprint != source.bundle.d_v_image_fingerprint
            or self.d_v_gt_fingerprint != source.bundle.d_v_gt_fingerprint
            or calibration_samples_fingerprint(self.base_samples)
            != self.base_samples_fingerprint
            or calibration_samples_fingerprint(self.cure_samples)
            != self.cure_samples_fingerprint
        ):
            raise RuntimeError("PAET fixed D_V samples changed after issuance")

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": PAET_FORMAL_DV_SAMPLE_SCHEMA,
            "runtime_split": "D_V",
            "D_T_accessed": False,
            "sample_count": len(self.ordered_sample_ids),
            "ordered_sample_ids": list(self.ordered_sample_ids),
            "base_samples_fingerprint": self.base_samples_fingerprint,
            "cure_samples_fingerprint": self.cure_samples_fingerprint,
            "field_decode": {
                "output_rule": PAET_FIXED_OUTPUT_RULE,
                "zero_tie_policy": PAET_ZERO_TIE_POLICY,
                "sigmoid_applied": False,
                "PAET_threshold_search_performed": False,
                "exact_zero_field_pixels": (
                    self.exact_zero_field_pixels
                ),
                "negative_field_pixels": self.negative_field_pixels,
                "completion_pixels": self.completion_pixels,
            },
            "artifact_binding_fingerprint": (
                self.artifact_binding_fingerprint
            ),
            "comparison_protocol_fingerprint": (
                self.comparison_protocol_fingerprint
            ),
            "D_V_cache_binding": {
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
    def adapter_fingerprint(self) -> str:
        self.verify_unchanged()
        return stable_fingerprint(self._canonical_payload())


@dataclass(frozen=True)
class PAETFixedOperatingPoints:
    """Shared-metric outputs before formal population/binding packaging."""

    base_at_a: AggregateEvaluation
    base_at_b: AggregateEvaluation
    base_at_a_plus_cure: AggregateEvaluation
    base_at_b_selected_threshold: float
    base_candidate_ledger: CalibrationCandidateLedger


def _evaluate_fixed_operating_points(
    base_samples: Sequence[CalibrationSample],
    cure_samples: Sequence[CalibrationSample],
    *,
    occupancy_config: OccupancyConfig,
    match_config: MatchConfig,
    base_threshold_grid: Sequence[float],
    budget: FalseAlarmBudget,
) -> PAETFixedOperatingPoints:
    """Evaluate fixed PAET completion and search only the Base@B control."""

    if occupancy_config != OccupancyConfig(
        threshold=PAET_FORMAL_BASE_THRESHOLD,
        connectivity=8,
        min_component_area=1,
    ):
        raise ValueError("PAET formal evaluation fixes Base@A at 0.72/CC8")
    grid = _base_grid(base_threshold_grid)
    if not isinstance(budget, FalseAlarmBudget):
        raise TypeError("budget must be FalseAlarmBudget")
    base_tuple = tuple(base_samples)
    cure_tuple = tuple(cure_samples)
    if (
        not base_tuple
        or len(base_tuple) != len(cure_tuple)
        or tuple(sample.sample_id for sample in base_tuple)
        != tuple(sample.sample_id for sample in cure_tuple)
    ):
        raise ValueError("base and PAET samples must be non-empty/aligned")

    context = prepare_calibration_context(
        base_tuple,
        occupancy_config,
        match_config,
    )
    # No PAET method is supplied to the candidate ledger.  Consequently, the
    # only tasks it can enumerate are Base@B probability thresholds.
    ledger = evaluate_candidate_ledger(
        context,
        {},
        base_thresholds=grid,
        residual_thresholds_by_method={},
        base_method="Base@B",
    )
    selection = ledger.select("Base@B", budget)
    if (
        not selection.feasible
        or selection.threshold is None
        or selection.metrics is None
    ):
        raise RuntimeError("frozen Base@B selection is infeasible")

    cure_by_id = {
        sample.sample_id: sample for sample in cure_tuple
    }
    evaluations = []
    for row in context.rows:
        _, completion_probability, cure_gt = cure_by_id[
            row.sample_id
        ].normalized()
        if (
            not torch.equal(cure_gt, row.gt_mask)
            or bool(
                torch.any(
                    (completion_probability != 0)
                    & (completion_probability != 1)
                )
            )
        ):
            raise ValueError(
                "PAET completion samples must share GT and be exactly binary"
            )
        completion = completion_probability.to(torch.bool)
        if bool(torch.any(completion & row.occupancy)):
            raise ValueError("PAET completion overlaps Base@A occupancy")
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

    def with_budget(metrics: AggregateEvaluation) -> AggregateEvaluation:
        return replace(
            metrics,
            budget_violation=not budget.accepts(metrics),
        )

    return PAETFixedOperatingPoints(
        base_at_a=with_budget(context.anchor_metrics),
        base_at_b=with_budget(selection.metrics),
        base_at_a_plus_cure=with_budget(
            aggregate_evaluations(evaluations)
        ),
        base_at_b_selected_threshold=float(selection.threshold),
        base_candidate_ledger=ledger,
    )


def build_paet_fixed_d_v_samples(
    artifact_binding: PAETFormalArtifactBinding,
    *,
    batch_size: int = 8,
) -> PAETFixedDVSamples:
    """Run the exact formal PAET artifact over one verified D_V base cache."""

    if type(artifact_binding) is not PAETFormalArtifactBinding:
        raise TypeError(
            "artifact_binding must be PAETFormalArtifactBinding"
        )
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    artifact_binding.verify_unchanged()
    sources = artifact_binding._sealed_inputs()
    bundle = sources.bundle
    comparison_protocol = sources.comparison_protocol
    model = sources.artifact.model
    artifact_binding.verify_cache_and_protocol(bundle, comparison_protocol)
    artifact_binding.verify_model(model)
    bundle.verify_unchanged()
    feature_channels = {
        int(row.base_output.feature.shape[1]) for row in bundle.rows
    }
    if feature_channels != {model.feature_channels}:
        raise RuntimeError(
            "D_V feature channels differ from the PAET artifact"
        )
    parameters = tuple(model.parameters())
    if not parameters:
        raise RuntimeError("PAET model unexpectedly has no parameters")
    devices = {parameter.device for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    if len(devices) != 1 or dtypes != {torch.float32}:
        raise RuntimeError(
            "PAET D_V evaluation requires one-device FP32 parameters"
        )
    device = next(iter(devices))

    base_samples: list[CalibrationSample] = []
    cure_samples: list[CalibrationSample] = []
    zero_pixels = 0
    negative_pixels = 0
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
            probabilities_cpu = torch.cat(
                [row.base_output.probability for row in rows],
                dim=0,
            ).detach().to(device="cpu", dtype=torch.float32)
            occupancies = (
                probabilities_cpu >= PAET_FORMAL_BASE_THRESHOLD
            ).to(device=device)
            with torch.no_grad():
                field = model(features, occupancies)
                if (
                    tuple(field.shape) != tuple(occupancies.shape)
                    or field.dtype != torch.float32
                    or not bool(torch.isfinite(field).all())
                ):
                    raise RuntimeError(
                        "PAET field violates its formal output contract"
                    )
                completion = fixed_paet_completion(
                    field,
                    occupancies,
                )
            zero_pixels += int(torch.count_nonzero(field == 0).item())
            negative_pixels += int(torch.count_nonzero(field < 0).item())
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
                base_candidate = CalibrationSample(
                    row.sample_id,
                    base,
                    torch.zeros_like(base),
                    gt,
                )
                cure_candidate = CalibrationSample(
                    row.sample_id,
                    base,
                    completion_cpu[index : index + 1].to(
                        dtype=torch.float32
                    ),
                    gt,
                )
                normalized_base, zero, normalized_gt = (
                    base_candidate.normalized()
                )
                cure_base, fixed_completion, cure_gt = (
                    cure_candidate.normalized()
                )
                base_samples.append(
                    CalibrationSample(
                        row.sample_id,
                        normalized_base,
                        zero,
                        normalized_gt,
                    )
                )
                cure_samples.append(
                    CalibrationSample(
                        row.sample_id,
                        cure_base,
                        fixed_completion,
                        cure_gt,
                    )
                )
    finally:
        model.train(was_training)

    if module_state_fingerprint(model) != initial_state:
        raise RuntimeError("PAET model changed during D_V evaluation")
    artifact_binding.verify_model(model)
    bundle.verify_unchanged()
    base_tuple = tuple(base_samples)
    cure_tuple = tuple(cure_samples)
    ordered_ids = tuple(row.sample_id for row in bundle.rows)
    result = PAETFixedDVSamples(
        base_samples=base_tuple,
        cure_samples=cure_tuple,
        ordered_sample_ids=ordered_ids,
        base_samples_fingerprint=calibration_samples_fingerprint(
            base_tuple
        ),
        cure_samples_fingerprint=calibration_samples_fingerprint(
            cure_tuple
        ),
        exact_zero_field_pixels=zero_pixels,
        negative_field_pixels=negative_pixels,
        completion_pixels=completion_pixels,
        artifact_binding_fingerprint=(
            artifact_binding.binding_fingerprint
        ),
        comparison_protocol_fingerprint=(
            comparison_protocol.comparison_protocol_fingerprint
        ),
        d_v_base_index_fingerprint=bundle.base_index_fingerprint,
        d_v_image_fingerprint=bundle.d_v_image_fingerprint,
        d_v_gt_fingerprint=bundle.d_v_gt_fingerprint,
        _seal=_PAETFixedDVSamplesSeal(
            issuer=_PAET_FIXED_DV_SAMPLES_ISSUER,
            binding=artifact_binding,
        ),
    )
    return result


@dataclass(frozen=True, slots=True)
class _PAETFormalDVEvaluationResultSeal:
    issuer: object
    samples: PAETFixedDVSamples
    binding: PAETFormalArtifactBinding
    base_at_a: AggregateEvaluation
    base_at_b: AggregateEvaluation
    base_at_a_plus_cure: AggregateEvaluation
    canonical_fingerprint: str = ""


_PAET_FORMAL_DV_RESULT_ISSUER = object()


@dataclass(frozen=True)
class PAETFormalDVEvaluationResult:
    """Sealed three-operating-point report for seed-42 formal development."""

    base_at_a: AggregateEvaluation
    base_at_b: AggregateEvaluation
    base_at_a_plus_cure: AggregateEvaluation
    base_at_b_selected_threshold: float
    base_threshold_grid: tuple[float, ...]
    budget: FalseAlarmBudget
    adapter_fingerprint: str
    artifact_binding_fingerprint: str
    model_artifact_fingerprint: str
    model_state_fingerprint: str
    formal_training_protocol_fingerprint: str
    formal_schedule_fingerprint: str
    formal_training_result_fingerprint: str
    comparison_protocol_fingerprint: str
    stage_a_config_sha256: str
    manifest_fingerprint: str
    manifest_file_sha256: str
    preprocessing_fingerprint: str
    base_fingerprint: str
    base_state_fingerprint: str
    d_v_base_index_fingerprint: str
    d_v_image_fingerprint: str
    d_v_gt_fingerprint: str
    seed: int = PAET_FORMAL_SEED
    _seal: _PAETFormalDVEvaluationResultSeal | None = None

    def __post_init__(self) -> None:
        seal = self._seal
        if (
            type(seal) is not _PAETFormalDVEvaluationResultSeal
            or seal.issuer is not _PAET_FORMAL_DV_RESULT_ISSUER
        ):
            raise PermissionError(
                "PAET formal D_V result must come from the sealed evaluator"
            )
        if self.seed != PAET_FORMAL_SEED:
            raise ValueError("PAET formal D_V result fixes seed 42")
        if self.base_threshold_grid != PAET_FORMAL_BASE_THRESHOLD_GRID:
            raise ValueError("PAET formal Base@B grid changed")
        if (
            isinstance(self.base_at_b_selected_threshold, bool)
            or not isinstance(
                self.base_at_b_selected_threshold,
                (int, float),
            )
            or not isfinite(float(self.base_at_b_selected_threshold))
            or float(self.base_at_b_selected_threshold)
            not in self.base_threshold_grid
            or float(self.base_at_b_selected_threshold)
            > PAET_FORMAL_BASE_THRESHOLD
        ):
            raise ValueError("Base@B selected threshold is invalid")
        if not isinstance(self.budget, FalseAlarmBudget):
            raise TypeError("budget must be FalseAlarmBudget")
        for metrics in (
            self.base_at_a,
            self.base_at_b,
            self.base_at_a_plus_cure,
        ):
            if not isinstance(metrics, AggregateEvaluation):
                raise TypeError(
                    "formal operating points must be AggregateEvaluation"
                )
            if (
                metrics.images != FORMAL_DV_IMAGES
                or metrics.total_anchor_misses
                != FORMAL_DV_ANCHOR_MISSES
                or metrics.total_anchor_covered
                != FORMAL_DV_ANCHOR_COVERED
                or (
                    metrics.total_anchor_misses
                    + metrics.total_anchor_covered
                )
                != FORMAL_DV_TOTAL_TARGETS
            ):
                raise ValueError(
                    "formal D_V population must remain 120/170/23/147"
                )
        for name in (
            "adapter_fingerprint",
            "artifact_binding_fingerprint",
            "model_artifact_fingerprint",
            "model_state_fingerprint",
            "formal_training_protocol_fingerprint",
            "formal_schedule_fingerprint",
            "formal_training_result_fingerprint",
            "comparison_protocol_fingerprint",
            "stage_a_config_sha256",
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

    def verify_unchanged(self) -> None:
        seal = self._seal
        if (
            type(seal) is not _PAETFormalDVEvaluationResultSeal
            or seal.issuer is not _PAET_FORMAL_DV_RESULT_ISSUER
        ):
            raise PermissionError("PAET formal D_V result is unsealed")
        if (
            self.base_at_a is not seal.base_at_a
            or self.base_at_b is not seal.base_at_b
            or self.base_at_a_plus_cure is not seal.base_at_a_plus_cure
        ):
            raise RuntimeError("PAET formal D_V result metrics were replaced")
        seal.samples.verify_unchanged()
        seal.binding.verify_unchanged()
        if (
            self.adapter_fingerprint != seal.samples.adapter_fingerprint
            or self.artifact_binding_fingerprint
            != seal.binding.binding_fingerprint
            or self.comparison_protocol_fingerprint
            != seal.binding.comparison_protocol_fingerprint
            or (
                seal.canonical_fingerprint
                and stable_fingerprint(self._canonical_payload())
                != seal.canonical_fingerprint
            )
        ):
            raise RuntimeError("PAET formal D_V result changed after issuance")

    def _canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": PAET_FORMAL_DV_RESULT_SCHEMA,
            "method": PAET_FORMAL_METHOD,
            "seed": self.seed,
            "runtime_split": "D_V",
            "D_T_accessed": False,
            "output_contract": {
                "rule": PAET_FIXED_OUTPUT_RULE,
                "zero_tie_policy": PAET_ZERO_TIE_POLICY,
                "sigmoid_applied": False,
                "PAET_threshold_search_performed": False,
            },
            "Base@B_selection": {
                "policy": PAET_BASE_AT_B_SELECTION_POLICY,
                "threshold_search_performed": True,
                "candidate_threshold_grid": list(
                    self.base_threshold_grid
                ),
                "selected_threshold": (
                    self.base_at_b_selected_threshold
                ),
                "stage_a_config_sha256": self.stage_a_config_sha256,
                "budget": {
                    "pixel_fa_budget": self.budget.pixel_fa_budget,
                    "component_fa_per_mp_budget": (
                        self.budget.component_fa_per_mp_budget
                    ),
                    "raw_background_fa_budget": (
                        self.budget.raw_background_fa_budget
                    ),
                    "minimum_retention": self.budget.minimum_retention,
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
            "bindings": {
                "adapter_fingerprint": self.adapter_fingerprint,
                "artifact_binding_fingerprint": (
                    self.artifact_binding_fingerprint
                ),
                "model_artifact_fingerprint": (
                    self.model_artifact_fingerprint
                ),
                "model_state_fingerprint": (
                    self.model_state_fingerprint
                ),
                "formal_training_protocol_fingerprint": (
                    self.formal_training_protocol_fingerprint
                ),
                "formal_schedule_fingerprint": (
                    self.formal_schedule_fingerprint
                ),
                "formal_training_result_fingerprint": (
                    self.formal_training_result_fingerprint
                ),
                "comparison_protocol_fingerprint": (
                    self.comparison_protocol_fingerprint
                ),
                "manifest_fingerprint": self.manifest_fingerprint,
                "manifest_file_sha256": self.manifest_file_sha256,
                "preprocessing_fingerprint": (
                    self.preprocessing_fingerprint
                ),
                "base_fingerprint": self.base_fingerprint,
                "base_state_fingerprint": (
                    self.base_state_fingerprint
                ),
                "D_V_base_index_fingerprint": (
                    self.d_v_base_index_fingerprint
                ),
                "D_V_image_fingerprint": self.d_v_image_fingerprint,
                "D_V_GT_fingerprint": self.d_v_gt_fingerprint,
            },
            "development_gate_only": True,
            "final_model_success_established": False,
        }

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return self._canonical_payload()

    @property
    def result_fingerprint(self) -> str:
        self.verify_unchanged()
        return stable_fingerprint(self._canonical_payload())


def evaluate_paet_formal_d_v(
    samples: PAETFixedDVSamples,
    artifact_binding: PAETFormalArtifactBinding,
) -> PAETFormalDVEvaluationResult:
    """Evaluate Base@A/Base@B/Base@A+CURE with no PAET selection."""

    if type(samples) is not PAETFixedDVSamples:
        raise TypeError("samples must be PAETFixedDVSamples")
    if type(artifact_binding) is not PAETFormalArtifactBinding:
        raise TypeError(
            "artifact_binding must be PAETFormalArtifactBinding"
        )
    samples.verify_unchanged()
    artifact_binding.verify_unchanged()
    sources = artifact_binding._sealed_inputs()
    comparison_protocol = sources.comparison_protocol
    grid = _verified_stage_a_base_grid()
    if (
        samples.artifact_binding_fingerprint
        != artifact_binding.binding_fingerprint
        or samples.comparison_protocol_fingerprint
        != comparison_protocol.comparison_protocol_fingerprint
        or artifact_binding.comparison_protocol_fingerprint
        != comparison_protocol.comparison_protocol_fingerprint
        or artifact_binding.stage_a_config_sha256
        != PAET_FORMAL_STAGE_A_CONFIG_SHA256
    ):
        raise RuntimeError(
            "PAET samples/artifact/comparison protocol bindings differ"
        )
    points = _evaluate_fixed_operating_points(
        samples.base_samples,
        samples.cure_samples,
        occupancy_config=comparison_protocol.occupancy_config,
        match_config=comparison_protocol.match_config,
        base_threshold_grid=grid,
        budget=comparison_protocol.budget,
    )
    if len(samples.ordered_sample_ids) != FORMAL_DV_IMAGES:
        raise ValueError("formal PAET D_V evaluation requires 120 images")
    seal = _PAETFormalDVEvaluationResultSeal(
        issuer=_PAET_FORMAL_DV_RESULT_ISSUER,
        samples=samples,
        binding=artifact_binding,
        base_at_a=points.base_at_a,
        base_at_b=points.base_at_b,
        base_at_a_plus_cure=points.base_at_a_plus_cure,
    )
    result = PAETFormalDVEvaluationResult(
        base_at_a=points.base_at_a,
        base_at_b=points.base_at_b,
        base_at_a_plus_cure=points.base_at_a_plus_cure,
        base_at_b_selected_threshold=(
            points.base_at_b_selected_threshold
        ),
        base_threshold_grid=grid,
        budget=comparison_protocol.budget,
        adapter_fingerprint=samples.adapter_fingerprint,
        artifact_binding_fingerprint=(
            artifact_binding.binding_fingerprint
        ),
        model_artifact_fingerprint=(
            artifact_binding.artifact_fingerprint
        ),
        model_state_fingerprint=(
            artifact_binding.model_state_fingerprint
        ),
        formal_training_protocol_fingerprint=(
            artifact_binding.formal_training_protocol_fingerprint
        ),
        formal_schedule_fingerprint=(
            artifact_binding.formal_schedule_fingerprint
        ),
        formal_training_result_fingerprint=(
            artifact_binding.formal_training_result_fingerprint
        ),
        comparison_protocol_fingerprint=(
            comparison_protocol.comparison_protocol_fingerprint
        ),
        stage_a_config_sha256=artifact_binding.stage_a_config_sha256,
        manifest_fingerprint=artifact_binding.manifest_fingerprint,
        manifest_file_sha256=(
            artifact_binding.manifest_file_sha256
        ),
        preprocessing_fingerprint=(
            artifact_binding.preprocessing_fingerprint
        ),
        base_fingerprint=artifact_binding.base_fingerprint,
        base_state_fingerprint=(
            artifact_binding.base_state_fingerprint
        ),
        d_v_base_index_fingerprint=(
            samples.d_v_base_index_fingerprint
        ),
        d_v_image_fingerprint=samples.d_v_image_fingerprint,
        d_v_gt_fingerprint=samples.d_v_gt_fingerprint,
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
    "PAET_BASE_AT_B_SELECTION_POLICY",
    "PAET_FIXED_OUTPUT_RULE",
    "PAET_FORMAL_ARTIFACT_BINDING_SCHEMA",
    "PAET_FORMAL_BASE_THRESHOLD",
    "PAET_FORMAL_BASE_THRESHOLD_GRID",
    "PAET_FORMAL_DV_RESULT_SCHEMA",
    "PAET_FORMAL_DV_SAMPLE_SCHEMA",
    "PAET_FORMAL_EPOCHS",
    "PAET_FORMAL_METHOD",
    "PAET_FORMAL_SEED",
    "PAET_FORMAL_STEPS_PER_EPOCH",
    "PAET_FORMAL_STAGE_A_CONFIG_SHA256",
    "PAET_FORMAL_UPDATES",
    "PAET_ZERO_TIE_POLICY",
    "PAETFixedDVSamples",
    "PAETFixedOperatingPoints",
    "PAETFormalArtifactBinding",
    "PAETFormalDVEvaluationResult",
    "bind_paet_formal_artifact",
    "build_paet_fixed_d_v_samples",
    "evaluate_paet_formal_d_v",
    "fixed_paet_completion",
    "paet_formal_model_config_fingerprint",
]
