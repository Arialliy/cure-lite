"""Label-independent scene materializer for CURE-Lite relation state v2.

The frozen NLCC-v12 population assigned unrelated float values to each
generated state.  That made target extent and feature--coverage role
recoverable through sample-specific numerical identity.  This independent
population instead follows one fixed scene process:

1. create latent target objects with prototype vectors and high-resolution
   pixel support;
2. encode the complete latent field with one global PixelUnshuffle mapping;
3. create Base occupancy by selecting detected object pixels;
4. derive completion truth from which latent objects remain undetected.

Feature tensors are created before endpoint occupancy is applied and are
shared by a pair.  Group, endpoint, relation, and target labels are never
written into model input channels.  The fixed analytic projection exposed
below is a representational preflight for this materializer, not a trained
model result or a production initialization rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from .cache.schema import stable_fingerprint
from .paired_types import tensor_content_fingerprint
from .phase_resolved_feature_coverage_relation import (
    PhaseResolvedFeatureCoverageRelation,
    PhaseResolvedFeatureCoverageRelationConfig,
    PhaseResolvedFeatureCoverageRelationFields,
)


PFCR_POPULATION_ALGORITHM_VERSION = (
    "cure-lite.phase-resolved-relation-population.v1"
)
PFCR_FEATURE_STRIDE = 4
PFCR_FEATURE_HEIGHT = 7
PFCR_FEATURE_WIDTH = 7
PFCR_OUTPUT_HEIGHT = PFCR_FEATURE_HEIGHT * PFCR_FEATURE_STRIDE
PFCR_OUTPUT_WIDTH = PFCR_FEATURE_WIDTH * PFCR_FEATURE_STRIDE
PFCR_LATENT_DIM = 2
PFCR_PHASE_COUNT = PFCR_FEATURE_STRIDE**2
PFCR_FEATURE_CHANNELS = PFCR_LATENT_DIM * PFCR_PHASE_COUNT
PFCR_ONE_PIXEL_PATTERN = ((0, 3),)
PFCR_THREE_PIXEL_PATTERN = ((0, 3), (1, 2), (3, 0))
PFCR_COVERAGE_PHASE = (2, 2)
PFCR_PROTOTYPES = (
    (1.0, 0.0),
    (0.0, 1.0),
)
_TARGET_CELLS = ((2, 2), (3, 3))
_COVERAGE_OFFSETS = ((0, 1), (1, 0))


@dataclass(frozen=True)
class PhaseResolvedRelationPairSpec:
    """One latent scene with two occupancy endpoints."""

    pair_index: int
    scene_id: str
    target_prototype: int
    coverage_prototype: int
    target_cell: tuple[int, int]
    coverage_cell: tuple[int, int]
    target_phases: tuple[tuple[int, int], ...]
    coverage_phase: tuple[int, int] = PFCR_COVERAGE_PHASE

    def __post_init__(self) -> None:
        if (
            isinstance(self.pair_index, bool)
            or not isinstance(self.pair_index, int)
            or self.pair_index < 0
        ):
            raise ValueError("pair_index must be nonnegative")
        if not isinstance(self.scene_id, str) or not self.scene_id:
            raise ValueError("scene_id must be non-empty")
        for name, value in (
            ("target_prototype", self.target_prototype),
            ("coverage_prototype", self.coverage_prototype),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < len(PFCR_PROTOTYPES)
            ):
                raise ValueError(f"{name} is outside the prototype set")
        for name, cell in (
            ("target_cell", self.target_cell),
            ("coverage_cell", self.coverage_cell),
        ):
            if (
                not isinstance(cell, tuple)
                or len(cell) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value < PFCR_FEATURE_HEIGHT
                    for value in cell
                )
            ):
                raise ValueError(f"{name} must be a feature-grid cell")
        if (
            sum(
                abs(left - right)
                for left, right in zip(
                    self.target_cell,
                    self.coverage_cell,
                    strict=True,
                )
            )
            != 1
        ):
            raise ValueError(
                "coverage_cell must be cardinally adjacent to target_cell"
            )
        if self.target_phases not in (
            PFCR_ONE_PIXEL_PATTERN,
            PFCR_THREE_PIXEL_PATTERN,
        ):
            raise ValueError("target_phases must use a frozen pattern")
        for phase in (*self.target_phases, self.coverage_phase):
            if (
                not isinstance(phase, tuple)
                or len(phase) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value < PFCR_FEATURE_STRIDE
                    for value in phase
                )
            ):
                raise ValueError("phase lies outside the feature cell")

    @property
    def same_object_relation(self) -> bool:
        return self.target_prototype == self.coverage_prototype

    def manifest(self) -> dict[str, object]:
        return {
            "pair_index": self.pair_index,
            "scene_id": self.scene_id,
            "target_prototype": self.target_prototype,
            "coverage_prototype": self.coverage_prototype,
            "same_object_relation": self.same_object_relation,
            "target_cell": list(self.target_cell),
            "coverage_cell": list(self.coverage_cell),
            "target_phases": [
                list(phase) for phase in self.target_phases
            ],
            "coverage_phase": list(self.coverage_phase),
        }


@dataclass(frozen=True, eq=False)
class PhaseResolvedRelationState:
    """One model input and its completion truth."""

    state_id: str
    endpoint_role: str
    pair_index: int
    feature: Tensor
    occupancy: Tensor
    completion_target: Tensor
    valid_mask: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id:
            raise ValueError("state_id must be non-empty")
        if self.endpoint_role not in {"occupancy_plus", "occupancy_minus"}:
            raise ValueError("endpoint_role is invalid")
        if (
            isinstance(self.pair_index, bool)
            or not isinstance(self.pair_index, int)
            or self.pair_index < 0
        ):
            raise ValueError("pair_index must be nonnegative")
        if (
            not isinstance(self.feature, Tensor)
            or self.feature.dtype != torch.float32
            or self.feature.shape
            != (
                1,
                PFCR_FEATURE_CHANNELS,
                PFCR_FEATURE_HEIGHT,
                PFCR_FEATURE_WIDTH,
            )
            or self.feature.requires_grad
            or not bool(torch.isfinite(self.feature).all())
        ):
            raise ValueError("feature has an invalid frozen contract")
        expected_mask_shape = (
            1,
            1,
            PFCR_OUTPUT_HEIGHT,
            PFCR_OUTPUT_WIDTH,
        )
        for name, value in (
            ("occupancy", self.occupancy),
            ("completion_target", self.completion_target),
            ("valid_mask", self.valid_mask),
        ):
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.bool
                or tuple(value.shape) != expected_mask_shape
            ):
                raise ValueError(f"{name} has an invalid mask contract")
        if torch.any(self.completion_target & self.occupancy):
            raise ValueError("completion_target overlaps Base occupancy")
        if torch.any(self.completion_target & ~self.valid_mask):
            raise ValueError("completion_target extends outside valid_mask")
        for name in (
            "feature",
            "occupancy",
            "completion_target",
            "valid_mask",
        ):
            value = getattr(self, name).detach().cpu().clone().contiguous()
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class AnalyticRelationCompletionFields:
    """Reference relation fields and native completion score."""

    relation: PhaseResolvedFeatureCoverageRelationFields
    native_phase_release: Tensor
    completion_score: Tensor


def _output_pixel(
    cell: tuple[int, int],
    phase: tuple[int, int],
) -> tuple[int, int]:
    return (
        PFCR_FEATURE_STRIDE * cell[0] + phase[0],
        PFCR_FEATURE_STRIDE * cell[1] + phase[1],
    )


def _prototype_tensor(index: int) -> Tensor:
    return torch.tensor(PFCR_PROTOTYPES[index], dtype=torch.float32)


def build_phase_resolved_relation_pair_specs(
) -> tuple[PhaseResolvedRelationPairSpec, ...]:
    """Return the fixed balanced latent-scene population."""

    specs: list[PhaseResolvedRelationPairSpec] = []
    for target_prototype in range(len(PFCR_PROTOTYPES)):
        for target_cell, coverage_offset in zip(
            _TARGET_CELLS,
            _COVERAGE_OFFSETS,
            strict=True,
        ):
            coverage_cell = (
                target_cell[0] + coverage_offset[0],
                target_cell[1] + coverage_offset[1],
            )
            for target_phases in (
                PFCR_ONE_PIXEL_PATTERN,
                PFCR_THREE_PIXEL_PATTERN,
            ):
                for relation in ("same", "different"):
                    coverage_prototype = (
                        target_prototype
                        if relation == "same"
                        else 1 - target_prototype
                    )
                    pair_index = len(specs)
                    specs.append(
                        PhaseResolvedRelationPairSpec(
                            pair_index=pair_index,
                            scene_id=(
                                "pfcr-v2-scene-"
                                f"{pair_index:04d}"
                            ),
                            target_prototype=target_prototype,
                            coverage_prototype=coverage_prototype,
                            target_cell=target_cell,
                            coverage_cell=coverage_cell,
                            target_phases=target_phases,
                        )
                    )
    result = tuple(specs)
    if len(result) != 16:
        raise AssertionError("relation population size changed")
    return result


def _render_scene(
    spec: PhaseResolvedRelationPairSpec,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return feature, target-object mask, and coverage-object mask."""

    high_resolution_feature = torch.zeros(
        1,
        PFCR_LATENT_DIM,
        PFCR_OUTPUT_HEIGHT,
        PFCR_OUTPUT_WIDTH,
        dtype=torch.float32,
    )
    target_mask = torch.zeros(
        1,
        1,
        PFCR_OUTPUT_HEIGHT,
        PFCR_OUTPUT_WIDTH,
        dtype=torch.bool,
    )
    coverage_mask = torch.zeros_like(target_mask)
    target_prototype = _prototype_tensor(spec.target_prototype)
    coverage_prototype = _prototype_tensor(spec.coverage_prototype)

    for phase in spec.target_phases:
        row, column = _output_pixel(spec.target_cell, phase)
        high_resolution_feature[0, :, row, column] += target_prototype
        target_mask[0, 0, row, column] = True
    coverage_row, coverage_column = _output_pixel(
        spec.coverage_cell,
        spec.coverage_phase,
    )
    high_resolution_feature[
        0, :, coverage_row, coverage_column
    ] += coverage_prototype
    coverage_mask[0, 0, coverage_row, coverage_column] = True

    feature = F.pixel_unshuffle(
        high_resolution_feature,
        PFCR_FEATURE_STRIDE,
    ).contiguous()
    if tuple(feature.shape) != (
        1,
        PFCR_FEATURE_CHANNELS,
        PFCR_FEATURE_HEIGHT,
        PFCR_FEATURE_WIDTH,
    ):
        raise AssertionError("global scene encoder changed shape")
    return feature, target_mask, coverage_mask


def materialize_phase_resolved_relation_pair(
    spec: PhaseResolvedRelationPairSpec,
) -> tuple[PhaseResolvedRelationState, PhaseResolvedRelationState]:
    """Materialize plus/minus endpoints from one shared latent scene."""

    if not isinstance(spec, PhaseResolvedRelationPairSpec):
        raise TypeError("spec must be PhaseResolvedRelationPairSpec")
    feature, target_mask, coverage_mask = _render_scene(spec)
    occupancy_plus = coverage_mask.clone()
    occupancy_minus = torch.zeros_like(coverage_mask)

    if spec.same_object_relation:
        completion_plus = torch.zeros_like(target_mask)
    else:
        completion_plus = target_mask.clone()
    completion_minus = target_mask | coverage_mask
    valid = torch.ones_like(target_mask)
    plus = PhaseResolvedRelationState(
        state_id=f"pfcr-state-{2 * spec.pair_index:04d}",
        endpoint_role="occupancy_plus",
        pair_index=spec.pair_index,
        feature=feature,
        occupancy=occupancy_plus,
        completion_target=completion_plus,
        valid_mask=valid,
    )
    minus = PhaseResolvedRelationState(
        state_id=f"pfcr-state-{2 * spec.pair_index + 1:04d}",
        endpoint_role="occupancy_minus",
        pair_index=spec.pair_index,
        feature=feature,
        occupancy=occupancy_minus,
        completion_target=completion_minus,
        valid_mask=valid,
    )
    if not torch.equal(plus.feature, minus.feature):
        raise AssertionError("paired endpoints must share exact features")
    return plus, minus


def materialize_phase_resolved_relation_population(
    specs: Iterable[PhaseResolvedRelationPairSpec] | None = None,
) -> tuple[PhaseResolvedRelationState, ...]:
    """Return all endpoint states in canonical pair order."""

    values = (
        build_phase_resolved_relation_pair_specs()
        if specs is None
        else tuple(specs)
    )
    if not values or any(
        not isinstance(value, PhaseResolvedRelationPairSpec)
        for value in values
    ):
        raise ValueError(
            "specs must contain PhaseResolvedRelationPairSpec values"
        )
    if len({value.pair_index for value in values}) != len(values):
        raise ValueError("pair_index values must be unique")
    states: list[PhaseResolvedRelationState] = []
    for spec in values:
        states.extend(materialize_phase_resolved_relation_pair(spec))
    return tuple(states)


def phase_resolved_relation_population_manifest(
    specs: Iterable[PhaseResolvedRelationPairSpec] | None = None,
) -> dict[str, object]:
    """Return the complete algorithm and tensor fingerprint manifest."""

    values = (
        build_phase_resolved_relation_pair_specs()
        if specs is None
        else tuple(specs)
    )
    states = materialize_phase_resolved_relation_population(values)
    payload: dict[str, object] = {
        "schema_version": PFCR_POPULATION_ALGORITHM_VERSION,
        "scene_process": {
            "order": [
                "latent object field",
                "global PixelUnshuffle feature encoding",
                "endpoint occupancy selection",
                "completion truth from undetected latent objects",
            ],
            "feature_created_before_endpoint_occupancy": True,
            "paired_feature_byte_identity_required": True,
            "model_inputs": ["feature", "occupancy"],
            "metadata_is_model_input": False,
            "sample_specific_float_identity_used": False,
            "feature_stride": PFCR_FEATURE_STRIDE,
            "latent_dim": PFCR_LATENT_DIM,
            "feature_channels": PFCR_FEATURE_CHANNELS,
        },
        "specs": [spec.manifest() for spec in values],
        "states": [
            {
                "state_id": state.state_id,
                "endpoint_role": state.endpoint_role,
                "pair_index": state.pair_index,
                "feature": tensor_content_fingerprint(state.feature),
                "occupancy": tensor_content_fingerprint(state.occupancy),
                "completion_target": tensor_content_fingerprint(
                    state.completion_target
                ),
                "valid_mask": tensor_content_fingerprint(
                    state.valid_mask
                ),
            }
            for state in states
        ],
    }
    payload["population_fingerprint"] = stable_fingerprint(payload)
    return payload


def analytic_reference_relation_config(
) -> PhaseResolvedFeatureCoverageRelationConfig:
    """Return the unique config compatible with the fixed scene encoder."""

    return PhaseResolvedFeatureCoverageRelationConfig(
        feature_channels=PFCR_FEATURE_CHANNELS,
        feature_stride=PFCR_FEATURE_STRIDE,
        relation_dim=PFCR_LATENT_DIM,
    )


def set_analytic_reference_projection(
    module: PhaseResolvedFeatureCoverageRelation,
) -> None:
    """Set the fixed scene-encoder inverse used only for sufficiency checks."""

    if not isinstance(module, PhaseResolvedFeatureCoverageRelation):
        raise TypeError(
            "module must be PhaseResolvedFeatureCoverageRelation"
        )
    expected = analytic_reference_relation_config()
    if (
        module.config.feature_channels != expected.feature_channels
        or module.config.feature_stride != expected.feature_stride
        or module.config.relation_dim != expected.relation_dim
    ):
        raise ValueError(
            "module config does not match the fixed scene encoder"
        )
    with torch.no_grad():
        module.projection.weight.zero_()
        for phase_index in range(PFCR_PHASE_COUNT):
            for latent_index in range(PFCR_LATENT_DIM):
                query_output = (
                    phase_index * PFCR_LATENT_DIM + latent_index
                )
                phase_input = (
                    latent_index * PFCR_PHASE_COUNT + phase_index
                )
                module.projection.weight[
                    query_output,
                    phase_input,
                    0,
                    0,
                ] = 1.0
        key_offset = PFCR_PHASE_COUNT * PFCR_LATENT_DIM
        for latent_index in range(PFCR_LATENT_DIM):
            key_output = key_offset + latent_index
            for phase_index in range(PFCR_PHASE_COUNT):
                phase_input = (
                    latent_index * PFCR_PHASE_COUNT + phase_index
                )
                module.projection.weight[
                    key_output,
                    phase_input,
                    0,
                    0,
                ] = 1.0


def analytic_relation_completion(
    module: PhaseResolvedFeatureCoverageRelation,
    state: PhaseResolvedRelationState,
) -> AnalyticRelationCompletionFields:
    """Return the deterministic relation-controlled completion score."""

    if not isinstance(state, PhaseResolvedRelationState):
        raise TypeError("state must be PhaseResolvedRelationState")
    relation = module.forward_fields(state.feature, state.occupancy)
    # This dataset-free sufficiency check asks whether the relation state
    # contains the required phase identity, not whether one particular
    # evidence calibration has already been learned.  Input normalization
    # deliberately removes sample amplitude, so the analytic reference uses
    # the exact non-zero query support as its unit evidence.
    native_release = (
        relation.phase_evidence_strength > 0.0
    ).to(dtype=state.feature.dtype) * (
        1.0 - relation.coverage_burden
    )
    completion_score = F.pixel_shuffle(
        native_release,
        PFCR_FEATURE_STRIDE,
    )
    if tuple(completion_score.shape) != tuple(
        state.completion_target.shape
    ):
        raise AssertionError("analytic completion shape changed")
    return AnalyticRelationCompletionFields(
        relation=relation,
        native_phase_release=native_release.contiguous(),
        completion_score=completion_score.contiguous(),
    )


__all__ = [
    "AnalyticRelationCompletionFields",
    "PFCR_COVERAGE_PHASE",
    "PFCR_FEATURE_CHANNELS",
    "PFCR_FEATURE_HEIGHT",
    "PFCR_FEATURE_STRIDE",
    "PFCR_FEATURE_WIDTH",
    "PFCR_LATENT_DIM",
    "PFCR_ONE_PIXEL_PATTERN",
    "PFCR_OUTPUT_HEIGHT",
    "PFCR_OUTPUT_WIDTH",
    "PFCR_PHASE_COUNT",
    "PFCR_POPULATION_ALGORITHM_VERSION",
    "PFCR_PROTOTYPES",
    "PFCR_THREE_PIXEL_PATTERN",
    "PhaseResolvedRelationPairSpec",
    "PhaseResolvedRelationState",
    "analytic_reference_relation_config",
    "analytic_relation_completion",
    "build_phase_resolved_relation_pair_specs",
    "materialize_phase_resolved_relation_pair",
    "materialize_phase_resolved_relation_population",
    "phase_resolved_relation_population_manifest",
    "set_analytic_reference_projection",
]
