"""Actual-input and structural-RF audit for CURE-Lite coverage states.

The audit is intentionally representation comparative.  It evaluates the
same frozen, full-grid ``D_R`` population under:

``scalar_max``
    the current max-projected occupancy input;

``phase_preserving``
    an invertible PixelUnshuffle occupancy basis.

No model is trained here.  The result only authorizes the representation
whose actual input can realize the fixed target-field contract within the
current radius-two feature-grid receptive field.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from .cache.schema import stable_fingerprint
from .coverage_state_level_set import (
    CSLF_FEATURE_POLICY,
    CSLF_NORMALIZATION_EPSILON,
    normalize_cslf_feature,
    truncated_signed_distance_field,
)
from .coverage_state_raw_catalog import (
    CoverageStateNaturalRecord,
    CoverageStatePairRecord,
    CoverageStateRawCatalog,
)
from .decoder import project_occupancy_to_feature_grid
from .paired_types import tensor_content_fingerprint


COVERAGE_STATE_OBSERVABILITY_SCHEMA = (
    "cure-lite-coverage-state-observability-v2"
)
COVERAGE_STATE_FEATURE_RADIUS = 2
CoverageStateRepresentation = Literal["scalar_max", "phase_preserving"]


class CoverageStateObservabilityDecision(str, Enum):
    """Frozen outcomes of the representation-selection gate."""

    RAW_CONTRACT_INVALID = "RAW_CONTRACT_INVALID"
    INSUFFICIENT_INFORMATIVE_POPULATION = (
        "INSUFFICIENT_INFORMATIVE_POPULATION"
    )
    STATE_TARGET_CONTRACT_UNREALIZABLE = (
        "STATE_TARGET_CONTRACT_UNREALIZABLE"
    )
    PHASE_RF_UNREACHABLE = "PHASE_RF_UNREACHABLE"
    AUTHORIZE_PP_CSLF = "AUTHORIZE_PP_CSLF"
    AUTHORIZE_SCALAR_CSLF = "AUTHORIZE_SCALAR_CSLF"


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_bool_grid(value: Tensor, *, name: str) -> None:
    if (
        not isinstance(value, Tensor)
        or value.dtype != torch.bool
        or value.ndim != 4
        or value.shape[0] < 1
        or value.shape[1] != 1
        or min(value.shape[-2:]) < 1
    ):
        raise TypeError(f"{name} must be bool [B,1,H,W]")


def occupancy_to_scalar_grid(
    occupancy: Tensor,
    *,
    feature_size: tuple[int, int],
) -> Tensor:
    """Return the exact max-projected occupancy used by scalar CSLF."""

    _validate_bool_grid(occupancy, name="occupancy")
    if (
        not isinstance(feature_size, tuple)
        or len(feature_size) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in feature_size
        )
    ):
        raise ValueError("feature_size must contain two positive integers")
    return project_occupancy_to_feature_grid(
        occupancy,
        feature_size,
    ).to(torch.bool).contiguous()


def occupancy_to_phase_grid(
    occupancy: Tensor,
    *,
    stride: int,
) -> Tensor:
    """Losslessly move output-grid occupancy phases into channels."""

    _validate_bool_grid(occupancy, name="occupancy")
    stride = _positive_int(stride, name="stride")
    height, width = (int(value) for value in occupancy.shape[-2:])
    if height % stride != 0 or width % stride != 0:
        raise ValueError("occupancy grid must be divisible by stride")
    phase = F.pixel_unshuffle(
        occupancy.to(dtype=torch.float32),
        stride,
    ).to(dtype=torch.bool).contiguous()
    reconstructed = F.pixel_shuffle(
        phase.to(dtype=torch.float32),
        stride,
    ).to(dtype=torch.bool)
    if not torch.equal(reconstructed, occupancy):
        raise AssertionError("phase occupancy roundtrip changed full-grid state")
    return phase


def changed_feature_cells(
    representation_plus: Tensor,
    representation_minus: Tensor,
) -> Tensor:
    """Return feature cells whose representation differs in any channel."""

    if (
        not isinstance(representation_plus, Tensor)
        or not isinstance(representation_minus, Tensor)
        or representation_plus.dtype != torch.bool
        or representation_minus.dtype != torch.bool
        or representation_plus.ndim != 4
        or representation_plus.shape != representation_minus.shape
    ):
        raise TypeError(
            "representation endpoints must be aligned bool [B,K,h,w]"
        )
    return (
        representation_plus.ne(representation_minus)
        .any(dim=1, keepdim=True)
        .contiguous()
    )


def structural_output_support(
    changed_cells: Tensor,
    *,
    stride: int,
    feature_radius: int = COVERAGE_STATE_FEATURE_RADIUS,
) -> Tensor:
    """Map changed feature cells through the current structural RF.

    Invalid output pixels do not block convolutional propagation.  A valid
    mask is therefore applied by the caller only when response support is
    counted.
    """

    _validate_bool_grid(changed_cells, name="changed_cells")
    stride = _positive_int(stride, name="stride")
    if feature_radius != COVERAGE_STATE_FEATURE_RADIUS:
        raise ValueError("current CSLF fixes feature_radius at 2")
    expanded_cells = F.max_pool2d(
        changed_cells.to(dtype=torch.float32),
        kernel_size=2 * feature_radius + 1,
        stride=1,
        padding=feature_radius,
    ).to(dtype=torch.bool)
    phase_channels = stride**2
    expanded_phases = expanded_cells.expand(
        -1,
        phase_channels,
        -1,
        -1,
    )
    return F.pixel_shuffle(
        expanded_phases.to(dtype=torch.float32),
        stride,
    ).to(dtype=torch.bool).contiguous()


def target_response_support(
    target_field_plus: Tensor,
    target_field_minus: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Return support of the fixed target-field finite response."""

    if (
        not isinstance(target_field_plus, Tensor)
        or not isinstance(target_field_minus, Tensor)
        or target_field_plus.dtype != torch.float32
        or target_field_minus.dtype != torch.float32
        or target_field_plus.shape != target_field_minus.shape
        or target_field_plus.ndim != 4
    ):
        raise TypeError("target fields must be aligned float32 [B,1,H,W]")
    _validate_bool_grid(valid_mask, name="valid_mask")
    if target_field_plus.shape != valid_mask.shape:
        raise ValueError("target fields and valid_mask must share a shape")
    if (
        target_field_plus.device != target_field_minus.device
        or target_field_plus.device != valid_mask.device
    ):
        raise ValueError("target fields and valid_mask must share a device")
    if not bool(torch.isfinite(target_field_plus).all()) or not bool(
        torch.isfinite(target_field_minus).all()
    ):
        raise ValueError("target fields must be finite")
    return (
        target_field_plus.ne(target_field_minus) & valid_mask
    ).contiguous()


def actual_input_fingerprint(
    encoded_feature: Tensor,
    occupancy_representation: Tensor,
    *,
    representation: CoverageStateRepresentation,
    stride: int,
) -> str:
    """Fingerprint only tensors and policies consumed by the model."""

    if representation not in {"scalar_max", "phase_preserving"}:
        raise ValueError("unknown coverage-state representation")
    if (
        not isinstance(encoded_feature, Tensor)
        or encoded_feature.dtype != torch.float32
        or encoded_feature.ndim != 4
        or encoded_feature.shape[0] < 1
        or not bool(torch.isfinite(encoded_feature).all())
    ):
        raise TypeError("encoded_feature must be finite float32 [B,C,h,w]")
    if (
        not isinstance(occupancy_representation, Tensor)
        or occupancy_representation.dtype != torch.bool
        or occupancy_representation.ndim != 4
        or occupancy_representation.shape[0] != encoded_feature.shape[0]
        or occupancy_representation.shape[-2:]
        != encoded_feature.shape[-2:]
    ):
        raise TypeError(
            "occupancy representation must be aligned bool [B,K,h,w]"
        )
    stride = _positive_int(stride, name="stride")
    expected_channels = 1 if representation == "scalar_max" else stride**2
    if occupancy_representation.shape[1] != expected_channels:
        raise ValueError("occupancy representation channel count is invalid")
    return stable_fingerprint(
        {
            "schema_version": "cure-lite-actual-coverage-input-v1",
            "representation": representation,
            "feature_stride": stride,
            "feature_policy": CSLF_FEATURE_POLICY,
            "normalization_epsilon_hex": (
                CSLF_NORMALIZATION_EPSILON.hex()
            ),
            "encoded_feature": tensor_content_fingerprint(encoded_feature),
            "occupancy_representation": tensor_content_fingerprint(
                occupancy_representation
            ),
        }
    )


def _changed_coordinates(value: Tensor) -> tuple[tuple[int, int], ...]:
    if value.shape[0] != 1 or value.shape[1] != 1:
        raise ValueError("pair audit requires one changed-cell grid")
    return tuple(
        (int(row), int(column))
        for row, column in torch.nonzero(
            value[0, 0],
            as_tuple=False,
        ).tolist()
    )


@dataclass(frozen=True)
class CoverageStateRepresentationAudit:
    """One pair under one actual model-input representation."""

    pair_id: str
    representation: CoverageStateRepresentation
    encoded_feature_sha256: str
    occupancy_plus_sha256: str
    occupancy_minus_sha256: str
    input_plus_sha256: str
    input_minus_sha256: str
    target_plus_sha256: str
    target_minus_sha256: str
    valid_mask_sha256: str
    changed_feature_cells: tuple[tuple[int, int], ...]
    receptive_output_pixels: int
    target_response_pixels: int
    target_response_outside_rf_pixels: int
    duplicate_input_target_conflict: bool

    def canonical_payload(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "representation": self.representation,
            "encoded_feature_sha256": self.encoded_feature_sha256,
            "occupancy_plus_sha256": self.occupancy_plus_sha256,
            "occupancy_minus_sha256": self.occupancy_minus_sha256,
            "input_plus_sha256": self.input_plus_sha256,
            "input_minus_sha256": self.input_minus_sha256,
            "target_plus_sha256": self.target_plus_sha256,
            "target_minus_sha256": self.target_minus_sha256,
            "valid_mask_sha256": self.valid_mask_sha256,
            "changed_feature_cells": [
                [row, column] for row, column in self.changed_feature_cells
            ],
            "receptive_output_pixels": self.receptive_output_pixels,
            "target_response_pixels": self.target_response_pixels,
            "target_response_outside_rf_pixels": (
                self.target_response_outside_rf_pixels
            ),
            "duplicate_input_target_conflict": (
                self.duplicate_input_target_conflict
            ),
        }


@dataclass(frozen=True)
class CoverageStatePairObservabilityAudit:
    """Scalar/phase comparison for one representation-neutral pair."""

    pair_id: str
    sample_id: str
    pair_kind: str
    full_grid_changed: bool
    scalar: CoverageStateRepresentationAudit
    phase: CoverageStateRepresentationAudit
    target_response_hidden_only_by_scalar_pixels: int

    @property
    def hidden_by_scalar_projection(self) -> bool:
        return (
            not self.scalar.changed_feature_cells
            and bool(self.phase.changed_feature_cells)
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "sample_id": self.sample_id,
            "pair_kind": self.pair_kind,
            "full_grid_changed": self.full_grid_changed,
            "hidden_by_scalar_projection": (
                self.hidden_by_scalar_projection
            ),
            "target_response_hidden_only_by_scalar_pixels": (
                self.target_response_hidden_only_by_scalar_pixels
            ),
            "scalar": self.scalar.canonical_payload(),
            "phase": self.phase.canonical_payload(),
        }


@dataclass(frozen=True)
class _EndpointAudit:
    endpoint_id: str
    scalar_input_sha256: str
    phase_input_sha256: str
    target_field: Tensor
    valid_mask: Tensor


@dataclass
class _ObservabilityTensorCache:
    """Process-local memoization over immutable raw-catalog tensors."""

    encoded_features: dict[str, Tensor]
    representations: dict[
        tuple[str, tuple[int, int], int],
        tuple[Tensor, Tensor],
    ]
    target_fields: dict[tuple[str, str, int], Tensor]

    @classmethod
    def empty(cls) -> "_ObservabilityTensorCache":
        return cls(
            encoded_features={},
            representations={},
            target_fields={},
        )

    def encoded_feature(self, feature: Tensor) -> Tensor:
        key = tensor_content_fingerprint(feature)
        value = self.encoded_features.get(key)
        if value is None:
            value = normalize_cslf_feature(feature)
            self.encoded_features[key] = value
        return value

    def occupancy_representations(
        self,
        occupancy: Tensor,
        *,
        feature_size: tuple[int, int],
        stride: int,
    ) -> tuple[Tensor, Tensor]:
        key = (
            tensor_content_fingerprint(occupancy),
            feature_size,
            stride,
        )
        value = self.representations.get(key)
        if value is None:
            value = _representations(
                occupancy,
                feature_size=feature_size,
                stride=stride,
            )
            self.representations[key] = value
        return value

    def target_field(
        self,
        target: Tensor,
        valid_mask: Tensor,
        *,
        stride: int,
    ) -> Tensor:
        key = (
            tensor_content_fingerprint(target),
            tensor_content_fingerprint(valid_mask),
            stride,
        )
        value = self.target_fields.get(key)
        if value is None:
            value = truncated_signed_distance_field(
                target,
                valid_mask,
                radius=stride,
            )
            self.target_fields[key] = value
        return value


def _representations(
    occupancy: Tensor,
    *,
    feature_size: tuple[int, int],
    stride: int,
) -> tuple[Tensor, Tensor]:
    scalar = occupancy_to_scalar_grid(
        occupancy,
        feature_size=feature_size,
    )
    phase = occupancy_to_phase_grid(
        occupancy,
        stride=stride,
    )
    if phase.shape[-2:] != feature_size:
        raise ValueError(
            "phase occupancy grid must equal the frozen feature grid"
        )
    if not torch.equal(
        scalar,
        phase.any(dim=1, keepdim=True),
    ):
        raise AssertionError("scalar projection differs from phase occupancy union")
    return scalar, phase


def _audit_pair(
    pair: CoverageStatePairRecord,
    *,
    stride: int,
    cache: _ObservabilityTensorCache,
) -> tuple[CoverageStatePairObservabilityAudit, tuple[_EndpointAudit, ...]]:
    feature_size = tuple(int(value) for value in pair.feature.shape[-2:])
    encoded = cache.encoded_feature(pair.feature)
    scalar_plus, phase_plus = cache.occupancy_representations(
        pair.occupancy_plus,
        feature_size=feature_size,
        stride=stride,
    )
    scalar_minus, phase_minus = cache.occupancy_representations(
        pair.occupancy_minus,
        feature_size=feature_size,
        stride=stride,
    )
    field_plus = cache.target_field(
        pair.target_plus,
        pair.valid_mask,
        stride=stride,
    )
    field_minus = cache.target_field(
        pair.target_minus,
        pair.valid_mask,
        stride=stride,
    )
    response = target_response_support(
        field_plus,
        field_minus,
        pair.valid_mask,
    )
    representation_values = (
        ("scalar_max", scalar_plus, scalar_minus),
        ("phase_preserving", phase_plus, phase_minus),
    )
    audits: dict[str, CoverageStateRepresentationAudit] = {}
    endpoint_inputs: dict[str, tuple[str, str]] = {}
    receptive_supports: dict[str, Tensor] = {}
    for representation, plus, minus in representation_values:
        changed = changed_feature_cells(plus, minus)
        receptive = structural_output_support(
            changed,
            stride=stride,
        )
        receptive_valid = receptive & pair.valid_mask
        input_plus = actual_input_fingerprint(
            encoded,
            plus,
            representation=representation,
            stride=stride,
        )
        input_minus = actual_input_fingerprint(
            encoded,
            minus,
            representation=representation,
            stride=stride,
        )
        audits[representation] = CoverageStateRepresentationAudit(
            pair_id=pair.pair_id,
            representation=representation,
            encoded_feature_sha256=tensor_content_fingerprint(encoded),
            occupancy_plus_sha256=tensor_content_fingerprint(plus),
            occupancy_minus_sha256=tensor_content_fingerprint(minus),
            input_plus_sha256=input_plus,
            input_minus_sha256=input_minus,
            target_plus_sha256=tensor_content_fingerprint(field_plus),
            target_minus_sha256=tensor_content_fingerprint(field_minus),
            valid_mask_sha256=tensor_content_fingerprint(pair.valid_mask),
            changed_feature_cells=_changed_coordinates(changed),
            receptive_output_pixels=int(torch.count_nonzero(receptive_valid)),
            target_response_pixels=int(torch.count_nonzero(response)),
            target_response_outside_rf_pixels=int(
                torch.count_nonzero(response & ~receptive)
            ),
            duplicate_input_target_conflict=(
                input_plus == input_minus and bool(torch.any(response))
            ),
        )
        endpoint_inputs[representation] = (input_plus, input_minus)
        receptive_supports[representation] = receptive
    scalar_receptive = receptive_supports["scalar_max"]
    phase_receptive = receptive_supports["phase_preserving"]
    if bool(torch.any(scalar_receptive & ~phase_receptive)):
        raise AssertionError("scalar RF must be a subset of phase RF")
    outside_scalar = response & ~scalar_receptive
    outside_phase = response & ~phase_receptive
    hidden_only = response & phase_receptive & ~scalar_receptive
    if not torch.equal(outside_scalar, outside_phase | hidden_only):
        raise AssertionError("scalar/phase RF response accounting changed")
    if bool(pair.occupancy_plus.ne(pair.occupancy_minus).any()) != bool(
        changed_feature_cells(phase_plus, phase_minus).any()
    ):
        raise AssertionError("phase representation is not full-grid faithful")
    if pair.pair_kind == "clean_positive" and not bool(torch.any(response)):
        raise ValueError("clean_positive must have a nonzero target-field response")
    if pair.pair_kind in {"component_null", "identity_null"} and bool(
        torch.any(response)
    ):
        raise ValueError(f"{pair.pair_kind} must have zero target-field response")
    scalar_audit = audits["scalar_max"]
    phase_audit = audits["phase_preserving"]
    pair_audit = CoverageStatePairObservabilityAudit(
        pair_id=pair.pair_id,
        sample_id=pair.sample_id,
        pair_kind=pair.pair_kind,
        full_grid_changed=not torch.equal(
            pair.occupancy_plus,
            pair.occupancy_minus,
        ),
        scalar=scalar_audit,
        phase=phase_audit,
        target_response_hidden_only_by_scalar_pixels=int(
            torch.count_nonzero(hidden_only)
        ),
    )
    endpoints = (
        _EndpointAudit(
            endpoint_id=f"pair:{pair.pair_id}:plus",
            scalar_input_sha256=endpoint_inputs["scalar_max"][0],
            phase_input_sha256=endpoint_inputs["phase_preserving"][0],
            target_field=field_plus,
            valid_mask=pair.valid_mask,
        ),
        _EndpointAudit(
            endpoint_id=f"pair:{pair.pair_id}:minus",
            scalar_input_sha256=endpoint_inputs["scalar_max"][1],
            phase_input_sha256=endpoint_inputs["phase_preserving"][1],
            target_field=field_minus,
            valid_mask=pair.valid_mask,
        ),
    )
    return pair_audit, endpoints


def _audit_natural(
    record: CoverageStateNaturalRecord,
    *,
    stride: int,
    cache: _ObservabilityTensorCache,
) -> _EndpointAudit:
    feature_size = tuple(int(value) for value in record.feature.shape[-2:])
    encoded = cache.encoded_feature(record.feature)
    scalar, phase = cache.occupancy_representations(
        record.occupancy,
        feature_size=feature_size,
        stride=stride,
    )
    field = cache.target_field(
        record.target,
        record.valid_mask,
        stride=stride,
    )
    return _EndpointAudit(
        endpoint_id=f"natural:{record.record_id}",
        scalar_input_sha256=actual_input_fingerprint(
            encoded,
            scalar,
            representation="scalar_max",
            stride=stride,
        ),
        phase_input_sha256=actual_input_fingerprint(
            encoded,
            phase,
            representation="phase_preserving",
            stride=stride,
        ),
        target_field=field,
        valid_mask=record.valid_mask,
    )


def _duplicate_conflicts(
    endpoints: tuple[_EndpointAudit, ...],
    *,
    representation: CoverageStateRepresentation,
) -> tuple[tuple[str, ...], int]:
    key_name = (
        "scalar_input_sha256"
        if representation == "scalar_max"
        else "phase_input_sha256"
    )
    groups: dict[str, list[_EndpointAudit]] = {}
    for endpoint in endpoints:
        groups.setdefault(getattr(endpoint, key_name), []).append(endpoint)
    conflicting_keys: list[str] = []
    conflicting_pairs = 0
    for input_key in sorted(groups):
        values = sorted(groups[input_key], key=lambda value: value.endpoint_id)
        key_conflict = False
        for left, right in combinations(values, 2):
            common_valid = left.valid_mask & right.valid_mask
            if bool(torch.any(common_valid)) and bool(
                torch.any(
                    left.target_field[common_valid]
                    != right.target_field[common_valid]
                )
            ):
                key_conflict = True
                conflicting_pairs += 1
        if key_conflict:
            conflicting_keys.append(input_key)
    return tuple(conflicting_keys), conflicting_pairs


@dataclass(frozen=True)
class CoverageStatePopulationObservabilityReceipt:
    """Deterministic representation decision over one complete raw catalog."""

    raw_catalog_fingerprint: str
    natural_record_count: int
    pair_record_count: int
    unique_encoded_feature_tensors: int
    unique_occupancy_states: int
    unique_target_fields: int
    informative_clean_positive_count: int
    full_grid_changed_pairs: int
    phase_changed_pairs: int
    scalar_projected_changed_pairs: int
    hidden_by_scalar_projection_pairs: int
    clean_positive_hidden_pairs: int
    component_null_hidden_pairs: int
    target_response_pixels: int
    target_response_outside_scalar_rf_pixels: int
    target_response_outside_phase_rf_pixels: int
    target_response_hidden_only_by_scalar_pixels: int
    identity_null_nonidentical_count: int
    scalar_duplicate_input_target_conflict_keys: tuple[str, ...]
    phase_duplicate_input_target_conflict_keys: tuple[str, ...]
    scalar_duplicate_input_target_conflicting_pairs: int
    phase_duplicate_input_target_conflicting_pairs: int
    pair_audits: tuple[CoverageStatePairObservabilityAudit, ...]
    decision: CoverageStateObservabilityDecision

    @property
    def scalar_duplicate_input_target_conflicts(self) -> int:
        return len(self.scalar_duplicate_input_target_conflict_keys)

    @property
    def phase_duplicate_input_target_conflicts(self) -> int:
        return len(self.phase_duplicate_input_target_conflict_keys)

    @property
    def scalar_authorized(self) -> bool:
        return self.decision is CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF

    @property
    def pp_authorized(self) -> bool:
        return self.decision is CoverageStateObservabilityDecision.AUTHORIZE_PP_CSLF

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": COVERAGE_STATE_OBSERVABILITY_SCHEMA,
            "raw_catalog_fingerprint": self.raw_catalog_fingerprint,
            "counts": {
                "natural_record_count": self.natural_record_count,
                "pair_record_count": self.pair_record_count,
                "unique_encoded_feature_tensors": (
                    self.unique_encoded_feature_tensors
                ),
                "unique_occupancy_states": self.unique_occupancy_states,
                "unique_target_fields": self.unique_target_fields,
                "informative_clean_positive_count": (
                    self.informative_clean_positive_count
                ),
                "full_grid_changed_pairs": self.full_grid_changed_pairs,
                "phase_changed_pairs": self.phase_changed_pairs,
                "scalar_projected_changed_pairs": (
                    self.scalar_projected_changed_pairs
                ),
                "hidden_by_scalar_projection_pairs": (
                    self.hidden_by_scalar_projection_pairs
                ),
                "clean_positive_hidden_pairs": (
                    self.clean_positive_hidden_pairs
                ),
                "component_null_hidden_pairs": (
                    self.component_null_hidden_pairs
                ),
                "target_response_pixels": self.target_response_pixels,
                "target_response_outside_scalar_rf_pixels": (
                    self.target_response_outside_scalar_rf_pixels
                ),
                "target_response_outside_phase_rf_pixels": (
                    self.target_response_outside_phase_rf_pixels
                ),
                "target_response_hidden_only_by_scalar_pixels": (
                    self.target_response_hidden_only_by_scalar_pixels
                ),
                "identity_null_nonidentical_count": (
                    self.identity_null_nonidentical_count
                ),
                "scalar_duplicate_input_target_conflicts": (
                    self.scalar_duplicate_input_target_conflicts
                ),
                "phase_duplicate_input_target_conflicts": (
                    self.phase_duplicate_input_target_conflicts
                ),
                "scalar_duplicate_input_target_conflicting_pairs": (
                    self.scalar_duplicate_input_target_conflicting_pairs
                ),
                "phase_duplicate_input_target_conflicting_pairs": (
                    self.phase_duplicate_input_target_conflicting_pairs
                ),
            },
            "conflicting_input_keys": {
                "scalar_max": list(
                    self.scalar_duplicate_input_target_conflict_keys
                ),
                "phase_preserving": list(
                    self.phase_duplicate_input_target_conflict_keys
                ),
            },
            "pair_audits": [
                value.canonical_payload() for value in self.pair_audits
            ],
            "decision": self.decision.value,
            "scalar_authorized": self.scalar_authorized,
            "pp_authorized": self.pp_authorized,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def decide_observability(
    *,
    informative_clean_positive_count: int,
    phase_duplicate_input_target_conflicts: int,
    target_response_outside_phase_rf_pixels: int,
    scalar_duplicate_input_target_conflicts: int,
    target_response_outside_scalar_rf_pixels: int,
) -> CoverageStateObservabilityDecision:
    """Apply the predeclared representation decision in fixed order."""

    for name, value in (
        ("informative_clean_positive_count", informative_clean_positive_count),
        (
            "phase_duplicate_input_target_conflicts",
            phase_duplicate_input_target_conflicts,
        ),
        (
            "target_response_outside_phase_rf_pixels",
            target_response_outside_phase_rf_pixels,
        ),
        (
            "scalar_duplicate_input_target_conflicts",
            scalar_duplicate_input_target_conflicts,
        ),
        (
            "target_response_outside_scalar_rf_pixels",
            target_response_outside_scalar_rf_pixels,
        ),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if informative_clean_positive_count == 0:
        return (
            CoverageStateObservabilityDecision.INSUFFICIENT_INFORMATIVE_POPULATION
        )
    if phase_duplicate_input_target_conflicts:
        return (
            CoverageStateObservabilityDecision.STATE_TARGET_CONTRACT_UNREALIZABLE
        )
    if target_response_outside_phase_rf_pixels:
        return CoverageStateObservabilityDecision.PHASE_RF_UNREACHABLE
    if (
        scalar_duplicate_input_target_conflicts
        or target_response_outside_scalar_rf_pixels
    ):
        return CoverageStateObservabilityDecision.AUTHORIZE_PP_CSLF
    return CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF


def audit_population_observability(
    catalog: CoverageStateRawCatalog,
) -> CoverageStatePopulationObservabilityReceipt:
    """Audit one complete, canonical ``D_R`` population without training."""

    if not isinstance(catalog, CoverageStateRawCatalog):
        raise TypeError("catalog must be CoverageStateRawCatalog")
    stride = catalog.feature_stride
    cache = _ObservabilityTensorCache.empty()
    pair_results: list[CoverageStatePairObservabilityAudit] = []
    endpoints: list[_EndpointAudit] = []
    for record in catalog.natural_records:
        endpoints.append(
            _audit_natural(
                record,
                stride=stride,
                cache=cache,
            )
        )
    for pair in catalog.pair_records:
        result, pair_endpoints = _audit_pair(
            pair,
            stride=stride,
            cache=cache,
        )
        pair_results.append(result)
        endpoints.extend(pair_endpoints)
    ordered_pairs = tuple(sorted(pair_results, key=lambda value: value.pair_id))
    ordered_endpoints = tuple(
        sorted(endpoints, key=lambda value: value.endpoint_id)
    )
    scalar_keys, scalar_conflicting_pairs = _duplicate_conflicts(
        ordered_endpoints,
        representation="scalar_max",
    )
    phase_keys, phase_conflicting_pairs = _duplicate_conflicts(
        ordered_endpoints,
        representation="phase_preserving",
    )
    informative = sum(
        value.pair_kind == "clean_positive"
        and value.scalar.target_response_pixels > 0
        for value in ordered_pairs
    )
    outside_scalar = sum(
        value.scalar.target_response_outside_rf_pixels
        for value in ordered_pairs
    )
    outside_phase = sum(
        value.phase.target_response_outside_rf_pixels
        for value in ordered_pairs
    )
    decision = decide_observability(
        informative_clean_positive_count=informative,
        phase_duplicate_input_target_conflicts=len(phase_keys),
        target_response_outside_phase_rf_pixels=outside_phase,
        scalar_duplicate_input_target_conflicts=len(scalar_keys),
        target_response_outside_scalar_rf_pixels=outside_scalar,
    )
    return CoverageStatePopulationObservabilityReceipt(
        raw_catalog_fingerprint=catalog.catalog_fingerprint,
        natural_record_count=len(catalog.natural_records),
        pair_record_count=len(catalog.pair_records),
        unique_encoded_feature_tensors=len(cache.encoded_features),
        unique_occupancy_states=len(cache.representations),
        unique_target_fields=len(cache.target_fields),
        informative_clean_positive_count=informative,
        full_grid_changed_pairs=sum(
            value.full_grid_changed for value in ordered_pairs
        ),
        phase_changed_pairs=sum(
            bool(value.phase.changed_feature_cells)
            for value in ordered_pairs
        ),
        scalar_projected_changed_pairs=sum(
            bool(value.scalar.changed_feature_cells)
            for value in ordered_pairs
        ),
        hidden_by_scalar_projection_pairs=sum(
            value.hidden_by_scalar_projection for value in ordered_pairs
        ),
        clean_positive_hidden_pairs=sum(
            value.pair_kind == "clean_positive"
            and value.hidden_by_scalar_projection
            for value in ordered_pairs
        ),
        component_null_hidden_pairs=sum(
            value.pair_kind == "component_null"
            and value.hidden_by_scalar_projection
            for value in ordered_pairs
        ),
        target_response_pixels=sum(
            value.scalar.target_response_pixels for value in ordered_pairs
        ),
        target_response_outside_scalar_rf_pixels=outside_scalar,
        target_response_outside_phase_rf_pixels=outside_phase,
        target_response_hidden_only_by_scalar_pixels=sum(
            value.target_response_hidden_only_by_scalar_pixels
            for value in ordered_pairs
        ),
        identity_null_nonidentical_count=0,
        scalar_duplicate_input_target_conflict_keys=scalar_keys,
        phase_duplicate_input_target_conflict_keys=phase_keys,
        scalar_duplicate_input_target_conflicting_pairs=(
            scalar_conflicting_pairs
        ),
        phase_duplicate_input_target_conflicting_pairs=(
            phase_conflicting_pairs
        ),
        pair_audits=ordered_pairs,
        decision=decision,
    )


__all__ = [
    "COVERAGE_STATE_FEATURE_RADIUS",
    "COVERAGE_STATE_OBSERVABILITY_SCHEMA",
    "CoverageStateObservabilityDecision",
    "CoverageStatePairObservabilityAudit",
    "CoverageStatePopulationObservabilityReceipt",
    "CoverageStateRepresentationAudit",
    "actual_input_fingerprint",
    "audit_population_observability",
    "changed_feature_cells",
    "decide_observability",
    "occupancy_to_phase_grid",
    "occupancy_to_scalar_grid",
    "structural_output_support",
    "target_response_support",
]
