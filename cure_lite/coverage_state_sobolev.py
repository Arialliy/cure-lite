"""Product-domain Sobolev objective for CURE-Lite coverage states.

Natural completion states provide absolute field values.  A legal component
deletion provides an edge on the discrete coverage-state graph.  The loss
uses one error field and measures its value and spatial variation under a
fixed target-balanced measure:

``e = phi_theta(F, O) - phi_star(Y)``.

The coverage response is therefore not an independent synthetic endpoint
risk.  It is the finite difference ``e_minus - e_plus`` for the same frozen
feature and the same source image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

from .coverage_state_level_set import (
    CSLF_FIELD_AMPLITUDE,
    truncated_signed_distance_field,
)
from .decoder import project_occupancy_to_feature_grid
from .paired_types import PAIR_KINDS, PairExample


CSLF_OBJECTIVE_POLICY = (
    "balanced_rooted_w1p4_spatial_coverage_graph_field_energy_v5"
)
CSLF_MEASURE_POLICY = (
    "equal_mass_focus_support_exterior_band_far_background_measure_v2"
)
CSLF_NORM_ORDER = 4
CSLF_NORM_EPSILON = 1.0e-3


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class CoverageStateSobolevConfig:
    """Frozen mathematical coordinates of the CSLF objective."""

    truncation_radius: int
    field_amplitude: float = CSLF_FIELD_AMPLITUDE
    norm_order: int = CSLF_NORM_ORDER
    norm_epsilon: float = CSLF_NORM_EPSILON
    objective_policy: str = CSLF_OBJECTIVE_POLICY
    measure_policy: str = CSLF_MEASURE_POLICY

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "truncation_radius",
            _positive_int(
                self.truncation_radius,
                name="truncation_radius",
            ),
        )
        if (
            isinstance(self.field_amplitude, bool)
            or not isinstance(self.field_amplitude, float)
            or self.field_amplitude != CSLF_FIELD_AMPLITUDE
        ):
            raise ValueError("CURE-Lite CSLF fixes field_amplitude")
        if self.norm_order != CSLF_NORM_ORDER:
            raise ValueError("CURE-Lite CSLF fixes norm_order")
        if (
            isinstance(self.norm_epsilon, bool)
            or not isinstance(self.norm_epsilon, float)
            or self.norm_epsilon != CSLF_NORM_EPSILON
        ):
            raise ValueError("CURE-Lite CSLF fixes norm_epsilon")
        if self.objective_policy != CSLF_OBJECTIVE_POLICY:
            raise ValueError("CURE-Lite CSLF fixes objective_policy")
        if self.measure_policy != CSLF_MEASURE_POLICY:
            raise ValueError("CURE-Lite CSLF fixes measure_policy")


@dataclass(frozen=True)
class CoverageStatePairBatch:
    """One feature copy and two coverage/target endpoints per state edge."""

    feature: Tensor
    occupancy_plus: Tensor
    occupancy_minus: Tensor
    target_plus: Tensor
    target_minus: Tensor
    valid_mask: Tensor
    pair_ids: tuple[str, ...]
    pair_kinds: tuple[str, ...]
    sample_ids: tuple[str, ...]

    def validate(self) -> None:
        tensors = (
            self.feature,
            self.occupancy_plus,
            self.occupancy_minus,
            self.target_plus,
            self.target_minus,
            self.valid_mask,
        )
        if any(not isinstance(value, Tensor) for value in tensors):
            raise TypeError("all coverage-state batch values must be tensors")
        if (
            self.feature.ndim != 4
            or self.feature.shape[0] < 1
            or not self.feature.is_floating_point()
            or not bool(torch.isfinite(self.feature).all())
        ):
            raise ValueError("feature must be finite floating [B,C,h,w]")
        output_shape = tuple(self.occupancy_plus.shape)
        if (
            len(output_shape) != 4
            or output_shape[0] != self.feature.shape[0]
            or output_shape[1] != 1
            or min(output_shape[-2:]) < 1
        ):
            raise ValueError(
                "occupancy_plus must align as nonempty [B,1,H,W]"
            )
        for name, value in (
            ("occupancy_plus", self.occupancy_plus),
            ("occupancy_minus", self.occupancy_minus),
            ("target_plus", self.target_plus),
            ("target_minus", self.target_minus),
            ("valid_mask", self.valid_mask),
        ):
            if value.dtype != torch.bool or tuple(value.shape) != output_shape:
                raise ValueError(
                    f"{name} must be bool and output-grid aligned"
                )
        if len({value.device for value in tensors}) != 1:
            raise ValueError("all coverage-state tensors must share a device")
        if bool(torch.any(self.occupancy_minus & ~self.occupancy_plus)):
            raise ValueError("occupancy_minus must be a subset of occupancy_plus")
        if bool(
            torch.any(
                (
                    self.occupancy_plus
                    | self.target_plus
                    | self.target_minus
                )
                & ~self.valid_mask
            )
        ):
            raise ValueError(
                "occupancy and targets must remain inside valid_mask"
            )
        if bool(torch.any(self.target_plus & ~self.target_minus)):
            raise ValueError(
                "a deletion edge may only add residual target support"
            )
        if bool(torch.any(self.target_plus & self.occupancy_plus)):
            raise ValueError(
                "target_plus must remain writable under occupancy_plus"
            )
        if bool(torch.any(self.target_minus & self.occupancy_minus)):
            raise ValueError(
                "target_minus must remain writable under occupancy_minus"
            )
        batch = int(self.feature.shape[0])
        metadata = (self.pair_ids, self.pair_kinds, self.sample_ids)
        if any(
            not isinstance(values, tuple) or len(values) != batch
            for values in metadata
        ):
            raise ValueError("coverage-state metadata must align with batch")
        if any(kind not in PAIR_KINDS for kind in self.pair_kinds):
            raise ValueError("coverage-state batch contains an unknown pair kind")
        feature_size = tuple(
            int(value) for value in self.feature.shape[-2:]
        )
        output_size = tuple(
            int(value) for value in self.occupancy_plus.shape[-2:]
        )
        if any(
            output % feature != 0
            for output, feature in zip(output_size, feature_size)
        ):
            raise ValueError(
                "output grid must be an integer multiple of feature grid"
            )
        projected_plus = project_occupancy_to_feature_grid(
            self.occupancy_plus,
            feature_size,
        )
        projected_minus = project_occupancy_to_feature_grid(
            self.occupancy_minus,
            feature_size,
        )
        for index, kind in enumerate(self.pair_kinds):
            occupancy_equal = torch.equal(
                self.occupancy_plus[index],
                self.occupancy_minus[index],
            )
            targets_equal = torch.equal(
                self.target_plus[index],
                self.target_minus[index],
            )
            projected_equal = torch.equal(
                projected_plus[index],
                projected_minus[index],
            )
            if kind == "clean_positive":
                if occupancy_equal or targets_equal or projected_equal:
                    raise ValueError(
                        "clean_positive must visibly change projected coverage "
                        "and target field"
                    )
            elif kind == "component_null":
                if occupancy_equal or not targets_equal or projected_equal:
                    raise ValueError(
                        "component_null visibly changes projected coverage but "
                        "not target field"
                    )
            elif (
                not occupancy_equal
                or not targets_equal
                or not projected_equal
            ):
                raise ValueError(
                    "identity_null must preserve both endpoints exactly"
                )


def stack_coverage_state_pairs(
    examples: Iterable[PairExample],
    *,
    device: torch.device | str,
) -> CoverageStatePairBatch:
    """Stack existing lineage-safe pairs with both absolute target states."""

    values = tuple(examples)
    if not values:
        raise ValueError("cannot stack an empty pair selection")
    if any(not isinstance(value, PairExample) for value in values):
        raise TypeError("examples must contain PairExample values")
    if len({tuple(value.feature.shape[1:]) for value in values}) != 1:
        raise ValueError("pair feature shapes differ")
    if len({tuple(value.occupancy_plus.shape) for value in values}) != 1:
        raise ValueError("pair output grids differ")
    batch = CoverageStatePairBatch(
        feature=torch.cat(
            [value.feature for value in values],
            dim=0,
        ).to(device),
        occupancy_plus=torch.stack(
            [value.occupancy_plus for value in values],
            dim=0,
        ).to(device),
        occupancy_minus=torch.stack(
            [value.occupancy_minus for value in values],
            dim=0,
        ).to(device),
        target_plus=torch.stack(
            [value.completion_plus for value in values],
            dim=0,
        ).to(device),
        target_minus=torch.stack(
            [value.completion_minus for value in values],
            dim=0,
        ).to(device),
        valid_mask=torch.stack(
            [value.image_valid_mask for value in values],
            dim=0,
        ).to(device),
        pair_ids=tuple(value.pair_id for value in values),
        pair_kinds=tuple(value.pair_kind for value in values),
        sample_ids=tuple(value.sample_id for value in values),
    )
    batch.validate()
    return batch


def _validate_field_inputs(
    field: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    *,
    name: str,
) -> None:
    if (
        not isinstance(field, Tensor)
        or not field.is_floating_point()
        or field.ndim != 4
        or field.shape[0] < 1
        or field.shape[1] != 1
        or not bool(torch.isfinite(field).all())
    ):
        raise ValueError(f"{name} must be finite floating [B,1,H,W]")
    for target_name, value in (
        ("target", target),
        ("valid_mask", valid_mask),
    ):
        if (
            not isinstance(value, Tensor)
            or value.dtype != torch.bool
            or tuple(value.shape) != tuple(field.shape)
            or value.device != field.device
        ):
            raise ValueError(
                f"{name} {target_name} must be aligned bool"
            )
    if bool(torch.any(target & ~valid_mask)):
        raise ValueError(f"{name} target extends outside valid_mask")
    if not bool(valid_mask.flatten(1).any(dim=1).all()):
        raise ValueError(f"{name} every state needs a nonempty valid domain")


def _balanced_field_measure(
    target: Tensor,
    target_field: Tensor,
    valid_mask: Tensor,
    *,
    amplitude: float,
) -> Tensor:
    """Assign equal mass to nonempty target, exterior-band, and far strata.

    This is one fixed integration measure for the scalar-field norm.  It
    prevents a one-pixel target from receiving ``1 / (H W)`` of the field
    risk while retaining a uniform measure for an empty completion state.
    """

    exterior_band = (
        valid_mask
        & ~target
        & (target_field < amplitude)
    )
    far_background = valid_mask & ~target & ~exterior_band
    strata = (target, exterior_band, far_background)
    measure = torch.zeros_like(target_field)
    active_count = torch.zeros(
        target.shape[0],
        dtype=torch.int64,
        device=target.device,
    )
    for stratum in strata:
        count = stratum.flatten(1).sum(dim=1)
        active = count > 0
        active_count = active_count + active.to(dtype=torch.int64)
        denominator = count.clamp_min(1).to(dtype=target_field.dtype)
        measure = measure + (
            stratum.to(dtype=target_field.dtype)
            / denominator[:, None, None, None]
        )
    if not bool((active_count > 0).all()):
        raise ValueError("balanced field measure has an empty valid domain")
    measure = measure / active_count.to(
        dtype=target_field.dtype
    )[:, None, None, None]
    mass = measure.flatten(1).sum(dim=1)
    if (
        not bool(torch.isfinite(measure).all())
        or not bool(torch.allclose(mass, torch.ones_like(mass)))
        or bool(torch.any(measure[~valid_mask] != 0.0))
    ):
        raise AssertionError("balanced field measure contract changed")
    return measure.contiguous()


def _per_state_vector_p4_power(
    values: tuple[Tensor, ...],
    measure: Tensor,
) -> Tensor:
    if not values:
        raise ValueError("vector field energy needs at least one component")
    if any(tuple(value.shape) != tuple(measure.shape) for value in values):
        raise ValueError("vector components and measure shapes differ")
    mass = measure.flatten(1).sum(dim=1)
    if not bool(torch.allclose(mass, torch.ones_like(mass))):
        raise ValueError("every state measure must have unit mass")
    squared_norm = torch.zeros_like(values[0])
    for value in values:
        squared_norm = squared_norm + value.square()
    vector_square = squared_norm / float(len(values))
    energy = vector_square.square()
    return (energy * measure).flatten(1).sum(dim=1)


def _per_state_spatial_vector_p4_power(
    values: tuple[Tensor, ...],
    valid_mask: Tensor,
    measure: Tensor,
) -> Tensor:
    horizontal_mask = valid_mask[..., 1:] & valid_mask[..., :-1]
    vertical_mask = valid_mask[..., 1:, :] & valid_mask[..., :-1, :]
    horizontal_measure = 0.5 * (
        measure[..., 1:] + measure[..., :-1]
    )
    vertical_measure = 0.5 * (
        measure[..., 1:, :] + measure[..., :-1, :]
    )
    batch = int(measure.shape[0])
    sums = torch.zeros(batch, dtype=measure.dtype, device=measure.device)
    masses = torch.zeros(
        batch,
        dtype=measure.dtype,
        device=measure.device,
    )
    for axis, mask, edge_measure in (
        ("horizontal", horizontal_mask, horizontal_measure),
        ("vertical", vertical_mask, vertical_measure),
    ):
        squared_norm = torch.zeros_like(edge_measure)
        for value in values:
            if axis == "horizontal":
                difference = value[..., 1:] - value[..., :-1]
            else:
                difference = value[..., 1:, :] - value[..., :-1, :]
            squared_norm = squared_norm + difference.square()
        vector_square = squared_norm / float(len(values))
        energy = vector_square.square()
        weighted = edge_measure * mask.to(dtype=edge_measure.dtype)
        sums = sums + (energy * weighted).flatten(1).sum(dim=1)
        masses = masses + weighted.flatten(1).sum(dim=1)
    if not bool((masses > 0.0).all()):
        raise ValueError(
            "spatial Sobolev term requires at least one valid grid edge"
        )
    return sums / masses


@dataclass(frozen=True)
class CoverageStateAbsoluteLossFields:
    """Absolute spatial-field error for natural factual/no-miss states."""

    loss: Tensor
    value_power: Tensor
    spatial_power: Tensor
    per_state_loss: Tensor
    per_state_value_power: Tensor
    per_state_spatial_power: Tensor
    target_field: Tensor
    integration_measure: Tensor


@dataclass(frozen=True)
class CoverageStateAbsoluteTargets:
    """Precomputed, reusable natural-state geometry."""

    target_field: Tensor
    integration_measure: Tensor
    valid_mask: Tensor

    def validate(self) -> None:
        values = (
            self.target_field,
            self.integration_measure,
            self.valid_mask,
        )
        if any(not isinstance(value, Tensor) for value in values):
            raise TypeError("absolute target geometry must contain tensors")
        if (
            not self.target_field.is_floating_point()
            or not self.integration_measure.is_floating_point()
            or self.valid_mask.dtype != torch.bool
            or self.target_field.ndim != 4
            or self.target_field.shape[0] < 1
            or self.target_field.shape[1] != 1
            or len({tuple(value.shape) for value in values}) != 1
            or len({value.device for value in values}) != 1
            or self.target_field.dtype != torch.float32
            or self.integration_measure.dtype != torch.float32
            or not bool(torch.isfinite(self.target_field).all())
            or not bool(torch.isfinite(self.integration_measure).all())
        ):
            raise ValueError("invalid absolute target geometry")
        mass = self.integration_measure.flatten(1).sum(dim=1)
        if (
            not bool(torch.allclose(mass, torch.ones_like(mass)))
            or bool(torch.any(self.integration_measure < 0.0))
            or bool(torch.any(self.integration_measure[~self.valid_mask] != 0.0))
        ):
            raise ValueError("absolute integration measure is invalid")


def prepare_coverage_state_absolute_targets(
    target: Tensor,
    valid_mask: Tensor,
    *,
    config: CoverageStateSobolevConfig,
) -> CoverageStateAbsoluteTargets:
    """Precompute a natural target field and its fixed integration measure."""

    if not isinstance(config, CoverageStateSobolevConfig):
        raise TypeError("config must be CoverageStateSobolevConfig")
    if not isinstance(target, Tensor):
        raise TypeError("target must be a tensor")
    reference = torch.zeros(
        target.shape,
        dtype=torch.float32,
        device=target.device,
    )
    _validate_field_inputs(
        reference,
        target,
        valid_mask,
        name="state",
    )
    target_field = truncated_signed_distance_field(
        target,
        valid_mask,
        radius=config.truncation_radius,
        amplitude=config.field_amplitude,
    )
    measure = _balanced_field_measure(
        target,
        target_field,
        valid_mask,
        amplitude=config.field_amplitude,
    )
    result = CoverageStateAbsoluteTargets(
        target_field=target_field,
        integration_measure=measure,
        valid_mask=valid_mask.contiguous(),
    )
    result.validate()
    return result


def coverage_state_absolute_sobolev_loss_from_targets(
    field: Tensor,
    targets: CoverageStateAbsoluteTargets,
    *,
    config: CoverageStateSobolevConfig,
) -> CoverageStateAbsoluteLossFields:
    """Measure a field against precomputed natural-state geometry."""

    if not isinstance(config, CoverageStateSobolevConfig):
        raise TypeError("config must be CoverageStateSobolevConfig")
    if not isinstance(targets, CoverageStateAbsoluteTargets):
        raise TypeError("targets must be CoverageStateAbsoluteTargets")
    targets.validate()
    if (
        tuple(field.shape) != tuple(targets.target_field.shape)
        or field.device != targets.target_field.device
        or field.dtype != torch.float32
        or not field.is_floating_point()
        or not bool(torch.isfinite(field).all())
    ):
        raise ValueError("field and absolute target geometry must align")
    target_field = targets.target_field
    measure = targets.integration_measure
    valid_mask = targets.valid_mask
    error = field - target_field
    per_value_power = _per_state_vector_p4_power(
        (error,),
        measure,
    )
    per_spatial_power = _per_state_spatial_vector_p4_power(
        (error,),
        valid_mask,
        measure,
    )
    per_loss = (
        0.5 * (per_value_power + per_spatial_power)
        + config.norm_epsilon**config.norm_order
    ).pow(1.0 / float(config.norm_order)) - config.norm_epsilon
    loss = per_loss.mean()
    return CoverageStateAbsoluteLossFields(
        loss=loss,
        value_power=per_value_power.mean(),
        spatial_power=per_spatial_power.mean(),
        per_state_loss=per_loss,
        per_state_value_power=per_value_power,
        per_state_spatial_power=per_spatial_power,
        target_field=target_field,
        integration_measure=measure,
    )


def coverage_state_absolute_sobolev_loss(
    field: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    *,
    config: CoverageStateSobolevConfig,
) -> CoverageStateAbsoluteLossFields:
    """Measure one natural state in the fixed rooted W1,p field energy."""

    targets = prepare_coverage_state_absolute_targets(
        target,
        valid_mask,
        config=config,
    )
    return coverage_state_absolute_sobolev_loss_from_targets(
        field,
        targets,
        config=config,
    )


@dataclass(frozen=True)
class CoverageStatePairLossFields:
    """Anchor and controlled-deletion finite response of one pair batch."""

    loss: Tensor
    value_power: Tensor
    spatial_power: Tensor
    per_state_loss: Tensor
    per_state_value_power: Tensor
    per_state_spatial_power: Tensor
    target_field_plus: Tensor
    target_field_minus: Tensor
    predicted_coverage_response: Tensor
    target_coverage_response: Tensor
    anchor_error: Tensor
    response_error: Tensor
    focus_support: Tensor
    focus_support_field: Tensor
    integration_measure: Tensor


@dataclass(frozen=True)
class CoverageStatePairTargets:
    """Precomputed geometry shared by coupled and independent controls."""

    target_field_plus: Tensor
    target_field_minus: Tensor
    focus_support: Tensor
    focus_support_field: Tensor
    integration_measure: Tensor
    valid_mask: Tensor

    def validate(self) -> None:
        floating = (
            self.target_field_plus,
            self.target_field_minus,
            self.focus_support_field,
            self.integration_measure,
        )
        binary = (self.focus_support, self.valid_mask)
        all_values = (*floating, *binary)
        if (
            any(not isinstance(value, Tensor) for value in all_values)
            or any(not value.is_floating_point() for value in floating)
            or any(value.dtype != torch.bool for value in binary)
            or self.target_field_plus.ndim != 4
            or self.target_field_plus.shape[0] < 1
            or self.target_field_plus.shape[1] != 1
            or len({tuple(value.shape) for value in all_values}) != 1
            or len({value.device for value in all_values}) != 1
            or any(value.dtype != torch.float32 for value in floating)
            or any(not bool(torch.isfinite(value).all()) for value in floating)
        ):
            raise ValueError("invalid pair target geometry")
        mass = self.integration_measure.flatten(1).sum(dim=1)
        if (
            not bool(torch.allclose(mass, torch.ones_like(mass)))
            or bool(torch.any(self.integration_measure < 0.0))
            or bool(torch.any(self.integration_measure[~self.valid_mask] != 0.0))
            or bool(torch.any(self.focus_support & ~self.valid_mask))
        ):
            raise ValueError("pair integration geometry is invalid")


def prepare_coverage_state_pair_targets(
    occupancy_plus: Tensor,
    occupancy_minus: Tensor,
    target_plus: Tensor,
    target_minus: Tensor,
    valid_mask: Tensor,
    *,
    config: CoverageStateSobolevConfig,
) -> CoverageStatePairTargets:
    """Precompute the exact geometry used by both pair objectives."""

    if not isinstance(config, CoverageStateSobolevConfig):
        raise TypeError("config must be CoverageStateSobolevConfig")
    if not isinstance(target_plus, Tensor):
        raise TypeError("target_plus must be a tensor")
    reference = torch.zeros(
        target_plus.shape,
        dtype=torch.float32,
        device=target_plus.device,
    )
    _validate_field_inputs(
        reference,
        target_plus,
        valid_mask,
        name="plus",
    )
    _validate_field_inputs(
        reference,
        target_minus,
        valid_mask,
        name="minus",
    )
    if bool(torch.any(target_plus & ~target_minus)):
        raise ValueError("pair deletion may only add residual support")
    for name, value in (
        ("occupancy_plus", occupancy_plus),
        ("occupancy_minus", occupancy_minus),
    ):
        if (
            not isinstance(value, Tensor)
            or value.dtype != torch.bool
            or tuple(value.shape) != tuple(reference.shape)
            or value.device != reference.device
        ):
            raise ValueError(f"{name} must be aligned bool")
    if bool(torch.any(occupancy_minus & ~occupancy_plus)):
        raise ValueError("occupancy_minus must be a subset of occupancy_plus")
    if bool(torch.any((occupancy_plus | occupancy_minus) & ~valid_mask)):
        raise ValueError("pair occupancy extends outside valid_mask")
    if bool(torch.any(target_plus & occupancy_plus)):
        raise ValueError("target_plus is not writable under occupancy_plus")
    if bool(torch.any(target_minus & occupancy_minus)):
        raise ValueError("target_minus is not writable under occupancy_minus")

    target_field_plus = truncated_signed_distance_field(
        target_plus,
        valid_mask,
        radius=config.truncation_radius,
        amplitude=config.field_amplitude,
    )
    target_field_minus = truncated_signed_distance_field(
        target_minus,
        valid_mask,
        radius=config.truncation_radius,
        amplitude=config.field_amplitude,
    )
    removed_coverage = occupancy_plus & ~occupancy_minus
    focus_support = removed_coverage | target_minus
    focus_support_field = truncated_signed_distance_field(
        focus_support,
        valid_mask,
        radius=config.truncation_radius,
        amplitude=config.field_amplitude,
    )
    measure = _balanced_field_measure(
        focus_support,
        focus_support_field,
        valid_mask,
        amplitude=config.field_amplitude,
    )
    result = CoverageStatePairTargets(
        target_field_plus=target_field_plus,
        target_field_minus=target_field_minus,
        focus_support=focus_support.contiguous(),
        focus_support_field=focus_support_field,
        integration_measure=measure,
        valid_mask=valid_mask.contiguous(),
    )
    result.validate()
    return result


def _pair_energy(
    components: tuple[Tensor, Tensor],
    targets: CoverageStatePairTargets,
    *,
    config: CoverageStateSobolevConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    per_value_power = _per_state_vector_p4_power(
        components,
        targets.integration_measure,
    )
    per_spatial_power = _per_state_spatial_vector_p4_power(
        components,
        targets.valid_mask,
        targets.integration_measure,
    )
    per_loss = (
        0.5 * (per_value_power + per_spatial_power)
        + config.norm_epsilon**config.norm_order
    ).pow(1.0 / float(config.norm_order)) - config.norm_epsilon
    loss = per_loss.mean()
    if not bool(
        torch.stack(
            (
                torch.isfinite(loss),
                torch.isfinite(per_value_power).all(),
                torch.isfinite(per_spatial_power).all(),
                torch.isfinite(per_loss).all(),
            )
        ).all()
    ):
        raise FloatingPointError("coverage-state field energy is non-finite")
    return loss, per_loss, per_value_power, per_spatial_power


def coverage_state_pair_sobolev_loss_from_targets(
    field_plus: Tensor,
    field_minus: Tensor,
    targets: CoverageStatePairTargets,
    *,
    config: CoverageStateSobolevConfig,
) -> CoverageStatePairLossFields:
    """Evaluate the coupled finite-response coordinates without rebuilding targets."""

    if not isinstance(config, CoverageStateSobolevConfig):
        raise TypeError("config must be CoverageStateSobolevConfig")
    if not isinstance(targets, CoverageStatePairTargets):
        raise TypeError("targets must be CoverageStatePairTargets")
    targets.validate()
    if (
        tuple(field_plus.shape) != tuple(targets.target_field_plus.shape)
        or tuple(field_minus.shape) != tuple(field_plus.shape)
        or field_plus.device != targets.target_field_plus.device
        or field_minus.device != field_plus.device
        or field_plus.dtype != torch.float32
        or field_minus.dtype != torch.float32
        or not bool(torch.isfinite(field_plus).all())
        or not bool(torch.isfinite(field_minus).all())
    ):
        raise ValueError("pair fields and precomputed geometry must align")
    anchor_error = field_plus - targets.target_field_plus
    predicted_response = field_minus - field_plus
    target_response = (
        targets.target_field_minus - targets.target_field_plus
    )
    response_error = predicted_response - target_response
    loss, per_loss, per_value_power, per_spatial_power = _pair_energy(
        (anchor_error, response_error),
        targets,
        config=config,
    )
    return CoverageStatePairLossFields(
        loss=loss,
        value_power=per_value_power.mean(),
        spatial_power=per_spatial_power.mean(),
        per_state_loss=per_loss,
        per_state_value_power=per_value_power,
        per_state_spatial_power=per_spatial_power,
        target_field_plus=targets.target_field_plus,
        target_field_minus=targets.target_field_minus,
        predicted_coverage_response=predicted_response,
        target_coverage_response=target_response,
        anchor_error=anchor_error,
        response_error=response_error,
        focus_support=targets.focus_support,
        focus_support_field=targets.focus_support_field,
        integration_measure=targets.integration_measure,
    )


@dataclass(frozen=True)
class CoverageStateIndependentLossFields:
    """Decisive same-endpoint control without finite-response coordinates."""

    loss: Tensor
    value_power: Tensor
    spatial_power: Tensor
    per_state_loss: Tensor
    per_state_value_power: Tensor
    per_state_spatial_power: Tensor
    error_plus: Tensor
    error_minus: Tensor
    target_field_plus: Tensor
    target_field_minus: Tensor
    focus_support: Tensor
    integration_measure: Tensor


def coverage_state_independent_endpoint_loss_from_targets(
    field_plus: Tensor,
    field_minus: Tensor,
    targets: CoverageStatePairTargets,
    *,
    config: CoverageStateSobolevConfig,
) -> CoverageStateIndependentLossFields:
    """Use ``[e_plus,e_minus]`` with the exact coupled-method geometry."""

    if not isinstance(config, CoverageStateSobolevConfig):
        raise TypeError("config must be CoverageStateSobolevConfig")
    if not isinstance(targets, CoverageStatePairTargets):
        raise TypeError("targets must be CoverageStatePairTargets")
    targets.validate()
    if (
        tuple(field_plus.shape) != tuple(targets.target_field_plus.shape)
        or tuple(field_minus.shape) != tuple(field_plus.shape)
        or field_plus.device != targets.target_field_plus.device
        or field_minus.device != field_plus.device
        or field_plus.dtype != torch.float32
        or field_minus.dtype != torch.float32
        or not bool(torch.isfinite(field_plus).all())
        or not bool(torch.isfinite(field_minus).all())
    ):
        raise ValueError("independent fields and target geometry must align")
    error_plus = field_plus - targets.target_field_plus
    error_minus = field_minus - targets.target_field_minus
    loss, per_loss, per_value_power, per_spatial_power = _pair_energy(
        (error_plus, error_minus),
        targets,
        config=config,
    )
    return CoverageStateIndependentLossFields(
        loss=loss,
        value_power=per_value_power.mean(),
        spatial_power=per_spatial_power.mean(),
        per_state_loss=per_loss,
        per_state_value_power=per_value_power,
        per_state_spatial_power=per_spatial_power,
        error_plus=error_plus,
        error_minus=error_minus,
        target_field_plus=targets.target_field_plus,
        target_field_minus=targets.target_field_minus,
        focus_support=targets.focus_support,
        integration_measure=targets.integration_measure,
    )


def coverage_state_independent_endpoint_loss(
    field_plus: Tensor,
    field_minus: Tensor,
    occupancy_plus: Tensor,
    occupancy_minus: Tensor,
    target_plus: Tensor,
    target_minus: Tensor,
    valid_mask: Tensor,
    *,
    config: CoverageStateSobolevConfig,
) -> CoverageStateIndependentLossFields:
    """Build the matched independent-endpoint comparator."""

    targets = prepare_coverage_state_pair_targets(
        occupancy_plus,
        occupancy_minus,
        target_plus,
        target_minus,
        valid_mask,
        config=config,
    )
    return coverage_state_independent_endpoint_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=config,
    )


def coverage_state_pair_sobolev_loss(
    field_plus: Tensor,
    field_minus: Tensor,
    occupancy_plus: Tensor,
    occupancy_minus: Tensor,
    target_plus: Tensor,
    target_minus: Tensor,
    valid_mask: Tensor,
    *,
    config: CoverageStateSobolevConfig,
) -> CoverageStatePairLossFields:
    """Measure one same-source controlled deletion in a vector field norm.

    The value vector is ``[e_plus, delta_e]``, where ``delta_e`` is the
    finite-response error induced by the legal coverage deletion.  Its
    spatial gradient is measured with the same fixed integration measure.
    """

    targets = prepare_coverage_state_pair_targets(
        occupancy_plus,
        occupancy_minus,
        target_plus,
        target_minus,
        valid_mask,
        config=config,
    )
    return coverage_state_pair_sobolev_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=config,
    )


class CoverageStateSobolevLoss(nn.Module):
    """Module wrapper over natural-state and pair-edge CSLF risks."""

    def __init__(self, config: CoverageStateSobolevConfig) -> None:
        super().__init__()
        if not isinstance(config, CoverageStateSobolevConfig):
            raise TypeError("config must be CoverageStateSobolevConfig")
        self.config = config

    def forward(
        self,
        field_plus: Tensor,
        field_minus: Tensor,
        occupancy_plus: Tensor,
        occupancy_minus: Tensor,
        target_plus: Tensor,
        target_minus: Tensor,
        valid_mask: Tensor,
    ) -> CoverageStatePairLossFields:
        return coverage_state_pair_sobolev_loss(
            field_plus,
            field_minus,
            occupancy_plus,
            occupancy_minus,
            target_plus,
            target_minus,
            valid_mask,
            config=self.config,
        )

    def natural(
        self,
        field: Tensor,
        target: Tensor,
        valid_mask: Tensor,
    ) -> CoverageStateAbsoluteLossFields:
        return coverage_state_absolute_sobolev_loss(
            field,
            target,
            valid_mask,
            config=self.config,
        )

    def paired_from_targets(
        self,
        field_plus: Tensor,
        field_minus: Tensor,
        targets: CoverageStatePairTargets,
    ) -> CoverageStatePairLossFields:
        return coverage_state_pair_sobolev_loss_from_targets(
            field_plus,
            field_minus,
            targets,
            config=self.config,
        )

    def natural_from_targets(
        self,
        field: Tensor,
        targets: CoverageStateAbsoluteTargets,
    ) -> CoverageStateAbsoluteLossFields:
        return coverage_state_absolute_sobolev_loss_from_targets(
            field,
            targets,
            config=self.config,
        )


class CoverageStateIndependentEndpointLoss(nn.Module):
    """Matched control that changes only the pair error coordinates."""

    def __init__(self, config: CoverageStateSobolevConfig) -> None:
        super().__init__()
        if not isinstance(config, CoverageStateSobolevConfig):
            raise TypeError("config must be CoverageStateSobolevConfig")
        self.config = config

    def forward(
        self,
        field_plus: Tensor,
        field_minus: Tensor,
        targets: CoverageStatePairTargets,
    ) -> CoverageStateIndependentLossFields:
        return coverage_state_independent_endpoint_loss_from_targets(
            field_plus,
            field_minus,
            targets,
            config=self.config,
        )


__all__ = [
    "CSLF_MEASURE_POLICY",
    "CSLF_NORM_EPSILON",
    "CSLF_NORM_ORDER",
    "CSLF_OBJECTIVE_POLICY",
    "CoverageStateAbsoluteLossFields",
    "CoverageStateAbsoluteTargets",
    "CoverageStateIndependentEndpointLoss",
    "CoverageStateIndependentLossFields",
    "CoverageStatePairBatch",
    "CoverageStatePairLossFields",
    "CoverageStatePairTargets",
    "CoverageStateSobolevConfig",
    "CoverageStateSobolevLoss",
    "coverage_state_absolute_sobolev_loss",
    "coverage_state_absolute_sobolev_loss_from_targets",
    "coverage_state_independent_endpoint_loss",
    "coverage_state_independent_endpoint_loss_from_targets",
    "coverage_state_pair_sobolev_loss",
    "coverage_state_pair_sobolev_loss_from_targets",
    "prepare_coverage_state_absolute_targets",
    "prepare_coverage_state_pair_targets",
    "stack_coverage_state_pairs",
]
