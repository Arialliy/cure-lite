"""Fixed training batches assembled only from the scalar CSLF cache."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .cache.schema import stable_fingerprint
from .coverage_state_precomputed_cache import (
    CoverageStateCachedNatural,
    CoverageStateCachedPair,
)
from .coverage_state_sobolev import (
    CoverageStateAbsoluteTargets,
    CoverageStatePairTargets,
)


COVERAGE_STATE_FUSED_BATCH_SCHEMA = "cure-lite-cslf-fused-batch-v1"
COVERAGE_STATE_FUSED_NATURAL_COUNT = 4
COVERAGE_STATE_FUSED_PAIR_COUNT = 2
COVERAGE_STATE_FUSED_LOGICAL_STATES = 12


def _cat(values: tuple[Tensor, ...], *, name: str) -> Tensor:
    if not values:
        raise ValueError(f"{name} cannot be empty")
    if any(not isinstance(value, Tensor) for value in values):
        raise TypeError(f"{name} must contain tensors")
    return torch.cat(values, dim=0).contiguous()


def _stack_absolute_targets(
    values: tuple[CoverageStateAbsoluteTargets, ...],
    *,
    device: torch.device | str,
    validate: bool = True,
) -> CoverageStateAbsoluteTargets:
    if not values:
        raise ValueError("absolute targets cannot be empty")
    if not isinstance(validate, bool):
        raise TypeError("validate must be bool")
    if validate:
        for value in values:
            value.validate()
    result = CoverageStateAbsoluteTargets(
        target_field=_cat(
            tuple(value.target_field for value in values),
            name="target_field",
        ).to(device),
        integration_measure=_cat(
            tuple(value.integration_measure for value in values),
            name="integration_measure",
        ).to(device),
        field_valid_mask=_cat(
            tuple(value.field_valid_mask for value in values),
            name="field_valid_mask",
        ).to(device),
        loss_valid_mask=_cat(
            tuple(value.loss_valid_mask for value in values),
            name="loss_valid_mask",
        ).to(device),
        focus_support=_cat(
            tuple(value.focus_support for value in values),
            name="focus_support",
        ).to(device),
        focus_support_field=_cat(
            tuple(value.focus_support_field for value in values),
            name="focus_support_field",
        ).to(device),
    )
    if validate:
        result.validate()
    return result


def _stack_pair_targets(
    values: tuple[CoverageStatePairTargets, ...],
    *,
    device: torch.device | str,
    validate: bool = True,
) -> CoverageStatePairTargets:
    if not values:
        raise ValueError("pair targets cannot be empty")
    if not isinstance(validate, bool):
        raise TypeError("validate must be bool")
    if validate:
        for value in values:
            value.validate()
    result = CoverageStatePairTargets(
        target_field_plus=_cat(
            tuple(value.target_field_plus for value in values),
            name="target_field_plus",
        ).to(device),
        target_field_minus=_cat(
            tuple(value.target_field_minus for value in values),
            name="target_field_minus",
        ).to(device),
        focus_support=_cat(
            tuple(value.focus_support for value in values),
            name="pair_focus_support",
        ).to(device),
        focus_support_field=_cat(
            tuple(value.focus_support_field for value in values),
            name="pair_focus_support_field",
        ).to(device),
        integration_measure=_cat(
            tuple(value.integration_measure for value in values),
            name="pair_integration_measure",
        ).to(device),
        valid_mask=_cat(
            tuple(value.valid_mask for value in values),
            name="pair_valid_mask",
        ).to(device),
    )
    if validate:
        result.validate()
    return result


def _validate_feature_occupancy(
    feature: Tensor,
    occupancy: Tensor,
    *,
    expected_batch: int,
) -> None:
    if (
        not isinstance(feature, Tensor)
        or feature.dtype != torch.float32
        or feature.ndim != 4
        or feature.shape[0] != expected_batch
        or feature.shape[1] < 1
        or feature.requires_grad
        or not bool(torch.isfinite(feature).all())
    ):
        raise ValueError("feature must be detached finite FP32 [B,C,h,w]")
    if (
        not isinstance(occupancy, Tensor)
        or occupancy.dtype != torch.bool
        or occupancy.ndim != 4
        or tuple(occupancy.shape[:2]) != (expected_batch, 1)
        or occupancy.device != feature.device
    ):
        raise ValueError("occupancy must be aligned bool [B,1,H,W]")


@dataclass(frozen=True)
class CoverageStateNaturalTrainBatch:
    """Four natural states sharing one state kind and precomputed geometry."""

    feature: Tensor
    occupancy: Tensor
    targets: CoverageStateAbsoluteTargets
    record_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    actual_input_fingerprints: tuple[str, ...]
    state_kind: str

    @property
    def batch_size(self) -> int:
        return int(self.feature.shape[0])

    def validate(self, *, expected_count: int | None = None) -> None:
        if self.state_kind not in {"factual_miss", "factual_no_miss"}:
            raise ValueError("natural batch has an unknown state kind")
        batch = self.batch_size
        if expected_count is not None and batch != expected_count:
            raise ValueError("natural batch has an unexpected state count")
        _validate_feature_occupancy(
            self.feature,
            self.occupancy,
            expected_batch=batch,
        )
        self.targets.validate()
        if (
            tuple(self.targets.target_field.shape)
            != tuple(self.occupancy.shape)
            or self.targets.target_field.device != self.feature.device
        ):
            raise ValueError("natural geometry and inputs differ")
        metadata = (
            self.record_ids,
            self.sample_ids,
            self.actual_input_fingerprints,
        )
        if any(
            not isinstance(value, tuple) or len(value) != batch
            for value in metadata
        ):
            raise ValueError("natural metadata does not align with batch")
        if (
            self.record_ids != tuple(dict.fromkeys(self.record_ids))
            or len(set(self.actual_input_fingerprints)) != batch
        ):
            raise ValueError(
                "natural batch must not repeat records or actual inputs"
            )


@dataclass(frozen=True)
class CoverageStatePairTrainBatch:
    """One clean and one visible component-null pair."""

    feature: Tensor
    occupancy_plus: Tensor
    occupancy_minus: Tensor
    joint_targets: CoverageStatePairTargets
    absolute_targets_plus: CoverageStateAbsoluteTargets
    absolute_targets_minus: CoverageStateAbsoluteTargets
    pair_ids: tuple[str, ...]
    pair_kinds: tuple[str, ...]
    sample_ids: tuple[str, ...]
    actual_input_plus_fingerprints: tuple[str, ...]
    actual_input_minus_fingerprints: tuple[str, ...]

    @property
    def batch_size(self) -> int:
        return int(self.feature.shape[0])

    def validate(self, *, expected_count: int | None = None) -> None:
        batch = self.batch_size
        if expected_count is not None and batch != expected_count:
            raise ValueError("pair batch has an unexpected pair count")
        _validate_feature_occupancy(
            self.feature,
            self.occupancy_plus,
            expected_batch=batch,
        )
        _validate_feature_occupancy(
            self.feature,
            self.occupancy_minus,
            expected_batch=batch,
        )
        if bool(torch.any(self.occupancy_minus & ~self.occupancy_plus)):
            raise ValueError("pair minus occupancy is not a subset")
        for targets in (
            self.joint_targets,
            self.absolute_targets_plus,
            self.absolute_targets_minus,
        ):
            targets.validate()
            reference = (
                targets.target_field_plus
                if isinstance(targets, CoverageStatePairTargets)
                else targets.target_field
            )
            if (
                tuple(reference.shape) != tuple(self.occupancy_plus.shape)
                or reference.device != self.feature.device
            ):
                raise ValueError("pair target geometry and inputs differ")
        metadata = (
            self.pair_ids,
            self.pair_kinds,
            self.sample_ids,
            self.actual_input_plus_fingerprints,
            self.actual_input_minus_fingerprints,
        )
        if any(
            not isinstance(value, tuple) or len(value) != batch
            for value in metadata
        ):
            raise ValueError("pair metadata does not align with batch")
        if self.pair_ids != tuple(dict.fromkeys(self.pair_ids)):
            raise ValueError("pair batch repeats an identity")
        if self.pair_kinds != ("clean_positive", "component_null"):
            raise ValueError(
                "pair batch must be ordered clean-positive then component-null"
            )
        if len(set(self.sample_ids)) != batch:
            raise ValueError(
                "clean and component-null pairs must use different sources"
            )
        if any(
            plus == minus
            for plus, minus in zip(
                self.actual_input_plus_fingerprints,
                self.actual_input_minus_fingerprints,
                strict=True,
            )
        ):
            raise ValueError(
                "every optimizer pair must be scalar-visible"
            )


@dataclass(frozen=True)
class CoverageStateFusedBatch:
    """The fixed 4 + 4 + 2 plus + 2 minus one-forward budget."""

    factual_miss: CoverageStateNaturalTrainBatch
    factual_no_miss: CoverageStateNaturalTrainBatch
    pairs: CoverageStatePairTrainBatch

    @property
    def selection_fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "schema_version": COVERAGE_STATE_FUSED_BATCH_SCHEMA,
                "factual_miss_record_ids": list(
                    self.factual_miss.record_ids
                ),
                "factual_no_miss_record_ids": list(
                    self.factual_no_miss.record_ids
                ),
                "pair_ids": list(self.pairs.pair_ids),
                "pair_kinds": list(self.pairs.pair_kinds),
                "input_order": [
                    "factual_miss",
                    "factual_no_miss",
                    "pair_plus",
                    "pair_minus",
                ],
            }
        )

    def validate(self) -> None:
        if not isinstance(
            self.factual_miss,
            CoverageStateNaturalTrainBatch,
        ) or not isinstance(
            self.factual_no_miss,
            CoverageStateNaturalTrainBatch,
        ) or not isinstance(self.pairs, CoverageStatePairTrainBatch):
            raise TypeError("fused batch contains an invalid sub-batch")
        self.factual_miss.validate(
            expected_count=COVERAGE_STATE_FUSED_NATURAL_COUNT
        )
        self.factual_no_miss.validate(
            expected_count=COVERAGE_STATE_FUSED_NATURAL_COUNT
        )
        self.pairs.validate(expected_count=COVERAGE_STATE_FUSED_PAIR_COUNT)
        if self.factual_miss.state_kind != "factual_miss":
            raise ValueError("first natural branch must contain factual misses")
        if self.factual_no_miss.state_kind != "factual_no_miss":
            raise ValueError(
                "second natural branch must contain factual no-miss states"
            )
        feature_shapes = {
            tuple(value.feature.shape[1:])
            for value in (
                self.factual_miss,
                self.factual_no_miss,
                self.pairs,
            )
        }
        output_shapes = {
            tuple(value.occupancy.shape[1:])
            for value in (
                self.factual_miss,
                self.factual_no_miss,
            )
        }
        output_shapes.add(tuple(self.pairs.occupancy_plus.shape[1:]))
        devices = {
            value.feature.device
            for value in (
                self.factual_miss,
                self.factual_no_miss,
                self.pairs,
            )
        }
        if len(feature_shapes) != 1 or len(output_shapes) != 1 or len(devices) != 1:
            raise ValueError("all fused branches must share shape and device")

    def model_inputs(self) -> tuple[Tensor, Tensor]:
        """Return the exact fixed-order 12-state model input."""

        self.validate()
        feature = torch.cat(
            (
                self.factual_miss.feature,
                self.factual_no_miss.feature,
                self.pairs.feature,
                self.pairs.feature,
            ),
            dim=0,
        ).contiguous()
        occupancy = torch.cat(
            (
                self.factual_miss.occupancy,
                self.factual_no_miss.occupancy,
                self.pairs.occupancy_plus,
                self.pairs.occupancy_minus,
            ),
            dim=0,
        ).contiguous()
        if feature.shape[0] != COVERAGE_STATE_FUSED_LOGICAL_STATES:
            raise AssertionError("fused model input is not exactly 12 states")
        return feature, occupancy


def make_coverage_state_natural_train_batch(
    values: tuple[CoverageStateCachedNatural, ...],
    *,
    state_kind: str,
    device: torch.device | str,
    validate: bool = True,
) -> CoverageStateNaturalTrainBatch:
    if not isinstance(validate, bool):
        raise TypeError("validate must be bool")
    if not values or any(
        not isinstance(value, CoverageStateCachedNatural)
        for value in values
    ):
        raise TypeError("values must contain cached natural records")
    if any(value.record.state_kind != state_kind for value in values):
        raise ValueError("cached natural state kind differs from request")
    result = CoverageStateNaturalTrainBatch(
        feature=_cat(
            tuple(value.record.feature for value in values),
            name="natural_feature",
        ).to(device),
        occupancy=_cat(
            tuple(value.record.occupancy for value in values),
            name="natural_occupancy",
        ).to(device),
        targets=_stack_absolute_targets(
            tuple(value.targets for value in values),
            device=device,
            validate=validate,
        ),
        record_ids=tuple(value.record.record_id for value in values),
        sample_ids=tuple(value.record.sample_id for value in values),
        actual_input_fingerprints=tuple(
            value.actual_scalar_input_fingerprint for value in values
        ),
        state_kind=state_kind,
    )
    if validate:
        result.validate()
    return result


def make_coverage_state_pair_train_batch(
    clean_positive: CoverageStateCachedPair,
    component_null: CoverageStateCachedPair,
    *,
    device: torch.device | str,
    validate: bool = True,
) -> CoverageStatePairTrainBatch:
    if not isinstance(validate, bool):
        raise TypeError("validate must be bool")
    values = (clean_positive, component_null)
    if any(not isinstance(value, CoverageStateCachedPair) for value in values):
        raise TypeError("pair selections must be cached pairs")
    if tuple(value.optimizer_role for value in values) != (
        "clean_positive",
        "component_null",
    ):
        raise ValueError(
            "optimizer pairs must be visible clean and component-null states"
        )
    result = CoverageStatePairTrainBatch(
        feature=_cat(
            tuple(value.record.feature for value in values),
            name="pair_feature",
        ).to(device),
        occupancy_plus=_cat(
            tuple(value.record.occupancy_plus for value in values),
            name="occupancy_plus",
        ).to(device),
        occupancy_minus=_cat(
            tuple(value.record.occupancy_minus for value in values),
            name="occupancy_minus",
        ).to(device),
        joint_targets=_stack_pair_targets(
            tuple(value.joint_targets for value in values),
            device=device,
            validate=validate,
        ),
        absolute_targets_plus=_stack_absolute_targets(
            tuple(value.absolute_targets_plus for value in values),
            device=device,
            validate=validate,
        ),
        absolute_targets_minus=_stack_absolute_targets(
            tuple(value.absolute_targets_minus for value in values),
            device=device,
            validate=validate,
        ),
        pair_ids=tuple(value.record.pair_id for value in values),
        pair_kinds=tuple(value.record.pair_kind for value in values),
        sample_ids=tuple(value.record.sample_id for value in values),
        actual_input_plus_fingerprints=tuple(
            value.actual_input_plus_fingerprint for value in values
        ),
        actual_input_minus_fingerprints=tuple(
            value.actual_input_minus_fingerprint for value in values
        ),
    )
    if validate:
        result.validate()
    return result


__all__ = [
    "COVERAGE_STATE_FUSED_BATCH_SCHEMA",
    "COVERAGE_STATE_FUSED_LOGICAL_STATES",
    "COVERAGE_STATE_FUSED_NATURAL_COUNT",
    "COVERAGE_STATE_FUSED_PAIR_COUNT",
    "CoverageStateFusedBatch",
    "CoverageStateNaturalTrainBatch",
    "CoverageStatePairTrainBatch",
    "make_coverage_state_natural_train_batch",
    "make_coverage_state_pair_train_batch",
]
