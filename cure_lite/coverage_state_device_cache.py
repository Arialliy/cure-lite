"""Device-resident packed tensors for scheduled scalar CSLF updates.

The scalar cache is the authoritative, verified CPU representation.  This
module performs one construction-time transfer of its training-eligible
payload and thereafter materializes scheduled batches only by device-local
row gathers.  Identity-null and diagnostic-only pairs are deliberately not
packed.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor

from .cache.schema import stable_fingerprint
from .coverage_state_batches import (
    CoverageStateFusedBatch,
    CoverageStateNaturalTrainBatch,
    CoverageStatePairTrainBatch,
)
from .coverage_state_precomputed_cache import (
    CoverageStateCachedNatural,
    CoverageStateCachedPair,
    CoverageStateScalarCache,
)
from .coverage_state_schedule import CoverageStateUpdateSelection
from .coverage_state_sobolev import (
    CoverageStateAbsoluteTargets,
    CoverageStatePairTargets,
)
from .paired_types import tensor_content_fingerprint


COVERAGE_STATE_DEVICE_CACHE_SCHEMA = (
    "cure-lite-scalar-coverage-state-device-cache-v1"
)
COVERAGE_STATE_DEVICE_CACHE_PAIR_ROLES = (
    "clean_positive",
    "component_null",
)


def _canonical_device(value: torch.device | str) -> torch.device:
    if not isinstance(value, (torch.device, str)):
        raise TypeError("device must be a torch.device or string")
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda", torch.cuda.current_device())
    return device


def _pack(
    values: tuple[Tensor, ...],
    *,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
    content_fingerprints: dict[str, str],
) -> Tensor:
    """Concatenate on CPU and transfer the packed payload exactly once."""

    if not values or any(not isinstance(value, Tensor) for value in values):
        raise TypeError(f"{name} must contain tensors")
    if any(
        value.device.type != "cpu"
        or value.dtype != dtype
        or value.requires_grad
        for value in values
    ):
        raise ValueError(
            f"{name} source tensors must be detached CPU {dtype}"
        )
    packed_cpu = torch.cat(values, dim=0).detach().contiguous()
    if name in content_fingerprints:
        raise ValueError(f"duplicate packed tensor name {name!r}")
    content_fingerprints[name] = tensor_content_fingerprint(packed_cpu)
    packed = packed_cpu.to(
        device=device,
        dtype=dtype,
        non_blocking=False,
        copy=device.type != "cpu",
    )
    if (
        packed.device != device
        or packed.dtype != dtype
        or packed.requires_grad
    ):
        raise RuntimeError(f"{name} transfer changed its binding")
    return packed.contiguous()


def _pack_row_index(
    count: int,
    *,
    name: str,
    device: torch.device,
    content_fingerprints: dict[str, str],
) -> Tensor:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError(f"{name} count must be a positive integer")
    rows_cpu = torch.arange(count, dtype=torch.long).contiguous()
    if name in content_fingerprints:
        raise ValueError(f"duplicate packed tensor name {name!r}")
    content_fingerprints[name] = tensor_content_fingerprint(rows_cpu)
    rows = rows_cpu.to(
        device=device,
        dtype=torch.long,
        non_blocking=False,
        copy=device.type != "cpu",
    )
    return rows.contiguous()


def _tensor_layout_payload(value: Tensor) -> dict[str, object]:
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "contiguous": value.is_contiguous(),
        "requires_grad": value.requires_grad,
        "nbytes": value.numel() * value.element_size(),
    }


@dataclass(frozen=True, eq=False)
class CoverageStateDeviceNaturalStore:
    """Dense device tensors for every natural scalar-cache record."""

    feature: Tensor
    occupancy: Tensor
    target_field: Tensor
    integration_measure: Tensor
    field_valid_mask: Tensor
    loss_valid_mask: Tensor
    focus_support: Tensor
    focus_support_field: Tensor
    record_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    actual_input_fingerprints: tuple[str, ...]
    state_kinds: tuple[str, ...]

    def named_tensors(self) -> tuple[tuple[str, Tensor], ...]:
        return (
            ("natural/feature", self.feature),
            ("natural/occupancy", self.occupancy),
            ("natural/target_field", self.target_field),
            (
                "natural/integration_measure",
                self.integration_measure,
            ),
            ("natural/field_valid_mask", self.field_valid_mask),
            ("natural/loss_valid_mask", self.loss_valid_mask),
            ("natural/focus_support", self.focus_support),
            (
                "natural/focus_support_field",
                self.focus_support_field,
            ),
        )

    def validate(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        count = len(self.record_ids)
        metadata = (
            self.record_ids,
            self.sample_ids,
            self.actual_input_fingerprints,
            self.state_kinds,
        )
        if (
            count < 1
            or any(len(value) != count for value in metadata)
            or len(set(self.record_ids)) != count
            or any(
                kind not in {"factual_miss", "factual_no_miss"}
                for kind in self.state_kinds
            )
        ):
            raise ValueError("natural packed metadata is invalid")
        tensors = self.named_tensors()
        if any(
            value.ndim != 4
            or value.shape[0] != count
            or value.device != device
            or not value.is_contiguous()
            or value.requires_grad
            for _, value in tensors
        ):
            raise ValueError("natural packed tensor layout is invalid")
        floating = (
            self.feature,
            self.target_field,
            self.integration_measure,
            self.focus_support_field,
        )
        binary = (
            self.occupancy,
            self.field_valid_mask,
            self.loss_valid_mask,
            self.focus_support,
        )
        if any(value.dtype != dtype for value in floating) or any(
            value.dtype != torch.bool for value in binary
        ):
            raise ValueError("natural packed tensor dtype is invalid")
        if (
            self.feature.shape[1] < 1
            or self.occupancy.shape[1] != 1
            or len(
                {
                    tuple(value.shape)
                    for value in (
                        self.occupancy,
                        self.target_field,
                        self.integration_measure,
                        self.field_valid_mask,
                        self.loss_valid_mask,
                        self.focus_support,
                        self.focus_support_field,
                    )
                }
            )
            != 1
        ):
            raise ValueError("natural packed grids differ")


@dataclass(frozen=True, eq=False)
class CoverageStateDevicePairStore:
    """Dense device tensors for eligible clean and component-null pairs."""

    feature: Tensor
    occupancy_plus: Tensor
    occupancy_minus: Tensor
    joint_target_field_plus: Tensor
    joint_target_field_minus: Tensor
    joint_focus_support: Tensor
    joint_focus_support_field: Tensor
    joint_integration_measure: Tensor
    joint_valid_mask: Tensor
    absolute_plus_target_field: Tensor
    absolute_plus_integration_measure: Tensor
    absolute_plus_field_valid_mask: Tensor
    absolute_plus_loss_valid_mask: Tensor
    absolute_plus_focus_support: Tensor
    absolute_plus_focus_support_field: Tensor
    absolute_minus_target_field: Tensor
    absolute_minus_integration_measure: Tensor
    absolute_minus_field_valid_mask: Tensor
    absolute_minus_loss_valid_mask: Tensor
    absolute_minus_focus_support: Tensor
    absolute_minus_focus_support_field: Tensor
    pair_ids: tuple[str, ...]
    pair_kinds: tuple[str, ...]
    sample_ids: tuple[str, ...]
    actual_input_plus_fingerprints: tuple[str, ...]
    actual_input_minus_fingerprints: tuple[str, ...]
    optimizer_roles: tuple[str, ...]

    def named_tensors(self) -> tuple[tuple[str, Tensor], ...]:
        return (
            ("pair/feature", self.feature),
            ("pair/occupancy_plus", self.occupancy_plus),
            ("pair/occupancy_minus", self.occupancy_minus),
            (
                "pair/joint_target_field_plus",
                self.joint_target_field_plus,
            ),
            (
                "pair/joint_target_field_minus",
                self.joint_target_field_minus,
            ),
            ("pair/joint_focus_support", self.joint_focus_support),
            (
                "pair/joint_focus_support_field",
                self.joint_focus_support_field,
            ),
            (
                "pair/joint_integration_measure",
                self.joint_integration_measure,
            ),
            ("pair/joint_valid_mask", self.joint_valid_mask),
            (
                "pair/absolute_plus_target_field",
                self.absolute_plus_target_field,
            ),
            (
                "pair/absolute_plus_integration_measure",
                self.absolute_plus_integration_measure,
            ),
            (
                "pair/absolute_plus_field_valid_mask",
                self.absolute_plus_field_valid_mask,
            ),
            (
                "pair/absolute_plus_loss_valid_mask",
                self.absolute_plus_loss_valid_mask,
            ),
            (
                "pair/absolute_plus_focus_support",
                self.absolute_plus_focus_support,
            ),
            (
                "pair/absolute_plus_focus_support_field",
                self.absolute_plus_focus_support_field,
            ),
            (
                "pair/absolute_minus_target_field",
                self.absolute_minus_target_field,
            ),
            (
                "pair/absolute_minus_integration_measure",
                self.absolute_minus_integration_measure,
            ),
            (
                "pair/absolute_minus_field_valid_mask",
                self.absolute_minus_field_valid_mask,
            ),
            (
                "pair/absolute_minus_loss_valid_mask",
                self.absolute_minus_loss_valid_mask,
            ),
            (
                "pair/absolute_minus_focus_support",
                self.absolute_minus_focus_support,
            ),
            (
                "pair/absolute_minus_focus_support_field",
                self.absolute_minus_focus_support_field,
            ),
        )

    def validate(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        count = len(self.pair_ids)
        metadata = (
            self.pair_ids,
            self.pair_kinds,
            self.sample_ids,
            self.actual_input_plus_fingerprints,
            self.actual_input_minus_fingerprints,
            self.optimizer_roles,
        )
        if (
            count < 2
            or any(len(value) != count for value in metadata)
            or len(set(self.pair_ids)) != count
            or self.pair_kinds != self.optimizer_roles
            or set(self.optimizer_roles)
            != set(COVERAGE_STATE_DEVICE_CACHE_PAIR_ROLES)
            or any(
                role not in COVERAGE_STATE_DEVICE_CACHE_PAIR_ROLES
                for role in self.optimizer_roles
            )
        ):
            raise ValueError("pair packed metadata is invalid")
        tensors = self.named_tensors()
        if any(
            value.ndim != 4
            or value.shape[0] != count
            or value.device != device
            or not value.is_contiguous()
            or value.requires_grad
            for _, value in tensors
        ):
            raise ValueError("pair packed tensor layout is invalid")
        floating = (
            self.feature,
            self.joint_target_field_plus,
            self.joint_target_field_minus,
            self.joint_focus_support_field,
            self.joint_integration_measure,
            self.absolute_plus_target_field,
            self.absolute_plus_integration_measure,
            self.absolute_plus_focus_support_field,
            self.absolute_minus_target_field,
            self.absolute_minus_integration_measure,
            self.absolute_minus_focus_support_field,
        )
        binary = (
            self.occupancy_plus,
            self.occupancy_minus,
            self.joint_focus_support,
            self.joint_valid_mask,
            self.absolute_plus_field_valid_mask,
            self.absolute_plus_loss_valid_mask,
            self.absolute_plus_focus_support,
            self.absolute_minus_field_valid_mask,
            self.absolute_minus_loss_valid_mask,
            self.absolute_minus_focus_support,
        )
        if any(value.dtype != dtype for value in floating) or any(
            value.dtype != torch.bool for value in binary
        ):
            raise ValueError("pair packed tensor dtype is invalid")
        output_tensors = tuple(
            value for _, value in tensors if value is not self.feature
        )
        if (
            self.feature.shape[1] < 1
            or any(value.shape[1] != 1 for value in output_tensors)
            or len({tuple(value.shape) for value in output_tensors}) != 1
        ):
            raise ValueError("pair packed grids differ")


def _absolute_targets_from_store(
    *,
    target_field: Tensor,
    integration_measure: Tensor,
    field_valid_mask: Tensor,
    loss_valid_mask: Tensor,
    focus_support: Tensor,
    focus_support_field: Tensor,
) -> CoverageStateAbsoluteTargets:
    return CoverageStateAbsoluteTargets(
        target_field=target_field,
        integration_measure=integration_measure,
        field_valid_mask=field_valid_mask,
        loss_valid_mask=loss_valid_mask,
        focus_support=focus_support,
        focus_support_field=focus_support_field,
    )


@dataclass(frozen=True, eq=False)
class CoverageStateDeviceCache:
    """A cache-bound, device-resident view of all optimizer-eligible states."""

    source_cache: CoverageStateScalarCache
    natural: CoverageStateDeviceNaturalStore
    pairs: CoverageStateDevicePairStore
    natural_id_to_index: Mapping[str, int]
    pair_id_to_index: Mapping[str, int]
    natural_row_index: Tensor
    pair_row_index: Tensor
    source_cache_fingerprint: str
    device: torch.device
    dtype: torch.dtype
    device_cache_fingerprint: str
    _tensor_versions: tuple[tuple[str, int], ...]
    _tensor_content_fingerprints: tuple[tuple[str, str], ...]

    def named_tensors(self) -> tuple[tuple[str, Tensor], ...]:
        return (
            *self.natural.named_tensors(),
            *self.pairs.named_tensors(),
            ("index/natural_rows", self.natural_row_index),
            ("index/pair_rows", self.pair_row_index),
        )

    @property
    def resident_tensor_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for _, value in self.named_tensors()
        )

    def memory_report(self) -> dict[str, object]:
        by_dtype: dict[str, int] = {}
        by_store = {"natural": 0, "pair": 0, "index": 0}
        for name, value in self.named_tensors():
            nbytes = value.numel() * value.element_size()
            dtype = str(value.dtype)
            by_dtype[dtype] = by_dtype.get(dtype, 0) + nbytes
            store = name.split("/", maxsplit=1)[0]
            by_store[store] += nbytes
        total = sum(by_store.values())
        return {
            "accounting": "resident_packed_tensor_payload_only",
            "tensor_count": len(self.named_tensors()),
            "resident_tensor_bytes": total,
            "resident_tensor_mib": total / float(1024**2),
            "by_dtype_bytes": dict(sorted(by_dtype.items())),
            "by_store_bytes": by_store,
            "natural_record_count": len(self.natural.record_ids),
            "eligible_pair_count": len(self.pairs.pair_ids),
            "retained_source_cache_not_counted": True,
        }

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": COVERAGE_STATE_DEVICE_CACHE_SCHEMA,
            "source_cache_fingerprint": self.source_cache_fingerprint,
            "device": str(self.device),
            "floating_dtype": str(self.dtype),
            "pair_roles": list(COVERAGE_STATE_DEVICE_CACHE_PAIR_ROLES),
            "natural": {
                "record_ids": list(self.natural.record_ids),
                "sample_ids": list(self.natural.sample_ids),
                "actual_input_fingerprints": list(
                    self.natural.actual_input_fingerprints
                ),
                "state_kinds": list(self.natural.state_kinds),
            },
            "pairs": {
                "pair_ids": list(self.pairs.pair_ids),
                "pair_kinds": list(self.pairs.pair_kinds),
                "sample_ids": list(self.pairs.sample_ids),
                "actual_input_plus_fingerprints": list(
                    self.pairs.actual_input_plus_fingerprints
                ),
                "actual_input_minus_fingerprints": list(
                    self.pairs.actual_input_minus_fingerprints
                ),
                "optimizer_roles": list(self.pairs.optimizer_roles),
            },
            "tensor_layout": {
                name: _tensor_layout_payload(value)
                for name, value in self.named_tensors()
            },
            "tensor_content_fingerprints": dict(
                self._tensor_content_fingerprints
            ),
            "memory": self.memory_report(),
            "identity_null_packed": False,
            "diagnostic_only_packed": False,
        }

    def verify_unchanged(
        self,
        *,
        verify_content: bool = True,
        verify_source: bool = True,
    ) -> None:
        """Verify source, layout, indexes, versions, and optionally all bytes.

        Exact content verification may copy resident tensors to CPU.  It is a
        preflight operation, not an update-loop operation.
        """

        if not isinstance(verify_content, bool) or not isinstance(
            verify_source,
            bool,
        ):
            raise TypeError(
                "verify_content and verify_source must be bool"
            )
        if verify_source:
            self.source_cache.verify_unchanged()
        if (
            self.source_cache.cache_fingerprint
            != self.source_cache_fingerprint
            or self.dtype != torch.float32
        ):
            raise RuntimeError("device cache source or dtype binding changed")
        self.natural.validate(device=self.device, dtype=self.dtype)
        self.pairs.validate(device=self.device, dtype=self.dtype)
        expected_natural_index = {
            identity: index
            for index, identity in enumerate(self.natural.record_ids)
        }
        expected_pair_index = {
            identity: index
            for index, identity in enumerate(self.pairs.pair_ids)
        }
        if (
            dict(self.natural_id_to_index) != expected_natural_index
            or dict(self.pair_id_to_index) != expected_pair_index
        ):
            raise RuntimeError("device cache ID index changed")
        if (
            self.natural_row_index.dtype != torch.long
            or self.pair_row_index.dtype != torch.long
            or self.natural_row_index.device != self.device
            or self.pair_row_index.device != self.device
            or not self.natural_row_index.is_contiguous()
            or not self.pair_row_index.is_contiguous()
            or tuple(self.natural_row_index.shape)
            != (len(self.natural.record_ids),)
            or tuple(self.pair_row_index.shape)
            != (len(self.pairs.pair_ids),)
        ):
            raise RuntimeError("device cache row-index binding changed")
        current_versions = tuple(
            (name, int(value._version))
            for name, value in self.named_tensors()
        )
        if current_versions != self._tensor_versions:
            raise RuntimeError("device cache packed tensor changed")
        if verify_content:
            current_content = tuple(
                (name, tensor_content_fingerprint(value))
                for name, value in self.named_tensors()
            )
            if current_content != self._tensor_content_fingerprints:
                raise RuntimeError(
                    "device cache packed tensor content changed"
                )
        if (
            stable_fingerprint(self.canonical_payload())
            != self.device_cache_fingerprint
        ):
            raise RuntimeError("device cache metadata binding changed")

    def materialize(
        self,
        selection: CoverageStateUpdateSelection,
        *,
        verify: bool = False,
        validate: bool = True,
    ) -> CoverageStateFusedBatch:
        return materialize_coverage_state_device_fused_batch(
            self,
            selection,
            verify=verify,
            validate=validate,
        )


def _pack_naturals(
    values: tuple[CoverageStateCachedNatural, ...],
    *,
    device: torch.device,
    content_fingerprints: dict[str, str],
) -> CoverageStateDeviceNaturalStore:
    return CoverageStateDeviceNaturalStore(
        feature=_pack(
            tuple(value.record.feature for value in values),
            name="natural/feature",
            device=device,
            dtype=torch.float32,
            content_fingerprints=content_fingerprints,
        ),
        occupancy=_pack(
            tuple(value.record.occupancy for value in values),
            name="natural/occupancy",
            device=device,
            dtype=torch.bool,
            content_fingerprints=content_fingerprints,
        ),
        target_field=_pack(
            tuple(value.targets.target_field for value in values),
            name="natural/target_field",
            device=device,
            dtype=torch.float32,
            content_fingerprints=content_fingerprints,
        ),
        integration_measure=_pack(
            tuple(value.targets.integration_measure for value in values),
            name="natural/integration_measure",
            device=device,
            dtype=torch.float32,
            content_fingerprints=content_fingerprints,
        ),
        field_valid_mask=_pack(
            tuple(value.targets.field_valid_mask for value in values),
            name="natural/field_valid_mask",
            device=device,
            dtype=torch.bool,
            content_fingerprints=content_fingerprints,
        ),
        loss_valid_mask=_pack(
            tuple(value.targets.loss_valid_mask for value in values),
            name="natural/loss_valid_mask",
            device=device,
            dtype=torch.bool,
            content_fingerprints=content_fingerprints,
        ),
        focus_support=_pack(
            tuple(value.targets.focus_support for value in values),
            name="natural/focus_support",
            device=device,
            dtype=torch.bool,
            content_fingerprints=content_fingerprints,
        ),
        focus_support_field=_pack(
            tuple(value.targets.focus_support_field for value in values),
            name="natural/focus_support_field",
            device=device,
            dtype=torch.float32,
            content_fingerprints=content_fingerprints,
        ),
        record_ids=tuple(value.record.record_id for value in values),
        sample_ids=tuple(value.record.sample_id for value in values),
        actual_input_fingerprints=tuple(
            value.actual_scalar_input_fingerprint for value in values
        ),
        state_kinds=tuple(value.record.state_kind for value in values),
    )


def _pack_pairs(
    values: tuple[CoverageStateCachedPair, ...],
    *,
    device: torch.device,
    content_fingerprints: dict[str, str],
) -> CoverageStateDevicePairStore:
    float_fields = {
        "feature": tuple(value.record.feature for value in values),
        "joint_target_field_plus": tuple(
            value.joint_targets.target_field_plus for value in values
        ),
        "joint_target_field_minus": tuple(
            value.joint_targets.target_field_minus for value in values
        ),
        "joint_focus_support_field": tuple(
            value.joint_targets.focus_support_field for value in values
        ),
        "joint_integration_measure": tuple(
            value.joint_targets.integration_measure for value in values
        ),
        "absolute_plus_target_field": tuple(
            value.absolute_targets_plus.target_field for value in values
        ),
        "absolute_plus_integration_measure": tuple(
            value.absolute_targets_plus.integration_measure
            for value in values
        ),
        "absolute_plus_focus_support_field": tuple(
            value.absolute_targets_plus.focus_support_field
            for value in values
        ),
        "absolute_minus_target_field": tuple(
            value.absolute_targets_minus.target_field for value in values
        ),
        "absolute_minus_integration_measure": tuple(
            value.absolute_targets_minus.integration_measure
            for value in values
        ),
        "absolute_minus_focus_support_field": tuple(
            value.absolute_targets_minus.focus_support_field
            for value in values
        ),
    }
    bool_fields = {
        "occupancy_plus": tuple(
            value.record.occupancy_plus for value in values
        ),
        "occupancy_minus": tuple(
            value.record.occupancy_minus for value in values
        ),
        "joint_focus_support": tuple(
            value.joint_targets.focus_support for value in values
        ),
        "joint_valid_mask": tuple(
            value.joint_targets.valid_mask for value in values
        ),
        "absolute_plus_field_valid_mask": tuple(
            value.absolute_targets_plus.field_valid_mask
            for value in values
        ),
        "absolute_plus_loss_valid_mask": tuple(
            value.absolute_targets_plus.loss_valid_mask
            for value in values
        ),
        "absolute_plus_focus_support": tuple(
            value.absolute_targets_plus.focus_support for value in values
        ),
        "absolute_minus_field_valid_mask": tuple(
            value.absolute_targets_minus.field_valid_mask
            for value in values
        ),
        "absolute_minus_loss_valid_mask": tuple(
            value.absolute_targets_minus.loss_valid_mask
            for value in values
        ),
        "absolute_minus_focus_support": tuple(
            value.absolute_targets_minus.focus_support for value in values
        ),
    }
    packed = {
        name: _pack(
            tensors,
            name=f"pair/{name}",
            device=device,
            dtype=torch.float32,
            content_fingerprints=content_fingerprints,
        )
        for name, tensors in float_fields.items()
    }
    packed.update(
        {
            name: _pack(
                tensors,
                name=f"pair/{name}",
                device=device,
                dtype=torch.bool,
                content_fingerprints=content_fingerprints,
            )
            for name, tensors in bool_fields.items()
        }
    )
    return CoverageStateDevicePairStore(
        **packed,
        pair_ids=tuple(value.record.pair_id for value in values),
        pair_kinds=tuple(value.record.pair_kind for value in values),
        sample_ids=tuple(value.record.sample_id for value in values),
        actual_input_plus_fingerprints=tuple(
            value.actual_input_plus_fingerprint for value in values
        ),
        actual_input_minus_fingerprints=tuple(
            value.actual_input_minus_fingerprint for value in values
        ),
        optimizer_roles=tuple(value.optimizer_role for value in values),
    )


def prepare_coverage_state_device_cache(
    cache: CoverageStateScalarCache,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> CoverageStateDeviceCache:
    """Verify and pack the complete optimizer-eligible scalar population."""

    if not isinstance(cache, CoverageStateScalarCache):
        raise TypeError("cache must be CoverageStateScalarCache")
    if dtype != torch.float32:
        raise ValueError("CoverageStateFusedBatch fixes floating dtype to FP32")
    target_device = _canonical_device(device)
    cache.verify_unchanged()
    naturals = cache.natural_records
    eligible_pairs = tuple(
        value
        for value in cache.pair_records
        if value.optimizer_role
        in COVERAGE_STATE_DEVICE_CACHE_PAIR_ROLES
    )
    if (
        not naturals
        or not eligible_pairs
        or {
            value.optimizer_role for value in eligible_pairs
        }
        != set(COVERAGE_STATE_DEVICE_CACHE_PAIR_ROLES)
    ):
        raise ValueError(
            "device cache requires natural, clean, and component populations"
        )
    content_fingerprints: dict[str, str] = {}
    natural_store = _pack_naturals(
        naturals,
        device=target_device,
        content_fingerprints=content_fingerprints,
    )
    pair_store = _pack_pairs(
        eligible_pairs,
        device=target_device,
        content_fingerprints=content_fingerprints,
    )
    natural_index = MappingProxyType(
        {
            identity: index
            for index, identity in enumerate(natural_store.record_ids)
        }
    )
    pair_index = MappingProxyType(
        {
            identity: index
            for index, identity in enumerate(pair_store.pair_ids)
        }
    )
    natural_row_index = _pack_row_index(
        len(natural_store.record_ids),
        name="index/natural_rows",
        device=target_device,
        content_fingerprints=content_fingerprints,
    )
    pair_row_index = _pack_row_index(
        len(pair_store.pair_ids),
        name="index/pair_rows",
        device=target_device,
        content_fingerprints=content_fingerprints,
    )
    named_tensors = (
        *natural_store.named_tensors(),
        *pair_store.named_tensors(),
        ("index/natural_rows", natural_row_index),
        ("index/pair_rows", pair_row_index),
    )
    if set(content_fingerprints) != {
        name for name, _ in named_tensors
    }:
        raise AssertionError("packed content fingerprint universe differs")
    tensor_versions = tuple(
        (name, int(value._version))
        for name, value in named_tensors
    )
    tensor_content_fingerprints = tuple(
        (name, content_fingerprints[name])
        for name, _ in named_tensors
    )
    provisional = CoverageStateDeviceCache(
        source_cache=cache,
        natural=natural_store,
        pairs=pair_store,
        natural_id_to_index=natural_index,
        pair_id_to_index=pair_index,
        natural_row_index=natural_row_index,
        pair_row_index=pair_row_index,
        source_cache_fingerprint=cache.cache_fingerprint,
        device=target_device,
        dtype=dtype,
        device_cache_fingerprint="",
        _tensor_versions=tensor_versions,
        _tensor_content_fingerprints=tensor_content_fingerprints,
    )
    result = CoverageStateDeviceCache(
        source_cache=cache,
        natural=natural_store,
        pairs=pair_store,
        natural_id_to_index=natural_index,
        pair_id_to_index=pair_index,
        natural_row_index=natural_row_index,
        pair_row_index=pair_row_index,
        source_cache_fingerprint=cache.cache_fingerprint,
        device=target_device,
        dtype=dtype,
        device_cache_fingerprint=stable_fingerprint(
            provisional.canonical_payload()
        ),
        _tensor_versions=tensor_versions,
        _tensor_content_fingerprints=tensor_content_fingerprints,
    )
    result.verify_unchanged(
        verify_content=False,
        verify_source=False,
    )
    return result


def _selection_indices(
    identities: tuple[str, ...],
    index: Mapping[str, int],
    *,
    expected_count: int,
    name: str,
    row_index: Tensor,
) -> tuple[tuple[int, ...], Tensor]:
    if (
        not isinstance(identities, tuple)
        or len(identities) != expected_count
        or len(set(identities)) != expected_count
    ):
        raise ValueError(f"{name} must contain {expected_count} unique IDs")
    try:
        rows = tuple(index[identity] for identity in identities)
    except KeyError as error:
        raise ValueError(f"{name} contains an ID outside the device cache") from error
    return rows, torch.cat(
        tuple(row_index[row : row + 1] for row in rows),
        dim=0,
    ).contiguous()


def _gather(value: Tensor, index: Tensor) -> Tensor:
    return torch.index_select(value, 0, index).contiguous()


def materialize_coverage_state_device_fused_batch(
    cache: CoverageStateDeviceCache,
    selection: CoverageStateUpdateSelection,
    *,
    verify: bool = False,
    validate: bool = True,
) -> CoverageStateFusedBatch:
    """Gather one schedule selection without another payload transfer.

    This is deliberately a low-level gather API: the caller must supply a
    selection from a cache-bound, already audited training schedule.
    ``epoch`` and ``step`` membership are enforced by that schedule layer,
    not reconstructed here.  A training loop should run one exact
    :meth:`CoverageStateDeviceCache.verify_unchanged` preflight and then use
    ``verify=False, validate=False``; the fused training step validates the
    returned batch once when it materializes model inputs.
    """

    if not isinstance(cache, CoverageStateDeviceCache):
        raise TypeError("cache must be CoverageStateDeviceCache")
    if not isinstance(selection, CoverageStateUpdateSelection):
        raise TypeError("selection must be CoverageStateUpdateSelection")
    if not isinstance(verify, bool) or not isinstance(validate, bool):
        raise TypeError("verify and validate must be bool")
    if verify:
        cache.verify_unchanged()
    miss_rows, miss_index = _selection_indices(
        selection.factual_miss_record_ids,
        cache.natural_id_to_index,
        expected_count=4,
        name="factual_miss",
        row_index=cache.natural_row_index,
    )
    no_miss_rows, no_miss_index = _selection_indices(
        selection.factual_no_miss_record_ids,
        cache.natural_id_to_index,
        expected_count=4,
        name="factual_no_miss",
        row_index=cache.natural_row_index,
    )
    pair_ids = (
        selection.clean_positive_pair_id,
        selection.component_null_pair_id,
    )
    pair_rows, pair_index = _selection_indices(
        pair_ids,
        cache.pair_id_to_index,
        expected_count=2,
        name="optimizer_pair",
        row_index=cache.pair_row_index,
    )
    if tuple(cache.natural.state_kinds[row] for row in miss_rows) != (
        "factual_miss",
    ) * 4:
        raise ValueError("factual_miss selection contains the wrong role")
    if tuple(
        cache.natural.state_kinds[row] for row in no_miss_rows
    ) != ("factual_no_miss",) * 4:
        raise ValueError("factual_no_miss selection contains the wrong role")
    if tuple(cache.pairs.optimizer_roles[row] for row in pair_rows) != (
        "clean_positive",
        "component_null",
    ):
        raise ValueError("optimizer pair selection contains the wrong roles")

    natural = cache.natural

    def natural_batch(
        rows: tuple[int, ...],
        index: Tensor,
        *,
        state_kind: str,
    ) -> CoverageStateNaturalTrainBatch:
        return CoverageStateNaturalTrainBatch(
            feature=_gather(natural.feature, index),
            occupancy=_gather(natural.occupancy, index),
            targets=_absolute_targets_from_store(
                target_field=_gather(natural.target_field, index),
                integration_measure=_gather(
                    natural.integration_measure,
                    index,
                ),
                field_valid_mask=_gather(
                    natural.field_valid_mask,
                    index,
                ),
                loss_valid_mask=_gather(
                    natural.loss_valid_mask,
                    index,
                ),
                focus_support=_gather(natural.focus_support, index),
                focus_support_field=_gather(
                    natural.focus_support_field,
                    index,
                ),
            ),
            record_ids=tuple(natural.record_ids[row] for row in rows),
            sample_ids=tuple(natural.sample_ids[row] for row in rows),
            actual_input_fingerprints=tuple(
                natural.actual_input_fingerprints[row] for row in rows
            ),
            state_kind=state_kind,
        )

    pairs = cache.pairs
    result = CoverageStateFusedBatch(
        factual_miss=natural_batch(
            miss_rows,
            miss_index,
            state_kind="factual_miss",
        ),
        factual_no_miss=natural_batch(
            no_miss_rows,
            no_miss_index,
            state_kind="factual_no_miss",
        ),
        pairs=CoverageStatePairTrainBatch(
            feature=_gather(pairs.feature, pair_index),
            occupancy_plus=_gather(
                pairs.occupancy_plus,
                pair_index,
            ),
            occupancy_minus=_gather(
                pairs.occupancy_minus,
                pair_index,
            ),
            joint_targets=CoverageStatePairTargets(
                target_field_plus=_gather(
                    pairs.joint_target_field_plus,
                    pair_index,
                ),
                target_field_minus=_gather(
                    pairs.joint_target_field_minus,
                    pair_index,
                ),
                focus_support=_gather(
                    pairs.joint_focus_support,
                    pair_index,
                ),
                focus_support_field=_gather(
                    pairs.joint_focus_support_field,
                    pair_index,
                ),
                integration_measure=_gather(
                    pairs.joint_integration_measure,
                    pair_index,
                ),
                valid_mask=_gather(
                    pairs.joint_valid_mask,
                    pair_index,
                ),
            ),
            absolute_targets_plus=_absolute_targets_from_store(
                target_field=_gather(
                    pairs.absolute_plus_target_field,
                    pair_index,
                ),
                integration_measure=_gather(
                    pairs.absolute_plus_integration_measure,
                    pair_index,
                ),
                field_valid_mask=_gather(
                    pairs.absolute_plus_field_valid_mask,
                    pair_index,
                ),
                loss_valid_mask=_gather(
                    pairs.absolute_plus_loss_valid_mask,
                    pair_index,
                ),
                focus_support=_gather(
                    pairs.absolute_plus_focus_support,
                    pair_index,
                ),
                focus_support_field=_gather(
                    pairs.absolute_plus_focus_support_field,
                    pair_index,
                ),
            ),
            absolute_targets_minus=_absolute_targets_from_store(
                target_field=_gather(
                    pairs.absolute_minus_target_field,
                    pair_index,
                ),
                integration_measure=_gather(
                    pairs.absolute_minus_integration_measure,
                    pair_index,
                ),
                field_valid_mask=_gather(
                    pairs.absolute_minus_field_valid_mask,
                    pair_index,
                ),
                loss_valid_mask=_gather(
                    pairs.absolute_minus_loss_valid_mask,
                    pair_index,
                ),
                focus_support=_gather(
                    pairs.absolute_minus_focus_support,
                    pair_index,
                ),
                focus_support_field=_gather(
                    pairs.absolute_minus_focus_support_field,
                    pair_index,
                ),
            ),
            pair_ids=tuple(pairs.pair_ids[row] for row in pair_rows),
            pair_kinds=tuple(
                pairs.pair_kinds[row] for row in pair_rows
            ),
            sample_ids=tuple(
                pairs.sample_ids[row] for row in pair_rows
            ),
            actual_input_plus_fingerprints=tuple(
                pairs.actual_input_plus_fingerprints[row]
                for row in pair_rows
            ),
            actual_input_minus_fingerprints=tuple(
                pairs.actual_input_minus_fingerprints[row]
                for row in pair_rows
            ),
        ),
    )
    if validate:
        result.validate()
    return result


__all__ = [
    "COVERAGE_STATE_DEVICE_CACHE_PAIR_ROLES",
    "COVERAGE_STATE_DEVICE_CACHE_SCHEMA",
    "CoverageStateDeviceCache",
    "CoverageStateDeviceNaturalStore",
    "CoverageStateDevicePairStore",
    "materialize_coverage_state_device_fused_batch",
    "prepare_coverage_state_device_cache",
]
