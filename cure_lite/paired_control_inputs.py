"""Pure input and pairing transforms for frozen CURE-Lite matched controls.

This module only constructs control inputs and a deterministic target
permutation plan.  It does not alter :class:`PairExample`, run a decoder,
detach endpoint scores, train a model, or access any dataset split.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Iterable

import torch
from torch import Tensor

from .cache.schema import stable_fingerprint
from .paired_types import PairExample, tensor_content_fingerprint
from .sampling import stable_hash


DCT_BASIS_SCHEMA = "cure-lite-dct-coordinate-basis-v1"
TARGET_PERMUTATION_SCHEMA = "cure-lite-target-permutation-v1"
TARGET_PERMUTATION_READY = "READY"
TARGET_PERMUTATION_INCONCLUSIVE = "COMPUTATIONALLY_INCONCLUSIVE"

_DCT_FORMULA = (
    "cos(pi*(2*y+1)*k_y/(2*h))*cos(pi*(2*x+1)*k_x/(2*w));"
    "per-channel spatial mean removal;per-channel RMS normalization"
)
_DCT_ORDER = "(k_y+k_x,k_y,k_x);exclude_(0,0)"


def _validate_feature_template(feature: Tensor) -> None:
    if not isinstance(feature, Tensor):
        raise TypeError("feature must be a tensor")
    if feature.ndim != 4 or min(feature.shape) < 1:
        raise ValueError("feature must have non-empty shape [B,C,h,w]")
    if not feature.is_floating_point():
        raise TypeError("feature must be floating point")


def nominal_zero_feature_like(feature: Tensor) -> Tensor:
    """Return the nominal occupancy-only feature without reading feature values."""

    _validate_feature_template(feature)
    return torch.zeros(
        tuple(feature.shape),
        dtype=feature.dtype,
        device=feature.device,
    )


def _validate_occupancy_pair(
    occupancy_plus: Tensor,
    occupancy_minus: Tensor,
) -> None:
    if not isinstance(occupancy_plus, Tensor) or not isinstance(
        occupancy_minus,
        Tensor,
    ):
        raise TypeError("occupancy endpoints must be tensors")
    if (
        occupancy_plus.ndim != 4
        or occupancy_plus.shape[0] < 1
        or occupancy_plus.shape[1] != 1
        or min(occupancy_plus.shape[-2:]) < 1
    ):
        raise ValueError("occupancy endpoints must have shape [B,1,H,W]")
    if occupancy_plus.shape != occupancy_minus.shape:
        raise ValueError("occupancy endpoints must have identical shapes")
    if occupancy_plus.dtype != torch.bool or occupancy_minus.dtype != torch.bool:
        raise TypeError("occupancy endpoints must be bool")
    if occupancy_plus.device != occupancy_minus.device:
        raise ValueError("occupancy endpoints must share a device")


def feature_only_zero_occupancy(
    occupancy_plus: Tensor,
    occupancy_minus: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return one fixed zero occupancy for both feature-only endpoints."""

    _validate_occupancy_pair(occupancy_plus, occupancy_minus)
    fixed = torch.zeros(
        tuple(occupancy_plus.shape),
        dtype=torch.bool,
        device=occupancy_plus.device,
    )
    return fixed, fixed


def _validate_basis_dtype(dtype: torch.dtype) -> None:
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype")
    probe = torch.empty((), dtype=dtype)
    if not probe.is_floating_point():
        raise TypeError("DCT coordinate basis dtype must be floating point")


def _dct_modes(
    height: int,
    width: int,
    channels: int,
) -> tuple[tuple[int, int], ...]:
    modes = tuple(
        sorted(
            (
                (k_y, k_x)
                for k_y in range(height)
                for k_x in range(width)
                if (k_y, k_x) != (0, 0)
            ),
            key=lambda mode: (mode[0] + mode[1], mode[0], mode[1]),
        )
    )
    if channels > len(modes):
        raise ValueError(
            "capacity-active DCT control requires C <= h*w-1 "
            f"({channels} > {len(modes)})"
        )
    return modes[:channels]


def _basis_payload(
    tensor: Tensor,
    modes: tuple[tuple[int, int], ...],
    tensor_fingerprint: str,
) -> dict[str, object]:
    return {
        "schema_version": DCT_BASIS_SCHEMA,
        "formula": _DCT_FORMULA,
        "mode_order": _DCT_ORDER,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "modes": [list(mode) for mode in modes],
        "tensor_fingerprint": tensor_fingerprint,
        "source_independent": True,
        "dc_excluded": True,
        "per_channel_centered": True,
        "per_channel_rms_normalized": True,
    }


@dataclass(frozen=True, eq=False)
class DCTCoordinateBasis:
    """One source-independent, capacity-active coordinate basis."""

    tensor: Tensor
    modes: tuple[tuple[int, int], ...]
    tensor_fingerprint: str
    basis_fingerprint: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tensor, Tensor)
            or self.tensor.device.type != "cpu"
            or self.tensor.ndim != 4
            or self.tensor.shape[0] != 1
            or not self.tensor.is_floating_point()
            or self.tensor.requires_grad
            or not torch.isfinite(self.tensor).all()
        ):
            raise TypeError(
                "DCT basis must be a finite detached CPU floating tensor "
                "with shape [1,C,h,w]"
            )
        channels = int(self.tensor.shape[1])
        if (
            not isinstance(self.modes, tuple)
            or len(self.modes) != channels
            or len(set(self.modes)) != channels
            or (0, 0) in self.modes
            or self.modes
            != tuple(
                sorted(
                    self.modes,
                    key=lambda mode: (
                        mode[0] + mode[1],
                        mode[0],
                        mode[1],
                    ),
                )
            )
        ):
            raise ValueError("DCT modes do not follow the frozen non-DC order")
        if tensor_content_fingerprint(self.tensor) != self.tensor_fingerprint:
            raise ValueError("DCT tensor_fingerprint does not reproduce")
        payload = _basis_payload(
            self.tensor,
            self.modes,
            self.tensor_fingerprint,
        )
        if stable_fingerprint(payload) != self.basis_fingerprint:
            raise ValueError("DCT basis_fingerprint does not reproduce")

        values = self.tensor.to(torch.float64)
        means = values.mean(dim=(-2, -1))
        rms = torch.sqrt((values.square()).mean(dim=(-2, -1)))
        tolerance = (
            5e-3
            if self.tensor.dtype in (torch.float16, torch.bfloat16)
            else 1e-5
            if self.tensor.dtype == torch.float32
            else 1e-10
        )
        if torch.any(means.abs() > tolerance):
            raise ValueError("DCT basis channels are not spatially centered")
        if torch.any((rms - 1.0).abs() > tolerance):
            raise ValueError("DCT basis channels are not RMS normalized")
        if torch.any(self.tensor.flatten(2).abs().sum(dim=2) == 0):
            raise ValueError("DCT basis contains an inactive all-zero channel")

    @property
    def canonical_payload(self) -> dict[str, object]:
        return _basis_payload(
            self.tensor,
            self.modes,
            self.tensor_fingerprint,
        )

    def expand(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
    ) -> Tensor:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer")
        target_device = self.tensor.device if device is None else device
        return (
            self.tensor.expand(batch_size, -1, -1, -1)
            .clone()
            .to(device=target_device)
            .contiguous()
        )


def build_dct_coordinate_basis(
    *,
    channels: int,
    height: int,
    width: int,
    dtype: torch.dtype = torch.float32,
) -> DCTCoordinateBasis:
    """Build the frozen non-DC 2-D DCT coordinate basis on CPU."""

    for value, name in (
        (channels, "channels"),
        (height, "height"),
        (width, "width"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    _validate_basis_dtype(dtype)
    modes = _dct_modes(height, width, channels)
    y = torch.arange(height, dtype=torch.float64)
    x = torch.arange(width, dtype=torch.float64)
    planes: list[Tensor] = []
    for k_y, k_x in modes:
        vertical = torch.cos(pi * (2.0 * y + 1.0) * k_y / (2.0 * height))
        horizontal = torch.cos(pi * (2.0 * x + 1.0) * k_x / (2.0 * width))
        plane = vertical[:, None] * horizontal[None, :]
        plane = plane - plane.mean()
        rms = torch.sqrt(plane.square().mean())
        if not torch.isfinite(rms) or float(rms) <= torch.finfo(torch.float64).eps:
            raise RuntimeError("a selected DCT channel cannot be normalized")
        plane = plane / rms

        # Re-center and normalize after dtype quantization so the actual tensor,
        # not only its float64 precursor, satisfies the frozen contract.
        quantized = plane.to(dtype=dtype).to(torch.float64)
        quantized = quantized - quantized.mean()
        quantized_rms = torch.sqrt(quantized.square().mean())
        if (
            not torch.isfinite(quantized_rms)
            or float(quantized_rms) <= torch.finfo(torch.float64).eps
        ):
            raise RuntimeError("DCT dtype quantization deactivated a channel")
        planes.append((quantized / quantized_rms).to(dtype=dtype))

    tensor = torch.stack(planes, dim=0).unsqueeze(0).contiguous()
    tensor_fingerprint = tensor_content_fingerprint(tensor)
    basis_fingerprint = stable_fingerprint(
        _basis_payload(tensor, modes, tensor_fingerprint)
    )
    return DCTCoordinateBasis(
        tensor=tensor,
        modes=modes,
        tensor_fingerprint=tensor_fingerprint,
        basis_fingerprint=basis_fingerprint,
    )


def capacity_active_dct_feature_like(
    feature: Tensor,
) -> tuple[Tensor, DCTCoordinateBasis]:
    """Replace any same-shaped source feature with one fixed DCT basis."""

    _validate_feature_template(feature)
    basis = build_dct_coordinate_basis(
        channels=int(feature.shape[1]),
        height=int(feature.shape[2]),
        width=int(feature.shape[3]),
        dtype=feature.dtype,
    )
    return basis.expand(int(feature.shape[0]), device=feature.device), basis


def _pair_signature(pair: PairExample) -> tuple[object, ...]:
    return (
        tuple(pair.feature.shape),
        str(pair.feature.dtype),
        tuple(pair.occupancy_plus.shape),
        str(pair.occupancy_plus.dtype),
        tuple(pair.occupancy_minus.shape),
        str(pair.occupancy_minus.dtype),
        tuple(pair.clean_increment.shape),
        str(pair.clean_increment.dtype),
        tuple(pair.image_valid_mask.shape),
        str(pair.image_valid_mask.dtype),
    )


def target_permutation_compatible(
    recipient: PairExample,
    donor: PairExample,
) -> bool:
    """Return the exact protocol-6.5 recipient/donor compatibility."""

    if not isinstance(recipient, PairExample) or not isinstance(
        donor,
        PairExample,
    ):
        raise TypeError("recipient and donor must be PairExample values")
    if (
        recipient.pair_kind != "clean_positive"
        or donor.pair_kind != "clean_positive"
    ):
        return False
    if recipient.pair_id == donor.pair_id:
        return False
    if recipient.sample_id == donor.sample_id:
        return False
    if _pair_signature(recipient) != _pair_signature(donor):
        return False
    return not bool(
        torch.any(donor.clean_increment & ~recipient.image_valid_mask)
    )


def _canonical_pairs(
    pairs: Iterable[PairExample],
) -> tuple[PairExample, ...]:
    values = tuple(pairs)
    if any(not isinstance(pair, PairExample) for pair in values):
        raise TypeError("pairs must contain only PairExample values")
    if any(pair.pair_kind != "clean_positive" for pair in values):
        raise ValueError("target permutation accepts only clean_positive pairs")
    pair_ids = tuple(pair.pair_id for pair in values)
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("target permutation pair IDs must be unique")
    return tuple(
        sorted(
            values,
            key=lambda pair: (
                stable_hash(
                    "target-permutation-canonical-v1",
                    pair.pair_id,
                ),
                pair.pair_id,
            ),
        )
    )


def _initial_perfect_matching(
    adjacency: tuple[tuple[int, ...], ...],
) -> tuple[list[int], list[int]] | None:
    size = len(adjacency)
    match_donor = [-1] * size

    def augment(recipient: int, seen: set[int]) -> bool:
        for donor in adjacency[recipient]:
            if donor in seen:
                continue
            seen.add(donor)
            owner = match_donor[donor]
            if owner == -1 or augment(owner, seen):
                match_donor[donor] = recipient
                return True
        return False

    for recipient in range(size):
        if not augment(recipient, set()):
            return None
    match_recipient = [-1] * size
    for donor, recipient in enumerate(match_donor):
        if recipient >= 0:
            match_recipient[recipient] = donor
    if any(donor < 0 for donor in match_recipient):
        return None
    return match_recipient, match_donor


def _force_matching_edge(
    adjacency: tuple[tuple[int, ...], ...],
    match_recipient: list[int],
    match_donor: list[int],
    *,
    recipient: int,
    donor: int,
    fixed_recipients: frozenset[int],
    fixed_donors: frozenset[int],
) -> tuple[list[int], list[int]] | None:
    if match_recipient[recipient] == donor:
        return match_recipient.copy(), match_donor.copy()
    displaced = match_donor[donor]
    old_donor = match_recipient[recipient]
    if displaced < 0 or old_donor < 0:
        raise AssertionError("forcing started from a non-perfect matching")

    trial_recipient = match_recipient.copy()
    trial_donor = match_donor.copy()
    trial_recipient[recipient] = -1
    trial_donor[old_donor] = -1
    trial_recipient[displaced] = -1
    trial_donor[donor] = -1
    trial_recipient[recipient] = donor
    trial_donor[donor] = recipient

    blocked_recipients = fixed_recipients | frozenset((recipient,))
    blocked_donors = fixed_donors | frozenset((donor,))

    def augment(current: int, seen: set[int]) -> bool:
        for candidate in adjacency[current]:
            if candidate in blocked_donors or candidate in seen:
                continue
            seen.add(candidate)
            owner = trial_donor[candidate]
            if owner == -1:
                trial_recipient[current] = candidate
                trial_donor[candidate] = current
                return True
            if owner in blocked_recipients:
                continue
            if augment(owner, seen):
                trial_recipient[current] = candidate
                trial_donor[candidate] = current
                return True
        return False

    if not augment(displaced, set()):
        return None
    return trial_recipient, trial_donor


@dataclass(frozen=True)
class TargetPermutationAssignment:
    recipient_pair_id: str
    recipient_sample_id: str
    donor_pair_id: str
    donor_sample_id: str
    donor_target_fingerprint: str

    def canonical_payload(self) -> dict[str, str]:
        return {
            "recipient_pair_id": self.recipient_pair_id,
            "recipient_sample_id": self.recipient_sample_id,
            "donor_pair_id": self.donor_pair_id,
            "donor_sample_id": self.donor_sample_id,
            "donor_target_fingerprint": self.donor_target_fingerprint,
        }


def _permutation_payload(
    *,
    status: str,
    canonical_pair_ids: tuple[str, ...],
    compatible_edges: int,
    assignments: tuple[TargetPermutationAssignment, ...],
    reason_code: str | None,
) -> dict[str, object]:
    return {
        "schema_version": TARGET_PERMUTATION_SCHEMA,
        "status": status,
        "canonical_pair_ids": list(canonical_pair_ids),
        "compatible_edges": compatible_edges,
        "assignments": [
            assignment.canonical_payload() for assignment in assignments
        ],
        "reason_code": reason_code,
        "source_disjoint": status == TARGET_PERMUTATION_READY,
        "fixed_point_free": status == TARGET_PERMUTATION_READY,
        "lexicographically_minimal_in_stable_hash_order": (
            status == TARGET_PERMUTATION_READY
        ),
    }


@dataclass(frozen=True)
class TargetPermutationPlan:
    """A complete perfect permutation or an explicit inconclusive outcome."""

    status: str
    canonical_pair_ids: tuple[str, ...]
    compatible_edges: int
    assignments: tuple[TargetPermutationAssignment, ...]
    reason_code: str | None
    plan_fingerprint: str

    def __post_init__(self) -> None:
        if self.status not in (
            TARGET_PERMUTATION_READY,
            TARGET_PERMUTATION_INCONCLUSIVE,
        ):
            raise ValueError("unknown target permutation status")
        if (
            isinstance(self.compatible_edges, bool)
            or not isinstance(self.compatible_edges, int)
            or self.compatible_edges < 0
        ):
            raise ValueError("compatible_edges must be a non-negative integer")
        if len(set(self.canonical_pair_ids)) != len(self.canonical_pair_ids):
            raise ValueError("canonical_pair_ids must be unique")
        if self.status == TARGET_PERMUTATION_READY:
            if self.reason_code is not None:
                raise ValueError("a ready permutation cannot have a reason_code")
            if len(self.assignments) != len(self.canonical_pair_ids):
                raise ValueError("a ready permutation must assign every pair")
            if tuple(
                assignment.recipient_pair_id
                for assignment in self.assignments
            ) != self.canonical_pair_ids:
                raise ValueError("assignments are not in canonical recipient order")
            donors = tuple(
                assignment.donor_pair_id for assignment in self.assignments
            )
            if set(donors) != set(self.canonical_pair_ids):
                raise ValueError("donor assignments are not a permutation")
            if any(
                assignment.recipient_pair_id == assignment.donor_pair_id
                or assignment.recipient_sample_id == assignment.donor_sample_id
                for assignment in self.assignments
            ):
                raise ValueError("permutation contains a fixed point or source reuse")
        else:
            if not self.reason_code or self.assignments:
                raise ValueError(
                    "an inconclusive permutation needs a reason and no assignments"
                )
        payload = _permutation_payload(
            status=self.status,
            canonical_pair_ids=self.canonical_pair_ids,
            compatible_edges=self.compatible_edges,
            assignments=self.assignments,
            reason_code=self.reason_code,
        )
        if stable_fingerprint(payload) != self.plan_fingerprint:
            raise ValueError("plan_fingerprint does not reproduce")

    @property
    def ready(self) -> bool:
        return self.status == TARGET_PERMUTATION_READY

    @property
    def canonical_payload(self) -> dict[str, object]:
        return _permutation_payload(
            status=self.status,
            canonical_pair_ids=self.canonical_pair_ids,
            compatible_edges=self.compatible_edges,
            assignments=self.assignments,
            reason_code=self.reason_code,
        )


def build_target_permutation(
    pairs: Iterable[PairExample],
) -> TargetPermutationPlan:
    """Build the lexicographically minimal compatible perfect permutation."""

    canonical = _canonical_pairs(pairs)
    pair_ids = tuple(pair.pair_id for pair in canonical)
    adjacency = tuple(
        tuple(
            donor_index
            for donor_index, donor in enumerate(canonical)
            if target_permutation_compatible(recipient, donor)
        )
        for recipient in canonical
    )
    compatible_edges = sum(len(edges) for edges in adjacency)
    initial = _initial_perfect_matching(adjacency)
    if initial is None:
        payload = _permutation_payload(
            status=TARGET_PERMUTATION_INCONCLUSIVE,
            canonical_pair_ids=pair_ids,
            compatible_edges=compatible_edges,
            assignments=(),
            reason_code="no_compatible_perfect_matching",
        )
        return TargetPermutationPlan(
            status=TARGET_PERMUTATION_INCONCLUSIVE,
            canonical_pair_ids=pair_ids,
            compatible_edges=compatible_edges,
            assignments=(),
            reason_code="no_compatible_perfect_matching",
            plan_fingerprint=stable_fingerprint(payload),
        )

    match_recipient, match_donor = initial
    fixed_donors: set[int] = set()
    for recipient in range(len(canonical)):
        selected: tuple[list[int], list[int]] | None = None
        for donor in adjacency[recipient]:
            if donor in fixed_donors:
                continue
            trial = _force_matching_edge(
                adjacency,
                match_recipient,
                match_donor,
                recipient=recipient,
                donor=donor,
                fixed_recipients=frozenset(range(recipient)),
                fixed_donors=frozenset(fixed_donors),
            )
            if trial is not None:
                selected = trial
                break
        if selected is None:
            raise AssertionError(
                "a previously verified perfect matching became infeasible"
            )
        match_recipient, match_donor = selected
        fixed_donors.add(match_recipient[recipient])

    assignments = tuple(
        TargetPermutationAssignment(
            recipient_pair_id=recipient.pair_id,
            recipient_sample_id=recipient.sample_id,
            donor_pair_id=canonical[match_recipient[index]].pair_id,
            donor_sample_id=canonical[match_recipient[index]].sample_id,
            donor_target_fingerprint=tensor_content_fingerprint(
                canonical[match_recipient[index]].clean_increment
            ),
        )
        for index, recipient in enumerate(canonical)
    )
    payload = _permutation_payload(
        status=TARGET_PERMUTATION_READY,
        canonical_pair_ids=pair_ids,
        compatible_edges=compatible_edges,
        assignments=assignments,
        reason_code=None,
    )
    return TargetPermutationPlan(
        status=TARGET_PERMUTATION_READY,
        canonical_pair_ids=pair_ids,
        compatible_edges=compatible_edges,
        assignments=assignments,
        reason_code=None,
        plan_fingerprint=stable_fingerprint(payload),
    )


def materialize_permuted_label_increments(
    pairs: Iterable[PairExample],
    plan: TargetPermutationPlan,
) -> tuple[Tensor, ...]:
    """Return donor ``A`` labels in canonical recipient order."""

    if not isinstance(plan, TargetPermutationPlan):
        raise TypeError("plan must be a TargetPermutationPlan")
    if not plan.ready:
        raise RuntimeError("target permutation is COMPUTATIONALLY_INCONCLUSIVE")
    canonical = _canonical_pairs(pairs)
    if tuple(pair.pair_id for pair in canonical) != plan.canonical_pair_ids:
        raise ValueError("pairs do not match the permutation plan population")
    by_id = {pair.pair_id: pair for pair in canonical}
    targets: list[Tensor] = []
    for assignment in plan.assignments:
        recipient = by_id[assignment.recipient_pair_id]
        donor = by_id[assignment.donor_pair_id]
        if not target_permutation_compatible(recipient, donor):
            raise RuntimeError("stored target permutation is no longer compatible")
        fingerprint = tensor_content_fingerprint(donor.clean_increment)
        if fingerprint != assignment.donor_target_fingerprint:
            raise RuntimeError("donor target fingerprint changed")
        targets.append(donor.clean_increment.to(torch.float32).clone())
    return tuple(targets)


__all__ = [
    "DCT_BASIS_SCHEMA",
    "TARGET_PERMUTATION_INCONCLUSIVE",
    "TARGET_PERMUTATION_READY",
    "TARGET_PERMUTATION_SCHEMA",
    "DCTCoordinateBasis",
    "TargetPermutationAssignment",
    "TargetPermutationPlan",
    "build_dct_coordinate_basis",
    "build_target_permutation",
    "capacity_active_dct_feature_like",
    "feature_only_zero_occupancy",
    "materialize_permuted_label_increments",
    "nominal_zero_feature_like",
    "target_permutation_compatible",
]
