"""Read-only real-``D_R`` identifiability gate for BFA-CMIF v20.

The gate consumes an already-built
:class:`~cure_lite.experiment.coverage_state_real_dr_inputs.CoverageStateRealDRInputs`
graph and its already-selected seed-42 bounded population.  It never rebuilds
the expensive real-data graph, constructs an optimizer, changes a parameter,
or opens ``D_V``/``D_T``.

For every declared factual-target, clean-target, writable-background, and
component coordinate, the gate evaluates the shared hidden energy at the
binary endpoints and midpoint:

``H0 = H(B, U_p=0)``, ``H1 = H(B, U_p=1)``,
``Hm = H(B, U_p=1/2)``.

It then records the binary odd basis ``o = (H0-H1)/2`` and midpoint
curvature ``e = (H0+H1)/2-Hm``.  The visible distributions are summarized
without dropping evidence: every ordered vector stream and every ordered
norm stream is content-fingerprinted, while fixed nearest-rank quantiles are
stored as hexadecimal floats.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import fsum, isfinite, sqrt
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import functional as F

from ..cache.schema import file_sha256, stable_fingerprint
from ..coverage_state_binary_flip_antisymmetric import (
    CSLF_BFA_EQUATION_POLICY,
    CSLF_BFA_FIELD_POLICY,
    CSLF_BFA_FLIP_POLICY,
    CURELiteBinaryFlipAntisymmetricLevelSet,
    CoverageStateBinaryFlipAntisymmetricConfig,
)
from ..coverage_state_sobolev import (
    CoverageStateAbsoluteTargets,
    CoverageStatePairTargets,
    coverage_state_absolute_sobolev_loss_from_targets,
    coverage_state_pmope_pair_loss_from_targets,
)
from ..paired_types import tensor_content_fingerprint
from .coverage_state_bfa_dataset_free import (
    COVERAGE_STATE_BFA_MARGIN,
    CoverageStateBFADatasetFreeReceipt,
)
from .coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_ROLE_COUNT,
    COVERAGE_STATE_BOUNDED_SEED,
    CoverageStateBoundedPopulation,
)
from .coverage_state_cmif_dataset_free import (
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_CMIF_FORMAL_WIDTH,
)
from .coverage_state_real_dr_inputs import CoverageStateRealDRInputs


COVERAGE_STATE_BFA_DR_GATE_SCHEMA = (
    "cure-lite-bfa-cmif-v20-real-dr-identifiability-gate-v1"
)
COVERAGE_STATE_BFA_DR_EXECUTION_SEED = 42
COVERAGE_STATE_BFA_DR_RATIO_MULTIPLIER = 128
COVERAGE_STATE_BFA_DR_FLOAT32_EPSILON = torch.finfo(torch.float32).eps
COVERAGE_STATE_BFA_DR_RATIO_THRESHOLD = (
    COVERAGE_STATE_BFA_DR_RATIO_MULTIPLIER
    * COVERAGE_STATE_BFA_DR_FLOAT32_EPSILON
)
COVERAGE_STATE_BFA_DR_PASS_DECISION = "BFA_D_R_IDENTIFIABILITY_PASS"
COVERAGE_STATE_BFA_DR_FAIL_DECISION = "BFA_D_R_IDENTIFIABILITY_FAIL"
COVERAGE_STATE_BFA_DR_QUANTILES = (
    ("q000", 0, 100),
    ("q001", 1, 100),
    ("q005", 5, 100),
    ("q025", 25, 100),
    ("q050", 50, 100),
    ("q075", 75, 100),
    ("q095", 95, 100),
    ("q099", 99, 100),
    ("q100", 100, 100),
)
COVERAGE_STATE_BFA_DR_IMPLEMENTATION_PATHS = (
    "cure_lite/coverage_state_binary_flip_antisymmetric.py",
    "cure_lite/coverage_state_sobolev.py",
    "cure_lite/experiment/coverage_state_bfa_dataset_free.py",
    "cure_lite/experiment/coverage_state_bounded_protocol.py",
    "cure_lite/experiment/coverage_state_real_dr_inputs.py",
    "cure_lite/experiment/coverage_state_bfa_dr_gate.py",
)


@dataclass(frozen=True)
class _CoordinateState:
    state_id: str
    sample_id: str
    state_kind: str
    endpoint: str
    feature: Tensor
    occupancy: Tensor
    valid_mask: Tensor
    target_mask: Tensor
    background_mask: Tensor
    component_mask: Tensor
    target_group_id: str | None
    component_writable: bool


@dataclass(frozen=True)
class _HiddenBasis:
    h0: Tensor
    h1: Tensor
    hm: Tensor
    odd: Tensor
    even: Tensor
    oriented_odd: Tensor


def _implementation_binding() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[2]
    rows: list[tuple[str, str]] = []
    for relative in COVERAGE_STATE_BFA_DR_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"BFA D_R implementation path is invalid: {relative}"
            )
        rows.append((relative, file_sha256(path)))
    return tuple(rows)


def _hex(value: float, *, name: str) -> str:
    number = float(value)
    if not isfinite(number):
        raise FloatingPointError(f"{name} is non-finite")
    return number.hex()


def _model_fingerprint(
    model: CURELiteBinaryFlipAntisymmetricLevelSet,
) -> str:
    return stable_fingerprint(
        {
            name: tensor_content_fingerprint(value)
            for name, value in sorted(model.state_dict().items())
        }
    )


def _cuda_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    if device.index is None:
        raise ValueError("CUDA device must have an explicit index")
    return [device.index]


def _phase_hidden_to_output(
    value: Tensor,
    *,
    stride: int,
) -> Tensor:
    """Map ``[B,P,W,h,w]`` phase-hidden values to ``[B,W,H,W]``."""

    if (
        not isinstance(value, Tensor)
        or value.ndim != 5
        or value.shape[1] != stride**2
        or not value.is_floating_point()
    ):
        raise ValueError("phase hidden tensor has an invalid shape")
    batch, phases, width, height, columns = value.shape
    native = (
        value.permute(0, 2, 1, 3, 4)
        .reshape(batch, width * phases, height, columns)
        .contiguous()
    )
    return F.pixel_shuffle(native, stride).contiguous()


def _hidden_basis(
    model: CURELiteBinaryFlipAntisymmetricLevelSet,
    feature: Tensor,
    occupancy: Tensor,
) -> _HiddenBasis:
    """Return endpoint, midpoint, odd, and even hidden coordinates."""

    fields = model.forward_fields(feature, occupancy)
    actual = fields.actual_feature_presence_hidden.unsqueeze(1)
    flipped = fields.flipped_feature_presence_hidden
    phase = fields.phase_occupancy.unsqueeze(2)
    h0_native = torch.where(phase, flipped, actual)
    h1_native = torch.where(phase, actual, flipped)

    center = model.config.coarse_radius
    center_weight = model.occupancy_weight[
        :, :, center, center
    ].transpose(0, 1)
    midpoint_delta = (
        0.5 - fields.phase_occupancy.to(dtype=torch.float32)
    ).unsqueeze(2) * center_weight[None, :, :, None, None]
    midpoint_occupancy_affine = (
        fields.occupancy_affine.unsqueeze(1) + midpoint_delta
    )
    midpoint_joint_affine = (
        fields.joint_affine.unsqueeze(1) + midpoint_delta
    )
    hm_native = (
        F.silu(midpoint_joint_affine)
        - F.silu(midpoint_occupancy_affine)
    )
    odd_native = 0.5 * (h0_native - h1_native)
    even_native = 0.5 * (h0_native + h1_native) - hm_native
    oriented_native = fields.odd_feature_presence_hidden
    if not torch.allclose(
        oriented_native,
        torch.where(phase, -odd_native, odd_native),
        rtol=2.0e-6,
        atol=2.0e-7,
    ):
        raise AssertionError("BFA oriented and canonical odd bases differ")

    kwargs = {
        "stride": model.config.feature_stride,
    }
    result = _HiddenBasis(
        h0=_phase_hidden_to_output(h0_native, **kwargs),
        h1=_phase_hidden_to_output(h1_native, **kwargs),
        hm=_phase_hidden_to_output(hm_native, **kwargs),
        odd=_phase_hidden_to_output(odd_native, **kwargs),
        even=_phase_hidden_to_output(even_native, **kwargs),
        oriented_odd=_phase_hidden_to_output(
            oriented_native,
            **kwargs,
        ),
    )
    expected = (
        feature.shape[0],
        model.config.width,
        occupancy.shape[-2],
        occupancy.shape[-1],
    )
    for name in ("h0", "h1", "hm", "odd", "even", "oriented_odd"):
        tensor = getattr(result, name)
        if (
            tuple(tensor.shape) != tuple(expected)
            or tensor.dtype != torch.float32
            or tensor.device != feature.device
            or not bool(torch.isfinite(tensor).all())
        ):
            raise FloatingPointError(
                f"BFA {name} basis is invalid"
            )
    return result


def _vectors_at(value: Tensor, mask: Tensor) -> Tensor:
    if (
        value.ndim != 4
        or mask.dtype != torch.bool
        or mask.ndim != 4
        or mask.shape[0] != value.shape[0]
        or mask.shape[1] != 1
        or tuple(mask.shape[-2:]) != tuple(value.shape[-2:])
    ):
        raise ValueError("basis and coordinate mask do not align")
    return value.permute(0, 2, 3, 1)[mask[:, 0]].contiguous()


def _nearest_rank(
    sorted_values: Tensor,
    *,
    numerator: int,
    denominator: int,
) -> float:
    if (
        sorted_values.ndim != 1
        or sorted_values.numel() < 1
        or numerator < 0
        or numerator > denominator
        or denominator < 1
    ):
        raise ValueError("invalid nearest-rank request")
    count = int(sorted_values.numel())
    index = ((count - 1) * numerator + denominator // 2) // denominator
    return float(sorted_values[index].item())


def _distribution(values: Tensor, *, name: str) -> dict[str, object]:
    value = values.detach().to("cpu", dtype=torch.float32).flatten().contiguous()
    if value.numel() < 1 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} distribution must be finite and nonempty")
    ordered = torch.sort(value).values
    double = value.to(dtype=torch.float64)
    count = int(value.numel())
    total = float(double.sum().item())
    square_total = float(double.square().sum().item())
    mean = total / count
    square_mean = square_total / count
    variance = max(0.0, square_mean - mean * mean)
    quantiles = {
        key: _hex(
            _nearest_rank(
                ordered,
                numerator=numerator,
                denominator=denominator,
            ),
            name=f"{name} {key}",
        )
        for key, numerator, denominator in COVERAGE_STATE_BFA_DR_QUANTILES
    }
    return {
        "count": count,
        "finite_count": count,
        "nonzero_count": int(torch.count_nonzero(value)),
        "exact_zero_count": int(torch.count_nonzero(value == 0.0)),
        "minimum_hex": _hex(float(ordered[0]), name=f"{name} minimum"),
        "maximum_hex": _hex(float(ordered[-1]), name=f"{name} maximum"),
        "mean_hex": _hex(mean, name=f"{name} mean"),
        "standard_deviation_hex": _hex(
            sqrt(variance),
            name=f"{name} standard deviation",
        ),
        "sum_squares_hex": _hex(
            square_total,
            name=f"{name} sum squares",
        ),
        "nearest_rank_quantiles": quantiles,
        "ordered_value_fingerprint": tensor_content_fingerprint(ordered),
        "quantile_policy": (
            "fixed_nearest_rank_round_half_up_over_count_minus_one_v1"
        ),
    }


def _stream_summary(values: Tensor, *, name: str) -> dict[str, object]:
    """Bind one state-role stream without an unnecessary per-state sort."""

    value = values.detach().to("cpu", dtype=torch.float32).flatten().contiguous()
    if value.numel() < 1 or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} stream must be finite and nonempty")
    double = value.to(dtype=torch.float64)
    count = int(value.numel())
    return {
        "count": count,
        "finite_count": count,
        "nonzero_count": int(torch.count_nonzero(value)),
        "exact_zero_count": int(torch.count_nonzero(value == 0.0)),
        "minimum_hex": _hex(float(value.amin()), name=f"{name} minimum"),
        "maximum_hex": _hex(float(value.amax()), name=f"{name} maximum"),
        "sum_hex": _hex(float(double.sum()), name=f"{name} sum"),
        "sum_squares_hex": _hex(
            float(double.square().sum()),
            name=f"{name} sum squares",
        ),
        "ordered_coordinate_value_fingerprint": (
            tensor_content_fingerprint(value)
        ),
        "aggregate_quantiles_reported_separately": True,
    }


def _mask(
    value: Tensor,
    *,
    occupancy: Tensor,
    valid: Tensor,
    excluded: Iterable[Tensor] = (),
) -> Tensor:
    result = value & valid & ~occupancy
    for item in excluded:
        result = result & ~item
    return result.contiguous()


def _state_specs(
    population: CoverageStateBoundedPopulation,
) -> tuple[tuple[_CoordinateState, ...], tuple[_CoordinateState, ...]]:
    """Return target-bearing states first, then every role-bearing context."""

    cache = population.cache
    factual_miss = sorted(
        (
            value
            for value in cache.natural_records
            if value.record.state_kind == "factual_miss"
        ),
        key=lambda value: value.record.record_id,
    )
    factual_no_miss = sorted(
        (
            value
            for value in cache.natural_records
            if value.record.state_kind == "factual_no_miss"
        ),
        key=lambda value: value.record.record_id,
    )
    clean = sorted(
        cache.clean_positive_records,
        key=lambda value: value.record.pair_id,
    )
    component = sorted(
        cache.component_null_records,
        key=lambda value: value.record.pair_id,
    )

    target_states: list[_CoordinateState] = []
    context_states: list[_CoordinateState] = []
    for value in factual_miss:
        record = value.record
        target = _mask(
            value.targets.focus_support,
            occupancy=record.occupancy,
            valid=record.valid_mask,
        )
        background = _mask(
            torch.ones_like(record.valid_mask),
            occupancy=record.occupancy,
            valid=record.valid_mask,
            excluded=(record.target,),
        )
        state = _CoordinateState(
            state_id=f"natural:{record.record_id}",
            sample_id=record.sample_id,
            state_kind="factual_miss",
            endpoint="natural",
            feature=record.feature,
            occupancy=record.occupancy,
            valid_mask=record.valid_mask,
            target_mask=target,
            background_mask=background,
            component_mask=torch.zeros_like(target),
            target_group_id=f"factual:{record.record_id}",
            component_writable=False,
        )
        target_states.append(state)
        context_states.append(state)

    for value in factual_no_miss:
        record = value.record
        empty = torch.zeros_like(record.valid_mask)
        context_states.append(
            _CoordinateState(
                state_id=f"natural:{record.record_id}",
                sample_id=record.sample_id,
                state_kind="factual_no_miss",
                endpoint="natural",
                feature=record.feature,
                occupancy=record.occupancy,
                valid_mask=record.valid_mask,
                target_mask=empty,
                background_mask=_mask(
                    torch.ones_like(record.valid_mask),
                    occupancy=record.occupancy,
                    valid=record.valid_mask,
                ),
                component_mask=empty,
                target_group_id=None,
                component_writable=False,
            )
        )

    for value in clean:
        record = value.record
        added = (
            record.target_minus
            & ~record.target_plus
            & record.valid_mask
        ).contiguous()
        empty = torch.zeros_like(added)
        plus = _CoordinateState(
            state_id=f"pair:{record.pair_id}:plus",
            sample_id=record.sample_id,
            state_kind="clean_positive",
            endpoint="plus",
            feature=record.feature,
            occupancy=record.occupancy_plus,
            valid_mask=record.valid_mask,
            target_mask=empty,
            background_mask=_mask(
                torch.ones_like(record.valid_mask),
                occupancy=record.occupancy_plus,
                valid=record.valid_mask,
                excluded=(record.target_plus,),
            ),
            component_mask=empty,
            target_group_id=None,
            component_writable=False,
        )
        minus_target = _mask(
            added,
            occupancy=record.occupancy_minus,
            valid=record.valid_mask,
        )
        minus = _CoordinateState(
            state_id=f"pair:{record.pair_id}:minus",
            sample_id=record.sample_id,
            state_kind="clean_positive",
            endpoint="minus",
            feature=record.feature,
            occupancy=record.occupancy_minus,
            valid_mask=record.valid_mask,
            target_mask=minus_target,
            background_mask=_mask(
                torch.ones_like(record.valid_mask),
                occupancy=record.occupancy_minus,
                valid=record.valid_mask,
                excluded=(record.target_minus,),
            ),
            component_mask=empty,
            target_group_id=f"clean:{record.pair_id}",
            component_writable=False,
        )
        target_states.append(minus)
        context_states.extend((plus, minus))

    for value in component:
        record = value.record
        component_support = (
            record.removed_component & record.valid_mask
        ).contiguous()
        empty = torch.zeros_like(component_support)
        for endpoint, occupancy, endpoint_target, writable in (
            (
                "plus",
                record.occupancy_plus,
                record.target_plus,
                False,
            ),
            (
                "minus",
                record.occupancy_minus,
                record.target_minus,
                True,
            ),
        ):
            context_states.append(
                _CoordinateState(
                    state_id=f"pair:{record.pair_id}:{endpoint}",
                    sample_id=record.sample_id,
                    state_kind="component_null",
                    endpoint=endpoint,
                    feature=record.feature,
                    occupancy=occupancy,
                    valid_mask=record.valid_mask,
                    target_mask=empty,
                    background_mask=_mask(
                        torch.ones_like(record.valid_mask),
                        occupancy=occupancy,
                        valid=record.valid_mask,
                        excluded=(
                            component_support,
                            endpoint_target,
                        ),
                    ),
                    component_mask=component_support,
                    target_group_id=None,
                    component_writable=writable,
                )
            )

    target_result = tuple(
        sorted(target_states, key=lambda value: value.state_id)
    )
    context_result = tuple(
        sorted(context_states, key=lambda value: value.state_id)
    )
    return target_result, context_result


def _coordinate_indices(mask: Tensor) -> Tensor:
    return torch.nonzero(mask[:, 0], as_tuple=False).to("cpu")


def _row_bit_hash(value: Tensor) -> Tensor:
    rows = value.detach().to("cpu", dtype=torch.float32).contiguous()
    if rows.ndim != 2:
        raise ValueError("row hash requires a matrix")
    bits = rows.view(torch.int32).to(torch.int64)
    coefficients = (
        torch.arange(
            1,
            rows.shape[1] + 1,
            dtype=torch.int64,
        )
        * 0x1F123BB5
        + 0x05F35649
    )
    # Signed int64 overflow is deterministic in the tensor kernel.  The hash
    # is only a candidate filter; every match is rechecked by exact bytes.
    return (bits * coefficients).sum(dim=1)


def _metric_vectors(basis: _HiddenBasis, mask: Tensor) -> dict[str, Tensor]:
    vectors = {
        "h0": _vectors_at(basis.h0, mask),
        "h1": _vectors_at(basis.h1, mask),
        "hm": _vectors_at(basis.hm, mask),
        "odd": _vectors_at(basis.odd, mask),
        "even": _vectors_at(basis.even, mask),
        "oriented_odd": _vectors_at(basis.oriented_odd, mask),
    }
    if not vectors["odd"].numel():
        raise ValueError("coordinate role has no vectors")
    norms = {
        name: torch.linalg.vector_norm(value, dim=1)
        for name, value in vectors.items()
        if name != "oriented_odd"
    }
    epsilon = torch.full(
        (),
        COVERAGE_STATE_BFA_DR_FLOAT32_EPSILON,
        dtype=torch.float32,
        device=norms["odd"].device,
    )
    norms["rho"] = norms["even"] / (norms["odd"] + epsilon)
    return {**vectors, **{f"{name}_norm": value for name, value in norms.items()}}


def _role_row(
    *,
    state: _CoordinateState,
    role: str,
    mask: Tensor,
    metrics: dict[str, Tensor],
) -> dict[str, object]:
    return {
        "state_id": state.state_id,
        "sample_id": state.sample_id,
        "state_kind": state.state_kind,
        "endpoint": state.endpoint,
        "role": role,
        "coordinate_count": int(torch.count_nonzero(mask)),
        "mask_fingerprint": tensor_content_fingerprint(mask),
        "h0_vector_fingerprint": tensor_content_fingerprint(metrics["h0"]),
        "h1_vector_fingerprint": tensor_content_fingerprint(metrics["h1"]),
        "hm_vector_fingerprint": tensor_content_fingerprint(metrics["hm"]),
        "odd_vector_fingerprint": tensor_content_fingerprint(metrics["odd"]),
        "even_vector_fingerprint": tensor_content_fingerprint(metrics["even"]),
        "oriented_odd_vector_fingerprint": tensor_content_fingerprint(
            metrics["oriented_odd"]
        ),
        "h0_norm_stream": _stream_summary(
            metrics["h0_norm"],
            name=f"{state.state_id} {role} H0 norm",
        ),
        "h1_norm_stream": _stream_summary(
            metrics["h1_norm"],
            name=f"{state.state_id} {role} H1 norm",
        ),
        "hm_norm_stream": _stream_summary(
            metrics["hm_norm"],
            name=f"{state.state_id} {role} Hm norm",
        ),
        "odd_norm_stream": _stream_summary(
            metrics["odd_norm"],
            name=f"{state.state_id} {role} odd norm",
        ),
        "even_norm_stream": _stream_summary(
            metrics["even_norm"],
            name=f"{state.state_id} {role} even norm",
        ),
        "rho_stream": _stream_summary(
            metrics["rho_norm"],
            name=f"{state.state_id} {role} rho",
        ),
        "component_writable": (
            state.component_writable if role == "component" else None
        ),
    }


def _aggregate_distributions(
    values: dict[str, dict[str, list[Tensor]]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for role in ("target", "background", "component"):
        role_values = values[role]
        if not role_values["odd"]:
            raise ValueError(f"BFA D_R has no {role} coordinates")
        result[role] = {
            name: _distribution(
                torch.cat(tensors, dim=0),
                name=f"aggregate {role} {name}",
            )
            for name, tensors in role_values.items()
        }
    return result


def _representation_probe(
    model: CURELiteBinaryFlipAntisymmetricLevelSet,
    population: CoverageStateBoundedPopulation,
    *,
    device: torch.device,
) -> dict[str, object]:
    target_states, context_states = _state_specs(population)
    metric_names = (
        "h0",
        "h1",
        "hm",
        "odd",
        "even",
        "rho",
    )
    collected: dict[str, dict[str, list[Tensor]]] = {
        role: {name: [] for name in metric_names}
        for role in ("target", "background", "component")
    }
    rows: list[dict[str, object]] = []
    target_groups: list[dict[str, object]] = []
    target_keys: dict[str, list[dict[str, object]]] = {}
    target_hashes: set[int] = set()
    target_coordinate_count = 0
    unique_state_forward_count = 0
    reused_basis_count = 0
    processed_state_ids: set[str] = set()
    pending_positive: list[
        tuple[_CoordinateState, str, Tensor, Tensor]
    ] = []
    context_by_id = {value.state_id: value for value in context_states}
    if len(context_by_id) != len(context_states):
        raise ValueError("BFA D_R context state IDs are not unique")

    def evaluate(state: _CoordinateState) -> _HiddenBasis:
        nonlocal unique_state_forward_count
        feature = state.feature.to(
            device=device,
            dtype=torch.float32,
            non_blocking=False,
        )
        occupancy = state.occupancy.to(
            device=device,
            non_blocking=False,
        )
        with torch.no_grad():
            result = _hidden_basis(model, feature, occupancy)
        unique_state_forward_count += 1
        return result

    def consume_positive_roles(
        state: _CoordinateState,
        basis: _HiddenBasis,
        *,
        defer_conflict_scan: bool,
    ) -> list[tuple[_CoordinateState, str, Tensor, Tensor]]:
        deferred: list[
            tuple[_CoordinateState, str, Tensor, Tensor]
        ] = []
        for role, mask_cpu in (
            ("background", state.background_mask),
            ("component", state.component_mask),
        ):
            if not bool(torch.any(mask_cpu)):
                continue
            metrics = _metric_vectors(
                basis,
                mask_cpu.to(device=device),
            )
            odd_cpu = metrics["odd"].detach().to("cpu").contiguous()
            coordinates = _coordinate_indices(mask_cpu)
            for name in metric_names:
                collected[role][name].append(
                    metrics[f"{name}_norm"].detach().to("cpu")
                )
            rows.append(
                _role_row(
                    state=state,
                    role=role,
                    mask=mask_cpu,
                    metrics=metrics,
                )
            )
            if defer_conflict_scan:
                deferred.append(
                    (state, role, odd_cpu, coordinates)
                )
            else:
                _scan_positive(
                    state,
                    role,
                    odd_cpu,
                    coordinates,
                )
        return deferred

    # Target-bearing contexts come first.  The one basis created for a state
    # is consumed immediately by both its target and background roles, then
    # released; no multi-gigabyte HiddenBasis cache is retained.
    for state in target_states:
        basis = evaluate(state)
        metrics = _metric_vectors(basis, state.target_mask.to(device=device))
        odd = metrics["odd"].detach().to("cpu").contiguous()
        coordinates = _coordinate_indices(state.target_mask)
        if odd.shape[0] != coordinates.shape[0]:
            raise AssertionError("target coordinate/vector counts differ")
        nonzero = bool(torch.any(odd != 0.0))
        target_groups.append(
            {
                "target_group_id": state.target_group_id,
                "state_id": state.state_id,
                "sample_id": state.sample_id,
                "state_kind": state.state_kind,
                "coordinate_count": int(odd.shape[0]),
                "finite": bool(torch.isfinite(odd).all()),
                "at_least_one_odd_coordinate_nonzero": nonzero,
                "odd_vector_fingerprint": tensor_content_fingerprint(odd),
            }
        )
        hashes = _row_bit_hash(odd)
        for index in range(odd.shape[0]):
            vector = odd[index].contiguous()
            key = tensor_content_fingerprint(vector)
            coordinate = coordinates[index].tolist()
            entry = {
                "target_group_id": state.target_group_id,
                "state_id": state.state_id,
                "coordinate": coordinate,
            }
            target_keys.setdefault(key, []).append(entry)
            target_hashes.add(int(hashes[index].item()))
        for name in metric_names:
            collected["target"][name].append(
                metrics[f"{name}_norm"].detach().to("cpu")
            )
        rows.append(
            _role_row(
                state=state,
                role="target",
                mask=state.target_mask,
                metrics=metrics,
            )
        )
        target_coordinate_count += int(odd.shape[0])

        context = context_by_id.get(state.state_id)
        if context is None:
            raise AssertionError("target state is absent from contexts")
        pending_positive.extend(
            consume_positive_roles(
                context,
                basis,
                defer_conflict_scan=True,
            )
        )
        reused_basis_count += 1
        processed_state_ids.add(state.state_id)

    conflict_count = 0
    conflict_examples: list[dict[str, object]] = []
    conflicting_vector_keys: set[str] = set()
    target_hash_tensor = torch.tensor(
        sorted(target_hashes),
        dtype=torch.int64,
    )

    def _scan_positive(
        state: _CoordinateState,
        role: str,
        odd_cpu: Tensor,
        coordinates: Tensor,
    ) -> None:
        nonlocal conflict_count
        hashes = _row_bit_hash(odd_cpu)
        candidate_indices = torch.nonzero(
            torch.isin(hashes, target_hash_tensor),
            as_tuple=False,
        ).flatten()
        for index_tensor in candidate_indices:
            index = int(index_tensor.item())
            key = tensor_content_fingerprint(
                odd_cpu[index].contiguous()
            )
            target_entries = target_keys.get(key, ())
            if not target_entries:
                continue
            conflicting_vector_keys.add(key)
            conflict_count += len(target_entries)
            if len(conflict_examples) >= 64:
                continue
            for target_entry in target_entries:
                if len(conflict_examples) >= 64:
                    break
                conflict_examples.append(
                    {
                        "odd_vector_fingerprint": key,
                        "negative_requirement": deepcopy(
                            target_entry
                        ),
                        "positive_requirement": {
                            "state_id": state.state_id,
                            "sample_id": state.sample_id,
                            "state_kind": state.state_kind,
                            "endpoint": state.endpoint,
                            "role": role,
                            "coordinate": coordinates[index].tolist(),
                            "component_writable": (
                                state.component_writable
                                if role == "component"
                                else None
                            ),
                        },
                    }
                )

    # The target dictionary is now complete, so deferred background streams
    # can be checked and released before the remaining states are evaluated.
    for state, role, odd_cpu, coordinates in pending_positive:
        _scan_positive(state, role, odd_cpu, coordinates)
    pending_positive.clear()

    for state in context_states:
        if state.state_id in processed_state_ids:
            continue
        basis = evaluate(state)
        consume_positive_roles(
            state,
            basis,
            defer_conflict_scan=False,
        )
        processed_state_ids.add(state.state_id)

    aggregate = _aggregate_distributions(collected)
    odd_square = fsum(
        float.fromhex(
            str(aggregate[role]["odd"]["sum_squares_hex"])
        )
        for role in ("target", "background", "component")
    )
    even_square = fsum(
        float.fromhex(
            str(aggregate[role]["even"]["sum_squares_hex"])
        )
        for role in ("target", "background", "component")
    )
    global_ratio = sqrt(even_square) / (
        sqrt(odd_square) + COVERAGE_STATE_BFA_DR_FLOAT32_EPSILON
    )
    return {
        "coordinate_policy": {
            "factual_target": (
                "bounded_factual_focus_support_writable_under_natural_O"
            ),
            "clean_target": (
                "bounded_clean_minus_added_target_writable_under_O_minus"
            ),
            "background": (
                "all_valid_writable_non_target_non_component_coordinates"
            ),
            "component": (
                "removed_component_coordinates_at_both_pair_endpoints"
            ),
            "conflict_basis": (
                "canonical_o_equals_half_H0_minus_H1_float32_exact_bytes"
            ),
            "target_interval": "d_le_negative_1p125",
            "background_interval": "d_ge_negative_0p675",
            "component_interval": "abs_d_le_0p675",
        },
        "state_counts": {
            "target_states": len(target_states),
            "context_states": len(context_states),
            "unique_context_states": len(context_by_id),
            "unique_state_forward_count": unique_state_forward_count,
            "reused_basis_count": reused_basis_count,
            "factual_target_groups": sum(
                row["state_kind"] == "factual_miss"
                for row in target_groups
            ),
            "clean_target_groups": sum(
                row["state_kind"] == "clean_positive"
                for row in target_groups
            ),
        },
        "coordinate_counts": {
            role: int(aggregate[role]["odd"]["count"])
            for role in ("target", "background", "component")
        },
        "target_group_rows": target_groups,
        "coordinate_rows": rows,
        "aggregate_distributions": aggregate,
        "global_odd_sum_squares_hex": _hex(
            odd_square,
            name="global odd sum squares",
        ),
        "global_curvature_sum_squares_hex": _hex(
            even_square,
            name="global curvature sum squares",
        ),
        "global_curvature_to_odd_ratio_hex": _hex(
            global_ratio,
            name="global curvature-to-odd ratio",
        ),
        "ratio_threshold_hex": (
            COVERAGE_STATE_BFA_DR_RATIO_THRESHOLD.hex()
        ),
        "ratio_threshold_policy": (
            "strictly_greater_than_128_times_float32_epsilon"
        ),
        "target_coordinate_count": target_coordinate_count,
        "unique_target_odd_representation_count": len(target_keys),
        "exact_mutually_exclusive_conflict_count": conflict_count,
        "conflicting_vector_fingerprints": sorted(
            conflicting_vector_keys
        ),
        "conflict_examples": conflict_examples,
        "conflict_examples_truncated": conflict_count > len(
            conflict_examples
        ),
    }


def _descent_row(
    *,
    role: str,
    state_id: str,
    sample_id: str,
    endpoint: str,
    loss: Tensor,
    field: Tensor,
    mask: Tensor,
    desired: str,
    loss_api: str,
) -> dict[str, object]:
    gradient = torch.autograd.grad(
        loss,
        field,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]
    selected = (-gradient)[mask]
    if selected.numel() < 1:
        raise ValueError(f"BFA {role} descent mask is empty")
    negative = int(torch.count_nonzero(selected < 0.0))
    positive = int(torch.count_nonzero(selected > 0.0))
    zero = int(torch.count_nonzero(selected == 0.0))
    total = float(selected.to(torch.float64).sum().detach().cpu())
    correct = total < 0.0 if desired == "negative" else total > 0.0
    return {
        "role": role,
        "state_id": state_id,
        "sample_id": sample_id,
        "endpoint": endpoint,
        "coordinate_count": int(selected.numel()),
        "actual_mask_fingerprint": tensor_content_fingerprint(mask),
        "loss_api": loss_api,
        "loss_hex": _hex(
            float(loss.detach().cpu()),
            name=f"BFA {role} direction loss",
        ),
        "descent_sum_hex": _hex(
            total,
            name=f"BFA {role} descent sum",
        ),
        "descent_negative_count": negative,
        "descent_positive_count": positive,
        "descent_zero_count": zero,
        "descent_finite": bool(torch.isfinite(selected).all()),
        "descent_nonzero": negative + positive > 0,
        "desired_field_direction": desired,
        "aggregate_descent_direction_correct": correct,
    }


def _absolute_targets_to_device(
    value: CoverageStateAbsoluteTargets,
    *,
    device: torch.device,
) -> CoverageStateAbsoluteTargets:
    result = CoverageStateAbsoluteTargets(
        target_field=value.target_field.to(device=device),
        integration_measure=value.integration_measure.to(device=device),
        field_valid_mask=value.field_valid_mask.to(device=device),
        loss_valid_mask=value.loss_valid_mask.to(device=device),
        focus_support=value.focus_support.to(device=device),
        focus_support_field=value.focus_support_field.to(device=device),
    )
    result.validate()
    return result


def _direction_probe(
    population: CoverageStateBoundedPopulation,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Use frozen losses on actual geometry with fixed violating fields."""

    factual_miss = sorted(
        (
            value
            for value in population.cache.natural_records
            if value.record.state_kind == "factual_miss"
        ),
        key=lambda value: value.record.record_id,
    )
    factual_no_miss = sorted(
        (
            value
            for value in population.cache.natural_records
            if value.record.state_kind == "factual_no_miss"
        ),
        key=lambda value: value.record.record_id,
    )
    clean = sorted(
        population.cache.clean_positive_records,
        key=lambda value: value.record.pair_id,
    )
    component = sorted(
        population.cache.component_null_records,
        key=lambda value: value.record.pair_id,
    )
    amplitude = float(
        CoverageStateBinaryFlipAntisymmetricConfig(
            feature_channels=COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
            feature_stride=COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
            width=COVERAGE_STATE_CMIF_FORMAL_WIDTH,
        ).field_amplitude
    )
    rows: list[dict[str, object]] = []

    # Natural factual targets: preserve the precomputed absolute target field
    # everywhere except the actual focus support, where a fixed positive field
    # violates the required negative target sign.
    for value in factual_miss:
        record = value.record
        targets = _absolute_targets_to_device(
            value.targets,
            device=device,
        )
        mask = (
            value.targets.focus_support
            & record.valid_mask
            & ~record.occupancy
        ).to(device=device)
        witness = targets.target_field.detach().clone()
        witness[mask] = amplitude
        witness.requires_grad_(True)
        result = coverage_state_absolute_sobolev_loss_from_targets(
            witness,
            targets,
            config=population.cache.sobolev_config,
            validate=True,
        )
        rows.append(
            _descent_row(
                role="factual_target",
                state_id=record.record_id,
                sample_id=record.sample_id,
                endpoint="natural",
                loss=result.loss,
                field=witness,
                mask=mask,
                desired="negative",
                loss_api=(
                    "coverage_state_absolute_sobolev_loss_from_targets"
                ),
            )
        )

    # Natural no-miss states provide actual writable background.  The exact
    # target field is retained except that all writable valid coordinates are
    # fixed to a negative violating value.
    for value in factual_no_miss:
        record = value.record
        targets = _absolute_targets_to_device(
            value.targets,
            device=device,
        )
        mask = (
            record.valid_mask & ~record.occupancy
        ).to(device=device)
        witness = targets.target_field.detach().clone()
        witness[mask] = -amplitude
        witness.requires_grad_(True)
        result = coverage_state_absolute_sobolev_loss_from_targets(
            witness,
            targets,
            config=population.cache.sobolev_config,
            validate=True,
        )
        rows.append(
            _descent_row(
                role="writable_background",
                state_id=record.record_id,
                sample_id=record.sample_id,
                endpoint="natural",
                loss=result.loss,
                field=witness,
                mask=mask,
                desired="positive",
                loss_api=(
                    "coverage_state_absolute_sobolev_loss_from_targets"
                ),
            )
        )

    # Clean-minus added support uses the actual two-endpoint PMOPE geometry.
    # Only the added target is overwritten with the fixed wrong positive sign.
    for value in clean:
        record = value.record
        targets = _pair_targets_to_device(
            value.joint_targets,
            device=device,
        )
        mask = (
            record.target_minus
            & ~record.target_plus
            & record.valid_mask
            & ~record.occupancy_minus
        ).to(device=device)
        plus = targets.target_field_plus.detach().clone()
        minus = targets.target_field_minus.detach().clone()
        minus[mask] = amplitude
        plus.requires_grad_(True)
        minus.requires_grad_(True)
        result = coverage_state_pmope_pair_loss_from_targets(
            plus,
            minus,
            targets,
            config=population.cache.sobolev_config,
            validate=True,
        )
        rows.append(
            _descent_row(
                role="clean_target",
                state_id=record.pair_id,
                sample_id=record.sample_id,
                endpoint="minus",
                loss=result.loss,
                field=minus,
                mask=mask,
                desired="negative",
                loss_api="coverage_state_pmope_pair_loss_from_targets",
            )
        )

    # Component-null minus is writable after deletion and must remain
    # exterior.  Set exactly the removed support to the wrong negative sign.
    for value in component:
        record = value.record
        targets = _pair_targets_to_device(
            value.joint_targets,
            device=device,
        )
        mask = (
            record.removed_component
            & record.valid_mask
            & ~record.occupancy_minus
        ).to(device=device)
        plus = targets.target_field_plus.detach().clone()
        minus = targets.target_field_minus.detach().clone()
        minus[mask] = -amplitude
        plus.requires_grad_(True)
        minus.requires_grad_(True)
        result = coverage_state_pmope_pair_loss_from_targets(
            plus,
            minus,
            targets,
            config=population.cache.sobolev_config,
            validate=True,
        )
        rows.append(
            _descent_row(
                role="writable_component",
                state_id=record.pair_id,
                sample_id=record.sample_id,
                endpoint="minus",
                loss=result.loss,
                field=minus,
                mask=mask,
                desired="positive",
                loss_api="coverage_state_pmope_pair_loss_from_targets",
            )
        )

    expected_counts = {
        "factual_target": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "clean_target": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "writable_background": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "writable_component": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
    }
    actual_counts = {
        role: sum(row["role"] == role for row in rows)
        for role in expected_counts
    }
    return {
        "witness_policy": (
            "actual_precomputed_geometry_fixed_wrong_sign_field_no_model_update_v1"
        ),
        "fixed_field_amplitude_hex": amplitude.hex(),
        "loss_apis": [
            "coverage_state_absolute_sobolev_loss_from_targets",
            "coverage_state_pmope_pair_loss_from_targets",
        ],
        "expected_role_rows": expected_counts,
        "actual_role_rows": actual_counts,
        "rows": rows,
        "all_roles_finite_nonzero_correct": (
            actual_counts == expected_counts
            and all(
                row["descent_finite"]
                and row["descent_nonzero"]
                and row["aggregate_descent_direction_correct"]
                and float.fromhex(str(row["loss_hex"])) > 0.0
                for row in rows
            )
        ),
        "uses_actual_target_geometry": True,
        "uses_actual_valid_and_writable_masks": True,
        "model_parameter_gradient": False,
    }


def _readout_probe(
    model: CURELiteBinaryFlipAntisymmetricLevelSet,
    population: CoverageStateBoundedPopulation,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Measure update-zero readout gradients on actual factual/clean targets."""

    factual = sorted(
        (
            value
            for value in population.cache.natural_records
            if value.record.state_kind == "factual_miss"
        ),
        key=lambda value: value.record.record_id,
    )
    clean = sorted(
        population.cache.clean_positive_records,
        key=lambda value: value.record.pair_id,
    )
    rows: list[dict[str, object]] = []
    for value in factual:
        record = value.record
        feature = record.feature.to(device=device, dtype=torch.float32)
        occupancy = record.occupancy.to(device=device)
        targets = _absolute_targets_to_device(
            value.targets,
            device=device,
        )
        field = model(feature, occupancy)
        result = coverage_state_absolute_sobolev_loss_from_targets(
            field,
            targets,
            config=population.cache.sobolev_config,
            validate=True,
        )
        gradient = torch.autograd.grad(
            result.loss,
            model.scalar_energy_weight,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        rows.append(
            {
                "kind": "factual_miss",
                "record_id": record.record_id,
                "sample_id": record.sample_id,
                "loss_hex": _hex(
                    float(result.loss.detach().cpu()),
                    name="BFA factual update-zero loss",
                ),
                "readout_gradient_l2_hex": _hex(
                    float(gradient.detach().norm().cpu()),
                    name="BFA factual readout gradient",
                ),
                "readout_gradient_finite": bool(
                    torch.isfinite(gradient).all()
                ),
                "readout_gradient_nonzero": bool(
                    torch.any(gradient != 0.0)
                ),
            }
        )

    for value in clean:
        record = value.record
        feature = torch.cat(
            (record.feature, record.feature),
            dim=0,
        ).to(device=device, dtype=torch.float32)
        occupancy = torch.cat(
            (record.occupancy_plus, record.occupancy_minus),
            dim=0,
        ).to(device=device)
        targets = _pair_targets_to_device(
            value.joint_targets,
            device=device,
        )
        field_plus, field_minus = model(feature, occupancy).split(1, dim=0)
        result = coverage_state_pmope_pair_loss_from_targets(
            field_plus,
            field_minus,
            targets,
            config=population.cache.sobolev_config,
            validate=True,
        )
        gradient = torch.autograd.grad(
            result.loss,
            model.scalar_energy_weight,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        rows.append(
            {
                "kind": "clean_positive",
                "record_id": record.pair_id,
                "sample_id": record.sample_id,
                "loss_hex": _hex(
                    float(result.loss.detach().cpu()),
                    name="BFA clean update-zero loss",
                ),
                "readout_gradient_l2_hex": _hex(
                    float(gradient.detach().norm().cpu()),
                    name="BFA clean readout gradient",
                ),
                "readout_gradient_finite": bool(
                    torch.isfinite(gradient).all()
                ),
                "readout_gradient_nonzero": bool(
                    torch.any(gradient != 0.0)
                ),
            }
        )
    return {
        "rows": rows,
        "factual_row_count": len(factual),
        "clean_row_count": len(clean),
        "all_losses_positive": all(
            float.fromhex(str(row["loss_hex"])) > 0.0 for row in rows
        ),
        "all_readout_gradients_finite_nonzero": all(
            bool(row["readout_gradient_finite"])
            and bool(row["readout_gradient_nonzero"])
            and float.fromhex(str(row["readout_gradient_l2_hex"])) > 0.0
            for row in rows
        ),
    }


def _pair_targets_to_device(
    value: CoverageStatePairTargets,
    *,
    device: torch.device,
) -> CoverageStatePairTargets:
    result = CoverageStatePairTargets(
        target_field_plus=value.target_field_plus.to(device=device),
        target_field_minus=value.target_field_minus.to(device=device),
        focus_support=value.focus_support.to(device=device),
        focus_support_field=value.focus_support_field.to(device=device),
        integration_measure=value.integration_measure.to(device=device),
        valid_mask=value.valid_mask.to(device=device),
    )
    result.validate()
    return result


def _nonzero_readout_joint_probe(
    model: CURELiteBinaryFlipAntisymmetricLevelSet,
    population: CoverageStateBoundedPopulation,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Use functional parameters to expose joint weight/bias gradients."""

    clean = sorted(
        population.cache.clean_positive_records,
        key=lambda value: value.record.pair_id,
    )
    if not clean:
        raise ValueError("BFA D_R joint witness requires a clean pair")
    value = clean[0]
    record = value.record
    feature = torch.cat(
        (record.feature, record.feature),
        dim=0,
    ).to(device=device, dtype=torch.float32)
    occupancy = torch.cat(
        (record.occupancy_plus, record.occupancy_minus),
        dim=0,
    ).to(device=device)
    targets = _pair_targets_to_device(
        value.joint_targets,
        device=device,
    )
    original_parameter_ids = {
        name: id(parameter)
        for name, parameter in model.named_parameters()
    }
    joint = model.joint_state_weight.detach().clone().requires_grad_(True)
    bias = model.joint_hidden_bias.detach().clone().requires_grad_(True)
    readout = torch.linspace(
        0.5,
        1.5,
        model.config.width,
        device=device,
        dtype=torch.float32,
    )
    parameters = {
        "joint_state_weight": joint,
        "joint_hidden_bias": bias,
        "scalar_energy_weight": readout,
    }
    functional_field = torch.func.functional_call(
        model,
        parameters,
        (feature, occupancy),
        strict=True,
    )
    plus, minus = functional_field.split(1, dim=0)
    result = coverage_state_pmope_pair_loss_from_targets(
        plus,
        minus,
        targets,
        config=population.cache.sobolev_config,
        validate=True,
    )
    joint_gradient, bias_gradient = torch.autograd.grad(
        result.loss,
        (joint, bias),
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    return {
        "pair_id": record.pair_id,
        "sample_id": record.sample_id,
        "readout_policy": (
            "fixed_positive_linspace_0p5_to_1p5_width32_no_model_mutation"
        ),
        "readout_fingerprint": tensor_content_fingerprint(readout),
        "loss_hex": _hex(
            float(result.loss.detach().cpu()),
            name="BFA nonzero-readout witness loss",
        ),
        "joint_weight_gradient_l2_hex": _hex(
            float(joint_gradient.detach().norm().cpu()),
            name="BFA joint-weight witness gradient",
        ),
        "joint_bias_gradient_l2_hex": _hex(
            float(bias_gradient.detach().norm().cpu()),
            name="BFA joint-bias witness gradient",
        ),
        "joint_weight_gradient_finite": bool(
            torch.isfinite(joint_gradient).all()
        ),
        "joint_bias_gradient_finite": bool(
            torch.isfinite(bias_gradient).all()
        ),
        "joint_weight_gradient_nonzero": bool(
            torch.any(joint_gradient != 0.0)
        ),
        "joint_bias_gradient_nonzero": bool(
            torch.any(bias_gradient != 0.0)
        ),
        "functional_call_did_not_replace_model_parameters": all(
            original_parameter_ids.get(name) == id(parameter)
            for name, parameter in model.named_parameters()
        ),
    }


def _probe(
    population: CoverageStateBoundedPopulation,
    *,
    device: torch.device,
) -> dict[str, object]:
    before_cpu_rng = torch.random.get_rng_state().clone()
    before_device_rng = (
        None
        if device.type != "cuda"
        else torch.cuda.get_rng_state(device).clone()
    )
    before_population = population.population_fingerprint
    before_cache = population.cache.cache_fingerprint
    runtime_splits = sorted(
        {
            population.source_cache.raw_catalog.split,
            population.cache.raw_catalog.split,
        }
    )
    with torch.random.fork_rng(
        devices=_cuda_rng_devices(device),
        device_type=("cuda" if device.type == "cuda" else None),
    ), torch.autocast(device_type=device.type, enabled=False):
        torch.random.default_generator.manual_seed(
            COVERAGE_STATE_BFA_DR_EXECUTION_SEED
        )
        config = CoverageStateBinaryFlipAntisymmetricConfig(
            feature_channels=COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
            feature_stride=COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
            width=COVERAGE_STATE_CMIF_FORMAL_WIDTH,
        )
        model = CURELiteBinaryFlipAntisymmetricLevelSet(config).to(
            device=device,
            dtype=torch.float32,
        )
        model.eval()
        initial_model = _model_fingerprint(model)
        representation = _representation_probe(
            model,
            population,
            device=device,
        )
        direction = _direction_probe(
            population,
            device=device,
        )
        readout = _readout_probe(
            model,
            population,
            device=device,
        )
        joint = _nonzero_readout_joint_probe(
            model,
            population,
            device=device,
        )
        final_model = _model_fingerprint(model)
        parameter_contract = [
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": parameter.requires_grad,
            }
            for name, parameter in model.named_parameters()
        ]
        parameter_grad_buffers_unretained = all(
            parameter.grad is None for parameter in model.parameters()
        )
        model_config = {
            "model_class": type(model).__name__,
            "feature_channels": config.feature_channels,
            "feature_stride": config.feature_stride,
            "width": config.width,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "parameter_tensor_count": len(
                tuple(model.parameters())
            ),
            "field_policy": config.field_policy,
            "equation_policy": config.equation_policy,
            "flip_policy": config.flip_policy,
            "margin_hex": COVERAGE_STATE_BFA_MARGIN.hex(),
        }
    population.verify_unchanged()
    after_device_rng = (
        None
        if device.type != "cuda"
        else torch.cuda.get_rng_state(device).clone()
    )
    return {
        "device": str(device),
        "execution_seed": COVERAGE_STATE_BFA_DR_EXECUTION_SEED,
        "runtime_splits": runtime_splits,
        "split_access_policy": (
            "already_built_real_D_R_inputs_and_seed42_bounded_population_only"
        ),
        "model_config": model_config,
        "parameter_contract": parameter_contract,
        "initial_model_fingerprint": initial_model,
        "final_model_fingerprint": final_model,
        "population_fingerprint_before": before_population,
        "population_fingerprint_after": population.population_fingerprint,
        "cache_fingerprint_before": before_cache,
        "cache_fingerprint_after": population.cache.cache_fingerprint,
        "representation": representation,
        "field_direction": direction,
        "update_zero_readout": readout,
        "nonzero_readout_joint_witness": joint,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_performed": False,
        "calibration_performed": False,
        "inference_performed": False,
        "historical_failure_coordinate_inputs": [],
        "parameter_grad_buffers_unretained": (
            parameter_grad_buffers_unretained
        ),
        "global_cpu_rng_preserved": torch.equal(
            before_cpu_rng,
            torch.random.get_rng_state(),
        ),
        "selected_device_rng_preserved": (
            before_device_rng is None
            or torch.equal(before_device_rng, after_device_rng)
        ),
    }


def recompute_coverage_state_bfa_dr_checks(
    *,
    dataset_free_receipt: CoverageStateBFADatasetFreeReceipt,
    real_inputs: CoverageStateRealDRInputs,
    bounded_population: CoverageStateBoundedPopulation,
    probe: dict[str, object],
) -> tuple[tuple[str, bool], ...]:
    representation = probe.get("representation", {})
    state_counts = (
        representation.get("state_counts", {})
        if isinstance(representation, dict)
        else {}
    )
    coordinate_counts = (
        representation.get("coordinate_counts", {})
        if isinstance(representation, dict)
        else {}
    )
    target_rows = (
        representation.get("target_group_rows", [])
        if isinstance(representation, dict)
        else []
    )
    aggregate = (
        representation.get("aggregate_distributions", {})
        if isinstance(representation, dict)
        else {}
    )
    direction = probe.get("field_direction", {})
    readout = probe.get("update_zero_readout", {})
    joint = probe.get("nonzero_readout_joint_witness", {})
    model_config = probe.get("model_config", {})
    parameter_contract = probe.get("parameter_contract", [])

    distributions_valid = (
        isinstance(aggregate, dict)
        and set(aggregate) == {"target", "background", "component"}
        and all(
            isinstance(aggregate.get(role), dict)
            and set(aggregate[role])
            == {"h0", "h1", "hm", "odd", "even", "rho"}
            and all(
                isinstance(aggregate[role].get(name), dict)
                and int(
                    aggregate[role][name].get("count", 0)
                )
                == int(coordinate_counts.get(role, -1))
                and int(
                    aggregate[role][name].get("finite_count", 0)
                )
                == int(coordinate_counts.get(role, -1))
                and isinstance(
                    aggregate[role][name].get(
                        "ordered_value_fingerprint"
                    ),
                    str,
                )
                for name in ("h0", "h1", "hm", "odd", "even", "rho")
            )
            for role in ("target", "background", "component")
        )
    )
    checks = {
        "dataset_free_gate_passed": (
            dataset_free_receipt.all_pass is True
        ),
        "real_inputs_and_population_are_exact_D_R": (
            real_inputs.source_binding.dataset == "IRSTD-1K"
            and real_inputs.source_binding.split == "D_R"
            and real_inputs.bundle.split == "D_R"
            and real_inputs.raw_catalog.split == "D_R"
            and real_inputs.scalar_cache.raw_catalog.split == "D_R"
            and bounded_population.source_cache is real_inputs.scalar_cache
            and bounded_population.cache.raw_catalog.split == "D_R"
            and probe.get("runtime_splits") == ["D_R"]
        ),
        "fixed_seed42_bounded_population": (
            bounded_population.seed
            == COVERAGE_STATE_BFA_DR_EXECUTION_SEED
            == COVERAGE_STATE_BOUNDED_SEED
            and state_counts.get("factual_target_groups")
            == COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and state_counts.get("clean_target_groups")
            == COVERAGE_STATE_BOUNDED_ROLE_COUNT
        ),
        "each_unique_state_forwarded_once": (
            state_counts.get("unique_context_states")
            == state_counts.get("context_states")
            == state_counts.get("unique_state_forward_count")
            and state_counts.get("reused_basis_count")
            == state_counts.get("target_states")
            == 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
        ),
        "actual_role_coordinates_nonempty": (
            isinstance(coordinate_counts, dict)
            and all(
                int(coordinate_counts.get(role, 0)) > 0
                for role in ("target", "background", "component")
            )
        ),
        "complete_hidden_norm_distributions_bound": distributions_valid,
        "odd_and_curvature_finite_nondegenerate": (
            distributions_valid
            and float.fromhex(
                str(
                    representation.get(
                        "global_odd_sum_squares_hex"
                    )
                )
            )
            > 0.0
            and int(
                aggregate["target"]["even"].get(
                    "nonzero_count",
                    0,
                )
            )
            > 0
            and sum(
                int(
                    aggregate[role]["even"].get("nonzero_count", 0)
                )
                for role in ("target", "background", "component")
            )
            > 0
            and float.fromhex(
                str(
                    representation.get(
                        "global_curvature_to_odd_ratio_hex"
                    )
                )
            )
            > COVERAGE_STATE_BFA_DR_RATIO_THRESHOLD
            and representation.get("ratio_threshold_hex")
            == COVERAGE_STATE_BFA_DR_RATIO_THRESHOLD.hex()
        ),
        "every_target_group_has_finite_nonzero_odd_basis": (
            isinstance(target_rows, list)
            and len(target_rows)
            == 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and len(
                {
                    row.get("target_group_id")
                    for row in target_rows
                    if isinstance(row, dict)
                }
            )
            == 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and all(
                isinstance(row, dict)
                and int(row.get("coordinate_count", 0)) > 0
                and row.get("finite") is True
                and row.get(
                    "at_least_one_odd_coordinate_nonzero"
                )
                is True
                for row in target_rows
            )
        ),
        "no_exact_odd_representation_interval_conflict": (
            representation.get(
                "exact_mutually_exclusive_conflict_count"
            )
            == 0
            and representation.get(
                "conflicting_vector_fingerprints"
            )
            == []
            and representation.get("conflict_examples") == []
        ),
        "target_background_component_descent_directions": (
            isinstance(direction, dict)
            and direction.get(
                "all_roles_finite_nonzero_correct"
            )
            is True
            and direction.get("uses_actual_target_geometry") is True
            and direction.get(
                "uses_actual_valid_and_writable_masks"
            )
            is True
            and direction.get("loss_apis")
            == [
                "coverage_state_absolute_sobolev_loss_from_targets",
                "coverage_state_pmope_pair_loss_from_targets",
            ]
            and direction.get("actual_role_rows")
            == {
                "factual_target": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
                "clean_target": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
                "writable_background": (
                    COVERAGE_STATE_BOUNDED_ROLE_COUNT
                ),
                "writable_component": (
                    COVERAGE_STATE_BOUNDED_ROLE_COUNT
                ),
            }
        ),
        "update_zero_readout_gradient_path": (
            isinstance(readout, dict)
            and readout.get("factual_row_count")
            == COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and readout.get("clean_row_count")
            == COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and readout.get("all_losses_positive") is True
            and readout.get(
                "all_readout_gradients_finite_nonzero"
            )
            is True
        ),
        "fixed_nonzero_readout_joint_weight_bias_path": (
            isinstance(joint, dict)
            and joint.get("joint_weight_gradient_finite") is True
            and joint.get("joint_bias_gradient_finite") is True
            and joint.get("joint_weight_gradient_nonzero") is True
            and joint.get("joint_bias_gradient_nonzero") is True
            and float.fromhex(
                str(joint.get("joint_weight_gradient_l2_hex"))
            )
            > 0.0
            and float.fromhex(
                str(joint.get("joint_bias_gradient_l2_hex"))
            )
            > 0.0
            and joint.get(
                "functional_call_did_not_replace_model_parameters"
            )
            is True
        ),
        "fixed_bfa_model_contract": (
            isinstance(model_config, dict)
            and model_config.get("model_class")
            == CURELiteBinaryFlipAntisymmetricLevelSet.__name__
            and model_config.get("feature_channels")
            == COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS
            and model_config.get("feature_stride")
            == COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE
            and model_config.get("width")
            == COVERAGE_STATE_CMIF_FORMAL_WIDTH
            and model_config.get("parameter_count")
            == COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT
            and model_config.get("parameter_tensor_count") == 3
            and model_config.get("field_policy")
            == CSLF_BFA_FIELD_POLICY
            and model_config.get("equation_policy")
            == CSLF_BFA_EQUATION_POLICY
            and model_config.get("flip_policy")
            == CSLF_BFA_FLIP_POLICY
            and model_config.get("margin_hex")
            == COVERAGE_STATE_BFA_MARGIN.hex()
            and isinstance(parameter_contract, list)
            and len(parameter_contract) == 3
            and {
                row.get("name")
                for row in parameter_contract
                if isinstance(row, dict)
            }
            == {
                "joint_state_weight",
                "joint_hidden_bias",
                "scalar_energy_weight",
            }
        ),
        "model_cache_and_rng_unchanged": (
            probe.get("initial_model_fingerprint")
            == probe.get("final_model_fingerprint")
            and probe.get("population_fingerprint_before")
            == probe.get("population_fingerprint_after")
            == bounded_population.population_fingerprint
            and probe.get("cache_fingerprint_before")
            == probe.get("cache_fingerprint_after")
            == bounded_population.cache.cache_fingerprint
            and probe.get("parameter_grad_buffers_unretained") is True
            and probe.get("global_cpu_rng_preserved") is True
            and probe.get("selected_device_rng_preserved") is True
        ),
        "read_only_zero_step_scope": (
            probe.get("execution_seed")
            == COVERAGE_STATE_BFA_DR_EXECUTION_SEED
            and probe.get("optimizer_constructed") is False
            and probe.get("optimizer_steps") == 0
            and probe.get("parameter_updates") == 0
            and probe.get("training_performed") is False
            and probe.get("calibration_performed") is False
            and probe.get("inference_performed") is False
            and probe.get("historical_failure_coordinate_inputs") == []
            and probe.get("runtime_splits") == ["D_R"]
        ),
    }
    return tuple(sorted(checks.items()))


@dataclass(frozen=True, eq=False)
class CoverageStateBFADRGateReceipt:
    """Immutable evidence from one BFA seed-42 real-``D_R`` gate."""

    dataset_free_receipt: CoverageStateBFADatasetFreeReceipt
    real_inputs: CoverageStateRealDRInputs
    bounded_population: CoverageStateBoundedPopulation
    implementation_binding: tuple[tuple[str, str], ...]
    probe: dict[str, object]
    checks: tuple[tuple[str, bool], ...]
    evidence_fingerprint: str

    def __post_init__(self) -> None:
        self.verify_unchanged()

    def _evidence_payload(self) -> dict[str, object]:
        return {
            "dataset_free_receipt_fingerprint": (
                self.dataset_free_receipt.receipt_fingerprint
            ),
            "real_inputs_build_fingerprint": (
                self.real_inputs.build_fingerprint
            ),
            "source_binding_fingerprint": (
                self.real_inputs.source_binding.binding_fingerprint
            ),
            "bounded_population_fingerprint": (
                self.bounded_population.population_fingerprint
            ),
            "bounded_cache_fingerprint": (
                self.bounded_population.cache.cache_fingerprint
            ),
            "implementation_binding": dict(
                self.implementation_binding
            ),
            "probe": deepcopy(self.probe),
        }

    def verify_unchanged(self) -> None:
        self.dataset_free_receipt.verify_unchanged()
        self.real_inputs.verify_unchanged()
        self.bounded_population.verify_unchanged()
        expected = recompute_coverage_state_bfa_dr_checks(
            dataset_free_receipt=self.dataset_free_receipt,
            real_inputs=self.real_inputs,
            bounded_population=self.bounded_population,
            probe=self.probe,
        )
        if (
            self.implementation_binding != _implementation_binding()
            or self.checks != expected
            or stable_fingerprint(self._evidence_payload())
            != self.evidence_fingerprint
        ):
            raise RuntimeError("BFA D_R evidence changed")

    @property
    def all_pass(self) -> bool:
        self.verify_unchanged()
        return bool(self.checks) and all(value for _, value in self.checks)

    @property
    def decision(self) -> str:
        return (
            COVERAGE_STATE_BFA_DR_PASS_DECISION
            if self.all_pass
            else COVERAGE_STATE_BFA_DR_FAIL_DECISION
        )

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        passed = bool(self.checks) and all(
            value for _, value in self.checks
        )
        return {
            "schema_version": COVERAGE_STATE_BFA_DR_GATE_SCHEMA,
            **self._evidence_payload(),
            "checks": dict(self.checks),
            "all_pass": passed,
            "decision": (
                COVERAGE_STATE_BFA_DR_PASS_DECISION
                if passed
                else COVERAGE_STATE_BFA_DR_FAIL_DECISION
            ),
            "execution": {
                "seed": COVERAGE_STATE_BFA_DR_EXECUTION_SEED,
                "runtime_splits": list(
                    self.probe.get("runtime_splits", [])
                ),
                "optimizer_constructed": False,
                "optimizer_steps": 0,
                "parameter_updates": 0,
                "training_performed": False,
                "calibration_performed": False,
                "inference_performed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            },
            # This receipt is a prerequisite, not a training-run permit by
            # itself.  The bounded runner binds it with the remaining frozen
            # inputs before executing the sole bounded-400 run.
            "bounded_400_authorized_by_this_receipt_alone": False,
            "formal_800_authorized": False,
            "evidence_fingerprint": self.evidence_fingerprint,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def run_coverage_state_bfa_dr_gate(
    *,
    dataset_free_receipt: CoverageStateBFADatasetFreeReceipt,
    real_inputs: CoverageStateRealDRInputs,
    bounded_population: CoverageStateBoundedPopulation,
    device: torch.device | str = "cpu",
) -> CoverageStateBFADRGateReceipt:
    """Run the sole BFA identifiability probe on already-built real inputs."""

    if not isinstance(
        dataset_free_receipt,
        CoverageStateBFADatasetFreeReceipt,
    ):
        raise TypeError(
            "dataset_free_receipt must be "
            "CoverageStateBFADatasetFreeReceipt"
        )
    if not isinstance(real_inputs, CoverageStateRealDRInputs):
        raise TypeError("real_inputs must be CoverageStateRealDRInputs")
    if not isinstance(
        bounded_population,
        CoverageStateBoundedPopulation,
    ):
        raise TypeError(
            "bounded_population must be CoverageStateBoundedPopulation"
        )
    dataset_free_receipt.verify_unchanged()
    real_inputs.verify_unchanged()
    bounded_population.verify_unchanged()
    if (
        not dataset_free_receipt.all_pass
        or bounded_population.source_cache is not real_inputs.scalar_cache
        or bounded_population.seed != COVERAGE_STATE_BFA_DR_EXECUTION_SEED
    ):
        raise PermissionError("BFA D_R prerequisites did not pass")

    resolved_device = torch.device(device)
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("BFA D_R gate supports only CPU or CUDA")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("requested CUDA device is unavailable")
    if resolved_device.type == "cuda" and resolved_device.index is None:
        resolved_device = torch.device(
            "cuda",
            torch.cuda.current_device(),
        )
    if resolved_device.type == "cuda" and (
        resolved_device.index is None
        or resolved_device.index < 0
        or resolved_device.index >= torch.cuda.device_count()
    ):
        raise ValueError("requested CUDA device is unavailable")

    probe = _probe(
        bounded_population,
        device=resolved_device,
    )
    checks = recompute_coverage_state_bfa_dr_checks(
        dataset_free_receipt=dataset_free_receipt,
        real_inputs=real_inputs,
        bounded_population=bounded_population,
        probe=probe,
    )
    implementation = _implementation_binding()
    evidence = {
        "dataset_free_receipt_fingerprint": (
            dataset_free_receipt.receipt_fingerprint
        ),
        "real_inputs_build_fingerprint": real_inputs.build_fingerprint,
        "source_binding_fingerprint": (
            real_inputs.source_binding.binding_fingerprint
        ),
        "bounded_population_fingerprint": (
            bounded_population.population_fingerprint
        ),
        "bounded_cache_fingerprint": (
            bounded_population.cache.cache_fingerprint
        ),
        "implementation_binding": dict(implementation),
        "probe": deepcopy(probe),
    }
    return CoverageStateBFADRGateReceipt(
        dataset_free_receipt=dataset_free_receipt,
        real_inputs=real_inputs,
        bounded_population=bounded_population,
        implementation_binding=implementation,
        probe=probe,
        checks=checks,
        evidence_fingerprint=stable_fingerprint(evidence),
    )


__all__ = [
    "COVERAGE_STATE_BFA_DR_EXECUTION_SEED",
    "COVERAGE_STATE_BFA_DR_FAIL_DECISION",
    "COVERAGE_STATE_BFA_DR_FLOAT32_EPSILON",
    "COVERAGE_STATE_BFA_DR_GATE_SCHEMA",
    "COVERAGE_STATE_BFA_DR_IMPLEMENTATION_PATHS",
    "COVERAGE_STATE_BFA_DR_PASS_DECISION",
    "COVERAGE_STATE_BFA_DR_RATIO_MULTIPLIER",
    "COVERAGE_STATE_BFA_DR_RATIO_THRESHOLD",
    "CoverageStateBFADRGateReceipt",
    "recompute_coverage_state_bfa_dr_checks",
    "run_coverage_state_bfa_dr_gate",
]
