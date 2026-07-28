"""Read-only real-``D_R`` identifiability gate for PAET-BFA.

This gate decides only whether the frozen PAET field is structurally
identifiable on the already selected seed-42 bounded population.  It does
not estimate performance, fit a probe, construct an optimizer, update a
parameter, or access ``D_V``/``D_T``.

The PAET-specific witness is evaluated in the actual output-phase coordinate
system.  For each factual/clean target group it binds one target phase ``p``
and one legal background phase ``q`` from ``state.background_mask`` that are
Chebyshev neighbours in the same coarse cell.  That *same pair* must exceed
one frozen scale-normalized separation threshold for phase features,
transported odd-hidden states, and the target's no-transport/BFA-common
counterfactual.

An exact target-versus-background/component collision scan is retained only
as a necessary check.  Collision absence is explicitly not claimed to prove
linear-readout feasibility.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from ..cache.schema import file_sha256, stable_fingerprint
from ..coverage_state_phase_aligned_evidence_transport import (
    CSLF_PAET_EQUATION_POLICY,
    CSLF_PAET_FIELD_POLICY,
    CSLF_PAET_FLIP_POLICY,
    CSLF_PAET_TRANSPORT_POLICY,
    CURELitePhaseAlignedEvidenceTransportLevelSet,
    CoverageStatePhaseAlignedEvidenceTransportConfig,
    CoverageStatePhaseAlignedEvidenceTransportFields,
    row_major_phase_unpack,
)
from ..coverage_state_sobolev import (
    coverage_state_pmope_pair_loss_from_targets,
)
from ..paired_types import tensor_content_fingerprint
from .coverage_state_bfa_dr_gate import (
    _direction_probe,
    _pair_targets_to_device,
    _row_bit_hash,
    _state_specs,
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
from .coverage_state_paet_dataset_free import (
    COVERAGE_STATE_PAET_MARGIN,
    CoverageStatePAETDatasetFreeReceipt,
)
from .coverage_state_real_dr_inputs import CoverageStateRealDRInputs


COVERAGE_STATE_PAET_DR_GATE_SCHEMA = (
    "cure-lite-paet-bfa-real-dr-identifiability-gate-v1"
)
COVERAGE_STATE_PAET_DR_EXECUTION_SEED = 42
COVERAGE_STATE_PAET_DR_FLOAT32_EPSILON = torch.finfo(
    torch.float32
).eps
COVERAGE_STATE_PAET_DR_SEPARATION_MULTIPLIER = 128
COVERAGE_STATE_PAET_DR_SEPARATION_THRESHOLD = (
    COVERAGE_STATE_PAET_DR_SEPARATION_MULTIPLIER
    * COVERAGE_STATE_PAET_DR_FLOAT32_EPSILON
)
COVERAGE_STATE_PAET_DR_PASS_DECISION = (
    "PAET_D_R_IDENTIFIABILITY_PASS"
)
COVERAGE_STATE_PAET_DR_FAIL_DECISION = (
    "PAET_D_R_IDENTIFIABILITY_FAIL"
)
COVERAGE_STATE_PAET_DR_IMPLEMENTATION_PATHS = (
    "cure_lite/coverage_state_phase_aligned_evidence_transport.py",
    "cure_lite/coverage_state_sobolev.py",
    "cure_lite/experiment/coverage_state_bfa_dr_gate.py",
    "cure_lite/experiment/coverage_state_bounded_protocol.py",
    "cure_lite/experiment/coverage_state_paet_dataset_free.py",
    "cure_lite/experiment/coverage_state_real_dr_inputs.py",
    "cure_lite/experiment/coverage_state_paet_dr_gate.py",
)
PAET_DR_BOUND_PHASE_PAIR_CHECK = (
    "each_target_group_has_one_same_coarse_cell_target_p_and_legal_"
    "background_q_pair_jointly_separated_in_phase_feature_transported_"
    "odd_hidden_and_no_transport_counterfactual_v1"
)
PAET_DR_CONFLICT_CHECK = (
    "necessary_no_exact_transported_odd_hidden_collision_between_"
    "mutually_exclusive_target_background_component_roles_v1"
)
PAET_DR_GRADIENT_CHECK = (
    "zero_readout_to_scalar_then_fixed_nonzero_readout_to_all_three_"
    "parameter_tensors_v1"
)


def _implementation_binding() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[2]
    rows: list[tuple[str, str]] = []
    for relative in COVERAGE_STATE_PAET_DR_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"PAET D_R implementation path is invalid: {relative}"
            )
        rows.append((relative, file_sha256(path)))
    return tuple(rows)


def _hex(value: float, *, name: str) -> str:
    number = float(value)
    if not isfinite(number):
        raise FloatingPointError(f"{name} is non-finite")
    return number.hex()


def _model_fingerprint(
    model: CURELitePhaseAlignedEvidenceTransportLevelSet,
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


def _determinism_flags() -> dict[str, bool]:
    return {
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": (
            torch.backends.cuda.matmul.allow_tf32
        ),
    }


@contextmanager
def _deterministic_execution_scope():
    """Fix and restore process-global deterministic execution flags."""

    before = _determinism_flags()
    ledger: dict[str, object] = {
        "policy": (
            "gate_internal_deterministic_algorithms_and_no_tf32_v1"
        ),
        "before": before,
    }
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        ledger["active"] = _determinism_flags()
        yield ledger
    finally:
        torch.use_deterministic_algorithms(
            before["deterministic_algorithms"],
            warn_only=before["deterministic_warn_only"],
        )
        torch.backends.cudnn.benchmark = before["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = before[
            "cudnn_deterministic"
        ]
        torch.backends.cudnn.allow_tf32 = before["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = before[
            "cuda_matmul_allow_tf32"
        ]
        after = _determinism_flags()
        ledger["after"] = after
        ledger["restored_exactly"] = after == before


def _phase_hidden_to_output(value: Tensor, *, stride: int) -> Tensor:
    """Map a PAET ``[B,P,W,h,w]`` tensor to ``[B,W,H,W]``."""

    output = row_major_phase_unpack(value, stride=stride)
    if (
        output.dtype != torch.float32
        or output.ndim != 4
        or not bool(torch.isfinite(output).all())
    ):
        raise FloatingPointError("PAET transported output is invalid")
    return output


def _no_transport_bfa_common_odd_hidden(
    fields: CoverageStatePhaseAlignedEvidenceTransportFields,
) -> Tensor:
    """Reproduce the BFA common-feature hidden state in exact FP32 order."""

    if not isinstance(
        fields,
        CoverageStatePhaseAlignedEvidenceTransportFields,
    ):
        raise TypeError("fields must be PAET forward fields")
    common_joint_affine = (
        fields.occupancy_affine + fields.coarse_feature_affine
    )
    common_actual_hidden = (
        F.silu(common_joint_affine)
        - F.silu(fields.occupancy_affine)
    )
    common_flipped_joint_affine = (
        common_joint_affine.unsqueeze(1) + fields.flip_delta
    )
    common_flipped_hidden = (
        F.silu(common_flipped_joint_affine)
        - F.silu(fields.flipped_occupancy_affine)
    )
    result = 0.5 * (
        common_actual_hidden.unsqueeze(1) - common_flipped_hidden
    )
    if (
        result.shape != fields.odd_feature_presence_hidden.shape
        or result.dtype != torch.float32
        or result.device != fields.field.device
        or not bool(torch.isfinite(result).all())
    ):
        raise FloatingPointError(
            "PAET no-transport BFA-common hidden state is invalid"
        )
    return result.contiguous()


def _vectors_at(value: Tensor, mask: Tensor) -> Tensor:
    if (
        value.ndim != 4
        or mask.dtype != torch.bool
        or mask.ndim != 4
        or mask.shape[0] != value.shape[0]
        or mask.shape[1] != 1
        or tuple(mask.shape[-2:]) != tuple(value.shape[-2:])
    ):
        raise ValueError("PAET representation and coordinate mask differ")
    return value.permute(0, 2, 3, 1)[mask[:, 0]].contiguous()


def _normalized_vector_separation(
    first: Tensor,
    second: Tensor,
) -> float:
    """Return scale-normalized L2 separation for two aligned vectors."""

    if (
        first.ndim != 1
        or second.ndim != 1
        or first.shape != second.shape
        or first.dtype != torch.float32
        or second.dtype != torch.float32
        or first.device != second.device
        or not bool(torch.isfinite(first).all())
        or not bool(torch.isfinite(second).all())
    ):
        raise ValueError("PAET separation vectors must be aligned FP32")
    numerator = torch.linalg.vector_norm(first - second)
    denominator = torch.maximum(
        torch.maximum(
            torch.linalg.vector_norm(first),
            torch.linalg.vector_norm(second),
        ),
        torch.ones((), dtype=torch.float32, device=first.device),
    )
    value = float((numerator / denominator).detach().cpu())
    if not isfinite(value) or value < 0.0:
        raise FloatingPointError("PAET normalized separation is invalid")
    return value


def _nearest_rank(values: list[float], numerator: int) -> float:
    if not values or numerator < 0 or numerator > 100:
        raise ValueError("invalid PAET nearest-rank request")
    ordered = sorted(values)
    index = ((len(ordered) - 1) * numerator + 50) // 100
    return ordered[index]


def _separation_distribution(
    values: list[float],
    *,
    name: str,
) -> dict[str, object]:
    if not values or any(not isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{name} separations must be finite and nonempty")
    return {
        "count": len(values),
        "minimum_hex": _hex(min(values), name=f"{name} minimum"),
        "maximum_hex": _hex(max(values), name=f"{name} maximum"),
        "nearest_rank_quantiles": {
            key: _hex(
                _nearest_rank(values, numerator),
                name=f"{name} {key}",
            )
            for key, numerator in (
                ("q000", 0),
                ("q025", 25),
                ("q050", 50),
                ("q075", 75),
                ("q100", 100),
            )
        },
        "threshold_hex": (
            COVERAGE_STATE_PAET_DR_SEPARATION_THRESHOLD.hex()
        ),
        "above_threshold_count": sum(
            value > COVERAGE_STATE_PAET_DR_SEPARATION_THRESHOLD
            for value in values
        ),
        "normalization_policy": (
            "l2_difference_over_max_l2_first_l2_second_one_v1"
        ),
    }


def _bound_phase_pair_witness(
    phase_feature: Tensor,
    transported_odd_hidden: Tensor,
    no_transport_odd_hidden: Tensor,
    *,
    target_mask: Tensor,
    background_mask: Tensor,
    stride: int,
) -> dict[str, object]:
    """Bind all three PAET separations to the same target/background pair."""

    target_coordinates = torch.nonzero(
        target_mask[:, 0],
        as_tuple=False,
    )
    if (
        target_mask.dtype != torch.bool
        or background_mask.dtype != torch.bool
        or target_mask.shape != background_mask.shape
        or phase_feature.shape != transported_odd_hidden.shape
        or transported_odd_hidden.shape != no_transport_odd_hidden.shape
        or target_mask.shape[0] != phase_feature.shape[0]
        or tuple(target_mask.shape[-2:])
        != tuple(phase_feature.shape[-2:])
    ):
        raise ValueError("PAET bound-pair tensors do not align")
    pair_rows: list[dict[str, object]] = []
    phase_separations: list[float] = []
    hidden_separations: list[float] = []
    counterfactual_separations: list[float] = []
    height, width = target_mask.shape[-2:]
    for coordinate in target_coordinates:
        batch, row, column = (
            int(value) for value in coordinate.tolist()
        )
        coarse_row = row // stride
        coarse_column = column // stride
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                if row_delta == 0 and column_delta == 0:
                    continue
                neighbour_row = row + row_delta
                neighbour_column = column + column_delta
                if (
                    neighbour_row < 0
                    or neighbour_row >= height
                    or neighbour_column < 0
                    or neighbour_column >= width
                    or not bool(
                        background_mask[
                            batch,
                            0,
                            neighbour_row,
                            neighbour_column,
                        ]
                    )
                    or neighbour_row // stride != coarse_row
                    or neighbour_column // stride != coarse_column
                ):
                    continue
                phase_separation = _normalized_vector_separation(
                    phase_feature[batch, :, row, column],
                    phase_feature[
                        batch,
                        :,
                        neighbour_row,
                        neighbour_column,
                    ],
                )
                hidden_separation = _normalized_vector_separation(
                    transported_odd_hidden[batch, :, row, column],
                    transported_odd_hidden[
                        batch,
                        :,
                        neighbour_row,
                        neighbour_column,
                    ],
                )
                counterfactual_separation = (
                    _normalized_vector_separation(
                        transported_odd_hidden[
                            batch,
                            :,
                            row,
                            column,
                        ],
                        no_transport_odd_hidden[
                            batch,
                            :,
                            row,
                            column,
                        ],
                    )
                )
                jointly_above = all(
                    value
                    > COVERAGE_STATE_PAET_DR_SEPARATION_THRESHOLD
                    for value in (
                        phase_separation,
                        hidden_separation,
                        counterfactual_separation,
                    )
                )
                phase_separations.append(phase_separation)
                hidden_separations.append(hidden_separation)
                counterfactual_separations.append(
                    counterfactual_separation
                )
                pair_rows.append(
                    {
                        "target_coordinate": [batch, row, column],
                        "background_coordinate": [
                            batch,
                            neighbour_row,
                            neighbour_column,
                        ],
                        "coarse_cell": [
                            batch,
                            coarse_row,
                            coarse_column,
                        ],
                        "target_phase_index": (
                            (row % stride) * stride + column % stride
                        ),
                        "background_phase_index": (
                            (neighbour_row % stride) * stride
                            + neighbour_column % stride
                        ),
                        "chebyshev_distance": max(
                            abs(row_delta),
                            abs(column_delta),
                        ),
                        "legal_background_from_state_mask": bool(
                            background_mask[
                                batch,
                                0,
                                neighbour_row,
                                neighbour_column,
                            ]
                        ),
                        "phase_feature_separation_hex": _hex(
                            phase_separation,
                            name="PAET phase-feature separation",
                        ),
                        "transported_odd_hidden_separation_hex": _hex(
                            hidden_separation,
                            name="PAET hidden separation",
                        ),
                        (
                            "target_odd_vs_no_transport_common_"
                            "separation_hex"
                        ): _hex(
                            counterfactual_separation,
                            name="PAET counterfactual separation",
                        ),
                        "all_three_strictly_above_threshold": (
                            jointly_above
                        ),
                    }
                )
    result: dict[str, object] = {
        "target_coordinate_count": int(target_coordinates.shape[0]),
        "target_coordinates_with_legal_background_q": len(
            {
                tuple(row["target_coordinate"])
                for row in pair_rows
            }
        ),
        "target_coordinates_without_legal_background_q": (
            int(target_coordinates.shape[0])
            - len(
                {
                    tuple(row["target_coordinate"])
                    for row in pair_rows
                }
            )
        ),
        "pair_count": len(pair_rows),
        "jointly_separated_pair_count": sum(
            row["all_three_strictly_above_threshold"]
            for row in pair_rows
        ),
        "at_least_one_bound_pair_jointly_separated": any(
            row["all_three_strictly_above_threshold"]
            for row in pair_rows
        ),
        "pair_rows": pair_rows,
        "pair_binding_fingerprint": stable_fingerprint(pair_rows),
        "separation_threshold_hex": (
            COVERAGE_STATE_PAET_DR_SEPARATION_THRESHOLD.hex()
        ),
        "separation_threshold_policy": (
            "strictly_greater_than_128_times_float32_epsilon"
        ),
        "same_target_p_background_q_pair_for_all_three_checks": True,
        "legal_background_policy": (
            "state_background_mask_same_coarse_cell_chebyshev1"
        ),
        "no_transport_counterfactual_policy": (
            "exact_BFA_common_joint_then_center_phase_flip_delta_"
            "float32_order_v1"
        ),
    }
    if pair_rows:
        result["separation_distributions"] = {
            "phase_feature_p_vs_q": _separation_distribution(
                phase_separations,
                name="phase feature p versus q",
            ),
            "transported_odd_hidden_p_vs_q": _separation_distribution(
                hidden_separations,
                name="transported odd hidden p versus q",
            ),
            "target_odd_vs_no_transport_common": (
                _separation_distribution(
                    counterfactual_separations,
                    name="target odd versus no-transport common",
                )
            ),
        }
    else:
        result["separation_distributions"] = {}
    return result


def _scan_exact_collisions(
    vectors: Tensor,
    *,
    coordinates: Tensor,
    state: object,
    role: str,
    target_hashes: Tensor,
    targets_by_hash: dict[int, list[dict[str, object]]],
    examples: list[dict[str, object]],
) -> int:
    """Count exact target/positive-role representation collisions."""

    hashes = _row_bit_hash(vectors)
    candidate_indices = torch.nonzero(
        torch.isin(hashes, target_hashes),
        as_tuple=False,
    ).flatten()
    count = 0
    for index_tensor in candidate_indices:
        index = int(index_tensor)
        hash_value = int(hashes[index].item())
        for target in targets_by_hash.get(hash_value, ()):
            target_vector = target["vector"]
            if not isinstance(target_vector, Tensor):
                raise TypeError("target conflict vector is invalid")
            if not torch.equal(vectors[index], target_vector):
                continue
            count += 1
            if len(examples) < 64:
                examples.append(
                    {
                        "transported_representation_fingerprint": (
                            tensor_content_fingerprint(
                                vectors[index].contiguous()
                            )
                        ),
                        "negative_requirement": {
                            key: deepcopy(value)
                            for key, value in target.items()
                            if key != "vector"
                        },
                        "positive_requirement": {
                            "state_id": getattr(state, "state_id"),
                            "sample_id": getattr(state, "sample_id"),
                            "state_kind": getattr(state, "state_kind"),
                            "endpoint": getattr(state, "endpoint"),
                            "role": role,
                            "coordinate": coordinates[index].tolist(),
                            "component_writable": (
                                getattr(state, "component_writable")
                                if role == "component"
                                else None
                            ),
                        },
                    }
                )
    return count


def _representation_probe(
    model: CURELitePhaseAlignedEvidenceTransportLevelSet,
    population: CoverageStateBoundedPopulation,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Audit PAET transport on actual target and positive-role coordinates."""

    target_states, context_states = _state_specs(population)
    target_state_ids = [state.state_id for state in target_states]
    context_state_ids = [state.state_id for state in context_states]
    if (
        len(set(target_state_ids)) != len(target_state_ids)
        or len(set(context_state_ids)) != len(context_state_ids)
    ):
        raise ValueError("PAET D_R state IDs must be unique per pass")
    reforwarded_target_state_ids = sorted(
        set(target_state_ids) & set(context_state_ids)
    )
    target_rows: list[dict[str, object]] = []
    target_hash_rows: dict[int, list[dict[str, object]]] = {}
    target_coordinate_count = 0
    target_forward_count = 0
    role_rows: list[dict[str, object]] = []
    role_counts = {"background": 0, "component": 0}

    def prepare_positive_roles(
        state: object,
        transported_hidden: Tensor,
    ) -> list[tuple[object, str, Tensor, Tensor]]:
        prepared: list[tuple[object, str, Tensor, Tensor]] = []
        for role, mask_cpu in (
            ("background", getattr(state, "background_mask")),
            ("component", getattr(state, "component_mask")),
        ):
            if not bool(torch.any(mask_cpu)):
                continue
            vectors = _vectors_at(
                transported_hidden,
                mask_cpu.to(device=device),
            ).detach().to("cpu").contiguous()
            coordinates = torch.nonzero(
                mask_cpu[:, 0],
                as_tuple=False,
            ).to("cpu")
            if vectors.shape[0] != coordinates.shape[0]:
                raise AssertionError(
                    "PAET role vectors and coordinates differ"
                )
            role_counts[role] += int(vectors.shape[0])
            role_rows.append(
                {
                    "state_id": getattr(state, "state_id"),
                    "sample_id": getattr(state, "sample_id"),
                    "state_kind": getattr(state, "state_kind"),
                    "endpoint": getattr(state, "endpoint"),
                    "role": role,
                    "coordinate_count": int(vectors.shape[0]),
                    "mask_fingerprint": tensor_content_fingerprint(
                        mask_cpu
                    ),
                    "transported_representation_finite": bool(
                        torch.isfinite(vectors).all()
                    ),
                    "transported_representation_fingerprint": (
                        tensor_content_fingerprint(vectors)
                    ),
                    "component_writable": (
                        getattr(state, "component_writable")
                        if role == "component"
                        else None
                    ),
                }
            )
            prepared.append((state, role, vectors, coordinates))
        return prepared

    for state in target_states:
        feature = state.feature.to(device=device, dtype=torch.float32)
        occupancy = state.occupancy.to(device=device)
        with torch.no_grad():
            fields = model.forward_fields(feature, occupancy)
        target_forward_count += 1
        stride = model.config.feature_stride
        transported_feature = _phase_hidden_to_output(
            fields.phase_feature_affine,
            stride=stride,
        )
        transported_hidden = _phase_hidden_to_output(
            fields.odd_feature_presence_hidden,
            stride=stride,
        )
        no_transport_odd_hidden = _phase_hidden_to_output(
            _no_transport_bfa_common_odd_hidden(fields),
            stride=stride,
        )
        target_mask = state.target_mask.to(device=device)
        target_vectors = _vectors_at(
            transported_hidden,
            target_mask,
        )
        target_coordinates = torch.nonzero(
            state.target_mask[:, 0],
            as_tuple=False,
        ).to("cpu")
        if (
            target_vectors.shape[0] != target_coordinates.shape[0]
            or target_vectors.shape[0] < 1
        ):
            raise ValueError("PAET target group has invalid coordinates")
        bound_pair = _bound_phase_pair_witness(
            transported_feature,
            transported_hidden,
            no_transport_odd_hidden,
            target_mask=target_mask,
            background_mask=state.background_mask.to(device=device),
            stride=stride,
        )
        target_rows.append(
            {
                "target_group_id": state.target_group_id,
                "state_id": state.state_id,
                "sample_id": state.sample_id,
                "state_kind": state.state_kind,
                "coordinate_count": int(target_vectors.shape[0]),
                "transported_representation_finite": bool(
                    torch.isfinite(target_vectors).all()
                ),
                "bound_phase_pair_witness": bound_pair,
                "target_mask_fingerprint": tensor_content_fingerprint(
                    state.target_mask
                ),
                "transported_representation_fingerprint": (
                    tensor_content_fingerprint(target_vectors)
                ),
                "no_transport_counterfactual_fingerprint": (
                    tensor_content_fingerprint(
                        _vectors_at(
                            no_transport_odd_hidden,
                            target_mask,
                        )
                    )
                ),
            }
        )
        hashes = _row_bit_hash(target_vectors.detach().to("cpu"))
        for index in range(target_vectors.shape[0]):
            hash_value = int(hashes[index].item())
            target_hash_rows.setdefault(hash_value, []).append(
                {
                    "target_group_id": state.target_group_id,
                    "state_id": state.state_id,
                    "sample_id": state.sample_id,
                    "coordinate": target_coordinates[index].tolist(),
                    "vector": target_vectors[index]
                    .detach()
                    .to("cpu")
                    .contiguous(),
                }
            )
        target_coordinate_count += int(target_vectors.shape[0])

    if not target_hash_rows:
        raise ValueError("PAET D_R has no target representations")
    target_hashes = torch.tensor(
        sorted(target_hash_rows),
        dtype=torch.int64,
    )
    collisions = 0
    collision_examples: list[dict[str, object]] = []
    positive_pass_forward_count = 0

    for state in context_states:
        feature = state.feature.to(device=device, dtype=torch.float32)
        occupancy = state.occupancy.to(device=device)
        with torch.no_grad():
            fields = model.forward_fields(feature, occupancy)
        positive_pass_forward_count += 1
        transported_hidden = _phase_hidden_to_output(
            fields.odd_feature_presence_hidden,
            stride=model.config.feature_stride,
        )
        for prepared in prepare_positive_roles(state, transported_hidden):
            prepared_state, role, vectors, coordinates = prepared
            collisions += _scan_exact_collisions(
                vectors,
                coordinates=coordinates,
                state=prepared_state,
                role=role,
                target_hashes=target_hashes,
                targets_by_hash=target_hash_rows,
                examples=collision_examples,
            )

    return {
        "coordinate_policy": {
            "bound_phase_pair_check": PAET_DR_BOUND_PHASE_PAIR_CHECK,
            "necessary_exact_collision_check": (
                PAET_DR_CONFLICT_CHECK
            ),
            "target_interval": "d_le_negative_1p125",
            "background_interval": "d_ge_negative_0p675",
            "component_interval": "abs_d_le_0p675",
        },
        "state_counts": {
            "target_states": len(target_states),
            "context_states": len(context_states),
            "factual_target_groups": sum(
                row["state_kind"] == "factual_miss"
                for row in target_rows
            ),
            "clean_target_groups": sum(
                row["state_kind"] == "clean_positive"
                for row in target_rows
            ),
            "target_pass_forward_count": target_forward_count,
            "positive_pass_forward_count": positive_pass_forward_count,
            "total_forward_count": (
                target_forward_count + positive_pass_forward_count
            ),
            "target_states_reforwarded_in_positive_pass": (
                len(reforwarded_target_state_ids)
            ),
            "two_pass_streaming_no_full_map_retention": True,
        },
        "state_id_ledger": {
            "target_pass_state_ids": target_state_ids,
            "positive_pass_state_ids": context_state_ids,
            "reforwarded_target_state_ids": (
                reforwarded_target_state_ids
            ),
            "target_pass_state_id_fingerprint": stable_fingerprint(
                target_state_ids
            ),
            "positive_pass_state_id_fingerprint": stable_fingerprint(
                context_state_ids
            ),
            "reforwarded_target_state_id_fingerprint": (
                stable_fingerprint(reforwarded_target_state_ids)
            ),
        },
        "coordinate_counts": {
            "target": target_coordinate_count,
            **role_counts,
        },
        "target_group_rows": target_rows,
        "positive_role_rows": role_rows,
        "target_representation_hash_count": len(target_hash_rows),
        "necessary_exact_collision_count": collisions,
        "collision_examples": collision_examples,
        "collision_examples_truncated": collisions > len(
            collision_examples
        ),
        "exact_collision_zero_is_readout_feasibility_proof": False,
        "exact_collision_zero_is_only_a_necessary_check": True,
        "transported_representation_binding": stable_fingerprint(
            {
                "target_group_rows": target_rows,
                "positive_role_rows": role_rows,
            }
        ),
    }


def _gradient_probe(
    model: CURELitePhaseAlignedEvidenceTransportLevelSet,
    population: CoverageStateBoundedPopulation,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Prove the two-stage gradient path without changing the model."""

    clean = sorted(
        population.cache.clean_positive_records,
        key=lambda value: value.record.pair_id,
    )
    if not clean:
        raise ValueError("PAET D_R gradient witness requires a clean pair")
    value = clean[0]
    record = value.record
    feature = torch.cat((record.feature, record.feature), dim=0).to(
        device=device,
        dtype=torch.float32,
    )
    occupancy = torch.cat(
        (record.occupancy_plus, record.occupancy_minus),
        dim=0,
    ).to(device=device)
    targets = _pair_targets_to_device(
        value.joint_targets,
        device=device,
    )

    zero_field = model(feature, occupancy)
    zero_plus, zero_minus = zero_field.split(1, dim=0)
    zero_loss = coverage_state_pmope_pair_loss_from_targets(
        zero_plus,
        zero_minus,
        targets,
        config=population.cache.sobolev_config,
        validate=True,
    ).loss
    zero_gradients = torch.autograd.grad(
        zero_loss,
        tuple(model.parameters()),
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    zero_by_name = {
        name: gradient
        for (name, _), gradient in zip(
            model.named_parameters(),
            zero_gradients,
            strict=True,
        )
    }

    original_parameter_ids = {
        name: id(parameter)
        for name, parameter in model.named_parameters()
    }
    functional_parameters = {
        "joint_state_weight": (
            model.joint_state_weight.detach().clone().requires_grad_(True)
        ),
        "joint_hidden_bias": (
            model.joint_hidden_bias.detach().clone().requires_grad_(True)
        ),
        "scalar_energy_weight": torch.linspace(
            0.5,
            1.5,
            model.config.width,
            device=device,
            dtype=torch.float32,
        ).requires_grad_(True),
    }
    nonzero_field = torch.func.functional_call(
        model,
        functional_parameters,
        (feature, occupancy),
        strict=True,
    )
    nonzero_plus, nonzero_minus = nonzero_field.split(1, dim=0)
    nonzero_loss = coverage_state_pmope_pair_loss_from_targets(
        nonzero_plus,
        nonzero_minus,
        targets,
        config=population.cache.sobolev_config,
        validate=True,
    ).loss
    nonzero_gradients = torch.autograd.grad(
        nonzero_loss,
        tuple(functional_parameters.values()),
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    nonzero_by_name = dict(
        zip(functional_parameters, nonzero_gradients, strict=True)
    )

    def gradient_row(value: Tensor, *, name: str) -> dict[str, object]:
        norm = float(value.detach().norm().cpu())
        return {
            "name": name,
            "shape": list(value.shape),
            "finite": bool(torch.isfinite(value).all()),
            "nonzero": bool(torch.any(value != 0.0)),
            "l2_hex": _hex(norm, name=f"PAET {name} gradient"),
            "fingerprint": tensor_content_fingerprint(
                value.detach().to("cpu").contiguous()
            ),
        }

    zero_rows = {
        name: gradient_row(value, name=f"zero_readout:{name}")
        for name, value in zero_by_name.items()
    }
    nonzero_rows = {
        name: gradient_row(value, name=f"fixed_nonzero:{name}")
        for name, value in nonzero_by_name.items()
    }
    return {
        "policy": PAET_DR_GRADIENT_CHECK,
        "pair_id": record.pair_id,
        "sample_id": record.sample_id,
        "zero_readout_loss_hex": _hex(
            float(zero_loss.detach().cpu()),
            name="PAET zero-readout loss",
        ),
        "fixed_nonzero_readout_loss_hex": _hex(
            float(nonzero_loss.detach().cpu()),
            name="PAET fixed-nonzero-readout loss",
        ),
        "zero_readout_gradients": zero_rows,
        "fixed_nonzero_readout_gradients": nonzero_rows,
        "zero_readout_scalar_path_finite_nonzero": (
            zero_rows["scalar_energy_weight"]["finite"] is True
            and zero_rows["scalar_energy_weight"]["nonzero"] is True
        ),
        "zero_readout_upstream_gradients_exactly_zero": (
            zero_rows["joint_state_weight"]["finite"] is True
            and zero_rows["joint_hidden_bias"]["finite"] is True
            and zero_rows["joint_state_weight"]["nonzero"] is False
            and zero_rows["joint_hidden_bias"]["nonzero"] is False
        ),
        "fixed_nonzero_readout_all_three_finite_nonzero": all(
            row["finite"] is True and row["nonzero"] is True
            for row in nonzero_rows.values()
        ),
        "functional_call_did_not_replace_model_parameters": all(
            original_parameter_ids.get(name) == id(parameter)
            for name, parameter in model.named_parameters()
        ),
    }


def _contains_performance_or_auc_key(value: object) -> bool:
    """Reject a hidden performance/AUC criterion anywhere in a probe."""

    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            if "performance" in normalized or "auc" in normalized:
                return True
            if _contains_performance_or_auc_key(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_performance_or_auc_key(item) for item in value)
    return False


def _bound_pair_group_is_valid(row: object) -> bool:
    """Recompute one target group's bound-pair and distribution receipt."""

    if not isinstance(row, dict):
        return False
    witness = row.get("bound_phase_pair_witness")
    if not isinstance(witness, dict):
        return False
    pairs = witness.get("pair_rows")
    if not isinstance(pairs, list) or not pairs:
        return False
    metric_keys = (
        "phase_feature_separation_hex",
        "transported_odd_hidden_separation_hex",
        "target_odd_vs_no_transport_common_separation_hex",
    )
    values: dict[str, list[float]] = {
        key: [] for key in metric_keys
    }
    jointly_separated = 0
    stride = COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE
    for pair in pairs:
        if not isinstance(pair, dict):
            return False
        target = pair.get("target_coordinate")
        background = pair.get("background_coordinate")
        coarse = pair.get("coarse_cell")
        if (
            not isinstance(target, list)
            or not isinstance(background, list)
            or not isinstance(coarse, list)
            or len(target) != 3
            or len(background) != 3
            or len(coarse) != 3
            or target[0] != background[0]
            or coarse
            != [
                target[0],
                target[1] // stride,
                target[2] // stride,
            ]
            or coarse
            != [
                background[0],
                background[1] // stride,
                background[2] // stride,
            ]
            or max(
                abs(target[1] - background[1]),
                abs(target[2] - background[2]),
            )
            != 1
            or pair.get("chebyshev_distance") != 1
            or pair.get("legal_background_from_state_mask") is not True
            or pair.get("target_phase_index")
            != (target[1] % stride) * stride + target[2] % stride
            or pair.get("background_phase_index")
            != (
                (background[1] % stride) * stride
                + background[2] % stride
            )
        ):
            return False
        try:
            separations = {
                key: float.fromhex(str(pair.get(key)))
                for key in metric_keys
            }
        except (TypeError, ValueError):
            return False
        if any(
            not isfinite(value) or value < 0.0
            for value in separations.values()
        ):
            return False
        expected_joint = all(
            value > COVERAGE_STATE_PAET_DR_SEPARATION_THRESHOLD
            for value in separations.values()
        )
        if (
            pair.get("all_three_strictly_above_threshold")
            != expected_joint
        ):
            return False
        jointly_separated += int(expected_joint)
        for key, value in separations.items():
            values[key].append(value)
    expected_distributions = {
        "phase_feature_p_vs_q": _separation_distribution(
            values["phase_feature_separation_hex"],
            name="phase feature p versus q",
        ),
        "transported_odd_hidden_p_vs_q": _separation_distribution(
            values["transported_odd_hidden_separation_hex"],
            name="transported odd hidden p versus q",
        ),
        "target_odd_vs_no_transport_common": _separation_distribution(
            values[
                "target_odd_vs_no_transport_common_separation_hex"
            ],
            name="target odd versus no-transport common",
        ),
    }
    unique_bound_targets = {
        tuple(pair["target_coordinate"]) for pair in pairs
    }
    return (
        int(row.get("coordinate_count", 0)) > 0
        and row.get("transported_representation_finite") is True
        and witness.get("pair_count") == len(pairs)
        and witness.get("target_coordinate_count")
        == int(row.get("coordinate_count", 0))
        and witness.get(
            "target_coordinates_with_legal_background_q"
        )
        == len(unique_bound_targets)
        and 0
        < len(unique_bound_targets)
        <= int(row.get("coordinate_count", 0))
        and witness.get(
            "target_coordinates_without_legal_background_q"
        )
        == (
            int(row.get("coordinate_count", 0))
            - len(unique_bound_targets)
        )
        and witness.get("jointly_separated_pair_count")
        == jointly_separated
        and witness.get(
            "at_least_one_bound_pair_jointly_separated"
        )
        is True
        and jointly_separated > 0
        and witness.get("pair_binding_fingerprint")
        == stable_fingerprint(pairs)
        and witness.get("separation_threshold_hex")
        == COVERAGE_STATE_PAET_DR_SEPARATION_THRESHOLD.hex()
        and witness.get("separation_threshold_policy")
        == "strictly_greater_than_128_times_float32_epsilon"
        and witness.get(
            "same_target_p_background_q_pair_for_all_three_checks"
        )
        is True
        and witness.get("legal_background_policy")
        == "state_background_mask_same_coarse_cell_chebyshev1"
        and witness.get("no_transport_counterfactual_policy")
        == (
            "exact_BFA_common_joint_then_center_phase_flip_delta_"
            "float32_order_v1"
        )
        and witness.get("separation_distributions")
        == expected_distributions
    )


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
    with (
        _deterministic_execution_scope() as determinism,
        torch.random.fork_rng(
            devices=_cuda_rng_devices(device),
            device_type=("cuda" if device.type == "cuda" else None),
        ),
        torch.autocast(device_type=device.type, enabled=False),
    ):
        torch.random.default_generator.manual_seed(
            COVERAGE_STATE_PAET_DR_EXECUTION_SEED
        )
        config = CoverageStatePhaseAlignedEvidenceTransportConfig(
            feature_channels=COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
            feature_stride=COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
            width=COVERAGE_STATE_CMIF_FORMAL_WIDTH,
        )
        model = CURELitePhaseAlignedEvidenceTransportLevelSet(config).to(
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
        direction = _direction_probe(population, device=device)
        gradient = _gradient_probe(model, population, device=device)
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
        model_config = {
            "model_class": type(model).__name__,
            "feature_channels": config.feature_channels,
            "feature_stride": config.feature_stride,
            "width": config.width,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "parameter_tensor_count": len(tuple(model.parameters())),
            "field_policy": config.field_policy,
            "equation_policy": config.equation_policy,
            "flip_policy": config.flip_policy,
            "transport_policy": config.transport_policy,
            "margin_hex": COVERAGE_STATE_PAET_MARGIN.hex(),
        }
        parameter_grad_buffers_unretained = all(
            parameter.grad is None for parameter in model.parameters()
        )
    population.verify_unchanged()
    after_device_rng = (
        None
        if device.type != "cuda"
        else torch.cuda.get_rng_state(device).clone()
    )
    return {
        "device": str(device),
        "execution_seed": COVERAGE_STATE_PAET_DR_EXECUTION_SEED,
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
        "gradient_path": gradient,
        "deterministic_execution": determinism,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_performed": False,
        "calibration_performed": False,
        "inference_performed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
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


def recompute_coverage_state_paet_dr_checks(
    *,
    dataset_free_receipt: CoverageStatePAETDatasetFreeReceipt,
    real_inputs: CoverageStateRealDRInputs,
    bounded_population: CoverageStateBoundedPopulation,
    probe: dict[str, object],
) -> tuple[tuple[str, bool], ...]:
    """Recompute every frozen PAET real-``D_R`` prerequisite check."""

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
    positive_rows = (
        representation.get("positive_role_rows", [])
        if isinstance(representation, dict)
        else []
    )
    state_id_ledger = (
        representation.get("state_id_ledger", {})
        if isinstance(representation, dict)
        else {}
    )
    direction = probe.get("field_direction", {})
    gradient = probe.get("gradient_path", {})
    deterministic = probe.get("deterministic_execution", {})
    model_config = probe.get("model_config", {})
    parameter_contract = probe.get("parameter_contract", [])
    expected_groups = 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
    expected_direction_counts = {
        "factual_target": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "clean_target": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "writable_background": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "writable_component": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
    }
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
        "fixed_seed42_bounded_factual_and_clean_target_groups": (
            bounded_population.seed
            == COVERAGE_STATE_PAET_DR_EXECUTION_SEED
            == COVERAGE_STATE_BOUNDED_SEED
            and state_counts.get("factual_target_groups")
            == COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and state_counts.get("clean_target_groups")
            == COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and state_counts.get("target_states") == expected_groups
            and len(target_rows) == expected_groups
        ),
        "complete_declared_state_forward_ledger": (
            state_counts.get("context_states")
            == 6 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and state_counts.get("target_pass_forward_count")
            == expected_groups
            and state_counts.get("positive_pass_forward_count")
            == 6 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and state_counts.get("total_forward_count")
            == 8 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and state_counts.get(
                "target_states_reforwarded_in_positive_pass"
            )
            == expected_groups
            and state_counts.get(
                "two_pass_streaming_no_full_map_retention"
            )
            is True
            and isinstance(state_id_ledger, dict)
            and isinstance(
                state_id_ledger.get("target_pass_state_ids"),
                list,
            )
            and isinstance(
                state_id_ledger.get("positive_pass_state_ids"),
                list,
            )
            and isinstance(
                state_id_ledger.get(
                    "reforwarded_target_state_ids"
                ),
                list,
            )
            and len(state_id_ledger["target_pass_state_ids"])
            == expected_groups
            and len(set(state_id_ledger["target_pass_state_ids"]))
            == expected_groups
            and len(state_id_ledger["positive_pass_state_ids"])
            == 6 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and len(set(state_id_ledger["positive_pass_state_ids"]))
            == 6 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and state_id_ledger["reforwarded_target_state_ids"]
            == sorted(
                set(state_id_ledger["target_pass_state_ids"])
                & set(state_id_ledger["positive_pass_state_ids"])
            )
            and len(
                state_id_ledger["reforwarded_target_state_ids"]
            )
            == state_counts.get(
                "target_states_reforwarded_in_positive_pass"
            )
            == expected_groups
            and state_id_ledger.get(
                "target_pass_state_id_fingerprint"
            )
            == stable_fingerprint(
                state_id_ledger["target_pass_state_ids"]
            )
            and state_id_ledger.get(
                "positive_pass_state_id_fingerprint"
            )
            == stable_fingerprint(
                state_id_ledger["positive_pass_state_ids"]
            )
            and state_id_ledger.get(
                "reforwarded_target_state_id_fingerprint"
            )
            == stable_fingerprint(
                state_id_ledger["reforwarded_target_state_ids"]
            )
        ),
        "actual_target_background_component_coordinates_nonempty": (
            isinstance(coordinate_counts, dict)
            and all(
                int(coordinate_counts.get(role, 0)) > 0
                for role in ("target", "background", "component")
            )
        ),
        "all_positive_role_representations_finite_and_bound": (
            isinstance(positive_rows, list)
            and len(positive_rows) > 0
            and all(
                isinstance(row, dict)
                and row.get("role") in {"background", "component"}
                and int(row.get("coordinate_count", 0)) > 0
                and row.get("transported_representation_finite") is True
                and isinstance(
                    row.get("transported_representation_fingerprint"),
                    str,
                )
                and len(
                    str(
                        row.get(
                            "transported_representation_fingerprint"
                        )
                    )
                )
                == 64
                for row in positive_rows
            )
            and sum(
                int(row.get("coordinate_count", 0))
                for row in positive_rows
                if row.get("role") == "background"
            )
            == int(coordinate_counts.get("background", -1))
            and sum(
                int(row.get("coordinate_count", 0))
                for row in positive_rows
                if row.get("role") == "component"
            )
            == int(coordinate_counts.get("component", -1))
            and isinstance(
                representation.get(
                    "transported_representation_binding"
                ),
                str,
            )
            and len(
                str(
                    representation.get(
                        "transported_representation_binding"
                    )
                )
            )
            == 64
            and representation.get(
                "transported_representation_binding"
            )
            == stable_fingerprint(
                {
                    "target_group_rows": target_rows,
                    "positive_role_rows": positive_rows,
                }
            )
        ),
        PAET_DR_BOUND_PHASE_PAIR_CHECK: (
            isinstance(target_rows, list)
            and len(target_rows) == expected_groups
            and len(
                {
                    row.get("target_group_id")
                    for row in target_rows
                    if isinstance(row, dict)
                }
            )
            == expected_groups
            and sum(
                int(row.get("coordinate_count", 0))
                for row in target_rows
                if isinstance(row, dict)
            )
            == int(coordinate_counts.get("target", -1))
            and all(_bound_pair_group_is_valid(row) for row in target_rows)
        ),
        PAET_DR_CONFLICT_CHECK: (
            representation.get("necessary_exact_collision_count")
            == 0
            and representation.get("collision_examples") == []
            and representation.get(
                "exact_collision_zero_is_readout_feasibility_proof"
            )
            is False
            and representation.get(
                "exact_collision_zero_is_only_a_necessary_check"
            )
            is True
        ),
        "actual_frozen_loss_directions_are_finite_nonzero": (
            isinstance(direction, dict)
            and direction.get("all_roles_finite_nonzero_correct") is True
            and direction.get("uses_actual_target_geometry") is True
            and direction.get(
                "uses_actual_valid_and_writable_masks"
            )
            is True
            and direction.get("actual_role_rows")
            == expected_direction_counts
            and len(direction.get("rows", []))
            == 4 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
            and direction.get("loss_apis")
            == [
                "coverage_state_absolute_sobolev_loss_from_targets",
                "coverage_state_pmope_pair_loss_from_targets",
            ]
            and all(
                isinstance(row, dict)
                and float.fromhex(str(row.get("loss_hex"))) > 0.0
                and row.get("descent_finite") is True
                and row.get("descent_nonzero") is True
                and row.get("aggregate_descent_direction_correct")
                is True
                and isfinite(
                    float.fromhex(str(row.get("descent_sum_hex")))
                )
                and float.fromhex(str(row.get("descent_sum_hex")))
                != 0.0
                for row in direction.get("rows", [])
            )
        ),
        "zero_readout_scalar_gradient_path_finite_nonzero": (
            isinstance(gradient, dict)
            and gradient.get("policy") == PAET_DR_GRADIENT_CHECK
            and float.fromhex(
                str(gradient.get("zero_readout_loss_hex"))
            )
            > 0.0
            and gradient.get(
                "zero_readout_scalar_path_finite_nonzero"
            )
            is True
            and gradient.get(
                "zero_readout_upstream_gradients_exactly_zero"
            )
            is True
            and isinstance(
                gradient.get("zero_readout_gradients"),
                dict,
            )
            and set(gradient["zero_readout_gradients"])
            == {
                "joint_state_weight",
                "joint_hidden_bias",
                "scalar_energy_weight",
            }
            and gradient["zero_readout_gradients"][
                "scalar_energy_weight"
            ].get("finite")
            is True
            and gradient["zero_readout_gradients"][
                "scalar_energy_weight"
            ].get("nonzero")
            is True
            and all(
                gradient["zero_readout_gradients"][name].get("finite")
                is True
                and gradient["zero_readout_gradients"][name].get(
                    "nonzero"
                )
                is False
                for name in (
                    "joint_state_weight",
                    "joint_hidden_bias",
                )
            )
        ),
        "fixed_nonzero_readout_all_three_gradients_finite_nonzero": (
            isinstance(gradient, dict)
            and float.fromhex(
                str(gradient.get("fixed_nonzero_readout_loss_hex"))
            )
            > 0.0
            and gradient.get(
                "fixed_nonzero_readout_all_three_finite_nonzero"
            )
            is True
            and gradient.get(
                "functional_call_did_not_replace_model_parameters"
            )
            is True
            and isinstance(
                gradient.get("fixed_nonzero_readout_gradients"),
                dict,
            )
            and set(gradient["fixed_nonzero_readout_gradients"])
            == {
                "joint_state_weight",
                "joint_hidden_bias",
                "scalar_energy_weight",
            }
            and all(
                row.get("finite") is True
                and row.get("nonzero") is True
                and float.fromhex(str(row.get("l2_hex"))) > 0.0
                for row in gradient[
                    "fixed_nonzero_readout_gradients"
                ].values()
            )
        ),
        "fixed_paet_bfa_model_contract": (
            isinstance(model_config, dict)
            and model_config.get("model_class")
            == CURELitePhaseAlignedEvidenceTransportLevelSet.__name__
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
            == CSLF_PAET_FIELD_POLICY
            and model_config.get("equation_policy")
            == CSLF_PAET_EQUATION_POLICY
            and model_config.get("flip_policy")
            == CSLF_PAET_FLIP_POLICY
            and model_config.get("transport_policy")
            == CSLF_PAET_TRANSPORT_POLICY
            and model_config.get("margin_hex")
            == COVERAGE_STATE_PAET_MARGIN.hex()
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
        "model_cache_population_and_rng_unchanged": (
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
        "gate_internal_deterministic_flags_fixed_and_restored": (
            isinstance(deterministic, dict)
            and deterministic.get("policy")
            == "gate_internal_deterministic_algorithms_and_no_tf32_v1"
            and isinstance(deterministic.get("before"), dict)
            and deterministic.get("active")
            == {
                "deterministic_algorithms": True,
                "deterministic_warn_only": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cudnn_allow_tf32": False,
                "cuda_matmul_allow_tf32": False,
            }
            and deterministic.get("after")
            == deterministic.get("before")
            and deterministic.get("restored_exactly") is True
        ),
        "read_only_zero_update_D_R_only_scope": (
            probe.get("execution_seed")
            == COVERAGE_STATE_PAET_DR_EXECUTION_SEED
            and probe.get("optimizer_constructed") is False
            and probe.get("optimizer_steps") == 0
            and probe.get("parameter_updates") == 0
            and probe.get("training_performed") is False
            and probe.get("calibration_performed") is False
            and probe.get("inference_performed") is False
            and probe.get("D_V_accessed") is False
            and probe.get("D_T_accessed") is False
            and probe.get("runtime_splits") == ["D_R"]
        ),
        "identifiability_only_no_performance_or_AUC_gate": (
            not _contains_performance_or_auc_key(probe)
        ),
    }
    return tuple(sorted(checks.items()))


@dataclass(frozen=True, eq=False)
class CoverageStatePAETDRGateReceipt:
    """Immutable evidence from one PAET seed-42 real-``D_R`` gate."""

    dataset_free_receipt: CoverageStatePAETDatasetFreeReceipt
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
        expected = recompute_coverage_state_paet_dr_checks(
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
            raise RuntimeError("PAET D_R evidence changed")

    @property
    def all_pass(self) -> bool:
        self.verify_unchanged()
        return bool(self.checks) and all(value for _, value in self.checks)

    @property
    def decision(self) -> str:
        return (
            COVERAGE_STATE_PAET_DR_PASS_DECISION
            if self.all_pass
            else COVERAGE_STATE_PAET_DR_FAIL_DECISION
        )

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        passed = bool(self.checks) and all(
            value for _, value in self.checks
        )
        return {
            "schema_version": COVERAGE_STATE_PAET_DR_GATE_SCHEMA,
            **self._evidence_payload(),
            "checks": dict(self.checks),
            "all_pass": passed,
            "decision": (
                COVERAGE_STATE_PAET_DR_PASS_DECISION
                if passed
                else COVERAGE_STATE_PAET_DR_FAIL_DECISION
            ),
            "execution": {
                "seed": COVERAGE_STATE_PAET_DR_EXECUTION_SEED,
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
                "gate_internal_deterministic_flags": True,
                "deterministic_flags_restored": True,
            },
            "identifiability_only": True,
            "performance_gate_present": False,
            "AUC_gate_present": False,
            "bound_phase_pair_contract": {
                "check": PAET_DR_BOUND_PHASE_PAIR_CHECK,
                "legal_background_policy": (
                    "state_background_mask_same_coarse_cell_chebyshev1"
                ),
                "normalization": (
                    "l2_difference_over_max_l2_first_l2_second_one_v1"
                ),
                "float32_epsilon_hex": (
                    COVERAGE_STATE_PAET_DR_FLOAT32_EPSILON.hex()
                ),
                "epsilon_multiplier": (
                    COVERAGE_STATE_PAET_DR_SEPARATION_MULTIPLIER
                ),
                "strict_threshold_hex": (
                    COVERAGE_STATE_PAET_DR_SEPARATION_THRESHOLD.hex()
                ),
                "same_pair_for_all_three_separations": True,
                (
                    "at_least_one_target_coordinate_has_legal_q_"
                    "per_group"
                ): True,
                "at_least_one_jointly_separated_pair_per_group": True,
                "two_pass_streaming": True,
            },
            "exact_collision_check_is_necessary_only": True,
            "linear_readout_feasibility_established": False,
            "positive_margin_readout_witness_present": False,
            "bounded_400_authorized_by_this_receipt_alone": False,
            "formal_800_authorized": False,
            "evidence_fingerprint": self.evidence_fingerprint,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def run_coverage_state_paet_dr_gate(
    *,
    dataset_free_receipt: CoverageStatePAETDatasetFreeReceipt,
    real_inputs: CoverageStateRealDRInputs,
    bounded_population: CoverageStateBoundedPopulation,
    device: torch.device | str = "cpu",
) -> CoverageStatePAETDRGateReceipt:
    """Run the PAET identifiability gate on already-built real inputs."""

    if not isinstance(
        dataset_free_receipt,
        CoverageStatePAETDatasetFreeReceipt,
    ):
        raise TypeError(
            "dataset_free_receipt must be "
            "CoverageStatePAETDatasetFreeReceipt"
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
        or bounded_population.seed != COVERAGE_STATE_PAET_DR_EXECUTION_SEED
    ):
        raise PermissionError("PAET D_R prerequisites did not pass")

    resolved_device = torch.device(device)
    if resolved_device.type not in {"cpu", "cuda"}:
        raise ValueError("PAET D_R gate supports only CPU or CUDA")
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

    probe = _probe(bounded_population, device=resolved_device)
    checks = recompute_coverage_state_paet_dr_checks(
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
    return CoverageStatePAETDRGateReceipt(
        dataset_free_receipt=dataset_free_receipt,
        real_inputs=real_inputs,
        bounded_population=bounded_population,
        implementation_binding=implementation,
        probe=probe,
        checks=checks,
        evidence_fingerprint=stable_fingerprint(evidence),
    )


__all__ = [
    "COVERAGE_STATE_PAET_DR_EXECUTION_SEED",
    "COVERAGE_STATE_PAET_DR_FAIL_DECISION",
    "COVERAGE_STATE_PAET_DR_FLOAT32_EPSILON",
    "COVERAGE_STATE_PAET_DR_GATE_SCHEMA",
    "COVERAGE_STATE_PAET_DR_IMPLEMENTATION_PATHS",
    "COVERAGE_STATE_PAET_DR_PASS_DECISION",
    "COVERAGE_STATE_PAET_DR_SEPARATION_MULTIPLIER",
    "COVERAGE_STATE_PAET_DR_SEPARATION_THRESHOLD",
    "CoverageStatePAETDRGateReceipt",
    "PAET_DR_BOUND_PHASE_PAIR_CHECK",
    "PAET_DR_CONFLICT_CHECK",
    "PAET_DR_GRADIENT_CHECK",
    "recompute_coverage_state_paet_dr_checks",
    "run_coverage_state_paet_dr_gate",
]
