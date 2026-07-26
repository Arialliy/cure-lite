"""D_R-only bounded model-code gate for CURE-Lite CC-SEA v8.

The outcome population, factual anchors, objective, optimizer step,
deterministic schedules, update budget, and computational thresholds are the
frozen SVEF-v4/PR-SVEF-v6 objects.  This additive executor instantiates one
fresh CC-SEA decoder and replaces only the mechanism-specific pre-training
audit.  It does not load a dataset itself and never reads D_V or D_T.

CC-SEA is audited as one state equation: an arithmetic phase common mode
creates one coverage-conditioned budget, while zero-mean phase contrast
allocates that budget over subpixels.  The audit therefore binds budget
construction, simplex allocation, mass conservation, occupancy invariance of
the allocation, phase-aware locality, both coordinate gradients, frozen
feature detachment, and the three-call/twelve-state optimizer path.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from math import ceil, isfinite, sqrt
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from ..cache.schema import stable_fingerprint
from ..config import DecoderConfig, LossConfig
from ..conservative_factorized_config import (
    ConservativeFactorizedDecoderConfig,
)
from ..conservative_factorized_decoder import (
    CURELiteConservativeFactorizedDecoder,
    ConservativeFactorizedDecoderFields,
    coverage_conserving_phase_evidence,
)
from ..crossing_factorized_decoder import crossing_recoverable_evidence
from ..factorized_config import FactorizedDecoderConfig
from ..factorized_decoder import CURELiteFactorizedDecoder
from ..losses import CURELiteLoss
from ..paired_outcome_losses import OutcomeCompleteTransitionLoss
from ..train.paired_outcome_step import outcome_complete_train_step
from ..train.paired_step import _paired_endpoint_logits
from ..train.pools import stack_state_examples
from .factorized_outcome_bounded import (
    FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE,
    FACTORIZED_FROZEN_LEARNING_RATE,
    FACTORIZED_FROZEN_SEED,
    FACTORIZED_FROZEN_WEIGHT_DECAY,
    FACTORIZED_JOINT_THRESHOLD,
    factorized_computational_gates,
)
from .paired_bounded_learnability import _deterministic_torch_runtime
from .paired_outcome_bounded import (
    COMPUTATIONAL_THRESHOLDS,
    DETERMINISM_SPECIFICATION,
    OutcomeBoundedAnchorPopulation,
    OutcomeFactualAnchorSchedule,
    _ForwardLedger,
    _evaluate_snapshot,
    _factual_batches,
    _validate_execution_inputs,
)
from .paired_outcome_inputs import PairedOutcomeInputMaterializer
from .paired_outcome_schedule import OutcomePairSchedule


CONSERVATIVE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA = (
    "cure-lite-cc-sea-v8-outcome-bounded-v1"
)
CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID = "cc_sea_v8"
CONSERVATIVE_OPERATOR_STRUCTURAL_CHECKS = (
    "v8_reference_v4_topology_and_initialization_exact",
    "v8_reference_parameter_budget_4385_and_six_tensors",
    "v8_common_mode_is_arithmetic_phase_mean",
    "v8_phase_contrast_is_zero_sum",
    "v8_budget_equals_frozen_continuously_recoverable_operator",
    "v8_phase_allocation_is_nonnegative_simplex",
    "v8_allocated_evidence_is_nonnegative",
    "v8_phase_mass_equals_single_budget",
    "v8_common_shift_preserves_phase_allocation",
    "v8_zero_mean_contrast_preserves_budget",
    "v8_zero_mean_contrast_changes_phase_allocation",
    "v8_occupancy_release_preserves_phase_allocation",
    "v8_occupancy_release_strictly_increases_budget",
    "v8_budget_coordinate_gradient_finite_nonzero",
    "v8_budget_coordinate_gradient_is_phase_common_mode",
    "v8_allocation_coordinate_gradient_finite_nonzero",
    "v8_allocation_coordinate_has_nonzero_phase_contrast",
    "v8_negative_80_recovery_gradient_finite_nonzero",
    "v8_negative_104_zero_recovery_fails_fast",
    "v8_positive_88_forward_gradient_finite",
    "v8_positive_89_nonfinite_fails_fast",
)


def _update_digest(digest: object, value: bytes) -> None:
    if not hasattr(digest, "update"):
        raise TypeError("digest must provide update()")
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def conservative_factorized_decoder_state_fingerprint(
    decoder: CURELiteConservativeFactorizedDecoder,
) -> str:
    """Hash the v8 class identity, module topology, and exact tensor state."""

    if not isinstance(decoder, CURELiteConservativeFactorizedDecoder):
        raise TypeError(
            "decoder must be CURELiteConservativeFactorizedDecoder"
        )
    digest = hashlib.sha256()
    digest.update(b"cure-lite-cc-sea-v8-decoder-state-v1")
    for name, child in decoder.named_modules():
        _update_digest(digest, name.encode("utf-8"))
        _update_digest(
            digest,
            (
                f"{type(child).__module__}.{type(child).__qualname__}"
            ).encode("utf-8"),
        )
    for name, value in sorted(decoder.state_dict().items()):
        tensor = value.detach().to(device="cpu").contiguous()
        _update_digest(digest, name.encode("utf-8"))
        _update_digest(digest, str(tensor.dtype).encode("ascii"))
        _update_digest(
            digest,
            json.dumps(
                list(tensor.shape),
                separators=(",", ":"),
            ).encode("ascii"),
        )
        raw = (
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            if tensor.numel()
            else b""
        )
        _update_digest(digest, raw)
    return digest.hexdigest()


def _maximum_abs(value: Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    detached = value.detach()
    finite = detached[torch.isfinite(detached)]
    if finite.numel() == 0:
        return 0.0
    return float(finite.abs().max().cpu())


class _ConservativeStateObserver:
    """Observe existing CC-SEA fields without another decoder computation."""

    def __init__(
        self,
        decoder: CURELiteConservativeFactorizedDecoder,
    ) -> None:
        if not isinstance(decoder, CURELiteConservativeFactorizedDecoder):
            raise TypeError(
                "decoder must be CURELiteConservativeFactorizedDecoder"
            )
        self._decoder = decoder
        self._original = decoder.forward_fields
        self._records: list[
            tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
        ] = []
        self._closed = False

        def observed_forward_fields(
            feature: Tensor,
            occupancy: Tensor,
        ) -> ConservativeFactorizedDecoderFields:
            fields = self._original(feature, occupancy)
            simplex_error = (
                fields.phase_allocation.sum(dim=1, keepdim=True) - 1.0
            ).abs().amax()
            conservation_error = (
                fields.allocated_phase_evidence.sum(
                    dim=1,
                    keepdim=True,
                )
                - fields.evidence_budget
            ).abs()
            conservation_relative_error = (
                conservation_error
                / fields.evidence_budget.abs().clamp_min(1.0)
            ).amax()
            finite = torch.stack(
                tuple(
                    torch.isfinite(value).all()
                    for value in (
                        fields.baseline_logits,
                        fields.raw_phase_evidence,
                        fields.common_mode_phase_evidence,
                        fields.occupancy_burden,
                        fields.budget_margin,
                        fields.evidence_budget,
                        fields.phase_allocation,
                        fields.allocated_phase_evidence,
                        fields.evidence,
                        fields.logits,
                        fields.local_occupancy_count,
                    )
                )
            ).all()
            self._records.append(
                (
                    fields.budget_margin.detach().abs().amax(),
                    simplex_error.detach(),
                    conservation_relative_error.detach(),
                    fields.phase_allocation.detach().amin(),
                    torch.stack(
                        (
                            fields.evidence_budget.detach().amin(),
                            fields.allocated_phase_evidence.detach().amin(),
                        )
                    ).amin(),
                    finite.detach(),
                )
            )
            return fields

        object.__setattr__(
            decoder,
            "forward_fields",
            observed_forward_fields,
        )

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> dict[str, object]:
        if self._closed:
            raise RuntimeError("CC-SEA state observer is already closed")
        self._closed = True
        object.__delattr__(self._decoder, "forward_fields")
        if not self._records:
            return {
                "maximum_observed_absolute_budget_margin": None,
                "maximum_observed_simplex_error": None,
                "maximum_observed_mass_conservation_relative_error": None,
                "minimum_observed_phase_allocation": None,
                "minimum_observed_evidence": None,
                "observed_forward_fields_calls": 0,
                "all_observed_fields_finite": True,
                "all_observed_allocations_nonnegative": True,
                "all_observed_evidence_nonnegative": True,
                "scope": (
                    "no_decoder_forward_observed_before_structural_stop"
                ),
                "additional_decoder_forward_calls": 0,
            }
        columns = tuple(
            torch.stack(tuple(record[index] for record in self._records))
            for index in range(6)
        )
        return {
            "maximum_observed_absolute_budget_margin": float(
                columns[0].max().cpu()
            ),
            "maximum_observed_simplex_error": float(
                columns[1].max().cpu()
            ),
            "maximum_observed_mass_conservation_relative_error": float(
                columns[2].max().cpu()
            ),
            "minimum_observed_phase_allocation": float(
                columns[3].min().cpu()
            ),
            "minimum_observed_evidence": float(columns[4].min().cpu()),
            "observed_forward_fields_calls": len(self._records),
            "all_observed_fields_finite": (
                all(
                    bool(torch.isfinite(column).all().cpu())
                    for column in columns[:5]
                )
                and bool(torch.all(columns[5]).cpu())
            ),
            "all_observed_allocations_nonnegative": bool(
                torch.all(columns[3] >= 0.0).cpu()
            ),
            "all_observed_evidence_nonnegative": bool(
                torch.all(columns[4] >= 0.0).cpu()
            ),
            "scope": (
                "all_existing_decoder_forward_fields_calls_without_"
                "additional_decoder_computation"
            ),
            "additional_decoder_forward_calls": 0,
        }


class _InputDetachLedger:
    """Count calls/states and reject any decoder input retaining autograd."""

    def __init__(
        self,
        decoder: CURELiteConservativeFactorizedDecoder,
    ) -> None:
        self.calls = 0
        self.states = 0
        self.requires_grad_violations = 0
        self._handle = decoder.register_forward_pre_hook(self._record)

    def _record(
        self,
        module: torch.nn.Module,
        inputs: tuple[object, ...],
    ) -> None:
        del module
        if len(inputs) != 2 or not isinstance(inputs[0], Tensor):
            raise RuntimeError("CC-SEA input ledger received invalid inputs")
        feature = inputs[0]
        self.calls += 1
        self.states += int(feature.shape[0])
        self.requires_grad_violations += int(feature.requires_grad)

    def snapshot(self) -> tuple[int, int, int]:
        return (
            self.calls,
            self.states,
            self.requires_grad_violations,
        )

    def close(self) -> None:
        self._handle.remove()


def _probe_rejected(value: float, *, device: torch.device) -> bool:
    probe = torch.tensor(
        [value],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    try:
        crossing_recoverable_evidence(probe)
    except ValueError:
        return True
    return False


def _audit_conservative_operator_contract(
    *,
    device: torch.device,
    decoder: CURELiteConservativeFactorizedDecoder | None = None,
    initialization_seed: int = 8321,
) -> dict[str, object]:
    """Evaluate the frozen CC-SEA equation, coordinates, and numerics."""

    if decoder is not None and not isinstance(
        decoder,
        CURELiteConservativeFactorizedDecoder,
    ):
        raise TypeError(
            "decoder must be CURELiteConservativeFactorizedDecoder or None"
        )
    if isinstance(initialization_seed, bool) or not isinstance(
        initialization_seed,
        int,
    ):
        raise TypeError("initialization_seed must be an integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        if decoder is None:
            audited_v8 = CURELiteConservativeFactorizedDecoder(
                ConservativeFactorizedDecoderConfig(64, 4)
            )
            torch.manual_seed(8321)
            reference_v4 = CURELiteFactorizedDecoder(
                FactorizedDecoderConfig(64, 4)
            )
        else:
            audited_v8 = decoder
            reference_v4 = CURELiteFactorizedDecoder(
                decoder.config.to_v4_topology_config()
            )
    topology_exact = (
        tuple(reference_v4.state_dict())
        == tuple(audited_v8.state_dict())
        and all(
            torch.equal(
                reference_v4.state_dict()[name],
                value.detach().cpu(),
            )
            for name, value in audited_v8.state_dict().items()
        )
        and tuple(
            type(module) for module in tuple(reference_v4.modules())[1:]
        )
        == tuple(
            type(module) for module in tuple(audited_v8.modules())[1:]
        )
    )
    reference_topology = FactorizedDecoderConfig(64, 4)
    reference_parameter_count = reference_topology.expected_parameter_count
    reference_parameter_tensors = 6

    raw = torch.tensor(
        [[[[1.0]], [[-1.0]], [[0.5]], [[-0.5]]]],
        dtype=torch.float64,
        device=device,
    )
    burden = torch.zeros(
        (1, 1, 1, 1),
        dtype=torch.float64,
        device=device,
    )
    base = coverage_conserving_phase_evidence(raw, burden)
    shifted = coverage_conserving_phase_evidence(raw + 2.0, burden)
    zero_mean = torch.tensor(
        [[[[1.5]], [[-1.5]], [[0.5]], [[-0.5]]]],
        dtype=torch.float64,
        device=device,
    )
    contrasted = coverage_conserving_phase_evidence(
        raw + zero_mean,
        burden,
    )
    release_raw = raw + 0.5
    occupied = coverage_conserving_phase_evidence(
        release_raw,
        torch.full_like(
            burden,
            float(torch.log(torch.tensor(2.0))),
        ),
    )
    released = coverage_conserving_phase_evidence(
        release_raw,
        burden,
    )
    states = (base, shifted, contrasted, occupied, released)
    phase_contrast_error = max(
        float(
            (
                state_raw - state[0]
            ).sum(dim=1, keepdim=True).abs().max().cpu()
        )
        for state_raw, state in (
            (raw, base),
            (raw + 2.0, shifted),
            (raw + zero_mean, contrasted),
            (release_raw, occupied),
            (release_raw, released),
        )
    )
    simplex_error = max(
        float(
            (state[3].sum(dim=1, keepdim=True) - 1.0)
            .abs()
            .max()
            .cpu()
        )
        for state in states
    )
    mass_error = max(
        float(
            (
                state[4].sum(dim=1, keepdim=True) - state[2]
            )
            .abs()
            .max()
            .cpu()
        )
        for state in states
    )
    budget_exact = all(
        torch.equal(
            state[2],
            crossing_recoverable_evidence(state[1]),
        )
        for state in states
    )

    raw_budget = torch.tensor(
        [[[[1.0]], [[0.5]], [[0.0]], [[-0.5]]]],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    budget_fields = coverage_conserving_phase_evidence(
        raw_budget,
        burden,
    )
    budget_gradient = torch.autograd.grad(
        budget_fields[2].sum(),
        raw_budget,
    )[0]
    raw_allocation = raw_budget.detach().clone().requires_grad_(True)
    allocation_fields = coverage_conserving_phase_evidence(
        raw_allocation,
        burden,
    )
    allocation_gradient = torch.autograd.grad(
        allocation_fields[4][:, 0].sum(),
        raw_allocation,
    )[0]
    allocation_gradient_contrast = (
        allocation_gradient
        - allocation_gradient.mean(dim=1, keepdim=True)
    )

    supported = torch.tensor(
        [-80.0, 88.0],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    supported_evidence = crossing_recoverable_evidence(supported)
    supported_gradient = torch.autograd.grad(
        supported_evidence.sum(),
        supported,
    )[0]
    negative_104_rejected = _probe_rejected(-104.0, device=device)
    positive_89_rejected = _probe_rejected(89.0, device=device)

    checks = {
        "v8_reference_v4_topology_and_initialization_exact": (
            topology_exact
        ),
        "v8_reference_parameter_budget_4385_and_six_tensors": (
            reference_parameter_count == 4385
            and reference_parameter_tensors == 6
        ),
        "v8_common_mode_is_arithmetic_phase_mean": all(
            torch.equal(
                state[0],
                state_raw.mean(dim=1, keepdim=True),
            )
            for state_raw, state in (
                (raw, base),
                (raw + 2.0, shifted),
                (raw + zero_mean, contrasted),
                (release_raw, occupied),
                (release_raw, released),
            )
        ),
        "v8_phase_contrast_is_zero_sum": (
            phase_contrast_error <= 1.0e-12
        ),
        "v8_budget_equals_frozen_continuously_recoverable_operator": (
            budget_exact
        ),
        "v8_phase_allocation_is_nonnegative_simplex": (
            simplex_error <= 1.0e-12
            and all(bool(torch.all(state[3] >= 0.0).cpu()) for state in states)
        ),
        "v8_allocated_evidence_is_nonnegative": all(
            bool(torch.all(state[4] >= 0.0).cpu()) for state in states
        ),
        "v8_phase_mass_equals_single_budget": mass_error <= 1.0e-12,
        "v8_common_shift_preserves_phase_allocation": torch.allclose(
            shifted[3],
            base[3],
            rtol=0.0,
            atol=1.0e-15,
        ),
        "v8_zero_mean_contrast_preserves_budget": torch.equal(
            contrasted[2],
            base[2],
        ),
        "v8_zero_mean_contrast_changes_phase_allocation": (
            float((contrasted[3] - base[3]).abs().max().cpu())
            > 1.0e-3
        ),
        "v8_occupancy_release_preserves_phase_allocation": torch.equal(
            occupied[3],
            released[3],
        ),
        "v8_occupancy_release_strictly_increases_budget": bool(
            torch.all(released[2] > occupied[2]).cpu()
        ),
        "v8_budget_coordinate_gradient_finite_nonzero": (
            bool(torch.isfinite(budget_gradient).all().cpu())
            and float(budget_gradient.norm().cpu()) > 0.0
        ),
        "v8_budget_coordinate_gradient_is_phase_common_mode": (
            float(
                (
                    budget_gradient
                    - budget_gradient.mean(dim=1, keepdim=True)
                )
                .abs()
                .max()
                .cpu()
            )
            <= 1.0e-12
        ),
        "v8_allocation_coordinate_gradient_finite_nonzero": (
            bool(torch.isfinite(allocation_gradient).all().cpu())
            and float(allocation_gradient.norm().cpu()) > 0.0
        ),
        "v8_allocation_coordinate_has_nonzero_phase_contrast": (
            float(allocation_gradient_contrast.norm().cpu()) > 0.0
        ),
        "v8_negative_80_recovery_gradient_finite_nonzero": (
            float(supported_evidence[0].detach().cpu()) == 0.0
            and bool(torch.isfinite(supported_gradient[0]).cpu())
            and float(supported_gradient[0].detach().cpu()) > 0.0
        ),
        "v8_negative_104_zero_recovery_fails_fast": (
            negative_104_rejected
        ),
        "v8_positive_88_forward_gradient_finite": (
            bool(torch.isfinite(supported_evidence[1]).cpu())
            and bool(torch.isfinite(supported_gradient[1]).cpu())
            and float(supported_evidence[1].detach().cpu()) > 0.0
            and float(supported_gradient[1].detach().cpu()) > 0.0
        ),
        "v8_positive_89_nonfinite_fails_fast": positive_89_rejected,
    }
    if tuple(checks) != CONSERVATIVE_OPERATOR_STRUCTURAL_CHECKS:
        raise AssertionError("v8 operator structural check set drifted")
    return {
        "scope": "CC_SEA_v8_frozen_state_equation_and_numeric_contract",
        "reference_topology": {
            "feature_channels": 64,
            "feature_stride": 4,
            "parameter_count": reference_parameter_count,
            "parameter_tensors": reference_parameter_tensors,
            "state_keys": list(audited_v8.state_dict()),
        },
        "coordinate_audit": {
            "budget_gradient_l2": float(budget_gradient.norm().cpu()),
            "allocation_gradient_l2": float(
                allocation_gradient.norm().cpu()
            ),
            "allocation_contrast_gradient_l2": float(
                allocation_gradient_contrast.norm().cpu()
            ),
        },
        "equation_errors": {
            "phase_contrast_sum_max_abs_error": phase_contrast_error,
            "simplex_max_abs_error": simplex_error,
            "mass_conservation_max_abs_error": mass_error,
            "zero_mean_contrast_allocation_max_abs_change": float(
                (contrasted[3] - base[3]).abs().max().cpu()
            ),
            "occupancy_release_budget_delta": float(
                (released[2] - occupied[2]).item()
            ),
        },
        "numeric_probes": {
            "negative_finite_nonzero_recovery": -80.0,
            "negative_zero_recovery_fail_fast": -104.0,
            "positive_largest_finite": 88.0,
            "positive_first_nonfinite_fail_fast": 89.0,
            "supported_forward": [
                float(value) for value in supported_evidence.detach().cpu()
            ],
            "supported_gradient": [
                float(value) for value in supported_gradient.detach().cpu()
            ],
            "negative_104_rejected": negative_104_rejected,
            "positive_89_rejected": positive_89_rejected,
        },
        "checks": checks,
        "all_pass": all(checks.values()),
        "autograd_gradient_calls": 3,
        "training_performed": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }


def _phase_aware_output_support(
    native_support: Tensor,
    *,
    phase_channels: int,
    feature_stride: int,
) -> Tensor:
    if (
        native_support.ndim != 4
        or native_support.shape[1] != 1
        or native_support.dtype != torch.bool
    ):
        raise ValueError("native_support must be bool [B,1,h,w]")
    expanded = native_support.expand(
        -1,
        phase_channels,
        -1,
        -1,
    )
    return F.pixel_shuffle(
        expanded.to(dtype=torch.float32),
        feature_stride,
    ).to(dtype=torch.bool)


def _state_equation_errors(
    fields: ConservativeFactorizedDecoderFields,
) -> dict[str, object]:
    phase_contrast = (
        fields.raw_phase_evidence
        - fields.common_mode_phase_evidence
    )
    expected_common = fields.raw_phase_evidence.mean(
        dim=1,
        keepdim=True,
    )
    expected_budget = crossing_recoverable_evidence(
        fields.budget_margin
    )
    expected_burden = torch.log1p(fields.local_occupancy_count)
    allocation_error = (
        fields.phase_allocation.sum(dim=1, keepdim=True) - 1.0
    ).abs()
    mass_error = (
        fields.allocated_phase_evidence.sum(dim=1, keepdim=True)
        - fields.evidence_budget
    ).abs()
    mass_relative_error = (
        mass_error / fields.evidence_budget.abs().clamp_min(1.0)
    )
    return {
        "common_mode_max_abs_error": _maximum_abs(
            fields.common_mode_phase_evidence - expected_common
        ),
        "phase_contrast_sum_max_abs_error": _maximum_abs(
            phase_contrast.sum(dim=1, keepdim=True)
        ),
        "occupancy_burden_max_abs_error": _maximum_abs(
            fields.occupancy_burden - expected_burden
        ),
        "allocation_sum_max_abs_error": _maximum_abs(allocation_error),
        "allocation_minimum": float(
            fields.phase_allocation.detach().min().cpu()
        ),
        "evidence_budget_minimum": float(
            fields.evidence_budget.detach().min().cpu()
        ),
        "allocated_evidence_minimum": float(
            fields.allocated_phase_evidence.detach().min().cpu()
        ),
        "mass_conservation_max_abs_error": _maximum_abs(mass_error),
        "mass_conservation_max_relative_error": _maximum_abs(
            mass_relative_error
        ),
        "budget_forward_exact": torch.equal(
            fields.evidence_budget,
            expected_budget,
        ),
        "logit_composition_exact": torch.equal(
            fields.logits,
            fields.baseline_logits + fields.evidence,
        ),
    }


def _audit_phase_aware_locality(
    decoder: CURELiteConservativeFactorizedDecoder,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Exercise changed and unchanged phase support on a controlled 5x5 grid."""

    parameter = next(decoder.parameters())
    feature_size = (5, 5)
    output_size = tuple(
        value * decoder.config.feature_stride for value in feature_size
    )
    feature = torch.linspace(
        -1.0,
        1.0,
        steps=(
            decoder.config.feature_channels
            * feature_size[0]
            * feature_size[1]
        ),
        dtype=parameter.dtype,
        device=device,
    ).reshape(
        1,
        decoder.config.feature_channels,
        *feature_size,
    )
    feature.requires_grad_(True)
    occupancy_plus = torch.zeros(
        1,
        1,
        *output_size,
        dtype=torch.bool,
        device=device,
    )
    occupancy_plus[
        :,
        :,
        output_size[0] // 2,
        output_size[1] // 2,
    ] = True
    occupancy_minus = torch.zeros_like(occupancy_plus)

    decoder.eval()
    plus = decoder.forward_fields(feature, occupancy_plus)
    identity_again = decoder(feature, occupancy_plus.clone())
    minus = decoder.forward_fields(feature, occupancy_minus)
    feature_gradient = torch.autograd.grad(
        plus.logits.sum(),
        feature,
        allow_unused=True,
    )[0]

    phase_channels = decoder.config.phase_channels
    phase_contrast = torch.linspace(
        -0.3,
        0.3,
        steps=phase_channels,
        dtype=parameter.dtype,
        device=device,
    )
    phase_contrast = phase_contrast - phase_contrast.mean()
    raw_phase_evidence = (
        2.0
        + phase_contrast.reshape(1, phase_channels, 1, 1)
    ).expand(1, phase_channels, *feature_size).contiguous()
    controlled_plus = coverage_conserving_phase_evidence(
        raw_phase_evidence,
        plus.occupancy_burden,
    )
    controlled_minus = coverage_conserving_phase_evidence(
        raw_phase_evidence,
        minus.occupancy_burden,
    )

    count_release = (
        plus.local_occupancy_count - minus.local_occupancy_count
    )
    support = _phase_aware_output_support(
        count_release > 0.0,
        phase_channels=phase_channels,
        feature_stride=decoder.config.feature_stride,
    )
    outside = ~support
    controlled_plus_evidence = F.pixel_shuffle(
        controlled_plus[4],
        decoder.config.feature_stride,
    )
    controlled_minus_evidence = F.pixel_shuffle(
        controlled_minus[4],
        decoder.config.feature_stride,
    )
    controlled_delta = (
        controlled_minus_evidence - controlled_plus_evidence
    )
    actual_delta = minus.logits - plus.logits
    probability_delta = (
        torch.sigmoid(minus.logits) - torch.sigmoid(plus.logits)
    )
    conservation_relative_error = max(
        _maximum_abs(
            (
                state[4].sum(dim=1, keepdim=True) - state[2]
            )
            / state[2].abs().clamp_min(1.0)
        )
        for state in (controlled_plus, controlled_minus)
    )
    support_pixels = int(support.sum().detach().cpu())
    outside_pixels = int(outside.sum().detach().cpu())
    controlled_strict = (
        controlled_minus[2] > controlled_plus[2]
    )
    checks = {
        "identity_endpoint_exact": torch.equal(
            plus.logits,
            identity_again,
        ),
        "feature_probe_requires_grad": feature.requires_grad,
        "feature_detached_inside_decoder": feature_gradient is None,
        "phase_aware_changed_support_nonempty": support_pixels > 0,
        "phase_aware_unchanged_support_nonempty": outside_pixels > 0,
        "controlled_allocation_occupancy_invariant": torch.equal(
            controlled_plus[3],
            controlled_minus[3],
        ),
        "controlled_budget_deletion_monotone": bool(
            torch.all(controlled_minus[2] >= controlled_plus[2]).cpu()
        ),
        "controlled_budget_deletion_strict_on_changed_cells": bool(
            torch.all(controlled_strict[count_release > 0.0]).cpu()
        ),
        "controlled_allocated_deletion_monotone": bool(
            torch.all(controlled_minus[4] >= controlled_plus[4]).cpu()
        ),
        "controlled_outside_count_support_exact": torch.equal(
            controlled_delta[outside],
            torch.zeros_like(controlled_delta[outside]),
        ),
        "controlled_inside_count_support_strict": bool(
            torch.all(controlled_delta[support] > 0.0).cpu()
        ),
        "actual_decoder_outside_count_support_exact": torch.equal(
            actual_delta[outside],
            torch.zeros_like(actual_delta[outside]),
        ),
        "actual_probability_outside_count_support_exact": torch.equal(
            probability_delta[outside],
            torch.zeros_like(probability_delta[outside]),
        ),
        "controlled_common_mode_exact": torch.equal(
            controlled_minus[0],
            torch.full_like(controlled_minus[0], 2.0),
        ),
        "controlled_mass_conservation": (
            conservation_relative_error <= 1.0e-6
        ),
        "all_fields_finite": all(
            bool(torch.isfinite(value).all().detach().cpu())
            for value in (
                plus.logits,
                minus.logits,
                count_release,
                controlled_delta,
                actual_delta,
                probability_delta,
            )
        ),
    }
    return {
        "probe_kind": "controlled_positive_budget_phase_aware_5x5",
        "feature_grid": list(feature_size),
        "output_grid": list(output_size),
        "changed_support_pixels": support_pixels,
        "unchanged_support_pixels": outside_pixels,
        "controlled_common_mode": 2.0,
        "controlled_phase_contrast_minimum": float(
            phase_contrast.min().detach().cpu()
        ),
        "controlled_phase_contrast_maximum": float(
            phase_contrast.max().detach().cpu()
        ),
        "controlled_support_max_abs_delta": _maximum_abs(
            controlled_delta[support]
        ),
        "controlled_outside_max_abs_delta": _maximum_abs(
            controlled_delta[outside]
        ),
        "actual_outside_max_abs_logit_delta": _maximum_abs(
            actual_delta[outside]
        ),
        "actual_outside_max_abs_probability_delta": _maximum_abs(
            probability_delta[outside]
        ),
        "mass_conservation_max_relative_error": (
            conservation_relative_error
        ),
        "feature_gradient_is_none": feature_gradient is None,
        "checks": checks,
        "all_pass": all(checks.values()),
        "decoder_calls": 3,
        "decoder_state_evaluations": 3,
    }


def _select_dual_endpoint_probe_ids(
    schedule: OutcomePairSchedule,
) -> tuple[str, str]:
    clean = tuple(
        pair for pair in schedule.pairs if pair.pair_kind == "clean_positive"
    )
    component = tuple(
        pair for pair in schedule.pairs if pair.pair_kind == "component_null"
    )
    for clean_pair in clean:
        for component_pair in component:
            if clean_pair.sample_id != component_pair.sample_id:
                return clean_pair.pair_id, component_pair.pair_id
    raise RuntimeError(
        "CC-SEA dual-endpoint audit requires source-disjoint pair roles"
    )


def _audit_dual_endpoint_gradients(
    decoder: CURELiteConservativeFactorizedDecoder,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Require both endpoint scores for clean and null roles to affect loss."""

    pair_ids = _select_dual_endpoint_probe_ids(schedule)
    outcome = materializer.materialize(pair_ids, device=device)
    batch = outcome.pair_batch
    observed_batches: list[int] = []

    def observe(
        module: torch.nn.Module,
        inputs: tuple[object, ...],
    ) -> None:
        del module
        if not inputs or not isinstance(inputs[0], Tensor):
            raise RuntimeError("invalid dual-endpoint decoder input")
        observed_batches.append(int(inputs[0].shape[0]))

    handle = decoder.register_forward_pre_hook(observe)
    try:
        logits_plus, logits_minus = _paired_endpoint_logits(
            decoder,
            feature=batch.feature,
            occupancy_plus=batch.occupancy_plus,
            occupancy_minus=batch.occupancy_minus,
        )
    finally:
        handle.remove()
    plus_leaf = logits_plus.detach().clone().requires_grad_(True)
    minus_leaf = logits_minus.detach().clone().requires_grad_(True)
    criterion = OutcomeCompleteTransitionLoss(LossConfig()).to(device)
    loss = criterion(
        plus_leaf,
        minus_leaf,
        outcome.completion_plus,
        batch.occupancy_plus,
        outcome.gt_union,
        batch.label_increment,
        batch.image_valid_mask,
        outcome.intervention_footprint,
    )["total"]
    plus_gradient, minus_gradient = torch.autograd.grad(
        loss,
        (plus_leaf, minus_leaf),
    )
    records: list[dict[str, object]] = []
    for index, pair_kind in enumerate(batch.pair_kinds):
        plus_local = plus_gradient[index]
        minus_local = minus_gradient[index]
        records.append(
            {
                "pair_id": batch.pair_ids[index],
                "sample_id": batch.sample_ids[index],
                "pair_kind": pair_kind,
                "plus_gradient_finite": bool(
                    torch.isfinite(plus_local).all().detach().cpu()
                ),
                "plus_gradient_nonzero_count": int(
                    torch.count_nonzero(plus_local).detach().cpu()
                ),
                "plus_gradient_l2": float(
                    plus_local.detach().double().norm().cpu()
                ),
                "minus_gradient_finite": bool(
                    torch.isfinite(minus_local).all().detach().cpu()
                ),
                "minus_gradient_nonzero_count": int(
                    torch.count_nonzero(minus_local).detach().cpu()
                ),
                "minus_gradient_l2": float(
                    minus_local.detach().double().norm().cpu()
                ),
            }
        )
    checks = {
        "clean_and_component_roles_present": tuple(
            record["pair_kind"] for record in records
        )
        == ("clean_positive", "component_null"),
        "one_2B_endpoint_forward": observed_batches == [4],
        "both_endpoints_finite_nonzero_for_each_pair_role": all(
            record["plus_gradient_finite"] is True
            and record["plus_gradient_nonzero_count"] > 0
            and record["plus_gradient_l2"] > 0.0
            and record["minus_gradient_finite"] is True
            and record["minus_gradient_nonzero_count"] > 0
            and record["minus_gradient_l2"] > 0.0
            for record in records
        ),
    }
    return {
        "pair_ids": list(pair_ids),
        "pair_kinds": list(batch.pair_kinds),
        "observed_decoder_batch_sizes": observed_batches,
        "records": records,
        "checks": checks,
        "all_pass": all(checks.values()),
        "decoder_calls": 1,
        "decoder_state_evaluations": 4,
        "autograd_gradient_calls": 1,
    }


def audit_conservative_outcome_population(
    decoder: CURELiteConservativeFactorizedDecoder,
    population: OutcomeBoundedAnchorPopulation,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    *,
    device: torch.device,
    chunk_size: int,
) -> dict[str, object]:
    """Audit the CC-SEA equation on every frozen D_R outcome pair."""

    if not isinstance(decoder, CURELiteConservativeFactorizedDecoder):
        raise TypeError(
            "decoder must be CURELiteConservativeFactorizedDecoder"
        )
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    pair_ids = tuple(pair.pair_id for pair in schedule.pairs)
    records: list[dict[str, object]] = []
    zero_feature_max = 0.0
    raw_occupancy_delta_max = 0.0
    common_occupancy_delta_max = 0.0
    baseline_occupancy_delta_max = 0.0
    allocation_occupancy_delta_max = 0.0
    outside_logit_max = 0.0
    outside_probability_max = 0.0
    count_burden_support_mismatch_pixels = 0
    count_monotonicity_violations = 0
    burden_monotonicity_violations = 0
    budget_monotonicity_violations = 0
    allocated_monotonicity_violations = 0
    logit_monotonicity_violations = 0
    probability_monotonicity_violations = 0
    nonfinite_field_values = 0
    resize_count = 0
    common_mode_error_max = 0.0
    phase_contrast_sum_error_max = 0.0
    burden_formula_error_max = 0.0
    simplex_error_max = 0.0
    mass_conservation_error_max = 0.0
    mass_conservation_relative_error_max = 0.0
    phase_mass_delta_error_max = 0.0
    budget_forward_mismatch_states = 0
    logit_composition_mismatch_states = 0
    minimum_allocation = float("inf")
    minimum_budget = float("inf")
    minimum_allocated_evidence = float("inf")
    clean_full_D_reachable = 0
    clean_D_reachable_pixels = 0
    clean_D_total_pixels = 0
    clean_nonempty_H_pairs = 0
    component_positive_support_pairs = 0
    clean_support_fractions: list[float] = []
    component_support_fractions: list[float] = []
    changed_support_pixels = 0
    unchanged_support_pixels = 0
    audit_decoder_calls = 0
    audit_decoder_state_evaluations = 0

    decoder.eval()
    with torch.no_grad():
        for start in range(0, len(pair_ids), chunk_size):
            selected = pair_ids[start : start + chunk_size]
            batch = materializer.materialize(selected, device=device)
            feature = batch.pair_batch.feature
            plus = decoder.forward_fields(
                feature,
                batch.pair_batch.occupancy_plus,
            )
            minus = decoder.forward_fields(
                feature,
                batch.pair_batch.occupancy_minus,
            )
            zero = torch.zeros_like(feature)
            zero_plus = decoder(
                zero,
                batch.pair_batch.occupancy_plus,
            )
            zero_minus = decoder(
                zero,
                batch.pair_batch.occupancy_minus,
            )
            audit_decoder_calls += 4
            audit_decoder_state_evaluations += 4 * len(selected)

            zero_feature_max = max(
                zero_feature_max,
                _maximum_abs(zero_minus - zero_plus),
            )
            raw_occupancy_delta_max = max(
                raw_occupancy_delta_max,
                _maximum_abs(
                    minus.raw_phase_evidence
                    - plus.raw_phase_evidence
                ),
            )
            common_occupancy_delta_max = max(
                common_occupancy_delta_max,
                _maximum_abs(
                    minus.common_mode_phase_evidence
                    - plus.common_mode_phase_evidence
                ),
            )
            baseline_occupancy_delta_max = max(
                baseline_occupancy_delta_max,
                _maximum_abs(
                    minus.baseline_logits - plus.baseline_logits
                ),
            )
            allocation_occupancy_delta_max = max(
                allocation_occupancy_delta_max,
                _maximum_abs(
                    minus.phase_allocation - plus.phase_allocation
                ),
            )

            count_release = (
                plus.local_occupancy_count
                - minus.local_occupancy_count
            )
            burden_release = (
                plus.occupancy_burden - minus.occupancy_burden
            )
            budget_release = (
                minus.evidence_budget - plus.evidence_budget
            )
            allocated_release = (
                minus.allocated_phase_evidence
                - plus.allocated_phase_evidence
            )
            native_support = count_release > 0.0
            burden_support = burden_release > 0.0
            output_support = _phase_aware_output_support(
                native_support,
                phase_channels=decoder.config.phase_channels,
                feature_stride=decoder.config.feature_stride,
            )
            outside = ~output_support
            count_burden_support_mismatch_pixels += int(
                (native_support != burden_support).sum().cpu()
            )
            logit_delta = minus.logits - plus.logits
            probability_delta = (
                torch.sigmoid(minus.logits)
                - torch.sigmoid(plus.logits)
            )
            outside_logit_max = max(
                outside_logit_max,
                _maximum_abs(logit_delta[outside]),
            )
            outside_probability_max = max(
                outside_probability_max,
                _maximum_abs(probability_delta[outside]),
            )
            count_monotonicity_violations += int(
                (count_release < 0.0).sum().cpu()
            )
            burden_monotonicity_violations += int(
                (burden_release < 0.0).sum().cpu()
            )
            budget_monotonicity_violations += int(
                (budget_release < 0.0).sum().cpu()
            )
            allocated_monotonicity_violations += int(
                (allocated_release < 0.0).sum().cpu()
            )
            logit_monotonicity_violations += int(
                (logit_delta < 0.0).sum().cpu()
            )
            probability_monotonicity_violations += int(
                (probability_delta < 0.0).sum().cpu()
            )

            endpoint_equations = (
                _state_equation_errors(plus),
                _state_equation_errors(minus),
            )
            common_mode_error_max = max(
                common_mode_error_max,
                *(
                    float(value["common_mode_max_abs_error"])
                    for value in endpoint_equations
                ),
            )
            phase_contrast_sum_error_max = max(
                phase_contrast_sum_error_max,
                *(
                    float(value["phase_contrast_sum_max_abs_error"])
                    for value in endpoint_equations
                ),
            )
            burden_formula_error_max = max(
                burden_formula_error_max,
                *(
                    float(value["occupancy_burden_max_abs_error"])
                    for value in endpoint_equations
                ),
            )
            simplex_error_max = max(
                simplex_error_max,
                *(
                    float(value["allocation_sum_max_abs_error"])
                    for value in endpoint_equations
                ),
            )
            mass_conservation_error_max = max(
                mass_conservation_error_max,
                *(
                    float(value["mass_conservation_max_abs_error"])
                    for value in endpoint_equations
                ),
            )
            mass_conservation_relative_error_max = max(
                mass_conservation_relative_error_max,
                *(
                    float(
                        value[
                            "mass_conservation_max_relative_error"
                        ]
                    )
                    for value in endpoint_equations
                ),
            )
            minimum_allocation = min(
                minimum_allocation,
                *(
                    float(value["allocation_minimum"])
                    for value in endpoint_equations
                ),
            )
            minimum_budget = min(
                minimum_budget,
                *(
                    float(value["evidence_budget_minimum"])
                    for value in endpoint_equations
                ),
            )
            minimum_allocated_evidence = min(
                minimum_allocated_evidence,
                *(
                    float(value["allocated_evidence_minimum"])
                    for value in endpoint_equations
                ),
            )
            budget_forward_mismatch_states += sum(
                value["budget_forward_exact"] is not True
                for value in endpoint_equations
            )
            logit_composition_mismatch_states += sum(
                value["logit_composition_exact"] is not True
                for value in endpoint_equations
            )
            phase_mass_delta = (
                minus.allocated_phase_evidence.sum(
                    dim=1,
                    keepdim=True,
                )
                - plus.allocated_phase_evidence.sum(
                    dim=1,
                    keepdim=True,
                )
            )
            phase_mass_delta_error_max = max(
                phase_mass_delta_error_max,
                _maximum_abs(
                    (
                        phase_mass_delta - budget_release
                    )
                    / budget_release.abs().clamp_min(1.0)
                ),
            )
            nonfinite_field_values += sum(
                int((~torch.isfinite(value)).sum().cpu())
                for value in (
                    plus.baseline_logits,
                    plus.raw_phase_evidence,
                    plus.common_mode_phase_evidence,
                    plus.occupancy_burden,
                    plus.budget_margin,
                    plus.evidence_budget,
                    plus.phase_allocation,
                    plus.allocated_phase_evidence,
                    plus.evidence,
                    plus.logits,
                    plus.local_occupancy_count,
                    minus.baseline_logits,
                    minus.raw_phase_evidence,
                    minus.common_mode_phase_evidence,
                    minus.occupancy_burden,
                    minus.budget_margin,
                    minus.evidence_budget,
                    minus.phase_allocation,
                    minus.allocated_phase_evidence,
                    minus.evidence,
                    minus.logits,
                    minus.local_occupancy_count,
                    count_release,
                    burden_release,
                    budget_release,
                    allocated_release,
                    logit_delta,
                    probability_delta,
                )
            )
            resize_count += int(plus.field_resize_applied)
            resize_count += int(minus.field_resize_applied)

            valid = batch.pair_batch.image_valid_mask
            changed_support_pixels += int(
                (output_support & valid).sum().cpu()
            )
            unchanged_support_pixels += int(
                (outside & valid).sum().cpu()
            )
            for index, pair_id in enumerate(selected):
                kind = batch.pair_batch.pair_kinds[index]
                response = batch.response_stratum[index]
                local_zero = batch.local_zero_stratum[index]
                response_count = int(response.sum().cpu())
                h_count = int(local_zero.sum().cpu())
                reachable = response & output_support[index]
                reachable_count = int(reachable.sum().cpu())
                valid_count = int(valid[index].sum().cpu())
                support_count = int(
                    (output_support[index] & valid[index]).sum().cpu()
                )
                support_fraction = support_count / valid_count
                if kind == "clean_positive":
                    clean_nonempty_H_pairs += int(h_count > 0)
                    clean_D_total_pixels += response_count
                    clean_D_reachable_pixels += reachable_count
                    clean_full_D_reachable += int(
                        reachable_count == response_count
                    )
                    clean_support_fractions.append(support_fraction)
                else:
                    component_support_fractions.append(support_fraction)
                    component_positive_support_pairs += int(
                        support_count > 0
                    )
                records.append(
                    {
                        "pair_id": pair_id,
                        "sample_id": batch.pair_batch.sample_ids[index],
                        "pair_kind": kind,
                        "D_pixels": response_count,
                        "H_pixels": h_count,
                        "D_reachable_pixels": reachable_count,
                        "D_reachability_fraction": (
                            None
                            if response_count == 0
                            else reachable_count / response_count
                        ),
                        "phase_aware_count_support_pixels": support_count,
                        "phase_aware_count_support_fraction": (
                            support_fraction
                        ),
                        "field_resize_applied": (
                            plus.field_resize_applied
                            or minus.field_resize_applied
                        ),
                    }
                )

    locality_probe = _audit_phase_aware_locality(
        decoder,
        device=device,
    )
    audit_decoder_calls += int(locality_probe["decoder_calls"])
    audit_decoder_state_evaluations += int(
        locality_probe["decoder_state_evaluations"]
    )
    dual_endpoint = _audit_dual_endpoint_gradients(
        decoder,
        schedule,
        materializer,
        device=device,
    )
    audit_decoder_calls += int(dual_endpoint["decoder_calls"])
    audit_decoder_state_evaluations += int(
        dual_endpoint["decoder_state_evaluations"]
    )

    decoder.eval()
    with torch.no_grad():
        factual = stack_state_examples(
            population.factual_miss,
            device=device,
        )
        factual_fields = decoder.forward_fields(
            factual.feature,
            factual.occupancy,
        )
        audit_decoder_calls += 1
        audit_decoder_state_evaluations += len(population.factual_miss)
        factual_recovery_native = torch.exp(
            factual_fields.budget_margin
        )
        factual_recoverable_native = (
            torch.isfinite(factual_recovery_native)
            & (factual_recovery_native > 0.0)
        )
        factual_recoverable = _phase_aware_output_support(
            factual_recoverable_native,
            phase_channels=decoder.config.phase_channels,
            feature_stride=decoder.config.feature_stride,
        )
        factual_target = factual.target > 0.5
        factual_total_by_anchor = factual_target.flatten(1).sum(dim=1)
        factual_reachable_by_anchor = (
            factual_target & factual_recoverable
        ).flatten(1).sum(dim=1)
        factual_full = (
            factual_total_by_anchor == factual_reachable_by_anchor
        )

    clean_pair_count = sum(
        record["pair_kind"] == "clean_positive" for record in records
    )
    component_pair_count = len(records) - clean_pair_count
    if len(records) != len(pair_ids):
        raise RuntimeError(
            "CC-SEA structural audit did not cover all pairs"
        )
    if (
        clean_pair_count < 1
        or component_pair_count < 1
        or clean_D_total_pixels < 1
        or not clean_support_fractions
        or not component_support_fractions
    ):
        raise RuntimeError(
            "CC-SEA structural audit found an empty required stratum"
        )

    factual_total = int(factual_total_by_anchor.sum().cpu())
    factual_reachable = int(factual_reachable_by_anchor.sum().cpu())
    expected_audit_decoder_calls = (
        4 * ceil(len(pair_ids) / chunk_size) + 5
    )
    expected_audit_decoder_state_evaluations = (
        4 * len(pair_ids)
        + 3
        + 4
        + len(population.factual_miss)
    )
    checks = {
        "zero_feature_occupancy_delta_exact_zero": (
            zero_feature_max == 0.0
        ),
        "raw_phase_evidence_occupancy_invariant_exact": (
            raw_occupancy_delta_max == 0.0
        ),
        "common_mode_occupancy_invariant_exact": (
            common_occupancy_delta_max == 0.0
        ),
        "baseline_occupancy_invariant_exact": (
            baseline_occupancy_delta_max == 0.0
        ),
        "phase_allocation_occupancy_invariant_exact": (
            allocation_occupancy_delta_max == 0.0
        ),
        "common_mode_equals_phase_mean": common_mode_error_max == 0.0,
        "phase_contrast_sum_zero": (
            phase_contrast_sum_error_max <= 1.0e-5
        ),
        "native_occupancy_burden_formula_exact": (
            burden_formula_error_max == 0.0
        ),
        "budget_equals_continuously_recoverable_margin": (
            budget_forward_mismatch_states == 0
        ),
        "phase_allocation_nonnegative_simplex": (
            minimum_allocation >= 0.0
            and simplex_error_max <= 1.0e-6
        ),
        "allocated_phase_evidence_nonnegative": (
            minimum_budget >= 0.0
            and minimum_allocated_evidence >= 0.0
        ),
        "phase_evidence_sum_equals_budget": (
            mass_conservation_relative_error_max <= 1.0e-6
        ),
        "phase_mass_delta_equals_budget_delta": (
            phase_mass_delta_error_max <= 1.0e-6
        ),
        "logit_composition_exact": (
            logit_composition_mismatch_states == 0
        ),
        "count_and_burden_change_support_exact": (
            count_burden_support_mismatch_pixels == 0
        ),
        "outside_count_support_logit_delta_exact_zero": (
            outside_logit_max == 0.0
        ),
        "outside_count_support_probability_delta_exact_zero": (
            outside_probability_max == 0.0
        ),
        "count_change_support_nonempty": changed_support_pixels > 0,
        "count_unchanged_support_nonempty": (
            unchanged_support_pixels > 0
            or int(locality_probe["unchanged_support_pixels"]) > 0
        ),
        "phase_aware_nonvacuous_locality_probe_passed": (
            locality_probe["all_pass"] is True
        ),
        "feature_detached_inside_decoder": (
            locality_probe["feature_gradient_is_none"] is True
        ),
        "dual_endpoint_gradients_finite_nonzero": (
            dual_endpoint["all_pass"] is True
        ),
        "all_audited_fields_finite": nonfinite_field_values == 0,
        "local_count_deletion_monotonicity_exact": (
            count_monotonicity_violations == 0
        ),
        "occupancy_burden_deletion_monotonicity_exact": (
            burden_monotonicity_violations == 0
        ),
        "evidence_budget_deletion_monotonicity_exact": (
            budget_monotonicity_violations == 0
        ),
        "allocated_evidence_deletion_monotonicity_exact": (
            allocated_monotonicity_violations == 0
        ),
        "deletion_logit_monotonicity_exact": (
            logit_monotonicity_violations == 0
        ),
        "deletion_probability_monotonicity_exact": (
            probability_monotonicity_violations == 0
        ),
        "native_subpixel_path_without_resize": resize_count == 0,
        "all_clean_D_pixels_in_phase_aware_count_support": (
            clean_full_D_reachable == clean_pair_count
            and clean_D_reachable_pixels == clean_D_total_pixels
        ),
        "all_clean_pairs_have_nonempty_H": (
            clean_nonempty_H_pairs == clean_pair_count
        ),
        "all_component_null_pairs_have_positive_count_support": (
            component_positive_support_pairs == component_pair_count
        ),
        "all_factual_targets_have_finite_nonzero_budget_recovery": (
            int(factual_full.sum().cpu()) == len(population.factual_miss)
            and factual_total == factual_reachable
        ),
        "structural_audit_decoder_budget_exact": (
            audit_decoder_calls == expected_audit_decoder_calls
            and audit_decoder_state_evaluations
            == expected_audit_decoder_state_evaluations
        ),
    }
    return {
        "scope": (
            "pretraining_D_R_full_population_CC_SEA_v8_state_equation"
        ),
        "pair_count": len(records),
        "clean_pair_count": clean_pair_count,
        "component_null_pair_count": component_pair_count,
        "zero_feature_max_abs_occupancy_delta": zero_feature_max,
        "raw_phase_evidence_max_abs_occupancy_delta": (
            raw_occupancy_delta_max
        ),
        "common_mode_max_abs_occupancy_delta": (
            common_occupancy_delta_max
        ),
        "baseline_max_abs_occupancy_delta": (
            baseline_occupancy_delta_max
        ),
        "phase_allocation_max_abs_occupancy_delta": (
            allocation_occupancy_delta_max
        ),
        "common_mode_max_abs_error": common_mode_error_max,
        "phase_contrast_sum_max_abs_error": (
            phase_contrast_sum_error_max
        ),
        "occupancy_burden_formula_max_abs_error": (
            burden_formula_error_max
        ),
        "phase_allocation_sum_max_abs_error": simplex_error_max,
        "minimum_phase_allocation": minimum_allocation,
        "minimum_evidence_budget": minimum_budget,
        "minimum_allocated_phase_evidence": (
            minimum_allocated_evidence
        ),
        "mass_conservation_max_abs_error": (
            mass_conservation_error_max
        ),
        "mass_conservation_max_relative_error": (
            mass_conservation_relative_error_max
        ),
        "phase_mass_delta_budget_delta_max_relative_error": (
            phase_mass_delta_error_max
        ),
        "budget_forward_mismatch_endpoint_batches": (
            budget_forward_mismatch_states
        ),
        "logit_composition_mismatch_endpoint_batches": (
            logit_composition_mismatch_states
        ),
        "count_burden_support_mismatch_pixels": (
            count_burden_support_mismatch_pixels
        ),
        "outside_count_support_max_abs_logit_delta": (
            outside_logit_max
        ),
        "outside_count_support_max_abs_probability_delta": (
            outside_probability_max
        ),
        "nonfinite_audited_field_values": nonfinite_field_values,
        "local_count_deletion_monotonicity_violations": (
            count_monotonicity_violations
        ),
        "occupancy_burden_deletion_monotonicity_violations": (
            burden_monotonicity_violations
        ),
        "evidence_budget_deletion_monotonicity_violations": (
            budget_monotonicity_violations
        ),
        "allocated_evidence_deletion_monotonicity_violations": (
            allocated_monotonicity_violations
        ),
        "deletion_logit_monotonicity_violations": (
            logit_monotonicity_violations
        ),
        "deletion_probability_monotonicity_violations": (
            probability_monotonicity_violations
        ),
        "field_resize_endpoint_count": resize_count,
        "changed_phase_aware_count_support_pixels": (
            changed_support_pixels
        ),
        "unchanged_phase_aware_count_support_pixels": (
            unchanged_support_pixels
        ),
        "outside_count_support_check_vacuous": (
            unchanged_support_pixels == 0
        ),
        "phase_aware_locality_probe": locality_probe,
        "dual_endpoint_gradient_audit": dual_endpoint,
        "clean_full_D_reachable_pairs": clean_full_D_reachable,
        "clean_nonempty_H_pairs": clean_nonempty_H_pairs,
        "clean_D_reachable_pixels": clean_D_reachable_pixels,
        "clean_D_total_pixels": clean_D_total_pixels,
        "component_positive_count_support_pairs": (
            component_positive_support_pairs
        ),
        "factual_full_target_reachable_anchors": int(
            factual_full.sum().cpu()
        ),
        "factual_target_reachable_pixels": factual_reachable,
        "factual_target_total_pixels": factual_total,
        "factual_budget_margin": {
            "minimum": float(
                factual_fields.budget_margin.min().cpu()
            ),
            "maximum": float(
                factual_fields.budget_margin.max().cpu()
            ),
        },
        "clean_count_support_fraction": {
            "minimum": min(clean_support_fractions),
            "mean": (
                sum(clean_support_fractions)
                / len(clean_support_fractions)
            ),
            "maximum": max(clean_support_fractions),
        },
        "component_count_support_fraction": {
            "minimum": min(component_support_fractions),
            "mean": (
                sum(component_support_fractions)
                / len(component_support_fractions)
            ),
            "maximum": max(component_support_fractions),
        },
        "compute_budget": {
            "decoder_calls": audit_decoder_calls,
            "decoder_state_evaluations": (
                audit_decoder_state_evaluations
            ),
            "expected_decoder_calls": expected_audit_decoder_calls,
            "expected_decoder_state_evaluations": (
                expected_audit_decoder_state_evaluations
            ),
            "population_chunk_decoder_calls": (
                4 * ceil(len(pair_ids) / chunk_size)
            ),
            "population_chunk_decoder_state_evaluations": (
                4 * len(pair_ids)
            ),
            "phase_aware_locality_decoder_calls": 3,
            "phase_aware_locality_decoder_state_evaluations": 3,
            "dual_endpoint_decoder_calls": 1,
            "dual_endpoint_decoder_state_evaluations": 4,
            "factual_forward_fields_calls": 1,
            "factual_forward_fields_states": len(
                population.factual_miss
            ),
        },
        "checks": checks,
        "all_pass": all(checks.values()),
        "per_pair": records,
        "training_performed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }


def _compose_pretraining_structural_audit(
    population_audit: Mapping[str, object],
    operator_audit: Mapping[str, object],
) -> dict[str, object]:
    population_checks = population_audit.get("checks")
    operator_checks = operator_audit.get("checks")
    if not isinstance(population_checks, Mapping):
        raise RuntimeError("malformed CC-SEA population structural audit")
    if not isinstance(operator_checks, Mapping):
        raise RuntimeError("malformed CC-SEA operator structural audit")
    if tuple(operator_checks) != CONSERVATIVE_OPERATOR_STRUCTURAL_CHECKS:
        raise RuntimeError("CC-SEA operator checks are incomplete")
    overlap = set(population_checks) & set(operator_checks)
    if overlap:
        raise RuntimeError(
            "CC-SEA structural check names overlap: "
            f"{sorted(overlap)}"
        )
    checks = {
        **{
            str(key): bool(value)
            for key, value in population_checks.items()
        },
        **{
            str(key): bool(value)
            for key, value in operator_checks.items()
        },
    }
    combined = dict(population_audit)
    combined.update(
        {
            "scope": (
                "pretraining_D_R_full_population_CC_SEA_v8_"
                "state_equation_plus_operator_contract"
            ),
            "population_audit_scope": population_audit.get("scope"),
            "operator_contract": dict(operator_audit),
            "checks": checks,
            "all_pass": (
                population_audit.get("all_pass") is True
                and operator_audit.get("all_pass") is True
                and all(checks.values())
            ),
        }
    )
    return combined


def _validate_conservative_execution_inputs(
    population: OutcomeBoundedAnchorPopulation,
    factual_schedule: OutcomeFactualAnchorSchedule,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    decoder_config: ConservativeFactorizedDecoderConfig,
    loss_config: LossConfig,
    optimization_budget: Mapping[str, object],
    device: torch.device | str,
    evaluation_chunk_size: int,
) -> tuple[torch.device, int, float, float]:
    if not isinstance(
        decoder_config,
        ConservativeFactorizedDecoderConfig,
    ):
        raise TypeError(
            "decoder_config must be "
            "ConservativeFactorizedDecoderConfig"
        )
    if not isinstance(loss_config, LossConfig):
        raise TypeError("loss_config must be LossConfig")
    if (
        loss_config.dice_weight != 1.0
        or loss_config.epsilon != 1.0e-6
    ):
        raise ValueError(
            "CC-SEA v8 fixes "
            "LossConfig(dice_weight=1.0, epsilon=1e-6)"
        )
    if not isinstance(optimization_budget, Mapping):
        raise TypeError("optimization_budget must be a mapping")
    if (
        optimization_budget.get("seed") != FACTORIZED_FROZEN_SEED
        or isinstance(optimization_budget.get("seed"), bool)
    ):
        raise ValueError("CC-SEA v8 fixes optimization seed at 42")
    if (
        optimization_budget.get("learning_rate")
        != FACTORIZED_FROZEN_LEARNING_RATE
        or isinstance(
            optimization_budget.get("learning_rate"),
            bool,
        )
    ):
        raise ValueError("CC-SEA v8 fixes learning_rate at 0.001")
    if (
        optimization_budget.get("weight_decay")
        != FACTORIZED_FROZEN_WEIGHT_DECAY
        or isinstance(
            optimization_budget.get("weight_decay"),
            bool,
        )
    ):
        raise ValueError("CC-SEA v8 fixes weight_decay at 0.0")
    if (
        isinstance(evaluation_chunk_size, bool)
        or evaluation_chunk_size
        != FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE
    ):
        raise ValueError("CC-SEA v8 fixes evaluation_chunk_size at 32")
    validated = _validate_execution_inputs(
        population,
        factual_schedule,
        schedule,
        materializer,
        DecoderConfig(
            feature_channels=decoder_config.feature_channels
        ),
        loss_config,
        optimization_budget,
        device,
        evaluation_chunk_size,
    )
    if decoder_config.feature_channels != materializer.feature_shape[1]:
        raise ValueError(
            "CC-SEA decoder channels differ from outcome inputs"
        )
    feature_height, feature_width = (
        int(value) for value in materializer.feature_shape[-2:]
    )
    evaluation_height, evaluation_width = (
        int(value) for value in materializer.evaluation_shape[-2:]
    )
    expected = (
        feature_height * decoder_config.feature_stride,
        feature_width * decoder_config.feature_stride,
    )
    if expected != (evaluation_height, evaluation_width):
        raise ValueError(
            "bounded CC-SEA requires a native subpixel path without field "
            f"resize ({expected} != "
            f"{(evaluation_height, evaluation_width)})"
        )
    return validated


def conservative_computational_gates(
    initial: Mapping[str, object],
    final: Mapping[str, object],
) -> dict[str, object]:
    """Reuse the exact twelve frozen gates under the v8-specific scope."""

    frozen = factorized_computational_gates(initial, final)
    expected_thresholds = {
        **dict(COMPUTATIONAL_THRESHOLDS),
        "clean_joint_D_ge_0_25_and_H_le_0_05_fraction_min": (
            FACTORIZED_JOINT_THRESHOLD
        ),
    }
    if (
        frozen.get("thresholds") != expected_thresholds
        or not isinstance(frozen.get("checks"), Mapping)
        or len(frozen["checks"]) != 12
    ):
        raise RuntimeError("the twelve frozen bounded gates changed")
    output = dict(frozen)
    output["scope"] = (
        "bounded_D_R_full_outcome_CC_SEA_v8_model_code_gate"
    )
    output["thresholds_unchanged_from_v4_v6_v7"] = True
    return output


def _structural_failure_result(
    decoder: CURELiteConservativeFactorizedDecoder,
    population: OutcomeBoundedAnchorPopulation,
    factual_schedule: OutcomeFactualAnchorSchedule,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    decoder_config: ConservativeFactorizedDecoderConfig,
    loss_config: LossConfig,
    optimization_budget: Mapping[str, object],
    *,
    target_device: torch.device,
    evaluation_chunk_size: int,
    structural_audit: Mapping[str, object],
    state_observation: Mapping[str, object],
) -> dict[str, object]:
    checks = structural_audit.get("checks")
    compute_budget = structural_audit.get("compute_budget")
    if (
        not isinstance(checks, Mapping)
        or structural_audit.get("all_pass") is not False
        or not isinstance(compute_budget, Mapping)
    ):
        raise RuntimeError("malformed failed CC-SEA structural audit")
    result: dict[str, object] = {
        "schema_version": (
            CONSERVATIVE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA
        ),
        "method_id": CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID,
        "execution_status": "completed",
        "decision": "CC_SEA_V8_STRUCTURAL_EXECUTION_FAIL",
        "device": str(target_device),
        "population_fingerprint": population.population_fingerprint,
        "outcome_schedule_fingerprint": (
            schedule.schedule_fingerprint
        ),
        "factual_schedule_fingerprint": (
            factual_schedule.schedule_fingerprint
        ),
        "materializer_fingerprint": (
            materializer.materializer_fingerprint
        ),
        "decoder_config": asdict(decoder_config),
        "loss_config": asdict(loss_config),
        "optimization_budget": dict(optimization_budget),
        "evaluation_chunk_size": evaluation_chunk_size,
        "optimizer_updates_completed": 0,
        "pretraining_structural_audit": dict(structural_audit),
        "state_equation_observation": dict(state_observation),
        "computational_gates": {
            "status": "NOT_EVALUATED_BY_STRUCTURAL_STOP_RULE",
            "all_pass": None,
        },
        "structural_checks": dict(checks),
        "structural_execution_pass": False,
        "computational_model_code_gate_pass": False,
        "training_performed": False,
        "parameters": {
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in decoder.parameters()
            ),
            "trainable_parameter_tensors": len(
                tuple(decoder.parameters())
            ),
            "expected_parameter_count": (
                decoder_config.expected_parameter_count
            ),
            "initial_decoder_fingerprint": (
                conservative_factorized_decoder_state_fingerprint(
                    decoder
                )
            ),
        },
        "forward_budget": {
            "pretraining_structural_audit": dict(compute_budget),
            "training": {"calls": 0, "state_evaluations": 0},
        },
        "trace": [],
        "interpretation": {
            "evidence_scope": (
                "fresh_CC_SEA_v8_decoder_pretraining_D_R_"
                "structural_audit"
            ),
            "not_detection_performance_evidence": True,
            "does_not_establish_Pd_or_FA": True,
            "does_not_authorize_formal_training": True,
            "does_not_directly_authorize_formal_800": True,
            "eligible_for_frozen_review": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "base_or_backbone_updated": False,
            "identity_null_optimizer_exposure": 0,
        },
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


def execute_conservative_factorized_outcome_bounded(
    population: OutcomeBoundedAnchorPopulation,
    factual_schedule: OutcomeFactualAnchorSchedule,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    decoder_config: ConservativeFactorizedDecoderConfig,
    loss_config: LossConfig,
    optimization_budget: Mapping[str, object],
    *,
    device: torch.device | str,
    evaluation_chunk_size: int = 32,
) -> dict[str, object]:
    """Train and audit one fresh CC-SEA decoder under 400 frozen updates."""

    (
        target_device,
        seed,
        learning_rate,
        weight_decay,
    ) = _validate_conservative_execution_inputs(
        population,
        factual_schedule,
        schedule,
        materializer,
        decoder_config,
        loss_config,
        optimization_budget,
        device,
        evaluation_chunk_size,
    )
    evaluation_chunk_size = int(evaluation_chunk_size)
    cuda_devices: list[int] = []
    if target_device.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device()
            if target_device.index is None
            else target_device.index
        ]

    with _deterministic_torch_runtime(
        target_device,
        DETERMINISM_SPECIFICATION,
    ) as deterministic_runtime, torch.random.fork_rng(
        devices=cuda_devices
    ):
        torch.manual_seed(seed)
        if target_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        decoder = CURELiteConservativeFactorizedDecoder(
            decoder_config
        ).to(target_device)
        state_observer = _ConservativeStateObserver(decoder)
        population_structural_audit = (
            audit_conservative_outcome_population(
                decoder,
                population,
                schedule,
                materializer,
                device=target_device,
                chunk_size=evaluation_chunk_size,
            )
        )
        operator_structural_audit = (
            _audit_conservative_operator_contract(
                device=target_device,
                decoder=decoder,
                initialization_seed=seed,
            )
        )
        structural_audit = _compose_pretraining_structural_audit(
            population_structural_audit,
            operator_structural_audit,
        )
        if structural_audit["all_pass"] is not True:
            materializer.verify_unchanged()
            state_observation = state_observer.close()
            return _structural_failure_result(
                decoder,
                population,
                factual_schedule,
                schedule,
                materializer,
                decoder_config,
                loss_config,
                optimization_budget,
                target_device=target_device,
                evaluation_chunk_size=evaluation_chunk_size,
                structural_audit=structural_audit,
                state_observation=state_observation,
            )

        absolute_criterion = CURELiteLoss(loss_config).to(target_device)
        outcome_criterion = OutcomeCompleteTransitionLoss(
            loss_config
        ).to(target_device)
        optimizer = torch.optim.Adam(
            decoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        named_parameters = tuple(decoder.named_parameters())
        parameter_count = sum(
            parameter.numel() for _, parameter in named_parameters
        )
        parameter_tensor_count = len(named_parameters)
        initial_decoder_fingerprint = (
            conservative_factorized_decoder_state_fingerprint(decoder)
        )
        initial_parameter_norm = sqrt(
            sum(
                float(parameter.detach().double().square().sum().cpu())
                for _, parameter in named_parameters
            )
        )

        ledger = _ForwardLedger(decoder)
        input_ledger = _InputDetachLedger(decoder)
        pair_exposure: Counter[str] = Counter()
        source_exposure: Counter[str] = Counter()
        miss_exposure: Counter[str] = Counter()
        no_miss_exposure: Counter[str] = Counter()
        trace: list[dict[str, object]] = []
        minimum_gradient_norm = float("inf")
        maximum_gradient_norm = 0.0
        nonfinite_gradient_updates = 0
        zero_gradient_updates = 0
        missing_parameter_gradient_updates = 0
        nonfinite_parameter_gradient_updates = 0
        zero_parameter_gradient_tensor_updates = 0
        optimizer_steps = 0
        backward_calls = 0
        try:
            before_initial = ledger.snapshot()
            before_initial_inputs = input_ledger.snapshot()
            initial = _evaluate_snapshot(
                decoder,
                population,
                schedule,
                materializer,
                absolute_criterion,
                outcome_criterion,
                device=target_device,
                chunk_size=evaluation_chunk_size,
            )
            after_initial = ledger.snapshot()
            after_initial_inputs = input_ledger.snapshot()
            initial_forward = {
                "calls": after_initial[0] - before_initial[0],
                "state_evaluations": (
                    after_initial[1] - before_initial[1]
                ),
            }
            initial_input_detach = {
                "calls": (
                    after_initial_inputs[0]
                    - before_initial_inputs[0]
                ),
                "state_evaluations": (
                    after_initial_inputs[1]
                    - before_initial_inputs[1]
                ),
                "requires_grad_violations": (
                    after_initial_inputs[2]
                    - before_initial_inputs[2]
                ),
            }

            training_start = ledger.snapshot()
            training_input_start = input_ledger.snapshot()
            for update in range(schedule.optimizer_updates):
                pair_ids = schedule.pair_ids_for_update(update)
                miss_indices = (
                    factual_schedule.factual_miss_indices[update]
                )
                no_miss_indices = (
                    factual_schedule.factual_no_miss_indices[update]
                )
                before_update = ledger.snapshot()
                before_update_inputs = input_ledger.snapshot()
                logs = outcome_complete_train_step(
                    decoder,
                    absolute_criterion,
                    outcome_criterion,
                    optimizer,
                    _factual_batches(
                        population,
                        factual_schedule,
                        update,
                        device=target_device,
                    ),
                    materializer.materialize(
                        pair_ids,
                        device=target_device,
                    ),
                )
                gradients = tuple(
                    parameter.grad
                    for _, parameter in named_parameters
                )
                missing_count = sum(
                    gradient is None for gradient in gradients
                )
                nonfinite_tensor_count = sum(
                    gradient is not None
                    and not bool(
                        torch.isfinite(gradient).all().detach().cpu()
                    )
                    for gradient in gradients
                )
                zero_tensor_count = sum(
                    gradient is not None
                    and int(
                        torch.count_nonzero(gradient).detach().cpu()
                    )
                    == 0
                    for gradient in gradients
                )
                squared_gradient_norm = sum(
                    float(
                        gradient.detach().double().square().sum().cpu()
                    )
                    for gradient in gradients
                    if gradient is not None
                )
                gradient_norm = sqrt(squared_gradient_norm)
                if not isfinite(gradient_norm):
                    nonfinite_gradient_updates += 1
                if gradient_norm <= 0.0:
                    zero_gradient_updates += 1
                missing_parameter_gradient_updates += int(
                    missing_count > 0
                )
                nonfinite_parameter_gradient_updates += int(
                    nonfinite_tensor_count > 0
                )
                zero_parameter_gradient_tensor_updates += (
                    zero_tensor_count
                )
                minimum_gradient_norm = min(
                    minimum_gradient_norm,
                    gradient_norm,
                )
                maximum_gradient_norm = max(
                    maximum_gradient_norm,
                    gradient_norm,
                )
                optimizer_steps += int(logs["optimizer_steps"])
                backward_calls += int(logs["backward_calls"])
                pair_exposure.update(pair_ids)
                source_exposure.update(
                    materializer.pair_by_id[pair_id].sample_id
                    for pair_id in pair_ids
                )
                miss_exposure.update(
                    population.factual_miss_ids[index]
                    for index in miss_indices
                )
                no_miss_exposure.update(
                    population.factual_no_miss_ids[index]
                    for index in no_miss_indices
                )
                after_update = ledger.snapshot()
                after_update_inputs = input_ledger.snapshot()
                trace.append(
                    {
                        "update": update,
                        "epoch": (
                            update // schedule.steps_per_epoch
                        ),
                        "step": update % schedule.steps_per_epoch,
                        "outcome_pair_ids": list(pair_ids),
                        "outcome_pair_kinds": [
                            materializer.pair_by_id[
                                pair_id
                            ].pair_kind
                            for pair_id in pair_ids
                        ],
                        "factual_miss_ids": [
                            population.factual_miss_ids[index]
                            for index in miss_indices
                        ],
                        "factual_no_miss_ids": [
                            population.factual_no_miss_ids[index]
                            for index in no_miss_indices
                        ],
                        "losses": logs,
                        "gradient_l2_norm": gradient_norm,
                        "parameter_gradient_tensors": (
                            parameter_tensor_count
                        ),
                        "missing_parameter_gradients": (
                            missing_count
                        ),
                        "nonfinite_parameter_gradients": (
                            nonfinite_tensor_count
                        ),
                        "zero_parameter_gradient_tensors": (
                            zero_tensor_count
                        ),
                        "decoder_forward_calls": (
                            after_update[0] - before_update[0]
                        ),
                        "decoder_state_evaluations": (
                            after_update[1] - before_update[1]
                        ),
                        "decoder_input_requires_grad_violations": (
                            after_update_inputs[2]
                            - before_update_inputs[2]
                        ),
                    }
                )
            training_end = ledger.snapshot()
            training_input_end = input_ledger.snapshot()
            training_forward = {
                "calls": training_end[0] - training_start[0],
                "state_evaluations": (
                    training_end[1] - training_start[1]
                ),
            }
            training_input_detach = {
                "calls": (
                    training_input_end[0]
                    - training_input_start[0]
                ),
                "state_evaluations": (
                    training_input_end[1]
                    - training_input_start[1]
                ),
                "requires_grad_violations": (
                    training_input_end[2]
                    - training_input_start[2]
                ),
            }

            materializer.verify_unchanged()
            before_final = ledger.snapshot()
            before_final_inputs = input_ledger.snapshot()
            final = _evaluate_snapshot(
                decoder,
                population,
                schedule,
                materializer,
                absolute_criterion,
                outcome_criterion,
                device=target_device,
                chunk_size=evaluation_chunk_size,
            )
            after_final = ledger.snapshot()
            after_final_inputs = input_ledger.snapshot()
            final_forward = {
                "calls": after_final[0] - before_final[0],
                "state_evaluations": (
                    after_final[1] - before_final[1]
                ),
            }
            final_input_detach = {
                "calls": (
                    after_final_inputs[0]
                    - before_final_inputs[0]
                ),
                "state_evaluations": (
                    after_final_inputs[1]
                    - before_final_inputs[1]
                ),
                "requires_grad_violations": (
                    after_final_inputs[2]
                    - before_final_inputs[2]
                ),
            }
            total_forward = {
                "calls": after_final[0],
                "state_evaluations": after_final[1],
            }
            total_input_detach = {
                "calls": after_final_inputs[0],
                "state_evaluations": after_final_inputs[1],
                "requires_grad_violations": after_final_inputs[2],
            }
        finally:
            input_ledger.close()
            ledger.close()

        state_observation = state_observer.close()
        final_decoder_fingerprint = (
            conservative_factorized_decoder_state_fingerprint(decoder)
        )
        final_parameter_norm = sqrt(
            sum(
                float(parameter.detach().double().square().sum().cpu())
                for _, parameter in named_parameters
            )
        )

    scheduled_ids = tuple(pair.pair_id for pair in schedule.pairs)
    actual_pair_counts = tuple(
        pair_exposure[pair_id] for pair_id in scheduled_ids
    )
    actual_miss_counts = tuple(
        miss_exposure[anchor_id]
        for anchor_id in population.factual_miss_ids
    )
    actual_no_miss_counts = tuple(
        no_miss_exposure[anchor_id]
        for anchor_id in population.factual_no_miss_ids
    )
    actual_source_counts = tuple(sorted(source_exposure.items()))
    expected_snapshot_forward = {
        "calls": ceil(len(scheduled_ids) / evaluation_chunk_size) + 3,
        "state_evaluations": (
            2 * len(scheduled_ids)
            + 2 * len(population.factual_miss)
            + 2 * len(population.factual_no_miss)
        ),
    }
    expected_training_forward = {
        "calls": 3 * schedule.optimizer_updates,
        "state_evaluations": 12 * schedule.optimizer_updates,
    }
    expected_total_forward = {
        "calls": (
            expected_training_forward["calls"]
            + 2 * expected_snapshot_forward["calls"]
        ),
        "state_evaluations": (
            expected_training_forward["state_evaluations"]
            + 2 * expected_snapshot_forward["state_evaluations"]
        ),
    }
    expected_state_observation_calls = (
        int(structural_audit["compute_budget"]["decoder_calls"])
        + int(expected_total_forward["calls"])
    )
    exposure = {
        "outcome_pairs": [
            {
                "pair_id": pair.pair_id,
                "pair_kind": pair.pair_kind,
                "sample_id": pair.sample_id,
                "count": pair_exposure[pair.pair_id],
            }
            for pair in schedule.pairs
        ],
        "source_images": [
            {"sample_id": sample_id, "count": count}
            for sample_id, count in actual_source_counts
        ],
        "factual_miss": [
            {
                "anchor_id": anchor_id,
                "sample_id": example.sample_id,
                "count": miss_exposure[anchor_id],
            }
            for anchor_id, example in zip(
                population.factual_miss_ids,
                population.factual_miss,
                strict=True,
            )
        ],
        "factual_no_miss": [
            {
                "anchor_id": anchor_id,
                "sample_id": example.sample_id,
                "count": no_miss_exposure[anchor_id],
            }
            for anchor_id, example in zip(
                population.factual_no_miss_ids,
                population.factual_no_miss,
                strict=True,
            )
        ],
        "outcome_pair_exposure_values": sorted(
            set(actual_pair_counts)
        ),
        "identity_null_optimizer_exposure": 0,
    }
    structural_checks = {
        "deterministic_runtime_contract_satisfied": (
            deterministic_runtime["contract_satisfied"] is True
            and deterministic_runtime[
                "flags_restored_after_execution"
            ]
            is True
        ),
        "CC_SEA_v8_pretraining_structural_audit_passed": (
            structural_audit["all_pass"] is True
        ),
        **{
            check_name: (
                structural_audit["operator_contract"]["checks"][
                    check_name
                ]
                is True
            )
            for check_name in CONSERVATIVE_OPERATOR_STRUCTURAL_CHECKS
        },
        "decoder_parameter_count_and_tensor_count_exact": (
            parameter_count == decoder_config.expected_parameter_count
            and parameter_tensor_count == 6
        ),
        "factual_anchor_and_identity_counts_exact": (
            len(population.factual_miss) == 16
            and len(population.factual_no_miss) == 16
            and len(population.identity_null) == 16
        ),
        "all_222_outcome_pairs_bound": (
            len(scheduled_ids) == 222
            and set(scheduled_ids)
            == set(materializer.canonical_pair_ids)
        ),
        "all_222_outcome_pairs_evaluated_initial": (
            initial["outcome_population"]["pair_ids"]
            == list(scheduled_ids)
        ),
        "all_222_outcome_pairs_evaluated_final": (
            final["outcome_population"]["pair_ids"]
            == list(scheduled_ids)
        ),
        "all_optimizer_updates_completed": (
            len(trace) == schedule.optimizer_updates == 400
        ),
        "one_backward_per_update": backward_calls == 400,
        "one_optimizer_step_per_update": optimizer_steps == 400,
        "all_parameter_gradients_present_each_update": (
            missing_parameter_gradient_updates == 0
        ),
        "all_parameter_gradients_finite_each_update": (
            nonfinite_parameter_gradient_updates == 0
            and nonfinite_gradient_updates == 0
        ),
        "every_update_total_gradient_norm_positive": (
            zero_gradient_updates == 0
        ),
        "decoder_parameters_changed": (
            final_decoder_fingerprint != initial_decoder_fingerprint
        ),
        "training_forward_budget_exact_three_calls_twelve_states": (
            training_forward == expected_training_forward
            and all(
                row["decoder_forward_calls"] == 3
                and row["decoder_state_evaluations"] == 12
                and row["losses"][
                    "decoder_forward_calls_per_update"
                ]
                == 3
                and row["losses"]["decoder_states_per_update"] == 12
                for row in trace
            )
        ),
        "training_inputs_detached": (
            training_input_detach
            == {
                **expected_training_forward,
                "requires_grad_violations": 0,
            }
            and all(
                row[
                    "decoder_input_requires_grad_violations"
                ]
                == 0
                for row in trace
            )
        ),
        "evaluation_forward_budget_exact": (
            initial_forward == expected_snapshot_forward
            and final_forward == expected_snapshot_forward
        ),
        "evaluation_inputs_detached": (
            initial_input_detach
            == {
                **expected_snapshot_forward,
                "requires_grad_violations": 0,
            }
            and final_input_detach
            == {
                **expected_snapshot_forward,
                "requires_grad_violations": 0,
            }
        ),
        "total_forward_budget_exact": (
            total_forward == expected_total_forward
            and total_input_detach
            == {
                **expected_total_forward,
                "requires_grad_violations": 0,
            }
        ),
        "state_equation_observation_budget_exact": (
            state_observation["observed_forward_fields_calls"]
            == expected_state_observation_calls
            and state_observation[
                "additional_decoder_forward_calls"
            ]
            == 0
        ),
        "all_observed_CC_SEA_fields_finite_nonnegative": (
            state_observation["all_observed_fields_finite"] is True
            and state_observation[
                "all_observed_allocations_nonnegative"
            ]
            is True
            and state_observation[
                "all_observed_evidence_nonnegative"
            ]
            is True
        ),
        "all_observed_phase_simplex_and_mass_conservation": (
            state_observation[
                "maximum_observed_simplex_error"
            ]
            <= 1.0e-6
            and state_observation[
                "maximum_observed_mass_conservation_relative_error"
            ]
            <= 1.0e-6
        ),
        "pair_exposure_ledger_exact": (
            actual_pair_counts == schedule.pair_exposure_counts
            and set(actual_pair_counts) == {3, 4}
        ),
        "source_exposure_ledger_exact": (
            actual_source_counts == schedule.source_exposure_counts
        ),
        "factual_exposure_ledgers_exact": (
            actual_miss_counts == factual_schedule.factual_miss_counts
            and actual_no_miss_counts
            == factual_schedule.factual_no_miss_counts
        ),
        "identity_null_excluded_from_optimizer": (
            exposure["identity_null_optimizer_exposure"] == 0
        ),
        "identity_null_diagnosed_without_autograd": all(
            snapshot["identity_null"]["autograd_enabled"] is False
            for snapshot in (initial, final)
        ),
    }
    structural_execution_pass = all(structural_checks.values())
    computational = conservative_computational_gates(initial, final)
    computational_pass = (
        structural_execution_pass and computational["all_pass"] is True
    )
    decision = (
        "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_PASS"
        if computational_pass
        else (
            "CC_SEA_V8_STRUCTURAL_EXECUTION_FAIL"
            if not structural_execution_pass
            else "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_FAIL"
        )
    )
    result: dict[str, object] = {
        "schema_version": (
            CONSERVATIVE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA
        ),
        "method_id": CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID,
        "execution_status": "completed",
        "decision": decision,
        "device": str(target_device),
        "population_fingerprint": population.population_fingerprint,
        "outcome_schedule_fingerprint": (
            schedule.schedule_fingerprint
        ),
        "factual_schedule_fingerprint": (
            factual_schedule.schedule_fingerprint
        ),
        "materializer_fingerprint": (
            materializer.materializer_fingerprint
        ),
        "decoder_config": asdict(decoder_config),
        "loss_config": asdict(loss_config),
        "optimization_budget": dict(optimization_budget),
        "evaluation_chunk_size": evaluation_chunk_size,
        "optimizer_updates_completed": len(trace),
        "training_performed": True,
        "pretraining_structural_audit": structural_audit,
        "state_equation_observation": {
            **dict(state_observation),
            "expected_forward_fields_calls": (
                expected_state_observation_calls
            ),
        },
        "initial": initial,
        "final": final,
        "computational_gates": computational,
        "structural_checks": structural_checks,
        "structural_execution_pass": structural_execution_pass,
        "computational_model_code_gate_pass": computational_pass,
        "parameters": {
            "trainable_parameter_count": parameter_count,
            "trainable_parameter_tensors": parameter_tensor_count,
            "expected_parameter_count": (
                decoder_config.expected_parameter_count
            ),
            "initial_decoder_fingerprint": (
                initial_decoder_fingerprint
            ),
            "final_decoder_fingerprint": (
                final_decoder_fingerprint
            ),
            "initial_l2_norm": initial_parameter_norm,
            "final_l2_norm": final_parameter_norm,
        },
        "gradients": {
            "minimum_update_l2_norm": minimum_gradient_norm,
            "maximum_update_l2_norm": maximum_gradient_norm,
            "nonfinite_updates": nonfinite_gradient_updates,
            "zero_norm_updates": zero_gradient_updates,
            "missing_parameter_gradient_updates": (
                missing_parameter_gradient_updates
            ),
            "nonfinite_parameter_gradient_updates": (
                nonfinite_parameter_gradient_updates
            ),
            "zero_parameter_gradient_tensor_updates": (
                zero_parameter_gradient_tensor_updates
            ),
        },
        "execution_ledger": {
            "backward_calls": backward_calls,
            "optimizer_steps": optimizer_steps,
            "expected_backward_calls": 400,
            "expected_optimizer_steps": 400,
        },
        "forward_budget": {
            "pretraining_structural_audit_is_separate": True,
            "pretraining_structural_audit": (
                structural_audit["compute_budget"]
            ),
            "initial_evaluation": initial_forward,
            "training": training_forward,
            "final_evaluation": final_forward,
            "total_excluding_structural_audit": total_forward,
            "expected_initial_evaluation": (
                expected_snapshot_forward
            ),
            "expected_training": expected_training_forward,
            "expected_final_evaluation": expected_snapshot_forward,
            "expected_total_excluding_structural_audit": (
                expected_total_forward
            ),
            "input_detach": {
                "initial_evaluation": initial_input_detach,
                "training": training_input_detach,
                "final_evaluation": final_input_detach,
                "total_excluding_structural_audit": (
                    total_input_detach
                ),
            },
        },
        "deterministic_runtime": deterministic_runtime,
        "exposure": exposure,
        "trace": trace,
        "interpretation": {
            "evidence_scope": (
                "fresh_CC_SEA_v8_decoder_bounded_D_R_"
                "full_outcome_population"
            ),
            "not_detection_performance_evidence": True,
            "does_not_establish_Pd_or_FA": True,
            "does_not_authorize_formal_training": True,
            "does_not_directly_authorize_formal_800": True,
            "eligible_for_frozen_review": computational_pass,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "base_or_backbone_updated": False,
            "identity_null_optimizer_exposure": 0,
        },
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


__all__ = [
    "CONSERVATIVE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA",
    "CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID",
    "CONSERVATIVE_OPERATOR_STRUCTURAL_CHECKS",
    "audit_conservative_outcome_population",
    "conservative_computational_gates",
    "conservative_factorized_decoder_state_fingerprint",
    "execute_conservative_factorized_outcome_bounded",
]
