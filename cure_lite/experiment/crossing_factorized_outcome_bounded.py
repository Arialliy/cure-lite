"""D_R-only bounded model-code gate for CURE-Lite CR-LVEC v7.

The outcome population, factual anchors, objective, optimizer step,
deterministic schedules, update budget, and computational thresholds are the
frozen SVEF-v4/PR-SVEF-v6 objects.  This additive executor instantiates one
fresh CR-LVEC decoder and replaces only the mechanism-specific pre-training
audit.  It does not load a dataset itself and never reads D_V or D_T.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from math import ceil, isfinite, sqrt
from typing import Mapping

import torch
from torch.nn import functional as F

from ..cache.schema import stable_fingerprint
from ..config import DecoderConfig, LossConfig
from ..crossing_factorized_config import (
    CrossingFactorizedDecoderConfig,
)
from ..crossing_factorized_decoder import (
    CURELiteCrossingFactorizedDecoder,
    crossing_recoverable_evidence,
)
from ..losses import CURELiteLoss
from ..paired_outcome_losses import OutcomeCompleteTransitionLoss
from ..train.paired_outcome_step import outcome_complete_train_step
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


CROSSING_FACTORIZED_OUTCOME_BOUNDED_SCHEMA = (
    "cure-lite-cr-lvec-v7-bounded-execution-v1"
)
CROSSING_FACTORIZED_OUTCOME_METHOD_ID = "cr_lvec_v7"
CROSSING_OPERATOR_STRUCTURAL_CHECKS = (
    "v7_nonpositive_margin_forward_exact_zero",
    "v7_positive_margin_forward_equals_expm1",
    "v7_supported_axis_gradient_equals_exp",
    "v7_zero_boundary_gradient_equals_one",
    "v7_negative_80_recovery_finite_nonzero",
    "v7_negative_104_zero_recovery_fails_fast",
    "v7_positive_88_forward_gradient_finite",
    "v7_positive_89_nonfinite_fails_fast",
    "v7_exponential_ratio_identity_matches",
)


def _update_digest(digest: object, value: bytes) -> None:
    if not hasattr(digest, "update"):
        raise TypeError("digest must provide update()")
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def crossing_factorized_decoder_state_fingerprint(
    decoder: CURELiteCrossingFactorizedDecoder,
) -> str:
    """Hash the v7 class identity, module topology, and exact tensor state."""

    if not isinstance(decoder, CURELiteCrossingFactorizedDecoder):
        raise TypeError(
            "decoder must be CURELiteCrossingFactorizedDecoder"
        )
    digest = hashlib.sha256()
    digest.update(b"cure-lite-cr-lvec-v7-decoder-state-v1")
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


def _maximum_abs(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    detached = value.detach()
    finite = detached[torch.isfinite(detached)]
    if finite.numel() == 0:
        return 0.0
    return float(finite.abs().max().cpu())


class _CrossingMarginObserver:
    """Collect existing forward margins without another decoder computation."""

    def __init__(
        self,
        decoder: CURELiteCrossingFactorizedDecoder,
    ) -> None:
        if not isinstance(decoder, CURELiteCrossingFactorizedDecoder):
            raise TypeError(
                "decoder must be CURELiteCrossingFactorizedDecoder"
            )
        self._decoder = decoder
        self._original = decoder.forward_fields
        self._maxima: list[torch.Tensor] = []
        self._closed = False

        def observed_forward_fields(
            feature: torch.Tensor,
            occupancy: torch.Tensor,
        ) -> object:
            fields = self._original(feature, occupancy)
            self._maxima.append(
                fields.crossing_margin.detach().abs().amax()
            )
            return fields

        object.__setattr__(
            decoder,
            "forward_fields",
            observed_forward_fields,
        )

    def close(self) -> dict[str, object]:
        if self._closed:
            raise RuntimeError("crossing margin observer is already closed")
        self._closed = True
        object.__delattr__(self._decoder, "forward_fields")
        if not self._maxima:
            return {
                "maximum_observed_absolute_margin": None,
                "observed_forward_fields_calls": 0,
                "all_observed_margins_finite": True,
                "scope": (
                    "no_decoder_forward_observed_before_structural_stop"
                ),
                "additional_decoder_forward_calls": 0,
            }
        maxima = torch.stack(self._maxima)
        return {
            "maximum_observed_absolute_margin": float(
                maxima.max().cpu()
            ),
            "observed_forward_fields_calls": len(self._maxima),
            "all_observed_margins_finite": bool(
                torch.isfinite(maxima).all().cpu()
            ),
            "scope": (
                "all_existing_decoder_forward_fields_calls_without_"
                "additional_decoder_computation"
            ),
            "additional_decoder_forward_calls": 0,
        }


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


def _audit_crossing_operator_contract(
    *,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate the frozen v7 forward, backward, and numeric contract."""

    margin = torch.tensor(
        [-4.0, -1.0, -1.0e-6, 0.0, 1.0e-4, 0.5, 4.0],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    evidence = crossing_recoverable_evidence(margin)
    evidence.sum().backward()
    gradient = margin.grad
    if gradient is None:
        raise RuntimeError("CR-LVEC operator audit produced no gradient")

    expected_forward = torch.where(
        margin.detach() <= 0.0,
        torch.zeros_like(margin.detach()),
        torch.expm1(margin.detach()),
    )
    expected_gradient = torch.exp(margin.detach())

    supported = torch.tensor(
        [-80.0, 88.0],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    supported_evidence = crossing_recoverable_evidence(supported)
    supported_evidence.sum().backward()
    supported_gradient = supported.grad
    if supported_gradient is None:
        raise RuntimeError(
            "CR-LVEC numerical audit produced no supported gradient"
        )

    negative_104_rejected = _probe_rejected(-104.0, device=device)
    positive_89_rejected = _probe_rejected(89.0, device=device)

    raw = torch.tensor(
        [0.5, 1.5, 4.0],
        dtype=torch.float64,
        device=device,
    )
    count = torch.tensor(
        [0.0, 1.0, 3.0],
        dtype=torch.float64,
        device=device,
    )
    ratio_left = torch.expm1(raw - torch.log1p(count))
    ratio_right = torch.exp(raw) / (1.0 + count) - 1.0

    checks = {
        "v7_nonpositive_margin_forward_exact_zero": torch.equal(
            evidence[:4],
            torch.zeros_like(evidence[:4]),
        ),
        "v7_positive_margin_forward_equals_expm1": torch.equal(
            evidence[4:],
            expected_forward[4:],
        ),
        "v7_supported_axis_gradient_equals_exp": torch.equal(
            gradient,
            expected_gradient,
        ),
        "v7_zero_boundary_gradient_equals_one": (
            float(gradient[3].detach().cpu()) == 1.0
        ),
        "v7_negative_80_recovery_finite_nonzero": (
            float(supported_evidence[0].detach().cpu()) == 0.0
            and bool(torch.isfinite(supported_gradient[0]).cpu())
            and float(supported_gradient[0].detach().cpu()) > 0.0
        ),
        "v7_negative_104_zero_recovery_fails_fast": (
            negative_104_rejected
        ),
        "v7_positive_88_forward_gradient_finite": (
            bool(torch.isfinite(supported_evidence[1]).cpu())
            and bool(torch.isfinite(supported_gradient[1]).cpu())
            and float(supported_evidence[1].detach().cpu()) > 0.0
            and float(supported_gradient[1].detach().cpu()) > 0.0
        ),
        "v7_positive_89_nonfinite_fails_fast": positive_89_rejected,
        "v7_exponential_ratio_identity_matches": torch.allclose(
            ratio_left,
            ratio_right,
            rtol=1.0e-12,
            atol=1.0e-15,
        ),
    }
    if tuple(checks) != CROSSING_OPERATOR_STRUCTURAL_CHECKS:
        raise AssertionError("v7 operator structural check set drifted")
    return {
        "scope": "CR_LVEC_v7_frozen_operator_and_numeric_contract",
        "probe_margin": [
            float(value) for value in margin.detach().cpu()
        ],
        "observed_forward": [
            float(value) for value in evidence.detach().cpu()
        ],
        "expected_forward": [
            float(value) for value in expected_forward.cpu()
        ],
        "observed_gradient": [
            float(value) for value in gradient.detach().cpu()
        ],
        "expected_gradient": [
            float(value) for value in expected_gradient.cpu()
        ],
        "numeric_probes": {
            "negative_finite_nonzero_recovery": -80.0,
            "negative_zero_recovery_fail_fast": -104.0,
            "positive_largest_finite": 88.0,
            "positive_first_nonfinite_fail_fast": 89.0,
            "supported_forward": [
                float(value)
                for value in supported_evidence.detach().cpu()
            ],
            "supported_gradient": [
                float(value)
                for value in supported_gradient.detach().cpu()
            ],
            "negative_104_rejected": negative_104_rejected,
            "positive_89_rejected": positive_89_rejected,
        },
        "ratio_probe": {
            "raw_evidence": [float(value) for value in raw.cpu()],
            "local_count": [float(value) for value in count.cpu()],
            "left": [float(value) for value in ratio_left.cpu()],
            "right": [float(value) for value in ratio_right.cpu()],
        },
        "checks": checks,
        "all_pass": all(checks.values()),
        "autograd_backward_calls": 2,
        "training_performed": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }


def _audit_nonvacuous_locality(
    decoder: CURELiteCrossingFactorizedDecoder,
    *,
    device: torch.device,
) -> dict[str, object]:
    """Exercise both changed and unchanged count support on a larger grid."""

    parameter = next(decoder.parameters())
    feature_size = (4, 5)
    output_size = tuple(
        value * decoder.config.feature_stride for value in feature_size
    )
    feature = torch.zeros(
        1,
        decoder.config.feature_channels,
        *feature_size,
        dtype=parameter.dtype,
        device=device,
    )
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
        decoder.config.feature_stride,
        2 * decoder.config.feature_stride,
    ] = True
    occupancy_minus = torch.zeros_like(occupancy_plus)
    plus = decoder.forward_fields(feature, occupancy_plus)
    minus = decoder.forward_fields(feature, occupancy_minus)
    count_release = (
        plus.local_occupancy_count - minus.local_occupancy_count
    )
    lifted = F.interpolate(
        count_release,
        size=output_size,
        mode="nearest",
    )
    changed = lifted > 0.0
    unchanged = ~changed
    logit_delta = minus.logits - plus.logits
    probability_delta = (
        torch.sigmoid(minus.logits) - torch.sigmoid(plus.logits)
    )
    changed_count = int(changed.sum().cpu())
    unchanged_count = int(unchanged.sum().cpu())
    burden_support = (
        plus.occupancy_burden - minus.occupancy_burden
    ) > 0.0
    checks = {
        "changed_support_nonempty": changed_count > 0,
        "unchanged_support_nonempty": unchanged_count > 0,
        "count_and_burden_support_exact": torch.equal(
            changed,
            burden_support,
        ),
        "unchanged_logits_bit_exact": torch.equal(
            logit_delta[unchanged],
            torch.zeros_like(logit_delta[unchanged]),
        ),
        "unchanged_probability_bit_exact": torch.equal(
            probability_delta[unchanged],
            torch.zeros_like(probability_delta[unchanged]),
        ),
        "deletion_logit_monotone": bool(
            torch.all(logit_delta >= 0.0).cpu()
        ),
        "deletion_probability_monotone": bool(
            torch.all(probability_delta >= 0.0).cpu()
        ),
        "all_fields_finite": all(
            bool(torch.isfinite(value).all().cpu())
            for value in (
                plus.logits,
                minus.logits,
                count_release,
                logit_delta,
                probability_delta,
            )
        ),
    }
    return {
        "feature_grid": list(feature_size),
        "output_grid": list(output_size),
        "changed_support_pixels": changed_count,
        "unchanged_support_pixels": unchanged_count,
        "checks": checks,
        "all_pass": all(checks.values()),
        "decoder_calls": 2,
        "decoder_state_evaluations": 2,
    }


def audit_crossing_outcome_population(
    decoder: CURELiteCrossingFactorizedDecoder,
    population: OutcomeBoundedAnchorPopulation,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    *,
    device: torch.device,
    chunk_size: int,
) -> dict[str, object]:
    """Audit v7 count, burden, crossing, and locality on all outcome pairs."""

    if not isinstance(decoder, CURELiteCrossingFactorizedDecoder):
        raise TypeError(
            "decoder must be CURELiteCrossingFactorizedDecoder"
        )
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    pair_ids = tuple(pair.pair_id for pair in schedule.pairs)
    records: list[dict[str, object]] = []
    zero_feature_max = 0.0
    raw_evidence_occupancy_delta_max = 0.0
    baseline_occupancy_delta_max = 0.0
    outside_count_logit_max = 0.0
    outside_count_probability_max = 0.0
    count_support_mismatch_pixels = 0
    count_monotonicity_violations = 0
    burden_monotonicity_violations = 0
    logit_monotonicity_violations = 0
    probability_monotonicity_violations = 0
    nonfinite_field_values = 0
    resize_count = 0
    clean_full_D_reachable = 0
    clean_D_reachable_pixels = 0
    clean_D_total_pixels = 0
    clean_nonempty_H_pairs = 0
    component_positive_support_pairs = 0
    clean_support_fractions: list[float] = []
    component_support_fractions: list[float] = []
    clean_D_burden_release: list[float] = []
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
            raw_evidence_occupancy_delta_max = max(
                raw_evidence_occupancy_delta_max,
                _maximum_abs(minus.raw_evidence - plus.raw_evidence),
            )
            baseline_occupancy_delta_max = max(
                baseline_occupancy_delta_max,
                _maximum_abs(
                    minus.baseline_logits - plus.baseline_logits
                ),
            )

            count_release = (
                plus.local_occupancy_count
                - minus.local_occupancy_count
            )
            lifted_count_release = F.interpolate(
                count_release,
                size=tuple(int(value) for value in plus.logits.shape[-2:]),
                mode="nearest",
            )
            burden_release = (
                plus.occupancy_burden - minus.occupancy_burden
            )
            count_support = lifted_count_release > 0.0
            burden_support = burden_release > 0.0
            count_support_mismatch_pixels += int(
                (count_support != burden_support).sum().cpu()
            )
            logit_delta = minus.logits - plus.logits
            probability_delta = (
                torch.sigmoid(minus.logits)
                - torch.sigmoid(plus.logits)
            )
            outside = ~count_support
            outside_count_logit_max = max(
                outside_count_logit_max,
                _maximum_abs(logit_delta[outside]),
            )
            outside_count_probability_max = max(
                outside_count_probability_max,
                _maximum_abs(probability_delta[outside]),
            )
            count_monotonicity_violations += int(
                (count_release < 0.0).sum().cpu()
            )
            burden_monotonicity_violations += int(
                (burden_release < 0.0).sum().cpu()
            )
            logit_monotonicity_violations += int(
                (logit_delta < 0.0).sum().cpu()
            )
            probability_monotonicity_violations += int(
                (probability_delta < 0.0).sum().cpu()
            )
            nonfinite_field_values += sum(
                int((~torch.isfinite(value)).sum().cpu())
                for value in (
                    plus.baseline_logits,
                    plus.raw_evidence,
                    plus.occupancy_burden,
                    plus.crossing_margin,
                    plus.evidence,
                    plus.logits,
                    plus.local_occupancy_count,
                    minus.baseline_logits,
                    minus.raw_evidence,
                    minus.occupancy_burden,
                    minus.crossing_margin,
                    minus.evidence,
                    minus.logits,
                    minus.local_occupancy_count,
                    count_release,
                    burden_release,
                    logit_delta,
                    probability_delta,
                )
            )
            resize_count += int(plus.field_resize_applied)
            resize_count += int(minus.field_resize_applied)

            valid = batch.pair_batch.image_valid_mask
            changed_support_pixels += int(
                (count_support & valid).sum().cpu()
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
                reachable = response & count_support[index]
                reachable_count = int(reachable.sum().cpu())
                valid_count = int(valid[index].sum().cpu())
                support_count = int(
                    (count_support[index] & valid[index]).sum().cpu()
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
                    clean_D_burden_release.extend(
                        float(value)
                        for value in burden_release[index][response].cpu()
                    )
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
                        "count_change_support_pixels": support_count,
                        "count_change_support_fraction": support_fraction,
                        "field_resize_applied": (
                            plus.field_resize_applied
                            or minus.field_resize_applied
                        ),
                    }
                )

        locality_probe = _audit_nonvacuous_locality(
            decoder,
            device=device,
        )
        audit_decoder_calls += int(locality_probe["decoder_calls"])
        audit_decoder_state_evaluations += int(
            locality_probe["decoder_state_evaluations"]
        )

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
        factual_recovery = torch.exp(factual_fields.crossing_margin)
        factual_target = factual.target > 0.5
        factual_recoverable = (
            torch.isfinite(factual_recovery)
            & (factual_recovery > 0.0)
        )
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
            "CR-LVEC structural audit did not cover all pairs"
        )
    if (
        clean_pair_count < 1
        or component_pair_count < 1
        or clean_D_total_pixels < 1
        or not clean_D_burden_release
    ):
        raise RuntimeError(
            "CR-LVEC structural audit found an empty required stratum"
        )

    factual_total = int(factual_total_by_anchor.sum().cpu())
    factual_reachable = int(factual_reachable_by_anchor.sum().cpu())
    expected_audit_decoder_calls = (
        4 * ceil(len(pair_ids) / chunk_size) + 3
    )
    expected_audit_decoder_state_evaluations = (
        4 * len(pair_ids) + len(population.factual_miss) + 2
    )
    checks = {
        "zero_feature_occupancy_delta_exact_zero": (
            zero_feature_max == 0.0
        ),
        "raw_evidence_occupancy_invariant_exact": (
            raw_evidence_occupancy_delta_max == 0.0
        ),
        "baseline_occupancy_invariant_exact": (
            baseline_occupancy_delta_max == 0.0
        ),
        "count_and_burden_change_support_exact": (
            count_support_mismatch_pixels == 0
        ),
        "count_support_outside_logit_delta_exact_zero": (
            outside_count_logit_max == 0.0
        ),
        "count_support_outside_probability_delta_exact_zero": (
            outside_count_probability_max == 0.0
        ),
        "count_change_support_nonempty": changed_support_pixels > 0,
        "independent_nonvacuous_locality_probe_passed": (
            locality_probe["all_pass"] is True
        ),
        "all_audited_fields_finite": nonfinite_field_values == 0,
        "local_count_deletion_monotonicity_exact": (
            count_monotonicity_violations == 0
        ),
        "occupancy_burden_deletion_monotonicity_exact": (
            burden_monotonicity_violations == 0
        ),
        "deletion_logit_monotonicity_exact": (
            logit_monotonicity_violations == 0
        ),
        "deletion_probability_monotonicity_exact": (
            probability_monotonicity_violations == 0
        ),
        "native_subpixel_path_without_resize": resize_count == 0,
        "all_clean_D_pixels_in_count_change_support": (
            clean_full_D_reachable == clean_pair_count
            and clean_D_reachable_pixels == clean_D_total_pixels
        ),
        "all_clean_pairs_have_nonempty_H": (
            clean_nonempty_H_pairs == clean_pair_count
        ),
        "all_component_null_pairs_have_positive_count_support": (
            component_positive_support_pairs == component_pair_count
        ),
        "all_factual_targets_have_finite_nonzero_recovery": (
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
        "scope": "pretraining_D_R_full_population_CR_LVEC_v7_structure",
        "pair_count": len(records),
        "clean_pair_count": clean_pair_count,
        "component_null_pair_count": component_pair_count,
        "zero_feature_max_abs_occupancy_delta": zero_feature_max,
        "raw_evidence_max_abs_occupancy_delta": (
            raw_evidence_occupancy_delta_max
        ),
        "baseline_max_abs_occupancy_delta": baseline_occupancy_delta_max,
        "count_burden_support_mismatch_pixels": (
            count_support_mismatch_pixels
        ),
        "outside_count_support_max_abs_logit_delta": (
            outside_count_logit_max
        ),
        "outside_count_support_max_abs_probability_delta": (
            outside_count_probability_max
        ),
        "nonfinite_audited_field_values": nonfinite_field_values,
        "local_count_deletion_monotonicity_violations": (
            count_monotonicity_violations
        ),
        "occupancy_burden_deletion_monotonicity_violations": (
            burden_monotonicity_violations
        ),
        "deletion_logit_monotonicity_violations": (
            logit_monotonicity_violations
        ),
        "deletion_probability_monotonicity_violations": (
            probability_monotonicity_violations
        ),
        "field_resize_endpoint_count": resize_count,
        "changed_count_support_pixels": changed_support_pixels,
        "unchanged_count_support_pixels": unchanged_support_pixels,
        "outside_count_support_check_vacuous": (
            unchanged_support_pixels == 0
        ),
        "independent_nonvacuous_locality_probe": locality_probe,
        "clean_full_D_reachable_pairs": clean_full_D_reachable,
        "clean_nonempty_H_pairs": clean_nonempty_H_pairs,
        "clean_D_reachable_pixels": clean_D_reachable_pixels,
        "clean_D_total_pixels": clean_D_total_pixels,
        "component_positive_count_support_pairs": (
            component_positive_support_pairs
        ),
        "factual_full_target_recoverable_anchors": int(
            factual_full.sum().cpu()
        ),
        "factual_target_recoverable_pixels": factual_reachable,
        "factual_target_total_pixels": factual_total,
        "factual_margin": {
            "minimum": float(
                factual_fields.crossing_margin.min().cpu()
            ),
            "maximum": float(
                factual_fields.crossing_margin.max().cpu()
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
        "clean_D_burden_release": {
            "minimum": min(clean_D_burden_release),
            "mean": (
                sum(clean_D_burden_release)
                / len(clean_D_burden_release)
            ),
            "maximum": max(clean_D_burden_release),
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
            "factual_forward_fields_calls": 1,
            "factual_forward_fields_states": len(
                population.factual_miss
            ),
            "independent_locality_decoder_calls": 2,
            "independent_locality_decoder_state_evaluations": 2,
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
        raise RuntimeError("malformed CR-LVEC population structural audit")
    if not isinstance(operator_checks, Mapping):
        raise RuntimeError("malformed CR-LVEC operator structural audit")
    if tuple(operator_checks) != CROSSING_OPERATOR_STRUCTURAL_CHECKS:
        raise RuntimeError("CR-LVEC operator checks are incomplete")
    overlap = set(population_checks) & set(operator_checks)
    if overlap:
        raise RuntimeError(
            "CR-LVEC structural check names overlap: "
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
                "pretraining_D_R_full_population_CR_LVEC_v7_structure_"
                "plus_operator_numeric_contract"
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


def _validate_crossing_execution_inputs(
    population: OutcomeBoundedAnchorPopulation,
    factual_schedule: OutcomeFactualAnchorSchedule,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    decoder_config: CrossingFactorizedDecoderConfig,
    loss_config: LossConfig,
    optimization_budget: Mapping[str, object],
    device: torch.device | str,
    evaluation_chunk_size: int,
) -> tuple[torch.device, int, float, float]:
    if not isinstance(decoder_config, CrossingFactorizedDecoderConfig):
        raise TypeError(
            "decoder_config must be CrossingFactorizedDecoderConfig"
        )
    if not isinstance(loss_config, LossConfig):
        raise TypeError("loss_config must be LossConfig")
    if (
        loss_config.dice_weight != 1.0
        or loss_config.epsilon != 1.0e-6
    ):
        raise ValueError(
            "CR-LVEC v7 fixes "
            "LossConfig(dice_weight=1.0, epsilon=1e-6)"
        )
    if not isinstance(optimization_budget, Mapping):
        raise TypeError("optimization_budget must be a mapping")
    if (
        optimization_budget.get("seed") != FACTORIZED_FROZEN_SEED
        or isinstance(optimization_budget.get("seed"), bool)
    ):
        raise ValueError("CR-LVEC v7 fixes optimization seed at 42")
    if (
        optimization_budget.get("learning_rate")
        != FACTORIZED_FROZEN_LEARNING_RATE
        or isinstance(optimization_budget.get("learning_rate"), bool)
    ):
        raise ValueError("CR-LVEC v7 fixes learning_rate at 0.001")
    if (
        optimization_budget.get("weight_decay")
        != FACTORIZED_FROZEN_WEIGHT_DECAY
        or isinstance(optimization_budget.get("weight_decay"), bool)
    ):
        raise ValueError("CR-LVEC v7 fixes weight_decay at 0.0")
    if (
        isinstance(evaluation_chunk_size, bool)
        or evaluation_chunk_size
        != FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE
    ):
        raise ValueError("CR-LVEC v7 fixes evaluation_chunk_size at 32")
    validated = _validate_execution_inputs(
        population,
        factual_schedule,
        schedule,
        materializer,
        DecoderConfig(feature_channels=decoder_config.feature_channels),
        loss_config,
        optimization_budget,
        device,
        evaluation_chunk_size,
    )
    if decoder_config.feature_channels != materializer.feature_shape[1]:
        raise ValueError(
            "crossing decoder channels differ from outcome inputs"
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
            "bounded CR-LVEC requires a native subpixel path without field "
            f"resize ({expected} != "
            f"{(evaluation_height, evaluation_width)})"
        )
    return validated


def crossing_computational_gates(
    initial: Mapping[str, object],
    final: Mapping[str, object],
) -> dict[str, object]:
    """Reuse the exact twelve frozen gates under a v7-specific scope."""

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
        "bounded_D_R_full_outcome_CR_LVEC_v7_model_code_gate"
    )
    output["thresholds_unchanged_from_v4_v6"] = True
    return output


def _structural_failure_result(
    decoder: CURELiteCrossingFactorizedDecoder,
    population: OutcomeBoundedAnchorPopulation,
    factual_schedule: OutcomeFactualAnchorSchedule,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    decoder_config: CrossingFactorizedDecoderConfig,
    loss_config: LossConfig,
    optimization_budget: Mapping[str, object],
    *,
    target_device: torch.device,
    evaluation_chunk_size: int,
    structural_audit: Mapping[str, object],
    margin_observation: Mapping[str, object],
) -> dict[str, object]:
    checks = structural_audit.get("checks")
    compute_budget = structural_audit.get("compute_budget")
    if (
        not isinstance(checks, Mapping)
        or structural_audit.get("all_pass") is not False
        or not isinstance(compute_budget, Mapping)
    ):
        raise RuntimeError("malformed failed CR-LVEC structural audit")
    result: dict[str, object] = {
        "schema_version": CROSSING_FACTORIZED_OUTCOME_BOUNDED_SCHEMA,
        "method_id": CROSSING_FACTORIZED_OUTCOME_METHOD_ID,
        "execution_status": "completed",
        "decision": "CR_LVEC_STRUCTURAL_EXECUTION_FAIL",
        "device": str(target_device),
        "population_fingerprint": population.population_fingerprint,
        "outcome_schedule_fingerprint": schedule.schedule_fingerprint,
        "factual_schedule_fingerprint": factual_schedule.schedule_fingerprint,
        "materializer_fingerprint": materializer.materializer_fingerprint,
        "decoder_config": asdict(decoder_config),
        "loss_config": asdict(loss_config),
        "optimization_budget": dict(optimization_budget),
        "evaluation_chunk_size": evaluation_chunk_size,
        "optimizer_updates_completed": 0,
        "pretraining_structural_audit": dict(structural_audit),
        "margin_observation": dict(margin_observation),
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
            "expected_parameter_count": (
                decoder_config.expected_parameter_count
            ),
            "initial_decoder_fingerprint": (
                crossing_factorized_decoder_state_fingerprint(decoder)
            ),
        },
        "forward_budget": {
            "pretraining_structural_audit": dict(compute_budget),
            "training": {"calls": 0, "state_evaluations": 0},
        },
        "trace": [],
        "interpretation": {
            "evidence_scope": (
                "fresh_CR_LVEC_v7_decoder_pretraining_D_R_"
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


def execute_crossing_factorized_outcome_bounded(
    population: OutcomeBoundedAnchorPopulation,
    factual_schedule: OutcomeFactualAnchorSchedule,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    decoder_config: CrossingFactorizedDecoderConfig,
    loss_config: LossConfig,
    optimization_budget: Mapping[str, object],
    *,
    device: torch.device | str,
    evaluation_chunk_size: int = 32,
) -> dict[str, object]:
    """Train and audit one fresh CR-LVEC decoder under 400 frozen updates."""

    (
        target_device,
        seed,
        learning_rate,
        weight_decay,
    ) = _validate_crossing_execution_inputs(
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
    ) as deterministic_runtime, torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if target_device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        decoder = CURELiteCrossingFactorizedDecoder(
            decoder_config
        ).to(target_device)
        margin_observer = _CrossingMarginObserver(decoder)
        population_structural_audit = audit_crossing_outcome_population(
            decoder,
            population,
            schedule,
            materializer,
            device=target_device,
            chunk_size=evaluation_chunk_size,
        )
        operator_structural_audit = _audit_crossing_operator_contract(
            device=target_device,
        )
        structural_audit = _compose_pretraining_structural_audit(
            population_structural_audit,
            operator_structural_audit,
        )
        if structural_audit["all_pass"] is not True:
            materializer.verify_unchanged()
            margin_observation = margin_observer.close()
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
                margin_observation=margin_observation,
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
        parameter_count = sum(
            parameter.numel() for parameter in decoder.parameters()
        )
        initial_decoder_fingerprint = (
            crossing_factorized_decoder_state_fingerprint(decoder)
        )
        initial_parameter_norm = sqrt(
            sum(
                float(parameter.detach().double().square().sum().cpu())
                for parameter in decoder.parameters()
            )
        )

        ledger = _ForwardLedger(decoder)
        pair_exposure: Counter[str] = Counter()
        source_exposure: Counter[str] = Counter()
        miss_exposure: Counter[str] = Counter()
        no_miss_exposure: Counter[str] = Counter()
        trace: list[dict[str, object]] = []
        minimum_gradient_norm = float("inf")
        maximum_gradient_norm = 0.0
        nonfinite_gradient_updates = 0
        zero_gradient_updates = 0
        optimizer_steps = 0
        backward_calls = 0
        try:
            before_initial = ledger.snapshot()
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
            initial_forward = {
                "calls": after_initial[0] - before_initial[0],
                "state_evaluations": after_initial[1] - before_initial[1],
            }

            training_start = ledger.snapshot()
            for update in range(schedule.optimizer_updates):
                pair_ids = schedule.pair_ids_for_update(update)
                miss_indices = factual_schedule.factual_miss_indices[update]
                no_miss_indices = (
                    factual_schedule.factual_no_miss_indices[update]
                )
                before_update = ledger.snapshot()
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
                squared_gradient_norm = sum(
                    float(
                        parameter.grad.detach().double().square().sum().cpu()
                    )
                    for parameter in decoder.parameters()
                    if parameter.grad is not None
                )
                gradient_norm = sqrt(squared_gradient_norm)
                if not isfinite(gradient_norm):
                    nonfinite_gradient_updates += 1
                if gradient_norm <= 0.0:
                    zero_gradient_updates += 1
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
                trace.append(
                    {
                        "update": update,
                        "epoch": update // schedule.steps_per_epoch,
                        "step": update % schedule.steps_per_epoch,
                        "outcome_pair_ids": list(pair_ids),
                        "outcome_pair_kinds": [
                            materializer.pair_by_id[pair_id].pair_kind
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
                        "decoder_forward_calls": (
                            after_update[0] - before_update[0]
                        ),
                        "decoder_state_evaluations": (
                            after_update[1] - before_update[1]
                        ),
                    }
                )
            training_end = ledger.snapshot()
            training_forward = {
                "calls": training_end[0] - training_start[0],
                "state_evaluations": training_end[1] - training_start[1],
            }

            materializer.verify_unchanged()
            before_final = ledger.snapshot()
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
            final_forward = {
                "calls": after_final[0] - before_final[0],
                "state_evaluations": after_final[1] - before_final[1],
            }
            total_forward = {
                "calls": after_final[0],
                "state_evaluations": after_final[1],
            }
        finally:
            ledger.close()

        margin_observation = margin_observer.close()
        final_decoder_fingerprint = (
            crossing_factorized_decoder_state_fingerprint(decoder)
        )
        final_parameter_norm = sqrt(
            sum(
                float(parameter.detach().double().square().sum().cpu())
                for parameter in decoder.parameters()
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
    expected_margin_observation_calls = (
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
        "CR_LVEC_pretraining_structural_audit_passed": (
            structural_audit["all_pass"] is True
        ),
        **{
            check_name: (
                structural_audit["operator_contract"]["checks"][
                    check_name
                ]
                is True
            )
            for check_name in CROSSING_OPERATOR_STRUCTURAL_CHECKS
        },
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
        "all_gradients_finite": nonfinite_gradient_updates == 0,
        "every_update_total_gradient_norm_positive": (
            zero_gradient_updates == 0
        ),
        "decoder_parameters_changed": (
            final_decoder_fingerprint != initial_decoder_fingerprint
        ),
        "training_forward_budget_exact": (
            training_forward == expected_training_forward
            and all(
                row["decoder_forward_calls"] == 3
                and row["decoder_state_evaluations"] == 12
                for row in trace
            )
        ),
        "evaluation_forward_budget_exact": (
            initial_forward == expected_snapshot_forward
            and final_forward == expected_snapshot_forward
        ),
        "total_forward_budget_exact": (
            total_forward == expected_total_forward
        ),
        "margin_observation_budget_exact": (
            margin_observation["observed_forward_fields_calls"]
            == expected_margin_observation_calls
            and margin_observation["additional_decoder_forward_calls"] == 0
        ),
        "all_observed_crossing_margins_finite": (
            margin_observation["all_observed_margins_finite"] is True
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
    computational = crossing_computational_gates(initial, final)
    computational_pass = (
        structural_execution_pass and computational["all_pass"] is True
    )
    decision = (
        "CR_LVEC_BOUNDED_MODEL_CODE_GATE_PASS"
        if computational_pass
        else (
            "CR_LVEC_STRUCTURAL_EXECUTION_FAIL"
            if not structural_execution_pass
            else "CR_LVEC_BOUNDED_MODEL_CODE_GATE_FAIL"
        )
    )
    result: dict[str, object] = {
        "schema_version": CROSSING_FACTORIZED_OUTCOME_BOUNDED_SCHEMA,
        "method_id": CROSSING_FACTORIZED_OUTCOME_METHOD_ID,
        "execution_status": "completed",
        "decision": decision,
        "device": str(target_device),
        "population_fingerprint": population.population_fingerprint,
        "outcome_schedule_fingerprint": schedule.schedule_fingerprint,
        "factual_schedule_fingerprint": (
            factual_schedule.schedule_fingerprint
        ),
        "materializer_fingerprint": materializer.materializer_fingerprint,
        "decoder_config": asdict(decoder_config),
        "loss_config": asdict(loss_config),
        "optimization_budget": dict(optimization_budget),
        "evaluation_chunk_size": evaluation_chunk_size,
        "optimizer_updates_completed": len(trace),
        "pretraining_structural_audit": structural_audit,
        "margin_observation": {
            **dict(margin_observation),
            "expected_forward_fields_calls": (
                expected_margin_observation_calls
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
            "expected_parameter_count": (
                decoder_config.expected_parameter_count
            ),
            "initial_decoder_fingerprint": initial_decoder_fingerprint,
            "final_decoder_fingerprint": final_decoder_fingerprint,
            "initial_l2_norm": initial_parameter_norm,
            "final_l2_norm": final_parameter_norm,
        },
        "gradients": {
            "minimum_update_l2_norm": minimum_gradient_norm,
            "maximum_update_l2_norm": maximum_gradient_norm,
            "nonfinite_updates": nonfinite_gradient_updates,
            "zero_norm_updates": zero_gradient_updates,
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
            "expected_initial_evaluation": expected_snapshot_forward,
            "expected_training": expected_training_forward,
            "expected_final_evaluation": expected_snapshot_forward,
            "expected_total_excluding_structural_audit": (
                expected_total_forward
            ),
        },
        "deterministic_runtime": deterministic_runtime,
        "exposure": exposure,
        "trace": trace,
        "interpretation": {
            "evidence_scope": (
                "fresh_CR_LVEC_v7_decoder_bounded_D_R_"
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
    "CROSSING_FACTORIZED_OUTCOME_BOUNDED_SCHEMA",
    "CROSSING_FACTORIZED_OUTCOME_METHOD_ID",
    "CROSSING_OPERATOR_STRUCTURAL_CHECKS",
    "audit_crossing_outcome_population",
    "crossing_computational_gates",
    "crossing_factorized_decoder_state_fingerprint",
    "execute_crossing_factorized_outcome_bounded",
]
