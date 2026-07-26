"""D_R-only bounded model-code gate for CURE-Lite SVEF v4.

The population, objective, optimizer step, and deterministic schedules are
the frozen OC-APTO v3 objects.  This additive runner changes only the decoder
factory and adds SVEF-specific structural reachability and per-pair joint
gates.  It is not a detector benchmark and never reads D_V or D_T.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
from math import ceil, isfinite, sqrt
from typing import Mapping

import torch

from ..cache.schema import stable_fingerprint
from ..config import DecoderConfig, LossConfig
from ..factorized_config import FactorizedDecoderConfig
from ..factorized_decoder import CURELiteFactorizedDecoder
from ..losses import CURELiteLoss
from ..paired_outcome_losses import OutcomeCompleteTransitionLoss
from ..train.paired_outcome_step import outcome_complete_train_step
from ..train.pools import stack_state_examples
from .paired_bounded_learnability import _deterministic_torch_runtime
from .paired_outcome_bounded import (
    COMPUTATIONAL_THRESHOLDS,
    DETERMINISM_SPECIFICATION,
    OutcomeBoundedAnchorPopulation,
    OutcomeFactualAnchorSchedule,
    _ForwardLedger,
    _computational_gates,
    _evaluate_snapshot,
    _factual_batches,
    _validate_execution_inputs,
)
from .paired_outcome_inputs import PairedOutcomeInputMaterializer
from .paired_outcome_schedule import OutcomePairSchedule


FACTORIZED_OUTCOME_BOUNDED_SCHEMA = (
    "cure-lite-svef-v4-bounded-execution-v1"
)
FACTORIZED_JOINT_THRESHOLD = 0.75
FACTORIZED_JOINT_D_THRESHOLD = 0.25
FACTORIZED_JOINT_H_THRESHOLD = 0.05
FACTORIZED_FROZEN_SEED = 42
FACTORIZED_FROZEN_LEARNING_RATE = 1.0e-3
FACTORIZED_FROZEN_WEIGHT_DECAY = 0.0
FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE = 32
TINY_TARGET_STRATA = (
    ("1_to_3", 1, 3),
    ("4_to_7", 4, 7),
    ("8_to_15", 8, 15),
    ("16_plus", 16, None),
)


def _update_digest(digest: object, value: bytes) -> None:
    if not hasattr(digest, "update"):
        raise TypeError("digest must provide update()")
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def factorized_decoder_state_fingerprint(
    decoder: CURELiteFactorizedDecoder,
) -> str:
    """Hash topology and exact tensor state, including scalar parameters."""

    if not isinstance(decoder, CURELiteFactorizedDecoder):
        raise TypeError("decoder must be CURELiteFactorizedDecoder")
    digest = hashlib.sha256()
    digest.update(b"cure-lite-svef-v4-decoder-state-v1")
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


def _validate_factorized_execution_inputs(
    population: OutcomeBoundedAnchorPopulation,
    factual_schedule: OutcomeFactualAnchorSchedule,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    decoder_config: FactorizedDecoderConfig,
    loss_config: LossConfig,
    optimization_budget: Mapping[str, object],
    device: torch.device | str,
    evaluation_chunk_size: int,
) -> tuple[torch.device, int, float, float]:
    if not isinstance(decoder_config, FactorizedDecoderConfig):
        raise TypeError(
            "decoder_config must be FactorizedDecoderConfig"
        )
    if not isinstance(loss_config, LossConfig):
        raise TypeError("loss_config must be LossConfig")
    if (
        loss_config.dice_weight != 1.0
        or loss_config.epsilon != 1.0e-6
    ):
        raise ValueError(
            "SVEF v4 fixes LossConfig(dice_weight=1.0, epsilon=1e-6)"
        )
    if not isinstance(optimization_budget, Mapping):
        raise TypeError("optimization_budget must be a mapping")
    if (
        optimization_budget.get("seed") != FACTORIZED_FROZEN_SEED
        or isinstance(optimization_budget.get("seed"), bool)
    ):
        raise ValueError("SVEF v4 fixes optimization seed at 42")
    if (
        optimization_budget.get("learning_rate")
        != FACTORIZED_FROZEN_LEARNING_RATE
        or isinstance(optimization_budget.get("learning_rate"), bool)
    ):
        raise ValueError("SVEF v4 fixes learning_rate at 0.001")
    if (
        optimization_budget.get("weight_decay")
        != FACTORIZED_FROZEN_WEIGHT_DECAY
        or isinstance(optimization_budget.get("weight_decay"), bool)
    ):
        raise ValueError("SVEF v4 fixes weight_decay at 0.0")
    if (
        isinstance(evaluation_chunk_size, bool)
        or evaluation_chunk_size
        != FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE
    ):
        raise ValueError("SVEF v4 fixes evaluation_chunk_size at 32")
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
            "factorized decoder channels differ from outcome inputs"
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
            "bounded SVEF requires a native subpixel path without field resize "
            f"({expected} != {(evaluation_height, evaluation_width)})"
        )
    return validated


def _maximum_abs(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    detached = value.detach()
    finite = detached[torch.isfinite(detached)]
    if finite.numel() == 0:
        return 0.0
    return float(finite.abs().max().cpu())


def audit_factorized_outcome_population(
    decoder: CURELiteFactorizedDecoder,
    population: OutcomeBoundedAnchorPopulation,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    *,
    device: torch.device,
    chunk_size: int,
) -> dict[str, object]:
    """Audit exact SVEF structure on all pairs before optimizer construction."""

    if not isinstance(decoder, CURELiteFactorizedDecoder):
        raise TypeError("decoder must be CURELiteFactorizedDecoder")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    pair_ids = tuple(pair.pair_id for pair in schedule.pairs)
    records: list[dict[str, object]] = []
    zero_feature_max = 0.0
    outside_gate_logit_max = 0.0
    outside_gate_probability_max = 0.0
    monotonic_logit_violations = 0
    monotonic_probability_violations = 0
    vacancy_monotonicity_violations = 0
    nonfinite_field_values = 0
    resize_count = 0
    clean_full_D_reachable = 0
    clean_D_reachable_pixels = 0
    clean_D_total_pixels = 0
    clean_support_fractions: list[float] = []
    clean_nonempty_H_pairs = 0
    component_support_fractions: list[float] = []
    component_positive_support_pairs = 0
    d_gate_values: list[float] = []
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
            gate_delta = minus.vacancy - plus.vacancy
            logit_delta = minus.logits - plus.logits
            probability_delta = (
                torch.sigmoid(minus.logits)
                - torch.sigmoid(plus.logits)
            )
            nonfinite_field_values += sum(
                int((~torch.isfinite(value)).sum().cpu())
                for value in (
                    plus.logits,
                    plus.baseline_logits,
                    plus.evidence,
                    plus.vacancy,
                    minus.logits,
                    minus.baseline_logits,
                    minus.evidence,
                    minus.vacancy,
                    zero_plus,
                    zero_minus,
                    gate_delta,
                    logit_delta,
                    probability_delta,
                )
            )
            support = gate_delta > 0.0
            outside = ~support
            outside_gate_logit_max = max(
                outside_gate_logit_max,
                _maximum_abs(logit_delta[outside]),
            )
            outside_gate_probability_max = max(
                outside_gate_probability_max,
                _maximum_abs(probability_delta[outside]),
            )
            vacancy_monotonicity_violations += int(
                (gate_delta < 0.0).sum().cpu()
            )
            monotonic_logit_violations += int(
                (logit_delta < 0.0).sum().cpu()
            )
            monotonic_probability_violations += int(
                (probability_delta < 0.0).sum().cpu()
            )
            resize_count += int(plus.field_resize_applied)
            resize_count += int(minus.field_resize_applied)

            valid = batch.pair_batch.image_valid_mask
            for index, pair_id in enumerate(selected):
                kind = batch.pair_batch.pair_kinds[index]
                response = batch.response_stratum[index]
                local_zero = batch.local_zero_stratum[index]
                response_count = int(response.sum().cpu())
                h_count = int(local_zero.sum().cpu())
                reachable = response & support[index]
                reachable_count = int(reachable.sum().cpu())
                valid_count = int(valid[index].sum().cpu())
                support_count = int((support[index] & valid[index]).sum().cpu())
                support_fraction = support_count / valid_count
                if kind == "clean_positive":
                    clean_nonempty_H_pairs += int(h_count > 0)
                    clean_D_total_pixels += response_count
                    clean_D_reachable_pixels += reachable_count
                    clean_full_D_reachable += int(
                        reachable_count == response_count
                    )
                    clean_support_fractions.append(support_fraction)
                    d_gate_values.extend(
                        float(value)
                        for value in gate_delta[index][response].cpu()
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
                        "gate_change_support_pixels": support_count,
                        "gate_change_support_fraction": support_fraction,
                        "field_resize_applied": (
                            plus.field_resize_applied
                            or minus.field_resize_applied
                        ),
                    }
                )

        factual = stack_state_examples(
            population.factual_miss,
            device=device,
        )
        _, _, factual_vacancy = decoder.vacancy_field(
            factual.occupancy,
            feature_size=tuple(int(value) for value in factual.feature.shape[-2:]),
        )
        factual_target = factual.target > 0.5
        factual_total_by_anchor = factual_target.flatten(1).sum(dim=1)
        factual_reachable_by_anchor = (
            factual_target & (factual_vacancy > 0.0)
        ).flatten(1).sum(dim=1)
        factual_full = (
            factual_total_by_anchor == factual_reachable_by_anchor
        )

    if len(records) != 222:
        raise RuntimeError("SVEF structural audit did not cover all pairs")
    if clean_D_total_pixels < 1 or not d_gate_values:
        raise RuntimeError("SVEF structural audit found no clean D pixels")

    factual_total = int(factual_total_by_anchor.sum().cpu())
    factual_reachable = int(factual_reachable_by_anchor.sum().cpu())
    expected_audit_decoder_calls = 4 * ceil(len(pair_ids) / chunk_size)
    expected_audit_decoder_state_evaluations = 4 * len(pair_ids)
    checks = {
        "zero_feature_occupancy_delta_exact_zero": (
            zero_feature_max == 0.0
        ),
        "gate_support_outside_logit_delta_exact_zero": (
            outside_gate_logit_max == 0.0
        ),
        "gate_support_outside_probability_delta_exact_zero": (
            outside_gate_probability_max == 0.0
        ),
        "all_audited_fields_finite": nonfinite_field_values == 0,
        "vacancy_deletion_monotonicity_exact": (
            vacancy_monotonicity_violations == 0
        ),
        "deletion_logit_monotonicity_exact": (
            monotonic_logit_violations == 0
        ),
        "deletion_probability_monotonicity_exact": (
            monotonic_probability_violations == 0
        ),
        "native_subpixel_path_without_resize": resize_count == 0,
        "all_clean_D_pixels_structurally_reachable": (
            clean_full_D_reachable == 206
            and clean_D_reachable_pixels == clean_D_total_pixels
        ),
        "all_clean_pairs_have_nonempty_H": (
            clean_nonempty_H_pairs == 206
        ),
        "all_component_null_pairs_have_positive_gate_support": (
            component_positive_support_pairs == 16
        ),
        "all_factual_targets_have_positive_vacancy": (
            int(factual_full.sum().cpu()) == 16
            and factual_total == factual_reachable
        ),
        "structural_audit_decoder_budget_exact": (
            audit_decoder_calls == expected_audit_decoder_calls
            and audit_decoder_state_evaluations
            == expected_audit_decoder_state_evaluations
        ),
    }
    return {
        "scope": "pretraining_D_R_full_population_SVEF_structure",
        "pair_count": len(records),
        "clean_pair_count": len(clean_support_fractions),
        "component_null_pair_count": len(component_support_fractions),
        "zero_feature_max_abs_occupancy_delta": zero_feature_max,
        "outside_gate_max_abs_logit_delta": outside_gate_logit_max,
        "outside_gate_max_abs_probability_delta": (
            outside_gate_probability_max
        ),
        "nonfinite_audited_field_values": nonfinite_field_values,
        "vacancy_deletion_monotonicity_violations": (
            vacancy_monotonicity_violations
        ),
        "deletion_logit_monotonicity_violations": (
            monotonic_logit_violations
        ),
        "deletion_probability_monotonicity_violations": (
            monotonic_probability_violations
        ),
        "field_resize_endpoint_count": resize_count,
        "clean_full_D_reachable_pairs": clean_full_D_reachable,
        "clean_nonempty_H_pairs": clean_nonempty_H_pairs,
        "clean_D_reachable_pixels": clean_D_reachable_pixels,
        "clean_D_total_pixels": clean_D_total_pixels,
        "component_positive_gate_support_pairs": (
            component_positive_support_pairs
        ),
        "factual_full_target_reachable_anchors": int(
            factual_full.sum().cpu()
        ),
        "factual_target_reachable_pixels": factual_reachable,
        "factual_target_total_pixels": factual_total,
        "clean_gate_support_fraction": {
            "minimum": min(clean_support_fractions),
            "mean": sum(clean_support_fractions)
            / len(clean_support_fractions),
            "maximum": max(clean_support_fractions),
        },
        "component_gate_support_fraction": {
            "minimum": min(component_support_fractions),
            "mean": sum(component_support_fractions)
            / len(component_support_fractions),
            "maximum": max(component_support_fractions),
        },
        "clean_D_gate_delta": {
            "minimum": min(d_gate_values),
            "mean": sum(d_gate_values) / len(d_gate_values),
            "maximum": max(d_gate_values),
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
            "factual_vacancy_field_calls": 1,
            "factual_vacancy_field_states": len(
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


def _tiny_target_report(records: list[Mapping[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for name, minimum, maximum in TINY_TARGET_STRATA:
        selected = [
            row
            for row in records
            if row.get("pair_kind") == "clean_positive"
            and int(row["D_pixels"]) >= minimum
            and (maximum is None or int(row["D_pixels"]) <= maximum)
        ]
        d_values = [float(row["D_mean_delta"]) for row in selected]
        h_values = [float(row["H_mean_abs_delta"]) for row in selected]
        joint = [
            d_value >= FACTORIZED_JOINT_D_THRESHOLD
            and h_value <= FACTORIZED_JOINT_H_THRESHOLD
            for d_value, h_value in zip(d_values, h_values, strict=True)
        ]
        output[name] = {
            "D_pixel_min": minimum,
            "D_pixel_max": maximum,
            "pair_count": len(selected),
            "D_pair_macro_mean_delta": (
                None if not d_values else sum(d_values) / len(d_values)
            ),
            "H_pair_macro_mean_abs_delta": (
                None if not h_values else sum(h_values) / len(h_values)
            ),
            "joint_pass_count": sum(joint),
            "joint_pass_fraction": (
                None if not joint else sum(joint) / len(joint)
            ),
        }
    return output


def factorized_computational_gates(
    initial: Mapping[str, object],
    final: Mapping[str, object],
) -> dict[str, object]:
    """Extend the frozen v3 means with a per-pair D/H joint requirement."""

    base = _computational_gates(initial, final)
    final_outcome = final.get("outcome_population")
    if not isinstance(final_outcome, Mapping):
        raise TypeError("final outcome metrics are malformed")
    records = final_outcome.get("per_pair")
    if not isinstance(records, list) or len(records) != 222:
        raise TypeError("final per-pair metrics are malformed")
    clean = [
        row
        for row in records
        if isinstance(row, Mapping)
        and row.get("pair_kind") == "clean_positive"
    ]
    if len(clean) != 206:
        raise RuntimeError("joint gate requires all 206 clean pairs")
    if any(
        int(row.get("H_pixels", 0)) < 1
        or row.get("H_mean_abs_delta") is None
        for row in clean
    ):
        raise RuntimeError(
            "joint gate requires a non-empty H stratum for every clean pair"
        )
    joint = [
        float(row["D_mean_delta"]) >= FACTORIZED_JOINT_D_THRESHOLD
        and float(row["H_mean_abs_delta"]) <= FACTORIZED_JOINT_H_THRESHOLD
        for row in clean
    ]
    joint_fraction = sum(joint) / len(joint)
    joint_check = {
        "value": joint_fraction,
        "direction": "min",
        "threshold": FACTORIZED_JOINT_THRESHOLD,
        "applicable": True,
        "status": "EVALUATED",
        "pass": joint_fraction >= FACTORIZED_JOINT_THRESHOLD,
    }
    checks = dict(base["checks"])
    checks[
        "clean_joint_D_ge_0_25_and_H_le_0_05_fraction"
    ] = joint_check
    observed = dict(base["observed"])
    observed[
        "clean_joint_D_ge_0_25_and_H_le_0_05_fraction"
    ] = joint_fraction
    thresholds = dict(base["thresholds"])
    thresholds[
        "clean_joint_D_ge_0_25_and_H_le_0_05_fraction_min"
    ] = FACTORIZED_JOINT_THRESHOLD
    return {
        "scope": "bounded_D_R_full_outcome_SVEF_model_code_gate",
        "not_detection_performance": True,
        "thresholds": thresholds,
        "observed": observed,
        "checks": checks,
        "tiny_target_strata": _tiny_target_report(clean),
        "all_pass": all(bool(value["pass"]) for value in checks.values()),
    }


def _structural_failure_result(
    decoder: CURELiteFactorizedDecoder,
    population: OutcomeBoundedAnchorPopulation,
    factual_schedule: OutcomeFactualAnchorSchedule,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    decoder_config: FactorizedDecoderConfig,
    loss_config: LossConfig,
    optimization_budget: Mapping[str, object],
    *,
    target_device: torch.device,
    evaluation_chunk_size: int,
    structural_audit: Mapping[str, object],
) -> dict[str, object]:
    checks = structural_audit.get("checks")
    compute_budget = structural_audit.get("compute_budget")
    if (
        not isinstance(checks, Mapping)
        or structural_audit.get("all_pass") is not False
        or not isinstance(compute_budget, Mapping)
    ):
        raise RuntimeError("malformed failed SVEF structural audit")
    result: dict[str, object] = {
        "schema_version": FACTORIZED_OUTCOME_BOUNDED_SCHEMA,
        "execution_status": "completed",
        "decision": "SVEF_STRUCTURAL_EXECUTION_FAIL",
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
                factorized_decoder_state_fingerprint(decoder)
            ),
        },
        "forward_budget": {
            "pretraining_structural_audit": dict(compute_budget),
            "training": {
                "calls": 0,
                "state_evaluations": 0,
            },
        },
        "trace": [],
        "interpretation": {
            "evidence_scope": (
                "fresh_SVEF_decoder_pretraining_D_R_structural_audit"
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


def execute_factorized_outcome_bounded(
    population: OutcomeBoundedAnchorPopulation,
    factual_schedule: OutcomeFactualAnchorSchedule,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    decoder_config: FactorizedDecoderConfig,
    loss_config: LossConfig,
    optimization_budget: Mapping[str, object],
    *,
    device: torch.device | str,
    evaluation_chunk_size: int = 32,
) -> dict[str, object]:
    """Train and audit one fresh SVEF decoder under the frozen 400 updates."""

    (
        target_device,
        seed,
        learning_rate,
        weight_decay,
    ) = _validate_factorized_execution_inputs(
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

        decoder = CURELiteFactorizedDecoder(decoder_config).to(target_device)
        structural_audit = audit_factorized_outcome_population(
            decoder,
            population,
            schedule,
            materializer,
            device=target_device,
            chunk_size=evaluation_chunk_size,
        )
        if structural_audit["all_pass"] is not True:
            materializer.verify_unchanged()
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
        initial_decoder_fingerprint = factorized_decoder_state_fingerprint(
            decoder
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

        final_decoder_fingerprint = factorized_decoder_state_fingerprint(
            decoder
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
        "calls": ceil(222 / evaluation_chunk_size) + 3,
        "state_evaluations": 2 * 222 + 2 * 16 + 2 * 16,
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
        "outcome_pair_exposure_values": sorted(set(actual_pair_counts)),
        "identity_null_optimizer_exposure": 0,
    }
    structural_checks = {
        "deterministic_runtime_contract_satisfied": (
            deterministic_runtime["contract_satisfied"] is True
            and deterministic_runtime["flags_restored_after_execution"] is True
        ),
        "SVEF_pretraining_structural_audit_passed": (
            structural_audit["all_pass"] is True
        ),
        "factual_anchor_and_identity_counts_exact": (
            len(population.factual_miss) == 16
            and len(population.factual_no_miss) == 16
            and len(population.identity_null) == 16
        ),
        "all_222_outcome_pairs_bound": (
            len(scheduled_ids) == 222
            and set(scheduled_ids) == set(materializer.canonical_pair_ids)
        ),
        "all_222_outcome_pairs_evaluated_initial": (
            initial["outcome_population"]["pair_ids"] == list(scheduled_ids)
        ),
        "all_222_outcome_pairs_evaluated_final": (
            final["outcome_population"]["pair_ids"] == list(scheduled_ids)
        ),
        "all_optimizer_updates_completed": len(trace) == 400,
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
    computational = factorized_computational_gates(initial, final)
    computational_pass = (
        structural_execution_pass and computational["all_pass"] is True
    )
    decision = (
        "SVEF_BOUNDED_MODEL_CODE_GATE_PASS"
        if computational_pass
        else (
            "SVEF_STRUCTURAL_EXECUTION_FAIL"
            if not structural_execution_pass
            else "SVEF_BOUNDED_MODEL_CODE_GATE_FAIL"
        )
    )
    result: dict[str, object] = {
        "schema_version": FACTORIZED_OUTCOME_BOUNDED_SCHEMA,
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
                "fresh_SVEF_decoder_bounded_D_R_full_outcome_population"
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
    "FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE",
    "FACTORIZED_FROZEN_LEARNING_RATE",
    "FACTORIZED_FROZEN_SEED",
    "FACTORIZED_FROZEN_WEIGHT_DECAY",
    "FACTORIZED_JOINT_D_THRESHOLD",
    "FACTORIZED_JOINT_H_THRESHOLD",
    "FACTORIZED_JOINT_THRESHOLD",
    "FACTORIZED_OUTCOME_BOUNDED_SCHEMA",
    "TINY_TARGET_STRATA",
    "audit_factorized_outcome_population",
    "execute_factorized_outcome_bounded",
    "factorized_decoder_state_fingerprint",
    "factorized_computational_gates",
]
