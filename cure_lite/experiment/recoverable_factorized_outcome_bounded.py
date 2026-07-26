"""D_R-only bounded model-code gate for CURE-Lite PR-SVEF v6.

The population, objective, optimizer step, deterministic schedules, vacancy
gate, topology, and thresholds are the frozen SVEF v4 objects.  This additive
runner instantiates a fresh polarity-recoverable decoder and changes no other
training or evaluation condition.  It never reads D_V or D_T.
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
from ..losses import CURELiteLoss
from ..paired_outcome_losses import OutcomeCompleteTransitionLoss
from ..recoverable_factorized_config import (
    RecoverableFactorizedDecoderConfig,
)
from ..recoverable_factorized_decoder import (
    CURELiteRecoverableFactorizedDecoder,
    polarity_recoverable_evidence,
)
from ..train.paired_outcome_step import outcome_complete_train_step
from .factorized_outcome_bounded import (
    FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE,
    FACTORIZED_FROZEN_LEARNING_RATE,
    FACTORIZED_FROZEN_SEED,
    FACTORIZED_FROZEN_WEIGHT_DECAY,
    audit_factorized_outcome_population,
    factorized_computational_gates,
)
from .paired_bounded_learnability import _deterministic_torch_runtime
from .paired_outcome_bounded import (
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


RECOVERABLE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA = (
    "cure-lite-pr-svef-v6-bounded-execution-v1"
)
RECOVERABLE_FACTORIZED_OUTCOME_METHOD_ID = "pr_svef_v6"
RECOVERABLE_OPERATOR_STRUCTURAL_CHECKS = (
    "v6_negative_half_forward_exact_zero",
    "v6_zero_forward_exact_zero",
    "v6_positive_forward_equals_v4",
    "v6_negative_half_recovery_gradient_matches_contract",
    "v6_zero_boundary_gradient_equals_half",
    "v6_positive_gradient_equals_v4",
)


def _update_digest(digest: object, value: bytes) -> None:
    if not hasattr(digest, "update"):
        raise TypeError("digest must provide update()")
    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def recoverable_factorized_decoder_state_fingerprint(
    decoder: CURELiteRecoverableFactorizedDecoder,
) -> str:
    """Hash v6 class identity, topology, and exact tensor state."""

    if not isinstance(decoder, CURELiteRecoverableFactorizedDecoder):
        raise TypeError(
            "decoder must be CURELiteRecoverableFactorizedDecoder"
        )
    digest = hashlib.sha256()
    digest.update(b"cure-lite-pr-svef-v6-decoder-state-v1")
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


def _audit_recoverable_operator_contract(
    *,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate all six frozen v6 forward/backward operator invariants."""

    raw = torch.tensor(
        [-4.0, -1.0, -1.0e-6, 0.0, 1.0e-4, 0.5, 4.0],
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    evidence = polarity_recoverable_evidence(raw)
    evidence.sum().backward()
    gradient = raw.grad
    if gradient is None:
        raise RuntimeError("PR-SVEF operator audit produced no gradient")

    positive = raw.detach()[4:]
    expected_positive_forward = (
        F.softplus(positive.square())
        - F.softplus(positive.new_zeros(()))
    )
    expected_negative_gradient = torch.sigmoid(raw.detach()[:3])
    expected_positive_gradient = (
        2.0
        * positive
        * torch.sigmoid(positive.square())
    )
    checks = {
        "v6_negative_half_forward_exact_zero": torch.equal(
            evidence[:3],
            torch.zeros_like(evidence[:3]),
        ),
        "v6_zero_forward_exact_zero": float(evidence[3].detach().cpu())
        == 0.0,
        "v6_positive_forward_equals_v4": torch.equal(
            evidence[4:],
            expected_positive_forward,
        ),
        "v6_negative_half_recovery_gradient_matches_contract": (
            torch.allclose(
                gradient[:3],
                expected_negative_gradient,
                rtol=1.0e-12,
                atol=1.0e-15,
            )
        ),
        "v6_zero_boundary_gradient_equals_half": (
            float(gradient[3].detach().cpu()) == 0.5
        ),
        "v6_positive_gradient_equals_v4": torch.allclose(
            gradient[4:],
            expected_positive_gradient,
            rtol=1.0e-12,
            atol=1.0e-15,
        ),
    }
    if tuple(checks) != RECOVERABLE_OPERATOR_STRUCTURAL_CHECKS:
        raise AssertionError("v6 operator structural check set drifted")
    return {
        "scope": "PR_SVEF_v6_frozen_operator_forward_backward_contract",
        "probe_raw": [float(value) for value in raw.detach().cpu()],
        "observed_forward": [
            float(value) for value in evidence.detach().cpu()
        ],
        "observed_gradient": [
            float(value) for value in gradient.detach().cpu()
        ],
        "expected_positive_forward_v4": [
            float(value)
            for value in expected_positive_forward.detach().cpu()
        ],
        "expected_negative_recovery_gradient": [
            float(value)
            for value in expected_negative_gradient.detach().cpu()
        ],
        "expected_positive_gradient_v4": [
            float(value)
            for value in expected_positive_gradient.detach().cpu()
        ],
        "checks": checks,
        "all_pass": all(checks.values()),
        "autograd_backward_calls": 1,
        "training_performed": False,
        "D_R_accessed": False,
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
        raise RuntimeError("malformed PR-SVEF population structural audit")
    if not isinstance(operator_checks, Mapping):
        raise RuntimeError("malformed PR-SVEF operator structural audit")
    if tuple(operator_checks) != RECOVERABLE_OPERATOR_STRUCTURAL_CHECKS:
        raise RuntimeError("PR-SVEF operator structural checks are incomplete")
    overlap = set(population_checks) & set(operator_checks)
    if overlap:
        raise RuntimeError(
            "PR-SVEF structural audit check names overlap: "
            f"{sorted(overlap)}"
        )

    checks = {
        **{str(key): bool(value) for key, value in population_checks.items()},
        **{str(key): bool(value) for key, value in operator_checks.items()},
    }
    combined = dict(population_audit)
    combined.update(
        {
            "scope": (
                "pretraining_D_R_full_population_SVEF_structure_plus_"
                "PR_SVEF_v6_operator"
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


def _validate_recoverable_execution_inputs(
    population: OutcomeBoundedAnchorPopulation,
    factual_schedule: OutcomeFactualAnchorSchedule,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    decoder_config: RecoverableFactorizedDecoderConfig,
    loss_config: LossConfig,
    optimization_budget: Mapping[str, object],
    device: torch.device | str,
    evaluation_chunk_size: int,
) -> tuple[torch.device, int, float, float]:
    if not isinstance(decoder_config, RecoverableFactorizedDecoderConfig):
        raise TypeError(
            "decoder_config must be RecoverableFactorizedDecoderConfig"
        )
    if not isinstance(loss_config, LossConfig):
        raise TypeError("loss_config must be LossConfig")
    if (
        loss_config.dice_weight != 1.0
        or loss_config.epsilon != 1.0e-6
    ):
        raise ValueError(
            "PR-SVEF v6 fixes "
            "LossConfig(dice_weight=1.0, epsilon=1e-6)"
        )
    if not isinstance(optimization_budget, Mapping):
        raise TypeError("optimization_budget must be a mapping")
    if (
        optimization_budget.get("seed") != FACTORIZED_FROZEN_SEED
        or isinstance(optimization_budget.get("seed"), bool)
    ):
        raise ValueError("PR-SVEF v6 fixes optimization seed at 42")
    if (
        optimization_budget.get("learning_rate")
        != FACTORIZED_FROZEN_LEARNING_RATE
        or isinstance(optimization_budget.get("learning_rate"), bool)
    ):
        raise ValueError("PR-SVEF v6 fixes learning_rate at 0.001")
    if (
        optimization_budget.get("weight_decay")
        != FACTORIZED_FROZEN_WEIGHT_DECAY
        or isinstance(optimization_budget.get("weight_decay"), bool)
    ):
        raise ValueError("PR-SVEF v6 fixes weight_decay at 0.0")
    if (
        isinstance(evaluation_chunk_size, bool)
        or evaluation_chunk_size
        != FACTORIZED_FROZEN_EVALUATION_CHUNK_SIZE
    ):
        raise ValueError("PR-SVEF v6 fixes evaluation_chunk_size at 32")
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
            "recoverable decoder channels differ from outcome inputs"
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
            "bounded PR-SVEF requires a native subpixel path without field "
            f"resize ({expected} != "
            f"{(evaluation_height, evaluation_width)})"
        )
    return validated


def _structural_failure_result(
    decoder: CURELiteRecoverableFactorizedDecoder,
    population: OutcomeBoundedAnchorPopulation,
    factual_schedule: OutcomeFactualAnchorSchedule,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    decoder_config: RecoverableFactorizedDecoderConfig,
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
        raise RuntimeError("malformed failed PR-SVEF structural audit")
    result: dict[str, object] = {
        "schema_version": RECOVERABLE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA,
        "method_id": RECOVERABLE_FACTORIZED_OUTCOME_METHOD_ID,
        "execution_status": "completed",
        "decision": "PR_SVEF_STRUCTURAL_EXECUTION_FAIL",
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
                recoverable_factorized_decoder_state_fingerprint(decoder)
            ),
        },
        "forward_budget": {
            "pretraining_structural_audit": dict(compute_budget),
            "training": {"calls": 0, "state_evaluations": 0},
        },
        "trace": [],
        "interpretation": {
            "evidence_scope": (
                "fresh_PR_SVEF_decoder_pretraining_D_R_structural_audit"
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


def execute_recoverable_factorized_outcome_bounded(
    population: OutcomeBoundedAnchorPopulation,
    factual_schedule: OutcomeFactualAnchorSchedule,
    schedule: OutcomePairSchedule,
    materializer: PairedOutcomeInputMaterializer,
    decoder_config: RecoverableFactorizedDecoderConfig,
    loss_config: LossConfig,
    optimization_budget: Mapping[str, object],
    *,
    device: torch.device | str,
    evaluation_chunk_size: int = 32,
) -> dict[str, object]:
    """Train and audit one fresh PR-SVEF decoder under 400 frozen updates."""

    (
        target_device,
        seed,
        learning_rate,
        weight_decay,
    ) = _validate_recoverable_execution_inputs(
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

        decoder = CURELiteRecoverableFactorizedDecoder(
            decoder_config
        ).to(target_device)
        population_structural_audit = audit_factorized_outcome_population(
            decoder,
            population,
            schedule,
            materializer,
            device=target_device,
            chunk_size=evaluation_chunk_size,
        )
        operator_structural_audit = _audit_recoverable_operator_contract(
            device=target_device,
        )
        structural_audit = _compose_pretraining_structural_audit(
            population_structural_audit,
            operator_structural_audit,
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
        initial_decoder_fingerprint = (
            recoverable_factorized_decoder_state_fingerprint(decoder)
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

        final_decoder_fingerprint = (
            recoverable_factorized_decoder_state_fingerprint(decoder)
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
        "PR_SVEF_pretraining_structural_audit_passed": (
            structural_audit["all_pass"] is True
        ),
        **{
            check_name: (
                structural_audit["operator_contract"]["checks"][check_name]
                is True
            )
            for check_name in RECOVERABLE_OPERATOR_STRUCTURAL_CHECKS
        },
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
        "total_forward_budget_exact": total_forward == expected_total_forward,
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
        "PR_SVEF_BOUNDED_MODEL_CODE_GATE_PASS"
        if computational_pass
        else (
            "PR_SVEF_STRUCTURAL_EXECUTION_FAIL"
            if not structural_execution_pass
            else "PR_SVEF_BOUNDED_MODEL_CODE_GATE_FAIL"
        )
    )
    result: dict[str, object] = {
        "schema_version": RECOVERABLE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA,
        "method_id": RECOVERABLE_FACTORIZED_OUTCOME_METHOD_ID,
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
                "fresh_PR_SVEF_decoder_bounded_D_R_full_outcome_population"
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
    "RECOVERABLE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA",
    "RECOVERABLE_FACTORIZED_OUTCOME_METHOD_ID",
    "RECOVERABLE_OPERATOR_STRUCTURAL_CHECKS",
    "execute_recoverable_factorized_outcome_bounded",
    "recoverable_factorized_decoder_state_fingerprint",
]
