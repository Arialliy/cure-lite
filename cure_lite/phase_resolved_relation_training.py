"""Deterministic learnability training for CURE-Lite relation decoder."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import torch
from torch import Tensor
from torch.nn import functional as F

from .cache.schema import stable_fingerprint
from .paired_types import tensor_content_fingerprint
from .phase_resolved_relation_decoder import (
    CURELitePhaseResolvedRelationDecoder,
    PhaseResolvedRelationDecoderConfig,
)
from .phase_resolved_relation_population import (
    PFCR_FEATURE_CHANNELS,
    PFCR_FEATURE_STRIDE,
    materialize_phase_resolved_relation_population,
    phase_resolved_relation_population_manifest,
)
from .phase_resolved_relation_preflight import (
    build_phase_resolved_relation_preflight_receipt,
)


PFCR_TRAINING_ALGORITHM_VERSION = (
    "cure-lite.phase-resolved-relation-training.v2"
)
PFCR_TRAIN_RELATION_DIM = 8
PFCR_TRAIN_POSITIVE_PROBABILITY = 0.951
PFCR_TRAIN_NEGATIVE_PROBABILITY = 0.049
PFCR_EVAL_POSITIVE_MIN_EXCLUSIVE = 0.95
PFCR_EVAL_NEGATIVE_MAX_EXCLUSIVE = 0.05


@dataclass(frozen=True)
class PhaseResolvedRelationTrainingConfig:
    """Frozen Development learnability schedule."""

    seed: int
    update_count: int = 320
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.0
    threshold: float = 0.5

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 0xFFFFFFFF
        ):
            raise ValueError("seed must be a uint32")
        if self.update_count != 320:
            raise ValueError("Development learnability fixes 320 updates")
        frozen_floats = {
            "learning_rate": 0.01,
            "beta1": 0.9,
            "beta2": 0.999,
            "weight_decay": 0.0,
            "threshold": 0.5,
        }
        for name, expected in frozen_floats.items():
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, float)
                or value != expected
            ):
                raise ValueError(
                    f"Development learnability fixes {name}"
                )

    @property
    def logit_margin(self) -> float:
        return log(
            PFCR_TRAIN_POSITIVE_PROBABILITY
            / PFCR_TRAIN_NEGATIVE_PROBABILITY
        )

    def manifest(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "update_count": self.update_count,
            "learning_rate": self.learning_rate,
            "betas": [self.beta1, self.beta2],
            "weight_decay": self.weight_decay,
            "threshold": self.threshold,
            "train_positive_probability": (
                PFCR_TRAIN_POSITIVE_PROBABILITY
            ),
            "train_negative_probability": (
                PFCR_TRAIN_NEGATIVE_PROBABILITY
            ),
            "logit_margin": self.logit_margin,
            "risk": (
                "statewise worst-positive and worst-negative absolute "
                "endpoint softplus margin"
            ),
        }


@dataclass(frozen=True)
class PhaseResolvedWorstEndpointLossFields:
    """Auditable statewise endpoint losses."""

    loss: Tensor
    per_state_loss: Tensor
    positive_loss: Tensor
    negative_loss: Tensor
    positive_state_mask: Tensor
    negative_state_mask: Tensor
    positive_min_logit: Tensor
    negative_max_logit: Tensor


def phase_resolved_worst_endpoint_loss(
    logits: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    occupancy: Tensor,
    *,
    logit_margin: float,
    audit: bool = True,
) -> PhaseResolvedWorstEndpointLossFields:
    """Align optimization with statewise strict positive/negative gates."""

    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError("logits must be a floating tensor")
    if logits.ndim != 4 or logits.shape[0] < 1 or logits.shape[1] != 1:
        raise ValueError("logits must be nonempty [B,1,H,W]")
    for name, value in (
        ("target", target),
        ("valid_mask", valid_mask),
        ("occupancy", occupancy),
    ):
        if (
            not isinstance(value, Tensor)
            or value.dtype != torch.bool
            or tuple(value.shape) != tuple(logits.shape)
        ):
            raise ValueError(
                f"{name} must be bool and align with logits"
            )
        if value.device != logits.device:
            raise ValueError(f"{name} and logits must share a device")
    if not isinstance(audit, bool):
        raise TypeError("audit must be bool")
    if audit:
        if target.any() and bool(torch.any(target & occupancy)):
            raise ValueError("target may not overlap occupancy")
        if torch.any(target & ~valid_mask):
            raise ValueError("target extends outside valid_mask")
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("logits must be finite")
    if (
        isinstance(logit_margin, bool)
        or not isinstance(logit_margin, float)
        or logit_margin <= 0.0
    ):
        raise ValueError("logit_margin must be a positive float")

    writable = valid_mask & ~occupancy
    positive = target & writable
    negative = ~target & writable
    batch = int(logits.shape[0])
    flat_logits = logits.reshape(batch, -1)
    flat_positive = positive.reshape(batch, -1)
    flat_negative = negative.reshape(batch, -1)
    positive_state = flat_positive.any(dim=1)
    negative_state = flat_negative.any(dim=1)
    if audit and not bool((positive_state | negative_state).all()):
        raise ValueError("every state requires writable supervision")

    positive_min = flat_logits.masked_fill(
        ~flat_positive,
        torch.inf,
    ).amin(dim=1)
    negative_max = flat_logits.masked_fill(
        ~flat_negative,
        -torch.inf,
    ).amax(dim=1)
    positive_loss = torch.where(
        positive_state,
        F.softplus(logit_margin - positive_min),
        torch.zeros_like(positive_min),
    )
    negative_loss = torch.where(
        negative_state,
        F.softplus(logit_margin + negative_max),
        torch.zeros_like(negative_max),
    )
    term_count = (
        positive_state.to(dtype=logits.dtype)
        + negative_state.to(dtype=logits.dtype)
    )
    per_state = (positive_loss + negative_loss) / term_count
    loss = per_state.mean()
    if audit and not bool(
        torch.stack(
            (
                torch.isfinite(loss),
                torch.isfinite(per_state).all(),
                torch.isfinite(positive_loss).all(),
                torch.isfinite(negative_loss).all(),
            )
        ).all()
    ):
        raise FloatingPointError("endpoint loss must be finite")
    return PhaseResolvedWorstEndpointLossFields(
        loss=loss,
        per_state_loss=per_state,
        positive_loss=positive_loss,
        negative_loss=negative_loss,
        positive_state_mask=positive_state,
        negative_state_mask=negative_state,
        positive_min_logit=positive_min,
        negative_max_logit=negative_max,
    )


def _development_batch(
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    states = materialize_phase_resolved_relation_population()
    return (
        torch.cat([state.feature for state in states], dim=0),
        torch.cat([state.occupancy for state in states], dim=0),
        torch.cat(
            [state.completion_target for state in states],
            dim=0,
        ),
        torch.cat([state.valid_mask for state in states], dim=0),
    )


def _model_state_fingerprint(
    model: CURELitePhaseResolvedRelationDecoder,
) -> str:
    return stable_fingerprint(
        {
            name: {
                "shape": list(value.shape),
                "content": tensor_content_fingerprint(
                    value.detach().cpu().reshape(-1)
                ),
            }
            for name, value in sorted(model.state_dict().items())
        }
    )


def _metrics(
    model: CURELitePhaseResolvedRelationDecoder,
    feature: Tensor,
    occupancy: Tensor,
    target: Tensor,
    valid_mask: Tensor,
    *,
    threshold: float,
) -> dict[str, object]:
    with torch.no_grad():
        fields = model.forward_fields(feature, occupancy)
        probability = fields.completion_probability
    writable = valid_mask & ~occupancy
    positive = target & writable
    negative = ~target & writable
    prediction = probability > threshold
    mismatch = (prediction != target) & writable
    return {
        "lossless_threshold_mismatch_pixel_count": int(
            mismatch.sum().item()
        ),
        "positive_pixel_count": int(positive.sum().item()),
        "negative_pixel_count": int(negative.sum().item()),
        "positive_probability_min": float(
            probability[positive].min().item()
        ),
        "negative_probability_max": float(
            probability[negative].max().item()
        ),
        "baseline_probability": float(
            torch.sigmoid(-F.softplus(model.baseline_raw)).item()
        ),
        "all_fields_finite": bool(
            torch.stack(
                (
                    torch.isfinite(fields.logits).all(),
                    torch.isfinite(fields.phase_evidence).all(),
                    torch.isfinite(
                        fields.relation.affinity
                    ).all(),
                    torch.isfinite(
                        fields.relation.relevant_coverage
                    ).all(),
                )
            ).all()
        ),
    }


def run_phase_resolved_relation_development(
    config: PhaseResolvedRelationTrainingConfig,
) -> dict[str, object]:
    """Run one exact deterministic 320-update learnability repeat."""

    if not isinstance(config, PhaseResolvedRelationTrainingConfig):
        raise TypeError(
            "config must be PhaseResolvedRelationTrainingConfig"
        )
    preflight = build_phase_resolved_relation_preflight_receipt(
        max_examples=1
    )
    if not preflight["decision"]["input_contract_v2_pass"]:
        raise RuntimeError("input contract v2 did not pass")
    feature, occupancy, target, valid_mask = _development_batch()
    population = phase_resolved_relation_population_manifest()
    decoder_config = PhaseResolvedRelationDecoderConfig(
        feature_channels=PFCR_FEATURE_CHANNELS,
        feature_stride=PFCR_FEATURE_STRIDE,
        relation_dim=PFCR_TRAIN_RELATION_DIM,
    )

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    trace: list[dict[str, object]] = []
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            model = CURELitePhaseResolvedRelationDecoder(decoder_config)
            initial_model_fingerprint = _model_state_fingerprint(model)
            initial_metrics = _metrics(
                model,
                feature,
                occupancy,
                target,
                valid_mask,
                threshold=config.threshold,
            )
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                weight_decay=config.weight_decay,
            )
            for update_index in range(config.update_count):
                logits = model(feature, occupancy)
                loss_fields = phase_resolved_worst_endpoint_loss(
                    logits,
                    target,
                    valid_mask,
                    occupancy,
                    logit_margin=config.logit_margin,
                )
                optimizer.zero_grad(set_to_none=True)
                loss_fields.loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is None or not bool(
                        torch.isfinite(parameter.grad).all()
                    ):
                        raise FloatingPointError(
                            "every model gradient must be finite"
                        )
                optimizer.step()
                for parameter in model.parameters():
                    if not bool(torch.isfinite(parameter).all()):
                        raise FloatingPointError(
                            "every model parameter must be finite"
                        )
                if update_index in {
                    0,
                    1,
                    3,
                    7,
                    15,
                    31,
                    63,
                    127,
                    255,
                    config.update_count - 1,
                }:
                    trace.append(
                        {
                            "update_index": update_index,
                            "loss": float(loss_fields.loss.item()),
                            "worst_positive_loss": float(
                                loss_fields.positive_loss.max().item()
                            ),
                            "worst_negative_loss": float(
                                loss_fields.negative_loss.max().item()
                            ),
                        }
                    )
            final_metrics = _metrics(
                model,
                feature,
                occupancy,
                target,
                valid_mask,
                threshold=config.threshold,
            )
            final_model_fingerprint = _model_state_fingerprint(model)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)

    gate_results = {
        "zero_threshold_mismatch": (
            final_metrics[
                "lossless_threshold_mismatch_pixel_count"
            ]
            == 0
        ),
        "positive_probability": (
            final_metrics["positive_probability_min"]
            > PFCR_EVAL_POSITIVE_MIN_EXCLUSIVE
        ),
        "negative_probability": (
            final_metrics["negative_probability_max"]
            < PFCR_EVAL_NEGATIVE_MAX_EXCLUSIVE
        ),
        "finite_fields": final_metrics["all_fields_finite"],
    }
    passed = all(gate_results.values())
    payload: dict[str, object] = {
        "schema_version": PFCR_TRAINING_ALGORITHM_VERSION,
        "scope": {
            "model": "CURE-Lite",
            "stage": "dataset-free learned relation decoder Development",
            "real_dataset_training": False,
            "dataset_metrics_read": False,
            "full_CURE_in_scope": False,
        },
        "config": config.manifest(),
        "decoder": {
            "feature_channels": decoder_config.feature_channels,
            "feature_stride": decoder_config.feature_stride,
            "relation_dim": decoder_config.relation_dim,
            "parameter_count": decoder_config.expected_parameter_count,
            "relation_policy": decoder_config.relation_policy,
            "feature_normalization_policy": (
                decoder_config.feature_normalization_policy
            ),
            "evidence_policy": decoder_config.evidence_policy,
            "evidence_ceiling": decoder_config.evidence_ceiling,
            "release_policy": decoder_config.release_policy,
            "output_policy": decoder_config.output_policy,
        },
        "input_contract_receipt_fingerprint": preflight[
            "receipt_fingerprint"
        ],
        "population_fingerprint": population[
            "population_fingerprint"
        ],
        "initial_model_fingerprint": initial_model_fingerprint,
        "final_model_fingerprint": final_model_fingerprint,
        "initial_metrics": initial_metrics,
        "trace": trace,
        "final_metrics": final_metrics,
        "gates": {
            "positive_probability_min_exclusive": (
                PFCR_EVAL_POSITIVE_MIN_EXCLUSIVE
            ),
            "negative_probability_max_exclusive": (
                PFCR_EVAL_NEGATIVE_MAX_EXCLUSIVE
            ),
            "results": gate_results,
        },
        "decision": {
            "development_learnability_pass": passed,
            "learned_relation_mechanism_supported_on_fixed_population": (
                passed
            ),
            "real_dataset_training_implementation_authorized": passed,
            "real_dataset_model_success_claimed": False,
            "full_CURE_authorized": False,
        },
    }
    payload["result_fingerprint"] = stable_fingerprint(payload)
    return payload


__all__ = [
    "PFCR_EVAL_NEGATIVE_MAX_EXCLUSIVE",
    "PFCR_EVAL_POSITIVE_MIN_EXCLUSIVE",
    "PFCR_TRAINING_ALGORITHM_VERSION",
    "PFCR_TRAIN_RELATION_DIM",
    "PFCR_TRAIN_NEGATIVE_PROBABILITY",
    "PFCR_TRAIN_POSITIVE_PROBABILITY",
    "PhaseResolvedRelationTrainingConfig",
    "PhaseResolvedWorstEndpointLossFields",
    "phase_resolved_worst_endpoint_loss",
    "run_phase_resolved_relation_development",
]
