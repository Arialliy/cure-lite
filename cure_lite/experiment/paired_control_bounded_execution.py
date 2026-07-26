"""Bounded D_R-only execution checks for the eight frozen matched controls.

The routines in this module are additive to the proposed paired route.  They
exercise each matched-control training path with the frozen 4 + 4 + 2-pair
state budget, but intentionally do not evaluate whether a control learns the
positive response.  Detection performance, calibration, D_V, D_T, and formal
800-epoch authorization are outside this module.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import sqrt
from typing import Mapping

import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from ..config import DecoderConfig, LossConfig
from ..decoder import CURELiteDecoder
from ..losses import CURELiteLoss
from ..paired_control_inputs import (
    DCTCoordinateBasis,
    TargetPermutationPlan,
    build_dct_coordinate_basis,
    build_target_permutation,
    feature_only_zero_occupancy,
    materialize_permuted_label_increments,
    nominal_zero_feature_like,
    target_permutation_compatible,
)
from ..paired_control_losses import (
    build_after_only_synthetic_supervision,
    build_geometry_matched_endpoint_supervision,
    minus_detached_paired_difference_loss,
    plus_detached_paired_difference_loss,
)
from ..paired_losses import PairedDifferenceLoss
from ..paired_types import (
    PairCatalog,
    PairExample,
    tensor_content_fingerprint,
)
from ..train.paired_control_step import (
    CONTROL_KINDS,
    paired_control_train_step,
)
from .artifacts import decoder_state_fingerprint
from .paired_bounded_learnability import (
    BoundedMicroPopulation,
    BoundedMicroSchedule,
    _ForwardLedger,
    _deterministic_torch_runtime,
    _factual_batches,
    _pair_batch,
)


CONTROL_BOUNDED_EXECUTION_SCHEMA = (
    "cure-lite-paired-control-bounded-execution-v1"
)
CONTROL_RUNTIME_BINDING_SCHEMA = (
    "cure-lite-paired-control-runtime-binding-v1"
)
CONTROL_SEMANTICS_SCHEMA = "cure-lite-paired-control-semantics-v1"


def _stack_bool(
    values: tuple[Tensor, ...],
    *,
    device: torch.device | str,
) -> Tensor:
    if not values:
        raise ValueError("cannot stack an empty control tensor sequence")
    return torch.stack(values, dim=0).to(device=device)


def _pair_ids(
    population: BoundedMicroPopulation,
    schedule: BoundedMicroSchedule,
    update: int,
) -> tuple[str, str]:
    first, second = schedule.pair_indices[update]
    return (
        population.clean_pairs[first].pair_id,
        population.clean_pairs[second].pair_id,
    )


@dataclass(frozen=True)
class ControlRuntimeBinding:
    """Immutable control inputs derived from the frozen D_R catalog."""

    pair_catalog_fingerprint: str
    population_fingerprint: str
    gt_union_by_sample: Mapping[str, Tensor]
    coordinate_basis: DCTCoordinateBasis
    permutation_plan: TargetPermutationPlan
    permuted_target_by_recipient: Mapping[str, Tensor]
    assignment_by_recipient: Mapping[str, Mapping[str, str]]
    binding_fingerprint: str

    def __post_init__(self) -> None:
        if not self.permutation_plan.ready:
            raise ValueError("target permutation must be READY")
        if set(self.permuted_target_by_recipient) != set(
            self.permutation_plan.canonical_pair_ids
        ):
            raise ValueError("permuted targets do not cover the full pair plan")
        if set(self.assignment_by_recipient) != set(
            self.permutation_plan.canonical_pair_ids
        ):
            raise ValueError("permutation assignments are incomplete")
        payload = self.canonical_payload()
        if stable_fingerprint(payload) != self.binding_fingerprint:
            raise ValueError("control runtime binding fingerprint changed")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": CONTROL_RUNTIME_BINDING_SCHEMA,
            "pair_catalog_fingerprint": self.pair_catalog_fingerprint,
            "population_fingerprint": self.population_fingerprint,
            "gt_union_by_sample": [
                {
                    "sample_id": sample_id,
                    "fingerprint": tensor_content_fingerprint(
                        self.gt_union_by_sample[sample_id]
                    ),
                }
                for sample_id in sorted(self.gt_union_by_sample)
            ],
            "coordinate_basis_fingerprint": (
                self.coordinate_basis.basis_fingerprint
            ),
            "target_permutation_plan_fingerprint": (
                self.permutation_plan.plan_fingerprint
            ),
            "assignments": [
                dict(self.assignment_by_recipient[pair_id])
                for pair_id in self.permutation_plan.canonical_pair_ids
            ],
        }


def build_control_runtime_binding(
    pair_catalog: PairCatalog,
    population: BoundedMicroPopulation,
    gt_union_by_sample: Mapping[str, Tensor],
    *,
    expected_permutation_fingerprint: str,
    expected_dct_basis_fingerprint: str,
) -> ControlRuntimeBinding:
    """Close real recipient -> donor -> target identities before training."""

    if not isinstance(pair_catalog, PairCatalog):
        raise TypeError("pair_catalog must be PairCatalog")
    if not isinstance(population, BoundedMicroPopulation):
        raise TypeError("population must be BoundedMicroPopulation")
    if pair_catalog.catalog_fingerprint != population.pair_catalog_fingerprint:
        raise ValueError("population and pair catalog fingerprints differ")
    selected_sources = {pair.sample_id for pair in population.clean_pairs}
    if not selected_sources <= set(gt_union_by_sample):
        raise ValueError("GT unions do not cover the selected clean-pair sources")
    normalized_gt: dict[str, Tensor] = {}
    expected_eval_shape = tuple(population.clean_pairs[0].image_valid_mask.shape)
    for sample_id in sorted(selected_sources):
        value = gt_union_by_sample[sample_id]
        if (
            not isinstance(value, Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.bool
            or tuple(value.shape) != expected_eval_shape
        ):
            raise TypeError(
                "every GT union must be a CPU bool tensor on the pair grid"
            )
        normalized_gt[sample_id] = value.detach().clone().contiguous()

    feature = population.clean_pairs[0].feature
    basis = build_dct_coordinate_basis(
        channels=int(feature.shape[1]),
        height=int(feature.shape[2]),
        width=int(feature.shape[3]),
        dtype=feature.dtype,
    )
    if basis.basis_fingerprint != expected_dct_basis_fingerprint:
        raise RuntimeError("runtime DCT basis differs from static preflight")

    plan = build_target_permutation(pair_catalog.clean_positive)
    if (
        not plan.ready
        or plan.plan_fingerprint != expected_permutation_fingerprint
    ):
        raise RuntimeError(
            "runtime target permutation differs from static preflight"
        )
    materialized = materialize_permuted_label_increments(
        pair_catalog.clean_positive,
        plan,
    )
    target_by_recipient = {
        pair_id: target
        for pair_id, target in zip(
            plan.canonical_pair_ids,
            materialized,
            strict=True,
        )
    }
    pair_by_id = {
        pair.pair_id: pair for pair in pair_catalog.clean_positive
    }
    assignment_by_recipient: dict[str, Mapping[str, str]] = {}
    for assignment in plan.assignments:
        recipient = pair_by_id[assignment.recipient_pair_id]
        donor = pair_by_id[assignment.donor_pair_id]
        target = target_by_recipient[assignment.recipient_pair_id]
        if (
            not target_permutation_compatible(recipient, donor)
            or tensor_content_fingerprint(donor.clean_increment)
            != assignment.donor_target_fingerprint
            or not torch.equal(
                target,
                donor.clean_increment.to(torch.float32),
            )
            or torch.any(target.to(torch.bool) & ~recipient.image_valid_mask)
        ):
            raise RuntimeError("target-permutation runtime closure failed")
        assignment_by_recipient[assignment.recipient_pair_id] = {
            **assignment.canonical_payload(),
            "runtime_target_fingerprint": tensor_content_fingerprint(target),
        }
    canonical = {
        "schema_version": CONTROL_RUNTIME_BINDING_SCHEMA,
        "pair_catalog_fingerprint": pair_catalog.catalog_fingerprint,
        "population_fingerprint": population.population_fingerprint,
        "gt_union_by_sample": [
            {
                "sample_id": sample_id,
                "fingerprint": tensor_content_fingerprint(
                    normalized_gt[sample_id]
                ),
            }
            for sample_id in sorted(normalized_gt)
        ],
        "coordinate_basis_fingerprint": basis.basis_fingerprint,
        "target_permutation_plan_fingerprint": plan.plan_fingerprint,
        "assignments": [
            dict(assignment_by_recipient[pair_id])
            for pair_id in plan.canonical_pair_ids
        ],
    }
    return ControlRuntimeBinding(
        pair_catalog_fingerprint=pair_catalog.catalog_fingerprint,
        population_fingerprint=population.population_fingerprint,
        gt_union_by_sample=normalized_gt,
        coordinate_basis=basis,
        permutation_plan=plan,
        permuted_target_by_recipient=target_by_recipient,
        assignment_by_recipient=assignment_by_recipient,
        binding_fingerprint=stable_fingerprint(canonical),
    )


def _selected_pairs(
    population: BoundedMicroPopulation,
    schedule: BoundedMicroSchedule,
    update: int,
) -> tuple[PairExample, PairExample]:
    first, second = schedule.pair_indices[update]
    return population.clean_pairs[first], population.clean_pairs[second]


def _control_kwargs(
    control_kind: str,
    pairs: tuple[PairExample, PairExample],
    binding: ControlRuntimeBinding,
    *,
    device: torch.device | str,
) -> dict[str, object]:
    gt_union = _stack_bool(
        tuple(binding.gt_union_by_sample[pair.sample_id] for pair in pairs),
        device=device,
    )
    if control_kind == "independent_endpoint":
        return {
            "gt_union": gt_union,
            "completion_plus": _stack_bool(
                tuple(pair.completion_plus for pair in pairs),
                device=device,
            ),
            "completion_minus": _stack_bool(
                tuple(pair.completion_minus for pair in pairs),
                device=device,
            ),
        }
    if control_kind == "after_only":
        return {"gt_union": gt_union}
    if control_kind == "coordinate_basis":
        return {"coordinate_basis": binding.coordinate_basis}
    if control_kind == "target_permutation":
        return {
            "permuted_label_increment": torch.stack(
                tuple(
                    binding.permuted_target_by_recipient[pair.pair_id]
                    for pair in pairs
                ),
                dim=0,
            ).to(device=device)
        }
    return {}


def build_control_semantics_receipt(
    population: BoundedMicroPopulation,
    schedule: BoundedMicroSchedule,
    binding: ControlRuntimeBinding,
) -> dict[str, object]:
    """Check all eight control semantics without a decoder forward."""

    pairs = _selected_pairs(population, schedule, 0)
    pair_batch = _pair_batch(population, schedule, 0, device="cpu")
    kwargs_independent = _control_kwargs(
        "independent_endpoint",
        pairs,
        binding,
        device="cpu",
    )
    plus = build_geometry_matched_endpoint_supervision(
        kwargs_independent["completion_plus"],
        pair_batch.occupancy_plus,
        kwargs_independent["gt_union"],
        pair_batch.image_valid_mask,
    )
    minus = build_geometry_matched_endpoint_supervision(
        kwargs_independent["completion_minus"],
        pair_batch.occupancy_minus,
        kwargs_independent["gt_union"],
        pair_batch.image_valid_mask,
    )
    after = build_after_only_synthetic_supervision(
        pair_batch.label_increment.to(torch.bool),
        pair_batch.occupancy_minus,
        kwargs_independent["gt_union"],
        pair_batch.image_valid_mask,
    )
    zero_feature = nominal_zero_feature_like(pair_batch.feature)
    feature_only_plus, feature_only_minus = feature_only_zero_occupancy(
        pair_batch.occupancy_plus,
        pair_batch.occupancy_minus,
    )
    permutation = _control_kwargs(
        "target_permutation",
        pairs,
        binding,
        device="cpu",
    )["permuted_label_increment"]

    label = pair_batch.label_increment
    valid = pair_batch.image_valid_mask
    plus_logits = torch.zeros_like(label, requires_grad=True)
    minus_logits = torch.full_like(label, 0.25, requires_grad=True)
    plus_detached = plus_detached_paired_difference_loss(
        plus_logits,
        minus_logits,
        label,
        valid,
    )["total"]
    plus_detached.backward()
    plus_detach_semantics = (
        plus_logits.grad is None
        and minus_logits.grad is not None
        and torch.isfinite(minus_logits.grad).all()
        and torch.count_nonzero(minus_logits.grad) > 0
    )

    plus_logits = torch.zeros_like(label, requires_grad=True)
    minus_logits = torch.full_like(label, 0.25, requires_grad=True)
    minus_detached = minus_detached_paired_difference_loss(
        plus_logits,
        minus_logits,
        label,
        valid,
    )["total"]
    minus_detached.backward()
    minus_detach_semantics = (
        minus_logits.grad is None
        and plus_logits.grad is not None
        and torch.isfinite(plus_logits.grad).all()
        and torch.count_nonzero(plus_logits.grad) > 0
    )

    checks = {
        "independent_endpoint_exact_supervision": (
            torch.equal(
                plus["target"],
                kwargs_independent["completion_plus"].to(torch.float32),
            )
            and torch.equal(
                minus["target"],
                kwargs_independent["completion_minus"].to(torch.float32),
            )
        ),
        "after_only_exact_selected_completion": torch.equal(
            after["target"],
            pair_batch.label_increment,
        ),
        "zero_feature_all_zero": torch.count_nonzero(zero_feature) == 0,
        "coordinate_basis_static_identity": (
            binding.coordinate_basis.basis_fingerprint
            == binding.coordinate_basis.basis_fingerprint
            and torch.all(
                binding.coordinate_basis.tensor.flatten(2).abs().sum(2) > 0
            )
        ),
        "feature_only_same_zero_occupancy": (
            feature_only_plus is feature_only_minus
            and torch.count_nonzero(feature_only_plus) == 0
        ),
        "target_permutation_runtime_targets_bound": (
            isinstance(permutation, Tensor)
            and permutation.shape == pair_batch.label_increment.shape
            and all(
                tensor_content_fingerprint(permutation[index].cpu())
                == binding.assignment_by_recipient[pair.pair_id][
                    "runtime_target_fingerprint"
                ]
                for index, pair in enumerate(pairs)
            )
        ),
        "plus_detach_exact_gradient_semantics": bool(
            plus_detach_semantics
        ),
        "minus_detach_exact_gradient_semantics": bool(
            minus_detach_semantics
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    return {
        "schema_version": CONTROL_SEMANTICS_SCHEMA,
        "checks": checks,
        "all_control_semantics_pass": all(bool(value) for value in checks.values()),
        "reference_pair_ids": list(pair_batch.pair_ids),
        "runtime_binding_fingerprint": binding.binding_fingerprint,
    }


def _parameter_gradient_norm(decoder: CURELiteDecoder) -> float:
    return sqrt(
        sum(
            float(parameter.grad.detach().double().square().sum().cpu())
            for parameter in decoder.parameters()
            if parameter.grad is not None
        )
    )


def execute_control_bounded_execution(
    population: BoundedMicroPopulation,
    schedule: BoundedMicroSchedule,
    binding: ControlRuntimeBinding,
    config: Mapping[str, object],
    *,
    device: torch.device | str,
) -> dict[str, object]:
    """Run all eight engineering-only 400-update control executions."""

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    optimization = config.get("optimization")
    budget = config.get("budget")
    determinism = config.get("determinism")
    controls = config.get("controls")
    if (
        not isinstance(optimization, Mapping)
        or not isinstance(budget, Mapping)
        or not isinstance(determinism, Mapping)
        or controls != list(CONTROL_KINDS)
    ):
        raise RuntimeError("control-bounded execution config is malformed")
    if (
        schedule.optimizer_updates != int(budget["updates_per_control"])
        or schedule.optimizer_updates != 400
    ):
        raise RuntimeError("control execution update budget changed")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    seed = int(optimization["seed"])
    decoder_config = DecoderConfig(**dict(optimization["decoder"]))
    loss_config = LossConfig(**dict(optimization["loss"]))
    semantic_receipt = build_control_semantics_receipt(
        population,
        schedule,
        binding,
    )
    if semantic_receipt["all_control_semantics_pass"] is not True:
        raise RuntimeError("control semantic preflight failed")

    cuda_devices: list[int] = []
    if target_device.type == "cuda":
        cuda_devices = [
            torch.cuda.current_device()
            if target_device.index is None
            else target_device.index
        ]
    control_results: dict[str, object] = {}
    initial_fingerprints: set[str] = set()
    permutation_exposure: Counter[
        tuple[str, str, str, str]
    ] = Counter()
    with _deterministic_torch_runtime(
        target_device,
        determinism,
    ) as deterministic_runtime, torch.random.fork_rng(devices=cuda_devices):
        for control_kind in CONTROL_KINDS:
            torch.manual_seed(seed)
            if target_device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            decoder = CURELiteDecoder(decoder_config).to(target_device)
            absolute = CURELiteLoss(loss_config)
            paired = PairedDifferenceLoss()
            optimizer = torch.optim.Adam(
                decoder.parameters(),
                lr=float(optimization["learning_rate"]),
                weight_decay=float(optimization["weight_decay"]),
            )
            initial_fingerprint = decoder_state_fingerprint(decoder)
            initial_fingerprints.add(initial_fingerprint)
            initial_norm = sqrt(
                sum(
                    float(
                        parameter.detach().double().square().sum().cpu()
                    )
                    for parameter in decoder.parameters()
                )
            )
            ledger = _ForwardLedger(decoder)
            minimum_gradient = float("inf")
            maximum_gradient = 0.0
            nonfinite_gradients = 0
            zero_gradients = 0
            nonfinite_losses = 0
            update_count = 0
            basis_fingerprints: set[str] = set()
            try:
                for update in range(schedule.optimizer_updates):
                    pairs = _selected_pairs(population, schedule, update)
                    before = ledger.snapshot()
                    logs = paired_control_train_step(
                        decoder,
                        absolute,
                        paired,
                        optimizer,
                        _factual_batches(
                            population,
                            schedule,
                            update,
                            device=target_device,
                        ),
                        _pair_batch(
                            population,
                            schedule,
                            update,
                            device=target_device,
                        ),
                        control_kind=control_kind,
                        **_control_kwargs(
                            control_kind,
                            pairs,
                            binding,
                            device=target_device,
                        ),
                    )
                    after = ledger.snapshot()
                    if after[0] - before[0] != 3 or after[1] - before[1] != 12:
                        raise RuntimeError(
                            "per-update control forward budget changed"
                        )
                    total = float(logs["total"])
                    if not torch.isfinite(torch.tensor(total)):
                        nonfinite_losses += 1
                    gradient = _parameter_gradient_norm(decoder)
                    if not torch.isfinite(torch.tensor(gradient)):
                        nonfinite_gradients += 1
                    if gradient <= 0.0:
                        zero_gradients += 1
                    minimum_gradient = min(minimum_gradient, gradient)
                    maximum_gradient = max(maximum_gradient, gradient)
                    if "control/basis_fingerprint" in logs:
                        basis_fingerprints.add(
                            str(logs["control/basis_fingerprint"])
                        )
                    if control_kind == "target_permutation":
                        for pair_id in _pair_ids(
                            population,
                            schedule,
                            update,
                        ):
                            assignment = binding.assignment_by_recipient[
                                pair_id
                            ]
                            target = binding.permuted_target_by_recipient[
                                pair_id
                            ]
                            runtime_fingerprint = tensor_content_fingerprint(
                                target
                            )
                            if (
                                runtime_fingerprint
                                != assignment["runtime_target_fingerprint"]
                            ):
                                raise RuntimeError(
                                    "runtime permutation target changed"
                                )
                            permutation_exposure[
                                (
                                    pair_id,
                                    assignment["donor_pair_id"],
                                    assignment["donor_target_fingerprint"],
                                    assignment["runtime_target_fingerprint"],
                                )
                            ] += 1
                    update_count += int(logs["optimizer_steps"])
            finally:
                ledger.close()
            calls, states = ledger.snapshot()
            final_fingerprint = decoder_state_fingerprint(decoder)
            final_norm = sqrt(
                sum(
                    float(
                        parameter.detach().double().square().sum().cpu()
                    )
                    for parameter in decoder.parameters()
                )
            )
            checks = {
                "optimizer_updates_exact": (
                    update_count
                    == schedule.optimizer_updates
                    == int(budget["updates_per_control"])
                ),
                "decoder_forward_calls_exact": (
                    calls == int(budget["forward_calls_per_control"])
                ),
                "decoder_state_evaluations_exact": (
                    states == int(budget["state_evaluations_per_control"])
                ),
                "losses_finite": nonfinite_losses == 0,
                "gradients_finite": nonfinite_gradients == 0,
                "total_gradient_nonzero_each_update": zero_gradients == 0,
                "decoder_parameters_changed": (
                    final_fingerprint != initial_fingerprint
                ),
                "control_kind_semantics_preflight_pass": (
                    semantic_receipt["checks"][
                        {
                            "independent_endpoint": (
                                "independent_endpoint_exact_supervision"
                            ),
                            "after_only": (
                                "after_only_exact_selected_completion"
                            ),
                            "zero_feature": "zero_feature_all_zero",
                            "coordinate_basis": (
                                "coordinate_basis_static_identity"
                            ),
                            "feature_only": (
                                "feature_only_same_zero_occupancy"
                            ),
                            "target_permutation": (
                                "target_permutation_runtime_targets_bound"
                            ),
                            "plus_detach": (
                                "plus_detach_exact_gradient_semantics"
                            ),
                            "minus_detach": (
                                "minus_detach_exact_gradient_semantics"
                            ),
                        }[control_kind]
                    ]
                    is True
                ),
                "coordinate_basis_fingerprint_exact": (
                    basis_fingerprints
                    == {binding.coordinate_basis.basis_fingerprint}
                    if control_kind == "coordinate_basis"
                    else not basis_fingerprints
                ),
            }
            control_results[control_kind] = {
                "checks": checks,
                "execution_pass": all(checks.values()),
                "optimizer_updates": update_count,
                "decoder_forward_calls": calls,
                "decoder_state_evaluations": states,
                "parameters": {
                    "initial_decoder_fingerprint": initial_fingerprint,
                    "final_decoder_fingerprint": final_fingerprint,
                    "initial_l2_norm": initial_norm,
                    "final_l2_norm": final_norm,
                },
                "gradients": {
                    "minimum_update_l2_norm": minimum_gradient,
                    "maximum_update_l2_norm": maximum_gradient,
                    "nonfinite_updates": nonfinite_gradients,
                    "zero_norm_updates": zero_gradients,
                },
                "nonfinite_loss_updates": nonfinite_losses,
            }

    expected_permutation_exposure = {
        (
            pair.pair_id,
            binding.assignment_by_recipient[pair.pair_id][
                "donor_pair_id"
            ],
            binding.assignment_by_recipient[pair.pair_id][
                "donor_target_fingerprint"
            ],
            binding.assignment_by_recipient[pair.pair_id][
                "runtime_target_fingerprint"
            ],
        ): schedule.pair_counts[index]
        for index, pair in enumerate(population.clean_pairs)
    }
    permutation_runtime_closure_pass = (
        dict(permutation_exposure) == expected_permutation_exposure
    )
    total_updates = sum(
        int(result["optimizer_updates"])
        for result in control_results.values()
    )
    total_calls = sum(
        int(result["decoder_forward_calls"])
        for result in control_results.values()
    )
    total_states = sum(
        int(result["decoder_state_evaluations"])
        for result in control_results.values()
    )
    global_checks = {
        "all_eight_controls_executed": (
            tuple(control_results) == CONTROL_KINDS
        ),
        "same_initial_decoder_across_controls": (
            len(initial_fingerprints) == 1
        ),
        "total_optimizer_updates_exact": (
            total_updates == int(budget["total_optimizer_updates"])
        ),
        "total_forward_calls_exact": (
            total_calls == int(budget["total_forward_calls"])
        ),
        "total_state_evaluations_exact": (
            total_states == int(budget["total_state_evaluations"])
        ),
        "target_permutation_runtime_closure_pass": (
            permutation_runtime_closure_pass
        ),
        "all_control_semantics_pass": (
            semantic_receipt["all_control_semantics_pass"] is True
        ),
        "all_control_executions_pass": all(
            result["execution_pass"] is True
            for result in control_results.values()
        ),
        "deterministic_runtime_contract_satisfied": (
            deterministic_runtime["contract_satisfied"] is True
            and deterministic_runtime["flags_restored_after_execution"] is True
        ),
    }
    return {
        "schema_version": CONTROL_BOUNDED_EXECUTION_SCHEMA,
        "execution_status": "completed",
        "split": "D_R",
        "device": str(target_device),
        "population_fingerprint": population.population_fingerprint,
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "runtime_binding_fingerprint": binding.binding_fingerprint,
        "control_order": list(CONTROL_KINDS),
        "controls": control_results,
        "control_semantics": semantic_receipt,
        "permutation_runtime_exposure": [
            {
                "recipient_pair_id": recipient,
                "donor_pair_id": donor,
                "donor_target_fingerprint": donor_target,
                "runtime_target_fingerprint": runtime_target,
                "exposure_count": count,
            }
            for (
                recipient,
                donor,
                donor_target,
                runtime_target,
            ), count in sorted(
                permutation_exposure.items()
            )
        ],
        "global_checks": global_checks,
        "engineering_execution_pass": all(global_checks.values()),
        "aggregate_budget": {
            "optimizer_updates": total_updates,
            "decoder_forward_calls": total_calls,
            "decoder_state_evaluations": total_states,
        },
        "deterministic_runtime": deterministic_runtime,
        "interpretation": {
            "engineering_only": True,
            "positive_response_learning_required": False,
            "not_performance_evidence": True,
            "authorizes_formal_800": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
        },
    }


__all__ = [
    "CONTROL_BOUNDED_EXECUTION_SCHEMA",
    "CONTROL_RUNTIME_BINDING_SCHEMA",
    "CONTROL_SEMANTICS_SCHEMA",
    "ControlRuntimeBinding",
    "build_control_runtime_binding",
    "build_control_semantics_receipt",
    "execute_control_bounded_execution",
]
