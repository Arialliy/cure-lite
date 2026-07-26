#!/usr/bin/env python3
"""Run the frozen pre-bounded CR-LVEC v7 six-case toy gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.config import LossConfig  # noqa: E402
from cure_lite.crossing_factorized_decoder import (  # noqa: E402
    CURELiteCrossingFactorizedDecoder,
    CrossingFactorizedDecoderFields,
    crossing_recoverable_evidence,
)
from cure_lite.losses import CURELiteLoss  # noqa: E402
from cure_lite.paired_outcome_losses import (  # noqa: E402
    OutcomeCompleteTransitionLoss,
)
from cure_lite.paired_outcome_types import (  # noqa: E402
    OutcomePairBatch,
    direct_projected_intervention_footprint,
)
from cure_lite.paired_types import PairBatch  # noqa: E402
from cure_lite.train.paired_outcome_step import (  # noqa: E402
    outcome_complete_train_step,
)
from cure_lite.train.paired_step import (  # noqa: E402
    _paired_endpoint_logits,
)
from cure_lite.train.step import BranchBatch  # noqa: E402
from tests.test_factorized_outcome_toy_overfit import (  # noqa: E402
    _subpixel_outcome_toy,
)


SCHEMA_VERSION = "cure-lite-cr-lvec-v7-toy-gate-result-v1"
METHOD_ID = "cr_lvec_v7"
FROZEN_SEED = 7817
FROZEN_UPDATES = 320
FROZEN_LEARNING_RATE = 0.004
EXPECTED_PARAMETER_TENSORS = 6
EXPECTED_PARAMETER_COUNT = 2593
EXPECTED_PROPOSAL_SHA256 = (
    "fa72f4ef850f72a65003e913db1b1230d7b0b45046faf61950fb1e4ef80d3c4f"
)
EXPECTED_PROPOSAL_FINGERPRINT = (
    "9d291e6ad9ec0869aa0ab0eaebcb219cd62678420375f56af480ba105208dbf2"
)
EXPECTED_CONFIG_SHA256 = (
    "5f0788b5a90b79ede07731489c81834c73f435dc30367e0af7c571bb397d48b5"
)
EXPECTED_CONFIG_FINGERPRINT = (
    "b2fb7984f7ef97dc111109fed0a859424f7914ff4832693102f9fe7bad1e7a46"
)

CASES = (
    (
        "legacy_component_contains_response",
        "legacy_one_pixel",
        ((1, 2),),
    ),
    (
        "legacy_component_contains_response",
        "legacy_two_pixels",
        ((1, 2), (2, 1)),
    ),
    (
        "legacy_component_contains_response",
        "legacy_three_pixels",
        ((1, 2), (2, 1), (2, 2)),
    ),
    (
        "response_outside_removed_component_inside_count_support",
        "support_one_pixel",
        ((1, 6),),
    ),
    (
        "response_outside_removed_component_inside_count_support",
        "support_two_pixels",
        ((1, 6), (2, 5)),
    ),
    (
        "response_outside_removed_component_inside_count_support",
        "support_three_pixels",
        ((1, 6), (2, 5), (2, 6)),
    ),
)

THRESHOLDS = {
    "total_loss_max_exclusive": 0.10,
    "plus_completion_min_exclusive": 0.95,
    "plus_background_max_exclusive": 0.05,
    "factual_miss_target_min_exclusive": 0.95,
    "factual_miss_background_max_exclusive": 0.05,
    "factual_no_miss_max_exclusive": 0.05,
    "clean_D_mean_min_inclusive": 0.80,
    "clean_H_max_abs_max_inclusive": 0.05,
    "clean_G_max_abs_max_inclusive": 0.05,
    "component_H_max_abs_max_inclusive": 0.05,
    "component_G_max_abs_max_inclusive": 0.05,
    "dual_endpoint_gradients_required": True,
    "all_parameter_gradients_finite_nonzero_required": True,
    "identity_exact_required": True,
    "outside_count_change_exact_required": True,
}

_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "continuously_recoverable_log_vacancy_evidence_crossing_v7"
)
_TOY_CONFIG = _PROTOCOL / "toy_config.json"


class _ObservedCrossingFactorizedDecoder(
    CURELiteCrossingFactorizedDecoder
):
    """Collect margins from existing forwards without another network call."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.maximum_observed_absolute_margin = 0.0
        self.observed_forward_fields_calls = 0

    def forward_fields(
        self,
        feature: Tensor,
        occupancy: Tensor,
    ) -> CrossingFactorizedDecoderFields:
        fields = super().forward_fields(feature, occupancy)
        observed = float(
            fields.crossing_margin.detach().abs().max()
        )
        self.maximum_observed_absolute_margin = max(
            self.maximum_observed_absolute_margin,
            observed,
        )
        self.observed_forward_fields_calls += 1
        return fields


def _load_json_object(path: Path, *, name: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be a JSON object")
    return payload


def _verify_fingerprint(
    payload: dict[str, object],
    *,
    field: str,
    expected: object,
    name: str,
) -> str:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    if not isinstance(observed, str):
        raise TypeError(f"{name}.{field} must be a string")
    if observed != stable_fingerprint(unsigned):
        raise RuntimeError(f"{name} fingerprint differs")
    if observed != expected:
        raise RuntimeError(f"{name} is not the frozen receipt")
    return observed


def _load_frozen_config() -> dict[str, object]:
    """Strictly load the config, proposal, and predecessor closure chain."""

    if file_sha256(_TOY_CONFIG) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("v7 toy config file hash differs")
    config = _load_json_object(_TOY_CONFIG, name="v7 toy config")
    _verify_fingerprint(
        config,
        field="config_fingerprint",
        expected=EXPECTED_CONFIG_FINGERPRINT,
        name="v7 toy config",
    )
    if config.get("schema_version") != (
        "cure-lite-cr-lvec-v7-toy-config-v1"
    ):
        raise RuntimeError("v7 toy config schema differs")
    if config.get("method_id") != METHOD_ID:
        raise RuntimeError("v7 toy config method differs")

    proposal_binding = config.get("proposal_binding")
    if not isinstance(proposal_binding, dict):
        raise TypeError("v7 proposal_binding must be an object")
    if proposal_binding != {
        "repo_path": (
            "protocols/IRSTD-1K/"
            "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
            "proposal_receipt.json"
        ),
        "file_sha256": EXPECTED_PROPOSAL_SHA256,
        "proposal_fingerprint": EXPECTED_PROPOSAL_FINGERPRINT,
    }:
        raise RuntimeError("v7 proposal binding differs")
    proposal_path = _ROOT / str(proposal_binding["repo_path"])
    if file_sha256(proposal_path) != EXPECTED_PROPOSAL_SHA256:
        raise RuntimeError("v7 proposal file hash differs")
    proposal = _load_json_object(proposal_path, name="v7 proposal")
    _verify_fingerprint(
        proposal,
        field="proposal_fingerprint",
        expected=EXPECTED_PROPOSAL_FINGERPRINT,
        name="v7 proposal",
    )
    if proposal.get("method_id") != METHOD_ID:
        raise RuntimeError("v7 proposal method differs")
    design_binding = proposal.get("design_document")
    if not isinstance(design_binding, dict):
        raise TypeError("v7 design_document must be an object")
    design_path = _ROOT / str(design_binding.get("repo_path"))
    if file_sha256(design_path) != design_binding.get("file_sha256"):
        raise RuntimeError("v7 design document hash differs")

    predecessor = config.get("predecessor_v6_closure")
    if not isinstance(predecessor, dict):
        raise TypeError("v7 predecessor closure binding must be an object")
    closure_path = _ROOT / str(predecessor.get("repo_path"))
    if file_sha256(closure_path) != predecessor.get("file_sha256"):
        raise RuntimeError("v6 predecessor closure file hash differs")
    closure = _load_json_object(closure_path, name="v6 closure")
    closure_fingerprint = _verify_fingerprint(
        closure,
        field="receipt_fingerprint",
        expected=predecessor.get("receipt_fingerprint"),
        name="v6 closure",
    )
    if closure.get("method_id") != "pr_svef_v6":
        raise RuntimeError("v6 predecessor method differs")
    if closure.get("decision") != (
        "PRSVEF_V6_BOUNDED_MODEL_CODE_GATE_FAIL"
    ):
        raise RuntimeError("v6 predecessor decision differs")
    proposal_predecessor = proposal.get("predecessor_v6")
    if not isinstance(proposal_predecessor, dict):
        raise TypeError("v7 proposal predecessor_v6 must be an object")
    if (
        proposal_predecessor.get("closure_repo_path")
        != predecessor.get("repo_path")
        or proposal_predecessor.get("closure_file_sha256")
        != predecessor.get("file_sha256")
        or proposal_predecessor.get("closure_receipt_fingerprint")
        != closure_fingerprint
    ):
        raise RuntimeError("v7 proposal/config predecessor bindings differ")

    expected_operator = {
        "occupancy_projection": "project_max_to_feature_grid",
        "local_count": "fixed_ones_3x3_convolution",
        "occupancy_burden": "nearest(log1p(local_count))",
        "crossing_margin": "raw_evidence-occupancy_burden",
        "ratio_identity": (
            "expm1(raw-log1p(count))=exp(raw)/(1+count)-1"
        ),
        "forward": (
            "expm1(margin) for margin>0, explicit zero for margin<=0"
        ),
        "backward": (
            "explicit exp(margin) surrogate on the supported numerical axis"
        ),
        "logit_composition": "baseline_logits+crossing_evidence",
        "trainable_parameters_added": 0,
        "nonfinite_policy": "fail_fast_without_clamp",
    }
    if config.get("operator") != expected_operator:
        raise RuntimeError("v7 toy operator differs")
    if config.get("decoder") != {
        "feature_channels": 8,
        "feature_stride": 4,
        "width": 32,
        "groups": 8,
        "trunk_residual_scale": 0.5,
        "baseline_probability": 0.1,
        "vacancy_kernel_size": 3,
        "resize_policy": (
            "bilinear_raw_nearest_burden_then_crossing_v1"
        ),
        "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
    }:
        raise RuntimeError("v7 toy decoder contract differs")
    if config.get("optimization") != {
        "seed": FROZEN_SEED,
        "optimizer": "adam",
        "updates": FROZEN_UPDATES,
        "learning_rate": FROZEN_LEARNING_RATE,
        "weight_decay": 0.0,
        "loss": (
            "unchanged_outcome_complete_transition_plus_absolute_factual"
        ),
        "training_step": "unchanged_outcome_complete_train_step",
        "automatic_retry_allowed": False,
    }:
        raise RuntimeError("v7 toy optimization differs")
    if config.get("thresholds") != THRESHOLDS:
        raise RuntimeError("v7 toy thresholds differ")

    observed_cases: list[
        tuple[str, str, tuple[tuple[int, int], ...]]
    ] = []
    families = config.get("case_families")
    if not isinstance(families, list) or len(families) != 2:
        raise RuntimeError("v7 toy case families differ")
    for family in families:
        if not isinstance(family, dict):
            raise TypeError("v7 toy family must be an object")
        family_id = family.get("family_id")
        rows = family.get("cases")
        if not isinstance(family_id, str) or not isinstance(rows, list):
            raise TypeError("v7 toy family identity/cases are invalid")
        for row in rows:
            if not isinstance(row, dict):
                raise TypeError("v7 toy case must be an object")
            observed_cases.append(
                (
                    family_id,
                    str(row.get("case_id")),
                    tuple(
                        tuple(int(value) for value in pixel)
                        for pixel in row.get("clean_pixels", [])
                    ),
                )
            )
    if tuple(observed_cases) != CASES:
        raise RuntimeError("v7 toy cases differ")
    support_family = families[1]
    if support_family.get("removed_component_pixels") != [[0, 0]]:
        raise RuntimeError("v7 support case removed component differs")
    if (
        support_family.get(
            "require_all_response_pixels_outside_removed_component"
        )
        is not True
        or support_family.get(
            "require_all_response_pixels_outside_direct_projected_change"
        )
        is not True
        or support_family.get(
            "require_all_response_pixels_inside_three_by_three_count_change_support"
        )
        is not True
    ):
        raise RuntimeError("v7 support-case geometry requirements differ")

    decision = config.get("decision_rule")
    if not isinstance(decision, dict):
        raise TypeError("v7 decision_rule must be an object")
    if (
        decision.get("required_passed_case_count") != 6
        or decision.get("required_passed_family_count") != 2
        or decision.get("per_case_all_checks_required") is not True
        or decision.get("mean_cannot_override_case_failure") is not True
        or decision.get("pass_decision") != "CR_LVEC_V7_TOY_GATE_PASS"
        or decision.get("fail_decision") != "CR_LVEC_V7_TOY_GATE_FAIL"
    ):
        raise RuntimeError("v7 decision rule differs")
    boundary = config.get("execution_boundary")
    if not isinstance(boundary, dict) or any(
        value is not False for value in boundary.values()
    ):
        raise RuntimeError("v7 toy execution boundary differs")
    numerical = config.get("numerical_contract")
    if not isinstance(numerical, dict):
        raise TypeError("v7 numerical_contract must be an object")
    if (
        numerical.get("dtype") != "float32"
        or numerical.get("finite_nonzero_negative_recovery_probe")
        != -80.0
        or numerical.get("first_zero_recovery_probe") != -104.0
        or numerical.get("largest_finite_positive_margin_probe") != 88.0
        or numerical.get("first_nonfinite_positive_margin_probe") != 89.0
        or numerical.get(
            "finite_nonzero_gradient_required_at_negative_probe"
        )
        is not True
        or numerical.get("zero_recovery_must_fail_fast") is not True
        or numerical.get(
            "finite_forward_and_gradient_required_at_largest_finite_probe"
        )
        is not True
        or numerical.get(
            "nonfinite_fail_fast_required_at_first_nonfinite_probe"
        )
        is not True
        or numerical.get("silent_clamp_allowed") is not False
        or numerical.get(
            "maximum_observed_absolute_margin_must_be_reported"
        )
        is not True
    ):
        raise RuntimeError("v7 numerical contract differs")
    return config


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _support_mismatch_outcome_toy(
    clean_pixels: tuple[tuple[int, int], ...],
) -> tuple[OutcomePairBatch, dict[str, object]]:
    """Use one-pixel deletions while D remains outside C but in count support."""

    legacy, factual = _subpixel_outcome_toy(clean_pixels)
    support_feature = legacy.pair_batch.feature.clone()
    support_feature[0, 0, 0, 0] = 0.0
    support_feature[0, 0, 0, 1] = 5.0
    occupancy_plus = torch.zeros_like(
        legacy.pair_batch.occupancy_plus
    )
    occupancy_plus[0, 0, 0, 0] = True
    occupancy_plus[1, 0, 7, 7] = True
    occupancy_minus = torch.zeros_like(occupancy_plus)
    pair_batch = PairBatch(
        feature=support_feature,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        label_increment=legacy.pair_batch.label_increment.clone(),
        image_valid_mask=legacy.pair_batch.image_valid_mask.clone(),
        pair_ids=(
            _sha("cr-lvec-support-clean"),
            _sha("cr-lvec-support-component"),
        ),
        sample_ids=(
            "cr-lvec-support-clean-source",
            "cr-lvec-support-component-source",
        ),
        group_ids=(
            "cr-lvec-support-clean-group",
            "cr-lvec-support-component-group",
        ),
        pair_kinds=("clean_positive", "component_null"),
        projection_visible=(True, True),
    )
    outcome = OutcomePairBatch(
        pair_batch=pair_batch,
        completion_plus=legacy.completion_plus.clone(),
        completion_minus=legacy.completion_minus.clone(),
        gt_union=legacy.gt_union.clone(),
        intervention_footprint=direct_projected_intervention_footprint(
            pair_batch
        ),
    )
    factual_miss = factual["factual_miss"]
    adjusted_factual = dict(factual)
    adjusted_factual["factual_miss"] = BranchBatch(
        feature=support_feature[0:1].repeat(4, 1, 1, 1),
        occupancy=factual_miss.occupancy.clone(),
        target=factual_miss.target.clone(),
        valid_mask=factual_miss.valid_mask.clone(),
    )
    return outcome, adjusted_factual


def _all_training_states(
    outcome: OutcomePairBatch,
    factual: dict[str, object],
) -> tuple[Tensor, Tensor]:
    miss = factual["factual_miss"]
    no_miss = factual["factual_no_miss"]
    feature = torch.cat(
        (
            miss.feature,
            no_miss.feature,
            outcome.pair_batch.feature,
            outcome.pair_batch.feature,
        ),
        dim=0,
    )
    occupancy = torch.cat(
        (
            miss.occupancy,
            no_miss.occupancy,
            outcome.pair_batch.occupancy_plus,
            outcome.pair_batch.occupancy_minus,
        ),
        dim=0,
    )
    return feature, occupancy


def _margin_snapshot(
    decoder: CURELiteCrossingFactorizedDecoder,
    outcome: OutcomePairBatch,
    factual: dict[str, object],
) -> dict[str, float | int | bool]:
    feature, occupancy = _all_training_states(outcome, factual)
    with torch.no_grad():
        fields = decoder.forward_fields(feature, occupancy)
        count = F.interpolate(
            fields.local_occupancy_count,
            size=fields.output_size,
            mode="nearest",
        )
        continuation = torch.expm1(fields.crossing_margin)
        raw64 = fields.raw_evidence.to(dtype=torch.float64)
        count64 = count.to(dtype=torch.float64)
        continuation64 = torch.expm1(
            raw64 - torch.log1p(count64)
        )
        ratio64 = (
            torch.exp(raw64)
            / (1.0 + count64)
            - 1.0
        )
        error = (continuation64 - ratio64).abs()
    return {
        "max_abs_margin": float(fields.crossing_margin.abs().max()),
        "minimum_margin": float(fields.crossing_margin.min()),
        "maximum_margin": float(fields.crossing_margin.max()),
        "positive_margin_count": int(
            torch.count_nonzero(fields.crossing_margin > 0.0)
        ),
        "nonpositive_margin_count": int(
            torch.count_nonzero(fields.crossing_margin <= 0.0)
        ),
        "all_fields_finite": bool(
            torch.isfinite(fields.raw_evidence).all()
            and torch.isfinite(fields.occupancy_burden).all()
            and torch.isfinite(fields.crossing_margin).all()
            and torch.isfinite(fields.evidence).all()
            and torch.isfinite(fields.logits).all()
            and torch.isfinite(continuation).all()
            and torch.isfinite(ratio64).all()
        ),
        "ratio_identity_max_abs_error": float(error.max()),
        "ratio_identity_allclose": bool(
            torch.allclose(
                continuation64,
                ratio64,
                rtol=1.0e-12,
                atol=1.0e-12,
            )
        ),
        "forward_crossing_exact": bool(
            torch.equal(
                fields.evidence,
                torch.where(
                    fields.crossing_margin <= 0.0,
                    torch.zeros_like(continuation),
                    continuation,
                ),
            )
        ),
    }


def _pair_operator_audit(
    decoder: CURELiteCrossingFactorizedDecoder,
    outcome: OutcomePairBatch,
) -> dict[str, object]:
    batch = outcome.pair_batch
    with torch.no_grad():
        plus = decoder.forward_fields(batch.feature, batch.occupancy_plus)
        minus = decoder.forward_fields(
            batch.feature,
            batch.occupancy_minus,
        )
        plus_count = F.interpolate(
            plus.local_occupancy_count,
            size=plus.output_size,
            mode="nearest",
        )
        minus_count = F.interpolate(
            minus.local_occupancy_count,
            size=minus.output_size,
            mode="nearest",
        )
        unchanged = plus_count == minus_count
        direct_changed = (
            plus.projected_occupancy ^ minus.projected_occupancy
        )
        direct_changed = F.interpolate(
            direct_changed.to(dtype=torch.float32),
            size=plus.output_size,
            mode="nearest",
        ).to(dtype=torch.bool)
        outside_count = int(torch.count_nonzero(unchanged))
        outside_delta = (
            minus.logits[unchanged] - plus.logits[unchanged]
        ).abs()
        outside_max = (
            0.0 if outside_count == 0 else float(outside_delta.max())
        )
        identity = decoder(
            batch.feature,
            batch.occupancy_plus,
        )
        identity_again = decoder(
            batch.feature,
            batch.occupancy_plus.clone(),
        )
        identity_delta = (identity - identity_again).abs()
        changed_support = ~unchanged
        clean_response = outcome.response_stratum[0:1]
        response_in_support = bool(
            torch.all(~clean_response | changed_support[0:1])
        )
        response_outside_component = bool(
            not torch.any(
                clean_response
                & outcome.removed_component[0:1]
            )
        )
        response_outside_direct_projection = bool(
            not torch.any(clean_response & direct_changed[0:1])
        )

    def endpoint(fields: object) -> dict[str, float | int]:
        return {
            "raw_evidence_min": float(fields.raw_evidence.min()),
            "raw_evidence_max": float(fields.raw_evidence.max()),
            "occupancy_burden_min": float(
                fields.occupancy_burden.min()
            ),
            "occupancy_burden_max": float(
                fields.occupancy_burden.max()
            ),
            "crossing_margin_min": float(
                fields.crossing_margin.min()
            ),
            "crossing_margin_max": float(
                fields.crossing_margin.max()
            ),
            "crossing_evidence_min": float(fields.evidence.min()),
            "crossing_evidence_max": float(fields.evidence.max()),
            "local_count_min": float(
                fields.local_occupancy_count.min()
            ),
            "local_count_max": float(
                fields.local_occupancy_count.max()
            ),
            "positive_margin_count": int(
                torch.count_nonzero(fields.crossing_margin > 0.0)
            ),
        }

    return {
        "plus": endpoint(plus),
        "minus": endpoint(minus),
        "identity_max_abs_logit_delta": float(identity_delta.max()),
        "identity_exact": bool(torch.equal(identity, identity_again)),
        "outside_count_change_pixel_count": outside_count,
        "outside_count_change_check_vacuous": outside_count == 0,
        "outside_count_change_max_abs_logit_delta": outside_max,
        "outside_count_change_exact": bool(
            outside_count == 0
            or torch.equal(
                minus.logits[unchanged],
                plus.logits[unchanged],
            )
        ),
        "clean_response_inside_count_change_support": response_in_support,
        "clean_response_outside_removed_component": (
            response_outside_component
        ),
        "clean_response_outside_direct_projected_xor_lift": (
            response_outside_direct_projection
        ),
    }


def _case_geometry(
    family_id: str,
    outcome: OutcomePairBatch,
    clean_pixels: tuple[tuple[int, int], ...],
    operator: dict[str, object],
) -> dict[str, object]:
    removed = outcome.removed_component
    clean_removed_coordinates = [
        [int(row), int(column)]
        for _, _, row, column in torch.nonzero(
            removed[0:1],
            as_tuple=False,
        ).tolist()
    ]
    component_removed_count = int(torch.count_nonzero(removed[1:2]))
    support_family = (
        family_id
        == "response_outside_removed_component_inside_count_support"
    )
    if support_family:
        passed = (
            clean_removed_coordinates == [[0, 0]]
            and component_removed_count == 1
            and operator[
                "clean_response_outside_removed_component"
            ]
            is True
            and operator[
                "clean_response_outside_direct_projected_xor_lift"
            ]
            is True
            and operator[
                "clean_response_inside_count_change_support"
            ]
            is True
        )
    else:
        passed = (
            len(clean_removed_coordinates) == 16
            and component_removed_count == 16
        )
    return {
        "clean_removed_component_pixels": clean_removed_coordinates,
        "clean_removed_component_pixel_count": len(
            clean_removed_coordinates
        ),
        "component_null_removed_component_pixel_count": (
            component_removed_count
        ),
        "clean_response_pixels": [list(value) for value in clean_pixels],
        "response_outside_removed_component": operator[
            "clean_response_outside_removed_component"
        ],
        "response_inside_count_change_support": operator[
            "clean_response_inside_count_change_support"
        ],
        "response_outside_direct_projected_xor_lift": operator[
            "clean_response_outside_direct_projected_xor_lift"
        ],
        "passed": passed,
    }


def _case(
    family_id: str,
    case_id: str,
    clean_pixels: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FROZEN_SEED)
        if family_id == "legacy_component_contains_response":
            outcome, factual = _subpixel_outcome_toy(clean_pixels)
        else:
            outcome, factual = _support_mismatch_outcome_toy(clean_pixels)
        decoder = _ObservedCrossingFactorizedDecoder(
            feature_channels=8,
            feature_stride=4,
        )
        named_parameters = tuple(decoder.named_parameters())
        if len(named_parameters) != EXPECTED_PARAMETER_TENSORS:
            raise AssertionError("v7 toy parameter tensor count differs")
        if (
            sum(parameter.numel() for _, parameter in named_parameters)
            != EXPECTED_PARAMETER_COUNT
        ):
            raise AssertionError("v7 toy parameter count differs")

        absolute = CURELiteLoss()
        criterion = OutcomeCompleteTransitionLoss(LossConfig())
        optimizer = torch.optim.Adam(
            decoder.parameters(),
            lr=FROZEN_LEARNING_RATE,
            weight_decay=0.0,
        )

        initial_plus, initial_minus = _paired_endpoint_logits(
            decoder,
            feature=outcome.pair_batch.feature,
            occupancy_plus=outcome.pair_batch.occupancy_plus,
            occupancy_minus=outcome.pair_batch.occupancy_minus,
        )
        initial_result = criterion(
            initial_plus,
            initial_minus,
            outcome.completion_plus,
            outcome.pair_batch.occupancy_plus,
            outcome.gt_union,
            outcome.pair_batch.label_increment,
            outcome.pair_batch.image_valid_mask,
            outcome.intervention_footprint,
        )
        plus_gradient, minus_gradient = torch.autograd.grad(
            initial_result["total"],
            (initial_plus, initial_minus),
        )
        endpoint_gradient_contract = {
            "plus_finite": bool(torch.isfinite(plus_gradient).all()),
            "minus_finite": bool(torch.isfinite(minus_gradient).all()),
            "plus_nonzero": bool(torch.count_nonzero(plus_gradient) > 0),
            "minus_nonzero": bool(torch.count_nonzero(minus_gradient) > 0),
            "plus_l2_norm": float(plus_gradient.double().norm()),
            "minus_l2_norm": float(minus_gradient.double().norm()),
        }

        initial_margin = _margin_snapshot(decoder, outcome, factual)
        gradient_snapshots: dict[str, dict[str, object]] = {
            name: {
                "numel": parameter.numel(),
                "first_update": None,
                "last_update": None,
            }
            for name, parameter in named_parameters
        }

        logs: dict[str, float | int] = {}
        for update_index in range(FROZEN_UPDATES):
            logs = outcome_complete_train_step(
                decoder,
                absolute,
                criterion,
                optimizer,
                factual,
                outcome,
            )
            if update_index in {0, FROZEN_UPDATES - 1}:
                snapshot_name = (
                    "first_update"
                    if update_index == 0
                    else "last_update"
                )
                for name, parameter in named_parameters:
                    gradient = parameter.grad
                    if gradient is None:
                        raise RuntimeError(
                            f"parameter {name} did not receive a gradient"
                        )
                    gradient_snapshots[name][snapshot_name] = {
                        "update": update_index + 1,
                        "finite": bool(torch.isfinite(gradient).all()),
                        "nonzero": bool(torch.count_nonzero(gradient) > 0),
                        "l2_norm": float(
                            gradient.detach().double().norm()
                        ),
                    }

        final_margin = _margin_snapshot(decoder, outcome, factual)
        all_snapshot_rows = [
            snapshot
            for row in gradient_snapshots.values()
            for snapshot in (
                row["first_update"],
                row["last_update"],
            )
        ]
        parameter_gradient_contract = {
            "expected_parameter_tensor_count": EXPECTED_PARAMETER_TENSORS,
            "observed_parameter_tensor_count": len(named_parameters),
            "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
            "observed_parameter_count": sum(
                parameter.numel()
                for _, parameter in named_parameters
            ),
            "training_step_enforces_finite_gradient_each_update": True,
            "snapshot_updates": [1, FROZEN_UPDATES],
            "parameters": gradient_snapshots,
            "all_six_finite_at_first_and_last_update": all(
                row is not None and row["finite"] is True
                for row in all_snapshot_rows
            ),
            "all_six_nonzero_at_first_and_last_update": all(
                row is not None and row["nonzero"] is True
                for row in all_snapshot_rows
            ),
        }

        decoder.eval()
        with torch.no_grad():
            logits_plus, logits_minus = _paired_endpoint_logits(
                decoder,
                feature=outcome.pair_batch.feature,
                occupancy_plus=outcome.pair_batch.occupancy_plus,
                occupancy_minus=outcome.pair_batch.occupancy_minus,
            )
            score_plus = torch.sigmoid(logits_plus)
            delta = (
                torch.sigmoid(logits_minus)
                - torch.sigmoid(logits_plus)
            )
            factual_miss = factual["factual_miss"]
            factual_no_miss = factual["factual_no_miss"]
            factual_miss_score = torch.sigmoid(
                decoder(
                    factual_miss.feature,
                    factual_miss.occupancy,
                )
            )
            factual_no_miss_score = torch.sigmoid(
                decoder(
                    factual_no_miss.feature,
                    factual_no_miss.occupancy,
                )
            )

        clean_D = outcome.response_stratum[0]
        clean_H = outcome.local_zero_stratum[0]
        clean_G = outcome.global_zero_stratum[0]
        component_H = outcome.local_zero_stratum[1]
        component_G = outcome.global_zero_stratum[1]
        anchor_background = (
            outcome.pair_batch.image_valid_mask
            & ~outcome.pair_batch.occupancy_plus
            & ~outcome.gt_union
        )
        factual_target = factual_miss.target > 0.5
        factual_background = factual_miss.valid_mask & ~factual_target
        observed = {
            "total_loss": float(logs["total"]),
            "plus_completion_min": float(
                score_plus[outcome.completion_plus].min()
            ),
            "plus_background_max": float(
                score_plus[anchor_background].max()
            ),
            "factual_miss_target_min": float(
                factual_miss_score[factual_target].min()
            ),
            "factual_miss_background_max": float(
                factual_miss_score[factual_background].max()
            ),
            "factual_no_miss_max": float(factual_no_miss_score.max()),
            "clean_D_mean": float(delta[0][clean_D].mean()),
            "clean_H_max_abs": float(delta[0][clean_H].abs().max()),
            "clean_G_max_abs": float(delta[0][clean_G].abs().max()),
            "component_H_max_abs": float(
                delta[1][component_H].abs().max()
            ),
            "component_G_max_abs": float(
                delta[1][component_G].abs().max()
            ),
        }
        operator = _pair_operator_audit(decoder, outcome)
        maximum_observed_margin = (
            decoder.maximum_observed_absolute_margin
        )
        observed_forward_fields_calls = (
            decoder.observed_forward_fields_calls
        )
        geometry = _case_geometry(
            family_id,
            outcome,
            clean_pixels,
            operator,
        )
        checks = {
            "total_loss": (
                observed["total_loss"]
                < THRESHOLDS["total_loss_max_exclusive"]
            ),
            "plus_completion": (
                observed["plus_completion_min"]
                > THRESHOLDS["plus_completion_min_exclusive"]
            ),
            "plus_background": (
                observed["plus_background_max"]
                < THRESHOLDS["plus_background_max_exclusive"]
            ),
            "factual_miss_target": (
                observed["factual_miss_target_min"]
                > THRESHOLDS["factual_miss_target_min_exclusive"]
            ),
            "factual_miss_background": (
                observed["factual_miss_background_max"]
                < THRESHOLDS["factual_miss_background_max_exclusive"]
            ),
            "factual_no_miss": (
                observed["factual_no_miss_max"]
                < THRESHOLDS["factual_no_miss_max_exclusive"]
            ),
            "clean_D": (
                observed["clean_D_mean"]
                >= THRESHOLDS["clean_D_mean_min_inclusive"]
            ),
            "clean_H": (
                observed["clean_H_max_abs"]
                <= THRESHOLDS["clean_H_max_abs_max_inclusive"]
            ),
            "clean_G": (
                observed["clean_G_max_abs"]
                <= THRESHOLDS["clean_G_max_abs_max_inclusive"]
            ),
            "component_H": (
                observed["component_H_max_abs"]
                <= THRESHOLDS[
                    "component_H_max_abs_max_inclusive"
                ]
            ),
            "component_G": (
                observed["component_G_max_abs"]
                <= THRESHOLDS[
                    "component_G_max_abs_max_inclusive"
                ]
            ),
            "dual_endpoint_gradients": all(
                endpoint_gradient_contract[name] is True
                for name in (
                    "plus_finite",
                    "minus_finite",
                    "plus_nonzero",
                    "minus_nonzero",
                )
            ),
            "all_parameter_gradients": (
                parameter_gradient_contract[
                    "all_six_finite_at_first_and_last_update"
                ]
                is True
                and parameter_gradient_contract[
                    "all_six_nonzero_at_first_and_last_update"
                ]
                is True
            ),
            "operator_ratio_and_forward": (
                final_margin["all_fields_finite"] is True
                and final_margin["ratio_identity_allclose"] is True
                and final_margin["forward_crossing_exact"] is True
            ),
            "identity_exact": operator["identity_exact"] is True,
            "outside_count_change_exact": (
                operator["outside_count_change_exact"] is True
            ),
            "case_geometry": geometry["passed"] is True,
            "margin_finite": (
                initial_margin["all_fields_finite"] is True
                and final_margin["all_fields_finite"] is True
                and bool(
                    torch.isfinite(
                        torch.tensor(maximum_observed_margin)
                    )
                )
            ),
        }
        return {
            "family_id": family_id,
            "case_id": case_id,
            "target_pixel_count": len(clean_pixels),
            "clean_pixels": [list(value) for value in clean_pixels],
            "observed": observed,
            "checks": checks,
            "passed": all(checks.values()),
            "failed_checks": sorted(
                name for name, passed in checks.items() if not passed
            ),
            "endpoint_gradient_contract": endpoint_gradient_contract,
            "parameter_gradient_contract": parameter_gradient_contract,
            "margin_contract": {
                "initial": initial_margin,
                "final": final_margin,
                "maximum_observed_absolute_margin": (
                    maximum_observed_margin
                ),
                "maximum_observed_scope": (
                    "all_decoder_forward_fields_calls_during_case_evaluation"
                ),
                "observed_forward_fields_calls": (
                    observed_forward_fields_calls
                ),
                "snapshot_count": 2,
                "snapshot_scope": "initial_and_final_training_states",
            },
            "ratio_operator_fields": operator,
            "geometry_contract": geometry,
        }


def _numerical_probe(config: dict[str, object]) -> dict[str, object]:
    numerical = config["numerical_contract"]
    negative_value = float(
        numerical["finite_nonzero_negative_recovery_probe"]
    )
    zero_recovery_value = float(
        numerical["first_zero_recovery_probe"]
    )
    safe_value = float(
        numerical["largest_finite_positive_margin_probe"]
    )
    failing_value = float(
        numerical["first_nonfinite_positive_margin_probe"]
    )
    safe = torch.tensor(
        safe_value,
        dtype=torch.float32,
        requires_grad=True,
    )
    output = crossing_recoverable_evidence(safe)
    output.backward()
    safe_finite = bool(
        torch.isfinite(output) and torch.isfinite(safe.grad)
    )
    overflow_fail_fast = False
    try:
        crossing_recoverable_evidence(
            torch.tensor(failing_value, dtype=torch.float32)
        )
    except ValueError:
        overflow_fail_fast = True
    negative = torch.tensor(
        negative_value,
        dtype=torch.float32,
        requires_grad=True,
    )
    negative_output = crossing_recoverable_evidence(negative)
    negative_output.backward()
    negative_recovery_pass = bool(
        float(negative_output.detach()) == 0.0
        and negative.grad is not None
        and torch.isfinite(negative.grad)
        and float(negative.grad.detach()) > 0.0
    )
    zero_recovery_fail_fast = False
    try:
        crossing_recoverable_evidence(
            torch.tensor(zero_recovery_value, dtype=torch.float32)
        )
    except ValueError:
        zero_recovery_fail_fast = True
    return {
        "dtype": "float32",
        "finite_nonzero_negative_recovery_probe": negative_value,
        "negative_probe_forward": float(negative_output.detach()),
        "negative_probe_gradient": float(negative.grad.detach()),
        "negative_probe_pass": negative_recovery_pass,
        "first_zero_recovery_probe": zero_recovery_value,
        "first_zero_recovery_probe_failed_fast": (
            zero_recovery_fail_fast
        ),
        "largest_finite_positive_margin_probe": safe_value,
        "largest_finite_forward": float(output.detach()),
        "largest_finite_gradient": float(safe.grad.detach()),
        "largest_finite_probe_pass": safe_finite,
        "first_nonfinite_positive_margin_probe": failing_value,
        "first_nonfinite_probe_failed_fast": overflow_fail_fast,
        "silent_clamp_observed": False,
        "passed": (
            negative_recovery_pass
            and zero_recovery_fail_fast
            and safe_finite
            and overflow_fail_fast
        ),
    }


def _nonvacuous_locality_probe() -> dict[str, object]:
    """Verify exact locality on a grid with a non-empty halo complement."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9271)
        decoder = CURELiteCrossingFactorizedDecoder(
            feature_channels=3,
            feature_stride=4,
        )
        feature = torch.randn(1, 3, 4, 5)
        occupancy_plus = torch.zeros(
            1,
            1,
            16,
            20,
            dtype=torch.bool,
        )
        occupancy_plus[0, 0, 4, 8] = True
        occupancy_minus = torch.zeros_like(occupancy_plus)
        with torch.no_grad():
            plus = decoder.forward_fields(feature, occupancy_plus)
            minus = decoder.forward_fields(feature, occupancy_minus)
            plus_count = F.interpolate(
                plus.local_occupancy_count,
                size=plus.output_size,
                mode="nearest",
            )
            minus_count = F.interpolate(
                minus.local_occupancy_count,
                size=minus.output_size,
                mode="nearest",
            )
            changed = plus_count != minus_count
            unchanged = ~changed
            changed_count = int(torch.count_nonzero(changed))
            unchanged_count = int(torch.count_nonzero(unchanged))
            unchanged_exact = bool(
                unchanged_count > 0
                and torch.equal(
                    plus.logits[unchanged],
                    minus.logits[unchanged],
                )
            )
            unchanged_probability_exact = bool(
                unchanged_count > 0
                and torch.equal(
                    torch.sigmoid(plus.logits)[unchanged],
                    torch.sigmoid(minus.logits)[unchanged],
                )
            )
            deletion_monotone = bool(
                torch.all(minus.logits >= plus.logits)
            )
            finite = bool(
                torch.isfinite(plus.logits).all()
                and torch.isfinite(minus.logits).all()
            )
    return {
        "feature_grid": [4, 5],
        "evaluation_grid": [16, 20],
        "removed_component_pixels": [[4, 8]],
        "changed_count_support_pixel_count": changed_count,
        "outside_count_support_pixel_count": unchanged_count,
        "outside_count_support_nonempty": unchanged_count > 0,
        "outside_count_change_exact": unchanged_exact,
        "outside_count_change_probability_exact": (
            unchanged_probability_exact
        ),
        "deletion_monotone": deletion_monotone,
        "all_fields_finite": finite,
        "passed": (
            changed_count > 0
            and unchanged_count > 0
            and unchanged_exact
            and unchanged_probability_exact
            and deletion_monotone
            and finite
        ),
    }


def evaluate() -> dict[str, object]:
    """Compute the complete toy result without writing any file."""

    config = _load_frozen_config()
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cases = [_case(*case) for case in CASES]
        numerical = _numerical_probe(config)
        locality = _nonvacuous_locality_probe()
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)

    passed = [row["case_id"] for row in cases if row["passed"] is True]
    failed = [row["case_id"] for row in cases if row["passed"] is False]
    families: dict[str, dict[str, object]] = {}
    for family_id in dict.fromkeys(row["family_id"] for row in cases):
        family_cases = [
            row for row in cases if row["family_id"] == family_id
        ]
        family_passed = all(row["passed"] is True for row in family_cases)
        families[family_id] = {
            "case_ids": [row["case_id"] for row in family_cases],
            "passed": family_passed,
        }
    passed_family_count = sum(
        row["passed"] is True for row in families.values()
    )
    all_pass = (
        len(passed) == 6
        and not failed
        and passed_family_count == 2
        and numerical["passed"] is True
        and locality["passed"] is True
    )
    decision_rule = config["decision_rule"]
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "operator": dict(config["operator"]),
        "protocol_binding": {
            "toy_config_repo_path": str(
                _TOY_CONFIG.relative_to(_ROOT)
            ),
            "toy_config_file_sha256": file_sha256(_TOY_CONFIG),
            "toy_config_fingerprint": config["config_fingerprint"],
            "proposal_repo_path": config["proposal_binding"][
                "repo_path"
            ],
            "proposal_file_sha256": config["proposal_binding"][
                "file_sha256"
            ],
            "proposal_fingerprint": config["proposal_binding"][
                "proposal_fingerprint"
            ],
            "predecessor_v6_closure": dict(
                config["predecessor_v6_closure"]
            ),
        },
        "contract": {
            "seed": FROZEN_SEED,
            "optimizer": "adam",
            "updates": FROZEN_UPDATES,
            "learning_rate": FROZEN_LEARNING_RATE,
            "weight_decay": 0.0,
            "loss": config["optimization"]["loss"],
            "training_step": config["optimization"]["training_step"],
            "feature_channels": 8,
            "feature_stride": 4,
            "expected_parameter_tensor_count": (
                EXPECTED_PARAMETER_TENSORS
            ),
            "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
            "torch_num_threads": 1,
            "deterministic_algorithms": True,
        },
        "thresholds": dict(THRESHOLDS),
        "numerical_contract_audit": numerical,
        "nonvacuous_locality_audit": locality,
        "case_families": families,
        "case_level_outside_count_check_vacuous_count": sum(
            row["ratio_operator_fields"][
                "outside_count_change_check_vacuous"
            ]
            is True
            for row in cases
        ),
        "cases": cases,
        "passed_cases": passed,
        "failed_cases": failed,
        "passed_case_count": len(passed),
        "failed_case_count": len(failed),
        "passed_family_count": passed_family_count,
        "all_pass": all_pass,
        "decision": (
            decision_rule["pass_decision"]
            if all_pass
            else decision_rule["fail_decision"]
        ),
        "implementation_contract_pass": all_pass,
        "bounded_code_creation_authorized": False,
        "bounded_code_creation_eligible_after_replay": all_pass,
        "real_D_R_bounded_authorized": False,
        "real_D_R_status": "NOT_RUN_TOY_PHASE",
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "detection_performance_evaluated": False,
        "formal_800_authorized": False,
        "automatic_retry_performed": False,
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


def _write_new(path: Path, payload: dict[str, object]) -> None:
    resolved = path.expanduser().resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with resolved.open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = evaluate()
    if args.output is not None:
        _write_new(args.output, result)
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    if result["all_pass"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
