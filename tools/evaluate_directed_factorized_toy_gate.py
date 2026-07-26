#!/usr/bin/env python3
"""Replay the frozen pre-bounded D-SVEF v5 toy/code gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import stable_fingerprint  # noqa: E402
from cure_lite.config import LossConfig  # noqa: E402
from cure_lite.directed_factorized_decoder import (  # noqa: E402
    CURELiteDirectedFactorizedDecoder,
)
from cure_lite.losses import CURELiteLoss  # noqa: E402
from cure_lite.paired_outcome_losses import (  # noqa: E402
    OutcomeCompleteTransitionLoss,
)
from cure_lite.train.paired_outcome_step import (  # noqa: E402
    outcome_complete_train_step,
)
from cure_lite.train.paired_step import (  # noqa: E402
    _paired_endpoint_logits,
)
from tests.test_factorized_outcome_toy_overfit import (  # noqa: E402
    _TOY_LEARNING_RATE,
    _TOY_UPDATES,
    _subpixel_outcome_toy,
)


SCHEMA_VERSION = "cure-lite-dsvef-v5-toy-gate-result-v1"
FROZEN_SEED = 7817
CASES = (
    ("one_pixel", ((1, 2),)),
    ("two_pixels", ((1, 2), (2, 1))),
    ("three_pixels", ((1, 2), (2, 1), (2, 2))),
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
}


def _case(
    case_id: str,
    clean_pixels: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(FROZEN_SEED)
        outcome, factual = _subpixel_outcome_toy(clean_pixels)
        decoder = CURELiteDirectedFactorizedDecoder(
            feature_channels=8,
            feature_stride=4,
        )
        absolute = CURELiteLoss()
        criterion = OutcomeCompleteTransitionLoss(LossConfig())
        optimizer = torch.optim.Adam(
            decoder.parameters(),
            lr=_TOY_LEARNING_RATE,
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
        }

        for _ in range(_TOY_UPDATES):
            logs = outcome_complete_train_step(
                decoder,
                absolute,
                criterion,
                optimizer,
                factual,
                outcome,
            )

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
            factual_miss_score = torch.sigmoid(
                decoder(
                    factual["factual_miss"].feature,
                    factual["factual_miss"].occupancy,
                )
            )
            factual_no_miss_score = torch.sigmoid(
                decoder(
                    factual["factual_no_miss"].feature,
                    factual["factual_no_miss"].occupancy,
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
        factual_target = factual["factual_miss"].target > 0.5
        factual_background = (
            factual["factual_miss"].valid_mask & ~factual_target
        )
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
                > THRESHOLDS[
                    "factual_miss_target_min_exclusive"
                ]
            ),
            "factual_miss_background": (
                observed["factual_miss_background_max"]
                < THRESHOLDS[
                    "factual_miss_background_max_exclusive"
                ]
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
                endpoint_gradient_contract.values()
            ),
        }
        return {
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
        }


def evaluate() -> dict[str, object]:
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cases = [_case(case_id, pixels) for case_id, pixels in CASES]
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)

    passed = [row["case_id"] for row in cases if row["passed"] is True]
    failed = [row["case_id"] for row in cases if row["passed"] is False]
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": "dsvef_v5",
        "activation": "relu(softplus(raw_evidence)-log(2))",
        "contract": {
            "seed": FROZEN_SEED,
            "optimizer": "adam",
            "updates": _TOY_UPDATES,
            "learning_rate": _TOY_LEARNING_RATE,
            "feature_channels": 8,
            "feature_stride": 4,
            "torch_num_threads": 1,
            "deterministic_algorithms": True,
        },
        "thresholds": dict(THRESHOLDS),
        "cases": cases,
        "passed_cases": passed,
        "failed_cases": failed,
        "passed_case_count": len(passed),
        "failed_case_count": len(failed),
        "all_pass": not failed,
        "decision": (
            "DSVEF_V5_TOY_GATE_PASS"
            if not failed
            else "DSVEF_V5_TOY_GATE_FAIL"
        ),
        "implementation_contract_pass": True,
        "real_D_R_authorized": False,
        "real_D_R_status": "NOT_RUN_BY_TOY_STOP_RULE",
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "detection_performance_evaluated": False,
        "formal_800_authorized": False,
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
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if result["all_pass"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
