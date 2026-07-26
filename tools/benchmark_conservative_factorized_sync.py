#!/usr/bin/env python3
"""Measure CC-SEA v8 fail-fast synchronization on synthetic tensors only.

The unchecked variant is a diagnostic lower bound, never a candidate model.
It is used solely to quantify the cost of the production numerical checks.
No dataset, cache tensor, checkpoint, or experiment result is loaded.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from math import isfinite
import os
from pathlib import Path
from statistics import median
import sys
import time
from typing import Callable, Iterator, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor
from torch.utils._python_dispatch import TorchDispatchMode

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cure_lite.conservative_factorized_decoder as conservative_module  # noqa: E402
from cure_lite.cache.schema import stable_fingerprint  # noqa: E402
from cure_lite.conservative_factorized_config import (  # noqa: E402
    ConservativeFactorizedDecoderConfig,
)


SCHEMA_VERSION = "cure-lite-cc-sea-v8-sync-benchmark-v1"
SEED = 20260725
BATCH_SIZE = 4
FEATURE_CHANNELS = 64
FEATURE_GRID = (64, 64)
OUTPUT_GRID = (256, 256)
FEATURE_STRIDE = 4
DECODER_CALLS_PER_UPDATE = 3
BOUNDED_CALLS = 400 * DECODER_CALLS_PER_UPDATE
FORMAL_CALLS = 32_000 * DECODER_CALLS_PER_UPDATE

CoverageOperator = Callable[
    [Tensor, Tensor],
    tuple[Tensor, Tensor, Tensor, Tensor, Tensor],
]
_PRODUCTION_COVERAGE = (
    conservative_module.coverage_conserving_phase_evidence
)


def _unchecked_coverage(
    raw_phase_evidence: Tensor,
    occupancy_burden: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Reproduce the exact equation without any numerical checks."""

    common_mode = raw_phase_evidence.mean(dim=1, keepdim=True)
    phase_contrast = raw_phase_evidence - common_mode
    budget_margin = common_mode - occupancy_burden
    continuation = torch.expm1(budget_margin)
    recovery = torch.exp(budget_margin)
    forward = torch.where(
        budget_margin <= 0.0,
        torch.zeros_like(continuation),
        continuation,
    )
    evidence_budget = (
        forward.detach() + (recovery - recovery.detach())
    )
    phase_allocation = torch.softmax(phase_contrast, dim=1)
    allocated = phase_allocation * evidence_budget
    return (
        common_mode,
        budget_margin,
        evidence_budget,
        phase_allocation,
        allocated,
    )


VARIANTS: dict[str, CoverageOperator] = {
    "production": _PRODUCTION_COVERAGE,
    "unchecked_diagnostic": _unchecked_coverage,
}


class _LocalScalarCounter(TorchDispatchMode):
    """Count tensor-to-Python scalar extraction operations."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def __torch_dispatch__(
        self,
        func: object,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if "local_scalar_dense" in str(func):
            self.count += 1
        return func(*args, **(kwargs or {}))  # type: ignore[operator]


@contextmanager
def _coverage_scope(operator: CoverageOperator) -> Iterator[None]:
    previous = conservative_module.coverage_conserving_phase_evidence
    conservative_module.coverage_conserving_phase_evidence = operator
    try:
        yield
    finally:
        conservative_module.coverage_conserving_phase_evidence = previous


def _resolve_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type == "cpu":
        return device
    if device.type != "cuda" or device.index not in {None, 0}:
        raise ValueError("benchmark device must be cpu or cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda:0")


def _positive_integer(
    value: int,
    *,
    name: str,
    allow_zero: bool,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{name} is below its minimum")
    return value


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _synthetic_inputs(
    device: torch.device,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    features = tuple(
        torch.randn(
            BATCH_SIZE,
            FEATURE_CHANNELS,
            *FEATURE_GRID,
            dtype=torch.float32,
            generator=generator,
        ).to(device)
        for _ in range(DECODER_CALLS_PER_UPDATE)
    )
    occupancies = tuple(
        (
            torch.rand(
                BATCH_SIZE,
                1,
                *OUTPUT_GRID,
                dtype=torch.float32,
                generator=generator,
            )
            < 0.01
        ).to(device)
        for _ in range(DECODER_CALLS_PER_UPDATE)
    )
    return features, occupancies


def _decoder(
    device: torch.device,
) -> conservative_module.CURELiteConservativeFactorizedDecoder:
    config = ConservativeFactorizedDecoderConfig(
        feature_channels=FEATURE_CHANNELS,
        feature_stride=FEATURE_STRIDE,
    )
    with torch.random.fork_rng(
        devices=[device] if device.type == "cuda" else [],
    ):
        torch.manual_seed(SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed(SEED)
        return (
            conservative_module.CURELiteConservativeFactorizedDecoder(
                config
            )
            .to(device)
            .eval()
        )


def audit_numerical_boundaries() -> dict[str, object]:
    """Prove that only the production variant enforces fail-fast checks."""

    raw = torch.zeros(1, 16, 2, 2, dtype=torch.float32)
    burden = torch.zeros(1, 1, 2, 2, dtype=torch.float32)
    safe_outputs = {
        name: operator(raw, burden)
        for name, operator in VARIANTS.items()
    }
    safe_forward_exact = all(
        torch.equal(observed, expected)
        for observed, expected in zip(
            safe_outputs["unchecked_diagnostic"],
            safe_outputs["production"],
            strict=True,
        )
    )

    rejected: dict[str, bool] = {}
    probes = {
        "raw_nan": (raw.clone(), burden.clone()),
        "negative_burden": (raw.clone(), burden.clone()),
        "zero_recovery": (raw.clone(), burden.clone()),
    }
    probes["raw_nan"][0][0, 0, 0, 0] = float("nan")
    probes["negative_burden"][1][0, 0, 0, 0] = -1.0
    probes["zero_recovery"][0].fill_(-104.0)
    for name, (probe_raw, probe_burden) in probes.items():
        try:
            _PRODUCTION_COVERAGE(probe_raw, probe_burden)
        except (TypeError, ValueError, RuntimeError):
            rejected[name] = True
        else:
            rejected[name] = False
    return {
        "safe_forward_bit_exact": safe_forward_exact,
        "production_invalid_rejected": rejected,
        "all_required_invalid_rejected": all(rejected.values()),
        "unchecked_is_diagnostic_only": True,
    }


def _outputs(
    decoder: conservative_module.CURELiteConservativeFactorizedDecoder,
    features: tuple[Tensor, ...],
    occupancies: tuple[Tensor, ...],
    *,
    operator: CoverageOperator,
) -> tuple[Tensor, ...]:
    with _coverage_scope(operator), torch.no_grad():
        result = tuple(
            decoder(feature, occupancy)
            for feature, occupancy in zip(
                features,
                occupancies,
                strict=True,
            )
        )
    _synchronize(features[0].device)
    return tuple(value.detach().clone() for value in result)


def _gradients(
    decoder: conservative_module.CURELiteConservativeFactorizedDecoder,
    features: tuple[Tensor, ...],
    occupancies: tuple[Tensor, ...],
    *,
    operator: CoverageOperator,
) -> tuple[Tensor, ...]:
    decoder.zero_grad(set_to_none=True)
    with _coverage_scope(operator):
        outputs = tuple(
            decoder(feature, occupancy)
            for feature, occupancy in zip(
                features,
                occupancies,
                strict=True,
            )
        )
        sum(value.square().mean() for value in outputs).backward()
    _synchronize(features[0].device)
    result: list[Tensor] = []
    for parameter in decoder.parameters():
        if parameter.grad is None:
            raise RuntimeError("every parameter must receive a gradient")
        result.append(parameter.grad.detach().clone())
    return tuple(result)


def _equivalence(
    decoder: conservative_module.CURELiteConservativeFactorizedDecoder,
    features: tuple[Tensor, ...],
    occupancies: tuple[Tensor, ...],
) -> dict[str, object]:
    outputs = {
        name: _outputs(
            decoder,
            features,
            occupancies,
            operator=operator,
        )
        for name, operator in VARIANTS.items()
    }
    gradients = {
        name: _gradients(
            decoder,
            features,
            occupancies,
            operator=operator,
        )
        for name, operator in VARIANTS.items()
    }
    reference_outputs = outputs["production"]
    reference_gradients = gradients["production"]
    return {
        "decoder_output_bit_exact_to_production": {
            name: all(
                torch.equal(observed, expected)
                for observed, expected in zip(
                    values,
                    reference_outputs,
                    strict=True,
                )
            )
            for name, values in outputs.items()
        },
        "parameter_gradient_bit_exact_to_production": {
            name: all(
                torch.equal(observed, expected)
                for observed, expected in zip(
                    values,
                    reference_gradients,
                    strict=True,
                )
            )
            for name, values in gradients.items()
        },
        "parameter_count": sum(
            parameter.numel() for parameter in decoder.parameters()
        ),
        "parameter_tensor_count": len(reference_gradients),
    }


def _local_scalar_count(
    decoder: conservative_module.CURELiteConservativeFactorizedDecoder,
    feature: Tensor,
    occupancy: Tensor,
    *,
    operator: CoverageOperator,
) -> int:
    counter = _LocalScalarCounter()
    with counter, _coverage_scope(operator), torch.no_grad():
        decoder(feature, occupancy)
    _synchronize(feature.device)
    return counter.count


def _timed_samples(
    action: Callable[[], None],
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmup):
        action()
    _synchronize(device)
    samples: list[float] = []
    for _ in range(repeats):
        _synchronize(device)
        started = time.perf_counter_ns()
        for _ in range(iterations):
            action()
        _synchronize(device)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000.0
        samples.append(elapsed / iterations)
    return samples


def _summary(samples: list[float]) -> dict[str, object]:
    if not samples or any(
        not isfinite(value) or value <= 0.0 for value in samples
    ):
        raise RuntimeError("benchmark produced an invalid timing sample")
    return {
        "samples_ms_per_update": samples,
        "median_ms_per_update": median(samples),
        "minimum_ms_per_update": min(samples),
        "maximum_ms_per_update": max(samples),
        "median_ms_per_decoder_call": (
            median(samples) / DECODER_CALLS_PER_UPDATE
        ),
    }


def _benchmark_variant(
    decoder: conservative_module.CURELiteConservativeFactorizedDecoder,
    features: tuple[Tensor, ...],
    occupancies: tuple[Tensor, ...],
    *,
    operator: CoverageOperator,
    device: torch.device,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, object]:
    def forward_action() -> None:
        with torch.no_grad():
            for feature, occupancy in zip(
                features,
                occupancies,
                strict=True,
            ):
                decoder(feature, occupancy)

    def training_action() -> None:
        decoder.zero_grad(set_to_none=True)
        outputs = tuple(
            decoder(feature, occupancy)
            for feature, occupancy in zip(
                features,
                occupancies,
                strict=True,
            )
        )
        sum(value.square().mean() for value in outputs).backward()

    with _coverage_scope(operator):
        forward = _timed_samples(
            forward_action,
            device=device,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        training = _timed_samples(
            training_action,
            device=device,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
    return {
        "forward": _summary(forward),
        "forward_backward": _summary(training),
    }


def run_benchmark(
    *,
    device: str | torch.device = "cpu",
    warmup: int = 2,
    iterations: int = 5,
    repeats: int = 3,
) -> dict[str, object]:
    """Run the deterministic synthetic audit and restore global state."""

    resolved = _resolve_device(device)
    warmup = _positive_integer(warmup, name="warmup", allow_zero=True)
    iterations = _positive_integer(
        iterations,
        name="iterations",
        allow_zero=False,
    )
    repeats = _positive_integer(
        repeats,
        name="repeats",
        allow_zero=False,
    )
    previous = {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        decoder = _decoder(resolved)
        features, occupancies = _synthetic_inputs(resolved)
        boundary = audit_numerical_boundaries()
        equivalence = _equivalence(
            decoder,
            features,
            occupancies,
        )
        local_scalars = {
            name: _local_scalar_count(
                decoder,
                features[0],
                occupancies[0],
                operator=operator,
            )
            for name, operator in VARIANTS.items()
        }
        variants = {
            name: _benchmark_variant(
                decoder,
                features,
                occupancies,
                operator=operator,
                device=resolved,
                warmup=warmup,
                iterations=iterations,
                repeats=repeats,
            )
            for name, operator in VARIANTS.items()
        }
        if (
            conservative_module.coverage_conserving_phase_evidence
            is not _PRODUCTION_COVERAGE
        ):
            raise RuntimeError("production operator was not restored")

        production_ms = float(
            variants["production"]["forward"][
                "median_ms_per_decoder_call"
            ]
        )
        unchecked_ms = float(
            variants["unchecked_diagnostic"]["forward"][
                "median_ms_per_decoder_call"
            ]
        )
        delta = production_ms - unchecked_ms
        result: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "method_id": "cc_sea_v8",
            "scope": {
                "synthetic_tensors_only": True,
                "dataset_or_cache_payload_loaded": False,
                "D_R_accessed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
                "detection_performance_evaluated": False,
                "production_decoder_modified": False,
                "latency_is_engineering_evidence_not_scientific_gate": True,
            },
            "contract": {
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "feature_channels": FEATURE_CHANNELS,
                "feature_grid": list(FEATURE_GRID),
                "output_grid": list(OUTPUT_GRID),
                "feature_stride": FEATURE_STRIDE,
                "decoder_calls_per_update": DECODER_CALLS_PER_UPDATE,
                "bounded_decoder_calls": BOUNDED_CALLS,
                "formal_decoder_calls": FORMAL_CALLS,
                "warmup_updates": warmup,
                "iterations_per_repeat": iterations,
                "repeats": repeats,
                "hard_latency_threshold": None,
            },
            "environment": {
                "python_version": sys.version.split()[0],
                "torch_version": torch.__version__,
                "cuda_build_version": torch.version.cuda,
                "device": str(resolved),
                "device_name": (
                    torch.cuda.get_device_name(resolved)
                    if resolved.type == "cuda"
                    else None
                ),
                "deterministic_algorithms": (
                    torch.are_deterministic_algorithms_enabled()
                ),
                "cudnn_deterministic": (
                    torch.backends.cudnn.deterministic
                ),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "timing_method": (
                    "perf_counter_with_cuda_boundary_synchronization"
                    if resolved.type == "cuda"
                    else "perf_counter"
                ),
            },
            "numerical_boundary_audit": boundary,
            "full_decoder_equivalence": equivalence,
            "local_scalar_dense_calls_per_decoder_forward": (
                local_scalars
            ),
            "variants": variants,
            "projections": {
                "production_minus_unchecked_ms_per_decoder_call": delta,
                "production_minus_unchecked_seconds_bounded_1200_calls": (
                    delta * BOUNDED_CALLS / 1000.0
                ),
                "production_minus_unchecked_seconds_formal_96000_calls": (
                    delta * FORMAL_CALLS / 1000.0
                ),
                "interpretation": (
                    "environment_measurement_only_no_model_or_run_"
                    "authorization"
                ),
            },
        }
        result["result_fingerprint"] = stable_fingerprint(result)
        return result
    finally:
        torch.use_deterministic_algorithms(previous["deterministic"])
        torch.backends.cudnn.deterministic = previous[
            "cudnn_deterministic"
        ]
        torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]
        torch.backends.cuda.matmul.allow_tf32 = previous["matmul_tf32"]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_tf32"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser


def _write_new(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_benchmark(
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    if args.output is None:
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        _write_new(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
