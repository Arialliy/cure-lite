#!/usr/bin/env python3
"""Benchmark CR-LVEC v7 numerical guards on synthetic tensors only.

The benchmark never loads a dataset, cache, checkpoint, or experiment result.
It compares the production guard with two diagnostic implementations while
keeping the decoder state, inputs, forward equation, and surrogate gradient
identical on the supported numerical domain.

Latency is environment evidence, not a scientific or model-quality gate.
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

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cure_lite.crossing_factorized_decoder as crossing_module  # noqa: E402
from cure_lite.cache.schema import stable_fingerprint  # noqa: E402
from cure_lite.crossing_factorized_config import (  # noqa: E402
    CrossingFactorizedDecoderConfig,
)


SCHEMA_VERSION = "cure-lite-cr-lvec-v7-sync-benchmark-v2"
VARIANT_ORDER = ("current", "unchecked", "async")
SEED = 20260725
BATCH_SIZE = 4
FEATURE_CHANNELS = 64
FEATURE_GRID = (64, 64)
OUTPUT_GRID = (256, 256)
FEATURE_STRIDE = 4
DECODER_CALLS_PER_UPDATE = 3
FORMAL_UPDATES = 32_000
FORMAL_DECODER_CALLS = FORMAL_UPDATES * DECODER_CALLS_PER_UPDATE

CrossingOperator = Callable[[Tensor], Tensor]
_PRODUCTION_OPERATOR = crossing_module.crossing_recoverable_evidence


def _validate_margin(margin: Tensor) -> None:
    if not isinstance(margin, Tensor):
        raise TypeError("crossing_margin must be a tensor")
    if not margin.is_floating_point():
        raise TypeError("crossing_margin must be floating point")


def _unchecked_operator(margin: Tensor) -> Tensor:
    """Reproduce the equation without a numerical guard.

    This is a timing lower bound only. It is not a candidate implementation.
    """

    _validate_margin(margin)
    continuation = torch.expm1(margin)
    recovery = torch.exp(margin)
    forward = torch.where(
        margin <= 0.0,
        torch.zeros_like(continuation),
        continuation,
    )
    return forward.detach() + (recovery - recovery.detach())


def _async_operator(margin: Tensor) -> Tensor:
    """Reproduce the production equation with a CUDA-asynchronous assertion."""

    _validate_margin(margin)
    if not hasattr(torch, "_assert_async"):
        raise RuntimeError("this PyTorch build does not provide torch._assert_async")
    continuation = torch.expm1(margin)
    recovery = torch.exp(margin)
    finite_contract = (
        torch.isfinite(margin)
        & torch.isfinite(continuation)
        & torch.isfinite(recovery)
        & (recovery > 0.0)
    )
    torch._assert_async(
        finite_contract.all(),
        "crossing margin, continuation, and recovery must remain "
        "finite with nonzero recovery",
    )
    forward = torch.where(
        margin <= 0.0,
        torch.zeros_like(continuation),
        continuation,
    )
    return forward.detach() + (recovery - recovery.detach())


OPERATORS: dict[str, CrossingOperator] = {
    "current": _PRODUCTION_OPERATOR,
    "unchecked": _unchecked_operator,
    "async": _async_operator,
}


@contextmanager
def _operator_scope(operator: CrossingOperator) -> Iterator[None]:
    """Temporarily select a diagnostic operator and always restore production."""

    previous = crossing_module.crossing_recoverable_evidence
    crossing_module.crossing_recoverable_evidence = operator
    try:
        yield
    finally:
        crossing_module.crossing_recoverable_evidence = previous


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("benchmark device must be cpu or cuda:0")
    if device.type == "cuda":
        if device.index not in {None, 0}:
            raise ValueError("the optional GPU benchmark is restricted to cuda:0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda:0")
    return torch.device("cpu")


def _positive_integer(value: int, *, name: str, allow_zero: bool) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}")
    return value


def audit_operator_boundaries() -> dict[str, object]:
    """Check supported equality and invalid boundaries on CPU float32.

    Invalid asynchronous assertions are intentionally never launched on CUDA,
    where a failed device assertion would invalidate the process context.
    """

    safe_values = (-80.0, -1.0, 0.0, 1.0e-7, 1.0, 88.0)
    outputs: dict[str, Tensor] = {}
    gradients: dict[str, Tensor] = {}
    for name in VARIANT_ORDER:
        margin = torch.tensor(
            safe_values,
            dtype=torch.float32,
            requires_grad=True,
        )
        output = OPERATORS[name](margin)
        output.sum().backward()
        if margin.grad is None:
            raise RuntimeError(f"{name} did not produce a margin gradient")
        outputs[name] = output.detach().clone()
        gradients[name] = margin.grad.detach().clone()

    reference_output = outputs["current"]
    reference_gradient = gradients["current"]
    forward_equal = {
        name: torch.equal(value, reference_output)
        for name, value in outputs.items()
    }
    gradient_equal = {
        name: torch.equal(value, reference_gradient)
        for name, value in gradients.items()
    }

    invalid_values: dict[str, float] = {
        "zero_recovery": -104.0,
        "nonfinite_positive": 89.0,
        "nan": float("nan"),
        "positive_inf": float("inf"),
    }
    rejected: dict[str, dict[str, bool]] = {}
    exception_types: dict[str, dict[str, str | None]] = {}
    for name in VARIANT_ORDER:
        rejected[name] = {}
        exception_types[name] = {}
        for boundary, value in invalid_values.items():
            try:
                OPERATORS[name](torch.tensor(value, dtype=torch.float32))
            except (TypeError, ValueError, RuntimeError) as error:
                rejected[name][boundary] = True
                exception_types[name][boundary] = type(error).__name__
            else:
                rejected[name][boundary] = False
                exception_types[name][boundary] = None

    expected_gradient = torch.exp(
        torch.tensor(safe_values, dtype=torch.float32)
    )
    return {
        "device": "cpu",
        "dtype": "float32",
        "safe_values": list(safe_values),
        "forward_bit_exact_to_current": forward_equal,
        "gradient_bit_exact_to_current": gradient_equal,
        "gradient_equals_exp_exact": {
            name: torch.equal(value, expected_gradient)
            for name, value in gradients.items()
        },
        "invalid_values": {
            "zero_recovery": -104.0,
            "nonfinite_positive": 89.0,
            "nan": "nan",
            "positive_inf": "+inf",
        },
        "invalid_rejected": rejected,
        "invalid_exception_types": exception_types,
        "current_contract_preserved": all(
            rejected["current"].values()
        ),
        "async_contract_preserved_on_cpu": all(
            rejected["async"].values()
        ),
        "unchecked_is_diagnostic_only": True,
        "invalid_async_cuda_probe_executed": False,
    }


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


def _decoder_outputs(
    decoder: crossing_module.CURELiteCrossingFactorizedDecoder,
    features: tuple[Tensor, ...],
    occupancies: tuple[Tensor, ...],
    *,
    operator: CrossingOperator,
) -> tuple[Tensor, ...]:
    with _operator_scope(operator), torch.no_grad():
        outputs = tuple(
            decoder(feature, occupancy)
            for feature, occupancy in zip(
                features,
                occupancies,
                strict=True,
            )
        )
    _synchronize(features[0].device)
    return tuple(value.detach().clone() for value in outputs)


def _decoder_gradients(
    decoder: crossing_module.CURELiteCrossingFactorizedDecoder,
    features: tuple[Tensor, ...],
    occupancies: tuple[Tensor, ...],
    *,
    operator: CrossingOperator,
) -> tuple[Tensor, ...]:
    decoder.zero_grad(set_to_none=True)
    with _operator_scope(operator):
        outputs = tuple(
            decoder(feature, occupancy)
            for feature, occupancy in zip(
                features,
                occupancies,
                strict=True,
            )
        )
        objective = sum(output.square().mean() for output in outputs)
        objective.backward()
    _synchronize(features[0].device)
    gradients: list[Tensor] = []
    for parameter in decoder.parameters():
        if parameter.grad is None:
            raise RuntimeError("every decoder parameter must receive a gradient")
        gradients.append(parameter.grad.detach().clone())
    return tuple(gradients)


def _equivalence_audit(
    decoder: crossing_module.CURELiteCrossingFactorizedDecoder,
    features: tuple[Tensor, ...],
    occupancies: tuple[Tensor, ...],
) -> dict[str, object]:
    outputs = {
        name: _decoder_outputs(
            decoder,
            features,
            occupancies,
            operator=OPERATORS[name],
        )
        for name in VARIANT_ORDER
    }
    gradients = {
        name: _decoder_gradients(
            decoder,
            features,
            occupancies,
            operator=OPERATORS[name],
        )
        for name in VARIANT_ORDER
    }
    reference_outputs = outputs["current"]
    reference_gradients = gradients["current"]
    return {
        "decoder_forward_bit_exact_to_current": {
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
        "decoder_parameter_gradient_bit_exact_to_current": {
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
        "parameter_tensor_count": len(reference_gradients),
        "parameter_count": sum(
            parameter.numel() for parameter in decoder.parameters()
        ),
    }


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
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        samples.append(elapsed_ms / iterations)
    return samples


def _summarize(samples: list[float]) -> dict[str, object]:
    if not samples or any(not isfinite(value) or value <= 0.0 for value in samples):
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
    decoder: crossing_module.CURELiteCrossingFactorizedDecoder,
    features: tuple[Tensor, ...],
    occupancies: tuple[Tensor, ...],
    *,
    operator: CrossingOperator,
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

    def forward_backward_action() -> None:
        decoder.zero_grad(set_to_none=True)
        outputs = tuple(
            decoder(feature, occupancy)
            for feature, occupancy in zip(
                features,
                occupancies,
                strict=True,
            )
        )
        sum(output.square().mean() for output in outputs).backward()

    with _operator_scope(operator):
        forward = _timed_samples(
            forward_action,
            device=device,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        forward_backward = _timed_samples(
            forward_backward_action,
            device=device,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
    return {
        "forward": _summarize(forward),
        "forward_backward": _summarize(forward_backward),
    }


def _run_benchmark_impl(
    *,
    device: str | torch.device = "cpu",
    warmup: int = 2,
    iterations: int = 5,
    repeats: int = 3,
) -> dict[str, object]:
    """Run the fixed-shape benchmark inside a deterministic runtime."""

    resolved = _resolve_device(device)
    warmup = _positive_integer(warmup, name="warmup", allow_zero=True)
    iterations = _positive_integer(
        iterations,
        name="iterations",
        allow_zero=False,
    )
    repeats = _positive_integer(repeats, name="repeats", allow_zero=False)

    config = CrossingFactorizedDecoderConfig(
        feature_channels=FEATURE_CHANNELS,
        feature_stride=FEATURE_STRIDE,
    )
    with torch.random.fork_rng(
        devices=[resolved] if resolved.type == "cuda" else [],
    ):
        torch.manual_seed(SEED)
        if resolved.type == "cuda":
            torch.cuda.manual_seed(SEED)
        decoder = (
            crossing_module.CURELiteCrossingFactorizedDecoder(config)
            .to(resolved)
            .eval()
        )
    features, occupancies = _synthetic_inputs(resolved)

    boundary = audit_operator_boundaries()
    equivalence = _equivalence_audit(
        decoder,
        features,
        occupancies,
    )
    variants = {
        name: _benchmark_variant(
            decoder,
            features,
            occupancies,
            operator=OPERATORS[name],
            device=resolved,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        for name in VARIANT_ORDER
    }
    current_call = float(
        variants["current"]["forward"]["median_ms_per_decoder_call"]
    )
    unchecked_call = float(
        variants["unchecked"]["forward"]["median_ms_per_decoder_call"]
    )
    async_call = float(
        variants["async"]["forward"]["median_ms_per_decoder_call"]
    )

    if crossing_module.crossing_recoverable_evidence is not _PRODUCTION_OPERATOR:
        raise RuntimeError("benchmark did not restore the production operator")
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "synthetic_tensors_only": True,
            "dataset_or_cache_loaded": False,
            "D_R_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "detection_performance_evaluated": False,
            "latency_is_not_a_scientific_gate": True,
            "production_decoder_modified": False,
        },
        "contract": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "feature_channels": FEATURE_CHANNELS,
            "feature_grid": list(FEATURE_GRID),
            "output_grid": list(OUTPUT_GRID),
            "feature_stride": FEATURE_STRIDE,
            "dtype": "float32",
            "decoder_calls_per_update": DECODER_CALLS_PER_UPDATE,
            "formal_updates_for_projection": FORMAL_UPDATES,
            "formal_decoder_calls_for_projection": FORMAL_DECODER_CALLS,
            "variant_order": list(VARIANT_ORDER),
            "current": "production_python_all_fail_fast",
            "unchecked": "diagnostic_lower_bound_not_a_candidate",
            "async": "torch_assert_async_same_contract",
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
            "torch_num_threads": torch.get_num_threads(),
            "torch_assert_async_available": hasattr(torch, "_assert_async"),
            "deterministic_algorithms": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "timing_method": (
                "perf_counter_with_cuda_boundary_synchronization"
                if resolved.type == "cuda"
                else "perf_counter"
            ),
        },
        "operator_boundary_audit": boundary,
        "full_decoder_equivalence": equivalence,
        "variants": variants,
        "projections": {
            "current_minus_unchecked_ms_per_decoder_call": (
                current_call - unchecked_call
            ),
            "current_minus_async_ms_per_decoder_call": (
                current_call - async_call
            ),
            "current_minus_unchecked_seconds_for_96000_calls": (
                (current_call - unchecked_call)
                * FORMAL_DECODER_CALLS
                / 1000.0
            ),
            "current_minus_async_seconds_for_96000_calls": (
                (current_call - async_call)
                * FORMAL_DECODER_CALLS
                / 1000.0
            ),
            "interpretation": (
                "environment_measurement_only_no_authorization_or_gate"
            ),
        },
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


def run_benchmark(
    *,
    device: str | torch.device = "cpu",
    warmup: int = 2,
    iterations: int = 5,
    repeats: int = 3,
) -> dict[str, object]:
    """Run the benchmark and restore every global runtime flag."""

    previous = {
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return _run_benchmark_impl(
            device=device,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
    finally:
        torch.use_deterministic_algorithms(
            previous["deterministic_algorithms"]
        )
        torch.backends.cudnn.benchmark = previous["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = previous[
            "cudnn_deterministic"
        ]
        torch.backends.cuda.matmul.allow_tf32 = previous[
            "matmul_allow_tf32"
        ]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_allow_tf32"]


def _write_new(path: Path, payload: dict[str, object]) -> None:
    resolved = path.expanduser().resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    with resolved.open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu or cuda:0; no other CUDA device is accepted",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional create-only JSON output; stdout is used when omitted",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    else:
        _write_new(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
