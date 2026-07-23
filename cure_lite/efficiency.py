"""Incremental decoder efficiency measurements for CURE-Lite.

The deterministic quantities in this module (parameter counts and Conv2d
MACs/FLOPs) are architecture/input-shape facts.  Latency and allocator memory
are execution-environment measurements and must not be used as scientific
gate metrics.  The formal Stage-A receipt keeps those two classes of evidence
separate; this module intentionally only provides the low-level measurement.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from math import ceil
import time

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class EfficiencyReport:
    parameters: int
    trainable_parameters: int
    multiply_accumulates: int
    flops: int
    median_latency_ms: float
    p95_latency_ms: float
    peak_allocated_bytes: int | None
    peak_incremental_allocated_bytes: int | None
    warmup: int
    repetitions: int
    device: str
    timing_method: str

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


def parameter_counts(module: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return total, trainable


def conv2d_macs(module: nn.Module, feature: Tensor, occupancy: Tensor) -> int:
    """Count Conv2d MACs for one decoder forward.

    One multiply followed by one accumulation is one MAC and two FLOPs.  Bias
    adds, normalization, activation, concatenation, occupancy projection and
    interpolation are deliberately excluded.  This narrow convention is
    stable and is stated verbatim in the formal receipt.
    """

    macs = 0
    handles = []

    def hook(layer: nn.Conv2d, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal macs
        if not isinstance(output, Tensor) or output.ndim != 4:
            raise RuntimeError("Conv2d hook received an unexpected output")
        batch, out_channels, height, width = output.shape
        kernel_height, kernel_width = layer.kernel_size
        per_output = (layer.in_channels // layer.groups) * kernel_height * kernel_width
        macs += int(batch * out_channels * height * width * per_output)

    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            handles.append(child.register_forward_hook(hook))
    try:
        with torch.no_grad():
            module(feature, occupancy)
    finally:
        for handle in handles:
            handle.remove()
    return macs


def measure_decoder_efficiency(
    decoder: nn.Module,
    feature: Tensor,
    occupancy: Tensor,
    *,
    warmup: int = 10,
    repetitions: int = 50,
) -> EfficiencyReport:
    if (
        isinstance(warmup, bool)
        or not isinstance(warmup, int)
        or isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or warmup < 0
        or repetitions < 1
    ):
        raise ValueError("warmup must be non-negative and repetitions must be positive")
    if feature.device != occupancy.device:
        raise ValueError("feature and occupancy must use the same device")
    device = feature.device
    parameters = tuple(decoder.parameters())
    if not parameters:
        raise ValueError("decoder must contain parameters")
    parameter_devices = {parameter.device for parameter in parameters}
    if parameter_devices != {device}:
        raise ValueError("decoder and benchmark inputs must use the same device")
    parameter_dtypes = {parameter.dtype for parameter in parameters}
    if len(parameter_dtypes) != 1 or feature.dtype not in parameter_dtypes:
        raise TypeError("feature dtype must match the decoder parameter dtype")
    prior_mode = decoder.training
    decoder.eval()
    total_parameters, trainable_parameters = parameter_counts(decoder)
    try:
        device_context = (
            torch.cuda.device(device) if device.type == "cuda" else nullcontext()
        )
        with device_context:
            macs = conv2d_macs(decoder, feature, occupancy)

            def synchronize() -> None:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)

            with torch.no_grad():
                for _ in range(warmup):
                    output = decoder(feature, occupancy)
                    del output
                synchronize()

                baseline_allocated: int | None = None
                if device.type == "cuda":
                    baseline_allocated = int(torch.cuda.memory_allocated(device))
                    torch.cuda.reset_peak_memory_stats(device)

                samples: list[float] = []
                if device.type == "cuda":
                    timing_method = "torch_cuda_event_elapsed_time"
                    stream = torch.cuda.current_stream(device)
                    for _ in range(repetitions):
                        started = torch.cuda.Event(enable_timing=True)
                        ended = torch.cuda.Event(enable_timing=True)
                        started.record(stream)
                        output = decoder(feature, occupancy)
                        ended.record(stream)
                        ended.synchronize()
                        samples.append(float(started.elapsed_time(ended)))
                        del output
                else:
                    timing_method = "perf_counter_ns"
                    for _ in range(repetitions):
                        started_ns = time.perf_counter_ns()
                        output = decoder(feature, occupancy)
                        samples.append(
                            (time.perf_counter_ns() - started_ns) / 1_000_000.0
                        )
                        del output
                synchronize()

                if device.type == "cuda":
                    assert baseline_allocated is not None
                    peak_allocated: int | None = int(
                        torch.cuda.max_memory_allocated(device)
                    )
                    peak_incremental: int | None = max(
                        0, peak_allocated - baseline_allocated
                    )
                else:
                    # CPU execution has no CUDA-VRAM measurement.  ``None`` is
                    # semantically different from a measured zero-byte peak.
                    peak_allocated = None
                    peak_incremental = None
    finally:
        decoder.train(prior_mode)

    ordered = sorted(samples)
    midpoint = len(ordered) // 2
    median_latency_ms = (
        ordered[midpoint]
        if len(ordered) % 2 == 1
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    )
    # Nearest-rank empirical p95 (one-indexed rank ceil(0.95*n)).
    p95_index = min(len(ordered) - 1, max(0, ceil(0.95 * len(ordered)) - 1))
    return EfficiencyReport(
        parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        multiply_accumulates=macs,
        flops=2 * macs,
        median_latency_ms=float(median_latency_ms),
        p95_latency_ms=float(ordered[p95_index]),
        peak_allocated_bytes=peak_allocated,
        peak_incremental_allocated_bytes=peak_incremental,
        warmup=warmup,
        repetitions=repetitions,
        device=str(device),
        timing_method=timing_method,
    )


__all__ = [
    "EfficiencyReport",
    "conv2d_macs",
    "measure_decoder_efficiency",
    "parameter_counts",
]
