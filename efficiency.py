"""Incremental decoder efficiency measurements for required CURE-Lite reporting."""

from __future__ import annotations

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
    peak_vram_bytes: int
    repetitions: int
    device: str

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def parameter_counts(module: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return total, trainable


def conv2d_macs(module: nn.Module, feature: Tensor, occupancy: Tensor) -> int:
    """Count Conv2d MACs for one decoder forward; interpolation is reported separately."""

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
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be non-negative and repetitions must be positive")
    if feature.device != occupancy.device:
        raise ValueError("feature and occupancy must use the same device")
    device = feature.device
    prior_mode = decoder.training
    decoder.eval()
    total_parameters, trainable_parameters = parameter_counts(decoder)
    macs = conv2d_macs(decoder, feature, occupancy)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            decoder(feature, occupancy)
        synchronize()
        samples = []
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            decoder(feature, occupancy)
            synchronize()
            samples.append((time.perf_counter_ns() - started) / 1_000_000.0)

    peak_vram = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
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
        peak_vram_bytes=peak_vram,
        repetitions=repetitions,
        device=str(device),
    )
