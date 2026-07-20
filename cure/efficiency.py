"""Reproducible incremental-inference measurements for full CURE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
import time

import torch
from torch import Tensor, nn

from .decoder import CUREResidualDecoder
from .descriptors import dilate_mask
from .model import noisy_or


@dataclass(frozen=True)
class CUREEfficiencyReport:
    """Measured cost of the inference-time add-on after frozen-base outputs.

    ``conv_macs`` covers every decoder ``Conv2d`` exactly.  Latency and peak
    memory cover the complete incremental path: feature projection,
    interpolation, decoder, occupancy suppression, sigmoid and noisy-OR.
    """

    parameters: int
    trainable_parameters: int
    conv_macs: int
    conv_flops: int
    median_incremental_latency_ms: float
    p95_incremental_latency_ms: float
    peak_incremental_vram_bytes: int
    repetitions: int
    batch_size: int
    feature_shape: tuple[int, ...]
    evaluation_shape: tuple[int, ...]
    device: str
    mac_scope: str = "decoder Conv2d only; interpolation/fusion excluded from MAC count"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _parameter_counts(module: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def _incremental_forward(
    decoder: CUREResidualDecoder,
    feature: Tensor,
    base_probability: Tensor,
    occupancy: Tensor,
) -> Tensor:
    logits = decoder(feature, base_probability, occupancy)
    exclusion = dilate_mask(occupancy, decoder.config.suppression_radius)
    residual = torch.sigmoid(logits).masked_fill(exclusion, 0.0)
    return noisy_or(base_probability, residual)


def _conv_macs(
    decoder: CUREResidualDecoder,
    feature: Tensor,
    base_probability: Tensor,
    occupancy: Tensor,
) -> int:
    macs = 0
    handles = []

    def hook(layer: nn.Conv2d, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        nonlocal macs
        if not isinstance(output, Tensor) or output.ndim != 4:
            raise RuntimeError("Conv2d hook received an unexpected output")
        batch, out_channels, height, width = output.shape
        kernel_height, kernel_width = layer.kernel_size
        per_output = (
            layer.in_channels // layer.groups
        ) * kernel_height * kernel_width
        macs += int(batch * out_channels * height * width * per_output)

    for child in decoder.modules():
        if isinstance(child, nn.Conv2d):
            handles.append(child.register_forward_hook(hook))
    try:
        with torch.no_grad():
            _incremental_forward(decoder, feature, base_probability, occupancy)
    finally:
        for handle in handles:
            handle.remove()
    return macs


def measure_cure_incremental_efficiency(
    decoder: CUREResidualDecoder,
    feature: Tensor,
    base_probability: Tensor,
    occupancy: Tensor,
    *,
    warmup: int = 10,
    repetitions: int = 50,
) -> CUREEfficiencyReport:
    """Measure the deploy-time CURE add-on without timing the frozen base.

    Base and CURE end-to-end latency must additionally be reported on the same
    hardware.  This function isolates the incremental carrier cost so training-
    only catalog/propensity work cannot be confused with deployment overhead.
    """

    if type(decoder) is not CUREResidualDecoder:
        raise TypeError("decoder must be the exact CUREResidualDecoder carrier")
    for name, value, minimum in (
        ("warmup", warmup, 0),
        ("repetitions", repetitions, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if any(
        not isinstance(value, Tensor)
        for value in (feature, base_probability, occupancy)
    ):
        raise TypeError("feature, base_probability, and occupancy must be tensors")
    if not (
        feature.device == base_probability.device == occupancy.device
    ):
        raise ValueError("efficiency inputs must share a device")

    prior_mode = decoder.training
    decoder.eval()
    # Exercise the decoder contract before installing timing hooks.
    with torch.no_grad():
        _incremental_forward(decoder, feature, base_probability, occupancy)
    total_parameters, trainable_parameters = _parameter_counts(decoder)
    macs = _conv_macs(decoder, feature, base_probability, occupancy)
    device = feature.device

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(warmup):
            _incremental_forward(decoder, feature, base_probability, occupancy)
        synchronize()
        if device.type == "cuda":
            baseline_vram = int(torch.cuda.memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)
        else:
            baseline_vram = 0
        samples = []
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            _incremental_forward(decoder, feature, base_probability, occupancy)
            synchronize()
            samples.append((time.perf_counter_ns() - started) / 1_000_000.0)

    peak_incremental_vram = (
        max(0, int(torch.cuda.max_memory_allocated(device)) - baseline_vram)
        if device.type == "cuda"
        else 0
    )
    decoder.train(prior_mode)
    ordered = sorted(samples)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    )
    p95_index = min(len(ordered) - 1, ceil(0.95 * len(ordered)) - 1)
    return CUREEfficiencyReport(
        parameters=total_parameters,
        trainable_parameters=trainable_parameters,
        conv_macs=macs,
        conv_flops=2 * macs,
        median_incremental_latency_ms=float(median),
        p95_incremental_latency_ms=float(ordered[p95_index]),
        peak_incremental_vram_bytes=peak_incremental_vram,
        repetitions=repetitions,
        batch_size=int(feature.shape[0]),
        feature_shape=tuple(int(value) for value in feature.shape),
        evaluation_shape=tuple(int(value) for value in base_probability.shape),
        device=str(device),
    )


__all__ = ["CUREEfficiencyReport", "measure_cure_incremental_efficiency"]
