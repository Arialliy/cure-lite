"""Stage-A efficiency receipt for the deployed U decoder.

The API accepts only a decoder, immutable identity digests, and tensor shapes.
It creates all-zero tensors itself, so it has no image/GT/split access path.
Static architecture counts and environment-dependent execution measurements
are deliberately separate and neither is a mechanism-gate metric.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from math import isfinite
import sys
from typing import Mapping, Sequence

import torch

from ..cache.schema import stable_fingerprint
from ..decoder import CURELiteDecoder
from ..efficiency import conv2d_macs, measure_decoder_efficiency, parameter_counts


STAGE_A_EFFICIENCY_SCHEMA = "cure-lite-stage-a-efficiency-receipt-v1"
DEFAULT_WARMUP = 10
DEFAULT_REPETITIONS = 50

_COUNTING_CONVENTION = (
    "Conv2d only: one multiply-accumulate is one MAC and two FLOPs; bias, "
    "normalization, activation, concatenation, occupancy projection, and "
    "interpolation are excluded"
)
_MEASUREMENT_SCOPE = (
    "one batch-1 decoder forward on all-zero synthetic tensors; excludes base "
    "inference, transfer, I/O, thresholding, union, calibration, and metrics"
)


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _shape(value: object, name: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, (tuple, list))
        or len(value) != 4
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in value
        )
    ):
        raise ValueError(f"{name} must contain four positive integers")
    return tuple(value)  # type: ignore[return-value]


def _integer(value: object, name: str, *, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _latency(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class EfficiencyBinding:
    """Identities supplied by the strict Stage-A artifact/cache loader."""

    decoder_artifact_fingerprint: str
    decoder_state_fingerprint: str
    decoder_receipt_sha256: str
    base_index_fingerprint: str
    preprocessing_fingerprint: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _digest(getattr(self, name), name)

    def canonical_payload(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_mapping(cls, value: object) -> "EfficiencyBinding":
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("efficiency binding fields are not canonical")
        result = cls(**dict(value))  # type: ignore[arg-type]
        if result.canonical_payload() != dict(value):
            raise ValueError("efficiency binding payload is not canonical")
        return result


@dataclass(frozen=True)
class StageAEfficiencyReceipt:
    """Canonical evidence; latency/allocator values are observations, not gates."""

    binding: EfficiencyBinding
    feature_shape: tuple[int, int, int, int]
    occupancy_shape: tuple[int, int, int, int]
    parameter_count: int
    conv2d_macs: int
    conv2d_flops: int
    requested_device: str
    resolved_device: str
    device_type: str
    device_name: str
    cuda_device_index: int | None
    cuda_compute_capability: str | None
    cuda_total_memory_bytes: int | None
    python_version: str
    torch_version: str
    torch_cuda_version: str | None
    cudnn_version: int | None
    torch_num_threads: int
    torch_num_interop_threads: int
    warmup: int
    repetitions: int
    timing_method: str
    median_latency_ms: float
    p95_latency_ms: float
    peak_allocated_bytes: int | None
    peak_incremental_allocated_bytes: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.binding, EfficiencyBinding):
            raise TypeError("binding must be EfficiencyBinding")
        object.__setattr__(
            self, "feature_shape", _shape(self.feature_shape, "feature_shape")
        )
        object.__setattr__(
            self,
            "occupancy_shape",
            _shape(self.occupancy_shape, "occupancy_shape"),
        )
        if self.feature_shape[0] != 1 or self.occupancy_shape[:2] != (1, 1):
            raise ValueError("efficiency receipt fixes batch/occupancy channels at one")
        if any(
            feature > occupancy
            for feature, occupancy in zip(
                self.feature_shape[-2:], self.occupancy_shape[-2:], strict=True
            )
        ):
            raise ValueError("feature grid may not exceed the occupancy grid")
        for name in ("parameter_count", "conv2d_macs", "conv2d_flops"):
            if _integer(getattr(self, name), name) == 0:
                raise ValueError(f"{name} must be positive")
        if self.conv2d_flops != 2 * self.conv2d_macs:
            raise ValueError("Conv2d FLOPs must equal two times MACs")

        if self.device_type not in {"cpu", "cuda"}:
            raise ValueError("efficiency supports only CPU or CUDA")
        for name in (
            "requested_device",
            "resolved_device",
            "device_name",
            "python_version",
            "torch_version",
            "timing_method",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if self.torch_cuda_version is not None and not isinstance(
            self.torch_cuda_version, str
        ):
            raise TypeError("torch_cuda_version must be a string or None")
        if self.cudnn_version is not None:
            _integer(self.cudnn_version, "cudnn_version")
        try:
            requested = torch.device(self.requested_device)
            resolved = torch.device(self.resolved_device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("receipt device strings are invalid") from error
        if requested.type != self.device_type or resolved.type != self.device_type:
            raise ValueError("requested/resolved device types are inconsistent")
        for name in ("torch_num_threads", "torch_num_interop_threads", "repetitions"):
            if _integer(getattr(self, name), name) == 0:
                raise ValueError(f"{name} must be positive")
        _integer(self.warmup, "warmup")
        median = _latency(self.median_latency_ms, "median_latency_ms")
        p95 = _latency(self.p95_latency_ms, "p95_latency_ms")
        object.__setattr__(self, "median_latency_ms", median)
        object.__setattr__(self, "p95_latency_ms", p95)
        if p95 < median:
            raise ValueError("p95 latency may not be below median latency")

        peak = _integer(
            self.peak_allocated_bytes, "peak_allocated_bytes", optional=True
        )
        incremental = _integer(
            self.peak_incremental_allocated_bytes,
            "peak_incremental_allocated_bytes",
            optional=True,
        )
        cuda_metadata = (
            self.cuda_device_index,
            self.cuda_compute_capability,
            self.cuda_total_memory_bytes,
        )
        if self.device_type == "cpu":
            if any(value is not None for value in cuda_metadata):
                raise ValueError("CPU receipt may not claim CUDA hardware")
            if peak is not None or incremental is not None:
                raise ValueError("CPU receipt must mark CUDA memory unavailable")
            if self.timing_method != "perf_counter_ns":
                raise ValueError("CPU receipt requires perf_counter_ns timing")
            if self.torch_cuda_version is not None and not self.torch_cuda_version:
                raise ValueError("torch_cuda_version may not be empty")
        else:
            if any(value is None for value in cuda_metadata):
                raise ValueError("CUDA receipt requires hardware metadata")
            cuda_index = _integer(self.cuda_device_index, "cuda_device_index")
            cuda_memory = _integer(
                self.cuda_total_memory_bytes, "cuda_total_memory_bytes"
            )
            if cuda_memory == 0:
                raise ValueError("cuda_total_memory_bytes must be positive")
            if resolved.index != cuda_index:
                raise ValueError("resolved CUDA device and hardware index differ")
            if (
                not isinstance(self.cuda_compute_capability, str)
                or not self.cuda_compute_capability
            ):
                raise ValueError("CUDA compute capability must be a non-empty string")
            if self.torch_cuda_version is None or not self.torch_cuda_version:
                raise ValueError("CUDA receipt requires a CUDA runtime version")
            if peak is None or incremental is None or incremental > peak:
                raise ValueError("CUDA allocator peaks are inconsistent")
            if self.timing_method != "torch_cuda_event_elapsed_time":
                raise ValueError("CUDA receipt requires CUDA-event timing")

    def _input_payload(self) -> dict[str, object]:
        return {
            "feature_shape": list(self.feature_shape),
            "occupancy_shape": list(self.occupancy_shape),
            "feature_dtype": "float32",
            "occupancy_dtype": "bool",
            "content_recipe": "all_zero_synthetic_shape_probe",
            "reads_image_or_gt_content": False,
        }

    def _environment_payload(self) -> dict[str, object]:
        names = (
            "requested_device",
            "resolved_device",
            "device_type",
            "device_name",
            "cuda_device_index",
            "cuda_compute_capability",
            "cuda_total_memory_bytes",
            "python_version",
            "torch_version",
            "torch_cuda_version",
            "cudnn_version",
            "torch_num_threads",
            "torch_num_interop_threads",
        )
        return {name: getattr(self, name) for name in names}

    def static_payload(self) -> dict[str, object]:
        core: dict[str, object] = {
            "deployed_method": "U",
            "decoder_variant": "uniform_legal",
            "scope": "incremental_CURELiteDecoder_only",
            "binding": self.binding.canonical_payload(),
            "input_contract": self._input_payload(),
            "parameter_count": self.parameter_count,
            "conv2d_macs": self.conv2d_macs,
            "conv2d_flops": self.conv2d_flops,
            "counting_convention": _COUNTING_CONVENTION,
            "deterministic_replayable": True,
        }
        return {**core, "static_fingerprint": stable_fingerprint(core)}

    @property
    def static_fingerprint(self) -> str:
        return self.static_payload()["static_fingerprint"]  # type: ignore[return-value]

    def execution_payload(self) -> dict[str, object]:
        core: dict[str, object] = {
            "role": "environment_dependent_execution_measurement",
            "scientific_gate_metric": False,
            "exact_cross_environment_numeric_replay_expected": False,
            "static_fingerprint": self.static_fingerprint,
            "measurement_scope": _MEASUREMENT_SCOPE,
            "environment": self._environment_payload(),
            "warmup": self.warmup,
            "repetitions": self.repetitions,
            "timing_method": self.timing_method,
            "median_latency_ms": self.median_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_incremental_allocated_bytes": self.peak_incremental_allocated_bytes,
        }
        return {**core, "measurement_fingerprint": stable_fingerprint(core)}

    def canonical_payload(self) -> dict[str, object]:
        core: dict[str, object] = {
            "schema_version": STAGE_A_EFFICIENCY_SCHEMA,
            "method": "CURE-Lite",
            "stage": "Stage-A",
            "deployed_method": "U",
            "efficiency_is_scientific_gate_metric": False,
            "static_evidence": self.static_payload(),
            "execution_measurement": self.execution_payload(),
        }
        return {**core, "receipt_fingerprint": stable_fingerprint(core)}

    @property
    def receipt_fingerprint(self) -> str:
        return self.canonical_payload()["receipt_fingerprint"]  # type: ignore[return-value]

    @classmethod
    def from_mapping(cls, value: object) -> "StageAEfficiencyReceipt":
        """Strictly parse and reproduce every fixed field and fingerprint."""

        root_keys = {
            "schema_version",
            "method",
            "stage",
            "deployed_method",
            "efficiency_is_scientific_gate_metric",
            "static_evidence",
            "execution_measurement",
            "receipt_fingerprint",
        }
        if not isinstance(value, Mapping) or set(value) != root_keys:
            raise ValueError("Stage-A efficiency receipt fields are not canonical")
        static = value["static_evidence"]
        execution = value["execution_measurement"]
        if not isinstance(static, Mapping) or not isinstance(execution, Mapping):
            raise TypeError("efficiency receipt sections must be mappings")
        binding = EfficiencyBinding.from_mapping(static.get("binding"))
        input_contract = static.get("input_contract")
        environment = execution.get("environment")
        if not isinstance(input_contract, Mapping) or not isinstance(environment, Mapping):
            raise TypeError("efficiency input/environment must be mappings")
        result = cls(
            binding=binding,
            feature_shape=_shape(input_contract.get("feature_shape"), "feature_shape"),
            occupancy_shape=_shape(
                input_contract.get("occupancy_shape"), "occupancy_shape"
            ),
            parameter_count=static.get("parameter_count"),  # type: ignore[arg-type]
            conv2d_macs=static.get("conv2d_macs"),  # type: ignore[arg-type]
            conv2d_flops=static.get("conv2d_flops"),  # type: ignore[arg-type]
            requested_device=environment.get("requested_device"),  # type: ignore[arg-type]
            resolved_device=environment.get("resolved_device"),  # type: ignore[arg-type]
            device_type=environment.get("device_type"),  # type: ignore[arg-type]
            device_name=environment.get("device_name"),  # type: ignore[arg-type]
            cuda_device_index=environment.get("cuda_device_index"),  # type: ignore[arg-type]
            cuda_compute_capability=environment.get("cuda_compute_capability"),  # type: ignore[arg-type]
            cuda_total_memory_bytes=environment.get("cuda_total_memory_bytes"),  # type: ignore[arg-type]
            python_version=environment.get("python_version"),  # type: ignore[arg-type]
            torch_version=environment.get("torch_version"),  # type: ignore[arg-type]
            torch_cuda_version=environment.get("torch_cuda_version"),  # type: ignore[arg-type]
            cudnn_version=environment.get("cudnn_version"),  # type: ignore[arg-type]
            torch_num_threads=environment.get("torch_num_threads"),  # type: ignore[arg-type]
            torch_num_interop_threads=environment.get("torch_num_interop_threads"),  # type: ignore[arg-type]
            warmup=execution.get("warmup"),  # type: ignore[arg-type]
            repetitions=execution.get("repetitions"),  # type: ignore[arg-type]
            timing_method=execution.get("timing_method"),  # type: ignore[arg-type]
            median_latency_ms=execution.get("median_latency_ms"),  # type: ignore[arg-type]
            p95_latency_ms=execution.get("p95_latency_ms"),  # type: ignore[arg-type]
            peak_allocated_bytes=execution.get("peak_allocated_bytes"),  # type: ignore[arg-type]
            peak_incremental_allocated_bytes=execution.get(
                "peak_incremental_allocated_bytes"
            ),  # type: ignore[arg-type]
        )
        if result.canonical_payload() != dict(value):
            raise ValueError("Stage-A efficiency receipt payload is not canonical")
        return result


def _resolve_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("efficiency device is invalid") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("efficiency supports only CPU or CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA efficiency requested but CUDA is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError("CUDA efficiency device index is unavailable")
        return torch.device("cuda", index)
    return torch.device("cpu")


def _environment(requested: str, device: torch.device) -> dict[str, object]:
    if device.type == "cuda":
        assert device.index is not None
        properties = torch.cuda.get_device_properties(device)
        cuda = {
            "cuda_device_index": device.index,
            "cuda_compute_capability": f"{properties.major}.{properties.minor}",
            "cuda_total_memory_bytes": int(properties.total_memory),
            "device_name": str(properties.name),
        }
    else:
        cuda = {
            "cuda_device_index": None,
            "cuda_compute_capability": None,
            "cuda_total_memory_bytes": None,
            "device_name": "CPU",
        }
    cudnn = torch.backends.cudnn.version()
    return {
        "requested_device": str(torch.device(requested)),
        "resolved_device": str(device),
        "device_type": device.type,
        **cuda,
        "python_version": (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn_version": None if cudnn is None else int(cudnn),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }


def measure_stage_a_efficiency(
    decoder: CURELiteDecoder,
    *,
    decoder_variant: str,
    binding: EfficiencyBinding,
    feature_shape: Sequence[int],
    occupancy_shape: Sequence[int],
    device: str,
    warmup: int = DEFAULT_WARMUP,
    repetitions: int = DEFAULT_REPETITIONS,
) -> StageAEfficiencyReceipt:
    """Measure only U using synthetic tensors; no data object is accepted."""

    if type(decoder) is not CURELiteDecoder:
        raise TypeError("decoder must be exactly CURELiteDecoder")
    if decoder_variant != "uniform_legal":
        raise ValueError("Stage-A efficiency measures only the deployed U decoder")
    if not isinstance(binding, EfficiencyBinding):
        raise TypeError("binding must be EfficiencyBinding")
    feature_shape = _shape(feature_shape, "feature_shape")
    occupancy_shape = _shape(occupancy_shape, "occupancy_shape")
    if feature_shape[1] != decoder.feature_channels:
        raise ValueError("feature shape differs from decoder channel contract")

    # Fresh topology: counts cannot depend on frozen requires_grad flags or weights.
    with torch.random.fork_rng(devices=[]):
        topology = CURELiteDecoder(decoder.config).eval()
    parameters, _ = parameter_counts(topology)
    static_feature = torch.zeros(feature_shape, dtype=torch.float32)
    static_occupancy = torch.zeros(occupancy_shape, dtype=torch.bool)
    macs = conv2d_macs(topology, static_feature, static_occupancy)

    resolved_device = _resolve_device(device)
    measured_decoder = copy.deepcopy(decoder).to(device=resolved_device).eval()
    feature = torch.zeros(feature_shape, dtype=torch.float32, device=resolved_device)
    occupancy = torch.zeros(occupancy_shape, dtype=torch.bool, device=resolved_device)
    report = measure_decoder_efficiency(
        measured_decoder,
        feature,
        occupancy,
        warmup=warmup,
        repetitions=repetitions,
    )
    if (
        report.parameters != parameters
        or report.multiply_accumulates != macs
        or report.flops != 2 * macs
    ):
        raise RuntimeError("static and execution efficiency counts differ")
    environment = _environment(device, resolved_device)
    return StageAEfficiencyReceipt(
        binding=binding,
        feature_shape=feature_shape,
        occupancy_shape=occupancy_shape,
        parameter_count=parameters,
        conv2d_macs=macs,
        conv2d_flops=2 * macs,
        **environment,
        warmup=report.warmup,
        repetitions=report.repetitions,
        timing_method=report.timing_method,
        median_latency_ms=report.median_latency_ms,
        p95_latency_ms=report.p95_latency_ms,
        peak_allocated_bytes=report.peak_allocated_bytes,
        peak_incremental_allocated_bytes=report.peak_incremental_allocated_bytes,
    )


def replay_static_efficiency(
    receipt: StageAEfficiencyReceipt,
    decoder: CURELiteDecoder,
) -> None:
    """Recompute deterministic facts without replaying latency or allocator peaks."""

    if not isinstance(receipt, StageAEfficiencyReceipt):
        raise TypeError("receipt must be StageAEfficiencyReceipt")
    if type(decoder) is not CURELiteDecoder:
        raise TypeError("decoder must be exactly CURELiteDecoder")
    if receipt.feature_shape[1] != decoder.feature_channels:
        raise RuntimeError("receipt input channels differ from decoder")
    with torch.random.fork_rng(devices=[]):
        topology = CURELiteDecoder(decoder.config).eval()
    parameters, _ = parameter_counts(topology)
    feature = torch.zeros(receipt.feature_shape, dtype=torch.float32)
    occupancy = torch.zeros(receipt.occupancy_shape, dtype=torch.bool)
    macs = conv2d_macs(topology, feature, occupancy)
    if (
        parameters != receipt.parameter_count
        or macs != receipt.conv2d_macs
        or 2 * macs != receipt.conv2d_flops
    ):
        raise RuntimeError("static Stage-A efficiency evidence does not reproduce")


__all__ = [
    "DEFAULT_REPETITIONS",
    "DEFAULT_WARMUP",
    "EfficiencyBinding",
    "StageAEfficiencyReceipt",
    "measure_stage_a_efficiency",
    "replay_static_efficiency",
]
