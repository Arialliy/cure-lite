"""Frozen numerical-runtime evidence for CURE-Lite v23 PACRE-VC.

The environment receipt is generated without reading any dataset.  It records
the numerical switches that can affect exact replay and includes an explicit
subnormal-arithmetic probe on both CPU and the selected accelerator.
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Final, Mapping

import torch

from cure_lite.cache.schema import stable_fingerprint


PACRE_VC_RUNTIME_ENVIRONMENT_SCHEMA: Final = (
    "cure-lite-v23-pacre-vc-runtime-environment-v1"
)
PACRE_VC_NUMERICAL_ENVIRONMENT_KEYS: Final = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_TF32_OVERRIDE",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "PYTHONHASHSEED",
    "PYTORCH_CUDA_ALLOC_CONF",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
)


def stabilize_pacre_vc_numerical_runtime() -> None:
    """Set the frozen deterministic/FP32 policy before taking a lock.

    Recent PyTorch builds lazily normalize ``fp32_precision`` from ``none``
    to ``ieee`` on the first CUDA deterministic scope.  Setting the policy
    explicitly prevents a generated parity probe from changing the otherwise
    identical runtime receipt.
    """

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # The transitive v21/v22 deterministic scope still reads the legacy
    # aggregate flags.  Use those setters here so every cuDNN operator family
    # is normalized together; mixing per-operator and aggregate APIs makes
    # the legacy getter intentionally raise in current PyTorch.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _cuda_matmul_tf32_policy() -> object:
    backend = torch.backends.cuda.matmul
    if hasattr(backend, "fp32_precision"):
        return str(backend.fp32_precision)
    return bool(backend.allow_tf32)


def _cudnn_tf32_policy() -> object:
    backend = torch.backends.cudnn
    convolution = getattr(backend, "conv", None)
    if convolution is not None and hasattr(
        convolution,
        "fp32_precision",
    ):
        return str(convolution.fp32_precision)
    return bool(backend.allow_tf32)


def _float32_raw_bits(value: torch.Tensor) -> int:
    cpu = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if cpu.numel() != 1:
        raise ValueError("raw-bit probe requires one float32 scalar")
    return int(cpu.view(torch.int32).item())


def subnormal_arithmetic_probe(
    device: torch.device | str,
) -> dict[str, object]:
    """Probe whether one normal-input subtraction preserves a subnormal.

    The two operands are normal FP32 values.  Their exact difference is the
    largest finite subnormal.  A zero result identifies an FTZ-like execution
    path.  PACRE-VC bounds are TINY32-safe either way, but the observed policy
    remains part of the runtime receipt.
    """

    resolved = torch.device(device)
    tiny = torch.tensor(
        torch.finfo(torch.float32).tiny,
        dtype=torch.float32,
        device=resolved,
    )
    zero = torch.tensor(0.0, dtype=torch.float32, device=resolved)
    upper = torch.nextafter(2.0 * tiny, zero)
    difference = tiny - upper
    expected = torch.nextafter(tiny, zero)
    gradual = bool(
        difference != 0.0
        and torch.equal(difference.abs(), expected)
    )
    return {
        "device": str(resolved),
        "lower_normal_hex": float(tiny.detach().cpu()).hex(),
        "upper_normal_hex": float(upper.detach().cpu()).hex(),
        "difference_hex": float(difference.detach().cpu()).hex(),
        "difference_raw_int32": _float32_raw_bits(difference),
        "expected_subnormal_hex": float(expected.detach().cpu()).hex(),
        "expected_subnormal_raw_int32": _float32_raw_bits(expected),
        "gradual_underflow_observed": gradual,
        "ftz_like_observed": bool(difference == 0.0),
    }


def runtime_environment_payload(
    selected_device: torch.device | str,
) -> dict[str, object]:
    """Return the canonical numerical environment for one selected device."""

    device = torch.device(selected_device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("selected CUDA device is unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(device)
        device_payload: dict[str, object] = {
            "type": "cuda",
            "logical_device": str(device),
            "name": properties.name,
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
            "total_memory": int(properties.total_memory),
            "multi_processor_count": int(
                properties.multi_processor_count
            ),
            "uuid": str(properties.uuid),
            "pci_domain_id": int(properties.pci_domain_id),
            "pci_bus_id": int(properties.pci_bus_id),
            "pci_device_id": int(properties.pci_device_id),
        }
    elif device.type == "cpu":
        device_payload = {
            "type": "cpu",
            "logical_device": "cpu",
            "processor": platform.processor(),
            "machine": platform.machine(),
        }
    else:
        raise ValueError("PACRE-VC supports only CPU or CUDA")

    environment = {
        key: os.environ.get(key)
        for key in PACRE_VC_NUMERICAL_ENVIRONMENT_KEYS
    }
    payload: dict[str, object] = {
        "schema_version": PACRE_VC_RUNTIME_ENVIRONMENT_SCHEMA,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "default_dtype": str(torch.get_default_dtype()),
        "default_device": str(torch.get_default_device()),
        "cpu_thread_count": int(torch.get_num_threads()),
        "cpu_interop_thread_count": int(torch.get_num_interop_threads()),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_tf32_policy": _cuda_matmul_tf32_policy(),
        "cudnn_tf32_policy": _cudnn_tf32_policy(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cpu_autocast_enabled": bool(torch.is_autocast_enabled("cpu")),
        "cuda_autocast_enabled": bool(torch.is_autocast_enabled("cuda")),
        "selected_device": device_payload,
        "environment_variables": environment,
        "cpu_subnormal_probe": subnormal_arithmetic_probe("cpu"),
        "selected_device_subnormal_probe": (
            subnormal_arithmetic_probe(device)
        ),
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "training_performed": False,
    }
    return payload


def fingerprinted_runtime_environment(
    selected_device: torch.device | str,
) -> dict[str, object]:
    """Return the environment payload plus a canonical fingerprint."""

    payload = runtime_environment_payload(selected_device)
    return {
        **payload,
        "environment_fingerprint": stable_fingerprint(payload),
    }


def verify_runtime_environment(
    locked: Mapping[str, object],
    selected_device: torch.device | str,
) -> str:
    """Require exact equality with a previously frozen environment receipt."""

    if not isinstance(locked, Mapping):
        raise TypeError("locked environment must be a mapping")
    current = fingerprinted_runtime_environment(selected_device)
    if dict(locked) != current:
        raise RuntimeError("PACRE-VC runtime environment differs from lock")
    fingerprint = current["environment_fingerprint"]
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise AssertionError("runtime environment fingerprint is malformed")
    return fingerprint


__all__ = [
    "PACRE_VC_NUMERICAL_ENVIRONMENT_KEYS",
    "PACRE_VC_RUNTIME_ENVIRONMENT_SCHEMA",
    "fingerprinted_runtime_environment",
    "runtime_environment_payload",
    "stabilize_pacre_vc_numerical_runtime",
    "subnormal_arithmetic_probe",
    "verify_runtime_environment",
]
