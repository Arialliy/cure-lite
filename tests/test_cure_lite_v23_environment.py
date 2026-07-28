from __future__ import annotations

import pytest
import torch

from cure_lite_v23.environment import (
    PACRE_VC_RUNTIME_ENVIRONMENT_SCHEMA,
    fingerprinted_runtime_environment,
    subnormal_arithmetic_probe,
    verify_runtime_environment,
)


def test_cpu_environment_receipt_is_exactly_recomputable() -> None:
    first = fingerprinted_runtime_environment("cpu")
    second = fingerprinted_runtime_environment(torch.device("cpu"))

    assert first == second
    assert first["schema_version"] == PACRE_VC_RUNTIME_ENVIRONMENT_SCHEMA
    assert first["D_R_accessed"] is False
    assert first["D_V_accessed"] is False
    assert first["D_T_accessed"] is False
    assert first["training_performed"] is False
    assert verify_runtime_environment(first, "cpu") == (
        first["environment_fingerprint"]
    )


def test_environment_lock_rejects_any_changed_field() -> None:
    locked = fingerprinted_runtime_environment("cpu")
    changed = dict(locked)
    changed["cpu_thread_count"] = int(changed["cpu_thread_count"]) + 1

    with pytest.raises(RuntimeError, match="differs from lock"):
        verify_runtime_environment(changed, "cpu")


def test_subnormal_probe_has_a_frozen_complete_contract() -> None:
    result = subnormal_arithmetic_probe("cpu")

    assert set(result) == {
        "device",
        "lower_normal_hex",
        "upper_normal_hex",
        "difference_hex",
        "difference_raw_int32",
        "expected_subnormal_hex",
        "expected_subnormal_raw_int32",
        "gradual_underflow_observed",
        "ftz_like_observed",
    }
    assert (
        result["gradual_underflow_observed"]
        or result["ftz_like_observed"]
    )
    assert not (
        result["gradual_underflow_observed"]
        and result["ftz_like_observed"]
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_cuda_environment_binds_physical_identity() -> None:
    receipt = fingerprinted_runtime_environment("cuda:0")
    selected = receipt["selected_device"]

    assert selected["logical_device"] == "cuda:0"
    assert selected["uuid"]
    assert selected["pci_bus_id"] >= 0
    assert receipt["selected_device_subnormal_probe"]["device"] == "cuda:0"
