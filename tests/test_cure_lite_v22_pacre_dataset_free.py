from __future__ import annotations

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite_v22.dataset_free import (
    PACRE_DATASET_FREE_CHECK_NAMES,
    recompute_pacre_dataset_free_checks,
    run_pacre_dataset_free_gate,
)


def test_pacre_dataset_free_gate_passes_all_frozen_checks() -> None:
    receipt = run_pacre_dataset_free_gate()

    assert receipt["gate_passed"] is True
    assert tuple(receipt["checks"]) == PACRE_DATASET_FREE_CHECK_NAMES
    assert all(receipt["checks"].values())
    assert receipt["dataset_accessed"] is False
    assert receipt["cache_accessed"] is False
    assert receipt["D_R_accessed"] is False
    assert receipt["D_V_accessed"] is False
    assert receipt["D_T_accessed"] is False
    assert receipt["training_performed"] is False
    assert receipt["threshold_search_performed"] is False
    assert receipt["additional_heads"] == 0
    assert receipt["additional_branches"] == 0
    assert (
        receipt["checks"]["10_frozen_initialization_gradient_path"]
        is True
    )
    assert receipt["checks"]["11_exact_training_factory"] is True
    body = dict(receipt)
    fingerprint = body.pop("receipt_fingerprint")
    assert fingerprint == stable_fingerprint(body)


def test_pacre_dataset_free_replay_is_deterministic() -> None:
    assert run_pacre_dataset_free_gate() == run_pacre_dataset_free_gate()
    assert (
        recompute_pacre_dataset_free_checks()
        == recompute_pacre_dataset_free_checks()
    )


def test_dataset_free_uses_only_the_cpu_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_all_device_seed(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("torch.manual_seed must not be called")

    monkeypatch.setattr(
        torch,
        "manual_seed",
        forbidden_all_device_seed,
    )
    assert run_pacre_dataset_free_gate()["gate_passed"] is True
