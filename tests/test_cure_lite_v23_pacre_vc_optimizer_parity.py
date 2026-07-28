from __future__ import annotations

import pytest
import torch

from cure_lite.cache.schema import canonical_json, stable_fingerprint
from cure_lite.paired_types import tensor_content_fingerprint
from cure_lite_v23 import parity
from cure_lite_v23.pacre_vc import (
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREVerifierCorrectedConfig,
)


def _assert_receipt_fingerprint(receipt: dict[str, object]) -> None:
    body = dict(receipt)
    fingerprint = body.pop("receipt_fingerprint")
    assert fingerprint == stable_fingerprint(body)
    assert isinstance(canonical_json(receipt), str)


def _assert_run_is_complete(run: dict[str, object]) -> None:
    assert run["gate_passed"] is True
    assert run["state_dict_parity"]["passed"] is True
    assert run["all_fields_raw_parity"]["passed"] is True
    assert run["probe_gradient_parity"]["passed"] is True
    assert run["probe_models_preserved"] is True
    assert run["global_cpu_rng_preserved"] is True
    assert run["selected_device_rng_preserved"] is True
    assert run["deterministic_execution"]["restored_exactly"] is True
    optimizer = run["optimizer_parity"]
    assert optimizer["fresh_optimizer_state_empty"] is True
    assert optimizer["initial_model_state"]["passed"] is True
    assert optimizer["passed"] is True
    assert len(optimizer["steps"]) == 3
    for index, step in enumerate(optimizer["steps"], start=1):
        assert step["step"] == index
        assert step["passed"] is True
        assert step["loss"]["passed"] is True
        assert step["model_state"]["passed"] is True
        assert step["optimizer_state"]["step"]["passed"] is True
        assert step["optimizer_state"]["exp_avg"]["passed"] is True
        assert step["optimizer_state"]["exp_avg_sq"]["passed"] is True
        assert len(step["batch_fingerprint"]) == 64
    assert run["dataset_accessed"] is False
    assert run["cache_accessed"] is False
    assert run["D_R_accessed"] is False
    assert run["D_V_accessed"] is False
    assert run["D_T_accessed"] is False
    _assert_receipt_fingerprint(run)


def test_cpu_receipt_is_complete_replayable_and_preserves_external_state() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(230020)
        sentinel = CURELitePACREVerifierCorrectedLevelSet(
            CoverageStatePACREVerifierCorrectedConfig(
                feature_channels=2,
                feature_stride=2,
                width=4,
            )
        )
    sentinel_before = {
        name: tensor_content_fingerprint(value)
        for name, value in sentinel.state_dict().items()
    }
    rng_before = torch.random.get_rng_state().clone()

    first = parity.run_pacre_vc_generated_parity_receipt(
        include_cuda=False
    )
    second = parity.run_pacre_vc_generated_parity_receipt(
        include_cuda=False
    )

    assert first == second
    assert first["gate_passed"] is True
    assert first["required_devices"] == ["cpu"]
    assert first["required_seeds"] == [42, 43, 44]
    assert first["expected_run_count"] == 3
    assert first["observed_run_count"] == 3
    assert [run["seed"] for run in first["runs"]] == [42, 43, 44]
    assert all(run["device"] == "cpu" for run in first["runs"])
    for run in first["runs"]:
        _assert_run_is_complete(run)
    _assert_receipt_fingerprint(first)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert sentinel_before == {
        name: tensor_content_fingerprint(value)
        for name, value in sentinel.state_dict().items()
    }
    assert all(parameter.grad is None for parameter in sentinel.parameters())


def test_receipt_uses_real_pair_target_and_pmope_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepare_calls = 0
    loss_calls = 0
    original_prepare = parity.prepare_coverage_state_pair_targets
    original_loss = parity.coverage_state_pmope_pair_loss_from_targets

    def counted_prepare(*args: object, **kwargs: object):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    def counted_loss(*args: object, **kwargs: object):
        nonlocal loss_calls
        loss_calls += 1
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(
        parity,
        "prepare_coverage_state_pair_targets",
        counted_prepare,
    )
    monkeypatch.setattr(
        parity,
        "coverage_state_pmope_pair_loss_from_targets",
        counted_loss,
    )

    receipt = parity.run_pacre_vc_generated_parity_run(
        device="cpu",
        seed=42,
    )

    assert receipt["gate_passed"] is True
    assert prepare_calls == 3
    assert loss_calls == 6


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_cuda_seed_matrix_has_exact_three_step_optimizer_parity() -> None:
    receipt = parity.run_pacre_vc_generated_parity_receipt(
        include_cuda=True
    )

    assert receipt["gate_passed"] is True
    assert receipt["required_devices"] == ["cpu", "cuda:0"]
    assert receipt["required_seeds"] == [42, 43, 44]
    assert receipt["expected_run_count"] == 6
    assert receipt["observed_run_count"] == 6
    assert [
        (run["device"], run["seed"]) for run in receipt["runs"]
    ] == [
        ("cpu", 42),
        ("cpu", 43),
        ("cpu", 44),
        ("cuda:0", 42),
        ("cuda:0", 43),
        ("cuda:0", 44),
    ]
    for run in receipt["runs"]:
        _assert_run_is_complete(run)
    _assert_receipt_fingerprint(receipt)
