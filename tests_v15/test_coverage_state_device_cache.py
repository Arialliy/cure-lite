from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

import pytest
import torch

from cure_lite.coverage_state_device_cache import (
    COVERAGE_STATE_DEVICE_CACHE_PAIR_ROLES,
    materialize_coverage_state_device_fused_batch,
    prepare_coverage_state_device_cache,
)
from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
    materialize_coverage_state_fused_batch,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_training_scalar_cache,
)


def _cache_schedule():
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=2,
            steps_per_epoch=3,
        ),
    )
    return cache, schedule


def _assert_exact_dataclass_tensors(first, second) -> None:
    assert type(first) is type(second)
    assert is_dataclass(first)
    for field in fields(first):
        left = getattr(first, field.name)
        right = getattr(second, field.name)
        if isinstance(left, torch.Tensor):
            assert isinstance(right, torch.Tensor)
            assert left.dtype == right.dtype
            assert left.device == right.device
            assert tuple(left.shape) == tuple(right.shape)
            assert torch.equal(left, right)
            assert left.is_contiguous()
            assert right.is_contiguous()
        elif is_dataclass(left):
            _assert_exact_dataclass_tensors(left, right)
        else:
            assert left == right


def test_cpu_device_cache_matches_existing_materializer_tensor_by_tensor() -> None:
    scalar, schedule = _cache_schedule()
    packed = prepare_coverage_state_device_cache(
        scalar,
        device="cpu",
    )
    selection = schedule.selections[4]
    reference = materialize_coverage_state_fused_batch(
        scalar,
        schedule,
        epoch=1,
        step=1,
        device="cpu",
    )
    actual = materialize_coverage_state_device_fused_batch(
        packed,
        selection,
    )
    _assert_exact_dataclass_tensors(actual, reference)
    assert actual.selection_fingerprint == reference.selection_fingerprint
    packed.verify_unchanged()


def test_device_cache_contains_only_eligible_pairs_and_readonly_indexes(
) -> None:
    scalar = make_training_scalar_cache()
    packed = prepare_coverage_state_device_cache(
        scalar,
        device=torch.device("cpu"),
    )
    eligible_ids = {
        value.record.pair_id
        for value in scalar.pair_records
        if value.optimizer_role
        in COVERAGE_STATE_DEVICE_CACHE_PAIR_ROLES
    }
    diagnostic_ids = {
        value.record.pair_id
        for value in scalar.pair_records
        if not value.optimization_eligible
    }
    assert set(packed.pair_id_to_index) == eligible_ids
    assert set(packed.pair_id_to_index).isdisjoint(diagnostic_ids)
    assert set(packed.pairs.optimizer_roles) == set(
        COVERAGE_STATE_DEVICE_CACHE_PAIR_ROLES
    )
    assert len(packed.natural_id_to_index) == len(
        scalar.natural_records
    )
    with pytest.raises(TypeError):
        packed.natural_id_to_index["new-id"] = 0  # type: ignore[index]
    with pytest.raises(TypeError):
        packed.pair_id_to_index["new-id"] = 0  # type: ignore[index]


def test_device_cache_is_bound_contiguous_and_reports_exact_tensor_memory() -> None:
    scalar = make_training_scalar_cache()
    packed = prepare_coverage_state_device_cache(
        scalar,
        device="cpu",
        dtype=torch.float32,
    )
    packed.verify_unchanged()
    assert packed.source_cache_fingerprint == scalar.cache_fingerprint
    assert packed.device == torch.device("cpu")
    assert packed.dtype == torch.float32
    for _, tensor in packed.named_tensors():
        assert tensor.device == packed.device
        assert tensor.is_contiguous()
        assert not tensor.requires_grad
        assert tensor.dtype in {
            torch.float32,
            torch.bool,
            torch.long,
        }
    expected_bytes = sum(
        tensor.numel() * tensor.element_size()
        for _, tensor in packed.named_tensors()
    )
    report = packed.memory_report()
    assert packed.resident_tensor_bytes == expected_bytes
    assert report["resident_tensor_bytes"] == expected_bytes
    assert report["tensor_count"] == len(packed.named_tensors())
    assert sum(report["by_dtype_bytes"].values()) == expected_bytes
    assert sum(report["by_store_bytes"].values()) == expected_bytes
    assert report["retained_source_cache_not_counted"] is True


def test_device_cache_construction_and_materialization_are_deterministic() -> None:
    first_scalar, first_schedule = _cache_schedule()
    second_scalar, second_schedule = _cache_schedule()
    first = prepare_coverage_state_device_cache(
        first_scalar,
        device="cpu",
    )
    second = prepare_coverage_state_device_cache(
        second_scalar,
        device="cpu",
    )
    assert first.device_cache_fingerprint == (
        second.device_cache_fingerprint
    )
    assert first.canonical_payload() == second.canonical_payload()
    assert dict(first.natural_id_to_index) == dict(
        second.natural_id_to_index
    )
    assert dict(first.pair_id_to_index) == dict(
        second.pair_id_to_index
    )
    for (first_name, first_tensor), (
        second_name,
        second_tensor,
    ) in zip(
        first.named_tensors(),
        second.named_tensors(),
        strict=True,
    ):
        assert first_name == second_name
        assert torch.equal(first_tensor, second_tensor)
    first_batch = first.materialize(first_schedule.selections[2])
    second_batch = second.materialize(second_schedule.selections[2])
    _assert_exact_dataclass_tensors(first_batch, second_batch)


def test_device_cache_detects_packed_tensor_mutation() -> None:
    packed = prepare_coverage_state_device_cache(
        make_training_scalar_cache(),
        device="cpu",
    )
    packed.natural.feature[0, 0, 0, 0] += 1.0
    with pytest.raises(
        RuntimeError,
        match="packed tensor changed",
    ):
        packed.verify_unchanged()


def test_device_cache_detects_data_mutation_that_bypasses_version() -> None:
    packed = prepare_coverage_state_device_cache(
        make_training_scalar_cache(),
        device="cpu",
    )
    original_version = packed.natural.feature._version
    packed.natural.feature.data[0, 0, 0, 0] += 1.0
    assert packed.natural.feature._version == original_version
    with pytest.raises(
        RuntimeError,
        match="packed tensor content changed",
    ):
        packed.verify_unchanged()


def test_device_cache_detects_source_cache_mutation() -> None:
    scalar = make_training_scalar_cache()
    packed = prepare_coverage_state_device_cache(
        scalar,
        device="cpu",
    )
    scalar.raw_catalog.natural_records[0].feature[0, 0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="raw catalog changed"):
        packed.verify_unchanged()


def test_materialized_batch_mutation_does_not_alias_device_cache() -> None:
    scalar, schedule = _cache_schedule()
    packed = prepare_coverage_state_device_cache(
        scalar,
        device="cpu",
    )
    selection = schedule.selections[0]
    batch = packed.materialize(selection)
    batch.factual_miss.feature[0, 0, 0, 0] += 1.0
    batch.factual_miss.targets.target_field[0, 0, 0, 0] += 1.0
    packed.verify_unchanged()
    replay = packed.materialize(selection)
    reference = materialize_coverage_state_fused_batch(
        scalar,
        schedule,
        epoch=0,
        step=0,
        device="cpu",
    )
    _assert_exact_dataclass_tensors(replay, reference)


def test_materialization_performs_no_second_payload_to_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scalar, schedule = _cache_schedule()
    calls = 0
    original_to = torch.Tensor.to

    def counted_to(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", counted_to)
    packed = prepare_coverage_state_device_cache(
        scalar,
        device="cpu",
    )
    calls_after_pack = calls
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError(
                "materialization rebuilt a host-origin index tensor"
            )
        ),
    )
    packed.materialize(
        schedule.selections[0],
        verify=False,
    )
    assert calls == calls_after_pack


def test_device_cache_rejects_diagnostic_selection_and_non_fp32_dtype() -> None:
    scalar, schedule = _cache_schedule()
    with pytest.raises(ValueError, match="fixes floating dtype"):
        prepare_coverage_state_device_cache(
            scalar,
            device="cpu",
            dtype=torch.float16,
        )
    packed = prepare_coverage_state_device_cache(
        scalar,
        device="cpu",
    )
    diagnostic = next(
        value
        for value in scalar.pair_records
        if value.optimizer_role == "diagnostic_only"
    )
    invalid = replace(
        schedule.selections[0],
        component_null_pair_id=diagnostic.record.pair_id,
    )
    with pytest.raises(ValueError, match="outside the device cache"):
        packed.materialize(invalid)
