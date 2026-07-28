from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cure_lite.coverage_state_binary_flip_antisymmetric import (
    CURELiteBinaryFlipAntisymmetricLevelSet,
    CoverageStateBinaryFlipAntisymmetricConfig,
)
from cure_lite.coverage_state_sobolev import (
    CoverageStateSobolevConfig,
    prepare_coverage_state_absolute_targets,
    prepare_coverage_state_pair_targets,
)
from cure_lite.experiment.coverage_state_bfa_dr_gate import (
    COVERAGE_STATE_BFA_DR_EXECUTION_SEED,
    COVERAGE_STATE_BFA_DR_FAIL_DECISION,
    COVERAGE_STATE_BFA_DR_PASS_DECISION,
    COVERAGE_STATE_BFA_DR_RATIO_THRESHOLD,
    CoverageStateBFADRGateReceipt,
    _direction_probe,
    _distribution,
    _hidden_basis,
    _probe,
    _representation_probe,
    _row_bit_hash,
    _state_specs,
    recompute_coverage_state_bfa_dr_checks,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_ROLE_COUNT,
)
from cure_lite.experiment.coverage_state_cmif_dataset_free import (
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_CMIF_FORMAL_WIDTH,
)
from cure_lite.experiment.coverage_state_bfa_dataset_free import (
    COVERAGE_STATE_BFA_MARGIN,
)


def _mask(size: int = 8) -> torch.Tensor:
    return torch.zeros(1, 1, size, size, dtype=torch.bool)


def _feature(
    *,
    channels: int = 2,
    size: int = 2,
    offset: float = 0.0,
) -> torch.Tensor:
    return (
        torch.arange(
            channels * size * size,
            dtype=torch.float32,
        )
        .reshape(1, channels, size, size)
        .add(1.0 + offset)
        .div(float(channels * size * size + 1))
    )


def _natural(
    index: int,
    *,
    miss: bool,
    channels: int = 2,
) -> object:
    valid = torch.ones_like(_mask())
    occupancy = _mask()
    target = _mask()
    if miss:
        target[:, :, 3, 3] = True
    config = CoverageStateSobolevConfig(truncation_radius=4)
    targets = prepare_coverage_state_absolute_targets(
        target,
        valid,
        config=config,
    )
    record = SimpleNamespace(
        record_id=f"{'miss' if miss else 'no-miss'}-{index:02d}",
        sample_id=f"sample-natural-{index:02d}",
        state_kind=("factual_miss" if miss else "factual_no_miss"),
        feature=_feature(channels=channels, offset=float(index)),
        occupancy=occupancy,
        target=target,
        valid_mask=valid,
    )
    return SimpleNamespace(record=record, targets=targets)


def _pair(
    index: int,
    *,
    clean: bool,
    channels: int = 2,
    component_target: bool = False,
) -> object:
    valid = torch.ones_like(_mask())
    empty = _mask()
    plus = _mask()
    minus = _mask()
    plus[:, :, 4, 4] = True
    target_plus = empty.clone()
    target_minus = empty.clone()
    if clean:
        target_minus[:, :, 4, 4] = True
    elif component_target:
        target_plus[:, :, 1, 2] = True
        target_minus[:, :, 1, 2] = True
    targets = prepare_coverage_state_pair_targets(
        plus,
        minus,
        target_plus,
        target_minus,
        valid,
        config=CoverageStateSobolevConfig(truncation_radius=4),
    )
    record = SimpleNamespace(
        pair_id=f"{'clean' if clean else 'component'}-{index:02d}",
        sample_id=f"sample-pair-{index:02d}",
        pair_kind=("clean_positive" if clean else "component_null"),
        feature=_feature(
            channels=channels,
            offset=float(100 + index),
        ),
        occupancy_plus=plus,
        occupancy_minus=minus,
        target_plus=target_plus,
        target_minus=target_minus,
        valid_mask=valid,
        removed_component=(plus & ~minus),
    )
    return SimpleNamespace(
        record=record,
        joint_targets=targets,
        optimizer_role=(
            "clean_positive" if clean else "component_null"
        ),
    )


def _population(
    *,
    count: int,
    channels: int = 2,
) -> object:
    naturals = tuple(
        [_natural(index, miss=True, channels=channels) for index in range(count)]
        + [
            _natural(index, miss=False, channels=channels)
            for index in range(count)
        ]
    )
    clean = tuple(
        _pair(index, clean=True, channels=channels)
        for index in range(count)
    )
    component = tuple(
        _pair(index, clean=False, channels=channels)
        for index in range(count)
    )
    cache = SimpleNamespace(
        natural_records=naturals,
        clean_positive_records=clean,
        component_null_records=component,
        sobolev_config=CoverageStateSobolevConfig(
            truncation_radius=4
        ),
    )
    return SimpleNamespace(cache=cache)


class _ProbePopulation:
    def __init__(self) -> None:
        value = _population(
            count=COVERAGE_STATE_BOUNDED_ROLE_COUNT,
            channels=COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
        )
        value.cache.raw_catalog = SimpleNamespace(split="D_R")
        value.cache.cache_fingerprint = "b" * 64
        self.cache = value.cache
        self.source_cache = SimpleNamespace(
            raw_catalog=SimpleNamespace(split="D_R"),
        )
        self.population_fingerprint = "c" * 64

    def verify_unchanged(self) -> None:
        return None


def test_hidden_basis_is_finite_and_uses_endpoint_midpoint_equations() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2020)
        config = CoverageStateBinaryFlipAntisymmetricConfig(
            feature_channels=2,
            feature_stride=4,
            width=3,
        )
        model = CURELiteBinaryFlipAntisymmetricLevelSet(config)
        occupancy = _mask()
        occupancy[:, :, 2, 5] = True
        basis = _hidden_basis(
            model,
            _feature(),
            occupancy,
        )

    for name in ("h0", "h1", "hm", "odd", "even", "oriented_odd"):
        value = getattr(basis, name)
        assert value.shape == (1, 3, 8, 8)
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()
    assert torch.allclose(
        basis.odd,
        0.5 * (basis.h0 - basis.h1),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.allclose(
        basis.even,
        0.5 * (basis.h0 + basis.h1) - basis.hm,
        rtol=0.0,
        atol=0.0,
    )
    sign = torch.where(
        occupancy,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    )
    assert torch.allclose(
        basis.oriented_odd,
        basis.odd * sign,
        rtol=2.0e-6,
        atol=2.0e-7,
    )


def test_distribution_is_deterministic_and_row_hash_preserves_float_bits() -> None:
    values = torch.tensor(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        dtype=torch.float32,
    )
    first = _distribution(values, name="toy")
    second = _distribution(values.clone(), name="toy")
    assert first == second
    assert first["count"] == 5
    assert float.fromhex(
        first["nearest_rank_quantiles"]["q050"]
    ) == 2.0

    rows = torch.tensor(
        [[0.0, 1.0], [-0.0, 1.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    hashes = _row_bit_hash(rows)
    assert hashes[0] == hashes[2]
    assert hashes[0] != hashes[1]


def test_actual_geometry_direction_probe_uses_frozen_losses() -> None:
    probe = _direction_probe(
        _population(count=COVERAGE_STATE_BOUNDED_ROLE_COUNT),
        device=torch.device("cpu"),
    )
    assert probe["all_roles_finite_nonzero_correct"] is True
    assert probe["uses_actual_target_geometry"] is True
    assert probe["uses_actual_valid_and_writable_masks"] is True
    assert probe["loss_apis"] == [
        "coverage_state_absolute_sobolev_loss_from_targets",
        "coverage_state_pmope_pair_loss_from_targets",
    ]
    assert probe["actual_role_rows"] == {
        "factual_target": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "clean_target": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "writable_background": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "writable_component": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
    }
    for row in probe["rows"]:
        assert float.fromhex(row["loss_hex"]) > 0.0
        assert row["descent_finite"] is True
        assert row["descent_nonzero"] is True
        assert row["aggregate_descent_direction_correct"] is True
        if row["desired_field_direction"] == "negative":
            assert float.fromhex(row["descent_sum_hex"]) < 0.0
        else:
            assert float.fromhex(row["descent_sum_hex"]) > 0.0


def test_representation_probe_reuses_target_basis_without_repeat_forward() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2021)
        model = CURELiteBinaryFlipAntisymmetricLevelSet(
            CoverageStateBinaryFlipAntisymmetricConfig(
                feature_channels=2,
                feature_stride=4,
                width=3,
            )
        )
        probe = _representation_probe(
            model,
            _population(count=1),
            device=torch.device("cpu"),
        )
    counts = probe["state_counts"]
    assert counts["target_states"] == 2
    assert counts["context_states"] == 6
    assert counts["unique_context_states"] == 6
    assert counts["unique_state_forward_count"] == 6
    assert counts["reused_basis_count"] == 2
    assert len(probe["target_group_rows"]) == 2
    assert set(probe["aggregate_distributions"]) == {
        "target",
        "background",
        "component",
    }


def test_component_null_background_excludes_endpoint_targets() -> None:
    component = _pair(0, clean=False, component_target=True)
    population = SimpleNamespace(
        cache=SimpleNamespace(
            natural_records=(),
            clean_positive_records=(),
            component_null_records=(component,),
        )
    )
    record = component.record
    joint = component.joint_targets

    _, context_states = _state_specs(population)
    component_states = {
        state.endpoint: state
        for state in context_states
        if state.state_kind == "component_null"
    }

    assert set(component_states) == {"plus", "minus"}
    for endpoint, endpoint_target in (
        ("plus", record.target_plus),
        ("minus", record.target_minus),
    ):
        state = component_states[endpoint]
        expected = (
            record.valid_mask
            & ~state.occupancy
            & ~record.removed_component
            & ~endpoint_target
        )
        assert torch.equal(state.background_mask, expected)
        assert not torch.any(state.background_mask & endpoint_target)
        assert bool(state.background_mask[:, :, 0, 0])
    assert torch.all(
        joint.target_field_plus[record.target_plus] < 0.0
    )
    assert torch.all(
        joint.target_field_minus[record.target_minus] < 0.0
    )


def test_complete_seed42_toy_probe_is_exact_and_read_only() -> None:
    population = _ProbePopulation()
    before = torch.random.get_rng_state().clone()
    first = _probe(population, device=torch.device("cpu"))
    second = _probe(population, device=torch.device("cpu"))
    assert first == second
    assert torch.equal(before, torch.random.get_rng_state())
    assert first["initial_model_fingerprint"] == (
        first["final_model_fingerprint"]
    )
    assert first["representation"]["state_counts"][
        "unique_state_forward_count"
    ] == 6 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
    assert first["representation"]["state_counts"][
        "reused_basis_count"
    ] == 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
    assert first["field_direction"][
        "all_roles_finite_nonzero_correct"
    ] is True
    assert first["update_zero_readout"][
        "all_readout_gradients_finite_nonzero"
    ] is True
    witness = first["nonzero_readout_joint_witness"]
    assert witness["joint_weight_gradient_nonzero"] is True
    assert witness["joint_bias_gradient_nonzero"] is True
    assert first["optimizer_constructed"] is False
    assert first["optimizer_steps"] == 0
    assert first["parameter_updates"] == 0


def _distribution_stub(*, nonzero: int = 1) -> dict[str, object]:
    return {
        "count": 1,
        "finite_count": 1,
        "nonzero_count": nonzero,
        "exact_zero_count": 1 - nonzero,
        "ordered_value_fingerprint": "a" * 64,
    }


def _passing_recompute_inputs() -> tuple[object, object, object, dict]:
    dataset_free = SimpleNamespace(all_pass=True)
    scalar_cache = SimpleNamespace(raw_catalog=SimpleNamespace(split="D_R"))
    real_inputs = SimpleNamespace(
        source_binding=SimpleNamespace(
            dataset="IRSTD-1K",
            split="D_R",
        ),
        bundle=SimpleNamespace(split="D_R"),
        raw_catalog=SimpleNamespace(split="D_R"),
        scalar_cache=scalar_cache,
    )
    bounded_cache = SimpleNamespace(
        raw_catalog=SimpleNamespace(split="D_R"),
        cache_fingerprint="b" * 64,
    )
    population = SimpleNamespace(
        source_cache=scalar_cache,
        cache=bounded_cache,
        seed=COVERAGE_STATE_BFA_DR_EXECUTION_SEED,
        population_fingerprint="c" * 64,
    )
    aggregate = {
        role: {
            name: _distribution_stub()
            for name in ("h0", "h1", "hm", "odd", "even", "rho")
        }
        for role in ("target", "background", "component")
    }
    target_rows = [
        {
            "target_group_id": f"target-{index:02d}",
            "coordinate_count": 1,
            "finite": True,
            "at_least_one_odd_coordinate_nonzero": True,
        }
        for index in range(2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT)
    ]
    direction_counts = {
        "factual_target": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "clean_target": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "writable_background": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "writable_component": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
    }
    probe = {
        "execution_seed": COVERAGE_STATE_BFA_DR_EXECUTION_SEED,
        "runtime_splits": ["D_R"],
        "representation": {
            "state_counts": {
                "target_states": 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT,
                "context_states": 6 * COVERAGE_STATE_BOUNDED_ROLE_COUNT,
                "unique_context_states": (
                    6 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
                ),
                "unique_state_forward_count": (
                    6 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
                ),
                "reused_basis_count": (
                    2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
                ),
                "factual_target_groups": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
                "clean_target_groups": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
            },
            "coordinate_counts": {
                "target": 1,
                "background": 1,
                "component": 1,
            },
            "target_group_rows": target_rows,
            "aggregate_distributions": aggregate,
            "global_odd_sum_squares_hex": float(1.0).hex(),
            "global_curvature_to_odd_ratio_hex": float(0.5).hex(),
            "ratio_threshold_hex": (
                COVERAGE_STATE_BFA_DR_RATIO_THRESHOLD.hex()
            ),
            "exact_mutually_exclusive_conflict_count": 0,
            "conflicting_vector_fingerprints": [],
            "conflict_examples": [],
        },
        "field_direction": {
            "all_roles_finite_nonzero_correct": True,
            "uses_actual_target_geometry": True,
            "uses_actual_valid_and_writable_masks": True,
            "loss_apis": [
                "coverage_state_absolute_sobolev_loss_from_targets",
                "coverage_state_pmope_pair_loss_from_targets",
            ],
            "actual_role_rows": direction_counts,
        },
        "update_zero_readout": {
            "factual_row_count": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
            "clean_row_count": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
            "all_losses_positive": True,
            "all_readout_gradients_finite_nonzero": True,
        },
        "nonzero_readout_joint_witness": {
            "joint_weight_gradient_finite": True,
            "joint_bias_gradient_finite": True,
            "joint_weight_gradient_nonzero": True,
            "joint_bias_gradient_nonzero": True,
            "joint_weight_gradient_l2_hex": float(1.0).hex(),
            "joint_bias_gradient_l2_hex": float(1.0).hex(),
            "functional_call_did_not_replace_model_parameters": True,
        },
        "model_config": {
            "model_class": (
                CURELiteBinaryFlipAntisymmetricLevelSet.__name__
            ),
            "feature_channels": (
                COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS
            ),
            "feature_stride": COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
            "width": COVERAGE_STATE_CMIF_FORMAL_WIDTH,
            "parameter_count": COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT,
            "parameter_tensor_count": 3,
            "field_policy": (
                CoverageStateBinaryFlipAntisymmetricConfig(
                    feature_channels=64,
                    feature_stride=4,
                    width=32,
                ).field_policy
            ),
            "equation_policy": (
                CoverageStateBinaryFlipAntisymmetricConfig(
                    feature_channels=64,
                    feature_stride=4,
                    width=32,
                ).equation_policy
            ),
            "flip_policy": (
                CoverageStateBinaryFlipAntisymmetricConfig(
                    feature_channels=64,
                    feature_stride=4,
                    width=32,
                ).flip_policy
            ),
            "margin_hex": COVERAGE_STATE_BFA_MARGIN.hex(),
        },
        "parameter_contract": [
            {"name": "joint_state_weight"},
            {"name": "joint_hidden_bias"},
            {"name": "scalar_energy_weight"},
        ],
        "initial_model_fingerprint": "d" * 64,
        "final_model_fingerprint": "d" * 64,
        "population_fingerprint_before": "c" * 64,
        "population_fingerprint_after": "c" * 64,
        "cache_fingerprint_before": "b" * 64,
        "cache_fingerprint_after": "b" * 64,
        "parameter_grad_buffers_unretained": True,
        "global_cpu_rng_preserved": True,
        "selected_device_rng_preserved": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_performed": False,
        "calibration_performed": False,
        "inference_performed": False,
        "historical_failure_coordinate_inputs": [],
    }
    return dataset_free, real_inputs, population, probe


def test_recompute_passes_and_rejects_ratio_conflict_and_repeat_forward() -> None:
    values = _passing_recompute_inputs()
    checks = dict(
        recompute_coverage_state_bfa_dr_checks(
            dataset_free_receipt=values[0],
            real_inputs=values[1],
            bounded_population=values[2],
            probe=values[3],
        )
    )
    assert checks
    assert all(checks.values())

    values = _passing_recompute_inputs()
    values[3]["representation"][
        "global_curvature_to_odd_ratio_hex"
    ] = COVERAGE_STATE_BFA_DR_RATIO_THRESHOLD.hex()
    checks = dict(
        recompute_coverage_state_bfa_dr_checks(
            dataset_free_receipt=values[0],
            real_inputs=values[1],
            bounded_population=values[2],
            probe=values[3],
        )
    )
    assert checks["odd_and_curvature_finite_nondegenerate"] is False

    values = _passing_recompute_inputs()
    values[3]["representation"][
        "exact_mutually_exclusive_conflict_count"
    ] = 1
    values[3]["representation"][
        "conflicting_vector_fingerprints"
    ] = ["e" * 64]
    checks = dict(
        recompute_coverage_state_bfa_dr_checks(
            dataset_free_receipt=values[0],
            real_inputs=values[1],
            bounded_population=values[2],
            probe=values[3],
        )
    )
    assert (
        checks["no_exact_odd_representation_interval_conflict"]
        is False
    )

    values = _passing_recompute_inputs()
    values[3]["representation"]["state_counts"][
        "unique_state_forward_count"
    ] += 1
    checks = dict(
        recompute_coverage_state_bfa_dr_checks(
            dataset_free_receipt=values[0],
            real_inputs=values[1],
            bounded_population=values[2],
            probe=values[3],
        )
    )
    assert checks["each_unique_state_forwarded_once"] is False


def test_target_curvature_is_required_even_when_background_is_nonlinear() -> None:
    values = _passing_recompute_inputs()
    values[3]["representation"]["aggregate_distributions"]["target"][
        "even"
    ]["nonzero_count"] = 0
    checks = dict(
        recompute_coverage_state_bfa_dr_checks(
            dataset_free_receipt=values[0],
            real_inputs=values[1],
            bounded_population=values[2],
            probe=values[3],
        )
    )
    assert checks["odd_and_curvature_finite_nondegenerate"] is False


def test_receipt_decision_and_fingerprint_are_recomputed_properties() -> None:
    assert isinstance(CoverageStateBFADRGateReceipt.all_pass, property)
    assert isinstance(CoverageStateBFADRGateReceipt.decision, property)
    assert isinstance(
        CoverageStateBFADRGateReceipt.receipt_fingerprint,
        property,
    )
    assert COVERAGE_STATE_BFA_DR_PASS_DECISION.endswith("_PASS")
    assert COVERAGE_STATE_BFA_DR_FAIL_DECISION.endswith("_FAIL")


def test_fixed_execution_seed_is_42() -> None:
    assert COVERAGE_STATE_BFA_DR_EXECUTION_SEED == 42


def test_distribution_rejects_empty_or_nonfinite() -> None:
    with pytest.raises(ValueError, match="finite and nonempty"):
        _distribution(torch.empty(0), name="empty")
    with pytest.raises(ValueError, match="finite and nonempty"):
        _distribution(torch.tensor([float("nan")]), name="nan")
