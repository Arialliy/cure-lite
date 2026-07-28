from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cure_lite.coverage_state_sobolev import (
    CoverageStatePairTargets,
    CoverageStateSobolevConfig,
    prepare_coverage_state_pair_targets,
)
from cure_lite.coverage_state_supremal_projection import (
    CSLF_USCOPE_POLICY,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_ROLE_COUNT,
    COVERAGE_STATE_BOUNDED_SEED,
)
from cure_lite.experiment.coverage_state_cmif_dataset_free import (
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_CMIF_FORMAL_WIDTH,
)
from cure_lite.experiment.coverage_state_uscope_dr_gate import (
    COVERAGE_STATE_USCOPE_DR_EXECUTION_SEED,
    COVERAGE_STATE_USCOPE_DR_MASS_ATOL,
    COVERAGE_STATE_USCOPE_DR_MASS_RTOL,
    CoverageStateUSCOPEDRGateReceipt,
    _cuda_rng_devices,
    _geometry,
    _probe,
    _stack_targets,
    recompute_coverage_state_uscope_dr_checks,
)


def _targets(
    *,
    size: int = 8,
    clean: bool,
) -> tuple[CoverageStatePairTargets, torch.Tensor, torch.Tensor]:
    valid = torch.ones(1, 1, size, size, dtype=torch.bool)
    empty = torch.zeros_like(valid)
    occupancy_plus = empty.clone()
    occupancy_minus = empty.clone()
    occupancy_plus[:, :, size // 2, size // 2] = True
    target_minus = empty.clone()
    if clean:
        target_minus[:, :, size // 2, size // 2] = True
    targets = prepare_coverage_state_pair_targets(
        occupancy_plus,
        occupancy_minus,
        empty,
        target_minus,
        valid,
        config=CoverageStateSobolevConfig(truncation_radius=4),
    )
    return targets, occupancy_plus, occupancy_minus


def _cached_pair(*, role: str, pair_id: str, clean: bool) -> object:
    targets, occupancy_plus, occupancy_minus = _targets(clean=clean)
    feature = (
        torch.arange(64 * 2 * 2, dtype=torch.float32)
        .reshape(1, 64, 2, 2)
        .add(1.0)
        .div(257.0)
    )
    record = SimpleNamespace(
        pair_id=pair_id,
        pair_kind=("clean_positive" if clean else "component_null"),
        sample_id=f"sample-{pair_id}",
        feature=feature,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        target_plus=torch.zeros_like(occupancy_plus),
        target_minus=(
            targets.target_field_minus < 0.0
        ),
        valid_mask=targets.valid_mask.clone(),
    )
    return SimpleNamespace(
        optimizer_role=role,
        record=record,
        joint_targets=targets,
    )


class _FakeCache:
    def __init__(self) -> None:
        self.raw_catalog = SimpleNamespace(split="D_R")
        self.clean_positive_records = (
            _cached_pair(
                role="clean_positive",
                pair_id="clean",
                clean=True,
            ),
        )
        self.component_null_records = (
            _cached_pair(
                role="component_null",
                pair_id="component",
                clean=False,
            ),
        )
        self.sobolev_config = CoverageStateSobolevConfig(
            truncation_radius=4
        )

    def verify_unchanged(self) -> None:
        return None


class _FakePopulation:
    def __init__(self) -> None:
        self.cache = _FakeCache()
        self.source_cache = self.cache
        self.population_fingerprint = "a" * 64

    def verify_unchanged(self) -> None:
        return None


def test_cuda_fork_rng_uses_integer_device_ids() -> None:
    assert _cuda_rng_devices(torch.device("cpu")) == []
    assert _cuda_rng_devices(torch.device("cuda:2")) == [2]
    with pytest.raises(ValueError, match="explicit index"):
        _cuda_rng_devices(torch.device("cuda"))


def test_stack_targets_preserves_geometry_and_rejects_empty() -> None:
    first, _, _ = _targets(clean=True)
    second, _, _ = _targets(clean=False)
    stacked = _stack_targets(
        (first, second),
        device=torch.device("cpu"),
    )
    stacked.validate()
    assert stacked.target_field_plus.shape == (2, 1, 8, 8)
    assert torch.equal(
        stacked.target_field_minus[:1],
        first.target_field_minus,
    )
    assert torch.equal(
        stacked.target_field_minus[1:],
        second.target_field_minus,
    )
    with pytest.raises(ValueError, match="nonempty"):
        _stack_targets((), device=torch.device("cpu"))


def test_cpu_probe_exercises_split_view_gradient_without_mutation() -> None:
    population = _FakePopulation()
    before_rng = torch.random.get_rng_state().clone()
    geometry = _geometry(population)
    probe = _probe(population, device=torch.device("cpu"))

    assert geometry["valid_domain_is_full_output"] is True
    assert geometry["valid_domain_nonempty_per_state"] is True
    assert geometry["clean_geometry_contract"] is True
    assert geometry["component_geometry_contract"] is True
    assert geometry["integration_measure_mass_one"] is True
    assert geometry["target_and_occupancy_inside_valid"] is True
    assert geometry["joint_valid_matches_record"] is True
    assert geometry["output_matches_feature_stride"] is True
    assert geometry["memory_plan"]["pair_batch_count"] == 2
    assert geometry["memory_plan"]["endpoint_batch_count"] == 4
    assert probe["model_forward_invocations"] == 1
    assert probe["field_and_gradients_finite"] is True
    assert probe["field_gradient_nonzero"] is True
    assert probe["scalar_energy_gradient_nonzero"] is True
    assert probe["parameter_grad_buffers_unretained"] is True
    assert (
        probe["initial_model_fingerprint"]
        == probe["final_model_fingerprint"]
    )
    assert probe["runtime_splits"] == ["D_R"]
    assert probe["split_access_evidence_policy"] == (
        "single_verified_real_input_graph_and_bounded_population_only"
    )
    assert probe["global_cpu_rng_preserved"] is True
    assert probe["selected_device_rng_preserved"] is True
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert len(probe["rows"]) == 2
    clean_row = next(
        row
        for row in probe["rows"]
        if row["optimizer_role"] == "clean_positive"
    )
    assert float.fromhex(clean_row["gamma_hex"]) > 0.0
    assert float.fromhex(clean_row["loss_hex"]) > 0.0
    assert clean_row["gamma_equals_global_endpoint_amax"] is True
    assert clean_row["violation_zero_outside_declared_V"] is True
    assert (
        float.fromhex(
            clean_row[
                "target_orthant_descent_alignment_hex"
            ]
        )
        > 0.0
    )
    assert (
        float.fromhex(
            clean_row[
                "full_tensor_orthant_descent_alignment_hex"
            ]
        )
        > 0.0
    )


def _passing_recompute_inputs() -> tuple[object, object, object, dict, dict]:
    scalar_cache = object()
    dataset_free = SimpleNamespace(all_pass=True)
    real_inputs = SimpleNamespace(
        bundle=SimpleNamespace(split="D_R"),
        source_binding=SimpleNamespace(
            split="D_R",
            dataset="IRSTD-1K",
        ),
        scalar_cache=scalar_cache,
    )
    population = SimpleNamespace(
        source_cache=scalar_cache,
        seed=COVERAGE_STATE_BOUNDED_SEED,
        population_fingerprint="b" * 64,
    )
    geometry = {
        "clean_positive_count": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "component_null_count": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "optimized_pair_count": 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "valid_domain_is_full_output": True,
        "valid_domain_nonempty_per_state": True,
        "target_fields_strictly_nonzero_on_valid": True,
        "endpoint_valid_pixel_count": 1024,
        "clean_geometry_contract": True,
        "component_geometry_contract": True,
        "integration_measure_mass_one": True,
        "integration_mass_rtol_hex": (
            COVERAGE_STATE_USCOPE_DR_MASS_RTOL.hex()
        ),
        "integration_mass_atol_hex": (
            COVERAGE_STATE_USCOPE_DR_MASS_ATOL.hex()
        ),
        "target_and_occupancy_inside_valid": True,
        "joint_valid_matches_record": True,
        "output_matches_feature_stride": True,
    }
    rows = [
        {
            "pair_id": f"clean-{index:02d}",
            "optimizer_role": "clean_positive",
            "gamma_hex": float(1.0).hex(),
            "gamma_plus_hex": float(0.25).hex(),
            "gamma_minus_hex": float(1.0).hex(),
            "loss_hex": float(0.5).hex(),
            "gamma_equals_global_endpoint_amax": True,
            "violation_zero_outside_declared_V": True,
            "target_orthant_descent_alignment_hex": (
                float(0.75).hex()
            ),
            "background_orthant_descent_alignment_hex": (
                float(0.0).hex()
            ),
            "full_tensor_orthant_descent_alignment_hex": (
                float(0.75).hex()
            ),
        }
        for index in range(COVERAGE_STATE_BOUNDED_ROLE_COUNT)
    ] + [
        {
            "pair_id": f"component-{index:02d}",
            "optimizer_role": "component_null",
            "gamma_hex": float(0.0).hex(),
            "gamma_plus_hex": float(0.0).hex(),
            "gamma_minus_hex": float(0.0).hex(),
            "loss_hex": float(0.0).hex(),
            "gamma_equals_global_endpoint_amax": True,
            "violation_zero_outside_declared_V": True,
            "target_orthant_descent_alignment_hex": (
                float(0.0).hex()
            ),
            "background_orthant_descent_alignment_hex": (
                float(0.0).hex()
            ),
            "full_tensor_orthant_descent_alignment_hex": (
                float(0.0).hex()
            ),
        }
        for index in range(COVERAGE_STATE_BOUNDED_ROLE_COUNT)
    ]
    geometry_rows = [
        {
            "pair_id": row["pair_id"],
            "integration_mass_one": True,
            "pair_targets_validate_passed": True,
            "integration_mass_hex": [float(1.0).hex()],
            "target_and_occupancy_inside_valid": True,
            "joint_valid_matches_record": True,
        }
        for row in rows
    ]
    geometry["geometry_rows"] = geometry_rows
    geometry["optimized_pair_ids"] = [
        row["pair_id"] for row in rows
    ]
    geometry["memory_plan"] = {
        "pair_batch_count": 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "endpoint_batch_count": 4 * COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "dominant_phase_width_tensor_bytes": 1024,
        "five_named_forward_tensors_bytes": 5120,
        "estimate_is_not_measured_peak": True,
        "execution_policy": (
            "one_batch_32_pairs_64_endpoints_one_model_forward"
        ),
    }
    probe = {
        "execution_seed": COVERAGE_STATE_USCOPE_DR_EXECUTION_SEED,
        "runtime_splits": ["D_R"],
        "split_access_evidence_policy": (
            "single_verified_real_input_graph_and_bounded_population_only"
        ),
        "single_batched_model_forward": True,
        "model_forward_invocations": 1,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "rows": rows,
        "field_and_gradients_finite": True,
        "field_gradient_nonzero": True,
        "scalar_energy_gradient_finite": True,
        "scalar_energy_gradient_nonzero": True,
        "initial_model_fingerprint": "c" * 64,
        "final_model_fingerprint": "c" * 64,
        "population_fingerprint_before": "b" * 64,
        "population_fingerprint_after": "b" * 64,
        "parameter_grad_buffers_unretained": True,
        "global_cpu_rng_preserved": True,
        "selected_device_rng_preserved": True,
        "model_config": {
            "feature_channels": (
                COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS
            ),
            "feature_stride": (
                COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE
            ),
            "width": COVERAGE_STATE_CMIF_FORMAL_WIDTH,
            "parameter_count": (
                COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT
            ),
            "objective_policy": CSLF_USCOPE_POLICY,
        },
    }
    return dataset_free, real_inputs, population, geometry, probe


def test_recompute_requires_exact_dr_read_only_scope() -> None:
    (
        dataset_free,
        real_inputs,
        population,
        geometry,
        probe,
    ) = _passing_recompute_inputs()
    checks = dict(
        recompute_coverage_state_uscope_dr_checks(
            dataset_free_receipt=dataset_free,
            real_inputs=real_inputs,
            population=population,
            geometry=geometry,
            probe=probe,
        )
    )
    assert checks
    assert all(checks.values())

    # V may exclude padding.  Full-output coverage is an observation rather
    # than a prerequisite, while every declared V must remain nonempty.
    geometry["valid_domain_is_full_output"] = False
    cropped_checks = dict(
        recompute_coverage_state_uscope_dr_checks(
            dataset_free_receipt=dataset_free,
            real_inputs=real_inputs,
            population=population,
            geometry=geometry,
            probe=probe,
        )
    )
    assert all(cropped_checks.values())

    probe["runtime_splits"] = ["D_R", "D_V"]
    changed = dict(
        recompute_coverage_state_uscope_dr_checks(
            dataset_free_receipt=dataset_free,
            real_inputs=real_inputs,
            population=population,
            geometry=geometry,
            probe=probe,
        )
    )
    assert changed["read_only_execution"] is False


def test_recompute_rejects_mass_gamma_direction_and_pair_coverage_drift() -> None:
    def recompute(
        dataset_free: object,
        real_inputs: object,
        population: object,
        geometry: dict,
        probe: dict,
    ) -> dict[str, bool]:
        return dict(
            recompute_coverage_state_uscope_dr_checks(
                dataset_free_receipt=dataset_free,
                real_inputs=real_inputs,
                population=population,
                geometry=geometry,
                probe=probe,
            )
        )

    values = _passing_recompute_inputs()
    values[3]["geometry_rows"][0]["integration_mass_one"] = False
    assert recompute(*values)[
        "pair_target_contracts_valid_on_declared_V"
    ] is False

    values = _passing_recompute_inputs()
    values[4]["rows"][0]["gamma_hex"] = float(0.75).hex()
    assert recompute(*values)[
        "gamma_is_exact_global_endpoint_amax"
    ] is False

    values = _passing_recompute_inputs()
    values[4]["rows"][0][
        "target_orthant_descent_alignment_hex"
    ] = float(-0.25).hex()
    assert recompute(*values)[
        "orthant_descent_direction_audited"
    ] is False

    values = _passing_recompute_inputs()
    values[4]["rows"].pop()
    assert recompute(*values)[
        "clean_and_component_pairs_covered_exactly_once"
    ] is False


def test_receipt_fingerprint_is_revalidated_not_cached() -> None:
    assert isinstance(
        CoverageStateUSCOPEDRGateReceipt.receipt_fingerprint,
        property,
    )
    assert isinstance(
        CoverageStateUSCOPEDRGateReceipt.all_pass,
        property,
    )
