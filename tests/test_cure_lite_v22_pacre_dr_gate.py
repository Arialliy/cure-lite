from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cure_lite.cache.schema import canonical_json, stable_fingerprint
from cure_lite_v22.dataset_free import run_pacre_dataset_free_gate
from cure_lite_v22.dr_gate import (
    PACRE_DR_CHECK_NAMES,
    PACRE_DR_PASS_DECISION,
    CoverageStatePACREDRGateReceipt,
    _algebra_checks,
    _bound_pacre_pair_witness,
    _implementation_binding,
    _validate_dataset_free_receipt,
    recompute_pacre_dr_checks,
)
from cure_lite_v22.factory import build_pacre_training_model
from cure_lite_v22.pacre import (
    CoverageStatePACREConfig,
    CoverageStatePACREFields,
)
from tests.test_coverage_state_bfa_dr_gate import _ProbePopulation


def _model_and_inputs():
    torch.manual_seed(220030)
    model = build_pacre_training_model(
        CoverageStatePACREConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    generator = torch.Generator().manual_seed(220031)
    feature = torch.randn(
        (1, 2, 3, 4),
        generator=generator,
        dtype=torch.float32,
    )
    occupancy = (
        torch.rand(
            (1, 1, 6, 8),
            generator=generator,
            dtype=torch.float32,
        )
        > 0.7
    )
    return model, feature, occupancy


def test_pacre_fields_satisfy_gate_algebra_at_zero_readout() -> None:
    model, feature, occupancy = _model_and_inputs()
    fields = model.forward_fields(feature, occupancy)

    assert type(fields) is CoverageStatePACREFields
    assert _algebra_checks(fields)
    assert torch.count_nonzero(fields.native_phase_interaction) == 0
    assert torch.equal(
        fields.field,
        torch.full_like(fields.field, model.config.field_amplitude),
    )
    latent = 0.5 * (
        fields.actual_compatibility_hidden
        - fields.flipped_compatibility_hidden
    )
    assert bool(torch.any(latent != 0.0))


def test_bound_witness_requires_one_same_cell_jointly_separated_pair() -> None:
    shape = (1, 3, 4, 4)
    residual = torch.zeros(shape, dtype=torch.float32)
    actual = torch.zeros(shape, dtype=torch.float32)
    flipped = torch.zeros(shape, dtype=torch.float32)
    latent = torch.zeros(shape, dtype=torch.float32)
    target = torch.zeros((1, 1, 4, 4), dtype=torch.bool)
    background = torch.zeros_like(target)
    target[:, :, 1, 1] = True
    background[:, :, 1, 2] = True
    residual[0, :, 1, 1] = torch.tensor([1.0, 0.0, 0.0])
    residual[0, :, 1, 2] = torch.tensor([0.0, 1.0, 0.0])
    actual[0, :, 1, 1] = torch.tensor([1.0, 1.0, 0.0])
    flipped[0, :, 1, 1] = torch.tensor([0.0, 1.0, 1.0])
    latent[0, :, 1, 1] = torch.tensor([0.5, 0.0, -0.5])
    latent[0, :, 1, 2] = torch.tensor([0.0, 0.5, -0.5])

    witness = _bound_pacre_pair_witness(
        residual,
        actual,
        flipped,
        latent,
        target_mask=target,
        background_mask=background,
        stride=4,
    )
    assert witness["legal_pair_count"] == 1
    assert witness["jointly_separated_pair_count"] == 1
    assert witness["at_least_one_jointly_separated_pair"] is True
    selected = witness["selected_first_joint_witness"]
    assert isinstance(selected, dict)
    assert selected["target_phase"] == 5
    assert selected["background_phase"] == 6
    assert selected["coarse_cell"] == [0, 0, 0]

    latent[:, :, 1, 2] = latent[:, :, 1, 1]
    failed = _bound_pacre_pair_witness(
        residual,
        actual,
        flipped,
        latent,
        target_mask=target,
        background_mask=background,
        stride=4,
    )
    assert failed["legal_pair_count"] == 1
    assert failed["jointly_separated_pair_count"] == 0
    assert failed["at_least_one_jointly_separated_pair"] is False


def test_dataset_free_binding_rejects_any_receipt_drift() -> None:
    receipt = run_pacre_dataset_free_gate()
    fingerprint = _validate_dataset_free_receipt(receipt)
    assert fingerprint == receipt["receipt_fingerprint"]

    changed = dict(receipt)
    changed["gate_passed"] = False
    with pytest.raises(PermissionError, match="prerequisite"):
        _validate_dataset_free_receipt(changed)

    rebound = dict(receipt)
    implementation = [
        dict(row) for row in rebound["implementation_binding"]
    ]
    implementation[0]["sha256"] = "0" * 64
    rebound["implementation_binding"] = implementation
    rebound.pop("receipt_fingerprint")
    rebound["receipt_fingerprint"] = stable_fingerprint(rebound)
    with pytest.raises(PermissionError, match="prerequisite"):
        _validate_dataset_free_receipt(rebound)


def _passing_probe() -> dict[str, object]:
    return {
        "model_fqcn": (
            "cure_lite_v22.pacre."
            "CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet"
        ),
        "config_fqcn": (
            "cure_lite_v22.pacre.CoverageStatePACREConfig"
        ),
        "model_contract": {
            "model_class": (
                "cure_lite_v22.pacre."
                "CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet"
            ),
            "config_class": (
                "cure_lite_v22.pacre.CoverageStatePACREConfig"
            ),
            "parameter_count": 64064,
            "config": {
                "feature_channels": 64,
                "feature_stride": 4,
                "width": 32,
                "field_policy": (
                    "phase_aligned_centered_residual_compatibility_"
                    "binary_flip_field_v1"
                ),
                "equation_policy": (
                    "phase_common_operating_point_specific_residual_"
                    "shared_silu_energy_binary_odd_projection_v1"
                ),
                "centering_policy": (
                    "exact_per_cell_hidden_channel_phase_mean_"
                    "quotient_v1"
                ),
            },
            "parameter_shapes": {
                "joint_hidden_bias": [32],
                "joint_state_weight": [32, 80, 5, 5],
                "scalar_energy_weight": [32],
            },
        },
        "model_contract_fingerprint": "",
        "initial_model_fingerprint": "1" * 64,
        "final_model_fingerprint": "1" * 64,
        "parameter_ids_preserved": True,
        "representation": {
            "target_group_count": 32,
            "target_forward_calls": 32,
            "context_state_count": 96,
            "context_forward_calls": 96,
            "all_fields_exact_pacre": True,
            "all_algebra_checks_passed": True,
            "all_target_groups_have_joint_witness": True,
            "exact_latent_collision_count": 0,
            "zero_readout_anchor_all_target_states": True,
            "fixed_readout_interaction_nonzero": True,
        },
        "gradient_path": {
            "initial_gradient_finite": {
                "joint_hidden_bias": True,
                "joint_state_weight": True,
                "scalar_energy_weight": True,
            },
            "initial_gradient_nonzero": {
                "joint_hidden_bias": False,
                "joint_state_weight": False,
                "scalar_energy_weight": True,
            },
            "readout_visible_upstream_dormant": True,
            "readout_to_upstream_cross_gradient_finite_nonzero": [
                True,
                True,
            ],
            "parameter_grad_buffers_unretained": True,
        },
        "field_direction": {
            "all_roles_finite_nonzero_correct": True,
        },
        "population_fingerprint_before": "2" * 64,
        "population_fingerprint_after": "2" * 64,
        "cache_fingerprint_before": "3" * 64,
        "cache_fingerprint_after": "3" * 64,
        "global_cpu_rng_preserved": True,
        "selected_device_rng_preserved": True,
        "deterministic_execution": {
            "restored_exactly": True,
        },
        "parameter_grad_buffers_unretained": True,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_performed": False,
        "D_R_accessed": True,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }


def test_check_recomputation_and_receipt_are_fail_closed() -> None:
    dataset_free_receipt = run_pacre_dataset_free_gate()
    dataset_free_fingerprint = dataset_free_receipt[
        "receipt_fingerprint"
    ]
    scalar_cache = SimpleNamespace(
        raw_catalog=SimpleNamespace(split="D_R"),
        cache_fingerprint="4" * 64,
    )
    real_inputs = SimpleNamespace(
        source_binding=SimpleNamespace(split="D_R"),
        scalar_cache=scalar_cache,
        build_fingerprint="6" * 64,
        verify_unchanged=lambda: None,
    )
    population = SimpleNamespace(
        seed=42,
        source_cache=scalar_cache,
        source_cache_fingerprint=scalar_cache.cache_fingerprint,
        cache=scalar_cache,
        population_fingerprint="7" * 64,
        verify_unchanged=lambda: None,
    )
    probe = _passing_probe()
    probe["model_contract_fingerprint"] = stable_fingerprint(
        probe["model_contract"]
    )
    checks = recompute_pacre_dr_checks(
        dataset_free_receipt_fingerprint=dataset_free_fingerprint,
        real_inputs=real_inputs,
        bounded_population=population,
        probe=probe,
    )
    assert tuple(name for name, _ in checks) == PACRE_DR_CHECK_NAMES
    assert all(value for _, value in checks)

    receipt = CoverageStatePACREDRGateReceipt(
        dataset_free_receipt_fingerprint=dataset_free_fingerprint,
        real_inputs_fingerprint="6" * 64,
        population_fingerprint="7" * 64,
        cache_fingerprint="4" * 64,
        implementation_binding=_implementation_binding(),
        probe_json=canonical_json(probe),
        checks=checks,
    )
    assert receipt.gate_passed
    assert receipt.decision == PACRE_DR_PASS_DECISION
    assert len(receipt.receipt_fingerprint) == 64
    assert receipt.canonical_payload()["D_V_accessed"] is False
    receipt.verify_unchanged(
        dataset_free_receipt=dataset_free_receipt,
        real_inputs=real_inputs,
        bounded_population=population,
    )

    failed_probe = dict(probe)
    failed_probe["D_V_accessed"] = True
    failed = recompute_pacre_dr_checks(
        dataset_free_receipt_fingerprint=dataset_free_fingerprint,
        real_inputs=real_inputs,
        bounded_population=population,
        probe=failed_probe,
    )
    assert dict(failed)["12_read_only_zero_update_D_R_scope"] is False

    forged = CoverageStatePACREDRGateReceipt(
        dataset_free_receipt_fingerprint=dataset_free_fingerprint,
        real_inputs_fingerprint="6" * 64,
        population_fingerprint="7" * 64,
        cache_fingerprint="4" * 64,
        implementation_binding=_implementation_binding(),
        probe_json=canonical_json(probe),
        checks=tuple(
            (name, False if index == 8 else value)
            for index, (name, value) in enumerate(checks)
        ),
    )
    with pytest.raises(RuntimeError, match="checks changed"):
        forged.verify_unchanged(
            dataset_free_receipt=dataset_free_receipt,
            real_inputs=real_inputs,
            bounded_population=population,
        )


def test_complete_generated_probe_is_json_safe_deterministic_and_read_only() -> None:
    population = _ProbePopulation()
    before_rng = torch.random.get_rng_state().clone()

    from cure_lite_v22.dr_gate import _probe

    first = _probe(population, device=torch.device("cpu"))
    second = _probe(population, device=torch.device("cpu"))

    assert first == second
    assert isinstance(
        first["gradient_path"]["readout_visible_upstream_dormant"],
        bool,
    )
    assert (
        first["field_direction"][
            "all_roles_finite_nonzero_correct"
        ]
        is True
    )
    assert isinstance(canonical_json(first), str)
    assert first["initial_model_fingerprint"] == (
        first["final_model_fingerprint"]
    )
    assert first["parameter_ids_preserved"] is True
    assert first["deterministic_execution"]["restored_exactly"] is True
    assert first["optimizer_constructed"] is False
    assert first["optimizer_steps"] == 0
    assert first["parameter_updates"] == 0
    assert first["training_performed"] is False
    assert first["D_R_accessed"] is True
    assert first["D_V_accessed"] is False
    assert first["D_T_accessed"] is False
    assert torch.equal(before_rng, torch.random.get_rng_state())
