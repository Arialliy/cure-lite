from __future__ import annotations

from types import SimpleNamespace

import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.coverage_state_binary_flip_antisymmetric import (
    CURELiteBinaryFlipAntisymmetricLevelSet,
    CoverageStateBinaryFlipAntisymmetricConfig,
)
from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
    CoverageStatePhaseAlignedEvidenceTransportConfig,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_ROLE_COUNT,
)
from cure_lite.experiment.coverage_state_bfa_dr_gate import _state_specs
from cure_lite.experiment.coverage_state_cmif_dataset_free import (
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_CMIF_FORMAL_WIDTH,
)
from cure_lite.experiment.coverage_state_paet_dataset_free import (
    COVERAGE_STATE_PAET_MARGIN,
)
from cure_lite.experiment.coverage_state_paet_dr_gate import (
    COVERAGE_STATE_PAET_DR_EXECUTION_SEED,
    COVERAGE_STATE_PAET_DR_FAIL_DECISION,
    COVERAGE_STATE_PAET_DR_PASS_DECISION,
    COVERAGE_STATE_PAET_DR_SEPARATION_THRESHOLD,
    PAET_DR_BOUND_PHASE_PAIR_CHECK,
    PAET_DR_CONFLICT_CHECK,
    PAET_DR_GRADIENT_CHECK,
    CoverageStatePAETDRGateReceipt,
    _bound_phase_pair_witness,
    _deterministic_execution_scope,
    _determinism_flags,
    _gradient_probe,
    _no_transport_bfa_common_odd_hidden,
    _phase_hidden_to_output,
    _probe,
    _representation_probe,
    recompute_coverage_state_paet_dr_checks,
)
from tests.test_coverage_state_bfa_dr_gate import (
    _ProbePopulation,
    _population,
)


def _model() -> CURELitePhaseAlignedEvidenceTransportLevelSet:
    return CURELitePhaseAlignedEvidenceTransportLevelSet(
        CoverageStatePhaseAlignedEvidenceTransportConfig(
            feature_channels=2,
            feature_stride=4,
            width=3,
        )
    )


def test_phase_hidden_to_output_keeps_canonical_phase_coordinates() -> None:
    native = torch.arange(
        1 * 16 * 3 * 2 * 2,
        dtype=torch.float32,
    ).reshape(1, 16, 3, 2, 2)
    output = _phase_hidden_to_output(native, stride=4)

    assert output.shape == (1, 3, 8, 8)
    for row in range(8):
        for column in range(8):
            phase = (row % 4) * 4 + column % 4
            assert torch.equal(
                output[0, :, row, column],
                native[0, phase, :, row // 4, column // 4],
            )


def test_no_transport_counterfactual_is_exact_bfa_common_hidden() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        paet = _model()
        bfa = CURELiteBinaryFlipAntisymmetricLevelSet(
            CoverageStateBinaryFlipAntisymmetricConfig(
                feature_channels=2,
                feature_stride=4,
                width=3,
            )
        )
        bfa.load_state_dict(paet.state_dict(), strict=True)
        feature = torch.randn(1, 2, 2, 2, dtype=torch.float32)
        occupancy = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
        occupancy[:, :, 3, 5] = True
        paet_fields = paet.forward_fields(feature, occupancy)
        bfa_fields = bfa.forward_fields(feature, occupancy)

    assert torch.equal(
        _no_transport_bfa_common_odd_hidden(paet_fields),
        bfa_fields.odd_feature_presence_hidden,
    )


def test_internal_deterministic_scope_restores_flags_on_error() -> None:
    before = _determinism_flags()
    try:
        with _deterministic_execution_scope() as ledger:
            assert ledger["active"] == {
                "deterministic_algorithms": True,
                "deterministic_warn_only": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cudnn_allow_tf32": False,
                "cuda_matmul_allow_tf32": False,
            }
            raise RuntimeError("intentional")
    except RuntimeError as error:
        assert str(error) == "intentional"
    else:
        raise AssertionError("deterministic-scope error was swallowed")
    assert _determinism_flags() == before
    assert ledger["after"] == before
    assert ledger["restored_exactly"] is True


def test_bound_pair_uses_complete_background_mask_and_same_coarse_cell() -> None:
    phase = torch.zeros(1, 2, 8, 8, dtype=torch.float32)
    hidden = torch.zeros_like(phase)
    common = torch.zeros_like(phase)
    target = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    valid = torch.ones_like(target)
    occupancy = torch.zeros_like(target)
    full_target = torch.zeros_like(target)
    target[:, :, 3, 3] = True
    full_target[:, :, 2:4, 2:4] = True
    background = valid & ~occupancy & ~full_target
    old_incomplete_legal = valid & ~occupancy & ~target

    # The old valid&~occupancy&~focus rule incorrectly calls (3,2) legal,
    # whereas the complete state.background_mask correctly excludes it.
    assert bool(old_incomplete_legal[:, :, 3, 2])
    assert not bool(background[:, :, 3, 2])
    # Every remaining adjacent background is across a coarse-cell boundary.
    phase[:, :, 3, 3] = torch.tensor([1.0, 0.0])
    hidden[:, :, 3, 3] = torch.tensor([1.0, 0.0])
    common[:, :, 3, 3] = torch.tensor([0.0, 1.0])
    phase[:, :, 3, 2] = torch.tensor([0.0, 2.0])
    hidden[:, :, 3, 2] = torch.tensor([0.0, 2.0])
    phase[:, :, 4, 3] = torch.tensor([0.0, 3.0])
    hidden[:, :, 4, 3] = torch.tensor([0.0, 3.0])

    rejected = _bound_phase_pair_witness(
        phase,
        hidden,
        common,
        target_mask=target,
        background_mask=background,
        stride=4,
    )
    assert rejected["pair_count"] == 0
    assert rejected["target_coordinate_count"] == 1
    assert rejected["target_coordinates_with_legal_background_q"] == 0
    assert rejected["target_coordinates_without_legal_background_q"] == 1
    assert (
        rejected["at_least_one_bound_pair_jointly_separated"]
        is False
    )

    # Only this true background phase is both adjacent and in the same cell.
    full_target[:, :, 2, 3] = False
    background = valid & ~occupancy & ~full_target
    phase[:, :, 2, 3] = torch.tensor([0.0, 1.0])
    hidden[:, :, 2, 3] = torch.tensor([0.0, 1.0])
    accepted = _bound_phase_pair_witness(
        phase,
        hidden,
        common,
        target_mask=target,
        background_mask=background,
        stride=4,
    )
    assert accepted["pair_count"] == 1
    assert accepted["target_coordinate_count"] == 1
    assert accepted["target_coordinates_with_legal_background_q"] == 1
    assert accepted["target_coordinates_without_legal_background_q"] == 0
    assert accepted["jointly_separated_pair_count"] == 1
    assert accepted[
        "at_least_one_bound_pair_jointly_separated"
    ] is True
    pair = accepted["pair_rows"][0]
    assert pair["target_coordinate"] == [0, 3, 3]
    assert pair["background_coordinate"] == [0, 2, 3]
    assert pair["coarse_cell"] == [0, 0, 0]
    for key in (
        "phase_feature_separation_hex",
        "transported_odd_hidden_separation_hex",
        "target_odd_vs_no_transport_common_separation_hex",
    ):
        assert (
            float.fromhex(pair[key])
            > COVERAGE_STATE_PAET_DR_SEPARATION_THRESHOLD
        )

    phase[:, :, 3, 3] = torch.tensor(
        [COVERAGE_STATE_PAET_DR_SEPARATION_THRESHOLD, 0.0]
    )
    phase[:, :, 2, 3] = 0.0
    boundary = _bound_phase_pair_witness(
        phase,
        hidden,
        common,
        target_mask=target,
        background_mask=background,
        stride=4,
    )
    assert boundary["jointly_separated_pair_count"] == 0
    assert (
        boundary["at_least_one_bound_pair_jointly_separated"]
        is False
    )


def test_multpixel_group_allows_interior_target_without_direct_q() -> None:
    grid = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    phase = torch.cat((grid + 1.0, 2.0 * grid + 3.0), dim=1)
    hidden = torch.cat((3.0 * grid + 1.0, grid + 2.0), dim=1)
    common = torch.zeros_like(hidden)
    target = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    target[:, :, 1:4, 1:4] = True
    background = ~target

    witness = _bound_phase_pair_witness(
        phase,
        hidden,
        common,
        target_mask=target,
        background_mask=background,
        stride=4,
    )

    assert witness["target_coordinate_count"] == 9
    assert 0 < witness[
        "target_coordinates_with_legal_background_q"
    ] < 9
    assert witness[
        "target_coordinates_without_legal_background_q"
    ] > 0
    assert (
        witness["target_coordinates_with_legal_background_q"]
        + witness["target_coordinates_without_legal_background_q"]
        == 9
    )
    assert witness["jointly_separated_pair_count"] > 0
    assert witness[
        "at_least_one_bound_pair_jointly_separated"
    ] is True


def test_representation_probe_exposes_all_paet_specific_witnesses() -> None:
    population = _population(count=1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        probe = _representation_probe(
            _model(),
            population,
            device=torch.device("cpu"),
        )
    target_states, _ = _state_specs(population)
    state_by_id = {state.state_id: state for state in target_states}

    assert probe["coordinate_policy"] == {
        "bound_phase_pair_check": PAET_DR_BOUND_PHASE_PAIR_CHECK,
        "necessary_exact_collision_check": PAET_DR_CONFLICT_CHECK,
        "target_interval": "d_le_negative_1p125",
        "background_interval": "d_ge_negative_0p675",
        "component_interval": "abs_d_le_0p675",
    }
    assert probe["state_counts"] == {
        "target_states": 2,
        "context_states": 6,
        "factual_target_groups": 1,
        "clean_target_groups": 1,
        "target_pass_forward_count": 2,
        "positive_pass_forward_count": 6,
        "total_forward_count": 8,
        "target_states_reforwarded_in_positive_pass": 2,
        "two_pass_streaming_no_full_map_retention": True,
    }
    assert probe["coordinate_counts"]["target"] == 2
    assert probe["coordinate_counts"]["background"] > 0
    assert probe["coordinate_counts"]["component"] > 0
    assert len(probe["target_group_rows"]) == 2
    for row in probe["target_group_rows"]:
        witness = row["bound_phase_pair_witness"]
        assert witness["pair_count"] > 0
        assert witness[
            "at_least_one_bound_pair_jointly_separated"
        ] is True
        assert witness["jointly_separated_pair_count"] > 0
        assert set(witness["separation_distributions"]) == {
            "phase_feature_p_vs_q",
            "transported_odd_hidden_p_vs_q",
            "target_odd_vs_no_transport_common",
        }
        for pair in witness["pair_rows"]:
            assert pair["legal_background_from_state_mask"] is True
            assert pair["chebyshev_distance"] == 1
            assert pair["target_coordinate"][0] == (
                pair["background_coordinate"][0]
            )
            assert pair["coarse_cell"] == [
                pair["target_coordinate"][0],
                pair["target_coordinate"][1] // 4,
                pair["target_coordinate"][2] // 4,
            ]
            assert pair["coarse_cell"] == [
                pair["background_coordinate"][0],
                pair["background_coordinate"][1] // 4,
                pair["background_coordinate"][2] // 4,
            ]
            batch, row_index, column_index = pair[
                "background_coordinate"
            ]
            assert bool(
                state_by_id[row["state_id"]].background_mask[
                    batch,
                    0,
                    row_index,
                    column_index,
                ]
            )
    # The synthetic fixture intentionally reuses one clean/component state;
    # the gate must expose, not hide, the resulting exact conflict.
    assert probe["necessary_exact_collision_count"] > 0
    assert probe["collision_examples"]
    assert (
        probe["exact_collision_zero_is_readout_feasibility_proof"]
        is False
    )


def test_gradient_probe_proves_zero_then_nonzero_readout_paths() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(456)
        model = _model()
        before = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
        probe = _gradient_probe(
            model,
            _population(count=1),
            device=torch.device("cpu"),
        )

    assert probe["policy"] == PAET_DR_GRADIENT_CHECK
    assert probe["zero_readout_scalar_path_finite_nonzero"] is True
    assert probe["zero_readout_upstream_gradients_exactly_zero"] is True
    assert (
        probe["fixed_nonzero_readout_all_three_finite_nonzero"]
        is True
    )
    assert (
        probe["functional_call_did_not_replace_model_parameters"]
        is True
    )
    assert set(probe["fixed_nonzero_readout_gradients"]) == {
        "joint_state_weight",
        "joint_hidden_bias",
        "scalar_energy_weight",
    }
    for row in probe["fixed_nonzero_readout_gradients"].values():
        assert row["finite"] is True
        assert row["nonzero"] is True
        assert float.fromhex(row["l2_hex"]) > 0.0
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_complete_toy_probe_is_deterministic_read_only_and_zero_step() -> None:
    population = _ProbePopulation()
    before_rng = torch.random.get_rng_state().clone()
    before_determinism = _determinism_flags()
    first = _probe(population, device=torch.device("cpu"))
    second = _probe(population, device=torch.device("cpu"))

    assert first == second
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert _determinism_flags() == before_determinism
    assert first["initial_model_fingerprint"] == (
        first["final_model_fingerprint"]
    )
    assert first["population_fingerprint_before"] == (
        first["population_fingerprint_after"]
    )
    assert first["cache_fingerprint_before"] == (
        first["cache_fingerprint_after"]
    )
    assert first["representation"]["state_counts"][
        "target_pass_forward_count"
    ] == 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
    assert first["representation"]["state_counts"][
        "positive_pass_forward_count"
    ] == 6 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
    assert first["representation"]["state_counts"][
        "two_pass_streaming_no_full_map_retention"
    ] is True
    determinism = first["deterministic_execution"]
    assert determinism["active"] == {
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_allow_tf32": False,
        "cuda_matmul_allow_tf32": False,
    }
    assert determinism["after"] == determinism["before"]
    assert determinism["restored_exactly"] is True
    assert first["optimizer_constructed"] is False
    assert first["optimizer_steps"] == 0
    assert first["parameter_updates"] == 0
    assert first["training_performed"] is False
    assert first["D_V_accessed"] is False
    assert first["D_T_accessed"] is False
    assert first["parameter_grad_buffers_unretained"] is True


def _passing_bound_witness() -> dict[str, object]:
    phase = torch.zeros(1, 2, 8, 8, dtype=torch.float32)
    hidden = torch.zeros_like(phase)
    common = torch.zeros_like(phase)
    target = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    background = torch.zeros_like(target)
    target[:, :, 3, 3] = True
    background[:, :, 2, 3] = True
    phase[:, :, 3, 3] = torch.tensor([1.0, 0.0])
    phase[:, :, 2, 3] = torch.tensor([0.0, 1.0])
    hidden[:, :, 3, 3] = torch.tensor([1.0, 0.0])
    hidden[:, :, 2, 3] = torch.tensor([0.0, 1.0])
    common[:, :, 3, 3] = torch.tensor([0.0, 1.0])
    return _bound_phase_pair_witness(
        phase,
        hidden,
        common,
        target_mask=target,
        background_mask=background,
        stride=4,
    )


def _passing_multpixel_bound_witness() -> dict[str, object]:
    grid = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    phase = torch.cat((grid + 1.0, 2.0 * grid + 3.0), dim=1)
    hidden = torch.cat((3.0 * grid + 1.0, grid + 2.0), dim=1)
    target = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    target[:, :, 1:4, 1:4] = True
    return _bound_phase_pair_witness(
        phase,
        hidden,
        torch.zeros_like(hidden),
        target_mask=target,
        background_mask=~target,
        stride=4,
    )


def _passing_recompute_inputs() -> tuple[object, object, object, dict]:
    dataset_free = SimpleNamespace(all_pass=True)
    scalar_cache = SimpleNamespace(
        raw_catalog=SimpleNamespace(split="D_R")
    )
    real_inputs = SimpleNamespace(
        source_binding=SimpleNamespace(
            dataset="IRSTD-1K",
            split="D_R",
        ),
        bundle=SimpleNamespace(split="D_R"),
        raw_catalog=SimpleNamespace(split="D_R"),
        scalar_cache=scalar_cache,
    )
    population = SimpleNamespace(
        source_cache=scalar_cache,
        cache=SimpleNamespace(
            raw_catalog=SimpleNamespace(split="D_R"),
            cache_fingerprint="b" * 64,
        ),
        seed=COVERAGE_STATE_PAET_DR_EXECUTION_SEED,
        population_fingerprint="c" * 64,
    )
    target_rows = [
        {
            "target_group_id": f"group-{index:02d}",
            "coordinate_count": 1,
            "transported_representation_finite": True,
            "bound_phase_pair_witness": _passing_bound_witness(),
        }
        for index in range(2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT)
    ]
    target_rows[0]["coordinate_count"] = 9
    target_rows[0][
        "bound_phase_pair_witness"
    ] = _passing_multpixel_bound_witness()
    direction_rows = [
        {
            "loss_hex": float(1.0).hex(),
            "descent_sum_hex": float(1.0).hex(),
            "descent_finite": True,
            "descent_nonzero": True,
            "aggregate_descent_direction_correct": True,
        }
        for _ in range(4 * COVERAGE_STATE_BOUNDED_ROLE_COUNT)
    ]
    positive_rows = [
        {
            "role": role,
            "coordinate_count": 1,
            "transported_representation_finite": True,
            "transported_representation_fingerprint": "e" * 64,
        }
        for role in ("background", "component")
    ]
    config = CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
        feature_stride=COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
        width=COVERAGE_STATE_CMIF_FORMAL_WIDTH,
    )
    target_state_ids = [
        f"target-state-{index:02d}"
        for index in range(2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT)
    ]
    positive_state_ids = target_state_ids + [
        f"positive-only-state-{index:02d}"
        for index in range(4 * COVERAGE_STATE_BOUNDED_ROLE_COUNT)
    ]
    probe = {
        "execution_seed": COVERAGE_STATE_PAET_DR_EXECUTION_SEED,
        "runtime_splits": ["D_R"],
        "representation": {
            "state_counts": {
                "target_states": (
                    2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
                ),
                "context_states": (
                    6 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
                ),
                "factual_target_groups": (
                    COVERAGE_STATE_BOUNDED_ROLE_COUNT
                ),
                "clean_target_groups": (
                    COVERAGE_STATE_BOUNDED_ROLE_COUNT
                ),
                "target_pass_forward_count": (
                    2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
                ),
                "positive_pass_forward_count": (
                    6 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
                ),
                "total_forward_count": (
                    8 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
                ),
                "target_states_reforwarded_in_positive_pass": (
                    2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
                ),
                "two_pass_streaming_no_full_map_retention": True,
            },
            "coordinate_counts": {
                "target": (
                    9
                    + 2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
                    - 1
                ),
                "background": 1,
                "component": 1,
            },
            "target_group_rows": target_rows,
            "positive_role_rows": positive_rows,
            "state_id_ledger": {
                "target_pass_state_ids": target_state_ids,
                "positive_pass_state_ids": positive_state_ids,
                "reforwarded_target_state_ids": target_state_ids,
                "target_pass_state_id_fingerprint": stable_fingerprint(
                    target_state_ids
                ),
                "positive_pass_state_id_fingerprint": stable_fingerprint(
                    positive_state_ids
                ),
                (
                    "reforwarded_target_state_id_fingerprint"
                ): stable_fingerprint(target_state_ids),
            },
            "necessary_exact_collision_count": 0,
            "collision_examples": [],
            "exact_collision_zero_is_readout_feasibility_proof": False,
            "exact_collision_zero_is_only_a_necessary_check": True,
            "transported_representation_binding": "",
        },
        "field_direction": {
            "all_roles_finite_nonzero_correct": True,
            "uses_actual_target_geometry": True,
            "uses_actual_valid_and_writable_masks": True,
            "actual_role_rows": {
                "factual_target": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
                "clean_target": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
                "writable_background": (
                    COVERAGE_STATE_BOUNDED_ROLE_COUNT
                ),
                "writable_component": (
                    COVERAGE_STATE_BOUNDED_ROLE_COUNT
                ),
            },
            "rows": direction_rows,
            "loss_apis": [
                "coverage_state_absolute_sobolev_loss_from_targets",
                "coverage_state_pmope_pair_loss_from_targets",
            ],
        },
        "gradient_path": {
            "policy": PAET_DR_GRADIENT_CHECK,
            "zero_readout_loss_hex": float(1.0).hex(),
            "fixed_nonzero_readout_loss_hex": float(1.0).hex(),
            "zero_readout_scalar_path_finite_nonzero": True,
            "zero_readout_upstream_gradients_exactly_zero": True,
            "fixed_nonzero_readout_all_three_finite_nonzero": True,
            "functional_call_did_not_replace_model_parameters": True,
            "zero_readout_gradients": {
                "joint_state_weight": {
                    "finite": True,
                    "nonzero": False,
                },
                "joint_hidden_bias": {
                    "finite": True,
                    "nonzero": False,
                },
                "scalar_energy_weight": {
                    "finite": True,
                    "nonzero": True,
                },
            },
            "fixed_nonzero_readout_gradients": {
                name: {
                    "finite": True,
                    "nonzero": True,
                    "l2_hex": float(1.0).hex(),
                }
                for name in (
                    "joint_state_weight",
                    "joint_hidden_bias",
                    "scalar_energy_weight",
                )
            },
        },
        "deterministic_execution": {
            "policy": (
                "gate_internal_deterministic_algorithms_and_no_tf32_v1"
            ),
            "before": {
                "deterministic_algorithms": False,
                "deterministic_warn_only": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": False,
                "cudnn_allow_tf32": True,
                "cuda_matmul_allow_tf32": False,
            },
            "active": {
                "deterministic_algorithms": True,
                "deterministic_warn_only": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cudnn_allow_tf32": False,
                "cuda_matmul_allow_tf32": False,
            },
            "after": {
                "deterministic_algorithms": False,
                "deterministic_warn_only": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": False,
                "cudnn_allow_tf32": True,
                "cuda_matmul_allow_tf32": False,
            },
            "restored_exactly": True,
        },
        "model_config": {
            "model_class": (
                CURELitePhaseAlignedEvidenceTransportLevelSet.__name__
            ),
            "feature_channels": config.feature_channels,
            "feature_stride": config.feature_stride,
            "width": config.width,
            "parameter_count": (
                COVERAGE_STATE_CMIF_FORMAL_PARAMETER_COUNT
            ),
            "parameter_tensor_count": 3,
            "field_policy": config.field_policy,
            "equation_policy": config.equation_policy,
            "flip_policy": config.flip_policy,
            "transport_policy": config.transport_policy,
            "margin_hex": COVERAGE_STATE_PAET_MARGIN.hex(),
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
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    representation = probe["representation"]
    representation["transported_representation_binding"] = (
        stable_fingerprint(
            {
                "target_group_rows": target_rows,
                "positive_role_rows": positive_rows,
            }
        )
    )
    return dataset_free, real_inputs, population, probe


def _checks(values: tuple[object, object, object, dict]) -> dict[str, bool]:
    return dict(
        recompute_coverage_state_paet_dr_checks(
            dataset_free_receipt=values[0],
            real_inputs=values[1],
            bounded_population=values[2],
            probe=values[3],
        )
    )


def test_recompute_freezes_paet_checks_and_rejects_each_failure_mode() -> None:
    values = _passing_recompute_inputs()
    checks = _checks(values)
    assert checks
    assert all(checks.values())

    mutations = (
        (
            PAET_DR_BOUND_PHASE_PAIR_CHECK,
            lambda probe: probe["representation"][
                "target_group_rows"
            ][0]["bound_phase_pair_witness"]["pair_rows"][0].update(
                {
                    "phase_feature_separation_hex": (
                        COVERAGE_STATE_PAET_DR_SEPARATION_THRESHOLD.hex()
                    )
                }
            ),
        ),
        (
            PAET_DR_BOUND_PHASE_PAIR_CHECK,
            lambda probe: probe["representation"][
                "target_group_rows"
            ][0]["bound_phase_pair_witness"]["pair_rows"][0].update(
                {
                    "legal_background_from_state_mask": False,
                }
            ),
        ),
        (
            PAET_DR_CONFLICT_CHECK,
            lambda probe: probe["representation"].update(
                {
                    "necessary_exact_collision_count": 1,
                    "collision_examples": [{"collision": True}],
                }
            ),
        ),
        (
            "complete_declared_state_forward_ledger",
            lambda probe: probe["representation"]["state_counts"].update(
                {"two_pass_streaming_no_full_map_retention": False}
            ),
        ),
        (
            "complete_declared_state_forward_ledger",
            lambda probe: probe["representation"][
                "state_id_ledger"
            ]["reforwarded_target_state_ids"].pop(),
        ),
        (
            "all_positive_role_representations_finite_and_bound",
            lambda probe: probe["representation"].update(
                {"transported_representation_binding": "0" * 64}
            ),
        ),
        (
            "fixed_nonzero_readout_all_three_gradients_finite_nonzero",
            lambda probe: probe["gradient_path"].update(
                {
                    "fixed_nonzero_readout_all_three_finite_nonzero": (
                        False
                    )
                }
            ),
        ),
        (
            "read_only_zero_update_D_R_only_scope",
            lambda probe: probe.update({"D_V_accessed": True}),
        ),
        (
            "gate_internal_deterministic_flags_fixed_and_restored",
            lambda probe: probe["deterministic_execution"].update(
                {"restored_exactly": False}
            ),
        ),
        (
            "identifiability_only_no_performance_or_AUC_gate",
            lambda probe: probe["representation"].update(
                {"hidden_auc_gate": True}
            ),
        ),
    )
    for check_name, mutate in mutations:
        values = _passing_recompute_inputs()
        mutate(values[3])
        assert _checks(values)[check_name] is False


def test_receipt_status_and_scope_are_frozen_properties() -> None:
    assert COVERAGE_STATE_PAET_DR_EXECUTION_SEED == 42
    assert isinstance(CoverageStatePAETDRGateReceipt.all_pass, property)
    assert isinstance(CoverageStatePAETDRGateReceipt.decision, property)
    assert isinstance(
        CoverageStatePAETDRGateReceipt.receipt_fingerprint,
        property,
    )
    assert COVERAGE_STATE_PAET_DR_PASS_DECISION.endswith("_PASS")
    assert COVERAGE_STATE_PAET_DR_FAIL_DECISION.endswith("_FAIL")
