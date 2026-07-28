from __future__ import annotations

import torch

from cure_lite.coverage_state_binary_flip_antisymmetric import (
    CURELiteBinaryFlipAntisymmetricLevelSet,
    CoverageStateBinaryFlipAntisymmetricConfig,
)
from cure_lite.experiment.coverage_state_bfa_curvature_audit import (
    COVERAGE_STATE_BFA_CURVATURE_DECISION_ACCEPT,
    COVERAGE_STATE_BFA_CURVATURE_DECISION_REJECT,
    _median,
    audit_coverage_state_bfa_curvature_checkpoint,
    curvature_gated_odd_delta,
    decide_coverage_state_bfa_curvature_audit,
    evaluate_bfa_scalar_odd_curvature,
    one_sided_exact_sign_test,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    build_coverage_state_bounded_population,
)
from cure_lite.frozen_base import module_state_fingerprint
from tests_v15.coverage_state_training_test_helpers import (
    make_bounded_training_scalar_cache,
)


def _model(
    *,
    feature_channels: int = 2,
    feature_stride: int = 4,
    width: int = 4,
):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1720)
        model = CURELiteBinaryFlipAntisymmetricLevelSet(
            CoverageStateBinaryFlipAntisymmetricConfig(
                feature_channels=feature_channels,
                feature_stride=feature_stride,
                width=width,
            )
        )
        with torch.no_grad():
            model.scalar_energy_weight.copy_(
                torch.linspace(-0.7, 0.9, width)
            )
    return model.eval()


def test_proxy_formula_is_odd_in_delta_and_has_no_search_parameter() -> None:
    delta = torch.tensor(
        [[-2.0, -0.25, 0.0, 0.25, 2.0]],
        dtype=torch.float32,
    )
    curvature = torch.tensor(
        [[-1.2, 0.4, 0.0, -0.7, 1.8]],
        dtype=torch.float32,
    )
    positive = curvature_gated_odd_delta(delta, curvature)
    negative = curvature_gated_odd_delta(-delta, curvature)

    assert torch.equal(negative, -positive)
    expected = delta * (1.0 - torch.tanh(curvature / 0.9))
    assert torch.equal(positive, expected)
    assert torch.equal(
        torch.sign(positive),
        torch.sign(delta),
    )


def test_shared_energy_scalar_curvature_is_flip_even_and_proxy_is_flip_odd(
) -> None:
    model = _model()
    feature = (
        torch.arange(2 * 2 * 2, dtype=torch.float32)
        .reshape(1, 2, 2, 2)
        .div(7.0)
    )
    occupancy0 = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    occupancy1 = occupancy0.clone()
    coordinate = (3, 5)
    occupancy1[0, 0, coordinate[0], coordinate[1]] = True

    with torch.inference_mode():
        value0 = evaluate_bfa_scalar_odd_curvature(
            model, feature, occupancy0
        )
        value1 = evaluate_bfa_scalar_odd_curvature(
            model, feature, occupancy1
        )
    index = (0, 0, *coordinate)

    assert torch.allclose(
        value0.canonical_odd_delta[index],
        value1.canonical_odd_delta[index],
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    assert torch.allclose(
        value0.midpoint_curvature[index],
        value1.midpoint_curvature[index],
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    assert torch.allclose(
        value0.oriented_odd_delta[index],
        -value1.oriented_odd_delta[index],
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    assert torch.allclose(
        value0.proxy_oriented_delta[index],
        -value1.proxy_oriented_delta[index],
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    assert torch.equal(value0.bfa_field, model(feature, occupancy0))


def test_exact_sign_test_fixes_the_n16_twelve_win_boundary() -> None:
    passed = one_sided_exact_sign_test(
        (-1.0,) * 12 + (1.0,) * 4,
        desired="negative",
    )
    failed = one_sided_exact_sign_test(
        (-1.0,) * 11 + (1.0,) * 5,
        desired="negative",
    )

    assert passed["non_tied_count"] == 16
    assert passed["win_count"] == 12
    assert passed["passed"] is True
    assert failed["win_count"] == 11
    assert failed["passed"] is False


def test_group_statistic_is_median_not_mean() -> None:
    values = torch.tensor([-100.0, 1.0, 2.0], dtype=torch.float32)
    assert _median(values) == 1.0
    assert float(values.mean()) < 0.0


def _passing_sign_tests() -> dict[str, dict[str, object]]:
    names = (
        "clean_target_e_negative",
        "spill_e_positive",
        "factual_target_e_negative",
        "same_pair_target_below_spill",
        "proxy_pair_pareto",
    )
    return {name: {"passed": True} for name in names}


def _passing_proxy_summary() -> dict[str, object]:
    return {
        "clean_target_negative_pixels": 116,
        "clean_outside_pixels": 53,
        "factual_recovered": 16,
        "factual_strict": 14,
        "factual_target_negative_pixels": 310,
        "factual_no_miss_passed": 16,
        "component_null_passed": 16,
        "identity_null_passed": 16,
        "diagnostic_null_passed": True,
        "invalid_completion_pixels": 0,
        "clean_compact_support_passed": 1,
    }


def _passing_global_curvature() -> dict[str, float]:
    return {
        "clean_added_target": -0.1,
        "v20_new_completion_outside": 0.1,
        "factual_target": -0.2,
    }


def _passing_multiplier_summary() -> dict[str, object]:
    return {
        "count": 100,
        "all_finite": True,
        "less_than_or_equal_zero_count": 0,
        "greater_than_or_equal_two_count": 0,
    }


def test_decision_is_conjunctive_and_has_only_two_outcomes() -> None:
    masks = {
        "clean_added_target": 149,
        "v20_new_completion_outside": 54,
        "factual_target": 335,
        "component_null_groups": 16,
        "clean_true_background": 1,
    }
    decision, checks = decide_coverage_state_bfa_curvature_audit(
        mask_counts=masks,
        sign_tests=_passing_sign_tests(),
        global_curvature_medians=_passing_global_curvature(),
        proxy_multiplier_summary=_passing_multiplier_summary(),
        proxy_summary=_passing_proxy_summary(),
    )
    assert decision == COVERAGE_STATE_BFA_CURVATURE_DECISION_ACCEPT
    assert all(value for _, value in checks)

    failed = _passing_proxy_summary()
    failed["clean_outside_pixels"] = 54
    decision, checks = decide_coverage_state_bfa_curvature_audit(
        mask_counts=masks,
        sign_tests=_passing_sign_tests(),
        global_curvature_medians=_passing_global_curvature(),
        proxy_multiplier_summary=_passing_multiplier_summary(),
        proxy_summary=failed,
    )
    assert decision == COVERAGE_STATE_BFA_CURVATURE_DECISION_REJECT
    assert dict(checks)["proxy_clean_outside_improved"] is False


def test_decision_rejects_global_median_or_saturated_multiplier() -> None:
    masks = {
        "clean_added_target": 149,
        "v20_new_completion_outside": 54,
        "clean_true_background": 1,
        "factual_target": 335,
        "component_null_groups": 16,
    }
    global_values = _passing_global_curvature()
    global_values["clean_added_target"] = 0.0
    decision, checks = decide_coverage_state_bfa_curvature_audit(
        mask_counts=masks,
        sign_tests=_passing_sign_tests(),
        global_curvature_medians=global_values,
        proxy_multiplier_summary=_passing_multiplier_summary(),
        proxy_summary=_passing_proxy_summary(),
    )
    assert decision == COVERAGE_STATE_BFA_CURVATURE_DECISION_REJECT
    assert dict(checks)["global_clean_target_curvature_negative"] is False

    multiplier = _passing_multiplier_summary()
    multiplier["less_than_or_equal_zero_count"] = 1
    decision, checks = decide_coverage_state_bfa_curvature_audit(
        mask_counts=masks,
        sign_tests=_passing_sign_tests(),
        global_curvature_medians=_passing_global_curvature(),
        proxy_multiplier_summary=multiplier,
        proxy_summary=_passing_proxy_summary(),
    )
    assert decision == COVERAGE_STATE_BFA_CURVATURE_DECISION_REJECT
    assert (
        dict(checks)["proxy_multiplier_finite_strict_open_interval"]
        is False
    )


def test_full_toy_audit_is_read_only_and_rejects_without_real_r2_masks(
) -> None:
    cache = make_bounded_training_scalar_cache()
    population = build_coverage_state_bounded_population(cache)
    feature_channels = cache.natural_records[0].record.feature.shape[1]
    model = _model(
        feature_channels=feature_channels,
        feature_stride=cache.raw_catalog.feature_stride,
    )
    state_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    fingerprint_before = module_state_fingerprint(model)
    rng_before = torch.random.get_rng_state().clone()

    receipt = audit_coverage_state_bfa_curvature_checkpoint(
        model,
        population,
        device="cpu",
        evidence_binding={"toy": "read-only"},
    )

    assert receipt.decision == COVERAGE_STATE_BFA_CURVATURE_DECISION_REJECT
    assert receipt.optimizer_constructed is False
    assert receipt.backward_performed is False
    assert receipt.training_performed is False
    assert receipt.d_v_accessed is False
    assert receipt.d_t_accessed is False
    assert receipt.checkpoint_fingerprint_before == fingerprint_before
    assert receipt.checkpoint_fingerprint_after == fingerprint_before
    assert torch.equal(rng_before, torch.random.get_rng_state())
    for name, value in model.state_dict().items():
        assert torch.equal(value, state_before[name])
    receipt.verify()
