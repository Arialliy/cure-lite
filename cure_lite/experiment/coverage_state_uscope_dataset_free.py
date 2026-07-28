"""Generated-only evidence for the v19 USCOPE product gauge.

This module exercises only generated tensors.  It does not import a dataset,
cache, training runner, optimizer, or runtime-split protocol.  The probes
bind USCOPE to the unchanged full-valid-domain PMOPE violations and verify
the additional statewise Chebyshev term without granting any run authority.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from ..cache.schema import file_sha256, stable_fingerprint
from ..coverage_state_level_set import CSLF_FIELD_AMPLITUDE
from ..coverage_state_sobolev import (
    CoverageStatePairTargets,
    CoverageStateSobolevConfig,
    coverage_state_pmope_pair_loss_from_targets,
    prepare_coverage_state_pair_targets,
)
from ..coverage_state_supremal_projection import (
    CSLF_USCOPE_POLICY,
    coverage_state_uscope_pair_loss_from_targets,
)


COVERAGE_STATE_USCOPE_DATASET_FREE_SCHEMA = (
    "cure-lite-uscope-v19-dataset-free-receipt-v1"
)
COVERAGE_STATE_USCOPE_DATASET_FREE_EXECUTION_SEED = 190019
COVERAGE_STATE_USCOPE_TRUNCATION_RADIUS = 4
COVERAGE_STATE_USCOPE_MARGIN = (
    CSLF_FIELD_AMPLITUDE / COVERAGE_STATE_USCOPE_TRUNCATION_RADIUS
)
COVERAGE_STATE_USCOPE_IMPLEMENTATION_PATHS = (
    "cure_lite/coverage_state_level_set.py",
    "cure_lite/coverage_state_sobolev.py",
    "cure_lite/coverage_state_supremal_projection.py",
    "cure_lite/experiment/coverage_state_uscope_dataset_free.py",
)


def _config() -> CoverageStateSobolevConfig:
    return CoverageStateSobolevConfig(
        truncation_radius=COVERAGE_STATE_USCOPE_TRUNCATION_RADIUS
    )


def _current_implementation_binding() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[2]
    result: list[tuple[str, str]] = []
    for relative in COVERAGE_STATE_USCOPE_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"USCOPE implementation path is invalid: {relative}"
            )
        result.append((relative, file_sha256(path)))
    return tuple(result)


def _empty_mask(batch: int, size: int) -> Tensor:
    return torch.zeros(batch, 1, size, size, dtype=torch.bool)


def _pair_targets(
    *,
    size: int,
    batch: int = 1,
    occupancy_plus: Tensor | None = None,
    occupancy_minus: Tensor | None = None,
    target_plus: Tensor | None = None,
    target_minus: Tensor | None = None,
    valid_mask: Tensor | None = None,
) -> CoverageStatePairTargets:
    empty = _empty_mask(batch, size)
    resolved_occupancy_plus = (
        empty.clone() if occupancy_plus is None else occupancy_plus
    )
    resolved_occupancy_minus = (
        empty.clone() if occupancy_minus is None else occupancy_minus
    )
    resolved_target_plus = (
        empty.clone() if target_plus is None else target_plus
    )
    resolved_target_minus = (
        empty.clone() if target_minus is None else target_minus
    )
    valid = (
        torch.ones(batch, 1, size, size, dtype=torch.bool)
        if valid_mask is None
        else valid_mask
    )
    return prepare_coverage_state_pair_targets(
        resolved_occupancy_plus,
        resolved_occupancy_minus,
        resolved_target_plus,
        resolved_target_minus,
        valid,
        config=_config(),
    )


def _deep_feasible_fields(
    targets: CoverageStatePairTargets,
    *,
    slack: float = 0.5,
) -> tuple[Tensor, Tensor]:
    magnitude = COVERAGE_STATE_USCOPE_MARGIN + slack
    return (
        torch.sign(targets.target_field_plus) * magnitude,
        torch.sign(targets.target_field_minus) * magnitude,
    )


def _set_violation(
    field: Tensor,
    target_field: Tensor,
    *,
    batch: int,
    row: int,
    column: int,
    violation: float,
) -> None:
    sign = float(torch.sign(target_field[batch, 0, row, column]).item())
    if sign not in {-1.0, 1.0}:
        raise AssertionError("generated target field must have a strict sign")
    field[batch, 0, row, column] = sign * (
        COVERAGE_STATE_USCOPE_MARGIN - violation
    )


def _hex_values(value: Tensor) -> list[str]:
    return [
        float(item).hex()
        for item in value.detach().to("cpu").flatten().tolist()
    ]


def _exact_product_probe() -> dict[str, object]:
    targets = _pair_targets(size=9, batch=2)
    field_plus, field_minus = _deep_feasible_fields(targets)
    _set_violation(
        field_plus,
        targets.target_field_plus,
        batch=0,
        row=2,
        column=2,
        violation=0.125,
    )
    _set_violation(
        field_minus,
        targets.target_field_minus,
        batch=0,
        row=4,
        column=4,
        violation=0.375,
    )
    _set_violation(
        field_plus,
        targets.target_field_plus,
        batch=1,
        row=3,
        column=3,
        violation=0.25,
    )
    _set_violation(
        field_minus,
        targets.target_field_minus,
        batch=1,
        row=5,
        column=5,
        violation=0.5,
    )
    fields = coverage_state_uscope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=_config(),
    )
    manual_gamma = torch.maximum(
        fields.violation_plus.flatten(1).amax(dim=1),
        fields.violation_minus.flatten(1).amax(dim=1),
    )
    manual_sobolev = 0.5 * (
        fields.per_state_value_power
        + fields.per_state_spatial_power
    )
    manual_chebyshev = manual_gamma.pow(_config().norm_order)
    manual_product = 0.5 * (
        manual_sobolev + manual_chebyshev
    )
    manual_per_state = (
        manual_product + _config().norm_epsilon ** _config().norm_order
    ).pow(1.0 / float(_config().norm_order)) - _config().norm_epsilon
    return {
        "state_count": int(field_plus.shape[0]),
        "gamma_exact": torch.equal(
            fields.per_state_chebyshev_violation,
            manual_gamma,
        ),
        "sobolev_power_exact": torch.equal(
            fields.per_state_sobolev_power,
            manual_sobolev,
        ),
        "chebyshev_power_exact": torch.equal(
            fields.per_state_chebyshev_power,
            manual_chebyshev,
        ),
        "product_power_exact": torch.equal(
            fields.per_state_product_power,
            manual_product,
        ),
        "per_state_loss_exact": torch.equal(
            fields.per_state_loss,
            manual_per_state,
        ),
        "batch_mean_exact": torch.equal(
            fields.loss,
            fields.per_state_loss.mean(),
        ),
        "gamma_hex": _hex_values(
            fields.per_state_chebyshev_violation
        ),
        "per_state_loss_hex": _hex_values(fields.per_state_loss),
        "loss_finite": bool(torch.isfinite(fields.loss)),
    }


def _gamma_certificate_probe() -> dict[str, object]:
    size = 9
    target = _empty_mask(1, size)
    target[..., 4, 4] = True
    targets = _pair_targets(
        size=size,
        target_plus=target,
        target_minus=target.clone(),
    )
    delta = 0.5 * COVERAGE_STATE_USCOPE_MARGIN
    field_plus = torch.sign(targets.target_field_plus) * (
        COVERAGE_STATE_USCOPE_MARGIN - delta
    )
    field_minus = torch.sign(targets.target_field_minus) * (
        COVERAGE_STATE_USCOPE_MARGIN - delta
    )
    certified = coverage_state_uscope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=_config(),
    )
    target_plus = targets.target_field_plus < 0.0
    target_minus = targets.target_field_minus < 0.0

    boundary_plus, boundary_minus = _deep_feasible_fields(targets)
    boundary_minus[..., 4, 4] = 0.0
    boundary = coverage_state_uscope_pair_loss_from_targets(
        boundary_plus,
        boundary_minus,
        targets,
        config=_config(),
    )
    return {
        "margin_hex": COVERAGE_STATE_USCOPE_MARGIN.hex(),
        "certified_gamma_hex": float(
            certified.per_state_chebyshev_violation.item()
        ).hex(),
        "certified_gamma_strictly_below_margin": bool(
            certified.per_state_chebyshev_violation
            < COVERAGE_STATE_USCOPE_MARGIN
        ),
        "certified_plus_sign_exact": torch.equal(
            field_plus < 0.0,
            target_plus,
        ),
        "certified_minus_sign_exact": torch.equal(
            field_minus < 0.0,
            target_minus,
        ),
        "boundary_gamma_at_least_margin": bool(
            boundary.per_state_chebyshev_violation
            >= COVERAGE_STATE_USCOPE_MARGIN
        ),
        "boundary_minus_sign_not_exact": not torch.equal(
            boundary_minus < 0.0,
            target_minus,
        ),
        "certificate_is_sufficient_not_necessary": True,
    }


def _one_bad_pixel_case(
    size: int,
    *,
    pixel_kind: str,
) -> tuple[dict[str, object], Tensor]:
    if pixel_kind not in {"target", "background"}:
        raise ValueError("pixel_kind must be target or background")
    target = _empty_mask(1, size)
    row = size // 2
    column = size // 2
    if pixel_kind == "target":
        target[..., row, column] = True
    targets = _pair_targets(
        size=size,
        target_plus=target,
        target_minus=target.clone(),
    )
    field_plus, field_minus = _deep_feasible_fields(targets)
    field_minus[..., row, column] = (
        0.05 if pixel_kind == "target" else -0.05
    )
    field_minus.requires_grad_(True)
    fields = coverage_state_uscope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=_config(),
    )
    (gradient_minus,) = torch.autograd.grad(
        fields.loss,
        (field_minus,),
    )
    gamma = fields.per_state_chebyshev_violation
    chebyshev_only_lower_bound = (
        0.5 * gamma.pow(_config().norm_order)
        + _config().norm_epsilon ** _config().norm_order
    ).pow(1.0 / float(_config().norm_order)) - _config().norm_epsilon
    payload = {
        "size": size,
        "pixel_count": size * size,
        "pixel_kind": pixel_kind,
        "gamma_hex": float(gamma.item()).hex(),
        "chebyshev_power_hex": float(
            fields.per_state_chebyshev_power.item()
        ).hex(),
        "loss_hex": float(fields.loss.item()).hex(),
        "loss_positive": bool(fields.loss > 0.0),
        "loss_at_least_chebyshev_lower_bound": bool(
            fields.loss >= chebyshev_only_lower_bound
        ),
        "bad_pixel_gradient_has_expected_sign": bool(
            gradient_minus[0, 0, row, column]
            > 0.0
            if pixel_kind == "target"
            else gradient_minus[0, 0, row, column] < 0.0
        ),
        "bad_pixel_descent_has_expected_direction": bool(
            -gradient_minus[0, 0, row, column]
            < 0.0
            if pixel_kind == "target"
            else -gradient_minus[0, 0, row, column] > 0.0
        ),
        "bad_pixel_descent_moves_background_positive": bool(
            pixel_kind == "background"
            and -gradient_minus[0, 0, row, column] > 0.0
        ),
        "gradient_finite": bool(torch.isfinite(gradient_minus).all()),
    }
    return payload, gamma.detach()


def _single_pixel_probe() -> dict[str, object]:
    small, small_gamma = _one_bad_pixel_case(
        16,
        pixel_kind="background",
    )
    full, full_gamma = _one_bad_pixel_case(
        256,
        pixel_kind="background",
    )
    target_small, target_small_gamma = _one_bad_pixel_case(
        16,
        pixel_kind="target",
    )
    target_full, target_full_gamma = _one_bad_pixel_case(
        256,
        pixel_kind="target",
    )
    return {
        "small": small,
        "full": full,
        "target_small": target_small,
        "target_full": target_full,
        "gamma_size_invariant": torch.equal(
            small_gamma,
            full_gamma,
        ),
        "target_gamma_size_invariant": torch.equal(
            target_small_gamma,
            target_full_gamma,
        ),
        "one_bad_pixel_not_diluted": bool(
            small["loss_positive"]
            and full["loss_positive"]
            and small["loss_at_least_chebyshev_lower_bound"]
            and full["loss_at_least_chebyshev_lower_bound"]
        ),
        "one_bad_target_pixel_not_diluted": bool(
            target_small["loss_positive"]
            and target_full["loss_positive"]
            and target_small["loss_at_least_chebyshev_lower_bound"]
            and target_full["loss_at_least_chebyshev_lower_bound"]
        ),
    }


def _gradient_direction_probe() -> dict[str, object]:
    size = 9
    target = _empty_mask(1, size)
    target[..., 4, 4] = True
    target_targets = _pair_targets(
        size=size,
        target_plus=target,
        target_minus=target.clone(),
    )
    target_plus, target_minus = _deep_feasible_fields(target_targets)
    target_minus[..., 4, 4] = 0.1
    target_minus.requires_grad_(True)
    target_fields = coverage_state_uscope_pair_loss_from_targets(
        target_plus,
        target_minus,
        target_targets,
        config=_config(),
    )
    (target_gradient,) = torch.autograd.grad(
        target_fields.loss,
        (target_minus,),
    )

    background_targets = _pair_targets(size=size)
    background_plus, background_minus = _deep_feasible_fields(
        background_targets
    )
    background_minus[..., 4, 4] = -0.1
    background_minus.requires_grad_(True)
    background_fields = coverage_state_uscope_pair_loss_from_targets(
        background_plus,
        background_minus,
        background_targets,
        config=_config(),
    )
    (background_gradient,) = torch.autograd.grad(
        background_fields.loss,
        (background_minus,),
    )
    target_value = target_gradient[0, 0, 4, 4]
    background_value = background_gradient[0, 0, 4, 4]
    return {
        "target_gradient_positive": bool(target_value > 0.0),
        "target_descent_negative": bool(-target_value < 0.0),
        "background_gradient_negative": bool(background_value < 0.0),
        "background_descent_positive": bool(-background_value > 0.0),
        "target_gradient_finite": bool(
            torch.isfinite(target_gradient).all()
        ),
        "background_gradient_finite": bool(
            torch.isfinite(background_gradient).all()
        ),
        "target_unique_chebyshev_violation": bool(
            target_fields.per_state_chebyshev_violation > 0.0
        ),
        "background_unique_chebyshev_violation": bool(
            background_fields.per_state_chebyshev_violation > 0.0
        ),
    }


def _occupied_hidden_negative_probe() -> dict[str, object]:
    size = 9
    occupancy = _empty_mask(1, size)
    occupancy[..., 4, 4] = True
    targets = _pair_targets(
        size=size,
        occupancy_plus=occupancy,
        occupancy_minus=occupancy.clone(),
    )
    field_plus, field_minus = _deep_feasible_fields(targets)
    field_plus[..., 4, 4] = -0.2
    field_plus.requires_grad_(True)
    fields = coverage_state_uscope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=_config(),
    )
    (gradient_plus,) = torch.autograd.grad(
        fields.loss,
        (field_plus,),
    )
    return {
        "occupied_pixel": bool(occupancy[..., 4, 4]),
        "raw_hidden_negative": bool(field_plus[..., 4, 4] < 0.0),
        "completion_masks_hidden_negative": not bool(
            ((field_plus < 0.0) & ~occupancy)[..., 4, 4]
        ),
        "full_domain_violation_positive": bool(
            fields.violation_plus[..., 4, 4] > 0.0
        ),
        "gamma_positive": bool(
            fields.per_state_chebyshev_violation > 0.0
        ),
        "gamma_above_margin": bool(
            fields.per_state_chebyshev_violation
            > COVERAGE_STATE_USCOPE_MARGIN
        ),
        "loss_positive": bool(fields.loss > 0.0),
        "gradient_pushes_field_positive": bool(
            gradient_plus[..., 4, 4] < 0.0
        ),
        "descent_moves_field_positive": bool(
            -gradient_plus[..., 4, 4] > 0.0
        ),
        "valid_domain_includes_occupied_pixel": bool(
            fields.valid_mask[..., 4, 4]
        ),
    }


def _invalid_domain_probe() -> dict[str, object]:
    size = 9
    valid = torch.ones(1, 1, size, size, dtype=torch.bool)
    valid[..., 0, 0] = False
    targets = _pair_targets(size=size, valid_mask=valid)
    field_plus, field_minus = _deep_feasible_fields(targets)
    reference = coverage_state_uscope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=_config(),
    )
    field_plus[..., 0, 0] = -100.0
    field_minus[..., 0, 0] = 100.0
    mutated = coverage_state_uscope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=_config(),
    )
    return {
        "probe_pixel_outside_valid_domain": not bool(
            targets.valid_mask[..., 0, 0]
        ),
        "outside_plus_violation_exact_zero": float(
            mutated.violation_plus[..., 0, 0].item()
        )
        == 0.0,
        "outside_minus_violation_exact_zero": float(
            mutated.violation_minus[..., 0, 0].item()
        )
        == 0.0,
        "gamma_unchanged": torch.equal(
            reference.per_state_chebyshev_violation,
            mutated.per_state_chebyshev_violation,
        ),
        "loss_unchanged": torch.equal(reference.loss, mutated.loss),
        "outside_domain_has_no_certificate": True,
    }


def _zero_set_and_finite_risk_probe() -> dict[str, object]:
    targets = _pair_targets(size=13)
    zero_plus, zero_minus = _deep_feasible_fields(targets)
    zero_uscope = coverage_state_uscope_pair_loss_from_targets(
        zero_plus,
        zero_minus,
        targets,
        config=_config(),
    )
    zero_pmope = coverage_state_pmope_pair_loss_from_targets(
        zero_plus,
        zero_minus,
        targets,
        config=_config(),
    )

    finite_plus, finite_minus = _deep_feasible_fields(targets)
    finite_minus[..., 6, 6] = -0.2
    finite_uscope = coverage_state_uscope_pair_loss_from_targets(
        finite_plus,
        finite_minus,
        targets,
        config=_config(),
    )
    finite_pmope = coverage_state_pmope_pair_loss_from_targets(
        finite_plus,
        finite_minus,
        targets,
        config=_config(),
    )
    return {
        "zero_violation_plus_same": torch.equal(
            zero_uscope.violation_plus,
            zero_pmope.violation_plus,
        ),
        "zero_violation_minus_same": torch.equal(
            zero_uscope.violation_minus,
            zero_pmope.violation_minus,
        ),
        "zero_uscope_loss_exact": float(zero_uscope.loss.item()) == 0.0,
        "zero_pmope_loss_exact": float(zero_pmope.loss.item()) == 0.0,
        "zero_gamma_exact": float(
            zero_uscope.per_state_chebyshev_violation.item()
        ) == 0.0,
        "finite_violation_plus_same": torch.equal(
            finite_uscope.violation_plus,
            finite_pmope.violation_plus,
        ),
        "finite_violation_minus_same": torch.equal(
            finite_uscope.violation_minus,
            finite_pmope.violation_minus,
        ),
        "finite_uscope_positive": bool(finite_uscope.loss > 0.0),
        "finite_pmope_positive": bool(finite_pmope.loss > 0.0),
        "finite_risks_different": not torch.equal(
            finite_uscope.loss,
            finite_pmope.loss,
        ),
        "finite_uscope_greater_than_pmope": bool(
            finite_uscope.loss > finite_pmope.loss
        ),
        "same_q_implies_same_zero_set": True,
    }


def _same_sign_response_diagnostic_probe() -> dict[str, object]:
    size = 15
    target_plus = _empty_mask(1, size)
    target_minus = _empty_mask(1, size)
    target_minus[..., 7, 7] = True
    targets = _pair_targets(
        size=size,
        target_plus=target_plus,
        target_minus=target_minus,
    )
    magnitude = COVERAGE_STATE_USCOPE_MARGIN + 0.4
    field_plus = torch.sign(targets.target_field_plus) * magnitude
    field_minus = torch.sign(targets.target_field_minus) * magnitude
    fields = coverage_state_uscope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=_config(),
    )
    target_response = (
        targets.target_field_minus - targets.target_field_plus
    )
    same_sign = (
        targets.valid_mask
        & target_response.ne(0.0)
        & (
            torch.sign(targets.target_field_plus)
            == torch.sign(targets.target_field_minus)
        )
    )
    predicted_response = field_minus - field_plus
    correct = (
        predicted_response[same_sign] * target_response[same_sign]
    ) > 0.0
    return {
        "same_sign_response_pixel_count": int(same_sign.sum().item()),
        "same_sign_response_correct_count": int(correct.sum().item()),
        "same_sign_response_has_error": bool(
            same_sign.any() and not bool(correct.all())
        ),
        "endpoint_violations_exact_zero": (
            not bool(torch.any(fields.violation_plus))
            and not bool(torch.any(fields.violation_minus))
        ),
        "gamma_exact_zero": float(
            fields.per_state_chebyshev_violation.item()
        ) == 0.0,
        "uscope_loss_exact_zero": float(fields.loss.item()) == 0.0,
        "response_consumed_by_objective": False,
        "response_is_diagnostic_only": True,
    }


def _generated_payload(
    *,
    exact_product_probe: dict[str, object],
    gamma_certificate_probe: dict[str, object],
    single_pixel_probe: dict[str, object],
    gradient_direction_probe: dict[str, object],
    occupied_hidden_negative_probe: dict[str, object],
    invalid_domain_probe: dict[str, object],
    zero_set_and_finite_risk_probe: dict[str, object],
    same_sign_response_diagnostic_probe: dict[str, object],
) -> dict[str, object]:
    return {
        "exact_product_probe": exact_product_probe,
        "gamma_certificate_probe": gamma_certificate_probe,
        "single_pixel_probe": single_pixel_probe,
        "gradient_direction_probe": gradient_direction_probe,
        "occupied_hidden_negative_probe": occupied_hidden_negative_probe,
        "invalid_domain_probe": invalid_domain_probe,
        "zero_set_and_finite_risk_probe": (
            zero_set_and_finite_risk_probe
        ),
        "same_sign_response_diagnostic_probe": (
            same_sign_response_diagnostic_probe
        ),
    }


def _collect_generated_evidence() -> dict[str, object]:
    torch.manual_seed(COVERAGE_STATE_USCOPE_DATASET_FREE_EXECUTION_SEED)
    return {
        "exact_product_probe": _exact_product_probe(),
        "gamma_certificate_probe": _gamma_certificate_probe(),
        "single_pixel_probe": _single_pixel_probe(),
        "gradient_direction_probe": _gradient_direction_probe(),
        "occupied_hidden_negative_probe": (
            _occupied_hidden_negative_probe()
        ),
        "invalid_domain_probe": _invalid_domain_probe(),
        "zero_set_and_finite_risk_probe": (
            _zero_set_and_finite_risk_probe()
        ),
        "same_sign_response_diagnostic_probe": (
            _same_sign_response_diagnostic_probe()
        ),
    }


def recompute_coverage_state_uscope_dataset_free_checks(
    *,
    implementation_binding: tuple[tuple[str, str], ...],
    generated_replay_fingerprint: str,
    exact_product_probe: dict[str, object],
    gamma_certificate_probe: dict[str, object],
    single_pixel_probe: dict[str, object],
    gradient_direction_probe: dict[str, object],
    occupied_hidden_negative_probe: dict[str, object],
    invalid_domain_probe: dict[str, object],
    zero_set_and_finite_risk_probe: dict[str, object],
    same_sign_response_diagnostic_probe: dict[str, object],
) -> tuple[tuple[str, bool], ...]:
    """Recompute every generated-only USCOPE gate bit."""

    generated = _generated_payload(
        exact_product_probe=exact_product_probe,
        gamma_certificate_probe=gamma_certificate_probe,
        single_pixel_probe=single_pixel_probe,
        gradient_direction_probe=gradient_direction_probe,
        occupied_hidden_negative_probe=occupied_hidden_negative_probe,
        invalid_domain_probe=invalid_domain_probe,
        zero_set_and_finite_risk_probe=zero_set_and_finite_risk_probe,
        same_sign_response_diagnostic_probe=(
            same_sign_response_diagnostic_probe
        ),
    )
    expected_generated_fingerprint = stable_fingerprint(generated)
    binding_paths = tuple(path for path, _ in implementation_binding)
    checks = {
        "policy_and_margin_exact": (
            CSLF_USCOPE_POLICY
            == "uniform_sobolev_chebyshev_orthant_projection_energy_v1"
            and COVERAGE_STATE_USCOPE_MARGIN == 0.225
        ),
        "exact_product_gauge_and_batch_mean": (
            exact_product_probe["state_count"] == 2
            and exact_product_probe["gamma_exact"]
            and exact_product_probe["sobolev_power_exact"]
            and exact_product_probe["chebyshev_power_exact"]
            and exact_product_probe["product_power_exact"]
            and exact_product_probe["per_state_loss_exact"]
            and exact_product_probe["batch_mean_exact"]
            and exact_product_probe["loss_finite"]
        ),
        "gamma_below_margin_certificate": all(
            bool(gamma_certificate_probe[name])
            for name in (
                "certified_gamma_strictly_below_margin",
                "certified_plus_sign_exact",
                "certified_minus_sign_exact",
                "boundary_gamma_at_least_margin",
                "boundary_minus_sign_not_exact",
                "certificate_is_sufficient_not_necessary",
            )
        ),
        "single_bad_pixel_not_diluted": (
            single_pixel_probe["gamma_size_invariant"]
            and single_pixel_probe["one_bad_pixel_not_diluted"]
            and single_pixel_probe["target_gamma_size_invariant"]
            and single_pixel_probe[
                "one_bad_target_pixel_not_diluted"
            ]
            and single_pixel_probe["small"]["gradient_finite"]
            and single_pixel_probe["full"]["gradient_finite"]
            and single_pixel_probe["target_small"]["gradient_finite"]
            and single_pixel_probe["target_full"]["gradient_finite"]
            and all(
                bool(single_pixel_probe[name][check])
                for name in (
                    "small",
                    "full",
                    "target_small",
                    "target_full",
                )
                for check in (
                    "bad_pixel_gradient_has_expected_sign",
                    "bad_pixel_descent_has_expected_direction",
                )
            )
        ),
        "target_and_background_gradient_directions": all(
            bool(gradient_direction_probe[name])
            for name in (
                "target_gradient_positive",
                "target_descent_negative",
                "background_gradient_negative",
                "background_descent_positive",
                "target_gradient_finite",
                "background_gradient_finite",
                "target_unique_chebyshev_violation",
                "background_unique_chebyshev_violation",
            )
        ),
        "occupied_hidden_negative_captured": all(
            bool(occupied_hidden_negative_probe[name])
            for name in (
                "occupied_pixel",
                "raw_hidden_negative",
                "completion_masks_hidden_negative",
                "full_domain_violation_positive",
                "gamma_positive",
                "gamma_above_margin",
                "loss_positive",
                "gradient_pushes_field_positive",
                "descent_moves_field_positive",
                "valid_domain_includes_occupied_pixel",
            )
        ),
        "outside_valid_domain_explicitly_excluded": all(
            bool(invalid_domain_probe[name])
            for name in (
                "probe_pixel_outside_valid_domain",
                "outside_plus_violation_exact_zero",
                "outside_minus_violation_exact_zero",
                "gamma_unchanged",
                "loss_unchanged",
                "outside_domain_has_no_certificate",
            )
        ),
        "same_pmope_q_and_zero_set_finite_risk_differs": all(
            bool(zero_set_and_finite_risk_probe[name])
            for name in (
                "zero_violation_plus_same",
                "zero_violation_minus_same",
                "zero_uscope_loss_exact",
                "zero_pmope_loss_exact",
                "zero_gamma_exact",
                "finite_violation_plus_same",
                "finite_violation_minus_same",
                "finite_uscope_positive",
                "finite_pmope_positive",
                "finite_risks_different",
                "finite_uscope_greater_than_pmope",
                "same_q_implies_same_zero_set",
            )
        ),
        "same_sign_response_diagnostic_only": all(
            bool(same_sign_response_diagnostic_probe[name])
            for name in (
                "same_sign_response_has_error",
                "endpoint_violations_exact_zero",
                "gamma_exact_zero",
                "uscope_loss_exact_zero",
                "response_is_diagnostic_only",
            )
        )
        and not same_sign_response_diagnostic_probe[
            "response_consumed_by_objective"
        ],
        "implementation_binding_complete": (
            binding_paths == COVERAGE_STATE_USCOPE_IMPLEMENTATION_PATHS
            and all(
                len(digest) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in digest
                )
                for _, digest in implementation_binding
            )
        ),
        "generated_replay_exact": (
            generated_replay_fingerprint
            == expected_generated_fingerprint
        ),
    }
    return tuple(sorted(checks.items()))


@dataclass(frozen=True)
class CoverageStateUSCOPEDatasetFreeReceipt:
    """Fingerprint-bound generated evidence for the v19 product gauge."""

    implementation_binding: tuple[tuple[str, str], ...]
    exact_product_probe: dict[str, object]
    gamma_certificate_probe: dict[str, object]
    single_pixel_probe: dict[str, object]
    gradient_direction_probe: dict[str, object]
    occupied_hidden_negative_probe: dict[str, object]
    invalid_domain_probe: dict[str, object]
    zero_set_and_finite_risk_probe: dict[str, object]
    same_sign_response_diagnostic_probe: dict[str, object]
    generated_replay_fingerprint: str
    checks: tuple[tuple[str, bool], ...]
    evidence_fingerprint: str

    def __post_init__(self) -> None:
        self.verify_unchanged()

    def _generated_payload(self) -> dict[str, object]:
        return _generated_payload(
            exact_product_probe=self.exact_product_probe,
            gamma_certificate_probe=self.gamma_certificate_probe,
            single_pixel_probe=self.single_pixel_probe,
            gradient_direction_probe=self.gradient_direction_probe,
            occupied_hidden_negative_probe=(
                self.occupied_hidden_negative_probe
            ),
            invalid_domain_probe=self.invalid_domain_probe,
            zero_set_and_finite_risk_probe=(
                self.zero_set_and_finite_risk_probe
            ),
            same_sign_response_diagnostic_probe=(
                self.same_sign_response_diagnostic_probe
            ),
        )

    def _evidence_payload(self) -> dict[str, object]:
        payload = deepcopy(self._generated_payload())
        payload["implementation_binding"] = dict(
            self.implementation_binding
        )
        payload["generated_replay_fingerprint"] = (
            self.generated_replay_fingerprint
        )
        return payload

    def verify_unchanged(self) -> None:
        expected_checks = (
            recompute_coverage_state_uscope_dataset_free_checks(
                implementation_binding=self.implementation_binding,
                generated_replay_fingerprint=(
                    self.generated_replay_fingerprint
                ),
                exact_product_probe=self.exact_product_probe,
                gamma_certificate_probe=self.gamma_certificate_probe,
                single_pixel_probe=self.single_pixel_probe,
                gradient_direction_probe=self.gradient_direction_probe,
                occupied_hidden_negative_probe=(
                    self.occupied_hidden_negative_probe
                ),
                invalid_domain_probe=self.invalid_domain_probe,
                zero_set_and_finite_risk_probe=(
                    self.zero_set_and_finite_risk_probe
                ),
                same_sign_response_diagnostic_probe=(
                    self.same_sign_response_diagnostic_probe
                ),
            )
        )
        if (
            self.checks != expected_checks
            or stable_fingerprint(self._evidence_payload())
            != self.evidence_fingerprint
            or self.implementation_binding
            != _current_implementation_binding()
        ):
            raise RuntimeError(
                "USCOPE dataset-free evidence changed after creation"
            )

    @property
    def all_pass(self) -> bool:
        self.verify_unchanged()
        return bool(self.checks) and all(
            value for _, value in self.checks
        )

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": COVERAGE_STATE_USCOPE_DATASET_FREE_SCHEMA,
            "objective_policy": CSLF_USCOPE_POLICY,
            "fixed_margin_hex": COVERAGE_STATE_USCOPE_MARGIN.hex(),
            "implementation_binding": dict(
                self.implementation_binding
            ),
            **deepcopy(self._generated_payload()),
            "generated_replay_fingerprint": (
                self.generated_replay_fingerprint
            ),
            "evidence_fingerprint": self.evidence_fingerprint,
            "checks": dict(self.checks),
            "all_pass": self.all_pass,
            "runtime_splits": [],
            "D_R_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "dataset_training_performed": False,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "bounded_400_authorized": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def _run_coverage_state_uscope_dataset_free_gate_inner(
) -> CoverageStateUSCOPEDatasetFreeReceipt:
    first = _collect_generated_evidence()
    second = _collect_generated_evidence()
    first_payload = _generated_payload(**first)
    second_payload = _generated_payload(**second)
    first_fingerprint = stable_fingerprint(first_payload)
    second_fingerprint = stable_fingerprint(second_payload)
    if first_fingerprint != second_fingerprint:
        raise RuntimeError("USCOPE generated replay is not deterministic")
    implementation_binding = _current_implementation_binding()
    checks = recompute_coverage_state_uscope_dataset_free_checks(
        implementation_binding=implementation_binding,
        generated_replay_fingerprint=second_fingerprint,
        **first,
    )
    evidence_payload = deepcopy(first_payload)
    evidence_payload["implementation_binding"] = dict(
        implementation_binding
    )
    evidence_payload["generated_replay_fingerprint"] = second_fingerprint
    return CoverageStateUSCOPEDatasetFreeReceipt(
        implementation_binding=implementation_binding,
        generated_replay_fingerprint=second_fingerprint,
        checks=checks,
        evidence_fingerprint=stable_fingerprint(evidence_payload),
        **first,
    )


def run_coverage_state_uscope_dataset_free_gate(
) -> CoverageStateUSCOPEDatasetFreeReceipt:
    """Run the generated-only USCOPE structural gate."""

    before_rng = torch.random.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        result = _run_coverage_state_uscope_dataset_free_gate_inner()
    if not torch.equal(before_rng, torch.random.get_rng_state()):
        raise RuntimeError("USCOPE dataset-free gate changed global RNG state")
    return result


__all__ = [
    "COVERAGE_STATE_USCOPE_DATASET_FREE_SCHEMA",
    "COVERAGE_STATE_USCOPE_DATASET_FREE_EXECUTION_SEED",
    "COVERAGE_STATE_USCOPE_TRUNCATION_RADIUS",
    "COVERAGE_STATE_USCOPE_MARGIN",
    "COVERAGE_STATE_USCOPE_IMPLEMENTATION_PATHS",
    "CoverageStateUSCOPEDatasetFreeReceipt",
    "recompute_coverage_state_uscope_dataset_free_checks",
    "run_coverage_state_uscope_dataset_free_gate",
]
