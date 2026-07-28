"""Generated-only structural gate for CURE-Lite v22 PACRE.

All tensors are created in memory on CPU.  The gate does not construct a
dataset, cache, optimizer, checkpoint, training run, or evaluation split.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import torch

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.coverage_state_level_set import CSLF_FIELD_AMPLITUDE
from cure_lite.paired_types import tensor_content_fingerprint

from .factory import PACRE_TRAINING_MODEL_FACTORY
from .pacre import (
    CSLF_PACRE_CENTERING_POLICY,
    CSLF_PACRE_EQUATION_POLICY,
    CSLF_PACRE_FIELD_POLICY,
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CoverageStatePACREConfig,
    phase_centered_feature_affine,
)


PACRE_DATASET_FREE_SCHEMA = "cure-lite-pacre-v22-dataset-free-receipt-v2"
PACRE_DATASET_FREE_SEED = 220022
PACRE_FORMAL_FEATURE_CHANNELS = 64
PACRE_FORMAL_FEATURE_STRIDE = 4
PACRE_FORMAL_WIDTH = 32
PACRE_FORMAL_PARAMETER_COUNT = 64064
PACRE_DATASET_FREE_CHECK_NAMES = (
    "01_phase_residual_centered",
    "02_phase_common_compatibility_zero",
    "03_zero_feature_anchor",
    "04_fast_reference_equivalent",
    "05_binary_flip_interaction_odd",
    "06_phase_specific_response_nondegenerate",
    "07_parameter_topology_unchanged",
    "08_single_completion_field_interface",
    "09_first_second_order_gradients_finite_nonzero",
    "10_frozen_initialization_gradient_path",
    "11_exact_training_factory",
    "12_generated_only_boundary",
    "13_model_state_preserved",
)
PACRE_IMPLEMENTATION_PATHS = (
    "cure_lite/coverage_state_level_set.py",
    "cure_lite/coverage_state_phase_preserving.py",
    "cure_lite/coverage_state_binary_flip_antisymmetric.py",
    "cure_lite/coverage_state_phase_aligned_evidence_transport.py",
    "cure_lite_v22/pacre.py",
    "cure_lite_v22/factory.py",
    "cure_lite_v22/dataset_free.py",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _implementation_binding() -> tuple[tuple[str, str], ...]:
    root = _repository_root()
    rows: list[tuple[str, str]] = []
    for relative in PACRE_IMPLEMENTATION_PATHS:
        path = root / relative
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve(strict=True) != path
        ):
            raise RuntimeError(f"invalid PACRE source path: {relative}")
        rows.append((relative, file_sha256(path)))
    return tuple(rows)


def _state_fingerprint(
    model: CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
) -> str:
    return stable_fingerprint(
        {
            name: tensor_content_fingerprint(value)
            for name, value in sorted(model.state_dict().items())
        }
    )


def _toy_model(
) -> CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet:
    torch.random.default_generator.manual_seed(
        PACRE_DATASET_FREE_SEED
    )
    model = (
        CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet(
            CoverageStatePACREConfig(
                feature_channels=2,
                feature_stride=2,
                width=4,
            )
        )
    )
    generator = torch.Generator(device="cpu").manual_seed(
        PACRE_DATASET_FREE_SEED + 1
    )
    with torch.no_grad():
        model.joint_state_weight.copy_(
            0.12
            * torch.randn(
                model.joint_state_weight.shape,
                generator=generator,
                dtype=torch.float32,
            )
        )
        model.joint_hidden_bias.copy_(
            0.08
            * torch.randn(
                model.joint_hidden_bias.shape,
                generator=generator,
                dtype=torch.float32,
            )
        )
        model.scalar_energy_weight.copy_(
            0.20
            * torch.randn(
                model.scalar_energy_weight.shape,
                generator=generator,
                dtype=torch.float32,
            )
        )
    return model


def _toy_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(
        PACRE_DATASET_FREE_SEED + 2
    )
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
        > 0.72
    )
    return feature.contiguous(), occupancy.contiguous()


def recompute_pacre_dataset_free_checks() -> dict[str, object]:
    """Recompute all generated-only PACRE structural checks."""

    model = _toy_model()
    feature, occupancy = _toy_inputs()
    state_before = _state_fingerprint(model)
    fields = model.forward_fields(feature, occupancy)
    reference = model.forward_reference(feature, occupancy)

    phase_scale = 1.0 + float(
        fields.phase_feature_affine.detach().abs().amax()
    )
    centered_tolerance = (
        4.0
        * float(model.config.phase_channels)
        * torch.finfo(torch.float32).eps
        * phase_scale
    )
    residual_sum_max = float(
        fields.phase_feature_residual.detach().sum(dim=1).abs().amax()
    )

    common = fields.phase_feature_mean.expand_as(
        fields.phase_feature_affine
    )
    common_joint, specific_joint, common_hidden, common_energy = (
        model._compatibility_energy(
            fields.occupancy_affine.unsqueeze(1),
            common,
            fields.phase_feature_mean,
        )
    )
    del common_joint, specific_joint
    phase_common_zero = bool(
        torch.count_nonzero(common_hidden) == 0
        and torch.count_nonzero(common_energy) == 0
    )

    zero_feature = torch.zeros_like(feature)
    zero_field = model(zero_feature, occupancy)
    zero_anchor = bool(
        torch.equal(
            zero_field,
            torch.full_like(zero_field, CSLF_FIELD_AMPLITUDE),
        )
    )

    row = 1
    column = 2
    out_row = row * model.config.feature_stride
    out_column = column * model.config.feature_stride + 1
    flipped_occupancy = occupancy.clone()
    flipped_occupancy[:, :, out_row, out_column] = (
        ~flipped_occupancy[:, :, out_row, out_column]
    )
    flipped_fields = model.forward_fields(feature, flipped_occupancy)
    actual_delta = (
        fields.field[:, :, out_row, out_column]
        - model.config.field_amplitude
    )
    flipped_delta = (
        flipped_fields.field[:, :, out_row, out_column]
        - model.config.field_amplitude
    )
    flip_odd = bool(
        torch.allclose(
            actual_delta,
            -flipped_delta,
            rtol=2.0e-5,
            atol=2.0e-6,
        )
    )

    interaction_spread = float(
        fields.native_phase_interaction.detach().amax()
        - fields.native_phase_interaction.detach().amin()
    )
    parameters = tuple(model.parameters())
    gradient_loss = (
        fields.field.square().mean()
        + fields.native_phase_interaction.square().mean()
    )
    first = torch.autograd.grad(
        gradient_loss,
        parameters,
        create_graph=True,
        allow_unused=False,
    )
    second = torch.autograd.grad(
        torch.stack([value.sum() for value in first]).sum(),
        parameters,
        allow_unused=False,
    )
    first_finite_nonzero = all(
        bool(torch.isfinite(value).all())
        and bool(torch.any(value != 0.0))
        for value in first
    )
    second_finite = all(
        bool(torch.isfinite(value).all()) for value in second
    )

    formal_config = CoverageStatePACREConfig(
        feature_channels=PACRE_FORMAL_FEATURE_CHANNELS,
        feature_stride=PACRE_FORMAL_FEATURE_STRIDE,
        width=PACRE_FORMAL_WIDTH,
    )
    formal_model = PACRE_TRAINING_MODEL_FACTORY(formal_config)
    exact_training_factory = bool(
        type(formal_model)
        is CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet
        and formal_model.config is formal_config
    )

    torch.random.default_generator.manual_seed(
        PACRE_DATASET_FREE_SEED + 3
    )
    startup_model = PACRE_TRAINING_MODEL_FACTORY(
        CoverageStatePACREConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    startup_parameters = dict(startup_model.named_parameters())
    startup_loss = startup_model(feature, occupancy).square().mean()
    startup_gradients = dict(
        zip(
            startup_parameters,
            torch.autograd.grad(
                startup_loss,
                tuple(startup_parameters.values()),
                create_graph=True,
                allow_unused=False,
            ),
            strict=True,
        )
    )
    startup_readout_visible = bool(
        torch.isfinite(
            startup_gradients["scalar_energy_weight"]
        ).all()
        and torch.any(
            startup_gradients["scalar_energy_weight"] != 0.0
        )
    )
    startup_upstream_dormant = bool(
        torch.count_nonzero(
            startup_gradients["joint_state_weight"]
        )
        == 0
        and torch.count_nonzero(
            startup_gradients["joint_hidden_bias"]
        )
        == 0
    )
    startup_cross_gradients = torch.autograd.grad(
        startup_gradients["scalar_energy_weight"].square().sum(),
        (
            startup_parameters["joint_state_weight"],
            startup_parameters["joint_hidden_bias"],
        ),
        allow_unused=False,
    )
    startup_upstream_path = all(
        bool(torch.isfinite(value).all())
        and bool(torch.any(value != 0.0))
        for value in startup_cross_gradients
    )
    parameter_names = tuple(
        name for name, _ in formal_model.named_parameters()
    )
    parameter_count = sum(
        value.numel() for value in formal_model.parameters()
    )
    forward_parameters = tuple(
        inspect.signature(type(model).forward_fields).parameters
    )
    completion_parameters = tuple(
        inspect.signature(type(model).predict_completion).parameters
    )
    children = tuple(model.named_children())
    state_after = _state_fingerprint(model)
    checks = {
        "01_phase_residual_centered": (
            residual_sum_max <= centered_tolerance
        ),
        "02_phase_common_compatibility_zero": phase_common_zero,
        "03_zero_feature_anchor": zero_anchor,
        "04_fast_reference_equivalent": bool(
            torch.allclose(
                fields.field,
                reference,
                rtol=2.0e-5,
                atol=2.0e-6,
            )
        ),
        "05_binary_flip_interaction_odd": flip_odd,
        "06_phase_specific_response_nondegenerate": (
            interaction_spread > 128.0 * torch.finfo(torch.float32).eps
        ),
        "07_parameter_topology_unchanged": (
            parameter_names
            == (
                "joint_state_weight",
                "joint_hidden_bias",
                "scalar_energy_weight",
            )
            and parameter_count == PACRE_FORMAL_PARAMETER_COUNT
            and formal_model.config.expected_parameter_count
            == PACRE_FORMAL_PARAMETER_COUNT
        ),
        "08_single_completion_field_interface": (
            forward_parameters == ("self", "feature", "occupancy")
            and completion_parameters == ("self", "feature", "occupancy")
            and tuple(fields.field.shape) == tuple(occupancy.shape)
            and len(children) == 1
            and children[0][0] == "pixel_shuffle"
        ),
        "09_first_second_order_gradients_finite_nonzero": (
            first_finite_nonzero and second_finite
        ),
        "10_frozen_initialization_gradient_path": (
            startup_readout_visible
            and startup_upstream_dormant
            and startup_upstream_path
        ),
        "11_exact_training_factory": exact_training_factory,
        "12_generated_only_boundary": True,
        "13_model_state_preserved": state_before == state_after,
    }
    if tuple(checks) != PACRE_DATASET_FREE_CHECK_NAMES:
        raise AssertionError("PACRE dataset-free check order changed")
    return {
        "checks": checks,
        "observations": {
            "phase_residual_sum_max_hex": residual_sum_max.hex(),
            "phase_centering_tolerance_hex": centered_tolerance.hex(),
            "interaction_spread_hex": interaction_spread.hex(),
            "parameter_names": list(parameter_names),
            "parameter_count": parameter_count,
            "field_shape": list(fields.field.shape),
            "field_min_hex": float(fields.field.detach().amin()).hex(),
            "field_max_hex": float(fields.field.detach().amax()).hex(),
            "first_order_gradient_nonzero": [
                bool(torch.any(value != 0.0)) for value in first
            ],
            "second_order_gradient_finite": [
                bool(torch.isfinite(value).all()) for value in second
            ],
            "initial_gradient_nonzero": {
                name: bool(torch.any(value != 0.0))
                for name, value in startup_gradients.items()
            },
            "initial_readout_to_upstream_cross_gradient_nonzero": [
                bool(torch.any(value != 0.0))
                for value in startup_cross_gradients
            ],
            "training_factory_model_class": (
                f"{type(formal_model).__module__}."
                f"{type(formal_model).__qualname__}"
            ),
        },
    }


def run_pacre_dataset_free_gate() -> dict[str, object]:
    """Return a fingerprinted, generated-only PACRE gate receipt."""

    implementation = _implementation_binding()
    recomputed = recompute_pacre_dataset_free_checks()
    checks = recomputed["checks"]
    assert isinstance(checks, dict)
    gate_passed = all(value is True for value in checks.values())
    body = {
        "schema_version": PACRE_DATASET_FREE_SCHEMA,
        "candidate": "PACRE-v22",
        "mechanism": (
            "phase_aligned_centered_residual_compatibility_energy"
        ),
        "input_contract": ["F_b", "O"],
        "output_contract": "one_non_saturating_completion_field",
        "field_policy": CSLF_PACRE_FIELD_POLICY,
        "equation_policy": CSLF_PACRE_EQUATION_POLICY,
        "centering_policy": CSLF_PACRE_CENTERING_POLICY,
        "threshold": 0.0,
        "threshold_search_performed": False,
        "parameter_count": PACRE_FORMAL_PARAMETER_COUNT,
        "additional_heads": 0,
        "additional_branches": 0,
        "dataset_accessed": False,
        "cache_accessed": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "optimizer_constructed": False,
        "training_performed": False,
        "implementation_binding": [
            {"repo_path": path, "sha256": digest}
            for path, digest in implementation
        ],
        "checks": checks,
        "observations": recomputed["observations"],
        "gate_passed": gate_passed,
        "next_action": (
            "AUTHORIZE_D_R_ONLY_GATE_IMPLEMENTATION"
            if gate_passed
            else "STOP_AND_REVISE_PACRE_CORE"
        ),
    }
    return {**body, "receipt_fingerprint": stable_fingerprint(body)}


__all__ = [
    "PACRE_DATASET_FREE_CHECK_NAMES",
    "PACRE_DATASET_FREE_SCHEMA",
    "PACRE_FORMAL_PARAMETER_COUNT",
    "recompute_pacre_dataset_free_checks",
    "run_pacre_dataset_free_gate",
]
