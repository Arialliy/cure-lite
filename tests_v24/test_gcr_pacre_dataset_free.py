"""Dataset-free structural and algebra tests for GCR-PACRE v24."""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from dataclasses import replace
import inspect
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest
import torch
from torch import Tensor

from cure_lite.coverage_state_level_set import CSLF_FIELD_AMPLITUDE
from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
)
from cure_lite_v23.pacre_vc import (
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREVerifierCorrectedConfig,
)
from cure_lite_v24 import (
    CSLF_GCR_PACRE_EQUATION_POLICY,
    CSLF_GCR_PACRE_FIELD_POLICY,
    CURELiteGatedCommonResidualPACRELevelSet,
    CoverageStateGCRPACREConfig,
    CoverageStateGCRPACREFields,
    GCRPACREFP64OracleFields,
    GCR_PACRE_CANDIDATE,
    GCR_PACRE_ENERGY_POLICY,
    GCR_PACRE_FIELDS_FQCN,
    GCR_PACRE_FP64_ORACLE_ABS_TOL,
    GCR_PACRE_FP64_ORACLE_MAX_ULP,
    GCR_PACRE_FORMAL_PARAMETER_COUNT,
    GCR_PACRE_GATE_STATISTICS_SCHEMA,
    GCR_PACRE_INTERACTION_POLICY,
    GCR_PACRE_METHOD_ID,
    GCR_PACRE_NUMERICAL_POLICY,
    GCR_PACRE_PARAMETER_NAMES,
    build_formal_gcr_pacre_training_model,
    build_gcr_pacre_training_model,
    compare_gcr_pacre_fp32_to_fp64_oracle,
    gcr_pacre_fp32_ulp_distance,
    summarize_gcr_pacre_gate_saturation,
    validate_gcr_pacre_fields,
    validate_gcr_pacre_fields_contract,
)
from cure_lite_v24 import gcr_pacre as gcr_pacre_module


REPO_ROOT = Path(__file__).resolve().parents[1]


def _config() -> CoverageStateGCRPACREConfig:
    return CoverageStateGCRPACREConfig(
        feature_channels=2,
        feature_stride=2,
        width=4,
    )


def _inputs(seed: int = 240002) -> tuple[Tensor, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
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
        > 0.68
    )
    return feature.contiguous(), occupancy.contiguous()


def _nonzero_model(
    seed: int = 240003,
) -> CURELiteGatedCommonResidualPACRELevelSet:
    return _nonzero_model_for_config(_config(), seed=seed)


def _nonzero_model_for_config(
    config: CoverageStateGCRPACREConfig,
    *,
    seed: int,
) -> CURELiteGatedCommonResidualPACRELevelSet:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = CURELiteGatedCommonResidualPACRELevelSet(config)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    with torch.no_grad():
        model.joint_state_weight.copy_(
            0.15
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
            0.25
            * torch.randn(
                model.scalar_energy_weight.shape,
                generator=generator,
                dtype=torch.float32,
            )
        )
    return model


def _assert_raw_equal(first: Tensor, second: Tensor) -> None:
    assert first.shape == second.shape
    assert first.dtype == second.dtype
    assert first.device == second.device
    assert torch.equal(
        first.detach().contiguous().view(torch.uint8),
        second.detach().contiguous().view(torch.uint8),
    )


def test_v24_policy_identity_is_canonical_and_fail_closed() -> None:
    config = _config()

    assert GCR_PACRE_CANDIDATE == "GCR-PACRE-v24"
    assert config.method_id == GCR_PACRE_METHOD_ID
    assert config.field_policy == CSLF_GCR_PACRE_FIELD_POLICY
    assert config.equation_policy == CSLF_GCR_PACRE_EQUATION_POLICY
    assert config.interaction_policy == GCR_PACRE_INTERACTION_POLICY
    assert config.energy_policy == GCR_PACRE_ENERGY_POLICY
    assert config.numerical_policy == GCR_PACRE_NUMERICAL_POLICY
    assert config.method_id == "cure_lite_gcr_pacre_v24"
    assert config.field_policy == (
        "gcr_pacre_single_zero_level_set_field_v1"
    )
    assert config.equation_policy == (
        "flip_even_common_gate_times_flip_odd_residual_v1"
    )

    with pytest.raises(
        ValueError,
        match="GCR-PACRE fixes equation_policy",
    ):
        CoverageStateGCRPACREConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
            equation_policy="legacy_or_unknown",
        )
    with pytest.raises(
        ValueError,
        match="GCR-PACRE fixes numerical_policy",
    ):
        CoverageStateGCRPACREConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
            numerical_policy="strict_open_interval",
        )
    for invalid_radius in (2.0, True):
        with pytest.raises(
            ValueError,
            match="GCR-PACRE fixes coarse_radius",
        ):
            CoverageStateGCRPACREConfig(
                feature_channels=2,
                feature_stride=2,
                width=4,
                coarse_radius=invalid_radius,
            )


def test_formal_factory_preserves_the_64_4_32_parameter_schema() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(240010)
        model = build_formal_gcr_pacre_training_model()

    assert type(model) is CURELiteGatedCommonResidualPACRELevelSet
    assert model.config.feature_channels == 64
    assert model.config.feature_stride == 4
    assert model.config.width == 32
    assert model.config.expected_parameter_count == 64064
    assert GCR_PACRE_FORMAL_PARAMETER_COUNT == 64064
    assert tuple(name for name, _ in model.named_parameters()) == (
        GCR_PACRE_PARAMETER_NAMES
    )
    assert {
        name: tuple(parameter.shape)
        for name, parameter in model.named_parameters()
    } == {
        "joint_state_weight": (32, 80, 5, 5),
        "joint_hidden_bias": (32,),
        "scalar_energy_weight": (32,),
    }
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        64064
    )
    assert all(
        parameter.dtype == torch.float32
        and parameter.requires_grad
        for parameter in model.parameters()
    )
    assert len(tuple(model.named_children())) == 1
    assert tuple(model.named_children())[0][0] == "pixel_shuffle"


def test_factory_rejects_legacy_and_derived_configs() -> None:
    with pytest.raises(TypeError, match="exact type"):
        build_gcr_pacre_training_model(
            CoverageStatePACREVerifierCorrectedConfig(
                feature_channels=2,
                feature_stride=2,
                width=4,
            )
        )

    class DerivedConfig(CoverageStateGCRPACREConfig):
        pass

    with pytest.raises(TypeError, match="exact type"):
        build_gcr_pacre_training_model(
            DerivedConfig(
                feature_channels=2,
                feature_stride=2,
                width=4,
            )
        )


def test_fields_restore_the_complete_replay_ledger() -> None:
    model = _nonzero_model()
    feature, occupancy = _inputs()
    fields = model.forward_fields(feature, occupancy)
    names = {item.name for item in dataclass_fields(type(fields))}
    inherited_replay_names = {
        "encoded_feature",
        "phase_occupancy",
        "occupancy_affine",
        "coarse_feature_affine",
        "upsampled_feature_affine",
        "phase_feature_affine",
        "phase_feature_mean",
        "phase_feature_residual",
        "actual_common_joint_affine",
        "actual_specific_joint_affine",
        "actual_common_silu",
        "actual_residual_hidden",
        "actual_residual_energy",
        "actual_common_hidden",
        "actual_common_energy",
        "center_phase_weight",
        "flip_delta",
        "flipped_center_phase_value",
        "flipped_occupancy_affine",
        "flipped_common_joint_affine",
        "flipped_specific_joint_affine",
        "flipped_common_silu",
        "flipped_residual_hidden",
        "flipped_residual_energy",
        "flipped_common_hidden",
        "flipped_common_energy",
        "residual_odd_interaction",
        "common_even_energy",
        "common_gate",
        "gated_interaction",
        "native_phase_field",
        "field",
        "output_size",
    }
    gcr_names = {
        "actual_occupancy_only_joint_affine",
        "flipped_occupancy_only_joint_affine",
        "common_gate_zero_saturation",
        "common_gate_two_saturation",
    }

    assert type(fields) is CoverageStateGCRPACREFields
    assert inherited_replay_names <= names
    assert gcr_names <= names
    assert (
        f"{type(fields).__module__}.{type(fields).__qualname__}"
        == GCR_PACRE_FIELDS_FQCN
    )
    assert "forward_reference" in type(model).__dict__
    assert type(model).forward_reference is not (
        CURELitePhaseAlignedEvidenceTransportLevelSet.forward_reference
    )


def test_forward_calls_the_fields_validator_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = (
        CURELiteGatedCommonResidualPACRELevelSet._validate_gcr_fields
    )

    def counted(
        self: CURELiteGatedCommonResidualPACRELevelSet,
        fields: CoverageStateGCRPACREFields,
        *,
        feature: Tensor,
        occupancy: Tensor,
    ) -> None:
        nonlocal calls
        calls += 1
        original(
            self,
            fields,
            feature=feature,
            occupancy=occupancy,
        )

    monkeypatch.setattr(
        CURELiteGatedCommonResidualPACRELevelSet,
        "_validate_gcr_fields",
        counted,
    )
    model = _nonzero_model(seed=240015)
    feature, occupancy = _inputs(seed=240016)

    model.forward_fields(feature, occupancy)

    assert calls == 1


def test_step_zero_forward_and_gradients_match_v23_with_frozen_bound() -> None:
    seed = 240020
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        v23 = CURELitePACREVerifierCorrectedLevelSet(
            CoverageStatePACREVerifierCorrectedConfig(
                feature_channels=2,
                feature_stride=2,
                width=4,
            )
        )
        torch.random.default_generator.manual_seed(seed)
        v24 = CURELiteGatedCommonResidualPACRELevelSet(_config())
    feature, occupancy = _inputs(seed + 1)

    assert tuple(v23.state_dict()) == tuple(v24.state_dict())
    for name in v23.state_dict():
        _assert_raw_equal(v23.state_dict()[name], v24.state_dict()[name])

    fields_v23 = v23.forward_fields(feature, occupancy)
    fields_v24 = v24.forward_fields(feature, occupancy)
    _assert_raw_equal(fields_v23.field, fields_v24.field)
    assert torch.count_nonzero(fields_v24.residual_odd_interaction) == 0
    assert torch.count_nonzero(fields_v24.common_even_energy) == 0
    assert torch.equal(
        fields_v24.common_gate,
        torch.ones_like(fields_v24.common_gate),
    )

    weights = torch.linspace(
        0.3,
        1.7,
        fields_v23.field.numel(),
        dtype=torch.float32,
    ).reshape_as(fields_v23.field)
    parameters_v23 = tuple(v23.parameters())
    parameters_v24 = tuple(v24.parameters())
    gradients_v23 = torch.autograd.grad(
        (fields_v23.field * weights).sum(),
        parameters_v23,
        allow_unused=False,
    )
    gradients_v24 = torch.autograd.grad(
        (fields_v24.field * weights).sum(),
        parameters_v24,
        allow_unused=False,
    )
    for first, second in zip(
        gradients_v23,
        gradients_v24,
        strict=True,
    ):
        gradient_roundoff_bound = 2.0 * torch.finfo(
            torch.float32
        ).eps * (
            1.0
            + float(first.detach().abs().amax())
            + float(second.detach().abs().amax())
        )
        torch.testing.assert_close(
            first,
            second,
            rtol=0.0,
            atol=gradient_roundoff_bound,
        )
        assert bool(torch.isfinite(first).all())
        assert bool(torch.isfinite(second).all())
    assert torch.count_nonzero(gradients_v24[2]) > 0


def test_post_warmup_gate_path_is_reachable_in_isolation() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(240030)
        model = CURELiteGatedCommonResidualPACRELevelSet(_config())
    feature, occupancy = _inputs(seed=240031)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0e-3,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    startup = model.forward_fields(feature, occupancy)
    warmup_target = torch.linspace(
        -0.9,
        0.9,
        startup.field.numel(),
        dtype=torch.float32,
    ).reshape_as(startup.field)
    warmup_loss = (startup.field - warmup_target).square().mean()
    warmup_loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Exactly one frozen generated update activates the shared readout.
    fields = model.forward_fields(feature, occupancy)
    weights = torch.linspace(
        0.2,
        1.4,
        fields.common_gate.numel(),
        dtype=torch.float32,
    ).reshape_as(fields.common_gate)
    residual_only_probe = (
        fields.residual_odd_interaction * weights
    ).sum()
    gate_only_probe = (
        fields.common_gate
        * fields.residual_odd_interaction.detach()
        * weights
    ).sum()
    total_probe = (fields.gated_interaction * weights).sum()
    residual_gradients = torch.autograd.grad(
        residual_only_probe,
        tuple(model.parameters()),
        retain_graph=True,
        allow_unused=False,
    )
    gate_gradients = torch.autograd.grad(
        gate_only_probe,
        tuple(model.parameters()),
        retain_graph=True,
        allow_unused=False,
    )
    total_gradients = torch.autograd.grad(
        total_probe,
        tuple(model.parameters()),
        allow_unused=False,
    )

    assert torch.count_nonzero(fields.residual_odd_interaction) > 0
    assert torch.count_nonzero(fields.common_even_energy) > 0
    for gradients in (
        residual_gradients,
        gate_gradients,
        total_gradients,
    ):
        assert all(bool(torch.isfinite(value).all()) for value in gradients)
        assert all(torch.count_nonzero(value) > 0 for value in gradients)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_zero_feature_is_the_exact_positive_point_nine_anchor() -> None:
    model = _nonzero_model(seed=240040)
    _, occupancy = _inputs(seed=240041)
    feature = torch.zeros((1, 2, 3, 4), dtype=torch.float32)

    fields = model.forward_fields(feature, occupancy)

    assert torch.count_nonzero(fields.residual_odd_interaction) == 0
    assert torch.count_nonzero(fields.common_even_energy) == 0
    assert torch.equal(
        fields.common_gate,
        torch.ones_like(fields.common_gate),
    )
    assert torch.equal(
        fields.field,
        torch.full_like(fields.field, CSLF_FIELD_AMPLITUDE),
    )


def test_fixed_zero_threshold_and_hard_union_are_unchanged() -> None:
    model = _nonzero_model(seed=240045)
    feature, occupancy = _inputs(seed=240046)
    field = model(feature, occupancy)
    completion = model.predict_completion(feature, occupancy)
    union = model.predict_union(feature, occupancy)

    assert torch.equal(completion, (field < 0.0) & ~occupancy)
    assert torch.equal(union, occupancy | completion)
    assert torch.equal(union & occupancy, occupancy)


def test_designated_binary_flip_has_exact_odd_even_parity() -> None:
    model = _nonzero_model(seed=240050)
    feature, occupancy = _inputs(seed=240051)
    native_row, native_column, phase_index = 1, 2, 1
    phase_row, phase_column = divmod(
        phase_index,
        model.config.feature_stride,
    )
    output_row = (
        native_row * model.config.feature_stride + phase_row
    )
    output_column = (
        native_column * model.config.feature_stride + phase_column
    )
    flipped = occupancy.clone()
    flipped[0, 0, output_row, output_column] = ~flipped[
        0,
        0,
        output_row,
        output_column,
    ]

    first = model.forward_fields(feature, occupancy)
    second = model.forward_fields(feature, flipped)
    coordinate = (0, phase_index, native_row, native_column)

    epsilon = torch.finfo(torch.float32).eps
    parity_cases = (
        (
            first.residual_odd_interaction[coordinate],
            -second.residual_odd_interaction[coordinate],
        ),
        (
            first.common_even_energy[coordinate],
            second.common_even_energy[coordinate],
        ),
        (
            first.common_gate[coordinate],
            second.common_gate[coordinate],
        ),
        (
            first.gated_interaction[coordinate],
            -second.gated_interaction[coordinate],
        ),
    )
    for actual, expected in parity_cases:
        analytic_roundoff_bound = 8.0 * epsilon * (
            1.0
            + float(actual.detach().abs())
            + float(expected.detach().abs())
        )
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.0,
            atol=analytic_roundoff_bound,
        )


def test_fast_field_matches_the_independent_literal_reference() -> None:
    model = _nonzero_model(seed=240060)
    feature, occupancy = _inputs(seed=240061)

    fields = model.forward_fields(feature, occupancy)
    fast = fields.field
    reference_fields = model.forward_reference_fields_fp64(
        feature,
        occupancy,
    )
    reference = model.forward_reference(feature, occupancy)
    legacy_ungated = model.pixel_shuffle(
        model.config.field_amplitude
        + fields.residual_odd_interaction
    )

    comparison = compare_gcr_pacre_fp32_to_fp64_oracle(
        fast,
        reference,
    )
    assert type(reference_fields) is GCRPACREFP64OracleFields
    assert reference.dtype == torch.float64
    assert torch.equal(reference, reference_fields.field)
    assert comparison.passed
    assert (
        comparison.absolute_tolerance
        == GCR_PACRE_FP64_ORACLE_ABS_TOL
    )
    assert (
        comparison.maximum_allowed_ulp
        == GCR_PACRE_FP64_ORACLE_MAX_ULP
    )
    assert bool(torch.any(fields.common_gate != 1.0))
    assert not torch.equal(fast, legacy_ungated)


def test_forced_unit_gate_is_read_only_same_weight_ablation() -> None:
    model = _nonzero_model(seed=240065)
    feature, occupancy = _inputs(seed=240066)
    fields = model.forward_fields(feature, occupancy)
    expected = model.pixel_shuffle(
        model.config.field_amplitude
        + fields.residual_odd_interaction
    ).contiguous()
    state_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    gradients_before = tuple(
        parameter.grad for parameter in model.parameters()
    )

    actual = model.forward_forced_unit_gate(feature, occupancy)

    assert torch.equal(actual, expected)
    assert not torch.equal(actual, fields.field)
    assert state_before.keys() == model.state_dict().keys()
    assert all(
        torch.equal(state_before[name], value)
        for name, value in model.state_dict().items()
    )
    assert gradients_before == tuple(
        parameter.grad for parameter in model.parameters()
    )
    assert tuple(name for name, _ in model.named_parameters()) == (
        GCR_PACRE_PARAMETER_NAMES
    )


def test_fp32_gate_endpoints_are_valid_and_saturation_is_recorded() -> None:
    model = CURELiteGatedCommonResidualPACRELevelSet(
        CoverageStateGCRPACREConfig(
            feature_channels=1,
            feature_stride=2,
            width=1,
        )
    )
    feature = torch.ones((1, 1, 2, 2), dtype=torch.float32)
    occupancy = torch.zeros((1, 1, 4, 4), dtype=torch.bool)
    with torch.no_grad():
        model.joint_state_weight.zero_()
        model.joint_hidden_bias.zero_()
        model.joint_state_weight[0, 0, 1, 1] = 20.0
        model.scalar_energy_weight.fill_(20.0)

    upper = model.forward_fields(feature, occupancy)
    upper_audit = summarize_gcr_pacre_gate_saturation(upper)
    assert torch.count_nonzero(upper.common_even_energy) > 0
    common_only_roundoff_bound = 8.0 * torch.finfo(
        torch.float32
    ).eps
    assert float(
        upper.residual_odd_interaction.detach().abs().amax()
    ) <= common_only_roundoff_bound
    assert float(
        upper.gated_interaction.detach().abs().amax()
    ) <= 2.0 * common_only_roundoff_bound
    torch.testing.assert_close(
        upper.field,
        torch.full_like(upper.field, CSLF_FIELD_AMPLITUDE),
        rtol=0.0,
        atol=2.0 * common_only_roundoff_bound,
    )
    assert bool(torch.all(upper.field > 0.0))
    assert upper_audit.two_count > 0
    assert upper_audit.zero_count == 0
    assert upper_audit.schema == GCR_PACRE_GATE_STATISTICS_SCHEMA
    assert (
        upper_audit.saturated_count
        == upper_audit.zero_count + upper_audit.two_count
    )
    assert (
        upper_audit.interior_count + upper_audit.saturated_count
        == upper_audit.element_count
    )
    assert torch.count_nonzero(
        upper.common_gate_two_saturation
    ) == upper_audit.two_count
    assert bool(torch.all(upper.common_gate <= 2.0))

    with torch.no_grad():
        model.scalar_energy_weight.fill_(-20.0)
    lower = model.forward_fields(feature, occupancy)
    lower_audit = summarize_gcr_pacre_gate_saturation(lower)
    assert lower_audit.zero_count > 0
    assert lower_audit.two_count == 0
    assert lower_audit.schema == GCR_PACRE_GATE_STATISTICS_SCHEMA
    assert torch.count_nonzero(
        lower.common_gate_zero_saturation
    ) == lower_audit.zero_count
    assert bool(torch.all(lower.common_gate >= 0.0))


def test_external_full_replay_verifier_passes_without_mutation() -> None:
    model = _nonzero_model(seed=240070)
    feature, occupancy = _inputs(seed=240071)
    fields = model.forward_fields(feature, occupancy)
    model_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    gradients_before = tuple(
        parameter.grad for parameter in model.parameters()
    )

    validate_gcr_pacre_fields(
        model,
        fields,
        feature=feature,
        occupancy=occupancy,
    )

    assert all(
        torch.equal(model_before[name], value)
        for name, value in model.state_dict().items()
    )
    assert gradients_before == tuple(
        parameter.grad for parameter in model.parameters()
    )


def test_external_full_replay_verifier_rejects_tampering() -> None:
    model = _nonzero_model(seed=240080)
    feature, occupancy = _inputs(seed=240081)
    fields = model.forward_fields(feature, occupancy)
    changed = fields.actual_common_silu.clone()
    changed[0, 1, 0, 1, 2] += 0.125
    tampered = replace(
        fields,
        actual_common_silu=changed,
    )

    with pytest.raises(
        AssertionError,
        match="actual_common_silu failed exact forward replay",
    ):
        validate_gcr_pacre_fields(
            model,
            tampered,
            feature=feature,
            occupancy=occupancy,
        )

    residual = fields.phase_feature_residual
    noncontiguous = (
        residual.transpose(-1, -2)
        .contiguous()
        .transpose(-1, -2)
    )
    assert noncontiguous.shape == residual.shape
    assert not noncontiguous.is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        validate_gcr_pacre_fields(
            model,
            replace(
                fields,
                phase_feature_residual=noncontiguous,
            ),
            feature=feature,
            occupancy=occupancy,
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "phase_occupancy",
        "occupancy_affine",
        "center_phase_weight",
        "flipped_center_phase_value",
        "flip_delta",
        "flipped_occupancy_affine",
        "actual_residual_energy",
        "flipped_residual_energy",
        "residual_odd_interaction",
        "actual_common_energy",
        "flipped_common_energy",
        "common_even_energy",
        "common_gate",
        "common_gate_zero_saturation",
        "common_gate_two_saturation",
        "gated_interaction",
        "native_phase_field",
        "field",
    ),
)
def test_lightweight_equation_validator_rejects_each_chain_tamper(
    field_name: str,
) -> None:
    model = _nonzero_model(seed=240090)
    feature, occupancy = _inputs(seed=240091)
    fields = model.forward_fields(feature, occupancy)
    changed = getattr(fields, field_name).clone()
    flattened = changed.reshape(-1)
    if changed.dtype == torch.bool:
        flattened[0] = ~flattened[0]
    else:
        flattened[0] += 0.125
    tampered = replace(fields, **{field_name: changed.contiguous()})

    with pytest.raises(
        FloatingPointError,
        match="finite/equation/gate contract failed",
    ):
        validate_gcr_pacre_fields_contract(
            model,
            tampered,
            feature=feature,
            occupancy=occupancy,
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "encoded_feature",
        "occupancy_affine",
        "coarse_feature_affine",
        "upsampled_feature_affine",
        "phase_feature_affine",
        "phase_feature_mean",
        "phase_feature_residual",
        "actual_occupancy_only_joint_affine",
        "actual_common_joint_affine",
        "actual_specific_joint_affine",
        "actual_common_silu",
        "actual_residual_hidden",
        "actual_residual_energy",
        "actual_common_hidden",
        "actual_common_energy",
        "center_phase_weight",
        "flip_delta",
        "flipped_occupancy_affine",
        "flipped_occupancy_only_joint_affine",
        "flipped_common_joint_affine",
        "flipped_specific_joint_affine",
        "flipped_common_silu",
        "flipped_residual_hidden",
        "flipped_residual_energy",
        "flipped_common_hidden",
        "flipped_common_energy",
        "residual_odd_interaction",
        "common_even_energy",
        "common_gate",
        "gated_interaction",
        "native_phase_field",
        "field",
    ),
)
def test_lightweight_validator_rejects_nonfinite_in_every_float_ledger_lane(
    field_name: str,
) -> None:
    model = _nonzero_model(seed=240095)
    feature, occupancy = _inputs(seed=240096)
    fields = model.forward_fields(feature, occupancy)
    changed = getattr(fields, field_name).clone()
    with torch.no_grad():
        changed.reshape(-1)[0] = float("nan")

    with pytest.raises(
        FloatingPointError,
        match="finite/equation/gate contract failed",
    ):
        validate_gcr_pacre_fields_contract(
            model,
            replace(fields, **{field_name: changed.contiguous()}),
            feature=feature,
            occupancy=occupancy,
        )


def test_lightweight_validator_has_one_aggregate_sync_and_no_full_replay() -> None:
    contract_source = inspect.getsource(
        gcr_pacre_module._validate_gcr_pacre_fields_contract
    )
    full_source = inspect.getsource(
        gcr_pacre_module.validate_gcr_pacre_fields
    )

    assert (
        contract_source.count(
            "bool(torch.stack(finite_checks).all())"
        )
        == 1
    )
    assert ".item(" not in contract_source
    assert "_affine_states" not in contract_source
    assert "_compatibility_components" not in contract_source
    assert "_affine_states" in full_source
    assert "_compatibility_components" in full_source


def test_local_phase_centering_has_no_v22_private_dependency() -> None:
    module_source = inspect.getsource(gcr_pacre_module)
    model = _nonzero_model(seed=240100)
    feature, occupancy = _inputs(seed=240101)
    fields = model.forward_fields(feature, occupancy)
    expected_mean = fields.phase_feature_affine.mean(
        dim=1,
        keepdim=True,
    ).contiguous()
    expected_residual = (
        fields.phase_feature_affine - expected_mean
    ).contiguous()

    assert "cure_lite_v22" not in module_source
    assert "_phase_centered_feature_affine_unchecked" not in module_source
    assert torch.equal(fields.phase_feature_mean, expected_mean)
    assert torch.equal(fields.phase_feature_residual, expected_residual)
    assert fields.flipped_center_phase_value.shape == (
        feature.shape[0],
        model.config.phase_channels,
        feature.shape[-2],
        feature.shape[-1],
    )
    assert torch.equal(
        fields.flipped_center_phase_value,
        ~fields.phase_occupancy,
    )


def test_fp64_oracle_is_independent_of_fast_private_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _nonzero_model(seed=240110)
    feature, occupancy = _inputs(seed=240111)
    fast = model.forward_fields(feature, occupancy).field.detach().clone()
    state_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("fast-path helper entered the FP64 oracle")

    monkeypatch.setattr(
        CURELiteGatedCommonResidualPACRELevelSet,
        "_affine_states",
        forbidden,
    )
    monkeypatch.setattr(
        CURELiteGatedCommonResidualPACRELevelSet,
        "_compatibility_components",
        forbidden,
    )
    monkeypatch.setattr(
        gcr_pacre_module,
        "normalize_cslf_feature",
        forbidden,
    )
    monkeypatch.setattr(
        gcr_pacre_module,
        "pixel_unshuffle_bool_occupancy",
        forbidden,
    )

    oracle = model.forward_reference_fields_fp64(feature, occupancy)
    comparison = compare_gcr_pacre_fp32_to_fp64_oracle(
        fast,
        oracle.field,
    )

    assert comparison.passed
    assert all(
        value.dtype == torch.float64
        and not value.requires_grad
        for value in (
            oracle.residual_odd_interaction,
            oracle.common_even_energy,
            oracle.common_gate,
            oracle.gated_interaction,
            oracle.native_phase_field,
            oracle.field,
        )
    )
    assert all(
        torch.equal(state_before[name], value)
        for name, value in model.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    oracle_source = inspect.getsource(
        CURELiteGatedCommonResidualPACRELevelSet
        .forward_reference_fields_fp64
    )
    assert "_affine_states" not in oracle_source
    assert "_compatibility_components" not in oracle_source
    assert "normalize_cslf_feature" not in oracle_source
    assert "pixel_unshuffle_bool_occupancy" not in oracle_source


@pytest.mark.parametrize(
    (
        "feature_channels",
        "stride",
        "hidden",
        "batch",
        "height",
        "width",
        "seed",
    ),
    (
        (2, 2, 3, 2, 1, 1, 240120),
        (3, 3, 4, 2, 2, 3, 240130),
        (1, 4, 2, 1, 1, 2, 240140),
    ),
)
def test_fp64_oracle_envelope_covers_boundaries_phases_and_batches(
    feature_channels: int,
    stride: int,
    hidden: int,
    batch: int,
    height: int,
    width: int,
    seed: int,
) -> None:
    config = CoverageStateGCRPACREConfig(
        feature_channels=feature_channels,
        feature_stride=stride,
        width=hidden,
    )
    model = _nonzero_model_for_config(config, seed=seed)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    feature = torch.randn(
        (batch, feature_channels, height, width),
        generator=generator,
        dtype=torch.float32,
    ).contiguous()
    occupancy = (
        torch.rand(
            (batch, 1, height * stride, width * stride),
            generator=generator,
            dtype=torch.float32,
        )
        > 0.55
    ).contiguous()

    fast = model.forward_fields(feature, occupancy)
    oracle = model.forward_reference_fields_fp64(feature, occupancy)
    comparison = compare_gcr_pacre_fp32_to_fp64_oracle(
        fast.field,
        oracle.field,
    )

    # This literal pair is the pre-frozen output envelope, not a tolerance
    # inferred from the observations produced by this parametrized test.
    assert GCR_PACRE_FP64_ORACLE_ABS_TOL == 2.0e-6
    assert GCR_PACRE_FP64_ORACLE_MAX_ULP == 32
    assert comparison.passed
    assert oracle.output_size == (height * stride, width * stride)
    assert oracle.field.shape == occupancy.shape
    assert oracle.common_gate.shape == (
        batch,
        stride * stride,
        height,
        width,
    )
    assert bool(torch.any(fast.common_gate != 1.0))
    assert bool(torch.any(oracle.common_gate != 1.0))

    for batch_index in range(batch):
        singleton = model.forward_reference_fields_fp64(
            feature[batch_index : batch_index + 1],
            occupancy[batch_index : batch_index + 1],
        )
        torch.testing.assert_close(
            oracle.field[batch_index : batch_index + 1],
            singleton.field,
            rtol=0.0,
            atol=1.0e-12,
        )


def test_fp64_oracle_designated_flip_parity_at_boundaries() -> None:
    config = CoverageStateGCRPACREConfig(
        feature_channels=3,
        feature_stride=3,
        width=4,
    )
    model = _nonzero_model_for_config(config, seed=240150)
    generator = torch.Generator(device="cpu").manual_seed(240151)
    feature = torch.randn(
        (2, 3, 1, 2),
        generator=generator,
        dtype=torch.float32,
    )
    occupancy = (
        torch.rand(
            (2, 1, 3, 6),
            generator=generator,
            dtype=torch.float32,
        )
        > 0.5
    )
    first = model.forward_reference_fields_fp64(feature, occupancy)

    coordinates = (
        (0, 0, 0, 0),
        (0, 4, 0, 1),
        (1, 8, 0, 0),
    )
    for batch_index, phase_index, row, column in coordinates:
        phase_row, phase_column = divmod(phase_index, config.feature_stride)
        output_row = row * config.feature_stride + phase_row
        output_column = (
            column * config.feature_stride + phase_column
        )
        flipped = occupancy.clone()
        flipped[
            batch_index,
            0,
            output_row,
            output_column,
        ] = ~flipped[
            batch_index,
            0,
            output_row,
            output_column,
        ]
        second = model.forward_reference_fields_fp64(feature, flipped)
        coordinate = (batch_index, phase_index, row, column)

        assert abs(
            float(first.residual_odd_interaction[coordinate])
        ) > 1.0e-12
        torch.testing.assert_close(
            first.residual_odd_interaction[coordinate],
            -second.residual_odd_interaction[coordinate],
            rtol=0.0,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            first.common_even_energy[coordinate],
            second.common_even_energy[coordinate],
            rtol=0.0,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            first.common_gate[coordinate],
            second.common_gate[coordinate],
            rtol=0.0,
            atol=1.0e-12,
        )
        torch.testing.assert_close(
            first.gated_interaction[coordinate],
            -second.gated_interaction[coordinate],
            rtol=0.0,
            atol=1.0e-12,
        )


def test_fp32_ulp_distance_has_exact_one_step_witnesses() -> None:
    positive = torch.tensor(1.0, dtype=torch.float32)
    negative = torch.tensor(-1.0, dtype=torch.float32)
    positive_next = torch.nextafter(
        positive,
        torch.tensor(float("inf"), dtype=torch.float32),
    )
    negative_next = torch.nextafter(
        negative,
        torch.tensor(float("-inf"), dtype=torch.float32),
    )
    actual = torch.tensor(
        [1.0, -1.0, -0.0],
        dtype=torch.float32,
    )
    reference = torch.stack(
        (
            positive_next.to(dtype=torch.float64),
            negative_next.to(dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
        )
    )

    distance = gcr_pacre_fp32_ulp_distance(actual, reference)

    assert torch.equal(
        distance,
        torch.tensor([1, 1, 0], dtype=torch.int64),
    )


def test_gate_statistics_schema_covers_interior_and_exact_moments() -> None:
    model = _nonzero_model(seed=240160)
    feature, occupancy = _inputs(seed=240161)
    fields = model.forward_fields(feature, occupancy)
    audit = summarize_gcr_pacre_gate_saturation(fields)
    element_count = fields.common_gate.numel()

    assert audit.schema == GCR_PACRE_GATE_STATISTICS_SCHEMA
    assert audit.element_count == element_count
    assert audit.zero_count == 0
    assert audit.two_count == 0
    assert audit.saturated_count == 0
    assert audit.interior_count == element_count
    assert audit.zero_fraction == 0.0
    assert audit.two_fraction == 0.0
    assert audit.saturated_fraction == 0.0
    assert audit.interior_fraction == 1.0
    assert audit.minimum == float(fields.common_gate.detach().amin())
    assert audit.maximum == float(fields.common_gate.detach().amax())
    assert audit.mean == float(fields.common_gate.detach().mean())


def test_target_like_and_background_like_gate_selectivity_fixture() -> None:
    model = _nonzero_model(seed=240003)
    feature, occupancy = _inputs(seed=241003)
    fields = model.forward_fields(feature, occupancy)
    target_like = (0, 0, 0, 0)
    background_like = (0, 0, 0, 1)
    legacy_native = (
        model.config.field_amplitude
        + fields.residual_odd_interaction
    )

    assert fields.residual_odd_interaction[target_like] < 0.0
    assert fields.common_even_energy[target_like] > 0.0
    assert fields.common_gate[target_like] > 1.0
    assert (
        fields.native_phase_field[target_like]
        < legacy_native[target_like]
    )

    assert fields.residual_odd_interaction[background_like] < 0.0
    assert fields.common_even_energy[background_like] < 0.0
    assert fields.common_gate[background_like] < 1.0
    assert (
        fields.native_phase_field[background_like]
        > legacy_native[background_like]
    )


def test_v24_wheel_contains_package_and_only_frozen_dependencies(
    tmp_path: Path,
) -> None:
    source = tmp_path / "cure_lite_v24"
    shutil.copytree(
        REPO_ROOT / "cure_lite_v24",
        source,
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.egg-info",
            "build",
        ),
    )
    wheel_directory = tmp_path / "wheel"
    wheel_directory.mkdir()
    subprocess.run(
        (
            "/usr/bin/python",
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_directory),
            str(source),
        ),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel_paths = tuple(wheel_directory.glob("*.whl"))
    assert len(wheel_paths) == 1

    with zipfile.ZipFile(wheel_paths[0]) as archive:
        names = set(archive.namelist())
        source_modules = {
            f"cure_lite_v24/{path.name}"
            for path in source.glob("*.py")
        }
        assert source_modules <= names
        assert not any(
            name.endswith((".pyc", "pyproject.toml"))
            or "/__pycache__/" in name
            or "/tests/" in name
            or name.startswith("tests")
            for name in names
        )
        metadata_name = next(
            name
            for name in names
            if name.endswith(".dist-info/METADATA")
        )
        metadata = archive.read(metadata_name).decode("utf-8")

    requires_dist = {
        line.removeprefix("Requires-Dist: ").strip()
        for line in metadata.splitlines()
        if line.startswith("Requires-Dist: ")
    }
    assert requires_dist == {
        "cure-lite ==0.2.0",
        "torch >=2.1",
    }
