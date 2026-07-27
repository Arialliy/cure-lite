from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.coverage_state_level_set import normalize_cslf_feature
from cure_lite.coverage_state_observability import (
    actual_input_fingerprint,
    occupancy_to_phase_grid,
    occupancy_to_scalar_grid,
)
from cure_lite.coverage_state_phase_preserving import (
    CURELitePhasePreservingCoverageStateLevelSet,
    CoverageStatePhasePreservingConfig,
)
from cure_lite.experiment.coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_BINARY_OUTPUT_RULE,
    COVERAGE_STATE_COMPACT_SUPPORT_POLICY,
    COVERAGE_STATE_PHASE_DIAGNOSTIC_NULL_POLICY,
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    COVERAGE_STATE_PHASE_ZERO_LEVEL_CONFIG_SCHEMA,
    COVERAGE_STATE_PHASE_ZERO_LEVEL_EVALUATION_SCHEMA,
    COVERAGE_STATE_SCALAR_INPUT_REPRESENTATION,
    CoverageStateZeroLevelEvaluationConfig,
    evaluate_coverage_state_zero_level_checkpoint,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_training_scalar_cache,
)


class LookupFieldCheckpoint(nn.Module):
    """Deterministic toy checkpoint keyed by the exact scalar model input."""

    feature_channels = 2
    feature_stride = 2

    def __init__(
        self,
        fields: dict[str, Tensor],
        *,
        return_half: bool = False,
        drift_on_replay: bool = False,
        input_representation: str = (
            COVERAGE_STATE_SCALAR_INPUT_REPRESENTATION
        ),
    ) -> None:
        super().__init__()
        self.return_half = return_half
        self.drift_on_replay = drift_on_replay
        self.config = SimpleNamespace(
            occupancy_representation=input_representation,
        )
        self.input_representation = input_representation
        self._forward_counts: dict[str, int] = {}
        self._names: dict[str, str] = {}
        for index, (key, field) in enumerate(sorted(fields.items())):
            name = f"field_{index:03d}"
            self.register_buffer(name, field.detach().clone())
            self._names[key] = name
        self.register_buffer(
            "checkpoint_anchor",
            torch.tensor([1.0], dtype=torch.float32),
        )

    def forward(self, feature: Tensor, occupancy: Tensor) -> Tensor:
        encoded = normalize_cslf_feature(feature)
        if (
            self.input_representation
            == COVERAGE_STATE_SCALAR_INPUT_REPRESENTATION
        ):
            representation = occupancy_to_scalar_grid(
                occupancy,
                feature_size=tuple(
                    int(value) for value in feature.shape[-2:]
                ),
            )
        else:
            representation = occupancy_to_phase_grid(
                occupancy,
                stride=self.feature_stride,
            )
        key = actual_input_fingerprint(
            encoded,
            representation,
            representation=self.input_representation,
            stride=self.feature_stride,
        )
        field = getattr(self, self._names[key]).to(feature.device).clone()
        count = self._forward_counts.get(key, 0)
        self._forward_counts[key] = count + 1
        if self.drift_on_replay and count > 0:
            field[..., 0, 0] -= 0.125
        return field.half() if self.return_half else field


def _perfect_fields(
    input_representation: str = (
        COVERAGE_STATE_SCALAR_INPUT_REPRESENTATION
    ),
) -> tuple[object, dict[str, Tensor]]:
    cache = make_training_scalar_cache()
    fields: dict[str, Tensor] = {}

    def insert(key: str, field: Tensor) -> None:
        current = fields.get(key)
        if current is not None:
            assert torch.equal(current, field)
        fields[key] = field.detach().clone()

    def input_key(
        feature: Tensor,
        occupancy: Tensor,
        *,
        scalar_fingerprint: str,
    ) -> str:
        if (
            input_representation
            == COVERAGE_STATE_SCALAR_INPUT_REPRESENTATION
        ):
            return scalar_fingerprint
        encoded = normalize_cslf_feature(feature)
        phase = occupancy_to_phase_grid(
            occupancy,
            stride=cache.raw_catalog.feature_stride,
        )
        return actual_input_fingerprint(
            encoded,
            phase,
            representation=input_representation,
            stride=cache.raw_catalog.feature_stride,
        )

    for value in cache.natural_records:
        insert(
            input_key(
                value.record.feature,
                value.record.occupancy,
                scalar_fingerprint=(
                    value.actual_scalar_input_fingerprint
                ),
            ),
            value.targets.target_field,
        )
    for value in cache.pair_records:
        if value.optimizer_role in {
            "diagnostic_only",
            "identity_diagnostic",
        }:
            field_plus = torch.full_like(
                value.record.occupancy_plus,
                0.9,
                dtype=torch.float32,
            )
            field_minus = torch.full_like(
                value.record.occupancy_minus,
                0.9,
                dtype=torch.float32,
            )
            if (
                value.optimizer_role == "diagnostic_only"
                and input_representation
                == COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
            ):
                # The phase-visible endpoints are genuinely different
                # inputs.  Their continuous fields may differ while both
                # encode the same empty completion.
                field_minus[..., 0, 0] = 0.8
        else:
            field_plus = value.joint_targets.target_field_plus
            field_minus = value.joint_targets.target_field_minus
        insert(
            input_key(
                value.record.feature,
                value.record.occupancy_plus,
                scalar_fingerprint=(
                    value.actual_input_plus_fingerprint
                ),
            ),
            field_plus,
        )
        insert(
            input_key(
                value.record.feature,
                value.record.occupancy_minus,
                scalar_fingerprint=(
                    value.actual_input_minus_fingerprint
                ),
            ),
            field_minus,
        )
    return cache, fields


def _diagnostics_by_id(result):
    natural = {
        value.record_id: value for value in result.natural_diagnostics
    }
    pairs = {value.pair_id: value for value in result.pair_diagnostics}
    return natural, pairs


def test_perfect_checkpoint_closes_defined_metrics_and_fails_compact_closed() -> None:
    cache, fields = _perfect_fields()
    model = LookupFieldCheckpoint(fields)
    model.eval()
    result = evaluate_coverage_state_zero_level_checkpoint(
        model,
        cache,
        device="cpu",
    )

    assert result.cache_fingerprint == cache.cache_fingerprint
    assert len(result.cache_fingerprint) == 64
    assert len(result.checkpoint_fingerprint) == 64
    assert len(result.result_fingerprint) == 64
    assert result.result_fingerprint == stable_fingerprint(
        result.canonical_payload()
    )
    assert result.config.binary_output_rule == (
        COVERAGE_STATE_BINARY_OUTPUT_RULE
    )
    assert result.config.residual_threshold == 0.0
    assert not result.config.threshold_search_performed
    assert result.split == "D_R"
    assert result.backward_calls == 0
    assert result.optimizer_steps == 0

    expected_references = len(cache.natural_records) + 2 * len(
        cache.pair_records
    )
    expected_unique = len(
        {
            *(
                value.actual_scalar_input_fingerprint
                for value in cache.natural_records
            ),
            *(
                fingerprint
                for value in cache.pair_records
                for fingerprint in (
                    value.actual_input_plus_fingerprint,
                    value.actual_input_minus_fingerprint,
                )
            ),
        }
    )
    assert result.diagnostic_state_references == expected_references
    assert result.unique_actual_input_states == expected_unique
    expected_replays = sum(
        value.optimizer_role
        in {"diagnostic_only", "identity_diagnostic"}
        for value in cache.pair_records
    )
    assert result.exact_replay_forward_invocations == expected_replays
    assert result.model_forward_invocations == (
        expected_unique + expected_replays
    )
    assert result.reused_state_references == (
        expected_references - result.model_forward_invocations
    )
    assert len(result.state_ledger) == expected_references

    assert result.factual_miss_gate_passed
    assert result.factual_no_miss_gate_passed
    assert result.clean_defined_metrics_passed
    assert result.clean_compact_support_gate_passed
    assert result.component_null_gate_passed
    assert result.identity_null_gate_passed
    assert result.scalar_hidden_diagnostic_gate_passed
    assert result.bounded_gate_passed
    assert result.config.compact_support_policy == (
        COVERAGE_STATE_COMPACT_SUPPORT_POLICY
    )
    assert result.fail_closed_reasons == ()

    natural, pairs = _diagnostics_by_id(result)
    misses = tuple(
        value
        for value in natural.values()
        if value.state_kind == "factual_miss"
    )
    assert misses
    assert all(value.target_recovered for value in misses)
    assert all(
        float.fromhex(value.target_negative_fraction_hex or "0x0p+0")
        == 1.0
        for value in misses
    )
    assert all(
        float.fromhex(value.connected_support_recall_hex or "0x0p+0")
        == 1.0
        for value in misses
    )
    no_misses = tuple(
        value
        for value in natural.values()
        if value.state_kind == "factual_no_miss"
    )
    assert all(value.negative_pixels == 0 for value in no_misses)
    assert all(value.negative_components == 0 for value in no_misses)

    clean = pairs["pair-clean-training"]
    assert clean.minus_added_target_all_negative
    assert clean.response_sign_all_correct
    assert clean.plus_writable_false_island_components == 0
    assert clean.new_completion_outside_added_target_pixels == 0
    assert clean.new_completion_pixels == clean.added_target_pixels
    assert (
        clean.new_completion_components
        == clean.added_target_components
    )
    assert clean.compact_support_exact_equal
    assert clean.compact_support_component_match
    assert clean.compact_support_passed

    component = pairs["pair-component-training"]
    assert component.new_negative_components == 0
    assert component.removed_footprint_negative_pixels == 0
    identity = pairs["pair-identity-diagnostic"]
    assert identity.field_exact_equal
    assert identity.completion_exact_equal
    assert identity.final_exact_equal
    hidden = pairs["pair-component-diagnostic"]
    assert hidden.scalar_hidden
    assert hidden.field_exact_equal
    assert hidden.completion_exact_equal
    assert not hidden.final_exact_equal


def test_result_and_state_ledgers_are_exactly_reproducible() -> None:
    cache, fields = _perfect_fields()
    model = LookupFieldCheckpoint(fields)
    model.eval()
    first = evaluate_coverage_state_zero_level_checkpoint(
        model,
        cache,
        device="cpu",
    )
    replay = evaluate_coverage_state_zero_level_checkpoint(
        model,
        cache,
        device="cpu",
    )
    assert first.canonical_payload() == replay.canonical_payload()
    assert first.result_fingerprint == replay.result_fingerprint
    reused = tuple(
        value
        for value in first.state_ledger
        if value.reused_actual_input
    )
    replays = tuple(
        value
        for value in first.state_ledger
        if value.independent_exact_replay
    )
    assert len(replays) == 2
    assert all(value.endpoint == "minus" for value in replays)
    assert all(value.model_forward_index >= 0 for value in reused)
    assert (
        first.model_forward_invocations
        + first.reused_state_references
        == first.diagnostic_state_references
    )


def test_phase_representation_distinguishes_scalar_aliases_without_reuse() -> None:
    cache, scalar_fields = _perfect_fields()
    scalar_model = LookupFieldCheckpoint(scalar_fields)
    scalar_model.eval()
    scalar = evaluate_coverage_state_zero_level_checkpoint(
        scalar_model,
        cache,
        device="cpu",
    )

    cache, phase_fields = _perfect_fields(
        COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
    )
    phase_model = LookupFieldCheckpoint(
        phase_fields,
        input_representation=(
            COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
        ),
    )
    phase_model.eval()
    phase = evaluate_coverage_state_zero_level_checkpoint(
        phase_model,
        cache,
        device="cpu",
        config=CoverageStateZeroLevelEvaluationConfig(
            input_representation=(
                COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
            ),
        ),
    )

    hidden = next(
        value
        for value in cache.pair_records
        if value.optimizer_role == "diagnostic_only"
    )
    assert (
        hidden.actual_input_plus_fingerprint
        == hidden.actual_input_minus_fingerprint
    )
    phase_ledger = {
        value.state_id: value for value in phase.state_ledger
    }
    plus = phase_ledger[f"pair:{hidden.record.pair_id}:plus"]
    minus = phase_ledger[f"pair:{hidden.record.pair_id}:minus"]
    assert plus.actual_input_fingerprint != minus.actual_input_fingerprint
    assert plus.model_forward_index != minus.model_forward_index
    assert not plus.reused_actual_input
    assert not minus.reused_actual_input
    assert phase.unique_actual_input_states == (
        scalar.unique_actual_input_states + 1
    )
    assert phase.config.canonical_payload()[
        "input_representation"
    ] == COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
    assert phase.config.canonical_payload()["schema_version"] == (
        COVERAGE_STATE_PHASE_ZERO_LEVEL_CONFIG_SCHEMA
    )
    assert phase.config.canonical_payload()[
        "diagnostic_null_policy"
    ] == COVERAGE_STATE_PHASE_DIAGNOSTIC_NULL_POLICY
    _, scalar_pairs = _diagnostics_by_id(scalar)
    _, phase_pairs = _diagnostics_by_id(phase)
    scalar_hidden = scalar_pairs[hidden.record.pair_id]
    phase_visible = phase_pairs[hidden.record.pair_id]
    assert scalar_hidden.field_exact_equal
    assert scalar_hidden.completion_exact_equal
    assert scalar_hidden.gate_passed
    assert not phase_visible.field_exact_equal
    assert phase_visible.completion_exact_equal
    assert phase_visible.gate_passed
    assert phase.scalar_hidden_diagnostic_gate_passed
    assert phase.diagnostic_null_gate_passed
    phase_payload = phase.canonical_payload()
    assert phase_payload["schema_version"] == (
        COVERAGE_STATE_PHASE_ZERO_LEVEL_EVALUATION_SCHEMA
    )
    assert "diagnostic_null" in phase_payload["gates"]
    assert "scalar_hidden_diagnostic" not in phase_payload["gates"]
    diagnostic_payload = next(
        value
        for value in phase_payload["pair_diagnostics"]
        if value["pair_id"] == hidden.record.pair_id
    )
    assert "scalar_hidden" not in diagnostic_payload
    assert not diagnostic_payload["actual_inputs_equal"]
    assert diagnostic_payload["input_relation"] == (
        "phase_visible_distinct_actual_inputs"
    )


def test_phase_visible_component_null_rejects_changed_completion() -> None:
    cache, phase_fields = _perfect_fields(
        COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
    )
    hidden = next(
        value
        for value in cache.pair_records
        if value.optimizer_role == "diagnostic_only"
    )
    minus_phase = occupancy_to_phase_grid(
        hidden.record.occupancy_minus,
        stride=cache.raw_catalog.feature_stride,
    )
    minus_key = actual_input_fingerprint(
        normalize_cslf_feature(hidden.record.feature),
        minus_phase,
        representation=COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
        stride=cache.raw_catalog.feature_stride,
    )
    writable = (
        hidden.record.valid_mask
        & ~hidden.record.occupancy_minus
    )
    changed_index = tuple(
        int(value)
        for value in torch.nonzero(
            writable,
            as_tuple=False,
        )[0].tolist()
    )
    phase_fields[minus_key][changed_index] = -0.9

    model = LookupFieldCheckpoint(
        phase_fields,
        input_representation=(
            COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
        ),
    )
    model.eval()
    result = evaluate_coverage_state_zero_level_checkpoint(
        model,
        cache,
        device="cpu",
        config=CoverageStateZeroLevelEvaluationConfig(
            input_representation=(
                COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
            ),
        ),
    )
    _, pairs = _diagnostics_by_id(result)
    diagnostic = pairs[hidden.record.pair_id]
    assert not diagnostic.field_exact_equal
    assert not diagnostic.completion_exact_equal
    assert not diagnostic.gate_passed
    assert not result.scalar_hidden_diagnostic_gate_passed
    assert not result.diagnostic_null_gate_passed
    assert (
        "defined_metric_gate_failed:diagnostic_null"
        in result.fail_closed_reasons
    )


def test_representation_binding_fails_closed_and_legacy_payload_is_exact() -> None:
    legacy = CoverageStateZeroLevelEvaluationConfig()
    assert legacy.input_representation == (
        COVERAGE_STATE_SCALAR_INPUT_REPRESENTATION
    )
    assert "input_representation" not in legacy.canonical_payload()
    assert legacy.config_fingerprint == (
        "bdff480cf7e098f9ec543d9043ea6d2c4b6a9a65f7bf26c2ec72710dc2a34be2"
    )

    cache, scalar_fields = _perfect_fields()
    scalar_model = LookupFieldCheckpoint(scalar_fields)
    scalar_model.eval()
    with pytest.raises(
        ValueError,
        match="input representations differ",
    ):
        evaluate_coverage_state_zero_level_checkpoint(
            scalar_model,
            cache,
            device="cpu",
            config=CoverageStateZeroLevelEvaluationConfig(
                input_representation=(
                    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
                ),
            ),
        )

    _, phase_fields = _perfect_fields(
        COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
    )
    phase_model = LookupFieldCheckpoint(
        phase_fields,
        input_representation=(
            COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
        ),
    )
    phase_model.eval()
    with pytest.raises(
        ValueError,
        match="input representations differ",
    ):
        evaluate_coverage_state_zero_level_checkpoint(
            phase_model,
            cache,
            device="cpu",
        )
    with pytest.raises(ValueError, match="input_representation"):
        CoverageStateZeroLevelEvaluationConfig(
            input_representation="unregistered",
        )


def test_real_ppce_checkpoint_requires_and_uses_phase_evaluation() -> None:
    cache = make_training_scalar_cache()
    model = CURELitePhasePreservingCoverageStateLevelSet(
        CoverageStatePhasePreservingConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    model.eval()
    with pytest.raises(
        ValueError,
        match="input representations differ",
    ):
        evaluate_coverage_state_zero_level_checkpoint(
            model,
            cache,
            device="cpu",
        )

    result = evaluate_coverage_state_zero_level_checkpoint(
        model,
        cache,
        device="cpu",
        config=CoverageStateZeroLevelEvaluationConfig(
            input_representation=(
                COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
            ),
        ),
    )
    assert result.config.input_representation == (
        COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
    )
    hidden = next(
        value
        for value in cache.pair_records
        if value.optimizer_role == "diagnostic_only"
    )
    ledger = {value.state_id: value for value in result.state_ledger}
    assert ledger[
        f"pair:{hidden.record.pair_id}:plus"
    ].actual_input_fingerprint != ledger[
        f"pair:{hidden.record.pair_id}:minus"
    ].actual_input_fingerprint


def test_independent_diagnostic_replay_detects_output_drift() -> None:
    cache, fields = _perfect_fields()
    model = LookupFieldCheckpoint(fields, drift_on_replay=True)
    model.eval()
    result = evaluate_coverage_state_zero_level_checkpoint(
        model,
        cache,
        device="cpu",
    )
    _, pairs = _diagnostics_by_id(result)

    identity = pairs["pair-identity-diagnostic"]
    hidden = pairs["pair-component-diagnostic"]
    assert not identity.field_exact_equal
    assert not identity.gate_passed
    assert not hidden.field_exact_equal
    assert not hidden.gate_passed
    assert not result.identity_null_gate_passed
    assert not result.scalar_hidden_diagnostic_gate_passed
    assert {
        "defined_metric_gate_failed:identity_null",
        "defined_metric_gate_failed:scalar_hidden_diagnostic",
    } <= set(result.fail_closed_reasons)


def test_clean_exact_no_spill_policy_rejects_one_extra_writable_pixel() -> None:
    cache, fields = _perfect_fields()
    fields = {key: value.clone() for key, value in fields.items()}
    clean = cache.clean_positive_records[0]
    fields[clean.actual_input_minus_fingerprint][..., 0, 0] = -0.9

    model = LookupFieldCheckpoint(fields)
    model.eval()
    result = evaluate_coverage_state_zero_level_checkpoint(
        model,
        cache,
        device="cpu",
    )
    _, pairs = _diagnostics_by_id(result)
    diagnostic = pairs[clean.record.pair_id]

    assert diagnostic.minus_added_target_all_negative
    assert diagnostic.plus_writable_false_island_components == 0
    assert diagnostic.new_completion_outside_added_target_pixels == 1
    assert (
        diagnostic.new_completion_pixels
        == diagnostic.added_target_pixels + 1
    )
    assert not diagnostic.compact_support_exact_equal
    assert not diagnostic.compact_support_passed
    assert not diagnostic.gate_passed
    assert not result.clean_compact_support_gate_passed
    assert not result.bounded_gate_passed
    assert (
        "defined_metric_gate_failed:clean_compact_support"
        in result.fail_closed_reasons
    )


def test_corrupted_checkpoint_reports_each_defined_role_failure() -> None:
    cache, fields = _perfect_fields()
    fields = {key: value.clone() for key, value in fields.items()}

    miss = next(
        value
        for value in cache.natural_records
        if value.record.state_kind == "factual_miss"
    )
    fields[miss.actual_scalar_input_fingerprint].fill_(0.9)

    no_miss = next(
        value
        for value in cache.natural_records
        if value.record.state_kind == "factual_no_miss"
    )
    fields[no_miss.actual_scalar_input_fingerprint][..., 0, 0] = -0.9

    clean = cache.clean_positive_records[0]
    added = clean.record.target_minus & ~clean.record.target_plus
    fields[clean.actual_input_plus_fingerprint][..., 0, 1] = -0.9
    fields[clean.actual_input_minus_fingerprint][added] = 0.9

    component = cache.component_null_records[0]
    fields[component.actual_input_minus_fingerprint][
        component.record.removed_component
    ] = -0.9

    model = LookupFieldCheckpoint(fields)
    model.eval()
    result = evaluate_coverage_state_zero_level_checkpoint(
        model,
        cache,
        device="cpu",
    )
    natural, pairs = _diagnostics_by_id(result)

    assert not natural[miss.record.record_id].gate_passed
    assert not natural[miss.record.record_id].target_recovered
    assert not natural[no_miss.record.record_id].gate_passed
    assert natural[no_miss.record.record_id].negative_pixels == 1
    assert natural[no_miss.record.record_id].negative_components == 1

    clean_result = pairs[clean.record.pair_id]
    assert not clean_result.minus_added_target_all_negative
    assert not clean_result.response_sign_all_correct
    assert clean_result.plus_writable_false_island_components == 1
    assert not clean_result.defined_metrics_passed

    component_result = pairs[component.record.pair_id]
    assert component_result.new_negative_components == 1
    assert component_result.removed_footprint_negative_pixels == 1
    assert not component_result.gate_passed

    assert not result.factual_miss_gate_passed
    assert not result.factual_no_miss_gate_passed
    assert not result.clean_defined_metrics_passed
    assert not result.clean_compact_support_gate_passed
    assert not result.component_null_gate_passed
    assert not result.bounded_gate_passed
    assert {
        "defined_metric_gate_failed:factual_miss",
        "defined_metric_gate_failed:factual_no_miss",
        "defined_metric_gate_failed:clean_positive",
        "defined_metric_gate_failed:clean_compact_support",
        "defined_metric_gate_failed:component_null",
    } <= set(result.fail_closed_reasons)


def test_evaluation_rejects_training_mode_non_fp32_and_threshold_changes() -> None:
    cache, fields = _perfect_fields()
    model = LookupFieldCheckpoint(fields)
    with pytest.raises(ValueError, match=r"model\.eval"):
        evaluate_coverage_state_zero_level_checkpoint(
            model,
            cache,
            device="cpu",
        )

    half_model = LookupFieldCheckpoint(fields, return_half=True)
    half_model.eval()
    with pytest.raises(TypeError, match="finite FP32 field"):
        evaluate_coverage_state_zero_level_checkpoint(
            half_model,
            cache,
            device="cpu",
        )

    with pytest.raises(ValueError, match="residual_threshold"):
        replace(
            CoverageStateZeroLevelEvaluationConfig(),
            residual_threshold=0.1,
        )
    with pytest.raises(ValueError, match="threshold_search_performed"):
        replace(
            CoverageStateZeroLevelEvaluationConfig(),
            threshold_search_performed=True,
        )
    with pytest.raises(ValueError, match="compact_support_policy"):
        replace(
            CoverageStateZeroLevelEvaluationConfig(),
            compact_support_policy="checkpoint_selected_policy",
        )
