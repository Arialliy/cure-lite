"""Read-only zero-level diagnostics for a trained CSLF/PP-CSLF checkpoint.

The evaluator consumes only a frozen :class:`CoverageStateScalarCache` on
``D_R`` and a checkpoint-loaded module in evaluation mode.  It never trains,
searches a threshold, reads ``D_V``/``D_T``, or changes the cache/model state.

The binary contract is fixed:

``completion = (field < 0) & ~occupancy``
``final = occupancy | completion``

Before any real trained checkpoint is read, the clean-pair compact-support
policy is frozen as exact zero-level support equality: within the valid,
writable domain, newly added completion must equal the added target pixel for
pixel, with no spill and identical 8-connected support.  No value is selected
from observed checkpoint behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import isfinite

import torch
from torch import Tensor, nn

from ..cache.schema import stable_fingerprint
from ..coverage_state_level_set import (
    CSLF_OUTPUT_POLICY,
    normalize_cslf_feature,
)
from ..coverage_state_observability import (
    actual_input_fingerprint,
    occupancy_to_phase_grid,
)
from ..coverage_state_phase_preserving import (
    CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
)
from ..coverage_state_precomputed_cache import (
    CoverageStateCachedNatural,
    CoverageStateCachedPair,
    CoverageStateScalarCache,
)
from ..frozen_base import module_state_fingerprint
from ..instances import instances_from_binary_mask
from ..paired_types import tensor_content_fingerprint


COVERAGE_STATE_ZERO_LEVEL_EVALUATION_SCHEMA = (
    "cure-lite-cslf-zero-level-evaluation-v2"
)
COVERAGE_STATE_ZERO_LEVEL_CONFIG_SCHEMA = (
    "cure-lite-cslf-zero-level-evaluation-config-v2"
)
COVERAGE_STATE_PHASE_ZERO_LEVEL_EVALUATION_SCHEMA = (
    "cure-lite-pp-cslf-zero-level-evaluation-v3"
)
COVERAGE_STATE_PHASE_ZERO_LEVEL_CONFIG_SCHEMA = (
    "cure-lite-pp-cslf-zero-level-evaluation-config-v3"
)
COVERAGE_STATE_PHASE_DIAGNOSTIC_NULL_POLICY = (
    "phase_visible_distinct_input_completion_semantic_null_v1"
)
COVERAGE_STATE_RESIDUAL_THRESHOLD = 0.0
COVERAGE_STATE_COMPONENT_CONNECTIVITY = 8
COVERAGE_STATE_FACTUAL_TARGET_NEGATIVE_FRACTION = 0.95
COVERAGE_STATE_CONNECTED_SUPPORT_POLICY = (
    "focus_component_hit_recall_connectivity8_v1"
)
COVERAGE_STATE_COMPACT_SUPPORT_POLICY = (
    "clean_added_target_zero_level_exact_no_spill_v1"
)
COVERAGE_STATE_BINARY_OUTPUT_RULE = (
    "completion=(field<0)&~occupancy;"
    "final=occupancy|completion"
)
COVERAGE_STATE_SCALAR_INPUT_REPRESENTATION = "scalar_max"
COVERAGE_STATE_PHASE_INPUT_REPRESENTATION = "phase_preserving"
COVERAGE_STATE_INPUT_REPRESENTATIONS = (
    COVERAGE_STATE_SCALAR_INPUT_REPRESENTATION,
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
)


def _hex_fraction(numerator: int, denominator: int) -> str | None:
    if (
        isinstance(numerator, bool)
        or isinstance(denominator, bool)
        or not isinstance(numerator, int)
        or not isinstance(denominator, int)
        or numerator < 0
        or denominator < 0
    ):
        raise ValueError("fraction counts must be nonnegative integers")
    if denominator == 0:
        return None
    value = numerator / denominator
    if not isfinite(value):
        raise FloatingPointError("diagnostic fraction is non-finite")
    return float(value).hex()


def _component_count(mask: Tensor) -> int:
    return len(
        instances_from_binary_mask(
            mask,
            connectivity=COVERAGE_STATE_COMPONENT_CONNECTIVITY,
        ).instances
    )


def _component_hit_counts(
    target: Tensor,
    completion: Tensor,
) -> tuple[int, int]:
    target_instances = instances_from_binary_mask(
        target,
        connectivity=COVERAGE_STATE_COMPONENT_CONNECTIVITY,
    )
    completion_cpu = completion.detach().to("cpu", dtype=torch.bool)
    hit = sum(
        bool(torch.any(value.mask & completion_cpu[0, 0]))
        for value in target_instances.instances
    )
    return hit, len(target_instances.instances)


def _new_component_count(
    before: Tensor,
    after: Tensor,
) -> int:
    before_cpu = before.detach().to("cpu", dtype=torch.bool)[0, 0]
    after_instances = instances_from_binary_mask(
        after,
        connectivity=COVERAGE_STATE_COMPONENT_CONNECTIVITY,
    )
    return sum(
        not bool(torch.any(value.mask & before_cpu))
        for value in after_instances.instances
    )


@dataclass(frozen=True)
class CoverageStateZeroLevelEvaluationConfig:
    """Frozen, selection-free policy for one read-only ``D_R`` evaluation."""

    split: str = "D_R"
    residual_threshold: float = COVERAGE_STATE_RESIDUAL_THRESHOLD
    threshold_search_performed: bool = False
    component_connectivity: int = COVERAGE_STATE_COMPONENT_CONNECTIVITY
    factual_target_negative_fraction: float = (
        COVERAGE_STATE_FACTUAL_TARGET_NEGATIVE_FRACTION
    )
    connected_support_policy: str = (
        COVERAGE_STATE_CONNECTED_SUPPORT_POLICY
    )
    compact_support_policy: str = COVERAGE_STATE_COMPACT_SUPPORT_POLICY
    binary_output_rule: str = COVERAGE_STATE_BINARY_OUTPUT_RULE
    output_policy: str = CSLF_OUTPUT_POLICY
    training_performed: bool = False
    d_v_accessed: bool = False
    d_t_accessed: bool = False
    input_representation: str = (
        COVERAGE_STATE_SCALAR_INPUT_REPRESENTATION
    )

    def __post_init__(self) -> None:
        if self.input_representation not in (
            COVERAGE_STATE_INPUT_REPRESENTATIONS
        ):
            raise ValueError(
                "zero-level evaluation fixes input_representation to a "
                "registered coverage-state representation"
            )
        frozen = {
            "split": "D_R",
            "residual_threshold": COVERAGE_STATE_RESIDUAL_THRESHOLD,
            "threshold_search_performed": False,
            "component_connectivity": (
                COVERAGE_STATE_COMPONENT_CONNECTIVITY
            ),
            "factual_target_negative_fraction": (
                COVERAGE_STATE_FACTUAL_TARGET_NEGATIVE_FRACTION
            ),
            "connected_support_policy": (
                COVERAGE_STATE_CONNECTED_SUPPORT_POLICY
            ),
            "compact_support_policy": (
                COVERAGE_STATE_COMPACT_SUPPORT_POLICY
            ),
            "binary_output_rule": COVERAGE_STATE_BINARY_OUTPUT_RULE,
            "output_policy": CSLF_OUTPUT_POLICY,
            "training_performed": False,
            "d_v_accessed": False,
            "d_t_accessed": False,
        }
        for name, expected in frozen.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"zero-level evaluation fixes {name}"
                )

    def canonical_payload(self) -> dict[str, object]:
        phase_visible = (
            self.input_representation
            == COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
        )
        payload: dict[str, object] = {
            "schema_version": (
                COVERAGE_STATE_PHASE_ZERO_LEVEL_CONFIG_SCHEMA
                if phase_visible
                else COVERAGE_STATE_ZERO_LEVEL_CONFIG_SCHEMA
            ),
            "split": self.split,
            "residual_threshold_hex": self.residual_threshold.hex(),
            "threshold_search_performed": (
                self.threshold_search_performed
            ),
            "component_connectivity": self.component_connectivity,
            "factual_target_negative_fraction_hex": (
                self.factual_target_negative_fraction.hex()
            ),
            "connected_support_policy": self.connected_support_policy,
            "compact_support_policy": self.compact_support_policy,
            "binary_output_rule": self.binary_output_rule,
            "output_policy": self.output_policy,
            "training_performed": self.training_performed,
            "D_V_accessed": self.d_v_accessed,
            "D_T_accessed": self.d_t_accessed,
        }
        # Keep the established scalar-max receipt byte-for-byte compatible.
        # The field is emitted only for the new phase-preserving protocol.
        if phase_visible:
            payload["input_representation"] = self.input_representation
            payload["diagnostic_null_policy"] = (
                COVERAGE_STATE_PHASE_DIAGNOSTIC_NULL_POLICY
            )
        return payload

    @cached_property
    def config_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


@dataclass(frozen=True)
class CoverageStateDiagnosticStateLedger:
    """One logical diagnostic state and its memoized checkpoint output."""

    state_id: str
    role: str
    endpoint: str
    actual_input_fingerprint: str
    model_forward_index: int
    reused_actual_input: bool
    independent_exact_replay: bool
    field_fingerprint: str
    completion_fingerprint: str
    final_fingerprint: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "state_id": self.state_id,
            "role": self.role,
            "endpoint": self.endpoint,
            "actual_input_fingerprint": (
                self.actual_input_fingerprint
            ),
            "model_forward_index": self.model_forward_index,
            "reused_actual_input": self.reused_actual_input,
            "independent_exact_replay": (
                self.independent_exact_replay
            ),
            "field_fingerprint": self.field_fingerprint,
            "completion_fingerprint": self.completion_fingerprint,
            "final_fingerprint": self.final_fingerprint,
        }


@dataclass(frozen=True)
class CoverageStateNaturalZeroLevelDiagnostic:
    """Fixed zero-level metrics for one natural state."""

    record_id: str
    sample_id: str
    state_kind: str
    field_valid_pixels: int
    invalid_completion_pixels: int
    negative_pixels: int
    negative_components: int
    focus_target_pixels: int
    focus_target_negative_pixels: int
    target_negative_fraction_hex: str | None
    target_recovered: bool | None
    connected_support_components: int
    connected_support_components_hit: int
    connected_support_recall_hex: str | None
    gate_passed: bool

    def canonical_payload(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "sample_id": self.sample_id,
            "state_kind": self.state_kind,
            "field_valid_pixels": self.field_valid_pixels,
            "invalid_completion_pixels": (
                self.invalid_completion_pixels
            ),
            "negative_pixels": self.negative_pixels,
            "negative_components": self.negative_components,
            "focus_target_pixels": self.focus_target_pixels,
            "focus_target_negative_pixels": (
                self.focus_target_negative_pixels
            ),
            "target_negative_fraction_hex": (
                self.target_negative_fraction_hex
            ),
            "target_recovered": self.target_recovered,
            "connected_support_components": (
                self.connected_support_components
            ),
            "connected_support_components_hit": (
                self.connected_support_components_hit
            ),
            "connected_support_recall_hex": (
                self.connected_support_recall_hex
            ),
            "gate_passed": self.gate_passed,
        }


@dataclass(frozen=True)
class CoverageStatePairZeroLevelDiagnostic:
    """Fixed pair diagnostics; non-applicable quantities remain ``None``."""

    pair_id: str
    sample_id: str
    pair_kind: str
    optimizer_role: str
    scalar_hidden: bool
    actual_inputs_equal: bool
    invalid_completion_pixels_plus: int
    invalid_completion_pixels_minus: int
    field_exact_equal: bool
    completion_exact_equal: bool
    final_exact_equal: bool
    maximum_abs_field_difference_hex: str
    added_target_pixels: int
    added_target_components: int
    minus_added_target_negative_pixels: int
    minus_added_target_all_negative: bool | None
    response_sign_pixels: int
    response_sign_correct_pixels: int
    response_sign_all_correct: bool | None
    plus_writable_false_island_components: int | None
    new_negative_pixels: int
    new_negative_components: int
    removed_footprint_negative_pixels: int
    new_completion_pixels: int
    new_completion_outside_added_target_pixels: int | None
    new_completion_components: int
    compact_support_exact_equal: bool | None
    compact_support_component_match: bool | None
    compact_support_passed: bool | None
    defined_metrics_passed: bool
    gate_passed: bool

    def canonical_payload(
        self,
        *,
        include_input_relation: bool = False,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "pair_id": self.pair_id,
            "sample_id": self.sample_id,
            "pair_kind": self.pair_kind,
            "optimizer_role": self.optimizer_role,
            "scalar_hidden": self.scalar_hidden,
            "invalid_completion_pixels_plus": (
                self.invalid_completion_pixels_plus
            ),
            "invalid_completion_pixels_minus": (
                self.invalid_completion_pixels_minus
            ),
            "field_exact_equal": self.field_exact_equal,
            "completion_exact_equal": self.completion_exact_equal,
            "final_exact_equal": self.final_exact_equal,
            "maximum_abs_field_difference_hex": (
                self.maximum_abs_field_difference_hex
            ),
            "added_target_pixels": self.added_target_pixels,
            "added_target_components": self.added_target_components,
            "minus_added_target_negative_pixels": (
                self.minus_added_target_negative_pixels
            ),
            "minus_added_target_all_negative": (
                self.minus_added_target_all_negative
            ),
            "response_sign_pixels": self.response_sign_pixels,
            "response_sign_correct_pixels": (
                self.response_sign_correct_pixels
            ),
            "response_sign_all_correct": self.response_sign_all_correct,
            "plus_writable_false_island_components": (
                self.plus_writable_false_island_components
            ),
            "new_negative_pixels": self.new_negative_pixels,
            "new_negative_components": self.new_negative_components,
            "removed_footprint_negative_pixels": (
                self.removed_footprint_negative_pixels
            ),
            "new_completion_pixels": self.new_completion_pixels,
            "new_completion_outside_added_target_pixels": (
                self.new_completion_outside_added_target_pixels
            ),
            "new_completion_components": (
                self.new_completion_components
            ),
            "compact_support_exact_equal": (
                self.compact_support_exact_equal
            ),
            "compact_support_component_match": (
                self.compact_support_component_match
            ),
            "compact_support_passed": self.compact_support_passed,
            "defined_metrics_passed": self.defined_metrics_passed,
            "gate_passed": self.gate_passed,
        }
        if include_input_relation:
            # ``scalar_hidden`` describes the historical scalar projection,
            # not the actual PP-CSLF input relation.  The phase schema
            # therefore replaces it with the representation-aware relation.
            payload.pop("scalar_hidden")
            payload["actual_inputs_equal"] = self.actual_inputs_equal
            payload["input_relation"] = (
                "exact_same_actual_input"
                if self.actual_inputs_equal
                else "phase_visible_distinct_actual_inputs"
            )
        return payload


@dataclass(frozen=True)
class CoverageStateZeroLevelEvaluationResult:
    """Fingerprintable read-only diagnostic result for one checkpoint/cache."""

    config: CoverageStateZeroLevelEvaluationConfig
    dataset: str
    split: str
    cache_fingerprint: str
    checkpoint_fingerprint: str
    state_ledger: tuple[CoverageStateDiagnosticStateLedger, ...]
    natural_diagnostics: tuple[
        CoverageStateNaturalZeroLevelDiagnostic,
        ...,
    ]
    pair_diagnostics: tuple[
        CoverageStatePairZeroLevelDiagnostic,
        ...,
    ]
    diagnostic_state_references: int
    unique_actual_input_states: int
    model_forward_invocations: int
    exact_replay_forward_invocations: int
    reused_state_references: int
    backward_calls: int
    optimizer_steps: int
    factual_miss_gate_passed: bool
    factual_no_miss_gate_passed: bool
    clean_defined_metrics_passed: bool
    clean_compact_support_gate_passed: bool
    component_null_gate_passed: bool
    identity_null_gate_passed: bool
    scalar_hidden_diagnostic_gate_passed: bool
    bounded_gate_passed: bool
    fail_closed_reasons: tuple[str, ...]

    @property
    def diagnostic_null_gate_passed(self) -> bool:
        """Representation-neutral alias for the historical field name."""

        return self.scalar_hidden_diagnostic_gate_passed

    def canonical_payload(self) -> dict[str, object]:
        phase_visible = (
            self.config.input_representation
            == COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
        )
        gates: dict[str, object] = {
            "factual_miss": self.factual_miss_gate_passed,
            "factual_no_miss": self.factual_no_miss_gate_passed,
            "clean_defined_metrics": (
                self.clean_defined_metrics_passed
            ),
            "clean_compact_support": (
                self.clean_compact_support_gate_passed
            ),
            "component_null": self.component_null_gate_passed,
            "identity_null": self.identity_null_gate_passed,
            (
                "diagnostic_null"
                if phase_visible
                else "scalar_hidden_diagnostic"
            ): self.diagnostic_null_gate_passed,
            "bounded_gate_passed": self.bounded_gate_passed,
        }
        return {
            "schema_version": (
                COVERAGE_STATE_PHASE_ZERO_LEVEL_EVALUATION_SCHEMA
                if phase_visible
                else COVERAGE_STATE_ZERO_LEVEL_EVALUATION_SCHEMA
            ),
            "config": self.config.canonical_payload(),
            "config_fingerprint": self.config.config_fingerprint,
            "dataset": self.dataset,
            "split": self.split,
            "cache_fingerprint": self.cache_fingerprint,
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "state_ledger": [
                value.canonical_payload() for value in self.state_ledger
            ],
            "natural_diagnostics": [
                value.canonical_payload()
                for value in self.natural_diagnostics
            ],
            "pair_diagnostics": [
                value.canonical_payload(
                    include_input_relation=phase_visible,
                )
                for value in self.pair_diagnostics
            ],
            "compute": {
                "diagnostic_state_references": (
                    self.diagnostic_state_references
                ),
                "unique_actual_input_states": (
                    self.unique_actual_input_states
                ),
                "model_forward_invocations": (
                    self.model_forward_invocations
                ),
                "exact_replay_forward_invocations": (
                    self.exact_replay_forward_invocations
                ),
                "reused_state_references": (
                    self.reused_state_references
                ),
                "backward_calls": self.backward_calls,
                "optimizer_steps": self.optimizer_steps,
            },
            "gates": gates,
            "fail_closed_reasons": list(self.fail_closed_reasons),
            "execution": {
                "training_performed": False,
                "backward_performed": False,
                "optimizer_step_performed": False,
                "threshold_search_performed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            },
        }

    @cached_property
    def result_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


@dataclass(frozen=True)
class _DiagnosticState:
    state_id: str
    role: str
    endpoint: str
    actual_input_fingerprint: str
    feature: Tensor
    occupancy: Tensor
    independent_exact_replay: bool


@dataclass(frozen=True)
class _StateOutput:
    field: Tensor
    completion: Tensor
    final: Tensor
    model_forward_index: int


def _state_specs(
    cache: CoverageStateScalarCache,
    *,
    input_representation: str = (
        COVERAGE_STATE_SCALAR_INPUT_REPRESENTATION
    ),
) -> tuple[_DiagnosticState, ...]:
    if input_representation not in COVERAGE_STATE_INPUT_REPRESENTATIONS:
        raise ValueError("input_representation is not registered")

    def fingerprint(
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
        stride = cache.raw_catalog.feature_stride
        phase = occupancy_to_phase_grid(
            occupancy,
            stride=stride,
        )
        encoded = normalize_cslf_feature(feature)
        if tuple(phase.shape[-2:]) != tuple(encoded.shape[-2:]):
            raise ValueError(
                "phase occupancy grid differs from frozen feature grid"
            )
        return actual_input_fingerprint(
            encoded,
            phase,
            representation=COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
            stride=stride,
        )

    result: list[_DiagnosticState] = []
    for value in cache.natural_records:
        result.append(
            _DiagnosticState(
                state_id=f"natural:{value.record.record_id}",
                role=value.record.state_kind,
                endpoint="natural",
                actual_input_fingerprint=(
                    fingerprint(
                        value.record.feature,
                        value.record.occupancy,
                        scalar_fingerprint=(
                            value.actual_scalar_input_fingerprint
                        ),
                    )
                ),
                feature=value.record.feature,
                occupancy=value.record.occupancy,
                independent_exact_replay=False,
            )
        )
    for value in cache.pair_records:
        for endpoint, occupancy, scalar_fingerprint in (
            (
                "plus",
                value.record.occupancy_plus,
                value.actual_input_plus_fingerprint,
            ),
            (
                "minus",
                value.record.occupancy_minus,
                value.actual_input_minus_fingerprint,
            ),
        ):
            result.append(
                _DiagnosticState(
                    state_id=f"pair:{value.record.pair_id}:{endpoint}",
                    role=value.optimizer_role,
                    endpoint=endpoint,
                    actual_input_fingerprint=fingerprint(
                        value.record.feature,
                        occupancy,
                        scalar_fingerprint=scalar_fingerprint,
                    ),
                    feature=value.record.feature,
                    occupancy=occupancy,
                    independent_exact_replay=(
                        endpoint == "minus"
                        and value.optimizer_role
                        in {
                            "diagnostic_only",
                            "identity_diagnostic",
                        }
                    ),
                )
            )
    return tuple(result)


def _validate_model(
    model: nn.Module,
    cache: CoverageStateScalarCache,
    *,
    device: torch.device,
    input_representation: str,
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module")
    if model.training:
        raise ValueError("zero-level evaluation requires model.eval()")
    if getattr(model, "feature_stride", None) != (
        cache.raw_catalog.feature_stride
    ):
        raise ValueError("model feature_stride differs from scalar cache")
    feature_channels = int(
        cache.raw_catalog.natural_records[0].feature.shape[1]
    )
    if getattr(model, "feature_channels", None) != feature_channels:
        raise ValueError("model feature_channels differs from scalar cache")
    model_config = getattr(model, "config", None)
    explicitly_declared = tuple(
        value
        for value in (
            getattr(
                model_config,
                "occupancy_representation",
                None,
            ),
            getattr(model_config, "input_representation", None),
        )
        if value is not None
    )
    coverage_policy = getattr(model_config, "coverage_policy", None)
    policy_representation = (
        COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
        if coverage_policy == CSLF_PHASE_PRESERVING_COVERAGE_POLICY
        else None
    )
    if coverage_policy is not None and policy_representation is None:
        raise ValueError(
            "model declares an invalid coverage-state coverage policy"
        )
    declared_representations = (
        *explicitly_declared,
        *(
            ()
            if policy_representation is None
            else (policy_representation,)
        ),
    )
    if (
        len(set(declared_representations)) > 1
        or any(
            value not in COVERAGE_STATE_INPUT_REPRESENTATIONS
            for value in declared_representations
        )
    ):
        raise ValueError(
            "model declares an invalid coverage-state representation"
        )
    model_representation = (
        declared_representations[0]
        if declared_representations
        else COVERAGE_STATE_SCALAR_INPUT_REPRESENTATION
    )
    if model_representation != input_representation:
        raise ValueError(
            "model and zero-level input representations differ"
        )
    state_values = tuple(model.state_dict().values())
    if not state_values:
        raise ValueError("checkpoint model must have registered tensor state")
    for value in state_values:
        if not isinstance(value, Tensor):
            raise TypeError("model state contains a non-tensor value")
        if value.device != device:
            raise ValueError("model state and evaluation device differ")
        if value.is_floating_point():
            if value.dtype != torch.float32:
                raise TypeError("checkpoint evaluation fixes FP32 model state")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("checkpoint state is non-finite")


def _evaluate_state(
    model: nn.Module,
    state: _DiagnosticState,
    *,
    device: torch.device,
    model_forward_index: int,
) -> _StateOutput:
    feature = state.feature.to(device=device)
    occupancy = state.occupancy.to(device=device)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, enabled=False),
    ):
        field = model(feature, occupancy)
    if (
        not isinstance(field, Tensor)
        or field.dtype != torch.float32
        or tuple(field.shape) != tuple(occupancy.shape)
        or field.device != device
        or not bool(torch.isfinite(field).all())
    ):
        raise TypeError(
            "model must return finite FP32 field aligned with occupancy"
        )
    field_cpu = field.detach().to("cpu").contiguous()
    occupancy_cpu = state.occupancy
    completion = ((field_cpu < 0.0) & ~occupancy_cpu).contiguous()
    final = (occupancy_cpu | completion).contiguous()
    if not torch.equal(final, occupancy_cpu | completion):
        raise AssertionError("hard-union output contract changed")
    return _StateOutput(
        field=field_cpu,
        completion=completion,
        final=final,
        model_forward_index=model_forward_index,
    )


def _natural_diagnostic(
    value: CoverageStateCachedNatural,
    output: _StateOutput,
    *,
    threshold: float,
) -> CoverageStateNaturalZeroLevelDiagnostic:
    record = value.record
    valid = record.valid_mask
    completion = output.completion
    completion_valid = completion & valid
    invalid_pixels = int(torch.count_nonzero(completion & ~valid))
    negative_pixels = int(torch.count_nonzero(completion_valid))
    negative_components = _component_count(completion_valid)
    valid_pixels = int(torch.count_nonzero(valid))
    if record.state_kind == "factual_no_miss":
        gate = (
            negative_pixels == 0
            and negative_components == 0
            and invalid_pixels == 0
        )
        return CoverageStateNaturalZeroLevelDiagnostic(
            record_id=record.record_id,
            sample_id=record.sample_id,
            state_kind=record.state_kind,
            field_valid_pixels=valid_pixels,
            invalid_completion_pixels=invalid_pixels,
            negative_pixels=negative_pixels,
            negative_components=negative_components,
            focus_target_pixels=0,
            focus_target_negative_pixels=0,
            target_negative_fraction_hex=None,
            target_recovered=None,
            connected_support_components=0,
            connected_support_components_hit=0,
            connected_support_recall_hex=None,
            gate_passed=gate,
        )

    focus = value.targets.focus_support & valid
    target_pixels = int(torch.count_nonzero(focus))
    target_negative = int(torch.count_nonzero(completion & focus))
    fraction_hex = _hex_fraction(target_negative, target_pixels)
    hit, components = _component_hit_counts(focus, completion)
    recall_hex = _hex_fraction(hit, components)
    fraction = (
        0.0 if fraction_hex is None else float.fromhex(fraction_hex)
    )
    recovered = target_negative > 0
    gate = (
        target_pixels > 0
        and components > 0
        and recovered
        and fraction >= threshold
        and invalid_pixels == 0
    )
    return CoverageStateNaturalZeroLevelDiagnostic(
        record_id=record.record_id,
        sample_id=record.sample_id,
        state_kind=record.state_kind,
        field_valid_pixels=valid_pixels,
        invalid_completion_pixels=invalid_pixels,
        negative_pixels=negative_pixels,
        negative_components=negative_components,
        focus_target_pixels=target_pixels,
        focus_target_negative_pixels=target_negative,
        target_negative_fraction_hex=fraction_hex,
        target_recovered=recovered,
        connected_support_components=components,
        connected_support_components_hit=hit,
        connected_support_recall_hex=recall_hex,
        gate_passed=gate,
    )


def _pair_diagnostic(
    value: CoverageStateCachedPair,
    plus: _StateOutput,
    minus: _StateOutput,
    *,
    actual_inputs_equal: bool,
) -> CoverageStatePairZeroLevelDiagnostic:
    record = value.record
    valid = record.valid_mask
    completion_plus = plus.completion
    completion_minus = minus.completion
    plus_valid = completion_plus & valid
    minus_valid = completion_minus & valid
    invalid_plus = int(torch.count_nonzero(completion_plus & ~valid))
    invalid_minus = int(torch.count_nonzero(completion_minus & ~valid))
    field_equal = torch.equal(plus.field, minus.field)
    completion_equal = torch.equal(completion_plus, completion_minus)
    final_equal = torch.equal(plus.final, minus.final)
    maximum_difference = float(
        (plus.field - minus.field).abs().max().item()
    )
    if not isfinite(maximum_difference):
        raise FloatingPointError("pair field difference is non-finite")

    writable_minus = valid & ~record.occupancy_minus
    added_valid = record.target_minus & ~record.target_plus & valid
    added = added_valid & writable_minus
    added_pixels = int(torch.count_nonzero(added))
    added_components = _component_count(added)
    minus_added_negative = int(
        torch.count_nonzero(completion_minus & added)
    )
    new_completion = (
        completion_minus
        & ~completion_plus
        & writable_minus
    )
    new_pixels = int(torch.count_nonzero(new_completion))
    new_components = _new_component_count(plus_valid, minus_valid)
    removed_negative = int(
        torch.count_nonzero(
            completion_minus & record.removed_component & valid
        )
    )
    new_completion_components = _component_count(new_completion)

    minus_all_negative: bool | None = None
    response_pixels = 0
    response_correct = 0
    response_all: bool | None = None
    plus_false_islands: int | None = None
    outside_added: int | None = None
    compact_exact: bool | None = None
    compact_component_match: bool | None = None
    compact_passed: bool | None = None

    if record.pair_kind == "clean_positive":
        predicted_response = minus.field - plus.field
        target_response = (
            value.joint_targets.target_field_minus
            - value.joint_targets.target_field_plus
        )
        response_support = target_response.ne(0.0) & valid
        response_pixels = int(torch.count_nonzero(response_support))
        if response_pixels:
            correct = (
                predicted_response[response_support]
                * target_response[response_support]
            ) > 0.0
            response_correct = int(torch.count_nonzero(correct))
            response_all = response_correct == response_pixels
        minus_all_negative = (
            added_pixels > 0
            and minus_added_negative == added_pixels
        )
        plus_false_islands = _component_count(
            plus_valid & ~record.target_plus,
        )
        outside_added = int(
            torch.count_nonzero(new_completion & ~added)
        )
        compact_exact = (
            torch.equal(added_valid, added)
            and torch.equal(new_completion, added)
        )
        compact_component_match = (
            new_completion_components == added_components
        )
        compact_passed = (
            added_pixels > 0
            and minus_all_negative
            and plus_false_islands == 0
            and outside_added == 0
            and new_pixels == added_pixels
            and compact_component_match
            and compact_exact
            and invalid_plus == 0
            and invalid_minus == 0
        )
        defined = (
            compact_passed
            and response_pixels > 0
            and response_all is True
        )
        gate = defined
    elif record.pair_kind == "component_null":
        if not torch.equal(record.target_plus, record.target_minus):
            raise ValueError(
                "component-null raw targets must be exactly equal"
            )
        defined = (
            new_components == 0
            and removed_negative == 0
            and invalid_plus == 0
            and invalid_minus == 0
        )
        if value.optimizer_role == "diagnostic_only":
            # A scalar-hidden pair is an exact replay: identical actual
            # inputs must produce an identical continuous field.  A
            # phase-visible pair deliberately has distinct actual inputs,
            # so continuous-field equality is neither expected nor a
            # semantic null requirement.  It must still leave the binary
            # completion state unchanged.
            defined = defined and completion_equal
            if actual_inputs_equal:
                defined = defined and field_equal
        gate = defined
    else:
        if not torch.equal(record.target_plus, record.target_minus):
            raise ValueError("identity-null raw targets must be exactly equal")
        defined = (
            field_equal
            and completion_equal
            and final_equal
            and invalid_plus == 0
            and invalid_minus == 0
        )
        gate = defined

    return CoverageStatePairZeroLevelDiagnostic(
        pair_id=record.pair_id,
        sample_id=record.sample_id,
        pair_kind=record.pair_kind,
        optimizer_role=value.optimizer_role,
        scalar_hidden=not value.scalar_visible,
        actual_inputs_equal=actual_inputs_equal,
        invalid_completion_pixels_plus=invalid_plus,
        invalid_completion_pixels_minus=invalid_minus,
        field_exact_equal=field_equal,
        completion_exact_equal=completion_equal,
        final_exact_equal=final_equal,
        maximum_abs_field_difference_hex=maximum_difference.hex(),
        added_target_pixels=added_pixels,
        added_target_components=added_components,
        minus_added_target_negative_pixels=minus_added_negative,
        minus_added_target_all_negative=minus_all_negative,
        response_sign_pixels=response_pixels,
        response_sign_correct_pixels=response_correct,
        response_sign_all_correct=response_all,
        plus_writable_false_island_components=plus_false_islands,
        new_negative_pixels=new_pixels,
        new_negative_components=new_components,
        removed_footprint_negative_pixels=removed_negative,
        new_completion_pixels=new_pixels,
        new_completion_outside_added_target_pixels=outside_added,
        new_completion_components=new_completion_components,
        compact_support_exact_equal=compact_exact,
        compact_support_component_match=compact_component_match,
        compact_support_passed=compact_passed,
        defined_metrics_passed=defined,
        gate_passed=gate,
    )


def evaluate_coverage_state_zero_level_checkpoint(
    model: nn.Module,
    cache: CoverageStateScalarCache,
    *,
    device: torch.device | str,
    config: CoverageStateZeroLevelEvaluationConfig | None = None,
) -> CoverageStateZeroLevelEvaluationResult:
    """Evaluate every cached ``D_R`` state without training or threshold search."""

    if not isinstance(cache, CoverageStateScalarCache):
        raise TypeError("cache must be CoverageStateScalarCache")
    resolved_config = (
        CoverageStateZeroLevelEvaluationConfig()
        if config is None
        else config
    )
    if not isinstance(
        resolved_config,
        CoverageStateZeroLevelEvaluationConfig,
    ):
        raise TypeError(
            "config must be CoverageStateZeroLevelEvaluationConfig"
        )
    if cache.raw_catalog.split != resolved_config.split:
        raise PermissionError("zero-level evaluation permits only D_R")
    cache.verify_unchanged()
    requested_device = torch.device(device)
    _validate_model(
        model,
        cache,
        device=requested_device,
        input_representation=resolved_config.input_representation,
    )
    checkpoint_fingerprint = module_state_fingerprint(model)

    state_specs = _state_specs(
        cache,
        input_representation=resolved_config.input_representation,
    )
    state_specs_by_id = {value.state_id: value for value in state_specs}
    if len(state_specs_by_id) != len(state_specs):
        raise RuntimeError("diagnostic state ids must be unique")
    outputs: dict[str, _StateOutput] = {}
    output_by_input: dict[str, _StateOutput] = {}
    state_ledger: list[CoverageStateDiagnosticStateLedger] = []
    for state in state_specs:
        output = (
            None
            if state.independent_exact_replay
            else output_by_input.get(state.actual_input_fingerprint)
        )
        reused = output is not None
        if output is None:
            output = _evaluate_state(
                model,
                state,
                device=requested_device,
                model_forward_index=sum(
                    not value.reused_actual_input
                    for value in state_ledger
                ),
            )
            output_by_input.setdefault(
                state.actual_input_fingerprint,
                output,
            )
        outputs[state.state_id] = output
        state_ledger.append(
            CoverageStateDiagnosticStateLedger(
                state_id=state.state_id,
                role=state.role,
                endpoint=state.endpoint,
                actual_input_fingerprint=(
                    state.actual_input_fingerprint
                ),
                model_forward_index=output.model_forward_index,
                reused_actual_input=reused,
                independent_exact_replay=(
                    state.independent_exact_replay
                ),
                field_fingerprint=tensor_content_fingerprint(
                    output.field
                ),
                completion_fingerprint=tensor_content_fingerprint(
                    output.completion
                ),
                final_fingerprint=tensor_content_fingerprint(
                    output.final
                ),
            )
        )

    if module_state_fingerprint(model) != checkpoint_fingerprint:
        raise RuntimeError("checkpoint state changed during evaluation")
    cache.verify_unchanged()

    natural_diagnostics = tuple(
        _natural_diagnostic(
            value,
            outputs[f"natural:{value.record.record_id}"],
            threshold=(
                resolved_config.factual_target_negative_fraction
            ),
        )
        for value in cache.natural_records
    )
    pair_diagnostics = tuple(
        _pair_diagnostic(
            value,
            outputs[f"pair:{value.record.pair_id}:plus"],
            outputs[f"pair:{value.record.pair_id}:minus"],
            actual_inputs_equal=(
                state_specs_by_id[
                    f"pair:{value.record.pair_id}:plus"
                ].actual_input_fingerprint
                == state_specs_by_id[
                    f"pair:{value.record.pair_id}:minus"
                ].actual_input_fingerprint
            ),
        )
        for value in cache.pair_records
    )

    factual_miss = tuple(
        value
        for value in natural_diagnostics
        if value.state_kind == "factual_miss"
    )
    factual_no_miss = tuple(
        value
        for value in natural_diagnostics
        if value.state_kind == "factual_no_miss"
    )
    clean = tuple(
        value
        for value in pair_diagnostics
        if value.pair_kind == "clean_positive"
    )
    component = tuple(
        value
        for value in pair_diagnostics
        if value.pair_kind == "component_null"
        and value.optimizer_role == "component_null"
    )
    identity = tuple(
        value
        for value in pair_diagnostics
        if value.pair_kind == "identity_null"
    )
    diagnostic_null = tuple(
        value
        for value in pair_diagnostics
        if value.optimizer_role == "diagnostic_only"
    )

    miss_gate = bool(factual_miss) and all(
        value.gate_passed for value in factual_miss
    )
    no_miss_gate = bool(factual_no_miss) and all(
        value.gate_passed for value in factual_no_miss
    )
    clean_defined = bool(clean) and all(
        value.defined_metrics_passed for value in clean
    )
    clean_compact_gate = bool(clean) and all(
        value.compact_support_passed is True for value in clean
    )
    component_gate = bool(component) and all(
        value.gate_passed for value in component
    )
    identity_gate = bool(identity) and all(
        value.gate_passed for value in identity
    )
    hidden_gate = bool(diagnostic_null) and all(
        value.gate_passed for value in diagnostic_null
    )
    bounded_gate = all(
        (
            miss_gate,
            no_miss_gate,
            clean_defined,
            clean_compact_gate,
            component_gate,
            identity_gate,
            hidden_gate,
        )
    )

    reasons: list[str] = []
    diagnostic_null_name = (
        "diagnostic_null"
        if (
            resolved_config.input_representation
            == COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
        )
        else "scalar_hidden_diagnostic"
    )
    for name, present, passed in (
        ("factual_miss", bool(factual_miss), miss_gate),
        ("factual_no_miss", bool(factual_no_miss), no_miss_gate),
        ("clean_positive", bool(clean), clean_defined),
        ("component_null", bool(component), component_gate),
        ("identity_null", bool(identity), identity_gate),
        (
            diagnostic_null_name,
            bool(diagnostic_null),
            hidden_gate,
        ),
    ):
        if not present:
            reasons.append(f"missing_required_role:{name}")
        elif not passed:
            reasons.append(f"defined_metric_gate_failed:{name}")
    if clean and not clean_compact_gate:
        reasons.append(
            "defined_metric_gate_failed:clean_compact_support"
        )
    fail_closed_reasons = tuple(sorted(set(reasons)))

    return CoverageStateZeroLevelEvaluationResult(
        config=resolved_config,
        dataset=cache.raw_catalog.dataset,
        split=cache.raw_catalog.split,
        cache_fingerprint=cache.cache_fingerprint,
        checkpoint_fingerprint=checkpoint_fingerprint,
        state_ledger=tuple(state_ledger),
        natural_diagnostics=natural_diagnostics,
        pair_diagnostics=pair_diagnostics,
        diagnostic_state_references=len(state_specs),
        unique_actual_input_states=len(output_by_input),
        model_forward_invocations=sum(
            not value.reused_actual_input for value in state_ledger
        ),
        exact_replay_forward_invocations=sum(
            value.independent_exact_replay for value in state_ledger
        ),
        reused_state_references=(
            sum(value.reused_actual_input for value in state_ledger)
        ),
        backward_calls=0,
        optimizer_steps=0,
        factual_miss_gate_passed=miss_gate,
        factual_no_miss_gate_passed=no_miss_gate,
        clean_defined_metrics_passed=clean_defined,
        clean_compact_support_gate_passed=clean_compact_gate,
        component_null_gate_passed=component_gate,
        identity_null_gate_passed=identity_gate,
        scalar_hidden_diagnostic_gate_passed=hidden_gate,
        bounded_gate_passed=bounded_gate,
        fail_closed_reasons=fail_closed_reasons,
    )


__all__ = [
    "COVERAGE_STATE_BINARY_OUTPUT_RULE",
    "COVERAGE_STATE_COMPACT_SUPPORT_POLICY",
    "COVERAGE_STATE_CONNECTED_SUPPORT_POLICY",
    "COVERAGE_STATE_FACTUAL_TARGET_NEGATIVE_FRACTION",
    "COVERAGE_STATE_INPUT_REPRESENTATIONS",
    "COVERAGE_STATE_PHASE_DIAGNOSTIC_NULL_POLICY",
    "COVERAGE_STATE_PHASE_INPUT_REPRESENTATION",
    "COVERAGE_STATE_PHASE_ZERO_LEVEL_CONFIG_SCHEMA",
    "COVERAGE_STATE_PHASE_ZERO_LEVEL_EVALUATION_SCHEMA",
    "COVERAGE_STATE_RESIDUAL_THRESHOLD",
    "COVERAGE_STATE_SCALAR_INPUT_REPRESENTATION",
    "COVERAGE_STATE_ZERO_LEVEL_CONFIG_SCHEMA",
    "COVERAGE_STATE_ZERO_LEVEL_EVALUATION_SCHEMA",
    "CoverageStateDiagnosticStateLedger",
    "CoverageStateNaturalZeroLevelDiagnostic",
    "CoverageStatePairZeroLevelDiagnostic",
    "CoverageStateZeroLevelEvaluationConfig",
    "CoverageStateZeroLevelEvaluationResult",
    "evaluate_coverage_state_zero_level_checkpoint",
]
