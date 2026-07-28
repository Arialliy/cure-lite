"""Persistent strict loader for the sole completed PAET Formal800 attempt.

The in-memory structural receipt is intentionally non-persistable.  This
module is its cross-process replacement: it accepts no caller-selected path
or JSON payload, reloads only the fixed Formal800 output directory, and binds
the terminal receipts to the strict final-model artifact before D_V can start.
It never constructs a D_R cache or reads D_V/D_T.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping

from ..cache.schema import file_sha256, stable_fingerprint
from ..coverage_state_paet_formal_source_closure import (
    verify_coverage_state_paet_formal_source_closure,
)
from ..coverage_state_sobolev import CSLF_PMOPE_POLICY
from .coverage_state_paet_formal_artifacts import (
    LoadedCoverageStatePAETFormalArtifact,
    load_coverage_state_paet_formal_artifact,
)
from .coverage_state_paet_formal_structural import (
    COVERAGE_STATE_PAET_FORMAL_STRUCTURAL_PERFORMANCE_STATUS,
    COVERAGE_STATE_PAET_FORMAL_STRUCTURAL_SCHEMA,
    COVERAGE_STATE_PAET_V21_BOUNDED_CACHE_FINGERPRINT,
    COVERAGE_STATE_PAET_V21_BOUNDED_POPULATION_FINGERPRINT,
    COVERAGE_STATE_PAET_V21_SOURCE_RECEIPT_FINGERPRINT,
)
from .coverage_state_paet_formal_training import (
    COVERAGE_STATE_PAET_FORMAL_AUTHORIZATION_SCHEMA,
    COVERAGE_STATE_PAET_FORMAL_EXPOSURE_GATE_FINGERPRINT,
    COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT,
    COVERAGE_STATE_PAET_FORMAL_INITIAL_MODEL_FINGERPRINT,
    COVERAGE_STATE_PAET_FORMAL_RUN_ID,
    COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT,
    _formal_model_config_payload,
    load_repository_coverage_state_paet_bounded_artifact_seal,
)
from .coverage_state_paet_decision import (
    COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
    decide_coverage_state_paet_bounded,
)
from .coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    CoverageStateDiagnosticStateLedger,
    CoverageStateNaturalZeroLevelDiagnostic,
    CoverageStatePairZeroLevelDiagnostic,
    CoverageStateZeroLevelEvaluationConfig,
    CoverageStateZeroLevelEvaluationResult,
)


PAET_FORMAL_ATTEMPT_SCHEMA = "cure-lite-paet-bfa-v21-formal800-attempt-v1"
PAET_FORMAL_STARTED_SCHEMA = "cure-lite-paet-bfa-v21-formal800-started-v1"
PAET_FORMAL_RUN_SCHEMA = "cure-lite-paet-bfa-v21-pmope-formal800-run-v1"
PAET_FORMAL_STRUCTURAL_DECISION_SCHEMA = (
    "cure-lite-paet-bfa-v21-pmope-formal800-structural-decision-v1"
)
PAET_FORMAL_STRUCTURAL_REPLAY_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal800-structural-replay-v1"
)
PAET_FORMAL_FINAL_ARTIFACT_RECEIPT_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal800-final-artifact-binding-v1"
)
PAET_FORMAL_TRAINING_RECEIPT_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal800-training-v1"
)
PAET_FORMAL_INPUTS_RECEIPT_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal800-inputs-v1"
)
PAET_FORMAL_AUTHORIZATION_RECEIPT_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal800-authorization-v1"
)
PAET_FORMAL_RESOURCE_RECEIPT_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal800-training-resource-measurement-v1"
)
PAET_FORMAL_EPOCH_PROGRESS_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal800-epoch-progress-v1"
)
_ROOT = Path(__file__).resolve().parents[2]
PAET_FORMAL_ATTEMPT_OUTPUT_PATH = _ROOT / (
    "runs/irstd1k_stage_a_seed42/"
    f"{COVERAGE_STATE_PAET_FORMAL_RUN_ID}"
)
_FINAL_MEMBERS = frozenset(
    {
        "model.safetensors",
        "formal_result.json",
        "training.json",
        "epoch_log.json",
        "receipt.json",
    }
)
_RECEIPT_MEMBERS = frozenset(
    {
        "config.json",
        "inputs.json",
        "authorization.json",
        "epoch_progress.jsonl",
        "training_resource.json",
        "formal_training.json",
        "final_artifact.json",
        "structural_replay.json",
        "decision.json",
    }
)
_CONTROL_MEMBERS = frozenset({"attempt.json", "STARTED.json", "COMPLETE.json"})
_HEX = frozenset("0123456789abcdef")
_SOURCE_CLOSURE_FIELD_NAMES = (
    "source_closure_manifest_sha256",
    "source_closure_archive_sha256",
    "source_closure_content_fingerprint",
    "source_closure_file_count",
)
_FROZEN_REAL_DR_INPUTS = {
    "manifest_path": {
        "repo_path": "protocols/IRSTD-1K/stage_a_seed42/manifest.json",
        "file_sha256": (
            "aa8e33529bd86f564ce6e163e0f9a7b1b3053e9c15054a59c6702a1523f35c02"
        ),
    },
    "state_index_path": {
        "repo_path": (
            "runs/irstd1k_stage_a_seed42/"
            "cure_lite_stage_a_fx_v3/d_r/state_cache/index.json"
        ),
        "file_sha256": (
            "075fc1ad217f365df85b1d29568ad215f06ce6e0b691ef78a5dd85f0affe6298"
        ),
    },
    "geometry_config_path": {
        "repo_path": (
            "protocols/IRSTD-1K/geometry_safe_p0_v2/config.json"
        ),
        "file_sha256": (
            "719e956b7c51b2b2c8294699fe26c2d36d5c8190b0d8bb5c1d5665a0f4344558"
        ),
    },
    "geometry_receipt_path": {
        "repo_path": (
            "runs/irstd1k_stage_a_seed42/"
            "cure_lite_geometry_safe_p0_v2_r1/receipts/"
            "geometry_catalog.json"
        ),
        "file_sha256": (
            "e2a9a986f8819433f3f5efd5c4f627504d10fb32d20f62769b2235b803209283"
        ),
    },
    "observability_config_path": {
        "repo_path": (
            "protocols/IRSTD-1K/"
            "coverage_state_observability_v1/config.json"
        ),
        "file_sha256": (
            "60d42e657f1daed3cb01c7ee93c8f3fe17417542931d853756ccbbeda1f95713"
        ),
    },
}
_FORMAL_EXPOSURE_BRANCHES = (
    "factual_miss",
    "factual_no_miss",
    "clean_positive",
    "component_null",
)
_FORMAL_EXPOSURE_STATISTICS = (
    "zero_exposure",
    "ess",
    "maximum_share",
)
_EXPECTED_EXPOSURE_GATE_CHECKS = {
    **{
        f"{branch}/{level}/{statistic}": True
        for branch in _FORMAL_EXPOSURE_BRANCHES
        for level in ("record", "source")
        for statistic in _FORMAL_EXPOSURE_STATISTICS
    },
    **{
        f"{role}/{statistic}": True
        for role in ("factual_focus_target", "clean_added_target")
        for statistic in _FORMAL_EXPOSURE_STATISTICS
    },
    "selection_exact_budget": True,
    "identity_null_optimizer_exposure": True,
    "diagnostic_only_optimizer_exposure": True,
}


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular file")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _verify_fingerprinted(
    payload: Mapping[str, object], *, field: str, name: str
) -> str:
    body = dict(payload)
    if field not in body:
        raise ValueError(f"{name} lacks {field}")
    digest = _digest(body.pop(field), name=f"{name}.{field}")
    if stable_fingerprint(body) != digest:
        raise ValueError(f"{name} fingerprint changed")
    return digest


def _require(value: object, expected: object, *, name: str) -> None:
    if value != expected:
        raise ValueError(f"{name} changed")


def _false(payload: Mapping[str, object], *names: str) -> None:
    if any(payload.get(name) is not False for name in names):
        raise ValueError("Formal800 receipt records forbidden D_V/D_T activity")


def _source_closure_fields(receipt: Mapping[str, object]) -> dict[str, object]:
    """Return the exact durable fields for the verified live source closure."""

    if receipt.get("sealed") is not True:
        raise ValueError("Formal800 source closure is not sealed")
    result: dict[str, object] = {}
    for source_name, field_name in (
        ("manifest_sha256", "source_closure_manifest_sha256"),
        ("archive_sha256", "source_closure_archive_sha256"),
        ("content_fingerprint", "source_closure_content_fingerprint"),
    ):
        result[field_name] = _digest(
            receipt.get(source_name),
            name=f"source closure {source_name}",
        )
    count = receipt.get("file_count")
    if type(count) is not int or count < 1:
        raise ValueError("Formal800 source closure file count changed")
    result["source_closure_file_count"] = count
    return result


def _require_source_closure_binding(
    payload: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    name: str,
) -> None:
    actual = {
        field: payload.get(field)
        for field in _SOURCE_CLOSURE_FIELD_NAMES
    }
    if actual != dict(expected):
        raise ValueError(f"{name} is not bound to the verified source closure")


def _dict(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _list(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _dataclass_from_exact_row(
    cls: type,
    row: object,
    *,
    name: str,
) -> object:
    payload = _dict(row, name=name)
    expected = {field.name for field in fields(cls)}
    if set(payload) != expected:
        raise ValueError(f"{name} fields changed")
    try:
        return cls(**payload)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is invalid") from error


def _bounded_population_role_ids(
    payload: object,
) -> dict[str, tuple[str, ...]]:
    population = _dict(payload, name="frozen bounded population")
    expected_fields = {
        "schema_version",
        "selection_policy",
        "seed",
        "split",
        "source_cache_fingerprint",
        "bounded_cache_fingerprint",
        "role_count",
        "factual_miss_record_ids",
        "factual_no_miss_record_ids",
        "clean_positive_pair_ids",
        "component_null_pair_ids",
        "identity_null_pair_ids",
        "scalar_hidden_diagnostic_pair_ids",
        "source_counts",
        "D_V_accessed",
        "D_T_accessed",
    }
    if (
        set(population) != expected_fields
        or population.get("schema_version")
        != "cure-lite-cslf-dr-bounded-population-v1"
        or population.get("selection_policy")
        != "deterministic_source_round_robin_then_record_rank_v1"
        or population.get("seed") != 42
        or population.get("split") != "D_R"
        or population.get("source_cache_fingerprint")
        != COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        or population.get("bounded_cache_fingerprint")
        != COVERAGE_STATE_PAET_V21_BOUNDED_CACHE_FINGERPRINT
        or population.get("role_count") != 16
        or population.get("D_V_accessed") is not False
        or population.get("D_T_accessed") is not False
    ):
        raise ValueError("frozen bounded population contract changed")

    result: dict[str, tuple[str, ...]] = {}
    role_counts = {
        "factual_miss_record_ids": 16,
        "factual_no_miss_record_ids": 16,
        "clean_positive_pair_ids": 16,
        "component_null_pair_ids": 16,
        "identity_null_pair_ids": 16,
        "scalar_hidden_diagnostic_pair_ids": 1,
    }
    for name, expected_count in role_counts.items():
        raw_ids = _list(
            population.get(name),
            name=f"frozen bounded population {name}",
        )
        ids = tuple(
            _digest(value, name=f"{name}[{index}]")
            for index, value in enumerate(raw_ids)
        )
        if len(ids) != expected_count or len(set(ids)) != len(ids):
            raise ValueError(
                f"frozen bounded population {name} changed"
            )
        result[name] = ids
    natural_ids = (
        *result["factual_miss_record_ids"],
        *result["factual_no_miss_record_ids"],
    )
    pair_ids = (
        *result["clean_positive_pair_ids"],
        *result["component_null_pair_ids"],
        *result["identity_null_pair_ids"],
        *result["scalar_hidden_diagnostic_pair_ids"],
    )
    if (
        len(set(natural_ids)) != len(natural_ids)
        or len(set(pair_ids)) != len(pair_ids)
    ):
        raise ValueError("frozen bounded population roles overlap")
    source_counts = _dict(
        population.get("source_counts"),
        name="frozen bounded population source counts",
    )
    if (
        not source_counts
        or any(
            not isinstance(sample_id, str)
            or not sample_id
            or type(count) is not int
            or count < 1
            for sample_id, count in source_counts.items()
        )
        or sum(source_counts.values()) != 81
    ):
        raise ValueError("frozen bounded population sources changed")
    return result


def _validate_diagnostic_row_scalars(
    natural: tuple[CoverageStateNaturalZeroLevelDiagnostic, ...],
    pairs: tuple[CoverageStatePairZeroLevelDiagnostic, ...],
) -> None:
    natural_integer_fields = (
        "field_valid_pixels",
        "invalid_completion_pixels",
        "negative_pixels",
        "negative_components",
        "focus_target_pixels",
        "focus_target_negative_pixels",
        "connected_support_components",
        "connected_support_components_hit",
    )
    pair_integer_fields = (
        "invalid_completion_pixels_plus",
        "invalid_completion_pixels_minus",
        "added_target_pixels",
        "added_target_components",
        "minus_added_target_negative_pixels",
        "response_sign_pixels",
        "response_sign_correct_pixels",
        "new_negative_pixels",
        "new_negative_components",
        "removed_footprint_negative_pixels",
        "new_completion_pixels",
        "new_completion_components",
    )
    pair_optional_integer_fields = (
        "plus_writable_false_island_components",
        "new_completion_outside_added_target_pixels",
    )
    pair_boolean_fields = (
        "scalar_hidden",
        "actual_inputs_equal",
        "field_exact_equal",
        "completion_exact_equal",
        "final_exact_equal",
        "defined_metrics_passed",
        "gate_passed",
    )
    pair_optional_boolean_fields = (
        "minus_added_target_all_negative",
        "response_sign_all_correct",
        "compact_support_exact_equal",
        "compact_support_component_match",
        "compact_support_passed",
    )
    for index, row in enumerate(natural):
        _digest(row.record_id, name=f"natural record_id {index}")
        if (
            not isinstance(row.sample_id, str)
            or not row.sample_id
            or row.state_kind
            not in {"factual_miss", "factual_no_miss"}
            or any(
                type(getattr(row, name)) is not int
                or getattr(row, name) < 0
                for name in natural_integer_fields
            )
            or (
                row.target_recovered is not None
                and type(row.target_recovered) is not bool
            )
            or type(row.gate_passed) is not bool
        ):
            raise ValueError(f"natural diagnostic {index} changed type")
        for name in (
            "target_negative_fraction_hex",
            "connected_support_recall_hex",
        ):
            value = getattr(row, name)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(
                        f"natural diagnostic {index} {name} changed type"
                    )
                try:
                    parsed = float.fromhex(value)
                except ValueError as error:
                    raise ValueError(
                        f"natural diagnostic {index} {name} is invalid"
                    ) from error
                if not parsed == parsed or parsed in {
                    float("inf"),
                    float("-inf"),
                }:
                    raise ValueError(
                        f"natural diagnostic {index} {name} is non-finite"
                    )
    for index, row in enumerate(pairs):
        _digest(row.pair_id, name=f"pair_id {index}")
        if (
            not isinstance(row.sample_id, str)
            or not row.sample_id
            or row.pair_kind
            not in {"clean_positive", "component_null", "identity_null"}
            or row.optimizer_role
            not in {
                "clean_positive",
                "component_null",
                "identity_diagnostic",
                "diagnostic_only",
            }
            or any(
                type(getattr(row, name)) is not int
                or getattr(row, name) < 0
                for name in pair_integer_fields
            )
            or any(
                getattr(row, name) is not None
                and (
                    type(getattr(row, name)) is not int
                    or getattr(row, name) < 0
                )
                for name in pair_optional_integer_fields
            )
            or any(
                type(getattr(row, name)) is not bool
                for name in pair_boolean_fields
            )
            or any(
                getattr(row, name) is not None
                and type(getattr(row, name)) is not bool
                for name in pair_optional_boolean_fields
            )
            or not isinstance(
                row.maximum_abs_field_difference_hex,
                str,
            )
        ):
            raise ValueError(f"pair diagnostic {index} changed type")
        try:
            maximum_difference = float.fromhex(
                row.maximum_abs_field_difference_hex
            )
        except ValueError as error:
            raise ValueError(
                f"pair diagnostic {index} field difference is invalid"
            ) from error
        if (
            maximum_difference < 0.0
            or maximum_difference != maximum_difference
            or maximum_difference == float("inf")
        ):
            raise ValueError(
                f"pair diagnostic {index} field difference is invalid"
            )


def _reconstruct_structural_diagnostic(
    payload: object,
    *,
    bounded_population: object,
) -> CoverageStateZeroLevelEvaluationResult:
    """Reconstruct and canonicalize the complete persisted D_R diagnostic."""

    value = _dict(payload, name="structural diagnostic")
    config = CoverageStateZeroLevelEvaluationConfig(
        input_representation=COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
    )
    if (
        value.get("config") != config.canonical_payload()
        or value.get("config_fingerprint") != config.config_fingerprint
    ):
        raise ValueError("structural diagnostic config changed")

    state_ledger = tuple(
        _dataclass_from_exact_row(
            CoverageStateDiagnosticStateLedger,
            row,
            name=f"structural state ledger row {index}",
        )
        for index, row in enumerate(
            _list(value.get("state_ledger"), name="structural state ledger")
        )
    )
    natural = tuple(
        _dataclass_from_exact_row(
            CoverageStateNaturalZeroLevelDiagnostic,
            row,
            name=f"structural natural diagnostic {index}",
        )
        for index, row in enumerate(
            _list(
                value.get("natural_diagnostics"),
                name="structural natural diagnostics",
            )
        )
    )
    pair_rows = _list(
        value.get("pair_diagnostics"),
        name="structural pair diagnostics",
    )
    pairs: list[CoverageStatePairZeroLevelDiagnostic] = []
    pair_fields = {
        field.name for field in fields(CoverageStatePairZeroLevelDiagnostic)
    }
    for index, raw_row in enumerate(pair_rows):
        row = dict(
            _dict(raw_row, name=f"structural pair diagnostic {index}")
        )
        relation = row.pop("input_relation", None)
        actual_inputs_equal = row.get("actual_inputs_equal")
        expected_relation = (
            "exact_same_actual_input"
            if actual_inputs_equal is True
            else "phase_visible_distinct_actual_inputs"
        )
        if (
            type(actual_inputs_equal) is not bool
            or relation != expected_relation
            or "scalar_hidden" in row
        ):
            raise ValueError(
                f"structural pair diagnostic {index} input relation changed"
            )
        row["scalar_hidden"] = False
        if set(row) != pair_fields:
            raise ValueError(
                f"structural pair diagnostic {index} fields changed"
            )
        try:
            pairs.append(CoverageStatePairZeroLevelDiagnostic(**row))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"structural pair diagnostic {index} is invalid"
            ) from error

    role_ids = _bounded_population_role_ids(bounded_population)
    natural_rows = tuple(natural)
    pair_values = tuple(pairs)
    _validate_diagnostic_row_scalars(natural_rows, pair_values)
    expected_natural_ids = tuple(
        sorted(
            (
                *role_ids["factual_miss_record_ids"],
                *role_ids["factual_no_miss_record_ids"],
            )
        )
    )
    expected_pair_ids = tuple(
        sorted(
            (
                *role_ids["clean_positive_pair_ids"],
                *role_ids["component_null_pair_ids"],
                *role_ids["identity_null_pair_ids"],
                *role_ids[
                    "scalar_hidden_diagnostic_pair_ids"
                ],
            )
        )
    )
    natural_kind_by_id = {
        **{
            record_id: "factual_miss"
            for record_id in role_ids["factual_miss_record_ids"]
        },
        **{
            record_id: "factual_no_miss"
            for record_id in role_ids["factual_no_miss_record_ids"]
        },
    }
    pair_contract_by_id = {
        **{
            pair_id: ("clean_positive", "clean_positive", False)
            for pair_id in role_ids["clean_positive_pair_ids"]
        },
        **{
            pair_id: ("component_null", "component_null", False)
            for pair_id in role_ids["component_null_pair_ids"]
        },
        **{
            pair_id: (
                "identity_null",
                "identity_diagnostic",
                True,
            )
            for pair_id in role_ids["identity_null_pair_ids"]
        },
        **{
            pair_id: (
                "component_null",
                "diagnostic_only",
                False,
            )
            for pair_id in role_ids[
                "scalar_hidden_diagnostic_pair_ids"
            ]
        },
    }
    if (
        tuple(row.record_id for row in natural_rows)
        != expected_natural_ids
        or any(
            row.state_kind != natural_kind_by_id[row.record_id]
            for row in natural_rows
        )
        or tuple(row.pair_id for row in pair_values)
        != expected_pair_ids
        or any(
            (
                row.pair_kind,
                row.optimizer_role,
                row.actual_inputs_equal,
            )
            != pair_contract_by_id[row.pair_id]
            for row in pair_values
        )
        or Counter(
            row.sample_id for row in (*natural_rows, *pair_values)
        )
        != Counter(
            _dict(
                _dict(
                    bounded_population,
                    name="frozen bounded population",
                ).get("source_counts"),
                name="frozen bounded population source counts",
            )
        )
    ):
        raise ValueError(
            "structural diagnostics differ from the frozen bounded population"
        )

    expected_states: list[tuple[str, str, str, bool]] = [
        (
            f"natural:{row.record_id}",
            row.state_kind,
            "natural",
            False,
        )
        for row in natural_rows
    ]
    for row in pair_values:
        expected_states.extend(
            (
                (
                    f"pair:{row.pair_id}:plus",
                    row.optimizer_role,
                    "plus",
                    False,
                ),
                (
                    f"pair:{row.pair_id}:minus",
                    row.optimizer_role,
                    "minus",
                    row.optimizer_role
                    in {"diagnostic_only", "identity_diagnostic"},
                ),
            )
        )
    if len(state_ledger) != len(expected_states):
        raise ValueError("structural state ledger population changed")

    first_state_by_input: dict[
        str,
        CoverageStateDiagnosticStateLedger,
    ] = {}
    next_forward_index = 0
    for index, (row, expected_state) in enumerate(
        zip(state_ledger, expected_states, strict=True)
    ):
        state_id, role, endpoint, independent = expected_state
        actual_input = _digest(
            row.actual_input_fingerprint,
            name=f"structural state {index} actual input",
        )
        for field_name in (
            "field_fingerprint",
            "completion_fingerprint",
            "final_fingerprint",
        ):
            _digest(
                getattr(row, field_name),
                name=f"structural state {index} {field_name}",
            )
        if (
            row.state_id != state_id
            or row.role != role
            or row.endpoint != endpoint
            or type(row.model_forward_index) is not int
            or row.model_forward_index < 0
            or type(row.reused_actual_input) is not bool
            or type(row.independent_exact_replay) is not bool
            or row.independent_exact_replay != independent
        ):
            raise ValueError(
                f"structural state ledger row {index} changed"
            )
        first = first_state_by_input.get(actual_input)
        expected_reuse = first is not None and not independent
        if expected_reuse:
            if (
                row.reused_actual_input is not True
                or row.model_forward_index
                != first.model_forward_index
                or row.field_fingerprint != first.field_fingerprint
                or row.completion_fingerprint
                != first.completion_fingerprint
                or row.final_fingerprint != first.final_fingerprint
            ):
                raise ValueError(
                    f"structural state reuse row {index} changed"
                )
        else:
            if (
                row.reused_actual_input is not False
                or row.model_forward_index != next_forward_index
            ):
                raise ValueError(
                    f"structural model forward row {index} changed"
                )
            next_forward_index += 1
            first_state_by_input.setdefault(actual_input, row)

    ledger_by_state = {row.state_id: row for row in state_ledger}
    if len(ledger_by_state) != len(state_ledger):
        raise ValueError("structural state ledger ids are not unique")
    for index, row in enumerate(pair_values):
        plus = ledger_by_state[f"pair:{row.pair_id}:plus"]
        minus = ledger_by_state[f"pair:{row.pair_id}:minus"]
        if (
            row.actual_inputs_equal
            != (
                plus.actual_input_fingerprint
                == minus.actual_input_fingerprint
            )
            or row.field_exact_equal
            != (plus.field_fingerprint == minus.field_fingerprint)
            or row.completion_exact_equal
            != (
                plus.completion_fingerprint
                == minus.completion_fingerprint
            )
            or row.final_exact_equal
            != (plus.final_fingerprint == minus.final_fingerprint)
        ):
            raise ValueError(
                f"structural pair diagnostic {index} and ledger differ"
            )

    compute = _dict(value.get("compute"), name="structural compute")
    expected_compute_fields = {
        "diagnostic_state_references",
        "unique_actual_input_states",
        "model_forward_invocations",
        "exact_replay_forward_invocations",
        "reused_state_references",
        "backward_calls",
        "optimizer_steps",
    }
    expected_compute = {
        "diagnostic_state_references": len(state_ledger),
        "unique_actual_input_states": len(first_state_by_input),
        "model_forward_invocations": next_forward_index,
        "exact_replay_forward_invocations": sum(
            row.independent_exact_replay for row in state_ledger
        ),
        "reused_state_references": sum(
            row.reused_actual_input for row in state_ledger
        ),
        "backward_calls": 0,
        "optimizer_steps": 0,
    }
    if (
        set(compute) != expected_compute_fields
        or compute != expected_compute
        or any(type(compute[name]) is not int or compute[name] < 0 for name in compute)
        or compute["backward_calls"] != 0
        or compute["optimizer_steps"] != 0
        or len(state_ledger) != 130
        or len(natural_rows) != 32
        or len(pair_values) != 49
        or len(
            {row.actual_input_fingerprint for row in state_ledger}
        )
        != expected_compute["unique_actual_input_states"]
        or expected_compute["unique_actual_input_states"] != 100
        or expected_compute["model_forward_invocations"] != 116
        or expected_compute["reused_state_references"] != 14
        or expected_compute[
            "exact_replay_forward_invocations"
        ]
        != 17
    ):
        raise ValueError("structural diagnostic compute ledger changed")
    gates = _dict(value.get("gates"), name="structural gates")
    expected_gate_fields = {
        "factual_miss",
        "factual_no_miss",
        "clean_defined_metrics",
        "clean_compact_support",
        "component_null",
        "identity_null",
        "diagnostic_null",
        "bounded_gate_passed",
    }
    factual_miss = tuple(
        row
        for row in natural_rows
        if row.state_kind == "factual_miss"
    )
    factual_no_miss = tuple(
        row
        for row in natural_rows
        if row.state_kind == "factual_no_miss"
    )
    clean = tuple(
        row
        for row in pair_values
        if row.pair_kind == "clean_positive"
    )
    component = tuple(
        row
        for row in pair_values
        if row.pair_kind == "component_null"
        and row.optimizer_role == "component_null"
    )
    identity = tuple(
        row
        for row in pair_values
        if row.pair_kind == "identity_null"
    )
    diagnostic_null = tuple(
        row
        for row in pair_values
        if row.optimizer_role == "diagnostic_only"
    )
    miss_gate = bool(factual_miss) and all(
        row.gate_passed for row in factual_miss
    )
    no_miss_gate = bool(factual_no_miss) and all(
        row.gate_passed for row in factual_no_miss
    )
    clean_defined = bool(clean) and all(
        row.defined_metrics_passed for row in clean
    )
    clean_compact = bool(clean) and all(
        row.compact_support_passed is True for row in clean
    )
    component_gate = bool(component) and all(
        row.gate_passed for row in component
    )
    identity_gate = bool(identity) and all(
        row.gate_passed for row in identity
    )
    diagnostic_gate = bool(diagnostic_null) and all(
        row.gate_passed for row in diagnostic_null
    )
    bounded_gate = all(
        (
            miss_gate,
            no_miss_gate,
            clean_defined,
            clean_compact,
            component_gate,
            identity_gate,
            diagnostic_gate,
        )
    )
    expected_gates = {
        "factual_miss": miss_gate,
        "factual_no_miss": no_miss_gate,
        "clean_defined_metrics": clean_defined,
        "clean_compact_support": clean_compact,
        "component_null": component_gate,
        "identity_null": identity_gate,
        "diagnostic_null": diagnostic_gate,
        "bounded_gate_passed": bounded_gate,
    }
    if (
        set(gates) != expected_gate_fields
        or any(type(gates[name]) is not bool for name in gates)
        or gates != expected_gates
    ):
        raise ValueError("structural diagnostic gates changed")
    expected_execution = {
        "training_performed": False,
        "backward_performed": False,
        "optimizer_step_performed": False,
        "threshold_search_performed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    if value.get("execution") != expected_execution:
        raise ValueError("structural diagnostic execution contract changed")
    reasons = _list(
        value.get("fail_closed_reasons"),
        name="structural fail-closed reasons",
    )
    expected_reasons: list[str] = []
    for name, present, passed in (
        ("factual_miss", bool(factual_miss), miss_gate),
        ("factual_no_miss", bool(factual_no_miss), no_miss_gate),
        ("clean_positive", bool(clean), clean_defined),
        ("component_null", bool(component), component_gate),
        ("identity_null", bool(identity), identity_gate),
        (
            "diagnostic_null",
            bool(diagnostic_null),
            diagnostic_gate,
        ),
    ):
        if not present:
            expected_reasons.append(f"missing_required_role:{name}")
        elif not passed:
            expected_reasons.append(
                f"defined_metric_gate_failed:{name}"
            )
    if clean and not clean_compact:
        expected_reasons.append(
            "defined_metric_gate_failed:clean_compact_support"
        )
    expected_fail_closed_reasons = sorted(set(expected_reasons))
    if (
        any(not isinstance(reason, str) for reason in reasons)
        or reasons != expected_fail_closed_reasons
    ):
        raise ValueError("structural fail-closed reasons are invalid")
    if (
        value.get("dataset") != "IRSTD-1K"
        or value.get("split") != "D_R"
        or value.get("cache_fingerprint")
        != COVERAGE_STATE_PAET_V21_BOUNDED_CACHE_FINGERPRINT
    ):
        raise ValueError("structural diagnostic population changed")

    result = CoverageStateZeroLevelEvaluationResult(
        config=config,
        dataset="IRSTD-1K",
        split="D_R",
        cache_fingerprint=str(value["cache_fingerprint"]),
        checkpoint_fingerprint=_digest(
            value.get("checkpoint_fingerprint"),
            name="structural checkpoint",
        ),
        state_ledger=state_ledger,
        natural_diagnostics=natural,
        pair_diagnostics=tuple(pairs),
        diagnostic_state_references=int(
            compute["diagnostic_state_references"]
        ),
        unique_actual_input_states=int(
            compute["unique_actual_input_states"]
        ),
        model_forward_invocations=int(
            compute["model_forward_invocations"]
        ),
        exact_replay_forward_invocations=int(
            compute["exact_replay_forward_invocations"]
        ),
        reused_state_references=int(compute["reused_state_references"]),
        backward_calls=0,
        optimizer_steps=0,
        factual_miss_gate_passed=miss_gate,
        factual_no_miss_gate_passed=no_miss_gate,
        clean_defined_metrics_passed=clean_defined,
        clean_compact_support_gate_passed=clean_compact,
        component_null_gate_passed=component_gate,
        identity_null_gate_passed=identity_gate,
        scalar_hidden_diagnostic_gate_passed=diagnostic_gate,
        bounded_gate_passed=bounded_gate,
        fail_closed_reasons=tuple(reasons),
    )
    if result.canonical_payload() != value:
        raise ValueError("structural diagnostic is not canonical")
    return result


def _validate_structural_result(
    payload: object,
    artifact: LoadedCoverageStatePAETFormalArtifact,
    *,
    bounded_population: object,
) -> tuple[str, bool]:
    """Recompute the frozen v21 decision from the persisted raw diagnostics."""

    result = _dict(payload, name="structural result")
    if (
        artifact.formal_result_payload.get(
            "structural_advancement_passed"
        )
        is not True
    ):
        raise ValueError(
            "formal result did not authorize structural advancement"
        )
    diagnostic = _reconstruct_structural_diagnostic(
        result.get("diagnostic"),
        bounded_population=bounded_population,
    )
    if (
        diagnostic.checkpoint_fingerprint
        != artifact.module_state_fingerprint
        or result.get("diagnostic_result_fingerprint")
        != diagnostic.result_fingerprint
    ):
        raise ValueError("structural diagnostic does not bind the final model")
    decision = decide_coverage_state_paet_bounded(
        diagnostic,
        run_id=COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
    )
    policy_payload = decision.canonical_payload()
    expected = {
        "schema_version": COVERAGE_STATE_PAET_FORMAL_STRUCTURAL_SCHEMA,
        "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
        "runtime_splits": ["D_R"],
        "formal_result_fingerprint": artifact.formal_result_fingerprint,
        "formal_authorization_fingerprint": (
            artifact.authorization_fingerprint
        ),
        "final_model_fingerprint": artifact.module_state_fingerprint,
        "bounded_cache_fingerprint": (
            COVERAGE_STATE_PAET_V21_BOUNDED_CACHE_FINGERPRINT
        ),
        "bounded_population_fingerprint": (
            COVERAGE_STATE_PAET_V21_BOUNDED_POPULATION_FINGERPRINT
        ),
        "source_receipt_fingerprint": (
            COVERAGE_STATE_PAET_V21_SOURCE_RECEIPT_FINGERPRINT
        ),
        "input_representation": (
            COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
        ),
        "field_threshold_hex": 0.0.hex(),
        "threshold_search_performed": False,
        "diagnostic": diagnostic.canonical_payload(),
        "diagnostic_result_fingerprint": diagnostic.result_fingerprint,
        "frozen_structural_policy": policy_payload,
        "frozen_structural_policy_fingerprint": (
            decision.decision_fingerprint
        ),
        "policy_origin_run_id": COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
        "policy_reused_without_change": True,
        "bounded400_structural_advancement_passed": True,
        "post_formal_structural_retention_passed": (
            decision.bounded_gate_passed
        ),
        "generic_population_gate_passed": (
            diagnostic.bounded_gate_passed
        ),
        "performance_status": (
            COVERAGE_STATE_PAET_FORMAL_STRUCTURAL_PERFORMANCE_STATUS
        ),
        "performance_gate_passed": None,
        "evaluation_invocations": 1,
        "training_performed_by_this_layer": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "performance_evaluation_performed": False,
        "performance_claim_supported": False,
    }
    if result != expected or not decision.bounded_gate_passed:
        raise ValueError(
            "persisted structural result is not the frozen passing replay"
        )
    return stable_fingerprint(expected), diagnostic.bounded_gate_passed


def _validate_epoch_progress(
    path: Path,
    expected_rows: tuple[dict[str, object], ...],
) -> None:
    """Require the durable 800-line callback ledger to equal the artifact."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("epoch progress must be a regular file")
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if len(lines) != 800 or len(expected_rows) != 800:
        raise ValueError("epoch progress must contain exactly 800 rows")

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("epoch progress contains duplicate JSON keys")
            result[key] = value
        return result

    for epoch, (line, expected_row) in enumerate(
        zip(lines, expected_rows, strict=True)
    ):
        try:
            event = json.loads(
                line.decode("utf-8"),
                object_pairs_hook=reject_duplicates,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("epoch progress is not strict JSONL") from error
        expected_event = {
            "schema_version": PAET_FORMAL_EPOCH_PROGRESS_SCHEMA,
            "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
            "objective": "pmope_joint",
            "epoch_result": expected_row,
        }
        encoded = (
            json.dumps(
                expected_event,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        if (
            event != expected_event
            or line != encoded
            or expected_row.get("epoch") != epoch
            or expected_row.get("completed_updates") != (epoch + 1) * 40
            or expected_row.get("objective") != "pmope_joint"
        ):
            raise ValueError("epoch progress differs from final epoch ledger")


def _regular_inventory(root: Path) -> dict[str, str]:
    if root.is_symlink() or root.resolve(strict=True) != root or not root.is_dir():
        raise ValueError("Formal800 output directory is not canonical")
    expected_root = _CONTROL_MEMBERS | {"final_model", "receipts"}
    entries = {path.name: path for path in root.iterdir()}
    if set(entries) != expected_root or any(path.is_symlink() for path in entries.values()):
        raise ValueError("Formal800 output inventory is incomplete or substituted")
    final = entries["final_model"]
    receipts = entries["receipts"]
    if not final.is_dir() or not receipts.is_dir():
        raise ValueError("Formal800 output directories are malformed")
    final_entries = {path.name: path for path in final.iterdir()}
    receipt_entries = {path.name: path for path in receipts.iterdir()}
    if (
        set(final_entries) != _FINAL_MEMBERS
        or set(receipt_entries) != _RECEIPT_MEMBERS
        or any(path.is_symlink() or not path.is_file() for path in final_entries.values())
        or any(path.is_symlink() or not path.is_file() for path in receipt_entries.values())
        or any(path.is_dir() for path in root.rglob("*") if path not in {final, receipts})
    ):
        raise ValueError("Formal800 terminal artifact inventory changed")
    files = {
        **{f"final_model/{name}": file_sha256(path) for name, path in final_entries.items()},
        **{f"receipts/{name}": file_sha256(path) for name, path in receipt_entries.items()},
    }
    return dict(sorted(files.items()))


@dataclass(frozen=True, slots=True)
class _LoadedPAETFormalAttemptSeal:
    issuer: object
    output: Path
    artifact: LoadedCoverageStatePAETFormalArtifact
    scientific_files: dict[str, str]
    complete_fingerprint: str
    source_closure_binding: tuple[str, str, str, int]


_LOADED_PAET_FORMAL_ATTEMPT_ISSUER = object()


@dataclass(frozen=True)
class LoadedCoverageStatePAETFormalAttempt:
    """Authenticated cross-process Formal800/D_R structural outcome."""

    artifact: LoadedCoverageStatePAETFormalArtifact
    formal_training_result_fingerprint: str
    authorization_fingerprint: str
    structural_result_fingerprint: str
    source_receipt_fingerprint: str
    post_formal_structural_retention_passed: bool
    generic_zero_level_population_gate_passed: bool
    complete_fingerprint: str
    source_closure_manifest_sha256: str = ""
    source_closure_archive_sha256: str = ""
    source_closure_content_fingerprint: str = ""
    source_closure_file_count: int = 0
    _seal: _LoadedPAETFormalAttemptSeal | None = None

    def __post_init__(self) -> None:
        if (
            type(self._seal) is not _LoadedPAETFormalAttemptSeal
            or self._seal.issuer is not _LOADED_PAET_FORMAL_ATTEMPT_ISSUER
        ):
            raise PermissionError("Formal800 attempt must come from its strict loader")
        self.verify_unchanged()

    def verify_unchanged(self) -> None:
        seal = self._seal
        if (
            type(seal) is not _LoadedPAETFormalAttemptSeal
            or seal.issuer is not _LOADED_PAET_FORMAL_ATTEMPT_ISSUER
            or self.artifact is not seal.artifact
            or self.complete_fingerprint != seal.complete_fingerprint
            or (
                self.source_closure_manifest_sha256,
                self.source_closure_archive_sha256,
                self.source_closure_content_fingerprint,
                self.source_closure_file_count,
            )
            != seal.source_closure_binding
        ):
            raise PermissionError("Formal800 attempt seal changed")
        self.artifact.verify_unchanged()
        if _regular_inventory(seal.output) != seal.scientific_files:
            raise RuntimeError("Formal800 output bytes changed after loading")


def load_coverage_state_paet_formal_attempt() -> LoadedCoverageStatePAETFormalAttempt:
    """Load the one fixed, completed Formal800 output without D_V/D_T access."""

    source_closure = _source_closure_fields(
        verify_coverage_state_paet_formal_source_closure()
    )
    bounded_seal = (
        load_repository_coverage_state_paet_bounded_artifact_seal()
    )
    bounded_seal.verify_unchanged()
    bounded_payload = bounded_seal.payload
    sealed_bounded_population = _dict(
        bounded_payload.get("bounded_population"),
        name="sealed v21 bounded population",
    )
    if (
        bounded_payload.get("structural_advancement_passed") is not True
        or bounded_payload.get("generic_population_gate_passed") is not False
        or bounded_payload.get("dataset_free_gate_passed") is not True
        or bounded_payload.get("D_R_gate_passed") is not True
        or bounded_payload.get("bounded_evidence_is_performance") is not False
        or bounded_payload.get("D_V_accessed") is not False
        or bounded_payload.get("D_T_accessed") is not False
        or bounded_payload.get("performance_claim_supported") is not False
        or bounded_payload.get("source_binding_fingerprint")
        != COVERAGE_STATE_PAET_V21_SOURCE_RECEIPT_FINGERPRINT
        or bounded_payload.get("full_D_R_scalar_cache_fingerprint")
        != COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        or bounded_payload.get("bounded_population_fingerprint")
        != COVERAGE_STATE_PAET_V21_BOUNDED_POPULATION_FINGERPRINT
        or stable_fingerprint(sealed_bounded_population)
        != COVERAGE_STATE_PAET_V21_BOUNDED_POPULATION_FINGERPRINT
    ):
        raise ValueError("bounded v21 prerequisite changed")
    _bounded_population_role_ids(sealed_bounded_population)
    root = PAET_FORMAL_ATTEMPT_OUTPUT_PATH
    files = _regular_inventory(root)
    attempt = _strict_json(root / "attempt.json", name="attempt")
    started = _strict_json(root / "STARTED.json", name="STARTED")
    complete = _strict_json(root / "COMPLETE.json", name="COMPLETE")
    attempt_fingerprint = _verify_fingerprinted(attempt, field="receipt_fingerprint", name="attempt")
    started_fingerprint = _verify_fingerprinted(
        started,
        field="receipt_fingerprint",
        name="STARTED",
    )
    complete_fingerprint = _verify_fingerprinted(complete, field="complete_fingerprint", name="COMPLETE")
    _require(attempt.get("schema_version"), PAET_FORMAL_ATTEMPT_SCHEMA, name="attempt schema")
    expected_attempt = {
        "schema_version": PAET_FORMAL_ATTEMPT_SCHEMA,
        "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
        "output_repo_path": (
            "runs/irstd1k_stage_a_seed42/"
            f"{COVERAGE_STATE_PAET_FORMAL_RUN_ID}"
        ),
        "seed": 42,
        "epochs": 800,
        "steps_per_epoch": 40,
        "updates": 32_000,
        "device": "cuda:0",
        "CUDA_VISIBLE_DEVICES": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "pause_temperature_c": 82,
        "resume_temperature_c": 75,
        "single_attempt": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        **source_closure,
    }
    if {key: value for key, value in attempt.items() if key != "receipt_fingerprint"} != expected_attempt:
        raise ValueError("attempt is not the fixed Formal800 claim")
    expected_started = {
        "schema_version": PAET_FORMAL_STARTED_SCHEMA,
        "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
        "status": "started_single_attempt",
        "attempt_fingerprint": attempt_fingerprint,
        "output_directory_reusable": False,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    if {key: value for key, value in started.items() if key != "receipt_fingerprint"} != expected_started:
        raise ValueError("STARTED is not bound to the fixed Formal800 claim")
    _require(complete.get("schema_version"), PAET_FORMAL_RUN_SCHEMA, name="COMPLETE schema")
    _require(complete.get("run_id"), COVERAGE_STATE_PAET_FORMAL_RUN_ID, name="COMPLETE run")
    _require(complete.get("status"), "complete", name="COMPLETE status")
    _require(complete.get("artifact_files"), files, name="COMPLETE artifact inventory")
    _require(complete.get("artifact_file_count"), len(files), name="COMPLETE artifact count")
    _require(
        complete.get("attempt_fingerprint"),
        attempt_fingerprint,
        name="COMPLETE attempt fingerprint",
    )
    _require(
        complete.get("started_fingerprint"),
        started_fingerprint,
        name="COMPLETE STARTED fingerprint",
    )
    _require_source_closure_binding(
        complete,
        source_closure,
        name="COMPLETE",
    )
    _false(complete, "D_V_accessed", "D_T_accessed")
    if complete.get("full_CURE_authorized") is not False or complete.get("cross_backbone_authorized") is not False:
        raise ValueError("Formal800 completion over-authorizes downstream work")

    config = _strict_json(root / "receipts/config.json", name="config receipt")
    _verify_fingerprinted(config, field="receipt_fingerprint", name="config receipt")
    _require_source_closure_binding(
        config,
        source_closure,
        name="config receipt",
    )
    inputs = _strict_json(
        root / "receipts/inputs.json",
        name="inputs receipt",
    )
    authorization = _strict_json(
        root / "receipts/authorization.json",
        name="authorization receipt",
    )
    resource = _strict_json(
        root / "receipts/training_resource.json",
        name="training resource receipt",
    )
    _verify_fingerprinted(
        inputs,
        field="receipt_fingerprint",
        name="inputs receipt",
    )
    _verify_fingerprinted(
        authorization,
        field="receipt_fingerprint",
        name="authorization receipt",
    )
    resource_fingerprint = _verify_fingerprinted(
        resource,
        field="receipt_fingerprint",
        name="training resource receipt",
    )
    final = _strict_json(root / "receipts/final_artifact.json", name="final artifact receipt")
    training = _strict_json(root / "receipts/formal_training.json", name="formal training receipt")
    structural = _strict_json(root / "receipts/structural_replay.json", name="structural replay receipt")
    decision = _strict_json(root / "receipts/decision.json", name="structural decision")
    _verify_fingerprinted(final, field="receipt_fingerprint", name="final artifact receipt")
    _verify_fingerprinted(training, field="receipt_fingerprint", name="formal training receipt")
    _verify_fingerprinted(structural, field="receipt_fingerprint", name="structural replay receipt")
    _verify_fingerprinted(decision, field="decision_fingerprint", name="structural decision")
    _require(final.get("schema_version"), PAET_FORMAL_FINAL_ARTIFACT_RECEIPT_SCHEMA, name="final artifact schema")
    _require(training.get("schema_version"), PAET_FORMAL_TRAINING_RECEIPT_SCHEMA, name="training schema")
    _require(structural.get("schema_version"), PAET_FORMAL_STRUCTURAL_REPLAY_SCHEMA, name="structural schema")
    _require(decision.get("schema_version"), PAET_FORMAL_STRUCTURAL_DECISION_SCHEMA, name="decision schema")
    for name, receipt in (("final", final), ("training", training), ("structural", structural), ("decision", decision)):
        _require(receipt.get("run_id"), COVERAGE_STATE_PAET_FORMAL_RUN_ID, name=f"{name} run")
        _false(receipt, "D_V_accessed", "D_T_accessed")

    artifact = load_coverage_state_paet_formal_artifact(
        root / "final_model",
        expected_authorization_fingerprint=_digest(final.get("authorization_fingerprint"), name="final authorization"),
        expected_result_fingerprint=_digest(final.get("formal_result_fingerprint"), name="final result"),
    )
    artifact.verify_unchanged()
    expected_members = {
        name: files[f"final_model/{name}"] for name in _FINAL_MEMBERS
    }
    config_body = {
        key: value
        for key, value in config.items()
        if key != "receipt_fingerprint"
    }
    expected_config_fields = {
        "schema_version",
        "run_id",
        "output_repo_path",
        "split",
        "runtime_splits",
        "real_inputs",
        "bounded_artifact_seal_fingerprint",
        "bounded_evidence_interpretation",
        "model",
        "budget",
        "full_D_R_contract",
        "execution",
        "final_artifact",
        "post_training_structural_replay",
        "implementation",
        "evidence_scope",
        *_SOURCE_CLOSURE_FIELD_NAMES,
    }
    config_model = _dict(config.get("model"), name="config model")
    config_budget = _dict(config.get("budget"), name="config budget")
    config_execution = _dict(
        config.get("execution"),
        name="config execution",
    )
    config_final_artifact = _dict(
        config.get("final_artifact"),
        name="config final artifact",
    )
    config_replay = _dict(
        config.get("post_training_structural_replay"),
        name="config structural replay",
    )
    config_evidence = _dict(
        config.get("evidence_scope"),
        name="config evidence scope",
    )
    config_implementation = _dict(
        config.get("implementation"),
        name="config implementation",
    )
    implementation_files = _dict(
        config_implementation.get("files"),
        name="config implementation files",
    )
    config_real_inputs = _dict(
        config.get("real_inputs"),
        name="config real inputs",
    )
    if (
        set(config_body) != expected_config_fields
        or config.get("schema_version") != PAET_FORMAL_RUN_SCHEMA
        or config.get("run_id") != COVERAGE_STATE_PAET_FORMAL_RUN_ID
        or config.get("output_repo_path")
        != (
            "runs/irstd1k_stage_a_seed42/"
            f"{COVERAGE_STATE_PAET_FORMAL_RUN_ID}"
        )
        or config.get("split") != "D_R"
        or config.get("runtime_splits") != ["D_R"]
        or config_real_inputs != _FROZEN_REAL_DR_INPUTS
        or config.get("bounded_artifact_seal_fingerprint")
        != bounded_seal.audit_fingerprint
        or config.get("bounded_evidence_interpretation")
        != "structural_advancement_only_not_performance"
        or config_model
        != {
            "class": (
                "CURELitePhaseAlignedEvidenceTransportLevelSet"
            ),
            "candidate": "PAET-BFA-v21",
            "input_interface": ["F_b", "O"],
            "config": asdict(artifact.model_config),
            "feature_channels": 64,
            "feature_stride": 4,
            "width": 32,
            "parameter_count": 64_064,
            "parameter_tensor_count": 3,
            "candidate_objective": "pmope_joint",
            "single_completion_field": True,
            "field_threshold_hex": 0.0.hex(),
            "threshold_search_performed": False,
        }
        or config_budget
        != {
            "seed": 42,
            "epochs": 800,
            "steps_per_epoch": 40,
            "updates": 32_000,
            "objectives": 1,
            "from_scratch": True,
        }
        or config.get("full_D_R_contract")
        != {
            "scalar_cache_fingerprint": (
                COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
            ),
            "formal_schedule_fingerprint": (
                COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT
            ),
        }
        or config_execution
        != {
            "device": "cuda:0",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "temperature_wrapper_repo_path": (
                "tools/run_with_gpu_temperature_control.py"
            ),
            "pause_temperature_c": 82,
            "resume_temperature_c": 75,
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
        }
        or config_final_artifact
        != {
            "directory": "final_model",
            "serialization": "safetensors",
            "checkpoint_policy": "final_model_only",
            "optimizer_state_saved": False,
            "intermediate_checkpoint_saved": False,
            "strict_loader_required": True,
            "training_and_module_state_fingerprints_separate": True,
        }
        or config_replay
        != {
            "source": (
                "same_full_D_R_cache_then_fixed_bounded_population"
            ),
            "population_seed": 42,
            "policy": "frozen_v21_bounded_structural_policy",
            "threshold_search_performed": False,
            "performance_evaluation": False,
            "D_V_authorized_only_if_structural_retention_passes": True,
            "generic_population_gate_reported_separately": True,
        }
        or stable_fingerprint(implementation_files)
        != config_implementation.get("implementation_fingerprint")
        or config_evidence
        != {
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
        }
    ):
        raise ValueError("Formal800 config receipt changed")

    inputs_body = {
        key: value
        for key, value in inputs.items()
        if key != "receipt_fingerprint"
    }
    real_inputs_payload = _dict(
        inputs.get("real_D_R_inputs"),
        name="real D_R inputs",
    )
    real_inputs_fingerprint = _digest(
        inputs.get("real_inputs_build_fingerprint"),
        name="real D_R inputs build fingerprint",
    )
    expected_inputs_body = {
        "schema_version": PAET_FORMAL_INPUTS_RECEIPT_SCHEMA,
        "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
        "real_D_R_inputs": real_inputs_payload,
        "real_inputs_build_fingerprint": real_inputs_fingerprint,
        "full_D_R_scalar_cache_fingerprint": (
            COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        ),
        "construction_invocations": 1,
        "bounded_population_constructed_before_training": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    input_fingerprints = _dict(
        real_inputs_payload.get("fingerprints"),
        name="real D_R input fingerprints",
    )
    if (
        inputs_body != expected_inputs_body
        or stable_fingerprint(real_inputs_payload)
        != real_inputs_fingerprint
        or real_inputs_fingerprint
        != bounded_payload.get("real_inputs_build_fingerprint")
        or real_inputs_payload.get("dataset") != "IRSTD-1K"
        or real_inputs_payload.get("split") != "D_R"
        or real_inputs_payload.get("runtime_splits") != ["D_R"]
        or real_inputs_payload.get("source_binding_fingerprint")
        != COVERAGE_STATE_PAET_V21_SOURCE_RECEIPT_FINGERPRINT
        or input_fingerprints.get("scalar_cache")
        != COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        or real_inputs_payload.get("execution_policy", {}).get(
            "D_V_accessed"
        )
        is not False
        or real_inputs_payload.get("execution_policy", {}).get(
            "D_T_accessed"
        )
        is not False
    ):
        raise ValueError("Formal800 inputs receipt changed")

    authorization_payload = _dict(
        authorization.get("authorization"),
        name="formal authorization payload",
    )
    authorization_fingerprint = _digest(
        authorization.get("authorization_fingerprint"),
        name="formal authorization fingerprint",
    )
    config_fingerprint = _digest(
        config.get("receipt_fingerprint"),
        name="config fingerprint",
    )
    config_implementation = _dict(
        config.get("implementation"),
        name="config implementation",
    )
    authorization_body = {
        key: value
        for key, value in authorization.items()
        if key != "receipt_fingerprint"
    }
    expected_authorization_body = {
        "schema_version": PAET_FORMAL_AUTHORIZATION_RECEIPT_SCHEMA,
        "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
        "authorization": authorization_payload,
        "authorization_fingerprint": authorization_fingerprint,
        "config_fingerprint": config_fingerprint,
        "implementation_fingerprint": config_implementation.get(
            "implementation_fingerprint"
        ),
        "formal_training_authorized": True,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    authorization_budget = _dict(
        authorization_payload.get("budget"),
        name="formal authorization budget",
    )
    authorization_exposure_checks = _dict(
        authorization_payload.get("exposure_gate_checks"),
        name="formal exposure checks",
    )
    authorization_implementation = _dict(
        authorization_payload.get("formal_implementation_binding"),
        name="formal authorization implementation",
    )
    expected_authorization_payload_fields = {
        "schema_version",
        "run_id",
        "scope",
        "runtime_splits",
        "bounded_artifact_seal_fingerprint",
        "bounded_evidence_interpretation",
        "structural_advancement_passed",
        "generic_population_gate_passed",
        "dataset_free_gate_passed",
        "D_R_identifiability_gate_passed",
        "real_inputs_build_fingerprint",
        "source_binding_fingerprint",
        "full_D_R_scalar_cache_fingerprint",
        "full_D_R_scalar_cache_counts",
        "schedule_fingerprint",
        "exposure_gate_fingerprint",
        "exposure_gate_checks",
        "budget",
        "model_config_fingerprint",
        "model_class",
        "expected_parameter_count",
        "expected_initial_model_fingerprint",
        "candidate_objective",
        "candidate_objective_policy",
        "field_threshold_hex",
        "threshold_search_performed",
        "formal_implementation_binding",
        "formal_implementation_fingerprint",
        "training_contract",
        "formal_D_R_training_authorized",
        "D_V_accessed",
        "D_T_accessed",
        "calibration_performed",
        "inference_performed",
        "performance_evaluation_performed",
        "performance_claim_supported",
        "full_CURE_authorized",
        "cross_backbone_authorized",
    }
    expected_training_contract = {
        "from_scratch": True,
        "process_local_single_attempt_claim": True,
        "cross_process_output_claim_required": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "continuation_checkpoint_consumed": False,
        "checkpoint_policy": "final_model_only",
        "intermediate_checkpoint_saved": False,
        "optimizer_state_saved": False,
    }
    if (
        authorization_body != expected_authorization_body
        or set(authorization_payload)
        != expected_authorization_payload_fields
        or stable_fingerprint(authorization_payload)
        != authorization_fingerprint
        or authorization_fingerprint
        != artifact.authorization_fingerprint
        or authorization_payload.get("run_id")
        != COVERAGE_STATE_PAET_FORMAL_RUN_ID
        or authorization_payload.get("schema_version")
        != COVERAGE_STATE_PAET_FORMAL_AUTHORIZATION_SCHEMA
        or authorization_payload.get("scope") != "D_R_formal_800"
        or authorization_payload.get("runtime_splits") != ["D_R"]
        or authorization_payload.get(
            "bounded_artifact_seal_fingerprint"
        )
        != bounded_seal.audit_fingerprint
        or authorization_payload.get(
            "bounded_evidence_interpretation"
        )
        != "structural_advancement_only_not_performance"
        or authorization_payload.get("structural_advancement_passed")
        is not True
        or authorization_payload.get("generic_population_gate_passed")
        is not False
        or authorization_payload.get("dataset_free_gate_passed")
        is not True
        or authorization_payload.get("D_R_identifiability_gate_passed")
        is not True
        or authorization_payload.get("real_inputs_build_fingerprint")
        != real_inputs_fingerprint
        or authorization_payload.get("source_binding_fingerprint")
        != COVERAGE_STATE_PAET_V21_SOURCE_RECEIPT_FINGERPRINT
        or authorization_payload.get(
            "full_D_R_scalar_cache_fingerprint"
        )
        != COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        or authorization_payload.get("full_D_R_scalar_cache_counts")
        != bounded_payload.get("full_D_R_scalar_cache_counts")
        or authorization_payload.get("schedule_fingerprint")
        != COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT
        or authorization_payload.get("exposure_gate_fingerprint")
        != COVERAGE_STATE_PAET_FORMAL_EXPOSURE_GATE_FINGERPRINT
        or authorization_exposure_checks
        != _EXPECTED_EXPOSURE_GATE_CHECKS
        or authorization_budget
        != {
            "seed": 42,
            "epochs": 800,
            "steps_per_epoch": 40,
            "updates": 32_000,
            "objectives": 1,
        }
        or authorization_payload.get("expected_parameter_count")
        != 64_064
        or authorization_payload.get("model_config_fingerprint")
        != stable_fingerprint(
            _formal_model_config_payload(artifact.model_config)
        )
        or authorization_payload.get("model_class")
        != "CURELitePhaseAlignedEvidenceTransportLevelSet"
        or authorization_payload.get(
            "expected_initial_model_fingerprint"
        )
        != COVERAGE_STATE_PAET_FORMAL_INITIAL_MODEL_FINGERPRINT
        or authorization_payload.get("candidate_objective")
        != "pmope_joint"
        or authorization_payload.get("candidate_objective_policy")
        != CSLF_PMOPE_POLICY
        or authorization_payload.get("field_threshold_hex")
        != 0.0.hex()
        or authorization_payload.get("threshold_search_performed")
        is not False
        or authorization_payload.get("training_contract")
        != expected_training_contract
        or authorization_payload.get("formal_D_R_training_authorized")
        is not True
        or authorization_payload.get("D_V_accessed") is not False
        or authorization_payload.get("D_T_accessed") is not False
        or stable_fingerprint(
            authorization_implementation
        )
        != authorization_payload.get(
            "formal_implementation_fingerprint"
        )
        or not authorization_implementation
        or any(
            implementation_files.get(name) != digest
            for name, digest in authorization_implementation.items()
        )
        or authorization_payload.get("calibration_performed") is not False
        or authorization_payload.get("inference_performed") is not False
        or authorization_payload.get(
            "performance_evaluation_performed"
        )
        is not False
        or authorization_payload.get("performance_claim_supported")
        is not False
        or authorization_payload.get("full_CURE_authorized") is not False
        or authorization_payload.get("cross_backbone_authorized")
        is not False
    ):
        raise ValueError("Formal800 authorization receipt changed")

    resource_body = {
        key: value
        for key, value in resource.items()
        if key != "receipt_fingerprint"
    }
    expected_resource_fields = {
        "schema_version",
        "run_id",
        "device",
        "scope",
        "updates",
        "baseline_allocated_bytes",
        "baseline_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "incremental_peak_allocated_bytes",
        "incremental_peak_reserved_bytes",
        "elapsed_ns",
        "ns_per_update",
        "oom_observed",
        "training_invocations",
        "performance_measurement",
        "D_V_accessed",
        "D_T_accessed",
    }
    integer_resource_fields = (
        "baseline_allocated_bytes",
        "baseline_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "incremental_peak_allocated_bytes",
        "incremental_peak_reserved_bytes",
        "elapsed_ns",
    )
    if (
        set(resource_body) != expected_resource_fields
        or resource.get("schema_version")
        != PAET_FORMAL_RESOURCE_RECEIPT_SCHEMA
        or resource.get("run_id") != COVERAGE_STATE_PAET_FORMAL_RUN_ID
        or resource.get("device") != "cuda:0"
        or resource.get("scope") != "single_formal_training_invocation"
        or type(resource.get("updates")) is not int
        or resource.get("updates") != 32_000
        or any(
            type(resource.get(name)) is not int
            or int(resource[name]) < (1 if name == "elapsed_ns" else 0)
            for name in integer_resource_fields
        )
        or resource.get("peak_allocated_bytes")
        < resource.get("baseline_allocated_bytes")
        or resource.get("peak_reserved_bytes")
        < resource.get("baseline_reserved_bytes")
        or resource.get("incremental_peak_allocated_bytes")
        != resource.get("peak_allocated_bytes")
        - resource.get("baseline_allocated_bytes")
        or resource.get("incremental_peak_reserved_bytes")
        != resource.get("peak_reserved_bytes")
        - resource.get("baseline_reserved_bytes")
        or type(resource.get("ns_per_update")) is not float
        or resource.get("ns_per_update") <= 0.0
        or float(resource["ns_per_update"])
        != resource["elapsed_ns"] / 32_000
        or resource.get("oom_observed") is not False
        or type(resource.get("training_invocations")) is not int
        or resource.get("training_invocations") != 1
        or resource.get("performance_measurement") is not False
        or resource.get("D_V_accessed") is not False
        or resource.get("D_T_accessed") is not False
    ):
        raise ValueError("Formal800 training resource receipt changed")
    _validate_epoch_progress(
        root / "receipts/epoch_progress.jsonl",
        artifact.epoch_logs,
    )
    final_body = {
        key: value
        for key, value in final.items()
        if key != "receipt_fingerprint"
    }
    expected_final_body = {
        "schema_version": PAET_FORMAL_FINAL_ARTIFACT_RECEIPT_SCHEMA,
        "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
        "artifact_repo_path": (
            "runs/irstd1k_stage_a_seed42/"
            f"{COVERAGE_STATE_PAET_FORMAL_RUN_ID}/final_model"
        ),
        "artifact_fingerprint": artifact.artifact_fingerprint,
        "artifact_receipt_sha256": artifact.receipt_sha256,
        "authorization_fingerprint": artifact.authorization_fingerprint,
        "formal_result_fingerprint": artifact.formal_result_fingerprint,
        "training_model_fingerprint": artifact.training_model_fingerprint,
        "module_state_fingerprint": artifact.module_state_fingerprint,
        "member_files": expected_members,
        "strict_loader_verified": True,
        "checkpoint_policy": "final_model_only",
        "optimizer_state_saved": False,
        "intermediate_checkpoint_saved": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    training_body = {
        key: value
        for key, value in training.items()
        if key != "receipt_fingerprint"
    }
    expected_training_body = {
        "schema_version": PAET_FORMAL_TRAINING_RECEIPT_SCHEMA,
        "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
        "formal_result": artifact.formal_result_payload,
        "formal_training_result_fingerprint": (
            artifact.formal_result_fingerprint
        ),
        "authorization_fingerprint": artifact.authorization_fingerprint,
        "training_invocations": 1,
        "completed_updates": 32_000,
        "epoch_callback_rows": 800,
        "resource_measurement_fingerprint": resource_fingerprint,
        "final_artifact_fingerprint": artifact.artifact_fingerprint,
        "training_model_fingerprint": artifact.training_model_fingerprint,
        "module_state_fingerprint": artifact.module_state_fingerprint,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "performance_evaluation_performed": False,
    }
    if (
        final_body != expected_final_body
        or training_body != expected_training_body
        or final.get("artifact_fingerprint") != artifact.artifact_fingerprint
        or final.get("artifact_receipt_sha256") != artifact.receipt_sha256
        or final.get("training_model_fingerprint") != artifact.training_model_fingerprint
        or final.get("module_state_fingerprint") != artifact.module_state_fingerprint
        or final.get("member_files") != expected_members
        or final.get("strict_loader_verified") is not True
        or training.get("formal_training_result_fingerprint") != artifact.formal_result_fingerprint
        or training.get("authorization_fingerprint") != artifact.authorization_fingerprint
        or training.get("formal_result") != artifact.formal_result_payload
        or training.get("final_artifact_fingerprint") != artifact.artifact_fingerprint
        or training.get("training_model_fingerprint") != artifact.training_model_fingerprint
        or training.get("module_state_fingerprint") != artifact.module_state_fingerprint
        or training.get("training_invocations") != 1
        or training.get("completed_updates") != 32_000
        or training.get("epoch_callback_rows") != 800
        or training.get("resource_measurement_fingerprint")
        != resource_fingerprint
        or complete.get("formal_training_result_fingerprint") != artifact.formal_result_fingerprint
        or complete.get("final_artifact_fingerprint") != artifact.artifact_fingerprint
        or complete.get("artifact_receipt_sha256") != artifact.receipt_sha256
        or complete.get("training_model_fingerprint") != artifact.training_model_fingerprint
        or complete.get("module_state_fingerprint") != artifact.module_state_fingerprint
    ):
        raise ValueError("Formal800 final artifact receipts do not bind the strict loader")

    bounded_population = _dict(
        structural.get("bounded_population"),
        name="bounded population",
    )
    if bounded_population != sealed_bounded_population:
        raise ValueError(
            "structural replay changed the frozen bounded population"
        )
    result = structural.get("structural_result")
    structural_fingerprint, generic_gate_passed = (
        _validate_structural_result(
            result,
            artifact,
            bounded_population=bounded_population,
        )
    )
    _require(
        structural.get("structural_result_fingerprint"),
        structural_fingerprint,
        name="structural result fingerprint",
    )
    structural_body = {
        key: value
        for key, value in structural.items()
        if key != "receipt_fingerprint"
    }
    expected_structural_body = {
        "schema_version": PAET_FORMAL_STRUCTURAL_REPLAY_SCHEMA,
        "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
        "source_full_D_R_scalar_cache_fingerprint": (
            COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        ),
        "bounded_population": bounded_population,
        "bounded_population_fingerprint": (
            COVERAGE_STATE_PAET_V21_BOUNDED_POPULATION_FINGERPRINT
        ),
        "structural_result": result,
        "structural_result_fingerprint": structural_fingerprint,
        "training_model_fingerprint": artifact.training_model_fingerprint,
        "module_state_fingerprint": artifact.module_state_fingerprint,
        "evaluation_invocations": 1,
        "paet_structural_retention_gate_passed": True,
        "generic_zero_level_population_gate_passed": generic_gate_passed,
        "performance_evaluation_performed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    if (
        structural_body != expected_structural_body
        or stable_fingerprint(bounded_population)
        != COVERAGE_STATE_PAET_V21_BOUNDED_POPULATION_FINGERPRINT
        or structural.get("bounded_population_fingerprint")
        != COVERAGE_STATE_PAET_V21_BOUNDED_POPULATION_FINGERPRINT
        or structural.get("structural_result_fingerprint") != complete.get("structural_result_fingerprint")
        or structural.get("paet_structural_retention_gate_passed") is not True
        or structural.get("generic_zero_level_population_gate_passed")
        != generic_gate_passed
        or structural.get("source_full_D_R_scalar_cache_fingerprint") != COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        or structural.get("evaluation_invocations") != 1
        or structural.get("training_model_fingerprint") != artifact.training_model_fingerprint
        or structural.get("module_state_fingerprint") != artifact.module_state_fingerprint
        or structural.get("performance_evaluation_performed") is not False
    ):
        raise ValueError("persisted structural replay is not a passing strict receipt")

    decision_body = {
        key: value
        for key, value in decision.items()
        if key != "decision_fingerprint"
    }
    expected_decision_checks = {
        "formal_training_complete": True,
        "strict_final_artifact_bound": True,
        "structural_replay_invoked_once": True,
        "frozen_paet_structural_retention_gate_passed": True,
        "D_V_and_D_T_not_accessed": True,
    }
    expected_decision_body = {
        "schema_version": PAET_FORMAL_STRUCTURAL_DECISION_SCHEMA,
        "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
        "status": (
            "PAET_BFA_V21_FORMAL800_STRUCTURAL_PASS_AUTHORIZE_D_V"
        ),
        "formal_training_complete": True,
        "strict_final_artifact_bound": True,
        "paet_structural_retention_gate_passed": True,
        "generic_zero_level_population_gate_passed": generic_gate_passed,
        "generic_gate_is_D_V_prerequisite": False,
        "structural_gate_and_generic_gate_are_separate": True,
        "checks": expected_decision_checks,
        "failed_checks": [],
        "D_V_authorized": True,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "performance_evaluation_performed": False,
        "performance_gate_passed": None,
        "performance_claim_supported": False,
        "final_model_success_established": False,
        "full_CURE_authorized": False,
        "cross_backbone_authorized": False,
        "next_action": "RUN_ONE_SEPARATE_STRICT_D_V_REVEAL",
        "bindings": {
            "authorization_fingerprint": (
                artifact.authorization_fingerprint
            ),
            "formal_training_result_fingerprint": (
                artifact.formal_result_fingerprint
            ),
            "final_artifact_fingerprint": artifact.artifact_fingerprint,
            "training_model_fingerprint": (
                artifact.training_model_fingerprint
            ),
            "module_state_fingerprint": (
                artifact.module_state_fingerprint
            ),
            "structural_result_fingerprint": structural_fingerprint,
        },
    }
    if (
        decision_body != expected_decision_body
        or decision.get("status") != "PAET_BFA_V21_FORMAL800_STRUCTURAL_PASS_AUTHORIZE_D_V"
        or decision.get("D_V_authorized") is not True
        or decision.get("paet_structural_retention_gate_passed") is not True
        or decision.get("generic_gate_is_D_V_prerequisite") is not False
        or decision.get("structural_gate_and_generic_gate_are_separate") is not True
        or decision.get("generic_zero_level_population_gate_passed")
        != generic_gate_passed
        or decision.get("bindings", {}).get("final_artifact_fingerprint") != artifact.artifact_fingerprint
        or decision.get("bindings", {}).get("authorization_fingerprint") != artifact.authorization_fingerprint
        or decision.get("bindings", {}).get("formal_training_result_fingerprint") != artifact.formal_result_fingerprint
        or decision.get("bindings", {}).get("structural_result_fingerprint") != structural_fingerprint
        or complete.get("decision") != decision.get("status")
        or complete.get("D_V_authorized") is not True
        or complete.get("paet_structural_retention_gate_passed") is not True
        or complete.get("generic_zero_level_population_gate_passed")
        != generic_gate_passed
    ):
        raise ValueError("Formal800 structural decision is not a D_V-only pass")

    complete_body = {
        key: value
        for key, value in complete.items()
        if key != "complete_fingerprint"
    }
    expected_complete_body = {
        "schema_version": PAET_FORMAL_RUN_SCHEMA,
        "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
        "status": "complete",
        "decision": expected_decision_body["status"],
        "formal_training_complete": True,
        "formal_training_result_fingerprint": (
            artifact.formal_result_fingerprint
        ),
        "final_artifact_fingerprint": artifact.artifact_fingerprint,
        "artifact_receipt_sha256": artifact.receipt_sha256,
        "training_model_fingerprint": artifact.training_model_fingerprint,
        "module_state_fingerprint": artifact.module_state_fingerprint,
        "structural_result_fingerprint": structural_fingerprint,
        "paet_structural_retention_gate_passed": True,
        "generic_zero_level_population_gate_passed": generic_gate_passed,
        "structural_gate_and_generic_gate_are_separate": True,
        "D_V_authorized": True,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "performance_evaluation_performed": False,
        "performance_gate_passed": None,
        "performance_claim_supported": False,
        "artifact_files": files,
        "artifact_file_count": len(files),
        "single_attempt": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "full_CURE_authorized": False,
        "cross_backbone_authorized": False,
        "attempt_fingerprint": attempt_fingerprint,
        "started_fingerprint": started_fingerprint,
        **source_closure,
    }
    if complete_body != expected_complete_body:
        raise ValueError("Formal800 COMPLETE receipt changed")

    return LoadedCoverageStatePAETFormalAttempt(
        artifact=artifact,
        formal_training_result_fingerprint=artifact.formal_result_fingerprint,
        authorization_fingerprint=artifact.authorization_fingerprint,
        structural_result_fingerprint=structural_fingerprint,
        source_receipt_fingerprint=_digest(
            result.get("source_receipt_fingerprint"),
            name="structural source receipt",
        ),
        post_formal_structural_retention_passed=True,
        generic_zero_level_population_gate_passed=generic_gate_passed,
        complete_fingerprint=complete_fingerprint,
        source_closure_manifest_sha256=str(
            source_closure["source_closure_manifest_sha256"]
        ),
        source_closure_archive_sha256=str(
            source_closure["source_closure_archive_sha256"]
        ),
        source_closure_content_fingerprint=str(
            source_closure["source_closure_content_fingerprint"]
        ),
        source_closure_file_count=int(
            source_closure["source_closure_file_count"]
        ),
        _seal=_LoadedPAETFormalAttemptSeal(
            issuer=_LOADED_PAET_FORMAL_ATTEMPT_ISSUER,
            output=root,
            artifact=artifact,
            scientific_files=files,
            complete_fingerprint=complete_fingerprint,
            source_closure_binding=(
                str(source_closure["source_closure_manifest_sha256"]),
                str(source_closure["source_closure_archive_sha256"]),
                str(source_closure["source_closure_content_fingerprint"]),
                int(source_closure["source_closure_file_count"]),
            ),
        ),
    )


__all__ = [
    "LoadedCoverageStatePAETFormalAttempt",
    "PAET_FORMAL_ATTEMPT_OUTPUT_PATH",
    "load_coverage_state_paet_formal_attempt",
]
