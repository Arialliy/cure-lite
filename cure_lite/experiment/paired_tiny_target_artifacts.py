"""Create-only artifacts for the late tiny-target representability audit.

The publication protocol is deliberately split into two commands:

``preflight``
    validates the frozen configuration and builds the exhaustive geometric
    case catalog.  It never calls the MILP solver.

``execute``
    requires a strictly loaded completed preflight, solves every catalog case
    in canonical order, re-aggregates the all-case decision, and publishes a
    self-contained certificate package.

The configuration binds the core, this artifact module, and the CLI by
repo-relative path and SHA256.  Neither implementation file embeds a
configuration fingerprint, avoiding a circular source/configuration hash.
All publications are create-only, contain no timestamps or machine/device
identity, and retain ``.incomplete`` if an exception interrupts publication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import multiprocessing
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from ..cache.schema import file_sha256, stable_fingerprint
from .paired_tiny_target_representability import (
    BIG_M,
    BILINEAR_WEIGHT_DENOMINATOR,
    CERTIFYING_MARGIN_MIN,
    DEFAULT_SOLVER_OPTIONS,
    EXPECTED_CONCRETE_PLACEMENTS,
    EXPECTED_SHAPE_COUNTS,
    EXPECTED_TOTAL_SHAPES,
    FP_COMPONENTS_PER_MP_MAX,
    GAMMA_MAX,
    LOW_GRID_SIZE,
    MARGIN_OBJECTIVE_SCALE,
    MAX_FALSE_ADDITION_PIXELS,
    OUTPUT_GRID_SIZE,
    PIXEL_FA_MAX,
    RAW_BACKGROUND_FA_MAX,
    RETENTION_REQUIRED,
    TARGET_AREAS,
    VERIFY_TOLERANCE,
    TinyTargetCaseCatalog,
    TinyTargetCaseCertificate,
    _build_tiny_target_decision_without_solver_replay,
    build_tiny_target_case_catalog,
    solve_tiny_target_case,
)


TINY_TARGET_CONFIG_SCHEMA = (
    "cure-lite-paired-tiny-target-representability-config-v1"
)
TINY_TARGET_PREFLIGHT_RECEIPT_SCHEMA = (
    "cure-lite-paired-tiny-target-representability-preflight-receipt-v1"
)
TINY_TARGET_PREFLIGHT_COMPLETE_SCHEMA = (
    "cure-lite-paired-tiny-target-representability-preflight-complete-v1"
)
TINY_TARGET_CERTIFICATE_SET_SCHEMA = (
    "cure-lite-paired-tiny-target-representability-certificate-set-v1"
)
TINY_TARGET_PREFLIGHT_BINDING_SCHEMA = (
    "cure-lite-paired-tiny-target-representability-preflight-binding-v1"
)
TINY_TARGET_RESULT_SCHEMA = (
    "cure-lite-paired-tiny-target-representability-result-v1"
)
TINY_TARGET_AUDIT_RECEIPT_SCHEMA = (
    "cure-lite-paired-tiny-target-representability-audit-receipt-v1"
)
TINY_TARGET_EXECUTION_COMPLETE_SCHEMA = (
    "cure-lite-paired-tiny-target-representability-execution-complete-v1"
)
TINY_TARGET_COMPARISON_SCHEMA = (
    "cure-lite-paired-tiny-target-representability-byte-comparison-v1"
)

_ROOT = Path(__file__).resolve().parents[2]
_CORE_REPO_PATH = (
    "cure_lite/experiment/paired_tiny_target_representability.py"
)
_ARTIFACT_REPO_PATH = (
    "cure_lite/experiment/paired_tiny_target_artifacts.py"
)
_CLI_REPO_PATH = "tools/run_paired_tiny_target_representability.py"
_TEMPLATE_REPO_PATH = (
    "protocols/IRSTD-1K/paired_tiny_target_representability_v1/"
    "config.template.json"
)
_IMPLEMENTATION_REPO_PATHS = frozenset(
    {_CORE_REPO_PATH, _ARTIFACT_REPO_PATH, _CLI_REPO_PATH}
)
_INCOMPLETE = ".incomplete"
_COMPLETE = "COMPLETE.json"
_PREFLIGHT_FILES = frozenset(
    {
        "frozen_config.json",
        "case_catalog.json",
        "preflight_receipt.json",
        _COMPLETE,
    }
)
_EXECUTION_FILES = frozenset(
    {
        "frozen_config.json",
        "preflight_binding.json",
        "case_catalog.json",
        "case_certificates.json",
        "result.json",
        "decision.json",
        "audit_receipt.json",
        _COMPLETE,
    }
)
_STRICT_LOAD_TOKEN = object()

_NON_AUTHORIZATION_BOUNDARY: dict[str, bool] = {
    "historical_pretraining_gate_satisfied": False,
    "historical_wave_a_decision_may_change": False,
    "current_paired_version_innovation_established": False,
    "training_authorized": False,
    "D_V_or_D_T_authorized": False,
    "calibration_or_inference_authorized": False,
    "full_cure_authorized": False,
    "cross_backbone_authorized": False,
}
_FORBIDDEN_TRUE_KEYS = frozenset(
    {
        *_NON_AUTHORIZATION_BOUNDARY,
        "backbone_integration_authorized",
        "confirmation_authorized",
        "authorizes_confirmation",
        "authorizes_formal_training",
        "authorizes_full_cure",
        "authorizes_cross_backbone",
        "authorizes_D_V_or_D_T",
        "D_V_accessed",
        "D_T_accessed",
        "runtime_dataset_access",
        "checkpoint_access",
        "model_weight_access",
        "training",
        "calibration",
        "learned_model_inference",
        "full_cure",
        "backbone_integration",
        "backbone_comparison",
        "decoder_architecture_change",
        "loss_change",
        "target_change",
        "target_dilation",
        "post_result_tuning",
    }
)


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 string")
    return value


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _fingerprinted(
    payload: Mapping[str, object],
    *,
    field_name: str,
) -> dict[str, object]:
    value = dict(payload)
    if field_name in value:
        raise ValueError(f"payload already contains {field_name}")
    value[field_name] = stable_fingerprint(value)
    return value


def _verify_fingerprint(
    payload: Mapping[str, Any],
    *,
    name: str,
    field_name: str,
) -> None:
    claimed = payload.get(field_name)
    unsigned = dict(payload)
    unsigned.pop(field_name, None)
    if not isinstance(claimed, str) or stable_fingerprint(unsigned) != claimed:
        raise RuntimeError(f"{name} fingerprint is inconsistent")


def _strict_json(
    path: Path,
    *,
    name: str,
    require_canonical_bytes: bool = True,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")

    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{name} contains non-finite value {value}")

    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} is not UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    if require_canonical_bytes and raw != _json_bytes(payload):
        raise ValueError(f"{name} is not canonical JSON")
    return payload


def _write_new_json(path: Path, payload: object) -> None:
    """Atomically create one canonical JSON file without replacement."""

    encoded = _json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite tiny-target artifact {path}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _canonical_input_file(path: str | Path, *, name: str) -> Path:
    candidate = Path(path).expanduser()
    absolute = Path(os.path.abspath(candidate))
    if candidate.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if resolved != absolute or resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{name} must be a canonical regular file")
    return resolved


def _canonical_input_directory(
    path: str | Path,
    *,
    name: str,
) -> Path:
    candidate = Path(path).expanduser()
    absolute = Path(os.path.abspath(candidate))
    if candidate.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if (
        resolved != absolute
        or resolved.is_symlink()
        or not resolved.is_dir()
    ):
        raise ValueError(f"{name} must be a canonical regular directory")
    return resolved


def _existing_ancestors(path: Path) -> tuple[Path, ...]:
    values: list[Path] = []
    current = path
    while True:
        if current.exists() or current.is_symlink():
            values.append(current)
        if current == current.parent:
            break
        current = current.parent
    return tuple(values)


def _new_output_path(path: str | Path, *, name: str) -> Path:
    candidate = Path(path).expanduser()
    absolute = Path(os.path.abspath(candidate))
    if absolute.exists() or absolute.is_symlink():
        raise FileExistsError(f"{name} already exists: {absolute}")
    for ancestor in _existing_ancestors(absolute.parent):
        if ancestor.is_symlink():
            raise ValueError(f"{name} may not traverse a symbolic link")
    return absolute


def _begin_publication(path: str | Path, *, name: str) -> Path:
    root = _new_output_path(path, name=name)
    root.mkdir(parents=True, exist_ok=False)
    marker = root / _INCOMPLETE
    with marker.open("xb") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    return root


def _finish_publication(root: Path, complete: Mapping[str, object]) -> None:
    if not (root / _INCOMPLETE).is_file():
        raise RuntimeError("tiny-target publication lost its incomplete marker")
    if (root / _COMPLETE).exists() or (root / _COMPLETE).is_symlink():
        raise RuntimeError("COMPLETE.json must be published exactly once")
    _write_new_json(root / _COMPLETE, complete)
    (root / _INCOMPLETE).unlink()


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {_INCOMPLETE, _COMPLETE}
    }


def _assert_no_forbidden_true(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if (
                isinstance(key, str)
                and key in _FORBIDDEN_TRUE_KEYS
                and child is True
            ):
                raise RuntimeError(f"{path}.{key} may not be true")
            _assert_no_forbidden_true(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_true(child, path=f"{path}[{index}]")


def _repo_file(path_text: object, *, name: str) -> Path:
    if (
        not isinstance(path_text, str)
        or not path_text
        or Path(path_text).is_absolute()
    ):
        raise ValueError(f"{name} must be a non-empty repo-relative path")
    path = _canonical_input_file(_ROOT / path_text, name=name)
    if path.relative_to(_ROOT).as_posix() != path_text:
        raise RuntimeError(f"{name} does not resolve to its declared path")
    return path


def _validate_file_binding(
    binding: object,
    *,
    name: str,
) -> tuple[str, str]:
    value = _require_mapping(binding, name=name)
    if set(value) != {"repo_path", "file_sha256"}:
        raise ValueError(f"{name} has an unexpected field inventory")
    path_text = value.get("repo_path")
    expected_sha = _require_sha256(
        value.get("file_sha256"),
        name=f"{name}.file_sha256",
    )
    path = _repo_file(path_text, name=f"{name}.repo_path")
    if file_sha256(path) != expected_sha:
        raise RuntimeError(f"{name} file SHA256 differs from the freeze")
    return str(path_text), expected_sha


def _implementation_bindings(
    config: Mapping[str, Any],
) -> dict[str, str]:
    section = _require_mapping(
        config.get("implementation_bindings"),
        name="implementation_bindings",
    )
    if set(section) != {
        "binding_direction",
        "config_self_hash_embedded",
        "source_files",
    }:
        raise ValueError("implementation_bindings inventory changed")
    if (
        section.get("binding_direction")
        != "config_to_implementation_files"
        or section.get("config_self_hash_embedded") is not False
    ):
        raise ValueError("implementation binding direction changed")
    source_files = section.get("source_files")
    if not isinstance(source_files, list) or len(source_files) != 3:
        raise ValueError(
            "implementation_bindings.source_files must contain three entries"
        )
    bindings = dict(
        _validate_file_binding(
            item,
            name=f"implementation_bindings.source_files[{index}]",
        )
        for index, item in enumerate(source_files)
    )
    if frozenset(bindings) != _IMPLEMENTATION_REPO_PATHS:
        raise RuntimeError("implementation source-file inventory changed")
    return dict(sorted(bindings.items()))


def _validate_template_binding(config: Mapping[str, Any]) -> None:
    section = _require_mapping(
        config.get("template_binding"),
        name="template_binding",
    )
    if set(section) != {"repo_path", "file_sha256"}:
        raise ValueError("template_binding inventory changed")
    path_text, _ = _validate_file_binding(
        section,
        name="template_binding",
    )
    if path_text != _TEMPLATE_REPO_PATH:
        raise RuntimeError("tiny-target template path changed")
    template = _strict_json(
        _ROOT / _TEMPLATE_REPO_PATH,
        name="tiny-target config template",
        require_canonical_bytes=False,
    )
    semantic = dict(template)
    for key in (
        "schema_version",
        "artifact_kind",
        "template_status",
        "future_materialization_requirements",
    ):
        semantic.pop(key, None)
    for key, value in semantic.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"executable config differs from template section {key}"
            )
    expected_keys = {
        *semantic,
        "schema_version",
        "artifact_kind",
        "template_binding",
        "implementation_bindings",
        "config_fingerprint",
    }
    if set(config) != expected_keys:
        raise ValueError("executable config field inventory changed")


def _validate_existing_bindings(config: Mapping[str, Any]) -> None:
    section = _require_mapping(
        config.get("frozen_existing_bindings"),
        name="frozen_existing_bindings",
    )
    if set(section) != {
        "paired_objective",
        "existing_source_files",
        "binding_policy",
    }:
        raise ValueError("frozen existing binding inventory changed")
    paired = _require_mapping(
        section.get("paired_objective"),
        name="frozen_existing_bindings.paired_objective",
    )
    if set(paired) != {
        "repo_path",
        "file_sha256",
        "receipt_fingerprint",
    }:
        raise ValueError("paired-objective binding inventory changed")
    path_text = paired.get("repo_path")
    path = _repo_file(path_text, name="paired-objective binding")
    expected_sha = _require_sha256(
        paired.get("file_sha256"),
        name="paired-objective file_sha256",
    )
    if file_sha256(path) != expected_sha:
        raise RuntimeError("paired-objective source hash changed")
    objective = _strict_json(
        path,
        name="paired-objective receipt",
        require_canonical_bytes=False,
    )
    if objective.get("receipt_fingerprint") != paired.get(
        "receipt_fingerprint"
    ):
        raise RuntimeError("paired-objective receipt fingerprint changed")
    existing = section.get("existing_source_files")
    if not isinstance(existing, list) or not existing:
        raise ValueError("existing source bindings must be a non-empty list")
    paths: list[str] = []
    for index, binding in enumerate(existing):
        path_name, _ = _validate_file_binding(
            binding,
            name=f"frozen_existing_bindings.existing_source_files[{index}]",
        )
        paths.append(path_name)
    if len(paths) != len(set(paths)):
        raise ValueError("existing source bindings contain duplicate paths")
    if (
        section.get("binding_policy")
        != "every_bound_path_and_sha_must_match_before_materialization_or_execution"
    ):
        raise ValueError("existing binding policy changed")


def _validate_config_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(payload)
    _verify_fingerprint(
        config,
        name="tiny-target config",
        field_name="config_fingerprint",
    )
    if (
        config.get("schema_version") != TINY_TARGET_CONFIG_SCHEMA
        or config.get("audit_name")
        != "stride_four_tiny_target_representability"
        or config.get("artifact_kind") != "executable_frozen_config"
    ):
        raise RuntimeError("tiny-target executable config identity changed")
    _validate_template_binding(config)
    if "future_materialization_requirements" in config:
        raise RuntimeError(
            "executable config may not retain template materialization tasks"
        )
    if "template_status" in config:
        raise RuntimeError("executable config may not retain template status")
    temporal = _require_mapping(
        config.get("temporal_semantics"),
        name="temporal_semantics",
    )
    required_temporal = {
        "designation": "late_compliance_audit",
        "original_required_timing": "before_paired_training",
        "original_required_timing_was_met": False,
        "retroactive_gate_satisfaction_is_possible": False,
        "purpose": "prospective_structural_diagnosis_only",
        "pass_cannot_change_historical_wave_a_decision": True,
        "pass_cannot_authorize_historical_paired_training": True,
        "pass_cannot_authorize_confirmation": True,
        "pass_cannot_authorize_full_cure": True,
        "pass_cannot_authorize_cross_backbone_evaluation": True,
    }
    if any(temporal.get(key) != value for key, value in required_temporal.items()):
        raise RuntimeError("late-audit temporal semantics changed")
    scope = _require_mapping(config.get("scope"), name="scope")
    for key in (
        "runtime_dataset_access",
        "D_R_access",
        "D_V_access",
        "D_T_access",
        "checkpoint_access",
        "model_weight_access",
        "training",
        "calibration",
        "learned_model_inference",
        "full_cure",
        "backbone_integration",
        "backbone_comparison",
        "decoder_architecture_change",
        "loss_change",
        "target_change",
        "target_dilation",
        "post_result_tuning",
    ):
        if scope.get(key) is not False:
            raise RuntimeError(f"scope.{key} must remain false")
    if scope.get("runtime_split_access") != []:
        raise RuntimeError("runtime split access must remain empty")

    geometry = _require_mapping(
        config.get("canonical_geometry"),
        name="canonical_geometry",
    )
    output = _require_mapping(
        geometry.get("output_grid"),
        name="canonical_geometry.output_grid",
    )
    low = _require_mapping(
        geometry.get("low_resolution_grid"),
        name="canonical_geometry.low_resolution_grid",
    )
    if (
        output != {"height": OUTPUT_GRID_SIZE, "width": OUTPUT_GRID_SIZE}
        or low != {"height": LOW_GRID_SIZE, "width": LOW_GRID_SIZE}
        or geometry.get("target_pixel_areas") != list(TARGET_AREAS)
        or geometry.get("target_connectivity") != 8
        or geometry.get("target_component_count") != 1
    ):
        raise RuntimeError("canonical tiny-target geometry changed")
    shapes = _require_mapping(
        config.get("shape_catalog"),
        name="shape_catalog",
    )
    if (
        shapes.get("expected_shape_count_by_area")
        != {str(key): value for key, value in EXPECTED_SHAPE_COUNTS.items()}
        or shapes.get("expected_total_oriented_shapes")
        != EXPECTED_TOTAL_SHAPES
        or shapes.get("rotation_quotient") is not False
        or shapes.get("reflection_quotient") is not False
    ):
        raise RuntimeError("shape-catalog freeze changed")
    placements = _require_mapping(
        config.get("concrete_placement_catalog"),
        name="concrete_placement_catalog",
    )
    if (
        placements.get("expected_total_concrete_placements")
        != EXPECTED_CONCRETE_PLACEMENTS
        or placements.get("every_concrete_placement_must_be_covered_exactly_once")
        is not True
    ):
        raise RuntimeError("concrete-placement freeze changed")
    operator = _require_mapping(
        config.get("bilinear_operator"),
        name="bilinear_operator",
    )
    if (
        operator.get("source_grid") != [LOW_GRID_SIZE, LOW_GRID_SIZE]
        or operator.get("destination_grid")
        != [OUTPUT_GRID_SIZE, OUTPUT_GRID_SIZE]
        or operator.get("mode") != "bilinear"
        or operator.get("align_corners") is not False
        or operator.get("antialias") is not False
        or _require_mapping(
            operator.get("coefficient_encoding"),
            name="bilinear_operator.coefficient_encoding",
        ).get("common_denominator")
        != BILINEAR_WEIGHT_DENOMINATOR
    ):
        raise RuntimeError("bilinear operator freeze changed")
    milp = _require_mapping(config.get("milp"), name="milp")
    solver_options = _require_mapping(
        milp.get("solver_options"),
        name="milp.solver_options",
    )
    expected_solver_options = {
        **dict(DEFAULT_SOLVER_OPTIONS),
        "time_limit": "none",
        "node_limit": "none",
    }
    exactness_search = _require_mapping(
        milp.get("exactness_search"),
        name="milp.exactness_search",
    )
    reconstruction = _require_mapping(
        milp.get("reconstruction_tolerances"),
        name="milp.reconstruction_tolerances",
    )
    if (
        milp.get("big_m") != BIG_M
        or milp.get("certifying_margin_min") != CERTIFYING_MARGIN_MIN
        or milp.get("gamma_search_bounds") != [0.0, GAMMA_MAX]
        or milp.get("margin_objective_scale") != MARGIN_OBJECTIVE_SCALE
        or dict(solver_options) != expected_solver_options
        or exactness_search.get("zero_margin_certificate")
        != (
            "reconstructed_gamma_and_scaled_primal_objective_and_scaled_"
            "dual_bound_must_all_equal_float64_zero_exactly"
        )
        or exactness_search.get("any_positive_margin_status")
        != "INCONCLUSIVE"
        or exactness_search.get("selected_witness")
        != "initial_valid_upper_only"
        or milp.get("symbolic_real_arithmetic_exactness_claim") is not False
        or set(reconstruction.values()) != {VERIFY_TOLERANCE}
    ):
        raise RuntimeError("MILP numerical or exactness-search freeze changed")
    prediction = _require_mapping(
        config.get("prediction_and_metric_reconstruction"),
        name="prediction_and_metric_reconstruction",
    )
    if (
        prediction.get("prediction_mask")
        != (
            "the_complete_reconstructed_256_by_256_output_logit_field_"
            "greater_than_or_equal_to_zero"
        )
        or prediction.get("retention_semantics")
        != "target_pixel_recall_not_stage_a_anchor_retention"
        or prediction.get("stage_a_anchor_retention_applicable") is not False
    ):
        raise RuntimeError("prediction reconstruction semantics changed")
    budgets = _require_mapping(
        config.get("prefrozen_case_budgets"),
        name="prefrozen_case_budgets",
    )
    if (
        budgets.get("retention") != RETENTION_REQUIRED
        or budgets.get("pixel_fa_max") != PIXEL_FA_MAX
        or budgets.get("raw_background_fa_max")
        != RAW_BACKGROUND_FA_MAX
        or budgets.get("fp_components_per_mp_max")
        != FP_COMPONENTS_PER_MP_MAX
        or budgets.get("maximum_false_addition_pixels")
        != MAX_FALSE_ADDITION_PIXELS
        or budgets.get("budget_violation") is not False
    ):
        raise RuntimeError("tiny-target case budgets changed")
    authorization = _require_mapping(
        config.get("authorization_semantics"),
        name="authorization_semantics",
    )
    for key in (
        "training_authorized",
        "D_V_or_D_T_evaluation_authorized",
        "calibration_or_inference_authorized",
        "full_cure_authorized",
        "backbone_integration_authorized",
    ):
        if authorization.get(key) is not False:
            raise RuntimeError(f"authorization_semantics.{key} changed")
    execution = _require_mapping(
        config.get("execution_policy"),
        name="execution_policy",
    )
    expected_execution = {
        "deterministic_process_workers",
        "dynamic_cpu_discovery",
        "process_start_method",
        "process_chunksize",
        "case_result_order",
        "parent_only_artifact_writer",
        "worker_recycling",
        "solver_threads_per_process",
        "solver_parallel",
        "solver_random_seed",
        "solver_output_flag",
    }
    workers = execution.get("deterministic_process_workers")
    if (
        set(execution) != expected_execution
        or isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers < 1
        or execution.get("dynamic_cpu_discovery") is not False
        or execution.get("process_start_method") != "spawn"
        or execution.get("process_chunksize") != 1
        or execution.get("case_result_order") != "catalog_order"
        or execution.get("parent_only_artifact_writer") is not True
        or execution.get("worker_recycling") is not False
        or execution.get("solver_threads_per_process") != 1
        or execution.get("solver_parallel") is not False
        or execution.get("solver_random_seed") != 0
        or execution.get("solver_output_flag") is not False
    ):
        raise RuntimeError(
            "deterministic process-execution policy changed"
        )
    _implementation_bindings(config)
    _validate_existing_bindings(config)
    _assert_no_forbidden_true(config, path="config")
    # Confirm the full payload remains JSON-only and finite.
    _json_bytes(config)
    return config


@dataclass(frozen=True)
class LoadedTinyTargetAuditConfig:
    """Validated executable config with a private strict-load provenance."""

    payload: Mapping[str, Any]
    config_fingerprint: str
    source_file_sha256: str
    source_path: Path
    implementation_bindings: Mapping[str, str]
    _token: object = field(repr=False, compare=False)


def load_tiny_target_audit_config(
    config_path: str | Path,
) -> LoadedTinyTargetAuditConfig:
    path = _canonical_input_file(config_path, name="tiny-target config")
    payload = _strict_json(
        path,
        name="tiny-target config",
        require_canonical_bytes=False,
    )
    validated = _validate_config_payload(payload)
    return LoadedTinyTargetAuditConfig(
        payload=validated,
        config_fingerprint=str(validated["config_fingerprint"]),
        source_file_sha256=file_sha256(path),
        source_path=path,
        implementation_bindings=_implementation_bindings(validated),
        _token=_STRICT_LOAD_TOKEN,
    )


def _require_loaded_config(
    config: LoadedTinyTargetAuditConfig,
) -> LoadedTinyTargetAuditConfig:
    if (
        not isinstance(config, LoadedTinyTargetAuditConfig)
        or config._token is not _STRICT_LOAD_TOKEN
    ):
        raise TypeError("config must come from the strict config loader")
    if (
        _validate_config_payload(config.payload) != dict(config.payload)
        or file_sha256(config.source_path) != config.source_file_sha256
        or _strict_json(
            config.source_path,
            name="tiny-target config",
            require_canonical_bytes=False,
        )
        != dict(config.payload)
    ):
        raise RuntimeError("strict-loaded tiny-target config changed")
    return config


def _non_authorization_payload() -> dict[str, bool]:
    return dict(_NON_AUTHORIZATION_BOUNDARY)


def _execution_parallelism(
    config: LoadedTinyTargetAuditConfig,
) -> dict[str, object]:
    execution = _require_mapping(
        config.payload.get("execution_policy"),
        name="execution_policy",
    )
    return {
        "deterministic_process_workers": execution[
            "deterministic_process_workers"
        ],
        "dynamic_cpu_discovery": False,
        "process_start_method": "spawn",
        "process_chunksize": 1,
        "case_result_order": "catalog_order",
        "parent_only_artifact_writer": True,
        "worker_recycling": False,
        "solver_threads_per_process": 1,
        "solver_parallel": False,
        "solver_random_seed": 0,
        "solver_output_flag": False,
    }


def _preflight_receipt(
    config: LoadedTinyTargetAuditConfig,
    catalog: TinyTargetCaseCatalog,
) -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": TINY_TARGET_PREFLIGHT_RECEIPT_SCHEMA,
            "execution_status": "PREFLIGHT_COMPLETE",
            "config_fingerprint": config.config_fingerprint,
            "source_config_file_sha256": config.source_file_sha256,
            "catalog_fingerprint": catalog.catalog_fingerprint,
            "shape_count": len(catalog.shapes),
            "equivalence_class_count": len(catalog.cases),
            "concrete_placement_count": catalog.concrete_placement_count,
            "solver_execution_performed": False,
            "case_certificates_constructed": False,
            "late_compliance_audit": True,
            **_non_authorization_payload(),
        },
        field_name="receipt_fingerprint",
    )


def _preflight_complete(
    config: LoadedTinyTargetAuditConfig,
    catalog: TinyTargetCaseCatalog,
    receipt: Mapping[str, object],
    artifact_files: Mapping[str, str],
) -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": TINY_TARGET_PREFLIGHT_COMPLETE_SCHEMA,
            "artifact_kind": "tiny_target_representability_preflight",
            "execution_status": "complete",
            "config_fingerprint": config.config_fingerprint,
            "source_config_file_sha256": config.source_file_sha256,
            "catalog_fingerprint": catalog.catalog_fingerprint,
            "preflight_receipt_fingerprint": receipt[
                "receipt_fingerprint"
            ],
            "artifact_files": dict(artifact_files),
            "artifact_file_count": len(artifact_files),
            "solver_execution_performed": False,
            "late_compliance_audit": True,
            **_non_authorization_payload(),
        },
        field_name="complete_fingerprint",
    )


@dataclass(frozen=True)
class PublishedTinyTargetPreflight:
    root: Path
    config_fingerprint: str
    source_config_file_sha256: str
    catalog: TinyTargetCaseCatalog
    preflight_receipt_fingerprint: str
    complete_fingerprint: str
    complete_file_sha256: str
    _token: object = field(repr=False, compare=False)

    def verify_unchanged(self) -> None:
        loaded = load_tiny_target_preflight(self.root)
        if loaded != self:
            raise RuntimeError("tiny-target preflight changed")


def publish_tiny_target_preflight(
    config: LoadedTinyTargetAuditConfig,
    output_dir: str | Path,
) -> PublishedTinyTargetPreflight:
    """Build and publish the exhaustive catalog without calling a solver."""

    config = _require_loaded_config(config)
    root = _begin_publication(
        output_dir,
        name="tiny-target preflight output",
    )
    catalog = build_tiny_target_case_catalog()
    # Detect source/config drift across catalog construction before any
    # completed receipt can be published.
    config = _require_loaded_config(config)
    receipt = _preflight_receipt(config, catalog)
    _write_new_json(root / "frozen_config.json", config.payload)
    _write_new_json(root / "case_catalog.json", catalog.payload())
    _write_new_json(root / "preflight_receipt.json", receipt)
    config = _require_loaded_config(config)
    artifact_files = _artifact_hashes(root)
    complete = _preflight_complete(
        config,
        catalog,
        receipt,
        artifact_files,
    )
    _finish_publication(root, complete)
    return load_tiny_target_preflight(root, config=config)


def _validate_flat_inventory(
    root: Path,
    *,
    expected: frozenset[str],
    name: str,
) -> None:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{name} root must be a regular directory")
    if (root / _INCOMPLETE).exists() or (root / _INCOMPLETE).is_symlink():
        raise RuntimeError(f"{name} publication is incomplete")
    entries = tuple(root.iterdir())
    if {entry.name for entry in entries} != set(expected):
        raise RuntimeError(f"{name} file inventory changed")
    if any(entry.is_symlink() or not entry.is_file() for entry in entries):
        raise RuntimeError(f"{name} inventory must contain regular files only")


def _expected_catalog(payload: Mapping[str, Any]) -> TinyTargetCaseCatalog:
    catalog = build_tiny_target_case_catalog()
    if dict(payload) != catalog.payload():
        raise RuntimeError("published case catalog differs from exact rebuild")
    return catalog


def load_tiny_target_preflight(
    output_dir: str | Path,
    *,
    config: LoadedTinyTargetAuditConfig | None = None,
) -> PublishedTinyTargetPreflight:
    root = _canonical_input_directory(
        output_dir,
        name="tiny-target preflight",
    )
    _validate_flat_inventory(
        root,
        expected=_PREFLIGHT_FILES,
        name="tiny-target preflight",
    )
    frozen = _strict_json(
        root / "frozen_config.json",
        name="preflight frozen config",
    )
    frozen = _validate_config_payload(frozen)
    catalog_payload = _strict_json(
        root / "case_catalog.json",
        name="preflight case catalog",
    )
    catalog = _expected_catalog(catalog_payload)
    receipt = _strict_json(
        root / "preflight_receipt.json",
        name="tiny-target preflight receipt",
    )
    complete = _strict_json(
        root / _COMPLETE,
        name="tiny-target preflight COMPLETE",
    )
    _verify_fingerprint(
        receipt,
        name="tiny-target preflight receipt",
        field_name="receipt_fingerprint",
    )
    _verify_fingerprint(
        complete,
        name="tiny-target preflight COMPLETE",
        field_name="complete_fingerprint",
    )
    _assert_no_forbidden_true(receipt, path="preflight_receipt")
    _assert_no_forbidden_true(complete, path="preflight_COMPLETE")
    if config is not None:
        config = _require_loaded_config(config)
        if (
            frozen != dict(config.payload)
            or receipt.get("source_config_file_sha256")
            != config.source_file_sha256
        ):
            raise RuntimeError("preflight does not bind the supplied config")
    config_fingerprint = frozen.get("config_fingerprint")
    source_sha = receipt.get("source_config_file_sha256")
    if not isinstance(config_fingerprint, str):
        raise RuntimeError("preflight config fingerprint is malformed")
    _require_sha256(source_sha, name="preflight source config SHA256")
    pseudo_config = LoadedTinyTargetAuditConfig(
        payload=frozen,
        config_fingerprint=config_fingerprint,
        source_file_sha256=str(source_sha),
        source_path=(
            config.source_path
            if config is not None
            else root / "frozen_config.json"
        ),
        implementation_bindings=_implementation_bindings(frozen),
        _token=_STRICT_LOAD_TOKEN,
    )
    expected_receipt = _preflight_receipt(pseudo_config, catalog)
    if receipt != expected_receipt:
        raise RuntimeError("preflight receipt does not reconstruct")
    hashes = _artifact_hashes(root)
    expected_complete = _preflight_complete(
        pseudo_config,
        catalog,
        receipt,
        hashes,
    )
    if complete != expected_complete:
        raise RuntimeError("preflight COMPLETE does not reconstruct")
    return PublishedTinyTargetPreflight(
        root=root,
        config_fingerprint=config_fingerprint,
        source_config_file_sha256=str(source_sha),
        catalog=catalog,
        preflight_receipt_fingerprint=str(
            receipt["receipt_fingerprint"]
        ),
        complete_fingerprint=str(complete["complete_fingerprint"]),
        complete_file_sha256=file_sha256(root / _COMPLETE),
        _token=_STRICT_LOAD_TOKEN,
    )


def _require_loaded_preflight(
    preflight: PublishedTinyTargetPreflight,
    config: LoadedTinyTargetAuditConfig,
) -> PublishedTinyTargetPreflight:
    if (
        not isinstance(preflight, PublishedTinyTargetPreflight)
        or preflight._token is not _STRICT_LOAD_TOKEN
    ):
        raise TypeError("preflight must come from the strict loader")
    current = load_tiny_target_preflight(preflight.root, config=config)
    if current != preflight:
        raise RuntimeError("strict-loaded tiny-target preflight changed")
    return preflight


def _preflight_binding(
    config: LoadedTinyTargetAuditConfig,
    preflight: PublishedTinyTargetPreflight,
) -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": TINY_TARGET_PREFLIGHT_BINDING_SCHEMA,
            "config_fingerprint": config.config_fingerprint,
            "catalog_fingerprint": preflight.catalog.catalog_fingerprint,
            "preflight_receipt_fingerprint": (
                preflight.preflight_receipt_fingerprint
            ),
            "preflight_complete_fingerprint": (
                preflight.complete_fingerprint
            ),
            "preflight_complete_file_sha256": (
                preflight.complete_file_sha256
            ),
            "strict_preflight_load_verified": True,
            "solver_execution_performed_by_preflight": False,
            **_non_authorization_payload(),
        },
        field_name="receipt_fingerprint",
    )


def _certificate_from_payload(
    payload: Mapping[str, Any],
) -> TinyTargetCaseCertificate:
    expected_keys = {
        "schema_version",
        "case_id",
        "case_status",
        "reason",
        "irreducible_false_addition_pixels",
        "localized_certifying_margin",
        "target_pixel_recall",
        "target_matched",
        "retention",
        "retention_semantics",
        "stage_a_anchor_retention_applicable",
        "pixel_fa",
        "raw_background_fa",
        "fp_components_per_mp",
        "active_value_hex",
        "positive_background_pixels",
        "budget_violations",
        "bound_normalization_max_abs",
        "dense_problem_fingerprint",
        "witness_fingerprint",
        "stage_1_solver",
        "stage_2_solver",
        "certificate_fingerprint",
    }
    if set(payload) != expected_keys:
        raise ValueError("certificate field inventory changed")
    if (
        payload.get("retention_semantics") != "target_pixel_recall"
        or payload.get("stage_a_anchor_retention_applicable") is not False
    ):
        raise ValueError("certificate retention semantics changed")
    active = payload.get("active_value_hex")
    positive = payload.get("positive_background_pixels")
    violations = payload.get("budget_violations")
    if not isinstance(active, list) or not all(
        isinstance(value, str) for value in active
    ):
        raise TypeError("certificate active values are malformed")
    if not isinstance(positive, list) or not all(
        isinstance(pixel, list)
        and len(pixel) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in pixel
        )
        for pixel in positive
    ):
        raise TypeError("certificate positive-background pixels are malformed")
    if not isinstance(violations, list) or not all(
        isinstance(value, str) for value in violations
    ):
        raise TypeError("certificate budget violations are malformed")
    stage_1 = _require_mapping(
        payload.get("stage_1_solver"),
        name="certificate.stage_1_solver",
    )
    stage_2 = _require_mapping(
        payload.get("stage_2_solver"),
        name="certificate.stage_2_solver",
    )
    return TinyTargetCaseCertificate(
        case_id=payload["case_id"],
        case_status=payload["case_status"],
        reason=payload["reason"],
        irreducible_false_addition_pixels=payload[
            "irreducible_false_addition_pixels"
        ],
        localized_certifying_margin=payload[
            "localized_certifying_margin"
        ],
        target_pixel_recall=payload["target_pixel_recall"],
        target_matched=payload["target_matched"],
        retention=payload["retention"],
        pixel_fa=payload["pixel_fa"],
        raw_background_fa=payload["raw_background_fa"],
        fp_components_per_mp=payload["fp_components_per_mp"],
        active_value_hex=tuple(active),
        positive_background_pixels=tuple(
            (int(pixel[0]), int(pixel[1])) for pixel in positive
        ),
        budget_violations=tuple(violations),
        bound_normalization_max_abs=payload[
            "bound_normalization_max_abs"
        ],
        dense_problem_fingerprint=payload[
            "dense_problem_fingerprint"
        ],
        witness_fingerprint=payload["witness_fingerprint"],
        stage_1_solver=dict(stage_1),
        stage_2_solver=dict(stage_2),
        certificate_fingerprint=payload["certificate_fingerprint"],
        schema_version=payload["schema_version"],
    )


def _certificate_set(
    catalog: TinyTargetCaseCatalog,
    certificates: Sequence[TinyTargetCaseCertificate],
) -> dict[str, object]:
    values = tuple(certificates)
    if tuple(certificate.case_id for certificate in values) != tuple(
        case.case_id for case in catalog.cases
    ):
        raise ValueError("certificate order differs from the case catalog")
    _build_tiny_target_decision_without_solver_replay(catalog, values)
    core: dict[str, object] = {
        "schema_version": TINY_TARGET_CERTIFICATE_SET_SCHEMA,
        "catalog_fingerprint": catalog.catalog_fingerprint,
        "certificate_count": len(values),
        "certificate_fingerprints": [
            certificate.certificate_fingerprint
            for certificate in values
        ],
        "certificates": [
            certificate.payload() for certificate in values
        ],
    }
    core["certificate_set_fingerprint"] = stable_fingerprint(core)
    return core


def _load_certificate_set(
    payload: Mapping[str, Any],
    catalog: TinyTargetCaseCatalog,
) -> tuple[tuple[TinyTargetCaseCertificate, ...], str]:
    _verify_fingerprint(
        payload,
        name="tiny-target certificate set",
        field_name="certificate_set_fingerprint",
    )
    if set(payload) != {
        "schema_version",
        "catalog_fingerprint",
        "certificate_count",
        "certificate_fingerprints",
        "certificates",
        "certificate_set_fingerprint",
    }:
        raise ValueError("certificate-set inventory changed")
    raw = payload.get("certificates")
    if not isinstance(raw, list):
        raise TypeError("certificate set must contain a list")
    values = tuple(
        _certificate_from_payload(
            _require_mapping(item, name=f"certificate[{index}]")
        )
        for index, item in enumerate(raw)
    )
    expected = _certificate_set(catalog, values)
    if dict(payload) != expected:
        raise RuntimeError("certificate set does not reconstruct")
    return values, str(expected["certificate_set_fingerprint"])


def _result(
    config: LoadedTinyTargetAuditConfig,
    catalog: TinyTargetCaseCatalog,
    certificate_set_fingerprint: str,
    decision: Mapping[str, object],
) -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": TINY_TARGET_RESULT_SCHEMA,
            "status": decision["status"],
            "config_fingerprint": config.config_fingerprint,
            "catalog_fingerprint": catalog.catalog_fingerprint,
            "certificate_set_fingerprint": certificate_set_fingerprint,
            "decision_fingerprint": decision["decision_fingerprint"],
            "equivalence_class_count": len(catalog.cases),
            "concrete_placement_count": catalog.concrete_placement_count,
            "case_status_counts": decision["case_status_counts"],
            "concrete_placement_status_counts": decision[
                "concrete_placement_status_counts"
            ],
            "all_case_conjunction": decision[
                "all_case_conjunction"
            ],
            "late_compliance_audit": True,
            **_non_authorization_payload(),
        },
        field_name="receipt_fingerprint",
    )


def _package_versions() -> dict[str, str]:
    values: dict[str, str] = {}
    for package in ("numpy", "scipy", "torch"):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = "not-installed"
    return values


def _audit_receipt(
    config: LoadedTinyTargetAuditConfig,
    preflight_binding: Mapping[str, object],
    catalog: TinyTargetCaseCatalog,
    certificate_set_fingerprint: str,
    result: Mapping[str, object],
    decision: Mapping[str, object],
    *,
    runtime_package_versions: Mapping[str, str] | None = None,
) -> dict[str, object]:
    versions = (
        _package_versions()
        if runtime_package_versions is None
        else dict(runtime_package_versions)
    )
    if set(versions) != {"numpy", "scipy", "torch"} or any(
        not isinstance(value, str) or not value for value in versions.values()
    ):
        raise ValueError("runtime package-version record is malformed")
    return _fingerprinted(
        {
            "schema_version": TINY_TARGET_AUDIT_RECEIPT_SCHEMA,
            "execution_status": "complete",
            "audit_kind": "late_static_output_grid_capacity_audit",
            "config_fingerprint": config.config_fingerprint,
            "implementation_bindings": dict(
                config.implementation_bindings
            ),
            "runtime_package_versions": versions,
            "execution_parallelism": _execution_parallelism(config),
            "preflight_binding_fingerprint": preflight_binding[
                "receipt_fingerprint"
            ],
            "catalog_fingerprint": catalog.catalog_fingerprint,
            "certificate_set_fingerprint": certificate_set_fingerprint,
            "result_fingerprint": result["receipt_fingerprint"],
            "decision_fingerprint": decision["decision_fingerprint"],
            "solver_execution_performed": True,
            "all_cases_processed_in_canonical_order": True,
            "runtime_dataset_access": False,
            "runtime_split_access": [],
            "checkpoint_access": False,
            "model_weight_access": False,
            "training": False,
            "calibration": False,
            "learned_model_inference": False,
            "decoder_architecture_change": False,
            "loss_change": False,
            "target_change": False,
            "target_dilation": False,
            "historical_artifact_overwrite": False,
            "late_compliance_audit": True,
            **_non_authorization_payload(),
        },
        field_name="receipt_fingerprint",
    )


def _execution_complete(
    config: LoadedTinyTargetAuditConfig,
    preflight_binding: Mapping[str, object],
    catalog: TinyTargetCaseCatalog,
    certificate_set_fingerprint: str,
    result: Mapping[str, object],
    decision: Mapping[str, object],
    audit_receipt: Mapping[str, object],
    artifact_files: Mapping[str, str],
) -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": TINY_TARGET_EXECUTION_COMPLETE_SCHEMA,
            "artifact_kind": "tiny_target_representability_execution",
            "execution_status": "complete",
            "status": decision["status"],
            "config_fingerprint": config.config_fingerprint,
            "source_config_file_sha256": config.source_file_sha256,
            "preflight_binding_fingerprint": preflight_binding[
                "receipt_fingerprint"
            ],
            "catalog_fingerprint": catalog.catalog_fingerprint,
            "certificate_set_fingerprint": certificate_set_fingerprint,
            "result_fingerprint": result["receipt_fingerprint"],
            "decision_fingerprint": decision["decision_fingerprint"],
            "audit_receipt_fingerprint": audit_receipt[
                "receipt_fingerprint"
            ],
            "artifact_files": dict(artifact_files),
            "artifact_file_count": len(artifact_files),
            "solver_execution_performed": True,
            "runtime_dataset_access": False,
            "runtime_split_access": [],
            "checkpoint_access": False,
            "model_weight_access": False,
            "training": False,
            "calibration": False,
            "learned_model_inference": False,
            "historical_artifact_overwrite": False,
            "late_compliance_audit": True,
            **_non_authorization_payload(),
        },
        field_name="complete_fingerprint",
    )


@dataclass(frozen=True)
class PublishedTinyTargetExecution:
    root: Path
    status: str
    config_fingerprint: str
    catalog: TinyTargetCaseCatalog
    certificates: tuple[TinyTargetCaseCertificate, ...]
    certificate_set_fingerprint: str
    decision_fingerprint: str
    complete_fingerprint: str
    _token: object = field(repr=False, compare=False)

    def verify_unchanged(self) -> None:
        loaded = load_tiny_target_execution(self.root)
        if loaded != self:
            raise RuntimeError("tiny-target execution artifact changed")


def _publish_execution_from_certificates(
    *,
    config: LoadedTinyTargetAuditConfig,
    preflight: PublishedTinyTargetPreflight,
    root: Path,
    certificates: Sequence[TinyTargetCaseCertificate],
) -> PublishedTinyTargetExecution:
    config = _require_loaded_config(config)
    preflight = _require_loaded_preflight(preflight, config)
    catalog = preflight.catalog
    certificate_set = _certificate_set(catalog, certificates)
    certificate_set_fingerprint = str(
        certificate_set["certificate_set_fingerprint"]
    )
    decision = _build_tiny_target_decision_without_solver_replay(
        catalog,
        certificates,
    )
    preflight_binding = _preflight_binding(config, preflight)
    result = _result(
        config,
        catalog,
        certificate_set_fingerprint,
        decision,
    )
    audit_receipt = _audit_receipt(
        config,
        preflight_binding,
        catalog,
        certificate_set_fingerprint,
        result,
        decision,
    )
    _write_new_json(root / "frozen_config.json", config.payload)
    _write_new_json(root / "preflight_binding.json", preflight_binding)
    _write_new_json(root / "case_catalog.json", catalog.payload())
    _write_new_json(root / "case_certificates.json", certificate_set)
    _write_new_json(root / "result.json", result)
    _write_new_json(root / "decision.json", decision)
    _write_new_json(root / "audit_receipt.json", audit_receipt)
    config = _require_loaded_config(config)
    preflight = _require_loaded_preflight(preflight, config)
    artifact_files = _artifact_hashes(root)
    complete = _execution_complete(
        config,
        preflight_binding,
        catalog,
        certificate_set_fingerprint,
        result,
        decision,
        audit_receipt,
        artifact_files,
    )
    _finish_publication(root, complete)
    return _load_tiny_target_execution_structure(root, config=config)


def execute_tiny_target_audit(
    config: LoadedTinyTargetAuditConfig,
    preflight: PublishedTinyTargetPreflight,
    output_dir: str | Path,
) -> PublishedTinyTargetExecution:
    """Solve all cases in preflight order and publish a complete execution."""

    config = _require_loaded_config(config)
    preflight = _require_loaded_preflight(preflight, config)
    root = _begin_publication(
        output_dir,
        name="tiny-target execution output",
    )
    workers = int(
        _execution_parallelism(config)["deterministic_process_workers"]
    )
    certificates: list[TinyTargetCaseCertificate] = []
    executor: ProcessPoolExecutor | None = None
    try:
        if workers == 1:
            solved = map(solve_tiny_target_case, preflight.catalog.cases)
        else:
            context = multiprocessing.get_context("spawn")
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
            )
            # Executor.map preserves input order.  No completion-order
            # collection or dynamic worker-count discovery is permitted.
            solved = executor.map(
                solve_tiny_target_case,
                preflight.catalog.cases,
                chunksize=1,
            )
        iterator = zip(preflight.catalog.cases, solved, strict=True)
        for case, certificate in iterator:
            if not isinstance(certificate, TinyTargetCaseCertificate):
                raise TypeError("solver returned a non-certificate value")
            if certificate.case_id != case.case_id:
                raise RuntimeError("solver certificate changed case identity")
            certificates.append(certificate)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    # A long all-case execution must not publish against implementation or
    # preflight files that changed while workers were running.
    config = _require_loaded_config(config)
    preflight = _require_loaded_preflight(preflight, config)
    return _publish_execution_from_certificates(
        config=config,
        preflight=preflight,
        root=root,
        certificates=certificates,
    )


def _load_tiny_target_execution_structure(
    output_dir: str | Path,
    *,
    config: LoadedTinyTargetAuditConfig | None = None,
) -> PublishedTinyTargetExecution:
    root = _canonical_input_directory(
        output_dir,
        name="tiny-target execution",
    )
    _validate_flat_inventory(
        root,
        expected=_EXECUTION_FILES,
        name="tiny-target execution",
    )
    frozen = _validate_config_payload(
        _strict_json(
            root / "frozen_config.json",
            name="execution frozen config",
        )
    )
    if config is not None:
        config = _require_loaded_config(config)
        if frozen != dict(config.payload):
            raise RuntimeError("execution does not bind the supplied config")
    source_sha = _strict_json(
        root / _COMPLETE,
        name="tiny-target execution COMPLETE",
    ).get("source_config_file_sha256")
    _require_sha256(source_sha, name="execution source config SHA256")
    if config is not None and source_sha != config.source_file_sha256:
        raise RuntimeError(
            "execution source-config SHA differs from supplied config"
        )
    pseudo_config = LoadedTinyTargetAuditConfig(
        payload=frozen,
        config_fingerprint=str(frozen["config_fingerprint"]),
        source_file_sha256=str(source_sha),
        source_path=(
            config.source_path
            if config is not None
            else root / "frozen_config.json"
        ),
        implementation_bindings=_implementation_bindings(frozen),
        _token=_STRICT_LOAD_TOKEN,
    )
    catalog = _expected_catalog(
        _strict_json(
            root / "case_catalog.json",
            name="execution case catalog",
        )
    )
    certificate_payload = _strict_json(
        root / "case_certificates.json",
        name="tiny-target case certificates",
    )
    certificates, certificate_set_fingerprint = _load_certificate_set(
        certificate_payload,
        catalog,
    )
    preflight_binding = _strict_json(
        root / "preflight_binding.json",
        name="tiny-target preflight binding",
    )
    result = _strict_json(
        root / "result.json",
        name="tiny-target result",
    )
    decision = _strict_json(
        root / "decision.json",
        name="tiny-target decision",
    )
    audit_receipt = _strict_json(
        root / "audit_receipt.json",
        name="tiny-target audit receipt",
    )
    complete = _strict_json(
        root / _COMPLETE,
        name="tiny-target execution COMPLETE",
    )
    for payload, name, field_name in (
        (
            preflight_binding,
            "tiny-target preflight binding",
            "receipt_fingerprint",
        ),
        (result, "tiny-target result", "receipt_fingerprint"),
        (
            audit_receipt,
            "tiny-target audit receipt",
            "receipt_fingerprint",
        ),
        (
            complete,
            "tiny-target execution COMPLETE",
            "complete_fingerprint",
        ),
    ):
        _verify_fingerprint(
            payload,
            name=name,
            field_name=field_name,
        )
        _assert_no_forbidden_true(payload, path=name)
    recomputed_decision = _build_tiny_target_decision_without_solver_replay(
        catalog,
        certificates,
    )
    if decision != recomputed_decision:
        raise RuntimeError("published all-case decision does not re-aggregate")
    _assert_no_forbidden_true(decision, path="tiny-target decision")
    expected_result = _result(
        pseudo_config,
        catalog,
        certificate_set_fingerprint,
        decision,
    )
    if result != expected_result:
        raise RuntimeError("tiny-target result does not reconstruct")
    # The preflight binding cannot prove that the original directory still
    # exists, but it must retain a complete hash/fingerprint identity and the
    # non-authorizing boundary.  CLI execution proves the live strict binding
    # before publication.
    if (
        set(preflight_binding)
        != {
            "schema_version",
            "config_fingerprint",
            "catalog_fingerprint",
            "preflight_receipt_fingerprint",
            "preflight_complete_fingerprint",
            "preflight_complete_file_sha256",
            "strict_preflight_load_verified",
            "solver_execution_performed_by_preflight",
            *_NON_AUTHORIZATION_BOUNDARY,
            "receipt_fingerprint",
        }
        or preflight_binding.get("schema_version")
        != TINY_TARGET_PREFLIGHT_BINDING_SCHEMA
        or preflight_binding.get("config_fingerprint")
        != pseudo_config.config_fingerprint
        or preflight_binding.get("catalog_fingerprint")
        != catalog.catalog_fingerprint
        or preflight_binding.get("strict_preflight_load_verified") is not True
        or preflight_binding.get("solver_execution_performed_by_preflight")
        is not False
    ):
        raise RuntimeError("execution preflight binding changed")
    for field_name in (
        "preflight_receipt_fingerprint",
        "preflight_complete_fingerprint",
        "preflight_complete_file_sha256",
    ):
        _require_sha256(
            preflight_binding.get(field_name),
            name=f"bound preflight {field_name}",
        )
    expected_audit = _audit_receipt(
        pseudo_config,
        preflight_binding,
        catalog,
        certificate_set_fingerprint,
        result,
        decision,
        runtime_package_versions=_require_mapping(
            audit_receipt.get("runtime_package_versions"),
            name="audit_receipt.runtime_package_versions",
        ),
    )
    if audit_receipt != expected_audit:
        raise RuntimeError("tiny-target audit receipt does not reconstruct")
    expected_complete = _execution_complete(
        pseudo_config,
        preflight_binding,
        catalog,
        certificate_set_fingerprint,
        result,
        decision,
        audit_receipt,
        _artifact_hashes(root),
    )
    if complete != expected_complete:
        raise RuntimeError("tiny-target execution COMPLETE does not reconstruct")
    return PublishedTinyTargetExecution(
        root=root,
        status=str(decision["status"]),
        config_fingerprint=pseudo_config.config_fingerprint,
        catalog=catalog,
        certificates=certificates,
        certificate_set_fingerprint=certificate_set_fingerprint,
        decision_fingerprint=str(decision["decision_fingerprint"]),
        complete_fingerprint=str(complete["complete_fingerprint"]),
        _token=_STRICT_LOAD_TOKEN,
    )


def _replay_certificate_set_with_bound_solver(
    catalog: TinyTargetCaseCatalog,
    certificates: Sequence[TinyTargetCaseCertificate],
    *,
    workers: int,
) -> None:
    """Independently solve all cases and require complete certificate equality."""

    executor: ProcessPoolExecutor | None = None
    try:
        if workers == 1:
            replayed = map(solve_tiny_target_case, catalog.cases)
        else:
            executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
            replayed = executor.map(
                solve_tiny_target_case,
                catalog.cases,
                chunksize=1,
            )
        for case, expected, actual in zip(
            catalog.cases,
            certificates,
            replayed,
            strict=True,
        ):
            if (
                actual.case_id != case.case_id
                or actual.payload() != expected.payload()
            ):
                raise RuntimeError(
                    "independent solver replay differs at case "
                    f"{case.case_id}"
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)


def load_tiny_target_execution(
    output_dir: str | Path,
    *,
    config: LoadedTinyTargetAuditConfig | None = None,
) -> PublishedTinyTargetExecution:
    """Strict-load an execution and independently replay every solver case."""

    loaded = _load_tiny_target_execution_structure(
        output_dir,
        config=config,
    )
    frozen = _validate_config_payload(
        _strict_json(
            loaded.root / "frozen_config.json",
            name="strict-load frozen config",
        )
    )
    workers = int(
        _require_mapping(
            frozen.get("execution_policy"),
            name="strict-load execution_policy",
        )["deterministic_process_workers"]
    )
    _replay_certificate_set_with_bound_solver(
        loaded.catalog,
        loaded.certificates,
        workers=workers,
    )
    current = _load_tiny_target_execution_structure(
        loaded.root,
        config=config,
    )
    if current != loaded:
        raise RuntimeError("execution changed during strict solver replay")
    return loaded


def compare_tiny_target_publications(
    first_dir: str | Path,
    second_dir: str | Path,
) -> dict[str, object]:
    """Strictly load and require byte identity for every published file."""

    first_root = _canonical_input_directory(
        first_dir,
        name="first tiny-target publication",
    )
    second_root = _canonical_input_directory(
        second_dir,
        name="second tiny-target publication",
    )
    if first_root == second_root:
        raise ValueError(
            "independent replay comparison requires two distinct roots"
        )
    first_complete = _strict_json(
        first_root / _COMPLETE,
        name="first tiny-target COMPLETE",
    )
    second_complete = _strict_json(
        second_root / _COMPLETE,
        name="second tiny-target COMPLETE",
    )
    schema = first_complete.get("schema_version")
    if schema != second_complete.get("schema_version"):
        raise RuntimeError("tiny-target publication kinds differ")
    if schema == TINY_TARGET_PREFLIGHT_COMPLETE_SCHEMA:
        first = load_tiny_target_preflight(first_root)
        second = load_tiny_target_preflight(second_root)
        artifact_kind = "preflight"
        first_fingerprint = first.complete_fingerprint
        second_fingerprint = second.complete_fingerprint
        solver_replay_verified = False
    elif schema == TINY_TARGET_EXECUTION_COMPLETE_SCHEMA:
        first = _load_tiny_target_execution_structure(first_root)
        second = _load_tiny_target_execution_structure(second_root)
        artifact_kind = "execution"
        first_fingerprint = first.complete_fingerprint
        second_fingerprint = second.complete_fingerprint
        solver_replay_verified = False
    else:
        raise RuntimeError("unsupported tiny-target publication kind")
    first_files = {
        path.relative_to(first_root).as_posix(): path
        for path in sorted(first_root.rglob("*"))
        if path.is_file()
    }
    second_files = {
        path.relative_to(second_root).as_posix(): path
        for path in sorted(second_root.rglob("*"))
        if path.is_file()
    }
    if set(first_files) != set(second_files):
        raise RuntimeError("tiny-target replay file inventories differ")
    hashes: dict[str, str] = {}
    for relative in sorted(first_files):
        first_bytes = first_files[relative].read_bytes()
        second_bytes = second_files[relative].read_bytes()
        if first_bytes != second_bytes:
            raise RuntimeError(
                f"tiny-target replay differs at {relative}"
            )
        hashes[relative] = _bytes_sha256(first_bytes)
    if first_fingerprint != second_fingerprint:
        raise RuntimeError("byte-identical publications changed fingerprint")
    if artifact_kind == "execution":
        frozen = _validate_config_payload(
            _strict_json(
                first_root / "frozen_config.json",
                name="comparison frozen config",
            )
        )
        workers = int(
            _require_mapping(
                frozen.get("execution_policy"),
                name="comparison execution_policy",
            )["deterministic_process_workers"]
        )
        _replay_certificate_set_with_bound_solver(
            first.catalog,
            first.certificates,
            workers=workers,
        )
        frozen_after = _validate_config_payload(
            _strict_json(
                first_root / "frozen_config.json",
                name="post-replay frozen config",
            )
        )
        if frozen_after != frozen:
            raise RuntimeError("configuration changed during solver replay")
        for relative, expected_hash in hashes.items():
            if (
                file_sha256(first_files[relative]) != expected_hash
                or file_sha256(second_files[relative]) != expected_hash
            ):
                raise RuntimeError(
                    "publication changed during solver replay at "
                    f"{relative}"
                )
        solver_replay_verified = True
    core: dict[str, object] = {
        "schema_version": TINY_TARGET_COMPARISON_SCHEMA,
        "artifact_kind": artifact_kind,
        "byte_identical": True,
        "file_count": len(hashes),
        "file_sha256": hashes,
        "complete_fingerprint": first_fingerprint,
        "independent_bound_solver_replay_verified": (
            solver_replay_verified
        ),
        **_non_authorization_payload(),
    }
    core["comparison_fingerprint"] = stable_fingerprint(core)
    return core


__all__ = [
    "LoadedTinyTargetAuditConfig",
    "PublishedTinyTargetExecution",
    "PublishedTinyTargetPreflight",
    "TINY_TARGET_CONFIG_SCHEMA",
    "compare_tiny_target_publications",
    "execute_tiny_target_audit",
    "load_tiny_target_audit_config",
    "load_tiny_target_execution",
    "load_tiny_target_preflight",
    "publish_tiny_target_preflight",
]
