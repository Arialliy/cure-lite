#!/usr/bin/env python3
"""Run the frozen D_R-only SVEF v4 bounded model-code gate.

The command reuses the frozen OC-APTO v3 D_R reconstruction, anchor
population, materializer, objective, and deterministic schedules.  It changes
only the decoder executed by ``execute_factorized_outcome_bounded``.  The
publication is create-only, has no resume path, never reads D_V or D_T, and
does not authorize formal training or detector-performance claims.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import isfinite
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.config import LossConfig  # noqa: E402
from cure_lite.experiment.factorized_outcome_bounded import (  # noqa: E402
    FACTORIZED_OUTCOME_BOUNDED_SCHEMA,
    execute_factorized_outcome_bounded,
    factorized_computational_gates,
)
from cure_lite.factorized_config import FactorizedDecoderConfig  # noqa: E402
from tools import run_paired_outcome_bounded as v3_runner  # noqa: E402


CONFIG_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "subpixel_vacancy_evidence_factorization_v4/bounded_config.json"
)
CONFIG_FILE_SHA256 = (
    "0462ecaa3cda30bbb8d8a57d1c4da396374607cdea6b2aa08cd43cc752ff3cc0"
)
CONFIG_FINGERPRINT = (
    "9637df9e8b84f61d914b38e4677374c8fd4cccbb17bb8afde5708daad518adde"
)
PROPOSAL_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "subpixel_vacancy_evidence_factorization_v4/proposal_receipt.json"
)
PROPOSAL_FILE_SHA256 = (
    "89a2dbc8fcbb0edfff8d33ce2e109a93cd24c3f7bec4a325d9ded2fe33e32115"
)
PROPOSAL_FINGERPRINT = (
    "d10ecd79f217ac692d965de1475529a1b9e0f6a38e8909db25560ec31bb28ccf"
)
PAIR_CATALOG_FINGERPRINT = (
    "4886e52d2cfb3392d0f4fdda376159d6e7f694fd449dc809cf8874793febde76"
)
PREPARED_CATALOG_FINGERPRINT = (
    "4955e5b4f1749b5f267db0ac1f031335a16cc48a470d6446ca6c99d04a5e85ed"
)
V3_IMPLEMENTATION_RECEIPT_FINGERPRINT = (
    "bc9f991d2a163691697e1abc21a6f1c180660dd6dea25fca5b650c123928b86e"
)
ANCHOR_POPULATION_FINGERPRINT = (
    "d251ed9061dd373aa0bf0e4ceeebbafc7ca32a4bab72c2f24601a20868d6d1cd"
)
MATERIALIZER_FINGERPRINT = (
    "8cc4eac43ad708265d8639c4b577b37bd81be8ccde73e79993ba18c65dca10ff"
)
FACTUAL_SCHEDULE_FINGERPRINT = (
    "57264042879d9850aa538e01563496a8d3de7b82556d2b5ef15ca7f32b66fac3"
)
OUTCOME_SCHEDULE_FINGERPRINT = (
    "747123867c88fd1444a514bf70e51013b739f39df2857e5ed021239e4847ec93"
)
OUTPUT_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/cure_lite_svef_v4_bounded_r1"
)
OUTPUT_VERSION_PREFIX = "cure_lite_svef_v4_bounded_"
FROZEN_DEVICE = "cuda:0"

RUN_SCHEMA = "cure-lite-svef-v4-bounded-run-v1"
DECISION_SCHEMA = "cure-lite-svef-v4-bounded-decision-v1"
FAILURE_SCHEMA = "cure-lite-svef-v4-bounded-failure-v1"
IMPLEMENTATION_SCHEMA = "cure-lite-svef-v4-implementation-binding-v1"
_ROOT = Path(__file__).resolve().parents[1]
_INCOMPLETE = ".incomplete"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _canonical_file(path: Path, *, name: str) -> Path:
    candidate = path.expanduser()
    absolute = Path(os.path.abspath(candidate))
    if candidate.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if resolved != absolute or not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{name} must be a canonical regular file")
    return resolved


def _repo_file(path_text: object, *, name: str) -> Path:
    if (
        not isinstance(path_text, str)
        or not path_text
        or Path(path_text).is_absolute()
    ):
        raise ValueError(f"{name} must be a non-empty repo-relative path")
    path = _canonical_file(_ROOT / path_text, name=name)
    if path.relative_to(_ROOT).as_posix() != path_text:
        raise RuntimeError(f"{name} does not resolve to its frozen path")
    return path


def _frozen_output_path() -> Path:
    return Path(os.path.abspath(_ROOT / OUTPUT_REPO_PATH))


def _validate_device(device: object) -> str:
    if device != FROZEN_DEVICE:
        raise ValueError("SVEF v4 bounded execution fixes --device at cuda:0")
    return FROZEN_DEVICE


def _prepare_output(path: Path) -> Path:
    candidate = path.expanduser()
    absolute = Path(os.path.abspath(candidate))
    expected = _frozen_output_path()
    if absolute != expected:
        raise ValueError(
            "SVEF v4 permits only its frozen r1 output path: "
            f"{expected}"
        )
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"SVEF bounded output already exists: {absolute}")
    run_root = expected.parent
    if run_root.exists():
        prior = tuple(
            sorted(
                entry
                for entry in run_root.iterdir()
                if entry.name.startswith(OUTPUT_VERSION_PREFIX)
            )
        )
        if prior:
            raise FileExistsError(
                "an SVEF v4 bounded run already exists: "
                + ", ".join(str(entry) for entry in prior)
            )
    for parent in (absolute.parent, *absolute.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError("SVEF bounded output may not traverse a symlink")
    return absolute


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _fingerprinted(
    payload: Mapping[str, object],
    *,
    field: str = "receipt_fingerprint",
) -> dict[str, object]:
    value = dict(payload)
    if field in value:
        raise ValueError(f"payload already contains {field}")
    value[field] = stable_fingerprint(value)
    return value


def _verify_fingerprinted(
    payload: Mapping[str, Any],
    *,
    name: str,
    field: str = "receipt_fingerprint",
) -> None:
    fingerprint = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if (
        not isinstance(fingerprint, str)
        or stable_fingerprint(unsigned) != fingerprint
    ):
        raise RuntimeError(f"{name} fingerprint is inconsistent")


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    return v3_runner._strict_json(path, name=name)


def _validate_config_payload(config: Mapping[str, Any]) -> None:
    unsigned = dict(config)
    fingerprint = unsigned.pop("config_fingerprint", None)
    if (
        fingerprint != CONFIG_FINGERPRINT
        or stable_fingerprint(unsigned) != fingerprint
    ):
        raise RuntimeError("SVEF bounded config fingerprint is inconsistent")
    if (
        config.get("schema_version")
        != "cure-lite-svef-v4-bounded-config-v1"
        or config.get("method_id") != "svef_v4"
        or config.get("dataset") != "IRSTD-1K"
        or config.get("split") != "D_R"
        or config.get("config_fingerprint_scope")
        != "all_fields_except_config_fingerprint"
    ):
        raise RuntimeError("SVEF bounded config identity changed")

    proposal = config.get("proposal_binding")
    source = config.get("source_reconstruction")
    anchors = config.get("anchor_population")
    outcomes = config.get("outcome_population")
    optimization = config.get("optimization")
    budget = config.get("budget")
    structural = config.get("structural_gates")
    bounded = config.get("bounded_gates")
    policy = config.get("execution_policy")
    semantics = config.get("decision_semantics")
    sections = (
        proposal,
        source,
        anchors,
        outcomes,
        optimization,
        budget,
        structural,
        bounded,
        policy,
        semantics,
    )
    if not all(isinstance(value, Mapping) for value in sections):
        raise RuntimeError("SVEF bounded config sections are malformed")
    if (
        proposal.get("path") != PROPOSAL_REPO_PATH
        or proposal.get("file_sha256") != PROPOSAL_FILE_SHA256
        or proposal.get("proposal_fingerprint") != PROPOSAL_FINGERPRINT
    ):
        raise RuntimeError("SVEF proposal binding changed")
    if (
        source.get("loader_role") != "frozen_D_R_input_reconstruction_only"
        or source.get("source_config_path")
        != v3_runner.legacy_runner.BOUNDED_CONFIG_REPO_PATH
        or source.get("source_config_file_sha256")
        != v3_runner.legacy_runner.BOUNDED_CONFIG_FILE_SHA256
        or source.get("source_config_fingerprint")
        != v3_runner.legacy_runner.BOUNDED_CONFIG_FINGERPRINT
        or source.get("source_config_is_not_method_or_loss_authority") is not True
        or source.get("required_pair_catalog_fingerprint")
        != PAIR_CATALOG_FINGERPRINT
    ):
        raise RuntimeError("SVEF D_R reconstruction binding changed")
    if {
        key: anchors.get(key)
        for key in (
            "seed",
            "factual_miss_anchors",
            "factual_no_miss_anchors",
            "identity_null_pairs",
        )
    } != {
        "seed": 42,
        "factual_miss_anchors": 16,
        "factual_no_miss_anchors": 16,
        "identity_null_pairs": 16,
    }:
        raise RuntimeError("SVEF anchor population changed")
    if (
        outcomes.get("clean_positive") != 206
        or outcomes.get("component_null") != 16
        or outcomes.get("union") != 222
        or outcomes.get("identity_null_optimizer_exposure") != 0
        or outcomes.get("sampling")
        != "pair_level_uniform_deterministic_over_outcome_union"
        or outcomes.get("source_disjoint_within_update") is not True
    ):
        raise RuntimeError("SVEF outcome population changed")

    decoder = optimization.get("decoder")
    loss = optimization.get("loss")
    if (
        not isinstance(decoder, Mapping)
        or not isinstance(loss, Mapping)
        or dict(decoder)
        != {
            "feature_channels": 64,
            "feature_stride": 4,
            "width": 32,
            "groups": 8,
            "trunk_residual_scale": 0.5,
            "baseline_probability": 0.1,
            "vacancy_kernel_size": 3,
            "evidence_transform": "zero_anchored_squared_softplus_v1",
            "resize_policy": "separate_fields_then_final_gate_v1",
        }
        or dict(loss) != {"dice_weight": 1.0, "epsilon": 0.000001}
        or optimization.get("optimizer") != "adam"
        or optimization.get("learning_rate") != 0.001
        or optimization.get("weight_decay") != 0.0
        or optimization.get("seed") != 42
        or optimization.get("trainable_scope")
        != "CURELiteFactorizedDecoder_only"
    ):
        raise RuntimeError("SVEF optimization contract changed")
    if dict(budget) != {
        "epochs": 10,
        "steps_per_epoch": 40,
        "optimizer_updates": 400,
        "factual_miss_states_per_update": 4,
        "factual_no_miss_states_per_update": 4,
        "outcome_pairs_per_update": 2,
        "outcome_endpoint_states_per_update": 4,
        "decoder_forward_calls_per_update": 3,
        "decoder_states_per_update": 12,
        "pair_slots": 800,
        "evaluation_chunk_size": 32,
        "resume_allowed": False,
    }:
        raise RuntimeError("SVEF bounded budget changed")
    if (
        structural.get("clean_full_D_reachable_pairs") != 206
        or structural.get("clean_D_reachable_pixels") != 2551
        or structural.get("factual_full_target_reachable_anchors") != 16
        or structural.get("factual_target_reachable_pixels") != 150
        or bounded.get(
            "clean_joint_D_ge_0_25_and_H_le_0_05_fraction_min"
        )
        != 0.75
        or bounded.get("tiny_target_strata_report_required") is not True
        or bounded.get("all_222_pairs_bound") is not True
        or bounded.get("pair_exposure_counts") != [3, 4]
    ):
        raise RuntimeError("SVEF structural or bounded gates changed")
    if (
        policy.get("create_only_output") is not True
        or policy.get("resume_allowed") is not False
        or policy.get("same_version_real_bounded_runs_max") != 1
        or policy.get("allowed_runtime_splits") != ["D_R"]
        or policy.get("D_V_access_allowed") is not False
        or policy.get("D_T_access_allowed") is not False
        or policy.get("base_or_backbone_update_allowed") is not False
        or policy.get("decoder_topology_change_allowed_during_execution")
        is not False
        or policy.get("loss_change_allowed") is not False
        or policy.get("calibration_allowed") is not False
        or policy.get("performance_evaluation_allowed") is not False
        or policy.get("formal_800_training_allowed_by_this_config") is not False
        or policy.get("full_cure_allowed") is not False
        or policy.get("other_detector_integration_allowed") is not False
        or semantics.get("not_detection_performance_evidence") is not True
        or semantics.get("directly_authorizes_formal_800") is not False
        or semantics.get(
            "pass_requires_separate_frozen_test_review_before_formal_800"
        )
        is not True
    ):
        raise RuntimeError("SVEF execution boundary changed")


def _load_config(path: Path) -> dict[str, Any]:
    expected = (_ROOT / CONFIG_REPO_PATH).resolve()
    if path != expected:
        raise RuntimeError("SVEF bounded config path differs from the freeze")
    if file_sha256(path) != CONFIG_FILE_SHA256:
        raise RuntimeError("SVEF bounded config is not the frozen file")
    config = _strict_json(path, name="SVEF bounded config")
    _validate_config_payload(config)
    return config


def _validate_proposal_payload(
    proposal: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    scope = proposal.get("proposal_fingerprint_scope")
    proposal_scope = proposal.get("scope")
    reused = proposal.get("reused_frozen_training")
    execution = proposal.get("execution_policy")
    if (
        not isinstance(scope, Mapping)
        or not isinstance(proposal_scope, Mapping)
        or not isinstance(reused, Mapping)
        or not isinstance(execution, Mapping)
        or proposal.get("proposal_fingerprint") != PROPOSAL_FINGERPRINT
        or stable_fingerprint(scope) != PROPOSAL_FINGERPRINT
        or proposal.get("schema_version")
        != "cure-lite-svef-v4-proposal-v1"
        or proposal.get("method_id") != "svef_v4"
        or scope.get("method_id") != "svef_v4"
        or scope.get("status") != "specified_not_implemented"
        or scope.get("pair_catalog_fingerprint")
        != PAIR_CATALOG_FINGERPRINT
        or proposal_scope.get("dataset") != "IRSTD-1K"
        or proposal_scope.get("split") != "D_R"
        or proposal_scope.get("D_V_access_allowed") is not False
        or proposal_scope.get("D_T_access_allowed") is not False
        or proposal_scope.get("loss_change_allowed") is not False
        or proposal_scope.get("paired_training_change_allowed") is not False
        or proposal_scope.get("formal_800_allowed_by_proposal") is not False
        or execution.get("same_version_real_bounded_runs_max") != 1
        or execution.get("automatic_retry_allowed") is not False
        or execution.get("D_V_access_allowed") is not False
        or execution.get("D_T_access_allowed") is not False
    ):
        raise RuntimeError("SVEF proposal contract changed")
    population = reused.get("outcome_population")
    if (
        not isinstance(population, Mapping)
        or population.get("clean_positive") != 206
        or population.get("component_null") != 16
        or population.get("union") != 222
        or reused.get("bounded_updates") != 400
        or reused.get("resume_allowed") is not False
        or config["proposal_binding"]["proposal_fingerprint"]
        != proposal["proposal_fingerprint"]
    ):
        raise RuntimeError("SVEF reused training contract changed")


def _load_proposal(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    binding = config["proposal_binding"]
    path = _repo_file(binding["path"], name="SVEF proposal")
    if file_sha256(path) != binding["file_sha256"]:
        raise RuntimeError("SVEF proposal file SHA256 changed")
    proposal = _strict_json(path, name="SVEF proposal")
    _validate_proposal_payload(proposal, config)
    design = proposal.get("design_document")
    if not isinstance(design, Mapping):
        raise RuntimeError("SVEF design-document binding is malformed")
    design_path = _repo_file(
        design.get("path"),
        name="SVEF design document",
    )
    if (
        file_sha256(design_path) != design.get("sha256")
        or design.get("sha256")
        != proposal["proposal_fingerprint_scope"]["document_sha256"]
    ):
        raise RuntimeError("SVEF design document changed")
    return proposal, path, design_path


def _load_source_config(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    binding = config["source_reconstruction"]
    path = _repo_file(
        binding["source_config_path"],
        name="frozen D_R source reconstruction config",
    )
    source = v3_runner.legacy_runner._load_config(path)
    if (
        file_sha256(path) != binding["source_config_file_sha256"]
        or source.get("config_fingerprint")
        != binding["source_config_fingerprint"]
        or source.get("split") != "D_R"
        or source["input_binding"]["real_pair_catalog_fingerprint"]
        != binding["required_pair_catalog_fingerprint"]
    ):
        raise RuntimeError("frozen D_R source reconstruction changed")
    return source, path


def _implementation_binding() -> dict[str, object]:
    v3 = v3_runner._implementation_binding()
    v3_all = v3.get("all_runtime_files")
    v3_fingerprint = stable_fingerprint(v3)
    if (
        not isinstance(v3_all, Mapping)
        or len(v3_all) != 40
        or v3_fingerprint != V3_IMPLEMENTATION_RECEIPT_FINGERPRINT
    ):
        raise RuntimeError("frozen v3 implementation binding changed")
    paths = (
        _ROOT / "cure_lite" / "factorized_config.py",
        _ROOT / "cure_lite" / "factorized_decoder.py",
        _ROOT / "cure_lite" / "factorized_model.py",
        _ROOT / "cure_lite" / "experiment" / "factorized_outcome_bounded.py",
        _ROOT / "tools" / "run_factorized_outcome_bounded.py",
    )
    v4 = {
        path.relative_to(_ROOT).as_posix(): file_sha256(path)
        for path in paths
    }
    merged = dict(v3_all)
    if set(merged) & set(v4):
        raise RuntimeError("v4 implementation must be additive")
    merged.update(v4)
    return {
        "schema_version": IMPLEMENTATION_SCHEMA,
        "v3_implementation_receipt_fingerprint": (
            v3_fingerprint
        ),
        "v3_runtime_files": dict(sorted(v3_all.items())),
        "v4_runtime_files": dict(sorted(v4.items())),
        "all_runtime_files": dict(sorted(merged.items())),
    }


def _verify_implementation_files(payload: Mapping[str, Any]) -> None:
    files = payload.get("all_runtime_files")
    v3_files = payload.get("v3_runtime_files")
    v4_files = payload.get("v4_runtime_files")
    current_v3 = v3_runner._implementation_binding()
    current_v3_files = current_v3.get("all_runtime_files")
    if (
        payload.get("schema_version") != IMPLEMENTATION_SCHEMA
        or payload.get("v3_implementation_receipt_fingerprint")
        != V3_IMPLEMENTATION_RECEIPT_FINGERPRINT
        or stable_fingerprint(current_v3)
        != V3_IMPLEMENTATION_RECEIPT_FINGERPRINT
        or not isinstance(current_v3_files, Mapping)
        or not isinstance(files, Mapping)
        or not isinstance(v3_files, Mapping)
        or not isinstance(v4_files, Mapping)
        or len(files) != 45
        or len(v3_files) != 40
        or len(v4_files) != 5
        or set(v3_files) & set(v4_files)
        or dict(files) != {**dict(v3_files), **dict(v4_files)}
        or dict(v3_files) != dict(current_v3_files)
    ):
        raise RuntimeError("SVEF implementation inventory changed")
    for path_text, expected in files.items():
        path = _repo_file(path_text, name="SVEF runtime file")
        if file_sha256(path) != expected:
            raise RuntimeError(
                f"SVEF runtime file hash changed: {path_text}"
            )


def _anchor_spec(config: Mapping[str, Any]) -> dict[str, object]:
    source = config["anchor_population"]
    return {
        key: source[key]
        for key in (
            "seed",
            "factual_miss_anchors",
            "factual_no_miss_anchors",
            "identity_null_pairs",
        )
    }


def _optimization_budget(config: Mapping[str, Any]) -> dict[str, object]:
    optimization = config["optimization"]
    budget = config["budget"]
    return {
        "seed": optimization["seed"],
        "optimizer_updates": budget["optimizer_updates"],
        "steps_per_epoch": budget["steps_per_epoch"],
        "factual_miss_states_per_update": budget[
            "factual_miss_states_per_update"
        ],
        "factual_no_miss_states_per_update": budget[
            "factual_no_miss_states_per_update"
        ],
        "outcome_pairs_per_update": budget["outcome_pairs_per_update"],
        "learning_rate": optimization["learning_rate"],
        "weight_decay": optimization["weight_decay"],
    }


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {_INCOMPLETE, "COMPLETE.json"}
    }


def _verify_internal_fingerprint(
    payload: Mapping[str, Any],
    *,
    field: str,
    name: str,
) -> None:
    unsigned = dict(payload)
    unsigned.pop("receipt_fingerprint", None)
    fingerprint = unsigned.pop(field, None)
    if (
        not isinstance(fingerprint, str)
        or stable_fingerprint(unsigned) != fingerprint
    ):
        raise RuntimeError(f"SVEF {name} internal fingerprint changed")


def _verify_structural_audit_payload(
    audit: Mapping[str, Any],
) -> bool:
    checks = audit.get("checks")
    budget = audit.get("compute_budget")
    records = audit.get("per_pair")
    if (
        audit.get("scope")
        != "pretraining_D_R_full_population_SVEF_structure"
        or audit.get("pair_count") != 222
        or audit.get("clean_pair_count") != 206
        or audit.get("component_null_pair_count") != 16
        or audit.get("clean_D_total_pixels") != 2551
        or audit.get("factual_target_total_pixels") != 150
        or not isinstance(checks, Mapping)
        or not isinstance(budget, Mapping)
        or not isinstance(records, list)
        or len(records) != 222
        or audit.get("training_performed") is not False
        or audit.get("D_V_accessed") is not False
        or audit.get("D_T_accessed") is not False
    ):
        raise RuntimeError("SVEF structural audit inventory changed")
    expected_budget = {
        "decoder_calls": 28,
        "decoder_state_evaluations": 888,
        "expected_decoder_calls": 28,
        "expected_decoder_state_evaluations": 888,
        "factual_vacancy_field_calls": 1,
        "factual_vacancy_field_states": 16,
    }
    if dict(budget) != expected_budget:
        raise RuntimeError("SVEF structural audit budget changed")

    pair_ids = [row.get("pair_id") for row in records]
    clean = [
        row for row in records if row.get("pair_kind") == "clean_positive"
    ]
    component = [
        row for row in records if row.get("pair_kind") == "component_null"
    ]
    if (
        len(set(pair_ids)) != 222
        or len(clean) != 206
        or len(component) != 16
        or sum(int(row.get("D_pixels", -1)) for row in clean) != 2551
    ):
        raise RuntimeError("SVEF structural audit pair evidence changed")

    expected_checks = {
        "zero_feature_occupancy_delta_exact_zero": (
            audit.get("zero_feature_max_abs_occupancy_delta") == 0.0
        ),
        "gate_support_outside_logit_delta_exact_zero": (
            audit.get("outside_gate_max_abs_logit_delta") == 0.0
        ),
        "gate_support_outside_probability_delta_exact_zero": (
            audit.get("outside_gate_max_abs_probability_delta") == 0.0
        ),
        "all_audited_fields_finite": (
            audit.get("nonfinite_audited_field_values") == 0
        ),
        "vacancy_deletion_monotonicity_exact": (
            audit.get("vacancy_deletion_monotonicity_violations") == 0
        ),
        "deletion_logit_monotonicity_exact": (
            audit.get("deletion_logit_monotonicity_violations") == 0
        ),
        "deletion_probability_monotonicity_exact": (
            audit.get("deletion_probability_monotonicity_violations") == 0
        ),
        "native_subpixel_path_without_resize": (
            audit.get("field_resize_endpoint_count") == 0
        ),
        "all_clean_D_pixels_structurally_reachable": (
            audit.get("clean_full_D_reachable_pairs") == 206
            and audit.get("clean_D_reachable_pixels")
            == audit.get("clean_D_total_pixels")
        ),
        "all_clean_pairs_have_nonempty_H": (
            audit.get("clean_nonempty_H_pairs") == 206
        ),
        "all_component_null_pairs_have_positive_gate_support": (
            audit.get("component_positive_gate_support_pairs") == 16
        ),
        "all_factual_targets_have_positive_vacancy": (
            audit.get("factual_full_target_reachable_anchors") == 16
            and audit.get("factual_target_reachable_pixels")
            == audit.get("factual_target_total_pixels")
        ),
        "structural_audit_decoder_budget_exact": (
            dict(budget) == expected_budget
        ),
    }
    if dict(checks) != expected_checks:
        raise RuntimeError("SVEF structural audit checks were not recomputable")
    all_pass = all(expected_checks.values())
    if audit.get("all_pass") is not all_pass:
        raise RuntimeError("SVEF structural audit all_pass is inconsistent")
    return all_pass


def _verify_computational_gate_payload(
    computational: Mapping[str, Any],
    *,
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
) -> bool:
    try:
        expected_from_snapshots = factorized_computational_gates(
            initial,
            final,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            "SVEF computational gates could not be recomputed from snapshots"
        ) from exc
    if dict(computational) != expected_from_snapshots:
        raise RuntimeError(
            "SVEF computational gates are not exactly bound to snapshots"
        )

    expected_thresholds = {
        "factual_miss_anchor_final_over_initial_max": 0.75,
        "factual_no_miss_anchor_final_over_initial_max": 0.75,
        "plus_baseline_final_over_initial_max": 0.75,
        "clean_transition_final_over_initial_max": 0.50,
        "clean_mean_delta_on_D_min": 0.50,
        "clean_pairs_delta_at_least_0_25_fraction_min": 0.75,
        "clean_zero_macro_mean_abs_delta_max": 0.05,
        "component_null_footprint_macro_mean_abs_delta_max": 0.05,
        "component_null_footprint_global_max_abs_delta_max": 0.25,
        "component_null_context_macro_mean_abs_delta_max": 0.05,
        "identity_null_max_abs_delta_max": 1.0e-7,
        "clean_joint_D_ge_0_25_and_H_le_0_05_fraction_min": 0.75,
    }
    rules = {
        "factual_miss_anchor_final_over_initial": (
            "max",
            "factual_miss_anchor_final_over_initial_max",
        ),
        "factual_no_miss_anchor_final_over_initial": (
            "max",
            "factual_no_miss_anchor_final_over_initial_max",
        ),
        "plus_baseline_final_over_initial": (
            "max",
            "plus_baseline_final_over_initial_max",
        ),
        "clean_transition_final_over_initial": (
            "max",
            "clean_transition_final_over_initial_max",
        ),
        "clean_mean_delta_on_D": ("min", "clean_mean_delta_on_D_min"),
        "clean_pairs_delta_at_least_0_25_fraction": (
            "min",
            "clean_pairs_delta_at_least_0_25_fraction_min",
        ),
        "clean_zero_macro_mean_abs_delta": (
            "max",
            "clean_zero_macro_mean_abs_delta_max",
        ),
        "component_null_footprint_macro_mean_abs_delta": (
            "max",
            "component_null_footprint_macro_mean_abs_delta_max",
        ),
        "component_null_footprint_global_max_abs_delta": (
            "max",
            "component_null_footprint_global_max_abs_delta_max",
        ),
        "component_null_context_macro_mean_abs_delta": (
            "max",
            "component_null_context_macro_mean_abs_delta_max",
        ),
        "identity_null_max_abs_delta": (
            "max",
            "identity_null_max_abs_delta_max",
        ),
        "clean_joint_D_ge_0_25_and_H_le_0_05_fraction": (
            "min",
            "clean_joint_D_ge_0_25_and_H_le_0_05_fraction_min",
        ),
    }
    observed = computational.get("observed")
    checks = computational.get("checks")
    tiny = computational.get("tiny_target_strata")
    if (
        computational.get("scope")
        != "bounded_D_R_full_outcome_SVEF_model_code_gate"
        or computational.get("not_detection_performance") is not True
        or computational.get("thresholds") != expected_thresholds
        or not isinstance(observed, Mapping)
        or set(observed) != set(rules)
        or not isinstance(checks, Mapping)
        or set(checks) != set(rules)
        or not isinstance(tiny, Mapping)
        or set(tiny) != {"1_to_3", "4_to_7", "8_to_15", "16_plus"}
    ):
        raise RuntimeError("SVEF computational gate inventory changed")

    recomputed: list[bool] = []
    for name, (direction, threshold_name) in rules.items():
        item = checks[name]
        value = observed[name]
        threshold = expected_thresholds[threshold_name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
            or not isinstance(item, Mapping)
            or item.get("value") != value
            or item.get("direction") != direction
            or item.get("threshold") != threshold
            or item.get("applicable") is not True
            or item.get("status") != "EVALUATED"
        ):
            raise RuntimeError(f"SVEF computational gate changed: {name}")
        passed = (
            float(value) >= threshold
            if direction == "min"
            else float(value) <= threshold
        )
        if item.get("pass") is not passed:
            raise RuntimeError(
                f"SVEF computational gate pass is inconsistent: {name}"
            )
        recomputed.append(passed)

    total_pairs = 0
    expected_bins = {
        "1_to_3": (1, 3),
        "4_to_7": (4, 7),
        "8_to_15": (8, 15),
        "16_plus": (16, None),
    }
    for name, (minimum, maximum) in expected_bins.items():
        row = tiny[name]
        if not isinstance(row, Mapping):
            raise RuntimeError(f"SVEF tiny-target stratum changed: {name}")
        count = row.get("pair_count")
        joint_count = row.get("joint_pass_count")
        if (
            row.get("D_pixel_min") != minimum
            or row.get("D_pixel_max") != maximum
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            or isinstance(joint_count, bool)
            or not isinstance(joint_count, int)
            or not 0 <= joint_count <= count
        ):
            raise RuntimeError(f"SVEF tiny-target stratum changed: {name}")
        total_pairs += count
        if count == 0:
            if any(
                row.get(field) is not None
                for field in (
                    "D_pair_macro_mean_delta",
                    "H_pair_macro_mean_abs_delta",
                    "joint_pass_fraction",
                )
            ):
                raise RuntimeError(
                    f"SVEF empty tiny-target stratum changed: {name}"
                )
        else:
            d_value = row.get("D_pair_macro_mean_delta")
            h_value = row.get("H_pair_macro_mean_abs_delta")
            fraction = row.get("joint_pass_fraction")
            if (
                not isinstance(d_value, (int, float))
                or not isfinite(float(d_value))
                or not isinstance(h_value, (int, float))
                or not isfinite(float(h_value))
                or fraction != joint_count / count
            ):
                raise RuntimeError(
                    f"SVEF tiny-target values changed: {name}"
                )
    if total_pairs != 206:
        raise RuntimeError("SVEF tiny-target strata do not cover 206 pairs")
    all_pass = all(recomputed)
    if computational.get("all_pass") is not all_pass:
        raise RuntimeError("SVEF computational all_pass is inconsistent")
    return all_pass


def _verify_core_result(result: Mapping[str, Any]) -> None:
    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint", None)
    interpretation = result.get("interpretation")
    structural_audit = result.get("pretraining_structural_audit")
    structural_checks = result.get("structural_checks")
    parameters = result.get("parameters")
    forward_budget = result.get("forward_budget")
    structural = result.get("structural_execution_pass")
    model_code_pass = result.get("computational_model_code_gate_pass")
    expected_decoder = {
        "feature_channels": 64,
        "feature_stride": 4,
        "width": 32,
        "groups": 8,
        "trunk_residual_scale": 0.5,
        "baseline_probability": 0.1,
        "vacancy_kernel_size": 3,
        "evidence_transform": "zero_anchored_squared_softplus_v1",
        "resize_policy": "separate_fields_then_final_gate_v1",
    }
    expected_budget = {
        "seed": 42,
        "optimizer_updates": 400,
        "steps_per_epoch": 40,
        "factual_miss_states_per_update": 4,
        "factual_no_miss_states_per_update": 4,
        "outcome_pairs_per_update": 2,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
    }
    if (
        not isinstance(fingerprint, str)
        or stable_fingerprint(unsigned) != fingerprint
        or result.get("schema_version") != FACTORIZED_OUTCOME_BOUNDED_SCHEMA
        or result.get("execution_status") != "completed"
        or result.get("device") != FROZEN_DEVICE
        or not isinstance(structural, bool)
        or not isinstance(model_code_pass, bool)
        or (model_code_pass and not structural)
        or result.get("population_fingerprint")
        != ANCHOR_POPULATION_FINGERPRINT
        or result.get("materializer_fingerprint")
        != MATERIALIZER_FINGERPRINT
        or result.get("factual_schedule_fingerprint")
        != FACTUAL_SCHEDULE_FINGERPRINT
        or result.get("outcome_schedule_fingerprint")
        != OUTCOME_SCHEDULE_FINGERPRINT
        or result.get("decoder_config") != expected_decoder
        or result.get("loss_config")
        != {"dice_weight": 1.0, "epsilon": 0.000001}
        or result.get("optimization_budget") != expected_budget
        or result.get("evaluation_chunk_size") != 32
        or not isinstance(structural_audit, Mapping)
        or not isinstance(structural_checks, Mapping)
        or not isinstance(parameters, Mapping)
        or parameters.get("trainable_parameter_count") != 4385
        or parameters.get("expected_parameter_count") != 4385
        or not isinstance(forward_budget, Mapping)
        or not isinstance(interpretation, Mapping)
        or interpretation.get("not_detection_performance_evidence") is not True
        or interpretation.get("does_not_authorize_formal_training") is not True
        or interpretation.get("does_not_directly_authorize_formal_800")
        is not True
        or interpretation.get("eligible_for_frozen_review")
        is not model_code_pass
        or interpretation.get("D_V_accessed") is not False
        or interpretation.get("D_T_accessed") is not False
        or interpretation.get("calibration_performed") is not False
        or interpretation.get("inference_performed") is not False
        or interpretation.get("base_or_backbone_updated") is not False
    ):
        raise RuntimeError("SVEF bounded result violates its frozen boundary")

    expected_decision = (
        "SVEF_BOUNDED_MODEL_CODE_GATE_PASS"
        if model_code_pass
        else (
            "SVEF_BOUNDED_MODEL_CODE_GATE_FAIL"
            if structural
            else "SVEF_STRUCTURAL_EXECUTION_FAIL"
        )
    )
    if result.get("decision") != expected_decision:
        raise RuntimeError("SVEF bounded core decision is inconsistent")

    audit_all_pass = _verify_structural_audit_payload(structural_audit)
    if not audit_all_pass:
        if (
            structural
            or model_code_pass
            or not any(value is False for value in structural_checks.values())
            or result.get("optimizer_updates_completed") != 0
            or result.get("training_performed") is not False
            or result.get("trace") != []
            or result.get("computational_gates")
            != {
                "status": "NOT_EVALUATED_BY_STRUCTURAL_STOP_RULE",
                "all_pass": None,
            }
            or forward_budget.get("training")
            != {"calls": 0, "state_evaluations": 0}
            or forward_budget.get("pretraining_structural_audit")
            != structural_audit.get("compute_budget")
        ):
            raise RuntimeError(
                "SVEF structural-stop result violates the frozen stop rule"
            )
        return

    expected_structural_checks = {
        "deterministic_runtime_contract_satisfied",
        "SVEF_pretraining_structural_audit_passed",
        "factual_anchor_and_identity_counts_exact",
        "all_222_outcome_pairs_bound",
        "all_222_outcome_pairs_evaluated_initial",
        "all_222_outcome_pairs_evaluated_final",
        "all_optimizer_updates_completed",
        "one_backward_per_update",
        "one_optimizer_step_per_update",
        "all_gradients_finite",
        "every_update_total_gradient_norm_positive",
        "decoder_parameters_changed",
        "training_forward_budget_exact",
        "evaluation_forward_budget_exact",
        "total_forward_budget_exact",
        "pair_exposure_ledger_exact",
        "source_exposure_ledger_exact",
        "factual_exposure_ledgers_exact",
        "identity_null_excluded_from_optimizer",
        "identity_null_diagnosed_without_autograd",
    }
    computational = result.get("computational_gates")
    execution_ledger = result.get("execution_ledger")
    exposure = result.get("exposure")
    deterministic = result.get("deterministic_runtime")
    gradients = result.get("gradients")
    trace = result.get("trace")
    initial = result.get("initial")
    final = result.get("final")
    if not isinstance(computational, Mapping):
        raise RuntimeError("SVEF full-run computational evidence is missing")
    computational_all_pass = _verify_computational_gate_payload(
        computational,
        initial=initial,
        final=final,
    )
    structural_from_checks = (
        set(structural_checks) == expected_structural_checks
        and all(isinstance(value, bool) for value in structural_checks.values())
        and all(structural_checks.values())
    )
    expected_snapshot = {"calls": 10, "state_evaluations": 508}
    expected_training = {"calls": 1200, "state_evaluations": 4800}
    expected_total = {"calls": 1220, "state_evaluations": 5816}
    if (
        set(structural_checks) != expected_structural_checks
        or not all(
            isinstance(value, bool) for value in structural_checks.values()
        )
        or structural is not structural_from_checks
        or model_code_pass is not (
            structural_from_checks and computational_all_pass
        )
        or result.get("optimizer_updates_completed") != 400
        or not isinstance(trace, list)
        or len(trace) != 400
        or not isinstance(initial, Mapping)
        or not isinstance(final, Mapping)
        or not isinstance(execution_ledger, Mapping)
        or not isinstance(exposure, Mapping)
        or not isinstance(gradients, Mapping)
        or not isinstance(deterministic, Mapping)
        or deterministic.get("contract_satisfied") is not True
        or deterministic.get("flags_restored_after_execution") is not True
        or not isinstance(exposure.get("outcome_pairs"), list)
        or len(exposure["outcome_pairs"]) != 222
        or exposure.get("identity_null_optimizer_exposure") != 0
        or forward_budget.get("pretraining_structural_audit")
        != structural_audit.get("compute_budget")
        or forward_budget.get("expected_initial_evaluation")
        != expected_snapshot
        or forward_budget.get("expected_training") != expected_training
        or forward_budget.get("expected_final_evaluation")
        != expected_snapshot
        or forward_budget.get("expected_total_excluding_structural_audit")
        != expected_total
    ):
        raise RuntimeError(
            "SVEF full-run structural evidence is inconsistent"
        )
    if structural and (
        execution_ledger.get("backward_calls") != 400
        or execution_ledger.get("optimizer_steps") != 400
        or gradients.get("nonfinite_updates") != 0
        or gradients.get("zero_norm_updates") != 0
        or forward_budget.get("initial_evaluation") != expected_snapshot
        or forward_budget.get("training") != expected_training
        or forward_budget.get("final_evaluation") != expected_snapshot
        or forward_budget.get("total_excluding_structural_audit")
        != expected_total
    ):
        raise RuntimeError(
            "SVEF structurally valid full run violates exact ledgers"
        )


def _decision(
    result: Mapping[str, object] | None,
    *,
    failure: Mapping[str, object] | None,
    evidence_receipt_fingerprint: str,
) -> dict[str, object]:
    if result is None:
        status = "STRUCTURAL_EXECUTION_ERROR"
        structural = False
        model_code_pass = False
    else:
        structural = result.get("structural_execution_pass") is True
        model_code_pass = (
            result.get("computational_model_code_gate_pass") is True
        )
        if model_code_pass and not structural:
            raise RuntimeError(
                "SVEF model-code pass requires structural execution pass"
            )
        status = (
            "BOUNDED_MODEL_CODE_GATE_PASS"
            if model_code_pass
            else (
                "BOUNDED_MODEL_CODE_GATE_FAIL"
                if structural
                else "STRUCTURAL_EXECUTION_FAIL"
            )
        )
        expected_core_status = {
            "BOUNDED_MODEL_CODE_GATE_PASS": (
                "SVEF_BOUNDED_MODEL_CODE_GATE_PASS"
            ),
            "BOUNDED_MODEL_CODE_GATE_FAIL": (
                "SVEF_BOUNDED_MODEL_CODE_GATE_FAIL"
            ),
            "STRUCTURAL_EXECUTION_FAIL": (
                "SVEF_STRUCTURAL_EXECUTION_FAIL"
            ),
        }[status]
        if result.get("decision") != expected_core_status:
            raise RuntimeError("SVEF core result and decision disagree")
    return _fingerprinted(
        {
            "schema_version": DECISION_SCHEMA,
            "status": status,
            "structural_execution_pass": structural,
            "bounded_model_code_gate_pass": model_code_pass,
            "not_detection_performance_evidence": True,
            "directly_authorizes_formal_800": False,
            "authorizes_D_V_or_D_T": False,
            "authorizes_full_cure": False,
            "authorizes_other_detector_integration": False,
            "evidence_kind": "result" if result is not None else "failure",
            "evidence_receipt_fingerprint": evidence_receipt_fingerprint,
            "failure": dict(failure) if failure is not None else None,
            "next_action": (
                "freeze_and_review_bounded_model_code_evidence"
                if model_code_pass
                else "preserve_failure_and_revise_model_code_before_training"
            ),
        }
    )


@dataclass(frozen=True)
class PublishedFactorizedOutcomeBounded:
    root: Path
    decision: str
    structural_execution_pass: bool
    bounded_model_code_gate_pass: bool
    pair_catalog_fingerprint: str
    complete_fingerprint: str

    def verify_unchanged(self) -> None:
        if load_factorized_outcome_bounded_artifact(self.root) != self:
            raise RuntimeError("published SVEF bounded artifact changed")


def load_factorized_outcome_bounded_artifact(
    output_dir: str | Path,
    *,
    _allow_incomplete: bool = False,
) -> PublishedFactorizedOutcomeBounded:
    candidate = Path(output_dir).expanduser()
    absolute = Path(os.path.abspath(candidate))
    if candidate.is_symlink():
        raise ValueError("SVEF bounded root may not be a symbolic link")
    root = candidate.resolve(strict=True)
    if root != absolute or not root.is_dir() or root.is_symlink():
        raise ValueError("SVEF bounded root must be a regular directory")
    incomplete_present = (root / _INCOMPLETE).exists()
    if incomplete_present and not _allow_incomplete:
        raise RuntimeError("SVEF bounded publication is incomplete")
    expected_top_level = {"receipts", "COMPLETE.json"}
    if _allow_incomplete:
        expected_top_level.add(_INCOMPLETE)
    if {item.name for item in root.iterdir()} != expected_top_level:
        raise RuntimeError("SVEF bounded top-level inventory changed")
    receipts_root = root / "receipts"
    if receipts_root.is_symlink() or not receipts_root.is_dir():
        raise RuntimeError("SVEF receipts must be a regular directory")
    common = {
        "anchor_population.json",
        "config_binding.json",
        "decision.json",
        "factual_schedule.json",
        "implementation_binding.json",
        "outcome_inputs.json",
        "outcome_schedule.json",
        "proposal_binding.json",
        "source_reconstruction.json",
    }
    names = {item.name for item in receipts_root.iterdir()}
    if names not in (common | {"result.json"}, common | {"failure.json"}):
        raise RuntimeError("SVEF bounded receipt inventory changed")
    if any(
        item.is_symlink() or not item.is_file()
        for item in receipts_root.iterdir()
    ):
        raise RuntimeError("SVEF receipts must be regular files")
    complete = _strict_json(root / "COMPLETE.json", name="SVEF COMPLETE")
    _verify_fingerprinted(
        complete,
        name="SVEF COMPLETE",
        field="complete_fingerprint",
    )
    payloads = {
        name[:-5]: _strict_json(
            receipts_root / name,
            name=f"SVEF {name[:-5]}",
        )
        for name in names
    }
    for name, payload in payloads.items():
        _verify_fingerprinted(payload, name=f"SVEF {name}")
    if (
        complete.get("artifact_files") != _artifact_hashes(root)
        or complete.get("artifact_file_count") != len(names)
        or complete.get("schema_version") != RUN_SCHEMA
        or complete.get("execution_status") != "complete"
        or complete.get("device") != FROZEN_DEVICE
        or complete.get("split") != "D_R"
        or complete.get("D_V_accessed") is not False
        or complete.get("D_T_accessed") is not False
        or complete.get("performance_evaluation_performed") is not False
        or complete.get("calibration_performed") is not False
        or complete.get("formal_800_training_performed") is not False
        or complete.get("directly_authorizes_formal_800") is not False
        or complete.get("not_detection_performance_evidence") is not True
        or complete.get("base_or_backbone_updated") is not False
        or complete.get("resume_used") is not False
    ):
        raise RuntimeError("SVEF COMPLETE boundary or hashes changed")

    config_binding = payloads["config_binding"]
    proposal_binding = payloads["proposal_binding"]
    source = payloads["source_reconstruction"]
    implementation = payloads["implementation_binding"]
    anchor = payloads["anchor_population"]
    factual = payloads["factual_schedule"]
    inputs = payloads["outcome_inputs"]
    schedule = payloads["outcome_schedule"]
    decision = payloads["decision"]
    evidence_kind = "result" if "result" in payloads else "failure"
    evidence = payloads[evidence_kind]
    embedded_config = config_binding.get("config")
    embedded_proposal = proposal_binding.get("proposal")
    if not isinstance(embedded_config, Mapping) or not isinstance(
        embedded_proposal, Mapping
    ):
        raise RuntimeError("SVEF embedded config/proposal is malformed")
    _validate_config_payload(embedded_config)
    _validate_proposal_payload(embedded_proposal, embedded_config)
    _verify_implementation_files(implementation)
    config_file = _repo_file(
        config_binding.get("config_repo_path"),
        name="published SVEF config",
    )
    proposal_file = _repo_file(
        proposal_binding.get("proposal_repo_path"),
        name="published SVEF proposal",
    )
    design_file = _repo_file(
        proposal_binding.get("design_document_repo_path"),
        name="published SVEF design document",
    )
    source_file = _repo_file(
        source.get("source_config_repo_path"),
        name="published SVEF source config",
    )
    if (
        config_binding.get("schema_version")
        != "cure-lite-svef-v4-config-binding-v1"
        or proposal_binding.get("schema_version")
        != "cure-lite-svef-v4-proposal-binding-v1"
        or source.get("schema_version")
        != "cure-lite-svef-v4-source-reconstruction-v1"
        or config_binding.get("config_file_sha256") != CONFIG_FILE_SHA256
        or file_sha256(config_file) != CONFIG_FILE_SHA256
        or complete.get("config_fingerprint") != CONFIG_FINGERPRINT
        or proposal_binding.get("proposal_file_sha256")
        != PROPOSAL_FILE_SHA256
        or file_sha256(proposal_file) != PROPOSAL_FILE_SHA256
        or file_sha256(design_file)
        != proposal_binding.get("design_document_file_sha256")
        or file_sha256(source_file)
        != source.get("source_config_file_sha256")
        or complete.get("proposal_fingerprint") != PROPOSAL_FINGERPRINT
        or source.get("split") != "D_R"
        or source.get("source_config_role")
        != "frozen_D_R_input_reconstruction_only"
        or source.get("source_config_is_not_method_or_loss_authority")
        is not True
        or source.get("pair_catalog_fingerprint")
        != PAIR_CATALOG_FINGERPRINT
        or source.get("prepared_catalog_fingerprint")
        != PREPARED_CATALOG_FINGERPRINT
        or source.get("D_V_accessed") is not False
        or source.get("D_T_accessed") is not False
    ):
        raise RuntimeError("SVEF config/proposal/source binding changed")
    for payload, field, name in (
        (anchor, "population_fingerprint", "anchor population"),
        (factual, "schedule_fingerprint", "factual schedule"),
        (inputs, "materializer_fingerprint", "outcome inputs"),
        (schedule, "schedule_fingerprint", "outcome schedule"),
    ):
        _verify_internal_fingerprint(payload, field=field, name=name)
    if evidence_kind == "result":
        result_unsigned = dict(evidence)
        result_unsigned.pop("receipt_fingerprint", None)
        _verify_core_result(result_unsigned)
        core_structural = (
            result_unsigned.get("structural_execution_pass") is True
        )
        core_model_pass = (
            result_unsigned.get("computational_model_code_gate_pass") is True
        )
        expected_decision = (
            "BOUNDED_MODEL_CODE_GATE_PASS"
            if core_model_pass
            else (
                "BOUNDED_MODEL_CODE_GATE_FAIL"
                if core_structural
                else "STRUCTURAL_EXECUTION_FAIL"
            )
        )
        if (
            decision.get("status") != expected_decision
            or decision.get("structural_execution_pass")
            is not core_structural
            or decision.get("bounded_model_code_gate_pass")
            is not core_model_pass
            or decision.get("failure") is not None
            or result_unsigned.get("population_fingerprint")
            != anchor.get("population_fingerprint")
            or result_unsigned.get("factual_schedule_fingerprint")
            != factual.get("schedule_fingerprint")
            or result_unsigned.get("outcome_schedule_fingerprint")
            != schedule.get("schedule_fingerprint")
            or result_unsigned.get("materializer_fingerprint")
            != inputs.get("materializer_fingerprint")
        ):
            raise RuntimeError(
                "SVEF core result and publication decision disagree"
            )
    else:
        failure_unsigned = dict(evidence)
        failure_unsigned.pop("receipt_fingerprint", None)
        if (
            evidence.get("schema_version") != FAILURE_SCHEMA
            or not isinstance(evidence.get("exception_type"), str)
            or not evidence.get("exception_type")
            or not isinstance(evidence.get("message"), str)
            or evidence.get("structural_execution_pass") is not False
            or evidence.get("bounded_model_code_gate_pass") is not False
            or evidence.get("budget_or_threshold_changed") is not False
            or evidence.get("D_V_accessed") is not False
            or evidence.get("D_T_accessed") is not False
            or decision.get("status") != "STRUCTURAL_EXECUTION_ERROR"
            or decision.get("structural_execution_pass") is not False
            or decision.get("bounded_model_code_gate_pass") is not False
            or decision.get("failure") != failure_unsigned
        ):
            raise RuntimeError(
                "SVEF execution failure decision is inconsistent"
            )
    expected_next_action = (
        "freeze_and_review_bounded_model_code_evidence"
        if decision.get("bounded_model_code_gate_pass") is True
        else "preserve_failure_and_revise_model_code_before_training"
    )
    if (
        decision.get("schema_version") != DECISION_SCHEMA
        or decision.get("evidence_kind") != evidence_kind
        or decision.get("evidence_receipt_fingerprint")
        != evidence.get("receipt_fingerprint")
        or decision.get("directly_authorizes_formal_800") is not False
        or decision.get("not_detection_performance_evidence") is not True
        or decision.get("authorizes_D_V_or_D_T") is not False
        or decision.get("authorizes_full_cure") is not False
        or decision.get("authorizes_other_detector_integration") is not False
        or decision.get("next_action") != expected_next_action
        or complete.get("decision") != decision.get("status")
        or complete.get("structural_execution_pass")
        is not decision.get("structural_execution_pass")
        or complete.get("bounded_model_code_gate_pass")
        is not decision.get("bounded_model_code_gate_pass")
        or complete.get("decision_fingerprint")
        != decision.get("receipt_fingerprint")
        or complete.get("evidence_kind") != evidence_kind
        or complete.get("evidence_receipt_fingerprint")
        != evidence.get("receipt_fingerprint")
        or complete.get("anchor_population_fingerprint")
        != anchor.get("population_fingerprint")
        or complete.get("anchor_population_fingerprint")
        != ANCHOR_POPULATION_FINGERPRINT
        or complete.get("factual_schedule_fingerprint")
        != factual.get("schedule_fingerprint")
        or complete.get("factual_schedule_fingerprint")
        != FACTUAL_SCHEDULE_FINGERPRINT
        or complete.get("outcome_schedule_fingerprint")
        != schedule.get("schedule_fingerprint")
        or complete.get("outcome_schedule_fingerprint")
        != OUTCOME_SCHEDULE_FINGERPRINT
        or complete.get("materializer_fingerprint")
        != inputs.get("materializer_fingerprint")
        or complete.get("materializer_fingerprint")
        != MATERIALIZER_FINGERPRINT
        or complete.get("implementation_receipt_fingerprint")
        != implementation.get("receipt_fingerprint")
        or complete.get("pair_catalog_fingerprint")
        != PAIR_CATALOG_FINGERPRINT
        or complete.get("prepared_catalog_fingerprint")
        != PREPARED_CATALOG_FINGERPRINT
        or anchor.get("prepared_catalog_fingerprint")
        != PREPARED_CATALOG_FINGERPRINT
        or inputs.get("prepared_catalog_fingerprint")
        != PREPARED_CATALOG_FINGERPRINT
    ):
        raise RuntimeError("SVEF decision or evidence binding changed")
    return PublishedFactorizedOutcomeBounded(
        root=root,
        decision=str(decision["status"]),
        structural_execution_pass=bool(
            decision["structural_execution_pass"]
        ),
        bounded_model_code_gate_pass=bool(
            decision["bounded_model_code_gate_pass"]
        ),
        pair_catalog_fingerprint=str(
            complete["pair_catalog_fingerprint"]
        ),
        complete_fingerprint=str(complete["complete_fingerprint"]),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    config_path = _canonical_file(args.config, name="SVEF bounded config")
    config = _load_config(config_path)
    device = _validate_device(args.device)
    output = _prepare_output(args.output)
    proposal, proposal_path, design_path = _load_proposal(config)
    source_config, source_config_path = _load_source_config(config)
    implementation = _implementation_binding()
    _verify_implementation_files(implementation)
    pair_catalog, prepared, bundle, immutable = (
        v3_runner.legacy_runner._load_real_catalog(source_config)
    )
    required_catalog = config["source_reconstruction"][
        "required_pair_catalog_fingerprint"
    ]
    if (
        pair_catalog.catalog_fingerprint != required_catalog
        or pair_catalog.split != "D_R"
        or len(pair_catalog.clean_positive) != 206
        or len(pair_catalog.component_null) != 16
    ):
        raise RuntimeError("reconstructed real SVEF outcome catalog changed")

    population = v3_runner.build_outcome_bounded_anchor_population(
        pair_catalog,
        prepared,
        _anchor_spec(config),
    )
    budget = config["budget"]
    factual_schedule = v3_runner.build_outcome_factual_anchor_schedule(
        population,
        optimizer_updates=budget["optimizer_updates"],
        steps_per_epoch=budget["steps_per_epoch"],
    )
    materializer = v3_runner.build_paired_outcome_input_materializer(
        pair_catalog,
        prepared,
    )
    outcome_schedule = v3_runner.build_outcome_pair_schedule(
        pair_catalog,
        seed=config["optimization"]["seed"],
        optimizer_updates=budget["optimizer_updates"],
        steps_per_epoch=budget["steps_per_epoch"],
    )
    if (
        materializer.pair_catalog_fingerprint != required_catalog
        or outcome_schedule.catalog_fingerprint != required_catalog
        or population.pair_catalog_fingerprint != required_catalog
        or population.prepared_catalog_fingerprint
        != PREPARED_CATALOG_FINGERPRINT
        or materializer.prepared_catalog_fingerprint
        != PREPARED_CATALOG_FINGERPRINT
        or population.population_fingerprint
        != ANCHOR_POPULATION_FINGERPRINT
        or materializer.materializer_fingerprint
        != MATERIALIZER_FINGERPRINT
        or factual_schedule.schedule_fingerprint
        != FACTUAL_SCHEDULE_FINGERPRINT
        or outcome_schedule.schedule_fingerprint
        != OUTCOME_SCHEDULE_FINGERPRINT
    ):
        raise RuntimeError(
            "SVEF inputs do not bind the frozen real D_R catalogs"
        )

    immutable.update(
        {
            str(config_path): file_sha256(config_path),
            str(proposal_path): file_sha256(proposal_path),
            str(design_path): file_sha256(design_path),
            str(source_config_path): file_sha256(source_config_path),
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / _INCOMPLETE
    incomplete.open("xb").close()
    receipts = output / "receipts"
    receipts.mkdir(exist_ok=False)

    config_receipt = _fingerprinted(
        {
            "schema_version": "cure-lite-svef-v4-config-binding-v1",
            "config_repo_path": CONFIG_REPO_PATH,
            "config_file_sha256": file_sha256(config_path),
            "config_fingerprint": config["config_fingerprint"],
            "config": config,
        }
    )
    proposal_receipt = _fingerprinted(
        {
            "schema_version": "cure-lite-svef-v4-proposal-binding-v1",
            "proposal_repo_path": PROPOSAL_REPO_PATH,
            "proposal_file_sha256": file_sha256(proposal_path),
            "proposal_fingerprint": proposal["proposal_fingerprint"],
            "design_document_repo_path": proposal["design_document"]["path"],
            "design_document_file_sha256": file_sha256(design_path),
            "design_time_status_preserved": "specified_not_implemented",
            "proposal": proposal,
        }
    )
    source_receipt = _fingerprinted(
        {
            "schema_version": "cure-lite-svef-v4-source-reconstruction-v1",
            "split": "D_R",
            "source_config_role": "frozen_D_R_input_reconstruction_only",
            "source_config_is_not_method_or_loss_authority": True,
            "source_config_repo_path": config["source_reconstruction"][
                "source_config_path"
            ],
            "source_config_file_sha256": file_sha256(source_config_path),
            "source_config_fingerprint": source_config["config_fingerprint"],
            "pair_catalog_fingerprint": pair_catalog.catalog_fingerprint,
            "prepared_catalog_fingerprint": (
                population.prepared_catalog_fingerprint
            ),
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )
    implementation_receipt = _fingerprinted(implementation)
    anchor_receipt = _fingerprinted(population.canonical_receipt())
    factual_receipt = _fingerprinted(factual_schedule.canonical_receipt())
    inputs_receipt = _fingerprinted(materializer.canonical_receipt())
    schedule_receipt = _fingerprinted(outcome_schedule.canonical_receipt())
    for name, payload in (
        ("config_binding.json", config_receipt),
        ("proposal_binding.json", proposal_receipt),
        ("source_reconstruction.json", source_receipt),
        ("implementation_binding.json", implementation_receipt),
        ("anchor_population.json", anchor_receipt),
        ("factual_schedule.json", factual_receipt),
        ("outcome_inputs.json", inputs_receipt),
        ("outcome_schedule.json", schedule_receipt),
    ):
        _write_new_json(receipts / name, payload)

    result: dict[str, object] | None = None
    failure: dict[str, object] | None = None
    execution_error: Exception | None = None
    try:
        result = execute_factorized_outcome_bounded(
            population,
            factual_schedule,
            outcome_schedule,
            materializer,
            FactorizedDecoderConfig(**config["optimization"]["decoder"]),
            LossConfig(**config["optimization"]["loss"]),
            _optimization_budget(config),
            device=device,
            evaluation_chunk_size=budget["evaluation_chunk_size"],
        )
    except Exception as error:
        execution_error = error

    try:
        bundle.verify_unchanged()
        if any(
            file_sha256(Path(path)) != digest
            for path, digest in immutable.items()
        ):
            raise RuntimeError("a frozen D_R input changed during execution")
        if _implementation_binding() != implementation:
            raise RuntimeError("SVEF implementation changed during execution")
    except Exception as error:
        if execution_error is None:
            execution_error = error

    evidence_receipt: dict[str, object]
    if execution_error is None:
        try:
            if result is None:
                raise RuntimeError("SVEF execution returned no result")
            _verify_core_result(result)
            evidence_receipt = _fingerprinted(result)
            json.dumps(evidence_receipt, allow_nan=False)
        except Exception as error:
            execution_error = error
    if execution_error is None:
        _write_new_json(receipts / "result.json", evidence_receipt)
    else:
        result = None
        failure = {
            "schema_version": FAILURE_SCHEMA,
            "exception_type": type(execution_error).__name__,
            "message": str(execution_error),
            "structural_execution_pass": False,
            "bounded_model_code_gate_pass": False,
            "budget_or_threshold_changed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
        evidence_receipt = _fingerprinted(failure)
        _write_new_json(receipts / "failure.json", evidence_receipt)
    decision = _decision(
        result,
        failure=failure,
        evidence_receipt_fingerprint=str(
            evidence_receipt["receipt_fingerprint"]
        ),
    )
    _write_new_json(receipts / "decision.json", decision)

    artifact_files = _artifact_hashes(output)
    complete = _fingerprinted(
        {
            "schema_version": RUN_SCHEMA,
            "execution_status": "complete",
            "decision": decision["status"],
            "structural_execution_pass": decision[
                "structural_execution_pass"
            ],
            "bounded_model_code_gate_pass": decision[
                "bounded_model_code_gate_pass"
            ],
            "not_detection_performance_evidence": True,
            "directly_authorizes_formal_800": False,
            "device": device,
            "split": "D_R",
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_evaluation_performed": False,
            "calibration_performed": False,
            "formal_800_training_performed": False,
            "base_or_backbone_updated": False,
            "resume_used": False,
            "config_fingerprint": config["config_fingerprint"],
            "proposal_fingerprint": proposal["proposal_fingerprint"],
            "pair_catalog_fingerprint": pair_catalog.catalog_fingerprint,
            "prepared_catalog_fingerprint": (
                population.prepared_catalog_fingerprint
            ),
            "anchor_population_fingerprint": population.population_fingerprint,
            "factual_schedule_fingerprint": (
                factual_schedule.schedule_fingerprint
            ),
            "outcome_schedule_fingerprint": (
                outcome_schedule.schedule_fingerprint
            ),
            "materializer_fingerprint": materializer.materializer_fingerprint,
            "implementation_receipt_fingerprint": implementation_receipt[
                "receipt_fingerprint"
            ],
            "evidence_kind": decision["evidence_kind"],
            "evidence_receipt_fingerprint": evidence_receipt[
                "receipt_fingerprint"
            ],
            "decision_fingerprint": decision["receipt_fingerprint"],
            "artifact_files": artifact_files,
            "artifact_file_count": len(artifact_files),
        },
        field="complete_fingerprint",
    )
    _write_new_json(output / "COMPLETE.json", complete)
    published = load_factorized_outcome_bounded_artifact(
        output,
        _allow_incomplete=True,
    )
    incomplete.unlink()
    return {
        "output": str(output),
        "decision": published.decision,
        "structural_execution_pass": published.structural_execution_pass,
        "bounded_model_code_gate_pass": (
            published.bounded_model_code_gate_pass
        ),
        "complete_fingerprint": published.complete_fingerprint,
        "not_detection_performance_evidence": True,
        "directly_authorizes_formal_800": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if result["bounded_model_code_gate_pass"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
