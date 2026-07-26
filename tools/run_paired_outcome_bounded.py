#!/usr/bin/env python3
"""Run the frozen D_R-only OC-APTO v3 bounded model-code gate.

This command reconstructs the already frozen real D_R catalog, builds the
independent v3 factual-anchor population and the complete 206+16 outcome
population, then executes one fresh 400-update decoder run.  It is create-only:
there is no resume path.  It never reads D_V or D_T, performs calibration, or
reports detector performance.  A bounded pass is evidence for model-code
review only and does not directly authorize the 800-epoch experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.config import DecoderConfig, LossConfig  # noqa: E402
from cure_lite.experiment.paired_outcome_bounded import (  # noqa: E402
    COMPUTATIONAL_THRESHOLDS,
    PAIRED_OUTCOME_BOUNDED_SCHEMA,
    build_outcome_bounded_anchor_population,
    build_outcome_factual_anchor_schedule,
    execute_paired_outcome_bounded,
)
from cure_lite.experiment.paired_outcome_inputs import (  # noqa: E402
    build_paired_outcome_input_materializer,
)
from cure_lite.experiment.paired_outcome_schedule import (  # noqa: E402
    build_outcome_pair_schedule,
)
from tools import run_paired_bounded_learnability as legacy_runner  # noqa: E402
from tools import run_paired_preflight as pair_preflight_runner  # noqa: E402


CONFIG_REPO_PATH = (
    "protocols/IRSTD-1K/outcome_complete_apto_v3/bounded_config.json"
)
CONFIG_FILE_SHA256 = (
    "a0068e44db91c8ce20fd1790abddb5a0d2c56e3a878cd9464ffedfeee0bf7be6"
)
CONFIG_FINGERPRINT = (
    "5448478e121981725845873a348d1414cd5791b40b94258e5a0f99ab5105c8c6"
)
PROPOSAL_REPO_PATH = (
    "protocols/IRSTD-1K/outcome_complete_apto_v3/proposal_receipt.json"
)
PROPOSAL_FILE_SHA256 = (
    "dff143f2f1e5afb27c67695b687dd05bf201ae7a6bc41715998e4d78df1f3de5"
)
PROPOSAL_FINGERPRINT = (
    "870ebb721e50343e74f7a2a9c5dd719e466fc318ac5d24e5ac00a5ec145d5fc0"
)

RUN_SCHEMA = "cure-lite-oc-apto-v3-bounded-run-v1"
DECISION_SCHEMA = "cure-lite-oc-apto-v3-bounded-decision-v1"
FAILURE_SCHEMA = "cure-lite-oc-apto-v3-bounded-failure-v1"
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


def _prepare_output(path: Path) -> Path:
    candidate = path.expanduser()
    absolute = Path(os.path.abspath(candidate))
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"OC-APTO bounded output already exists: {absolute}")
    for parent in (absolute.parent, *absolute.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError(
                "OC-APTO bounded output may not traverse a symbolic link"
            )
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
    return pair_preflight_runner._strict_json(path, name=name)


def _expected_gate_payload() -> dict[str, object]:
    return {
        "all_222_pairs_bound": True,
        "pair_exposure_counts": [3, 4],
        **dict(COMPUTATIONAL_THRESHOLDS),
    }


def _validate_config_payload(config: Mapping[str, Any]) -> None:
    unsigned = dict(config)
    fingerprint = unsigned.pop("config_fingerprint", None)
    if (
        fingerprint != CONFIG_FINGERPRINT
        or stable_fingerprint(unsigned) != fingerprint
    ):
        raise RuntimeError("OC-APTO bounded config fingerprint is inconsistent")
    if (
        config.get("schema_version")
        != "cure-lite-oc-apto-v3-bounded-config-v1"
        or config.get("method_id") != "oc_apto_v3"
        or config.get("dataset") != "IRSTD-1K"
        or config.get("split") != "D_R"
        or config.get("config_fingerprint_scope")
        != "all_fields_except_config_fingerprint"
    ):
        raise RuntimeError("OC-APTO bounded config identity changed")
    if config.get("bounded_gates") != _expected_gate_payload():
        raise RuntimeError("OC-APTO bounded gates differ from the implementation")

    proposal = config.get("proposal_binding")
    source = config.get("source_reconstruction")
    anchors = config.get("anchor_population")
    outcomes = config.get("outcome_population")
    optimization = config.get("optimization")
    budget = config.get("budget")
    policy = config.get("execution_policy")
    semantics = config.get("decision_semantics")
    if not all(
        isinstance(value, Mapping)
        for value in (
            proposal,
            source,
            anchors,
            outcomes,
            optimization,
            budget,
            policy,
            semantics,
        )
    ):
        raise RuntimeError("OC-APTO bounded config sections are malformed")
    if (
        proposal.get("path") != PROPOSAL_REPO_PATH
        or proposal.get("file_sha256") != PROPOSAL_FILE_SHA256
        or proposal.get("proposal_fingerprint") != PROPOSAL_FINGERPRINT
    ):
        raise RuntimeError("OC-APTO proposal binding changed")
    if (
        source.get("loader_role") != "frozen_D_R_input_reconstruction_only"
        or source.get("source_config_path")
        != legacy_runner.BOUNDED_CONFIG_REPO_PATH
        or source.get("source_config_file_sha256")
        != legacy_runner.BOUNDED_CONFIG_FILE_SHA256
        or source.get("source_config_fingerprint")
        != legacy_runner.BOUNDED_CONFIG_FINGERPRINT
        or source.get("source_config_is_not_method_or_loss_authority") is not True
        or source.get("required_pair_catalog_fingerprint")
        != "4886e52d2cfb3392d0f4fdda376159d6e7f694fd449dc809cf8874793febde76"
    ):
        raise RuntimeError("D_R source reconstruction binding changed")
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
        raise RuntimeError("bounded anchor population changed")
    if (
        outcomes.get("clean_positive") != 206
        or outcomes.get("component_null") != 16
        or outcomes.get("union") != 222
        or outcomes.get("identity_null_optimizer_exposure") != 0
        or outcomes.get("sampling")
        != "pair_level_uniform_deterministic_over_outcome_union"
        or outcomes.get("source_disjoint_within_update") is not True
    ):
        raise RuntimeError("bounded outcome population changed")
    if (
        optimization.get("optimizer") != "adam"
        or optimization.get("seed") != 42
        or optimization.get("learning_rate") != 0.001
        or optimization.get("weight_decay") != 0.0
        or optimization.get("trainable_scope") != "CURELiteDecoder_only"
        or optimization.get("decoder")
        != {"feature_channels": 64, "width": 32, "groups": 8}
        or optimization.get("loss")
        != {"dice_weight": 1.0, "epsilon": 0.000001}
    ):
        raise RuntimeError("bounded optimization changed")
    expected_budget = {
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
    }
    if dict(budget) != expected_budget:
        raise RuntimeError("bounded execution budget changed")
    if (
        policy.get("create_only_output") is not True
        or policy.get("resume_allowed") is not False
        or policy.get("allowed_runtime_splits") != ["D_R"]
        or policy.get("D_V_access_allowed") is not False
        or policy.get("D_T_access_allowed") is not False
        or policy.get("base_or_backbone_update_allowed") is not False
        or policy.get("decoder_topology_change_allowed") is not False
        or policy.get("calibration_allowed") is not False
        or policy.get("performance_evaluation_allowed") is not False
        or policy.get("formal_800_training_allowed_by_this_config") is not False
        or policy.get("full_cure_allowed") is not False
        or policy.get("other_detector_integration_allowed") is not False
        or semantics.get("not_detection_performance_evidence") is not True
        or semantics.get("pass_alone_does_not_authorize_formal_800") is not True
    ):
        raise RuntimeError("bounded execution or decision boundary changed")


def _load_config(path: Path) -> dict[str, Any]:
    expected = _ROOT / CONFIG_REPO_PATH
    if path != expected:
        raise RuntimeError("OC-APTO bounded config path differs from the freeze")
    if file_sha256(path) != CONFIG_FILE_SHA256:
        raise RuntimeError("OC-APTO bounded config is not the exact frozen file")
    config = _strict_json(path, name="OC-APTO bounded config")
    _validate_config_payload(config)
    return config


def _validate_proposal_payload(
    proposal: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    scope = proposal.get("proposal_fingerprint_scope")
    if (
        not isinstance(scope, Mapping)
        or proposal.get("proposal_fingerprint") != PROPOSAL_FINGERPRINT
        or stable_fingerprint(scope) != proposal.get("proposal_fingerprint")
    ):
        raise RuntimeError("OC-APTO proposal fingerprint is inconsistent")
    source = config["source_reconstruction"]
    if (
        proposal.get("schema_version") != "cure-lite-oc-apto-v3-proposal-v1"
        or proposal.get("method_id") != "oc_apto_v3"
        or scope.get("method_id") != "oc_apto_v3"
        or scope.get("status") != "specified_not_implemented"
        or scope.get("pair_catalog_fingerprint")
        != source["required_pair_catalog_fingerprint"]
        or scope.get("population_counts")
        != {"clean_positive": 206, "component_null": 16, "union": 222}
    ):
        raise RuntimeError("OC-APTO proposal identity changed")
    proposal_scope = proposal.get("scope")
    population = proposal.get("population")
    budget = proposal.get("optimization_budget")
    gates = proposal.get("bounded_gates")
    if not all(
        isinstance(value, Mapping)
        for value in (proposal_scope, population, budget, gates)
    ):
        raise RuntimeError("OC-APTO proposal sections are malformed")
    if (
        proposal_scope.get("dataset") != "IRSTD-1K"
        or proposal_scope.get("split") != "D_R"
        or proposal_scope.get("D_V_access_allowed") is not False
        or proposal_scope.get("D_T_access_allowed") is not False
        or proposal_scope.get("base_or_backbone_update_allowed") is not False
        or proposal_scope.get("decoder_topology_change_allowed") is not False
        or proposal_scope.get("inference_change_allowed") is not False
        or proposal_scope.get("calibration_change_allowed") is not False
        or population.get("clean_positive") != 206
        or population.get("component_null") != 16
        or population.get("outcome_union") != 222
        or population.get("identity_null_optimizer_exposure") != 0
        or budget.get("bounded_updates") != 400
        or budget.get("outcome_pairs_per_update") != 2
        or budget.get("resume_allowed") is not False
        or gates.get("performance_evidence") is not False
        or any(
            gates.get(name) != value
            for name, value in config["bounded_gates"].items()
        )
    ):
        raise RuntimeError("OC-APTO proposal contract changed")


def _load_proposal(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    binding = config["proposal_binding"]
    proposal_path = _repo_file(binding["path"], name="OC-APTO proposal")
    if file_sha256(proposal_path) != binding["file_sha256"]:
        raise RuntimeError("OC-APTO proposal file SHA256 changed")
    proposal = _strict_json(proposal_path, name="OC-APTO proposal")
    _validate_proposal_payload(proposal, config)

    design = proposal.get("design_document")
    if not isinstance(design, Mapping):
        raise RuntimeError("OC-APTO design-document binding is malformed")
    design_path = _repo_file(design.get("path"), name="OC-APTO design document")
    if (
        file_sha256(design_path) != design.get("sha256")
        or design.get("sha256")
        != proposal["proposal_fingerprint_scope"]["document_sha256"]
    ):
        raise RuntimeError("OC-APTO design document changed")
    frozen = proposal.get("frozen_dependencies")
    if not isinstance(frozen, Mapping):
        raise RuntimeError("OC-APTO frozen dependencies are malformed")
    expected_files = {
        "decoder_sha256": _ROOT / "cure_lite" / "decoder.py",
        "model_sha256": _ROOT / "cure_lite" / "model.py",
        "paired_types_v1_sha256": _ROOT / "cure_lite" / "paired_types.py",
    }
    if any(
        file_sha256(path) != frozen.get(field)
        for field, path in expected_files.items()
    ):
        raise RuntimeError("a proposal-frozen v1 dependency changed")
    return proposal, proposal_path, design_path


def _load_source_config(config: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    binding = config["source_reconstruction"]
    path = _repo_file(
        binding["source_config_path"],
        name="frozen D_R source reconstruction config",
    )
    source = legacy_runner._load_config(path)
    if (
        file_sha256(path) != binding["source_config_file_sha256"]
        or source.get("config_fingerprint")
        != binding["source_config_fingerprint"]
        or source.get("split") != "D_R"
        or source["input_binding"]["real_pair_catalog_fingerprint"]
        != binding["required_pair_catalog_fingerprint"]
    ):
        raise RuntimeError("frozen D_R source reconstruction config changed")
    return source, path


def _implementation_binding() -> dict[str, object]:
    v3_paths = (
        _ROOT / "tools" / "run_paired_outcome_bounded.py",
        _ROOT / "cure_lite" / "paired_outcome_types.py",
        _ROOT / "cure_lite" / "paired_outcome_losses.py",
        _ROOT / "cure_lite" / "train" / "paired_outcome_step.py",
        _ROOT / "cure_lite" / "experiment" / "paired_outcome_inputs.py",
        _ROOT / "cure_lite" / "experiment" / "paired_outcome_schedule.py",
        _ROOT / "cure_lite" / "experiment" / "paired_outcome_bounded.py",
    )
    v3 = {
        path.relative_to(_ROOT).as_posix(): file_sha256(path)
        for path in v3_paths
    }
    source = legacy_runner._implementation_binding()
    merged = dict(source)
    merged.update(v3)
    return {
        "schema_version": "cure-lite-oc-apto-v3-implementation-binding-v1",
        "v3_runtime_files": dict(sorted(v3.items())),
        "source_reconstruction_dependency_files": dict(sorted(source.items())),
        "all_runtime_files": dict(sorted(merged.items())),
    }


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


def _verify_core_result(result: Mapping[str, Any]) -> None:
    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint", None)
    interpretation = result.get("interpretation")
    if (
        fingerprint is None
        or stable_fingerprint(unsigned) != fingerprint
        or result.get("schema_version") != PAIRED_OUTCOME_BOUNDED_SCHEMA
        or result.get("execution_status") != "completed"
        or not isinstance(interpretation, Mapping)
        or interpretation.get("not_detection_performance_evidence") is not True
        or interpretation.get("D_V_accessed") is not False
        or interpretation.get("D_T_accessed") is not False
        or interpretation.get("calibration_performed") is not False
        or interpretation.get("inference_performed") is not False
        or interpretation.get("base_or_backbone_updated") is not False
    ):
        raise RuntimeError("OC-APTO bounded result violates its frozen boundary")


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
        status = (
            "BOUNDED_MODEL_CODE_GATE_PASS"
            if model_code_pass
            else (
                "BOUNDED_MODEL_CODE_GATE_FAIL"
                if structural
                else "STRUCTURAL_EXECUTION_FAIL"
            )
        )
        if result.get("decision") != status:
            raise RuntimeError("core result and publication decision disagree")
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
                else "preserve_failure_and_revise_model_code_before_new_training"
            ),
        }
    )


@dataclass(frozen=True)
class PublishedOutcomeBounded:
    root: Path
    decision: str
    structural_execution_pass: bool
    bounded_model_code_gate_pass: bool
    pair_catalog_fingerprint: str
    complete_fingerprint: str

    def verify_unchanged(self) -> None:
        if load_paired_outcome_bounded_artifact(self.root) != self:
            raise RuntimeError("published OC-APTO bounded artifact changed")


def load_paired_outcome_bounded_artifact(
    output_dir: str | Path,
) -> PublishedOutcomeBounded:
    root = Path(output_dir).expanduser().resolve(strict=False)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("OC-APTO bounded root must be a regular directory")
    if (root / _INCOMPLETE).exists():
        raise RuntimeError("OC-APTO bounded publication is incomplete")
    if {item.name for item in root.iterdir()} != {"receipts", "COMPLETE.json"}:
        raise RuntimeError("OC-APTO bounded top-level inventory changed")
    receipts_root = root / "receipts"
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
        raise RuntimeError("OC-APTO bounded receipt inventory changed")
    complete = _strict_json(root / "COMPLETE.json", name="OC-APTO COMPLETE")
    _verify_fingerprinted(
        complete,
        name="OC-APTO COMPLETE",
        field="complete_fingerprint",
    )
    payloads = {
        name[:-5]: _strict_json(
            receipts_root / name,
            name=f"OC-APTO {name[:-5]}",
        )
        for name in names
    }
    for name, payload in payloads.items():
        _verify_fingerprinted(payload, name=f"OC-APTO {name}")
    evidence_kind = "result" if "result" in payloads else "failure"
    evidence = payloads[evidence_kind]
    decision = payloads["decision"]
    config_binding = payloads["config_binding"]
    proposal_binding = payloads["proposal_binding"]
    source_binding = payloads["source_reconstruction"]
    anchor = payloads["anchor_population"]
    factual = payloads["factual_schedule"]
    outcome_inputs = payloads["outcome_inputs"]
    outcome_schedule = payloads["outcome_schedule"]

    if (
        complete.get("artifact_files") != _artifact_hashes(root)
        or complete.get("artifact_file_count") != len(names)
        or complete.get("schema_version") != RUN_SCHEMA
        or complete.get("execution_status") != "complete"
        or complete.get("split") != "D_R"
        or complete.get("D_V_accessed") is not False
        or complete.get("D_T_accessed") is not False
        or complete.get("performance_evaluation_performed") is not False
        or complete.get("calibration_performed") is not False
        or complete.get("formal_800_training_performed") is not False
        or complete.get("directly_authorizes_formal_800") is not False
        or complete.get("resume_used") is not False
    ):
        raise RuntimeError("OC-APTO COMPLETE boundary or hashes changed")
    embedded_config = config_binding.get("config")
    embedded_proposal = proposal_binding.get("proposal")
    if not isinstance(embedded_config, Mapping) or not isinstance(
        embedded_proposal, Mapping
    ):
        raise RuntimeError("OC-APTO embedded config/proposal is malformed")
    _validate_config_payload(embedded_config)
    _validate_proposal_payload(embedded_proposal, embedded_config)
    if (
        config_binding.get("config_file_sha256") != CONFIG_FILE_SHA256
        or complete.get("config_fingerprint") != CONFIG_FINGERPRINT
        or proposal_binding.get("proposal_file_sha256") != PROPOSAL_FILE_SHA256
        or complete.get("proposal_fingerprint") != PROPOSAL_FINGERPRINT
        or source_binding.get("split") != "D_R"
        or source_binding.get("source_config_role")
        != "frozen_D_R_input_reconstruction_only"
        or source_binding.get("source_config_is_not_method_or_loss_authority")
        is not True
    ):
        raise RuntimeError("OC-APTO config/proposal/source binding changed")
    for payload, fingerprint_field, name in (
        (anchor, "population_fingerprint", "anchor population"),
        (factual, "schedule_fingerprint", "factual schedule"),
        (outcome_inputs, "materializer_fingerprint", "outcome inputs"),
        (outcome_schedule, "schedule_fingerprint", "outcome schedule"),
    ):
        unsigned = dict(payload)
        unsigned.pop("receipt_fingerprint", None)
        fingerprint = unsigned.pop(fingerprint_field, None)
        if fingerprint is None or stable_fingerprint(unsigned) != fingerprint:
            raise RuntimeError(f"OC-APTO {name} internal fingerprint changed")
    if evidence_kind == "result":
        result_unsigned = dict(evidence)
        result_unsigned.pop("receipt_fingerprint", None)
        _verify_core_result(result_unsigned)
    elif evidence.get("schema_version") != FAILURE_SCHEMA:
        raise RuntimeError("OC-APTO failure schema changed")
    if (
        decision.get("schema_version") != DECISION_SCHEMA
        or decision.get("evidence_kind") != evidence_kind
        or decision.get("evidence_receipt_fingerprint")
        != evidence.get("receipt_fingerprint")
        or decision.get("directly_authorizes_formal_800") is not False
        or decision.get("authorizes_D_V_or_D_T") is not False
        or complete.get("decision") != decision.get("status")
        or complete.get("decision_fingerprint")
        != decision.get("receipt_fingerprint")
        or complete.get("evidence_kind") != evidence_kind
        or complete.get("evidence_receipt_fingerprint")
        != evidence.get("receipt_fingerprint")
        or complete.get("anchor_population_fingerprint")
        != anchor.get("population_fingerprint")
        or complete.get("factual_schedule_fingerprint")
        != factual.get("schedule_fingerprint")
        or complete.get("outcome_schedule_fingerprint")
        != outcome_schedule.get("schedule_fingerprint")
        or complete.get("materializer_fingerprint")
        != outcome_inputs.get("materializer_fingerprint")
    ):
        raise RuntimeError("OC-APTO decision or evidence binding changed")
    return PublishedOutcomeBounded(
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
    config_path = _canonical_file(args.config, name="OC-APTO bounded config")
    config = _load_config(config_path)
    output = _prepare_output(args.output)
    proposal, proposal_path, design_path = _load_proposal(config)
    source_config, source_config_path = _load_source_config(config)
    implementation = _implementation_binding()
    pair_catalog, prepared, bundle, immutable = legacy_runner._load_real_catalog(
        source_config
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
        raise RuntimeError("reconstructed real outcome catalog changed")

    population = build_outcome_bounded_anchor_population(
        pair_catalog,
        prepared,
        _anchor_spec(config),
    )
    budget = config["budget"]
    factual_schedule = build_outcome_factual_anchor_schedule(
        population,
        optimizer_updates=budget["optimizer_updates"],
        steps_per_epoch=budget["steps_per_epoch"],
    )
    materializer = build_paired_outcome_input_materializer(
        pair_catalog,
        prepared,
    )
    outcome_schedule = build_outcome_pair_schedule(
        pair_catalog,
        seed=config["optimization"]["seed"],
        optimizer_updates=budget["optimizer_updates"],
        steps_per_epoch=budget["steps_per_epoch"],
    )
    if (
        materializer.pair_catalog_fingerprint != required_catalog
        or outcome_schedule.catalog_fingerprint != required_catalog
        or population.pair_catalog_fingerprint != required_catalog
    ):
        raise RuntimeError("OC-APTO v3 inputs do not bind one real catalog")

    immutable.update(
        {
            str(config_path): file_sha256(config_path),
            str(proposal_path): file_sha256(proposal_path),
            str(design_path): file_sha256(design_path),
            str(source_config_path): file_sha256(source_config_path),
        }
    )
    implementation_files = implementation["all_runtime_files"]
    if not isinstance(implementation_files, Mapping):
        raise RuntimeError("implementation file binding is malformed")

    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / _INCOMPLETE
    incomplete.open("xb").close()
    receipts = output / "receipts"
    receipts.mkdir(exist_ok=False)

    config_binding = _fingerprinted(
        {
            "schema_version": "cure-lite-oc-apto-v3-config-binding-v1",
            "config_repo_path": CONFIG_REPO_PATH,
            "config_file_sha256": file_sha256(config_path),
            "config_fingerprint": config["config_fingerprint"],
            "config": config,
        }
    )
    proposal_binding = _fingerprinted(
        {
            "schema_version": "cure-lite-oc-apto-v3-proposal-binding-v1",
            "proposal_repo_path": PROPOSAL_REPO_PATH,
            "proposal_file_sha256": file_sha256(proposal_path),
            "proposal_fingerprint": proposal["proposal_fingerprint"],
            "design_document_repo_path": proposal["design_document"]["path"],
            "design_document_file_sha256": file_sha256(design_path),
            "design_time_status_preserved": "specified_not_implemented",
            "proposal": proposal,
        }
    )
    source_binding = _fingerprinted(
        {
            "schema_version": "cure-lite-oc-apto-v3-source-reconstruction-v1",
            "split": "D_R",
            "source_config_role": "frozen_D_R_input_reconstruction_only",
            "source_config_is_not_method_or_loss_authority": True,
            "source_config_repo_path": config["source_reconstruction"][
                "source_config_path"
            ],
            "source_config_file_sha256": file_sha256(source_config_path),
            "source_config_fingerprint": source_config["config_fingerprint"],
            "pair_catalog_fingerprint": pair_catalog.catalog_fingerprint,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )
    implementation_receipt = _fingerprinted(implementation)
    anchor_receipt = _fingerprinted(population.canonical_receipt())
    factual_receipt = _fingerprinted(factual_schedule.canonical_receipt())
    materializer_receipt = _fingerprinted(materializer.canonical_receipt())
    outcome_schedule_receipt = _fingerprinted(
        outcome_schedule.canonical_receipt()
    )
    for name, payload in (
        ("config_binding.json", config_binding),
        ("proposal_binding.json", proposal_binding),
        ("source_reconstruction.json", source_binding),
        ("implementation_binding.json", implementation_receipt),
        ("anchor_population.json", anchor_receipt),
        ("factual_schedule.json", factual_receipt),
        ("outcome_inputs.json", materializer_receipt),
        ("outcome_schedule.json", outcome_schedule_receipt),
    ):
        _write_new_json(receipts / name, payload)

    result: dict[str, object] | None = None
    failure: dict[str, object] | None = None
    execution_error: Exception | None = None
    try:
        result = execute_paired_outcome_bounded(
            population,
            factual_schedule,
            outcome_schedule,
            materializer,
            DecoderConfig(**config["optimization"]["decoder"]),
            LossConfig(**config["optimization"]["loss"]),
            _optimization_budget(config),
            device=args.device,
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
            raise RuntimeError("OC-APTO implementation changed during execution")
    except Exception as error:
        if execution_error is None:
            execution_error = error

    if execution_error is None:
        try:
            if result is None:
                raise RuntimeError("OC-APTO execution returned no result")
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
    incomplete.unlink()
    published = load_paired_outcome_bounded_artifact(output)
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
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    if result["bounded_model_code_gate_pass"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
