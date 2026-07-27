#!/usr/bin/env python3
"""Read-only terminal audit for the CMIF/PMOPE-v18 bounded protocol.

The auditor has no dependency on PyTorch or on the ``cure_lite`` package.  It
never deserializes a checkpoint, evaluates a model, trains, or reads dataset
tensors.  It verifies only regular-file bytes, strict JSON, canonical
fingerprints, the frozen implementation/source closure, and the terminal
claim graph.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile
from typing import Any, Iterable, Mapping, Sequence


_ROOT = Path(__file__).resolve().parents[1]

RUN_ID = "cure_lite_cmif_v18_pmope_bounded_400_r1"
RUN_REPO_PATH = f"runs/irstd1k_stage_a_seed42/{RUN_ID}"
DEFAULT_RUN_PATH = _ROOT / RUN_REPO_PATH

RUN_SCHEMA = "cure-lite-cmif-v18-pmope-bounded-400-run-v1"
ATTEMPT_SCHEMA = "cure-lite-cmif-v18-pmope-bounded-400-attempt-v1"
FAILURE_SCHEMA = "cure-lite-cmif-v18-pmope-bounded-400-failure-v1"
CHECKPOINT_SCHEMA = "cure-lite-cmif-v18-pmope-bounded-400-checkpoint-v1"
DECISION_SCHEMA = "cure-lite-cmif-v18-pmope-bounded-400-decision-v1"
AUDIT_SCHEMA = "cure-lite-cmif-v18-pmope-terminal-audit-v1"

V17_RUN_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/"
    "cure_lite_cmif_v17_support_oriented_bounded_400_r1"
)
V17_SOURCE_MANIFEST_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_cmif_v17_support_oriented_bounded_400_50a9963ae620.json"
)
V17_SOURCE_ARCHIVE_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_cmif_v17_support_oriented_bounded_400_50a9963ae620.tar"
)

EXPECTED_IMPLEMENTATION_PATHS = (
    "cure_lite/cache/schema.py",
    "cure_lite/coverage_state_batches.py",
    "cure_lite/coverage_state_device_cache.py",
    "cure_lite/coverage_state_level_set.py",
    "cure_lite/coverage_state_observability.py",
    "cure_lite/coverage_state_precomputed_cache.py",
    "cure_lite/coverage_state_raw_catalog.py",
    "cure_lite/coverage_state_schedule.py",
    "cure_lite/coverage_state_sobolev.py",
    "cure_lite/data.py",
    "cure_lite/frozen_base.py",
    "cure_lite/intervention.py",
    "cure_lite/instances.py",
    "cure_lite/matching.py",
    "cure_lite/paired_types.py",
    "cure_lite/splits.py",
    "cure_lite/types.py",
    "cure_lite/train/coverage_state_fused_step.py",
    "cure_lite/experiment/cache_pipeline.py",
    "cure_lite/experiment/coverage_state_bounded_protocol.py",
    "cure_lite/experiment/coverage_state_bounded_runner.py",
    "cure_lite/experiment/coverage_state_dataset_free.py",
    "cure_lite/experiment/coverage_state_observability_protocol.py",
    "cure_lite/experiment/coverage_state_raw_catalog.py",
    "cure_lite/experiment/coverage_state_real_dr_inputs.py",
    "cure_lite/experiment/coverage_state_training.py",
    "cure_lite/experiment/coverage_state_zero_level_evaluation.py",
    "cure_lite/experiment/geometry_catalog_protocol.py",
    "cure_lite/experiment/geometry_safe_catalog.py",
    "cure_lite/experiment/training_pipeline.py",
    "cure_lite/coverage_state_phase_preserving.py",
    "cure_lite/coverage_state_centered_mixed_interaction.py",
    "cure_lite/experiment/coverage_state_pmope_dataset_free.py",
    "cure_lite/experiment/coverage_state_pmope_dr_gate.py",
    "cure_lite/experiment/coverage_state_pmope_sealed_v17.py",
    "cure_lite/experiment/coverage_state_pmope_bounded_runner.py",
    "tools/audit_coverage_state_cmif_pmope_v18.py",
    "tools/run_coverage_state_cmif_pmope_bounded_400.py",
    "tools/run_coverage_state_cslf_ppce_support_oriented_bounded_400.py",
    "tools/run_coverage_state_cslf_support_oriented_bounded_400.py",
    "tools/run_with_gpu_temperature_control.py",
)

EXPECTED_BOUNDED_ARTIFACT_PATHS = (
    "attempt.json",
    "checkpoints/pmope_joint.checkpoint.json",
    "checkpoints/pmope_joint.safetensors",
    "receipts/authorization.json",
    "receipts/bounded_result.json",
    "receipts/config.json",
    "receipts/dataset_free.json",
    "receipts/decision.json",
    "receipts/device_memory_preflight.json",
    "receipts/dr_gate.json",
    "receipts/inputs.json",
    "receipts/preflight.json",
    "receipts/sealed_v17_controls.json",
    "receipts/training.json",
    "receipts/zero_level.json",
)

EXPECTED_GATE_STOP_ARTIFACT_PATHS = (
    "attempt.json",
    "receipts/config.json",
    "receipts/dataset_free.json",
    "receipts/decision.json",
    "receipts/dr_gate.json",
    "receipts/inputs.json",
    "receipts/preflight.json",
    "receipts/sealed_v17_controls.json",
)

_RECEIPT_SCHEMA_BY_PATH = {
    "attempt.json": ATTEMPT_SCHEMA,
    "checkpoints/pmope_joint.checkpoint.json": CHECKPOINT_SCHEMA,
    "receipts/authorization.json": (
        "cure-lite-cmif-v18-pmope-bounded-400-authorization-v1"
    ),
    "receipts/bounded_result.json": (
        "cure-lite-cmif-v18-pmope-bounded-400-result-v1"
    ),
    "receipts/config.json": RUN_SCHEMA,
    "receipts/dataset_free.json": (
        "cure-lite-cmif-v18-pmope-bounded-400-dataset-free-v1"
    ),
    "receipts/decision.json": DECISION_SCHEMA,
    "receipts/device_memory_preflight.json": (
        "cure-lite-cmif-v18-pmope-bounded-400-"
        "device-memory-preflight-v1"
    ),
    "receipts/dr_gate.json": (
        "cure-lite-cmif-v18-pmope-bounded-400-real-D_R-gate-v1"
    ),
    "receipts/inputs.json": (
        "cure-lite-cmif-v18-pmope-bounded-400-inputs-v1"
    ),
    "receipts/preflight.json": (
        "cure-lite-cmif-v18-pmope-bounded-400-preflight-v1"
    ),
    "receipts/sealed_v17_controls.json": (
        "cure-lite-cmif-v18-pmope-bounded-400-sealed-v17-controls-v1"
    ),
    "receipts/training.json": (
        "cure-lite-cmif-v18-pmope-bounded-400-training-v1"
    ),
    "receipts/zero_level.json": (
        "cure-lite-cmif-v18-pmope-bounded-400-zero-level-v1"
    ),
}

_FALSE_CLAIM_KEYS = {
    "D_V_accessed",
    "D_T_accessed",
    "resume_allowed",
    "automatic_retry_allowed",
    "formal_800_authorized",
    "full_CURE_authorized",
    "cross_backbone_authorized",
    "performance_claim_supported",
    "multi_seed_claim_supported",
    "historical_controls_retrained",
    "historical_controls_reevaluated",
    "historical_control_outcomes_are_candidate_gates",
}


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _stable_fingerprint(payload: Any) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalized_relative(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or "." in pure.parts
        or ".." in pure.parts
        or str(pure) != value
    ):
        raise ValueError(f"{name} is not a normalized relative path")
    return value


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite value {item}")
            ),
        )
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _verified_fingerprint(
    payload: Mapping[str, Any],
    *,
    field: str,
    name: str,
) -> str:
    value = dict(payload)
    recorded = value.pop(field, None)
    if not _is_sha256(recorded) or _stable_fingerprint(value) != recorded:
        raise RuntimeError(f"{name} fingerprint changed")
    return str(recorded)


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key), item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _assert_claim_boundary(payload: Any, *, name: str) -> None:
    for key, value in _walk(payload):
        if key in _FALSE_CLAIM_KEYS and value is not False:
            raise RuntimeError(f"{name} illegally sets {key}")
        if key == "contemporaneous_controls" and value is not False:
            raise RuntimeError(f"{name} makes controls contemporaneous")
        if key in {"seed", "execution_seed"} and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value != 42
        ):
            raise RuntimeError(f"{name} contains a non-frozen seed")


def _canonical_regular_file(
    root: Path,
    relative: str,
    *,
    name: str,
) -> Path:
    relative = _normalized_relative(relative, name=name)
    candidate = root / relative
    absolute = Path(os.path.abspath(candidate))
    if candidate.is_symlink():
        raise RuntimeError(f"{name} may not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if (
        resolved != absolute
        or not resolved.is_file()
        or resolved.is_symlink()
        or not resolved.is_relative_to(root)
    ):
        raise RuntimeError(f"{name} must be a canonical regular file")
    return resolved


def _tree_files(root: Path) -> tuple[str, ...]:
    result: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("PMOPE terminal tree contains a symbolic link")
        if path.is_file():
            result.append(str(path.relative_to(root)))
        elif not path.is_dir():
            raise RuntimeError("PMOPE terminal tree contains a special file")
    return tuple(result)


def _artifact_map(
    complete: Mapping[str, Any],
    *,
    root: Path,
    expected: tuple[str, ...],
) -> dict[str, str]:
    raw = complete.get("artifact_files")
    if not isinstance(raw, Mapping):
        raise RuntimeError("COMPLETE artifact map is missing")
    result: dict[str, str] = {}
    for key, value in raw.items():
        relative = _normalized_relative(key, name="artifact path")
        if not _is_sha256(value):
            raise RuntimeError(f"artifact hash is invalid: {relative}")
        result[relative] = str(value)
    if (
        tuple(sorted(result)) != tuple(sorted(expected))
        or complete.get("artifact_file_count") != len(expected)
    ):
        raise RuntimeError("COMPLETE artifact population changed")
    for relative, expected_hash in result.items():
        path = _canonical_regular_file(
            root,
            relative,
            name=f"artifact {relative}",
        )
        if _file_sha256(path) != expected_hash:
            raise RuntimeError(f"artifact bytes changed: {relative}")
    return result


def _read_receipts(
    root: Path,
    paths: Iterable[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for relative in paths:
        if not relative.endswith(".json"):
            continue
        payload = _strict_json(
            _canonical_regular_file(
                root,
                relative,
                name=f"receipt {relative}",
            ),
            name=f"receipt {relative}",
        )
        expected_schema = _RECEIPT_SCHEMA_BY_PATH[relative]
        if payload.get("schema_version") != expected_schema:
            raise RuntimeError(f"receipt schema changed: {relative}")
        _verified_fingerprint(
            payload,
            field="receipt_fingerprint",
            name=relative,
        )
        _assert_claim_boundary(payload, name=relative)
        result[relative] = payload
    return result


def _verify_implementation(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
) -> str:
    implementation = config.get("implementation")
    if not isinstance(implementation, Mapping):
        raise RuntimeError("config implementation binding is missing")
    raw_files = implementation.get("files")
    if not isinstance(raw_files, Mapping):
        raise RuntimeError("config implementation file map is missing")
    files: dict[str, str] = {}
    for raw_relative, raw_digest in raw_files.items():
        relative = _normalized_relative(
            raw_relative,
            name="implementation path",
        )
        if not _is_sha256(raw_digest):
            raise RuntimeError(
                f"implementation hash is invalid: {relative}"
            )
        files[relative] = str(raw_digest)
    if tuple(sorted(files)) != tuple(sorted(EXPECTED_IMPLEMENTATION_PATHS)):
        raise RuntimeError("PMOPE implementation source closure is incomplete")
    for relative, expected_hash in files.items():
        path = _canonical_regular_file(
            repository_root,
            relative,
            name=f"implementation {relative}",
        )
        if _file_sha256(path) != expected_hash:
            raise RuntimeError(
                f"implementation source changed: {relative}"
            )
    fingerprint = implementation.get("implementation_fingerprint")
    if (
        not _is_sha256(fingerprint)
        or fingerprint != _stable_fingerprint(files)
    ):
        raise RuntimeError("implementation fingerprint changed")
    return str(fingerprint)


def _tar_member_hashes(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with tarfile.open(path, mode="r:") as archive:
        for member in archive.getmembers():
            relative = _normalized_relative(
                member.name,
                name="v17 source archive member",
            )
            if not member.isfile() or relative in result:
                raise RuntimeError("v17 source archive is not regular and exact")
            handle = archive.extractfile(member)
            if handle is None:
                raise RuntimeError("v17 source archive member is unreadable")
            result[relative] = sha256(handle.read()).hexdigest()
    return result


def _verify_sealed_v17(
    wrapper: Mapping[str, Any],
    *,
    repository_root: Path,
) -> str:
    sealed = wrapper.get("sealed_v17")
    if not isinstance(sealed, Mapping):
        raise RuntimeError("sealed-v17 canonical receipt is missing")
    fingerprint = _stable_fingerprint(sealed)
    if (
        wrapper.get("sealed_v17_receipt_fingerprint") != fingerprint
        or sealed.get("historical_frozen_controls") is not True
        or sealed.get("contemporaneous_controls") is not False
        or sealed.get("control_outcomes_are_not_candidate_gates") is not True
        or sealed.get("model_deserialization_performed") is not False
        or sealed.get("evaluator_called") is not False
        or sealed.get("training_performed") is not False
        or sealed.get("D_R_cached_tensor_payload_accessed") is not False
        or sealed.get("D_V_accessed") is not False
        or sealed.get("D_T_accessed") is not False
        or sealed.get("runtime_splits") != []
        or sealed.get("run_repo_path") != V17_RUN_REPO_PATH
    ):
        raise RuntimeError("sealed-v17 claim boundary changed")
    _assert_claim_boundary(sealed, name="sealed v17")

    artifact_files = sealed.get("artifact_files")
    if not isinstance(artifact_files, Mapping) or len(artifact_files) != 17:
        raise RuntimeError("sealed-v17 artifact map changed")
    v17_root = repository_root / V17_RUN_REPO_PATH
    v17_root = v17_root.resolve(strict=True)
    if not v17_root.is_dir() or v17_root.is_symlink():
        raise RuntimeError("sealed-v17 run directory changed")
    expected_v17_tree = set()
    for raw_relative, raw_digest in artifact_files.items():
        relative = _normalized_relative(
            raw_relative,
            name="sealed-v17 artifact path",
        )
        if not _is_sha256(raw_digest):
            raise RuntimeError("sealed-v17 artifact hash is invalid")
        path = _canonical_regular_file(
            v17_root,
            relative,
            name=f"sealed-v17 artifact {relative}",
        )
        if _file_sha256(path) != raw_digest:
            raise RuntimeError(f"sealed-v17 artifact changed: {relative}")
        expected_v17_tree.add(relative)
    complete_path = _canonical_regular_file(
        v17_root,
        "COMPLETE.json",
        name="sealed-v17 COMPLETE",
    )
    if _file_sha256(complete_path) != sealed.get("complete_file_sha256"):
        raise RuntimeError("sealed-v17 COMPLETE bytes changed")
    v17_complete = _strict_json(
        complete_path,
        name="sealed-v17 COMPLETE",
    )
    if (
        _verified_fingerprint(
            v17_complete,
            field="complete_fingerprint",
            name="sealed-v17 COMPLETE",
        )
        != sealed.get("complete_fingerprint")
        or v17_complete.get("artifact_files") != dict(artifact_files)
        or v17_complete.get("artifact_file_count") != 17
    ):
        raise RuntimeError("sealed-v17 COMPLETE graph changed")
    expected_v17_tree.add("COMPLETE.json")
    if set(_tree_files(v17_root)) != expected_v17_tree:
        raise RuntimeError("sealed-v17 run tree changed")

    source = sealed.get("source_closure")
    if not isinstance(source, Mapping):
        raise RuntimeError("sealed-v17 source closure is missing")
    manifest_path = _canonical_regular_file(
        repository_root,
        V17_SOURCE_MANIFEST_REPO_PATH,
        name="sealed-v17 source manifest",
    )
    archive_path = _canonical_regular_file(
        repository_root,
        V17_SOURCE_ARCHIVE_REPO_PATH,
        name="sealed-v17 source archive",
    )
    if (
        _file_sha256(manifest_path) != source.get("manifest_file_sha256")
        or _file_sha256(archive_path) != source.get("archive_file_sha256")
    ):
        raise RuntimeError("sealed-v17 source closure bytes changed")
    source_members = source.get("source_members")
    if not isinstance(source_members, Mapping) or len(source_members) != 40:
        raise RuntimeError("sealed-v17 source member map changed")
    if _tar_member_hashes(archive_path) != dict(source_members):
        raise RuntimeError("sealed-v17 source archive members changed")
    manifest = _strict_json(
        manifest_path,
        name="sealed-v17 source manifest",
    )
    if (
        manifest.get("run_repo_path") != V17_RUN_REPO_PATH
        or manifest.get("complete_file_sha256")
        != sealed.get("complete_file_sha256")
        or manifest.get("complete_fingerprint")
        != sealed.get("complete_fingerprint")
        or manifest.get("archive_repo_path")
        != V17_SOURCE_ARCHIVE_REPO_PATH
        or manifest.get("archive_sha256")
        != source.get("archive_file_sha256")
        or manifest.get("source_file_count") != 40
        or manifest.get("implementation_fingerprint")
        != source.get("implementation_fingerprint")
    ):
        raise RuntimeError("sealed-v17 source manifest binding changed")

    controls = sealed.get("controls")
    if (
        not isinstance(controls, list)
        or [value.get("objective") for value in controls if isinstance(value, Mapping)]
        != [
            "support_oriented_response_joint",
            "identity_joint",
            "separable_endpoint",
        ]
    ):
        raise RuntimeError("sealed-v17 control set changed")
    for control in controls:
        if not isinstance(control, Mapping):
            raise RuntimeError("sealed-v17 control is malformed")
        objective = str(control["objective"])
        checkpoint_path = f"checkpoints/{objective}.safetensors"
        checkpoint_receipt_path = (
            f"checkpoints/{objective}.checkpoint.json"
        )
        if (
            artifact_files.get(checkpoint_path)
            != control.get("checkpoint_file_sha256")
        ):
            raise RuntimeError(f"sealed-v17 checkpoint changed: {objective}")
        checkpoint_receipt = _strict_json(
            v17_root / checkpoint_receipt_path,
            name=f"sealed-v17 checkpoint receipt {objective}",
        )
        if (
            _verified_fingerprint(
                checkpoint_receipt,
                field="receipt_fingerprint",
                name=f"sealed-v17 checkpoint receipt {objective}",
            )
            != control.get("checkpoint_receipt_fingerprint")
            or checkpoint_receipt.get("objective") != objective
            or checkpoint_receipt.get("checkpoint_file_sha256")
            != control.get("checkpoint_file_sha256")
            or checkpoint_receipt.get("module_state_fingerprint")
            != control.get("module_state_fingerprint")
        ):
            raise RuntimeError(
                f"sealed-v17 checkpoint receipt changed: {objective}"
            )
    return fingerprint


def _exact_value(
    payload: Mapping[str, Any],
    key: str,
    expected: Any,
    *,
    name: str,
) -> None:
    if payload.get(key) != expected:
        raise RuntimeError(f"{name} changed {key}")


def _verify_common_config(
    config: Mapping[str, Any],
    *,
    repository_root: Path,
) -> str:
    _exact_value(config, "run_id", RUN_ID, name="config")
    _exact_value(config, "output_repo_path", RUN_REPO_PATH, name="config")
    _exact_value(config, "runtime_splits", ["D_R"], name="config")
    model = config.get("model")
    budget = config.get("budget")
    evidence = config.get("evidence_scope")
    if (
        not isinstance(model, Mapping)
        or model.get("class")
        != "CURELiteCenteredMixedInteractionLevelSet"
        or model.get("objective_suite") != ["pmope_joint"]
        or model.get("candidate_objective") != "pmope_joint"
        or not isinstance(model.get("candidate_objective_policy"), str)
        or model.get("parameter_count") != 64064
        or model.get("field_threshold") != 0.0
        or model.get("threshold_search_performed") is not False
        or not isinstance(budget, Mapping)
        or budget.get("seed") != 42
        or budget.get("epochs") != 10
        or budget.get("steps_per_epoch") != 40
        or budget.get("updates_per_objective") != 400
        or budget.get("objectives") != 1
        or not isinstance(evidence, Mapping)
        or evidence.get("bounded_400_authorized") is not False
    ):
        raise RuntimeError("singleton PMOPE config changed")
    return _verify_implementation(
        config,
        repository_root=repository_root,
    )


def _receipt_graph_value(
    complete: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    complete_key: str,
    name: str,
) -> None:
    if complete.get(complete_key) != receipt.get("receipt_fingerprint"):
        raise RuntimeError(f"COMPLETE no longer binds {name}")


def _verify_bounded_complete(
    root: Path,
    complete: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    if set(_tree_files(root)) != {
        *EXPECTED_BOUNDED_ARTIFACT_PATHS,
        "COMPLETE.json",
    }:
        raise RuntimeError("bounded COMPLETE tree must contain exactly 16 files")
    _artifact_map(
        complete,
        root=root,
        expected=EXPECTED_BOUNDED_ARTIFACT_PATHS,
    )
    receipts = _read_receipts(root, EXPECTED_BOUNDED_ARTIFACT_PATHS)
    config = receipts["receipts/config.json"]
    implementation_fp = _verify_common_config(
        config,
        repository_root=repository_root,
    )
    attempt = receipts["attempt.json"]
    if (
        attempt.get("run_id") != RUN_ID
        or attempt.get("candidate_objective") != "pmope_joint"
        or attempt.get("objectives") != 1
        or attempt.get("single_attempt") is not True
        or attempt.get("config_fingerprint")
        != config.get("receipt_fingerprint")
    ):
        raise RuntimeError("bounded attempt contract changed")

    dataset_free = receipts["receipts/dataset_free.json"]
    dr_gate = receipts["receipts/dr_gate.json"]
    sealed = receipts["receipts/sealed_v17_controls.json"]
    authorization = receipts["receipts/authorization.json"]
    memory = receipts["receipts/device_memory_preflight.json"]
    training = receipts["receipts/training.json"]
    zero = receipts["receipts/zero_level.json"]
    bounded = receipts["receipts/bounded_result.json"]
    decision = receipts["receipts/decision.json"]
    if (
        dataset_free.get("all_pass") is not True
        or dr_gate.get("all_pass") is not True
        or dr_gate.get("gate_run_count") != 1
        or dr_gate.get("optimizer_steps") != 0
        or dr_gate.get("training_performed") is not False
        or authorization.get("training_authorized") is not True
        or memory.get("all_pass") is not True
        or training.get("bounded_training_performed") is not True
        or training.get("formal_training_performed") is not False
        or training.get("candidate_count") != 1
        or training.get("candidate_objective") != "pmope_joint"
        or zero.get("candidate_objective") != "pmope_joint"
    ):
        raise RuntimeError("bounded prerequisite/training receipt changed")
    sealed_fp = _verify_sealed_v17(
        sealed,
        repository_root=repository_root,
    )

    checkpoint_receipt = receipts[
        "checkpoints/pmope_joint.checkpoint.json"
    ]
    checkpoint_path = root / "checkpoints/pmope_joint.safetensors"
    checkpoint_fp = checkpoint_receipt["receipt_fingerprint"]
    checkpoint_model_config = checkpoint_receipt.get("model_config")
    expected_checkpoint_repo_path = (
        f"{RUN_REPO_PATH}/checkpoints/pmope_joint.safetensors"
    )
    if (
        checkpoint_receipt.get("objective") != "pmope_joint"
        or checkpoint_receipt.get("objective_policy")
        != config["model"]["candidate_objective_policy"]
        or checkpoint_receipt.get("model_class")
        != "CURELiteCenteredMixedInteractionLevelSet"
        or not isinstance(checkpoint_model_config, Mapping)
        or checkpoint_model_config.get("parameter_count") != 64064
        or checkpoint_model_config.get("fixed_margin_hex")
        != config["model"].get("fixed_margin_hex")
        or checkpoint_receipt.get("repo_relative_path")
        != expected_checkpoint_repo_path
        or checkpoint_receipt.get("serialization") != "safetensors"
        or checkpoint_receipt.get("tensor_only_state_dict") is not True
        or checkpoint_receipt.get("weights_only_roundtrip_verified") is not True
        or checkpoint_receipt.get("device_policy") != "cpu_checkpoint"
        or checkpoint_receipt.get("checkpoint_file_sha256")
        != _file_sha256(checkpoint_path)
        or not _is_sha256(
            checkpoint_receipt.get("module_state_fingerprint")
        )
    ):
        raise RuntimeError("singleton PMOPE checkpoint contract changed")
    expected_checkpoint_map = {"pmope_joint": checkpoint_fp}
    if (
        training.get("checkpoint_receipt_fingerprints")
        != expected_checkpoint_map
        or decision.get("checkpoint_receipt_fingerprints")
        != expected_checkpoint_map
    ):
        raise RuntimeError("checkpoint receipt graph changed")
    training_payload = training.get("training")
    training_objectives = (
        training_payload.get("objectives")
        if isinstance(training_payload, Mapping)
        else None
    )
    candidate_diagnostic = zero.get("candidate_diagnostic")
    if (
        not isinstance(training_payload, Mapping)
        or training_payload.get("objective_suite") != ["pmope_joint"]
        or not isinstance(training_objectives, list)
        or len(training_objectives) != 1
        or not isinstance(training_objectives[0], Mapping)
        or training_objectives[0].get("objective") != "pmope_joint"
        or training_objectives[0].get("seed") != 42
        or training_objectives[0].get("epochs") != 10
        or training_objectives[0].get("steps_per_epoch") != 40
        or training_objectives[0].get("completed_updates") != 400
        # ``final_model_fingerprint`` and ``module_state_fingerprint`` are
        # deliberately different contracts: training fingerprints the
        # tensor map under the coverage-state training schema, whereas the
        # checkpoint/evaluator use the generic module-state schema.  Their
        # values therefore must each be valid and are not expected to be
        # byte-identical.
        or not _is_sha256(
            training_objectives[0].get("final_model_fingerprint")
        )
        or not isinstance(candidate_diagnostic, Mapping)
        or candidate_diagnostic.get("checkpoint_fingerprint")
        != checkpoint_receipt.get("module_state_fingerprint")
    ):
        raise RuntimeError("candidate checkpoint/training/diagnostic graph changed")

    bounded_passed = complete.get("bounded_gate_passed")
    if (
        not isinstance(bounded_passed, bool)
        or decision.get("bounded_gate_passed") is not bounded_passed
        or decision.get("candidate_gate_passed") is not bounded_passed
        or decision.get("formal800_eligible") is not bounded_passed
        or complete.get("formal800_eligible") is not bounded_passed
        or decision.get("status")
        != (
            "PMOPE_V18_BOUNDED_400_GATE_PASS"
            if bounded_passed
            else "PMOPE_V18_BOUNDED_400_GATE_FAIL"
        )
        or complete.get("decision") != decision.get("status")
        or complete.get("D_R_gate_evidence_fingerprint")
        != dr_gate.get("D_R_gate_evidence_fingerprint")
        or complete.get("sealed_v17_evidence_fingerprint") != sealed_fp
    ):
        raise RuntimeError("bounded decision graph changed")

    graph = (
        ("config_fingerprint", "receipts/config.json"),
        ("input_receipt_fingerprint", "receipts/inputs.json"),
        ("preflight_receipt_fingerprint", "receipts/preflight.json"),
        (
            "dataset_free_receipt_fingerprint",
            "receipts/dataset_free.json",
        ),
        ("D_R_gate_receipt_fingerprint", "receipts/dr_gate.json"),
        (
            "sealed_v17_receipt_fingerprint",
            "receipts/sealed_v17_controls.json",
        ),
        (
            "authorization_receipt_fingerprint",
            "receipts/authorization.json",
        ),
        (
            "device_memory_preflight_receipt_fingerprint",
            "receipts/device_memory_preflight.json",
        ),
        ("training_receipt_fingerprint", "receipts/training.json"),
        ("zero_level_receipt_fingerprint", "receipts/zero_level.json"),
        (
            "bounded_result_receipt_fingerprint",
            "receipts/bounded_result.json",
        ),
        ("decision_fingerprint", "receipts/decision.json"),
    )
    for complete_key, relative in graph:
        _receipt_graph_value(
            complete,
            receipts[relative],
            complete_key=complete_key,
            name=relative,
        )
    if (
        bounded.get("result_fingerprint")
        != decision.get("result_fingerprint")
        or complete.get("single_attempt") is not True
        or complete.get("calibration_performed") is not False
        or complete.get("performance_evaluation_performed") is not False
    ):
        raise RuntimeError("bounded terminal claim changed")
    return {
        "terminal_state": "complete_bounded",
        "decision": str(decision["status"]),
        "bounded_gate_passed": bounded_passed,
        "artifact_file_count": 15,
        "tree_file_count": 16,
        "checkpoint_count": 1,
        "receipt_count": 12,
        "implementation_fingerprint": implementation_fp,
        "sealed_v17_receipt_fingerprint": sealed_fp,
    }


def _verify_gate_stop_complete(
    root: Path,
    complete: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    if set(_tree_files(root)) != {
        *EXPECTED_GATE_STOP_ARTIFACT_PATHS,
        "COMPLETE.json",
    }:
        raise RuntimeError("D_R gate-stop tree is not exact")
    _artifact_map(
        complete,
        root=root,
        expected=EXPECTED_GATE_STOP_ARTIFACT_PATHS,
    )
    receipts = _read_receipts(root, EXPECTED_GATE_STOP_ARTIFACT_PATHS)
    config = receipts["receipts/config.json"]
    implementation_fp = _verify_common_config(
        config,
        repository_root=repository_root,
    )
    attempt = receipts["attempt.json"]
    dr_gate = receipts["receipts/dr_gate.json"]
    decision = receipts["receipts/decision.json"]
    sealed = receipts["receipts/sealed_v17_controls.json"]
    sealed_fp = _verify_sealed_v17(
        sealed,
        repository_root=repository_root,
    )
    if (
        attempt.get("config_fingerprint")
        != config.get("receipt_fingerprint")
        or attempt.get("single_attempt") is not True
        or receipts["receipts/dataset_free.json"].get("all_pass") is not True
        or dr_gate.get("all_pass") is not False
        or dr_gate.get("gate_run_count") != 1
        or dr_gate.get("training_performed") is not False
        or decision.get("status") != "PMOPE_V18_DR_GATE_FAIL"
        or decision.get("D_R_gate_passed") is not False
        or decision.get("bounded_gate_passed") is not False
        or decision.get("authorization_created") is not False
        or decision.get("bounded_training_performed") is not False
        or decision.get("checkpoint_count") != 0
        or decision.get("formal800_eligible") is not False
        or complete.get("decision") != "PMOPE_V18_DR_GATE_FAIL"
        or complete.get("D_R_gate_passed") is not False
        or complete.get("bounded_gate_passed") is not False
        or complete.get("authorization_created") is not False
        or complete.get("bounded_training_performed") is not False
        or complete.get("checkpoint_count") != 0
        or complete.get("formal800_eligible") is not False
    ):
        raise RuntimeError("D_R gate-stop terminal claim changed")
    graph = (
        ("config_fingerprint", "receipts/config.json"),
        ("input_receipt_fingerprint", "receipts/inputs.json"),
        ("preflight_receipt_fingerprint", "receipts/preflight.json"),
        (
            "dataset_free_receipt_fingerprint",
            "receipts/dataset_free.json",
        ),
        ("D_R_gate_receipt_fingerprint", "receipts/dr_gate.json"),
        (
            "sealed_v17_receipt_fingerprint",
            "receipts/sealed_v17_controls.json",
        ),
        ("decision_fingerprint", "receipts/decision.json"),
    )
    for complete_key, relative in graph:
        _receipt_graph_value(
            complete,
            receipts[relative],
            complete_key=complete_key,
            name=relative,
        )
    return {
        "terminal_state": "complete_D_R_gate_stop",
        "decision": "PMOPE_V18_DR_GATE_FAIL",
        "bounded_gate_passed": False,
        "artifact_file_count": 8,
        "tree_file_count": 9,
        "checkpoint_count": 0,
        "receipt_count": 7,
        "implementation_fingerprint": implementation_fp,
        "sealed_v17_receipt_fingerprint": sealed_fp,
    }


def _verify_failure(
    root: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    files = set(_tree_files(root))
    if "COMPLETE.json" in files:
        raise RuntimeError("FAILURE terminal may not contain COMPLETE")
    if ".incomplete" not in files or "FAILURE.json" not in files:
        raise RuntimeError("failure terminal markers are incomplete")
    marker = root / ".incomplete"
    if not marker.is_file() or marker.is_symlink() or marker.stat().st_size != 0:
        raise RuntimeError("failure .incomplete marker changed")
    failure = _strict_json(root / "FAILURE.json", name="FAILURE")
    _verified_fingerprint(
        failure,
        field="receipt_fingerprint",
        name="FAILURE",
    )
    _assert_claim_boundary(failure, name="FAILURE")
    if (
        failure.get("schema_version") != FAILURE_SCHEMA
        or failure.get("status") != "failed_incomplete_attempt"
        or not _is_sha256(failure.get("attempt_fingerprint"))
    ):
        raise RuntimeError("FAILURE contract changed")
    attempt = _strict_json(root / "attempt.json", name="attempt")
    _verified_fingerprint(
        attempt,
        field="receipt_fingerprint",
        name="attempt",
    )
    _assert_claim_boundary(attempt, name="attempt")
    if (
        attempt.get("schema_version") != ATTEMPT_SCHEMA
        or attempt.get("receipt_fingerprint")
        != failure.get("attempt_fingerprint")
        or attempt.get("single_attempt") is not True
    ):
        raise RuntimeError("FAILURE attempt binding changed")
    expected_partial = {
        relative: _file_sha256(root / relative)
        for relative in files
        if relative not in {".incomplete", "FAILURE.json"}
    }
    if failure.get("artifact_files_before_failure") != expected_partial:
        raise RuntimeError("FAILURE partial artifact map changed")
    for relative in expected_partial:
        if relative.endswith(".json"):
            payload = _strict_json(root / relative, name=relative)
            _verified_fingerprint(
                payload,
                field="receipt_fingerprint",
                name=relative,
            )
            _assert_claim_boundary(payload, name=relative)
    implementation_fp: str | None = None
    config_path = root / "receipts/config.json"
    if config_path.is_file():
        config = _strict_json(config_path, name="failure config")
        implementation_fp = _verify_common_config(
            config,
            repository_root=repository_root,
        )
    return {
        "terminal_state": "failed_incomplete",
        "decision": None,
        "bounded_gate_passed": False,
        "artifact_file_count": len(expected_partial),
        "tree_file_count": len(files),
        "checkpoint_count": sum(
            relative.endswith(".safetensors") for relative in files
        ),
        "receipt_count": sum(
            relative.startswith("receipts/")
            and relative.endswith(".json")
            for relative in files
        ),
        "implementation_fingerprint": implementation_fp,
        "sealed_v17_receipt_fingerprint": None,
    }


def audit_coverage_state_cmif_pmope_v18(
    run_path: str | Path = DEFAULT_RUN_PATH,
    *,
    repository_root: str | Path = _ROOT,
) -> dict[str, Any]:
    """Audit one v18 terminal without loading model or dataset tensors."""

    repository = Path(repository_root)
    repository = repository.resolve(strict=True)
    if not repository.is_dir() or repository.is_symlink():
        raise ValueError("repository_root must be a canonical directory")
    root = Path(run_path)
    root = root.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("run_path must be a canonical directory")

    files = set(_tree_files(root))
    has_complete = "COMPLETE.json" in files
    has_failure = "FAILURE.json" in files
    if has_complete and has_failure:
        raise RuntimeError("terminal may not contain COMPLETE and FAILURE")
    if has_complete:
        if ".incomplete" in files:
            raise RuntimeError("complete terminal retained .incomplete")
        complete = _strict_json(root / "COMPLETE.json", name="COMPLETE")
        complete_fp = _verified_fingerprint(
            complete,
            field="complete_fingerprint",
            name="COMPLETE",
        )
        _assert_claim_boundary(complete, name="COMPLETE")
        if (
            complete.get("schema_version") != RUN_SCHEMA
            or complete.get("status") != "complete"
            or complete.get("run_id") != RUN_ID
            or complete.get("single_attempt") is not True
        ):
            raise RuntimeError("COMPLETE contract changed")
        if complete.get("decision") == "PMOPE_V18_DR_GATE_FAIL":
            summary = _verify_gate_stop_complete(
                root,
                complete,
                repository_root=repository,
            )
        else:
            summary = _verify_bounded_complete(
                root,
                complete,
                repository_root=repository,
            )
        terminal_fingerprint = complete_fp
    elif has_failure:
        summary = _verify_failure(
            root,
            repository_root=repository,
        )
        terminal = _strict_json(root / "FAILURE.json", name="FAILURE")
        terminal_fingerprint = str(terminal["receipt_fingerprint"])
    else:
        raise RuntimeError("run has no recognized terminal marker")

    payload = {
        "schema_version": AUDIT_SCHEMA,
        "run_path": str(root),
        **summary,
        "terminal_fingerprint": terminal_fingerprint,
        "checks": {
            "strict_regular_file_tree": True,
            "all_json_fingerprints_verified": True,
            "all_declared_file_hashes_verified": True,
            "single_seed42_only": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "model_deserialization_performed": False,
            "evaluator_called": False,
            "training_performed_by_auditor": False,
            "filesystem_writes_performed_by_auditor": False,
        },
    }
    payload["audit_fingerprint"] = _stable_fingerprint(payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_path",
        nargs="?",
        default=str(DEFAULT_RUN_PATH),
        help="terminal run directory; defaults to the frozen formal v18 path",
    )
    parser.add_argument(
        "--repository-root",
        default=str(_ROOT),
        help="repository root used to verify implementation and sealed v17 bytes",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = audit_coverage_state_cmif_pmope_v18(
        args.run_path,
        repository_root=args.repository_root,
    )
    print(
        json.dumps(
            result,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
