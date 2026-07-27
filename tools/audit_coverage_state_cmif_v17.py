#!/usr/bin/env python3
"""Validate or run the frozen create-only CMIF-v17 P0 contract.

``--validate-create-only`` checks immutable inputs, the frozen v16
negative-result closure, the generated-only CMIF gate, implementation files,
and the two reserved output locations without loading cached ``D_R`` tensors.
``--run-once r1`` and ``--run-once r2`` each perform exactly one independent
P0 audit.  Only a complete persisted r1 and byte-identical persisted r2 core
receipt may produce the bounded-400 decision.  No mode trains a model or
accesses ``D_V`` or ``D_T``.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.experiment.coverage_state_cmif_dataset_free import (  # noqa: E402
    run_coverage_state_cmif_dataset_free_gate,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (  # noqa: E402
    build_coverage_state_bounded_population,
)
from cure_lite.experiment.coverage_state_cmif_p0 import (  # noqa: E402
    run_coverage_state_cmif_p0_single,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (  # noqa: E402
    bind_coverage_state_real_dr_sources,
    build_coverage_state_real_dr_inputs,
)


_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_SCHEMA = "cure-lite-cmif-v17-p0-create-only-validation-v1"
ATTEMPT_SCHEMA = "cure-lite-cmif-v17-p0-create-only-attempt-v1"
CONFIG_SCHEMA = "cure-lite-cmif-v17-p0-create-only-config-v1"
INPUTS_SCHEMA = "cure-lite-cmif-v17-p0-create-only-inputs-v1"
DATASET_FREE_SCHEMA = (
    "cure-lite-cmif-v17-p0-create-only-dataset-free-v1"
)
REPLAY_COMPARISON_SCHEMA = (
    "cure-lite-cmif-v17-p0-create-only-persisted-replay-comparison-v2"
)
DECISION_SCHEMA = "cure-lite-cmif-v17-p0-create-only-decision-v2"
COMPLETE_SCHEMA = "cure-lite-cmif-v17-p0-create-only-complete-v1"
FAILURE_SCHEMA = "cure-lite-cmif-v17-p0-create-only-failure-v1"
INCOMPLETE_MARKER = ".incomplete"

FROZEN_DATASET = "IRSTD-1K"
FROZEN_SPLIT = "D_R"
FROZEN_REAL_DR_SOURCE_BINDING_FINGERPRINT = (
    "9689ac7dc4cd95bd0e9bcf79e12e83bc1c8606a96e99ca27945dc07baf4fc74d"
)
FROZEN_REAL_DR_INPUTS = (
    (
        "manifest_path",
        "protocols/IRSTD-1K/stage_a_seed42/manifest.json",
        "aa8e33529bd86f564ce6e163e0f9a7b1b3053e9c15054a59c6702a1523f35c02",
    ),
    (
        "state_index_path",
        (
            "runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3/"
            "d_r/state_cache/index.json"
        ),
        "075fc1ad217f365df85b1d29568ad215f06ce6e0b691ef78a5dd85f0affe6298",
    ),
    (
        "geometry_config_path",
        "protocols/IRSTD-1K/geometry_safe_p0_v2/config.json",
        "719e956b7c51b2b2c8294699fe26c2d36d5c8190b0d8bb5c1d5665a0f4344558",
    ),
    (
        "geometry_receipt_path",
        (
            "runs/irstd1k_stage_a_seed42/"
            "cure_lite_geometry_safe_p0_v2_r1/"
            "receipts/geometry_catalog.json"
        ),
        "e2a9a986f8819433f3f5efd5c4f627504d10fb32d20f62769b2235b803209283",
    ),
    (
        "observability_config_path",
        (
            "protocols/IRSTD-1K/"
            "coverage_state_observability_v1/config.json"
        ),
        "60d42e657f1daed3cb01c7ee93c8f3fe17417542931d853756ccbbeda1f95713",
    ),
)

PARENT_V16_RUN_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/"
    "cure_lite_cslf_v16_ppce_support_oriented_bounded_400_r1"
)
PARENT_V16_COMPLETE_REPO_PATH = (
    f"{PARENT_V16_RUN_REPO_PATH}/COMPLETE.json"
)
PARENT_V16_COMPLETE_SHA256 = (
    "c73bcd79848f1b57fbfdcf92ded572cde8dae08f81334b0b1eaded62952ba649"
)
PARENT_V16_COMPLETE_FINGERPRINT = (
    "7eba70fc32f70411f915a1d63261c32ac814232613bc68868cd6bb441b5bf599"
)
PARENT_V16_DECISION = (
    "BOUNDED_PPCE_SUPPORT_ORIENTED_CSLF_GATE_FAIL"
)
PARENT_V16_ARTIFACT_FILE_COUNT = 17
PARENT_V16_SOURCE_MANIFEST_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_cslf_v16_ppce_support_oriented_bounded_400_7eba70fc32f7.json"
)
PARENT_V16_SOURCE_MANIFEST_SHA256 = (
    "d36cf9450d5974eafa45926c45f35103015e61676576ce346bb2e7d23cec501f"
)
PARENT_V16_SOURCE_ARCHIVE_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_cslf_v16_ppce_support_oriented_bounded_400_7eba70fc32f7.tar"
)
PARENT_V16_SOURCE_ARCHIVE_SHA256 = (
    "ef0d38a487ee14ed48dc79817c7e61042d61ee7aaca30a7b95cb2fd165409c4c"
)

P0_CORE_REPO_PATH = (
    "cure_lite/experiment/coverage_state_cmif_p0.py"
)
P0_SPEC_REPO_PATH = (
    "CURE_Lite_v16_PPCE失败归因与v17_FRCE修改方案.md"
)
STATIC_IMPLEMENTATION_REPO_PATHS = (
    P0_SPEC_REPO_PATH,
    "cure_lite/coverage_state_level_set.py",
    "cure_lite/coverage_state_phase_preserving.py",
    "cure_lite/coverage_state_centered_mixed_interaction.py",
    "cure_lite/coverage_state_precomputed_cache.py",
    "cure_lite/coverage_state_raw_catalog.py",
    "cure_lite/coverage_state_sobolev.py",
    "cure_lite/experiment/coverage_state_bounded_protocol.py",
    "cure_lite/experiment/coverage_state_cmif_dataset_free.py",
    P0_CORE_REPO_PATH,
    "cure_lite/experiment/coverage_state_real_dr_inputs.py",
    "tools/audit_coverage_state_cmif_v17.py",
)

REPLICATE_OUTPUT_REPO_PATHS = {
    "r1": (
        "runs/irstd1k_stage_a_seed42/"
        "cure_lite_cmif_v17_p0_r1"
    ),
    "r2": (
        "runs/irstd1k_stage_a_seed42/"
        "cure_lite_cmif_v17_p0_r2"
    ),
}


def _validate_repo_relative_path(relative: str, *, name: str) -> str:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{name} repository path is invalid")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or str(pure) != relative
    ):
        raise ValueError(f"{name} must be a normalized repository path")
    return relative


def _canonical_repo_file(relative: str, *, name: str) -> Path:
    relative = _validate_repo_relative_path(relative, name=name)
    candidate = _ROOT / relative
    absolute = Path(os.path.abspath(candidate))
    if candidate.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if (
        resolved != absolute
        or not resolved.is_file()
        or resolved.is_symlink()
        or not resolved.is_relative_to(_ROOT)
    ):
        raise ValueError(f"{name} must be a canonical repository file")
    return resolved


def _canonical_repo_directory(relative: str, *, name: str) -> Path:
    relative = _validate_repo_relative_path(relative, name=name)
    candidate = _ROOT / relative
    absolute = Path(os.path.abspath(candidate))
    if candidate.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if (
        resolved != absolute
        or not resolved.is_dir()
        or resolved.is_symlink()
        or not resolved.is_relative_to(_ROOT)
    ):
        raise ValueError(f"{name} must be a canonical repository directory")
    return resolved


def _strict_json(path: Path, *, name: str) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
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
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain a JSON object")
    return dict(value)


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write one canonical JSON artifact without replacing an existing path."""

    _write_new_bytes(path, _json_bytes(payload))


def _write_new_bytes(path: Path, encoded: bytes) -> None:
    if not isinstance(encoded, bytes):
        raise TypeError("encoded artifact must be bytes")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fingerprinted(
    payload: Mapping[str, object],
    *,
    field: str = "receipt_fingerprint",
) -> dict[str, object]:
    result = dict(payload)
    if field in result:
        raise ValueError(f"payload already contains {field}")
    result[field] = stable_fingerprint(result)
    return result


def _prepare_fixed_output(replicate: str) -> Path:
    """Return an unused fixed output path without creating it."""

    if replicate not in REPLICATE_OUTPUT_REPO_PATHS:
        raise ValueError("replicate must be exactly r1 or r2")
    relative = REPLICATE_OUTPUT_REPO_PATHS[replicate]
    _validate_repo_relative_path(relative, name=f"{replicate} output")
    output = Path(os.path.abspath(_ROOT / relative))
    if not output.is_relative_to(_ROOT):
        raise ValueError("P0 output must remain inside the repository")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"fixed P0 output already exists: {output}")
    current = output.parent
    while current != _ROOT:
        if current.exists() and current.is_symlink():
            raise ValueError("P0 output path may not traverse a symbolic link")
        current = current.parent
    if _ROOT.is_symlink() or _ROOT.resolve(strict=True) != _ROOT:
        raise ValueError("repository root must be canonical")
    return output


def _claim_fixed_output(
    replicate: str,
    *,
    attempt: Mapping[str, object],
) -> tuple[Path, Path]:
    """Claim one fixed output exactly once."""

    output = _prepare_fixed_output(replicate)
    output.mkdir(parents=True, exist_ok=False)
    marker = output / INCOMPLETE_MARKER
    with marker.open("xb") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    _write_new_json(output / "attempt.json", attempt)
    receipts = output / "receipts"
    receipts.mkdir(exist_ok=False)
    _fsync_directory(receipts)
    _fsync_directory(output)
    _fsync_directory(output.parent)
    return output, receipts


def _verify_frozen_real_dr_sources() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, relative, expected_sha256 in FROZEN_REAL_DR_INPUTS:
        path = _canonical_repo_file(relative, name=name)
        if file_sha256(path) != expected_sha256:
            raise RuntimeError(f"frozen real D_R input changed: {name}")
        paths[name] = path
    return paths


def _verify_parent_v16_closure() -> dict[str, object]:
    parent = _canonical_repo_directory(
        PARENT_V16_RUN_REPO_PATH,
        name="parent v16 result",
    )
    complete_path = _canonical_repo_file(
        PARENT_V16_COMPLETE_REPO_PATH,
        name="parent v16 COMPLETE",
    )
    if file_sha256(complete_path) != PARENT_V16_COMPLETE_SHA256:
        raise RuntimeError("parent v16 COMPLETE file changed")
    complete = _strict_json(complete_path, name="parent v16 COMPLETE")
    complete_without_fingerprint = dict(complete)
    recorded_complete_fingerprint = complete_without_fingerprint.pop(
        "complete_fingerprint",
        None,
    )
    if (
        recorded_complete_fingerprint
        != PARENT_V16_COMPLETE_FINGERPRINT
        or stable_fingerprint(complete_without_fingerprint)
        != recorded_complete_fingerprint
        or complete.get("status") != "complete"
        or complete.get("decision") != PARENT_V16_DECISION
        or complete.get("bounded_gate_passed") is not False
        or complete.get("artifact_file_count")
        != PARENT_V16_ARTIFACT_FILE_COUNT
    ):
        raise RuntimeError("parent v16 result contract changed")
    artifacts = complete.get("artifact_files")
    if (
        not isinstance(artifacts, Mapping)
        or len(artifacts) != PARENT_V16_ARTIFACT_FILE_COUNT
    ):
        raise RuntimeError("parent v16 artifact map is incomplete")
    expected_paths: set[str] = set()
    for raw_relative, raw_sha256 in artifacts.items():
        if (
            not isinstance(raw_relative, str)
            or not isinstance(raw_sha256, str)
            or len(raw_sha256) != 64
            or raw_sha256 != raw_sha256.lower()
            or any(
                character not in "0123456789abcdef"
                for character in raw_sha256
            )
        ):
            raise RuntimeError("parent v16 artifact map is malformed")
        relative = _validate_repo_relative_path(
            raw_relative,
            name="parent v16 artifact",
        )
        path = parent / relative
        absolute = Path(os.path.abspath(path))
        if path.is_symlink():
            raise RuntimeError("parent v16 artifact became a symbolic link")
        resolved = path.resolve(strict=True)
        if (
            resolved != absolute
            or not resolved.is_file()
            or resolved.is_symlink()
            or not resolved.is_relative_to(parent)
            or file_sha256(resolved) != raw_sha256
        ):
            raise RuntimeError(
                f"parent v16 artifact changed: {relative}"
            )
        expected_paths.add(relative)
    actual_paths = {
        str(path.relative_to(parent))
        for path in parent.rglob("*")
        if path.is_file() and path.name != "COMPLETE.json"
    }
    if actual_paths != expected_paths:
        raise RuntimeError("parent v16 artifact population changed")

    source_manifest_path = _canonical_repo_file(
        PARENT_V16_SOURCE_MANIFEST_REPO_PATH,
        name="parent v16 source manifest",
    )
    source_archive_path = _canonical_repo_file(
        PARENT_V16_SOURCE_ARCHIVE_REPO_PATH,
        name="parent v16 source archive",
    )
    if (
        file_sha256(source_manifest_path)
        != PARENT_V16_SOURCE_MANIFEST_SHA256
        or file_sha256(source_archive_path)
        != PARENT_V16_SOURCE_ARCHIVE_SHA256
    ):
        raise RuntimeError("parent v16 source closure changed")
    source_manifest = _strict_json(
        source_manifest_path,
        name="parent v16 source manifest",
    )
    if (
        source_manifest.get("complete_fingerprint")
        != PARENT_V16_COMPLETE_FINGERPRINT
        or source_manifest.get("complete_file_sha256")
        != PARENT_V16_COMPLETE_SHA256
        or source_manifest.get("run_repo_path")
        != PARENT_V16_RUN_REPO_PATH
        or source_manifest.get("archive_repo_path")
        != PARENT_V16_SOURCE_ARCHIVE_REPO_PATH
        or source_manifest.get("archive_sha256")
        != PARENT_V16_SOURCE_ARCHIVE_SHA256
    ):
        raise RuntimeError("parent v16 source manifest changed")
    return {
        "run_repo_path": PARENT_V16_RUN_REPO_PATH,
        "complete_repo_path": PARENT_V16_COMPLETE_REPO_PATH,
        "complete_sha256": PARENT_V16_COMPLETE_SHA256,
        "complete_fingerprint": PARENT_V16_COMPLETE_FINGERPRINT,
        "decision": PARENT_V16_DECISION,
        "bounded_gate_passed": False,
        "artifact_file_count": PARENT_V16_ARTIFACT_FILE_COUNT,
        "source_manifest_repo_path": (
            PARENT_V16_SOURCE_MANIFEST_REPO_PATH
        ),
        "source_manifest_sha256": (
            PARENT_V16_SOURCE_MANIFEST_SHA256
        ),
        "source_archive_repo_path": (
            PARENT_V16_SOURCE_ARCHIVE_REPO_PATH
        ),
        "source_archive_sha256": PARENT_V16_SOURCE_ARCHIVE_SHA256,
    }


def _static_implementation_binding() -> tuple[tuple[str, str], ...]:
    result = []
    for relative in STATIC_IMPLEMENTATION_REPO_PATHS:
        path = _canonical_repo_file(relative, name="implementation file")
        result.append((relative, file_sha256(path)))
    return tuple(result)


def _artifact_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("P0 artifact may not be a symbolic link")
        if not path.is_file() or path.name in {
            INCOMPLETE_MARKER,
            "COMPLETE.json",
        }:
            continue
        result[str(path.relative_to(root))] = file_sha256(path)
    return result


def _static_config_payload(
    *,
    source_paths: Mapping[str, Path],
    parent: Mapping[str, object],
    dataset_free_receipt_fingerprint: str,
    dataset_free_evidence_fingerprint: str,
    implementation: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    return {
        "schema_version": CONFIG_SCHEMA,
        "method_id": "CMIF-CSLF-v17",
        "stage": "frozen_D_R_P0_create_only",
        "dataset": FROZEN_DATASET,
        "split": FROZEN_SPLIT,
        "runtime_splits": [FROZEN_SPLIT],
        "real_inputs": {
            name: {
                "repo_path": str(path.relative_to(_ROOT)),
                "file_sha256": dict(
                    (key, digest)
                    for key, _, digest in FROZEN_REAL_DR_INPUTS
                )[name],
            }
            for name, path in sorted(source_paths.items())
        },
        "source_binding_fingerprint": (
            FROZEN_REAL_DR_SOURCE_BINDING_FINGERPRINT
        ),
        "parent_v16_negative_result": dict(parent),
        "dataset_free_gate": {
            "receipt_fingerprint": dataset_free_receipt_fingerprint,
            "evidence_fingerprint": dataset_free_evidence_fingerprint,
            "all_pass_required": True,
        },
        "model_contract": {
            "representation": "phase_preserving",
            "feature_channels": 64,
            "feature_stride": 4,
            "occupancy_phase_channels": 16,
            "coarse_radius": 2,
            "field_threshold": 0.0,
            "threshold_search_performed": False,
        },
        "replicates": {
            name: {"output_repo_path": relative}
            for name, relative in sorted(
                REPLICATE_OUTPUT_REPO_PATHS.items()
            )
        },
        "implementation": {
            "files": dict(implementation),
            "implementation_fingerprint": stable_fingerprint(
                dict(implementation)
            ),
        },
        "execution_policy": {
            "create_only": True,
            "single_use_per_replicate": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "D_R_training_performed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
        "authorization_policy": {
            "r1_single_can_authorize": False,
            "r2_requires_complete_r1": True,
            "canonical_p0_core_bytes_must_match": True,
            "each_replicate_runs_exactly_one_single_audit": True,
            "in_memory_replay_can_authorize": False,
            "bounded_400_requires_persisted_replay_decision": True,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
        },
    }


def _failure_payload(
    error: BaseException,
    *,
    replicate: str,
    attempt_fingerprint: str,
    artifact_files: Mapping[str, str],
) -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": FAILURE_SCHEMA,
            "status": "failed_incomplete_attempt",
            "replicate": replicate,
            "exception_type": type(error).__name__,
            "message": str(error),
            "attempt_fingerprint": attempt_fingerprint,
            "artifact_files_before_failure": dict(artifact_files),
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "bounded_400_authorized": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "training_performed": False,
        }
    )


def _verify_complete_replicate(
    replicate: str,
) -> tuple[dict[str, object], bytes]:
    if replicate not in REPLICATE_OUTPUT_REPO_PATHS:
        raise ValueError("replicate must be exactly r1 or r2")
    root = _canonical_repo_directory(
        REPLICATE_OUTPUT_REPO_PATHS[replicate],
        name=f"CMIF P0 {replicate} output",
    )
    if (
        (root / INCOMPLETE_MARKER).exists()
        or (root / "FAILURE.json").exists()
    ):
        raise RuntimeError(f"CMIF P0 {replicate} is not complete")
    complete_path = root / "COMPLETE.json"
    if complete_path.is_symlink():
        raise RuntimeError(f"CMIF P0 {replicate} COMPLETE is a symlink")
    complete = _strict_json(
        complete_path.resolve(strict=True),
        name=f"CMIF P0 {replicate} COMPLETE",
    )
    complete_without_fingerprint = dict(complete)
    recorded_complete_fingerprint = complete_without_fingerprint.pop(
        "complete_fingerprint",
        None,
    )
    if (
        not isinstance(recorded_complete_fingerprint, str)
        or stable_fingerprint(complete_without_fingerprint)
        != recorded_complete_fingerprint
        or complete.get("schema_version") != COMPLETE_SCHEMA
        or complete.get("status") != "complete"
        or complete.get("replicate") != replicate
        or complete.get("split") != FROZEN_SPLIT
        or complete.get("D_V_accessed") is not False
        or complete.get("D_T_accessed") is not False
        or complete.get("training_performed") is not False
    ):
        raise RuntimeError(f"CMIF P0 {replicate} COMPLETE changed")
    expected_artifacts = complete.get("artifact_files")
    if (
        not isinstance(expected_artifacts, Mapping)
        or dict(expected_artifacts) != _artifact_hashes(root)
        or complete.get("artifact_file_count")
        != len(expected_artifacts)
    ):
        raise RuntimeError(
            f"CMIF P0 {replicate} artifact population changed"
        )
    core_path = root / "receipts" / "p0_core.json"
    if core_path.is_symlink() or not core_path.is_file():
        raise RuntimeError(f"CMIF P0 {replicate} core receipt is missing")
    core_payload = _strict_json(
        core_path,
        name=f"CMIF P0 {replicate} core receipt",
    )
    core_bytes = core_path.read_bytes()
    if core_bytes != _json_bytes(core_payload):
        raise RuntimeError(
            f"CMIF P0 {replicate} core receipt is not canonical"
        )
    core_sha256 = sha256(core_bytes).hexdigest()
    if (
        complete.get("p0_core_file_sha256") != core_sha256
        or complete.get("p0_core_receipt_fingerprint")
        != core_payload.get("receipt_fingerprint")
    ):
        raise RuntimeError(f"CMIF P0 {replicate} core receipt changed")
    return complete, core_bytes


def _decision_payload(
    *,
    replicate: str,
    single_eligible: bool,
    single_receipt_fingerprint: str,
    persisted_replay_payload: Mapping[str, object] | None,
    persisted_r1_byte_identity: bool,
) -> dict[str, object]:
    if replicate == "r1":
        bounded = False
        status = (
            "CMIF_V17_P0_REPLAY_PENDING"
            if single_eligible
            else "CMIF_V17_P0_FAIL"
        )
    else:
        if persisted_replay_payload is None:
            raise ValueError(
                "r2 decision requires persisted replay evidence"
            )
        replay_checks = persisted_replay_payload.get("checks")
        if not isinstance(replay_checks, Mapping):
            raise ValueError("persisted replay checks are malformed")
        bounded = bool(
            single_eligible
            and persisted_r1_byte_identity
            and persisted_replay_payload.get(
                "persisted_replay_passed"
            )
            is True
            and all(
                value is True
                for value in replay_checks.values()
            )
        )
        status = (
            "CMIF_V17_P0_PASS"
            if bounded
            else "CMIF_V17_P0_FAIL"
        )
    return _fingerprinted(
        {
            "schema_version": DECISION_SCHEMA,
            "status": status,
            "replicate": replicate,
            "single_run_eligible_for_replay": single_eligible,
            "single_receipt_fingerprint": single_receipt_fingerprint,
            "persisted_r1_byte_identity": (
                persisted_r1_byte_identity
                if replicate == "r2"
                else None
            ),
            "persisted_replay_comparison_fingerprint": (
                persisted_replay_payload.get("receipt_fingerprint")
                if persisted_replay_payload is not None
                else None
            ),
            "bounded_400_authorized": bounded,
            "training_authorized": bounded,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "performance_supported": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "training_performed": False,
        }
    )


def validate_create_only() -> dict[str, object]:
    """Validate static P0 inputs without loading cached D_R tensors."""

    source_paths = _verify_frozen_real_dr_sources()
    source_binding, protocol, _, _ = bind_coverage_state_real_dr_sources(
        **source_paths
    )
    if (
        source_binding.dataset != FROZEN_DATASET
        or source_binding.split != FROZEN_SPLIT
        or protocol.dataset != FROZEN_DATASET
        or protocol.split != FROZEN_SPLIT
        or source_binding.binding_fingerprint
        != FROZEN_REAL_DR_SOURCE_BINDING_FINGERPRINT
    ):
        raise RuntimeError("frozen real D_R semantic binding changed")
    parent = _verify_parent_v16_closure()
    dataset_free = run_coverage_state_cmif_dataset_free_gate()
    if not dataset_free.all_pass:
        raise RuntimeError("CMIF dataset-free gate did not pass")
    implementation = _static_implementation_binding()
    output_status: dict[str, object] = {}
    for replicate, relative in REPLICATE_OUTPUT_REPO_PATHS.items():
        output = Path(os.path.abspath(_ROOT / relative))
        exists = output.exists() or output.is_symlink()
        output_status[replicate] = {
            "repo_path": relative,
            "exists": exists,
            "available": not exists,
        }
        if not exists:
            _prepare_fixed_output(replicate)
    p0_core_present = (_ROOT / P0_CORE_REPO_PATH).is_file()
    r1_exists = bool(
        dict(output_status["r1"]).get("exists")
    )
    r2_exists = bool(
        dict(output_status["r2"]).get("exists")
    )
    r1_complete_valid = False
    if r1_exists and not r2_exists:
        try:
            _verify_complete_replicate("r1")
        except (OSError, RuntimeError, ValueError):
            r1_complete_valid = False
        else:
            r1_complete_valid = True
    r1_run_available = (
        p0_core_present and not r1_exists and not r2_exists
    )
    r2_run_available = (
        p0_core_present
        and r1_complete_valid
        and not r2_exists
    )
    return _fingerprinted(
        {
            "schema_version": VALIDATION_SCHEMA,
            "mode": "validate_create_only",
            "static_contract_valid": True,
            "dataset": FROZEN_DATASET,
            "split": FROZEN_SPLIT,
            "runtime_splits": [FROZEN_SPLIT],
            "frozen_real_D_R_inputs": {
                name: {
                    "repo_path": relative,
                    "file_sha256": expected_sha256,
                }
                for name, relative, expected_sha256
                in FROZEN_REAL_DR_INPUTS
            },
            "real_D_R_source_binding": (
                source_binding.canonical_payload()
            ),
            "real_D_R_source_binding_fingerprint": (
                source_binding.binding_fingerprint
            ),
            "parent_v16_negative_result": parent,
            "dataset_free_receipt_fingerprint": (
                dataset_free.receipt_fingerprint
            ),
            "dataset_free_evidence_fingerprint": (
                dataset_free.evidence_fingerprint
            ),
            "dataset_free_gate_passed": dataset_free.all_pass,
            "implementation_files": dict(implementation),
            "implementation_fingerprint": stable_fingerprint(
                dict(implementation)
            ),
            "reserved_outputs": output_status,
            "p0_core_repo_path": P0_CORE_REPO_PATH,
            "p0_core_present": p0_core_present,
            "run_once_implemented": True,
            "r1_run_available": r1_run_available,
            "r2_requires_complete_r1": True,
            "r1_complete_valid": r1_complete_valid,
            "r2_run_available": r2_run_available,
            "run_once_available": (
                r1_run_available or r2_run_available
            ),
            "output_claimed": False,
            "D_R_cached_tensor_payload_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "training_performed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "bounded_400_authorized": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "not_a_P0_result": True,
        }
    )


def _single_core_payload(single: object) -> dict[str, object]:
    payload = dict(single.canonical_payload())
    if "receipt_fingerprint" in payload:
        raise RuntimeError("single P0 payload already has a fingerprint")
    payload["receipt_fingerprint"] = single.receipt_fingerprint
    return payload


def run_once(replicate: str) -> dict[str, object]:
    """Run one fixed create-only replicate without training."""

    if replicate not in REPLICATE_OUTPUT_REPO_PATHS:
        raise ValueError("replicate must be exactly r1 or r2")
    if replicate == "r1":
        if any(
            (_ROOT / relative).exists()
            or (_ROOT / relative).is_symlink()
            for relative in REPLICATE_OUTPUT_REPO_PATHS.values()
        ):
            raise FileExistsError(
                "r1 requires both fixed P0 outputs to be unused"
            )
        persisted_r1_complete = None
        persisted_r1_bytes = None
    else:
        persisted_r1_complete, persisted_r1_bytes = (
            _verify_complete_replicate("r1")
        )
        if (
            persisted_r1_complete.get("replicate") != "r1"
            or persisted_r1_complete.get("bounded_400_authorized")
            is not False
        ):
            raise RuntimeError("r1 single-run boundary changed")
        _prepare_fixed_output("r2")

    source_paths = _verify_frozen_real_dr_sources()
    source_binding, protocol, _, _ = bind_coverage_state_real_dr_sources(
        **source_paths
    )
    if (
        source_binding.dataset != FROZEN_DATASET
        or source_binding.split != FROZEN_SPLIT
        or protocol.dataset != FROZEN_DATASET
        or protocol.split != FROZEN_SPLIT
        or source_binding.binding_fingerprint
        != FROZEN_REAL_DR_SOURCE_BINDING_FINGERPRINT
    ):
        raise RuntimeError("frozen real D_R semantic binding changed")
    parent = _verify_parent_v16_closure()
    dataset_free = run_coverage_state_cmif_dataset_free_gate()
    if not dataset_free.all_pass:
        raise PermissionError("CMIF dataset-free gate did not pass")
    implementation = _static_implementation_binding()
    config = _fingerprinted(
        _static_config_payload(
            source_paths=source_paths,
            parent=parent,
            dataset_free_receipt_fingerprint=(
                dataset_free.receipt_fingerprint
            ),
            dataset_free_evidence_fingerprint=(
                dataset_free.evidence_fingerprint
            ),
            implementation=implementation,
        )
    )
    attempt = _fingerprinted(
        {
            "schema_version": ATTEMPT_SCHEMA,
            "replicate": replicate,
            "output_repo_path": (
                REPLICATE_OUTPUT_REPO_PATHS[replicate]
            ),
            "config_fingerprint": config["receipt_fingerprint"],
            "source_binding_fingerprint": (
                source_binding.binding_fingerprint
            ),
            "parent_v16_complete_fingerprint": (
                PARENT_V16_COMPLETE_FINGERPRINT
            ),
            "dataset_free_receipt_fingerprint": (
                dataset_free.receipt_fingerprint
            ),
            "implementation_fingerprint": stable_fingerprint(
                dict(implementation)
            ),
            "r1_complete_fingerprint": (
                persisted_r1_complete.get("complete_fingerprint")
                if persisted_r1_complete is not None
                else None
            ),
            "create_only": True,
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "bounded_400_authorized": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "training_performed": False,
        }
    )
    output, receipts = _claim_fixed_output(
        replicate,
        attempt=attempt,
    )
    try:
        _write_new_json(receipts / "config.json", config)
        real_inputs = build_coverage_state_real_dr_inputs(**source_paths)
        if (
            real_inputs.source_binding.binding_fingerprint
            != FROZEN_REAL_DR_SOURCE_BINDING_FINGERPRINT
        ):
            raise RuntimeError("real D_R build left the frozen binding")
        bounded_population = build_coverage_state_bounded_population(
            real_inputs.scalar_cache
        )
        bounded_population.verify_unchanged()
        inputs_receipt = _fingerprinted(
            {
                "schema_version": INPUTS_SCHEMA,
                "frozen_population_container": (
                    real_inputs.canonical_payload()
                ),
                "source_binding": (
                    real_inputs.source_binding.canonical_payload()
                ),
                "bounded_population": (
                    bounded_population.canonical_payload()
                ),
                "CMIF_model_input_representation": "phase_preserving",
                "legacy_scalar_observability_is_CMIF_authority": False,
                "split": FROZEN_SPLIT,
                "D_V_accessed": False,
                "D_T_accessed": False,
                "training_performed": False,
            }
        )
        _write_new_json(receipts / "inputs.json", inputs_receipt)
        dataset_free_receipt = _fingerprinted(
            {
                "schema_version": DATASET_FREE_SCHEMA,
                "dataset_free": dataset_free.canonical_payload(),
                "dataset_free_receipt_fingerprint": (
                    dataset_free.receipt_fingerprint
                ),
                "dataset_free_evidence_fingerprint": (
                    dataset_free.evidence_fingerprint
                ),
                "all_pass": dataset_free.all_pass,
                "bounded_400_authorized": False,
                "formal_800_authorized": False,
            }
        )
        _write_new_json(
            receipts / "dataset_free.json",
            dataset_free_receipt,
        )

        replay_comparison: dict[str, object] | None = None
        single = run_coverage_state_cmif_p0_single(
            dataset_free_receipt=dataset_free,
            real_inputs=real_inputs,
            bounded_population=bounded_population,
        )
        single.verify_unchanged()
        single_core_payload = _single_core_payload(single)
        current_core_bytes = _json_bytes(single_core_payload)
        persisted_r1_byte_identity = False
        if replicate == "r2":
            if persisted_r1_bytes is None:
                raise AssertionError("r2 lost the persisted r1 receipt")
            persisted_r1_core = json.loads(
                persisted_r1_bytes.decode("utf-8")
            )
            if not isinstance(persisted_r1_core, Mapping):
                raise RuntimeError("persisted r1 core is malformed")
            persisted_r1_byte_identity = (
                persisted_r1_bytes == current_core_bytes
            )
            r1_file_sha256 = sha256(persisted_r1_bytes).hexdigest()
            r2_file_sha256 = sha256(current_core_bytes).hexdigest()
            r1_receipt_fingerprint = persisted_r1_core.get(
                "receipt_fingerprint"
            )
            replay_checks = {
                "r1_COMPLETE_verified": True,
                "r1_single_eligible_for_replay": (
                    persisted_r1_core.get("eligible_for_replay")
                    is True
                ),
                "r2_single_eligible_for_replay": (
                    single.eligible_for_replay
                ),
                "p0_core_canonical_bytes_identical": (
                    persisted_r1_byte_identity
                ),
                "p0_core_file_sha256_identical": (
                    r1_file_sha256 == r2_file_sha256
                ),
                "p0_core_receipt_fingerprint_identical": (
                    r1_receipt_fingerprint
                    == single.receipt_fingerprint
                ),
            }
            replay_comparison = _fingerprinted(
                {
                    "schema_version": REPLAY_COMPARISON_SCHEMA,
                    "r1_output_repo_path": (
                        REPLICATE_OUTPUT_REPO_PATHS["r1"]
                    ),
                    "r2_output_repo_path": (
                        REPLICATE_OUTPUT_REPO_PATHS["r2"]
                    ),
                    "r1_p0_core_file_sha256": r1_file_sha256,
                    "r2_p0_core_file_sha256": r2_file_sha256,
                    "persisted_canonical_bytes_identical": (
                        persisted_r1_byte_identity
                    ),
                    "persisted_file_sha256_identical": (
                        r1_file_sha256 == r2_file_sha256
                    ),
                    "r1_p0_core_receipt_fingerprint": (
                        r1_receipt_fingerprint
                    ),
                    "r2_p0_core_receipt_fingerprint": (
                        single.receipt_fingerprint
                    ),
                    "checks": replay_checks,
                    "persisted_replay_passed": (
                        bool(replay_checks)
                        and all(replay_checks.values())
                    ),
                    "in_memory_replay_used": False,
                    "each_replicate_ran_exactly_one_single_audit": True,
                    "D_V_accessed": False,
                    "D_T_accessed": False,
                    "training_performed": False,
                }
            )

        core_bytes = _json_bytes(single_core_payload)
        _write_new_bytes(receipts / "p0_core.json", core_bytes)
        if replicate == "r2":
            if replay_comparison is None:
                raise AssertionError("r2 replay comparison is missing")
            _write_new_json(
                receipts / "replay_comparison.json",
                replay_comparison,
            )
        decision = _decision_payload(
            replicate=replicate,
            single_eligible=single.eligible_for_replay,
            single_receipt_fingerprint=single.receipt_fingerprint,
            persisted_replay_payload=replay_comparison,
            persisted_r1_byte_identity=persisted_r1_byte_identity,
        )
        _write_new_json(receipts / "decision.json", decision)

        real_inputs.verify_unchanged()
        bounded_population.verify_unchanged()
        single.verify_unchanged()
        if _verify_frozen_real_dr_sources() != source_paths:
            raise RuntimeError("frozen D_R source paths changed")
        if _verify_parent_v16_closure() != parent:
            raise RuntimeError("parent v16 closure changed")
        replay_dataset_free = (
            run_coverage_state_cmif_dataset_free_gate()
        )
        if (
            replay_dataset_free.receipt_fingerprint
            != dataset_free.receipt_fingerprint
            or replay_dataset_free.evidence_fingerprint
            != dataset_free.evidence_fingerprint
        ):
            raise RuntimeError("CMIF dataset-free receipt changed")
        if _static_implementation_binding() != implementation:
            raise RuntimeError("CMIF P0 implementation changed")

        artifacts = _artifact_hashes(output)
        complete = _fingerprinted(
            {
                "schema_version": COMPLETE_SCHEMA,
                "status": "complete",
                "replicate": replicate,
                "decision": decision["status"],
                "decision_fingerprint": (
                    decision["receipt_fingerprint"]
                ),
                "config_fingerprint": config["receipt_fingerprint"],
                "inputs_receipt_fingerprint": (
                    inputs_receipt["receipt_fingerprint"]
                ),
                "dataset_free_receipt_fingerprint": (
                    dataset_free_receipt["receipt_fingerprint"]
                ),
                "p0_core_receipt_fingerprint": (
                    single.receipt_fingerprint
                ),
                "p0_core_file_sha256": sha256(core_bytes).hexdigest(),
                "replay_comparison_fingerprint": (
                    replay_comparison["receipt_fingerprint"]
                    if replay_comparison is not None
                    else None
                ),
                "bounded_400_authorized": (
                    decision["bounded_400_authorized"]
                ),
                "formal_800_authorized": False,
                "full_CURE_authorized": False,
                "cross_backbone_authorized": False,
                "dataset": FROZEN_DATASET,
                "split": FROZEN_SPLIT,
                "runtime_splits": [FROZEN_SPLIT],
                "artifact_files": artifacts,
                "artifact_file_count": len(artifacts),
                "single_attempt": True,
                "resume_allowed": False,
                "automatic_retry_allowed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
                "training_performed": False,
                "calibration_performed": False,
                "inference_performed": False,
            },
            field="complete_fingerprint",
        )
        _write_new_json(output / "COMPLETE.json", complete)
        _fsync_directory(output)
        (output / INCOMPLETE_MARKER).unlink()
        _fsync_directory(output)
        return {
            "output": str(output),
            "replicate": replicate,
            "decision": decision["status"],
            "p0_core_file_sha256": sha256(core_bytes).hexdigest(),
            "p0_core_receipt_fingerprint": single.receipt_fingerprint,
            "complete_fingerprint": complete["complete_fingerprint"],
            "bounded_400_authorized": (
                decision["bounded_400_authorized"]
            ),
            "D_V_accessed": False,
            "D_T_accessed": False,
            "training_performed": False,
        }
    except BaseException as error:
        try:
            failure_path = output / "FAILURE.json"
            if not failure_path.exists() and not failure_path.is_symlink():
                _write_new_json(
                    failure_path,
                    _failure_payload(
                        error,
                        replicate=replicate,
                        attempt_fingerprint=str(
                            attempt["receipt_fingerprint"]
                        ),
                        artifact_files=_artifact_hashes(output),
                    ),
                )
                _fsync_directory(output)
        except BaseException:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--validate-create-only",
        action="store_true",
        help=(
            "validate frozen sources and reserved outputs without loading "
            "cached D_R tensors"
        ),
    )
    mode.add_argument(
        "--run-once",
        choices=tuple(sorted(REPLICATE_OUTPUT_REPO_PATHS)),
        metavar="{r1,r2}",
        help=(
            "consume exactly one fixed create-only P0 replicate; r2 "
            "requires a complete immutable r1"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = (
        validate_create_only()
        if args.validate_create_only
        else run_once(str(args.run_once))
    )
    print(
        json.dumps(
            result,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    )
    if (
        args.run_once == "r2"
        and result["bounded_400_authorized"] is not True
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
