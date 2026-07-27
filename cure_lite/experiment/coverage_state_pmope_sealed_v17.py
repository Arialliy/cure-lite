"""Read-only binding of the sealed v17 CMIF bounded controls for PMOPE.

The verifier treats the completed v17 run and its source closure as immutable
historical evidence.  It reads bytes, strict JSON, and tar members only.  It
never deserializes a checkpoint, constructs a model, calls an evaluator, or
enters a training path.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Iterable

from ..cache.schema import file_sha256, stable_fingerprint


COVERAGE_STATE_PMOPE_SEALED_V17_SCHEMA = (
    "cure-lite-pmope-v18-sealed-v17-controls-v1"
)
COVERAGE_STATE_PMOPE_V17_RUN_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/"
    "cure_lite_cmif_v17_support_oriented_bounded_400_r1"
)
COVERAGE_STATE_PMOPE_V17_SOURCE_MANIFEST_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_cmif_v17_support_oriented_bounded_400_50a9963ae620.json"
)
COVERAGE_STATE_PMOPE_V17_SOURCE_ARCHIVE_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_cmif_v17_support_oriented_bounded_400_50a9963ae620.tar"
)
COVERAGE_STATE_PMOPE_V17_COMPLETE_FINGERPRINT = (
    "50a9963ae620dc7140deebf604a4344f78af5560af2a1737d58efb256070aeb0"
)
COVERAGE_STATE_PMOPE_V17_COMPLETE_SHA256 = (
    "aed5ab56fa0ec786bd3ea684f44194fe5d581aeae7c61c320379e43ff489334b"
)
COVERAGE_STATE_PMOPE_V17_DECISION_FINGERPRINT = (
    "a7760aa8028b9ac8e4fc8657af5d7dae8422f28744abd505c0db095837a69fa4"
)
COVERAGE_STATE_PMOPE_V17_RESULT_FINGERPRINT = (
    "5dfcb5a5674e3941b30245fc6612b4c75699edefc70566cb1ebaad117d7abcbf"
)
COVERAGE_STATE_PMOPE_V17_SOURCE_MANIFEST_SHA256 = (
    "9d04e13ba781163e7114f33607d4e67f633eac9765ea8313676200fe1f906d98"
)
COVERAGE_STATE_PMOPE_V17_SOURCE_ARCHIVE_SHA256 = (
    "4f333efe993151c36dcab80d83f266387d4a8e0ade0059b3a2538522a20f532c"
)
COVERAGE_STATE_PMOPE_V17_ARTIFACT_COUNT = 17
COVERAGE_STATE_PMOPE_V17_SOURCE_FILE_COUNT = 40
COVERAGE_STATE_PMOPE_V17_CONTROL_OBJECTIVES = (
    "support_oriented_response_joint",
    "identity_joint",
    "separable_endpoint",
)
COVERAGE_STATE_PMOPE_V17_ARTIFACT_PATHS = (
    "attempt.json",
    "checkpoints/identity_joint.checkpoint.json",
    "checkpoints/identity_joint.safetensors",
    "checkpoints/separable_endpoint.checkpoint.json",
    "checkpoints/separable_endpoint.safetensors",
    "checkpoints/support_oriented_response_joint.checkpoint.json",
    "checkpoints/support_oriented_response_joint.safetensors",
    "receipts/authorization.json",
    "receipts/bounded_result.json",
    "receipts/config.json",
    "receipts/dataset_free.json",
    "receipts/decision.json",
    "receipts/device_memory_preflight.json",
    "receipts/inputs.json",
    "receipts/preflight.json",
    "receipts/training.json",
    "receipts/zero_level.json",
)

_EXPECTED_RUN_ID = (
    "cure_lite_cmif_v17_support_oriented_bounded_400_r1"
)
_EXPECTED_COMPLETE_SCHEMA = (
    "cure-lite-cmif-v17-support-oriented-bounded-400-run-v1"
)
_EXPECTED_DECISION_SCHEMA = (
    "cure-lite-cmif-v17-support-oriented-bounded-400-decision-v1"
)
_EXPECTED_CHECKPOINT_SCHEMA = (
    "cure-lite-cmif-v17-support-oriented-bounded-400-checkpoint-v1"
)
_EXPECTED_SOURCE_CLOSURE_SCHEMA = "cure-lite-cmif-source-closure-v1"
_EXPECTED_SOURCE_MANIFEST_POINTER = (
    "receipts/config.json:implementation.files"
)
_EXPECTED_MODEL_CLASS = "CURELiteCenteredMixedInteractionLevelSet"
_EXPECTED_PARAMETER_COUNT = 64064
_EXPECTED_STATE_KEYS = (
    "joint_hidden_bias",
    "joint_state_weight",
    "scalar_energy_weight",
)


def _reject_duplicate_keys(
    pairs: Iterable[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{name} must be a regular non-symlink file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must contain a JSON object")
    return value


def _safe_relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{name} must be a nonempty relative path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise RuntimeError(f"{name} is not a safe canonical relative path")
    return value


def _required_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"{name} must be a lowercase SHA256 digest")
    return value


def _verify_embedded_fingerprint(
    payload: dict[str, Any],
    *,
    field: str,
    name: str,
) -> str:
    fingerprint = _required_sha256(payload.get(field), name=f"{name}.{field}")
    unsigned = dict(payload)
    unsigned.pop(field)
    if stable_fingerprint(unsigned) != fingerprint:
        raise RuntimeError(f"{name} embedded fingerprint changed")
    return fingerprint


def _assert_all_split_access_false(
    value: object,
    *,
    location: str,
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"D_V_accessed", "D_T_accessed"} and item is not False:
                raise RuntimeError(f"{location}.{key} must remain false")
            _assert_all_split_access_false(
                item,
                location=f"{location}.{key}",
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_all_split_access_false(
                item,
                location=f"{location}[{index}]",
            )


def _regular_tree_files(root: Path) -> tuple[str, ...]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("sealed v17 run must be a non-symlink directory")
    values: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("sealed v17 run contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError("sealed v17 run contains a non-regular entry")
        values.append(path.relative_to(root).as_posix())
    return tuple(sorted(values))


def _archive_member_hashes(
    archive_path: Path,
) -> tuple[tuple[str, str], ...]:
    if not archive_path.is_file() or archive_path.is_symlink():
        raise RuntimeError("v17 source archive must be a regular file")
    result: dict[str, str] = {}
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            for member in archive.getmembers():
                name = _safe_relative_path(
                    member.name,
                    name="source archive member",
                )
                if (
                    name in result
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                ):
                    raise RuntimeError(
                        "v17 source archive contains a duplicate or "
                        "non-regular member"
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise RuntimeError(
                        "v17 source archive member cannot be read"
                    )
                content = handle.read()
                if len(content) != member.size:
                    raise RuntimeError(
                        "v17 source archive member size changed"
                    )
                result[name] = sha256(content).hexdigest()
    except (tarfile.TarError, OSError) as error:
        raise RuntimeError("v17 source archive is invalid") from error
    return tuple(sorted(result.items()))


@dataclass(frozen=True)
class CoverageStatePMOPESealedV17Control:
    """One historical checkpoint and its already-computed diagnostic."""

    objective: str
    objective_policy: str
    final_model_fingerprint: str
    module_state_fingerprint: str
    checkpoint_file_sha256: str
    checkpoint_receipt_fingerprint: str
    zero_level_checkpoint_fingerprint: str
    bounded_gate_passed: bool

    def canonical_payload(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "objective_policy": self.objective_policy,
            "final_model_fingerprint": self.final_model_fingerprint,
            "module_state_fingerprint": self.module_state_fingerprint,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "checkpoint_receipt_fingerprint": (
                self.checkpoint_receipt_fingerprint
            ),
            "zero_level_checkpoint_fingerprint": (
                self.zero_level_checkpoint_fingerprint
            ),
            "bounded_gate_passed": self.bounded_gate_passed,
        }


@dataclass(frozen=True)
class CoverageStatePMOPESealedV17Receipt:
    """Canonical proof that the three v17 controls remain sealed."""

    run_repo_path: str
    complete_fingerprint: str
    complete_file_sha256: str
    decision_fingerprint: str
    bounded_result_fingerprint: str
    source_manifest_file_sha256: str
    source_archive_file_sha256: str
    source_implementation_fingerprint: str
    artifact_files: tuple[tuple[str, str], ...]
    source_members: tuple[tuple[str, str], ...]
    controls: tuple[CoverageStatePMOPESealedV17Control, ...]
    checks: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        if (
            self.run_repo_path != COVERAGE_STATE_PMOPE_V17_RUN_REPO_PATH
            or self.complete_fingerprint
            != COVERAGE_STATE_PMOPE_V17_COMPLETE_FINGERPRINT
            or self.complete_file_sha256
            != COVERAGE_STATE_PMOPE_V17_COMPLETE_SHA256
            or self.decision_fingerprint
            != COVERAGE_STATE_PMOPE_V17_DECISION_FINGERPRINT
            or self.bounded_result_fingerprint
            != COVERAGE_STATE_PMOPE_V17_RESULT_FINGERPRINT
            or self.source_manifest_file_sha256
            != COVERAGE_STATE_PMOPE_V17_SOURCE_MANIFEST_SHA256
            or self.source_archive_file_sha256
            != COVERAGE_STATE_PMOPE_V17_SOURCE_ARCHIVE_SHA256
            or len(self.artifact_files)
            != COVERAGE_STATE_PMOPE_V17_ARTIFACT_COUNT
            or len(self.source_members)
            != COVERAGE_STATE_PMOPE_V17_SOURCE_FILE_COUNT
            or tuple(value.objective for value in self.controls)
            != COVERAGE_STATE_PMOPE_V17_CONTROL_OBJECTIVES
            or self.checks != tuple(sorted(self.checks))
            or not self.checks
            or not all(value for _, value in self.checks)
        ):
            raise ValueError("sealed v17 receipt binding changed")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": COVERAGE_STATE_PMOPE_SEALED_V17_SCHEMA,
            "run_repo_path": self.run_repo_path,
            "complete_fingerprint": self.complete_fingerprint,
            "complete_file_sha256": self.complete_file_sha256,
            "decision_fingerprint": self.decision_fingerprint,
            "bounded_result_fingerprint": (
                self.bounded_result_fingerprint
            ),
            "source_closure": {
                "manifest_file_sha256": (
                    self.source_manifest_file_sha256
                ),
                "archive_file_sha256": (
                    self.source_archive_file_sha256
                ),
                "implementation_fingerprint": (
                    self.source_implementation_fingerprint
                ),
                "source_members": dict(self.source_members),
            },
            "artifact_files": dict(self.artifact_files),
            "controls": [
                value.canonical_payload() for value in self.controls
            ],
            "checks": dict(self.checks),
            "all_pass": True,
            "historical_frozen_controls": True,
            "contemporaneous_controls": False,
            "control_outcomes_are_not_candidate_gates": True,
            "verification_mode": "read_only_bytes_strict_json_and_tar_members",
            "model_deserialization_performed": False,
            "evaluator_called": False,
            "training_performed": False,
            "D_R_cached_tensor_payload_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "runtime_splits": [],
        }

    @property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def verify_coverage_state_pmope_sealed_v17_controls(
    run_path: str | Path,
    *,
    source_manifest_path: str | Path,
    source_archive_path: str | Path,
) -> CoverageStatePMOPESealedV17Receipt:
    """Strictly verify the relocated or in-repository sealed v17 evidence."""

    root = Path(run_path)
    manifest_path = Path(source_manifest_path)
    archive_path = Path(source_archive_path)
    complete_path = root / "COMPLETE.json"
    complete = _strict_json(complete_path, name="v17 COMPLETE")
    complete_fingerprint = _verify_embedded_fingerprint(
        complete,
        field="complete_fingerprint",
        name="v17 COMPLETE",
    )
    complete_sha256 = file_sha256(complete_path)
    if (
        complete_fingerprint
        != COVERAGE_STATE_PMOPE_V17_COMPLETE_FINGERPRINT
        or complete_sha256 != COVERAGE_STATE_PMOPE_V17_COMPLETE_SHA256
        or complete.get("schema_version") != _EXPECTED_COMPLETE_SCHEMA
        or complete.get("run_id") != _EXPECTED_RUN_ID
        or complete.get("status") != "complete"
        or complete.get("decision")
        != "CMIF_V17_BOUNDED_400_GATE_FAIL"
        or complete.get("bounded_gate_passed") is not False
        or complete.get("D_V_accessed") is not False
        or complete.get("D_T_accessed") is not False
        or complete.get("artifact_file_count")
        != COVERAGE_STATE_PMOPE_V17_ARTIFACT_COUNT
    ):
        raise RuntimeError("sealed v17 COMPLETE binding changed")

    artifact_mapping = complete.get("artifact_files")
    if not isinstance(artifact_mapping, dict):
        raise RuntimeError("sealed v17 artifact map is missing")
    artifact_files = tuple(
        sorted(
            (
                _safe_relative_path(key, name="v17 artifact path"),
                _required_sha256(value, name=f"artifact hash {key}"),
            )
            for key, value in artifact_mapping.items()
        )
    )
    if tuple(path for path, _ in artifact_files) != (
        COVERAGE_STATE_PMOPE_V17_ARTIFACT_PATHS
    ):
        raise RuntimeError("sealed v17 artifact set changed")
    actual_tree = _regular_tree_files(root)
    if actual_tree != tuple(
        sorted((*COVERAGE_STATE_PMOPE_V17_ARTIFACT_PATHS, "COMPLETE.json"))
    ):
        raise RuntimeError("sealed v17 run tree is not exact")

    parsed_artifacts: dict[str, dict[str, Any]] = {}
    for relative, expected_hash in artifact_files:
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or file_sha256(path) != expected_hash
        ):
            raise RuntimeError(f"sealed v17 artifact hash changed: {relative}")
        if relative.endswith(".json"):
            payload = _strict_json(path, name=f"v17 artifact {relative}")
            _verify_embedded_fingerprint(
                payload,
                field="receipt_fingerprint",
                name=f"v17 artifact {relative}",
            )
            _assert_all_split_access_false(
                payload,
                location=relative,
            )
            parsed_artifacts[relative] = payload

    config = parsed_artifacts["receipts/config.json"]
    training_receipt = parsed_artifacts["receipts/training.json"]
    zero_receipt = parsed_artifacts["receipts/zero_level.json"]
    bounded_receipt = parsed_artifacts["receipts/bounded_result.json"]
    decision = parsed_artifacts["receipts/decision.json"]
    if (
        config.get("run_id") != _EXPECTED_RUN_ID
        or config.get("output_repo_path")
        != COVERAGE_STATE_PMOPE_V17_RUN_REPO_PATH
        or config.get("model", {}).get("objective_suite")
        != list(COVERAGE_STATE_PMOPE_V17_CONTROL_OBJECTIVES)
        or config.get("budget", {}).get("seed") != 42
        or config.get("budget", {}).get("epochs") != 10
        or config.get("budget", {}).get("steps_per_epoch") != 40
        or config.get("budget", {}).get("updates_per_objective") != 400
        or config.get("budget", {}).get("objectives") != 3
    ):
        raise RuntimeError("sealed v17 configuration changed")

    bounded_result_fingerprint = _required_sha256(
        bounded_receipt.get("result_fingerprint"),
        name="v17 bounded result fingerprint",
    )
    bounded_payload = bounded_receipt.get("result")
    if (
        bounded_result_fingerprint
        != COVERAGE_STATE_PMOPE_V17_RESULT_FINGERPRINT
        or not isinstance(bounded_payload, dict)
        or stable_fingerprint(bounded_payload)
        != bounded_result_fingerprint
        or decision.get("schema_version") != _EXPECTED_DECISION_SCHEMA
        or decision.get("receipt_fingerprint")
        != COVERAGE_STATE_PMOPE_V17_DECISION_FINGERPRINT
        or decision.get("result_fingerprint")
        != bounded_result_fingerprint
        or decision.get("status")
        != "CMIF_V17_BOUNDED_400_GATE_FAIL"
        or decision.get("bounded_gate_passed") is not False
    ):
        raise RuntimeError("sealed v17 decision/result binding changed")

    training = training_receipt.get("training")
    diagnostics = zero_receipt.get("diagnostics")
    if (
        not isinstance(training, dict)
        or training.get("objective_suite")
        != list(COVERAGE_STATE_PMOPE_V17_CONTROL_OBJECTIVES)
        or not isinstance(training.get("objectives"), list)
        or not isinstance(diagnostics, dict)
        or tuple(diagnostics) != (
            "identity_joint",
            "separable_endpoint",
            "support_oriented_response_joint",
        )
    ):
        raise RuntimeError("sealed v17 training/diagnostic suite changed")
    training_by_objective = {
        value.get("objective"): value
        for value in training["objectives"]
        if isinstance(value, dict)
    }
    if tuple(training_by_objective) != (
        COVERAGE_STATE_PMOPE_V17_CONTROL_OBJECTIVES
    ):
        raise RuntimeError("sealed v17 objective ledger changed")

    decision_checkpoint_receipts = decision.get(
        "checkpoint_receipt_fingerprints"
    )
    training_checkpoint_receipts = training_receipt.get(
        "checkpoint_receipt_fingerprints"
    )
    if (
        not isinstance(decision_checkpoint_receipts, dict)
        or not isinstance(training_checkpoint_receipts, dict)
        or decision_checkpoint_receipts != training_checkpoint_receipts
    ):
        raise RuntimeError("sealed v17 checkpoint receipt map changed")

    controls: list[CoverageStatePMOPESealedV17Control] = []
    for objective in COVERAGE_STATE_PMOPE_V17_CONTROL_OBJECTIVES:
        checkpoint_relative = (
            f"checkpoints/{objective}.checkpoint.json"
        )
        weights_relative = f"checkpoints/{objective}.safetensors"
        checkpoint = parsed_artifacts[checkpoint_relative]
        objective_training = training_by_objective.get(objective)
        diagnostic = diagnostics.get(objective)
        if (
            not isinstance(objective_training, dict)
            or not isinstance(diagnostic, dict)
            or checkpoint.get("schema_version")
            != _EXPECTED_CHECKPOINT_SCHEMA
            or checkpoint.get("objective") != objective
            or checkpoint.get("model_class") != _EXPECTED_MODEL_CLASS
            or checkpoint.get("model_config", {}).get("parameter_count")
            != _EXPECTED_PARAMETER_COUNT
            or checkpoint.get("serialization") != "safetensors"
            or checkpoint.get("tensor_only_state_dict") is not True
            or checkpoint.get("weights_only_roundtrip_verified") is not True
            or tuple(checkpoint.get("state_keys", ()))
            != _EXPECTED_STATE_KEYS
            or checkpoint.get("checkpoint_file_sha256")
            != dict(artifact_files)[weights_relative]
            or checkpoint.get("receipt_fingerprint")
            != decision_checkpoint_receipts.get(objective)
            or diagnostic.get("checkpoint_fingerprint")
            != checkpoint.get("module_state_fingerprint")
            or objective_training.get("completed_updates") != 400
            or objective_training.get("seed") != 42
            or objective_training.get("compute", {}).get(
                "optimizer_steps"
            )
            != 400
        ):
            raise RuntimeError(
                f"sealed v17 control binding changed: {objective}"
            )
        gates = diagnostic.get("gates")
        if (
            not isinstance(gates, dict)
            or not isinstance(gates.get("bounded_gate_passed"), bool)
        ):
            raise RuntimeError(
                f"sealed v17 control gates changed: {objective}"
            )
        controls.append(
            CoverageStatePMOPESealedV17Control(
                objective=objective,
                objective_policy=str(
                    checkpoint.get("objective_policy")
                ),
                final_model_fingerprint=_required_sha256(
                    objective_training.get("final_model_fingerprint"),
                    name=f"{objective} final model fingerprint",
                ),
                module_state_fingerprint=_required_sha256(
                    checkpoint.get("module_state_fingerprint"),
                    name=f"{objective} module state fingerprint",
                ),
                checkpoint_file_sha256=_required_sha256(
                    checkpoint.get("checkpoint_file_sha256"),
                    name=f"{objective} checkpoint file hash",
                ),
                checkpoint_receipt_fingerprint=_required_sha256(
                    checkpoint.get("receipt_fingerprint"),
                    name=f"{objective} checkpoint receipt",
                ),
                zero_level_checkpoint_fingerprint=_required_sha256(
                    diagnostic.get("checkpoint_fingerprint"),
                    name=f"{objective} diagnostic checkpoint",
                ),
                bounded_gate_passed=bool(
                    gates["bounded_gate_passed"]
                ),
            )
        )

    if (
        complete.get("decision_fingerprint")
        != COVERAGE_STATE_PMOPE_V17_DECISION_FINGERPRINT
        or complete.get("bounded_result_receipt_fingerprint")
        != bounded_receipt.get("receipt_fingerprint")
        or complete.get("training_receipt_fingerprint")
        != training_receipt.get("receipt_fingerprint")
        or complete.get("zero_level_receipt_fingerprint")
        != zero_receipt.get("receipt_fingerprint")
    ):
        raise RuntimeError("sealed v17 COMPLETE receipt links changed")

    source_manifest = _strict_json(
        manifest_path,
        name="v17 source closure manifest",
    )
    manifest_sha256 = file_sha256(manifest_path)
    archive_sha256 = file_sha256(archive_path)
    source_implementation = config.get("implementation")
    implementation_files = (
        source_implementation.get("files")
        if isinstance(source_implementation, dict)
        else None
    )
    if (
        manifest_sha256
        != COVERAGE_STATE_PMOPE_V17_SOURCE_MANIFEST_SHA256
        or archive_sha256
        != COVERAGE_STATE_PMOPE_V17_SOURCE_ARCHIVE_SHA256
        or source_manifest.get("schema_version")
        != _EXPECTED_SOURCE_CLOSURE_SCHEMA
        or source_manifest.get("run_repo_path")
        != COVERAGE_STATE_PMOPE_V17_RUN_REPO_PATH
        or source_manifest.get("complete_fingerprint")
        != COVERAGE_STATE_PMOPE_V17_COMPLETE_FINGERPRINT
        or source_manifest.get("complete_file_sha256")
        != COVERAGE_STATE_PMOPE_V17_COMPLETE_SHA256
        or source_manifest.get("archive_repo_path")
        != COVERAGE_STATE_PMOPE_V17_SOURCE_ARCHIVE_REPO_PATH
        or source_manifest.get("archive_sha256")
        != COVERAGE_STATE_PMOPE_V17_SOURCE_ARCHIVE_SHA256
        or source_manifest.get("archive_bytes") != archive_path.stat().st_size
        or source_manifest.get("source_file_count")
        != COVERAGE_STATE_PMOPE_V17_SOURCE_FILE_COUNT
        or source_manifest.get("source_manifest")
        != _EXPECTED_SOURCE_MANIFEST_POINTER
        or not isinstance(implementation_files, dict)
        or source_manifest.get("implementation_fingerprint")
        != source_implementation.get("implementation_fingerprint")
        or source_manifest.get("config_receipt_sha256")
        != dict(artifact_files)["receipts/config.json"]
        or source_manifest.get("training_receipt_sha256")
        != dict(artifact_files)["receipts/training.json"]
        or source_manifest.get("zero_level_receipt_sha256")
        != dict(artifact_files)["receipts/zero_level.json"]
        or source_manifest.get("bounded_result_receipt_sha256")
        != dict(artifact_files)["receipts/bounded_result.json"]
        or source_manifest.get("decision_receipt_sha256")
        != dict(artifact_files)["receipts/decision.json"]
    ):
        raise RuntimeError("sealed v17 source closure manifest changed")
    source_members = _archive_member_hashes(archive_path)
    normalized_implementation_files = tuple(
        sorted(
            (
                _safe_relative_path(
                    path,
                    name="v17 implementation source path",
                ),
                _required_sha256(
                    digest,
                    name=f"v17 implementation source hash {path}",
                ),
            )
            for path, digest in implementation_files.items()
        )
    )
    if (
        len(source_members)
        != COVERAGE_STATE_PMOPE_V17_SOURCE_FILE_COUNT
        or source_members != normalized_implementation_files
    ):
        raise RuntimeError("sealed v17 source archive members changed")

    checks = tuple(
        sorted(
            {
                "artifact_set_and_hashes_exact": True,
                "all_json_fingerprints_exact": True,
                "complete_identity_bound": True,
                "decision_and_result_bound": True,
                "three_checkpoint_files_and_receipts_bound": True,
                "historical_budget_seed42_400_bound": True,
                "source_manifest_identity_bound": True,
                "source_archive_identity_bound": True,
                "source_archive_members_match_implementation": True,
                "D_V_not_accessed": True,
                "D_T_not_accessed": True,
                "no_model_deserialization": True,
                "no_evaluator_call": True,
                "no_training": True,
            }.items()
        )
    )
    return CoverageStatePMOPESealedV17Receipt(
        run_repo_path=str(source_manifest["run_repo_path"]),
        complete_fingerprint=complete_fingerprint,
        complete_file_sha256=complete_sha256,
        decision_fingerprint=str(decision["receipt_fingerprint"]),
        bounded_result_fingerprint=bounded_result_fingerprint,
        source_manifest_file_sha256=manifest_sha256,
        source_archive_file_sha256=archive_sha256,
        source_implementation_fingerprint=_required_sha256(
            source_manifest.get("implementation_fingerprint"),
            name="v17 source implementation fingerprint",
        ),
        artifact_files=artifact_files,
        source_members=source_members,
        controls=tuple(controls),
        checks=checks,
    )


__all__ = [
    "COVERAGE_STATE_PMOPE_SEALED_V17_SCHEMA",
    "COVERAGE_STATE_PMOPE_V17_ARTIFACT_COUNT",
    "COVERAGE_STATE_PMOPE_V17_ARTIFACT_PATHS",
    "COVERAGE_STATE_PMOPE_V17_COMPLETE_FINGERPRINT",
    "COVERAGE_STATE_PMOPE_V17_COMPLETE_SHA256",
    "COVERAGE_STATE_PMOPE_V17_CONTROL_OBJECTIVES",
    "COVERAGE_STATE_PMOPE_V17_DECISION_FINGERPRINT",
    "COVERAGE_STATE_PMOPE_V17_RESULT_FINGERPRINT",
    "COVERAGE_STATE_PMOPE_V17_RUN_REPO_PATH",
    "COVERAGE_STATE_PMOPE_V17_SOURCE_ARCHIVE_REPO_PATH",
    "COVERAGE_STATE_PMOPE_V17_SOURCE_ARCHIVE_SHA256",
    "COVERAGE_STATE_PMOPE_V17_SOURCE_FILE_COUNT",
    "COVERAGE_STATE_PMOPE_V17_SOURCE_MANIFEST_REPO_PATH",
    "COVERAGE_STATE_PMOPE_V17_SOURCE_MANIFEST_SHA256",
    "CoverageStatePMOPESealedV17Control",
    "CoverageStatePMOPESealedV17Receipt",
    "verify_coverage_state_pmope_sealed_v17_controls",
]
