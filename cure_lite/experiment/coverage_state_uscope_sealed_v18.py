"""Read-only binding of the sealed v18 PMOPE negative result for v19.

This module verifies only immutable file bytes, strict JSON contracts, and
the source-closure tar.  In particular, the PMOPE checkpoint is never
deserialized: its ``.safetensors`` file is treated as an opaque byte stream
whose SHA256 digest is bound by ``COMPLETE.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any, Iterable

from ..cache.schema import file_sha256, stable_fingerprint


COVERAGE_STATE_USCOPE_SEALED_V18_SCHEMA = (
    "cure-lite-uscope-v19-sealed-v18-negative-result-v1"
)
COVERAGE_STATE_USCOPE_V18_RUN_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/"
    "cure_lite_cmif_v18_pmope_bounded_400_r1"
)
COVERAGE_STATE_USCOPE_V18_SOURCE_MANIFEST_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_cmif_v18_pmope_bounded_400_bd791fd17a6e.json"
)
COVERAGE_STATE_USCOPE_V18_SOURCE_ARCHIVE_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_cmif_v18_pmope_bounded_400_bd791fd17a6e.tar"
)
COVERAGE_STATE_USCOPE_V18_COMPLETE_FINGERPRINT = (
    "bd791fd17a6eaf9884e0b590a0f5b8ebabb046016ee289b26e21ba81dd87645b"
)
COVERAGE_STATE_USCOPE_V18_COMPLETE_SHA256 = (
    "558abdf8eee405c44d5e65fcd379177412732d997ae5707c8c259cee683511ef"
)
COVERAGE_STATE_USCOPE_V18_DECISION_FINGERPRINT = (
    "0e388542ad035a1a9f8a5f747297c9f6834b775d0c7a2e86556697c95fb0fe58"
)
COVERAGE_STATE_USCOPE_V18_DECISION_SHA256 = (
    "bca61e6502498cf81e1431631f71e2fc9e1321e7bcee7ec745f4ee56dd044d79"
)
COVERAGE_STATE_USCOPE_V18_RESULT_FINGERPRINT = (
    "101e5ffc87ba534ce02fb37f280d3249f594b4b8b0e1a7cde04f3be35a649559"
)
COVERAGE_STATE_USCOPE_V18_RESULT_RECEIPT_FINGERPRINT = (
    "0e1445041251e6f260ba8e95251a0a63b2199f0eb19a24f3c23c64f62dcf1da3"
)
COVERAGE_STATE_USCOPE_V18_SOURCE_MANIFEST_SHA256 = (
    "c7320a0ea7c2393f4ddfc647ec32be9c2ae15ee01e8167b1f8550f8e48cdba33"
)
COVERAGE_STATE_USCOPE_V18_SOURCE_ARCHIVE_SHA256 = (
    "e32d32cc3ea0e01a5f823a69bc3250e0091dcfdda3ec7d45f2abce05c0bed1a0"
)
COVERAGE_STATE_USCOPE_V18_IMPLEMENTATION_FINGERPRINT = (
    "036b59ab0a96c44c27df88bdbaf1f019ba2d96752369bef165e298b220a94402"
)
COVERAGE_STATE_USCOPE_V18_SEALED_RECEIPT_FINGERPRINT = (
    "d66a2c9b8a7a5ced87e0265715ccd6cc3ca79fc91fdb3da29528185fa09f6d76"
)
COVERAGE_STATE_USCOPE_V18_ARTIFACT_COUNT = 15
COVERAGE_STATE_USCOPE_V18_SOURCE_FILE_COUNT = 41
COVERAGE_STATE_USCOPE_V18_ARTIFACT_PATHS = (
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

_EXPECTED_RUN_ID = "cure_lite_cmif_v18_pmope_bounded_400_r1"
_EXPECTED_COMPLETE_SCHEMA = (
    "cure-lite-cmif-v18-pmope-bounded-400-run-v1"
)
_EXPECTED_DECISION_SCHEMA = (
    "cure-lite-cmif-v18-pmope-bounded-400-decision-v1"
)
_EXPECTED_RESULT_SCHEMA = (
    "cure-lite-cmif-v18-pmope-bounded-400-result-v1"
)
_EXPECTED_CHECKPOINT_SCHEMA = (
    "cure-lite-cmif-v18-pmope-bounded-400-checkpoint-v1"
)
_EXPECTED_SOURCE_CLOSURE_SCHEMA = "cure-lite-cmif-source-closure-v1"
_EXPECTED_SOURCE_MANIFEST_POINTER = (
    "receipts/config.json:implementation.files"
)
_EXPECTED_DECISION = "PMOPE_V18_BOUNDED_400_GATE_FAIL"
_EXPECTED_NEXT_ACTION = (
    "freeze_pmope_v18_negative_result_and_review_structure"
)
_EXPECTED_OBJECTIVE = "pmope_joint"
_EXPECTED_OBJECTIVE_POLICY = (
    "paired_minimum_sdf_margin_target_orthant_projection_joint_w1p4_energy_v1"
)
_EXPECTED_MODEL_CLASS = "CURELiteCenteredMixedInteractionLevelSet"
_EXPECTED_PARAMETER_COUNT = 64064
_EXPECTED_STATE_KEYS = (
    "joint_hidden_bias",
    "joint_state_weight",
    "scalar_energy_weight",
)
_EXPECTED_CHECKPOINT_FILE_SHA256 = (
    "b171f452ad8bcf6cc739f756d2a233b35715551ff33d4a95c94bce30bb859ba8"
)
_EXPECTED_CHECKPOINT_RECEIPT_FINGERPRINT = (
    "aa636adffa306a49fcd3d7b27bd0c3aeb1819fdb88f2444393a12fc152bc4574"
)
_EXPECTED_FINAL_MODEL_FINGERPRINT = (
    "a3dc3a548e2f4a1c89505df15bcc2f1f0356ac9eb7ece0478753e7fb3734f51e"
)
_EXPECTED_MODULE_STATE_FINGERPRINT = (
    "c58d74d33df5a611fcb461c8f5276b2f479edac666be354f4800d752a2d14569"
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
        raise RuntimeError("sealed v18 run must be a non-symlink directory")
    values: list[str] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("sealed v18 run contains a symbolic link")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError("sealed v18 run contains a non-regular entry")
        values.append(path.relative_to(root).as_posix())
    return tuple(sorted(values))


def _archive_member_hashes(
    archive_path: Path,
) -> tuple[tuple[str, str], ...]:
    if not archive_path.is_file() or archive_path.is_symlink():
        raise RuntimeError("v18 source archive must be a regular file")
    result: dict[str, str] = {}
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            for member in archive.getmembers():
                name = _safe_relative_path(
                    member.name,
                    name="v18 source archive member",
                )
                if (
                    name in result
                    or not member.isfile()
                    or member.issym()
                    or member.islnk()
                ):
                    raise RuntimeError(
                        "v18 source archive contains a duplicate or "
                        "non-regular member"
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise RuntimeError(
                        "v18 source archive member cannot be read"
                    )
                content = handle.read()
                if len(content) != member.size:
                    raise RuntimeError(
                        "v18 source archive member size changed"
                    )
                result[name] = sha256(content).hexdigest()
    except (tarfile.TarError, OSError) as error:
        raise RuntimeError("v18 source archive is invalid") from error
    return tuple(sorted(result.items()))


@dataclass(frozen=True)
class CoverageStateUSCOPESealedV18Receipt:
    """Canonical evidence that the v18 PMOPE result remains a negative."""

    run_repo_path: str
    complete_fingerprint: str
    complete_file_sha256: str
    decision_fingerprint: str
    bounded_result_fingerprint: str
    bounded_result_receipt_fingerprint: str
    source_manifest_file_sha256: str
    source_archive_file_sha256: str
    source_implementation_fingerprint: str
    artifact_files: tuple[tuple[str, str], ...]
    source_members: tuple[tuple[str, str], ...]
    candidate_objective: str
    candidate_objective_policy: str
    final_model_fingerprint: str
    module_state_fingerprint: str
    checkpoint_file_sha256: str
    checkpoint_receipt_fingerprint: str
    checks: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        if (
            self.run_repo_path != COVERAGE_STATE_USCOPE_V18_RUN_REPO_PATH
            or self.complete_fingerprint
            != COVERAGE_STATE_USCOPE_V18_COMPLETE_FINGERPRINT
            or self.complete_file_sha256
            != COVERAGE_STATE_USCOPE_V18_COMPLETE_SHA256
            or self.decision_fingerprint
            != COVERAGE_STATE_USCOPE_V18_DECISION_FINGERPRINT
            or self.bounded_result_fingerprint
            != COVERAGE_STATE_USCOPE_V18_RESULT_FINGERPRINT
            or self.bounded_result_receipt_fingerprint
            != COVERAGE_STATE_USCOPE_V18_RESULT_RECEIPT_FINGERPRINT
            or self.source_manifest_file_sha256
            != COVERAGE_STATE_USCOPE_V18_SOURCE_MANIFEST_SHA256
            or self.source_archive_file_sha256
            != COVERAGE_STATE_USCOPE_V18_SOURCE_ARCHIVE_SHA256
            or self.source_implementation_fingerprint
            != COVERAGE_STATE_USCOPE_V18_IMPLEMENTATION_FINGERPRINT
            or len(self.artifact_files)
            != COVERAGE_STATE_USCOPE_V18_ARTIFACT_COUNT
            or len(self.source_members)
            != COVERAGE_STATE_USCOPE_V18_SOURCE_FILE_COUNT
            or self.candidate_objective != _EXPECTED_OBJECTIVE
            or self.candidate_objective_policy
            != _EXPECTED_OBJECTIVE_POLICY
            or self.final_model_fingerprint
            != _EXPECTED_FINAL_MODEL_FINGERPRINT
            or self.module_state_fingerprint
            != _EXPECTED_MODULE_STATE_FINGERPRINT
            or self.checkpoint_file_sha256
            != _EXPECTED_CHECKPOINT_FILE_SHA256
            or self.checkpoint_receipt_fingerprint
            != _EXPECTED_CHECKPOINT_RECEIPT_FINGERPRINT
            or self.checks != tuple(sorted(self.checks))
            or not self.checks
            or not all(value for _, value in self.checks)
        ):
            raise ValueError("sealed v18 negative-result binding changed")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": COVERAGE_STATE_USCOPE_SEALED_V18_SCHEMA,
            "run_repo_path": self.run_repo_path,
            "complete_fingerprint": self.complete_fingerprint,
            "complete_file_sha256": self.complete_file_sha256,
            "decision_fingerprint": self.decision_fingerprint,
            "bounded_result_fingerprint": (
                self.bounded_result_fingerprint
            ),
            "bounded_result_receipt_fingerprint": (
                self.bounded_result_receipt_fingerprint
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
            "negative_result": {
                "candidate_objective": self.candidate_objective,
                "candidate_objective_policy": (
                    self.candidate_objective_policy
                ),
                "status": _EXPECTED_DECISION,
                "bounded_gate_passed": False,
                "failed_checks": ["candidate_seven_zero_level_gates"],
                "next_action": _EXPECTED_NEXT_ACTION,
                "seed": 42,
                "completed_updates": 400,
                "final_model_fingerprint": (
                    self.final_model_fingerprint
                ),
                "module_state_fingerprint": (
                    self.module_state_fingerprint
                ),
                "checkpoint_file_sha256": (
                    self.checkpoint_file_sha256
                ),
                "checkpoint_receipt_fingerprint": (
                    self.checkpoint_receipt_fingerprint
                ),
            },
            "checks": dict(self.checks),
            "all_pass": True,
            "historical_negative_result": True,
            "contemporaneous_candidate_result": False,
            "verification_mode": (
                "read_only_bytes_strict_json_and_tar_members"
            ),
            "checkpoint_treated_as_opaque_bytes": True,
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

    def verify_unchanged(
        self,
        repository_root: str | Path,
    ) -> None:
        """Re-read the canonical repository evidence and require identity."""

        replay = verify_repository_coverage_state_uscope_sealed_v18(
            repository_root
        )
        if replay != self or replay.receipt_fingerprint != (
            self.receipt_fingerprint
        ):
            raise RuntimeError("sealed v18 negative result changed")


def verify_coverage_state_uscope_sealed_v18_negative(
    run_path: str | Path,
    *,
    source_manifest_path: str | Path,
    source_archive_path: str | Path,
) -> CoverageStateUSCOPESealedV18Receipt:
    """Verify a fixed or byte-identical relocated v18 result."""

    root = Path(run_path)
    manifest_path = Path(source_manifest_path)
    archive_path = Path(source_archive_path)
    complete_path = root / "COMPLETE.json"
    complete = _strict_json(complete_path, name="v18 COMPLETE")
    complete_fingerprint = _verify_embedded_fingerprint(
        complete,
        field="complete_fingerprint",
        name="v18 COMPLETE",
    )
    complete_sha256 = file_sha256(complete_path)
    if (
        complete_fingerprint
        != COVERAGE_STATE_USCOPE_V18_COMPLETE_FINGERPRINT
        or complete_sha256 != COVERAGE_STATE_USCOPE_V18_COMPLETE_SHA256
        or complete.get("schema_version") != _EXPECTED_COMPLETE_SCHEMA
        or complete.get("run_id") != _EXPECTED_RUN_ID
        or complete.get("status") != "complete"
        or complete.get("decision") != _EXPECTED_DECISION
        or complete.get("bounded_gate_passed") is not False
        or complete.get("single_attempt") is not True
        or complete.get("runtime_splits") != ["D_R"]
        or complete.get("D_V_accessed") is not False
        or complete.get("D_T_accessed") is not False
        or complete.get("formal800_eligible") is not False
        or complete.get("formal_800_authorized") is not False
        or complete.get("performance_evaluation_performed") is not False
        or complete.get("artifact_file_count")
        != COVERAGE_STATE_USCOPE_V18_ARTIFACT_COUNT
    ):
        raise RuntimeError("sealed v18 COMPLETE binding changed")

    artifact_mapping = complete.get("artifact_files")
    if not isinstance(artifact_mapping, dict):
        raise RuntimeError("sealed v18 artifact map is missing")
    artifact_files = tuple(
        sorted(
            (
                _safe_relative_path(key, name="v18 artifact path"),
                _required_sha256(value, name=f"artifact hash {key}"),
            )
            for key, value in artifact_mapping.items()
        )
    )
    if tuple(path for path, _ in artifact_files) != (
        COVERAGE_STATE_USCOPE_V18_ARTIFACT_PATHS
    ):
        raise RuntimeError("sealed v18 artifact set changed")
    actual_tree = _regular_tree_files(root)
    if actual_tree != tuple(
        sorted((*COVERAGE_STATE_USCOPE_V18_ARTIFACT_PATHS, "COMPLETE.json"))
    ):
        raise RuntimeError("sealed v18 run tree is not exact")

    parsed_artifacts: dict[str, dict[str, Any]] = {}
    artifact_hashes = dict(artifact_files)
    for relative, expected_hash in artifact_files:
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or file_sha256(path) != expected_hash
        ):
            raise RuntimeError(f"sealed v18 artifact hash changed: {relative}")
        if relative.endswith(".json"):
            payload = _strict_json(path, name=f"v18 artifact {relative}")
            _verify_embedded_fingerprint(
                payload,
                field="receipt_fingerprint",
                name=f"v18 artifact {relative}",
            )
            _assert_all_split_access_false(payload, location=relative)
            parsed_artifacts[relative] = payload

    config = parsed_artifacts["receipts/config.json"]
    decision = parsed_artifacts["receipts/decision.json"]
    bounded_receipt = parsed_artifacts["receipts/bounded_result.json"]
    training_receipt = parsed_artifacts["receipts/training.json"]
    zero_receipt = parsed_artifacts["receipts/zero_level.json"]
    checkpoint = parsed_artifacts[
        "checkpoints/pmope_joint.checkpoint.json"
    ]
    if (
        config.get("run_id") != _EXPECTED_RUN_ID
        or config.get("output_repo_path")
        != COVERAGE_STATE_USCOPE_V18_RUN_REPO_PATH
        or config.get("model", {}).get("objective_suite")
        != [_EXPECTED_OBJECTIVE]
        or config.get("model", {}).get("candidate_objective_policy")
        != _EXPECTED_OBJECTIVE_POLICY
        or config.get("budget", {}).get("seed") != 42
        or config.get("budget", {}).get("epochs") != 10
        or config.get("budget", {}).get("steps_per_epoch") != 40
        or config.get("budget", {}).get("updates_per_objective") != 400
        or config.get("budget", {}).get("objectives") != 1
    ):
        raise RuntimeError("sealed v18 configuration changed")

    decision_fingerprint = str(decision.get("receipt_fingerprint"))
    bounded_result_fingerprint = _required_sha256(
        bounded_receipt.get("result_fingerprint"),
        name="v18 bounded result fingerprint",
    )
    bounded_payload = bounded_receipt.get("result")
    if (
        decision.get("schema_version") != _EXPECTED_DECISION_SCHEMA
        or decision_fingerprint
        != COVERAGE_STATE_USCOPE_V18_DECISION_FINGERPRINT
        or artifact_hashes["receipts/decision.json"]
        != COVERAGE_STATE_USCOPE_V18_DECISION_SHA256
        or decision.get("status") != _EXPECTED_DECISION
        or decision.get("candidate_objective") != _EXPECTED_OBJECTIVE
        or decision.get("bounded_gate_passed") is not False
        or decision.get("candidate_gate_passed") is not False
        or decision.get("failed_checks")
        != ["candidate_seven_zero_level_gates"]
        or decision.get("next_action") != _EXPECTED_NEXT_ACTION
        or decision.get("formal800_eligible") is not False
        or decision.get("formal_800_authorized") is not False
        or decision.get("performance_claim_supported") is not False
        or decision.get("result_fingerprint")
        != COVERAGE_STATE_USCOPE_V18_RESULT_FINGERPRINT
        or bounded_receipt.get("schema_version")
        != _EXPECTED_RESULT_SCHEMA
        or bounded_receipt.get("receipt_fingerprint")
        != COVERAGE_STATE_USCOPE_V18_RESULT_RECEIPT_FINGERPRINT
        or bounded_result_fingerprint
        != COVERAGE_STATE_USCOPE_V18_RESULT_FINGERPRINT
        or not isinstance(bounded_payload, dict)
        or stable_fingerprint(bounded_payload)
        != bounded_result_fingerprint
        or bounded_payload.get("bounded_gate_passed") is not False
        or bounded_payload.get("candidate_objective")
        != _EXPECTED_OBJECTIVE
        or bounded_payload.get("failed_checks")
        != ["candidate_seven_zero_level_gates"]
        or bounded_payload.get("formal800_eligible") is not False
        or bounded_payload.get("performance_claim_supported") is not False
    ):
        raise RuntimeError("sealed v18 decision/result binding changed")

    training = training_receipt.get("training")
    objectives = training.get("objectives") if isinstance(training, dict) else None
    objective_training = (
        objectives[0]
        if isinstance(objectives, list)
        and len(objectives) == 1
        and isinstance(objectives[0], dict)
        else None
    )
    diagnostic = zero_receipt.get("candidate_diagnostic")
    gates = diagnostic.get("gates") if isinstance(diagnostic, dict) else None
    if (
        not isinstance(objective_training, dict)
        or training.get("objective_suite") != [_EXPECTED_OBJECTIVE]
        or objective_training.get("objective") != _EXPECTED_OBJECTIVE
        or objective_training.get("objective_policy")
        != _EXPECTED_OBJECTIVE_POLICY
        or objective_training.get("seed") != 42
        or objective_training.get("completed_updates") != 400
        or objective_training.get("compute", {}).get("optimizer_steps") != 400
        or objective_training.get("final_model_fingerprint")
        != _EXPECTED_FINAL_MODEL_FINGERPRINT
        or not isinstance(diagnostic, dict)
        or diagnostic.get("checkpoint_fingerprint")
        != _EXPECTED_MODULE_STATE_FINGERPRINT
        or not isinstance(gates, dict)
        or gates.get("bounded_gate_passed") is not False
        or zero_receipt.get("candidate_bounded_gate_passed") is not False
    ):
        raise RuntimeError("sealed v18 training/diagnostic binding changed")

    if (
        checkpoint.get("schema_version") != _EXPECTED_CHECKPOINT_SCHEMA
        or checkpoint.get("objective") != _EXPECTED_OBJECTIVE
        or checkpoint.get("objective_policy") != _EXPECTED_OBJECTIVE_POLICY
        or checkpoint.get("model_class") != _EXPECTED_MODEL_CLASS
        or checkpoint.get("model_config", {}).get("parameter_count")
        != _EXPECTED_PARAMETER_COUNT
        or checkpoint.get("serialization") != "safetensors"
        or checkpoint.get("tensor_only_state_dict") is not True
        or checkpoint.get("weights_only_roundtrip_verified") is not True
        or tuple(checkpoint.get("state_keys", ())) != _EXPECTED_STATE_KEYS
        or checkpoint.get("module_state_fingerprint")
        != _EXPECTED_MODULE_STATE_FINGERPRINT
        or checkpoint.get("checkpoint_file_sha256")
        != _EXPECTED_CHECKPOINT_FILE_SHA256
        or checkpoint.get("checkpoint_file_sha256")
        != artifact_hashes["checkpoints/pmope_joint.safetensors"]
        or checkpoint.get("receipt_fingerprint")
        != _EXPECTED_CHECKPOINT_RECEIPT_FINGERPRINT
        or decision.get("checkpoint_receipt_fingerprints")
        != {_EXPECTED_OBJECTIVE: _EXPECTED_CHECKPOINT_RECEIPT_FINGERPRINT}
    ):
        raise RuntimeError("sealed v18 opaque checkpoint binding changed")

    if (
        complete.get("decision_fingerprint") != decision_fingerprint
        or complete.get("bounded_result_receipt_fingerprint")
        != bounded_receipt.get("receipt_fingerprint")
        or complete.get("training_receipt_fingerprint")
        != training_receipt.get("receipt_fingerprint")
        or complete.get("zero_level_receipt_fingerprint")
        != zero_receipt.get("receipt_fingerprint")
    ):
        raise RuntimeError("sealed v18 COMPLETE receipt links changed")

    source_manifest = _strict_json(
        manifest_path,
        name="v18 source closure manifest",
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
        != COVERAGE_STATE_USCOPE_V18_SOURCE_MANIFEST_SHA256
        or archive_sha256
        != COVERAGE_STATE_USCOPE_V18_SOURCE_ARCHIVE_SHA256
        or source_manifest.get("schema_version")
        != _EXPECTED_SOURCE_CLOSURE_SCHEMA
        or source_manifest.get("run_repo_path")
        != COVERAGE_STATE_USCOPE_V18_RUN_REPO_PATH
        or source_manifest.get("complete_fingerprint")
        != COVERAGE_STATE_USCOPE_V18_COMPLETE_FINGERPRINT
        or source_manifest.get("complete_file_sha256")
        != COVERAGE_STATE_USCOPE_V18_COMPLETE_SHA256
        or source_manifest.get("archive_repo_path")
        != COVERAGE_STATE_USCOPE_V18_SOURCE_ARCHIVE_REPO_PATH
        or source_manifest.get("archive_sha256")
        != COVERAGE_STATE_USCOPE_V18_SOURCE_ARCHIVE_SHA256
        or source_manifest.get("archive_bytes") != archive_path.stat().st_size
        or source_manifest.get("source_file_count")
        != COVERAGE_STATE_USCOPE_V18_SOURCE_FILE_COUNT
        or source_manifest.get("source_manifest")
        != _EXPECTED_SOURCE_MANIFEST_POINTER
        or source_manifest.get("implementation_fingerprint")
        != COVERAGE_STATE_USCOPE_V18_IMPLEMENTATION_FINGERPRINT
        or not isinstance(implementation_files, dict)
        or source_implementation.get("implementation_fingerprint")
        != COVERAGE_STATE_USCOPE_V18_IMPLEMENTATION_FINGERPRINT
        or source_manifest.get("config_receipt_sha256")
        != artifact_hashes["receipts/config.json"]
        or source_manifest.get("training_receipt_sha256")
        != artifact_hashes["receipts/training.json"]
        or source_manifest.get("zero_level_receipt_sha256")
        != artifact_hashes["receipts/zero_level.json"]
        or source_manifest.get("bounded_result_receipt_sha256")
        != artifact_hashes["receipts/bounded_result.json"]
        or source_manifest.get("decision_receipt_sha256")
        != artifact_hashes["receipts/decision.json"]
    ):
        raise RuntimeError("sealed v18 source closure manifest changed")

    source_members = _archive_member_hashes(archive_path)
    normalized_implementation_files = tuple(
        sorted(
            (
                _safe_relative_path(
                    path,
                    name="v18 implementation source path",
                ),
                _required_sha256(
                    digest,
                    name=f"v18 implementation source hash {path}",
                ),
            )
            for path, digest in implementation_files.items()
        )
    )
    if (
        len(source_members) != COVERAGE_STATE_USCOPE_V18_SOURCE_FILE_COUNT
        or source_members != normalized_implementation_files
    ):
        raise RuntimeError("sealed v18 source archive members changed")

    checks = tuple(
        sorted(
            {
                "artifact_set_and_hashes_exact": True,
                "all_json_fingerprints_exact": True,
                "complete_identity_and_negative_decision_bound": True,
                "decision_and_bounded_result_bound": True,
                "seed42_and_400_updates_bound": True,
                "opaque_checkpoint_hash_and_receipt_bound": True,
                "source_manifest_identity_bound": True,
                "source_archive_identity_bound": True,
                "source_archive_members_match_implementation": True,
                "formal800_not_authorized": True,
                "performance_claim_not_supported": True,
                "D_V_not_accessed": True,
                "D_T_not_accessed": True,
                "no_checkpoint_deserialization": True,
                "no_evaluator_call": True,
                "no_training": True,
            }.items()
        )
    )
    receipt = CoverageStateUSCOPESealedV18Receipt(
        run_repo_path=str(source_manifest["run_repo_path"]),
        complete_fingerprint=complete_fingerprint,
        complete_file_sha256=complete_sha256,
        decision_fingerprint=decision_fingerprint,
        bounded_result_fingerprint=bounded_result_fingerprint,
        bounded_result_receipt_fingerprint=str(
            bounded_receipt["receipt_fingerprint"]
        ),
        source_manifest_file_sha256=manifest_sha256,
        source_archive_file_sha256=archive_sha256,
        source_implementation_fingerprint=str(
            source_manifest["implementation_fingerprint"]
        ),
        artifact_files=artifact_files,
        source_members=source_members,
        candidate_objective=str(decision["candidate_objective"]),
        candidate_objective_policy=str(
            bounded_payload["candidate_objective_policy"]
        ),
        final_model_fingerprint=str(
            objective_training["final_model_fingerprint"]
        ),
        module_state_fingerprint=str(
            checkpoint["module_state_fingerprint"]
        ),
        checkpoint_file_sha256=str(
            checkpoint["checkpoint_file_sha256"]
        ),
        checkpoint_receipt_fingerprint=str(
            checkpoint["receipt_fingerprint"]
        ),
        checks=checks,
    )
    if receipt.receipt_fingerprint != (
        COVERAGE_STATE_USCOPE_V18_SEALED_RECEIPT_FINGERPRINT
    ):
        raise RuntimeError("sealed v18 receipt fingerprint changed")
    return receipt


def verify_repository_coverage_state_uscope_sealed_v18(
    repository_root: str | Path,
) -> CoverageStateUSCOPESealedV18Receipt:
    """Verify the canonical sealed v18 evidence beneath one repository root."""

    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("repository_root must be a canonical directory")
    return verify_coverage_state_uscope_sealed_v18_negative(
        root / COVERAGE_STATE_USCOPE_V18_RUN_REPO_PATH,
        source_manifest_path=(
            root / COVERAGE_STATE_USCOPE_V18_SOURCE_MANIFEST_REPO_PATH
        ),
        source_archive_path=(
            root / COVERAGE_STATE_USCOPE_V18_SOURCE_ARCHIVE_REPO_PATH
        ),
    )


__all__ = [
    "COVERAGE_STATE_USCOPE_SEALED_V18_SCHEMA",
    "COVERAGE_STATE_USCOPE_V18_ARTIFACT_COUNT",
    "COVERAGE_STATE_USCOPE_V18_ARTIFACT_PATHS",
    "COVERAGE_STATE_USCOPE_V18_COMPLETE_FINGERPRINT",
    "COVERAGE_STATE_USCOPE_V18_COMPLETE_SHA256",
    "COVERAGE_STATE_USCOPE_V18_DECISION_FINGERPRINT",
    "COVERAGE_STATE_USCOPE_V18_DECISION_SHA256",
    "COVERAGE_STATE_USCOPE_V18_IMPLEMENTATION_FINGERPRINT",
    "COVERAGE_STATE_USCOPE_V18_RESULT_FINGERPRINT",
    "COVERAGE_STATE_USCOPE_V18_RESULT_RECEIPT_FINGERPRINT",
    "COVERAGE_STATE_USCOPE_V18_RUN_REPO_PATH",
    "COVERAGE_STATE_USCOPE_V18_SEALED_RECEIPT_FINGERPRINT",
    "COVERAGE_STATE_USCOPE_V18_SOURCE_ARCHIVE_REPO_PATH",
    "COVERAGE_STATE_USCOPE_V18_SOURCE_ARCHIVE_SHA256",
    "COVERAGE_STATE_USCOPE_V18_SOURCE_FILE_COUNT",
    "COVERAGE_STATE_USCOPE_V18_SOURCE_MANIFEST_REPO_PATH",
    "COVERAGE_STATE_USCOPE_V18_SOURCE_MANIFEST_SHA256",
    "CoverageStateUSCOPESealedV18Receipt",
    "verify_coverage_state_uscope_sealed_v18_negative",
    "verify_repository_coverage_state_uscope_sealed_v18",
]
