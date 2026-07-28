"""Append-only evaluation-v3 wrapper for the sealed PAET Formal800 D_V.

The original D_V runner is executed only inside two strict, restoring
contexts:

1. the already verified evaluation-v2 Formal800 schema aliases; and
2. four evaluation-v3 substitutions: a distinct run ID, distinct output and
   staging paths, and the mechanically corrected fixed-sample builder.

Every scientific input and operation remains delegated to the original
runner.  The failed evaluation-v2 staging directory is neither resumed nor
modified.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterator, Mapping

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.experiment import (
    coverage_state_paet_formal_d_v_runner as _d_v,
)
from cure_lite.experiment import (
    coverage_state_paet_formal_evaluation as _evaluation,
)
from cure_lite_eval_v2 import formal_d_v_runner_v2 as _v2

from .evaluation_source_closure import (
    verify_evaluation_v3_source_closure,
)
from .evaluation_v3_amendment import (
    FAILED_RUN_ID,
    FAILED_STAGING_REPO_PATH,
    FAILURE_RECEIPT_REPO_PATH,
    NEW_EXTERNAL_BINDING_REPO_PATH,
    NEW_OUTPUT_REPO_PATH,
    NEW_RUN_ID,
    NEW_STAGING_REPO_PATH,
    verify_evaluation_v3_amendment,
)
from .fixed_sample_builder_v3 import (
    build_paet_fixed_d_v_samples,
    verify_fixed_sample_builder_correction,
)


EVALUATION_V3_RECOVERY_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal-d-v-evaluation-recovery-v3"
)
EVALUATION_V3_EVIDENCE_BINDING_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal-d-v-external-evidence-binding-v3"
)
_REPO_ROOT = Path(__file__).resolve().parents[1]
NEW_OUTPUT_PATH = _REPO_ROOT / NEW_OUTPUT_REPO_PATH
NEW_STAGING_PATH = _REPO_ROOT / NEW_STAGING_REPO_PATH
NEW_EXTERNAL_BINDING_PATH = (
    _REPO_ROOT / NEW_EXTERNAL_BINDING_REPO_PATH
)
NEW_EXTERNAL_BINDING_STAGING_PATH = (
    NEW_EXTERNAL_BINDING_PATH.with_name(
        f".{NEW_EXTERNAL_BINDING_PATH.name}.incomplete"
    )
)
_ORIGINAL_OUTPUT_PATH = (
    _REPO_ROOT
    / "runs/irstd1k_stage_a_seed42"
    / FAILED_RUN_ID
)
_ORIGINAL_STAGING_PATH = _REPO_ROOT / FAILED_STAGING_REPO_PATH
_ORIGINAL_BUILDER = _evaluation.build_paet_fixed_d_v_samples
_D_V_MEMBER_NAMES = (
    "COMPLETE.json",
    "decision.json",
    "receipt.json",
)
_PATCH_LOCK = threading.RLock()


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
    ):
        raise RuntimeError(f"{name} must be a canonical regular file")

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"{name} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} is not strict UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} must be a JSON object")
    return payload


def _validate_chain() -> dict[str, object]:
    amendment = verify_evaluation_v3_amendment(_REPO_ROOT)
    source_closure = verify_evaluation_v3_source_closure(_REPO_ROOT)
    parents = source_closure.get("parent_bindings")
    if not isinstance(parents, Mapping):
        raise RuntimeError(
            "evaluation-v3 source closure lacks parent bindings"
        )
    failure = parents.get("evaluation_v2_failure")
    v2 = parents.get("evaluation_v2_source_closure")
    original = parents.get("original_formal_source_closure")
    schema_erratum = parents.get("formal800_schema_erratum")
    v3_amendment = parents.get("evaluation_v3_amendment")
    if (
        not isinstance(failure, Mapping)
        or not isinstance(v2, Mapping)
        or not isinstance(original, Mapping)
        or not isinstance(schema_erratum, Mapping)
        or not isinstance(v3_amendment, Mapping)
        or failure.get("sha256")
        != amendment["failure_receipt_sha256"]
        or failure.get("failure_fingerprint")
        != amendment["failure_fingerprint"]
        or failure.get("D_V_accessed") is not True
        or failure.get("D_T_accessed") is not False
        or v2.get("manifest_sha256")
        != amendment["evaluation_v2_closure_manifest_sha256"]
        or v2.get("archive_sha256")
        != amendment["evaluation_v2_closure_archive_sha256"]
        or original.get("manifest_sha256")
        != amendment["original_source_closure_manifest_sha256"]
        or original.get("archive_sha256")
        != amendment["original_source_closure_archive_sha256"]
        or schema_erratum.get("sha256")
        != amendment["schema_erratum_sha256"]
        or schema_erratum.get("erratum_fingerprint")
        != amendment["schema_erratum_fingerprint"]
        or v3_amendment.get("sha256")
        != amendment["amendment_sha256"]
        or v3_amendment.get("amendment_fingerprint")
        != amendment["amendment_fingerprint"]
    ):
        raise RuntimeError(
            "evaluation-v3 source closure is not bound to its full chain"
        )
    return {
        "amendment": amendment,
        "evaluation_v3_source_closure": source_closure,
    }


def _expected_original_runner_globals() -> dict[str, object]:
    return {
        "PAET_FORMAL_DV_RUN_ID": FAILED_RUN_ID,
        "PAET_FORMAL_DV_OUTPUT_PATH": _ORIGINAL_OUTPUT_PATH,
        "PAET_FORMAL_DV_STAGING_PATH": _ORIGINAL_STAGING_PATH,
        "build_paet_fixed_d_v_samples": _ORIGINAL_BUILDER,
    }


def _actual_runner_globals() -> dict[str, object]:
    return {
        "PAET_FORMAL_DV_RUN_ID": _d_v.PAET_FORMAL_DV_RUN_ID,
        "PAET_FORMAL_DV_OUTPUT_PATH": (
            _d_v.PAET_FORMAL_DV_OUTPUT_PATH
        ),
        "PAET_FORMAL_DV_STAGING_PATH": (
            _d_v.PAET_FORMAL_DV_STAGING_PATH
        ),
        "build_paet_fixed_d_v_samples": (
            _d_v.build_paet_fixed_d_v_samples
        ),
    }


def _runner_globals_are(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
) -> bool:
    return (
        actual["PAET_FORMAL_DV_RUN_ID"]
        == expected["PAET_FORMAL_DV_RUN_ID"]
        and actual["PAET_FORMAL_DV_OUTPUT_PATH"]
        == expected["PAET_FORMAL_DV_OUTPUT_PATH"]
        and actual["PAET_FORMAL_DV_STAGING_PATH"]
        == expected["PAET_FORMAL_DV_STAGING_PATH"]
        and actual["build_paet_fixed_d_v_samples"]
        is expected["build_paet_fixed_d_v_samples"]
    )


@contextmanager
def _exact_evaluation_v3_runner_patch() -> Iterator[None]:
    """Patch exactly four original-runner globals and always restore them."""

    with _PATCH_LOCK:
        before = _actual_runner_globals()
        expected_before = _expected_original_runner_globals()
        if not _runner_globals_are(before, expected_before):
            raise RuntimeError(
                "original D_V runner globals changed; v3 patch is refused"
            )
        verify_fixed_sample_builder_correction()
        _d_v.PAET_FORMAL_DV_RUN_ID = NEW_RUN_ID
        _d_v.PAET_FORMAL_DV_OUTPUT_PATH = NEW_OUTPUT_PATH
        _d_v.PAET_FORMAL_DV_STAGING_PATH = NEW_STAGING_PATH
        _d_v.build_paet_fixed_d_v_samples = (
            build_paet_fixed_d_v_samples
        )
        try:
            expected_during = {
                "PAET_FORMAL_DV_RUN_ID": NEW_RUN_ID,
                "PAET_FORMAL_DV_OUTPUT_PATH": NEW_OUTPUT_PATH,
                "PAET_FORMAL_DV_STAGING_PATH": NEW_STAGING_PATH,
                "build_paet_fixed_d_v_samples": (
                    build_paet_fixed_d_v_samples
                ),
            }
            if not _runner_globals_are(
                _actual_runner_globals(),
                expected_during,
            ):
                raise RuntimeError(
                    "evaluation-v3 runner patch did not apply exactly"
                )
            yield
        finally:
            _d_v.PAET_FORMAL_DV_RUN_ID = before[
                "PAET_FORMAL_DV_RUN_ID"
            ]
            _d_v.PAET_FORMAL_DV_OUTPUT_PATH = before[
                "PAET_FORMAL_DV_OUTPUT_PATH"
            ]
            _d_v.PAET_FORMAL_DV_STAGING_PATH = before[
                "PAET_FORMAL_DV_STAGING_PATH"
            ]
            _d_v.build_paet_fixed_d_v_samples = before[
                "build_paet_fixed_d_v_samples"
            ]
            if not _runner_globals_are(
                _actual_runner_globals(),
                expected_before,
            ):
                raise RuntimeError(
                    "original D_V runner globals were not restored"
                )


def _new_destinations_are_fresh() -> None:
    for path, label in (
        (NEW_OUTPUT_PATH, "evaluation-v3 D_V output"),
        (NEW_STAGING_PATH, "evaluation-v3 D_V staging"),
        (
            NEW_EXTERNAL_BINDING_PATH,
            "evaluation-v3 external evidence binding",
        ),
        (
            NEW_EXTERNAL_BINDING_STAGING_PATH,
            "evaluation-v3 evidence-binding staging",
        ),
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"{label} already exists")


def validate_paet_formal_d_v_create_only_v3() -> dict[str, object]:
    """Validate v3 without opening D_V/D_T or creating any output."""

    chain = _validate_chain()
    _new_destinations_are_fresh()
    loaded = _v2._load_original_attempt_with_aliases()  # noqa: SLF001
    with _exact_evaluation_v3_runner_patch():
        original_plan = _d_v.validate_paet_formal_d_v_create_only()
    if (
        original_plan.get("run_id") != NEW_RUN_ID
        or original_plan.get("output_repo_path")
        != NEW_OUTPUT_REPO_PATH
        or original_plan.get("staging_repo_path")
        != NEW_STAGING_REPO_PATH
        or original_plan.get("D_V_accessed") is not False
        or original_plan.get("D_T_accessed") is not False
        or original_plan.get("output_created") is not False
        or original_plan.get("create_only") is not True
    ):
        raise RuntimeError(
            "original create-only plan changed under evaluation-v3"
        )
    if not _runner_globals_are(
        _actual_runner_globals(),
        _expected_original_runner_globals(),
    ):
        raise RuntimeError("original D_V runner globals were not restored")
    _new_destinations_are_fresh()
    amendment = chain["amendment"]
    source_closure = chain["evaluation_v3_source_closure"]
    assert isinstance(amendment, Mapping)
    assert isinstance(source_closure, Mapping)
    body = {
        "schema_version": EVALUATION_V3_RECOVERY_SCHEMA,
        "mode": "validate_create_only",
        "status": "valid",
        "new_attempt": {
            "run_id": NEW_RUN_ID,
            "output_repo_path": NEW_OUTPUT_REPO_PATH,
            "staging_repo_path": NEW_STAGING_REPO_PATH,
            "distinct_from_failed_run": True,
            "destinations_fresh": True,
        },
        "prior_failed_attempt": {
            "run_id": FAILED_RUN_ID,
            "failure_receipt_repo_path": (
                FAILURE_RECEIPT_REPO_PATH
            ),
            "failure_receipt_sha256": (
                amendment["failure_receipt_sha256"]
            ),
            "failure_fingerprint": amendment["failure_fingerprint"],
            "D_V_accessed": True,
            "D_T_accessed": False,
            "model_forward_calls": 0,
            "failed_staging_preserved": True,
            "resume_or_reuse_performed": False,
        },
        "runtime_patch": {
            "count": 4,
            "symbols": [
                "PAET_FORMAL_DV_RUN_ID",
                "PAET_FORMAL_DV_OUTPUT_PATH",
                "PAET_FORMAL_DV_STAGING_PATH",
                "build_paet_fixed_d_v_samples",
            ],
            "original_values_restored": True,
            "builder_correction": (
                verify_fixed_sample_builder_correction()
            ),
        },
        "formal_attempt": {
            "complete_fingerprint": loaded.complete_fingerprint,
            "artifact_fingerprint": (
                loaded.artifact.artifact_fingerprint
            ),
            "module_state_fingerprint": (
                loaded.artifact.module_state_fingerprint
            ),
            "source_closure_manifest_sha256": (
                loaded.source_closure_manifest_sha256
            ),
            "source_closure_archive_sha256": (
                loaded.source_closure_archive_sha256
            ),
            "source_closure_content_fingerprint": (
                loaded.source_closure_content_fingerprint
            ),
        },
        "evaluation_v3_amendment": {
            "repo_path": amendment["amendment_repo_path"],
            "sha256": amendment["amendment_sha256"],
            "amendment_fingerprint": (
                amendment["amendment_fingerprint"]
            ),
        },
        "evaluation_v3_source_closure": {
            "manifest_repo_path": source_closure[
                "manifest_repo_path"
            ],
            "manifest_sha256": source_closure["manifest_sha256"],
            "archive_repo_path": source_closure[
                "archive_repo_path"
            ],
            "archive_sha256": source_closure["archive_sha256"],
            "content_fingerprint": source_closure[
                "content_fingerprint"
            ],
            "file_count": source_closure["file_count"],
        },
        "original_create_only_plan": original_plan,
        "Formal800_attempt_loaded": True,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "output_created": False,
        "metric_or_gate_changed": False,
        "data_or_model_changed": False,
    }
    return {**body, "validation_fingerprint": stable_fingerprint(body)}


def _validated_new_d_v_files() -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
]:
    expected_during = {
        "PAET_FORMAL_DV_RUN_ID": NEW_RUN_ID,
        "PAET_FORMAL_DV_OUTPUT_PATH": NEW_OUTPUT_PATH,
        "PAET_FORMAL_DV_STAGING_PATH": NEW_STAGING_PATH,
        "build_paet_fixed_d_v_samples": (
            build_paet_fixed_d_v_samples
        ),
    }
    if not _runner_globals_are(
        _actual_runner_globals(),
        expected_during,
    ):
        raise RuntimeError(
            "new D_V output may be validated only inside the v3 patch"
        )
    result = _d_v._validate_published_output(  # noqa: SLF001
        NEW_OUTPUT_PATH
    )
    members = {path.name: path for path in NEW_OUTPUT_PATH.iterdir()}
    if set(members) != set(_D_V_MEMBER_NAMES):
        raise RuntimeError("published evaluation-v3 D_V inventory changed")
    rows: dict[str, dict[str, object]] = {}
    for name in _D_V_MEMBER_NAMES:
        path = members[name]
        rows[name] = {
            "repo_path": path.relative_to(_REPO_ROOT).as_posix(),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    return result, rows


def _build_external_evidence_binding(
    chain: Mapping[str, object],
) -> dict[str, object]:
    result, files = _validated_new_d_v_files()
    complete = _strict_json(
        NEW_OUTPUT_PATH / "COMPLETE.json",
        name="evaluation-v3 D_V COMPLETE",
    )
    receipt = _strict_json(
        NEW_OUTPUT_PATH / "receipt.json",
        name="evaluation-v3 D_V receipt",
    )
    decision = _strict_json(
        NEW_OUTPUT_PATH / "decision.json",
        name="evaluation-v3 D_V decision",
    )
    amendment = chain.get("amendment")
    source_closure = chain.get("evaluation_v3_source_closure")
    if not isinstance(amendment, Mapping) or not isinstance(
        source_closure,
        Mapping,
    ):
        raise TypeError("verified evaluation-v3 chain is malformed")
    body = {
        "schema_version": EVALUATION_V3_EVIDENCE_BINDING_SCHEMA,
        "status": "complete",
        "binding_is_external_to_immutable_D_V_output": True,
        "D_V": {
            "run_id": NEW_RUN_ID,
            "output_repo_path": NEW_OUTPUT_REPO_PATH,
            "files": files,
            "complete_fingerprint": complete.get(
                "complete_fingerprint"
            ),
            "receipt_fingerprint": receipt.get(
                "receipt_fingerprint"
            ),
            "decision_fingerprint": decision.get(
                "decision_fingerprint"
            ),
            "evaluation_result_fingerprint": receipt.get(
                "evaluation_result_fingerprint"
            ),
            "gate_passed": result["gate_passed"],
            "authorizes_D_T": result["authorizes_D_T"],
            "D_T_accessed": False,
        },
        "Formal800": {
            "complete_sha256": amendment["formal_complete_sha256"],
            "complete_fingerprint": (
                amendment["formal_complete_fingerprint"]
            ),
            "source_closure_manifest_sha256": (
                amendment[
                    "original_source_closure_manifest_sha256"
                ]
            ),
            "source_closure_archive_sha256": (
                amendment["original_source_closure_archive_sha256"]
            ),
            "source_closure_content_fingerprint": (
                amendment[
                    "original_source_closure_content_fingerprint"
                ]
            ),
            "source_closure_file_count": (
                amendment["original_source_closure_file_count"]
            ),
        },
        "formal800_schema_erratum": {
            "repo_path": amendment["schema_erratum_repo_path"],
            "sha256": amendment["schema_erratum_sha256"],
            "erratum_fingerprint": (
                amendment["schema_erratum_fingerprint"]
            ),
            "authorized_alias_count": (
                amendment["authorized_alias_count"]
            ),
        },
        "evaluation_v2_source_closure": {
            "manifest_sha256": (
                amendment["evaluation_v2_closure_manifest_sha256"]
            ),
            "archive_sha256": (
                amendment["evaluation_v2_closure_archive_sha256"]
            ),
            "content_fingerprint": (
                amendment[
                    "evaluation_v2_closure_content_fingerprint"
                ]
            ),
            "file_count": (
                amendment["evaluation_v2_closure_file_count"]
            ),
        },
        "failed_evaluation_v2_attempt": {
            "run_id": FAILED_RUN_ID,
            "failure_receipt_repo_path": (
                amendment["failure_receipt_repo_path"]
            ),
            "failure_receipt_sha256": (
                amendment["failure_receipt_sha256"]
            ),
            "failure_fingerprint": amendment["failure_fingerprint"],
            "D_V_accessed": True,
            "D_T_accessed": False,
            "model_forward_calls": 0,
            "failed_staging_preserved": True,
            "failed_attempt_resumed_or_reused": False,
        },
        "evaluation_v3_amendment": {
            "repo_path": amendment["amendment_repo_path"],
            "sha256": amendment["amendment_sha256"],
            "amendment_fingerprint": (
                amendment["amendment_fingerprint"]
            ),
        },
        "evaluation_v3_source_closure": {
            "manifest_repo_path": source_closure[
                "manifest_repo_path"
            ],
            "manifest_sha256": source_closure["manifest_sha256"],
            "archive_repo_path": source_closure[
                "archive_repo_path"
            ],
            "archive_sha256": source_closure["archive_sha256"],
            "content_fingerprint": source_closure[
                "content_fingerprint"
            ],
            "file_count": source_closure["file_count"],
        },
        "correction_scope": {
            "runtime_patch_symbol_count": 4,
            "formal800_schema_alias_count": 2,
            "fixed_sample_builder_access_path_corrections": 1,
            "artifact_bytes_modified": False,
            "failed_attempt_modified": False,
            "model_retrained": False,
            "model_state_updated": False,
            "data_modified": False,
            "metric_or_gate_modified": False,
            "inference_rule_modified": False,
        },
    }
    return {**body, "binding_fingerprint": stable_fingerprint(body)}


def _publish_external_binding(
    payload: Mapping[str, object],
) -> None:
    if (
        NEW_EXTERNAL_BINDING_PATH.exists()
        or NEW_EXTERNAL_BINDING_PATH.is_symlink()
    ):
        raise FileExistsError(
            "evaluation-v3 external evidence binding already exists"
        )
    if (
        NEW_EXTERNAL_BINDING_STAGING_PATH.exists()
        or NEW_EXTERNAL_BINDING_STAGING_PATH.is_symlink()
    ):
        raise FileExistsError(
            "evaluation-v3 evidence-binding staging already exists"
        )
    NEW_EXTERNAL_BINDING_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    descriptor = os.open(
        NEW_EXTERNAL_BINDING_STAGING_PATH,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(_canonical_json(payload))
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    os.link(
        NEW_EXTERNAL_BINDING_STAGING_PATH,
        NEW_EXTERNAL_BINDING_PATH,
        follow_symlinks=False,
    )
    directory = os.open(
        NEW_EXTERNAL_BINDING_PATH.parent,
        os.O_RDONLY | os.O_DIRECTORY,
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    NEW_EXTERNAL_BINDING_STAGING_PATH.unlink()


def _verify_external_binding(
    expected: Mapping[str, object],
) -> dict[str, object]:
    actual = _strict_json(
        NEW_EXTERNAL_BINDING_PATH,
        name="evaluation-v3 external evidence binding",
    )
    if actual != expected:
        raise RuntimeError(
            "evaluation-v3 external evidence binding changed"
        )
    body = dict(actual)
    fingerprint = body.pop("binding_fingerprint", None)
    if fingerprint != stable_fingerprint(body):
        raise RuntimeError(
            "evaluation-v3 evidence-binding fingerprint changed"
        )
    return {
        "repo_path": NEW_EXTERNAL_BINDING_REPO_PATH,
        "sha256": file_sha256(NEW_EXTERNAL_BINDING_PATH),
        "binding_fingerprint": fingerprint,
    }


def finalize_paet_formal_d_v_evidence_binding_v3() -> dict[str, object]:
    """Finalize only the external receipt after a completed v3 D_V run."""

    chain = _validate_chain()
    with _exact_evaluation_v3_runner_patch():
        expected = _build_external_evidence_binding(chain)
    if NEW_EXTERNAL_BINDING_PATH.exists():
        return _verify_external_binding(expected)
    _publish_external_binding(expected)
    return _verify_external_binding(expected)


def run_paet_formal_d_v_once_v3() -> dict[str, object]:
    """Run the distinct fixed D_V attempt under the one-point correction."""

    chain = _validate_chain()
    _new_destinations_are_fresh()
    with _v2._exact_formal800_schema_aliases():  # noqa: SLF001
        with _exact_evaluation_v3_runner_patch():
            original_result = _d_v.run_paet_formal_d_v_once()
            external = _build_external_evidence_binding(chain)
    if not _runner_globals_are(
        _actual_runner_globals(),
        _expected_original_runner_globals(),
    ):
        raise RuntimeError("original D_V runner globals were not restored")
    if (
        _v2._attempt.PAET_FORMAL_ATTEMPT_SCHEMA  # noqa: SLF001
        != _v2.ATTEMPT_CONSUMER_SCHEMA
        or _v2._attempt.PAET_FORMAL_STARTED_SCHEMA  # noqa: SLF001
        != _v2.STARTED_CONSUMER_SCHEMA
    ):
        raise RuntimeError(
            "Formal800 schema aliases were not restored"
        )
    _publish_external_binding(external)
    binding = _verify_external_binding(external)
    return {
        "schema_version": EVALUATION_V3_RECOVERY_SCHEMA,
        "mode": "run_once",
        "status": "complete",
        "new_run_id": NEW_RUN_ID,
        "original_D_V_result": original_result,
        "external_evidence_binding": binding,
        "prior_failed_attempt_preserved": True,
        "D_T_accessed": False,
        "model_retrained": False,
        "artifact_bytes_modified": False,
        "metric_or_gate_modified": False,
        "data_or_model_changed": False,
    }


__all__ = [
    "NEW_EXTERNAL_BINDING_PATH",
    "NEW_OUTPUT_PATH",
    "NEW_STAGING_PATH",
    "finalize_paet_formal_d_v_evidence_binding_v3",
    "run_paet_formal_d_v_once_v3",
    "validate_paet_formal_d_v_create_only_v3",
]
