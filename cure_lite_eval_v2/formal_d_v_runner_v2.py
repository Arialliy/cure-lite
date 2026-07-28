"""Append-only v2 wrapper around the sealed PAET Formal800 D_V runner.

Only two exact producer schema names are substituted into the original
consumer module, and only for the duration of the original loader/runner
call.  No artifact payload is edited.  The original loader consequently
performs every remaining check unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
from typing import Any, Iterator, Mapping

from cure_lite.cache.schema import file_sha256, stable_fingerprint
import cure_lite.experiment.coverage_state_paet_formal_attempt as _attempt
import cure_lite.experiment.coverage_state_paet_formal_d_v_runner as _d_v

from .evaluation_source_closure import verify_evaluation_source_closure
from .formal800_schema_erratum import (
    ATTEMPT_CONSUMER_SCHEMA,
    ATTEMPT_PRODUCER_SCHEMA,
    STARTED_CONSUMER_SCHEMA,
    STARTED_PRODUCER_SCHEMA,
    verify_formal800_schema_erratum,
)


EVALUATION_RECOVERY_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal-d-v-evaluation-recovery-v2"
)
EVIDENCE_BINDING_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal-d-v-external-evidence-binding-v2"
)
_REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_BINDING_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/"
    f"{_d_v.PAET_FORMAL_DV_RUN_ID}.evaluation_v2_evidence_binding.json"
)
EVIDENCE_BINDING_PATH = _REPO_ROOT / EVIDENCE_BINDING_REPO_PATH
EVIDENCE_BINDING_STAGING_PATH = EVIDENCE_BINDING_PATH.with_name(
    f".{EVIDENCE_BINDING_PATH.name}.incomplete"
)
_D_V_MEMBER_NAMES = (
    "COMPLETE.json",
    "decision.json",
    "receipt.json",
)
_ALIAS_LOCK = threading.RLock()


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
                raise RuntimeError(f"{name} contains duplicate keys")
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


def _validate_closure_chain() -> dict[str, object]:
    erratum = verify_formal800_schema_erratum(_REPO_ROOT)
    evaluation = verify_evaluation_source_closure(_REPO_ROOT)
    original = evaluation.get("original_formal_source_closure")
    evaluation_erratum = evaluation.get("formal800_schema_erratum")
    if (
        not isinstance(original, dict)
        or not isinstance(evaluation_erratum, dict)
        or original.get("manifest_sha256")
        != erratum["original_source_closure_manifest_sha256"]
        or original.get("archive_sha256")
        != erratum["original_source_closure_archive_sha256"]
        or original.get("content_fingerprint")
        != erratum["original_source_closure_content_fingerprint"]
        or original.get("file_count")
        != erratum["original_source_closure_file_count"]
        or evaluation_erratum.get("repo_path")
        != erratum["erratum_repo_path"]
        or evaluation_erratum.get("sha256")
        != erratum["erratum_sha256"]
        or evaluation_erratum.get("erratum_fingerprint")
        != erratum["erratum_fingerprint"]
    ):
        raise RuntimeError(
            "evaluation closure is not bound to the verified erratum chain"
        )
    return {"erratum": erratum, "evaluation_closure": evaluation}


@contextmanager
def _exact_formal800_schema_aliases() -> Iterator[None]:
    """Temporarily repair only the two names listed in the erratum."""

    with _ALIAS_LOCK:
        before = {
            "PAET_FORMAL_ATTEMPT_SCHEMA": getattr(
                _attempt,
                "PAET_FORMAL_ATTEMPT_SCHEMA",
                None,
            ),
            "PAET_FORMAL_STARTED_SCHEMA": getattr(
                _attempt,
                "PAET_FORMAL_STARTED_SCHEMA",
                None,
            ),
        }
        expected_before = {
            "PAET_FORMAL_ATTEMPT_SCHEMA": ATTEMPT_CONSUMER_SCHEMA,
            "PAET_FORMAL_STARTED_SCHEMA": STARTED_CONSUMER_SCHEMA,
        }
        if before != expected_before:
            raise RuntimeError(
                "original Formal800 consumer constants changed; "
                "the two-alias erratum cannot be applied"
            )
        _attempt.PAET_FORMAL_ATTEMPT_SCHEMA = ATTEMPT_PRODUCER_SCHEMA
        _attempt.PAET_FORMAL_STARTED_SCHEMA = STARTED_PRODUCER_SCHEMA
        try:
            yield
        finally:
            _attempt.PAET_FORMAL_ATTEMPT_SCHEMA = before[
                "PAET_FORMAL_ATTEMPT_SCHEMA"
            ]
            _attempt.PAET_FORMAL_STARTED_SCHEMA = before[
                "PAET_FORMAL_STARTED_SCHEMA"
            ]


def _load_original_attempt_with_aliases():
    with _exact_formal800_schema_aliases():
        loaded = _attempt.load_coverage_state_paet_formal_attempt()
    if type(loaded) is not _attempt.LoadedCoverageStatePAETFormalAttempt:
        raise TypeError(
            "original Formal800 loader returned a substituted object"
        )
    loaded.verify_unchanged()
    return loaded


def load_coverage_state_paet_formal_attempt_v2():
    """Verify both closures, then invoke the original strict loader."""

    _validate_closure_chain()
    return _load_original_attempt_with_aliases()


def validate_paet_formal_d_v_create_only_v2() -> dict[str, object]:
    """Validate recovery without opening D_V/D_T or creating any output."""

    chain = _validate_closure_chain()
    loaded = _load_original_attempt_with_aliases()
    original_plan = _d_v.validate_paet_formal_d_v_create_only()
    if (
        original_plan.get("D_V_accessed") is not False
        or original_plan.get("D_T_accessed") is not False
        or original_plan.get("output_created") is not False
        or original_plan.get("create_only") is not True
    ):
        raise RuntimeError(
            "original D_V create-only contract unexpectedly changed"
        )
    body = {
        "schema_version": EVALUATION_RECOVERY_SCHEMA,
        "mode": "validate_create_only",
        "status": "valid",
        "alias_policy": {
            "count": 2,
            "payload_bytes_patched": False,
            "constants_restored": (
                _attempt.PAET_FORMAL_ATTEMPT_SCHEMA
                == ATTEMPT_CONSUMER_SCHEMA
                and _attempt.PAET_FORMAL_STARTED_SCHEMA
                == STARTED_CONSUMER_SCHEMA
            ),
            "all_other_checks_delegated_to_original_loader": True,
        },
        "formal_attempt": {
            "complete_fingerprint": loaded.complete_fingerprint,
            "artifact_fingerprint": (
                loaded.artifact.artifact_fingerprint
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
        "erratum": chain["erratum"],
        "evaluation_source_closure": chain["evaluation_closure"],
        "original_create_only_plan": original_plan,
        "Formal800_attempt_loaded": True,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "output_created": False,
    }
    return {**body, "validation_fingerprint": stable_fingerprint(body)}


def _validated_d_v_files() -> tuple[
    dict[str, object],
    dict[str, dict[str, object]],
]:
    original_result = _d_v._validate_published_output(  # noqa: SLF001
        _d_v.PAET_FORMAL_DV_OUTPUT_PATH
    )
    root = _d_v.PAET_FORMAL_DV_OUTPUT_PATH
    members = {path.name: path for path in root.iterdir()}
    if set(members) != set(_D_V_MEMBER_NAMES):
        raise RuntimeError("published D_V inventory changed")
    rows: dict[str, dict[str, object]] = {}
    for name in _D_V_MEMBER_NAMES:
        path = members[name]
        rows[name] = {
            "repo_path": path.relative_to(_REPO_ROOT).as_posix(),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
    return original_result, rows


def _build_external_evidence_binding(
    chain: Mapping[str, object],
) -> dict[str, object]:
    original_result, files = _validated_d_v_files()
    complete = _strict_json(
        _d_v.PAET_FORMAL_DV_OUTPUT_PATH / "COMPLETE.json",
        name="published D_V COMPLETE",
    )
    receipt = _strict_json(
        _d_v.PAET_FORMAL_DV_OUTPUT_PATH / "receipt.json",
        name="published D_V receipt",
    )
    decision = _strict_json(
        _d_v.PAET_FORMAL_DV_OUTPUT_PATH / "decision.json",
        name="published D_V decision",
    )
    erratum = chain.get("erratum")
    evaluation = chain.get("evaluation_closure")
    if not isinstance(erratum, Mapping) or not isinstance(
        evaluation,
        Mapping,
    ):
        raise TypeError("verified closure chain is malformed")
    body = {
        "schema_version": EVIDENCE_BINDING_SCHEMA,
        "status": "complete",
        "binding_is_external_to_immutable_D_V_output": True,
        "D_V": {
            "run_id": _d_v.PAET_FORMAL_DV_RUN_ID,
            "output_repo_path": (
                _d_v.PAET_FORMAL_DV_OUTPUT_PATH.relative_to(
                    _REPO_ROOT
                ).as_posix()
            ),
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
            "gate_passed": original_result["gate_passed"],
            "authorizes_D_T": original_result["authorizes_D_T"],
            "D_T_accessed": False,
        },
        "Formal800": {
            "complete_sha256": erratum["formal_complete_sha256"],
            "complete_fingerprint": (
                erratum["formal_complete_fingerprint"]
            ),
        },
        "original_formal_source_closure": {
            "manifest_sha256": (
                erratum[
                    "original_source_closure_manifest_sha256"
                ]
            ),
            "archive_sha256": (
                erratum[
                    "original_source_closure_archive_sha256"
                ]
            ),
            "content_fingerprint": (
                erratum[
                    "original_source_closure_content_fingerprint"
                ]
            ),
            "file_count": (
                erratum["original_source_closure_file_count"]
            ),
        },
        "schema_erratum": {
            "repo_path": erratum["erratum_repo_path"],
            "sha256": erratum["erratum_sha256"],
            "erratum_fingerprint": erratum["erratum_fingerprint"],
            "authorized_alias_count": erratum[
                "authorized_alias_count"
            ],
        },
        "evaluation_source_closure": {
            "manifest_repo_path": evaluation[
                "manifest_repo_path"
            ],
            "manifest_sha256": evaluation["manifest_sha256"],
            "archive_repo_path": evaluation["archive_repo_path"],
            "archive_sha256": evaluation["archive_sha256"],
            "content_fingerprint": evaluation[
                "content_fingerprint"
            ],
            "file_count": evaluation["file_count"],
        },
        "recovery_scope": {
            "in_memory_schema_alias_count": 2,
            "artifact_bytes_modified": False,
            "original_D_V_output_modified": False,
            "model_retrained": False,
            "model_state_updated": False,
            "metric_or_gate_modified": False,
        },
    }
    return {**body, "binding_fingerprint": stable_fingerprint(body)}


def _publish_external_binding(
    payload: Mapping[str, object],
) -> None:
    if EVIDENCE_BINDING_PATH.exists() or EVIDENCE_BINDING_PATH.is_symlink():
        raise FileExistsError(
            "external evaluation-v2 evidence binding already exists"
        )
    if (
        EVIDENCE_BINDING_STAGING_PATH.exists()
        or EVIDENCE_BINDING_STAGING_PATH.is_symlink()
    ):
        raise FileExistsError(
            "external evidence-binding staging file already exists"
        )
    EVIDENCE_BINDING_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical_json(payload)
    descriptor = os.open(
        EVIDENCE_BINDING_STAGING_PATH,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    os.link(
        EVIDENCE_BINDING_STAGING_PATH,
        EVIDENCE_BINDING_PATH,
        follow_symlinks=False,
    )
    directory = os.open(
        EVIDENCE_BINDING_PATH.parent,
        os.O_RDONLY | os.O_DIRECTORY,
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    EVIDENCE_BINDING_STAGING_PATH.unlink()


def _verify_external_binding(
    expected: Mapping[str, object],
) -> dict[str, object]:
    actual = _strict_json(
        EVIDENCE_BINDING_PATH,
        name="evaluation-v2 external evidence binding",
    )
    if actual != expected:
        raise RuntimeError(
            "evaluation-v2 external evidence binding changed"
        )
    body = dict(actual)
    fingerprint = body.pop("binding_fingerprint", None)
    if fingerprint != stable_fingerprint(body):
        raise RuntimeError(
            "evaluation-v2 evidence-binding fingerprint changed"
        )
    return {
        "repo_path": EVIDENCE_BINDING_REPO_PATH,
        "sha256": file_sha256(EVIDENCE_BINDING_PATH),
        "binding_fingerprint": fingerprint,
    }


def finalize_paet_formal_d_v_evidence_binding_v2() -> dict[str, object]:
    """Finalize only the external receipt after a completed one-shot D_V run."""

    chain = _validate_closure_chain()
    expected = _build_external_evidence_binding(chain)
    if EVIDENCE_BINDING_PATH.exists():
        return _verify_external_binding(expected)
    _publish_external_binding(expected)
    return _verify_external_binding(expected)


def run_paet_formal_d_v_once_v2() -> dict[str, object]:
    """Run the original fixed D_V once under the exact two-alias erratum."""

    chain = _validate_closure_chain()
    if (
        EVIDENCE_BINDING_PATH.exists()
        or EVIDENCE_BINDING_PATH.is_symlink()
        or EVIDENCE_BINDING_STAGING_PATH.exists()
        or EVIDENCE_BINDING_STAGING_PATH.is_symlink()
    ):
        raise FileExistsError(
            "evaluation-v2 external binding destination is not fresh"
        )
    with _exact_formal800_schema_aliases():
        original_result = _d_v.run_paet_formal_d_v_once()
    if (
        _attempt.PAET_FORMAL_ATTEMPT_SCHEMA
        != ATTEMPT_CONSUMER_SCHEMA
        or _attempt.PAET_FORMAL_STARTED_SCHEMA
        != STARTED_CONSUMER_SCHEMA
    ):
        raise RuntimeError(
            "original Formal800 consumer constants were not restored"
        )
    external = _build_external_evidence_binding(chain)
    _publish_external_binding(external)
    binding = _verify_external_binding(external)
    return {
        "schema_version": EVALUATION_RECOVERY_SCHEMA,
        "mode": "run_once",
        "status": "complete",
        "original_D_V_result": original_result,
        "external_evidence_binding": binding,
        "D_T_accessed": False,
        "model_retrained": False,
        "artifact_bytes_modified": False,
    }


__all__ = [
    "EVIDENCE_BINDING_PATH",
    "EVIDENCE_BINDING_REPO_PATH",
    "finalize_paet_formal_d_v_evidence_binding_v2",
    "load_coverage_state_paet_formal_attempt_v2",
    "run_paet_formal_d_v_once_v2",
    "validate_paet_formal_d_v_create_only_v2",
]
