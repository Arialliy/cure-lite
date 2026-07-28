"""Strict append-only erratum for two Formal800 schema-name mismatches.

The completed Formal800 bytes and its original source closure are immutable.
The producer wrote two schema names containing ``pmope`` while the later
strict consumer declared the otherwise identical names without ``pmope``.
This erratum authorizes exactly those two name aliases in memory.  It does
not authorize editing an artifact, weakening another check, retraining, or
changing any metric, gate, model, loss, schedule, or inference rule.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


ERRATUM_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal800-schema-name-erratum-v2"
)
ERRATUM_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "paet_bfa_v21_formal800_schema_name_erratum_v2.json"
)
ERRATUM_SHA256 = (
    "66bb284510dce91bd0d1588266b64a28df1a245ad896c821da540a6fd84e65bd"
)
FORMAL_RUN_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/"
    "cure_lite_paet_bfa_v21_pmope_formal_800_seed42_r1"
)
ORIGINAL_CLOSURE_MANIFEST_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_paet_bfa_v21_pmope_formal800_source_closure.json"
)
ORIGINAL_CLOSURE_ARCHIVE_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_paet_bfa_v21_pmope_formal800_source_closure.tar"
)

ORIGINAL_CLOSURE_MANIFEST_SHA256 = (
    "1afb838456525f136cfe73a5b52debdd93dc939ed421c743530da69621f190f5"
)
ORIGINAL_CLOSURE_ARCHIVE_SHA256 = (
    "6b82972662bae12fb4d6892a92e7a6417fc96d94cdad6363ad92514c16f72218"
)
ORIGINAL_CLOSURE_CONTENT_FINGERPRINT = (
    "fc82768309cce3c7f911f2e4c71e615fc323e2c54bb0e25e99c368acf0b2ba9d"
)
ORIGINAL_CLOSURE_FILE_COUNT = 223

FORMAL_COMPLETE_SHA256 = (
    "e5a75a171d455bb31d8194b65981cdd618f8a26f49fa5a224334f9f14d182f1a"
)
FORMAL_COMPLETE_FINGERPRINT = (
    "116fbe67c5f9ec74cc2317ee9dc283845b4da9f5699cd07537b72917338bb74c"
)
FORMAL_ATTEMPT_SHA256 = (
    "4647599403f8f2a475e94d89cff51ee12a05a652764ba52a218115855a0988bf"
)
FORMAL_ATTEMPT_FINGERPRINT = (
    "323946c94e40935ba315e569b13744b59636660d2d88b99556d1b49010ae766a"
)
FORMAL_STARTED_SHA256 = (
    "07aae4cb12144db8f00822a5a4166d18101ae379d419f7db67076049c8f36ea3"
)
FORMAL_STARTED_FINGERPRINT = (
    "3e8d6fca2c3708f52b12cfa1df1d9d167d60901ea573099d77308ec1c2f723d5"
)

ATTEMPT_CONSUMER_SCHEMA = "cure-lite-paet-bfa-v21-formal800-attempt-v1"
ATTEMPT_PRODUCER_SCHEMA = (
    "cure-lite-paet-bfa-v21-pmope-formal800-attempt-v1"
)
STARTED_CONSUMER_SCHEMA = "cure-lite-paet-bfa-v21-formal800-started-v1"
STARTED_PRODUCER_SCHEMA = (
    "cure-lite-paet-bfa-v21-pmope-formal800-started-v1"
)

_HEX = frozenset("0123456789abcdef")


class Formal800SchemaErratumError(RuntimeError):
    """Raised when the append-only erratum or its frozen parents differ."""


def _repository_root(repository_root: Path | None) -> Path:
    root = (
        Path(repository_root)
        if repository_root is not None
        else Path(__file__).resolve().parents[1]
    )
    absolute = Path(os.path.abspath(root))
    if (
        absolute.is_symlink()
        or not absolute.is_dir()
        or absolute.resolve(strict=True) != absolute
    ):
        raise Formal800SchemaErratumError(
            "repository root must be a canonical directory"
        )
    return absolute


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
    ):
        raise Formal800SchemaErratumError(
            f"{name} must be a canonical regular file"
        )

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise Formal800SchemaErratumError(
                    f"{name} contains duplicate JSON keys"
                )
            result[key] = value
        return result

    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Formal800SchemaErratumError(
            f"{name} is not strict UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise Formal800SchemaErratumError(
            f"{name} must be a JSON object"
        )
    return value


def _expected_erratum_body() -> dict[str, object]:
    return {
        "schema_version": ERRATUM_SCHEMA,
        "status": "append_only_frozen_artifact_interpretation_erratum",
        "formal_run": {
            "run_id": (
                "cure_lite_paet_bfa_v21_pmope_formal_800_seed42_r1"
            ),
            "repo_path": FORMAL_RUN_REPO_PATH,
            "COMPLETE_sha256": FORMAL_COMPLETE_SHA256,
            "complete_fingerprint": FORMAL_COMPLETE_FINGERPRINT,
            "formal_training_complete": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
        "original_source_closure": {
            "manifest_repo_path": (
                ORIGINAL_CLOSURE_MANIFEST_REPO_PATH
            ),
            "manifest_sha256": ORIGINAL_CLOSURE_MANIFEST_SHA256,
            "archive_repo_path": ORIGINAL_CLOSURE_ARCHIVE_REPO_PATH,
            "archive_sha256": ORIGINAL_CLOSURE_ARCHIVE_SHA256,
            "content_fingerprint": (
                ORIGINAL_CLOSURE_CONTENT_FINGERPRINT
            ),
            "file_count": ORIGINAL_CLOSURE_FILE_COUNT,
        },
        "producer_receipts": [
            {
                "repo_path": f"{FORMAL_RUN_REPO_PATH}/attempt.json",
                "file_sha256": FORMAL_ATTEMPT_SHA256,
                "fingerprint_field": "receipt_fingerprint",
                "fingerprint": FORMAL_ATTEMPT_FINGERPRINT,
                "producer_schema": ATTEMPT_PRODUCER_SCHEMA,
            },
            {
                "repo_path": f"{FORMAL_RUN_REPO_PATH}/STARTED.json",
                "file_sha256": FORMAL_STARTED_SHA256,
                "fingerprint_field": "receipt_fingerprint",
                "fingerprint": FORMAL_STARTED_FINGERPRINT,
                "producer_schema": STARTED_PRODUCER_SCHEMA,
            },
        ],
        "authorized_in_memory_aliases": [
            {
                "consumer_module": (
                    "cure_lite.experiment."
                    "coverage_state_paet_formal_attempt"
                ),
                "consumer_symbol": "PAET_FORMAL_ATTEMPT_SCHEMA",
                "consumer_schema": ATTEMPT_CONSUMER_SCHEMA,
                "producer_schema": ATTEMPT_PRODUCER_SCHEMA,
                "receipt_repo_path": (
                    f"{FORMAL_RUN_REPO_PATH}/attempt.json"
                ),
            },
            {
                "consumer_module": (
                    "cure_lite.experiment."
                    "coverage_state_paet_formal_attempt"
                ),
                "consumer_symbol": "PAET_FORMAL_STARTED_SCHEMA",
                "consumer_schema": STARTED_CONSUMER_SCHEMA,
                "producer_schema": STARTED_PRODUCER_SCHEMA,
                "receipt_repo_path": (
                    f"{FORMAL_RUN_REPO_PATH}/STARTED.json"
                ),
            },
        ],
        "alias_count": 2,
        "delegation": {
            "loader": (
                "cure_lite.experiment."
                "coverage_state_paet_formal_attempt."
                "load_coverage_state_paet_formal_attempt"
            ),
            "all_non_schema_checks_remain_in_original_loader": True,
            "payload_bytes_are_not_patched": True,
            "constants_are_restored_after_call": True,
        },
        "scientific_scope": {
            "model_changed": False,
            "weights_changed": False,
            "training_changed": False,
            "data_changed": False,
            "metric_changed": False,
            "gate_changed": False,
            "inference_changed": False,
            "retraining_authorized": False,
            "artifact_rewrite_authorized": False,
            "additional_aliases_authorized": False,
        },
    }


def expected_formal800_schema_erratum() -> dict[str, object]:
    """Return the exact fingerprinted erratum payload."""

    body = _expected_erratum_body()
    return {**body, "erratum_fingerprint": _fingerprint(body)}


def verify_formal800_schema_erratum(
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Verify the erratum, original closure, and frozen Formal800 receipts."""

    root = _repository_root(repository_root)
    erratum_path = root / ERRATUM_REPO_PATH
    payload = _strict_json(erratum_path, name="Formal800 schema erratum")
    expected = expected_formal800_schema_erratum()
    if (
        payload != expected
        or _file_sha256(erratum_path) != ERRATUM_SHA256
        or erratum_path.read_bytes()
        != _canonical_json(expected) + b"\n"
    ):
        raise Formal800SchemaErratumError(
            "Formal800 schema erratum payload changed"
        )
    body = dict(payload)
    actual_fingerprint = body.pop("erratum_fingerprint", None)
    if actual_fingerprint != _fingerprint(body):
        raise Formal800SchemaErratumError(
            "Formal800 schema erratum fingerprint changed"
        )

    from cure_lite.coverage_state_paet_formal_source_closure import (
        verify_coverage_state_paet_formal_source_closure,
    )

    original = verify_coverage_state_paet_formal_source_closure(root)
    expected_original = expected["original_source_closure"]
    assert isinstance(expected_original, Mapping)
    if (
        original.get("sealed") is not True
        or original.get("manifest_repo_path")
        != expected_original["manifest_repo_path"]
        or original.get("manifest_sha256")
        != expected_original["manifest_sha256"]
        or original.get("archive_repo_path")
        != expected_original["archive_repo_path"]
        or original.get("archive_sha256")
        != expected_original["archive_sha256"]
        or original.get("content_fingerprint")
        != expected_original["content_fingerprint"]
        or original.get("file_count") != expected_original["file_count"]
    ):
        raise Formal800SchemaErratumError(
            "original Formal800 source closure binding changed"
        )

    for filename, expected_sha, expected_schema, expected_receipt in (
        (
            "attempt.json",
            FORMAL_ATTEMPT_SHA256,
            ATTEMPT_PRODUCER_SCHEMA,
            FORMAL_ATTEMPT_FINGERPRINT,
        ),
        (
            "STARTED.json",
            FORMAL_STARTED_SHA256,
            STARTED_PRODUCER_SCHEMA,
            FORMAL_STARTED_FINGERPRINT,
        ),
    ):
        path = root / FORMAL_RUN_REPO_PATH / filename
        receipt = _strict_json(path, name=f"Formal800 {filename}")
        if (
            _file_sha256(path) != expected_sha
            or receipt.get("schema_version") != expected_schema
            or receipt.get("receipt_fingerprint") != expected_receipt
        ):
            raise Formal800SchemaErratumError(
                f"frozen Formal800 {filename} binding changed"
            )

    complete_path = root / FORMAL_RUN_REPO_PATH / "COMPLETE.json"
    complete = _strict_json(complete_path, name="Formal800 COMPLETE")
    if (
        _file_sha256(complete_path) != FORMAL_COMPLETE_SHA256
        or complete.get("complete_fingerprint")
        != FORMAL_COMPLETE_FINGERPRINT
        or complete.get("formal_training_complete") is not True
        or complete.get("D_V_accessed") is not False
        or complete.get("D_T_accessed") is not False
    ):
        raise Formal800SchemaErratumError(
            "frozen Formal800 COMPLETE binding changed"
        )
    return {
        "schema_version": (
            "cure-lite-paet-bfa-v21-formal800-schema-erratum-receipt-v2"
        ),
        "verified": True,
        "erratum_repo_path": ERRATUM_REPO_PATH,
        "erratum_sha256": ERRATUM_SHA256,
        "erratum_fingerprint": actual_fingerprint,
        "formal_complete_sha256": FORMAL_COMPLETE_SHA256,
        "formal_complete_fingerprint": FORMAL_COMPLETE_FINGERPRINT,
        "original_source_closure_manifest_sha256": (
            ORIGINAL_CLOSURE_MANIFEST_SHA256
        ),
        "original_source_closure_archive_sha256": (
            ORIGINAL_CLOSURE_ARCHIVE_SHA256
        ),
        "original_source_closure_content_fingerprint": (
            ORIGINAL_CLOSURE_CONTENT_FINGERPRINT
        ),
        "original_source_closure_file_count": (
            ORIGINAL_CLOSURE_FILE_COUNT
        ),
        "authorized_alias_count": 2,
    }


__all__ = [
    "ATTEMPT_CONSUMER_SCHEMA",
    "ATTEMPT_PRODUCER_SCHEMA",
    "ERRATUM_REPO_PATH",
    "Formal800SchemaErratumError",
    "STARTED_CONSUMER_SCHEMA",
    "STARTED_PRODUCER_SCHEMA",
    "expected_formal800_schema_erratum",
    "verify_formal800_schema_erratum",
]
