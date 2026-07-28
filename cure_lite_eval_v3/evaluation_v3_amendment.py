"""Strict append-only amendment for the PAET D_V evaluation-v3 attempt.

Evaluation-v2 reached the fixed D_V inputs but failed before the first model
forward because the copied evaluator addressed the sealed artifact at
``sources.artifact`` instead of ``sources.attempt.artifact``.  The failed
staging directory and failure receipt remain immutable.  This amendment
authorizes a distinct run identity and exactly one code correction; it does
not authorize resuming or reusing the failed attempt, changing scientific
logic, or accessing D_T.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from cure_lite_eval_v2.evaluation_source_closure import (
    verify_evaluation_source_closure as verify_evaluation_v2_source_closure,
)
from cure_lite_eval_v2.formal800_schema_erratum import (
    verify_formal800_schema_erratum,
)

from .fixed_sample_builder_v3 import (
    CORRECTED_MODEL_ACCESS,
    ORIGINAL_BUILDER_SOURCE_SHA256,
    ORIGINAL_MODEL_ACCESS,
    verify_fixed_sample_builder_correction,
)


EVALUATION_V3_AMENDMENT_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal-d-v-evaluation-v3-amendment-v1"
)
EVALUATION_V3_AMENDMENT_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "paet_bfa_v21_formal_d_v_evaluation_v3_amendment.json"
)
EVALUATION_V3_AMENDMENT_SHA256 = (
    "c88e0cc6304d02b8f6e553765628d26514e3f6809077c3d3665e16df839f154f"
)

FAILED_RUN_ID = "cure_lite_paet_bfa_v21_formal_d_v_seed42_r1"
FAILED_STAGING_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/"
    ".cure_lite_paet_bfa_v21_formal_d_v_seed42_r1.incomplete"
)
FAILURE_RECEIPT_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/"
    "cure_lite_paet_bfa_v21_formal_d_v_seed42_r1."
    "evaluation_v2_failure.json"
)
FAILURE_RECEIPT_SHA256 = (
    "44248fcfe7f5ccbf6f8eebecdcc69bd90c257f066723f571ab16150624397407"
)
FAILURE_FINGERPRINT = (
    "dd0b3f77d9cd36dc484e2c4582914cc65f5221fa02ae6276c80578ff396d1c1f"
)
FAILED_CLAIM_SHA256 = (
    "ee86da60a4b304e60c0306ef0c5aa950db92b07e95d22bea77fef20fa0b2fba6"
)
FAILED_CLAIM_FINGERPRINT = (
    "9b4e9ef66be8fdd01d2536d91188986c9448437243ff88a3101cee22d1bad58d"
)

EVALUATION_V2_CLOSURE_MANIFEST_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_paet_bfa_v21_formal_d_v_evaluation_v2_source_closure.json"
)
EVALUATION_V2_CLOSURE_MANIFEST_SHA256 = (
    "b1195acc8b6fcf69b95684c69507c2d9bddc551ca4532052cd360919f4821329"
)
EVALUATION_V2_CLOSURE_ARCHIVE_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_paet_bfa_v21_formal_d_v_evaluation_v2_source_closure.tar"
)
EVALUATION_V2_CLOSURE_ARCHIVE_SHA256 = (
    "b6fd310027b4229bb76c5a1b45f2a1cc4d703dd365bf481a7405328675e7db23"
)
EVALUATION_V2_CLOSURE_CONTENT_FINGERPRINT = (
    "69d2570169013facd9152008a4a84eb663e7c50afbfe069f4cafc967bbcae197"
)
EVALUATION_V2_CLOSURE_FILE_COUNT = 7

NEW_RUN_ID = "cure_lite_paet_bfa_v21_formal_d_v_seed42_r2"
NEW_OUTPUT_REPO_PATH = f"runs/irstd1k_stage_a_seed42/{NEW_RUN_ID}"
NEW_STAGING_REPO_PATH = (
    f"runs/irstd1k_stage_a_seed42/.{NEW_RUN_ID}.incomplete"
)
NEW_EXTERNAL_BINDING_REPO_PATH = (
    f"runs/irstd1k_stage_a_seed42/{NEW_RUN_ID}."
    "evaluation_v3_evidence_binding.json"
)

ORIGINAL_EVALUATION_SOURCE_REPO_PATH = (
    "cure_lite/experiment/coverage_state_paet_formal_evaluation.py"
)
ORIGINAL_EVALUATION_SOURCE_SHA256 = (
    "98de5c9b71cd23d089ee6643e3d48ac1bceb1c619964b65c316859e0d45717bf"
)
_HEX = frozenset("0123456789abcdef")


class EvaluationV3AmendmentError(RuntimeError):
    """Raised when the amendment or any frozen parent evidence differs."""


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
        raise EvaluationV3AmendmentError(
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


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise EvaluationV3AmendmentError(
            f"{name} must be a lowercase SHA256 digest"
        )
    return value


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
    ):
        raise EvaluationV3AmendmentError(
            f"{name} must be a canonical regular file"
        )

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationV3AmendmentError(
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
        raise EvaluationV3AmendmentError(
            f"{name} is not strict UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise EvaluationV3AmendmentError(f"{name} must be an object")
    return value


def _expected_amendment_body() -> dict[str, object]:
    return {
        "schema_version": EVALUATION_V3_AMENDMENT_SCHEMA,
        "status": "append_only_new_attempt_after_terminal_v2_failure",
        "parent_chain": {
            "original_formal_source_closure": {
                "manifest_sha256": (
                    "1afb838456525f136cfe73a5b52debdd93dc939ed421c743530da69621f190f5"
                ),
                "archive_sha256": (
                    "6b82972662bae12fb4d6892a92e7a6417fc96d94cdad6363ad92514c16f72218"
                ),
                "content_fingerprint": (
                    "fc82768309cce3c7f911f2e4c71e615fc323e2c54bb0e25e99c368acf0b2ba9d"
                ),
                "file_count": 223,
            },
            "formal800_schema_erratum": {
                "repo_path": (
                    "protocols/IRSTD-1K/"
                    "paet_bfa_v21_formal800_schema_name_erratum_v2.json"
                ),
                "sha256": (
                    "66bb284510dce91bd0d1588266b64a28df1a245ad896c821da540a6fd84e65bd"
                ),
                "erratum_fingerprint": (
                    "ceeeedf381d0ddaa4d85f85cebe0fee82d713c03d51ca9ad562f255073306916"
                ),
                "authorized_alias_count": 2,
            },
            "evaluation_v2_source_closure": {
                "manifest_repo_path": (
                    EVALUATION_V2_CLOSURE_MANIFEST_REPO_PATH
                ),
                "manifest_sha256": (
                    EVALUATION_V2_CLOSURE_MANIFEST_SHA256
                ),
                "archive_repo_path": (
                    EVALUATION_V2_CLOSURE_ARCHIVE_REPO_PATH
                ),
                "archive_sha256": (
                    EVALUATION_V2_CLOSURE_ARCHIVE_SHA256
                ),
                "content_fingerprint": (
                    EVALUATION_V2_CLOSURE_CONTENT_FINGERPRINT
                ),
                "file_count": EVALUATION_V2_CLOSURE_FILE_COUNT,
            },
        },
        "failed_evaluation_v2_attempt": {
            "run_id": FAILED_RUN_ID,
            "failure_receipt_repo_path": FAILURE_RECEIPT_REPO_PATH,
            "failure_receipt_sha256": FAILURE_RECEIPT_SHA256,
            "failure_fingerprint": FAILURE_FINGERPRINT,
            "status": "terminal_infrastructure_failure_before_prediction",
            "D_V_accessed": True,
            "D_T_accessed": False,
            "model_forward_calls": 0,
            "prediction_samples_created": False,
            "performance_metrics_computed": False,
            "performance_decision_computed": False,
            "method_failure_established": False,
            "staging_repo_path": FAILED_STAGING_REPO_PATH,
            "staging_inventory": [".incomplete.json"],
            "claim_sha256": FAILED_CLAIM_SHA256,
            "claim_fingerprint": FAILED_CLAIM_FINGERPRINT,
            "staging_reusable": False,
            "resume_allowed": False,
            "same_attempt_retry_allowed": False,
            "failed_evidence_preserved": True,
        },
        "mechanical_correction": {
            "original_source_repo_path": (
                ORIGINAL_EVALUATION_SOURCE_REPO_PATH
            ),
            "original_source_sha256": (
                ORIGINAL_EVALUATION_SOURCE_SHA256
            ),
            "original_builder_source_sha256": (
                ORIGINAL_BUILDER_SOURCE_SHA256
            ),
            "original_function": (
                "cure_lite.experiment."
                "coverage_state_paet_formal_evaluation."
                "build_paet_fixed_d_v_samples"
            ),
            "corrected_function": (
                "cure_lite_eval_v3.fixed_sample_builder_v3."
                "build_paet_fixed_d_v_samples"
            ),
            "from": ORIGINAL_MODEL_ACCESS,
            "to": CORRECTED_MODEL_ACCESS,
            "correction_count": 1,
            "byte_equivalent_after_inverse_substitution": True,
            "ast_equivalent_after_inverse_substitution": True,
        },
        "new_evaluation_v3_attempt": {
            "run_id": NEW_RUN_ID,
            "output_repo_path": NEW_OUTPUT_REPO_PATH,
            "staging_repo_path": NEW_STAGING_REPO_PATH,
            "external_binding_repo_path": (
                NEW_EXTERNAL_BINDING_REPO_PATH
            ),
            "distinct_from_failed_run": True,
            "new_claim_required_before_D_V_materialization": True,
            "failed_staging_reused": False,
            "failed_staging_modified": False,
            "single_attempt": True,
            "resume_allowed": False,
            "retry_allowed": False,
            "overwrite_allowed": False,
        },
        "runtime_patch_scope": {
            "target_module": (
                "cure_lite.experiment."
                "coverage_state_paet_formal_d_v_runner"
            ),
            "symbols": [
                "PAET_FORMAL_DV_RUN_ID",
                "PAET_FORMAL_DV_OUTPUT_PATH",
                "PAET_FORMAL_DV_STAGING_PATH",
                "build_paet_fixed_d_v_samples",
            ],
            "symbol_count": 4,
            "strict_context_only": True,
            "original_values_verified_before_patch": True,
            "original_values_restored_after_context": True,
            "formal800_schema_aliases_reused_from_verified_v2_chain": True,
        },
        "scientific_scope": {
            "model_changed": False,
            "weights_changed": False,
            "training_changed": False,
            "dataset_changed": False,
            "split_changed": False,
            "D_V_population_changed": False,
            "base_cache_changed": False,
            "preprocessing_changed": False,
            "batch_size_changed": False,
            "field_decode_changed": False,
            "inference_changed": False,
            "Base_at_B_grid_changed": False,
            "metric_changed": False,
            "gate_changed": False,
            "model_retraining_authorized": False,
            "D_T_access_authorized": False,
            "Full_CURE_authorized": False,
            "cross_backbone_authorized": False,
        },
        "execution_policy": {
            "evaluation_v3_source_closure_required_before_run": True,
            "validate_create_only_must_not_access_D_V": True,
            "validate_create_only_must_not_access_D_T": True,
            "run_requires_explicit_run_once_invocation": True,
            "new_D_V_output_is_create_once": True,
            "external_evidence_binding_is_create_once": True,
            "D_T_requires_new_D_V_gate_authorization": True,
        },
    }


def expected_evaluation_v3_amendment() -> dict[str, object]:
    """Return the exact fingerprinted evaluation-v3 amendment."""

    body = _expected_amendment_body()
    return {
        **body,
        "amendment_fingerprint": _fingerprint(body),
    }


def verify_evaluation_v3_amendment(
    repository_root: Path | None = None,
) -> dict[str, object]:
    """Verify the amendment, v2 chain, and preserved failure evidence."""

    root = _repository_root(repository_root)
    erratum = verify_formal800_schema_erratum(root)
    evaluation_v2 = verify_evaluation_v2_source_closure(root)
    expected = expected_evaluation_v3_amendment()
    amendment_path = root / EVALUATION_V3_AMENDMENT_REPO_PATH
    amendment = _strict_json(
        amendment_path,
        name="evaluation-v3 amendment",
    )
    if (
        amendment != expected
        or amendment_path.read_bytes()
        != _canonical_json(expected) + b"\n"
        or _file_sha256(amendment_path)
        != EVALUATION_V3_AMENDMENT_SHA256
    ):
        raise EvaluationV3AmendmentError(
            "evaluation-v3 amendment bytes changed"
        )
    body = dict(amendment)
    fingerprint = _digest(
        body.pop("amendment_fingerprint", None),
        name="amendment fingerprint",
    )
    if fingerprint != _fingerprint(body):
        raise EvaluationV3AmendmentError(
            "evaluation-v3 amendment fingerprint changed"
        )

    parent = expected["parent_chain"]
    assert isinstance(parent, Mapping)
    original = parent["original_formal_source_closure"]
    schema_erratum = parent["formal800_schema_erratum"]
    v2_binding = parent["evaluation_v2_source_closure"]
    assert isinstance(original, Mapping)
    assert isinstance(schema_erratum, Mapping)
    assert isinstance(v2_binding, Mapping)
    if (
        erratum["original_source_closure_manifest_sha256"]
        != original["manifest_sha256"]
        or erratum["original_source_closure_archive_sha256"]
        != original["archive_sha256"]
        or erratum["original_source_closure_content_fingerprint"]
        != original["content_fingerprint"]
        or erratum["original_source_closure_file_count"]
        != original["file_count"]
        or erratum["erratum_sha256"] != schema_erratum["sha256"]
        or erratum["erratum_fingerprint"]
        != schema_erratum["erratum_fingerprint"]
        or erratum["authorized_alias_count"]
        != schema_erratum["authorized_alias_count"]
        or evaluation_v2["manifest_sha256"]
        != v2_binding["manifest_sha256"]
        or evaluation_v2["archive_sha256"]
        != v2_binding["archive_sha256"]
        or evaluation_v2["content_fingerprint"]
        != v2_binding["content_fingerprint"]
        or evaluation_v2["file_count"] != v2_binding["file_count"]
    ):
        raise EvaluationV3AmendmentError(
            "evaluation-v3 parent closure chain changed"
        )

    failure_path = root / FAILURE_RECEIPT_REPO_PATH
    failure = _strict_json(
        failure_path,
        name="evaluation-v2 failure receipt",
    )
    boundary = failure.get("execution_boundary")
    output_state = failure.get("output_state")
    scientific = failure.get("scientific_status")
    if (
        _file_sha256(failure_path) != FAILURE_RECEIPT_SHA256
        or failure.get("schema_version")
        != "cure-lite-paet-bfa-v21-formal-d-v-evaluation-v2-failure-v1"
        or failure.get("run_id") != FAILED_RUN_ID
        or failure.get("status")
        != "terminal_infrastructure_failure_before_prediction"
        or failure.get("failure_fingerprint") != FAILURE_FINGERPRINT
        or not isinstance(boundary, Mapping)
        or boundary.get("D_V_accessed") is not True
        or boundary.get("D_T_accessed") is not False
        or boundary.get("model_forward_calls") != 0
        or boundary.get("prediction_samples_created") is not False
        or boundary.get("performance_metrics_computed") is not False
        or boundary.get("performance_decision_computed") is not False
        or not isinstance(output_state, Mapping)
        or output_state.get("staging_repo_path")
        != FAILED_STAGING_REPO_PATH
        or output_state.get("staging_inventory")
        != [".incomplete.json"]
        or output_state.get("claim_sha256") != FAILED_CLAIM_SHA256
        or output_state.get("claim_fingerprint")
        != FAILED_CLAIM_FINGERPRINT
        or output_state.get("staging_reusable") is not False
        or not isinstance(scientific, Mapping)
        or scientific.get("method_failure_established") is not False
        or scientific.get("performance_claim_supported") is not False
    ):
        raise EvaluationV3AmendmentError(
            "evaluation-v2 failure evidence changed"
        )

    failed_staging = root / FAILED_STAGING_REPO_PATH
    if (
        failed_staging.is_symlink()
        or not failed_staging.is_dir()
        or failed_staging.resolve(strict=True) != failed_staging
        or {item.name for item in failed_staging.iterdir()}
        != {".incomplete.json"}
        or _file_sha256(failed_staging / ".incomplete.json")
        != FAILED_CLAIM_SHA256
    ):
        raise EvaluationV3AmendmentError(
            "failed evaluation-v2 staging evidence changed"
        )
    original_source = root / ORIGINAL_EVALUATION_SOURCE_REPO_PATH
    if (
        original_source.is_symlink()
        or not original_source.is_file()
        or _file_sha256(original_source)
        != ORIGINAL_EVALUATION_SOURCE_SHA256
    ):
        raise EvaluationV3AmendmentError(
            "original fixed-sample builder source changed"
        )
    correction = verify_fixed_sample_builder_correction()
    return {
        "schema_version": EVALUATION_V3_AMENDMENT_SCHEMA,
        "verified": True,
        "amendment_repo_path": EVALUATION_V3_AMENDMENT_REPO_PATH,
        "amendment_sha256": EVALUATION_V3_AMENDMENT_SHA256,
        "amendment_fingerprint": fingerprint,
        "formal_complete_sha256": erratum["formal_complete_sha256"],
        "formal_complete_fingerprint": (
            erratum["formal_complete_fingerprint"]
        ),
        "original_source_closure_manifest_sha256": (
            erratum["original_source_closure_manifest_sha256"]
        ),
        "original_source_closure_archive_sha256": (
            erratum["original_source_closure_archive_sha256"]
        ),
        "original_source_closure_content_fingerprint": (
            erratum["original_source_closure_content_fingerprint"]
        ),
        "original_source_closure_file_count": (
            erratum["original_source_closure_file_count"]
        ),
        "schema_erratum_repo_path": erratum["erratum_repo_path"],
        "schema_erratum_sha256": erratum["erratum_sha256"],
        "schema_erratum_fingerprint": erratum["erratum_fingerprint"],
        "authorized_alias_count": erratum["authorized_alias_count"],
        "evaluation_v2_closure_manifest_sha256": (
            evaluation_v2["manifest_sha256"]
        ),
        "evaluation_v2_closure_archive_sha256": (
            evaluation_v2["archive_sha256"]
        ),
        "evaluation_v2_closure_content_fingerprint": (
            evaluation_v2["content_fingerprint"]
        ),
        "evaluation_v2_closure_file_count": evaluation_v2["file_count"],
        "failure_receipt_repo_path": FAILURE_RECEIPT_REPO_PATH,
        "failure_receipt_sha256": FAILURE_RECEIPT_SHA256,
        "failure_fingerprint": FAILURE_FINGERPRINT,
        "prior_D_V_accessed": True,
        "prior_D_T_accessed": False,
        "prior_model_forward_calls": 0,
        "failed_staging_preserved": True,
        "new_run_id": NEW_RUN_ID,
        "new_output_repo_path": NEW_OUTPUT_REPO_PATH,
        "new_staging_repo_path": NEW_STAGING_REPO_PATH,
        "new_external_binding_repo_path": (
            NEW_EXTERNAL_BINDING_REPO_PATH
        ),
        "builder_correction": correction,
        "metric_or_gate_changed": False,
        "data_or_model_changed": False,
    }


__all__ = [
    "EVALUATION_V3_AMENDMENT_REPO_PATH",
    "EVALUATION_V3_AMENDMENT_SHA256",
    "FAILED_RUN_ID",
    "FAILED_STAGING_REPO_PATH",
    "FAILURE_RECEIPT_REPO_PATH",
    "NEW_EXTERNAL_BINDING_REPO_PATH",
    "NEW_OUTPUT_REPO_PATH",
    "NEW_RUN_ID",
    "NEW_STAGING_REPO_PATH",
    "EvaluationV3AmendmentError",
    "expected_evaluation_v3_amendment",
    "verify_evaluation_v3_amendment",
]
