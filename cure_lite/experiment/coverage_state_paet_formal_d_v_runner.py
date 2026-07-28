"""One-shot fixed D_V evaluation for the completed PAET-BFA Formal800 run.

The public runner has no caller-selected path, seed, threshold, split, model,
or batch-size argument.  It first authenticates the sole completed Formal800
attempt without opening D_V, then atomically claims a fixed staging directory.
Only after that claim does it open the frozen IRSTD-1K D_V manifest/cache and
evaluate the single PAET completion field

``completion = (field < 0) & ~occupancy``.

The only threshold selection performed is the already frozen 51-point
Base@B probability control.  No PAET threshold is searched, no model is
trained, and no D_T sample is opened.  A failed staging directory is never
resumed or reused.  Success publishes exactly ``receipt.json``,
``decision.json``, and ``COMPLETE.json`` by an atomic no-replace rename.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from ..cache.schema import file_sha256, stable_fingerprint
from ..data import ManifestImageDataset, PreprocessConfig
from ..splits import load_and_validate_manifest
from .cache_pipeline import LoadedDVCacheBundle, load_d_v_cache_bundle
from .coverage_state_paet_formal_attempt import (
    LoadedCoverageStatePAETFormalAttempt,
    load_coverage_state_paet_formal_attempt,
)
from .coverage_state_paet_formal_decision import (
    assess_paet_formal_d_v_result,
)
from .coverage_state_paet_formal_evaluation import (
    PAET_BASE_AT_B_SELECTION_POLICY,
    PAET_FIXED_OUTPUT_RULE,
    PAET_FORMAL_BASE_THRESHOLD_GRID,
    PAET_FORMAL_METHOD,
    PAET_FORMAL_SEED,
    PAET_FORMAL_STAGE_A_CONFIG_SHA256,
    PAET_ZERO_TIE_POLICY,
    bind_paet_formal_artifact,
    build_paet_fixed_d_v_samples,
    evaluate_paet_formal_d_v,
)
from .paired_formal_evaluation import (
    FrozenComparisonProtocol,
    load_frozen_comparison_protocol,
)


PAET_FORMAL_DV_RUN_ID = (
    "cure_lite_paet_bfa_v21_formal_d_v_seed42_r1"
)
PAET_FORMAL_DV_RUN_RECEIPT_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal-d-v-run-receipt-v1"
)
PAET_FORMAL_DV_COMPLETE_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal-d-v-complete-v1"
)
PAET_FORMAL_DV_CLAIM_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal-d-v-claim-v1"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
PAET_FORMAL_DV_OUTPUT_PATH = (
    _REPO_ROOT
    / "runs/irstd1k_stage_a_seed42"
    / PAET_FORMAL_DV_RUN_ID
)
PAET_FORMAL_DV_STAGING_PATH = PAET_FORMAL_DV_OUTPUT_PATH.with_name(
    f".{PAET_FORMAL_DV_OUTPUT_PATH.name}.incomplete"
)
PAET_FORMAL_DV_COMPARISON_PROTOCOL_PATH = (
    _REPO_ROOT
    / "protocols/IRSTD-1K/paired_formal_evaluation_v1/config.json"
)
PAET_FORMAL_DV_MANIFEST_PATH = (
    _REPO_ROOT / "protocols/IRSTD-1K/stage_a_seed42/manifest.json"
)
PAET_FORMAL_DV_BASE_INDEX_PATH = (
    _REPO_ROOT
    / "runs/irstd1k_stage_a_seed42/reference_base_cache_fx_v2"
    / "D_V/index.json"
)
PAET_FORMAL_DV_COMPARISON_PROTOCOL_SHA256 = (
    "a322530eec57ffa6f8a34684a19e96b5881c06aa876a17956a2f8625283199cc"
)
PAET_FORMAL_DV_COMPARISON_PROTOCOL_FINGERPRINT = (
    "cb2fb09c3ec7dbbb0f057d94f7f159e2b4a733296e6ea4a144d6302387014884"
)
PAET_FORMAL_DV_MANIFEST_SHA256 = (
    "aa8e33529bd86f564ce6e163e0f9a7b1b3053e9c15054a59c6702a1523f35c02"
)
PAET_FORMAL_DV_MANIFEST_FINGERPRINT = (
    "87d63d1a6aa1414c06dc08cdb5547080a18cd54baf08e72cd5a77175758e1820"
)
PAET_FORMAL_DV_BASE_INDEX_SHA256 = (
    "86da975813b2b17afe5ddfc2477de72680f941e170b24678510000bbd23351c1"
)
PAET_FORMAL_DV_BASE_INDEX_FINGERPRINT = (
    "3431162d68fc79c50352adb828b3ff158b335f17f35a0e2a120251c63ec356d9"
)
PAET_FORMAL_DV_IMAGE_FINGERPRINT = (
    "57a6b4dd1ec44ce2c25c9c0c4ac5ae85ff9e0982f1894cb9f1963ae38dada68e"
)
PAET_FORMAL_DV_GT_FINGERPRINT = (
    "6407d397d0c0db7fa9e82f8b4b650d83efb0124642f961f7622f3d33074f0eda"
)
PAET_FORMAL_DV_BASE_FINGERPRINT = (
    "5f69986b95d11a89c5a5e91d6bdd63add865eda102be8ce486722fee8cd00dce"
)
PAET_FORMAL_DV_BASE_STATE_FINGERPRINT = (
    "1e17bc11465bf4fd63b5a697dd466cd2d78505d44ba83f9862ffbce3bd39f3c4"
)
PAET_FORMAL_DV_PREPROCESSING_FINGERPRINT = (
    "db4b3b4b37c513c3ad5547f1d05a756d9c422f8f55d846b19b50acfd59f210ca"
)
PAET_FORMAL_DV_PREPROCESSING = {
    "color_mode": "L",
    "height": 256,
    "image_interpolation": "bilinear",
    "mask_interpolation": "nearest",
    "mean": [0.5],
    "range": "float32-[0,1]-then-normalize",
    "std": [0.5],
    "width": 256,
}
PAET_FORMAL_DV_BATCH_SIZE = 8
PAET_FORMAL_DV_DEVICE = "cuda:0"
PAET_FORMAL_DV_CUDA_VISIBLE_DEVICES = "0"
PAET_FORMAL_DV_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_CLAIM_NAME = ".incomplete.json"
_RECEIPT_NAME = "receipt.json"
_DECISION_NAME = "decision.json"
_COMPLETE_NAME = "COMPLETE.json"
_FINAL_MEMBERS = frozenset(
    {_RECEIPT_NAME, _DECISION_NAME, _COMPLETE_NAME}
)
_HEX = frozenset("0123456789abcdef")


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fingerprinted(
    payload: Mapping[str, object],
    *,
    field: str,
) -> dict[str, object]:
    body = dict(payload)
    if field in body:
        raise ValueError(f"{field} already exists")
    return {**body, field: stable_fingerprint(body)}


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular file")

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _verify_fingerprint(
    payload: Mapping[str, object],
    *,
    field: str,
    name: str,
) -> str:
    body = dict(payload)
    if field not in body:
        raise ValueError(f"{name} lacks {field}")
    digest = _digest(body.pop(field), name=f"{name}.{field}")
    if stable_fingerprint(body) != digest:
        raise ValueError(f"{name} fingerprint changed")
    return digest


def _write_new(path: Path, payload: Mapping[str, object]) -> None:
    data = _json_bytes(payload)
    descriptor = os.open(
        path,
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verify_fixed_file(
    path: Path,
    expected_sha256: str,
    *,
    name: str,
) -> None:
    expected = _digest(expected_sha256, name=f"{name} SHA256")
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path
        or file_sha256(path) != expected
    ):
        raise RuntimeError(f"{name} differs from its fixed binding")


def _atomic_rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory rename is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            "fixed D_V final output appeared before publication"
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        f"{source} -> {target}",
    )


def _require_atomic_rename_noreplace() -> None:
    if getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None) is None:
        raise RuntimeError("atomic no-replace directory rename is unavailable")


def _fixed_plan_payload() -> dict[str, object]:
    return {
        "schema_version": "cure-lite-paet-bfa-v21-formal-d-v-plan-v1",
        "run_id": PAET_FORMAL_DV_RUN_ID,
        "method": PAET_FORMAL_METHOD,
        "dataset": "IRSTD-1K",
        "runtime_split": "D_V",
        "seed": PAET_FORMAL_SEED,
        "output_repo_path": (
            "runs/irstd1k_stage_a_seed42/"
            f"{PAET_FORMAL_DV_RUN_ID}"
        ),
        "staging_repo_path": (
            "runs/irstd1k_stage_a_seed42/"
            f".{PAET_FORMAL_DV_RUN_ID}.incomplete"
        ),
        "fixed_device": PAET_FORMAL_DV_DEVICE,
        "CUDA_VISIBLE_DEVICES": (
            PAET_FORMAL_DV_CUDA_VISIBLE_DEVICES
        ),
        "CUBLAS_WORKSPACE_CONFIG": (
            PAET_FORMAL_DV_CUBLAS_WORKSPACE_CONFIG
        ),
        "comparison_protocol": {
            "repo_path": (
                "protocols/IRSTD-1K/"
                "paired_formal_evaluation_v1/config.json"
            ),
            "file_sha256": (
                PAET_FORMAL_DV_COMPARISON_PROTOCOL_SHA256
            ),
            "comparison_protocol_fingerprint": (
                PAET_FORMAL_DV_COMPARISON_PROTOCOL_FINGERPRINT
            ),
        },
        "D_V_bundle": {
            "manifest_repo_path": (
                "protocols/IRSTD-1K/stage_a_seed42/manifest.json"
            ),
            "manifest_file_sha256": PAET_FORMAL_DV_MANIFEST_SHA256,
            "manifest_fingerprint": (
                PAET_FORMAL_DV_MANIFEST_FINGERPRINT
            ),
            "base_index_repo_path": (
                "runs/irstd1k_stage_a_seed42/"
                "reference_base_cache_fx_v2/D_V/index.json"
            ),
            "base_index_file_sha256": (
                PAET_FORMAL_DV_BASE_INDEX_SHA256
            ),
            "base_index_fingerprint": (
                PAET_FORMAL_DV_BASE_INDEX_FINGERPRINT
            ),
            "image_fingerprint": PAET_FORMAL_DV_IMAGE_FINGERPRINT,
            "GT_fingerprint": PAET_FORMAL_DV_GT_FINGERPRINT,
            "preprocessing": dict(PAET_FORMAL_DV_PREPROCESSING),
            "preprocessing_fingerprint": (
                PAET_FORMAL_DV_PREPROCESSING_FINGERPRINT
            ),
            "base_fingerprint": PAET_FORMAL_DV_BASE_FINGERPRINT,
            "base_state_fingerprint": (
                PAET_FORMAL_DV_BASE_STATE_FINGERPRINT
            ),
            "images": 120,
            "batch_size": PAET_FORMAL_DV_BATCH_SIZE,
        },
        "fixed_output": {
            "rule": PAET_FIXED_OUTPUT_RULE,
            "zero_tie_policy": PAET_ZERO_TIE_POLICY,
            "sigmoid_applied": False,
            "PAET_threshold_search_performed": False,
        },
        "Base@B": {
            "selection_policy": PAET_BASE_AT_B_SELECTION_POLICY,
            "candidate_count": len(PAET_FORMAL_BASE_THRESHOLD_GRID),
            "candidate_grid": list(PAET_FORMAL_BASE_THRESHOLD_GRID),
            "stage_a_config_sha256": (
                PAET_FORMAL_STAGE_A_CONFIG_SHA256
            ),
        },
        "execution_policy": {
            "formal_attempt_loader_fixed": True,
            "claim_before_D_V_materialization": True,
            "single_attempt": True,
            "resume_allowed": False,
            "retry_allowed": False,
            "overwrite_allowed": False,
            "model_training_performed": False,
            "model_state_update_performed": False,
            "D_T_accessed": False,
            "final_members": sorted(_FINAL_MEMBERS),
            "complete_written_last": True,
            "atomic_no_replace_publication": True,
        },
    }


def validate_paet_formal_d_v_create_only() -> dict[str, object]:
    """Validate the fixed plan without opening D_V or creating output."""

    _require_atomic_rename_noreplace()
    if PAET_FORMAL_DV_OUTPUT_PATH.exists():
        raise FileExistsError("fixed PAET Formal D_V output already exists")
    if PAET_FORMAL_DV_STAGING_PATH.exists():
        raise FileExistsError(
            "fixed PAET Formal D_V staging attempt already exists"
        )
    plan = _fixed_plan_payload()
    return {
        **plan,
        "plan_fingerprint": stable_fingerprint(plan),
        "create_only": True,
        "Formal800_attempt_loaded": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "output_created": False,
    }


def _claim_before_d_v(
    attempt: LoadedCoverageStatePAETFormalAttempt,
) -> dict[str, object]:
    if type(attempt) is not LoadedCoverageStatePAETFormalAttempt:
        raise TypeError(
            "attempt must come from the fixed Formal800 attempt loader"
        )
    attempt.verify_unchanged()
    if PAET_FORMAL_DV_OUTPUT_PATH.exists():
        raise FileExistsError("fixed PAET Formal D_V output already exists")
    PAET_FORMAL_DV_STAGING_PATH.mkdir(
        parents=False,
        exist_ok=False,
    )
    plan = _fixed_plan_payload()
    claim = _fingerprinted(
        {
            "schema_version": PAET_FORMAL_DV_CLAIM_SCHEMA,
            "run_id": PAET_FORMAL_DV_RUN_ID,
            "status": "claimed_before_D_V_materialization",
            "plan_fingerprint": stable_fingerprint(plan),
            "formal_attempt_complete_fingerprint": (
                attempt.complete_fingerprint
            ),
            "source_closure_content_fingerprint": (
                attempt.source_closure_content_fingerprint
            ),
            "source_closure_manifest_sha256": (
                attempt.source_closure_manifest_sha256
            ),
            "source_closure_archive_sha256": (
                attempt.source_closure_archive_sha256
            ),
            "single_attempt": True,
            "resume_allowed": False,
            "retry_allowed": False,
            "output_reusable": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
        field="claim_fingerprint",
    )
    _write_new(PAET_FORMAL_DV_STAGING_PATH / _CLAIM_NAME, claim)
    _fsync_directory(PAET_FORMAL_DV_STAGING_PATH)
    _fsync_directory(PAET_FORMAL_DV_STAGING_PATH.parent)
    return claim


def _reverify_formal_attempt(
    expected: LoadedCoverageStatePAETFormalAttempt,
) -> None:
    """Reload the fixed attempt so live source-closure bytes are rechecked."""

    if type(expected) is not LoadedCoverageStatePAETFormalAttempt:
        raise TypeError(
            "expected must be the fixed loaded Formal800 attempt"
        )
    expected.verify_unchanged()
    reloaded = load_coverage_state_paet_formal_attempt()
    if type(reloaded) is not LoadedCoverageStatePAETFormalAttempt:
        raise TypeError("Formal800 reloader returned a substituted object")
    reloaded.verify_unchanged()
    if (
        reloaded.complete_fingerprint != expected.complete_fingerprint
        or reloaded.formal_training_result_fingerprint
        != expected.formal_training_result_fingerprint
        or reloaded.authorization_fingerprint
        != expected.authorization_fingerprint
        or reloaded.structural_result_fingerprint
        != expected.structural_result_fingerprint
        or reloaded.source_receipt_fingerprint
        != expected.source_receipt_fingerprint
        or reloaded.source_closure_manifest_sha256
        != expected.source_closure_manifest_sha256
        or reloaded.source_closure_archive_sha256
        != expected.source_closure_archive_sha256
        or reloaded.source_closure_content_fingerprint
        != expected.source_closure_content_fingerprint
        or reloaded.source_closure_file_count
        != expected.source_closure_file_count
        or reloaded.artifact.artifact_fingerprint
        != expected.artifact.artifact_fingerprint
        or reloaded.artifact.module_state_fingerprint
        != expected.artifact.module_state_fingerprint
        or reloaded.artifact.receipt_sha256
        != expected.artifact.receipt_sha256
    ):
        raise RuntimeError(
            "Formal800 attempt or source closure changed during D_V"
        )


def _require_fixed_runtime() -> None:
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES")
        != PAET_FORMAL_DV_CUDA_VISIBLE_DEVICES
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        != PAET_FORMAL_DV_CUBLAS_WORKSPACE_CONFIG
    ):
        raise RuntimeError(
            "fixed D_V evaluation requires GPU0-only deterministic environment"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "fixed D_V evaluation requires exactly one visible CUDA device"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def _load_fixed_d_v_inputs() -> tuple[
    FrozenComparisonProtocol,
    LoadedDVCacheBundle,
]:
    """Open the fixed D_V sources; caller must already own the claim."""

    if (
        not PAET_FORMAL_DV_STAGING_PATH.is_dir()
        or PAET_FORMAL_DV_STAGING_PATH.is_symlink()
        or {
            path.name for path in PAET_FORMAL_DV_STAGING_PATH.iterdir()
        }
        != {_CLAIM_NAME}
    ):
        raise RuntimeError("fixed D_V claim is absent or changed")
    _verify_fixed_file(
        PAET_FORMAL_DV_COMPARISON_PROTOCOL_PATH,
        PAET_FORMAL_DV_COMPARISON_PROTOCOL_SHA256,
        name="comparison protocol",
    )
    _verify_fixed_file(
        PAET_FORMAL_DV_MANIFEST_PATH,
        PAET_FORMAL_DV_MANIFEST_SHA256,
        name="D_V manifest",
    )
    _verify_fixed_file(
        PAET_FORMAL_DV_BASE_INDEX_PATH,
        PAET_FORMAL_DV_BASE_INDEX_SHA256,
        name="D_V base index",
    )
    protocol = load_frozen_comparison_protocol(
        PAET_FORMAL_DV_COMPARISON_PROTOCOL_PATH
    )
    if (
        type(protocol) is not FrozenComparisonProtocol
        or protocol.comparison_protocol_fingerprint
        != PAET_FORMAL_DV_COMPARISON_PROTOCOL_FINGERPRINT
        or protocol.manifest_fingerprint
        != PAET_FORMAL_DV_MANIFEST_FINGERPRINT
        or protocol.manifest_file_sha256
        != PAET_FORMAL_DV_MANIFEST_SHA256
        or protocol.preprocessing_fingerprint
        != PAET_FORMAL_DV_PREPROCESSING_FINGERPRINT
        or protocol.base_fingerprint
        != PAET_FORMAL_DV_BASE_FINGERPRINT
    ):
        raise RuntimeError("fixed D_V comparison protocol changed")
    manifest = load_and_validate_manifest(PAET_FORMAL_DV_MANIFEST_PATH)
    preprocessing = PreprocessConfig.from_fingerprint_payload(
        PAET_FORMAL_DV_PREPROCESSING
    )
    if (
        stable_fingerprint(preprocessing.fingerprint_payload())
        != PAET_FORMAL_DV_PREPROCESSING_FINGERPRINT
    ):
        raise RuntimeError("fixed D_V preprocessing changed")
    dataset = ManifestImageDataset(
        manifest,
        "D_V",
        preprocessing,
        manifest_path=PAET_FORMAL_DV_MANIFEST_PATH,
    )
    bundle = load_d_v_cache_bundle(
        PAET_FORMAL_DV_BASE_INDEX_PATH,
        dataset,
        expected_base_fingerprint=PAET_FORMAL_DV_BASE_FINGERPRINT,
    )
    if (
        type(bundle) is not LoadedDVCacheBundle
        or bundle.split != "D_V"
        or len(bundle.rows) != 120
        or bundle.base_index_sha256
        != PAET_FORMAL_DV_BASE_INDEX_SHA256
        or bundle.base_index_fingerprint
        != PAET_FORMAL_DV_BASE_INDEX_FINGERPRINT
        or bundle.d_v_image_fingerprint
        != PAET_FORMAL_DV_IMAGE_FINGERPRINT
        or bundle.d_v_gt_fingerprint != PAET_FORMAL_DV_GT_FINGERPRINT
        or bundle.base_fingerprint
        != PAET_FORMAL_DV_BASE_FINGERPRINT
        or bundle.base_state_fingerprint
        != PAET_FORMAL_DV_BASE_STATE_FINGERPRINT
        or bundle.split_manifest_fingerprint
        != PAET_FORMAL_DV_MANIFEST_FINGERPRINT
        or bundle.split_manifest_file_sha256
        != PAET_FORMAL_DV_MANIFEST_SHA256
        or bundle.preprocessing_fingerprint
        != PAET_FORMAL_DV_PREPROCESSING_FINGERPRINT
    ):
        raise RuntimeError("fixed D_V bundle identity changed")
    protocol.verify_bundle(bundle)
    return protocol, bundle


@dataclass(frozen=True)
class _EvaluationEvidence:
    artifact_binding_payload: dict[str, object]
    artifact_binding_fingerprint: str
    sample_payload: dict[str, object]
    sample_fingerprint: str
    evaluation_payload: dict[str, object]
    evaluation_result_fingerprint: str
    decision: dict[str, object]

    def verify(self) -> None:
        for name, value in (
            (
                "artifact_binding_fingerprint",
                self.artifact_binding_fingerprint,
            ),
            ("sample_fingerprint", self.sample_fingerprint),
            (
                "evaluation_result_fingerprint",
                self.evaluation_result_fingerprint,
            ),
        ):
            _digest(value, name=name)
        if (
            stable_fingerprint(self.artifact_binding_payload)
            != self.artifact_binding_fingerprint
            or stable_fingerprint(self.sample_payload)
            != self.sample_fingerprint
            or stable_fingerprint(self.evaluation_payload)
            != self.evaluation_result_fingerprint
        ):
            raise RuntimeError(
                "fixed D_V evidence payload fingerprint changed"
            )
        decision_fingerprint = _verify_fingerprint(
            self.decision,
            field="decision_fingerprint",
            name="D_V decision",
        )
        if (
            self.decision.get("D_T_accessed") is not False
            or self.decision.get("seed") != PAET_FORMAL_SEED
            or self.decision.get("runtime_split") != "D_V"
            or self.decision.get("bindings", {}).get(
                "evaluation_result_fingerprint"
            )
            != self.evaluation_result_fingerprint
            or self.decision.get("bindings", {}).get(
                "artifact_binding_fingerprint"
            )
            != self.artifact_binding_fingerprint
            or decision_fingerprint
            != self.decision["decision_fingerprint"]
        ):
            raise RuntimeError("fixed D_V decision binding changed")


def _execute_fixed_evaluation(
    attempt: LoadedCoverageStatePAETFormalAttempt,
) -> _EvaluationEvidence:
    """Materialize D_V once and evaluate the fixed Formal800 artifact."""

    _require_fixed_runtime()
    attempt.verify_unchanged()
    protocol, bundle = _load_fixed_d_v_inputs()
    model = attempt.artifact.model
    model.to(device=torch.device(PAET_FORMAL_DV_DEVICE))
    binding = bind_paet_formal_artifact(attempt, protocol, bundle)
    samples = build_paet_fixed_d_v_samples(
        binding,
        batch_size=PAET_FORMAL_DV_BATCH_SIZE,
    )
    result = evaluate_paet_formal_d_v(samples, binding)
    decision = assess_paet_formal_d_v_result(result)
    result.verify_unchanged()
    samples.verify_unchanged()
    binding.verify_unchanged()
    bundle.verify_unchanged()
    attempt.verify_unchanged()
    evidence = _EvaluationEvidence(
        artifact_binding_payload=binding.canonical_payload(),
        artifact_binding_fingerprint=binding.binding_fingerprint,
        sample_payload=samples.canonical_payload(),
        sample_fingerprint=samples.adapter_fingerprint,
        evaluation_payload=result.canonical_payload(),
        evaluation_result_fingerprint=result.result_fingerprint,
        decision=dict(decision),
    )
    evidence.verify()
    return evidence


def _build_run_receipt(
    attempt: LoadedCoverageStatePAETFormalAttempt,
    claim: Mapping[str, object],
    evidence: _EvaluationEvidence,
) -> dict[str, object]:
    attempt.verify_unchanged()
    evidence.verify()
    claim_fingerprint = _verify_fingerprint(
        claim,
        field="claim_fingerprint",
        name="D_V claim",
    )
    core = {
        "schema_version": PAET_FORMAL_DV_RUN_RECEIPT_SCHEMA,
        "run_id": PAET_FORMAL_DV_RUN_ID,
        "status": "complete",
        "plan": _fixed_plan_payload(),
        "plan_fingerprint": stable_fingerprint(_fixed_plan_payload()),
        "claim_fingerprint": claim_fingerprint,
        "formal_attempt": {
            "complete_fingerprint": attempt.complete_fingerprint,
            "formal_training_result_fingerprint": (
                attempt.formal_training_result_fingerprint
            ),
            "authorization_fingerprint": (
                attempt.authorization_fingerprint
            ),
            "structural_result_fingerprint": (
                attempt.structural_result_fingerprint
            ),
            "structural_source_receipt_fingerprint": (
                attempt.source_receipt_fingerprint
            ),
            "source_closure_manifest_sha256": (
                attempt.source_closure_manifest_sha256
            ),
            "source_closure_archive_sha256": (
                attempt.source_closure_archive_sha256
            ),
            "source_closure_content_fingerprint": (
                attempt.source_closure_content_fingerprint
            ),
            "source_closure_file_count": (
                attempt.source_closure_file_count
            ),
            "post_formal_structural_retention_passed": (
                attempt.post_formal_structural_retention_passed
            ),
        },
        "artifact_binding": evidence.artifact_binding_payload,
        "artifact_binding_fingerprint": (
            evidence.artifact_binding_fingerprint
        ),
        "fixed_D_V_samples": evidence.sample_payload,
        "fixed_D_V_samples_fingerprint": evidence.sample_fingerprint,
        "evaluation_result": evidence.evaluation_payload,
        "evaluation_result_fingerprint": (
            evidence.evaluation_result_fingerprint
        ),
        "decision_fingerprint": evidence.decision[
            "decision_fingerprint"
        ],
        "model_training_performed": False,
        "model_state_update_performed": False,
        "PAET_threshold_search_performed": False,
        "Base@B_51_point_selection_performed": True,
        "runtime_splits": ["D_V"],
        "D_T_accessed": False,
        "authorizes_full_CURE": False,
        "authorizes_cross_backbone": False,
    }
    return _fingerprinted(core, field="receipt_fingerprint")


def _validate_published_output(root: Path) -> dict[str, object]:
    if (
        root.is_symlink()
        or not root.is_dir()
        or root.resolve(strict=True) != root
    ):
        raise RuntimeError("published D_V output is not canonical")
    members = {path.name: path for path in root.iterdir()}
    if (
        set(members) != _FINAL_MEMBERS
        or any(
            path.is_symlink() or not path.is_file()
            for path in members.values()
        )
    ):
        raise RuntimeError("published D_V output inventory changed")
    receipt = _strict_json(members[_RECEIPT_NAME], name="D_V receipt")
    decision = _strict_json(
        members[_DECISION_NAME],
        name="D_V decision",
    )
    complete = _strict_json(
        members[_COMPLETE_NAME],
        name="D_V COMPLETE",
    )
    receipt_fingerprint = _verify_fingerprint(
        receipt,
        field="receipt_fingerprint",
        name="D_V receipt",
    )
    decision_fingerprint = _verify_fingerprint(
        decision,
        field="decision_fingerprint",
        name="D_V decision",
    )
    complete_fingerprint = _verify_fingerprint(
        complete,
        field="complete_fingerprint",
        name="D_V COMPLETE",
    )
    if (
        receipt.get("schema_version")
        != PAET_FORMAL_DV_RUN_RECEIPT_SCHEMA
        or complete.get("schema_version")
        != PAET_FORMAL_DV_COMPLETE_SCHEMA
        or receipt.get("run_id") != PAET_FORMAL_DV_RUN_ID
        or complete.get("run_id") != PAET_FORMAL_DV_RUN_ID
        or complete.get("status") != "complete"
        or complete.get("receipt_fingerprint")
        != receipt_fingerprint
        or complete.get("decision_fingerprint")
        != decision_fingerprint
        or receipt.get("decision_fingerprint")
        != decision_fingerprint
        or complete.get("formal_attempt_complete_fingerprint")
        != receipt.get("formal_attempt", {}).get(
            "complete_fingerprint"
        )
        or complete.get("source_closure_content_fingerprint")
        != receipt.get("formal_attempt", {}).get(
            "source_closure_content_fingerprint"
        )
        or complete.get("source_closure_manifest_sha256")
        != receipt.get("formal_attempt", {}).get(
            "source_closure_manifest_sha256"
        )
        or complete.get("source_closure_archive_sha256")
        != receipt.get("formal_attempt", {}).get(
            "source_closure_archive_sha256"
        )
        or complete.get("source_closure_file_count")
        != receipt.get("formal_attempt", {}).get(
            "source_closure_file_count"
        )
        or complete.get("receipt_sha256")
        != file_sha256(members[_RECEIPT_NAME])
        or complete.get("decision_sha256")
        != file_sha256(members[_DECISION_NAME])
        or complete.get("final_members")
        != sorted(_FINAL_MEMBERS)
        or receipt.get("D_T_accessed") is not False
        or decision.get("D_T_accessed") is not False
        or complete.get("D_T_accessed") is not False
        or receipt.get("model_training_performed") is not False
        or receipt.get("model_state_update_performed") is not False
        or receipt.get("PAET_threshold_search_performed") is not False
    ):
        raise RuntimeError("published D_V output bindings changed")
    return {
        "run_id": PAET_FORMAL_DV_RUN_ID,
        "status": "complete",
        "receipt_fingerprint": receipt_fingerprint,
        "decision_fingerprint": decision_fingerprint,
        "evaluation_result_fingerprint": (
            receipt["evaluation_result_fingerprint"]
        ),
        "gate_passed": decision["gate_passed"],
        "authorizes_D_T": decision["authorizes_D_T"],
        "D_T_accessed": False,
        "complete_fingerprint": complete_fingerprint,
    }


def run_paet_formal_d_v_once() -> dict[str, object]:
    """Run and publish the sole fixed seed-42 PAET Formal800 D_V result."""

    _require_atomic_rename_noreplace()
    if PAET_FORMAL_DV_OUTPUT_PATH.exists():
        raise FileExistsError("fixed PAET Formal D_V output already exists")
    if PAET_FORMAL_DV_STAGING_PATH.exists():
        raise FileExistsError(
            "fixed PAET Formal D_V attempt already exists and is not reusable"
        )
    # This loader authenticates only Formal800/D_R/source-closure state.
    # D_V is not opened until the staging claim below exists.
    attempt = load_coverage_state_paet_formal_attempt()
    attempt.verify_unchanged()
    claim = _claim_before_d_v(attempt)
    evidence = _execute_fixed_evaluation(attempt)
    _reverify_formal_attempt(attempt)
    receipt = _build_run_receipt(attempt, claim, evidence)
    decision = dict(evidence.decision)
    _write_new(
        PAET_FORMAL_DV_STAGING_PATH / _RECEIPT_NAME,
        receipt,
    )
    _write_new(
        PAET_FORMAL_DV_STAGING_PATH / _DECISION_NAME,
        decision,
    )
    complete_core = {
        "schema_version": PAET_FORMAL_DV_COMPLETE_SCHEMA,
        "run_id": PAET_FORMAL_DV_RUN_ID,
        "status": "complete",
        "final_members": sorted(_FINAL_MEMBERS),
        "receipt_sha256": file_sha256(
            PAET_FORMAL_DV_STAGING_PATH / _RECEIPT_NAME
        ),
        "decision_sha256": file_sha256(
            PAET_FORMAL_DV_STAGING_PATH / _DECISION_NAME
        ),
        "receipt_fingerprint": receipt["receipt_fingerprint"],
        "decision_fingerprint": decision["decision_fingerprint"],
        "evaluation_result_fingerprint": (
            evidence.evaluation_result_fingerprint
        ),
        "formal_attempt_complete_fingerprint": (
            attempt.complete_fingerprint
        ),
        "source_closure_content_fingerprint": (
            attempt.source_closure_content_fingerprint
        ),
        "source_closure_manifest_sha256": (
            attempt.source_closure_manifest_sha256
        ),
        "source_closure_archive_sha256": (
            attempt.source_closure_archive_sha256
        ),
        "source_closure_file_count": (
            attempt.source_closure_file_count
        ),
        "artifact_binding_fingerprint": (
            evidence.artifact_binding_fingerprint
        ),
        "fixed_D_V_samples_fingerprint": (
            evidence.sample_fingerprint
        ),
        "gate_passed": decision["gate_passed"],
        "authorizes_D_T": decision["authorizes_D_T"],
        "D_T_accessed": False,
        "model_training_performed": False,
        "authorizes_full_CURE": False,
        "authorizes_cross_backbone": False,
    }
    complete = _fingerprinted(
        complete_core,
        field="complete_fingerprint",
    )
    _write_new(
        PAET_FORMAL_DV_STAGING_PATH / _COMPLETE_NAME,
        complete,
    )
    _reverify_formal_attempt(attempt)
    evidence.verify()
    _fsync_directory(PAET_FORMAL_DV_STAGING_PATH)
    (PAET_FORMAL_DV_STAGING_PATH / _CLAIM_NAME).unlink()
    _fsync_directory(PAET_FORMAL_DV_STAGING_PATH)
    _atomic_rename_noreplace(
        PAET_FORMAL_DV_STAGING_PATH,
        PAET_FORMAL_DV_OUTPUT_PATH,
    )
    _fsync_directory(PAET_FORMAL_DV_OUTPUT_PATH.parent)
    return _validate_published_output(PAET_FORMAL_DV_OUTPUT_PATH)


__all__ = [
    "PAET_FORMAL_DV_OUTPUT_PATH",
    "PAET_FORMAL_DV_RUN_ID",
    "PAET_FORMAL_DV_STAGING_PATH",
    "run_paet_formal_d_v_once",
    "validate_paet_formal_d_v_create_only",
]
