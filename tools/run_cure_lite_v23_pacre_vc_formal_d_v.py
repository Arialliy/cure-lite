#!/usr/bin/env python3
"""Validate or execute the sole PACRE-VC v23 adaptive D_V evaluation.

The command has no caller-selectable model, split, threshold, seed, output,
or batch size.  ``--validate-create-only`` authenticates the completed
Formal800 terminal and frozen metadata without opening a D_V tensor.
``--run-once`` first claims the fixed staging directory, then opens the
strict v21 D_V bundle and evaluates the one final v23 model.  It never opens
D_T and exposes no training, calibration, checkpoint-selection, or retry
path.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import errno
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

# Fix the physical mapping before importing torch through any project module.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.data import ManifestImageDataset, PreprocessConfig  # noqa: E402
from cure_lite.experiment.cache_pipeline import (  # noqa: E402
    LoadedDVCacheBundle,
    load_d_v_cache_bundle,
)
from cure_lite.experiment.paired_formal_evaluation import (  # noqa: E402
    FrozenComparisonProtocol,
    load_frozen_comparison_protocol,
)
from cure_lite.splits import load_and_validate_manifest  # noqa: E402
from cure_lite_v23.formal_artifacts import (  # noqa: E402
    LoadedPACREVCFormalArtifact,
    VerifiedPACREVCFormalTerminal,
)
from cure_lite_v23.formal_evaluation import (  # noqa: E402
    PACRE_VC_BASE_AT_B_SELECTION_POLICY,
    PACRE_VC_FIXED_OUTPUT_RULE,
    PACRE_VC_FORMAL_BATCH_SIZE,
    PACRE_VC_FORMAL_BASE_THRESHOLD,
    PACRE_VC_FORMAL_BASE_THRESHOLD_GRID,
    PACRE_VC_FORMAL_METHOD,
    PACRE_VC_FORMAL_STAGE_A_CONFIG_SHA256,
    PACRE_VC_MAXIMUM_FP_COMPONENTS_PER_MP,
    PACRE_VC_MAXIMUM_PIXEL_FA,
    PACRE_VC_MAXIMUM_RAW_BACKGROUND_FA,
    PACRE_VC_ZERO_TIE_POLICY,
    PACREVCFormalDVEvaluationResult,
    PACREVCFormalModelBinding,
    bind_pacre_vc_formal_model,
    evaluate_pacre_vc_formal_d_v,
)
from cure_lite_v23.formal_training import (  # noqa: E402
    PACRE_VC_FORMAL_RUN_ID,
)
from cure_lite_v23.authorization import protocol_root  # noqa: E402
from cure_lite_v23.environment import (  # noqa: E402
    stabilize_pacre_vc_numerical_runtime,
    verify_runtime_environment,
)
from cure_lite_v23.protocol import (  # noqa: E402
    fingerprinted,
    read_strict_json,
    verify_fingerprinted,
    write_new_json,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "cure_lite_pacre_v23_vc_formal_d_v_seed42_r1"
OUTPUT_REPO_PATH = f"runs/irstd1k_stage_a_seed42/{RUN_ID}"
OUTPUT_PATH = ROOT / OUTPUT_REPO_PATH
STAGING_REPO_PATH = (
    f"runs/irstd1k_stage_a_seed42/.{RUN_ID}.incomplete"
)
STAGING_PATH = ROOT / STAGING_REPO_PATH

FORMAL_OUTPUT_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/"
    f"{PACRE_VC_FORMAL_RUN_ID}"
)
FORMAL_OUTPUT_PATH = ROOT / FORMAL_OUTPUT_REPO_PATH
COMPARISON_PROTOCOL_REPO_PATH = (
    "protocols/IRSTD-1K/paired_formal_evaluation_v1/config.json"
)
MANIFEST_REPO_PATH = (
    "protocols/IRSTD-1K/stage_a_seed42/manifest.json"
)
BASE_INDEX_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/"
    "reference_base_cache_fx_v2/D_V/index.json"
)
STAGE_A_CONFIG_REPO_PATH = (
    "protocols/IRSTD-1K/stage_a_seed42_fx_v3/stage_a_config.json"
)
COMPARISON_PROTOCOL_PATH = ROOT / COMPARISON_PROTOCOL_REPO_PATH
MANIFEST_PATH = ROOT / MANIFEST_REPO_PATH
BASE_INDEX_PATH = ROOT / BASE_INDEX_REPO_PATH
STAGE_A_CONFIG_PATH = ROOT / STAGE_A_CONFIG_REPO_PATH
RUNTIME_LOCK_PATH = protocol_root() / "runtime_environment_lock.json"

COMPARISON_PROTOCOL_SHA256 = (
    "a322530eec57ffa6f8a34684a19e96b5881c06aa876a17956a2f8625283199cc"
)
COMPARISON_PROTOCOL_FINGERPRINT = (
    "cb2fb09c3ec7dbbb0f057d94f7f159e2b4a733296e6ea4a144d6302387014884"
)
MANIFEST_SHA256 = (
    "aa8e33529bd86f564ce6e163e0f9a7b1b3053e9c15054a59c6702a1523f35c02"
)
MANIFEST_FINGERPRINT = (
    "87d63d1a6aa1414c06dc08cdb5547080a18cd54baf08e72cd5a77175758e1820"
)
BASE_INDEX_SHA256 = (
    "86da975813b2b17afe5ddfc2477de72680f941e170b24678510000bbd23351c1"
)
BASE_INDEX_FINGERPRINT = (
    "3431162d68fc79c50352adb828b3ff158b335f17f35a0e2a120251c63ec356d9"
)
D_V_IMAGE_FINGERPRINT = (
    "57a6b4dd1ec44ce2c25c9c0c4ac5ae85ff9e0982f1894cb9f1963ae38dada68e"
)
D_V_GT_FINGERPRINT = (
    "6407d397d0c0db7fa9e82f8b4b650d83efb0124642f961f7622f3d33074f0eda"
)
BASE_FINGERPRINT = (
    "5f69986b95d11a89c5a5e91d6bdd63add865eda102be8ce486722fee8cd00dce"
)
BASE_STATE_FINGERPRINT = (
    "1e17bc11465bf4fd63b5a697dd466cd2d78505d44ba83f9862ffbce3bd39f3c4"
)
PREPROCESSING_FINGERPRINT = (
    "db4b3b4b37c513c3ad5547f1d05a756d9c422f8f55d846b19b50acfd59f210ca"
)
PREPROCESSING = {
    "color_mode": "L",
    "height": 256,
    "image_interpolation": "bilinear",
    "mask_interpolation": "nearest",
    "mean": [0.5],
    "range": "float32-[0,1]-then-normalize",
    "std": [0.5],
    "width": 256,
}

DEVICE = "cuda:0"
BATCH_SIZE = PACRE_VC_FORMAL_BATCH_SIZE
SEED = 42
EPOCHS = 800
STEPS_PER_EPOCH = 40
UPDATES = 32_000

PLAN_SCHEMA = "cure-lite-v23-pacre-vc-formal-d-v-plan-v1"
CLAIM_SCHEMA = "cure-lite-v23-pacre-vc-formal-d-v-claim-v1"
RECEIPT_SCHEMA = "cure-lite-v23-pacre-vc-formal-d-v-receipt-v1"
DECISION_SCHEMA = "cure-lite-v23-pacre-vc-formal-d-v-decision-v1"
COMPLETE_SCHEMA = "cure-lite-v23-pacre-vc-formal-d-v-complete-v1"
FAILURE_SCHEMA = "cure-lite-v23-pacre-vc-formal-d-v-failure-v1"
VALIDATION_SCHEMA = (
    "cure-lite-v23-pacre-vc-formal-d-v-create-only-validation-v1"
)

CLAIM_FILE = "claim.json"
RECEIPT_FILE = "receipt.json"
DECISION_FILE = "decision.json"
COMPLETE_FILE = "COMPLETE.json"
FAILURE_FILE = "FAILURE.json"
FINAL_MEMBERS = frozenset(
    {CLAIM_FILE, RECEIPT_FILE, DECISION_FILE, COMPLETE_FILE}
)


@dataclass(frozen=True)
class _FormalTerminal:
    """Strict local binding to the independently verified Formal800 output."""

    verified_terminal: VerifiedPACREVCFormalTerminal
    payload: dict[str, object]

    @property
    def artifact(self) -> LoadedPACREVCFormalArtifact:
        return self.verified_terminal.artifact

    @property
    def binding_fingerprint(self) -> str:
        return stable_fingerprint(self.payload)

    def verify_unchanged(self) -> None:
        self.verified_terminal.verify_unchanged()
        current = _load_formal_terminal()
        if (
            current.payload != self.payload
            or current.artifact.artifact_json
            != self.artifact.artifact_json
        ):
            raise RuntimeError("Formal800 terminal changed during D_V")


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_digest(value: object, *, name: str) -> str:
    if not _is_digest(value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return str(value)


def _verify_fixed_file(
    path: Path,
    expected_sha256: str,
    *,
    name: str,
) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
        or file_sha256(path) != expected_sha256
    ):
        raise RuntimeError(f"frozen {name} changed")


def _verify_fixed_metadata() -> None:
    """Verify only metadata files; this does not materialize D_V tensors."""

    for path, digest, name in (
        (
            COMPARISON_PROTOCOL_PATH,
            COMPARISON_PROTOCOL_SHA256,
            "comparison protocol",
        ),
        (MANIFEST_PATH, MANIFEST_SHA256, "split manifest"),
        (BASE_INDEX_PATH, BASE_INDEX_SHA256, "D_V base index"),
        (
            STAGE_A_CONFIG_PATH,
            PACRE_VC_FORMAL_STAGE_A_CONFIG_SHA256,
            "stage-a threshold metadata",
        ),
    ):
        _verify_fixed_file(path, digest, name=name)


def _load_formal_terminal() -> _FormalTerminal:
    """Independently verify and strictly load the one Formal800 final model."""

    from tools.verify_cure_lite_v23_pacre_vc_formal_800_receipt import (
        verify_terminal_sealed as verify_formal_terminal,
    )

    verified_terminal = verify_formal_terminal(FORMAL_OUTPUT_PATH)
    if (
        type(verified_terminal)
        is not VerifiedPACREVCFormalTerminal
    ):
        raise TypeError("Formal800 verifier returned a substituted result")
    verified_terminal.verify_unchanged()

    complete = read_strict_json(FORMAL_OUTPUT_PATH / COMPLETE_FILE)
    complete_fingerprint = verify_fingerprinted(
        complete,
        field="complete_fingerprint",
    )
    attempt = read_strict_json(FORMAL_OUTPUT_PATH / "attempt.json")
    attempt_fingerprint = verify_fingerprinted(attempt)
    authorization = read_strict_json(
        FORMAL_OUTPUT_PATH / "receipts/authorization.json"
    )
    verify_fingerprinted(authorization)
    training = read_strict_json(
        FORMAL_OUTPUT_PATH / "receipts/training.json"
    )
    verify_fingerprinted(training)
    decision = read_strict_json(
        FORMAL_OUTPUT_PATH / "receipts/decision.json"
    )
    verify_fingerprinted(decision)
    artifact_payload = read_strict_json(
        FORMAL_OUTPUT_PATH / "final_model/artifact.json"
    )
    artifact_fingerprint = verify_fingerprinted(
        artifact_payload,
        field="artifact_fingerprint",
    )
    artifact = verified_terminal.artifact
    if artifact.receipt != artifact_payload:
        raise RuntimeError(
            "Formal800 verifier-issued artifact differs from disk receipt"
        )

    authorization_core = authorization.get("authorization")
    compute = training.get("compute_ledger")
    if not isinstance(authorization_core, Mapping):
        raise RuntimeError("Formal800 authorization receipt is incomplete")
    if not isinstance(compute, Mapping):
        raise RuntimeError("Formal800 compute ledger is incomplete")
    authorization_fingerprint = _require_digest(
        authorization.get("authorization_fingerprint"),
        name="Formal800 authorization fingerprint",
    )
    formal_result_fingerprint = _require_digest(
        training.get("formal_result_fingerprint"),
        name="Formal800 result fingerprint",
    )
    training_result_fingerprint = _require_digest(
        training.get("training_result_fingerprint"),
        name="Formal800 training-result fingerprint",
    )
    source_closure_fingerprint = _require_digest(
        artifact_payload.get("source_closure_fingerprint"),
        name="Formal800 source-closure fingerprint",
    )
    if (
        complete.get("schema_version")
        != "cure-lite-v23-pacre-vc-formal800-complete-v1"
        or complete.get("run_id") != PACRE_VC_FORMAL_RUN_ID
        or complete.get("status")
        != "FORMAL800_TRAINING_COMPLETE_D_V_PREREGISTRATION_ELIGIBLE"
        or complete.get("attempt_fingerprint") != attempt_fingerprint
        or complete.get("authorization_fingerprint")
        != authorization_fingerprint
        or complete.get("formal_result_fingerprint")
        != formal_result_fingerprint
        or complete.get("final_model_artifact_fingerprint")
        != artifact_fingerprint
        or complete.get("seed") != SEED
        or complete.get("epochs") != EPOCHS
        or complete.get("steps_per_epoch") != STEPS_PER_EPOCH
        or complete.get("updates") != UPDATES
        or complete.get("training_invocations") != 1
        or complete.get("final_checkpoint_only") is not True
        or complete.get("optimizer_state_saved") is not False
        or complete.get("intermediate_checkpoint_saved") is not False
        or complete.get("D_V_preregistration_eligible") is not True
        or complete.get("D_V_execution_authorized") is not False
        or complete.get("D_T_execution_authorized") is not False
        or complete.get("D_V_tensor_payload_accessed") is not False
        or complete.get("D_T_tensor_payload_accessed") is not False
        or complete.get("performance_evaluation_performed") is not False
        or compute.get("seed") != SEED
        or compute.get("epochs") != EPOCHS
        or compute.get("steps_per_epoch") != STEPS_PER_EPOCH
        or compute.get("completed_updates") != UPDATES
        or compute.get("forward_calls") != UPDATES
        or compute.get("backward_calls") != UPDATES
        or compute.get("optimizer_steps") != UPDATES
        or compute.get("epoch_progress_rows") != EPOCHS
        or compute.get("training_invocations") != 1
        or artifact_payload.get("training_result_fingerprint")
        != training_result_fingerprint
        or artifact_payload.get("formal_result_fingerprint")
        != formal_result_fingerprint
        or artifact_payload.get("authorization_fingerprint")
        != authorization_fingerprint
        or authorization_core.get("source_closure_fingerprint")
        != source_closure_fingerprint
        or artifact_payload.get("D_V_payload_accessed") is not False
        or artifact_payload.get("D_T_payload_accessed") is not False
        or artifact_payload.get("performance_evaluation_performed")
        is not False
        or decision.get("D_V_preregistration_eligible") is not True
        or decision.get("D_V_execution_authorized") is not False
        or decision.get("D_T_execution_authorized") is not False
        or decision.get("D_V_tensor_payload_accessed") is not False
        or decision.get("D_T_tensor_payload_accessed") is not False
    ):
        raise PermissionError(
            "D_V requires the exact final-only seed42 Formal800 terminal"
        )
    payload = {
        "formal_run_id": PACRE_VC_FORMAL_RUN_ID,
        "formal_output_repo_path": FORMAL_OUTPUT_REPO_PATH,
        "complete_fingerprint": complete_fingerprint,
        "attempt_fingerprint": attempt_fingerprint,
        "authorization_fingerprint": authorization_fingerprint,
        "formal_result_fingerprint": formal_result_fingerprint,
        "training_result_fingerprint": training_result_fingerprint,
        "artifact_fingerprint": artifact_fingerprint,
        "model_file_sha256": artifact_payload["model_file_sha256"],
        "module_state_fingerprint": artifact_payload[
            "module_state_fingerprint"
        ],
        "source_closure_fingerprint": source_closure_fingerprint,
        "seed": SEED,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "updates": UPDATES,
        "trained_from_scratch": True,
        "final_checkpoint_only": True,
        "D_V_payload_accessed_during_training": False,
        "D_T_payload_accessed_during_training": False,
        "formal_terminal_independently_verified": True,
    }
    for name in (
        "model_file_sha256",
        "module_state_fingerprint",
    ):
        _require_digest(payload[name], name=name)
    return _FormalTerminal(
        verified_terminal=verified_terminal,
        payload=payload,
    )


def _fixed_plan() -> dict[str, object]:
    return {
        "schema_version": PLAN_SCHEMA,
        "run_id": RUN_ID,
        "method": PACRE_VC_FORMAL_METHOD,
        "dataset": "IRSTD-1K",
        "runtime_split": "D_V",
        "D_V_adaptive": True,
        "seed": SEED,
        "output_repo_path": OUTPUT_REPO_PATH,
        "staging_repo_path": STAGING_REPO_PATH,
        "formal_output_repo_path": FORMAL_OUTPUT_REPO_PATH,
        "device": DEVICE,
        "CUDA_VISIBLE_DEVICES": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "runtime_environment_lock_repo_path": str(
            RUNTIME_LOCK_PATH.relative_to(ROOT)
        ),
        "batch_size": BATCH_SIZE,
        "comparison_protocol": {
            "repo_path": COMPARISON_PROTOCOL_REPO_PATH,
            "file_sha256": COMPARISON_PROTOCOL_SHA256,
            "comparison_protocol_fingerprint": (
                COMPARISON_PROTOCOL_FINGERPRINT
            ),
        },
        "D_V_bundle": {
            "manifest_repo_path": MANIFEST_REPO_PATH,
            "manifest_file_sha256": MANIFEST_SHA256,
            "manifest_fingerprint": MANIFEST_FINGERPRINT,
            "base_index_repo_path": BASE_INDEX_REPO_PATH,
            "base_index_file_sha256": BASE_INDEX_SHA256,
            "base_index_fingerprint": BASE_INDEX_FINGERPRINT,
            "image_fingerprint": D_V_IMAGE_FINGERPRINT,
            "GT_fingerprint": D_V_GT_FINGERPRINT,
            "preprocessing": dict(PREPROCESSING),
            "preprocessing_fingerprint": PREPROCESSING_FINGERPRINT,
            "base_fingerprint": BASE_FINGERPRINT,
            "base_state_fingerprint": BASE_STATE_FINGERPRINT,
            "images": 120,
        },
        "fixed_output": {
            "rule": PACRE_VC_FIXED_OUTPUT_RULE,
            "base_threshold": PACRE_VC_FORMAL_BASE_THRESHOLD,
            "field_threshold": 0.0,
            "zero_tie_policy": PACRE_VC_ZERO_TIE_POLICY,
            "hard_union": True,
            "sigmoid_applied": False,
            "PACRE_threshold_search_performed": False,
        },
        "Base@B": {
            "selection_policy": PACRE_VC_BASE_AT_B_SELECTION_POLICY,
            "candidate_count": len(PACRE_VC_FORMAL_BASE_THRESHOLD_GRID),
            "candidate_grid": list(PACRE_VC_FORMAL_BASE_THRESHOLD_GRID),
            "stage_a_config_repo_path": STAGE_A_CONFIG_REPO_PATH,
            "stage_a_config_sha256": (
                PACRE_VC_FORMAL_STAGE_A_CONFIG_SHA256
            ),
        },
        "development_gate": {
            "comparison": "best_valid_Base",
            "strict_target_improvement": True,
            "strict_recovered_miss_improvement": True,
            "minimum_fixed_uplift_margin": None,
            "plus_one_is_sufficient": True,
            "mIoU_non_regression": True,
            "nIoU_non_regression": True,
            "retention": 1.0,
            "maximum_pixel_Fa": PACRE_VC_MAXIMUM_PIXEL_FA,
            "maximum_raw_background_Fa": (
                PACRE_VC_MAXIMUM_RAW_BACKGROUND_FA
            ),
            "maximum_false_positive_components_per_megapixel": (
                PACRE_VC_MAXIMUM_FP_COMPONENTS_PER_MP
            ),
            "budget_violation": False,
        },
        "execution_policy": {
            "claim_before_D_V_materialization": True,
            "single_attempt": True,
            "resume_allowed": False,
            "retry_allowed": False,
            "overwrite_allowed": False,
            "training_performed": False,
            "model_update_performed": False,
            "checkpoint_selection_performed": False,
            "D_T_accessed": False,
            "complete_written_last": True,
            "atomic_no_replace_publication": True,
            "final_members": sorted(FINAL_MEMBERS),
        },
    }


def _require_atomic_rename_noreplace() -> None:
    if getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None) is None:
        raise RuntimeError("atomic no-replace directory rename is unavailable")


def _atomic_rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory rename is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result != 0:
        code = ctypes.get_errno()
        if code in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(f"fixed D_V output exists: {target}")
        raise OSError(code, os.strerror(code), str(target))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_outputs_absent() -> None:
    if OUTPUT_PATH.exists() or OUTPUT_PATH.is_symlink():
        raise FileExistsError("fixed PACRE-VC D_V output already exists")
    if STAGING_PATH.exists() or STAGING_PATH.is_symlink():
        raise FileExistsError(
            "fixed PACRE-VC D_V attempt already exists and is not reusable"
        )


def validate_create_only() -> dict[str, object]:
    """Validate fixed prerequisites without opening D_V or creating output."""

    _require_atomic_rename_noreplace()
    _ensure_outputs_absent()
    _verify_fixed_metadata()
    runtime = _require_runtime()
    formal = _load_formal_terminal()
    formal.artifact.verify_unchanged()
    plan = _fixed_plan()
    return fingerprinted(
        {
            "schema_version": VALIDATION_SCHEMA,
            "run_id": RUN_ID,
            "mode": "validate_create_only",
            "plan": plan,
            "plan_fingerprint": stable_fingerprint(plan),
            "formal_terminal": formal.payload,
            "formal_terminal_binding_fingerprint": (
                formal.binding_fingerprint
            ),
            "runtime": runtime,
            "output_absent": True,
            "staging_absent": True,
            "output_claimed": False,
            "D_V_metadata_verified": True,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
            "evaluation_performed": False,
            "not_a_performance_result": True,
        }
    )


def _claim(
    formal: _FormalTerminal,
    runtime: Mapping[str, object],
) -> dict[str, object]:
    _ensure_outputs_absent()
    STAGING_PATH.mkdir(parents=False, exist_ok=False)
    plan = _fixed_plan()
    claim = fingerprinted(
        {
            "schema_version": CLAIM_SCHEMA,
            "run_id": RUN_ID,
            "status": "claimed_before_D_V_materialization",
            "plan_fingerprint": stable_fingerprint(plan),
            "formal_terminal_binding_fingerprint": (
                formal.binding_fingerprint
            ),
            "formal_complete_fingerprint": formal.payload[
                "complete_fingerprint"
            ],
            "formal_artifact_fingerprint": formal.payload[
                "artifact_fingerprint"
            ],
            "source_closure_fingerprint": formal.payload[
                "source_closure_fingerprint"
            ],
            "runtime": dict(runtime),
            "single_attempt": True,
            "resume_allowed": False,
            "retry_allowed": False,
            "overwrite_allowed": False,
            "output_reusable": False,
            "D_V_metadata_verified": True,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        },
        field="claim_fingerprint",
    )
    write_new_json(STAGING_PATH / CLAIM_FILE, claim)
    _fsync_directory(STAGING_PATH)
    _fsync_directory(STAGING_PATH.parent)
    return claim


def _require_runtime() -> dict[str, object]:
    stabilize_pacre_vc_numerical_runtime()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise RuntimeError("D_V fixes GPU0 and deterministic CUBLAS")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "fixed D_V evaluation requires exactly one visible CUDA device"
        )
    locked = read_strict_json(RUNTIME_LOCK_PATH)
    fingerprint = verify_runtime_environment(locked, DEVICE)
    return {
        "device": DEVICE,
        "CUDA_VISIBLE_DEVICES": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "runtime_environment_lock_repo_path": str(
            RUNTIME_LOCK_PATH.relative_to(ROOT)
        ),
        "runtime_environment_fingerprint": fingerprint,
        "exactly_one_visible_cuda_device": True,
        "stabilized_before_lock_verification": True,
    }


def _load_fixed_d_v_inputs() -> tuple[
    FrozenComparisonProtocol,
    LoadedDVCacheBundle,
]:
    """Load the strict D_V payload only after the exclusive claim exists."""

    if (
        not STAGING_PATH.is_dir()
        or STAGING_PATH.is_symlink()
        or {path.name for path in STAGING_PATH.iterdir()}
        != {CLAIM_FILE}
    ):
        raise RuntimeError("D_V output claim is absent or changed")
    _verify_fixed_metadata()
    protocol = load_frozen_comparison_protocol(COMPARISON_PROTOCOL_PATH)
    manifest = load_and_validate_manifest(MANIFEST_PATH)
    preprocessing = PreprocessConfig.from_fingerprint_payload(PREPROCESSING)
    if (
        stable_fingerprint(preprocessing.fingerprint_payload())
        != PREPROCESSING_FINGERPRINT
    ):
        raise RuntimeError("frozen D_V preprocessing changed")
    dataset = ManifestImageDataset(
        manifest,
        "D_V",
        preprocessing,
        manifest_path=MANIFEST_PATH,
    )
    bundle = load_d_v_cache_bundle(
        BASE_INDEX_PATH,
        dataset,
        expected_base_fingerprint=BASE_FINGERPRINT,
    )
    if (
        type(protocol) is not FrozenComparisonProtocol
        or protocol.comparison_protocol_fingerprint
        != COMPARISON_PROTOCOL_FINGERPRINT
        or protocol.manifest_fingerprint != MANIFEST_FINGERPRINT
        or protocol.manifest_file_sha256 != MANIFEST_SHA256
        or protocol.preprocessing_fingerprint
        != PREPROCESSING_FINGERPRINT
        or protocol.base_fingerprint != BASE_FINGERPRINT
        or type(bundle) is not LoadedDVCacheBundle
        or bundle.split != "D_V"
        or len(bundle.rows) != 120
        or bundle.base_index_sha256 != BASE_INDEX_SHA256
        or bundle.base_index_fingerprint != BASE_INDEX_FINGERPRINT
        or bundle.d_v_image_fingerprint != D_V_IMAGE_FINGERPRINT
        or bundle.d_v_gt_fingerprint != D_V_GT_FINGERPRINT
        or bundle.base_fingerprint != BASE_FINGERPRINT
        or bundle.base_state_fingerprint != BASE_STATE_FINGERPRINT
        or bundle.split_manifest_fingerprint != MANIFEST_FINGERPRINT
        or bundle.split_manifest_file_sha256 != MANIFEST_SHA256
        or bundle.preprocessing_fingerprint
        != PREPROCESSING_FINGERPRINT
    ):
        raise RuntimeError("strict D_V protocol/bundle identity changed")
    protocol.verify_bundle(bundle)
    return protocol, bundle


def _result_contract(
    result: PACREVCFormalDVEvaluationResult,
) -> dict[str, object]:
    if type(result) is not PACREVCFormalDVEvaluationResult:
        raise TypeError("D_V evaluator returned a substituted result")
    result.verify_unchanged()
    payload = result.canonical_payload()
    gate = payload.get("development_gate")
    output = payload.get("output_contract")
    base_selection = payload.get("Base@B_selection")
    if (
        not isinstance(gate, Mapping)
        or not isinstance(output, Mapping)
        or not isinstance(base_selection, Mapping)
    ):
        raise RuntimeError("D_V result lacks its frozen contracts")
    candidate_ledger = base_selection.get("candidate_ledger")
    candidate_entries = (
        candidate_ledger.get("entries")
        if isinstance(candidate_ledger, Mapping)
        else None
    )
    if (
        payload.get("method") != PACRE_VC_FORMAL_METHOD
        or payload.get("seed") != SEED
        or payload.get("runtime_split") != "D_V"
        or payload.get("batch_size") != BATCH_SIZE
        or payload.get("D_V_adaptive") is not True
        or payload.get("D_V_payload_accessed") is not True
        or payload.get("D_T_payload_accessed") is not False
        or output.get("rule") != PACRE_VC_FIXED_OUTPUT_RULE
        or output.get("field_threshold") != 0.0
        or output.get("zero_tie_policy") != PACRE_VC_ZERO_TIE_POLICY
        or output.get("hard_union") is not True
        or output.get("sigmoid_applied") is not False
        or output.get("PACRE_threshold_search_performed") is not False
        or base_selection.get("candidate_threshold_grid")
        != list(PACRE_VC_FORMAL_BASE_THRESHOLD_GRID)
        or base_selection.get("candidate_count")
        != len(PACRE_VC_FORMAL_BASE_THRESHOLD_GRID)
        or not isinstance(candidate_ledger, Mapping)
        or candidate_ledger.get("schema_version")
        != "cure-lite-v23-pacre-vc-base-at-b-51-ledger-v1"
        or candidate_ledger.get("candidate_count")
        != len(PACRE_VC_FORMAL_BASE_THRESHOLD_GRID)
        or not isinstance(candidate_entries, list)
        or len(candidate_entries)
        != len(PACRE_VC_FORMAL_BASE_THRESHOLD_GRID)
        or gate.get("comparison") != "best_valid_Base"
        or gate.get("gate_passed") is not result.gate_passed
        or payload.get("eligible_for_D_T_confirmation")
        is not result.gate_passed
        or payload.get("authorizes_D_T") is not False
        or payload.get("final_model_success_established") is not False
    ):
        raise RuntimeError("D_V evaluator result contract changed")
    verify_fingerprinted(
        candidate_ledger,
        field="ledger_fingerprint",
    )
    return payload


def _decision(
    result: PACREVCFormalDVEvaluationResult,
    result_payload: Mapping[str, object],
    *,
    formal: _FormalTerminal,
    binding: PACREVCFormalModelBinding,
) -> dict[str, object]:
    gate = result_payload["development_gate"]
    if not isinstance(gate, Mapping):
        raise TypeError("D_V development gate must be a mapping")
    status = (
        "PACRE_V23_FORMAL_D_V_ADAPTIVE_PASS"
        if result.gate_passed
        else "PACRE_V23_FORMAL_D_V_ADAPTIVE_FAIL"
    )
    return fingerprinted(
        {
            "schema_version": DECISION_SCHEMA,
            "run_id": RUN_ID,
            "status": status,
            "D_V_adaptive": True,
            "gate_passed": result.gate_passed,
            "failed_checks": list(result.failed_checks),
            "checks": dict(result.checks),
            "comparison": "best_valid_Base",
            "best_valid_Base": gate["best_valid_Base"],
            "CURE_margins": gate["CURE_margins"],
            "minimum_fixed_uplift_margin": None,
            "plus_one_is_sufficient": True,
            "evaluation_result_fingerprint": result.result_fingerprint,
            "model_binding_fingerprint": binding.binding_fingerprint,
            "formal_complete_fingerprint": formal.payload[
                "complete_fingerprint"
            ],
            "formal_artifact_fingerprint": formal.payload[
                "artifact_fingerprint"
            ],
            "eligible_for_future_separately_preregistered_D_T": (
                result.gate_passed
            ),
            "authorizes_D_T": False,
            "D_T_payload_accessed": False,
            "model_training_performed": False,
            "model_state_update_performed": False,
            "checkpoint_selection_performed": False,
            "final_model_success_established": False,
        },
        field="decision_fingerprint",
    )


def _receipt(
    *,
    formal: _FormalTerminal,
    claim: Mapping[str, object],
    binding: PACREVCFormalModelBinding,
    result: PACREVCFormalDVEvaluationResult,
    result_payload: Mapping[str, object],
    decision: Mapping[str, object],
) -> dict[str, object]:
    claim_fingerprint = verify_fingerprinted(
        claim,
        field="claim_fingerprint",
    )
    decision_fingerprint = verify_fingerprinted(
        decision,
        field="decision_fingerprint",
    )
    return fingerprinted(
        {
            "schema_version": RECEIPT_SCHEMA,
            "run_id": RUN_ID,
            "status": "complete",
            "plan": _fixed_plan(),
            "plan_fingerprint": stable_fingerprint(_fixed_plan()),
            "claim_fingerprint": claim_fingerprint,
            "formal_terminal": formal.payload,
            "formal_terminal_binding_fingerprint": (
                formal.binding_fingerprint
            ),
            "model_binding": binding.canonical_payload(),
            "model_binding_fingerprint": binding.binding_fingerprint,
            "evaluation_result": dict(result_payload),
            "evaluation_result_fingerprint": result.result_fingerprint,
            "decision_fingerprint": decision_fingerprint,
            "runtime_splits": ["D_V"],
            "D_V_adaptive": True,
            "D_V_payload_accessed": True,
            "D_T_payload_accessed": False,
            "model_training_performed": False,
            "model_state_update_performed": False,
            "checkpoint_selection_performed": False,
            "PACRE_threshold_search_performed": False,
            "Base@B_51_point_selection_performed": True,
            "minimum_fixed_uplift_margin": None,
            "authorizes_D_T": False,
            "final_model_success_established": False,
        }
    )


def _file_hashes(root: Path, names: Sequence[str]) -> dict[str, str]:
    return {name: file_sha256(root / name) for name in sorted(names)}


def _failure(
    error: BaseException,
    *,
    stage: str,
    d_v_payload_accessed: bool,
) -> None:
    if (
        not STAGING_PATH.is_dir()
        or (STAGING_PATH / FAILURE_FILE).exists()
        or (STAGING_PATH / COMPLETE_FILE).exists()
    ):
        return
    existing = sorted(
        path.name
        for path in STAGING_PATH.iterdir()
        if path.is_file() and not path.is_symlink()
    )
    payload = fingerprinted(
        {
            "schema_version": FAILURE_SCHEMA,
            "run_id": RUN_ID,
            "status": "failed_incomplete_single_attempt",
            "failed_stage": stage,
            "exception_type": type(error).__name__,
            "message": str(error),
            "existing_members": existing,
            "output_reusable": False,
            "resume_allowed": False,
            "retry_allowed": False,
            "D_V_adaptive": True,
            "D_V_payload_accessed": d_v_payload_accessed,
            "D_T_payload_accessed": False,
            "authorizes_D_T": False,
            "final_model_success_established": False,
        },
        field="failure_fingerprint",
    )
    write_new_json(STAGING_PATH / FAILURE_FILE, payload)
    _fsync_directory(STAGING_PATH)


def run_once() -> dict[str, object]:
    """Run and publish the one fixed adaptive D_V attempt."""

    _require_atomic_rename_noreplace()
    _ensure_outputs_absent()
    _verify_fixed_metadata()
    runtime = _require_runtime()
    formal = _load_formal_terminal()
    formal.artifact.verify_unchanged()
    claim = _claim(formal, runtime)
    stage = "post_claim_runtime"
    d_v_payload_accessed = False
    try:
        post_claim_runtime = _require_runtime()
        if post_claim_runtime != runtime:
            raise RuntimeError("D_V runtime changed across output claim")
        stage = "D_V_materialization"
        # From this point a failing strict loader may already have opened a
        # subset of D_V assets, so failure evidence reports access
        # conservatively.
        d_v_payload_accessed = True
        protocol, bundle = _load_fixed_d_v_inputs()
        stage = "fixed_model_evaluation"
        model = formal.artifact.model
        model.to(device=torch.device(DEVICE), dtype=torch.float32)
        binding = bind_pacre_vc_formal_model(
            formal.verified_terminal,
            protocol,
            bundle,
        )
        result = evaluate_pacre_vc_formal_d_v(
            model,
            bundle,
            protocol,
            binding,
            batch_size=BATCH_SIZE,
        )
        result_payload = _result_contract(result)
        result.verify_unchanged()
        binding.verify_unchanged()
        bundle.verify_unchanged()

        stage = "post_evaluation_formal_reverification"
        formal.verify_unchanged()
        decision = _decision(
            result,
            result_payload,
            formal=formal,
            binding=binding,
        )
        receipt = _receipt(
            formal=formal,
            claim=claim,
            binding=binding,
            result=result,
            result_payload=result_payload,
            decision=decision,
        )
        write_new_json(STAGING_PATH / RECEIPT_FILE, receipt)
        write_new_json(STAGING_PATH / DECISION_FILE, decision)

        stage = "terminal_seal"
        scientific_hashes = _file_hashes(
            STAGING_PATH,
            (CLAIM_FILE, RECEIPT_FILE, DECISION_FILE),
        )
        complete = fingerprinted(
            {
                "schema_version": COMPLETE_SCHEMA,
                "run_id": RUN_ID,
                "status": (
                    "PACRE_V23_FORMAL_D_V_ADAPTIVE_PASS"
                    if result.gate_passed
                    else "PACRE_V23_FORMAL_D_V_ADAPTIVE_FAIL"
                ),
                "final_members": sorted(FINAL_MEMBERS),
                "artifact_files": scientific_hashes,
                "artifact_count": len(scientific_hashes),
                "claim_fingerprint": claim["claim_fingerprint"],
                "receipt_fingerprint": receipt[
                    "receipt_fingerprint"
                ],
                "decision_fingerprint": decision[
                    "decision_fingerprint"
                ],
                "evaluation_result_fingerprint": (
                    result.result_fingerprint
                ),
                "model_binding_fingerprint": (
                    binding.binding_fingerprint
                ),
                "formal_complete_fingerprint": formal.payload[
                    "complete_fingerprint"
                ],
                "formal_artifact_fingerprint": formal.payload[
                    "artifact_fingerprint"
                ],
                "source_closure_fingerprint": formal.payload[
                    "source_closure_fingerprint"
                ],
                "D_V_adaptive": True,
                "gate_passed": result.gate_passed,
                "eligible_for_future_separately_preregistered_D_T": (
                    result.gate_passed
                ),
                "authorizes_D_T": False,
                "D_T_payload_accessed": False,
                "model_training_performed": False,
                "model_state_update_performed": False,
                "checkpoint_selection_performed": False,
                "final_model_success_established": False,
            },
            field="complete_fingerprint",
        )
        write_new_json(STAGING_PATH / COMPLETE_FILE, complete)
        _fsync_directory(STAGING_PATH)
        if {path.name for path in STAGING_PATH.iterdir()} != FINAL_MEMBERS:
            raise RuntimeError("D_V terminal population changed")
        formal.verify_unchanged()
        result.verify_unchanged()
        _atomic_rename_noreplace(STAGING_PATH, OUTPUT_PATH)
        _fsync_directory(OUTPUT_PATH.parent)
        from tools.verify_cure_lite_v23_pacre_vc_formal_d_v_receipt import (
            verify_terminal,
        )

        return verify_terminal(OUTPUT_PATH)
    except BaseException as error:
        _failure(
            error,
            stage=stage,
            d_v_payload_accessed=d_v_payload_accessed,
        )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-create-only", action="store_true")
    modes.add_argument("--run-once", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = (
        validate_create_only()
        if args.validate_create_only
        else run_once()
    )
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
