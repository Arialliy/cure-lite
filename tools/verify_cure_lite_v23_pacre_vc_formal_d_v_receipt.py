#!/usr/bin/env python3
"""Independently verify the PACRE-VC v23 adaptive D_V terminal graph.

The verifier does not rerun inference and does not open D_V or D_T tensor
payloads.  It strictly reloads the final Formal800 artifact, rechecks its live
source/runtime closure, authenticates the frozen D_V metadata, and recomputes
the relative-performance decision from the persisted aggregate metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from math import isclose, isfinite
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.calibration import FalseAlarmBudget  # noqa: E402
from cure_lite.calibration_ledger import (  # noqa: E402
    CalibrationCandidateLedger,
    CandidateEvaluation,
)
from cure_lite.experiment.paired_formal_evaluation import (  # noqa: E402
    FORMAL_DV_ANCHOR_COVERED,
    FORMAL_DV_ANCHOR_MISSES,
    FORMAL_DV_IMAGES,
    FORMAL_DV_TOTAL_TARGETS,
    FrozenComparisonProtocol,
    load_frozen_comparison_protocol,
)
from cure_lite.metrics import AggregateEvaluation  # noqa: E402
from cure_lite_v23.authorization import protocol_root  # noqa: E402
from cure_lite_v23.environment import (  # noqa: E402
    stabilize_pacre_vc_numerical_runtime,
    verify_runtime_environment,
)
from cure_lite_v23.formal_artifacts import (  # noqa: E402
    LoadedPACREVCFormalArtifact,
    load_pacre_vc_formal_final_model,
)
from cure_lite_v23.formal_evaluation import (  # noqa: E402
    PACRE_VC_BASE_AT_B_SELECTION_POLICY,
    PACRE_VC_FIXED_OUTPUT_RULE,
    PACRE_VC_FORMAL_BATCH_SIZE,
    PACRE_VC_FORMAL_BASE_THRESHOLD,
    PACRE_VC_FORMAL_BASE_THRESHOLD_GRID,
    PACRE_VC_FORMAL_DV_RESULT_SCHEMA,
    PACRE_VC_FORMAL_METHOD,
    PACRE_VC_FORMAL_MODEL_BINDING_SCHEMA,
    PACRE_VC_FORMAL_STAGE_A_CONFIG_SHA256,
    PACRE_VC_MAXIMUM_FP_COMPONENTS_PER_MP,
    PACRE_VC_MAXIMUM_PIXEL_FA,
    PACRE_VC_MAXIMUM_RAW_BACKGROUND_FA,
    PACRE_VC_ZERO_TIE_POLICY,
)
from cure_lite_v23.formal_training import (  # noqa: E402
    PACRE_VC_FORMAL_RUN_ID,
)
from cure_lite_v23.protocol import (  # noqa: E402
    read_strict_json,
    verify_fingerprinted,
    verify_source_closure,
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
SOURCE_CLOSURE_PATH = protocol_root() / "implementation_closure.json"

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

CLAIM_FILE = "claim.json"
RECEIPT_FILE = "receipt.json"
DECISION_FILE = "decision.json"
COMPLETE_FILE = "COMPLETE.json"
FINAL_MEMBERS = frozenset(
    {CLAIM_FILE, RECEIPT_FILE, DECISION_FILE, COMPLETE_FILE}
)
_AGGREGATE_FIELDS = tuple(field.name for field in fields(AggregateEvaluation))
_HEX = frozenset("0123456789abcdef")


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


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


def _verify_metadata_and_protocol() -> FrozenComparisonProtocol:
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
    protocol = load_frozen_comparison_protocol(COMPARISON_PROTOCOL_PATH)
    if (
        type(protocol) is not FrozenComparisonProtocol
        or protocol.comparison_protocol_fingerprint
        != COMPARISON_PROTOCOL_FINGERPRINT
        or protocol.manifest_fingerprint != MANIFEST_FINGERPRINT
        or protocol.manifest_file_sha256 != MANIFEST_SHA256
        or protocol.preprocessing_fingerprint
        != PREPROCESSING_FINGERPRINT
        or protocol.base_fingerprint != BASE_FINGERPRINT
        or protocol.d_v_base_index_fingerprint
        != BASE_INDEX_FINGERPRINT
        or protocol.d_v_base_index_sha256 != BASE_INDEX_SHA256
        or protocol.d_v_image_fingerprint != D_V_IMAGE_FINGERPRINT
        or protocol.d_v_gt_fingerprint != D_V_GT_FINGERPRINT
        or tuple(protocol.residual_thresholds)
        != PACRE_VC_FORMAL_BASE_THRESHOLD_GRID
    ):
        raise RuntimeError("frozen D_V protocol identity changed")
    return protocol


def _verify_runtime() -> dict[str, object]:
    stabilize_pacre_vc_numerical_runtime()
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != "0"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError(
            "D_V terminal verification requires fixed visible GPU0"
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


def _load_formal_terminal() -> tuple[
    LoadedPACREVCFormalArtifact,
    dict[str, object],
]:
    from tools.verify_cure_lite_v23_pacre_vc_formal_800_receipt import (
        verify_terminal as verify_formal_terminal,
    )

    verified = verify_formal_terminal(FORMAL_OUTPUT_PATH)
    if not isinstance(verified, Mapping):
        raise TypeError("Formal800 verifier returned a substituted result")
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
    artifact = load_pacre_vc_formal_final_model(
        FORMAL_OUTPUT_PATH / "final_model",
        artifact_payload,
    )
    authorization_core = authorization.get("authorization")
    compute = training.get("compute_ledger")
    if not isinstance(authorization_core, Mapping):
        raise RuntimeError("Formal800 authorization receipt is incomplete")
    if not isinstance(compute, Mapping):
        raise RuntimeError("Formal800 compute ledger is incomplete")
    authorization_fingerprint = _digest(
        authorization.get("authorization_fingerprint"),
        name="Formal800 authorization fingerprint",
    )
    formal_result_fingerprint = _digest(
        training.get("formal_result_fingerprint"),
        name="Formal800 result fingerprint",
    )
    training_result_fingerprint = _digest(
        training.get("training_result_fingerprint"),
        name="Formal800 training-result fingerprint",
    )
    source_closure_fingerprint = _digest(
        artifact_payload.get("source_closure_fingerprint"),
        name="Formal800 source-closure fingerprint",
    )
    live_source_fingerprint = verify_source_closure(
        read_strict_json(SOURCE_CLOSURE_PATH)
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
        or source_closure_fingerprint != live_source_fingerprint
        or artifact_payload.get("D_V_payload_accessed") is not False
        or artifact_payload.get("D_T_payload_accessed") is not False
        or decision.get("D_V_preregistration_eligible") is not True
        or decision.get("D_V_execution_authorized") is not False
        or decision.get("D_T_execution_authorized") is not False
    ):
        raise PermissionError("Formal800 terminal/source closure changed")
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
    for name in ("model_file_sha256", "module_state_fingerprint"):
        _digest(payload[name], name=name)
    return artifact, payload


def _expected_plan() -> dict[str, object]:
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


def _real(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise TypeError(f"{name} must be finite real")
    return float(value)


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _parse_metrics(
    value: object,
    *,
    name: str,
) -> AggregateEvaluation:
    if not isinstance(value, Mapping) or set(value) != set(_AGGREGATE_FIELDS):
        raise ValueError(f"{name} aggregate fields changed")
    payload = dict(value)
    for field_name in (
        "pd",
        "rmr",
        "gross_rmr",
        "net_rmr",
        "retention",
        "reachable_rmr",
        "oracle_upper_bound",
        "overlap_supported_rmr",
        "pixel_fa",
        "raw_background_fa",
        "fp_components_per_mp",
        "miou",
        "niou",
    ):
        payload[field_name] = _real(
            payload[field_name],
            name=f"{name}.{field_name}",
        )
    for field_name in (
        "images",
        "recovered_anchor_misses",
        "net_recovered_anchor_misses",
        "total_anchor_misses",
        "retained_anchor_covered",
        "total_anchor_covered",
        "recovered_reachable_anchor_misses",
        "total_reachable_anchor_misses",
    ):
        payload[field_name] = _integer(
            payload[field_name],
            name=f"{name}.{field_name}",
        )
    if type(payload["budget_violation"]) is not bool:
        raise TypeError(f"{name}.budget_violation must be bool")
    metrics = AggregateEvaluation(**payload)
    if (
        metrics.images != FORMAL_DV_IMAGES
        or metrics.total_anchor_misses != FORMAL_DV_ANCHOR_MISSES
        or metrics.total_anchor_covered != FORMAL_DV_ANCHOR_COVERED
        or metrics.total_anchor_misses + metrics.total_anchor_covered
        != FORMAL_DV_TOTAL_TARGETS
        or not 0
        <= metrics.recovered_anchor_misses
        <= metrics.total_anchor_misses
        or not 0
        <= metrics.retained_anchor_covered
        <= metrics.total_anchor_covered
        or not 0
        <= metrics.recovered_reachable_anchor_misses
        <= metrics.total_reachable_anchor_misses
        or not 0
        <= metrics.total_reachable_anchor_misses
        <= metrics.total_anchor_misses
    ):
        raise ValueError(f"{name} population/counts changed")
    true_targets = (
        metrics.retained_anchor_covered
        + metrics.recovered_anchor_misses
    )
    derived = {
        "pd": true_targets / FORMAL_DV_TOTAL_TARGETS,
        "rmr": (
            metrics.recovered_anchor_misses
            / FORMAL_DV_ANCHOR_MISSES
        ),
        "gross_rmr": (
            metrics.recovered_anchor_misses
            / FORMAL_DV_ANCHOR_MISSES
        ),
        "net_rmr": (
            metrics.net_recovered_anchor_misses
            / FORMAL_DV_ANCHOR_MISSES
        ),
        "retention": (
            metrics.retained_anchor_covered
            / FORMAL_DV_ANCHOR_COVERED
        ),
        "reachable_rmr": (
            metrics.recovered_reachable_anchor_misses
            / metrics.total_reachable_anchor_misses
            if metrics.total_reachable_anchor_misses
            else 0.0
        ),
        "oracle_upper_bound": (
            metrics.total_reachable_anchor_misses
            / FORMAL_DV_ANCHOR_MISSES
        ),
    }
    if any(
        not isclose(
            getattr(metrics, field_name),
            expected,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for field_name, expected in derived.items()
    ):
        raise ValueError(f"{name} derived metrics are inconsistent")
    if (
        metrics.net_recovered_anchor_misses
        != true_targets - FORMAL_DV_ANCHOR_COVERED
        or any(
            not 0.0 <= value <= 1.0
            for value in (
                metrics.pd,
                metrics.rmr,
                metrics.gross_rmr,
                metrics.retention,
                metrics.reachable_rmr,
                metrics.oracle_upper_bound,
                metrics.overlap_supported_rmr,
                metrics.miou,
                metrics.niou,
            )
        )
        or any(
            value < 0.0
            for value in (
                metrics.pixel_fa,
                metrics.raw_background_fa,
                metrics.fp_components_per_mp,
            )
        )
    ):
        raise ValueError(f"{name} metric ranges changed")
    return metrics


def _summary(metrics: AggregateEvaluation) -> dict[str, object]:
    return {
        "true_targets": (
            metrics.retained_anchor_covered
            + metrics.recovered_anchor_misses
        ),
        "Pd": metrics.pd,
        "mIoU": metrics.miou,
        "nIoU": metrics.niou,
        "pixel_Fa": metrics.pixel_fa,
        "raw_background_Fa": metrics.raw_background_fa,
        "false_positive_components_per_megapixel": (
            metrics.fp_components_per_mp
        ),
        "recovered_anchor_misses": metrics.recovered_anchor_misses,
        "retained_anchor_covered": metrics.retained_anchor_covered,
        "total_anchor_misses": metrics.total_anchor_misses,
        "total_anchor_covered": metrics.total_anchor_covered,
        "retention": metrics.retention,
        "budget_violation": metrics.budget_violation,
    }


def _budget_accepts(metrics: AggregateEvaluation) -> bool:
    return (
        metrics.pixel_fa <= PACRE_VC_MAXIMUM_PIXEL_FA
        and metrics.raw_background_fa
        <= PACRE_VC_MAXIMUM_RAW_BACKGROUND_FA
        and metrics.fp_components_per_mp
        <= PACRE_VC_MAXIMUM_FP_COMPONENTS_PER_MP
        and metrics.retention >= 0.99
    )


def _recompute_gate(
    result: Mapping[str, object],
) -> dict[str, object]:
    expected_top = {
        "schema_version",
        "method",
        "seed",
        "batch_size",
        "runtime_split",
        "D_V_adaptive",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "output_contract",
        "Base@B_selection",
        "operating_points",
        "development_gate",
        "bindings",
        "eligible_for_D_T_confirmation",
        "authorizes_D_T",
        "final_model_success_established",
    }
    if set(result) != expected_top:
        raise ValueError("D_V evaluation-result fields changed")
    output = result["output_contract"]
    base_selection = result["Base@B_selection"]
    operating = result["operating_points"]
    stored_gate = result["development_gate"]
    if (
        result["schema_version"] != PACRE_VC_FORMAL_DV_RESULT_SCHEMA
        or result["method"] != PACRE_VC_FORMAL_METHOD
        or result["seed"] != SEED
        or result["batch_size"] != BATCH_SIZE
        or result["runtime_split"] != "D_V"
        or result["D_V_adaptive"] is not True
        or result["D_V_payload_accessed"] is not True
        or result["D_T_payload_accessed"] is not False
        or result["authorizes_D_T"] is not False
        or result["final_model_success_established"] is not False
        or not isinstance(output, Mapping)
        or set(output)
        != {
            "rule",
            "field_threshold",
            "zero_tie_policy",
            "hard_union",
            "sigmoid_applied",
            "PACRE_threshold_search_performed",
            "exact_zero_field_pixels",
            "negative_field_pixels",
            "completion_pixels",
        }
        or output.get("rule") != PACRE_VC_FIXED_OUTPUT_RULE
        or output.get("field_threshold") != 0.0
        or output.get("zero_tie_policy") != PACRE_VC_ZERO_TIE_POLICY
        or output.get("hard_union") is not True
        or output.get("sigmoid_applied") is not False
        or output.get("PACRE_threshold_search_performed") is not False
        or not isinstance(base_selection, Mapping)
        or set(base_selection)
        != {
            "policy",
            "base_threshold_search_performed",
            "candidate_threshold_grid",
            "candidate_count",
            "candidate_ledger",
            "selected_threshold",
            "stage_a_config_sha256",
            "budget",
        }
        or base_selection.get("policy")
        != PACRE_VC_BASE_AT_B_SELECTION_POLICY
        or base_selection.get("base_threshold_search_performed")
        is not True
        or base_selection.get("candidate_threshold_grid")
        != list(PACRE_VC_FORMAL_BASE_THRESHOLD_GRID)
        or base_selection.get("candidate_count")
        != len(PACRE_VC_FORMAL_BASE_THRESHOLD_GRID)
        or base_selection.get("selected_threshold")
        not in PACRE_VC_FORMAL_BASE_THRESHOLD_GRID
        or base_selection.get("stage_a_config_sha256")
        != PACRE_VC_FORMAL_STAGE_A_CONFIG_SHA256
        or base_selection.get("budget")
        != {
            "pixel_fa_budget": PACRE_VC_MAXIMUM_PIXEL_FA,
            "component_fa_per_mp_budget": (
                PACRE_VC_MAXIMUM_FP_COMPONENTS_PER_MP
            ),
            "raw_background_fa_budget": (
                PACRE_VC_MAXIMUM_RAW_BACKGROUND_FA
            ),
            "minimum_retention": 0.99,
        }
        or not isinstance(operating, Mapping)
        or set(operating) != {"Base@A", "Base@B", "Base@A+CURE"}
        or not isinstance(stored_gate, Mapping)
    ):
        raise ValueError("D_V fixed evaluation contract changed")
    for name in (
        "exact_zero_field_pixels",
        "negative_field_pixels",
        "completion_pixels",
    ):
        if (
            isinstance(output[name], bool)
            or not isinstance(output[name], int)
            or output[name] < 0
        ):
            raise ValueError(f"D_V output_contract.{name} changed")
    selected_threshold = base_selection["selected_threshold"]
    if (
        isinstance(selected_threshold, bool)
        or not isinstance(selected_threshold, (int, float))
        or not isfinite(float(selected_threshold))
    ):
        raise ValueError("Base@B selected threshold changed")
    metrics: dict[str, AggregateEvaluation] = {}
    for name in ("Base@A", "Base@B", "Base@A+CURE"):
        row = operating[name]
        if (
            not isinstance(row, Mapping)
            or set(row) != {"aggregate_evaluation", "summary"}
        ):
            raise ValueError(f"{name} persisted row changed")
        parsed = _parse_metrics(
            row["aggregate_evaluation"],
            name=name,
        )
        if row["summary"] != _summary(parsed):
            raise ValueError(f"{name} summary differs from aggregate")
        metrics[name] = parsed
    ledger_payload = base_selection["candidate_ledger"]
    if not isinstance(ledger_payload, Mapping):
        raise TypeError("Base@B 51-point ledger must be a mapping")
    if set(ledger_payload) != {
        "schema_version",
        "method",
        "mode",
        "anchor_threshold",
        "candidate_count",
        "entries",
        "ledger_fingerprint",
    }:
        raise ValueError("Base@B 51-point ledger fields changed")
    verify_fingerprinted(
        ledger_payload,
        field="ledger_fingerprint",
    )
    candidate_rows = ledger_payload.get("entries")
    if (
        ledger_payload.get("schema_version")
        != "cure-lite-v23-pacre-vc-base-at-b-51-ledger-v1"
        or ledger_payload.get("method") != "Base@B"
        or ledger_payload.get("mode") != "base"
        or ledger_payload.get("anchor_threshold")
        != PACRE_VC_FORMAL_BASE_THRESHOLD
        or ledger_payload.get("candidate_count")
        != len(PACRE_VC_FORMAL_BASE_THRESHOLD_GRID)
        or not isinstance(candidate_rows, list)
        or len(candidate_rows)
        != len(PACRE_VC_FORMAL_BASE_THRESHOLD_GRID)
    ):
        raise ValueError("Base@B 51-point ledger contract changed")
    candidate_entries: list[CandidateEvaluation] = []
    for expected_threshold, candidate_row in zip(
        PACRE_VC_FORMAL_BASE_THRESHOLD_GRID,
        candidate_rows,
        strict=True,
    ):
        if (
            not isinstance(candidate_row, Mapping)
            or set(candidate_row)
            != {
                "threshold",
                "aggregate_evaluation",
                "budget_accepted",
            }
            or type(candidate_row.get("threshold")) is not float
            or candidate_row.get("threshold") != expected_threshold
        ):
            raise ValueError("Base@B candidate row/order changed")
        candidate_metrics = _parse_metrics(
            candidate_row["aggregate_evaluation"],
            name=f"Base@B[{expected_threshold}]",
        )
        accepted = _budget_accepts(candidate_metrics)
        if (
            candidate_row.get("budget_accepted") is not accepted
            or candidate_metrics.budget_violation is accepted
        ):
            raise ValueError("Base@B candidate budget flag changed")
        candidate_entries.append(
            CandidateEvaluation(
                method="Base@B",
                mode="base",
                threshold=expected_threshold,
                metrics=candidate_metrics,
            )
        )
    anchor_candidates = tuple(
        entry
        for entry in candidate_entries
        if entry.threshold == PACRE_VC_FORMAL_BASE_THRESHOLD
    )
    if len(anchor_candidates) != 1:
        raise ValueError("Base@B ledger lacks the Base@A anchor")
    ledger = CalibrationCandidateLedger(
        base_method="Base@B",
        anchor_threshold=PACRE_VC_FORMAL_BASE_THRESHOLD,
        anchor_metrics=anchor_candidates[0].metrics,
        entries=tuple(candidate_entries),
    )
    budget = FalseAlarmBudget(
        pixel_fa_budget=PACRE_VC_MAXIMUM_PIXEL_FA,
        component_fa_per_mp_budget=(
            PACRE_VC_MAXIMUM_FP_COMPONENTS_PER_MP
        ),
        raw_background_fa_budget=(
            PACRE_VC_MAXIMUM_RAW_BACKGROUND_FA
        ),
        minimum_retention=0.99,
    )
    selection = ledger.select("Base@B", budget)
    if (
        ledger.anchor_metrics != metrics["Base@A"]
        or not selection.feasible
        or selection.threshold != float(selected_threshold)
        or selection.metrics != metrics["Base@B"]
    ):
        raise RuntimeError(
            "stored Base@B differs from independent 51-point reselection"
        )
    valid = tuple(
        name
        for name in ("Base@A", "Base@B")
        if metrics[name].budget_violation is False
        and _budget_accepts(metrics[name])
    )
    if not valid:
        raise ValueError("no valid frozen Base comparator remains")
    best_true = max(
        metrics[name].retained_anchor_covered
        + metrics[name].recovered_anchor_misses
        for name in valid
    )
    best_recovered = max(
        metrics[name].recovered_anchor_misses for name in valid
    )
    best_miou = max(metrics[name].miou for name in valid)
    best_niou = max(metrics[name].niou for name in valid)
    cure = metrics["Base@A+CURE"]
    cure_true = (
        cure.retained_anchor_covered + cure.recovered_anchor_misses
    )
    margins = {
        "true_targets": cure_true - best_true,
        "recovered_anchor_misses": (
            cure.recovered_anchor_misses - best_recovered
        ),
    }
    checks = dict(
        sorted(
            {
                "CURE_true_targets_strictly_above_best_valid_Base": (
                    cure_true > best_true
                ),
                "CURE_recovered_anchor_misses_strictly_above_best_valid_Base": (
                    cure.recovered_anchor_misses > best_recovered
                ),
                "CURE_mIoU_not_below_best_valid_Base": (
                    cure.miou >= best_miou
                ),
                "CURE_nIoU_not_below_best_valid_Base": (
                    cure.niou >= best_niou
                ),
                "CURE_retention_equal_1": isclose(
                    cure.retention,
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ),
                "CURE_pixel_Fa_le_1e-4": (
                    cure.pixel_fa <= PACRE_VC_MAXIMUM_PIXEL_FA
                ),
                "CURE_raw_background_Fa_le_1e-4": (
                    cure.raw_background_fa
                    <= PACRE_VC_MAXIMUM_RAW_BACKGROUND_FA
                ),
                "CURE_false_positive_components_per_megapixel_le_100": (
                    cure.fp_components_per_mp
                    <= PACRE_VC_MAXIMUM_FP_COMPONENTS_PER_MP
                ),
                "CURE_budget_violation_false": (
                    cure.budget_violation is False
                ),
                "D_T_payload_accessed_false": True,
            }.items()
        )
    )
    gate_passed = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    status = (
        "PACRE_V23_FORMAL_D_V_GATE_PASS"
        if gate_passed
        else "PACRE_V23_FORMAL_D_V_GATE_FAIL"
    )
    expected_gate = {
        "comparison": "best_valid_Base",
        "valid_base_names": list(valid),
        "requirements": {
            "true_targets": "strictly_greater_than_best_valid_Base",
            "recovered_anchor_misses": (
                "strictly_greater_than_best_valid_Base"
            ),
            "mIoU": "not_below_best_valid_Base",
            "nIoU": "not_below_best_valid_Base",
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
        "best_valid_Base": {
            "true_targets": best_true,
            "recovered_anchor_misses": best_recovered,
            "mIoU": best_miou,
            "nIoU": best_niou,
        },
        "CURE_margins": margins,
        "checks": checks,
        "failed_checks": failed,
        "gate_passed": gate_passed,
        "status": status,
    }
    if dict(stored_gate) != expected_gate:
        raise RuntimeError(
            "stored D_V decision differs from independent recomputation"
        )
    if result["eligible_for_D_T_confirmation"] is not gate_passed:
        raise RuntimeError("stored D_T eligibility differs from recomputation")
    return {
        "gate_passed": gate_passed,
        "failed_checks": failed,
        "checks": checks,
        "best_valid_Base": expected_gate["best_valid_Base"],
        "CURE_margins": margins,
    }


def _verify_model_binding(
    binding: object,
    *,
    formal: Mapping[str, object],
    artifact: LoadedPACREVCFormalArtifact,
) -> str:
    if not isinstance(binding, Mapping):
        raise TypeError("model binding must be a mapping")
    payload = dict(binding)
    formal_budget = payload.get("formal_budget")
    formal_artifact = payload.get("formal_artifact")
    d_v_protocol = payload.get("D_V_protocol")
    if (
        set(payload)
        != {
            "schema_version",
            "method",
            "formal_budget",
            "formal_artifact",
            "D_V_protocol",
        }
        or payload.get("schema_version")
        != PACRE_VC_FORMAL_MODEL_BINDING_SCHEMA
        or payload.get("method") != PACRE_VC_FORMAL_METHOD
        or not isinstance(formal_budget, Mapping)
        or dict(formal_budget)
        != {
            "seed": SEED,
            "epochs": EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "completed_updates": UPDATES,
            "trained_from_scratch": True,
            "resumed": False,
            "runtime_splits": ["D_R"],
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
        or not isinstance(formal_artifact, Mapping)
        or set(formal_artifact)
        != {
            "artifact_fingerprint",
            "training_result_fingerprint",
            "authorization_fingerprint",
            "source_closure_fingerprint",
            "model_state_fingerprint",
            "model_config_fingerprint",
        }
        or formal_artifact.get("artifact_fingerprint")
        != formal["artifact_fingerprint"]
        or formal_artifact.get("training_result_fingerprint")
        != formal["training_result_fingerprint"]
        or formal_artifact.get("authorization_fingerprint")
        != formal["authorization_fingerprint"]
        or formal_artifact.get("source_closure_fingerprint")
        != formal["source_closure_fingerprint"]
        or formal_artifact.get("model_state_fingerprint")
        != formal["module_state_fingerprint"]
        or not isinstance(d_v_protocol, Mapping)
        or set(d_v_protocol)
        != {
            "comparison_protocol_fingerprint",
            "manifest_fingerprint",
            "manifest_file_sha256",
            "preprocessing_fingerprint",
            "base_fingerprint",
            "base_state_fingerprint",
            "base_index_fingerprint",
            "image_fingerprint",
            "GT_fingerprint",
        }
        or d_v_protocol.get("comparison_protocol_fingerprint")
        != COMPARISON_PROTOCOL_FINGERPRINT
        or d_v_protocol.get("manifest_fingerprint")
        != MANIFEST_FINGERPRINT
        or d_v_protocol.get("manifest_file_sha256") != MANIFEST_SHA256
        or d_v_protocol.get("preprocessing_fingerprint")
        != PREPROCESSING_FINGERPRINT
        or d_v_protocol.get("base_fingerprint") != BASE_FINGERPRINT
        or d_v_protocol.get("base_state_fingerprint")
        != BASE_STATE_FINGERPRINT
        or d_v_protocol.get("base_index_fingerprint")
        != BASE_INDEX_FINGERPRINT
        or d_v_protocol.get("image_fingerprint")
        != D_V_IMAGE_FINGERPRINT
        or d_v_protocol.get("GT_fingerprint") != D_V_GT_FINGERPRINT
    ):
        raise RuntimeError("persisted model/D_V binding changed")
    artifact.verify_unchanged()
    return stable_fingerprint(payload)


def _verify_result_bindings(
    result: Mapping[str, object],
    *,
    model_binding_fingerprint: str,
    formal: Mapping[str, object],
) -> None:
    bindings = result.get("bindings")
    if not isinstance(bindings, Mapping):
        raise TypeError("evaluation-result bindings must be a mapping")
    for name in (
        "base_samples_fingerprint",
        "cure_samples_fingerprint",
        "model_binding_fingerprint",
        "artifact_fingerprint",
        "model_state_fingerprint",
        "comparison_protocol_fingerprint",
        "manifest_fingerprint",
        "base_state_fingerprint",
        "D_V_base_index_fingerprint",
        "D_V_image_fingerprint",
        "D_V_GT_fingerprint",
    ):
        _digest(bindings.get(name), name=f"result.bindings.{name}")
    if (
        bindings.get("model_binding_fingerprint")
        != model_binding_fingerprint
        or bindings.get("artifact_fingerprint")
        != formal["artifact_fingerprint"]
        or bindings.get("model_state_fingerprint")
        != formal["module_state_fingerprint"]
        or bindings.get("comparison_protocol_fingerprint")
        != COMPARISON_PROTOCOL_FINGERPRINT
        or bindings.get("manifest_fingerprint") != MANIFEST_FINGERPRINT
        or bindings.get("base_state_fingerprint")
        != BASE_STATE_FINGERPRINT
        or bindings.get("D_V_base_index_fingerprint")
        != BASE_INDEX_FINGERPRINT
        or bindings.get("D_V_image_fingerprint")
        != D_V_IMAGE_FINGERPRINT
        or bindings.get("D_V_GT_fingerprint") != D_V_GT_FINGERPRINT
    ):
        raise RuntimeError("evaluation-result source bindings changed")


def verify_terminal(output: Path = OUTPUT_PATH) -> dict[str, object]:
    """Verify one exact published terminal without reopening D_V tensors."""

    output = Path(output)
    if output != OUTPUT_PATH:
        raise ValueError("only the fixed PACRE-VC D_V output is verifiable")
    if (
        not output.is_dir()
        or output.is_symlink()
        or output.resolve(strict=True) != output
        or STAGING_PATH.exists()
        or STAGING_PATH.is_symlink()
    ):
        raise RuntimeError("fixed PACRE-VC D_V output is not canonical")
    members = {path.name: path for path in output.iterdir()}
    if (
        set(members) != FINAL_MEMBERS
        or any(
            path.is_symlink() or not path.is_file()
            for path in members.values()
        )
    ):
        raise RuntimeError("PACRE-VC D_V terminal population changed")

    protocol = _verify_metadata_and_protocol()
    runtime = _verify_runtime()
    artifact, formal = _load_formal_terminal()
    artifact.verify_unchanged()
    if (
        protocol.comparison_protocol_fingerprint
        != COMPARISON_PROTOCOL_FINGERPRINT
    ):
        raise AssertionError("strict protocol verification changed")

    claim = read_strict_json(members[CLAIM_FILE])
    receipt = read_strict_json(members[RECEIPT_FILE])
    decision = read_strict_json(members[DECISION_FILE])
    complete = read_strict_json(members[COMPLETE_FILE])
    claim_fingerprint = verify_fingerprinted(
        claim,
        field="claim_fingerprint",
    )
    receipt_fingerprint = verify_fingerprinted(receipt)
    decision_fingerprint = verify_fingerprinted(
        decision,
        field="decision_fingerprint",
    )
    complete_fingerprint = verify_fingerprinted(
        complete,
        field="complete_fingerprint",
    )
    if set(claim) != {
        "schema_version",
        "run_id",
        "status",
        "plan_fingerprint",
        "formal_terminal_binding_fingerprint",
        "formal_complete_fingerprint",
        "formal_artifact_fingerprint",
        "source_closure_fingerprint",
        "runtime",
        "single_attempt",
        "resume_allowed",
        "retry_allowed",
        "overwrite_allowed",
        "output_reusable",
        "D_V_metadata_verified",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "claim_fingerprint",
    }:
        raise ValueError("D_V claim fields changed")
    if set(receipt) != {
        "schema_version",
        "run_id",
        "status",
        "plan",
        "plan_fingerprint",
        "claim_fingerprint",
        "formal_terminal",
        "formal_terminal_binding_fingerprint",
        "model_binding",
        "model_binding_fingerprint",
        "evaluation_result",
        "evaluation_result_fingerprint",
        "decision_fingerprint",
        "runtime_splits",
        "D_V_adaptive",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "model_training_performed",
        "model_state_update_performed",
        "checkpoint_selection_performed",
        "PACRE_threshold_search_performed",
        "Base@B_51_point_selection_performed",
        "minimum_fixed_uplift_margin",
        "authorizes_D_T",
        "final_model_success_established",
        "receipt_fingerprint",
    }:
        raise ValueError("D_V receipt fields changed")
    if set(decision) != {
        "schema_version",
        "run_id",
        "status",
        "D_V_adaptive",
        "gate_passed",
        "failed_checks",
        "checks",
        "comparison",
        "best_valid_Base",
        "CURE_margins",
        "minimum_fixed_uplift_margin",
        "plus_one_is_sufficient",
        "evaluation_result_fingerprint",
        "model_binding_fingerprint",
        "formal_complete_fingerprint",
        "formal_artifact_fingerprint",
        "eligible_for_future_separately_preregistered_D_T",
        "authorizes_D_T",
        "D_T_payload_accessed",
        "model_training_performed",
        "model_state_update_performed",
        "checkpoint_selection_performed",
        "final_model_success_established",
        "decision_fingerprint",
    }:
        raise ValueError("D_V decision fields changed")
    if set(complete) != {
        "schema_version",
        "run_id",
        "status",
        "final_members",
        "artifact_files",
        "artifact_count",
        "claim_fingerprint",
        "receipt_fingerprint",
        "decision_fingerprint",
        "evaluation_result_fingerprint",
        "model_binding_fingerprint",
        "formal_complete_fingerprint",
        "formal_artifact_fingerprint",
        "source_closure_fingerprint",
        "D_V_adaptive",
        "gate_passed",
        "eligible_for_future_separately_preregistered_D_T",
        "authorizes_D_T",
        "D_T_payload_accessed",
        "model_training_performed",
        "model_state_update_performed",
        "checkpoint_selection_performed",
        "final_model_success_established",
        "complete_fingerprint",
    }:
        raise ValueError("D_V COMPLETE fields changed")
    plan = _expected_plan()
    formal_binding_fingerprint = stable_fingerprint(formal)
    if (
        claim.get("schema_version") != CLAIM_SCHEMA
        or claim.get("run_id") != RUN_ID
        or claim.get("status")
        != "claimed_before_D_V_materialization"
        or claim.get("plan_fingerprint") != stable_fingerprint(plan)
        or claim.get("formal_terminal_binding_fingerprint")
        != formal_binding_fingerprint
        or claim.get("formal_complete_fingerprint")
        != formal["complete_fingerprint"]
        or claim.get("formal_artifact_fingerprint")
        != formal["artifact_fingerprint"]
        or claim.get("source_closure_fingerprint")
        != formal["source_closure_fingerprint"]
        or claim.get("runtime") != runtime
        or claim.get("single_attempt") is not True
        or claim.get("resume_allowed") is not False
        or claim.get("retry_allowed") is not False
        or claim.get("overwrite_allowed") is not False
        or claim.get("output_reusable") is not False
        or claim.get("D_V_metadata_verified") is not True
        or claim.get("D_V_payload_accessed") is not False
        or claim.get("D_T_payload_accessed") is not False
    ):
        raise RuntimeError("D_V pre-materialization claim changed")

    result = receipt.get("evaluation_result")
    if not isinstance(result, Mapping):
        raise TypeError("D_V receipt has no aggregate evaluation result")
    result_payload = dict(result)
    result_fingerprint = stable_fingerprint(result_payload)
    gate = _recompute_gate(result_payload)
    model_binding_fingerprint = _verify_model_binding(
        receipt.get("model_binding"),
        formal=formal,
        artifact=artifact,
    )
    _verify_result_bindings(
        result_payload,
        model_binding_fingerprint=model_binding_fingerprint,
        formal=formal,
    )

    expected_status = (
        "PACRE_V23_FORMAL_D_V_ADAPTIVE_PASS"
        if gate["gate_passed"]
        else "PACRE_V23_FORMAL_D_V_ADAPTIVE_FAIL"
    )
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA
        or receipt.get("run_id") != RUN_ID
        or receipt.get("status") != "complete"
        or receipt.get("plan") != plan
        or receipt.get("plan_fingerprint") != stable_fingerprint(plan)
        or receipt.get("claim_fingerprint") != claim_fingerprint
        or receipt.get("formal_terminal") != formal
        or receipt.get("formal_terminal_binding_fingerprint")
        != formal_binding_fingerprint
        or receipt.get("model_binding_fingerprint")
        != model_binding_fingerprint
        or receipt.get("evaluation_result_fingerprint")
        != result_fingerprint
        or receipt.get("decision_fingerprint")
        != decision_fingerprint
        or receipt.get("runtime_splits") != ["D_V"]
        or receipt.get("D_V_adaptive") is not True
        or receipt.get("D_V_payload_accessed") is not True
        or receipt.get("D_T_payload_accessed") is not False
        or receipt.get("model_training_performed") is not False
        or receipt.get("model_state_update_performed") is not False
        or receipt.get("checkpoint_selection_performed") is not False
        or receipt.get("PACRE_threshold_search_performed") is not False
        or receipt.get("Base@B_51_point_selection_performed") is not True
        or receipt.get("minimum_fixed_uplift_margin") is not None
        or receipt.get("authorizes_D_T") is not False
        or receipt.get("final_model_success_established") is not False
    ):
        raise RuntimeError("D_V receipt associations changed")
    if (
        decision.get("schema_version") != DECISION_SCHEMA
        or decision.get("run_id") != RUN_ID
        or decision.get("status") != expected_status
        or decision.get("D_V_adaptive") is not True
        or decision.get("gate_passed") is not gate["gate_passed"]
        or decision.get("failed_checks") != gate["failed_checks"]
        or decision.get("checks") != gate["checks"]
        or decision.get("comparison") != "best_valid_Base"
        or decision.get("best_valid_Base") != gate["best_valid_Base"]
        or decision.get("CURE_margins") != gate["CURE_margins"]
        or decision.get("minimum_fixed_uplift_margin") is not None
        or decision.get("plus_one_is_sufficient") is not True
        or decision.get("evaluation_result_fingerprint")
        != result_fingerprint
        or decision.get("model_binding_fingerprint")
        != model_binding_fingerprint
        or decision.get("formal_complete_fingerprint")
        != formal["complete_fingerprint"]
        or decision.get("formal_artifact_fingerprint")
        != formal["artifact_fingerprint"]
        or decision.get(
            "eligible_for_future_separately_preregistered_D_T"
        )
        is not gate["gate_passed"]
        or decision.get("authorizes_D_T") is not False
        or decision.get("D_T_payload_accessed") is not False
        or decision.get("model_training_performed") is not False
        or decision.get("model_state_update_performed") is not False
        or decision.get("checkpoint_selection_performed") is not False
        or decision.get("final_model_success_established") is not False
    ):
        raise RuntimeError(
            "stored D_V decision differs from recomputed decision"
        )

    expected_hashes = {
        name: file_sha256(members[name])
        for name in sorted(
            (CLAIM_FILE, RECEIPT_FILE, DECISION_FILE)
        )
    }
    if (
        complete.get("schema_version") != COMPLETE_SCHEMA
        or complete.get("run_id") != RUN_ID
        or complete.get("status") != expected_status
        or complete.get("final_members") != sorted(FINAL_MEMBERS)
        or complete.get("artifact_files") != expected_hashes
        or complete.get("artifact_count") != len(expected_hashes)
        or complete.get("claim_fingerprint") != claim_fingerprint
        or complete.get("receipt_fingerprint")
        != receipt_fingerprint
        or complete.get("decision_fingerprint")
        != decision_fingerprint
        or complete.get("evaluation_result_fingerprint")
        != result_fingerprint
        or complete.get("model_binding_fingerprint")
        != model_binding_fingerprint
        or complete.get("formal_complete_fingerprint")
        != formal["complete_fingerprint"]
        or complete.get("formal_artifact_fingerprint")
        != formal["artifact_fingerprint"]
        or complete.get("source_closure_fingerprint")
        != formal["source_closure_fingerprint"]
        or complete.get("D_V_adaptive") is not True
        or complete.get("gate_passed") is not gate["gate_passed"]
        or complete.get(
            "eligible_for_future_separately_preregistered_D_T"
        )
        is not gate["gate_passed"]
        or complete.get("authorizes_D_T") is not False
        or complete.get("D_T_payload_accessed") is not False
        or complete.get("model_training_performed") is not False
        or complete.get("model_state_update_performed") is not False
        or complete.get("checkpoint_selection_performed") is not False
        or complete.get("final_model_success_established") is not False
    ):
        raise RuntimeError("D_V COMPLETE terminal bindings changed")
    return {
        "run_id": RUN_ID,
        "output": str(output),
        "status": expected_status,
        "D_V_adaptive": True,
        "gate_passed": gate["gate_passed"],
        "failed_checks": gate["failed_checks"],
        "CURE_margins": gate["CURE_margins"],
        "receipt_fingerprint": receipt_fingerprint,
        "decision_fingerprint": decision_fingerprint,
        "complete_fingerprint": complete_fingerprint,
        "evaluation_result_fingerprint": result_fingerprint,
        "model_binding_fingerprint": model_binding_fingerprint,
        "formal_complete_fingerprint": formal["complete_fingerprint"],
        "formal_artifact_fingerprint": formal["artifact_fingerprint"],
        "runtime_environment_fingerprint": runtime[
            "runtime_environment_fingerprint"
        ],
        "D_V_payload_reopened_by_verifier": False,
        "D_T_payload_accessed": False,
        "authorizes_D_T": False,
        "final_model_success_established": False,
        "terminal_verified": True,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    return argparse.Namespace()


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    result = verify_terminal()
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
