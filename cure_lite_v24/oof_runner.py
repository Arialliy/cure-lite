"""Concrete real-D_R OOF-4 orchestrator for GCR-PACRE v24.

There are intentionally no caller-supplied factories, loaders, trainers,
evaluators, or callbacks in this module.  Every runtime component is the
fixed implementation imported below and bound by the unified source closure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Mapping

import torch

from cure_lite.cache.schema import canonical_json, stable_fingerprint
from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
)
from cure_lite.data import PreprocessConfig
from cure_lite.experiment.coverage_state_observability_protocol import (
    CoverageStateObservabilityProtocol,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    CoverageStateRealDRSourceBinding,
)
from cure_lite.experiment.geometry_catalog_protocol import (
    GeometryCatalogProtocol,
)
from cure_lite.paired_types import tensor_content_fingerprint
from tools.gcr_pacre_v24_protocol import (
    BASE_A_THRESHOLD,
    BASE_B_THRESHOLD_GRID,
    OOF_ARMS,
    VerifiedAccessAudit,
    VerifiedOOF4Split,
    VerifiedOOFDecision,
    canonical_json as protocol_canonical_json,
    combine_oof4_factual_pools,
    decide_oof4_pooled,
    pool_factual_only_rows,
    stable_fingerprint as protocol_fingerprint,
    validate_oof_fold_execution_receipt,
    verify_access_audit_receipt,
    verify_gate_path_receipt,
)

from .artifact_io import (
    atomic_write_new_canonical_json,
    read_canonical_json,
    regular_file_receipt,
)
from .gcr_pacre import CoverageStateGCRPACREConfig
from .oof_cache import (
    OOF_EVENT_HOLDOUT_CACHE_CREATED,
    OOF_EVENT_HOLDOUT_FIRST_OPEN,
    OOF_EVENT_TERMINALS_SEALED,
    OOF_EVENT_TRAIN_CACHE_CREATED,
    OOF_EVENT_TRAINING_RUN_START,
    VerifiedOOFCacheArtifact,
    VerifiedOOFTerminalSeal,
    issue_oof_cache_reader,
    load_oof_cache_payload,
    save_oof_cache_artifact_new,
    seal_oof_training_terminals,
    verify_oof_six_cache_independence,
)
from .oof_evaluation import (
    OOFConcreteEvaluator,
    OOFEvaluationDataset,
    OOFEvaluationLedger,
    OOF_BASE_A_ARM,
    OOF_BASE_B_ARM,
    OOF_G1_ARM,
    OOF_V23_ARM,
    OOF_V24_ARM,
)
from .oof_inputs import (
    build_oof_restricted_holdout_inputs,
    build_oof_restricted_train_inputs,
)
from .oof_run_start import (
    OOF_EPOCHS,
    OOF_SEED,
    OOF_STEPS_PER_EPOCH,
    VerifiedOOFExecutionAuthorization,
    create_oof_training_run_start_new,
    require_verified_oof_execution_authorization,
    run_start_artifact_receipt,
)
from .oof_split import (
    VerifiedOOFFoldClosure,
    verify_all_oof_fold_closures,
)
from .oof_training import (
    OOF_CANDIDATE_ARM,
    OOF_CONTROL_ARM,
    OOF_OBJECTIVE,
    OOFPairedTrainingResult,
    load_oof_terminal_model_strict,
    require_verified_oof_completed_400_capability,
    run_paired_oof_training_400,
)
from .source_closure import (
    gcr_pacre_v24_source_closure_fingerprint,
    gcr_pacre_v24_source_closure_hashes,
)


OOF_REAL_RESULT_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-real-D_R-oof4-result-v1"
)
OOF_FOLD_RUNTIME_RECEIPT_SCHEMA: Final = (
    "cure-lite-v24-oof-fold-execution-v3"
)
_GRID_SOURCE = (
    "protocols/IRSTD-1K/stage_a_seed42_fx_v3/stage_a_config.json"
)
_GRID_SOURCE_SHA256 = (
    "6eecdc10f87a043cafb945db40d0b767b5f0a2ccb64963c1043160f165ce9d6c"
)
_BASE_B_SELECTOR_POLICY = (
    "maximize_pd",
    "maximize_retention",
    "minimize_pixel_fa",
    "minimize_raw_background_fa",
    "minimize_fp_components_per_mp",
    "maximize_threshold",
)


def preflight_oof_execution_device(
    device: torch.device | str,
) -> str:
    """Resolve and allocate the fixed execution device before any writes."""

    try:
        resolved = torch.device(device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError("OOF execution device is invalid") from error
    if resolved.type == "cuda":
        if resolved.index is None:
            raise ValueError("OOF CUDA device requires an explicit index")
        if not torch.cuda.is_available():
            raise RuntimeError("OOF CUDA execution is unavailable")
        device_count = torch.cuda.device_count()
        if (
            isinstance(device_count, bool)
            or not isinstance(device_count, int)
            or resolved.index < 0
            or resolved.index >= device_count
        ):
            raise ValueError("OOF CUDA device index is out of range")
    elif resolved.type != "cpu":
        raise ValueError("OOF execution device must be CPU or explicit CUDA")
    try:
        probe = torch.empty(1, dtype=torch.float32, device=resolved)
        if probe.device.type != resolved.type:
            raise RuntimeError("OOF execution device probe changed device")
        if (
            resolved.type == "cuda"
            and probe.device.index != resolved.index
        ):
            raise RuntimeError("OOF CUDA context probe changed index")
        del probe
    except Exception as error:
        raise RuntimeError(
            "OOF execution device context allocation failed"
        ) from error
    return str(resolved)


def _seal(body: Mapping[str, object], *, field: str) -> dict[str, object]:
    value = dict(body)
    if field in value:
        raise ValueError(f"{field} is already present")
    return {**value, field: protocol_fingerprint(value)}


def _artifact_model_view(
    artifact: Mapping[str, object],
) -> dict[str, object]:
    # Persist the complete terminal ledger, not a lossy path/hash projection:
    # the protocol verifier must be able to recompute the exact capability
    # fingerprint after the producer process has exited.
    return dict(artifact)


def _save_evaluation_ledger_new(
    path: Path,
    ledger: OOFEvaluationLedger,
) -> dict[str, object]:
    ledger.verify_unchanged()
    payload = {
        **ledger.canonical_payload(),
        "ledger_fingerprint": ledger.ledger_fingerprint,
    }
    target = atomic_write_new_canonical_json(path, payload)
    return {
        **regular_file_receipt(target),
        "ledger_fingerprint": ledger.ledger_fingerprint,
    }


def _access_receipt(
    *,
    fold_id: int,
    source_binding_fingerprint: str,
    cache_artifacts: tuple[VerifiedOOFCacheArtifact, ...],
) -> dict[str, object]:
    observed = [
        {
            "split": "D_R",
            "logical_id": cache.cache_id,
            "purpose": (
                "train_cache_materialization"
                if cache.partition == "train"
                else "read_only_holdout_cache_materialization"
            ),
            "source_fingerprint": cache.file_sha256,
        }
        for cache in sorted(
            cache_artifacts,
            key=lambda value: (value.partition, value.arm),
        )
    ]
    body = {
        "schema_version": "cure-lite-v24-split-access-audit-v1",
        "stage_id": f"oof4_fold_{fold_id}",
        "allowed_splits": ["D_R"],
        "observed_payloads": observed,
        "source_manifest_fingerprint": source_binding_fingerprint,
        "event_log_fingerprint": protocol_fingerprint(observed),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return _seal(body, field="receipt_fingerprint")


def _terminal_seal_payload(
    seal: VerifiedOOFTerminalSeal,
    training: OOFPairedTrainingResult,
) -> dict[str, object]:
    control = require_verified_oof_completed_400_capability(
        training.control_capability,
        arm=OOF_CONTROL_ARM,
    )
    candidate = require_verified_oof_completed_400_capability(
        training.candidate_capability,
        arm=OOF_CANDIDATE_ARM,
    )
    body = {
        "schema_version": (
            "cure-lite-v24-gcr-pacre-oof4-training-terminal-seal-v1"
        ),
        "fold_id": seal.fold_id,
        "closure_fingerprint": seal.closure_fingerprint,
        "terminal_artifact_fingerprints": dict(
            seal.terminal_artifact_fingerprints
        ),
        "completed_400_capability_fingerprints": dict(
            seal.completed_400_capability_fingerprints
        ),
        "run_start_marker_fingerprint": (
            seal.run_start_marker_fingerprint
        ),
        "shared_initial_parameter_fingerprint": (
            seal.shared_initial_parameter_fingerprint
        ),
        "initial_parameters": list(control.initial_parameters),
        "schedule_fingerprint": control.schedule_fingerprint,
        "batch_sequence_fingerprint": control.batch_sequence_fingerprint,
        "semantic_cache_fingerprint": (
            control.semantic_cache_fingerprint
        ),
        "optimizer_config_fingerprint": (
            control.optimizer_config_fingerprint
        ),
        "objective_policy_fingerprint": (
            control.objective_policy_fingerprint
        ),
        "event_index": OOF_EVENT_TERMINALS_SEALED,
    }
    if (
        control.initial_parameters != candidate.initial_parameters
        or stable_fingerprint(body) != seal.seal_fingerprint
    ):
        raise RuntimeError("OOF terminal seal payload cannot be recomputed")
    return {**body, "seal_fingerprint": seal.seal_fingerprint}


def _training_arm_receipts(
    *,
    closure: VerifiedOOFFoldClosure,
    training: OOFPairedTrainingResult,
    source_hashes: tuple[tuple[str, str], ...],
    device: torch.device | str,
) -> dict[str, object]:
    control_cap = require_verified_oof_completed_400_capability(
        training.control_capability,
        fold_closure=closure,
        arm=OOF_CONTROL_ARM,
    )
    candidate_cap = require_verified_oof_completed_400_capability(
        training.candidate_capability,
        fold_closure=closure,
        arm=OOF_CANDIDATE_ARM,
    )
    rows = {}
    for arm, capability, result, artifact in (
        (
            OOF_CONTROL_ARM,
            control_cap,
            training.control_training_result,
            training.control_terminal_artifact,
        ),
        (
            OOF_CANDIDATE_ARM,
            candidate_cap,
            training.candidate_training_result,
            training.candidate_terminal_artifact,
        ),
    ):
        rows[arm] = {
            "seed": OOF_SEED,
            "epochs": OOF_EPOCHS,
            "steps_per_epoch": OOF_STEPS_PER_EPOCH,
            "completed_updates": result.completed_updates,
            "training_invocations": 1,
            "from_scratch": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "checkpoint_policy": "final_only",
            "optimizer_state_initial_empty": True,
            "train_root_source_ids": list(
                closure.train_root_source_ids
            ),
            "train_sample_ids": list(closure.train_sample_ids),
            "schedule_fingerprint": capability.schedule_fingerprint,
            "batch_sequence_fingerprint": (
                capability.batch_sequence_fingerprint
            ),
            "training_population_fingerprint": (
                capability.semantic_cache_fingerprint
            ),
            "initial_shared_parameter_fingerprint": (
                capability.shared_initial_parameter_fingerprint
            ),
            "initial_parameters": list(capability.initial_parameters),
            "completed_400_capability_fingerprint": (
                capability.capability_fingerprint
            ),
            "completed_400_capability": capability.payload,
            "run_start_marker_fingerprint": (
                capability.run_start_marker_fingerprint
            ),
            "PMOPE_fingerprint": (
                capability.objective_policy_fingerprint
            ),
            "Adam_policy_fingerprint": (
                capability.optimizer_config_fingerprint
            ),
            "dtype_device_policy_fingerprint": protocol_fingerprint(
                {
                    "dtype": "torch.float32",
                    "device": str(torch.device(device)),
                    "deterministic_algorithms": True,
                    "tf32": False,
                }
            ),
            "source_hashes": dict(source_hashes),
            "module_instance_id": capability.module_instance_id,
            "optimizer_instance_id": capability.optimizer_instance_id,
            "parameter_storage_ledger": list(
                capability.parameter_storage_ledger
            ),
            "parameter_storage_ledger_fingerprint": (
                capability.parameter_storage_ledger_fingerprint
            ),
            "initial_model_fingerprint": (
                result.initial_model_fingerprint
            ),
            "final_model_fingerprint": result.final_model_fingerprint,
            "terminal_artifact_fingerprint": (
                capability.terminal_artifact_fingerprint
            ),
            "terminal_artifact": _artifact_model_view(artifact),
        }
    return rows


def _base_b_selection_receipt(
    *,
    closure: VerifiedOOFFoldClosure,
    train_cache: VerifiedOOFCacheArtifact,
    access: VerifiedAccessAudit,
    selected_threshold: float,
    candidate_rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    normalized = [
        {
            **row,
            "train_sample_ids": list(closure.train_sample_ids),
            "train_root_source_ids": list(
                closure.train_root_source_ids
            ),
            "input_train_cache_fingerprint": train_cache.file_sha256,
            "access_audit_receipt_fingerprint": (
                access.receipt_fingerprint
            ),
        }
        for row in candidate_rows
    ]
    ledger_fp = protocol_fingerprint(normalized)
    return {
        "selection_root_source_ids": list(
            closure.train_root_source_ids
        ),
        "selection_sample_ids": list(closure.train_sample_ids),
        "evaluation_root_source_ids": list(
            closure.held_out_root_source_ids
        ),
        "evaluation_sample_ids": list(closure.held_out_sample_ids),
        "holdout_labels_used_for_selection": False,
        "complete_51_point_grid_evaluated": True,
        "D_V_threshold_reused": False,
        "grid_source_repo_path": _GRID_SOURCE,
        "grid_source_file_sha256": _GRID_SOURCE_SHA256,
        "threshold_grid": list(BASE_B_THRESHOLD_GRID),
        "candidate_rows": normalized,
        "candidate_ledger_fingerprint": ledger_fp,
        "selector_policy": list(_BASE_B_SELECTOR_POLICY),
        "selector_policy_fingerprint": protocol_fingerprint(
            list(_BASE_B_SELECTOR_POLICY)
        ),
        "selected_threshold": selected_threshold,
        "input_train_cache_fingerprint": train_cache.file_sha256,
        "access_audit_receipt_fingerprint": access.receipt_fingerprint,
    }


def _factual_rows(
    *,
    closure: VerifiedOOFFoldClosure,
    dataset: OOFEvaluationDataset,
    ledgers: Mapping[str, OOFEvaluationLedger],
    evaluation_fingerprints: Mapping[str, str],
    evaluator: OOFConcreteEvaluator,
) -> list[dict[str, object]]:
    sample_by_id = {row.sample_id: row for row in dataset.rows}
    rows = []
    for arm in OOF_ARMS:
        ledger = ledgers[arm]
        for row in ledger.per_sample_rows:
            sample = sample_by_id[str(row["sample_id"])]
            valid = sample.valid_mask
            anchor = (sample.base_probability >= BASE_A_THRESHOLD) & valid
            rows.append({
                "split": "D_R",
                "evidence_role": "factual_only",
                "fold_id": closure.fold_id,
                "arm": arm,
                "sample_id": sample.sample_id,
                "root_source_id": sample.root_source_id,
                "gt_fingerprint": tensor_content_fingerprint(
                    sample.gt_mask & valid
                ),
                "anchor_state_fingerprint": (
                    tensor_content_fingerprint(anchor)
                ),
                "evaluation_contract_fingerprint": (
                    evaluator.evaluator_fingerprint
                ),
                "terminal_artifact_fingerprint": (
                    evaluation_fingerprints[arm]
                ),
                "sufficient_statistics": dict(row["statistics"]),
            })
    return rows


@dataclass(frozen=True)
class OOFFoldRuntimeResult:
    fold_id: int
    fold_receipt: dict[str, object]
    access_receipt: dict[str, object]
    factual_rows: tuple[dict[str, object], ...]
    candidate_field_rows: tuple[dict[str, object], ...]
    candidate_prediction_rows: tuple[dict[str, object], ...]
    artifact_paths: tuple[str, ...]


@dataclass(frozen=True)
class OOF4RuntimeResult:
    decision: VerifiedOOFDecision
    result_payload: dict[str, object]
    result_path: str


def _run_fold(
    *,
    execution_authorization: VerifiedOOFExecutionAuthorization,
    verified_split: VerifiedOOF4Split,
    closure: VerifiedOOFFoldClosure,
    source_binding: CoverageStateRealDRSourceBinding,
    protocol: CoverageStateObservabilityProtocol,
    geometry_protocol: GeometryCatalogProtocol,
    preprocess: PreprocessConfig,
    candidate_config: CoverageStateGCRPACREConfig,
    evaluator: OOFConcreteEvaluator,
    device: torch.device | str,
) -> tuple[OOFFoldRuntimeResult, object]:
    authorization = require_verified_oof_execution_authorization(
        execution_authorization
    )
    fold_directory = (
        Path(authorization.runtime_root) / f"fold_{closure.fold_id}"
    )
    train = build_oof_restricted_train_inputs(
        source_binding=source_binding,
        protocol=protocol,
        geometry_protocol=geometry_protocol,
        preprocess=preprocess,
        fold_closure=closure,
    )
    train_base = save_oof_cache_artifact_new(
        train.evaluation_dataset,
        fold_directory / "train/base_eval/cache.pt",
        fold_closure=closure,
        partition="train",
        arm="base_eval",
        creation_event=OOF_EVENT_TRAIN_CACHE_CREATED,
    )
    train_control = save_oof_cache_artifact_new(
        train.scalar_cache,
        fold_directory / "train/v23_control/cache.pt",
        fold_closure=closure,
        partition="train",
        arm=OOF_CONTROL_ARM,
        creation_event=OOF_EVENT_TRAIN_CACHE_CREATED,
    )
    train_candidate = save_oof_cache_artifact_new(
        train.scalar_cache,
        fold_directory / "train/candidate/cache.pt",
        fold_closure=closure,
        partition="train",
        arm=OOF_CANDIDATE_ARM,
        creation_event=OOF_EVENT_TRAIN_CACHE_CREATED,
    )
    schedule = build_coverage_state_training_schedule(
        train.scalar_cache,
        CoverageStateScheduleConfig(
            seed=OOF_SEED,
            epochs=OOF_EPOCHS,
            steps_per_epoch=OOF_STEPS_PER_EPOCH,
        ),
    )
    schedule_path = atomic_write_new_canonical_json(
        fold_directory / "schedule.json",
        schedule.canonical_payload(),
    )
    run_start = create_oof_training_run_start_new(
        authorization,
        closure,
        schedule=schedule,
        control_cache_artifact=train_control,
        candidate_cache_artifact=train_candidate,
    )

    base_train_reader = issue_oof_cache_reader(
        train_base,
        reader_id="BaseB_train_fold_selector",
    )
    base_train_dataset = load_oof_cache_payload(base_train_reader)
    if type(base_train_dataset) is not OOFEvaluationDataset:
        raise TypeError("BaseB reader did not return an evaluation dataset")
    selected_threshold, base_rows = evaluator.select_base_b_train_only(
        base_train_dataset
    )

    control_reader = issue_oof_cache_reader(
        train_control,
        reader_id="PACRE_VC_v23_control_train_runner",
    )
    candidate_reader = issue_oof_cache_reader(
        train_candidate,
        reader_id="GCR_PACRE_v24_train_runner",
    )
    training = run_paired_oof_training_400(
        fold_closure=closure,
        run_start_token=run_start,
        control_cache_reader=control_reader,
        candidate_cache_reader=candidate_reader,
        schedule=schedule,
        candidate_config=candidate_config,
        device=device,
    )
    training.verify_unchanged()
    seal = seal_oof_training_terminals(
        closure,
        completed_400_capabilities=(
            training.completed_400_capabilities
        ),
    )
    # Evaluation evidence is produced from the same persisted CPU terminal
    # representation that the independent finalizer will load.  This avoids
    # trusting a still-live training module and removes CPU/GPU numerical
    # drift from the byte-exact ledger replay contract.
    replay_control_model = load_oof_terminal_model_strict(
        training.control_terminal_artifact,
        arm=OOF_CONTROL_ARM,
        expected_path=(
            fold_directory
            / "terminal"
            / "v23_control_terminal.safetensors"
        ),
    )
    replay_candidate_model = load_oof_terminal_model_strict(
        training.candidate_terminal_artifact,
        arm=OOF_CANDIDATE_ARM,
        expected_path=(
            fold_directory
            / "terminal"
            / "candidate_terminal.safetensors"
        ),
    )

    holdout = build_oof_restricted_holdout_inputs(
        source_binding=source_binding,
        protocol=protocol,
        preprocess=preprocess,
        fold_closure=closure,
        terminal_seal=seal,
    )
    holdout_base = save_oof_cache_artifact_new(
        holdout.evaluation_dataset,
        fold_directory / "holdout/base_eval/cache.pt",
        fold_closure=closure,
        partition="holdout",
        arm="base_eval",
        creation_event=OOF_EVENT_HOLDOUT_CACHE_CREATED,
        terminal_seal=seal,
    )
    holdout_control = save_oof_cache_artifact_new(
        holdout.evaluation_dataset,
        fold_directory / "holdout/v23_control/cache.pt",
        fold_closure=closure,
        partition="holdout",
        arm=OOF_CONTROL_ARM,
        creation_event=OOF_EVENT_HOLDOUT_CACHE_CREATED,
        terminal_seal=seal,
    )
    holdout_candidate = save_oof_cache_artifact_new(
        holdout.evaluation_dataset,
        fold_directory / "holdout/candidate/cache.pt",
        fold_closure=closure,
        partition="holdout",
        arm=OOF_CANDIDATE_ARM,
        creation_event=OOF_EVENT_HOLDOUT_CACHE_CREATED,
        terminal_seal=seal,
    )
    caches = (
        train_base,
        train_control,
        train_candidate,
        holdout_base,
        holdout_control,
        holdout_candidate,
    )
    cache_set = verify_oof_six_cache_independence(caches)
    access_receipt = _access_receipt(
        fold_id=closure.fold_id,
        source_binding_fingerprint=source_binding.binding_fingerprint,
        cache_artifacts=caches,
    )
    access_path = atomic_write_new_canonical_json(
        fold_directory / "access_receipt.json",
        access_receipt,
    )
    access = verify_access_audit_receipt(
        access_receipt,
        expected_stage_id=f"oof4_fold_{closure.fold_id}",
        allowed_splits=["D_R"],
    )

    base_holdout_reader = issue_oof_cache_reader(
        holdout_base,
        reader_id="OOF4_read_only_holdout_evaluator",
        terminal_seal=seal,
    )
    control_holdout_reader = issue_oof_cache_reader(
        holdout_control,
        reader_id="OOF4_read_only_holdout_evaluator",
        terminal_seal=seal,
    )
    candidate_holdout_reader = issue_oof_cache_reader(
        holdout_candidate,
        reader_id="OOF4_read_only_holdout_evaluator",
        terminal_seal=seal,
    )
    base_dataset = load_oof_cache_payload(base_holdout_reader)
    control_dataset = load_oof_cache_payload(control_holdout_reader)
    candidate_dataset = load_oof_cache_payload(candidate_holdout_reader)
    if (
        type(base_dataset) is not OOFEvaluationDataset
        or type(control_dataset) is not OOFEvaluationDataset
        or type(candidate_dataset) is not OOFEvaluationDataset
        or base_dataset.dataset_fingerprint
        != control_dataset.dataset_fingerprint
        or base_dataset.dataset_fingerprint
        != candidate_dataset.dataset_fingerprint
    ):
        raise PermissionError("OOF holdout evaluator datasets differ")
    ledgers = {
        OOF_BASE_A_ARM: evaluator.evaluate_base(
            base_dataset,
            threshold=BASE_A_THRESHOLD,
            arm=OOF_BASE_A_ARM,
        ),
        OOF_BASE_B_ARM: evaluator.evaluate_base(
            base_dataset,
            threshold=selected_threshold,
            arm=OOF_BASE_B_ARM,
        ),
        OOF_V23_ARM: evaluator.evaluate_model(
            control_dataset,
            replay_control_model,
            arm=OOF_V23_ARM,
            device="cpu",
        ),
        OOF_V24_ARM: evaluator.evaluate_model(
            candidate_dataset,
            replay_candidate_model,
            arm=OOF_V24_ARM,
            device="cpu",
        ),
        OOF_G1_ARM: evaluator.evaluate_model(
            candidate_dataset,
            replay_candidate_model,
            arm=OOF_G1_ARM,
            forced_unit_gate=True,
            device="cpu",
        ),
    }
    evaluation_artifacts = {
        arm: _save_evaluation_ledger_new(
            fold_directory / "evaluation" / f"{arm}.json",
            ledger,
        )
        for arm, ledger in ledgers.items()
    }
    control_cap = training.control_capability
    candidate_cap = training.candidate_capability
    evaluation_fingerprints = {
        OOF_BASE_A_ARM: protocol_fingerprint(
            {"arm": OOF_BASE_A_ARM, "threshold": BASE_A_THRESHOLD}
        ),
        OOF_BASE_B_ARM: protocol_fingerprint(
            {
                "arm": OOF_BASE_B_ARM,
                "candidate_ledger_fingerprint": (
                    protocol_fingerprint([
                        {
                            **row,
                            "train_sample_ids": list(
                                closure.train_sample_ids
                            ),
                            "train_root_source_ids": list(
                                closure.train_root_source_ids
                            ),
                            "input_train_cache_fingerprint": (
                                train_base.file_sha256
                            ),
                            "access_audit_receipt_fingerprint": (
                                access.receipt_fingerprint
                            ),
                        }
                        for row in base_rows
                    ])
                ),
                "selected_threshold": selected_threshold,
            }
        ),
        OOF_V23_ARM: control_cap.terminal_model_fingerprint,
        OOF_V24_ARM: candidate_cap.terminal_model_fingerprint,
        OOF_G1_ARM: candidate_cap.terminal_model_fingerprint,
    }
    source_hashes = gcr_pacre_v24_source_closure_hashes()
    training_arms = _training_arm_receipts(
        closure=closure,
        training=training,
        source_hashes=source_hashes,
        device=device,
    )
    base_selection = _base_b_selection_receipt(
        closure=closure,
        train_cache=train_base,
        access=access,
        selected_threshold=selected_threshold,
        candidate_rows=base_rows,
    )
    factual = _factual_rows(
        closure=closure,
        dataset=base_dataset,
        ledgers=ledgers,
        evaluation_fingerprints=evaluation_fingerprints,
        evaluator=evaluator,
    )
    factual_body = {
        "schema_version": (
            "cure-lite-v24-gcr-pacre-oof4-factual-rows-artifact-v1"
        ),
        "fold_id": closure.fold_id,
        "closure_fingerprint": closure.closure_fingerprint,
        "rows": factual,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    factual_payload = _seal(
        factual_body,
        field="ledger_fingerprint",
    )
    factual_path = atomic_write_new_canonical_json(
        fold_directory / "factual_rows.json",
        factual_payload,
    )
    factual_artifact = {
        **regular_file_receipt(factual_path),
        "ledger_fingerprint": factual_payload["ledger_fingerprint"],
        "payload": factual_payload,
    }
    fold_body = {
        "schema_version": OOF_FOLD_RUNTIME_RECEIPT_SCHEMA,
        "split_preregistration_fingerprint": (
            verified_split.receipt_fingerprint
        ),
        "root_by_sample_fingerprint": (
            verified_split.root_by_sample_fingerprint
        ),
        "plan_fingerprint": verified_split.plan_fingerprint,
        "fold_id": closure.fold_id,
        "train_root_source_ids": list(closure.train_root_source_ids),
        "held_out_root_source_ids": list(
            closure.held_out_root_source_ids
        ),
        "train_sample_ids": list(closure.train_sample_ids),
        "held_out_sample_ids": list(closure.held_out_sample_ids),
        "access_audit_receipt_fingerprint": access.receipt_fingerprint,
        "events": {
            "train_cache_materialized": OOF_EVENT_TRAIN_CACHE_CREATED,
            "training_claimed": OOF_EVENT_TRAINING_RUN_START,
            "training_terminals_sealed": OOF_EVENT_TERMINALS_SEALED,
            "holdout_cache_materialized": (
                OOF_EVENT_HOLDOUT_CACHE_CREATED
            ),
            "holdout_cache_first_open": OOF_EVENT_HOLDOUT_FIRST_OPEN,
        },
        "run_start_artifact": {
            **run_start_artifact_receipt(run_start),
            "payload": run_start.payload,
        },
        "terminal_seal": _terminal_seal_payload(seal, training),
        "cache_set_fingerprint": cache_set.set_fingerprint,
        "cache_entries": list(cache_set.protocol_entries),
        "training_arms": training_arms,
        "BaseB_train_fold_selection": base_selection,
        "evaluation_artifact_fingerprints": evaluation_fingerprints,
        "evaluation_ledger_artifacts": evaluation_artifacts,
        "factual_rows_artifact": factual_artifact,
        "held_out_prediction_role": "factual_only",
        "source_closure": {
            "schema_version": (
                "cure-lite-v24-gcr-pacre-unified-source-closure-v1"
            ),
            "source_hashes": dict(source_hashes),
            "source_closure_fingerprint": (
                gcr_pacre_v24_source_closure_fingerprint(source_hashes)
            ),
        },
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    fold_receipt = _seal(fold_body, field="receipt_fingerprint")
    fold_path = atomic_write_new_canonical_json(
        fold_directory / "fold_receipt.json",
        fold_receipt,
    )
    verified_fold = validate_oof_fold_execution_receipt(
        fold_receipt,
        verified_split,
        access_audit=access,
        execution_authorization=authorization,
        repository_root=Path(__file__).resolve().parents[1],
    )
    pool = pool_factual_only_rows(
        factual,
        verified_fold,
        access_audit=access,
    )
    candidate_rows = {
        row["sample_id"]: row
        for row in ledgers[OOF_V24_ARM].per_sample_rows
    }
    g1_rows = {
        row["sample_id"]: row
        for row in ledgers[OOF_G1_ARM].per_sample_rows
    }
    field_differences = tuple(
        {
            "fold_id": closure.fold_id,
            "sample_id": sample_id,
            "natural_output_fingerprint": (
                candidate_rows[sample_id]["field_fingerprint"]
            ),
            "forced_G1_output_fingerprint": (
                g1_rows[sample_id]["field_fingerprint"]
            ),
        }
        for sample_id in closure.held_out_sample_ids
        if candidate_rows[sample_id]["field_fingerprint"]
        != g1_rows[sample_id]["field_fingerprint"]
    )
    prediction_differences = tuple(
        {
            "fold_id": closure.fold_id,
            "sample_id": sample_id,
            "natural_output_fingerprint": (
                candidate_rows[sample_id]["prediction_fingerprint"]
            ),
            "forced_G1_output_fingerprint": (
                g1_rows[sample_id]["prediction_fingerprint"]
            ),
        }
        for sample_id in closure.held_out_sample_ids
        if candidate_rows[sample_id]["prediction_fingerprint"]
        != g1_rows[sample_id]["prediction_fingerprint"]
    )
    artifacts = (
        str(schedule_path),
        str(access_path),
        str(factual_path),
        str(fold_path),
        *(
            str(value["path"]) for value in evaluation_artifacts.values()
        ),
    )
    return (
        OOFFoldRuntimeResult(
            fold_id=closure.fold_id,
            fold_receipt=fold_receipt,
            access_receipt=access_receipt,
            factual_rows=tuple(factual),
            candidate_field_rows=field_differences,
            candidate_prediction_rows=prediction_differences,
            artifact_paths=tuple(artifacts),
        ),
        pool,
    )


def _run_real_oof4_in_process_disabled(
    *,
    execution_authorization: VerifiedOOFExecutionAuthorization,
    verified_split: VerifiedOOF4Split,
    source_binding: CoverageStateRealDRSourceBinding,
    protocol: CoverageStateObservabilityProtocol,
    geometry_protocol: GeometryCatalogProtocol,
    preprocess: PreprocessConfig,
    candidate_config: CoverageStateGCRPACREConfig,
    available_sample_ids: tuple[str, ...],
    device: torch.device | str = "cpu",
) -> OOF4RuntimeResult:
    """Disabled: real execution must use four independent run-fold processes."""

    raise PermissionError(
        "one-process four-fold OOF execution is forbidden; use the fixed "
        "authorize -> 4x run-fold -> finalize process sequence"
    )

    authorization = require_verified_oof_execution_authorization(
        execution_authorization
    )
    if (
        type(candidate_config) is not CoverageStateGCRPACREConfig
        or type(protocol) is not CoverageStateObservabilityProtocol
        or type(geometry_protocol) is not GeometryCatalogProtocol
        or type(preprocess) is not PreprocessConfig
    ):
        raise TypeError("OOF real runner accepts only fixed built-in types")
    if (
        candidate_config.feature_channels,
        candidate_config.feature_stride,
        candidate_config.width,
        candidate_config.expected_parameter_count,
    ) != (64, 4, 32, 64_064):
        raise ValueError("OOF candidate config must be frozen 64/4/32/64064")
    closures = verify_all_oof_fold_closures(
        verified_split,
        available_sample_ids=available_sample_ids,
    )
    evaluator = OOFConcreteEvaluator.fixed()
    fold_results = []
    pools = []
    for closure in closures:
        result, pool = _run_fold(
            execution_authorization=authorization,
            verified_split=verified_split,
            closure=closure,
            source_binding=source_binding,
            protocol=protocol,
            geometry_protocol=geometry_protocol,
            preprocess=preprocess,
            candidate_config=candidate_config,
            evaluator=evaluator,
            device=device,
        )
        fold_results.append(result)
        pools.append(pool)
    pooled = combine_oof4_factual_pools(pools, verified_split)
    field_rows = [
        row
        for fold in fold_results
        for row in fold.candidate_field_rows
    ]
    prediction_rows = [
        row
        for fold in fold_results
        for row in fold.candidate_prediction_rows
    ]
    pooled_payload = pooled.payload
    artifacts_by_fold = pooled_payload[
        "evaluation_artifact_fingerprints_by_fold"
    ]
    v24_fingerprints = [
        artifacts_by_fold[str(fold_id)][OOF_V24_ARM]
        for fold_id in range(4)
    ]
    gate_body = {
        "schema_version": "cure-lite-v24-gcr-pacre-forced-G1-path-v2",
        "pooled_evidence_fingerprint": pooled.evidence_fingerprint,
        "sample_roots_fingerprint": pooled_payload[
            "sample_roots_fingerprint"
        ],
        "v24_terminal_artifact_fingerprints": v24_fingerprints,
        "forced_G1_terminal_artifact_fingerprints": v24_fingerprints,
        "forced_G1_retrained": False,
        "field_difference_count": len(field_rows),
        "prediction_difference_count": len(prediction_rows),
        "field_difference_ledger": field_rows,
        "field_difference_ledger_fingerprint": protocol_fingerprint(
            field_rows
        ),
        "prediction_difference_ledger": prediction_rows,
        "prediction_difference_ledger_fingerprint": protocol_fingerprint(
            prediction_rows
        ),
        "access_audit_receipt_fingerprints": list(
            pooled.access_audit_receipt_fingerprints
        ),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    gate_receipt = _seal(gate_body, field="receipt_fingerprint")
    gate = verify_gate_path_receipt(gate_receipt, pooled)
    decision = decide_oof4_pooled(
        pooled,
        gate_path_evidence=gate,
    )
    result_body = {
        "schema_version": OOF_REAL_RESULT_SCHEMA,
        "execution_authorization_fingerprint": (
            authorization.authorization_fingerprint
        ),
        "split_receipt_fingerprint": (
            verified_split.receipt_fingerprint
        ),
        "fold_access_receipts": [
            value.access_receipt for value in fold_results
        ],
        "fold_receipts": [
            value.fold_receipt for value in fold_results
        ],
        "fold_factual_rows": [
            list(value.factual_rows) for value in fold_results
        ],
        "gate_path_receipt": gate_receipt,
        "pooled_evidence": pooled.payload,
        "decision": decision.payload,
        "source_closure_fingerprint": (
            gcr_pacre_v24_source_closure_fingerprint()
        ),
        "fixed_relative_uplift_threshold": None,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    result_payload = _seal(
        result_body,
        field="result_fingerprint",
    )
    result_path = atomic_write_new_canonical_json(
        Path(authorization.runtime_root) / "result.json",
        result_payload,
    )
    return OOF4RuntimeResult(
        decision=decision,
        result_payload=result_payload,
        result_path=str(result_path),
    )


def run_real_oof4_fold(
    *,
    fold_id: int,
    execution_authorization: VerifiedOOFExecutionAuthorization,
    verified_split: VerifiedOOF4Split,
    source_binding: CoverageStateRealDRSourceBinding,
    protocol: CoverageStateObservabilityProtocol,
    geometry_protocol: GeometryCatalogProtocol,
    preprocess: PreprocessConfig,
    available_sample_ids: tuple[str, ...],
    device: torch.device | str = "cpu",
) -> OOFFoldRuntimeResult:
    """Run exactly one pre-registered fold into its fixed create-only path."""

    if isinstance(fold_id, bool) or fold_id not in range(4):
        raise ValueError("OOF fold_id must be one of 0,1,2,3")
    resolved_device = preflight_oof_execution_device(device)
    authorization = require_verified_oof_execution_authorization(
        execution_authorization
    )
    closures = verify_all_oof_fold_closures(
        verified_split,
        available_sample_ids=available_sample_ids,
    )
    result, _ = _run_fold(
        execution_authorization=authorization,
        verified_split=verified_split,
        closure=closures[fold_id],
        source_binding=source_binding,
        protocol=protocol,
        geometry_protocol=geometry_protocol,
        preprocess=preprocess,
        candidate_config=CoverageStateGCRPACREConfig(
            feature_channels=64,
            feature_stride=4,
            width=32,
        ),
        evaluator=OOFConcreteEvaluator.fixed(),
        device=resolved_device,
    )
    return result


def finalize_real_oof4(
    *,
    verified_split: VerifiedOOF4Split,
    execution_authorization: VerifiedOOFExecutionAuthorization,
) -> OOF4RuntimeResult:
    """Finalize the four fixed persisted folds without reopening D_R payloads."""

    authorization = require_verified_oof_execution_authorization(
        execution_authorization
    )
    fold_access_receipts: list[dict[str, object]] = []
    fold_receipts: list[dict[str, object]] = []
    fold_factual_rows: list[list[dict[str, object]]] = []
    pools = []
    for fold_id in range(4):
        fold_directory = (
            Path(authorization.runtime_root) / f"fold_{fold_id}"
        )
        access_receipt = read_canonical_json(
            fold_directory / "access_receipt.json"
        )
        fold_receipt = read_canonical_json(
            fold_directory / "fold_receipt.json"
        )
        factual_payload = read_canonical_json(
            fold_directory / "factual_rows.json"
        )
        factual = factual_payload.get("rows")
        if not isinstance(factual, list):
            raise TypeError("OOF factual artifact rows must be a list")
        access = verify_access_audit_receipt(
            access_receipt,
            expected_stage_id=f"oof4_fold_{fold_id}",
            allowed_splits=["D_R"],
        )
        fold = validate_oof_fold_execution_receipt(
            fold_receipt,
            verified_split,
            access_audit=access,
            execution_authorization=authorization,
            repository_root=Path(__file__).resolve().parents[1],
        )
        pools.append(
            pool_factual_only_rows(
                factual,
                fold,
                access_audit=access,
            )
        )
        fold_access_receipts.append(access_receipt)
        fold_receipts.append(fold_receipt)
        fold_factual_rows.append([dict(row) for row in factual])
    pooled = combine_oof4_factual_pools(pools, verified_split)
    artifacts_by_fold = pooled.payload[
        "evaluation_artifact_fingerprints_by_fold"
    ]
    v24_fingerprints = [
        artifacts_by_fold[str(fold_id)][OOF_V24_ARM]
        for fold_id in range(4)
    ]
    field_rows = list(
        pooled.payload["verified_field_difference_ledger"]
    )
    prediction_rows = list(
        pooled.payload["verified_prediction_difference_ledger"]
    )
    gate_body = {
        "schema_version": "cure-lite-v24-gcr-pacre-forced-G1-path-v2",
        "pooled_evidence_fingerprint": pooled.evidence_fingerprint,
        "sample_roots_fingerprint": pooled.payload[
            "sample_roots_fingerprint"
        ],
        "v24_terminal_artifact_fingerprints": v24_fingerprints,
        "forced_G1_terminal_artifact_fingerprints": v24_fingerprints,
        "forced_G1_retrained": False,
        "field_difference_count": len(field_rows),
        "prediction_difference_count": len(prediction_rows),
        "field_difference_ledger": field_rows,
        "field_difference_ledger_fingerprint": protocol_fingerprint(
            field_rows
        ),
        "prediction_difference_ledger": prediction_rows,
        "prediction_difference_ledger_fingerprint": protocol_fingerprint(
            prediction_rows
        ),
        "access_audit_receipt_fingerprints": list(
            pooled.access_audit_receipt_fingerprints
        ),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    gate_receipt = _seal(gate_body, field="receipt_fingerprint")
    gate = verify_gate_path_receipt(gate_receipt, pooled)
    decision = decide_oof4_pooled(
        pooled,
        gate_path_evidence=gate,
    )
    result_body = {
        "schema_version": OOF_REAL_RESULT_SCHEMA,
        "execution_authorization_fingerprint": (
            authorization.authorization_fingerprint
        ),
        "split_receipt_fingerprint": verified_split.receipt_fingerprint,
        "fold_access_receipts": fold_access_receipts,
        "fold_receipts": fold_receipts,
        "fold_factual_rows": fold_factual_rows,
        "gate_path_receipt": gate_receipt,
        "pooled_evidence": pooled.payload,
        "decision": decision.payload,
        "source_closure_fingerprint": (
            gcr_pacre_v24_source_closure_fingerprint()
        ),
        "fixed_relative_uplift_threshold": None,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    result_payload = _seal(result_body, field="result_fingerprint")
    result_path = atomic_write_new_canonical_json(
        Path(authorization.runtime_root) / "result.json",
        result_payload,
    )
    return OOF4RuntimeResult(
        decision=decision,
        result_payload=result_payload,
        result_path=str(result_path),
    )


def verify_real_oof4_result_artifact(
    *,
    verified_split: VerifiedOOF4Split,
    execution_authorization: VerifiedOOFExecutionAuthorization,
) -> VerifiedOOFDecision:
    """Rebuild the OOF decision token from the fixed persisted evidence chain."""

    authorization = require_verified_oof_execution_authorization(
        execution_authorization
    )
    if authorization.split_receipt_fingerprint != (
        verified_split.receipt_fingerprint
    ):
        raise PermissionError("OOF result authorization/split changed")
    result_path = Path(authorization.runtime_root) / "result.json"
    result = read_canonical_json(result_path)
    if result_path.stat().st_mode & 0o777 != 0o444:
        raise PermissionError("OOF result artifact must be immutable mode 0444")
    expected_fields = {
        "schema_version",
        "execution_authorization_fingerprint",
        "split_receipt_fingerprint",
        "fold_access_receipts",
        "fold_receipts",
        "fold_factual_rows",
        "gate_path_receipt",
        "pooled_evidence",
        "decision",
        "source_closure_fingerprint",
        "fixed_relative_uplift_threshold",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "result_fingerprint",
    }
    if set(result) != expected_fields:
        raise ValueError("OOF result artifact fields changed")
    body = dict(result)
    result_fingerprint = body.pop("result_fingerprint", None)
    if (
        result.get("schema_version") != OOF_REAL_RESULT_SCHEMA
        or result_fingerprint != protocol_fingerprint(body)
        or result.get("execution_authorization_fingerprint")
        != authorization.authorization_fingerprint
        or result.get("split_receipt_fingerprint")
        != verified_split.receipt_fingerprint
        or result.get("source_closure_fingerprint")
        != gcr_pacre_v24_source_closure_fingerprint()
        or result.get("fixed_relative_uplift_threshold") is not None
        or result.get("D_V_payload_accessed") is not False
        or result.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError("OOF persisted result identity changed")
    raw_access = result.get("fold_access_receipts")
    raw_folds = result.get("fold_receipts")
    raw_rows = result.get("fold_factual_rows")
    if (
        not isinstance(raw_access, list)
        or not isinstance(raw_folds, list)
        or not isinstance(raw_rows, list)
        or len(raw_access) != 4
        or len(raw_folds) != 4
        or len(raw_rows) != 4
    ):
        raise ValueError("OOF result requires exactly four fold evidence sets")
    pools = []
    for fold_id in range(4):
        access_receipt = dict(raw_access[fold_id])
        fold_receipt = dict(raw_folds[fold_id])
        factual_rows = raw_rows[fold_id]
        if not isinstance(factual_rows, list):
            raise TypeError("OOF persisted factual rows must be a list")
        fold_directory = (
            Path(authorization.runtime_root) / f"fold_{fold_id}"
        )
        persisted_access = read_canonical_json(
            fold_directory / "access_receipt.json"
        )
        persisted_fold = read_canonical_json(
            fold_directory / "fold_receipt.json"
        )
        if (
            persisted_access != access_receipt
            or persisted_fold != fold_receipt
            or persisted_access.get("source_manifest_fingerprint")
            != authorization.source_binding_fingerprint
        ):
            raise RuntimeError("OOF persisted fold/access evidence changed")
        access = verify_access_audit_receipt(
            access_receipt,
            expected_stage_id=f"oof4_fold_{fold_id}",
            allowed_splits=["D_R"],
        )
        fold = validate_oof_fold_execution_receipt(
            fold_receipt,
            verified_split,
            access_audit=access,
            execution_authorization=authorization,
            repository_root=Path(__file__).resolve().parents[1],
        )
        if fold.fold_id != fold_id:
            raise ValueError("OOF persisted fold ordering changed")
        pools.append(
            pool_factual_only_rows(
                factual_rows,
                fold,
                access_audit=access,
            )
        )
    pooled = combine_oof4_factual_pools(pools, verified_split)
    if result.get("pooled_evidence") != pooled.payload:
        raise ValueError("OOF persisted pooled evidence changed")
    gate_receipt = result.get("gate_path_receipt")
    if not isinstance(gate_receipt, Mapping):
        raise TypeError("OOF persisted gate receipt must be a mapping")
    gate = verify_gate_path_receipt(gate_receipt, pooled)
    decision = decide_oof4_pooled(
        pooled,
        gate_path_evidence=gate,
    )
    if result.get("decision") != decision.payload:
        raise ValueError("OOF persisted decision differs from recomputation")
    return decision


__all__ = [
    "OOF4RuntimeResult",
    "OOFFoldRuntimeResult",
    "OOF_FOLD_RUNTIME_RECEIPT_SCHEMA",
    "OOF_REAL_RESULT_SCHEMA",
    "finalize_real_oof4",
    "preflight_oof_execution_device",
    "run_real_oof4_fold",
    "verify_real_oof4_result_artifact",
]
