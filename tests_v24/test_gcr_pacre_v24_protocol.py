from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import struct
from typing import Callable

import pytest
import torch

from cure_lite.experiment.coverage_state_training import (
    coverage_state_model_contract_payload,
    coverage_state_model_fingerprint,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    bind_coverage_state_real_dr_sources,
)
from cure_lite.coverage_state_observability import (
    CoverageStateObservabilityDecision,
    audit_population_observability,
)
from cure_lite.coverage_state_precomputed_cache import (
    prepare_scalar_coverage_state_cache,
)
from cure_lite.coverage_state_raw_catalog import (
    make_coverage_state_raw_catalog,
)
from cure_lite.coverage_state_sobolev import CoverageStateSobolevConfig
from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
)
from cure_lite.paired_types import tensor_content_fingerprint
from cure_lite_v24.formal_cache_artifacts import (
    FORMAL_CACHE_PAYLOAD_SCHEMA,
    VerifiedFormalCacheArtifact,
    build_formal_cache_neutral_envelope,
    require_verified_formal_cache_artifact,
    require_verified_formal_cache_origin_artifact,
    save_formal_cache_neutral_artifact_new,
    verify_formal_cache_artifact,
    verify_formal_cache_pair_independence,
)
from cure_lite_v24.artifact_io import save_terminal_safetensors_new
from cure_lite_v24.bounded_runner import (
    GCR_PACRE_CANDIDATE_ARM,
    GCR_PACRE_CONTROL_ARM,
)
import cure_lite_v24.formal_cache_artifacts as formal_cache_artifacts
import cure_lite_v24.oof_evaluation as oof_evaluation_module
import cure_lite_v24.oof_run_start as oof_run_start
from cure_lite_v24.oof_run_start import (
    authorize_real_oof4_execution_new,
)
from cure_lite_v24.oof_evaluation import OOFConcreteEvaluator
from cure_lite_v24.source_closure import (
    GCR_PACRE_V24_SOURCE_CLOSURE_PATHS,
    GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
)
from cure_lite_v23.factory import build_pacre_vc_training_model
from cure_lite_v23.pacre_vc import (
    CoverageStatePACREVerifierCorrectedConfig,
)
from cure_lite_v24.factory import (
    GCR_PACRE_PARAMETER_NAMES,
    build_formal_gcr_pacre_training_model,
)
from cure_lite_v24.terminal_evidence import (
    mechanically_recompute_bounded_arm,
    mechanically_recompute_formal_terminal,
)
from cure_lite_v24.training_trace import (
    build_training_trace_payload,
    save_training_trace_new,
    trace_finite_audit,
    verify_training_trace_artifact,
)
import tools.gcr_pacre_v24_protocol as protocol_module
from tools.gcr_pacre_v24_protocol import (
    BASE_A_THRESHOLD,
    BASE_B_THRESHOLD_GRID,
    BOUNDED_UPDATES,
    FORMAL_UPDATES,
    OOF_ARMS,
    VerifiedOOF4Split,
    canonical_json,
    combine_oof4_factual_pools,
    decide_oof4_pooled,
    decide_paired_bounded400,
    decide_relative_dv_gate,
    derive_root_source_ids,
    deterministic_oof4_plan,
    load_exact_baseline_envelope,
    pool_factual_only_rows,
    require_verified_access_audit,
    require_verified_bounded_decision,
    require_verified_oof_decision,
    stable_fingerprint,
    validate_d_t_preregistration,
    validate_d_t_seed42_model_binding,
    validate_formal_training_receipt,
    validate_oof_fold_execution_receipt,
    validate_paired_bounded_receipt,
    validate_protocol_artifact_manifest,
    verify_access_audit_receipt,
    verify_dv_candidate_evidence,
    verify_formal800_training_independence,
    verify_gate_path_receipt,
    verify_oof4_split_preregistration,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_training_scalar_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ROOT = REPO_ROOT / "protocols/IRSTD-1K/gcr_pacre_v24"
SHA_A = "a" * 64
SHA_B = "b" * 64


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _fp(label: str) -> str:
    return stable_fingerprint({"label": label})


def _seal(
    body: dict[str, object],
    field: str = "receipt_fingerprint",
) -> dict[str, object]:
    return {**body, field: stable_fingerprint(body)}


def _reseal(
    payload: dict[str, object],
    field: str = "receipt_fingerprint",
) -> dict[str, object]:
    body = deepcopy(payload)
    body.pop(field, None)
    return _seal(body, field)


def _access_receipt(
    stage: str,
    allowed: list[str],
    observed: list[dict[str, object]],
) -> dict[str, object]:
    body = {
        "schema_version": "cure-lite-v24-split-access-audit-v1",
        "stage_id": stage,
        "allowed_splits": allowed,
        "observed_payloads": observed,
        "source_manifest_fingerprint": _fp(f"{stage}-manifest"),
        "event_log_fingerprint": stable_fingerprint(observed),
        "D_V_payload_accessed": "D_V" in allowed,
        "D_T_payload_accessed": "D_T" in allowed,
    }
    return _seal(body)


def _file_meta(path: Path, model_fingerprint: str) -> dict[str, object]:
    stat_result = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat_result.st_size,
        "file_sha256": _sha(path),
        "model_fingerprint": model_fingerprint,
    }


def _write_immutable_json(
    path: Path,
    payload: dict[str, object],
) -> dict[str, object]:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    path.chmod(0o444)
    stat_result = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat_result.st_size,
        "file_sha256": _sha(path),
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "hardlink_count": stat_result.st_nlink,
    }


def _oof_cache_entry(
    path: Path,
    *,
    fold_id: int,
    partition: str,
    arm: str,
    roots: list[str],
    samples: list[str],
    closure_fingerprint: str,
    terminal_seal_fingerprint: str | None,
    semantic_payload_fingerprint: str,
) -> dict[str, object]:
    cache_id = f"oof4-fold-{fold_id}-{partition}-{arm}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cache_id.encode())
    flags = list(formal_cache_artifacts._fiemap_flags(path))
    stat_result = path.stat()
    creation_event = 1 if partition == "train" else 4
    phase = (
        "pre_training_train_only"
        if partition == "train"
        else "post_terminal_seal_holdout_only"
    )
    readers = (
        ["BaseB_train_fold_selector"]
        if partition == "train" and arm == "base_eval"
        else (
            [f"{arm}_train_runner"]
            if partition == "train"
            else ["OOF4_read_only_holdout_evaluator"]
        )
    )
    tensor_ledger_fp = _fp(
        "holdout-shared-tensor-ledger"
        if partition == "holdout"
        else f"fold{fold_id}-{partition}-{arm}-tensor-ledger"
    )
    body = {
        "schema_version": (
            "cure-lite-v24-gcr-pacre-oof4-cache-artifact-v1"
        ),
        "cache_id": cache_id,
        "fold_id": fold_id,
        "partition": partition,
        "arm": arm,
        "closure_fingerprint": closure_fingerprint,
        "terminal_seal_fingerprint": terminal_seal_fingerprint,
        "semantic_payload_fingerprint": semantic_payload_fingerprint,
        "root_source_ids": roots,
        "sample_ids": samples,
        "realpath": str(path.resolve()),
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "size_bytes": stat_result.st_size,
        "file_sha256": _sha(path),
        "creation_phase": phase,
        "creation_event": creation_event,
        "reader_allowlist": readers,
        "tensor_ledger_fingerprint": tensor_ledger_fp,
        "fiemap_extent_flags": flags,
        "loader": {
            "torch_load": True,
            "weights_only": True,
            "mmap_used": False,
            "neutral_object_graph": True,
        },
    }
    return {
        "cache_id": cache_id,
        "artifact_fingerprint": stable_fingerprint(body),
        "tensor_ledger_fingerprint": tensor_ledger_fp,
        "partition": partition,
        "arm": arm,
        "closure_fingerprint": closure_fingerprint,
        "terminal_seal_fingerprint": terminal_seal_fingerprint,
        "semantic_payload_fingerprint": semantic_payload_fingerprint,
        "root_source_ids": roots,
        "sample_ids": samples,
        "realpath": str(path.resolve()),
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "size_bytes": stat_result.st_size,
        "file_sha256": _sha(path),
        "creation_phase": phase,
        "creation_event": creation_event,
        "reader_allowlist": readers,
        "is_symlink": False,
        "hardlink_count": 1,
        "fiemap_extent_flags": flags,
        "is_reflink": False,
        "shared_tensor_storage": False,
        "mmap_reused": False,
        "process_cache_reused": False,
    }


def _sum_stats(
    row: dict[str, object],
    count: int,
) -> dict[str, object]:
    if count < 1:
        raise ValueError("statistics count must be positive")
    total = dict(row)
    for _ in range(count - 1):
        total = {
            key: total[key] + value
            for key, value in row.items()
        }
    return total


def _write_formal_cache(
    path: Path,
) -> str:
    cache = make_training_scalar_cache()
    torch.save(build_formal_cache_neutral_envelope(cache), path)
    return cache.cache_fingerprint


def _make_formal_coordinate_scalar_cache():
    """Lift the neutral toy population to the frozen 64/4 coordinates."""

    base = make_training_scalar_cache()

    def feature(value: torch.Tensor) -> torch.Tensor:
        return value.repeat(1, 32, 1, 1).contiguous()

    def mask(value: torch.Tensor) -> torch.Tensor:
        return value.repeat_interleave(2, dim=-2).repeat_interleave(
            2,
            dim=-1,
        ).contiguous()

    naturals = tuple(
        replace(
            row,
            feature=feature(row.feature),
            occupancy=mask(row.occupancy),
            target=mask(row.target),
            valid_mask=mask(row.valid_mask),
            loss_valid_mask=mask(row.loss_valid_mask),
        )
        for row in base.raw_catalog.natural_records
    )
    pairs = tuple(
        replace(
            row,
            feature=feature(row.feature),
            occupancy_plus=mask(row.occupancy_plus),
            occupancy_minus=mask(row.occupancy_minus),
            target_plus=mask(row.target_plus),
            target_minus=mask(row.target_minus),
            valid_mask=mask(row.valid_mask),
            removed_component=mask(row.removed_component),
        )
        for row in base.raw_catalog.pair_records
    )
    catalog = make_coverage_state_raw_catalog(
        dataset="toy-formal-coordinates",
        feature_stride=4,
        source_fingerprint=_fp("toy-formal-coordinate-cache"),
        natural_records=naturals,
        pair_records=pairs,
        exclusions=base.raw_catalog.exclusions,
    )
    observability = audit_population_observability(catalog)
    assert (
        observability.decision
        is CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF
    )
    return prepare_scalar_coverage_state_cache(
        catalog,
        observability,
        CoverageStateSobolevConfig(truncation_radius=4),
    )


def _formal_schedule_meta(
    path: Path,
    *,
    schedule_fingerprint: str,
    semantic_cache_fingerprint: str,
    seed: int,
) -> dict[str, object]:
    policy = {
        "schema_version": (
            "cure-lite-v24-formal800-schedule-policy-without-seed-v1"
        ),
        "semantic_cache_fingerprint": semantic_cache_fingerprint,
        "epochs": 800,
        "steps_per_epoch": 40,
        "updates": FORMAL_UPDATES,
        "logical_states_per_update": 12,
        "objective_invariant": True,
        "optimizer_exposure_accounting": (
            "recomputed_against_current_cache_before_use"
        ),
    }
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "file_sha256": _sha(path),
        "schedule_fingerprint": schedule_fingerprint,
        "seed": seed,
        "epochs": 800,
        "steps_per_epoch": 40,
        "updates": FORMAL_UPDATES,
        "semantic_cache_fingerprint": semantic_cache_fingerprint,
        "policy_without_seed_fingerprint": stable_fingerprint(policy),
    }


def _finite(updates: int) -> dict[str, object]:
    body = {
        "schema_version": "cure-lite-v24-training-finite-audit-v1",
        "expected_updates": updates,
        "loss_values_checked": updates,
        "gradient_tensors_checked": updates * 3,
        "parameter_tensors_checked": (updates + 1) * 3,
        "nonfinite_values": 0,
    }
    return _seal(body, "audit_fingerprint")


def _selector_metrics(pd: float = 0.5) -> dict[str, object]:
    return {
        "pd": pd,
        "retention": 1.0,
        "pixel_fa": 0.0,
        "raw_background_fa": 0.0,
        "fp_components_per_mp": 0.0,
        "budget_violation": False,
    }


def _bounded_metrics(
    *,
    arm: str,
    initial: float,
    terminal: float,
) -> dict[str, object]:
    candidate = arm == "GCR_PACRE_v24"
    terminal_field_fp = _fp(f"{arm}-terminal-field")
    terminal_role_fp = _fp(f"{arm}-terminal-role")

    def distribution(*, forced: bool) -> dict[str, object]:
        return {
            "schema_version": (
                "cure-lite-v24-gcr-pacre-gate-role-summary-v1"
            ),
            "endpoint_counts": {
                "G_equal_0": 0 if forced else 1,
                "G_equal_2": 0 if forced else 1,
                "G_strict_interior": 16 if forced else 14,
            },
            "target_G": {
                "count": 4,
                "minimum": 1.0 if forced else 0.0,
                "maximum": 1.0 if forced else 2.0,
                "mean": 1.0,
            },
            "background_G": {
                "count": 12,
                "minimum": 1.0 if forced else 0.25,
                "maximum": 1.0 if forced else 1.75,
                "mean": 1.0,
            },
            "target_E": {
                "count": 4,
                "minimum": -1.0,
                "maximum": 1.0,
                "mean": 0.0,
            },
            "background_E": {
                "count": 12,
                "minimum": -0.5,
                "maximum": 0.5,
                "mean": 0.0,
            },
        }

    return {
        "true_targets": 10,
        "recovered_anchor_misses": 1,
        "mIoU": 0.6,
        "nIoU": 0.55,
        "pd": 0.5,
        "retention": 1.0,
        "pixel_fa": 0.0,
        "raw_background_fa": 0.0,
        "fp_components_per_mp": 0.0,
        "budget_violation": False,
        "initial_PMOPE": initial,
        "terminal_PMOPE": terminal,
        "terminal_target_role_violation": 0.3,
        "terminal_background_role_violation": 0.2,
        "terminal_zero_crossed_target_states": 2,
        "terminal_false_completion_states": 1,
        "terminal_field_fingerprint": terminal_field_fp,
        "terminal_role_prediction_fingerprint": terminal_role_fp,
        "G1_PMOPE": 1.05 if candidate else terminal,
        "G1_target_role_violation": 0.35 if candidate else 0.3,
        "G1_background_role_violation": 0.25 if candidate else 0.2,
        "G1_zero_crossed_target_states": 1 if candidate else 2,
        "G1_false_completion_states": 2 if candidate else 1,
        "G1_field_fingerprint": (
            _fp(f"{arm}-G1-field") if candidate else terminal_field_fp
        ),
        "G1_role_prediction_fingerprint": (
            _fp(f"{arm}-G1-role") if candidate else terminal_role_fp
        ),
        "terminal_gate_distribution": (
            distribution(forced=False) if candidate else None
        ),
        "G1_gate_distribution": (
            distribution(forced=True) if candidate else None
        ),
        "gate_role_distributions_present": candidate,
    }


def _frozen_seeded_model(arm: str, *, seed: int = 42):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        if arm == GCR_PACRE_CONTROL_ARM:
            return build_pacre_vc_training_model(
                CoverageStatePACREVerifierCorrectedConfig(
                    feature_channels=64,
                    feature_stride=4,
                    width=32,
                )
            )
        if arm == GCR_PACRE_CANDIDATE_ARM:
            return build_formal_gcr_pacre_training_model()
    raise ValueError("unknown frozen test arm")


def _initial_parameter_rows(model: torch.nn.Module) -> list[dict[str, object]]:
    rows = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "numel": parameter.numel(),
            "dtype": str(parameter.dtype),
            "byte_count": parameter.numel() * parameter.element_size(),
            "content_fingerprint": tensor_content_fingerprint(parameter),
        }
        for name, parameter in model.named_parameters()
    ]
    assert [row["name"] for row in rows] == list(
        GCR_PACRE_PARAMETER_NAMES
    )
    return rows


def _bounded_initial_parameter_fingerprint(
    model: torch.nn.Module,
) -> str:
    rows = _initial_parameter_rows(model)
    for row in rows:
        row.pop("byte_count")
    return stable_fingerprint(rows)


def _subnormal_terminal_model(arm: str, *, seed: int = 42):
    model = _frozen_seeded_model(arm, seed=seed)
    with torch.no_grad():
        bias = model.joint_hidden_bias.view(-1)
        assert float(bias[0].item()) == 0.0
        bias[0] = torch.nextafter(
            bias[0],
            torch.tensor(float("inf"), dtype=bias.dtype),
        )
    return model


def _save_test_terminal_safetensors(
    path: Path,
    *,
    model: torch.nn.Module,
    role: str,
    seed: int,
    run: str,
) -> dict[str, object]:
    model_fp = coverage_state_model_fingerprint(model)
    saved = save_terminal_safetensors_new(
        path.resolve(),
        model,
        metadata={
            "schema": "cure-lite-v24-test-terminal-safetensors-v1",
            "run": run,
            "seed": str(seed),
            "role": role,
            "model_fingerprint": model_fp,
        },
    )
    return {**saved, "model_fingerprint": model_fp}


def _test_trace_rows(
    schedule,
    *,
    arm_names: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for update, selection in enumerate(schedule.selections):
        rows.append(
            {
                "update": update,
                "epoch": selection.epoch,
                "step": selection.step,
                "selection_fingerprint": (
                    selection.selection_fingerprint
                ),
                "arms": {
                    arm: {
                        "loss": 1.0 + update / 100_000.0,
                        "gradient_l2_norm": (
                            0.5 + update / 200_000.0
                        ),
                        "optimizer_step_counter": update + 1,
                        "parameter_state_digest": _fp(
                            f"{arm}-parameter-state-{update}"
                        ),
                        "optimizer_state_digest": _fp(
                            f"{arm}-optimizer-state-{update}"
                        ),
                        "loss_finite": True,
                        "gradients_finite": True,
                        "parameters_finite": True,
                        "optimizer_state_finite": True,
                    }
                    for arm in arm_names
                },
            }
        )
    return rows


def _paired_bounded_diagnostics(
    arms: dict[str, object],
    trace_payload: dict[str, object],
) -> dict[str, object]:
    control = arms["PACRE_VC_v23_control"]["metrics"]
    candidate = arms["GCR_PACRE_v24"]["metrics"]
    per_update = []
    for update, trace_row in enumerate(trace_payload["rows"]):
        trace_arms = trace_row["arms"]
        control_trace = trace_arms[GCR_PACRE_CONTROL_ARM]
        candidate_trace = trace_arms[GCR_PACRE_CANDIDATE_ARM]
        control_loss = control_trace["loss"]
        candidate_loss = candidate_trace["loss"]
        control_gradient = control_trace["gradient_l2_norm"]
        candidate_gradient = candidate_trace["gradient_l2_norm"]
        per_update.append(
            {
                "update": update,
                "selection_fingerprint": trace_row[
                    "selection_fingerprint"
                ],
                "control_loss": control_loss,
                "candidate_loss": candidate_loss,
                "candidate_minus_control_loss": (
                    candidate_loss - control_loss
                ),
                "control_gradient_l2_norm": control_gradient,
                "candidate_gradient_l2_norm": candidate_gradient,
                "candidate_minus_control_gradient_l2_norm": (
                    candidate_gradient - control_gradient
                ),
            }
        )
    return {
        "interpretation": (
            "paired_deltas_are_diagnostic_only_without_a_fixed_threshold"
        ),
        "candidate_minus_control": {
            "PMOPE": (
                candidate["terminal_PMOPE"]
                - control["terminal_PMOPE"]
            ),
            "target_role_violation": (
                candidate["terminal_target_role_violation"]
                - control["terminal_target_role_violation"]
            ),
            "background_role_violation": (
                candidate["terminal_background_role_violation"]
                - control["terminal_background_role_violation"]
            ),
            "zero_crossed_target_states": (
                candidate["terminal_zero_crossed_target_states"]
                - control["terminal_zero_crossed_target_states"]
            ),
            "false_completion_states": (
                candidate["terminal_false_completion_states"]
                - control["terminal_false_completion_states"]
            ),
        },
        "candidate_minus_same_weight_G1": {
            "PMOPE": candidate["terminal_PMOPE"] - candidate["G1_PMOPE"],
            "target_role_violation": (
                candidate["terminal_target_role_violation"]
                - candidate["G1_target_role_violation"]
            ),
            "background_role_violation": (
                candidate["terminal_background_role_violation"]
                - candidate["G1_background_role_violation"]
            ),
            "zero_crossed_target_states": (
                candidate["terminal_zero_crossed_target_states"]
                - candidate["G1_zero_crossed_target_states"]
            ),
            "false_completion_states": (
                candidate["terminal_false_completion_states"]
                - candidate["G1_false_completion_states"]
            ),
            "field_nonidentity_witness": True,
            "role_prediction_nonidentity_witness": True,
        },
        "per_update_fingerprint": stable_fingerprint(per_update),
        "per_update": per_update,
    }


def _one_image_stats(arm: str) -> dict[str, object]:
    candidate = arm == "GCR_PACRE_v24"
    forced = arm == "GCR_PACRE_v24_forced_G1"
    matched = 2 if candidate else 1
    recovered = 1 if candidate else 0
    intersection = 6 if candidate or forced else 5
    return {
        "images": 1,
        "matched_gt": matched,
        "total_gt": 2,
        "recovered_anchor_misses": recovered,
        "overlap_supported_recovered_anchor_misses": recovered,
        "total_anchor_misses": 1,
        "retained_anchor_covered": 1,
        "total_anchor_covered": 1,
        "recovered_reachable_anchor_misses": recovered,
        "total_reachable_anchor_misses": 1,
        "unmatched_pred_pixels": 0,
        "unmatched_pred_components": 0,
        "raw_background_fp": 0,
        "total_pixels": 1_000_000,
        "intersection": intersection,
        "union": 10,
        "sum_image_iou": intersection / 10,
    }


def _force_full_endpoint_gate(
    metrics: dict[str, object],
) -> None:
    distribution = metrics["terminal_gate_distribution"]
    total = (
        distribution["target_G"]["count"]
        + distribution["background_G"]["count"]
    )
    distribution["endpoint_counts"] = {
        "G_equal_0": total,
        "G_equal_2": 0,
        "G_strict_interior": 0,
    }
    for role in ("target_G", "background_G"):
        distribution[role].update(
            {
                "minimum": 0.0,
                "maximum": 0.0,
                "mean": 0.0,
            }
        )


def _reseal_bounded_diagnostic_row_against_itself(
    receipt: dict[str, object],
) -> None:
    diagnostics = receipt["paired_diagnostics"]
    rows = diagnostics["per_update"]
    row = rows[0]
    row["candidate_loss"] += 0.25
    row["candidate_minus_control_loss"] = (
        row["candidate_loss"] - row["control_loss"]
    )
    diagnostics["per_update_fingerprint"] = stable_fingerprint(rows)


@pytest.fixture(scope="module")
def protocol_chain(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
) -> dict[str, object]:
    work = tmp_path_factory.mktemp("gcr_pacre_protocol")
    split_payload = json.loads(
        (PROTOCOL_ROOT / "D_R_OOF4_split_preregistration.json").read_text()
    )
    split = verify_oof4_split_preregistration(
        split_payload,
        repository_root=REPO_ROOT,
    )
    source_rows = tuple(
        (relative, _sha(REPO_ROOT / relative))
        for relative in GCR_PACRE_V24_SOURCE_CLOSURE_PATHS
    )
    source_hashes = dict(source_rows)
    source_closure_fp = stable_fingerprint(
        {
            "schema_version": GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
            "source_hashes": source_hashes,
        }
    )
    fixed_sources = oof_run_start.required_oof_dr_source_paths()
    source_binding, _, _, _ = bind_coverage_state_real_dr_sources(
        **fixed_sources
    )
    generated_dr_body = {
        "decision": {
            "status": "GCR_PACRE_V24_D_R_STRUCTURAL_PASS",
        },
        "input_binding": {
            "source_binding_fingerprint": (
                source_binding.binding_fingerprint
            ),
        },
        "real_inputs_fingerprint": _fp("generated-real-inputs"),
        "cache_fingerprint": _fp("generated-full-cache"),
    }
    generated_dr_receipt = _seal(generated_dr_body)
    generated_dr_path = work / "generated-D_R-structural-receipt.json"
    generated_dr_path.write_text(
        canonical_json(generated_dr_receipt) + "\n",
        encoding="utf-8",
    )
    runtime_root = work / "oof_runtime"
    patcher = pytest.MonkeyPatch()
    request.addfinalizer(patcher.undo)
    # This broad protocol fixture supplies hand-authored factual rows so the
    # later bounded/Formal schema tests remain compact.  The actual neutral
    # cache/terminal/schedule/evaluator replay is exercised without mocking
    # in test_gcr_pacre_v24_oof_mechanical_replay.py.
    patcher.setattr(
        oof_evaluation_module,
        "mechanically_replay_oof_fold_evidence",
        lambda *_args, **_kwargs: _fp("fixture-oof-mechanical-replay"),
    )
    patcher.setattr(
        oof_run_start,
        "required_oof_runtime_root",
        lambda: runtime_root,
    )
    patcher.setattr(
        oof_run_start,
        "required_oof_dr_receipt_path",
        lambda: generated_dr_path,
    )
    patcher.setattr(
        oof_run_start,
        "verify_gcr_pacre_dr_receipt",
        lambda receipt: str(receipt["receipt_fingerprint"]),
    )
    patcher.setattr(
        oof_run_start,
        "gcr_pacre_v24_source_closure_hashes",
        lambda: source_rows,
    )
    patcher.setattr(
        oof_run_start,
        "gcr_pacre_v24_source_closure_fingerprint",
        lambda rows=None: source_closure_fp,
    )
    execution_authorization = authorize_real_oof4_execution_new(
        verified_split=split,
        source_binding=source_binding,
    )
    fold_receipts: list[dict[str, object]] = []
    fold_access_receipts: list[dict[str, object]] = []
    fold_access_tokens = []
    fold_tokens = []
    factual_rows_by_fold: list[list[dict[str, object]]] = []
    fold_pools = []

    for raw_fold in split.plan["folds"]:  # type: ignore[index]
        fold = dict(raw_fold)
        fold_id = int(fold["fold_id"])
        train_samples = list(fold["train_sample_ids"])
        holdout_samples = list(fold["held_out_sample_ids"])
        train_roots = list(fold["train_root_source_ids"])
        holdout_roots = list(fold["held_out_root_source_ids"])
        events = {
            "train_cache_materialized": 1,
            "training_claimed": 2,
            "training_terminals_sealed": 3,
            "holdout_cache_materialized": 4,
            "holdout_cache_first_open": 5,
        }
        closure_body = {
            "schema_version": (
                "cure-lite-v24-gcr-pacre-oof4-fold-closure-v1"
            ),
            "fold_id": fold_id,
            "split_receipt_fingerprint": split.receipt_fingerprint,
            "plan_fingerprint": split.plan_fingerprint,
            "root_by_sample_fingerprint": (
                split.root_by_sample_fingerprint
            ),
            "train_root_source_ids": train_roots,
            "held_out_root_source_ids": holdout_roots,
            "train_sample_ids": train_samples,
            "held_out_sample_ids": holdout_samples,
            "root_by_sample": dict(sorted(split.root_by_sample.items())),
            "checks": {
                "root_sets_disjoint": True,
                "root_union_exact": True,
                "sample_sets_disjoint": True,
                "sample_union_exact": True,
                "sample_to_root_closure_exact": True,
                "D_V_payload_accessed": False,
                "D_T_payload_accessed": False,
            },
        }
        closure_fp = stable_fingerprint(closure_body)
        scalar_semantic_fp = _fp(f"fold{fold_id}-train-scalar")
        train_entries = [
            _oof_cache_entry(
                runtime_root
                / f"fold_{fold_id}"
                / "train"
                / {
                    "base_eval": "base_eval",
                    "PACRE_VC_v23_control": "v23_control",
                    "GCR_PACRE_v24": "candidate",
                }[arm]
                / "cache.pt",
                fold_id=fold_id,
                partition="train",
                arm=arm,
                roots=train_roots,
                samples=train_samples,
                closure_fingerprint=closure_fp,
                terminal_seal_fingerprint=None,
                semantic_payload_fingerprint=(
                    _fp(f"fold{fold_id}-train-base")
                    if arm == "base_eval"
                    else scalar_semantic_fp
                ),
            )
            for arm in (
                "base_eval",
                "PACRE_VC_v23_control",
                "GCR_PACRE_v24",
            )
        ]
        train_by_arm = {str(row["arm"]): row for row in train_entries}
        schedule_fp = _fp(f"fold{fold_id}-schedule")
        batch_fp = _fp(f"fold{fold_id}-batches")
        process_instance_fp = _fp(f"fold{fold_id}-process")
        marker_path = runtime_root / f"fold_{fold_id}" / "run_start.json"
        run_start_body = {
            "schema_version": (
                "cure-lite-v24-gcr-pacre-oof4-fold-persistent-run-start-v1"
            ),
            "fold_id": fold_id,
            "closure_fingerprint": closure_fp,
            "split_receipt_fingerprint": split.receipt_fingerprint,
            "authorization_fingerprint": (
                execution_authorization.authorization_fingerprint
            ),
            "authorization_artifact_file_sha256": (
                execution_authorization.artifact_file_sha256
            ),
            "source_binding_fingerprint": (
                source_binding.binding_fingerprint
            ),
            "source_closure_fingerprint": source_closure_fp,
            "seed": 42,
            "epochs": 10,
            "steps_per_epoch": 40,
            "updates_per_arm": BOUNDED_UPDATES,
            "event_index": 2,
            "event": "training_claimed",
            "process_instance_fingerprint": process_instance_fp,
            "schedule_fingerprint": schedule_fp,
            "batch_sequence_fingerprint": batch_fp,
            "training_population_fingerprint": scalar_semantic_fp,
            "control_cache_artifact_fingerprint": train_by_arm[
                "PACRE_VC_v23_control"
            ]["artifact_fingerprint"],
            "candidate_cache_artifact_fingerprint": train_by_arm[
                "GCR_PACRE_v24"
            ]["artifact_fingerprint"],
            "output_directory": str(marker_path.parent),
            "marker_path": str(marker_path),
            "from_scratch": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "checkpoint_policy": "final_only",
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
        run_start_payload = _seal(
            run_start_body,
            field="marker_fingerprint",
        )
        run_start_artifact = {
            **_write_immutable_json(marker_path, run_start_payload),
            "marker_fingerprint": run_start_payload[
                "marker_fingerprint"
            ],
            "payload": run_start_payload,
        }
        initial_parameters = [
            {
                "name": "joint_state_weight",
                "shape": [32, 80, 5, 5],
                "dtype": "torch.float32",
                "numel": 64_000,
                "byte_count": 256_000,
                "content_fingerprint": _fp("shared-joint-state"),
            },
            {
                "name": "joint_hidden_bias",
                "shape": [32],
                "dtype": "torch.float32",
                "numel": 32,
                "byte_count": 128,
                "content_fingerprint": _fp("shared-hidden-bias"),
            },
            {
                "name": "scalar_energy_weight",
                "shape": [32],
                "dtype": "torch.float32",
                "numel": 32,
                "byte_count": 128,
                "content_fingerprint": _fp("shared-scalar-energy"),
            },
        ]
        initial_ledger_fp = stable_fingerprint(initial_parameters)
        common = {
            "seed": 42,
            "epochs": 10,
            "steps_per_epoch": 40,
            "completed_updates": BOUNDED_UPDATES,
            "training_invocations": 1,
            "from_scratch": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "checkpoint_policy": "final_only",
            "optimizer_state_initial_empty": True,
            "train_root_source_ids": train_roots,
            "train_sample_ids": train_samples,
            "schedule_fingerprint": schedule_fp,
            "batch_sequence_fingerprint": batch_fp,
            "training_population_fingerprint": scalar_semantic_fp,
            "initial_shared_parameter_fingerprint": initial_ledger_fp,
            "initial_parameters": initial_parameters,
            "run_start_marker_fingerprint": run_start_payload[
                "marker_fingerprint"
            ],
            "PMOPE_fingerprint": _fp("PMOPE"),
            "Adam_policy_fingerprint": _fp("Adam"),
            "dtype_device_policy_fingerprint": _fp("float32-cpu"),
            "source_hashes": source_hashes,
        }
        training_arms: dict[str, object] = {}
        final_fps: dict[str, str] = {}
        terminal_fps: dict[str, str] = {}
        capability_fps: dict[str, str] = {}
        for arm in ("PACRE_VC_v23_control", "GCR_PACRE_v24"):
            terminal_name = (
                "v23_control_terminal.safetensors"
                if arm == "PACRE_VC_v23_control"
                else "candidate_terminal.safetensors"
            )
            artifact_path = (
                runtime_root
                / f"fold_{fold_id}"
                / "terminal"
                / terminal_name
            )
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            torch.manual_seed(10_000 + 10 * fold_id)
            model = (
                build_pacre_vc_training_model(
                    CoverageStatePACREVerifierCorrectedConfig(
                        feature_channels=64,
                        feature_stride=4,
                        width=32,
                    )
                )
                if arm == "PACRE_VC_v23_control"
                else build_formal_gcr_pacre_training_model()
            )
            initial_fp = _fp(f"fold{fold_id}-shared-initial-model")
            final_fp = coverage_state_model_fingerprint(model)
            final_fps[arm] = final_fp
            training_result_fp = _fp(f"fold{fold_id}-{arm}-training")
            storage_rows = [
                {
                    "name": parameter["name"],
                    "device": "cpu",
                    "nbytes": parameter["byte_count"],
                    "storage_identity_fingerprint": _fp(
                        f"fold{fold_id}-{arm}-{parameter['name']}-storage"
                    ),
                }
                for parameter in initial_parameters
            ]
            storage_fp = stable_fingerprint(storage_rows)
            saved_terminal = save_terminal_safetensors_new(
                artifact_path,
                model,
                metadata={
                    "schema": (
                        "cure-lite-v24-gcr-pacre-oof4-"
                        "terminal-safetensors-v1"
                    ),
                    "run": f"oof4-fold-{fold_id}",
                    "seed": "42",
                    "role": arm,
                    "arm": arm,
                    "model_fingerprint": final_fp,
                    "epochs": "10",
                    "steps_per_epoch": "40",
                    "updates": "400",
                    "checkpoint_policy": "final_only",
                    "run_start_marker_fingerprint": str(
                        run_start_payload["marker_fingerprint"]
                    ),
                },
            )
            terminal_body = {
                "schema_version": (
                    "cure-lite-v24-gcr-pacre-oof4-terminal-safetensors-v1"
                ),
                "fold_id": fold_id,
                "arm": arm,
                "seed": 42,
                "epochs": 10,
                "steps_per_epoch": 40,
                "completed_updates": BOUNDED_UPDATES,
                "path": saved_terminal["path"],
                "size_bytes": saved_terminal["size_bytes"],
                "file_sha256": saved_terminal["file_sha256"],
                "device": saved_terminal["device"],
                "inode": saved_terminal["inode"],
                "hardlink_count": saved_terminal["hardlink_count"],
                "state_keys": saved_terminal["state_keys"],
                "state_shapes": saved_terminal["state_shapes"],
                "state_dtypes": saved_terminal["state_dtypes"],
                "parameter_count": saved_terminal["parameter_count"],
                "model_fingerprint": final_fp,
                "training_result_fingerprint": training_result_fp,
                "run_start_marker_fingerprint": run_start_payload[
                    "marker_fingerprint"
                ],
                "serialization": "safetensors",
                "final_checkpoint_only": True,
                "optimizer_state_saved": False,
                "intermediate_checkpoint_saved": False,
                "D_V_payload_accessed": False,
                "D_T_payload_accessed": False,
            }
            terminal = _seal(
                terminal_body,
                field="terminal_artifact_fingerprint",
            )
            module_fp = _fp(f"fold{fold_id}-{arm}-module")
            optimizer_fp = _fp(f"fold{fold_id}-{arm}-optimizer")
            capability_body = {
                "schema_version": (
                    "cure-lite-v24-gcr-pacre-oof4-"
                    "completed-400-capability-v1"
                ),
                "fold_id": fold_id,
                "arm": arm,
                "closure_fingerprint": closure_fp,
                "run_start": {
                    "marker_fingerprint": run_start_payload[
                        "marker_fingerprint"
                    ],
                    "marker_path": str(marker_path),
                    "marker_file_sha256": run_start_artifact["file_sha256"],
                    "authorization_fingerprint": (
                        execution_authorization.authorization_fingerprint
                    ),
                    "source_closure_fingerprint": source_closure_fp,
                },
                "train_cache": {
                    "artifact_fingerprint": train_by_arm[arm][
                        "artifact_fingerprint"
                    ],
                    "semantic_payload_fingerprint": scalar_semantic_fp,
                    "reader_authorization_fingerprint": _fp(
                        f"fold{fold_id}-{arm}-reader"
                    ),
                },
                "seed": 42,
                "epochs": 10,
                "steps_per_epoch": 40,
                "completed_updates": BOUNDED_UPDATES,
                "training_invocations": 1,
                "schedule_fingerprint": schedule_fp,
                "batch_sequence_fingerprint": batch_fp,
                "shared_initial_parameter_fingerprint": initial_ledger_fp,
                "initial_parameters": initial_parameters,
                "model_config": {
                    "feature_channels": 64,
                    "feature_stride": 4,
                    "width": 32,
                    "parameter_count": 64_064,
                },
                "module_instance_id": module_fp,
                "optimizer_instance_id": optimizer_fp,
                "parameter_storage_ledger": storage_rows,
                "parameter_storage_ledger_fingerprint": storage_fp,
                "optimizer_fqcn": "torch.optim.adam.Adam",
                "optimizer_config_fingerprint": _fp("Adam"),
                "objective": "pmope_joint",
                "objective_policy_fingerprint": _fp("PMOPE"),
                "training_result_fingerprint": training_result_fp,
                "terminal_artifact": terminal,
                "source_hashes": source_hashes,
                "from_scratch": True,
                "resume_allowed": False,
                "automatic_retry_allowed": False,
                "checkpoint_policy": "final_only",
                "holdout_payload_accessed": False,
                "D_V_payload_accessed": False,
                "D_T_payload_accessed": False,
            }
            capability = _seal(
                capability_body,
                field="capability_fingerprint",
            )
            terminal_fps[arm] = str(
                terminal["terminal_artifact_fingerprint"]
            )
            capability_fps[arm] = str(capability["capability_fingerprint"])
            training_arms[arm] = {
                **common,
                "completed_400_capability": capability,
                "completed_400_capability_fingerprint": capability[
                    "capability_fingerprint"
                ],
                "module_instance_id": module_fp,
                "optimizer_instance_id": optimizer_fp,
                "parameter_storage_ledger": storage_rows,
                "parameter_storage_ledger_fingerprint": storage_fp,
                "initial_model_fingerprint": initial_fp,
                "final_model_fingerprint": final_fp,
                "terminal_artifact_fingerprint": terminal[
                    "terminal_artifact_fingerprint"
                ],
                "terminal_artifact": terminal,
            }
        terminal_seal_body = {
            "schema_version": (
                "cure-lite-v24-gcr-pacre-oof4-training-terminal-seal-v1"
            ),
            "fold_id": fold_id,
            "closure_fingerprint": closure_fp,
            "terminal_artifact_fingerprints": terminal_fps,
            "completed_400_capability_fingerprints": capability_fps,
            "run_start_marker_fingerprint": run_start_payload[
                "marker_fingerprint"
            ],
            "shared_initial_parameter_fingerprint": initial_ledger_fp,
            "initial_parameters": initial_parameters,
            "schedule_fingerprint": schedule_fp,
            "batch_sequence_fingerprint": batch_fp,
            "semantic_cache_fingerprint": scalar_semantic_fp,
            "optimizer_config_fingerprint": _fp("Adam"),
            "objective_policy_fingerprint": _fp("PMOPE"),
            "event_index": 3,
        }
        terminal_seal = _seal(
            terminal_seal_body,
            field="seal_fingerprint",
        )
        holdout_semantic_fp = _fp(f"fold{fold_id}-holdout-dataset")
        holdout_entries = [
            _oof_cache_entry(
                runtime_root
                / f"fold_{fold_id}"
                / "holdout"
                / {
                    "base_eval": "base_eval",
                    "PACRE_VC_v23_control": "v23_control",
                    "GCR_PACRE_v24": "candidate",
                }[arm]
                / "cache.pt",
                fold_id=fold_id,
                partition="holdout",
                arm=arm,
                roots=holdout_roots,
                samples=holdout_samples,
                closure_fingerprint=closure_fp,
                terminal_seal_fingerprint=str(
                    terminal_seal["seal_fingerprint"]
                ),
                semantic_payload_fingerprint=holdout_semantic_fp,
            )
            for arm in (
                "base_eval",
                "PACRE_VC_v23_control",
                "GCR_PACRE_v24",
            )
        ]
        cache_entries = sorted(
            [*train_entries, *holdout_entries],
            key=lambda row: (str(row["partition"]), str(row["arm"])),
        )
        cache_set_fp = stable_fingerprint(
            {
                "schema_version": (
                    "cure-lite-v24-gcr-pacre-oof4-"
                    "six-cache-independence-v1"
                ),
                "fold_id": fold_id,
                "cache_artifact_fingerprints": sorted(
                    str(row["artifact_fingerprint"])
                    for row in cache_entries
                ),
                "entries": cache_entries,
            }
        )
        observed = [
            {
                "split": "D_R",
                "logical_id": row["cache_id"],
                "purpose": (
                    "train_cache_materialization"
                    if row["partition"] == "train"
                    else "read_only_holdout_cache_materialization"
                ),
                "source_fingerprint": row["file_sha256"],
            }
            for row in cache_entries
        ]
        access_body = {
            **_access_receipt(
                f"oof4_fold_{fold_id}",
                ["D_R"],
                observed,
            ),
            "source_manifest_fingerprint": (
                source_binding.binding_fingerprint
            ),
        }
        access_body.pop("receipt_fingerprint")
        access_receipt = _seal(access_body)
        access = verify_access_audit_receipt(
            access_receipt,
            expected_stage_id=f"oof4_fold_{fold_id}",
            allowed_splits=["D_R"],
        )
        fold_access_receipts.append(access_receipt)
        fold_access_tokens.append(access)
        base_cache_sha = next(
            str(row["file_sha256"])
            for row in cache_entries
            if row["partition"] == "train" and row["arm"] == "base_eval"
        )
        candidate_rows = [
            {
                "threshold": threshold,
                "selection_split_role": "OOF_train_fold",
                "train_sample_ids": train_samples,
                "train_root_source_ids": train_roots,
                "input_train_cache_fingerprint": base_cache_sha,
                "access_audit_receipt_fingerprint": access.receipt_fingerprint,
                "metrics": _selector_metrics(),
                "D_V_payload_accessed": False,
                "D_T_payload_accessed": False,
            }
            for threshold in BASE_B_THRESHOLD_GRID
        ]
        ledger_fp = stable_fingerprint(candidate_rows)
        selected = 1.0
        base_b_selection = {
            "selection_root_source_ids": train_roots,
            "selection_sample_ids": train_samples,
            "evaluation_root_source_ids": holdout_roots,
            "evaluation_sample_ids": holdout_samples,
            "holdout_labels_used_for_selection": False,
            "complete_51_point_grid_evaluated": True,
            "D_V_threshold_reused": False,
            "grid_source_repo_path": (
                "protocols/IRSTD-1K/stage_a_seed42_fx_v3/"
                "stage_a_config.json"
            ),
            "grid_source_file_sha256": (
                "6eecdc10f87a043cafb945db40d0b767b5f0a2ccb64963c1043160f165ce9d6c"
            ),
            "threshold_grid": list(BASE_B_THRESHOLD_GRID),
            "candidate_rows": candidate_rows,
            "candidate_ledger_fingerprint": ledger_fp,
            "selector_policy": [
                "maximize_pd",
                "maximize_retention",
                "minimize_pixel_fa",
                "minimize_raw_background_fa",
                "minimize_fp_components_per_mp",
                "maximize_threshold",
            ],
            "selector_policy_fingerprint": stable_fingerprint(
                [
                    "maximize_pd",
                    "maximize_retention",
                    "minimize_pixel_fa",
                    "minimize_raw_background_fa",
                    "minimize_fp_components_per_mp",
                    "maximize_threshold",
                ]
            ),
            "selected_threshold": selected,
            "input_train_cache_fingerprint": base_cache_sha,
            "access_audit_receipt_fingerprint": access.receipt_fingerprint,
        }
        evaluation_fps = {
            "BaseA": stable_fingerprint(
                {"arm": "BaseA", "threshold": BASE_A_THRESHOLD}
            ),
            "BaseB_train_fold_selected": stable_fingerprint(
                {
                    "arm": "BaseB_train_fold_selected",
                    "candidate_ledger_fingerprint": ledger_fp,
                    "selected_threshold": selected,
                }
            ),
            "PACRE_VC_v23_control": final_fps["PACRE_VC_v23_control"],
            "GCR_PACRE_v24": final_fps["GCR_PACRE_v24"],
            "GCR_PACRE_v24_forced_G1": final_fps["GCR_PACRE_v24"],
        }
        evaluator_fp = OOFConcreteEvaluator.fixed().evaluator_fingerprint
        evaluation_ledger_artifacts: dict[str, object] = {}
        evaluation_rows_by_arm: dict[str, list[dict[str, object]]] = {}
        witness_sample = holdout_samples[0] if fold_id == 0 else None
        for arm in OOF_ARMS:
            evaluation_rows: list[dict[str, object]] = []
            for sample_id in holdout_samples:
                if arm in {
                    "GCR_PACRE_v24",
                    "GCR_PACRE_v24_forced_G1",
                }:
                    field_fp = _fp(
                        f"fold{fold_id}-{sample_id}-v24-shared-field"
                    )
                    prediction_fp = _fp(
                        f"fold{fold_id}-{sample_id}-v24-shared-prediction"
                    )
                    if (
                        arm == "GCR_PACRE_v24_forced_G1"
                        and sample_id == witness_sample
                    ):
                        field_fp = _fp(
                            f"fold{fold_id}-{sample_id}-forced-G1-field"
                        )
                        prediction_fp = _fp(
                            f"fold{fold_id}-{sample_id}-forced-G1-prediction"
                        )
                else:
                    field_fp = _fp(
                        f"fold{fold_id}-{sample_id}-{arm}-field"
                    )
                    prediction_fp = _fp(
                        f"fold{fold_id}-{sample_id}-{arm}-prediction"
                    )
                evaluation_rows.append(
                    {
                        "sample_id": sample_id,
                        "root_source_id": split.root_by_sample[sample_id],
                        "statistics": _one_image_stats(arm),
                        "field_fingerprint": field_fp,
                        "prediction_fingerprint": prediction_fp,
                        "role_statistics": {},
                    }
                )
            evaluation_rows_by_arm[arm] = evaluation_rows
            model_fp = (
                None
                if arm in {"BaseA", "BaseB_train_fold_selected"}
                else (
                    final_fps["PACRE_VC_v23_control"]
                    if arm == "PACRE_VC_v23_control"
                    else final_fps["GCR_PACRE_v24"]
                )
            )
            evaluation_body = {
                "schema_version": (
                    "cure-lite-v24-gcr-pacre-oof4-"
                    "evaluation-ledger-v1"
                ),
                "fold_id": fold_id,
                "partition": "holdout",
                "arm": arm,
                "operating_point": (
                    selected
                    if arm == "BaseB_train_fold_selected"
                    else BASE_A_THRESHOLD
                ),
                "dataset_fingerprint": holdout_semantic_fp,
                "model_fingerprint": model_fp,
                "per_sample_rows": evaluation_rows,
                "pooled_statistics": _sum_stats(
                    _one_image_stats(arm),
                    len(holdout_samples),
                ),
                "field_ledger_fingerprint": stable_fingerprint(
                    [row["field_fingerprint"] for row in evaluation_rows]
                ),
                "prediction_ledger_fingerprint": stable_fingerprint(
                    [
                        row["prediction_fingerprint"]
                        for row in evaluation_rows
                    ]
                ),
                "role_ledger_fingerprint": stable_fingerprint(
                    [row["role_statistics"] for row in evaluation_rows]
                ),
                "evaluator_fingerprint": evaluator_fp,
                "D_V_payload_accessed": False,
                "D_T_payload_accessed": False,
            }
            evaluation_payload = _seal(
                evaluation_body,
                field="ledger_fingerprint",
            )
            evaluation_path = (
                runtime_root
                / f"fold_{fold_id}"
                / "evaluation"
                / f"{arm}.json"
            )
            evaluation_path.parent.mkdir(parents=True, exist_ok=True)
            evaluation_ledger_artifacts[arm] = {
                **_write_immutable_json(
                    evaluation_path,
                    evaluation_payload,
                ),
                "ledger_fingerprint": evaluation_payload[
                    "ledger_fingerprint"
                ],
            }
        factual_rows: list[dict[str, object]] = []
        for arm in OOF_ARMS:
            for sample_id in holdout_samples:
                factual_rows.append(
                    {
                        "split": "D_R",
                        "evidence_role": "factual_only",
                        "fold_id": fold_id,
                        "arm": arm,
                        "sample_id": sample_id,
                        "root_source_id": split.root_by_sample[sample_id],
                        "gt_fingerprint": _fp(f"{sample_id}-gt"),
                        "anchor_state_fingerprint": _fp(
                            f"{sample_id}-anchor"
                        ),
                        "evaluation_contract_fingerprint": evaluator_fp,
                        "terminal_artifact_fingerprint": evaluation_fps[arm],
                        "sufficient_statistics": _one_image_stats(arm),
                    }
                )
        factual_body = {
            "schema_version": (
                "cure-lite-v24-gcr-pacre-oof4-"
                "factual-rows-artifact-v1"
            ),
            "fold_id": fold_id,
            "closure_fingerprint": closure_fp,
            "rows": factual_rows,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
        factual_payload = _seal(
            factual_body,
            field="ledger_fingerprint",
        )
        factual_path = (
            runtime_root / f"fold_{fold_id}" / "factual_rows.json"
        )
        factual_rows_artifact = {
            **_write_immutable_json(factual_path, factual_payload),
            "ledger_fingerprint": factual_payload["ledger_fingerprint"],
            "payload": factual_payload,
        }
        fold_body = {
            "schema_version": "cure-lite-v24-oof-fold-execution-v3",
            "split_preregistration_fingerprint": split.receipt_fingerprint,
            "root_by_sample_fingerprint": split.root_by_sample_fingerprint,
            "plan_fingerprint": split.plan_fingerprint,
            "fold_id": fold_id,
            "train_root_source_ids": train_roots,
            "held_out_root_source_ids": holdout_roots,
            "train_sample_ids": train_samples,
            "held_out_sample_ids": holdout_samples,
            "access_audit_receipt_fingerprint": access.receipt_fingerprint,
            "events": events,
            "run_start_artifact": run_start_artifact,
            "terminal_seal": terminal_seal,
            "cache_set_fingerprint": cache_set_fp,
            "cache_entries": cache_entries,
            "training_arms": training_arms,
            "BaseB_train_fold_selection": base_b_selection,
            "evaluation_artifact_fingerprints": evaluation_fps,
            "evaluation_ledger_artifacts": evaluation_ledger_artifacts,
            "factual_rows_artifact": factual_rows_artifact,
            "held_out_prediction_role": "factual_only",
            "source_closure": {
                "schema_version": GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
                "source_hashes": source_hashes,
                "source_closure_fingerprint": source_closure_fp,
            },
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
        fold_receipt = _seal(fold_body)
        fold_token = validate_oof_fold_execution_receipt(
            fold_receipt,
            split,
            access_audit=access,
            execution_authorization=execution_authorization,
            repository_root=REPO_ROOT,
        )
        fold_receipts.append(fold_receipt)
        fold_tokens.append(fold_token)
        factual_rows_by_fold.append(factual_rows)
        fold_pools.append(
            pool_factual_only_rows(
                factual_rows,
                fold_token,
                access_audit=access,
            )
        )

    pooled = combine_oof4_factual_pools(fold_pools, split)
    artifacts_by_fold = pooled.payload[
        "evaluation_artifact_fingerprints_by_fold"
    ]
    v24_fps = [
        artifacts_by_fold[str(index)]["GCR_PACRE_v24"]
        for index in range(4)
    ]
    field_difference_ledger = pooled.payload[
        "verified_field_difference_ledger"
    ]
    prediction_difference_ledger = pooled.payload[
        "verified_prediction_difference_ledger"
    ]
    gate_body = {
        "schema_version": "cure-lite-v24-gcr-pacre-forced-G1-path-v2",
        "pooled_evidence_fingerprint": pooled.evidence_fingerprint,
        "sample_roots_fingerprint": pooled.payload[
            "sample_roots_fingerprint"
        ],
        "v24_terminal_artifact_fingerprints": v24_fps,
        "forced_G1_terminal_artifact_fingerprints": v24_fps,
        "forced_G1_retrained": False,
        "field_difference_count": len(field_difference_ledger),
        "prediction_difference_count": len(prediction_difference_ledger),
        "field_difference_ledger": field_difference_ledger,
        "field_difference_ledger_fingerprint": stable_fingerprint(
            field_difference_ledger
        ),
        "prediction_difference_ledger": prediction_difference_ledger,
        "prediction_difference_ledger_fingerprint": stable_fingerprint(
            prediction_difference_ledger
        ),
        "access_audit_receipt_fingerprints": list(
            pooled.access_audit_receipt_fingerprints
        ),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    gate_receipt = _seal(gate_body)
    gate = verify_gate_path_receipt(gate_receipt, pooled)
    oof_decision = decide_oof4_pooled(
        pooled,
        gate_path_evidence=gate,
    )
    assert oof_decision["gate_passed"] is True

    dataset_free_fp = _fp("dataset-free")
    structural_fp = _fp("D_R-structural")
    formal_scalar_cache = _make_formal_coordinate_scalar_cache()
    semantic_formal_cache_fp = formal_scalar_cache.cache_fingerprint
    bounded_full_cache_path = work / "bounded-full-D_R-materialization.pt"
    bounded_full_cache_token = save_formal_cache_neutral_artifact_new(
        formal_scalar_cache,
        bounded_full_cache_path.resolve(),
        cache_id="paired-bounded400-full-D_R-materialization",
    )
    bounded_access_receipt = _access_receipt(
        "paired_bounded400",
        ["D_R"],
        [
            {
                "split": "D_R",
                "logical_id": bounded_full_cache_token.cache_id,
                "purpose": (
                    "paired_bounded400_full_D_R_materialization"
                ),
                "source_fingerprint": (
                    bounded_full_cache_token.file_sha256
                ),
            }
        ],
    )
    bounded_access = verify_access_audit_receipt(
        bounded_access_receipt,
        expected_stage_id="paired_bounded400",
        allowed_splits=["D_R"],
    )
    evidence_runtime = work / "evidence_runtime"
    patcher.setattr(
        protocol_module,
        "_gcr_pacre_v24_evidence_runtime_root",
        lambda repository_root: evidence_runtime,
    )
    bounded_runtime = evidence_runtime / "bounded"
    bounded_output = bounded_runtime / "paired_bounded400"
    bounded_output.mkdir(parents=True)
    bounded_chain_path = bounded_runtime / "execution_chain_config.json"
    bounded_cache_binding = {
        "receipt_fingerprint": (
            bounded_full_cache_token.receipt_fingerprint
        ),
        "cache_id": bounded_full_cache_token.cache_id,
        "path": bounded_full_cache_token.path,
        "file_sha256": bounded_full_cache_token.file_sha256,
        "device": bounded_full_cache_token.device,
        "inode": bounded_full_cache_token.inode,
        "hardlink_count": bounded_full_cache_token.hardlink_count,
        "semantic_cache_fingerprint": (
            bounded_full_cache_token.semantic_cache_fingerprint
        ),
        "neutral_payload_fingerprint": (
            bounded_full_cache_token.neutral_payload_fingerprint
        ),
    }
    bounded_chain_body = {
        "schema_version": (
            "cure-lite-v24-gcr-pacre-paired-bounded400-chain-config-v1"
        ),
        "protocol_id": "irstd1k-gcr-pacre-v24-evidence-v1",
        "path_policy": (
            "fixed_runtime_root_bounded_paired_bounded400_"
            "run_start_json_v1"
        ),
        "repository_root": str(REPO_ROOT),
        "runtime_root": str(evidence_runtime),
        "chain_config_path": str(bounded_chain_path),
        "source_closure_schema": GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
        "source_hashes": dict(source_hashes),
        "source_closure_fingerprint": source_closure_fp,
        "predecessors": {
            "dataset_free_receipt_fingerprint": dataset_free_fp,
            "D_R_structural_receipt_fingerprint": structural_fp,
            "OOF4_decision_fingerprint": oof_decision.decision_fingerprint,
            "access_audit_receipt_fingerprint": (
                bounded_access.receipt_fingerprint
            ),
        },
        "access_audit_receipt": bounded_access.payload,
        "full_D_R_cache_artifact": bounded_cache_binding,
        "requested_device": "cpu",
        "output_directory": str(bounded_output),
        "run_start_marker_path": str(bounded_output / "run_start.json"),
        "authorization_artifact_path": str(
            bounded_output / "authorization.json"
        ),
        "schedule_artifact_path": str(bounded_output / "schedule.json"),
        "control_terminal_artifact_path": str(
            bounded_output / "control_terminal.safetensors"
        ),
        "candidate_terminal_artifact_path": str(
            bounded_output / "candidate_terminal.safetensors"
        ),
        "result_artifact_path": str(
            bounded_output / "bounded_400_result.json"
        ),
        "diagnostics_artifact_path": str(
            bounded_output / "bounded_400_diagnostics.json"
        ),
        "decision_artifact_path": str(
            bounded_output / "bounded_400_decision.json"
        ),
        "budget": {
            "seed": 42,
            "epochs": 10,
            "steps_per_epoch": 40,
            "updates_per_arm": BOUNDED_UPDATES,
            "training_invocations_per_arm": 1,
        },
        "attempt_policy": {
            "from_scratch": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "persistent_O_EXCL_run_start_required": True,
            "run_start_marker_never_removed_after_creation": True,
            "checkpoint_policy": "final_only",
            "fixed_relative_promotion_threshold": None,
            "relative_diagnostics_authorize": False,
        },
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    bounded_chain_payload = {
        **bounded_chain_body,
        "config_fingerprint": stable_fingerprint(bounded_chain_body),
    }
    bounded_chain_meta = _write_immutable_json(
        bounded_chain_path,
        bounded_chain_payload,
    )
    bounded_chain_artifact = {
        "path": bounded_chain_meta["path"],
        "file_sha256": bounded_chain_meta["file_sha256"],
        "config_fingerprint": bounded_chain_payload[
            "config_fingerprint"
        ],
    }
    bounded_authorization_payload = {
        "schema_version": "cure-lite-v24-test-bounded-authorization-v1",
        "chain_config_fingerprint": bounded_chain_payload[
            "config_fingerprint"
        ],
        "OOF4_decision_fingerprint": oof_decision.decision_fingerprint,
        "cache_receipt_fingerprint": (
            bounded_full_cache_token.receipt_fingerprint
        ),
    }
    _write_immutable_json(
        bounded_output / "authorization.json",
        bounded_authorization_payload,
    )
    bounded_authorization_fp = stable_fingerprint(
        bounded_authorization_payload
    )
    bounded_intent = {
        "execution_kind": "paired_bounded400_D_R_training",
        "split": "D_R",
        "requested_device": "cpu",
        "output_directory": str(bounded_output),
        "seed": 42,
        "epochs": 10,
        "steps_per_epoch": 40,
        "optimizer_steps_authorized_per_arm": BOUNDED_UPDATES,
        "parameter_updates_authorized_per_arm": BOUNDED_UPDATES,
        "training_invocations_authorized_per_arm": 1,
        "from_scratch": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "D_V_materialization_intended": False,
        "D_T_materialization_intended": False,
    }
    bounded_run_start_path = bounded_output / "run_start.json"
    bounded_run_start_body = {
        "schema_version": (
            "cure-lite-v24-gcr-pacre-paired-bounded400-"
            "persistent-run-start-v1"
        ),
        "protocol_id": "irstd1k-gcr-pacre-v24-evidence-v1",
        "path_policy": (
            "fixed_runtime_root_bounded_paired_bounded400_"
            "run_start_json_v1"
        ),
        "marker_path": str(bounded_run_start_path),
        "stage_id": "paired_bounded400",
        "chain_config": bounded_chain_artifact,
        "authorization_fingerprint": bounded_authorization_fp,
        "OOF4_decision_fingerprint": oof_decision.decision_fingerprint,
        "access_audit_receipt_fingerprint": (
            bounded_access.receipt_fingerprint
        ),
        "full_D_R_cache_artifact": bounded_cache_binding,
        "source_closure": {
            "schema_version": GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
            "fingerprint": source_closure_fp,
            "source_hashes": dict(source_hashes),
        },
        "intent": bounded_intent,
        "intent_fingerprint": stable_fingerprint(bounded_intent),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    bounded_run_start_payload = {
        **bounded_run_start_body,
        "marker_fingerprint": stable_fingerprint(
            bounded_run_start_body
        ),
    }
    bounded_run_start_artifact = {
        **_write_immutable_json(
            bounded_run_start_path,
            bounded_run_start_payload,
        ),
        "marker_fingerprint": bounded_run_start_payload[
            "marker_fingerprint"
        ],
        "payload": bounded_run_start_payload,
    }
    bounded_schedule = build_coverage_state_training_schedule(
        formal_scalar_cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=10,
            steps_per_epoch=40,
        ),
    )
    schedule_path = bounded_output / "schedule.json"
    _write_immutable_json(
        schedule_path,
        bounded_schedule.canonical_payload(),
    )
    schedule_fp = bounded_schedule.schedule_fingerprint
    bounded_batch_sequence_fp = stable_fingerprint(
        [
            selection.canonical_payload()
            for selection in bounded_schedule.selections
        ]
    )
    paired_population_fp = semantic_formal_cache_fp
    bounded_arms: dict[str, object] = {}
    for arm, role in (
        (GCR_PACRE_CONTROL_ARM, "control"),
        (GCR_PACRE_CANDIDATE_ARM, "candidate"),
    ):
        artifact_path = bounded_output / (
            "control_terminal.safetensors"
            if arm == GCR_PACRE_CONTROL_ARM
            else "candidate_terminal.safetensors"
        )
        initial_model = _frozen_seeded_model(arm)
        initial_fp = coverage_state_model_fingerprint(initial_model)
        initial_parameter_fp = _bounded_initial_parameter_fingerprint(
            initial_model
        )
        terminal_model = _subnormal_terminal_model(arm)
        saved_terminal = _save_test_terminal_safetensors(
            artifact_path,
            model=terminal_model,
            role=role,
            seed=42,
            run="paired_bounded400",
        )
        final_fp = str(saved_terminal["model_fingerprint"])
        terminal_artifact = {
            key: saved_terminal[key]
            for key in (
                "path",
                "size_bytes",
                "file_sha256",
                "model_fingerprint",
            )
        }
        mechanical_metrics = mechanically_recompute_bounded_arm(
            arm=arm,
            terminal_artifact_path=str(saved_terminal["path"]),
            expected_initial_model_fingerprint=initial_fp,
            expected_final_model_fingerprint=final_fp,
            expected_initial_parameter_fingerprint=initial_parameter_fp,
            full_d_r_cache_artifact=bounded_full_cache_token,
            requested_device="cpu",
        )
        bounded_arms[arm] = {
            "role": role,
            "seed": 42,
            "epochs": 10,
            "steps_per_epoch": 40,
            "completed_updates": BOUNDED_UPDATES,
            "training_invocations": 1,
            "from_scratch": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "checkpoint_policy": "final_only",
            "optimizer_state_initial_empty": True,
            "population_fingerprint": paired_population_fp,
            "schedule_fingerprint": schedule_fp,
            "batch_sequence_fingerprint": bounded_batch_sequence_fp,
            "initial_shared_parameter_fingerprint": initial_parameter_fp,
            "PMOPE_fingerprint": _fp("PMOPE"),
            "Adam_policy_fingerprint": _fp("Adam"),
            "dtype_device_policy_fingerprint": _fp("float32-cpu"),
            "source_hashes": dict(source_hashes),
            "cache_fingerprint": semantic_formal_cache_fp,
            "neutral_payload_fingerprint": (
                bounded_full_cache_token.neutral_payload_fingerprint
            ),
            "cache_instance_id": f"bounded-{arm}-cache-instance",
            "rng_instance_id": f"bounded-{arm}-rng",
            "module_instance_id": f"bounded-{arm}-module",
            "optimizer_instance_id": f"bounded-{arm}-optimizer",
            "parameter_storage_ids": [f"bounded-{arm}-storage"],
            "initial_model_fingerprint": initial_fp,
            "final_model_fingerprint": final_fp,
            "terminal_artifact": terminal_artifact,
            "finite_audit": {},
            "metrics": mechanical_metrics,
        }
    bounded_trace_payload = build_training_trace_payload(
        stage_id="paired_bounded400",
        authorization_fingerprint=bounded_authorization_fp,
        schedule=bounded_schedule,
        arm_names=(
            GCR_PACRE_CONTROL_ARM,
            GCR_PACRE_CANDIDATE_ARM,
        ),
        terminal_model_fingerprints={
            arm: str(value["final_model_fingerprint"])
            for arm, value in bounded_arms.items()
        },
        raw_rows=_test_trace_rows(
            bounded_schedule,
            arm_names=(
                GCR_PACRE_CONTROL_ARM,
                GCR_PACRE_CANDIDATE_ARM,
            ),
        ),
    )
    bounded_trace_artifact = save_training_trace_new(
        bounded_output / "training_trace.json",
        bounded_trace_payload,
    )
    for arm in (
        GCR_PACRE_CONTROL_ARM,
        GCR_PACRE_CANDIDATE_ARM,
    ):
        bounded_arms[arm]["finite_audit"] = trace_finite_audit(
            bounded_trace_payload,
            arm=arm,
        )
    bounded_body = {
        "schema_version": (
            "cure-lite-v24-gcr-pacre-paired-bounded400-receipt-v6"
        ),
        "budget": {
            "epochs": 10,
            "steps_per_epoch": 40,
            "updates": BOUNDED_UPDATES,
            "training_invocations_per_arm": 1,
        },
        "prerequisites": {
            "dataset_free_receipt_fingerprint": dataset_free_fp,
            "D_R_structural_receipt_fingerprint": structural_fp,
            "OOF4_decision_fingerprint": oof_decision.decision_fingerprint,
        },
        "access_audit_receipt_fingerprint": (
            bounded_access.receipt_fingerprint
        ),
        "paired_population_fingerprint": paired_population_fp,
        "full_D_R_cache_materialization": (
            bounded_full_cache_token.payload
        ),
        "run_start_artifact": bounded_run_start_artifact,
        "schedule_artifact": {
            "path": str(schedule_path.resolve()),
            "size_bytes": schedule_path.stat().st_size,
            "file_sha256": _sha(schedule_path),
            "schedule_fingerprint": schedule_fp,
        },
        "training_trace_artifact": bounded_trace_artifact,
        "arms": bounded_arms,
        "paired_diagnostics": _paired_bounded_diagnostics(
            bounded_arms,
            bounded_trace_payload,
        ),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    bounded_receipt = _seal(bounded_body)
    bounded_evidence = validate_paired_bounded_receipt(
        bounded_receipt,
        oof_decision=oof_decision,
        access_audit=bounded_access,
        full_d_r_cache_artifact=bounded_full_cache_token,
        dataset_free_receipt_fingerprint=dataset_free_fp,
        d_r_structural_receipt_fingerprint=structural_fp,
        repository_root=REPO_ROOT,
    )
    bounded_decision = decide_paired_bounded400(bounded_evidence)
    assert bounded_decision["gate_passed"] is True

    formal_cache_path = work / "formal-seed42-full-D_R-cache.pt"
    formal_cache_token = save_formal_cache_neutral_artifact_new(
        formal_scalar_cache,
        formal_cache_path.resolve(),
        cache_id="formal800-seed42-primary-full-D_R-cache",
    )
    formal_access_receipt = _access_receipt(
        "formal800_seed42_primary",
        ["D_R"],
        [
            {
                "split": "D_R",
                "logical_id": formal_cache_token.cache_id,
                "purpose": "Formal800_seed42_primary_training_cache",
                "source_fingerprint": formal_cache_token.file_sha256,
            }
        ],
    )
    formal_access = verify_access_audit_receipt(
        formal_access_receipt,
        expected_stage_id="formal800_seed42_primary",
        allowed_splits=["D_R"],
    )
    formal43_cache_path = work / "formal-seed43-full-D_R-cache.pt"
    formal43_cache_token = save_formal_cache_neutral_artifact_new(
        formal_scalar_cache,
        formal43_cache_path.resolve(),
        cache_id=(
            "formal800-seed43-training_integrity_only-full-D_R-cache"
        ),
    )
    formal43_access_receipt = _access_receipt(
        "formal800_seed43_training_integrity_only",
        ["D_R"],
        [
            {
                "split": "D_R",
                "logical_id": formal43_cache_token.cache_id,
                "purpose": "Formal800_seed43_training_integrity_cache",
                "source_fingerprint": formal43_cache_token.file_sha256,
            }
        ],
    )
    formal43_access = verify_access_audit_receipt(
        formal43_access_receipt,
        expected_stage_id="formal800_seed43_training_integrity_only",
        allowed_splits=["D_R"],
    )
    formal_runtime = evidence_runtime / "formal"
    formal42_output = formal_runtime / "seed42_primary"
    formal43_output = formal_runtime / "seed43_training_integrity_only"
    formal42_output.mkdir(parents=True)
    formal43_output.mkdir()
    formal_chain_path = formal_runtime / "execution_chain_config.json"

    def _formal_cache_binding(token: object) -> dict[str, object]:
        return {
            "receipt_fingerprint": token.receipt_fingerprint,
            "cache_id": token.cache_id,
            "path": token.path,
            "file_sha256": token.file_sha256,
            "device": token.device,
            "inode": token.inode,
            "hardlink_count": token.hardlink_count,
            "semantic_cache_fingerprint": (
                token.semantic_cache_fingerprint
            ),
            "neutral_payload_fingerprint": (
                token.neutral_payload_fingerprint
            ),
        }

    def _formal_run_binding(
        *,
        seed: int,
        role: str,
        output: Path,
        access: object,
        cache: object,
    ) -> dict[str, object]:
        return {
            "seed": seed,
            "role": role,
            "stage_id": f"formal800_seed{seed}_{role}",
            "requested_device": "cpu",
            "output_directory": str(output),
            "run_start_marker_path": str(output / "run_start.json"),
            "authorization_artifact_path": str(
                output / "authorization.json"
            ),
            "schedule_artifact_path": str(output / "schedule.json"),
            "terminal_artifact_directory": str(output / "terminal"),
            "evidence_artifact_path": str(
                output / "formal800_evidence.json"
            ),
            "access_audit_receipt_fingerprint": (
                access.receipt_fingerprint
            ),
            "access_audit_receipt": access.payload,
            "cache_artifact": _formal_cache_binding(cache),
            "selection_effect": (
                "predeclared_primary" if seed == 42 else "none"
            ),
            "may_replace_seed42_primary": False,
            "D_V_execution_authorized": False,
            "D_T_execution_authorized": False,
        }

    formal_chain_body = {
        "schema_version": (
            "cure-lite-v24-gcr-pacre-formal800-chain-config-v1"
        ),
        "protocol_id": "irstd1k-gcr-pacre-v24-evidence-v1",
        "path_policy": (
            "fixed_runtime_root_seed_role_directory_run_start_json_v1"
        ),
        "repository_root": str(REPO_ROOT),
        "runtime_root": str(evidence_runtime),
        "chain_config_path": str(formal_chain_path),
        "source_closure_schema": GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
        "source_hashes": dict(source_hashes),
        "source_closure_fingerprint": source_closure_fp,
        "predecessors": {
            "dataset_free_receipt_fingerprint": dataset_free_fp,
            "D_R_structural_receipt_fingerprint": structural_fp,
            "OOF4_decision_fingerprint": oof_decision.decision_fingerprint,
            "paired_bounded400_decision_fingerprint": (
                bounded_decision.decision_fingerprint
            ),
        },
        "runs": {
            "seed42_primary": _formal_run_binding(
                seed=42,
                role="primary",
                output=formal42_output,
                access=formal_access,
                cache=formal_cache_token,
            ),
            "seed43_training_integrity_only": _formal_run_binding(
                seed=43,
                role="training_integrity_only",
                output=formal43_output,
                access=formal43_access,
                cache=formal43_cache_token,
            ),
        },
        "formal_pair_receipt_path": str(
            formal_runtime / "formal800_pair_receipt.json"
        ),
        "budget": {
            "epochs_per_seed": 800,
            "steps_per_epoch": 40,
            "updates_per_seed": FORMAL_UPDATES,
            "training_invocations_per_seed": 1,
        },
        "attempt_policy": {
            "from_scratch": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "persistent_O_EXCL_run_start_required": True,
            "run_start_marker_never_removed_after_creation": True,
            "checkpoint_policy": "final_only",
            "seed43_selection_effect": "none",
        },
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    formal_chain_payload = {
        **formal_chain_body,
        "config_fingerprint": stable_fingerprint(formal_chain_body),
    }
    formal_chain_meta = _write_immutable_json(
        formal_chain_path,
        formal_chain_payload,
    )
    formal_chain_artifact = {
        "path": formal_chain_meta["path"],
        "file_sha256": formal_chain_meta["file_sha256"],
        "config_fingerprint": formal_chain_payload["config_fingerprint"],
    }
    formal_source_closure = {
        "schema_version": GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
        "source_hashes": dict(source_hashes),
        "source_closure_fingerprint": source_closure_fp,
    }
    formal_marker_source_closure = {
        "schema_version": GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
        "fingerprint": source_closure_fp,
        "source_hashes": dict(source_hashes),
    }

    def _formal_run_start_artifact(
        *,
        seed: int,
        role: str,
        output: Path,
        access: object,
        cache: object,
        process_instance_fingerprint: str,
        authorization_fingerprint: str,
    ) -> dict[str, object]:
        marker_path = output / "run_start.json"
        intent = {
            "execution_kind": "Formal800_D_R_training",
            "split": "D_R",
            "requested_device": "cpu",
            "output_directory": str(output),
            "epochs": 800,
            "steps_per_epoch": 40,
            "optimizer_steps_authorized": FORMAL_UPDATES,
            "parameter_updates_authorized": FORMAL_UPDATES,
            "training_invocations_authorized": 1,
            "from_scratch": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "D_V_materialization_intended": False,
            "D_T_materialization_intended": False,
        }
        cache_binding = _formal_cache_binding(cache)
        body = {
            "schema_version": (
                "cure-lite-v24-gcr-pacre-formal800-"
                "persistent-run-start-v2"
            ),
            "protocol_id": "irstd1k-gcr-pacre-v24-evidence-v1",
            "path_policy": (
                "fixed_runtime_root_seed_role_directory_run_start_json_v1"
            ),
            "marker_path": str(marker_path),
            "seed": seed,
            "role": role,
            "stage_id": f"formal800_seed{seed}_{role}",
            "process_instance_fingerprint": (
                process_instance_fingerprint
            ),
            "chain_config": formal_chain_artifact,
            "authorization_fingerprint": authorization_fingerprint,
            "access_audit_receipt_fingerprint": (
                access.receipt_fingerprint
            ),
            "cache_artifact": {
                "path": cache_binding["path"],
                "file_sha256": cache_binding["file_sha256"],
                "receipt_fingerprint": cache_binding[
                    "receipt_fingerprint"
                ],
                "semantic_cache_fingerprint": cache_binding[
                    "semantic_cache_fingerprint"
                ],
                "neutral_payload_fingerprint": cache_binding[
                    "neutral_payload_fingerprint"
                ],
            },
            "source_closure": deepcopy(formal_marker_source_closure),
            "intent": intent,
            "intent_fingerprint": stable_fingerprint(intent),
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
        payload = {
            **body,
            "marker_fingerprint": stable_fingerprint(body),
        }
        return {
            **_write_immutable_json(marker_path, payload),
            "marker_fingerprint": payload["marker_fingerprint"],
            "payload": payload,
        }

    formal42_process_instance_fp = _fp(
        "formal-seed42-independent-process-instance"
    )
    formal43_process_instance_fp = _fp(
        "formal-seed43-independent-process-instance"
    )
    formal42_authorization_payload = {
        "schema_version": "cure-lite-v24-test-formal-authorization-v1",
        "seed": 42,
        "role": "primary",
        "chain_config_fingerprint": formal_chain_payload[
            "config_fingerprint"
        ],
        "cache_receipt_fingerprint": (
            formal_cache_token.receipt_fingerprint
        ),
    }
    _write_immutable_json(
        formal42_output / "authorization.json",
        formal42_authorization_payload,
    )
    formal42_authorization_fp = stable_fingerprint(
        formal42_authorization_payload
    )
    formal43_authorization_payload = {
        "schema_version": "cure-lite-v24-test-formal-authorization-v1",
        "seed": 43,
        "role": "training_integrity_only",
        "chain_config_fingerprint": formal_chain_payload[
            "config_fingerprint"
        ],
        "cache_receipt_fingerprint": (
            formal43_cache_token.receipt_fingerprint
        ),
    }
    _write_immutable_json(
        formal43_output / "authorization.json",
        formal43_authorization_payload,
    )
    formal43_authorization_fp = stable_fingerprint(
        formal43_authorization_payload
    )
    formal_run_start_artifact = _formal_run_start_artifact(
        seed=42,
        role="primary",
        output=formal42_output,
        access=formal_access,
        cache=formal_cache_token,
        process_instance_fingerprint=formal42_process_instance_fp,
        authorization_fingerprint=formal42_authorization_fp,
    )
    formal43_run_start_artifact = _formal_run_start_artifact(
        seed=43,
        role="training_integrity_only",
        output=formal43_output,
        access=formal43_access,
        cache=formal43_cache_token,
        process_instance_fingerprint=formal43_process_instance_fp,
        authorization_fingerprint=formal43_authorization_fp,
    )
    formal_schedule = build_coverage_state_training_schedule(
        formal_scalar_cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=800,
            steps_per_epoch=40,
        ),
    )
    formal_schedule_path = formal42_output / "schedule.json"
    _write_immutable_json(
        formal_schedule_path,
        formal_schedule.canonical_payload(),
    )
    formal_schedule_fp = formal_schedule.schedule_fingerprint
    formal_contract = coverage_state_model_contract_payload(
        build_formal_gcr_pacre_training_model()
    )
    formal_initial_model = _frozen_seeded_model(
        GCR_PACRE_CANDIDATE_ARM,
        seed=42,
    )
    parameters = _initial_parameter_rows(formal_initial_model)
    initial_model_fp = coverage_state_model_fingerprint(
        formal_initial_model
    )
    formal_terminal_model = _subnormal_terminal_model(
        GCR_PACRE_CANDIDATE_ARM,
        seed=42,
    )
    (formal42_output / "terminal").mkdir()
    formal_artifact_path = (
        formal42_output / "terminal/model.safetensors"
    )
    saved_formal_terminal = _save_test_terminal_safetensors(
        formal_artifact_path,
        model=formal_terminal_model,
        role="primary",
        seed=42,
        run="formal800-seed42",
    )
    final_model_fp = str(saved_formal_terminal["model_fingerprint"])
    formal_artifact = {
        key: saved_formal_terminal[key]
        for key in (
            "path",
            "size_bytes",
            "file_sha256",
            "model_fingerprint",
        )
    }
    (
        formal_terminal_evaluation,
        formal_terminal_evaluation_fp,
    ) = mechanically_recompute_formal_terminal(
        terminal_artifact_path=str(saved_formal_terminal["path"]),
        expected_final_model_fingerprint=final_model_fp,
        cache_artifact=formal_cache_token,
        requested_device="cpu",
        seed=42,
        role="primary",
    )
    formal_trace_payload = build_training_trace_payload(
        stage_id="formal800_seed42_primary",
        authorization_fingerprint=formal42_authorization_fp,
        schedule=formal_schedule,
        arm_names=("primary",),
        terminal_model_fingerprints={
            "primary": final_model_fp,
        },
        raw_rows=_test_trace_rows(
            formal_schedule,
            arm_names=("primary",),
        ),
    )
    formal_trace_artifact = save_training_trace_new(
        formal42_output / "training_trace.json",
        formal_trace_payload,
    )
    contract_json = canonical_json(formal_contract)
    policy = {
        "role": "primary",
        "seed": 42,
        "scope": "D_R_formal_800",
        "budget": {
            "epochs": 800,
            "steps_per_epoch": 40,
            "updates": FORMAL_UPDATES,
        },
        "objective": "pmope_joint",
        "optimizer_fqcn": "torch.optim.adam.Adam",
        "learning_rate_hex": (0.001).hex(),
        "weight_decay_hex": (0.0).hex(),
        "betas_hex": [(0.9).hex(), (0.999).hex()],
        "epsilon_hex": (1.0e-8).hex(),
        "training_invocations": 1,
        "from_scratch": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "checkpoint_policy": "final_only",
        "D_V_execution_authorized": False,
        "D_T_execution_authorized": False,
        "eligible_for_future_D_V_authorization_after_all_external_prerequisites": True,
        "eligible_for_future_D_T_authorization_after_all_external_prerequisites": True,
    }
    policy_json = canonical_json(policy)
    core_source_hashes = {
        relative: _sha(REPO_ROOT / relative)
        for relative in (
            "cure_lite_v24/gcr_pacre.py",
            "cure_lite_v24/factory.py",
            "cure_lite_v24/training.py",
        )
    }
    training_receipt = {
        "schema_version": (
            "cure-lite-v24-gcr-pacre-pmope-training-receipt-v1"
        ),
        "role": "primary",
        "evaluation_role": "primary",
        "seed": 42,
        "scope": "D_R_formal_800",
        "objective": "pmope_joint",
        "optimizer_fqcn": "torch.optim.adam.Adam",
        "policy_json": policy_json,
        "policy_fingerprint": sha256(policy_json.encode()).hexdigest(),
        "model": {
            "model_fqcn": (
                "cure_lite_v24.gcr_pacre."
                "CURELiteGatedCommonResidualPACRELevelSet"
            ),
            "config_fqcn": (
                "cure_lite_v24.gcr_pacre.CoverageStateGCRPACREConfig"
            ),
            "contract_json": contract_json,
            "contract_fingerprint": sha256(
                contract_json.encode()
            ).hexdigest(),
            "parameter_count": 64064,
            "initial_parameters": parameters,
            "initial_parameter_state_fingerprint": stable_fingerprint(
                parameters
            ),
            "initial_fingerprint": initial_model_fp,
            "final_fingerprint": final_model_fp,
        },
        "source_hashes": core_source_hashes,
        "cache_fingerprint": semantic_formal_cache_fp,
        "schedule_fingerprint": formal_schedule_fp,
        "optimizer_config_fingerprint": _fp("formal-optimizer"),
        "training_result_fingerprint": _fp("formal-result"),
        "budget": {
            "epochs": 800,
            "steps_per_epoch": 40,
            "updates": FORMAL_UPDATES,
            "training_invocations": 1,
        },
        "compute": {
            "completed_updates": FORMAL_UPDATES,
            "forward_calls": FORMAL_UPDATES,
            "backward_calls": FORMAL_UPDATES,
            "optimizer_steps": FORMAL_UPDATES,
        },
        "from_scratch": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "checkpoint_policy": "final_only",
        "eligible_for_future_D_V_authorization_after_all_external_prerequisites": True,
        "eligible_for_future_D_T_authorization_after_all_external_prerequisites": True,
        "D_V_execution_authorized": False,
        "D_T_execution_authorized": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "selection_effect": "predeclared_primary",
        "may_replace_seed42_primary": False,
    }
    formal_body = {
        "schema_version": "cure-lite-v24-gcr-pacre-formal800-evidence-v6",
        "seed": 42,
        "evaluation_role": "primary",
        "prerequisites": {
            "dataset_free_receipt_fingerprint": dataset_free_fp,
            "D_R_structural_receipt_fingerprint": structural_fp,
            "OOF4_decision_fingerprint": oof_decision.decision_fingerprint,
            "paired_bounded400_decision_fingerprint": (
                bounded_decision.decision_fingerprint
            ),
        },
        "access_audit_receipt_fingerprint": (
            formal_access.receipt_fingerprint
        ),
        "training_receipt": training_receipt,
        "training_receipt_fingerprint": stable_fingerprint(training_receipt),
        "finite_audit": trace_finite_audit(
            formal_trace_payload,
            arm="primary",
        ),
        "cache_artifact": formal_cache_token.payload,
        "run_start_artifact": formal_run_start_artifact,
        "schedule_artifact": _formal_schedule_meta(
            formal_schedule_path,
            schedule_fingerprint=formal_schedule_fp,
            semantic_cache_fingerprint=semantic_formal_cache_fp,
            seed=42,
        ),
        "training_trace_artifact": formal_trace_artifact,
        "terminal_artifact": formal_artifact,
        "terminal_D_R_evaluation": formal_terminal_evaluation,
        "terminal_D_R_evaluation_fingerprint": (
            formal_terminal_evaluation_fp
        ),
        "source_closure": deepcopy(formal_source_closure),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    formal_receipt = _seal(formal_body)
    formal = validate_formal_training_receipt(
        formal_receipt,
        expected_seed=42,
        expected_role="primary",
        oof_decision=oof_decision,
        bounded_decision=bounded_decision,
        access_audit=formal_access,
        cache_artifact=formal_cache_token,
        dataset_free_receipt_fingerprint=dataset_free_fp,
        d_r_structural_receipt_fingerprint=structural_fp,
        repository_root=REPO_ROOT,
    )

    training43 = deepcopy(training_receipt)
    policy43 = deepcopy(policy)
    policy43["role"] = "training_integrity_only"
    policy43["seed"] = 43
    policy43[
        "eligible_for_future_D_V_authorization_after_all_external_prerequisites"
    ] = False
    policy43[
        "eligible_for_future_D_T_authorization_after_all_external_prerequisites"
    ] = False
    policy43_json = canonical_json(policy43)
    training43["role"] = "training_integrity_only"
    training43["evaluation_role"] = "training_integrity_only"
    training43["seed"] = 43
    training43["policy_json"] = policy43_json
    training43["policy_fingerprint"] = sha256(
        policy43_json.encode()
    ).hexdigest()
    formal43_initial_model = _frozen_seeded_model(
        GCR_PACRE_CANDIDATE_ARM,
        seed=43,
    )
    parameters43 = _initial_parameter_rows(formal43_initial_model)
    training43["model"]["initial_parameters"] = parameters43
    training43["model"]["initial_parameter_state_fingerprint"] = (
        stable_fingerprint(parameters43)
    )
    training43["model"]["initial_fingerprint"] = (
        coverage_state_model_fingerprint(formal43_initial_model)
    )
    formal43_terminal_model = _subnormal_terminal_model(
        GCR_PACRE_CANDIDATE_ARM,
        seed=43,
    )
    (formal43_output / "terminal").mkdir()
    formal43_artifact_path = (
        formal43_output / "terminal/model.safetensors"
    )
    saved_formal43_terminal = _save_test_terminal_safetensors(
        formal43_artifact_path,
        model=formal43_terminal_model,
        role="training_integrity_only",
        seed=43,
        run="formal800-seed43",
    )
    training43["model"]["final_fingerprint"] = saved_formal43_terminal[
        "model_fingerprint"
    ]
    training43["cache_fingerprint"] = semantic_formal_cache_fp
    training43["training_result_fingerprint"] = _fp("formal43-result")
    training43[
        "eligible_for_future_D_V_authorization_after_all_external_prerequisites"
    ] = False
    training43[
        "eligible_for_future_D_T_authorization_after_all_external_prerequisites"
    ] = False
    training43["selection_effect"] = "none"
    formal43_schedule = build_coverage_state_training_schedule(
        formal_scalar_cache,
        CoverageStateScheduleConfig(
            seed=43,
            epochs=800,
            steps_per_epoch=40,
        ),
    )
    formal43_schedule_path = formal43_output / "schedule.json"
    _write_immutable_json(
        formal43_schedule_path,
        formal43_schedule.canonical_payload(),
    )
    training43["schedule_fingerprint"] = (
        formal43_schedule.schedule_fingerprint
    )
    formal43_artifact = {
        key: saved_formal43_terminal[key]
        for key in (
            "path",
            "size_bytes",
            "file_sha256",
            "model_fingerprint",
        )
    }
    (
        formal43_terminal_evaluation,
        formal43_terminal_evaluation_fp,
    ) = mechanically_recompute_formal_terminal(
        terminal_artifact_path=str(saved_formal43_terminal["path"]),
        expected_final_model_fingerprint=str(
            training43["model"]["final_fingerprint"]
        ),
        cache_artifact=formal43_cache_token,
        requested_device="cpu",
        seed=43,
        role="training_integrity_only",
    )
    formal43_trace_payload = build_training_trace_payload(
        stage_id="formal800_seed43_training_integrity_only",
        authorization_fingerprint=formal43_authorization_fp,
        schedule=formal43_schedule,
        arm_names=("training_integrity_only",),
        terminal_model_fingerprints={
            "training_integrity_only": str(
                training43["model"]["final_fingerprint"]
            ),
        },
        raw_rows=_test_trace_rows(
            formal43_schedule,
            arm_names=("training_integrity_only",),
        ),
    )
    formal43_trace_artifact = save_training_trace_new(
        formal43_output / "training_trace.json",
        formal43_trace_payload,
    )
    formal43_body = {
        "schema_version": "cure-lite-v24-gcr-pacre-formal800-evidence-v6",
        "seed": 43,
        "evaluation_role": "training_integrity_only",
        "prerequisites": deepcopy(formal_body["prerequisites"]),
        "access_audit_receipt_fingerprint": (
            formal43_access.receipt_fingerprint
        ),
        "training_receipt": training43,
        "training_receipt_fingerprint": stable_fingerprint(training43),
        "finite_audit": trace_finite_audit(
            formal43_trace_payload,
            arm="training_integrity_only",
        ),
        "cache_artifact": formal43_cache_token.payload,
        "run_start_artifact": formal43_run_start_artifact,
        "schedule_artifact": _formal_schedule_meta(
            formal43_schedule_path,
            schedule_fingerprint=training43["schedule_fingerprint"],
            semantic_cache_fingerprint=semantic_formal_cache_fp,
            seed=43,
        ),
        "training_trace_artifact": formal43_trace_artifact,
        "terminal_artifact": formal43_artifact,
        "terminal_D_R_evaluation": formal43_terminal_evaluation,
        "terminal_D_R_evaluation_fingerprint": (
            formal43_terminal_evaluation_fp
        ),
        "source_closure": deepcopy(formal_source_closure),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    formal43_receipt = _seal(formal43_body)
    formal43 = validate_formal_training_receipt(
        formal43_receipt,
        expected_seed=43,
        expected_role="training_integrity_only",
        oof_decision=oof_decision,
        bounded_decision=bounded_decision,
        access_audit=formal43_access,
        cache_artifact=formal43_cache_token,
        dataset_free_receipt_fingerprint=dataset_free_fp,
        d_r_structural_receipt_fingerprint=structural_fp,
        repository_root=REPO_ROOT,
    )
    formal_pair = verify_formal800_training_independence(formal, formal43)

    d_t_payload = json.loads(
        (PROTOCOL_ROOT / "D_T_preregistration.json").read_text()
    )
    d_t_prereg = validate_d_t_preregistration(
        d_t_payload,
        repository_root=REPO_ROOT,
    )
    binding_access_receipt = _access_receipt(
        "d_t_seed42_model_binding",
        [],
        [],
    )
    binding_access = verify_access_audit_receipt(
        binding_access_receipt,
        expected_stage_id="d_t_seed42_model_binding",
        allowed_splits=[],
    )
    binding_body = {
        "schema_version": "cure-lite-v24-D_T-seed42-model-binding-v1",
        "protocol_preregistration_fingerprint": (
            d_t_prereg.protocol_preregistration_fingerprint
        ),
        "D_T_preregistration_fingerprint": (
            d_t_prereg.preregistration_fingerprint
        ),
        "Formal800_evidence_receipt_fingerprint": formal.receipt_fingerprint,
        "Formal800_pair_fingerprint": formal_pair.pair_fingerprint,
        "training_receipt_fingerprint": formal.training_receipt_fingerprint,
        "seed": 42,
        "role": "primary",
        "final_model_fingerprint": formal.final_model_fingerprint,
        "model_contract_fingerprint": formal.model_contract_fingerprint,
        "terminal_artifact": formal_artifact,
        "access_audit_receipt_fingerprint": (
            binding_access.receipt_fingerprint
        ),
        "events": {
            "Formal800_seed42_terminal_sealed": 100,
            "D_T_model_binding_sealed": 101,
            "D_V_authorization_created": None,
        },
        "D_V_authorization_created": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    binding_receipt = _seal(binding_body)
    binding = validate_d_t_seed42_model_binding(
        binding_receipt,
        d_t_preregistration=d_t_prereg,
        formal_seed42=formal,
        formal_pair=formal_pair,
        access_audit=binding_access,
    )

    baseline_binding = json.loads(
        (PROTOCOL_ROOT / "exact_baseline_ledger_binding.json").read_text()
    )
    baseline_access_receipt = _access_receipt(
        "exact_baseline_envelope",
        ["SEALED_D_V_AGGREGATE_METADATA"],
        [
            {
                "split": "SEALED_D_V_AGGREGATE_METADATA",
                "logical_id": baseline_binding["source_repo_path"],
                "purpose": "sealed_aggregate_baseline_envelope",
                "source_fingerprint": baseline_binding[
                    "source_file_sha256"
                ],
            }
        ],
    )
    baseline_access = verify_access_audit_receipt(
        baseline_access_receipt,
        expected_stage_id="exact_baseline_envelope",
        allowed_splits=["SEALED_D_V_AGGREGATE_METADATA"],
    )
    baseline = load_exact_baseline_envelope(
        PROTOCOL_ROOT / "exact_baseline_ledger_binding.json",
        repository_root=REPO_ROOT,
        access_audit=baseline_access,
    )
    envelope = baseline.payload["envelope"]
    d_v_access_receipt = _access_receipt(
        "v24_D_V_one_shot",
        ["D_V"],
        [
            {
                "split": "D_V",
                "logical_id": "candidate-aggregate",
                "purpose": "authorized_one_shot_aggregate_evaluation",
                "source_fingerprint": _fp("candidate-aggregate"),
            }
        ],
    )
    d_v_access = verify_access_audit_receipt(
        d_v_access_receipt,
        expected_stage_id="v24_D_V_one_shot",
        allowed_splits=["D_V"],
    )
    d_v_body = {
        "schema_version": (
            "cure-lite-v24-gcr-pacre-D_V-candidate-evidence-v1"
        ),
        "model_binding_receipt_fingerprint": binding.receipt_fingerprint,
        "final_model_fingerprint": binding.final_model_fingerprint,
        "access_audit_receipt_fingerprint": d_v_access.receipt_fingerprint,
        "candidate_metrics": {
            "true_targets": int(envelope["true_targets"]) + 1,
            "recovered_anchor_misses": (
                int(envelope["recovered_anchor_misses"]) + 1
            ),
            "mIoU": float(envelope["mIoU"]),
            "nIoU": float(envelope["nIoU"]),
            "retention": 1.0,
            "pixel_fa": 0.0,
            "raw_background_fa": 0.0,
            "fp_components_per_mp": 0.0,
            "budget_violation": False,
        },
        "D_T_payload_accessed": False,
    }
    d_v_receipt = _seal(d_v_body)
    d_v_evidence = verify_dv_candidate_evidence(
        d_v_receipt,
        model_binding=binding,
        access_audit=d_v_access,
    )
    return locals()


def test_root_mapping_and_real_frozen_split_are_exact(
    protocol_chain: dict[str, object],
) -> None:
    split = protocol_chain["split"]
    assert isinstance(split, VerifiedOOF4Split)
    assert len(split.root_by_sample) == 160
    assert len(set(split.root_by_sample.values())) == 156
    assert split.plan_fingerprint == (
        "40727e0ae728aed0cceb9e11244645bddf5da7366507826791f793948f156089"
    )
    rows = [
        {"sample_id": "a", "split": "D_R", "scene_id": "x"},
        {
            "sample_id": "b",
            "split": "D_R",
            "scene_id": "x",
            "sequence_id": "y",
        },
        {"sample_id": "c", "split": "D_R", "sequence_id": "y"},
        *[
            {"sample_id": f"s{i}", "split": "D_R", "group_id": f"g{i}"}
            for i in range(4)
        ],
    ]
    roots = derive_root_source_ids(rows)
    assert roots["a"] == roots["b"] == roots["c"]
    assert deterministic_oof4_plan(roots) == deterministic_oof4_plan(
        dict(reversed(tuple(roots.items())))
    )


def test_split_tamper_and_forged_plan_are_rejected(
    protocol_chain: dict[str, object],
) -> None:
    payload = deepcopy(protocol_chain["split_payload"])
    payload["plan"]["fold_summaries"][0][
        "held_out_sample_ids_fingerprint"
    ] = SHA_A
    payload = _reseal(payload)
    with pytest.raises(ValueError, match="compact OOF4 plan"):
        verify_oof4_split_preregistration(
            payload,
            repository_root=REPO_ROOT,
        )
    with pytest.raises(TypeError, match="verified_split"):
        validate_oof_fold_execution_receipt(
            protocol_chain["fold_receipts"][0],
            protocol_chain["split"].plan,
            access_audit=protocol_chain["fold_access_tokens"][0],
            execution_authorization=protocol_chain[
                "execution_authorization"
            ],
            repository_root=REPO_ROOT,
        )


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda row: row.update(
                {"reader_allowlist": ["unauthorized-reader"]}
            ),
            "isolation metadata",
        ),
        (
            lambda row: row.update({"creation_phase": "after_holdout_open"}),
            "isolation metadata",
        ),
        (
            lambda row: row.update({"file_sha256": SHA_A}),
            "bytes changed",
        ),
    ],
)
def test_fold_cache_real_file_metadata_tamper_fails(
    protocol_chain: dict[str, object],
    mutate: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    receipt = deepcopy(protocol_chain["fold_receipts"][0])
    entry = receipt["cache_entries"][0]
    mutate(entry)
    if match == "bytes changed":
        artifact_body = {
            "schema_version": (
                "cure-lite-v24-gcr-pacre-oof4-cache-artifact-v1"
            ),
            "cache_id": entry["cache_id"],
            "fold_id": receipt["fold_id"],
            "partition": entry["partition"],
            "arm": entry["arm"],
            "closure_fingerprint": entry["closure_fingerprint"],
            "terminal_seal_fingerprint": entry[
                "terminal_seal_fingerprint"
            ],
            "semantic_payload_fingerprint": entry[
                "semantic_payload_fingerprint"
            ],
            "root_source_ids": entry["root_source_ids"],
            "sample_ids": entry["sample_ids"],
            "realpath": entry["realpath"],
            "device": entry["device"],
            "inode": entry["inode"],
            "size_bytes": entry["size_bytes"],
            "file_sha256": entry["file_sha256"],
            "creation_phase": entry["creation_phase"],
            "creation_event": entry["creation_event"],
            "reader_allowlist": entry["reader_allowlist"],
            "tensor_ledger_fingerprint": entry[
                "tensor_ledger_fingerprint"
            ],
            "fiemap_extent_flags": entry["fiemap_extent_flags"],
            "loader": {
                "torch_load": True,
                "weights_only": True,
                "mmap_used": False,
                "neutral_object_graph": True,
            },
        }
        entry["artifact_fingerprint"] = stable_fingerprint(artifact_body)
    receipt = _reseal(receipt)
    with pytest.raises((ValueError, RuntimeError, PermissionError), match=match):
        validate_oof_fold_execution_receipt(
            receipt,
            protocol_chain["split"],
            access_audit=protocol_chain["fold_access_tokens"][0],
            execution_authorization=protocol_chain[
                "execution_authorization"
            ],
            repository_root=REPO_ROOT,
        )


@pytest.mark.parametrize(
    "field, forged_value",
    [
        ("dataset_fingerprint", SHA_A),
        ("evaluator_fingerprint", SHA_B),
    ],
)
def test_fold_ledgers_cannot_substitute_resealed_artifacts(
    protocol_chain: dict[str, object],
    tmp_path: Path,
    field: str,
    forged_value: str,
) -> None:
    receipt = deepcopy(protocol_chain["fold_receipts"][0])
    artifacts = receipt["evaluation_ledger_artifacts"]
    for arm in OOF_ARMS:
        original = artifacts[arm]
        payload = json.loads(
            Path(original["path"]).read_text(encoding="utf-8")
        )
        payload[field] = forged_value
        payload = _reseal(payload, field="ledger_fingerprint")
        replacement_path = tmp_path / f"{arm}-{field}.json"
        artifacts[arm] = {
            **_write_immutable_json(replacement_path, payload),
            "ledger_fingerprint": payload["ledger_fingerprint"],
        }
    receipt = _reseal(receipt)
    with pytest.raises(
        PermissionError,
        match="hard-link alias|verified held-out cache",
    ):
        validate_oof_fold_execution_receipt(
            receipt,
            protocol_chain["split"],
            access_audit=protocol_chain["fold_access_tokens"][0],
            execution_authorization=protocol_chain[
                "execution_authorization"
            ],
            repository_root=REPO_ROOT,
        )


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("non_sha_pair", "schedule_fingerprint"),
        ("missing_grid_row", "51 rows"),
        ("selector_tamper", "selector policy"),
        ("heldout_leak", "train-only"),
    ],
)
def test_fold_pairing_and_complete_train_only_base_b_fail_closed(
    protocol_chain: dict[str, object],
    mutation: str,
    match: str,
) -> None:
    receipt = deepcopy(protocol_chain["fold_receipts"][0])
    if mutation == "non_sha_pair":
        receipt["training_arms"]["GCR_PACRE_v24"][
            "schedule_fingerprint"
        ] = None
    elif mutation == "missing_grid_row":
        receipt["BaseB_train_fold_selection"]["candidate_rows"].pop()
    elif mutation == "selector_tamper":
        receipt["BaseB_train_fold_selection"]["selector_policy"][0] = "minimize_pd"
    else:
        row = receipt["BaseB_train_fold_selection"]["candidate_rows"][0]
        leaked = list(row["train_sample_ids"])
        leaked[0] = receipt["held_out_sample_ids"][0]
        row["train_sample_ids"] = leaked
    receipt = _reseal(receipt)
    with pytest.raises((ValueError, PermissionError), match=match):
        validate_oof_fold_execution_receipt(
            receipt,
            protocol_chain["split"],
            access_audit=protocol_chain["fold_access_tokens"][0],
            execution_authorization=protocol_chain[
                "execution_authorization"
            ],
            repository_root=REPO_ROOT,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "root",
        "images",
        "denominator",
        "gt",
    ],
)
def test_factual_pool_rejects_any_row_not_in_persisted_artifact(
    protocol_chain: dict[str, object],
    mutation: str,
) -> None:
    rows = deepcopy(protocol_chain["factual_rows_by_fold"][0])
    if mutation == "root":
        rows[0]["root_source_id"] = "forged-root"
    elif mutation == "images":
        rows[0]["sufficient_statistics"]["images"] = 2
    elif mutation == "denominator":
        rows[0]["sufficient_statistics"]["total_pixels"] += 1
    else:
        rows[0]["gt_fingerprint"] = SHA_A
    with pytest.raises(
        PermissionError,
        match="persisted factual artifact",
    ):
        pool_factual_only_rows(
            rows,
            protocol_chain["fold_tokens"][0],
            access_audit=protocol_chain["fold_access_tokens"][0],
        )


def test_oof_decision_requires_verified_pooled_and_gate_receipts(
    protocol_chain: dict[str, object],
) -> None:
    assert protocol_chain["oof_decision"]["gate_passed"] is True
    with pytest.raises(TypeError, match="pooled_evidence"):
        decide_oof4_pooled(
            {"GCR_PACRE_v24": {}},
            gate_path_evidence=protocol_chain["gate"],
        )
    receipt = deepcopy(protocol_chain["gate_receipt"])
    receipt["pooled_evidence_fingerprint"] = SHA_A
    receipt = _reseal(receipt)
    with pytest.raises(PermissionError, match="gate-path"):
        verify_gate_path_receipt(receipt, protocol_chain["pooled"])
    forged = deepcopy(protocol_chain["gate_receipt"])
    forged["field_difference_ledger"][0][
        "forced_G1_output_fingerprint"
    ] = forged["field_difference_ledger"][0][
        "natural_output_fingerprint"
    ]
    forged["field_difference_ledger_fingerprint"] = stable_fingerprint(
        forged["field_difference_ledger"]
    )
    forged = _reseal(forged)
    with pytest.raises(ValueError, match="difference evidence"):
        verify_gate_path_receipt(forged, protocol_chain["pooled"])


@pytest.mark.parametrize("mutation", ["fake_sha", "missing", "extra"])
def test_gate_path_ledger_must_equal_evaluated_derived_rows(
    protocol_chain: dict[str, object],
    mutation: str,
) -> None:
    receipt = deepcopy(protocol_chain["gate_receipt"])
    rows = receipt["field_difference_ledger"]
    if mutation == "fake_sha":
        rows[0]["forced_G1_output_fingerprint"] = _fp(
            "syntactically-valid-but-fake-field"
        )
    elif mutation == "missing":
        rows.pop()
    else:
        extra = deepcopy(
            protocol_chain["pooled"].payload[
                "verified_field_difference_ledger"
            ][0]
        )
        extra["sample_id"] = protocol_chain["pooled"].payload[
            "held_out_sample_ids_by_fold"
        ]["0"][1]
        extra["natural_output_fingerprint"] = _fp("fake-extra-natural")
        extra["forced_G1_output_fingerprint"] = _fp("fake-extra-forced")
        rows.append(extra)
    receipt["field_difference_count"] = len(rows)
    receipt["field_difference_ledger_fingerprint"] = stable_fingerprint(
        rows
    )
    receipt = _reseal(receipt)
    with pytest.raises(PermissionError, match="gate-path"):
        verify_gate_path_receipt(receipt, protocol_chain["pooled"])


def test_oof_gate_is_relative_improvement_without_fixed_uplift_margin(
    protocol_chain: dict[str, object],
) -> None:
    decision = protocol_chain["oof_decision"].payload
    assert decision["gate_passed"] is True
    assert decision["comparison"] == (
        "pooled_factual_only_sufficient_statistics"
    )
    assert decision["checks"][
        "strict_true_targets_above_Base_envelope"
    ] is True
    assert decision["checks"][
        "strict_recovery_above_PACRE_VC_v23_control"
    ] is True
    assert "minimum_uplift" not in decision
    assert "fixed_uplift_threshold" not in decision


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("schema", "identity"),
        ("seed", "execution identity"),
        ("finite", "finite"),
        ("source", "source bytes"),
        ("artifact", "bytes changed"),
        ("prerequisite", "prerequisites"),
    ],
)
def test_bounded_exact_schema_seed_finite_source_artifact_prereq_tamper(
    protocol_chain: dict[str, object],
    mutation: str,
    match: str,
) -> None:
    receipt = deepcopy(protocol_chain["bounded_receipt"])
    arm = receipt["arms"]["GCR_PACRE_v24"]
    if mutation == "schema":
        receipt["schema_version"] = "wrong"
    elif mutation == "seed":
        arm["seed"] = 43
    elif mutation == "finite":
        arm["finite_audit"]["nonfinite_values"] = 1
    elif mutation == "source":
        arm["source_hashes"]["tools/gcr_pacre_v24_protocol.py"] = SHA_A
    elif mutation == "artifact":
        arm["terminal_artifact"]["file_sha256"] = SHA_A
    else:
        receipt["prerequisites"]["OOF4_decision_fingerprint"] = SHA_A
    receipt = _reseal(receipt)
    with pytest.raises((ValueError, RuntimeError, PermissionError), match=match):
        validate_paired_bounded_receipt(
            receipt,
            oof_decision=protocol_chain["oof_decision"],
            access_audit=protocol_chain["bounded_access"],
            full_d_r_cache_artifact=protocol_chain[
                "bounded_full_cache_token"
            ],
            dataset_free_receipt_fingerprint=protocol_chain[
                "dataset_free_fp"
            ],
            d_r_structural_receipt_fingerprint=protocol_chain[
                "structural_fp"
            ],
            repository_root=REPO_ROOT,
        )


def test_bounded_full_reseal_rejects_non_safetensors_and_metric_spoof(
    protocol_chain: dict[str, object],
    tmp_path: Path,
) -> None:
    fake_path = (tmp_path / "forged-terminal.safetensors").resolve()
    fake_path.write_bytes(b"caller-authored-terminal-bytes")
    forged = deepcopy(protocol_chain["bounded_receipt"])
    arm = forged["arms"][GCR_PACRE_CANDIDATE_ARM]
    arm["terminal_artifact"] = _file_meta(
        fake_path,
        arm["final_model_fingerprint"],
    )
    forged = _reseal(forged)
    with pytest.raises(
        (ValueError, RuntimeError, PermissionError),
        match="safetensors|header|deserialize|terminal",
    ):
        validate_paired_bounded_receipt(
            forged,
            oof_decision=protocol_chain["oof_decision"],
            access_audit=protocol_chain["bounded_access"],
            full_d_r_cache_artifact=protocol_chain[
                "bounded_full_cache_token"
            ],
            dataset_free_receipt_fingerprint=protocol_chain[
                "dataset_free_fp"
            ],
            d_r_structural_receipt_fingerprint=protocol_chain[
                "structural_fp"
            ],
            repository_root=REPO_ROOT,
        )

    forged = deepcopy(protocol_chain["bounded_receipt"])
    forged["arms"][GCR_PACRE_CANDIDATE_ARM]["metrics"][
        "terminal_PMOPE"
    ] += 1.0
    forged = _reseal(forged)
    with pytest.raises(PermissionError, match="receipt metrics differ"):
        validate_paired_bounded_receipt(
            forged,
            oof_decision=protocol_chain["oof_decision"],
            access_audit=protocol_chain["bounded_access"],
            full_d_r_cache_artifact=protocol_chain[
                "bounded_full_cache_token"
            ],
            dataset_free_receipt_fingerprint=protocol_chain[
                "dataset_free_fp"
            ],
            d_r_structural_receipt_fingerprint=protocol_chain[
                "structural_fp"
            ],
            repository_root=REPO_ROOT,
        )


@pytest.mark.parametrize(
    "mutation, match",
    [
        (
            lambda value: value["paired_diagnostics"].update(
                {"interpretation": "fixed_uplift_threshold"}
            ),
            "fixed gate interpretation",
        ),
        (
            lambda value: value["paired_diagnostics"][
                "candidate_minus_control"
            ].update({"PMOPE": 123.0}),
            "exact terminal metric difference",
        ),
        (
            lambda value: value["paired_diagnostics"][
                "candidate_minus_same_weight_G1"
            ].update({"field_nonidentity_witness": False}),
            "nonidentity witnesses",
        ),
        (
            lambda value: value["paired_diagnostics"]["per_update"].pop(),
            "exactly 400 rows",
        ),
        (
            lambda value: value["paired_diagnostics"]["per_update"][1].update(
                {"update": 0}
            ),
            "ordering or exact deltas",
        ),
        (
            lambda value: value["paired_diagnostics"]["per_update"][0].update(
                {"candidate_minus_control_loss": 5.0}
            ),
            "ordering or exact deltas",
        ),
        (
            _reseal_bounded_diagnostic_row_against_itself,
            "persisted training trace",
        ),
    ],
)
def test_bounded_paired_diagnostics_are_complete_and_mechanically_bound(
    protocol_chain: dict[str, object],
    mutation: Callable[[dict[str, object]], object],
    match: str,
) -> None:
    receipt = deepcopy(protocol_chain["bounded_receipt"])
    mutation(receipt)
    receipt = _reseal(receipt)
    with pytest.raises((ValueError, PermissionError), match=match):
        validate_paired_bounded_receipt(
            receipt,
            oof_decision=protocol_chain["oof_decision"],
            access_audit=protocol_chain["bounded_access"],
            full_d_r_cache_artifact=protocol_chain[
                "bounded_full_cache_token"
            ],
            dataset_free_receipt_fingerprint=protocol_chain[
                "dataset_free_fp"
            ],
            d_r_structural_receipt_fingerprint=protocol_chain[
                "structural_fp"
            ],
            repository_root=REPO_ROOT,
        )


@pytest.mark.parametrize(
    "mutation, match",
    [
        (
            lambda metrics: metrics.update(
                {"terminal_gate_distribution": None}
            ),
            "must be a mapping",
        ),
        (
            lambda metrics: metrics["terminal_gate_distribution"][
                "endpoint_counts"
            ].update({"G_strict_interior": 13}),
            "counts are inconsistent",
        ),
        (
            lambda metrics: metrics["terminal_gate_distribution"][
                "target_G"
            ].update({"maximum": 2.1}),
            "G in",
        ),
        (
            lambda metrics: metrics["terminal_gate_distribution"][
                "target_E"
            ].update({"count": 3}),
            "counts are inconsistent",
        ),
        (
            lambda metrics: metrics["terminal_gate_distribution"][
                "background_E"
            ].update({"mean": "nan"}),
            "must be real",
        ),
        (
            _force_full_endpoint_gate,
            "strict-interior gate requirement",
        ),
    ],
)
def test_bounded_gate_role_distributions_fail_closed_on_tamper(
    protocol_chain: dict[str, object],
    mutation: Callable[[dict[str, object]], object],
    match: str,
) -> None:
    receipt = deepcopy(protocol_chain["bounded_receipt"])
    metrics = receipt["arms"]["GCR_PACRE_v24"]["metrics"]
    mutation(metrics)
    receipt = _reseal(receipt)
    with pytest.raises((TypeError, ValueError, PermissionError), match=match):
        validate_paired_bounded_receipt(
            receipt,
            oof_decision=protocol_chain["oof_decision"],
            access_audit=protocol_chain["bounded_access"],
            full_d_r_cache_artifact=protocol_chain[
                "bounded_full_cache_token"
            ],
            dataset_free_receipt_fingerprint=protocol_chain[
                "dataset_free_fp"
            ],
            d_r_structural_receipt_fingerprint=protocol_chain[
                "structural_fp"
            ],
            repository_root=REPO_ROOT,
        )


def test_training_trace_hash_chain_detects_accidental_row_tamper(
    protocol_chain: dict[str, object],
    tmp_path: Path,
) -> None:
    payload = deepcopy(protocol_chain["bounded_trace_payload"])
    assert payload["evidence_limitation"] == (
        "accidental_tamper_and_internal_consistency_evidence_not_"
        "cryptographic_proof_against_same_user_malicious_fabrication_"
        "without_external_signature_or_trusted_hardware"
    )
    payload["rows"][7]["arms"][GCR_PACRE_CANDIDATE_ARM]["loss"] += 0.5
    path = (tmp_path / "training_trace.json").resolve()
    artifact = save_training_trace_new(path, payload)
    with pytest.raises(PermissionError, match="hash chain broke"):
        verify_training_trace_artifact(
            artifact,
            expected_path=path,
            stage_id="paired_bounded400",
            authorization_fingerprint=protocol_chain[
                "bounded_authorization_fp"
            ],
            schedule=protocol_chain["bounded_schedule"],
            arm_names=(
                GCR_PACRE_CONTROL_ARM,
                GCR_PACRE_CANDIDATE_ARM,
            ),
            terminal_model_fingerprints={
                arm: value["final_model_fingerprint"]
                for arm, value in protocol_chain[
                    "bounded_arms"
                ].items()
            },
        )


def test_run_start_fingerprint_binds_current_authorization_bytes(
    protocol_chain: dict[str, object],
    tmp_path: Path,
) -> None:
    altered = deepcopy(protocol_chain["bounded_authorization_payload"])
    altered["seed"] = 43
    path = (tmp_path / "authorization.json").resolve()
    _write_immutable_json(path, altered)
    with pytest.raises(
        PermissionError,
        match="differs from the current authorization artifact",
    ):
        protocol_module._validate_current_authorization_artifact(
            path,
            expected_authorization_fingerprint=protocol_chain[
                "bounded_authorization_fp"
            ],
            name="bounded test",
        )


def test_bounded_decision_is_absolute_smoke_not_paired_delta_gate(
    protocol_chain: dict[str, object],
) -> None:
    assert protocol_chain["bounded_decision"]["gate_passed"] is True
    with pytest.raises(TypeError, match="bounded_evidence"):
        decide_paired_bounded400(
            {"candidate": {"terminal_PMOPE": 0.0}}
        )


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("extra_schema", "fields changed"),
        ("seed", "role/evaluation"),
        ("finite", "finite"),
        ("source", "source bytes"),
        ("schedule", "bytes changed"),
        ("artifact", "bytes changed"),
        ("prerequisite", "prerequisite"),
    ],
)
def test_formal_exact_800x40_schema_source_schedule_artifact_prereq_tamper(
    protocol_chain: dict[str, object],
    mutation: str,
    match: str,
) -> None:
    receipt = deepcopy(protocol_chain["formal_receipt"])
    if mutation == "extra_schema":
        receipt["training_receipt"]["unexpected"] = True
    elif mutation == "seed":
        receipt["training_receipt"]["seed"] = 43
    elif mutation == "finite":
        receipt["finite_audit"]["loss_values_checked"] -= 1
    elif mutation == "source":
        receipt["training_receipt"]["source_hashes"][
            "cure_lite_v24/training.py"
        ] = SHA_A
    elif mutation == "schedule":
        receipt["schedule_artifact"]["file_sha256"] = SHA_A
    elif mutation == "artifact":
        receipt["terminal_artifact"]["file_sha256"] = SHA_A
    else:
        receipt["prerequisites"][
            "paired_bounded400_decision_fingerprint"
        ] = SHA_A
    receipt["training_receipt_fingerprint"] = stable_fingerprint(
        receipt["training_receipt"]
    )
    receipt = _reseal(receipt)
    with pytest.raises((ValueError, RuntimeError, PermissionError), match=match):
        validate_formal_training_receipt(
            receipt,
            expected_seed=42,
            expected_role="primary",
            oof_decision=protocol_chain["oof_decision"],
            bounded_decision=protocol_chain["bounded_decision"],
            access_audit=protocol_chain["formal_access"],
            cache_artifact=protocol_chain["formal_cache_token"],
            dataset_free_receipt_fingerprint=protocol_chain[
                "dataset_free_fp"
            ],
            d_r_structural_receipt_fingerprint=protocol_chain[
                "structural_fp"
            ],
            repository_root=REPO_ROOT,
        )


def test_formal_full_reseal_rejects_non_safetensors_and_metric_spoof(
    protocol_chain: dict[str, object],
    tmp_path: Path,
) -> None:
    fake_path = (tmp_path / "forged-formal.safetensors").resolve()
    fake_path.write_bytes(b"caller-authored-formal-terminal")
    forged = deepcopy(protocol_chain["formal_receipt"])
    forged["terminal_artifact"] = _file_meta(
        fake_path,
        forged["training_receipt"]["model"]["final_fingerprint"],
    )
    forged = _reseal(forged)
    with pytest.raises(
        (ValueError, RuntimeError, PermissionError),
        match="safetensors|header|deserialize|terminal",
    ):
        validate_formal_training_receipt(
            forged,
            expected_seed=42,
            expected_role="primary",
            oof_decision=protocol_chain["oof_decision"],
            bounded_decision=protocol_chain["bounded_decision"],
            access_audit=protocol_chain["formal_access"],
            cache_artifact=protocol_chain["formal_cache_token"],
            dataset_free_receipt_fingerprint=protocol_chain[
                "dataset_free_fp"
            ],
            d_r_structural_receipt_fingerprint=protocol_chain[
                "structural_fp"
            ],
            repository_root=REPO_ROOT,
        )

    forged = deepcopy(protocol_chain["formal_receipt"])
    forged["terminal_D_R_evaluation"]["metrics"]["metrics"][
        "PMOPE"
    ] += 1.0
    forged["terminal_D_R_evaluation_fingerprint"] = stable_fingerprint(
        forged["terminal_D_R_evaluation"]
    )
    forged = _reseal(forged)
    with pytest.raises(
        PermissionError,
        match="terminal D_R evidence differs",
    ):
        validate_formal_training_receipt(
            forged,
            expected_seed=42,
            expected_role="primary",
            oof_decision=protocol_chain["oof_decision"],
            bounded_decision=protocol_chain["bounded_decision"],
            access_audit=protocol_chain["formal_access"],
            cache_artifact=protocol_chain["formal_cache_token"],
            dataset_free_receipt_fingerprint=protocol_chain[
                "dataset_free_fp"
            ],
            d_r_structural_receipt_fingerprint=protocol_chain[
                "structural_fp"
            ],
            repository_root=REPO_ROOT,
        )


def test_formal_receipt_matches_training_core_eligibility_semantics(
    protocol_chain: dict[str, object],
) -> None:
    formal = protocol_chain["formal"]
    assert formal.seed == 42
    assert formal.role == "primary"
    assert formal.payload["schema_version"].endswith("evidence-v6")
    inner = formal.payload["training_receipt"]
    assert inner["budget"] == {
        "epochs": 800,
        "steps_per_epoch": 40,
        "updates": 32_000,
        "training_invocations": 1,
    }
    assert inner[
        "eligible_for_future_D_V_authorization_after_all_external_prerequisites"
    ] is True
    assert inner["D_V_execution_authorized"] is False
    assert inner["D_V_payload_accessed"] is False
    pair = protocol_chain["formal_pair"]
    assert pair.payload["schema_version"].endswith("training-pair-v2")
    assert pair["checks"]["different_interpreter_process_instances"] is True
    assert (
        protocol_chain["formal"].process_instance_fingerprint
        != protocol_chain["formal43"].process_instance_fingerprint
    )
    assert pair["seed43_selection_effect"] == "none"
    assert pair["D_V_payload_accessed_by_seed43"] is False
    with pytest.raises((TypeError, PermissionError), match="seed43_integrity|seed/role"):
        verify_formal800_training_independence(
            protocol_chain["formal"],
            protocol_chain["formal"],
        )


def test_formal_cache_requires_private_issuer_and_exact_outer_token(
    protocol_chain: dict[str, object],
) -> None:
    token = protocol_chain["formal_cache_token"]
    assert isinstance(token, VerifiedFormalCacheArtifact)
    fake = replace(token, _issuer=object())
    with pytest.raises(TypeError, match="fixed Formal cache verifier"):
        require_verified_formal_cache_artifact(fake)
    bounded_origin = protocol_chain["bounded_full_cache_token"]
    assert (
        require_verified_formal_cache_origin_artifact(bounded_origin)
        is bounded_origin
    )

    receipt = deepcopy(protocol_chain["formal_receipt"])
    receipt["cache_artifact"][
        "shared_tensor_storage_with_other_formal_cache"
    ] = False
    receipt = _reseal(receipt)
    with pytest.raises(PermissionError, match="mechanically verified"):
        validate_formal_training_receipt(
            receipt,
            expected_seed=42,
            expected_role="primary",
            oof_decision=protocol_chain["oof_decision"],
            bounded_decision=protocol_chain["bounded_decision"],
            access_audit=protocol_chain["formal_access"],
            cache_artifact=token,
            dataset_free_receipt_fingerprint=protocol_chain[
                "dataset_free_fp"
            ],
            d_r_structural_receipt_fingerprint=protocol_chain[
                "structural_fp"
            ],
            repository_root=REPO_ROOT,
        )


def test_bounded_receipt_accepts_cross_process_cache_reverification(
    protocol_chain: dict[str, object],
) -> None:
    origin = protocol_chain["bounded_full_cache_token"]
    public_only = verify_formal_cache_artifact(
        origin.path,
        cache_id=origin.cache_id,
        expected_semantic_cache_fingerprint=(
            origin.semantic_cache_fingerprint
        ),
    )
    assert public_only.payload == origin.payload
    reverified = validate_paired_bounded_receipt(
        protocol_chain["bounded_receipt"],
        oof_decision=protocol_chain["oof_decision"],
        access_audit=protocol_chain["bounded_access"],
        full_d_r_cache_artifact=public_only,
        dataset_free_receipt_fingerprint=protocol_chain[
            "dataset_free_fp"
        ],
        d_r_structural_receipt_fingerprint=protocol_chain[
            "structural_fp"
        ],
        repository_root=REPO_ROOT,
    )
    assert (
        reverified.receipt_fingerprint
        == protocol_chain["bounded_evidence"].receipt_fingerprint
    )


def test_public_protocol_token_authenticity_helpers_reject_replaced_issuers(
    protocol_chain: dict[str, object],
) -> None:
    cases = (
        (
            protocol_chain["formal_access"],
            require_verified_access_audit,
        ),
        (
            protocol_chain["oof_decision"],
            require_verified_oof_decision,
        ),
        (
            protocol_chain["bounded_decision"],
            require_verified_bounded_decision,
        ),
    )
    for token, verifier in cases:
        assert verifier(token) is token
        with pytest.raises(TypeError, match="issued by"):
            verifier(replace(token, _issuer=object()))


def test_every_protocol_capability_rejects_retain_issuer_replace(
    protocol_chain: dict[str, object],
) -> None:
    tokens = [
        protocol_chain["formal_access"],
        protocol_chain["split"],
        protocol_chain["fold_tokens"][0],
        protocol_chain["fold_pools"][0],
        protocol_chain["pooled"],
        protocol_chain["gate"],
        protocol_chain["oof_decision"],
        protocol_chain["bounded_evidence"],
        protocol_chain["bounded_decision"],
        protocol_chain["formal"],
        protocol_chain["formal43"],
        protocol_chain["formal_pair"],
        protocol_chain["d_t_prereg"],
        protocol_chain["binding"],
        protocol_chain["baseline"],
        protocol_chain["d_v_evidence"],
    ]
    for token in tokens:
        forged = replace(token)
        assert forged._issuer is token._issuer
        with pytest.raises(TypeError, match="protocol verifier"):
            protocol_module._token(
                forged,
                type(token),
                name=type(token).__name__,
            )


def test_formal_cache_capability_rejects_retain_issuer_replace(
    protocol_chain: dict[str, object],
) -> None:
    for token, verifier in (
        (
            protocol_chain["formal_cache_token"],
            require_verified_formal_cache_artifact,
        ),
        (
            protocol_chain["bounded_full_cache_token"],
            require_verified_formal_cache_origin_artifact,
        ),
    ):
        forged = replace(token)
        assert forged._issuer is token._issuer
        with pytest.raises(TypeError, match="fixed Formal cache verifier"):
            verifier(forged)


def test_formal_cache_pair_detects_actual_process_storage_reuse(
    protocol_chain: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = protocol_chain["formal_cache_token"]
    integrity = protocol_chain["formal43_cache_token"]
    shared_loaded = torch.load(
        primary.path,
        map_location="cpu",
        weights_only=True,
        mmap=False,
    )
    monkeypatch.setattr(
        formal_cache_artifacts.torch,
        "load",
        lambda *args, **kwargs: shared_loaded,
    )
    with pytest.raises(
        PermissionError,
        match="actual_loaded_tensor_storages_disjoint",
    ):
        verify_formal_cache_pair_independence(primary, integrity)


def test_formal_cache_rejects_arbitrary_payload_with_correct_semantic_string(
    protocol_chain: dict[str, object],
    tmp_path: Path,
) -> None:
    semantic = protocol_chain["semantic_formal_cache_fp"]
    wrong_path = tmp_path / "wrong-but-claims-right-semantic.pt"
    wrong_envelope = build_formal_cache_neutral_envelope(
        protocol_chain["formal_scalar_cache"]
    )
    tensor_map = wrong_envelope["payload"]["tensors"]
    first_name = next(iter(tensor_map))
    tensor_map[first_name] = tensor_map[first_name].clone()
    tensor_map[first_name].view(-1)[0] += 999
    for row in wrong_envelope["payload"]["tensor_ledger"]:
        if row["logical_path"] == first_name:
            row["content_fingerprint"] = tensor_content_fingerprint(
                tensor_map[first_name]
            )
    torch.save(wrong_envelope, wrong_path)
    wrong_token = verify_formal_cache_artifact(
        wrong_path.resolve(),
        cache_id="formal800-seed42-primary-full-D_R-cache",
        expected_semantic_cache_fingerprint=semantic,
    )
    with pytest.raises(TypeError, match="exact CoverageStateScalarCache"):
        require_verified_formal_cache_origin_artifact(wrong_token)
    wrong_access_receipt = _access_receipt(
        "formal800_seed42_primary",
        ["D_R"],
        [
            {
                "split": "D_R",
                "logical_id": wrong_token.cache_id,
                "purpose": "Formal800_seed42_primary_training_cache",
                "source_fingerprint": wrong_token.file_sha256,
            }
        ],
    )
    wrong_access = verify_access_audit_receipt(
        wrong_access_receipt,
        expected_stage_id="formal800_seed42_primary",
        allowed_splits=["D_R"],
    )
    receipt = deepcopy(protocol_chain["formal_receipt"])
    receipt["access_audit_receipt_fingerprint"] = (
        wrong_access.receipt_fingerprint
    )
    receipt["cache_artifact"] = wrong_token.payload
    receipt = _reseal(receipt)
    with pytest.raises(
        PermissionError,
        match="verified bounded predecessor",
    ):
        validate_formal_training_receipt(
            receipt,
            expected_seed=42,
            expected_role="primary",
            oof_decision=protocol_chain["oof_decision"],
            bounded_decision=protocol_chain["bounded_decision"],
            access_audit=wrong_access,
            cache_artifact=wrong_token,
            dataset_free_receipt_fingerprint=protocol_chain[
                "dataset_free_fp"
            ],
            d_r_structural_receipt_fingerprint=protocol_chain[
                "structural_fp"
            ],
            repository_root=REPO_ROOT,
        )


@pytest.mark.parametrize(
    "mutation, match",
    [
        (
            lambda envelope: envelope["payload"].update(
                {
                    "canonical_cache_payload_json": (
                        envelope["payload"][
                            "canonical_cache_payload_json"
                        ]
                        + " "
                    )
                }
            ),
            "canonical self-fingerprint",
        ),
        (
            lambda envelope: envelope["payload"]["tensor_ledger"].pop(),
            "tensor_ledger paths/content",
        ),
        (
            lambda envelope: envelope["payload"]["tensor_ledger"].append(
                deepcopy(envelope["payload"]["tensor_ledger"][0])
            ),
            "tensor_ledger paths/content",
        ),
    ],
)
def test_formal_cache_exact_payload_and_tensor_ledger_fail_closed(
    protocol_chain: dict[str, object],
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], object],
    match: str,
) -> None:
    cache = protocol_chain["formal_scalar_cache"]
    envelope = build_formal_cache_neutral_envelope(cache)
    mutation(envelope)
    path = tmp_path / f"invalid-envelope-{stable_fingerprint(match)}.pt"
    torch.save(envelope, path)
    with pytest.raises(ValueError, match=match):
        verify_formal_cache_artifact(
            path.resolve(),
            cache_id="generated-invalid-envelope",
            expected_semantic_cache_fingerprint=cache.cache_fingerprint,
        )


def test_formal_cache_fiemap_rejects_actual_reflink(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    clone = tmp_path / "clone.pt"
    semantic = _write_formal_cache(source)
    source_fd = os.open(source, os.O_RDONLY)
    clone_fd = os.open(clone, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        # Linux FICLONE = _IOW(0x94, 9, int).
        try:
            fcntl.ioctl(clone_fd, 0x40049409, source_fd)
        except OSError as error:
            pytest.skip(f"filesystem does not support reflink: {error}")
    finally:
        os.close(clone_fd)
        os.close(source_fd)
    with pytest.raises(PermissionError, match=r"\{0,LAST\} whitelist"):
        verify_formal_cache_artifact(
            clone.resolve(),
            cache_id="generated-reflink-negative",
            expected_semantic_cache_fingerprint=semantic,
        )


@pytest.mark.parametrize(
    "forbidden_flag",
    [0x2, 0x4, 0x8, 0x10, 0x1000, 0x2000],
)
def test_formal_cache_fiemap_forbidden_flags_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forbidden_flag: int,
) -> None:
    path = tmp_path / f"fiemap-{forbidden_flag}.pt"
    semantic = _write_formal_cache(path)

    def forged_fiemap(
        descriptor: int,
        request: int,
        buffer: bytearray,
        mutate: bool,
    ) -> int:
        del descriptor, request, mutate
        struct.pack_into("=I", buffer, 20, 1)
        struct.pack_into(
            "=QQQQQIIII",
            buffer,
            32,
            0,
            4096,
            path.stat().st_size,
            0,
            0,
            0x1 | forbidden_flag,
            0,
            0,
            0,
        )
        return 0

    monkeypatch.setattr(
        formal_cache_artifacts.fcntl,
        "ioctl",
        forged_fiemap,
    )
    with pytest.raises(PermissionError, match=r"\{0,LAST\} whitelist"):
        verify_formal_cache_artifact(
            path.resolve(),
            cache_id=f"generated-fiemap-negative-{forbidden_flag}",
            expected_semantic_cache_fingerprint=semantic,
        )


def test_d_t_preregistration_every_field_and_anchor_are_fail_closed(
    protocol_chain: dict[str, object],
) -> None:
    assert protocol_chain["d_t_prereg"].payload[
        "threshold_search_allowed"
    ] is False
    payload = deepcopy(protocol_chain["d_t_payload"])
    payload["Base_validity_and_envelope"][
        "D_T_evaluated_Base_operating_point_count"
    ] = 51
    payload = _reseal(payload, "preregistration_fingerprint")
    with pytest.raises((ValueError, PermissionError), match="Base envelope"):
        validate_d_t_preregistration(payload, repository_root=REPO_ROOT)


@pytest.mark.parametrize(
    "mutation, match",
    [
        ("seed", "identity"),
        ("artifact", "bytes changed"),
        ("event", "predate"),
        ("formal_fp", "identity"),
    ],
)
def test_d_t_seed42_model_binding_tamper_fails(
    protocol_chain: dict[str, object],
    mutation: str,
    match: str,
) -> None:
    receipt = deepcopy(protocol_chain["binding_receipt"])
    if mutation == "seed":
        receipt["seed"] = 43
    elif mutation == "artifact":
        receipt["terminal_artifact"]["file_sha256"] = SHA_A
    elif mutation == "event":
        receipt["events"]["D_V_authorization_created"] = 100
    else:
        receipt["Formal800_evidence_receipt_fingerprint"] = SHA_A
    receipt = _reseal(receipt)
    with pytest.raises((ValueError, RuntimeError, PermissionError), match=match):
        validate_d_t_seed42_model_binding(
            receipt,
            d_t_preregistration=protocol_chain["d_t_prereg"],
            formal_seed42=protocol_chain["formal"],
            formal_pair=protocol_chain["formal_pair"],
            access_audit=protocol_chain["binding_access"],
        )


def test_relative_dv_decision_has_no_naked_unread_boolean(
    protocol_chain: dict[str, object],
) -> None:
    decision = decide_relative_dv_gate(
        protocol_chain["d_v_evidence"],
        protocol_chain["baseline"],
    )
    assert decision["gate_passed"] is True
    assert decision["minimum_fixed_uplift_margin"] is None
    with pytest.raises(TypeError, match="candidate_evidence"):
        decide_relative_dv_gate(
            {"candidate_metrics": {}},
            protocol_chain["baseline"],
        )


def test_protocol_artifact_fingerprints_are_mutually_anchored(
    tmp_path: Path,
) -> None:
    main = json.loads((PROTOCOL_ROOT / "preregistration.json").read_text())
    main_fp = main.pop("preregistration_fingerprint")
    assert main_fp == stable_fingerprint(main)
    for filename, self_field in (
        ("D_R_OOF4_split_preregistration.json", "receipt_fingerprint"),
        ("D_T_preregistration.json", "preregistration_fingerprint"),
        ("exact_baseline_ledger_binding.json", "binding_fingerprint"),
    ):
        payload = json.loads((PROTOCOL_ROOT / filename).read_text())
        assert payload["protocol_preregistration_fingerprint"] == main_fp
        fingerprint = payload.pop(self_field)
        assert fingerprint == stable_fingerprint(payload)
    template = json.loads(
        (
            PROTOCOL_ROOT / "D_T_seed42_model_binding.template.json"
        ).read_text()
    )
    assert template["status"] == "TEMPLATE_NOT_A_VALID_BINDING_RECEIPT"
    assert template["final_model_fingerprint"] is None
    artifact_manifest = json.loads(
        (PROTOCOL_ROOT / "artifact_manifest.json").read_text()
    )
    assert validate_protocol_artifact_manifest(
        PROTOCOL_ROOT / "artifact_manifest.json",
        repository_root=REPO_ROOT,
    ) == artifact_manifest["manifest_fingerprint"]
    tampered = deepcopy(artifact_manifest)
    tampered["artifacts"]["protocol.schema.json"]["file_sha256"] = SHA_A
    tampered = _reseal(tampered, "manifest_fingerprint")
    tampered_path = tmp_path / "artifact_manifest.tampered.json"
    tampered_path.write_text(canonical_json(tampered))
    with pytest.raises(RuntimeError, match="artifact bytes"):
        validate_protocol_artifact_manifest(
            tampered_path,
            repository_root=REPO_ROOT,
        )
