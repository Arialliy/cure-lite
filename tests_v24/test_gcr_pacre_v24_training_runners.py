from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import cure_lite_v24.formal_training as formal_training
import cure_lite_v24.formal_run_start as formal_run_start
import cure_lite_v24.oof_runner as oof_runner
import cure_lite_v24.real_input_factory as real_input_factory
import cure_lite_v24.source_closure as source_closure
import cure_lite_v24.training as training
import tools.run_cure_lite_v24_gcr_pacre_bounded_400 as bounded_cli
import tools.run_cure_lite_v24_gcr_pacre_formal_800 as formal_cli
import tools.run_cure_lite_v24_gcr_pacre_oof4 as oof_cli
import tools.prepare_cure_lite_v24_gcr_pacre_training_chain as chain_setup_cli
from cure_lite.cache.schema import canonical_json, stable_fingerprint
from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
)
from cure_lite.experiment.coverage_state_training import (
    COVERAGE_STATE_FORMAL_SCOPE,
    CoverageStateTrainingResult,
    coverage_state_model_fingerprint,
    coverage_state_optimizer_config_fingerprint,
)
from cure_lite.paired_types import tensor_content_fingerprint
from cure_lite.train.coverage_state_fused_step import (
    CoverageStatePairObjective,
    coverage_state_pair_objective_policy,
)
from cure_lite_v24.bounded_runner import (
    GCR_PACRE_BOUNDED_UPDATES,
    GCR_PACRE_CANDIDATE_ARM,
    GCR_PACRE_CONTROL_ARM,
    GCR_PACRE_FORCED_G1_MODE,
    GCRPACREBoundedEvaluation,
    GCRPACREBoundedEvaluator,
    build_paired_bounded_receipt,
    prepare_gcr_pacre_paired_bounded_authorization,
    run_gcr_pacre_paired_bounded_400,
)
from cure_lite_v24.bounded_run_start import (
    VerifiedGCRPACREBoundedChainConfig,
    create_gcr_pacre_bounded_run_start_marker,
    seal_gcr_pacre_bounded_chain_config_new,
)
from cure_lite_v24.factory import build_gcr_pacre_training_model
from cure_lite_v24.formal_artifacts import (
    build_formal_evidence_receipt,
    load_and_verify_gcr_pacre_formal_terminal,
    save_gcr_pacre_formal_schedule_atomic,
    save_gcr_pacre_formal_terminal_atomic,
)
from cure_lite_v24.formal_cache_artifacts import (
    build_formal_cache_neutral_envelope,
    save_formal_cache_neutral_artifact_new,
    verify_formal_cache_artifact,
    verify_formal_cache_pair_independence,
)
from cure_lite_v24.formal_run_start import (
    create_gcr_pacre_formal_run_start_marker,
    seal_gcr_pacre_formal_chain_config_new,
)
from cure_lite_v24.fixed_dr_evaluator import FrozenGCRPACREDREvaluator
from cure_lite_v24.gcr_pacre import CoverageStateGCRPACREConfig
from tools.gcr_pacre_v24_protocol import (
    VerifiedBoundedDecision,
    VerifiedOOFDecision,
    verify_access_audit_receipt,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_training_scalar_cache,
)


def _fp(label: str) -> str:
    return stable_fingerprint({"generated": label})


def _seal(body: dict[str, object]) -> dict[str, object]:
    return {**body, "receipt_fingerprint": stable_fingerprint(body)}


def _assert_cross_process_marker_replay_rejected(path: str) -> None:
    replay = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os,sys;"
                "fd=os.open(sys.argv[1],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444);"
                "os.close(fd)"
            ),
            path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert replay.returncode != 0


def _access(stage: str, observed: list[dict[str, object]]):
    body = {
        "schema_version": "cure-lite-v24-split-access-audit-v1",
        "stage_id": stage,
        "allowed_splits": ["D_R"],
        "observed_payloads": observed,
        "source_manifest_fingerprint": _fp(f"{stage}-manifest"),
        "event_log_fingerprint": stable_fingerprint(observed),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return verify_access_audit_receipt(
        _seal(body),
        expected_stage_id=stage,
        allowed_splits=["D_R"],
    )


def _forged_oof() -> VerifiedOOFDecision:
    return VerifiedOOFDecision(
        payload_json=canonical_json({"gate_passed": True}),
        decision_fingerprint=_fp("oof-decision"),
        pooled_evidence_fingerprint=_fp("pooled"),
        gate_path_receipt_fingerprint=_fp("g1-path"),
        _issuer=object(),
    )


def _forged_bounded(
    *,
    semantic_cache_fingerprint: str,
    neutral_payload_fingerprint: str,
    materialization_receipt_fingerprint: str,
) -> VerifiedBoundedDecision:
    binding = {
        "semantic_cache_fingerprint": semantic_cache_fingerprint,
        "neutral_payload_fingerprint": neutral_payload_fingerprint,
        "materialization_receipt_fingerprint": (
            materialization_receipt_fingerprint
        ),
    }
    return VerifiedBoundedDecision(
        payload_json=canonical_json(
            {
                "gate_passed": True,
                "full_D_R_cache_binding": binding,
            }
        ),
        decision_fingerprint=_fp("bounded-decision"),
        bounded_receipt_fingerprint=_fp("bounded-receipt"),
        oof_decision_fingerprint=_fp("oof-decision"),
        full_d_r_semantic_cache_fingerprint=(
            semantic_cache_fingerprint
        ),
        full_d_r_neutral_payload_fingerprint=(
            neutral_payload_fingerprint
        ),
        full_d_r_materialization_receipt_fingerprint=(
            materialization_receipt_fingerprint
        ),
        _issuer=object(),
    )


def _generated_gate_distribution() -> dict[str, object]:
    return {
        "schema_version": "cure-lite-v24-gcr-pacre-gate-role-summary-v1",
        "endpoint_counts": {
            "G_equal_0": 0,
            "G_equal_2": 0,
            "G_strict_interior": 8,
        },
        "target_G": {
            "count": 4,
            "minimum": 0.5,
            "maximum": 1.5,
            "mean": 1.0,
        },
        "background_G": {
            "count": 4,
            "minimum": 0.5,
            "maximum": 1.5,
            "mean": 1.0,
        },
        "target_E": {
            "count": 4,
            "minimum": -1.0,
            "maximum": 1.0,
            "mean": 0.0,
        },
        "background_E": {
            "count": 4,
            "minimum": -1.0,
            "maximum": 1.0,
            "mean": 0.0,
        },
    }


class _GeneratedBoundedEvaluator(GCRPACREBoundedEvaluator):
    @property
    def evaluator_fingerprint(self) -> str:
        return _fp("generated-bounded-evaluator")

    def evaluate(
        self,
        model,
        cache,
        *,
        arm: str,
        checkpoint: str,
        forward_mode: str,
    ) -> GCRPACREBoundedEvaluation:
        del model, cache
        initial = checkpoint == "initial"
        forced = forward_mode == GCR_PACRE_FORCED_G1_MODE
        gate_distribution = (
            _generated_gate_distribution()
            if arm == GCR_PACRE_CANDIDATE_ARM
            else None
        )
        return GCRPACREBoundedEvaluation(
            true_targets=4,
            recovered_anchor_misses=1,
            mIoU=0.5,
            nIoU=0.5,
            pd=0.5,
            retention=1.0,
            pixel_fa=0.0,
            raw_background_fa=0.0,
            fp_components_per_mp=0.0,
            budget_violation=False,
            PMOPE=2.0 if initial else (1.1 if forced else 1.0),
            target_role_violation=0.4 if initial else (0.3 if forced else 0.2),
            background_role_violation=(
                0.4 if initial else (0.3 if forced else 0.2)
            ),
            zero_crossed_target_states=0 if initial else (1 if forced else 2),
            false_completion_states=0,
            gate_role_distributions_present=gate_distribution is not None,
            gate_role_distribution_json=(
                None
                if gate_distribution is None
                else canonical_json(gate_distribution)
            ),
            field_fingerprint=_fp(
                f"{arm}-{checkpoint}-{forward_mode}-field"
            ),
            role_prediction_fingerprint=_fp(
                f"{arm}-{checkpoint}-{forward_mode}-roles"
            ),
        )


class _GeneratedFormalEvaluator(
    formal_training.GCRPACREFormalTerminalEvaluator
):
    @property
    def evaluator_fingerprint(self) -> str:
        return _fp("generated-formal-evaluator")

    def evaluate_terminal_d_r(
        self,
        model,
        cache,
        *,
        seed: int,
        role: str,
    ):
        del cache
        return {
            "generated_only": True,
            "seed": seed,
            "role": role,
            "model_fingerprint": coverage_state_model_fingerprint(model),
            "finite_metric": 1.0,
        }


def _generated_bounded_chain(
    *,
    oof,
    access,
    full_cache,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "cure_lite_v24.bounded_run_start._required_runtime_root",
        lambda: (tmp_path / "runtime").resolve(),
    )
    monkeypatch.setattr(
        "cure_lite_v24.bounded_run_start.require_verified_oof_decision",
        lambda value: value,
    )
    return seal_gcr_pacre_bounded_chain_config_new(
        oof_decision=oof,
        access_audit=access,
        full_d_r_cache_artifact=full_cache,
        dataset_free_receipt_fingerprint=_fp("dataset-free"),
        d_r_structural_receipt_fingerprint=_fp("structural"),
        device="cpu",
    )


def _generated_formal_chain(
    *,
    seed: int,
    cache,
    formal_cache,
    bounded,
    oof,
    access,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    other_seed = 43 if seed == 42 else 42
    other_role = (
        "training_integrity_only" if other_seed == 43 else "primary"
    )
    other_cache = save_formal_cache_neutral_artifact_new(
        deepcopy(cache),
        (tmp_path / f"formal-cache-{other_seed}.pt").resolve(),
        cache_id=f"formal800-seed{other_seed}-{other_role}-full-D_R-cache",
    )
    other_access = _access(
        f"formal800_seed{other_seed}_{other_role}",
        [
            {
                "split": "D_R",
                "logical_id": other_cache.cache_id,
                "purpose": (
                    "Formal800_seed42_primary_training_cache"
                    if other_seed == 42
                    else "Formal800_seed43_training_integrity_cache"
                ),
                "source_fingerprint": other_cache.file_sha256,
            }
        ],
    )
    monkeypatch.setattr(
        "cure_lite_v24.formal_run_start._required_runtime_root",
        lambda: (tmp_path / "runtime").resolve(),
    )
    monkeypatch.setattr(
        "cure_lite_v24.formal_run_start.require_verified_oof_decision",
        lambda value: value,
    )
    monkeypatch.setattr(
        "cure_lite_v24.formal_run_start.require_verified_bounded_decision",
        lambda value: value,
    )
    return seal_gcr_pacre_formal_chain_config_new(
        oof_decision=oof,
        bounded_decision=bounded,
        seed42_access_audit=(
            access if seed == 42 else other_access
        ),
        seed43_access_audit=(
            access if seed == 43 else other_access
        ),
        seed42_cache_artifact=(
            formal_cache if seed == 42 else other_cache
        ),
        seed43_cache_artifact=(
            formal_cache if seed == 43 else other_cache
        ),
        dataset_free_receipt_fingerprint=_fp("dataset-free"),
        d_r_structural_receipt_fingerprint=_fp("structural"),
        seed42_device="cpu",
        seed43_device="cpu",
    )


def test_actual_generated_paired_bounded_run_is_synchronized_and_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = make_training_scalar_cache()
    second_cache = deepcopy(cache)
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=10,
            steps_per_epoch=40,
        ),
    )
    full_cache = save_formal_cache_neutral_artifact_new(
        cache,
        (tmp_path / "bounded-full-D_R.pt").resolve(),
        cache_id="paired-bounded400-full-D_R-materialization",
    )
    access = _access(
        "paired_bounded400",
        [
            {
                "split": "D_R",
                "logical_id": full_cache.cache_id,
                "purpose": (
                    "paired_bounded400_full_D_R_materialization"
                ),
                "source_fingerprint": full_cache.file_sha256,
            }
        ],
    )
    oof = _forged_oof()
    monkeypatch.setattr(
        "cure_lite_v24.bounded_runner.require_verified_oof_decision",
        lambda value: value,
    )
    chain_config = _generated_bounded_chain(
        oof=oof,
        access=access,
        full_cache=full_cache,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    authorization = prepare_gcr_pacre_paired_bounded_authorization(
        oof_decision=oof,
        access_audit=access,
        full_d_r_cache_artifact=full_cache,
        chain_config=chain_config,
        dataset_free_receipt_fingerprint=_fp("dataset-free"),
        d_r_structural_receipt_fingerprint=_fp("structural"),
        control_cache=cache,
        candidate_cache=second_cache,
        schedule=schedule,
        candidate_config=CoverageStateGCRPACREConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        ),
        evaluator=_GeneratedBoundedEvaluator(),
    )
    run_start_token = create_gcr_pacre_bounded_run_start_marker(
        authorization
    )
    _assert_cross_process_marker_replay_rejected(
        run_start_token.marker_path
    )
    result = run_gcr_pacre_paired_bounded_400(
        authorization,
        run_start_token=run_start_token,
        output_directory=Path(authorization.output_directory),
        device="cpu",
    )
    receipt = build_paired_bounded_receipt(result)

    assert len(result.updates) == GCR_PACRE_BOUNDED_UPDATES
    assert all(len(row.selection_fingerprint) == 64 for row in result.updates)
    assert (
        result.control.training_result.initial_model_fingerprint
        == result.candidate.training_result.initial_model_fingerprint
    )
    assert (
        result.control.parameter_storage_ids
        != result.candidate.parameter_storage_ids
    )
    assert (
        result.diagnostic_payload["interpretation"]
        == "paired_deltas_are_diagnostic_only_without_a_fixed_threshold"
    )
    assert len(result.diagnostic_payload["per_update"]) == 400
    assert receipt["schema_version"].endswith("receipt-v6")
    assert receipt["training_trace_artifact"] == (
        result.training_trace_artifact
    )
    assert receipt["run_start_artifact"]["marker_fingerprint"] == (
        run_start_token.marker_fingerprint
    )
    assert receipt["paired_diagnostics"] == result.diagnostic_payload
    assert receipt["full_D_R_cache_materialization"] == full_cache.payload
    assert {
        receipt["arms"][arm]["neutral_payload_fingerprint"]
        for arm in (GCR_PACRE_CONTROL_ARM, GCR_PACRE_CANDIDATE_ARM)
    } == {full_cache.neutral_payload_fingerprint}
    with pytest.raises(PermissionError, match="no longer available"):
        run_gcr_pacre_paired_bounded_400(
            authorization,
            run_start_token=run_start_token,
            output_directory=Path(authorization.output_directory),
        )


def _fake_public_trainer():
    def run(
        model,
        optimizer,
        cache,
        schedule,
        *,
        objective,
        device,
        expected_initial_model_fingerprint,
        authorization,
        epoch_callback,
        update_callback,
    ):
        del epoch_callback
        assert objective is CoverageStatePairObjective.PMOPE_JOINT
        authorization.verify_for_run(
            cache=cache,
            schedule=schedule,
            scope=COVERAGE_STATE_FORMAL_SCOPE,
        )
        optimizer_fp = coverage_state_optimizer_config_fingerprint(
            model,
            optimizer,
        )
        with torch.no_grad():
            for index, parameter in enumerate(model.parameters(), start=1):
                parameter.add_(index * 1.0e-4)
        updates = schedule.config.updates
        for update, selection in enumerate(schedule.selections):
            update_callback(
                {
                    "update": update,
                    "epoch": selection.epoch,
                    "step": selection.step,
                    "selection_fingerprint": (
                        selection.selection_fingerprint
                    ),
                    "loss": 1.0 + update / 100_000.0,
                    "gradient_l2_norm": (
                        0.5 + update / 200_000.0
                    ),
                    "optimizer_step_counter": update + 1,
                    "parameter_state_digest": _fp(
                        f"fake-parameter-state-{update}"
                    ),
                    "optimizer_state_digest": _fp(
                        f"fake-optimizer-state-{update}"
                    ),
                    "loss_finite": True,
                    "gradients_finite": True,
                    "parameters_finite": True,
                    "optimizer_state_finite": True,
                }
            )
        return CoverageStateTrainingResult(
            objective=CoverageStatePairObjective.PMOPE_JOINT.value,
            objective_policy=coverage_state_pair_objective_policy(
                CoverageStatePairObjective.PMOPE_JOINT
            ),
            seed=schedule.config.seed,
            epochs=schedule.config.epochs,
            steps_per_epoch=schedule.config.steps_per_epoch,
            completed_updates=updates,
            schedule_fingerprint=schedule.schedule_fingerprint,
            cache_fingerprint=cache.cache_fingerprint,
            execution_device=str(torch.device(device)),
            device_cache_fingerprint=_fp(
                f"generated-device-cache-{schedule.config.seed}"
            ),
            device_cache_resident_bytes=1,
            optimizer_config_fingerprint=optimizer_fp,
            initial_model_fingerprint=expected_initial_model_fingerprint,
            final_model_fingerprint=coverage_state_model_fingerprint(model),
            epoch_logs=tuple(
                {"epoch": epoch} for epoch in range(schedule.config.epochs)
            ),
            first_nonzero_gradient_update=tuple(
                (name, 0) for name, _ in model.named_parameters()
            ),
            forward_calls=updates,
            backward_calls=updates,
            optimizer_steps=updates,
            logical_state_evaluations=updates * 12,
            finite_state_audits=updates + 1,
        )

    return run


def test_training_state_digest_detects_same_shape_element_permutation() -> None:
    model_a = build_gcr_pacre_training_model(
        CoverageStateGCRPACREConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=0.001)
    sum(parameter.sum() for parameter in model_a.parameters()).backward()
    optimizer_a.step()

    model_b = deepcopy(model_a)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=0.001)
    optimizer_b.load_state_dict(deepcopy(optimizer_a.state_dict()))
    with torch.no_grad():
        flattened = model_b.joint_state_weight.view(-1)
        assert flattened.numel() >= 2
        assert flattened[0].item() != flattened[1].item()
        first = flattened[0].clone()
        flattened[0].copy_(flattened[1])
        flattened[1].copy_(first)

    parameter_a, optimizer_state_a, step_a = (
        training.gcr_pacre_training_state_summary_fingerprint(
            model_a,
            optimizer_a,
        )
    )
    parameter_b, optimizer_state_b, step_b = (
        training.gcr_pacre_training_state_summary_fingerprint(
            model_b,
            optimizer_b,
        )
    )
    assert parameter_a != parameter_b
    assert optimizer_state_a == optimizer_state_b
    assert step_a == step_b == 1


@pytest.mark.parametrize(
    ("seed", "role"),
    ((42, "primary"), (43, "training_integrity_only")),
)
def test_generated_formal800_role_firewall_and_artifact_roundtrip(
    seed: int,
    role: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = make_training_scalar_cache()
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=seed,
            epochs=800,
            steps_per_epoch=40,
        ),
    )
    bounded_cache = save_formal_cache_neutral_artifact_new(
        cache,
        (tmp_path / f"bounded-source-{seed}.pt").resolve(),
        cache_id="paired-bounded400-full-D_R-materialization",
    )
    formal_cache = save_formal_cache_neutral_artifact_new(
        deepcopy(cache),
        (tmp_path / f"formal-cache-{seed}.pt").resolve(),
        cache_id=f"formal800-seed{seed}-{role}-full-D_R-cache",
    )
    bounded = _forged_bounded(
        semantic_cache_fingerprint=cache.cache_fingerprint,
        neutral_payload_fingerprint=(
            bounded_cache.neutral_payload_fingerprint
        ),
        materialization_receipt_fingerprint=(
            bounded_cache.receipt_fingerprint
        ),
    )
    access = _access(
        f"formal800_seed{seed}_{role}",
        [
            {
                "split": "D_R",
                "logical_id": formal_cache.cache_id,
                "purpose": (
                    "Formal800_seed42_primary_training_cache"
                    if seed == 42
                    else "Formal800_seed43_training_integrity_cache"
                ),
                "source_fingerprint": formal_cache.file_sha256,
            }
        ],
    )
    monkeypatch.setattr(
        formal_training,
        "require_verified_oof_decision",
        lambda value: value,
    )
    monkeypatch.setattr(
        formal_training,
        "require_verified_bounded_decision",
        lambda value: value,
    )
    monkeypatch.setattr(
        training,
        "train_coverage_state_objective",
        _fake_public_trainer(),
    )
    oof = _forged_oof()
    chain_config = _generated_formal_chain(
        seed=seed,
        cache=cache,
        formal_cache=formal_cache,
        bounded=bounded,
        oof=oof,
        access=access,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    authorization = formal_training.prepare_gcr_pacre_formal_authorization(
        seed=seed,
        role=role,
        oof_decision=oof,
        bounded_decision=bounded,
        access_audit=access,
        cache_artifact=formal_cache,
        chain_config=chain_config,
        dataset_free_receipt_fingerprint=_fp("dataset-free"),
        d_r_structural_receipt_fingerprint=_fp("structural"),
        cache=cache,
        schedule=schedule,
        evaluator=_GeneratedFormalEvaluator(),
    )
    run_start_token = create_gcr_pacre_formal_run_start_marker(
        authorization
    )
    _assert_cross_process_marker_replay_rejected(
        run_start_token.marker_path
    )
    schedule_artifact = save_gcr_pacre_formal_schedule_atomic(
        Path(
            str(
                authorization.chain_run_binding[
                    "schedule_artifact_path"
                ]
            )
        ),
        authorization=authorization,
    )
    result = formal_training.run_gcr_pacre_formal_800(
        authorization,
        run_start_token=run_start_token,
        device="cpu",
    )
    terminal_receipt = save_gcr_pacre_formal_terminal_atomic(
        Path(
            str(
                authorization.chain_run_binding[
                    "terminal_artifact_directory"
                ]
            )
        ),
        formal_result=result,
    )
    loaded = load_and_verify_gcr_pacre_formal_terminal(
        Path(str(terminal_receipt["model_file_absolute_path"])).parent,
        expected_receipt=terminal_receipt,
    )
    outer = build_formal_evidence_receipt(
        result,
        schedule_artifact=schedule_artifact,
        terminal_artifact_receipt=terminal_receipt,
    )

    assert result.training_result.completed_updates == 32_000
    assert loaded.receipt["model_fingerprint"] == (
        result.training_receipt.final_model_fingerprint
    )
    assert outer["schema_version"].endswith("evidence-v6")
    assert outer["training_trace_artifact"] == (
        result.training_trace_artifact
    )
    assert outer["cache_artifact"] == formal_cache.payload
    assert outer["run_start_artifact"]["marker_fingerprint"] == (
        run_start_token.marker_fingerprint
    )
    assert outer["source_closure"]["source_hashes"] == dict(
        result.source_hashes_after
    )
    assert outer["D_V_payload_accessed"] is False
    assert outer["D_T_payload_accessed"] is False
    if seed == 43:
        assert result.training_receipt.selection_effect == "none"
        assert (
            result.training_receipt.eligible_for_future_D_V_authorization_after_all_external_prerequisites
            is False
        )
        object.__setattr__(
            result.training_bundle.receipt,
            "eligible_for_future_D_V_authorization_after_all_external_prerequisites",
            True,
        )
        with pytest.raises(
            PermissionError,
            match="seed43 evaluation/selection firewall changed",
        ):
            result.verify_unchanged()


def test_formal_cache_pair_and_arbitrary_payload_binding_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = make_training_scalar_cache()
    first = save_formal_cache_neutral_artifact_new(
        cache,
        (tmp_path / "cache-first.pt").resolve(),
        cache_id="formal800-seed42-primary-full-D_R-cache",
    )
    second = save_formal_cache_neutral_artifact_new(
        deepcopy(cache),
        (tmp_path / "cache-second.pt").resolve(),
        cache_id=(
            "formal800-seed43-training_integrity_only-full-D_R-cache"
        ),
    )
    pair = verify_formal_cache_pair_independence(first, second)
    assert pair.payload["checks"]["same_semantic_cache_fingerprint"] is True
    assert pair.payload["checks"]["different_device_inode"] is True
    assert (
        pair.payload["checks"]["actual_loaded_tensor_storages_disjoint"]
        is True
    )

    envelope = build_formal_cache_neutral_envelope(cache)
    arbitrary = deepcopy(envelope)
    tensor_map = arbitrary["payload"]["tensors"]
    first_name = next(iter(tensor_map))
    tensor_map[first_name] = tensor_map[first_name].clone()
    tensor_map[first_name].view(-1)[0] += 1
    for row in arbitrary["payload"]["tensor_ledger"]:
        if row["logical_path"] == first_name:
            row["content_fingerprint"] = tensor_content_fingerprint(
                tensor_map[first_name]
            )
    arbitrary_path = (tmp_path / "arbitrary.pt").resolve()
    torch.save(arbitrary, arbitrary_path)
    arbitrary_token = verify_formal_cache_artifact(
        arbitrary_path,
        cache_id="formal800-seed42-primary-full-D_R-cache",
        expected_semantic_cache_fingerprint=cache.cache_fingerprint,
    )
    assert (
        arbitrary_token.semantic_cache_fingerprint
        == first.semantic_cache_fingerprint
    )
    assert (
        arbitrary_token.neutral_payload_fingerprint
        != first.neutral_payload_fingerprint
    )

    bounded = _forged_bounded(
        semantic_cache_fingerprint=cache.cache_fingerprint,
        neutral_payload_fingerprint=first.neutral_payload_fingerprint,
        materialization_receipt_fingerprint=first.receipt_fingerprint,
    )
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=800,
            steps_per_epoch=40,
        ),
    )
    access = _access(
        "formal800_seed42_primary",
        [
            {
                "split": "D_R",
                "logical_id": arbitrary_token.cache_id,
                "purpose": "Formal800_seed42_primary_training_cache",
                "source_fingerprint": arbitrary_token.file_sha256,
            }
        ],
    )
    monkeypatch.setattr(
        formal_training,
        "require_verified_oof_decision",
        lambda value: value,
    )
    monkeypatch.setattr(
        formal_training,
        "require_verified_bounded_decision",
        lambda value: value,
    )
    oof = _forged_oof()
    second_access = _access(
        "formal800_seed43_training_integrity_only",
        [
            {
                "split": "D_R",
                "logical_id": second.cache_id,
                "purpose": (
                    "Formal800_seed43_training_integrity_cache"
                ),
                "source_fingerprint": second.file_sha256,
            }
        ],
    )
    monkeypatch.setattr(
        "cure_lite_v24.formal_run_start._required_runtime_root",
        lambda: (tmp_path / "runtime").resolve(),
    )
    monkeypatch.setattr(
        "cure_lite_v24.formal_run_start.require_verified_oof_decision",
        lambda value: value,
    )
    monkeypatch.setattr(
        "cure_lite_v24.formal_run_start.require_verified_bounded_decision",
        lambda value: value,
    )
    chain_config = seal_gcr_pacre_formal_chain_config_new(
        oof_decision=oof,
        bounded_decision=bounded,
        seed42_access_audit=access,
        seed43_access_audit=second_access,
        seed42_cache_artifact=first,
        seed43_cache_artifact=second,
        dataset_free_receipt_fingerprint=_fp("dataset-free"),
        d_r_structural_receipt_fingerprint=_fp("structural"),
        seed42_device="cpu",
        seed43_device="cpu",
    )
    with pytest.raises(PermissionError, match="differs from verified"):
        formal_training.prepare_gcr_pacre_formal_authorization(
            seed=42,
            role="primary",
            oof_decision=oof,
            bounded_decision=bounded,
            access_audit=access,
            cache_artifact=arbitrary_token,
            chain_config=chain_config,
            dataset_free_receipt_fingerprint=_fp("dataset-free"),
            d_r_structural_receipt_fingerprint=_fp("structural"),
            cache=cache,
            schedule=schedule,
            evaluator=_GeneratedFormalEvaluator(),
        )


def test_forged_protocol_tokens_are_rejected_before_runner_allocation(
    tmp_path: Path,
) -> None:
    cache = make_training_scalar_cache()
    second = deepcopy(cache)
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=10,
            steps_per_epoch=40,
        ),
    )
    full_cache = save_formal_cache_neutral_artifact_new(
        cache,
        (tmp_path / "full-cache.pt").resolve(),
        cache_id="paired-bounded400-full-D_R-materialization",
    )
    access = _access(
        "paired_bounded400",
        [
            {
                "split": "D_R",
                "logical_id": full_cache.cache_id,
                "purpose": (
                    "paired_bounded400_full_D_R_materialization"
                ),
                "source_fingerprint": full_cache.file_sha256,
            }
        ],
    )
    with pytest.raises(TypeError, match="oof_decision"):
        prepare_gcr_pacre_paired_bounded_authorization(
            oof_decision=_forged_oof(),
            access_audit=access,
            full_d_r_cache_artifact=full_cache,
            chain_config=VerifiedGCRPACREBoundedChainConfig(
                payload_json=canonical_json({}),
                path=str(tmp_path / "forged.json"),
                file_sha256=_fp("forged-chain-file"),
                config_fingerprint=_fp("forged-chain"),
                source_closure_fingerprint=_fp("forged-source"),
                _issuer=object(),
            ),
            dataset_free_receipt_fingerprint=_fp("dataset-free"),
            d_r_structural_receipt_fingerprint=_fp("structural"),
            control_cache=cache,
            candidate_cache=second,
            schedule=schedule,
            candidate_config=CoverageStateGCRPACREConfig(
                feature_channels=2,
                feature_stride=2,
                width=4,
            ),
            evaluator=_GeneratedBoundedEvaluator(),
        )
    assert not (tmp_path / "forbidden-output").exists()


def test_unified_transitive_source_dependency_tamper_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = make_training_scalar_cache()
    second = deepcopy(cache)
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=10,
            steps_per_epoch=40,
        ),
    )
    full_cache = save_formal_cache_neutral_artifact_new(
        cache,
        (tmp_path / "source-closure-full-cache.pt").resolve(),
        cache_id="paired-bounded400-full-D_R-materialization",
    )
    access = _access(
        "paired_bounded400",
        [
            {
                "split": "D_R",
                "logical_id": full_cache.cache_id,
                "purpose": (
                    "paired_bounded400_full_D_R_materialization"
                ),
                "source_fingerprint": full_cache.file_sha256,
            }
        ],
    )
    monkeypatch.setattr(
        "cure_lite_v24.bounded_runner.require_verified_oof_decision",
        lambda value: value,
    )
    oof = _forged_oof()
    chain_config = _generated_bounded_chain(
        oof=oof,
        access=access,
        full_cache=full_cache,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
    )
    authorization = prepare_gcr_pacre_paired_bounded_authorization(
        oof_decision=oof,
        access_audit=access,
        full_d_r_cache_artifact=full_cache,
        chain_config=chain_config,
        dataset_free_receipt_fingerprint=_fp("dataset-free"),
        d_r_structural_receipt_fingerprint=_fp("structural"),
        control_cache=cache,
        candidate_cache=second,
        schedule=schedule,
        candidate_config=CoverageStateGCRPACREConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        ),
        evaluator=_GeneratedBoundedEvaluator(),
    )
    original = source_closure.file_sha256

    def tampered(path: Path) -> str:
        if path.name == "metrics.py":
            return _fp("tampered-transitive-metrics")
        return original(path)

    monkeypatch.setattr(source_closure, "file_sha256", tampered)
    with pytest.raises(
        (RuntimeError, PermissionError),
        match="source closure changed|chain config identity changed",
    ):
        authorization.verify_unchanged()


def test_fixed_cache_only_evaluator_persists_native_and_g1_distributions(
) -> None:
    cache = make_training_scalar_cache()
    model = build_gcr_pacre_training_model(
        CoverageStateGCRPACREConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        )
    )
    evaluator = FrozenGCRPACREDREvaluator()
    native = evaluator.evaluate(
        model,
        cache,
        arm=GCR_PACRE_CANDIDATE_ARM,
        checkpoint="initial",
        forward_mode="native",
    )
    forced = evaluator.evaluate(
        model,
        cache,
        arm=GCR_PACRE_CANDIDATE_ARM,
        checkpoint="initial",
        forward_mode=GCR_PACRE_FORCED_G1_MODE,
    )

    for value in (native, forced):
        distribution = value.gate_role_distribution
        assert distribution is not None
        assert set(distribution) == {
            "schema_version",
            "endpoint_counts",
            "target_G",
            "background_G",
            "target_E",
            "background_E",
        }
        assert sum(distribution["endpoint_counts"].values()) == (
            distribution["target_G"]["count"]
            + distribution["background_G"]["count"]
        )
    assert forced.gate_role_distribution["target_G"]["minimum"] == 1.0
    assert forced.gate_role_distribution["target_G"]["maximum"] == 1.0
    assert forced.gate_role_distribution["background_G"]["minimum"] == 1.0
    assert forced.gate_role_distribution["background_G"]["maximum"] == 1.0


def test_off_registry_cli_factory_is_rejected_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = False

    def forbidden_import(name: str):
        nonlocal imported
        imported = True
        raise AssertionError(f"unexpected import of {name}")

    monkeypatch.setattr(formal_cli.importlib, "import_module", forbidden_import)
    with pytest.raises(PermissionError, match="frozen real input factory"):
        formal_cli._factory("malicious.module:side_effect")
    with pytest.raises(PermissionError, match="frozen real input factory"):
        bounded_cli._load_authorization(
            "malicious.module:side_effect",
            chain_config=object(),  # type: ignore[arg-type]
        )
    assert imported is False


def test_runtime_source_audit_fails_before_persistent_marker_or_training(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A late-loaded missing module must not consume either real attempt."""

    events: list[str] = []

    def forbidden_marker(*args, **kwargs):
        del args, kwargs
        events.append("marker")
        raise AssertionError("persistent marker was created")

    def forbidden_training(*args, **kwargs):
        del args, kwargs
        events.append("model_or_optimizer")
        raise AssertionError("training allocated runtime state")

    loaded = source_closure.gcr_pacre_v24_loaded_runtime_source_paths()
    monkeypatch.setattr(
        source_closure,
        "gcr_pacre_v24_loaded_runtime_source_paths",
        lambda: (
            *loaded,
            "cure_lite_v24/sentinel_missing_runtime_module.py",
        ),
    )
    bounded_output = tmp_path.resolve()
    bounded_authorization_path = bounded_output / "authorization.json"
    bounded_chain = SimpleNamespace(
        payload={
            "authorization_artifact_path": str(
                bounded_authorization_path
            )
        }
    )
    bounded_authorization = SimpleNamespace(
        output_directory=str(bounded_output),
        requested_device="cpu",
    )
    monkeypatch.setattr(
        bounded_cli,
        "load_and_verify_gcr_pacre_bounded_chain_config",
        lambda path: bounded_chain,
    )
    monkeypatch.setattr(
        bounded_cli,
        "_load_authorization",
        lambda specification, *, chain_config: bounded_authorization,
    )
    monkeypatch.setattr(
        bounded_cli,
        "create_gcr_pacre_bounded_run_start_marker",
        forbidden_marker,
    )
    monkeypatch.setattr(
        bounded_cli,
        "run_gcr_pacre_paired_bounded_400",
        forbidden_training,
    )
    with pytest.raises(RuntimeError, match="sentinel_missing"):
        bounded_cli.run_once(SimpleNamespace(
            chain_config="/fixed/chain.json",
            input_factory=bounded_cli.FROZEN_INPUT_FACTORY,
            output=str(bounded_output),
            authorization_out=str(bounded_authorization_path),
            device="cpu",
        ))
    assert events == []

    events.clear()
    formal_authorization = SimpleNamespace(
        output_directory=str(bounded_output),
        requested_device="cpu",
    )
    monkeypatch.setattr(
        formal_cli,
        "create_gcr_pacre_formal_run_start_marker",
        forbidden_marker,
    )
    monkeypatch.setattr(
        formal_cli,
        "run_gcr_pacre_formal_800",
        forbidden_training,
    )
    with pytest.raises(RuntimeError, match="sentinel_missing"):
        formal_cli._run_one(
            formal_authorization,  # type: ignore[arg-type]
            output=bounded_output,
            device="cpu",
        )
    assert events == []


def test_formal_cli_has_no_sequential_production_entry() -> None:
    with pytest.raises(SystemExit):
        formal_cli.parse_args([
            "run-sequential",
            "--input-factory",
            formal_cli.FROZEN_INPUT_FACTORY,
            "--chain-config",
            "/fixed/formal.json",
            "--output",
            "/fixed/output",
        ])


def test_chain_and_formal_clis_expose_no_epoch_or_uplift_override() -> None:
    for forbidden in (
        ("--epochs", "1"),
        ("--steps-per-epoch", "1"),
        ("--margin", "0.01"),
    ):
        with pytest.raises(SystemExit):
            chain_setup_cli.parse_args([
                "seal-formal",
                "--seed42-device",
                "cuda:0",
                "--seed43-device",
                "cuda:1",
                *forbidden,
            ])
    for forbidden in (
        ("--fixed-uplift-threshold", "0.01"),
        ("--steps-per-epoch", "1"),
        ("--margin", "0.01"),
    ):
        with pytest.raises(SystemExit):
            formal_cli.parse_args([
                "run-seed",
                "--input-factory",
                formal_cli.FROZEN_INPUT_FACTORY,
                "--chain-config",
                "/fixed/formal.json",
                "--seed",
                "42",
                "--device",
                "cuda:0",
                "--output",
                "/fixed/output",
                *forbidden,
            ])
    for forbidden in (
        ("--steps-per-epoch", "1"),
        ("--margin", "0.01"),
    ):
        with pytest.raises(SystemExit):
            bounded_cli.parse_args([
                "--input-factory",
                bounded_cli.FROZEN_INPUT_FACTORY,
                "--chain-config",
                "/fixed/bounded.json",
                "--run-once",
                "--output",
                "/fixed/output",
                "--authorization-out",
                "/fixed/authorization.json",
                "--device",
                "cuda:0",
                *forbidden,
            ])
        with pytest.raises(SystemExit):
            oof_cli._parser().parse_args([
                "run-fold",
                "--fold-id",
                "0",
                "--device",
                "cuda:0",
                "--execute-real-dr-oof",
                *forbidden,
            ])


def test_real_factory_imports_and_calls_frozen_oof_verifier_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the cross-process OOF verifier module/signature."""

    import cure_lite_v24.oof_run_start as oof_run_start
    import cure_lite_v24.oof_runner as oof_runner

    split = object()
    execution = object()
    decision = SimpleNamespace(decision_fingerprint=_fp("oof"))
    calls: list[tuple[str, object, object | None]] = []

    def verify_split(path, *, repository_root):
        assert path == real_input_factory._OOF_SPLIT_PREREGISTRATION
        assert repository_root == real_input_factory._REPOSITORY_ROOT
        calls.append(("split", path, repository_root))
        return split

    def load_authorization(*, verified_split):
        assert verified_split is split
        calls.append(("authorization", verified_split, None))
        return execution

    def verify_result(*, verified_split, execution_authorization):
        assert verified_split is split
        assert execution_authorization is execution
        calls.append(
            ("result", verified_split, execution_authorization)
        )
        return decision

    monkeypatch.setattr(
        real_input_factory,
        "verify_oof4_split_preregistration",
        verify_split,
    )
    monkeypatch.setattr(
        oof_run_start,
        "load_and_verify_real_oof4_execution_authorization",
        load_authorization,
    )
    monkeypatch.setattr(
        oof_runner,
        "verify_real_oof4_result_artifact",
        verify_result,
    )
    monkeypatch.setattr(
        real_input_factory,
        "require_verified_oof_decision",
        lambda value: value,
    )
    assert real_input_factory._rebuild_oof_chain() == (
        decision,
        execution,
        split,
    )
    assert [name for name, _, _ in calls] == [
        "split",
        "authorization",
        "result",
    ]


def test_oof_source_audit_fails_before_fold_marker_or_model_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    split = object()
    authorization = object()

    class Parser:
        @staticmethod
        def parse_args():
            return SimpleNamespace(
                mode="run-fold",
                fold_id=0,
                device="cpu",
                execute_real_dr_oof=True,
            )

    def forbidden_fold(*args, **kwargs):
        del args, kwargs
        events.append("marker_or_model")
        raise AssertionError("OOF fold execution started")

    loaded = source_closure.gcr_pacre_v24_loaded_runtime_source_paths()
    monkeypatch.setattr(
        source_closure,
        "gcr_pacre_v24_loaded_runtime_source_paths",
        lambda: (
            *loaded,
            "cure_lite_v24/sentinel_missing_oof_runtime_module.py",
        ),
    )
    monkeypatch.setattr(oof_cli, "_parser", lambda: Parser())
    monkeypatch.setattr(oof_cli, "_split", lambda: split)
    monkeypatch.setattr(
        oof_cli,
        "load_and_verify_real_oof4_execution_authorization",
        lambda *, verified_split: authorization,
    )
    monkeypatch.setattr(
        oof_cli,
        "_bound_context",
        lambda value: (object(), object(), object(), object()),
    )
    monkeypatch.setattr(oof_cli, "run_real_oof4_fold", forbidden_fold)
    with pytest.raises(RuntimeError, match="sentinel_missing_oof"):
        oof_cli.main()
    assert events == []


def test_oof_invalid_cuda_fails_before_context_marker_or_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    split = object()
    authorization = object()

    class Parser:
        @staticmethod
        def parse_args():
            return SimpleNamespace(
                mode="run-fold",
                fold_id=0,
                device="cuda:999",
                execute_real_dr_oof=True,
            )

    monkeypatch.setattr(oof_cli, "_parser", lambda: Parser())
    monkeypatch.setattr(oof_cli, "_split", lambda: split)
    monkeypatch.setattr(
        oof_cli,
        "load_and_verify_real_oof4_execution_authorization",
        lambda *, verified_split: authorization,
    )
    monkeypatch.setattr(
        oof_runner.torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        oof_runner.torch.cuda,
        "device_count",
        lambda: 1,
    )

    def forbidden_context(*args, **kwargs):
        del args, kwargs
        events.append("D_R_context")
        raise AssertionError("D_R context must not be opened")

    def forbidden_fold(*args, **kwargs):
        del args, kwargs
        events.append("marker_or_model")
        raise AssertionError("OOF fold execution must not start")

    monkeypatch.setattr(oof_cli, "_bound_context", forbidden_context)
    monkeypatch.setattr(oof_cli, "run_real_oof4_fold", forbidden_fold)
    with pytest.raises(ValueError, match="out of range"):
        oof_cli.main()
    assert events == []
    assert list(tmp_path.iterdir()) == []


def test_formal_process_instance_fingerprint_is_interpreter_scoped() -> None:
    script = (
        "from cure_lite_v24.formal_run_start import "
        "_PROCESS_INSTANCE_FINGERPRINT;"
        "print(_PROCESS_INSTANCE_FINGERPRINT)"
    )
    values = [
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip()
        for _ in range(2)
    ]
    assert all(len(value) == 64 for value in values)
    assert values[0] != values[1]
    assert (
        formal_run_start._PROCESS_INSTANCE_FINGERPRINT
        == formal_run_start._PROCESS_INSTANCE_FINGERPRINT
    )
