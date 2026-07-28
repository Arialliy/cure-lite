from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CoverageStatePhaseAlignedEvidenceTransportConfig,
)
from cure_lite.coverage_state_sobolev import CSLF_PMOPE_POLICY
from cure_lite.experiment import coverage_state_paet_formal_attempt as module
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_POPULATION_SCHEMA,
    COVERAGE_STATE_BOUNDED_SELECTION_POLICY,
)
from cure_lite.experiment.coverage_state_paet_formal_training import (
    COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT,
    COVERAGE_STATE_PAET_FORMAL_RUN_ID,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _fingerprinted(payload: dict[str, object], field: str = "receipt_fingerprint") -> dict[str, object]:
    return {**payload, field: stable_fingerprint(payload)}


def _passing_diagnostic(
    checkpoint_fingerprint: str,
) -> module.CoverageStateZeroLevelEvaluationResult:
    config = module.CoverageStateZeroLevelEvaluationConfig(
        input_representation=module.COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
    )
    ledger: list[module.CoverageStateDiagnosticStateLedger] = []
    first_output_by_input: dict[
        str, module.CoverageStateDiagnosticStateLedger
    ] = {}
    next_forward_index = 0

    def digest(namespace: str, identity: str) -> str:
        return stable_fingerprint(
            {"namespace": namespace, "identity": identity}
        )

    def state(
        *,
        state_id: str,
        role: str,
        endpoint: str,
        input_identity: str,
        independent_exact_replay: bool = False,
        completion_fingerprint: str | None = None,
    ) -> module.CoverageStateDiagnosticStateLedger:
        nonlocal next_forward_index
        input_fingerprint = digest("actual-input", input_identity)
        cached = (
            None
            if independent_exact_replay
            else first_output_by_input.get(input_fingerprint)
        )
        if cached is not None:
            row = module.CoverageStateDiagnosticStateLedger(
                state_id=state_id,
                role=role,
                endpoint=endpoint,
                actual_input_fingerprint=input_fingerprint,
                model_forward_index=cached.model_forward_index,
                reused_actual_input=True,
                independent_exact_replay=False,
                field_fingerprint=cached.field_fingerprint,
                completion_fingerprint=cached.completion_fingerprint,
                final_fingerprint=cached.final_fingerprint,
            )
        else:
            row = module.CoverageStateDiagnosticStateLedger(
                state_id=state_id,
                role=role,
                endpoint=endpoint,
                actual_input_fingerprint=input_fingerprint,
                model_forward_index=next_forward_index,
                reused_actual_input=False,
                independent_exact_replay=independent_exact_replay,
                field_fingerprint=digest("field", input_identity),
                completion_fingerprint=(
                    completion_fingerprint
                    if completion_fingerprint is not None
                    else digest("completion", input_identity)
                ),
                final_fingerprint=digest("final", input_identity),
            )
            next_forward_index += 1
            first_output_by_input.setdefault(input_fingerprint, row)
        ledger.append(row)
        return row

    target_pixels = (21,) * 15 + (20,)
    natural: list[module.CoverageStateNaturalZeroLevelDiagnostic] = []
    for index, pixels in enumerate(target_pixels):
        negative_pixels = 19 if index == 0 else pixels
        gate_passed = index != 0
        record_id = digest("natural-record", f"miss-{index:02d}")
        state(
            state_id=f"natural:{record_id}",
            role="factual_miss",
            endpoint="natural",
            input_identity=f"natural-miss-{index:02d}",
        )
        natural.append(
            module.CoverageStateNaturalZeroLevelDiagnostic(
                record_id=record_id,
                sample_id=f"miss-image-{index:02d}",
                state_kind="factual_miss",
                field_valid_pixels=65_536,
                invalid_completion_pixels=0,
                negative_pixels=negative_pixels,
                negative_components=1,
                focus_target_pixels=pixels,
                focus_target_negative_pixels=negative_pixels,
                target_negative_fraction_hex=(
                    negative_pixels / pixels
                ).hex(),
                target_recovered=True,
                connected_support_components=1,
                connected_support_components_hit=1,
                connected_support_recall_hex=1.0.hex(),
                gate_passed=gate_passed,
            )
        )
    for index in range(16):
        record_id = digest(
            "natural-record", f"no-miss-{index:02d}"
        )
        state(
            state_id=f"natural:{record_id}",
            role="factual_no_miss",
            endpoint="natural",
            input_identity=f"natural-no-miss-{index:02d}",
        )
        natural.append(
            module.CoverageStateNaturalZeroLevelDiagnostic(
                record_id=record_id,
                sample_id=f"no-miss-image-{index:02d}",
                state_kind="factual_no_miss",
                field_valid_pixels=65_536,
                invalid_completion_pixels=0,
                negative_pixels=0,
                negative_components=0,
                focus_target_pixels=0,
                focus_target_negative_pixels=0,
                target_negative_fraction_hex=None,
                target_recovered=None,
                connected_support_components=0,
                connected_support_components_hit=0,
                connected_support_recall_hex=None,
                gate_passed=True,
            )
        )

    def pair_diagnostic(
        pair_id: str,
        *,
        pair_kind: str,
        optimizer_role: str,
        actual_inputs_equal: bool,
        field_exact_equal: bool,
        completion_exact_equal: bool,
        final_exact_equal: bool,
        added_pixels: int = 0,
        clean: bool = False,
    ) -> module.CoverageStatePairZeroLevelDiagnostic:
        return module.CoverageStatePairZeroLevelDiagnostic(
            pair_id=pair_id,
            sample_id=f"pair-image-{pair_id}",
            pair_kind=pair_kind,
            optimizer_role=optimizer_role,
            scalar_hidden=False,
            actual_inputs_equal=actual_inputs_equal,
            invalid_completion_pixels_plus=0,
            invalid_completion_pixels_minus=0,
            field_exact_equal=field_exact_equal,
            completion_exact_equal=completion_exact_equal,
            final_exact_equal=final_exact_equal,
            maximum_abs_field_difference_hex=(
                0.0 if field_exact_equal else 1.0
            ).hex(),
            added_target_pixels=added_pixels,
            added_target_components=1 if added_pixels else 0,
            minus_added_target_negative_pixels=added_pixels,
            minus_added_target_all_negative=True if clean else None,
            response_sign_pixels=added_pixels if clean else 0,
            response_sign_correct_pixels=added_pixels if clean else 0,
            response_sign_all_correct=True if clean else None,
            plus_writable_false_island_components=0 if clean else None,
            new_negative_pixels=added_pixels,
            new_negative_components=1 if added_pixels else 0,
            removed_footprint_negative_pixels=0,
            new_completion_pixels=added_pixels,
            new_completion_outside_added_target_pixels=0 if clean else None,
            new_completion_components=1 if added_pixels else 0,
            compact_support_exact_equal=True if clean else None,
            compact_support_component_match=True if clean else None,
            compact_support_passed=True if clean else None,
            defined_metrics_passed=True,
            gate_passed=True,
        )

    pairs: list[module.CoverageStatePairZeroLevelDiagnostic] = []
    clean_pixels = (9,) * 15 + (14,)
    for index, pixels in enumerate(clean_pixels):
        pair_id = digest("pair-record", f"clean-{index:02d}")
        state(
            state_id=f"pair:{pair_id}:plus",
            role="clean_positive",
            endpoint="plus",
            input_identity=f"{pair_id}-plus",
        )
        state(
            state_id=f"pair:{pair_id}:minus",
            role="clean_positive",
            endpoint="minus",
            input_identity=f"{pair_id}-minus",
        )
        pairs.append(
            pair_diagnostic(
                pair_id,
                pair_kind="clean_positive",
                optimizer_role="clean_positive",
                actual_inputs_equal=False,
                field_exact_equal=False,
                completion_exact_equal=False,
                final_exact_equal=False,
                added_pixels=pixels,
                clean=True,
            )
        )

    for index in range(16):
        pair_id = digest("pair-record", f"component-{index:02d}")
        plus = state(
            state_id=f"pair:{pair_id}:plus",
            role="component_null",
            endpoint="plus",
            input_identity=(
                f"natural-miss-{index:02d}"
                if index < 14
                else f"{pair_id}-plus"
            ),
        )
        state(
            state_id=f"pair:{pair_id}:minus",
            role="component_null",
            endpoint="minus",
            input_identity=f"{pair_id}-minus",
            completion_fingerprint=plus.completion_fingerprint,
        )
        pairs.append(
            pair_diagnostic(
                pair_id,
                pair_kind="component_null",
                optimizer_role="component_null",
                actual_inputs_equal=False,
                field_exact_equal=False,
                completion_exact_equal=True,
                final_exact_equal=False,
            )
        )

    for index in range(16):
        pair_id = digest("pair-record", f"identity-{index:02d}")
        state(
            state_id=f"pair:{pair_id}:plus",
            role="identity_diagnostic",
            endpoint="plus",
            input_identity=f"{pair_id}-shared",
        )
        state(
            state_id=f"pair:{pair_id}:minus",
            role="identity_diagnostic",
            endpoint="minus",
            input_identity=f"{pair_id}-shared",
            independent_exact_replay=True,
        )
        pairs.append(
            pair_diagnostic(
                pair_id,
                pair_kind="identity_null",
                optimizer_role="identity_diagnostic",
                actual_inputs_equal=True,
                field_exact_equal=True,
                completion_exact_equal=True,
                final_exact_equal=True,
            )
        )

    pair_id = digest("pair-record", "diagnostic-00")
    diagnostic_plus = state(
        state_id=f"pair:{pair_id}:plus",
        role="diagnostic_only",
        endpoint="plus",
        input_identity=f"{pair_id}-plus",
    )
    state(
        state_id=f"pair:{pair_id}:minus",
        role="diagnostic_only",
        endpoint="minus",
        input_identity=f"{pair_id}-minus",
        independent_exact_replay=True,
        completion_fingerprint=(
            diagnostic_plus.completion_fingerprint
        ),
    )
    pairs.append(
        pair_diagnostic(
            pair_id,
            pair_kind="component_null",
            optimizer_role="diagnostic_only",
            actual_inputs_equal=False,
            field_exact_equal=False,
            completion_exact_equal=True,
            final_exact_equal=False,
        )
    )

    # The bounded cache exposes natural records and pairs in canonical
    # identity order.  Rebuild the execution ledger in that same order so
    # forward indices and reuse flags follow the evaluator's actual loop.
    natural.sort(key=lambda row: row.record_id)
    pairs.sort(key=lambda row: row.pair_id)
    old_ledger_by_state_id = {
        row.state_id: row for row in ledger
    }
    ordered_state_ids = [
        *(f"natural:{row.record_id}" for row in natural),
        *(
            state_id
            for row in pairs
            for state_id in (
                f"pair:{row.pair_id}:plus",
                f"pair:{row.pair_id}:minus",
            )
        ),
    ]
    ledger = []
    first_output_by_input = {}
    next_forward_index = 0
    for state_id in ordered_state_ids:
        old = old_ledger_by_state_id[state_id]
        first = first_output_by_input.get(
            old.actual_input_fingerprint
        )
        reused = (
            first is not None
            and not old.independent_exact_replay
        )
        if reused:
            assert first is not None
            row = module.CoverageStateDiagnosticStateLedger(
                state_id=old.state_id,
                role=old.role,
                endpoint=old.endpoint,
                actual_input_fingerprint=(
                    old.actual_input_fingerprint
                ),
                model_forward_index=first.model_forward_index,
                reused_actual_input=True,
                independent_exact_replay=False,
                field_fingerprint=first.field_fingerprint,
                completion_fingerprint=(
                    first.completion_fingerprint
                ),
                final_fingerprint=first.final_fingerprint,
            )
        else:
            row = module.CoverageStateDiagnosticStateLedger(
                state_id=old.state_id,
                role=old.role,
                endpoint=old.endpoint,
                actual_input_fingerprint=(
                    old.actual_input_fingerprint
                ),
                model_forward_index=next_forward_index,
                reused_actual_input=False,
                independent_exact_replay=(
                    old.independent_exact_replay
                ),
                field_fingerprint=old.field_fingerprint,
                completion_fingerprint=(
                    old.completion_fingerprint
                ),
                final_fingerprint=old.final_fingerprint,
            )
            next_forward_index += 1
            first_output_by_input.setdefault(
                row.actual_input_fingerprint,
                row,
            )
        ledger.append(row)

    assert len(ledger) == 130
    assert len(first_output_by_input) == 100
    assert next_forward_index == 116
    assert sum(row.reused_actual_input for row in ledger) == 14
    assert (
        sum(row.independent_exact_replay for row in ledger) == 17
    )
    assert len(natural) == 32
    assert len(pairs) == 49
    return module.CoverageStateZeroLevelEvaluationResult(
        config=config,
        dataset="IRSTD-1K",
        split="D_R",
        cache_fingerprint=(
            module.COVERAGE_STATE_PAET_V21_BOUNDED_CACHE_FINGERPRINT
        ),
        checkpoint_fingerprint=checkpoint_fingerprint,
        state_ledger=tuple(ledger),
        natural_diagnostics=tuple(natural),
        pair_diagnostics=tuple(pairs),
        diagnostic_state_references=130,
        unique_actual_input_states=100,
        model_forward_invocations=116,
        exact_replay_forward_invocations=17,
        reused_state_references=14,
        backward_calls=0,
        optimizer_steps=0,
        factual_miss_gate_passed=False,
        factual_no_miss_gate_passed=True,
        clean_defined_metrics_passed=True,
        clean_compact_support_gate_passed=True,
        component_null_gate_passed=True,
        identity_null_gate_passed=True,
        scalar_hidden_diagnostic_gate_passed=True,
        bounded_gate_passed=False,
        fail_closed_reasons=(
            "defined_metric_gate_failed:factual_miss",
        ),
    )


def _bounded_population_from_diagnostic(
    diagnostic: module.CoverageStateZeroLevelEvaluationResult,
) -> dict[str, object]:
    natural = diagnostic.natural_diagnostics
    pairs = diagnostic.pair_diagnostics
    source_counts: dict[str, int] = {}
    for row in (*natural, *pairs):
        source_counts[row.sample_id] = source_counts.get(row.sample_id, 0) + 1
    return {
        "schema_version": COVERAGE_STATE_BOUNDED_POPULATION_SCHEMA,
        "selection_policy": COVERAGE_STATE_BOUNDED_SELECTION_POLICY,
        "seed": 42,
        "split": "D_R",
        "source_cache_fingerprint": (
            COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        ),
        "bounded_cache_fingerprint": (
            module.COVERAGE_STATE_PAET_V21_BOUNDED_CACHE_FINGERPRINT
        ),
        "role_count": 16,
        "factual_miss_record_ids": [
            row.record_id
            for row in natural
            if row.state_kind == "factual_miss"
        ],
        "factual_no_miss_record_ids": [
            row.record_id
            for row in natural
            if row.state_kind == "factual_no_miss"
        ],
        "clean_positive_pair_ids": [
            row.pair_id
            for row in pairs
            if row.pair_kind == "clean_positive"
            and row.optimizer_role == "clean_positive"
        ],
        "component_null_pair_ids": [
            row.pair_id
            for row in pairs
            if row.pair_kind == "component_null"
            and row.optimizer_role == "component_null"
        ],
        "identity_null_pair_ids": [
            row.pair_id
            for row in pairs
            if row.pair_kind == "identity_null"
        ],
        "scalar_hidden_diagnostic_pair_ids": [
            row.pair_id
            for row in pairs
            if row.optimizer_role == "diagnostic_only"
        ],
        "source_counts": dict(sorted(source_counts.items())),
        "D_V_accessed": False,
        "D_T_accessed": False,
    }


def _formal_exposure_checks() -> dict[str, bool]:
    checks = {
        f"{population}/{unit}/{statistic}": True
        for population in (
            "factual_miss",
            "factual_no_miss",
            "clean_positive",
            "component_null",
        )
        for unit in ("record", "source")
        for statistic in ("zero_exposure", "ess", "maximum_share")
    }
    checks.update(
        {
            f"{population}/{statistic}": True
            for population in (
                "factual_focus_target",
                "clean_added_target",
            )
            for statistic in ("zero_exposure", "ess", "maximum_share")
        }
    )
    checks.update(
        {
            "selection_exact_budget": True,
            "identity_null_optimizer_exposure": True,
            "diagnostic_only_optimizer_exposure": True,
        }
    )
    assert len(checks) == 33
    return dict(sorted(checks.items()))


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / COVERAGE_STATE_PAET_FORMAL_RUN_ID
    final, receipts = root / "final_model", root / "receipts"
    final.mkdir(parents=True)
    receipts.mkdir()
    for name in module._FINAL_MEMBERS:
        (final / name).write_bytes(name.encode())

    model_config = CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    epoch_logs = tuple(
        {
            "epoch": epoch,
            "completed_updates": (epoch + 1) * 40,
            "objective": "pmope_joint",
        }
        for epoch in range(800)
    )
    diagnostic = _passing_diagnostic("e" * 64)
    bounded_population = _bounded_population_from_diagnostic(diagnostic)
    bounded_population_fingerprint = stable_fingerprint(
        bounded_population
    )
    monkeypatch.setattr(
        module,
        "COVERAGE_STATE_PAET_V21_BOUNDED_POPULATION_FINGERPRINT",
        bounded_population_fingerprint,
    )
    real_inputs_payload = {
        "schema_version": "fixture-real-d-r-inputs-v1",
        "dataset": "IRSTD-1K",
        "split": "D_R",
        "runtime_splits": ["D_R"],
        "source_binding_fingerprint": (
            module.COVERAGE_STATE_PAET_V21_SOURCE_RECEIPT_FINGERPRINT
        ),
        "fingerprints": {
            "scalar_cache": (
                COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
            )
        },
        "execution_policy": {
            "D_V_accessed": False,
            "D_T_accessed": False,
        },
    }
    real_inputs_fingerprint = stable_fingerprint(real_inputs_payload)
    formal_implementation = {"formal.py": "4" * 64}
    bounded_payload = {
        "structural_advancement_passed": True,
        "generic_population_gate_passed": False,
        "dataset_free_gate_passed": True,
        "D_R_gate_passed": True,
        "bounded_evidence_is_performance": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "performance_claim_supported": False,
        "source_binding_fingerprint": (
            module.COVERAGE_STATE_PAET_V21_SOURCE_RECEIPT_FINGERPRINT
        ),
        "full_D_R_scalar_cache_fingerprint": (
            COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        ),
        "full_D_R_scalar_cache_counts": {
            "natural_total": 167,
            "pair_total": 383,
            "clean_positive_optimization_eligible": 206,
            "component_null_total": 17,
            "component_null_optimization_eligible": 16,
            "component_null_diagnostic_only": 1,
            "identity_null_diagnostic": 160,
        },
        "real_inputs_build_fingerprint": real_inputs_fingerprint,
        "bounded_population": bounded_population,
        "bounded_population_fingerprint": (
            bounded_population_fingerprint
        ),
    }
    bounded_seal = SimpleNamespace(
        audit_fingerprint="6" * 64,
        payload=bounded_payload,
        verify_unchanged=lambda: None,
    )
    monkeypatch.setattr(
        module,
        "load_repository_coverage_state_paet_bounded_artifact_seal",
        lambda: bounded_seal,
    )
    authorization_payload = {
        "schema_version": (
            module.COVERAGE_STATE_PAET_FORMAL_AUTHORIZATION_SCHEMA
        ),
        "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
        "scope": "D_R_formal_800",
        "runtime_splits": ["D_R"],
        "bounded_artifact_seal_fingerprint": (
            bounded_seal.audit_fingerprint
        ),
        "bounded_evidence_interpretation": (
            "structural_advancement_only_not_performance"
        ),
        "structural_advancement_passed": True,
        "generic_population_gate_passed": False,
        "dataset_free_gate_passed": True,
        "D_R_identifiability_gate_passed": True,
        "real_inputs_build_fingerprint": real_inputs_fingerprint,
        "source_binding_fingerprint": (
            module.COVERAGE_STATE_PAET_V21_SOURCE_RECEIPT_FINGERPRINT
        ),
        "full_D_R_scalar_cache_fingerprint": (
            COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        ),
        "full_D_R_scalar_cache_counts": (
            bounded_payload["full_D_R_scalar_cache_counts"]
        ),
        "schedule_fingerprint": (
            module.COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT
        ),
        "exposure_gate_fingerprint": (
            module.COVERAGE_STATE_PAET_FORMAL_EXPOSURE_GATE_FINGERPRINT
        ),
        "exposure_gate_checks": _formal_exposure_checks(),
        "budget": {
            "seed": 42,
            "epochs": 800,
            "steps_per_epoch": 40,
            "updates": 32_000,
            "objectives": 1,
        },
        "model_config_fingerprint": stable_fingerprint(
            module._formal_model_config_payload(model_config)
        ),
        "model_class": (
            "CURELitePhaseAlignedEvidenceTransportLevelSet"
        ),
        "expected_parameter_count": 64_064,
        "expected_initial_model_fingerprint": (
            module.COVERAGE_STATE_PAET_FORMAL_INITIAL_MODEL_FINGERPRINT
        ),
        "candidate_objective": "pmope_joint",
        "candidate_objective_policy": CSLF_PMOPE_POLICY,
        "field_threshold_hex": 0.0.hex(),
        "threshold_search_performed": False,
        "formal_implementation_binding": formal_implementation,
        "formal_implementation_fingerprint": stable_fingerprint(
            formal_implementation
        ),
        "training_contract": {
            "from_scratch": True,
            "process_local_single_attempt_claim": True,
            "cross_process_output_claim_required": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "continuation_checkpoint_consumed": False,
            "checkpoint_policy": "final_model_only",
            "intermediate_checkpoint_saved": False,
            "optimizer_state_saved": False,
        },
        "formal_D_R_training_authorized": True,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "calibration_performed": False,
        "inference_performed": False,
        "performance_evaluation_performed": False,
        "performance_claim_supported": False,
        "full_CURE_authorized": False,
        "cross_backbone_authorized": False,
    }
    authorization_fingerprint = stable_fingerprint(
        authorization_payload
    )
    formal_result_payload = {
        "formal": "result",
        "structural_advancement_passed": True,
    }
    artifact = SimpleNamespace(
        artifact_fingerprint="a" * 64,
        receipt_sha256=file_sha256(final / "receipt.json"),
        authorization_fingerprint=authorization_fingerprint,
        formal_result_fingerprint=stable_fingerprint(
            formal_result_payload
        ),
        training_model_fingerprint="d" * 64,
        module_state_fingerprint="e" * 64,
        formal_result_payload=formal_result_payload,
        model_config=model_config,
        epoch_logs=epoch_logs,
        verify_unchanged=lambda: None,
    )
    monkeypatch.setattr(module, "PAET_FORMAL_ATTEMPT_OUTPUT_PATH", root)
    monkeypatch.setattr(
        module,
        "load_coverage_state_paet_formal_artifact",
        lambda *args, **kwargs: artifact,
    )
    source_closure_receipt = {
        "sealed": True,
        "manifest_sha256": "1" * 64,
        "archive_sha256": "2" * 64,
        "content_fingerprint": "3" * 64,
        "file_count": 223,
    }
    monkeypatch.setattr(
        module,
        "verify_coverage_state_paet_formal_source_closure",
        lambda: source_closure_receipt,
    )
    source_closure_fields = module._source_closure_fields(
        source_closure_receipt
    )
    attempt = _fingerprinted(
        {
            "schema_version": module.PAET_FORMAL_ATTEMPT_SCHEMA,
            "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
            "output_repo_path": (
                "runs/irstd1k_stage_a_seed42/"
                f"{COVERAGE_STATE_PAET_FORMAL_RUN_ID}"
            ),
            "seed": 42,
            "epochs": 800,
            "steps_per_epoch": 40,
            "updates": 32_000,
            "device": "cuda:0",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "pause_temperature_c": 82,
            "resume_temperature_c": 75,
            "single_attempt": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "D_R_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            **source_closure_fields,
        }
    )
    _write_json(root / "attempt.json", attempt)
    started = _fingerprinted(
        {
            "schema_version": module.PAET_FORMAL_STARTED_SCHEMA,
            "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
            "status": "started_single_attempt",
            "attempt_fingerprint": attempt["receipt_fingerprint"],
            "output_directory_reusable": False,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "D_R_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )
    _write_json(root / "STARTED.json", started)

    implementation_files = {
        **formal_implementation,
        "implementation.py": "b" * 64,
    }
    config = _fingerprinted(
        {
            "schema_version": module.PAET_FORMAL_RUN_SCHEMA,
            "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
            "output_repo_path": (
                "runs/irstd1k_stage_a_seed42/"
                f"{COVERAGE_STATE_PAET_FORMAL_RUN_ID}"
            ),
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "real_inputs": module._FROZEN_REAL_DR_INPUTS,
            "bounded_artifact_seal_fingerprint": (
                bounded_seal.audit_fingerprint
            ),
            "bounded_evidence_interpretation": (
                "structural_advancement_only_not_performance"
            ),
            "model": {
                "class": (
                    "CURELitePhaseAlignedEvidenceTransportLevelSet"
                ),
                "candidate": "PAET-BFA-v21",
                "input_interface": ["F_b", "O"],
                "config": asdict(model_config),
                "feature_channels": 64,
                "feature_stride": 4,
                "width": 32,
                "parameter_count": 64_064,
                "parameter_tensor_count": 3,
                "candidate_objective": "pmope_joint",
                "single_completion_field": True,
                "field_threshold_hex": 0.0.hex(),
                "threshold_search_performed": False,
            },
            "budget": {
                "seed": 42,
                "epochs": 800,
                "steps_per_epoch": 40,
                "updates": 32_000,
                "objectives": 1,
                "from_scratch": True,
            },
            "full_D_R_contract": {
                "scalar_cache_fingerprint": (
                    COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
                ),
                "formal_schedule_fingerprint": (
                    module.COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT
                ),
            },
            "execution": {
                "device": "cuda:0",
                "CUDA_VISIBLE_DEVICES": "0",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "temperature_wrapper_repo_path": (
                    "tools/run_with_gpu_temperature_control.py"
                ),
                "pause_temperature_c": 82,
                "resume_temperature_c": 75,
                "single_attempt": True,
                "resume_allowed": False,
                "automatic_retry_allowed": False,
            },
            "final_artifact": {
                "directory": "final_model",
                "serialization": "safetensors",
                "checkpoint_policy": "final_model_only",
                "optimizer_state_saved": False,
                "intermediate_checkpoint_saved": False,
                "strict_loader_required": True,
                "training_and_module_state_fingerprints_separate": True,
            },
            "post_training_structural_replay": {
                "source": (
                    "same_full_D_R_cache_then_fixed_bounded_population"
                ),
                "population_seed": 42,
                "policy": "frozen_v21_bounded_structural_policy",
                "threshold_search_performed": False,
                "performance_evaluation": False,
                "D_V_authorized_only_if_structural_retention_passes": True,
                "generic_population_gate_reported_separately": True,
            },
            "implementation": {
                "files": implementation_files,
                "implementation_fingerprint": stable_fingerprint(
                    implementation_files
                ),
            },
            "evidence_scope": {
                "D_V_accessed": False,
                "D_T_accessed": False,
                "calibration_performed": False,
                "inference_performed": False,
                "performance_evaluation_performed": False,
                "performance_claim_supported": False,
                "full_CURE_authorized": False,
                "cross_backbone_authorized": False,
            },
            **source_closure_fields,
        }
    )
    _write_json(receipts / "config.json", config)
    _write_json(
        receipts / "inputs.json",
        _fingerprinted(
            {
                "schema_version": module.PAET_FORMAL_INPUTS_RECEIPT_SCHEMA,
                "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
                "real_D_R_inputs": real_inputs_payload,
                "real_inputs_build_fingerprint": (
                    real_inputs_fingerprint
                ),
                "full_D_R_scalar_cache_fingerprint": (
                    COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
                ),
                "construction_invocations": 1,
                "bounded_population_constructed_before_training": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        ),
    )
    _write_json(
        receipts / "authorization.json",
        _fingerprinted(
            {
                "schema_version": (
                    module.PAET_FORMAL_AUTHORIZATION_RECEIPT_SCHEMA
                ),
                "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
                "authorization": authorization_payload,
                "authorization_fingerprint": authorization_fingerprint,
                "config_fingerprint": config["receipt_fingerprint"],
                "implementation_fingerprint": stable_fingerprint(
                    implementation_files
                ),
                "formal_training_authorized": True,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        ),
    )
    resource = _fingerprinted(
        {
            "schema_version": module.PAET_FORMAL_RESOURCE_RECEIPT_SCHEMA,
            "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
            "device": "cuda:0",
            "scope": "single_formal_training_invocation",
            "updates": 32_000,
            "baseline_allocated_bytes": 0,
            "baseline_reserved_bytes": 0,
            "peak_allocated_bytes": 10,
            "peak_reserved_bytes": 20,
            "incremental_peak_allocated_bytes": 10,
            "incremental_peak_reserved_bytes": 20,
            "elapsed_ns": 32_000,
            "ns_per_update": 1.0,
            "oom_observed": False,
            "training_invocations": 1,
            "performance_measurement": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
    )
    _write_json(receipts / "training_resource.json", resource)
    with (receipts / "epoch_progress.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in epoch_logs:
            handle.write(
                json.dumps(
                    {
                        "schema_version": (
                            module.PAET_FORMAL_EPOCH_PROGRESS_SCHEMA
                        ),
                        "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
                        "objective": "pmope_joint",
                        "epoch_result": row,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

    members = {
        name: file_sha256(final / name)
        for name in module._FINAL_MEMBERS
    }
    _write_json(
        receipts / "final_artifact.json",
        _fingerprinted(
            {
                "schema_version": (
                    module.PAET_FORMAL_FINAL_ARTIFACT_RECEIPT_SCHEMA
                ),
                "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
                "artifact_repo_path": (
                    "runs/irstd1k_stage_a_seed42/"
                    f"{COVERAGE_STATE_PAET_FORMAL_RUN_ID}/final_model"
                ),
                "artifact_fingerprint": artifact.artifact_fingerprint,
                "artifact_receipt_sha256": artifact.receipt_sha256,
                "authorization_fingerprint": (
                    artifact.authorization_fingerprint
                ),
                "formal_result_fingerprint": (
                    artifact.formal_result_fingerprint
                ),
                "training_model_fingerprint": (
                    artifact.training_model_fingerprint
                ),
                "module_state_fingerprint": (
                    artifact.module_state_fingerprint
                ),
                "member_files": members,
                "strict_loader_verified": True,
                "checkpoint_policy": "final_model_only",
                "optimizer_state_saved": False,
                "intermediate_checkpoint_saved": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        ),
    )
    _write_json(
        receipts / "formal_training.json",
        _fingerprinted(
            {
                "schema_version": (
                    module.PAET_FORMAL_TRAINING_RECEIPT_SCHEMA
                ),
                "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
                "formal_result": artifact.formal_result_payload,
                "formal_training_result_fingerprint": (
                    artifact.formal_result_fingerprint
                ),
                "authorization_fingerprint": (
                    artifact.authorization_fingerprint
                ),
                "training_invocations": 1,
                "completed_updates": 32_000,
                "epoch_callback_rows": 800,
                "resource_measurement_fingerprint": (
                    resource["receipt_fingerprint"]
                ),
                "final_artifact_fingerprint": (
                    artifact.artifact_fingerprint
                ),
                "training_model_fingerprint": (
                    artifact.training_model_fingerprint
                ),
                "module_state_fingerprint": (
                    artifact.module_state_fingerprint
                ),
                "D_V_accessed": False,
                "D_T_accessed": False,
                "performance_evaluation_performed": False,
            }
        ),
    )

    assert (
        diagnostic.checkpoint_fingerprint
        == artifact.module_state_fingerprint
    )
    policy = module.decide_coverage_state_paet_bounded(
        diagnostic,
        run_id=module.COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
    )
    assert policy.bounded_gate_passed
    structural_result = {
        "schema_version": (
            module.COVERAGE_STATE_PAET_FORMAL_STRUCTURAL_SCHEMA
        ),
        "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
        "runtime_splits": ["D_R"],
        "formal_result_fingerprint": artifact.formal_result_fingerprint,
        "formal_authorization_fingerprint": (
            artifact.authorization_fingerprint
        ),
        "final_model_fingerprint": artifact.module_state_fingerprint,
        "bounded_cache_fingerprint": (
            module.COVERAGE_STATE_PAET_V21_BOUNDED_CACHE_FINGERPRINT
        ),
        "bounded_population_fingerprint": (
            bounded_population_fingerprint
        ),
        "source_receipt_fingerprint": (
            module.COVERAGE_STATE_PAET_V21_SOURCE_RECEIPT_FINGERPRINT
        ),
        "input_representation": (
            module.COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
        ),
        "field_threshold_hex": 0.0.hex(),
        "threshold_search_performed": False,
        "diagnostic": diagnostic.canonical_payload(),
        "diagnostic_result_fingerprint": diagnostic.result_fingerprint,
        "frozen_structural_policy": policy.canonical_payload(),
        "frozen_structural_policy_fingerprint": (
            policy.decision_fingerprint
        ),
        "policy_origin_run_id": (
            module.COVERAGE_STATE_PAET_BOUNDED_RUN_ID
        ),
        "policy_reused_without_change": True,
        "bounded400_structural_advancement_passed": True,
        "post_formal_structural_retention_passed": True,
        "generic_population_gate_passed": False,
        "performance_status": (
            module.COVERAGE_STATE_PAET_FORMAL_STRUCTURAL_PERFORMANCE_STATUS
        ),
        "performance_gate_passed": None,
        "evaluation_invocations": 1,
        "training_performed_by_this_layer": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "performance_evaluation_performed": False,
        "performance_claim_supported": False,
    }
    structural_fp = stable_fingerprint(structural_result)
    _write_json(
        receipts / "structural_replay.json",
        _fingerprinted(
            {
                "schema_version": (
                    module.PAET_FORMAL_STRUCTURAL_REPLAY_SCHEMA
                ),
                "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
                "source_full_D_R_scalar_cache_fingerprint": (
                    COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
                ),
                "bounded_population": bounded_population,
                "bounded_population_fingerprint": (
                    bounded_population_fingerprint
                ),
                "structural_result": structural_result,
                "structural_result_fingerprint": structural_fp,
                "training_model_fingerprint": (
                    artifact.training_model_fingerprint
                ),
                "module_state_fingerprint": (
                    artifact.module_state_fingerprint
                ),
                "evaluation_invocations": 1,
                "paet_structural_retention_gate_passed": True,
                "generic_zero_level_population_gate_passed": False,
                "performance_evaluation_performed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        ),
    )
    decision_checks = {
        "formal_training_complete": True,
        "strict_final_artifact_bound": True,
        "structural_replay_invoked_once": True,
        "frozen_paet_structural_retention_gate_passed": True,
        "D_V_and_D_T_not_accessed": True,
    }
    decision = _fingerprinted(
        {
            "schema_version": (
                module.PAET_FORMAL_STRUCTURAL_DECISION_SCHEMA
            ),
            "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
            "status": (
                "PAET_BFA_V21_FORMAL800_STRUCTURAL_PASS_AUTHORIZE_D_V"
            ),
            "formal_training_complete": True,
            "strict_final_artifact_bound": True,
            "paet_structural_retention_gate_passed": True,
            "generic_zero_level_population_gate_passed": False,
            "generic_gate_is_D_V_prerequisite": False,
            "structural_gate_and_generic_gate_are_separate": True,
            "checks": decision_checks,
            "failed_checks": [],
            "D_V_authorized": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_evaluation_performed": False,
            "performance_gate_passed": None,
            "performance_claim_supported": False,
            "final_model_success_established": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "next_action": "RUN_ONE_SEPARATE_STRICT_D_V_REVEAL",
            "bindings": {
                "authorization_fingerprint": (
                    artifact.authorization_fingerprint
                ),
                "formal_training_result_fingerprint": (
                    artifact.formal_result_fingerprint
                ),
                "final_artifact_fingerprint": (
                    artifact.artifact_fingerprint
                ),
                "training_model_fingerprint": (
                    artifact.training_model_fingerprint
                ),
                "module_state_fingerprint": (
                    artifact.module_state_fingerprint
                ),
                "structural_result_fingerprint": structural_fp,
            },
        },
        "decision_fingerprint",
    )
    _write_json(receipts / "decision.json", decision)

    files = dict(
        sorted(
            {
                **{
                    f"final_model/{name}": file_sha256(final / name)
                    for name in module._FINAL_MEMBERS
                },
                **{
                    f"receipts/{name}": file_sha256(receipts / name)
                    for name in module._RECEIPT_MEMBERS
                },
            }.items()
        )
    )
    _write_json(
        root / "COMPLETE.json",
        _fingerprinted(
            {
                "schema_version": module.PAET_FORMAL_RUN_SCHEMA,
                "run_id": COVERAGE_STATE_PAET_FORMAL_RUN_ID,
                "status": "complete",
                "decision": decision["status"],
                "formal_training_complete": True,
                "formal_training_result_fingerprint": (
                    artifact.formal_result_fingerprint
                ),
                "final_artifact_fingerprint": (
                    artifact.artifact_fingerprint
                ),
                "artifact_receipt_sha256": artifact.receipt_sha256,
                "training_model_fingerprint": (
                    artifact.training_model_fingerprint
                ),
                "module_state_fingerprint": (
                    artifact.module_state_fingerprint
                ),
                "structural_result_fingerprint": structural_fp,
                "paet_structural_retention_gate_passed": True,
                "generic_zero_level_population_gate_passed": False,
                "structural_gate_and_generic_gate_are_separate": True,
                "D_V_authorized": True,
                "D_V_accessed": False,
                "D_T_accessed": False,
                "performance_evaluation_performed": False,
                "performance_gate_passed": None,
                "performance_claim_supported": False,
                "artifact_files": files,
                "artifact_file_count": len(files),
                "single_attempt": True,
                "resume_allowed": False,
                "automatic_retry_allowed": False,
                "full_CURE_authorized": False,
                "cross_backbone_authorized": False,
                "attempt_fingerprint": attempt["receipt_fingerprint"],
                "started_fingerprint": started["receipt_fingerprint"],
                **source_closure_fields,
            },
            "complete_fingerprint",
        ),
    )
    return root


def _rehash_complete_inventory(root: Path) -> None:
    complete_path = root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete.pop("complete_fingerprint")
    complete["artifact_files"] = module._regular_inventory(root)
    complete["artifact_file_count"] = len(complete["artifact_files"])
    _write_json(
        complete_path,
        _fingerprinted(complete, "complete_fingerprint"),
    )


def test_persistent_attempt_loader_accepts_only_complete_bound_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fixture(tmp_path, monkeypatch)
    loaded = module.load_coverage_state_paet_formal_attempt()
    assert loaded.post_formal_structural_retention_passed is True
    assert loaded.generic_zero_level_population_gate_passed is False


@pytest.mark.parametrize("target", ("COMPLETE.json", "receipts/structural_replay.json", "final_model/model.safetensors"))
def test_persistent_attempt_loader_rejects_tampered_or_incomplete_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    root = _fixture(tmp_path, monkeypatch)
    (root / target).write_text("{}", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        module.load_coverage_state_paet_formal_attempt()


@pytest.mark.parametrize("marker", (".incomplete", "FAILURE.json"))
def test_persistent_attempt_loader_rejects_failure_or_incomplete_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: str) -> None:
    root = _fixture(tmp_path, monkeypatch)
    (root / marker).write_bytes(b"")
    with pytest.raises(ValueError, match="inventory"):
        module.load_coverage_state_paet_formal_attempt()


def test_persistent_attempt_loader_rejects_structural_fail_even_when_rehashed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _fixture(tmp_path, monkeypatch)
    replay_path = root / "receipts/structural_replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay.pop("receipt_fingerprint")
    result = replay["structural_result"]
    assert isinstance(result, dict)
    result["post_formal_structural_retention_passed"] = False
    replay["structural_result_fingerprint"] = stable_fingerprint(result)
    replay["paet_structural_retention_gate_passed"] = False
    _write_json(replay_path, _fingerprinted(replay))
    _rehash_complete_inventory(root)
    with pytest.raises(
        ValueError,
        match="frozen passing replay|structural",
    ):
        module.load_coverage_state_paet_formal_attempt()


def test_persistent_attempt_loader_rejects_nested_policy_change_even_when_chain_is_rehashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _fixture(tmp_path, monkeypatch)
    replay_path = root / "receipts/structural_replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay.pop("receipt_fingerprint")
    result = replay["structural_result"]
    assert isinstance(result, dict)
    policy = result["frozen_structural_policy"]
    assert isinstance(policy, dict)
    policy["bounded_gate_passed"] = False
    result["frozen_structural_policy_fingerprint"] = stable_fingerprint(
        policy
    )
    structural_fingerprint = stable_fingerprint(result)
    replay["structural_result_fingerprint"] = structural_fingerprint
    _write_json(replay_path, _fingerprinted(replay))

    decision_path = root / "receipts/decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision.pop("decision_fingerprint")
    bindings = decision["bindings"]
    assert isinstance(bindings, dict)
    bindings["structural_result_fingerprint"] = structural_fingerprint
    _write_json(
        decision_path,
        _fingerprinted(decision, "decision_fingerprint"),
    )

    complete_path = root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete.pop("complete_fingerprint")
    complete["structural_result_fingerprint"] = structural_fingerprint
    complete["artifact_files"] = module._regular_inventory(root)
    complete["artifact_file_count"] = len(complete["artifact_files"])
    _write_json(
        complete_path,
        _fingerprinted(complete, "complete_fingerprint"),
    )

    with pytest.raises(
        ValueError,
        match="frozen passing replay|structural",
    ):
        module.load_coverage_state_paet_formal_attempt()


@pytest.mark.parametrize(
    "target",
    (
        "receipts/inputs.json",
        "receipts/authorization.json",
        "receipts/epoch_progress.jsonl",
        "receipts/training_resource.json",
    ),
)
def test_persistent_attempt_loader_rejects_broken_required_receipt_after_complete_inventory_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    root = _fixture(tmp_path, monkeypatch)
    (root / target).write_bytes(b"not-json-at-all\n")
    _rehash_complete_inventory(root)
    with pytest.raises(ValueError):
        module.load_coverage_state_paet_formal_attempt()
