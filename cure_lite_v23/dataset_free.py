"""Aggregate generated-only PACRE-VC evidence without data access.

The first thirteen checks are copied verbatim from the sealed v22
dataset-free receipt.  Checks 14--22 consume already-generated parity,
counterexample, formal-stress, and runtime evidence.  This aggregator never
constructs an optimizer, performs training, or reads D_R, D_V, or D_T.
"""

from __future__ import annotations

from typing import Final, Mapping, Sequence

from cure_lite.cache.schema import stable_fingerprint
from cure_lite_v22.dataset_free import (
    PACRE_DATASET_FREE_CHECK_NAMES,
    PACRE_FORMAL_FEATURE_CHANNELS,
    PACRE_FORMAL_FEATURE_STRIDE,
    PACRE_FORMAL_PARAMETER_COUNT,
    PACRE_FORMAL_WIDTH,
    run_pacre_dataset_free_gate,
)

from .numeric_stress import (
    PACRE_VC_FORMAL_STRESS_RUN_SCHEMA,
    PACRE_VC_FORMAL_STRESS_SCHEMA,
    PACRE_VC_FORMAL_STRESS_SEEDS,
    PACRE_VC_SCALAR_COUNTEREXAMPLE_SCHEMA,
    run_pacre_vc_scalar_counterexample_receipt,
)
from .parity import (
    PACRE_VC_PARITY_ADAM_STEPS,
    PACRE_VC_PARITY_RUN_SCHEMA,
    PACRE_VC_PARITY_SCHEMA,
)


PACRE_VC_DATASET_FREE_SCHEMA: Final = (
    "cure-lite-v23-pacre-vc-dataset-free-receipt-v1"
)
PACRE_VC_DATASET_FREE_NEW_CHECK_NAMES: Final = (
    "14_v22_v23_forward_field_gradient_parity",
    "15_scalar_cancellation_counterexamples_reproduced",
    "16_formal_shape_cpu_algebra_stress",
    "17_formal_shape_selected_device_algebra_stress",
    "18_phase_reconstruction_bound_valid",
    "19_phase_centering_bound_valid",
    "20_legacy_subtraction_is_diagnostic_only",
    "21_fp64_oracle_and_swallow_ledger_complete",
    "22_runtime_environment_frozen",
)
PACRE_VC_DATASET_FREE_CHECK_NAMES: Final = (
    *PACRE_DATASET_FREE_CHECK_NAMES,
    *PACRE_VC_DATASET_FREE_NEW_CHECK_NAMES,
)
PACRE_VC_REQUIRED_DEVICES: Final = ("cpu", "cuda:0")


def _verify_self_fingerprint(
    payload: Mapping[str, object],
    *,
    field: str,
) -> str:
    if not isinstance(payload, Mapping):
        raise TypeError("evidence receipt must be a mapping")
    body = dict(payload)
    observed = body.pop(field, None)
    if (
        not isinstance(observed, str)
        or len(observed) != 64
        or observed != stable_fingerprint(body)
    ):
        raise ValueError(f"invalid {field}")
    return observed


def _boundary_closed(payload: Mapping[str, object]) -> bool:
    return (
        payload.get("generated_only") is True
        and payload.get("dataset_accessed") is False
        and payload.get("cache_accessed") is False
        and payload.get("D_R_accessed") is False
        and payload.get("D_V_accessed") is False
        and payload.get("D_T_accessed") is False
    )


def _verify_environment_lock(payload: Mapping[str, object]) -> str:
    return _verify_self_fingerprint(
        payload,
        field="environment_fingerprint",
    )


def _verify_source_lock(payload: Mapping[str, object]) -> str:
    return _verify_self_fingerprint(
        payload,
        field="closure_fingerprint",
    )


def _verify_manifest(payload: Mapping[str, object]) -> str:
    return _verify_self_fingerprint(
        payload,
        field="manifest_fingerprint",
    )


def _verify_parity_receipt(
    payload: Mapping[str, object],
) -> tuple[str, bool]:
    fingerprint = _verify_self_fingerprint(
        payload,
        field="receipt_fingerprint",
    )
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("parity runs are missing")
    required = {
        (device, seed)
        for device in PACRE_VC_REQUIRED_DEVICES
        for seed in PACRE_VC_FORMAL_STRESS_SEEDS
    }
    observed: set[tuple[str, int]] = set()
    runs_valid = True
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError("parity run must be a mapping")
        _verify_self_fingerprint(run, field="receipt_fingerprint")
        device = run.get("device")
        seed = run.get("seed")
        if not isinstance(device, str) or not isinstance(seed, int):
            raise ValueError("parity run identity is malformed")
        observed.add((device, seed))
        state = run.get("state_dict_parity")
        fields = run.get("all_fields_raw_parity")
        gradient = run.get("probe_gradient_parity")
        optimizer = run.get("optimizer_parity")
        determinism = run.get("deterministic_execution")
        optimizer_steps = (
            optimizer.get("steps")
            if isinstance(optimizer, Mapping)
            else None
        )
        optimizer_valid = (
            isinstance(optimizer, Mapping)
            and optimizer.get("fresh_optimizer_state_empty") is True
            and isinstance(
                optimizer.get("initial_model_state"),
                Mapping,
            )
            and optimizer["initial_model_state"].get("passed") is True
            and optimizer.get("passed") is True
            and isinstance(optimizer_steps, list)
            and len(optimizer_steps) == PACRE_VC_PARITY_ADAM_STEPS
            and all(
                isinstance(step, Mapping)
                and step.get("step") == index
                and step.get("passed") is True
                and isinstance(step.get("loss"), Mapping)
                and step["loss"].get("passed") is True
                and isinstance(step.get("model_state"), Mapping)
                and step["model_state"].get("passed") is True
                and isinstance(step.get("optimizer_state"), Mapping)
                and all(
                    isinstance(step["optimizer_state"].get(name), Mapping)
                    and step["optimizer_state"][name].get("passed")
                    is True
                    for name in ("step", "exp_avg", "exp_avg_sq")
                )
                for index, step in enumerate(optimizer_steps, start=1)
            )
        )
        runs_valid &= (
            run.get("schema_version") == PACRE_VC_PARITY_RUN_SCHEMA
            and isinstance(state, Mapping)
            and state.get("passed") is True
            and isinstance(fields, Mapping)
            and fields.get("passed") is True
            and isinstance(gradient, Mapping)
            and gradient.get("passed") is True
            and run.get("probe_models_preserved") is True
            and optimizer_valid
            and run.get("optimizer_steps_per_model")
            == PACRE_VC_PARITY_ADAM_STEPS
            and run.get("global_cpu_rng_preserved") is True
            and run.get("selected_device_rng_preserved") is True
            and isinstance(determinism, Mapping)
            and determinism.get("restored_exactly") is True
            and run.get("gate_passed") is True
            and run.get("dataset_accessed") is False
            and run.get("cache_accessed") is False
            and run.get("D_R_accessed") is False
            and run.get("D_V_accessed") is False
            and run.get("D_T_accessed") is False
        )
    valid = (
        payload.get("schema_version") == PACRE_VC_PARITY_SCHEMA
        and observed == required
        and len(runs) == len(required)
        and payload.get("expected_run_count") == len(required)
        and payload.get("observed_run_count") == len(required)
        and payload.get("all_required_runs_present") is True
        and runs_valid
        and payload.get("gate_passed") is True
        and _boundary_closed(payload)
    )
    return fingerprint, valid


def _verify_counterexample_receipt(
    payload: Mapping[str, object],
) -> tuple[str, bool]:
    fingerprint = _verify_self_fingerprint(
        payload,
        field="receipt_fingerprint",
    )
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("counterexample cases are missing")
    valid_cases = (
        len(cases) == 2
        and all(
            isinstance(case, Mapping)
            and case.get("legacy_failure_reproduced") is True
            and case.get("legacy_v22_allclose") is False
            and case.get("same_operand_exact_forward_replay") is True
            and isinstance(case.get("phase_semantics"), Mapping)
            and case["phase_semantics"].get("passed") is True
            for case in cases
        )
    )
    valid = (
        payload.get("schema_version")
        == PACRE_VC_SCALAR_COUNTEREXAMPLE_SCHEMA
        and payload.get("gate_passed") is True
        and payload.get("legacy_formula_gate_eligible") is False
        and payload.get("legacy_formula_decision_weight") == 0
        and valid_cases
        and _boundary_closed(payload)
        and payload.get("optimizer_constructed") is False
        and payload.get("training_performed") is False
    )
    return fingerprint, valid


def _verify_stress_run(
    run: Mapping[str, object],
    *,
    device: str,
) -> dict[str, bool]:
    _verify_self_fingerprint(run, field="run_fingerprint")
    if (
        run.get("schema_version") != PACRE_VC_FORMAL_STRESS_RUN_SCHEMA
        or run.get("device") != device
        or not isinstance(run.get("seed"), int)
    ):
        raise ValueError("formal stress run identity is malformed")
    manifest = run.get("input_manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("formal stress input manifest is missing")
    _verify_manifest(manifest)

    algebra = run.get("algebra_verification")
    if not isinstance(algebra, Mapping):
        raise ValueError("algebra verification is missing")
    phase = algebra.get("phase_semantics")
    if not isinstance(phase, Mapping):
        raise ValueError("phase semantics are missing")
    reconstruction = phase.get("reconstruction")
    centering = phase.get("centering")
    if not isinstance(reconstruction, Mapping) or not isinstance(
        centering,
        Mapping,
    ):
        raise ValueError("phase semantic subchecks are missing")

    legacy = run.get("legacy_subtraction_diagnostics")
    legacy_six = run.get("legacy_six_check_replay")
    if not isinstance(legacy, Mapping) or not isinstance(
        legacy_six,
        Mapping,
    ):
        raise ValueError("legacy diagnostic ledger is missing")
    legacy_rows = legacy_six.get("rows")
    legacy_diagnostic_only = (
        legacy.get("gate_eligible") is False
        and legacy.get("decision_weight") == 0
        and legacy_six.get("complete") is True
        and legacy_six.get("all_diagnostic_only") is True
        and legacy_six.get("legacy_outcomes_gate_eligible") is False
        and isinstance(legacy_rows, list)
        and len(legacy_rows) == 6
        and all(
            isinstance(row, Mapping)
            and row.get("gate_eligible") is False
            and row.get("decision_weight") == 0
            for row in legacy_rows
        )
    )

    oracle = run.get("fp64_oracle")
    swallow = run.get("signal_swallow_ledger_integrity")
    fixed = run.get("fixed_readout_nonzero_observation")
    if (
        not isinstance(oracle, Mapping)
        or not isinstance(swallow, Mapping)
        or not isinstance(fixed, Mapping)
    ):
        raise ValueError("FP64 evidence is incomplete")
    integrity = oracle.get("integrity")
    oracle_complete = (
        isinstance(integrity, Mapping)
        and integrity.get("passed") is True
        and swallow.get("passed") is True
        and swallow.get("all_diagnostic_only") is True
        and swallow.get("decision_uses_swallow_counts") is False
        and fixed.get("readout_exact_zero") is False
        and fixed.get("passed") is True
    )
    fields_parity = run.get("all_fields_raw_parity")
    run_gate = (
        isinstance(fields_parity, Mapping)
        and fields_parity.get("passed") is True
        and algebra.get("passed") is True
        and run.get("gate_passed") is True
        and _boundary_closed(run)
        and run.get("optimizer_constructed") is False
        and run.get("training_performed") is False
    )
    return {
        "gate": run_gate,
        "reconstruction": reconstruction.get("passed") is True,
        "centering": centering.get("passed") is True,
        "legacy_diagnostic_only": legacy_diagnostic_only,
        "oracle_complete": oracle_complete,
    }


def _verify_stress_receipt(
    payload: Mapping[str, object],
    *,
    device: str,
) -> tuple[str, dict[str, bool], str, str, str]:
    fingerprint = _verify_self_fingerprint(
        payload,
        field="receipt_fingerprint",
    )
    if (
        payload.get("schema_version") != PACRE_VC_FORMAL_STRESS_SCHEMA
        or payload.get("device") != device
    ):
        raise ValueError("formal stress receipt identity is malformed")
    environment = payload.get("runtime_environment")
    source = payload.get("source_closure")
    aggregate_manifest = payload.get("canonical_cpu_input_manifest")
    if (
        not isinstance(environment, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(aggregate_manifest, Mapping)
    ):
        raise ValueError("formal stress bindings are incomplete")
    environment_fingerprint = _verify_environment_lock(environment)
    source_fingerprint = _verify_source_lock(source)
    manifest_fingerprint = _verify_manifest(aggregate_manifest)

    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("formal stress runs are missing")
    observed_seeds: set[int] = set()
    run_manifests: dict[int, str] = {}
    outcomes: list[dict[str, bool]] = []
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError("formal stress run must be a mapping")
        outcomes.append(_verify_stress_run(run, device=device))
        seed = run.get("seed")
        assert isinstance(seed, int)
        observed_seeds.add(seed)
        run_manifest = run["input_manifest"]
        assert isinstance(run_manifest, Mapping)
        run_manifests[seed] = _verify_manifest(run_manifest)
        if (
            run.get("runtime_environment_fingerprint")
            != environment_fingerprint
            or run.get("source_closure_fingerprint")
            != source_fingerprint
        ):
            raise ValueError("formal stress run binding differs")

    expected_seeds = set(PACRE_VC_FORMAL_STRESS_SEEDS)
    aggregate_rows = aggregate_manifest.get("manifests")
    if not isinstance(aggregate_rows, list):
        raise ValueError("aggregate input manifests are missing")
    aggregate_by_seed: dict[int, str] = {}
    for manifest in aggregate_rows:
        if not isinstance(manifest, Mapping):
            raise ValueError("aggregate input manifest is malformed")
        seed = manifest.get("seed")
        if not isinstance(seed, int):
            raise ValueError("aggregate input seed is malformed")
        aggregate_by_seed[seed] = _verify_manifest(manifest)
    formal_shape = payload.get("formal_shape")
    formal_shape_valid = (
        isinstance(formal_shape, Mapping)
        and formal_shape.get("batch_size") == 1
        and formal_shape.get("feature_channels") == 64
        and formal_shape.get("feature_stride") == 4
        and formal_shape.get("hidden_width") == 32
        and formal_shape.get("feature_grid") == [64, 64]
        and formal_shape.get("occupancy_grid") == [256, 256]
        and formal_shape.get("dtype") == "torch.float32"
    )
    matrix_complete = (
        observed_seeds == expected_seeds
        and len(runs) == len(expected_seeds)
        and aggregate_manifest.get("generator_device") == "cpu"
        and aggregate_manifest.get("seeds")
        == list(PACRE_VC_FORMAL_STRESS_SEEDS)
        and aggregate_by_seed == run_manifests
        and formal_shape_valid
    )
    summary = {
        "gate": (
            matrix_complete
            and payload.get("gate_passed") is True
            and all(row["gate"] for row in outcomes)
            and _boundary_closed(payload)
            and payload.get("optimizer_constructed") is False
            and payload.get("training_performed") is False
        ),
        "reconstruction": (
            matrix_complete
            and all(row["reconstruction"] for row in outcomes)
        ),
        "centering": (
            matrix_complete
            and all(row["centering"] for row in outcomes)
        ),
        "legacy_diagnostic_only": (
            matrix_complete
            and all(
                row["legacy_diagnostic_only"] for row in outcomes
            )
        ),
        "oracle_complete": (
            matrix_complete
            and all(row["oracle_complete"] for row in outcomes)
        ),
        "runtime_frozen": (
            payload.get(
                "runtime_environment_verified_pre_input_generation"
            )
            is True
            and payload.get(
                "source_closure_verified_pre_input_generation"
            )
            is True
            and payload.get(
                "runtime_environment_verified_post_execution"
            )
            is True
            and payload.get(
                "source_closure_verified_post_execution"
            )
            is True
        ),
    }
    return (
        fingerprint,
        summary,
        environment_fingerprint,
        source_fingerprint,
        manifest_fingerprint,
    )


def _verify_v22_receipt(
    payload: Mapping[str, object],
) -> tuple[str, dict[str, bool]]:
    fingerprint = _verify_self_fingerprint(
        payload,
        field="receipt_fingerprint",
    )
    checks = payload.get("checks")
    if (
        not isinstance(checks, Mapping)
        or tuple(checks) != PACRE_DATASET_FREE_CHECK_NAMES
        or any(type(value) is not bool for value in checks.values())
    ):
        raise ValueError("v22 dataset-free checks differ from the contract")
    return fingerprint, dict(checks)


def _verify_optional_runtime_receipts(
    payloads: Mapping[str, Mapping[str, object]] | None,
    *,
    expected: Mapping[str, str],
) -> bool:
    if payloads is None:
        return True
    if not isinstance(payloads, Mapping) or set(payloads) != set(expected):
        raise ValueError("runtime receipt device matrix differs")
    return all(
        _verify_environment_lock(payloads[device])
        == expected[device]
        for device in expected
    )


def run_pacre_vc_dataset_free_gate(
    *,
    parity_receipt: Mapping[str, object],
    cpu_stress_receipt: Mapping[str, object],
    selected_device_stress_receipt: Mapping[str, object],
    counterexample_receipt: Mapping[str, object] | None = None,
    v22_dataset_free_receipt: Mapping[str, object] | None = None,
    runtime_environment_receipts: (
        Mapping[str, Mapping[str, object]] | None
    ) = None,
    source_closure_receipt: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Aggregate frozen generated evidence into checks 01--22."""

    v22_receipt = (
        run_pacre_dataset_free_gate()
        if v22_dataset_free_receipt is None
        else dict(v22_dataset_free_receipt)
    )
    counter_receipt = (
        run_pacre_vc_scalar_counterexample_receipt()
        if counterexample_receipt is None
        else dict(counterexample_receipt)
    )
    v22_fingerprint, v22_checks = _verify_v22_receipt(v22_receipt)
    parity_fingerprint, parity_valid = _verify_parity_receipt(
        parity_receipt
    )
    counter_fingerprint, counter_valid = (
        _verify_counterexample_receipt(counter_receipt)
    )
    (
        cpu_fingerprint,
        cpu,
        cpu_environment,
        cpu_source,
        cpu_manifest,
    ) = _verify_stress_receipt(cpu_stress_receipt, device="cpu")
    (
        device_fingerprint,
        selected,
        selected_environment,
        selected_source,
        selected_manifest,
    ) = _verify_stress_receipt(
        selected_device_stress_receipt,
        device="cuda:0",
    )
    if cpu_source != selected_source:
        raise ValueError("CPU and selected-device source closures differ")
    if cpu_manifest != selected_manifest:
        raise ValueError(
            "CPU and selected-device canonical inputs differ"
        )
    if source_closure_receipt is not None:
        supplied_source = _verify_source_lock(source_closure_receipt)
        if supplied_source != cpu_source:
            raise ValueError("supplied source closure differs from stress")
    runtime_receipts_valid = _verify_optional_runtime_receipts(
        runtime_environment_receipts,
        expected={
            "cpu": cpu_environment,
            "cuda:0": selected_environment,
        },
    )

    new_checks = {
        "14_v22_v23_forward_field_gradient_parity": parity_valid,
        "15_scalar_cancellation_counterexamples_reproduced": (
            counter_valid
        ),
        "16_formal_shape_cpu_algebra_stress": cpu["gate"],
        "17_formal_shape_selected_device_algebra_stress": (
            selected["gate"]
        ),
        "18_phase_reconstruction_bound_valid": (
            cpu["reconstruction"] and selected["reconstruction"]
        ),
        "19_phase_centering_bound_valid": (
            cpu["centering"] and selected["centering"]
        ),
        "20_legacy_subtraction_is_diagnostic_only": (
            cpu["legacy_diagnostic_only"]
            and selected["legacy_diagnostic_only"]
        ),
        "21_fp64_oracle_and_swallow_ledger_complete": (
            cpu["oracle_complete"] and selected["oracle_complete"]
        ),
        "22_runtime_environment_frozen": (
            cpu["runtime_frozen"]
            and selected["runtime_frozen"]
            and runtime_receipts_valid
        ),
    }
    checks = {**v22_checks, **new_checks}
    if tuple(checks) != PACRE_VC_DATASET_FREE_CHECK_NAMES:
        raise AssertionError("PACRE-VC dataset-free check order changed")
    gate_passed = all(value is True for value in checks.values())
    body: dict[str, object] = {
        "schema_version": PACRE_VC_DATASET_FREE_SCHEMA,
        "candidate": "PACRE-VC-v23",
        "role": "post_v22_D_R_failure_adaptive_verifier_correction",
        "parameter_count": PACRE_FORMAL_PARAMETER_COUNT,
        "checks": checks,
        "evidence_bindings": {
            "v22_dataset_free_receipt_fingerprint": v22_fingerprint,
            "parity_receipt_fingerprint": parity_fingerprint,
            "counterexample_receipt_fingerprint": counter_fingerprint,
            "cpu_stress_receipt_fingerprint": cpu_fingerprint,
            "selected_device_stress_receipt_fingerprint": (
                device_fingerprint
            ),
            "cpu_runtime_environment_fingerprint": cpu_environment,
            "selected_runtime_environment_fingerprint": (
                selected_environment
            ),
            "source_closure_fingerprint": cpu_source,
            "canonical_cpu_input_manifest_fingerprint": cpu_manifest,
        },
        "required_devices": list(PACRE_VC_REQUIRED_DEVICES),
        "required_seeds": list(PACRE_VC_FORMAL_STRESS_SEEDS),
        "gate_passed": gate_passed,
        "next_action": (
            "AUTHORIZE_D_R_ONLY_GATE_IMPLEMENTATION"
            if gate_passed
            else "STOP_AND_REVISE_PACRE_VC_EVIDENCE"
        ),
        "generated_only": True,
        "dataset_accessed": False,
        "cache_accessed": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "optimizer_constructed": False,
        "training_performed": False,
        "threshold_search_performed": False,
    }
    return {**body, "receipt_fingerprint": stable_fingerprint(body)}


__all__ = [
    "PACRE_VC_DATASET_FREE_CHECK_NAMES",
    "PACRE_VC_DATASET_FREE_NEW_CHECK_NAMES",
    "PACRE_VC_DATASET_FREE_SCHEMA",
    "PACRE_VC_REQUIRED_DEVICES",
    "PACRE_FORMAL_FEATURE_CHANNELS",
    "PACRE_FORMAL_FEATURE_STRIDE",
    "PACRE_FORMAL_PARAMETER_COUNT",
    "PACRE_FORMAL_WIDTH",
    "run_pacre_vc_dataset_free_gate",
]
