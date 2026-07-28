from __future__ import annotations

from copy import deepcopy

import pytest

from cure_lite.cache.schema import stable_fingerprint
from cure_lite_v22.dataset_free import PACRE_DATASET_FREE_CHECK_NAMES
from cure_lite_v23.dataset_free import (
    PACRE_VC_DATASET_FREE_CHECK_NAMES,
    PACRE_VC_DATASET_FREE_NEW_CHECK_NAMES,
    run_pacre_vc_dataset_free_gate,
)
from cure_lite_v23.numeric_stress import (
    PACRE_VC_FORMAL_STRESS_RUN_SCHEMA,
    PACRE_VC_FORMAL_STRESS_SCHEMA,
    PACRE_VC_FORMAL_STRESS_SEEDS,
    run_pacre_vc_scalar_counterexample_receipt,
)
from cure_lite_v23.parity import (
    PACRE_VC_PARITY_RUN_SCHEMA,
    PACRE_VC_PARITY_SCHEMA,
)


def _fingerprinted(
    body: dict[str, object],
    field: str,
) -> dict[str, object]:
    return {**body, field: stable_fingerprint(body)}


def _boundary() -> dict[str, object]:
    return {
        "generated_only": True,
        "dataset_accessed": False,
        "cache_accessed": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }


def _environment(device: str) -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": "test-runtime-environment",
            "logical_device": device,
        },
        "environment_fingerprint",
    )


def _source() -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": "test-source-closure",
            "files": [{"repo_path": "frozen.py", "sha256": "a" * 64}],
        },
        "closure_fingerprint",
    )


def _manifest(seed: int) -> dict[str, object]:
    return _fingerprinted(
        {
            "generator_device": "cpu",
            "seed": seed,
            "tensors": {
                "feature": {"raw_fingerprint": f"{seed:064x}"},
                "occupancy": {"raw_fingerprint": f"{seed + 1:064x}"},
            },
        },
        "manifest_fingerprint",
    )


def _parity_receipt() -> dict[str, object]:
    runs = []
    for device in ("cpu", "cuda:0"):
        for seed in PACRE_VC_FORMAL_STRESS_SEEDS:
            steps = [
                {
                    "step": index,
                    "loss": {"passed": True},
                    "model_state": {"passed": True},
                    "optimizer_state": {
                        name: {"passed": True}
                        for name in ("step", "exp_avg", "exp_avg_sq")
                    },
                    "passed": True,
                }
                for index in range(1, 4)
            ]
            body = {
                "schema_version": PACRE_VC_PARITY_RUN_SCHEMA,
                "device": device,
                "seed": seed,
                "state_dict_parity": {"passed": True},
                "all_fields_raw_parity": {"passed": True},
                "probe_gradient_parity": {"passed": True},
                "probe_models_preserved": True,
                "optimizer_parity": {
                    "fresh_optimizer_state_empty": True,
                    "initial_model_state": {"passed": True},
                    "steps": steps,
                    "passed": True,
                },
                "optimizer_steps_per_model": 3,
                "global_cpu_rng_preserved": True,
                "selected_device_rng_preserved": True,
                "deterministic_execution": {
                    "restored_exactly": True,
                },
                "gate_passed": True,
                **{
                    key: value
                    for key, value in _boundary().items()
                    if key != "generated_only"
                },
            }
            runs.append(_fingerprinted(body, "receipt_fingerprint"))
    return _fingerprinted(
        {
            "schema_version": PACRE_VC_PARITY_SCHEMA,
            "required_devices": ["cpu", "cuda:0"],
            "required_seeds": list(PACRE_VC_FORMAL_STRESS_SEEDS),
            "runs": runs,
            "expected_run_count": len(runs),
            "observed_run_count": len(runs),
            "all_required_runs_present": True,
            "gate_passed": True,
            **_boundary(),
        },
        "receipt_fingerprint",
    )


def _stress_receipt(
    device: str,
    source: dict[str, object],
) -> dict[str, object]:
    environment = _environment(device)
    runs = []
    for seed in PACRE_VC_FORMAL_STRESS_SEEDS:
        legacy_rows = [
            {
                "name": f"legacy_{index}",
                "gate_eligible": False,
                "decision_weight": 0,
                "passed_under_v22_check": index not in (2, 3),
                "failed_element_count": 7 if index in (2, 3) else 0,
            }
            for index in range(6)
        ]
        body = {
            "schema_version": PACRE_VC_FORMAL_STRESS_RUN_SCHEMA,
            "device": device,
            "seed": seed,
            "input_manifest": _manifest(seed),
            "all_fields_raw_parity": {"passed": True},
            "algebra_verification": {
                "passed": True,
                "phase_semantics": {
                    "reconstruction": {"passed": True},
                    "centering": {"passed": True},
                },
            },
            "legacy_subtraction_diagnostics": {
                "gate_eligible": False,
                "decision_weight": 0,
            },
            "legacy_six_check_replay": {
                "complete": True,
                "all_diagnostic_only": True,
                "legacy_outcomes_gate_eligible": False,
                "rows": legacy_rows,
            },
            "fp64_oracle": {"integrity": {"passed": True}},
            "signal_swallow_ledger_integrity": {
                "passed": True,
                "all_diagnostic_only": True,
                "decision_uses_swallow_counts": False,
            },
            "fixed_readout_nonzero_observation": {
                "readout_exact_zero": False,
                "passed": True,
            },
            "runtime_environment_fingerprint": environment[
                "environment_fingerprint"
            ],
            "source_closure_fingerprint": source[
                "closure_fingerprint"
            ],
            "gate_passed": True,
            "optimizer_constructed": False,
            "training_performed": False,
            **_boundary(),
        }
        runs.append(_fingerprinted(body, "run_fingerprint"))
    aggregate_manifest = _fingerprinted(
        {
            "generator_device": "cpu",
            "seeds": list(PACRE_VC_FORMAL_STRESS_SEEDS),
            "manifests": [
                _manifest(seed) for seed in PACRE_VC_FORMAL_STRESS_SEEDS
            ],
        },
        "manifest_fingerprint",
    )
    body = {
        "schema_version": PACRE_VC_FORMAL_STRESS_SCHEMA,
        "device": device,
        "formal_shape": {
            "batch_size": 1,
            "feature_channels": 64,
            "feature_stride": 4,
            "hidden_width": 32,
            "feature_grid": [64, 64],
            "occupancy_grid": [256, 256],
            "dtype": "torch.float32",
        },
        "canonical_cpu_input_manifest": aggregate_manifest,
        "runtime_environment": environment,
        "runtime_environment_verified_pre_input_generation": True,
        "runtime_environment_verified_post_execution": True,
        "source_closure": source,
        "source_closure_verified_pre_input_generation": True,
        "source_closure_verified_post_execution": True,
        "runs": runs,
        "gate_passed": True,
        "optimizer_constructed": False,
        "training_performed": False,
        **_boundary(),
    }
    return _fingerprinted(body, "receipt_fingerprint")


def _v22_receipt() -> dict[str, object]:
    body = {
        "checks": {
            name: True for name in PACRE_DATASET_FREE_CHECK_NAMES
        },
        "gate_passed": True,
    }
    return _fingerprinted(body, "receipt_fingerprint")


def _evidence() -> dict[str, object]:
    source = _source()
    cpu = _stress_receipt("cpu", source)
    selected = _stress_receipt("cuda:0", source)
    return {
        "parity_receipt": _parity_receipt(),
        "cpu_stress_receipt": cpu,
        "selected_device_stress_receipt": selected,
        "counterexample_receipt": (
            run_pacre_vc_scalar_counterexample_receipt()
        ),
        "v22_dataset_free_receipt": _v22_receipt(),
        "runtime_environment_receipts": {
            "cpu": cpu["runtime_environment"],
            "cuda:0": selected["runtime_environment"],
        },
        "source_closure_receipt": source,
    }


def test_dataset_free_aggregates_v22_checks_and_new_14_through_22() -> None:
    receipt = run_pacre_vc_dataset_free_gate(**_evidence())

    assert tuple(receipt["checks"]) == PACRE_VC_DATASET_FREE_CHECK_NAMES
    assert tuple(receipt["checks"])[13:] == (
        PACRE_VC_DATASET_FREE_NEW_CHECK_NAMES
    )
    assert len(receipt["checks"]) == 22
    assert all(receipt["checks"].values())
    assert receipt["gate_passed"] is True
    assert receipt["candidate"] == "PACRE-VC-v23"
    assert receipt["parameter_count"] == 64_064
    assert receipt["dataset_accessed"] is False
    assert receipt["cache_accessed"] is False
    assert receipt["D_R_accessed"] is False
    assert receipt["D_V_accessed"] is False
    assert receipt["D_T_accessed"] is False
    assert receipt["optimizer_constructed"] is False
    assert receipt["training_performed"] is False
    body = dict(receipt)
    fingerprint = body.pop("receipt_fingerprint")
    assert fingerprint == stable_fingerprint(body)


def test_legacy_failures_are_recorded_but_have_zero_decision_weight() -> None:
    receipt = run_pacre_vc_dataset_free_gate(**_evidence())

    assert (
        receipt["checks"][
            "20_legacy_subtraction_is_diagnostic_only"
        ]
        is True
    )
    assert receipt["gate_passed"] is True


def test_dataset_free_rejects_any_tampered_nested_evidence() -> None:
    evidence = _evidence()
    tampered = deepcopy(evidence["cpu_stress_receipt"])
    tampered["runs"][0]["gate_passed"] = False
    evidence["cpu_stress_receipt"] = tampered

    with pytest.raises(ValueError, match="receipt_fingerprint"):
        run_pacre_vc_dataset_free_gate(**evidence)
