"""Read-only real-``D_R`` structural gate for PACRE-VC v23.

The model forward is the inherited PACRE-v22 forward.  This gate replaces
only the numerically invalid subtractive algebra assertion with:

* same-device exact replay of the operations actually stored by the forward;
* FTZ-safe analytic phase reconstruction and centering bounds;
* a complete, attribution-only replay of the six legacy v22 subchecks; and
* integrity-gated FP64 and signal-swallow diagnostics.

The gate accepts an already-built seed-42 bounded population.  It never
constructs an optimizer, updates a parameter, or accesses ``D_V``/``D_T``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path
from typing import Final, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from cure_lite.cache.schema import (
    canonical_json,
    file_sha256,
    stable_fingerprint,
)
from cure_lite.experiment.coverage_state_bfa_dr_gate import (
    _direction_probe,
    _state_specs,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_ROLE_COUNT,
    COVERAGE_STATE_BOUNDED_SEED,
    CoverageStateBoundedPopulation,
)
from cure_lite.experiment.coverage_state_paet_dr_gate import (
    _cuda_rng_devices,
    _deterministic_execution_scope,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    CoverageStateRealDRInputs,
)
from cure_lite.experiment.coverage_state_training import (
    coverage_state_model_contract_payload,
    coverage_state_model_fingerprint,
)
from cure_lite.paired_types import tensor_content_fingerprint
from cure_lite_v22.dr_gate import (
    _gradient_probe as _v22_gradient_probe,
    _representation_probe as _v22_representation_probe,
)
from cure_lite_v22.pacre import (
    CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet,
    CoverageStatePACREConfig,
    CoverageStatePACREFields,
)

from .algebra_verifier import (
    PACRE_VC_ALGEBRA_POLICY,
    verify_pacre_v22_forward_fields,
)
from .dataset_free import (
    PACRE_FORMAL_FEATURE_CHANNELS,
    PACRE_FORMAL_FEATURE_STRIDE,
    PACRE_FORMAL_PARAMETER_COUNT,
    PACRE_FORMAL_WIDTH,
)
from .environment import verify_runtime_environment
from .factory import (
    PACRE_VC_PARAMETER_NAMES,
    PACRE_VC_TRAINING_MODEL_FACTORY,
)
from .numerical_diagnostics import (
    PACRE_VC_DIAGNOSTIC_POLICY,
    PACRE_VC_LEGACY_ATOL,
    PACRE_VC_LEGACY_RTOL,
    legacy_subtraction_diagnostics,
    run_pacre_fp64_oracle,
)
from .pacre_vc import (
    PACRE_VC_CANDIDATE,
    PACRE_VC_FIELDS_FQCN,
    PACRE_VC_VERIFIER_POLICY,
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREVerifierCorrectedConfig,
)
from .protocol import (
    source_closure_payload,
    verify_fingerprinted,
)


PACRE_VC_DR_GATE_SCHEMA: Final = (
    "cure-lite-pacre-v23-verifier-corrected-real-dr-gate-v1"
)
PACRE_VC_DR_RUN_ID: Final = (
    "pacre_v23_verifier_corrected_D_R_structural_r1"
)
PACRE_VC_DR_EXECUTION_SEED: Final = 42
PACRE_VC_DR_PASS_DECISION: Final = "PACRE_V23_D_R_STRUCTURAL_PASS"
PACRE_VC_DR_FAIL_DECISION: Final = "PACRE_V23_D_R_STRUCTURAL_FAIL"
PACRE_DR_PASS_DECISION: Final = PACRE_VC_DR_PASS_DECISION
PACRE_DR_FAIL_DECISION: Final = PACRE_VC_DR_FAIL_DECISION
PACRE_VC_DR_TARGET_STATE_COUNT: Final = (
    2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
)
PACRE_VC_DR_CONTEXT_STATE_COUNT: Final = (
    6 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
)
PACRE_VC_LEGACY_FORMULA_POLICY: Final = (
    "exact_v22_six_subchecks_original_formula_rtol2e-6_atol2e-7_v1"
)
PACRE_VC_DR_CHECK_NAMES: Final = (
    "01_dataset_free_prerequisite_exact_and_passed",
    "02_real_D_R_seed42_population_bound",
    "03_exact_pacre_vc_model_config_factory_and_parameter_contract",
    "04_complete_state_forward_ledger_and_exact_v22_fields_type",
    "05_target_state_forward_algebra_and_phase_semantics_valid",
    "06_each_target_group_has_one_bound_residual_flip_latent_witness",
    "07_no_exact_target_positive_latent_collision",
    "08_zero_readout_anchor_and_fixed_readout_witness",
    "09_real_pmope_initialization_gradient_path",
    "10_field_loss_direction_correct_for_all_roles",
    "11_model_population_cache_rng_and_grad_buffers_preserved",
    "12_read_only_zero_update_D_R_scope",
    "13_context_state_forward_algebra_and_phase_semantics_valid",
)
PACRE_DR_CHECK_NAMES: Final = PACRE_VC_DR_CHECK_NAMES
PACRE_VC_DR_IMPLEMENTATION_PATHS: Final = (
    "cure_lite/coverage_state_sobolev.py",
    "cure_lite/experiment/coverage_state_bfa_dr_gate.py",
    "cure_lite/experiment/coverage_state_bounded_protocol.py",
    "cure_lite/experiment/coverage_state_paet_dr_gate.py",
    "cure_lite/experiment/coverage_state_real_dr_inputs.py",
    "cure_lite_v22/pacre.py",
    "cure_lite_v22/dr_gate.py",
    "cure_lite_v23/pacre_vc.py",
    "cure_lite_v23/factory.py",
    "cure_lite_v23/algebra_verifier.py",
    "cure_lite_v23/numerical_diagnostics.py",
    "cure_lite_v23/environment.py",
    "cure_lite_v23/dataset_free.py",
    "cure_lite_v23/dr_gate.py",
)


def _implementation_binding() -> tuple[tuple[str, str], ...]:
    root = Path(__file__).resolve().parents[1]
    rows: list[tuple[str, str]] = []
    for relative in PACRE_VC_DR_IMPLEMENTATION_PATHS:
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.resolve(strict=True) != path
        ):
            raise RuntimeError(
                f"invalid PACRE-VC D_R source: {relative}"
            )
        rows.append((relative, file_sha256(path)))
    return tuple(rows)


def _validate_dataset_free_receipt(
    receipt: Mapping[str, object],
) -> str:
    """Validate the generated-only prerequisite used by the real-data gate."""

    if not isinstance(receipt, Mapping):
        raise TypeError("dataset-free receipt must be a mapping")
    body = dict(receipt)
    fingerprint = body.pop("receipt_fingerprint", None)
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or stable_fingerprint(body) != fingerprint
        or body.get("candidate") != PACRE_VC_CANDIDATE
        or body.get("gate_passed") is not True
        or body.get("parameter_count") != PACRE_FORMAL_PARAMETER_COUNT
        or body.get("D_R_accessed") is not False
        or body.get("D_V_accessed") is not False
        or body.get("D_T_accessed") is not False
        or body.get("training_performed") is not False
    ):
        raise PermissionError(
            "PACRE-VC dataset-free prerequisite is invalid"
        )
    return fingerprint


def _legacy_six_subchecks(
    fields: CoverageStatePACREFields,
) -> dict[str, object]:
    """Replay the exact v22 formulas; outcomes remain attribution-only."""

    residual = fields.phase_feature_residual
    phases = int(residual.shape[1])
    tolerance = (
        4.0
        * float(phases)
        * torch.finfo(torch.float32).eps
        * (
            1.0
            + float(
                fields.phase_feature_affine.detach().abs().amax()
            )
        )
    )
    outcomes = {
        "legacy_01_phase_reconstruction_allclose": bool(
            torch.allclose(
                fields.phase_feature_mean + residual,
                fields.phase_feature_affine,
                rtol=0.0,
                atol=tolerance,
            )
        ),
        "legacy_02_phase_centering_bound": bool(
            torch.all(residual.sum(dim=1).abs() <= tolerance)
        ),
        "legacy_03_actual_joint_subtraction_allclose": bool(
            torch.allclose(
                fields.actual_specific_joint_affine
                - fields.actual_common_joint_affine,
                residual,
                rtol=PACRE_VC_LEGACY_RTOL,
                atol=PACRE_VC_LEGACY_ATOL,
            )
        ),
        "legacy_04_flipped_joint_subtraction_allclose": bool(
            torch.allclose(
                fields.flipped_specific_joint_affine
                - fields.flipped_common_joint_affine,
                residual,
                rtol=PACRE_VC_LEGACY_RTOL,
                atol=PACRE_VC_LEGACY_ATOL,
            )
        ),
        "legacy_05_actual_hidden_allclose": bool(
            torch.allclose(
                fields.actual_compatibility_hidden,
                F.silu(fields.actual_specific_joint_affine)
                - F.silu(fields.actual_common_joint_affine),
                rtol=PACRE_VC_LEGACY_RTOL,
                atol=PACRE_VC_LEGACY_ATOL,
            )
        ),
        "legacy_06_flipped_hidden_allclose": bool(
            torch.allclose(
                fields.flipped_compatibility_hidden,
                F.silu(fields.flipped_specific_joint_affine)
                - F.silu(fields.flipped_common_joint_affine),
                rtol=PACRE_VC_LEGACY_RTOL,
                atol=PACRE_VC_LEGACY_ATOL,
            )
        ),
    }
    formula = {
        "policy": PACRE_VC_LEGACY_FORMULA_POLICY,
        "subcheck_names": list(outcomes),
        "rtol_hex": PACRE_VC_LEGACY_RTOL.hex(),
        "atol_hex": PACRE_VC_LEGACY_ATOL.hex(),
        "centering_tolerance_hex": float(tolerance).hex(),
    }
    return {
        "gate_eligible": False,
        "attribution_only": True,
        "observed_subcheck_count": len(outcomes),
        "formula": formula,
        "formula_fingerprint": stable_fingerprint(formula),
        "outcomes": outcomes,
        "failed_subchecks": [
            name for name, passed in outcomes.items() if not passed
        ],
    }


def _state_manifest_row(
    state: object,
    *,
    scope: str,
) -> dict[str, object]:
    state_id = str(getattr(state, "state_id"))
    feature = getattr(state, "feature")
    occupancy = getattr(state, "occupancy")
    if not isinstance(feature, Tensor) or not isinstance(occupancy, Tensor):
        raise TypeError("D_R state tensors are incomplete")
    row = {
        "scoped_state_id": f"{scope}::{state_id}",
        "source_state_id": state_id,
        "scope": scope,
        "sample_id": str(getattr(state, "sample_id")),
        "state_kind": str(getattr(state, "state_kind")),
        "endpoint": str(getattr(state, "endpoint")),
        "feature_fingerprint": tensor_content_fingerprint(feature),
        "occupancy_fingerprint": tensor_content_fingerprint(occupancy),
    }
    return {**row, "state_manifest_fingerprint": stable_fingerprint(row)}


@torch.no_grad()
def _state_algebra_row(
    model: CURELitePACREVerifierCorrectedLevelSet,
    state: object,
    *,
    scope: str,
    device: torch.device,
) -> dict[str, object]:
    manifest = _state_manifest_row(state, scope=scope)
    feature = getattr(state, "feature").to(
        device=device,
        dtype=torch.float32,
    )
    occupancy = getattr(state, "occupancy").to(device=device)
    fields = model.forward_fields(feature, occupancy)
    verification = verify_pacre_v22_forward_fields(model, fields)
    legacy = _legacy_six_subchecks(fields)
    legacy_diagnostics = legacy_subtraction_diagnostics(fields)
    oracle = run_pacre_fp64_oracle(model, fields)
    fields_fqcn = (
        f"{type(fields).__module__}.{type(fields).__qualname__}"
    )
    integrity_passed = bool(
        verification.passed
        and oracle.integrity.passed
        and fields_fqcn == PACRE_VC_FIELDS_FQCN
        and legacy["observed_subcheck_count"] == 6
    )
    return {
        **manifest,
        "selected_device": str(device),
        "fields_fqcn": fields_fqcn,
        "gate_eligible_integrity_passed": integrity_passed,
        "algebra": verification.canonical_payload(),
        "legacy_six_check_replay": legacy,
        "legacy_subtraction_diagnostics": (
            legacy_diagnostics.canonical_payload()
        ),
        "fp64_oracle": oracle.canonical_payload(),
    }


def _scope_summary(
    rows: list[dict[str, object]],
    *,
    scope: str,
    expected_count: int,
) -> dict[str, object]:
    scoped_ids = [str(row["scoped_state_id"]) for row in rows]
    source_ids = [str(row["source_state_id"]) for row in rows]
    if any(row.get("scope") != scope for row in rows):
        raise AssertionError("state ledger scope changed")
    legacy_failed: dict[str, int] = {}
    actual_mismatches = 0
    flipped_mismatches = 0
    for row in rows:
        legacy = row["legacy_six_check_replay"]
        diagnostics = row["legacy_subtraction_diagnostics"]
        if not isinstance(legacy, Mapping) or not isinstance(
            diagnostics,
            Mapping,
        ):
            raise TypeError("state diagnostic ledger is incomplete")
        for name in legacy.get("failed_subchecks", ()):
            legacy_failed[str(name)] = legacy_failed.get(str(name), 0) + 1
        actual = diagnostics.get("actual")
        flipped = diagnostics.get("flipped")
        if not isinstance(actual, Mapping) or not isinstance(
            flipped,
            Mapping,
        ):
            raise TypeError("legacy lanes are incomplete")
        actual_mismatches += int(
            actual["failed_under_v22_allclose_count"]
        )
        flipped_mismatches += int(
            flipped["failed_under_v22_allclose_count"]
        )
    body = {
        "scope": scope,
        "expected_state_count": expected_count,
        "observed_state_count": len(rows),
        "state_ids": scoped_ids,
        "state_ids_fingerprint": stable_fingerprint(scoped_ids),
        "state_ids_unique": len(scoped_ids) == len(set(scoped_ids)),
        "ordered_source_state_ids": source_ids,
        "ordered_source_state_ids_fingerprint": stable_fingerprint(
            source_ids
        ),
        "source_state_ids_unique": len(source_ids) == len(set(source_ids)),
        "all_gate_eligible_integrity_passed": all(
            row.get("gate_eligible_integrity_passed") is True
            for row in rows
        ),
        "legacy_replay_complete": all(
            isinstance(row.get("legacy_six_check_replay"), Mapping)
            and row["legacy_six_check_replay"].get(
                "observed_subcheck_count"
            )
            == 6
            for row in rows
        ),
        "legacy_failed_state_counts": dict(
            sorted(legacy_failed.items())
        ),
        "legacy_actual_mismatch_element_count": actual_mismatches,
        "legacy_flipped_mismatch_element_count": flipped_mismatches,
    }
    return {**body, "scope_summary_fingerprint": stable_fingerprint(body)}


def _algebra_ledger(
    model: CURELitePACREVerifierCorrectedLevelSet,
    population: CoverageStateBoundedPopulation,
    *,
    device: torch.device,
) -> dict[str, object]:
    target_states, context_states = _state_specs(population)
    target_rows = [
        _state_algebra_row(
            model,
            state,
            scope="target",
            device=device,
        )
        for state in target_states
    ]
    context_rows = [
        _state_algebra_row(
            model,
            state,
            scope="context",
            device=device,
        )
        for state in context_states
    ]
    target_summary = _scope_summary(
        target_rows,
        scope="target",
        expected_count=PACRE_VC_DR_TARGET_STATE_COUNT,
    )
    context_summary = _scope_summary(
        context_rows,
        scope="context",
        expected_count=PACRE_VC_DR_CONTEXT_STATE_COUNT,
    )
    target_ids = set(target_summary["state_ids"])
    context_ids = set(context_summary["state_ids"])
    body = {
        "schema_version": (
            "cure-lite-pacre-v23-verifier-state-ledger-v1"
        ),
        "algebra_policy": PACRE_VC_ALGEBRA_POLICY,
        "diagnostic_policy": PACRE_VC_DIAGNOSTIC_POLICY,
        "legacy_formula_policy": PACRE_VC_LEGACY_FORMULA_POLICY,
        "scope_qualification_policy": (
            "target_or_context_prefix_preserves_128_forward_call_ledger_v1"
        ),
        "target_summary": target_summary,
        "context_summary": context_summary,
        "target_rows": target_rows,
        "context_rows": context_rows,
        "target_context_scoped_ids_disjoint": target_ids.isdisjoint(
            context_ids
        ),
        "union_unique_state_call_count": len(target_ids | context_ids),
        "expected_union_state_call_count": (
            PACRE_VC_DR_TARGET_STATE_COUNT
            + PACRE_VC_DR_CONTEXT_STATE_COUNT
        ),
    }
    return {**body, "ledger_fingerprint": stable_fingerprint(body)}


def _raw_state_fingerprints_equal(
    first: torch.nn.Module,
    second: torch.nn.Module,
) -> bool:
    first_state = dict(first.state_dict())
    second_state = dict(second.state_dict())
    return (
        tuple(first_state) == tuple(second_state)
        and all(
            tensor_content_fingerprint(first_state[name])
            == tensor_content_fingerprint(second_state[name])
            for name in first_state
        )
    )


def _probe(
    population: CoverageStateBoundedPopulation,
    *,
    device: torch.device,
) -> dict[str, object]:
    before_cpu_rng = torch.get_rng_state().clone()
    before_cuda_rng = (
        torch.cuda.get_rng_state(device).clone()
        if device.type == "cuda"
        else None
    )
    before_population = population.population_fingerprint
    before_cache = population.cache.cache_fingerprint
    with _deterministic_execution_scope() as deterministic_execution:
        with torch.random.fork_rng(
            devices=_cuda_rng_devices(device),
            device_type=(
                "cuda" if device.type == "cuda" else None
            ),
        ):
            torch.random.default_generator.manual_seed(
                PACRE_VC_DR_EXECUTION_SEED
            )
            config = CoverageStatePACREVerifierCorrectedConfig(
                feature_channels=PACRE_FORMAL_FEATURE_CHANNELS,
                feature_stride=PACRE_FORMAL_FEATURE_STRIDE,
                width=PACRE_FORMAL_WIDTH,
            )
            model = PACRE_VC_TRAINING_MODEL_FACTORY(config).to(
                device=device,
                dtype=torch.float32,
            )
            model.eval()
            initial_model = coverage_state_model_fingerprint(model)
            initial_ids = {
                name: id(parameter)
                for name, parameter in model.named_parameters()
            }

            torch.random.default_generator.manual_seed(
                PACRE_VC_DR_EXECUTION_SEED
            )
            v22_model = (
                CURELitePhaseAlignedCenteredResidualCompatibilityEnergyLevelSet(
                    CoverageStatePACREConfig(
                        feature_channels=PACRE_FORMAL_FEATURE_CHANNELS,
                        feature_stride=PACRE_FORMAL_FEATURE_STRIDE,
                        width=PACRE_FORMAL_WIDTH,
                    )
                )
                .to(device=device, dtype=torch.float32)
                .eval()
            )
            v22_initial_model = coverage_state_model_fingerprint(
                v22_model
            )
            v22_v23_initial_raw_parity = (
                _raw_state_fingerprints_equal(model, v22_model)
            )
            del v22_model

            # The inherited v22 representation probe supplies the unchanged
            # witness/collision/zero-readout/functional-readout contracts.
            representation = _v22_representation_probe(
                model,
                population,
                device=device,
            )
            ledger = _algebra_ledger(
                model,
                population,
                device=device,
            )
            gradient = _v22_gradient_probe(
                model,
                population,
                device=device,
            )
            direction = _direction_probe(
                population,
                device=device,
            )
            final_model = coverage_state_model_fingerprint(model)
            final_ids = {
                name: id(parameter)
                for name, parameter in model.named_parameters()
            }
            model_contract = coverage_state_model_contract_payload(model)
            parameter_grad_buffers_unretained = all(
                parameter.grad is None
                for parameter in model.parameters()
            )
    population.verify_unchanged()
    after_cuda_rng = (
        torch.cuda.get_rng_state(device).clone()
        if device.type == "cuda"
        else None
    )
    return {
        "device": str(device),
        "execution_seed": PACRE_VC_DR_EXECUTION_SEED,
        "model_fqcn": (
            f"{type(model).__module__}.{type(model).__qualname__}"
        ),
        "config_fqcn": (
            f"{type(config).__module__}.{type(config).__qualname__}"
        ),
        "fields_fqcn": PACRE_VC_FIELDS_FQCN,
        "model_contract": model_contract,
        "model_contract_fingerprint": stable_fingerprint(model_contract),
        "initial_model_fingerprint": initial_model,
        "v22_initial_model_fingerprint": v22_initial_model,
        "v22_v23_initial_raw_state_parity": (
            v22_v23_initial_raw_parity
        ),
        "final_model_fingerprint": final_model,
        "parameter_ids_preserved": initial_ids == final_ids,
        "representation": representation,
        "algebra_ledger": ledger,
        "gradient_path": gradient,
        "field_direction": direction,
        "deterministic_execution": deterministic_execution,
        "population_fingerprint_before": before_population,
        "population_fingerprint_after": (
            population.population_fingerprint
        ),
        "cache_fingerprint_before": before_cache,
        "cache_fingerprint_after": population.cache.cache_fingerprint,
        "global_cpu_rng_preserved": torch.equal(
            before_cpu_rng,
            torch.get_rng_state(),
        ),
        "selected_device_rng_preserved": (
            before_cuda_rng is None
            or torch.equal(before_cuda_rng, after_cuda_rng)
        ),
        "parameter_grad_buffers_unretained": (
            parameter_grad_buffers_unretained
            and gradient["parameter_grad_buffers_unretained"]
        ),
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_performed": False,
        "D_R_accessed": True,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }


def recompute_pacre_vc_dr_checks(
    *,
    dataset_free_receipt_fingerprint: str,
    real_inputs: CoverageStateRealDRInputs,
    bounded_population: CoverageStateBoundedPopulation,
    probe: Mapping[str, object],
) -> tuple[tuple[str, bool], ...]:
    representation = probe.get("representation")
    ledger = probe.get("algebra_ledger")
    gradient = probe.get("gradient_path")
    direction = probe.get("field_direction")
    model_contract = probe.get("model_contract")
    sealed_replay = probe.get("sealed_v22_replay_binding")
    if not all(
        isinstance(value, Mapping)
        for value in (
            representation,
            ledger,
            gradient,
            direction,
            model_contract,
            sealed_replay,
        )
    ):
        raise TypeError("PACRE-VC D_R probe is incomplete")
    target = ledger.get("target_summary")
    context = ledger.get("context_summary")
    contract_config = model_contract.get("config")
    parameter_shapes = model_contract.get("parameter_shapes")
    if not all(
        isinstance(value, Mapping)
        for value in (
            target,
            context,
            contract_config,
            parameter_shapes,
        )
    ):
        raise TypeError("PACRE-VC D_R contract is incomplete")
    model_fqcn = (
        "cure_lite_v23.pacre_vc."
        "CURELitePACREVerifierCorrectedLevelSet"
    )
    config_fqcn = (
        "cure_lite_v23.pacre_vc."
        "CoverageStatePACREVerifierCorrectedConfig"
    )
    checks = {
        PACRE_VC_DR_CHECK_NAMES[0]: (
            isinstance(dataset_free_receipt_fingerprint, str)
            and len(dataset_free_receipt_fingerprint) == 64
        ),
        PACRE_VC_DR_CHECK_NAMES[1]: (
            real_inputs.source_binding.split == "D_R"
            and real_inputs.scalar_cache.raw_catalog.split == "D_R"
            and bounded_population.seed == COVERAGE_STATE_BOUNDED_SEED
            and bounded_population.source_cache
            is real_inputs.scalar_cache
            and bounded_population.source_cache_fingerprint
            == real_inputs.scalar_cache.cache_fingerprint
        ),
        PACRE_VC_DR_CHECK_NAMES[2]: (
            probe.get("model_fqcn") == model_fqcn
            and probe.get("config_fqcn") == config_fqcn
            and probe.get("fields_fqcn") == PACRE_VC_FIELDS_FQCN
            and model_contract.get("model_class") == model_fqcn
            and model_contract.get("config_class") == config_fqcn
            and model_contract.get("parameter_count")
            == PACRE_FORMAL_PARAMETER_COUNT
            and contract_config.get("feature_channels")
            == PACRE_FORMAL_FEATURE_CHANNELS
            and contract_config.get("feature_stride")
            == PACRE_FORMAL_FEATURE_STRIDE
            and contract_config.get("width") == PACRE_FORMAL_WIDTH
            and contract_config.get("verifier_policy")
            == PACRE_VC_VERIFIER_POLICY
            and tuple(sorted(parameter_shapes))
            == tuple(sorted(PACRE_VC_PARAMETER_NAMES))
            and probe.get("model_contract_fingerprint")
            == stable_fingerprint(model_contract)
            and probe.get("v22_v23_initial_raw_state_parity") is True
            and probe.get("v22_initial_model_fingerprint")
            == probe.get("initial_model_fingerprint")
            and sealed_replay.get(
                "expected_initial_model_fingerprint"
            )
            == sealed_replay.get(
                "observed_initial_model_fingerprint"
            )
            == probe.get("initial_model_fingerprint")
            and sealed_replay.get(
                "initial_model_fingerprint_matches"
            )
            is True
        ),
        PACRE_VC_DR_CHECK_NAMES[3]: (
            target.get("observed_state_count")
            == target.get("expected_state_count")
            == PACRE_VC_DR_TARGET_STATE_COUNT
            and context.get("observed_state_count")
            == context.get("expected_state_count")
            == PACRE_VC_DR_CONTEXT_STATE_COUNT
            and target.get("state_ids_unique") is True
            and context.get("state_ids_unique") is True
            and target.get("source_state_ids_unique") is True
            and context.get("source_state_ids_unique") is True
            and sealed_replay.get(
                "expected_ordered_target_state_ids_fingerprint"
            )
            == sealed_replay.get(
                "observed_ordered_target_state_ids_fingerprint"
            )
            == target.get(
                "ordered_source_state_ids_fingerprint"
            )
            and sealed_replay.get("ordered_target_state_ids_match")
            is True
            and target.get("legacy_replay_complete") is True
            and context.get("legacy_replay_complete") is True
            and ledger.get("target_context_scoped_ids_disjoint") is True
            and ledger.get("union_unique_state_call_count")
            == ledger.get("expected_union_state_call_count")
            == (
                PACRE_VC_DR_TARGET_STATE_COUNT
                + PACRE_VC_DR_CONTEXT_STATE_COUNT
            )
            and representation.get("all_fields_exact_pacre") is True
        ),
        PACRE_VC_DR_CHECK_NAMES[4]: (
            target.get("all_gate_eligible_integrity_passed") is True
        ),
        PACRE_VC_DR_CHECK_NAMES[5]: (
            representation.get(
                "all_target_groups_have_joint_witness"
            )
            is True
        ),
        PACRE_VC_DR_CHECK_NAMES[6]: (
            representation.get("exact_latent_collision_count") == 0
        ),
        PACRE_VC_DR_CHECK_NAMES[7]: (
            representation.get(
                "zero_readout_anchor_all_target_states"
            )
            is True
            and representation.get(
                "fixed_readout_interaction_nonzero"
            )
            is True
        ),
        PACRE_VC_DR_CHECK_NAMES[8]: (
            gradient.get("initial_gradient_finite")
            == {
                "joint_hidden_bias": True,
                "joint_state_weight": True,
                "scalar_energy_weight": True,
            }
            and gradient.get("initial_gradient_nonzero")
            == {
                "joint_hidden_bias": False,
                "joint_state_weight": False,
                "scalar_energy_weight": True,
            }
            and gradient.get("readout_visible_upstream_dormant") is True
            and gradient.get(
                "readout_to_upstream_cross_gradient_finite_nonzero"
            )
            == [True, True]
            and gradient.get("parameter_grad_buffers_unretained")
            is True
        ),
        PACRE_VC_DR_CHECK_NAMES[9]: (
            direction.get("all_roles_finite_nonzero_correct") is True
        ),
        PACRE_VC_DR_CHECK_NAMES[10]: (
            probe.get("initial_model_fingerprint")
            == probe.get("final_model_fingerprint")
            and probe.get("parameter_ids_preserved") is True
            and probe.get("population_fingerprint_before")
            == probe.get("population_fingerprint_after")
            and probe.get("cache_fingerprint_before")
            == probe.get("cache_fingerprint_after")
            and probe.get("global_cpu_rng_preserved") is True
            and probe.get("selected_device_rng_preserved") is True
            and probe.get("parameter_grad_buffers_unretained") is True
            and isinstance(
                probe.get("deterministic_execution"),
                Mapping,
            )
            and probe["deterministic_execution"].get(
                "restored_exactly"
            )
            is True
        ),
        PACRE_VC_DR_CHECK_NAMES[11]: (
            probe.get("D_R_accessed") is True
            and probe.get("D_V_accessed") is False
            and probe.get("D_T_accessed") is False
            and probe.get("optimizer_constructed") is False
            and probe.get("optimizer_steps") == 0
            and probe.get("parameter_updates") == 0
            and probe.get("training_performed") is False
        ),
        PACRE_VC_DR_CHECK_NAMES[12]: (
            context.get("all_gate_eligible_integrity_passed") is True
        ),
    }
    if tuple(checks) != PACRE_VC_DR_CHECK_NAMES:
        raise AssertionError("PACRE-VC D_R check order changed")
    return tuple(checks.items())


recompute_pacre_dr_checks = recompute_pacre_vc_dr_checks


@dataclass(frozen=True)
class CoverageStatePACREVerifierCorrectedDRGateReceipt:
    """Immutable binding of one authorized read-only v23 ``D_R`` probe."""

    dataset_free_receipt_fingerprint: str
    pre_run_authorization_fingerprint: str
    runtime_environment_fingerprint: str
    source_closure_fingerprint: str
    real_inputs_fingerprint: str
    population_fingerprint: str
    cache_fingerprint: str
    implementation_binding: tuple[tuple[str, str], ...]
    prerequisite_fingerprints: tuple[tuple[str, str], ...]
    probe_json: str
    checks: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        digests = (
            self.dataset_free_receipt_fingerprint,
            self.pre_run_authorization_fingerprint,
            self.runtime_environment_fingerprint,
            self.source_closure_fingerprint,
            self.real_inputs_fingerprint,
            self.population_fingerprint,
            self.cache_fingerprint,
            *(value for _, value in self.implementation_binding),
            *(value for _, value in self.prerequisite_fingerprints),
        )
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in digests
        ):
            raise ValueError("PACRE-VC D_R digest is malformed")
        if (
            tuple(name for name, _ in self.implementation_binding)
            != PACRE_VC_DR_IMPLEMENTATION_PATHS
            or tuple(name for name, _ in self.checks)
            != PACRE_VC_DR_CHECK_NAMES
            or any(not isinstance(value, bool) for _, value in self.checks)
            or tuple(name for name, _ in self.prerequisite_fingerprints)
            != tuple(
                sorted(name for name, _ in self.prerequisite_fingerprints)
            )
        ):
            raise ValueError("PACRE-VC D_R receipt contract changed")
        value = json.loads(self.probe_json)
        if not isinstance(value, dict):
            raise ValueError("PACRE-VC probe JSON must be an object")

    @property
    def gate_passed(self) -> bool:
        return bool(self.checks) and all(value for _, value in self.checks)

    @property
    def decision(self) -> str:
        return (
            PACRE_VC_DR_PASS_DECISION
            if self.gate_passed
            else PACRE_VC_DR_FAIL_DECISION
        )

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(name for name, passed in self.checks if not passed)

    @property
    def probe(self) -> dict[str, object]:
        value = json.loads(self.probe_json)
        if not isinstance(value, dict):
            raise AssertionError("PACRE-VC probe changed")
        return value

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": PACRE_VC_DR_GATE_SCHEMA,
            "run_id": PACRE_VC_DR_RUN_ID,
            "candidate": PACRE_VC_CANDIDATE,
            "verifier_policy": PACRE_VC_VERIFIER_POLICY,
            "algebra_policy": PACRE_VC_ALGEBRA_POLICY,
            "fields_fqcn": PACRE_VC_FIELDS_FQCN,
            "dataset_free_receipt_fingerprint": (
                self.dataset_free_receipt_fingerprint
            ),
            "pre_run_authorization_fingerprint": (
                self.pre_run_authorization_fingerprint
            ),
            "runtime_environment_fingerprint": (
                self.runtime_environment_fingerprint
            ),
            "source_closure_fingerprint": (
                self.source_closure_fingerprint
            ),
            "prerequisite_fingerprints": dict(
                self.prerequisite_fingerprints
            ),
            "real_inputs_fingerprint": self.real_inputs_fingerprint,
            "population_fingerprint": self.population_fingerprint,
            "cache_fingerprint": self.cache_fingerprint,
            "implementation_binding": dict(
                self.implementation_binding
            ),
            "probe": self.probe,
            "checks": dict(self.checks),
            "failed_checks": list(self.failed_checks),
            "gate_passed": self.gate_passed,
            "decision": self.decision,
            "D_R_adaptive_verifier_correction": True,
            "independent_confirmation": False,
            "identifiability_only": True,
            "performance_claim_supported": False,
            "formal_800_route_granted": self.gate_passed,
            "formal_800_seed": 42,
            "formal_800_epochs": 800,
            "formal_800_steps_per_epoch": 40,
            "formal_800_updates": 32000,
            "formal_800_from_scratch": True,
            "formal_800_execution_authorized": False,
            "bounded_400_required": False,
            "bounded_400_authorized": False,
            "bounded_400_authorization_effect": False,
            "training_performed": False,
            "D_R_accessed": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "split_manifest_metadata_read": True,
            "D_V_tensor_payload_accessed": False,
            "D_T_tensor_payload_accessed": False,
        }

    @cached_property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify_sources_unchanged(self) -> None:
        if self.implementation_binding != _implementation_binding():
            raise RuntimeError("PACRE-VC D_R implementation changed")
        closure = source_closure_payload()
        if (
            closure.get("closure_fingerprint")
            != self.source_closure_fingerprint
        ):
            raise RuntimeError("PACRE-VC source closure changed")

    def verify_unchanged(
        self,
        *,
        dataset_free_receipt: Mapping[str, object],
        real_inputs: CoverageStateRealDRInputs,
        bounded_population: CoverageStateBoundedPopulation,
    ) -> None:
        self.verify_sources_unchanged()
        dataset_fingerprint = _validate_dataset_free_receipt(
            dataset_free_receipt
        )
        real_inputs.verify_unchanged()
        bounded_population.verify_unchanged()
        if (
            dataset_fingerprint
            != self.dataset_free_receipt_fingerprint
            or real_inputs.build_fingerprint
            != self.real_inputs_fingerprint
            or bounded_population.population_fingerprint
            != self.population_fingerprint
            or bounded_population.cache.cache_fingerprint
            != self.cache_fingerprint
            or bounded_population.source_cache
            is not real_inputs.scalar_cache
        ):
            raise RuntimeError("PACRE-VC D_R receipt inputs changed")
        recomputed = recompute_pacre_vc_dr_checks(
            dataset_free_receipt_fingerprint=dataset_fingerprint,
            real_inputs=real_inputs,
            bounded_population=bounded_population,
            probe=self.probe,
        )
        if recomputed != self.checks:
            raise RuntimeError("PACRE-VC D_R checks changed")


CoverageStatePACREDRGateReceipt = (
    CoverageStatePACREVerifierCorrectedDRGateReceipt
)


def pacre_vc_dr_receipt_from_payload(
    payload: Mapping[str, object],
) -> CoverageStatePACREVerifierCorrectedDRGateReceipt:
    """Reconstruct and strictly validate one persisted canonical receipt."""

    if not isinstance(payload, Mapping):
        raise TypeError("PACRE-VC D_R payload must be a mapping")
    probe = payload.get("probe")
    checks = payload.get("checks")
    implementation = payload.get("implementation_binding")
    prerequisites = payload.get("prerequisite_fingerprints")
    if not all(
        isinstance(value, Mapping)
        for value in (
            probe,
            checks,
            implementation,
            prerequisites,
        )
    ):
        raise ValueError("PACRE-VC D_R payload is incomplete")
    receipt = CoverageStatePACREVerifierCorrectedDRGateReceipt(
        dataset_free_receipt_fingerprint=str(
            payload.get("dataset_free_receipt_fingerprint")
        ),
        pre_run_authorization_fingerprint=str(
            payload.get("pre_run_authorization_fingerprint")
        ),
        runtime_environment_fingerprint=str(
            payload.get("runtime_environment_fingerprint")
        ),
        source_closure_fingerprint=str(
            payload.get("source_closure_fingerprint")
        ),
        real_inputs_fingerprint=str(
            payload.get("real_inputs_fingerprint")
        ),
        population_fingerprint=str(
            payload.get("population_fingerprint")
        ),
        cache_fingerprint=str(payload.get("cache_fingerprint")),
        implementation_binding=tuple(
            (name, str(implementation[name]))
            for name in PACRE_VC_DR_IMPLEMENTATION_PATHS
        ),
        prerequisite_fingerprints=tuple(
            sorted(
                (str(name), str(value))
                for name, value in prerequisites.items()
            )
        ),
        probe_json=canonical_json(dict(probe)),
        checks=tuple(
            (str(name), value)
            for name, value in checks.items()
        ),
    )
    if (
        payload.get("schema_version") != PACRE_VC_DR_GATE_SCHEMA
        or payload.get("run_id") != PACRE_VC_DR_RUN_ID
        or dict(payload) != receipt.canonical_payload()
    ):
        raise ValueError("persisted PACRE-VC D_R receipt changed")
    return receipt


def _validated_authorization(
    authorization: Mapping[str, object],
    *,
    dataset_fingerprint: str,
    runtime_fingerprint: str,
    source_fingerprint: str,
) -> tuple[
    str,
    tuple[tuple[str, str], ...],
    dict[str, str],
]:
    fingerprint = verify_fingerprinted(
        authorization,
        field="authorization_fingerprint",
    )
    prerequisites = authorization.get("prerequisite_fingerprints")
    v22_failure = authorization.get("v22_sealed_failure")
    if (
        authorization.get("schema_version")
        != "cure-lite-pacre-v23-D_R-pre-run-authorization-v1"
        or authorization.get("run_id") != PACRE_VC_DR_RUN_ID
        or authorization.get("candidate") != PACRE_VC_CANDIDATE
        or authorization.get("status")
        != "PACRE_V23_D_R_PRE_RUN_AUTHORIZED"
        or authorization.get("dataset_free_receipt_fingerprint")
        != dataset_fingerprint
        or authorization.get("runtime_environment_fingerprint")
        != runtime_fingerprint
        or authorization.get("source_closure_fingerprint")
        != source_fingerprint
        or authorization.get("D_R_accessed") is not False
        or authorization.get("D_V_accessed") is not False
        or authorization.get("D_T_accessed") is not False
        or authorization.get("training_performed") is not False
        or not isinstance(prerequisites, Mapping)
        or not prerequisites
        or not isinstance(v22_failure, Mapping)
        or not isinstance(
            v22_failure.get("initial_model_fingerprint"),
            str,
        )
        or len(v22_failure["initial_model_fingerprint"]) != 64
        or not isinstance(
            v22_failure.get(
                "ordered_target_state_ids_fingerprint"
            ),
            str,
        )
        or len(
            v22_failure[
                "ordered_target_state_ids_fingerprint"
            ]
        )
        != 64
        or any(
            not isinstance(name, str)
            or not isinstance(value, str)
            or len(value) != 64
            for name, value in prerequisites.items()
        )
    ):
        raise PermissionError("PACRE-VC D_R authorization is invalid")
    return (
        fingerprint,
        tuple(
            sorted(
                (str(name), str(value))
                for name, value in prerequisites.items()
            )
        ),
        {
            "initial_model_fingerprint": str(
                v22_failure["initial_model_fingerprint"]
            ),
            "ordered_target_state_ids_fingerprint": str(
                v22_failure[
                    "ordered_target_state_ids_fingerprint"
                ]
            ),
        },
    )


def _resolve_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cpu":
        if resolved.index is not None:
            raise ValueError("CPU D_R device must not have an index")
        return resolved
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("PACRE-VC D_R gate requires CPU or available CUDA")
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    if (
        resolved.index is None
        or resolved.index < 0
        or resolved.index >= torch.cuda.device_count()
    ):
        raise ValueError("requested CUDA device is unavailable")
    return resolved


def run_pacre_vc_dr_gate(
    *,
    dataset_free_receipt: Mapping[str, object],
    pre_run_authorization: Mapping[str, object],
    runtime_environment_lock: Mapping[str, object],
    real_inputs: CoverageStateRealDRInputs,
    bounded_population: CoverageStateBoundedPopulation,
    device: torch.device | str = "cpu",
) -> CoverageStatePACREVerifierCorrectedDRGateReceipt:
    """Consume one authorization and run the zero-update v23 ``D_R`` gate."""

    dataset_fingerprint = _validate_dataset_free_receipt(
        dataset_free_receipt
    )
    if not isinstance(real_inputs, CoverageStateRealDRInputs):
        raise TypeError("real_inputs must be CoverageStateRealDRInputs")
    if not isinstance(
        bounded_population,
        CoverageStateBoundedPopulation,
    ):
        raise TypeError(
            "bounded_population must be CoverageStateBoundedPopulation"
        )
    real_inputs.verify_unchanged()
    bounded_population.verify_unchanged()
    if (
        bounded_population.source_cache is not real_inputs.scalar_cache
        or bounded_population.seed != COVERAGE_STATE_BOUNDED_SEED
    ):
        raise PermissionError("PACRE-VC D_R input bindings differ")
    resolved = _resolve_device(device)
    runtime_fingerprint = verify_runtime_environment(
        runtime_environment_lock,
        resolved,
    )
    closure = source_closure_payload()
    source_fingerprint = str(closure["closure_fingerprint"])
    (
        authorization_fingerprint,
        prerequisites,
        sealed_v22_binding,
    ) = _validated_authorization(
        pre_run_authorization,
        dataset_fingerprint=dataset_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        source_fingerprint=source_fingerprint,
    )
    probe = _probe(bounded_population, device=resolved)
    ledger = probe.get("algebra_ledger")
    target_summary = (
        ledger.get("target_summary")
        if isinstance(ledger, Mapping)
        else None
    )
    if not isinstance(target_summary, Mapping):
        raise RuntimeError("PACRE-VC target ledger is absent")
    replay_binding = {
        "expected_initial_model_fingerprint": (
            sealed_v22_binding["initial_model_fingerprint"]
        ),
        "observed_initial_model_fingerprint": (
            probe["initial_model_fingerprint"]
        ),
        "initial_model_fingerprint_matches": (
            probe["initial_model_fingerprint"]
            == sealed_v22_binding["initial_model_fingerprint"]
        ),
        "expected_ordered_target_state_ids_fingerprint": (
            sealed_v22_binding[
                "ordered_target_state_ids_fingerprint"
            ]
        ),
        "observed_ordered_target_state_ids_fingerprint": (
            target_summary[
                "ordered_source_state_ids_fingerprint"
            ]
        ),
        "ordered_target_state_ids_match": (
            target_summary[
                "ordered_source_state_ids_fingerprint"
            ]
            == sealed_v22_binding[
                "ordered_target_state_ids_fingerprint"
            ]
        ),
    }
    probe = {
        **probe,
        "sealed_v22_replay_binding": replay_binding,
    }
    checks = recompute_pacre_vc_dr_checks(
        dataset_free_receipt_fingerprint=dataset_fingerprint,
        real_inputs=real_inputs,
        bounded_population=bounded_population,
        probe=probe,
    )
    receipt = CoverageStatePACREVerifierCorrectedDRGateReceipt(
        dataset_free_receipt_fingerprint=dataset_fingerprint,
        pre_run_authorization_fingerprint=authorization_fingerprint,
        runtime_environment_fingerprint=runtime_fingerprint,
        source_closure_fingerprint=source_fingerprint,
        real_inputs_fingerprint=real_inputs.build_fingerprint,
        population_fingerprint=bounded_population.population_fingerprint,
        cache_fingerprint=bounded_population.cache.cache_fingerprint,
        implementation_binding=_implementation_binding(),
        prerequisite_fingerprints=prerequisites,
        probe_json=canonical_json(probe),
        checks=checks,
    )
    receipt.verify_unchanged(
        dataset_free_receipt=dataset_free_receipt,
        real_inputs=real_inputs,
        bounded_population=bounded_population,
    )
    return receipt


run_pacre_dr_gate = run_pacre_vc_dr_gate


__all__ = [
    "PACRE_DR_CHECK_NAMES",
    "PACRE_DR_FAIL_DECISION",
    "PACRE_DR_PASS_DECISION",
    "PACRE_VC_DR_CHECK_NAMES",
    "PACRE_VC_DR_CONTEXT_STATE_COUNT",
    "PACRE_VC_DR_EXECUTION_SEED",
    "PACRE_VC_DR_FAIL_DECISION",
    "PACRE_VC_DR_GATE_SCHEMA",
    "PACRE_VC_DR_PASS_DECISION",
    "PACRE_VC_DR_RUN_ID",
    "PACRE_VC_DR_TARGET_STATE_COUNT",
    "CoverageStatePACREDRGateReceipt",
    "CoverageStatePACREVerifierCorrectedDRGateReceipt",
    "recompute_pacre_dr_checks",
    "recompute_pacre_vc_dr_checks",
    "pacre_vc_dr_receipt_from_payload",
    "run_pacre_dr_gate",
    "run_pacre_vc_dr_gate",
]
