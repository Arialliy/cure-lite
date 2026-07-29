"""Read-only zero-update structural gate for v24 GCR-PACRE.

The real-data entry point accepts only the frozen ``D_R`` input graph and a
pre-access token produced by an external protocol verifier.  This module has
no ``D_V`` or ``D_T`` loader, constructs no optimizer, performs no parameter
update, and makes no performance claim.

The gate uses a deterministic nonzero diagnostic readout so that the residual
and gate paths are observable before training.  Every model tensor, buffer,
gradient slot, RNG state, and deterministic-runtime flag is restored after the
probe.  The final decision is derived from raw observations rather than from
caller-supplied pass booleans.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from functools import cached_property
import importlib
import json
from math import isfinite
import os
from pathlib import Path
import struct
from typing import Final, Iterator, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from cure_lite.cache.schema import (
    canonical_json,
    file_sha256,
    stable_fingerprint,
)
from cure_lite.coverage_state_precomputed_cache import (
    CoverageStateCachedNatural,
    CoverageStateCachedPair,
)
from cure_lite.coverage_state_sobolev import (
    CoverageStateAbsoluteTargets,
    CoverageStatePairTargets,
    CoverageStateSobolevConfig,
    coverage_state_absolute_sobolev_loss_from_targets,
    coverage_state_pmope_pair_loss_from_targets,
    prepare_coverage_state_focused_absolute_targets,
    prepare_coverage_state_pair_targets,
)
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_ROLE_COUNT,
    COVERAGE_STATE_BOUNDED_SEED,
    CoverageStateBoundedPopulation,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    CoverageStateRealDRInputs,
)
from cure_lite.experiment.coverage_state_training import (
    coverage_state_model_contract_payload,
    coverage_state_model_fingerprint,
)
from cure_lite.paired_types import tensor_content_fingerprint

from .dataset_free import (
    verify_gcr_pacre_dataset_free_receipt,
)
from .factory import (
    GCR_PACRE_FORMAL_FEATURE_CHANNELS,
    GCR_PACRE_FORMAL_FEATURE_STRIDE,
    GCR_PACRE_FORMAL_PARAMETER_COUNT,
    GCR_PACRE_FORMAL_WIDTH,
    GCR_PACRE_PARAMETER_NAMES,
    build_formal_gcr_pacre_training_model,
)
from .gcr_pacre import (
    GCR_PACRE_CANDIDATE,
    GCR_PACRE_FIELDS_FQCN,
    GCR_PACRE_FP64_ORACLE_ABS_TOL,
    GCR_PACRE_FP64_ORACLE_MAX_ULP,
    CURELiteGatedCommonResidualPACRELevelSet,
    CoverageStateGCRPACREFields,
    compare_gcr_pacre_fp32_to_fp64_oracle,
    summarize_gcr_pacre_gate_saturation,
    validate_gcr_pacre_fields,
)


GCR_PACRE_DR_GATE_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-real-dr-structural-gate-v1"
)
GCR_PACRE_DR_GENERATED_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-generated-dr-contract-audit-v1"
)
GCR_PACRE_DR_RUN_ID: Final = (
    "gcr_pacre_v24_D_R_zero_update_structural_r1"
)
GCR_PACRE_DR_EXECUTION_SEED: Final = 42
GCR_PACRE_DR_TARGET_STATE_COUNT: Final = (
    2 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
)
GCR_PACRE_DR_CONTEXT_STATE_COUNT: Final = (
    6 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
)
GCR_PACRE_DR_PASS_DECISION: Final = (
    "GCR_PACRE_V24_D_R_STRUCTURAL_PASS"
)
GCR_PACRE_DR_FAIL_DECISION: Final = (
    "GCR_PACRE_V24_D_R_STRUCTURAL_FAIL"
)
GCR_PACRE_DR_FIXED_READOUT_POLICY: Final = (
    "linspace_0.5_to_1.5_width_temporary_exact_restore_v1"
)
GCR_PACRE_DR_COLLISION_POLICY: Final = (
    "exact_fp32_value_collision_signed_zero_coalesced_v1"
)
GCR_PACRE_DR_GATE_GRADIENT_POLICY: Final = (
    "isolated_D_detach_times_G_autograd_grad_no_buffer_retention_v1"
)
GCR_PACRE_DR_PREACCESS_SCHEMA: Final = (
    "cure-lite-v24-D_R-structural-preaccess-authorization-v1"
)
GCR_PACRE_DR_PREACCESS_STAGE_ID: Final = (
    "gcr_pacre_v24_D_R_structural"
)
GCR_PACRE_DR_PREACCESS_STATUS: Final = (
    "GCR_PACRE_V24_D_R_STRUCTURAL_PREACCESS_AUTHORIZED"
)
GCR_PACRE_DR_ACCESS_AUDIT_SCHEMA: Final = (
    "cure-lite-v24-split-access-audit-v1"
)
GCR_PACRE_DR_FROZEN_V23_METADATA_SCHEMA: Final = (
    "cure-lite-v24-D_R-frozen-v23-input-metadata-binding-v1"
)
GCR_PACRE_DR_RUN_START_SCHEMA: Final = (
    "cure-lite-v24-D_R-persistent-run-start-v1"
)
GCR_PACRE_DR_RUN_START_PATH_POLICY: Final = (
    "fixed_repository_run_root_authorization_fingerprint_filename_v1"
)
GCR_PACRE_DR_RUN_START_PARENT: Final = (
    "runs/irstd1k_stage_a_seed42"
)
GCR_PACRE_DR_DATASET_FREE_RECEIPT_R2_PATH: Final = (
    "protocols/IRSTD-1K/gcr_pacre_v24/"
    "dataset_free_receipt_r2.json"
)
GCR_PACRE_DR_ACCESS_AUDIT_PATH: Final = (
    "protocols/IRSTD-1K/gcr_pacre_v24/"
    "D_R_structural_access_audit.json"
)
GCR_PACRE_DR_PREACCESS_AUTHORIZATION_PATH: Final = (
    "protocols/IRSTD-1K/gcr_pacre_v24/"
    "D_R_structural_authorization.json"
)
GCR_PACRE_DR_RECEIPT_PATH: Final = (
    "protocols/IRSTD-1K/gcr_pacre_v24/"
    "D_R_structural_receipt.json"
)
GCR_PACRE_DR_SOURCE_PATHS: Final = {
    "manifest_path": "protocols/IRSTD-1K/stage_a_seed42/manifest.json",
    "state_index_path": (
        "runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3/"
        "d_r/state_cache/index.json"
    ),
    "geometry_config_path": (
        "protocols/IRSTD-1K/geometry_safe_p0_v2/config.json"
    ),
    "geometry_receipt_path": (
        "runs/irstd1k_stage_a_seed42/"
        "cure_lite_geometry_safe_p0_v2_r1/receipts/"
        "geometry_catalog.json"
    ),
    "observability_config_path": (
        "protocols/IRSTD-1K/coverage_state_observability_v1/config.json"
    ),
}
GCR_PACRE_DR_V23_AUTHORIZATION_PATH: Final = (
    "protocols/IRSTD-1K/pacre_v23_verifier_corrected/"
    "D_R_pre_run_authorization.json"
)
GCR_PACRE_DR_V23_RUN_ROOT: Final = (
    "runs/irstd1k_stage_a_seed42/"
    "pacre_v23_verifier_corrected_D_R_structural_r1"
)
GCR_PACRE_DR_V23_AUTHORIZATION_FINGERPRINT: Final = (
    "1b87029350a5cff92d6319d189dc3425bca0254dd5fa4df568961e93ed46b5fe"
)
GCR_PACRE_DR_V23_COMPLETE_FINGERPRINT: Final = (
    "ee9d4b30a225ff73d56224874057475fac45785884a3148047942ef8641e35d3"
)
GCR_PACRE_DR_V23_INPUTS_FINGERPRINT: Final = (
    "c05d289600f3ce6a60434b50428a324ce9e421ff5c1109806b676d7763b9a8a5"
)
GCR_PACRE_DR_V23_GATE_RECEIPT_FINGERPRINT: Final = (
    "0edb3e99259e55b4591b38c7adee1261acf261cd6a7e847cbe261cdef6250d82"
)
GCR_PACRE_DR_V23_GATE_WRAPPER_FINGERPRINT: Final = (
    "901b8bdc273cfb150f94e9ca3389be1dd390a45c2823b2125acc655dd1680f4f"
)

GCR_PACRE_DR_CHECK_NAMES: Final = (
    "01_dataset_free_prerequisite_exact_and_passed",
    "02_real_D_R_seed42_population_bound",
    "03_exact_gcr_pacre_model_config_factory_and_parameter_contract",
    "04_complete_32_target_96_context_state_ledger",
    "05_target_state_forward_algebra_and_phase_semantics_valid",
    "06_each_target_group_has_bound_residual_flip_latent_witness",
    "07_no_exact_target_positive_latent_collision",
    "08_zero_readout_anchor_and_fixed_readout_witness",
    "09_real_pmope_initialization_gradient_path",
    "10_field_loss_direction_correct_for_all_roles",
    "11_model_population_cache_rng_and_grad_buffers_preserved",
    "12_read_only_zero_update_D_R_scope",
    "13_context_state_forward_algebra_and_phase_semantics_valid",
    "14_common_compatibility_finite_and_non_degenerate",
    "15_common_gate_even_and_residual_interaction_odd",
    "16_common_evidence_cannot_create_completion_without_residual",
    "17_target_groups_have_bound_residual_direction_and_gate_gradient_witness",
    "18_no_exact_target_background_gated_latent_collision",
    "19_model_population_cache_rng_and_grad_buffers_preserved",
    "20_read_only_zero_update_D_R_scope",
    "21_gate_saturation_distribution_recorded_without_post_hoc_threshold",
    "22_fast_reference_and_complete_fields_ledger_agree",
    "23_efficiency_receipt_bound",
)

GCR_PACRE_DR_IMPLEMENTATION_PATHS: Final = (
    "cure_lite/__init__.py",
    "cure_lite/cache/__init__.py",
    "cure_lite/cache/base_cache.py",
    "cure_lite/cache/schema.py",
    "cure_lite/cache/state_cache.py",
    "cure_lite/base_identity.py",
    "cure_lite/calibration.py",
    "cure_lite/calibration_ledger.py",
    "cure_lite/config.py",
    "cure_lite/coverage_state_batches.py",
    "cure_lite/coverage_state_binary_flip_antisymmetric.py",
    "cure_lite/coverage_state_centered_mixed_interaction.py",
    "cure_lite/coverage_state_device_cache.py",
    "cure_lite/coverage_state_level_set.py",
    "cure_lite/coverage_state_observability.py",
    "cure_lite/coverage_state_phase_aligned_evidence_transport.py",
    "cure_lite/coverage_state_phase_preserving.py",
    "cure_lite/coverage_state_precomputed_cache.py",
    "cure_lite/coverage_state_raw_catalog.py",
    "cure_lite/coverage_state_schedule.py",
    "cure_lite/coverage_state_sobolev.py",
    "cure_lite/coverage_state_supremal_projection.py",
    "cure_lite/data.py",
    "cure_lite/decoder.py",
    "cure_lite/efficiency.py",
    "cure_lite/experiment/__init__.py",
    "cure_lite/experiment/artifacts.py",
    "cure_lite/experiment/cache_pipeline.py",
    "cure_lite/experiment/coverage_state_bfa_dataset_free.py",
    "cure_lite/experiment/coverage_state_bfa_dr_gate.py",
    "cure_lite/experiment/coverage_state_bounded_protocol.py",
    "cure_lite/experiment/coverage_state_cmif_dataset_free.py",
    "cure_lite/experiment/coverage_state_observability_protocol.py",
    "cure_lite/experiment/coverage_state_paet_dataset_free.py",
    "cure_lite/experiment/coverage_state_paet_dr_gate.py",
    "cure_lite/experiment/coverage_state_raw_catalog.py",
    "cure_lite/experiment/coverage_state_real_dr_inputs.py",
    "cure_lite/experiment/coverage_state_training.py",
    "cure_lite/experiment/deployment.py",
    "cure_lite/experiment/efficiency_evidence.py",
    "cure_lite/experiment/evaluation_pipeline.py",
    "cure_lite/experiment/formal_anchor.py",
    "cure_lite/experiment/formal_evaluation.py",
    "cure_lite/experiment/formal_training.py",
    "cure_lite/experiment/geometry_catalog_protocol.py",
    "cure_lite/experiment/geometry_safe_catalog.py",
    "cure_lite/experiment/paired_artifacts.py",
    "cure_lite/experiment/p0_geometry.py",
    "cure_lite/experiment/p0_protocol.py",
    "cure_lite/experiment/seed_registry.py",
    "cure_lite/experiment/stage_a_m_extension.py",
    "cure_lite/experiment/stage_a_m_runner.py",
    "cure_lite/experiment/stage_a_runner.py",
    "cure_lite/experiment/training_pipeline.py",
    "cure_lite/frozen_base.py",
    "cure_lite/instances.py",
    "cure_lite/intervention.py",
    "cure_lite/losses.py",
    "cure_lite/matching.py",
    "cure_lite/metrics.py",
    "cure_lite/model.py",
    "cure_lite/occupancy.py",
    "cure_lite/paired_control_inputs.py",
    "cure_lite/paired_control_losses.py",
    "cure_lite/paired_losses.py",
    "cure_lite/paired_types.py",
    "cure_lite/sampling.py",
    "cure_lite/splits.py",
    "cure_lite/stage_a.py",
    "cure_lite/supervision.py",
    "cure_lite/train/__init__.py",
    "cure_lite/train/coverage_state_fused_step.py",
    "cure_lite/train/engine.py",
    "cure_lite/train/paired_control_step.py",
    "cure_lite/train/paired_step.py",
    "cure_lite/train/pools.py",
    "cure_lite/train/step.py",
    "cure_lite/types.py",
    "cure_lite_v22/__init__.py",
    "cure_lite_v22/dataset_free.py",
    "cure_lite_v22/dr_gate.py",
    "cure_lite_v22/factory.py",
    "cure_lite_v22/pacre.py",
    "cure_lite_v23/__init__.py",
    "cure_lite_v23/algebra_verifier.py",
    "cure_lite_v23/dataset_free.py",
    "cure_lite_v23/dr_gate.py",
    "cure_lite_v23/environment.py",
    "cure_lite_v23/factory.py",
    "cure_lite_v23/numeric_stress.py",
    "cure_lite_v23/numerical_diagnostics.py",
    "cure_lite_v23/pacre_vc.py",
    "cure_lite_v23/parity.py",
    "cure_lite_v23/protocol.py",
    "cure_lite_v24/__init__.py",
    "cure_lite_v24/gcr_pacre.py",
    "cure_lite_v24/factory.py",
    "cure_lite_v24/dataset_free.py",
    "cure_lite_v24/dr_gate.py",
    "cure_lite_v24/formal_cache_artifacts.py",
    "tools/__init__.py",
    "tools/gcr_pacre_v24_protocol.py",
    "tools/run_cure_lite_v24_gcr_pacre_dr_gate.py",
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fqcn(value: object) -> str:
    value_type = value if isinstance(value, type) else type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _implementation_binding() -> tuple[tuple[str, str], ...]:
    root = _repository_root()
    rows: list[tuple[str, str]] = []
    for relative in GCR_PACRE_DR_IMPLEMENTATION_PATHS:
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.resolve(strict=True) != path
        ):
            raise RuntimeError(
                f"invalid GCR-PACRE D_R implementation source: {relative}"
            )
        rows.append((relative, file_sha256(path)))
    return tuple(rows)


def _canonical_regular_file(path: str | Path, *, name: str) -> Path:
    candidate = Path(path).expanduser()
    absolute = Path(os.path.abspath(candidate))
    if candidate.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if (
        resolved != absolute
        or not resolved.is_file()
        or resolved.is_symlink()
    ):
        raise ValueError(f"{name} must be a canonical regular file")
    return resolved


def _read_strict_json(path: str | Path, *, name: str) -> dict[str, object]:
    resolved = _canonical_regular_file(path, name=name)

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite value {item}")
            ),
        )
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must contain one JSON object")
    return dict(value)


def _verified_self_fingerprint(
    payload: Mapping[str, object],
    *,
    field: str,
    expected: str,
    name: str,
) -> str:
    body = dict(payload)
    fingerprint = body.pop(field, None)
    if (
        fingerprint != expected
        or not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(body)
    ):
        raise ValueError(f"{name} frozen fingerprint changed")
    return str(fingerprint)


def _frozen_v23_dr_input_metadata() -> dict[str, object]:
    """Verify and expose only sealed v23 aggregate ``D_R`` metadata.

    This function opens JSON receipts only.  It never opens the manifest,
    state-cache index payloads, feature tensors, masks, images, or labels.
    """

    root = _repository_root()
    run_root = root / GCR_PACRE_DR_V23_RUN_ROOT
    authorization_path = (
        root / GCR_PACRE_DR_V23_AUTHORIZATION_PATH
    )
    complete_path = run_root / "COMPLETE.json"
    inputs_path = run_root / "receipts/inputs.json"
    gate_path = run_root / "receipts/dr_gate.json"
    authorization = _read_strict_json(
        authorization_path,
        name="frozen v23 D_R authorization metadata",
    )
    complete = _read_strict_json(
        complete_path,
        name="frozen v23 D_R COMPLETE metadata",
    )
    inputs = _read_strict_json(
        inputs_path,
        name="frozen v23 D_R inputs metadata",
    )
    gate = _read_strict_json(
        gate_path,
        name="frozen v23 D_R gate metadata",
    )
    authorization_fingerprint = _verified_self_fingerprint(
        authorization,
        field="authorization_fingerprint",
        expected=GCR_PACRE_DR_V23_AUTHORIZATION_FINGERPRINT,
        name="v23 D_R authorization",
    )
    complete_fingerprint = _verified_self_fingerprint(
        complete,
        field="complete_fingerprint",
        expected=GCR_PACRE_DR_V23_COMPLETE_FINGERPRINT,
        name="v23 D_R COMPLETE",
    )
    inputs_fingerprint = _verified_self_fingerprint(
        inputs,
        field="receipt_fingerprint",
        expected=GCR_PACRE_DR_V23_INPUTS_FINGERPRINT,
        name="v23 D_R inputs",
    )
    gate_wrapper_fingerprint = _verified_self_fingerprint(
        gate,
        field="wrapper_fingerprint",
        expected=GCR_PACRE_DR_V23_GATE_WRAPPER_FINGERPRINT,
        name="v23 D_R gate wrapper",
    )
    artifact_files = complete.get("artifact_files")
    allowed_artifacts = {
        "attempt.json",
        "receipts/decision.json",
        "receipts/dr_gate.json",
        "receipts/inputs.json",
        "receipts/preflight.json",
    }
    if (
        not isinstance(artifact_files, Mapping)
        or set(artifact_files) != allowed_artifacts
        or complete.get("artifact_count") != len(allowed_artifacts)
    ):
        raise ValueError("v23 D_R COMPLETE artifact inventory changed")
    for relative, digest in artifact_files.items():
        if not isinstance(relative, str) or not _is_sha256(digest):
            raise TypeError("v23 D_R artifact binding is invalid")
        artifact_path = _canonical_regular_file(
            run_root / relative,
            name=f"frozen v23 D_R artifact {relative}",
        )
        if file_sha256(artifact_path) != digest:
            raise ValueError(
                f"frozen v23 D_R artifact bytes changed: {relative}"
            )

    run_id = "pacre_v23_verifier_corrected_D_R_structural_r1"
    false_boundaries = (
        authorization.get("D_V_accessed") is False
        and authorization.get("D_T_accessed") is False
        and authorization.get("D_V_tensor_payload_accessed") is False
        and authorization.get("D_T_tensor_payload_accessed") is False
        and authorization.get("training_performed") is False
        and complete.get("D_V_accessed") is False
        and complete.get("D_T_accessed") is False
        and complete.get("D_V_tensor_payload_accessed") is False
        and complete.get("D_T_tensor_payload_accessed") is False
        and complete.get("training_performed") is False
        and inputs.get("D_V_accessed") is False
        and inputs.get("D_T_accessed") is False
        and inputs.get("D_V_tensor_payload_accessed") is False
        and inputs.get("D_T_tensor_payload_accessed") is False
        and inputs.get("training_performed") is False
        and gate.get("D_V_accessed") is False
        and gate.get("D_T_accessed") is False
        and gate.get("D_V_tensor_payload_accessed") is False
        and gate.get("D_T_tensor_payload_accessed") is False
        and gate.get("training_performed") is False
    )
    if (
        authorization.get("schema_version")
        != "cure-lite-pacre-v23-D_R-pre-run-authorization-v1"
        or authorization.get("run_id") != run_id
        or authorization.get("candidate") != "PACRE-VC-v23"
        or authorization.get("status")
        != "PACRE_V23_D_R_PRE_RUN_AUTHORIZED"
        or authorization.get("D_R_accessed") is not False
        or authorization.get("execution_seed") != 42
        or authorization.get("target_state_count") != 32
        or authorization.get("context_state_count") != 96
        or authorization.get("optimizer_allowed") is not False
        or authorization.get("parameter_updates_allowed") != 0
        or authorization.get("single_use") is not True
        or complete.get("schema_version")
        != "cure-lite-pacre-v23-D_R-terminal-complete-v1"
        or complete.get("run_id") != run_id
        or complete.get("status") != "PACRE_V23_D_R_STRUCTURAL_PASS"
        or complete.get("D_R_accessed") is not True
        or complete.get("D_R_gate_passed") is not True
        or complete.get("authorization_fingerprint")
        != authorization_fingerprint
        or complete.get("D_R_gate_receipt_fingerprint")
        != GCR_PACRE_DR_V23_GATE_RECEIPT_FINGERPRINT
        or complete.get("gate_invocations") != 1
        or complete.get("optimizer_steps") != 0
        or complete.get("parameter_updates") != 0
        or complete.get("formal_800_epochs") != 800
        or complete.get("formal_800_steps_per_epoch") != 40
        or complete.get("formal_800_updates") != 32_000
        or complete.get("formal_800_execution_authorized") is not False
        or inputs.get("schema_version")
        != "cure-lite-pacre-v23-D_R-inputs-v1"
        or inputs.get("run_id") != run_id
        or inputs.get("authorization_fingerprint")
        != authorization_fingerprint
        or inputs.get("D_R_accessed") is not True
        or inputs.get("construction_invocations")
        != {"population": 1, "real_inputs": 1}
        or gate.get("schema_version")
        != "cure-lite-pacre-v23-D_R-gate-wrapper-v1"
        or gate.get("run_id") != run_id
        or gate.get("authorization_fingerprint")
        != authorization_fingerprint
        or gate.get("receipt_fingerprint")
        != GCR_PACRE_DR_V23_GATE_RECEIPT_FINGERPRINT
        or gate.get("decision") != "PACRE_V23_D_R_STRUCTURAL_PASS"
        or gate.get("gate_passed") is not True
        or gate.get("failed_checks") != []
        or gate.get("gate_invocations") != 1
        or gate.get("optimizer_steps") != 0
        or gate.get("parameter_updates") != 0
        or not false_boundaries
    ):
        raise PermissionError("frozen v23 D_R metadata chain changed")

    source_files = inputs.get("source_files")
    expected_source_files = authorization.get(
        "expected_real_input_bindings"
    )
    real_inputs = inputs.get("real_inputs")
    population = inputs.get("population")
    if (
        not isinstance(source_files, Mapping)
        or not isinstance(expected_source_files, Mapping)
        or dict(source_files) != dict(expected_source_files)
        or set(source_files)
        != {
            "geometry_config_path",
            "geometry_receipt_path",
            "manifest_path",
            "observability_config_path",
            "state_index_path",
        }
        or not isinstance(real_inputs, Mapping)
        or not isinstance(population, Mapping)
        or real_inputs.get("schema_version")
        != "cure-lite-coverage-state-real-dr-inputs-v1"
        or real_inputs.get("split") != "D_R"
        or real_inputs.get("runtime_splits") != ["D_R"]
        or real_inputs.get("representation") != "scalar_max"
        or real_inputs.get("source_binding_fingerprint") is None
        or population.get("schema_version")
        != "cure-lite-cslf-dr-bounded-population-v1"
        or population.get("split") != "D_R"
        or population.get("seed") != 42
        or population.get("role_count") != 16
        or population.get("D_V_accessed") is not False
        or population.get("D_T_accessed") is not False
    ):
        raise ValueError("frozen v23 D_R input metadata changed")
    for name, raw in source_files.items():
        if (
            not isinstance(name, str)
            or not isinstance(raw, Mapping)
            or set(raw) != {"repo_path", "file_sha256"}
            or not isinstance(raw.get("repo_path"), str)
            or not _is_sha256(raw.get("file_sha256"))
        ):
            raise TypeError("frozen v23 D_R source binding changed")

    expected_real_inputs = inputs.get("real_inputs_fingerprint")
    expected_population = inputs.get("population_fingerprint")
    expected_cache = inputs.get("cache_fingerprint")
    expected_source_binding = real_inputs.get(
        "source_binding_fingerprint"
    )
    if (
        any(
            not _is_sha256(value)
            for value in (
                expected_real_inputs,
                expected_population,
                expected_cache,
                expected_source_binding,
            )
        )
        or authorization.get("expected_real_inputs_fingerprint")
        != expected_real_inputs
        or authorization.get("expected_population_fingerprint")
        != expected_population
        or authorization.get("expected_cache_fingerprint")
        != expected_cache
        or population.get("bounded_cache_fingerprint")
        != expected_cache
    ):
        raise ValueError("frozen v23 D_R fingerprint binding changed")

    preregistration_path = (
        root
        / "protocols/IRSTD-1K/gcr_pacre_v24/preregistration.json"
    )
    preregistration = _read_strict_json(
        preregistration_path,
        name="v24 preregistration",
    )
    protocol_fingerprint = _protocol_preregistration_fingerprint()
    frozen_sources = preregistration.get("frozen_D_R_sources")
    if not isinstance(frozen_sources, Mapping):
        raise TypeError("v24 frozen D_R source metadata is absent")
    manifest = source_files["manifest_path"]
    state_index = source_files["state_index_path"]
    if (
        frozen_sources.get("manifest") != manifest
        or frozen_sources.get("state_index") != state_index
    ):
        raise PermissionError(
            "v23 aggregate inputs differ from v24 frozen D_R sources"
        )

    provenance: dict[str, object] = {
        "schema_version": GCR_PACRE_DR_FROZEN_V23_METADATA_SCHEMA,
        "protocol_preregistration_fingerprint": protocol_fingerprint,
        "v23_authorization": {
            "repo_path": GCR_PACRE_DR_V23_AUTHORIZATION_PATH,
            "file_sha256": file_sha256(authorization_path),
            "authorization_fingerprint": authorization_fingerprint,
        },
        "v23_complete": {
            "repo_path": (
                f"{GCR_PACRE_DR_V23_RUN_ROOT}/COMPLETE.json"
            ),
            "file_sha256": file_sha256(complete_path),
            "complete_fingerprint": complete_fingerprint,
        },
        "v23_inputs": {
            "repo_path": (
                f"{GCR_PACRE_DR_V23_RUN_ROOT}/receipts/inputs.json"
            ),
            "file_sha256": file_sha256(inputs_path),
            "receipt_fingerprint": inputs_fingerprint,
        },
        "v23_gate": {
            "repo_path": (
                f"{GCR_PACRE_DR_V23_RUN_ROOT}/receipts/dr_gate.json"
            ),
            "file_sha256": file_sha256(gate_path),
            "receipt_fingerprint": (
                GCR_PACRE_DR_V23_GATE_RECEIPT_FINGERPRINT
            ),
            "wrapper_fingerprint": gate_wrapper_fingerprint,
        },
        "source_files": dict(source_files),
        "source_binding_fingerprint": expected_source_binding,
        "real_inputs_fingerprint": expected_real_inputs,
        "population_fingerprint": expected_population,
        "cache_fingerprint": expected_cache,
        "D_R_tensor_payload_accessed": False,
        "D_V_tensor_payload_accessed": False,
        "D_T_tensor_payload_accessed": False,
    }
    source_manifest_fingerprint = stable_fingerprint(provenance)
    return {
        "provenance": provenance,
        "source_manifest_fingerprint": source_manifest_fingerprint,
        "source_binding_fingerprint": str(expected_source_binding),
        "manifest_file_sha256": str(manifest["file_sha256"]),
        "state_index_file_sha256": str(
            state_index["file_sha256"]
        ),
        "real_inputs_fingerprint": str(expected_real_inputs),
        "population_fingerprint": str(expected_population),
        "cache_fingerprint": str(expected_cache),
    }


def _resolve_device(device: torch.device | str) -> torch.device:
    try:
        resolved = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise TypeError("device must be CPU or an explicit CUDA device") from error
    if resolved.type == "cpu":
        if resolved.index is not None:
            raise ValueError("CPU device must not have an index")
        return resolved
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("D_R gate requires CPU or available CUDA")
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    if (
        resolved.index is None
        or resolved.index < 0
        or resolved.index >= torch.cuda.device_count()
    ):
        raise ValueError("requested CUDA device is unavailable")
    return resolved


def _cuda_rng_devices(device: torch.device) -> list[int]:
    if device.type != "cuda":
        return []
    if device.index is None:
        raise ValueError("CUDA device index is required")
    return [device.index]


@contextmanager
def _deterministic_execution_scope() -> Iterator[dict[str, object]]:
    algorithms = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    observation = {
        "deterministic_algorithms_enabled": True,
        "deterministic_warn_only": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    try:
        yield observation
    finally:
        torch.use_deterministic_algorithms(
            algorithms,
            warn_only=warn_only,
        )
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_allow_tf32
        observation["restored_exactly"] = (
            torch.are_deterministic_algorithms_enabled() == algorithms
            and (
                torch.is_deterministic_algorithms_warn_only_enabled()
                == warn_only
            )
            and torch.backends.cudnn.benchmark == cudnn_benchmark
            and (
                torch.backends.cudnn.deterministic
                == cudnn_deterministic
            )
            and torch.backends.cuda.matmul.allow_tf32 == allow_tf32
            and torch.backends.cudnn.allow_tf32 == cudnn_allow_tf32
        )


def _tensor_map_fingerprint(values: Mapping[str, Tensor]) -> str:
    return stable_fingerprint(
        {
            name: tensor_content_fingerprint(value)
            for name, value in sorted(values.items())
        }
    )


def _model_state_fingerprint(model: torch.nn.Module) -> str:
    return _tensor_map_fingerprint(dict(model.state_dict()))


def _gradient_slots_fingerprint(model: torch.nn.Module) -> str:
    return stable_fingerprint(
        {
            name: (
                None
                if parameter.grad is None
                else tensor_content_fingerprint(parameter.grad)
            )
            for name, parameter in model.named_parameters()
        }
    )


@contextmanager
def _exact_model_restore(
    model: CURELiteGatedCommonResidualPACRELevelSet,
) -> Iterator[dict[str, object]]:
    state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    gradients = {
        name: (
            None
            if parameter.grad is None
            else parameter.grad.detach().clone()
        )
        for name, parameter in model.named_parameters()
    }
    parameter_ids = {
        name: id(parameter)
        for name, parameter in model.named_parameters()
    }
    buffer_ids = {
        name: id(buffer)
        for name, buffer in model.named_buffers()
    }
    was_training = model.training
    before_state = _model_state_fingerprint(model)
    before_gradients = _gradient_slots_fingerprint(model)
    observation: dict[str, object] = {
        "initial_state_fingerprint": before_state,
        "initial_gradient_slots_fingerprint": before_gradients,
        "parameter_ids_before": parameter_ids,
        "buffer_ids_before": buffer_ids,
    }
    try:
        yield observation
    finally:
        with torch.no_grad():
            current = dict(model.state_dict())
            if tuple(current) != tuple(state):
                raise RuntimeError("model state inventory changed during probe")
            for name, value in state.items():
                current[name].copy_(value)
        for name, parameter in model.named_parameters():
            saved = gradients[name]
            parameter.grad = (
                None if saved is None else saved.detach().clone()
            )
        model.train(was_training)
        observation.update(
            {
                "final_state_fingerprint": _model_state_fingerprint(
                    model
                ),
                "final_gradient_slots_fingerprint": (
                    _gradient_slots_fingerprint(model)
                ),
                "parameter_ids_after": {
                    name: id(parameter)
                    for name, parameter in model.named_parameters()
                },
                "buffer_ids_after": {
                    name: id(buffer)
                    for name, buffer in model.named_buffers()
                },
            }
        )
        observation["restored_exactly"] = (
            observation["initial_state_fingerprint"]
            == observation["final_state_fingerprint"]
            and observation["initial_gradient_slots_fingerprint"]
            == observation["final_gradient_slots_fingerprint"]
            and observation["parameter_ids_before"]
            == observation["parameter_ids_after"]
            and observation["buffer_ids_before"]
            == observation["buffer_ids_after"]
            and model.training == was_training
        )


@dataclass(frozen=True)
class _CoordinateState:
    state_id: str
    sample_id: str
    state_kind: str
    endpoint: str
    feature: Tensor
    occupancy: Tensor
    valid_mask: Tensor
    target_mask: Tensor
    background_mask: Tensor
    component_mask: Tensor
    target_group_id: str | None
    component_writable: bool


@dataclass(frozen=True)
class _GradientFixture:
    pair_id: str
    sample_id: str
    feature: Tensor
    occupancy_plus: Tensor
    occupancy_minus: Tensor
    targets: CoverageStatePairTargets
    sobolev_config: CoverageStateSobolevConfig


@dataclass(frozen=True)
class _DirectionFixture:
    role: str
    state_id: str
    sample_id: str
    endpoint: str
    desired: str
    mask: Tensor
    absolute_targets: CoverageStateAbsoluteTargets | None
    pair_targets: CoverageStatePairTargets | None


@dataclass(frozen=True)
class _PopulationAdapter:
    mode: str
    split: str
    seed: int
    target_states: tuple[_CoordinateState, ...]
    context_states: tuple[_CoordinateState, ...]
    gradient_fixture: _GradientFixture
    direction_fixtures: tuple[_DirectionFixture, ...]
    feature_channels: int
    feature_stride: int
    population_fingerprint: str
    cache_fingerprint: str
    source_cache_fingerprint: str
    adapter_fingerprint: str


def _mask(
    value: Tensor,
    *,
    occupancy: Tensor,
    valid: Tensor,
    excluded: tuple[Tensor, ...] = (),
) -> Tensor:
    result = value & valid & ~occupancy
    for item in excluded:
        result = result & ~item
    return result.contiguous()


def _state_from_natural(
    value: CoverageStateCachedNatural,
    *,
    target_bearing: bool,
) -> _CoordinateState:
    record = value.record
    target = (
        _mask(
            value.targets.focus_support,
            occupancy=record.occupancy,
            valid=record.valid_mask,
        )
        if target_bearing
        else torch.zeros_like(record.valid_mask)
    )
    background = _mask(
        torch.ones_like(record.valid_mask),
        occupancy=record.occupancy,
        valid=record.valid_mask,
        excluded=((record.target,) if target_bearing else ()),
    )
    return _CoordinateState(
        state_id=f"natural:{record.record_id}",
        sample_id=record.sample_id,
        state_kind=record.state_kind,
        endpoint="natural",
        feature=record.feature,
        occupancy=record.occupancy,
        valid_mask=record.valid_mask,
        target_mask=target,
        background_mask=background,
        component_mask=torch.zeros_like(target),
        target_group_id=(
            f"factual:{record.record_id}" if target_bearing else None
        ),
        component_writable=False,
    )


def _selected_states_from_population(
    population: CoverageStateBoundedPopulation,
) -> tuple[tuple[_CoordinateState, ...], tuple[_CoordinateState, ...]]:
    cache = population.cache
    factual_miss = sorted(
        (
            value
            for value in cache.natural_records
            if value.record.state_kind == "factual_miss"
        ),
        key=lambda value: value.record.record_id,
    )
    factual_no_miss = sorted(
        (
            value
            for value in cache.natural_records
            if value.record.state_kind == "factual_no_miss"
        ),
        key=lambda value: value.record.record_id,
    )
    clean = sorted(
        cache.clean_positive_records,
        key=lambda value: value.record.pair_id,
    )
    component = sorted(
        cache.component_null_records,
        key=lambda value: value.record.pair_id,
    )
    targets: list[_CoordinateState] = []
    contexts: list[_CoordinateState] = []
    for value in factual_miss:
        state = _state_from_natural(value, target_bearing=True)
        targets.append(state)
        contexts.append(state)
    for value in factual_no_miss:
        contexts.append(_state_from_natural(value, target_bearing=False))
    for value in clean:
        record = value.record
        added = (
            record.target_minus
            & ~record.target_plus
            & record.valid_mask
        ).contiguous()
        empty = torch.zeros_like(added)
        plus = _CoordinateState(
            state_id=f"pair:{record.pair_id}:plus",
            sample_id=record.sample_id,
            state_kind="clean_positive",
            endpoint="plus",
            feature=record.feature,
            occupancy=record.occupancy_plus,
            valid_mask=record.valid_mask,
            target_mask=empty,
            background_mask=_mask(
                torch.ones_like(record.valid_mask),
                occupancy=record.occupancy_plus,
                valid=record.valid_mask,
                excluded=(record.target_plus,),
            ),
            component_mask=empty,
            target_group_id=None,
            component_writable=False,
        )
        minus_target = _mask(
            added,
            occupancy=record.occupancy_minus,
            valid=record.valid_mask,
        )
        minus = _CoordinateState(
            state_id=f"pair:{record.pair_id}:minus",
            sample_id=record.sample_id,
            state_kind="clean_positive",
            endpoint="minus",
            feature=record.feature,
            occupancy=record.occupancy_minus,
            valid_mask=record.valid_mask,
            target_mask=minus_target,
            background_mask=_mask(
                torch.ones_like(record.valid_mask),
                occupancy=record.occupancy_minus,
                valid=record.valid_mask,
                excluded=(record.target_minus,),
            ),
            component_mask=empty,
            target_group_id=f"clean:{record.pair_id}",
            component_writable=False,
        )
        targets.append(minus)
        contexts.extend((plus, minus))
    for value in component:
        record = value.record
        component_support = (
            record.removed_component & record.valid_mask
        ).contiguous()
        empty = torch.zeros_like(component_support)
        for endpoint, occupancy, endpoint_target, writable in (
            (
                "plus",
                record.occupancy_plus,
                record.target_plus,
                False,
            ),
            (
                "minus",
                record.occupancy_minus,
                record.target_minus,
                True,
            ),
        ):
            contexts.append(
                _CoordinateState(
                    state_id=f"pair:{record.pair_id}:{endpoint}",
                    sample_id=record.sample_id,
                    state_kind="component_null",
                    endpoint=endpoint,
                    feature=record.feature,
                    occupancy=occupancy,
                    valid_mask=record.valid_mask,
                    target_mask=empty,
                    background_mask=_mask(
                        torch.ones_like(record.valid_mask),
                        occupancy=occupancy,
                        valid=record.valid_mask,
                        excluded=(component_support, endpoint_target),
                    ),
                    component_mask=component_support,
                    target_group_id=None,
                    component_writable=writable,
                )
            )
    return (
        tuple(sorted(targets, key=lambda value: value.state_id)),
        tuple(sorted(contexts, key=lambda value: value.state_id)),
    )


def _move_pair_targets(
    value: CoverageStatePairTargets,
    *,
    device: torch.device,
) -> CoverageStatePairTargets:
    result = CoverageStatePairTargets(
        target_field_plus=value.target_field_plus.to(device=device),
        target_field_minus=value.target_field_minus.to(device=device),
        focus_support=value.focus_support.to(device=device),
        focus_support_field=value.focus_support_field.to(device=device),
        integration_measure=value.integration_measure.to(device=device),
        valid_mask=value.valid_mask.to(device=device),
    )
    result.validate()
    return result


def _move_absolute_targets(
    value: CoverageStateAbsoluteTargets,
    *,
    device: torch.device,
) -> CoverageStateAbsoluteTargets:
    result = CoverageStateAbsoluteTargets(
        target_field=value.target_field.to(device=device),
        integration_measure=value.integration_measure.to(device=device),
        field_valid_mask=value.field_valid_mask.to(device=device),
        loss_valid_mask=value.loss_valid_mask.to(device=device),
        focus_support=value.focus_support.to(device=device),
        focus_support_field=value.focus_support_field.to(device=device),
    )
    result.validate()
    return result


def _fixtures_from_population(
    population: CoverageStateBoundedPopulation,
) -> tuple[_GradientFixture, tuple[_DirectionFixture, ...]]:
    cache = population.cache
    factual_miss = sorted(
        (
            value
            for value in cache.natural_records
            if value.record.state_kind == "factual_miss"
        ),
        key=lambda value: value.record.record_id,
    )
    factual_no_miss = sorted(
        (
            value
            for value in cache.natural_records
            if value.record.state_kind == "factual_no_miss"
        ),
        key=lambda value: value.record.record_id,
    )
    clean = sorted(
        cache.clean_positive_records,
        key=lambda value: value.record.pair_id,
    )
    component = sorted(
        cache.component_null_records,
        key=lambda value: value.record.pair_id,
    )
    if not clean:
        raise ValueError("D_R gradient fixture requires a clean pair")
    first = clean[0]
    gradient = _GradientFixture(
        pair_id=first.record.pair_id,
        sample_id=first.record.sample_id,
        feature=first.record.feature,
        occupancy_plus=first.record.occupancy_plus,
        occupancy_minus=first.record.occupancy_minus,
        targets=first.joint_targets,
        sobolev_config=cache.sobolev_config,
    )
    directions: list[_DirectionFixture] = []
    for value in factual_miss:
        record = value.record
        directions.append(
            _DirectionFixture(
                role="factual_target",
                state_id=record.record_id,
                sample_id=record.sample_id,
                endpoint="natural",
                desired="negative",
                mask=(
                    value.targets.focus_support
                    & record.valid_mask
                    & ~record.occupancy
                ).contiguous(),
                absolute_targets=value.targets,
                pair_targets=None,
            )
        )
    for value in factual_no_miss:
        record = value.record
        directions.append(
            _DirectionFixture(
                role="writable_background",
                state_id=record.record_id,
                sample_id=record.sample_id,
                endpoint="natural",
                desired="positive",
                mask=(
                    record.valid_mask & ~record.occupancy
                ).contiguous(),
                absolute_targets=value.targets,
                pair_targets=None,
            )
        )
    for value in clean:
        record = value.record
        directions.append(
            _DirectionFixture(
                role="clean_target",
                state_id=record.pair_id,
                sample_id=record.sample_id,
                endpoint="minus",
                desired="negative",
                mask=(
                    record.target_minus
                    & ~record.target_plus
                    & record.valid_mask
                    & ~record.occupancy_minus
                ).contiguous(),
                absolute_targets=None,
                pair_targets=value.joint_targets,
            )
        )
    for value in component:
        record = value.record
        directions.append(
            _DirectionFixture(
                role="writable_component",
                state_id=record.pair_id,
                sample_id=record.sample_id,
                endpoint="minus",
                desired="positive",
                mask=(
                    record.removed_component
                    & record.valid_mask
                    & ~record.occupancy_minus
                ).contiguous(),
                absolute_targets=None,
                pair_targets=value.joint_targets,
            )
        )
    return gradient, tuple(directions)


def _adapter_fingerprint(
    *,
    mode: str,
    split: str,
    seed: int,
    targets: tuple[_CoordinateState, ...],
    contexts: tuple[_CoordinateState, ...],
    population_fingerprint: str,
    cache_fingerprint: str,
) -> str:
    def rows(values: tuple[_CoordinateState, ...]) -> list[dict[str, object]]:
        return [
            {
                "state_id": value.state_id,
                "sample_id": value.sample_id,
                "kind": value.state_kind,
                "endpoint": value.endpoint,
                "feature": tensor_content_fingerprint(value.feature),
                "occupancy": tensor_content_fingerprint(value.occupancy),
                "target": tensor_content_fingerprint(value.target_mask),
                "background": tensor_content_fingerprint(
                    value.background_mask
                ),
                "component": tensor_content_fingerprint(
                    value.component_mask
                ),
            }
            for value in values
        ]

    return stable_fingerprint(
        {
            "mode": mode,
            "split": split,
            "seed": seed,
            "population_fingerprint": population_fingerprint,
            "cache_fingerprint": cache_fingerprint,
            "targets": rows(targets),
            "contexts": rows(contexts),
        }
    )


def _real_population_adapter(
    real_inputs: CoverageStateRealDRInputs,
    population: CoverageStateBoundedPopulation,
) -> _PopulationAdapter:
    if type(real_inputs) is not CoverageStateRealDRInputs:
        raise TypeError("real_inputs must be exact CoverageStateRealDRInputs")
    if type(population) is not CoverageStateBoundedPopulation:
        raise TypeError(
            "population must be exact CoverageStateBoundedPopulation"
        )
    real_inputs.verify_unchanged()
    population.verify_unchanged()
    if (
        real_inputs.source_binding.split != "D_R"
        or real_inputs.scalar_cache.raw_catalog.split != "D_R"
        or population.cache.raw_catalog.split != "D_R"
        or population.seed != COVERAGE_STATE_BOUNDED_SEED
        or population.source_cache is not real_inputs.scalar_cache
        or population.source_cache_fingerprint
        != real_inputs.scalar_cache.cache_fingerprint
    ):
        raise PermissionError("real input graph is not the frozen D_R population")
    targets, contexts = _selected_states_from_population(population)
    gradient, directions = _fixtures_from_population(population)
    first = population.cache.natural_records[0].record.feature
    adapter_fingerprint = _adapter_fingerprint(
        mode="real_D_R",
        split="D_R",
        seed=population.seed,
        targets=targets,
        contexts=contexts,
        population_fingerprint=population.population_fingerprint,
        cache_fingerprint=population.cache.cache_fingerprint,
    )
    return _PopulationAdapter(
        mode="real_D_R",
        split="D_R",
        seed=population.seed,
        target_states=targets,
        context_states=contexts,
        gradient_fixture=gradient,
        direction_fixtures=directions,
        feature_channels=int(first.shape[1]),
        feature_stride=population.cache.raw_catalog.feature_stride,
        population_fingerprint=population.population_fingerprint,
        cache_fingerprint=population.cache.cache_fingerprint,
        source_cache_fingerprint=population.source_cache_fingerprint,
        adapter_fingerprint=adapter_fingerprint,
    )


def _generated_feature(index: int) -> Tensor:
    """Return a deterministic, finite formal-channel fixture."""

    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("generated feature index must be nonnegative")
    coordinates = torch.arange(
        GCR_PACRE_FORMAL_FEATURE_CHANNELS * 2 * 2,
        dtype=torch.float32,
    ).reshape(1, GCR_PACRE_FORMAL_FEATURE_CHANNELS, 2, 2)
    phase = coordinates * 0.017 + float(index) * 0.113
    return (
        torch.sin(phase)
        + 0.35 * torch.cos(phase * 0.41 + float(index) * 0.07)
    ).contiguous()


def _single_coordinate(
    row: int,
    column: int,
    *,
    height: int = 8,
    width: int = 8,
) -> Tensor:
    value = torch.zeros((1, 1, height, width), dtype=torch.bool)
    value[0, 0, row, column] = True
    return value


def _generated_population_adapter() -> _PopulationAdapter:
    """Build 32/96 generated states without touching any dataset."""

    stride = GCR_PACRE_FORMAL_FEATURE_STRIDE
    valid = torch.ones((1, 1, 8, 8), dtype=torch.bool)
    empty = torch.zeros_like(valid)
    sobolev = CoverageStateSobolevConfig(truncation_radius=stride)
    targets: list[_CoordinateState] = []
    contexts: list[_CoordinateState] = []
    directions: list[_DirectionFixture] = []
    gradient: _GradientFixture | None = None

    for index in range(COVERAGE_STATE_BOUNDED_ROLE_COUNT):
        feature = _generated_feature(index)
        occupancy = _single_coordinate(0, index % 4)
        scene_target = _single_coordinate(
            1 + index % 2,
            1 + (index // 2) % 2,
        )
        absolute = prepare_coverage_state_focused_absolute_targets(
            scene_target,
            valid,
            valid,
            config=sobolev,
        )
        target_mask = (scene_target & ~occupancy).contiguous()
        state = _CoordinateState(
            state_id=f"generated:factual-miss:{index:02d}",
            sample_id=f"generated-sample-fm-{index:02d}",
            state_kind="factual_miss",
            endpoint="natural",
            feature=feature,
            occupancy=occupancy,
            valid_mask=valid,
            target_mask=target_mask,
            background_mask=(
                valid & ~occupancy & ~scene_target
            ).contiguous(),
            component_mask=empty,
            target_group_id=f"generated:factual:{index:02d}",
            component_writable=False,
        )
        targets.append(state)
        contexts.append(state)
        directions.append(
            _DirectionFixture(
                role="factual_target",
                state_id=state.state_id,
                sample_id=state.sample_id,
                endpoint="natural",
                desired="negative",
                mask=target_mask,
                absolute_targets=absolute,
                pair_targets=None,
            )
        )

    for index in range(COVERAGE_STATE_BOUNDED_ROLE_COUNT):
        feature = _generated_feature(100 + index)
        occupancy = _single_coordinate(7, index % 4)
        absolute = prepare_coverage_state_focused_absolute_targets(
            empty,
            valid,
            valid,
            config=sobolev,
        )
        state = _CoordinateState(
            state_id=f"generated:factual-no-miss:{index:02d}",
            sample_id=f"generated-sample-fn-{index:02d}",
            state_kind="factual_no_miss",
            endpoint="natural",
            feature=feature,
            occupancy=occupancy,
            valid_mask=valid,
            target_mask=empty,
            background_mask=(valid & ~occupancy).contiguous(),
            component_mask=empty,
            target_group_id=None,
            component_writable=False,
        )
        contexts.append(state)
        directions.append(
            _DirectionFixture(
                role="writable_background",
                state_id=state.state_id,
                sample_id=state.sample_id,
                endpoint="natural",
                desired="positive",
                mask=state.background_mask,
                absolute_targets=absolute,
                pair_targets=None,
            )
        )

    for index in range(COVERAGE_STATE_BOUNDED_ROLE_COUNT):
        feature = _generated_feature(200 + index)
        removed = _single_coordinate(
            1 + index % 2,
            1 + (index // 2) % 2,
        )
        occupancy_plus = (
            removed | _single_coordinate(6, 6)
        ).contiguous()
        occupancy_minus = (occupancy_plus & ~removed).contiguous()
        target_plus = _single_coordinate(5, 1)
        added_target = removed
        target_minus = (target_plus | added_target).contiguous()
        pair_targets = prepare_coverage_state_pair_targets(
            occupancy_plus,
            occupancy_minus,
            target_plus,
            target_minus,
            valid,
            config=sobolev,
        )
        plus = _CoordinateState(
            state_id=f"generated:clean:{index:02d}:plus",
            sample_id=f"generated-sample-clean-{index:02d}",
            state_kind="clean_positive",
            endpoint="plus",
            feature=feature,
            occupancy=occupancy_plus,
            valid_mask=valid,
            target_mask=empty,
            background_mask=(
                valid & ~occupancy_plus & ~target_plus
            ).contiguous(),
            component_mask=empty,
            target_group_id=None,
            component_writable=False,
        )
        minus = _CoordinateState(
            state_id=f"generated:clean:{index:02d}:minus",
            sample_id=f"generated-sample-clean-{index:02d}",
            state_kind="clean_positive",
            endpoint="minus",
            feature=feature,
            occupancy=occupancy_minus,
            valid_mask=valid,
            target_mask=added_target,
            background_mask=(
                valid & ~occupancy_minus & ~target_minus
            ).contiguous(),
            component_mask=empty,
            target_group_id=f"generated:clean:{index:02d}",
            component_writable=False,
        )
        targets.append(minus)
        contexts.extend((plus, minus))
        directions.append(
            _DirectionFixture(
                role="clean_target",
                state_id=minus.state_id,
                sample_id=minus.sample_id,
                endpoint="minus",
                desired="negative",
                mask=added_target,
                absolute_targets=None,
                pair_targets=pair_targets,
            )
        )
        if gradient is None:
            gradient = _GradientFixture(
                pair_id=f"generated:clean:{index:02d}",
                sample_id=minus.sample_id,
                feature=feature,
                occupancy_plus=occupancy_plus,
                occupancy_minus=occupancy_minus,
                targets=pair_targets,
                sobolev_config=sobolev,
            )

    for index in range(COVERAGE_STATE_BOUNDED_ROLE_COUNT):
        feature = _generated_feature(300 + index)
        removed = _single_coordinate(
            5 + index % 2,
            1 + (index // 2) % 2,
        )
        occupancy_plus = (
            removed | _single_coordinate(1, 6)
        ).contiguous()
        occupancy_minus = (occupancy_plus & ~removed).contiguous()
        target = _single_coordinate(3, 5)
        pair_targets = prepare_coverage_state_pair_targets(
            occupancy_plus,
            occupancy_minus,
            target,
            target,
            valid,
            config=sobolev,
        )
        for endpoint, occupancy, writable in (
            ("plus", occupancy_plus, False),
            ("minus", occupancy_minus, True),
        ):
            contexts.append(
                _CoordinateState(
                    state_id=(
                        f"generated:component:{index:02d}:{endpoint}"
                    ),
                    sample_id=(
                        f"generated-sample-component-{index:02d}"
                    ),
                    state_kind="component_null",
                    endpoint=endpoint,
                    feature=feature,
                    occupancy=occupancy,
                    valid_mask=valid,
                    target_mask=empty,
                    background_mask=(
                        valid & ~occupancy & ~removed & ~target
                    ).contiguous(),
                    component_mask=removed,
                    target_group_id=None,
                    component_writable=writable,
                )
            )
        directions.append(
            _DirectionFixture(
                role="writable_component",
                state_id=f"generated:component:{index:02d}",
                sample_id=f"generated-sample-component-{index:02d}",
                endpoint="minus",
                desired="positive",
                mask=removed,
                absolute_targets=None,
                pair_targets=pair_targets,
            )
        )

    if gradient is None:
        raise AssertionError("generated clean-pair fixture is absent")
    target_states = tuple(
        sorted(targets, key=lambda value: value.state_id)
    )
    context_states = tuple(
        sorted(contexts, key=lambda value: value.state_id)
    )
    population_fingerprint = stable_fingerprint(
        {
            "schema": GCR_PACRE_DR_GENERATED_SCHEMA,
            "seed": GCR_PACRE_DR_EXECUTION_SEED,
            "role_count": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
            "feature_policy": "analytic_sine_cosine_formal_64x2x2_v1",
        }
    )
    cache_fingerprint = stable_fingerprint(
        {
            "schema": "generated-gcr-pacre-dr-cache-v1",
            "sobolev": asdict(sobolev),
            "target_count": len(target_states),
            "context_count": len(context_states),
        }
    )
    adapter_fingerprint = _adapter_fingerprint(
        mode="generated",
        split="generated",
        seed=GCR_PACRE_DR_EXECUTION_SEED,
        targets=target_states,
        contexts=context_states,
        population_fingerprint=population_fingerprint,
        cache_fingerprint=cache_fingerprint,
    )
    return _PopulationAdapter(
        mode="generated",
        split="generated",
        seed=GCR_PACRE_DR_EXECUTION_SEED,
        target_states=target_states,
        context_states=context_states,
        gradient_fixture=gradient,
        direction_fixtures=tuple(directions),
        feature_channels=GCR_PACRE_FORMAL_FEATURE_CHANNELS,
        feature_stride=GCR_PACRE_FORMAL_FEATURE_STRIDE,
        population_fingerprint=population_fingerprint,
        cache_fingerprint=cache_fingerprint,
        source_cache_fingerprint=cache_fingerprint,
        adapter_fingerprint=adapter_fingerprint,
    )


def _finite_hex(value: float, *, name: str) -> str:
    result = float(value)
    if not isfinite(result):
        raise FloatingPointError(f"{name} must be finite")
    return result.hex()


def _phase_hidden_to_output(value: Tensor, *, stride: int) -> Tensor:
    if (
        not isinstance(value, Tensor)
        or value.ndim != 5
        or value.shape[1] != stride**2
        or value.dtype != torch.float32
    ):
        raise ValueError("phase hidden tensor has an invalid shape")
    batch, phases, width, height, columns = value.shape
    native = (
        value.permute(0, 2, 1, 3, 4)
        .reshape(batch, width * phases, height, columns)
        .contiguous()
    )
    return F.pixel_shuffle(native, stride).contiguous()


def _vectors_at(value: Tensor, mask: Tensor) -> Tensor:
    if (
        value.ndim != 4
        or mask.dtype != torch.bool
        or mask.ndim != 4
        or mask.shape[0] != value.shape[0]
        or mask.shape[1] != 1
        or tuple(mask.shape[-2:]) != tuple(value.shape[-2:])
    ):
        raise ValueError("representation and coordinate mask do not align")
    return value.permute(0, 2, 3, 1)[mask[:, 0]].contiguous()


def _row_bit_hash(value: Tensor) -> Tensor:
    rows = value.detach().to("cpu", dtype=torch.float32).contiguous()
    if rows.ndim != 2:
        raise ValueError("row hash requires a matrix")
    canonical = torch.where(rows == 0.0, torch.zeros_like(rows), rows)
    bits = canonical.contiguous().view(torch.int32).to(torch.int64)
    coefficients = (
        torch.arange(1, rows.shape[1] + 1, dtype=torch.int64)
        * 0x1F123BB5
        + 0x05F35649
    )
    return (bits * coefficients).sum(dim=1)


def _gradient_summary(
    gradients: Mapping[str, Tensor | None],
) -> dict[str, object]:
    rows: dict[str, object] = {}
    for name, raw in gradients.items():
        if raw is None:
            rows[name] = {
                "present": False,
                "finite": False,
                "nonzero_count": 0,
                "l2_norm_hex": 0.0.hex(),
                "fingerprint": None,
            }
            continue
        value = raw.detach().to("cpu", dtype=torch.float32).contiguous()
        finite = bool(torch.isfinite(value).all())
        norm = float(
            torch.linalg.vector_norm(value.to(dtype=torch.float64))
        )
        rows[name] = {
            "present": True,
            "finite": finite,
            "nonzero_count": int(torch.count_nonzero(value)),
            "l2_norm_hex": _finite_hex(
                norm,
                name=f"{name} gradient norm",
            ),
            "fingerprint": tensor_content_fingerprint(value),
        }
    return rows


def _all_gradient_rows_finite(
    rows: object,
    *,
    require_any_nonzero: bool,
) -> bool:
    if not isinstance(rows, Mapping) or not rows:
        return False
    finite = all(
        isinstance(value, Mapping)
        and value.get("present") is True
        and value.get("finite") is True
        and _is_sha256(value.get("fingerprint"))
        and isinstance(value.get("nonzero_count"), int)
        and not isinstance(value.get("nonzero_count"), bool)
        for value in rows.values()
    )
    return finite and (
        not require_any_nonzero
        or any(
            int(value["nonzero_count"]) > 0
            for value in rows.values()
            if isinstance(value, Mapping)
        )
    )


def _comparison_payload(actual: Tensor, reference: Tensor) -> dict[str, object]:
    comparison = compare_gcr_pacre_fp32_to_fp64_oracle(
        actual,
        reference,
    )
    return {
        "maximum_absolute_error_hex": _finite_hex(
            comparison.maximum_absolute_error,
            name="FP64 comparison absolute error",
        ),
        "maximum_ulp_distance": comparison.maximum_ulp_distance,
        "absolute_tolerance_hex": (
            comparison.absolute_tolerance.hex()
        ),
        "maximum_allowed_ulp": comparison.maximum_allowed_ulp,
        "absolute_envelope_passed": (
            comparison.maximum_absolute_error
            <= comparison.absolute_tolerance
        ),
        "passed": comparison.passed,
    }


def _comparison_ledger_agrees(
    comparisons: Mapping[str, object],
) -> bool:
    """Apply the frozen FP64 envelope to the two produced field tensors.

    The cancellation-scale intermediate lanes are still recorded in full,
    including their absolute/ULP errors.  They are not mislabeled as output
    fields: the public comparison contract freezes an envelope for a field,
    while the complete FP32 ledger validator mechanically replays every
    intermediate tensor.
    """

    if set(comparisons) != {
        "residual_odd_interaction",
        "common_even_energy",
        "common_gate",
        "gated_interaction",
        "native_phase_field",
        "field",
    }:
        return False
    return all(
        isinstance(comparisons[name], Mapping)
        and comparisons[name].get("passed") is True
        for name in ("native_phase_field", "field")
    )


def _fields_ledger_fingerprint(
    fields: CoverageStateGCRPACREFields,
) -> str:
    payload: dict[str, object] = {}
    for name in fields.__dataclass_fields__:
        value = getattr(fields, name)
        payload[name] = (
            tensor_content_fingerprint(value)
            if isinstance(value, Tensor)
            else list(value)
        )
    return stable_fingerprint(payload)


def _descent_row(
    *,
    fixture: _DirectionFixture,
    loss: Tensor,
    field: Tensor,
    mask: Tensor,
    loss_api: str,
) -> dict[str, object]:
    gradient = torch.autograd.grad(
        loss,
        field,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]
    selected = (-gradient)[mask]
    if selected.numel() < 1:
        raise ValueError(f"{fixture.role} descent mask is empty")
    negative = int(torch.count_nonzero(selected < 0.0))
    positive = int(torch.count_nonzero(selected > 0.0))
    zero = int(torch.count_nonzero(selected == 0.0))
    total = float(
        selected.detach().to("cpu", dtype=torch.float64).sum()
    )
    return {
        "role": fixture.role,
        "state_id": fixture.state_id,
        "sample_id": fixture.sample_id,
        "endpoint": fixture.endpoint,
        "coordinate_count": int(selected.numel()),
        "actual_mask_fingerprint": tensor_content_fingerprint(
            mask.detach().to("cpu")
        ),
        "loss_api": loss_api,
        "loss_hex": _finite_hex(
            float(loss.detach().cpu()),
            name=f"{fixture.role} direction loss",
        ),
        "descent_sum_hex": _finite_hex(
            total,
            name=f"{fixture.role} descent sum",
        ),
        "descent_negative_count": negative,
        "descent_positive_count": positive,
        "descent_zero_count": zero,
        "descent_finite": bool(torch.isfinite(selected).all()),
        "descent_nonzero": negative + positive > 0,
        "desired_field_direction": fixture.desired,
        "aggregate_descent_direction_correct": (
            total < 0.0
            if fixture.desired == "negative"
            else total > 0.0
        ),
    }


def _direction_probe(
    adapter: _PopulationAdapter,
    *,
    device: torch.device,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    amplitude = 0.9
    for fixture in adapter.direction_fixtures:
        mask = fixture.mask.to(device=device)
        if fixture.absolute_targets is not None:
            targets = _move_absolute_targets(
                fixture.absolute_targets,
                device=device,
            )
            witness = targets.target_field.detach().clone()
            witness[mask] = (
                amplitude
                if fixture.desired == "negative"
                else -amplitude
            )
            witness.requires_grad_(True)
            loss = coverage_state_absolute_sobolev_loss_from_targets(
                witness,
                targets,
                config=adapter.gradient_fixture.sobolev_config,
                validate=True,
            ).loss
            rows.append(
                _descent_row(
                    fixture=fixture,
                    loss=loss,
                    field=witness,
                    mask=mask,
                    loss_api=(
                        "coverage_state_absolute_sobolev_loss_from_targets"
                    ),
                )
            )
            continue
        if fixture.pair_targets is None:
            raise TypeError("pair direction fixture has no pair targets")
        targets = _move_pair_targets(
            fixture.pair_targets,
            device=device,
        )
        plus = targets.target_field_plus.detach().clone()
        minus = targets.target_field_minus.detach().clone()
        minus[mask] = (
            amplitude
            if fixture.desired == "negative"
            else -amplitude
        )
        plus.requires_grad_(True)
        minus.requires_grad_(True)
        loss = coverage_state_pmope_pair_loss_from_targets(
            plus,
            minus,
            targets,
            config=adapter.gradient_fixture.sobolev_config,
            validate=True,
        ).loss
        rows.append(
            _descent_row(
                fixture=fixture,
                loss=loss,
                field=minus,
                mask=mask,
                loss_api=(
                    "coverage_state_pmope_pair_loss_from_targets"
                ),
            )
        )
    expected_counts = {
        "factual_target": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "clean_target": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "writable_background": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        "writable_component": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
    }
    observed_counts = {
        role: sum(row["role"] == role for row in rows)
        for role in expected_counts
    }
    body: dict[str, object] = {
        "witness_policy": (
            "actual_precomputed_geometry_fixed_wrong_sign_"
            "field_no_model_update_v1"
        ),
        "fixed_field_amplitude_hex": amplitude.hex(),
        "expected_role_rows": expected_counts,
        "observed_role_rows": observed_counts,
        "rows": rows,
        "all_roles_finite_nonzero_correct": (
            observed_counts == expected_counts
            and all(
                row["descent_finite"] is True
                and row["descent_nonzero"] is True
                and row["aggregate_descent_direction_correct"] is True
                and float.fromhex(str(row["loss_hex"])) > 0.0
                for row in rows
            )
        ),
        "uses_actual_target_geometry": True,
        "uses_actual_valid_and_writable_masks": True,
        "model_parameter_gradient": False,
    }
    return {**body, "probe_fingerprint": stable_fingerprint(body)}


def _initial_gradient_probe(
    model: CURELiteGatedCommonResidualPACRELevelSet,
    fixture: _GradientFixture,
    *,
    device: torch.device,
) -> dict[str, object]:
    feature = torch.cat((fixture.feature, fixture.feature), dim=0).to(
        device=device,
        dtype=torch.float32,
    )
    occupancy = torch.cat(
        (fixture.occupancy_plus, fixture.occupancy_minus),
        dim=0,
    ).to(device=device)
    targets = _move_pair_targets(fixture.targets, device=device)
    field_plus, field_minus = model(feature, occupancy).split(1, dim=0)
    result = coverage_state_pmope_pair_loss_from_targets(
        field_plus,
        field_minus,
        targets,
        config=fixture.sobolev_config,
        validate=True,
    )
    parameters = dict(model.named_parameters())
    gradients_tuple = torch.autograd.grad(
        result.loss,
        tuple(parameters.values()),
        create_graph=True,
        allow_unused=False,
    )
    gradients = dict(
        zip(parameters, gradients_tuple, strict=True)
    )
    cross = torch.autograd.grad(
        gradients["scalar_energy_weight"].square().sum(),
        (
            parameters["joint_state_weight"],
            parameters["joint_hidden_bias"],
        ),
        create_graph=False,
        allow_unused=False,
    )
    gradient_rows = _gradient_summary(gradients)
    cross_rows = _gradient_summary(
        {
            "joint_state_weight": cross[0],
            "joint_hidden_bias": cross[1],
        }
    )
    return {
        "pair_id": fixture.pair_id,
        "sample_id": fixture.sample_id,
        "selection_policy": (
            "lexicographically_first_clean_positive_pair_v1"
        ),
        "loss_hex": _finite_hex(
            float(result.loss.detach().cpu()),
            name="GCR-PACRE PMOPE initialization loss",
        ),
        "initial_gradient_rows": gradient_rows,
        "readout_to_upstream_cross_gradient_rows": cross_rows,
        "readout_visible_upstream_dormant": (
            gradient_rows["scalar_energy_weight"]["nonzero_count"] > 0
            and gradient_rows["joint_state_weight"]["nonzero_count"] == 0
            and gradient_rows["joint_hidden_bias"]["nonzero_count"] == 0
        ),
        "parameter_grad_buffers_unretained": all(
            parameter.grad is None for parameter in model.parameters()
        ),
    }


def _zero_readout_anchor_probe(
    model: CURELiteGatedCommonResidualPACRELevelSet,
    adapter: _PopulationAdapter,
    *,
    device: torch.device,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for state in adapter.target_states:
            feature = state.feature.to(device=device, dtype=torch.float32)
            occupancy = state.occupancy.to(device=device)
            fields = model.forward_fields(feature, occupancy)
            expected = torch.full_like(
                fields.field,
                model.config.field_amplitude,
            )
            rows.append(
                {
                    "state_id": state.state_id,
                    "field_exact_anchor": torch.equal(
                        fields.field,
                        expected,
                    ),
                    "residual_exact_zero": bool(
                        torch.count_nonzero(
                            fields.residual_odd_interaction
                        )
                        == 0
                    ),
                    "common_even_exact_zero": bool(
                        torch.count_nonzero(fields.common_even_energy)
                        == 0
                    ),
                    "gate_exact_unit": torch.equal(
                        fields.common_gate,
                        torch.ones_like(fields.common_gate),
                    ),
                }
            )
    return {
        "target_state_count": len(rows),
        "rows_fingerprint": stable_fingerprint(rows),
        "all_target_states_exact_anchor": all(
            all(
                row[name] is True
                for name in (
                    "field_exact_anchor",
                    "residual_exact_zero",
                    "common_even_exact_zero",
                    "gate_exact_unit",
                )
            )
            for row in rows
        ),
    }


def _common_only_probe(
    model: CURELiteGatedCommonResidualPACRELevelSet,
    state: _CoordinateState,
    *,
    device: torch.device,
) -> dict[str, object]:
    feature = state.feature.to(device=device, dtype=torch.float32)
    occupancy = state.occupancy.to(device=device)
    feature_flat = feature.detach().abs().reshape(
        feature.shape[1],
        -1,
    )
    channel = int(
        torch.argmax(feature_flat.amax(dim=1)).detach().cpu()
    )
    center = model.config.coarse_radius
    with _exact_model_restore(model) as restoration:
        with torch.no_grad():
            model.joint_state_weight.zero_()
            model.joint_hidden_bias.zero_()
            coefficients = torch.linspace(
                0.25,
                1.25,
                model.config.width,
                device=device,
                dtype=torch.float32,
            )
            model.joint_state_weight[
                :,
                channel,
                center,
                center,
            ].copy_(coefficients)
            model.scalar_energy_weight.copy_(
                torch.linspace(
                    0.5,
                    1.5,
                    model.config.width,
                    device=device,
                    dtype=torch.float32,
                )
            )
        with torch.no_grad():
            fields = model.forward_fields(feature, occupancy)
            validate_gcr_pacre_fields(
                model,
                fields,
                feature=feature,
                occupancy=occupancy,
            )
            oracle = model.forward_reference_fields_fp64(
                feature,
                occupancy,
            )
            comparison = _comparison_payload(
                fields.field,
                oracle.field,
            )
            body = {
                "state_id": state.state_id,
                "selected_feature_channel": channel,
                "parameter_policy": (
                    "zero_occupancy_weights_single_feature_center_"
                    "deterministic_readout_temporary_v1"
                ),
                "residual_element_count": (
                    fields.residual_odd_interaction.numel()
                ),
                "residual_exact_zero_count": int(
                    torch.count_nonzero(
                        fields.residual_odd_interaction == 0.0
                    )
                ),
                "common_even_nonzero_count": int(
                    torch.count_nonzero(
                        fields.common_even_energy != 0.0
                    )
                ),
                "gate_nonunit_count": int(
                    torch.count_nonzero(fields.common_gate != 1.0)
                ),
                "gated_interaction_exact_zero_count": int(
                    torch.count_nonzero(
                        fields.gated_interaction == 0.0
                    )
                ),
                "field_exact_anchor": torch.equal(
                    fields.field,
                    torch.full_like(
                        fields.field,
                        model.config.field_amplitude,
                    ),
                ),
                "fast_fp64_field_comparison": comparison,
                "field_fingerprint": tensor_content_fingerprint(
                    fields.field.detach().to("cpu")
                ),
            }
    return {
        **body,
        "model_restoration": restoration,
    }


@dataclass(frozen=True)
class _StateRuntimeEvidence:
    state: _CoordinateState
    row: dict[str, object]
    residual_latent: Tensor
    gated_latent: Tensor


def _target_path_probe(
    model: CURELiteGatedCommonResidualPACRELevelSet,
    fields: CoverageStateGCRPACREFields,
    target_mask: Tensor,
) -> dict[str, object]:
    if not bool(torch.any(target_mask)):
        raise ValueError("target path probe needs a nonempty target mask")
    parameters = dict(model.named_parameters())
    residual_only_native = (
        fields.common_gate.detach()
        * fields.residual_odd_interaction
    )
    gate_only_native = (
        fields.common_gate
        * fields.residual_odd_interaction.detach()
    )
    residual_only_field = model.pixel_shuffle(residual_only_native)
    gate_only_field = model.pixel_shuffle(gate_only_native)
    residual_objective = residual_only_field[target_mask].mean()
    gate_objective = gate_only_field[target_mask].mean()
    residual_tuple = torch.autograd.grad(
        residual_objective,
        tuple(parameters.values()),
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    gate_tuple = torch.autograd.grad(
        gate_objective,
        tuple(parameters.values()),
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    residual = _gradient_summary(
        dict(zip(parameters, residual_tuple, strict=True))
    )
    gate = _gradient_summary(
        dict(zip(parameters, gate_tuple, strict=True))
    )

    def derivative(rows: Mapping[str, object], *, name: str) -> str:
        square_total = 0.0
        for row in rows.values():
            if not isinstance(row, Mapping):
                raise TypeError(f"{name} gradient row is invalid")
            norm = float.fromhex(str(row["l2_norm_hex"]))
            square_total += norm * norm
        return _finite_hex(
            -square_total,
            name=f"{name} descent directional derivative",
        )

    return {
        "policy": GCR_PACRE_DR_GATE_GRADIENT_POLICY,
        "target_coordinate_count": int(
            torch.count_nonzero(target_mask)
        ),
        "residual_only_objective_hex": _finite_hex(
            float(residual_objective.detach().cpu()),
            name="target residual-only objective",
        ),
        "gate_only_objective_hex": _finite_hex(
            float(gate_objective.detach().cpu()),
            name="target gate-only objective",
        ),
        "residual_only_gradient_rows": residual,
        "gate_only_gradient_rows": gate,
        "residual_descent_directional_derivative_hex": derivative(
            residual,
            name="residual-only",
        ),
        "gate_descent_directional_derivative_hex": derivative(
            gate,
            name="gate-only",
        ),
        "parameter_grad_buffers_unretained": all(
            parameter.grad is None for parameter in model.parameters()
        ),
    }


@torch.no_grad()
def _target_flip_parity_probe(
    model: CURELiteGatedCommonResidualPACRELevelSet,
    state: _CoordinateState,
    original_oracle: object,
    *,
    feature: Tensor,
    occupancy: Tensor,
    device: torch.device,
) -> dict[str, object]:
    coordinates = torch.nonzero(
        state.target_mask[:, 0],
        as_tuple=False,
    )
    if coordinates.shape[0] < 1:
        raise ValueError("target flip parity needs a target coordinate")
    batch, row, column = (
        int(value) for value in coordinates[0].tolist()
    )
    toggled = occupancy.clone()
    toggled[batch, 0, row, column] = ~toggled[
        batch,
        0,
        row,
        column,
    ]
    flipped_fields = model.forward_fields(feature, toggled)
    validate_gcr_pacre_fields(
        model,
        flipped_fields,
        feature=feature,
        occupancy=toggled,
    )
    flipped_oracle = model.forward_reference_fields_fp64(
        feature,
        toggled,
    )
    stride = model.config.feature_stride
    phase = (row % stride) * stride + column % stride
    coarse_row = row // stride
    coarse_column = column // stride

    def scalar(value: object, field: str) -> Tensor:
        tensor = getattr(value, field)
        return tensor[batch, phase, coarse_row, coarse_column]

    original_values = {
        name: scalar(original_oracle, name)
        for name in (
            "residual_odd_interaction",
            "common_even_energy",
            "common_gate",
            "gated_interaction",
        )
    }
    flipped_values = {
        name: scalar(flipped_oracle, name)
        for name in original_values
    }
    comparisons = {
        "residual_odd_interaction": _comparison_payload(
            flipped_fields.residual_odd_interaction,
            flipped_oracle.residual_odd_interaction,
        ),
        "common_even_energy": _comparison_payload(
            flipped_fields.common_even_energy,
            flipped_oracle.common_even_energy,
        ),
        "common_gate": _comparison_payload(
            flipped_fields.common_gate,
            flipped_oracle.common_gate,
        ),
        "gated_interaction": _comparison_payload(
            flipped_fields.gated_interaction,
            flipped_oracle.gated_interaction,
        ),
        "native_phase_field": _comparison_payload(
            flipped_fields.native_phase_field,
            flipped_oracle.native_phase_field,
        ),
        "field": _comparison_payload(
            flipped_fields.field,
            flipped_oracle.field,
        ),
    }
    return {
        "coordinate": [batch, row, column],
        "phase_coordinate": [
            batch,
            phase,
            coarse_row,
            coarse_column,
        ],
        "occupancy_toggled_exactly_once": int(
            torch.count_nonzero(toggled != occupancy)
        )
        == 1,
        "fp64_residual_exact_odd": torch.equal(
            flipped_values["residual_odd_interaction"],
            -original_values["residual_odd_interaction"],
        ),
        "fp64_common_even_exact_even": torch.equal(
            flipped_values["common_even_energy"],
            original_values["common_even_energy"],
        ),
        "fp64_gate_exact_even": torch.equal(
            flipped_values["common_gate"],
            original_values["common_gate"],
        ),
        "fp64_gated_interaction_exact_odd": torch.equal(
            flipped_values["gated_interaction"],
            -original_values["gated_interaction"],
        ),
        "original_values_hex": {
            name: _finite_hex(
                float(value.detach().cpu()),
                name=f"original parity {name}",
            )
            for name, value in original_values.items()
        },
        "flipped_values_hex": {
            name: _finite_hex(
                float(value.detach().cpu()),
                name=f"flipped parity {name}",
            )
            for name, value in flipped_values.items()
        },
        "flipped_fast_fp64_comparisons": comparisons,
        "flipped_complete_fields_ledger_fingerprint": (
            _fields_ledger_fingerprint(flipped_fields)
        ),
    }


def _state_forward_observation(
    model: CURELiteGatedCommonResidualPACRELevelSet,
    state: _CoordinateState,
    *,
    scope: str,
    device: torch.device,
) -> _StateRuntimeEvidence:
    if scope == "context" and torch.is_grad_enabled():
        with torch.no_grad():
            return _state_forward_observation(
                model,
                state,
                scope=scope,
                device=device,
            )
    if scope == "target" and not torch.is_grad_enabled():
        with torch.enable_grad():
            return _state_forward_observation(
                model,
                state,
                scope=scope,
                device=device,
            )
    feature = state.feature.to(device=device, dtype=torch.float32)
    occupancy = state.occupancy.to(device=device)
    fields = model.forward_fields(feature, occupancy)
    validate_gcr_pacre_fields(
        model,
        fields,
        feature=feature,
        occupancy=occupancy,
    )
    oracle = model.forward_reference_fields_fp64(feature, occupancy)
    comparisons = {
        "residual_odd_interaction": _comparison_payload(
            fields.residual_odd_interaction,
            oracle.residual_odd_interaction,
        ),
        "common_even_energy": _comparison_payload(
            fields.common_even_energy,
            oracle.common_even_energy,
        ),
        "common_gate": _comparison_payload(
            fields.common_gate,
            oracle.common_gate,
        ),
        "gated_interaction": _comparison_payload(
            fields.gated_interaction,
            oracle.gated_interaction,
        ),
        "native_phase_field": _comparison_payload(
            fields.native_phase_field,
            oracle.native_phase_field,
        ),
        "field": _comparison_payload(fields.field, oracle.field),
    }
    float_tensors = {
        name: getattr(fields, name)
        for name in fields.__dataclass_fields__
        if isinstance(getattr(fields, name), Tensor)
        and getattr(fields, name).is_floating_point()
    }
    finite_rows = {
        name: bool(torch.isfinite(value).all())
        for name, value in float_tensors.items()
    }
    gate_statistics = asdict(
        summarize_gcr_pacre_gate_saturation(fields)
    )
    residual_hidden = 0.5 * (
        fields.actual_residual_hidden
        - fields.flipped_residual_hidden
    )
    gated_hidden = (
        fields.common_gate.unsqueeze(2) * residual_hidden
    )
    residual_latent = _phase_hidden_to_output(
        residual_hidden,
        stride=model.config.feature_stride,
    ).detach().to("cpu").contiguous()
    gated_latent = _phase_hidden_to_output(
        gated_hidden,
        stride=model.config.feature_stride,
    ).detach().to("cpu").contiguous()
    target_mask = state.target_mask.to(device=device)
    target_path = (
        _target_path_probe(model, fields, target_mask)
        if scope == "target"
        else None
    )
    flip_parity = (
        _target_flip_parity_probe(
            model,
            state,
            oracle,
            feature=feature,
            occupancy=occupancy,
            device=device,
        )
        if scope == "target"
        else None
    )
    row = {
        "scoped_state_id": f"{scope}::{state.state_id}",
        "source_state_id": state.state_id,
        "scope": scope,
        "sample_id": state.sample_id,
        "state_kind": state.state_kind,
        "endpoint": state.endpoint,
        "target_group_id": state.target_group_id,
        "feature_fingerprint": tensor_content_fingerprint(
            state.feature
        ),
        "occupancy_fingerprint": tensor_content_fingerprint(
            state.occupancy
        ),
        "target_mask_fingerprint": tensor_content_fingerprint(
            state.target_mask
        ),
        "background_mask_fingerprint": tensor_content_fingerprint(
            state.background_mask
        ),
        "component_mask_fingerprint": tensor_content_fingerprint(
            state.component_mask
        ),
        "fields_fqcn": _fqcn(fields),
        "complete_fields_validator_called": True,
        "complete_fields_ledger_fingerprint": (
            _fields_ledger_fingerprint(fields)
        ),
        "float_ledger_tensor_count": len(finite_rows),
        "float_ledger_finite": finite_rows,
        "all_float_ledger_tensors_finite": all(finite_rows.values()),
        "fast_fp64_comparisons": comparisons,
        "fast_fp64_ledger_agrees": _comparison_ledger_agrees(
            comparisons
        ),
        "common_nonzero_counts": {
            "actual_common_hidden": int(
                torch.count_nonzero(fields.actual_common_hidden)
            ),
            "flipped_common_hidden": int(
                torch.count_nonzero(fields.flipped_common_hidden)
            ),
            "actual_common_energy": int(
                torch.count_nonzero(fields.actual_common_energy)
            ),
            "flipped_common_energy": int(
                torch.count_nonzero(fields.flipped_common_energy)
            ),
            "common_even_energy": int(
                torch.count_nonzero(fields.common_even_energy)
            ),
            "common_gate_nonunit": int(
                torch.count_nonzero(fields.common_gate != 1.0)
            ),
        },
        "mechanism_nonzero_counts": {
            "residual_odd_interaction": int(
                torch.count_nonzero(
                    fields.residual_odd_interaction
                )
            ),
            "gated_interaction": int(
                torch.count_nonzero(fields.gated_interaction)
            ),
            "field_not_anchor": int(
                torch.count_nonzero(
                    fields.field != model.config.field_amplitude
                )
            ),
        },
        "gate_statistics": gate_statistics,
        "gate_tensor_fingerprint": tensor_content_fingerprint(
            fields.common_gate.detach().to("cpu")
        ),
        "residual_latent_fingerprint": tensor_content_fingerprint(
            residual_latent
        ),
        "gated_latent_fingerprint": tensor_content_fingerprint(
            gated_latent
        ),
        "target_path": target_path,
        "flip_parity": flip_parity,
    }
    row["state_observation_fingerprint"] = stable_fingerprint(row)
    return _StateRuntimeEvidence(
        state=state,
        row=row,
        residual_latent=residual_latent,
        gated_latent=gated_latent,
    )


def _bound_target_witness(
    evidence: _StateRuntimeEvidence,
    *,
    stride: int,
) -> dict[str, object]:
    state = evidence.state
    target_coordinates = torch.nonzero(
        state.target_mask[:, 0],
        as_tuple=False,
    ).to("cpu")
    background = state.background_mask.to("cpu")
    height, width = background.shape[-2:]
    selected: dict[str, object] | None = None
    legal_pair_count = 0
    exact_separated_count = 0
    residual_nonzero_target_count = 0
    binding_rows: list[dict[str, object]] = []
    for coordinate in target_coordinates:
        batch, row, column = (
            int(value) for value in coordinate.tolist()
        )
        target_residual = evidence.residual_latent[
            batch,
            :,
            row,
            column,
        ]
        target_gated = evidence.gated_latent[
            batch,
            :,
            row,
            column,
        ]
        residual_nonzero = bool(torch.any(target_residual != 0.0))
        residual_nonzero_target_count += int(residual_nonzero)
        for row_delta in (-1, 0, 1):
            for column_delta in (-1, 0, 1):
                if row_delta == 0 and column_delta == 0:
                    continue
                other_row = row + row_delta
                other_column = column + column_delta
                if (
                    not 0 <= other_row < height
                    or not 0 <= other_column < width
                    or not bool(
                        background[
                            batch,
                            0,
                            other_row,
                            other_column,
                        ]
                    )
                    or row // stride != other_row // stride
                    or column // stride != other_column // stride
                ):
                    continue
                legal_pair_count += 1
                background_gated = evidence.gated_latent[
                    batch,
                    :,
                    other_row,
                    other_column,
                ]
                exactly_separated = not torch.equal(
                    torch.where(
                        target_gated == 0.0,
                        torch.zeros_like(target_gated),
                        target_gated,
                    ),
                    torch.where(
                        background_gated == 0.0,
                        torch.zeros_like(background_gated),
                        background_gated,
                    ),
                )
                exact_separated_count += int(exactly_separated)
                raw = {
                    "target_coordinate": [batch, row, column],
                    "background_coordinate": [
                        batch,
                        other_row,
                        other_column,
                    ],
                    "coarse_cell": [
                        batch,
                        row // stride,
                        column // stride,
                    ],
                    "target_residual_nonzero": residual_nonzero,
                    "target_background_gated_latent_exactly_distinct": (
                        exactly_separated
                    ),
                    "target_residual_fingerprint": (
                        tensor_content_fingerprint(
                            target_residual.contiguous()
                        )
                    ),
                    "target_gated_fingerprint": (
                        tensor_content_fingerprint(
                            target_gated.contiguous()
                        )
                    ),
                    "background_gated_fingerprint": (
                        tensor_content_fingerprint(
                            background_gated.contiguous()
                        )
                    ),
                }
                binding_rows.append(raw)
                if residual_nonzero and exactly_separated and selected is None:
                    selected = raw
    target_path = evidence.row.get("target_path")
    residual_rows = (
        target_path.get("residual_only_gradient_rows")
        if isinstance(target_path, Mapping)
        else None
    )
    gate_rows = (
        target_path.get("gate_only_gradient_rows")
        if isinstance(target_path, Mapping)
        else None
    )
    path_valid = (
        isinstance(target_path, Mapping)
        and _all_gradient_rows_finite(
            residual_rows,
            require_any_nonzero=True,
        )
        and _all_gradient_rows_finite(
            gate_rows,
            require_any_nonzero=True,
        )
        and float.fromhex(
            str(
                target_path[
                    "residual_descent_directional_derivative_hex"
                ]
            )
        )
        < 0.0
        and float.fromhex(
            str(
                target_path[
                    "gate_descent_directional_derivative_hex"
                ]
            )
        )
        < 0.0
        and target_path.get("parameter_grad_buffers_unretained") is True
    )
    return {
        "target_group_id": state.target_group_id,
        "state_id": state.state_id,
        "sample_id": state.sample_id,
        "target_coordinate_count": int(target_coordinates.shape[0]),
        "residual_nonzero_target_count": residual_nonzero_target_count,
        "legal_same_cell_background_pair_count": legal_pair_count,
        "exactly_separated_pair_count": exact_separated_count,
        "selected_first_exact_witness": selected,
        "binding_fingerprint": stable_fingerprint(binding_rows),
        "no_numeric_separation_threshold": True,
        "residual_and_gate_paths_finite_nonzero": path_valid,
        "witness_passed": selected is not None and path_valid,
    }


def _collision_probe(
    target_rows: list[_StateRuntimeEvidence],
    context_rows: list[_StateRuntimeEvidence],
    *,
    attribute: str,
) -> dict[str, object]:
    if attribute not in {"residual_latent", "gated_latent"}:
        raise ValueError("unknown collision representation")
    targets_by_hash: dict[int, list[dict[str, object]]] = {}
    target_coordinate_count = 0
    for evidence in target_rows:
        value = getattr(evidence, attribute)
        vectors = _vectors_at(value, evidence.state.target_mask)
        coordinates = torch.nonzero(
            evidence.state.target_mask[:, 0],
            as_tuple=False,
        ).to("cpu")
        canonical = torch.where(
            vectors == 0.0,
            torch.zeros_like(vectors),
            vectors,
        ).contiguous()
        hashes = _row_bit_hash(canonical)
        target_coordinate_count += int(vectors.shape[0])
        for index in range(vectors.shape[0]):
            key = int(hashes[index])
            targets_by_hash.setdefault(key, []).append(
                {
                    "state_id": evidence.state.state_id,
                    "sample_id": evidence.state.sample_id,
                    "target_group_id": evidence.state.target_group_id,
                    "coordinate": coordinates[index].tolist(),
                    "vector": canonical[index],
                }
            )
    if not targets_by_hash:
        raise RuntimeError("collision probe has no target vectors")
    target_hashes = torch.tensor(
        sorted(targets_by_hash),
        dtype=torch.int64,
    )
    collisions = 0
    examples: list[dict[str, object]] = []
    role_counts = {"background": 0, "writable_component": 0}
    role_stream_rows: list[dict[str, object]] = []
    for evidence in context_rows:
        value = getattr(evidence, attribute)
        roles = [("background", evidence.state.background_mask)]
        if evidence.state.component_writable:
            roles.append(
                (
                    "writable_component",
                    evidence.state.component_mask,
                )
            )
        for role, mask in roles:
            if not bool(torch.any(mask)):
                continue
            vectors = _vectors_at(value, mask)
            canonical = torch.where(
                vectors == 0.0,
                torch.zeros_like(vectors),
                vectors,
            ).contiguous()
            coordinates = torch.nonzero(
                mask[:, 0],
                as_tuple=False,
            ).to("cpu")
            role_counts[role] += int(vectors.shape[0])
            role_stream_rows.append(
                {
                    "state_id": evidence.state.state_id,
                    "role": role,
                    "coordinate_count": int(vectors.shape[0]),
                    "vectors_fingerprint": tensor_content_fingerprint(
                        canonical
                    ),
                }
            )
            hashes = _row_bit_hash(canonical)
            candidates = torch.nonzero(
                torch.isin(hashes, target_hashes),
                as_tuple=False,
            ).flatten()
            for raw_index in candidates:
                index = int(raw_index)
                key = int(hashes[index])
                for target in targets_by_hash.get(key, ()):
                    target_vector = target["vector"]
                    if not isinstance(target_vector, Tensor):
                        raise TypeError("target collision vector changed")
                    if not torch.equal(canonical[index], target_vector):
                        continue
                    collisions += 1
                    if len(examples) < 64:
                        examples.append(
                            {
                                "representation": attribute,
                                "vector_fingerprint": (
                                    tensor_content_fingerprint(
                                        canonical[index]
                                    )
                                ),
                                "negative_requirement": {
                                    name: value
                                    for name, value in target.items()
                                    if name != "vector"
                                },
                                "positive_requirement": {
                                    "state_id": evidence.state.state_id,
                                    "sample_id": evidence.state.sample_id,
                                    "state_kind": (
                                        evidence.state.state_kind
                                    ),
                                    "endpoint": evidence.state.endpoint,
                                    "role": role,
                                    "coordinate": (
                                        coordinates[index].tolist()
                                    ),
                                },
                            }
                        )
    return {
        "representation": attribute,
        "collision_policy": GCR_PACRE_DR_COLLISION_POLICY,
        "target_coordinate_count": target_coordinate_count,
        "target_hash_bucket_count": len(targets_by_hash),
        "positive_role_coordinate_counts": role_counts,
        "positive_role_stream_fingerprint": stable_fingerprint(
            role_stream_rows
        ),
        "exact_collision_count": collisions,
        "exact_collision_examples": examples,
        "collision_examples_truncated": collisions > len(examples),
    }


class _StreamingCollisionAccumulator:
    """Exact collision scan retaining only target vectors and raw summaries."""

    def __init__(self, attribute: str) -> None:
        if attribute not in {"residual_latent", "gated_latent"}:
            raise ValueError("unknown collision representation")
        self.attribute = attribute
        self.targets_by_hash: dict[
            int,
            list[dict[str, object]],
        ] = {}
        self.target_coordinate_count = 0
        self.collisions = 0
        self.examples: list[dict[str, object]] = []
        self.role_counts = {
            "background": 0,
            "writable_component": 0,
        }
        self.role_stream_rows: list[dict[str, object]] = []
        self._target_hashes: Tensor | None = None

    def add_target(self, evidence: _StateRuntimeEvidence) -> None:
        if self._target_hashes is not None:
            raise RuntimeError("target collision index is already sealed")
        value = getattr(evidence, self.attribute)
        vectors = _vectors_at(value, evidence.state.target_mask)
        coordinates = torch.nonzero(
            evidence.state.target_mask[:, 0],
            as_tuple=False,
        ).to("cpu")
        canonical = torch.where(
            vectors == 0.0,
            torch.zeros_like(vectors),
            vectors,
        ).contiguous()
        hashes = _row_bit_hash(canonical)
        self.target_coordinate_count += int(vectors.shape[0])
        for index in range(vectors.shape[0]):
            key = int(hashes[index])
            self.targets_by_hash.setdefault(key, []).append(
                {
                    "state_id": evidence.state.state_id,
                    "sample_id": evidence.state.sample_id,
                    "target_group_id": evidence.state.target_group_id,
                    "coordinate": coordinates[index].tolist(),
                    "vector": canonical[index].clone(),
                }
            )

    def seal_targets(self) -> None:
        if not self.targets_by_hash:
            raise RuntimeError("collision probe has no target vectors")
        self._target_hashes = torch.tensor(
            sorted(self.targets_by_hash),
            dtype=torch.int64,
        )

    def scan_context(self, evidence: _StateRuntimeEvidence) -> None:
        if self._target_hashes is None:
            raise RuntimeError("target collision index is not sealed")
        value = getattr(evidence, self.attribute)
        roles = [("background", evidence.state.background_mask)]
        if evidence.state.component_writable:
            roles.append(
                (
                    "writable_component",
                    evidence.state.component_mask,
                )
            )
        for role, mask in roles:
            if not bool(torch.any(mask)):
                continue
            vectors = _vectors_at(value, mask)
            canonical = torch.where(
                vectors == 0.0,
                torch.zeros_like(vectors),
                vectors,
            ).contiguous()
            coordinates = torch.nonzero(
                mask[:, 0],
                as_tuple=False,
            ).to("cpu")
            self.role_counts[role] += int(vectors.shape[0])
            self.role_stream_rows.append(
                {
                    "state_id": evidence.state.state_id,
                    "role": role,
                    "coordinate_count": int(vectors.shape[0]),
                    "vectors_fingerprint": tensor_content_fingerprint(
                        canonical
                    ),
                }
            )
            hashes = _row_bit_hash(canonical)
            candidates = torch.nonzero(
                torch.isin(hashes, self._target_hashes),
                as_tuple=False,
            ).flatten()
            for raw_index in candidates:
                index = int(raw_index)
                key = int(hashes[index])
                for target in self.targets_by_hash.get(key, ()):
                    target_vector = target["vector"]
                    if not isinstance(target_vector, Tensor):
                        raise TypeError("target collision vector changed")
                    if not torch.equal(canonical[index], target_vector):
                        continue
                    self.collisions += 1
                    if len(self.examples) < 64:
                        self.examples.append(
                            {
                                "representation": self.attribute,
                                "vector_fingerprint": (
                                    tensor_content_fingerprint(
                                        canonical[index]
                                    )
                                ),
                                "negative_requirement": {
                                    name: value
                                    for name, value in target.items()
                                    if name != "vector"
                                },
                                "positive_requirement": {
                                    "state_id": evidence.state.state_id,
                                    "sample_id": evidence.state.sample_id,
                                    "state_kind": (
                                        evidence.state.state_kind
                                    ),
                                    "endpoint": evidence.state.endpoint,
                                    "role": role,
                                    "coordinate": (
                                        coordinates[index].tolist()
                                    ),
                                },
                            }
                        )

    def payload(self) -> dict[str, object]:
        if self._target_hashes is None:
            raise RuntimeError("collision accumulator is incomplete")
        return {
            "representation": self.attribute,
            "collision_policy": GCR_PACRE_DR_COLLISION_POLICY,
            "target_coordinate_count": self.target_coordinate_count,
            "target_hash_bucket_count": len(self.targets_by_hash),
            "positive_role_coordinate_counts": dict(self.role_counts),
            "positive_role_stream_fingerprint": stable_fingerprint(
                self.role_stream_rows
            ),
            "exact_collision_count": self.collisions,
            "exact_collision_examples": self.examples,
            "collision_examples_truncated": (
                self.collisions > len(self.examples)
            ),
            "streaming_context_state_retention": 1,
        }


def _ledger_probe(
    model: CURELiteGatedCommonResidualPACRELevelSet,
    adapter: _PopulationAdapter,
    *,
    device: torch.device,
) -> dict[str, object]:
    residual_accumulator = _StreamingCollisionAccumulator(
        "residual_latent"
    )
    gated_accumulator = _StreamingCollisionAccumulator("gated_latent")
    target_rows: list[dict[str, object]] = []
    witnesses: list[dict[str, object]] = []
    for state in adapter.target_states:
        evidence = _state_forward_observation(
            model,
            state,
            scope="target",
            device=device,
        )
        target_rows.append(evidence.row)
        witnesses.append(
            _bound_target_witness(
                evidence,
                stride=adapter.feature_stride,
            )
        )
        residual_accumulator.add_target(evidence)
        gated_accumulator.add_target(evidence)
        del evidence
    residual_accumulator.seal_targets()
    gated_accumulator.seal_targets()

    context_rows: list[dict[str, object]] = []
    for state in adapter.context_states:
        evidence = _state_forward_observation(
            model,
            state,
            scope="context",
            device=device,
        )
        context_rows.append(evidence.row)
        residual_accumulator.scan_context(evidence)
        gated_accumulator.scan_context(evidence)
        del evidence
    residual_collisions = residual_accumulator.payload()
    gated_collisions = gated_accumulator.payload()
    all_rows = target_rows + context_rows
    target_ids = [str(row["scoped_state_id"]) for row in target_rows]
    context_ids = [str(row["scoped_state_id"]) for row in context_rows]
    target_group_ids = [
        str(row["target_group_id"]) for row in target_rows
    ]
    common_counts = {
        name: sum(
            int(row["common_nonzero_counts"][name])
            for row in all_rows
        )
        for name in (
            "actual_common_hidden",
            "flipped_common_hidden",
            "actual_common_energy",
            "flipped_common_energy",
            "common_even_energy",
            "common_gate_nonunit",
        )
    }
    gate_rows = [
        {
            "scoped_state_id": row["scoped_state_id"],
            "gate_statistics": row["gate_statistics"],
            "gate_tensor_fingerprint": row[
                "gate_tensor_fingerprint"
            ],
        }
        for row in all_rows
    ]
    gate_element_count = sum(
        int(row["gate_statistics"]["element_count"])
        for row in all_rows
    )
    gate_zero_count = sum(
        int(row["gate_statistics"]["zero_count"]) for row in all_rows
    )
    gate_two_count = sum(
        int(row["gate_statistics"]["two_count"]) for row in all_rows
    )
    gate_interior_count = sum(
        int(row["gate_statistics"]["interior_count"])
        for row in all_rows
    )
    gate_minimum = min(
        float(row["gate_statistics"]["minimum"]) for row in all_rows
    )
    gate_maximum = max(
        float(row["gate_statistics"]["maximum"]) for row in all_rows
    )
    weighted_mean = sum(
        float(row["gate_statistics"]["mean"])
        * int(row["gate_statistics"]["element_count"])
        for row in all_rows
    ) / gate_element_count
    gate_distribution = {
        "threshold_or_ratio_gate": None,
        "state_count": len(all_rows),
        "element_count": gate_element_count,
        "zero_count": gate_zero_count,
        "two_count": gate_two_count,
        "interior_count": gate_interior_count,
        "minimum_hex": _finite_hex(
            gate_minimum,
            name="aggregate gate minimum",
        ),
        "maximum_hex": _finite_hex(
            gate_maximum,
            name="aggregate gate maximum",
        ),
        "mean_hex": _finite_hex(
            weighted_mean,
            name="aggregate gate mean",
        ),
        "per_state_rows_fingerprint": stable_fingerprint(gate_rows),
        "interpretation": (
            "descriptive_endpoint_and_interior_distribution_only"
        ),
    }
    target_summary = {
        "scope": "target",
        "expected_state_count": GCR_PACRE_DR_TARGET_STATE_COUNT,
        "observed_state_count": len(target_rows),
        "scoped_state_ids": target_ids,
        "scoped_state_ids_unique": len(target_ids) == len(set(target_ids)),
        "target_group_ids": target_group_ids,
        "target_group_ids_unique": (
            len(target_group_ids) == len(set(target_group_ids))
        ),
        "all_complete_fields_validated": all(
            row["complete_fields_validator_called"] is True
            for row in target_rows
        ),
        "all_float_ledger_tensors_finite": all(
            row["all_float_ledger_tensors_finite"] is True
            for row in target_rows
        ),
        "all_fast_fp64_ledgers_agree": all(
            row["fast_fp64_ledger_agrees"] is True
            for row in target_rows
        ),
    }
    context_summary = {
        "scope": "context",
        "expected_state_count": GCR_PACRE_DR_CONTEXT_STATE_COUNT,
        "observed_state_count": len(context_rows),
        "scoped_state_ids": context_ids,
        "scoped_state_ids_unique": (
            len(context_ids) == len(set(context_ids))
        ),
        "all_complete_fields_validated": all(
            row["complete_fields_validator_called"] is True
            for row in context_rows
        ),
        "all_float_ledger_tensors_finite": all(
            row["all_float_ledger_tensors_finite"] is True
            for row in context_rows
        ),
        "all_fast_fp64_ledgers_agree": all(
            row["fast_fp64_ledger_agrees"] is True
            for row in context_rows
        ),
    }
    body: dict[str, object] = {
        "schema_version": (
            "cure-lite-v24-gcr-pacre-complete-state-ledger-v1"
        ),
        "target_summary": target_summary,
        "context_summary": context_summary,
        "target_rows": target_rows,
        "context_rows": context_rows,
        "target_context_scoped_ids_disjoint": set(
            target_ids
        ).isdisjoint(context_ids),
        "union_scoped_state_count": len(
            set(target_ids) | set(context_ids)
        ),
        "expected_union_scoped_state_count": (
            GCR_PACRE_DR_TARGET_STATE_COUNT
            + GCR_PACRE_DR_CONTEXT_STATE_COUNT
        ),
        "target_witnesses": witnesses,
        "all_target_groups_have_exact_bound_witness": all(
            row["witness_passed"] is True for row in witnesses
        ),
        "residual_latent_collisions": residual_collisions,
        "gated_latent_collisions": gated_collisions,
        "common_nonzero_counts": common_counts,
        "all_target_flip_parity_exact": all(
            isinstance(row["flip_parity"], Mapping)
            and row["flip_parity"].get(
                "occupancy_toggled_exactly_once"
            )
            is True
            and row["flip_parity"].get(
                "fp64_residual_exact_odd"
            )
            is True
            and row["flip_parity"].get(
                "fp64_common_even_exact_even"
            )
            is True
            and row["flip_parity"].get(
                "fp64_gate_exact_even"
            )
            is True
            and row["flip_parity"].get(
                "fp64_gated_interaction_exact_odd"
            )
            is True
            and _comparison_ledger_agrees(
                row["flip_parity"][
                    "flipped_fast_fp64_comparisons"
                ]
            )
            for row in target_rows
        ),
        "gate_distribution": gate_distribution,
        "fixed_readout_mechanism_nonzero": any(
            int(row["mechanism_nonzero_counts"]["gated_interaction"])
            > 0
            for row in target_rows
        ),
    }
    return {**body, "ledger_fingerprint": stable_fingerprint(body)}


def _efficiency_observation(
    dataset_free_receipt: Mapping[str, object],
) -> dict[str, object]:
    evidence = dataset_free_receipt.get("evidence")
    efficiency = (
        evidence.get("efficiency")
        if isinstance(evidence, Mapping)
        else None
    )
    if not isinstance(efficiency, Mapping):
        raise TypeError("dataset-free efficiency evidence is absent")
    payload = dict(efficiency)
    section_fingerprint = payload.pop("section_fingerprint", None)
    if (
        not _is_sha256(section_fingerprint)
        or section_fingerprint != stable_fingerprint(payload)
    ):
        raise ValueError("efficiency section fingerprint changed")
    conditions = efficiency.get("common_conditions")
    arms = efficiency.get("arms")
    inventory = efficiency.get("additional_op_inventory")
    if (
        not isinstance(conditions, Mapping)
        or not isinstance(arms, Mapping)
        or set(arms) != {"PACRE_VC_v23", "GCR_PACRE_v24"}
        or not isinstance(inventory, Mapping)
        or conditions.get("formal_model_config")
        != {
            "feature_channels": GCR_PACRE_FORMAL_FEATURE_CHANNELS,
            "feature_stride": GCR_PACRE_FORMAL_FEATURE_STRIDE,
            "width": GCR_PACRE_FORMAL_WIDTH,
            "parameter_count": GCR_PACRE_FORMAL_PARAMETER_COUNT,
        }
        or conditions.get("threshold_or_ratio_gate") is not None
        or efficiency.get("interpretation")
        != "measurement_only_no_post_hoc_lite_overhead_threshold"
    ):
        raise ValueError("efficiency common conditions changed")
    arm_rows: dict[str, object] = {}
    common_device = conditions.get("device")
    common_dtype = conditions.get("dtype")
    common_output_shape: list[object] | None = None
    common_initial_fingerprint: str | None = None
    for arm_name in ("PACRE_VC_v23", "GCR_PACRE_v24"):
        raw = arms.get(arm_name)
        if not isinstance(raw, Mapping):
            raise TypeError(f"efficiency arm {arm_name} is absent")
        latencies: dict[str, object] = {}
        for name, repeat_name in (
            ("forward_latency", "forward_repeats"),
            ("train_step_latency", "train_step_repeats"),
        ):
            latency = raw.get(name)
            samples = (
                latency.get("samples_ns")
                if isinstance(latency, Mapping)
                else None
            )
            expected = conditions.get(repeat_name)
            if (
                not isinstance(latency, Mapping)
                or not isinstance(samples, list)
                or isinstance(expected, bool)
                or not isinstance(expected, int)
                or latency.get("sample_count") != expected
                or len(samples) != expected
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 1
                    for value in samples
                )
                or isinstance(latency.get("median_ns"), bool)
                or not isinstance(
                    latency.get("median_ns"),
                    (int, float),
                )
                or not isfinite(float(latency["median_ns"]))
                or float(latency["median_ns"]) <= 0.0
                or isinstance(latency.get("p95_ns"), bool)
                or not isinstance(latency.get("p95_ns"), (int, float))
                or not isfinite(float(latency["p95_ns"]))
                or float(latency["p95_ns"]) <= 0.0
            ):
                raise ValueError(
                    f"efficiency {arm_name} {name} is incomplete"
                )
            ordered = sorted(samples)
            median_expected = (
                float(ordered[len(ordered) // 2])
                if len(ordered) % 2 == 1
                else 0.5
                * (
                    float(ordered[len(ordered) // 2 - 1])
                    + float(ordered[len(ordered) // 2])
                )
            )
            p95_index = max(
                0,
                (95 * len(ordered) + 99) // 100 - 1,
            )
            if (
                float(latency["median_ns"]) != median_expected
                or float(latency["p95_ns"])
                != float(ordered[p95_index])
            ):
                raise ValueError(
                    f"efficiency {arm_name} latency summary changed"
                )
            latencies[name] = {
                "sample_count": expected,
                "samples_fingerprint": stable_fingerprint(samples),
                "median_ns_hex": float(
                    latency["median_ns"]
                ).hex(),
                "p95_ns_hex": float(latency["p95_ns"]).hex(),
            }
        output_shape = raw.get("output_shape")
        initial_fingerprint = raw.get(
            "initial_parameter_fingerprint"
        )
        if (
            raw.get("arm") != arm_name
            or raw.get("device") != common_device
            or raw.get("dtype") != common_dtype
            or raw.get("oom") is not False
            or raw.get("nonfinite") is not False
            or raw.get("model_config")
            != {
                "feature_channels": GCR_PACRE_FORMAL_FEATURE_CHANNELS,
                "feature_stride": GCR_PACRE_FORMAL_FEATURE_STRIDE,
                "width": GCR_PACRE_FORMAL_WIDTH,
            }
            or raw.get("parameter_count")
            != GCR_PACRE_FORMAL_PARAMETER_COUNT
            or raw.get("parameter_tensor_count") != 3
            or raw.get("forward_flop_counter_supported") is not True
            or isinstance(raw.get("forward_flops"), bool)
            or not isinstance(raw.get("forward_flops"), int)
            or int(raw["forward_flops"]) < 1
            or isinstance(raw.get("checkpoint_bytes"), bool)
            or not isinstance(raw.get("checkpoint_bytes"), int)
            or int(raw["checkpoint_bytes"]) < 1
            or isinstance(raw.get("parameter_bytes"), bool)
            or not isinstance(raw.get("parameter_bytes"), int)
            or int(raw["parameter_bytes"]) < 1
            or isinstance(raw.get("field_tensor_bytes"), bool)
            or not isinstance(raw.get("field_tensor_bytes"), int)
            or int(raw["field_tensor_bytes"]) < 1
            or not isinstance(output_shape, list)
            or not output_shape
            or not _is_sha256(initial_fingerprint)
            or raw.get("forward_warmups")
            != conditions.get("forward_warmups")
            or raw.get("forward_repeats")
            != conditions.get("forward_repeats")
            or raw.get("train_step_warmups")
            != conditions.get("train_step_warmups")
            or raw.get("train_step_repeats")
            != conditions.get("train_step_repeats")
        ):
            raise ValueError(f"efficiency arm {arm_name} changed")
        if common_output_shape is None:
            common_output_shape = list(output_shape)
            common_initial_fingerprint = str(initial_fingerprint)
        elif (
            output_shape != common_output_shape
            or initial_fingerprint != common_initial_fingerprint
        ):
            raise ValueError("efficiency arms lack common conditions")
        arm_rows[arm_name] = {
            "oom": raw["oom"],
            "nonfinite": raw["nonfinite"],
            "forward_flops": raw["forward_flops"],
            "parameter_count": raw["parameter_count"],
            "parameter_bytes": raw["parameter_bytes"],
            "checkpoint_bytes": raw["checkpoint_bytes"],
            "field_tensor_bytes": raw["field_tensor_bytes"],
            "output_shape": output_shape,
            "initial_parameter_fingerprint": initial_fingerprint,
            "latencies": latencies,
        }
    observation: dict[str, object] = {
        "section_fingerprint": section_fingerprint,
        "efficiency_receipt_sha256": stable_fingerprint(
            dict(efficiency)
        ),
        "common_conditions_fingerprint": stable_fingerprint(
            dict(conditions)
        ),
        "device": common_device,
        "dtype": common_dtype,
        "formal_model_config": dict(
            conditions["formal_model_config"]
        ),
        "threshold_or_ratio_gate": conditions[
            "threshold_or_ratio_gate"
        ],
        "arms": arm_rows,
        "additional_op_inventory_fingerprint": stable_fingerprint(
            dict(inventory)
        ),
        "v24_additional_ops_recorded": bool(
            inventory.get("v24_additions_over_v23")
        ),
        "interpretation": efficiency["interpretation"],
        "both_arms_complete_finite_no_oom": all(
            isinstance(row, Mapping)
            and row.get("oom") is False
            and row.get("nonfinite") is False
            for row in arm_rows.values()
        ),
        "performance_ratio_or_absolute_gate": None,
    }
    return {
        **observation,
        "observation_fingerprint": stable_fingerprint(observation),
    }


_REAL_PREACCESS_ISSUER = object()
_REAL_RUN_START_ISSUER = object()
_GENERATED_AUDIT_ISSUER = object()
_CONSUMED_REAL_PREACCESS_TOKENS: set[int] = set()
_STARTED_REAL_GATE_TOKENS: set[int] = set()
_ISSUED_REAL_PREACCESS_TOKENS: dict[
    int,
    tuple["GCRPACREDRPreaccessToken", str],
] = {}
_ISSUED_REAL_RUN_START_TOKENS: dict[
    int,
    tuple["GCRPACREDRRunStartToken", str],
] = {}


@dataclass(frozen=True, slots=True)
class GCRPACREDRPreaccessToken:
    """Opaque one-use token issued only by the fixed verifier below."""

    dataset_free_receipt_fingerprint: str
    dataset_free_receipt_file_sha256: str
    efficiency_section_fingerprint: str
    efficiency_receipt_sha256: str
    authorization_fingerprint: str
    authorization_receipt_file_sha256: str
    access_audit_receipt_fingerprint: str
    access_audit_receipt_file_sha256: str
    protocol_preregistration_fingerprint: str
    source_closure_fingerprint: str
    expected_source_binding_fingerprint: str
    expected_manifest_file_sha256: str
    expected_state_index_file_sha256: str
    expected_real_inputs_fingerprint: str
    expected_population_fingerprint: str
    expected_cache_fingerprint: str
    efficiency_observation_json: str
    _issuer: object


@dataclass(frozen=True, slots=True)
class GCRPACREDRRunStartToken:
    """Opaque binding to one persistent, create-only run-start marker."""

    marker_path: str
    marker_fingerprint: str
    marker_file_sha256: str
    authorization_fingerprint: str
    access_audit_receipt_fingerprint: str
    source_closure_fingerprint: str
    requested_device: str
    requested_receipt_output: str
    _issuer: object


@dataclass(frozen=True, slots=True)
class _GeneratedAuditToken:
    adapter_fingerprint: str
    _issuer: object


def _preaccess_token_payload(
    token: GCRPACREDRPreaccessToken,
) -> dict[str, object]:
    return {
        name: getattr(token, name)
        for name in token.__dataclass_fields__
        if name != "_issuer"
    }


def _register_real_preaccess_token(
    token: GCRPACREDRPreaccessToken,
) -> GCRPACREDRPreaccessToken:
    token_id = id(token)
    if token_id in _ISSUED_REAL_PREACCESS_TOKENS:
        raise AssertionError("preaccess token identity was reused")
    _ISSUED_REAL_PREACCESS_TOKENS[token_id] = (
        token,
        stable_fingerprint(_preaccess_token_payload(token)),
    )
    return token


def _is_live_real_preaccess_token(
    token: object,
) -> bool:
    if (
        type(token) is not GCRPACREDRPreaccessToken
        or token._issuer is not _REAL_PREACCESS_ISSUER
    ):
        return False
    issued = _ISSUED_REAL_PREACCESS_TOKENS.get(id(token))
    return (
        issued is not None
        and issued[0] is token
        and issued[1]
        == stable_fingerprint(_preaccess_token_payload(token))
    )


def _run_start_token_payload(
    token: GCRPACREDRRunStartToken,
) -> dict[str, object]:
    return {
        name: getattr(token, name)
        for name in token.__dataclass_fields__
        if name != "_issuer"
    }


def _register_real_run_start_token(
    token: GCRPACREDRRunStartToken,
) -> GCRPACREDRRunStartToken:
    token_id = id(token)
    if token_id in _ISSUED_REAL_RUN_START_TOKENS:
        raise AssertionError("run-start token identity was reused")
    _ISSUED_REAL_RUN_START_TOKENS[token_id] = (
        token,
        stable_fingerprint(_run_start_token_payload(token)),
    )
    return token


def _is_live_real_run_start_token(
    token: object,
) -> bool:
    if (
        type(token) is not GCRPACREDRRunStartToken
        or token._issuer is not _REAL_RUN_START_ISSUER
    ):
        return False
    issued = _ISSUED_REAL_RUN_START_TOKENS.get(id(token))
    return (
        issued is not None
        and issued[0] is token
        and issued[1]
        == stable_fingerprint(_run_start_token_payload(token))
    )


def _required_run_start_marker_path_from_fingerprint(
    authorization_fingerprint: str,
) -> Path:
    if not _is_sha256(authorization_fingerprint):
        raise ValueError("authorization fingerprint must be SHA-256")
    parent = (
        _repository_root() / GCR_PACRE_DR_RUN_START_PARENT
    )
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or parent.resolve(strict=True) != parent
    ):
        raise RuntimeError(
            "fixed D_R run-start marker parent is unavailable"
        )
    return (
        parent
        / (
            "gcr_pacre_v24_D_R_structural_run_start_"
            f"{authorization_fingerprint}.json"
        )
    )


def required_gcr_pacre_dr_run_start_marker_path(
    preaccess_token: GCRPACREDRPreaccessToken,
) -> Path:
    """Return the sole repository path valid for this authorization."""

    if not _is_live_real_preaccess_token(preaccess_token):
        raise PermissionError(
            "run-start path requires a private preaccess token"
        )
    return _required_run_start_marker_path_from_fingerprint(
        preaccess_token.authorization_fingerprint
    )


def _canonical_absent_output_path(
    path: str | Path,
    *,
    name: str,
) -> Path:
    candidate = Path(path).expanduser()
    absolute = Path(os.path.abspath(candidate))
    parent = absolute.parent
    resolved_parent = parent.resolve(strict=True)
    if (
        resolved_parent != parent
        or not parent.is_dir()
        or parent.is_symlink()
        or absolute.exists()
        or absolute.is_symlink()
    ):
        raise ValueError(f"{name} must be a new canonical path")
    return absolute


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_gcr_pacre_dr_run_start_marker(
    preaccess_token: GCRPACREDRPreaccessToken,
    *,
    marker_path: str | Path,
    requested_device: str,
    requested_receipt_output: str | Path,
) -> GCRPACREDRRunStartToken:
    """Persist the single-use run intent before any real-data materialization."""

    if not _is_live_real_preaccess_token(preaccess_token):
        raise PermissionError(
            "run-start creation requires a private preaccess token"
        )
    required_path = required_gcr_pacre_dr_run_start_marker_path(
        preaccess_token
    )
    raw_marker = Path(marker_path).expanduser()
    absolute_marker = Path(os.path.abspath(raw_marker))
    if absolute_marker != required_path:
        raise PermissionError(
            f"run-start marker path must be exactly {required_path}"
        )
    if not isinstance(requested_device, str) or not requested_device:
        raise TypeError("requested_device must be a nonempty string")
    canonical_device = str(_resolve_device(requested_device))
    receipt_output = _canonical_absent_output_path(
        requested_receipt_output,
        name="requested D_R receipt output",
    )
    implementation = _implementation_binding()
    if (
        preaccess_token.source_closure_fingerprint
        != stable_fingerprint(dict(implementation))
        or preaccess_token.protocol_preregistration_fingerprint
        != _protocol_preregistration_fingerprint()
    ):
        raise PermissionError(
            "D_R source closure changed before persistent run start"
        )
    intent: dict[str, object] = {
        "execution_kind": "real_D_R",
        "split": "D_R",
        "requested_device": canonical_device,
        "requested_receipt_output": str(receipt_output),
        "D_R_materialization_intended": True,
        "D_V_materialization_intended": False,
        "D_T_materialization_intended": False,
        "optimizer_steps_authorized": 0,
        "parameter_updates_authorized": 0,
        "training_authorized": False,
    }
    body: dict[str, object] = {
        "schema_version": GCR_PACRE_DR_RUN_START_SCHEMA,
        "path_policy": GCR_PACRE_DR_RUN_START_PATH_POLICY,
        "stage_id": GCR_PACRE_DR_PREACCESS_STAGE_ID,
        "run_id": GCR_PACRE_DR_RUN_ID,
        "candidate": GCR_PACRE_CANDIDATE,
        "marker_path": str(required_path),
        "authorization_fingerprint": (
            preaccess_token.authorization_fingerprint
        ),
        "authorization_receipt_file_sha256": (
            preaccess_token.authorization_receipt_file_sha256
        ),
        "access_audit_receipt_fingerprint": (
            preaccess_token.access_audit_receipt_fingerprint
        ),
        "access_audit_receipt_file_sha256": (
            preaccess_token.access_audit_receipt_file_sha256
        ),
        "dataset_free_receipt_fingerprint": (
            preaccess_token.dataset_free_receipt_fingerprint
        ),
        "dataset_free_receipt_file_sha256": (
            preaccess_token.dataset_free_receipt_file_sha256
        ),
        "protocol_preregistration_fingerprint": (
            preaccess_token.protocol_preregistration_fingerprint
        ),
        "source_closure_fingerprint": (
            preaccess_token.source_closure_fingerprint
        ),
        "implementation_binding": dict(implementation),
        "expected_source_binding_fingerprint": (
            preaccess_token.expected_source_binding_fingerprint
        ),
        "expected_real_inputs_fingerprint": (
            preaccess_token.expected_real_inputs_fingerprint
        ),
        "expected_population_fingerprint": (
            preaccess_token.expected_population_fingerprint
        ),
        "expected_cache_fingerprint": (
            preaccess_token.expected_cache_fingerprint
        ),
        "intent": intent,
        "intent_fingerprint": stable_fingerprint(intent),
    }
    marker = {
        **body,
        "marker_fingerprint": stable_fingerprint(body),
    }
    encoded = (canonical_json(marker) + "\n").encode("utf-8")
    descriptor = os.open(
        required_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o444,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_parent(required_path)
    except BaseException:
        # The marker is intentionally never rolled back after O_EXCL succeeds.
        try:
            _fsync_parent(required_path)
        finally:
            raise
    stored = _read_strict_json(
        required_path,
        name="persistent D_R run-start marker",
    )
    stat_result = required_path.stat()
    if (
        stored != marker
        or stat_result.st_nlink != 1
        or stat_result.st_mode & 0o222
        or required_path.read_bytes() != encoded
    ):
        raise RuntimeError(
            "persistent D_R run-start marker failed self-verification"
        )
    return _register_real_run_start_token(GCRPACREDRRunStartToken(
        marker_path=str(required_path),
        marker_fingerprint=str(marker["marker_fingerprint"]),
        marker_file_sha256=file_sha256(required_path),
        authorization_fingerprint=(
            preaccess_token.authorization_fingerprint
        ),
        access_audit_receipt_fingerprint=(
            preaccess_token.access_audit_receipt_fingerprint
        ),
        source_closure_fingerprint=(
            preaccess_token.source_closure_fingerprint
        ),
        requested_device=canonical_device,
        requested_receipt_output=str(receipt_output),
        _issuer=_REAL_RUN_START_ISSUER,
    ))


def _verify_real_run_start_token(
    preaccess_token: GCRPACREDRPreaccessToken,
    run_start_token: GCRPACREDRRunStartToken,
) -> dict[str, object]:
    if (
        not _is_live_real_preaccess_token(preaccess_token)
        or not _is_live_real_run_start_token(run_start_token)
        or run_start_token.authorization_fingerprint
        != preaccess_token.authorization_fingerprint
        or run_start_token.access_audit_receipt_fingerprint
        != preaccess_token.access_audit_receipt_fingerprint
        or run_start_token.source_closure_fingerprint
        != preaccess_token.source_closure_fingerprint
    ):
        raise PermissionError("persistent D_R run-start token is invalid")
    path = _canonical_regular_file(
        run_start_token.marker_path,
        name="persistent D_R run-start marker",
    )
    required = required_gcr_pacre_dr_run_start_marker_path(
        preaccess_token
    )
    payload = _read_strict_json(
        path,
        name="persistent D_R run-start marker",
    )
    body = dict(payload)
    marker_fingerprint = body.pop("marker_fingerprint", None)
    stat_result = path.stat()
    if (
        path != required
        or stat_result.st_nlink != 1
        or stat_result.st_mode & 0o222
        or marker_fingerprint != run_start_token.marker_fingerprint
        or marker_fingerprint != stable_fingerprint(body)
        or file_sha256(path) != run_start_token.marker_file_sha256
        or payload.get("authorization_fingerprint")
        != preaccess_token.authorization_fingerprint
        or payload.get("authorization_receipt_file_sha256")
        != preaccess_token.authorization_receipt_file_sha256
        or payload.get("access_audit_receipt_fingerprint")
        != preaccess_token.access_audit_receipt_fingerprint
        or payload.get("access_audit_receipt_file_sha256")
        != preaccess_token.access_audit_receipt_file_sha256
        or payload.get("source_closure_fingerprint")
        != preaccess_token.source_closure_fingerprint
        or payload.get("marker_path") != str(required)
    ):
        raise PermissionError("persistent D_R run-start marker changed")
    intent = payload.get("intent")
    if (
        not isinstance(intent, Mapping)
        or intent.get("requested_device")
        != run_start_token.requested_device
        or intent.get("requested_receipt_output")
        != run_start_token.requested_receipt_output
        or _canonical_absent_output_path(
            run_start_token.requested_receipt_output,
            name="requested D_R receipt output",
        )
        != Path(run_start_token.requested_receipt_output)
    ):
        raise PermissionError("persistent D_R run intent changed")
    return payload


def begin_gcr_pacre_dr_materialization(
    preaccess_token: GCRPACREDRPreaccessToken,
    run_start_token: GCRPACREDRRunStartToken,
) -> GCRPACREDRPreaccessToken:
    """Irreversibly consume preaccess immediately before the D_R loader."""

    if not _is_live_real_preaccess_token(preaccess_token):
        raise PermissionError(
            "D_R materialization requires a private preaccess token"
        )
    _verify_real_run_start_token(
        preaccess_token,
        run_start_token,
    )
    token_id = id(preaccess_token)
    if token_id in _CONSUMED_REAL_PREACCESS_TOKENS:
        raise PermissionError(
            "D_R preaccess token already materialized once"
        )
    _CONSUMED_REAL_PREACCESS_TOKENS.add(token_id)
    return preaccess_token


def _protocol_preregistration_fingerprint() -> str:
    path = (
        _repository_root()
        / "protocols/IRSTD-1K/gcr_pacre_v24/preregistration.json"
    )
    payload = _read_strict_json(path, name="v24 preregistration")
    body = dict(payload)
    fingerprint = body.pop("preregistration_fingerprint", None)
    if (
        not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(body)
        or body.get("schema_version")
        != "cure-lite-v24-gcr-pacre-evidence-preregistration-v1"
    ):
        raise ValueError("v24 preregistration fingerprint changed")
    return str(fingerprint)


def _validate_preaccess_authorization_mapping(
    authorization: Mapping[str, object],
    *,
    dataset_path: Path,
    dataset_fingerprint: str,
    efficiency: Mapping[str, object],
    source_closure_fingerprint: str,
    protocol_fingerprint: str,
    access_fingerprint: str,
    frozen_metadata: Mapping[str, object],
) -> str:
    expected_keys = {
        "schema_version",
        "stage_id",
        "run_id",
        "candidate",
        "status",
        "protocol_preregistration_fingerprint",
        "dataset_free_receipt_fingerprint",
        "dataset_free_receipt_file_sha256",
        "efficiency_section_fingerprint",
        "efficiency_receipt_sha256",
        "source_closure_fingerprint",
        "source_binding_fingerprint",
        "manifest_file_sha256",
        "state_index_file_sha256",
        "expected_real_inputs_fingerprint",
        "expected_population_fingerprint",
        "expected_cache_fingerprint",
        "access_audit_receipt_fingerprint",
        "allowed_splits",
        "allowed_purposes",
        "D_R_payload_authorized",
        "D_V_payload_authorized",
        "D_T_payload_authorized",
        "training_authorized",
        "expires_after_single_materialization",
        "authorization_fingerprint",
    }
    if set(authorization) != expected_keys:
        raise PermissionError("D_R authorization field inventory changed")
    authorization_body = dict(authorization)
    authorization_fingerprint = authorization_body.pop(
        "authorization_fingerprint",
        None,
    )
    if (
        not _is_sha256(authorization_fingerprint)
        or authorization_fingerprint
        != stable_fingerprint(authorization_body)
        or authorization.get("schema_version")
        != GCR_PACRE_DR_PREACCESS_SCHEMA
        or authorization.get("stage_id")
        != GCR_PACRE_DR_PREACCESS_STAGE_ID
        or authorization.get("run_id") != GCR_PACRE_DR_RUN_ID
        or authorization.get("candidate") != GCR_PACRE_CANDIDATE
        or authorization.get("status")
        != GCR_PACRE_DR_PREACCESS_STATUS
        or authorization.get(
            "protocol_preregistration_fingerprint"
        )
        != protocol_fingerprint
        or authorization.get(
            "dataset_free_receipt_fingerprint"
        )
        != dataset_fingerprint
        or authorization.get(
            "dataset_free_receipt_file_sha256"
        )
        != file_sha256(dataset_path)
        or authorization.get("efficiency_section_fingerprint")
        != efficiency["section_fingerprint"]
        or authorization.get("efficiency_receipt_sha256")
        != efficiency["efficiency_receipt_sha256"]
        or authorization.get("source_closure_fingerprint")
        != source_closure_fingerprint
        or authorization.get("source_binding_fingerprint")
        != frozen_metadata["source_binding_fingerprint"]
        or authorization.get("manifest_file_sha256")
        != frozen_metadata["manifest_file_sha256"]
        or authorization.get("state_index_file_sha256")
        != frozen_metadata["state_index_file_sha256"]
        or authorization.get("expected_real_inputs_fingerprint")
        != frozen_metadata["real_inputs_fingerprint"]
        or authorization.get("expected_population_fingerprint")
        != frozen_metadata["population_fingerprint"]
        or authorization.get("expected_cache_fingerprint")
        != frozen_metadata["cache_fingerprint"]
        or authorization.get("access_audit_receipt_fingerprint")
        != access_fingerprint
        or authorization.get("allowed_splits") != ["D_R"]
        or authorization.get("allowed_purposes")
        != ["zero_update_structural_gate"]
        or authorization.get("D_R_payload_authorized") is not True
        or authorization.get("D_V_payload_authorized") is not False
        or authorization.get("D_T_payload_authorized") is not False
        or authorization.get("training_authorized") is not False
        or authorization.get(
            "expires_after_single_materialization"
        )
        is not True
    ):
        raise PermissionError("D_R preaccess authorization is invalid")
    return str(authorization_fingerprint)


def build_gcr_pacre_dr_preaccess_artifacts() -> tuple[
    dict[str, object],
    dict[str, object],
]:
    """Build exact metadata-only access and authorization artifacts.

    The fixed r2 receipt and frozen v23 aggregate receipts are the only
    evidence inputs.  No real-data loader is imported or invoked here.
    """

    root = _repository_root()
    dataset_path = _canonical_regular_file(
        root / GCR_PACRE_DR_DATASET_FREE_RECEIPT_R2_PATH,
        name="dataset-free r2 receipt",
    )
    dataset_receipt = _read_strict_json(
        dataset_path,
        name="dataset-free r2 receipt",
    )
    dataset_fingerprint = verify_gcr_pacre_dataset_free_receipt(
        dataset_receipt
    )
    efficiency = _efficiency_observation(dataset_receipt)
    implementation = _implementation_binding()
    source_closure_fingerprint = stable_fingerprint(
        dict(implementation)
    )
    protocol_fingerprint = _protocol_preregistration_fingerprint()
    frozen = _frozen_v23_dr_input_metadata()
    if (
        frozen["provenance"][
            "protocol_preregistration_fingerprint"
        ]
        != protocol_fingerprint
    ):
        raise PermissionError("frozen v23 metadata protocol binding changed")

    access_body: dict[str, object] = {
        "schema_version": GCR_PACRE_DR_ACCESS_AUDIT_SCHEMA,
        "stage_id": GCR_PACRE_DR_PREACCESS_STAGE_ID,
        "allowed_splits": ["D_R"],
        "observed_payloads": [],
        "source_manifest_fingerprint": frozen[
            "source_manifest_fingerprint"
        ],
        "event_log_fingerprint": stable_fingerprint([]),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    access_receipt = {
        **access_body,
        "receipt_fingerprint": stable_fingerprint(access_body),
    }
    protocol = importlib.import_module(
        "tools.gcr_pacre_v24_protocol"
    )
    fixed_access_verifier = getattr(
        protocol,
        "verify_access_audit_receipt",
    )
    access_token = fixed_access_verifier(
        access_receipt,
        expected_stage_id=GCR_PACRE_DR_PREACCESS_STAGE_ID,
        allowed_splits=("D_R",),
    )
    access_fingerprint = getattr(
        access_token,
        "receipt_fingerprint",
        None,
    )
    if (
        not _is_sha256(access_fingerprint)
        or getattr(access_token, "observed_payloads", None) != ()
    ):
        raise PermissionError("generated D_R access audit is invalid")

    authorization_body: dict[str, object] = {
        "schema_version": GCR_PACRE_DR_PREACCESS_SCHEMA,
        "stage_id": GCR_PACRE_DR_PREACCESS_STAGE_ID,
        "run_id": GCR_PACRE_DR_RUN_ID,
        "candidate": GCR_PACRE_CANDIDATE,
        "status": GCR_PACRE_DR_PREACCESS_STATUS,
        "protocol_preregistration_fingerprint": protocol_fingerprint,
        "dataset_free_receipt_fingerprint": dataset_fingerprint,
        "dataset_free_receipt_file_sha256": file_sha256(
            dataset_path
        ),
        "efficiency_section_fingerprint": efficiency[
            "section_fingerprint"
        ],
        "efficiency_receipt_sha256": efficiency[
            "efficiency_receipt_sha256"
        ],
        "source_closure_fingerprint": source_closure_fingerprint,
        "source_binding_fingerprint": frozen[
            "source_binding_fingerprint"
        ],
        "manifest_file_sha256": frozen["manifest_file_sha256"],
        "state_index_file_sha256": frozen[
            "state_index_file_sha256"
        ],
        "expected_real_inputs_fingerprint": frozen[
            "real_inputs_fingerprint"
        ],
        "expected_population_fingerprint": frozen[
            "population_fingerprint"
        ],
        "expected_cache_fingerprint": frozen["cache_fingerprint"],
        "access_audit_receipt_fingerprint": access_fingerprint,
        "allowed_splits": ["D_R"],
        "allowed_purposes": ["zero_update_structural_gate"],
        "D_R_payload_authorized": True,
        "D_V_payload_authorized": False,
        "D_T_payload_authorized": False,
        "training_authorized": False,
        "expires_after_single_materialization": True,
    }
    authorization = {
        **authorization_body,
        "authorization_fingerprint": stable_fingerprint(
            authorization_body
        ),
    }
    _validate_preaccess_authorization_mapping(
        authorization,
        dataset_path=dataset_path,
        dataset_fingerprint=dataset_fingerprint,
        efficiency=efficiency,
        source_closure_fingerprint=source_closure_fingerprint,
        protocol_fingerprint=protocol_fingerprint,
        access_fingerprint=str(access_fingerprint),
        frozen_metadata=frozen,
    )
    return access_receipt, authorization


def verify_and_issue_gcr_pacre_dr_preaccess(
    *,
    dataset_free_receipt_path: str | Path,
    authorization_receipt_path: str | Path,
    access_audit_receipt_path: str | Path,
) -> GCRPACREDRPreaccessToken:
    """Verify exact metadata prerequisites before any ``D_R`` loader call."""

    dataset_path = _canonical_regular_file(
        dataset_free_receipt_path,
        name="dataset-free receipt",
    )
    authorization_path = _canonical_regular_file(
        authorization_receipt_path,
        name="D_R preaccess authorization",
    )
    access_path = _canonical_regular_file(
        access_audit_receipt_path,
        name="D_R access-audit receipt",
    )
    dataset_receipt = _read_strict_json(
        dataset_path,
        name="dataset-free receipt",
    )
    dataset_fingerprint = verify_gcr_pacre_dataset_free_receipt(
        dataset_receipt
    )
    efficiency = _efficiency_observation(dataset_receipt)
    implementation = _implementation_binding()
    source_closure_fingerprint = stable_fingerprint(
        dict(implementation)
    )
    protocol_fingerprint = _protocol_preregistration_fingerprint()
    frozen = _frozen_v23_dr_input_metadata()

    # This import is deliberately metadata-only.  The protocol module has no
    # dataset/model entry point and issues its own private access token.
    protocol = importlib.import_module("tools.gcr_pacre_v24_protocol")
    fixed_access_verifier = getattr(
        protocol,
        "verify_access_audit_receipt",
    )
    access_receipt = _read_strict_json(
        access_path,
        name="D_R access-audit receipt",
    )
    access_token = fixed_access_verifier(
        access_receipt,
        expected_stage_id=GCR_PACRE_DR_PREACCESS_STAGE_ID,
        allowed_splits=("D_R",),
    )
    if (
        getattr(access_token, "stage_id", None)
        != GCR_PACRE_DR_PREACCESS_STAGE_ID
        or getattr(access_token, "allowed_splits", None) != ("D_R",)
        or access_receipt.get("observed_payloads") != []
        or access_receipt.get("source_manifest_fingerprint")
        != frozen["source_manifest_fingerprint"]
        or access_receipt.get("event_log_fingerprint")
        != stable_fingerprint([])
    ):
        raise PermissionError(
            "D_R access audit must precede payload materialization"
        )
    access_fingerprint = getattr(
        access_token,
        "receipt_fingerprint",
        None,
    )
    if not _is_sha256(access_fingerprint):
        raise PermissionError("D_R access-audit token is invalid")

    authorization = _read_strict_json(
        authorization_path,
        name="D_R preaccess authorization",
    )
    authorization_fingerprint = (
        _validate_preaccess_authorization_mapping(
            authorization,
            dataset_path=dataset_path,
            dataset_fingerprint=dataset_fingerprint,
            efficiency=efficiency,
            source_closure_fingerprint=source_closure_fingerprint,
            protocol_fingerprint=protocol_fingerprint,
            access_fingerprint=str(access_fingerprint),
            frozen_metadata=frozen,
        )
    )
    return _register_real_preaccess_token(GCRPACREDRPreaccessToken(
        dataset_free_receipt_fingerprint=dataset_fingerprint,
        dataset_free_receipt_file_sha256=file_sha256(dataset_path),
        efficiency_section_fingerprint=str(
            efficiency["section_fingerprint"]
        ),
        efficiency_receipt_sha256=str(
            efficiency["efficiency_receipt_sha256"]
        ),
        authorization_fingerprint=str(authorization_fingerprint),
        authorization_receipt_file_sha256=file_sha256(
            authorization_path
        ),
        access_audit_receipt_fingerprint=str(access_fingerprint),
        access_audit_receipt_file_sha256=file_sha256(access_path),
        protocol_preregistration_fingerprint=protocol_fingerprint,
        source_closure_fingerprint=source_closure_fingerprint,
        expected_source_binding_fingerprint=str(
            authorization["source_binding_fingerprint"]
        ),
        expected_manifest_file_sha256=str(
            authorization["manifest_file_sha256"]
        ),
        expected_state_index_file_sha256=str(
            authorization["state_index_file_sha256"]
        ),
        expected_real_inputs_fingerprint=str(
            authorization["expected_real_inputs_fingerprint"]
        ),
        expected_population_fingerprint=str(
            authorization["expected_population_fingerprint"]
        ),
        expected_cache_fingerprint=str(
            authorization["expected_cache_fingerprint"]
        ),
        efficiency_observation_json=canonical_json(efficiency),
        _issuer=_REAL_PREACCESS_ISSUER,
    ))


def _current_adapter_fingerprint(adapter: _PopulationAdapter) -> str:
    return _adapter_fingerprint(
        mode=adapter.mode,
        split=adapter.split,
        seed=adapter.seed,
        targets=adapter.target_states,
        contexts=adapter.context_states,
        population_fingerprint=adapter.population_fingerprint,
        cache_fingerprint=adapter.cache_fingerprint,
    )


def _run_probe(
    adapter: _PopulationAdapter,
    *,
    device: torch.device,
    execution_token: GCRPACREDRPreaccessToken | _GeneratedAuditToken,
) -> dict[str, object]:
    if type(execution_token) is GCRPACREDRPreaccessToken:
        if (
            execution_token._issuer is not _REAL_PREACCESS_ISSUER
            or adapter.mode != "real_D_R"
            or adapter.split != "D_R"
        ):
            raise PermissionError("real D_R preaccess token is invalid")
    elif type(execution_token) is _GeneratedAuditToken:
        if (
            execution_token._issuer is not _GENERATED_AUDIT_ISSUER
            or adapter.mode != "generated"
            or adapter.split != "generated"
            or execution_token.adapter_fingerprint
            != adapter.adapter_fingerprint
        ):
            raise PermissionError("generated audit token is invalid")
    else:
        raise TypeError("execution token has the wrong exact type")

    before_cpu_rng = torch.get_rng_state().clone()
    before_cuda_rng = (
        torch.cuda.get_rng_state(device).clone()
        if device.type == "cuda"
        else None
    )
    before_adapter = _current_adapter_fingerprint(adapter)
    with _deterministic_execution_scope() as deterministic_execution:
        with torch.random.fork_rng(
            devices=_cuda_rng_devices(device),
            enabled=True,
        ):
            torch.manual_seed(GCR_PACRE_DR_EXECUTION_SEED)
            model = build_formal_gcr_pacre_training_model().to(
                device=device,
                dtype=torch.float32,
            )
            model.eval()
            initial_model_fingerprint = (
                coverage_state_model_fingerprint(model)
            )
            model_contract = coverage_state_model_contract_payload(
                model
            )
            parameter_names = tuple(
                name for name, _ in model.named_parameters()
            )
            parameter_ids = {
                name: id(parameter)
                for name, parameter in model.named_parameters()
            }
            buffer_ids = {
                name: id(buffer)
                for name, buffer in model.named_buffers()
            }
            with _exact_model_restore(model) as model_restoration:
                zero_anchor = _zero_readout_anchor_probe(
                    model,
                    adapter,
                    device=device,
                )
                initial_gradient = _initial_gradient_probe(
                    model,
                    adapter.gradient_fixture,
                    device=device,
                )
                with torch.no_grad():
                    fixed_readout = torch.linspace(
                        0.5,
                        1.5,
                        model.config.width,
                        device=device,
                        dtype=torch.float32,
                    )
                    model.scalar_energy_weight.copy_(fixed_readout)
                fixed_readout_fingerprint = (
                    tensor_content_fingerprint(
                        fixed_readout.detach().to("cpu")
                    )
                )
                ledger = _ledger_probe(
                    model,
                    adapter,
                    device=device,
                )
                common_only = _common_only_probe(
                    model,
                    adapter.target_states[0],
                    device=device,
                )
                direction = _direction_probe(
                    adapter,
                    device=device,
                )
            final_model_fingerprint = coverage_state_model_fingerprint(
                model
            )
            final_parameter_ids = {
                name: id(parameter)
                for name, parameter in model.named_parameters()
            }
            final_buffer_ids = {
                name: id(buffer)
                for name, buffer in model.named_buffers()
            }
            grad_buffers_unretained = all(
                parameter.grad is None
                for parameter in model.parameters()
            )
    after_cuda_rng = (
        torch.cuda.get_rng_state(device).clone()
        if device.type == "cuda"
        else None
    )
    after_adapter = _current_adapter_fingerprint(adapter)
    boundary = {
        "execution_kind": adapter.mode,
        "split": adapter.split,
        "D_R_accessed": adapter.mode == "real_D_R",
        "D_V_accessed": False,
        "D_T_accessed": False,
        "D_V_tensor_payload_accessed": False,
        "D_T_tensor_payload_accessed": False,
        "optimizer_module_referenced": False,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "training_performed": False,
        "performance_gate_present": False,
        "performance_claim_supported": False,
        "threshold_or_ratio_gate": None,
    }
    body: dict[str, object] = {
        "device": str(device),
        "execution_seed": GCR_PACRE_DR_EXECUTION_SEED,
        "adapter_mode": adapter.mode,
        "adapter_split": adapter.split,
        "adapter_fingerprint_before": before_adapter,
        "adapter_fingerprint_after": after_adapter,
        "population_fingerprint": adapter.population_fingerprint,
        "cache_fingerprint": adapter.cache_fingerprint,
        "source_cache_fingerprint": (
            adapter.source_cache_fingerprint
        ),
        "model_fqcn": _fqcn(model),
        "config_fqcn": _fqcn(model.config),
        "factory_fqcn": (
            f"{build_formal_gcr_pacre_training_model.__module__}."
            f"{build_formal_gcr_pacre_training_model.__qualname__}"
        ),
        "model_contract": model_contract,
        "model_contract_fingerprint": stable_fingerprint(
            model_contract
        ),
        "parameter_names": list(parameter_names),
        "initial_model_fingerprint": initial_model_fingerprint,
        "final_model_fingerprint": final_model_fingerprint,
        "parameter_ids_preserved": parameter_ids == final_parameter_ids,
        "buffer_ids_preserved": buffer_ids == final_buffer_ids,
        "model_restoration": model_restoration,
        "zero_readout_anchor": zero_anchor,
        "fixed_readout_policy": GCR_PACRE_DR_FIXED_READOUT_POLICY,
        "fixed_readout_fingerprint": fixed_readout_fingerprint,
        "initial_gradient_path": initial_gradient,
        "complete_state_ledger": ledger,
        "common_only_probe": common_only,
        "field_direction_probe": direction,
        "deterministic_execution": deterministic_execution,
        "global_cpu_rng_preserved": torch.equal(
            before_cpu_rng,
            torch.get_rng_state(),
        ),
        "selected_device_rng_preserved": (
            before_cuda_rng is None
            or torch.equal(before_cuda_rng, after_cuda_rng)
        ),
        "parameter_grad_buffers_unretained": (
            grad_buffers_unretained
            and initial_gradient[
                "parameter_grad_buffers_unretained"
            ]
        ),
        "boundary": boundary,
    }
    return {**body, "probe_fingerprint": stable_fingerprint(body)}


def _observation(value: Mapping[str, object]) -> dict[str, object]:
    body = dict(value)
    body.pop("observation_fingerprint", None)
    return {
        **body,
        "observation_fingerprint": stable_fingerprint(body),
    }


def _static_scope_observation() -> dict[str, object]:
    source = Path(__file__).read_text(encoding="utf-8")
    optimizer_namespace = "torch." + "optim"
    optimizer_step = "optimizer" + ".step"
    backward_call = "." + "backward("
    forbidden_loader_names = (
        "build_" + "D_V" + "_inputs",
        "build_" + "D_T" + "_inputs",
        "load_" + "D_V",
        "load_" + "D_T",
    )
    return {
        "module_path": "cure_lite_v24/dr_gate.py",
        "module_sha256": file_sha256(Path(__file__).resolve()),
        "optimizer_namespace_occurrences": source.count(
            optimizer_namespace
        ),
        "optimizer_step_occurrences": source.count(optimizer_step),
        "backward_call_occurrences": source.count(backward_call),
        "forbidden_split_loader_occurrences": {
            name: source.count(name) for name in forbidden_loader_names
        },
        "public_real_loader_imported": (
            (
                "build_coverage_state_"
                + "real_dr_inputs"
            )
            in source
        ),
        "autograd_grad_api_used": "torch.autograd.grad(" in source,
        "temporary_parameter_probe_restore_context_used": (
            "with _exact_model_restore(model)" in source
        ),
    }


def _raw_observations(
    *,
    probe: Mapping[str, object],
    prerequisite: Mapping[str, object],
    input_binding: Mapping[str, object],
    efficiency: Mapping[str, object],
) -> dict[str, object]:
    ledger = probe.get("complete_state_ledger")
    if not isinstance(ledger, Mapping):
        raise TypeError("complete state ledger is absent")
    target = ledger.get("target_summary")
    context = ledger.get("context_summary")
    residual_collisions = ledger.get("residual_latent_collisions")
    gated_collisions = ledger.get("gated_latent_collisions")
    common_only = probe.get("common_only_probe")
    gradient = probe.get("initial_gradient_path")
    direction = probe.get("field_direction_probe")
    boundary = probe.get("boundary")
    if not all(
        isinstance(value, Mapping)
        for value in (
            target,
            context,
            residual_collisions,
            gated_collisions,
            common_only,
            gradient,
            direction,
            boundary,
        )
    ):
        raise TypeError("probe sections are incomplete")
    preservation = {
        "initial_model_fingerprint": probe.get(
            "initial_model_fingerprint"
        ),
        "final_model_fingerprint": probe.get(
            "final_model_fingerprint"
        ),
        "parameter_ids_preserved": probe.get(
            "parameter_ids_preserved"
        ),
        "buffer_ids_preserved": probe.get("buffer_ids_preserved"),
        "model_restoration": probe.get("model_restoration"),
        "adapter_fingerprint_before": probe.get(
            "adapter_fingerprint_before"
        ),
        "adapter_fingerprint_after": probe.get(
            "adapter_fingerprint_after"
        ),
        "population_fingerprint": probe.get(
            "population_fingerprint"
        ),
        "cache_fingerprint": probe.get("cache_fingerprint"),
        "source_cache_fingerprint": probe.get(
            "source_cache_fingerprint"
        ),
        "global_cpu_rng_preserved": probe.get(
            "global_cpu_rng_preserved"
        ),
        "selected_device_rng_preserved": probe.get(
            "selected_device_rng_preserved"
        ),
        "parameter_grad_buffers_unretained": probe.get(
            "parameter_grad_buffers_unretained"
        ),
        "deterministic_execution": probe.get(
            "deterministic_execution"
        ),
    }
    fast_full = {
        "fields_fqcn": GCR_PACRE_FIELDS_FQCN,
        "target_summary": target,
        "context_summary": context,
        "target_complete_validator_call_count": sum(
            int(row.get("complete_fields_validator_called") is True)
            for row in ledger.get("target_rows", [])
            if isinstance(row, Mapping)
        ),
        "context_complete_validator_call_count": sum(
            int(row.get("complete_fields_validator_called") is True)
            for row in ledger.get("context_rows", [])
            if isinstance(row, Mapping)
        ),
        "target_fast_fp64_agreement_count": sum(
            int(row.get("fast_fp64_ledger_agrees") is True)
            for row in ledger.get("target_rows", [])
            if isinstance(row, Mapping)
        ),
        "context_fast_fp64_agreement_count": sum(
            int(row.get("fast_fp64_ledger_agrees") is True)
            for row in ledger.get("context_rows", [])
            if isinstance(row, Mapping)
        ),
        "ledger_fingerprint": ledger.get("ledger_fingerprint"),
    }
    observations = {
        GCR_PACRE_DR_CHECK_NAMES[0]: _observation(prerequisite),
        GCR_PACRE_DR_CHECK_NAMES[1]: _observation(input_binding),
        GCR_PACRE_DR_CHECK_NAMES[2]: _observation(
            {
                "model_fqcn": probe.get("model_fqcn"),
                "config_fqcn": probe.get("config_fqcn"),
                "factory_fqcn": probe.get("factory_fqcn"),
                "model_contract": probe.get("model_contract"),
                "model_contract_fingerprint": probe.get(
                    "model_contract_fingerprint"
                ),
                "parameter_names": probe.get("parameter_names"),
            }
        ),
        GCR_PACRE_DR_CHECK_NAMES[3]: _observation(
            {
                "target_summary": target,
                "context_summary": context,
                "target_context_scoped_ids_disjoint": ledger.get(
                    "target_context_scoped_ids_disjoint"
                ),
                "union_scoped_state_count": ledger.get(
                    "union_scoped_state_count"
                ),
                "expected_union_scoped_state_count": ledger.get(
                    "expected_union_scoped_state_count"
                ),
            }
        ),
        GCR_PACRE_DR_CHECK_NAMES[4]: _observation(
            {
                "target_summary": target,
                "target_row_count": len(ledger.get("target_rows", [])),
                "fields_fqcn_values": sorted(
                    {
                        str(row.get("fields_fqcn"))
                        for row in ledger.get("target_rows", [])
                        if isinstance(row, Mapping)
                    }
                ),
            }
        ),
        GCR_PACRE_DR_CHECK_NAMES[5]: _observation(
            {
                "target_group_count": len(
                    ledger.get("target_witnesses", [])
                ),
                "target_witnesses": ledger.get(
                    "target_witnesses"
                ),
            }
        ),
        GCR_PACRE_DR_CHECK_NAMES[6]: _observation(
            dict(residual_collisions)
        ),
        GCR_PACRE_DR_CHECK_NAMES[7]: _observation(
            {
                "zero_readout_anchor": probe.get(
                    "zero_readout_anchor"
                ),
                "fixed_readout_policy": probe.get(
                    "fixed_readout_policy"
                ),
                "fixed_readout_fingerprint": probe.get(
                    "fixed_readout_fingerprint"
                ),
                "fixed_readout_mechanism_nonzero": ledger.get(
                    "fixed_readout_mechanism_nonzero"
                ),
            }
        ),
        GCR_PACRE_DR_CHECK_NAMES[8]: _observation(dict(gradient)),
        GCR_PACRE_DR_CHECK_NAMES[9]: _observation(dict(direction)),
        GCR_PACRE_DR_CHECK_NAMES[10]: _observation(preservation),
        GCR_PACRE_DR_CHECK_NAMES[11]: _observation(
            {
                "boundary": boundary,
                "static_scope": _static_scope_observation(),
            }
        ),
        GCR_PACRE_DR_CHECK_NAMES[12]: _observation(
            {
                "context_summary": context,
                "context_row_count": len(
                    ledger.get("context_rows", [])
                ),
                "fields_fqcn_values": sorted(
                    {
                        str(row.get("fields_fqcn"))
                        for row in ledger.get("context_rows", [])
                        if isinstance(row, Mapping)
                    }
                ),
            }
        ),
        GCR_PACRE_DR_CHECK_NAMES[13]: _observation(
            {
                "common_nonzero_counts": ledger.get(
                    "common_nonzero_counts"
                ),
                "target_state_count": target.get(
                    "observed_state_count"
                ),
                "context_state_count": context.get(
                    "observed_state_count"
                ),
            }
        ),
        GCR_PACRE_DR_CHECK_NAMES[14]: _observation(
            {
                "all_target_flip_parity_exact": ledger.get(
                    "all_target_flip_parity_exact"
                ),
                "target_flip_rows": [
                    {
                        "scoped_state_id": row.get(
                            "scoped_state_id"
                        ),
                        "flip_parity": row.get("flip_parity"),
                    }
                    for row in ledger.get("target_rows", [])
                    if isinstance(row, Mapping)
                ],
            }
        ),
        GCR_PACRE_DR_CHECK_NAMES[15]: _observation(
            dict(common_only)
        ),
        GCR_PACRE_DR_CHECK_NAMES[16]: _observation(
            {
                "target_witnesses": ledger.get(
                    "target_witnesses"
                ),
                "gradient_policy": (
                    GCR_PACRE_DR_GATE_GRADIENT_POLICY
                ),
            }
        ),
        GCR_PACRE_DR_CHECK_NAMES[17]: _observation(
            dict(gated_collisions)
        ),
        GCR_PACRE_DR_CHECK_NAMES[18]: _observation(preservation),
        GCR_PACRE_DR_CHECK_NAMES[19]: _observation(
            {
                "boundary": boundary,
                "static_scope": _static_scope_observation(),
            }
        ),
        GCR_PACRE_DR_CHECK_NAMES[20]: _observation(
            dict(ledger["gate_distribution"])
        ),
        GCR_PACRE_DR_CHECK_NAMES[21]: _observation(fast_full),
        GCR_PACRE_DR_CHECK_NAMES[22]: _observation(efficiency),
    }
    if tuple(observations) != GCR_PACRE_DR_CHECK_NAMES:
        raise AssertionError("D_R raw observation order changed")
    return observations


def _preservation_passed(value: Mapping[str, object]) -> bool:
    restoration = value.get("model_restoration")
    deterministic = value.get("deterministic_execution")
    return (
        value.get("initial_model_fingerprint")
        == value.get("final_model_fingerprint")
        and value.get("parameter_ids_preserved") is True
        and value.get("buffer_ids_preserved") is True
        and isinstance(restoration, Mapping)
        and restoration.get("restored_exactly") is True
        and value.get("adapter_fingerprint_before")
        == value.get("adapter_fingerprint_after")
        and _is_sha256(value.get("population_fingerprint"))
        and _is_sha256(value.get("cache_fingerprint"))
        and _is_sha256(value.get("source_cache_fingerprint"))
        and value.get("global_cpu_rng_preserved") is True
        and value.get("selected_device_rng_preserved") is True
        and value.get("parameter_grad_buffers_unretained") is True
        and isinstance(deterministic, Mapping)
        and deterministic.get("restored_exactly") is True
    )


def _scope_passed(
    value: Mapping[str, object],
    *,
    execution_kind: str,
) -> bool:
    boundary = value.get("boundary")
    static = value.get("static_scope")
    return (
        isinstance(boundary, Mapping)
        and boundary.get("execution_kind") == execution_kind
        and boundary.get("D_R_accessed")
        is (execution_kind == "real_D_R")
        and boundary.get("D_V_accessed") is False
        and boundary.get("D_T_accessed") is False
        and boundary.get("D_V_tensor_payload_accessed") is False
        and boundary.get("D_T_tensor_payload_accessed") is False
        and boundary.get("optimizer_constructed") is False
        and boundary.get("optimizer_steps") == 0
        and boundary.get("parameter_updates") == 0
        and boundary.get("training_performed") is False
        and boundary.get("performance_gate_present") is False
        and boundary.get("threshold_or_ratio_gate") is None
        and isinstance(static, Mapping)
        and static.get("optimizer_namespace_occurrences") == 0
        and static.get("optimizer_step_occurrences") == 0
        and static.get("backward_call_occurrences") == 0
        and all(
            count == 0
            for count in static.get(
                "forbidden_split_loader_occurrences",
                {},
            ).values()
        )
        and static.get("public_real_loader_imported") is False
        and static.get("autograd_grad_api_used") is True
        and static.get(
            "temporary_parameter_probe_restore_context_used"
        )
        is True
    )


def recompute_gcr_pacre_dr_checks(
    raw_observations: Mapping[str, object],
    *,
    execution_kind: str,
) -> tuple[tuple[str, bool], ...]:
    """Derive all 23 decisions from sealed raw observations."""

    if execution_kind not in {"generated", "real_D_R"}:
        raise ValueError("execution_kind must be generated or real_D_R")
    if (
        not isinstance(raw_observations, Mapping)
        or tuple(raw_observations) != GCR_PACRE_DR_CHECK_NAMES
    ):
        raise ValueError("D_R raw observation inventory changed")
    raw: dict[str, dict[str, object]] = {}
    for name in GCR_PACRE_DR_CHECK_NAMES:
        value = raw_observations[name]
        if not isinstance(value, Mapping):
            raise TypeError(f"raw observation {name} must be a mapping")
        body = dict(value)
        fingerprint = body.pop("observation_fingerprint", None)
        if (
            not _is_sha256(fingerprint)
            or fingerprint != stable_fingerprint(body)
        ):
            raise ValueError(f"raw observation {name} was altered")
        raw[name] = body

    prerequisite = raw[GCR_PACRE_DR_CHECK_NAMES[0]]
    binding = raw[GCR_PACRE_DR_CHECK_NAMES[1]]
    model = raw[GCR_PACRE_DR_CHECK_NAMES[2]]
    inventory = raw[GCR_PACRE_DR_CHECK_NAMES[3]]
    target_algebra = raw[GCR_PACRE_DR_CHECK_NAMES[4]]
    witnesses = raw[GCR_PACRE_DR_CHECK_NAMES[5]]
    residual_collision = raw[GCR_PACRE_DR_CHECK_NAMES[6]]
    readout = raw[GCR_PACRE_DR_CHECK_NAMES[7]]
    gradient = raw[GCR_PACRE_DR_CHECK_NAMES[8]]
    direction = raw[GCR_PACRE_DR_CHECK_NAMES[9]]
    preservation_11 = raw[GCR_PACRE_DR_CHECK_NAMES[10]]
    scope_12 = raw[GCR_PACRE_DR_CHECK_NAMES[11]]
    context_algebra = raw[GCR_PACRE_DR_CHECK_NAMES[12]]
    common = raw[GCR_PACRE_DR_CHECK_NAMES[13]]
    parity = raw[GCR_PACRE_DR_CHECK_NAMES[14]]
    common_only = raw[GCR_PACRE_DR_CHECK_NAMES[15]]
    target_paths = raw[GCR_PACRE_DR_CHECK_NAMES[16]]
    gated_collision = raw[GCR_PACRE_DR_CHECK_NAMES[17]]
    preservation_19 = raw[GCR_PACRE_DR_CHECK_NAMES[18]]
    scope_20 = raw[GCR_PACRE_DR_CHECK_NAMES[19]]
    gate_distribution = raw[GCR_PACRE_DR_CHECK_NAMES[20]]
    fast_full = raw[GCR_PACRE_DR_CHECK_NAMES[21]]
    efficiency = raw[GCR_PACRE_DR_CHECK_NAMES[22]]

    contract = model.get("model_contract")
    config = (
        contract.get("config")
        if isinstance(contract, Mapping)
        else None
    )
    shapes = (
        contract.get("parameter_shapes")
        if isinstance(contract, Mapping)
        else None
    )
    target_summary = inventory.get("target_summary")
    context_summary = inventory.get("context_summary")
    witness_rows = witnesses.get("target_witnesses")
    zero = readout.get("zero_readout_anchor")
    gradient_rows = gradient.get("initial_gradient_rows")
    cross_rows = gradient.get(
        "readout_to_upstream_cross_gradient_rows"
    )
    direction_rows = direction.get("rows")
    common_counts = common.get("common_nonzero_counts")
    parity_rows = parity.get("target_flip_rows")
    common_restore = common_only.get("model_restoration")
    path_rows = target_paths.get("target_witnesses")
    arms = efficiency.get("arms")

    exact_witnesses = (
        isinstance(witness_rows, list)
        and len(witness_rows) == GCR_PACRE_DR_TARGET_STATE_COUNT
        and len(
            {
                str(row.get("target_group_id"))
                for row in witness_rows
                if isinstance(row, Mapping)
            }
        )
        == GCR_PACRE_DR_TARGET_STATE_COUNT
        and all(
            isinstance(row, Mapping)
            and row.get("target_group_id") is not None
            and isinstance(row.get("target_coordinate_count"), int)
            and int(row["target_coordinate_count"]) > 0
            and isinstance(
                row.get("legal_same_cell_background_pair_count"),
                int,
            )
            and int(row["legal_same_cell_background_pair_count"]) > 0
            and isinstance(row.get("selected_first_exact_witness"), Mapping)
            and row.get("no_numeric_separation_threshold") is True
            and _is_sha256(row.get("binding_fingerprint"))
            and row.get("witness_passed") is True
            for row in witness_rows
        )
    )
    initial_gradient_passed = (
        isinstance(gradient_rows, Mapping)
        and set(gradient_rows) == set(GCR_PACRE_PARAMETER_NAMES)
        and _all_gradient_rows_finite(
            gradient_rows,
            require_any_nonzero=True,
        )
        and gradient_rows["scalar_energy_weight"].get(
            "nonzero_count"
        )
        > 0
        and gradient_rows["joint_state_weight"].get(
            "nonzero_count"
        )
        == 0
        and gradient_rows["joint_hidden_bias"].get(
            "nonzero_count"
        )
        == 0
        and isinstance(cross_rows, Mapping)
        and set(cross_rows)
        == {"joint_state_weight", "joint_hidden_bias"}
        and all(
            isinstance(row, Mapping)
            and row.get("present") is True
            and row.get("finite") is True
            and isinstance(row.get("nonzero_count"), int)
            and int(row["nonzero_count"]) > 0
            and _is_sha256(row.get("fingerprint"))
            for row in cross_rows.values()
        )
        and gradient.get("readout_visible_upstream_dormant") is True
        and gradient.get("parameter_grad_buffers_unretained") is True
    )
    direction_passed = (
        isinstance(direction_rows, list)
        and len(direction_rows)
        == 4 * COVERAGE_STATE_BOUNDED_ROLE_COUNT
        and direction.get("expected_role_rows")
        == direction.get("observed_role_rows")
        == {
            "factual_target": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
            "clean_target": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
            "writable_background": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
            "writable_component": COVERAGE_STATE_BOUNDED_ROLE_COUNT,
        }
        and all(
            isinstance(row, Mapping)
            and row.get("coordinate_count", 0) > 0
            and row.get("descent_finite") is True
            and row.get("descent_nonzero") is True
            and row.get("aggregate_descent_direction_correct") is True
            and float.fromhex(str(row.get("loss_hex"))) > 0.0
            for row in direction_rows
        )
    )
    target_path_passed = (
        isinstance(path_rows, list)
        and len(path_rows) == GCR_PACRE_DR_TARGET_STATE_COUNT
        and all(
            isinstance(row, Mapping)
            and row.get(
                "residual_and_gate_paths_finite_nonzero"
            )
            is True
            and row.get("witness_passed") is True
            for row in path_rows
        )
        and target_paths.get("gradient_policy")
        == GCR_PACRE_DR_GATE_GRADIENT_POLICY
    )
    efficiency_passed = (
        _is_sha256(efficiency.get("section_fingerprint"))
        and _is_sha256(efficiency.get("efficiency_receipt_sha256"))
        and _is_sha256(
            efficiency.get("common_conditions_fingerprint")
        )
        and efficiency.get("formal_model_config")
        == {
            "feature_channels": GCR_PACRE_FORMAL_FEATURE_CHANNELS,
            "feature_stride": GCR_PACRE_FORMAL_FEATURE_STRIDE,
            "width": GCR_PACRE_FORMAL_WIDTH,
            "parameter_count": GCR_PACRE_FORMAL_PARAMETER_COUNT,
        }
        and efficiency.get("threshold_or_ratio_gate") is None
        and efficiency.get("performance_ratio_or_absolute_gate")
        is None
        and efficiency.get("interpretation")
        == "measurement_only_no_post_hoc_lite_overhead_threshold"
        and efficiency.get("both_arms_complete_finite_no_oom")
        is True
        and efficiency.get("v24_additional_ops_recorded") is True
        and isinstance(arms, Mapping)
        and set(arms) == {"PACRE_VC_v23", "GCR_PACRE_v24"}
        and all(
            isinstance(row, Mapping)
            and row.get("oom") is False
            and row.get("nonfinite") is False
            and row.get("parameter_count")
            == GCR_PACRE_FORMAL_PARAMETER_COUNT
            and isinstance(row.get("forward_flops"), int)
            and not isinstance(row.get("forward_flops"), bool)
            and int(row["forward_flops"]) > 0
            and all(
                isinstance(row.get(name), int)
                and not isinstance(row.get(name), bool)
                and int(row[name]) > 0
                for name in (
                    "parameter_bytes",
                    "checkpoint_bytes",
                    "field_tensor_bytes",
                )
            )
            and isinstance(row.get("latencies"), Mapping)
            and set(row["latencies"])
            == {"forward_latency", "train_step_latency"}
            for row in arms.values()
        )
    )
    checks = {
        GCR_PACRE_DR_CHECK_NAMES[0]: (
            prerequisite.get("official_verifier_called") is True
            and _is_sha256(
                prerequisite.get(
                    "dataset_free_receipt_fingerprint"
                )
            )
            and _is_sha256(
                prerequisite.get(
                    "dataset_free_receipt_file_sha256"
                )
            )
            and prerequisite.get("dataset_free_gate_passed") is True
            and prerequisite.get("D_R_authorized_by_dataset_free")
            is False
        ),
        GCR_PACRE_DR_CHECK_NAMES[1]: (
            binding.get("execution_kind") == execution_kind
            and binding.get("seed") == GCR_PACRE_DR_EXECUTION_SEED
            and binding.get("target_state_count")
            == GCR_PACRE_DR_TARGET_STATE_COUNT
            and binding.get("context_state_count")
            == GCR_PACRE_DR_CONTEXT_STATE_COUNT
            and _is_sha256(binding.get("adapter_fingerprint"))
            and _is_sha256(binding.get("population_fingerprint"))
            and _is_sha256(binding.get("cache_fingerprint"))
            and _is_sha256(binding.get("source_cache_fingerprint"))
            and (
                (
                    execution_kind == "generated"
                    and binding.get("split") == "generated"
                    and binding.get("real_dataset_accessed") is False
                    and binding.get("real_preaccess_token_used") is False
                )
                or (
                    execution_kind == "real_D_R"
                    and binding.get("split") == "D_R"
                    and binding.get("real_dataset_accessed") is True
                    and binding.get("real_preaccess_token_used") is True
                    and binding.get(
                        "authorization_matches_live_inputs"
                    )
                    is True
                )
            )
        ),
        GCR_PACRE_DR_CHECK_NAMES[2]: (
            model.get("model_fqcn")
            == (
                "cure_lite_v24.gcr_pacre."
                "CURELiteGatedCommonResidualPACRELevelSet"
            )
            and model.get("config_fqcn")
            == (
                "cure_lite_v24.gcr_pacre."
                "CoverageStateGCRPACREConfig"
            )
            and model.get("factory_fqcn")
            == (
                "cure_lite_v24.factory."
                "build_formal_gcr_pacre_training_model"
            )
            and isinstance(contract, Mapping)
            and contract.get("parameter_count")
            == GCR_PACRE_FORMAL_PARAMETER_COUNT
            and model.get("parameter_names")
            == list(GCR_PACRE_PARAMETER_NAMES)
            and isinstance(config, Mapping)
            and config.get("feature_channels")
            == GCR_PACRE_FORMAL_FEATURE_CHANNELS
            and config.get("feature_stride")
            == GCR_PACRE_FORMAL_FEATURE_STRIDE
            and config.get("width") == GCR_PACRE_FORMAL_WIDTH
            and shapes
            == {
                "joint_hidden_bias": [GCR_PACRE_FORMAL_WIDTH],
                "joint_state_weight": [
                    GCR_PACRE_FORMAL_WIDTH,
                    (
                        GCR_PACRE_FORMAL_FEATURE_CHANNELS
                        + GCR_PACRE_FORMAL_FEATURE_STRIDE**2
                    ),
                    5,
                    5,
                ],
                "scalar_energy_weight": [
                    GCR_PACRE_FORMAL_WIDTH
                ],
            }
            and model.get("model_contract_fingerprint")
            == stable_fingerprint(contract)
        ),
        GCR_PACRE_DR_CHECK_NAMES[3]: (
            isinstance(target_summary, Mapping)
            and isinstance(context_summary, Mapping)
            and target_summary.get("observed_state_count")
            == GCR_PACRE_DR_TARGET_STATE_COUNT
            and context_summary.get("observed_state_count")
            == GCR_PACRE_DR_CONTEXT_STATE_COUNT
            and target_summary.get("scoped_state_ids_unique") is True
            and context_summary.get("scoped_state_ids_unique") is True
            and inventory.get("target_context_scoped_ids_disjoint")
            is True
            and inventory.get("union_scoped_state_count")
            == inventory.get("expected_union_scoped_state_count")
            == (
                GCR_PACRE_DR_TARGET_STATE_COUNT
                + GCR_PACRE_DR_CONTEXT_STATE_COUNT
            )
        ),
        GCR_PACRE_DR_CHECK_NAMES[4]: (
            target_algebra.get("target_row_count")
            == GCR_PACRE_DR_TARGET_STATE_COUNT
            and target_algebra.get("fields_fqcn_values")
            == [GCR_PACRE_FIELDS_FQCN]
            and isinstance(
                target_algebra.get("target_summary"),
                Mapping,
            )
            and target_algebra["target_summary"].get(
                "all_complete_fields_validated"
            )
            is True
            and target_algebra["target_summary"].get(
                "all_float_ledger_tensors_finite"
            )
            is True
        ),
        GCR_PACRE_DR_CHECK_NAMES[5]: exact_witnesses,
        GCR_PACRE_DR_CHECK_NAMES[6]: (
            residual_collision.get("representation")
            == "residual_latent"
            and residual_collision.get("collision_policy")
            == GCR_PACRE_DR_COLLISION_POLICY
            and residual_collision.get("target_coordinate_count", 0)
            > 0
            and residual_collision.get("exact_collision_count") == 0
        ),
        GCR_PACRE_DR_CHECK_NAMES[7]: (
            isinstance(zero, Mapping)
            and zero.get("target_state_count")
            == GCR_PACRE_DR_TARGET_STATE_COUNT
            and zero.get("all_target_states_exact_anchor") is True
            and _is_sha256(zero.get("rows_fingerprint"))
            and readout.get("fixed_readout_policy")
            == GCR_PACRE_DR_FIXED_READOUT_POLICY
            and _is_sha256(
                readout.get("fixed_readout_fingerprint")
            )
            and readout.get("fixed_readout_mechanism_nonzero") is True
        ),
        GCR_PACRE_DR_CHECK_NAMES[8]: initial_gradient_passed,
        GCR_PACRE_DR_CHECK_NAMES[9]: direction_passed,
        GCR_PACRE_DR_CHECK_NAMES[10]: _preservation_passed(
            preservation_11
        ),
        GCR_PACRE_DR_CHECK_NAMES[11]: _scope_passed(
            scope_12,
            execution_kind=execution_kind,
        ),
        GCR_PACRE_DR_CHECK_NAMES[12]: (
            context_algebra.get("context_row_count")
            == GCR_PACRE_DR_CONTEXT_STATE_COUNT
            and context_algebra.get("fields_fqcn_values")
            == [GCR_PACRE_FIELDS_FQCN]
            and isinstance(
                context_algebra.get("context_summary"),
                Mapping,
            )
            and context_algebra["context_summary"].get(
                "all_complete_fields_validated"
            )
            is True
            and context_algebra["context_summary"].get(
                "all_float_ledger_tensors_finite"
            )
            is True
        ),
        GCR_PACRE_DR_CHECK_NAMES[13]: (
            isinstance(common_counts, Mapping)
            and set(common_counts)
            == {
                "actual_common_hidden",
                "flipped_common_hidden",
                "actual_common_energy",
                "flipped_common_energy",
                "common_even_energy",
                "common_gate_nonunit",
            }
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value > 0
                for value in common_counts.values()
            )
            and common.get("target_state_count")
            == GCR_PACRE_DR_TARGET_STATE_COUNT
            and common.get("context_state_count")
            == GCR_PACRE_DR_CONTEXT_STATE_COUNT
        ),
        GCR_PACRE_DR_CHECK_NAMES[14]: (
            parity.get("all_target_flip_parity_exact") is True
            and isinstance(parity_rows, list)
            and len(parity_rows) == GCR_PACRE_DR_TARGET_STATE_COUNT
            and all(
                isinstance(row, Mapping)
                and isinstance(row.get("flip_parity"), Mapping)
                and all(
                    row["flip_parity"].get(name) is True
                    for name in (
                        "occupancy_toggled_exactly_once",
                        "fp64_residual_exact_odd",
                        "fp64_common_even_exact_even",
                        "fp64_gate_exact_even",
                        "fp64_gated_interaction_exact_odd",
                    )
                )
                for row in parity_rows
            )
        ),
        GCR_PACRE_DR_CHECK_NAMES[15]: (
            common_only.get("residual_element_count", 0) > 0
            and common_only.get("residual_exact_zero_count")
            == common_only.get("residual_element_count")
            and common_only.get("common_even_nonzero_count", 0) > 0
            and common_only.get("gate_nonunit_count", 0) > 0
            and common_only.get(
                "gated_interaction_exact_zero_count"
            )
            == common_only.get("residual_element_count")
            and common_only.get("field_exact_anchor") is True
            and isinstance(
                common_only.get("fast_fp64_field_comparison"),
                Mapping,
            )
            and common_only["fast_fp64_field_comparison"].get(
                "passed"
            )
            is True
            and isinstance(common_restore, Mapping)
            and common_restore.get("restored_exactly") is True
        ),
        GCR_PACRE_DR_CHECK_NAMES[16]: target_path_passed,
        GCR_PACRE_DR_CHECK_NAMES[17]: (
            gated_collision.get("representation") == "gated_latent"
            and gated_collision.get("collision_policy")
            == GCR_PACRE_DR_COLLISION_POLICY
            and gated_collision.get("target_coordinate_count", 0) > 0
            and gated_collision.get("exact_collision_count") == 0
        ),
        GCR_PACRE_DR_CHECK_NAMES[18]: _preservation_passed(
            preservation_19
        ),
        GCR_PACRE_DR_CHECK_NAMES[19]: _scope_passed(
            scope_20,
            execution_kind=execution_kind,
        ),
        GCR_PACRE_DR_CHECK_NAMES[20]: (
            gate_distribution.get("threshold_or_ratio_gate") is None
            and gate_distribution.get("state_count")
            == (
                GCR_PACRE_DR_TARGET_STATE_COUNT
                + GCR_PACRE_DR_CONTEXT_STATE_COUNT
            )
            and isinstance(
                gate_distribution.get("element_count"),
                int,
            )
            and gate_distribution.get("element_count", 0) > 0
            and gate_distribution.get("zero_count", 0)
            + gate_distribution.get("two_count", 0)
            + gate_distribution.get("interior_count", 0)
            == gate_distribution.get("element_count")
            and all(
                isfinite(
                    float.fromhex(
                        str(gate_distribution.get(name))
                    )
                )
                for name in (
                    "minimum_hex",
                    "maximum_hex",
                    "mean_hex",
                )
            )
            and _is_sha256(
                gate_distribution.get(
                    "per_state_rows_fingerprint"
                )
            )
        ),
        GCR_PACRE_DR_CHECK_NAMES[21]: (
            fast_full.get("fields_fqcn") == GCR_PACRE_FIELDS_FQCN
            and fast_full.get(
                "target_complete_validator_call_count"
            )
            == GCR_PACRE_DR_TARGET_STATE_COUNT
            and fast_full.get(
                "context_complete_validator_call_count"
            )
            == GCR_PACRE_DR_CONTEXT_STATE_COUNT
            and fast_full.get("target_fast_fp64_agreement_count")
            == GCR_PACRE_DR_TARGET_STATE_COUNT
            and fast_full.get("context_fast_fp64_agreement_count")
            == GCR_PACRE_DR_CONTEXT_STATE_COUNT
            and _is_sha256(fast_full.get("ledger_fingerprint"))
        ),
        GCR_PACRE_DR_CHECK_NAMES[22]: efficiency_passed,
    }
    if (
        tuple(checks) != GCR_PACRE_DR_CHECK_NAMES
        or any(type(value) is not bool for value in checks.values())
    ):
        raise AssertionError("D_R check derivation contract changed")
    return tuple(checks.items())


def _decision_payload(
    checks: tuple[tuple[str, bool], ...],
    *,
    execution_kind: str,
) -> dict[str, object]:
    failed = [name for name, passed in checks if not passed]
    body: dict[str, object] = {
        "schema_version": (
            "cure-lite-v24-gcr-pacre-D_R-structural-decision-v1"
        ),
        "candidate": GCR_PACRE_CANDIDATE,
        "execution_kind": execution_kind,
        "check_names": list(GCR_PACRE_DR_CHECK_NAMES),
        "checks_fingerprint": stable_fingerprint(dict(checks)),
        "gate_passed": not failed,
        "failed_checks": failed,
        "decision": (
            GCR_PACRE_DR_PASS_DECISION
            if not failed
            else GCR_PACRE_DR_FAIL_DECISION
        ),
        "performance_claim_supported": False,
        "performance_threshold_present": False,
        "D_V_execution_authorized": False,
        "D_T_execution_authorized": False,
        "formal_800_execution_authorized": False,
    }
    return {**body, "decision_fingerprint": stable_fingerprint(body)}


def _verify_decision(
    decision: object,
    checks: tuple[tuple[str, bool], ...],
    *,
    execution_kind: str,
) -> None:
    if not isinstance(decision, Mapping):
        raise TypeError("D_R decision must be a mapping")
    if dict(decision) != _decision_payload(
        checks,
        execution_kind=execution_kind,
    ):
        raise ValueError("D_R decision differs from raw observations")


def run_gcr_pacre_generated_dr_contract_audit(
    *,
    dataset_free_receipt_path: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Exercise all structural mechanics on generated 32/96 fixtures."""

    resolved = _resolve_device(device)
    dataset_path = _canonical_regular_file(
        dataset_free_receipt_path,
        name="dataset-free receipt",
    )
    dataset_receipt = _read_strict_json(
        dataset_path,
        name="dataset-free receipt",
    )
    dataset_fingerprint = verify_gcr_pacre_dataset_free_receipt(
        dataset_receipt
    )
    efficiency = _efficiency_observation(dataset_receipt)
    adapter = _generated_population_adapter()
    token = _GeneratedAuditToken(
        adapter_fingerprint=adapter.adapter_fingerprint,
        _issuer=_GENERATED_AUDIT_ISSUER,
    )
    probe = _run_probe(
        adapter,
        device=resolved,
        execution_token=token,
    )
    prerequisite = {
        "official_verifier_called": True,
        "dataset_free_receipt_fingerprint": dataset_fingerprint,
        "dataset_free_receipt_file_sha256": file_sha256(
            dataset_path
        ),
        "dataset_free_gate_passed": True,
        "D_R_authorized_by_dataset_free": False,
        "efficiency_section_fingerprint": efficiency[
            "section_fingerprint"
        ],
    }
    input_binding = {
        "execution_kind": "generated",
        "split": adapter.split,
        "seed": adapter.seed,
        "target_state_count": len(adapter.target_states),
        "context_state_count": len(adapter.context_states),
        "adapter_fingerprint": adapter.adapter_fingerprint,
        "population_fingerprint": adapter.population_fingerprint,
        "cache_fingerprint": adapter.cache_fingerprint,
        "source_cache_fingerprint": (
            adapter.source_cache_fingerprint
        ),
        "real_dataset_accessed": False,
        "real_preaccess_token_used": False,
        "authorization_matches_live_inputs": None,
    }
    raw = _raw_observations(
        probe=probe,
        prerequisite=prerequisite,
        input_binding=input_binding,
        efficiency=efficiency,
    )
    checks = recompute_gcr_pacre_dr_checks(
        raw,
        execution_kind="generated",
    )
    implementation = _implementation_binding()
    decision = _decision_payload(
        checks,
        execution_kind="generated",
    )
    body: dict[str, object] = {
        "schema_version": GCR_PACRE_DR_GENERATED_SCHEMA,
        "run_id": GCR_PACRE_DR_RUN_ID,
        "candidate": GCR_PACRE_CANDIDATE,
        "execution_kind": "generated",
        "execution_seed": GCR_PACRE_DR_EXECUTION_SEED,
        "device": str(resolved),
        "dataset_free_receipt_fingerprint": dataset_fingerprint,
        "dataset_free_receipt_file_sha256": file_sha256(
            dataset_path
        ),
        "efficiency_section_fingerprint": efficiency[
            "section_fingerprint"
        ],
        "efficiency_receipt_sha256": efficiency[
            "efficiency_receipt_sha256"
        ],
        "implementation_binding": dict(implementation),
        "source_closure_fingerprint": stable_fingerprint(
            dict(implementation)
        ),
        "adapter_fingerprint": adapter.adapter_fingerprint,
        "population_fingerprint": adapter.population_fingerprint,
        "cache_fingerprint": adapter.cache_fingerprint,
        "raw_observations": raw,
        "raw_observations_fingerprint": stable_fingerprint(raw),
        "checks": dict(checks),
        "decision": decision,
        "boundary": probe["boundary"],
        "real_preaccess_token_issued": False,
        "real_D_R_execution_authorized": False,
    }
    receipt = {**body, "receipt_fingerprint": stable_fingerprint(body)}
    verify_gcr_pacre_generated_dr_contract_receipt(receipt)
    return receipt


def verify_gcr_pacre_generated_dr_contract_receipt(
    receipt: Mapping[str, object],
) -> str:
    if not isinstance(receipt, Mapping):
        raise TypeError("generated D_R receipt must be a mapping")
    payload = dict(receipt)
    fingerprint = payload.pop("receipt_fingerprint", None)
    if (
        not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(payload)
        or payload.get("schema_version")
        != GCR_PACRE_DR_GENERATED_SCHEMA
        or payload.get("candidate") != GCR_PACRE_CANDIDATE
        or payload.get("execution_kind") != "generated"
        or payload.get("execution_seed")
        != GCR_PACRE_DR_EXECUTION_SEED
        or payload.get("real_preaccess_token_issued") is not False
        or payload.get("real_D_R_execution_authorized") is not False
    ):
        raise ValueError("generated D_R receipt identity changed")
    implementation = payload.get("implementation_binding")
    if (
        not isinstance(implementation, Mapping)
        or dict(implementation) != dict(_implementation_binding())
        or payload.get("source_closure_fingerprint")
        != stable_fingerprint(dict(implementation))
    ):
        raise ValueError("generated D_R source closure changed")
    raw = payload.get("raw_observations")
    if (
        not isinstance(raw, Mapping)
        or payload.get("raw_observations_fingerprint")
        != stable_fingerprint(dict(raw))
    ):
        raise ValueError("generated D_R raw observations changed")
    checks = recompute_gcr_pacre_dr_checks(
        raw,
        execution_kind="generated",
    )
    if payload.get("checks") != dict(checks):
        raise ValueError("generated D_R checks changed")
    _verify_decision(
        payload.get("decision"),
        checks,
        execution_kind="generated",
    )
    boundary = payload.get("boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("D_R_accessed") is not False
        or boundary.get("D_V_accessed") is not False
        or boundary.get("D_T_accessed") is not False
        or boundary.get("training_performed") is not False
    ):
        raise PermissionError("generated D_R boundary changed")
    return str(fingerprint)


def run_gcr_pacre_dr_gate(
    *,
    preaccess_token: GCRPACREDRPreaccessToken,
    run_start_token: GCRPACREDRRunStartToken,
    real_inputs: CoverageStateRealDRInputs,
    bounded_population: CoverageStateBoundedPopulation,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    """Consume one private token and run the real zero-update ``D_R`` gate."""

    if not _is_live_real_preaccess_token(preaccess_token):
        raise PermissionError(
            "real D_R requires the fixed verifier's private token"
        )
    run_start_payload = _verify_real_run_start_token(
        preaccess_token,
        run_start_token,
    )
    resolved = _resolve_device(device)
    if run_start_token.requested_device != str(resolved):
        raise PermissionError(
            "real D_R device differs from persistent run intent"
        )
    token_id = id(preaccess_token)
    if token_id not in _CONSUMED_REAL_PREACCESS_TOKENS:
        raise PermissionError(
            "real D_R loader materialization was not claimed"
        )
    if token_id in _STARTED_REAL_GATE_TOKENS:
        raise PermissionError("real D_R gate token was already executed")
    _STARTED_REAL_GATE_TOKENS.add(token_id)
    implementation = _implementation_binding()
    if (
        preaccess_token.source_closure_fingerprint
        != stable_fingerprint(dict(implementation))
        or preaccess_token.protocol_preregistration_fingerprint
        != _protocol_preregistration_fingerprint()
    ):
        raise PermissionError("real D_R source closure changed after authorization")
    adapter = _real_population_adapter(
        real_inputs,
        bounded_population,
    )
    authorization_matches = (
        real_inputs.source_binding.binding_fingerprint
        == preaccess_token.expected_source_binding_fingerprint
        and real_inputs.source_binding.manifest_file_sha256
        == preaccess_token.expected_manifest_file_sha256
        and real_inputs.source_binding.state_index_file_sha256
        == preaccess_token.expected_state_index_file_sha256
        and real_inputs.build_fingerprint
        == preaccess_token.expected_real_inputs_fingerprint
        and bounded_population.population_fingerprint
        == preaccess_token.expected_population_fingerprint
        and bounded_population.cache.cache_fingerprint
        == preaccess_token.expected_cache_fingerprint
    )
    if not authorization_matches:
        raise PermissionError(
            "live D_R graph differs from external authorization"
        )
    probe = _run_probe(
        adapter,
        device=resolved,
        execution_token=preaccess_token,
    )
    real_inputs.verify_unchanged()
    bounded_population.verify_unchanged()
    efficiency = json.loads(
        preaccess_token.efficiency_observation_json
    )
    if not isinstance(efficiency, Mapping):
        raise AssertionError("preaccess efficiency token changed")
    prerequisite = {
        "official_verifier_called": True,
        "dataset_free_receipt_fingerprint": (
            preaccess_token.dataset_free_receipt_fingerprint
        ),
        "dataset_free_receipt_file_sha256": (
            preaccess_token.dataset_free_receipt_file_sha256
        ),
        "dataset_free_gate_passed": True,
        "D_R_authorized_by_dataset_free": False,
        "efficiency_section_fingerprint": (
            preaccess_token.efficiency_section_fingerprint
        ),
    }
    input_binding = {
        "execution_kind": "real_D_R",
        "split": adapter.split,
        "seed": adapter.seed,
        "target_state_count": len(adapter.target_states),
        "context_state_count": len(adapter.context_states),
        "adapter_fingerprint": adapter.adapter_fingerprint,
        "population_fingerprint": adapter.population_fingerprint,
        "cache_fingerprint": adapter.cache_fingerprint,
        "source_cache_fingerprint": (
            adapter.source_cache_fingerprint
        ),
        "real_dataset_accessed": True,
        "real_preaccess_token_used": True,
        "authorization_matches_live_inputs": authorization_matches,
        "source_binding_fingerprint": (
            real_inputs.source_binding.binding_fingerprint
        ),
        "manifest_file_sha256": (
            real_inputs.source_binding.manifest_file_sha256
        ),
        "state_index_file_sha256": (
            real_inputs.source_binding.state_index_file_sha256
        ),
        "real_inputs_fingerprint": real_inputs.build_fingerprint,
    }
    raw = _raw_observations(
        probe=probe,
        prerequisite=prerequisite,
        input_binding=input_binding,
        efficiency=efficiency,
    )
    checks = recompute_gcr_pacre_dr_checks(
        raw,
        execution_kind="real_D_R",
    )
    decision = _decision_payload(
        checks,
        execution_kind="real_D_R",
    )
    body: dict[str, object] = {
        "schema_version": GCR_PACRE_DR_GATE_SCHEMA,
        "run_id": GCR_PACRE_DR_RUN_ID,
        "candidate": GCR_PACRE_CANDIDATE,
        "execution_kind": "real_D_R",
        "execution_seed": GCR_PACRE_DR_EXECUTION_SEED,
        "device": str(resolved),
        "requested_receipt_output": (
            run_start_token.requested_receipt_output
        ),
        "dataset_free_receipt_fingerprint": (
            preaccess_token.dataset_free_receipt_fingerprint
        ),
        "dataset_free_receipt_file_sha256": (
            preaccess_token.dataset_free_receipt_file_sha256
        ),
        "efficiency_section_fingerprint": (
            preaccess_token.efficiency_section_fingerprint
        ),
        "efficiency_receipt_sha256": (
            preaccess_token.efficiency_receipt_sha256
        ),
        "preaccess_authorization_fingerprint": (
            preaccess_token.authorization_fingerprint
        ),
        "preaccess_authorization_file_sha256": (
            preaccess_token.authorization_receipt_file_sha256
        ),
        "access_audit_receipt_fingerprint": (
            preaccess_token.access_audit_receipt_fingerprint
        ),
        "access_audit_receipt_file_sha256": (
            preaccess_token.access_audit_receipt_file_sha256
        ),
        "protocol_preregistration_fingerprint": (
            preaccess_token.protocol_preregistration_fingerprint
        ),
        "implementation_binding": dict(implementation),
        "source_closure_fingerprint": stable_fingerprint(
            dict(implementation)
        ),
        "source_binding_fingerprint": (
            real_inputs.source_binding.binding_fingerprint
        ),
        "real_inputs_fingerprint": real_inputs.build_fingerprint,
        "population_fingerprint": (
            bounded_population.population_fingerprint
        ),
        "cache_fingerprint": (
            bounded_population.cache.cache_fingerprint
        ),
        "adapter_fingerprint": adapter.adapter_fingerprint,
        "run_start_marker": {
            "path": run_start_token.marker_path,
            "file_sha256": run_start_token.marker_file_sha256,
            "marker_fingerprint": (
                run_start_token.marker_fingerprint
            ),
            "payload": run_start_payload,
        },
        "artifact_hashes": {
            "dataset_free_receipt": (
                preaccess_token.dataset_free_receipt_file_sha256
            ),
            "preaccess_authorization": (
                preaccess_token.authorization_receipt_file_sha256
            ),
            "preaccess_access_audit": (
                preaccess_token.access_audit_receipt_file_sha256
            ),
            "persistent_run_start_marker": (
                run_start_token.marker_file_sha256
            ),
        },
        "raw_observations": raw,
        "raw_observations_fingerprint": stable_fingerprint(raw),
        "checks": dict(checks),
        "decision": decision,
        "boundary": probe["boundary"],
    }
    receipt = {**body, "receipt_fingerprint": stable_fingerprint(body)}
    verify_gcr_pacre_dr_receipt(
        receipt,
        real_inputs=real_inputs,
        bounded_population=bounded_population,
    )
    return receipt


def verify_gcr_pacre_dr_receipt(
    receipt: Mapping[str, object],
    *,
    real_inputs: CoverageStateRealDRInputs | None = None,
    bounded_population: CoverageStateBoundedPopulation | None = None,
) -> str:
    """Verify a sealed real receipt without rerunning the structural probe."""

    if not isinstance(receipt, Mapping):
        raise TypeError("real D_R receipt must be a mapping")
    payload = dict(receipt)
    fingerprint = payload.pop("receipt_fingerprint", None)
    if (
        not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(payload)
        or payload.get("schema_version") != GCR_PACRE_DR_GATE_SCHEMA
        or payload.get("run_id") != GCR_PACRE_DR_RUN_ID
        or payload.get("candidate") != GCR_PACRE_CANDIDATE
        or payload.get("execution_kind") != "real_D_R"
        or payload.get("execution_seed")
        != GCR_PACRE_DR_EXECUTION_SEED
    ):
        raise ValueError("real D_R receipt identity changed")
    implementation = payload.get("implementation_binding")
    if (
        not isinstance(implementation, Mapping)
        or dict(implementation) != dict(_implementation_binding())
        or payload.get("source_closure_fingerprint")
        != stable_fingerprint(dict(implementation))
    ):
        raise ValueError("real D_R source closure changed")
    run_start = payload.get("run_start_marker")
    marker = (
        run_start.get("payload")
        if isinstance(run_start, Mapping)
        else None
    )
    if not isinstance(marker, Mapping):
        raise PermissionError("real D_R run-start marker is absent")
    marker_body = dict(marker)
    marker_fingerprint = marker_body.pop(
        "marker_fingerprint",
        None,
    )
    intent = marker.get("intent")
    artifact_hashes = payload.get("artifact_hashes")
    authorization_fingerprint = payload.get(
        "preaccess_authorization_fingerprint"
    )
    required_marker_path = (
        _required_run_start_marker_path_from_fingerprint(
            str(authorization_fingerprint)
        )
        if _is_sha256(authorization_fingerprint)
        else None
    )
    if (
        set(run_start)
        != {"path", "file_sha256", "marker_fingerprint", "payload"}
        or marker.get("schema_version")
        != GCR_PACRE_DR_RUN_START_SCHEMA
        or marker.get("path_policy")
        != GCR_PACRE_DR_RUN_START_PATH_POLICY
        or marker.get("stage_id")
        != GCR_PACRE_DR_PREACCESS_STAGE_ID
        or marker.get("run_id") != GCR_PACRE_DR_RUN_ID
        or marker.get("candidate") != GCR_PACRE_CANDIDATE
        or not _is_sha256(marker_fingerprint)
        or marker_fingerprint != stable_fingerprint(marker_body)
        or run_start.get("marker_fingerprint")
        != marker_fingerprint
        or marker.get("authorization_fingerprint")
        != authorization_fingerprint
        or marker.get("authorization_receipt_file_sha256")
        != payload.get("preaccess_authorization_file_sha256")
        or marker.get("access_audit_receipt_fingerprint")
        != payload.get("access_audit_receipt_fingerprint")
        or marker.get("access_audit_receipt_file_sha256")
        != payload.get("access_audit_receipt_file_sha256")
        or marker.get("source_closure_fingerprint")
        != payload.get("source_closure_fingerprint")
        or marker.get("implementation_binding")
        != dict(implementation)
        or required_marker_path is None
        or marker.get("marker_path") != str(required_marker_path)
        or run_start.get("path") != str(required_marker_path)
        or not _is_sha256(run_start.get("file_sha256"))
        or not isinstance(intent, Mapping)
        or intent.get("execution_kind") != "real_D_R"
        or intent.get("split") != "D_R"
        or intent.get("requested_device") != payload.get("device")
        or intent.get("requested_receipt_output")
        != payload.get("requested_receipt_output")
        or intent.get("D_R_materialization_intended") is not True
        or intent.get("D_V_materialization_intended") is not False
        or intent.get("D_T_materialization_intended") is not False
        or intent.get("optimizer_steps_authorized") != 0
        or intent.get("parameter_updates_authorized") != 0
        or intent.get("training_authorized") is not False
        or marker.get("intent_fingerprint")
        != stable_fingerprint(dict(intent))
        or not isinstance(artifact_hashes, Mapping)
        or set(artifact_hashes)
        != {
            "dataset_free_receipt",
            "preaccess_authorization",
            "preaccess_access_audit",
            "persistent_run_start_marker",
        }
        or artifact_hashes.get("dataset_free_receipt")
        != payload.get("dataset_free_receipt_file_sha256")
        or artifact_hashes.get("preaccess_authorization")
        != payload.get("preaccess_authorization_file_sha256")
        or artifact_hashes.get("persistent_run_start_marker")
        != run_start.get("file_sha256")
        or artifact_hashes.get("preaccess_access_audit")
        != payload.get("access_audit_receipt_file_sha256")
    ):
        raise PermissionError("real D_R run-start binding changed")
    marker_path = _canonical_regular_file(
        required_marker_path,
        name="real D_R persistent run-start marker",
    )
    if (
        marker_path.stat().st_nlink != 1
        or marker_path.stat().st_mode & 0o222
        or file_sha256(marker_path) != run_start["file_sha256"]
        or _read_strict_json(
            marker_path,
            name="real D_R persistent run-start marker",
        )
        != dict(marker)
    ):
        raise PermissionError("real D_R persistent marker bytes changed")
    raw = payload.get("raw_observations")
    if (
        not isinstance(raw, Mapping)
        or payload.get("raw_observations_fingerprint")
        != stable_fingerprint(dict(raw))
    ):
        raise ValueError("real D_R raw observations changed")
    checks = recompute_gcr_pacre_dr_checks(
        raw,
        execution_kind="real_D_R",
    )
    if payload.get("checks") != dict(checks):
        raise ValueError("real D_R checks changed")
    _verify_decision(
        payload.get("decision"),
        checks,
        execution_kind="real_D_R",
    )
    boundary = payload.get("boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("D_R_accessed") is not True
        or boundary.get("D_V_accessed") is not False
        or boundary.get("D_T_accessed") is not False
        or boundary.get("optimizer_steps") != 0
        or boundary.get("parameter_updates") != 0
        or boundary.get("training_performed") is not False
    ):
        raise PermissionError("real D_R receipt boundary changed")
    if (real_inputs is None) is not (bounded_population is None):
        raise TypeError(
            "real_inputs and bounded_population must be supplied together"
        )
    if real_inputs is not None and bounded_population is not None:
        real_inputs.verify_unchanged()
        bounded_population.verify_unchanged()
        if (
            bounded_population.source_cache
            is not real_inputs.scalar_cache
            or payload.get("source_binding_fingerprint")
            != real_inputs.source_binding.binding_fingerprint
            or payload.get("real_inputs_fingerprint")
            != real_inputs.build_fingerprint
            or payload.get("population_fingerprint")
            != bounded_population.population_fingerprint
            or payload.get("cache_fingerprint")
            != bounded_population.cache.cache_fingerprint
        ):
            raise RuntimeError("real D_R receipt input graph changed")
    return str(fingerprint)


__all__ = [
    "GCR_PACRE_DR_ACCESS_AUDIT_PATH",
    "GCR_PACRE_DR_ACCESS_AUDIT_SCHEMA",
    "GCR_PACRE_DR_CHECK_NAMES",
    "GCR_PACRE_DR_CONTEXT_STATE_COUNT",
    "GCR_PACRE_DR_DATASET_FREE_RECEIPT_R2_PATH",
    "GCR_PACRE_DR_EXECUTION_SEED",
    "GCR_PACRE_DR_FAIL_DECISION",
    "GCR_PACRE_DR_GATE_SCHEMA",
    "GCR_PACRE_DR_PASS_DECISION",
    "GCR_PACRE_DR_PREACCESS_SCHEMA",
    "GCR_PACRE_DR_PREACCESS_AUTHORIZATION_PATH",
    "GCR_PACRE_DR_PREACCESS_STAGE_ID",
    "GCR_PACRE_DR_PREACCESS_STATUS",
    "GCR_PACRE_DR_RUN_START_PATH_POLICY",
    "GCR_PACRE_DR_RUN_START_SCHEMA",
    "GCR_PACRE_DR_RUN_ID",
    "GCR_PACRE_DR_RECEIPT_PATH",
    "GCR_PACRE_DR_SOURCE_PATHS",
    "GCR_PACRE_DR_TARGET_STATE_COUNT",
    "GCRPACREDRPreaccessToken",
    "GCRPACREDRRunStartToken",
    "begin_gcr_pacre_dr_materialization",
    "build_gcr_pacre_dr_preaccess_artifacts",
    "create_gcr_pacre_dr_run_start_marker",
    "recompute_gcr_pacre_dr_checks",
    "run_gcr_pacre_dr_gate",
    "run_gcr_pacre_generated_dr_contract_audit",
    "required_gcr_pacre_dr_run_start_marker_path",
    "verify_and_issue_gcr_pacre_dr_preaccess",
    "verify_gcr_pacre_dr_receipt",
    "verify_gcr_pacre_generated_dr_contract_receipt",
]
