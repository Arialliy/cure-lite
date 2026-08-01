#!/usr/bin/env python3
"""Append-only closure for CURE-Lite v24 runtime compatibility generation c5.

This module is deliberately not a scientific runner.  It can only authorize
and seal the metadata-only c5 compatibility lane after the c4 B4 receipt-seal
failure has been terminalized.  It never imports
torch, opens a dataset, starts a unit, or creates a runtime specification.

The c5 lane retains scientific attempt ordinal 2 and the original scientific
output paths.  All old, c1, c2, c3, c4, and scientific runtime paths remain
protected.  The c4 terminal is a valid sealed failure boundary, never a PASS.
A compatibility receipt is archival: its short authorization may expire after
sealing, but the receipt chronology must remain inside the original
authorization window.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from types import ModuleType
from typing import Callable, Mapping, Sequence


REPOSITORY = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = (
    REPOSITORY / "protocols/IRSTD-1K/gcr_pacre_v24/runtime_evidence_r2"
).resolve()
RUNS_ROOT = (REPOSITORY / "runs/irstd1k_stage_a_seed42").resolve()

CANDIDATE = "GCR-PACRE-v24"
STAGE_ID = "gcr_pacre_v24_D_R_structural_r2"
SCIENTIFIC_ATTEMPT_ID = "gcr_pacre_v24_D_R_zero_update_structural_r2"
SCIENTIFIC_ATTEMPT_ORDINAL = 2
RUNTIME_COMPATIBILITY_ID = "c5"
INSTRUCTION_ID = "user-2026-07-31-modify-after-c4-failure-v1"
AUTHORIZATION_BASIS = "user instruction: 修改后继续"

OLD_UNIT_NAME = "cure-lite-v24-gcr-pacre-dr-r2.service"
C1_UNIT_NAME = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c1.service"
)
C2_UNIT_NAME = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c2.service"
)
C3_UNIT_NAME = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c3.service"
)
C4_UNIT_NAME = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c4.service"
)
C5_UNIT_NAME = (
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c5.service"
)
USER_SYSTEMD_UNIT_DIRECTORY = Path(
    f"/run/user/{os.getuid()}/systemd/user"
)
OLD_UNIT_FRAGMENT_PATH = USER_SYSTEMD_UNIT_DIRECTORY / OLD_UNIT_NAME
C1_UNIT_FRAGMENT_PATH = USER_SYSTEMD_UNIT_DIRECTORY / C1_UNIT_NAME
C2_UNIT_FRAGMENT_PATH = USER_SYSTEMD_UNIT_DIRECTORY / C2_UNIT_NAME
C3_UNIT_FRAGMENT_PATH = USER_SYSTEMD_UNIT_DIRECTORY / C3_UNIT_NAME
C4_UNIT_FRAGMENT_PATH = USER_SYSTEMD_UNIT_DIRECTORY / C4_UNIT_NAME
C5_UNIT_FRAGMENT_PATH = USER_SYSTEMD_UNIT_DIRECTORY / C5_UNIT_NAME

C1_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c1_authorization.json"
)
C1_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c1_receipt.json"
)
C1_FAILURE_TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_schema_compat_c1_expired_prewrite_terminal.json"
)
C2_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c2_authorization.json"
)
C2_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c2_receipt.json"
)
C2_PREWRITE_FAILURE_TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_schema_compat_c2_prewrite_failure_terminal.json"
)
C2_MODE_CONTRACT_FAILURE_TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_schema_compat_c2_mode_contract_failure_terminal.json"
)
C3_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c3_authorization.json"
)
C3_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c3_receipt.json"
)
C3_ENVIRONMENT_FAILURE_TERMINAL_PATH = (
    EVIDENCE_ROOT
    / (
        "r2_preaccess_schema_compat_c3_"
        "environment_stability_failure_terminal.json"
    )
)
C4_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c4_authorization.json"
)
C4_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c4_receipt.json"
)
C4_RECEIPT_SEAL_FAILURE_TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_schema_compat_c4_receipt_seal_failure_terminal.json"
)
C5_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c5_authorization.json"
)
C5_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c5_receipt.json"
)
C5_TERMINAL_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_schema_compat_c5_terminal.json"
)

R10_ROOT = (
    EVIDENCE_ROOT
    / "supervisor_v2_systemd_integration_preaccess_compat_c1_r10"
)
R10_AUTHORIZATION_PATH = R10_ROOT / "control/authorization.json"
R10_RECEIPT_PATH = R10_ROOT / "control/integration-receipt.json"
R10_TERMINAL_PATH = R10_ROOT / "control/integration-terminal.json"
R10_REMOVAL_STATE_PATH = R10_ROOT / "control/removal-state.json"

C3_ENVIRONMENT_POLICY_PATH = (
    EVIDENCE_ROOT / "runtime_environment_policy_preaccess_compat_c3.json"
)
C3_ENVIRONMENT_STABILITY_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_receipt_preaccess_compat_c3.json"
)
C3_ENVIRONMENT_POSTCLEANUP_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_postcleanup_receipt_preaccess_compat_c3.json"
)
C3_UNIT_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c3_unit_realization_authorization.json"
)
C3_UNIT_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c3_unit_realization_receipt.json"
)
C3_UNIT_TERMINAL_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c3_unit_realization_terminal.json"
)

C3_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c3_runtime_spec.json"
)
C3_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / (
        "D_R_structural_attempt_r2_preaccess_compat_c3_"
        "runtime_launch_authorization.json"
    )
)
C3_RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c3_runtime_artifacts"
)
C3_GPU_LEASE_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c3_gpu_lease"
)
C3_RUN_ROOT_ALIAS_PATH = (
    RUNS_ROOT / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c3"
)
C3_RESULT_RECEIPT_ALIAS_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c3_receipt.json"
)

C4_ENVIRONMENT_POLICY_PATH = (
    EVIDENCE_ROOT / "runtime_environment_policy_preaccess_compat_c4.json"
)
C4_ENVIRONMENT_SCOPE_HANDOFF_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_scope_handoff_preaccess_compat_c4.json"
)
C4_ENVIRONMENT_STABILITY_ATTEMPT_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_attempt_preaccess_compat_c4.json"
)
C4_ENVIRONMENT_STABILITY_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_receipt_preaccess_compat_c4.json"
)
C4_ENVIRONMENT_STABILITY_TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_terminal_preaccess_compat_c4.json"
)
C4_ENVIRONMENT_POSTCLEANUP_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_postcleanup_receipt_preaccess_compat_c4.json"
)
C4_UNIT_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c4_unit_realization_authorization.json"
)
C4_UNIT_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c4_unit_realization_receipt.json"
)
C4_UNIT_TERMINAL_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c4_unit_realization_terminal.json"
)
C4_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c4_runtime_spec.json"
)
C4_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / (
        "D_R_structural_attempt_r2_preaccess_compat_c4_"
        "runtime_launch_authorization.json"
    )
)
C4_RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c4_runtime_artifacts"
)
C4_GPU_LEASE_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c4_gpu_lease"
)
C4_RUN_ROOT_ALIAS_PATH = (
    RUNS_ROOT / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c4"
)
C4_RESULT_RECEIPT_ALIAS_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c4_receipt.json"
)

C5_ENVIRONMENT_POLICY_PATH = (
    EVIDENCE_ROOT / "runtime_environment_policy_preaccess_compat_c5.json"
)
C5_ENVIRONMENT_STABILITY_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_receipt_preaccess_compat_c5.json"
)
C5_ENVIRONMENT_SCOPE_HANDOFF_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_scope_handoff_preaccess_compat_c5.json"
)
C5_ENVIRONMENT_STABILITY_ATTEMPT_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_attempt_preaccess_compat_c5.json"
)
C5_ENVIRONMENT_STABILITY_TERMINAL_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_terminal_preaccess_compat_c5.json"
)
C5_ENVIRONMENT_POSTCLEANUP_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_postcleanup_receipt_preaccess_compat_c5.json"
)
C5_UNIT_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c5_unit_realization_authorization.json"
)
C5_UNIT_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c5_unit_realization_receipt.json"
)
C5_UNIT_TERMINAL_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c5_unit_realization_terminal.json"
)

C5_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c5_runtime_spec.json"
)
C5_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / (
        "D_R_structural_attempt_r2_preaccess_compat_c5_"
        "runtime_launch_authorization.json"
    )
)
C5_RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c5_runtime_artifacts"
)
C5_GPU_LEASE_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c5_gpu_lease"
)
C5_RUN_ROOT_ALIAS_PATH = (
    RUNS_ROOT / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c5"
)
C5_RESULT_RECEIPT_ALIAS_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c5_receipt.json"
)

C2_ENVIRONMENT_POLICY_PATH = (
    EVIDENCE_ROOT / "runtime_environment_policy_preaccess_compat_c2.json"
)
C2_ENVIRONMENT_STABILITY_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_stability_receipt_preaccess_compat_c2.json"
)
C2_ENVIRONMENT_POSTCLEANUP_PATH = (
    EVIDENCE_ROOT
    / "runtime_environment_postcleanup_receipt_preaccess_compat_c2.json"
)
C2_UNIT_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "r2_preaccess_compat_c2_unit_realization_authorization.json"
)
C2_UNIT_RECEIPT_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c2_unit_realization_receipt.json"
)
C2_UNIT_TERMINAL_PATH = (
    EVIDENCE_ROOT / "r2_preaccess_compat_c2_unit_realization_terminal.json"
)
C2_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c2_runtime_spec.json"
)
C2_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / (
        "D_R_structural_attempt_r2_preaccess_compat_c2_"
        "runtime_launch_authorization.json"
    )
)
C2_RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c2_runtime_artifacts"
)
C2_GPU_LEASE_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c2_gpu_lease"
)
C2_RUN_ROOT_ALIAS_PATH = (
    RUNS_ROOT / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c2"
)
C2_RESULT_RECEIPT_ALIAS_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c2_receipt.json"
)

C1_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_spec.json"
)
C1_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / (
        "D_R_structural_attempt_r2_preaccess_compat_c1_"
        "runtime_launch_authorization.json"
    )
)
C1_RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_runtime_artifacts"
)
C1_GPU_LEASE_ROOT = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_gpu_lease"
)
C1_RUN_ROOT_ALIAS_PATH = (
    RUNS_ROOT / "gcr_pacre_v24_D_R_structural_attempt_r2_preaccess_compat_c1"
)
C1_RESULT_RECEIPT_ALIAS_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_preaccess_compat_c1_receipt.json"
)

OLD_RUNTIME_SPEC_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_spec.json"
)
OLD_RUNTIME_LAUNCH_AUTHORIZATION_PATH = (
    EVIDENCE_ROOT
    / "D_R_structural_attempt_r2_runtime_launch_authorization.json"
)
OLD_RUNTIME_ARTIFACT_ROOT = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_runtime_artifacts"
)
OLD_GPU_LEASE_ROOT = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_gpu_lease"
)
SCIENTIFIC_RUN_ROOT = (
    RUNS_ROOT / "gcr_pacre_v24_D_R_structural_attempt_r2"
)
SCIENTIFIC_RESULT_RECEIPT_PATH = (
    EVIDENCE_ROOT / "D_R_structural_attempt_r2_receipt.json"
)

C5_BRIDGE_SOURCE_PATH = Path(__file__).resolve()
C4_BRIDGE_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_schema_compatibility_c4.py"
).resolve()
C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SOURCE_PATH = (
    REPOSITORY
    / (
        "tools/cure_lite_v24_preaccess_compat_c4_"
        "receipt_seal_failure_terminal.py"
    )
).resolve()
C1_FAILURE_TERMINALIZER_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_compat_c1_expired_prewrite_terminal.py"
).resolve()
C5_ENVIRONMENT_WRAPPER_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_environment_preaccess_compat_c5.py"
).resolve()
C5_RELEASE_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_actual_runtime_release_preaccess_compat_c5.py"
).resolve()
C5_SUPERVISOR_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_runtime_supervisor_preaccess_compat_c5.py"
).resolve()
C5_ADAPTER_SOURCE_PATH = (
    REPOSITORY
    / "tools/run_cure_lite_v24_gcr_pacre_dr_gate_r2_preaccess_compat_c5.py"
).resolve()
C5_UNIT_REALIZER_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_actual_unit_realization_preaccess_compat_c5.py"
).resolve()
C5_UNIT_TEMPLATE_PATH = (
    REPOSITORY
    / "deploy/systemd/"
    "cure-lite-v24-gcr-pacre-dr-r2-preaccess-compat-c5.service.template"
).resolve()
C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_compat_c2_mode_contract_failure_terminal.py"
).resolve()
C2_PREWRITE_FAILURE_TERMINALIZER_SOURCE_PATH = (
    REPOSITORY
    / "tools/cure_lite_v24_preaccess_compat_c2_prewrite_failure_terminal.py"
).resolve()
C3_ENVIRONMENT_FAILURE_TERMINALIZER_SOURCE_PATH = (
    REPOSITORY
    / (
        "tools/cure_lite_v24_preaccess_compat_c3_"
        "environment_stability_failure_terminal.py"
    )
).resolve()
R14_INTEGRATION_WRAPPER_SOURCE_PATH = (
    REPOSITORY / "tools/cure_lite_v24_user_systemd_integration.py"
).resolve()
R14_SHARED_REALIZER_SOURCE_PATH = (
    REPOSITORY / "tools/cure_lite_v24_realize_systemd_unit.py"
).resolve()
R14_DUMMY_CHILD_SOURCE_PATH = (
    REPOSITORY / "tools/cure_lite_v24_dummy_child.py"
).resolve()
R14_DUMMY_UNIT_TEMPLATE_PATH = (
    REPOSITORY
    / "deploy/systemd/cure-lite-v24-supervisor-integration.service.template"
).resolve()

C1_FAILURE_TERMINALIZER_SHA256 = (
    "72d7f8846d9bdccbdb2d15d6790d5e021b6d2db75523fe5dbe11a3d4246ca880"
)
C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SHA256 = (
    "86181ffbb584381754c2eafe40759f8caaf387d270a8b8c71a45f2eefa099126"
)
C2_MODE_CONTRACT_FAILURE_TERMINAL_SHA256 = (
    "e478e0cc3516c97b5eea91c615a64cd7ee4020a9d22ddc863c7b04375331f9e7"
)
C2_MODE_CONTRACT_FAILURE_TERMINAL_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c2-"
    "mode-contract-failure-terminal-v1"
)
C2_PREWRITE_FAILURE_TERMINALIZER_SHA256 = (
    "17ef3a0420c4b3d978f23270bde490805997e21dcb21f395ce7e5ac06659dc5f"
)
C2_PREWRITE_FAILURE_TERMINAL_SHA256 = (
    "6984dc9df2c905a5b7bc3b1577a4d5e8c21d1e1f895217997ed6915050e0f43d"
)
C2_PREWRITE_FAILURE_TERMINAL_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c2-"
    "prewrite-failure-terminal-v1"
)
C3_ENVIRONMENT_FAILURE_TERMINAL_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c3-"
    "environment-stability-failure-terminal-v1"
)

# These two values are intentionally not permissive placeholders.  Production
# authorization fails before loading either predecessor file until the final
# terminalizer and create-once terminal bytes have been independently frozen.
_TO_BE_FROZEN_SHA256 = "__TO_BE_FROZEN__"
C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256 = (
    "b55a916dade97b9d49f1cd80758aeaac316d55725eb7ee4e1148c7c206aa9d9f"
)
C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256 = (
    "527eb5c12c92e19dac8f797868de2bc8462e53b8113c24f6e701e0e54a26180a"
)
C3_ENVIRONMENT_FAILURE_TERMINAL_FINGERPRINT = (
    "c31159e7033450ecc2a8dea071fd125ab756e43afbc8d8c433c425a045713670"
)

C4_BRIDGE_SHA256 = (
    "ad660b7afe7ca87f690bc9565bd6674684c2b62824394751a39114a6efcf178a"
)
C4_RECEIPT_SEAL_FAILURE_TERMINAL_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c4-"
    "receipt-seal-failure-terminal-v1"
)
# Independently frozen C4 receipt-seal FAIL producer and create-once record.
C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SHA256 = (
    "3cf56e803d6d7b39c995125d17d145b5c8625a4eea03de6cf4c6118c9bc777c0"
)
C4_RECEIPT_SEAL_FAILURE_TERMINAL_SHA256 = (
    "567b22e9839dad2d27168c36206b66be9b2b91d98269e9b9ce087ee3becea733"
)
C4_RECEIPT_SEAL_FAILURE_TERMINAL_FINGERPRINT = (
    "d86ef0c432237043e39119c56cfb6602b7df7f8b62069f836ac6c3d08b75b622"
)

AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c5-authorization-v1"
)
RECEIPT_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c5-receipt-v1"
)
C5_TERMINAL_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c5-terminal-v1"
)
C4_AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c4-authorization-v1"
)
C1_AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-r2-preaccess-schema-compat-c1-authorization-v1"
)
R10_TERMINAL_SCHEMA = (
    "cure-lite-v24-supervisor-v2-systemd-integration-terminal-v1"
)
ENVIRONMENT_POLICY_SCHEMA = "cure-lite-v24-runtime-environment-policy-v1"
ENVIRONMENT_SCOPE_HANDOFF_SCHEMA = (
    "cure-lite-v24-runtime-environment-scope-handoff-"
    "preaccess-compat-c5-v1"
)
ENVIRONMENT_STABILITY_ATTEMPT_SCHEMA = (
    "cure-lite-v24-runtime-environment-stability-attempt-"
    "preaccess-compat-c5-v1"
)
ENVIRONMENT_STABILITY_TERMINAL_SCHEMA = (
    "cure-lite-v24-runtime-environment-stability-terminal-"
    "preaccess-compat-c5-v1"
)
ENVIRONMENT_STABILITY_SCHEMA = (
    "cure-lite-v24-runtime-environment-stability-receipt-v1"
)
ENVIRONMENT_RECEIPT_SCHEMA = (
    "cure-lite-v24-runtime-environment-audit-receipt-v1"
)
UNIT_AUTHORIZATION_SCHEMA = (
    "cure-lite-v24-actual-unit-realization-authorization-v1"
)
UNIT_RECEIPT_SCHEMA = (
    "cure-lite-v24-actual-unit-realization-receipt-v1"
)
RUNTIME_SPEC_SCHEMA = "cure-lite-v24-dr-runtime-supervisor-spec-v2"
R2_RESULT_SCHEMA = (
    "cure-lite-v24-gcr-pacre-real-dr-structural-gate-r2-v1"
)
R2_RUN_START_SCHEMA = (
    "cure-lite-v24-D_R-persistent-run-start-r2-v1"
)
R2_RUN_START_STAGE_ID = "gcr_pacre_v24_D_R_structural_r2"
R2_RUN_START_PATH_POLICY = (
    "fixed_repository_run_root_authorization_fingerprint_filename_v1"
)
R2_EXECUTION_KIND = "real_D_R"
R2_EXECUTION_SEED = 42
AUTHORITATIVE_ACCESS_AUDIT_SCHEMA = (
    "cure-lite-v24-split-access-audit-v1"
)
FICTIONAL_ACCESS_AUDIT_SCHEMA = (
    "cure-lite-v24-split-access-audit-r2-v1"
)
R2_RUN_START_FILENAME_PREFIX = (
    "gcr_pacre_v24_D_R_structural_run_start_"
)

RUNTIME_PHASE_PREACTIVATION = "preactivation"
RUNTIME_PHASE_COMMIT = "commit"
RUNTIME_PHASE_CLAIM = "claim"
RUNTIME_PHASE_VERIFY = "verify"
RUNTIME_PHASE_RUN_ONCE = "run_once"
RUNTIME_PHASE_FINALIZE_SUCCESS = "finalize_success"
RUNTIME_PHASE_FINALIZE_FAILURE = "finalize_failure"
RUNTIME_PHASES = frozenset(
    {
        RUNTIME_PHASE_PREACTIVATION,
        RUNTIME_PHASE_COMMIT,
        RUNTIME_PHASE_CLAIM,
        RUNTIME_PHASE_VERIFY,
        RUNTIME_PHASE_RUN_ONCE,
        RUNTIME_PHASE_FINALIZE_SUCCESS,
        RUNTIME_PHASE_FINALIZE_FAILURE,
    }
)
_PRESPAWN_ABSENT_PHASES = frozenset(
    {
        RUNTIME_PHASE_PREACTIVATION,
        RUNTIME_PHASE_COMMIT,
        RUNTIME_PHASE_CLAIM,
        RUNTIME_PHASE_VERIFY,
        RUNTIME_PHASE_RUN_ONCE,
    }
)

_R2_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "run_id",
        "candidate",
        "execution_kind",
        "execution_seed",
        "device",
        "requested_receipt_output",
        "dataset_free_receipt_fingerprint",
        "dataset_free_receipt_file_sha256",
        "efficiency_section_fingerprint",
        "efficiency_receipt_sha256",
        "preaccess_authorization_fingerprint",
        "preaccess_authorization_file_sha256",
        "access_audit_receipt_fingerprint",
        "access_audit_receipt_file_sha256",
        "protocol_preregistration_fingerprint",
        "implementation_binding",
        "source_closure_fingerprint",
        "source_binding_fingerprint",
        "real_inputs_fingerprint",
        "population_fingerprint",
        "cache_fingerprint",
        "adapter_fingerprint",
        "run_start_marker",
        "artifact_hashes",
        "raw_observations",
        "raw_observations_fingerprint",
        "checks",
        "decision",
        "boundary",
        "receipt_fingerprint",
    }
)
_R2_RUN_START_ENVELOPE_KEYS = frozenset(
    {"path", "file_sha256", "marker_fingerprint", "payload"}
)
_R2_RUN_START_MARKER_KEYS = frozenset(
    {
        "schema_version",
        "path_policy",
        "stage_id",
        "run_id",
        "candidate",
        "marker_path",
        "authorization_fingerprint",
        "authorization_receipt_file_sha256",
        "access_audit_receipt_fingerprint",
        "access_audit_receipt_file_sha256",
        "dataset_free_receipt_fingerprint",
        "dataset_free_receipt_file_sha256",
        "protocol_preregistration_fingerprint",
        "source_closure_fingerprint",
        "implementation_binding",
        "expected_source_binding_fingerprint",
        "expected_real_inputs_fingerprint",
        "expected_population_fingerprint",
        "expected_cache_fingerprint",
        "intent",
        "intent_fingerprint",
        "marker_fingerprint",
    }
)
_R2_RUN_INTENT_KEYS = frozenset(
    {
        "execution_kind",
        "split",
        "requested_device",
        "requested_receipt_output",
        "D_R_materialization_intended",
        "D_V_materialization_intended",
        "D_T_materialization_intended",
        "optimizer_steps_authorized",
        "parameter_updates_authorized",
        "training_authorized",
    }
)
_R2_BOUNDARY_KEYS = frozenset(
    {
        "execution_kind",
        "split",
        "D_R_accessed",
        "D_V_accessed",
        "D_T_accessed",
        "D_V_tensor_payload_accessed",
        "D_T_tensor_payload_accessed",
        "optimizer_module_referenced",
        "optimizer_constructed",
        "optimizer_steps",
        "parameter_updates",
        "training_performed",
        "performance_gate_present",
        "performance_claim_supported",
        "threshold_or_ratio_gate",
    }
)
_R2_ARTIFACT_HASH_KEYS = frozenset(
    {
        "dataset_free_receipt",
        "preaccess_authorization",
        "preaccess_access_audit",
        "persistent_run_start_marker",
    }
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_UTC_SUFFIX = "Z"
_PAYLOAD_FLAGS = (
    "D_R_payload_accessed",
    "D_V_payload_accessed",
    "D_T_payload_accessed",
)
_STAT_GENERATION_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_uid",
    "st_gid",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
_STATE_FIELDS = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "NRestarts",
    "FragmentPath",
    "InvocationID",
    "DropInPaths",
    "NeedDaemonReload",
    "Transient",
    "Restart",
    "ExecMainPID",
    "ExecMainCode",
    "ExecMainStatus",
    "Result",
    "StateChangeTimestamp",
    "ActiveEnterTimestamp",
    "InactiveEnterTimestamp",
    "ExecMainStartTimestamp",
    "ExecMainExitTimestamp",
)
_BRIDGE_STATE_FIELDS = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "UnitFileState",
    "NRestarts",
    "FragmentPath",
    "InvocationID",
)
_SCOPE_FIELDS = (
    "target_unit_id",
    "conflict_unit_ids",
    "dependency_unit_ids",
    "allowed_failed_unit_ids",
    "allowed_unit_ids",
    "allowed_manager_states",
    "require_target_ready",
    "strict_all_gpu_consumers",
)
_SOURCE_LABELS = frozenset(
    {
        "compat_bridge",
        "compat_policy",
        "c4_bridge",
        "c4_receipt_seal_failure_terminalizer",
        "c1_failure_terminalizer",
        "compat_environment_wrapper",
        "compat_release",
        "compat_supervisor",
        "compat_adapter",
        "compat_unit_realizer",
        "compat_unit_template",
        "c2_mode_contract_failure_terminalizer",
        "c2_prewrite_failure_terminalizer",
        "c3_environment_failure_terminalizer",
        "r14_integration_wrapper",
        "r14_shared_realizer",
        "r14_dummy_child",
        "r14_dummy_unit_template",
    }
)
_EVIDENCE_LABELS = frozenset(
    {
        "c1_failure_terminal",
        "c2_mode_contract_failure_terminal",
        "c2_prewrite_failure_terminal",
        "c3_environment_failure_terminal",
        "c4_authorization",
        "c4_receipt_seal_failure_terminal",
        "r10_authorization",
        "r10_receipt",
        "environment_policy",
        "environment_scope_handoff",
        "environment_stability_attempt",
        "environment_stability",
        "environment_postcleanup",
        "unit_realization_authorization",
        "unit_realization_receipt",
    }
)
_AUTHORIZATION_KEYS = frozenset(
    {
        "schema_version",
        "candidate",
        "stage_id",
        "scientific_attempt_id",
        "scientific_attempt_ordinal",
        "runtime_compatibility_id",
        "instruction_id",
        "authorization_basis",
        "authorized_uid",
        "created_at_utc",
        "issued_at_utc",
        "expires_at_utc",
        "c1_failure_terminal_root",
        "c2_mode_contract_failure_terminal_root",
        "c2_prewrite_failure_terminal_root",
        "c3_environment_failure_terminal_root",
        "c4_authorization_root",
        "c4_receipt_seal_failure_terminal_root",
        "c1_expired_authorization_root",
        "r10_roots",
        "compatibility_source_roots",
        "protected_unit_states",
        "preauthorization_target_unit_state",
        "expected_evidence_paths",
        "scientific_output_contract",
        "scientific_authority",
        "mutation_authority",
        "D_R_payload_accessed",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "gpu_accessed",
        "training_started",
        "materialization_consumed",
        "authorization_fingerprint",
    }
)
_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "candidate",
        "stage_id",
        "scientific_attempt_id",
        "scientific_attempt_ordinal",
        "runtime_compatibility_id",
        "created_at_utc",
        "compatibility_authorization_root",
        "compatibility_source_roots",
        "compatibility_evidence_roots",
        "historical_environment_contract",
        "current_environment_contract",
        "scientific_output_contract",
        "scientific_authority",
        "schema_compatibility",
        "compatibility_closure_passed",
        "runtime_launch_authorized",
        "systemd_start_authorized",
        "automatic_retry",
        "resume",
        "D_R_payload_accessed",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "gpu_accessed",
        "training_started",
        "materialization_consumed",
        "receipt_fingerprint",
    }
)

UnitStateReader = Callable[[str], Mapping[str, object]]

# Compatibility-facing names intentionally mirror the c1 bridge interface.
# Downstream c5 realizer/supervisor/release modules must not need a permissive
# fallback or generation-specific import shim.
COMPAT_AUTHORIZATION_PATH = C5_AUTHORIZATION_PATH
COMPAT_RECEIPT_PATH = C5_RECEIPT_PATH
COMPATIBILITY_RECEIPT_PATH = C5_RECEIPT_PATH
COMPAT_UNIT_REALIZER_SOURCE_PATH = C5_UNIT_REALIZER_SOURCE_PATH
COMPAT_UNIT_NAME = C5_UNIT_NAME


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        # C5 owns this UTF-8 profile.  Foreign C4/R5/E5 evidence is never
        # reinterpreted with this helper; its fixed producer validates it.
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_fingerprint(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(dict(value)).encode()).hexdigest()


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", _UTC_SUFFIX)


def _parse_utc(value: object, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith(_UTC_SUFFIX):
        raise PermissionError(f"{name} is not canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise PermissionError(f"{name} is malformed") from error
    if parsed.tzinfo is None or _format_utc(parsed) != value:
        raise PermissionError(f"{name} is not canonical UTC")
    return parsed.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _require_fixed_path(
    supplied: str | Path,
    expected: Path,
    *,
    name: str,
) -> Path:
    path = Path(supplied).absolute()
    if path != Path(expected).absolute():
        raise PermissionError(f"{name} path changed")
    return path


def _safe_parent(path: Path) -> os.stat_result:
    parent = path.parent
    before = parent.lstat()
    if (
        not stat.S_ISDIR(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or parent.resolve(strict=True) != parent
        or before.st_uid != os.getuid()
    ):
        raise PermissionError(f"unsafe evidence parent: {parent}")
    return before


def _read_regular_bytes(
    path: Path,
    *,
    sealed: bool = True,
) -> tuple[bytes, os.stat_result]:
    target = Path(path).absolute()
    parent_before = _safe_parent(target)
    flags = os.O_RDONLY
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required")
    flags |= os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    directory_fd = os.open(
        target.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        parent_opened = os.fstat(directory_fd)
        if (
            parent_opened.st_dev,
            parent_opened.st_ino,
            parent_opened.st_uid,
        ) != (
            parent_before.st_dev,
            parent_before.st_ino,
            parent_before.st_uid,
        ):
            raise PermissionError("evidence parent generation changed")
        fd = os.open(target.name, flags, dir_fd=directory_fd)
        try:
            before = os.fstat(fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.getuid()
                or before.st_nlink != 1
                or (sealed and stat.S_IMODE(before.st_mode) != 0o444)
            ):
                raise PermissionError(f"unsafe sealed file: {target}")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(fd)
        finally:
            os.close(fd)
    finally:
        os.close(directory_fd)
    parent_after = target.parent.lstat()
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_uid,
        before.st_gid,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_uid,
        after.st_gid,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) or (
        parent_before.st_dev,
        parent_before.st_ino,
        parent_before.st_uid,
    ) != (
        parent_after.st_dev,
        parent_after.st_ino,
        parent_after.st_uid,
    ):
        raise PermissionError(f"sealed file changed while reading: {target}")
    return b"".join(chunks), before


def _source_root(path: Path) -> dict[str, object]:
    # Repository sources are owner-owned and may use the workspace's existing
    # 0664 mode.  Their exact bytes and file generation are frozen below; the
    # stricter non-writable rule remains mandatory for sealed evidence.
    raw, observed = _read_regular_bytes(path, sealed=False)
    target = Path(path).absolute()
    return {
        "path": str(target),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": stat.S_IMODE(observed.st_mode),
        "owner_uid": observed.st_uid,
        "size": observed.st_size,
    }


def _validate_source_root(
    root: object,
    *,
    expected_path: Path,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    if not isinstance(root, Mapping):
        raise PermissionError("source root is malformed")
    observed = _source_root(expected_path)
    if dict(root) != observed:
        raise PermissionError(f"source generation changed: {expected_path}")
    if (
        expected_sha256 is not None
        and observed["file_sha256"] != expected_sha256
    ):
        raise PermissionError(f"frozen source hash changed: {expected_path}")
    return observed


def _write_sealed(
    path: Path,
    body: Mapping[str, object],
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    target = Path(path).absolute()
    parent_before = _safe_parent(target)
    if fingerprint_field in body:
        raise ValueError("fingerprint field must not be pre-populated")
    payload = dict(body)
    payload[fingerprint_field] = stable_fingerprint(payload)
    raw = (_canonical_json(payload) + "\n").encode()
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required")
    flags |= os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    directory_fd = os.open(
        target.parent,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        parent_opened = os.fstat(directory_fd)
        if (
            parent_opened.st_dev,
            parent_opened.st_ino,
            parent_opened.st_uid,
        ) != (
            parent_before.st_dev,
            parent_before.st_ino,
            parent_before.st_uid,
        ):
            raise PermissionError("sealed-write parent generation changed")
        fd = os.open(target.name, flags, 0o400, dir_fd=directory_fd)
        try:
            offset = 0
            while offset < len(raw):
                written = os.write(fd, raw[offset:])
                if written <= 0:
                    raise OSError("short sealed evidence write")
                offset += written
            os.fsync(fd)
            os.fchmod(fd, 0o444)
            os.fsync(fd)
            sealed = os.fstat(fd)
            linked = os.stat(
                target.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            readback = os.pread(fd, len(raw) + 1, 0)
            finished = os.fstat(fd)
            identity = tuple(
                getattr(sealed, field)
                for field in _STAT_GENERATION_FIELDS
            )
            if (
                not stat.S_ISREG(sealed.st_mode)
                or sealed.st_uid != os.getuid()
                or sealed.st_nlink != 1
                or stat.S_IMODE(sealed.st_mode) != 0o444
                or sealed.st_size != len(raw)
                or identity
                != tuple(
                    getattr(linked, field)
                    for field in _STAT_GENERATION_FIELDS
                )
                or identity
                != tuple(
                    getattr(finished, field)
                    for field in _STAT_GENERATION_FIELDS
                )
                or readback != raw
            ):
                raise PermissionError("sealed evidence fd seal/readback changed")
        finally:
            os.close(fd)
        os.fsync(directory_fd)
        parent_finished = os.fstat(directory_fd)
        parent_linked = target.parent.lstat()
        parent_identity = (
            parent_opened.st_dev,
            parent_opened.st_ino,
            parent_opened.st_mode,
            parent_opened.st_uid,
            parent_opened.st_gid,
        )
        if parent_identity != (
            parent_finished.st_dev,
            parent_finished.st_ino,
            parent_finished.st_mode,
            parent_finished.st_uid,
            parent_finished.st_gid,
        ) or parent_identity != (
            parent_linked.st_dev,
            parent_linked.st_ino,
            parent_linked.st_mode,
            parent_linked.st_uid,
            parent_linked.st_gid,
        ):
            raise PermissionError("sealed-write parent generation changed")
    finally:
        os.close(directory_fd)
    loaded, _root = _load_sealed(
        target,
        fingerprint_field=fingerprint_field,
    )
    if loaded != payload:
        raise RuntimeError("sealed write verification failed")
    return payload


def _load_sealed(
    path: Path,
    *,
    fingerprint_field: str,
    schema: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    target = Path(path).absolute()
    raw, observed = _read_regular_bytes(target)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(f"sealed JSON is malformed: {target}") from error
    if (
        not isinstance(value, dict)
        or not isinstance(value.get(fingerprint_field), str)
        or _SHA256.fullmatch(value[fingerprint_field]) is None
    ):
        raise PermissionError(f"sealed fingerprint is absent: {target}")
    body = dict(value)
    fingerprint = body.pop(fingerprint_field)
    if fingerprint != stable_fingerprint(body):
        raise PermissionError(f"sealed fingerprint changed: {target}")
    if schema is not None and value.get("schema_version") != schema:
        raise PermissionError(f"sealed schema changed: {target}")
    if raw != (_canonical_json(value) + "\n").encode():
        raise PermissionError(f"sealed JSON is not canonical: {target}")
    root = {
        "path": str(target),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "fingerprint_field": fingerprint_field,
        "fingerprint": fingerprint,
        "schema_version": value.get("schema_version"),
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": stat.S_IMODE(observed.st_mode),
        "owner_uid": observed.st_uid,
        "size": observed.st_size,
    }
    return value, root


def _validate_sealed_root(
    root: object,
    *,
    expected_path: Path,
    fingerprint_field: str,
    schema: str | None = None,
) -> dict[str, object]:
    if not isinstance(root, Mapping):
        raise PermissionError("sealed root is malformed")
    _payload, observed = _load_sealed(
        expected_path,
        fingerprint_field=fingerprint_field,
        schema=schema,
    )
    if dict(root) != observed:
        raise PermissionError(f"sealed evidence changed: {expected_path}")
    return observed


def _require_no_payload(value: Mapping[str, object]) -> None:
    if any(value.get(field) is not False for field in _PAYLOAD_FLAGS):
        raise PermissionError("compatibility evidence accessed payload")
    if value.get("gpu_accessed", False) is not False:
        raise PermissionError("compatibility evidence accessed a GPU")
    if value.get("training_started", False) is not False:
        raise PermissionError("compatibility evidence started training")
    if value.get("materialization_consumed", False) is not False:
        raise PermissionError("scientific materialization was consumed")


def _expected_scientific_output_contract() -> dict[str, object]:
    return {
        "run_root": str(SCIENTIFIC_RUN_ROOT),
        "result_receipt": str(SCIENTIFIC_RESULT_RECEIPT_PATH),
        "compat_run_root_alias": str(C5_RUN_ROOT_ALIAS_PATH),
        "compat_result_receipt_alias": str(
            C5_RESULT_RECEIPT_ALIAS_PATH
        ),
        "original_r2_paths_retained": True,
        "compatibility_aliases_forbidden": True,
    }


def _expected_schema_compatibility() -> dict[str, object]:
    """Exact compatibility projection required by the frozen C1 consumer."""

    return {
        "producer_schema": AUTHORITATIVE_ACCESS_AUDIT_SCHEMA,
        "scientific_authorization_bound_schema": (
            AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
        ),
        "compatibility_consumer_required_schema": (
            AUTHORITATIVE_ACCESS_AUDIT_SCHEMA
        ),
        "buggy_frozen_consumer_expected_schema": (
            FICTIONAL_ACCESS_AUDIT_SCHEMA
        ),
        "accept_either_schema": False,
    }


def _expected_scientific_authority() -> dict[str, object]:
    return {
        "D_R_payload_authorized": False,
        "D_V_payload_authorized": False,
        "D_T_payload_authorized": False,
        "training_authorized": False,
        "materialization_authorized": False,
        "automatic_retry": False,
        "resume": False,
        "fresh_scientific_attempt": False,
    }


def _expected_mutation_authority() -> dict[str, object]:
    return {
        "compatibility_receipt_creation_authorized": True,
        "compatibility_terminal_creation_authorized": True,
        "environment_scope_handoff_authorized": True,
        "environment_metadata_audit_authorized": True,
        "c5_unit_realization_authorized": True,
        "c4_unit_mutation_authorized": False,
        "c4_evidence_mutation_authorized": False,
        "runtime_spec_creation_authorized": False,
        "runtime_launch_authorization_creation_authorized": False,
        "unit_start_authorized": False,
        "unit_enable_authorized": False,
        "payload_access_authorized": False,
    }


def _default_unit_state_reader(unit_name: str) -> Mapping[str, object]:
    command = [
        "/usr/bin/systemctl",
        "--user",
        "show",
        unit_name,
        *[f"--property={field}" for field in _STATE_FIELDS],
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    state: dict[str, object] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in _STATE_FIELDS:
            state[key] = value
    if set(state) != set(_STATE_FIELDS):
        raise PermissionError(f"unit state is incomplete: {unit_name}")
    return state


def _normalized_state(
    reader: UnitStateReader,
    unit_name: str,
) -> dict[str, object]:
    state = dict(reader(unit_name))
    if not set(_BRIDGE_STATE_FIELDS).issubset(state):
        raise PermissionError(f"unit state fields changed: {unit_name}")
    result = {field: state[field] for field in _BRIDGE_STATE_FIELDS}
    if result["Id"] != unit_name:
        raise PermissionError(f"unit identity changed: {unit_name}")
    return result


def _require_inert_state(
    state: Mapping[str, object],
    *,
    unit_name: str,
    fragment_path: str | None = None,
) -> None:
    if (
        state.get("Id") != unit_name
        or state.get("LoadState") != "loaded"
        or state.get("ActiveState") != "inactive"
        or state.get("SubState") != "dead"
        or state.get("UnitFileState") != "static"
        or state.get("NRestarts") != "0"
        or state.get("InvocationID") not in ("", None)
        or (
            fragment_path is not None
            and state.get("FragmentPath") != fragment_path
        )
    ):
        raise PermissionError(f"unit is not exact static/inert: {unit_name}")


def _require_missing_inert_state(
    state: Mapping[str, object],
    *,
    unit_name: str,
) -> None:
    if (
        state.get("Id") != unit_name
        or state.get("LoadState") != "not-found"
        or state.get("ActiveState") != "inactive"
        or state.get("SubState") != "dead"
        or state.get("UnitFileState") not in ("", None)
        or state.get("NRestarts") != "0"
        or state.get("FragmentPath") not in ("", None)
        or state.get("InvocationID") not in ("", None)
    ):
        raise PermissionError(f"unit is not exact not-found/inert: {unit_name}")


def _source_paths() -> dict[str, Path]:
    return {
        "compat_bridge": C5_BRIDGE_SOURCE_PATH,
        # Strict alias retained for the frozen C1 supervisor's consumer
        # contract; both labels must bind the same B5 file generation.
        "compat_policy": C5_BRIDGE_SOURCE_PATH,
        "c4_bridge": C4_BRIDGE_SOURCE_PATH,
        "c4_receipt_seal_failure_terminalizer": (
            C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SOURCE_PATH
        ),
        "c1_failure_terminalizer": (
            C1_FAILURE_TERMINALIZER_SOURCE_PATH
        ),
        "compat_environment_wrapper": (
            C5_ENVIRONMENT_WRAPPER_SOURCE_PATH
        ),
        "compat_release": C5_RELEASE_SOURCE_PATH,
        "compat_supervisor": C5_SUPERVISOR_SOURCE_PATH,
        "compat_adapter": C5_ADAPTER_SOURCE_PATH,
        "compat_unit_realizer": C5_UNIT_REALIZER_SOURCE_PATH,
        "compat_unit_template": C5_UNIT_TEMPLATE_PATH,
        "c2_mode_contract_failure_terminalizer": (
            C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SOURCE_PATH
        ),
        "c2_prewrite_failure_terminalizer": (
            C2_PREWRITE_FAILURE_TERMINALIZER_SOURCE_PATH
        ),
        "c3_environment_failure_terminalizer": (
            C3_ENVIRONMENT_FAILURE_TERMINALIZER_SOURCE_PATH
        ),
        "r14_integration_wrapper": R14_INTEGRATION_WRAPPER_SOURCE_PATH,
        "r14_shared_realizer": R14_SHARED_REALIZER_SOURCE_PATH,
        "r14_dummy_child": R14_DUMMY_CHILD_SOURCE_PATH,
        "r14_dummy_unit_template": R14_DUMMY_UNIT_TEMPLATE_PATH,
    }


def _has_unfrozen_binding(raw: bytes, *, path: Path) -> bool:
    """Detect sentinel assignments, including parenthesized multiline pins."""

    if path.suffix != ".py":
        return _TO_BE_FROZEN_SHA256.encode("ascii") in raw
    try:
        tree = ast.parse(raw.decode("utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as error:
        raise PermissionError(
            f"c5 source cannot be parsed while checking bindings: {path}"
        ) from error
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            value = node.value
        elif isinstance(node, ast.NamedExpr):
            value = node.value
        else:
            continue
        if value is not None and any(
            isinstance(item, ast.Constant)
            and item.value == _TO_BE_FROZEN_SHA256
            for item in ast.walk(value)
        ):
            return True
    return False


def _require_frozen_c3_failure_hashes() -> None:
    values = (
        C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256,
        C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256,
    )
    if any(
        value == _TO_BE_FROZEN_SHA256
        or not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        for value in values
    ):
        raise PermissionError(
            "c3 failure terminal source/evidence hashes are not frozen"
        )


def _require_frozen_c4_failure_hashes() -> None:
    values = (
        C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SHA256,
        C4_RECEIPT_SEAL_FAILURE_TERMINAL_SHA256,
        C4_RECEIPT_SEAL_FAILURE_TERMINAL_FINGERPRINT,
    )
    if any(
        value == _TO_BE_FROZEN_SHA256
        or not isinstance(value, str)
        or _SHA256.fullmatch(value) is None
        for value in values
    ):
        raise PermissionError(
            "c4 receipt-seal failure source/evidence hashes are not frozen"
        )


def _evidence_paths() -> dict[str, Path]:
    return {
        "c1_failure_terminal": C1_FAILURE_TERMINAL_PATH,
        "c2_mode_contract_failure_terminal": (
            C2_MODE_CONTRACT_FAILURE_TERMINAL_PATH
        ),
        "c2_prewrite_failure_terminal": (
            C2_PREWRITE_FAILURE_TERMINAL_PATH
        ),
        "c3_environment_failure_terminal": (
            C3_ENVIRONMENT_FAILURE_TERMINAL_PATH
        ),
        "c4_authorization": C4_AUTHORIZATION_PATH,
        "c4_receipt_seal_failure_terminal": (
            C4_RECEIPT_SEAL_FAILURE_TERMINAL_PATH
        ),
        "r10_authorization": R10_AUTHORIZATION_PATH,
        "r10_receipt": R10_RECEIPT_PATH,
        "environment_policy": C5_ENVIRONMENT_POLICY_PATH,
        "environment_scope_handoff": C5_ENVIRONMENT_SCOPE_HANDOFF_PATH,
        "environment_stability_attempt": (
            C5_ENVIRONMENT_STABILITY_ATTEMPT_PATH
        ),
        "environment_stability_terminal": (
            C5_ENVIRONMENT_STABILITY_TERMINAL_PATH
        ),
        "environment_stability": C5_ENVIRONMENT_STABILITY_PATH,
        "environment_postcleanup": C5_ENVIRONMENT_POSTCLEANUP_PATH,
        "unit_realization_authorization": C5_UNIT_AUTHORIZATION_PATH,
        "unit_realization_receipt": C5_UNIT_RECEIPT_PATH,
    }


def _collect_source_roots() -> dict[str, dict[str, object]]:
    _require_frozen_c4_failure_hashes()
    _require_frozen_c3_failure_hashes()
    for label in (
        "compat_environment_wrapper",
        "compat_release",
        "compat_supervisor",
        "compat_adapter",
        "compat_unit_realizer",
        "compat_unit_template",
        "r14_integration_wrapper",
        "r14_shared_realizer",
        "r14_dummy_child",
        "r14_dummy_unit_template",
    ):
        raw, _observed = _read_regular_bytes(
            _source_paths()[label],
            sealed=False,
        )
        if _has_unfrozen_binding(raw, path=_source_paths()[label]):
            raise PermissionError(
                f"c5 source still has an unfrozen binding: {label}"
            )
    roots = {
        label: _source_root(path)
        for label, path in _source_paths().items()
    }
    if set(roots) != _SOURCE_LABELS:
        raise AssertionError("c5 source labels changed")
    if (
        roots["c1_failure_terminalizer"]["file_sha256"]
        != C1_FAILURE_TERMINALIZER_SHA256
        or roots["compat_policy"] != roots["compat_bridge"]
        or roots["c4_bridge"]["file_sha256"] != C4_BRIDGE_SHA256
        or roots["c4_receipt_seal_failure_terminalizer"][
            "file_sha256"
        ] != C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SHA256
        or roots["c2_mode_contract_failure_terminalizer"]["file_sha256"]
        != C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SHA256
        or roots["c2_prewrite_failure_terminalizer"]["file_sha256"]
        != C2_PREWRITE_FAILURE_TERMINALIZER_SHA256
        or roots["c3_environment_failure_terminalizer"]["file_sha256"]
        != C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256
    ):
        raise PermissionError("compatibility terminalizer source hash changed")
    return roots


def _validate_source_roots(roots: object) -> None:
    _require_frozen_c4_failure_hashes()
    _require_frozen_c3_failure_hashes()
    if not isinstance(roots, Mapping) or set(roots) != _SOURCE_LABELS:
        raise PermissionError("c5 source-root labels changed")
    for label, path in _source_paths().items():
        expected = None
        if label == "c1_failure_terminalizer":
            expected = C1_FAILURE_TERMINALIZER_SHA256
        elif label == "c4_bridge":
            expected = C4_BRIDGE_SHA256
        elif label == "c4_receipt_seal_failure_terminalizer":
            expected = C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SHA256
        elif label == "c2_mode_contract_failure_terminalizer":
            expected = C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SHA256
        elif label == "c2_prewrite_failure_terminalizer":
            expected = C2_PREWRITE_FAILURE_TERMINALIZER_SHA256
        elif label == "c3_environment_failure_terminalizer":
            expected = C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256
        _validate_source_root(
            roots[label],
            expected_path=path,
            expected_sha256=expected,
        )
    if roots["compat_policy"] != roots["compat_bridge"]:
        raise PermissionError("c5 bridge/policy source alias diverged")


def _always_absent_paths() -> dict[str, Path]:
    return {
        "c1_compatibility_receipt": C1_RECEIPT_PATH,
        "c1_runtime_spec": C1_RUNTIME_SPEC_PATH,
        "c1_runtime_launch_authorization": (
            C1_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        ),
        "c1_runtime_artifacts": C1_RUNTIME_ARTIFACT_ROOT,
        "c1_gpu_lease": C1_GPU_LEASE_ROOT,
        "c1_run_alias": C1_RUN_ROOT_ALIAS_PATH,
        "c1_result_alias": C1_RESULT_RECEIPT_ALIAS_PATH,
        "c2_compatibility_receipt": C2_RECEIPT_PATH,
        "c2_environment_policy": C2_ENVIRONMENT_POLICY_PATH,
        "c2_environment_stability": C2_ENVIRONMENT_STABILITY_PATH,
        "c2_environment_postcleanup": C2_ENVIRONMENT_POSTCLEANUP_PATH,
        "c2_unit_authorization": C2_UNIT_AUTHORIZATION_PATH,
        "c2_unit_receipt": C2_UNIT_RECEIPT_PATH,
        "c2_unit_terminal": C2_UNIT_TERMINAL_PATH,
        "c2_unit_fragment": C2_UNIT_FRAGMENT_PATH,
        "c2_runtime_spec": C2_RUNTIME_SPEC_PATH,
        "c2_runtime_launch_authorization": C2_RUNTIME_LAUNCH_AUTHORIZATION_PATH,
        "c2_runtime_artifacts": C2_RUNTIME_ARTIFACT_ROOT,
        "c2_gpu_lease": C2_GPU_LEASE_ROOT,
        "c2_run_alias": C2_RUN_ROOT_ALIAS_PATH,
        "c2_result_alias": C2_RESULT_RECEIPT_ALIAS_PATH,
        "c3_compatibility_receipt": C3_RECEIPT_PATH,
        "c3_environment_stability": C3_ENVIRONMENT_STABILITY_PATH,
        "c3_environment_postcleanup": C3_ENVIRONMENT_POSTCLEANUP_PATH,
        "c3_unit_terminal": C3_UNIT_TERMINAL_PATH,
        "c3_runtime_spec": C3_RUNTIME_SPEC_PATH,
        "c3_runtime_launch_authorization": (
            C3_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        ),
        "c3_runtime_artifacts": C3_RUNTIME_ARTIFACT_ROOT,
        "c3_gpu_lease": C3_GPU_LEASE_ROOT,
        "c3_run_alias": C3_RUN_ROOT_ALIAS_PATH,
        "c3_result_alias": C3_RESULT_RECEIPT_ALIAS_PATH,
        "c4_compatibility_receipt": C4_RECEIPT_PATH,
        "c4_environment_stability_terminal": (
            C4_ENVIRONMENT_STABILITY_TERMINAL_PATH
        ),
        "c4_unit_terminal": C4_UNIT_TERMINAL_PATH,
        "c4_runtime_spec": C4_RUNTIME_SPEC_PATH,
        "c4_runtime_launch_authorization": (
            C4_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        ),
        "c4_runtime_artifacts": C4_RUNTIME_ARTIFACT_ROOT,
        "c4_gpu_lease": C4_GPU_LEASE_ROOT,
        "c4_run_alias": C4_RUN_ROOT_ALIAS_PATH,
        "c4_result_alias": C4_RESULT_RECEIPT_ALIAS_PATH,
        "old_runtime_spec": OLD_RUNTIME_SPEC_PATH,
        "old_runtime_launch_authorization": (
            OLD_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        ),
        "old_runtime_artifacts": OLD_RUNTIME_ARTIFACT_ROOT,
        "old_gpu_lease": OLD_GPU_LEASE_ROOT,
        "c5_run_alias": C5_RUN_ROOT_ALIAS_PATH,
        "c5_result_alias": C5_RESULT_RECEIPT_ALIAS_PATH,
    }


def _preactivation_scientific_paths() -> dict[str, Path]:
    return {
        "scientific_run_root": SCIENTIFIC_RUN_ROOT,
        "scientific_result_receipt": SCIENTIFIC_RESULT_RECEIPT_PATH,
    }


def _c5_preauthorization_paths() -> dict[str, Path]:
    return {
        "c5_environment_policy": C5_ENVIRONMENT_POLICY_PATH,
        "c5_environment_scope_handoff": (
            C5_ENVIRONMENT_SCOPE_HANDOFF_PATH
        ),
        "c5_environment_stability_attempt": (
            C5_ENVIRONMENT_STABILITY_ATTEMPT_PATH
        ),
        "c5_environment_stability_terminal": (
            C5_ENVIRONMENT_STABILITY_TERMINAL_PATH
        ),
        "c5_environment_stability": C5_ENVIRONMENT_STABILITY_PATH,
        "c5_environment_postcleanup": C5_ENVIRONMENT_POSTCLEANUP_PATH,
        "c5_unit_authorization": C5_UNIT_AUTHORIZATION_PATH,
        "c5_unit_receipt": C5_UNIT_RECEIPT_PATH,
        "c5_unit_terminal": C5_UNIT_TERMINAL_PATH,
        "c5_unit_fragment": C5_UNIT_FRAGMENT_PATH,
        "c5_compatibility_terminal": C5_TERMINAL_PATH,
    }


def _c5_future_paths() -> dict[str, Path]:
    return {
        "c5_runtime_spec": C5_RUNTIME_SPEC_PATH,
        "c5_runtime_launch_authorization": (
            C5_RUNTIME_LAUNCH_AUTHORIZATION_PATH
        ),
        "c5_runtime_artifacts": C5_RUNTIME_ARTIFACT_ROOT,
        "c5_gpu_lease": C5_GPU_LEASE_ROOT,
    }


def _require_absent(paths: Mapping[str, Path]) -> None:
    present = [
        f"{label}:{path}"
        for label, path in paths.items()
        if os.path.lexists(path)
    ]
    if present:
        raise PermissionError(
            "protected compatibility/scientific path exists: "
            + ",".join(present)
        )


def _resolve_runtime_phase(
    *,
    allow_runtime_activation: bool,
    runtime_phase: str | None,
) -> str:
    """Resolve one explicit phase while retaining only safe legacy behavior."""

    if type(allow_runtime_activation) is not bool:
        raise TypeError("allow_runtime_activation must be boolean")
    if runtime_phase is None:
        if allow_runtime_activation:
            raise PermissionError(
                "active runtime verification requires an explicit phase"
            )
        return RUNTIME_PHASE_PREACTIVATION
    if not isinstance(runtime_phase, str) or runtime_phase not in RUNTIME_PHASES:
        raise PermissionError("runtime phase is not an exact closed state")
    expected_activation = runtime_phase != RUNTIME_PHASE_PREACTIVATION
    if allow_runtime_activation is not expected_activation:
        raise PermissionError(
            "runtime activation flag and explicit phase disagree"
        )
    return runtime_phase


def _validate_runtime_scientific_run_root(
    *,
    require_empty: bool,
) -> None:
    """Require the exact private original-r2 run directory generation."""

    if type(require_empty) is not bool:
        raise TypeError("require_empty must be boolean")
    target = SCIENTIFIC_RUN_ROOT.absolute()
    if not os.path.lexists(target):
        raise PermissionError("scientific r2 run root is absent")
    parent_before = _safe_parent(target)
    try:
        before = target.lstat()
        if (
            not stat.S_ISDIR(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or target.resolve(strict=True) != target
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o700
            or before.st_nlink < 2
        ):
            raise PermissionError(
                "scientific r2 run root is not exact private canonical"
            )
        if not hasattr(os, "O_NOFOLLOW"):
            raise RuntimeError("O_NOFOLLOW is required")
        descriptor = os.open(
            target,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            opened = os.fstat(descriptor)
            entries = tuple(os.listdir(descriptor)) if require_empty else ()
            after = os.fstat(descriptor)
            linked = target.lstat()
        finally:
            os.close(descriptor)
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        raise PermissionError(
            "scientific r2 run root changed while validating"
        ) from error
    parent_after = target.parent.lstat()
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(before, field) != getattr(observed, field)
        for field in identity_fields
        for observed in (opened, after, linked)
    ) or (
        parent_before.st_dev,
        parent_before.st_ino,
        parent_before.st_uid,
    ) != (
        parent_after.st_dev,
        parent_after.st_ino,
        parent_after.st_uid,
    ):
        raise PermissionError(
            "scientific r2 run root generation changed while validating"
        )
    if entries:
        raise PermissionError(
            "scientific r2 run root is not empty in pre-execution phase"
        )


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PermissionError(f"{name} is not exact SHA-256")
    return value


def _validate_r2_run_start_binding(
    receipt: Mapping[str, object],
) -> None:
    run_start = receipt.get("run_start_marker")
    marker = (
        run_start.get("payload")
        if isinstance(run_start, Mapping)
        else None
    )
    intent = marker.get("intent") if isinstance(marker, Mapping) else None
    if (
        not isinstance(run_start, Mapping)
        or set(run_start) != _R2_RUN_START_ENVELOPE_KEYS
        or not isinstance(marker, Mapping)
        or set(marker) != _R2_RUN_START_MARKER_KEYS
        or not isinstance(intent, Mapping)
        or set(intent) != _R2_RUN_INTENT_KEYS
    ):
        raise PermissionError("scientific r2 run-start structure changed")
    authorization_fingerprint = _require_sha256(
        receipt.get("preaccess_authorization_fingerprint"),
        name="r2 preaccess authorization fingerprint",
    )
    marker_fingerprint = _require_sha256(
        marker.get("marker_fingerprint"),
        name="r2 run-start marker fingerprint",
    )
    marker_body = dict(marker)
    marker_body.pop("marker_fingerprint")
    expected_marker_path = SCIENTIFIC_RUN_ROOT.absolute() / (
        R2_RUN_START_FILENAME_PREFIX
        + authorization_fingerprint
        + ".json"
    )
    expected_result_path = str(
        SCIENTIFIC_RESULT_RECEIPT_PATH.absolute()
    )
    implementation = receipt.get("implementation_binding")
    if (
        marker.get("schema_version") != R2_RUN_START_SCHEMA
        or marker.get("path_policy") != R2_RUN_START_PATH_POLICY
        or marker.get("stage_id") != R2_RUN_START_STAGE_ID
        or marker.get("run_id") != SCIENTIFIC_ATTEMPT_ID
        or marker.get("candidate") != CANDIDATE
        or marker.get("marker_path") != str(expected_marker_path)
        or marker.get("authorization_fingerprint")
        != authorization_fingerprint
        or marker.get("authorization_receipt_file_sha256")
        != receipt.get("preaccess_authorization_file_sha256")
        or marker.get("access_audit_receipt_fingerprint")
        != receipt.get("access_audit_receipt_fingerprint")
        or marker.get("access_audit_receipt_file_sha256")
        != receipt.get("access_audit_receipt_file_sha256")
        or marker.get("dataset_free_receipt_fingerprint")
        != receipt.get("dataset_free_receipt_fingerprint")
        or marker.get("dataset_free_receipt_file_sha256")
        != receipt.get("dataset_free_receipt_file_sha256")
        or marker.get("protocol_preregistration_fingerprint")
        != receipt.get("protocol_preregistration_fingerprint")
        or marker.get("source_closure_fingerprint")
        != receipt.get("source_closure_fingerprint")
        or marker.get("implementation_binding") != implementation
        or marker.get("expected_source_binding_fingerprint")
        != receipt.get("source_binding_fingerprint")
        or marker.get("expected_real_inputs_fingerprint")
        != receipt.get("real_inputs_fingerprint")
        or marker.get("expected_population_fingerprint")
        != receipt.get("population_fingerprint")
        or marker.get("expected_cache_fingerprint")
        != receipt.get("cache_fingerprint")
        or marker_fingerprint != stable_fingerprint(marker_body)
        or run_start.get("path") != str(expected_marker_path)
        or run_start.get("marker_fingerprint") != marker_fingerprint
        or intent.get("execution_kind") != R2_EXECUTION_KIND
        or intent.get("split") != "D_R"
        or intent.get("requested_device") != receipt.get("device")
        or intent.get("requested_receipt_output")
        != expected_result_path
        or intent.get("D_R_materialization_intended") is not True
        or intent.get("D_V_materialization_intended") is not False
        or intent.get("D_T_materialization_intended") is not False
        or type(intent.get("optimizer_steps_authorized")) is not int
        or intent.get("optimizer_steps_authorized") != 0
        or type(intent.get("parameter_updates_authorized")) is not int
        or intent.get("parameter_updates_authorized") != 0
        or intent.get("training_authorized") is not False
        or marker.get("intent_fingerprint")
        != stable_fingerprint(dict(intent))
    ):
        raise PermissionError("scientific r2 run-start binding changed")
    stored_marker, marker_root = _load_sealed(
        expected_marker_path,
        fingerprint_field="marker_fingerprint",
        schema=R2_RUN_START_SCHEMA,
    )
    if (
        stored_marker != dict(marker)
        or marker_root.get("path") != str(expected_marker_path)
        or marker_root.get("mode") != 0o444
        or marker_root.get("owner_uid") != os.getuid()
        or expected_marker_path.resolve(strict=True)
        != expected_marker_path
        or run_start.get("file_sha256")
        != marker_root.get("file_sha256")
    ):
        raise PermissionError(
            "scientific r2 persistent run-start marker changed"
        )


def _validate_r2_result_payload(
    receipt: Mapping[str, object],
) -> None:
    if set(receipt) != _R2_RESULT_KEYS:
        raise PermissionError("scientific r2 result keys changed")
    expected_result_path = str(
        SCIENTIFIC_RESULT_RECEIPT_PATH.absolute()
    )
    implementation = receipt.get("implementation_binding")
    raw = receipt.get("raw_observations")
    boundary = receipt.get("boundary")
    artifacts = receipt.get("artifact_hashes")
    if (
        receipt.get("schema_version") != R2_RESULT_SCHEMA
        or receipt.get("run_id") != SCIENTIFIC_ATTEMPT_ID
        or receipt.get("candidate") != CANDIDATE
        or receipt.get("execution_kind") != R2_EXECUTION_KIND
        or type(receipt.get("execution_seed")) is not int
        or receipt.get("execution_seed") != R2_EXECUTION_SEED
        or not isinstance(receipt.get("device"), str)
        or not receipt.get("device")
        or receipt.get("requested_receipt_output")
        != expected_result_path
        or not isinstance(implementation, Mapping)
        or receipt.get("source_closure_fingerprint")
        != stable_fingerprint(dict(implementation))
        or not isinstance(raw, Mapping)
        or receipt.get("raw_observations_fingerprint")
        != stable_fingerprint(dict(raw))
        or not isinstance(receipt.get("checks"), Mapping)
        or not isinstance(receipt.get("decision"), Mapping)
        or not isinstance(boundary, Mapping)
        or set(boundary) != _R2_BOUNDARY_KEYS
        or boundary.get("execution_kind") != R2_EXECUTION_KIND
        or boundary.get("split") != "D_R"
        or boundary.get("D_R_accessed") is not True
        or boundary.get("D_V_accessed") is not False
        or boundary.get("D_T_accessed") is not False
        or boundary.get("D_V_tensor_payload_accessed") is not False
        or boundary.get("D_T_tensor_payload_accessed") is not False
        or boundary.get("optimizer_module_referenced") is not False
        or boundary.get("optimizer_constructed") is not False
        or type(boundary.get("optimizer_steps")) is not int
        or boundary.get("optimizer_steps") != 0
        or type(boundary.get("parameter_updates")) is not int
        or boundary.get("parameter_updates") != 0
        or boundary.get("training_performed") is not False
        or boundary.get("performance_gate_present") is not False
        or boundary.get("performance_claim_supported") is not False
        or boundary.get("threshold_or_ratio_gate") is not None
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != _R2_ARTIFACT_HASH_KEYS
    ):
        raise PermissionError("scientific r2 result structure changed")
    sha_fields = (
        "dataset_free_receipt_fingerprint",
        "dataset_free_receipt_file_sha256",
        "efficiency_section_fingerprint",
        "efficiency_receipt_sha256",
        "preaccess_authorization_fingerprint",
        "preaccess_authorization_file_sha256",
        "access_audit_receipt_fingerprint",
        "access_audit_receipt_file_sha256",
        "protocol_preregistration_fingerprint",
        "source_closure_fingerprint",
        "source_binding_fingerprint",
        "real_inputs_fingerprint",
        "population_fingerprint",
        "cache_fingerprint",
        "adapter_fingerprint",
        "raw_observations_fingerprint",
        "receipt_fingerprint",
    )
    for field in sha_fields:
        _require_sha256(receipt.get(field), name=f"r2 result {field}")
    _validate_r2_run_start_binding(receipt)
    run_start = receipt["run_start_marker"]
    if (
        artifacts.get("dataset_free_receipt")
        != receipt.get("dataset_free_receipt_file_sha256")
        or artifacts.get("preaccess_authorization")
        != receipt.get("preaccess_authorization_file_sha256")
        or artifacts.get("preaccess_access_audit")
        != receipt.get("access_audit_receipt_file_sha256")
        or artifacts.get("persistent_run_start_marker")
        != run_start.get("file_sha256")
    ):
        raise PermissionError("scientific r2 artifact binding changed")


def _validate_runtime_scientific_result_receipt(
    *,
    required: bool,
) -> None:
    """Validate the exact sealed real-r2 receipt without scientific imports."""

    if type(required) is not bool:
        raise TypeError("required must be boolean")

    target = SCIENTIFIC_RESULT_RECEIPT_PATH.absolute()
    if not os.path.lexists(target):
        if required:
            raise PermissionError("scientific r2 result receipt is absent")
        return
    try:
        payload, root = _load_sealed(
            target,
            fingerprint_field="receipt_fingerprint",
            schema=R2_RESULT_SCHEMA,
        )
        if (
            root.get("path") != str(target)
            or root.get("mode") != 0o444
            or root.get("owner_uid") != os.getuid()
            or target.resolve(strict=True) != target
        ):
            raise PermissionError(
                "scientific r2 result receipt is not exactly sealed"
            )
        _validate_r2_result_payload(payload)
    except PermissionError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise PermissionError(
            "scientific r2 result receipt is not exact canonical sealed"
        ) from error


def _validate_scientific_output_phase(
    *,
    allow_runtime_activation: bool,
    runtime_phase: str | None = None,
) -> None:
    """Enforce the exact original-r2 output state for one runtime phase."""

    phase = _resolve_runtime_phase(
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=runtime_phase,
    )
    _require_absent(_always_absent_paths())
    if phase in _PRESPAWN_ABSENT_PHASES:
        _require_absent(_preactivation_scientific_paths())
        return
    if phase == RUNTIME_PHASE_FINALIZE_FAILURE:
        _require_absent(
            {"scientific_result_receipt": SCIENTIFIC_RESULT_RECEIPT_PATH}
        )
        if os.path.lexists(SCIENTIFIC_RUN_ROOT):
            _validate_runtime_scientific_run_root(require_empty=False)
        return
    if phase != RUNTIME_PHASE_FINALIZE_SUCCESS:
        raise AssertionError("unhandled c5 runtime phase")
    _validate_runtime_scientific_run_root(require_empty=False)
    _validate_runtime_scientific_result_receipt(required=True)


def _validate_common_identity(value: Mapping[str, object]) -> None:
    if (
        value.get("candidate") != CANDIDATE
        or value.get("stage_id") != STAGE_ID
        or value.get("scientific_attempt_id") != SCIENTIFIC_ATTEMPT_ID
        or value.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
    ):
        raise PermissionError("c5 scientific identity changed")


def _load_verified_c4_bridge() -> tuple[ModuleType, dict[str, object]]:
    """Load the exact B4 producer so B4 evidence keeps B4's JSON profile."""

    raw, _observed = _read_regular_bytes(C4_BRIDGE_SOURCE_PATH, sealed=False)
    if hashlib.sha256(raw).hexdigest() != C4_BRIDGE_SHA256:
        raise PermissionError("frozen c4 bridge source changed")
    name = "tools._cure_lite_v24_compat_c4_verified_for_c5_bridge"
    module = ModuleType(name)
    module.__file__ = str(C4_BRIDGE_SOURCE_PATH)
    module.__package__ = "tools"
    exec(
        compile(
            raw,
            str(C4_BRIDGE_SOURCE_PATH),
            "exec",
            dont_inherit=True,
        ),
        module.__dict__,
    )
    root = _source_root(C4_BRIDGE_SOURCE_PATH)
    if (
        root.get("file_sha256") != C4_BRIDGE_SHA256
        or Path(module.C4_AUTHORIZATION_PATH).absolute()
        != C4_AUTHORIZATION_PATH.absolute()
        or Path(module.C4_RECEIPT_PATH).absolute()
        != C4_RECEIPT_PATH.absolute()
        or module.AUTHORIZATION_SCHEMA != C4_AUTHORIZATION_SCHEMA
        or module.CANDIDATE != CANDIDATE
        or module.STAGE_ID != STAGE_ID
        or module.SCIENTIFIC_ATTEMPT_ID != SCIENTIFIC_ATTEMPT_ID
        or module.SCIENTIFIC_ATTEMPT_ORDINAL != SCIENTIFIC_ATTEMPT_ORDINAL
        or module.RUNTIME_COMPATIBILITY_ID != "c4"
        or module.C4_UNIT_NAME != C4_UNIT_NAME
        or not callable(module.validate_c4_authorization)
    ):
        raise PermissionError("frozen c4 bridge producer interface changed")
    return module, root


def _validate_c4_authorization_archival(
    *,
    unit_state_reader: UnitStateReader,
    allow_runtime_activation: bool,
    runtime_phase: str,
    now: Callable[[], datetime],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Validate B4 authorization only through its exact frozen producer."""

    module, source_root = _load_verified_c4_bridge()
    source_before, source_stat_before = _read_regular_bytes(
        C4_BRIDGE_SOURCE_PATH,
        sealed=False,
    )
    auth_before, auth_stat_before = _read_regular_bytes(
        C4_AUTHORIZATION_PATH,
        sealed=True,
    )
    authorization, root = module.validate_c4_authorization(
        C4_AUTHORIZATION_PATH,
        unit_state_reader=unit_state_reader,
        require_fresh=False,
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=runtime_phase,
        now=now,
    )
    source_after, source_stat_after = _read_regular_bytes(
        C4_BRIDGE_SOURCE_PATH,
        sealed=False,
    )
    auth_after, auth_stat_after = _read_regular_bytes(
        C4_AUTHORIZATION_PATH,
        sealed=True,
    )
    if (
        source_after != source_before
        or auth_after != auth_before
        or any(
            getattr(source_stat_before, field)
            != getattr(source_stat_after, field)
            for field in _STAT_GENERATION_FIELDS
        )
        or any(
            getattr(auth_stat_before, field) != getattr(auth_stat_after, field)
            for field in _STAT_GENERATION_FIELDS
        )
        or _source_root(C4_BRIDGE_SOURCE_PATH) != source_root
        or hashlib.sha256(source_after).hexdigest() != C4_BRIDGE_SHA256
        or not isinstance(authorization, Mapping)
        or not isinstance(root, Mapping)
        or root.get("path") != str(C4_AUTHORIZATION_PATH.absolute())
        or root.get("file_sha256") != hashlib.sha256(auth_after).hexdigest()
        or authorization.get("schema_version") != C4_AUTHORIZATION_SCHEMA
        or authorization.get("runtime_compatibility_id") != "c4"
        or authorization.get("scientific_attempt_id")
        != SCIENTIFIC_ATTEMPT_ID
        or authorization.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or authorization.get("scientific_authority", {}).get(
            "fresh_scientific_attempt"
        )
        is not False
        or authorization.get("mutation_authority", {}).get(
            "runtime_spec_creation_authorized"
        )
        is not False
    ):
        raise PermissionError("archival c4 authorization producer diverged")
    _require_no_payload(authorization)
    return dict(authorization), dict(root), source_root


def _load_verified_c4_receipt_seal_failure_terminalizer(
) -> tuple[ModuleType, dict[str, object]]:
    """Load only the frozen producer for the direct C4 FAIL boundary."""

    _require_frozen_c4_failure_hashes()
    raw, _observed = _read_regular_bytes(
        C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SOURCE_PATH,
        sealed=False,
    )
    if (
        hashlib.sha256(raw).hexdigest()
        != C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SHA256
    ):
        raise PermissionError("c4 receipt-seal terminalizer source changed")
    name = (
        "tools._cure_lite_v24_c4_receipt_seal_failure_terminal_"
        "verified_for_c5_bridge"
    )
    module = ModuleType(name)
    module.__file__ = str(C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SOURCE_PATH)
    module.__package__ = "tools"
    exec(
        compile(
            raw,
            str(C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SOURCE_PATH),
            "exec",
            dont_inherit=True,
        ),
        module.__dict__,
    )
    root = _source_root(C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SOURCE_PATH)
    if (
        root.get("file_sha256")
        != C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SHA256
        or Path(module.TERMINAL_PATH).absolute()
        != C4_RECEIPT_SEAL_FAILURE_TERMINAL_PATH.absolute()
        or module.SCHEMA != C4_RECEIPT_SEAL_FAILURE_TERMINAL_SCHEMA
        or module.CANDIDATE != CANDIDATE
        or module.STAGE_ID != STAGE_ID
        or module.SCIENTIFIC_ATTEMPT_ID != SCIENTIFIC_ATTEMPT_ID
        or module.SCIENTIFIC_ATTEMPT_ORDINAL != SCIENTIFIC_ATTEMPT_ORDINAL
        or module.RUNTIME_COMPATIBILITY_ID != "c4"
        or module.C4_UNIT_NAME != C4_UNIT_NAME
        or not callable(module.validate_archival)
    ):
        raise PermissionError("c4 receipt-seal terminalizer interface changed")
    return module, root


def _validate_c4_receipt_seal_failure_terminal(
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Validate C4 as a sealed FAIL, never as a compatibility PASS."""

    module, source_root = (
        _load_verified_c4_receipt_seal_failure_terminalizer()
    )
    source_before, source_stat_before = _read_regular_bytes(
        C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SOURCE_PATH,
        sealed=False,
    )
    terminal_before, terminal_stat_before = _read_regular_bytes(
        C4_RECEIPT_SEAL_FAILURE_TERMINAL_PATH,
        sealed=True,
    )
    payload, root = module.validate_archival(
        C4_RECEIPT_SEAL_FAILURE_TERMINAL_PATH
    )
    source_after, source_stat_after = _read_regular_bytes(
        C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SOURCE_PATH,
        sealed=False,
    )
    terminal_after, terminal_stat_after = _read_regular_bytes(
        C4_RECEIPT_SEAL_FAILURE_TERMINAL_PATH,
        sealed=True,
    )
    identity = payload.get("identity") if isinstance(payload, Mapping) else None
    failure = (
        payload.get("b4_receipt_seal_failure")
        if isinstance(payload, Mapping)
        else None
    )
    observation = (
        payload.get("original_execution_observation")
        if isinstance(payload, Mapping)
        else None
    )
    closure = (
        payload.get("metadata_success_closure")
        if isinstance(payload, Mapping)
        else None
    )
    continuation = (
        payload.get("continuation_policy")
        if isinstance(payload, Mapping)
        else None
    )
    payload_observation = (
        payload.get("payload_observation")
        if isinstance(payload, Mapping)
        else None
    )
    expiry = (
        payload.get("authorization_expiry")
        if isinstance(payload, Mapping)
        else None
    )
    historical = (
        payload.get("historical_state_observation")
        if isinstance(payload, Mapping)
        else None
    )
    reproduction = (
        payload.get("deterministic_reproduction")
        if isinstance(payload, Mapping)
        else None
    )
    expected_terminalizer_source_root = {
        "path": str(
            C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SOURCE_PATH.absolute()
        ),
        "file_sha256": hashlib.sha256(source_after).hexdigest(),
        "device": source_stat_after.st_dev,
        "inode": source_stat_after.st_ino,
        "owner_uid": source_stat_after.st_uid,
        "owner_gid": source_stat_after.st_gid,
        "mode": stat.S_IMODE(source_stat_after.st_mode),
        "nlink": source_stat_after.st_nlink,
        "size": source_stat_after.st_size,
        "mtime_ns": source_stat_after.st_mtime_ns,
        "ctime_ns": source_stat_after.st_ctime_ns,
    }
    if (
        source_after != source_before
        or terminal_after != terminal_before
        or any(
            getattr(source_stat_before, field)
            != getattr(source_stat_after, field)
            for field in _STAT_GENERATION_FIELDS
        )
        or any(
            getattr(terminal_stat_before, field)
            != getattr(terminal_stat_after, field)
            for field in _STAT_GENERATION_FIELDS
        )
        or hashlib.sha256(source_after).hexdigest()
        != C4_RECEIPT_SEAL_FAILURE_TERMINALIZER_SHA256
        or hashlib.sha256(terminal_after).hexdigest()
        != C4_RECEIPT_SEAL_FAILURE_TERMINAL_SHA256
        or not isinstance(root, Mapping)
        or root.get("path")
        != str(C4_RECEIPT_SEAL_FAILURE_TERMINAL_PATH.absolute())
        or root.get("file_sha256")
        != C4_RECEIPT_SEAL_FAILURE_TERMINAL_SHA256
        or root.get("terminal_fingerprint")
        != C4_RECEIPT_SEAL_FAILURE_TERMINAL_FINGERPRINT
        or root.get("schema_version")
        != C4_RECEIPT_SEAL_FAILURE_TERMINAL_SCHEMA
        or root.get("terminalizer_source_root")
        != expected_terminalizer_source_root
        or not isinstance(identity, Mapping)
        or identity.get("candidate") != CANDIDATE
        or identity.get("stage_id") != STAGE_ID
        or identity.get("scientific_attempt_id") != SCIENTIFIC_ATTEMPT_ID
        or identity.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or identity.get("runtime_compatibility_id") != "c4"
        or identity.get("failure_stage")
        != "B4_compatibility_receipt_seal"
        or not isinstance(failure, Mapping)
        or failure.get("first_rejected_path")
        != str(C4_UNIT_AUTHORIZATION_PATH.absolute())
        or failure.get("fingerprint_field") != "authorization_fingerprint"
        or failure.get("producer_canonical_profile")
        != "compact_sorted_ensure_ascii_false_utf8"
        or failure.get("consumer_canonical_profile")
        != "compact_sorted_ensure_ascii_true_utf8"
        or failure.get("producer_fingerprint")
        != "543f794fd27e6277471eb2e52ab290a228415091c3071070cf3f0920c3d28c10"
        or failure.get("consumer_recomputed_fingerprint")
        != "11b4f19ae10d7b032af4eb7611e8b36155be6cf577149450128d6b439b14cb44"
        or failure.get("profile_mismatch") is not True
        or failure.get("receipt_writer_reached") is not False
        or failure.get("receipt_sealed") is not False
        or not isinstance(observation, Mapping)
        or observation.get("attempt_count") != 1
        or observation.get("control_plane_observed_exit_code") != 1
        or observation.get("durable_original_execution_artifact") is not False
        or observation.get("exit_code_independently_verifiable") is not False
        or observation.get("original_argv_claimed") is not False
        or observation.get("original_stdout_claimed") is not False
        or observation.get("original_stderr_claimed") is not False
        or observation.get("original_traceback_claimed") is not False
        or not isinstance(closure, Mapping)
        or closure.get("r4_unit_realization_passed") is not True
        or closure.get("r4_static_unit_verified") is not True
        or closure.get("e4_scope_handoff_present") is not True
        or closure.get("e4_stability_attempt_count") != 1
        or closure.get("e4_environment_sample_count") != 2
        or closure.get("e4_stability_passed") is not True
        or closure.get("e4_postcleanup_passed") is not True
        or closure.get("c4_compatibility_receipt_present") is not False
        or not isinstance(continuation, Mapping)
        or continuation.get("automatic_retry") is not False
        or continuation.get("same_c4_reentry") is not False
        or continuation.get("same_c4_reauthorization") is not False
        or continuation.get("same_c4_source_repair") is not False
        or continuation.get("same_c4_loader_patch") is not False
        or continuation.get("same_c4_receipt_seal_reentry") is not False
        or continuation.get("r4_e4_reentry") is not False
        or continuation.get("r14_l4_runtime_scientific_launch") is not False
        or continuation.get("c5_required") is not True
        or continuation.get("new_explicit_authorization_required") is not True
        or continuation.get("scientific_attempt_consumed") is not False
        or continuation.get("terminal_grants_c5_reuse_authority") is not False
        or continuation.get("b4_authorization_consumed") is not True
        or continuation.get("r4_unit_realization_consumed") is not True
        or continuation.get("e4_metadata_attempt_consumed") is not True
        or continuation.get("runtime_launch_consumed") is not False
        or continuation.get("runtime_materialization_consumed") is not False
        or continuation.get("scientific_attempt_id")
        != SCIENTIFIC_ATTEMPT_ID
        or continuation.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or continuation.get("scientific_attempt_id_unchanged") is not True
        or continuation.get("scientific_attempt_ordinal_unchanged") is not True
        or not isinstance(payload_observation, Mapping)
        or payload_observation.get("D_R_payload_accessed") is not False
        or payload_observation.get("D_V_payload_accessed") is not False
        or payload_observation.get("D_T_payload_accessed") is not False
        or payload_observation.get("gpu_compute_accessed") is not False
        or payload_observation.get("training_started") is not False
        or payload_observation.get("scientific_samples_processed") != 0
        or payload_observation.get("optimizer_steps") != 0
        or payload_observation.get("parameter_updates") != 0
        or payload_observation.get("scientific_attempt_consumed") is not False
        or not isinstance(expiry, Mapping)
        or expiry.get("B4_expired") is not True
        or expiry.get("R4_expired") is not True
        or expiry.get("B4_compatibility_receipt_absent_at_observation")
        is not True
        or expiry.get("B4_sealed_by_compatibility_receipt") is not False
        or not isinstance(historical, Mapping)
        or historical.get("historical_observation_only") is not True
        or historical.get("future_state_authority") is not False
        or historical.get("archival_live_absence_recheck_required")
        is not False
        or historical.get("archival_live_manager_recheck_required")
        is not False
        or not isinstance(reproduction, Mapping)
        or reproduction.get("first_failure_stage")
        != "R4_authorization_fingerprint_validation"
        or reproduction.get("first_failure_reproduced") is not True
        or reproduction.get("retry_or_replay_performed") is not False
        or reproduction.get("systemd_mutation_performed") is not False
        or reproduction.get("gpu_or_payload_accessed") is not False
        or reproduction.get("old_B4_seal_called") is not False
        or reproduction.get("R4_E4_writer_called") is not False
    ):
        raise PermissionError("c4 receipt-seal failure transition changed")
    return dict(payload), dict(root), source_root


def _load_verified_terminalizer() -> tuple[ModuleType, dict[str, object]]:
    raw, _observed = _read_regular_bytes(
        C1_FAILURE_TERMINALIZER_SOURCE_PATH,
        sealed=False,
    )
    if hashlib.sha256(raw).hexdigest() != C1_FAILURE_TERMINALIZER_SHA256:
        raise PermissionError("c1 terminalizer source changed")
    name = "tools._cure_lite_v24_c1_expired_terminal_verified_for_c2"
    module = ModuleType(name)
    module.__file__ = str(C1_FAILURE_TERMINALIZER_SOURCE_PATH)
    module.__package__ = "tools"
    exec(
        compile(
            raw,
            str(C1_FAILURE_TERMINALIZER_SOURCE_PATH),
            "exec",
            dont_inherit=True,
        ),
        module.__dict__,
    )
    root = _source_root(C1_FAILURE_TERMINALIZER_SOURCE_PATH)
    if (
        root["file_sha256"] != C1_FAILURE_TERMINALIZER_SHA256
        or Path(module.TERMINAL_PATH).absolute()
        != C1_FAILURE_TERMINAL_PATH.absolute()
        or module.CANDIDATE != CANDIDATE
        or module.STAGE_ID != STAGE_ID
        or module.SCIENTIFIC_ATTEMPT_ID != SCIENTIFIC_ATTEMPT_ID
        or module.SCIENTIFIC_ATTEMPT_ORDINAL
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or module.RUNTIME_COMPATIBILITY_ID != "c1"
        or module.UNIT_NAME != C1_UNIT_NAME
        or not isinstance(module.SCHEMA, str)
        or not module.SCHEMA
        or not callable(module.validate_terminal)
    ):
        raise PermissionError("c1 terminalizer interface changed")
    return module, root


def _load_verified_environment_wrapper(
    expected_root: Mapping[str, object],
) -> tuple[ModuleType, dict[str, object]]:
    """Load exactly the wrapper generation captured by c5 authorization.

    The wrapper is intentionally not hash-pinned in this bridge: doing so
    would form a source-hash cycle once the wrapper pins the c5 realizer.  The
    create-once authorization captures the wrapper's full source root, and
    every later phase requires that same file generation before and after
    execution.
    """

    if not isinstance(expected_root, Mapping):
        raise PermissionError("c5 environment wrapper root is malformed")
    expected = dict(expected_root)
    _validate_source_root(
        expected,
        expected_path=C5_ENVIRONMENT_WRAPPER_SOURCE_PATH,
    )
    raw, _observed = _read_regular_bytes(
        C5_ENVIRONMENT_WRAPPER_SOURCE_PATH,
        sealed=False,
    )
    if hashlib.sha256(raw).hexdigest() != expected.get("file_sha256"):
        raise PermissionError("c5 environment wrapper bytes changed")
    name = "tools._cure_lite_v24_environment_compat_c5_verified_for_bridge"
    module = ModuleType(name)
    module.__file__ = str(C5_ENVIRONMENT_WRAPPER_SOURCE_PATH)
    module.__package__ = "tools"
    sys.modules[name] = module
    try:
        exec(
            compile(
                raw,
                str(C5_ENVIRONMENT_WRAPPER_SOURCE_PATH),
                "exec",
                dont_inherit=True,
            ),
            module.__dict__,
        )
        after = _source_root(C5_ENVIRONMENT_WRAPPER_SOURCE_PATH)
        if after != expected:
            raise PermissionError(
                "c5 environment wrapper generation changed while loading"
            )
        if (
            Path(module.C5_POLICY_PATH).absolute()
            != C5_ENVIRONMENT_POLICY_PATH.absolute()
            or Path(module.C5_SCOPE_HANDOFF_PATH).absolute()
            != C5_ENVIRONMENT_SCOPE_HANDOFF_PATH.absolute()
            or Path(module.C5_STABILITY_ATTEMPT_PATH).absolute()
            != C5_ENVIRONMENT_STABILITY_ATTEMPT_PATH.absolute()
            or Path(module.C5_STABILITY_TERMINAL_PATH).absolute()
            != C5_ENVIRONMENT_STABILITY_TERMINAL_PATH.absolute()
            or Path(module.C5_STABILITY_PATH).absolute()
            != C5_ENVIRONMENT_STABILITY_PATH.absolute()
            or Path(module.C5_POSTCLEANUP_PATH).absolute()
            != C5_ENVIRONMENT_POSTCLEANUP_PATH.absolute()
            or Path(module.C5_REALIZATION_AUTHORIZATION_PATH).absolute()
            != C5_UNIT_AUTHORIZATION_PATH.absolute()
            or Path(module.C5_REALIZATION_RECEIPT_PATH).absolute()
            != C5_UNIT_RECEIPT_PATH.absolute()
            or module.C5_TARGET_UNIT != C5_UNIT_NAME
            or not callable(module.replay_old_scope_and_handoff)
            or not callable(module.validate_c5_environment_closure)
            or not callable(module.load_c5_environment_closure)
        ):
            raise PermissionError("c5 environment wrapper interface changed")
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, after


def _load_verified_unit_realizer(
    expected_root: Mapping[str, object],
) -> tuple[ModuleType, dict[str, object]]:
    """Load the exact R5 producer generation captured by B5 authorization."""

    if not isinstance(expected_root, Mapping):
        raise PermissionError("c5 unit realizer root is malformed")
    expected = dict(expected_root)
    _validate_source_root(
        expected,
        expected_path=C5_UNIT_REALIZER_SOURCE_PATH,
    )
    raw, _observed = _read_regular_bytes(
        C5_UNIT_REALIZER_SOURCE_PATH,
        sealed=False,
    )
    if hashlib.sha256(raw).hexdigest() != expected.get("file_sha256"):
        raise PermissionError("c5 unit realizer bytes changed")
    name = "tools._cure_lite_v24_realizer_compat_c5_verified_for_bridge"
    module = ModuleType(name)
    module.__file__ = str(C5_UNIT_REALIZER_SOURCE_PATH)
    module.__package__ = "tools"
    sys.modules[name] = module
    try:
        exec(
            compile(
                raw,
                str(C5_UNIT_REALIZER_SOURCE_PATH),
                "exec",
                dont_inherit=True,
            ),
            module.__dict__,
        )
        after = _source_root(C5_UNIT_REALIZER_SOURCE_PATH)
        if (
            after != expected
            or module.COMPAT_UNIT != C5_UNIT_NAME
            or Path(module.COMPAT_BRIDGE_AUTHORIZATION_PATH).absolute()
            != C5_AUTHORIZATION_PATH.absolute()
            or Path(module.COMPAT_AUTHORIZATION_PATH).absolute()
            != C5_UNIT_AUTHORIZATION_PATH.absolute()
            or Path(module.COMPAT_RECEIPT_PATH).absolute()
            != C5_UNIT_RECEIPT_PATH.absolute()
            or Path(module.COMPAT_TERMINAL_PATH).absolute()
            != C5_UNIT_TERMINAL_PATH.absolute()
            or not callable(module.validate_archival_realization_chain)
        ):
            raise PermissionError("c5 unit realizer producer interface changed")
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, after


def _normalize_environment_contract(contract: object) -> dict[str, object]:
    if not is_dataclass(contract) or isinstance(contract, type):
        raise PermissionError("c5 environment contract is not a dataclass")
    value = asdict(contract)
    if not isinstance(value, dict):
        raise PermissionError("c5 environment contract is malformed")
    return json.loads(_canonical_json(value))


def _validate_c2_prewrite_failure_terminal(
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    source_raw_before, source_before = _read_regular_bytes(
        C2_PREWRITE_FAILURE_TERMINALIZER_SOURCE_PATH,
        sealed=False,
    )
    terminal_raw_before, terminal_before = _read_regular_bytes(
        C2_PREWRITE_FAILURE_TERMINAL_PATH,
        sealed=True,
    )
    source_digest = hashlib.sha256(source_raw_before).hexdigest()
    terminal_digest = hashlib.sha256(terminal_raw_before).hexdigest()
    if (
        source_digest != C2_PREWRITE_FAILURE_TERMINALIZER_SHA256
        or terminal_digest != C2_PREWRITE_FAILURE_TERMINAL_SHA256
        or stat.S_IMODE(terminal_before.st_mode) != 0o444
    ):
        raise PermissionError("c2 prewrite failure lineage changed")
    try:
        payload = json.loads(terminal_raw_before)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(
            "c2 prewrite failure terminal is malformed"
        ) from error
    if (
        not isinstance(payload, dict)
        or terminal_raw_before
        != (_canonical_json(payload) + "\n").encode()
        or payload.get("schema_version")
        != C2_PREWRITE_FAILURE_TERMINAL_SCHEMA
    ):
        raise PermissionError("c2 prewrite failure terminal layout changed")
    fingerprint = payload.get("terminal_fingerprint")
    body = dict(payload)
    body.pop("terminal_fingerprint", None)
    identity = payload.get("identity")
    if (
        fingerprint != stable_fingerprint(body)
        or not isinstance(identity, Mapping)
        or identity.get("candidate") != CANDIDATE
        or identity.get("stage_id") != STAGE_ID
        or identity.get("scientific_attempt_id")
        != SCIENTIFIC_ATTEMPT_ID
        or identity.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or identity.get("runtime_compatibility_id") != "c2"
    ):
        raise PermissionError("c2 prewrite failure terminal identity changed")
    source_root = {
        "path": str(
            C2_PREWRITE_FAILURE_TERMINALIZER_SOURCE_PATH.absolute()
        ),
        "file_sha256": source_digest,
        "device": source_before.st_dev,
        "inode": source_before.st_ino,
        "mode": stat.S_IMODE(source_before.st_mode),
        "owner_uid": source_before.st_uid,
        "size": source_before.st_size,
    }
    if payload.get("terminalizer_source_root") != {
        **source_root,
        "nlink": source_before.st_nlink,
    }:
        raise PermissionError("c2 prewrite terminalizer lineage changed")
    source_raw_after, source_after = _read_regular_bytes(
        C2_PREWRITE_FAILURE_TERMINALIZER_SOURCE_PATH,
        sealed=False,
    )
    terminal_raw_after, terminal_after = _read_regular_bytes(
        C2_PREWRITE_FAILURE_TERMINAL_PATH,
        sealed=True,
    )
    if (
        source_raw_after != source_raw_before
        or terminal_raw_after != terminal_raw_before
        or any(
            getattr(source_before, field) != getattr(source_after, field)
            for field in _STAT_GENERATION_FIELDS
        )
        or any(
            getattr(terminal_before, field) != getattr(terminal_after, field)
            for field in _STAT_GENERATION_FIELDS
        )
    ):
        raise PermissionError("c2 prewrite failure lineage changed")
    terminal_root = {
        "path": str(C2_PREWRITE_FAILURE_TERMINAL_PATH.absolute()),
        "file_sha256": terminal_digest,
        "device": terminal_after.st_dev,
        "inode": terminal_after.st_ino,
        "mode": stat.S_IMODE(terminal_after.st_mode),
        "owner_uid": terminal_after.st_uid,
        "nlink": terminal_after.st_nlink,
        "size": terminal_after.st_size,
        "terminal_fingerprint": fingerprint,
        "schema_version": C2_PREWRITE_FAILURE_TERMINAL_SCHEMA,
    }
    return dict(payload), terminal_root, source_root


def _load_verified_mode_contract_failure_terminalizer(
) -> tuple[ModuleType, dict[str, object]]:
    raw, _observed = _read_regular_bytes(
        C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SOURCE_PATH,
        sealed=False,
    )
    if (
        hashlib.sha256(raw).hexdigest()
        != C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SHA256
    ):
        raise PermissionError("c2 mode-contract terminalizer source changed")

    name = "tools._cure_lite_v24_c2_mode_contract_terminal_verified_for_c5_bridge"
    module = ModuleType(name)
    module.__file__ = str(C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SOURCE_PATH)
    module.__package__ = "tools"
    exec(
        compile(
            raw,
            str(C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SOURCE_PATH),
            "exec",
            dont_inherit=True,
        ),
        module.__dict__,
    )
    root = _source_root(C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SOURCE_PATH)
    if (
        root.get("file_sha256")
        != C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SHA256
        or Path(module.TERMINAL_PATH).absolute()
        != C2_MODE_CONTRACT_FAILURE_TERMINAL_PATH.absolute()
        or module.SCHEMA != C2_MODE_CONTRACT_FAILURE_TERMINAL_SCHEMA
        or not callable(module.validate_archival)
    ):
        raise PermissionError("c2 mode-contract terminalizer interface changed")
    return module, root


def _validate_mode_contract_failure_terminal(
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    # T and F are independently byte-pinned before archival code can run.
    module, source_root = _load_verified_mode_contract_failure_terminalizer()
    terminalizer_raw_before, terminalizer_before = _read_regular_bytes(
        C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SOURCE_PATH,
        sealed=False,
    )
    if (
        hashlib.sha256(terminalizer_raw_before).hexdigest()
        != C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SHA256
        or _source_root(C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SOURCE_PATH)
        != source_root
    ):
        raise PermissionError("c2 mode-contract terminalizer source changed")
    raw_before, before = _read_regular_bytes(
        C2_MODE_CONTRACT_FAILURE_TERMINAL_PATH,
        sealed=True,
    )
    if (
        hashlib.sha256(raw_before).hexdigest()
        != C2_MODE_CONTRACT_FAILURE_TERMINAL_SHA256
        or stat.S_IMODE(before.st_mode) != 0o444
    ):
        raise PermissionError("c2 mode-contract failure terminal changed")

    payload, root = module.validate_archival(
        C2_MODE_CONTRACT_FAILURE_TERMINAL_PATH
    )
    terminalizer_raw_after, terminalizer_after = _read_regular_bytes(
        C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SOURCE_PATH,
        sealed=False,
    )
    raw_after, after = _read_regular_bytes(
        C2_MODE_CONTRACT_FAILURE_TERMINAL_PATH,
        sealed=True,
    )
    generation_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_uid",
        "st_gid",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        terminalizer_raw_after != terminalizer_raw_before
        or any(
            getattr(terminalizer_before, field)
            != getattr(terminalizer_after, field)
            for field in generation_fields
        )
        or hashlib.sha256(terminalizer_raw_after).hexdigest()
        != C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SHA256
        or _source_root(C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SOURCE_PATH)
        != source_root
        or raw_after != raw_before
        or any(
            getattr(before, field) != getattr(after, field)
            for field in generation_fields
        )
        or hashlib.sha256(raw_after).hexdigest()
        != C2_MODE_CONTRACT_FAILURE_TERMINAL_SHA256
        or not isinstance(payload, Mapping)
        or not isinstance(root, Mapping)
    ):
        raise PermissionError("c2 mode-contract failure terminal changed")

    terminalizer_archival_root = {
        "path": str(
            C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SOURCE_PATH.absolute()
        ),
        "file_sha256": C2_MODE_CONTRACT_FAILURE_TERMINALIZER_SHA256,
        "device": terminalizer_after.st_dev,
        "inode": terminalizer_after.st_ino,
        "owner_uid": terminalizer_after.st_uid,
        "owner_gid": terminalizer_after.st_gid,
        "mode": stat.S_IMODE(terminalizer_after.st_mode),
        "nlink": terminalizer_after.st_nlink,
        "size": terminalizer_after.st_size,
        "mtime_ns": terminalizer_after.st_mtime_ns,
        "ctime_ns": terminalizer_after.st_ctime_ns,
    }
    expected_root = {
        "path": str(C2_MODE_CONTRACT_FAILURE_TERMINAL_PATH.absolute()),
        "file_sha256": C2_MODE_CONTRACT_FAILURE_TERMINAL_SHA256,
        "device": after.st_dev,
        "inode": after.st_ino,
        "owner_uid": after.st_uid,
        "owner_gid": after.st_gid,
        "mode": stat.S_IMODE(after.st_mode),
        "nlink": after.st_nlink,
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
        "terminal_fingerprint": payload.get("terminal_fingerprint"),
        "schema_version": C2_MODE_CONTRACT_FAILURE_TERMINAL_SCHEMA,
        "terminalizer_source_root": terminalizer_archival_root,
    }
    try:
        decoded = json.loads(raw_before)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(
            "c2 mode-contract failure terminal is malformed"
        ) from error
    if dict(root) != expected_root or decoded != dict(payload):
        raise PermissionError("c2 mode-contract failure terminal root diverged")

    identity = payload.get("identity")
    failure = payload.get("mode_contract_failure")
    reproduction = payload.get("deterministic_reproduction")
    historical = payload.get("historical_absence_observation")
    continuation = payload.get("continuation_policy")
    if (
        not isinstance(identity, Mapping)
        or identity.get("candidate") != CANDIDATE
        or identity.get("stage_id") != STAGE_ID
        or identity.get("scientific_attempt_id")
        != SCIENTIFIC_ATTEMPT_ID
        or identity.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or identity.get("runtime_compatibility_id") != "c2"
        or not isinstance(failure, Mapping)
        or failure.get("producer") != (
            "cure_lite_v24_preaccess_schema_compatibility_c2._write_sealed"
        )
        or failure.get("consumer") != (
            "cure_lite_v24_actual_unit_realization_preaccess_compat_c2."
            "_validate_c2_bridge_authorization"
        )
        or failure.get("producer_observed_mode") != 0o400
        or failure.get("consumer_required_mode") != 0o444
        or failure.get("mode_contract_mismatch") is not True
        or failure.get("failed_before_unit_authorization_write") is not True
        or failure.get("original_call_artifact_claimed") is not False
        or failure.get("original_failure_time_claimed") is not False
        or failure.get("original_write_capable_entrypoint_invoked") is not True
        or failure.get("unit_authorization_written") is not False
        or failure.get("unit_terminal_written") is not False
        or not isinstance(reproduction, Mapping)
        or reproduction.get("validator")
        != "_validate_c2_bridge_authorization"
        or reproduction.get("observation_kind")
        != "read_only_post_hoc_reproduction"
        or reproduction.get("require_fresh") is not False
        or reproduction.get("require_future_absence") is not True
        or reproduction.get("write_capable_entrypoint_invoked") is not False
        or reproduction.get("exception_type") != "PermissionError"
        or reproduction.get("exception_message")
        != "c2 bridge does not authorize the narrow unit lane"
        or reproduction.get("exception_args")
        != ["c2 bridge does not authorize the narrow unit lane"]
        or reproduction.get("reproduced") is not True
        or not isinstance(historical, Mapping)
        or historical.get("historical_observation_only") is not True
        or historical.get("future_state_authority") is not False
        or historical.get("archival_live_absence_recheck_required") is not False
        or not isinstance(continuation, Mapping)
        or continuation.get("automatic_retry") is not False
        or continuation.get("same_c2_reentry") is not False
        or continuation.get("same_c2_reauthorization_allowed") is not False
        or continuation.get("same_c2_metadata_repair_allowed") is not False
        or continuation.get("c3_required") is not True
        or continuation.get("new_explicit_authorization_required") is not True
        or continuation.get("scientific_attempt_id")
        != SCIENTIFIC_ATTEMPT_ID
        or continuation.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or continuation.get("scientific_attempt_consumed") is not False
        or continuation.get("unit_realization_consumed") is not False
        or continuation.get("runtime_launch_consumed") is not False
        or continuation.get("materialization_consumed") is not False
    ):
        raise PermissionError("c2 mode-contract failure transition changed")

    # F's payload and historical-live claims are not C5 authority.  Only the
    # fixed source/evidence lineage and the historical C2 -> C3 transition are
    # consumed.  C3 only authorizes the historical C3 -> C4 transition; direct
    # C5 continuation authority comes from the sealed C4 failure terminal.
    return dict(payload), expected_root, source_root


def _load_verified_c3_environment_failure_terminalizer(
) -> tuple[ModuleType, dict[str, object]]:
    """Byte-pin the historical C3 predecessor before archival validation."""

    _require_frozen_c3_failure_hashes()
    raw, _observed = _read_regular_bytes(
        C3_ENVIRONMENT_FAILURE_TERMINALIZER_SOURCE_PATH,
        sealed=False,
    )
    if (
        hashlib.sha256(raw).hexdigest()
        != C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256
    ):
        raise PermissionError(
            "c3 environment-failure terminalizer source changed"
        )
    name = (
        "tools._cure_lite_v24_c3_environment_failure_terminal_"
        "verified_for_c5_bridge"
    )
    module = ModuleType(name)
    module.__file__ = str(
        C3_ENVIRONMENT_FAILURE_TERMINALIZER_SOURCE_PATH
    )
    module.__package__ = "tools"
    exec(
        compile(
            raw,
            str(C3_ENVIRONMENT_FAILURE_TERMINALIZER_SOURCE_PATH),
            "exec",
            dont_inherit=True,
        ),
        module.__dict__,
    )
    root = _source_root(
        C3_ENVIRONMENT_FAILURE_TERMINALIZER_SOURCE_PATH
    )
    if (
        root.get("file_sha256")
        != C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256
        or Path(module.TERMINAL_PATH).absolute()
        != C3_ENVIRONMENT_FAILURE_TERMINAL_PATH.absolute()
        or module.SCHEMA != C3_ENVIRONMENT_FAILURE_TERMINAL_SCHEMA
        or module.CANDIDATE != CANDIDATE
        or module.STAGE_ID != STAGE_ID
        or module.SCIENTIFIC_ATTEMPT_ID != SCIENTIFIC_ATTEMPT_ID
        or module.SCIENTIFIC_ATTEMPT_ORDINAL
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or module.RUNTIME_COMPATIBILITY_ID != "c3"
        or module.C3_UNIT_NAME != C3_UNIT_NAME
        or not callable(module.validate_archival)
    ):
        raise PermissionError(
            "c3 environment-failure terminalizer interface changed"
        )
    return module, root


def _validate_c3_environment_failure_terminal(
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Validate the exact sealed C3 FAIL boundary without live re-entry."""

    module, source_root = (
        _load_verified_c3_environment_failure_terminalizer()
    )
    source_raw_before, source_before = _read_regular_bytes(
        C3_ENVIRONMENT_FAILURE_TERMINALIZER_SOURCE_PATH,
        sealed=False,
    )
    terminal_raw_before, terminal_before = _read_regular_bytes(
        C3_ENVIRONMENT_FAILURE_TERMINAL_PATH,
        sealed=True,
    )
    if (
        hashlib.sha256(source_raw_before).hexdigest()
        != C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256
        or hashlib.sha256(terminal_raw_before).hexdigest()
        != C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256
        or _source_root(
            C3_ENVIRONMENT_FAILURE_TERMINALIZER_SOURCE_PATH
        )
        != source_root
        or stat.S_IMODE(terminal_before.st_mode) != 0o444
    ):
        raise PermissionError("c3 environment-failure lineage changed")

    payload, terminal_root = module.validate_archival(
        C3_ENVIRONMENT_FAILURE_TERMINAL_PATH
    )
    source_raw_after, source_after = _read_regular_bytes(
        C3_ENVIRONMENT_FAILURE_TERMINALIZER_SOURCE_PATH,
        sealed=False,
    )
    terminal_raw_after, terminal_after = _read_regular_bytes(
        C3_ENVIRONMENT_FAILURE_TERMINAL_PATH,
        sealed=True,
    )
    if (
        source_raw_after != source_raw_before
        or terminal_raw_after != terminal_raw_before
        or any(
            getattr(source_before, field) != getattr(source_after, field)
            for field in _STAT_GENERATION_FIELDS
        )
        or any(
            getattr(terminal_before, field)
            != getattr(terminal_after, field)
            for field in _STAT_GENERATION_FIELDS
        )
        or hashlib.sha256(source_raw_after).hexdigest()
        != C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256
        or hashlib.sha256(terminal_raw_after).hexdigest()
        != C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256
        or _source_root(
            C3_ENVIRONMENT_FAILURE_TERMINALIZER_SOURCE_PATH
        )
        != source_root
        or not isinstance(payload, Mapping)
        or not isinstance(terminal_root, Mapping)
    ):
        raise PermissionError("c3 environment-failure lineage changed")
    try:
        decoded = json.loads(terminal_raw_before)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(
            "c3 environment-failure terminal is malformed"
        ) from error
    identity = payload.get("identity")
    closure = payload.get("unit_realization_closure")
    failure = payload.get("environment_stability_failure")
    reproduction = payload.get("deterministic_reproduction")
    continuation = payload.get("continuation_policy")
    payload_observation = payload.get("payload_observation")
    returned_terminalizer_root = terminal_root.get(
        "terminalizer_source_root"
    )
    if (
        decoded != dict(payload)
        or terminal_root.get("path")
        != str(C3_ENVIRONMENT_FAILURE_TERMINAL_PATH.absolute())
        or terminal_root.get("file_sha256")
        != C3_ENVIRONMENT_FAILURE_TERMINAL_SHA256
        or terminal_root.get("mode") != 0o444
        or terminal_root.get("schema_version")
        != C3_ENVIRONMENT_FAILURE_TERMINAL_SCHEMA
        or terminal_root.get("terminal_fingerprint")
        != C3_ENVIRONMENT_FAILURE_TERMINAL_FINGERPRINT
        or not isinstance(returned_terminalizer_root, Mapping)
        or returned_terminalizer_root.get("path")
        != str(C3_ENVIRONMENT_FAILURE_TERMINALIZER_SOURCE_PATH.absolute())
        or returned_terminalizer_root.get("file_sha256")
        != C3_ENVIRONMENT_FAILURE_TERMINALIZER_SHA256
        or not isinstance(identity, Mapping)
        or identity.get("candidate") != CANDIDATE
        or identity.get("stage_id") != STAGE_ID
        or identity.get("scientific_attempt_id")
        != SCIENTIFIC_ATTEMPT_ID
        or identity.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or identity.get("runtime_compatibility_id") != "c3"
        or not isinstance(closure, Mapping)
        or closure.get("R3_receipt_passed") is not True
        or closure.get("static") is not True
        or closure.get("enabled") is not False
        or closure.get("started") is not False
        or closure.get("removed") is not False
        or closure.get("payload_authority") != "none"
        or closure.get("unit_name") != C3_UNIT_NAME
        or not isinstance(failure, Mapping)
        or failure.get("known_subcommand") != "stability-gate"
        or failure.get("attempt_count") != 1
        or failure.get("retry") is not False
        or failure.get("samples_collected") != 0
        or failure.get("expected_exception_type") != "PermissionError"
        or failure.get("expected_exception_message")
        != "precleanup inventory unit scope changed"
        or not isinstance(reproduction, Mapping)
        or reproduction.get("reproduced") is not True
        or reproduction.get("samples_collected") != 0
        or not isinstance(continuation, Mapping)
        or continuation.get("automatic_retry") is not False
        or continuation.get("same_c3_reentry") is not False
        or continuation.get("same_c3_reauthorization_allowed") is not False
        or continuation.get("same_c3_metadata_repair_allowed") is not False
        or continuation.get("c3_environment_gate_reentry_allowed")
        is not False
        or continuation.get("c3_environment_gate_repair_allowed")
        is not False
        or continuation.get("c4_required") is not True
        or continuation.get("new_explicit_authorization_required")
        is not True
        or continuation.get("scientific_attempt_id")
        != SCIENTIFIC_ATTEMPT_ID
        or continuation.get("scientific_attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or continuation.get("scientific_attempt_consumed") is not False
        or continuation.get("c3_authorization_consumed") is not True
        or continuation.get("unit_realization_consumed") is not True
        or continuation.get("environment_metadata_attempt_consumed")
        is not True
        or continuation.get("runtime_launch_consumed") is not False
        or continuation.get("materialization_consumed") is not False
        or not isinstance(payload_observation, Mapping)
        or any(
            payload_observation.get(field) is not False
            for field in _PAYLOAD_FLAGS
        )
        or payload_observation.get("gpu_accessed") is not False
        or payload_observation.get("training_started") is not False
        or payload_observation.get("samples_processed") != 0
        or payload_observation.get("optimizer_steps") != 0
        or payload_observation.get("parameter_updates") != 0
    ):
        raise PermissionError(
            "c3 environment-failure transition changed"
        )
    # This VALID/SEALED FAIL proves only the historical C3 -> C4 transition.
    # It is never promoted to direct C5 authority or converted into a
    # compatibility, environment, or scientific PASS.
    return dict(payload), dict(terminal_root), source_root


def _validate_c1_historical_absence_snapshot(
    module: ModuleType,
    value: object,
) -> None:
    if not isinstance(value, Mapping):
        raise PermissionError("c1 historical absences are malformed")

    paths = module.ABSENCE_PATHS
    row_keys = {
        "path",
        "basename",
        "lexists",
        "parent_path",
        "parent_device",
        "parent_inode",
        "parent_owner_uid",
        "parent_owner_gid",
        "parent_mode",
        "parent_nlink",
        "parent_size",
        "parent_mtime_ns",
        "parent_ctime_ns",
    }
    if not isinstance(paths, Mapping) or set(value) != set(paths):
        raise PermissionError("c1 historical absence keys changed")

    parent_fields = tuple(
        sorted(key for key in row_keys if key.startswith("parent_"))
    )
    integer_fields = set(parent_fields) - {"parent_path"}
    parents: dict[str, tuple[object, ...]] = {}
    for name, expected_path in paths.items():
        row = value[name]
        target = Path(expected_path).absolute()
        if (
            not isinstance(row, Mapping)
            or set(row) != row_keys
            or row.get("path") != str(target)
            or row.get("basename") != target.name
            or row.get("lexists") is not False
            or row.get("parent_path") != str(target.parent)
            or any(
                isinstance(row.get(field), bool)
                or not isinstance(row.get(field), int)
                or row[field] < 0
                for field in integer_fields
            )
            or row["parent_inode"] <= 0
            or row["parent_nlink"] < 2
            or row["parent_owner_uid"] != os.getuid()
            or row["parent_mode"] > 0o7777
            or row["parent_mode"] & 0o002
        ):
            raise PermissionError(f"c1 historical absence changed: {name}")

        parent_identity = tuple(row[field] for field in parent_fields)
        previous = parents.setdefault(str(target.parent), parent_identity)
        if previous != parent_identity:
            raise PermissionError("c1 historical absence parent diverged")


def _validate_c1_historical_terminal(
    module: ModuleType,
    terminal: Mapping[str, object],
    *,
    c1_reader: Callable[[], Mapping[str, str]],
    now: datetime,
) -> None:
    identity = {
        "schema_version": module.SCHEMA,
        "candidate": module.CANDIDATE,
        "stage_id": module.STAGE_ID,
        "scientific_attempt_id": module.SCIENTIFIC_ATTEMPT_ID,
        "scientific_attempt_ordinal": module.SCIENTIFIC_ATTEMPT_ORDINAL,
        "runtime_compatibility_id": module.RUNTIME_COMPATIBILITY_ID,
        "unit_name": module.UNIT_NAME,
        "instruction_id": module.INSTRUCTION_ID,
        "authorization_basis": module.AUTHORIZATION_BASIS,
    }
    if (
        set(terminal) != set(module._BODY_KEYS) | {"terminal_fingerprint"}
        or any(
            terminal.get(key) != expected
            for key, expected in identity.items()
        )
    ):
        raise PermissionError("c1 historical terminal identity changed")
    if (
        terminal.get("payload_observation") != module._PAYLOAD_OBSERVATION
        or terminal.get("continuation_policy")
        != module._CONTINUATION_POLICY
        or terminal.get("outcome") != module._OUTCOME
        or terminal.get("derived_runtime_absences")
        != module._derived_runtime_absences()
    ):
        raise PermissionError("c1 historical terminal contract changed")

    current = now.astimezone(timezone.utc)
    created = module._parse_utc(
        terminal.get("created_at_utc"),
        name="terminal created_at_utc",
    )
    if created > current:
        raise PermissionError("c1 historical terminal is future-dated")

    session = module._observe_session_failure()
    evidence_roots, payloads = module._observe_evidence()
    source_roots = module._observe_source_roots()
    fragment_root = module._observe_fragment_root()
    expiry = module._validate_evidence_semantics(
        payloads,
        now=current,
    )
    current_live = module._validate_live_state(
        c1_reader(),
        unit_receipt=payloads["unit_receipt"],
    )
    receipt_fragment = payloads["unit_receipt"].get("fragment_identity")
    fragment_fields = (
        "path",
        "file_sha256",
        "device",
        "inode",
        "owner_uid",
        "mode",
        "nlink",
    )
    if (
        not isinstance(receipt_fragment, Mapping)
        or terminal.get("session_failure") != session
        or terminal.get("evidence_roots") != evidence_roots
        or terminal.get("source_roots") != source_roots
        or terminal.get("fragment_root") != fragment_root
        or terminal.get("live_unit_state") != current_live
        or terminal.get("authorization_expiry") != expiry
        or any(
            receipt_fragment.get(key) != fragment_root.get(key)
            for key in fragment_fields
        )
    ):
        raise PermissionError("c1 historical immutable snapshot changed")

    _validate_c1_historical_absence_snapshot(
        module,
        terminal.get("absence_generation_roots"),
    )
    for name, path in module.ABSENCE_PATHS.items():
        if name not in {
            "scientific_run_root",
            "scientific_result_receipt",
        }:
            module._observe_absence(path)

    bridge_expiry = module._parse_utc(
        expiry["bridge_expires_at_utc"],
        name="terminal bridge expiry",
    )
    if created <= bridge_expiry:
        raise PermissionError("c1 historical terminal predates bridge expiry")


def _validate_c1_failure_terminal(
    *,
    unit_state_reader: UnitStateReader,
    now: datetime,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    module, source_root = _load_verified_terminalizer()

    def c1_reader() -> Mapping[str, str]:
        observed = dict(unit_state_reader(C1_UNIT_NAME))
        live_keys = set(module._LIVE_KEYS)
        if not live_keys.issubset(observed):
            raise PermissionError("c1 terminal live-state fields are absent")
        return {key: str(observed[key]) for key in live_keys}

    allowed_errors = {
        "expired-prewrite terminal live closure changed",
        "required absent path exists: "
        + str(Path(module.ABSENCE_PATHS["scientific_run_root"]).absolute()),
        "required absent path exists: "
        + str(
            Path(
                module.ABSENCE_PATHS["scientific_result_receipt"]
            ).absolute()
        ),
    }
    try:
        terminal = module.validate_terminal(
            terminal_path=C1_FAILURE_TERMINAL_PATH,
            unit_state_reader=c1_reader,
            now=lambda: now,
        )
    except PermissionError as error:
        if (
            type(error) is not PermissionError
            or len(error.args) != 1
            or error.args[0] not in allowed_errors
        ):
            raise
        terminal, root = module._load_sealed(
            C1_FAILURE_TERMINAL_PATH,
            fingerprint_field="terminal_fingerprint",
            schema=module.SCHEMA,
        )
        _validate_c1_historical_terminal(
            module,
            terminal,
            c1_reader=c1_reader,
            now=now,
        )
    else:
        sealed, root = module._load_sealed(
            C1_FAILURE_TERMINAL_PATH,
            fingerprint_field="terminal_fingerprint",
            schema=module.SCHEMA,
        )
        if terminal != sealed:
            raise PermissionError(
                "c1 terminalizer returned a different terminal"
            )
    continuation = terminal["continuation_policy"]
    outcome = terminal["outcome"]
    payload = terminal["payload_observation"]
    evidence = terminal["evidence_roots"]
    expiry = terminal["authorization_expiry"]
    if (
        continuation.get("same_c1_reauthorization_allowed") is not False
        or continuation.get("same_c1_receipt_sealing_allowed") is not False
        or continuation.get("automatic_retry_allowed") is not False
        or continuation.get("resume_allowed") is not False
        or continuation.get("new_compatibility_generation_required")
        is not True
        or outcome.get("scientific_attempt_consumed") is not False
        or outcome.get("runtime_launch_consumed") is not False
        or outcome.get("materialization_consumed") is not False
        or any(payload.get(field) is not False for field in _PAYLOAD_FLAGS)
        or payload.get("gpu_accessed") is not False
        or payload.get("training_started") is not False
        or not isinstance(evidence, Mapping)
        or "r10_authorization" not in evidence
        or "r10_receipt" not in evidence
        or not isinstance(expiry, Mapping)
    ):
        raise PermissionError("c1 terminal continuation semantics changed")
    return terminal, dict(root), source_root


def _r10_roots_from_terminal(
    terminal: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    evidence = terminal.get("evidence_roots")
    if not isinstance(evidence, Mapping):
        raise PermissionError("c1 terminal evidence roots are absent")
    roots = {
        "authorization": dict(evidence["r10_authorization"]),
        "receipt": dict(evidence["r10_receipt"]),
    }
    return roots


def _authorization_times(
    authorization: Mapping[str, object],
    *,
    current: datetime,
    require_fresh: bool,
) -> tuple[datetime, datetime, datetime]:
    created = _parse_utc(
        authorization.get("created_at_utc"),
        name="c5 authorization creation",
    )
    issued = _parse_utc(
        authorization.get("issued_at_utc"),
        name="c5 authorization issuance",
    )
    expires = _parse_utc(
        authorization.get("expires_at_utc"),
        name="c5 authorization expiry",
    )
    if (
        not issued <= created <= expires
        or expires - issued > timedelta(seconds=300)
        or (require_fresh and not issued <= current <= expires)
    ):
        raise PermissionError("c5 authorization is stale or malformed")
    return created, issued, expires


def _collect_protected_unit_states(
    unit_state_reader: UnitStateReader,
) -> dict[str, dict[str, object]]:
    old_state = _normalized_state(unit_state_reader, OLD_UNIT_NAME)
    c1_state = _normalized_state(unit_state_reader, C1_UNIT_NAME)
    c2_state = _normalized_state(unit_state_reader, C2_UNIT_NAME)
    c3_state = _normalized_state(unit_state_reader, C3_UNIT_NAME)
    c4_state = _normalized_state(unit_state_reader, C4_UNIT_NAME)
    _require_inert_state(
        old_state,
        unit_name=OLD_UNIT_NAME,
        fragment_path=str(OLD_UNIT_FRAGMENT_PATH),
    )
    _require_inert_state(
        c1_state,
        unit_name=C1_UNIT_NAME,
        fragment_path=str(C1_UNIT_FRAGMENT_PATH),
    )
    _require_missing_inert_state(c2_state, unit_name=C2_UNIT_NAME)
    _require_inert_state(
        c3_state,
        unit_name=C3_UNIT_NAME,
        fragment_path=str(C3_UNIT_FRAGMENT_PATH),
    )
    _require_inert_state(
        c4_state,
        unit_name=C4_UNIT_NAME,
        fragment_path=str(C4_UNIT_FRAGMENT_PATH),
    )
    return {
        "old": old_state,
        "c1": c1_state,
        "c2": c2_state,
        "c3": c3_state,
        "c4": c4_state,
    }


def _collect_preauthorization_target_unit_state(
    unit_state_reader: UnitStateReader,
) -> dict[str, object]:
    state = _normalized_state(unit_state_reader, C5_UNIT_NAME)
    _require_missing_inert_state(state, unit_name=C5_UNIT_NAME)
    return state


def authorize_c5(
    *,
    instruction_id: str,
    authorization_basis: str,
    validity_seconds: int = 300,
    authorization_path: Path | None = None,
    unit_state_reader: UnitStateReader = _default_unit_state_reader,
    now: Callable[[], datetime] = _utc_now,
) -> dict[str, object]:
    if authorization_path is None:
        authorization_path = C5_AUTHORIZATION_PATH
    _require_fixed_path(
        authorization_path,
        C5_AUTHORIZATION_PATH,
        name="c5 authorization",
    )
    if (
        instruction_id != INSTRUCTION_ID
        or authorization_basis != AUTHORIZATION_BASIS
        or isinstance(validity_seconds, bool)
        or not isinstance(validity_seconds, int)
        or not 1 <= validity_seconds <= 300
    ):
        raise ValueError("c5 authorization input changed")
    if any(
        os.path.lexists(path)
        for path in (C5_AUTHORIZATION_PATH, C5_RECEIPT_PATH, C5_TERMINAL_PATH)
    ):
        raise FileExistsError("c5 compatibility identity is consumed")
    _require_frozen_c4_failure_hashes()
    _require_frozen_c3_failure_hashes()
    _validate_scientific_output_phase(
        allow_runtime_activation=False,
        runtime_phase=RUNTIME_PHASE_PREACTIVATION,
    )
    _require_absent(_c5_preauthorization_paths())
    _require_absent(_c5_future_paths())
    issued = now().astimezone(timezone.utc)
    _prewrite, prewrite_root, prewrite_source_root = (
        _validate_c2_prewrite_failure_terminal()
    )
    _failure, failure_root, failure_source_root = (
        _validate_mode_contract_failure_terminal()
    )
    c3_failure, c3_failure_root, c3_failure_source_root = (
        _validate_c3_environment_failure_terminal()
    )
    _c4_authorization, c4_authorization_root, c4_bridge_root = (
        _validate_c4_authorization_archival(
            unit_state_reader=unit_state_reader,
            allow_runtime_activation=False,
            runtime_phase=RUNTIME_PHASE_PREACTIVATION,
            now=lambda: issued,
        )
    )
    c4_failure, c4_failure_root, c4_failure_source_root = (
        _validate_c4_receipt_seal_failure_terminal()
    )
    terminal, terminal_root, _terminalizer_root = (
        _validate_c1_failure_terminal(
            unit_state_reader=unit_state_reader,
            now=issued,
        )
    )
    terminal_evidence = terminal["evidence_roots"]
    c1_root = dict(terminal_evidence["bridge_authorization"])
    r10_roots = _r10_roots_from_terminal(terminal)
    sources = _collect_source_roots()
    if (
        sources["c2_mode_contract_failure_terminalizer"]
        != failure_source_root
        or sources["c2_prewrite_failure_terminalizer"]
        != prewrite_source_root
        or sources["c3_environment_failure_terminalizer"]
        != c3_failure_source_root
        or sources["c4_bridge"] != c4_bridge_root
        or sources["c4_receipt_seal_failure_terminalizer"]
        != c4_failure_source_root
    ):
        raise PermissionError(
            "c2 predecessor terminalizer root diverged"
        )
    _load_verified_environment_wrapper(
        sources["compat_environment_wrapper"]
    )
    _load_verified_unit_realizer(sources["compat_unit_realizer"])
    protected_states = _collect_protected_unit_states(unit_state_reader)
    c5_preauthorization_state = (
        _collect_preauthorization_target_unit_state(unit_state_reader)
    )
    c1_expires = _parse_utc(
        terminal["authorization_expiry"]["bridge_expires_at_utc"],
        name="c1 authorization expiry",
    )
    c2_failure_sealed = _parse_utc(
        _failure["identity"]["sealed_at_utc"],
        name="c2 mode-contract failure sealing",
    )
    c3_failure_sealed = _parse_utc(
        c3_failure["identity"]["sealed_at_utc"],
        name="c3 environment-failure sealing",
    )
    c4_failure_sealed = _parse_utc(
        c4_failure["identity"]["sealed_at_utc"],
        name="c4 receipt-seal failure sealing",
    )
    if (
        issued <= c1_expires
        or issued <= c2_failure_sealed
        or issued <= c3_failure_sealed
        or issued <= c4_failure_sealed
    ):
        raise PermissionError("c5 cannot precede its sealed predecessors")
    body: dict[str, object] = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "candidate": CANDIDATE,
        "stage_id": STAGE_ID,
        "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
        "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
        "runtime_compatibility_id": RUNTIME_COMPATIBILITY_ID,
        "instruction_id": instruction_id,
        "authorization_basis": authorization_basis,
        "authorized_uid": os.getuid(),
        "created_at_utc": _format_utc(issued),
        "issued_at_utc": _format_utc(issued),
        "expires_at_utc": _format_utc(
            issued + timedelta(seconds=validity_seconds)
        ),
        "c1_failure_terminal_root": terminal_root,
        "c2_mode_contract_failure_terminal_root": failure_root,
        "c2_prewrite_failure_terminal_root": prewrite_root,
        "c3_environment_failure_terminal_root": c3_failure_root,
        "c4_authorization_root": c4_authorization_root,
        "c4_receipt_seal_failure_terminal_root": c4_failure_root,
        "c1_expired_authorization_root": c1_root,
        "r10_roots": r10_roots,
        "compatibility_source_roots": sources,
        "protected_unit_states": protected_states,
        "preauthorization_target_unit_state": c5_preauthorization_state,
        "expected_evidence_paths": {
            label: str(path)
            for label, path in _evidence_paths().items()
            if label not in {
                "c1_failure_terminal",
                "r10_authorization",
                "r10_receipt",
            }
        },
        "scientific_output_contract": (
            _expected_scientific_output_contract()
        ),
        "scientific_authority": _expected_scientific_authority(),
        "mutation_authority": _expected_mutation_authority(),
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "gpu_accessed": False,
        "training_started": False,
        "materialization_consumed": False,
    }
    return _write_sealed(
        C5_AUTHORIZATION_PATH,
        body,
        fingerprint_field="authorization_fingerprint",
    )


def validate_c5_authorization(
    path: Path | None = None,
    *,
    unit_state_reader: UnitStateReader = _default_unit_state_reader,
    require_fresh: bool = True,
    allow_runtime_activation: bool = False,
    runtime_phase: str | None = None,
    now: Callable[[], datetime] = _utc_now,
) -> tuple[dict[str, object], dict[str, object]]:
    phase = _resolve_runtime_phase(
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=runtime_phase,
    )
    _require_frozen_c4_failure_hashes()
    _require_frozen_c3_failure_hashes()
    if path is None:
        path = C5_AUTHORIZATION_PATH
    fixed = _require_fixed_path(
        path,
        C5_AUTHORIZATION_PATH,
        name="c5 authorization",
    )
    authorization, root = _load_sealed(
        fixed,
        fingerprint_field="authorization_fingerprint",
        schema=AUTHORIZATION_SCHEMA,
    )
    _validate_common_identity(authorization)
    _require_no_payload(authorization)
    if set(authorization) != _AUTHORIZATION_KEYS:
        raise PermissionError("c5 authorization keys changed")
    _validate_scientific_output_phase(
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=phase,
    )
    current = now().astimezone(timezone.utc)
    _authorization_times(
        authorization,
        current=current,
        require_fresh=require_fresh,
    )
    _prewrite, prewrite_root, prewrite_source_root = (
        _validate_c2_prewrite_failure_terminal()
    )
    _failure, failure_root, failure_source_root = (
        _validate_mode_contract_failure_terminal()
    )
    c3_failure, c3_failure_root, c3_failure_source_root = (
        _validate_c3_environment_failure_terminal()
    )
    _c4_authorization, c4_authorization_root, c4_bridge_root = (
        _validate_c4_authorization_archival(
            unit_state_reader=unit_state_reader,
            allow_runtime_activation=allow_runtime_activation,
            runtime_phase=phase,
            now=lambda: current,
        )
    )
    c4_failure, c4_failure_root, c4_failure_source_root = (
        _validate_c4_receipt_seal_failure_terminal()
    )
    terminal, terminal_root, _terminalizer_root = (
        _validate_c1_failure_terminal(
            unit_state_reader=unit_state_reader,
            now=current,
        )
    )
    c1_root = dict(terminal["evidence_roots"]["bridge_authorization"])
    if authorization.get("r10_roots") != _r10_roots_from_terminal(
        terminal
    ):
        raise PermissionError("c5 authorization r10 lineage changed")
    sources = authorization.get("compatibility_source_roots")
    _validate_source_roots(sources)
    if not isinstance(sources, Mapping):
        raise PermissionError("c5 authorization source roots are malformed")
    protected_states = _collect_protected_unit_states(unit_state_reader)
    preauthorization_target_state = authorization.get(
        "preauthorization_target_unit_state"
    )
    if not isinstance(preauthorization_target_state, Mapping):
        raise PermissionError("c5 preauthorization target state is absent")
    _require_missing_inert_state(
        preauthorization_target_state,
        unit_name=C5_UNIT_NAME,
    )
    authorized_at = _parse_utc(
        authorization["created_at_utc"],
        name="c5 authorization creation",
    )
    c2_failure_sealed = _parse_utc(
        _failure["identity"]["sealed_at_utc"],
        name="c2 mode-contract failure sealing",
    )
    c3_failure_sealed = _parse_utc(
        c3_failure["identity"]["sealed_at_utc"],
        name="c3 environment-failure sealing",
    )
    c4_failure_sealed = _parse_utc(
        c4_failure["identity"]["sealed_at_utc"],
        name="c4 receipt-seal failure sealing",
    )
    if (
        authorized_at <= c2_failure_sealed
        or authorized_at <= c3_failure_sealed
        or authorized_at <= c4_failure_sealed
        or authorization.get("runtime_compatibility_id") != "c5"
        or authorization.get("instruction_id") != INSTRUCTION_ID
        or authorization.get("authorization_basis") != AUTHORIZATION_BASIS
        or authorization.get("authorized_uid") != os.getuid()
        or authorization.get("scientific_output_contract")
        != _expected_scientific_output_contract()
        or authorization.get("scientific_authority")
        != _expected_scientific_authority()
        or authorization.get("mutation_authority")
        != _expected_mutation_authority()
        or authorization.get("expected_evidence_paths")
        != {
            label: str(path)
            for label, path in _evidence_paths().items()
            if label
            not in {
                "c1_failure_terminal",
                "r10_authorization",
                "r10_receipt",
            }
        }
        or authorization.get("c1_failure_terminal_root") != terminal_root
        or authorization.get("c2_mode_contract_failure_terminal_root")
        != failure_root
        or authorization.get("c2_prewrite_failure_terminal_root")
        != prewrite_root
        or authorization.get("c3_environment_failure_terminal_root")
        != c3_failure_root
        or sources.get("c4_bridge") != c4_bridge_root
        or authorization.get("c4_authorization_root")
        != c4_authorization_root
        or sources.get("c4_receipt_seal_failure_terminalizer")
        != c4_failure_source_root
        or authorization.get("c4_receipt_seal_failure_terminal_root")
        != c4_failure_root
        or authorization.get("compatibility_source_roots", {}).get(
            "c2_prewrite_failure_terminalizer"
        )
        != prewrite_source_root
        or authorization.get("compatibility_source_roots", {}).get(
            "c2_mode_contract_failure_terminalizer"
        ) != failure_source_root
        or authorization.get("compatibility_source_roots", {}).get(
            "c3_environment_failure_terminalizer"
        ) != c3_failure_source_root
        or sources.get("c4_bridge") != c4_bridge_root
        or sources.get("c4_receipt_seal_failure_terminalizer")
        != c4_failure_source_root
        or authorization.get("c1_expired_authorization_root") != c1_root
        or authorization.get("protected_unit_states")
        != protected_states
        or authorization.get("scientific_authority", {}).get(
            "automatic_retry"
        )
        is not False
        or authorization.get("scientific_authority", {}).get("resume")
        is not False
        or authorization.get("scientific_authority", {}).get(
            "materialization_authorized"
        )
        is not False
    ):
        raise PermissionError("c5 authorization closure changed")
    return authorization, root


def validate_compat_authorization(
    path: Path | None = None,
    *,
    unit_state_reader: UnitStateReader = _default_unit_state_reader,
    require_fresh: bool = True,
    require_future_absence: bool = True,
    allow_runtime_activation: bool = False,
    runtime_phase: str | None = None,
    now: Callable[[], datetime] = _utc_now,
) -> tuple[dict[str, object], dict[str, object]]:
    """c1-compatible consumer interface for the c5 authorization.

    ``require_future_absence`` controls only the c5 runtime spec/launch/
    artifact/lease namespace.  Fresh realization is preactivation-only;
    archival consumers must preserve an explicit, internally consistent
    runtime phase while original scientific outputs and every alias stay
    subject to that phase's exact absence contract.
    """

    if not isinstance(require_future_absence, bool):
        raise TypeError("require_future_absence must be boolean")
    authorization, root = validate_c5_authorization(
        path=path,
        unit_state_reader=unit_state_reader,
        require_fresh=require_fresh,
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=runtime_phase,
        now=now,
    )
    _validate_scientific_output_phase(
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=_resolve_runtime_phase(
            allow_runtime_activation=allow_runtime_activation,
            runtime_phase=runtime_phase,
        ),
    )
    if require_future_absence:
        _require_absent(_c5_future_paths())
    return authorization, root


def _validate_unit_chain(
    *,
    realizer_root: Mapping[str, object],
    unit_state_reader: UnitStateReader,
    allow_runtime_activation: bool,
    runtime_phase: str,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    datetime,
]:
    producer, _producer_root = _load_verified_unit_realizer(realizer_root)
    archival = producer.validate_archival_realization_chain(
        C5_UNIT_AUTHORIZATION_PATH,
        C5_UNIT_RECEIPT_PATH,
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=runtime_phase,
    )
    if not isinstance(archival, Mapping):
        raise PermissionError("R5 producer returned malformed archival evidence")
    authorization = archival.get("authorization")
    auth_root = archival.get("authorization_identity")
    if auth_root is None:
        auth_root = archival.get("authorization_root")
    receipt = archival.get("receipt")
    receipt_root = archival.get("receipt_identity")
    if receipt_root is None:
        receipt_root = archival.get("receipt_root")
    if not all(
        isinstance(item, Mapping)
        for item in (authorization, auth_root, receipt, receipt_root)
    ):
        raise PermissionError("R5 producer omitted archival roots")
    authorization = dict(authorization)
    auth_root = dict(auth_root)
    receipt = dict(receipt)
    receipt_root = dict(receipt_root)
    _require_no_payload(authorization)
    _require_no_payload(receipt)
    issued = _parse_utc(
        authorization.get("issued_at_utc"),
        name="c5 realization issuance",
    )
    expires = _parse_utc(
        authorization.get("expires_at_utc"),
        name="c5 realization expiry",
    )
    receipt_time = _parse_utc(
        receipt.get("created_at_utc"),
        name="c5 realization receipt creation",
    )
    full_shadow = receipt.get("full_static_shadow")
    fragment = receipt.get("fragment_identity")
    if (
        authorization.get("unit_name") != C5_UNIT_NAME
        or receipt.get("unit_name") != C5_UNIT_NAME
        or receipt.get("passed") is not True
        or receipt.get("static") is not True
        or receipt.get("started") is not False
        or receipt.get("enabled") is not False
        or receipt.get("removed") is not False
        or not issued <= receipt_time <= expires
        or expires - issued > timedelta(seconds=300)
        or not isinstance(full_shadow, Mapping)
        or not isinstance(fragment, Mapping)
        or not set(_BRIDGE_STATE_FIELDS).issubset(full_shadow)
        or full_shadow.get("Id") != C5_UNIT_NAME
        or full_shadow.get("FragmentPath") != fragment.get("path")
        or receipt.get("manager_generation")
        != authorization.get("manager_generation")
    ):
        raise PermissionError("c5 unit realization chain changed")
    live = _normalized_state(unit_state_reader, C5_UNIT_NAME)
    if live.get("FragmentPath") != fragment.get("path"):
        raise PermissionError("c5 fragment path changed")
    fragment_root = _source_root(Path(str(fragment["path"])))
    if (
        fragment_root.get("file_sha256") != fragment.get("file_sha256")
        or fragment_root.get("device") != fragment.get("device")
        or fragment_root.get("inode") != fragment.get("inode")
    ):
        raise PermissionError("c5 fragment generation changed")
    # R5's fixed producer has already validated its complete systemd shadow.
    # This independent B5 reader intentionally observes the narrower bridge
    # state projection, so compare exactly that shared projection here.
    for field in _BRIDGE_STATE_FIELDS:
        if live.get(field) != full_shadow[field]:
            if (
                allow_runtime_activation
                and field in ("ActiveState", "SubState", "InvocationID")
            ):
                continue
            raise PermissionError(f"c5 live unit shadow changed: {field}")
    if not allow_runtime_activation:
        _require_inert_state(
            live,
            unit_name=C5_UNIT_NAME,
            fragment_path=str(fragment["path"]),
        )
    return authorization, auth_root, receipt, receipt_root, receipt_time


def _load_environment_evidence(
    environment: ModuleType,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, dict[str, object]],
]:
    _require_absent(
        {
            "c5_environment_stability_terminal": (
                C5_ENVIRONMENT_STABILITY_TERMINAL_PATH
            )
        }
    )
    loaded = environment.load_c5_environment_closure()
    if not isinstance(loaded, Mapping):
        raise PermissionError("E5 producer returned malformed closure")
    scope_handoff = loaded.get("scope_handoff")
    stability_attempt = loaded.get("stability_attempt")
    policy = loaded.get("policy")
    stability = loaded.get("stability")
    postcleanup = loaded.get("postcleanup")
    roots = loaded.get("evidence_roots")
    if not all(
        isinstance(item, Mapping)
        for item in (
            scope_handoff,
            stability_attempt,
            policy,
            stability,
            postcleanup,
            roots,
        )
    ):
        raise PermissionError("E5 producer omitted closure roots")
    scope_handoff = dict(scope_handoff)
    stability_attempt = dict(stability_attempt)
    policy = dict(policy)
    stability = dict(stability)
    postcleanup = dict(postcleanup)
    roots = dict(roots)
    expected_root_labels = {
        "environment_scope_handoff",
        "environment_stability_attempt",
        "environment_policy",
        "environment_stability",
        "environment_postcleanup",
    }
    if set(roots) != expected_root_labels or not all(
        isinstance(root, Mapping) for root in roots.values()
    ):
        raise PermissionError("E5 producer root labels changed")
    for value in (
        scope_handoff,
        stability_attempt,
        policy,
        stability,
        postcleanup,
    ):
        _require_no_payload(value)
    _require_absent(
        {
            "c5_environment_stability_terminal": (
                C5_ENVIRONMENT_STABILITY_TERMINAL_PATH
            )
        }
    )
    return (
        scope_handoff,
        stability_attempt,
        policy,
        stability,
        postcleanup,
        {label: dict(root) for label, root in roots.items()},
    )


def _collect_full_closure(
    *,
    authorization: Mapping[str, object],
    authorization_root: Mapping[str, object],
    unit_state_reader: UnitStateReader,
    allow_runtime_activation: bool,
    runtime_phase: str,
    receipt_time: datetime,
) -> dict[str, object]:
    _prewrite, prewrite_root, prewrite_source_root = (
        _validate_c2_prewrite_failure_terminal()
    )
    _failure, failure_root, failure_source_root = (
        _validate_mode_contract_failure_terminal()
    )
    c3_failure, c3_failure_root, c3_failure_source_root = (
        _validate_c3_environment_failure_terminal()
    )
    _c4_authorization, c4_authorization_root, c4_bridge_root = (
        _validate_c4_authorization_archival(
            unit_state_reader=unit_state_reader,
            allow_runtime_activation=allow_runtime_activation,
            runtime_phase=runtime_phase,
            now=lambda: receipt_time,
        )
    )
    c4_failure, c4_failure_root, c4_failure_source_root = (
        _validate_c4_receipt_seal_failure_terminal()
    )
    terminal, terminal_root, _terminalizer_root = (
        _validate_c1_failure_terminal(
            unit_state_reader=unit_state_reader,
            now=receipt_time,
        )
    )
    protected_states = _collect_protected_unit_states(unit_state_reader)
    if authorization.get("protected_unit_states") != protected_states:
        raise PermissionError("protected predecessor unit states changed")
    preauthorization_target_state = authorization.get(
        "preauthorization_target_unit_state"
    )
    if not isinstance(preauthorization_target_state, Mapping):
        raise PermissionError("c5 preauthorization target state is absent")
    _require_missing_inert_state(
        preauthorization_target_state,
        unit_name=C5_UNIT_NAME,
    )
    terminal_evidence = terminal.get("evidence_roots")
    if not isinstance(terminal_evidence, Mapping):
        raise PermissionError("c1 terminal evidence roots are absent")
    sources = authorization.get("compatibility_source_roots")
    _validate_source_roots(sources)
    if not isinstance(sources, Mapping):
        raise PermissionError("c5 source roots are malformed")
    if (
        sources.get("c2_mode_contract_failure_terminalizer")
        != failure_source_root
        or authorization.get("c2_mode_contract_failure_terminal_root")
        != failure_root
        or sources.get("c2_prewrite_failure_terminalizer")
        != prewrite_source_root
        or authorization.get("c2_prewrite_failure_terminal_root")
        != prewrite_root
        or sources.get("c3_environment_failure_terminalizer")
        != c3_failure_source_root
        or authorization.get("c3_environment_failure_terminal_root")
        != c3_failure_root
        or sources.get("c4_bridge") != c4_bridge_root
        or authorization.get("c4_authorization_root")
        != c4_authorization_root
        or sources.get("c4_receipt_seal_failure_terminalizer")
        != c4_failure_source_root
        or authorization.get("c4_receipt_seal_failure_terminal_root")
        != c4_failure_root
    ):
        raise PermissionError(
            "c2 predecessor failure lineage changed"
        )
    environment, wrapper_root = _load_verified_environment_wrapper(
        sources["compat_environment_wrapper"]
    )
    if wrapper_root != dict(sources["compat_environment_wrapper"]):
        raise PermissionError("c5 environment wrapper generation changed")
    (
        unit_authorization,
        unit_auth_root,
        unit_receipt,
        unit_receipt_root,
        unit_receipt_time,
    ) = _validate_unit_chain(
        realizer_root=sources["compat_unit_realizer"],
        unit_state_reader=unit_state_reader,
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=runtime_phase,
    )
    archival = {
        "authorization": unit_authorization,
        "authorization_identity": unit_auth_root,
        "receipt": unit_receipt,
        "receipt_identity": unit_receipt_root,
    }
    historical_contract, current_contract, _replay_roots = (
        environment.replay_old_scope_and_handoff()
    )
    (
        scope_handoff,
        stability_attempt,
        policy,
        stability,
        postcleanup,
        environment_roots,
    ) = (
        _load_environment_evidence(environment)
    )
    validated = environment.validate_c5_environment_closure(
        scope_handoff,
        stability_attempt,
        policy,
        stability,
        postcleanup,
        archival=archival,
        c5_contract=current_contract,
    )
    if (
        not isinstance(validated, Mapping)
        or validated.get("scope_handoff") != scope_handoff
        or validated.get("stability_attempt") != stability_attempt
        or validated.get("policy") != policy
        or validated.get("stability") != stability
        or validated.get("postcleanup") != postcleanup
    ):
        raise PermissionError("c5 environment wrapper returned a different closure")
    policy_time = _parse_utc(
        policy.get("created_at_utc"),
        name="c5 environment policy creation",
    )
    postcleanup_time = _parse_utc(
        postcleanup.get("created_at_utc"),
        name="c5 postcleanup creation",
    )
    c1_root = terminal_evidence.get("bridge_authorization")
    c3_failure_time = _parse_utc(
        c3_failure["identity"]["sealed_at_utc"],
        name="c3 environment-failure sealing",
    )
    c4_failure_time = _parse_utc(
        c4_failure["identity"]["sealed_at_utc"],
        name="c4 receipt-seal failure sealing",
    )
    if (
        authorization.get("c1_failure_terminal_root") != terminal_root
        or authorization.get("c1_expired_authorization_root") != c1_root
        or authorization.get("r10_roots")
        != _r10_roots_from_terminal(terminal)
        or authorization.get("c3_environment_failure_terminal_root")
        != c3_failure_root
        or authorization.get("c4_authorization_root")
        != c4_authorization_root
        or authorization.get("c4_receipt_seal_failure_terminal_root")
        != c4_failure_root
        or not c3_failure_time < c4_failure_time < unit_receipt_time
        or not unit_receipt_time < policy_time
        or policy_time > postcleanup_time
        or postcleanup_time > receipt_time
    ):
        raise PermissionError("c5 closure chronology/lineage changed")
    evidence_roots = {
        "c1_failure_terminal": terminal_root,
        "c2_mode_contract_failure_terminal": failure_root,
        "c2_prewrite_failure_terminal": prewrite_root,
        "c3_environment_failure_terminal": c3_failure_root,
        "c4_authorization": c4_authorization_root,
        "c4_receipt_seal_failure_terminal": c4_failure_root,
        "r10_authorization": authorization["r10_roots"]["authorization"],
        "r10_receipt": authorization["r10_roots"]["receipt"],
        **environment_roots,
        "unit_realization_authorization": unit_auth_root,
        "unit_realization_receipt": unit_receipt_root,
    }
    if set(evidence_roots) != _EVIDENCE_LABELS:
        raise AssertionError("c5 evidence labels changed")
    return {
        "authorization_root": dict(authorization_root),
        "source_roots": dict(sources),
        "evidence_roots": evidence_roots,
        "historical_contract": _normalize_environment_contract(
            historical_contract
        ),
        "current_contract": _normalize_environment_contract(
            current_contract
        ),
        "unit_authorization": unit_authorization,
        "unit_receipt": unit_receipt,
        "policy": policy,
        "scope_handoff": scope_handoff,
        "stability_attempt": stability_attempt,
        "stability": stability,
        "postcleanup": postcleanup,
    }


def seal_receipt(
    *,
    receipt_path: Path | None = None,
    unit_state_reader: UnitStateReader = _default_unit_state_reader,
    now: Callable[[], datetime] = _utc_now,
) -> dict[str, object]:
    if receipt_path is None:
        receipt_path = C5_RECEIPT_PATH
    _require_fixed_path(
        receipt_path,
        C5_RECEIPT_PATH,
        name="c5 receipt",
    )
    if os.path.lexists(C5_RECEIPT_PATH) or os.path.lexists(C5_TERMINAL_PATH):
        raise FileExistsError("c5 compatibility identity is consumed")
    _validate_scientific_output_phase(
        allow_runtime_activation=False,
        runtime_phase=RUNTIME_PHASE_PREACTIVATION,
    )
    _require_absent(_c5_future_paths())
    created = now().astimezone(timezone.utc)
    authorization, authorization_root = validate_c5_authorization(
        unit_state_reader=unit_state_reader,
        require_fresh=True,
        allow_runtime_activation=False,
        runtime_phase=RUNTIME_PHASE_PREACTIVATION,
        now=lambda: created,
    )
    _created, issued, expires = _authorization_times(
        authorization,
        current=created,
        require_fresh=True,
    )
    closure = _collect_full_closure(
        authorization=authorization,
        authorization_root=authorization_root,
        unit_state_reader=unit_state_reader,
        allow_runtime_activation=False,
        runtime_phase=RUNTIME_PHASE_PREACTIVATION,
        receipt_time=created,
    )
    if not issued <= created <= expires:
        raise PermissionError("c5 receipt is outside authorization window")
    body: dict[str, object] = {
        "schema_version": RECEIPT_SCHEMA,
        "candidate": CANDIDATE,
        "stage_id": STAGE_ID,
        "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
        "scientific_attempt_ordinal": SCIENTIFIC_ATTEMPT_ORDINAL,
        "runtime_compatibility_id": RUNTIME_COMPATIBILITY_ID,
        "created_at_utc": _format_utc(created),
        "compatibility_authorization_root": authorization_root,
        "compatibility_source_roots": closure["source_roots"],
        "compatibility_evidence_roots": closure["evidence_roots"],
        "historical_environment_contract": (
            closure["historical_contract"]
        ),
        "current_environment_contract": closure["current_contract"],
        "scientific_output_contract": (
            authorization["scientific_output_contract"]
        ),
        "scientific_authority": authorization["scientific_authority"],
        "schema_compatibility": _expected_schema_compatibility(),
        "compatibility_closure_passed": True,
        "runtime_launch_authorized": False,
        "systemd_start_authorized": False,
        "automatic_retry": False,
        "resume": False,
        "D_R_payload_accessed": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "gpu_accessed": False,
        "training_started": False,
        "materialization_consumed": False,
    }
    return _write_sealed(
        C5_RECEIPT_PATH,
        body,
        fingerprint_field="receipt_fingerprint",
    )


def _validate_receipt_evidence_roots(
    roots: object,
    expected: Mapping[str, object],
) -> None:
    if (
        not isinstance(roots, Mapping)
        or set(roots) != _EVIDENCE_LABELS
        or dict(roots) != dict(expected)
    ):
        raise PermissionError("c5 receipt evidence-root labels changed")


def _validate_expected_runtime_spec_contract(
    expected_spec: Mapping[str, object],
) -> None:
    systemd = expected_spec.get("systemd")
    artifacts = expected_spec.get("artifacts")
    if (
        not isinstance(systemd, Mapping)
        or not isinstance(artifacts, Mapping)
        or expected_spec.get("attempt_ordinal")
        != SCIENTIFIC_ATTEMPT_ORDINAL
        or systemd.get("unit_name") != C5_UNIT_NAME
        or artifacts.get("root") != str(C5_RUNTIME_ARTIFACT_ROOT)
    ):
        raise PermissionError("expected spec is outside c5")


def verify_compatibility_receipt(
    path: Path | None = None,
    expected_spec: Mapping[str, object] | None = None,
    require_spec_binding: bool = False,
    allow_runtime_activation: bool = False,
    *,
    runtime_phase: str | None = None,
    unit_state_reader: UnitStateReader = _default_unit_state_reader,
    now: Callable[[], datetime] = _utc_now,
) -> dict[str, object]:
    phase = _resolve_runtime_phase(
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=runtime_phase,
    )
    if path is None:
        path = C5_RECEIPT_PATH
    fixed = _require_fixed_path(
        path,
        C5_RECEIPT_PATH,
        name="c5 receipt",
    )
    if type(require_spec_binding) is not bool:
        raise TypeError("c5 verification phase flags must be boolean")
    if (
        (expected_spec is None and require_spec_binding)
        or (expected_spec is not None and not require_spec_binding)
    ):
        raise PermissionError("c5 runtime-spec verification phase changed")
    receipt, receipt_root = _load_sealed(
        fixed,
        fingerprint_field="receipt_fingerprint",
        schema=RECEIPT_SCHEMA,
    )
    _validate_common_identity(receipt)
    _require_no_payload(receipt)
    if set(receipt) != _RECEIPT_KEYS:
        raise PermissionError("c5 receipt keys changed")
    if (
        receipt.get("runtime_compatibility_id") != "c5"
        or receipt.get("compatibility_closure_passed") is not True
        or receipt.get("runtime_launch_authorized") is not False
        or receipt.get("systemd_start_authorized") is not False
        or receipt.get("automatic_retry") is not False
        or receipt.get("resume") is not False
        or receipt.get("scientific_output_contract")
        != _expected_scientific_output_contract()
        or receipt.get("scientific_authority")
        != _expected_scientific_authority()
        or receipt.get("schema_compatibility")
        != _expected_schema_compatibility()
    ):
        raise PermissionError("c5 receipt semantics changed")
    authorization, authorization_root = validate_c5_authorization(
        unit_state_reader=unit_state_reader,
        require_fresh=False,
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=phase,
        now=now,
    )
    receipt_time = _parse_utc(
        receipt.get("created_at_utc"),
        name="c5 receipt creation",
    )
    _created, issued, expires = _authorization_times(
        authorization,
        current=receipt_time,
        require_fresh=False,
    )
    if not issued <= receipt_time <= expires:
        raise PermissionError("archival c5 receipt chronology changed")
    closure = _collect_full_closure(
        authorization=authorization,
        authorization_root=authorization_root,
        unit_state_reader=unit_state_reader,
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=phase,
        receipt_time=receipt_time,
    )
    _validate_receipt_evidence_roots(
        receipt.get("compatibility_evidence_roots"),
        closure["evidence_roots"],
    )
    _validate_source_roots(
        receipt.get("compatibility_source_roots")
    )
    if (
        receipt.get("compatibility_authorization_root")
        != authorization_root
        or receipt.get("compatibility_source_roots")
        != closure["source_roots"]
        or receipt.get("compatibility_evidence_roots")
        != closure["evidence_roots"]
        or receipt.get("historical_environment_contract")
        != closure["historical_contract"]
        or receipt.get("current_environment_contract")
        != closure["current_contract"]
        or receipt.get("scientific_output_contract")
        != authorization["scientific_output_contract"]
    ):
        raise PermissionError("c5 compatibility receipt lineage changed")
    _validate_scientific_output_phase(
        allow_runtime_activation=allow_runtime_activation,
        runtime_phase=phase,
    )
    if expected_spec is None:
        _require_absent(_c5_future_paths())
    else:
        if not isinstance(expected_spec, Mapping):
            raise TypeError("expected c5 runtime spec must be a mapping")
        sealed_spec, _root = _load_sealed(
            C5_RUNTIME_SPEC_PATH,
            fingerprint_field="runtime_spec_fingerprint",
            schema=RUNTIME_SPEC_SCHEMA,
        )
        if sealed_spec != dict(expected_spec):
            raise PermissionError("expected c5 runtime spec changed")
        _validate_expected_runtime_spec_contract(sealed_spec)
    result = dict(receipt)
    result["receipt_root"] = receipt_root
    return result


def verify_compatibility_prewrite_spec(
    path: Path,
    expected_spec: Mapping[str, object],
    *,
    unit_state_reader: UnitStateReader = _default_unit_state_reader,
    now: Callable[[], datetime] = _utc_now,
) -> dict[str, object]:
    """Validate one exact producer preview while every runtime path is absent."""

    if not isinstance(expected_spec, Mapping):
        raise TypeError("expected_spec must be a mapping")
    receipt = verify_compatibility_receipt(
        path=path,
        expected_spec=None,
        require_spec_binding=False,
        allow_runtime_activation=False,
        runtime_phase=RUNTIME_PHASE_PREACTIVATION,
        unit_state_reader=unit_state_reader,
        now=now,
    )
    _validate_expected_runtime_spec_contract(expected_spec)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CURE-Lite v24 c5 preaccess compatibility closure",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    authorize = subparsers.add_parser("authorize-c5")
    authorize.add_argument("--instruction-id", required=True)
    authorize.add_argument("--authorization-basis", required=True)
    authorize.add_argument(
        "--validity-seconds",
        type=int,
        default=300,
    )
    subparsers.add_parser("seal-receipt")
    verify = subparsers.add_parser("verify-compatibility-receipt")
    verify.add_argument("--expected-spec", type=Path)
    verify.add_argument("--allow-runtime-activation", action="store_true")
    verify.add_argument("--runtime-phase", choices=sorted(RUNTIME_PHASES))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "authorize-c5":
        result = authorize_c5(
            instruction_id=args.instruction_id,
            authorization_basis=args.authorization_basis,
            validity_seconds=args.validity_seconds,
        )
        summary = {
            "authorization_fingerprint": result[
                "authorization_fingerprint"
            ],
            "expires_at_utc": result["expires_at_utc"],
        }
    elif args.command == "seal-receipt":
        result = seal_receipt()
        summary = {
            "receipt_fingerprint": result["receipt_fingerprint"],
            "compatibility_closure_passed": result[
                "compatibility_closure_passed"
            ],
        }
    else:
        expected = None
        if args.expected_spec is not None:
            expected, _root = _load_sealed(
                _require_fixed_path(
                    args.expected_spec,
                    C5_RUNTIME_SPEC_PATH,
                    name="c5 runtime spec",
                ),
                fingerprint_field="runtime_spec_fingerprint",
                schema=RUNTIME_SPEC_SCHEMA,
            )
        result = verify_compatibility_receipt(
            expected_spec=expected,
            require_spec_binding=expected is not None,
            allow_runtime_activation=args.allow_runtime_activation,
            runtime_phase=args.runtime_phase,
        )
        summary = {
            "receipt_fingerprint": result["receipt_fingerprint"],
            "compatibility_closure_passed": result[
                "compatibility_closure_passed"
            ],
        }
    print(_canonical_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
