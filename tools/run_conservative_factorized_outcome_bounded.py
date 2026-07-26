#!/usr/bin/env python3
"""Run the one authorized D_R-only CC-SEA v8 bounded model-code gate.

This entrypoint is deliberately independent of the v7 bounded runner and
executor.  It binds the authoritative CC-SEA dry-run v3 chain, the frozen
bounded implementation proposal/configuration, a future signed implementation
closure, and a *separate* one-run authorization.  Every closure and
authorization check completes before the output is claimed and before
``_load_frozen_real_inputs`` can be called.

The output is create-only.  A claimed reconstruction or execution failure
consumes the one authorized run and is published as a strictly reloadable
failure artifact.  Resume, retry, D_V/D_T access, calibration, detector
performance evaluation, formal-800 training, Full CURE, and detector
integration are outside this entrypoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from math import isfinite
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.config import LossConfig  # noqa: E402
from cure_lite.conservative_factorized_config import (  # noqa: E402
    ConservativeFactorizedDecoderConfig,
)
from tools import run_factorized_outcome_bounded as v4_runner  # noqa: E402
from tools import run_paired_outcome_bounded as v3_runner  # noqa: E402


CONSERVATIVE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA = (
    "cure-lite-cc-sea-v8-outcome-bounded-v1"
)
CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID = "cc_sea_v8"

_PROTOCOL_PREFIX = (
    "protocols/IRSTD-1K/"
    "coverage_conserving_subpixel_evidence_allocation_v8/"
)
METHOD_PROPOSAL_REPO_PATH = _PROTOCOL_PREFIX + "proposal_receipt.json"
METHOD_PROPOSAL_FILE_SHA256 = (
    "4590a681990a5332de233262510e0918f1d08d7b01ea6ad5e3c4ed7b8749c9bc"
)
METHOD_PROPOSAL_FINGERPRINT = (
    "14bb96e03598a613c5c201e891d7c5b690f8cc38dbf83d380aa3dbc17e82370b"
)
TOY_CLOSURE_REPO_PATH = _PROTOCOL_PREFIX + "toy_gate_closure_receipt.json"
TOY_CLOSURE_FILE_SHA256 = (
    "63affcf21c59f0808b2fcc18e1fc6e1054fc781708fa521d202d2a9ac8b16b0d"
)
TOY_CLOSURE_FINGERPRINT = (
    "be05a38ca53975f48f429e16d0df31b365a76bfb994c8a911e8ed636af4a2f67"
)
DRY_PROPOSAL_REPO_PATH = (
    _PROTOCOL_PREFIX + "bounded_dry_run_proposal_receipt_v3.json"
)
DRY_PROPOSAL_FILE_SHA256 = (
    "1a1fc75c23991373d584f91041f3af73319c1e5e539dc728bd9d8f4cc41b9949"
)
DRY_PROPOSAL_FINGERPRINT = (
    "509584774a52dbaa585f8d0860c16baf630046e94c1305bb2e1ca384cf45d746"
)
DRY_CONFIG_REPO_PATH = _PROTOCOL_PREFIX + "bounded_dry_run_config_v3.json"
DRY_CONFIG_FILE_SHA256 = (
    "5187b2f5516fd33b3eba9ae74092ba10ce42d0a85ac9d22918cbefb322e835c6"
)
DRY_CONFIG_FINGERPRINT = (
    "c985d1598d490c202397b0483cc2ac02abe98ed1ea9de26127a935386ac5b863"
)
DRY_RESULT_REPO_PATH = _PROTOCOL_PREFIX + "bounded_dry_run_result_v3.json"
DRY_RESULT_FILE_SHA256 = (
    "7fb88a27bb37ae7e28713d2de427ad9f42bf82dd07fcfa9f60b0eee04745140e"
)
DRY_RESULT_FINGERPRINT = (
    "18e7d5511b7f37d5e6060fb02e30dc34659516e905ce6736772322c4e865a586"
)
DRY_CLOSURE_REPO_PATH = (
    _PROTOCOL_PREFIX + "bounded_dry_run_closure_receipt.json"
)
DRY_CLOSURE_FILE_SHA256 = (
    "811e5582b9dc99b860fc866faa350b538ad06094a997195da6152cd6013fc935"
)
DRY_CLOSURE_FINGERPRINT = (
    "020485ba9e1feb37a64b1e17272113d23596842cfb76eefb94b9e3f2b3c036c6"
)

IMPLEMENTATION_PROPOSAL_REPO_PATH = (
    _PROTOCOL_PREFIX + "bounded_implementation_proposal_receipt.json"
)
IMPLEMENTATION_PROPOSAL_FILE_SHA256 = (
    "c9e06e4ae488b2b3b5e93e2c794cc4bbb55ad0bd5d2558ed6ff5b09a0787054d"
)
IMPLEMENTATION_PROPOSAL_FINGERPRINT = (
    "8f3008e707d03f23be9760a200a9af219fc0eb9f3f284a3635d066370f3da754"
)
CONFIG_REPO_PATH = _PROTOCOL_PREFIX + "bounded_config.json"
CONFIG_FILE_SHA256 = (
    "19ebde5b42643e65177084cb52d456e065e7ee9349852e1c68f4f6778a6c9b47"
)
CONFIG_FINGERPRINT = (
    "baf120fdd7886877e70df3c2186035ab78df9c417aebe549250a142652b417ba"
)
IMPLEMENTATION_CLOSURE_REPO_PATH = (
    _PROTOCOL_PREFIX + "bounded_implementation_closure_receipt.json"
)
AUTHORIZATION_REPO_PATH = (
    _PROTOCOL_PREFIX + "bounded_run_authorization_receipt.json"
)

V4_IMPLEMENTATION_RECEIPT_FINGERPRINT = (
    "1e01ea64f64f27a59dec84cf071eaefdc6c6bfbceec360cfc1bc66b9365cf975"
)
PAIR_CATALOG_FINGERPRINT = (
    "4886e52d2cfb3392d0f4fdda376159d6e7f694fd449dc809cf8874793febde76"
)
PREPARED_CATALOG_FINGERPRINT = (
    "4955e5b4f1749b5f267db0ac1f031335a16cc48a470d6446ca6c99d04a5e85ed"
)
ANCHOR_POPULATION_FINGERPRINT = (
    "d251ed9061dd373aa0bf0e4ceeebbafc7ca32a4bab72c2f24601a20868d6d1cd"
)
MATERIALIZER_FINGERPRINT = (
    "8cc4eac43ad708265d8639c4b577b37bd81be8ccde73e79993ba18c65dca10ff"
)
FACTUAL_SCHEDULE_FINGERPRINT = (
    "57264042879d9850aa538e01563496a8d3de7b82556d2b5ef15ca7f32b66fac3"
)
OUTCOME_SCHEDULE_FINGERPRINT = (
    "747123867c88fd1444a514bf70e51013b739f39df2857e5ed021239e4847ec93"
)
ALL_PAIR_INPUTS_FINGERPRINT = (
    "f3573b469464015865870440427deed341b7e2cddd8e866bdede2ee44c509b6c"
)
GT_UNION_POPULATION_FINGERPRINT = (
    "afa80e88581fa5ee5f832dc70624d9ee54ca9541d1b33b7f5957ef0ed08e3ae5"
)
OUTCOME_SEQUENCE_FINGERPRINT = (
    "6f4c45d51cfa8364d97a620af1bad1ea565f9ce4fc72c4d638d141fb056cffd0"
)

OUTPUT_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/cure_lite_cc_sea_v8_bounded_r1"
)
OUTPUT_VERSION_PREFIX = "cure_lite_cc_sea_v8_bounded_"
FROZEN_DEVICE = "cuda:0"
TEMPERATURE_WRAPPER_REPO_PATH = "tools/run_with_gpu_temperature_control.py"
TEMPERATURE_WRAPPER_FILE_SHA256 = (
    "026b751fbb59530721da1436af32f3bc924c9ed2ab3576df062a45bca7ec5e86"
)
SYNC_BENCHMARK_REPO_PATH = (
    _PROTOCOL_PREFIX + "sync_benchmark_result.json"
)
SYNC_BENCHMARK_FILE_SHA256 = (
    "52caa08511aebf26e5e7e746cd1d59017e14d5e8ea86a841f36623901a36f152"
)
SYNC_BENCHMARK_FINGERPRINT = (
    "d2304d3428daadebc40fc9047c9aca20b4c9492d95ccc6c554ae5d541471a0c0"
)
SYNC_BENCHMARK_TOOL_REPO_PATH = (
    "tools/benchmark_conservative_factorized_sync.py"
)
SYNC_BENCHMARK_TOOL_FILE_SHA256 = (
    "ab41a52ee5db19069c99b1ff306f1583326e96c928136eb43df8d95b5d1f40e7"
)
PYTHON_EXECUTABLE = "/home/md0/ly/MSHNet/.venv/bin/python"

RUN_SCHEMA = "cure-lite-cc-sea-v8-bounded-run-v1"
DECISION_SCHEMA = "cure-lite-cc-sea-v8-bounded-decision-v1"
FAILURE_SCHEMA = "cure-lite-cc-sea-v8-bounded-failure-v1"
IMPLEMENTATION_SCHEMA = "cure-lite-cc-sea-v8-runtime-implementation-v1"
IMPLEMENTATION_CLOSURE_SCHEMA = (
    "cure-lite-cc-sea-v8-bounded-implementation-closure-v1"
)
AUTHORIZATION_SCHEMA = (
    "cure-lite-cc-sea-v8-bounded-run-authorization-v1"
)

_ROOT = Path(__file__).resolve().parents[1]
_INCOMPLETE = ".incomplete"
_PRE_RUN_RECEIPTS = {
    "authorization_binding.json",
    "config_binding.json",
    "dry_run_closure_binding.json",
    "dry_run_config_binding.json",
    "dry_run_proposal_binding.json",
    "dry_run_result_binding.json",
    "implementation_binding.json",
    "implementation_closure_binding.json",
    "implementation_proposal_binding.json",
    "method_proposal_binding.json",
    "run_claim.json",
    "toy_gate_binding.json",
}
_INPUT_RECEIPTS = {
    "anchor_population.json",
    "factual_schedule.json",
    "outcome_inputs.json",
    "outcome_schedule.json",
    "source_reconstruction.json",
}
_CLOSURE_STATIC_TEST_REPO_PATH = (
    "tests_v8/"
    "test_conservative_factorized_bounded_implementation_closure.py"
)
_REAL_RUNNER_TEST_REPO_PATH = (
    "tests_v8/test_run_conservative_factorized_outcome_bounded.py"
)
_CORE_EXECUTOR_TEST_REPO_PATH = (
    "tests_v8/test_conservative_factorized_outcome_bounded.py"
)
_PROTOCOL_TEST_REPO_PATH = (
    "tests_v8/test_conservative_factorized_bounded_protocol.py"
)
_SYNC_TEST_REPO_PATH = (
    "tests_v8/test_conservative_factorized_sync.py"
)
_V7_CLOSURE_TEST_REPO_PATH = (
    "tests/test_crossing_factorized_bounded_implementation_closure.py"
)

_BOUND_GATE_CONTRACT = {
    "all_222_pairs_bound": True,
    "clean_joint_D_ge_0_25_and_H_le_0_05_fraction_min": 0.75,
    "clean_mean_delta_on_D_min": 0.5,
    "clean_pairs_delta_at_least_0_25_fraction_min": 0.75,
    "clean_transition_final_over_initial_max": 0.5,
    "clean_zero_macro_mean_abs_delta_max": 0.05,
    "component_null_context_macro_mean_abs_delta_max": 0.05,
    "component_null_footprint_global_max_abs_delta_max": 0.25,
    "component_null_footprint_macro_mean_abs_delta_max": 0.05,
    "factual_miss_anchor_final_over_initial_max": 0.75,
    "factual_no_miss_anchor_final_over_initial_max": 0.75,
    "identity_null_max_abs_delta_max": 1.0e-7,
    "pair_exposure_counts": [3, 4],
    "plus_baseline_final_over_initial_max": 0.75,
    "tiny_target_strata_report_required": True,
}
_BUDGET_CONTRACT = {
    "backward_calls_per_update": 1,
    "decoder_forward_calls_per_update": 3,
    "decoder_states_per_update": 12,
    "epochs": 10,
    "evaluation_chunk_size": 32,
    "factual_miss_states_per_update": 4,
    "factual_no_miss_states_per_update": 4,
    "optimizer_steps_per_update": 1,
    "optimizer_updates": 400,
    "outcome_endpoint_states_per_update": 4,
    "outcome_pairs_per_update": 2,
    "pair_slots": 800,
    "resume_allowed": False,
    "steps_per_epoch": 40,
}
_OUTCOME_POPULATION_CONTRACT = {
    "clean_positive": 206,
    "component_null": 16,
    "identity_null_optimizer_exposure": 0,
    "sampling": "pair_level_uniform_deterministic_over_outcome_union",
    "source_disjoint_within_update": True,
    "union": 222,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _canonical_file(path: Path, *, name: str) -> Path:
    candidate = path.expanduser()
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


def _repo_file(path_text: object, *, name: str) -> Path:
    if (
        not isinstance(path_text, str)
        or not path_text
        or Path(path_text).is_absolute()
        or ".." in Path(path_text).parts
    ):
        raise ValueError(f"{name} must be a non-empty repo-relative path")
    path = _canonical_file(_ROOT / path_text, name=name)
    if path.relative_to(_ROOT).as_posix() != path_text:
        raise RuntimeError(f"{name} does not resolve to its frozen path")
    return path


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must be a JSON object")
    return payload


def _fingerprinted(
    payload: Mapping[str, object],
    *,
    field: str = "receipt_fingerprint",
) -> dict[str, object]:
    output = dict(payload)
    if field in output:
        raise ValueError(f"payload already contains {field}")
    output[field] = stable_fingerprint(output)
    return output


def _verify_fingerprinted(
    payload: Mapping[str, Any],
    *,
    name: str,
    field: str = "receipt_fingerprint",
) -> None:
    observed = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if (
        not isinstance(observed, str)
        or stable_fingerprint(unsigned) != observed
    ):
        raise RuntimeError(f"{name} fingerprint is inconsistent")


def _verified_unsigned_receipt(
    payload: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    _verify_fingerprinted(payload, name=name)
    unsigned = dict(payload)
    unsigned.pop("receipt_fingerprint")
    return unsigned


def _write_new_json(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)


def _load_exact_signed(
    *,
    path_text: str,
    expected_sha256: str,
    expected_fingerprint: str,
    name: str,
    fingerprint_field: str = "receipt_fingerprint",
) -> tuple[dict[str, Any], Path]:
    path = _repo_file(path_text, name=name)
    if file_sha256(path) != expected_sha256:
        raise RuntimeError(f"{name} file SHA256 changed")
    payload = _strict_json(path, name=name)
    _verify_fingerprinted(
        payload,
        name=name,
        field=fingerprint_field,
    )
    if payload.get(fingerprint_field) != expected_fingerprint:
        raise RuntimeError(f"{name} fingerprint changed")
    return payload, path


def _binding_matches(
    binding: object,
    *,
    repo_path: str,
    file_sha256_value: str,
    fingerprint: str,
) -> bool:
    if not isinstance(binding, Mapping):
        return False
    bound_fingerprint = (
        binding.get("fingerprint")
        or binding.get("receipt_fingerprint")
        or binding.get("proposal_fingerprint")
        or binding.get("config_fingerprint")
        or binding.get("result_fingerprint")
    )
    return (
        binding.get("repo_path") == repo_path
        and binding.get("file_sha256") == file_sha256_value
        and bound_fingerprint == fingerprint
    )


def _validate_device(device: object) -> str:
    if device != FROZEN_DEVICE:
        raise ValueError("CC-SEA v8 bounded execution fixes --device at cuda:0")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError(
            "CC-SEA v8 must be launched by the frozen GPU-0 temperature "
            "wrapper with CUDA_VISIBLE_DEVICES=0"
        )
    return FROZEN_DEVICE


def _frozen_output_path() -> Path:
    return Path(os.path.abspath(_ROOT / OUTPUT_REPO_PATH))


def _validate_output_target(path: Path) -> Path:
    candidate = path.expanduser()
    absolute = Path(os.path.abspath(candidate))
    expected = _frozen_output_path()
    if absolute != expected:
        raise ValueError(
            "CC-SEA v8 permits only its frozen r1 output path: "
            f"{expected}"
        )
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"CC-SEA v8 output already exists: {absolute}")
    run_root = expected.parent
    if run_root.exists():
        prior = tuple(
            sorted(
                item
                for item in run_root.iterdir()
                if item.name.startswith(OUTPUT_VERSION_PREFIX)
            )
        )
        if prior:
            raise FileExistsError(
                "a CC-SEA v8 bounded run already exists: "
                + ", ".join(str(item) for item in prior)
            )
    for parent in (absolute.parent, *absolute.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError("CC-SEA v8 output may not traverse a symlink")
    return absolute


def _validate_config_payload(config: Mapping[str, Any]) -> None:
    unsigned = dict(config)
    fingerprint = unsigned.pop("config_fingerprint", None)
    if (
        fingerprint != CONFIG_FINGERPRINT
        or stable_fingerprint(unsigned) != fingerprint
        or config.get("schema_version")
        != "cure-lite-cc-sea-v8-bounded-config-v1"
        or config.get("method_id")
        != CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID
        or config.get("dataset") != "IRSTD-1K"
        or config.get("split") != "D_R"
    ):
        raise RuntimeError("CC-SEA v8 bounded config identity changed")

    if (
        config.get("bounded_gates") != _BOUND_GATE_CONTRACT
        or config.get("budget") != _BUDGET_CONTRACT
        or config.get("outcome_population")
        != _OUTCOME_POPULATION_CONTRACT
    ):
        raise RuntimeError("CC-SEA v8 bounded comparison contract changed")

    implementation = config.get("bounded_implementation_proposal_binding")
    dry = config.get("dry_run_closure_binding")
    optimization = config.get("optimization")
    source = config.get("source_reconstruction")
    policy = config.get("execution_policy")
    closure = config.get("implementation_closure_contract")
    authorization = config.get("future_pre_run_authorization_contract")
    semantics = config.get("decision_semantics")
    if not all(
        isinstance(value, Mapping)
        for value in (
            implementation,
            dry,
            optimization,
            source,
            policy,
            closure,
            authorization,
            semantics,
        )
    ):
        raise RuntimeError("CC-SEA v8 bounded config sections are malformed")

    expected_decoder = ConservativeFactorizedDecoderConfig(
        feature_channels=64,
        feature_stride=4,
    )
    if (
        dict(optimization.get("decoder", {})) != vars(expected_decoder)
        or optimization.get("loss")
        != {"dice_weight": 1.0, "epsilon": 1.0e-6}
        or optimization.get("optimizer") != "adam"
        or optimization.get("learning_rate") != 1.0e-3
        or optimization.get("weight_decay") != 0.0
        or optimization.get("seed") != 42
        or optimization.get("trainable_scope")
        != "CURELiteConservativeFactorizedDecoder_only"
    ):
        raise RuntimeError("CC-SEA v8 optimization contract changed")

    if (
        not _binding_matches(
            implementation,
            repo_path=IMPLEMENTATION_PROPOSAL_REPO_PATH,
            file_sha256_value=IMPLEMENTATION_PROPOSAL_FILE_SHA256,
            fingerprint=IMPLEMENTATION_PROPOSAL_FINGERPRINT,
        )
        or not _binding_matches(
            dry,
            repo_path=DRY_CLOSURE_REPO_PATH,
            file_sha256_value=DRY_CLOSURE_FILE_SHA256,
            fingerprint=DRY_CLOSURE_FINGERPRINT,
        )
        or source.get("required_pair_catalog_fingerprint")
        != PAIR_CATALOG_FINGERPRINT
        or source.get("required_prepared_catalog_fingerprint")
        != PREPARED_CATALOG_FINGERPRINT
        or source.get("required_anchor_population_fingerprint")
        != ANCHOR_POPULATION_FINGERPRINT
        or source.get("required_materializer_fingerprint")
        != MATERIALIZER_FINGERPRINT
        or source.get("required_factual_schedule_fingerprint")
        != FACTUAL_SCHEDULE_FINGERPRINT
        or source.get("required_outcome_schedule_fingerprint")
        != OUTCOME_SCHEDULE_FINGERPRINT
        or source.get("required_all_pair_inputs_fingerprint")
        != ALL_PAIR_INPUTS_FINGERPRINT
        or source.get("required_gt_union_population_fingerprint")
        != GT_UNION_POPULATION_FINGERPRINT
        or source.get("required_outcome_sequence_fingerprint")
        != OUTCOME_SEQUENCE_FINGERPRINT
    ):
        raise RuntimeError("CC-SEA v8 protocol/input bindings changed")

    if (
        closure.get("repo_path") != IMPLEMENTATION_CLOSURE_REPO_PATH
        or closure.get("schema_version") != IMPLEMENTATION_CLOSURE_SCHEMA
        or closure.get("required_before_any_D_R_payload_access") is not True
        or closure.get("may_directly_authorize_real_D_R_run") is not False
        or authorization.get("future_repo_path") != AUTHORIZATION_REPO_PATH
        or authorization.get("required_after_implementation_closure")
        is not True
        or authorization.get("must_authorize_exactly_one_real_D_R_run")
        is not True
        or authorization.get("may_authorize_D_V_or_D_T") is not False
        or authorization.get("may_authorize_formal_800") is not False
        or policy.get("create_only_output") is not True
        or policy.get("resume_allowed") is not False
        or policy.get("automatic_retry_allowed") is not False
        or policy.get("same_version_real_bounded_runs_max") != 1
        or policy.get("required_device") != FROZEN_DEVICE
        or policy.get("required_gpu_index") != 0
        or policy.get("pause_temperature_celsius") != 82
        or policy.get("resume_temperature_celsius") != 75
        or policy.get("frozen_output_repo_path") != OUTPUT_REPO_PATH
        or policy.get("D_V_access_allowed") is not False
        or policy.get("D_T_access_allowed") is not False
        or policy.get("performance_evaluation_allowed") is not False
        or policy.get("calibration_allowed") is not False
        or policy.get("formal_800_training_allowed_by_this_config")
        is not False
        or policy.get("full_CURE_allowed") is not False
        or policy.get("other_detector_integration_allowed") is not False
        or semantics.get("not_detection_performance_evidence") is not True
        or semantics.get("directly_authorizes_formal_800") is not False
        or semantics.get("bounded_pass")
        != "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_PASS"
        or semantics.get("bounded_nonpass")
        != "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_FAIL"
        or semantics.get("structural_failure")
        != "CC_SEA_V8_STRUCTURAL_EXECUTION_FAIL"
        or semantics.get("execution_error")
        != "CC_SEA_V8_BOUNDED_EXECUTION_ERROR"
    ):
        raise RuntimeError("CC-SEA v8 execution boundary changed")


def _load_config(path: Path) -> dict[str, Any]:
    expected = (_ROOT / CONFIG_REPO_PATH).resolve()
    if path != expected:
        raise RuntimeError("CC-SEA v8 config path differs from the freeze")
    if file_sha256(path) != CONFIG_FILE_SHA256:
        raise RuntimeError("CC-SEA v8 config is not the frozen file")
    config = _strict_json(path, name="CC-SEA v8 bounded config")
    _validate_config_payload(config)
    return config


def _load_implementation_proposal(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    proposal, path = _load_exact_signed(
        path_text=IMPLEMENTATION_PROPOSAL_REPO_PATH,
        expected_sha256=IMPLEMENTATION_PROPOSAL_FILE_SHA256,
        expected_fingerprint=IMPLEMENTATION_PROPOSAL_FINGERPRINT,
        name="CC-SEA v8 bounded implementation proposal",
        fingerprint_field="proposal_fingerprint",
    )
    boundary = proposal.get("current_boundary")
    scope = proposal.get("implementation_scope")
    if (
        proposal.get("schema_version")
        != "cure-lite-cc-sea-v8-bounded-implementation-proposal-v1"
        or proposal.get("method_id")
        != CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID
        or proposal.get("decision")
        != "CC_SEA_V8_REAL_BOUNDED_CODE_CREATION_AUTHORIZED"
        or not isinstance(boundary, Mapping)
        or boundary.get("real_D_R_bounded_execution_authorized")
        is not False
        or boundary.get(
            "D_R_dataset_or_cached_tensor_payload_access_allowed"
        )
        is not False
        or boundary.get("D_V_access_allowed") is not False
        or boundary.get("D_T_access_allowed") is not False
        or not isinstance(scope, Mapping)
        or scope.get("v7_executor_or_runner_import_allowed") is not False
        or scope.get("real_run_authorization_created_in_this_stage")
        is not False
        or config.get("bounded_implementation_proposal_binding", {}).get(
            "file_sha256"
        )
        != file_sha256(path)
    ):
        raise RuntimeError(
            "CC-SEA v8 bounded implementation proposal changed"
        )
    return proposal, path


def _load_frozen_dry_evidence() -> dict[str, tuple[dict[str, Any], Path]]:
    method, method_path = _load_exact_signed(
        path_text=METHOD_PROPOSAL_REPO_PATH,
        expected_sha256=METHOD_PROPOSAL_FILE_SHA256,
        expected_fingerprint=METHOD_PROPOSAL_FINGERPRINT,
        name="CC-SEA v8 method proposal",
        fingerprint_field="proposal_fingerprint",
    )
    toy, toy_path = _load_exact_signed(
        path_text=TOY_CLOSURE_REPO_PATH,
        expected_sha256=TOY_CLOSURE_FILE_SHA256,
        expected_fingerprint=TOY_CLOSURE_FINGERPRINT,
        name="CC-SEA v8 toy closure",
    )
    dry_proposal, dry_proposal_path = _load_exact_signed(
        path_text=DRY_PROPOSAL_REPO_PATH,
        expected_sha256=DRY_PROPOSAL_FILE_SHA256,
        expected_fingerprint=DRY_PROPOSAL_FINGERPRINT,
        name="CC-SEA v8 dry-run v3 proposal",
        fingerprint_field="proposal_fingerprint",
    )
    dry_config, dry_config_path = _load_exact_signed(
        path_text=DRY_CONFIG_REPO_PATH,
        expected_sha256=DRY_CONFIG_FILE_SHA256,
        expected_fingerprint=DRY_CONFIG_FINGERPRINT,
        name="CC-SEA v8 dry-run v3 config",
        fingerprint_field="config_fingerprint",
    )
    dry_result, dry_result_path = _load_exact_signed(
        path_text=DRY_RESULT_REPO_PATH,
        expected_sha256=DRY_RESULT_FILE_SHA256,
        expected_fingerprint=DRY_RESULT_FINGERPRINT,
        name="CC-SEA v8 dry-run v3 result",
        fingerprint_field="result_fingerprint",
    )
    dry_closure, dry_closure_path = _load_exact_signed(
        path_text=DRY_CLOSURE_REPO_PATH,
        expected_sha256=DRY_CLOSURE_FILE_SHA256,
        expected_fingerprint=DRY_CLOSURE_FINGERPRINT,
        name="CC-SEA v8 dry-run closure",
    )

    protocol = dry_closure.get("protocol_bindings")
    scope = dry_closure.get("authorization_scope")
    if (
        method.get("method_id")
        != CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID
        or toy.get("method_id")
        != CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID
        or toy.get("decision")
        != "CC_SEA_V8_TOY_GATE_PASS_AND_DRY_RUN_CODE_AUTHORIZED"
        or dry_proposal.get("schema_version")
        != "cure-lite-cc-sea-v8-dry-run-proposal-v3"
        or dry_proposal.get("method_id")
        != CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID
        or dry_config.get("schema_version")
        != "cure-lite-cc-sea-v8-bounded-dry-run-config-v3"
        or dry_config.get("method_id")
        != CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID
        or dry_result.get("schema_version")
        != "cure-lite-cc-sea-v8-bounded-dry-run-result-v3"
        or dry_result.get("method_id")
        != CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID
        or dry_result.get("decision")
        != "CC_SEA_V8_DRY_RUN_SINGLE_PROCESS_PASS"
        or dry_result.get("all_pass") is not True
        or dry_closure.get("schema_version")
        != "cure-lite-cc-sea-v8-bounded-dry-run-closure-v1"
        or dry_closure.get("method_id")
        != CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID
        or dry_closure.get("decision")
        != (
            "CC_SEA_V8_DRY_RUN_CLOSURE_PASS_AND_REAL_BOUNDED_CODE_AUTHORIZED"
        )
        or not isinstance(protocol, Mapping)
        or not _binding_matches(
            protocol.get("dry_run_proposal_v3"),
            repo_path=DRY_PROPOSAL_REPO_PATH,
            file_sha256_value=DRY_PROPOSAL_FILE_SHA256,
            fingerprint=DRY_PROPOSAL_FINGERPRINT,
        )
        or not _binding_matches(
            protocol.get("dry_run_config_v3"),
            repo_path=DRY_CONFIG_REPO_PATH,
            file_sha256_value=DRY_CONFIG_FILE_SHA256,
            fingerprint=DRY_CONFIG_FINGERPRINT,
        )
        or dry_closure.get("single_process_result_binding", {}).get(
            "file_sha256"
        )
        != DRY_RESULT_FILE_SHA256
        or dry_closure.get("single_process_result_binding", {}).get(
            "result_fingerprint"
        )
        != DRY_RESULT_FINGERPRINT
        or not isinstance(scope, Mapping)
        or scope.get("real_D_R_bounded_code_creation_authorized")
        is not True
        or scope.get("real_D_R_bounded_execution_authorized") is not False
        or scope.get("real_D_R_payload_access_authorized") is not False
        or scope.get("real_run_authorization_receipt_created") is not False
    ):
        raise RuntimeError("authoritative CC-SEA dry-run v3 chain changed")

    return {
        "method_proposal": (method, method_path),
        "toy_closure": (toy, toy_path),
        "dry_proposal": (dry_proposal, dry_proposal_path),
        "dry_config": (dry_config, dry_config_path),
        "dry_result": (dry_result, dry_result_path),
        "dry_closure": (dry_closure, dry_closure_path),
    }


def _implementation_binding() -> dict[str, object]:
    """Build the additive v8 runtime inventory without importing v7 code."""

    inherited = v4_runner._implementation_binding()
    inherited_files = inherited.get("all_runtime_files")
    inherited_fingerprint = stable_fingerprint(inherited)
    if (
        not isinstance(inherited_files, Mapping)
        or len(inherited_files) != 45
        or inherited_fingerprint
        != V4_IMPLEMENTATION_RECEIPT_FINGERPRINT
    ):
        raise RuntimeError("frozen v4 runtime binding changed")

    dry_closure, dry_closure_path = _load_exact_signed(
        path_text=DRY_CLOSURE_REPO_PATH,
        expected_sha256=DRY_CLOSURE_FILE_SHA256,
        expected_fingerprint=DRY_CLOSURE_FINGERPRINT,
        fingerprint_field="receipt_fingerprint",
        name="CC-SEA v8 dry-run closure",
    )
    dry_source_binding = dry_closure.get("source_bindings")
    dry_source_files = (
        dry_source_binding.get("files")
        if isinstance(dry_source_binding, Mapping)
        else None
    )
    if (
        not isinstance(dry_source_files, Mapping)
        or len(dry_source_files) != 62
    ):
        raise RuntimeError("CC-SEA v8 dry source binding changed")
    for path_text, digest in dry_source_files.items():
        path = _repo_file(path_text, name="CC-SEA v8 dry source file")
        if file_sha256(path) != digest:
            raise RuntimeError(
                f"CC-SEA v8 dry source file changed: {path_text}"
            )

    paths = (
        _ROOT / "cure_lite" / "conservative_factorized_config.py",
        _ROOT / "cure_lite" / "conservative_factorized_decoder.py",
        _ROOT / "cure_lite" / "crossing_factorized_config.py",
        _ROOT / "cure_lite" / "crossing_factorized_decoder.py",
        _ROOT / "cure_lite" / "experiment" / "p0_geometry.py",
        _ROOT / "cure_lite" / "experiment" / "p0_protocol.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "paired_formal_schedule.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "conservative_factorized_outcome_bounded.py",
        _ROOT / "tools" / "run_conservative_factorized_outcome_bounded.py",
    )
    v8_files: dict[str, str] = {}
    for path in paths:
        canonical = _canonical_file(path, name="CC-SEA v8 runtime file")
        v8_files[canonical.relative_to(_ROOT).as_posix()] = file_sha256(
            canonical
        )
    inventories = (
        dict(inherited_files),
        dict(dry_source_files),
        v8_files,
    )
    for first_index, first in enumerate(inventories):
        for second in inventories[first_index + 1 :]:
            for path_text in set(first) & set(second):
                if first[path_text] != second[path_text]:
                    raise RuntimeError(
                        "CC-SEA v8 runtime inventories disagree on "
                        f"{path_text}"
                    )
    all_files: dict[str, object] = {}
    for inventory in inventories:
        all_files.update(inventory)
    return {
        "schema_version": IMPLEMENTATION_SCHEMA,
        "method_id": CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID,
        "v4_implementation_receipt_fingerprint": inherited_fingerprint,
        "v4_runtime_files": dict(sorted(inherited_files.items())),
        "dry_v3_loaded_source_files": dict(
            sorted(dry_source_files.items())
        ),
        "v8_runtime_files": dict(sorted(v8_files.items())),
        "all_runtime_files": dict(sorted(all_files.items())),
        "v7_executor_or_runner_imported": False,
    }


def _verify_implementation_files(
    unsigned: Mapping[str, Any],
) -> None:
    all_files = unsigned.get("all_runtime_files")
    v4_files = unsigned.get("v4_runtime_files")
    dry_files = unsigned.get("dry_v3_loaded_source_files")
    v8_files = unsigned.get("v8_runtime_files")
    if (
        unsigned.get("schema_version") != IMPLEMENTATION_SCHEMA
        or unsigned.get("method_id")
        != CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID
        or unsigned.get("v4_implementation_receipt_fingerprint")
        != V4_IMPLEMENTATION_RECEIPT_FINGERPRINT
        or unsigned.get("v7_executor_or_runner_imported") is not False
        or not isinstance(all_files, Mapping)
        or not isinstance(v4_files, Mapping)
        or not isinstance(dry_files, Mapping)
        or not isinstance(v8_files, Mapping)
        or len(v4_files) != 45
        or len(dry_files) != 62
        or len(v8_files) != 9
        or set(all_files)
        != set(v4_files) | set(dry_files) | set(v8_files)
    ):
        raise RuntimeError("CC-SEA v8 runtime inventory changed")
    for inventory in (v4_files, dry_files, v8_files):
        if any(all_files[path] != digest for path, digest in inventory.items()):
            raise RuntimeError("CC-SEA v8 runtime inventory disagrees")
    for path_text, digest in all_files.items():
        path = _repo_file(path_text, name="CC-SEA v8 runtime file")
        if file_sha256(path) != digest:
            raise RuntimeError(
                f"CC-SEA v8 runtime file hash changed: {path_text}"
            )


def _expected_temperature_command() -> list[str]:
    return [
        PYTHON_EXECUTABLE,
        TEMPERATURE_WRAPPER_REPO_PATH,
        "--gpu",
        "0",
        "--pause-temp",
        "82",
        "--resume-temp",
        "75",
        "--",
        PYTHON_EXECUTABLE,
        "tools/run_conservative_factorized_outcome_bounded.py",
        "--config",
        CONFIG_REPO_PATH,
        "--device",
        FROZEN_DEVICE,
        "--output",
        OUTPUT_REPO_PATH,
    ]


def _expected_post_signing_closure_test_command() -> list[str]:
    return [
        PYTHON_EXECUTABLE,
        "-m",
        "pytest",
        "-q",
        _CLOSURE_STATIC_TEST_REPO_PATH,
    ]


def _protocol_binding(
    evidence: Mapping[str, tuple[dict[str, Any], Path]],
    name: str,
    *,
    fingerprint_field: str,
) -> dict[str, object]:
    payload, path = evidence[name]
    return {
        "repo_path": path.relative_to(_ROOT).as_posix(),
        "file_sha256": file_sha256(path),
        "fingerprint": payload[fingerprint_field],
        "fingerprint_field": fingerprint_field,
    }


def _validate_test_record(
    record: object,
    *,
    required_test_path: str,
    post_signing: bool,
) -> bool:
    if not isinstance(record, Mapping):
        return False
    command = record.get("command")
    command_text = (
        " ".join(str(value) for value in command)
        if isinstance(command, list)
        else ""
    )
    passed = record.get("passed_count")
    selected = record.get("selected_count")
    collected = record.get("collected_count")
    if (
        "pytest" not in command_text
        or required_test_path not in command_text
        or record.get("exit_code") != 0
        or record.get("failed_count") != 0
        or record.get("skipped_count") != 0
        or record.get("deselected_count") != 0
        or not isinstance(passed, int)
        or isinstance(passed, bool)
        or passed < 1
        or selected != passed
        or collected != selected
        or record.get("D_R_payload_accessed") is not False
        or record.get("D_V_accessed") is not False
        or record.get("D_T_accessed") is not False
        or record.get("evidence_stage")
        != ("post_signing" if post_signing else "pre_signing")
        or record.get("closure_receipt_present_during_execution")
        is not post_signing
    ):
        return False
    return True


def _load_implementation_closure(
    config: Mapping[str, Any],
    implementation_unsigned: Mapping[str, Any],
    dry_evidence: Mapping[
        str, tuple[dict[str, Any], Path]
    ] | None = None,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Load the signed code closure; never loads a D_R payload."""

    if dry_evidence is None:
        dry_evidence = _load_frozen_dry_evidence()
    path = _repo_file(
        config["implementation_closure_contract"]["repo_path"],
        name="CC-SEA v8 bounded implementation closure",
    )
    if path.relative_to(_ROOT).as_posix() != IMPLEMENTATION_CLOSURE_REPO_PATH:
        raise RuntimeError("CC-SEA v8 closure path changed")
    closure = _strict_json(
        path,
        name="CC-SEA v8 bounded implementation closure",
    )
    _verify_fingerprinted(
        closure,
        name="CC-SEA v8 bounded implementation closure",
    )
    runtime_signed = closure.get("runtime_implementation_binding")
    if not isinstance(runtime_signed, Mapping):
        raise RuntimeError("CC-SEA v8 closure runtime binding is missing")
    runtime_unsigned = _verified_unsigned_receipt(
        runtime_signed,
        name="CC-SEA v8 closure runtime implementation",
    )
    _verify_implementation_files(runtime_unsigned)

    proposal_path = _repo_file(
        IMPLEMENTATION_PROPOSAL_REPO_PATH,
        name="CC-SEA v8 implementation proposal",
    )
    config_path = _repo_file(CONFIG_REPO_PATH, name="CC-SEA v8 config")
    wrapper_path = _repo_file(
        TEMPERATURE_WRAPPER_REPO_PATH,
        name="CC-SEA v8 GPU temperature wrapper",
    )
    protocol = closure.get("protocol_bindings")
    gate = closure.get("gate_summary")
    boundary = closure.get("boundary")
    eligibility = closure.get("authorization_eligibility")
    tests = closure.get("test_evidence")
    temperature = closure.get("gpu_temperature_control_evidence")
    sync = closure.get("sync_benchmark_binding")
    sync_result, sync_result_path = _load_exact_signed(
        path_text=SYNC_BENCHMARK_REPO_PATH,
        expected_sha256=SYNC_BENCHMARK_FILE_SHA256,
        expected_fingerprint=SYNC_BENCHMARK_FINGERPRINT,
        fingerprint_field="result_fingerprint",
        name="CC-SEA v8 synchronization benchmark",
    )
    sync_tool_path = _repo_file(
        SYNC_BENCHMARK_TOOL_REPO_PATH,
        name="CC-SEA v8 synchronization benchmark tool",
    )
    sync_scope = sync_result.get("scope")
    sync_equivalence = sync_result.get("full_decoder_equivalence")
    sync_boundary = sync_result.get("numerical_boundary_audit")
    sync_counts = sync_result.get(
        "local_scalar_dense_calls_per_decoder_forward"
    )
    if (
        closure.get("schema_version") != IMPLEMENTATION_CLOSURE_SCHEMA
        or closure.get("method_id")
        != CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID
        or closure.get("phase_status")
        != "FROZEN_BOUNDED_IMPLEMENTATION_PASS"
        or closure.get("decision")
        != "CC_SEA_V8_BOUNDED_IMPLEMENTATION_GATE_PASS"
        or runtime_unsigned != dict(implementation_unsigned)
        or not isinstance(protocol, Mapping)
        or not _binding_matches(
            protocol.get("bounded_implementation_proposal"),
            repo_path=IMPLEMENTATION_PROPOSAL_REPO_PATH,
            file_sha256_value=file_sha256(proposal_path),
            fingerprint=IMPLEMENTATION_PROPOSAL_FINGERPRINT,
        )
        or not _binding_matches(
            protocol.get("bounded_config"),
            repo_path=CONFIG_REPO_PATH,
            file_sha256_value=file_sha256(config_path),
            fingerprint=CONFIG_FINGERPRINT,
        )
        or not _binding_matches(
            protocol.get("dry_run_proposal_v3"),
            repo_path=DRY_PROPOSAL_REPO_PATH,
            file_sha256_value=DRY_PROPOSAL_FILE_SHA256,
            fingerprint=DRY_PROPOSAL_FINGERPRINT,
        )
        or not _binding_matches(
            protocol.get("dry_run_config_v3"),
            repo_path=DRY_CONFIG_REPO_PATH,
            file_sha256_value=DRY_CONFIG_FILE_SHA256,
            fingerprint=DRY_CONFIG_FINGERPRINT,
        )
        or not _binding_matches(
            protocol.get("dry_run_result_v3"),
            repo_path=DRY_RESULT_REPO_PATH,
            file_sha256_value=DRY_RESULT_FILE_SHA256,
            fingerprint=DRY_RESULT_FINGERPRINT,
        )
        or not _binding_matches(
            protocol.get("dry_run_closure"),
            repo_path=DRY_CLOSURE_REPO_PATH,
            file_sha256_value=DRY_CLOSURE_FILE_SHA256,
            fingerprint=DRY_CLOSURE_FINGERPRINT,
        )
        or not isinstance(gate, Mapping)
        or gate.get("all_required_checks_pass") is not True
        or gate.get("mock_core_pass_and_nonpass_verified") is not True
        or gate.get("three_publication_outcomes_verified") is not True
        or gate.get("strict_loader_verified") is not True
        or gate.get(
            "closure_failure_precedes_D_R_loader_and_output_claim"
        )
        is not True
        or gate.get(
            "authorization_failure_precedes_D_R_loader_and_output_claim"
        )
        is not True
        or not isinstance(boundary, Mapping)
        or boundary.get("D_R_payload_accessed") is not False
        or boundary.get("D_V_accessed") is not False
        or boundary.get("D_T_accessed") is not False
        or boundary.get("real_D_R_bounded_execution_authorized")
        is not False
        or boundary.get("formal_800_authorized") is not False
        or not isinstance(eligibility, Mapping)
        or eligibility.get("single_real_D_R_run_eligible") is not True
        or eligibility.get("directly_authorizes_real_D_R_run") is not False
        or eligibility.get("formal_800_authorized") is not False
        or not isinstance(tests, Mapping)
        or not _validate_test_record(
            tests.get("core_executor_tests"),
            required_test_path=_CORE_EXECUTOR_TEST_REPO_PATH,
            post_signing=False,
        )
        or not _validate_test_record(
            tests.get("protocol_tests"),
            required_test_path=_PROTOCOL_TEST_REPO_PATH,
            post_signing=False,
        )
        or not _validate_test_record(
            tests.get("sync_tests"),
            required_test_path=_SYNC_TEST_REPO_PATH,
            post_signing=False,
        )
        or not _validate_test_record(
            tests.get("real_runner_publication_tests"),
            required_test_path=_REAL_RUNNER_TEST_REPO_PATH,
            post_signing=False,
        )
        or not _validate_test_record(
            tests.get("full_v8_regression"),
            required_test_path="tests_v8",
            post_signing=False,
        )
        or not _validate_test_record(
            tests.get("v7_closure_regression"),
            required_test_path=_V7_CLOSURE_TEST_REPO_PATH,
            post_signing=False,
        )
        or not isinstance(temperature, Mapping)
        or temperature.get("wrapper_repo_path")
        != TEMPERATURE_WRAPPER_REPO_PATH
        or temperature.get("wrapper_file_sha256")
        != TEMPERATURE_WRAPPER_FILE_SHA256
        or file_sha256(wrapper_path) != TEMPERATURE_WRAPPER_FILE_SHA256
        or temperature.get("gpu_index") != 0
        or temperature.get("pause_temperature_celsius") != 82
        or temperature.get("resume_temperature_celsius") != 75
        or not isinstance(sync, Mapping)
        or sync.get("result_repo_path") != SYNC_BENCHMARK_REPO_PATH
        or sync.get("result_file_sha256") != SYNC_BENCHMARK_FILE_SHA256
        or sync.get("result_fingerprint") != SYNC_BENCHMARK_FINGERPRINT
        or file_sha256(sync_result_path) != SYNC_BENCHMARK_FILE_SHA256
        or sync.get("tool_repo_path") != SYNC_BENCHMARK_TOOL_REPO_PATH
        or sync.get("tool_file_sha256")
        != SYNC_BENCHMARK_TOOL_FILE_SHA256
        or file_sha256(sync_tool_path)
        != SYNC_BENCHMARK_TOOL_FILE_SHA256
        or sync.get("production_local_scalar_calls_per_forward") != 9
        or sync.get("unchecked_local_scalar_calls_per_forward") != 0
        or sync.get("unchecked_is_diagnostic_only") is not True
        or sync.get("production_decoder_modified") is not False
        or not isinstance(sync_scope, Mapping)
        or sync_scope.get("synthetic_tensors_only") is not True
        or sync_scope.get("dataset_or_cache_payload_loaded") is not False
        or sync_scope.get("D_R_accessed") is not False
        or sync_scope.get("D_V_accessed") is not False
        or sync_scope.get("D_T_accessed") is not False
        or not isinstance(sync_equivalence, Mapping)
        or not all(
            sync_equivalence.get(
                "decoder_output_bit_exact_to_production",
                {},
            ).values()
        )
        or not all(
            sync_equivalence.get(
                "parameter_gradient_bit_exact_to_production",
                {},
            ).values()
        )
        or not isinstance(sync_boundary, Mapping)
        or sync_boundary.get("all_required_invalid_rejected") is not True
        or sync_counts
        != {"production": 9, "unchecked_diagnostic": 0}
    ):
        raise RuntimeError(
            "CC-SEA v8 bounded implementation closure changed"
        )

    # Also prove that the objects loaded above are the exact authoritative
    # dry-v3 objects.  This avoids accepting a closure that merely repeats the
    # expected strings while the local files differ.
    for name, repo_path, fingerprint_field in (
        ("method_proposal", METHOD_PROPOSAL_REPO_PATH, "proposal_fingerprint"),
        ("toy_closure", TOY_CLOSURE_REPO_PATH, "receipt_fingerprint"),
        ("dry_proposal", DRY_PROPOSAL_REPO_PATH, "proposal_fingerprint"),
        ("dry_config", DRY_CONFIG_REPO_PATH, "config_fingerprint"),
        ("dry_result", DRY_RESULT_REPO_PATH, "result_fingerprint"),
        ("dry_closure", DRY_CLOSURE_REPO_PATH, "receipt_fingerprint"),
    ):
        payload, loaded_path = dry_evidence[name]
        if (
            loaded_path.relative_to(_ROOT).as_posix() != repo_path
            or not isinstance(payload.get(fingerprint_field), str)
        ):
            raise RuntimeError("CC-SEA v8 dry evidence binding changed")
    return closure, path, dict(runtime_signed)


def _validate_authorization_payload(
    receipt: Mapping[str, Any],
    *,
    closure: Mapping[str, Any],
    closure_path: Path,
    implementation_unsigned: Mapping[str, Any],
) -> None:
    _verify_fingerprinted(
        receipt,
        name="CC-SEA v8 bounded run authorization",
    )
    authorization = receipt.get("authorization")
    config_binding = receipt.get("bounded_config_binding")
    closure_binding = receipt.get("implementation_closure_binding")
    runtime = receipt.get("runtime_implementation_binding")
    control = receipt.get("execution_control_binding")
    post_signing = receipt.get("post_signing_closure_test_evidence")
    if (
        receipt.get("schema_version") != AUTHORIZATION_SCHEMA
        or receipt.get("method_id")
        != CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID
        or receipt.get("split") != "D_R"
        or receipt.get("phase_status")
        != "FROZEN_SINGLE_REAL_D_R_RUN_AUTHORIZATION"
        or receipt.get("decision")
        != "CC_SEA_V8_ONE_REAL_D_R_BOUNDED_RUN_AUTHORIZED"
        or not isinstance(authorization, Mapping)
        or authorization.get("real_D_R_bounded_execution") is not True
        or authorization.get("exact_run_count") != 1
        or authorization.get("device") != FROZEN_DEVICE
        or authorization.get("output_repo_path") != OUTPUT_REPO_PATH
        or authorization.get("resume_allowed") is not False
        or authorization.get("automatic_retry_allowed") is not False
        or authorization.get("D_V_access_allowed") is not False
        or authorization.get("D_T_access_allowed") is not False
        or authorization.get("formal_800_allowed") is not False
        or not _binding_matches(
            config_binding,
            repo_path=CONFIG_REPO_PATH,
            file_sha256_value=CONFIG_FILE_SHA256,
            fingerprint=CONFIG_FINGERPRINT,
        )
        or not _binding_matches(
            closure_binding,
            repo_path=IMPLEMENTATION_CLOSURE_REPO_PATH,
            file_sha256_value=file_sha256(closure_path),
            fingerprint=str(closure.get("receipt_fingerprint")),
        )
        or not isinstance(runtime, Mapping)
        or runtime.get("implementation_fingerprint")
        != stable_fingerprint(implementation_unsigned)
        or runtime.get("all_runtime_files")
        != implementation_unsigned.get("all_runtime_files")
        or not isinstance(control, Mapping)
        or control.get("gpu_index") != 0
        or control.get("pause_temperature_celsius") != 82
        or control.get("resume_temperature_celsius") != 75
        or control.get("wrapper_repo_path")
        != TEMPERATURE_WRAPPER_REPO_PATH
        or control.get("wrapper_file_sha256")
        != TEMPERATURE_WRAPPER_FILE_SHA256
        or control.get("wrapped_command") != _expected_temperature_command()
        or not _validate_test_record(
            post_signing,
            required_test_path=_CLOSURE_STATIC_TEST_REPO_PATH,
            post_signing=True,
        )
        or post_signing.get("command")
        != _expected_post_signing_closure_test_command()
        or post_signing.get("closure_repo_path")
        != IMPLEMENTATION_CLOSURE_REPO_PATH
        or post_signing.get("closure_file_sha256")
        != file_sha256(closure_path)
        or post_signing.get("closure_receipt_fingerprint")
        != closure.get("receipt_fingerprint")
    ):
        raise RuntimeError("CC-SEA v8 bounded run authorization changed")


def _load_authorization(
    config: Mapping[str, Any],
    closure: Mapping[str, Any],
    closure_path: Path,
    implementation_unsigned: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Load separate one-run authorization; never loads a D_R payload."""

    path = _repo_file(
        config["future_pre_run_authorization_contract"][
            "future_repo_path"
        ],
        name="CC-SEA v8 bounded run authorization",
    )
    if path.relative_to(_ROOT).as_posix() != AUTHORIZATION_REPO_PATH:
        raise RuntimeError("CC-SEA v8 authorization path changed")
    receipt = _strict_json(
        path,
        name="CC-SEA v8 bounded run authorization",
    )
    _validate_authorization_payload(
        receipt,
        closure=closure,
        closure_path=closure_path,
        implementation_unsigned=implementation_unsigned,
    )
    return receipt, path


def _anchor_spec(config: Mapping[str, Any]) -> dict[str, object]:
    anchors = config["anchor_population"]
    return {
        key: anchors[key]
        for key in (
            "seed",
            "factual_miss_anchors",
            "factual_no_miss_anchors",
            "identity_null_pairs",
        )
    }


def _optimization_budget(
    config: Mapping[str, Any],
) -> dict[str, object]:
    budget = config["budget"]
    optimization = config["optimization"]
    return {
        "seed": optimization["seed"],
        "optimizer_updates": budget["optimizer_updates"],
        "steps_per_epoch": budget["steps_per_epoch"],
        "factual_miss_states_per_update": budget[
            "factual_miss_states_per_update"
        ],
        "factual_no_miss_states_per_update": budget[
            "factual_no_miss_states_per_update"
        ],
        "outcome_pairs_per_update": budget["outcome_pairs_per_update"],
        "learning_rate": optimization["learning_rate"],
        "weight_decay": optimization["weight_decay"],
    }


@dataclass(frozen=True)
class _FrozenRealInputs:
    source_config: Mapping[str, Any]
    source_config_path: Path
    bundle: object
    immutable: Mapping[str, str]
    population: object
    factual_schedule: object
    materializer: object
    outcome_schedule: object
    pair_catalog_fingerprint: str
    prepared_catalog_fingerprint: str


def _load_frozen_real_inputs(
    config: Mapping[str, Any],
) -> _FrozenRealInputs:
    """The sole function allowed to invoke the frozen real D_R loader."""

    source_binding = config["source_reconstruction"]
    source_path = _repo_file(
        source_binding["source_config_path"],
        name="frozen D_R source reconstruction config",
    )
    source_config = v3_runner.legacy_runner._load_config(source_path)
    if (
        file_sha256(source_path)
        != source_binding["source_config_file_sha256"]
        or source_config.get("config_fingerprint")
        != source_binding["source_config_fingerprint"]
        or source_config.get("split") != "D_R"
    ):
        raise RuntimeError("frozen CC-SEA v8 D_R source config changed")

    pair_catalog, prepared, bundle, immutable = (
        v3_runner.legacy_runner._load_real_catalog(source_config)
    )
    if (
        pair_catalog.catalog_fingerprint != PAIR_CATALOG_FINGERPRINT
        or pair_catalog.split != "D_R"
        or len(pair_catalog.clean_positive) != 206
        or len(pair_catalog.component_null) != 16
    ):
        raise RuntimeError("frozen CC-SEA v8 outcome catalog changed")

    population = v3_runner.build_outcome_bounded_anchor_population(
        pair_catalog,
        prepared,
        _anchor_spec(config),
    )
    budget = config["budget"]
    factual_schedule = v3_runner.build_outcome_factual_anchor_schedule(
        population,
        optimizer_updates=budget["optimizer_updates"],
        steps_per_epoch=budget["steps_per_epoch"],
    )
    materializer = v3_runner.build_paired_outcome_input_materializer(
        pair_catalog,
        prepared,
    )
    outcome_schedule = v3_runner.build_outcome_pair_schedule(
        pair_catalog,
        seed=config["optimization"]["seed"],
        optimizer_updates=budget["optimizer_updates"],
        steps_per_epoch=budget["steps_per_epoch"],
    )
    materializer_receipt = materializer.canonical_receipt()
    if (
        population.prepared_catalog_fingerprint
        != PREPARED_CATALOG_FINGERPRINT
        or materializer.prepared_catalog_fingerprint
        != PREPARED_CATALOG_FINGERPRINT
        or population.population_fingerprint
        != ANCHOR_POPULATION_FINGERPRINT
        or materializer.materializer_fingerprint
        != MATERIALIZER_FINGERPRINT
        or materializer_receipt.get("all_outcome_pair_input_fingerprint")
        != ALL_PAIR_INPUTS_FINGERPRINT
        or materializer_receipt.get("gt_union_population_fingerprint")
        != GT_UNION_POPULATION_FINGERPRINT
        or factual_schedule.schedule_fingerprint
        != FACTUAL_SCHEDULE_FINGERPRINT
        or outcome_schedule.schedule_fingerprint
        != OUTCOME_SCHEDULE_FINGERPRINT
        or outcome_schedule.sequence_fingerprint
        != OUTCOME_SEQUENCE_FINGERPRINT
    ):
        raise RuntimeError("CC-SEA v8 frozen D_R inputs changed")
    return _FrozenRealInputs(
        source_config=source_config,
        source_config_path=source_path,
        bundle=bundle,
        immutable=dict(immutable),
        population=population,
        factual_schedule=factual_schedule,
        materializer=materializer,
        outcome_schedule=outcome_schedule,
        pair_catalog_fingerprint=pair_catalog.catalog_fingerprint,
        prepared_catalog_fingerprint=(
            population.prepared_catalog_fingerprint
        ),
    )


def execute_conservative_factorized_outcome_bounded(
    *args: object,
    **kwargs: object,
) -> dict[str, object]:
    """Lazy v8 executor bridge, kept as a monkeypatchable runner boundary."""

    from cure_lite.experiment.conservative_factorized_outcome_bounded import (
        execute_conservative_factorized_outcome_bounded as execute,
    )

    return execute(*args, **kwargs)


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {_INCOMPLETE, "COMPLETE.json"}
    }


def _verify_internal_fingerprint(
    payload: Mapping[str, Any],
    *,
    field: str,
    name: str,
) -> None:
    unsigned = dict(payload)
    unsigned.pop("receipt_fingerprint", None)
    observed = unsigned.pop(field, None)
    if (
        not isinstance(observed, str)
        or stable_fingerprint(unsigned) != observed
    ):
        raise RuntimeError(f"CC-SEA v8 {name} fingerprint changed")


def _finite_number(value: object, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
        and (not positive or float(value) > 0.0)
    )


def _verify_trace_and_exposure(
    trace: object,
    exposure: object,
) -> None:
    if (
        not isinstance(trace, list)
        or len(trace) != 400
        or not isinstance(exposure, Mapping)
    ):
        raise RuntimeError("CC-SEA v8 trace or exposure ledger is missing")
    pair_counts: Counter[str] = Counter()
    miss_counts: Counter[str] = Counter()
    no_miss_counts: Counter[str] = Counter()
    for update, row in enumerate(trace):
        if not isinstance(row, Mapping):
            raise RuntimeError("CC-SEA v8 trace row is malformed")
        pair_ids = row.get("outcome_pair_ids")
        pair_kinds = row.get("outcome_pair_kinds")
        miss_ids = row.get("factual_miss_ids")
        no_miss_ids = row.get("factual_no_miss_ids")
        if (
            row.get("update") != update
            or row.get("epoch") != update // 40
            or row.get("step") != update % 40
            or not isinstance(pair_ids, list)
            or len(pair_ids) != 2
            or len(set(pair_ids)) != 2
            or not all(isinstance(value, str) for value in pair_ids)
            or not isinstance(pair_kinds, list)
            or len(pair_kinds) != 2
            or any(
                value not in {"clean_positive", "component_null"}
                for value in pair_kinds
            )
            or not isinstance(miss_ids, list)
            or len(miss_ids) != 4
            or not isinstance(no_miss_ids, list)
            or len(no_miss_ids) != 4
            or row.get("decoder_forward_calls") != 3
            or row.get("decoder_state_evaluations") != 12
            or not _finite_number(
                row.get("gradient_l2_norm"),
                positive=True,
            )
        ):
            raise RuntimeError("CC-SEA v8 exact update trace changed")
        pair_counts.update(pair_ids)
        miss_counts.update(str(value) for value in miss_ids)
        no_miss_counts.update(str(value) for value in no_miss_ids)

    pair_rows = exposure.get("outcome_pairs")
    miss_rows = exposure.get("factual_miss")
    no_miss_rows = exposure.get("factual_no_miss")
    source_rows = exposure.get("source_images")
    if (
        not isinstance(pair_rows, list)
        or len(pair_rows) != 222
        or not isinstance(miss_rows, list)
        or len(miss_rows) != 16
        or not isinstance(no_miss_rows, list)
        or len(no_miss_rows) != 16
        or not isinstance(source_rows, list)
        or exposure.get("outcome_pair_exposure_values") != [3, 4]
        or exposure.get("identity_null_optimizer_exposure") != 0
    ):
        raise RuntimeError("CC-SEA v8 exposure population changed")

    declared_pair_counts = {
        str(row.get("pair_id")): row.get("count")
        for row in pair_rows
        if isinstance(row, Mapping)
    }
    declared_miss_counts = {
        str(row.get("anchor_id")): row.get("count")
        for row in miss_rows
        if isinstance(row, Mapping)
    }
    declared_no_miss_counts = {
        str(row.get("anchor_id")): row.get("count")
        for row in no_miss_rows
        if isinstance(row, Mapping)
    }
    declared_source_counts = {
        str(row.get("sample_id")): row.get("count")
        for row in source_rows
        if isinstance(row, Mapping)
    }
    recomputed_source_counts: Counter[str] = Counter()
    for row in pair_rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("CC-SEA v8 pair exposure row is malformed")
        sample_id = row.get("sample_id")
        count = row.get("count")
        if not isinstance(sample_id, str) or not isinstance(count, int):
            raise RuntimeError("CC-SEA v8 pair exposure row changed")
        recomputed_source_counts[sample_id] += count
    if (
        declared_pair_counts != dict(pair_counts)
        or declared_miss_counts != dict(miss_counts)
        or declared_no_miss_counts != dict(no_miss_counts)
        or sum(pair_counts.values()) != 800
        or sum(miss_counts.values()) != 1600
        or sum(no_miss_counts.values()) != 1600
        or set(pair_counts.values()) != {3, 4}
        or declared_source_counts != dict(recomputed_source_counts)
        or sum(recomputed_source_counts.values()) != 800
        or len(declared_source_counts) != len(source_rows)
    ):
        raise RuntimeError("CC-SEA v8 exposure ledgers do not reproduce")


def _verify_core_result(result: Mapping[str, Any]) -> None:
    """Strictly validate the unsigned executor result.

    Deep numerical/state-equation validity is produced by the independent v8
    executor.  This publication layer rechecks its identity, fixed experiment
    contract, decision algebra, stop rule, complete ledgers, and bounded
    computational evidence before accepting it.
    """

    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint", None)
    structural = result.get("structural_execution_pass")
    model_pass = result.get("computational_model_code_gate_pass")
    audit = result.get("pretraining_structural_audit")
    structural_checks = result.get("structural_checks")
    interpretation = result.get("interpretation")
    expected_budget = {
        "seed": 42,
        "optimizer_updates": 400,
        "steps_per_epoch": 40,
        "factual_miss_states_per_update": 4,
        "factual_no_miss_states_per_update": 4,
        "outcome_pairs_per_update": 2,
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
    }
    if (
        not isinstance(fingerprint, str)
        or stable_fingerprint(unsigned) != fingerprint
        or result.get("schema_version")
        != CONSERVATIVE_FACTORIZED_OUTCOME_BOUNDED_SCHEMA
        or result.get("method_id")
        != CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID
        or result.get("execution_status") != "completed"
        or result.get("device") != FROZEN_DEVICE
        or not isinstance(structural, bool)
        or not isinstance(model_pass, bool)
        or (model_pass and not structural)
        or result.get("population_fingerprint")
        != ANCHOR_POPULATION_FINGERPRINT
        or result.get("materializer_fingerprint")
        != MATERIALIZER_FINGERPRINT
        or result.get("factual_schedule_fingerprint")
        != FACTUAL_SCHEDULE_FINGERPRINT
        or result.get("outcome_schedule_fingerprint")
        != OUTCOME_SCHEDULE_FINGERPRINT
        or result.get("decoder_config")
        != vars(ConservativeFactorizedDecoderConfig(64, 4))
        or result.get("loss_config")
        != {"dice_weight": 1.0, "epsilon": 1.0e-6}
        or result.get("optimization_budget") != expected_budget
        or result.get("evaluation_chunk_size") != 32
        or not isinstance(audit, Mapping)
        or not isinstance(audit.get("checks"), Mapping)
        or not all(
            isinstance(value, bool)
            for value in audit["checks"].values()
        )
        or audit.get("all_pass") is not all(audit["checks"].values())
        or audit.get("pair_count") != 222
        or audit.get("clean_pair_count") != 206
        or audit.get("component_null_pair_count") != 16
        or audit.get("training_performed") is not False
        or audit.get("D_V_accessed") is not False
        or audit.get("D_T_accessed") is not False
        or not isinstance(structural_checks, Mapping)
        or not structural_checks
        or not all(
            isinstance(value, bool)
            for value in structural_checks.values()
        )
        or not isinstance(interpretation, Mapping)
        or interpretation.get("not_detection_performance_evidence")
        is not True
        or interpretation.get("does_not_establish_Pd_or_FA") is not True
        or interpretation.get("does_not_authorize_formal_training")
        is not True
        or interpretation.get("does_not_directly_authorize_formal_800")
        is not True
        or interpretation.get("eligible_for_frozen_review")
        is not model_pass
        or interpretation.get("D_V_accessed") is not False
        or interpretation.get("D_T_accessed") is not False
        or interpretation.get("calibration_performed") is not False
        or interpretation.get("inference_performed") is not False
        or interpretation.get("base_or_backbone_updated") is not False
        or interpretation.get("identity_null_optimizer_exposure") != 0
    ):
        raise RuntimeError(
            "CC-SEA v8 result violates its frozen bounded boundary"
        )

    expected_decision = (
        "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_PASS"
        if model_pass
        else (
            "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_FAIL"
            if structural
            else "CC_SEA_V8_STRUCTURAL_EXECUTION_FAIL"
        )
    )
    if result.get("decision") != expected_decision:
        raise RuntimeError("CC-SEA v8 core decision is inconsistent")

    audit_pass = audit.get("all_pass") is True
    if not audit_pass:
        if (
            structural
            or model_pass
            or result.get("optimizer_updates_completed") != 0
            or result.get("training_performed") is not False
            or result.get("trace") != []
            or result.get("computational_gates")
            != {
                "status": "NOT_EVALUATED_BY_STRUCTURAL_STOP_RULE",
                "all_pass": None,
            }
            or not any(
                value is False for value in structural_checks.values()
            )
            or result.get("forward_budget", {}).get("training")
            != {"calls": 0, "state_evaluations": 0}
        ):
            raise RuntimeError(
                "CC-SEA v8 structural zero-update stop rule changed"
            )
        return

    initial = result.get("initial")
    final = result.get("final")
    computational = result.get("computational_gates")
    parameters = result.get("parameters")
    gradients = result.get("gradients")
    ledger = result.get("execution_ledger")
    forward = result.get("forward_budget")
    deterministic = result.get("deterministic_runtime")
    observation = result.get("state_equation_observation")
    if (
        not isinstance(initial, Mapping)
        or not isinstance(final, Mapping)
        or not isinstance(computational, Mapping)
        or not isinstance(computational.get("checks"), Mapping)
        or len(computational["checks"]) != 12
        or not all(
            isinstance(value, bool)
            for value in computational["checks"].values()
        )
        or computational.get("all_pass")
        is not all(computational["checks"].values())
        or computational.get("all_pass") is not model_pass
        or structural is not all(structural_checks.values())
        or result.get("optimizer_updates_completed") != 400
        or result.get("training_performed") is not True
        or not isinstance(parameters, Mapping)
        or parameters.get("trainable_parameter_count") != 4385
        or parameters.get("expected_parameter_count") != 4385
        or any(
            not isinstance(parameters.get(name), str)
            or len(str(parameters.get(name))) != 64
            for name in (
                "initial_decoder_fingerprint",
                "final_decoder_fingerprint",
            )
        )
        or parameters.get("initial_decoder_fingerprint")
        == parameters.get("final_decoder_fingerprint")
        or not _finite_number(
            parameters.get("initial_l2_norm"),
            positive=True,
        )
        or not _finite_number(
            parameters.get("final_l2_norm"),
            positive=True,
        )
        or not isinstance(gradients, Mapping)
        or not _finite_number(
            gradients.get("minimum_update_l2_norm"),
            positive=True,
        )
        or not _finite_number(
            gradients.get("maximum_update_l2_norm"),
            positive=True,
        )
        or float(gradients["minimum_update_l2_norm"])
        > float(gradients["maximum_update_l2_norm"])
        or gradients.get("nonfinite_updates") != 0
        or gradients.get("zero_norm_updates") != 0
        or not isinstance(ledger, Mapping)
        or dict(ledger)
        != {
            "backward_calls": 400,
            "optimizer_steps": 400,
            "expected_backward_calls": 400,
            "expected_optimizer_steps": 400,
        }
        or not isinstance(forward, Mapping)
        or forward.get("training")
        != {"calls": 1200, "state_evaluations": 4800}
        or forward.get("expected_training")
        != {"calls": 1200, "state_evaluations": 4800}
        or not isinstance(deterministic, Mapping)
        or deterministic.get("contract_satisfied") is not True
        or deterministic.get("flags_restored_after_execution") is not True
        or not isinstance(observation, Mapping)
        or observation.get("additional_decoder_forward_calls") != 0
    ):
        raise RuntimeError("CC-SEA v8 full bounded evidence changed")
    _verify_trace_and_exposure(result.get("trace"), result.get("exposure"))


def _decision(
    result: Mapping[str, object] | None,
    *,
    failure: Mapping[str, object] | None,
    evidence_receipt_fingerprint: str,
) -> dict[str, object]:
    if result is None:
        status = "CC_SEA_V8_BOUNDED_EXECUTION_ERROR"
        structural = False
        model_pass = False
    else:
        structural = result.get("structural_execution_pass") is True
        model_pass = (
            result.get("computational_model_code_gate_pass") is True
        )
        status = (
            "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_PASS"
            if model_pass
            else (
                "CC_SEA_V8_BOUNDED_MODEL_CODE_GATE_FAIL"
                if structural
                else "CC_SEA_V8_STRUCTURAL_EXECUTION_FAIL"
            )
        )
    return _fingerprinted(
        {
            "schema_version": DECISION_SCHEMA,
            "method_id": CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID,
            "status": status,
            "structural_execution_pass": structural,
            "bounded_model_code_gate_pass": model_pass,
            "not_detection_performance_evidence": True,
            "directly_authorizes_formal_800": False,
            "authorizes_D_V_or_D_T": False,
            "authorizes_full_CURE": False,
            "authorizes_other_detector_integration": False,
            "evidence_kind": (
                "result" if result is not None else "failure"
            ),
            "evidence_receipt_fingerprint": (
                evidence_receipt_fingerprint
            ),
            "failure": dict(failure) if failure is not None else None,
            "next_action": (
                "freeze_and_review_CC_SEA_v8_bounded_model_code_evidence"
                if model_pass
                else "preserve_v8_evidence_and_stop_without_retry"
            ),
        }
    )


@dataclass(frozen=True)
class PublishedConservativeFactorizedOutcomeBounded:
    root: Path
    decision: str
    structural_execution_pass: bool
    bounded_model_code_gate_pass: bool
    complete_fingerprint: str

    def verify_unchanged(self) -> None:
        if load_conservative_factorized_outcome_bounded_artifact(
            self.root
        ) != self:
            raise RuntimeError("published CC-SEA v8 artifact changed")


def _verify_published_input_receipts(
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    source = payloads["source_reconstruction"]
    anchor = payloads["anchor_population"]
    factual = payloads["factual_schedule"]
    inputs = payloads["outcome_inputs"]
    schedule = payloads["outcome_schedule"]
    for payload, field, name in (
        (anchor, "population_fingerprint", "anchor population"),
        (factual, "schedule_fingerprint", "factual schedule"),
        (inputs, "materializer_fingerprint", "outcome inputs"),
        (schedule, "schedule_fingerprint", "outcome schedule"),
    ):
        _verify_internal_fingerprint(payload, field=field, name=name)
    source_path = _repo_file(
        source.get("source_config_repo_path"),
        name="published CC-SEA v8 D_R source config",
    )
    source_config = v3_runner.legacy_runner._load_config(source_path)
    if (
        source.get("schema_version")
        != "cure-lite-cc-sea-v8-source-reconstruction-v1"
        or source.get("split") != "D_R"
        or source.get("source_config_repo_path")
        != v3_runner.legacy_runner.BOUNDED_CONFIG_REPO_PATH
        or source.get("source_config_file_sha256")
        != v3_runner.legacy_runner.BOUNDED_CONFIG_FILE_SHA256
        or file_sha256(source_path)
        != v3_runner.legacy_runner.BOUNDED_CONFIG_FILE_SHA256
        or source.get("source_config_fingerprint")
        != v3_runner.legacy_runner.BOUNDED_CONFIG_FINGERPRINT
        or source_config.get("config_fingerprint")
        != v3_runner.legacy_runner.BOUNDED_CONFIG_FINGERPRINT
        or source.get("pair_catalog_fingerprint")
        != PAIR_CATALOG_FINGERPRINT
        or source.get("prepared_catalog_fingerprint")
        != PREPARED_CATALOG_FINGERPRINT
        or source.get("D_V_accessed") is not False
        or source.get("D_T_accessed") is not False
        or anchor.get("pair_catalog_fingerprint")
        != PAIR_CATALOG_FINGERPRINT
        or anchor.get("prepared_catalog_fingerprint")
        != PREPARED_CATALOG_FINGERPRINT
        or anchor.get("population_fingerprint")
        != ANCHOR_POPULATION_FINGERPRINT
        or factual.get("population_fingerprint")
        != ANCHOR_POPULATION_FINGERPRINT
        or factual.get("schedule_fingerprint")
        != FACTUAL_SCHEDULE_FINGERPRINT
        or inputs.get("pair_catalog_fingerprint")
        != PAIR_CATALOG_FINGERPRINT
        or inputs.get("prepared_catalog_fingerprint")
        != PREPARED_CATALOG_FINGERPRINT
        or inputs.get("materializer_fingerprint")
        != MATERIALIZER_FINGERPRINT
        or inputs.get("all_outcome_pair_input_fingerprint")
        != ALL_PAIR_INPUTS_FINGERPRINT
        or inputs.get("gt_union_population_fingerprint")
        != GT_UNION_POPULATION_FINGERPRINT
        or schedule.get("catalog_fingerprint")
        != PAIR_CATALOG_FINGERPRINT
        or schedule.get("sequence_fingerprint")
        != OUTCOME_SEQUENCE_FINGERPRINT
        or schedule.get("schedule_fingerprint")
        != OUTCOME_SCHEDULE_FINGERPRINT
    ):
        raise RuntimeError("published CC-SEA v8 D_R inputs changed")
    return {
        "population_fingerprint": ANCHOR_POPULATION_FINGERPRINT,
        "factual_schedule_fingerprint": FACTUAL_SCHEDULE_FINGERPRINT,
        "materializer_fingerprint": MATERIALIZER_FINGERPRINT,
        "outcome_schedule_fingerprint": OUTCOME_SCHEDULE_FINGERPRINT,
        "all_pair_inputs_fingerprint": ALL_PAIR_INPUTS_FINGERPRINT,
        "gt_union_population_fingerprint": (
            GT_UNION_POPULATION_FINGERPRINT
        ),
        "outcome_sequence_fingerprint": OUTCOME_SEQUENCE_FINGERPRINT,
    }


def _binding_receipt(
    *,
    schema_version: str,
    repo_path: str,
    path: Path,
    payload: Mapping[str, Any],
    fingerprint_field: str,
    payload_field: str,
) -> dict[str, object]:
    return _fingerprinted(
        {
            "schema_version": schema_version,
            "repo_path": repo_path,
            "file_sha256": file_sha256(path),
            "bound_fingerprint": payload[fingerprint_field],
            "fingerprint_field": fingerprint_field,
            payload_field: dict(payload),
        }
    )


def _pre_run_receipts(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    implementation_proposal: Mapping[str, Any],
    implementation_proposal_path: Path,
    dry_evidence: Mapping[str, tuple[dict[str, Any], Path]],
    closure: Mapping[str, Any],
    closure_path: Path,
    authorization: Mapping[str, Any],
    authorization_path: Path,
    implementation_signed: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    method, method_path = dry_evidence["method_proposal"]
    toy, toy_path = dry_evidence["toy_closure"]
    dry_proposal, dry_proposal_path = dry_evidence["dry_proposal"]
    dry_config, dry_config_path = dry_evidence["dry_config"]
    dry_result, dry_result_path = dry_evidence["dry_result"]
    dry_closure, dry_closure_path = dry_evidence["dry_closure"]
    return {
        "config_binding.json": _binding_receipt(
            schema_version="cure-lite-cc-sea-v8-config-binding-v1",
            repo_path=CONFIG_REPO_PATH,
            path=config_path,
            payload=config,
            fingerprint_field="config_fingerprint",
            payload_field="config",
        ),
        "method_proposal_binding.json": _binding_receipt(
            schema_version=(
                "cure-lite-cc-sea-v8-method-proposal-binding-v1"
            ),
            repo_path=METHOD_PROPOSAL_REPO_PATH,
            path=method_path,
            payload=method,
            fingerprint_field="proposal_fingerprint",
            payload_field="proposal",
        ),
        "toy_gate_binding.json": _binding_receipt(
            schema_version="cure-lite-cc-sea-v8-toy-binding-v1",
            repo_path=TOY_CLOSURE_REPO_PATH,
            path=toy_path,
            payload=toy,
            fingerprint_field="receipt_fingerprint",
            payload_field="closure",
        ),
        "dry_run_proposal_binding.json": _binding_receipt(
            schema_version=(
                "cure-lite-cc-sea-v8-dry-proposal-binding-v1"
            ),
            repo_path=DRY_PROPOSAL_REPO_PATH,
            path=dry_proposal_path,
            payload=dry_proposal,
            fingerprint_field="proposal_fingerprint",
            payload_field="proposal",
        ),
        "dry_run_config_binding.json": _binding_receipt(
            schema_version="cure-lite-cc-sea-v8-dry-config-binding-v1",
            repo_path=DRY_CONFIG_REPO_PATH,
            path=dry_config_path,
            payload=dry_config,
            fingerprint_field="config_fingerprint",
            payload_field="config",
        ),
        "dry_run_result_binding.json": _binding_receipt(
            schema_version="cure-lite-cc-sea-v8-dry-result-binding-v1",
            repo_path=DRY_RESULT_REPO_PATH,
            path=dry_result_path,
            payload=dry_result,
            fingerprint_field="result_fingerprint",
            payload_field="result",
        ),
        "dry_run_closure_binding.json": _binding_receipt(
            schema_version="cure-lite-cc-sea-v8-dry-closure-binding-v1",
            repo_path=DRY_CLOSURE_REPO_PATH,
            path=dry_closure_path,
            payload=dry_closure,
            fingerprint_field="receipt_fingerprint",
            payload_field="closure",
        ),
        "implementation_proposal_binding.json": _binding_receipt(
            schema_version=(
                "cure-lite-cc-sea-v8-implementation-proposal-binding-v1"
            ),
            repo_path=IMPLEMENTATION_PROPOSAL_REPO_PATH,
            path=implementation_proposal_path,
            payload=implementation_proposal,
            fingerprint_field="proposal_fingerprint",
            payload_field="proposal",
        ),
        "implementation_closure_binding.json": _binding_receipt(
            schema_version=(
                "cure-lite-cc-sea-v8-implementation-closure-binding-v1"
            ),
            repo_path=IMPLEMENTATION_CLOSURE_REPO_PATH,
            path=closure_path,
            payload=closure,
            fingerprint_field="receipt_fingerprint",
            payload_field="closure",
        ),
        "authorization_binding.json": _binding_receipt(
            schema_version=(
                "cure-lite-cc-sea-v8-authorization-binding-v1"
            ),
            repo_path=AUTHORIZATION_REPO_PATH,
            path=authorization_path,
            payload=authorization,
            fingerprint_field="receipt_fingerprint",
            payload_field="authorization",
        ),
        "implementation_binding.json": dict(implementation_signed),
        "run_claim.json": _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cc-sea-v8-single-run-claim-v1"
                ),
                "method_id": CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID,
                "split": "D_R",
                "device": FROZEN_DEVICE,
                "real_D_R_run_count_claimed": 1,
                "claim_consumed_before_first_D_R_payload_loader": True,
                "resume_allowed": False,
                "automatic_retry_allowed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        ),
    }


def _verify_embedded_binding(
    receipt: Mapping[str, Any],
    *,
    expected_schema: str,
    expected_repo_path: str,
    current_payload: Mapping[str, Any],
    current_path: Path,
    fingerprint_field: str,
    payload_field: str,
) -> None:
    if (
        receipt.get("schema_version") != expected_schema
        or receipt.get("repo_path") != expected_repo_path
        or receipt.get("file_sha256") != file_sha256(current_path)
        or receipt.get("bound_fingerprint")
        != current_payload.get(fingerprint_field)
        or receipt.get("fingerprint_field") != fingerprint_field
        or receipt.get(payload_field) != current_payload
    ):
        raise RuntimeError(
            f"published CC-SEA v8 {payload_field} binding changed"
        )


def load_conservative_factorized_outcome_bounded_artifact(
    output_dir: str | Path,
    *,
    _allow_incomplete: bool = False,
) -> PublishedConservativeFactorizedOutcomeBounded:
    candidate = Path(output_dir).expanduser()
    absolute = Path(os.path.abspath(candidate))
    if candidate.is_symlink():
        raise ValueError("CC-SEA v8 bounded root may not be a symlink")
    root = candidate.resolve(strict=True)
    if root != absolute or not root.is_dir() or root.is_symlink():
        raise ValueError("CC-SEA v8 root must be a regular directory")
    incomplete = (root / _INCOMPLETE).exists()
    if incomplete and not _allow_incomplete:
        raise RuntimeError("CC-SEA v8 publication is incomplete")
    expected_top = {"receipts", "COMPLETE.json"}
    if _allow_incomplete:
        expected_top.add(_INCOMPLETE)
    if {item.name for item in root.iterdir()} != expected_top:
        raise RuntimeError("CC-SEA v8 top-level inventory changed")

    receipts_root = root / "receipts"
    if (
        receipts_root.is_symlink()
        or not receipts_root.is_dir()
        or any(
            item.is_symlink() or not item.is_file()
            for item in receipts_root.iterdir()
        )
    ):
        raise RuntimeError("CC-SEA v8 receipts must be regular files")
    names = {item.name for item in receipts_root.iterdir()}
    pre_failure = _PRE_RUN_RECEIPTS | {
        "decision.json",
        "failure.json",
    }
    full_result = (
        _PRE_RUN_RECEIPTS
        | _INPUT_RECEIPTS
        | {"decision.json", "result.json"}
    )
    full_failure = (
        _PRE_RUN_RECEIPTS
        | _INPUT_RECEIPTS
        | {"decision.json", "failure.json"}
    )
    if names not in (pre_failure, full_result, full_failure):
        raise RuntimeError("CC-SEA v8 receipt inventory changed")

    complete = _strict_json(
        root / "COMPLETE.json",
        name="CC-SEA v8 COMPLETE",
    )
    _verify_fingerprinted(
        complete,
        name="CC-SEA v8 COMPLETE",
        field="complete_fingerprint",
    )
    payloads = {
        name[:-5]: _strict_json(
            receipts_root / name,
            name=f"CC-SEA v8 {name[:-5]}",
        )
        for name in names
    }
    for name, payload in payloads.items():
        _verify_fingerprinted(payload, name=f"CC-SEA v8 {name}")

    if (
        complete.get("artifact_files") != _artifact_hashes(root)
        or complete.get("artifact_file_count")
        != len(_artifact_hashes(root))
        or complete.get("schema_version") != RUN_SCHEMA
        or complete.get("method_id")
        != CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID
        or complete.get("execution_status") != "complete"
        or complete.get("device") != FROZEN_DEVICE
        or complete.get("split") != "D_R"
        or complete.get("D_V_accessed") is not False
        or complete.get("D_T_accessed") is not False
        or complete.get("performance_evaluation_performed") is not False
        or complete.get("calibration_performed") is not False
        or complete.get("formal_800_training_performed") is not False
        or complete.get("resume_used") is not False
        or complete.get("automatic_retry_performed") is not False
        or complete.get("real_D_R_run_count") != 1
        or complete.get("real_D_R_run_claim_consumed") is not True
    ):
        raise RuntimeError("CC-SEA v8 COMPLETE boundary changed")

    config_path = _repo_file(CONFIG_REPO_PATH, name="CC-SEA v8 config")
    config = _load_config(config_path)
    dry_evidence = _load_frozen_dry_evidence()
    proposal, proposal_path = _load_implementation_proposal(config)
    implementation_unsigned = _implementation_binding()
    _verify_implementation_files(implementation_unsigned)
    implementation_signed = _fingerprinted(implementation_unsigned)
    closure, closure_path, closure_runtime_signed = (
        _load_implementation_closure(
            config,
            implementation_unsigned,
            dry_evidence,
        )
    )
    if closure_runtime_signed != implementation_signed:
        raise RuntimeError("CC-SEA v8 closure runtime binding changed")
    authorization, authorization_path = _load_authorization(
        config,
        closure,
        closure_path,
        implementation_unsigned,
    )

    _verify_embedded_binding(
        payloads["config_binding"],
        expected_schema="cure-lite-cc-sea-v8-config-binding-v1",
        expected_repo_path=CONFIG_REPO_PATH,
        current_payload=config,
        current_path=config_path,
        fingerprint_field="config_fingerprint",
        payload_field="config",
    )
    binding_specs = (
        (
            "method_proposal_binding",
            "cure-lite-cc-sea-v8-method-proposal-binding-v1",
            "method_proposal",
            METHOD_PROPOSAL_REPO_PATH,
            "proposal_fingerprint",
            "proposal",
        ),
        (
            "toy_gate_binding",
            "cure-lite-cc-sea-v8-toy-binding-v1",
            "toy_closure",
            TOY_CLOSURE_REPO_PATH,
            "receipt_fingerprint",
            "closure",
        ),
        (
            "dry_run_proposal_binding",
            "cure-lite-cc-sea-v8-dry-proposal-binding-v1",
            "dry_proposal",
            DRY_PROPOSAL_REPO_PATH,
            "proposal_fingerprint",
            "proposal",
        ),
        (
            "dry_run_config_binding",
            "cure-lite-cc-sea-v8-dry-config-binding-v1",
            "dry_config",
            DRY_CONFIG_REPO_PATH,
            "config_fingerprint",
            "config",
        ),
        (
            "dry_run_result_binding",
            "cure-lite-cc-sea-v8-dry-result-binding-v1",
            "dry_result",
            DRY_RESULT_REPO_PATH,
            "result_fingerprint",
            "result",
        ),
        (
            "dry_run_closure_binding",
            "cure-lite-cc-sea-v8-dry-closure-binding-v1",
            "dry_closure",
            DRY_CLOSURE_REPO_PATH,
            "receipt_fingerprint",
            "closure",
        ),
    )
    for (
        receipt_name,
        schema,
        evidence_name,
        repo_path,
        fingerprint_field,
        payload_field,
    ) in binding_specs:
        current_payload, current_path = dry_evidence[evidence_name]
        _verify_embedded_binding(
            payloads[receipt_name],
            expected_schema=schema,
            expected_repo_path=repo_path,
            current_payload=current_payload,
            current_path=current_path,
            fingerprint_field=fingerprint_field,
            payload_field=payload_field,
        )
    _verify_embedded_binding(
        payloads["implementation_proposal_binding"],
        expected_schema=(
            "cure-lite-cc-sea-v8-implementation-proposal-binding-v1"
        ),
        expected_repo_path=IMPLEMENTATION_PROPOSAL_REPO_PATH,
        current_payload=proposal,
        current_path=proposal_path,
        fingerprint_field="proposal_fingerprint",
        payload_field="proposal",
    )
    _verify_embedded_binding(
        payloads["implementation_closure_binding"],
        expected_schema=(
            "cure-lite-cc-sea-v8-implementation-closure-binding-v1"
        ),
        expected_repo_path=IMPLEMENTATION_CLOSURE_REPO_PATH,
        current_payload=closure,
        current_path=closure_path,
        fingerprint_field="receipt_fingerprint",
        payload_field="closure",
    )
    _verify_embedded_binding(
        payloads["authorization_binding"],
        expected_schema=(
            "cure-lite-cc-sea-v8-authorization-binding-v1"
        ),
        expected_repo_path=AUTHORIZATION_REPO_PATH,
        current_payload=authorization,
        current_path=authorization_path,
        fingerprint_field="receipt_fingerprint",
        payload_field="authorization",
    )
    if payloads["implementation_binding"] != implementation_signed:
        raise RuntimeError(
            "published CC-SEA v8 implementation binding changed"
        )
    claim = payloads["run_claim"]
    if (
        claim.get("schema_version")
        != "cure-lite-cc-sea-v8-single-run-claim-v1"
        or claim.get("method_id")
        != CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID
        or claim.get("split") != "D_R"
        or claim.get("device") != FROZEN_DEVICE
        or claim.get("real_D_R_run_count_claimed") != 1
        or claim.get(
            "claim_consumed_before_first_D_R_payload_loader"
        )
        is not True
        or claim.get("resume_allowed") is not False
        or claim.get("automatic_retry_allowed") is not False
        or claim.get("D_V_accessed") is not False
        or claim.get("D_T_accessed") is not False
    ):
        raise RuntimeError("published CC-SEA v8 run claim changed")

    has_inputs = _INPUT_RECEIPTS <= names
    if has_inputs is not (
        complete.get("input_receipts_present") is True
    ):
        raise RuntimeError("CC-SEA v8 input receipt status changed")
    input_fingerprints: dict[str, str] | None = None
    if has_inputs:
        input_fingerprints = _verify_published_input_receipts(
            {
                name[:-5]: payloads[name[:-5]]
                for name in _INPUT_RECEIPTS
            }
        )

    decision = payloads["decision"]
    if "result" in payloads:
        evidence_kind = "result"
        result_receipt = payloads["result"]
        result_unsigned = dict(result_receipt)
        result_unsigned.pop("receipt_fingerprint")
        _verify_core_result(result_unsigned)
        structural = (
            result_unsigned["structural_execution_pass"] is True
        )
        model_pass = (
            result_unsigned["computational_model_code_gate_pass"] is True
        )
        expected_status = str(result_unsigned["decision"])
        failure_unsigned = None
        if input_fingerprints is None or any(
            result_unsigned.get(key) != value
            for key, value in input_fingerprints.items()
            if key
            in {
                "population_fingerprint",
                "factual_schedule_fingerprint",
                "materializer_fingerprint",
                "outcome_schedule_fingerprint",
            }
        ):
            raise RuntimeError("CC-SEA v8 result/input binding changed")
    else:
        evidence_kind = "failure"
        failure_receipt = payloads["failure"]
        failure_unsigned = dict(failure_receipt)
        failure_unsigned.pop("receipt_fingerprint")
        if (
            failure_unsigned.get("schema_version") != FAILURE_SCHEMA
            or failure_unsigned.get("phase")
            not in {
                "D_R_RECONSTRUCTION",
                "BOUNDED_EXECUTION",
                "POST_EXECUTION_IMMUTABILITY",
            }
            or not isinstance(
                failure_unsigned.get("exception_type"),
                str,
            )
            or not isinstance(failure_unsigned.get("message"), str)
            or failure_unsigned.get("real_D_R_run_claim_consumed")
            is not True
            or failure_unsigned.get("structural_execution_pass")
            is not False
            or failure_unsigned.get("bounded_model_code_gate_pass")
            is not False
            or failure_unsigned.get("budget_or_threshold_changed")
            is not False
            or failure_unsigned.get("D_V_accessed") is not False
            or failure_unsigned.get("D_T_accessed") is not False
        ):
            raise RuntimeError("CC-SEA v8 failure receipt changed")
        structural = False
        model_pass = False
        expected_status = "CC_SEA_V8_BOUNDED_EXECUTION_ERROR"

    evidence = payloads[evidence_kind]
    if (
        decision.get("schema_version") != DECISION_SCHEMA
        or decision.get("method_id")
        != CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID
        or decision.get("status") != expected_status
        or decision.get("structural_execution_pass") is not structural
        or decision.get("bounded_model_code_gate_pass") is not model_pass
        or decision.get("evidence_kind") != evidence_kind
        or decision.get("evidence_receipt_fingerprint")
        != evidence.get("receipt_fingerprint")
        or decision.get("not_detection_performance_evidence") is not True
        or decision.get("directly_authorizes_formal_800") is not False
        or decision.get("authorizes_D_V_or_D_T") is not False
        or decision.get("authorizes_full_CURE") is not False
        or decision.get("authorizes_other_detector_integration")
        is not False
        or decision.get("failure") != failure_unsigned
        or decision.get("next_action")
        != (
            "freeze_and_review_CC_SEA_v8_bounded_model_code_evidence"
            if model_pass
            else "preserve_v8_evidence_and_stop_without_retry"
        )
        or complete.get("decision") != expected_status
        or complete.get("decision_fingerprint")
        != decision.get("receipt_fingerprint")
        or complete.get("evidence_kind") != evidence_kind
        or complete.get("evidence_receipt_fingerprint")
        != evidence.get("receipt_fingerprint")
        or complete.get("implementation_receipt_fingerprint")
        != implementation_signed.get("receipt_fingerprint")
    ):
        raise RuntimeError("CC-SEA v8 decision binding changed")

    return PublishedConservativeFactorizedOutcomeBounded(
        root=root,
        decision=expected_status,
        structural_execution_pass=structural,
        bounded_model_code_gate_pass=model_pass,
        complete_fingerprint=str(complete["complete_fingerprint"]),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    config_path = _canonical_file(
        args.config,
        name="CC-SEA v8 bounded config",
    )
    config = _load_config(config_path)
    device = _validate_device(args.device)
    output = _validate_output_target(args.output)

    # This entire chain is deliberately before output.mkdir and before the
    # sole real loader.  A missing/invalid closure or authorization therefore
    # cannot consume a run, touch a D_R payload, or publish any output claim.
    implementation_proposal, implementation_proposal_path = (
        _load_implementation_proposal(config)
    )
    dry_evidence = _load_frozen_dry_evidence()
    implementation_unsigned = _implementation_binding()
    _verify_implementation_files(implementation_unsigned)
    closure, closure_path, closure_runtime_signed = (
        _load_implementation_closure(
            config,
            implementation_unsigned,
            dry_evidence,
        )
    )
    implementation_signed = _fingerprinted(implementation_unsigned)
    if implementation_signed != closure_runtime_signed:
        raise RuntimeError(
            "closure binds a different CC-SEA v8 runtime implementation"
        )
    authorization, authorization_path = _load_authorization(
        config,
        closure,
        closure_path,
        implementation_unsigned,
    )

    # Exact-one-run claim: only now, after every static authorization check,
    # and still before the first possible D_R payload loader call.
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / _INCOMPLETE
    incomplete.open("xb").close()
    receipts = output / "receipts"
    receipts.mkdir(exist_ok=False)
    for name, payload in _pre_run_receipts(
        config=config,
        config_path=config_path,
        implementation_proposal=implementation_proposal,
        implementation_proposal_path=implementation_proposal_path,
        dry_evidence=dry_evidence,
        closure=closure,
        closure_path=closure_path,
        authorization=authorization,
        authorization_path=authorization_path,
        implementation_signed=implementation_signed,
    ).items():
        _write_new_json(receipts / name, payload)

    real_inputs: _FrozenRealInputs | None = None
    result: dict[str, object] | None = None
    execution_error: Exception | None = None
    failure_phase = "D_R_RECONSTRUCTION"
    try:
        real_inputs = _load_frozen_real_inputs(config)
        source_receipt = _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cc-sea-v8-source-reconstruction-v1"
                ),
                "split": "D_R",
                "source_config_repo_path": (
                    real_inputs.source_config_path
                    .relative_to(_ROOT)
                    .as_posix()
                ),
                "source_config_file_sha256": file_sha256(
                    real_inputs.source_config_path
                ),
                "source_config_fingerprint": real_inputs.source_config[
                    "config_fingerprint"
                ],
                "pair_catalog_fingerprint": (
                    real_inputs.pair_catalog_fingerprint
                ),
                "prepared_catalog_fingerprint": (
                    real_inputs.prepared_catalog_fingerprint
                ),
                "D_V_accessed": False,
                "D_T_accessed": False,
            }
        )
        input_receipts = {
            "source_reconstruction.json": source_receipt,
            "anchor_population.json": _fingerprinted(
                real_inputs.population.canonical_receipt()
            ),
            "factual_schedule.json": _fingerprinted(
                real_inputs.factual_schedule.canonical_receipt()
            ),
            "outcome_inputs.json": _fingerprinted(
                real_inputs.materializer.canonical_receipt()
            ),
            "outcome_schedule.json": _fingerprinted(
                real_inputs.outcome_schedule.canonical_receipt()
            ),
        }
        for name, payload in input_receipts.items():
            _write_new_json(receipts / name, payload)

        failure_phase = "BOUNDED_EXECUTION"
        result = execute_conservative_factorized_outcome_bounded(
            real_inputs.population,
            real_inputs.factual_schedule,
            real_inputs.outcome_schedule,
            real_inputs.materializer,
            ConservativeFactorizedDecoderConfig(
                **config["optimization"]["decoder"]
            ),
            LossConfig(**config["optimization"]["loss"]),
            _optimization_budget(config),
            device=device,
            evaluation_chunk_size=config["budget"][
                "evaluation_chunk_size"
            ],
        )
        _verify_core_result(result)
    except Exception as error:
        execution_error = error

    post_attempt_error: Exception | None = None
    try:
        if real_inputs is not None:
            real_inputs.bundle.verify_unchanged()
            if any(
                file_sha256(Path(path)) != digest
                for path, digest in real_inputs.immutable.items()
            ):
                raise RuntimeError(
                    "a frozen CC-SEA v8 D_R input changed during execution"
                )
        if (
            _load_config(config_path) != config
            or _load_implementation_proposal(config)[0]
            != implementation_proposal
            or _load_frozen_dry_evidence() != dry_evidence
            or _implementation_binding() != implementation_unsigned
        ):
            raise RuntimeError(
                "CC-SEA v8 static inputs changed during execution"
            )
        current_closure, current_closure_path, current_runtime_signed = (
            _load_implementation_closure(
                config,
                implementation_unsigned,
                dry_evidence,
            )
        )
        current_authorization, current_authorization_path = (
            _load_authorization(
                config,
                closure,
                closure_path,
                implementation_unsigned,
            )
        )
        if (
            current_closure != closure
            or file_sha256(current_closure_path)
            != file_sha256(closure_path)
            or current_runtime_signed != implementation_signed
            or current_authorization != authorization
            or file_sha256(current_authorization_path)
            != file_sha256(authorization_path)
        ):
            raise RuntimeError(
                "CC-SEA v8 closure or authorization changed "
                "during execution"
            )
    except Exception as error:
        post_attempt_error = error
        if execution_error is None:
            execution_error = error
            failure_phase = "POST_EXECUTION_IMMUTABILITY"

    if execution_error is None:
        if result is None:
            raise RuntimeError("CC-SEA v8 execution returned no result")
        evidence = _fingerprinted(result)
        _write_new_json(receipts / "result.json", evidence)
        failure = None
    else:
        result = None
        failure = {
            "schema_version": FAILURE_SCHEMA,
            "method_id": CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID,
            "phase": failure_phase,
            "exception_type": type(execution_error).__name__,
            "message": str(execution_error),
            "post_attempt_verification_passed": (
                post_attempt_error is None
            ),
            "post_attempt_exception_type": (
                None
                if post_attempt_error is None
                else type(post_attempt_error).__name__
            ),
            "post_attempt_exception_message": (
                None
                if post_attempt_error is None
                else str(post_attempt_error)
            ),
            "real_D_R_run_claim_consumed": True,
            "structural_execution_pass": False,
            "bounded_model_code_gate_pass": False,
            "budget_or_threshold_changed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
        }
        evidence = _fingerprinted(failure)
        _write_new_json(receipts / "failure.json", evidence)

    decision = _decision(
        result,
        failure=failure,
        evidence_receipt_fingerprint=str(
            evidence["receipt_fingerprint"]
        ),
    )
    _write_new_json(receipts / "decision.json", decision)

    artifact_files = _artifact_hashes(output)
    complete = _fingerprinted(
        {
            "schema_version": RUN_SCHEMA,
            "method_id": CONSERVATIVE_FACTORIZED_OUTCOME_METHOD_ID,
            "execution_status": "complete",
            "decision": decision["status"],
            "structural_execution_pass": decision[
                "structural_execution_pass"
            ],
            "bounded_model_code_gate_pass": decision[
                "bounded_model_code_gate_pass"
            ],
            "device": device,
            "split": "D_R",
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_evaluation_performed": False,
            "calibration_performed": False,
            "formal_800_training_performed": False,
            "resume_used": False,
            "automatic_retry_performed": False,
            "real_D_R_run_count": 1,
            "real_D_R_run_claim_consumed": True,
            "post_attempt_verification_passed": (
                post_attempt_error is None
            ),
            "input_receipts_present": real_inputs is not None,
            "config_fingerprint": CONFIG_FINGERPRINT,
            "implementation_proposal_fingerprint": (
                IMPLEMENTATION_PROPOSAL_FINGERPRINT
            ),
            "dry_run_proposal_fingerprint": DRY_PROPOSAL_FINGERPRINT,
            "dry_run_config_fingerprint": DRY_CONFIG_FINGERPRINT,
            "dry_run_result_fingerprint": DRY_RESULT_FINGERPRINT,
            "dry_run_closure_fingerprint": DRY_CLOSURE_FINGERPRINT,
            "implementation_closure_fingerprint": closure[
                "receipt_fingerprint"
            ],
            "authorization_fingerprint": authorization[
                "receipt_fingerprint"
            ],
            "pair_catalog_fingerprint": (
                None
                if real_inputs is None
                else PAIR_CATALOG_FINGERPRINT
            ),
            "prepared_catalog_fingerprint": (
                None
                if real_inputs is None
                else PREPARED_CATALOG_FINGERPRINT
            ),
            "population_fingerprint": (
                None
                if real_inputs is None
                else ANCHOR_POPULATION_FINGERPRINT
            ),
            "factual_schedule_fingerprint": (
                None
                if real_inputs is None
                else FACTUAL_SCHEDULE_FINGERPRINT
            ),
            "materializer_fingerprint": (
                None
                if real_inputs is None
                else MATERIALIZER_FINGERPRINT
            ),
            "outcome_schedule_fingerprint": (
                None
                if real_inputs is None
                else OUTCOME_SCHEDULE_FINGERPRINT
            ),
            "all_pair_inputs_fingerprint": (
                None
                if real_inputs is None
                else ALL_PAIR_INPUTS_FINGERPRINT
            ),
            "gt_union_population_fingerprint": (
                None
                if real_inputs is None
                else GT_UNION_POPULATION_FINGERPRINT
            ),
            "outcome_sequence_fingerprint": (
                None
                if real_inputs is None
                else OUTCOME_SEQUENCE_FINGERPRINT
            ),
            "implementation_receipt_fingerprint": implementation_signed[
                "receipt_fingerprint"
            ],
            "evidence_kind": decision["evidence_kind"],
            "evidence_receipt_fingerprint": evidence[
                "receipt_fingerprint"
            ],
            "decision_fingerprint": decision["receipt_fingerprint"],
            "artifact_files": artifact_files,
            "artifact_file_count": len(artifact_files),
        },
        field="complete_fingerprint",
    )
    _write_new_json(output / "COMPLETE.json", complete)
    published = load_conservative_factorized_outcome_bounded_artifact(
        output,
        _allow_incomplete=True,
    )
    incomplete.unlink()
    return {
        "output": str(output),
        "decision": published.decision,
        "structural_execution_pass": (
            published.structural_execution_pass
        ),
        "bounded_model_code_gate_pass": (
            published.bounded_model_code_gate_pass
        ),
        "complete_fingerprint": published.complete_fingerprint,
        "real_D_R_run_claim_consumed": True,
        "not_detection_performance_evidence": True,
        "directly_authorizes_formal_800": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    if result["bounded_model_code_gate_pass"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
