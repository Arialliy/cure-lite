#!/usr/bin/env python3
"""Run the single frozen D_R-only CR-LVEC v7 bounded model-code gate.

All protocol, implementation-closure, authorization, device, and output-path
checks complete before the unique r1 directory is claimed.  The claim and its
``.incomplete`` marker are written before the first D_R payload loader call,
so reconstruction or execution failure still consumes the one authorized run
and leaves a fingerprinted failure artifact.  D_V, D_T, calibration, detector
performance evaluation, resume, and automatic retry are not available here.
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
from cure_lite.crossing_factorized_config import (  # noqa: E402
    CrossingFactorizedDecoderConfig,
)
from cure_lite.experiment.crossing_factorized_outcome_bounded import (  # noqa: E402
    CROSSING_FACTORIZED_OUTCOME_BOUNDED_SCHEMA,
    CROSSING_FACTORIZED_OUTCOME_METHOD_ID,
    CROSSING_OPERATOR_STRUCTURAL_CHECKS,
    crossing_computational_gates,
    execute_crossing_factorized_outcome_bounded,
)
from tools import run_factorized_outcome_bounded as v4_runner  # noqa: E402
from tools import run_paired_outcome_bounded as v3_runner  # noqa: E402


CONFIG_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
    "bounded_config.json"
)
CONFIG_FILE_SHA256 = (
    "352c0c235134c1017b851854278255c2c678973929d3fda614389392502c4b96"
)
CONFIG_FINGERPRINT = (
    "9bdc7f5567065c02d37cc82f94b5bc49c589dfee271487f4cbce7dd831c45818"
)
PROPOSAL_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
    "proposal_receipt.json"
)
PROPOSAL_FILE_SHA256 = (
    "fa72f4ef850f72a65003e913db1b1230d7b0b45046faf61950fb1e4ef80d3c4f"
)
PROPOSAL_FINGERPRINT = (
    "9d291e6ad9ec0869aa0ab0eaebcb219cd62678420375f56af480ba105208dbf2"
)
TOY_CLOSURE_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
    "toy_gate_closure_receipt.json"
)
TOY_CLOSURE_FILE_SHA256 = (
    "25c3317045533f4116b8873d892fcd2c0e866d3e991843a4c0c8e872142f0fe5"
)
TOY_CLOSURE_FINGERPRINT = (
    "f95573edd8b842980d5b175b1aac8caf753f6c279342da8e29f54e165b1e255f"
)
IMPLEMENTATION_PROPOSAL_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
    "bounded_implementation_proposal_receipt.json"
)
IMPLEMENTATION_PROPOSAL_FILE_SHA256 = (
    "65a45dc6d73d8cbf6bcb2c6b6204251f3583e0354ba3161b633f1547fbaa11dd"
)
IMPLEMENTATION_PROPOSAL_FINGERPRINT = (
    "d33f710348dec255fd73790b3c97c643472d115d3098e1469409f7dd57fad896"
)
DRY_CONFIG_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
    "bounded_dry_run_config.json"
)
DRY_CONFIG_FILE_SHA256 = (
    "709f72bc4d17798be4fecb01f96afb1b91a9fb39f6a5da80315a71b6b501e55c"
)
DRY_CONFIG_FINGERPRINT = (
    "d5421a162822ad9962b9790a10c49c4bfe8cd7844c88c4ec5e80a7ca54559e97"
)
DRY_RESULT_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
    "bounded_dry_run_result.json"
)
DRY_RESULT_FILE_SHA256 = (
    "01f98d35602942887e1f3003894be92beb428802b7989d4b2bbd2d04756ee490"
)
DRY_RESULT_FINGERPRINT = (
    "47cf682cf16023a8c14a468e2d7a83e0630a2a1bf02ee8d69c38254baee02993"
)
IMPLEMENTATION_CLOSURE_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
    "bounded_implementation_closure_receipt.json"
)
AUTHORIZATION_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
    "bounded_run_authorization_receipt.json"
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
    "runs/irstd1k_stage_a_seed42/cure_lite_cr_lvec_v7_bounded_r1"
)
OUTPUT_VERSION_PREFIX = "cure_lite_cr_lvec_v7_bounded_"
FROZEN_DEVICE = "cuda:0"
TEMPERATURE_WRAPPER_REPO_PATH = "tools/run_with_gpu_temperature_control.py"
TEMPERATURE_WRAPPER_FILE_SHA256 = (
    "026b751fbb59530721da1436af32f3bc924c9ed2ab3576df062a45bca7ec5e86"
)
SYNC_BENCHMARK_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "continuously_recoverable_log_vacancy_evidence_crossing_v7/"
    "sync_benchmark_result_v2.json"
)
SYNC_BENCHMARK_FILE_SHA256 = (
    "b73613881ce3bb530f450a62721ceb532e7aaa6881ace76945751d672d5744ad"
)
SYNC_BENCHMARK_FINGERPRINT = (
    "0ce0df9ad5be90b4730aaea739c848d8564d462116540d3e9ce5e1cd7afd9742"
)
PYTHON_EXECUTABLE = "/home/md0/ly/MSHNet/.venv/bin/python"

RUN_SCHEMA = "cure-lite-cr-lvec-v7-bounded-run-v1"
DECISION_SCHEMA = "cure-lite-cr-lvec-v7-bounded-decision-v1"
FAILURE_SCHEMA = "cure-lite-cr-lvec-v7-bounded-failure-v1"
IMPLEMENTATION_SCHEMA = (
    "cure-lite-cr-lvec-v7-runtime-implementation-v1"
)
IMPLEMENTATION_CLOSURE_SCHEMA = (
    "cure-lite-cr-lvec-v7-bounded-implementation-closure-v1"
)
AUTHORIZATION_SCHEMA = (
    "cure-lite-cr-lvec-v7-bounded-run-authorization-v1"
)

_ROOT = Path(__file__).resolve().parents[1]
_INCOMPLETE = ".incomplete"
_PRE_RUN_RECEIPTS = {
    "authorization_binding.json",
    "config_binding.json",
    "implementation_binding.json",
    "implementation_closure_binding.json",
    "implementation_proposal_binding.json",
    "proposal_binding.json",
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
_FOCUSED_TEST_REPO_PATHS = (
    "tests/test_crossing_factorized_config.py",
    "tests/test_crossing_factorized_decoder.py",
    "tests/test_crossing_factorized_model.py",
    "tests/test_crossing_factorized_outcome_bounded.py",
    "tests/test_crossing_factorized_outcome_toy_overfit.py",
    "tests/test_crossing_factorized_toy_gate_closure.py",
    "tests/test_crossing_factorized_sync_benchmark.py",
    "tests/test_crossing_factorized_bounded_protocol.py",
    "tests/test_dry_run_crossing_factorized_outcome_bounded_cli.py",
    "tests/test_run_crossing_factorized_outcome_bounded_cli.py",
    "tests/test_gpu_temperature_control.py",
)
_REAL_RUNNER_TEST_REPO_PATH = (
    "tests/test_run_crossing_factorized_outcome_bounded_cli.py"
)
_TEMPERATURE_TEST_REPO_PATH = "tests/test_gpu_temperature_control.py"
_CLOSURE_STATIC_TEST_REPO_PATH = (
    "tests/test_crossing_factorized_bounded_implementation_closure.py"
)
_PRE_SIGNING_REAL_PAYLOAD_TEST_NODE = (
    "tests/test_paired_formal_controls.py::"
    "test_real_206_pair_static_provider_closes_authoritative_preflight"
)
_PRE_SIGNING_REAL_PAYLOAD_DESELECTION_REASON = (
    "implementation_closure_forbids_D_R_payload_access"
)


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
    ):
        raise ValueError(f"{name} must be a non-empty repo-relative path")
    path = _canonical_file(_ROOT / path_text, name=name)
    if path.relative_to(_ROOT).as_posix() != path_text:
        raise RuntimeError(f"{name} does not resolve to its frozen path")
    return path


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    return v3_runner._strict_json(path, name=name)


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
    fingerprint = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if (
        not isinstance(fingerprint, str)
        or stable_fingerprint(unsigned) != fingerprint
    ):
        raise RuntimeError(f"{name} fingerprint is inconsistent")


def _verified_unsigned_receipt(
    payload: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    """Verify a signed receipt, then return its canonical unsigned object."""

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


def _validate_device(device: object) -> str:
    if device != FROZEN_DEVICE:
        raise ValueError(
            "CR-LVEC v7 bounded execution fixes --device at cuda:0"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError(
            "CR-LVEC v7 must be launched by the frozen GPU-0 temperature "
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
            "CR-LVEC v7 permits only its frozen r1 output path: "
            f"{expected}"
        )
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(
            f"CR-LVEC bounded output already exists: {absolute}"
        )
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
                "a CR-LVEC v7 bounded run already exists: "
                + ", ".join(str(item) for item in prior)
            )
    for parent in (absolute.parent, *absolute.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError(
                "CR-LVEC bounded output may not traverse a symlink"
            )
    return absolute


def _validate_config_payload(config: Mapping[str, Any]) -> None:
    unsigned = dict(config)
    fingerprint = unsigned.pop("config_fingerprint", None)
    if (
        fingerprint != CONFIG_FINGERPRINT
        or stable_fingerprint(unsigned) != fingerprint
        or config.get("schema_version")
        != "cure-lite-cr-lvec-v7-bounded-config-v1"
        or config.get("method_id") != CROSSING_FACTORIZED_OUTCOME_METHOD_ID
        or config.get("dataset") != "IRSTD-1K"
        or config.get("split") != "D_R"
    ):
        raise RuntimeError("CR-LVEC bounded config identity changed")

    proposal = config.get("proposal_binding")
    toy = config.get("toy_gate_authorization")
    implementation_proposal = config.get(
        "bounded_implementation_proposal_binding"
    )
    source = config.get("source_reconstruction")
    optimization = config.get("optimization")
    budget = config.get("budget")
    closure = config.get("implementation_closure_contract")
    authorization = config.get("future_pre_run_authorization_contract")
    policy = config.get("execution_policy")
    semantics = config.get("decision_semantics")
    if not all(
        isinstance(value, Mapping)
        for value in (
            proposal,
            toy,
            implementation_proposal,
            source,
            optimization,
            budget,
            closure,
            authorization,
            policy,
            semantics,
        )
    ):
        raise RuntimeError("CR-LVEC bounded config sections are malformed")
    if (
        proposal.get("repo_path") != PROPOSAL_REPO_PATH
        or proposal.get("file_sha256") != PROPOSAL_FILE_SHA256
        or proposal.get("proposal_fingerprint") != PROPOSAL_FINGERPRINT
        or toy.get("closure_repo_path") != TOY_CLOSURE_REPO_PATH
        or toy.get("closure_file_sha256") != TOY_CLOSURE_FILE_SHA256
        or toy.get("closure_receipt_fingerprint")
        != TOY_CLOSURE_FINGERPRINT
        or toy.get("decision") != "CR_LVEC_V7_TOY_GATE_PASS"
        or toy.get("bounded_code_creation_authorized") is not True
        or toy.get("real_D_R_bounded_authorized_by_toy_closure")
        is not False
        or implementation_proposal.get("repo_path")
        != IMPLEMENTATION_PROPOSAL_REPO_PATH
        or implementation_proposal.get("file_sha256")
        != IMPLEMENTATION_PROPOSAL_FILE_SHA256
        or implementation_proposal.get("receipt_fingerprint")
        != IMPLEMENTATION_PROPOSAL_FINGERPRINT
    ):
        raise RuntimeError("CR-LVEC protocol bindings changed")

    decoder = optimization.get("decoder")
    loss = optimization.get("loss")
    expected_decoder = CrossingFactorizedDecoderConfig(
        feature_channels=64,
        feature_stride=4,
    )
    if (
        not isinstance(decoder, Mapping)
        or dict(decoder)
        != {
            key: value
            for key, value in vars(expected_decoder).items()
        }
        or dict(loss) != {"dice_weight": 1.0, "epsilon": 1.0e-6}
        or optimization.get("optimizer") != "adam"
        or optimization.get("learning_rate") != 1.0e-3
        or optimization.get("weight_decay") != 0.0
        or optimization.get("seed") != 42
        or optimization.get("trainable_scope")
        != "CURELiteCrossingFactorizedDecoder_only"
    ):
        raise RuntimeError("CR-LVEC optimization contract changed")
    if dict(budget) != {
        "epochs": 10,
        "steps_per_epoch": 40,
        "optimizer_updates": 400,
        "factual_miss_states_per_update": 4,
        "factual_no_miss_states_per_update": 4,
        "outcome_pairs_per_update": 2,
        "outcome_endpoint_states_per_update": 4,
        "decoder_forward_calls_per_update": 3,
        "decoder_states_per_update": 12,
        "pair_slots": 800,
        "evaluation_chunk_size": 32,
        "resume_allowed": False,
    }:
        raise RuntimeError("CR-LVEC bounded budget changed")
    if (
        source.get("source_config_path")
        != v3_runner.legacy_runner.BOUNDED_CONFIG_REPO_PATH
        or source.get("source_config_file_sha256")
        != v3_runner.legacy_runner.BOUNDED_CONFIG_FILE_SHA256
        or source.get("source_config_fingerprint")
        != v3_runner.legacy_runner.BOUNDED_CONFIG_FINGERPRINT
        or source.get("required_pair_catalog_fingerprint")
        != PAIR_CATALOG_FINGERPRINT
        or source.get("required_prepared_catalog_fingerprint")
        != PREPARED_CATALOG_FINGERPRINT
        or source.get("required_anchor_population_fingerprint")
        != ANCHOR_POPULATION_FINGERPRINT
        or source.get("required_materializer_fingerprint")
        != MATERIALIZER_FINGERPRINT
        or source.get("required_all_pair_inputs_fingerprint")
        != ALL_PAIR_INPUTS_FINGERPRINT
        or source.get("required_gt_union_population_fingerprint")
        != GT_UNION_POPULATION_FINGERPRINT
        or source.get("required_factual_schedule_fingerprint")
        != FACTUAL_SCHEDULE_FINGERPRINT
        or source.get("required_outcome_schedule_fingerprint")
        != OUTCOME_SCHEDULE_FINGERPRINT
        or source.get("required_outcome_sequence_fingerprint")
        != OUTCOME_SEQUENCE_FINGERPRINT
    ):
        raise RuntimeError("CR-LVEC D_R reconstruction binding changed")
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
        or policy.get("create_only_output") is not True
        or policy.get("resume_allowed") is not False
        or policy.get("same_version_real_bounded_runs_max") != 1
        or policy.get("automatic_retry_allowed") is not False
        or policy.get("required_device") != FROZEN_DEVICE
        or policy.get("required_gpu_index") != 0
        or policy.get("pause_temperature_celsius") != 82
        or policy.get("resume_temperature_celsius") != 75
        or policy.get("frozen_output_repo_path") != OUTPUT_REPO_PATH
        or policy.get("D_V_access_allowed") is not False
        or policy.get("D_T_access_allowed") is not False
        or policy.get("formal_800_training_allowed_by_this_config")
        is not False
        or semantics.get("not_detection_performance_evidence") is not True
        or semantics.get("directly_authorizes_formal_800") is not False
    ):
        raise RuntimeError("CR-LVEC execution boundary changed")


def _load_config(path: Path) -> dict[str, Any]:
    expected = (_ROOT / CONFIG_REPO_PATH).resolve()
    if path != expected:
        raise RuntimeError(
            "CR-LVEC bounded config path differs from the freeze"
        )
    if file_sha256(path) != CONFIG_FILE_SHA256:
        raise RuntimeError("CR-LVEC bounded config is not the frozen file")
    config = _strict_json(path, name="CR-LVEC bounded config")
    _validate_config_payload(config)
    return config


def _load_exact_signed(
    *,
    path_text: str,
    expected_sha256: str,
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
    return payload, path


def _load_proposal(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    binding = config["proposal_binding"]
    proposal, path = _load_exact_signed(
        path_text=str(binding["repo_path"]),
        expected_sha256=str(binding["file_sha256"]),
        name="CR-LVEC proposal",
        fingerprint_field="proposal_fingerprint",
    )
    if (
        proposal.get("schema_version")
        != "cure-lite-cr-lvec-v7-proposal-v1"
        or proposal.get("method_id") != CROSSING_FACTORIZED_OUTCOME_METHOD_ID
        or proposal.get("proposal_fingerprint") != PROPOSAL_FINGERPRINT
        or proposal.get("stage_decision", {}).get(
            "real_D_R_run_authorized_at_proposal_time"
        )
        is not False
    ):
        raise RuntimeError("CR-LVEC proposal contract changed")
    return proposal, path


def _load_toy_closure(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    binding = config["toy_gate_authorization"]
    closure, path = _load_exact_signed(
        path_text=str(binding["closure_repo_path"]),
        expected_sha256=str(binding["closure_file_sha256"]),
        name="CR-LVEC toy closure",
    )
    gate = closure.get("gate_summary")
    boundary = closure.get("boundary")
    if (
        closure.get("schema_version")
        != "cure-lite-cr-lvec-v7-toy-gate-closure-v1"
        or closure.get("method_id") != CROSSING_FACTORIZED_OUTCOME_METHOD_ID
        or closure.get("phase_status") != "FROZEN_TOY_GATE_PASS"
        or closure.get("decision") != "CR_LVEC_V7_TOY_GATE_PASS"
        or closure.get("receipt_fingerprint") != TOY_CLOSURE_FINGERPRINT
        or not isinstance(gate, Mapping)
        or gate.get("toy_gate_pass") is not True
        or gate.get("bounded_code_creation_authorized") is not True
        or gate.get("real_D_R_bounded_authorized") is not False
        or not isinstance(boundary, Mapping)
        or boundary.get("D_V_accessed") is not False
        or boundary.get("D_T_accessed") is not False
        or boundary.get("formal_800_authorized") is not False
    ):
        raise RuntimeError("CR-LVEC toy closure contract changed")
    return closure, path


def _load_implementation_proposal(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    binding = config["bounded_implementation_proposal_binding"]
    proposal, path = _load_exact_signed(
        path_text=str(binding["repo_path"]),
        expected_sha256=str(binding["file_sha256"]),
        name="CR-LVEC bounded implementation proposal",
    )
    boundary = proposal.get("execution_boundary")
    if (
        proposal.get("schema_version")
        != "cure-lite-cr-lvec-v7-bounded-implementation-proposal-v1"
        or proposal.get("method_id") != CROSSING_FACTORIZED_OUTCOME_METHOD_ID
        or proposal.get("decision")
        != "CR_LVEC_V7_BOUNDED_IMPLEMENTATION_CREATION_AUTHORIZED"
        or proposal.get("receipt_fingerprint")
        != IMPLEMENTATION_PROPOSAL_FINGERPRINT
        or not isinstance(boundary, Mapping)
        or boundary.get("real_D_R_bounded_execution_authorized")
        is not False
        or boundary.get("D_V_access_allowed") is not False
        or boundary.get("D_T_access_allowed") is not False
    ):
        raise RuntimeError(
            "CR-LVEC bounded implementation proposal changed"
        )
    return proposal, path


def _implementation_binding() -> dict[str, object]:
    v4 = v4_runner._implementation_binding()
    v4_files = v4.get("all_runtime_files")
    v4_fingerprint = stable_fingerprint(v4)
    if (
        not isinstance(v4_files, Mapping)
        or len(v4_files) != 45
        or v4_fingerprint != V4_IMPLEMENTATION_RECEIPT_FINGERPRINT
    ):
        raise RuntimeError("frozen v4 runtime binding changed")
    paths = (
        _ROOT / "cure_lite" / "crossing_factorized_config.py",
        _ROOT / "cure_lite" / "crossing_factorized_decoder.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "crossing_factorized_outcome_bounded.py",
        _ROOT / "tools" / "dry_run_crossing_factorized_outcome_bounded.py",
        _ROOT / "tools" / "run_crossing_factorized_outcome_bounded.py",
    )
    v7_files = {
        path.relative_to(_ROOT).as_posix(): file_sha256(path)
        for path in paths
    }
    if set(v4_files) & set(v7_files):
        raise RuntimeError("v7 runtime implementation must be additive")
    all_files = {**dict(v4_files), **v7_files}
    return {
        "schema_version": IMPLEMENTATION_SCHEMA,
        "v4_implementation_receipt_fingerprint": v4_fingerprint,
        "v4_runtime_files": dict(sorted(v4_files.items())),
        "v7_runtime_files": dict(sorted(v7_files.items())),
        "all_runtime_files": dict(sorted(all_files.items())),
    }


def _verify_implementation_files(
    unsigned: Mapping[str, Any],
) -> None:
    files = unsigned.get("all_runtime_files")
    v4_files = unsigned.get("v4_runtime_files")
    v7_files = unsigned.get("v7_runtime_files")
    if (
        unsigned.get("schema_version") != IMPLEMENTATION_SCHEMA
        or unsigned.get("v4_implementation_receipt_fingerprint")
        != V4_IMPLEMENTATION_RECEIPT_FINGERPRINT
        or not isinstance(files, Mapping)
        or not isinstance(v4_files, Mapping)
        or not isinstance(v7_files, Mapping)
        or len(v4_files) != 45
        or len(v7_files) != 5
        or len(files) != 50
        or set(v4_files) & set(v7_files)
        or dict(files) != {**dict(v4_files), **dict(v7_files)}
    ):
        raise RuntimeError("CR-LVEC runtime inventory changed")
    for path_text, digest in files.items():
        path = _repo_file(path_text, name="CR-LVEC runtime file")
        if file_sha256(path) != digest:
            raise RuntimeError(
                f"CR-LVEC runtime file hash changed: {path_text}"
            )


def _binding_matches(
    binding: object,
    *,
    repo_path: str,
    file_sha256_value: str,
    fingerprint: str,
) -> bool:
    return (
        isinstance(binding, Mapping)
        and binding.get("repo_path") == repo_path
        and binding.get("file_sha256") == file_sha256_value
        and (
            binding.get("receipt_fingerprint") == fingerprint
            or binding.get("config_fingerprint") == fingerprint
            or binding.get("proposal_fingerprint") == fingerprint
            or binding.get("result_fingerprint") == fingerprint
        )
    )


def _test_record_passes(
    record: object,
    *,
    require_three_publication_outcomes: bool = False,
    required_test_paths: Sequence[str] = (),
    require_full_inventory: bool = False,
) -> bool:
    if not isinstance(record, Mapping):
        return False
    command = record.get("command")
    if not isinstance(command, list) or not command:
        return False
    command_text = " ".join(str(value) for value in command)
    if (
        "pytest" not in command_text
        or any(path not in command_text for path in required_test_paths)
    ):
        return False
    test_files = record.get("test_file_sha256")
    if not isinstance(test_files, Mapping):
        return False
    for path_text in required_test_paths:
        path = _ROOT / path_text
        if (
            not path.is_file()
            or test_files.get(path_text) != file_sha256(path)
        ):
            return False
    if require_full_inventory:
        current = {
            path.relative_to(_ROOT).as_posix(): file_sha256(path)
            for path in sorted((_ROOT / "tests").glob("test_*.py"))
        }
        if (
            record.get("test_inventory_file_count") != len(current)
            or record.get("test_inventory_fingerprint")
            != stable_fingerprint(current)
            or record.get("pre_signing_excluded_test_files")
            != [_CLOSURE_STATIC_TEST_REPO_PATH]
            or record.get("pre_signing_deselected_real_payload_tests")
            != [_PRE_SIGNING_REAL_PAYLOAD_TEST_NODE]
            or record.get("deselected_count") != 1
            or record.get("deselection_reason")
            != _PRE_SIGNING_REAL_PAYLOAD_DESELECTION_REASON
            or "--ignore" not in command_text
            or _CLOSURE_STATIC_TEST_REPO_PATH not in command_text
            or "--deselect" not in command_text
            or _PRE_SIGNING_REAL_PAYLOAD_TEST_NODE not in command_text
        ):
            return False
    deselected = record.get("deselected_count")
    passed = record.get("passed_count")
    skipped = record.get("skipped_count")
    selected = record.get("selected_count")
    collected = record.get("collected_count")
    if (
        record.get("exit_code") != 0
        or record.get("failed_count") != 0
        or not isinstance(passed, int)
        or isinstance(passed, bool)
        or passed < 1
        or not isinstance(skipped, int)
        or isinstance(skipped, bool)
        or skipped < 0
        or not isinstance(deselected, int)
        or isinstance(deselected, bool)
        or deselected < 0
        or (require_full_inventory and deselected != 1)
        or (not require_full_inventory and deselected != 0)
        or selected != passed + skipped
        or collected != selected + deselected
        or record.get("evidence_stage") != "pre_signing"
        or record.get("closure_receipt_present_during_execution")
        is not False
        or record.get("D_R_payload_accessed") is not False
        or record.get("D_V_accessed") is not False
        or record.get("D_T_accessed") is not False
    ):
        return False
    if require_three_publication_outcomes and (
        record.get("completed_pass_publication_verified") is not True
        or record.get("completed_nonpass_publication_verified") is not True
        or record.get("execution_error_publication_verified") is not True
        or record.get("strict_loader_verified") is not True
        or record.get("signed_outer_to_unsigned_core_verified") is not True
        or record.get(
            "closure_failure_precedes_D_R_loader_verified"
        )
        is not True
        or record.get(
            "authorization_failure_precedes_D_R_loader_verified"
        )
        is not True
        or record.get(
            "authorization_verified_before_first_D_R_loader"
        )
        is not True
        or record.get(
            "D_R_reconstruction_failure_publication_verified"
        )
        is not True
        or record.get("D_R_payload_accessed") is not False
    ):
        return False
    return True


def _load_frozen_dry_evidence() -> tuple[
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
]:
    dry_config, dry_config_path = _load_exact_signed(
        path_text=DRY_CONFIG_REPO_PATH,
        expected_sha256=DRY_CONFIG_FILE_SHA256,
        name="CR-LVEC bounded dry-run config",
        fingerprint_field="config_fingerprint",
    )
    dry_result, dry_result_path = _load_exact_signed(
        path_text=DRY_RESULT_REPO_PATH,
        expected_sha256=DRY_RESULT_FILE_SHA256,
        name="CR-LVEC bounded dry-run result",
        fingerprint_field="result_fingerprint",
    )
    if (
        dry_config.get("schema_version")
        != "cure-lite-cr-lvec-v7-bounded-dry-run-config-v1"
        or dry_config.get("method_id")
        != CROSSING_FACTORIZED_OUTCOME_METHOD_ID
        or dry_config.get("config_fingerprint")
        != DRY_CONFIG_FINGERPRINT
        or dry_config.get("data_contract", {}).get(
            "D_R_dataset_or_cached_tensor_payload_access_allowed"
        )
        is not False
        or dry_result.get("schema_version")
        != "cure-lite-cr-lvec-v7-bounded-dry-run-result-v1"
        or dry_result.get("method_id")
        != CROSSING_FACTORIZED_OUTCOME_METHOD_ID
        or dry_result.get("result_fingerprint")
        != DRY_RESULT_FINGERPRINT
        or dry_result.get("decision")
        != "CR_LVEC_V7_BOUNDED_DRY_RUN_PASS"
        or dry_result.get("all_pass") is not True
        or dry_result.get("real_catalog_loader_call_count") != 0
        or dry_result.get("real_loader_imported_by_dry_entrypoint")
        is not False
        or dry_result.get("real_loader_symbol_reachable_from_dry_execution")
        is not False
        or dry_result.get("D_R_payload_accessed") is not False
        or dry_result.get("D_V_accessed") is not False
        or dry_result.get("D_T_accessed") is not False
        or dry_result.get("artifact_roundtrip_audit", {}).get(
            "covers_real_runner_publication"
        )
        is not False
        or dry_result.get("artifact_roundtrip_audit", {}).get(
            "scope"
        )
        != "canonical_json_serialization_probe_only"
    ):
        raise RuntimeError("CR-LVEC dry-run evidence contract changed")
    return (
        dry_config,
        dry_config_path,
        dry_result,
        dry_result_path,
    )


def _load_implementation_closure(
    config: Mapping[str, Any],
    implementation_unsigned: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    path = _repo_file(
        config["implementation_closure_contract"]["repo_path"],
        name="CR-LVEC bounded implementation closure",
    )
    closure = _strict_json(
        path,
        name="CR-LVEC bounded implementation closure",
    )
    _verify_fingerprinted(
        closure,
        name="CR-LVEC bounded implementation closure",
    )
    runtime_signed = closure.get("runtime_implementation_binding")
    if not isinstance(runtime_signed, Mapping):
        raise RuntimeError(
            "CR-LVEC closure runtime receipt is missing"
        )
    runtime_unsigned = _verified_unsigned_receipt(
        runtime_signed,
        name="CR-LVEC closure runtime implementation",
    )
    _verify_implementation_files(runtime_unsigned)
    _, dry_config_path, _, dry_result_path = _load_frozen_dry_evidence()
    wrapper_path = _repo_file(
        TEMPERATURE_WRAPPER_REPO_PATH,
        name="CR-LVEC GPU temperature wrapper",
    )
    sync_result, sync_path = _load_exact_signed(
        path_text=SYNC_BENCHMARK_REPO_PATH,
        expected_sha256=SYNC_BENCHMARK_FILE_SHA256,
        name="CR-LVEC sync benchmark",
        fingerprint_field="result_fingerprint",
    )

    protocol = closure.get("protocol_bindings")
    dry = closure.get("dry_run_result_binding")
    gate = closure.get("gate_summary")
    dependency = closure.get("dependency_audit")
    tests = closure.get("test_evidence")
    closure_static_test = (
        tests.get("closure_static_test")
        if isinstance(tests, Mapping)
        else None
    )
    temperature = closure.get("gpu_temperature_control_evidence")
    sync = closure.get("sync_benchmark_binding")
    boundary = closure.get("boundary")
    eligibility = closure.get("authorization_eligibility")
    if (
        closure.get("schema_version") != IMPLEMENTATION_CLOSURE_SCHEMA
        or closure.get("method_id") != CROSSING_FACTORIZED_OUTCOME_METHOD_ID
        or closure.get("phase_status")
        != "FROZEN_BOUNDED_IMPLEMENTATION_PASS"
        or closure.get("decision")
        != "CR_LVEC_V7_BOUNDED_IMPLEMENTATION_GATE_PASS"
        or not isinstance(protocol, Mapping)
        or not _binding_matches(
            protocol.get("bounded_implementation_proposal"),
            repo_path=IMPLEMENTATION_PROPOSAL_REPO_PATH,
            file_sha256_value=IMPLEMENTATION_PROPOSAL_FILE_SHA256,
            fingerprint=IMPLEMENTATION_PROPOSAL_FINGERPRINT,
        )
        or not _binding_matches(
            protocol.get("bounded_config"),
            repo_path=CONFIG_REPO_PATH,
            file_sha256_value=CONFIG_FILE_SHA256,
            fingerprint=CONFIG_FINGERPRINT,
        )
        or not _binding_matches(
            protocol.get("toy_gate_closure"),
            repo_path=TOY_CLOSURE_REPO_PATH,
            file_sha256_value=TOY_CLOSURE_FILE_SHA256,
            fingerprint=TOY_CLOSURE_FINGERPRINT,
        )
        or not _binding_matches(
            protocol.get("bounded_dry_run_config"),
            repo_path=DRY_CONFIG_REPO_PATH,
            file_sha256_value=file_sha256(dry_config_path),
            fingerprint=DRY_CONFIG_FINGERPRINT,
        )
        or not _binding_matches(
            protocol.get("bounded_dry_run_result"),
            repo_path=DRY_RESULT_REPO_PATH,
            file_sha256_value=file_sha256(dry_result_path),
            fingerprint=DRY_RESULT_FINGERPRINT,
        )
        or runtime_unsigned != dict(implementation_unsigned)
        or not isinstance(dry, Mapping)
        or not _binding_matches(
            dry,
            repo_path=DRY_RESULT_REPO_PATH,
            file_sha256_value=DRY_RESULT_FILE_SHA256,
            fingerprint=DRY_RESULT_FINGERPRINT,
        )
        or dry.get("process_replay_count") != 2
        or dry.get("byte_identical") is not True
        or dry.get("D_R_payload_accessed") is not False
        or dry.get("real_catalog_loader_call_count") != 0
        or dry.get("real_runner_publication_covered") is not False
        or not isinstance(dependency, Mapping)
        or dependency.get("v4_runtime_file_count") != 45
        or dependency.get("v7_runtime_file_count") != 5
        or dependency.get("all_runtime_file_count") != 50
        or dependency.get("all_runtime_hashes_verified") is not True
        or dependency.get("v4_runtime_fingerprint")
        != V4_IMPLEMENTATION_RECEIPT_FINGERPRINT
        or not isinstance(tests, Mapping)
        or not _test_record_passes(
            tests.get("focused_tests"),
            required_test_paths=_FOCUSED_TEST_REPO_PATHS,
        )
        or not _test_record_passes(
            tests.get("full_regression"),
            require_full_inventory=True,
        )
        or not _test_record_passes(
            tests.get("real_runner_publication_tests"),
            require_three_publication_outcomes=True,
            required_test_paths=(_REAL_RUNNER_TEST_REPO_PATH,),
        )
        or not _test_record_passes(
            tests.get("gpu_temperature_wrapper_tests"),
            required_test_paths=(_TEMPERATURE_TEST_REPO_PATH,),
        )
        or not isinstance(closure_static_test, Mapping)
        or closure_static_test.get("repo_path")
        != _CLOSURE_STATIC_TEST_REPO_PATH
        or closure_static_test.get("file_sha256")
        != file_sha256(_ROOT / _CLOSURE_STATIC_TEST_REPO_PATH)
        or closure_static_test.get("excluded_from_pre_signing_run")
        is not True
        or closure_static_test.get("post_signing_execution_required")
        is not True
        or closure_static_test.get("D_R_payload_access_allowed")
        is not False
        or not isinstance(temperature, Mapping)
        or temperature.get("wrapper_repo_path")
        != TEMPERATURE_WRAPPER_REPO_PATH
        or temperature.get("wrapper_file_sha256")
        != TEMPERATURE_WRAPPER_FILE_SHA256
        or file_sha256(wrapper_path) != TEMPERATURE_WRAPPER_FILE_SHA256
        or temperature.get("gpu_index") != 0
        or temperature.get("pause_temperature_celsius") != 82
        or temperature.get("resume_temperature_celsius") != 75
        or temperature.get("tests_passed") is not True
        or not _binding_matches(
            sync,
            repo_path=SYNC_BENCHMARK_REPO_PATH,
            file_sha256_value=file_sha256(sync_path),
            fingerprint=SYNC_BENCHMARK_FINGERPRINT,
        )
        or sync.get("bounded_400_policy") != "retain_strict_current_operator"
        or sync.get("formal_800_authorized") is not False
        or sync_result.get("schema_version")
        != "cure-lite-cr-lvec-v7-sync-benchmark-v2"
        or sync_result.get("result_fingerprint")
        != SYNC_BENCHMARK_FINGERPRINT
        or sync_result.get("operator_boundary_audit", {}).get(
            "current_contract_preserved"
        )
        is not True
        or sync_result.get("operator_boundary_audit", {}).get(
            "unchecked_is_diagnostic_only"
        )
        is not True
        or sync_result.get("scope", {}).get("D_R_accessed") is not False
        or sync_result.get("scope", {}).get("D_V_accessed") is not False
        or sync_result.get("scope", {}).get("D_T_accessed") is not False
        or sync_result.get("scope", {}).get("production_decoder_modified")
        is not False
        or not isinstance(gate, Mapping)
        or gate.get("all_required_checks_pass") is not True
        or gate.get("real_runner_three_outcomes_verified") is not True
        or gate.get("signed_outer_to_unsigned_core_verified") is not True
        or gate.get(
            "D_R_reconstruction_failure_publication_verified"
        )
        is not True
        or not isinstance(boundary, Mapping)
        or boundary.get("D_R_payload_accessed") is not False
        or boundary.get("real_D_R_bounded_authorized") is not False
        or boundary.get("D_V_accessed") is not False
        or boundary.get("D_T_accessed") is not False
        or boundary.get("formal_800_authorized") is not False
        or not isinstance(eligibility, Mapping)
        or eligibility.get("single_real_D_R_run_eligible") is not True
        or eligibility.get("directly_authorizes_real_D_R_run") is not False
        or eligibility.get("formal_800_authorized") is not False
    ):
        raise RuntimeError(
            "CR-LVEC bounded implementation closure changed"
        )
    return closure, path, dict(runtime_signed)


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
        "tools/run_crossing_factorized_outcome_bounded.py",
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


def _validate_authorization_payload(
    receipt: Mapping[str, Any],
    *,
    closure: Mapping[str, Any],
    closure_path: Path,
    implementation_unsigned: Mapping[str, Any],
) -> None:
    _verify_fingerprinted(
        receipt,
        name="CR-LVEC bounded run authorization",
    )
    authorization = receipt.get("authorization")
    config_binding = receipt.get("bounded_config_binding")
    closure_binding = receipt.get("implementation_closure_binding")
    runtime = receipt.get("runtime_implementation_binding")
    control = receipt.get("execution_control_binding")
    post_signing = receipt.get("post_signing_closure_test_evidence")
    if (
        receipt.get("schema_version") != AUTHORIZATION_SCHEMA
        or receipt.get("method_id") != CROSSING_FACTORIZED_OUTCOME_METHOD_ID
        or receipt.get("split") != "D_R"
        or receipt.get("phase_status")
        != "FROZEN_SINGLE_REAL_D_R_RUN_AUTHORIZATION"
        or receipt.get("decision")
        != "CR_LVEC_V7_ONE_REAL_D_R_BOUNDED_RUN_AUTHORIZED"
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
        or control.get("wrapped_command")
        != _expected_temperature_command()
        or not isinstance(post_signing, Mapping)
        or post_signing.get("evidence_stage") != "post_signing"
        or post_signing.get("closure_receipt_present_during_execution")
        is not True
        or post_signing.get("repo_path")
        != _CLOSURE_STATIC_TEST_REPO_PATH
        or post_signing.get("file_sha256")
        != file_sha256(_ROOT / _CLOSURE_STATIC_TEST_REPO_PATH)
        or post_signing.get("command")
        != _expected_post_signing_closure_test_command()
        or post_signing.get("exit_code") != 0
        or post_signing.get("passed_count") != 1
        or post_signing.get("failed_count") != 0
        or post_signing.get("skipped_count") != 0
        or post_signing.get("deselected_count") != 0
        or post_signing.get("selected_count") != 1
        or post_signing.get("collected_count") != 1
        or post_signing.get("closure_repo_path")
        != IMPLEMENTATION_CLOSURE_REPO_PATH
        or post_signing.get("closure_file_sha256")
        != file_sha256(closure_path)
        or post_signing.get("closure_receipt_fingerprint")
        != closure.get("receipt_fingerprint")
        or post_signing.get("D_R_payload_accessed") is not False
        or post_signing.get("D_V_accessed") is not False
        or post_signing.get("D_T_accessed") is not False
    ):
        raise RuntimeError(
            "CR-LVEC bounded run authorization changed"
        )


def _load_authorization(
    config: Mapping[str, Any],
    closure: Mapping[str, Any],
    closure_path: Path,
    implementation_unsigned: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    path = _repo_file(
        config["future_pre_run_authorization_contract"][
            "future_repo_path"
        ],
        name="CR-LVEC bounded run authorization",
    )
    receipt = _strict_json(
        path,
        name="CR-LVEC bounded run authorization",
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
        "outcome_pairs_per_update": budget[
            "outcome_pairs_per_update"
        ],
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
    """First and only function allowed to invoke the real D_R catalog loader."""

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
        raise RuntimeError("frozen D_R source config changed")

    pair_catalog, prepared, bundle, immutable = (
        v3_runner.legacy_runner._load_real_catalog(source_config)
    )
    if (
        pair_catalog.catalog_fingerprint != PAIR_CATALOG_FINGERPRINT
        or pair_catalog.split != "D_R"
        or len(pair_catalog.clean_positive) != 206
        or len(pair_catalog.component_null) != 16
    ):
        raise RuntimeError("frozen CR-LVEC outcome catalog changed")
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
        or materializer_receipt.get(
            "all_outcome_pair_input_fingerprint"
        )
        != ALL_PAIR_INPUTS_FINGERPRINT
        or materializer_receipt.get(
            "gt_union_population_fingerprint"
        )
        != GT_UNION_POPULATION_FINGERPRINT
        or factual_schedule.schedule_fingerprint
        != FACTUAL_SCHEDULE_FINGERPRINT
        or outcome_schedule.schedule_fingerprint
        != OUTCOME_SCHEDULE_FINGERPRINT
        or outcome_schedule.sequence_fingerprint
        != OUTCOME_SEQUENCE_FINGERPRINT
    ):
        raise RuntimeError("CR-LVEC frozen D_R inputs changed")
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
    fingerprint = unsigned.pop(field, None)
    if (
        not isinstance(fingerprint, str)
        or stable_fingerprint(unsigned) != fingerprint
    ):
        raise RuntimeError(f"CR-LVEC {name} internal fingerprint changed")


_POPULATION_STRUCTURAL_CHECKS = {
    "zero_feature_occupancy_delta_exact_zero",
    "raw_evidence_occupancy_invariant_exact",
    "baseline_occupancy_invariant_exact",
    "count_and_burden_change_support_exact",
    "count_support_outside_logit_delta_exact_zero",
    "count_support_outside_probability_delta_exact_zero",
    "count_change_support_nonempty",
    "independent_nonvacuous_locality_probe_passed",
    "all_audited_fields_finite",
    "local_count_deletion_monotonicity_exact",
    "occupancy_burden_deletion_monotonicity_exact",
    "deletion_logit_monotonicity_exact",
    "deletion_probability_monotonicity_exact",
    "native_subpixel_path_without_resize",
    "all_clean_D_pixels_in_count_change_support",
    "all_clean_pairs_have_nonempty_H",
    "all_component_null_pairs_have_positive_count_support",
    "all_factual_targets_have_finite_nonzero_recovery",
    "structural_audit_decoder_budget_exact",
}
_FULL_STRUCTURAL_CHECKS = {
    "deterministic_runtime_contract_satisfied",
    "CR_LVEC_pretraining_structural_audit_passed",
    *CROSSING_OPERATOR_STRUCTURAL_CHECKS,
    "factual_anchor_and_identity_counts_exact",
    "all_222_outcome_pairs_bound",
    "all_222_outcome_pairs_evaluated_initial",
    "all_222_outcome_pairs_evaluated_final",
    "all_optimizer_updates_completed",
    "one_backward_per_update",
    "one_optimizer_step_per_update",
    "all_gradients_finite",
    "every_update_total_gradient_norm_positive",
    "decoder_parameters_changed",
    "training_forward_budget_exact",
    "evaluation_forward_budget_exact",
    "total_forward_budget_exact",
    "margin_observation_budget_exact",
    "all_observed_crossing_margins_finite",
    "pair_exposure_ledger_exact",
    "source_exposure_ledger_exact",
    "factual_exposure_ledgers_exact",
    "identity_null_excluded_from_optimizer",
    "identity_null_diagnosed_without_autograd",
}


def _finite_number(value: object, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
        and (not positive or float(value) > 0.0)
    )


def _verify_pretraining_audit(
    audit: object,
) -> bool:
    if not isinstance(audit, Mapping):
        raise RuntimeError("CR-LVEC pretraining structural audit is missing")
    checks = audit.get("checks")
    operator = audit.get("operator_contract")
    operator_checks = (
        operator.get("checks") if isinstance(operator, Mapping) else None
    )
    budget = audit.get("compute_budget")
    locality = audit.get("independent_nonvacuous_locality_probe")
    records = audit.get("per_pair")
    if (
        not isinstance(checks, Mapping)
        or set(checks)
        != _POPULATION_STRUCTURAL_CHECKS
        | set(CROSSING_OPERATOR_STRUCTURAL_CHECKS)
        or not all(isinstance(value, bool) for value in checks.values())
        or not isinstance(operator, Mapping)
        or not isinstance(operator_checks, Mapping)
        or set(operator_checks) != set(CROSSING_OPERATOR_STRUCTURAL_CHECKS)
        or not all(
            isinstance(value, bool) for value in operator_checks.values()
        )
        or operator.get("all_pass") is not all(operator_checks.values())
        or operator.get("autograd_backward_calls") != 2
        or operator.get("training_performed") is not False
        or operator.get("D_R_accessed") is not False
        or operator.get("D_V_accessed") is not False
        or operator.get("D_T_accessed") is not False
        or audit.get("all_pass") is not all(checks.values())
        or audit.get("pair_count") != 222
        or audit.get("clean_pair_count") != 206
        or audit.get("component_null_pair_count") != 16
        or audit.get("clean_full_D_reachable_pairs") != 206
        or audit.get("clean_nonempty_H_pairs") != 206
        or audit.get("component_positive_count_support_pairs") != 16
        or audit.get("factual_full_target_recoverable_anchors") != 16
        or audit.get("factual_target_recoverable_pixels")
        != audit.get("factual_target_total_pixels")
        or audit.get("field_resize_endpoint_count") != 0
        or audit.get("count_burden_support_mismatch_pixels") != 0
        or audit.get("nonfinite_audited_field_values") != 0
        or audit.get("local_count_deletion_monotonicity_violations") != 0
        or audit.get(
            "occupancy_burden_deletion_monotonicity_violations"
        )
        != 0
        or audit.get("deletion_logit_monotonicity_violations") != 0
        or audit.get("deletion_probability_monotonicity_violations") != 0
        or not isinstance(budget, Mapping)
        or dict(budget)
        != {
            "decoder_calls": 31,
            "decoder_state_evaluations": 906,
            "expected_decoder_calls": 31,
            "expected_decoder_state_evaluations": 906,
            "factual_forward_fields_calls": 1,
            "factual_forward_fields_states": 16,
            "independent_locality_decoder_calls": 2,
            "independent_locality_decoder_state_evaluations": 2,
        }
        or not isinstance(locality, Mapping)
        or locality.get("all_pass") is not True
        or not _finite_number(
            locality.get("changed_support_pixels"),
            positive=True,
        )
        or not _finite_number(
            locality.get("unchanged_support_pixels"),
            positive=True,
        )
        or not isinstance(records, list)
        or len(records) != 222
        or len(
            {
                row.get("pair_id")
                for row in records
                if isinstance(row, Mapping)
            }
        )
        != 222
        or sum(
            row.get("pair_kind") == "clean_positive"
            for row in records
            if isinstance(row, Mapping)
        )
        != 206
        or sum(
            row.get("pair_kind") == "component_null"
            for row in records
            if isinstance(row, Mapping)
        )
        != 16
        or audit.get("training_performed") is not False
        or audit.get("D_V_accessed") is not False
        or audit.get("D_T_accessed") is not False
    ):
        raise RuntimeError("CR-LVEC structural audit contract changed")
    return all(checks.values())


def _verify_trace_and_exposure(
    trace: object,
    exposure: object,
) -> None:
    if (
        not isinstance(trace, list)
        or len(trace) != 400
        or not isinstance(exposure, Mapping)
    ):
        raise RuntimeError("CR-LVEC trace or exposure ledger is missing")
    pair_counts: Counter[str] = Counter()
    miss_counts: Counter[str] = Counter()
    no_miss_counts: Counter[str] = Counter()
    for update, row in enumerate(trace):
        if not isinstance(row, Mapping):
            raise RuntimeError("CR-LVEC trace row is malformed")
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
            raise RuntimeError("CR-LVEC exact update trace changed")
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
        raise RuntimeError("CR-LVEC exposure population changed")
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
            raise RuntimeError("CR-LVEC pair exposure row is malformed")
        sample_id = row.get("sample_id")
        count = row.get("count")
        if not isinstance(sample_id, str) or not isinstance(count, int):
            raise RuntimeError("CR-LVEC pair exposure row changed")
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
        raise RuntimeError("CR-LVEC exposure ledgers do not reproduce")


def _verify_core_result(result: Mapping[str, Any]) -> None:
    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint", None)
    structural = result.get("structural_execution_pass")
    model_pass = result.get("computational_model_code_gate_pass")
    interpretation = result.get("interpretation")
    audit = result.get("pretraining_structural_audit")
    audit_pass = _verify_pretraining_audit(audit)
    structural_checks = result.get("structural_checks")
    if (
        not isinstance(fingerprint, str)
        or stable_fingerprint(unsigned) != fingerprint
        or result.get("schema_version")
        != CROSSING_FACTORIZED_OUTCOME_BOUNDED_SCHEMA
        or result.get("method_id")
        != CROSSING_FACTORIZED_OUTCOME_METHOD_ID
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
        != vars(CrossingFactorizedDecoderConfig(64, 4))
        or result.get("loss_config")
        != {"dice_weight": 1.0, "epsilon": 1.0e-6}
        or result.get("optimization_budget")
        != {
            "seed": 42,
            "optimizer_updates": 400,
            "steps_per_epoch": 40,
            "factual_miss_states_per_update": 4,
            "factual_no_miss_states_per_update": 4,
            "outcome_pairs_per_update": 2,
            "learning_rate": 1.0e-3,
            "weight_decay": 0.0,
        }
        or result.get("evaluation_chunk_size") != 32
        or not isinstance(structural_checks, Mapping)
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
            "CR-LVEC bounded result violates its frozen boundary"
        )
    expected_decision = (
        "CR_LVEC_BOUNDED_MODEL_CODE_GATE_PASS"
        if model_pass
        else (
            "CR_LVEC_BOUNDED_MODEL_CODE_GATE_FAIL"
            if structural
            else "CR_LVEC_STRUCTURAL_EXECUTION_FAIL"
        )
    )
    if result.get("decision") != expected_decision:
        raise RuntimeError("CR-LVEC core decision is inconsistent")

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
            or not any(value is False for value in structural_checks.values())
            or result.get("forward_budget", {}).get("training")
            != {"calls": 0, "state_evaluations": 0}
            or result.get("forward_budget", {}).get(
                "pretraining_structural_audit"
            )
            != audit.get("compute_budget")
        ):
            raise RuntimeError(
                "CR-LVEC structural stop rule was not preserved"
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
    margin = result.get("margin_observation")
    expected_snapshot = {"calls": 10, "state_evaluations": 508}
    expected_training = {"calls": 1200, "state_evaluations": 4800}
    expected_total = {"calls": 1220, "state_evaluations": 5816}
    parameters_changed = (
        isinstance(parameters, Mapping)
        and parameters.get("initial_decoder_fingerprint")
        != parameters.get("final_decoder_fingerprint")
    )
    recomputed_structural_checks = {
        name: (
            parameters_changed
            if name == "decoder_parameters_changed"
            else True
        )
        for name in _FULL_STRUCTURAL_CHECKS
    }
    if (
        not isinstance(initial, Mapping)
        or not isinstance(final, Mapping)
        or not isinstance(computational, Mapping)
        or computational != crossing_computational_gates(initial, final)
        or len(computational.get("checks", {})) != 12
        or computational.get("all_pass") is not model_pass
        or set(structural_checks) != _FULL_STRUCTURAL_CHECKS
        or dict(structural_checks) != recomputed_structural_checks
        or structural is not all(structural_checks.values())
        or result.get("optimizer_updates_completed") != 400
        or not isinstance(parameters, Mapping)
        or parameters.get("trainable_parameter_count") != 4385
        or parameters.get("expected_parameter_count") != 4385
        or any(
            not isinstance(parameters.get(name), str)
            or len(str(parameters.get(name))) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(parameters.get(name))
            )
            for name in (
                "initial_decoder_fingerprint",
                "final_decoder_fingerprint",
            )
        )
        or not _finite_number(parameters.get("initial_l2_norm"), positive=True)
        or not _finite_number(parameters.get("final_l2_norm"), positive=True)
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
        or forward.get("pretraining_structural_audit_is_separate")
        is not True
        or forward.get("pretraining_structural_audit")
        != audit.get("compute_budget")
        or forward.get("initial_evaluation") != expected_snapshot
        or forward.get("training") != expected_training
        or forward.get("final_evaluation") != expected_snapshot
        or forward.get("total_excluding_structural_audit")
        != expected_total
        or forward.get("expected_initial_evaluation") != expected_snapshot
        or forward.get("expected_training") != expected_training
        or forward.get("expected_final_evaluation") != expected_snapshot
        or forward.get("expected_total_excluding_structural_audit")
        != expected_total
        or not isinstance(deterministic, Mapping)
        or deterministic.get("contract_satisfied") is not True
        or deterministic.get("flags_restored_after_execution") is not True
        or not isinstance(margin, Mapping)
        or margin.get("observed_forward_fields_calls") != 1251
        or margin.get("expected_forward_fields_calls") != 1251
        or margin.get("additional_decoder_forward_calls") != 0
        or margin.get("all_observed_margins_finite") is not True
        or not _finite_number(
            margin.get("maximum_observed_absolute_margin"),
            positive=True,
        )
    ):
        raise RuntimeError("CR-LVEC full bounded evidence changed")
    _verify_trace_and_exposure(result.get("trace"), result.get("exposure"))


def _decision(
    result: Mapping[str, object] | None,
    *,
    failure: Mapping[str, object] | None,
    evidence_receipt_fingerprint: str,
) -> dict[str, object]:
    if result is None:
        status = "STRUCTURAL_EXECUTION_ERROR"
        structural = False
        model_pass = False
    else:
        structural = result.get("structural_execution_pass") is True
        model_pass = (
            result.get("computational_model_code_gate_pass") is True
        )
        status = (
            "BOUNDED_MODEL_CODE_GATE_PASS"
            if model_pass
            else (
                "BOUNDED_MODEL_CODE_GATE_FAIL"
                if structural
                else "STRUCTURAL_EXECUTION_FAIL"
            )
        )
    return _fingerprinted(
        {
            "schema_version": DECISION_SCHEMA,
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
                "freeze_and_review_bounded_model_code_evidence"
                if model_pass
                else "preserve_failure_and_stop_v7_without_retry"
            ),
        }
    )


@dataclass(frozen=True)
class PublishedCrossingFactorizedOutcomeBounded:
    root: Path
    decision: str
    structural_execution_pass: bool
    bounded_model_code_gate_pass: bool
    complete_fingerprint: str

    def verify_unchanged(self) -> None:
        if load_crossing_factorized_outcome_bounded_artifact(
            self.root
        ) != self:
            raise RuntimeError("published CR-LVEC artifact changed")


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
        name="published CR-LVEC D_R source config",
    )
    source_config = v3_runner.legacy_runner._load_config(source_path)
    if (
        source.get("schema_version")
        != "cure-lite-cr-lvec-v7-source-reconstruction-v1"
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
        raise RuntimeError("published CR-LVEC D_R input binding changed")
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


def load_crossing_factorized_outcome_bounded_artifact(
    output_dir: str | Path,
    *,
    _allow_incomplete: bool = False,
) -> PublishedCrossingFactorizedOutcomeBounded:
    candidate = Path(output_dir).expanduser()
    absolute = Path(os.path.abspath(candidate))
    if candidate.is_symlink():
        raise ValueError("CR-LVEC bounded root may not be a symlink")
    root = candidate.resolve(strict=True)
    if root != absolute or not root.is_dir() or root.is_symlink():
        raise ValueError("CR-LVEC bounded root must be a regular directory")
    incomplete = (root / _INCOMPLETE).exists()
    if incomplete and not _allow_incomplete:
        raise RuntimeError("CR-LVEC bounded publication is incomplete")
    expected_top = {"receipts", "COMPLETE.json"}
    if _allow_incomplete:
        expected_top.add(_INCOMPLETE)
    if {item.name for item in root.iterdir()} != expected_top:
        raise RuntimeError("CR-LVEC top-level inventory changed")

    receipts_root = root / "receipts"
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
        raise RuntimeError("CR-LVEC receipt inventory changed")
    if any(
        item.is_symlink() or not item.is_file()
        for item in receipts_root.iterdir()
    ):
        raise RuntimeError("CR-LVEC receipts must be regular files")

    complete = _strict_json(root / "COMPLETE.json", name="CR-LVEC COMPLETE")
    _verify_fingerprinted(
        complete,
        name="CR-LVEC COMPLETE",
        field="complete_fingerprint",
    )
    payloads = {
        name[:-5]: _strict_json(
            receipts_root / name,
            name=f"CR-LVEC {name[:-5]}",
        )
        for name in names
    }
    for name, payload in payloads.items():
        _verify_fingerprinted(payload, name=f"CR-LVEC {name}")
    if (
        complete.get("artifact_files") != _artifact_hashes(root)
        or complete.get("artifact_file_count") != len(names)
        or complete.get("schema_version") != RUN_SCHEMA
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
    ):
        raise RuntimeError("CR-LVEC COMPLETE boundary changed")

    config_receipt = payloads["config_binding"]
    embedded_config = config_receipt.get("config")
    if not isinstance(embedded_config, Mapping):
        raise RuntimeError("embedded CR-LVEC config is malformed")
    _validate_config_payload(embedded_config)
    current_config_path = _repo_file(
        config_receipt.get("repo_path"),
        name="published CR-LVEC config",
    )
    current_config = _load_config(current_config_path)
    if (
        config_receipt.get("schema_version")
        != "cure-lite-cr-lvec-v7-config-binding-v1"
        or config_receipt.get("repo_path") != CONFIG_REPO_PATH
        or config_receipt.get("file_sha256") != CONFIG_FILE_SHA256
        or config_receipt.get("config_fingerprint")
        != CONFIG_FINGERPRINT
        or embedded_config != current_config
    ):
        raise RuntimeError("published CR-LVEC config binding changed")

    proposal_receipt = payloads["proposal_binding"]
    toy_receipt = payloads["toy_gate_binding"]
    implementation_proposal_receipt = payloads[
        "implementation_proposal_binding"
    ]
    embedded_proposal = proposal_receipt.get("proposal")
    embedded_toy = toy_receipt.get("closure")
    embedded_implementation_proposal = (
        implementation_proposal_receipt.get("proposal")
    )
    if (
        not isinstance(embedded_proposal, Mapping)
        or not isinstance(embedded_toy, Mapping)
        or not isinstance(embedded_implementation_proposal, Mapping)
    ):
        raise RuntimeError("published CR-LVEC protocol receipt is malformed")
    current_proposal, current_proposal_path = _load_proposal(
        embedded_config
    )
    current_toy, current_toy_path = _load_toy_closure(embedded_config)
    current_implementation_proposal, current_ip_path = (
        _load_implementation_proposal(embedded_config)
    )
    if (
        proposal_receipt.get("schema_version")
        != "cure-lite-cr-lvec-v7-proposal-binding-v1"
        or proposal_receipt.get("repo_path") != PROPOSAL_REPO_PATH
        or proposal_receipt.get("file_sha256")
        != file_sha256(current_proposal_path)
        or proposal_receipt.get("proposal_fingerprint")
        != PROPOSAL_FINGERPRINT
        or embedded_proposal != current_proposal
        or toy_receipt.get("schema_version")
        != "cure-lite-cr-lvec-v7-toy-binding-v1"
        or toy_receipt.get("repo_path") != TOY_CLOSURE_REPO_PATH
        or toy_receipt.get("file_sha256") != file_sha256(current_toy_path)
        or toy_receipt.get("closure_fingerprint")
        != TOY_CLOSURE_FINGERPRINT
        or embedded_toy != current_toy
        or implementation_proposal_receipt.get("schema_version")
        != "cure-lite-cr-lvec-v7-implementation-proposal-binding-v1"
        or implementation_proposal_receipt.get("repo_path")
        != IMPLEMENTATION_PROPOSAL_REPO_PATH
        or implementation_proposal_receipt.get("file_sha256")
        != file_sha256(current_ip_path)
        or implementation_proposal_receipt.get("proposal_fingerprint")
        != IMPLEMENTATION_PROPOSAL_FINGERPRINT
        or embedded_implementation_proposal
        != current_implementation_proposal
    ):
        raise RuntimeError("published CR-LVEC protocol binding changed")

    implementation_signed = payloads["implementation_binding"]
    implementation_unsigned = _verified_unsigned_receipt(
        implementation_signed,
        name="published CR-LVEC runtime implementation",
    )
    _verify_implementation_files(implementation_unsigned)

    claim = payloads["run_claim"]
    claim_unsigned = dict(claim)
    claim_unsigned.pop("receipt_fingerprint")
    if claim_unsigned != {
        "schema_version": "cure-lite-cr-lvec-v7-single-run-claim-v1",
        "method_id": CROSSING_FACTORIZED_OUTCOME_METHOD_ID,
        "split": "D_R",
        "device": FROZEN_DEVICE,
        "real_D_R_run_count_claimed": 1,
        "claim_consumed_before_first_D_R_payload_loader": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }:
        raise RuntimeError("published CR-LVEC run claim changed")

    closure_receipt = payloads["implementation_closure_binding"]
    closure = closure_receipt.get("closure")
    if not isinstance(closure, Mapping):
        raise RuntimeError("embedded CR-LVEC closure is malformed")
    current_closure, current_closure_path, runtime_signed = (
        _load_implementation_closure(
            embedded_config,
            implementation_unsigned,
        )
    )
    if (
        closure != current_closure
        or closure_receipt.get("repo_path")
        != IMPLEMENTATION_CLOSURE_REPO_PATH
        or closure_receipt.get("file_sha256")
        != file_sha256(current_closure_path)
        or closure_receipt.get("closure_fingerprint")
        != current_closure.get("receipt_fingerprint")
        or dict(runtime_signed) != dict(implementation_signed)
    ):
        raise RuntimeError("published CR-LVEC closure binding changed")

    authorization_receipt = payloads["authorization_binding"]
    authorization = authorization_receipt.get("authorization")
    if not isinstance(authorization, Mapping):
        raise RuntimeError("embedded CR-LVEC authorization is malformed")
    current_authorization, authorization_path = _load_authorization(
        embedded_config,
        current_closure,
        current_closure_path,
        implementation_unsigned,
    )
    if (
        authorization != current_authorization
        or authorization_receipt.get("repo_path")
        != AUTHORIZATION_REPO_PATH
        or authorization_receipt.get("file_sha256")
        != file_sha256(authorization_path)
        or authorization_receipt.get("authorization_fingerprint")
        != current_authorization.get("receipt_fingerprint")
    ):
        raise RuntimeError("published CR-LVEC authorization changed")

    input_binding = (
        None
        if not _INPUT_RECEIPTS.issubset(names)
        else _verify_published_input_receipts(payloads)
    )
    expected_input_complete = (
        {
            "pair_catalog_fingerprint": None,
            "prepared_catalog_fingerprint": None,
            "population_fingerprint": None,
            "factual_schedule_fingerprint": None,
            "materializer_fingerprint": None,
            "outcome_schedule_fingerprint": None,
            "all_pair_inputs_fingerprint": None,
            "gt_union_population_fingerprint": None,
            "outcome_sequence_fingerprint": None,
        }
        if input_binding is None
        else {
            "pair_catalog_fingerprint": PAIR_CATALOG_FINGERPRINT,
            "prepared_catalog_fingerprint": PREPARED_CATALOG_FINGERPRINT,
            **input_binding,
        }
    )
    if (
        complete.get("input_receipts_present")
        is not (input_binding is not None)
        or complete.get("config_fingerprint") != CONFIG_FINGERPRINT
        or complete.get("proposal_fingerprint") != PROPOSAL_FINGERPRINT
        or complete.get("toy_closure_fingerprint")
        != TOY_CLOSURE_FINGERPRINT
        or complete.get("implementation_proposal_fingerprint")
        != IMPLEMENTATION_PROPOSAL_FINGERPRINT
        or complete.get("implementation_closure_fingerprint")
        != current_closure.get("receipt_fingerprint")
        or complete.get("authorization_fingerprint")
        != current_authorization.get("receipt_fingerprint")
        or any(
            complete.get(key) != value
            for key, value in expected_input_complete.items()
        )
    ):
        raise RuntimeError("CR-LVEC COMPLETE cross-binding changed")
    decision = payloads["decision"]
    evidence_kind = "result" if "result" in payloads else "failure"
    evidence = payloads[evidence_kind]
    if evidence_kind == "result":
        core = dict(evidence)
        core.pop("receipt_fingerprint")
        _verify_core_result(core)
        if (
            input_binding is None
            or core.get("population_fingerprint")
            != input_binding["population_fingerprint"]
            or core.get("factual_schedule_fingerprint")
            != input_binding["factual_schedule_fingerprint"]
            or core.get("materializer_fingerprint")
            != input_binding["materializer_fingerprint"]
            or core.get("outcome_schedule_fingerprint")
            != input_binding["outcome_schedule_fingerprint"]
        ):
            raise RuntimeError(
                "CR-LVEC result does not bind its published D_R inputs"
            )
        structural = core.get("structural_execution_pass") is True
        model_pass = (
            core.get("computational_model_code_gate_pass") is True
        )
        expected_status = (
            "BOUNDED_MODEL_CODE_GATE_PASS"
            if model_pass
            else (
                "BOUNDED_MODEL_CODE_GATE_FAIL"
                if structural
                else "STRUCTURAL_EXECUTION_FAIL"
            )
        )
    else:
        structural = False
        model_pass = False
        expected_status = "STRUCTURAL_EXECUTION_ERROR"
        failure_unsigned = dict(evidence)
        failure_unsigned.pop("receipt_fingerprint")
        post_passed = failure_unsigned.get(
            "post_attempt_verification_passed"
        )
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
            or not failure_unsigned.get("exception_type")
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
            or not isinstance(post_passed, bool)
            or (
                post_passed
                and (
                    failure_unsigned.get("post_attempt_exception_type")
                    is not None
                    or failure_unsigned.get(
                        "post_attempt_exception_message"
                    )
                    is not None
                )
            )
            or (
                not post_passed
                and (
                    not isinstance(
                        failure_unsigned.get(
                            "post_attempt_exception_type"
                        ),
                        str,
                    )
                    or not isinstance(
                        failure_unsigned.get(
                            "post_attempt_exception_message"
                        ),
                        str,
                    )
                )
            )
        ):
            raise RuntimeError("CR-LVEC failure receipt changed")
    if (
        decision.get("schema_version") != DECISION_SCHEMA
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
        or decision.get("failure")
        != (
            None
            if evidence_kind == "result"
            else failure_unsigned
        )
        or decision.get("next_action")
        != (
            "freeze_and_review_bounded_model_code_evidence"
            if model_pass
            else "preserve_failure_and_stop_v7_without_retry"
        )
        or complete.get("decision") != expected_status
        or complete.get("decision_fingerprint")
        != decision.get("receipt_fingerprint")
        or complete.get("implementation_receipt_fingerprint")
        != implementation_signed.get("receipt_fingerprint")
    ):
        raise RuntimeError("CR-LVEC decision binding changed")
    return PublishedCrossingFactorizedOutcomeBounded(
        root=root,
        decision=str(expected_status),
        structural_execution_pass=structural,
        bounded_model_code_gate_pass=model_pass,
        complete_fingerprint=str(complete["complete_fingerprint"]),
    )


def _pre_run_receipts(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    proposal: Mapping[str, Any],
    proposal_path: Path,
    toy: Mapping[str, Any],
    toy_path: Path,
    implementation_proposal: Mapping[str, Any],
    implementation_proposal_path: Path,
    closure: Mapping[str, Any],
    closure_path: Path,
    authorization: Mapping[str, Any],
    authorization_path: Path,
    implementation_signed: Mapping[str, Any],
) -> dict[str, dict[str, object]]:
    return {
        "config_binding.json": _fingerprinted(
            {
                "schema_version": "cure-lite-cr-lvec-v7-config-binding-v1",
                "repo_path": CONFIG_REPO_PATH,
                "file_sha256": file_sha256(config_path),
                "config_fingerprint": config["config_fingerprint"],
                "config": config,
            }
        ),
        "proposal_binding.json": _fingerprinted(
            {
                "schema_version": "cure-lite-cr-lvec-v7-proposal-binding-v1",
                "repo_path": PROPOSAL_REPO_PATH,
                "file_sha256": file_sha256(proposal_path),
                "proposal_fingerprint": proposal["proposal_fingerprint"],
                "proposal": proposal,
            }
        ),
        "toy_gate_binding.json": _fingerprinted(
            {
                "schema_version": "cure-lite-cr-lvec-v7-toy-binding-v1",
                "repo_path": TOY_CLOSURE_REPO_PATH,
                "file_sha256": file_sha256(toy_path),
                "closure_fingerprint": toy["receipt_fingerprint"],
                "closure": toy,
            }
        ),
        "implementation_proposal_binding.json": _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cr-lvec-v7-implementation-proposal-binding-v1"
                ),
                "repo_path": IMPLEMENTATION_PROPOSAL_REPO_PATH,
                "file_sha256": file_sha256(
                    implementation_proposal_path
                ),
                "proposal_fingerprint": implementation_proposal[
                    "receipt_fingerprint"
                ],
                "proposal": implementation_proposal,
            }
        ),
        "implementation_closure_binding.json": _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cr-lvec-v7-implementation-closure-binding-v1"
                ),
                "repo_path": IMPLEMENTATION_CLOSURE_REPO_PATH,
                "file_sha256": file_sha256(closure_path),
                "closure_fingerprint": closure["receipt_fingerprint"],
                "closure": closure,
            }
        ),
        "authorization_binding.json": _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cr-lvec-v7-authorization-binding-v1"
                ),
                "repo_path": AUTHORIZATION_REPO_PATH,
                "file_sha256": file_sha256(authorization_path),
                "authorization_fingerprint": authorization[
                    "receipt_fingerprint"
                ],
                "authorization": authorization,
            }
        ),
        "implementation_binding.json": dict(implementation_signed),
        "run_claim.json": _fingerprinted(
            {
                "schema_version": (
                    "cure-lite-cr-lvec-v7-single-run-claim-v1"
                ),
                "method_id": CROSSING_FACTORIZED_OUTCOME_METHOD_ID,
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


def run(args: argparse.Namespace) -> dict[str, object]:
    config_path = _canonical_file(
        args.config,
        name="CR-LVEC bounded config",
    )
    config = _load_config(config_path)
    device = _validate_device(args.device)
    output = _validate_output_target(args.output)

    proposal, proposal_path = _load_proposal(config)
    toy, toy_path = _load_toy_closure(config)
    implementation_proposal, implementation_proposal_path = (
        _load_implementation_proposal(config)
    )
    implementation_unsigned = _implementation_binding()
    _verify_implementation_files(implementation_unsigned)
    closure, closure_path, closure_runtime_signed = (
        _load_implementation_closure(
            config,
            implementation_unsigned,
        )
    )
    implementation_signed = _fingerprinted(
        implementation_unsigned
    )
    if implementation_signed != closure_runtime_signed:
        raise RuntimeError(
            "closure binds a different signed runtime implementation"
        )
    authorization, authorization_path = _load_authorization(
        config,
        closure,
        closure_path,
        implementation_unsigned,
    )

    # The exact-one-run claim occurs only after every static authorization
    # check and before the first possible D_R payload loader call.
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / _INCOMPLETE
    incomplete.open("xb").close()
    receipts = output / "receipts"
    receipts.mkdir(exist_ok=False)
    for name, payload in _pre_run_receipts(
        config=config,
        config_path=config_path,
        proposal=proposal,
        proposal_path=proposal_path,
        toy=toy,
        toy_path=toy_path,
        implementation_proposal=implementation_proposal,
        implementation_proposal_path=implementation_proposal_path,
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
                    "cure-lite-cr-lvec-v7-source-reconstruction-v1"
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
        result = execute_crossing_factorized_outcome_bounded(
            real_inputs.population,
            real_inputs.factual_schedule,
            real_inputs.outcome_schedule,
            real_inputs.materializer,
            CrossingFactorizedDecoderConfig(
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
                    "a frozen D_R input changed during execution"
                )
        if (
            _load_config(config_path) != config
            or _load_proposal(config)[0] != proposal
            or _load_toy_closure(config)[0] != toy
            or _load_implementation_proposal(config)[0]
            != implementation_proposal
            or _implementation_binding() != implementation_unsigned
        ):
            raise RuntimeError(
                "CR-LVEC static implementation inputs changed "
                "during execution"
            )
        current_closure, current_closure_path, current_runtime_signed = (
            _load_implementation_closure(
                config,
                implementation_unsigned,
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
                "CR-LVEC closure or authorization changed "
                "during execution"
            )
    except Exception as error:
        post_attempt_error = error
        if execution_error is None:
            execution_error = error
            failure_phase = "POST_EXECUTION_IMMUTABILITY"

    if execution_error is None:
        if result is None:
            raise RuntimeError("CR-LVEC execution returned no result")
        evidence = _fingerprinted(result)
        _write_new_json(receipts / "result.json", evidence)
        failure = None
    else:
        result = None
        failure = {
            "schema_version": FAILURE_SCHEMA,
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
            "proposal_fingerprint": PROPOSAL_FINGERPRINT,
            "toy_closure_fingerprint": TOY_CLOSURE_FINGERPRINT,
            "implementation_proposal_fingerprint": (
                IMPLEMENTATION_PROPOSAL_FINGERPRINT
            ),
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
    published = load_crossing_factorized_outcome_bounded_artifact(
        output,
        _allow_incomplete=True,
    )
    incomplete.unlink()
    return {
        "output": str(output),
        "decision": published.decision,
        "structural_execution_pass": published.structural_execution_pass,
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
