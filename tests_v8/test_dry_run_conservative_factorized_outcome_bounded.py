from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import pytest
import numpy as np
import torch

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from tools import dry_run_conservative_factorized_outcome_bounded as dry_runner


_ROOT = Path(__file__).resolve().parents[1]
_RUNNER = (
    _ROOT
    / "tools"
    / "dry_run_conservative_factorized_outcome_bounded.py"
)
_PROTOCOL = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "coverage_conserving_subpixel_evidence_allocation_v8"
)
_CONFIG = _PROTOCOL / "bounded_dry_run_config_v3.json"
_PROPOSAL = _PROTOCOL / "bounded_dry_run_proposal_receipt_v3.json"
_TOY_CLOSURE = _PROTOCOL / "toy_gate_closure_receipt.json"

_CONFIG_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "coverage_conserving_subpixel_evidence_allocation_v8/"
    "bounded_dry_run_config_v3.json"
)
_PROPOSAL_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "coverage_conserving_subpixel_evidence_allocation_v8/"
    "bounded_dry_run_proposal_receipt_v3.json"
)
_TOY_CLOSURE_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "coverage_conserving_subpixel_evidence_allocation_v8/"
    "toy_gate_closure_receipt.json"
)

_CONFIG_SHA256 = (
    "5187b2f5516fd33b3eba9ae74092ba10ce42d0a85ac9d22918cbefb322e835c6"
)
_CONFIG_FINGERPRINT = (
    "c985d1598d490c202397b0483cc2ac02abe98ed1ea9de26127a935386ac5b863"
)
_PROPOSAL_SHA256 = (
    "1a1fc75c23991373d584f91041f3af73319c1e5e539dc728bd9d8f4cc41b9949"
)
_PROPOSAL_FINGERPRINT = (
    "509584774a52dbaa585f8d0860c16baf630046e94c1305bb2e1ca384cf45d746"
)
_TOY_CLOSURE_SHA256 = (
    "63affcf21c59f0808b2fcc18e1fc6e1054fc781708fa521d202d2a9ac8b16b0d"
)
_TOY_CLOSURE_FINGERPRINT = (
    "be05a38ca53975f48f429e16d0df31b365a76bfb994c8a911e8ed636af4a2f67"
)

_EXPECTED_PARAMETER_NAMES = (
    "baseline_raw",
    "stem.weight",
    "depthwise.weight",
    "pointwise.weight",
    "baseline_head.weight",
    "evidence_head.weight",
)


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _verify_fingerprint(
    value: Mapping[str, object],
    *,
    field: str,
) -> str:
    unsigned = dict(value)
    observed = unsigned.pop(field)
    assert isinstance(observed, str)
    assert observed == stable_fingerprint(unsigned)
    return observed


def _run_cli(output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_RUNNER),
            "--output",
            str(output),
        ],
        cwd=_ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        },
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def cli_replays(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    directory = tmp_path_factory.mktemp("cc-sea-v8-dry-replays")
    first = directory / "first.json"
    second = directory / "second.json"
    first_process = _run_cli(first)
    second_process = _run_cli(second)
    assert first_process.returncode == 0, first_process.stderr
    assert second_process.returncode == 0, second_process.stderr
    return {
        "first_path": first,
        "second_path": second,
        "first_process": first_process,
        "second_process": second_process,
        "first_bytes": first.read_bytes(),
        "second_bytes": second.read_bytes(),
        "result": _load_object(first),
    }


def test_two_independent_cli_processes_are_byte_identical(
    cli_replays: Mapping[str, object],
) -> None:
    assert cli_replays["first_bytes"] == cli_replays["second_bytes"]
    result = cli_replays["result"]
    assert isinstance(result, dict)
    assert result["schema_version"] == (
        "cure-lite-cc-sea-v8-bounded-dry-run-result-v3"
    )
    assert result["method_id"] == "cc_sea_v8"
    assert result["decision"] == (
        "CC_SEA_V8_DRY_RUN_SINGLE_PROCESS_PASS"
    )
    assert result["all_pass"] is True
    assert result["single_process_gate_pass"] is True
    assert result["closure_eligible_after_replay"] is True
    assert result["closure_required_evidence_status"] == (
        "NOT_EVALUATED_BY_SINGLE_PROCESS"
    )
    assert result["real_D_R_bounded_code_creation_authorized"] is False

    unsigned = dict(result)
    result_fingerprint = unsigned.pop("result_fingerprint")
    assert result_fingerprint == stable_fingerprint(unsigned)

    for key in ("first_process", "second_process"):
        process = cli_replays[key]
        assert isinstance(process, subprocess.CompletedProcess)
        summary = json.loads(process.stdout)
        assert summary["decision"] == (
            "CC_SEA_V8_DRY_RUN_SINGLE_PROCESS_PASS"
        )
        assert summary["result_fingerprint"] == result_fingerprint


def test_protocol_chain_is_exact_and_independently_verifiable(
    cli_replays: Mapping[str, object],
) -> None:
    config = _load_object(_CONFIG)
    proposal = _load_object(_PROPOSAL)
    closure = _load_object(_TOY_CLOSURE)

    assert file_sha256(_CONFIG) == _CONFIG_SHA256
    assert file_sha256(_PROPOSAL) == _PROPOSAL_SHA256
    assert file_sha256(_TOY_CLOSURE) == _TOY_CLOSURE_SHA256
    assert _verify_fingerprint(
        config,
        field="config_fingerprint",
    ) == _CONFIG_FINGERPRINT
    assert _verify_fingerprint(
        proposal,
        field="proposal_fingerprint",
    ) == _PROPOSAL_FINGERPRINT
    assert _verify_fingerprint(
        closure,
        field="receipt_fingerprint",
    ) == _TOY_CLOSURE_FINGERPRINT

    assert config["schema_version"] == (
        "cure-lite-cc-sea-v8-bounded-dry-run-config-v3"
    )
    assert proposal["schema_version"] == (
        "cure-lite-cc-sea-v8-dry-run-proposal-v3"
    )
    assert closure["schema_version"] == (
        "cure-lite-cc-sea-v8-toy-gate-closure-v1"
    )
    assert proposal["toy_closure_binding"] == {
        "repo_path": _TOY_CLOSURE_REPO_PATH,
        "file_sha256": _TOY_CLOSURE_SHA256,
        "receipt_fingerprint": _TOY_CLOSURE_FINGERPRINT,
    }
    assert config["proposal_binding"] == {
        "repo_path": _PROPOSAL_REPO_PATH,
        "file_sha256": _PROPOSAL_SHA256,
        "proposal_fingerprint": _PROPOSAL_FINGERPRINT,
    }
    assert config["toy_closure_binding"] == proposal[
        "toy_closure_binding"
    ]
    assert config["single_process_required_evidence"] == proposal[
        "single_process_required_evidence"
    ]
    assert config["closure_required_evidence"] == proposal[
        "closure_required_evidence"
    ]
    assert config["decision_rule"] == {
        "all_single_process_checks_must_pass": True,
        "single_process_pass_decision": (
            "CC_SEA_V8_DRY_RUN_SINGLE_PROCESS_PASS"
        ),
        "single_process_fail_decision": (
            "CC_SEA_V8_DRY_RUN_SINGLE_PROCESS_FAIL"
        ),
        "closure_pass_decision": (
            "CC_SEA_V8_DRY_RUN_CLOSURE_PASS_AND_"
            "REAL_BOUNDED_CODE_AUTHORIZED"
        ),
        "single_process_pass_authorizes": "nothing",
        "closure_pass_authorizes": (
            "real_D_R_bounded_code_creation_only"
        ),
    }
    assert closure["decision"] == (
        "CC_SEA_V8_TOY_GATE_PASS_AND_DRY_RUN_CODE_AUTHORIZED"
    )

    result = cli_replays["result"]
    assert isinstance(result, dict)
    assert result["protocol_binding"] == {
        "config_repo_path": _CONFIG_REPO_PATH,
        "config_file_sha256": _CONFIG_SHA256,
        "config_fingerprint": _CONFIG_FINGERPRINT,
        "proposal_repo_path": _PROPOSAL_REPO_PATH,
        "proposal_file_sha256": _PROPOSAL_SHA256,
        "proposal_fingerprint": _PROPOSAL_FINGERPRINT,
        "toy_closure_repo_path": _TOY_CLOSURE_REPO_PATH,
        "toy_closure_file_sha256": _TOY_CLOSURE_SHA256,
        "toy_closure_fingerprint": _TOY_CLOSURE_FINGERPRINT,
        "toy_closure_bound_files_verified": 25,
    }


def test_exact_24_calls_96_states_and_all_eight_updates(
    cli_replays: Mapping[str, object],
) -> None:
    result = cli_replays["result"]
    assert isinstance(result, dict)
    runtime = result["runtime"]
    assert isinstance(runtime, dict)
    assert runtime == {
        "device": "cpu",
        "seed": 7817,
        "optimizer_updates": 8,
        "learning_rate": 0.001,
        "decoder_calls": 24,
        "decoder_states": 96,
        "observed_device_types": ["cpu"],
        "base_detector_instances": 0,
        "real_loader_calls": 0,
        "dataset_or_cache_payload_accesses": 0,
        "package_initialization_loader_modules_observed": [
            "cure_lite.cache.base_cache",
            "cure_lite.cache.state_cache",
            "cure_lite.data",
            "cure_lite.experiment.cache_pipeline",
            "cure_lite.experiment.training_pipeline",
        ],
        "package_initialization_imports_disclosed": True,
        "entrypoint_direct_import_audit": (
            "PENDING_CLOSURE_STATIC_TEST"
        ),
    }

    trace = result["trace"]
    assert isinstance(trace, list)
    assert len(trace) == 8
    assert [row["update"] for row in trace] == list(range(8))
    assert [row["epoch"] for row in trace] == [0] * 4 + [1] * 4
    assert [row["step"] for row in trace] == [0, 1, 2, 3] * 2
    assert sum(
        int(row["losses"]["decoder_forward_calls_per_update"])
        for row in trace
    ) == 24
    assert sum(
        int(row["losses"]["decoder_states_per_update"])
        for row in trace
    ) == 96
    assert all(
        row["losses"]["backward_calls"] == 1
        and row["losses"]["optimizer_steps"] == 1
        for row in trace
    )

    structural = result["structural_checks"]
    assert isinstance(structural, dict)
    assert all(structural.values())
    assert structural["exact_eight_updates"] is True
    assert structural["exact_24_training_forward_calls"] is True
    assert structural["exact_96_training_states"] is True
    assert structural[
        "every_update_execution_budget_and_feature_detachment"
    ] is True
    assert structural["dual_endpoint_gradients"] is True

    locality = result["state_equation_audit"]
    assert locality["all_pass"] is True
    assert locality["probe_kind"] == (
        "controlled_positive_budget_operator_probe"
    )
    assert locality["feature_grid"] == [5, 5]
    assert locality["evaluation_grid"] == [20, 20]
    assert locality["support_pixel_count"] == 144
    assert locality["outside_support_pixel_count"] == 256
    assert locality["support_max_abs_delta"] > 0.0
    assert locality["outside_count_support_max_abs_delta"] == 0.0
    assert locality[
        "actual_decoder_outside_count_support_max_abs_delta"
    ] == 0.0
    assert all(locality["checks"].values())


def test_each_update_has_six_finite_nonzero_gradients_and_updates(
    cli_replays: Mapping[str, object],
) -> None:
    result = cli_replays["result"]
    assert isinstance(result, dict)
    trace = result["trace"]
    assert isinstance(trace, list)
    for update, row in enumerate(trace):
        checks = row["parameter_checks"]
        assert checks == {
            "all_six_gradients_present_finite_nonzero": True,
            "all_six_parameters_updated": True,
        }
        records = row["parameter_records"]
        assert isinstance(records, list)
        assert len(records) == 6
        assert tuple(record["name"] for record in records) == (
            _EXPECTED_PARAMETER_NAMES
        )
        assert sum(int(record["numel"]) for record in records) == 2593
        for record in records:
            assert record["gradient_present"] is True
            assert record["gradient_finite"] is True
            assert int(record["gradient_nonzero_count"]) > 0
            assert float(record["gradient_l2"]) > 0.0
            assert float(record["gradient_max_abs"]) > 0.0
            assert int(record["parameter_delta_nonzero_count"]) > 0
            assert float(record["parameter_delta_l2"]) > 0.0
        assert row["update"] == update
        assert all(row["execution_checks"].values())
        assert len(row["decoder_input_fingerprint"]) == 64
        assert len(row["training_example_fingerprint"]) == 64

    assert len(
        {row["training_example_fingerprint"] for row in trace[:6]}
    ) == 6
    assert trace[0]["training_example_fingerprint"] == trace[6][
        "training_example_fingerprint"
    ]
    assert trace[3]["training_example_fingerprint"] == trace[7][
        "training_example_fingerprint"
    ]


def test_training_uses_one_exact_2b_pair_call_per_update(
    cli_replays: Mapping[str, object],
) -> None:
    result = cli_replays["result"]
    assert isinstance(result, dict)
    trace = result["trace"]
    assert isinstance(trace, list)
    for row in trace:
        assert row["call_checks"] == {
            "exactly_three_calls": True,
            "factual_miss_input_exact": True,
            "factual_no_miss_input_exact": True,
            "paired_2B_feature_exact": True,
            "paired_2B_occupancy_exact": True,
        }
        losses = row["losses"]
        assert losses["decoder_forward_calls_per_update"] == 3
        assert losses["factual_miss/states"] == 4
        assert losses["factual_no_miss/states"] == 4
        assert losses["outcome/endpoints"] == 4
        assert losses["decoder_states_per_update"] == 12

    paired = result["paired_equivalence_audit"]
    assert paired == {
        "checks": {
            "plus_bit_exact": True,
            "minus_bit_exact": True,
        },
        "plus_max_abs_error": 0.0,
        "minus_max_abs_error": 0.0,
        "all_pass": True,
    }
    assert result["structural_checks"]["paired_2B_equivalence"] is True
    assert result["structural_checks"][
        "every_update_input_binding_exact"
    ] is True

    endpoint = result["dual_endpoint_gradient_audit"]
    assert endpoint["all_pass"] is True
    assert [record["pair_kind"] for record in endpoint["records"]] == [
        "clean_positive",
        "component_null",
    ]
    for record in endpoint["records"]:
        assert record["plus_gradient_finite"] is True
        assert record["minus_gradient_finite"] is True
        assert record["plus_gradient_nonzero_count"] > 0
        assert record["minus_gradient_nonzero_count"] > 0
        assert record["plus_gradient_l2"] > 0.0
        assert record["minus_gradient_l2"] > 0.0


def test_all_data_and_stage_boundaries_remain_false(
    cli_replays: Mapping[str, object],
) -> None:
    result = cli_replays["result"]
    assert isinstance(result, dict)
    boundary = result["execution_boundary"]
    assert boundary == {
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "detection_performance_evaluated": False,
        "real_D_R_bounded_execution_authorized": False,
        "formal_800_authorized": False,
        "full_CURE_authorized": False,
        "cross_detector_authorized": False,
    }
    assert all(value is False for value in boundary.values())

    config = _load_object(_CONFIG)
    config_boundary = config["execution_boundary"]
    assert config_boundary == {
        "D_R_payload_access_allowed": False,
        "D_V_access_allowed": False,
        "D_T_access_allowed": False,
        "dataset_or_cache_payload_access_allowed": False,
        "real_loader_call_allowed": False,
        "package_initialization_module_import_allowed": True,
        "detection_performance_allowed": False,
        "real_D_R_bounded_execution_authorized": False,
        "formal_800_authorized": False,
        "full_CURE_authorized": False,
        "cross_detector_authorized": False,
    }
    assert result["interpretation"] == (
        "in_memory_execution_connectivity_not_performance"
    )


def test_cli_is_create_only_and_never_replaces_existing_output(
    cli_replays: Mapping[str, object],
) -> None:
    output = cli_replays["first_path"]
    assert isinstance(output, Path)
    original = output.read_bytes()
    completed = _run_cli(output)
    assert completed.returncode != 0
    assert "FileExistsError" in completed.stderr
    assert output.read_bytes() == original


def test_runner_ast_has_no_real_data_v7_runner_or_test_import_edges() -> None:
    tree = ast.parse(_RUNNER.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    dynamic_import_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported_modules.add(node.module or "")
        elif isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in {"__import__", "eval", "exec"}
            ) or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
            ):
                dynamic_import_calls.append(node)

    forbidden_exact = {
        "cure_lite.data",
        "cure_lite.experiment.cache_pipeline",
        "cure_lite.experiment.geometry_safe_catalog",
        "cure_lite.experiment.paired_catalog",
        "cure_lite.experiment.paired_outcome_inputs",
        "cure_lite.experiment.training_pipeline",
        "cure_lite.experiment.crossing_factorized_outcome_bounded",
        "tools.run_crossing_factorized_outcome_bounded",
    }
    forbidden_observed = {
        module
        for module in imported_modules
        if module in forbidden_exact
        or module.startswith("tests")
        or "crossing_factorized_outcome_bounded" in module
        or (
            module.startswith("cure_lite.experiment.")
            and module
            != "cure_lite.experiment.conservative_toy_inputs"
        )
    }
    assert forbidden_observed == set()
    assert dynamic_import_calls == []
    assert "cure_lite.experiment.conservative_toy_inputs" in (
        imported_modules
    )
    assert "cure_lite.conservative_factorized_decoder" in imported_modules


def test_evaluate_has_zero_loader_calls_and_only_frozen_file_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_object(_CONFIG)
    proposal = _load_object(_PROPOSAL)
    toy_closure = _load_object(_TOY_CLOSURE)
    allowed_reads = {
        _CONFIG.resolve(),
        _PROPOSAL.resolve(),
        _TOY_CLOSURE.resolve(),
    }
    for binding_name in ("software_bindings", "test_bindings"):
        for repo_path in toy_closure[binding_name]:
            allowed_reads.add((_ROOT / repo_path).resolve())

    opened_paths: list[Path] = []
    original_path_open = Path.open

    def tracked_open(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> object:
        opened_paths.append(path.resolve())
        return original_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)

    def forbidden_payload_call(*args: object, **kwargs: object) -> object:
        raise AssertionError("dataset/cache payload access is forbidden")

    import cure_lite.cache.base_cache as base_cache
    import cure_lite.cache.state_cache as state_cache
    import cure_lite.data as data_module
    import cure_lite.experiment.cache_pipeline as cache_pipeline
    import cure_lite.experiment.training_pipeline as training_pipeline

    for module in (
        base_cache,
        state_cache,
        data_module,
        cache_pipeline,
        training_pipeline,
    ):
        for name, value in tuple(vars(module).items()):
            if inspect.isfunction(value) and value.__module__ == module.__name__:
                monkeypatch.setattr(module, name, forbidden_payload_call)
    monkeypatch.setattr(torch, "load", forbidden_payload_call)
    monkeypatch.setattr(np, "load", forbidden_payload_call)

    result = dry_runner.evaluate()
    assert result["all_pass"] is True
    assert result["runtime"]["real_loader_calls"] == 0
    assert result["runtime"]["dataset_or_cache_payload_accesses"] == 0
    environment_root = Path("/home/md0/ly/MSHNet/.venv").resolve()
    package_metadata = (
        _ROOT / "cure_lite.egg-info" / "entry_points.txt"
    ).resolve()
    workspace_root = Path("/home/md0/ly").resolve()
    workspace_payload_reads = {
        path
        for path in opened_paths
        if workspace_root in path.parents
        and environment_root not in path.parents
        and path != package_metadata
    }
    assert workspace_payload_reads <= allowed_reads
    assert not any(
        part.lower() in {"dataset", "datasets", "runs"}
        for path in opened_paths
        for part in path.parts
    )
    assert not any(
        path.suffix.lower()
        in {".pt", ".pth", ".npy", ".npz", ".pkl", ".pickle"}
        for path in opened_paths
    )
    assert _CONFIG.resolve() in opened_paths
    assert _PROPOSAL.resolve() in opened_paths
    assert _TOY_CLOSURE.resolve() in opened_paths
    assert config["proposal_binding"]["repo_path"] == _PROPOSAL_REPO_PATH
    assert proposal["toy_closure_binding"]["repo_path"] == (
        _TOY_CLOSURE_REPO_PATH
    )
