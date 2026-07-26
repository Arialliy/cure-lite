from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from tools import dry_run_crossing_factorized_outcome_bounded as runner


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "continuously_recoverable_log_vacancy_evidence_crossing_v7"
    / "bounded_dry_run_config.json"
)


@pytest.fixture(scope="module")
def dry_result() -> dict[str, object]:
    return runner.evaluate(_CONFIG)


def test_dry_run_executes_exact_synthetic_cpu_contract(
    dry_result: dict[str, object],
) -> None:
    result = dry_result
    assert result["schema_version"] == (
        "cure-lite-cr-lvec-v7-bounded-dry-run-result-v1"
    )
    assert result["method_id"] == "cr_lvec_v7"
    assert result["mode"] == "synthetic_bounded_implementation_dry_run"
    assert result["decision"] == "CR_LVEC_V7_BOUNDED_DRY_RUN_PASS"
    assert result["all_pass"] is True
    assert all(result["checks"].values())

    contract = result["contract"]
    assert contract["device"] == "cpu"
    assert contract["seed"] == 7817
    assert contract["epochs"] == 2
    assert contract["steps_per_epoch"] == 4
    assert contract["optimizer_updates"] == 8
    assert contract["decoder_forward_calls_per_update"] == 3
    assert contract["decoder_states_per_update"] == 12
    assert contract["data_source"] == (
        "fixed_in_memory_synthetic_fixtures_only"
    )

    training = result["training_audit"]
    assert training["optimizer_updates"] == 8
    assert training["decoder_forward_calls"] == 24
    assert training["expected_decoder_forward_calls"] == 24
    assert training["decoder_forward_batch_sizes"] == [4] * 24
    assert training["backward_calls"] == 8
    assert training["optimizer_steps"] == 8
    assert len(training["losses"]) == 8
    assert training["all_losses_finite"] is True
    assert training["parameter_gradient_updates_passed"] == 8
    assert len(training["parameter_gradient_audits"]) == 8
    assert [
        audit["update"]
        for audit in training["parameter_gradient_audits"]
    ] == list(range(1, 9))
    assert all(
        audit["parameter_tensor_count"] == 6
        and audit["missing_gradient_names"] == []
        and audit["nonfinite_gradient_names"] == []
        and audit["all_present"] is True
        and audit["all_finite"] is True
        and audit["passed"] is True
        for audit in training["parameter_gradient_audits"]
    )
    assert (
        training["all_parameter_gradients_finite_each_update"] is True
    )
    assert training["decoder_state_changed"] is True
    assert training["input_features_detached"] is True
    assert training["fixed_budget_pass"] is True


def test_dry_run_covers_operator_pairs_gradients_and_stops_before_bad_step(
    dry_result: dict[str, object],
) -> None:
    result = dry_result
    topology = result["topology_audit"]
    assert topology["parameter_count"] == 4385
    assert topology["parameter_tensor_count"] == 6
    assert topology["state_keys_equal_v4"] is True
    assert topology["initial_state_values_equal_v4"] is True
    assert topology["module_types_equal_v4"] is True
    assert topology["passed"] is True

    operator = result["operator_audit"]
    assert operator["safe_forward_exact"] is True
    assert operator["safe_gradient_exact"] is True
    assert operator["negative_probe_gradient"] > 0.0
    assert operator["zero_recovery_probe_failed_fast"] is True
    assert operator["nonfinite_positive_probe_failed_fast"] is True
    assert operator["occupancy_burden_exact"] is True
    assert operator["crossing_margin_exact"] is True
    assert operator["crossing_forward_exact"] is True
    assert operator["logit_composition_exact"] is True
    assert operator["passed"] is True

    fixtures = result["fixture_audit"]
    assert fixtures["target_pixel_counts"] == [1, 2, 3]
    assert fixtures["pair_kinds_covered"] == [
        "clean_positive",
        "component_null",
        "identity_null",
    ]
    assert fixtures["source_disjoint_within_update"] is True
    assert fixtures["pre_mask_contract"] is True
    assert fixtures["clean_and_component_null_controls"] is True
    assert fixtures["passed"] is True

    gradients = result["dual_endpoint_gradient_audit"]
    assert gradients["plus_finite"] is True
    assert gradients["minus_finite"] is True
    assert gradients["plus_nonzero"] is True
    assert gradients["minus_nonzero"] is True
    assert gradients["passed"] is True
    stop = result["structural_stop_audit"]
    assert stop["invalid_factual_batch_rejected"] is True
    assert stop["decoder_state_unchanged"] is True
    assert stop["decoder_forward_calls"] == 0
    assert stop["optimizer_state_untouched"] is True
    assert stop["passed"] is True


def test_dry_run_structurally_excludes_real_loader_and_authorization() -> None:
    assert not hasattr(runner, "_load_real_catalog")
    result = runner.evaluate(_CONFIG)
    assert result["real_catalog_loader_call_count"] == 0
    assert result["real_catalog_loader_call_count_basis"] == (
        "structural_isolation_no_real_loader_import_or_call_edge"
    )
    assert result["real_loader_imported_by_dry_entrypoint"] is False
    assert (
        result["real_loader_symbol_reachable_from_dry_execution"] is False
    )
    assert result["D_R_payload_accessed"] is False
    assert result["D_V_accessed"] is False
    assert result["D_T_accessed"] is False
    assert result["real_D_R_bounded_authorized"] is False
    assert result["real_run_authorization_created"] is False
    assert result["formal_800_authorized"] is False
    assert result["full_CURE_authorized"] is False
    assert result["other_detector_integration_authorized"] is False
    assert result["bounded_implementation_closure_authorized"] is False
    assert (
        result["bounded_implementation_closure_eligible_after_replay"]
        is True
    )


def test_roundtrip_audit_is_scoped_to_serialization_probe(
    dry_result: dict[str, object],
) -> None:
    audit = dry_result["artifact_roundtrip_audit"]
    assert audit["scope"] == "canonical_json_serialization_probe_only"
    assert audit["covers_real_runner_publication"] is False
    assert audit["real_runner_result_variants_exercised"] is False
    assert dry_result["artifact_roundtrip_scope"] == (
        "serialization_probe_only_not_real_runner_publication"
    )


def test_result_fingerprint_and_protocol_chain_are_exact(
    dry_result: dict[str, object],
) -> None:
    result = dry_result
    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)

    binding = result["protocol_binding"]
    assert binding["bounded_implementation_proposal"]["file_sha256"] == (
        runner.EXPECTED_BOUNDED_PROPOSAL_SHA256
    )
    assert binding["bounded_implementation_proposal"][
        "receipt_fingerprint"
    ] == runner.EXPECTED_BOUNDED_PROPOSAL_FINGERPRINT
    assert binding["bounded_config"]["file_sha256"] == (
        runner.EXPECTED_BOUNDED_CONFIG_SHA256
    )
    assert binding["bounded_config"]["config_fingerprint"] == (
        runner.EXPECTED_BOUNDED_CONFIG_FINGERPRINT
    )
    assert binding["bounded_dry_run_config"]["file_sha256"] == (
        runner.EXPECTED_DRY_CONFIG_SHA256
    )
    assert binding["bounded_dry_run_config"]["config_fingerprint"] == (
        runner.EXPECTED_DRY_CONFIG_FINGERPRINT
    )
    assert file_sha256(_CONFIG) == runner.EXPECTED_DRY_CONFIG_SHA256


def test_protocol_payload_changes_are_rejected() -> None:
    dry = json.loads(_CONFIG.read_text(encoding="utf-8"))
    changed = deepcopy(dry)
    changed["optimization"]["optimizer_updates"] = 9
    unsigned = dict(changed)
    unsigned.pop("config_fingerprint")
    changed["config_fingerprint"] = stable_fingerprint(unsigned)
    with pytest.raises(
        RuntimeError,
        match="fingerprint|optimization contract",
    ):
        runner._validate_dry_config(changed)

    bounded = json.loads(
        runner._BOUNDED_CONFIG.read_text(encoding="utf-8")
    )
    changed_bounded = deepcopy(bounded)
    changed_bounded["cuda_synchronization_policy"][
        "bounded_potential_host_synchronization_check_sites"
    ] = 1199
    unsigned_bounded = dict(changed_bounded)
    unsigned_bounded.pop("config_fingerprint")
    changed_bounded["config_fingerprint"] = stable_fingerprint(
        unsigned_bounded
    )
    with pytest.raises(
        RuntimeError,
        match="fingerprint|execution contract",
    ):
        runner._validate_bounded_config(changed_bounded)


def test_only_canonical_config_is_accepted(tmp_path: Path) -> None:
    copied = tmp_path / "bounded_dry_run_config.json"
    copied.write_bytes(_CONFIG.read_bytes())
    with pytest.raises(ValueError, match="canonical bounded dry config"):
        runner.evaluate(copied)


def test_create_only_output_and_cli_payload_round_trip(
    dry_result: dict[str, object],
    tmp_path: Path,
) -> None:
    output = tmp_path / "dry-result.json"
    runner._write_new(output, dry_result)
    assert json.loads(output.read_text(encoding="utf-8")) == dry_result
    with pytest.raises(FileExistsError):
        runner._write_new(output, dry_result)


def test_two_independent_cli_processes_are_byte_identical(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    command_prefix = [
        sys.executable,
        str(_ROOT / "tools" / "dry_run_crossing_factorized_outcome_bounded.py"),
        "--config",
        str(_CONFIG),
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    run_first = subprocess.run(
        [*command_prefix, "--output", str(first)],
        cwd=_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    run_second = subprocess.run(
        [*command_prefix, "--output", str(second)],
        cwd=_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert run_first.returncode == 0, run_first.stderr
    assert run_second.returncode == 0, run_second.stderr
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8"))["all_pass"] is True


def test_cli_rejects_existing_output_without_replacing_it(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing.json"
    sentinel = b"preserve-me\n"
    output.write_bytes(sentinel)
    completed = subprocess.run(
        [
            sys.executable,
            str(
                _ROOT
                / "tools"
                / "dry_run_crossing_factorized_outcome_bounded.py"
            ),
            "--config",
            str(_CONFIG),
            "--output",
            str(output),
        ],
        cwd=_ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert output.read_bytes() == sentinel
