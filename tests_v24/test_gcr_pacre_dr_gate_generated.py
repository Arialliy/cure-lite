from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cure_lite.cache.schema import stable_fingerprint
import cure_lite_v24.dr_gate as dr_gate
import tools.run_cure_lite_v24_gcr_pacre_dr_gate as dr_gate_cli
from cure_lite_v24.dr_gate import (
    GCR_PACRE_DR_CHECK_NAMES,
    GCRPACREDRPreaccessToken,
    begin_gcr_pacre_dr_materialization,
    build_gcr_pacre_dr_preaccess_artifacts,
    create_gcr_pacre_dr_run_start_marker,
    recompute_gcr_pacre_dr_checks,
    run_gcr_pacre_dr_gate,
    run_gcr_pacre_generated_dr_contract_audit,
    verify_and_issue_gcr_pacre_dr_preaccess,
    verify_gcr_pacre_generated_dr_contract_receipt,
)
from tools.run_cure_lite_v24_gcr_pacre_dr_gate import main as cli_main


REPOSITORY = Path(__file__).resolve().parents[1]
DATASET_FREE_RECEIPT = (
    REPOSITORY
    / "protocols/IRSTD-1K/gcr_pacre_v24/"
    "dataset_free_receipt_r2.json"
)


def _seal(body: dict[str, object], field: str) -> dict[str, object]:
    return {**body, field: stable_fingerprint(body)}


@pytest.fixture(scope="module")
def generated_receipt() -> dict[str, object]:
    return run_gcr_pacre_generated_dr_contract_audit(
        dataset_free_receipt_path=DATASET_FREE_RECEIPT,
        device="cpu",
    )


def test_generated_gate_has_exact_23_of_23_pass(
    generated_receipt: dict[str, object],
) -> None:
    assert tuple(generated_receipt["checks"]) == GCR_PACRE_DR_CHECK_NAMES
    assert len(generated_receipt["raw_observations"]) == 23
    assert all(generated_receipt["checks"].values())
    assert generated_receipt["decision"]["gate_passed"] is True
    assert generated_receipt["decision"]["failed_checks"] == []
    assert (
        verify_gcr_pacre_generated_dr_contract_receipt(
            generated_receipt
        )
        == generated_receipt["receipt_fingerprint"]
    )


def test_generated_gate_never_claims_real_split_or_training(
    generated_receipt: dict[str, object],
) -> None:
    boundary = generated_receipt["boundary"]
    assert boundary == {
        "execution_kind": "generated",
        "split": "generated",
        "D_R_accessed": False,
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
    assert generated_receipt["real_preaccess_token_issued"] is False
    assert generated_receipt["real_D_R_execution_authorized"] is False


def test_generated_raw_mechanism_evidence_is_complete(
    generated_receipt: dict[str, object],
) -> None:
    raw = generated_receipt["raw_observations"]
    witness = dict(raw[GCR_PACRE_DR_CHECK_NAMES[16]])
    witness.pop("observation_fingerprint")
    assert len(witness["target_witnesses"]) == 32
    assert all(
        row["residual_and_gate_paths_finite_nonzero"]
        and row["witness_passed"]
        for row in witness["target_witnesses"]
    )
    parity = raw[GCR_PACRE_DR_CHECK_NAMES[14]]
    assert parity["all_target_flip_parity_exact"] is True
    assert len(parity["target_flip_rows"]) == 32
    common_only = raw[GCR_PACRE_DR_CHECK_NAMES[15]]
    assert (
        common_only["residual_exact_zero_count"]
        == common_only["residual_element_count"]
    )
    assert common_only["common_even_nonzero_count"] > 0
    assert common_only["gate_nonunit_count"] > 0
    assert common_only["field_exact_anchor"] is True
    assert (
        raw[GCR_PACRE_DR_CHECK_NAMES[6]]["exact_collision_count"]
        == 0
    )
    assert (
        raw[GCR_PACRE_DR_CHECK_NAMES[17]]["exact_collision_count"]
        == 0
    )


def test_efficiency_is_measurement_only_without_fixed_threshold(
    generated_receipt: dict[str, object],
) -> None:
    efficiency = generated_receipt["raw_observations"][
        GCR_PACRE_DR_CHECK_NAMES[22]
    ]
    assert efficiency["threshold_or_ratio_gate"] is None
    assert efficiency["performance_ratio_or_absolute_gate"] is None
    assert efficiency["both_arms_complete_finite_no_oom"] is True
    assert set(efficiency["arms"]) == {
        "PACRE_VC_v23",
        "GCR_PACRE_v24",
    }
    assert all(
        row["oom"] is False and row["nonfinite"] is False
        for row in efficiency["arms"].values()
    )


def test_transitive_scientific_source_closure_is_bound() -> None:
    required = {
        "cure_lite/cache/schema.py",
        "cure_lite/coverage_state_level_set.py",
        "cure_lite/coverage_state_observability.py",
        "cure_lite/coverage_state_precomputed_cache.py",
        "cure_lite/coverage_state_raw_catalog.py",
        "cure_lite/coverage_state_schedule.py",
        "cure_lite/coverage_state_sobolev.py",
        "cure_lite/data.py",
        "cure_lite/paired_types.py",
        "cure_lite/sampling.py",
        "cure_lite/splits.py",
        "cure_lite/experiment/cache_pipeline.py",
        "cure_lite/experiment/coverage_state_observability_protocol.py",
        "cure_lite/experiment/coverage_state_raw_catalog.py",
        "cure_lite/experiment/geometry_catalog_protocol.py",
        "cure_lite/experiment/geometry_safe_catalog.py",
        "cure_lite/experiment/training_pipeline.py",
        "cure_lite_v24/dr_gate.py",
        "tools/gcr_pacre_v24_protocol.py",
    }
    assert required <= set(dr_gate.GCR_PACRE_DR_IMPLEMENTATION_PATHS)
    binding = dr_gate._implementation_binding()
    assert len(binding) == len(
        set(dr_gate.GCR_PACRE_DR_IMPLEMENTATION_PATHS)
    )
    assert set(dict(binding)) == set(
        dr_gate.GCR_PACRE_DR_IMPLEMENTATION_PATHS
    )


def test_runtime_import_closure_has_no_unbound_repository_module() -> None:
    script = r"""
import json
from pathlib import Path
import sys

root = Path.cwd().resolve()
import cure_lite_v24.dr_gate as gate
import tools.gcr_pacre_v24_protocol
import tools.run_cure_lite_v24_gcr_pacre_dr_gate

bound = set(gate.GCR_PACRE_DR_IMPLEMENTATION_PATHS)
imported = set()
for module in sys.modules.values():
    raw = getattr(module, "__file__", None)
    if not raw:
        continue
    try:
        path = Path(raw).resolve(strict=True)
        relative = path.relative_to(root)
    except (FileNotFoundError, RuntimeError, ValueError):
        continue
    if relative.suffix == ".py":
        imported.add(str(relative))
print(json.dumps(sorted(imported - bound)))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_real_cli_orders_fixed_preaccess_before_direct_dr_builders() -> None:
    path = (
        REPOSITORY
        / "tools/run_cure_lite_v24_gcr_pacre_dr_gate.py"
    )
    source = path.read_text(encoding="utf-8")
    real_scope = source[source.index("def _run_real(") :]
    verify_offset = real_scope.index(
        "verify_and_issue_gcr_pacre_dr_preaccess("
    )
    consume_offset = real_scope.index(
        "begin_gcr_pacre_dr_materialization("
    )
    marker_offset = real_scope.index(
        "create_gcr_pacre_dr_run_start_marker("
    )
    binding_import_offset = real_scope.index(
        "from cure_lite.experiment.coverage_state_real_dr_inputs import"
    )
    binding_call_offset = real_scope.index(
        "bind_coverage_state_real_dr_sources("
    )
    inputs_import_offset = real_scope.index(
        "from cure_lite.experiment.coverage_state_real_dr_inputs import",
        marker_offset,
    )
    population_import_offset = real_scope.index(
        "from cure_lite.experiment.coverage_state_bounded_protocol import"
    )
    inputs_call_offset = real_scope.index(
        "build_coverage_state_real_dr_inputs("
    )
    population_call_offset = real_scope.index(
        "build_coverage_state_bounded_population("
    )
    assert (
        verify_offset
        < binding_import_offset
        < binding_call_offset
        < marker_offset
        < consume_offset
        < min(inputs_import_offset, population_import_offset)
        < min(inputs_call_offset, population_call_offset)
    )
    assert "build_d_v" not in real_scope.casefold()
    assert "build_d_t" not in real_scope.casefold()


def test_raw_observation_tamper_cannot_be_resealed_as_pass(
    generated_receipt: dict[str, object],
) -> None:
    tampered = deepcopy(generated_receipt)
    name = GCR_PACRE_DR_CHECK_NAMES[17]
    observation = dict(tampered["raw_observations"][name])
    observation.pop("observation_fingerprint")
    observation["exact_collision_count"] = 1
    tampered["raw_observations"][name] = _seal(
        observation,
        "observation_fingerprint",
    )
    tampered["raw_observations_fingerprint"] = stable_fingerprint(
        tampered["raw_observations"]
    )
    body = dict(tampered)
    body.pop("receipt_fingerprint")
    tampered["receipt_fingerprint"] = stable_fingerprint(body)
    with pytest.raises(ValueError, match="checks changed"):
        verify_gcr_pacre_generated_dr_contract_receipt(tampered)


def _authorization_files(
    temporary: Path,
) -> tuple[Path, Path]:
    access, authorization = (
        build_gcr_pacre_dr_preaccess_artifacts()
    )
    access_path = temporary / "access.json"
    access_path.write_text(
        json.dumps(access, sort_keys=True),
        encoding="utf-8",
    )
    authorization_path = temporary / "authorization.json"
    authorization_path.write_text(
        json.dumps(authorization, sort_keys=True),
        encoding="utf-8",
    )
    return authorization_path, access_path


def _run_start_token(
    token: GCRPACREDRPreaccessToken,
    temporary: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> object:
    marker = temporary / "fixed-run-start.json"
    output = temporary / "future-real-receipt.json"
    monkeypatch.setattr(
        dr_gate,
        "_required_run_start_marker_path_from_fingerprint",
        lambda _: marker,
    )
    return create_gcr_pacre_dr_run_start_marker(
        token,
        marker_path=marker,
        requested_device="cpu",
        requested_receipt_output=output,
    )


def test_fixed_private_preaccess_issuer_and_negative_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_path, access_path = _authorization_files(tmp_path)
    token = verify_and_issue_gcr_pacre_dr_preaccess(
        dataset_free_receipt_path=DATASET_FREE_RECEIPT,
        authorization_receipt_path=authorization_path,
        access_audit_receipt_path=access_path,
    )
    assert type(token) is GCRPACREDRPreaccessToken
    assert token.efficiency_section_fingerprint
    run_start = _run_start_token(token, tmp_path, monkeypatch)

    called = False

    def fake_callback(*_: object) -> str:
        nonlocal called
        called = True
        return "0" * 64

    with pytest.raises(TypeError):
        verify_and_issue_gcr_pacre_dr_preaccess(
            dataset_free_receipt_path=DATASET_FREE_RECEIPT,
            authorization_receipt_path=authorization_path,
            access_audit_receipt_path=access_path,
            authorization_verifier=fake_callback,  # type: ignore[call-arg]
        )
    assert called is False

    with pytest.raises(PermissionError, match="private token"):
        run_gcr_pacre_dr_gate(
            preaccess_token={"authorization": "naked"},  # type: ignore[arg-type]
            run_start_token=object(),  # type: ignore[arg-type]
            real_inputs=object(),  # type: ignore[arg-type]
            bounded_population=object(),  # type: ignore[arg-type]
        )
    forged = replace(token, _issuer=object())
    with pytest.raises(PermissionError, match="private token"):
        run_gcr_pacre_dr_gate(
            preaccess_token=forged,
            run_start_token=object(),  # type: ignore[arg-type]
            real_inputs=object(),  # type: ignore[arg-type]
            bounded_population=object(),  # type: ignore[arg-type]
        )
    source_tampered = replace(
        token,
        source_closure_fingerprint="0" * 64,
    )
    with pytest.raises(PermissionError, match="private token"):
        run_gcr_pacre_dr_gate(
            preaccess_token=source_tampered,
            run_start_token=object(),  # type: ignore[arg-type]
            real_inputs=object(),  # type: ignore[arg-type]
            bounded_population=object(),  # type: ignore[arg-type]
        )
    for field in (
        "expected_source_binding_fingerprint",
        "access_audit_receipt_file_sha256",
        "expected_real_inputs_fingerprint",
        "expected_population_fingerprint",
        "expected_cache_fingerprint",
    ):
        input_tampered = replace(token, **{field: "0" * 64})
        with pytest.raises(PermissionError, match="private token"):
            run_gcr_pacre_dr_gate(
                preaccess_token=input_tampered,
                run_start_token=object(),  # type: ignore[arg-type]
                real_inputs=object(),  # type: ignore[arg-type]
                bounded_population=object(),  # type: ignore[arg-type]
            )
    altered_run_start = replace(
        run_start,
        requested_device="cuda:0",
    )
    with pytest.raises(PermissionError, match="run-start token"):
        begin_gcr_pacre_dr_materialization(
            token,
            altered_run_start,
        )
    assert (
        begin_gcr_pacre_dr_materialization(token, run_start)
        is token
    )
    with pytest.raises(PermissionError, match="already materialized"):
        begin_gcr_pacre_dr_materialization(token, run_start)
    with pytest.raises(TypeError, match="real_inputs"):
        run_gcr_pacre_dr_gate(
            preaccess_token=token,
            run_start_token=run_start,  # type: ignore[arg-type]
            real_inputs=object(),  # type: ignore[arg-type]
            bounded_population=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(PermissionError, match="already executed"):
        run_gcr_pacre_dr_gate(
            preaccess_token=token,
            run_start_token=run_start,  # type: ignore[arg-type]
            real_inputs=object(),  # type: ignore[arg-type]
            bounded_population=object(),  # type: ignore[arg-type]
        )


def test_persistent_run_start_blocks_cross_process_crash_replay(
    tmp_path: Path,
) -> None:
    authorization_path, access_path = _authorization_files(tmp_path)
    marker_path = tmp_path / "persistent-run-start.json"
    output_path = tmp_path / "never-written-real-receipt.json"
    script = r"""
from pathlib import Path
import sys

import cure_lite_v24.dr_gate as gate

marker = Path(sys.argv[3]).resolve()
gate._required_run_start_marker_path_from_fingerprint = lambda _: marker

import cure_lite.experiment.coverage_state_real_dr_inputs as real_inputs

def simulated_crash(**_: object) -> object:
    raise RuntimeError("SIMULATED_CRASH_AFTER_PERSISTENT_RUN_START")

real_inputs.build_coverage_state_real_dr_inputs = simulated_crash

import tools.run_cure_lite_v24_gcr_pacre_dr_gate as cli

fixed_outputs = {
    cli.GCR_PACRE_DR_PREACCESS_AUTHORIZATION_PATH: Path(sys.argv[2]).resolve(),
    cli.GCR_PACRE_DR_ACCESS_AUDIT_PATH: Path(sys.argv[4]).resolve(),
    cli.GCR_PACRE_DR_RECEIPT_PATH: Path(sys.argv[5]).resolve(),
}
original_fixed_path = cli._fixed_path
cli._fixed_path = lambda relative: fixed_outputs.get(
    relative, original_fixed_path(relative)
)
cli._fixed_dataset_free_r2_path = lambda: Path(sys.argv[1]).resolve()
cli._fixed_source_paths = lambda: {
    "manifest_path": Path(sys.argv[6]).resolve(),
    "state_index_path": Path(sys.argv[7]).resolve(),
    "geometry_config_path": Path(sys.argv[8]).resolve(),
    "geometry_receipt_path": Path(sys.argv[9]).resolve(),
    "observability_config_path": Path(sys.argv[10]).resolve(),
}

cli.main([
    "real",
    "--execute-real-dr",
    "--device", "cpu",
])
"""
    arguments = [
        sys.executable,
        "-c",
        script,
        str(DATASET_FREE_RECEIPT),
        str(authorization_path),
        str(marker_path),
        str(access_path),
        str(output_path),
        str(
            REPOSITORY
            / "protocols/IRSTD-1K/stage_a_seed42/manifest.json"
        ),
        str(
            REPOSITORY
            / (
                "runs/irstd1k_stage_a_seed42/"
                "cure_lite_stage_a_fx_v3/d_r/state_cache/index.json"
            )
        ),
        str(
            REPOSITORY
            / "protocols/IRSTD-1K/geometry_safe_p0_v2/config.json"
        ),
        str(
            REPOSITORY
            / (
                "runs/irstd1k_stage_a_seed42/"
                "cure_lite_geometry_safe_p0_v2_r1/"
                "receipts/geometry_catalog.json"
            )
        ),
        str(
            REPOSITORY
            / (
                "protocols/IRSTD-1K/"
                "coverage_state_observability_v1/config.json"
            )
        ),
    ]
    first = subprocess.run(
        arguments,
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode != 0
    assert (
        "SIMULATED_CRASH_AFTER_PERSISTENT_RUN_START"
        in first.stderr
    )
    marker_bytes = marker_path.read_bytes()
    marker = json.loads(marker_bytes)
    assert marker["intent"]["D_R_materialization_intended"] is True
    assert marker["intent"]["D_V_materialization_intended"] is False
    assert marker["intent"]["D_T_materialization_intended"] is False
    assert output_path.exists() is False

    second = subprocess.run(
        arguments,
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode != 0
    assert "FileExistsError" in second.stderr
    assert (
        "SIMULATED_CRASH_AFTER_PERSISTENT_RUN_START"
        not in second.stderr
    )
    assert marker_path.read_bytes() == marker_bytes
    assert output_path.exists() is False


def test_preaccess_rejects_source_closure_hash_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_path, access_path = _authorization_files(tmp_path)
    original = dr_gate._implementation_binding()
    dependency = "cure_lite/data.py"
    changed = tuple(
        (name, "0" * 64 if name == dependency else digest)
        for name, digest in original
    )
    assert changed != original
    monkeypatch.setattr(
        dr_gate,
        "_implementation_binding",
        lambda: changed,
    )
    with pytest.raises(PermissionError, match="authorization is invalid"):
        verify_and_issue_gcr_pacre_dr_preaccess(
            dataset_free_receipt_path=DATASET_FREE_RECEIPT,
            authorization_receipt_path=authorization_path,
            access_audit_receipt_path=access_path,
        )


def test_issued_token_rejects_transitive_dependency_hash_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization_path, access_path = _authorization_files(tmp_path)
    token = verify_and_issue_gcr_pacre_dr_preaccess(
        dataset_free_receipt_path=DATASET_FREE_RECEIPT,
        authorization_receipt_path=authorization_path,
        access_audit_receipt_path=access_path,
    )
    run_start = _run_start_token(token, tmp_path, monkeypatch)
    begin_gcr_pacre_dr_materialization(token, run_start)
    original = dr_gate._implementation_binding()
    dependency = "cure_lite/coverage_state_precomputed_cache.py"
    changed = tuple(
        (name, "0" * 64 if name == dependency else digest)
        for name, digest in original
    )
    assert changed != original
    monkeypatch.setattr(
        dr_gate,
        "_implementation_binding",
        lambda: changed,
    )
    with pytest.raises(PermissionError, match="source closure changed"):
        run_gcr_pacre_dr_gate(
            preaccess_token=token,
            run_start_token=run_start,  # type: ignore[arg-type]
            real_inputs=object(),  # type: ignore[arg-type]
            bounded_population=object(),  # type: ignore[arg-type]
        )


def test_cli_generated_writes_new_receipt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "generated.json"
    assert (
        cli_main(
            [
                "generated",
                "--dataset-free-receipt",
                str(DATASET_FREE_RECEIPT),
                "--device",
                "cpu",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["decision"]["gate_passed"] is True
    assert len(payload["checks"]) == 23
    with pytest.raises(FileExistsError):
        cli_main(
            [
                "generated",
                "--dataset-free-receipt",
                str(DATASET_FREE_RECEIPT),
                "--output",
                str(output),
            ]
        )


def test_preaccess_create_verify_and_refuse_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    access_path = tmp_path / "D_R_access_audit.json"
    authorization_path = tmp_path / "D_R_authorization.json"
    fixed_paths = {
        dr_gate_cli.GCR_PACRE_DR_ACCESS_AUDIT_PATH: access_path,
        dr_gate_cli.GCR_PACRE_DR_PREACCESS_AUTHORIZATION_PATH: (
            authorization_path
        ),
    }
    monkeypatch.setattr(
        dr_gate_cli,
        "_fixed_path",
        lambda relative: fixed_paths.get(relative, REPOSITORY / relative),
    )
    monkeypatch.setattr(
        dr_gate_cli,
        "required_gcr_pacre_dr_run_start_marker_path",
        lambda token: tmp_path / "fixed-run-start.json",
    )
    with pytest.raises(SystemExit):
        cli_main(
            [
                "preaccess-create",
                "--access-audit-output",
                str(access_path),
            ]
        )
    assert (
        cli_main(["preaccess-create"])
        == 0
    )
    access = json.loads(access_path.read_text(encoding="utf-8"))
    authorization = json.loads(
        authorization_path.read_text(encoding="utf-8")
    )
    assert access["observed_payloads"] == []
    assert access["allowed_splits"] == ["D_R"]
    assert access["D_V_payload_accessed"] is False
    assert access["D_T_payload_accessed"] is False
    assert authorization["D_R_payload_authorized"] is True
    assert authorization["D_V_payload_authorized"] is False
    assert authorization["D_T_payload_authorized"] is False
    assert authorization["training_authorized"] is False
    assert authorization["expected_real_inputs_fingerprint"] == (
        "ee717a7e13461fb86cacc65d33efd331a"
        "bcf9b27611f254f981082d45eb7bfb4"
    )
    assert authorization["expected_population_fingerprint"] == (
        "1a53467d57bea595afcc1edd3330708d1"
        "dda39e0e2d606325e552e8993e7841c"
    )
    assert authorization["expected_cache_fingerprint"] == (
        "c1627d7e838ff57e27f4753e689bd407"
        "5d2b8a8f4d2ca00754c206092aaf66d8"
    )
    access_bytes = access_path.read_bytes()
    authorization_bytes = authorization_path.read_bytes()
    assert (
        cli_main(["preaccess-verify"])
        == 0
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        cli_main(["preaccess-create"])
    assert access_path.read_bytes() == access_bytes
    assert authorization_path.read_bytes() == authorization_bytes


def test_recomputed_checks_are_boolean_and_ordered(
    generated_receipt: dict[str, object],
) -> None:
    checks = recompute_gcr_pacre_dr_checks(
        generated_receipt["raw_observations"],
        execution_kind="generated",
    )
    assert tuple(name for name, _ in checks) == GCR_PACRE_DR_CHECK_NAMES
    assert all(type(value) is bool and value for _, value in checks)
