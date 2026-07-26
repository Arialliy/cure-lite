from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
import shutil

import pytest

from cure_lite.cache.schema import file_sha256, stable_fingerprint
import cure_lite.experiment.paired_tiny_target_artifacts as artifacts
from tools.run_paired_tiny_target_representability import build_parser


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "paired_tiny_target_representability_v1"
    / "config.template.json"
)
IMPLEMENTATION_PATHS = (
    "cure_lite/experiment/paired_tiny_target_representability.py",
    "cure_lite/experiment/paired_tiny_target_artifacts.py",
    "tools/run_paired_tiny_target_representability.py",
)


def _materialized_payload() -> dict[str, object]:
    payload = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    payload.pop("template_status")
    payload.pop("future_materialization_requirements")
    payload["schema_version"] = artifacts.TINY_TARGET_CONFIG_SCHEMA
    payload["artifact_kind"] = "executable_frozen_config"
    payload["template_binding"] = {
        "repo_path": TEMPLATE.relative_to(ROOT).as_posix(),
        "file_sha256": file_sha256(TEMPLATE),
    }
    payload["implementation_bindings"] = {
        "binding_direction": "config_to_implementation_files",
        "config_self_hash_embedded": False,
        "source_files": [
            {
                "repo_path": path,
                "file_sha256": file_sha256(ROOT / path),
            }
            for path in IMPLEMENTATION_PATHS
        ],
    }
    payload["config_fingerprint"] = stable_fingerprint(payload)
    return payload


def _write_config(path: Path, payload: dict[str, object]) -> None:
    path.write_bytes(artifacts._json_bytes(payload))


def test_config_binds_template_and_all_three_implementation_files(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    _write_config(path, _materialized_payload())
    loaded = artifacts.load_tiny_target_audit_config(path)
    assert loaded.payload["execution_policy"][
        "deterministic_process_workers"
    ] == 16
    assert set(loaded.implementation_bindings) == set(IMPLEMENTATION_PATHS)

    changed = deepcopy(_materialized_payload())
    changed["milp"]["margin_objective_scale"] = 1.0
    changed["config_fingerprint"] = stable_fingerprint(
        {
            key: value
            for key, value in changed.items()
            if key != "config_fingerprint"
        }
    )
    changed_path = tmp_path / "changed.json"
    _write_config(changed_path, changed)
    with pytest.raises(RuntimeError, match="template section"):
        artifacts.load_tiny_target_audit_config(changed_path)


def test_config_loader_rejects_symbolic_link(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write_config(path, _materialized_payload())
    link = tmp_path / "config-link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symbolic link"):
        artifacts.load_tiny_target_audit_config(link)


def test_config_loader_rejects_bound_source_sha_drift_and_bad_json(
    tmp_path: Path,
) -> None:
    changed = _materialized_payload()
    changed["implementation_bindings"]["source_files"][0][
        "file_sha256"
    ] = "0" * 64
    changed["config_fingerprint"] = stable_fingerprint(
        {
            key: value
            for key, value in changed.items()
            if key != "config_fingerprint"
        }
    )
    changed_path = tmp_path / "changed-sha.json"
    _write_config(changed_path, changed)
    with pytest.raises(RuntimeError, match="SHA256 differs"):
        artifacts.load_tiny_target_audit_config(changed_path)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value":1,"value":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        artifacts._strict_json(duplicate, name="duplicate")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        artifacts._strict_json(nonfinite, name="nonfinite")


def test_output_path_rejects_existing_target_and_symlink_ancestor(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        artifacts._new_output_path(existing, name="existing")

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        artifacts._new_output_path(
            linked_parent / "new-output",
            name="linked output",
        )


def test_preflight_never_calls_solver_and_two_roots_are_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "config.json"
    _write_config(path, _materialized_payload())
    config = artifacts.load_tiny_target_audit_config(path)

    def forbidden_solver(*args: object, **kwargs: object) -> object:
        raise AssertionError("preflight called the solver")

    monkeypatch.setattr(
        artifacts,
        "solve_tiny_target_case",
        forbidden_solver,
    )
    first = artifacts.publish_tiny_target_preflight(
        config,
        tmp_path / "preflight-r1",
    )
    second = artifacts.publish_tiny_target_preflight(
        config,
        tmp_path / "preflight-r2",
    )
    comparison = artifacts.compare_tiny_target_publications(
        first.root,
        second.root,
    )
    assert comparison["artifact_kind"] == "preflight"
    assert comparison["byte_identical"] is True
    assert comparison["file_count"] == 4
    assert (
        comparison["independent_bound_solver_replay_verified"] is False
    )
    with pytest.raises(ValueError, match="distinct roots"):
        artifacts.compare_tiny_target_publications(first.root, first.root)

    extra = tmp_path / "preflight-extra"
    shutil.copytree(first.root, extra)
    (extra / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inventory"):
        artifacts.load_tiny_target_preflight(extra)

    incomplete = tmp_path / "preflight-incomplete"
    shutil.copytree(first.root, incomplete)
    (incomplete / "COMPLETE.json").unlink()
    (incomplete / ".incomplete").touch()
    with pytest.raises(RuntimeError, match="incomplete"):
        artifacts.load_tiny_target_preflight(incomplete)

    changed = tmp_path / "preflight-changed"
    shutil.copytree(second.root, changed)
    receipt = changed / "preflight_receipt.json"
    receipt.write_bytes(receipt.read_bytes().replace(b"6400", b"6401", 1))
    with pytest.raises((RuntimeError, ValueError)):
        artifacts.compare_tiny_target_publications(first.root, changed)


def test_execution_failure_retains_incomplete_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "config.json"
    _write_config(path, _materialized_payload())
    config = artifacts.load_tiny_target_audit_config(path)
    preflight = artifacts.publish_tiny_target_preflight(
        config,
        tmp_path / "preflight",
    )

    def fail_immediately(*args: object, **kwargs: object) -> object:
        raise RuntimeError("intentional solver interruption")

    monkeypatch.setattr(
        artifacts,
        "_execution_parallelism",
        lambda _config: {
            "deterministic_process_workers": 1,
            "dynamic_cpu_discovery": False,
            "process_start_method": "spawn",
            "process_chunksize": 1,
            "case_result_order": "catalog_order",
            "parent_only_artifact_writer": True,
            "worker_recycling": False,
            "solver_threads_per_process": 1,
            "solver_parallel": False,
            "solver_random_seed": 0,
            "solver_output_flag": False,
        },
    )
    monkeypatch.setattr(
        artifacts,
        "solve_tiny_target_case",
        fail_immediately,
    )
    output = tmp_path / "execution"
    with pytest.raises(RuntimeError, match="intentional"):
        artifacts.execute_tiny_target_audit(config, preflight, output)
    assert (output / ".incomplete").is_file()
    assert not (output / "COMPLETE.json").exists()


def test_public_execute_has_no_solver_injection_and_cli_has_no_model_inputs() -> None:
    assert "solver" not in inspect.signature(
        artifacts.execute_tiny_target_audit
    ).parameters
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "execute",
                "--config",
                "config.json",
                "--preflight",
                "preflight",
                "--output",
                "output",
                "--device",
                "cuda",
            ]
        )
