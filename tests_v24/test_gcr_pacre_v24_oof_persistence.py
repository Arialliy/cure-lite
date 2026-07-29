from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Mapping

import pytest

import cure_lite_v24.oof_run_start as oof_run_start
from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    CoverageStateRealDRSourceBinding,
)
from cure_lite_v24.artifact_io import atomic_write_new_canonical_json
from cure_lite_v24.dr_gate import GCR_PACRE_DR_PASS_DECISION


_SOURCE_ROWS = (
    (
        "generated/oof-persistence-source.py",
        stable_fingerprint({"generated": "source"}),
    ),
)


def _source_closure_fingerprint(
    rows: tuple[tuple[str, str], ...] | None = None,
) -> str:
    resolved = _SOURCE_ROWS if rows is None else tuple(rows)
    return stable_fingerprint(
        {
            "schema_version": "generated-oof-source-closure-v1",
            "source_hashes": dict(resolved),
        }
    )


def _sealed(payload: Mapping[str, object], *, field: str) -> dict[str, object]:
    body = dict(payload)
    if field in body:
        raise ValueError(f"{field} is already present")
    return {**body, field: stable_fingerprint(body)}


def _verify_generated_dr_receipt(
    payload: Mapping[str, object],
) -> str:
    body = dict(payload)
    fingerprint = body.pop("receipt_fingerprint", None)
    if fingerprint != stable_fingerprint(body):
        raise ValueError("generated D_R receipt fingerprint changed")
    return str(fingerprint)


def _write_generated_json(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_new_canonical_json(path.resolve(), payload)


def _source_binding(
    source_paths: Mapping[str, Path],
) -> CoverageStateRealDRSourceBinding:
    source_fingerprints = {
        "observability_config": stable_fingerprint(
            {"generated": "observability"}
        ),
        "geometry_protocol_config": stable_fingerprint(
            {"generated": "geometry-protocol"}
        ),
        "geometry_catalog": stable_fingerprint(
            {"generated": "geometry-catalog"}
        ),
        "state_index": stable_fingerprint(
            {"generated": "state-index"}
        ),
        "base": stable_fingerprint({"generated": "base"}),
        "base_state": stable_fingerprint(
            {"generated": "base-state"}
        ),
        "state": stable_fingerprint({"generated": "state"}),
        "gt": stable_fingerprint({"generated": "gt"}),
    }
    provisional = CoverageStateRealDRSourceBinding(
        manifest_path=source_paths["manifest_path"],
        state_index_path=source_paths["state_index_path"],
        geometry_config_path=source_paths["geometry_config_path"],
        geometry_receipt_path=source_paths["geometry_receipt_path"],
        observability_config_path=source_paths[
            "observability_config_path"
        ],
        manifest_file_sha256=file_sha256(
            source_paths["manifest_path"]
        ),
        state_index_file_sha256=file_sha256(
            source_paths["state_index_path"]
        ),
        geometry_config_file_sha256=file_sha256(
            source_paths["geometry_config_path"]
        ),
        geometry_receipt_file_sha256=file_sha256(
            source_paths["geometry_receipt_path"]
        ),
        observability_config_file_sha256=file_sha256(
            source_paths["observability_config_path"]
        ),
        observability_config_fingerprint=source_fingerprints[
            "observability_config"
        ],
        geometry_protocol_config_fingerprint=source_fingerprints[
            "geometry_protocol_config"
        ],
        geometry_catalog_fingerprint=source_fingerprints[
            "geometry_catalog"
        ],
        state_index_fingerprint=source_fingerprints["state_index"],
        base_fingerprint=source_fingerprints["base"],
        base_state_fingerprint=source_fingerprints["base_state"],
        state_fingerprint=source_fingerprints["state"],
        gt_fingerprint=source_fingerprints["gt"],
        dataset="GENERATED-OOF",
        split="D_R",
        binding_fingerprint="0" * 64,
    )
    return replace(
        provisional,
        binding_fingerprint=stable_fingerprint(
            provisional.canonical_payload()
        ),
    )


@dataclass(frozen=True)
class _GeneratedAuthorization:
    runtime_root: Path
    receipt_path: Path
    source_paths: dict[str, Path]
    split_fingerprints: dict[str, str]
    authorization_fingerprint: str
    authorization_payload: dict[str, object]

    def child_config(
        self,
        *,
        runtime_root: Path | None = None,
    ) -> dict[str, object]:
        return {
            "runtime_root": str(
                self.runtime_root if runtime_root is None else runtime_root
            ),
            "receipt_path": str(self.receipt_path),
            "source_paths": {
                name: str(path)
                for name, path in self.source_paths.items()
            },
            "split_fingerprints": dict(self.split_fingerprints),
            "source_rows": [list(row) for row in _SOURCE_ROWS],
        }


def _generated_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _GeneratedAuthorization:
    source_root = (tmp_path / "metadata").resolve()
    source_paths = {
        "manifest_path": source_root / "manifest.json",
        "state_index_path": source_root / "state_index.json",
        "geometry_config_path": source_root / "geometry_config.json",
        "geometry_receipt_path": source_root / "geometry_receipt.json",
        "observability_config_path": (
            source_root / "observability_config.json"
        ),
    }
    for name, path in source_paths.items():
        _write_generated_json(path, {"generated_metadata": name})
    binding = _source_binding(source_paths)
    receipt_path = (tmp_path / "D_R_structural_receipt.json").resolve()
    receipt = _sealed(
        {
            "schema_version": "generated-D_R-structural-receipt-v1",
            "decision": {"status": GCR_PACRE_DR_PASS_DECISION},
            "input_binding": {
                "source_binding_fingerprint": (
                    binding.binding_fingerprint
                )
            },
            "real_inputs_fingerprint": stable_fingerprint(
                {"generated": "real-inputs"}
            ),
            "cache_fingerprint": stable_fingerprint(
                {"generated": "full-cache"}
            ),
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        },
        field="receipt_fingerprint",
    )
    _write_generated_json(receipt_path, receipt)
    runtime_root = (tmp_path / "runtime").resolve()
    split_fingerprints = {
        "receipt_fingerprint": stable_fingerprint(
            {"generated": "split-receipt"}
        ),
        "plan_fingerprint": stable_fingerprint(
            {"generated": "split-plan"}
        ),
        "root_by_sample_fingerprint": stable_fingerprint(
            {"generated": "root-by-sample"}
        ),
    }
    split = SimpleNamespace(**split_fingerprints)

    monkeypatch.setattr(
        CoverageStateRealDRSourceBinding,
        "verify_unchanged",
        lambda self: None,
    )
    monkeypatch.setattr(
        oof_run_start,
        "require_verified_oof4_split",
        lambda value: value,
    )
    monkeypatch.setattr(
        oof_run_start,
        "required_oof_runtime_root",
        lambda: runtime_root,
    )
    monkeypatch.setattr(
        oof_run_start,
        "required_oof_dr_receipt_path",
        lambda: receipt_path,
    )
    monkeypatch.setattr(
        oof_run_start,
        "required_oof_dr_source_paths",
        lambda: dict(source_paths),
    )
    monkeypatch.setattr(
        oof_run_start,
        "verify_gcr_pacre_dr_receipt",
        _verify_generated_dr_receipt,
    )
    monkeypatch.setattr(
        oof_run_start,
        "gcr_pacre_v24_source_closure_hashes",
        lambda: _SOURCE_ROWS,
    )
    monkeypatch.setattr(
        oof_run_start,
        "gcr_pacre_v24_source_closure_fingerprint",
        _source_closure_fingerprint,
    )
    authorization = oof_run_start.authorize_real_oof4_execution_new(
        verified_split=split,
        source_binding=binding,
        runtime_root=runtime_root,
    )
    return _GeneratedAuthorization(
        runtime_root=runtime_root,
        receipt_path=receipt_path,
        source_paths=source_paths,
        split_fingerprints=split_fingerprints,
        authorization_fingerprint=(
            authorization.authorization_fingerprint
        ),
        authorization_payload=authorization.payload,
    )


def _reseal_authorization(
    payload: Mapping[str, object],
) -> dict[str, object]:
    body = deepcopy(dict(payload))
    body.pop("authorization_fingerprint", None)
    return {
        **body,
        "authorization_fingerprint": stable_fingerprint(body),
    }


_CHILD_PROGRAM = r"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

import cure_lite.experiment.coverage_state_real_dr_inputs as real_inputs
import cure_lite_v24.oof_run_start as run_start
from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    CoverageStateRealDRSourceBinding,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_training_scalar_cache,
)


config = json.loads(os.environ["OOF_PERSISTENCE_CONFIG"])
mode = os.environ["OOF_PERSISTENCE_MODE"]
runtime_root = Path(config["runtime_root"])
receipt_path = Path(config["receipt_path"])
source_paths = {
    name: Path(path) for name, path in config["source_paths"].items()
}
source_rows = tuple(tuple(row) for row in config["source_rows"])
split = SimpleNamespace(**config["split_fingerprints"])


def source_closure_fingerprint(rows=None):
    resolved = source_rows if rows is None else tuple(rows)
    return stable_fingerprint(
        {
            "schema_version": "generated-oof-source-closure-v1",
            "source_hashes": dict(resolved),
        }
    )


def verify_generated_dr_receipt(payload):
    body = dict(payload)
    fingerprint = body.pop("receipt_fingerprint", None)
    if fingerprint != stable_fingerprint(body):
        raise ValueError("generated D_R receipt fingerprint changed")
    return str(fingerprint)


def binding_from_authorization(payload):
    source = payload["source_binding"]
    paths = source["paths"]
    digests = source["file_sha256"]
    fingerprints = source["fingerprints"]
    return CoverageStateRealDRSourceBinding(
        manifest_path=Path(paths["manifest"]),
        state_index_path=Path(paths["state_index"]),
        geometry_config_path=Path(paths["geometry_config"]),
        geometry_receipt_path=Path(paths["geometry_receipt"]),
        observability_config_path=Path(paths["observability_config"]),
        manifest_file_sha256=digests["manifest"],
        state_index_file_sha256=digests["state_index"],
        geometry_config_file_sha256=digests["geometry_config"],
        geometry_receipt_file_sha256=digests["geometry_receipt"],
        observability_config_file_sha256=digests[
            "observability_config"
        ],
        observability_config_fingerprint=fingerprints[
            "observability_config"
        ],
        geometry_protocol_config_fingerprint=fingerprints[
            "geometry_protocol_config"
        ],
        geometry_catalog_fingerprint=fingerprints["geometry_catalog"],
        state_index_fingerprint=fingerprints["state_index"],
        base_fingerprint=fingerprints["base"],
        base_state_fingerprint=fingerprints["base_state"],
        state_fingerprint=fingerprints["state"],
        gt_fingerprint=fingerprints["gt"],
        dataset=source["dataset"],
        split=source["split"],
        binding_fingerprint=payload["source_binding_fingerprint"],
    )


run_start.require_verified_oof4_split = lambda value: value
run_start.required_oof_runtime_root = lambda: runtime_root
run_start.required_oof_dr_receipt_path = lambda: receipt_path
run_start.required_oof_dr_source_paths = lambda: dict(source_paths)
run_start.verify_gcr_pacre_dr_receipt = verify_generated_dr_receipt
run_start.gcr_pacre_v24_source_closure_hashes = lambda: source_rows
run_start.gcr_pacre_v24_source_closure_fingerprint = (
    source_closure_fingerprint
)


if mode == "wrong_receipt_path":
    original_read = run_start.read_canonical_json
    authorization_path = runtime_root / "authorization.json"
    sentinel_calls = []

    def guarded_read(path):
        resolved = Path(path)
        if resolved == authorization_path:
            return original_read(resolved)
        sentinel_calls.append(str(resolved))
        raise AssertionError("wrong D_R path reached read sentinel")

    run_start.read_canonical_json = guarded_read
    try:
        run_start.load_and_verify_real_oof4_execution_authorization(
            verified_split=split,
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("wrong sealed D_R path was accepted")
    if sentinel_calls:
        raise AssertionError(
            "wrong sealed D_R path was read before rejection"
        )
    print(json.dumps({"rejected_before_read": True}))
    raise SystemExit(0)


authorization_payload = run_start.read_canonical_json(
    runtime_root / "authorization.json"
)
binding = binding_from_authorization(authorization_payload)


if mode == "wrong_source_path":
    bind_calls = []

    def forbidden_bind(**kwargs):
        bind_calls.append(dict(kwargs))
        raise AssertionError("wrong source path reached bind sentinel")

    real_inputs.bind_coverage_state_real_dr_sources = forbidden_bind
    try:
        run_start.load_and_verify_real_oof4_execution_authorization(
            verified_split=split,
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("wrong sealed source path was accepted")
    if bind_calls:
        raise AssertionError(
            "wrong sealed source path was bound before rejection"
        )
    print(json.dumps({"rejected_before_bind": True}))
    raise SystemExit(0)


def generated_bind(**kwargs):
    expected = {
        "manifest_path": str(binding.manifest_path),
        "state_index_path": str(binding.state_index_path),
        "geometry_config_path": str(binding.geometry_config_path),
        "geometry_receipt_path": str(binding.geometry_receipt_path),
        "observability_config_path": str(
            binding.observability_config_path
        ),
    }
    if kwargs != expected:
        raise PermissionError("loader requested an unsealed source path")
    return binding, None, None, None


real_inputs.bind_coverage_state_real_dr_sources = generated_bind
authorization = (
    run_start.load_and_verify_real_oof4_execution_authorization(
        verified_split=split,
    )
)


if mode == "load":
    print(
        json.dumps(
            {
                "authorization_fingerprint": (
                    authorization.authorization_fingerprint
                ),
                "runtime_root": authorization.runtime_root,
                "loaded_in_fresh_process": True,
            }
        )
    )
    raise SystemExit(0)


cache = make_training_scalar_cache()
schedule = build_coverage_state_training_schedule(
    cache,
    CoverageStateScheduleConfig(
        seed=42,
        epochs=10,
        steps_per_epoch=40,
    ),
)
sample_ids = tuple(
    sorted(
        {
            value.record.sample_id for value in cache.natural_records
        }
        | {
            value.record.sample_id for value in cache.pair_records
        }
    )
)
closure = SimpleNamespace(
    fold_id=0,
    closure_fingerprint=stable_fingerprint(
        {"generated": "fold-closure"}
    ),
    split_receipt_fingerprint=split.receipt_fingerprint,
    train_sample_ids=sample_ids,
    held_out_sample_ids=("generated-holdout",),
    train_root_source_ids=tuple(
        f"generated-root-{sample_id}" for sample_id in sample_ids
    ),
    held_out_root_source_ids=("generated-holdout-root",),
)
control = SimpleNamespace(
    fold_id=0,
    partition="train",
    arm="PACRE_VC_v23_control",
    artifact_fingerprint=stable_fingerprint(
        {"generated": "control-cache"}
    ),
    payload={
        "semantic_payload_fingerprint": schedule.cache_fingerprint,
    },
)
candidate = SimpleNamespace(
    fold_id=0,
    partition="train",
    arm="GCR_PACRE_v24",
    artifact_fingerprint=stable_fingerprint(
        {"generated": "candidate-cache"}
    ),
    payload={
        "semantic_payload_fingerprint": schedule.cache_fingerprint,
    },
)
run_start.require_verified_oof_fold_closure = lambda value: value
run_start.require_verified_oof_cache_artifact = lambda value: value

model_or_optimizer_calls = []


def forbidden_model_or_optimizer(*args, **kwargs):
    model_or_optimizer_calls.append((args, kwargs))
    raise AssertionError("model/optimizer allocation preceded run claim")


import cure_lite_v24.oof_training as oof_training
oof_training.build_pacre_vc_training_model = (
    forbidden_model_or_optimizer
)
oof_training.build_gcr_pacre_training_model = (
    forbidden_model_or_optimizer
)
torch.optim.Adam = forbidden_model_or_optimizer
marker_path = runtime_root / "fold_0" / "run_start.json"


if mode == "marker_crash":
    token = run_start.create_oof_training_run_start_new(
        authorization,
        closure,
        schedule=schedule,
        control_cache_artifact=control,
        candidate_cache_artifact=candidate,
    )
    if model_or_optimizer_calls:
        raise AssertionError("model/optimizer was allocated before crash")
    print(
        json.dumps(
            {
                "marker_fingerprint": token.marker_fingerprint,
                "marker_file_sha256": file_sha256(marker_path),
                "process_instance_fingerprint": (
                    token.process_instance_fingerprint
                ),
            }
        ),
        flush=True,
    )
    raise SystemExit(73)


if mode != "marker_retry":
    raise AssertionError(f"unknown child mode: {mode}")
persisted = run_start.read_canonical_json(marker_path)
if persisted["process_instance_fingerprint"] == (
    run_start.OOF_PROCESS_INSTANCE_FINGERPRINT
):
    raise AssertionError("retry did not run in a fresh process")
before = marker_path.read_bytes()
try:
    run_start.create_oof_training_run_start_new(
        authorization,
        closure,
        schedule=schedule,
        control_cache_artifact=control,
        candidate_cache_artifact=candidate,
    )
except FileExistsError:
    pass
else:
    raise AssertionError("fresh process adopted an existing fold marker")
after = marker_path.read_bytes()
if before != after:
    raise AssertionError("retry changed persistent marker bytes")
if model_or_optimizer_calls:
    raise AssertionError("retry reached model/optimizer allocation")
print(
    json.dumps(
        {
            "retry_rejected": True,
            "marker_unchanged": True,
            "model_or_optimizer_calls": 0,
            "marker_file_sha256": file_sha256(marker_path),
        }
    )
)
"""


def _run_child(
    mode: str,
    config: Mapping[str, object],
    *,
    expected_returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OOF_PERSISTENCE_MODE"] = mode
    environment["OOF_PERSISTENCE_CONFIG"] = json.dumps(config)
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode != expected_returncode:
        raise AssertionError(
            f"child mode {mode!r} returned {result.returncode}, "
            f"expected {expected_returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result


def _child_json(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise AssertionError("child process produced no JSON")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise AssertionError("child process JSON is not an object")
    return value


def test_authorization_v2_reloads_in_fresh_process_and_rejects_wrong_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _generated_authorization(tmp_path, monkeypatch)
    loaded = _child_json(
        _run_child("load", generated.child_config())
    )
    assert loaded == {
        "authorization_fingerprint": (
            generated.authorization_fingerprint
        ),
        "runtime_root": str(generated.runtime_root),
        "loaded_in_fresh_process": True,
    }

    wrong_receipt_root = (tmp_path / "wrong_receipt_runtime").resolve()
    wrong_receipt_root.mkdir()
    wrong_receipt = deepcopy(generated.authorization_payload)
    receipt_artifact = wrong_receipt[
        "D_R_structural_receipt_artifact"
    ]
    assert isinstance(receipt_artifact, dict)
    receipt_artifact["path"] = str(
        (tmp_path / "D_V_payload_must_not_be_read.json").resolve()
    )
    _write_generated_json(
        wrong_receipt_root / "authorization.json",
        _reseal_authorization(wrong_receipt),
    )
    receipt_rejection = _child_json(
        _run_child(
            "wrong_receipt_path",
            generated.child_config(runtime_root=wrong_receipt_root),
        )
    )
    assert receipt_rejection == {"rejected_before_read": True}

    wrong_source_root = (tmp_path / "wrong_source_runtime").resolve()
    wrong_source_root.mkdir()
    wrong_source = deepcopy(generated.authorization_payload)
    source_binding = wrong_source["source_binding"]
    assert isinstance(source_binding, dict)
    sealed_paths = source_binding["paths"]
    assert isinstance(sealed_paths, dict)
    sealed_paths["manifest"] = str(
        (tmp_path / "D_T_source_must_not_be_bound.json").resolve()
    )
    _write_generated_json(
        wrong_source_root / "authorization.json",
        _reseal_authorization(wrong_source),
    )
    source_rejection = _child_json(
        _run_child(
            "wrong_source_path",
            generated.child_config(runtime_root=wrong_source_root),
        )
    )
    assert source_rejection == {"rejected_before_bind": True}


def test_run_start_marker_survives_crash_and_blocks_fresh_process_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _generated_authorization(tmp_path, monkeypatch)
    first = _child_json(
        _run_child(
            "marker_crash",
            generated.child_config(),
            expected_returncode=73,
        )
    )
    marker = generated.runtime_root / "fold_0" / "run_start.json"
    assert marker.is_file()
    assert marker.stat().st_mode & 0o777 == 0o444
    assert first["marker_file_sha256"] == file_sha256(marker)
    before = marker.read_bytes()

    retry = _child_json(
        _run_child("marker_retry", generated.child_config())
    )
    assert retry == {
        "retry_rejected": True,
        "marker_unchanged": True,
        "model_or_optimizer_calls": 0,
        "marker_file_sha256": file_sha256(marker),
    }
    assert marker.read_bytes() == before
