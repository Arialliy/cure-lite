#!/usr/bin/env python3
"""Run the frozen D_R-only bounded paired computational learnability gate.

The command reconstructs the exact real pair catalog sealed by the paired
preflight, selects the fixed identity-only 16-unit micro-populations, and
trains one fresh decoder for 400 updates.  It never opens D_V/D_T, performs
calibration, reports detection performance, or authorizes the formal
800-epoch experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

# CUDA requires this process contract before torch-backed modules are loaded.
# An incompatible caller-supplied value is preserved here and rejected by the
# frozen execution policy rather than being silently replaced.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.data import ManifestImageDataset  # noqa: E402
from cure_lite.experiment.cache_pipeline import load_d_r_cache_bundle  # noqa: E402
from cure_lite.experiment.geometry_catalog_protocol import (  # noqa: E402
    load_geometry_catalog_protocol,
)
from cure_lite.experiment.geometry_safe_catalog import (  # noqa: E402
    build_geometry_safe_catalog,
    build_geometry_safe_p0_view,
    build_p0_a1_receipt,
)
from cure_lite.experiment.paired_bounded_learnability import (  # noqa: E402
    BOUNDED_EXECUTION_SCHEMA,
    BOUNDED_MICRO_POPULATION_SCHEMA,
    BOUNDED_MICRO_SCHEDULE_SCHEMA,
    build_bounded_micro_population,
    build_bounded_micro_schedule,
    execute_bounded_learnability,
)
from cure_lite.experiment.paired_catalog import build_pair_catalog  # noqa: E402
from cure_lite.experiment.training_pipeline import (  # noqa: E402
    CachedTrainingSource,
    prepare_training_catalog,
)
from cure_lite.splits import load_and_validate_manifest  # noqa: E402
from tools import run_paired_preflight as pair_preflight_runner  # noqa: E402


BOUNDED_CONFIG_REPO_PATH = (
    "protocols/IRSTD-1K/paired_bounded_learnability_v1/config.json"
)
BOUNDED_CONFIG_FILE_SHA256 = (
    "90bb22fb6e46add584b99b24f54b9926a9cda8e63d73eda5a67447f6668eabd5"
)
BOUNDED_CONFIG_FINGERPRINT = (
    "0c2895c843bcd5ed5dd3865118e8006d8bc3a2d74665052cddb2c1572f007785"
)
BOUNDED_RUN_SCHEMA = "cure-lite-paired-bounded-learnability-run-v1"
BOUNDED_DECISION_SCHEMA = (
    "cure-lite-paired-bounded-learnability-decision-v1"
)
_ROOT = Path(__file__).resolve().parents[1]
_INCOMPLETE = ".incomplete"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--control-preflight-complete",
        type=Path,
        required=True,
    )
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
    if resolved != absolute or not resolved.is_file() or resolved.is_symlink():
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


def _repo_directory(path_text: object, *, name: str) -> Path:
    if (
        not isinstance(path_text, str)
        or not path_text
        or Path(path_text).is_absolute()
    ):
        raise ValueError(f"{name} must be a non-empty repo-relative path")
    candidate = _ROOT / path_text
    if candidate.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    path = candidate.resolve(strict=True)
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"{name} must be a regular directory")
    if path.relative_to(_ROOT).as_posix() != path_text:
        raise RuntimeError(f"{name} does not resolve to its frozen path")
    return path


def _prepare_output(path: Path) -> Path:
    candidate = path.expanduser()
    absolute = Path(os.path.abspath(candidate))
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(
            f"bounded learnability output already exists: {absolute}"
        )
    for parent in (absolute.parent, *absolute.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError(
                "bounded learnability output may not traverse a symbolic link"
            )
    return absolute


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
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


def _fingerprinted(
    payload: Mapping[str, object],
    *,
    field: str = "receipt_fingerprint",
) -> dict[str, object]:
    value = dict(payload)
    if field in value:
        raise ValueError(f"payload already contains {field}")
    value[field] = stable_fingerprint(value)
    return value


def _verify_fingerprinted(
    payload: Mapping[str, Any],
    *,
    name: str,
    field: str,
) -> None:
    fingerprint = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if (
        not isinstance(fingerprint, str)
        or stable_fingerprint(unsigned) != fingerprint
    ):
        raise RuntimeError(f"{name} fingerprint is inconsistent")


def _load_config(path: Path) -> dict[str, Any]:
    expected = _ROOT / BOUNDED_CONFIG_REPO_PATH
    if path != expected:
        raise RuntimeError("bounded config path differs from the freeze")
    if file_sha256(path) != BOUNDED_CONFIG_FILE_SHA256:
        raise RuntimeError("bounded config is not the exact frozen file")
    config = pair_preflight_runner._strict_json(
        path,
        name="bounded learnability config",
    )
    _verify_fingerprinted(
        config,
        name="bounded learnability config",
        field="config_fingerprint",
    )
    if config.get("config_fingerprint") != BOUNDED_CONFIG_FINGERPRINT:
        raise RuntimeError("bounded config fingerprint differs from the freeze")
    if (
        config.get("schema_version")
        != "cure-lite-paired-bounded-learnability-config-v1"
        or config.get("dataset") != "IRSTD-1K"
        or config.get("split") != "D_R"
        or config.get("not_performance_evidence") is not True
    ):
        raise RuntimeError("bounded config identity differs from the freeze")
    budget = config.get("budget")
    optimization = config.get("optimization")
    if not isinstance(budget, Mapping) or not isinstance(
        optimization,
        Mapping,
    ):
        raise RuntimeError("bounded budget/optimization is malformed")
    expected_budget = {
        "epochs": 10,
        "steps_per_epoch": 40,
        "optimizer_updates": 400,
        "decoder_states_per_update": 12,
        "training_decoder_forward_calls": 1200,
        "training_decoder_state_evaluations": 4800,
        "total_decoder_forward_calls": 1210,
        "total_decoder_state_evaluations": 5056,
    }
    for field, value in expected_budget.items():
        if budget.get(field) != value:
            raise RuntimeError(f"bounded budget {field} differs from the freeze")
    if (
        optimization.get("optimizer") != "adam"
        or optimization.get("learning_rate") != 0.001
        or optimization.get("weight_decay") != 0.0
        or optimization.get("seed") != 42
    ):
        raise RuntimeError("bounded optimizer differs from the formal freeze")
    return config


def _verify_control_preflight(
    path: Path,
    contract: Mapping[str, object],
) -> dict[str, Any]:
    authority = _repo_file(
        contract.get("authority_complete_path"),
        name="matched-control authority COMPLETE",
    )
    replay = _repo_file(
        contract.get("replay_complete_path"),
        name="matched-control replay COMPLETE",
    )
    if path != authority:
        raise RuntimeError(
            "matched-control preflight must be the frozen r1 authority"
        )
    if (
        file_sha256(authority)
        != contract.get("authority_complete_file_sha256")
        or file_sha256(replay)
        != contract.get("replay_complete_file_sha256")
    ):
        raise RuntimeError("matched-control COMPLETE file changed")

    payload = pair_preflight_runner._strict_json(
        authority,
        name="matched-control preflight COMPLETE",
    )
    replay_payload = pair_preflight_runner._strict_json(
        replay,
        name="matched-control replay COMPLETE",
    )
    _verify_fingerprinted(
        payload,
        name="matched-control preflight COMPLETE",
        field="complete_fingerprint",
    )
    _verify_fingerprinted(
        replay_payload,
        name="matched-control replay COMPLETE",
        field="complete_fingerprint",
    )
    if (
        payload != replay_payload
        or payload.get("complete_fingerprint")
        != contract.get("authority_complete_fingerprint")
    ):
        raise RuntimeError(
            "matched-control r1/r2 COMPLETE identities are not frozen-identical"
        )

    receipt_relative = contract.get("run_receipt_relative_path")
    if (
        not isinstance(receipt_relative, str)
        or Path(receipt_relative).is_absolute()
    ):
        raise RuntimeError("matched-control run receipt path is malformed")
    authority_root = authority.parent
    replay_root = replay.parent
    authority_receipt = _canonical_file(
        authority_root / receipt_relative,
        name="matched-control authority run receipt",
    )
    replay_receipt = _canonical_file(
        replay_root / receipt_relative,
        name="matched-control replay run receipt",
    )
    if (
        authority_receipt.relative_to(authority_root).as_posix()
        != receipt_relative
        or replay_receipt.relative_to(replay_root).as_posix()
        != receipt_relative
    ):
        raise RuntimeError("matched-control run receipt escaped its artifact")
    if (
        file_sha256(authority_receipt)
        != contract.get("run_receipt_file_sha256")
        or file_sha256(replay_receipt)
        != contract.get("run_receipt_file_sha256")
    ):
        raise RuntimeError("matched-control run receipt changed")
    run_receipt = pair_preflight_runner._strict_json(
        authority_receipt,
        name="matched-control authority run receipt",
    )
    replay_run_receipt = pair_preflight_runner._strict_json(
        replay_receipt,
        name="matched-control replay run receipt",
    )
    _verify_fingerprinted(
        run_receipt,
        name="matched-control authority run receipt",
        field="receipt_fingerprint",
    )
    _verify_fingerprinted(
        replay_run_receipt,
        name="matched-control replay run receipt",
        field="receipt_fingerprint",
    )
    if (
        run_receipt != replay_run_receipt
        or run_receipt.get("receipt_fingerprint")
        != contract.get("run_receipt_fingerprint")
        or payload.get("run_receipt_fingerprint")
        != run_receipt.get("receipt_fingerprint")
    ):
        raise RuntimeError(
            "matched-control r1/r2 run receipts are not frozen-identical"
        )

    expected_receipts = {
        "control_contracts.json",
        "dct_basis.json",
        "run_receipt.json",
        "target_permutation.json",
    }
    for root, complete in (
        (authority_root, payload),
        (replay_root, replay_payload),
    ):
        if {item.name for item in root.iterdir()} != {
            "receipts",
            "COMPLETE.json",
        }:
            raise RuntimeError(
                "matched-control top-level artifact inventory changed"
            )
        receipts_root = root / "receipts"
        if (
            receipts_root.is_symlink()
            or not receipts_root.is_dir()
            or {item.name for item in receipts_root.iterdir()}
            != expected_receipts
        ):
            raise RuntimeError(
                "matched-control receipt inventory changed"
            )
        artifact_files = {
            item.relative_to(root).as_posix(): file_sha256(item)
            for item in sorted(receipts_root.iterdir())
        }
        if (
            complete.get("artifact_files") != artifact_files
            or complete.get("artifact_file_count") != len(artifact_files)
        ):
            raise RuntimeError(
                "matched-control artifact hashes or count changed"
            )
        for receipt_path in sorted(receipts_root.iterdir()):
            receipt = pair_preflight_runner._strict_json(
                receipt_path,
                name=f"matched-control receipt {receipt_path.name}",
            )
            _verify_fingerprinted(
                receipt,
                name=f"matched-control receipt {receipt_path.name}",
                field="receipt_fingerprint",
            )

    r1_files = {
        item.relative_to(authority_root).as_posix(): file_sha256(item)
        for item in sorted(authority_root.rglob("*"))
        if item.is_file()
    }
    r2_files = {
        item.relative_to(replay_root).as_posix(): file_sha256(item)
        for item in sorted(replay_root.rglob("*"))
        if item.is_file()
    }
    if (
        contract.get("require_byte_identical_r1_r2") is not True
        or r1_files != r2_files
    ):
        raise RuntimeError(
            "matched-control r1/r2 artifacts are not byte-identical"
        )

    expected = {
        "schema_version": contract.get("schema_version"),
        "execution_status": contract.get("required_execution_status"),
        "status": contract.get("required_status"),
        "target_permutation_status": contract.get(
            "required_target_permutation_status"
        ),
        "paired_protocol_fingerprint": contract.get(
            "required_paired_protocol_fingerprint"
        ),
        "pair_catalog_fingerprint": contract.get(
            "required_pair_catalog_fingerprint"
        ),
        "training_performed": contract.get("training_performed"),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(
                f"matched-control preflight {field} differs from the contract"
            )
    complete_gate = contract.get("required_complete_gate")
    if (
        not isinstance(complete_gate, str)
        or payload.get(complete_gate) is not True
    ):
        raise RuntimeError(
            "matched-control COMPLETE static preflight gate has not passed"
        )
    expected_run = {
        "schema_version": contract.get("run_receipt_schema_version"),
        "execution_status": contract.get(
            "required_run_receipt_execution_status"
        ),
        "paired_protocol_fingerprint": contract.get(
            "required_paired_protocol_fingerprint"
        ),
        "pair_catalog_fingerprint": contract.get(
            "required_pair_catalog_fingerprint"
        ),
        "split": "D_R",
    }
    for field, value in expected_run.items():
        if run_receipt.get(field) != value:
            raise RuntimeError(
                "matched-control run receipt "
                f"{field} differs from the contract"
            )
    gate = contract.get("required_run_receipt_gate")
    static_gate = contract.get("required_run_receipt_static_gate")
    gates = run_receipt.get("gates")
    if (
        not isinstance(gate, str)
        or not isinstance(static_gate, str)
        or not isinstance(gates, Mapping)
        or gates.get(gate) is not True
        or gates.get(static_gate) is not True
        or gates.get("target_permutation_status")
        != contract.get("required_target_permutation_status")
    ):
        raise RuntimeError("matched-control static preflight has not passed")
    policy = run_receipt.get("execution_policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("training_performed")
        is not contract.get("training_performed")
        or payload.get("training_performed")
        is not contract.get("training_performed")
        or policy.get("d_v_accessed") is not False
        or policy.get("d_t_accessed") is not False
    ):
        raise RuntimeError(
            "matched-control preflight execution boundary changed"
        )
    return {
        **payload,
        "run_receipt_file_sha256": file_sha256(authority_receipt),
        "run_receipt_fingerprint_verified": run_receipt[
            "receipt_fingerprint"
        ],
        "byte_identical_replay_verified": True,
    }


def _control_artifact_hashes(
    contract: Mapping[str, object],
) -> dict[str, str]:
    """Return the exact frozen r1/r2 control files for mid-run rechecks."""

    roots = (
        _repo_file(
            contract.get("authority_complete_path"),
            name="matched-control authority COMPLETE",
        ).parent,
        _repo_file(
            contract.get("replay_complete_path"),
            name="matched-control replay COMPLETE",
        ).parent,
    )
    return {
        item.relative_to(_ROOT).as_posix(): file_sha256(item)
        for root in roots
        for item in sorted(root.rglob("*"))
        if item.is_file()
    }


def _verify_formal_optimizer(
    config: Mapping[str, Any],
    path: Path,
) -> None:
    binding = config["input_binding"]
    if not isinstance(binding, Mapping):
        raise RuntimeError("bounded input binding is malformed")
    if file_sha256(path) != binding["formal_seed42_config_file_sha256"]:
        raise RuntimeError("formal seed-42 config changed")
    formal = pair_preflight_runner._strict_json(
        path,
        name="formal seed-42 config",
    )
    optimization = config["optimization"]
    expected = {
        "optimizer": formal["training"]["optimizer"],
        "learning_rate": formal["training"]["learning_rate"],
        "weight_decay": formal["training"]["weight_decay"],
        "seed": formal["training"]["global_seed"],
        "decoder": formal["training"]["decoder_config"],
        "loss": formal["training"]["loss_config"],
    }
    for field, value in expected.items():
        if optimization[field] != value:
            raise RuntimeError(
                f"bounded optimization {field} differs from formal seed 42"
            )


def _implementation_binding() -> dict[str, str]:
    paths = (
        _ROOT / "tools" / "run_paired_bounded_learnability.py",
        _ROOT / "tools" / "run_paired_preflight.py",
        _ROOT / "cure_lite" / "cache" / "schema.py",
        _ROOT / "cure_lite" / "config.py",
        _ROOT / "cure_lite" / "sampling.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "paired_bounded_learnability.py",
        _ROOT / "cure_lite" / "experiment" / "artifacts.py",
        _ROOT / "cure_lite" / "experiment" / "cache_pipeline.py",
        _ROOT / "cure_lite" / "experiment" / "paired_catalog.py",
        _ROOT / "cure_lite" / "experiment" / "paired_preflight.py",
        _ROOT / "cure_lite" / "experiment" / "paired_exposure.py",
        _ROOT / "cure_lite" / "experiment" / "geometry_safe_catalog.py",
        _ROOT / "cure_lite" / "experiment" / "training_pipeline.py",
        _ROOT / "cure_lite" / "train" / "paired_step.py",
        _ROOT / "cure_lite" / "train" / "paired_pools.py",
        _ROOT / "cure_lite" / "train" / "pools.py",
        _ROOT / "cure_lite" / "train" / "step.py",
        _ROOT / "cure_lite" / "paired_types.py",
        _ROOT / "cure_lite" / "paired_losses.py",
        _ROOT / "cure_lite" / "decoder.py",
        _ROOT / "cure_lite" / "losses.py",
        _ROOT / "cure_lite" / "model.py",
    )
    bounded = {
        path.relative_to(_ROOT).as_posix(): file_sha256(path)
        for path in paths
    }
    upstream = pair_preflight_runner._implementation_binding()
    upstream.update(bounded)
    return dict(sorted(upstream.items()))


def _load_real_catalog(
    config: Mapping[str, Any],
) -> tuple[object, object, object, dict[str, str]]:
    binding = config["input_binding"]
    if not isinstance(binding, Mapping):
        raise RuntimeError("bounded input binding is malformed")
    paths = {
        name: _repo_file(binding[f"{name}_path"], name=name)
        for name in (
            "paired_protocol",
            "formal_seed42_config",
            "manifest",
            "state_index",
            "geometry_config",
            "geometry_catalog",
            "p0_a1",
            "eligible_view",
            "geometry_complete",
        )
    }
    paths["real_pair_preflight"] = _repo_directory(
        binding["real_pair_preflight_path"],
        name="real_pair_preflight",
    )
    expected_sha = {
        "paired_protocol": binding["paired_protocol_file_sha256"],
        "real_pair_preflight": binding[
            "real_pair_preflight_complete_file_sha256"
        ],
        "formal_seed42_config": binding[
            "formal_seed42_config_file_sha256"
        ],
        "manifest": binding["manifest_file_sha256"],
        "state_index": binding["state_index_file_sha256"],
        "geometry_config": binding["geometry_config_file_sha256"],
        "geometry_catalog": binding["geometry_catalog_file_sha256"],
        "p0_a1": binding["p0_a1_file_sha256"],
        "eligible_view": binding["eligible_view_file_sha256"],
        "geometry_complete": binding["geometry_complete_file_sha256"],
    }
    # The paired preflight binding names a directory; bind its COMPLETE file.
    actual_sha_paths = dict(paths)
    actual_sha_paths["real_pair_preflight"] = (
        paths["real_pair_preflight"] / "COMPLETE.json"
    )
    for name, path in actual_sha_paths.items():
        if file_sha256(path) != expected_sha[name]:
            raise RuntimeError(f"frozen bounded input changed: {name}")

    pair_preflight = pair_preflight_runner.load_paired_run_artifact(
        paths["real_pair_preflight"]
    )
    if (
        pair_preflight.complete_fingerprint
        != binding["real_pair_preflight_complete_fingerprint"]
        or pair_preflight.catalog_fingerprint
        != binding["real_pair_catalog_fingerprint"]
    ):
        raise RuntimeError("real paired preflight identity changed")
    pair_protocol = pair_preflight_runner._load_paired_protocol(
        paths["paired_protocol"]
    )
    if (
        pair_protocol["receipt_fingerprint"]
        != binding["paired_protocol_fingerprint"]
    ):
        raise RuntimeError("paired protocol binding changed")
    _verify_formal_optimizer(config, paths["formal_seed42_config"])

    geometry_protocol = load_geometry_catalog_protocol(
        paths["geometry_config"]
    )
    (
        upstream_geometry,
        upstream_a1,
        upstream_view,
        _,
    ) = pair_preflight_runner._upstream_binding(
        pair_protocol,
        geometry_catalog_path=paths["geometry_catalog"],
        p0_a1_path=paths["p0_a1"],
        eligible_view_path=paths["eligible_view"],
        geometry_complete_path=paths["geometry_complete"],
        geometry_protocol=geometry_protocol,
    )
    manifest = load_and_validate_manifest(paths["manifest"])
    state_index = pair_preflight_runner._strict_json(
        paths["state_index"],
        name="D_R state index",
    )
    preprocess = pair_preflight_runner._verify_geometry_input_binding(
        geometry_protocol,
        paths["manifest"],
        paths["state_index"],
        state_index,
    )
    dataset = ManifestImageDataset(
        manifest,
        "D_R",
        preprocess,
        manifest_path=paths["manifest"],
    )
    bundle = load_d_r_cache_bundle(
        paths["state_index"],
        dataset,
        expected_base_fingerprint=(
            geometry_protocol.input_binding.base_fingerprint
        ),
    )
    sources = tuple(
        CachedTrainingSource(
            row.sample_id,
            row.base_output.feature,
            row.base_output.probability,
            row.state,
        )
        for row in bundle.rows
    )
    prepared = prepare_training_catalog(
        sources,
        occupancy_config=bundle.occupancy_config,
        match_config=bundle.match_config,
        intervention_config=bundle.intervention_config,
    )
    geometry = build_geometry_safe_catalog(
        bundle,
        prepared,
        manifest,
        geometry_protocol,
    )
    if pair_preflight_runner._fingerprinted(
        geometry.canonical_payload()
    ) != upstream_geometry:
        raise RuntimeError("reconstructed geometry catalog changed")
    if pair_preflight_runner._fingerprinted(
        build_p0_a1_receipt(
            geometry,
            geometry_protocol,
            a0_receipt_fingerprint=upstream_a1[
                "a0_receipt_fingerprint"
            ],
        )
    ) != upstream_a1:
        raise RuntimeError("reconstructed P0-A1 receipt changed")
    view = build_geometry_safe_p0_view(prepared, geometry)
    if pair_preflight_runner._reconstructed_eligible_view_receipt(
        geometry,
        view,
        str(upstream_a1["eligible_catalog_fingerprint"]),
    ) != upstream_view:
        raise RuntimeError("reconstructed eligible view changed")
    pair_catalog = build_pair_catalog(
        prepared,
        geometry,
        manifest,
        paired_protocol_fingerprint=binding[
            "paired_protocol_fingerprint"
        ],
        match_config=bundle.match_config,
    )
    if pair_catalog.catalog_fingerprint != binding[
        "real_pair_catalog_fingerprint"
    ]:
        raise RuntimeError("reconstructed pair catalog fingerprint changed")
    persisted_manifest = pair_preflight_runner._strict_json(
        paths["real_pair_preflight"]
        / "pair_preflight"
        / "pair_catalog_manifest.json",
        name="persisted pair manifest",
    )
    if (
        file_sha256(
            paths["real_pair_preflight"]
            / "pair_preflight"
            / "pair_catalog_manifest.json"
        )
        != binding["real_pair_manifest_file_sha256"]
        or persisted_manifest.get("canonical_pair_catalog")
        != pair_catalog.canonical_payload()
    ):
        raise RuntimeError("reconstructed pair catalog differs from preflight")
    immutable = {
        str(path): file_sha256(path)
        for path in (
            *(
                value
                for key, value in paths.items()
                if key != "real_pair_preflight"
            ),
            paths["real_pair_preflight"] / "COMPLETE.json",
            paths["real_pair_preflight"]
            / "pair_preflight"
            / "pair_catalog_manifest.json",
        )
    }
    bundle.verify_unchanged()
    return pair_catalog, prepared, bundle, immutable


def _decision(
    result: Mapping[str, object] | None,
    *,
    failure: Mapping[str, object] | None,
    evidence_receipt_fingerprint: str,
) -> dict[str, object]:
    if result is not None:
        structural = result["structural_execution_pass"] is True
        computational = (
            result["computational_learnability_pass"] is True
        )
        status = (
            "COMPUTATIONAL_LEARNABILITY_PASS"
            if computational
            else "COMPUTATIONAL_LEARNABILITY_FAIL"
            if structural
            else "STRUCTURAL_EXECUTION_FAIL"
        )
    else:
        structural = False
        computational = False
        status = "STRUCTURAL_EXECUTION_ERROR"
    return _fingerprinted(
        {
            "schema_version": BOUNDED_DECISION_SCHEMA,
            "status": status,
            "structural_execution_pass": structural,
            "computational_learnability_pass": computational,
            "not_performance_evidence": True,
            "authorizes_formal_800": False,
            "authorizes_D_V_or_D_T": False,
            "threshold_or_budget_changed_after_result": False,
            "evidence_kind": "result" if result is not None else "failure",
            "evidence_receipt_fingerprint": (
                evidence_receipt_fingerprint
            ),
            "failure": dict(failure) if failure is not None else None,
            "next_route": (
                "review_bounded_evidence_without_threshold_or_budget_change"
            ),
        }
    )


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {_INCOMPLETE, "COMPLETE.json"}
    }


@dataclass(frozen=True)
class PublishedBoundedLearnability:
    """Fully verified identity of one sealed bounded learnability artifact."""

    root: Path
    decision: str
    structural_execution_pass: bool
    computational_learnability_pass: bool
    pair_catalog_fingerprint: str
    micro_population_fingerprint: str
    schedule_fingerprint: str
    complete_fingerprint: str

    def verify_unchanged(self) -> None:
        if load_bounded_learnability_artifact(self.root) != self:
            raise RuntimeError(
                "published bounded learnability artifact identity changed"
            )


def _canonical_micro_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    fields = (
        "schema_version",
        "seed",
        "pair_catalog_fingerprint",
        "prepared_catalog_fingerprint",
        "selection_rule",
        "clean_pairs",
        "factual_miss",
        "factual_no_miss",
        "component_null",
        "identity_null",
    )
    return {field: payload.get(field) for field in fields}


def _canonical_schedule_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    fields = (
        "schema_version",
        "optimizer_updates",
        "steps_per_epoch",
        "pair_indices",
        "factual_miss_indices",
        "factual_no_miss_indices",
        "pair_counts",
        "factual_miss_counts",
        "factual_no_miss_counts",
    )
    return {field: payload.get(field) for field in fields}


def load_bounded_learnability_artifact(
    output_dir: str | Path,
) -> PublishedBoundedLearnability:
    """Load and cross-check every file in one completed bounded run."""

    root = Path(output_dir).expanduser().resolve(strict=False)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(
            "bounded learnability root must be a regular directory"
        )
    if (root / _INCOMPLETE).exists():
        raise RuntimeError("bounded learnability publication is incomplete")
    if {item.name for item in root.iterdir()} != {
        "receipts",
        "COMPLETE.json",
    }:
        raise RuntimeError(
            "bounded learnability top-level inventory changed"
        )
    receipts = root / "receipts"
    if receipts.is_symlink() or not receipts.is_dir():
        raise ValueError("bounded receipt directory must be regular")
    names = {item.name for item in receipts.iterdir()}
    common = {
        "config_binding.json",
        "micro_population.json",
        "schedule.json",
        "decision.json",
    }
    if names not in (
        common | {"result.json"},
        common | {"failure.json"},
    ):
        raise RuntimeError("bounded learnability receipt inventory changed")

    complete = pair_preflight_runner._strict_json(
        root / "COMPLETE.json",
        name="bounded COMPLETE",
    )
    config_binding = pair_preflight_runner._strict_json(
        receipts / "config_binding.json",
        name="bounded config binding",
    )
    micro = pair_preflight_runner._strict_json(
        receipts / "micro_population.json",
        name="bounded micro population",
    )
    schedule = pair_preflight_runner._strict_json(
        receipts / "schedule.json",
        name="bounded schedule",
    )
    decision = pair_preflight_runner._strict_json(
        receipts / "decision.json",
        name="bounded decision",
    )
    evidence_name = (
        "result.json" if "result.json" in names else "failure.json"
    )
    evidence = pair_preflight_runner._strict_json(
        receipts / evidence_name,
        name=f"bounded {evidence_name[:-5]}",
    )
    _verify_fingerprinted(
        complete,
        name="bounded COMPLETE",
        field="complete_fingerprint",
    )
    for payload, name in (
        (config_binding, "bounded config binding"),
        (micro, "bounded micro population"),
        (schedule, "bounded schedule"),
        (decision, "bounded decision"),
        (evidence, f"bounded {evidence_name[:-5]}"),
    ):
        _verify_fingerprinted(
            payload,
            name=name,
            field="receipt_fingerprint",
        )
    if (
        complete.get("artifact_files") != _artifact_hashes(root)
        or complete.get("artifact_file_count") != len(names)
    ):
        raise RuntimeError("bounded artifact hashes or count changed")

    if (
        complete.get("schema_version") != BOUNDED_RUN_SCHEMA
        or complete.get("execution_status") != "complete"
        or complete.get("split") != "D_R"
        or complete.get("not_performance_evidence") is not True
        or complete.get("authorizes_formal_800") is not False
        or complete.get("formal_training_performed") is not False
        or complete.get("D_V_accessed") is not False
        or complete.get("D_T_accessed") is not False
        or complete.get("calibration_performed") is not False
        or complete.get("performance_evaluation_performed") is not False
        or complete.get("deterministic_runtime_contract_required") is not True
        or complete.get(
            "exact_replay_required_under_same_frozen_environment"
        )
        is not True
    ):
        raise RuntimeError("bounded COMPLETE execution boundary changed")

    embedded_config = config_binding.get("config")
    if not isinstance(embedded_config, Mapping):
        raise RuntimeError("bounded embedded config is malformed")
    _verify_fingerprinted(
        embedded_config,
        name="bounded embedded config",
        field="config_fingerprint",
    )
    if (
        config_binding.get("schema_version")
        != "cure-lite-paired-bounded-config-binding-v1"
        or embedded_config.get("config_fingerprint")
        != BOUNDED_CONFIG_FINGERPRINT
        or config_binding.get("config_file_sha256")
        != BOUNDED_CONFIG_FILE_SHA256
        or complete.get("config_fingerprint")
        != BOUNDED_CONFIG_FINGERPRINT
        or complete.get("config_binding_fingerprint")
        != config_binding.get("receipt_fingerprint")
    ):
        raise RuntimeError("bounded config binding changed")
    runtime = config_binding.get("runtime")
    contract = embedded_config.get("control_preflight_contract")
    if not isinstance(runtime, Mapping) or not isinstance(
        contract,
        Mapping,
    ):
        raise RuntimeError("bounded runtime/control binding is malformed")
    if (
        runtime.get("allowed_split") != "D_R"
        or runtime.get("cublas_workspace_config") != ":4096:8"
        or config_binding.get(
            "control_preflight_byte_identical_replay_verified"
        )
        is not True
        or config_binding.get(
            "control_preflight_complete_fingerprint"
        )
        != contract.get("authority_complete_fingerprint")
        or complete.get("control_preflight_complete_fingerprint")
        != contract.get("authority_complete_fingerprint")
        or config_binding.get(
            "control_preflight_run_receipt_fingerprint"
        )
        != contract.get("run_receipt_fingerprint")
    ):
        raise RuntimeError("bounded matched-control binding changed")

    micro_fingerprint = micro.get("population_fingerprint")
    if (
        micro.get("schema_version") != BOUNDED_MICRO_POPULATION_SCHEMA
        or not isinstance(micro_fingerprint, str)
        or stable_fingerprint(_canonical_micro_payload(micro))
        != micro_fingerprint
        or complete.get("micro_population_fingerprint")
        != micro_fingerprint
        or complete.get("micro_population_receipt_fingerprint")
        != micro.get("receipt_fingerprint")
        or micro.get("pair_catalog_fingerprint")
        != complete.get("pair_catalog_fingerprint")
    ):
        raise RuntimeError("bounded micro-population binding changed")
    schedule_fingerprint = schedule.get("schedule_fingerprint")
    if (
        schedule.get("schema_version") != BOUNDED_MICRO_SCHEDULE_SCHEMA
        or not isinstance(schedule_fingerprint, str)
        or stable_fingerprint(_canonical_schedule_payload(schedule))
        != schedule_fingerprint
        or complete.get("schedule_fingerprint") != schedule_fingerprint
        or complete.get("schedule_receipt_fingerprint")
        != schedule.get("receipt_fingerprint")
    ):
        raise RuntimeError("bounded schedule binding changed")
    exposure = schedule.get("exposure")
    if not isinstance(exposure, Mapping) or any(
        exposure.get(name) != schedule.get(name)
        for name in (
            "pair_counts",
            "factual_miss_counts",
            "factual_no_miss_counts",
        )
    ):
        raise RuntimeError("bounded schedule exposure ledger changed")

    evidence_kind = "result" if evidence_name == "result.json" else "failure"
    if (
        decision.get("schema_version") != BOUNDED_DECISION_SCHEMA
        or decision.get("evidence_kind") != evidence_kind
        or decision.get("evidence_receipt_fingerprint")
        != evidence.get("receipt_fingerprint")
        or complete.get("evidence_kind") != evidence_kind
        or complete.get("evidence_receipt_fingerprint")
        != evidence.get("receipt_fingerprint")
        or complete.get("decision_fingerprint")
        != decision.get("receipt_fingerprint")
        or complete.get("decision") != decision.get("status")
        or complete.get("structural_execution_pass")
        is not decision.get("structural_execution_pass")
        or complete.get("computational_learnability_pass")
        is not decision.get("computational_learnability_pass")
    ):
        raise RuntimeError("bounded decision/evidence cross-binding changed")

    if evidence_kind == "result":
        expected_status = (
            "COMPUTATIONAL_LEARNABILITY_PASS"
            if evidence.get("computational_learnability_pass") is True
            else "COMPUTATIONAL_LEARNABILITY_FAIL"
            if evidence.get("structural_execution_pass") is True
            else "STRUCTURAL_EXECUTION_FAIL"
        )
        if (
            evidence.get("schema_version") != BOUNDED_EXECUTION_SCHEMA
            or evidence.get("execution_status") != "completed"
            or evidence.get("population_fingerprint")
            != micro_fingerprint
            or evidence.get("schedule_fingerprint")
            != schedule_fingerprint
            or decision.get("failure") is not None
            or decision.get("status") != expected_status
            or decision.get("structural_execution_pass")
            is not evidence.get("structural_execution_pass")
            or decision.get("computational_learnability_pass")
            is not evidence.get("computational_learnability_pass")
        ):
            raise RuntimeError("bounded result decision changed")
        deterministic_runtime = evidence.get("deterministic_runtime")
        if (
            not isinstance(deterministic_runtime, Mapping)
            or deterministic_runtime.get("contract_satisfied") is not True
            or deterministic_runtime.get("flags_restored_after_execution")
            is not True
        ):
            raise RuntimeError(
                "bounded deterministic runtime evidence changed"
            )
    else:
        unsigned_failure = dict(evidence)
        unsigned_failure.pop("receipt_fingerprint", None)
        if (
            evidence.get("schema_version")
            != "cure-lite-paired-bounded-execution-failure-v1"
            or decision.get("status") != "STRUCTURAL_EXECUTION_ERROR"
            or decision.get("failure") != unsigned_failure
            or decision.get("structural_execution_pass") is not False
            or decision.get("computational_learnability_pass") is not False
        ):
            raise RuntimeError("bounded sealed failure decision changed")

    if (
        decision.get("not_performance_evidence") is not True
        or decision.get("authorizes_formal_800") is not False
        or decision.get("authorizes_D_V_or_D_T") is not False
        or decision.get("threshold_or_budget_changed_after_result") is not False
    ):
        raise RuntimeError("bounded decision boundary changed")
    return PublishedBoundedLearnability(
        root=root,
        decision=str(decision["status"]),
        structural_execution_pass=bool(
            decision["structural_execution_pass"]
        ),
        computational_learnability_pass=bool(
            decision["computational_learnability_pass"]
        ),
        pair_catalog_fingerprint=str(
            complete["pair_catalog_fingerprint"]
        ),
        micro_population_fingerprint=str(micro_fingerprint),
        schedule_fingerprint=str(schedule_fingerprint),
        complete_fingerprint=str(complete["complete_fingerprint"]),
    )


def _bind_runner_structural_gates(
    result: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, object]:
    """Combine core execution checks with runner-only frozen input gates."""

    checks_value = result.get("structural_checks")
    gates_value = config.get("gates")
    if not isinstance(checks_value, Mapping) or not isinstance(
        gates_value,
        Mapping,
    ):
        raise RuntimeError("bounded structural gate payload is malformed")
    expected_value = gates_value.get("structural_execution")
    computational_value = result.get("computational_gates")
    if not isinstance(expected_value, Mapping) or not isinstance(
        computational_value,
        Mapping,
    ):
        raise RuntimeError("bounded frozen structural gates are malformed")
    if any(value is not True for value in expected_value.values()):
        raise RuntimeError(
            "bounded structural gate specification must require every check"
        )
    checks = dict(checks_value)
    checks["all_input_fingerprints_verified"] = True
    checks["control_preflight_verified"] = True
    if set(checks) != set(expected_value):
        raise RuntimeError(
            "bounded structural gate implementation differs from the freeze"
        )
    structural = all(checks[name] is True for name in sorted(checks))
    computational = (
        structural and computational_value.get("all_pass") is True
    )
    return {
        **dict(result),
        "structural_checks": checks,
        "structural_execution_pass": structural,
        "computational_learnability_pass": computational,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    config_path = _canonical_file(args.config, name="bounded config")
    config = _load_config(config_path)
    output = _prepare_output(args.output)
    control_path = _canonical_file(
        args.control_preflight_complete,
        name="matched-control preflight COMPLETE",
    )
    control_contract = config["control_preflight_contract"]
    if not isinstance(control_contract, Mapping):
        raise RuntimeError("control preflight contract is malformed")
    control = _verify_control_preflight(control_path, control_contract)
    control_artifact_hashes = _control_artifact_hashes(control_contract)

    implementation = _implementation_binding()
    pair_catalog, prepared, bundle, immutable = _load_real_catalog(config)
    immutable[str(config_path)] = file_sha256(config_path)
    immutable.update(
        {
            str(_ROOT / relative): digest
            for relative, digest in control_artifact_hashes.items()
        }
    )
    population = build_bounded_micro_population(
        pair_catalog,
        prepared,
        config["micro_population"],
    )
    schedule = build_bounded_micro_schedule(
        population,
        config["budget"],
    )

    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / _INCOMPLETE
    incomplete.open("xb").close()
    receipts = output / "receipts"
    receipts.mkdir(exist_ok=False)
    config_binding = _fingerprinted(
        {
            "schema_version": (
                "cure-lite-paired-bounded-config-binding-v1"
            ),
            "config": config,
            "config_file_sha256": file_sha256(config_path),
            "control_preflight_complete_file_sha256": file_sha256(
                control_path
            ),
            "control_preflight_complete_fingerprint": control[
                "complete_fingerprint"
            ],
            "control_preflight_run_receipt_file_sha256": control[
                "run_receipt_file_sha256"
            ],
            "control_preflight_run_receipt_fingerprint": control[
                "run_receipt_fingerprint_verified"
            ],
            "control_preflight_byte_identical_replay_verified": control[
                "byte_identical_replay_verified"
            ],
            "control_preflight_artifact_files": control_artifact_hashes,
            "implementation_files": implementation,
            "runtime": {
                "device": args.device,
                "allowed_split": "D_R",
                "cublas_workspace_config": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"
                ),
            },
        }
    )
    _write_new_json(receipts / "config_binding.json", config_binding)
    micro_receipt = _fingerprinted(
        {
            **population.canonical_payload(),
            "population_fingerprint": population.population_fingerprint,
        }
    )
    _write_new_json(receipts / "micro_population.json", micro_receipt)
    schedule_receipt = _fingerprinted(
        {
            **schedule.canonical_payload(),
            "schedule_fingerprint": schedule.schedule_fingerprint,
            "exposure": {
                "pair_counts": list(schedule.pair_counts),
                "factual_miss_counts": list(
                    schedule.factual_miss_counts
                ),
                "factual_no_miss_counts": list(
                    schedule.factual_no_miss_counts
                ),
            },
        }
    )
    _write_new_json(receipts / "schedule.json", schedule_receipt)

    result: dict[str, object] | None = None
    failure: dict[str, object] | None = None
    evidence_receipt: dict[str, object]
    execution_error: Exception | None = None
    try:
        result = execute_bounded_learnability(
            population,
            schedule,
            config,
            device=args.device,
        )
    except Exception as error:
        execution_error = error

    try:
        bundle.verify_unchanged()
        if any(
            file_sha256(Path(path)) != digest
            for path, digest in immutable.items()
        ):
            raise RuntimeError(
                "a frozen bounded input changed during execution"
            )
        if _implementation_binding() != implementation:
            raise RuntimeError(
                "bounded learnability implementation changed during execution"
            )
    except Exception as error:
        if execution_error is None:
            execution_error = error

    if execution_error is None:
        try:
            if result is None:
                raise RuntimeError("bounded execution returned no evidence")
            result = _bind_runner_structural_gates(result, config)
            evidence_receipt = _fingerprinted(result)
            # Serialize before file creation so a non-JSON value cannot leave
            # a misleading partial result next to a sealed failure.
            json.dumps(evidence_receipt, allow_nan=False)
        except Exception as error:
            execution_error = error
    if execution_error is None:
        _write_new_json(receipts / "result.json", evidence_receipt)
    else:
        result = None
        failure = {
            "schema_version": (
                "cure-lite-paired-bounded-execution-failure-v1"
            ),
            "exception_type": type(execution_error).__name__,
            "message": str(execution_error),
            "structural_execution_pass": False,
            "computational_learnability_pass": False,
            "threshold_or_budget_changed": False,
        }
        evidence_receipt = _fingerprinted(failure)
        json.dumps(evidence_receipt, allow_nan=False)
        _write_new_json(receipts / "failure.json", evidence_receipt)
    decision = _decision(
        result,
        failure=failure,
        evidence_receipt_fingerprint=str(
            evidence_receipt["receipt_fingerprint"]
        ),
    )
    _write_new_json(receipts / "decision.json", decision)

    artifact_files = _artifact_hashes(output)
    complete = _fingerprinted(
        {
            "schema_version": BOUNDED_RUN_SCHEMA,
            "execution_status": "complete",
            "decision": decision["status"],
            "structural_execution_pass": decision[
                "structural_execution_pass"
            ],
            "computational_learnability_pass": decision[
                "computational_learnability_pass"
            ],
            "not_performance_evidence": True,
            "authorizes_formal_800": False,
            "split": "D_R",
            "pair_catalog_fingerprint": (
                pair_catalog.catalog_fingerprint
            ),
            "config_fingerprint": config["config_fingerprint"],
            "config_binding_fingerprint": config_binding[
                "receipt_fingerprint"
            ],
            "control_preflight_complete_fingerprint": control[
                "complete_fingerprint"
            ],
            "micro_population_fingerprint": (
                population.population_fingerprint
            ),
            "micro_population_receipt_fingerprint": micro_receipt[
                "receipt_fingerprint"
            ],
            "schedule_fingerprint": schedule.schedule_fingerprint,
            "schedule_receipt_fingerprint": schedule_receipt[
                "receipt_fingerprint"
            ],
            "evidence_kind": decision["evidence_kind"],
            "evidence_receipt_fingerprint": evidence_receipt[
                "receipt_fingerprint"
            ],
            "decision_fingerprint": decision["receipt_fingerprint"],
            "artifact_files": artifact_files,
            "artifact_file_count": len(artifact_files),
            "training_scope": "fresh_decoder_only_bounded_400_updates",
            "formal_training_performed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "performance_evaluation_performed": False,
            "deterministic_runtime_contract_required": True,
            "exact_replay_required_under_same_frozen_environment": True,
        },
        field="complete_fingerprint",
    )
    _write_new_json(output / "COMPLETE.json", complete)
    incomplete.unlink()
    published = load_bounded_learnability_artifact(output)
    return {
        "output": str(output),
        "decision": published.decision,
        "structural_execution_pass": (
            published.structural_execution_pass
        ),
        "computational_learnability_pass": (
            published.computational_learnability_pass
        ),
        "micro_population_fingerprint": (
            population.population_fingerprint
        ),
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "complete_fingerprint": complete["complete_fingerprint"],
        "not_performance_evidence": True,
        "formal_training_performed": False,
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
    if result["computational_learnability_pass"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
