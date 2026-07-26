#!/usr/bin/env python3
"""Run the frozen D_R-only paired spatial-tail companion diagnostic.

The command rebuilds the exact bounded micro-population and schedule, replays
the sealed 400-update proposed run from a fresh decoder, verifies exact
agreement with the existing r1/r2 bounded authority, and writes descriptive
spatial-tail metrics.  It never recovers a checkpoint, changes an existing
gate, accesses D_V/D_T, evaluates detection performance, or authorizes the
formal 800-epoch run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.experiment.paired_bounded_learnability import (  # noqa: E402
    build_bounded_micro_population,
    build_bounded_micro_schedule,
)
from cure_lite.experiment.paired_spatial_tail_diagnostic import (  # noqa: E402
    SPATIAL_TAIL_EXECUTION_SCHEMA,
    execute_spatial_tail_replay,
    validate_spatial_tail_specification,
)
from tools import run_paired_bounded_learnability as bounded_runner  # noqa: E402
from tools import run_paired_preflight as pair_preflight_runner  # noqa: E402


SPATIAL_CONFIG_REPO_PATH = (
    "protocols/IRSTD-1K/paired_spatial_tail_diagnostic_v1/config.json"
)
SPATIAL_CONFIG_FILE_SHA256 = (
    "b83336e4fa820ca1821a9e8da9d1018b75a0ffeb6be20a4fd6bdcc1f06da3747"
)
SPATIAL_CONFIG_FINGERPRINT = (
    "16d86d98833daadf9a13e85ec26809ddf97fb266673fe280c2bc6180d04b413c"
)
SPATIAL_RUN_SCHEMA = "cure-lite-paired-spatial-tail-diagnostic-run-v1"
SPATIAL_DECISION_SCHEMA = (
    "cure-lite-paired-spatial-tail-diagnostic-decision-v1"
)
SPATIAL_CONFIG_BINDING_SCHEMA = (
    "cure-lite-paired-spatial-tail-config-binding-v1"
)
_ROOT = Path(__file__).resolve().parents[1]
_INCOMPLETE = ".incomplete"


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
            f"spatial-tail diagnostic output already exists: {absolute}"
        )
    for parent in (absolute.parent, *absolute.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError(
                "spatial-tail output may not traverse a symbolic link"
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
    expected = _ROOT / SPATIAL_CONFIG_REPO_PATH
    if path != expected:
        raise RuntimeError("spatial-tail config path differs from freeze")
    if file_sha256(path) != SPATIAL_CONFIG_FILE_SHA256:
        raise RuntimeError("spatial-tail config is not the frozen file")
    config = pair_preflight_runner._strict_json(
        path,
        name="spatial-tail config",
    )
    _verify_fingerprinted(
        config,
        name="spatial-tail config",
        field="config_fingerprint",
    )
    if (
        config.get("config_fingerprint") != SPATIAL_CONFIG_FINGERPRINT
        or config.get("schema_version")
        != "cure-lite-paired-spatial-tail-diagnostic-config-v1"
        or config.get("dataset") != "IRSTD-1K"
        or config.get("split") != "D_R"
    ):
        raise RuntimeError("spatial-tail config identity differs from freeze")
    validate_spatial_tail_specification(config["diagnostic"])
    replay = config.get("replay_contract")
    decision = config.get("decision_semantics")
    execution = config.get("execution_policy")
    if not all(
        isinstance(value, Mapping)
        for value in (replay, decision, execution)
    ):
        raise RuntimeError("spatial-tail config sections are malformed")
    if (
        replay.get("optimizer_updates") != 400
        or replay.get("steps_per_epoch") != 40
        or replay.get("seed") != 42
        or replay.get("checkpoint_recovery") is not False
        or replay.get("checkpoint_persistence") is not False
        or replay.get("budget_extension") is not False
        or decision.get("retroactive_bounded_gate_added") is not False
        or decision.get("bounded_decision_may_change") is not False
        or decision.get("authorizes_formal_800") is not False
        or decision.get("authorizes_D_V_or_D_T") is not False
        or decision.get("is_performance_evidence") is not False
        or execution.get("allowed_runtime_splits") != ["D_R"]
        or execution.get("allow_D_V") is not False
        or execution.get("allow_D_T") is not False
        or execution.get("allow_formal_800") is not False
        or execution.get("overwrite_existing_bounded_artifacts") is not False
    ):
        raise RuntimeError("spatial-tail execution boundary differs from freeze")
    return config


def _directory_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _load_bounded_authority(
    config: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
    dict[str, str],
]:
    binding = config.get("input_binding")
    if not isinstance(binding, Mapping):
        raise RuntimeError("spatial-tail input binding is malformed")
    authority_root = _repo_directory(
        binding.get("authority_root_path"),
        name="bounded authority root",
    )
    replay_root = _repo_directory(
        binding.get("replay_root_path"),
        name="bounded replay root",
    )
    authority_complete = authority_root / "COMPLETE.json"
    replay_complete = replay_root / "COMPLETE.json"
    if (
        file_sha256(authority_complete)
        != binding.get("authority_complete_file_sha256")
        or file_sha256(replay_complete)
        != binding.get("replay_complete_file_sha256")
    ):
        raise RuntimeError("bounded authority COMPLETE changed")
    authority = bounded_runner.load_bounded_learnability_artifact(
        authority_root
    )
    replay = bounded_runner.load_bounded_learnability_artifact(replay_root)
    if (
        authority.complete_fingerprint
        != binding.get("authority_complete_fingerprint")
        or (
            authority.decision,
            authority.structural_execution_pass,
            authority.computational_learnability_pass,
            authority.pair_catalog_fingerprint,
            authority.micro_population_fingerprint,
            authority.schedule_fingerprint,
            authority.complete_fingerprint,
        )
        != (
            replay.decision,
            replay.structural_execution_pass,
            replay.computational_learnability_pass,
            replay.pair_catalog_fingerprint,
            replay.micro_population_fingerprint,
            replay.schedule_fingerprint,
            replay.complete_fingerprint,
        )
        or authority.decision != binding.get("required_bounded_decision")
        or authority.computational_learnability_pass is not True
        or binding.get("require_authority_replay_byte_identity") is not True
        or _directory_hashes(authority_root)
        != _directory_hashes(replay_root)
    ):
        raise RuntimeError("bounded r1/r2 authority identity changed")

    relative = binding.get("authority_result_relative_path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
    ):
        raise RuntimeError("bounded authority result path is malformed")
    authority_result_path = _canonical_file(
        authority_root / relative,
        name="bounded authority result",
    )
    replay_result_path = _canonical_file(
        replay_root / relative,
        name="bounded replay result",
    )
    if (
        authority_result_path.relative_to(authority_root).as_posix()
        != relative
        or replay_result_path.relative_to(replay_root).as_posix()
        != relative
        or file_sha256(authority_result_path)
        != binding.get("authority_result_file_sha256")
        or file_sha256(replay_result_path)
        != binding.get("authority_result_file_sha256")
    ):
        raise RuntimeError("bounded authority result changed")
    authority_result = pair_preflight_runner._strict_json(
        authority_result_path,
        name="bounded authority result",
    )
    replay_result = pair_preflight_runner._strict_json(
        replay_result_path,
        name="bounded replay result",
    )
    _verify_fingerprinted(
        authority_result,
        name="bounded authority result",
        field="receipt_fingerprint",
    )
    if (
        authority_result != replay_result
        or authority_result.get("receipt_fingerprint")
        != binding.get("authority_result_receipt_fingerprint")
        or authority_result.get("population_fingerprint")
        != binding.get("authority_micro_population_fingerprint")
        or authority_result.get("schedule_fingerprint")
        != binding.get("authority_schedule_fingerprint")
        or authority_result.get("parameters", {}).get(
            "final_decoder_fingerprint"
        )
        != binding.get("authority_final_decoder_fingerprint")
    ):
        raise RuntimeError("bounded result identity differs from freeze")

    bounded_config_path = _repo_file(
        binding.get("bounded_config_path"),
        name="bounded config",
    )
    if (
        file_sha256(bounded_config_path)
        != binding.get("bounded_config_file_sha256")
    ):
        raise RuntimeError("bounded config changed")
    bounded_config = bounded_runner._load_config(bounded_config_path)
    if (
        bounded_config.get("config_fingerprint")
        != binding.get("bounded_config_fingerprint")
    ):
        raise RuntimeError("bounded config fingerprint changed")
    immutable = {
        path.relative_to(_ROOT).as_posix(): file_sha256(path)
        for root in (authority_root, replay_root)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    immutable[bounded_config_path.relative_to(_ROOT).as_posix()] = (
        file_sha256(bounded_config_path)
    )
    return (
        bounded_config,
        authority_result,
        authority_root,
        replay_root,
        immutable,
    )


def _verify_reconstructed_population_schedule(
    config: Mapping[str, Any],
    population: object,
    schedule: object,
    authority_root: Path,
) -> None:
    binding = config["input_binding"]
    if not isinstance(binding, Mapping):
        raise RuntimeError("spatial-tail input binding is malformed")
    micro_path = authority_root / "receipts" / "micro_population.json"
    schedule_path = authority_root / "receipts" / "schedule.json"
    if (
        file_sha256(micro_path)
        != binding.get("authority_micro_population_file_sha256")
        or file_sha256(schedule_path)
        != binding.get("authority_schedule_file_sha256")
    ):
        raise RuntimeError("bounded population/schedule receipt changed")
    if (
        getattr(population, "population_fingerprint", None)
        != binding.get("authority_micro_population_fingerprint")
        or getattr(schedule, "schedule_fingerprint", None)
        != binding.get("authority_schedule_fingerprint")
    ):
        raise RuntimeError(
            "reconstructed population or schedule differs from authority"
        )


def _implementation_binding() -> dict[str, str]:
    paths = (
        _ROOT / "tools" / "run_paired_spatial_tail_diagnostic.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "paired_spatial_tail_diagnostic.py",
        _ROOT / SPATIAL_CONFIG_REPO_PATH,
    )
    values = bounded_runner._implementation_binding()
    values.update(
        {
            path.relative_to(_ROOT).as_posix(): file_sha256(path)
            for path in paths
        }
    )
    return dict(sorted(values.items()))


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {_INCOMPLETE, "COMPLETE.json"}
    }


def _decision(
    result: Mapping[str, object] | None,
    *,
    failure: Mapping[str, object] | None,
    evidence_receipt_fingerprint: str,
) -> dict[str, object]:
    complete = (
        result is not None
        and result.get("exact_bounded_replay_verified") is True
    )
    return _fingerprinted(
        {
            "schema_version": SPATIAL_DECISION_SCHEMA,
            "status": (
                "SPATIAL_TAIL_DIAGNOSTIC_COMPLETE"
                if complete
                else "SPATIAL_TAIL_DIAGNOSTIC_EXECUTION_ERROR"
            ),
            "exact_bounded_replay_verified": complete,
            "spatial_tail_report_complete": complete,
            "descriptive_only": True,
            "retroactive_bounded_gate_added": False,
            "bounded_decision_changed": False,
            "authorizes_formal_800": False,
            "authorizes_D_V_or_D_T": False,
            "not_performance_evidence": True,
            "evidence_kind": "result" if result is not None else "failure",
            "evidence_receipt_fingerprint": (
                evidence_receipt_fingerprint
            ),
            "failure": dict(failure) if failure is not None else None,
            "next_route": (
                "compare_spatial_tail_with_frozen_matched_controls_"
                "without_retroactive_threshold_change"
            ),
        }
    )


@dataclass(frozen=True)
class PublishedSpatialTailDiagnostic:
    root: Path
    status: str
    exact_bounded_replay_verified: bool
    config_fingerprint: str
    population_fingerprint: str
    schedule_fingerprint: str
    complete_fingerprint: str

    def verify_unchanged(self) -> None:
        if load_spatial_tail_diagnostic_artifact(self.root) != self:
            raise RuntimeError("published spatial-tail artifact changed")


def load_spatial_tail_diagnostic_artifact(
    output_dir: str | Path,
) -> PublishedSpatialTailDiagnostic:
    root = Path(output_dir).expanduser().resolve(strict=False)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("spatial-tail root must be a regular directory")
    if (root / _INCOMPLETE).exists():
        raise RuntimeError("spatial-tail publication is incomplete")
    if {item.name for item in root.iterdir()} != {
        "receipts",
        "COMPLETE.json",
    }:
        raise RuntimeError("spatial-tail top-level inventory changed")
    receipts = root / "receipts"
    names = {item.name for item in receipts.iterdir()}
    common = {"config_binding.json", "decision.json"}
    if names not in (
        common | {"result.json"},
        common | {"failure.json"},
    ):
        raise RuntimeError("spatial-tail receipt inventory changed")
    complete = pair_preflight_runner._strict_json(
        root / "COMPLETE.json",
        name="spatial-tail COMPLETE",
    )
    config_binding = pair_preflight_runner._strict_json(
        receipts / "config_binding.json",
        name="spatial-tail config binding",
    )
    decision = pair_preflight_runner._strict_json(
        receipts / "decision.json",
        name="spatial-tail decision",
    )
    evidence_name = (
        "result.json" if "result.json" in names else "failure.json"
    )
    evidence = pair_preflight_runner._strict_json(
        receipts / evidence_name,
        name="spatial-tail evidence",
    )
    _verify_fingerprinted(
        complete,
        name="spatial-tail COMPLETE",
        field="complete_fingerprint",
    )
    for payload, name in (
        (config_binding, "spatial-tail config binding"),
        (decision, "spatial-tail decision"),
        (evidence, "spatial-tail evidence"),
    ):
        _verify_fingerprinted(
            payload,
            name=name,
            field="receipt_fingerprint",
        )
    if (
        complete.get("artifact_files") != _artifact_hashes(root)
        or complete.get("artifact_file_count") != len(names)
        or complete.get("schema_version") != SPATIAL_RUN_SCHEMA
        or complete.get("execution_status") != "complete"
        or complete.get("split") != "D_R"
        or complete.get("descriptive_only") is not True
        or complete.get("retroactive_bounded_gate_added") is not False
        or complete.get("bounded_decision_changed") is not False
        or complete.get("authorizes_formal_800") is not False
        or complete.get("D_V_accessed") is not False
        or complete.get("D_T_accessed") is not False
        or complete.get("checkpoint_recovery_performed") is not False
        or complete.get("checkpoint_persisted") is not False
        or complete.get("formal_training_performed") is not False
        or complete.get("performance_evaluation_performed") is not False
    ):
        raise RuntimeError("spatial-tail COMPLETE boundary changed")
    embedded_config = config_binding.get("config")
    if not isinstance(embedded_config, Mapping):
        raise RuntimeError("spatial-tail embedded config is malformed")
    _verify_fingerprinted(
        embedded_config,
        name="spatial-tail embedded config",
        field="config_fingerprint",
    )
    if (
        config_binding.get("schema_version")
        != SPATIAL_CONFIG_BINDING_SCHEMA
        or embedded_config.get("config_fingerprint")
        != SPATIAL_CONFIG_FINGERPRINT
        or config_binding.get("config_file_sha256")
        != SPATIAL_CONFIG_FILE_SHA256
        or complete.get("config_fingerprint")
        != SPATIAL_CONFIG_FINGERPRINT
        or complete.get("config_binding_fingerprint")
        != config_binding.get("receipt_fingerprint")
        or complete.get("decision_fingerprint")
        != decision.get("receipt_fingerprint")
        or complete.get("evidence_receipt_fingerprint")
        != evidence.get("receipt_fingerprint")
        or decision.get("evidence_receipt_fingerprint")
        != evidence.get("receipt_fingerprint")
        or complete.get("decision") != decision.get("status")
    ):
        raise RuntimeError("spatial-tail cross-binding changed")
    evidence_kind = "result" if evidence_name == "result.json" else "failure"
    if (
        decision.get("evidence_kind") != evidence_kind
        or complete.get("evidence_kind") != evidence_kind
        or decision.get("descriptive_only") is not True
        or decision.get("retroactive_bounded_gate_added") is not False
        or decision.get("bounded_decision_changed") is not False
        or decision.get("authorizes_formal_800") is not False
    ):
        raise RuntimeError("spatial-tail decision boundary changed")
    if evidence_kind == "result":
        interpretation = evidence.get("interpretation")
        if (
            evidence.get("schema_version") != SPATIAL_TAIL_EXECUTION_SCHEMA
            or evidence.get("execution_status") != "completed"
            or evidence.get("exact_bounded_replay_verified") is not True
            or not isinstance(interpretation, Mapping)
            or interpretation.get("descriptive_companion_only") is not True
            or interpretation.get("retroactive_gate_added") is not False
            or interpretation.get("bounded_decision_changed") is not False
            or interpretation.get("authorizes_formal_800") is not False
            or interpretation.get("D_V_accessed") is not False
            or interpretation.get("D_T_accessed") is not False
            or interpretation.get("checkpoint_recovery_performed") is not False
            or interpretation.get("checkpoint_persisted") is not False
            or decision.get("status")
            != "SPATIAL_TAIL_DIAGNOSTIC_COMPLETE"
            or decision.get("exact_bounded_replay_verified") is not True
        ):
            raise RuntimeError("spatial-tail result identity changed")
    else:
        if (
            decision.get("status")
            != "SPATIAL_TAIL_DIAGNOSTIC_EXECUTION_ERROR"
            or decision.get("exact_bounded_replay_verified") is not False
        ):
            raise RuntimeError("spatial-tail failure identity changed")
    return PublishedSpatialTailDiagnostic(
        root=root,
        status=str(decision["status"]),
        exact_bounded_replay_verified=bool(
            decision["exact_bounded_replay_verified"]
        ),
        config_fingerprint=str(complete["config_fingerprint"]),
        population_fingerprint=str(
            complete["micro_population_fingerprint"]
        ),
        schedule_fingerprint=str(complete["schedule_fingerprint"]),
        complete_fingerprint=str(complete["complete_fingerprint"]),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    config_path = _canonical_file(args.config, name="spatial-tail config")
    config = _load_config(config_path)
    output = _prepare_output(args.output)
    (
        bounded_config,
        authority_result,
        authority_root,
        _,
        immutable,
    ) = _load_bounded_authority(config)
    implementation = _implementation_binding()
    pair_catalog, prepared, bundle, upstream_immutable = (
        bounded_runner._load_real_catalog(bounded_config)
    )
    immutable.update(
        {
            Path(path).relative_to(_ROOT).as_posix(): digest
            for path, digest in upstream_immutable.items()
        }
    )
    immutable[config_path.relative_to(_ROOT).as_posix()] = file_sha256(
        config_path
    )
    population = build_bounded_micro_population(
        pair_catalog,
        prepared,
        bounded_config["micro_population"],
    )
    schedule = build_bounded_micro_schedule(
        population,
        bounded_config["budget"],
    )
    _verify_reconstructed_population_schedule(
        config,
        population,
        schedule,
        authority_root,
    )

    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / _INCOMPLETE
    incomplete.open("xb").close()
    receipts = output / "receipts"
    receipts.mkdir(exist_ok=False)
    config_binding = _fingerprinted(
        {
            "schema_version": SPATIAL_CONFIG_BINDING_SCHEMA,
            "config": config,
            "config_file_sha256": file_sha256(config_path),
            "bounded_config_file_sha256": (
                config["input_binding"]["bounded_config_file_sha256"]
            ),
            "bounded_authority_complete_fingerprint": (
                config["input_binding"]["authority_complete_fingerprint"]
            ),
            "bounded_authority_result_receipt_fingerprint": (
                authority_result["receipt_fingerprint"]
            ),
            "bounded_r1_r2_byte_identity_verified": True,
            "micro_population_fingerprint": (
                population.population_fingerprint
            ),
            "schedule_fingerprint": schedule.schedule_fingerprint,
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

    result: dict[str, object] | None = None
    failure: dict[str, object] | None = None
    execution_error: Exception | None = None
    try:
        result = execute_spatial_tail_replay(
            population,
            schedule,
            bounded_config,
            config,
            authority_result,
            device=args.device,
        )
    except Exception as error:
        execution_error = error
    try:
        bundle.verify_unchanged()
        for relative, digest in immutable.items():
            path = _ROOT / relative
            if file_sha256(path) != digest:
                raise RuntimeError(
                    f"frozen spatial-tail input changed: {relative}"
                )
        if _implementation_binding() != implementation:
            raise RuntimeError(
                "spatial-tail implementation changed during execution"
            )
    except Exception as error:
        if execution_error is None:
            execution_error = error

    evidence_receipt: dict[str, object]
    if execution_error is None:
        try:
            if result is None:
                raise RuntimeError(
                    "spatial-tail execution returned no evidence"
                )
            evidence_receipt = _fingerprinted(result)
            json.dumps(evidence_receipt, allow_nan=False)
        except Exception as error:
            execution_error = error
    if execution_error is None:
        _write_new_json(receipts / "result.json", evidence_receipt)
    if execution_error is not None:
        result = None
        failure = {
            "schema_version": (
                "cure-lite-paired-spatial-tail-diagnostic-failure-v1"
            ),
            "exception_type": type(execution_error).__name__,
            "message": str(execution_error),
            "exact_bounded_replay_verified": False,
            "spatial_tail_report_complete": False,
            "retroactive_bounded_gate_added": False,
            "bounded_decision_changed": False,
            "authorizes_formal_800": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
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
            "schema_version": SPATIAL_RUN_SCHEMA,
            "execution_status": "complete",
            "decision": decision["status"],
            "exact_bounded_replay_verified": decision[
                "exact_bounded_replay_verified"
            ],
            "spatial_tail_report_complete": decision[
                "spatial_tail_report_complete"
            ],
            "descriptive_only": True,
            "retroactive_bounded_gate_added": False,
            "bounded_decision_changed": False,
            "authorizes_formal_800": False,
            "split": "D_R",
            "config_fingerprint": config["config_fingerprint"],
            "config_binding_fingerprint": config_binding[
                "receipt_fingerprint"
            ],
            "bounded_authority_complete_fingerprint": config[
                "input_binding"
            ]["authority_complete_fingerprint"],
            "bounded_authority_result_receipt_fingerprint": (
                authority_result["receipt_fingerprint"]
            ),
            "micro_population_fingerprint": (
                population.population_fingerprint
            ),
            "schedule_fingerprint": schedule.schedule_fingerprint,
            "evidence_kind": decision["evidence_kind"],
            "evidence_receipt_fingerprint": evidence_receipt[
                "receipt_fingerprint"
            ],
            "decision_fingerprint": decision["receipt_fingerprint"],
            "artifact_files": artifact_files,
            "artifact_file_count": len(artifact_files),
            "training_scope": (
                "exact_fresh-decoder_replay_of_bounded_400_updates"
            ),
            "formal_training_performed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "performance_evaluation_performed": False,
            "checkpoint_recovery_performed": False,
            "checkpoint_persisted": False,
            "existing_bounded_artifacts_overwritten": False,
        },
        field="complete_fingerprint",
    )
    _write_new_json(output / "COMPLETE.json", complete)
    incomplete.unlink()
    published = load_spatial_tail_diagnostic_artifact(output)
    return {
        "output": str(output),
        "status": published.status,
        "exact_bounded_replay_verified": (
            published.exact_bounded_replay_verified
        ),
        "complete_fingerprint": published.complete_fingerprint,
        "descriptive_only": True,
        "authorizes_formal_800": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run(args)
    print(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    if result["status"] != "SPATIAL_TAIL_DIAGNOSTIC_COMPLETE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
