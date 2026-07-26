#!/usr/bin/env python3
"""Run the frozen D_R-only bounded execution checks for all matched controls.

This command executes each of the eight frozen controls for 400 updates on
the same seed-42 micro-population.  It seals a seed-43 static recipe but does
not train seed 43.  It never opens D_V/D_T, calibrates a threshold, evaluates
detection performance, or authorizes formal 800-epoch training.
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
    BOUNDED_MICRO_POPULATION_SCHEMA,
    BOUNDED_MICRO_SCHEDULE_SCHEMA,
    build_bounded_micro_population,
    build_bounded_micro_schedule,
)
from cure_lite.experiment.paired_control_bounded_execution import (  # noqa: E402
    CONTROL_BOUNDED_EXECUTION_SCHEMA,
    CONTROL_RUNTIME_BINDING_SCHEMA,
    CONTROL_SEMANTICS_SCHEMA,
    build_control_runtime_binding,
    build_control_semantics_receipt,
    execute_control_bounded_execution,
)
from cure_lite.train.paired_control_step import CONTROL_KINDS  # noqa: E402
from tools import run_paired_bounded_learnability as bounded_runner  # noqa: E402
from tools import run_paired_preflight as pair_preflight_runner  # noqa: E402


CONTROL_BOUNDED_CONFIG_REPO_PATH = (
    "protocols/IRSTD-1K/paired_control_bounded_execution_v1/config.json"
)
CONTROL_BOUNDED_CONFIG_FILE_SHA256 = (
    "a0d0d50ea3cf38cee80459a8d6b1bc8f908acc6d9eacf833d9286216ba89251e"
)
CONTROL_BOUNDED_CONFIG_FINGERPRINT = (
    "cc042e167863601d40c6652312722141ba5dcbac0b39cee819aa4a4e2055dd91"
)
CONTROL_BOUNDED_RUN_SCHEMA = (
    "cure-lite-paired-control-bounded-run-v1"
)
CONTROL_BOUNDED_DECISION_SCHEMA = (
    "cure-lite-paired-control-bounded-decision-v1"
)
CONTROL_BOUNDED_SEED43_SCHEMA = (
    "cure-lite-paired-control-seed43-static-recipe-v1"
)
CONTROL_BOUNDED_CONFIG_BINDING_SCHEMA = (
    "cure-lite-paired-control-bounded-config-binding-v1"
)
CONTROL_BOUNDED_FAILURE_SCHEMA = (
    "cure-lite-paired-control-bounded-failure-v1"
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
    parser.add_argument(
        "--paired-bounded-complete",
        type=Path,
        required=True,
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _fingerprinted(
    payload: Mapping[str, object],
    *,
    field: str = "receipt_fingerprint",
) -> dict[str, object]:
    return bounded_runner._fingerprinted(payload, field=field)


def _verify_fingerprinted(
    payload: Mapping[str, Any],
    *,
    name: str,
    field: str = "receipt_fingerprint",
) -> None:
    bounded_runner._verify_fingerprinted(
        payload,
        name=name,
        field=field,
    )


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    bounded_runner._write_new_json(path, payload)


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {_INCOMPLETE, "COMPLETE.json"}
    }


def _load_config(path: Path) -> dict[str, Any]:
    expected = _ROOT / CONTROL_BOUNDED_CONFIG_REPO_PATH
    if path != expected:
        raise RuntimeError("control-bounded config path differs from the freeze")
    if file_sha256(path) != CONTROL_BOUNDED_CONFIG_FILE_SHA256:
        raise RuntimeError("control-bounded config is not the exact frozen file")
    config = pair_preflight_runner._strict_json(
        path,
        name="control-bounded config",
    )
    _verify_fingerprinted(
        config,
        name="control-bounded config",
        field="config_fingerprint",
    )
    if (
        config.get("config_fingerprint")
        != CONTROL_BOUNDED_CONFIG_FINGERPRINT
        or config.get("schema_version")
        != "cure-lite-paired-control-bounded-execution-config-v1"
        or config.get("dataset") != "IRSTD-1K"
        or config.get("split") != "D_R"
        or config.get("not_performance_evidence") is not True
        or config.get("controls") != list(CONTROL_KINDS)
    ):
        raise RuntimeError("control-bounded config identity changed")
    budget = config.get("budget")
    optimization = config.get("optimization")
    policy = config.get("execution_policy")
    if (
        not isinstance(budget, Mapping)
        or not isinstance(optimization, Mapping)
        or not isinstance(policy, Mapping)
    ):
        raise RuntimeError("control-bounded config sections are malformed")
    expected_budget = {
        "steps_per_epoch": 40,
        "updates_per_control": 400,
        "control_count": 8,
        "decoder_states_per_update": 12,
        "forward_calls_per_update": 3,
        "forward_calls_per_control": 1200,
        "state_evaluations_per_control": 4800,
        "total_optimizer_updates": 3200,
        "total_forward_calls": 9600,
        "total_state_evaluations": 38400,
    }
    if any(budget.get(key) != value for key, value in expected_budget.items()):
        raise RuntimeError("control-bounded execution budget changed")
    if (
        optimization.get("optimizer") != "adam"
        or optimization.get("learning_rate") != 0.001
        or optimization.get("weight_decay") != 0.0
        or optimization.get("seed") != 42
    ):
        raise RuntimeError("control-bounded optimizer changed")
    if any(
        policy.get(field) is not False
        for field in (
            "allow_D_V",
            "allow_D_T",
            "allow_calibration",
            "allow_performance_evaluation",
            "allow_formal_800",
            "allow_full_cure",
            "allow_backbone_integration",
            "resume",
            "overwrite",
        )
    ):
        raise RuntimeError("control-bounded execution boundary changed")
    return config


def _schedule_budget(config: Mapping[str, Any]) -> dict[str, int]:
    budget = config["budget"]
    if not isinstance(budget, Mapping):
        raise RuntimeError("control-bounded budget is malformed")
    return {
        "optimizer_updates": int(budget["updates_per_control"]),
        "steps_per_epoch": int(budget["steps_per_epoch"]),
        "factual_miss_states_per_update": 4,
        "factual_no_miss_states_per_update": 4,
        "clean_pairs_per_update": 2,
    }


def _verify_seed43_formal_recipe(config: Mapping[str, Any]) -> None:
    binding = config["input_binding"]
    recipe = config["seed43_static_recipe"]
    if not isinstance(binding, Mapping) or not isinstance(recipe, Mapping):
        raise RuntimeError("seed-43 recipe binding is malformed")
    path = bounded_runner._repo_file(
        recipe.get("formal_config_path"),
        name="formal seed-43 config",
    )
    if (
        file_sha256(path) != recipe.get("formal_config_file_sha256")
        or file_sha256(path) != binding.get("formal_seed43_config_file_sha256")
        or recipe.get("seed") != 43
        or recipe.get("execution_performed") is not False
    ):
        raise RuntimeError("formal seed-43 recipe file changed")
    payload = pair_preflight_runner._strict_json(
        path,
        name="formal seed-43 config",
    )
    training = payload.get("training")
    optimization = config["optimization"]
    if not isinstance(training, Mapping) or not isinstance(
        optimization,
        Mapping,
    ):
        raise RuntimeError("formal seed-43 training recipe is malformed")
    expected = {
        "global_seed": 43,
        "optimizer": optimization["optimizer"],
        "learning_rate": optimization["learning_rate"],
        "weight_decay": optimization["weight_decay"],
        "decoder_config": optimization["decoder"],
        "loss_config": optimization["loss"],
    }
    if any(training.get(key) != value for key, value in expected.items()):
        raise RuntimeError("seed-43 formal optimizer recipe changed")


def _verify_paired_bounded_pass(
    path: Path,
    contract: Mapping[str, object],
) -> dict[str, object]:
    authority = bounded_runner._repo_file(
        contract.get("authority_complete_path"),
        name="paired bounded authority COMPLETE",
    )
    replay = bounded_runner._repo_file(
        contract.get("replay_complete_path"),
        name="paired bounded replay COMPLETE",
    )
    if path != authority:
        raise RuntimeError("paired bounded pass must use the r1 authority")
    if (
        file_sha256(authority)
        != contract.get("authority_complete_file_sha256")
        or file_sha256(replay)
        != contract.get("replay_complete_file_sha256")
    ):
        raise RuntimeError("paired bounded COMPLETE changed")
    first = bounded_runner.load_bounded_learnability_artifact(authority.parent)
    second = bounded_runner.load_bounded_learnability_artifact(replay.parent)
    identity_first = (
        first.decision,
        first.structural_execution_pass,
        first.computational_learnability_pass,
        first.pair_catalog_fingerprint,
        first.micro_population_fingerprint,
        first.schedule_fingerprint,
        first.complete_fingerprint,
    )
    identity_second = (
        second.decision,
        second.structural_execution_pass,
        second.computational_learnability_pass,
        second.pair_catalog_fingerprint,
        second.micro_population_fingerprint,
        second.schedule_fingerprint,
        second.complete_fingerprint,
    )
    first_files = {
        item.relative_to(authority.parent).as_posix(): file_sha256(item)
        for item in sorted(authority.parent.rglob("*"))
        if item.is_file()
    }
    second_files = {
        item.relative_to(replay.parent).as_posix(): file_sha256(item)
        for item in sorted(replay.parent.rglob("*"))
        if item.is_file()
    }
    if (
        contract.get("require_byte_identical_r1_r2") is not True
        or identity_first != identity_second
        or first_files != second_files
        or first.complete_fingerprint != contract.get("complete_fingerprint")
        or first.decision != contract.get("required_decision")
        or first.structural_execution_pass
        is not contract.get("required_structural_execution_pass")
        or first.computational_learnability_pass
        is not contract.get("required_computational_learnability_pass")
    ):
        raise RuntimeError("paired bounded pass contract is not satisfied")
    return {
        "complete_fingerprint": first.complete_fingerprint,
        "complete_file_sha256": file_sha256(authority),
        "byte_identical_replay_verified": True,
    }


def _load_static_control_identities(
    config: Mapping[str, Any],
    authority_complete: Path,
) -> dict[str, str]:
    contract = config["control_preflight_contract"]
    if not isinstance(contract, Mapping):
        raise RuntimeError("control static contract is malformed")
    root = authority_complete.parent
    fields = {
        "target_permutation": (
            contract["target_permutation_relative_path"],
            contract["target_permutation_file_sha256"],
        ),
        "dct_basis": (
            contract["dct_basis_relative_path"],
            contract["dct_basis_file_sha256"],
        ),
    }
    payloads: dict[str, dict[str, Any]] = {}
    for name, (relative, expected_sha) in fields.items():
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise RuntimeError(f"{name} receipt path is malformed")
        path = bounded_runner._canonical_file(
            root / relative,
            name=f"{name} static receipt",
        )
        if (
            path.relative_to(root).as_posix() != relative
            or file_sha256(path) != expected_sha
        ):
            raise RuntimeError(f"{name} static receipt changed")
        payload = pair_preflight_runner._strict_json(
            path,
            name=f"{name} static receipt",
        )
        _verify_fingerprinted(payload, name=f"{name} static receipt")
        payloads[name] = payload
    if (
        payloads["target_permutation"].get("status") != "READY"
        or payloads["target_permutation"].get("plan_fingerprint")
        != contract.get("target_permutation_plan_fingerprint")
        or payloads["dct_basis"].get("basis_fingerprint")
        != contract.get("dct_basis_fingerprint")
    ):
        raise RuntimeError("static control identities changed")
    return {
        "target_permutation_plan_fingerprint": str(
            payloads["target_permutation"]["plan_fingerprint"]
        ),
        "dct_basis_fingerprint": str(
            payloads["dct_basis"]["basis_fingerprint"]
        ),
    }


def _implementation_binding() -> dict[str, str]:
    paths = (
        _ROOT / "tools" / "run_paired_control_bounded_execution.py",
        _ROOT
        / "cure_lite"
        / "experiment"
        / "paired_control_bounded_execution.py",
        _ROOT / "cure_lite" / "train" / "paired_control_step.py",
        _ROOT / "cure_lite" / "paired_control_inputs.py",
        _ROOT / "cure_lite" / "paired_control_losses.py",
        _ROOT / "cure_lite" / "paired_losses.py",
        _ROOT / "cure_lite" / "paired_types.py",
        _ROOT / "cure_lite" / "decoder.py",
        _ROOT / "cure_lite" / "losses.py",
        _ROOT / "cure_lite" / "model.py",
    )
    values = bounded_runner._implementation_binding()
    values.update(
        {
            path.relative_to(_ROOT).as_posix(): file_sha256(path)
            for path in paths
        }
    )
    return dict(sorted(values.items()))


def _micro_receipt(population) -> dict[str, object]:
    return _fingerprinted(
        {
            **population.canonical_payload(),
            "population_fingerprint": population.population_fingerprint,
        }
    )


def _schedule_receipt(schedule) -> dict[str, object]:
    return _fingerprinted(
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


def _seed43_static_receipt(
    population,
    schedule,
    config: Mapping[str, Any],
) -> dict[str, object]:
    recipe = config["seed43_static_recipe"]
    if not isinstance(recipe, Mapping):
        raise RuntimeError("seed-43 static recipe is malformed")
    return _fingerprinted(
        {
            "schema_version": CONTROL_BOUNDED_SEED43_SCHEMA,
            "seed": 43,
            "execution_performed": False,
            "formal_config_path": recipe["formal_config_path"],
            "formal_config_file_sha256": recipe[
                "formal_config_file_sha256"
            ],
            "population": {
                **population.canonical_payload(),
                "population_fingerprint": (
                    population.population_fingerprint
                ),
            },
            "schedule": {
                **schedule.canonical_payload(),
                "schedule_fingerprint": schedule.schedule_fingerprint,
            },
            "optimizer": {
                **dict(config["optimization"]),
                "seed": 43,
            },
            "D_V_accessed": False,
            "D_T_accessed": False,
            "authorizes_formal_800": False,
        }
    )


def _decision(
    result: Mapping[str, object] | None,
    *,
    failure: Mapping[str, object] | None,
    evidence_receipt_fingerprint: str,
) -> dict[str, object]:
    if result is None:
        passed = False
        status = "ENGINEERING_EXECUTION_ERROR"
    else:
        passed = result.get("engineering_execution_pass") is True
        status = (
            "ENGINEERING_EXECUTION_PASS"
            if passed
            else "ENGINEERING_EXECUTION_FAIL"
        )
    return _fingerprinted(
        {
            "schema_version": CONTROL_BOUNDED_DECISION_SCHEMA,
            "status": status,
            "engineering_execution_pass": passed,
            "not_performance_evidence": True,
            "positive_response_learning_required": False,
            "authorizes_formal_800": False,
            "authorizes_D_V_or_D_T": False,
            "evidence_kind": "result" if result is not None else "failure",
            "evidence_receipt_fingerprint": evidence_receipt_fingerprint,
            "failure": dict(failure) if failure is not None else None,
            "budget_extended_after_result": False,
            "core_mechanism_changed": False,
            "next_route": (
                "freeze_formal_paired_matched_control_protocol"
                if passed
                else "review_control_execution_without_core_change"
            ),
        }
    )


@dataclass(frozen=True)
class PublishedControlBoundedExecution:
    root: Path
    decision: str
    engineering_execution_pass: bool
    pair_catalog_fingerprint: str
    seed42_population_fingerprint: str
    seed42_schedule_fingerprint: str
    seed43_population_fingerprint: str
    seed43_schedule_fingerprint: str
    complete_fingerprint: str

    def verify_unchanged(self) -> None:
        if load_control_bounded_execution_artifact(self.root) != self:
            raise RuntimeError("control-bounded artifact identity changed")


def load_control_bounded_execution_artifact(
    output_dir: str | Path,
) -> PublishedControlBoundedExecution:
    """Load and cross-check every file in a completed control execution."""

    root = Path(output_dir).expanduser().resolve(strict=False)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("control-bounded root must be a regular directory")
    if (root / _INCOMPLETE).exists():
        raise RuntimeError("control-bounded publication is incomplete")
    if {item.name for item in root.iterdir()} != {
        "receipts",
        "COMPLETE.json",
    }:
        raise RuntimeError("control-bounded top-level inventory changed")
    receipts = root / "receipts"
    if receipts.is_symlink() or not receipts.is_dir():
        raise ValueError("control-bounded receipts must be a regular directory")
    common = {
        "config_binding.json",
        "seed42_micro_population.json",
        "seed42_schedule.json",
        "seed43_static_recipe.json",
        "control_semantics.json",
        "runtime_binding.json",
        "decision.json",
    }
    names = {item.name for item in receipts.iterdir()}
    if names not in (
        common | {"result.json"},
        common | {"failure.json"},
    ):
        raise RuntimeError("control-bounded receipt inventory changed")
    payloads = {
        "complete": pair_preflight_runner._strict_json(
            root / "COMPLETE.json",
            name="control-bounded COMPLETE",
        ),
        **{
            name[:-5]: pair_preflight_runner._strict_json(
                receipts / name,
                name=f"control-bounded {name}",
            )
            for name in sorted(names)
        },
    }
    _verify_fingerprinted(
        payloads["complete"],
        name="control-bounded COMPLETE",
        field="complete_fingerprint",
    )
    for name, payload in payloads.items():
        if name != "complete":
            _verify_fingerprinted(
                payload,
                name=f"control-bounded {name}",
            )
    complete = payloads["complete"]
    if (
        complete.get("artifact_files") != _artifact_hashes(root)
        or complete.get("artifact_file_count") != len(names)
    ):
        raise RuntimeError("control-bounded artifact hashes changed")
    if (
        complete.get("schema_version") != CONTROL_BOUNDED_RUN_SCHEMA
        or complete.get("execution_status") != "complete"
        or complete.get("split") != "D_R"
        or complete.get("not_performance_evidence") is not True
        or complete.get("authorizes_formal_800") is not False
        or complete.get("D_V_accessed") is not False
        or complete.get("D_T_accessed") is not False
        or complete.get("calibration_performed") is not False
        or complete.get("performance_evaluation_performed") is not False
        or complete.get("formal_training_performed") is not False
    ):
        raise RuntimeError("control-bounded COMPLETE boundary changed")
    config_binding = payloads["config_binding"]
    embedded = config_binding.get("config")
    if not isinstance(embedded, Mapping):
        raise RuntimeError("embedded control-bounded config is malformed")
    _verify_fingerprinted(
        embedded,
        name="embedded control-bounded config",
        field="config_fingerprint",
    )
    if (
        embedded.get("config_fingerprint")
        != CONTROL_BOUNDED_CONFIG_FINGERPRINT
        or config_binding.get("config_file_sha256")
        != CONTROL_BOUNDED_CONFIG_FILE_SHA256
        or complete.get("config_fingerprint")
        != CONTROL_BOUNDED_CONFIG_FINGERPRINT
        or complete.get("config_binding_fingerprint")
        != config_binding.get("receipt_fingerprint")
    ):
        raise RuntimeError("control-bounded config binding changed")
    micro = payloads["seed42_micro_population"]
    schedule = payloads["seed42_schedule"]
    if (
        micro.get("schema_version") != BOUNDED_MICRO_POPULATION_SCHEMA
        or stable_fingerprint(
            bounded_runner._canonical_micro_payload(micro)
        )
        != micro.get("population_fingerprint")
        or schedule.get("schema_version") != BOUNDED_MICRO_SCHEDULE_SCHEMA
        or stable_fingerprint(
            bounded_runner._canonical_schedule_payload(schedule)
        )
        != schedule.get("schedule_fingerprint")
        or complete.get("seed42_population_fingerprint")
        != micro.get("population_fingerprint")
        or complete.get("seed42_schedule_fingerprint")
        != schedule.get("schedule_fingerprint")
    ):
        raise RuntimeError("seed-42 population or schedule changed")
    seed43 = payloads["seed43_static_recipe"]
    seed43_population = seed43.get("population")
    seed43_schedule = seed43.get("schedule")
    if (
        seed43.get("schema_version") != CONTROL_BOUNDED_SEED43_SCHEMA
        or seed43.get("seed") != 43
        or seed43.get("execution_performed") is not False
        or not isinstance(seed43_population, Mapping)
        or not isinstance(seed43_schedule, Mapping)
        or stable_fingerprint(
            bounded_runner._canonical_micro_payload(seed43_population)
        )
        != seed43_population.get("population_fingerprint")
        or stable_fingerprint(
            bounded_runner._canonical_schedule_payload(seed43_schedule)
        )
        != seed43_schedule.get("schedule_fingerprint")
        or complete.get("seed43_population_fingerprint")
        != seed43_population.get("population_fingerprint")
        or complete.get("seed43_schedule_fingerprint")
        != seed43_schedule.get("schedule_fingerprint")
    ):
        raise RuntimeError("seed-43 static recipe changed")
    runtime = payloads["runtime_binding"]
    semantics = payloads["control_semantics"]
    if (
        runtime.get("schema_version") != CONTROL_RUNTIME_BINDING_SCHEMA
        or semantics.get("schema_version") != CONTROL_SEMANTICS_SCHEMA
        or semantics.get("all_control_semantics_pass") is not True
        or complete.get("runtime_binding_fingerprint")
        != runtime.get("binding_fingerprint")
        or semantics.get("runtime_binding_fingerprint")
        != runtime.get("binding_fingerprint")
    ):
        raise RuntimeError("control runtime or semantic binding changed")
    evidence_name = "result" if "result.json" in names else "failure"
    evidence = payloads[evidence_name]
    decision = payloads["decision"]
    expected_status = (
        "ENGINEERING_EXECUTION_PASS"
        if evidence_name == "result"
        and evidence.get("engineering_execution_pass") is True
        else "ENGINEERING_EXECUTION_FAIL"
        if evidence_name == "result"
        else "ENGINEERING_EXECUTION_ERROR"
    )
    if (
        decision.get("schema_version") != CONTROL_BOUNDED_DECISION_SCHEMA
        or decision.get("status") != expected_status
        or decision.get("evidence_kind") != evidence_name
        or decision.get("evidence_receipt_fingerprint")
        != evidence.get("receipt_fingerprint")
        or decision.get("authorizes_formal_800") is not False
        or decision.get("authorizes_D_V_or_D_T") is not False
        or complete.get("decision") != expected_status
        or complete.get("decision_fingerprint")
        != decision.get("receipt_fingerprint")
        or complete.get("evidence_receipt_fingerprint")
        != evidence.get("receipt_fingerprint")
    ):
        raise RuntimeError("control-bounded decision binding changed")
    if evidence_name == "result":
        controls = evidence.get("controls")
        if (
            evidence.get("schema_version")
            != CONTROL_BOUNDED_EXECUTION_SCHEMA
            or evidence.get("execution_status") != "completed"
            or evidence.get("control_order") != list(CONTROL_KINDS)
            or not isinstance(controls, Mapping)
            or set(controls) != set(CONTROL_KINDS)
            or any(
                not isinstance(controls[name], Mapping)
                for name in CONTROL_KINDS
            )
            or complete.get("engineering_execution_pass")
            is not evidence.get("engineering_execution_pass")
        ):
            raise RuntimeError("control-bounded result changed")
    else:
        if (
            evidence.get("schema_version") != CONTROL_BOUNDED_FAILURE_SCHEMA
            or complete.get("engineering_execution_pass") is not False
        ):
            raise RuntimeError("control-bounded failure changed")
    return PublishedControlBoundedExecution(
        root=root,
        decision=str(decision["status"]),
        engineering_execution_pass=bool(
            decision["engineering_execution_pass"]
        ),
        pair_catalog_fingerprint=str(
            complete["pair_catalog_fingerprint"]
        ),
        seed42_population_fingerprint=str(
            micro["population_fingerprint"]
        ),
        seed42_schedule_fingerprint=str(
            schedule["schedule_fingerprint"]
        ),
        seed43_population_fingerprint=str(
            seed43_population["population_fingerprint"]
        ),
        seed43_schedule_fingerprint=str(
            seed43_schedule["schedule_fingerprint"]
        ),
        complete_fingerprint=str(complete["complete_fingerprint"]),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    config_path = bounded_runner._canonical_file(
        args.config,
        name="control-bounded config",
    )
    config = _load_config(config_path)
    output = bounded_runner._prepare_output(args.output)
    control_path = bounded_runner._canonical_file(
        args.control_preflight_complete,
        name="matched-control preflight COMPLETE",
    )
    bounded_path = bounded_runner._canonical_file(
        args.paired_bounded_complete,
        name="paired bounded COMPLETE",
    )
    control_contract = config["control_preflight_contract"]
    bounded_contract = config["paired_bounded_pass_contract"]
    if not isinstance(control_contract, Mapping) or not isinstance(
        bounded_contract,
        Mapping,
    ):
        raise RuntimeError("upstream contracts are malformed")
    control = bounded_runner._verify_control_preflight(
        control_path,
        control_contract,
    )
    bounded_pass = _verify_paired_bounded_pass(
        bounded_path,
        bounded_contract,
    )
    static = _load_static_control_identities(config, control_path)
    _verify_seed43_formal_recipe(config)
    implementation = _implementation_binding()
    pair_catalog, prepared, bundle, immutable = (
        bounded_runner._load_real_catalog(config)
    )
    immutable[str(config_path)] = file_sha256(config_path)
    control_artifact_hashes = bounded_runner._control_artifact_hashes(
        control_contract
    )
    immutable.update(
        {
            str(_ROOT / relative): digest
            for relative, digest in control_artifact_hashes.items()
        }
    )
    for root in (bounded_path.parent,):
        immutable.update(
            {
                str(item): file_sha256(item)
                for item in sorted(root.rglob("*"))
                if item.is_file()
            }
        )

    population42 = build_bounded_micro_population(
        pair_catalog,
        prepared,
        config["micro_population"],
    )
    schedule42 = build_bounded_micro_schedule(
        population42,
        _schedule_budget(config),
    )
    micro43_spec = dict(config["micro_population"])
    micro43_spec["seed"] = 43
    population43 = build_bounded_micro_population(
        pair_catalog,
        prepared,
        micro43_spec,
    )
    schedule43 = build_bounded_micro_schedule(
        population43,
        _schedule_budget(config),
    )
    gt_unions = {
        entry.sample_id: entry.gt.occupancy.unsqueeze(0)
        for entry in prepared.entries
    }
    runtime_binding = build_control_runtime_binding(
        pair_catalog,
        population42,
        gt_unions,
        expected_permutation_fingerprint=static[
            "target_permutation_plan_fingerprint"
        ],
        expected_dct_basis_fingerprint=static[
            "dct_basis_fingerprint"
        ],
    )
    semantics = build_control_semantics_receipt(
        population42,
        schedule42,
        runtime_binding,
    )
    if semantics["all_control_semantics_pass"] is not True:
        raise RuntimeError("control semantics did not pass before publication")

    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / _INCOMPLETE
    incomplete.open("xb").close()
    receipts = output / "receipts"
    receipts.mkdir(exist_ok=False)
    config_binding = _fingerprinted(
        {
            "schema_version": CONTROL_BOUNDED_CONFIG_BINDING_SCHEMA,
            "config": config,
            "config_file_sha256": file_sha256(config_path),
            "control_preflight_complete_fingerprint": control[
                "complete_fingerprint"
            ],
            "control_preflight_byte_identical_replay_verified": control[
                "byte_identical_replay_verified"
            ],
            "paired_bounded_complete_fingerprint": bounded_pass[
                "complete_fingerprint"
            ],
            "paired_bounded_byte_identical_replay_verified": bounded_pass[
                "byte_identical_replay_verified"
            ],
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
    micro42_receipt = _micro_receipt(population42)
    schedule42_receipt = _schedule_receipt(schedule42)
    seed43_receipt = _seed43_static_receipt(
        population43,
        schedule43,
        config,
    )
    runtime_receipt = _fingerprinted(
        {
            **runtime_binding.canonical_payload(),
            "binding_fingerprint": runtime_binding.binding_fingerprint,
            "runtime_training_binding_closed": True,
            "recipient_donor_target_fingerprint_closure": True,
        }
    )
    semantics_receipt = _fingerprinted(semantics)
    _write_new_json(receipts / "config_binding.json", config_binding)
    _write_new_json(
        receipts / "seed42_micro_population.json",
        micro42_receipt,
    )
    _write_new_json(receipts / "seed42_schedule.json", schedule42_receipt)
    _write_new_json(
        receipts / "seed43_static_recipe.json",
        seed43_receipt,
    )
    _write_new_json(receipts / "runtime_binding.json", runtime_receipt)
    _write_new_json(receipts / "control_semantics.json", semantics_receipt)

    result: dict[str, object] | None = None
    failure: dict[str, object] | None = None
    execution_error: Exception | None = None
    evidence_receipt: dict[str, object]
    try:
        result = execute_control_bounded_execution(
            population42,
            schedule42,
            runtime_binding,
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
                "a frozen control-bounded input changed during execution"
            )
        if _implementation_binding() != implementation:
            raise RuntimeError(
                "control-bounded implementation changed during execution"
            )
    except Exception as error:
        if execution_error is None:
            execution_error = error
    if execution_error is None:
        if result is None:
            execution_error = RuntimeError(
                "control-bounded execution returned no evidence"
            )
        else:
            evidence_receipt = _fingerprinted(result)
            json.dumps(evidence_receipt, allow_nan=False)
            _write_new_json(receipts / "result.json", evidence_receipt)
    if execution_error is not None:
        result = None
        failure = {
            "schema_version": CONTROL_BOUNDED_FAILURE_SCHEMA,
            "exception_type": type(execution_error).__name__,
            "message": str(execution_error),
            "engineering_execution_pass": False,
            "budget_extended_after_result": False,
            "core_mechanism_changed": False,
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
            "schema_version": CONTROL_BOUNDED_RUN_SCHEMA,
            "execution_status": "complete",
            "decision": decision["status"],
            "engineering_execution_pass": decision[
                "engineering_execution_pass"
            ],
            "not_performance_evidence": True,
            "authorizes_formal_800": False,
            "split": "D_R",
            "pair_catalog_fingerprint": pair_catalog.catalog_fingerprint,
            "config_fingerprint": config["config_fingerprint"],
            "config_binding_fingerprint": config_binding[
                "receipt_fingerprint"
            ],
            "control_preflight_complete_fingerprint": control[
                "complete_fingerprint"
            ],
            "paired_bounded_complete_fingerprint": bounded_pass[
                "complete_fingerprint"
            ],
            "seed42_population_fingerprint": (
                population42.population_fingerprint
            ),
            "seed42_schedule_fingerprint": (
                schedule42.schedule_fingerprint
            ),
            "seed43_population_fingerprint": (
                population43.population_fingerprint
            ),
            "seed43_schedule_fingerprint": (
                schedule43.schedule_fingerprint
            ),
            "runtime_binding_fingerprint": (
                runtime_binding.binding_fingerprint
            ),
            "evidence_kind": decision["evidence_kind"],
            "evidence_receipt_fingerprint": evidence_receipt[
                "receipt_fingerprint"
            ],
            "decision_fingerprint": decision["receipt_fingerprint"],
            "artifact_files": artifact_files,
            "artifact_file_count": len(artifact_files),
            "training_scope": (
                "eight_fresh_decoders_each_400_D_R_only_updates"
            ),
            "formal_training_performed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "performance_evaluation_performed": False,
            "seed42_exact_replay_required": True,
            "seed43_execution_performed": False,
        },
        field="complete_fingerprint",
    )
    _write_new_json(output / "COMPLETE.json", complete)
    incomplete.unlink()
    published = load_control_bounded_execution_artifact(output)
    return {
        "output": str(output),
        "decision": published.decision,
        "engineering_execution_pass": (
            published.engineering_execution_pass
        ),
        "seed42_population_fingerprint": (
            published.seed42_population_fingerprint
        ),
        "seed42_schedule_fingerprint": (
            published.seed42_schedule_fingerprint
        ),
        "seed43_population_fingerprint": (
            published.seed43_population_fingerprint
        ),
        "seed43_schedule_fingerprint": (
            published.seed43_schedule_fingerprint
        ),
        "complete_fingerprint": published.complete_fingerprint,
        "not_performance_evidence": True,
        "authorizes_formal_800": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if result["engineering_execution_pass"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
