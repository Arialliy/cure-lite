#!/usr/bin/env python3
"""Complete Stage-A after all three frozen decoder runs already exist.

This entry point is intentionally post-training-only.  It accepts one exact
incomplete Stage-A directory, strictly reloads the existing F/Fx/U decoder
artifacts, computes the frozen D_V calibration/results and efficiency receipt,
then publishes the ordinary Stage-A COMPLETE receipt.  It contains no decoder
training or artifact-writing path.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.data import ManifestImageDataset, PreprocessConfig  # noqa: E402
from cure_lite.experiment.artifacts import (  # noqa: E402
    LoadedDecoderArtifact,
    load_decoder_artifact,
)
from cure_lite.experiment.cache_pipeline import (  # noqa: E402
    load_base_cache_pair_contract,
    load_d_r_cache_bundle,
    load_d_v_cache_bundle,
)
from cure_lite.experiment.formal_training import prepare_gate2_training  # noqa: E402
from cure_lite.experiment.stage_a_runner import (  # noqa: E402
    _COMPLETE_NAME,
    _INCOMPLETE_NAME,
    _anchor_receipt_payload,
    _bind_published_stage_a_run,
    _build_downstream_state,
    _calibration_receipt_payload,
    _check_dataset_pair,
    _complete_receipt,
    _config_receipt,
    _measure_stage_a_efficiency,
    _preflight_stage_a_device,
    _receipt_paths,
    _results_receipt_payload,
    _source_tree_digest,
    _strict_json,
    _support_receipt_payload,
    _tree_inventory,
    _verified_base_run_payload,
    _write_new_json,
)
from cure_lite.reference_base import (  # noqa: E402
    load_verified_reference_base_run_identity,
)
from cure_lite.splits import load_and_validate_manifest  # noqa: E402
from tools.assess_stage_a import (  # noqa: E402
    _sha256,
    _validate_decision_rule,
    validate_protocol_freeze,
)
from tools.run_stage_a import (  # noqa: E402
    DEFAULT_CALIBRATION_WORKERS,
    _calibration_progress,
    _json_object,
    _positive_int,
    _summary_payload,
    load_stage_a_config,
)


FINALIZATION_SCHEMA = "cure-lite-stage-a-post-training-finalization-v1"
MINIMUM_PARALLEL_NOFILE = 8192
_ROOT = Path(__file__).resolve().parents[1]
_FINALIZATION_NAME = "finalization.json"
_POST_TRAINING_RECEIPT_NAMES = frozenset(
    {
        "receipts/calibration.json",
        "receipts/results.json",
        "receipts/efficiency.json",
        f"receipts/{_FINALIZATION_NAME}",
    }
)
_DECODER_DIRECTORIES = {
    "F": "factual_only",
    "F×": "factual_exposure_matched",
    "U": "uniform_legal",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--d-r-base-index", type=Path, required=True)
    parser.add_argument("--d-v-base-index", type=Path, required=True)
    parser.add_argument("--reference-base-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decision-rule", type=Path, required=True)
    parser.add_argument("--protocol-freeze", type=Path, required=True)
    parser.add_argument("--stage-run", type=Path, required=True)
    parser.add_argument(
        "--calibration-workers",
        type=_positive_int,
        default=DEFAULT_CALIBRATION_WORKERS,
        help=(
            "candidate-evaluation worker processes; execution-only and does "
            "not alter the frozen scientific protocol"
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _resolve_frozen_path(value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"protocol freeze {name} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = _ROOT / path
    return path.resolve(strict=False)


def _require_regular_directory(path: Path, *, name: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{name} may not be addressed through a symbolic link")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink directory")
    return resolved


def _require_stage_marker(root: Path) -> Path:
    marker = root / _INCOMPLETE_NAME
    if marker.is_symlink() or not marker.is_file():
        raise RuntimeError("Stage-A post-training completion requires .incomplete")
    if marker.stat().st_size != 0:
        raise RuntimeError("Stage-A .incomplete marker must be empty")
    return marker


def _guard_stage_tree(root: Path) -> Path:
    marker = _require_stage_marker(root)
    expected_top_level = {
        _INCOMPLETE_NAME,
        "d_r",
        "d_v",
        "decoders",
        "receipts",
    }
    complete_path = root / _COMPLETE_NAME
    if complete_path.exists() or complete_path.is_symlink():
        expected_top_level.add(_COMPLETE_NAME)
        if complete_path.is_symlink() or not complete_path.is_file():
            raise RuntimeError("existing Stage-A COMPLETE must be a regular file")
    actual_top_level = {path.name for path in root.iterdir()}
    if actual_top_level != expected_top_level:
        raise RuntimeError("Stage-A incomplete directory has unexpected top-level members")

    expected_receipts = {
        "config.json",
        "anchor.json",
        "support.json",
        "calibration.json",
        "results.json",
        "efficiency.json",
        _FINALIZATION_NAME,
    }
    receipts = root / "receipts"
    if receipts.is_symlink() or not receipts.is_dir():
        raise RuntimeError("Stage-A receipts directory is invalid")
    actual_receipts = {path.name for path in receipts.iterdir()}
    if not {"config.json", "anchor.json", "support.json"}.issubset(actual_receipts):
        raise RuntimeError("Stage-A pre-training receipts are incomplete")
    if not actual_receipts.issubset(expected_receipts):
        raise RuntimeError("Stage-A receipts directory has unexpected members")

    decoders = root / "decoders"
    if decoders.is_symlink() or not decoders.is_dir():
        raise RuntimeError("Stage-A decoder directory is invalid")
    if {path.name for path in decoders.iterdir()} != set(
        _DECODER_DIRECTORIES.values()
    ):
        raise RuntimeError("Stage-A must contain exactly the F/Fx/U decoder directories")
    return marker


def _require_execution_capacity(calibration_workers: int) -> tuple[int, int]:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    soft_value = int(soft)
    hard_value = int(hard)
    if calibration_workers > 1 and soft_value < MINIMUM_PARALLEL_NOFILE:
        raise RuntimeError(
            "parallel Stage-A calibration requires RLIMIT_NOFILE soft limit "
            f">= {MINIMUM_PARALLEL_NOFILE}; observed {soft_value}"
        )
    return soft_value, hard_value


def _scientific_input_inventory(
    root: Path,
) -> tuple[list[str], dict[str, str], str]:
    directories, files = _tree_inventory(root)
    input_files = {
        name: digest
        for name, digest in files.items()
        if name != _INCOMPLETE_NAME
        and name != _COMPLETE_NAME
        and name not in _POST_TRAINING_RECEIPT_NAMES
    }
    input_directories = list(directories)
    fingerprint = stable_fingerprint(
        {
            "schema_version": "cure-lite-stage-a-scientific-input-tree-v1",
            "directories": input_directories,
            "files": input_files,
        }
    )
    return input_directories, input_files, fingerprint


def _require_same_input_inventory(
    root: Path,
    expected_directories: list[str],
    expected_files: Mapping[str, str],
    expected_fingerprint: str,
) -> None:
    directories, files, fingerprint = _scientific_input_inventory(root)
    if (
        directories != expected_directories
        or files != dict(expected_files)
        or fingerprint != expected_fingerprint
    ):
        raise RuntimeError("Stage-A scientific input files changed during completion")


def _load_decoders(root: Path) -> dict[str, LoadedDecoderArtifact]:
    artifacts = {
        method: load_decoder_artifact(root / "decoders" / directory)
        for method, directory in _DECODER_DIRECTORIES.items()
    }
    initial_fingerprints = {
        artifact.config.initial_decoder_fingerprint
        for artifact in artifacts.values()
    }
    if len(initial_fingerprints) != 1:
        raise RuntimeError("F/Fx/U decoder artifacts do not share initialization")
    return artifacts


def _decoder_payload(artifact: LoadedDecoderArtifact) -> dict[str, object]:
    config = artifact.config
    return {
        "variant": config.variant,
        "global_seed": config.global_seed,
        "trained_epochs": config.trained_epochs,
        "steps_per_epoch": config.steps_per_epoch,
        "optimizer_updates": config.trained_epochs * config.steps_per_epoch,
        "initial_decoder_fingerprint": config.initial_decoder_fingerprint,
        "artifact_fingerprint": artifact.artifact_fingerprint,
        "decoder_state_fingerprint": artifact.decoder_state_fingerprint,
        "receipt_sha256": artifact.receipt_sha256,
        "weights_sha256": artifact.weights_sha256,
        "train_log_sha256": artifact.train_log_sha256,
        "train_log_fingerprint": artifact.train_log_fingerprint,
    }


def _finalization_payload(
    *,
    root: Path,
    manifest_path: Path,
    config_path: Path,
    decision_rule_path: Path,
    freeze_path: Path,
    source_digest: str,
    config: object,
    d_r_bundle: object,
    d_v_bundle: object,
    artifacts: Mapping[str, LoadedDecoderArtifact],
    verified_base_payload: Mapping[str, str],
    calibration_workers: int,
    nofile_soft: int,
    nofile_hard: int,
    scientific_input_fingerprint: str,
) -> dict[str, object]:
    training = config.training
    return {
        "schema_version": FINALIZATION_SCHEMA,
        "method": "CURE-Lite",
        "stage": "Stage-A",
        "operation": "post_training_calibration_evaluation_and_publication",
        "training_entrypoint_called": False,
        "optimizer_updates_during_this_operation": 0,
        "runtime_splits": ["D_R", "D_V"],
        "unused_split": "D_T",
        "stage_run": str(root),
        "global_seed": training.global_seed,
        "trained_epochs": training.epochs,
        "steps_per_epoch": training.steps_per_epoch,
        "source_tree_digest": source_digest,
        "manifest_file_sha256": file_sha256(manifest_path),
        "stage_config_sha256": file_sha256(config_path),
        "decision_rule_sha256": file_sha256(decision_rule_path),
        "protocol_freeze_sha256": file_sha256(freeze_path),
        "finalization_tool_sha256": file_sha256(Path(__file__).resolve(strict=True)),
        "verified_base_run_identity": dict(verified_base_payload),
        "cache_indexes": {
            "d_r_base_index_fingerprint": d_r_bundle.base_index_fingerprint,
            "d_r_base_index_sha256": d_r_bundle.base_index_sha256,
            "d_r_state_index_fingerprint": d_r_bundle.state_index_fingerprint,
            "d_r_state_index_sha256": d_r_bundle.state_index_sha256,
            "d_v_base_index_fingerprint": d_v_bundle.base_index_fingerprint,
            "d_v_base_index_sha256": d_v_bundle.base_index_sha256,
        },
        "decoders": {
            method: _decoder_payload(artifacts[method])
            for method in ("F", "F×", "U")
        },
        "shared_initial_decoder_fingerprint": (
            artifacts["F"].config.initial_decoder_fingerprint
        ),
        "calibration_workers": calibration_workers,
        "execution_file_limit": {
            "soft": nofile_soft,
            "hard": nofile_hard,
        },
        "scientific_input_tree_fingerprint": scientific_input_fingerprint,
    }


def _write_or_require_same(path: Path, payload: Mapping[str, object]) -> None:
    if path.is_symlink():
        raise RuntimeError(f"Stage-A receipt may not be a symlink: {path.name}")
    if path.exists():
        if _strict_json(path, name=f"existing {path.name}") != dict(payload):
            raise RuntimeError(f"existing Stage-A receipt differs: {path.name}")
        return
    _write_new_json(path, payload)


def _validate_frozen_inputs(
    *,
    args: argparse.Namespace,
    root: Path,
    manifest_path: Path,
    config_path: Path,
    decision_rule_path: Path,
    freeze_path: Path,
    reference_base_root: Path,
    config: object,
    manifest: object,
) -> tuple[dict[str, Any], object]:
    freeze = _json_object(freeze_path, name="Stage-A protocol freeze")
    rule = _json_object(decision_rule_path, name="Stage-A decision rule")
    contract = load_base_cache_pair_contract(
        args.d_r_base_index,
        args.d_v_base_index,
    )
    validate_protocol_freeze(
        freeze,
        manifest_path=manifest_path,
        stage_config_path=config_path,
        decision_rule_path=decision_rule_path,
        d_r_index_path=contract.d_r_index_path,
        d_v_index_path=contract.d_v_index_path,
        stage_run_path=root,
    )
    if _resolve_frozen_path(
        freeze.get("reference_base_output"),
        name="reference_base_output",
    ) != reference_base_root:
        raise RuntimeError("reference Base run differs from protocol freeze")
    _validate_decision_rule(
        rule,
        dataset=manifest.dataset,
        seed=config.training.global_seed,
        stage_config_sha256=file_sha256(config_path),
    )
    return freeze, contract


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = _require_regular_directory(args.stage_run, name="Stage-A run")
    marker = _guard_stage_tree(root)
    nofile_soft, nofile_hard = _require_execution_capacity(
        args.calibration_workers
    )

    manifest_path = args.manifest.expanduser().resolve(strict=True)
    config_path = args.config.expanduser().resolve(strict=True)
    decision_rule_path = args.decision_rule.expanduser().resolve(strict=True)
    freeze_path = args.protocol_freeze.expanduser().resolve(strict=True)
    reference_base_root = _require_regular_directory(
        args.reference_base_run,
        name="reference Base run",
    )
    manifest = load_and_validate_manifest(manifest_path)
    config = load_stage_a_config(config_path)
    _, contract = _validate_frozen_inputs(
        args=args,
        root=root,
        manifest_path=manifest_path,
        config_path=config_path,
        decision_rule_path=decision_rule_path,
        freeze_path=freeze_path,
        reference_base_root=reference_base_root,
        config=config,
        manifest=manifest,
    )

    preprocess: PreprocessConfig = contract.preprocessing
    d_r_dataset = ManifestImageDataset(
        manifest,
        "D_R",
        preprocess,
        manifest_path=manifest_path,
    )
    d_v_dataset = ManifestImageDataset(
        manifest,
        "D_V",
        preprocess,
        manifest_path=manifest_path,
    )
    _check_dataset_pair(d_r_dataset, d_v_dataset)
    _preflight_stage_a_device(config.device)

    source_digest = _source_tree_digest()
    paths = _receipt_paths(root)
    if _strict_json(
        paths["config"],
        name="Stage-A config receipt",
    ) != _config_receipt(config, source_digest):
        raise RuntimeError("Stage-A config receipt differs from frozen config/source")

    verified_base_identity = load_verified_reference_base_run_identity(
        reference_base_root
    )
    verified_base_payload = _verified_base_run_payload(verified_base_identity)
    base_fingerprint = verified_base_payload["base_fingerprint"]
    local_d_r_base_index = root / "d_r" / "base_cache" / "index.json"
    local_d_v_base_index = root / "d_v" / "base_cache" / "index.json"
    if (
        file_sha256(local_d_r_base_index)
        != file_sha256(contract.d_r_index_path)
        or file_sha256(local_d_v_base_index)
        != file_sha256(contract.d_v_index_path)
    ):
        raise RuntimeError("materialized Stage-A base-cache indexes changed")

    d_v_bundle = load_d_v_cache_bundle(
        local_d_v_base_index,
        d_v_dataset,
        expected_base_fingerprint=base_fingerprint,
    )
    d_r_bundle = load_d_r_cache_bundle(
        root / "d_r" / "state_cache" / "index.json",
        d_r_dataset,
        expected_base_fingerprint=base_fingerprint,
    )
    _verified_base_run_payload(
        verified_base_identity,
        expected_base_fingerprint=d_v_bundle.base_fingerprint,
        expected_base_state_fingerprint=d_v_bundle.base_state_fingerprint,
    )
    artifacts = _load_decoders(root)
    prepared_training = prepare_gate2_training(d_r_bundle)

    input_directories, input_files, input_fingerprint = (
        _scientific_input_inventory(root)
    )
    state = _build_downstream_state(
        config=config,
        d_r_bundle=d_r_bundle,
        d_v_bundle=d_v_bundle,
        factual_artifact=artifacts["F"],
        factual_exposure_matched_artifact=artifacts["F×"],
        uniform_artifact=artifacts["U"],
        prepared_training=prepared_training,
        calibration_workers=args.calibration_workers,
        calibration_progress=_calibration_progress,
    )
    if _strict_json(
        paths["anchor"],
        name="Stage-A anchor receipt",
    ) != _anchor_receipt_payload(state.anchor):
        raise RuntimeError("recomputed Stage-A anchor differs from its existing receipt")
    if _strict_json(
        paths["support"],
        name="Stage-A support receipt",
    ) != _support_receipt_payload(
        state.support_summary,
        state.config.support_requirements,
    ):
        raise RuntimeError("recomputed Stage-A support differs from its existing receipt")

    efficiency = _measure_stage_a_efficiency(state)
    calibration_payload = _calibration_receipt_payload(state.calibration)
    results_payload = _results_receipt_payload(state.results, state.calibration)
    efficiency_payload = efficiency.canonical_payload()
    finalization_payload = _finalization_payload(
        root=root,
        manifest_path=manifest_path,
        config_path=config_path,
        decision_rule_path=decision_rule_path,
        freeze_path=freeze_path,
        source_digest=source_digest,
        config=config,
        d_r_bundle=d_r_bundle,
        d_v_bundle=d_v_bundle,
        artifacts=artifacts,
        verified_base_payload=verified_base_payload,
        calibration_workers=args.calibration_workers,
        nofile_soft=nofile_soft,
        nofile_hard=nofile_hard,
        scientific_input_fingerprint=input_fingerprint,
    )

    _require_same_input_inventory(
        root,
        input_directories,
        input_files,
        input_fingerprint,
    )
    for artifact in artifacts.values():
        artifact.verify_unchanged()
    verified_base_identity.verify_unchanged()
    if _source_tree_digest() != source_digest:
        raise RuntimeError("CURE-Lite method source changed during completion")

    _write_or_require_same(paths["calibration"], calibration_payload)
    _write_or_require_same(paths["results"], results_payload)
    _write_or_require_same(paths["efficiency"], efficiency_payload)
    _write_or_require_same(
        paths["config"].parent / _FINALIZATION_NAME,
        finalization_payload,
    )

    _require_same_input_inventory(
        root,
        input_directories,
        input_files,
        input_fingerprint,
    )
    for artifact in artifacts.values():
        artifact.verify_unchanged()
    _verified_base_run_payload(
        verified_base_identity,
        expected_base_fingerprint=d_v_bundle.base_fingerprint,
        expected_base_state_fingerprint=d_v_bundle.base_state_fingerprint,
    )
    if _source_tree_digest() != source_digest:
        raise RuntimeError("CURE-Lite method source changed before publication")

    directories_with_marker, files_with_marker = _tree_inventory(root)
    if files_with_marker.get(_INCOMPLETE_NAME) != file_sha256(marker):
        raise RuntimeError("Stage-A .incomplete marker changed before publication")
    artifact_files = dict(files_with_marker)
    artifact_files.pop(_INCOMPLETE_NAME)
    complete = _complete_receipt(
        state,
        verified_base_identity=verified_base_identity,
        efficiency=efficiency,
        source_digest=source_digest,
        artifact_directories=directories_with_marker,
        artifact_files=artifact_files,
    )
    _write_or_require_same(root / _COMPLETE_NAME, complete)
    marker.unlink()

    completed = _bind_published_stage_a_run(
        root,
        state,
        efficiency,
        complete,
        d_r_dataset,
        d_v_dataset,
        verified_base_identity=verified_base_identity,
        calibration_workers=args.calibration_workers,
        calibration_progress=_calibration_progress,
    )
    print(
        json.dumps(
            _summary_payload(completed, manifest),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
