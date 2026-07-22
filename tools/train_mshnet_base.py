#!/usr/bin/env python3
"""Train one formal MSHNet base with D_B-fit/D_B-select model selection.

The command builds a new data view containing exactly the manifest's D_B rows,
partitions those rows by grouping keys, and starts the CURE-Lite-owned MSHNet
runner.  D_R, D_V, and D_T asset paths are not copied or passed to that runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.provenance import (
    CURE_LITE_METHOD_VERSION,
    FORMAL_BASE_FINAL_SCHEMA,
    FORMAL_BASE_PREFLIGHT_SCHEMA,
    BaseCheckpointSelection,
    BaseTrainingProvenance,
    deterministic_base_checkpoint_selection,
    validate_formal_base_training_run,
)
from cure_lite.splits import SplitManifest, SplitRecord, load_and_validate_manifest


VIEW_SCHEMA = "cure-lite-mshnet-db-view-v2"
RUNNER_CONTRACT = "cure-lite-pinned-mshnet-runner-v1"
SOURCE_FILES = (
    "model/MSHNet.py",
    "model/loss.py",
    "utils/data.py",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be strictly between zero and one")
    return parsed


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError("value must be true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mshnet-repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--selection-fraction", type=_fraction, default=0.2)
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=_positive_int, default=4)
    parser.add_argument("--epochs", type=_positive_int, default=800)
    parser.add_argument("--lr", type=_positive_float, default=0.05)
    parser.add_argument("--warm-epoch", type=_nonnegative_int, default=5)
    parser.add_argument("--base-size", type=_positive_int, default=256)
    parser.add_argument("--crop-size", type=_positive_int, default=256)
    parser.add_argument("--num-workers", type=_nonnegative_int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--multi-gpus", type=_boolean, default=False)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a JSON object")
    return value


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _git_value(repository: Path, *arguments: str) -> str:
    return _git_bytes(repository, *arguments).decode("utf-8").strip()


def _repository_identity(
    repository: Path,
    *,
    expected_commit: str,
) -> tuple[str, str, dict[str, str]]:
    commit = _git_value(repository, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise RuntimeError(
            f"MSHNet commit mismatch: expected {expected_commit}, got {commit}"
        )
    tree = _git_value(repository, "rev-parse", "HEAD^{tree}")
    source_hashes: dict[str, str] = {}
    for relative in SOURCE_FILES:
        source = repository / relative
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"required MSHNet source is not a regular file: {relative}")
        worktree_hash = file_sha256(source)
        committed_content = _git_bytes(repository, "show", f"{commit}:{relative}")
        committed_hash = hashlib.sha256(committed_content).hexdigest()
        if worktree_hash != committed_hash:
            raise RuntimeError(
                f"MSHNet source differs from commit {commit}: {relative}"
            )
        source_hashes[relative] = worktree_hash
    return commit, tree, source_hashes


def _resolve_manifest_file(manifest_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    if candidate.is_symlink():
        raise ValueError(f"dataset asset must be a regular file: {raw_path}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"dataset asset must be a regular file: {raw_path}")
    return resolved


def _copy_exact(source: Path, destination: Path) -> str:
    source_hash = file_sha256(source)
    shutil.copyfile(source, destination)
    if file_sha256(source) != source_hash:
        raise RuntimeError(f"dataset asset changed while being copied: {source}")
    if file_sha256(destination) != source_hash:
        raise RuntimeError(f"dataset-view copy differs from its source: {source}")
    return source_hash


def _view_row(
    *,
    manifest_path: Path,
    record: SplitRecord,
    role: str,
    view_id: str,
    images: Path,
    masks: Path,
) -> dict[str, str]:
    if record.split != "D_B":
        raise ValueError("base-training data view accepts only D_B records")
    if record.mask is None:
        raise ValueError(f"D_B sample {record.sample_id!r} has no mask")
    source_image = _resolve_manifest_file(manifest_path, record.image)
    source_mask = _resolve_manifest_file(manifest_path, record.mask)
    view_image = images / f"{view_id}.png"
    view_mask = masks / f"{view_id}.png"
    image_hash = _copy_exact(source_image, view_image)
    mask_hash = _copy_exact(source_mask, view_mask)
    return {
        "view_id": view_id,
        "sample_id": record.sample_id,
        "split": "D_B",
        "role": role,
        "image": str(view_image.relative_to(images.parent)),
        "mask": str(view_mask.relative_to(masks.parent)),
        "image_sha256": image_hash,
        "mask_sha256": mask_hash,
    }


def build_d_b_dataset_view(
    manifest: SplitManifest,
    manifest_path: Path,
    selection: BaseCheckpointSelection,
    destination: Path,
) -> dict[str, Any]:
    """Build the exact D_B-fit/D_B-select root consumed by the runner."""

    selection.validate_against(manifest)
    destination.mkdir(parents=True, exist_ok=False)
    images = destination / "images"
    masks = destination / "masks"
    images.mkdir()
    masks.mkdir()
    by_id = {record.sample_id: record for record in manifest.records_for("D_B")}
    fit_records = tuple(by_id[sample_id] for sample_id in selection.fit_sample_ids)
    select_records = tuple(by_id[sample_id] for sample_id in selection.select_sample_ids)
    if not fit_records or not select_records:
        raise ValueError("D_B-fit and D_B-select must both be non-empty")

    rows: list[dict[str, str]] = []
    fit_view_ids: list[str] = []
    select_view_ids: list[str] = []
    for index, record in enumerate(fit_records):
        view_id = f"db_fit_{index:06d}"
        fit_view_ids.append(view_id)
        rows.append(
            _view_row(
                manifest_path=manifest_path,
                record=record,
                role="train",
                view_id=view_id,
                images=images,
                masks=masks,
            )
        )
    for index, record in enumerate(select_records):
        view_id = f"db_select_{index:06d}"
        select_view_ids.append(view_id)
        rows.append(
            _view_row(
                manifest_path=manifest_path,
                record=record,
                role="validation",
                view_id=view_id,
                images=images,
                masks=masks,
            )
        )

    train_file = destination / "trainval.txt"
    select_file = destination / "test.txt"
    train_file.write_text(
        "".join(f"{view_id}\n" for view_id in fit_view_ids),
        encoding="utf-8",
    )
    select_file.write_text(
        "".join(f"{view_id}\n" for view_id in select_view_ids),
        encoding="utf-8",
    )
    payload: dict[str, Any] = {
        "schema_version": VIEW_SCHEMA,
        "method_version": CURE_LITE_METHOD_VERSION,
        "split_manifest_fingerprint": manifest.fingerprint,
        "roles": {"train": "D_B-fit", "validation": "D_B-select"},
        "native_split_files_sha256": {
            "trainval.txt": file_sha256(train_file),
            "test.txt": file_sha256(select_file),
        },
        "records": rows,
    }
    payload["view_fingerprint"] = stable_fingerprint(payload)
    _write_json(destination / "index.json", payload)
    return payload


def _verify_dataset_view(view_root: Path, index: Mapping[str, Any]) -> None:
    records = index.get("records")
    split_hashes = index.get("native_split_files_sha256")
    if not isinstance(records, list) or not isinstance(split_hashes, Mapping):
        raise RuntimeError("dataset-view index is incomplete")
    index_payload = dict(index)
    declared_fingerprint = index_payload.pop("view_fingerprint", None)
    if (
        not isinstance(declared_fingerprint, str)
        or declared_fingerprint != stable_fingerprint(index_payload)
    ):
        raise RuntimeError("dataset-view fingerprint does not match its contents")
    if index.get("roles") != {"train": "D_B-fit", "validation": "D_B-select"}:
        raise RuntimeError("dataset-view roles must be D_B-fit and D_B-select")
    expected_files = {"index.json", "trainval.txt", "test.txt"}
    for filename in ("trainval.txt", "test.txt"):
        expected_hash = split_hashes.get(filename)
        path = view_root / filename
        if not isinstance(expected_hash, str) or file_sha256(path) != expected_hash:
            raise RuntimeError(f"dataset-view {filename} differs from its index")
    seen_view_ids: set[str] = set()
    role_view_ids: dict[str, list[str]] = {"train": [], "validation": []}
    for row in records:
        if not isinstance(row, Mapping) or row.get("split") != "D_B":
            raise RuntimeError("dataset-view records must all belong to D_B")
        role = row.get("role")
        view_id = row.get("view_id")
        if role not in role_view_ids:
            raise RuntimeError("dataset-view record has an unknown role")
        if (
            not isinstance(view_id, str)
            or not view_id
            or view_id in seen_view_ids
        ):
            raise RuntimeError("dataset-view record has an invalid view_id")
        seen_view_ids.add(view_id)
        role_view_ids[str(role)].append(view_id)
        for kind in ("image", "mask"):
            relative = row.get(kind)
            expected_hash = row.get(f"{kind}_sha256")
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                raise RuntimeError("dataset-view asset record is incomplete")
            candidate = view_root / relative
            if candidate.is_symlink() or not candidate.is_file():
                raise RuntimeError("dataset-view asset is not a regular file")
            if file_sha256(candidate) != expected_hash:
                raise RuntimeError("dataset-view asset differs from its index")
            expected_files.add(relative)
    if (view_root / "trainval.txt").read_text(encoding="utf-8").splitlines() != (
        role_view_ids["train"]
    ):
        raise RuntimeError("trainval.txt does not match the D_B-fit records")
    if (view_root / "test.txt").read_text(encoding="utf-8").splitlines() != (
        role_view_ids["validation"]
    ):
        raise RuntimeError("test.txt does not match the D_B-select records")
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in view_root.rglob("*"):
        relative = str(path.relative_to(view_root))
        if path.is_symlink():
            raise RuntimeError("dataset view may not contain symbolic links")
        if path.is_dir():
            actual_directories.add(relative)
        elif path.is_file():
            actual_files.add(relative)
        else:
            raise RuntimeError("dataset view contains an unknown filesystem entry")
    if actual_directories != {"images", "masks"}:
        raise RuntimeError("dataset view has unexpected directories")
    if actual_files != expected_files:
        raise RuntimeError("dataset view has unexpected or missing files")


def _recipe(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "native_entrypoint": RUNNER_CONTRACT,
        "mode": "train",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "warm_epoch": args.warm_epoch,
        "base_size": args.base_size,
        "crop_size": args.crop_size,
        "num_workers": args.num_workers,
        "device": args.device,
        "multi_gpus": args.multi_gpus,
        "seed": args.seed,
        "selection_fraction": args.selection_fraction,
        "selection_seed": args.selection_seed,
        "selection_metric": "D_B-select/global-binary-mIoU@logit>0",
        "selection_tie_break": "earliest_epoch",
        "model_output_format": "raw_state_dict",
        "training_split": "D_B-fit",
        "validation_split": "D_B-select",
        "checkpoint_selection_split": "D_B-select",
        "external_validation_split": None,
        "resume": False,
        "resume_path": None,
        "input_checkpoint": None,
    }


def _runner_contract(
    runner: Path,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "contract": RUNNER_CONTRACT,
        "entrypoint": runner.name,
        "runner_sha256": file_sha256(runner),
        "upstream_sources_sha256": dict(source_hashes),
    }


def _native_command(
    args: argparse.Namespace,
    *,
    python_executable: Path,
    runner: Path,
    repository: Path,
    commit: str,
    tree: str,
    source_hashes: Mapping[str, str],
    view_root: Path,
    native_output: Path,
) -> list[str]:
    return [
        str(python_executable),
        str(runner),
        "--mshnet-repo",
        str(repository),
        "--expected-commit",
        commit,
        "--expected-tree",
        tree,
        "--model-source-sha256",
        source_hashes["model/MSHNet.py"],
        "--loss-source-sha256",
        source_hashes["model/loss.py"],
        "--data-source-sha256",
        source_hashes["utils/data.py"],
        "--dataset-dir",
        str(view_root),
        "--output-dir",
        str(native_output),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--warm-epoch",
        str(args.warm_epoch),
        "--base-size",
        str(args.base_size),
        "--crop-size",
        str(args.crop_size),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--multi-gpus",
        "true" if args.multi_gpus else "false",
        "--seed",
        str(args.seed),
    ]


def _child_environment(seed: int) -> dict[str, str]:
    inherited = (
        "CUDA_HOME",
        "CUDA_VISIBLE_DEVICES",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "MKL_NUM_THREADS",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NVIDIA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "PATH",
    )
    environment = {key: os.environ[key] for key in inherited if key in os.environ}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = str(seed)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _runtime_identity(python_executable: Path) -> dict[str, Any]:
    resolved = python_executable.resolve(strict=True)
    return {
        "python_executable": str(python_executable),
        "python_executable_resolved": str(resolved),
        "python_executable_sha256": file_sha256(resolved),
        "python_version": sys.version.split()[0],
    }


def _validate_child_outputs(
    native_output: Path,
    *,
    seed: int,
    source_hashes: Mapping[str, str],
) -> tuple[Path, Path, Path, dict[str, Any]]:
    run_dir = native_output / f"MSHNet-seed{seed:06d}"
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise RuntimeError("runner output directory is missing")
    checkpoint = run_dir / "weight.pkl"
    metric_path = run_dir / "epoch_metrics.jsonl"
    summary_path = run_dir / "training_summary.json"
    expected_files = {checkpoint.name, metric_path.name, summary_path.name}
    if {path.name for path in run_dir.iterdir()} != expected_files:
        raise RuntimeError("runner output has unexpected or missing files")
    for path in (checkpoint, metric_path, summary_path):
        if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"runner output is incomplete: {path.name}")
    summary = _load_json(summary_path, name="training summary")
    required = {
        "schema_version": "cure-lite-pinned-mshnet-training-summary-v1",
        "runner_contract": RUNNER_CONTRACT,
        "status": "completed",
        "seed": seed,
        "selection_metric": "D_B-select/global-binary-mIoU@logit>0",
        "selection_tie_break": "earliest_epoch",
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"training summary mismatch for {key}")
    upstream = summary.get("upstream")
    if not isinstance(upstream, Mapping) or upstream.get("sources_sha256") != dict(
        source_hashes
    ):
        raise RuntimeError("training summary source identities do not match")
    checkpoint_block = summary.get("checkpoint")
    metric_block = summary.get("epoch_metrics")
    if not isinstance(checkpoint_block, Mapping) or not isinstance(
        metric_block, Mapping
    ):
        raise RuntimeError("training summary artifact records are incomplete")
    if checkpoint_block.get("sha256") != file_sha256(checkpoint):
        raise RuntimeError("training summary model SHA256 mismatch")
    if metric_block.get("sha256") != file_sha256(metric_path):
        raise RuntimeError("training summary metric SHA256 mismatch")
    return checkpoint, metric_path, summary_path, summary


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    repository = args.mshnet_repo.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    python_executable = args.python_executable.expanduser().absolute()
    launcher = Path(__file__).resolve(strict=True)
    runner = launcher.with_name("run_pinned_mshnet_train.py").resolve(strict=True)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not repository.is_dir():
        raise ValueError("mshnet-repo must be a directory")
    try:
        output.relative_to(repository)
    except ValueError:
        pass
    else:
        raise ValueError("output must be outside the MSHNet repository")
    if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
        raise ValueError(f"Python executable is not executable: {python_executable}")
    if not runner.is_file() or runner.is_symlink():
        raise ValueError("run_pinned_mshnet_train.py must be a regular file")
    if args.base_size % 16 != 0 or args.crop_size % 16 != 0:
        raise ValueError("base-size and crop-size must be divisible by 16")

    commit, tree, source_hashes = _repository_identity(
        repository,
        expected_commit=args.expected_commit,
    )
    manifest_file_hash = file_sha256(manifest_path)
    manifest = load_and_validate_manifest(manifest_path)
    selection = deterministic_base_checkpoint_selection(
        manifest,
        select_fraction=args.selection_fraction,
        seed=args.selection_seed,
        min_fit_samples=args.batch_size,
        min_select_samples=1,
    )
    if len(selection.fit_sample_ids) < args.batch_size:
        raise ValueError("D_B-fit must contain at least one complete training batch")

    output.mkdir(parents=True, exist_ok=False)
    view_root = output / "dataset_view"
    view = build_d_b_dataset_view(manifest, manifest_path, selection, view_root)
    _verify_dataset_view(view_root, view)
    native_output = output / "native_runs"
    native_output.mkdir()
    logs = output / "logs"
    logs.mkdir()
    command = _native_command(
        args,
        python_executable=python_executable,
        runner=runner,
        repository=repository,
        commit=commit,
        tree=tree,
        source_hashes=source_hashes,
        view_root=view_root,
        native_output=native_output,
    )
    recipe = _recipe(args)
    trainer_contract = _runner_contract(runner, source_hashes)
    environment = _child_environment(args.seed)
    launcher_hash = file_sha256(launcher)
    preflight: dict[str, Any] = {
        "schema_version": FORMAL_BASE_PREFLIGHT_SCHEMA,
        "method_version": CURE_LITE_METHOD_VERSION,
        "status": "ready_for_fresh_native_training",
        "split_manifest_fingerprint": manifest.fingerprint,
        "split_manifest_file_sha256": manifest_file_hash,
        "dataset_view_fingerprint": view["view_fingerprint"],
        "dataset_view_index_sha256": file_sha256(view_root / "index.json"),
        "dataset_view_roles": {"train": "D_B-fit", "validation": "D_B-select"},
        "dataset_view_records": view["records"],
        "checkpoint_selection": selection.to_mapping(),
        "recipe": recipe,
        "recipe_fingerprint": stable_fingerprint(recipe),
        "upstream": {
            "commit": commit,
            "tree": tree,
            "tracked_python_sources_sha256": source_hashes,
        },
        "native_trainer_contract": trainer_contract,
        "launcher_sha256": launcher_hash,
        "runtime": _runtime_identity(python_executable),
        "native_command": command,
        "native_child_dataset_root": str(view_root),
        "native_child_environment": environment,
        "fresh_output_policy": "new_output_no_resume_no_checkpoint_fallback",
    }
    preflight_path = output / "preflight_receipt.json"
    _write_json(preflight_path, preflight)
    preflight_hash = file_sha256(preflight_path)

    stdout_path = logs / "native.stdout.log"
    stderr_path = logs / "native.stderr.log"
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        completed = subprocess.run(
            command,
            cwd=output,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"MSHNet base training exited with code {completed.returncode}; "
            f"see {stderr_path}"
        )

    commit_after, tree_after, source_hashes_after = _repository_identity(
        repository,
        expected_commit=args.expected_commit,
    )
    if (commit_after, tree_after, source_hashes_after) != (commit, tree, source_hashes):
        raise RuntimeError("MSHNet experiment version changed during training")
    if file_sha256(manifest_path) != manifest_file_hash:
        raise RuntimeError("split manifest changed during training")
    if file_sha256(launcher) != launcher_hash:
        raise RuntimeError("base-training launcher changed during training")
    if _runner_contract(runner, source_hashes) != trainer_contract:
        raise RuntimeError("MSHNet runner contract changed during training")
    if file_sha256(preflight_path) != preflight_hash:
        raise RuntimeError("preflight record changed during training")
    if file_sha256(view_root / "index.json") != preflight[
        "dataset_view_index_sha256"
    ]:
        raise RuntimeError("dataset-view index changed during training")
    _verify_dataset_view(view_root, view)

    checkpoint, metric_path, summary_path, summary = _validate_child_outputs(
        native_output,
        seed=args.seed,
        source_hashes=source_hashes,
    )
    provenance = BaseTrainingProvenance.from_manifest(manifest, checkpoint)
    provenance_path = output / "base_training_provenance.json"
    _write_json(provenance_path, provenance.to_mapping())
    final: dict[str, Any] = {
        "schema_version": FORMAL_BASE_FINAL_SCHEMA,
        "method_version": CURE_LITE_METHOD_VERSION,
        "status": "completed",
        "split_manifest_fingerprint": manifest.fingerprint,
        "native_exit_code": completed.returncode,
        "preflight_receipt_sha256": preflight_hash,
        "dataset_view_fingerprint": view["view_fingerprint"],
        "recipe_fingerprint": preflight["recipe_fingerprint"],
        "checkpoint_selection_fingerprint": selection.fingerprint,
        "upstream_commit": commit,
        "upstream_tree": tree,
        "launcher_sha256": launcher_hash,
        "native_trainer_contract": trainer_contract,
        "checkpoint": {
            "path": str(checkpoint.relative_to(output)),
            "name": checkpoint.name,
            "format": "raw_state_dict",
            "sha256": file_sha256(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        },
        "base_training_provenance": {
            "path": str(provenance_path.relative_to(output)),
            "sha256": file_sha256(provenance_path),
            "fingerprint": provenance.fingerprint,
        },
        "training_metrics": {
            "path": str(metric_path.relative_to(output)),
            "sha256": file_sha256(metric_path),
            "selection_metric": summary["selection_metric"],
            "best_epoch": summary["best_epoch"],
            "best_d_b_select_miou": summary["best_d_b_select_miou"],
        },
        "training_summary": {
            "path": str(summary_path.relative_to(output)),
            "sha256": file_sha256(summary_path),
        },
        "logs": {
            "stdout": str(stdout_path.relative_to(output)),
            "stderr": str(stderr_path.relative_to(output)),
            "stdout_sha256": file_sha256(stdout_path),
            "stderr_sha256": file_sha256(stderr_path),
        },
    }
    final_path = output / "final_receipt.json"
    _write_json(final_path, final)
    if file_sha256(checkpoint) != final["checkpoint"]["sha256"]:
        raise RuntimeError("selected model changed while writing the final record")
    formal_identity = validate_formal_base_training_run(
        final_path,
        provenance_path,
        manifest,
        checkpoint,
    )

    print(f"checkpoint={checkpoint}")
    print(f"checkpoint_sha256={final['checkpoint']['sha256']}")
    print(f"best_epoch={summary['best_epoch']}")
    print(f"best_d_b_select_miou={summary['best_d_b_select_miou']:.12f}")
    print(f"preflight_receipt={preflight_path}")
    print(f"final_receipt={final_path}")
    print(f"base_training_provenance={provenance_path}")
    print(f"base_training_identity={formal_identity.provenance_fingerprint}")


if __name__ == "__main__":
    main()
