#!/usr/bin/env python3
"""Run the fixed MSHNet training recipe on a prepared D_B data view.

This runner owns the optimization loop.  It imports exactly three source files
from the requested MSHNet checkout: ``model/MSHNet.py``, ``model/loss.py``, and
``utils/data.py``.  The runner has no parameter for loading an earlier model.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adagrad
from torch.utils.data import DataLoader


# The imported upstream sources remain byte-for-byte unchanged during the run.
sys.dont_write_bytecode = True


RUNNER_CONTRACT = "cure-lite-pinned-mshnet-runner-v1"
SOURCE_FILES = (
    "model/MSHNet.py",
    "model/loss.py",
    "utils/data.py",
)
SELECTION_METRIC = "D_B-select/global-binary-mIoU@logit>0"


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


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise argparse.ArgumentTypeError("value must be true or false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mshnet-repo", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--model-source-sha256", required=True)
    parser.add_argument("--loss-source-sha256", required=True)
    parser.add_argument("--data-source-sha256", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_paths(args: argparse.Namespace, repository: Path) -> dict[str, Path]:
    expected_hashes = {
        "model/MSHNet.py": args.model_source_sha256.lower(),
        "model/loss.py": args.loss_source_sha256.lower(),
        "utils/data.py": args.data_source_sha256.lower(),
    }
    paths: dict[str, Path] = {}
    for relative in SOURCE_FILES:
        path = repository / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"required MSHNet source is not a regular file: {relative}")
        actual = _file_sha256(path)
        if actual != expected_hashes[relative]:
            raise RuntimeError(f"MSHNet source SHA256 mismatch: {relative}")
        paths[relative] = path.resolve(strict=True)
    return paths


def _verify_repository(
    args: argparse.Namespace,
    repository: Path,
) -> dict[str, Path]:
    if _git_value(repository, "rev-parse", "HEAD") != args.expected_commit:
        raise RuntimeError("MSHNet commit differs from the configured experiment version")
    if _git_value(repository, "rev-parse", "HEAD^{tree}") != args.expected_tree:
        raise RuntimeError("MSHNet tree differs from the configured experiment version")
    return _source_paths(args, repository)


def _load_module(module_name: str, source: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create a module specification for {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")


def _raw_state_dict(model: nn.Module) -> OrderedDict[str, Tensor]:
    source = model.module if isinstance(model, nn.DataParallel) else model
    return OrderedDict(
        (name, value.detach().cpu().contiguous())
        for name, value in source.state_dict().items()
    )


def _save_raw_state_dict(model: nn.Module, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.writing")
    if temporary.exists() or destination.exists():
        raise FileExistsError(f"model output already exists: {destination}")
    torch.save(_raw_state_dict(model), temporary)
    os.replace(temporary, destination)


def _train_epoch(
    *,
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: Adagrad,
    loss_function: nn.Module,
    downsample: nn.Module,
    device: torch.device,
    warm_epoch: int,
    epoch: int,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0
    use_auxiliary = epoch > warm_epoch
    for images, target in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        target = target.to(device, non_blocking=device.type == "cuda")
        auxiliary, logits = model(images, use_auxiliary)
        loss = loss_function(logits, target, warm_epoch, epoch)
        scaled_target = target
        for index, auxiliary_logits in enumerate(auxiliary):
            if index > 0:
                scaled_target = downsample(scaled_target)
            loss = loss + loss_function(
                auxiliary_logits,
                scaled_target,
                warm_epoch,
                epoch,
            )
        loss = loss / (len(auxiliary) + 1)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = int(images.shape[0])
        loss_sum += float(loss.detach().item()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        raise RuntimeError("D_B-fit produced no complete training batch")
    return loss_sum / sample_count


@torch.no_grad()
def _evaluate_miou(
    *,
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    warm_epoch: int,
    epoch: int,
) -> tuple[float, int, int]:
    model.eval()
    intersection = 0
    union = 0
    use_auxiliary = epoch > warm_epoch
    for images, target in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        target = target.to(device, non_blocking=device.type == "cuda")
        _, logits = model(images, use_auxiliary)
        prediction = logits > 0.0
        foreground = target > 0.5
        intersection += int(torch.logical_and(prediction, foreground).sum().item())
        union += int(torch.logical_or(prediction, foreground).sum().item())
    miou = float(intersection / union) if union > 0 else 0.0
    if not math.isfinite(miou):
        raise RuntimeError("D_B-select mIoU is not finite")
    return miou, intersection, union


def main() -> None:
    args = parse_args()
    if args.base_size % 16 != 0 or args.crop_size % 16 != 0:
        raise ValueError("base-size and crop-size must be divisible by 16")
    repository = args.mshnet_repo.expanduser().resolve(strict=True)
    dataset_root = args.dataset_dir.expanduser().resolve(strict=True)
    output_root = args.output_dir.expanduser().resolve(strict=True)
    if not repository.is_dir() or not dataset_root.is_dir() or not output_root.is_dir():
        raise ValueError("repository, dataset-dir, and output-dir must be directories")
    if any(output_root.iterdir()):
        raise FileExistsError("output-dir must be empty at the start of training")

    source_paths = _verify_repository(args, repository)
    model_module = _load_module(
        f"_cure_lite_mshnet_model_{args.model_source_sha256[:12]}",
        source_paths["model/MSHNet.py"],
    )
    loss_module = _load_module(
        f"_cure_lite_mshnet_loss_{args.loss_source_sha256[:12]}",
        source_paths["model/loss.py"],
    )
    data_module = _load_module(
        f"_cure_lite_mshnet_data_{args.data_source_sha256[:12]}",
        source_paths["utils/data.py"],
    )
    model_type = getattr(model_module, "MSHNet")
    loss_type = getattr(loss_module, "SLSIoULoss")
    dataset_type = getattr(data_module, "IRSTD_Dataset")

    _seed_everything(args.seed)
    device = _select_device(args.device)
    train_dataset = dataset_type(args, mode="train")
    select_dataset = dataset_type(args, mode="val")
    if len(train_dataset) < args.batch_size:
        raise ValueError("D_B-fit must contain at least one complete batch")
    if len(select_dataset) < 1:
        raise ValueError("D_B-select must contain at least one sample")
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    common_loader = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": _seed_worker,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
        **common_loader,
    )
    select_loader = DataLoader(
        select_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        **common_loader,
    )

    model: nn.Module = model_type(3)
    if args.multi_gpus:
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise RuntimeError("multi-gpus requires at least two visible CUDA devices")
        model = nn.DataParallel(model)
    model.to(device)
    optimizer = Adagrad(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
    )
    loss_function: nn.Module = loss_type()
    downsample = nn.MaxPool2d(2, 2)

    run_dir = output_root / f"MSHNet-seed{args.seed:06d}"
    run_dir.mkdir(parents=False, exist_ok=False)
    metric_path = run_dir / "epoch_metrics.jsonl"
    checkpoint_path = run_dir / "weight.pkl"
    best_miou = -1.0
    best_epoch = -1
    start_time = time.monotonic()
    with metric_path.open("x", encoding="utf-8") as metric_log:
        for epoch in range(args.epochs):
            train_loss = _train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                loss_function=loss_function,
                downsample=downsample,
                device=device,
                warm_epoch=args.warm_epoch,
                epoch=epoch,
            )
            miou, intersection, union = _evaluate_miou(
                model=model,
                loader=select_loader,
                device=device,
                warm_epoch=args.warm_epoch,
                epoch=epoch,
            )
            selected = miou > best_miou
            if selected:
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                _save_raw_state_dict(model, checkpoint_path)
                best_miou = miou
                best_epoch = epoch
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "d_b_select_miou": miou,
                "d_b_select_intersection": intersection,
                "d_b_select_union": union,
                "selected": selected,
                "best_epoch": best_epoch,
                "best_d_b_select_miou": best_miou,
            }
            metric_log.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            metric_log.write("\n")
            metric_log.flush()
            print(json.dumps(row, sort_keys=True), flush=True)

    if best_epoch < 0 or not checkpoint_path.is_file():
        raise RuntimeError("training did not produce a selected model")
    source_paths_after = _verify_repository(args, repository)
    if source_paths_after != source_paths:
        raise RuntimeError("MSHNet source paths changed during training")
    summary = {
        "schema_version": "cure-lite-pinned-mshnet-training-summary-v1",
        "runner_contract": RUNNER_CONTRACT,
        "status": "completed",
        "seed": args.seed,
        "device": str(device),
        "multi_gpus": args.multi_gpus,
        "epochs": args.epochs,
        "d_b_fit_samples": len(train_dataset),
        "d_b_select_samples": len(select_dataset),
        "selection_metric": SELECTION_METRIC,
        "selection_tie_break": "earliest_epoch",
        "best_epoch": best_epoch,
        "best_d_b_select_miou": best_miou,
        "elapsed_seconds": time.monotonic() - start_time,
        "checkpoint": {
            "path": checkpoint_path.name,
            "format": "raw_state_dict",
            "sha256": _file_sha256(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "epoch_metrics": {
            "path": metric_path.name,
            "sha256": _file_sha256(metric_path),
        },
        "upstream": {
            "commit": args.expected_commit,
            "tree": args.expected_tree,
            "sources_sha256": {
                relative: _file_sha256(source_paths[relative])
                for relative in SOURCE_FILES
            },
        },
    }
    _write_json(run_dir / "training_summary.json", summary)
    print(f"checkpoint={checkpoint_path}", flush=True)
    print(f"best_epoch={best_epoch}", flush=True)
    print(f"best_d_b_select_miou={best_miou:.12f}", flush=True)


if __name__ == "__main__":
    main()
