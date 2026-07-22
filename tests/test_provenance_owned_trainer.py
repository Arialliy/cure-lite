from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.provenance import (
    CURE_LITE_METHOD_VERSION,
    FORMAL_BASE_FINAL_SCHEMA,
    FORMAL_BASE_PREFLIGHT_SCHEMA,
    OWNED_MSHNET_SOURCE_PATHS,
    OWNED_NATIVE_TRAINER_CONTRACT,
    OWNED_NATIVE_TRAINER_ENTRYPOINT,
    BaseCheckpointSelection,
    BaseTrainingProvenance,
    BaseTrainingProvenanceError,
    validate_formal_base_training_run,
)
from cure_lite.splits import SplitManifest, SplitRecord


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _build_owned_runner_record(tmp_path: Path) -> dict[str, Any]:
    assets = tmp_path / "assets"
    assets.mkdir()

    def asset(name: str) -> tuple[str, str]:
        image = assets / f"{name}.png"
        mask = assets / f"{name}-mask.png"
        image.write_bytes(f"image-{name}".encode())
        mask.write_bytes(f"mask-{name}".encode())
        return str(image), str(mask)

    fit_image, fit_mask = asset("fit")
    select_image, select_mask = asset("select")
    dr_image, dr_mask = asset("dr")
    dv_image, dv_mask = asset("dv")
    dt_image, dt_mask = asset("dt")
    manifest = SplitManifest(
        dataset="owned-runner-toy",
        records=(
            SplitRecord("fit", "D_B", "fit-group", fit_image, fit_mask),
            SplitRecord(
                "select", "D_B", "select-group", select_image, select_mask
            ),
            SplitRecord("dr", "D_R", "dr-group", dr_image, dr_mask),
            SplitRecord("dv", "D_V", "dv-group", dv_image, dv_mask),
            SplitRecord("dt", "D_T", "dt-group", dt_image, dt_mask),
        ),
    )
    selection = BaseCheckpointSelection.from_manifest(
        manifest,
        fit_sample_ids=["fit"],
        select_sample_ids=["select"],
    )

    mshnet_repo = tmp_path / "MSHNet"
    upstream_source_sha256: dict[str, str] = {}
    for relative in OWNED_MSHNET_SOURCE_PATHS:
        source = mshnet_repo / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# {relative}\n", encoding="utf-8")
        upstream_source_sha256[relative] = file_sha256(source)

    tool_directory = tmp_path / "tools"
    tool_directory.mkdir()
    runner = tool_directory / OWNED_NATIVE_TRAINER_ENTRYPOINT
    runner.write_text("# CURE-Lite MSHNet runner\n", encoding="utf-8")

    run = tmp_path / "run"
    view = run / "dataset_view"
    images = view / "images"
    masks = view / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    rows: list[dict[str, str]] = []
    for view_id, sample_id, role, source_image, source_mask in (
        ("db_fit", "fit", "train", Path(fit_image), Path(fit_mask)),
        ("db_select", "select", "validation", Path(select_image), Path(select_mask)),
    ):
        image_path = images / f"{view_id}.png"
        mask_path = masks / f"{view_id}.png"
        image_path.write_bytes(source_image.read_bytes())
        mask_path.write_bytes(source_mask.read_bytes())
        rows.append(
            {
                "view_id": view_id,
                "sample_id": sample_id,
                "split": "D_B",
                "role": role,
                "image": str(image_path.relative_to(view)),
                "mask": str(mask_path.relative_to(view)),
                "image_sha256": file_sha256(source_image),
                "mask_sha256": file_sha256(source_mask),
            }
        )
    train_file = view / "trainval.txt"
    select_file = view / "test.txt"
    train_file.write_text("db_fit\n", encoding="utf-8")
    select_file.write_text("db_select\n", encoding="utf-8")
    index_without_fingerprint = {
        "schema_version": "owned-runner-test-view-v1",
        "method_version": CURE_LITE_METHOD_VERSION,
        "split_manifest_fingerprint": manifest.fingerprint,
        "roles": {"train": "D_B-fit", "validation": "D_B-select"},
        "native_split_files_sha256": {
            "trainval.txt": file_sha256(train_file),
            "test.txt": file_sha256(select_file),
        },
        "records": rows,
    }
    index = dict(index_without_fingerprint)
    index["view_fingerprint"] = stable_fingerprint(index_without_fingerprint)
    index_path = view / "index.json"
    _write_json(index_path, index)

    native_output = run / "native_output"
    native_output.mkdir()
    checkpoint = native_output / "weight.pkl"
    checkpoint.write_bytes(b"owned-runner-checkpoint")
    logs = run / "logs"
    logs.mkdir()
    stdout = logs / "stdout.log"
    stderr = logs / "stderr.log"
    stdout.write_text("completed\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")

    recipe = {
        "training_split": "D_B-fit",
        "validation_split": "D_B-select",
        "checkpoint_selection_split": "D_B-select",
        "external_validation_split": None,
        "resume": False,
        "resume_path": None,
        "input_checkpoint": None,
    }
    trainer_contract = {
        "contract": OWNED_NATIVE_TRAINER_CONTRACT,
        "entrypoint": OWNED_NATIVE_TRAINER_ENTRYPOINT,
        "runner_sha256": file_sha256(runner),
        "upstream_sources_sha256": upstream_source_sha256,
    }
    preflight = {
        "schema_version": FORMAL_BASE_PREFLIGHT_SCHEMA,
        "method_version": CURE_LITE_METHOD_VERSION,
        "status": "ready_for_fresh_native_training",
        "split_manifest_fingerprint": manifest.fingerprint,
        "dataset_view_fingerprint": index["view_fingerprint"],
        "dataset_view_index_sha256": file_sha256(index_path),
        "dataset_view_roles": index["roles"],
        "dataset_view_records": rows,
        "checkpoint_selection": selection.to_mapping(),
        "recipe": recipe,
        "recipe_fingerprint": stable_fingerprint(recipe),
        "upstream": {
            "commit": "pinned-commit",
            "tree": "pinned-tree",
            "tracked_python_sources_sha256": upstream_source_sha256,
        },
        "native_trainer_contract": trainer_contract,
        "launcher_sha256": "d" * 64,
        "native_command": [
            "/python",
            str(runner),
            "--mshnet-repo",
            str(mshnet_repo),
            "--dataset-dir",
            str(view),
            "--output-dir",
            str(native_output),
            "--epochs",
            "1",
        ],
        "native_child_dataset_root": str(view),
        "fresh_output_policy": "new_output_no_resume_no_checkpoint_fallback",
    }
    preflight_path = run / "preflight_receipt.json"
    _write_json(preflight_path, preflight)

    provenance = BaseTrainingProvenance.from_manifest(manifest, checkpoint)
    provenance_path = run / "base_training_provenance.json"
    _write_json(provenance_path, provenance.to_mapping())
    final = {
        "schema_version": FORMAL_BASE_FINAL_SCHEMA,
        "method_version": CURE_LITE_METHOD_VERSION,
        "status": "completed",
        "split_manifest_fingerprint": manifest.fingerprint,
        "native_exit_code": 0,
        "preflight_receipt_sha256": file_sha256(preflight_path),
        "dataset_view_fingerprint": index["view_fingerprint"],
        "recipe_fingerprint": stable_fingerprint(recipe),
        "checkpoint_selection_fingerprint": selection.fingerprint,
        "upstream_commit": "pinned-commit",
        "upstream_tree": "pinned-tree",
        "launcher_sha256": "d" * 64,
        "native_trainer_contract": trainer_contract,
        "checkpoint": {
            "path": str(checkpoint.relative_to(run)),
            "sha256": file_sha256(checkpoint),
        },
        "base_training_provenance": {
            "path": str(provenance_path.relative_to(run)),
            "sha256": file_sha256(provenance_path),
            "fingerprint": provenance.fingerprint,
        },
        "logs": {
            "stdout": str(stdout.relative_to(run)),
            "stderr": str(stderr.relative_to(run)),
            "stdout_sha256": file_sha256(stdout),
            "stderr_sha256": file_sha256(stderr),
        },
    }
    final_path = run / "final_receipt.json"
    _write_json(final_path, final)
    return {
        "manifest": manifest,
        "selection": selection,
        "mshnet_repo": mshnet_repo,
        "runner": runner,
        "run": run,
        "preflight": preflight,
        "preflight_path": preflight_path,
        "final": final,
        "final_path": final_path,
        "provenance_path": provenance_path,
        "checkpoint": checkpoint,
        "upstream_source_sha256": upstream_source_sha256,
    }


def _rewrite_preflight(
    record: dict[str, Any], transform: Callable[[dict[str, Any]], None]
) -> None:
    transform(record["preflight"])
    _write_json(record["preflight_path"], record["preflight"])
    record["final"]["preflight_receipt_sha256"] = file_sha256(
        record["preflight_path"]
    )
    _write_json(record["final_path"], record["final"])


def _validate(record: dict[str, Any]):
    return validate_formal_base_training_run(
        record["final_path"],
        record["provenance_path"],
        record["manifest"],
        record["checkpoint"],
    )


def test_owned_runner_contract_binds_runner_sources_and_d_b_roles(tmp_path) -> None:
    record = _build_owned_runner_record(tmp_path)

    identity = _validate(record)

    assert identity.checkpoint_selection_fingerprint == record["selection"].fingerprint
    assert identity.model_source_sha256 == record["upstream_source_sha256"][
        "model/MSHNet.py"
    ]


def test_owned_runner_contract_requires_explicit_mshnet_repository(tmp_path) -> None:
    record = _build_owned_runner_record(tmp_path)

    def remove_repository(command_record: dict[str, Any]) -> None:
        command = command_record["native_command"]
        position = command.index("--mshnet-repo")
        del command[position : position + 2]

    _rewrite_preflight(record, remove_repository)

    with pytest.raises(
        BaseTrainingProvenanceError,
        match="exactly one explicit --mshnet-repo",
    ):
        _validate(record)


@pytest.mark.parametrize("option", ["--resume", "--input-checkpoint", "--weight-path"])
def test_owned_runner_contract_rejects_resume_and_input_weight_options(
    tmp_path, option
) -> None:
    record = _build_owned_runner_record(tmp_path)

    def add_option(command_record: dict[str, Any]) -> None:
        command_record["native_command"].extend([option, "unused.pt"])

    _rewrite_preflight(record, add_option)

    with pytest.raises(
        BaseTrainingProvenanceError,
        match="resume or input-weight option",
    ):
        _validate(record)


@pytest.mark.parametrize(
    ("changed_path", "message"),
    [
        ("runner", "runner SHA256 differs"),
        ("model/MSHNet.py", "model/MSHNet.py SHA256 differs"),
        ("model/loss.py", "model/loss.py SHA256 differs"),
        ("utils/data.py", "utils/data.py SHA256 differs"),
    ],
)
def test_owned_runner_contract_checks_runner_and_imported_sources(
    tmp_path, changed_path, message
) -> None:
    record = _build_owned_runner_record(tmp_path)
    changed_file = (
        record["runner"]
        if changed_path == "runner"
        else record["mshnet_repo"] / changed_path
    )
    changed_file.write_text("# changed after preflight\n", encoding="utf-8")

    with pytest.raises(BaseTrainingProvenanceError, match=message):
        _validate(record)


def test_owned_runner_contract_requires_same_final_contract(tmp_path) -> None:
    record = _build_owned_runner_record(tmp_path)
    record["final"]["native_trainer_contract"] = dict(
        record["final"]["native_trainer_contract"]
    )
    record["final"]["native_trainer_contract"]["runner_sha256"] = "e" * 64
    _write_json(record["final_path"], record["final"])

    with pytest.raises(
        BaseTrainingProvenanceError,
        match="preflight/final CURE-Lite runner contracts differ",
    ):
        _validate(record)

