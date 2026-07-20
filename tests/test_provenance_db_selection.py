from __future__ import annotations

import json
from pathlib import Path

import pytest

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.provenance import (
    CURE_LITE_METHOD_VERSION,
    FORMAL_BASE_FINAL_SCHEMA,
    FORMAL_BASE_PREFLIGHT_SCHEMA,
    LEGACY_FORMAL_BASE_FINAL_SCHEMA,
    LEGACY_FORMAL_BASE_PREFLIGHT_SCHEMA,
    BaseCheckpointSelection,
    BaseTrainingProvenance,
    BaseTrainingProvenanceError,
    validate_formal_base_training_run,
)
from cure_lite.splits import SplitManifest, SplitRecord


def _manifest(*, shared_scene: bool = False) -> SplitManifest:
    return SplitManifest(
        dataset="toy",
        records=(
            SplitRecord(
                "fit-1",
                "D_B",
                "fit-group",
                "fit-1.png",
                scene_id="shared-scene" if shared_scene else "fit-scene",
            ),
            SplitRecord(
                "fit-2",
                "D_B",
                "fit-group",
                "fit-2.png",
                scene_id="shared-scene" if shared_scene else "fit-scene",
            ),
            SplitRecord(
                "select-1",
                "D_B",
                "select-group",
                "select-1.png",
                scene_id="shared-scene" if shared_scene else "select-scene",
            ),
            SplitRecord("dr", "D_R", "dr-group", "dr.png"),
            SplitRecord("dv", "D_V", "dv-group", "dv.png"),
            SplitRecord("dt", "D_T", "dt-group", "dt.png"),
        ),
    )


def test_checkpoint_selection_round_trip_binds_exact_d_b_partition() -> None:
    manifest = _manifest()
    selection = BaseCheckpointSelection.from_manifest(
        manifest,
        fit_sample_ids=["fit-2", "fit-1"],
        select_sample_ids=["select-1"],
    )

    assert selection.fit_sample_ids == ("fit-1", "fit-2")
    assert selection.select_sample_ids == ("select-1",)
    assert BaseCheckpointSelection.from_mapping(selection.to_mapping()) == selection
    selection.validate_against(manifest)


def test_checkpoint_selection_rejects_cross_partition_grouping_keys() -> None:
    with pytest.raises(BaseTrainingProvenanceError, match="crosses D_B-fit/D_B-select"):
        BaseCheckpointSelection.from_manifest(
            _manifest(shared_scene=True),
            fit_sample_ids=["fit-1", "fit-2"],
            select_sample_ids=["select-1"],
        )


def test_checkpoint_selection_requires_an_exhaustive_d_b_partition() -> None:
    with pytest.raises(BaseTrainingProvenanceError, match="exact manifest D_B partition"):
        BaseCheckpointSelection.from_manifest(
            _manifest(),
            fit_sample_ids=["fit-1"],
            select_sample_ids=["select-1"],
        )


def test_formal_validator_rejects_legacy_d_v_selection_receipt(tmp_path) -> None:
    final = tmp_path / "final_receipt.json"
    final.write_text(
        json.dumps({"schema_version": LEGACY_FORMAL_BASE_FINAL_SCHEMA}),
        encoding="utf-8",
    )

    with pytest.raises(BaseTrainingProvenanceError, match="used D_V"):
        validate_formal_base_training_run(
            final,
            tmp_path / "missing-provenance.json",
            _manifest(),
            tmp_path / "missing-checkpoint.pt",
        )


def test_formal_validator_rejects_legacy_preflight_under_v2_final(tmp_path) -> None:
    manifest = _manifest()
    preflight = tmp_path / "preflight_receipt.json"
    preflight.write_text(
        json.dumps({"schema_version": LEGACY_FORMAL_BASE_PREFLIGHT_SCHEMA}),
        encoding="utf-8",
    )
    final = tmp_path / "final_receipt.json"
    final.write_text(
        json.dumps(
            {
                "schema_version": FORMAL_BASE_FINAL_SCHEMA,
                "method_version": CURE_LITE_METHOD_VERSION,
                "status": "completed",
                "split_manifest_fingerprint": manifest.fingerprint,
                "native_exit_code": 0,
                "preflight_receipt_sha256": file_sha256(preflight),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BaseTrainingProvenanceError, match="assigned D_V"):
        validate_formal_base_training_run(
            final,
            tmp_path / "missing-provenance.json",
            manifest,
            tmp_path / "missing-checkpoint.pt",
        )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_formal_v2_receipt_binds_selection_and_native_split_roles(tmp_path) -> None:
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
        dataset="formal-toy",
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

    run = tmp_path / "run"
    view = run / "dataset_view"
    images = view / "images"
    masks = view / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    rows = []
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
    index = {
        "schema_version": "test-view-v1",
        "method_version": CURE_LITE_METHOD_VERSION,
        "split_manifest_fingerprint": manifest.fingerprint,
        "roles": {"train": "D_B-fit", "validation": "D_B-select"},
        "native_split_files_sha256": {
            "trainval.txt": file_sha256(train_file),
            "test.txt": file_sha256(select_file),
        },
        "records": rows,
    }
    index["view_fingerprint"] = stable_fingerprint(index)
    index_path = view / "index.json"
    _write_json(index_path, index)

    native_output = run / "native_runs"
    checkpoint_dir = native_output / "seed"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "weight.pkl"
    checkpoint.write_bytes(b"fresh-checkpoint")
    logs = run / "logs"
    logs.mkdir()
    stdout = logs / "stdout.log"
    stderr = logs / "stderr.log"
    stdout.write_text("ok", encoding="utf-8")
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
            "commit": "commit",
            "tree": "tree",
            "tracked_python_sources_sha256": {"model/MSHNet.py": "a" * 64},
        },
        "native_trainer_contract": {
            "contract": "external-wrapper-native-train-cli-v1",
            "entrypoint": "run_pinned_mshnet_train.py",
            "native_module": "main.py",
            "fresh_output_option": "--save-dir",
            "resume_disabled_argv": ["--if-checkpoint", "false"],
            "boolean_parser": "str2bool",
            "wrapper_sha256": "b" * 64,
            "main_source_sha256": "c" * 64,
        },
        "launcher_sha256": "d" * 64,
        "native_command": [
            "/python",
            "/launcher/run_pinned_mshnet_train.py",
            "--dataset-dir",
            str(view),
            "--save-dir",
            str(native_output),
            "--mode",
            "train",
            "--if-checkpoint",
            "false",
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
        "upstream_commit": "commit",
        "upstream_tree": "tree",
        "launcher_sha256": "d" * 64,
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

    identity = validate_formal_base_training_run(
        final_path,
        provenance_path,
        manifest,
        checkpoint,
    )
    assert identity.checkpoint_selection_fingerprint == selection.fingerprint

    (images / "db_fit.png").write_bytes(b"tampered-view-asset")
    with pytest.raises(
        BaseTrainingProvenanceError,
        match="isolated dataset-view asset differs",
    ):
        validate_formal_base_training_run(
            final_path,
            provenance_path,
            manifest,
            checkpoint,
        )
