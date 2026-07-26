from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

import cure_lite.experiment.paired_preflight as preflight_module
from cure_lite.cache.schema import stable_fingerprint
from cure_lite.experiment.paired_preflight import (
    build_pair_preflight_manifest,
    build_pair_preflight_receipt,
    load_pair_preflight_artifact,
    write_pair_preflight_artifact,
)
from cure_lite.paired_types import (
    PairCatalog,
    PairCatalogExclusion,
    PairExample,
    tensor_content_fingerprint,
)
from tools.run_paired_preflight import (
    PAIRED_PROTOCOL_FILE_SHA256,
    PAIRED_PROTOCOL_FINGERPRINT,
    PAIRED_PROTOCOL_REPO_PATH,
    _load_paired_protocol,
    build_parser,
    load_paired_run_artifact,
    write_paired_run_artifact,
)

_ROOT = Path(__file__).resolve().parents[1]


def _example(kind: str, sample: str, digit: str) -> PairExample:
    feature = torch.full(
        (1, 1, 2, 2),
        1.0 if sample == "source-a" else 2.0,
        dtype=torch.float32,
    )
    valid = torch.ones((1, 4, 4), dtype=torch.bool)
    plus = torch.zeros_like(valid)
    plus[:, 0, 0] = True
    minus = torch.zeros_like(valid)
    removed = plus.clone()
    empty = torch.zeros_like(valid)
    if kind == "clean_positive":
        increment = torch.zeros_like(valid)
        increment[:, 1, 1] = True
        completion_minus = increment
        return PairExample(
            pair_id=digit * 64,
            pair_kind=kind,
            sample_id=sample,
            group_id=f"group-{sample}",
            feature=feature,
            occupancy_plus=plus,
            occupancy_minus=minus,
            removed_component=removed,
            image_valid_mask=valid,
            completion_plus=empty,
            completion_minus=completion_minus,
            label_increment=increment.float(),
            clean_increment=increment,
            evaluation_gt_id=1,
            native_gt_id=1,
            pred_id=1,
            feature_fingerprint=tensor_content_fingerprint(feature),
            before_match_fingerprint="a" * 64,
            after_match_fingerprint="b" * 64,
            projected_occupancy_plus_fingerprint="c" * 64,
            projected_occupancy_minus_fingerprint="d" * 64,
            projection_visible=True,
            geometry_safe_bijective_lineage=True,
            selected_gt_is_only_new_unmatched=True,
            other_match_identities_unchanged=True,
            preexisting_unmatched_gt_noninterference=True,
        )
    if kind == "component_null":
        return PairExample(
            pair_id=digit * 64,
            pair_kind=kind,
            sample_id=sample,
            group_id=f"group-{sample}",
            feature=feature,
            occupancy_plus=plus,
            occupancy_minus=minus,
            removed_component=removed,
            image_valid_mask=valid,
            completion_plus=empty,
            completion_minus=empty,
            label_increment=empty.float(),
            clean_increment=empty,
            evaluation_gt_id=None,
            native_gt_id=None,
            pred_id=1,
            feature_fingerprint=tensor_content_fingerprint(feature),
            before_match_fingerprint="a" * 64,
            after_match_fingerprint="b" * 64,
            projected_occupancy_plus_fingerprint="c" * 64,
            projected_occupancy_minus_fingerprint="d" * 64,
            projection_visible=True,
            geometry_safe_bijective_lineage=None,
            selected_gt_is_only_new_unmatched=None,
            other_match_identities_unchanged=None,
            preexisting_unmatched_gt_noninterference=None,
        )
    return PairExample(
        pair_id=digit * 64,
        pair_kind=kind,
        sample_id=sample,
        group_id=f"group-{sample}",
        feature=feature,
        occupancy_plus=empty,
        occupancy_minus=empty,
        removed_component=empty,
        image_valid_mask=valid,
        completion_plus=empty,
        completion_minus=empty,
        label_increment=empty.float(),
        clean_increment=empty,
        evaluation_gt_id=None,
        native_gt_id=None,
        pred_id=None,
        feature_fingerprint=tensor_content_fingerprint(feature),
        before_match_fingerprint="a" * 64,
        after_match_fingerprint="a" * 64,
        projected_occupancy_plus_fingerprint="c" * 64,
        projected_occupancy_minus_fingerprint="c" * 64,
        projection_visible=False,
        geometry_safe_bijective_lineage=None,
        selected_gt_is_only_new_unmatched=None,
        other_match_identities_unchanged=None,
        preexisting_unmatched_gt_noninterference=None,
    )


@pytest.fixture()
def pair_catalog() -> PairCatalog:
    unsealed = PairCatalog(
        dataset="paired-toy",
        split="D_R",
        paired_protocol_fingerprint="1" * 64,
        geometry_catalog_fingerprint="2" * 64,
        source_catalog_fingerprint="3" * 64,
        manifest_fingerprint="4" * 64,
        clean_positive=(
            _example("clean_positive", "source-a", "5"),
            _example("clean_positive", "source-b", "6"),
        ),
        component_null=(_example("component_null", "source-a", "7"),),
        identity_null=(
            _example("identity_null", "source-a", "8"),
            _example("identity_null", "source-b", "9"),
        ),
        exclusions=(
            PairCatalogExclusion(
                pair_kind="clean_positive",
                sample_id="source-b",
                group_id="group-source-b",
                evaluation_gt_id=2,
                native_gt_id=2,
                pred_id=2,
                reason_codes=("projection_invisible",),
            ),
        ),
        catalog_fingerprint="",
    )
    return replace(
        unsealed,
        catalog_fingerprint=stable_fingerprint(unsealed.canonical_payload()),
    )


def test_preflight_manifest_and_receipt_are_tensor_free_and_bound(
    pair_catalog: PairCatalog,
) -> None:
    manifest = build_pair_preflight_manifest(pair_catalog)
    receipt = build_pair_preflight_receipt(pair_catalog, manifest)
    json.dumps(manifest, allow_nan=False)
    assert manifest["canonical_pair_catalog"] == pair_catalog.canonical_payload()
    assert manifest["storage_contract"]["raw_tensor_payloads_written"] is False
    assert receipt["input_bindings"][
        "prepared_analysis_population_fingerprint"
    ] == pair_catalog.source_catalog_fingerprint
    assert receipt["counts"] == {
        "clean_positive": 2,
        "component_null": 1,
        "identity_null": 2,
        "included_pairs": 5,
        "trainable_pairs": 2,
        "control_pairs": 3,
        "exclusions": 1,
    }
    assert receipt["exclusion_accounting"]["by_reason"] == {
        "projection_invisible": 1
    }
    assert receipt["source_accounting"]["prepared_source_images"] == 2
    assert receipt["integrity_gates"]["preflight_passed"] is True
    assert receipt["execution_policy"]["training_performed"] is False
    assert receipt["runner_boundary"] == {
        "in_memory_catalog_artifact_writer_implemented": True,
        "verified_real_cache_loader_entrypoint_implemented": True,
        "entrypoint": "tools/run_paired_preflight.py",
        "remaining_gap": None,
    }


def test_real_runner_cli_is_fixed_to_d_r_and_both_frozen_seeds() -> None:
    options = {
        action.dest
        for action in build_parser()._actions
        if action.dest != "help"
    }
    assert options == {
        "manifest",
        "state_index",
        "geometry_config",
        "geometry_catalog_receipt",
        "p0_a1_receipt",
        "eligible_view_receipt",
        "geometry_complete",
        "paired_protocol",
        "output",
    }
    assert not any(
        fragment in option
        for option in options
        for fragment in ("seed", "d_v", "d_t", "device", "train")
    )
    protocol = _load_paired_protocol(_ROOT / PAIRED_PROTOCOL_REPO_PATH)
    assert protocol["receipt_fingerprint"] == PAIRED_PROTOCOL_FINGERPRINT
    assert protocol["future_performance_gate"]["development_seeds"] == [
        42,
        43,
    ]


def _relative_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_two_publications_are_byte_identical_and_loadable(
    tmp_path: Path,
    pair_catalog: PairCatalog,
) -> None:
    first = write_pair_preflight_artifact(pair_catalog, tmp_path / "r1")
    second = write_pair_preflight_artifact(pair_catalog, tmp_path / "r2")
    assert _relative_bytes(first.root) == _relative_bytes(second.root)
    assert set(_relative_bytes(first.root)) == {
        "COMPLETE.json",
        "pair_catalog_manifest.json",
        "preflight_receipt.json",
    }
    assert not (first.root / ".incomplete").exists()
    first.verify_unchanged()
    assert load_pair_preflight_artifact(first.root) == first
    with pytest.raises(FileExistsError):
        write_pair_preflight_artifact(pair_catalog, first.root)


def test_tampering_is_detected(
    tmp_path: Path,
    pair_catalog: PairCatalog,
) -> None:
    published = write_pair_preflight_artifact(pair_catalog, tmp_path / "run")
    path = published.root / "preflight_receipt.json"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_pair_preflight_artifact(published.root)


def test_failed_publication_keeps_incomplete_marker(
    tmp_path: Path,
    pair_catalog: PairCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = preflight_module._write_new_json

    def fail_on_receipt(path: Path, payload: object) -> None:
        if path.name == "preflight_receipt.json":
            raise RuntimeError("injected publication failure")
        original(path, payload)

    monkeypatch.setattr(preflight_module, "_write_new_json", fail_on_receipt)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected"):
        write_pair_preflight_artifact(pair_catalog, output)
    assert (output / ".incomplete").is_file()
    assert not (output / "COMPLETE.json").exists()
    with pytest.raises(RuntimeError, match="incomplete"):
        load_pair_preflight_artifact(output)


def _real_runner_catalog(catalog: PairCatalog) -> PairCatalog:
    unsealed = replace(
        catalog,
        paired_protocol_fingerprint=PAIRED_PROTOCOL_FINGERPRINT,
        catalog_fingerprint="",
    )
    return replace(
        unsealed,
        catalog_fingerprint=stable_fingerprint(unsealed.canonical_payload()),
    )


def _real_runner_bindings(catalog: PairCatalog) -> dict[str, object]:
    return {
        "dataset": catalog.dataset,
        "split": "D_R",
        "runtime_splits": ["D_R"],
        "paired_protocol_fingerprint": PAIRED_PROTOCOL_FINGERPRINT,
        "paired_protocol_file_sha256": PAIRED_PROTOCOL_FILE_SHA256,
        "paired_protocol_repo_path": PAIRED_PROTOCOL_REPO_PATH,
        "geometry_catalog_fingerprint": (
            catalog.geometry_catalog_fingerprint
        ),
        "geometry_catalog_repo_path": "authority/geometry_catalog.json",
        "geometry_catalog_frozen_repo_path": (
            "authority/geometry_catalog.json"
        ),
        "geometry_catalog_path_matches_frozen_binding": True,
        "p0_a1_repo_path": "authority/p0_a1.json",
        "p0_a1_frozen_repo_path": "authority/p0_a1.json",
        "p0_a1_path_matches_frozen_binding": True,
        "eligible_view_repo_path": "authority/eligible_view.json",
        "eligible_view_frozen_repo_path": "authority/eligible_view.json",
        "eligible_view_path_matches_frozen_binding": True,
        "source_catalog_fingerprint": catalog.source_catalog_fingerprint,
        "manifest_fingerprint": catalog.manifest_fingerprint,
        "state_index_fingerprint": "a" * 64,
        "state_index_file_sha256": "b" * 64,
    }


def test_real_runner_publication_replays_and_refuses_overwrite(
    tmp_path: Path,
    pair_catalog: PairCatalog,
) -> None:
    catalog = _real_runner_catalog(pair_catalog)
    bindings = _real_runner_bindings(catalog)
    implementation = {"tools/run_paired_preflight.py": "c" * 64}
    first = write_paired_run_artifact(
        catalog,
        tmp_path / "r1",
        input_bindings=bindings,
        implementation_files=implementation,
    )
    second = write_paired_run_artifact(
        catalog,
        tmp_path / "r2",
        input_bindings=bindings,
        implementation_files=implementation,
    )

    assert _relative_bytes(first.root) == _relative_bytes(second.root)
    assert not (first.root / ".incomplete").exists()
    assert {
        path.relative_to(first.root).as_posix()
        for path in first.root.rglob("*")
        if path.is_file()
    } == {
        "COMPLETE.json",
        "pair_preflight/COMPLETE.json",
        "pair_preflight/pair_catalog_manifest.json",
        "pair_preflight/preflight_receipt.json",
        "receipts/exposure_seed42.json",
        "receipts/exposure_seed43.json",
        "receipts/run_receipt.json",
    }
    assert first == load_paired_run_artifact(first.root)
    first.verify_unchanged()
    with pytest.raises(FileExistsError, match="already exists"):
        write_paired_run_artifact(
            catalog,
            first.root,
            input_bindings=bindings,
            implementation_files=implementation,
        )


def test_real_runner_rejects_input_binding_mismatch_before_output(
    tmp_path: Path,
    pair_catalog: PairCatalog,
) -> None:
    catalog = _real_runner_catalog(pair_catalog)
    bindings = _real_runner_bindings(catalog)
    bindings["geometry_catalog_fingerprint"] = "d" * 64
    output = tmp_path / "mismatch"

    with pytest.raises(RuntimeError, match="geometry_catalog_fingerprint"):
        write_paired_run_artifact(
            catalog,
            output,
            input_bindings=bindings,
            implementation_files={
                "tools/run_paired_preflight.py": "c" * 64
            },
        )
    assert not output.exists()


def test_real_runner_rejects_same_content_at_nonfrozen_authority_path(
    tmp_path: Path,
    pair_catalog: PairCatalog,
) -> None:
    catalog = _real_runner_catalog(pair_catalog)
    bindings = _real_runner_bindings(catalog)
    bindings["geometry_catalog_repo_path"] = (
        "runs/another-byte-identical-copy/geometry_catalog.json"
    )
    output = tmp_path / "wrong-path"

    with pytest.raises(RuntimeError, match="input path binding mismatch"):
        write_paired_run_artifact(
            catalog,
            output,
            input_bindings=bindings,
            implementation_files={
                "tools/run_paired_preflight.py": "c" * 64
            },
        )
    assert not output.exists()
