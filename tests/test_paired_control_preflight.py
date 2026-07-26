from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.decoder import project_occupancy_to_feature_grid
from cure_lite.experiment.paired_control_preflight import (
    CONTROL_PREFLIGHT_PROTOCOL_FINGERPRINT,
    REQUIRED_CONTROL_SOURCE_PATHS,
    build_control_contracts_receipt,
    build_dct_basis_receipt,
    build_target_permutation_receipt,
    load_control_preflight_artifact,
    write_control_preflight_artifact,
)
from cure_lite.paired_control_inputs import (
    TARGET_PERMUTATION_INCONCLUSIVE,
    TARGET_PERMUTATION_READY,
)
from cure_lite.paired_types import (
    PairCatalog,
    PairExample,
    tensor_content_fingerprint,
)
from tools.run_paired_control_preflight import build_parser


_ROOT = Path(__file__).resolve().parents[1]


def _clean_pair(
    sample_id: str,
    target_id: int,
    *,
    target_pixel: tuple[int, int],
) -> PairExample:
    feature = torch.full(
        (1, 2, 2, 2),
        float(target_id),
        dtype=torch.float32,
    )
    valid = torch.ones((1, 4, 4), dtype=torch.bool)
    plus = torch.zeros_like(valid)
    plus[0, target_pixel[0], target_pixel[1]] = True
    minus = torch.zeros_like(plus)
    clean = plus.clone()
    empty = torch.zeros_like(valid)
    projected_plus = project_occupancy_to_feature_grid(
        plus.unsqueeze(0),
        (2, 2),
    )
    projected_minus = project_occupancy_to_feature_grid(
        minus.unsqueeze(0),
        (2, 2),
    )
    pair_id = stable_fingerprint(
        {
            "sample_id": sample_id,
            "target_id": target_id,
            "target_pixel": target_pixel,
        }
    )
    return PairExample(
        pair_id=pair_id,
        pair_kind="clean_positive",
        sample_id=sample_id,
        group_id=f"group-{sample_id}",
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
        removed_component=plus.clone(),
        image_valid_mask=valid,
        completion_plus=empty,
        completion_minus=clean,
        label_increment=clean.to(torch.float32),
        clean_increment=clean,
        evaluation_gt_id=target_id,
        native_gt_id=target_id,
        pred_id=target_id,
        feature_fingerprint=tensor_content_fingerprint(feature),
        before_match_fingerprint=stable_fingerprint(
            {"pair_id": pair_id, "endpoint": "plus"}
        ),
        after_match_fingerprint=stable_fingerprint(
            {"pair_id": pair_id, "endpoint": "minus"}
        ),
        projected_occupancy_plus_fingerprint=tensor_content_fingerprint(
            projected_plus
        ),
        projected_occupancy_minus_fingerprint=tensor_content_fingerprint(
            projected_minus
        ),
        projection_visible=True,
        geometry_safe_bijective_lineage=True,
        selected_gt_is_only_new_unmatched=True,
        other_match_identities_unchanged=True,
        preexisting_unmatched_gt_noninterference=True,
    )


def _catalog(*, all_same_source: bool = False) -> PairCatalog:
    pairs = tuple(
        sorted(
            (
                _clean_pair(
                    "source-a",
                    1,
                    target_pixel=(0, 0),
                ),
                _clean_pair(
                    "source-a" if all_same_source else "source-b",
                    2,
                    target_pixel=(1, 1),
                ),
                _clean_pair(
                    "source-a" if all_same_source else "source-c",
                    3,
                    target_pixel=(2, 2),
                ),
            ),
            key=lambda pair: (
                pair.sample_id,
                pair.evaluation_gt_id,
                pair.pred_id,
                pair.pair_id,
            ),
        )
    )
    unsealed = PairCatalog(
        dataset="paired-control-toy",
        split="D_R",
        paired_protocol_fingerprint=CONTROL_PREFLIGHT_PROTOCOL_FINGERPRINT,
        geometry_catalog_fingerprint="2" * 64,
        source_catalog_fingerprint="3" * 64,
        manifest_fingerprint="4" * 64,
        clean_positive=pairs,
        component_null=(),
        identity_null=(),
        exclusions=(),
        catalog_fingerprint="",
    )
    return replace(
        unsealed,
        catalog_fingerprint=stable_fingerprint(unsealed.canonical_payload()),
    )


def _bindings(catalog: PairCatalog) -> dict[str, object]:
    return {
        "dataset": catalog.dataset,
        "split": "D_R",
        "runtime_splits": ["D_R"],
        "paired_protocol_fingerprint": catalog.paired_protocol_fingerprint,
        "pair_catalog_fingerprint": catalog.catalog_fingerprint,
        "upstream_paired_preflight_complete_fingerprint": "a" * 64,
        "upstream_paired_preflight_complete_sha256": "b" * 64,
    }


def _source_hashes() -> dict[str, str]:
    return {
        path: stable_fingerprint({"path": path})
        for path in REQUIRED_CONTROL_SOURCE_PATHS
    }


def _publish(
    catalog: PairCatalog,
    output: Path,
):
    return write_control_preflight_artifact(
        catalog,
        output,
        input_bindings=_bindings(catalog),
        control_source_hashes=_source_hashes(),
        expected_catalog_fingerprint=catalog.catalog_fingerprint,
        expected_protocol_fingerprint=catalog.paired_protocol_fingerprint,
        expected_clean_pair_count=len(catalog.clean_positive),
    )


def _load(catalog: PairCatalog, output: Path):
    return load_control_preflight_artifact(
        output,
        expected_catalog_fingerprint=catalog.catalog_fingerprint,
        expected_protocol_fingerprint=catalog.paired_protocol_fingerprint,
        expected_clean_pair_count=len(catalog.clean_positive),
    )


def _relative_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_real_contract_receipts_are_derived_without_raw_tensors() -> None:
    catalog = _catalog()
    contracts = build_control_contracts_receipt(catalog)
    dct = build_dct_basis_receipt(catalog)
    permutation = build_target_permutation_receipt(catalog)

    assert contracts["real_feature_signature"] == {
        "shape": [1, 2, 2, 2],
        "dtype": "torch.float32",
        "device": "cpu",
        "uniform_over_clean_population": True,
    }
    assert contracts["nominal_zero_feature"]["all_elements_zero"] is True
    assert contracts["nominal_zero_feature"]["source_values_read"] is False
    assert contracts["feature_only_zero_occupancy"][
        "shared_tensor_object"
    ] is True
    assert dct["real_feature_shape"] == [1, 2, 2, 2]
    assert dct["basis"]["modes"] == [[0, 1], [1, 0]]
    assert dct["basis"]["shape"] == [1, 2, 2, 2]
    assert dct["basis"]["tensor_fingerprint"]
    assert dct["basis_fingerprint"]
    assert permutation["status"] == TARGET_PERMUTATION_READY
    assert permutation["assignment_count"] == 3
    assert permutation["source_disjoint"] is True
    assert permutation["fixed_point_free"] is True
    assert permutation["full_donor_marginal"] is True
    assert all(
        row["assignment_count"] == 1
        for row in permutation["donor_marginal"]
    )
    materialization = permutation["training_materialization_contract"]
    assert materialization[
        "recipient_batch_pair_ids_must_bind_exact_assignments"
    ] is True
    assert materialization[
        "donor_target_fingerprint_must_match_assignment"
    ] is True
    assert materialization[
        "runtime_training_binding_implemented_by_preflight"
    ] is False
    assert permutation["training_performed"] is False
    json.dumps((contracts, dct, permutation), allow_nan=False)


def test_publications_are_create_only_byte_identical_and_fully_loadable(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    first = _publish(catalog, tmp_path / "r1")
    second = _publish(catalog, tmp_path / "r2")

    assert _relative_bytes(first.root) == _relative_bytes(second.root)
    assert set(_relative_bytes(first.root)) == {
        "COMPLETE.json",
        "receipts/control_contracts.json",
        "receipts/dct_basis.json",
        "receipts/run_receipt.json",
        "receipts/target_permutation.json",
    }
    assert not (first.root / ".incomplete").exists()
    assert _load(catalog, first.root) == first
    complete = json.loads(
        (first.root / "COMPLETE.json").read_text(encoding="utf-8")
    )
    assert complete["status"] == "complete"
    assert complete["target_permutation_status"] == TARGET_PERMUTATION_READY
    assert complete["matched_controls_static_preflight_pass"] is True
    first.verify_unchanged(
        expected_catalog_fingerprint=catalog.catalog_fingerprint,
        expected_protocol_fingerprint=catalog.paired_protocol_fingerprint,
        expected_clean_pair_count=3,
    )
    with pytest.raises(FileExistsError):
        _publish(catalog, first.root)


def test_tamper_or_incomplete_publication_is_rejected(tmp_path: Path) -> None:
    catalog = _catalog()
    published = _publish(catalog, tmp_path / "published")
    permutation_path = (
        published.root / "receipts" / "target_permutation.json"
    )
    payload = json.loads(permutation_path.read_text(encoding="utf-8"))
    payload["compatible_edges"] += 1
    permutation_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        _load(catalog, published.root)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / ".incomplete").touch()
    with pytest.raises(RuntimeError, match="incomplete"):
        _load(catalog, incomplete)


def test_no_source_disjoint_perfect_matching_is_explicitly_inconclusive(
    tmp_path: Path,
) -> None:
    catalog = _catalog(all_same_source=True)
    permutation = build_target_permutation_receipt(catalog)
    assert permutation["status"] == TARGET_PERMUTATION_INCONCLUSIVE
    assert permutation["reason_code"] == "no_compatible_perfect_matching"
    assert permutation["assignments"] == []
    assert permutation["full_donor_marginal"] is False

    published = _publish(catalog, tmp_path / "inconclusive")
    assert published.status == TARGET_PERMUTATION_INCONCLUSIVE
    receipt = json.loads(
        (published.root / "receipts" / "run_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["gates"]["matched_controls_ready"] is False
    assert (
        receipt["next_route"]
        == "report_target_permutation_computationally_inconclusive"
    )
    assert receipt["execution_policy"]["training_performed"] is False


def test_source_binding_includes_the_production_control_step() -> None:
    assert "cure_lite/train/paired_control_step.py" in (
        REQUIRED_CONTROL_SOURCE_PATHS
    )
    assert {
        "cure_lite/paired_control_inputs.py",
        "cure_lite/paired_control_losses.py",
        "cure_lite/train/paired_control_step.py",
        "cure_lite/decoder.py",
    }.issubset(REQUIRED_CONTROL_SOURCE_PATHS)


def test_real_runner_cli_has_no_training_or_non_d_r_controls() -> None:
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
        "paired_preflight_complete",
        "output",
    }
    assert not any(
        fragment in option
        for option in options
        for fragment in ("seed", "d_v", "d_t", "device", "train", "model")
    )


def test_bounded_config_gate_matches_control_complete_contract() -> None:
    config = json.loads(
        (
            _ROOT
            / "protocols"
            / "IRSTD-1K"
            / "paired_bounded_learnability_v1"
            / "config.json"
        ).read_text(encoding="utf-8")
    )
    contract = config["control_preflight_contract"]
    assert contract["schema_version"] == (
        "cure-lite-paired-control-preflight-complete-v1"
    )
    assert contract["required_execution_status"] == "complete"
    assert contract["required_status"] == "complete"
    assert (
        contract["required_target_permutation_status"] == "READY"
    )
    assert (
        contract["required_complete_gate"]
        == "matched_controls_static_preflight_pass"
    )
