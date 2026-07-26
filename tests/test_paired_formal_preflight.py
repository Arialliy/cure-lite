from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.decoder import project_occupancy_to_feature_grid
from cure_lite.experiment.paired_formal_preflight import (
    FORMAL_PREFLIGHT_SEEDS,
    PAIRED_FORMAL_PREFLIGHT_CONFIG_SCHEMA,
    build_formal_method_bindings,
    build_formal_schedule_receipt,
    load_paired_formal_preflight_artifact,
    validate_paired_formal_preflight_config,
    write_paired_formal_preflight_artifact,
)
from cure_lite.experiment.paired_formal_schedule import (
    FORMAL_METHOD_KINDS,
    build_paired_formal_schedule_from_epoch_pool_builder,
)
from cure_lite.paired_types import (
    PairCatalog,
    PairExample,
    tensor_content_fingerprint,
)
from cure_lite.train.paired_pools import build_paired_schedule
from cure_lite.train.pools import BranchPools, StateExample
from cure_lite.types import BranchSupervision


def _clean_pair(*, sample_id: str, target_id: int) -> PairExample:
    feature = torch.full(
        (1, 2, 2, 2),
        float(target_id) / 10.0,
        dtype=torch.float32,
    )
    valid = torch.ones((1, 4, 4), dtype=torch.bool)
    plus = torch.zeros_like(valid)
    row = target_id % 4
    column = (target_id * 3) % 4
    plus[0, row, column] = True
    minus = torch.zeros_like(plus)
    clean = plus.clone()
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
            "kind": "formal-preflight-clean-positive",
            "sample_id": sample_id,
            "target_id": target_id,
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
        completion_plus=torch.zeros_like(valid),
        completion_minus=clean.clone(),
        label_increment=clean.to(torch.float32),
        clean_increment=clean,
        evaluation_gt_id=target_id,
        native_gt_id=target_id,
        pred_id=target_id,
        feature_fingerprint=tensor_content_fingerprint(feature),
        before_match_fingerprint=stable_fingerprint(
            {"sample_id": sample_id, "state": "before"}
        ),
        after_match_fingerprint=stable_fingerprint(
            {"sample_id": sample_id, "state": "after"}
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


def _catalog() -> PairCatalog:
    pairs = tuple(
        _clean_pair(sample_id=f"source-{index}", target_id=index + 1)
        for index in range(4)
    )
    return PairCatalog(
        dataset="IRSTD-1K",
        split="D_R",
        paired_protocol_fingerprint="1" * 64,
        geometry_catalog_fingerprint="2" * 64,
        source_catalog_fingerprint="3" * 64,
        manifest_fingerprint="4" * 64,
        clean_positive=pairs,
        component_null=(),
        identity_null=(),
        exclusions=(),
        catalog_fingerprint="5" * 64,
    )


def _factual(
    *,
    branch: str,
    sample_id: str,
    target_id: int | None,
) -> StateExample:
    feature = torch.full(
        (1, 2, 2, 2),
        float(target_id or 1),
        dtype=torch.float32,
    )
    occupancy = torch.zeros((1, 4, 4), dtype=torch.bool)
    target = torch.zeros((1, 4, 4), dtype=torch.float32)
    if target_id is not None:
        target[0, target_id % 4, (target_id * 3) % 4] = 1.0
    return StateExample(
        sample_id,
        feature,
        BranchSupervision(
            occupancy=occupancy,
            target=target,
            valid_mask=torch.ones_like(occupancy),
            branch=branch,
            positive_gt_ids=(() if target_id is None else (target_id,)),
            reachable_gt_ids=(() if target_id is None else (target_id,)),
        ),
    )


@pytest.fixture(scope="module")
def formal_schedules():
    miss = tuple(
        _factual(
            branch="factual_miss",
            sample_id=f"miss-{index}",
            target_id=index + 1,
        )
        for index in range(5)
    )
    no_miss = tuple(
        _factual(
            branch="factual_no_miss",
            sample_id=f"no-miss-{index}",
            target_id=None,
        )
        for index in range(4)
    )
    schedules = {}
    for seed in FORMAL_PREFLIGHT_SEEDS:
        paired = build_paired_schedule(_catalog(), seed=seed)
        schedules[seed] = (
            build_paired_formal_schedule_from_epoch_pool_builder(
                paired,
                prepared_catalog_fingerprint="a" * 64,
                expected_factual_miss=miss,
                expected_factual_no_miss=no_miss,
                epoch_pool_builder=lambda _epoch: BranchPools(
                    factual_miss=miss,
                    factual_no_miss=no_miss,
                ),
            )
        )
    return schedules


def _config(pair_catalog_fingerprint: str) -> dict[str, object]:
    core: dict[str, object] = {
        "schema_version": PAIRED_FORMAL_PREFLIGHT_CONFIG_SCHEMA,
        "protocol_id": "test-paired-formal-preflight-v1",
        "dataset": "IRSTD-1K",
        "split": "D_R",
        "seeds": [42, 43],
        "methods": list(FORMAL_METHOD_KINDS),
        "input_binding": {
            "real_pair_catalog_fingerprint": pair_catalog_fingerprint,
        },
        "implementation_binding": {"fake.py": "b" * 64},
        "budget": {
            "epochs": 800,
            "steps_per_epoch": 40,
            "optimizer_updates_per_seed": 32_000,
            "factual_miss_states_per_update": 4,
            "factual_no_miss_states_per_update": 4,
            "clean_pairs_per_update": 2,
            "paired_endpoint_states_per_update": 4,
            "decoder_states_per_update": 12,
            "decoder_forwards_per_update": 3,
        },
        "execution_policy": {
            "create_only_output": True,
            "allowed_runtime_splits": ["D_R"],
            "schedule_only": True,
            "training_performed": False,
            "allow_D_V": False,
            "allow_D_T": False,
            "allow_calibration": False,
            "allow_inference": False,
            "allow_scientific_overrides": False,
            "resume": False,
            "overwrite": False,
        },
    }
    return {**core, "config_fingerprint": stable_fingerprint(core)}


def test_schedule_receipts_seal_sequences_ledgers_and_budget(
    formal_schedules,
) -> None:
    receipt = build_formal_schedule_receipt(formal_schedules[42])
    assert receipt["split"] == "D_R"
    assert receipt["budget"]["optimizer_updates"] == 32_000
    assert receipt["budget"]["decoder_state_evaluations"] == 384_000
    assert receipt["budget"]["decoder_forward_calls"] == 96_000
    assert set(receipt["sequence_fingerprints"]) == {
        "pair",
        "factual_miss",
        "factual_no_miss",
        "combined",
    }
    ledgers = receipt["exposure_ledgers"]
    assert ledgers["pair"]["total"] == 64_000
    assert ledgers["factual_miss"]["total"] == 128_000
    assert ledgers["factual_no_miss"]["total"] == 128_000
    assert ledgers["source"]["total"] == 320_000
    assert all(value is True for value in receipt["gates"].values())


def test_all_nine_method_labels_share_schedule_per_seed(
    formal_schedules,
) -> None:
    bindings = build_formal_method_bindings(formal_schedules)
    assert FORMAL_METHOD_KINDS[0] == "paired_difference"
    assert len(FORMAL_METHOD_KINDS) == 9
    assert bindings["method_inventory"] == list(FORMAL_METHOD_KINDS)
    for seed_row in bindings["seeds"]:
        assert {
            row["shared_formal_schedule_fingerprint"]
            for row in seed_row["methods"]
        } == {seed_row["formal_schedule_fingerprint"]}
        assert all(
            row["method_label_affects_schedule"] is False
            for row in seed_row["methods"]
        )


def test_create_load_exact_replay_and_tamper_rejection(
    formal_schedules,
    tmp_path: Path,
) -> None:
    config = _config(
        formal_schedules[42].paired_schedule.catalog_fingerprint
    )
    kwargs = {
        "config": config,
        "config_file_sha256": "c" * 64,
        "input_file_sha256": {"input.json": "d" * 64},
        "implementation_file_sha256": {"fake.py": "b" * 64},
    }
    first_root = tmp_path / "r1"
    second_root = tmp_path / "r2"
    first = write_paired_formal_preflight_artifact(
        formal_schedules,
        output_dir=first_root,
        **kwargs,
    )
    second = write_paired_formal_preflight_artifact(
        formal_schedules,
        output_dir=second_root,
        **kwargs,
    )
    assert first.complete_fingerprint == second.complete_fingerprint
    names = sorted(path.name for path in first_root.iterdir())
    assert names == sorted(path.name for path in second_root.iterdir())
    for name in names:
        assert (first_root / name).read_bytes() == (
            second_root / name
        ).read_bytes()
    assert load_paired_formal_preflight_artifact(first_root) == first
    with pytest.raises(FileExistsError, match="overwrite"):
        write_paired_formal_preflight_artifact(
            formal_schedules,
            output_dir=first_root,
            **kwargs,
        )

    schedule_path = second_root / "seed42_schedule.json"
    payload = json.loads(schedule_path.read_text(encoding="utf-8"))
    payload["exposure_ledgers"]["pair"]["identities"][0]["count"] += 1
    schedule_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        load_paired_formal_preflight_artifact(second_root)


def test_config_rejects_method_and_scientific_policy_changes(
    formal_schedules,
) -> None:
    original = _config(
        formal_schedules[42].paired_schedule.catalog_fingerprint
    )
    changed = dict(original)
    changed["methods"] = ["proposed", *list(FORMAL_METHOD_KINDS[1:])]
    changed.pop("config_fingerprint")
    changed["config_fingerprint"] = stable_fingerprint(changed)
    with pytest.raises(ValueError, match="method inventory"):
        validate_paired_formal_preflight_config(changed)

    changed = json.loads(json.dumps(original))
    changed["execution_policy"]["allow_D_V"] = True
    changed.pop("config_fingerprint")
    changed["config_fingerprint"] = stable_fingerprint(changed)
    with pytest.raises(ValueError, match="execution policy"):
        validate_paired_formal_preflight_config(changed)


def test_loader_rejects_self_consistent_receipt_policy_flip(
    formal_schedules,
    tmp_path: Path,
) -> None:
    root = tmp_path / "policy-tamper"
    config = _config(
        formal_schedules[42].paired_schedule.catalog_fingerprint
    )
    write_paired_formal_preflight_artifact(
        formal_schedules,
        output_dir=root,
        config=config,
        config_file_sha256="c" * 64,
        input_file_sha256={"input.json": "d" * 64},
        implementation_file_sha256={"fake.py": "b" * 64},
    )

    receipt_path = root / "preflight_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["execution_policy"][
        "formal_training_authorized_by_this_artifact"
    ] = True
    receipt_core = dict(receipt)
    receipt_core.pop("receipt_fingerprint")
    receipt["receipt_fingerprint"] = stable_fingerprint(receipt_core)
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    complete_path = root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["artifact_files"]["preflight_receipt.json"] = file_sha256(
        receipt_path
    )
    complete["receipt_fingerprint"] = receipt["receipt_fingerprint"]
    complete_core = dict(complete)
    complete_core.pop("complete_fingerprint")
    complete["complete_fingerprint"] = stable_fingerprint(complete_core)
    complete_path.write_text(
        json.dumps(complete, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="gate or policy"):
        load_paired_formal_preflight_artifact(root)
