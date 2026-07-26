from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.decoder import project_occupancy_to_feature_grid
from cure_lite.experiment.paired_bounded_learnability import (
    BoundedMicroPopulation,
    build_bounded_micro_schedule,
)
from cure_lite.experiment.paired_control_bounded_execution import (
    CONTROL_RUNTIME_BINDING_SCHEMA,
    CONTROL_SEMANTICS_SCHEMA,
    build_control_runtime_binding,
    build_control_semantics_receipt,
)
from cure_lite.paired_control_inputs import (
    build_dct_coordinate_basis,
    build_target_permutation,
)
from cure_lite.paired_types import (
    PairCatalog,
    PairExample,
    tensor_content_fingerprint,
)
from cure_lite.train.paired_control_step import CONTROL_KINDS
from tools import run_paired_bounded_learnability as bounded_runner
from tools import run_paired_control_bounded_execution as runner


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "paired_control_bounded_execution_v1"
    / "config.json"
)
_RUN_R1 = (
    _ROOT
    / "runs"
    / "irstd1k_stage_a_seed42"
    / "cure_lite_paired_control_bounded_execution_v1_r1"
)
_RUN_R2 = (
    _ROOT
    / "runs"
    / "irstd1k_stage_a_seed42"
    / "cure_lite_paired_control_bounded_execution_v1_r2"
)


def _config() -> dict[str, object]:
    payload = json.loads(_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _pair(
    sample_id: str,
    target_id: int,
    pixel: tuple[int, int],
) -> PairExample:
    feature = torch.full((1, 2, 2, 2), float(target_id))
    valid = torch.ones((1, 4, 4), dtype=torch.bool)
    plus = torch.zeros_like(valid)
    plus[0, pixel[0], pixel[1]] = True
    minus = torch.zeros_like(plus)
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
        {"sample_id": sample_id, "target_id": target_id}
    )
    return PairExample(
        pair_id=pair_id,
        pair_kind="clean_positive",
        sample_id=sample_id,
        group_id=f"group-{sample_id}",
        feature=feature,
        occupancy_plus=plus,
        occupancy_minus=minus,
        removed_component=plus,
        image_valid_mask=valid,
        completion_plus=empty,
        completion_minus=plus,
        label_increment=plus.to(torch.float32),
        clean_increment=plus,
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


def _catalog() -> PairCatalog:
    pairs = tuple(
        sorted(
            (
                _pair("source-a", 1, (0, 0)),
                _pair("source-b", 2, (1, 1)),
                _pair("source-c", 3, (2, 2)),
            ),
            key=lambda pair: (
                pair.sample_id,
                pair.evaluation_gt_id,
                pair.pair_id,
            ),
        )
    )
    unsealed = PairCatalog(
        dataset="control-bounded-toy",
        split="D_R",
        paired_protocol_fingerprint="1" * 64,
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


def _population(catalog: PairCatalog) -> BoundedMicroPopulation:
    population = object.__new__(BoundedMicroPopulation)
    object.__setattr__(
        population,
        "clean_pairs",
        catalog.clean_positive,
    )
    object.__setattr__(
        population,
        "factual_miss",
        tuple(object() for _ in range(3)),
    )
    object.__setattr__(
        population,
        "factual_no_miss",
        tuple(object() for _ in range(3)),
    )
    object.__setattr__(
        population,
        "pair_catalog_fingerprint",
        catalog.catalog_fingerprint,
    )
    object.__setattr__(
        population,
        "population_fingerprint",
        stable_fingerprint({"population": "toy"}),
    )
    return population


def _schedule(population: BoundedMicroPopulation):
    return build_bounded_micro_schedule(
        population,
        {
            "optimizer_updates": 3,
            "steps_per_epoch": 3,
            "factual_miss_states_per_update": 4,
            "factual_no_miss_states_per_update": 4,
            "clean_pairs_per_update": 2,
        },
    )


def test_frozen_config_and_cli_have_no_scientific_override() -> None:
    config = runner._load_config(_CONFIG.resolve())
    assert config["controls"] == list(CONTROL_KINDS)
    assert config["gates"]["require_positive_response_learning"] is False
    assert config["decision_semantics"]["pass_authorizes_formal_800"] is False
    runner._verify_seed43_formal_recipe(config)

    destinations = {action.dest for action in runner.build_parser()._actions}
    assert destinations == {
        "help",
        "config",
        "control_preflight_complete",
        "paired_bounded_complete",
        "device",
        "output",
    }
    base = [
        "--config",
        str(_CONFIG),
        "--control-preflight-complete",
        "control.json",
        "--paired-bounded-complete",
        "bounded.json",
        "--device",
        "cpu",
        "--output",
        "out",
    ]
    for forbidden in ("--seed", "--budget", "--D_V", "--D_T", "--resume"):
        with pytest.raises(SystemExit):
            runner.parse_args([*base, forbidden, "1"])


def test_upstream_bounded_pass_and_control_preflight_are_exact() -> None:
    config = _config()
    bounded_contract = config["paired_bounded_pass_contract"]
    control_contract = config["control_preflight_contract"]
    bounded_path = _ROOT / bounded_contract["authority_complete_path"]
    control_path = _ROOT / control_contract["authority_complete_path"]

    bounded = runner._verify_paired_bounded_pass(
        bounded_path.resolve(),
        bounded_contract,
    )
    control = bounded_runner._verify_control_preflight(
        control_path.resolve(),
        control_contract,
    )
    assert bounded["byte_identical_replay_verified"] is True
    assert control["byte_identical_replay_verified"] is True
    assert control["target_permutation_status"] == "READY"


def test_runtime_binding_closes_real_control_identities_on_toy() -> None:
    catalog = _catalog()
    population = _population(catalog)
    basis = build_dct_coordinate_basis(
        channels=2,
        height=2,
        width=2,
    )
    plan = build_target_permutation(catalog.clean_positive)
    gt_unions = {
        pair.sample_id: pair.clean_increment.clone()
        for pair in catalog.clean_positive
    }
    binding = build_control_runtime_binding(
        catalog,
        population,
        gt_unions,
        expected_permutation_fingerprint=plan.plan_fingerprint,
        expected_dct_basis_fingerprint=basis.basis_fingerprint,
    )

    assert binding.canonical_payload()["schema_version"] == (
        CONTROL_RUNTIME_BINDING_SCHEMA
    )
    assert binding.permutation_plan.ready
    assert set(binding.permuted_target_by_recipient) == {
        pair.pair_id for pair in catalog.clean_positive
    }
    assert all(
        row["recipient_pair_id"] != row["donor_pair_id"]
        and row["recipient_sample_id"] != row["donor_sample_id"]
        and len(row["donor_target_fingerprint"]) == 64
        and len(row["runtime_target_fingerprint"]) == 64
        for row in binding.assignment_by_recipient.values()
    )

    semantics = build_control_semantics_receipt(
        population,
        _schedule(population),
        binding,
    )
    assert semantics["schema_version"] == CONTROL_SEMANTICS_SCHEMA
    assert semantics["all_control_semantics_pass"] is True
    assert all(semantics["checks"].values())


def test_runtime_binding_rejects_static_identity_tampering() -> None:
    catalog = _catalog()
    population = _population(catalog)
    plan = build_target_permutation(catalog.clean_positive)
    gt_unions = {
        pair.sample_id: pair.clean_increment.clone()
        for pair in catalog.clean_positive
    }
    with pytest.raises(RuntimeError, match="DCT basis"):
        build_control_runtime_binding(
            catalog,
            population,
            gt_unions,
            expected_permutation_fingerprint=plan.plan_fingerprint,
            expected_dct_basis_fingerprint="f" * 64,
        )
    basis = build_dct_coordinate_basis(channels=2, height=2, width=2)
    with pytest.raises(RuntimeError, match="target permutation"):
        build_control_runtime_binding(
            catalog,
            population,
            gt_unions,
            expected_permutation_fingerprint="e" * 64,
            expected_dct_basis_fingerprint=basis.basis_fingerprint,
        )


def test_loader_rejects_incomplete_or_unknown_inventory(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / ".incomplete").touch()
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.load_control_bounded_execution_artifact(incomplete)

    unknown = tmp_path / "unknown"
    unknown.mkdir()
    (unknown / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="inventory"):
        runner.load_control_bounded_execution_artifact(unknown)


def test_real_r1_r2_are_byte_identical_loadable_and_tamper_rejected(
    tmp_path: Path,
) -> None:
    first = runner.load_control_bounded_execution_artifact(_RUN_R1)
    second = runner.load_control_bounded_execution_artifact(_RUN_R2)
    assert first.engineering_execution_pass is True
    assert second.engineering_execution_pass is True
    assert first.decision == second.decision == "ENGINEERING_EXECUTION_PASS"
    assert first.complete_fingerprint == second.complete_fingerprint

    def relative_bytes(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    assert relative_bytes(_RUN_R1) == relative_bytes(_RUN_R2)

    copied = tmp_path / "tampered"
    shutil.copytree(_RUN_R1, copied)
    result_path = copied / "receipts" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["aggregate_budget"]["optimizer_updates"] += 1
    result_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        runner.load_control_bounded_execution_artifact(copied)
