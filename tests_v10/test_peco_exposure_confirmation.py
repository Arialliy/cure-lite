from __future__ import annotations

from collections import Counter, defaultdict
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import tools.evaluate_peco_exposure_confirmation as confirmation_evaluator

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.config import LossConfig
from cure_lite.conservative_factorized_decoder import (
    CURELiteConservativeFactorizedDecoder,
)
from cure_lite.experiment.peco_exposure_confirmation import (
    CLEAN_PAIR_COUNT,
    CLEAN_SLOT_COUNT,
    COMPONENT_PAIR_COUNT,
    COMPONENT_SLOT_COUNT,
    CONTAINS_FAMILY,
    FACTUAL_BATCH_SIZE,
    FACTUAL_EXPOSURES_PER_STATE,
    FACTUAL_POPULATION_SIZE,
    FACTUAL_SLOTS_PER_BRANCH,
    OUTSIDE_FAMILY,
    build_confirmation_factual_batches,
    build_confirmation_factual_population,
    build_confirmation_outcome_batch,
    build_confirmation_pair_specs,
    build_confirmation_schedule,
    build_identical_input_conflict_control,
    catalog_fingerprint,
    factual_indices_for_update,
    factual_schedule_fingerprint,
    schedule_fingerprint,
)
from cure_lite.losses import CURELiteLoss
from cure_lite.paired_endpoint_crossing_losses import (
    PairedEndpointCrossingLoss,
)
from cure_lite.train.paired_outcome_step import (
    outcome_complete_train_step,
)
from cure_lite.train.paired_step import _paired_endpoint_logits
from tools.evaluate_peco_exposure_confirmation import (
    EXPECTED_GROUP_CONTRACT,
    _build_frozen_adam,
    _component_update_histogram,
    _decoder_fingerprint,
    _exact_group_contract,
    _group_results,
    _load_implementation_closure,
    _load_pre_run_receipt,
    _load_protocol,
    _write_result,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "paired_endpoint_crossing_objective_v10"
)
CONFIG = PROTOCOL / "exposure_confirmation_config_r2.json"
RECEIPT = PROTOCOL / "exposure_confirmation_design_receipt_r2.json"
INVALIDATION = (
    PROTOCOL
    / "exposure_confirmation_design_r1_invalidation_receipt.json"
)
CLOSURE = (
    PROTOCOL / "exposure_confirmation_implementation_closure_r3.json"
)
PRE_RUN = (
    PROTOCOL / "exposure_confirmation_r3_pre_run_verification_receipt.json"
)
R2_CLOSURE = (
    PROTOCOL / "exposure_confirmation_implementation_closure_r2.json"
)
R2_FAILURE = (
    PROTOCOL / "exposure_confirmation_r2_execution_failure_receipt.json"
)
POPULATION_SOURCE = (
    ROOT / "cure_lite" / "experiment" / "peco_exposure_confirmation.py"
)
EVALUATOR = ROOT / "tools" / "evaluate_peco_exposure_confirmation.py"

CATALOG_FINGERPRINT = (
    "0391b44f28cecfff3e05b6bfe55c0cfc364df64df04e42bb712f83b6554b0f4c"
)
SCHEDULE_FINGERPRINT = (
    "6b7235fe52a12065c516869b7e445f42e1a418b42d9c6efe8194074345a216c2"
)
INITIAL_DECODER_FINGERPRINT = (
    "88be60d9e2297b03a291cb6c52f345ce00e4afc1f6d9d0ed21e6dc6affa76886"
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _verify_fingerprint(
    value: dict[str, object],
    *,
    field: str,
) -> None:
    unsigned = dict(value)
    observed = unsigned.pop(field)
    assert stable_fingerprint(unsigned) == observed


def test_population_is_exact_deterministic_and_exposure_balanced_by_group() -> None:
    first = build_confirmation_pair_specs()
    second = build_confirmation_pair_specs()

    assert first == second
    assert len(first) == 222
    assert Counter(spec.pair_kind for spec in first) == {
        "clean_positive": CLEAN_PAIR_COUNT,
        "component_null": COMPONENT_PAIR_COUNT,
    }
    assert len({spec.pair_id for spec in first}) == 222
    assert len({spec.sample_id for spec in first}) == 222
    assert catalog_fingerprint(first) == CATALOG_FINGERPRINT

    clean = [
        spec for spec in first if spec.pair_kind == "clean_positive"
    ]
    component = [
        spec for spec in first if spec.pair_kind == "component_null"
    ]
    assert Counter(spec.exposure_count for spec in clean) == {4: 121, 3: 85}
    assert Counter(spec.exposure_count for spec in component) == {4: 13, 3: 3}
    assert sum(spec.exposure_count for spec in clean) == CLEAN_SLOT_COUNT
    assert sum(spec.exposure_count for spec in component) == (
        COMPONENT_SLOT_COUNT
    )

    by_group: dict[str, Counter[int]] = defaultdict(Counter)
    for spec in first:
        by_group[spec.group_id][spec.exposure_count] += 1
    assert set(by_group) == {
        "clean_contains_1px",
        "clean_contains_2px",
        "clean_contains_3px",
        "clean_outside_1px",
        "clean_outside_2px",
        "clean_outside_3px",
        "component_null_block",
        "component_null_sparse",
    }
    assert all(counter[3] > 0 and counter[4] > 0 for counter in by_group.values())


def test_schedule_exactly_replays_the_800_slot_exposure_contract() -> None:
    specs = build_confirmation_pair_specs()
    first = build_confirmation_schedule(specs)
    second = build_confirmation_schedule(specs)

    assert first == second
    assert len(first) == 400
    assert schedule_fingerprint(specs) == SCHEDULE_FINGERPRINT
    assert Counter(update.component_count for update in first) == {
        0: 340,
        1: 59,
        2: 1,
    }
    assert sum(len(update.pair_ids) for update in first) == 800
    assert all(len(set(update.pair_ids)) == 2 for update in first)
    assert all(len(set(update.sample_ids)) == 2 for update in first)
    histogram = _component_update_histogram(first)
    assert histogram == {"0": 340, "1": 59, "2": 1}
    assert all(isinstance(key, str) for key in histogram)
    assert isinstance(
        stable_fingerprint({"component_update_histogram": histogram}),
        str,
    )
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        stable_fingerprint(
            {"component_update_histogram": {0: 340, 1: 59, 2: 1}}
        )

    exposure = Counter(
        population_index
        for update in first
        for population_index in update.population_indices
    )
    assert exposure == Counter(
        {
            spec.population_index: spec.exposure_count
            for spec in specs
        }
    )


def test_factual_schedule_replays_16_plus_16_states_exactly_100_times() -> None:
    populations = build_confirmation_factual_population()
    assert populations["factual_miss"].feature.shape[0] == (
        FACTUAL_POPULATION_SIZE
    )
    assert populations["factual_no_miss"].feature.shape[0] == (
        FACTUAL_POPULATION_SIZE
    )
    assert FACTUAL_BATCH_SIZE == 4
    assert FACTUAL_SLOTS_PER_BRANCH == 1600
    assert FACTUAL_EXPOSURES_PER_STATE == 100

    exposure = Counter(
        index
        for update_index in range(400)
        for index in factual_indices_for_update(update_index)
    )
    assert exposure == Counter(
        {index: 100 for index in range(FACTUAL_POPULATION_SIZE)}
    )
    assert factual_indices_for_update(0) == (0, 1, 2, 3)
    assert factual_indices_for_update(1) == (4, 5, 6, 7)
    assert factual_indices_for_update(3) == (12, 13, 14, 15)
    assert factual_indices_for_update(4) == (0, 1, 2, 3)
    with pytest.raises(ValueError):
        factual_indices_for_update(400)

    for update_index in (0, 1, 2, 3, 399):
        batches = build_confirmation_factual_batches(
            update_index=update_index,
        )
        assert batches["factual_miss"].feature.shape[0] == 4
        assert batches["factual_no_miss"].feature.shape[0] == 4


def test_all_geometry_groups_materialize_valid_D_H_G_states() -> None:
    specs = build_confirmation_pair_specs()
    observed_clean_shapes: set[tuple[str, int]] = set()
    for offset in range(0, len(specs), 2):
        selected = (specs[offset], specs[offset + 1])
        outcome = build_confirmation_outcome_batch(selected)
        outcome.validate()
        assert outcome.pair_batch.feature.requires_grad is False
        assert outcome.pair_batch.feature.shape == (2, 8, 2, 2)
        assert outcome.pair_batch.occupancy_plus.shape == (2, 1, 8, 8)
        assert torch.all(
            outcome.global_zero_stratum.flatten(1).any(dim=1)
        )
        for index, spec in enumerate(selected):
            response_count = int(
                outcome.response_stratum[index].sum()
            )
            assert response_count == spec.response_pixel_count
            assert bool(outcome.local_zero_stratum[index].any())
            if spec.pair_kind == "clean_positive":
                observed_clean_shapes.add(
                    (spec.geometry_family, response_count)
                )
                overlap = (
                    outcome.response_stratum[index]
                    & outcome.removed_component[index]
                )
                if spec.geometry_family == CONTAINS_FAMILY:
                    assert int(overlap.sum()) == response_count
                elif spec.geometry_family == OUTSIDE_FAMILY:
                    assert not bool(overlap.any())
                else:
                    raise AssertionError("unknown clean geometry")
            else:
                assert response_count == 0
    assert observed_clean_shapes == {
        (CONTAINS_FAMILY, 1),
        (CONTAINS_FAMILY, 2),
        (CONTAINS_FAMILY, 3),
        (OUTSIDE_FAMILY, 1),
        (OUTSIDE_FAMILY, 2),
        (OUTSIDE_FAMILY, 3),
    }


def test_identical_input_conflict_is_structural_and_has_no_role_channel() -> None:
    outcome = build_identical_input_conflict_control()
    assert outcome.pair_batch.pair_kinds == (
        "clean_positive",
        "component_null",
    )
    assert torch.equal(
        outcome.pair_batch.feature[0],
        outcome.pair_batch.feature[1],
    )
    assert torch.equal(
        outcome.pair_batch.occupancy_plus[0],
        outcome.pair_batch.occupancy_plus[1],
    )
    assert torch.equal(
        outcome.pair_batch.occupancy_minus[0],
        outcome.pair_batch.occupancy_minus[1],
    )
    assert outcome.response_stratum.flatten(1).sum(1).tolist() == [1, 0]
    conflict_pixel = outcome.response_stratum[0]
    assert bool(outcome.local_zero_stratum[1][conflict_pixel].all())

    torch.manual_seed(42)
    decoder = CURELiteConservativeFactorizedDecoder(
        feature_channels=8,
        feature_stride=4,
    )
    with torch.no_grad():
        plus, minus = _paired_endpoint_logits(
            decoder,
            feature=outcome.pair_batch.feature,
            occupancy_plus=outcome.pair_batch.occupancy_plus,
            occupancy_minus=outcome.pair_batch.occupancy_minus,
        )
    assert torch.equal(plus[0], plus[1])
    assert torch.equal(minus[0], minus[1])

    loss_signature = inspect.signature(PairedEndpointCrossingLoss.forward)
    decoder_signature = inspect.signature(decoder.forward)
    assert all("pair_kind" not in name for name in loss_signature.parameters)
    assert all("pair_kind" not in name for name in decoder_signature.parameters)
    assert "pair_kind" not in inspect.getsource(
        PairedEndpointCrossingLoss.forward
    )


def test_one_update_uses_2B_endpoint_forward_and_detached_features() -> None:
    specs = build_confirmation_pair_specs()
    selected = (specs[0], specs[CLEAN_PAIR_COUNT])
    outcome = build_confirmation_outcome_batch(selected)
    factual = build_confirmation_factual_batches(update_index=0)
    pair_feature = outcome.pair_batch.feature.clone().requires_grad_()
    outcome = type(outcome)(
        pair_batch=type(outcome.pair_batch)(
            feature=pair_feature,
            occupancy_plus=outcome.pair_batch.occupancy_plus,
            occupancy_minus=outcome.pair_batch.occupancy_minus,
            label_increment=outcome.pair_batch.label_increment,
            image_valid_mask=outcome.pair_batch.image_valid_mask,
            pair_ids=outcome.pair_batch.pair_ids,
            sample_ids=outcome.pair_batch.sample_ids,
            group_ids=outcome.pair_batch.group_ids,
            pair_kinds=outcome.pair_batch.pair_kinds,
            projection_visible=outcome.pair_batch.projection_visible,
        ),
        completion_plus=outcome.completion_plus,
        completion_minus=outcome.completion_minus,
        gt_union=outcome.gt_union,
        intervention_footprint=outcome.intervention_footprint,
    )
    torch.manual_seed(42)
    decoder = CURELiteConservativeFactorizedDecoder(
        feature_channels=8,
        feature_stride=4,
    )
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
    forward_sizes: list[int] = []
    handle = decoder.register_forward_pre_hook(
        lambda _module, args: forward_sizes.append(int(args[0].shape[0]))
    )
    try:
        logs = outcome_complete_train_step(
            decoder,
            CURELiteLoss(),
            PairedEndpointCrossingLoss(LossConfig()),
            optimizer,
            factual,
            outcome,
        )
    finally:
        handle.remove()

    assert forward_sizes == [4, 4, 4]
    assert logs["decoder_forward_calls_per_update"] == 3
    assert logs["decoder_states_per_update"] == 12
    assert logs["outcome/clean_pairs"] == 1
    assert logs["outcome/component_null_pairs"] == 1
    assert pair_feature.grad is None
    assert all(parameter.grad is not None for parameter in decoder.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in decoder.parameters()
    )


def test_confirmation_adam_is_explicit_and_checked_before_training() -> None:
    torch.manual_seed(42)
    decoder = CURELiteConservativeFactorizedDecoder(
        feature_channels=8,
        feature_stride=4,
    )
    optimizer, contract = _build_frozen_adam(decoder.parameters())

    assert isinstance(optimizer, torch.optim.Adam)
    assert contract == {
        "name": "Adam",
        "learning_rate": 0.001,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": False,
    }


def test_group_gate_uses_worst_pair_not_a_population_mean() -> None:
    thresholds = _load(CONFIG)["thresholds"]
    assert isinstance(thresholds, dict)
    good = {
        "pair_id": "a",
        "group_id": "clean_contains_1px",
        "pair_kind": "clean_positive",
        "exposure_count": 4,
        "plus_completion_min": 0.99,
        "plus_background_max": 0.01,
        "H_max_abs": 0.01,
        "G_max_abs": 0.01,
        "D_plus_max": 0.01,
        "D_minus_min": 0.99,
        "D_delta_mean": 0.98,
    }
    bad = dict(good)
    bad["pair_id"] = "b"
    bad["D_delta_mean"] = 0.79
    results, all_pass = _group_results([good, bad], thresholds)

    assert len(results) == 1
    assert results[0]["metrics"]["D_delta_mean_min"] == 0.79
    assert results[0]["checks"]["D_delta"] is False
    assert results[0]["all_pass"] is False
    assert all_pass is False


def test_exact_group_gate_rejects_a_missing_group() -> None:
    groups = [
        {
            "group_id": group_id,
            "metrics": counts,
        }
        for group_id, counts in EXPECTED_GROUP_CONTRACT.items()
    ]
    complete = _exact_group_contract(groups)
    missing = _exact_group_contract(groups[:-1])

    assert complete["all_pass"] is True
    assert missing["checks"]["exact_eight_group_set"] is False
    assert missing["all_pass"] is False


def test_protocol_binds_sources_counts_optimizer_and_closed_boundaries() -> None:
    config = _load(CONFIG)
    receipt = _load(RECEIPT)
    _verify_fingerprint(config, field="config_fingerprint")
    _verify_fingerprint(receipt, field="receipt_fingerprint")
    loaded, binding = _load_protocol()
    assert loaded == config
    assert binding["config_file_sha256"] == file_sha256(CONFIG)

    assert config["population"]["catalog_fingerprint"] == CATALOG_FINGERPRINT
    assert config["schedule"]["schedule_fingerprint"] == SCHEDULE_FINGERPRINT
    assert config["optimization"] == {
        "device": "cpu",
        "seed": 42,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "adam_betas": [0.9, 0.999],
        "adam_epsilon": 1.0e-8,
        "adam_amsgrad": False,
        "adam_maximize": False,
        "adam_foreach": None,
        "adam_capturable": False,
        "adam_differentiable": False,
        "adam_fused": None,
        "adam_decoupled_weight_decay": False,
        "all_fields_explicitly_passed": True,
        "exact_defaults_checked_before_training": True,
        "updates": 400,
        "automatic_retry_allowed": False,
        "hyperparameter_search_allowed": False,
    }
    assert config["exact_group_contract"] == EXPECTED_GROUP_CONTRACT
    assert config["r1_invalidation_binding"] == {
        "repo_path": (
            "protocols/IRSTD-1K/"
            "paired_endpoint_crossing_objective_v10/"
            "exposure_confirmation_design_r1_invalidation_receipt.json"
        ),
        "file_sha256": file_sha256(INVALIDATION),
        "receipt_fingerprint": _load(INVALIDATION)[
            "receipt_fingerprint"
        ],
        "decision": (
            "PECO_V10_EXPOSURE_CONFIRMATION_R1_DESIGN_INVALIDATED"
        ),
    }
    assert config["decoder"]["expected_seed42_initial_fingerprint"] == (
        INITIAL_DECODER_FINGERPRINT
    )
    assert config["schedule"][
        "factual_miss_population_size"
    ] == FACTUAL_POPULATION_SIZE
    assert config["schedule"][
        "factual_no_miss_population_size"
    ] == FACTUAL_POPULATION_SIZE
    assert config["schedule"]["factual_miss_slots"] == 1600
    assert config["schedule"]["factual_no_miss_slots"] == 1600
    assert config["schedule"]["factual_exposures_per_state"] == 100
    assert config["schedule"]["factual_schedule_fingerprint"] == (
        factual_schedule_fingerprint()
    )
    assert config["matched_comparator"]["same_initialization"] is True
    assert config["crossing_semantics"] == {
        "boundary": "training_logit_zero",
        "equivalent_probability": 0.5,
        "is_final_frozen_calibration_threshold": False,
        "detection_threshold_claim_allowed": False,
    }
    dependencies = config["transitive_dependency_bindings"]
    assert isinstance(dependencies, list)
    paths = [binding["repo_path"] for binding in dependencies]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    assert all(
        file_sha256(ROOT / binding["repo_path"])
        == binding["file_sha256"]
        for binding in dependencies
    )
    assert receipt["transitive_dependency_bindings"] == dependencies
    assert config["decision_rule"][
        "matched_comparator_cannot_change_peco_decision"
    ] is True
    assert config["decision_rule"][
        "every_update_all_six_gradients_finite_nonzero"
    ] is True
    assert config["decision_rule"][
        "pass_authorizes_only_deterministic_dry_run"
    ] is True
    assert config["decision_rule"][
        "runtime_contract_must_be_reported"
    ] is True
    assert config["execution_boundary"] == {
        "dataset_free_confirmation_run_authorized": True,
        "dry_run_authorized": False,
        "D_R_access_allowed": False,
        "D_V_access_allowed": False,
        "D_T_access_allowed": False,
        "detection_performance_allowed": False,
        "real_bounded_authorized": False,
        "formal_800_authorized": False,
        "full_CURE_authorized": False,
        "cross_detector_authorized": False,
    }
    assert receipt["pre_run_status"]["authority_run_performed"] is False
    assert receipt["interpretation"] == (
        "frozen_r2_dataset_free_confirmation_design_not_a_model_result"
    )
    assert receipt["test_binding"] == {
        "repo_path": "tests_v10/test_peco_exposure_confirmation.py",
        "file_sha256": (
            "8608213a9dd506d3cd9d8dcf27b18ad42c7fefe8cf64e1aea674d604eeb4a063"
        ),
        "policy": (
            "schedule_geometry_optimizer_group_negative_control_"
            "protocol_and_one_step_without_authority_run"
        ),
    }


def test_seed42_initialization_and_bound_source_hashes_are_exact() -> None:
    config = _load(CONFIG)
    torch.manual_seed(42)
    decoder = CURELiteConservativeFactorizedDecoder(
        feature_channels=8,
        feature_stride=4,
    )
    assert _decoder_fingerprint(decoder) == INITIAL_DECODER_FINGERPRINT
    assert len(tuple(decoder.parameters())) == 6
    assert sum(parameter.numel() for parameter in decoder.parameters()) == 2593
    assert file_sha256(POPULATION_SOURCE) == config[
        "population_source_binding"
    ]["file_sha256"]
    historical_sources = _load(R2_CLOSURE)["source_bindings"]
    historical_evaluator = next(
        binding
        for binding in historical_sources
        if binding["repo_path"]
        == "tools/evaluate_peco_exposure_confirmation.py"
    )
    assert historical_evaluator == {
        "repo_path": config["evaluator_binding"]["repo_path"],
        "file_sha256": config["evaluator_binding"]["file_sha256"],
    }
    assert file_sha256(EVALUATOR) != historical_evaluator["file_sha256"]


def test_r3_closure_and_pre_run_receipt_bind_the_bounded_correction() -> None:
    code = """
import json
from tools.evaluate_peco_exposure_confirmation import (
    _load_implementation_closure,
    _load_pre_run_receipt,
    _load_protocol,
)
_, protocol_binding = _load_protocol()
implementation_binding = _load_implementation_closure(protocol_binding)
pre_run_binding = _load_pre_run_receipt(
    protocol_binding,
    implementation_binding,
)
print(json.dumps({
    "implementation": implementation_binding,
    "pre_run": pre_run_binding,
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)
    implementation_binding = observed["implementation"]
    pre_run_binding = observed["pre_run"]
    closure = _load(CLOSURE)
    pre_run = _load(PRE_RUN)
    failure = _load(R2_FAILURE)

    _verify_fingerprint(closure, field="receipt_fingerprint")
    _verify_fingerprint(pre_run, field="receipt_fingerprint")
    assert implementation_binding["file_sha256"] == file_sha256(CLOSURE)
    assert pre_run_binding["file_sha256"] == file_sha256(PRE_RUN)
    assert closure["r2_execution_failure_binding"][
        "receipt_fingerprint"
    ] == failure["receipt_fingerprint"]
    corrected = {
        binding["repo_path"]: binding["file_sha256"]
        for binding in closure["corrected_source_bindings"]
    }
    assert corrected == {
        "tests_v10/test_peco_exposure_confirmation.py": file_sha256(
            Path(__file__)
        ),
        "tools/evaluate_peco_exposure_confirmation.py": file_sha256(
            EVALUATOR
        ),
    }


def test_authority_writer_is_exclusive_and_tests_do_not_run_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "already-exists.json"
    output.write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _write_result(output, {"all_pass": True})
    monkeypatch.setattr(
        confirmation_evaluator,
        "evaluate",
        lambda: (_ for _ in ()).throw(
            AssertionError("evaluate must not run for an existing output")
        ),
    )
    with pytest.raises(FileExistsError):
        confirmation_evaluator.main(["--output", str(output)])

    source = EVALUATOR.read_text(encoding="utf-8")
    assert "datasets" not in source
    assert "D_R" in source and "D_V" in source and "D_T" in source
    assert 'with path.open("x"' in source


def test_authority_entry_rejects_noncanonical_absent_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "not-canonical.json"
    called = False

    def forbidden_evaluate() -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("evaluate must not run for a noncanonical path")

    monkeypatch.setattr(
        confirmation_evaluator,
        "evaluate",
        forbidden_evaluate,
    )
    with pytest.raises(ValueError, match="single frozen canonical path"):
        confirmation_evaluator.main(["--output", str(output)])
    assert called is False
