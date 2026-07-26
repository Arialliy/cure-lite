from __future__ import annotations

from collections import Counter
import math
from pathlib import Path

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.nlcc_dataset_free_decision import InvalidResultError
from cure_lite.nlcc_dataset_free_inputs import (
    build_factual_batches,
    build_outcome_batch,
)
from cure_lite.nlcc_dataset_free_runner import (
    RESULT_SCHEMA,
    PreRunAuthorization,
    build_training_components,
    claim_execution,
    decoder_fingerprint,
    evaluate_cached_logits,
    evaluate_trained_decoder,
    load_pre_run_authorization,
    materialize_profile,
    publish_failure,
    publish_result,
    runtime_import_boundary,
)
from cure_lite.null_anchored_local_count_crossing_decoder import (
    CURELiteNullAnchoredLocalCountCrossingDecoder,
)
from cure_lite.nlcc_dataset_free_runner_config import (
    development_runner_config,
    holdout_runner_config,
)


ZERO_SHA = "0" * 64


@pytest.fixture(scope="module")
def development_cache():
    return materialize_profile(development_runner_config())


@pytest.fixture(scope="module")
def holdout_cache():
    return materialize_profile(holdout_runner_config())


def _authorization(config) -> PreRunAuthorization:
    return PreRunAuthorization(
        profile_id=config.profile.profile_id,
        profile_kind=config.profile.kind,
        attempt_ordinal=1,
        repo_path=config.profile.pre_run_authorization,
        file_sha256=ZERO_SHA,
        authorization_fingerprint=ZERO_SHA,
        implementation_closure_fingerprint=ZERO_SHA,
        source_bindings={},
    )


def _assert_same_branch(left, right) -> None:
    for field in ("feature", "occupancy", "target", "valid_mask"):
        assert torch.equal(getattr(left, field), getattr(right, field))


def _assert_same_outcome(left, right) -> None:
    for field in (
        "feature",
        "occupancy_plus",
        "occupancy_minus",
        "label_increment",
        "image_valid_mask",
    ):
        assert torch.equal(
            getattr(left.pair_batch, field),
            getattr(right.pair_batch, field),
        )
    for field in (
        "pair_ids",
        "sample_ids",
        "group_ids",
        "pair_kinds",
        "projection_visible",
    ):
        assert getattr(left.pair_batch, field) == getattr(right.pair_batch, field)
    for field in (
        "completion_plus",
        "completion_minus",
        "gt_union",
        "intervention_footprint",
    ):
        assert torch.equal(getattr(left, field), getattr(right, field))


@pytest.mark.parametrize("update_index", [0, 17, 319])
def test_cached_development_views_are_bit_exact_to_direct_builders(
    development_cache,
    update_index: int,
) -> None:
    cached_factual, cached_outcome = development_cache.training_batches(update_index)
    update = development_cache.schedule[update_index]
    direct_specs = tuple(
        development_cache.specs[index] for index in update.population_indices
    )
    direct_outcome = build_outcome_batch(
        development_cache.input_profile,
        direct_specs,
    )
    direct_factual = build_factual_batches(
        development_cache.input_profile,
        update_index=update_index,
    )
    _assert_same_outcome(cached_outcome, direct_outcome)
    for branch in ("factual_miss", "factual_no_miss"):
        _assert_same_branch(cached_factual[branch], direct_factual[branch])


def test_materialized_profiles_replay_exact_exposures_without_builder_reentry(
    development_cache,
    holdout_cache,
) -> None:
    for cache, expected_updates, expected_slots in (
        (development_cache, 320, 640),
        (holdout_cache, 400, 800),
    ):
        manifest = cache.manifest()
        assert manifest["updates"] == expected_updates
        assert manifest["pair_slots"] == expected_slots
        assert manifest["scientific_pair_population_builder_calls"] == 1
        assert manifest["scientific_factual_population_builder_calls"] == 1
        assert manifest["scientific_schedule_builder_calls"] == 1
        assert manifest["per_update_builder_reentry"] is False
        observed = Counter(
            index
            for update in cache.schedule
            for index in update.population_indices
        )
        expected = Counter(
            {spec.population_index: spec.exposure_count for spec in cache.specs}
        )
        assert observed == expected
        assert all(
            set(update.anchor_roles) == {"anchor_positive", "anchor_null"}
            for update in cache.schedule
        )


def test_holdout_static_group_rows_and_D_pixels_are_exact(holdout_cache) -> None:
    expected = {
        "clean_same_cell_1px": (36, 36),
        "clean_same_cell_3px": (34, 102),
        "clean_adjacent_cell_1px": (34, 34),
        "clean_adjacent_cell_3px": (34, 102),
        "clean_multicount_2to1": (34, 66),
        "clean_multicount_3to2": (34, 66),
        "component_null_block": (8, 0),
        "component_null_sparse": (8, 0),
    }
    for group_id, (row_count, D_pixels) in expected.items():
        rows = torch.tensor(
            [spec.group_id == group_id for spec in holdout_cache.specs],
            dtype=torch.bool,
        ).reshape(-1, 1, 1, 1)
        assert int(rows.sum()) == row_count
        assert int((rows & holdout_cache.strata.D).sum()) == D_pixels


def _passing_logits(cache):
    pair_shape = cache.pair_population.pair_batch.label_increment.shape
    plus = torch.full(pair_shape, -20.0)
    minus = plus.clone()
    positive_anchor = cache.pair_population.completion_plus
    plus[positive_anchor] = 20.0
    minus[positive_anchor] = 20.0
    D = cache.strata.D
    plus[D] = -20.0
    minus[D] = 20.0

    miss = cache.factual_population["factual_miss"]
    miss_logits = torch.full_like(miss.target, -20.0)
    miss_logits[miss.target > 0.5] = 20.0
    no_miss = cache.factual_population["factual_no_miss"]
    no_miss_logits = torch.full_like(no_miss.target, -20.0)
    return plus, minus, miss_logits, no_miss_logits


def test_full_population_gate_algebra_and_component_D_semantics(
    development_cache,
) -> None:
    plus, minus, miss, no_miss = _passing_logits(development_cache)
    result = evaluate_cached_logits(
        development_cache,
        logits_plus=plus,
        logits_minus=minus,
        factual_miss_logits=miss,
        factual_no_miss_logits=no_miss,
        structural_training_contract={"all_pass": True},
    )
    assert result["all_pass"] is True
    assert result["numeric_gate_count"] == 76
    assert len(result["groups"]) == 8
    expected_D = {
        "clean_same_cell_1px": 4,
        "clean_same_cell_3px": 12,
        "clean_adjacent_cell_1px": 4,
        "clean_adjacent_cell_3px": 12,
        "clean_multicount_2to1": 8,
        "clean_multicount_3to2": 8,
    }
    for group in result["groups"]:
        if group["pair_kind"] == "clean_positive":
            assert group["D_gate_status"] == "APPLICABLE"
            assert group["metrics"]["clean_D_pixel_count"] == expected_D[
                group["group_id"]
            ]
            assert group["metrics"]["D_wrong_direction_pixel_count"] == 0
        else:
            assert group["D_gate_status"] == "NOT_APPLICABLE_EMPTY_D"
            assert group["metrics"]["clean_D_pixel_count"] == 0
            assert group["metrics"]["clean_D_delta_mean"] is None
            assert "clean_D_delta_mean" not in group["checks"]
        gaps = group["metrics"]["matched_twin_gap"]
        assert len(gaps["matches"]) == group["metrics"]["row_count"] // 2
        assert gaps["is_gate"] is False
    assert result["final_forward_contract"]["unique_pair_rows_equal_weight"] is True
    assert result["final_forward_contract"]["exposure_weighted"] is False


def test_final_evaluation_uses_only_three_full_population_forward_fields_calls(
    development_cache,
) -> None:
    class CountingDecoder(CURELiteNullAnchoredLocalCountCrossingDecoder):
        def __init__(self) -> None:
            super().__init__(feature_channels=8, feature_stride=4)
            self.observed_batches: list[int] = []

        def forward_fields(self, feature, occupancy):
            self.observed_batches.append(int(feature.shape[0]))
            return super().forward_fields(feature, occupancy)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        decoder = CountingDecoder()
    result = evaluate_trained_decoder(
        decoder,
        development_cache,
        structural_training_contract={"all_pass": False},
    )
    assert decoder.observed_batches == [
        2 * len(development_cache.specs),
        16,
        16,
    ]
    assert result["final_forward_contract"]["total_decoder_calls"] == 3
    assert result["final_forward_contract"]["repeated_group_forwards"] is False


def test_exclusive_and_inclusive_threshold_boundaries(development_cache) -> None:
    plus, minus, miss, no_miss = _passing_logits(development_cache)
    target_group = "clean_same_cell_1px"
    spec_index = next(
        index
        for index, spec in enumerate(development_cache.specs)
        if spec.group_id == target_group and spec.anchor_role == "anchor_positive"
    )
    anchor = development_cache.pair_population.completion_plus[spec_index]
    boundary = math.log(0.95 / 0.05)
    plus[spec_index][anchor] = boundary
    minus[spec_index][anchor] = boundary
    result = evaluate_cached_logits(
        development_cache,
        logits_plus=plus,
        logits_minus=minus,
        factual_miss_logits=miss,
        factual_no_miss_logits=no_miss,
        structural_training_contract={"all_pass": True},
    )
    group = next(row for row in result["groups"] if row["group_id"] == target_group)
    assert group["metrics"]["positive_anchor_min"] == pytest.approx(0.95)
    assert group["checks"]["positive_anchor"] is False
    # Inclusive zero-response gates accept exact equality by frozen algebra.
    assert 0.05 <= development_cache.config.thresholds.zero_H_max_abs_max_inclusive


def test_runtime_boundary_and_missing_authority_fail_before_attempt(tmp_path: Path) -> None:
    assert runtime_import_boundary()["all_pass"] is True
    config = development_runner_config()
    with pytest.raises(FileNotFoundError, match="authorization is absent"):
        load_pre_run_authorization(config, repo_root=tmp_path)
    assert not (tmp_path / config.profile.canonical_artifact_directory).exists()


def test_create_only_claim_is_concurrent_safe_and_precedes_adam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = development_runner_config()
    authority = claim_execution(config, _authorization(config), repo_root=tmp_path)
    attempt = authority.artifact_directory / "attempt.json"
    assert attempt.is_file()
    assert (authority.artifact_directory / ".incomplete").is_file()
    with pytest.raises(FileExistsError):
        claim_execution(config, _authorization(config), repo_root=tmp_path)

    original_adam = torch.optim.Adam
    training_started = authority.artifact_directory / "training_started.json"
    observed = {
        "attempt_exists_at_optimizer_construction": False,
        "training_start_exists_at_optimizer_construction": False,
    }

    def checked_adam(*args, **kwargs):
        observed["attempt_exists_at_optimizer_construction"] = attempt.is_file()
        observed["training_start_exists_at_optimizer_construction"] = (
            training_started.is_file()
        )
        return original_adam(*args, **kwargs)

    monkeypatch.setattr(torch.optim, "Adam", checked_adam)
    components = build_training_components(authority, config)
    assert observed["attempt_exists_at_optimizer_construction"] is True
    assert observed["training_start_exists_at_optimizer_construction"] is True
    assert components.optimizer_state_initially_empty is True
    assert not components.optimizer.state
    with pytest.raises(RuntimeError, match="state differs|already consumed"):
        build_training_components(authority, config)


def _passing_result(
    cache,
    *,
    initial_decoder_fingerprint: str = "1" * 64,
) -> dict[str, object]:
    config = cache.config
    plus, minus, miss_logits, no_miss_logits = _passing_logits(cache)
    structural = {
        "updates_executed": config.profile.updates,
        "expected_updates": config.profile.updates,
        "training_forward_call_count": 3 * config.profile.updates,
        "expected_training_forward_call_count": 3 * config.profile.updates,
        "training_forward_pattern_counts": {"4,4,4": config.profile.updates},
        "all_update_forward_patterns_4_4_4": True,
        "step_contract_failure_count": 0,
        "gradient_failure_count": 0,
        "finite_state_audit_count": config.profile.updates + 1,
        "expected_finite_state_audit_count": config.profile.updates + 1,
        "finite_state_nonfinite_element_count": 0,
        "all_six_gradients_finite_nonzero_every_update": True,
        "feature_cache_grad_tensor_count": 0,
        "feature_cache_leaves_remain_without_grad": True,
        "one_backward_and_one_step_per_update": True,
        "population_builder_reentry": False,
        "from_scratch_seed_42": True,
        "fresh_adam_state_before_first_update": True,
        "development_checkpoint_loaded": False,
        "development_optimizer_state_loaded": False,
        "all_pass": True,
    }
    miss = cache.factual_population["factual_miss"]
    no_miss = cache.factual_population["factual_no_miss"]
    final = evaluate_cached_logits(
        cache,
        logits_plus=plus,
        logits_minus=minus,
        factual_miss_logits=miss_logits,
        factual_no_miss_logits=no_miss_logits,
        structural_training_contract=structural,
        operator_field_diagnostics={
            "crossing_margin": {"minimum": -1.0, "maximum": 1.0},
            "recovery_factor": {"minimum": 0.1, "maximum": 2.0},
            "forward_fields_call_count": 3,
            "forward_fields_batch_sizes": [
                2 * len(cache.specs),
                int(miss.feature.shape[0]),
                int(no_miss.feature.shape[0]),
            ],
            "field_tensor_count": 45,
            "field_element_count": 1,
            "field_nonfinite_element_count": 0,
        },
    )
    initial_audit = {
        "phase": "before_first_update",
        "update_index": -1,
        "parameter_tensor_count": 6,
        "buffer_tensor_count": 0,
        "optimizer_state_tensor_count": 0,
        "total_tensor_count": 6,
        "total_element_count": 2593,
        "nonfinite_element_count": 0,
        "nonfinite_tensor_paths": [],
        "maximum_absolute_value": 1.0,
        "global_l2_norm": 1.0,
        "all_finite": True,
    }
    final_audit = {
        **initial_audit,
        "phase": "after_optimizer_step",
        "update_index": config.profile.updates - 1,
        "optimizer_state_tensor_count": 18,
        "total_tensor_count": 24,
        "total_element_count": 3 * 2593 + 6,
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "method_id": "nlcc_v12",
        "profile_id": config.profile.profile_id,
        "profile_kind": config.profile.kind,
        "decision": "NLCC_V12_DEVELOPMENT_PASS",
        "all_pass": True,
        "config": config.manifest(),
        "materialized_cache": cache.manifest(),
        "training": {
            "structural_contract": structural,
            "gradient_failures": [],
            "step_contract_failures": [],
            "finite_state_audit": {
                "call_count": config.profile.updates + 1,
                "expected_call_count": config.profile.updates + 1,
                "initial": initial_audit,
                "final": final_audit,
                "nonfinite_element_count": 0,
            },
        },
        "final_evaluation": final,
        "initial_decoder_fingerprint": initial_decoder_fingerprint,
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


def test_holdout_requires_sealed_development_pass_and_starts_from_scratch(
    tmp_path: Path,
) -> None:
    holdout = holdout_runner_config()
    with pytest.raises(RuntimeError, match="sealed development PASS"):
        claim_execution(holdout, _authorization(holdout), repo_root=tmp_path)
    assert not (tmp_path / holdout.profile.canonical_artifact_directory).exists()

    development = development_runner_config()
    development_authority = claim_execution(
        development,
        _authorization(development),
        repo_root=tmp_path,
    )
    development_components = build_training_components(
        development_authority,
        development,
    )
    publish_result(
        development_authority,
        _passing_result(
            materialize_profile(development),
            initial_decoder_fingerprint=(
                development_components.initial_decoder_fingerprint
            ),
        ),
    )

    holdout_authority = claim_execution(
        holdout,
        _authorization(holdout),
        repo_root=tmp_path,
    )
    holdout_components = build_training_components(holdout_authority, holdout)
    assert development_components.initial_decoder_fingerprint == (
        holdout_components.initial_decoder_fingerprint
    )
    assert decoder_fingerprint(development_components.decoder) == (
        decoder_fingerprint(holdout_components.decoder)
    )
    assert not development_components.optimizer.state
    assert not holdout_components.optimizer.state
    assert development_components.optimizer is not holdout_components.optimizer
    assert development_components.decoder is not holdout_components.decoder


def test_result_publication_is_create_only_and_independently_sealed(
    tmp_path: Path,
) -> None:
    config = development_runner_config()
    authority = claim_execution(config, _authorization(config), repo_root=tmp_path)
    build_training_components(authority, config)
    sealed = publish_result(
        authority,
        _passing_result(materialize_profile(config)),
    )
    directory = authority.artifact_directory
    assert {path.name for path in directory.iterdir()} == {
        "attempt.json",
        "training_started.json",
        "result.json",
        "decision.json",
        "COMPLETE.json",
    }
    assert sealed["decision"]["all_pass"] is True
    assert sealed["complete"]["files"].keys() == {
        "attempt.json",
        "training_started.json",
        "result.json",
        "decision.json",
    }
    with pytest.raises(FileExistsError):
        publish_result(authority, _passing_result(materialize_profile(config)))


def test_result_publication_rejects_skeletal_or_derived_only_pass(
    tmp_path: Path,
) -> None:
    config = development_runner_config()
    authority = claim_execution(config, _authorization(config), repo_root=tmp_path)
    build_training_components(authority, config)
    result = _passing_result(materialize_profile(config))
    result.pop("final_evaluation")
    result.pop("result_fingerprint")
    result["result_fingerprint"] = stable_fingerprint(result)

    with pytest.raises(InvalidResultError, match="missing keys"):
        publish_result(authority, result)
    assert not (authority.artifact_directory / "result.json").exists()


def test_existing_result_terminal_blocks_failure_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cure_lite import nlcc_dataset_free_runner as runner

    config = development_runner_config()
    authority = claim_execution(config, _authorization(config), repo_root=tmp_path)
    build_training_components(authority, config)

    def fail_after_result_write(*args, **kwargs):
        raise RuntimeError("simulated seal failure")

    monkeypatch.setattr(runner, "_seal_directory", fail_after_result_write)
    with pytest.raises(RuntimeError, match="simulated seal failure"):
        publish_result(authority, _passing_result(materialize_profile(config)))
    assert (authority.artifact_directory / "result.json").is_file()
    assert not (authority.artifact_directory / "failure.json").exists()
    with pytest.raises(FileExistsError, match="terminal"):
        publish_failure(authority, RuntimeError("must not create second terminal"))
    assert not (authority.artifact_directory / "failure.json").exists()
