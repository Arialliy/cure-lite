from __future__ import annotations

from bisect import bisect_right
from dataclasses import replace
import math

import pytest
import torch

from cure_lite.cure import (
    CURELossConfig,
    CUREProtocol,
    CUREResidualConfig,
    CUREResidualDecoder,
    CUREStateExample,
    CURETrainingPolicy,
    CUREUncensoringLoss,
    CounterfactualBackgroundPolicy,
    CounterfactualSamplingPolicy,
    CounterfactualTargetPolicy,
    DescriptorConfig,
    PropensityConfig,
    bind_weighted_candidates,
    build_counterfactual_residual_set_supervision,
    build_cure_state_pool,
    build_eligible_sample_catalog,
    build_factual_residual_set_supervision,
    build_fair_sampling_policy_family,
    cross_fit_miss_propensity,
    draw_fixed_exposure_batch,
    extract_frozen_source_record,
    train_cure_step,
)
from cure_lite.cure.training import CURESamplingReceipt, _score_strata_weights
from cure_lite.cure.protocol import module_state_fingerprint
from cure_lite.instances import instances_from_binary_mask
from cure_lite.matching import match_components
from cure_lite.provenance import BaseCheckpointSelection
from cure_lite.splits import SplitManifest, SplitRecord
from cure_lite.toy import ToyFrozenBaseAdapter


def _manifest() -> SplitManifest:
    records = [
        SplitRecord(
            f"source-{index}",
            "D_R",
            f"source-group-{index}",
            f"source-{index}.png",
        )
        for index in range(4)
    ]
    records.extend(
        (
            SplitRecord("base-fit", "D_B", "base-fit-group", "base-fit.png"),
            SplitRecord("base-select", "D_B", "base-select-group", "base-select.png"),
            SplitRecord("validation", "D_V", "validation-group", "validation.png"),
            SplitRecord("test", "D_T", "test-group", "test.png"),
        )
    )
    return SplitManifest(dataset="toy", records=tuple(records))


def _training_artifacts(
    *,
    sampling_policy: CounterfactualSamplingPolicy = (
        CounterfactualSamplingPolicy.ODDS
    ),
    target_policy: CounterfactualTargetPolicy = (
        CounterfactualTargetPolicy.SELECTED_DELETED_TARGET_ONLY
    ),
    background_policy: CounterfactualBackgroundPolicy = (
        CounterfactualBackgroundPolicy.EMPTY
    ),
):
    manifest = _manifest()
    residual_config = CUREResidualConfig(
        feature_channels=3,
        width=8,
        groups=4,
        occupancy_threshold=0.5,
        suppression_radius=0,
    )
    base = ToyFrozenBaseAdapter()
    protocol = CUREProtocol.from_manifest(
        manifest,
        base_fingerprint="toy-base",
        base_state_fingerprint=module_state_fingerprint(base),
        adapter_fingerprint=base.fingerprint,
        preprocessing_fingerprint="toy-preprocessing",
        residual_config=residual_config,
        base_checkpoint_selection=BaseCheckpointSelection.from_manifest(
            manifest,
            fit_sample_ids=("base-fit",),
            select_sample_ids=("base-select",),
        ),
        loss_config=CURELossConfig(background_bce_weight=1.0),
        descriptor_config=DescriptorConfig(
            ring_inner_radius=1,
            ring_outer_radius=3,
        ),
        propensity_config=PropensityConfig(
            folds=2,
            l2=0.1,
            max_iterations=100,
        ),
    )
    training_policy = CURETrainingPolicy.bind(
        protocol,
        sampling_policy=sampling_policy,
        factual_count=2,
        counterfactual_count=3,
        global_seed=5,
        target_policy=target_policy,
        background_policy=background_policy,
        score_strata_count=3,
        placebo_seed=17,
    )
    catalogs = []
    gt_by_sample = {}
    factual_states = []
    counterfactual_states = {}
    frozen = {}
    for index in range(4):
        sample_id = f"source-{index}"
        gt_mask = torch.zeros(9, 9, dtype=torch.bool)
        gt_mask[2, 2] = True
        gt_mask[6, 6] = True
        gt = instances_from_binary_mask(gt_mask)
        desired_miss_probability = 0.2 + 0.02 * index
        desired_covered_probability = 0.55 + 0.10 * index
        image = torch.zeros(1, 1, 9, 9)
        image[0, 0, 2, 2] = (desired_covered_probability - 0.1) / 0.8
        image[0, 0, 6, 6] = (desired_miss_probability - 0.1) / 0.8
        source = extract_frozen_source_record(
            base=base,
            images=image,
            gt=gt,
            sample_id=sample_id,
            group_id=f"source-group-{index}",
            protocol=protocol,
            manifest=manifest,
        )
        catalog = build_eligible_sample_catalog(
            source=source,
            protocol=protocol,
            manifest=manifest,
        )
        catalogs.append(catalog)
        occupancy = source.probability[0, 0] >= 0.5
        before = match_components(instances_from_binary_mask(occupancy), source.gt)
        gt_by_sample[sample_id] = source.gt
        frozen[sample_id] = (
            source.feature,
            source.probability,
            occupancy,
            before,
        )

    catalogs = tuple(catalogs)
    propensity = cross_fit_miss_propensity(
        catalogs,
        protocol=protocol,
        manifest=manifest,
    )
    intervention_catalog = bind_weighted_candidates(
        catalogs,
        gt_by_sample,
        propensity,
        protocol=protocol,
    )
    candidate_by_sample = {
        item.sample_id: item for item in intervention_catalog.candidates
    }
    for sample_id in sorted(frozen):
        feature, probability, occupancy, before = frozen[sample_id]
        gt = gt_by_sample[sample_id]
        factual_supervision = build_factual_residual_set_supervision(
            occupancy,
            gt,
            before,
            protocol.match_config,
            suppression_radius=protocol.residual_config.suppression_radius,
        )
        factual_states.append(
            CUREStateExample.bind(
                sample_id,
                feature,
                probability,
                factual_supervision,
                protocol,
            )
        )
        candidate = candidate_by_sample[sample_id]
        counter_supervision = build_counterfactual_residual_set_supervision(
            candidate.deletion,
            gt,
            before,
            occupancy,
            protocol.match_config,
            suppression_radius=protocol.residual_config.suppression_radius,
            target_policy=training_policy.target_policy,
            background_policy=training_policy.background_policy,
        )
        counterfactual_states[candidate.key] = CUREStateExample.bind(
            sample_id,
            feature.clone(),
            probability.clone(),
            counter_supervision,
            protocol,
        )
    return (
        tuple(factual_states),
        counterfactual_states,
        intervention_catalog,
        gt_by_sample,
        protocol,
        training_policy,
    )


def _draw():
    (
        factual,
        counterfactual,
        catalog,
        gt_by_sample,
        protocol,
        training_policy,
    ) = _training_artifacts()
    state_pool = build_cure_state_pool(
        factual,
        counterfactual,
        catalog,
        gt_by_sample,
        training_policy=training_policy,
    )
    batch = draw_fixed_exposure_batch(
        state_pool,
        epoch=0,
        step=0,
        device="cpu",
    )
    return batch, protocol


def test_fixed_exposure_draw_and_train_step_update_only_decoder() -> None:
    batch, protocol = _draw()
    assert batch.feature.shape[0] == 5
    assert sum(item.branch == "factual" for item in batch.supervisions) == 2
    assert sum(item.branch == "counterfactual" for item in batch.supervisions) == 3

    decoder = CUREResidualDecoder(protocol.residual_config)
    criterion = CUREUncensoringLoss(protocol.loss_config)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    before = {
        name: value.detach().clone() for name, value in decoder.state_dict().items()
    }
    losses = train_cure_step(decoder, criterion, optimizer, batch)
    assert torch.isfinite(losses["total"])
    assert any(
        not torch.equal(before[name], value)
        for name, value in decoder.state_dict().items()
    )


def test_draw_rejects_feature_probability_or_supervision_substitution() -> None:
    factual, counterfactual, catalog, gt_by_sample, _, policy = _training_artifacts()
    substitutions = []
    changed_probability = factual[0].base_probability.clone()
    changed_probability[0, 0, 0, 0] += 0.01
    substitutions.append(replace(factual[0], base_probability=changed_probability))
    changed_feature = factual[0].feature.clone()
    changed_feature[0, 0, 0, 0] += 1.0
    substitutions.append(replace(factual[0], feature=changed_feature))
    for forged in substitutions:
        with pytest.raises(ValueError, match="receipt|differs"):
            build_cure_state_pool(
                (forged, *factual[1:]),
                counterfactual,
                catalog,
                gt_by_sample,
                training_policy=policy,
            )

    first_key = sorted(counterfactual)[0]
    counter = counterfactual[first_key]
    # Move the positive to another editable pixel while keeping the supervision
    # object internally valid.  The canonical builder receipt must still reject it.
    forged_target = torch.zeros_like(counter.supervision.target)
    forged_target[0, 4, 4] = 1.0
    forged_supervision = replace(
        counter.supervision,
        target=forged_target,
        object_masks=forged_target.to(torch.bool),
    )
    forged_counter = replace(counter, supervision=forged_supervision)
    changed_states = dict(counterfactual)
    changed_states[first_key] = forged_counter
    with pytest.raises(ValueError, match="bound policy builder"):
        build_cure_state_pool(
            factual,
            changed_states,
            catalog,
            gt_by_sample,
            training_policy=policy,
        )


def test_train_rejects_decoder_loss_or_receipt_bypass() -> None:
    batch, protocol = _draw()
    wrong_decoder = CUREResidualDecoder(
        replace(protocol.residual_config, suppression_radius=1)
    )
    optimizer = torch.optim.Adam(wrong_decoder.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="decoder configuration"):
        train_cure_step(
            wrong_decoder,
            CUREUncensoringLoss(protocol.loss_config),
            optimizer,
            batch,
        )

    class DoubleWeightedLoss(CUREUncensoringLoss):
        pass

    decoder = CUREResidualDecoder(protocol.residual_config)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    with pytest.raises(TypeError, match="criterion"):
        train_cure_step(
            decoder,
            DoubleWeightedLoss(protocol.loss_config),
            optimizer,
            batch,
        )

    with pytest.raises(ValueError, match="not issued"):
        CURESamplingReceipt(
            protocol_fingerprint=protocol.fingerprint,
            catalog_fingerprint="0" * 64,
            state_pool_fingerprint="0" * 64,
            state_universe_fingerprint="0" * 64,
            policy_fingerprint="0" * 64,
            policy_schema_version="cure-training-policy-v1",
            sampler_implementation="global-with-replacement-v1",
            sampling_policy="odds",
            target_policy="selected-deleted-target-only",
            background_policy="empty",
            schedule_name="fixed-per-step-v1",
            control_weight_floor=1e-6,
            score_strata_transported_mass_fraction=None,
            score_strata_unmatched_mass_fraction=None,
            epoch=0,
            step=0,
            global_seed=0,
            factual_count=1,
            counterfactual_count=0,
            selected_factual_sample_ids=("source-0",),
            selected_candidate_keys=(),
            selected_state_fingerprints=("0" * 64,),
            _seal=object(),
        )

    for field in ("feature", "target"):
        tampered, tampered_protocol = _draw()
        if field == "feature":
            tampered.feature[0, 0, 0, 0] += 1.0
        else:
            tampered.supervisions[0].target[0, 0, 0] = 1.0
        decoder = CUREResidualDecoder(tampered_protocol.residual_config)
        optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
        with pytest.raises(ValueError, match="state differs"):
            train_cure_step(
                decoder,
                CUREUncensoringLoss(tampered_protocol.loss_config),
                optimizer,
                tampered,
            )


def test_five_sampling_controls_share_support_states_and_fixed_exposure() -> None:
    factual, counterfactual, catalog, gt_by_sample, _, reference = (
        _training_artifacts()
    )
    policies = build_fair_sampling_policy_family(reference)
    assert tuple(item.sampling_policy for item in policies) == tuple(
        CounterfactualSamplingPolicy
    )
    invariant_fields = {
        (
            item.factual_count,
            item.counterfactual_count,
            item.global_seed,
            item.target_policy,
            item.background_policy,
            item.schedule_name,
            item.sampler_implementation,
            item.control_weight_floor,
        )
        for item in policies
    }
    assert len(invariant_fields) == 1

    pools = tuple(
        build_cure_state_pool(
            factual,
            counterfactual,
            catalog,
            gt_by_sample,
            training_policy=policy,
        )
        for policy in policies
    )
    assert len({pool.catalog_fingerprint for pool in pools}) == 1
    assert len({pool.state_universe_fingerprint for pool in pools}) == 1
    assert len(
        {
            tuple(key for key, _ in pool.counterfactual_states)
            for pool in pools
        }
    ) == 1
    for pool in pools:
        assert len(pool.candidate_sampling_weights) == len(catalog.candidates)
        assert all(weight > 0.0 for weight in pool.candidate_sampling_weights)
        batch = draw_fixed_exposure_batch(pool, epoch=2, step=7, device="cpu")
        receipt = batch.sampling_receipt
        assert receipt.policy_fingerprint == pool.training_policy.fingerprint
        assert receipt.sampling_policy == pool.training_policy.sampling_policy.value
        assert (receipt.factual_count, receipt.counterfactual_count) == (2, 3)
        assert receipt.global_seed == 5
        assert receipt.state_universe_fingerprint == pool.state_universe_fingerprint

    by_policy = {pool.training_policy.sampling_policy: pool for pool in pools}
    odds = by_policy[CounterfactualSamplingPolicy.ODDS]
    uniform = by_policy[CounterfactualSamplingPolicy.UNIFORM]
    placebo = by_policy[CounterfactualSamplingPolicy.ODDS_PLACEBO]
    score_hard = by_policy[CounterfactualSamplingPolicy.SCORE_HARD]
    score_strata = by_policy[CounterfactualSamplingPolicy.SCORE_STRATA]
    assert odds.candidate_sampling_weights == tuple(
        item.weight for item in catalog.candidates
    )
    assert uniform.candidate_sampling_weights == (1.0,) * len(catalog.candidates)
    assert sorted(placebo.candidate_sampling_weights) == sorted(
        odds.candidate_sampling_weights
    )
    assert math.fsum(placebo.candidate_sampling_weights) == math.fsum(
        odds.candidate_sampling_weights
    )
    assert placebo.total_candidate_weight == odds.total_candidate_weight
    placebo_ess = math.fsum(placebo.candidate_sampling_weights) ** 2 / math.fsum(
        value * value for value in placebo.candidate_sampling_weights
    )
    odds_ess = math.fsum(odds.candidate_sampling_weights) ** 2 / math.fsum(
        value * value for value in odds.candidate_sampling_weights
    )
    assert placebo_ess == odds_ess
    assert placebo.candidate_effective_sample_size == odds.candidate_effective_sample_size

    score_weight_pairs = sorted(
        zip(
            (value for _, value in score_hard.candidate_target_scores),
            score_hard.candidate_sampling_weights,
            strict=True,
        )
    )
    assert all(
        left_weight >= right_weight
        for (_, left_weight), (_, right_weight) in zip(
            score_weight_pairs, score_weight_pairs[1:], strict=False
        )
    )

    boundaries = score_strata.score_strata_boundaries
    factual_bins: dict[int, int] = {}
    legal_weight_by_bin: dict[int, float] = {}
    for _, score in score_strata.factual_target_scores:
        index = bisect_right(boundaries, score)
        factual_bins[index] = factual_bins.get(index, 0) + 1
    for (_, score), weight in zip(
        score_strata.candidate_target_scores,
        score_strata.candidate_sampling_weights,
        strict=True,
    ):
        index = bisect_right(boundaries, score)
        legal_weight_by_bin[index] = legal_weight_by_bin.get(index, 0.0) + weight
    factual_total = len(score_strata.factual_target_scores)
    for index, legal_mass in legal_weight_by_bin.items():
        assert legal_mass == pytest.approx(factual_bins[index] / factual_total)
    transported = sum(factual_bins[index] for index in legal_weight_by_bin) / factual_total
    assert score_strata.score_strata_transported_mass_fraction == pytest.approx(
        transported
    )
    assert score_strata.score_strata_unmatched_mass_fraction == pytest.approx(
        1.0 - transported
    )


def test_score_strata_floor_retains_a_zero_factual_mass_legal_bin() -> None:
    weights, transported, unmatched = _score_strata_weights(
        boundaries=(0.5,),
        factual_values=(0.2,),
        legal_values=(0.2, 0.8),
        weight_floor=1e-4,
    )
    assert weights == (1.0, 1e-4)
    assert transported == 1.0
    assert unmatched == 0.0
    with pytest.raises(ValueError, match="zero U_M/U_L"):
        _score_strata_weights(
            boundaries=(0.5,),
            factual_values=(0.2,),
            legal_values=(0.8,),
            weight_floor=1e-4,
        )


def test_bound_exposure_seed_and_policy_cannot_be_overridden() -> None:
    factual, counterfactual, catalog, gt_by_sample, _, policy = _training_artifacts()
    pool = build_cure_state_pool(
        factual,
        counterfactual,
        catalog,
        gt_by_sample,
        training_policy=policy,
    )
    for override in (
        {"factual_count": 3},
        {"counterfactual_count": 2},
        {"global_seed": 6},
    ):
        with pytest.raises(ValueError, match="cannot override"):
            draw_fixed_exposure_batch(
                pool,
                epoch=0,
                step=0,
                device="cpu",
                **override,
            )
    with pytest.raises(ValueError, match="content differs"):
        replace(policy, factual_count=3)


@pytest.mark.parametrize(
    ("target_policy", "background_policy"),
    (
        (
            CounterfactualTargetPolicy.ALL_UNCOVERED_TARGETS,
            CounterfactualBackgroundPolicy.EMPTY,
        ),
        (
            CounterfactualTargetPolicy.SELECTED_DELETED_TARGET_ONLY,
            CounterfactualBackgroundPolicy.BCE,
        ),
    ),
)
def test_supervision_ablations_are_policy_bound_and_sampling_orthogonal(
    target_policy: CounterfactualTargetPolicy,
    background_policy: CounterfactualBackgroundPolicy,
) -> None:
    factual, counterfactual, catalog, gt_by_sample, _, policy = _training_artifacts(
        target_policy=target_policy,
        background_policy=background_policy,
    )
    assert policy.sampling_policy is CounterfactualSamplingPolicy.ODDS
    assert (policy.factual_count, policy.counterfactual_count, policy.global_seed) == (
        2,
        3,
        5,
    )
    pool = build_cure_state_pool(
        factual,
        counterfactual,
        catalog,
        gt_by_sample,
        training_policy=policy,
    )
    batch = draw_fixed_exposure_batch(pool, epoch=0, step=0, device="cpu")
    counter_states = [
        item for item in batch.supervisions if item.branch == "counterfactual"
    ]
    if target_policy is CounterfactualTargetPolicy.ALL_UNCOVERED_TARGETS:
        assert all(len(item.positive_gt_ids) == 2 for item in counter_states)
    if background_policy is CounterfactualBackgroundPolicy.BCE:
        assert all(torch.any(item.background_mask) for item in counter_states)

    default = policy.with_supervision_policy(
        target_policy=CounterfactualTargetPolicy.SELECTED_DELETED_TARGET_ONLY,
        background_policy=CounterfactualBackgroundPolicy.EMPTY,
    )
    assert default.sampling_policy == policy.sampling_policy
    assert (default.factual_count, default.counterfactual_count, default.global_seed) == (
        policy.factual_count,
        policy.counterfactual_count,
        policy.global_seed,
    )
    if default.fingerprint != policy.fingerprint:
        with pytest.raises(ValueError, match="bound policy builder"):
            build_cure_state_pool(
                factual,
                counterfactual,
                catalog,
                gt_by_sample,
                training_policy=default,
            )
