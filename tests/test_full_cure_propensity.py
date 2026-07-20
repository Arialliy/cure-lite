from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cure_lite.cure import (
    CUREProtocol,
    CUREResidualConfig,
    DescriptorConfig,
    PropensityConfig,
    bind_weighted_candidates,
    build_eligible_sample_catalog,
    choose_weighted_candidate,
    cross_fit_miss_propensity,
    extract_frozen_source_record,
    miss_odds,
)
from cure_lite.cure.types import WeightedCounterfactualCandidate
from cure_lite.cure.protocol import module_state_fingerprint
from cure_lite.instances import instances_from_binary_mask
from cure_lite.provenance import BaseCheckpointSelection
from cure_lite.splits import SplitManifest, SplitRecord
from cure_lite.toy import ToyFrozenBaseAdapter


def _manifest() -> SplitManifest:
    records = [
        SplitRecord(
            sample_id=f"image-{group:02d}",
            split="D_R",
            group_id=f"sequence-{group:02d}",
            image=f"/tmp/cure-image-{group:02d}.png",
            sequence_id=f"sequence-{group:02d}",
        )
        for group in range(20)
    ]
    records.extend(
        (
            SplitRecord("base-fit", "D_B", "base-fit-group", "/tmp/cure-base-fit.png"),
            SplitRecord("base-select", "D_B", "base-select-group", "/tmp/cure-base-select.png"),
            SplitRecord("valid", "D_V", "valid-group", "/tmp/cure-valid.png"),
            SplitRecord("test", "D_T", "test-group", "/tmp/cure-test.png"),
        )
    )
    return SplitManifest(dataset="toy", records=tuple(records))


def _protocol(
    manifest: SplitManifest,
    base: ToyFrozenBaseAdapter,
) -> CUREProtocol:
    return CUREProtocol.from_manifest(
        manifest,
        base_fingerprint="toy-base",
        adapter_fingerprint=base.fingerprint,
        base_state_fingerprint=module_state_fingerprint(base),
        preprocessing_fingerprint="toy-preprocessing",
        residual_config=CUREResidualConfig(feature_channels=base.feature_channels),
        base_checkpoint_selection=BaseCheckpointSelection.from_manifest(
            manifest,
            fit_sample_ids=("base-fit",),
            select_sample_ids=("base-select",),
        ),
        descriptor_config=DescriptorConfig(
            ring_inner_radius=1,
            ring_outer_radius=3,
        ),
        propensity_config=PropensityConfig(
            folds=5,
            l2=0.1,
            max_iterations=80,
            seed=7,
        ),
    )


def _canonical_catalogs(*, all_covered: bool = False):
    manifest = _manifest()
    base = ToyFrozenBaseAdapter()
    protocol = _protocol(manifest, base)
    catalogs = []
    gt_by_sample = {}
    for group in range(20):
        gt_mask = torch.zeros(9, 9, dtype=torch.bool)
        gt_mask[2, 2] = True
        if not all_covered:
            gt_mask[6, 6] = True
        gt = instances_from_binary_mask(gt_mask)
        covered_probability = 0.90 - 0.019 * group
        image = torch.zeros(1, 1, 9, 9)
        image[0, 0, 2, 2] = (covered_probability - 0.1) / 0.8
        if not all_covered:
            missed_probability = 0.15 + 0.015 * group
            image[0, 0, 6, 6] = (missed_probability - 0.1) / 0.8
        sample_id = f"image-{group:02d}"
        source = extract_frozen_source_record(
            base=base,
            images=image,
            gt=gt,
            sample_id=sample_id,
            group_id=f"sequence-{group:02d}",
            protocol=protocol,
            manifest=manifest,
        )
        catalog = build_eligible_sample_catalog(
            source=source,
            protocol=protocol,
            manifest=manifest,
        )
        catalogs.append(catalog)
        gt_by_sample[sample_id] = source.gt
    return tuple(catalogs), gt_by_sample, protocol, manifest


def test_propensity_is_group_oof_and_assigns_harder_legal_targets_more_weight() -> None:
    catalogs, _, protocol, manifest = _canonical_catalogs()
    result = cross_fit_miss_propensity(
        catalogs,
        protocol=protocol,
        manifest=manifest,
    )
    fold_by_group = dict(result.fold_by_group)
    assert len(fold_by_group) == 20
    assert set(fold_by_group.values()) == set(range(5))
    estimates = [item for item in result.estimates if item.covered]
    low = next(item for item in estimates if item.sample_id == "image-00")
    high = next(item for item in estimates if item.sample_id == "image-19")
    assert high.clipped_probability > low.clipped_probability
    assert high.sampling_weight > low.sampling_weight
    assert 0.0 < result.effective_sample_size <= len(estimates)
    assert 0.0 <= result.brier_score <= 1.0
    assert len(result.smd_before) == len(result.smd_after) == 7
    assert result.protocol_fingerprint == protocol.fingerprint


def test_propensity_accepts_only_canonical_catalogs_and_both_roles() -> None:
    catalogs, _, protocol, manifest = _canonical_catalogs(all_covered=True)
    with pytest.raises(ValueError, match="covered and missed"):
        cross_fit_miss_propensity(
            catalogs,
            protocol=protocol,
            manifest=manifest,
        )
    with pytest.raises(ValueError, match="exactly one catalog"):
        cross_fit_miss_propensity(
            catalogs[1:],
            protocol=protocol,
            manifest=manifest,
        )


def test_miss_odds_clips_and_caps() -> None:
    config = PropensityConfig(clip_epsilon=0.02, max_odds=3.0)
    assert miss_odds(0.999, config) == 3.0
    assert miss_odds(0.001, config) == pytest.approx(0.02 / 0.98)


def test_global_weighted_draw_is_deterministic_and_uses_oof_odds() -> None:
    catalogs, gt_by_sample, protocol, manifest = _canonical_catalogs()
    propensity = cross_fit_miss_propensity(
        catalogs,
        protocol=protocol,
        manifest=manifest,
    )
    receipt = bind_weighted_candidates(
        catalogs,
        gt_by_sample,
        propensity,
        protocol=protocol,
    )
    draws = [
        choose_weighted_candidate(
            receipt,
            epoch=1,
            step=4,
            draw_index=index,
            global_seed=13,
        )
        for index in range(2000)
    ]
    repeated = [
        choose_weighted_candidate(
            receipt,
            epoch=1,
            step=4,
            draw_index=index,
            global_seed=13,
        )
        for index in range(2000)
    ]
    assert [item.key for item in draws] == [item.key for item in repeated]
    easiest = min(receipt.candidates, key=lambda item: item.weight)
    hardest = max(receipt.candidates, key=lambda item: item.weight)
    assert sum(item.key == hardest.key for item in draws) > sum(
        item.key == easiest.key for item in draws
    )


def test_weighted_candidate_rejects_a_forged_non_odds_weight() -> None:
    catalogs, _, _, _ = _canonical_catalogs()
    deletion = catalogs[0].legal_deletions[0]
    with pytest.raises(ValueError, match="capped odds"):
        WeightedCounterfactualCandidate(
            "image-00", deletion, 0.8, 9.0, 10.0
        )


def test_binding_rejects_protocol_or_gt_substitution() -> None:
    catalogs, gt_by_sample, protocol, manifest = _canonical_catalogs()
    propensity = cross_fit_miss_propensity(
        catalogs,
        protocol=protocol,
        manifest=manifest,
    )
    receipt = bind_weighted_candidates(
        catalogs,
        gt_by_sample,
        propensity,
        protocol=protocol,
    )
    assert len(receipt.candidates) == 20
    assert receipt.protocol_fingerprint == protocol.fingerprint

    with pytest.raises(ValueError, match="exactly one catalog"):
        bind_weighted_candidates(
            catalogs[1:],
            gt_by_sample,
            propensity,
            protocol=protocol,
        )

    forged_gt = dict(gt_by_sample)
    forged_gt["image-00"] = instances_from_binary_mask(
        torch.zeros(9, 9, dtype=torch.bool)
    )
    with pytest.raises(ValueError, match="differs from its catalog"):
        bind_weighted_candidates(
            catalogs,
            forged_gt,
            propensity,
            protocol=protocol,
        )


def test_content_addressed_receipts_reject_post_issue_mutation() -> None:
    catalogs, gt_by_sample, protocol, manifest = _canonical_catalogs()
    propensity = cross_fit_miss_propensity(
        catalogs,
        protocol=protocol,
        manifest=manifest,
    )
    receipt = bind_weighted_candidates(
        catalogs,
        gt_by_sample,
        propensity,
        protocol=protocol,
    )
    with pytest.raises(ValueError, match="content differs"):
        replace(catalogs[0], excluded_factual_gt_ids=(99,))
    with pytest.raises(ValueError, match="content differs"):
        replace(propensity, brier_score=min(1.0, propensity.brier_score + 0.01))
    with pytest.raises(ValueError, match="content differs"):
        replace(receipt, propensity_fingerprint="0" * 64)

    mutable_catalogs, _, mutable_protocol, mutable_manifest = _canonical_catalogs()
    mutable_catalogs[0].descriptors[0].values[0] += 1.0
    with pytest.raises(ValueError, match="content differs"):
        cross_fit_miss_propensity(
            mutable_catalogs,
            protocol=mutable_protocol,
            manifest=mutable_manifest,
        )
