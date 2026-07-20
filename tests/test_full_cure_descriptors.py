from __future__ import annotations

import pytest
import torch

from cure_lite.cure import (
    CUREProtocol,
    DescriptorConfig,
    CUREResidualConfig,
    build_eligible_sample_catalog,
    extract_frozen_source_record,
)
from cure_lite.cure.descriptors import extract_target_descriptor
from cure_lite.cure.protocol import module_state_fingerprint
from cure_lite.instances import instances_from_binary_mask
from cure_lite.matching import match_components
from cure_lite.provenance import BaseCheckpointSelection
from cure_lite.splits import SplitManifest, SplitRecord
from cure_lite.toy import ToyFrozenBaseAdapter


def _manifest() -> SplitManifest:
    return SplitManifest(
        dataset="toy",
        records=(
            SplitRecord("base-fit", "D_B", "base-fit-group", "base-fit.png"),
            SplitRecord("base-select", "D_B", "base-select-group", "base-select.png"),
            SplitRecord("image", "D_R", "sequence", "image.png"),
            SplitRecord("validation", "D_V", "validation-group", "validation.png"),
            SplitRecord("test", "D_T", "test-group", "test.png"),
        ),
    )


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
    )


def _state():
    gt_mask = torch.zeros(9, 9, dtype=torch.bool)
    gt_mask[2, 2] = True
    gt_mask[6, 6] = True
    occupancy = torch.zeros_like(gt_mask)
    occupancy[2, 2] = True
    gt = instances_from_binary_mask(gt_mask)
    match = match_components(instances_from_binary_mask(occupancy), gt)
    probability = torch.full((9, 9), 0.1)
    probability[2, 2] = 0.9
    probability[6, 6] = 0.3
    intensity = torch.zeros(9, 9)
    intensity[2, 2] = 1.0
    intensity[6, 6] = 0.4
    return occupancy, gt, match, probability, intensity


def test_descriptor_role_is_derived_from_common_eligible_universe() -> None:
    _, gt, match, probability, intensity = _state()
    legal = extract_target_descriptor(
        sample_id="image",
        group_id="sequence",
        gt=gt,
        gt_id=1,
        match=match,
        eligible_factual_gt_ids=(2,),
        legal_covered_gt_ids=(1,),
        probability=probability,
        intensity=intensity,
        config=DescriptorConfig(ring_inner_radius=1, ring_outer_radius=3),
    )
    factual = extract_target_descriptor(
        sample_id="image",
        group_id="sequence",
        gt=gt,
        gt_id=2,
        match=match,
        eligible_factual_gt_ids=(2,),
        legal_covered_gt_ids=(1,),
        probability=probability,
        intensity=intensity,
        config=DescriptorConfig(ring_inner_radius=1, ring_outer_radius=3),
    )
    assert legal.role == "legal_covered"
    assert factual.role == "factual_miss"
    assert legal.values.shape == factual.values.shape == (7,)
    assert legal.values.dtype == factual.values.dtype == torch.float64


def test_descriptor_rejects_target_outside_eligible_pool() -> None:
    _, gt, match, probability, intensity = _state()
    with pytest.raises(ValueError, match="common eligible"):
        extract_target_descriptor(
            sample_id="image",
            group_id="sequence",
            gt=gt,
            gt_id=1,
            match=match,
            eligible_factual_gt_ids=(2,),
            legal_covered_gt_ids=(),
            probability=probability,
            intensity=intensity,
        )


def test_canonical_catalog_derives_factual_and_legal_pools_from_state() -> None:
    _, gt, _, _, intensity = _state()
    manifest = _manifest()
    base = ToyFrozenBaseAdapter()
    protocol = _protocol(manifest, base)
    source = extract_frozen_source_record(
        base=base,
        images=intensity[None, None],
        gt=gt,
        sample_id="image",
        group_id="sequence",
        protocol=protocol,
        manifest=manifest,
    )
    catalog = build_eligible_sample_catalog(
        source=source,
        protocol=protocol,
        manifest=manifest,
    )
    assert tuple((item.gt_id, item.role) for item in catalog.descriptors) == (
        (1, "legal_covered"),
        (2, "factual_miss"),
    )
    assert tuple(item.gt_id for item in catalog.legal_deletions) == (1,)
    assert catalog.excluded_factual_gt_ids == ()
    assert catalog.excluded_covered_gt_ids == ()


def test_canonical_source_fixes_intensity_and_rejects_post_issue_mutation() -> None:
    _, gt, _, _, intensity = _state()
    manifest = _manifest()
    base = ToyFrozenBaseAdapter()
    protocol = _protocol(manifest, base)
    images = intensity[None, None].clone()
    source = extract_frozen_source_record(
        base=base,
        images=images,
        gt=gt,
        sample_id="image",
        group_id="sequence",
        protocol=protocol,
        manifest=manifest,
    )
    torch.testing.assert_close(source.intensity, intensity)
    assert source.adapter_state_fingerprint == module_state_fingerprint(base)

    # The issued record owns a snapshot, so later caller mutation is harmless.
    images[0, 0, 0, 0] = 1.0
    source.validate_receipt()

    # In-place mutation of a signed source tensor is detected at consumption.
    source.probability[0, 0, 0, 0] = 0.2
    with pytest.raises(ValueError, match="content differs"):
        build_eligible_sample_catalog(
            source=source,
            protocol=protocol,
            manifest=manifest,
        )


def test_canonical_source_rejects_adapter_not_bound_to_protocol() -> None:
    _, gt, _, _, intensity = _state()
    manifest = _manifest()
    base = ToyFrozenBaseAdapter()
    protocol = CUREProtocol.from_manifest(
        manifest,
        base_fingerprint="toy-base",
        adapter_fingerprint="different-adapter",
        base_state_fingerprint=module_state_fingerprint(base),
        preprocessing_fingerprint="toy-preprocessing",
        residual_config=CUREResidualConfig(feature_channels=base.feature_channels),
        base_checkpoint_selection=BaseCheckpointSelection.from_manifest(
            manifest,
            fit_sample_ids=("base-fit",),
            select_sample_ids=("base-select",),
        ),
    )
    with pytest.raises(ValueError, match="adapter differs"):
        extract_frozen_source_record(
            base=base,
            images=intensity[None, None],
            gt=gt,
            sample_id="image",
            group_id="sequence",
            protocol=protocol,
            manifest=manifest,
        )


def test_canonical_source_rejects_reused_identity_with_changed_base_state() -> None:
    _, gt, _, _, intensity = _state()
    manifest = _manifest()
    base = ToyFrozenBaseAdapter()
    protocol = CUREProtocol.from_manifest(
        manifest,
        base_fingerprint="toy-base",
        adapter_fingerprint=base.fingerprint,
        base_state_fingerprint="0" * 64,
        preprocessing_fingerprint="toy-preprocessing",
        residual_config=CUREResidualConfig(feature_channels=base.feature_channels),
        base_checkpoint_selection=BaseCheckpointSelection.from_manifest(
            manifest,
            fit_sample_ids=("base-fit",),
            select_sample_ids=("base-select",),
        ),
    )
    with pytest.raises(ValueError, match="parameter/buffer state differs"):
        extract_frozen_source_record(
            base=base,
            images=intensity[None, None],
            gt=gt,
            sample_id="image",
            group_id="sequence",
            protocol=protocol,
            manifest=manifest,
        )


def test_canonical_source_rejects_stateful_adapter_execution() -> None:
    class StatefulAdapter(ToyFrozenBaseAdapter):
        def extract(self, images: torch.Tensor):
            with torch.no_grad():
                self.base.probability_head.bias.add_(0.01)
            return super().extract(images)

    _, gt, _, _, intensity = _state()
    manifest = _manifest()
    base = StatefulAdapter()
    protocol = _protocol(manifest, base)
    with pytest.raises(RuntimeError, match="module state changed"):
        extract_frozen_source_record(
            base=base,
            images=intensity[None, None],
            gt=gt,
            sample_id="image",
            group_id="sequence",
            protocol=protocol,
            manifest=manifest,
        )
