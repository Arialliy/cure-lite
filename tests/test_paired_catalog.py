from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cure_lite.cache.state_cache import StateCacheRecord
from cure_lite.experiment.geometry_safe_catalog import (
    GeometrySafeCatalog,
    GeometryTargetRecord,
)
from cure_lite.experiment.paired_catalog import (
    PAIR_CATALOG_SCHEMA,
    build_pair_catalog,
)
from cure_lite.experiment.training_pipeline import (
    CachedTrainingSource,
    prepare_training_catalog,
)
from cure_lite.instances import instances_from_binary_mask
from cure_lite.intervention import enumerate_legal_deletions
from cure_lite.matching import match_components
from cure_lite.occupancy import build_occupancy
from cure_lite.paired_types import (
    PAIR_KINDS,
    PairCatalog,
    PairExample,
    stack_pair_examples,
    tensor_content_fingerprint,
)
from cure_lite.splits import SplitManifest, SplitRecord
from cure_lite.supervision import build_factual_supervision
from cure_lite.toy import (
    ToyFrozenBaseAdapter,
    ToyScene,
    make_factual_miss_scene,
    make_two_target_scene,
)


_PAIRED_PROTOCOL_FINGERPRINT = "1" * 64


def _pairs(rows) -> torch.Tensor:
    return torch.tensor(
        [[row.gt_id, row.pred_id] for row in rows],
        dtype=torch.int64,
    ).reshape(-1, 2)


def _cached_source(scene: ToyScene) -> CachedTrainingSource:
    output = ToyFrozenBaseAdapter()(scene.image_batch())
    occupancy, pred = build_occupancy(output.probability)
    gt = instances_from_binary_mask(scene.gt_mask)
    match = match_components(pred, gt)
    factual = build_factual_supervision(occupancy, gt, match)
    legal = enumerate_legal_deletions(pred, gt, match, occupancy)
    state = StateCacheRecord(
        sample_id=scene.sample_id,
        occupancy=occupancy,
        pred_labels=pred.labels,
        gt_labels=gt.labels,
        base_match_pairs=_pairs(match.pairs),
        real_miss_ids=torch.tensor(
            sorted(match.unmatched_gt_ids),
            dtype=torch.int64,
        ),
        reachable_miss_ids=torch.tensor(
            factual.reachable_gt_ids,
            dtype=torch.int64,
        ),
        legal_pairs=_pairs(legal),
        image_valid_mask=torch.ones_like(occupancy),
    )
    return CachedTrainingSource(
        scene.sample_id,
        output.feature,
        output.probability,
        state,
    )


def _false_positive_scene() -> ToyScene:
    base = make_two_target_scene()
    image = base.image.clone()
    image[0, 15, 15] = 1.0
    return ToyScene(
        sample_id="toy-two-targets-with-fp",
        image=image,
        gt_mask=base.gt_mask.clone(),
        target_masks=tuple(mask.clone() for mask in base.target_masks),
    )


def _manifest(sample_ids: tuple[str, ...]) -> SplitManifest:
    records = [
        SplitRecord("dummy-db", "D_B", "group-db", "dummy-db.png"),
        SplitRecord("dummy-dv", "D_V", "group-dv", "dummy-dv.png"),
        SplitRecord("dummy-dt", "D_T", "group-dt", "dummy-dt.png"),
    ]
    records.extend(
        SplitRecord(
            sample_id,
            "D_R",
            f"group-{sample_id}",
            f"{sample_id}.png",
        )
        for sample_id in sample_ids
    )
    return SplitManifest(dataset="paired-toy", records=tuple(records))


def _geometry_record(entry, candidate, ordinal: int) -> GeometryTargetRecord:
    target = entry.gt.by_id(candidate.gt_id)
    feature_fingerprint = tensor_content_fingerprint(entry.source.feature)
    occupancy_fingerprint = tensor_content_fingerprint(
        candidate.occupancy_after
    )
    return GeometryTargetRecord(
        sample_id=entry.sample_id,
        group_id=f"group-{entry.sample_id}",
        role="legal",
        evaluation_gt_id=candidate.gt_id,
        pred_id=candidate.pred_id,
        candidate_ordinal=ordinal,
        analysis_candidate=True,
        native_ancestor_ids=(candidate.gt_id,),
        native_gt_id=candidate.gt_id,
        native_descendant_evaluation_ids=(candidate.gt_id,),
        reciprocal_one_to_one=True,
        exact_component_projection=True,
        native_area=target.area,
        projected_area=target.area,
        evaluation_area=target.area,
        expected_scaled_area=float(target.area),
        area_ratio=1.0,
        native_centroid=target.centroid,
        expected_evaluation_centroid=target.centroid,
        evaluation_centroid=target.centroid,
        centroid_shift_evaluation_px=0.0,
        geometry_eligible=True,
        geometry_reason_codes=(),
        analysis_eligible=True,
        analysis_exclusion_reasons=(),
        evaluation_mask_fingerprint=tensor_content_fingerprint(target.mask),
        projected_mask_fingerprint=tensor_content_fingerprint(target.mask),
        candidate_occupancy_fingerprint=occupancy_fingerprint,
        synthetic_occupancy_fingerprint=occupancy_fingerprint,
        synthetic_target_fingerprint=tensor_content_fingerprint(
            entry.synthetic_examples[ordinal].supervision.target
        ),
        synthetic_valid_mask_fingerprint=tensor_content_fingerprint(
            entry.synthetic_examples[ordinal].supervision.valid_mask
        ),
        source_state_content_fingerprint=feature_fingerprint,
    )


def _geometry(prepared) -> GeometrySafeCatalog:
    records = tuple(
        _geometry_record(entry, candidate, ordinal)
        for entry in prepared.entries
        for ordinal, candidate in enumerate(
            entry.decoder_visible_legal_candidates
        )
    )
    return GeometrySafeCatalog(
        protocol_fingerprint="2" * 64,
        source_catalog_fingerprint="3" * 64,
        sample_audits=(),
        factual_records=(),
        legal_records=records,
        outside_population_records=(),
        catalog_fingerprint="4" * 64,
    )


@pytest.fixture()
def catalog_inputs():
    sources = (
        _cached_source(make_factual_miss_scene(missed_gt_id=1)),
        _cached_source(make_two_target_scene()),
        _cached_source(_false_positive_scene()),
    )
    prepared = prepare_training_catalog(sources)
    manifest = _manifest(prepared.source_ids)
    geometry = _geometry(prepared)
    return prepared, geometry, manifest


@pytest.fixture()
def pair_catalog(catalog_inputs) -> PairCatalog:
    prepared, geometry, manifest = catalog_inputs
    return build_pair_catalog(
        prepared,
        geometry,
        manifest,
        paired_protocol_fingerprint=_PAIRED_PROTOCOL_FINGERPRINT,
    )


def test_pair_catalog_materializes_clean_and_null_roles(
    pair_catalog: PairCatalog,
) -> None:
    assert PAIR_KINDS == (
        "clean_positive",
        "component_null",
        "identity_null",
    )
    assert pair_catalog.canonical_payload()["schema_version"] == (
        PAIR_CATALOG_SCHEMA
    )
    assert len(pair_catalog.clean_positive) == 5
    assert len(pair_catalog.component_null) == 1
    assert len(pair_catalog.identity_null) == 3
    assert pair_catalog.trainable_pairs is pair_catalog.clean_positive
    assert all(
        row.pair_kind == "clean_positive"
        for row in pair_catalog.trainable_pairs
    )
    assert {
        reason
        for row in pair_catalog.exclusions
        for reason in row.reason_codes
    } == {"actual_increment_nonempty"}


def test_clean_pairs_recompute_exact_instance_completion_truth(
    pair_catalog: PairCatalog,
) -> None:
    for pair in pair_catalog.clean_positive:
        increment = pair.label_increment.to(torch.bool)
        assert torch.equal(
            increment,
            pair.completion_minus & ~pair.completion_plus,
        )
        assert torch.equal(increment, pair.clean_increment)
        assert torch.any(increment)
        assert torch.any(pair.image_valid_mask & ~increment)
        assert torch.equal(
            pair.removed_component,
            pair.occupancy_plus & ~pair.occupancy_minus,
        )
        assert not torch.equal(pair.occupancy_plus, pair.occupancy_minus)
        assert pair.projection_visible is True
        assert pair.feature.requires_grad is False
        assert pair.feature_fingerprint == tensor_content_fingerprint(
            pair.feature
        )


def test_null_pairs_are_controls_and_not_trainable(
    pair_catalog: PairCatalog,
) -> None:
    component = pair_catalog.component_null[0]
    assert component.pair_kind == "component_null"
    assert not torch.any(component.label_increment)
    assert not torch.equal(component.occupancy_plus, component.occupancy_minus)
    assert component.projection_visible
    assert component.evaluation_gt_id is None

    for identity in pair_catalog.identity_null:
        assert identity.pair_kind == "identity_null"
        assert torch.equal(identity.occupancy_plus, identity.occupancy_minus)
        assert torch.equal(identity.completion_plus, identity.completion_minus)
        assert not torch.any(identity.label_increment)
        assert not identity.projection_visible


def test_pair_batch_stores_one_feature_per_pair_and_validates_roles(
    pair_catalog: PairCatalog,
) -> None:
    examples = (
        pair_catalog.clean_positive[0],
        pair_catalog.component_null[0],
        pair_catalog.identity_null[0],
    )
    batch = stack_pair_examples(examples, device="cpu")
    batch.validate()
    assert batch.feature.shape[0] == len(examples)
    assert batch.occupancy_plus.shape[0] == len(examples)
    assert batch.occupancy_minus.shape[0] == len(examples)
    assert batch.pair_kinds == (
        "clean_positive",
        "component_null",
        "identity_null",
    )
    assert batch.feature.numel() == sum(
        example.feature.numel() for example in examples
    )


def test_catalog_is_canonical_and_deterministic(
    pair_catalog: PairCatalog,
    catalog_inputs,
) -> None:
    prepared, geometry, manifest = catalog_inputs
    repeated = build_pair_catalog(
        prepared,
        geometry,
        manifest,
        paired_protocol_fingerprint=_PAIRED_PROTOCOL_FINGERPRINT,
    )
    assert repeated.catalog_fingerprint == pair_catalog.catalog_fingerprint
    assert repeated.canonical_payload() == pair_catalog.canonical_payload()
    all_ids = [
        row.pair_id
        for row in (
            *pair_catalog.clean_positive,
            *pair_catalog.component_null,
            *pair_catalog.identity_null,
        )
    ]
    assert len(all_ids) == len(set(all_ids))


def test_geometry_ineligible_candidate_is_explicitly_excluded(
    catalog_inputs,
) -> None:
    prepared, geometry, manifest = catalog_inputs
    rejected = replace(
        geometry.legal_records[0],
        geometry_eligible=False,
        geometry_reason_codes=("area_ratio_below_minimum",),
        analysis_eligible=False,
        analysis_exclusion_reasons=("area_ratio_below_minimum",),
    )
    altered = replace(
        geometry,
        legal_records=(rejected, *geometry.legal_records[1:]),
        catalog_fingerprint="5" * 64,
    )
    result = build_pair_catalog(
        prepared,
        altered,
        manifest,
        paired_protocol_fingerprint=_PAIRED_PROTOCOL_FINGERPRINT,
    )
    assert len(result.clean_positive) == 4
    exclusion = next(
        row
        for row in result.exclusions
        if row.pair_kind == "clean_positive"
    )
    assert exclusion.reason_codes == (
        "geometry:area_ratio_below_minimum",
    )


def test_pair_example_rejects_a_corrupted_actual_increment(
    pair_catalog: PairCatalog,
) -> None:
    pair = pair_catalog.clean_positive[0]
    corrupted = pair.label_increment.clone()
    corrupted.zero_()
    with pytest.raises(
        ValueError,
        match="completion_minus setminus completion_plus",
    ):
        replace(pair, label_increment=corrupted)
