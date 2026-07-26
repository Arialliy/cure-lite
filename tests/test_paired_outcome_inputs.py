from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.experiment.paired_catalog import build_pair_catalog
from cure_lite.experiment.paired_outcome_inputs import (
    PAIRED_OUTCOME_INPUT_SCHEMA,
    PairedOutcomeInputMaterializer,
    build_paired_outcome_input_materializer,
)
from cure_lite.experiment.training_pipeline import prepare_training_catalog
from cure_lite.paired_outcome_types import OutcomePairBatch
from cure_lite.paired_types import PairCatalog, tensor_content_fingerprint
from cure_lite.toy import make_factual_miss_scene, make_two_target_scene
from tests.test_paired_catalog import (
    _PAIRED_PROTOCOL_FINGERPRINT,
    _cached_source,
    _false_positive_scene,
    _geometry,
    _manifest,
)


def _toy_inputs() -> tuple[object, PairCatalog]:
    sources = (
        _cached_source(make_factual_miss_scene(missed_gt_id=1)),
        _cached_source(make_two_target_scene()),
        _cached_source(_false_positive_scene()),
    )
    prepared = prepare_training_catalog(sources)
    catalog = build_pair_catalog(
        prepared,
        _geometry(prepared),
        _manifest(prepared.source_ids),
        paired_protocol_fingerprint=_PAIRED_PROTOCOL_FINGERPRINT,
    )
    assert catalog.clean_positive
    assert catalog.component_null
    return prepared, catalog


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def _small_catalog(catalog: PairCatalog) -> PairCatalog:
    unsealed = replace(
        catalog,
        clean_positive=catalog.clean_positive[:1],
        component_null=catalog.component_null[:1],
        catalog_fingerprint="",
    )
    return replace(
        unsealed,
        catalog_fingerprint=stable_fingerprint(unsealed.canonical_payload()),
    )


def test_receipt_binds_full_outcome_union_without_raw_tensors() -> None:
    prepared, catalog = _toy_inputs()
    materializer = build_paired_outcome_input_materializer(
        catalog,
        prepared,
    )
    receipt = materializer.canonical_receipt()
    outcome_pairs = (*catalog.clean_positive, *catalog.component_null)

    assert receipt["schema_version"] == PAIRED_OUTCOME_INPUT_SCHEMA
    assert receipt["split"] == "D_R"
    assert receipt["materializer_fingerprint"] == (
        materializer.materializer_fingerprint
    )
    assert receipt["counts"] == {
        "clean_positive_pairs": len(catalog.clean_positive),
        "component_null_pairs": len(catalog.component_null),
        "outcome_pairs": len(outcome_pairs),
        "outcome_pair_sources": len(
            {pair.sample_id for pair in outcome_pairs}
        ),
        "prepared_sources": len(prepared.source_ids),
    }
    rows = {
        row["pair_id"]: row
        for row in receipt["all_outcome_pair_inputs"]
    }
    assert set(rows) == {pair.pair_id for pair in outcome_pairs}
    assert {row["pair_kind"] for row in rows.values()} == {
        "clean_positive",
        "component_null",
    }
    assert not any(pair.pair_id in rows for pair in catalog.identity_null)
    assert receipt["materialization_contract"] == {
        "selection_key": "pair_id",
        "selection_order_preserved": True,
        "optimizer_pair_kinds": ["clean_positive", "component_null"],
        "identity_null_bound": False,
        "all_clean_component_union_pairs_bound": True,
        "exact_completion_endpoints_bound": True,
        "exact_gt_union_bound": True,
        "exact_intervention_footprint_bound": True,
        "stack_function": "stack_outcome_pair_examples",
        "raw_tensor_payloads_written": False,
        "population_integrity_verification": (
            "explicit_once_before_and_once_after_training_not_per_update"
        ),
    }
    assert receipt["execution_policy"] == {
        "runtime_split_access": ["D_R"],
        "D_V_accessed": False,
        "D_T_accessed": False,
        "training_performed": False,
        "calibration_performed": False,
        "inference_performed": False,
    }
    assert not _contains_tensor(receipt)


def test_pair_rows_bind_exact_completion_gt_and_footprint() -> None:
    prepared, catalog = _toy_inputs()
    materializer = build_paired_outcome_input_materializer(
        catalog,
        prepared,
    )
    receipt = materializer.canonical_receipt()
    rows = {
        row["pair_id"]: row
        for row in receipt["all_outcome_pair_inputs"]
    }

    for pair_id in materializer.canonical_pair_ids:
        pair = materializer.pair_by_id[pair_id]
        outcome = materializer.materialize((pair_id,), device="cpu")
        row = rows[pair_id]
        assert row["completion_plus_fingerprint"] == (
            tensor_content_fingerprint(outcome.completion_plus[0])
        )
        assert row["completion_minus_fingerprint"] == (
            tensor_content_fingerprint(outcome.completion_minus[0])
        )
        assert row["gt_union_fingerprint"] == (
            tensor_content_fingerprint(outcome.gt_union[0])
        )
        assert row["intervention_footprint_fingerprint"] == (
            tensor_content_fingerprint(outcome.intervention_footprint[0])
        )
        assert row["pair_manifest_row_fingerprint"] == stable_fingerprint(
            pair.canonical_payload()
        )


def test_mixed_batch_is_selected_by_pair_id_and_preserves_order() -> None:
    prepared, catalog = _toy_inputs()
    materializer = build_paired_outcome_input_materializer(
        catalog,
        prepared,
    )
    pair_ids = (
        materializer.component_null_pair_ids[0],
        materializer.clean_positive_pair_ids[0],
    )
    batch = materializer.materialize(
        pair_ids,
        device=torch.device("cpu"),
    )

    assert isinstance(batch, OutcomePairBatch)
    assert batch.pair_batch.pair_ids == pair_ids
    assert batch.pair_batch.pair_kinds == (
        "component_null",
        "clean_positive",
    )
    assert batch.pair_batch.sample_ids == tuple(
        materializer.pair_by_id[pair_id].sample_id
        for pair_id in pair_ids
    )
    for index, pair_id in enumerate(pair_ids):
        pair = materializer.pair_by_id[pair_id]
        assert torch.equal(batch.pair_batch.feature[index], pair.feature[0])
        assert torch.equal(batch.completion_plus[index], pair.completion_plus)
        assert torch.equal(batch.completion_minus[index], pair.completion_minus)
        assert torch.equal(
            batch.gt_union[index],
            materializer.gt_union_by_sample[pair.sample_id],
        )


def test_materialize_does_not_enter_full_population_rehash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, catalog = _toy_inputs()
    materializer = build_paired_outcome_input_materializer(
        catalog,
        prepared,
    )

    def fail_verify(self: PairedOutcomeInputMaterializer) -> None:
        del self
        raise AssertionError("full population verification entered batch path")

    def fail_payload(self: PairedOutcomeInputMaterializer) -> str:
        del self
        raise AssertionError("full population rehash entered batch path")

    monkeypatch.setattr(
        type(materializer),
        "verify_unchanged",
        fail_verify,
    )
    monkeypatch.setattr(
        type(materializer),
        "_canonical_payload_fingerprint",
        fail_payload,
    )
    pair_ids = (
        materializer.clean_positive_pair_ids[0],
        materializer.component_null_pair_ids[0],
    )
    batch = materializer.materialize(pair_ids, device="cpu")

    assert batch.pair_batch.pair_ids == pair_ids


def test_verify_unchanged_detects_gt_pair_and_footprint_tampering() -> None:
    prepared, catalog = _toy_inputs()
    materializer = build_paired_outcome_input_materializer(
        catalog,
        prepared,
    )
    sample_id = materializer.prepared_source_ids[0]
    materializer.gt_union_by_sample[sample_id].zero_()
    with pytest.raises(RuntimeError, match="inputs changed"):
        materializer.verify_unchanged()

    prepared, catalog = _toy_inputs()
    materializer = build_paired_outcome_input_materializer(
        catalog,
        prepared,
    )
    pair = materializer.pair_by_id[
        materializer.clean_positive_pair_ids[0]
    ]
    pair.feature.add_(1.0)
    with pytest.raises(RuntimeError, match="inputs changed"):
        materializer.verify_unchanged()

    prepared, catalog = _toy_inputs()
    materializer = build_paired_outcome_input_materializer(
        catalog,
        prepared,
    )
    pair = materializer.pair_by_id[
        materializer.component_null_pair_ids[0]
    ]
    coordinate = torch.nonzero(pair.occupancy_minus, as_tuple=False)[0]
    pair.occupancy_minus[tuple(coordinate)] = False
    with pytest.raises(RuntimeError, match="inputs changed"):
        materializer.verify_unchanged()


def test_identity_unknown_duplicate_and_empty_selections_are_rejected() -> None:
    prepared, catalog = _toy_inputs()
    materializer = build_paired_outcome_input_materializer(
        catalog,
        prepared,
    )
    identity_id = catalog.identity_null[0].pair_id
    clean_id = materializer.clean_positive_pair_ids[0]

    with pytest.raises(KeyError, match="unknown outcome pair IDs"):
        materializer.materialize((identity_id,), device="cpu")
    with pytest.raises(ValueError, match="unique"):
        materializer.materialize((clean_id, clean_id), device="cpu")
    with pytest.raises(ValueError, match="non-empty"):
        materializer.materialize((), device="cpu")


def test_non_dr_catalog_is_rejected_before_materialization() -> None:
    prepared, catalog = _toy_inputs()
    object.__setattr__(catalog, "split", "D_V")
    with pytest.raises(ValueError, match="only D_R"):
        build_paired_outcome_input_materializer(catalog, prepared)


def test_real_and_small_catalog_sizes_use_the_same_materializer() -> None:
    prepared, catalog = _toy_inputs()
    full = build_paired_outcome_input_materializer(catalog, prepared)
    small_catalog = _small_catalog(catalog)
    small = build_paired_outcome_input_materializer(
        small_catalog,
        prepared,
    )

    assert len(full.canonical_pair_ids) == (
        len(catalog.clean_positive) + len(catalog.component_null)
    )
    assert len(small.canonical_pair_ids) == 2
    assert len(small.clean_positive_pair_ids) == 1
    assert len(small.component_null_pair_ids) == 1
    batch = small.materialize(
        (
            small.clean_positive_pair_ids[0],
            small.component_null_pair_ids[0],
        ),
        device="cpu",
    )
    assert batch.pair_batch.pair_kinds == (
        "clean_positive",
        "component_null",
    )


def test_build_and_receipt_are_deterministic() -> None:
    prepared, catalog = _toy_inputs()
    first = build_paired_outcome_input_materializer(catalog, prepared)
    second = build_paired_outcome_input_materializer(catalog, prepared)

    assert first.materializer_fingerprint == second.materializer_fingerprint
    assert first.canonical_receipt() == second.canonical_receipt()
    first.verify_unchanged()
    second.verify_unchanged()


def test_incomplete_prepared_binding_and_missing_role_are_rejected() -> None:
    prepared, catalog = _toy_inputs()
    incomplete = prepare_training_catalog(prepared.sources[:1])
    with pytest.raises((ValueError, RuntimeError)):
        build_paired_outcome_input_materializer(catalog, incomplete)

    no_component = replace(
        catalog,
        component_null=(),
        catalog_fingerprint="",
    )
    no_component = replace(
        no_component,
        catalog_fingerprint=stable_fingerprint(
            no_component.canonical_payload()
        ),
    )
    with pytest.raises(ValueError, match="clean-positive and component-null"):
        build_paired_outcome_input_materializer(no_component, prepared)
