from __future__ import annotations

import pytest
import torch

from cure_lite.experiment.paired_catalog import build_pair_catalog
from cure_lite.experiment.paired_transition_inputs import (
    PAIRED_TRANSITION_INPUT_SCHEMA,
    build_paired_transition_input_materializer,
)
from cure_lite.experiment.training_pipeline import prepare_training_catalog
from cure_lite.paired_transition_types import AnchoredPairBatch
from cure_lite.toy import make_factual_miss_scene, make_two_target_scene
from tests.test_paired_catalog import (
    _PAIRED_PROTOCOL_FINGERPRINT,
    _cached_source,
    _false_positive_scene,
    _geometry,
    _manifest,
)


def _toy_inputs():
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
    return prepared, catalog


def _contains_tensor(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, dict):
        return any(_contains_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_tensor(item) for item in value)
    return False


def test_full_population_receipt_and_materialization_are_exact() -> None:
    prepared, catalog = _toy_inputs()
    materializer = build_paired_transition_input_materializer(
        catalog,
        prepared,
    )
    receipt = materializer.canonical_receipt()

    assert receipt["schema_version"] == PAIRED_TRANSITION_INPUT_SCHEMA
    assert receipt["split"] == "D_R"
    assert receipt["materializer_fingerprint"] == (
        materializer.materializer_fingerprint
    )
    assert receipt["counts"] == {
        "clean_pairs": len(catalog.clean_positive),
        "clean_pair_sources": len(
            {pair.sample_id for pair in catalog.clean_positive}
        ),
        "prepared_sources": len(prepared.source_ids),
    }
    assert {
        row["pair_id"] for row in receipt["all_clean_pair_inputs"]
    } == {pair.pair_id for pair in catalog.clean_positive}
    assert {
        row["sample_id"] for row in receipt["gt_unions"]
    } == set(prepared.source_ids)
    assert receipt["materialization_contract"] == {
        "selection_key": "pair_id",
        "selection_order_preserved": True,
        "all_clean_pairs_bound": True,
        "stack_function": "stack_anchored_pair_examples",
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

    pair_ids = materializer.canonical_pair_ids
    batch = materializer.materialize(pair_ids, device="cpu")
    assert isinstance(batch, AnchoredPairBatch)
    assert batch.pair_ids == pair_ids
    assert batch.feature.shape[0] == len(catalog.clean_positive)
    for index, pair_id in enumerate(pair_ids):
        pair = materializer.pair_by_id[pair_id]
        assert batch.sample_ids[index] == pair.sample_id
        assert torch.equal(batch.feature[index], pair.feature[0])
        assert torch.equal(
            batch.completion_plus[index],
            pair.completion_plus,
        )
        assert torch.equal(
            batch.gt_union[index],
            materializer.gt_union_by_sample[pair.sample_id],
        )


def test_batch_is_selected_only_by_pair_id_and_preserves_order() -> None:
    prepared, catalog = _toy_inputs()
    materializer = build_paired_transition_input_materializer(
        catalog,
        prepared,
    )
    pair_ids = tuple(reversed(materializer.canonical_pair_ids[:3]))
    batch = materializer.materialize(pair_ids, device=torch.device("cpu"))

    assert batch.pair_ids == pair_ids
    assert batch.sample_ids == tuple(
        materializer.pair_by_id[pair_id].sample_id
        for pair_id in pair_ids
    )
    with pytest.raises(KeyError, match="unknown clean pair IDs"):
        materializer.materialize(("f" * 64,), device="cpu")
    with pytest.raises(ValueError, match="unique"):
        materializer.materialize(
            (pair_ids[0], pair_ids[0]),
            device="cpu",
        )


def test_per_update_materialization_does_not_rehash_full_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, catalog = _toy_inputs()
    materializer = build_paired_transition_input_materializer(
        catalog,
        prepared,
    )

    def fail_rehash(self) -> None:
        del self
        raise AssertionError("full-population verification entered batch path")

    monkeypatch.setattr(
        type(materializer),
        "verify_unchanged",
        fail_rehash,
    )
    pair_ids = materializer.canonical_pair_ids[:2]
    batch = materializer.materialize(pair_ids, device="cpu")

    assert batch.pair_ids == pair_ids


def test_verify_unchanged_rejects_gt_and_pair_tensor_tampering() -> None:
    prepared, catalog = _toy_inputs()
    materializer = build_paired_transition_input_materializer(
        catalog,
        prepared,
    )
    sample_id = materializer.prepared_source_ids[0]
    materializer.gt_union_by_sample[sample_id].zero_()
    with pytest.raises(RuntimeError, match="inputs changed"):
        materializer.verify_unchanged()

    prepared, catalog = _toy_inputs()
    materializer = build_paired_transition_input_materializer(
        catalog,
        prepared,
    )
    pair = materializer.pair_by_id[materializer.canonical_pair_ids[0]]
    pair.feature.add_(1.0)
    with pytest.raises(RuntimeError, match="inputs changed"):
        materializer.verify_unchanged()


def test_non_dr_catalog_is_rejected_before_materialization() -> None:
    prepared, catalog = _toy_inputs()
    object.__setattr__(catalog, "split", "D_V")
    with pytest.raises(ValueError, match="only D_R"):
        build_paired_transition_input_materializer(catalog, prepared)


def test_build_and_receipt_are_deterministic() -> None:
    prepared, catalog = _toy_inputs()
    first = build_paired_transition_input_materializer(catalog, prepared)
    second = build_paired_transition_input_materializer(catalog, prepared)

    assert first.materializer_fingerprint == second.materializer_fingerprint
    assert first.canonical_receipt() == second.canonical_receipt()
    first.verify_unchanged()
    second.verify_unchanged()


def test_incomplete_or_inconsistent_prepared_binding_is_rejected() -> None:
    prepared, catalog = _toy_inputs()
    incomplete = prepare_training_catalog(prepared.sources[:1])
    with pytest.raises((ValueError, RuntimeError)):
        build_paired_transition_input_materializer(catalog, incomplete)
