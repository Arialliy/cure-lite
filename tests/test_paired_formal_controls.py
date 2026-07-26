from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from pathlib import Path

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.experiment.paired_catalog import build_pair_catalog
from cure_lite.experiment.paired_formal_controls import (
    PAIRED_FORMAL_CONTROL_PROVIDER_SCHEMA,
    FrozenControlPreflightFingerprints,
    build_paired_formal_control_provider,
    load_frozen_control_preflight_fingerprints,
)
from cure_lite.experiment.training_pipeline import prepare_training_catalog
from cure_lite.paired_control_inputs import (
    build_dct_coordinate_basis,
    build_target_permutation,
)
from cure_lite.paired_types import stack_pair_examples
from cure_lite.toy import (
    make_factual_miss_scene,
    make_two_target_scene,
)
from cure_lite.train.paired_control_step import CONTROL_KINDS
from tests.test_paired_catalog import (
    _PAIRED_PROTOCOL_FINGERPRINT,
    _cached_source,
    _false_positive_scene,
    _geometry,
    _manifest,
)
from tools import run_paired_bounded_learnability as bounded_runner


_ROOT = Path(__file__).resolve().parents[1]
_REAL_CONFIG = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "paired_bounded_learnability_v1"
    / "config.json"
)
_REAL_CONTROL_PREFLIGHT = (
    _ROOT
    / "runs"
    / "irstd1k_stage_a_seed42"
    / "cure_lite_paired_control_preflight_v1_r1"
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
    return sources, prepared, catalog


def _preflight(catalog) -> FrozenControlPreflightFingerprints:
    pair = catalog.clean_positive[0]
    basis = build_dct_coordinate_basis(
        channels=int(pair.feature.shape[1]),
        height=int(pair.feature.shape[2]),
        width=int(pair.feature.shape[3]),
        dtype=pair.feature.dtype,
    )
    plan = build_target_permutation(catalog.clean_positive)
    assert plan.ready
    return FrozenControlPreflightFingerprints(
        complete_fingerprint="a" * 64,
        pair_catalog_fingerprint=catalog.catalog_fingerprint,
        dct_basis_fingerprint=basis.basis_fingerprint,
        target_permutation_plan_fingerprint=plan.plan_fingerprint,
        target_assignment_fingerprint=stable_fingerprint(
            [
                assignment.canonical_payload()
                for assignment in plan.assignments
            ]
        ),
    )


@pytest.fixture(scope="module")
def toy_provider():
    _, prepared, catalog = _toy_inputs()
    provider = build_paired_formal_control_provider(
        catalog,
        prepared,
        _preflight(catalog),
    )
    return prepared, catalog, provider


def _source_disjoint_pairs(catalog):
    return next(
        (first, second)
        for first, second in combinations(catalog.clean_positive, 2)
        if first.sample_id != second.sample_id
    )


def _assert_equal_kwargs(
    first: dict[str, object],
    second: dict[str, object],
) -> None:
    assert set(first) == set(second)
    for key in first:
        left = first[key]
        right = second[key]
        if isinstance(left, torch.Tensor):
            assert isinstance(right, torch.Tensor)
            assert torch.equal(left, right)
        else:
            assert left is right


def test_provider_receipt_seals_all_pairs_sources_and_control_only_scope(
    toy_provider,
) -> None:
    prepared, catalog, provider = toy_provider
    receipt = provider.canonical_receipt()
    assert receipt["schema_version"] == PAIRED_FORMAL_CONTROL_PROVIDER_SCHEMA
    assert receipt["provider_fingerprint"] == provider.provider_fingerprint
    assert receipt["counts"] == {
        "clean_pairs": len(catalog.clean_positive),
        "prepared_sources": len(prepared.entries),
        "permutation_assignments": len(catalog.clean_positive),
    }
    assert {
        row["sample_id"] for row in receipt["gt_unions"]
    } == set(prepared.source_ids)
    assert {
        row["pair_id"] for row in receipt["all_pair_inputs"]
    } == {pair.pair_id for pair in catalog.clean_positive}
    assert receipt["target_permutation"]["source_disjoint"] is True
    assert receipt["target_permutation"]["fixed_point_free"] is True
    assert (
        receipt["target_permutation"]["full_recipient_and_donor_marginals"]
        is True
    )
    assert receipt["selection_contract"] == {
        "selected_only_by_pair_ids": True,
        "epoch_used_for_selection": False,
        "step_used_for_selection": False,
        "method_changes_schedule": False,
        "seed_specific_data_owned_by_provider": False,
        "raw_tensor_payloads_written": False,
    }
    assert receipt["D_V_accessed"] is False
    assert receipt["D_T_accessed"] is False
    assert receipt["training_performed"] is False
    assert "paired_difference" not in receipt["control_kwarg_keys"]
    with pytest.raises(ValueError, match="unknown formal matched control"):
        pairs = _source_disjoint_pairs(catalog)
        provider(
            control_kind="paired_difference",
            pairs=pairs,
            pair_batch=stack_pair_examples(pairs, device="cpu"),
            epoch=0,
            step=0,
            device=torch.device("cpu"),
        )


def test_provider_returns_exact_keys_and_is_epoch_step_independent(
    toy_provider,
) -> None:
    _, catalog, provider = toy_provider
    pairs = _source_disjoint_pairs(catalog)
    batch = stack_pair_examples(pairs, device="cpu")
    expected_keys = {
        "independent_endpoint": {
            "gt_union",
            "completion_plus",
            "completion_minus",
        },
        "after_only": {"gt_union"},
        "zero_feature": set(),
        "coordinate_basis": {"coordinate_basis"},
        "feature_only": set(),
        "target_permutation": {"permuted_label_increment"},
        "plus_detach": set(),
        "minus_detach": set(),
    }
    for control_kind in CONTROL_KINDS:
        first = dict(
            provider(
                control_kind=control_kind,
                pairs=pairs,
                pair_batch=batch,
                epoch=0,
                step=0,
                device=torch.device("cpu"),
            )
        )
        last = dict(
            provider(
                control_kind=control_kind,
                pairs=pairs,
                pair_batch=batch,
                epoch=799,
                step=39,
                device=torch.device("cpu"),
            )
        )
        assert set(first) == expected_keys[control_kind]
        _assert_equal_kwargs(first, last)

    permutation = provider(
        control_kind="target_permutation",
        pairs=pairs,
        pair_batch=batch,
        epoch=4,
        step=17,
        device=torch.device("cpu"),
    )["permuted_label_increment"]
    for index, pair in enumerate(pairs):
        assignment = provider.assignment_by_recipient[pair.pair_id]
        donor = provider.pair_by_id[assignment["donor_pair_id"]]
        assert pair.sample_id != donor.sample_id
        assert torch.equal(
            permutation[index],
            donor.clean_increment.to(torch.float32),
        )


def test_provider_rejects_wrong_runtime_shape_dtype_device_and_identity(
    toy_provider,
) -> None:
    _, catalog, provider = toy_provider
    pairs = _source_disjoint_pairs(catalog)
    batch = stack_pair_examples(pairs, device="cpu")
    common = {
        "control_kind": "after_only",
        "pairs": pairs,
        "epoch": 0,
        "step": 0,
        "device": torch.device("cpu"),
    }
    with pytest.raises(ValueError, match="feature shape"):
        provider(
            **common,
            pair_batch=replace(
                batch,
                feature=batch.feature[:, :, :1, :],
            ),
        )
    with pytest.raises(TypeError, match="feature dtype"):
        provider(
            **common,
            pair_batch=replace(batch, feature=batch.feature.to(torch.float64)),
        )
    with pytest.raises(ValueError, match="requested device"):
        provider(
            **{**common, "device": torch.device("meta")},
            pair_batch=batch,
        )
    with pytest.raises(ValueError, match="identities differ"):
        provider(
            **{**common, "pairs": (pairs[1], pairs[0])},
            pair_batch=batch,
        )


def test_builder_rejects_missing_source_and_wrong_preflight_fingerprints() -> None:
    sources, prepared, catalog = _toy_inputs()
    preflight = _preflight(catalog)
    with pytest.raises(RuntimeError, match="pair catalog differs"):
        build_paired_formal_control_provider(
            catalog,
            prepared,
            replace(preflight, pair_catalog_fingerprint="f" * 64),
        )
    with pytest.raises(RuntimeError, match="DCT basis differs"):
        build_paired_formal_control_provider(
            catalog,
            prepared,
            replace(preflight, dct_basis_fingerprint="e" * 64),
        )
    with pytest.raises(RuntimeError, match="target permutation"):
        build_paired_formal_control_provider(
            catalog,
            prepared,
            replace(
                preflight,
                target_permutation_plan_fingerprint="d" * 64,
            ),
        )
    with pytest.raises(RuntimeError, match="assignments differ"):
        build_paired_formal_control_provider(
            catalog,
            prepared,
            replace(preflight, target_assignment_fingerprint="c" * 64),
        )

    incomplete_prepared = prepare_training_catalog(sources[:2])
    assert set(incomplete_prepared.source_ids) != set(prepared.source_ids)
    with pytest.raises(ValueError, match="missing a clean-pair source"):
        build_paired_formal_control_provider(
            catalog,
            incomplete_prepared,
            preflight,
        )


def test_provider_detects_post_build_tensor_mutation(toy_provider) -> None:
    _, _, provider = toy_provider
    sample_id = next(iter(provider.gt_union_by_sample))
    tensor = provider.gt_union_by_sample[sample_id]
    original = tensor.clone()
    try:
        tensor.logical_not_()
        with pytest.raises(RuntimeError, match="inputs changed"):
            provider.verify_unchanged()
    finally:
        tensor.copy_(original)
    provider.verify_unchanged()
    with pytest.raises(TypeError):
        provider.gt_union_by_sample["new-source"] = original


@pytest.mark.skipif(
    not _REAL_CONFIG.is_file() or not _REAL_CONTROL_PREFLIGHT.is_dir(),
    reason="local authoritative D_R artifacts are unavailable",
)
def test_real_206_pair_static_provider_closes_authoritative_preflight() -> None:
    config = bounded_runner._load_config(_REAL_CONFIG.resolve())
    pair_catalog, prepared, bundle, _ = bounded_runner._load_real_catalog(
        config
    )
    preflight = load_frozen_control_preflight_fingerprints(
        _REAL_CONTROL_PREFLIGHT,
        pair_catalog,
    )
    provider = build_paired_formal_control_provider(
        pair_catalog,
        prepared,
        preflight,
    )
    receipt = provider.canonical_receipt()
    assert receipt["counts"] == {
        "clean_pairs": 206,
        "prepared_sources": 160,
        "permutation_assignments": 206,
    }
    assert provider.pair_catalog_fingerprint == (
        "4886e52d2cfb3392d0f4fdda376159d6e7f694fd449dc809cf8874793febde76"
    )
    assert (
        provider.coordinate_basis.basis_fingerprint
        == "24d9e9bfe99d864c49167b3b9d088847eb2802e060f7148a5596a2d3970beb39"
    )
    assert (
        provider.preflight.target_permutation_plan_fingerprint
        == "4932cc226753700916feb712ff99b1192f7d38927476e8041b6adc7fcd915798"
    )
    provider.verify_unchanged()
    bundle.verify_unchanged()
