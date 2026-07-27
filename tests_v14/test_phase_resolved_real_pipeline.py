from __future__ import annotations

import json
from copy import deepcopy
from math import log
from pathlib import Path

import pytest
import torch

from cure_lite.data import ManifestImageDataset, PreprocessConfig
from cure_lite.experiment.cache_pipeline import load_d_r_cache_bundle
from cure_lite.experiment.geometry_catalog_protocol import (
    load_geometry_catalog_protocol,
)
from cure_lite.experiment.phase_resolved_real_training import (
    PFCRRealFormalTrainingConfig,
    PFCRRealPreflightConfig,
    _PFCRForwardLedger,
    _validate_final_adam,
    pfcr_real_formal_schedule_payload,
)
from cure_lite.phase_resolved_real_cache import adapt_pfcr_d_r_cache
from cure_lite.phase_resolved_real_states import (
    build_pfcr_epoch_pools,
    build_pfcr_real_state_catalog,
    load_pfcr_lineage_allowlist,
)
from cure_lite.phase_resolved_relation_decoder import (
    CURELitePhaseResolvedRelationDecoder,
    PhaseResolvedRelationDecoderConfig,
)
from cure_lite.splits import load_and_validate_manifest
from cure_lite.train.phase_resolved_relation_step import (
    phase_resolved_real_train_step,
)
from cure_lite.train.step import BranchBatch


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT / "protocols/IRSTD-1K/stage_a_seed42/manifest.json"
)
STATE_INDEX = (
    ROOT
    / "runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3"
    / "d_r/state_cache/index.json"
)
GEOMETRY_CONFIG = (
    ROOT / "protocols/IRSTD-1K/geometry_safe_p0_v2/config.json"
)
P0_A1 = (
    ROOT
    / "runs/irstd1k_stage_a_seed42"
    / "cure_lite_geometry_safe_p0_v2_r1"
    / "receipts/p0_a1_population_eligibility.json"
)


@pytest.fixture(scope="module")
def real_inputs():
    required = (MANIFEST, STATE_INDEX, GEOMETRY_CONFIG, P0_A1)
    if any(not path.is_file() for path in required):
        pytest.skip("local frozen IRSTD-1K Stage-A inputs are unavailable")
    manifest = load_and_validate_manifest(MANIFEST)
    state_index = json.loads(STATE_INDEX.read_text(encoding="utf-8"))
    preprocess = PreprocessConfig.from_fingerprint_payload(
        state_index["preprocessing"]
    )
    dataset = ManifestImageDataset(
        manifest,
        "D_R",
        preprocess,
        manifest_path=MANIFEST,
    )
    protocol = load_geometry_catalog_protocol(GEOMETRY_CONFIG)
    bundle = load_d_r_cache_bundle(
        STATE_INDEX,
        dataset,
        expected_base_fingerprint=(
            protocol.input_binding.base_fingerprint
        ),
    )
    cache = adapt_pfcr_d_r_cache(bundle)
    allowlist = load_pfcr_lineage_allowlist(P0_A1)
    catalog = build_pfcr_real_state_catalog(cache, allowlist)
    return cache, catalog


def test_real_cache_derives_native_pfcr_dimensions(real_inputs) -> None:
    cache, _ = real_inputs
    contract = cache.contract

    assert contract.dataset == "IRSTD-1K"
    assert contract.sample_count == 160
    assert contract.feature_channels == 64
    assert contract.feature_shape == (64, 64)
    assert contract.output_shape == (256, 256)
    assert contract.feature_stride == 4
    assert contract.occupancy_threshold == 0.72
    assert contract.canonical_payload()["base_forward_executed"] is False

    decoder = CURELitePhaseResolvedRelationDecoder(
        contract.decoder_config()
    )
    first = cache.bundle.rows[0]
    occupancy = first.state.occupancy[None, None]
    with torch.no_grad():
        logits = decoder(first.base_output.feature, occupancy)
    assert logits.shape == (1, 1, 256, 256)


def test_real_state_catalog_is_exactly_lineage_safe_206(
    real_inputs,
) -> None:
    _, catalog = real_inputs

    assert catalog.factual_target_count == 32
    assert catalog.factual_source_count == 24
    assert catalog.factual_no_miss_source_count == 135
    assert catalog.legal_target_count == 206
    assert catalog.legal_source_count == 149
    assert catalog.allowlist.excluded_legal_identities == (
        ("XDU486", 1, 1),
        ("XDU526", 1, 1),
        ("XDU965", 1, 1),
    )
    selected = {
        tuple(value)
        for value in catalog.canonical_payload()[
            "selected_legal_identities"
        ]
    }
    assert not selected & set(
        catalog.allowlist.excluded_legal_identities
    )

    pools = build_pfcr_epoch_pools(
        catalog,
        epoch=0,
        global_seed=42,
    )
    assert len(pools.factual_miss) == 24
    assert len(pools.factual_no_miss) == 135
    assert len(pools.synthetic) == 149


def _branch(
    *,
    positive: bool,
    feature: torch.Tensor,
) -> BranchBatch:
    occupancy = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    target = torch.zeros(1, 1, 8, 8, dtype=torch.float32)
    if positive:
        target[0, 0, 3, 3] = 1.0
    valid = torch.ones_like(occupancy)
    return BranchBatch(feature, occupancy, target, valid)


def test_real_step_fuses_three_branches_and_updates_only_decoder() -> None:
    torch.manual_seed(7)
    decoder = CURELitePhaseResolvedRelationDecoder(
        PhaseResolvedRelationDecoderConfig(
            feature_channels=4,
            feature_stride=2,
            relation_dim=4,
        )
    )
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
    feature = torch.randn(1, 4, 4, 4)
    feature_before = feature.clone()
    state_before = {
        name: value.detach().clone()
        for name, value in decoder.state_dict().items()
    }
    batches = {
        "factual_miss": _branch(positive=True, feature=feature),
        "factual_no_miss": _branch(
            positive=False,
            feature=feature,
        ),
        "synthetic": _branch(positive=True, feature=feature),
    }

    logs = phase_resolved_real_train_step(
        decoder,
        optimizer,
        batches,
        logit_margin=log(0.951 / 0.049),
    )

    assert logs["decoder_forward_calls"] == 1
    assert logs["total_states"] == 3
    assert logs["gradient_l2_norm"] > 0.0
    assert torch.equal(feature, feature_before)
    assert any(
        not torch.equal(state_before[name], value)
        for name, value in decoder.state_dict().items()
    )
    assert all(
        torch.isfinite(value).all()
        for value in decoder.state_dict().values()
    )


def test_decoder_is_invariant_to_positive_feature_rescaling() -> None:
    torch.manual_seed(9)
    decoder = CURELitePhaseResolvedRelationDecoder(
        PhaseResolvedRelationDecoderConfig(
            feature_channels=4,
            feature_stride=2,
            relation_dim=4,
        )
    )
    feature = torch.randn(2, 4, 4, 4)
    occupancy = torch.zeros(2, 1, 8, 8, dtype=torch.bool)

    with torch.no_grad():
        reference = decoder(feature, occupancy)
        scaled = decoder(feature * 1000.0, occupancy)

    assert torch.allclose(reference, scaled, atol=1.0e-6, rtol=0.0)


def test_formal_fast_path_is_one_update_equivalent() -> None:
    torch.manual_seed(17)
    audited = CURELitePhaseResolvedRelationDecoder(
        PhaseResolvedRelationDecoderConfig(
            feature_channels=4,
            feature_stride=2,
            relation_dim=8,
        )
    )
    fast = deepcopy(audited)
    audited_optimizer = torch.optim.Adam(
        audited.parameters(),
        lr=0.001,
    )
    fast_optimizer = torch.optim.Adam(
        fast.parameters(),
        lr=0.001,
    )
    ledger = _PFCRForwardLedger(fast)
    feature = torch.randn(1, 4, 4, 4)
    batches = {
        "factual_miss": _branch(
            positive=True,
            feature=feature,
        ),
        "factual_no_miss": _branch(
            positive=False,
            feature=feature,
        ),
        "synthetic": _branch(
            positive=True,
            feature=feature,
        ),
    }
    margin = log(0.951 / 0.049)

    audited_logs = phase_resolved_real_train_step(
        audited,
        audited_optimizer,
        batches,
        logit_margin=margin,
        audit=True,
    )
    fast_logs = phase_resolved_real_train_step(
        fast,
        fast_optimizer,
        batches,
        logit_margin=margin,
        audit=False,
    )
    ledger.close()

    assert ledger.calls == 1
    assert ledger.states == 3
    optimizer_fingerprint, minimum_step, maximum_step = (
        _validate_final_adam(
            fast,
            fast_optimizer,
            expected_steps=1,
            learning_rate=0.001,
            weight_decay=0.0,
        )
    )
    assert len(optimizer_fingerprint) == 64
    assert minimum_step == maximum_step == 1
    assert audited_logs["total"] == pytest.approx(
        fast_logs["total"],
        rel=0.0,
        abs=1.0e-7,
    )
    assert audited_logs["gradient_l2_norm"] == pytest.approx(
        fast_logs["gradient_l2_norm"],
        rel=0.0,
        abs=1.0e-7,
    )
    assert all(
        torch.equal(audited.state_dict()[name], value)
        for name, value in fast.state_dict().items()
    )


def test_real_step_requires_all_three_independent_branches() -> None:
    decoder = CURELitePhaseResolvedRelationDecoder(
        PhaseResolvedRelationDecoderConfig(
            feature_channels=4,
            feature_stride=2,
            relation_dim=4,
        )
    )
    optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
    feature = torch.randn(1, 4, 4, 4)

    with pytest.raises(ValueError, match="requires factual_miss"):
        phase_resolved_real_train_step(
            decoder,
            optimizer,
            {
                "factual_miss": _branch(
                    positive=True,
                    feature=feature,
                )
            },
            logit_margin=log(0.951 / 0.049),
        )


def test_formal_schedule_is_exactly_800_by_40_without_continuation() -> None:
    config = PFCRRealFormalTrainingConfig(seed=42)

    assert config.epochs == 800
    assert config.steps_per_epoch == 40
    assert config.optimizer_updates == 32_000
    assert config.canonical_payload()["continuation_supported"] is False
    with pytest.raises(ValueError, match="fixes epochs"):
        PFCRRealFormalTrainingConfig(seed=42, epochs=1)
    with pytest.raises(ValueError, match="fixes update_count"):
        PFCRRealPreflightConfig(seed=42, update_count=9)


def test_formal_schedule_binds_actual_target_and_source_exposure(
    real_inputs,
) -> None:
    _, catalog = real_inputs
    payload = pfcr_real_formal_schedule_payload(
        catalog,
        PFCRRealFormalTrainingConfig(seed=42),
    )
    exposure = payload["exposure"]
    branches = exposure["branches"]

    assert exposure["optimizer_updates"] == 32_000
    assert len(exposure["combined_sequence_fingerprint"]) == 64
    assert {
        branch: fields["state_exposure_total"]
        for branch, fields in branches.items()
    } == {
        "factual_miss": 128_000,
        "factual_no_miss": 128_000,
        "synthetic": 128_000,
    }
    assert {
        branch: fields["state_population"]
        for branch, fields in branches.items()
    } == {
        "factual_miss": 32,
        "factual_no_miss": 135,
        "synthetic": 206,
    }
    assert all(
        fields["zero_exposure_states"] == 0
        for fields in branches.values()
    )
