from __future__ import annotations

import pytest
import torch

from cure_lite.coverage_state_binary_flip_antisymmetric import (
    CURELiteBinaryFlipAntisymmetricLevelSet,
    CoverageStateBinaryFlipAntisymmetricConfig,
)
from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
    CoverageStatePhaseAlignedEvidenceTransportConfig,
)
from cure_lite.experiment.coverage_state_paet_certificate import (
    COVERAGE_STATE_PAET_CERTIFICATE_DEFAULT_PAIR_BATCH_SIZE,
    COVERAGE_STATE_PAET_CERTIFICATE_PAIR_COUNT,
    audit_coverage_state_paet_pair_certificate,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_bounded_training_scalar_cache,
)


def _paet_model(cache):
    return CURELitePhaseAlignedEvidenceTransportLevelSet(
        CoverageStatePhaseAlignedEvidenceTransportConfig(
            feature_channels=(
                cache.clean_positive_records[0].record.feature.shape[1]
            ),
            feature_stride=cache.raw_catalog.feature_stride,
            width=4,
        )
    )


def test_paet_certificate_reports_all_pairs_without_gating() -> None:
    cache = make_bounded_training_scalar_cache()
    model = _paet_model(cache).eval()
    state_before = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    cpu_rng_before = torch.random.get_rng_state().clone()

    receipt = audit_coverage_state_paet_pair_certificate(
        model,
        cache,
        device="cpu",
        pair_batch_size=(
            COVERAGE_STATE_PAET_CERTIFICATE_DEFAULT_PAIR_BATCH_SIZE
        ),
    )

    assert receipt.integrity_passed
    assert len(receipt.pair_certificates) == (
        COVERAGE_STATE_PAET_CERTIFICATE_PAIR_COUNT
    )
    assert receipt.model_forward_invocations == 8
    assert receipt.pair_batch_size == 4
    assert not receipt.all_pairs_passed
    assert receipt.total_raw_sign_error_pixels > 0
    assert receipt.canonical_payload()["diagnostic_summary"][
        "pair_result_is_bounded_gate"
    ] is False
    assert torch.equal(cpu_rng_before, torch.random.get_rng_state())
    for name, value in model.state_dict().items():
        assert torch.equal(value, state_before[name])
    receipt.verify()


def test_paet_certificate_requires_exact_paet_model() -> None:
    cache = make_bounded_training_scalar_cache()
    model = CURELiteBinaryFlipAntisymmetricLevelSet(
        CoverageStateBinaryFlipAntisymmetricConfig(
            feature_channels=(
                cache.clean_positive_records[0].record.feature.shape[1]
            ),
            feature_stride=cache.raw_catalog.feature_stride,
            width=4,
        )
    )

    with pytest.raises(
        TypeError,
        match="CURELitePhaseAlignedEvidenceTransportLevelSet",
    ):
        audit_coverage_state_paet_pair_certificate(
            model,  # type: ignore[arg-type]
            cache,
            device="cpu",
        )


def test_paet_certificate_rejects_invalid_batch_size() -> None:
    cache = make_bounded_training_scalar_cache()
    model = _paet_model(cache)

    with pytest.raises(ValueError, match=r"integer in \[1, 32\]"):
        audit_coverage_state_paet_pair_certificate(
            model,
            cache,
            device="cpu",
            pair_batch_size=0,
        )
