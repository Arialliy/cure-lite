from __future__ import annotations

import json
from pathlib import Path

import pytest

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.factorized_config import SVEF_RESIZE_POLICY
from cure_lite.recoverable_factorized_config import (
    RECOVERABLE_BACKWARD_SURROGATE_POLICY,
    RECOVERABLE_FORWARD_EVIDENCE_TRANSFORM,
    RECOVERABLE_ZERO_BOUNDARY_POLICY,
    RecoverableFactorizedDecoderConfig,
)
from cure_lite.recoverable_factorized_decoder import (
    CURELiteRecoverableFactorizedDecoder,
)


_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "polarity_recoverable_subpixel_vacancy_evidence_factorization_v6"
)


def test_reference_config_preserves_v4_topology_and_parameter_count() -> None:
    config = RecoverableFactorizedDecoderConfig(
        feature_channels=64,
        feature_stride=4,
    )

    assert config.width == 32
    assert config.groups == 8
    assert config.trunk_residual_scale == 0.5
    assert config.baseline_probability == 0.1
    assert config.vacancy_kernel_size == 3
    assert (
        config.forward_evidence_transform
        == RECOVERABLE_FORWARD_EVIDENCE_TRANSFORM
    )
    assert (
        config.backward_surrogate_policy
        == RECOVERABLE_BACKWARD_SURROGATE_POLICY
    )
    assert config.zero_boundary_policy == RECOVERABLE_ZERO_BOUNDARY_POLICY
    assert config.resize_policy == SVEF_RESIZE_POLICY
    assert config.phase_channels == 16
    assert config.expected_parameter_count == 4385
    assert config.to_v4_topology_config().expected_parameter_count == 4385

    decoder = CURELiteRecoverableFactorizedDecoder(config)
    assert sum(parameter.numel() for parameter in decoder.parameters()) == 4385


@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_channels": True, "feature_stride": 4},
        {"feature_channels": 0, "feature_stride": 4},
        {"feature_channels": 4, "feature_stride": False},
        {"feature_channels": 4, "feature_stride": 0},
        {"feature_channels": 4, "feature_stride": 2, "width": 16},
        {"feature_channels": 4, "feature_stride": 2, "groups": 4},
        {
            "feature_channels": 4,
            "feature_stride": 2,
            "trunk_residual_scale": 1.0,
        },
        {
            "feature_channels": 4,
            "feature_stride": 2,
            "baseline_probability": 0.2,
        },
        {
            "feature_channels": 4,
            "feature_stride": 2,
            "vacancy_kernel_size": 5,
        },
        {
            "feature_channels": 4,
            "feature_stride": 2,
            "forward_evidence_transform": "other",
        },
        {
            "feature_channels": 4,
            "feature_stride": 2,
            "backward_surrogate_policy": "other",
        },
        {
            "feature_channels": 4,
            "feature_stride": 2,
            "zero_boundary_policy": "other",
        },
        {
            "feature_channels": 4,
            "feature_stride": 2,
            "resize_policy": "other",
        },
    ],
)
def test_config_rejects_any_operator_or_topology_variant(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        RecoverableFactorizedDecoderConfig(**kwargs)


def test_decoder_constructor_requires_one_truthful_v6_config() -> None:
    config = RecoverableFactorizedDecoderConfig(
        feature_channels=3,
        feature_stride=2,
    )
    with pytest.raises(ValueError):
        CURELiteRecoverableFactorizedDecoder(
            config,
            feature_channels=3,
        )
    with pytest.raises(TypeError):
        CURELiteRecoverableFactorizedDecoder(feature_channels=3)
    with pytest.raises(TypeError):
        CURELiteRecoverableFactorizedDecoder(3)  # type: ignore[arg-type]


def test_frozen_v6_proposal_and_toy_config_fingerprints() -> None:
    proposal_path = _PROTOCOL / "proposal_receipt.json"
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal_unsigned = dict(proposal)
    proposal_fingerprint = proposal_unsigned.pop("proposal_fingerprint")
    assert proposal_fingerprint == stable_fingerprint(proposal_unsigned)

    config_path = _PROTOCOL / "toy_config.json"
    toy = json.loads(config_path.read_text(encoding="utf-8"))
    toy_unsigned = dict(toy)
    config_fingerprint = toy_unsigned.pop("config_fingerprint")
    assert config_fingerprint == stable_fingerprint(toy_unsigned)
    assert toy["proposal_binding"]["file_sha256"] == file_sha256(
        proposal_path
    )
    assert toy["proposal_binding"]["proposal_fingerprint"] == (
        proposal_fingerprint
    )

    design = proposal["design_document"]
    assert file_sha256(_ROOT / design["repo_path"]) == design["file_sha256"]
    closure = proposal["predecessor_v5"]
    assert file_sha256(_ROOT / closure["closure_repo_path"]) == (
        closure["closure_file_sha256"]
    )
