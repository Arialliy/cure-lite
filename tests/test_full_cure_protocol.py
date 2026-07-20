from __future__ import annotations

from dataclasses import replace

import pytest

from cure_lite.cure import CUREProtocol, CUREResidualConfig
from cure_lite.provenance import BaseCheckpointSelection
from cure_lite.splits import SplitManifest, SplitRecord


def _manifest(*, source_image: str = "source.png") -> SplitManifest:
    return SplitManifest(
        dataset="toy",
        records=(
            SplitRecord("base-fit", "D_B", "base-fit-group", "base-fit.png"),
            SplitRecord("base-select", "D_B", "base-select-group", "base-select.png"),
            SplitRecord("source", "D_R", "source-group", source_image),
            SplitRecord("validation", "D_V", "validation-group", "validation.png"),
            SplitRecord("test", "D_T", "test-group", "test.png"),
        ),
    )


def _protocol(
    manifest: SplitManifest,
    *,
    occupancy_threshold: float = 0.5,
    base_state_fingerprint: str = "0" * 64,
) -> CUREProtocol:
    return CUREProtocol.from_manifest(
        manifest,
        base_fingerprint="base-checkpoint",
        adapter_fingerprint="adapter",
        base_state_fingerprint=base_state_fingerprint,
        preprocessing_fingerprint="preprocessing",
        residual_config=CUREResidualConfig(
            feature_channels=3,
            width=8,
            groups=4,
            occupancy_threshold=occupancy_threshold,
        ),
        base_checkpoint_selection=BaseCheckpointSelection.from_manifest(
            manifest,
            fit_sample_ids=("base-fit",),
            select_sample_ids=("base-select",),
        ),
    )


def test_protocol_fingerprint_changes_with_any_method_configuration() -> None:
    manifest = _manifest()
    protocol = _protocol(manifest)
    changed = _protocol(manifest, occupancy_threshold=0.6)
    assert changed.fingerprint != protocol.fingerprint
    changed_base_state = _protocol(manifest, base_state_fingerprint="1" * 64)
    assert changed_base_state.fingerprint != protocol.fingerprint
    with pytest.raises(ValueError, match="content differs"):
        replace(
            protocol,
            residual_config=replace(
                protocol.residual_config,
                occupancy_threshold=0.6,
            ),
        )


def test_protocol_rejects_manifest_payload_substitution() -> None:
    protocol = _protocol(_manifest())
    with pytest.raises(ValueError, match="differs from the frozen"):
        protocol.validate_manifest(_manifest(source_image="changed-source.png"))
