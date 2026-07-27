from __future__ import annotations

import os
from pathlib import Path
import shutil

import pytest

from cure_lite.coverage_state_observability import (
    CoverageStateObservabilityDecision,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    COVERAGE_STATE_OBSERVABILITY_CONFIG_FILE_SHA256,
    bind_coverage_state_real_dr_sources,
    build_coverage_state_real_dr_inputs,
)


_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST = (
    _ROOT / "protocols" / "IRSTD-1K" / "stage_a_seed42" / "manifest.json"
)
_STATE_INDEX = (
    _ROOT
    / "runs"
    / "irstd1k_stage_a_seed42"
    / "cure_lite_stage_a_fx_v3"
    / "d_r"
    / "state_cache"
    / "index.json"
)
_GEOMETRY_CONFIG = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "geometry_safe_p0_v2"
    / "config.json"
)
_GEOMETRY_RECEIPT = (
    _ROOT
    / "runs"
    / "irstd1k_stage_a_seed42"
    / "cure_lite_geometry_safe_p0_v2_r1"
    / "receipts"
    / "geometry_catalog.json"
)
_OBSERVABILITY_CONFIG = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "coverage_state_observability_v1"
    / "config.json"
)


def _paths() -> dict[str, Path]:
    return {
        "manifest_path": _MANIFEST,
        "state_index_path": _STATE_INDEX,
        "geometry_config_path": _GEOMETRY_CONFIG,
        "geometry_receipt_path": _GEOMETRY_RECEIPT,
        "observability_config_path": _OBSERVABILITY_CONFIG,
    }


def test_real_dr_source_binding_closes_the_frozen_create_only_contract() -> None:
    binding, protocol, geometry, preprocess = (
        bind_coverage_state_real_dr_sources(**_paths())
    )
    assert binding.split == protocol.split == "D_R"
    assert binding.dataset == protocol.dataset == "IRSTD-1K"
    assert binding.observability_config_file_sha256 == (
        COVERAGE_STATE_OBSERVABILITY_CONFIG_FILE_SHA256
    )
    assert binding.geometry_protocol_config_fingerprint == (
        geometry.fingerprint
    )
    assert binding.canonical_payload()["execution_policy"] == {
        "create_only": True,
        "training_performed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    assert len(preprocess.fingerprint_payload()) > 0
    assert len(binding.binding_fingerprint) == 64
    binding.verify_unchanged()


def test_real_dr_source_binding_rejects_state_index_drift(
    tmp_path: Path,
) -> None:
    changed = tmp_path / "index.json"
    shutil.copyfile(_STATE_INDEX, changed)
    with changed.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    paths = _paths()
    paths["state_index_path"] = changed
    with pytest.raises(
        RuntimeError,
        match="state_index_sha256 differs",
    ):
        bind_coverage_state_real_dr_sources(**paths)


def test_real_dr_source_binding_detects_post_bind_receipt_drift(
    tmp_path: Path,
) -> None:
    copied: dict[str, Path] = {}
    for name, source in _paths().items():
        target = tmp_path / f"{name}.json"
        shutil.copyfile(source, target)
        copied[name] = target
    binding, _, _, _ = bind_coverage_state_real_dr_sources(**copied)
    with copied["geometry_receipt_path"].open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write("\n")
    with pytest.raises(RuntimeError, match="geometry receipt file changed"):
        binding.verify_unchanged()


@pytest.mark.skipif(
    os.environ.get("CURE_LITE_REAL_DR_CREATE_ONLY_TEST") != "1",
    reason="opt-in real D_R cache construction; never trains or reads D_V/D_T",
)
def test_real_dr_scalar_inputs_create_only_integration() -> None:
    inputs = build_coverage_state_real_dr_inputs(**_paths())
    assert inputs.observability.decision is (
        CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF
    )
    assert inputs.raw_catalog.feature_stride == (
        inputs.scalar_cache.sobolev_config.truncation_radius
    )
    assert inputs.canonical_payload()["execution_policy"] == {
        "create_only": True,
        "training_performed": False,
        "calibration_performed": False,
        "inference_performed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    inputs.verify_unchanged()
