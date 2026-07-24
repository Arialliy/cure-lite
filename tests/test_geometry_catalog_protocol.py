from __future__ import annotations

import json
from pathlib import Path

import pytest

from cure_lite.cache.schema import file_sha256
from cure_lite.experiment.geometry_catalog_protocol import (
    GEOMETRY_CATALOG_CONFIG_SCHEMA,
    GeometryCatalogProtocol,
    load_geometry_catalog_protocol,
)
from tools.run_geometry_safe_p0 import (
    GEOMETRY_SAFE_FROZEN_CONFIG_FILE_SHA256,
    build_parser,
)


_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "protocols"
    / "IRSTD-1K"
    / "geometry_safe_p0_v2"
    / "config.json"
)


def test_geometry_safe_config_is_canonical_and_d_r_only() -> None:
    config = load_geometry_catalog_protocol(_CONFIG)
    assert config.schema_version == GEOMETRY_CATALOG_CONFIG_SCHEMA
    assert config.split == "D_R"
    assert config.predecessor.predecessor_p0_a_pass is False
    assert config.predecessor.v1_outcome_remains_valid is True
    assert config.predecessor.result_reinterpretation is False
    assert config.geometry.area_ratio_min_inclusive == 0.5
    assert config.geometry.area_ratio_max_inclusive == 2.0
    assert config.geometry.centroid_shift_max_evaluation_px_inclusive == 1.0
    assert config.analysis_population.hardcoded_identity_exclusions == ()
    assert len(config.fingerprint) == 64
    assert file_sha256(_CONFIG) == GEOMETRY_SAFE_FROZEN_CONFIG_FILE_SHA256


def test_geometry_safe_config_rejects_d_v_and_unknown_fields() -> None:
    payload = json.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["split"] = "D_V"
    with pytest.raises(ValueError, match="only D_R"):
        GeometryCatalogProtocol.from_mapping(payload)
    payload["split"] = "D_R"
    payload["unknown"] = True
    with pytest.raises(ValueError, match="fields are not canonical"):
        GeometryCatalogProtocol.from_mapping(payload)


def test_geometry_safe_config_rejects_hardcoded_exclusions() -> None:
    payload = json.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["analysis_population"]["hardcoded_identity_exclusions"] = [
        ["XDU486", 1, 1]
    ]
    with pytest.raises(ValueError, match="must be empty"):
        GeometryCatalogProtocol.from_mapping(payload)


def test_geometry_safe_config_cannot_reinterpret_v1() -> None:
    payload = json.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["predecessor"]["result_reinterpretation"] = True
    with pytest.raises(ValueError, match="must be false"):
        GeometryCatalogProtocol.from_mapping(payload)


def test_geometry_safe_cli_has_no_training_or_validation_input() -> None:
    options = {
        action.dest
        for action in build_parser()._actions
        if action.dest != "help"
    }
    assert options == {"manifest", "state_index", "config", "output"}
    assert not any("D_V" in option for option in options)
