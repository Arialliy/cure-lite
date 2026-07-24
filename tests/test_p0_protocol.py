from __future__ import annotations

import json
from pathlib import Path

import pytest

from cure_lite.experiment.p0_protocol import (
    P0_CONFIG_SCHEMA,
    P0DiagnosticConfig,
    load_p0_config,
)
from cure_lite.cache.schema import file_sha256
from tools.run_p0_diagnostics import (
    P0_FROZEN_CONFIG_FILE_SHA256,
    build_parser,
)


_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "protocols"
    / "IRSTD-1K"
    / "p0_v1"
    / "p0_config.json"
)


def test_frozen_p0_config_is_canonical_and_d_r_only() -> None:
    config = load_p0_config(_CONFIG)
    assert config.schema_version == P0_CONFIG_SCHEMA
    assert config.split == "D_R"
    assert config.exposure.epochs == 800
    assert config.exposure.steps_per_epoch == 40
    assert config.exposure.synthetic_batch == 4
    assert config.exposure.seeds == (42, 43)
    assert config.overlap.coverage_minimum == 0.9
    assert config.separability.auc_maximum == 0.7
    assert len(config.fingerprint) == 64
    assert file_sha256(_CONFIG) == P0_FROZEN_CONFIG_FILE_SHA256


def test_p0_config_rejects_d_v_and_unknown_fields() -> None:
    payload = json.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["split"] = "D_V"
    with pytest.raises(ValueError, match="only D_R"):
        P0DiagnosticConfig.from_mapping(payload)
    payload["split"] = "D_R"
    payload["unknown"] = True
    with pytest.raises(ValueError, match="fields are not canonical"):
        P0DiagnosticConfig.from_mapping(payload)


def test_p0_config_requires_explicit_geometry_thresholds() -> None:
    payload = json.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["geometry"].pop("legal_area_ratio_min_inclusive")
    with pytest.raises(ValueError, match="geometry fields"):
        P0DiagnosticConfig.from_mapping(payload)


def test_p0_config_rejects_invalid_logit_clip() -> None:
    payload = json.loads(_CONFIG.read_text(encoding="utf-8"))
    payload["overlap"]["probability_clip"] = 0.5
    with pytest.raises(ValueError, match="below 0.5"):
        P0DiagnosticConfig.from_mapping(payload)


def test_p0_cli_exposes_no_validation_or_training_argument() -> None:
    options = {
        action.dest
        for action in build_parser()._actions
        if action.dest != "help"
    }
    assert options == {"manifest", "state_index", "config", "output"}
    assert not any("D_V" in option for option in options)
