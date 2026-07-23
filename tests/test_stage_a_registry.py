from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from cure_lite.calibration import THRESHOLD_SELECTION_RULE
from cure_lite.config import (
    DecoderConfig,
    InterventionConfig,
    LossConfig,
    MatchConfig,
    OccupancyConfig,
    TrainingConfig,
    config_to_dict,
)
from cure_lite.data import PreprocessConfig
from cure_lite.stage_a import (
    BASE_RUN_IDENTITY_FIELDS,
    MASTER_REGISTRY_SCHEMA_VERSION,
    SEED_REGISTRY_SCHEMA_VERSION,
    STAGE_A_METHOD_ORDER,
    STAGE_A_VARIANTS,
    _validate_protocols,
    build_master_registry,
    load_seed_registry,
    threshold_grid_fingerprint,
    validate_master_registry_mapping,
    validate_seed_registry_mapping,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _protocols(seed: int) -> dict[str, dict[str, object]]:
    return {
        "A": {
            "evaluation_mode": "anchor",
            "decoder_variant": None,
            "protocol_fingerprint": _digest(f"seed-{seed}-anchor"),
            "decoder_artifact_fingerprint": None,
            "selected_threshold": 0.5,
        },
        "Base@B": {
            "evaluation_mode": "base_at_budget",
            "decoder_variant": None,
            "protocol_fingerprint": _digest(f"seed-{seed}-base-at-budget"),
            "decoder_artifact_fingerprint": None,
            "selected_threshold": 0.3,
        },
        "F": {
            "evaluation_mode": "residual",
            "decoder_variant": "factual_only",
            "protocol_fingerprint": _digest(f"seed-{seed}-f-protocol"),
            "decoder_artifact_fingerprint": _digest(f"seed-{seed}-f-artifact"),
            "selected_threshold": 0.4,
        },
        "F×": {
            "evaluation_mode": "residual",
            "decoder_variant": "factual_exposure_matched",
            "protocol_fingerprint": _digest(f"seed-{seed}-fx-protocol"),
            "decoder_artifact_fingerprint": _digest(f"seed-{seed}-fx-artifact"),
            "selected_threshold": 0.6,
        },
        "U": {
            "evaluation_mode": "residual",
            "decoder_variant": "uniform_legal",
            "protocol_fingerprint": _digest(f"seed-{seed}-u-protocol"),
            "decoder_artifact_fingerprint": _digest(f"seed-{seed}-u-artifact"),
            "selected_threshold": None,
        },
    }


def _common(seed: int) -> dict[str, object]:
    anchor_grid = [0.3, 0.5, 0.7]
    residual_grid = [0.4, 0.6]
    base_grid = [0.3, 0.5]
    seed_digest_fields = (
        "base_fingerprint",
        "base_state_fingerprint",
        "stage_a_complete_fingerprint",
        "d_v_base_cache_index_fingerprint",
        "d_v_base_cache_index_sha256",
        "d_r_base_cache_index_fingerprint",
        "d_r_base_cache_index_sha256",
        "d_r_state_cache_index_fingerprint",
        "d_r_state_cache_index_sha256",
        "state_fingerprint",
        "efficiency_static_fingerprint",
        "efficiency_receipt_fingerprint",
    )
    common: dict[str, object] = {
        "manifest_fingerprint": _digest("manifest"),
        "d_v_image_fingerprint": _digest("d-v-images"),
        "d_v_gt_fingerprint": _digest("d-v-gt"),
        "anchor_protocol_sha256": _digest(f"seed-{seed}-anchor"),
        "tau_o": 0.5,
        "tau_B": 0.3,
        "pixel_fa_budget": 0.0001,
        "component_fa_per_mp_budget": 100.0,
        "raw_background_fa_budget": 0.0001,
        "minimum_retention": 0.99,
        "null_residual_candidate": True,
        "threshold_selection_rule": THRESHOLD_SELECTION_RULE,
        "anchor_threshold_grid": anchor_grid,
        "anchor_threshold_grid_fingerprint": threshold_grid_fingerprint(anchor_grid),
        "residual_threshold_grid": residual_grid,
        "residual_threshold_grid_fingerprint": threshold_grid_fingerprint(
            residual_grid
        ),
        "base_threshold_grid": base_grid,
        "base_threshold_grid_fingerprint": threshold_grid_fingerprint(base_grid),
        "occupancy_config": config_to_dict(OccupancyConfig(threshold=0.5)),
        "matching_config": config_to_dict(MatchConfig()),
        "intervention_config": config_to_dict(InterventionConfig()),
        "preprocessing": PreprocessConfig(
            height=32,
            width=32,
            color_mode="L",
            mean=(0.5,),
            std=(0.25,),
        ).fingerprint_payload(),
        "decoder_config": config_to_dict(DecoderConfig(feature_channels=4)),
        "loss_config": config_to_dict(LossConfig()),
        "training_config": config_to_dict(TrainingConfig()),
        "optimization_config": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
        "branch_batch_sizes": {
            "factual_miss": 4,
            "factual_no_miss": 4,
            "synthetic": 4,
        },
        "data_augmentation": "none_frozen_base_cache",
        "fixed_stopping_rule": {"epochs": 20, "steps_per_epoch": 40},
        "global_seed": seed,
        "steps_per_epoch": 40,
        "trained_epochs": 20,
        "efficiency_device_type": "cuda",
        "efficiency_warmup": 10,
        "efficiency_repetitions": 50,
    }
    for field in seed_digest_fields:
        common[field] = _digest(f"seed-{seed}-{field}")
    common["base_run_identity"] = {
        "producer_schema": "cure-lite-reference-base-run-v1",
        "base_fingerprint": common["base_fingerprint"],
        "base_state_fingerprint": common["base_state_fingerprint"],
        "training_run_fingerprint": _digest(f"seed-{seed}-base-run"),
        "completion_receipt_sha256": _digest(f"seed-{seed}-base-complete"),
        "checkpoint_sha256": _digest(f"seed-{seed}-base-checkpoint"),
        "selection_fingerprint": _digest(f"seed-{seed}-base-selection"),
        "source_fingerprint": _digest("reference-base-source"),
    }
    assert set(common["base_run_identity"]) == set(BASE_RUN_IDENTITY_FIELDS)
    return common


def _seed_registry(seed: int) -> dict[str, object]:
    return {
        "schema_version": SEED_REGISTRY_SCHEMA_VERSION,
        "artifact_type": "stage_a_frozen_registry",
        "method_version": "cure-lite-v0.1",
        "stage": "Stage A",
        "split": "D_V",
        "thresholds_frozen": True,
        "common_config": _common(seed),
        "protocols": _protocols(seed),
    }


def test_seed_registry_matches_formal_five_way_contract() -> None:
    payload = _seed_registry(7)
    validated = validate_seed_registry_mapping(payload)

    assert SEED_REGISTRY_SCHEMA_VERSION == "stage-a-frozen-registry-v6"
    assert STAGE_A_METHOD_ORDER == ("A", "Base@B", "F", "F×", "U")
    assert tuple(STAGE_A_VARIANTS) == STAGE_A_METHOD_ORDER
    assert tuple(validated.protocols) == ("A", "Base@B", "F", "F×", "U")
    for method in ("A", "Base@B"):
        assert validated.protocols[method]["decoder_variant"] is None
        assert validated.protocols[method]["decoder_artifact_fingerprint"] is None
    assert validated.protocols["F×"]["decoder_variant"] == (
        "factual_exposure_matched"
    )


def test_seed_registry_binds_the_exact_threshold_selection_rule() -> None:
    payload = _seed_registry(7)
    payload["common_config"]["threshold_selection_rule"] = "legacy-rule"
    with pytest.raises(RuntimeError, match="threshold_selection_rule"):
        validate_seed_registry_mapping(payload)


def test_seed_registry_v5_uses_neutral_base_and_exact_cache_identities() -> None:
    payload = _seed_registry(9)
    common = payload["common_config"]
    assert "base_run_identity" in common
    assert not any("base_training_" in key for key in common)
    assert {
        "d_v_base_cache_index_fingerprint",
        "d_v_base_cache_index_sha256",
        "d_r_base_cache_index_fingerprint",
        "d_r_base_cache_index_sha256",
        "d_r_state_cache_index_fingerprint",
        "d_r_state_cache_index_sha256",
    } <= set(common)
    validated = validate_seed_registry_mapping(payload)
    assert validated.base_run_identity.producer_schema == (
        "cure-lite-reference-base-run-v1"
    )


def test_seed_registry_rejects_base_identity_or_efficiency_drift() -> None:
    payload = _seed_registry(9)
    payload["common_config"]["base_run_identity"]["base_fingerprint"] = _digest(
        "another-base"
    )
    with pytest.raises(RuntimeError, match="Base run identity"):
        validate_seed_registry_mapping(payload)

    payload = _seed_registry(9)
    payload["common_config"]["efficiency_device_type"] = "mps"
    with pytest.raises(ValueError, match="efficiency_device_type"):
        validate_seed_registry_mapping(payload)


def test_registry_rejects_obsolete_five_decoder_contract() -> None:
    legacy = _seed_registry(7)
    legacy["schema_version"] = "stage-a-frozen-registry-v4"
    legacy["protocols"] = {
        method: {
            "decoder_variant": variant,
            "protocol_sha256": _digest(f"legacy-{method}-protocol"),
            "decoder_checkpoint_sha256": _digest(f"legacy-{method}-checkpoint"),
            "tau_r": 0.4,
        }
        for method, variant in {
            "P": "parallel_all_gt",
            "F": "factual_only",
            "F×": "factual_exposure_matched",
            "U": "uniform_legal",
            "S": "score_hard",
        }.items()
    }

    with pytest.raises(RuntimeError, match="schema_version"):
        validate_seed_registry_mapping(legacy)


def test_decoder_free_controls_reject_fabricated_artifacts_and_threshold_drift() -> None:
    for method in ("A", "Base@B"):
        protocols = _protocols(11)
        protocols[method]["decoder_artifact_fingerprint"] = _digest(
            f"fabricated-{method}"
        )
        with pytest.raises(RuntimeError, match="decoder-free"):
            _validate_protocols(
                protocols,
                residual_grid=[0.4, 0.6],
                null_residual_candidate=True,
                tau_o=0.5,
                tau_b=0.3,
                anchor_protocol_sha256=_digest("seed-11-anchor"),
            )

        protocols = _protocols(11)
        protocols[method]["decoder_variant"] = "uniform_legal"
        with pytest.raises(RuntimeError, match="must be None"):
            _validate_protocols(
                protocols,
                residual_grid=[0.4, 0.6],
                null_residual_candidate=True,
                tau_o=0.5,
                tau_b=0.3,
                anchor_protocol_sha256=_digest("seed-11-anchor"),
            )

    protocols = _protocols(11)
    protocols["Base@B"]["selected_threshold"] = 0.5
    with pytest.raises(RuntimeError, match="common_config.tau_B"):
        _validate_protocols(
            protocols,
            residual_grid=[0.4, 0.6],
            null_residual_candidate=True,
            tau_o=0.5,
            tau_b=0.3,
            anchor_protocol_sha256=_digest("seed-11-anchor"),
        )


def test_residual_methods_require_exact_distinct_decoder_artifacts() -> None:
    protocols = _protocols(13)
    protocols["F×"]["decoder_artifact_fingerprint"] = protocols["F"][
        "decoder_artifact_fingerprint"
    ]
    with pytest.raises(RuntimeError, match="unique across F/F×/U"):
        _validate_protocols(
            protocols,
            residual_grid=[0.4, 0.6],
            null_residual_candidate=True,
            tau_o=0.5,
            tau_b=0.3,
            anchor_protocol_sha256=_digest("seed-13-anchor"),
        )


def test_master_registry_accepts_five_way_seed_registries(tmp_path) -> None:
    loaded = {}
    for seed in range(5):
        path = tmp_path / f"seed-{seed}.json"
        path.write_text(
            json.dumps(_seed_registry(seed), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        loaded[f"seed-{seed}"] = load_seed_registry(path)

    master = build_master_registry(
        loaded,
    )
    validated = validate_master_registry_mapping(deepcopy(master))

    assert MASTER_REGISTRY_SCHEMA_VERSION == "stage-a-master-registry-v6"
    assert master["schema_version"] == MASTER_REGISTRY_SCHEMA_VERSION
    assert master["seed_count"] == 5
    assert tuple(validated.seed_registries) == tuple(f"seed-{seed}" for seed in range(5))

    legacy_master = deepcopy(master)
    assert master["efficiency_protocol"] == {
        "device_type": "cuda",
        "warmup": 10,
        "repetitions": 50,
    }

    legacy_master["schema_version"] = "stage-a-master-registry-v4"
    with pytest.raises(RuntimeError, match="schema_version"):
        validate_master_registry_mapping(legacy_master)

    inconsistent = deepcopy(master)
    inconsistent["efficiency_protocol"]["repetitions"] = 51
    with pytest.raises(RuntimeError, match="differs from seed registry receipts"):
        validate_master_registry_mapping(inconsistent)
