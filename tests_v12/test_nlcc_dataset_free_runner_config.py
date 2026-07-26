from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cure_lite.nlcc_dataset_free_runner_config import (
    DEVELOPMENT,
    EXPECTED_ADDITIVE_PATHS,
    HOLDOUT,
    INPUT_FREEZE_FILE_SHA256,
    NLCCDatasetFreeThresholds,
    PROFILE_INDEPENDENCE_FILE_SHA256,
    RUNNER_CLARIFICATION_FILE_SHA256,
    RUNNER_EVIDENCE_AMENDMENT_FILE_SHA256,
    RUNNER_PREREGISTRATION_FILE_SHA256,
    development_runner_config,
    holdout_runner_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_both_profiles_are_frozen_together_and_independent() -> None:
    development = development_runner_config()
    holdout = holdout_runner_config()
    assert development.profile.kind == DEVELOPMENT
    assert holdout.profile.kind == HOLDOUT
    assert development.profile.updates == 320
    assert holdout.profile.updates == 400
    assert development.profile.pair_slots == 640
    assert holdout.profile.pair_slots == 800
    assert development.decoder_seed == holdout.decoder_seed == 42
    assert development.decoder_initialization == "from_scratch_seed_42"
    assert holdout.decoder_initialization == "from_scratch_seed_42"
    assert development.optimizer_initialization == "fresh_empty_state"
    assert holdout.optimizer_initialization == "fresh_empty_state"
    assert development.development_state_carry_into_holdout is False
    assert holdout.development_state_carry_into_holdout is False
    assert development.profile.input_fingerprint != holdout.profile.input_fingerprint
    assert development.profile.schedule_fingerprint != holdout.profile.schedule_fingerprint


def test_loss_adam_batch_and_threshold_contracts_are_exact() -> None:
    config = development_runner_config()
    assert config.loss_dice_weight == 1.0
    assert config.loss_epsilon == 1e-6
    assert config.learning_rate == 0.001
    assert config.betas == (0.9, 0.999)
    assert config.optimizer_epsilon == 1e-8
    assert config.weight_decay == 0.0
    assert config.amsgrad is False
    assert config.maximize is False
    assert config.foreach is None
    assert config.capturable is False
    assert config.differentiable is False
    assert config.fused is None
    assert config.decoupled_weight_decay is False
    assert config.decoder_forward_batch_sizes_per_update == (4, 4, 4)
    assert config.decoder_states_per_update == 12
    assert config.parameter_tensors == 6
    assert config.parameters == 2593
    assert config.thresholds.manifest() == {
        "population_total_loss_max_exclusive": 0.1,
        "positive_anchor_min_exclusive": 0.95,
        "matched_anchor_null_max_exclusive": 0.05,
        "plus_background_max_exclusive": 0.05,
        "factual_miss_target_min_exclusive": 0.95,
        "factual_miss_background_max_exclusive": 0.05,
        "factual_no_miss_max_exclusive": 0.05,
        "clean_D_delta_mean_min_inclusive": 0.8,
        "clean_D_plus_max_exclusive": 0.05,
        "clean_D_minus_min_exclusive": 0.95,
        "D_wrong_direction_pixel_count_max_inclusive": 0,
        "zero_H_max_abs_max_inclusive": 0.05,
        "zero_G_near_max_abs_max_inclusive": 0.05,
        "zero_G_norm_tail_max_abs_max_inclusive": 0.05,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("population_total_loss_max_exclusive", 0.1000001),
        ("positive_anchor_min_exclusive", 0.949),
        ("zero_H_max_abs_max_inclusive", 0.051),
        ("D_wrong_direction_pixel_count_max_inclusive", 1),
    ],
)
def test_thresholds_reject_every_override(field: str, value: float | int) -> None:
    with pytest.raises(ValueError, match="freezes"):
        NLCCDatasetFreeThresholds(**{field: value})


def test_runner_rejects_hyperparameter_or_state_carry_override() -> None:
    config = development_runner_config()
    with pytest.raises(ValueError, match="freezes"):
        replace(config, learning_rate=0.002)
    with pytest.raises(ValueError, match="freezes"):
        replace(config, loss_epsilon=1e-5)
    with pytest.raises(ValueError, match="freezes"):
        replace(config, development_state_carry_into_holdout=True)


def test_effective_paths_and_all_runner_receipts_are_bound() -> None:
    assert "cure_lite/nlcc_dataset_free_runner.py" in EXPECTED_ADDITIVE_PATHS
    assert "cure_lite/experiment/nlcc_dataset_free_runner.py" not in (
        EXPECTED_ADDITIVE_PATHS
    )
    assert not (
        ROOT / "cure_lite/experiment/nlcc_dataset_free_runner.py"
    ).exists()
    manifest = development_runner_config().manifest()
    assert manifest["runner_preregistration"]["file_sha256"] == (
        RUNNER_PREREGISTRATION_FILE_SHA256
    )
    assert manifest["input_freeze"]["file_sha256"] == INPUT_FREEZE_FILE_SHA256
    assert manifest["runner_path_and_metric_clarification"]["file_sha256"] == (
        RUNNER_CLARIFICATION_FILE_SHA256
    )
    assert manifest["profile_independence_clarification"]["file_sha256"] == (
        PROFILE_INDEPENDENCE_FILE_SHA256
    )
    assert manifest["runner_evidence_r2_amendment"]["file_sha256"] == (
        RUNNER_EVIDENCE_AMENDMENT_FILE_SHA256
    )


def test_clis_expose_no_output_update_seed_or_hyperparameter_override() -> None:
    for relative in (
        "tools/evaluate_nlcc_development_regression.py",
        "tools/evaluate_nlcc_exposure_holdout.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert '"--preflight"' in source
        assert '"--output"' not in source
        assert '"--updates"' not in source
        assert '"--seed"' not in source
        assert '"--learning-rate"' not in source
