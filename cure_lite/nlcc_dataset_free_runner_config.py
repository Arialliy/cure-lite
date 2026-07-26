"""Frozen runner configuration for NLCC-v12 dataset-free gates.

The development and exposure-holdout profiles are fixed together in this
module.  They deliberately expose no training, threshold, output-path, or
retry override.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path


METHOD_ID = "nlcc_v12"
RUNNER_PREREGISTRATION_REPO_PATH = (
    "protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/"
    "dataset_free_runner_evaluation_preregistration.json"
)
RUNNER_PREREGISTRATION_FILE_SHA256 = (
    "014a5df9b9b7088b504e53ef1921a2f0c43e9ca392997b3cde2085e50fa430bd"
)
RUNNER_PREREGISTRATION_FINGERPRINT = (
    "3a3ca440e974a6810d047730ea993c553eb6266a8fb0e68d99a137842011a0f9"
)
INPUT_FREEZE_REPO_PATH = (
    "protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/"
    "dataset_free_input_freeze_and_reachability_receipt.json"
)
INPUT_FREEZE_FILE_SHA256 = (
    "6f489860755fd4329549116fd61267ba5aa5eaa26c4c0cfd879cb04d95fd4800"
)
INPUT_FREEZE_FINGERPRINT = (
    "2867331bcc695c9a5a31bee98233f5f7db4aeefbf2385d18450422ffd0ad870a"
)
RUNNER_CLARIFICATION_REPO_PATH = (
    "protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/"
    "dataset_free_runner_path_and_metric_clarification.json"
)
RUNNER_CLARIFICATION_FILE_SHA256 = (
    "b252391e65149f28340560912d09210c3f972b88360d1330610fd1696647e290"
)
RUNNER_CLARIFICATION_FINGERPRINT = (
    "e4e91db7b93eec33fac321cf07329e3afdb4f41e5b69f4e68e6510c2c20e4166"
)
PROFILE_INDEPENDENCE_REPO_PATH = (
    "protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/"
    "dataset_free_profile_independence_clarification.json"
)
PROFILE_INDEPENDENCE_FILE_SHA256 = (
    "20b1b31d610c2ec19ad46ab89b3973f6674bdb521a43455d4de0b094fc1efd9e"
)
PROFILE_INDEPENDENCE_FINGERPRINT = (
    "de212d80195dc0e59a2ba44dc5b70bc87196c1b8dcdb0c24ee1380bf28087560"
)

EXPECTED_ADDITIVE_PATHS = (
    "cure_lite/nlcc_dataset_free_runner_config.py",
    "cure_lite/nlcc_dataset_free_runner.py",
    "tools/evaluate_nlcc_development_regression.py",
    "tools/evaluate_nlcc_exposure_holdout.py",
    "tests_v12/test_nlcc_dataset_free_runner_config.py",
    "tests_v12/test_nlcc_dataset_free_runner.py",
)

DEVELOPMENT = "development"
HOLDOUT = "holdout"
PROFILE_KINDS = (DEVELOPMENT, HOLDOUT)


@dataclass(frozen=True)
class NLCCDatasetFreeThresholds:
    """The exact preregistered NLCC-v12 decision thresholds."""

    population_total_loss_max_exclusive: float = 0.1
    positive_anchor_min_exclusive: float = 0.95
    matched_anchor_null_max_exclusive: float = 0.05
    plus_background_max_exclusive: float = 0.05
    factual_miss_target_min_exclusive: float = 0.95
    factual_miss_background_max_exclusive: float = 0.05
    factual_no_miss_max_exclusive: float = 0.05
    clean_D_delta_mean_min_inclusive: float = 0.8
    clean_D_plus_max_exclusive: float = 0.05
    clean_D_minus_min_exclusive: float = 0.95
    D_wrong_direction_pixel_count_max_inclusive: int = 0
    zero_H_max_abs_max_inclusive: float = 0.05
    zero_G_near_max_abs_max_inclusive: float = 0.05
    zero_G_norm_tail_max_abs_max_inclusive: float = 0.05

    def __post_init__(self) -> None:
        expected = {
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
        for name, frozen in expected.items():
            value = getattr(self, name)
            if isinstance(frozen, int):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{name} must be an integer")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
            ):
                raise TypeError(f"{name} must be a finite real number")
            if value != frozen:
                raise ValueError(f"NLCC-v12 freezes {name}={frozen!r}")

    def manifest(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class NLCCDatasetFreeProfileConfig:
    """One exact frozen development or independent-holdout profile."""

    kind: str
    profile_id: str
    updates: int
    pair_slots: int
    input_fingerprint: str
    catalog_fingerprint: str
    schedule_fingerprint: str
    factual_population_fingerprint: str
    factual_schedule_fingerprint: str
    canonical_artifact_directory: str
    pre_run_authorization: str
    evidentiary_role: str
    attempt_ordinal_max: int = 1

    def __post_init__(self) -> None:
        if self.kind not in PROFILE_KINDS:
            raise ValueError("unknown NLCC dataset-free profile kind")
        expected = _PROFILE_VALUES[self.kind]
        for name, frozen in expected.items():
            if getattr(self, name) != frozen:
                raise ValueError(
                    f"NLCC-v12 {self.kind} freezes {name}={frozen!r}"
                )
        if self.attempt_ordinal_max != 1:
            raise ValueError("NLCC-v12 permits at most one profile attempt")
        for name in (
            "input_fingerprint",
            "catalog_fingerprint",
            "schedule_fingerprint",
            "factual_population_fingerprint",
            "factual_schedule_fingerprint",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} must be a lowercase SHA256 value")
        for name in (
            "canonical_artifact_directory",
            "pre_run_authorization",
        ):
            path = Path(getattr(self, name))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{name} must be a repository-relative path")

    def manifest(self) -> dict[str, object]:
        return asdict(self)


_PROFILE_VALUES: dict[str, dict[str, object]] = {
    DEVELOPMENT: {
        "kind": DEVELOPMENT,
        "profile_id": "nlcc_v12_development",
        "updates": 320,
        "pair_slots": 640,
        "input_fingerprint": (
            "4f387e3e513a93a1cee58ee68d9d67eb5b2746688da42ec40572ae6fc1df55a7"
        ),
        "catalog_fingerprint": (
            "1105dd6c086b2482217a28336ba588b6d87aa09c33a100ed932dc85e8e2ca257"
        ),
        "schedule_fingerprint": (
            "828f261a62ee9bd8486d2bf24131a5031383fae488ef435638a13c32adca16bc"
        ),
        "factual_population_fingerprint": (
            "3e467075c5b4ae12cc09663574b0a637a34cf6a9d82988621b0645c836d9213e"
        ),
        "factual_schedule_fingerprint": (
            "bcb8e7a26d19256b4d901dc280128776d9ffd9184666ca5eec7ead9c25d1e49e"
        ),
        "canonical_artifact_directory": (
            "protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/"
            "development_regression_r1"
        ),
        "pre_run_authorization": (
            "protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/"
            "development_pre_run_authorization.json"
        ),
        "evidentiary_role": (
            "development_learnability_gate_not_independent_confirmation"
        ),
        "attempt_ordinal_max": 1,
    },
    HOLDOUT: {
        "kind": HOLDOUT,
        "profile_id": "nlcc_v12_exposure_holdout",
        "updates": 400,
        "pair_slots": 800,
        "input_fingerprint": (
            "d475e7037656016233949e28fb3877b5e92af8745bc168aa8660284c93fa5335"
        ),
        "catalog_fingerprint": (
            "e0f292c2283e4b5d750af3f3507113cb144314afd3f89856ffecc889c29f4619"
        ),
        "schedule_fingerprint": (
            "cde2e6c3868d12bced218ce01c798441fa01067050b01daa9d8f50b1fe914326"
        ),
        "factual_population_fingerprint": (
            "10dd8efba4ae2dac8ea515055558cea012c1d0d8ad8a6562b91e927756628739"
        ),
        "factual_schedule_fingerprint": (
            "18d1a702d62299d072328c407233cfc5f65e94ebd0eaa71eb2c7004af1764073"
        ),
        "canonical_artifact_directory": (
            "protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/"
            "exposure_holdout_r1"
        ),
        "pre_run_authorization": (
            "protocols/IRSTD-1K/null_anchored_local_count_crossing_v12/"
            "holdout_pre_run_authorization.json"
        ),
        "evidentiary_role": (
            "unique_exposure_confirmation_not_real_detection_performance"
        ),
        "attempt_ordinal_max": 1,
    },
}


@dataclass(frozen=True)
class NLCCDatasetFreeRunnerConfig:
    """Complete immutable execution contract for one profile."""

    profile: NLCCDatasetFreeProfileConfig
    thresholds: NLCCDatasetFreeThresholds = field(
        default_factory=NLCCDatasetFreeThresholds
    )
    method_id: str = METHOD_ID
    feature_channels: int = 8
    feature_stride: int = 4
    decoder_seed: int = 42
    device: str = "cpu"
    torch_threads: int = 2
    deterministic_algorithms: bool = True
    optimizer: str = "Adam"
    learning_rate: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    optimizer_epsilon: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False
    maximize: bool = False
    foreach: None = None
    capturable: bool = False
    differentiable: bool = False
    fused: None = None
    decoupled_weight_decay: bool = False
    loss_dice_weight: float = 1.0
    loss_epsilon: float = 1e-6
    factual_miss_states_per_update: int = 4
    factual_no_miss_states_per_update: int = 4
    pair_rows_per_update: int = 2
    decoder_forward_batch_sizes_per_update: tuple[int, int, int] = (4, 4, 4)
    decoder_states_per_update: int = 12
    backward_calls_per_update: int = 1
    optimizer_steps_per_update: int = 1
    parameter_tensors: int = 6
    parameters: int = 2593
    decoder_initialization: str = "from_scratch_seed_42"
    optimizer_initialization: str = "fresh_empty_state"
    development_state_carry_into_holdout: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.profile, NLCCDatasetFreeProfileConfig):
            raise TypeError("profile must be NLCCDatasetFreeProfileConfig")
        if not isinstance(self.thresholds, NLCCDatasetFreeThresholds):
            raise TypeError("thresholds must be NLCCDatasetFreeThresholds")
        expected = {
            "method_id": METHOD_ID,
            "feature_channels": 8,
            "feature_stride": 4,
            "decoder_seed": 42,
            "device": "cpu",
            "torch_threads": 2,
            "deterministic_algorithms": True,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "betas": (0.9, 0.999),
            "optimizer_epsilon": 1e-8,
            "weight_decay": 0.0,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "decoupled_weight_decay": False,
            "loss_dice_weight": 1.0,
            "loss_epsilon": 1e-6,
            "factual_miss_states_per_update": 4,
            "factual_no_miss_states_per_update": 4,
            "pair_rows_per_update": 2,
            "decoder_forward_batch_sizes_per_update": (4, 4, 4),
            "decoder_states_per_update": 12,
            "backward_calls_per_update": 1,
            "optimizer_steps_per_update": 1,
            "parameter_tensors": 6,
            "parameters": 2593,
            "decoder_initialization": "from_scratch_seed_42",
            "optimizer_initialization": "fresh_empty_state",
            "development_state_carry_into_holdout": False,
        }
        for name, frozen in expected.items():
            if getattr(self, name) != frozen:
                raise ValueError(f"NLCC-v12 freezes {name}={frozen!r}")

    def manifest(self) -> dict[str, object]:
        return {
            **asdict(self),
            "runner_preregistration": {
                "repo_path": RUNNER_PREREGISTRATION_REPO_PATH,
                "file_sha256": RUNNER_PREREGISTRATION_FILE_SHA256,
                "fingerprint": RUNNER_PREREGISTRATION_FINGERPRINT,
            },
            "input_freeze": {
                "repo_path": INPUT_FREEZE_REPO_PATH,
                "file_sha256": INPUT_FREEZE_FILE_SHA256,
                "fingerprint": INPUT_FREEZE_FINGERPRINT,
            },
            "runner_path_and_metric_clarification": {
                "repo_path": RUNNER_CLARIFICATION_REPO_PATH,
                "file_sha256": RUNNER_CLARIFICATION_FILE_SHA256,
                "fingerprint": RUNNER_CLARIFICATION_FINGERPRINT,
            },
            "profile_independence_clarification": {
                "repo_path": PROFILE_INDEPENDENCE_REPO_PATH,
                "file_sha256": PROFILE_INDEPENDENCE_FILE_SHA256,
                "fingerprint": PROFILE_INDEPENDENCE_FINGERPRINT,
            },
        }


def _profile(kind: str) -> NLCCDatasetFreeProfileConfig:
    if kind not in PROFILE_KINDS:
        raise ValueError("kind must be development or holdout")
    return NLCCDatasetFreeProfileConfig(**_PROFILE_VALUES[kind])


def development_runner_config() -> NLCCDatasetFreeRunnerConfig:
    return NLCCDatasetFreeRunnerConfig(profile=_profile(DEVELOPMENT))


def holdout_runner_config() -> NLCCDatasetFreeRunnerConfig:
    return NLCCDatasetFreeRunnerConfig(profile=_profile(HOLDOUT))


__all__ = [
    "DEVELOPMENT",
    "EXPECTED_ADDITIVE_PATHS",
    "HOLDOUT",
    "INPUT_FREEZE_FILE_SHA256",
    "INPUT_FREEZE_FINGERPRINT",
    "INPUT_FREEZE_REPO_PATH",
    "METHOD_ID",
    "NLCCDatasetFreeProfileConfig",
    "NLCCDatasetFreeRunnerConfig",
    "NLCCDatasetFreeThresholds",
    "PROFILE_KINDS",
    "PROFILE_INDEPENDENCE_FILE_SHA256",
    "PROFILE_INDEPENDENCE_FINGERPRINT",
    "PROFILE_INDEPENDENCE_REPO_PATH",
    "RUNNER_PREREGISTRATION_FILE_SHA256",
    "RUNNER_PREREGISTRATION_FINGERPRINT",
    "RUNNER_PREREGISTRATION_REPO_PATH",
    "RUNNER_CLARIFICATION_FILE_SHA256",
    "RUNNER_CLARIFICATION_FINGERPRINT",
    "RUNNER_CLARIFICATION_REPO_PATH",
    "development_runner_config",
    "holdout_runner_config",
]
