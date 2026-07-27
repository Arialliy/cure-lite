"""P0-bound CMIF/SORR bounded-400 training protocol.

This module is the only package-level entry point for the v17 bounded run.
It binds the frozen seed-42 bounded population and schedule, the generated
CMIF receipt, and the two independently persisted real-``D_R`` P0 receipts
before dispatching the unchanged SORR/identity/separable objective suite.

It does not expose a CLI, create artifacts, access ``D_V``/``D_T``, retry a
run, or authorize Formal800.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
from typing import Mapping

import torch

from ..cache.schema import file_sha256, stable_fingerprint
from ..coverage_state_centered_mixed_interaction import (
    CMIF_ENERGY_POLICY,
    CMIF_INPUT_REPRESENTATION,
    CMIF_INTERACTION_POLICY,
    CURELiteCenteredMixedInteractionLevelSet,
    CoverageStateCenteredMixedInteractionConfig,
)
from ..coverage_state_phase_preserving import (
    CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
)
from ..coverage_state_precomputed_cache import CoverageStateScalarCache
from ..coverage_state_schedule import CoverageStateTrainingSchedule
from ..coverage_state_sobolev import (
    CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY,
)
from .coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_SEED,
    CoverageStateBoundedPreflight,
)
from .coverage_state_bounded_runner import (
    COVERAGE_STATE_BOUNDED_IMPLEMENTATION_PATHS,
    CoverageStateBoundedRunResult,
    _bounded_result_checks,
    _deterministic_execution,
)
from .coverage_state_cmif_dataset_free import (
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_CMIF_FORMAL_WIDTH,
    CoverageStateCMIFDatasetFreeReceipt,
)
from .coverage_state_training import (
    COVERAGE_STATE_BOUNDED_SCOPE,
    CoverageStateMatchedTrainingConfig,
    CoverageStateMatchedTrainingResult,
    CoverageStateRunAuthorization,
    train_matched_coverage_state_cmif_support_oriented_objectives,
)
from .coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    CoverageStateZeroLevelEvaluationConfig,
    CoverageStateZeroLevelEvaluationResult,
    evaluate_coverage_state_zero_level_checkpoint,
)
from ..train.coverage_state_fused_step import (
    COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES,
    CoverageStatePairObjective,
)


COVERAGE_STATE_CMIF_BOUNDED_AUTHORIZATION_SCHEMA = (
    "cure-lite-cmif-v17-bounded-run-authorization-v1"
)
COVERAGE_STATE_CMIF_BOUNDED_RESULT_SCHEMA = (
    "cure-lite-cmif-v17-bounded-run-result-v1"
)
COVERAGE_STATE_CMIF_P0_R1_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/cure_lite_cmif_v17_p0_r1"
)
COVERAGE_STATE_CMIF_P0_R2_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/cure_lite_cmif_v17_p0_r2"
)
COVERAGE_STATE_CMIF_P0_R1_COMPLETE_FINGERPRINT = (
    "a997bee968af9149888c442152b40ab1f1043e6dac9d2f8ec890d4308b6c1342"
)
COVERAGE_STATE_CMIF_P0_R1_COMPLETE_SHA256 = (
    "2d27236789c5bcf25187a7031b665eeedc47a7340f0ccf83476802dba883bbe6"
)
COVERAGE_STATE_CMIF_P0_R2_COMPLETE_FINGERPRINT = (
    "942d739b1288170a370471116ebfb8e11c6752523871c27828e898a6a3aae05b"
)
COVERAGE_STATE_CMIF_P0_R2_COMPLETE_SHA256 = (
    "c021487cb5253cee3133c3812ee82f5bd7b03f4dd5fedf446f26daae69464237"
)
COVERAGE_STATE_CMIF_P0_CORE_SHA256 = (
    "84cdcfdbc58a38eb04f1bbd419309f8d58700b05bdfc0b40523390cec6809c04"
)
COVERAGE_STATE_CMIF_P0_CORE_RECEIPT_FINGERPRINT = (
    "ec4320ccb82d8777d1e27fbe77563b2a08071b169b02229341f5ba12745385ab"
)
COVERAGE_STATE_CMIF_P0_BOUNDED_POPULATION_FINGERPRINT = (
    "1a53467d57bea595afcc1edd3330708d1dda39e0e2d606325e552e8993e7841c"
)
COVERAGE_STATE_CMIF_P0_DATASET_FREE_RECEIPT_FINGERPRINT = (
    "9af196be896216003997e79e7781bd4a057ea6eff4e0ea587de3544fc1e772b7"
)
COVERAGE_STATE_CMIF_P0_SOURCE_BINDING_FINGERPRINT = (
    "9689ac7dc4cd95bd0e9bcf79e12e83bc1c8606a96e99ca27945dc07baf4fc74d"
)
COVERAGE_STATE_CMIF_P0_REPLAY_COMPARISON_FINGERPRINT = (
    "c48a1581c1b6e5a439c10b384b9f7b58f077cbe113ce3dcc2df7e70913a976e3"
)
COVERAGE_STATE_CMIF_P0_REPLAY_COMPARISON_SHA256 = (
    "abad7e94bc20502eb513f4ed96faa87d3f142c4f046cc3b6b25eb1a5f82c24ae"
)
COVERAGE_STATE_CMIF_P0_DECISION_FINGERPRINT = (
    "5e0bb9971db1d6ac90c5269e4deddc91036b6d231cc137a2a0433ceff42db8ea"
)
COVERAGE_STATE_CMIF_P0_DECISION_SHA256 = (
    "7b882a4601c178203f160e0ae047d5c7041e7c3120c1cc40371ca1056ebfe431"
)
COVERAGE_STATE_CMIF_P0_EVIDENCE_FINGERPRINT = (
    "77a4e8a1daf1a701ffe5277fb1a34d444822e112ac7402a4a6d70b6c53b6082c"
)
COVERAGE_STATE_CMIF_P0_R1_DECISION = "CMIF_V17_P0_REPLAY_PENDING"
COVERAGE_STATE_CMIF_P0_R2_DECISION = "CMIF_V17_P0_PASS"
COVERAGE_STATE_CMIF_BOUNDED_IMPLEMENTATION_PATHS = tuple(
    dict.fromkeys(
        (
            *COVERAGE_STATE_BOUNDED_IMPLEMENTATION_PATHS,
            "cure_lite/coverage_state_phase_preserving.py",
            "cure_lite/coverage_state_centered_mixed_interaction.py",
            "cure_lite/experiment/coverage_state_cmif_dataset_free.py",
            "cure_lite/experiment/coverage_state_cmif_p0.py",
            "cure_lite/experiment/coverage_state_cmif_bounded_runner.py",
            "tools/audit_coverage_state_cmif_v17.py",
        )
    )
)

_INCOMPLETE = ".incomplete"


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _current_cmif_implementation_binding(
) -> tuple[tuple[str, str], ...]:
    root = _repository_root()
    result: list[tuple[str, str]] = []
    for relative in COVERAGE_STATE_CMIF_BOUNDED_IMPLEMENTATION_PATHS:
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.resolve(strict=True)
            != Path(os.path.abspath(path))
        ):
            raise RuntimeError(
                f"CMIF bounded implementation path is invalid: {relative}"
            )
        result.append((relative, file_sha256(path)))
    return tuple(result)


def _cmif_model_config_payload(
    config: CoverageStateCenteredMixedInteractionConfig,
) -> dict[str, object]:
    if not isinstance(
        config,
        CoverageStateCenteredMixedInteractionConfig,
    ):
        raise TypeError(
            "config must be CoverageStateCenteredMixedInteractionConfig"
        )
    return {
        "feature_channels": config.feature_channels,
        "feature_stride": config.feature_stride,
        "width": config.width,
        "normalization_epsilon_hex": (
            config.normalization_epsilon.hex()
        ),
        "field_amplitude_hex": config.field_amplitude.hex(),
        "initial_field_value_hex": config.initial_field_value.hex(),
        "field_policy": config.field_policy,
        "target_policy": config.target_policy,
        "output_policy": config.output_policy,
        "feature_policy": config.feature_policy,
        "numerical_policy": config.numerical_policy,
        "coverage_policy": config.coverage_policy,
        "input_representation": config.input_representation,
        "interaction_policy": config.interaction_policy,
        "energy_policy": config.energy_policy,
        "coarse_radius": config.coarse_radius,
        "neutral_phase_hex": config.neutral_phase.hex(),
        "phase_occupancy_channels": (
            config.phase_occupancy_channels
        ),
        "expected_parameter_count": config.expected_parameter_count,
        "model_class": "CURELiteCenteredMixedInteractionLevelSet",
    }


def expected_coverage_state_cmif_config(
    preflight: CoverageStateBoundedPreflight,
) -> CoverageStateCenteredMixedInteractionConfig:
    """Return the only CMIF config permitted by the bounded protocol."""

    if not isinstance(preflight, CoverageStateBoundedPreflight):
        raise TypeError("preflight must be CoverageStateBoundedPreflight")
    return CoverageStateCenteredMixedInteractionConfig(
        feature_channels=COVERAGE_STATE_CMIF_FORMAL_FEATURE_CHANNELS,
        feature_stride=COVERAGE_STATE_CMIF_FORMAL_FEATURE_STRIDE,
        width=COVERAGE_STATE_CMIF_FORMAL_WIDTH,
    )


def _validate_repo_relative_path(relative: str, *, name: str) -> str:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{name} repository path is invalid")
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or str(pure) != relative
    ):
        raise ValueError(f"{name} must be a normalized repository path")
    return relative


def _canonical_directory(relative: str, *, name: str) -> Path:
    relative = _validate_repo_relative_path(relative, name=name)
    candidate = _repository_root() / relative
    absolute = Path(os.path.abspath(candidate))
    if candidate.is_symlink():
        raise RuntimeError(f"{name} may not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if (
        resolved != absolute
        or not resolved.is_dir()
        or resolved.is_symlink()
        or not resolved.is_relative_to(_repository_root())
    ):
        raise RuntimeError(f"{name} must be a canonical directory")
    return resolved


def _strict_json(path: Path, *, name: str) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(
                    f"{name} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} must be a regular file")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                RuntimeError(
                    f"{name} contains non-finite value {item}"
                )
            ),
        )
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must contain a JSON object")
    return dict(value)


def _artifact_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("CMIF P0 artifact may not be a symlink")
        if not path.is_file() or path.name in {
            _INCOMPLETE,
            "COMPLETE.json",
        }:
            continue
        result[str(path.relative_to(root))] = file_sha256(path)
    return result


def _verify_complete(
    root: Path,
    *,
    replicate: str,
    expected_sha256: str,
    expected_fingerprint: str,
    expected_decision: str,
    expected_bounded: bool,
) -> dict[str, object]:
    if (
        (root / _INCOMPLETE).exists()
        or (root / "FAILURE.json").exists()
    ):
        raise PermissionError(f"CMIF P0 {replicate} is incomplete")
    path = root / "COMPLETE.json"
    if file_sha256(path) != expected_sha256:
        raise RuntimeError(f"CMIF P0 {replicate} COMPLETE changed")
    complete = _strict_json(path, name=f"CMIF P0 {replicate} COMPLETE")
    payload = dict(complete)
    fingerprint = payload.pop("complete_fingerprint", None)
    if (
        fingerprint != expected_fingerprint
        or stable_fingerprint(payload) != fingerprint
        or complete.get("status") != "complete"
        or complete.get("replicate") != replicate
        or complete.get("decision") != expected_decision
        or complete.get("bounded_400_authorized")
        is not expected_bounded
        or complete.get("formal_800_authorized") is not False
        or complete.get("full_CURE_authorized") is not False
        or complete.get("cross_backbone_authorized") is not False
        or complete.get("split") != "D_R"
        or complete.get("D_V_accessed") is not False
        or complete.get("D_T_accessed") is not False
        or complete.get("training_performed") is not False
    ):
        raise RuntimeError(f"CMIF P0 {replicate} contract changed")
    artifacts = complete.get("artifact_files")
    if (
        not isinstance(artifacts, Mapping)
        or dict(artifacts) != _artifact_hashes(root)
        or complete.get("artifact_file_count") != len(artifacts)
    ):
        raise RuntimeError(
            f"CMIF P0 {replicate} artifact population changed"
        )
    return complete


def _verify_receipt_fingerprint(
    payload: Mapping[str, object],
    *,
    expected: str,
    name: str,
) -> None:
    candidate = dict(payload)
    fingerprint = candidate.pop("receipt_fingerprint", None)
    if (
        fingerprint != expected
        or stable_fingerprint(candidate) != fingerprint
    ):
        raise RuntimeError(f"{name} fingerprint changed")


def _verify_persisted_cmif_p0_authorization(
) -> dict[str, object]:
    """Verify and summarize the fixed persisted r1/r2 authorization."""

    r1 = _canonical_directory(
        COVERAGE_STATE_CMIF_P0_R1_REPO_PATH,
        name="CMIF P0 r1",
    )
    r2 = _canonical_directory(
        COVERAGE_STATE_CMIF_P0_R2_REPO_PATH,
        name="CMIF P0 r2",
    )
    r1_complete = _verify_complete(
        r1,
        replicate="r1",
        expected_sha256=COVERAGE_STATE_CMIF_P0_R1_COMPLETE_SHA256,
        expected_fingerprint=(
            COVERAGE_STATE_CMIF_P0_R1_COMPLETE_FINGERPRINT
        ),
        expected_decision=COVERAGE_STATE_CMIF_P0_R1_DECISION,
        expected_bounded=False,
    )
    r2_complete = _verify_complete(
        r2,
        replicate="r2",
        expected_sha256=COVERAGE_STATE_CMIF_P0_R2_COMPLETE_SHA256,
        expected_fingerprint=(
            COVERAGE_STATE_CMIF_P0_R2_COMPLETE_FINGERPRINT
        ),
        expected_decision=COVERAGE_STATE_CMIF_P0_R2_DECISION,
        expected_bounded=True,
    )

    r1_core_path = r1 / "receipts" / "p0_core.json"
    r2_core_path = r2 / "receipts" / "p0_core.json"
    r1_core_bytes = r1_core_path.read_bytes()
    r2_core_bytes = r2_core_path.read_bytes()
    r1_core = _strict_json(r1_core_path, name="CMIF P0 r1 core")
    r2_core = _strict_json(r2_core_path, name="CMIF P0 r2 core")
    real_binding = r2_core.get("real_D_R_binding")
    if not isinstance(real_binding, Mapping):
        raise RuntimeError("CMIF P0 real D_R binding is malformed")
    if (
        r1_core_bytes != r2_core_bytes
        or sha256(r1_core_bytes).hexdigest()
        != COVERAGE_STATE_CMIF_P0_CORE_SHA256
        or sha256(r2_core_bytes).hexdigest()
        != COVERAGE_STATE_CMIF_P0_CORE_SHA256
        or r1_core != r2_core
        or r2_core.get("receipt_fingerprint")
        != COVERAGE_STATE_CMIF_P0_CORE_RECEIPT_FINGERPRINT
        or r2_core.get("eligible_for_replay") is not True
        or r2_core.get("training_authorized") is not False
        or r2_core.get("dataset") != "IRSTD-1K"
        or r2_core.get("split") != "D_R"
        or r2_core.get("bounded_population_fingerprint")
        != COVERAGE_STATE_CMIF_P0_BOUNDED_POPULATION_FINGERPRINT
        or r2_core.get("dataset_free_receipt_fingerprint")
        != COVERAGE_STATE_CMIF_P0_DATASET_FREE_RECEIPT_FINGERPRINT
        or real_binding.get("source_binding_fingerprint")
        != COVERAGE_STATE_CMIF_P0_SOURCE_BINDING_FINGERPRINT
    ):
        raise RuntimeError("CMIF P0 core replay binding changed")
    _verify_receipt_fingerprint(
        r2_core,
        expected=COVERAGE_STATE_CMIF_P0_CORE_RECEIPT_FINGERPRINT,
        name="CMIF P0 core",
    )

    replay_path = r2 / "receipts" / "replay_comparison.json"
    decision_path = r2 / "receipts" / "decision.json"
    if (
        file_sha256(replay_path)
        != COVERAGE_STATE_CMIF_P0_REPLAY_COMPARISON_SHA256
        or file_sha256(decision_path)
        != COVERAGE_STATE_CMIF_P0_DECISION_SHA256
    ):
        raise RuntimeError("CMIF P0 persisted authorization files changed")
    replay = _strict_json(replay_path, name="CMIF P0 replay comparison")
    decision = _strict_json(decision_path, name="CMIF P0 decision")
    _verify_receipt_fingerprint(
        replay,
        expected=(
            COVERAGE_STATE_CMIF_P0_REPLAY_COMPARISON_FINGERPRINT
        ),
        name="CMIF P0 replay comparison",
    )
    _verify_receipt_fingerprint(
        decision,
        expected=COVERAGE_STATE_CMIF_P0_DECISION_FINGERPRINT,
        name="CMIF P0 decision",
    )
    replay_checks = replay.get("checks")
    if (
        not isinstance(replay_checks, Mapping)
        or not replay_checks
        or not all(value is True for value in replay_checks.values())
        or replay.get("persisted_replay_passed") is not True
        or replay.get("persisted_canonical_bytes_identical") is not True
        or replay.get("persisted_file_sha256_identical") is not True
        or decision.get("status") != COVERAGE_STATE_CMIF_P0_R2_DECISION
        or decision.get("bounded_400_authorized") is not True
        or decision.get("training_authorized") is not True
        or decision.get("formal_800_authorized") is not False
        or decision.get("D_V_accessed") is not False
        or decision.get("D_T_accessed") is not False
        or decision.get("training_performed") is not False
    ):
        raise PermissionError("CMIF persisted P0 did not authorize training")

    payload = {
        "schema_version": "cure-lite-cmif-v17-persisted-p0-binding-v1",
        "r1_complete_fingerprint": (
            r1_complete["complete_fingerprint"]
        ),
        "r1_complete_sha256": (
            COVERAGE_STATE_CMIF_P0_R1_COMPLETE_SHA256
        ),
        "r2_complete_fingerprint": (
            r2_complete["complete_fingerprint"]
        ),
        "r2_complete_sha256": (
            COVERAGE_STATE_CMIF_P0_R2_COMPLETE_SHA256
        ),
        "p0_core_sha256": COVERAGE_STATE_CMIF_P0_CORE_SHA256,
        "p0_core_receipt_fingerprint": (
            COVERAGE_STATE_CMIF_P0_CORE_RECEIPT_FINGERPRINT
        ),
        "bounded_population_fingerprint": (
            COVERAGE_STATE_CMIF_P0_BOUNDED_POPULATION_FINGERPRINT
        ),
        "dataset_free_receipt_fingerprint": (
            COVERAGE_STATE_CMIF_P0_DATASET_FREE_RECEIPT_FINGERPRINT
        ),
        "source_binding_fingerprint": (
            COVERAGE_STATE_CMIF_P0_SOURCE_BINDING_FINGERPRINT
        ),
        "replay_comparison_fingerprint": (
            COVERAGE_STATE_CMIF_P0_REPLAY_COMPARISON_FINGERPRINT
        ),
        "replay_comparison_sha256": (
            COVERAGE_STATE_CMIF_P0_REPLAY_COMPARISON_SHA256
        ),
        "decision_fingerprint": (
            COVERAGE_STATE_CMIF_P0_DECISION_FINGERPRINT
        ),
        "decision_sha256": COVERAGE_STATE_CMIF_P0_DECISION_SHA256,
        "checks": dict(replay_checks),
        "training_authorized": True,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "training_performed": False,
    }
    payload["evidence_fingerprint"] = stable_fingerprint(payload)
    return payload


@dataclass(frozen=True, eq=False)
class CoverageStateCMIFBoundedRunAuthorization(
    CoverageStateRunAuthorization,
):
    """Bind one bounded preflight to generated and persisted CMIF gates."""

    preflight: CoverageStateBoundedPreflight
    dataset_free_receipt: CoverageStateCMIFDatasetFreeReceipt
    dataset_free_receipt_fingerprint: str
    p0_evidence_fingerprint: str
    p0_r1_complete_fingerprint: str
    p0_r2_complete_fingerprint: str
    p0_core_receipt_fingerprint: str
    p0_replay_comparison_fingerprint: str
    p0_decision_fingerprint: str
    p0_bounded_population_fingerprint: str
    implementation_binding: tuple[tuple[str, str], ...]
    implementation_fingerprint: str
    model_config_fingerprint: str
    expected_parameter_count: int
    objective_suite: tuple[str, ...]
    candidate_objective: str
    candidate_objective_policy: str
    coverage_policy: str
    interaction_policy: str
    energy_policy: str

    def __post_init__(self) -> None:
        expected_config = expected_coverage_state_cmif_config(
            self.preflight
        )
        expected_suite = tuple(
            value.value
            for value in COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES
        )
        if (
            not isinstance(self.preflight, CoverageStateBoundedPreflight)
            or not isinstance(
                self.dataset_free_receipt,
                CoverageStateCMIFDatasetFreeReceipt,
            )
            or self.dataset_free_receipt.receipt_fingerprint
            != self.dataset_free_receipt_fingerprint
            or self.dataset_free_receipt_fingerprint
            != COVERAGE_STATE_CMIF_P0_DATASET_FREE_RECEIPT_FINGERPRINT
            or self.p0_evidence_fingerprint
            != COVERAGE_STATE_CMIF_P0_EVIDENCE_FINGERPRINT
            or self.p0_r1_complete_fingerprint
            != COVERAGE_STATE_CMIF_P0_R1_COMPLETE_FINGERPRINT
            or self.p0_r2_complete_fingerprint
            != COVERAGE_STATE_CMIF_P0_R2_COMPLETE_FINGERPRINT
            or self.p0_core_receipt_fingerprint
            != COVERAGE_STATE_CMIF_P0_CORE_RECEIPT_FINGERPRINT
            or self.p0_replay_comparison_fingerprint
            != COVERAGE_STATE_CMIF_P0_REPLAY_COMPARISON_FINGERPRINT
            or self.p0_decision_fingerprint
            != COVERAGE_STATE_CMIF_P0_DECISION_FINGERPRINT
            or self.p0_bounded_population_fingerprint
            != self.preflight.population.population_fingerprint
            or self.p0_bounded_population_fingerprint
            != COVERAGE_STATE_CMIF_P0_BOUNDED_POPULATION_FINGERPRINT
            or self.implementation_binding
            != _current_cmif_implementation_binding()
            or self.implementation_fingerprint
            != stable_fingerprint(dict(self.implementation_binding))
            or self.model_config_fingerprint
            != stable_fingerprint(
                _cmif_model_config_payload(expected_config)
            )
            or self.expected_parameter_count
            != expected_config.expected_parameter_count
            or self.objective_suite != expected_suite
            or self.candidate_objective
            != CoverageStatePairObjective.SUPPORT_ORIENTED_RESPONSE_JOINT.value
            or self.objective_suite[0] != self.candidate_objective
            or self.candidate_objective_policy
            != CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY
            or self.coverage_policy
            != CSLF_PHASE_PRESERVING_COVERAGE_POLICY
            or self.interaction_policy != CMIF_INTERACTION_POLICY
            or self.energy_policy != CMIF_ENERGY_POLICY
        ):
            raise ValueError("CMIF bounded authorization binding changed")

    @property
    def training_authorized(self) -> bool:
        return (
            self.preflight.training_authorized
            and self.dataset_free_receipt.all_pass
            and self.p0_r2_complete_fingerprint
            == COVERAGE_STATE_CMIF_P0_R2_COMPLETE_FINGERPRINT
            and self.p0_bounded_population_fingerprint
            == self.preflight.population.population_fingerprint
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                COVERAGE_STATE_CMIF_BOUNDED_AUTHORIZATION_SCHEMA
            ),
            "scope": COVERAGE_STATE_BOUNDED_SCOPE,
            "runtime_splits": ["D_R"],
            "preflight_fingerprint": self.preflight.preflight_fingerprint,
            "population_fingerprint": (
                self.preflight.population.population_fingerprint
            ),
            "cache_fingerprint": (
                self.preflight.population.bounded_cache_fingerprint
            ),
            "schedule_fingerprint": (
                self.preflight.schedule.schedule_fingerprint
            ),
            "dataset_free_receipt_fingerprint": (
                self.dataset_free_receipt_fingerprint
            ),
            "persisted_p0": {
                "evidence_fingerprint": self.p0_evidence_fingerprint,
                "r1_complete_fingerprint": (
                    self.p0_r1_complete_fingerprint
                ),
                "r2_complete_fingerprint": (
                    self.p0_r2_complete_fingerprint
                ),
                "core_receipt_fingerprint": (
                    self.p0_core_receipt_fingerprint
                ),
                "replay_comparison_fingerprint": (
                    self.p0_replay_comparison_fingerprint
                ),
                "decision_fingerprint": (
                    self.p0_decision_fingerprint
                ),
                "bounded_population_fingerprint": (
                    self.p0_bounded_population_fingerprint
                ),
            },
            "implementation_binding": dict(self.implementation_binding),
            "implementation_fingerprint": (
                self.implementation_fingerprint
            ),
            "model_config_fingerprint": self.model_config_fingerprint,
            "model_class": "CURELiteCenteredMixedInteractionLevelSet",
            "expected_parameter_count": self.expected_parameter_count,
            "input_representation": CMIF_INPUT_REPRESENTATION,
            "coverage_policy": self.coverage_policy,
            "interaction_policy": self.interaction_policy,
            "energy_policy": self.energy_policy,
            "objective_suite": list(self.objective_suite),
            "candidate_objective": self.candidate_objective,
            "candidate_objective_policy": (
                self.candidate_objective_policy
            ),
            "checks": {
                "preflight_passed": self.preflight.training_authorized,
                "dataset_free_gate_passed": (
                    self.dataset_free_receipt.all_pass
                ),
                "persisted_p0_authorized": True,
                "p0_population_matches_preflight": (
                    self.p0_bounded_population_fingerprint
                    == self.preflight.population.population_fingerprint
                ),
            },
            "training_authorized": self.training_authorized,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_claim_supported": False,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
        }

    @cached_property
    def authorization_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify_unchanged(self) -> None:
        self.preflight.verify_unchanged()
        self.dataset_free_receipt.verify_unchanged()
        p0 = _verify_persisted_cmif_p0_authorization()
        expected_config = expected_coverage_state_cmif_config(
            self.preflight
        )
        if (
            p0["evidence_fingerprint"]
            != self.p0_evidence_fingerprint
            or p0["r2_complete_fingerprint"]
            != self.p0_r2_complete_fingerprint
            or p0["r1_complete_fingerprint"]
            != self.p0_r1_complete_fingerprint
            or p0["p0_core_receipt_fingerprint"]
            != self.p0_core_receipt_fingerprint
            or p0["replay_comparison_fingerprint"]
            != self.p0_replay_comparison_fingerprint
            or p0["decision_fingerprint"]
            != self.p0_decision_fingerprint
            or p0["bounded_population_fingerprint"]
            != self.p0_bounded_population_fingerprint
            or self.implementation_binding
            != _current_cmif_implementation_binding()
            or self.implementation_fingerprint
            != stable_fingerprint(dict(self.implementation_binding))
            or self.model_config_fingerprint
            != stable_fingerprint(
                _cmif_model_config_payload(expected_config)
            )
            or self.expected_parameter_count
            != expected_config.expected_parameter_count
            or stable_fingerprint(self.canonical_payload())
            != self.authorization_fingerprint
        ):
            raise RuntimeError(
                "CMIF bounded authorization changed after creation"
            )

    def verify_model_config(
        self,
        model_config: CoverageStateCenteredMixedInteractionConfig,
    ) -> None:
        if (
            not isinstance(
                model_config,
                CoverageStateCenteredMixedInteractionConfig,
            )
            or stable_fingerprint(
                _cmif_model_config_payload(model_config)
            )
            != self.model_config_fingerprint
            or model_config.expected_parameter_count
            != self.expected_parameter_count
        ):
            raise PermissionError(
                "CMIF authorization does not permit this model config"
            )

    def verify_for_run(
        self,
        *,
        cache: CoverageStateScalarCache,
        schedule: CoverageStateTrainingSchedule,
        scope: str,
    ) -> None:
        self.verify_unchanged()
        if (
            scope != COVERAGE_STATE_BOUNDED_SCOPE
            or cache.cache_fingerprint
            != self.preflight.population.bounded_cache_fingerprint
            or schedule.schedule_fingerprint
            != self.preflight.schedule.schedule_fingerprint
            or not self.training_authorized
        ):
            raise PermissionError(
                "CMIF authorization does not permit this training run"
            )


def prepare_coverage_state_cmif_bounded_run_authorization(
    preflight: CoverageStateBoundedPreflight,
    dataset_free_receipt: CoverageStateCMIFDatasetFreeReceipt,
) -> CoverageStateCMIFBoundedRunAuthorization:
    """Bind the fixed persisted P0 authorization to one real preflight."""

    if not isinstance(preflight, CoverageStateBoundedPreflight):
        raise TypeError("preflight must be CoverageStateBoundedPreflight")
    if not isinstance(
        dataset_free_receipt,
        CoverageStateCMIFDatasetFreeReceipt,
    ):
        raise TypeError(
            "dataset_free_receipt must be "
            "CoverageStateCMIFDatasetFreeReceipt"
        )
    preflight.verify_unchanged()
    dataset_free_receipt.verify_unchanged()
    p0 = _verify_persisted_cmif_p0_authorization()
    implementation_binding = _current_cmif_implementation_binding()
    model_config = expected_coverage_state_cmif_config(preflight)
    objective_suite = tuple(
        value.value
        for value in COVERAGE_STATE_SUPPORT_ORIENTED_MATCHED_OBJECTIVES
    )
    result = CoverageStateCMIFBoundedRunAuthorization(
        preflight=preflight,
        dataset_free_receipt=dataset_free_receipt,
        dataset_free_receipt_fingerprint=(
            dataset_free_receipt.receipt_fingerprint
        ),
        p0_evidence_fingerprint=str(p0["evidence_fingerprint"]),
        p0_r1_complete_fingerprint=str(
            p0["r1_complete_fingerprint"]
        ),
        p0_r2_complete_fingerprint=str(
            p0["r2_complete_fingerprint"]
        ),
        p0_core_receipt_fingerprint=str(
            p0["p0_core_receipt_fingerprint"]
        ),
        p0_replay_comparison_fingerprint=str(
            p0["replay_comparison_fingerprint"]
        ),
        p0_decision_fingerprint=str(p0["decision_fingerprint"]),
        p0_bounded_population_fingerprint=str(
            p0["bounded_population_fingerprint"]
        ),
        implementation_binding=implementation_binding,
        implementation_fingerprint=stable_fingerprint(
            dict(implementation_binding)
        ),
        model_config_fingerprint=stable_fingerprint(
            _cmif_model_config_payload(model_config)
        ),
        expected_parameter_count=model_config.expected_parameter_count,
        objective_suite=objective_suite,
        candidate_objective=objective_suite[0],
        candidate_objective_policy=CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY,
        coverage_policy=CSLF_PHASE_PRESERVING_COVERAGE_POLICY,
        interaction_policy=CMIF_INTERACTION_POLICY,
        energy_policy=CMIF_ENERGY_POLICY,
    )
    result.verify_unchanged()
    return result


def _cmif_bounded_result_checks(
    authorization: CoverageStateCMIFBoundedRunAuthorization,
    training: CoverageStateMatchedTrainingResult,
    diagnostics: tuple[
        tuple[str, CoverageStateZeroLevelEvaluationResult],
        ...,
    ],
) -> tuple[tuple[str, bool], ...]:
    """Reuse fixed execution checks and qualify only the SORR candidate."""

    generic = dict(
        _bounded_result_checks(
            authorization,
            training,
            diagnostics,
        )
    )
    generic.pop("zero_level_gates")
    models = training.models
    expected = authorization.objective_suite
    names = tuple(value.objective for value in training.results)
    diagnostic_by_name = dict(diagnostics)
    candidate = diagnostic_by_name.get(
        authorization.candidate_objective
    )
    controls = expected[1:]
    exact_models = (
        len(models) == 3
        and all(
            type(model)
            is CURELiteCenteredMixedInteractionLevelSet
            for _, model in models
        )
    )
    generic["authorized_model_config"] = all(
        type(model)
        is CURELiteCenteredMixedInteractionLevelSet
        and stable_fingerprint(
            _cmif_model_config_payload(model.config)
        )
        == authorization.model_config_fingerprint
        for _, model in models
    )
    generic.update(
        {
            "cmif_objective_suite": names == expected,
            "candidate_original_zero_level_gates": (
                candidate is not None
                and candidate.bounded_gate_passed
            ),
            "control_diagnostics_complete": (
                tuple(name for name, _ in diagnostics) == expected
                and all(name in diagnostic_by_name for name in controls)
                and len(controls) == 2
            ),
            "all_models_exact_cmif_class": exact_models,
            "all_models_same_cmif_config": (
                exact_models
                and len(
                    {
                        stable_fingerprint(
                            _cmif_model_config_payload(model.config)
                        )
                        for _, model in models
                    }
                )
                == 1
            ),
            "all_models_expected_parameter_count": (
                len(models) == 3
                and all(
                    sum(
                        parameter.numel()
                        for parameter in model.parameters()
                    )
                    == authorization.expected_parameter_count
                    for _, model in models
                )
            ),
            "candidate_policy_bound": (
                authorization.candidate_objective_policy
                == CSLF_SUPPORT_ORIENTED_RESPONSE_POLICY
            ),
            "coverage_policy_bound": (
                authorization.coverage_policy
                == CSLF_PHASE_PRESERVING_COVERAGE_POLICY
            ),
            "interaction_policy_bound": (
                authorization.interaction_policy
                == CMIF_INTERACTION_POLICY
            ),
            "energy_policy_bound": (
                authorization.energy_policy == CMIF_ENERGY_POLICY
            ),
            "phase_preserving_zero_evaluation": all(
                getattr(
                    getattr(diagnostic, "config", None),
                    "input_representation",
                    None,
                )
                == COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
                for _, diagnostic in diagnostics
            ),
            "persisted_p0_authorization_bound": (
                authorization.p0_r2_complete_fingerprint
                == COVERAGE_STATE_CMIF_P0_R2_COMPLETE_FINGERPRINT
                and authorization.p0_core_receipt_fingerprint
                == COVERAGE_STATE_CMIF_P0_CORE_RECEIPT_FINGERPRINT
                and authorization.p0_replay_comparison_fingerprint
                == COVERAGE_STATE_CMIF_P0_REPLAY_COMPARISON_FINGERPRINT
                and authorization.p0_decision_fingerprint
                == COVERAGE_STATE_CMIF_P0_DECISION_FINGERPRINT
                and authorization.p0_bounded_population_fingerprint
                == authorization.preflight.population.population_fingerprint
            ),
        }
    )
    return tuple(sorted(generic.items()))


@dataclass(frozen=True, eq=False)
class CoverageStateCMIFBoundedRunResult(CoverageStateBoundedRunResult):
    """CMIF candidate result with complete non-gating controls."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(
            self.authorization,
            CoverageStateCMIFBoundedRunAuthorization,
        ):
            raise ValueError("CMIF result requires CMIF authorization")
        if tuple(
            value.objective for value in self.training.results
        ) != self.authorization.objective_suite:
            raise ValueError("CMIF result objective suite changed")

    def verify_unchanged(self) -> None:
        self.authorization.verify_unchanged()
        self.training.verify_unchanged()
        if self.checks != _cmif_bounded_result_checks(
            self.authorization,
            self.training,
            self.diagnostics,
        ):
            raise RuntimeError("CMIF bounded result checks changed")

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        diagnostic_by_name = dict(self.diagnostics)
        controls = self.authorization.objective_suite[1:]
        return {
            "schema_version": COVERAGE_STATE_CMIF_BOUNDED_RESULT_SCHEMA,
            "runtime_splits": ["D_R"],
            "authorization_fingerprint": (
                self.authorization.authorization_fingerprint
            ),
            "persisted_p0": {
                "evidence_fingerprint": (
                    self.authorization.p0_evidence_fingerprint
                ),
                "r1_complete_fingerprint": (
                    self.authorization.p0_r1_complete_fingerprint
                ),
                "r2_complete_fingerprint": (
                    self.authorization.p0_r2_complete_fingerprint
                ),
                "core_receipt_fingerprint": (
                    self.authorization.p0_core_receipt_fingerprint
                ),
                "replay_comparison_fingerprint": (
                    self.authorization
                    .p0_replay_comparison_fingerprint
                ),
                "decision_fingerprint": (
                    self.authorization.p0_decision_fingerprint
                ),
            },
            "model_class": "CURELiteCenteredMixedInteractionLevelSet",
            "model_config_fingerprint": (
                self.authorization.model_config_fingerprint
            ),
            "expected_parameter_count": (
                self.authorization.expected_parameter_count
            ),
            "candidate_objective": (
                self.authorization.candidate_objective
            ),
            "candidate_diagnostic": diagnostic_by_name[
                self.authorization.candidate_objective
            ].canonical_payload(),
            "control_diagnostics": {
                name: diagnostic_by_name[name].canonical_payload()
                for name in controls
            },
            "training": self.training.canonical_payload(),
            "checks": dict(self.checks),
            "failed_checks": list(self.failed_checks),
            "bounded_gate_passed": self.bounded_gate_passed,
            "candidate_qualification_uses_original_gates_only": True,
            "control_outcomes_are_not_candidate_gates": True,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_claim_supported": False,
        }


def run_coverage_state_cmif_support_oriented_bounded_400(
    authorization: CoverageStateCMIFBoundedRunAuthorization,
    model_config: CoverageStateCenteredMixedInteractionConfig,
    *,
    device: torch.device | str,
) -> CoverageStateCMIFBoundedRunResult:
    """Run the fixed CMIF/SORR suite; caller owns artifact policy."""

    if not isinstance(
        authorization,
        CoverageStateCMIFBoundedRunAuthorization,
    ):
        raise TypeError("authorization must be CMIF bounded authorization")
    authorization.verify_model_config(model_config)
    preflight = authorization.preflight
    authorization.verify_for_run(
        cache=preflight.population.cache,
        schedule=preflight.schedule,
        scope=COVERAGE_STATE_BOUNDED_SCOPE,
    )
    with _deterministic_execution(device):
        training = (
            train_matched_coverage_state_cmif_support_oriented_objectives(
                model_config,
                preflight.population.cache,
                preflight.schedule,
                config=CoverageStateMatchedTrainingConfig(
                    seed=COVERAGE_STATE_BOUNDED_SEED
                ),
                device=device,
                authorization=authorization,
            )
        )
        if any(
            type(model)
            is not CURELiteCenteredMixedInteractionLevelSet
            for _, model in training.models
        ):
            raise RuntimeError(
                "matched CMIF training returned a different model class"
            )
        diagnostics = tuple(
            (
                name,
                evaluate_coverage_state_zero_level_checkpoint(
                    model.eval(),
                    preflight.population.cache,
                    device=device,
                    config=CoverageStateZeroLevelEvaluationConfig(
                        input_representation=(
                            COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
                        )
                    ),
                ),
            )
            for name, model in training.models
        )
    result = CoverageStateCMIFBoundedRunResult(
        authorization=authorization,
        training=training,
        diagnostics=diagnostics,
        checks=_cmif_bounded_result_checks(
            authorization,
            training,
            diagnostics,
        ),
    )
    result.verify_unchanged()
    return result


__all__ = [
    "COVERAGE_STATE_CMIF_BOUNDED_AUTHORIZATION_SCHEMA",
    "COVERAGE_STATE_CMIF_BOUNDED_IMPLEMENTATION_PATHS",
    "COVERAGE_STATE_CMIF_BOUNDED_RESULT_SCHEMA",
    "COVERAGE_STATE_CMIF_P0_BOUNDED_POPULATION_FINGERPRINT",
    "COVERAGE_STATE_CMIF_P0_CORE_RECEIPT_FINGERPRINT",
    "COVERAGE_STATE_CMIF_P0_R2_COMPLETE_FINGERPRINT",
    "CoverageStateCMIFBoundedRunAuthorization",
    "CoverageStateCMIFBoundedRunResult",
    "expected_coverage_state_cmif_config",
    "prepare_coverage_state_cmif_bounded_run_authorization",
    "run_coverage_state_cmif_support_oriented_bounded_400",
]
