from __future__ import annotations

from pathlib import Path

import cure_lite
import cure_lite.experiment as cure_experiment


_EXPECTED_PUBLIC_API = {
    "BranchBatch",
    "BranchPools",
    "BranchSupervision",
    "CURELiteDecoder",
    "CURELiteLoss",
    "CURELiteModel",
    "CURELiteOutput",
    "CURELiteTrainEngine",
    "CalibrationConfig",
    "CalibrationSample",
    "DecoderConfig",
    "FrozenBaseAdapter",
    "FrozenBaseOutput",
    "FrozenThresholdProtocol",
    "InstanceMap",
    "InterventionConfig",
    "LegalDeletion",
    "LossConfig",
    "MatchConfig",
    "MatchResult",
    "OccupancyConfig",
    "StateExample",
    "ThresholdSelection",
    "TrainingConfig",
    "build_epoch_factual_supervision_from_catalog",
    "build_factual_supervision",
    "build_factual_supervision_from_catalog",
    "build_occupancy",
    "build_synthetic_supervision",
    "choose_uniform_factual_gt_id",
    "choose_uniform_legal_deletion",
    "enumerate_legal_deletions",
    "instances_from_binary_mask",
    "iter_fixed_branch_batches",
    "match_components",
    "run_training_epoch",
    "select_anchor_threshold_by_miou",
    "select_residual_threshold",
}

_EXPECTED_EXPERIMENT_API = {
    "BaseCachePairContract",
    "CalibratedCURELiteModel",
    "CalibratedDeploymentReceipt",
    "EfficiencyBinding",
    "LoadedStageARun",
    "PairedGate2TrainingConfig",
    "StageARunConfig",
    "StageAEfficiencyReceipt",
    "TrainingSupportRequirements",
    "TrainingSupportSummary",
    "build_loaded_d_v_method_run",
    "build_seed_registry_from_stage_a_run",
    "build_calibrated_cure_lite_model",
    "cache_d_r_states",
    "cache_manifest_split",
    "calibrate_paired_gate2",
    "evaluate_paired_gate2",
    "load_base_cache_pair_contract",
    "load_d_r_cache_bundle",
    "load_d_v_cache_bundle",
    "load_decoder_artifact",
    "load_stage_a_run",
    "measure_stage_a_efficiency",
    "materialize_base_cache_bundle",
    "run_paired_gate2_training",
    "replay_static_efficiency",
    "run_stage_a",
    "run_stage_a_from_base_caches",
    "save_completed_decoder_run",
    "summarize_gate2_training_support",
}


def test_installed_package_resolves_to_nested_source_tree() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    package_root = repository_root / "cure_lite"
    assert Path(cure_lite.__file__).resolve() == package_root / "__init__.py"


def test_public_api_is_unique_resolvable_and_contains_main_lite_types() -> None:
    exported = tuple(cure_lite.__all__)
    assert len(exported) == len(set(exported))
    assert set(exported) == _EXPECTED_PUBLIC_API
    assert all(hasattr(cure_lite, name) for name in exported)
    assert {
        "CURELiteDecoder",
        "CURELiteLoss",
        "CURELiteModel",
        "CURELiteOutput",
        "FrozenBaseAdapter",
        "FrozenBaseOutput",
    } <= set(exported)
    assert not hasattr(cure_lite, "CUREModel")
    assert not hasattr(cure_lite, "CUREProtocol")
    assert not hasattr(cure_lite, "FormalBaseTrainingIdentity")


def test_public_root_does_not_import_adapter_specific_or_future_namespaces() -> None:
    package_root = Path(__file__).resolve().parents[1] / "cure_lite"
    root_source = (package_root / "__init__.py").read_text(encoding="utf-8")
    forbidden = (".provenance", ".counterfactual", ".cure", "MSHNet", "propensity")
    assert all(token not in root_source for token in forbidden)


def test_experiment_api_exposes_only_the_paired_gate_2_route() -> None:
    assert set(cure_experiment.__all__) == _EXPECTED_EXPERIMENT_API
    assert all(hasattr(cure_experiment, name) for name in _EXPECTED_EXPERIMENT_API)
    assert not hasattr(cure_experiment, "save_decoder_artifact")
    assert not hasattr(cure_experiment, "select_formal_residual_threshold")
    assert not hasattr(cure_experiment, "evaluate_formal_base_threshold")
