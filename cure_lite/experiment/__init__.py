"""Fixed Gate-2 experiment helpers kept outside the method core.

The public path here is intentionally limited to exact ``D_R`` training and
``D_V`` calibration.  No final-test or Full-CURE entry point is exported.
"""

from .artifacts import load_decoder_artifact
from .cache_pipeline import (
    BaseCachePairContract,
    cache_d_r_states,
    cache_manifest_split,
    load_base_cache_pair_contract,
    load_d_r_cache_bundle,
    load_d_v_cache_bundle,
    materialize_base_cache_bundle,
)
from .formal_evaluation import (
    build_loaded_d_v_method_run,
    calibrate_paired_gate2,
    evaluate_paired_gate2,
)
from .formal_training import (
    PairedGate2TrainingConfig,
    run_paired_gate2_training,
    save_completed_decoder_run,
    summarize_gate2_training_support,
)
from .stage_a_runner import (
    LoadedStageARun,
    StageARunConfig,
    load_stage_a_run,
    run_stage_a,
    run_stage_a_from_base_caches,
)
from .deployment import (
    CalibratedCURELiteModel,
    CalibratedDeploymentReceipt,
    build_calibrated_cure_lite_model,
)
from .efficiency_evidence import (
    EfficiencyBinding,
    StageAEfficiencyReceipt,
    measure_stage_a_efficiency,
    replay_static_efficiency,
)
from .seed_registry import build_seed_registry_from_stage_a_run
from .training_pipeline import TrainingSupportRequirements, TrainingSupportSummary

__all__ = [
    "LoadedStageARun",
    "BaseCachePairContract",
    "CalibratedCURELiteModel",
    "CalibratedDeploymentReceipt",
    "EfficiencyBinding",
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
    "materialize_base_cache_bundle",
    "load_decoder_artifact",
    "load_stage_a_run",
    "measure_stage_a_efficiency",
    "run_paired_gate2_training",
    "replay_static_efficiency",
    "run_stage_a",
    "run_stage_a_from_base_caches",
    "save_completed_decoder_run",
    "summarize_gate2_training_support",
]
