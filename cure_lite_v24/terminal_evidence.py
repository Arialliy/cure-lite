"""Mechanical terminal-model evidence verification for v24.

This module is deliberately cache-only.  It reconstructs an already verified
full-D_R scalar cache, loads final model state exclusively through the strict
safetensors decoder, rebuilds the frozen model topology through the exact arm
factories, and recomputes the receipt metrics on the device sealed by the
persistent run-start artifact.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Final, Mapping

import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.experiment.coverage_state_training import (
    coverage_state_model_fingerprint,
)
from cure_lite.paired_types import tensor_content_fingerprint
from cure_lite_v23.factory import build_pacre_vc_training_model
from cure_lite_v23.pacre_vc import (
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREVerifierCorrectedConfig,
)

from .artifact_io import load_terminal_safetensors_strict
from .bounded_runner import (
    GCR_PACRE_CANDIDATE_ARM,
    GCR_PACRE_CONTROL_ARM,
    GCR_PACRE_FORCED_G1_MODE,
    GCR_PACRE_NATIVE_MODE,
    GCRPACREBoundedEvaluation,
)
from .factory import (
    GCR_PACRE_FORMAL_FEATURE_CHANNELS,
    GCR_PACRE_FORMAL_FEATURE_STRIDE,
    GCR_PACRE_FORMAL_PARAMETER_COUNT,
    GCR_PACRE_FORMAL_WIDTH,
    GCR_PACRE_PARAMETER_NAMES,
    build_formal_gcr_pacre_training_model,
)
from .fixed_dr_evaluator import FrozenGCRPACREDREvaluator
from .formal_cache_artifacts import (
    VerifiedFormalCacheArtifact,
    load_formal_scalar_cache_artifact,
)
from .formal_training import (
    GCR_PACRE_FORMAL_TERMINAL_EVALUATION_SCHEMA,
)
from .gcr_pacre import (
    CURELiteGatedCommonResidualPACRELevelSet,
)


_STATE_SHAPES: Final = {
    "joint_state_weight": [32, 80, 5, 5],
    "joint_hidden_bias": [32],
    "scalar_energy_weight": [32],
}


def _resolved_device(value: object) -> torch.device:
    if not isinstance(value, str) or not value:
        raise TypeError("persistent run-start device must be non-empty text")
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as error:
        raise ValueError("persistent run-start device is invalid") from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "persistent run-start requires CUDA but CUDA is unavailable"
            )
        if device.index is None:
            raise ValueError(
                "persistent run-start CUDA device must have an explicit index"
            )
    elif device.type != "cpu":
        raise ValueError("terminal evidence permits only CPU or CUDA")
    return device


@contextmanager
def _deterministic_evaluation(device: torch.device):
    old_algorithms = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_cudnn_benchmark = torch.backends.cudnn.benchmark
    old_cudnn_deterministic = torch.backends.cudnn.deterministic
    old_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    old_tf32_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        yield
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        torch.use_deterministic_algorithms(
            old_algorithms,
            warn_only=old_warn_only,
        )
        torch.backends.cudnn.benchmark = old_cudnn_benchmark
        torch.backends.cudnn.deterministic = old_cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = old_tf32_matmul
        torch.backends.cudnn.allow_tf32 = old_tf32_cudnn


def _validate_terminal_state(
    path: str,
) -> dict[str, torch.Tensor]:
    state = load_terminal_safetensors_strict(path)
    if (
        set(state) != set(GCR_PACRE_PARAMETER_NAMES)
        or {
            name: list(tensor.shape)
            for name, tensor in state.items()
        }
        != _STATE_SHAPES
        or any(tensor.dtype != torch.float32 for tensor in state.values())
        or sum(tensor.numel() for tensor in state.values())
        != GCR_PACRE_FORMAL_PARAMETER_COUNT
    ):
        raise ValueError(
            "terminal safetensors is not the frozen 64/4/32/64064 state"
        )
    return state


def _seeded_bounded_model(arm: str) -> torch.nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        if arm == GCR_PACRE_CONTROL_ARM:
            model = build_pacre_vc_training_model(
                CoverageStatePACREVerifierCorrectedConfig(
                    feature_channels=GCR_PACRE_FORMAL_FEATURE_CHANNELS,
                    feature_stride=GCR_PACRE_FORMAL_FEATURE_STRIDE,
                    width=GCR_PACRE_FORMAL_WIDTH,
                )
            )
            if type(model) is not CURELitePACREVerifierCorrectedLevelSet:
                raise AssertionError("bounded control factory type changed")
            return model
        if arm == GCR_PACRE_CANDIDATE_ARM:
            model = build_formal_gcr_pacre_training_model()
            if type(model) is not CURELiteGatedCommonResidualPACRELevelSet:
                raise AssertionError("bounded candidate factory type changed")
            return model
    raise ValueError("unknown bounded arm")


def _initial_parameter_fingerprint(model: torch.nn.Module) -> str:
    rows = [
        {
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "numel": parameter.numel(),
            "content_fingerprint": tensor_content_fingerprint(parameter),
        }
        for name, parameter in model.named_parameters()
    ]
    if [str(row["name"]) for row in rows] != list(
        GCR_PACRE_PARAMETER_NAMES
    ):
        raise RuntimeError("bounded initial parameter inventory changed")
    return stable_fingerprint(rows)


def _bounded_metrics_payload(
    initial: GCRPACREBoundedEvaluation,
    terminal: GCRPACREBoundedEvaluation,
    forced_g1: GCRPACREBoundedEvaluation,
) -> dict[str, object]:
    return {
        "true_targets": terminal.true_targets,
        "recovered_anchor_misses": terminal.recovered_anchor_misses,
        "mIoU": terminal.mIoU,
        "nIoU": terminal.nIoU,
        "pd": terminal.pd,
        "retention": terminal.retention,
        "pixel_fa": terminal.pixel_fa,
        "raw_background_fa": terminal.raw_background_fa,
        "fp_components_per_mp": terminal.fp_components_per_mp,
        "budget_violation": terminal.budget_violation,
        "initial_PMOPE": initial.PMOPE,
        "terminal_PMOPE": terminal.PMOPE,
        "terminal_target_role_violation": (
            terminal.target_role_violation
        ),
        "terminal_background_role_violation": (
            terminal.background_role_violation
        ),
        "terminal_zero_crossed_target_states": (
            terminal.zero_crossed_target_states
        ),
        "terminal_false_completion_states": (
            terminal.false_completion_states
        ),
        "terminal_field_fingerprint": terminal.field_fingerprint,
        "terminal_role_prediction_fingerprint": (
            terminal.role_prediction_fingerprint
        ),
        "G1_PMOPE": forced_g1.PMOPE,
        "G1_target_role_violation": forced_g1.target_role_violation,
        "G1_background_role_violation": (
            forced_g1.background_role_violation
        ),
        "G1_zero_crossed_target_states": (
            forced_g1.zero_crossed_target_states
        ),
        "G1_false_completion_states": (
            forced_g1.false_completion_states
        ),
        "G1_field_fingerprint": forced_g1.field_fingerprint,
        "G1_role_prediction_fingerprint": (
            forced_g1.role_prediction_fingerprint
        ),
        "terminal_gate_distribution": terminal.gate_role_distribution,
        "G1_gate_distribution": forced_g1.gate_role_distribution,
        "gate_role_distributions_present": (
            terminal.gate_role_distributions_present
        ),
    }


def mechanically_recompute_bounded_arm(
    *,
    arm: str,
    terminal_artifact_path: str,
    expected_initial_model_fingerprint: str,
    expected_final_model_fingerprint: str,
    expected_initial_parameter_fingerprint: str,
    full_d_r_cache_artifact: VerifiedFormalCacheArtifact,
    requested_device: str,
) -> dict[str, object]:
    """Strictly load and recompute one bounded arm's complete metric payload."""

    device = _resolved_device(requested_device)
    cache = load_formal_scalar_cache_artifact(full_d_r_cache_artifact)
    initial_model = _seeded_bounded_model(arm)
    if (
        coverage_state_model_fingerprint(initial_model)
        != expected_initial_model_fingerprint
        or _initial_parameter_fingerprint(initial_model)
        != expected_initial_parameter_fingerprint
    ):
        raise PermissionError(
            f"{arm} initial model bytes differ from the exact seed-42 factory"
        )
    terminal_model = _seeded_bounded_model(arm)
    state = _validate_terminal_state(terminal_artifact_path)
    terminal_model.load_state_dict(state, strict=True)
    terminal_fingerprint = coverage_state_model_fingerprint(terminal_model)
    if terminal_fingerprint != expected_final_model_fingerprint:
        raise PermissionError(
            f"{arm} terminal safetensors model fingerprint changed"
        )
    initial_model = initial_model.to(device=device, dtype=torch.float32)
    terminal_model = terminal_model.to(device=device, dtype=torch.float32)
    if (
        coverage_state_model_fingerprint(initial_model)
        != expected_initial_model_fingerprint
        or coverage_state_model_fingerprint(terminal_model)
        != terminal_fingerprint
    ):
        raise RuntimeError(f"{arm} model state changed during device transfer")
    evaluator = FrozenGCRPACREDREvaluator()
    with _deterministic_evaluation(device):
        initial = evaluator.evaluate(
            initial_model,
            cache,
            arm=arm,
            checkpoint="initial",
            forward_mode=GCR_PACRE_NATIVE_MODE,
        )
        terminal = evaluator.evaluate(
            terminal_model,
            cache,
            arm=arm,
            checkpoint="terminal",
            forward_mode=GCR_PACRE_NATIVE_MODE,
        )
        forced_g1 = (
            terminal
            if arm == GCR_PACRE_CONTROL_ARM
            else evaluator.evaluate(
                terminal_model,
                cache,
                arm=arm,
                checkpoint="terminal",
                forward_mode=GCR_PACRE_FORCED_G1_MODE,
            )
        )
    if (
        coverage_state_model_fingerprint(initial_model)
        != expected_initial_model_fingerprint
        or coverage_state_model_fingerprint(terminal_model)
        != terminal_fingerprint
    ):
        raise RuntimeError(f"{arm} mechanical evaluator mutated model state")
    return _bounded_metrics_payload(initial, terminal, forced_g1)


def mechanically_recompute_formal_terminal(
    *,
    terminal_artifact_path: str,
    expected_final_model_fingerprint: str,
    cache_artifact: VerifiedFormalCacheArtifact,
    requested_device: str,
    seed: int,
    role: str,
) -> tuple[dict[str, object], str]:
    """Strictly load and recompute one Formal terminal D_R evaluation."""

    if (seed, role) not in {
        (42, "primary"),
        (43, "training_integrity_only"),
    }:
        raise ValueError("Formal seed/role pair changed")
    device = _resolved_device(requested_device)
    cache = load_formal_scalar_cache_artifact(cache_artifact)
    state = _validate_terminal_state(terminal_artifact_path)
    model = build_formal_gcr_pacre_training_model()
    model.load_state_dict(state, strict=True)
    model_fingerprint = coverage_state_model_fingerprint(model)
    if model_fingerprint != expected_final_model_fingerprint:
        raise PermissionError(
            "Formal terminal safetensors model fingerprint changed"
        )
    model = model.to(device=device, dtype=torch.float32)
    if coverage_state_model_fingerprint(model) != model_fingerprint:
        raise RuntimeError("Formal model state changed during device transfer")
    evaluator = FrozenGCRPACREDREvaluator()
    with _deterministic_evaluation(device):
        metrics = evaluator.evaluate_terminal_d_r(
            model,
            cache,
            seed=seed,
            role=role,
        )
    if coverage_state_model_fingerprint(model) != model_fingerprint:
        raise RuntimeError("Formal mechanical evaluator mutated model state")
    body = {
        "schema_version": GCR_PACRE_FORMAL_TERMINAL_EVALUATION_SCHEMA,
        "seed": seed,
        "role": role,
        "split": "D_R",
        "evaluator_fingerprint": evaluator.evaluator_fingerprint,
        "model_fingerprint": model_fingerprint,
        "metrics": metrics,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    return body, stable_fingerprint(body)


__all__ = [
    "mechanically_recompute_bounded_arm",
    "mechanically_recompute_formal_terminal",
]
