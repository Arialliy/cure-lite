from __future__ import annotations

from dataclasses import fields as dataclass_fields, replace
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cure_lite.cache.schema import canonical_json, stable_fingerprint
from cure_lite.calibration import CalibrationSample, FalseAlarmBudget
from cure_lite.calibration_ledger import (
    CalibrationCandidateLedger,
    CandidateEvaluation,
)
from cure_lite.config import MatchConfig, OccupancyConfig
from cure_lite.experiment.cache_pipeline import LoadedDVCacheBundle
from cure_lite.experiment.paired_formal_evaluation import (
    FrozenComparisonProtocol,
)
from cure_lite.frozen_base import module_state_fingerprint
from cure_lite.metrics import AggregateEvaluation
from cure_lite_v23 import formal_evaluation as module
from cure_lite_v23.formal_artifacts import (
    PACRE_VC_FORMAL_MODEL_ARTIFACT_SCHEMA,
    LoadedPACREVCFormalArtifact,
    formal_model_config_payload,
)
from cure_lite_v23.formal_evaluation import (
    PACRE_VC_FORMAL_BASE_THRESHOLD_GRID,
    PACREVCFormalModelBinding,
    bind_pacre_vc_formal_model,
    evaluate_pacre_vc_formal_d_v,
    fixed_pacre_vc_completion,
)
from cure_lite_v23.pacre_vc import (
    CURELitePACREVerifierCorrectedLevelSet,
    CoverageStatePACREVerifierCorrectedConfig,
)


def _formal_model() -> CURELitePACREVerifierCorrectedLevelSet:
    config = CoverageStatePACREVerifierCorrectedConfig(
        feature_channels=64,
        feature_stride=4,
        width=32,
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        return CURELitePACREVerifierCorrectedLevelSet(config)


def _metric(
    recovered: int,
    *,
    retained: int = 147,
    miou: float = 0.80,
    niou: float = 0.81,
    pixel_fa: float = 0.0,
    raw_background_fa: float = 0.0,
    fp_components_per_mp: float = 0.0,
    budget_violation: bool = False,
) -> AggregateEvaluation:
    gross_rmr = recovered / 23
    return AggregateEvaluation(
        pd=(retained + recovered) / 170,
        rmr=gross_rmr,
        gross_rmr=gross_rmr,
        net_rmr=gross_rmr,
        retention=retained / 147,
        reachable_rmr=0.0,
        oracle_upper_bound=0.0,
        overlap_supported_rmr=0.0,
        pixel_fa=pixel_fa,
        raw_background_fa=raw_background_fa,
        fp_components_per_mp=fp_components_per_mp,
        miou=miou,
        niou=niou,
        images=120,
        recovered_anchor_misses=recovered,
        net_recovered_anchor_misses=recovered,
        total_anchor_misses=23,
        retained_anchor_covered=retained,
        total_anchor_covered=147,
        recovered_reachable_anchor_misses=0,
        total_reachable_anchor_misses=0,
        budget_violation=budget_violation,
    )


def _install_synthetic_strict_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    CURELitePACREVerifierCorrectedLevelSet,
    LoadedDVCacheBundle,
    FrozenComparisonProtocol,
    PACREVCFormalModelBinding,
]:
    model = _formal_model()

    def zero_field(
        self: CURELitePACREVerifierCorrectedLevelSet,
        feature: torch.Tensor,
        occupancy: torch.Tensor,
    ) -> torch.Tensor:
        del self, feature
        return torch.zeros_like(occupancy, dtype=torch.float32)

    monkeypatch.setattr(
        CURELitePACREVerifierCorrectedLevelSet,
        "forward",
        zero_field,
    )
    monkeypatch.setattr(
        LoadedPACREVCFormalArtifact,
        "verify_unchanged",
        lambda self: None,
    )
    monkeypatch.setattr(
        LoadedDVCacheBundle,
        "verify_unchanged",
        lambda self: None,
    )
    monkeypatch.setattr(
        FrozenComparisonProtocol,
        "verify_bundle",
        lambda self, bundle: None,
    )
    monkeypatch.setattr(
        FrozenComparisonProtocol,
        "comparison_protocol_fingerprint",
        property(lambda self: "f" * 64),
    )
    monkeypatch.setattr(
        module,
        "verify_formal_training_ledger",
        lambda *args, **kwargs: "9" * 64,
    )

    receipt = {
        "schema_version": (
            PACRE_VC_FORMAL_MODEL_ARTIFACT_SCHEMA
        ),
        "artifact_fingerprint": "a" * 64,
        "formal_result_fingerprint": "f" * 64,
        "training_result_fingerprint": "b" * 64,
        "formal_training_ledger": {},
        "authorization_fingerprint": "c" * 64,
        "source_closure_fingerprint": "d" * 64,
        "module_state_fingerprint": module_state_fingerprint(model),
        "model_config_fingerprint": stable_fingerprint(
            formal_model_config_payload(model.config)
        ),
        "final_checkpoint_only": True,
        "optimizer_state_saved": False,
        "intermediate_checkpoint_saved": False,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
        "performance_evaluation_performed": False,
    }
    artifact = object.__new__(LoadedPACREVCFormalArtifact)
    object.__setattr__(artifact, "directory", Path("/synthetic/formal"))
    object.__setattr__(artifact, "model", model)
    object.__setattr__(artifact, "artifact_json", canonical_json(receipt))
    object.__setattr__(artifact, "_seal", object())

    class _SyntheticVerifiedTerminal:
        def __init__(
            self,
            bound_artifact: LoadedPACREVCFormalArtifact,
        ) -> None:
            self.artifact = bound_artifact

        def verify_unchanged(self) -> None:
            self.artifact.verify_unchanged()

    monkeypatch.setattr(
        module,
        "VerifiedPACREVCFormalTerminal",
        _SyntheticVerifiedTerminal,
    )
    terminal = _SyntheticVerifiedTerminal(artifact)

    rows = []
    for index in range(120):
        probability = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
        gt = torch.zeros((1, 4, 4), dtype=torch.bool)
        gt[0, 0, 0] = True
        rows.append(
            SimpleNamespace(
                sample_id=f"synthetic-{index:03d}",
                base_output=SimpleNamespace(
                    feature=torch.zeros(
                        (1, 64, 1, 1),
                        dtype=torch.float32,
                    ),
                    probability=probability,
                ),
                gt_mask=gt,
            )
        )
    bundle = object.__new__(LoadedDVCacheBundle)
    object.__setattr__(bundle, "rows", tuple(rows))
    object.__setattr__(
        bundle,
        "split_manifest_fingerprint",
        "1" * 64,
    )
    object.__setattr__(
        bundle,
        "split_manifest_file_sha256",
        "2" * 64,
    )
    object.__setattr__(
        bundle,
        "preprocessing_fingerprint",
        "3" * 64,
    )
    object.__setattr__(bundle, "base_fingerprint", "4" * 64)
    object.__setattr__(bundle, "base_state_fingerprint", "5" * 64)
    object.__setattr__(bundle, "base_index_fingerprint", "6" * 64)
    object.__setattr__(bundle, "d_v_image_fingerprint", "7" * 64)
    object.__setattr__(bundle, "d_v_gt_fingerprint", "8" * 64)

    protocol = object.__new__(FrozenComparisonProtocol)
    object.__setattr__(
        protocol,
        "occupancy_config",
        OccupancyConfig(
            threshold=0.72,
            connectivity=8,
            min_component_area=1,
        ),
    )
    object.__setattr__(
        protocol,
        "match_config",
        MatchConfig(
            max_distance=3.0,
            distance_quantization=1_000_000,
            iou_quantization=1_000_000,
        ),
    )
    object.__setattr__(
        protocol,
        "budget",
        FalseAlarmBudget(1.0e-4, 100.0, 1.0e-4, 0.99),
    )
    object.__setattr__(
        protocol,
        "residual_thresholds",
        PACRE_VC_FORMAL_BASE_THRESHOLD_GRID,
    )
    binding = bind_pacre_vc_formal_model(
        terminal,
        protocol,
        bundle,
    )
    return model, bundle, protocol, binding


def _install_points(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cure: AggregateEvaluation,
) -> None:
    base_at_a = _metric(0, miou=0.78, niou=0.79)
    base_at_b = _metric(1, miou=0.80, niou=0.81)
    ledger = CalibrationCandidateLedger(
        base_method="Base@B",
        anchor_threshold=0.72,
        anchor_metrics=base_at_a,
        entries=tuple(
            CandidateEvaluation(
                method="Base@B",
                mode="base",
                threshold=threshold,
                metrics=base_at_b if threshold == 0.60 else base_at_a,
            )
            for threshold in PACRE_VC_FORMAL_BASE_THRESHOLD_GRID
        ),
    )
    points = module._PACREVCFixedOperatingPoints(
        base_at_a=base_at_a,
        base_at_b=base_at_b,
        base_at_a_plus_cure=cure,
        base_at_b_selected_threshold=0.60,
        base_candidate_ledger=ledger,
    )
    monkeypatch.setattr(
        module,
        "_evaluate_fixed_operating_points",
        lambda *args, **kwargs: points,
    )
    monkeypatch.setattr(
        module,
        "_verified_stage_a_base_grid",
        lambda: PACRE_VC_FORMAL_BASE_THRESHOLD_GRID,
    )


def test_fixed_completion_uses_zero_threshold_strictly_and_hard_masks_base() -> None:
    field = torch.tensor(
        [[[[-1.0, -0.0, 0.0, 1.0, -2.0]]]],
        dtype=torch.float32,
    )
    occupancy = torch.tensor(
        [[[[False, False, False, False, True]]]],
        dtype=torch.bool,
    )

    assert fixed_pacre_vc_completion(field, occupancy).tolist() == [
        [[[True, False, False, False, False]]]
    ]
    with pytest.raises(ValueError, match="finite"):
        fixed_pacre_vc_completion(
            torch.full((1, 1, 1, 1), float("nan")),
            torch.zeros((1, 1, 1, 1), dtype=torch.bool),
        )


def test_low_level_core_searches_only_base_b_and_uses_hard_union() -> None:
    base = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    base[0, 0, 1, 1] = 0.9
    base[0, 0, 6, 6] = 0.6
    gt = torch.zeros((1, 1, 8, 8), dtype=torch.bool)
    gt[0, 0, 1, 1] = True
    gt[0, 0, 6, 6] = True
    completion = torch.zeros_like(base)
    completion[0, 0, 6, 6] = 1.0

    points = module._evaluate_fixed_operating_points(
        (
            CalibrationSample(
                "synthetic",
                base,
                torch.zeros_like(base),
                gt,
            ),
        ),
        (
            CalibrationSample(
                "synthetic",
                base,
                completion,
                gt,
            ),
        ),
        occupancy_config=OccupancyConfig(threshold=0.72),
        match_config=MatchConfig(),
        base_threshold_grid=PACRE_VC_FORMAL_BASE_THRESHOLD_GRID,
        budget=FalseAlarmBudget(1.0e-4, 100.0, 1.0e-4, 0.99),
    )

    assert points.base_at_b_selected_threshold == 0.6
    assert points.base_candidate_ledger.methods == ("Base@B",)
    assert len(points.base_candidate_ledger.entries) == 51
    assert tuple(
        entry.threshold for entry in points.base_candidate_ledger.entries
    ) == PACRE_VC_FORMAL_BASE_THRESHOLD_GRID
    assert points.base_candidate_ledger.entries[-1].threshold == 1.0
    assert points.base_at_a.recovered_anchor_misses == 0
    assert points.base_at_a_plus_cure.recovered_anchor_misses == 1


def test_binding_requires_exact_loaded_artifact_and_cannot_be_resealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, bundle, protocol, binding = _install_synthetic_strict_sources(
        monkeypatch
    )
    assert binding.seed == 42
    assert binding.completed_updates == 32_000
    assert binding.trained_from_scratch is True
    assert binding.resumed is False
    binding.verify_inputs(model, bundle, protocol)

    raw_artifact = binding._sealed_sources().terminal.artifact
    with pytest.raises(TypeError, match="verifier-issued"):
        bind_pacre_vc_formal_model(
            raw_artifact,  # type: ignore[arg-type]
            protocol,
            bundle,
        )
    with pytest.raises(TypeError, match="verifier-issued"):
        bind_pacre_vc_formal_model(
            object(),  # type: ignore[arg-type]
            protocol,
            bundle,
        )
    with pytest.raises(PermissionError, match="strict factory"):
        replace(binding, _seal=None)
    substituted_model = _formal_model()
    substituted_model.load_state_dict(model.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="substituted"):
        binding.verify_inputs(substituted_model, bundle, protocol)

    with pytest.raises(TypeError, match="exact CURELite"):
        evaluate_pacre_vc_formal_d_v(
            object(),  # type: ignore[arg-type]
            bundle,
            protocol,
            binding,
        )
    with pytest.raises(TypeError, match="exact LoadedDVCacheBundle"):
        evaluate_pacre_vc_formal_d_v(
            model,
            object(),  # type: ignore[arg-type]
            protocol,
            binding,
        )
    with pytest.raises(TypeError, match="exact FrozenComparisonProtocol"):
        evaluate_pacre_vc_formal_d_v(
            model,
            bundle,
            object(),  # type: ignore[arg-type]
            binding,
        )


@pytest.mark.parametrize(
    ("cure_recovered", "expected_pass"),
    ((2, True), (1, False)),
)
def test_best_valid_base_strict_improvement_allows_plus_one_but_not_plus_zero(
    monkeypatch: pytest.MonkeyPatch,
    cure_recovered: int,
    expected_pass: bool,
) -> None:
    model, bundle, protocol, binding = _install_synthetic_strict_sources(
        monkeypatch
    )
    _install_points(
        monkeypatch,
        cure=_metric(
            cure_recovered,
            miou=0.80,
            niou=0.81,
        ),
    )

    result = evaluate_pacre_vc_formal_d_v(
        model,
        bundle,
        protocol,
        binding,
        batch_size=8,
    )
    payload = result.canonical_payload()

    assert result.gate_passed is expected_pass
    assert result.true_target_margin == cure_recovered - 1
    assert result.recovered_anchor_miss_margin == cure_recovered - 1
    assert payload["method"] == "PACRE-VC-v23"
    assert payload["D_V_adaptive"] is True
    assert payload["D_T_payload_accessed"] is False
    assert payload["batch_size"] == 8
    base_selection = payload["Base@B_selection"]
    assert base_selection["candidate_count"] == 51
    assert (
        base_selection["candidate_ledger"]["candidate_count"] == 51
    )
    assert len(base_selection["candidate_ledger"]["entries"]) == 51
    assert payload["output_contract"]["field_threshold"] == 0.0
    assert (
        payload["output_contract"][
            "PACRE_threshold_search_performed"
        ]
        is False
    )
    for name in ("Base@A", "Base@B", "Base@A+CURE"):
        operating_point = payload["operating_points"][name]
        assert set(operating_point["aggregate_evaluation"]) == {
            item.name for item in dataclass_fields(AggregateEvaluation)
        }
        assert {
            "true_targets",
            "Pd",
            "mIoU",
            "nIoU",
            "pixel_Fa",
            "raw_background_Fa",
            "false_positive_components_per_megapixel",
            "recovered_anchor_misses",
            "retention",
            "budget_violation",
        } <= set(operating_point["summary"])


@pytest.mark.parametrize(
    "cure",
    (
        _metric(
            2,
            miou=0.80,
            niou=0.81,
            pixel_fa=1.00001e-4,
            budget_violation=True,
        ),
        _metric(2, miou=0.799, niou=0.81),
        _metric(2, miou=0.80, niou=0.809),
        _metric(
            2,
            miou=0.80,
            niou=0.81,
            raw_background_fa=1.00001e-4,
            budget_violation=True,
        ),
        _metric(
            2,
            miou=0.80,
            niou=0.81,
            fp_components_per_mp=100.00001,
            budget_violation=True,
        ),
        _metric(
            3,
            retained=146,
            miou=0.80,
            niou=0.81,
        ),
    ),
)
def test_false_addition_and_iou_guards_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    cure: AggregateEvaluation,
) -> None:
    model, bundle, protocol, binding = _install_synthetic_strict_sources(
        monkeypatch
    )
    _install_points(monkeypatch, cure=cure)

    result = evaluate_pacre_vc_formal_d_v(
        model,
        bundle,
        protocol,
        binding,
        batch_size=8,
    )

    assert result.gate_passed is False
    assert result.failed_checks
    assert result.canonical_payload()["authorizes_D_T"] is False


def test_inclusive_safety_boundaries_and_iou_equality_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, bundle, protocol, binding = _install_synthetic_strict_sources(
        monkeypatch
    )
    _install_points(
        monkeypatch,
        cure=_metric(
            2,
            miou=0.80,
            niou=0.81,
            pixel_fa=1.0e-4,
            raw_background_fa=1.0e-4,
            fp_components_per_mp=100.0,
        ),
    )
    result = evaluate_pacre_vc_formal_d_v(
        model,
        bundle,
        protocol,
        binding,
        batch_size=8,
    )
    assert result.gate_passed is True
    assert result.true_target_margin == 1
    assert result.recovered_anchor_miss_margin == 1


def test_formal_evaluator_rejects_nonfrozen_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, bundle, protocol, binding = _install_synthetic_strict_sources(
        monkeypatch
    )
    with pytest.raises(ValueError, match="batch_size=8"):
        evaluate_pacre_vc_formal_d_v(
            model,
            bundle,
            protocol,
            binding,
            batch_size=16,
        )


def test_public_api_has_exact_inputs_no_pacre_threshold_and_no_writers() -> None:
    bind = inspect.signature(bind_pacre_vc_formal_model)
    evaluate = inspect.signature(evaluate_pacre_vc_formal_d_v)
    assert tuple(bind.parameters) == (
        "terminal",
        "comparison_protocol",
        "bundle",
    )
    assert tuple(evaluate.parameters) == (
        "model",
        "bundle",
        "comparison_protocol",
        "model_binding",
        "batch_size",
    )
    assert "threshold" not in evaluate.parameters
    source = inspect.getsource(module)
    for writer in (
        "write_text(",
        "write_bytes(",
        "json.dump(",
        ".open(\"w",
        ".open('w",
    ):
        assert writer not in source
