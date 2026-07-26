from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cure_lite.calibration import FalseAlarmBudget
from cure_lite.experiment.formal_evaluation import (
    FormalDVThresholdReceipt,
    LoadedDVMethodRun,
)
from cure_lite.experiment.paired_artifacts import LoadedPairedDecoderArtifact
from cure_lite.experiment import paired_formal_evaluation as module
from cure_lite.experiment.paired_formal_evaluation import (
    FORMAL_DV_ANCHOR_COVERED,
    FORMAL_DV_ANCHOR_MISSES,
    FORMAL_DV_IMAGES,
    FORMAL_DV_TOTAL_TARGETS,
    adapt_frozen_historical_method_evidence,
    load_frozen_comparison_protocol,
    load_paired_formal_d_v_result,
    result_from_selected_paired_evaluation,
    save_paired_formal_d_v_result,
)
from cure_lite.metrics import (
    AggregateEvaluation,
    FORMAL_STAGE_A_METRIC_FIELDS,
    formal_stage_a_metrics_payload,
)

_PROTOCOL_PATH = (
    Path(__file__).resolve().parents[1]
    / "protocols"
    / "IRSTD-1K"
    / "paired_formal_evaluation_v1"
    / "config.json"
)


def _protocol():
    return load_frozen_comparison_protocol(_PROTOCOL_PATH)


def _historical_sources(method: str, seed: int = 42) -> dict[str, object]:
    binding = _protocol().historical_binding(seed)
    return {
        "comparison_protocol": _protocol(),
        "source_results_fingerprint": binding.results_fingerprint,
        "source_protocol_fingerprint": (
            binding.protocol_fingerprint_for(method)
        ),
        "source_run_config_fingerprint": (
            binding.run_config_fingerprint
        ),
        "source_config_file_sha256": binding.config_file_sha256,
        "source_calibration_receipt_fingerprint": (
            binding.calibration_receipt_fingerprint
        ),
        "source_complete_fingerprint": binding.complete_fingerprint,
    }


def _frozen_historical_metrics(method: str) -> dict[str, float | bool]:
    return {
        "Base@B": {
            "budget_violation": False,
            "fp_components_per_mp": 3.4332275390625,
            "miou": 0.6076294277929155,
            "niou": 0.5640138505062713,
            "pd": 150 / 170,
            "pixel_fa": 2.4286905924479165e-05,
            "raw_background_fa": 8.20159912109375e-05,
            "retention": 1.0,
        },
        "F": {
            "budget_violation": False,
            "fp_components_per_mp": 8.900960286458334,
            "miou": 0.5948505474992601,
            "niou": 0.5340285324302402,
            "pd": 154 / 170,
            "pixel_fa": 3.31878662109375e-05,
            "raw_background_fa": 9.167989095052084e-05,
            "retention": 1.0,
        },
        "F×": {
            "budget_violation": False,
            "fp_components_per_mp": 7.756551106770833,
            "miou": 0.5891701000588582,
            "niou": 0.5285164505007406,
            "pd": 149 / 170,
            "pixel_fa": 3.4968058268229164e-05,
            "raw_background_fa": 9.409586588541667e-05,
            "retention": 1.0,
        },
        "U": {
            "budget_violation": False,
            "fp_components_per_mp": 6.103515625,
            "miou": 0.5970597059705971,
            "niou": 0.5359605406098087,
            "pd": 151 / 170,
            "pixel_fa": 3.255208333333333e-05,
            "raw_background_fa": 8.58306884765625e-05,
            "retention": 1.0,
        },
    }[method]


def _metrics(
    *,
    recovered: int = 8,
    retained: int = FORMAL_DV_ANCHOR_COVERED,
    pixel_fa: float = 1e-5,
    raw_background_fa: float = 2e-5,
    fp_components_per_mp: float = 10.0,
) -> AggregateEvaluation:
    true_targets = retained + recovered
    net_recovered = true_targets - FORMAL_DV_ANCHOR_COVERED
    return AggregateEvaluation(
        pd=true_targets / FORMAL_DV_TOTAL_TARGETS,
        rmr=recovered / FORMAL_DV_ANCHOR_MISSES,
        gross_rmr=recovered / FORMAL_DV_ANCHOR_MISSES,
        net_rmr=net_recovered / FORMAL_DV_ANCHOR_MISSES,
        retention=retained / FORMAL_DV_ANCHOR_COVERED,
        reachable_rmr=recovered / FORMAL_DV_ANCHOR_MISSES,
        oracle_upper_bound=1.0,
        overlap_supported_rmr=recovered / FORMAL_DV_ANCHOR_MISSES,
        pixel_fa=pixel_fa,
        raw_background_fa=raw_background_fa,
        fp_components_per_mp=fp_components_per_mp,
        miou=0.5,
        niou=0.4,
        images=FORMAL_DV_IMAGES,
        recovered_anchor_misses=recovered,
        net_recovered_anchor_misses=net_recovered,
        total_anchor_misses=FORMAL_DV_ANCHOR_MISSES,
        retained_anchor_covered=retained,
        total_anchor_covered=FORMAL_DV_ANCHOR_COVERED,
        recovered_reachable_anchor_misses=recovered,
        total_reachable_anchor_misses=FORMAL_DV_ANCHOR_MISSES,
        budget_violation=False,
    )


def _result(
    *,
    method: str = "paired_difference",
    seed: int = 42,
    metrics: AggregateEvaluation | None = None,
):
    return module._new_result(
        method=method,
        seed=seed,
        comparison_protocol_fingerprint=(
            _protocol().comparison_protocol_fingerprint
        ),
        selected_threshold=0.75,
        metrics=_metrics() if metrics is None else metrics,
        budget=FalseAlarmBudget(
            pixel_fa_budget=1e-4,
            component_fa_per_mp_budget=100.0,
            raw_background_fa_budget=1e-4,
            minimum_retention=0.99,
        ),
        d_v_run_fingerprint="0" * 64,
        threshold_protocol_fingerprint="1" * 64,
        manifest_fingerprint="2" * 64,
        manifest_file_sha256="3" * 64,
        preprocessing_fingerprint="4" * 64,
        base_fingerprint="5" * 64,
        d_v_base_index_fingerprint="6" * 64,
        d_v_base_index_sha256="7" * 64,
        d_v_image_fingerprint="8" * 64,
        d_v_gt_fingerprint="9" * 64,
        residual_samples_fingerprint="a" * 64,
        decoder_artifact_fingerprint="b" * 64,
        decoder_receipt_sha256="c" * 64,
        decoder_state_fingerprint="d" * 64,
        formal_protocol_fingerprint="e" * 64,
        paired_objective_fingerprint="f" * 64,
        pair_catalog_fingerprint="0" * 64,
        paired_schedule_fingerprint="1" * 64,
        formal_schedule_fingerprint="2" * 64,
        runtime_input_fingerprint="7" * 64,
        control_preflight_fingerprint="3" * 64,
        control_provider_fingerprint=(
            None if method == "paired_difference" else "8" * 64
        ),
        method_contract_fingerprint="4" * 64,
        paired_criterion_fingerprint="5" * 64,
        method_objective_fingerprint="6" * 64,
    )


def _fake_strict_sources(
    monkeypatch: pytest.MonkeyPatch,
    metrics: AggregateEvaluation,
    *,
    method: str = "paired_difference",
    seed: int = 42,
):
    config_payload = {
        "paired_criterion": {
            "id": "balanced_pre_mask_score_difference_regression_v1"
        },
        "method_objective": {
            "family": "coupled_paired_difference",
            "method": method,
        },
    }
    config = SimpleNamespace(
        method=method,
        seed=seed,
        formal_protocol_fingerprint="0" * 64,
        paired_objective_fingerprint="1" * 64,
        pair_catalog_fingerprint="2" * 64,
        paired_schedule_fingerprint="3" * 64,
        formal_schedule_fingerprint="4" * 64,
        runtime_input_fingerprint="a" * 64,
        control_preflight_fingerprint="5" * 64,
        control_provider_fingerprint=(
            None if method == "paired_difference" else "b" * 64
        ),
        method_contract_fingerprint="6" * 64,
        canonical_payload=lambda: config_payload,
    )
    artifact = object.__new__(LoadedPairedDecoderArtifact)
    for name, value in {
        "config": config,
        "artifact_fingerprint": "7" * 64,
        "receipt_sha256": "8" * 64,
        "decoder_state_fingerprint": "9" * 64,
    }.items():
        object.__setattr__(artifact, name, value)
    bundle = SimpleNamespace(
        split_manifest_fingerprint="a" * 64,
        split_manifest_file_sha256="b" * 64,
        preprocessing_fingerprint="c" * 64,
        base_fingerprint="d" * 64,
        base_index_fingerprint="e" * 64,
        base_index_sha256="f" * 64,
        d_v_image_fingerprint="0" * 64,
        d_v_gt_fingerprint="1" * 64,
    )
    run = object.__new__(LoadedDVMethodRun)
    for name, value in {
        "artifact": artifact,
        "bundle": bundle,
        "residual_samples_fingerprint": "2" * 64,
    }.items():
        object.__setattr__(run, name, value)
    budget = FalseAlarmBudget(
        pixel_fa_budget=1e-4,
        component_fa_per_mp_budget=100.0,
        raw_background_fa_budget=1e-4,
        minimum_retention=0.99,
    )
    protocol = SimpleNamespace(
        selected_metrics=metrics,
        budget=budget,
        selected_threshold=0.75,
        receipt_fingerprint="3" * 64,
    )
    receipt = object.__new__(FormalDVThresholdReceipt)
    for name, value in {
        "decoder_variant": method,
        "global_seed": seed,
        "protocol": protocol,
    }.items():
        object.__setattr__(receipt, name, value)

    monkeypatch.setattr(
        LoadedDVMethodRun,
        "verify_unchanged",
        lambda self: None,
    )
    monkeypatch.setattr(
        LoadedPairedDecoderArtifact,
        "verify_unchanged",
        lambda self: None,
    )
    monkeypatch.setattr(
        LoadedDVMethodRun,
        "run_fingerprint",
        property(lambda self: "4" * 64),
    )
    monkeypatch.setattr(
        module,
        "evaluate_formal_residual_threshold",
        lambda actual_run, actual_receipt: metrics,
    )
    monkeypatch.setattr(
        module.FrozenComparisonProtocol,
        "verify_selected_receipt",
        lambda self, actual_run, actual_receipt: None,
    )
    return run, receipt


def test_result_contains_complete_metrics_bindings_and_decision_evidence() -> None:
    result = _result()
    payload = result.canonical_payload()
    evidence = result.to_formal_method_evidence()

    assert payload["runtime_split"] == "D_V"
    assert payload["D_T_accessed"] is False
    assert set(payload["aggregate_evaluation"]) == {
        field.name for field in __import__("dataclasses").fields(AggregateEvaluation)
    }
    assert payload["bindings"]["formal_schedule_fingerprint"] == "2" * 64
    assert payload["bindings"]["paired_criterion_fingerprint"] == "5" * 64
    assert evidence.method == "paired_difference"
    assert evidence.seed == 42
    assert evidence.total_targets == 170
    assert evidence.true_targets == 155
    assert evidence.pd == 155 / 170
    assert evidence.total_anchor_misses == 23
    assert evidence.recovered_anchor_misses == 8
    assert evidence.result_fingerprint == result.result_fingerprint
    assert payload["formal_method_evidence"] == evidence.canonical_payload()


def test_population_pd_and_recovery_denominators_are_exact() -> None:
    with pytest.raises(ValueError, match="exactly 120 images"):
        _result(metrics=replace(_metrics(), images=100))
    with pytest.raises(ValueError, match="denominator.*23"):
        _result(
            metrics=replace(
                _metrics(),
                total_anchor_misses=22,
            )
        )
    with pytest.raises(ValueError, match="Pd"):
        _result(metrics=replace(_metrics(), pd=0.9))
    with pytest.raises(ValueError, match="anchor-covered.*147"):
        _result(
            metrics=replace(
                _metrics(),
                total_anchor_covered=146,
            )
        )


def test_frozen_budget_is_checked_on_construction_and_load() -> None:
    with pytest.raises(ValueError, match="budget"):
        _result(metrics=_metrics(pixel_fa=2e-4))
    with pytest.raises(ValueError, match="budget"):
        _result(metrics=replace(_metrics(), budget_violation=True))


def test_strict_selected_evaluation_binds_method_seed_and_current_config_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = _metrics()
    run, receipt = _fake_strict_sources(monkeypatch, metrics)
    result = result_from_selected_paired_evaluation(
        run,
        receipt,
        _protocol(),
    )

    assert result.method == "paired_difference"
    assert result.seed == 42
    assert result.formal_schedule_fingerprint == "4" * 64
    assert result.formal_protocol_fingerprint == "0" * 64
    assert result.runtime_input_fingerprint == "a" * 64
    assert result.control_provider_fingerprint is None
    assert result.paired_criterion_fingerprint == module.stable_fingerprint(
        run.artifact.config.canonical_payload()["paired_criterion"]
    )
    assert result.method_objective_fingerprint == module.stable_fingerprint(
        run.artifact.config.canonical_payload()["method_objective"]
    )

    object.__setattr__(receipt, "decoder_variant", "independent_endpoint")
    with pytest.raises(RuntimeError, match="method/seed"):
        result_from_selected_paired_evaluation(
            run,
            receipt,
            _protocol(),
        )


def test_replayed_metrics_must_equal_selected_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, receipt = _fake_strict_sources(monkeypatch, _metrics())
    monkeypatch.setattr(
        module,
        "evaluate_formal_residual_threshold",
        lambda actual_run, actual_receipt: _metrics(recovered=7),
    )
    with pytest.raises(RuntimeError, match="differs"):
        result_from_selected_paired_evaluation(
            run,
            receipt,
            _protocol(),
        )


def test_create_only_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    result = _result()
    path = tmp_path / "paired-formal-result.json"
    fingerprint = save_paired_formal_d_v_result(path, result)
    loaded = load_paired_formal_d_v_result(path)

    assert fingerprint == result.receipt_fingerprint
    assert loaded.canonical_payload() == result.canonical_payload()
    assert loaded.to_formal_method_evidence() == (
        result.to_formal_method_evidence()
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_paired_formal_d_v_result(path, result)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["bindings"]["decoder_state_fingerprint"] = "f" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint|binding"):
        load_paired_formal_d_v_result(path)


@pytest.mark.parametrize(
    ("method", "pd", "expected_true", "expected_recovered"),
    (
        ("Base@B", 150 / 170, 150, 3),
        ("F", 154 / 170, 154, 7),
        ("F×", 149 / 170, 149, 2),
        ("U", 151 / 170, 151, 4),
    ),
)
def test_historical_receipt_adapter_never_requires_reevaluation(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    pd: float,
    expected_true: int,
    expected_recovered: int,
) -> None:
    monkeypatch.setattr(
        module,
        "evaluate_formal_residual_threshold",
        lambda *args, **kwargs: pytest.fail("historical adapter reevaluated D_V"),
    )
    metrics = _frozen_historical_metrics(method)
    evidence = adapt_frozen_historical_method_evidence(
        method=method,
        seed=42,
        selected_metrics=metrics,
        **_historical_sources(method),
    )
    assert evidence.true_targets == expected_true
    assert evidence.recovered_anchor_misses == expected_recovered
    assert evidence.pd == pd
    assert evidence.retention == metrics["retention"]
    assert evidence.pixel_fa == metrics["pixel_fa"]
    assert evidence.raw_background_fa == metrics["raw_background_fa"]
    assert evidence.fp_components_per_mp == metrics["fp_components_per_mp"]


def test_historical_adapter_rejects_rounded_or_noncanonical_values() -> None:
    metrics = dict(_frozen_historical_metrics("F"))
    metrics["pd"] = 0.901
    with pytest.raises(RuntimeError, match="frozen fx_v3"):
        adapt_frozen_historical_method_evidence(
            method="F",
            seed=42,
            selected_metrics=metrics,
            **_historical_sources("F"),
        )
    metrics = dict(_frozen_historical_metrics("U"))
    metrics["extra"] = 1.0
    with pytest.raises(ValueError, match="canonical"):
        adapt_frozen_historical_method_evidence(
            method="U",
            seed=42,
            selected_metrics=metrics,
            **_historical_sources("U"),
        )


def test_authoritative_comparison_protocol_is_frozen_to_fx_v3() -> None:
    protocol = _protocol()
    assert FORMAL_DV_IMAGES == 120
    assert protocol.comparison_protocol_fingerprint == (
        "cb2fb09c3ec7dbbb0f057d94f7f159e2b4a733296e6ea4a144d6302387014884"
    )
    assert protocol.ordered_d_v_sample_ids_fingerprint == (
        "02fe0ff4392563c4471487bf5d2e11bf39854e1d40ba714a9567dae0aa8b68b5"
    )
    assert protocol.residual_thresholds == tuple(
        index / 50 for index in range(51)
    )
    assert protocol.budget == FalseAlarmBudget(
        pixel_fa_budget=1e-4,
        component_fa_per_mp_budget=100.0,
        raw_background_fa_budget=1e-4,
        minimum_retention=0.99,
    )
    assert protocol.occupancy_config.threshold == 0.72
    assert [binding.seed for binding in protocol.historical_fx_v3] == [
        42,
        43,
    ]


def test_protocol_loader_rejects_100_image_assumption_duplicate_and_tamper(
    tmp_path: Path,
) -> None:
    payload = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["population"]["images"] = 100
    wrong_population = tmp_path / "wrong-population.json"
    wrong_population.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="constants"):
        load_frozen_comparison_protocol(wrong_population)

    payload = json.loads(_PROTOCOL_PATH.read_text(encoding="utf-8"))
    payload["ordered_d_v_sample_ids_fingerprint"] = "f" * 64
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint|payload"):
        load_frozen_comparison_protocol(tampered)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"a","schema_version":"b"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_frozen_comparison_protocol(duplicate)


def test_historical_adapter_rejects_wrong_method_seed_protocol_binding() -> None:
    metrics = _frozen_historical_metrics("F")
    sources = _historical_sources("F")
    sources["source_protocol_fingerprint"] = (
        _protocol()
        .historical_binding(42)
        .protocol_fingerprint_for("U")
    )
    with pytest.raises(RuntimeError, match="frozen fx_v3"):
        adapt_frozen_historical_method_evidence(
            method="F",
            seed=42,
            selected_metrics=metrics,
            **sources,
        )


def test_control_result_requires_provider_but_proposed_forbids_it() -> None:
    control = _result(method="independent_endpoint")
    assert control.runtime_input_fingerprint == "7" * 64
    assert control.control_provider_fingerprint == "8" * 64
    values = {
        field.name: getattr(_result(), field.name)
        for field in __import__("dataclasses").fields(
            module.PairedFormalDVResult
        )
        if field.name != "_verification_token"
    }
    values["control_provider_fingerprint"] = "f" * 64
    with pytest.raises(ValueError, match="must not bind"):
        module._new_result(**values)


def test_module_exposes_no_split_or_d_t_selection_argument() -> None:
    selection_signature = inspect.signature(
        module.select_and_evaluate_paired_formal_method
    )
    assert "split" not in selection_signature.parameters
    assert "residual_thresholds" not in selection_signature.parameters
    assert "budget" not in selection_signature.parameters
    assert "comparison_protocol" in selection_signature.parameters
    assert all(
        "d_t" not in name.lower()
        for name in module.__all__
    )
    assert set(FORMAL_STAGE_A_METRIC_FIELDS) == {
        "pd",
        "miou",
        "niou",
        "pixel_fa",
        "fp_components_per_mp",
        "raw_background_fa",
        "retention",
        "budget_violation",
    }
