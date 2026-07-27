from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cure_lite.calibration import FalseAlarmBudget
from cure_lite.experiment import (
    phase_resolved_real_formal_evaluation as module,
)
from cure_lite.experiment.paired_formal_evaluation import (
    FORMAL_DV_ANCHOR_COVERED,
    FORMAL_DV_ANCHOR_MISSES,
    FORMAL_DV_IMAGES,
    FORMAL_DV_TOTAL_TARGETS,
    load_frozen_comparison_protocol,
)
from cure_lite.experiment.phase_resolved_real_artifacts import (
    LoadedPFCRRealDecoderArtifact,
)
from cure_lite.experiment.phase_resolved_real_formal_evaluation import (
    LoadedPFCRFormalDVRun,
    PFCRFormalDVResult,
    build_loaded_pfcr_formal_d_v_run,
    load_pfcr_formal_d_v_result,
    save_pfcr_formal_d_v_result,
    select_and_evaluate_pfcr_formal_method,
)
from cure_lite.experiment.phase_resolved_real_formal_runner import (
    PublishedPFCRRealFormalAttempt,
)
from cure_lite.metrics import AggregateEvaluation


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "protocols/IRSTD-1K/paired_formal_evaluation_v1/config.json"
)


def _protocol():
    return load_frozen_comparison_protocol(PROTOCOL)


def _metrics(
    *,
    recovered: int = 9,
    retained: int = FORMAL_DV_ANCHOR_COVERED,
    pixel_fa: float = 1.0e-5,
    raw_background_fa: float = 2.0e-5,
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
    seed: int = 42,
    metrics: AggregateEvaluation | None = None,
) -> PFCRFormalDVResult:
    values = {
        "seed": seed,
        "execution_device": "cpu",
        "comparison_protocol_fingerprint": (
            _protocol().comparison_protocol_fingerprint
        ),
        "selected_threshold": 0.74,
        "metrics": _metrics() if metrics is None else metrics,
        "budget": _protocol().budget,
    }
    names = (
        "pfcr_d_v_run_fingerprint",
        "threshold_protocol_fingerprint",
        "manifest_fingerprint",
        "manifest_file_sha256",
        "preprocessing_fingerprint",
        "base_fingerprint",
        "d_v_base_index_fingerprint",
        "d_v_base_index_sha256",
        "d_v_image_fingerprint",
        "d_v_gt_fingerprint",
        "base_samples_fingerprint",
        "residual_samples_fingerprint",
        "sample_adapter_fingerprint",
        "cache_contract_fingerprint",
        "formal_attempt_run_receipt_fingerprint",
        "formal_attempt_complete_fingerprint",
        "decoder_artifact_fingerprint",
        "decoder_receipt_sha256",
        "decoder_state_fingerprint",
        "formal_schedule_fingerprint",
        "state_catalog_fingerprint",
        "lineage_allowlist_fingerprint",
        "preflight_result_fingerprint",
    )
    values.update(
        {
            name: format(index % 16, "x") * 64
            for index, name in enumerate(names)
        }
    )
    return module._new_result(**values)


def test_result_is_pfcr_specific_and_contains_complete_attempt_bindings() -> None:
    result = _result()
    payload = result.canonical_payload()
    evidence = result.to_formal_method_evidence()

    assert result.method == "PFCR"
    assert payload["method"] == "PFCR"
    assert payload["runtime_split"] == "D_V"
    assert payload["D_T_accessed"] is False
    assert payload["execution_device"] == "cpu"
    assert payload["bindings"]["formal_attempt_complete_fingerprint"]
    assert payload["bindings"]["sample_adapter_fingerprint"]
    assert payload["bindings"]["base_samples_fingerprint"]
    assert evidence.method == "PFCR"
    assert evidence.seed == 42
    assert evidence.total_targets == 170
    assert evidence.total_anchor_misses == 23
    assert evidence.true_targets == 156
    assert evidence.recovered_anchor_misses == 9
    assert evidence.result_fingerprint == result.result_fingerprint


def test_result_enforces_population_count_identities_and_frozen_budget() -> None:
    with pytest.raises(ValueError, match="exactly 120 images"):
        _result(metrics=replace(_metrics(), images=119))
    with pytest.raises(ValueError, match="denominator.*23"):
        _result(
            metrics=replace(
                _metrics(),
                total_anchor_misses=22,
            )
        )
    with pytest.raises(ValueError, match="Pd"):
        _result(metrics=replace(_metrics(), pd=0.5))
    with pytest.raises(ValueError, match="budget"):
        _result(metrics=_metrics(pixel_fa=2.0e-4))
    with pytest.raises(ValueError, match="budget"):
        _result(metrics=replace(_metrics(), budget_violation=True))


def test_result_create_only_round_trip_and_tamper_rejection(
    tmp_path: Path,
) -> None:
    result = _result()
    target = tmp_path / "pfcr-seed42.json"

    fingerprint = save_pfcr_formal_d_v_result(target, result)
    loaded = load_pfcr_formal_d_v_result(target)

    assert fingerprint == result.receipt_fingerprint
    assert loaded.canonical_payload() == result.canonical_payload()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_pfcr_formal_d_v_result(target, result)

    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["bindings"]["decoder_state_fingerprint"] = "f" * 64
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint|binding"):
        load_pfcr_formal_d_v_result(target)


@pytest.mark.parametrize(
    "entry",
    (
        build_loaded_pfcr_formal_d_v_run,
        select_and_evaluate_pfcr_formal_method,
    ),
)
def test_public_entry_rejects_bare_pfcr_artifact_before_d_v(
    entry,
) -> None:
    bare = object.__new__(LoadedPFCRRealDecoderArtifact)
    with pytest.raises(
        TypeError,
        match="PublishedPFCRRealFormalAttempt|bare PFCR artifacts",
    ):
        entry(
            None,
            None,
            bare,
            comparison_protocol=None,
            device="cpu",
        )


def test_formal_builder_resolves_private_pfcr_d_v_sample_builder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    digest = "1" * 64
    empty_samples_fingerprint = module.calibration_samples_fingerprint(())

    class FakeBundle:
        split_manifest_fingerprint = digest
        preprocessing_fingerprint = digest
        base_fingerprint = digest
        base_state_fingerprint = digest
        manifest_path = tmp_path / "manifest.json"
        rows: tuple[object, ...] = ()

        def verify_unchanged(self) -> None:
            return None

    class FakeContract:
        contract_fingerprint = digest
        dataset = "IRSTD-1K"
        split_manifest_fingerprint = digest
        preprocessing_fingerprint = digest
        base_fingerprint = digest
        base_state_fingerprint = digest
        occupancy_threshold = 0.5

    class FakeDRCache:
        contract = FakeContract()

        def verify_unchanged(self) -> None:
            return None

    class FakeProtocol:
        dataset = "IRSTD-1K"
        occupancy_config = SimpleNamespace(threshold=0.5)
        base_samples_fingerprint = empty_samples_fingerprint

        def verify_bundle(self, bundle) -> None:
            assert isinstance(bundle, FakeBundle)

    class FakeAccess:
        def __init__(self, manifest) -> None:
            self.manifest = manifest

        def records_for(self, split: str) -> tuple[object, ...]:
            assert split == "D_V"
            return ()

    artifact = SimpleNamespace(
        config=SimpleNamespace(cache_contract_fingerprint=digest),
        verify_unchanged=lambda: None,
    )
    attempt = PublishedPFCRRealFormalAttempt(
        root=tmp_path,
        seed=42,
        artifact=artifact,  # type: ignore[arg-type]
        run_receipt_fingerprint="2" * 64,
        complete_fingerprint="3" * 64,
    )
    decoder = object()
    built_samples = SimpleNamespace(samples=())
    calls: list[tuple[object, ...]] = []

    def build_samples(
        bundle,
        contract,
        actual_decoder,
        occupancy_config,
        *,
        batch_size: int,
    ):
        calls.append(
            (
                bundle,
                contract,
                actual_decoder,
                occupancy_config,
                batch_size,
            )
        )
        return built_samples

    monkeypatch.setattr(module, "LoadedDVCacheBundle", FakeBundle)
    monkeypatch.setattr(module, "PFCRRealCacheAdapter", FakeDRCache)
    monkeypatch.setattr(module, "FrozenComparisonProtocol", FakeProtocol)
    monkeypatch.setattr(
        module,
        "_verify_published_attempt",
        lambda actual_attempt: None,
    )
    monkeypatch.setattr(
        module,
        "load_and_validate_manifest",
        lambda path: object(),
    )
    monkeypatch.setattr(module, "DevelopmentSplitAccess", FakeAccess)
    monkeypatch.setattr(
        module,
        "_frozen_decoder_clone",
        lambda actual_artifact, *, device: decoder,
    )
    monkeypatch.setattr(module, "_build_pfcr_d_v_samples", build_samples)
    monkeypatch.setattr(
        LoadedPFCRFormalDVRun,
        "verify_unchanged",
        lambda self: None,
    )

    bundle = FakeBundle()
    d_r_cache = FakeDRCache()
    protocol = FakeProtocol()
    run = build_loaded_pfcr_formal_d_v_run(
        bundle,  # type: ignore[arg-type]
        d_r_cache,  # type: ignore[arg-type]
        attempt,
        comparison_protocol=protocol,  # type: ignore[arg-type]
        device="cpu",
    )

    assert run.d_v_samples is built_samples
    assert calls == [
        (
            bundle,
            d_r_cache.contract,
            decoder,
            protocol.occupancy_config,
            module.PFCR_FORMAL_DV_BATCH_SIZE,
        )
    ]


def test_evaluator_consumes_only_the_frozen_comparison_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    metrics = _metrics()
    threshold_protocol = SimpleNamespace(
        selected_threshold=0.82,
        selected_metrics=metrics,
        receipt_fingerprint="1" * 64,
    )
    artifact_config = SimpleNamespace(
        cache_contract_fingerprint="2" * 64,
        formal_schedule_fingerprint="3" * 64,
        state_catalog_fingerprint="4" * 64,
        lineage_allowlist_fingerprint="5" * 64,
        preflight_result_fingerprint="6" * 64,
    )
    artifact = SimpleNamespace(
        config=artifact_config,
        artifact_fingerprint="7" * 64,
        receipt_sha256="8" * 64,
        decoder_state_fingerprint="9" * 64,
    )
    attempt = PublishedPFCRRealFormalAttempt(
        root=tmp_path,
        seed=42,
        artifact=artifact,  # type: ignore[arg-type]
        run_receipt_fingerprint="a" * 64,
        complete_fingerprint="b" * 64,
    )
    bundle = SimpleNamespace(
        split_manifest_fingerprint="c" * 64,
        split_manifest_file_sha256="d" * 64,
        preprocessing_fingerprint="e" * 64,
        base_fingerprint="f" * 64,
        base_index_fingerprint="0" * 64,
        base_index_sha256="1" * 64,
        d_v_image_fingerprint="2" * 64,
        d_v_gt_fingerprint="3" * 64,
    )
    run = object.__new__(LoadedPFCRFormalDVRun)
    for name, value in {
        "access": object(),
        "d_v_samples": SimpleNamespace(
            samples=(),
            adapter_fingerprint="4" * 64,
            sample_tensor_fingerprint="6" * 64,
        ),
        "base_samples_fingerprint": "5" * 64,
    }.items():
        object.__setattr__(run, name, value)

    monkeypatch.setattr(
        module,
        "build_loaded_pfcr_formal_d_v_run",
        lambda *args, **kwargs: run,
    )
    monkeypatch.setattr(
        LoadedPFCRFormalDVRun,
        "verify_unchanged",
        lambda self: None,
    )
    monkeypatch.setattr(
        LoadedPFCRFormalDVRun,
        "run_fingerprint",
        property(lambda self: "7" * 64),
    )
    captured: dict[str, object] = {}

    def select(
        access,
        samples,
        thresholds,
        occupancy,
        match,
        budget,
    ):
        captured.update(
            {
                "thresholds": thresholds,
                "occupancy": occupancy,
                "match": match,
                "budget": budget,
            }
        )
        return threshold_protocol

    monkeypatch.setattr(
        module,
        "select_residual_threshold_on_d_v",
        select,
    )
    monkeypatch.setattr(
        module,
        "_verify_selected_protocol",
        lambda actual_run, actual_protocol: None,
    )
    monkeypatch.setattr(
        module,
        "evaluate_frozen_residual_threshold",
        lambda *args: metrics,
    )

    result = select_and_evaluate_pfcr_formal_method(
        bundle,  # type: ignore[arg-type]
        object(),  # builder is replaced in this composition test
        attempt,
        comparison_protocol=protocol,
        device="cpu",
    )

    assert captured == {
        "thresholds": protocol.residual_thresholds,
        "occupancy": protocol.occupancy_config,
        "match": protocol.match_config,
        "budget": protocol.budget,
    }
    assert result.seed == 42
    assert result.selected_threshold == 0.82
    assert result.formal_attempt_complete_fingerprint == "b" * 64


def test_replayed_metrics_must_equal_the_selected_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    selected = _metrics(recovered=9)
    replayed = _metrics(recovered=8)
    threshold_protocol = SimpleNamespace(
        selected_threshold=None,
        selected_metrics=selected,
        receipt_fingerprint="1" * 64,
    )
    attempt = PublishedPFCRRealFormalAttempt(
        root=tmp_path,
        seed=42,
        artifact=SimpleNamespace(),  # type: ignore[arg-type]
        run_receipt_fingerprint="2" * 64,
        complete_fingerprint="3" * 64,
    )
    run = object.__new__(LoadedPFCRFormalDVRun)
    object.__setattr__(run, "access", object())
    object.__setattr__(
        run,
        "d_v_samples",
        SimpleNamespace(samples=()),
    )
    monkeypatch.setattr(
        module,
        "build_loaded_pfcr_formal_d_v_run",
        lambda *args, **kwargs: run,
    )
    monkeypatch.setattr(
        LoadedPFCRFormalDVRun,
        "verify_unchanged",
        lambda self: None,
    )
    monkeypatch.setattr(
        module,
        "select_residual_threshold_on_d_v",
        lambda *args: threshold_protocol,
    )
    monkeypatch.setattr(
        module,
        "_verify_selected_protocol",
        lambda actual_run, actual_protocol: None,
    )
    monkeypatch.setattr(
        module,
        "evaluate_frozen_residual_threshold",
        lambda *args: replayed,
    )

    with pytest.raises(RuntimeError, match="replayed PFCR metrics differ"):
        select_and_evaluate_pfcr_formal_method(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            attempt,
            comparison_protocol=protocol,
            device="cpu",
        )


def test_public_api_exposes_only_device_not_protocol_overrides() -> None:
    for entry in (
        build_loaded_pfcr_formal_d_v_run,
        select_and_evaluate_pfcr_formal_method,
    ):
        parameters = inspect.signature(entry).parameters
        assert "attempt" in parameters
        assert "comparison_protocol" in parameters
        assert "device" in parameters
        assert "split" not in parameters
        assert "thresholds" not in parameters
        assert "budget" not in parameters
    assert all("d_t" not in name.lower() for name in module.__all__)
    assert module.PFCR_FORMAL_DV_BATCH_SIZE == 8
    assert _protocol().residual_thresholds == tuple(
        index / 50 for index in range(51)
    )
    assert _protocol().budget == FalseAlarmBudget(
        pixel_fa_budget=1.0e-4,
        component_fa_per_mp_budget=100.0,
        raw_background_fa_budget=1.0e-4,
        minimum_retention=0.99,
    )
