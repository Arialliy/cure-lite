from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cure_lite.calibration import FalseAlarmBudget
from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.experiment.cache_pipeline import LoadedDVCacheBundle
from cure_lite.experiment.paired_artifacts import LoadedPairedDecoderArtifact
from cure_lite.experiment.paired_formal_decision import FormalMethodEvidence
from cure_lite.experiment import paired_formal_evaluation as evaluation
from cure_lite.experiment import paired_formal_wave_reveal as module
from cure_lite.experiment.paired_formal_runner import (
    PublishedPairedFormalAttempt,
)
from cure_lite.metrics import AggregateEvaluation


_REAL_PROTOCOL = (
    Path(__file__).resolve().parents[1]
    / "protocols"
    / "IRSTD-1K"
    / "paired_formal_evaluation_v1"
    / "config.json"
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _metrics(recovered: int) -> AggregateEvaluation:
    retained = 147
    true_targets = retained + recovered
    return AggregateEvaluation(
        pd=true_targets / 170,
        rmr=recovered / 23,
        gross_rmr=recovered / 23,
        net_rmr=recovered / 23,
        retention=1.0,
        reachable_rmr=recovered / 23,
        oracle_upper_bound=1.0,
        overlap_supported_rmr=recovered / 23,
        pixel_fa=1e-5,
        raw_background_fa=2e-5,
        fp_components_per_mp=10.0,
        miou=0.5,
        niou=0.4,
        images=120,
        recovered_anchor_misses=recovered,
        net_recovered_anchor_misses=recovered,
        total_anchor_misses=23,
        retained_anchor_covered=retained,
        total_anchor_covered=retained,
        recovered_reachable_anchor_misses=recovered,
        total_reachable_anchor_misses=23,
        budget_violation=False,
    )


def _result(
    protocol_fingerprint: str,
    method: str,
    seed: int,
    *,
    recovered: int,
):
    return evaluation._new_result(
        method=method,
        seed=seed,
        comparison_protocol_fingerprint=protocol_fingerprint,
        selected_threshold=0.5,
        metrics=_metrics(recovered),
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
        runtime_input_fingerprint="3" * 64,
        control_preflight_fingerprint="4" * 64,
        control_provider_fingerprint=(
            None if method == "paired_difference" else "5" * 64
        ),
        method_contract_fingerprint="6" * 64,
        paired_criterion_fingerprint="7" * 64,
        method_objective_fingerprint="8" * 64,
    )


def _historical_evidence(
    protocol_fingerprint: str,
) -> tuple[FormalMethodEvidence, ...]:
    recovered = {"Base@B": 3, "F": 7, "F×": 2, "U": 4}
    return tuple(
        FormalMethodEvidence(
            method=method,
            seed=seed,
            total_targets=170,
            true_targets=147 + recovered[method],
            pd=(147 + recovered[method]) / 170,
            total_anchor_misses=23,
            recovered_anchor_misses=recovered[method],
            retention=1.0,
            pixel_fa=1e-5,
            raw_background_fa=2e-5,
            fp_components_per_mp=10.0,
            budget_violation=False,
            comparison_protocol_fingerprint=protocol_fingerprint,
            result_fingerprint=f"{seed - 40:x}" * 64,
        )
        for seed in (42, 43)
        for method in ("Base@B", "F", "F×", "U")
    )


def _bound_file(root: Path, relative: str, content: bytes) -> dict[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"repo_path": relative, "file_sha256": file_sha256(path)}


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[tuple[int, str], LoadedPairedDecoderArtifact]]:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "reveals").mkdir()
    monkeypatch.setattr(module, "_REPO_ROOT", repository)
    monkeypatch.setattr(
        module,
        "_IMPLEMENTATION_PATHS",
        ("impl/reveal.py", "impl/cli.py"),
    )
    implementation = {}
    for relative in module._IMPLEMENTATION_PATHS:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        implementation[relative] = file_sha256(path)

    comparison = _bound_file(
        repository,
        "protocol/comparison.json",
        _REAL_PROTOCOL.read_bytes(),
    )
    protocol = evaluation.load_frozen_comparison_protocol(
        repository / comparison["repo_path"]
    )
    protocol_fp = protocol.comparison_protocol_fingerprint
    runner_fp = "f" * 64
    artifacts: dict[tuple[int, str], LoadedPairedDecoderArtifact] = {}
    attempts = []
    for index, (seed, method) in enumerate(module.WAVE_A_ATTEMPTS):
        relative = f"runs/{index}-{method}-{seed}"
        root = repository / relative
        complete = root / "COMPLETE.json"
        receipt = root / "decoder_artifact" / "receipt.json"
        _write(complete, {"attempt": index})
        _write(receipt, {"artifact": index})
        artifact = object.__new__(LoadedPairedDecoderArtifact)
        config = SimpleNamespace(method=method, seed=seed)
        for name, value in {
            "config": config,
            "artifact_fingerprint": f"{index + 1:x}" * 64,
            "decoder_state_fingerprint": f"{index + 5:x}" * 64,
        }.items():
            object.__setattr__(artifact, name, value)
        artifacts[(seed, method)] = artifact
        attempts.append(
            {
                "seed": seed,
                "method": method,
                "repo_path": relative,
                "complete_file_sha256": file_sha256(complete),
                "complete_fingerprint": f"{index + 9:x}" * 64,
                "run_receipt_fingerprint": f"{index + 1:x}" * 64,
                "paired_artifact_fingerprint": artifact.artifact_fingerprint,
                "artifact_receipt_sha256": file_sha256(receipt),
                "decoder_state_fingerprint": artifact.decoder_state_fingerprint,
            }
        )

    historical_bindings = []
    source_by_seed = {}
    for offset, seed in enumerate((42, 43)):
        run_relative = f"historical/run-{seed}"
        protocol_relative = f"historical/protocol-{seed}"
        run = repository / run_relative
        frozen = repository / protocol_relative
        _write(run / "COMPLETE.json", {"seed": seed})
        _write(frozen / "protocol_freeze.json", {"seed": seed})
        _write(frozen / "stage_a_config.json", {"seed": seed})
        _write(frozen / "stage_a_decision_rule.json", {"seed": seed})
        complete_fp = f"{offset + 13:x}" * 64
        historical_bindings.append(
            {
                "seed": seed,
                "run_repo_path": run_relative,
                "complete_file_sha256": file_sha256(run / "COMPLETE.json"),
                "complete_fingerprint": complete_fp,
                "protocol_repo_path": protocol_relative,
                "protocol_freeze_sha256": file_sha256(
                    frozen / "protocol_freeze.json"
                ),
                "stage_a_config_sha256": file_sha256(
                    frozen / "stage_a_config.json"
                ),
                "decision_rule_sha256": file_sha256(
                    frozen / "stage_a_decision_rule.json"
                ),
            }
        )
        source_by_seed[seed] = SimpleNamespace(
            run_root=run.resolve(),
            protocol_root=frozen.resolve(),
            methods=(
                SimpleNamespace(source_complete_fingerprint=complete_fp),
            ),
            stage_a_config_sha256=historical_bindings[-1][
                "stage_a_config_sha256"
            ],
            decision_rule_sha256=historical_bindings[-1][
                "decision_rule_sha256"
            ],
        )
    historical_evidence = _historical_evidence(protocol_fp)
    sources = SimpleNamespace(
        source_for_seed=lambda seed: source_by_seed[seed],
        adapted_evidence=lambda actual_protocol: historical_evidence,
    )

    manifest = _bound_file(repository, "d_v/manifest.json", b"manifest\n")
    base_index = _bound_file(repository, "d_v/index.json", b"index\n")
    preprocessing = {
        "height": 256,
        "width": 256,
        "color_mode": "RGB",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "image_interpolation": "bilinear",
        "mask_interpolation": "nearest",
        "range": "float32-[0,1]-then-normalize",
    }
    unsigned = {
        "schema_version": module.PAIRED_FORMAL_WAVE_A_REVEAL_CONFIG_SCHEMA,
        "protocol_id": "irstd1k-paired-formal-wave-a-reveal-v1",
        "dataset": "IRSTD-1K",
        "wave": "A",
        "formal_runner_config_fingerprint": runner_fp,
        "comparison_protocol": comparison,
        "attempts": attempts,
        "historical_sources": historical_bindings,
        "d_v_bundle": {
            "manifest": manifest,
            "base_index": base_index,
            "expected_base_fingerprint": "a" * 64,
            "preprocessing": preprocessing,
            "preprocessing_fingerprint": stable_fingerprint(preprocessing),
        },
        "output_repo_path": "reveals/wave-a",
        "implementation_binding": implementation,
        "execution_policy": {
            "create_only_output": True,
            "failed_staging_reuse": False,
            "resume": False,
            "overwrite": False,
            "runtime_split": "D_V",
            "allow_D_T": False,
            "bundle_materializations": 1,
            "new_method_evaluations": 4,
            "historical_receipt_adaptations": 8,
            "wave_decisions": 1,
            "all_non_d_v_inputs_verified_before_d_v": True,
            "complete_written_last": True,
            "atomic_final_rename": True,
            "stdout_only_after_success": True,
        },
        "config_fingerprint_scope": "all-fields-except-config_fingerprint",
    }
    config = {**unsigned, "config_fingerprint": stable_fingerprint(unsigned)}
    config_path = repository / "wave-a-config.json"
    _write(config_path, config)

    attempts_by_root = {
        (repository / row["repo_path"]).resolve(): PublishedPairedFormalAttempt(
            root=(repository / row["repo_path"]).resolve(),
            method=row["method"],
            seed=row["seed"],
            config_fingerprint=runner_fp,
            formal_schedule_fingerprint="0" * 64,
            runtime_input_fingerprint="1" * 64,
            initial_decoder_fingerprint="2" * 64,
            final_decoder_fingerprint=row["decoder_state_fingerprint"],
            paired_artifact_fingerprint=row["paired_artifact_fingerprint"],
            provider_fingerprint=(
                None if row["method"] == "paired_difference" else "3" * 64
            ),
            run_receipt_fingerprint=row["run_receipt_fingerprint"],
            complete_fingerprint=row["complete_fingerprint"],
        )
        for row in attempts
    }
    monkeypatch.setattr(
        module,
        "load_paired_formal_attempt",
        lambda root: attempts_by_root[root],
    )
    monkeypatch.setattr(
        module,
        "load_paired_decoder_artifact",
        lambda path: artifacts[
            (
                int(path.parent.name.rsplit("-", 1)[1]),
                path.parent.name.split("-", 1)[1].rsplit("-", 1)[0],
            )
        ],
    )
    monkeypatch.setattr(
        module,
        "load_frozen_historical_fx_v3_sources",
        lambda **kwargs: sources,
    )
    monkeypatch.setattr(
        PublishedPairedFormalAttempt,
        "verify_unchanged",
        lambda self: None,
    )
    monkeypatch.setattr(
        LoadedPairedDecoderArtifact,
        "verify_unchanged",
        lambda self: None,
    )
    monkeypatch.setattr(
        LoadedDVCacheBundle,
        "verify_unchanged",
        lambda self: None,
    )
    monkeypatch.setattr(
        module.FrozenComparisonProtocol,
        "verify_bundle",
        lambda self, bundle: None,
    )
    return config_path, artifacts


def _fake_bundle() -> LoadedDVCacheBundle:
    return object.__new__(LoadedDVCacheBundle)


def test_one_shot_reveal_materializes_once_evaluates_four_and_decides_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _ = _fixture(tmp_path, monkeypatch)
    protocol = evaluation.load_frozen_comparison_protocol(
        tmp_path / "repo/protocol/comparison.json"
    )
    bundle = _fake_bundle()
    materializations = 0
    seen_bundles: list[LoadedDVCacheBundle] = []
    decisions = 0
    real_decision = module.assess_formal_wave

    def materialize(inputs):
        nonlocal materializations
        materializations += 1
        return bundle

    def evaluate(actual_bundle, artifact):
        seen_bundles.append(actual_bundle)
        recovered = 13 if artifact.config.method == "paired_difference" else 7
        return _result(
            protocol.comparison_protocol_fingerprint,
            artifact.config.method,
            artifact.config.seed,
            recovered=recovered,
        )

    def decide(*args, **kwargs):
        nonlocal decisions
        decisions += 1
        return real_decision(*args, **kwargs)

    monkeypatch.setattr(module, "assess_formal_wave", decide)
    published = module.run_wave_a_reveal(
        config_path,
        bundle_materializer=materialize,
        evaluator=evaluate,
    )

    assert materializations == 1
    assert seen_bundles == [bundle] * 4
    assert decisions == 1
    assert published.decision["all_seeds_pass"] is True
    assert published.decision["status"] == "FORMAL_WAVE_PASS"
    assert published.root.is_dir()
    assert {
        path.relative_to(published.root).as_posix()
        for path in published.root.rglob("*")
        if path.is_file()
    } == {
        "COMPLETE.json",
        "decision.json",
        "historical_evidence.json",
        "reveal_receipt.json",
        "results/independent_endpoint_seed42.json",
        "results/independent_endpoint_seed43.json",
        "results/paired_difference_seed42.json",
        "results/paired_difference_seed43.json",
    }
    complete = json.loads(
        (published.root / "COMPLETE.json").read_text(encoding="utf-8")
    )
    assert complete["complete_written_last"] is True
    assert complete["atomic_final_rename"] is True
    assert len(complete["artifact_files"]) == 7
    assert "COMPLETE.json" not in complete["artifact_files"]


def test_non_d_v_failure_occurs_before_bundle_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _ = _fixture(tmp_path, monkeypatch)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["attempts"][0]["complete_fingerprint"] = "e" * 64
    unsigned = dict(payload)
    unsigned.pop("config_fingerprint")
    payload["config_fingerprint"] = stable_fingerprint(unsigned)
    _write(config_path, payload)
    called = False

    def forbidden(_inputs):
        nonlocal called
        called = True
        return _fake_bundle()

    with pytest.raises(RuntimeError, match="formal attempt"):
        module.run_wave_a_reveal(
            config_path,
            bundle_materializer=forbidden,
        )
    assert called is False
    assert not (tmp_path / "repo/reveals/wave-a").exists()


def test_failed_publication_never_creates_final_and_staging_cannot_be_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _ = _fixture(tmp_path, monkeypatch)
    protocol = evaluation.load_frozen_comparison_protocol(
        tmp_path / "repo/protocol/comparison.json"
    )
    writes = 0
    real_write = module._write_new_json

    def evaluate(bundle, artifact):
        recovered = 13 if artifact.config.method == "paired_difference" else 7
        return _result(
            protocol.comparison_protocol_fingerprint,
            artifact.config.method,
            artifact.config.seed,
            recovered=recovered,
        )

    def fail_during_staging(path, value):
        nonlocal writes
        writes += 1
        if writes == 2:
            raise RuntimeError("synthetic staging failure")
        real_write(path, value)

    monkeypatch.setattr(module, "_write_new_json", fail_during_staging)
    with pytest.raises(RuntimeError, match="synthetic staging failure"):
        module.run_wave_a_reveal(
            config_path,
            bundle_materializer=lambda inputs: _fake_bundle(),
            evaluator=evaluate,
        )
    final = tmp_path / "repo/reveals/wave-a"
    config = module.load_wave_a_reveal_config(config_path)
    staging = final.with_name(
        f".wave-a.staging-{config.config_fingerprint}"
    )
    assert not final.exists()
    assert staging.is_dir()

    monkeypatch.setattr(module, "_write_new_json", real_write)
    with pytest.raises(FileExistsError, match="no resume or reuse"):
        module.run_wave_a_reveal(
            config_path,
            bundle_materializer=lambda inputs: pytest.fail(
                "D_V touched while failed staging existed"
            ),
            evaluator=evaluate,
        )


def test_failed_evaluation_claims_before_d_v_and_cannot_be_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _ = _fixture(tmp_path, monkeypatch)
    protocol = evaluation.load_frozen_comparison_protocol(
        tmp_path / "repo/protocol/comparison.json"
    )
    evaluations = 0

    def evaluate(bundle, artifact):
        nonlocal evaluations
        evaluations += 1
        if evaluations == 2:
            raise RuntimeError("synthetic evaluation failure")
        return _result(
            protocol.comparison_protocol_fingerprint,
            artifact.config.method,
            artifact.config.seed,
            recovered=13,
        )

    with pytest.raises(RuntimeError, match="synthetic evaluation failure"):
        module.run_wave_a_reveal(
            config_path,
            bundle_materializer=lambda inputs: _fake_bundle(),
            evaluator=evaluate,
        )
    final = tmp_path / "repo/reveals/wave-a"
    config = module.load_wave_a_reveal_config(config_path)
    staging = final.with_name(
        f".wave-a.staging-{config.config_fingerprint}"
    )
    assert not final.exists()
    assert staging.is_dir()
    assert (staging / ".INCOMPLETE.json").is_file()

    with pytest.raises(FileExistsError, match="no resume or reuse"):
        module.run_wave_a_reveal(
            config_path,
            bundle_materializer=lambda inputs: pytest.fail(
                "D_V touched while failed reveal claim existed"
            ),
            evaluator=evaluate,
        )


def test_published_loader_rejects_result_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _ = _fixture(tmp_path, monkeypatch)
    protocol = evaluation.load_frozen_comparison_protocol(
        tmp_path / "repo/protocol/comparison.json"
    )

    def evaluate(bundle, artifact):
        recovered = 13 if artifact.config.method == "paired_difference" else 7
        return _result(
            protocol.comparison_protocol_fingerprint,
            artifact.config.method,
            artifact.config.seed,
            recovered=recovered,
        )

    published = module.run_wave_a_reveal(
        config_path,
        bundle_materializer=lambda inputs: _fake_bundle(),
        evaluator=evaluate,
    )
    assert (
        module.load_published_wave_a_reveal(published.root).complete_fingerprint
        == published.complete_fingerprint
    )
    result_path = (
        published.root / "results/paired_difference_seed42.json"
    )
    result_path.write_bytes(result_path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="COMPLETE semantics"):
        module.load_published_wave_a_reveal(published.root)


def test_decision_is_bound_to_reveal_config_and_recomputed_from_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _ = _fixture(tmp_path, monkeypatch)
    protocol = evaluation.load_frozen_comparison_protocol(
        tmp_path / "repo/protocol/comparison.json"
    )

    def evaluate(bundle, artifact):
        recovered = 13 if artifact.config.method == "paired_difference" else 7
        return _result(
            protocol.comparison_protocol_fingerprint,
            artifact.config.method,
            artifact.config.seed,
            recovered=recovered,
        )

    published = module.run_wave_a_reveal(
        config_path,
        bundle_materializer=lambda inputs: _fake_bundle(),
        evaluator=evaluate,
    )
    config = module.load_wave_a_reveal_config(config_path)
    assert published.decision["protocol_fingerprint"] == (
        config.config_fingerprint
    )
    decision_path = published.root / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["status"] = "PERFORMANCE_FAIL"
    core = dict(decision)
    core.pop("decision_fingerprint")
    decision["decision_fingerprint"] = stable_fingerprint(core)
    _write(decision_path, decision)
    receipt_path = published.root / "reveal_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["decision_fingerprint"] = decision["decision_fingerprint"]
    receipt["artifact_files_before_receipt"]["decision.json"] = file_sha256(
        decision_path
    )
    receipt_core = dict(receipt)
    receipt_core.pop("receipt_fingerprint")
    receipt["receipt_fingerprint"] = stable_fingerprint(receipt_core)
    _write(receipt_path, receipt)
    complete_path = published.root / "COMPLETE.json"
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["decision_fingerprint"] = decision["decision_fingerprint"]
    complete["reveal_receipt_fingerprint"] = receipt["receipt_fingerprint"]
    complete["artifact_files"]["decision.json"] = file_sha256(decision_path)
    complete["artifact_files"]["reveal_receipt.json"] = file_sha256(receipt_path)
    complete_core = dict(complete)
    complete_core.pop("complete_fingerprint")
    complete["complete_fingerprint"] = stable_fingerprint(complete_core)
    _write(complete_path, complete)
    with pytest.raises(RuntimeError, match="twelve evidence rows"):
        module.load_published_wave_a_reveal(published.root)


def test_config_rejects_extra_attempt_and_policy_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path, _ = _fixture(tmp_path, monkeypatch)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["attempts"].append(dict(payload["attempts"][0]))
    unsigned = dict(payload)
    unsigned.pop("config_fingerprint")
    payload["config_fingerprint"] = stable_fingerprint(unsigned)
    with pytest.raises(ValueError, match="exact ordered Wave-A quartet"):
        module.validate_wave_a_reveal_config(payload)

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["execution_policy"]["resume"] = True
    unsigned = dict(payload)
    unsigned.pop("config_fingerprint")
    payload["config_fingerprint"] = stable_fingerprint(unsigned)
    with pytest.raises(ValueError, match="execution policy"):
        module.validate_wave_a_reveal_config(payload)
