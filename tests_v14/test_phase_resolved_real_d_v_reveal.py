from __future__ import annotations

import inspect
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cure_lite.calibration import FalseAlarmBudget
from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.experiment.cache_pipeline import LoadedDVCacheBundle
from cure_lite.experiment.paired_formal_decision import (
    FormalMethodEvidence,
)
from cure_lite.experiment.phase_resolved_real_formal_runner import (
    PublishedPFCRRealFormalAttempt,
)
from cure_lite.metrics import AggregateEvaluation
from cure_lite.experiment import (
    phase_resolved_real_d_v_reveal as module,
)
from cure_lite.experiment import (
    phase_resolved_real_formal_evaluation as evaluation,
)


_ROOT = Path(__file__).resolve().parents[1]
_REAL_COMPARISON_PATH = (
    _ROOT
    / "protocols/IRSTD-1K/paired_formal_evaluation_v1/config.json"
)
_REAL_COMPARISON_PAYLOAD = json.loads(
    _REAL_COMPARISON_PATH.read_text(encoding="utf-8")
)
_COMPARISON_FP = _REAL_COMPARISON_PAYLOAD[
    "comparison_protocol_fingerprint"
]
_COMPARISON_BUNDLE = _REAL_COMPARISON_PAYLOAD["bundle_binding"]


def _digest(label: str) -> str:
    return stable_fingerprint({"label": label})


def _preprocessing() -> dict[str, object]:
    return {
        "color_mode": "L",
        "height": 256,
        "image_interpolation": "bilinear",
        "mask_interpolation": "nearest",
        "mean": [0.5],
        "range": "float32-[0,1]-then-normalize",
        "std": [0.5],
        "width": 256,
    }


def _attempt_binding(seed: int) -> dict[str, object]:
    return {
        "seed": seed,
        "device": module.PFCR_REAL_DEVICE_MAP[seed],
        "repo_path": f"runs/pfcr-seed{seed}",
        "complete_file_sha256": _digest(f"complete-sha-{seed}"),
        "complete_fingerprint": _digest(f"complete-fp-{seed}"),
        "run_receipt_fingerprint": _digest(f"run-{seed}"),
        "artifact_fingerprint": _digest(f"artifact-{seed}"),
        "artifact_receipt_sha256": _digest(f"artifact-sha-{seed}"),
        "decoder_state_fingerprint": _digest(f"decoder-{seed}"),
        "cache_contract_fingerprint": _digest("cache-contract"),
        "state_catalog_fingerprint": _digest("state-catalog"),
        "lineage_allowlist_fingerprint": _digest("lineage"),
        "formal_schedule_fingerprint": _digest(f"schedule-{seed}"),
        "preflight_result_fingerprint": _digest(f"preflight-{seed}"),
    }


def _config_payload() -> dict[str, object]:
    preprocessing = _preprocessing()
    unsigned = {
        "schema_version": module.PFCR_REAL_DV_REVEAL_CONFIG_SCHEMA,
        "protocol_id": module.PFCR_REAL_DV_REVEAL_PROTOCOL_ID,
        "dataset": "IRSTD-1K",
        "model": "CURE-Lite",
        "comparison_protocol": {
            "repo_path": (
                "protocols/IRSTD-1K/"
                "paired_formal_evaluation_v1/config.json"
            ),
            "file_sha256": file_sha256(_REAL_COMPARISON_PATH),
            "comparison_protocol_fingerprint": _COMPARISON_FP,
        },
        "attempts": [
            _attempt_binding(42),
            _attempt_binding(43),
        ],
        "comparator_wave_a": {
            "repo_path": "runs/comparator-wave-a",
            "complete_file_sha256": _digest("wave-complete-sha"),
            "decision_file_sha256": _digest("wave-decision-sha"),
            "complete_fingerprint": _digest("wave-complete-fp"),
            "decision_fingerprint": _digest("wave-decision-fp"),
        },
        "decision_gate": dict(module._EXPECTED_DECISION_GATE),
        "backend_policy": dict(module._EXPECTED_BACKEND_POLICY),
        "d_r_bundle": {
            "manifest": {
                "repo_path": "protocols/manifest.json",
                "file_sha256": _COMPARISON_BUNDLE[
                    "manifest_file_sha256"
                ],
            },
            "index": {
                "repo_path": "runs/dr-state-index.json",
                "file_sha256": _digest("dr-index"),
            },
            "expected_base_fingerprint": _COMPARISON_BUNDLE[
                "base_fingerprint"
            ],
            "expected_cache_contract_fingerprint": _digest(
                "cache-contract"
            ),
            "preprocessing": preprocessing,
            "preprocessing_fingerprint": stable_fingerprint(
                preprocessing
            ),
        },
        "d_v_bundle": {
            "manifest": {
                "repo_path": "protocols/manifest.json",
                "file_sha256": _COMPARISON_BUNDLE[
                    "manifest_file_sha256"
                ],
            },
            "index": {
                "repo_path": "runs/dv-base-index.json",
                "file_sha256": _COMPARISON_BUNDLE[
                    "d_v_base_index_sha256"
                ],
            },
            "expected_base_fingerprint": _COMPARISON_BUNDLE[
                "base_fingerprint"
            ],
            "preprocessing": preprocessing,
            "preprocessing_fingerprint": stable_fingerprint(
                preprocessing
            ),
        },
        "output_repo_path": "reveals/pfcr-formal",
        "implementation_binding": {
            path: _digest(f"implementation:{path}")
            for path in module._IMPLEMENTATION_PATHS
        },
        "execution_policy": {
            "create_only_output": True,
            "failed_staging_reuse": False,
            "resume": False,
            "overwrite": False,
            "runtime_split": "D_V",
            "allow_D_T": False,
            "d_r_bundle_materializations": 1,
            "d_v_bundle_materializations": 1,
            "pfcr_method_evaluations": 2,
            "frozen_comparator_evidence_rows": 12,
            "formal_decision_evidence_rows": 14,
            "formal_decisions": 1,
            "canonical_bundle_materializer_only": True,
            "canonical_evaluator_only": True,
            "seed_order": [42, 43],
            "seed_device_map": {"42": "cuda:0", "43": "cuda:2"},
            "sequential_evaluation": True,
            "shared_in_memory_d_v_bundle": True,
            "all_non_d_v_inputs_verified_before_d_v": True,
            "all_frozen_inputs_reverified_before_publication": True,
            "complete_written_last": True,
            "atomic_final_rename": True,
            "stdout_only_after_success": True,
        },
        "config_fingerprint_scope": (
            "all-fields-except-config_fingerprint"
        ),
    }
    return {
        **unsigned,
        "config_fingerprint": stable_fingerprint(unsigned),
    }


def _write_config(path: Path) -> module.LoadedPFCRRealDVRevealConfig:
    path.write_text(
        json.dumps(
            _config_payload(),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return module.load_pfcr_real_d_v_reveal_config(path)


def _metrics(recovered: int) -> AggregateEvaluation:
    retained = 147
    return AggregateEvaluation(
        pd=(retained + recovered) / 170,
        rmr=recovered / 23,
        gross_rmr=recovered / 23,
        net_rmr=recovered / 23,
        retention=1.0,
        reachable_rmr=recovered / 23,
        oracle_upper_bound=1.0,
        overlap_supported_rmr=recovered / 23,
        pixel_fa=1.0e-5,
        raw_background_fa=2.0e-5,
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
    seed: int,
    *,
    execution_device: str | None = None,
) -> evaluation.PFCRFormalDVResult:
    binding = _attempt_binding(seed)
    values: dict[str, object] = {
        "seed": seed,
        "execution_device": (
            module.PFCR_REAL_DEVICE_MAP[seed]
            if execution_device is None
            else execution_device
        ),
        "comparison_protocol_fingerprint": _COMPARISON_FP,
        "selected_threshold": 0.5,
        "metrics": _metrics(9 if seed == 42 else 7),
        "budget": FalseAlarmBudget(
            pixel_fa_budget=1.0e-4,
            component_fa_per_mp_budget=100.0,
            raw_background_fa_budget=1.0e-4,
            minimum_retention=0.99,
        ),
        "pfcr_d_v_run_fingerprint": _digest(f"run-dv-{seed}"),
        "threshold_protocol_fingerprint": _digest(
            f"threshold-{seed}"
        ),
        "manifest_fingerprint": _COMPARISON_BUNDLE[
            "manifest_fingerprint"
        ],
        "manifest_file_sha256": _COMPARISON_BUNDLE[
            "manifest_file_sha256"
        ],
        "preprocessing_fingerprint": _COMPARISON_BUNDLE[
            "preprocessing_fingerprint"
        ],
        "base_fingerprint": _COMPARISON_BUNDLE[
            "base_fingerprint"
        ],
        "d_v_base_index_fingerprint": _COMPARISON_BUNDLE[
            "d_v_base_index_fingerprint"
        ],
        "d_v_base_index_sha256": _COMPARISON_BUNDLE[
            "d_v_base_index_sha256"
        ],
        "d_v_image_fingerprint": _COMPARISON_BUNDLE[
            "d_v_image_fingerprint"
        ],
        "d_v_gt_fingerprint": _COMPARISON_BUNDLE[
            "d_v_gt_fingerprint"
        ],
        "base_samples_fingerprint": _COMPARISON_BUNDLE[
            "base_samples_fingerprint"
        ],
        "residual_samples_fingerprint": _digest(
            f"residual-{seed}"
        ),
        "sample_adapter_fingerprint": _digest(f"adapter-{seed}"),
        "cache_contract_fingerprint": binding[
            "cache_contract_fingerprint"
        ],
        "formal_attempt_run_receipt_fingerprint": binding[
            "run_receipt_fingerprint"
        ],
        "formal_attempt_complete_fingerprint": binding[
            "complete_fingerprint"
        ],
        "decoder_artifact_fingerprint": binding[
            "artifact_fingerprint"
        ],
        "decoder_receipt_sha256": binding[
            "artifact_receipt_sha256"
        ],
        "decoder_state_fingerprint": binding[
            "decoder_state_fingerprint"
        ],
        "formal_schedule_fingerprint": binding[
            "formal_schedule_fingerprint"
        ],
        "state_catalog_fingerprint": binding[
            "state_catalog_fingerprint"
        ],
        "lineage_allowlist_fingerprint": binding[
            "lineage_allowlist_fingerprint"
        ],
        "preflight_result_fingerprint": binding[
            "preflight_result_fingerprint"
        ],
    }
    return evaluation._new_result(**values)


def _comparators() -> tuple[FormalMethodEvidence, ...]:
    values = {
        42: {
            "Base@B": (150, 3),
            "F": (154, 7),
            "F×": (149, 2),
            "U": (151, 4),
            "paired_difference": (147, 0),
            "independent_endpoint": (154, 7),
        },
        43: {
            "Base@B": (150, 3),
            "F": (152, 5),
            "F×": (147, 0),
            "U": (152, 5),
            "paired_difference": (152, 5),
            "independent_endpoint": (152, 5),
        },
    }
    return tuple(
        FormalMethodEvidence(
            method=method,
            seed=seed,
            total_targets=170,
            true_targets=values[seed][method][0],
            pd=values[seed][method][0] / 170,
            total_anchor_misses=23,
            recovered_anchor_misses=values[seed][method][1],
            retention=1.0,
            pixel_fa=1.0e-5,
            raw_background_fa=2.0e-5,
            fp_components_per_mp=10.0,
            budget_violation=False,
            comparison_protocol_fingerprint=_COMPARISON_FP,
            result_fingerprint=_digest(f"{seed}:{method}"),
        )
        for seed in (42, 43)
        for method in module.PFCR_FORMAL_COMPARATORS
    )


def _attempt(seed: int) -> PublishedPFCRRealFormalAttempt:
    binding = _attempt_binding(seed)
    artifact = SimpleNamespace(
        artifact_fingerprint=binding["artifact_fingerprint"],
        receipt_sha256=binding["artifact_receipt_sha256"],
        decoder_state_fingerprint=binding[
            "decoder_state_fingerprint"
        ],
    )
    return PublishedPFCRRealFormalAttempt(
        root=Path(f"/synthetic/pfcr-seed{seed}"),
        seed=seed,
        artifact=artifact,  # type: ignore[arg-type]
        run_receipt_fingerprint=binding["run_receipt_fingerprint"],
        complete_fingerprint=binding["complete_fingerprint"],
    )


def _inputs(
    repository: Path,
    config: module.LoadedPFCRRealDVRevealConfig,
) -> module._VerifiedInputs:
    comparison = _COMPARISON_FP
    protocol = SimpleNamespace(
        comparison_protocol_fingerprint=comparison,
        verify_bundle=lambda bundle: None,
    )
    comparator = SimpleNamespace(
        complete_fingerprint=_digest("wave-complete-fp"),
        decision={
            "decision_fingerprint": _digest("wave-decision-fp")
        },
    )
    output = repository / "reveals/pfcr-formal"
    return module._VerifiedInputs(
        protocol=protocol,  # type: ignore[arg-type]
        comparison_protocol_payload=_REAL_COMPARISON_PAYLOAD,
        attempts=(_attempt(42), _attempt(43)),
        devices=(torch.device("cuda:0"), torch.device("cuda:2")),
        comparator_reveal=comparator,  # type: ignore[arg-type]
        comparator_evidence=_comparators(),
        d_r_cache=SimpleNamespace(  # type: ignore[arg-type]
            contract=SimpleNamespace(
                contract_fingerprint=_digest("cache-contract")
            )
        ),
        d_v_manifest_path=repository / "unused-manifest.json",
        d_v_base_index_path=repository / "unused-index.json",
        d_v_preprocessing=SimpleNamespace(),  # type: ignore[arg-type]
        d_v_expected_base_fingerprint=_digest("base"),
        output=output,
        staging=output.with_name(
            f".{output.name}.staging-{config.config_fingerprint}"
        ),
    )


def _publish_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    results: tuple[evaluation.PFCRFormalDVResult, ...] | None = None,
    mutate_decision: bool = False,
):
    repository = tmp_path / "repo"
    (repository / "reveals").mkdir(parents=True)
    monkeypatch.setattr(module, "_REPO_ROOT", repository)
    monkeypatch.setattr(
        module,
        "_verify_snapshot_sources",
        lambda snapshot, rows: None,
    )
    config = _write_config(repository / "config.json")
    inputs = _inputs(repository, config)
    actual_results = (
        (_result(42), _result(43)) if results is None else results
    )
    decision = module.assess_pfcr_formal_d_v_gate(
        (
            *(
                result.to_formal_method_evidence()
                for result in actual_results
            ),
            *inputs.comparator_evidence,
        ),
        protocol_fingerprint=config.config_fingerprint,
        comparison_protocol_fingerprint=_COMPARISON_FP,
    )
    if mutate_decision:
        decision = dict(decision)
        decision["status"] = "FORGED"
        core = dict(decision)
        core.pop("decision_fingerprint")
        decision["decision_fingerprint"] = stable_fingerprint(core)
    module._claim_staging(config, inputs)
    return module._publish(
        config,
        inputs,
        actual_results,
        decision,
    )


def test_public_runner_and_cli_expose_only_config() -> None:
    parameters = inspect.signature(
        module.run_pfcr_real_d_v_reveal
    ).parameters
    assert tuple(parameters) == ("config_path",)
    assert all(
        forbidden not in parameters
        for forbidden in (
            "evaluator",
            "bundle_materializer",
            "device",
            "seed",
            "thresholds",
            "budget",
            "split",
            "resume",
            "overwrite",
        )
    )
    assert all("d_t" not in name.lower() for name in module.__all__)
    cli_path = (
        Path(__file__).resolve().parents[1]
        / "tools/run_phase_resolved_relation_real_d_v_reveal.py"
    )
    spec = importlib.util.spec_from_file_location("pfcr_reveal_cli", cli_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    options = {
        option
        for action in cli._parser()._actions
        for option in action.option_strings
        if option != "--help" and option != "-h"
    }
    assert options == {"--config"}


def test_config_freezes_gate_devices_backend_and_execution_policy() -> None:
    payload = _config_payload()
    assert module.validate_pfcr_real_d_v_reveal_config(payload) == payload

    changed = json.loads(json.dumps(payload))
    changed["attempts"][0]["device"] = "cuda:1"
    unsigned = dict(changed)
    unsigned.pop("config_fingerprint")
    changed["config_fingerprint"] = stable_fingerprint(unsigned)
    with pytest.raises(ValueError, match="device"):
        module.validate_pfcr_real_d_v_reveal_config(changed)

    changed = json.loads(json.dumps(payload))
    changed["decision_gate"]["minimum_true_target_margin"] = 1
    unsigned = dict(changed)
    unsigned.pop("config_fingerprint")
    changed["config_fingerprint"] = stable_fingerprint(unsigned)
    with pytest.raises(ValueError, match="decision gate"):
        module.validate_pfcr_real_d_v_reveal_config(changed)

    changed = json.loads(json.dumps(payload))
    changed["execution_policy"]["resume"] = True
    unsigned = dict(changed)
    unsigned.pop("config_fingerprint")
    changed["config_fingerprint"] = stable_fingerprint(unsigned)
    with pytest.raises(ValueError, match="execution policy"):
        module.validate_pfcr_real_d_v_reveal_config(changed)


def test_publish_is_complete_last_self_contained_and_strictly_loadable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = _publish_fixture(tmp_path, monkeypatch)
    loaded = module.load_published_pfcr_real_d_v_reveal(
        published.root
    )

    assert loaded.decision["all_seeds_pass"] is True
    assert [result.execution_device for result in loaded.results] == [
        "cuda:0",
        "cuda:2",
    ]
    assert len(loaded.comparator_evidence) == 12
    assert (loaded.root / "protocol_config.json").is_file()
    complete = json.loads(
        (loaded.root / "COMPLETE.json").read_text(encoding="utf-8")
    )
    assert complete["complete_written_last"] is True
    assert complete["canonical_evaluator_used"] is True
    assert "COMPLETE.json" not in complete["artifact_files"]


def test_strict_loader_rejects_an_extra_empty_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = _publish_fixture(tmp_path, monkeypatch)
    (published.root / "empty-extra").mkdir()

    with pytest.raises(RuntimeError, match="top-level inventory"):
        module.load_published_pfcr_real_d_v_reveal(published.root)


def test_strict_loader_rejects_incomplete_and_symlink_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published = _publish_fixture(tmp_path, monkeypatch)
    complete = published.root / "COMPLETE.json"
    complete.unlink()
    with pytest.raises(RuntimeError, match="top-level inventory"):
        module.load_published_pfcr_real_d_v_reveal(published.root)

    other_root = tmp_path / "second"
    published = _publish_fixture(other_root, monkeypatch)
    (published.root / "extra-link").symlink_to(
        published.root / "decision.json"
    )
    with pytest.raises(RuntimeError, match="top-level inventory"):
        module.load_published_pfcr_real_d_v_reveal(published.root)


def test_staging_claim_is_create_only_and_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    (repository / "reveals").mkdir(parents=True)
    monkeypatch.setattr(module, "_REPO_ROOT", repository)
    monkeypatch.setattr(
        module,
        "_verify_snapshot_sources",
        lambda snapshot, rows: None,
    )
    config = _write_config(repository / "config.json")
    inputs = _inputs(repository, config)

    module._claim_staging(config, inputs)
    assert (inputs.staging / ".INCOMPLETE.json").is_file()
    assert not (inputs.staging / "COMPLETE.json").exists()
    with pytest.raises(FileExistsError):
        module._claim_staging(config, inputs)


def test_strict_loader_recomputes_all_fourteen_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="fourteen evidence"):
        _publish_fixture(
            tmp_path,
            monkeypatch,
            mutate_decision=True,
        )


def test_strict_loader_rejects_wrong_per_seed_execution_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="frozen config/protocol"):
        _publish_fixture(
            tmp_path,
            monkeypatch,
            results=(
                _result(42, execution_device="cuda:2"),
                _result(43),
            ),
        )


def test_non_d_v_failure_precedes_materialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.json"
    _write_config(config_path)
    touched = False

    def fail_non_d_v(config):
        raise RuntimeError("synthetic non-D_V failure")

    def forbidden(inputs):
        nonlocal touched
        touched = True
        return object.__new__(LoadedDVCacheBundle)

    monkeypatch.setattr(module, "_verify_non_d_v_inputs", fail_non_d_v)
    monkeypatch.setattr(module, "_materialize_one_d_v_bundle", forbidden)
    with pytest.raises(RuntimeError, match="non-D_V failure"):
        module.run_pfcr_real_d_v_reveal(config_path)
    assert touched is False


def test_public_runner_materializes_once_and_evaluates_sequentially(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    (repository / "reveals").mkdir(parents=True)
    monkeypatch.setattr(module, "_REPO_ROOT", repository)
    config_path = repository / "config.json"
    config = _write_config(config_path)
    inputs = _inputs(repository, config)
    bundle = object.__new__(LoadedDVCacheBundle)
    materializations = 0
    calls: list[tuple[int, str, int]] = []

    monkeypatch.setattr(
        module,
        "_verify_non_d_v_inputs",
        lambda loaded: inputs,
    )

    def materialize(actual_inputs):
        nonlocal materializations
        materializations += 1
        assert actual_inputs is inputs
        return bundle

    def evaluate(
        actual_bundle,
        d_r_cache,
        attempt,
        *,
        comparison_protocol,
        device,
    ):
        calls.append((attempt.seed, str(device), id(actual_bundle)))
        return _result(attempt.seed)

    monkeypatch.setattr(
        module,
        "_materialize_one_d_v_bundle",
        materialize,
    )
    monkeypatch.setattr(
        module,
        "select_and_evaluate_pfcr_formal_method",
        evaluate,
    )
    monkeypatch.setattr(
        module,
        "_verify_frozen_sources_unchanged",
        lambda *args: None,
    )
    monkeypatch.setattr(
        module,
        "_verify_snapshot_sources",
        lambda snapshot, rows: None,
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    published = module.run_pfcr_real_d_v_reveal(config_path)

    assert materializations == 1
    assert calls == [
        (42, "cuda:0", id(bundle)),
        (43, "cuda:2", id(bundle)),
    ]
    assert published.decision["all_seeds_pass"] is True


def test_backend_policy_must_precede_cuda_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        module._configure_and_verify_backend_policy(
            require_uninitialized_cuda=True
        )


def test_snapshot_source_reload_rejects_a_changed_source_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    monkeypatch.setattr(module, "_REPO_ROOT", repository)
    snapshot = _config_payload()

    implementation: dict[str, str] = {}
    for relative in module._IMPLEMENTATION_PATHS:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        implementation[relative] = file_sha256(path)
    snapshot["implementation_binding"] = implementation

    comparison = repository / snapshot["comparison_protocol"]["repo_path"]
    comparison.parent.mkdir(parents=True, exist_ok=True)
    comparison.write_text("comparison\n", encoding="utf-8")
    snapshot["comparison_protocol"]["file_sha256"] = file_sha256(
        comparison
    )

    for binding in snapshot["attempts"]:
        root = repository / binding["repo_path"]
        (root / "decoder_artifact").mkdir(parents=True)
        (root / "COMPLETE.json").write_text(
            f"{binding['seed']}\n",
            encoding="utf-8",
        )
        (root / "decoder_artifact/receipt.json").write_text(
            f"artifact-{binding['seed']}\n",
            encoding="utf-8",
        )
        binding["complete_file_sha256"] = file_sha256(
            root / "COMPLETE.json"
        )
        binding["artifact_receipt_sha256"] = file_sha256(
            root / "decoder_artifact/receipt.json"
        )

    wave = repository / snapshot["comparator_wave_a"]["repo_path"]
    wave.mkdir(parents=True)
    (wave / "COMPLETE.json").write_text("complete\n", encoding="utf-8")
    (wave / "decision.json").write_text("decision\n", encoding="utf-8")
    snapshot["comparator_wave_a"]["complete_file_sha256"] = file_sha256(
        wave / "COMPLETE.json"
    )
    snapshot["comparator_wave_a"]["decision_file_sha256"] = file_sha256(
        wave / "decision.json"
    )

    manifest = repository / "protocols/manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("manifest\n", encoding="utf-8")
    for key in ("d_r_bundle", "d_v_bundle"):
        snapshot[key]["manifest"]["file_sha256"] = file_sha256(manifest)
        index = repository / snapshot[key]["index"]["repo_path"]
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(key, encoding="utf-8")
        snapshot[key]["index"]["file_sha256"] = file_sha256(index)

    unsigned = dict(snapshot)
    unsigned.pop("config_fingerprint")
    snapshot["config_fingerprint"] = stable_fingerprint(unsigned)
    module.validate_pfcr_real_d_v_reveal_config(snapshot)
    comparators = _comparators()
    monkeypatch.setattr(
        module,
        "load_frozen_comparison_protocol",
        lambda path: SimpleNamespace(
            comparison_protocol_fingerprint=_COMPARISON_FP
        ),
    )
    monkeypatch.setattr(
        module,
        "load_pfcr_real_formal_attempt",
        lambda root: _attempt(int(root.name.rsplit("seed", 1)[1])),
    )
    monkeypatch.setattr(
        module,
        "_verify_attempt_binding",
        lambda attempt, binding: None,
    )
    monkeypatch.setattr(
        module,
        "load_published_wave_a_reveal",
        lambda root: SimpleNamespace(
            complete_fingerprint=_digest("wave-complete-fp"),
            decision={
                "decision_fingerprint": _digest("wave-decision-fp")
            },
        ),
    )
    monkeypatch.setattr(
        module,
        "_extract_comparator_evidence",
        lambda reveal, comparison_protocol_fingerprint: comparators,
    )

    module._verify_snapshot_sources(snapshot, comparators)
    comparison.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="comparison protocol"):
        module._verify_snapshot_sources(snapshot, comparators)
