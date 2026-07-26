from __future__ import annotations

from dataclasses import replace
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.config import (
    InterventionConfig,
    MatchConfig,
    OccupancyConfig,
)
from cure_lite.experiment.artifacts import decoder_state_fingerprint
from cure_lite.experiment.paired_artifacts import PairedExecutionLedger
from cure_lite.experiment import paired_formal_runner as runner
from cure_lite.experiment.paired_formal_training import (
    PairedFormalTrainingResult,
)
from tests.test_paired_formal_schedule import (
    _factual_example,
    _fake_formal_schedule,
    _pair_catalog,
)


_ROOT = Path(__file__).resolve().parents[1]
_DRAFT = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "paired_formal_runner_v1"
    / "config.json"
)


def _initial_fingerprint(seed: int) -> str:
    from cure_lite.config import DecoderConfig
    from cure_lite.decoder import CURELiteDecoder

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        decoder = CURELiteDecoder(DecoderConfig(feature_channels=64))
    return decoder_state_fingerprint(decoder)


def _config_payload() -> dict[str, object]:
    payload = json.loads(_DRAFT.read_text(encoding="utf-8"))
    payload["initial_decoder_fingerprints"] = {
        "42": _initial_fingerprint(42),
        "43": _initial_fingerprint(43),
    }
    implementation = (
        _ROOT / "cure_lite" / "experiment" / "paired_formal_runner.py"
    )
    payload["implementation_binding"] = {
        implementation.relative_to(_ROOT).as_posix(): file_sha256(
            implementation
        )
    }
    payload.pop("config_fingerprint")
    payload["config_fingerprint"] = stable_fingerprint(payload)
    return payload


def _loaded_config(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(_config_payload(), sort_keys=True),
        encoding="utf-8",
    )
    return runner.load_paired_formal_runner_config(path)


def test_config_is_canonical_and_tamper_evident(tmp_path: Path) -> None:
    payload = _config_payload()
    validated = runner.validate_paired_formal_runner_config(payload)
    assert validated["budget"]["optimizer_updates"] == 32_000
    assert validated["execution_policy"]["resume"] is False
    assert validated["execution_policy"]["allow_D_V"] is False
    assert validated["execution_policy"]["allow_D_T"] is False

    changed = json.loads(json.dumps(payload))
    changed["budget"]["epochs"] = 799
    with pytest.raises(ValueError, match="fingerprint"):
        runner.validate_paired_formal_runner_config(changed)
    changed["config_fingerprint"] = stable_fingerprint(
        {
            key: value
            for key, value in changed.items()
            if key != "config_fingerprint"
        }
    )
    with pytest.raises(ValueError, match="budget"):
        runner.validate_paired_formal_runner_config(changed)

    loaded = _loaded_config(tmp_path)
    loaded.verify_unchanged()
    loaded.source_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        loaded.verify_unchanged()


def test_cpu_seed_initialization_is_method_independent(
    tmp_path: Path,
) -> None:
    config = _loaded_config(tmp_path)
    first = runner.build_seeded_formal_decoder(
        config,
        seed=42,
        device=torch.device("cpu"),
    )
    second = runner.build_seeded_formal_decoder(
        config,
        seed=42,
        device=torch.device("cpu"),
    )
    other_seed = runner.build_seeded_formal_decoder(
        config,
        seed=43,
        device=torch.device("cpu"),
    )
    assert decoder_state_fingerprint(first) == decoder_state_fingerprint(
        second
    )
    assert decoder_state_fingerprint(first) == config.payload[
        "initial_decoder_fingerprints"
    ]["42"]
    assert decoder_state_fingerprint(other_seed) == config.payload[
        "initial_decoder_fingerprints"
    ]["43"]
    assert decoder_state_fingerprint(first) != decoder_state_fingerprint(
        other_seed
    )


def _runtime_fingerprint_fixture(seed: int):
    factual_miss = tuple(
        _factual_example(
            branch="factual_miss",
            sample_id=f"runtime-miss-{index}",
            target_id=index + 1,
        )
        for index in range(5)
    )
    factual_no_miss = tuple(
        _factual_example(
            branch="factual_no_miss",
            sample_id=f"runtime-no-miss-{index}",
            target_id=None,
        )
        for index in range(4)
    )
    schedule = _fake_formal_schedule(
        seed,
        (factual_miss, factual_no_miss),
    )
    raw_catalog = _pair_catalog(
        ("pair-a", "pair-a", "pair-b", "pair-c")
    )
    unsigned_catalog = replace(raw_catalog, catalog_fingerprint="")
    catalog = replace(
        raw_catalog,
        catalog_fingerprint=stable_fingerprint(
            unsigned_catalog.canonical_payload()
        ),
    )
    return catalog, schedule


def test_runtime_input_fingerprint_is_seed_specific_and_tensor_complete() -> None:
    catalog42, schedule42 = _runtime_fingerprint_fixture(42)
    catalog43, schedule43 = _runtime_fingerprint_fixture(43)
    fingerprint42 = runner.paired_formal_runtime_input_fingerprint(
        catalog42,
        schedule42,
    )
    fingerprint43 = runner.paired_formal_runtime_input_fingerprint(
        catalog43,
        schedule43,
    )
    assert fingerprint42 != fingerprint43
    payload = runner.paired_formal_runtime_input_payload(
        catalog42,
        schedule42,
    )
    assert len(payload["factual_miss_anchors"]) == 5
    assert len(payload["factual_no_miss_anchors"]) == 4
    for row in (
        *payload["factual_miss_anchors"],
        *payload["factual_no_miss_anchors"],
    ):
        assert set(row["tensor_fingerprints"]) == {
            "feature",
            "occupancy",
            "target",
            "valid_mask",
        }

    schedule42.factual_miss_anchors[
        0
    ].example.supervision.target.zero_()
    with pytest.raises(ValueError, match="non-empty positive target"):
        runner.paired_formal_runtime_input_fingerprint(
            catalog42,
            schedule42,
        )

    catalog_mutated, schedule_mutated = _runtime_fingerprint_fixture(42)
    catalog_mutated.clean_positive[0].feature[0, 0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="feature tensor changed"):
        runner.paired_formal_runtime_input_fingerprint(
            catalog_mutated,
            schedule_mutated,
        )

    catalog_schedule, schedule_pair_mutated = (
        _runtime_fingerprint_fixture(42)
    )
    schedule_pair_mutated.paired_schedule.pairs[
        0
    ].feature[0, 0, 0, 0] += 1.0
    with pytest.raises(RuntimeError, match="schedule pair feature"):
        runner.paired_formal_runtime_input_fingerprint(
            catalog_schedule,
            schedule_pair_mutated,
        )


class _FakeBundle:
    split_manifest_fingerprint = "1" * 64
    split_manifest_file_sha256 = "2" * 64
    preprocessing_fingerprint = "3" * 64
    base_fingerprint = "4" * 64
    state_fingerprint = "5" * 64
    gt_fingerprint = "6" * 64
    base_index_fingerprint = "7" * 64
    base_index_sha256 = "8" * 64
    state_index_fingerprint = "9" * 64
    state_index_sha256 = "a" * 64
    occupancy_config = OccupancyConfig()
    match_config = MatchConfig()
    intervention_config = InterventionConfig()

    def verify_unchanged(self) -> None:
        return None


class _FakePreflight:
    complete_fingerprint = "b" * 64
    method_bindings_fingerprint = "c" * 64

    def verify_unchanged(self) -> None:
        return None


def _runtime() -> runner.PairedFormalRuntimeInputs:
    paired_schedule = SimpleNamespace(
        schedule_fingerprint="d" * 64,
    )
    schedule = SimpleNamespace(
        schedule_fingerprint="e" * 64,
        paired_schedule=paired_schedule,
    )
    pair_catalog = SimpleNamespace(catalog_fingerprint="f" * 64)
    return runner.PairedFormalRuntimeInputs(
        bundle=_FakeBundle(),
        pair_catalog=pair_catalog,
        prepared_catalog=SimpleNamespace(),
        schedule=schedule,
        control_provider=None,
    )


def _complete_fake_training(
    decoder,
    absolute_criterion,
    paired_criterion,
    optimizer,
    schedule,
    config,
    *,
    control_kwargs_provider,
):
    del absolute_criterion, paired_criterion, optimizer, schedule
    assert control_kwargs_provider is None
    with torch.no_grad():
        next(decoder.parameters()).add_(0.01)
    logs = tuple(
        {
            "epoch": epoch,
            "steps": 40,
            "metrics": {
                "mean_total_loss": 1.0,
                "mean_factual_miss_loss": 0.3,
                "mean_factual_no_miss_loss": 0.2,
                "mean_paired_or_control_loss": 0.5,
                "minimum_total_loss": 0.5,
                "maximum_total_loss": 1.5,
            },
        }
        for epoch in range(800)
    )
    ledger = PairedExecutionLedger(
        method=config.method,
        seed=config.seed,
        formal_schedule_fingerprint=config.formal_schedule_fingerprint,
        runtime_input_fingerprint=config.runtime_input_fingerprint,
        control_provider_fingerprint=config.control_provider_fingerprint,
        pair_exposure_fingerprint="1" * 64,
        factual_miss_exposure_fingerprint="2" * 64,
        factual_no_miss_exposure_fingerprint="3" * 64,
        initial_decoder_fingerprint=config.initial_decoder_fingerprint,
        final_decoder_fingerprint=decoder_state_fingerprint(decoder),
        trainable_parameter_count=sum(
            parameter.numel() for parameter in decoder.parameters()
        ),
        minimum_gradient_l2_norm=0.1,
        maximum_gradient_l2_norm=1.0,
    )
    return PairedFormalTrainingResult(logs, ledger)


def test_failed_attempt_is_not_reusable_and_writes_no_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _loaded_config(tmp_path)
    monkeypatch.setattr(
        runner,
        "_validate_runtime",
        lambda *a, **k: "0" * 64,
    )
    monkeypatch.setattr(
        runner,
        "paired_formal_runtime_input_fingerprint",
        lambda *a, **k: "0" * 64,
    )

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate bounded failure")

    monkeypatch.setattr(runner, "execute_paired_formal_training", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="deliberate"):
        runner.run_paired_formal_attempt(
            config,
            _FakePreflight(),
            _runtime(),
            method="paired_difference",
            seed=42,
            output_dir=output,
            device=torch.device("cpu"),
        )
    assert {path.name for path in output.iterdir()} == {
        ".INCOMPLETE.json"
    }
    assert not (output / "decoder_artifact").exists()
    assert not (output / "COMPLETE.json").exists()
    with pytest.raises(FileExistsError, match="cannot be reused"):
        runner.run_paired_formal_attempt(
            config,
            _FakePreflight(),
            _runtime(),
            method="paired_difference",
            seed=42,
            output_dir=output,
            device=torch.device("cpu"),
        )


def test_bounded_fake_completion_publishes_strict_artifact_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _loaded_config(tmp_path)
    monkeypatch.setattr(
        runner,
        "_validate_runtime",
        lambda *a, **k: "0" * 64,
    )
    monkeypatch.setattr(
        runner,
        "paired_formal_runtime_input_fingerprint",
        lambda *a, **k: "0" * 64,
    )
    monkeypatch.setattr(
        runner,
        "execute_paired_formal_training",
        _complete_fake_training,
    )
    output = tmp_path / "complete"
    published = runner.run_paired_formal_attempt(
        config,
        _FakePreflight(),
        _runtime(),
        method="paired_difference",
        seed=42,
        output_dir=output,
        device=torch.device("cpu"),
    )
    assert published.method == "paired_difference"
    assert published.provider_fingerprint is None
    assert not (output / ".INCOMPLETE.json").exists()
    assert (output / "COMPLETE.json").is_file()
    assert runner.load_paired_formal_attempt(output) == published
    provider = json.loads(
        (output / "control_provider_receipt.json").read_text()
    )
    assert provider["control_provider_used"] is False
    assert provider["provider_fingerprint"] is None
    assert provider["provider_receipt"] is None
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert complete["complete_800_by_40"] is True
    assert complete["resume_used"] is False
    assert complete["D_V_accessed"] is False
    assert complete["D_T_accessed"] is False
    assert complete["calibration_performed"] is False
    assert complete["inference_performed"] is False
    assert complete["wave_decision_performed"] is False


def test_post_training_runtime_change_prevents_artifact_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _loaded_config(tmp_path)
    catalog, schedule = _runtime_fingerprint_fixture(42)
    initial_runtime_fingerprint = (
        runner.paired_formal_runtime_input_fingerprint(catalog, schedule)
    )
    monkeypatch.setattr(
        runner,
        "_validate_runtime",
        lambda *a, **k: initial_runtime_fingerprint,
    )

    def complete_then_mutate(*args, **kwargs):
        result = _complete_fake_training(*args, **kwargs)
        supplied_schedule = args[4]
        supplied_schedule.factual_miss_anchors[
            0
        ].example.supervision.target.zero_()
        return result

    monkeypatch.setattr(
        runner,
        "execute_paired_formal_training",
        complete_then_mutate,
    )
    runtime = runner.PairedFormalRuntimeInputs(
        bundle=_FakeBundle(),
        pair_catalog=catalog,
        prepared_catalog=SimpleNamespace(),
        schedule=schedule,
        control_provider=None,
    )
    output = tmp_path / "post-change"
    with pytest.raises(ValueError, match="non-empty positive target"):
        runner.run_paired_formal_attempt(
            config,
            _FakePreflight(),
            runtime,
            method="paired_difference",
            seed=42,
            output_dir=output,
            device=torch.device("cpu"),
        )
    assert (output / ".INCOMPLETE.json").is_file()
    assert not (output / "decoder_artifact").exists()
    assert not (output / "COMPLETE.json").exists()


def test_production_api_has_no_horizon_or_evaluation_override() -> None:
    parameters = inspect.signature(
        runner.run_paired_formal_attempt
    ).parameters
    assert set(parameters) == {
        "protocol",
        "preflight",
        "runtime",
        "method",
        "seed",
        "output_dir",
        "device",
    }
    forbidden = {
        "epochs",
        "steps",
        "steps_per_epoch",
        "max_updates",
        "checkpoint",
        "resume",
        "D_V",
        "D_T",
        "calibration",
        "inference",
        "wave",
    }
    assert not forbidden.intersection(parameters)
    source = inspect.getsource(runner.run_paired_formal_attempt)
    assert "32_000" not in source or "execute_paired_formal_training" in source
