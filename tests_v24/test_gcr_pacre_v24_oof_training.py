from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import cure_lite_v24.oof_cache as oof_cache
import cure_lite_v24.oof_training as oof_training
from cure_lite.cache.schema import stable_fingerprint
from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
)
from cure_lite_v24.gcr_pacre import CoverageStateGCRPACREConfig
from cure_lite_v24.oof_cache import (
    issue_oof_cache_reader,
    save_oof_cache_artifact_new,
    seal_oof_training_terminals,
)
from cure_lite_v24.oof_training import (
    OOF_CANDIDATE_ARM,
    OOF_CONTROL_ARM,
    require_verified_oof_completed_400_capability,
    run_paired_oof_training_400,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_training_scalar_cache,
)


def _sample_ids(cache) -> tuple[str, ...]:
    return tuple(sorted(
        {
            value.record.sample_id for value in cache.natural_records
        }
        | {
            value.record.sample_id for value in cache.pair_records
        }
    ))


def _generated_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fold = (tmp_path / "oof4/fold_0").resolve()
    for partition in ("train", "holdout"):
        for arm in ("base_eval", "v23_control", "candidate"):
            (fold / partition / arm).mkdir(parents=True)
    (fold / "terminal").mkdir()
    control_cache = make_training_scalar_cache()
    candidate_cache = deepcopy(control_cache)
    samples = _sample_ids(control_cache)
    closure_fp = stable_fingerprint({"generated_closure": str(tmp_path)})
    closure = SimpleNamespace(
        fold_id=0,
        closure_fingerprint=closure_fp,
        train_sample_ids=samples,
        held_out_sample_ids=("generated-holdout-never-opened",),
        train_root_source_ids=tuple(
            f"generated-root-{value}" for value in samples
        ),
        held_out_root_source_ids=("generated-holdout-root",),
    )
    monkeypatch.setattr(
        oof_cache,
        "require_verified_oof_fold_closure",
        lambda value: value,
    )
    monkeypatch.setattr(
        oof_training,
        "require_verified_oof_fold_closure",
        lambda value: value,
    )
    control_artifact = save_oof_cache_artifact_new(
        control_cache,
        (fold / "train/v23_control/cache.pt").resolve(),
        fold_closure=closure,
        partition="train",
        arm=OOF_CONTROL_ARM,
        creation_event=1,
    )
    candidate_artifact = save_oof_cache_artifact_new(
        candidate_cache,
        (fold / "train/candidate/cache.pt").resolve(),
        fold_closure=closure,
        partition="train",
        arm=OOF_CANDIDATE_ARM,
        creation_event=1,
    )
    control_reader = issue_oof_cache_reader(
        control_artifact,
        reader_id="PACRE_VC_v23_control_train_runner",
    )
    candidate_reader = issue_oof_cache_reader(
        candidate_artifact,
        reader_id="GCR_PACRE_v24_train_runner",
    )
    schedule = build_coverage_state_training_schedule(
        control_cache,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=10,
            steps_per_epoch=40,
        ),
    )
    marker_fp = stable_fingerprint(
        {"generated_marker": str(tmp_path)}
    )
    run_start = SimpleNamespace(
        fold_id=0,
        closure_fingerprint=closure_fp,
        event_index=2,
        schedule_fingerprint=schedule.schedule_fingerprint,
        training_population_fingerprint=(
            schedule.cache_fingerprint
        ),
        output_directory=str(fold),
        marker_fingerprint=marker_fp,
        marker_path=str(fold / "run_start.json"),
        marker_file_sha256=stable_fingerprint(
            {"generated_marker_file": str(tmp_path)}
        ),
        authorization_fingerprint=stable_fingerprint(
            {"generated_authorization": str(tmp_path)}
        ),
        source_closure_fingerprint=stable_fingerprint(
            {"generated_source_closure": str(tmp_path)}
        ),
        payload={
            "event_index": 2,
            "batch_sequence_fingerprint": stable_fingerprint(
                [
                    selection.selection_fingerprint
                    for selection in schedule.selections
                ]
            ),
            "updates_per_arm": 400,
            "control_cache_artifact_fingerprint": (
                control_artifact.artifact_fingerprint
            ),
            "candidate_cache_artifact_fingerprint": (
                candidate_artifact.artifact_fingerprint
            ),
        },
    )
    monkeypatch.setattr(
        oof_training,
        "require_verified_oof_training_run_start",
        lambda value: value,
    )
    return SimpleNamespace(
        fold=fold,
        closure=closure,
        run_start=run_start,
        control_cache=control_cache,
        control_reader=control_reader,
        candidate_reader=candidate_reader,
        schedule=schedule,
        config=CoverageStateGCRPACREConfig(
            feature_channels=2,
            feature_stride=2,
            width=4,
        ),
    )


def _run_pair(generated):
    return run_paired_oof_training_400(
        fold_closure=generated.closure,
        run_start_token=generated.run_start,
        control_cache_reader=generated.control_reader,
        candidate_cache_reader=generated.candidate_reader,
        schedule=generated.schedule,
        candidate_config=generated.config,
        device="cpu",
    )


def test_generated_pair_really_runs_400_and_issues_strong_capabilities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _generated_pair(tmp_path, monkeypatch)
    result = _run_pair(generated)
    terminal_seal = seal_oof_training_terminals(
        generated.closure,
        completed_400_capabilities=(
            result.completed_400_capabilities
        ),
    )

    assert result.control_training_result.completed_updates == 400
    assert result.candidate_training_result.completed_updates == 400
    assert len(result.control_training_result.epoch_logs) == 10
    assert len(result.candidate_training_result.epoch_logs) == 10
    assert (
        result.control_training_result.initial_model_fingerprint
        == result.candidate_training_result.initial_model_fingerprint
    )
    assert (
        result.control_training_result.initial_model_fingerprint
        != result.control_training_result.final_model_fingerprint
    )
    assert (
        result.candidate_training_result.initial_model_fingerprint
        != result.candidate_training_result.final_model_fingerprint
    )
    assert {
        value.arm
        for value in result.completed_400_capabilities.values()
    } == {OOF_CONTROL_ARM, OOF_CANDIDATE_ARM}
    assert terminal_seal.fold_id == 0
    assert terminal_seal.run_start_marker_fingerprint == (
        generated.run_start.marker_fingerprint
    )
    assert dict(
        terminal_seal.completed_400_capability_fingerprints
    ) == {
        arm: value.capability_fingerprint
        for arm, value in result.completed_400_capabilities.items()
    }
    for arm, capability in result.completed_400_capabilities.items():
        assert (
            require_verified_oof_completed_400_capability(
                capability,
                arm=arm,
            )
            is capability
        )
        terminal = Path(capability.terminal_artifact_path)
        assert terminal.is_file()
        assert terminal.stat().st_nlink == 1
        assert terminal.stat().st_mode & 0o222 == 0
        assert capability.completed_updates == 400
        assert capability.payload["holdout_payload_accessed"] is False
        assert capability.payload["D_V_payload_accessed"] is False
        assert capability.payload["D_T_payload_accessed"] is False
    assert (
        result.control_terminal_artifact["device"],
        result.control_terminal_artifact["inode"],
    ) != (
        result.candidate_terminal_artifact["device"],
        result.candidate_terminal_artifact["inode"],
    )
    assert result.control_capability.module_instance_id != (
        result.candidate_capability.module_instance_id
    )
    assert result.control_capability.optimizer_instance_id != (
        result.candidate_capability.optimizer_instance_id
    )
    assert not (
        {
            row["storage_identity_fingerprint"]
            for row in (
                result.control_capability.parameter_storage_ledger
            )
        }
        & {
            row["storage_identity_fingerprint"]
            for row in (
                result.candidate_capability.parameter_storage_ledger
            )
        }
    )
    assert not any(
        (generated.fold / "holdout" / arm / "cache.pt").exists()
        for arm in ("base_eval", "v23_control", "candidate")
    )
    forged = replace(result.control_capability)
    with pytest.raises(TypeError, match="paired OOF runner"):
        require_verified_oof_completed_400_capability(forged)
    with pytest.raises(PermissionError, match="already consumed"):
        _run_pair(generated)


def test_initial_byte_mismatch_fails_before_any_pmope_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _generated_pair(tmp_path, monkeypatch)
    original_factory = oof_training.build_gcr_pacre_training_model
    step_called = False

    def mismatched_factory(config):
        model = original_factory(config)
        with torch.no_grad():
            next(model.parameters()).view(-1)[0] += 1.0
        return model

    def forbidden_step(*args, **kwargs):
        nonlocal step_called
        step_called = True
        raise AssertionError("PMOPE step must not run")

    monkeypatch.setattr(
        oof_training,
        "build_gcr_pacre_training_model",
        mismatched_factory,
    )
    monkeypatch.setattr(
        oof_training,
        "coverage_state_fused_train_step",
        forbidden_step,
    )
    with pytest.raises(RuntimeError, match="initial parameter"):
        _run_pair(generated)
    assert step_called is False
    assert not any((generated.fold / "terminal").iterdir())


def test_preexisting_exact_terminal_is_rejected_before_model_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _generated_pair(tmp_path, monkeypatch)
    existing = (
        generated.fold
        / "terminal"
        / "v23_control_terminal.safetensors"
    )
    with existing.open("xb") as handle:
        handle.write(b"generated-preexisting-terminal")
    allocated = False

    def forbidden_factory(*args, **kwargs):
        nonlocal allocated
        allocated = True
        raise AssertionError("model allocation must not occur")

    monkeypatch.setattr(
        oof_training,
        "build_pacre_vc_training_model",
        forbidden_factory,
    )
    with pytest.raises(FileExistsError, match="exact new frozen path"):
        _run_pair(generated)
    assert allocated is False


def test_scalar_cache_cannot_hide_a_full_or_wrong_fold_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = make_training_scalar_cache()
    samples = _sample_ids(cache)
    closure = SimpleNamespace(
        fold_id=0,
        closure_fingerprint=stable_fingerprint(
            {"generated": "wrong-population"}
        ),
        train_sample_ids=samples[:-1],
        held_out_sample_ids=(samples[-1],),
        train_root_source_ids=("train-root",),
        held_out_root_source_ids=("holdout-root",),
    )
    monkeypatch.setattr(
        oof_cache,
        "require_verified_oof_fold_closure",
        lambda value: value,
    )
    destination = (
        tmp_path / "oof4/fold_0/train/candidate/cache.pt"
    )
    destination.parent.mkdir(parents=True)
    with pytest.raises(PermissionError, match="sample population"):
        save_oof_cache_artifact_new(
            cache,
            destination.resolve(),
            fold_closure=closure,
            partition="train",
            arm=OOF_CANDIDATE_ARM,
            creation_event=1,
        )
    assert not destination.exists()
