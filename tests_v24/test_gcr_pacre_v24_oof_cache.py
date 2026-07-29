from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import cure_lite_v24.oof_cache as oof_cache
import cure_lite_v24.oof_training as oof_training
from cure_lite.cache.schema import stable_fingerprint
from cure_lite_v24.oof_cache import (
    OOF_ARMS,
    issue_oof_cache_reader,
    load_oof_cache_payload,
    require_verified_oof_cache_artifact,
    save_oof_cache_artifact_new,
    seal_oof_training_terminals,
    verify_oof_six_cache_independence,
)
from cure_lite_v24.oof_evaluation import (
    OOFEvaluationDataset,
    seal_oof_evaluation_dataset,
    seal_oof_evaluation_sample,
)
from cure_lite_v24.oof_split import verify_oof_fold_closure
from tools.gcr_pacre_v24_protocol import verify_oof4_split_preregistration
from tests_v15.coverage_state_training_test_helpers import (
    make_training_scalar_cache,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = (
    REPO_ROOT
    / "protocols/IRSTD-1K/gcr_pacre_v24/"
    "D_R_OOF4_split_preregistration.json"
)
DIR_BY_ARM = {
    "base_eval": "base_eval",
    "PACRE_VC_v23_control": "v23_control",
    "GCR_PACRE_v24": "candidate",
}


def _fp(value: str) -> str:
    return stable_fingerprint({"generated": value})


def _closure():
    split = verify_oof4_split_preregistration(
        json.loads(SPLIT_PATH.read_text(encoding="utf-8")),
        repository_root=REPO_ROOT,
    )
    return verify_oof_fold_closure(
        split,
        fold_id=0,
        available_sample_ids=split.root_by_sample,
    )


def _generated_completed_capabilities():
    initial_parameters = (
        {
            "name": "generated",
            "shape": [1],
            "dtype": "torch.float32",
            "numel": 1,
            "byte_count": 4,
            "content_fingerprint": _fp("initial-parameter"),
        },
    )
    shared = {
        "completed_updates": 400,
        "run_start_marker_fingerprint": _fp("run-start"),
        "schedule_fingerprint": _fp("schedule"),
        "batch_sequence_fingerprint": _fp("batch-sequence"),
        "semantic_cache_fingerprint": _fp("semantic-cache"),
        "optimizer_config_fingerprint": _fp("adam"),
        "objective_policy_fingerprint": _fp("PMOPE"),
        "shared_initial_parameter_fingerprint": stable_fingerprint(
            list(initial_parameters)
        ),
        "initial_parameters": initial_parameters,
        "source_fingerprint": _fp("source"),
    }
    return {
        arm: SimpleNamespace(
            **shared,
            arm=arm,
            module_instance_id=_fp(f"{arm}-module-instance"),
            optimizer_instance_id=_fp(
                f"{arm}-optimizer-instance"
            ),
            parameter_storage_ledger=(
                {
                    "name": "generated",
                    "device": "cpu",
                    "nbytes": 4,
                    "storage_identity_fingerprint": _fp(
                        f"{arm}-storage"
                    ),
                },
            ),
            terminal_artifact_fingerprint=_fp(f"{arm}-terminal"),
            capability_fingerprint=_fp(f"{arm}-capability"),
            payload={
                "terminal_artifact": {
                    "device": 1,
                    "inode": index + 1,
                }
            },
        )
        for index, arm in enumerate(
            ("PACRE_VC_v23_control", "GCR_PACRE_v24")
        )
    }


def _evaluation_dataset(
    *,
    fold_id: int,
    partition: str,
    closure_fingerprint: str,
) -> OOFEvaluationDataset:
    row = seal_oof_evaluation_sample(
        sample_id=f"generated-{partition}",
        root_source_id=f"generated-{partition}-root",
        base_probability=torch.zeros(1, 1, 4, 4),
        feature=torch.zeros(1, 64, 1, 1),
        gt_mask=torch.ones(1, 1, 4, 4, dtype=torch.bool),
        valid_mask=torch.ones(1, 1, 4, 4, dtype=torch.bool),
        anchor_miss_ids=(1,),
        reachable_anchor_miss_ids=(1,),
    )
    return seal_oof_evaluation_dataset(
        fold_id=fold_id,
        partition=partition,
        closure_fingerprint=closure_fingerprint,
        rows=(row,),
    )


def _make_six(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    closure = _closure()
    # The cache codec is under test here; exact real-fold population closure
    # is exercised separately by the OOF runner/protocol tests.
    monkeypatch.setattr(
        oof_cache,
        "_verify_payload_partition",
        lambda *_args, **_kwargs: None,
    )
    scalar_cache = make_training_scalar_cache()
    train_dataset = _evaluation_dataset(
        fold_id=0,
        partition="train",
        closure_fingerprint=closure.closure_fingerprint,
    )
    holdout_dataset = _evaluation_dataset(
        fold_id=0,
        partition="holdout",
        closure_fingerprint=closure.closure_fingerprint,
    )
    tokens = []
    for index, arm in enumerate(OOF_ARMS):
        directory = (
            tmp_path / "oof4/fold_0/train" / DIR_BY_ARM[arm]
        )
        directory.mkdir(parents=True)
        tokens.append(
            save_oof_cache_artifact_new(
                (
                    train_dataset
                    if arm == "base_eval"
                    else scalar_cache
                ),
                (directory / "cache.pt").resolve(),
                fold_closure=closure,
                partition="train",
                arm=arm,
                creation_event=1,
            )
        )
    monkeypatch.setattr(
        oof_training,
        "require_verified_oof_completed_400_capability",
        lambda value, **_kwargs: value,
    )
    seal = seal_oof_training_terminals(
        closure,
        completed_400_capabilities=(
            _generated_completed_capabilities()
        ),
    )
    for index, arm in enumerate(OOF_ARMS):
        directory = (
            tmp_path / "oof4/fold_0/holdout" / DIR_BY_ARM[arm]
        )
        directory.mkdir(parents=True)
        tokens.append(
            save_oof_cache_artifact_new(
                holdout_dataset,
                (directory / "cache.pt").resolve(),
                fold_closure=closure,
                partition="holdout",
                arm=arm,
                creation_event=4,
                terminal_seal=seal,
            )
        )
    return closure, seal, tokens


def test_six_cache_files_are_physical_and_holdout_open_is_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, seal, tokens = _make_six(tmp_path, monkeypatch)
    verified = verify_oof_six_cache_independence(tokens)
    assert len(verified.protocol_entries) == 6
    assert len(
        {
            (row["device"], row["inode"])
            for row in verified.protocol_entries
        }
    ) == 6
    holdout = next(
        value
        for value in tokens
        if value.partition == "holdout"
        and value.arm == "GCR_PACRE_v24"
    )
    with pytest.raises(TypeError, match="terminal_seal"):
        issue_oof_cache_reader(
            holdout,
            reader_id="OOF4_read_only_holdout_evaluator",
        )
    reader = issue_oof_cache_reader(
        holdout,
        reader_id="OOF4_read_only_holdout_evaluator",
        terminal_seal=seal,
    )
    payload = load_oof_cache_payload(reader)
    assert type(payload) is OOFEvaluationDataset
    assert payload.dataset_fingerprint == holdout.semantic_payload_fingerprint


def test_train_reader_cannot_open_holdout_or_another_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, tokens = _make_six(tmp_path, monkeypatch)
    train = next(
        value
        for value in tokens
        if value.partition == "train"
        and value.arm == "GCR_PACRE_v24"
    )
    with pytest.raises(PermissionError, match="allowlist"):
        issue_oof_cache_reader(
            train,
            reader_id="OOF4_read_only_holdout_evaluator",
        )
    with pytest.raises(PermissionError, match="allowlist"):
        issue_oof_cache_reader(
            train,
            reader_id="PACRE_VC_v23_control_train_runner",
        )


def test_cache_tokens_reject_retain_issuer_replace_and_use_safe_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, tokens = _make_six(tmp_path, monkeypatch)
    forged = replace(tokens[0])
    assert forged._issuer is tokens[0]._issuer
    with pytest.raises(TypeError, match="OOF cache verifier"):
        require_verified_oof_cache_artifact(forged)

    original_load = oof_cache.torch.load
    calls: list[dict[str, object]] = []

    def observed_load(*args, **kwargs):
        calls.append(dict(kwargs))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(oof_cache.torch, "load", observed_load)
    verify_oof_six_cache_independence(tokens)
    assert calls
    assert all(
        row.get("weights_only") is True
        and row.get("mmap") is False
        and row.get("map_location") == "cpu"
        for row in calls
    )
