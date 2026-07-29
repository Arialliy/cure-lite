from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import cure_lite_v24.oof_cache as oof_cache
from cure_lite.cache.schema import stable_fingerprint
from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
)
from cure_lite.experiment.coverage_state_training import (
    coverage_state_model_fingerprint,
)
from cure_lite.paired_types import tensor_content_fingerprint
from cure_lite_v23.factory import build_pacre_vc_training_model
from cure_lite_v23.pacre_vc import (
    CoverageStatePACREVerifierCorrectedConfig,
)
from cure_lite_v24.artifact_io import (
    atomic_write_new_canonical_json,
    read_canonical_json,
    save_terminal_safetensors_new,
)
from cure_lite_v24.factory import build_formal_gcr_pacre_training_model
from cure_lite_v24.oof_cache import (
    save_oof_cache_artifact_new,
    verify_oof_six_cache_independence,
)
from cure_lite_v24.oof_evaluation import (
    OOFConcreteEvaluator,
    OOF_ARMS,
    OOF_BASE_A_ARM,
    OOF_BASE_B_ARM,
    OOF_G1_ARM,
    OOF_V23_ARM,
    OOF_V24_ARM,
    mechanically_replay_oof_fold_evidence,
    seal_oof_evaluation_dataset,
    seal_oof_evaluation_sample,
)
from cure_lite_v24.oof_training import (
    OOF_CANDIDATE_ARM,
    OOF_CONTROL_ARM,
    load_oof_terminal_model_strict,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_training_scalar_cache,
)
from tools.gcr_pacre_v24_protocol import (
    BASE_A_THRESHOLD,
    BASE_B_THRESHOLD_GRID,
)


DIR_BY_ARM = {
    "base_eval": "base_eval",
    OOF_CONTROL_ARM: "v23_control",
    OOF_CANDIDATE_ARM: "candidate",
}


def _sample_ids(cache) -> tuple[str, ...]:
    return tuple(sorted(
        {
            row.record.sample_id for row in cache.natural_records
        }
        | {
            row.record.sample_id for row in cache.pair_records
        }
    ))


def _dataset(
    *,
    fold_id: int,
    partition: str,
    closure_fingerprint: str,
    sample_ids: tuple[str, ...],
) -> object:
    torch.manual_seed(900 + len(sample_ids))
    feature = torch.randn(1, 64, 1, 1)
    base = torch.zeros(1, 1, 4, 4)
    gt = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    gt[..., 1:3, 1:3] = True
    valid = torch.ones_like(gt)
    rows = tuple(
        seal_oof_evaluation_sample(
            sample_id=sample_id,
            root_source_id=f"root::{sample_id}",
            base_probability=base,
            feature=feature,
            gt_mask=gt,
            valid_mask=valid,
            anchor_miss_ids=(1,),
            reachable_anchor_miss_ids=(1,),
        )
        for sample_id in sample_ids
    )
    return seal_oof_evaluation_dataset(
        fold_id=fold_id,
        partition=partition,
        closure_fingerprint=closure_fingerprint,
        rows=rows,
    )


def _terminal(
    path: Path,
    *,
    arm: str,
) -> tuple[dict[str, object], torch.nn.Module]:
    torch.manual_seed(1234)
    model = (
        build_pacre_vc_training_model(
            CoverageStatePACREVerifierCorrectedConfig(
                feature_channels=64,
                feature_stride=4,
                width=32,
            )
        )
        if arm == OOF_CONTROL_ARM
        else build_formal_gcr_pacre_training_model()
    )
    model_fp = coverage_state_model_fingerprint(model)
    saved = save_terminal_safetensors_new(
        path,
        model,
        metadata={
            "schema": "generated-oof-terminal-v1",
            "run": "generated-oof-fold-0",
            "seed": "42",
            "role": arm,
            "arm": arm,
            "model_fingerprint": model_fp,
        },
    )
    return {
        **saved,
        "model_fingerprint": model_fp,
    }, model


def _build_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, object]]:
    runtime_root = (tmp_path / "runtime").resolve()
    fold = runtime_root / "fold_0"
    scalar = make_training_scalar_cache()
    samples = _sample_ids(scalar)
    holdout_samples = ("held-out-generated",)
    closure_fp = stable_fingerprint({"generated": "closure"})
    closure = SimpleNamespace(
        fold_id=0,
        closure_fingerprint=closure_fp,
        train_sample_ids=samples,
        held_out_sample_ids=holdout_samples,
        train_root_source_ids=tuple(f"root::{value}" for value in samples),
        held_out_root_source_ids=(
            f"root::{holdout_samples[0]}",
        ),
    )
    monkeypatch.setattr(
        oof_cache,
        "require_verified_oof_fold_closure",
        lambda value: value,
    )
    train_dataset = _dataset(
        fold_id=0,
        partition="train",
        closure_fingerprint=closure_fp,
        sample_ids=samples,
    )
    holdout_dataset = _dataset(
        fold_id=0,
        partition="holdout",
        closure_fingerprint=closure_fp,
        sample_ids=holdout_samples,
    )
    tokens = []
    for partition in ("train", "holdout"):
        for arm in ("base_eval", OOF_CONTROL_ARM, OOF_CANDIDATE_ARM):
            directory = fold / partition / DIR_BY_ARM[arm]
            directory.mkdir(parents=True, exist_ok=True)
            payload = (
                train_dataset
                if partition == "train" and arm == "base_eval"
                else (
                    scalar
                    if partition == "train"
                    else holdout_dataset
                )
            )
            seal = (
                None
                if partition == "train"
                else SimpleNamespace(
                    fold_id=0,
                    closure_fingerprint=closure_fp,
                    event_index=3,
                    seal_fingerprint=stable_fingerprint(
                        {"generated": "terminal-seal"}
                    ),
                )
            )
            if seal is not None:
                monkeypatch.setattr(
                    oof_cache,
                    "require_verified_oof_terminal_seal",
                    lambda value: value,
                )
            tokens.append(save_oof_cache_artifact_new(
                payload,
                (directory / "cache.pt").resolve(),
                fold_closure=closure,
                partition=partition,
                arm=arm,
                creation_event=1 if partition == "train" else 4,
                terminal_seal=seal,
            ))
    cache_set = verify_oof_six_cache_independence(tokens)
    entries = list(cache_set.protocol_entries)
    by_slot = {
        (row["partition"], row["arm"]): row for row in entries
    }

    schedule = build_coverage_state_training_schedule(
        scalar,
        CoverageStateScheduleConfig(
            seed=42,
            epochs=10,
            steps_per_epoch=40,
        ),
    )
    atomic_write_new_canonical_json(
        fold / "schedule.json",
        schedule.canonical_payload(),
    )
    batch_fp = stable_fingerprint(
        [
            selection.selection_fingerprint
            for selection in schedule.selections
        ]
    )
    access_fp = stable_fingerprint({"generated": "access"})
    evaluator = OOFConcreteEvaluator.fixed()
    selected, raw_rows = evaluator.select_base_b_train_only(train_dataset)
    normalized_rows = [
        {
            **row,
            "train_sample_ids": list(train_dataset.sample_ids),
            "train_root_source_ids": list(train_dataset.root_source_ids),
            "input_train_cache_fingerprint": by_slot[
                ("train", "base_eval")
            ]["file_sha256"],
            "access_audit_receipt_fingerprint": access_fp,
        }
        for row in raw_rows
    ]
    selector = [
        "maximize_pd",
        "maximize_retention",
        "minimize_pixel_fa",
        "minimize_raw_background_fa",
        "minimize_fp_components_per_mp",
        "maximize_threshold",
    ]
    base_selection = {
        "threshold_grid": list(BASE_B_THRESHOLD_GRID),
        "candidate_rows": normalized_rows,
        "candidate_ledger_fingerprint": stable_fingerprint(
            normalized_rows
        ),
        "selector_policy": selector,
        "selector_policy_fingerprint": stable_fingerprint(selector),
        "selected_threshold": selected,
    }

    terminal_dir = fold / "terminal"
    terminal_dir.mkdir()
    control_terminal, control_model = _terminal(
        terminal_dir / "v23_control_terminal.safetensors",
        arm=OOF_CONTROL_ARM,
    )
    candidate_terminal, candidate_model = _terminal(
        terminal_dir / "candidate_terminal.safetensors",
        arm=OOF_CANDIDATE_ARM,
    )
    ledgers = {
        OOF_BASE_A_ARM: evaluator.evaluate_base(
            holdout_dataset,
            threshold=BASE_A_THRESHOLD,
            arm=OOF_BASE_A_ARM,
        ),
        OOF_BASE_B_ARM: evaluator.evaluate_base(
            holdout_dataset,
            threshold=selected,
            arm=OOF_BASE_B_ARM,
        ),
        OOF_V23_ARM: evaluator.evaluate_model(
            holdout_dataset,
            control_model,
            arm=OOF_V23_ARM,
        ),
        OOF_V24_ARM: evaluator.evaluate_model(
            holdout_dataset,
            candidate_model,
            arm=OOF_V24_ARM,
        ),
        OOF_G1_ARM: evaluator.evaluate_model(
            holdout_dataset,
            candidate_model,
            arm=OOF_G1_ARM,
            forced_unit_gate=True,
        ),
    }
    evaluation_dir = fold / "evaluation"
    evaluation_dir.mkdir()
    for arm, ledger in ledgers.items():
        atomic_write_new_canonical_json(
            evaluation_dir / f"{arm}.json",
            {
                **ledger.canonical_payload(),
                "ledger_fingerprint": ledger.ledger_fingerprint,
            },
        )
    evaluation_fps = {
        OOF_BASE_A_ARM: stable_fingerprint(
            {"arm": OOF_BASE_A_ARM, "threshold": BASE_A_THRESHOLD}
        ),
        OOF_BASE_B_ARM: stable_fingerprint({
            "arm": OOF_BASE_B_ARM,
            "candidate_ledger_fingerprint": stable_fingerprint(
                normalized_rows
            ),
            "selected_threshold": selected,
        }),
        OOF_V23_ARM: control_terminal["model_fingerprint"],
        OOF_V24_ARM: candidate_terminal["model_fingerprint"],
        OOF_G1_ARM: candidate_terminal["model_fingerprint"],
    }
    sample = holdout_dataset.rows[0]
    anchor = (
        sample.base_probability >= BASE_A_THRESHOLD
    ) & sample.valid_mask
    factual_rows = [
        {
            "split": "D_R",
            "evidence_role": "factual_only",
            "fold_id": 0,
            "arm": arm,
            "sample_id": sample.sample_id,
            "root_source_id": sample.root_source_id,
            "gt_fingerprint": tensor_content_fingerprint(
                sample.gt_mask & sample.valid_mask
            ),
            "anchor_state_fingerprint": tensor_content_fingerprint(anchor),
            "evaluation_contract_fingerprint": evaluator.evaluator_fingerprint,
            "terminal_artifact_fingerprint": evaluation_fps[arm],
            "sufficient_statistics": dict(
                ledgers[arm].per_sample_rows[0]["statistics"]
            ),
        }
        for arm in OOF_ARMS
    ]
    atomic_write_new_canonical_json(
        fold / "factual_rows.json",
        {"rows": factual_rows},
    )
    receipt = {
        "fold_id": 0,
        "cache_entries": entries,
        "run_start_artifact": {
            "payload": {
                "schedule_fingerprint": schedule.schedule_fingerprint,
                "batch_sequence_fingerprint": batch_fp,
                "training_population_fingerprint": scalar.cache_fingerprint,
            },
        },
        "BaseB_train_fold_selection": base_selection,
        "training_arms": {
            OOF_CONTROL_ARM: {
                "terminal_artifact": control_terminal,
            },
            OOF_CANDIDATE_ARM: {
                "terminal_artifact": candidate_terminal,
            },
        },
        "evaluation_artifact_fingerprints": evaluation_fps,
        "access_audit_receipt_fingerprint": access_fp,
    }
    return runtime_root, receipt


def test_mechanical_replay_rebuilds_safe_caches_schedule_models_and_ledgers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root, receipt = _build_chain(tmp_path, monkeypatch)
    replay = mechanically_replay_oof_fold_evidence(
        receipt,
        runtime_root=runtime_root,
    )
    assert isinstance(replay, str) and len(replay) == 64


def test_mechanical_replay_rejects_self_resealed_fake_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root, receipt = _build_chain(tmp_path, monkeypatch)
    path = (
        runtime_root / "fold_0/evaluation/GCR_PACRE_v24.json"
    )
    payload = read_canonical_json(path)
    payload["pooled_statistics"]["matched_gt"] = (
        int(payload["pooled_statistics"]["matched_gt"]) + 1
    )
    body = dict(payload)
    body.pop("ledger_fingerprint")
    payload["ledger_fingerprint"] = stable_fingerprint(body)
    path.chmod(0o644)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o444)
    with pytest.raises(PermissionError, match="mechanical replay"):
        mechanically_replay_oof_fold_evidence(
            receipt,
            runtime_root=runtime_root,
        )


def test_mechanical_replay_rejects_changed_weight_after_terminal_reseal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_root, receipt = _build_chain(tmp_path, monkeypatch)
    terminal = receipt["training_arms"][OOF_CANDIDATE_ARM][
        "terminal_artifact"
    ]
    path = Path(terminal["path"])
    model = load_oof_terminal_model_strict(
        terminal,
        arm=OOF_CANDIDATE_ARM,
        expected_path=path,
    )
    with torch.no_grad():
        next(model.parameters()).view(-1)[0] += 0.25
    path.chmod(0o644)
    path.unlink()
    new_model_fp = coverage_state_model_fingerprint(model)
    saved = save_terminal_safetensors_new(
        path,
        model,
        metadata={
            "schema": "generated-oof-terminal-v1",
            "run": "generated-oof-fold-0",
            "seed": "42",
            "role": OOF_CANDIDATE_ARM,
            "arm": OOF_CANDIDATE_ARM,
            "model_fingerprint": new_model_fp,
        },
    )
    terminal.update(saved)
    terminal["model_fingerprint"] = new_model_fp
    with pytest.raises(PermissionError, match="mechanical replay"):
        mechanically_replay_oof_fold_evidence(
            receipt,
            runtime_root=runtime_root,
        )


def test_ordinary_bytes_and_wrong_terminal_weight_fail_strict_rebuild(
    tmp_path: Path,
) -> None:
    ordinary = tmp_path / "ordinary.safetensors"
    ordinary.write_bytes(b"ordinary bytes are not safetensors")
    ordinary.chmod(0o444)
    with pytest.raises(Exception):
        load_oof_terminal_model_strict(
            {
                "path": str(ordinary.resolve()),
                "state_keys": [
                    "joint_state_weight",
                    "joint_hidden_bias",
                    "scalar_energy_weight",
                ],
                "state_shapes": {},
                "state_dtypes": {},
                "parameter_count": 64_064,
                "model_fingerprint": "0" * 64,
            },
            arm=OOF_CANDIDATE_ARM,
            expected_path=ordinary.resolve(),
        )

    valid_path = tmp_path / "candidate_terminal.safetensors"
    terminal, _ = _terminal(valid_path, arm=OOF_CANDIDATE_ARM)
    terminal["model_fingerprint"] = "f" * 64
    with pytest.raises(RuntimeError, match="fingerprint"):
        load_oof_terminal_model_strict(
            terminal,
            arm=OOF_CANDIDATE_ARM,
            expected_path=valid_path.resolve(),
        )
