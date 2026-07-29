from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from cure_lite_v24.oof_split import (
    require_verified_oof_fold_closure,
    verify_all_oof_fold_closures,
    verify_oof_fold_closure,
)
from tools.gcr_pacre_v24_protocol import verify_oof4_split_preregistration


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = (
    REPO_ROOT
    / "protocols/IRSTD-1K/gcr_pacre_v24/"
    "D_R_OOF4_split_preregistration.json"
)


def _split():
    return verify_oof4_split_preregistration(
        json.loads(SPLIT_PATH.read_text(encoding="utf-8")),
        repository_root=REPO_ROOT,
    )


def test_four_verified_closures_are_source_disjoint_and_cover_once() -> None:
    split = _split()
    closures = verify_all_oof_fold_closures(
        split,
        available_sample_ids=split.root_by_sample,
    )
    assert tuple(value.fold_id for value in closures) == (0, 1, 2, 3)
    assert {
        sample_id
        for closure in closures
        for sample_id in closure.held_out_sample_ids
    } == set(split.root_by_sample)
    for closure in closures:
        assert not (
            set(closure.train_root_source_ids)
            & set(closure.held_out_root_source_ids)
        )


def test_fold_closure_rejects_unknown_missing_and_retain_issuer_replace() -> None:
    split = _split()
    samples = list(split.root_by_sample)
    with pytest.raises(PermissionError, match="sample universe"):
        verify_oof_fold_closure(
            split,
            fold_id=0,
            available_sample_ids=[*samples[:-1], "unknown-sample"],
        )
    token = verify_oof_fold_closure(
        split,
        fold_id=0,
        available_sample_ids=samples,
    )
    forged = replace(token)
    assert forged._issuer is token._issuer
    with pytest.raises(TypeError, match="fixed OOF split expander"):
        require_verified_oof_fold_closure(forged)
