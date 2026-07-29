from __future__ import annotations

import json
from pathlib import Path

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite_v24.dr_gate import GCR_PACRE_DR_IMPLEMENTATION_PATHS


ROOT = Path(__file__).resolve().parents[1]
R1_MARKER = (
    ROOT
    / "runs/irstd1k_stage_a_seed42/"
    "gcr_pacre_v24_D_R_structural_run_start_"
    "3500d1fe352d0b7fc4622bab128f37f3fa6232769ee489fa49c4f9cd40e3063f"
    ".json"
)
EXPECTED_R1_SOURCE_CLOSURE_FINGERPRINT = (
    "28d26759a68785e9c99917fcfa8b36430c7f6e5463282d66eeab5c711e425e9f"
)


def test_new_outer_supervision_files_do_not_mutate_r1_103_file_closure() -> None:
    marker = json.loads(R1_MARKER.read_text(encoding="utf-8"))
    bound = marker["implementation_binding"]

    assert len(GCR_PACRE_DR_IMPLEMENTATION_PATHS) == 103
    assert len(bound) == 103
    assert set(bound) == set(GCR_PACRE_DR_IMPLEMENTATION_PATHS)
    assert marker["source_closure_fingerprint"] == (
        EXPECTED_R1_SOURCE_CLOSURE_FINGERPRINT
    )
    assert stable_fingerprint(bound) == EXPECTED_R1_SOURCE_CLOSURE_FINGERPRINT
    assert {
        relative: file_sha256(ROOT / relative)
        for relative in GCR_PACRE_DR_IMPLEMENTATION_PATHS
    } == bound

    new_paths = {
        "tools/cure_lite_v24_runtime_supervisor.py",
        (
            "deploy/systemd/"
            "cure-lite-v24-gcr-pacre-dr-r2.service.template"
        ),
    }
    assert new_paths.isdisjoint(GCR_PACRE_DR_IMPLEMENTATION_PATHS)
    for relative in new_paths:
        path = ROOT / relative
        assert path.is_file()
        assert not path.is_symlink()
