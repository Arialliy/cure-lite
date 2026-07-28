#!/usr/bin/env python3
"""Verify the frozen PACRE-VC v23 D_V terminal with one schema corrigendum.

The frozen verifier omitted the canonical ``formal_result_fingerprint`` key
from one exact-key whitelist.  This wrapper accepts exactly that one key,
checks it against the independently verified Formal terminal, delegates every
other model-binding check to the byte-frozen original function, and then runs
the original terminal verifier unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from tools import (  # noqa: E402
    verify_cure_lite_v23_pacre_vc_formal_d_v_receipt as frozen_verifier,
)


AUDIT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = AUDIT_ROOT / "verification.json"
ORIGINAL_VERIFIER_PATH = (
    ROOT / "tools/verify_cure_lite_v23_pacre_vc_formal_d_v_receipt.py"
)
D_V_OUTPUT_PATH = (
    ROOT
    / "runs/irstd1k_stage_a_seed42"
    / "cure_lite_pacre_v23_vc_formal_d_v_seed42_r1"
)
SOURCE_CLOSURE_PATH = (
    ROOT
    / "protocols/IRSTD-1K/pacre_v23_verifier_corrected"
    / "implementation_closure.json"
)

ORIGINAL_VERIFIER_SHA256 = (
    "bb85c2746589ce60f5b9d59c834cba00a4ac39580084bfee7f6813b87d442ce6"
)
SOURCE_CLOSURE_FINGERPRINT = (
    "d08a1d84348d8caf8ecee3b0fef3d5efcd56e05e50f46e25b1cf17bd71dfe48c"
)
EXPECTED_INPUT_SHA256 = {
    "claim.json": (
        "49782cfa6d4b4933733c476d89c3d827356ae573dbf940905a5fe2d19cb50cbf"
    ),
    "receipt.json": (
        "89527eb76eeded00a9fcf8a8cc96fc69fd8970f30ea53b13d7f11f43101fb1ad"
    ),
    "decision.json": (
        "8bb002b2b73c974e9bce0b03931625cc49e284d49afda9a8d75feb4aa9aa4b81"
    ),
    "COMPLETE.json": (
        "2a9b648933dd1d9635943f2a8debd255ac91635722efe17f75b85438ee43a86b"
    ),
}
LEGACY_FORMAL_ARTIFACT_KEYS = frozenset(
    {
        "artifact_fingerprint",
        "training_result_fingerprint",
        "authorization_fingerprint",
        "source_closure_fingerprint",
        "model_state_fingerprint",
        "model_config_fingerprint",
    }
)
CORRECTED_FORMAL_ARTIFACT_KEYS = (
    LEGACY_FORMAL_ARTIFACT_KEYS | {"formal_result_fingerprint"}
)
ORIGINAL_MODEL_BINDING_VERIFIER = frozen_verifier._verify_model_binding


def _read_strict_json(path: Path) -> dict[str, object]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.resolve(strict=True) != path
    ):
        raise RuntimeError(f"non-regular corrigendum input: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"corrigendum input must be an object: {path}")
    return payload


def _verify_frozen_inputs() -> dict[str, str]:
    if file_sha256(ORIGINAL_VERIFIER_PATH) != ORIGINAL_VERIFIER_SHA256:
        raise RuntimeError("frozen original D_V verifier changed")
    closure = _read_strict_json(SOURCE_CLOSURE_PATH)
    if (
        closure.get("closure_fingerprint")
        != SOURCE_CLOSURE_FINGERPRINT
        or closure.get("D_V_accessed") is not False
        or closure.get("D_T_accessed") is not False
    ):
        raise RuntimeError("frozen source-closure receipt changed")
    observed = {
        name: file_sha256(D_V_OUTPUT_PATH / name)
        for name in sorted(EXPECTED_INPUT_SHA256)
    }
    if observed != {
        name: EXPECTED_INPUT_SHA256[name]
        for name in sorted(EXPECTED_INPUT_SHA256)
    }:
        raise RuntimeError("published D_V terminal files changed")
    return observed


def _verify_model_binding_corrigendum(
    binding: object,
    *,
    formal: Mapping[str, object],
    artifact: object,
) -> str:
    if not isinstance(binding, Mapping):
        raise TypeError("model binding must be a mapping")
    payload = dict(binding)
    formal_artifact = payload.get("formal_artifact")
    if (
        not isinstance(formal_artifact, Mapping)
        or set(formal_artifact) != CORRECTED_FORMAL_ARTIFACT_KEYS
        or formal_artifact.get("formal_result_fingerprint")
        != formal.get("formal_result_fingerprint")
    ):
        raise RuntimeError(
            "corrected formal_result_fingerprint binding changed"
        )

    legacy_payload = dict(payload)
    legacy_formal_artifact = dict(formal_artifact)
    del legacy_formal_artifact["formal_result_fingerprint"]
    legacy_payload["formal_artifact"] = legacy_formal_artifact

    delegated_fingerprint = ORIGINAL_MODEL_BINDING_VERIFIER(
        legacy_payload,
        formal=formal,
        artifact=artifact,
    )
    if delegated_fingerprint != stable_fingerprint(legacy_payload):
        raise RuntimeError("frozen model-binding verifier delegation changed")
    return stable_fingerprint(payload)


def _write_once(payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        OUTPUT_PATH,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    directory_descriptor = os.open(AUDIT_ROOT, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-once", action="store_true", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    before = _verify_frozen_inputs()
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"corrigendum receipt already exists: {OUTPUT_PATH}")

    frozen_verifier._verify_model_binding = _verify_model_binding_corrigendum
    verification_result = frozen_verifier.verify_terminal(D_V_OUTPUT_PATH)
    after = _verify_frozen_inputs()
    if after != before:
        raise RuntimeError("D_V terminal changed during corrigendum verification")

    correction = {
        "kind": "exact_key_whitelist_omission",
        "field": "formal_result_fingerprint",
        "legacy_formal_artifact_keys": sorted(LEGACY_FORMAL_ARTIFACT_KEYS),
        "corrected_formal_artifact_keys": sorted(
            CORRECTED_FORMAL_ARTIFACT_KEYS
        ),
        "delegates_all_remaining_checks_to_frozen_original": True,
        "performance_gate_changed": False,
        "threshold_changed": False,
    }
    receipt: dict[str, object] = {
        "schema_version": (
            "cure-lite-v23-pacre-vc-d-v-verifier-corrigendum-v1"
        ),
        "status": "CORRIGENDUM_TERMINAL_VERIFIED",
        "original_verifier_sha256": ORIGINAL_VERIFIER_SHA256,
        "source_closure_fingerprint": SOURCE_CLOSURE_FINGERPRINT,
        "frozen_D_V_terminal_file_sha256": after,
        "correction": correction,
        "correction_fingerprint": stable_fingerprint(correction),
        "verification_result": verification_result,
        "D_V_inference_reexecuted": False,
        "D_V_payload_reopened": False,
        "D_T_payload_accessed": False,
        "model_training_performed": False,
        "model_state_update_performed": False,
        "checkpoint_selection_performed": False,
        "D_T_authorized": False,
    }
    receipt["corrigendum_receipt_fingerprint"] = stable_fingerprint(receipt)
    _write_once(receipt)
    print(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
