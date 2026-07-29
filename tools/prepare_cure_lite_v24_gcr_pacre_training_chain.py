#!/usr/bin/env python3
"""Create the fixed bounded and Formal v24 training-chain configurations.

This command has two create-only stages.  ``seal-bounded`` is legal only
after the persisted OOF-4 result verifies and passes; it materializes the
single full-D_R bounded cache and seals the bounded execution config.
``seal-formal`` is legal only after the persisted bounded receipt verifies
and passes; it creates two physically independent cache artifacts and seals
the two-seed Formal800 config.

No artifact path, retry switch, epoch count, performance margin, evaluator,
factory, D_V input, or D_T input is caller-configurable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import canonical_json, stable_fingerprint
from cure_lite.coverage_state_schedule import CoverageStateScheduleConfig
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_SEED,
    build_coverage_state_bounded_population,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    bind_coverage_state_real_dr_sources,
    build_coverage_state_real_dr_inputs,
)
from cure_lite_v24.artifact_io import read_canonical_json
from cure_lite_v24.bounded_run_start import (
    load_and_verify_gcr_pacre_bounded_chain_config,
    required_gcr_pacre_bounded_chain_config_path,
    seal_gcr_pacre_bounded_chain_config_new,
)
from cure_lite_v24.formal_cache_artifacts import (
    load_formal_scalar_cache_artifact,
    save_formal_cache_neutral_artifact_new,
)
from cure_lite_v24.formal_run_start import (
    required_gcr_pacre_formal_chain_config_path,
    seal_gcr_pacre_formal_chain_config_new,
)
from cure_lite_v24.oof_run_start import required_oof_dr_source_paths
from cure_lite_v24.real_input_factory import (
    _cache_from_binding,
    _rebuild_bounded_decision,
    _rebuild_oof_chain,
)
from cure_lite_v24.source_closure import (
    assert_gcr_pacre_v24_loaded_source_closure_complete,
    audit_gcr_pacre_v24_loaded_source_closure,
)
from tools.gcr_pacre_v24_protocol import verify_access_audit_receipt


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_EVIDENCE_ROOT = (
    _REPOSITORY_ROOT
    / "runs/irstd1k_stage_a_seed42/gcr_pacre_v24_evidence_r1"
)
_BOUNDED_CACHE_PATH = (
    _EVIDENCE_ROOT
    / "bounded/cache_materializations/full_D_R_cache.pt"
)
_FORMAL_CACHE_PATHS = {
    42: (
        _EVIDENCE_ROOT
        / "formal/cache_materializations/seed42_primary_full_D_R_cache.pt"
    ),
    43: (
        _EVIDENCE_ROOT
        / "formal/cache_materializations/"
        "seed43_training_integrity_only_full_D_R_cache.pt"
    ),
}


def _preflight_execution_device(value: object) -> str:
    """Validate and initialize one exact CPU or explicitly indexed CUDA device."""

    if type(value) is not str or not value:
        raise TypeError("device must be a non-empty string")
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid torch device: {value!r}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device type must be cpu or cuda")
    if device.type == "cuda":
        prefix, separator, index_text = value.partition(":")
        if (
            prefix != "cuda"
            or not separator
            or not index_text.isdecimal()
            or device.index is None
        ):
            raise ValueError("CUDA device index must be explicit")
        requested_index = int(index_text)
        if requested_index != device.index:
            raise ValueError("CUDA device index is out of range")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device_count = torch.cuda.device_count()
        if requested_index >= device_count:
            raise ValueError(
                "CUDA device index is out of range: "
                f"{requested_index} >= {device_count}"
            )
    try:
        torch.empty(1, device=device)
    except Exception as exc:
        raise RuntimeError(
            f"failed to initialize execution device {device}"
        ) from exc
    return str(device)


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be lowercase SHA256")
    return value


def _source_audit() -> dict[str, object]:
    assert_gcr_pacre_v24_loaded_source_closure_complete()
    audit = audit_gcr_pacre_v24_loaded_source_closure()
    if audit.get("missing_count") != 0 or audit.get("passed") is not True:
        raise RuntimeError("training-chain setup source closure is incomplete")
    return audit


def _access_token(
    *,
    stage_id: str,
    logical_id: str,
    purpose: str,
    source_fingerprint: str,
    source_manifest_fingerprint: str,
):
    events = [
        {
            "split": "D_R",
            "logical_id": logical_id,
            "purpose": purpose,
            "source_fingerprint": _sha256(
                source_fingerprint,
                name="access source fingerprint",
            ),
        }
    ]
    body = {
        "schema_version": "cure-lite-v24-split-access-audit-v1",
        "stage_id": stage_id,
        "allowed_splits": ["D_R"],
        "observed_payloads": events,
        "source_manifest_fingerprint": _sha256(
            source_manifest_fingerprint,
            name="source manifest fingerprint",
        ),
        "event_log_fingerprint": stable_fingerprint(events),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    receipt = {**body, "receipt_fingerprint": stable_fingerprint(body)}
    return verify_access_audit_receipt(
        receipt,
        expected_stage_id=stage_id,
        allowed_splits=("D_R",),
    )


def _create_parent_once(path: Path) -> None:
    parent = path.parent
    if path.exists() or path.is_symlink() or parent.exists():
        raise FileExistsError(
            f"create-only cache materialization path already used: {path}"
        )
    parent.mkdir(parents=True, exist_ok=False)
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or parent.resolve(strict=True) != parent
    ):
        raise RuntimeError("cache materialization directory is not canonical")


def _fixed_real_inputs():
    paths = required_oof_dr_source_paths()
    binding, _, _, _ = bind_coverage_state_real_dr_sources(**paths)
    real_inputs = build_coverage_state_real_dr_inputs(**paths)
    if real_inputs.source_binding.binding_fingerprint != (
        binding.binding_fingerprint
    ):
        raise RuntimeError("full-D_R materialization source binding changed")
    return binding, real_inputs


def seal_bounded(*, device: str) -> dict[str, object]:
    normalized_device = _preflight_execution_device(device)
    target = required_gcr_pacre_bounded_chain_config_path()
    if target.exists() or target.is_symlink():
        raise FileExistsError("bounded chain configuration already exists")
    oof, execution, _ = _rebuild_oof_chain()
    if oof.payload.get("gate_passed") is not True:
        raise PermissionError("bounded sealing requires OOF-4 PASS")
    # All OOF/factory/setup modules are now loaded.  Fail before reopening
    # the authorized full-D_R tensor graph.
    pre_materialization_audit = _source_audit()
    binding, real_inputs = _fixed_real_inputs()
    if binding.binding_fingerprint != execution.source_binding_fingerprint:
        raise PermissionError("OOF/full-D_R source binding changed")
    population = build_coverage_state_bounded_population(
        real_inputs.scalar_cache,
        seed=COVERAGE_STATE_BOUNDED_SEED,
    )
    cache = population.cache
    if cache.cache_fingerprint != execution.expected_full_cache_fingerprint:
        raise PermissionError(
            "full-D_R bounded cache differs from the structural receipt"
        )
    _create_parent_once(_BOUNDED_CACHE_PATH)
    cache_id = "paired-bounded400-full-D_R-materialization"
    artifact = save_formal_cache_neutral_artifact_new(
        cache,
        _BOUNDED_CACHE_PATH,
        cache_id=cache_id,
    )
    access = _access_token(
        stage_id="paired_bounded400",
        logical_id=cache_id,
        purpose="paired_bounded400_full_D_R_materialization",
        source_fingerprint=artifact.file_sha256,
        source_manifest_fingerprint=binding.binding_fingerprint,
    )
    d_r_receipt = read_canonical_json(execution.d_r_receipt_path)
    chain = seal_gcr_pacre_bounded_chain_config_new(
        oof_decision=oof,
        access_audit=access,
        full_d_r_cache_artifact=artifact,
        dataset_free_receipt_fingerprint=_sha256(
            d_r_receipt.get("dataset_free_receipt_fingerprint"),
            name="D_R dataset-free predecessor",
        ),
        d_r_structural_receipt_fingerprint=(
            execution.d_r_receipt_fingerprint
        ),
        device=normalized_device,
    )
    return {
        "mode": "seal-bounded",
        "chain_config": chain.path,
        "chain_config_fingerprint": chain.config_fingerprint,
        "cache_artifact": artifact.path,
        "cache_receipt_fingerprint": artifact.receipt_fingerprint,
        "pre_materialization_source_closure_audit": (
            pre_materialization_audit
        ),
        "final_source_closure_audit": _source_audit(),
        "epochs": CoverageStateScheduleConfig(
            seed=42,
            epochs=10,
            steps_per_epoch=40,
        ).epochs,
        "fixed_relative_uplift_threshold": None,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }


def seal_formal(
    *,
    seed42_device: str,
    seed43_device: str,
) -> dict[str, object]:
    normalized_seed42_device = _preflight_execution_device(seed42_device)
    normalized_seed43_device = _preflight_execution_device(seed43_device)
    target = required_gcr_pacre_formal_chain_config_path()
    if target.exists() or target.is_symlink():
        raise FileExistsError("Formal chain configuration already exists")
    bounded_chain = load_and_verify_gcr_pacre_bounded_chain_config(
        required_gcr_pacre_bounded_chain_config_path()
    )
    predecessors = _mapping(
        bounded_chain.payload.get("predecessors"),
        name="bounded predecessors",
    )
    oof, bounded = _rebuild_bounded_decision(
        expected_oof_decision_fingerprint=_sha256(
            predecessors.get("OOF4_decision_fingerprint"),
            name="bounded OOF predecessor",
        ),
    )
    if bounded.payload.get("gate_passed") is not True:
        raise PermissionError("Formal sealing requires bounded-400 PASS")
    _source_audit()
    bounded_artifact = _cache_from_binding(
        bounded_chain.payload.get("full_D_R_cache_artifact"),
        name="bounded full-D_R cache",
    )
    cache42 = load_formal_scalar_cache_artifact(bounded_artifact)
    cache43 = load_formal_scalar_cache_artifact(bounded_artifact)
    materialization_parent = _FORMAL_CACHE_PATHS[42].parent
    if (
        materialization_parent.exists()
        or materialization_parent.is_symlink()
        or _FORMAL_CACHE_PATHS[43].parent != materialization_parent
    ):
        raise FileExistsError(
            "Formal create-only cache materialization path already used"
        )
    materialization_parent.mkdir(parents=True, exist_ok=False)
    roles = {42: "primary", 43: "training_integrity_only"}
    caches = {42: cache42, 43: cache43}
    artifacts = {}
    accesses = {}
    for seed in (42, 43):
        role = roles[seed]
        cache_id = f"formal800-seed{seed}-{role}-full-D_R-cache"
        artifact = save_formal_cache_neutral_artifact_new(
            caches[seed],
            _FORMAL_CACHE_PATHS[seed],
            cache_id=cache_id,
        )
        artifacts[seed] = artifact
        accesses[seed] = _access_token(
            stage_id=f"formal800_seed{seed}_{role}",
            logical_id=cache_id,
            purpose=(
                "Formal800_seed42_primary_training_cache"
                if seed == 42
                else "Formal800_seed43_training_integrity_cache"
            ),
            source_fingerprint=artifact.file_sha256,
            source_manifest_fingerprint=(
                artifact.semantic_cache_fingerprint
            ),
        )
    chain = seal_gcr_pacre_formal_chain_config_new(
        oof_decision=oof,
        bounded_decision=bounded,
        seed42_access_audit=accesses[42],
        seed43_access_audit=accesses[43],
        seed42_cache_artifact=artifacts[42],
        seed43_cache_artifact=artifacts[43],
        dataset_free_receipt_fingerprint=_sha256(
            predecessors.get("dataset_free_receipt_fingerprint"),
            name="Formal dataset-free predecessor",
        ),
        d_r_structural_receipt_fingerprint=_sha256(
            predecessors.get("D_R_structural_receipt_fingerprint"),
            name="Formal D_R structural predecessor",
        ),
        seed42_device=normalized_seed42_device,
        seed43_device=normalized_seed43_device,
    )
    return {
        "mode": "seal-formal",
        "chain_config": chain.path,
        "chain_config_fingerprint": chain.config_fingerprint,
        "seed42_cache_artifact": artifacts[42].path,
        "seed43_cache_artifact": artifacts[43].path,
        "epochs_per_seed": 800,
        "steps_per_epoch": 40,
        "updates_per_seed": 32_000,
        "fixed_relative_uplift_threshold": None,
        "source_closure_audit": _source_audit(),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    bounded = subparsers.add_parser("seal-bounded")
    bounded.add_argument("--device", required=True)
    formal = subparsers.add_parser("seal-formal")
    formal.add_argument("--seed42-device", required=True)
    formal.add_argument("--seed43-device", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    value = (
        seal_bounded(device=arguments.device)
        if arguments.mode == "seal-bounded"
        else seal_formal(
            seed42_device=arguments.seed42_device,
            seed43_device=arguments.seed43_device,
        )
    )
    print(canonical_json(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
