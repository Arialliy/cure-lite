"""Fixed cross-process input factory for the real v24 training stages.

The public entry point deliberately accepts no path, loader, evaluator, model
factory, threshold, or retry override.  Every file location comes from the
immutable bounded/Formal chain capability or from the frozen OOF protocol
location.  JSON artifacts are verifier inputs only: this module reconstructs
fresh private capabilities in every process before it loads the authorized
``D_R`` cache.

There is no ``D_V`` or ``D_T`` import, path constant, parameter, or loader in
this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Mapping

from cure_lite.coverage_state_schedule import (
    CoverageStateScheduleConfig,
    build_coverage_state_training_schedule,
)
from tools.gcr_pacre_v24_protocol import (
    VerifiedBoundedDecision,
    VerifiedOOFDecision,
    decide_paired_bounded400,
    require_verified_bounded_decision,
    require_verified_oof_decision,
    validate_paired_bounded_receipt,
    verify_access_audit_receipt,
    verify_oof4_split_preregistration,
)

from .artifact_io import read_canonical_json
from .bounded_run_start import (
    VerifiedGCRPACREBoundedChainConfig,
    load_and_verify_gcr_pacre_bounded_chain_config,
    require_verified_gcr_pacre_bounded_chain_config,
    required_gcr_pacre_bounded_chain_config_path,
)
from .bounded_runner import (
    GCRPACREBoundedAuthorization,
    GCR_PACRE_BOUNDED_EPOCHS,
    GCR_PACRE_BOUNDED_SEED,
    GCR_PACRE_BOUNDED_STEPS_PER_EPOCH,
    prepare_gcr_pacre_paired_bounded_authorization,
)
from .fixed_dr_evaluator import FrozenGCRPACREDREvaluator
from .formal_cache_artifacts import (
    VerifiedFormalCacheArtifact,
    load_formal_scalar_cache_artifact,
    verify_formal_cache_artifact,
)
from .formal_run_start import (
    VerifiedGCRPACREFormalChainConfig,
    require_verified_gcr_pacre_formal_chain_config,
)
from .formal_training import (
    GCRPACREFormalAuthorization,
    GCR_PACRE_FORMAL_EPOCHS,
    GCR_PACRE_FORMAL_SEED_ROLES,
    GCR_PACRE_FORMAL_STEPS_PER_EPOCH,
    prepare_gcr_pacre_formal_authorization,
)
from .gcr_pacre import CoverageStateGCRPACREConfig


_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
_OOF_SPLIT_PREREGISTRATION: Final = (
    _REPOSITORY_ROOT
    / "protocols/IRSTD-1K/gcr_pacre_v24/"
    "D_R_OOF4_split_preregistration.json"
)
_FORMAL_MODEL_COORDINATES: Final = (64, 4, 32)


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


def _cache_from_binding(
    raw: object,
    *,
    name: str,
) -> VerifiedFormalCacheArtifact:
    binding = _mapping(raw, name=name)
    cache_id = binding.get("cache_id")
    path = binding.get("path")
    if not isinstance(cache_id, str) or not cache_id:
        raise ValueError(f"{name}.cache_id is invalid")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise ValueError(f"{name}.path must be absolute")
    token = verify_formal_cache_artifact(
        path,
        cache_id=cache_id,
        expected_semantic_cache_fingerprint=_sha256(
            binding.get("semantic_cache_fingerprint"),
            name=f"{name}.semantic_cache_fingerprint",
        ),
        expected_neutral_payload_fingerprint=_sha256(
            binding.get("neutral_payload_fingerprint"),
            name=f"{name}.neutral_payload_fingerprint",
        ),
    )
    if (
        token.receipt_fingerprint
        != binding.get("receipt_fingerprint")
        or token.file_sha256 != binding.get("file_sha256")
        or token.device != binding.get("device")
        or token.inode != binding.get("inode")
        or token.hardlink_count != binding.get("hardlink_count")
    ):
        raise PermissionError(f"{name} differs from the sealed chain")
    return token


def _access_from_receipt(
    raw: object,
    *,
    stage_id: str,
):
    receipt = _mapping(raw, name=f"{stage_id} access receipt")
    return verify_access_audit_receipt(
        receipt,
        expected_stage_id=stage_id,
        allowed_splits=("D_R",),
    )


def _rebuild_oof_decision(
    *,
    expected_decision_fingerprint: str,
) -> VerifiedOOFDecision:
    """Replay the fixed OOF authorization/result chain in this process."""

    verified, _, _ = _rebuild_oof_chain()
    if verified.decision_fingerprint != _sha256(
        expected_decision_fingerprint,
        name="expected OOF decision fingerprint",
    ):
        raise PermissionError("OOF result differs from the sealed successor")
    return verified


def _rebuild_oof_chain():
    """Return freshly verified OOF decision/auth/source/split capabilities."""

    # Kept local so importing this factory cannot itself create or adopt an
    # OOF execution capability.  Both functions re-read the fixed artifacts
    # and issue new private tokens in the current process.
    from .oof_run_start import (
        load_and_verify_real_oof4_execution_authorization,
    )
    from .oof_runner import (
        verify_real_oof4_result_artifact,
    )

    split = verify_oof4_split_preregistration(
        _OOF_SPLIT_PREREGISTRATION,
        repository_root=_REPOSITORY_ROOT,
    )
    execution = load_and_verify_real_oof4_execution_authorization(
        verified_split=split,
    )
    decision = verify_real_oof4_result_artifact(
        verified_split=split,
        execution_authorization=execution,
    )
    verified = require_verified_oof_decision(decision)
    return verified, execution, split


def _rebuild_bounded_decision(
    *,
    expected_oof_decision_fingerprint: str,
    expected_bounded_decision_fingerprint: str | None = None,
) -> tuple[VerifiedOOFDecision, VerifiedBoundedDecision]:
    """Replay bounded receipt verification from its fixed chain config."""

    chain = load_and_verify_gcr_pacre_bounded_chain_config(
        required_gcr_pacre_bounded_chain_config_path()
    )
    payload = chain.payload
    predecessors = _mapping(
        payload.get("predecessors"),
        name="bounded predecessors",
    )
    oof = _rebuild_oof_decision(
        expected_decision_fingerprint=(
            expected_oof_decision_fingerprint
        )
    )
    if (
        predecessors.get("OOF4_decision_fingerprint")
        != oof.decision_fingerprint
    ):
        raise PermissionError("bounded/OOF predecessor binding changed")
    access = _access_from_receipt(
        payload.get("access_audit_receipt"),
        stage_id="paired_bounded400",
    )
    cache = _cache_from_binding(
        payload.get("full_D_R_cache_artifact"),
        name="bounded full-D_R cache",
    )
    result_path = payload.get("result_artifact_path")
    if not isinstance(result_path, str) or not Path(result_path).is_absolute():
        raise ValueError("bounded result path is not fixed")
    evidence = validate_paired_bounded_receipt(
        read_canonical_json(result_path),
        oof_decision=oof,
        access_audit=access,
        full_d_r_cache_artifact=cache,
        dataset_free_receipt_fingerprint=_sha256(
            predecessors.get("dataset_free_receipt_fingerprint"),
            name="dataset-free predecessor",
        ),
        d_r_structural_receipt_fingerprint=_sha256(
            predecessors.get("D_R_structural_receipt_fingerprint"),
            name="D_R structural predecessor",
        ),
        repository_root=_REPOSITORY_ROOT,
    )
    decision = require_verified_bounded_decision(
        decide_paired_bounded400(evidence)
    )
    if (
        expected_bounded_decision_fingerprint is not None
        and decision.decision_fingerprint
        != _sha256(
            expected_bounded_decision_fingerprint,
            name="expected bounded decision fingerprint",
        )
    ):
        raise PermissionError(
            "bounded result differs from the sealed Formal successor"
        )
    return oof, decision


def _candidate_config() -> CoverageStateGCRPACREConfig:
    channels, stride, width = _FORMAL_MODEL_COORDINATES
    return CoverageStateGCRPACREConfig(
        feature_channels=channels,
        feature_stride=stride,
        width=width,
    )


def _build_bounded_authorization(
    chain_config: object,
) -> GCRPACREBoundedAuthorization:
    chain = require_verified_gcr_pacre_bounded_chain_config(chain_config)
    payload = chain.payload
    predecessors = _mapping(
        payload.get("predecessors"),
        name="bounded predecessors",
    )
    oof = _rebuild_oof_decision(
        expected_decision_fingerprint=_sha256(
            predecessors.get("OOF4_decision_fingerprint"),
            name="bounded OOF predecessor",
        )
    )
    access = _access_from_receipt(
        payload.get("access_audit_receipt"),
        stage_id="paired_bounded400",
    )
    artifact = _cache_from_binding(
        payload.get("full_D_R_cache_artifact"),
        name="bounded full-D_R cache",
    )
    control_cache = load_formal_scalar_cache_artifact(artifact)
    candidate_cache = load_formal_scalar_cache_artifact(artifact)
    schedule = build_coverage_state_training_schedule(
        control_cache,
        CoverageStateScheduleConfig(
            seed=GCR_PACRE_BOUNDED_SEED,
            epochs=GCR_PACRE_BOUNDED_EPOCHS,
            steps_per_epoch=GCR_PACRE_BOUNDED_STEPS_PER_EPOCH,
        ),
    )
    return prepare_gcr_pacre_paired_bounded_authorization(
        oof_decision=oof,
        access_audit=access,
        full_d_r_cache_artifact=artifact,
        chain_config=chain,
        dataset_free_receipt_fingerprint=_sha256(
            predecessors.get("dataset_free_receipt_fingerprint"),
            name="bounded dataset-free predecessor",
        ),
        d_r_structural_receipt_fingerprint=_sha256(
            predecessors.get("D_R_structural_receipt_fingerprint"),
            name="bounded D_R structural predecessor",
        ),
        control_cache=control_cache,
        candidate_cache=candidate_cache,
        schedule=schedule,
        candidate_config=_candidate_config(),
        evaluator=FrozenGCRPACREDREvaluator(),
    )


def _build_formal_one(
    *,
    seed: int,
    chain: VerifiedGCRPACREFormalChainConfig,
    oof: VerifiedOOFDecision,
    bounded: VerifiedBoundedDecision,
) -> GCRPACREFormalAuthorization:
    try:
        role = dict(GCR_PACRE_FORMAL_SEED_ROLES)[seed]
    except KeyError as error:
        raise ValueError("Formal seed must be exactly 42 or 43") from error
    run = chain.run(seed)
    access = _access_from_receipt(
        run.get("access_audit_receipt"),
        stage_id=f"formal800_seed{seed}_{role}",
    )
    artifact = _cache_from_binding(
        run.get("cache_artifact"),
        name=f"Formal seed{seed} cache",
    )
    cache = load_formal_scalar_cache_artifact(artifact)
    schedule = build_coverage_state_training_schedule(
        cache,
        CoverageStateScheduleConfig(
            seed=seed,
            epochs=GCR_PACRE_FORMAL_EPOCHS,
            steps_per_epoch=GCR_PACRE_FORMAL_STEPS_PER_EPOCH,
        ),
    )
    predecessors = _mapping(
        chain.payload.get("predecessors"),
        name="Formal predecessors",
    )
    return prepare_gcr_pacre_formal_authorization(
        seed=seed,
        role=role,
        oof_decision=oof,
        bounded_decision=bounded,
        access_audit=access,
        cache_artifact=artifact,
        chain_config=chain,
        dataset_free_receipt_fingerprint=_sha256(
            predecessors.get("dataset_free_receipt_fingerprint"),
            name="Formal dataset-free predecessor",
        ),
        d_r_structural_receipt_fingerprint=_sha256(
            predecessors.get("D_R_structural_receipt_fingerprint"),
            name="Formal D_R structural predecessor",
        ),
        cache=cache,
        schedule=schedule,
        evaluator=FrozenGCRPACREDREvaluator(),
    )


def _build_formal_authorization(
    seed: int | None,
    chain_config: object,
):
    chain = require_verified_gcr_pacre_formal_chain_config(chain_config)
    predecessors = _mapping(
        chain.payload.get("predecessors"),
        name="Formal predecessors",
    )
    oof, bounded = _rebuild_bounded_decision(
        expected_oof_decision_fingerprint=_sha256(
            predecessors.get("OOF4_decision_fingerprint"),
            name="Formal OOF predecessor",
        ),
        expected_bounded_decision_fingerprint=_sha256(
            predecessors.get("paired_bounded400_decision_fingerprint"),
            name="Formal bounded predecessor",
        ),
    )
    if seed is None:
        return {
            formal_seed: _build_formal_one(
                seed=formal_seed,
                chain=chain,
                oof=oof,
                bounded=bounded,
            )
            for formal_seed in (42, 43)
        }
    if isinstance(seed, bool) or seed not in {42, 43}:
        raise ValueError("Formal seed must be exactly 42, 43, or None")
    return _build_formal_one(
        seed=seed,
        chain=chain,
        oof=oof,
        bounded=bounded,
    )


def build_gcr_pacre_v24_stage_authorization(
    seed_or_chain_config: int
    | None
    | VerifiedGCRPACREBoundedChainConfig,
    chain_config: VerifiedGCRPACREFormalChainConfig | None = None,
):
    """Build the exact bounded or Formal authorization expected by the CLIs.

    ``factory(bounded_chain)`` returns one paired bounded-400 authorization.
    ``factory(seed, formal_chain)`` returns one Formal800 authorization, while
    ``factory(None, formal_chain)`` returns the independently materialized
    seed-42/seed-43 pair used by finalization.
    """

    if chain_config is None:
        if type(seed_or_chain_config) is not VerifiedGCRPACREBoundedChainConfig:
            raise TypeError(
                "one-argument factory use requires the exact bounded chain "
                "capability"
            )
        return _build_bounded_authorization(seed_or_chain_config)
    if (
        type(chain_config) is not VerifiedGCRPACREFormalChainConfig
        or (
            seed_or_chain_config is not None
            and (
                isinstance(seed_or_chain_config, bool)
                or not isinstance(seed_or_chain_config, int)
            )
        )
    ):
        raise TypeError(
            "two-argument factory use requires seed/None and the exact "
            "Formal chain capability"
        )
    return _build_formal_authorization(
        seed_or_chain_config,
        chain_config,
    )


__all__ = ["build_gcr_pacre_v24_stage_authorization"]
