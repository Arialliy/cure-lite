"""Persistent cross-process attempt gate for paired bounded-400.

The fixed chain configuration is sealed after OOF4 and before bounded
training.  Its output directory is pre-created as empty infrastructure.
Starting the single bounded attempt then consists of creating the exact
``run_start.json`` path with ``O_EXCL``; the marker is never removed even if
the process crashes.  No model, optimizer, or dataset payload is opened here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Final, Mapping

import torch

from cure_lite.cache.schema import (
    canonical_json,
    file_sha256,
    stable_fingerprint,
)
from tools.gcr_pacre_v24_protocol import (
    PROTOCOL_ID,
    VerifiedAccessAudit,
    VerifiedOOFDecision,
    require_verified_access_audit,
    require_verified_oof_decision,
)

from .artifact_io import read_canonical_json
from .formal_cache_artifacts import (
    VerifiedFormalCacheArtifact,
    require_verified_formal_cache_artifact,
    require_verified_formal_cache_origin_artifact,
    verify_formal_cache_artifact,
)
from .source_closure import (
    GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
    gcr_pacre_v24_source_closure_fingerprint,
    gcr_pacre_v24_source_closure_hashes,
)


GCR_PACRE_BOUNDED_CHAIN_CONFIG_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-paired-bounded400-chain-config-v1"
)
GCR_PACRE_BOUNDED_RUN_START_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-paired-bounded400-persistent-run-start-v1"
)
GCR_PACRE_BOUNDED_PATH_POLICY: Final = (
    "fixed_runtime_root_bounded_paired_bounded400_run_start_json_v1"
)
GCR_PACRE_BOUNDED_RUNTIME_RELATIVE: Final = (
    "runs/irstd1k_stage_a_seed42/gcr_pacre_v24_evidence_r1"
)
GCR_PACRE_BOUNDED_CONFIG_RELATIVE: Final = (
    "bounded/execution_chain_config.json"
)
GCR_PACRE_BOUNDED_EPOCHS: Final = 10
GCR_PACRE_BOUNDED_STEPS_PER_EPOCH: Final = 40
GCR_PACRE_BOUNDED_UPDATES: Final = 400

_CHAIN_ISSUER = object()
_RUN_START_ISSUER = object()
_CHAIN_REGISTRY: dict[int, tuple[object, str]] = {}
_RUN_START_REGISTRY: dict[int, tuple[object, str]] = {}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _repository_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if (
        not root.is_dir()
        or root.is_symlink()
        or root.resolve(strict=True) != root
    ):
        raise RuntimeError("repository root is not canonical")
    return root


def _required_runtime_root() -> Path:
    return _repository_root() / GCR_PACRE_BOUNDED_RUNTIME_RELATIVE


def required_gcr_pacre_bounded_chain_config_path() -> Path:
    return _required_runtime_root() / GCR_PACRE_BOUNDED_CONFIG_RELATIVE


def required_gcr_pacre_bounded_output_directory() -> Path:
    return _required_runtime_root() / "bounded" / "paired_bounded400"


def required_gcr_pacre_bounded_run_start_path() -> Path:
    return required_gcr_pacre_bounded_output_directory() / "run_start.json"


def _canonical_device(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("bounded device must be non-empty text")
    return str(torch.device(value))


def _write_persistent_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (canonical_json(dict(payload)) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            directory_descriptor = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            raise
    if (
        path.read_bytes() != encoded
        or path.is_symlink()
        or path.resolve(strict=True) != path
        or path.stat().st_nlink != 1
        or path.stat().st_mode & 0o222
    ):
        raise RuntimeError("persistent bounded JSON failed verification")


def _ensure_infrastructure(*, must_be_empty: bool) -> None:
    runtime = _required_runtime_root()
    bounded = runtime / "bounded"
    output = required_gcr_pacre_bounded_output_directory()
    runtime.mkdir(parents=True, exist_ok=True)
    bounded.mkdir(parents=False, exist_ok=True)
    output.mkdir(parents=False, exist_ok=True)
    for directory in (runtime, bounded, output):
        if (
            not directory.is_dir()
            or directory.is_symlink()
            or directory.resolve(strict=True) != directory
        ):
            raise RuntimeError("bounded runtime directory is not canonical")
    if must_be_empty and any(output.iterdir()):
        raise FileExistsError(
            "bounded fixed output must be empty before chain sealing"
        )


@dataclass(frozen=True, slots=True)
class VerifiedGCRPACREBoundedChainConfig:
    payload_json: str
    path: str
    file_sha256: str
    config_fingerprint: str
    source_closure_fingerprint: str
    _issuer: object = field(repr=False, compare=False)

    @property
    def payload(self) -> dict[str, object]:
        value = json.loads(self.payload_json)
        if not isinstance(value, dict):
            raise AssertionError("verified bounded chain config changed")
        return value


@dataclass(frozen=True, slots=True)
class GCRPACREBoundedRunStartToken:
    marker_path: str
    marker_size_bytes: int
    marker_file_sha256: str
    marker_device: int
    marker_inode: int
    marker_fingerprint: str
    chain_config_fingerprint: str
    authorization_fingerprint: str
    requested_device: str
    output_directory: str
    _issuer: object = field(repr=False, compare=False)


def _chain_token_payload(
    token: VerifiedGCRPACREBoundedChainConfig,
) -> dict[str, object]:
    return {
        "payload_json": token.payload_json,
        "path": token.path,
        "file_sha256": token.file_sha256,
        "config_fingerprint": token.config_fingerprint,
        "source_closure_fingerprint": token.source_closure_fingerprint,
    }


def _run_start_token_payload(
    token: GCRPACREBoundedRunStartToken,
) -> dict[str, object]:
    return {
        name: getattr(token, name)
        for name in token.__dataclass_fields__
        if name != "_issuer"
    }


def _register(
    registry: dict[int, tuple[object, str]],
    token: object,
    payload: Mapping[str, object],
) -> object:
    identity = id(token)
    if identity in registry:
        raise AssertionError("bounded capability identity was reused")
    registry[identity] = (token, stable_fingerprint(dict(payload)))
    return token


def _live(
    registry: dict[int, tuple[object, str]],
    token: object,
    issuer: object,
    payload: Mapping[str, object],
) -> bool:
    issued = registry.get(id(token))
    return (
        getattr(token, "_issuer", None) is issuer
        and issued is not None
        and issued[0] is token
        and issued[1] == stable_fingerprint(dict(payload))
    )


def _config_path(path: str | Path) -> Path:
    source = Path(path)
    required = required_gcr_pacre_bounded_chain_config_path()
    if (
        not source.is_absolute()
        or source != required
        or not source.is_file()
        or source.is_symlink()
        or source.resolve(strict=True) != source
        or source.stat().st_nlink != 1
        or source.stat().st_mode & 0o222
    ):
        raise PermissionError(
            f"bounded chain config must be immutable file {required}"
        )
    return source


def _cache_binding(
    cache: VerifiedFormalCacheArtifact,
) -> dict[str, object]:
    return {
        "receipt_fingerprint": cache.receipt_fingerprint,
        "cache_id": cache.cache_id,
        "path": cache.path,
        "file_sha256": cache.file_sha256,
        "device": cache.device,
        "inode": cache.inode,
        "hardlink_count": cache.hardlink_count,
        "semantic_cache_fingerprint": cache.semantic_cache_fingerprint,
        "neutral_payload_fingerprint": cache.neutral_payload_fingerprint,
    }


def _validate_config(
    payload: Mapping[str, object],
    *,
    path: Path,
) -> dict[str, object]:
    value = dict(payload)
    body = dict(value)
    fingerprint = body.pop("config_fingerprint", None)
    source_rows = gcr_pacre_v24_source_closure_hashes()
    source_fp = gcr_pacre_v24_source_closure_fingerprint(source_rows)
    output = required_gcr_pacre_bounded_output_directory()
    cache = value.get("full_D_R_cache_artifact")
    budget = value.get("budget")
    expected = {
        "schema_version",
        "protocol_id",
        "path_policy",
        "repository_root",
        "runtime_root",
        "chain_config_path",
        "source_closure_schema",
        "source_hashes",
        "source_closure_fingerprint",
        "predecessors",
        "access_audit_receipt",
        "full_D_R_cache_artifact",
        "requested_device",
        "output_directory",
        "run_start_marker_path",
        "authorization_artifact_path",
        "schedule_artifact_path",
        "control_terminal_artifact_path",
        "candidate_terminal_artifact_path",
        "result_artifact_path",
        "diagnostics_artifact_path",
        "decision_artifact_path",
        "budget",
        "attempt_policy",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "config_fingerprint",
    }
    predecessors = value.get("predecessors")
    access_receipt = value.get("access_audit_receipt")
    if (
        set(value) != expected
        or value.get("schema_version")
        != GCR_PACRE_BOUNDED_CHAIN_CONFIG_SCHEMA
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("path_policy") != GCR_PACRE_BOUNDED_PATH_POLICY
        or value.get("repository_root") != str(_repository_root())
        or value.get("runtime_root") != str(_required_runtime_root())
        or value.get("chain_config_path") != str(path)
        or path != required_gcr_pacre_bounded_chain_config_path()
        or value.get("source_closure_schema")
        != GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA
        or value.get("source_hashes") != dict(source_rows)
        or value.get("source_closure_fingerprint") != source_fp
        or not isinstance(predecessors, Mapping)
        or set(predecessors)
        != {
            "dataset_free_receipt_fingerprint",
            "D_R_structural_receipt_fingerprint",
            "OOF4_decision_fingerprint",
            "access_audit_receipt_fingerprint",
        }
        or any(not _is_sha256(item) for item in predecessors.values())
        or not isinstance(access_receipt, Mapping)
        or access_receipt.get("receipt_fingerprint")
        != predecessors.get("access_audit_receipt_fingerprint")
        or access_receipt.get("stage_id") != "paired_bounded400"
        or access_receipt.get("allowed_splits") != ["D_R"]
        or access_receipt.get("D_V_payload_accessed") is not False
        or access_receipt.get("D_T_payload_accessed") is not False
        or stable_fingerprint(
            {
                key: item
                for key, item in access_receipt.items()
                if key != "receipt_fingerprint"
            }
        )
        != access_receipt.get("receipt_fingerprint")
        or not isinstance(cache, Mapping)
        or set(cache)
        != {
            "receipt_fingerprint",
            "cache_id",
            "path",
            "file_sha256",
            "device",
            "inode",
            "hardlink_count",
            "semantic_cache_fingerprint",
            "neutral_payload_fingerprint",
        }
        or not all(
            _is_sha256(cache.get(name))
            for name in (
                "receipt_fingerprint",
                "file_sha256",
                "semantic_cache_fingerprint",
                "neutral_payload_fingerprint",
            )
        )
        or value.get("requested_device")
        != _canonical_device(str(value.get("requested_device")))
        or value.get("output_directory") != str(output)
        or value.get("run_start_marker_path")
        != str(output / "run_start.json")
        or value.get("authorization_artifact_path")
        != str(output / "authorization.json")
        or value.get("schedule_artifact_path")
        != str(output / "schedule.json")
        or value.get("control_terminal_artifact_path")
        != str(output / "control_terminal.safetensors")
        or value.get("candidate_terminal_artifact_path")
        != str(output / "candidate_terminal.safetensors")
        or value.get("result_artifact_path")
        != str(output / "bounded_400_result.json")
        or value.get("diagnostics_artifact_path")
        != str(output / "bounded_400_diagnostics.json")
        or value.get("decision_artifact_path")
        != str(output / "bounded_400_decision.json")
        or budget
        != {
            "seed": 42,
            "epochs": GCR_PACRE_BOUNDED_EPOCHS,
            "steps_per_epoch": GCR_PACRE_BOUNDED_STEPS_PER_EPOCH,
            "updates_per_arm": GCR_PACRE_BOUNDED_UPDATES,
            "training_invocations_per_arm": 1,
        }
        or value.get("attempt_policy")
        != {
            "from_scratch": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "persistent_O_EXCL_run_start_required": True,
            "run_start_marker_never_removed_after_creation": True,
            "checkpoint_policy": "final_only",
            "fixed_relative_promotion_threshold": None,
            "relative_diagnostics_authorize": False,
        }
        or value.get("D_V_payload_accessed") is not False
        or value.get("D_T_payload_accessed") is not False
        or not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(body)
        or not output.is_dir()
        or output.is_symlink()
        or output.resolve(strict=True) != output
    ):
        raise PermissionError("bounded chain config identity changed")
    return value


def seal_gcr_pacre_bounded_chain_config_new(
    *,
    oof_decision: VerifiedOOFDecision,
    access_audit: VerifiedAccessAudit,
    full_d_r_cache_artifact: VerifiedFormalCacheArtifact,
    dataset_free_receipt_fingerprint: str,
    d_r_structural_receipt_fingerprint: str,
    device: str,
) -> VerifiedGCRPACREBoundedChainConfig:
    """Seal the sole bounded attempt path and exact predecessor binding."""

    oof = require_verified_oof_decision(oof_decision)
    access = require_verified_access_audit(access_audit)
    cache = require_verified_formal_cache_origin_artifact(
        full_d_r_cache_artifact
    )
    reverified = verify_formal_cache_artifact(
        cache.path,
        cache_id=cache.cache_id,
        expected_semantic_cache_fingerprint=(
            cache.semantic_cache_fingerprint
        ),
        expected_neutral_payload_fingerprint=(
            cache.neutral_payload_fingerprint
        ),
    )
    if (
        oof.payload.get("gate_passed") is not True
        or access.stage_id != "paired_bounded400"
        or access.allowed_splits != ("D_R",)
        or cache.cache_id
        != "paired-bounded400-full-D_R-materialization"
        or reverified.receipt_fingerprint != cache.receipt_fingerprint
        or not _is_sha256(dataset_free_receipt_fingerprint)
        or not _is_sha256(d_r_structural_receipt_fingerprint)
    ):
        raise PermissionError("bounded chain predecessors are incoherent")
    _ensure_infrastructure(must_be_empty=True)
    target = required_gcr_pacre_bounded_chain_config_path()
    source_rows = gcr_pacre_v24_source_closure_hashes()
    output = required_gcr_pacre_bounded_output_directory()
    body = {
        "schema_version": GCR_PACRE_BOUNDED_CHAIN_CONFIG_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "path_policy": GCR_PACRE_BOUNDED_PATH_POLICY,
        "repository_root": str(_repository_root()),
        "runtime_root": str(_required_runtime_root()),
        "chain_config_path": str(target),
        "source_closure_schema": GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
        "source_hashes": dict(source_rows),
        "source_closure_fingerprint": (
            gcr_pacre_v24_source_closure_fingerprint(source_rows)
        ),
        "predecessors": {
            "dataset_free_receipt_fingerprint": (
                dataset_free_receipt_fingerprint
            ),
            "D_R_structural_receipt_fingerprint": (
                d_r_structural_receipt_fingerprint
            ),
            "OOF4_decision_fingerprint": oof.decision_fingerprint,
            "access_audit_receipt_fingerprint": access.receipt_fingerprint,
        },
        "access_audit_receipt": access.payload,
        "full_D_R_cache_artifact": _cache_binding(cache),
        "requested_device": _canonical_device(device),
        "output_directory": str(output),
        "run_start_marker_path": str(output / "run_start.json"),
        "authorization_artifact_path": str(output / "authorization.json"),
        "schedule_artifact_path": str(output / "schedule.json"),
        "control_terminal_artifact_path": str(
            output / "control_terminal.safetensors"
        ),
        "candidate_terminal_artifact_path": str(
            output / "candidate_terminal.safetensors"
        ),
        "result_artifact_path": str(output / "bounded_400_result.json"),
        "diagnostics_artifact_path": str(
            output / "bounded_400_diagnostics.json"
        ),
        "decision_artifact_path": str(
            output / "bounded_400_decision.json"
        ),
        "budget": {
            "seed": 42,
            "epochs": GCR_PACRE_BOUNDED_EPOCHS,
            "steps_per_epoch": GCR_PACRE_BOUNDED_STEPS_PER_EPOCH,
            "updates_per_arm": GCR_PACRE_BOUNDED_UPDATES,
            "training_invocations_per_arm": 1,
        },
        "attempt_policy": {
            "from_scratch": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "persistent_O_EXCL_run_start_required": True,
            "run_start_marker_never_removed_after_creation": True,
            "checkpoint_policy": "final_only",
            "fixed_relative_promotion_threshold": None,
            "relative_diagnostics_authorize": False,
        },
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    payload = {**body, "config_fingerprint": stable_fingerprint(body)}
    _write_persistent_json(target, payload)
    return load_and_verify_gcr_pacre_bounded_chain_config(target)


def load_and_verify_gcr_pacre_bounded_chain_config(
    path: str | Path,
) -> VerifiedGCRPACREBoundedChainConfig:
    source = _config_path(path)
    payload = _validate_config(read_canonical_json(source), path=source)
    token = VerifiedGCRPACREBoundedChainConfig(
        payload_json=canonical_json(payload),
        path=str(source),
        file_sha256=file_sha256(source),
        config_fingerprint=str(payload["config_fingerprint"]),
        source_closure_fingerprint=str(
            payload["source_closure_fingerprint"]
        ),
        _issuer=_CHAIN_ISSUER,
    )
    return _register(
        _CHAIN_REGISTRY,
        token,
        _chain_token_payload(token),
    )  # type: ignore[return-value]


def require_verified_gcr_pacre_bounded_chain_config(
    token: object,
) -> VerifiedGCRPACREBoundedChainConfig:
    if (
        type(token) is not VerifiedGCRPACREBoundedChainConfig
        or not _live(
            _CHAIN_REGISTRY,
            token,
            _CHAIN_ISSUER,
            _chain_token_payload(token),
        )
    ):
        raise TypeError(
            "bounded chain config must be issued by its fixed verifier"
        )
    current = load_and_verify_gcr_pacre_bounded_chain_config(token.path)
    if (
        current.payload_json != token.payload_json
        or current.file_sha256 != token.file_sha256
        or current.config_fingerprint != token.config_fingerprint
    ):
        raise PermissionError("bounded chain config bytes changed")
    return token


def verify_gcr_pacre_bounded_chain_authorization_binding(
    chain_config: VerifiedGCRPACREBoundedChainConfig,
    *,
    oof_decision: VerifiedOOFDecision,
    access_audit: VerifiedAccessAudit,
    full_d_r_cache_artifact: VerifiedFormalCacheArtifact,
    dataset_free_receipt_fingerprint: str,
    d_r_structural_receipt_fingerprint: str,
) -> dict[str, object]:
    chain = require_verified_gcr_pacre_bounded_chain_config(chain_config)
    oof = require_verified_oof_decision(oof_decision)
    access = require_verified_access_audit(access_audit)
    cache = require_verified_formal_cache_artifact(
        full_d_r_cache_artifact
    )
    predecessors = chain.payload["predecessors"]
    if (
        not isinstance(predecessors, Mapping)
        or predecessors.get("dataset_free_receipt_fingerprint")
        != dataset_free_receipt_fingerprint
        or predecessors.get("D_R_structural_receipt_fingerprint")
        != d_r_structural_receipt_fingerprint
        or predecessors.get("OOF4_decision_fingerprint")
        != oof.decision_fingerprint
        or predecessors.get("access_audit_receipt_fingerprint")
        != access.receipt_fingerprint
        or chain.payload.get("full_D_R_cache_artifact")
        != _cache_binding(cache)
    ):
        raise PermissionError(
            "bounded authorization differs from frozen chain config"
        )
    return chain.payload


def create_gcr_pacre_bounded_run_start_marker(
    authorization: object,
) -> GCRPACREBoundedRunStartToken:
    from .bounded_runner import GCRPACREBoundedAuthorization

    if type(authorization) is not GCRPACREBoundedAuthorization:
        raise TypeError("authorization must be exact bounded authorization")
    authorization.verify_unchanged()
    output = Path(authorization.output_directory)
    marker = required_gcr_pacre_bounded_run_start_path()
    if (
        output != required_gcr_pacre_bounded_output_directory()
        or not output.is_dir()
        or output.is_symlink()
        or output.resolve(strict=True) != output
        or any(output.iterdir())
    ):
        raise FileExistsError(
            "bounded fixed output is not pristine before persistent claim"
        )
    source_rows = gcr_pacre_v24_source_closure_hashes()
    source_fp = gcr_pacre_v24_source_closure_fingerprint(source_rows)
    intent = {
        "execution_kind": "paired_bounded400_D_R_training",
        "split": "D_R",
        "requested_device": authorization.requested_device,
        "output_directory": authorization.output_directory,
        "seed": 42,
        "epochs": GCR_PACRE_BOUNDED_EPOCHS,
        "steps_per_epoch": GCR_PACRE_BOUNDED_STEPS_PER_EPOCH,
        "optimizer_steps_authorized_per_arm": (
            GCR_PACRE_BOUNDED_UPDATES
        ),
        "parameter_updates_authorized_per_arm": (
            GCR_PACRE_BOUNDED_UPDATES
        ),
        "training_invocations_authorized_per_arm": 1,
        "from_scratch": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "D_V_materialization_intended": False,
        "D_T_materialization_intended": False,
    }
    body = {
        "schema_version": GCR_PACRE_BOUNDED_RUN_START_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "path_policy": GCR_PACRE_BOUNDED_PATH_POLICY,
        "marker_path": str(marker),
        "stage_id": "paired_bounded400",
        "chain_config": {
            "path": authorization.chain_config.path,
            "file_sha256": authorization.chain_config.file_sha256,
            "config_fingerprint": (
                authorization.chain_config.config_fingerprint
            ),
        },
        "authorization_fingerprint": (
            authorization.authorization_fingerprint
        ),
        "OOF4_decision_fingerprint": (
            authorization.oof_decision.decision_fingerprint
        ),
        "access_audit_receipt_fingerprint": (
            authorization.access_audit.receipt_fingerprint
        ),
        "full_D_R_cache_artifact": _cache_binding(
            authorization.full_d_r_cache_artifact
        ),
        "source_closure": {
            "schema_version": GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
            "fingerprint": source_fp,
            "source_hashes": dict(source_rows),
        },
        "intent": intent,
        "intent_fingerprint": stable_fingerprint(intent),
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    payload = {**body, "marker_fingerprint": stable_fingerprint(body)}
    _write_persistent_json(marker, payload)
    stat_result = marker.stat()
    token = GCRPACREBoundedRunStartToken(
        marker_path=str(marker),
        marker_size_bytes=stat_result.st_size,
        marker_file_sha256=file_sha256(marker),
        marker_device=stat_result.st_dev,
        marker_inode=stat_result.st_ino,
        marker_fingerprint=str(payload["marker_fingerprint"]),
        chain_config_fingerprint=(
            authorization.chain_config.config_fingerprint
        ),
        authorization_fingerprint=authorization.authorization_fingerprint,
        requested_device=authorization.requested_device,
        output_directory=authorization.output_directory,
        _issuer=_RUN_START_ISSUER,
    )
    return _register(
        _RUN_START_REGISTRY,
        token,
        _run_start_token_payload(token),
    )  # type: ignore[return-value]


def verify_gcr_pacre_bounded_run_start_token(
    authorization: object,
    token: object,
) -> dict[str, object]:
    from .bounded_runner import GCRPACREBoundedAuthorization

    if (
        type(authorization) is not GCRPACREBoundedAuthorization
        or type(token) is not GCRPACREBoundedRunStartToken
        or not _live(
            _RUN_START_REGISTRY,
            token,
            _RUN_START_ISSUER,
            _run_start_token_payload(token),
        )
    ):
        raise TypeError("run_start_token is not a live bounded capability")
    authorization.verify_unchanged()
    path = Path(token.marker_path)
    payload = read_canonical_json(path)
    body = dict(payload)
    marker_fp = body.pop("marker_fingerprint", None)
    stat_result = path.stat()
    if (
        path != required_gcr_pacre_bounded_run_start_path()
        or path.is_symlink()
        or path.resolve(strict=True) != path
        or stat_result.st_nlink != 1
        or stat_result.st_mode & 0o222
        or stat_result.st_size != token.marker_size_bytes
        or stat_result.st_dev != token.marker_device
        or stat_result.st_ino != token.marker_inode
        or file_sha256(path) != token.marker_file_sha256
        or marker_fp != token.marker_fingerprint
        or marker_fp != stable_fingerprint(body)
        or token.chain_config_fingerprint
        != authorization.chain_config.config_fingerprint
        or token.authorization_fingerprint
        != authorization.authorization_fingerprint
        or token.requested_device != authorization.requested_device
        or token.output_directory != authorization.output_directory
        or payload.get("authorization_fingerprint")
        != authorization.authorization_fingerprint
        or payload.get("D_V_payload_accessed") is not False
        or payload.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError("persistent bounded run-start marker changed")
    return {
        "path": str(path),
        "size_bytes": stat_result.st_size,
        "file_sha256": token.marker_file_sha256,
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "hardlink_count": stat_result.st_nlink,
        "marker_fingerprint": token.marker_fingerprint,
        "payload": payload,
    }


__all__ = [
    "GCR_PACRE_BOUNDED_CHAIN_CONFIG_SCHEMA",
    "GCR_PACRE_BOUNDED_PATH_POLICY",
    "GCR_PACRE_BOUNDED_RUN_START_SCHEMA",
    "GCRPACREBoundedRunStartToken",
    "VerifiedGCRPACREBoundedChainConfig",
    "create_gcr_pacre_bounded_run_start_marker",
    "load_and_verify_gcr_pacre_bounded_chain_config",
    "require_verified_gcr_pacre_bounded_chain_config",
    "required_gcr_pacre_bounded_chain_config_path",
    "required_gcr_pacre_bounded_output_directory",
    "required_gcr_pacre_bounded_run_start_path",
    "seal_gcr_pacre_bounded_chain_config_new",
    "verify_gcr_pacre_bounded_chain_authorization_binding",
    "verify_gcr_pacre_bounded_run_start_token",
]
