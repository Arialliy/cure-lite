"""Frozen Formal800 execution configuration and persistent run-start markers.

The in-memory Formal authorization is deliberately insufficient to start a
real run.  A caller must also present a verifier-issued token for one sealed
two-seed chain configuration.  Immediately before any model/optimizer
allocation, the runner creates the seed-specific marker with ``O_EXCL``.
That marker is never removed, including when training fails, so a new process
cannot silently replay either Formal800 attempt.

This module handles metadata and already-verified cache/access capabilities
only.  It never opens D_R, D_V, or D_T payloads.
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
    VerifiedBoundedDecision,
    VerifiedOOFDecision,
    require_verified_access_audit,
    require_verified_bounded_decision,
    require_verified_oof_decision,
)

from .artifact_io import read_canonical_json
from .formal_cache_artifacts import (
    VerifiedFormalCacheArtifact,
    require_verified_formal_cache_artifact,
    verify_formal_cache_artifact,
    verify_formal_cache_pair_independence,
)
from .source_closure import (
    GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
    gcr_pacre_v24_source_closure_fingerprint,
    gcr_pacre_v24_source_closure_hashes,
)


GCR_PACRE_FORMAL_CHAIN_CONFIG_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-formal800-chain-config-v1"
)
GCR_PACRE_FORMAL_RUN_START_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-formal800-persistent-run-start-v2"
)
GCR_PACRE_FORMAL_RUNTIME_RELATIVE: Final = (
    "runs/irstd1k_stage_a_seed42/gcr_pacre_v24_evidence_r1"
)
GCR_PACRE_FORMAL_CHAIN_CONFIG_RELATIVE: Final = (
    "formal/execution_chain_config.json"
)
GCR_PACRE_FORMAL_RUN_START_PATH_POLICY: Final = (
    "fixed_runtime_root_seed_role_directory_run_start_json_v1"
)
GCR_PACRE_FORMAL_EPOCHS: Final = 800
GCR_PACRE_FORMAL_STEPS_PER_EPOCH: Final = 40
GCR_PACRE_FORMAL_UPDATES: Final = 32_000

_SEED_ROLES: Final = {
    42: "primary",
    43: "training_integrity_only",
}
_RUN_KEYS: Final = {
    42: "seed42_primary",
    43: "seed43_training_integrity_only",
}
_CHAIN_ISSUER = object()
_RUN_START_ISSUER = object()
_CHAIN_REGISTRY: dict[int, tuple[object, str]] = {}
_RUN_START_REGISTRY: dict[int, tuple[object, str]] = {}


def _formal_process_instance_fingerprint() -> str:
    """Bind one interpreter to its Linux process/start-time identity."""

    try:
        process_stat = Path("/proc/self/stat").read_text(
            encoding="utf-8"
        ).strip()
        closing_parenthesis = process_stat.rfind(")")
        fields_after_comm = process_stat[closing_parenthesis + 2 :].split()
        # /proc/[pid]/stat field 22 is process start time.  The suffix begins
        # at field 3, so its zero-based index is 19.
        start_time_ticks = fields_after_comm[19]
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except (OSError, IndexError, ValueError) as error:
        raise RuntimeError(
            "Formal independent-process identity is unavailable"
        ) from error
    if not start_time_ticks.isdigit() or not boot_id:
        raise RuntimeError("Formal process identity metadata is invalid")
    return stable_fingerprint(
        {
            "schema_version": (
                "cure-lite-v24-formal-independent-process-instance-v1"
            ),
            "boot_id": boot_id,
            "pid": os.getpid(),
            "process_start_time_ticks": start_time_ticks,
        }
    )


_PROCESS_INSTANCE_FINGERPRINT: Final = (
    _formal_process_instance_fingerprint()
)


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
        raise RuntimeError("repository root is not a canonical directory")
    return root


def _required_runtime_root() -> Path:
    return _repository_root() / GCR_PACRE_FORMAL_RUNTIME_RELATIVE


def required_gcr_pacre_formal_chain_config_path() -> Path:
    return (
        _required_runtime_root()
        / GCR_PACRE_FORMAL_CHAIN_CONFIG_RELATIVE
    )


def _run_directory(seed: int) -> Path:
    try:
        run_key = _RUN_KEYS[seed]
    except KeyError as error:
        raise ValueError("Formal seed must be exactly 42 or 43") from error
    return _required_runtime_root() / "formal" / run_key


def required_gcr_pacre_formal_run_start_path(seed: int) -> Path:
    return _run_directory(seed) / "run_start.json"


def _canonical_device(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("Formal device must be non-empty text")
    return str(torch.device(value))


def _regular_config_path(path: str | Path) -> Path:
    source = Path(path)
    required = required_gcr_pacre_formal_chain_config_path()
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
            f"Formal chain config must be the immutable file {required}"
        )
    return source


def _source_snapshot() -> tuple[tuple[str, str], ...]:
    return gcr_pacre_v24_source_closure_hashes()


def _run_binding(
    *,
    seed: int,
    device: str,
    access_audit: VerifiedAccessAudit,
    cache_artifact: VerifiedFormalCacheArtifact,
) -> dict[str, object]:
    role = _SEED_ROLES[seed]
    stage_id = f"formal800_seed{seed}_{role}"
    access = require_verified_access_audit(access_audit)
    cache = require_verified_formal_cache_artifact(cache_artifact)
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
        access.stage_id != stage_id
        or access.allowed_splits != ("D_R",)
        or reverified.receipt_fingerprint != cache.receipt_fingerprint
    ):
        raise PermissionError("Formal chain seed binding is incoherent")
    output = _run_directory(seed)
    return {
        "seed": seed,
        "role": role,
        "stage_id": stage_id,
        "requested_device": _canonical_device(device),
        "output_directory": str(output),
        "run_start_marker_path": str(output / "run_start.json"),
        "authorization_artifact_path": str(output / "authorization.json"),
        "schedule_artifact_path": str(output / "schedule.json"),
        "terminal_artifact_directory": str(output / "terminal"),
        "evidence_artifact_path": str(
            output / "formal800_evidence.json"
        ),
        "access_audit_receipt_fingerprint": access.receipt_fingerprint,
        "access_audit_receipt": access.payload,
        "cache_artifact": {
            "receipt_fingerprint": cache.receipt_fingerprint,
            "cache_id": cache.cache_id,
            "path": cache.path,
            "file_sha256": cache.file_sha256,
            "device": cache.device,
            "inode": cache.inode,
            "hardlink_count": cache.hardlink_count,
            "semantic_cache_fingerprint": (
                cache.semantic_cache_fingerprint
            ),
            "neutral_payload_fingerprint": (
                cache.neutral_payload_fingerprint
            ),
        },
        "selection_effect": (
            "predeclared_primary" if seed == 42 else "none"
        ),
        "may_replace_seed42_primary": False,
        "D_V_execution_authorized": False,
        "D_T_execution_authorized": False,
    }


def _mkdir_chain_parent() -> Path:
    runtime = _required_runtime_root()
    formal = runtime / "formal"
    if runtime.exists() and (
        not runtime.is_dir()
        or runtime.is_symlink()
        or runtime.resolve(strict=True) != runtime
    ):
        raise RuntimeError("Formal runtime root is not canonical")
    runtime.mkdir(parents=True, exist_ok=True)
    formal.mkdir(parents=False, exist_ok=True)
    seed_directories = tuple(_run_directory(seed) for seed in _SEED_ROLES)
    for directory in seed_directories:
        directory.mkdir(parents=False, exist_ok=True)
    for directory in (runtime, formal, *seed_directories):
        if (
            not directory.is_dir()
            or directory.is_symlink()
            or directory.resolve(strict=True) != directory
        ):
            raise RuntimeError("Formal runtime directory is not canonical")
    if any(any(directory.iterdir()) for directory in seed_directories):
        raise FileExistsError(
            "Formal seed directories must be empty before chain sealing"
        )
    return formal


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
        # Once O_EXCL succeeds the attempt evidence is intentionally retained.
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
        raise RuntimeError("persistent JSON publication failed verification")


@dataclass(frozen=True, slots=True)
class VerifiedGCRPACREFormalChainConfig:
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
            raise AssertionError("verified chain config changed")
        return value

    def run(self, seed: int) -> dict[str, object]:
        runs = self.payload["runs"]
        if not isinstance(runs, dict):
            raise AssertionError("verified run mapping changed")
        value = runs[_RUN_KEYS[seed]]
        if not isinstance(value, dict):
            raise AssertionError("verified run binding changed")
        return value


@dataclass(frozen=True, slots=True)
class GCRPACREFormalRunStartToken:
    marker_path: str
    marker_size_bytes: int
    marker_file_sha256: str
    marker_device: int
    marker_inode: int
    marker_fingerprint: str
    chain_config_fingerprint: str
    authorization_fingerprint: str
    seed: int
    role: str
    process_instance_fingerprint: str
    requested_device: str
    output_directory: str
    _issuer: object = field(repr=False, compare=False)


def _register(
    registry: dict[int, tuple[object, str]],
    token: object,
    payload: Mapping[str, object],
) -> object:
    identity = id(token)
    if identity in registry:
        raise AssertionError("Formal capability identity was reused")
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


def _chain_token_payload(
    token: VerifiedGCRPACREFormalChainConfig,
) -> dict[str, object]:
    return {
        "payload_json": token.payload_json,
        "path": token.path,
        "file_sha256": token.file_sha256,
        "config_fingerprint": token.config_fingerprint,
        "source_closure_fingerprint": token.source_closure_fingerprint,
    }


def _run_start_token_payload(
    token: GCRPACREFormalRunStartToken,
) -> dict[str, object]:
    return {
        name: getattr(token, name)
        for name in token.__dataclass_fields__
        if name != "_issuer"
    }


def _validate_chain_payload(
    payload: Mapping[str, object],
    *,
    path: Path,
) -> dict[str, object]:
    value = dict(payload)
    body = dict(value)
    fingerprint = body.pop("config_fingerprint", None)
    source_rows = _source_snapshot()
    source_fp = gcr_pacre_v24_source_closure_fingerprint(source_rows)
    runs = value.get("runs")
    predecessors = value.get("predecessors")
    budget = value.get("budget")
    policy = value.get("attempt_policy")
    expected_top = {
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
        "runs",
        "formal_pair_receipt_path",
        "budget",
        "attempt_policy",
        "D_V_payload_accessed",
        "D_T_payload_accessed",
        "config_fingerprint",
    }
    expected_run_fields = {
        "seed",
        "role",
        "stage_id",
        "requested_device",
        "output_directory",
        "run_start_marker_path",
        "authorization_artifact_path",
        "schedule_artifact_path",
        "terminal_artifact_directory",
        "evidence_artifact_path",
        "access_audit_receipt_fingerprint",
        "access_audit_receipt",
        "cache_artifact",
        "selection_effect",
        "may_replace_seed42_primary",
        "D_V_execution_authorized",
        "D_T_execution_authorized",
    }
    expected_cache_fields = {
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
    if (
        set(value) != expected_top
        or value.get("schema_version")
        != GCR_PACRE_FORMAL_CHAIN_CONFIG_SCHEMA
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("path_policy")
        != GCR_PACRE_FORMAL_RUN_START_PATH_POLICY
        or value.get("repository_root") != str(_repository_root())
        or value.get("runtime_root") != str(_required_runtime_root())
        or value.get("chain_config_path") != str(path)
        or path != required_gcr_pacre_formal_chain_config_path()
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
            "paired_bounded400_decision_fingerprint",
        }
        or any(not _is_sha256(item) for item in predecessors.values())
        or not isinstance(runs, Mapping)
        or set(runs) != set(_RUN_KEYS.values())
        or budget
        != {
            "epochs_per_seed": GCR_PACRE_FORMAL_EPOCHS,
            "steps_per_epoch": GCR_PACRE_FORMAL_STEPS_PER_EPOCH,
            "updates_per_seed": GCR_PACRE_FORMAL_UPDATES,
            "training_invocations_per_seed": 1,
        }
        or policy
        != {
            "from_scratch": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "persistent_O_EXCL_run_start_required": True,
            "run_start_marker_never_removed_after_creation": True,
            "checkpoint_policy": "final_only",
            "seed43_selection_effect": "none",
        }
        or value.get("formal_pair_receipt_path")
        != str(_required_runtime_root() / "formal" / "formal800_pair_receipt.json")
        or value.get("D_V_payload_accessed") is not False
        or value.get("D_T_payload_accessed") is not False
        or not _is_sha256(fingerprint)
        or fingerprint != stable_fingerprint(body)
    ):
        raise PermissionError("Formal chain config identity changed")
    for seed, key in _RUN_KEYS.items():
        run = runs.get(key)
        if not isinstance(run, Mapping) or set(run) != expected_run_fields:
            raise ValueError("Formal chain run schema changed")
        cache = run.get("cache_artifact")
        access_receipt = run.get("access_audit_receipt")
        output = _run_directory(seed)
        if (
            run.get("seed") != seed
            or run.get("role") != _SEED_ROLES[seed]
            or run.get("stage_id")
            != f"formal800_seed{seed}_{_SEED_ROLES[seed]}"
            or run.get("requested_device")
            != _canonical_device(str(run.get("requested_device")))
            or run.get("output_directory") != str(output)
            or run.get("run_start_marker_path")
            != str(output / "run_start.json")
            or run.get("authorization_artifact_path")
            != str(output / "authorization.json")
            or run.get("schedule_artifact_path")
            != str(output / "schedule.json")
            or run.get("terminal_artifact_directory")
            != str(output / "terminal")
            or run.get("evidence_artifact_path")
            != str(output / "formal800_evidence.json")
            or not _is_sha256(
                run.get("access_audit_receipt_fingerprint")
            )
            or not isinstance(access_receipt, Mapping)
            or access_receipt.get("receipt_fingerprint")
            != run.get("access_audit_receipt_fingerprint")
            or access_receipt.get("stage_id")
            != f"formal800_seed{seed}_{_SEED_ROLES[seed]}"
            or access_receipt.get("allowed_splits") != ["D_R"]
            or access_receipt.get("D_V_payload_accessed") is not False
            or access_receipt.get("D_T_payload_accessed") is not False
            or stable_fingerprint(
                {
                    name: item
                    for name, item in access_receipt.items()
                    if name != "receipt_fingerprint"
                }
            )
            != access_receipt.get("receipt_fingerprint")
            or not isinstance(cache, Mapping)
            or set(cache) != expected_cache_fields
            or not all(
                _is_sha256(cache.get(name))
                for name in (
                    "receipt_fingerprint",
                    "file_sha256",
                    "semantic_cache_fingerprint",
                    "neutral_payload_fingerprint",
                )
            )
            or not isinstance(cache.get("cache_id"), str)
            or not cache.get("cache_id")
            or not isinstance(cache.get("path"), str)
            or not Path(str(cache.get("path"))).is_absolute()
            or not all(
                isinstance(cache.get(name), int)
                and not isinstance(cache.get(name), bool)
                and int(cache[name]) >= 1
                for name in ("device", "inode", "hardlink_count")
            )
            or run.get("selection_effect")
            != ("predeclared_primary" if seed == 42 else "none")
            or run.get("may_replace_seed42_primary") is not False
            or run.get("D_V_execution_authorized") is not False
            or run.get("D_T_execution_authorized") is not False
        ):
            raise PermissionError("Formal chain run binding changed")
        if (
            not output.is_dir()
            or output.is_symlink()
            or output.resolve(strict=True) != output
        ):
            raise RuntimeError("Formal fixed seed directory is unavailable")
    return value


def seal_gcr_pacre_formal_chain_config_new(
    *,
    oof_decision: VerifiedOOFDecision,
    bounded_decision: VerifiedBoundedDecision,
    seed42_access_audit: VerifiedAccessAudit,
    seed43_access_audit: VerifiedAccessAudit,
    seed42_cache_artifact: VerifiedFormalCacheArtifact,
    seed43_cache_artifact: VerifiedFormalCacheArtifact,
    dataset_free_receipt_fingerprint: str,
    d_r_structural_receipt_fingerprint: str,
    seed42_device: str,
    seed43_device: str,
) -> VerifiedGCRPACREFormalChainConfig:
    """Seal the sole Formal chain configuration before either run starts."""

    oof = require_verified_oof_decision(oof_decision)
    bounded = require_verified_bounded_decision(bounded_decision)
    if (
        oof.payload.get("gate_passed") is not True
        or bounded.payload.get("gate_passed") is not True
        or bounded.oof_decision_fingerprint != oof.decision_fingerprint
        or not _is_sha256(dataset_free_receipt_fingerprint)
        or not _is_sha256(d_r_structural_receipt_fingerprint)
    ):
        raise PermissionError("Formal chain predecessors are incoherent")
    verify_formal_cache_pair_independence(
        seed42_cache_artifact,
        seed43_cache_artifact,
    )
    parent = _mkdir_chain_parent()
    target = required_gcr_pacre_formal_chain_config_path()
    if target.parent != parent:
        raise AssertionError("Formal chain config parent changed")
    source_rows = _source_snapshot()
    body = {
        "schema_version": GCR_PACRE_FORMAL_CHAIN_CONFIG_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "path_policy": GCR_PACRE_FORMAL_RUN_START_PATH_POLICY,
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
            "paired_bounded400_decision_fingerprint": (
                bounded.decision_fingerprint
            ),
        },
        "runs": {
            _RUN_KEYS[42]: _run_binding(
                seed=42,
                device=seed42_device,
                access_audit=seed42_access_audit,
                cache_artifact=seed42_cache_artifact,
            ),
            _RUN_KEYS[43]: _run_binding(
                seed=43,
                device=seed43_device,
                access_audit=seed43_access_audit,
                cache_artifact=seed43_cache_artifact,
            ),
        },
        "formal_pair_receipt_path": str(
            _required_runtime_root()
            / "formal"
            / "formal800_pair_receipt.json"
        ),
        "budget": {
            "epochs_per_seed": GCR_PACRE_FORMAL_EPOCHS,
            "steps_per_epoch": GCR_PACRE_FORMAL_STEPS_PER_EPOCH,
            "updates_per_seed": GCR_PACRE_FORMAL_UPDATES,
            "training_invocations_per_seed": 1,
        },
        "attempt_policy": {
            "from_scratch": True,
            "resume_allowed": False,
            "automatic_retry_allowed": False,
            "persistent_O_EXCL_run_start_required": True,
            "run_start_marker_never_removed_after_creation": True,
            "checkpoint_policy": "final_only",
            "seed43_selection_effect": "none",
        },
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    payload = {**body, "config_fingerprint": stable_fingerprint(body)}
    _write_persistent_json(target, payload)
    return load_and_verify_gcr_pacre_formal_chain_config(target)


def load_and_verify_gcr_pacre_formal_chain_config(
    path: str | Path,
) -> VerifiedGCRPACREFormalChainConfig:
    """Reissue a private capability after replaying all config checks."""

    source = _regular_config_path(path)
    payload = _validate_chain_payload(
        read_canonical_json(source),
        path=source,
    )
    token = VerifiedGCRPACREFormalChainConfig(
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


def require_verified_gcr_pacre_formal_chain_config(
    token: object,
) -> VerifiedGCRPACREFormalChainConfig:
    if type(token) is not VerifiedGCRPACREFormalChainConfig or not _live(
        _CHAIN_REGISTRY,
        token,
        _CHAIN_ISSUER,
        _chain_token_payload(token),
    ):
        raise TypeError(
            "chain_config must be issued by the fixed Formal chain verifier"
        )
    current = load_and_verify_gcr_pacre_formal_chain_config(token.path)
    if (
        current.payload_json != token.payload_json
        or current.file_sha256 != token.file_sha256
        or current.config_fingerprint != token.config_fingerprint
    ):
        raise PermissionError("Formal chain config bytes changed")
    return token


def verify_gcr_pacre_formal_chain_authorization_binding(
    chain_config: VerifiedGCRPACREFormalChainConfig,
    *,
    seed: int,
    role: str,
    oof_decision: VerifiedOOFDecision,
    bounded_decision: VerifiedBoundedDecision,
    access_audit: VerifiedAccessAudit,
    cache_artifact: VerifiedFormalCacheArtifact,
    dataset_free_receipt_fingerprint: str,
    d_r_structural_receipt_fingerprint: str,
) -> dict[str, object]:
    """Bind one authorization to the exact frozen seed entry."""

    chain = require_verified_gcr_pacre_formal_chain_config(chain_config)
    oof = require_verified_oof_decision(oof_decision)
    bounded = require_verified_bounded_decision(bounded_decision)
    access = require_verified_access_audit(access_audit)
    cache = require_verified_formal_cache_artifact(cache_artifact)
    run = chain.run(seed)
    predecessors = chain.payload["predecessors"]
    cache_binding = run.get("cache_artifact")
    if (
        role != _SEED_ROLES.get(seed)
        or not isinstance(predecessors, Mapping)
        or predecessors.get("dataset_free_receipt_fingerprint")
        != dataset_free_receipt_fingerprint
        or predecessors.get("D_R_structural_receipt_fingerprint")
        != d_r_structural_receipt_fingerprint
        or predecessors.get("OOF4_decision_fingerprint")
        != oof.decision_fingerprint
        or predecessors.get("paired_bounded400_decision_fingerprint")
        != bounded.decision_fingerprint
        or run.get("access_audit_receipt_fingerprint")
        != access.receipt_fingerprint
        or not isinstance(cache_binding, Mapping)
        or cache_binding
        != {
            "receipt_fingerprint": cache.receipt_fingerprint,
            "cache_id": cache.cache_id,
            "path": cache.path,
            "file_sha256": cache.file_sha256,
            "device": cache.device,
            "inode": cache.inode,
            "hardlink_count": cache.hardlink_count,
            "semantic_cache_fingerprint": (
                cache.semantic_cache_fingerprint
            ),
            "neutral_payload_fingerprint": (
                cache.neutral_payload_fingerprint
            ),
        }
    ):
        raise PermissionError("Formal authorization differs from chain config")
    return run


def create_gcr_pacre_formal_run_start_marker(
    authorization: object,
) -> GCRPACREFormalRunStartToken:
    """Create and permanently retain the seed-specific attempt marker."""

    # Avoid an import cycle at module import time.
    from .formal_training import GCRPACREFormalAuthorization

    if type(authorization) is not GCRPACREFormalAuthorization:
        raise TypeError("authorization must be exact Formal authorization")
    authorization.verify_unchanged()
    run = authorization.chain_run_binding
    output = Path(str(run["output_directory"]))
    marker_path = Path(str(run["run_start_marker_path"]))
    if (
        output != _run_directory(authorization.seed)
        or marker_path
        != required_gcr_pacre_formal_run_start_path(authorization.seed)
        or marker_path.parent != output
        or not output.is_dir()
        or output.is_symlink()
        or output.resolve(strict=True) != output
    ):
        raise PermissionError("Formal run directory differs from chain config")
    if any(output.iterdir()):
        raise FileExistsError(
            "Formal run directory is not pristine before persistent claim"
        )
    source_rows = _source_snapshot()
    source_fp = gcr_pacre_v24_source_closure_fingerprint(source_rows)
    if (
        source_rows != authorization.source_hashes
        or source_fp
        != authorization.chain_config.source_closure_fingerprint
    ):
        raise RuntimeError("Formal source closure changed before run start")
    intent = {
        "execution_kind": "Formal800_D_R_training",
        "split": "D_R",
        "requested_device": authorization.requested_device,
        "output_directory": authorization.output_directory,
        "epochs": GCR_PACRE_FORMAL_EPOCHS,
        "steps_per_epoch": GCR_PACRE_FORMAL_STEPS_PER_EPOCH,
        "optimizer_steps_authorized": GCR_PACRE_FORMAL_UPDATES,
        "parameter_updates_authorized": GCR_PACRE_FORMAL_UPDATES,
        "training_invocations_authorized": 1,
        "from_scratch": True,
        "resume_allowed": False,
        "automatic_retry_allowed": False,
        "D_V_materialization_intended": False,
        "D_T_materialization_intended": False,
    }
    body = {
        "schema_version": GCR_PACRE_FORMAL_RUN_START_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "path_policy": GCR_PACRE_FORMAL_RUN_START_PATH_POLICY,
        "marker_path": str(marker_path),
        "seed": authorization.seed,
        "role": authorization.role,
        "stage_id": authorization.stage_id,
        "process_instance_fingerprint": _PROCESS_INSTANCE_FINGERPRINT,
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
        "access_audit_receipt_fingerprint": (
            authorization.access_audit.receipt_fingerprint
        ),
        "cache_artifact": {
            "path": authorization.cache_artifact.path,
            "file_sha256": authorization.cache_artifact.file_sha256,
            "receipt_fingerprint": (
                authorization.cache_artifact.receipt_fingerprint
            ),
            "semantic_cache_fingerprint": (
                authorization.cache_artifact.semantic_cache_fingerprint
            ),
            "neutral_payload_fingerprint": (
                authorization.cache_artifact.neutral_payload_fingerprint
            ),
        },
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
    _write_persistent_json(marker_path, payload)
    stat_result = marker_path.stat()
    token = GCRPACREFormalRunStartToken(
        marker_path=str(marker_path),
        marker_size_bytes=stat_result.st_size,
        marker_file_sha256=file_sha256(marker_path),
        marker_device=stat_result.st_dev,
        marker_inode=stat_result.st_ino,
        marker_fingerprint=str(payload["marker_fingerprint"]),
        chain_config_fingerprint=(
            authorization.chain_config.config_fingerprint
        ),
        authorization_fingerprint=authorization.authorization_fingerprint,
        seed=authorization.seed,
        role=authorization.role,
        process_instance_fingerprint=_PROCESS_INSTANCE_FINGERPRINT,
        requested_device=authorization.requested_device,
        output_directory=authorization.output_directory,
        _issuer=_RUN_START_ISSUER,
    )
    return _register(
        _RUN_START_REGISTRY,
        token,
        _run_start_token_payload(token),
    )  # type: ignore[return-value]


def verify_gcr_pacre_formal_run_start_token(
    authorization: object,
    token: object,
) -> dict[str, object]:
    """Re-read the persistent marker and return its evidence wrapper."""

    from .formal_training import GCRPACREFormalAuthorization

    if (
        type(authorization) is not GCRPACREFormalAuthorization
        or type(token) is not GCRPACREFormalRunStartToken
        or not _live(
            _RUN_START_REGISTRY,
            token,
            _RUN_START_ISSUER,
            _run_start_token_payload(token),
        )
    ):
        raise TypeError("run_start_token is not a live Formal capability")
    authorization.verify_unchanged()
    path = Path(token.marker_path)
    expected_path = required_gcr_pacre_formal_run_start_path(
        authorization.seed
    )
    payload = read_canonical_json(path)
    body = dict(payload)
    marker_fp = body.pop("marker_fingerprint", None)
    stat_result = path.stat()
    if (
        path != expected_path
        or path.is_symlink()
        or path.resolve(strict=True) != path
        or stat_result.st_nlink != 1
        or stat_result.st_mode & 0o222
        or token.marker_size_bytes != stat_result.st_size
        or token.marker_file_sha256 != file_sha256(path)
        or token.marker_device != stat_result.st_dev
        or token.marker_inode != stat_result.st_ino
        or marker_fp != token.marker_fingerprint
        or marker_fp != stable_fingerprint(body)
        or token.chain_config_fingerprint
        != authorization.chain_config.config_fingerprint
        or token.authorization_fingerprint
        != authorization.authorization_fingerprint
        or (token.seed, token.role)
        != (authorization.seed, authorization.role)
        or token.process_instance_fingerprint
        != _PROCESS_INSTANCE_FINGERPRINT
        or payload.get("process_instance_fingerprint")
        != token.process_instance_fingerprint
        or token.requested_device != authorization.requested_device
        or token.output_directory != authorization.output_directory
        or payload.get("marker_path") != str(expected_path)
        or payload.get("authorization_fingerprint")
        != authorization.authorization_fingerprint
        or payload.get("access_audit_receipt_fingerprint")
        != authorization.access_audit.receipt_fingerprint
        or payload.get("D_V_payload_accessed") is not False
        or payload.get("D_T_payload_accessed") is not False
    ):
        raise PermissionError("persistent Formal run-start marker changed")
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
    "GCR_PACRE_FORMAL_CHAIN_CONFIG_SCHEMA",
    "GCR_PACRE_FORMAL_RUN_START_PATH_POLICY",
    "GCR_PACRE_FORMAL_RUN_START_SCHEMA",
    "GCRPACREFormalRunStartToken",
    "VerifiedGCRPACREFormalChainConfig",
    "create_gcr_pacre_formal_run_start_marker",
    "load_and_verify_gcr_pacre_formal_chain_config",
    "require_verified_gcr_pacre_formal_chain_config",
    "required_gcr_pacre_formal_chain_config_path",
    "required_gcr_pacre_formal_run_start_path",
    "seal_gcr_pacre_formal_chain_config_new",
    "verify_gcr_pacre_formal_chain_authorization_binding",
    "verify_gcr_pacre_formal_run_start_token",
]
