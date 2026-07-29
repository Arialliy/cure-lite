"""Create-only, physically isolated cache artifacts for D_R OOF-4.

Every cache is a distinct regular file created with ``O_EXCL``.  Verification
is performed against the actual file, Linux FIEMAP, a fixed non-mmap loader,
and simultaneously live tensor storages.  Holdout artifacts cannot be
created or opened until a verifier-issued terminal seal exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Final, Iterator, Mapping, Sequence

import torch
from torch import Tensor

from cure_lite.cache.schema import canonical_json, stable_fingerprint
from cure_lite.coverage_state_precomputed_cache import (
    CoverageStateScalarCache,
)
from cure_lite.paired_types import tensor_content_fingerprint

from .formal_cache_artifacts import (
    _fiemap_flags,
    _regular_file,
    build_formal_cache_neutral_envelope,
    rebuild_formal_scalar_cache_from_neutral_envelope,
)
from .oof_evaluation import (
    OOF_EVALUATION_DATASET_SCHEMA,
    OOFEvaluationDataset,
    seal_oof_evaluation_dataset,
    seal_oof_evaluation_sample,
)
from .oof_split import (
    VerifiedOOFFoldClosure,
    require_verified_oof_fold_closure,
)


OOF_CACHE_ENVELOPE_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-oof4-physical-cache-envelope-v2"
)
OOF_EVALUATION_DATASET_CODEC_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-oof4-evaluation-dataset-neutral-codec-v1"
)
OOF_CACHE_ARTIFACT_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-oof4-cache-artifact-v1"
)
OOF_CACHE_SET_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-oof4-six-cache-independence-v1"
)
OOF_TERMINAL_SEAL_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-oof4-training-terminal-seal-v1"
)
OOF_EVENT_TRAIN_CACHE_CREATED: Final = 1
OOF_EVENT_TRAINING_RUN_START: Final = 2
OOF_EVENT_TERMINALS_SEALED: Final = 3
OOF_EVENT_HOLDOUT_CACHE_CREATED: Final = 4
OOF_EVENT_HOLDOUT_FIRST_OPEN: Final = 5
OOF_ARMS: Final = (
    "base_eval",
    "PACRE_VC_v23_control",
    "GCR_PACRE_v24",
)
_DIRECTORY_BY_ARM = {
    "base_eval": "base_eval",
    "PACRE_VC_v23_control": "v23_control",
    "GCR_PACRE_v24": "candidate",
}
_READERS = {
    ("train", "base_eval"): ("BaseB_train_fold_selector",),
    ("train", "PACRE_VC_v23_control"): (
        "PACRE_VC_v23_control_train_runner",
    ),
    ("train", "GCR_PACRE_v24"): ("GCR_PACRE_v24_train_runner",),
    ("holdout", "base_eval"): ("OOF4_read_only_holdout_evaluator",),
    ("holdout", "PACRE_VC_v23_control"): (
        "OOF4_read_only_holdout_evaluator",
    ),
    ("holdout", "GCR_PACRE_v24"): (
        "OOF4_read_only_holdout_evaluator",
    ),
}
_PAYLOAD_KIND_SCALAR_CACHE: Final = "coverage_state_scalar_cache"
_PAYLOAD_KIND_EVALUATION_DATASET: Final = "oof_evaluation_dataset"
_PAYLOAD_KIND_BY_SLOT: Final = {
    ("train", "base_eval"): _PAYLOAD_KIND_EVALUATION_DATASET,
    ("train", "PACRE_VC_v23_control"): _PAYLOAD_KIND_SCALAR_CACHE,
    ("train", "GCR_PACRE_v24"): _PAYLOAD_KIND_SCALAR_CACHE,
    ("holdout", "base_eval"): _PAYLOAD_KIND_EVALUATION_DATASET,
    ("holdout", "PACRE_VC_v23_control"): (
        _PAYLOAD_KIND_EVALUATION_DATASET
    ),
    ("holdout", "GCR_PACRE_v24"): _PAYLOAD_KIND_EVALUATION_DATASET,
}
_FIXED_LOADER_POLICY: Final = {
    "torch_load": True,
    "weights_only": True,
    "mmap_used": False,
    "neutral_object_graph": True,
}
_TOKEN_ISSUER = object()
_TOKEN_REGISTRY: dict[int, object] = {}


def _register(value: object) -> object:
    if getattr(value, "_issuer", None) is not _TOKEN_ISSUER:
        raise AssertionError("attempted to register an unsigned OOF token")
    identity = id(value)
    prior = _TOKEN_REGISTRY.get(identity)
    if prior is not None and prior is not value:
        raise RuntimeError("OOF cache token identity collision")
    _TOKEN_REGISTRY[identity] = value
    return value


def _require(value: object, kind: type, *, name: str) -> object:
    if (
        type(value) is not kind
        or getattr(value, "_issuer", None) is not _TOKEN_ISSUER
        or _TOKEN_REGISTRY.get(id(value)) is not value
    ):
        raise TypeError(f"{name} must be issued by the OOF cache verifier")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk_tensors(
    value: object,
    *,
    path: str,
    seen: set[int],
) -> Iterator[tuple[str, Tensor]]:
    if isinstance(value, Tensor):
        if (
            value.device.type != "cpu"
            or value.layout != torch.strided
            or value.requires_grad
            or not value.is_contiguous()
            or value.numel() < 1
        ):
            raise ValueError(
                "OOF cache tensors must be detached contiguous CPU strided"
            )
        yield path, value
        return
    if value is None or isinstance(value, (str, int, float, bool, Path)):
        return
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if is_dataclass(value) and not isinstance(value, type):
        for definition in fields(value):
            yield from _walk_tensors(
                getattr(value, definition.name),
                path=f"{path}.{definition.name}",
                seen=seen,
            )
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("OOF cache mappings require text keys")
        for key in sorted(value):
            yield from _walk_tensors(
                value[key],
                path=f"{path}.{key}",
                seen=seen,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_tensors(
                item,
                path=f"{path}[{index}]",
                seen=seen,
            )


def _tensor_ledger(value: object) -> tuple[dict[str, object], ...]:
    rows = tuple(_walk_tensors(value, path="payload", seen=set()))
    if not rows:
        raise ValueError("OOF cache payload contains no tensors")
    return tuple(
        {
            "logical_path": path,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "content_fingerprint": tensor_content_fingerprint(tensor),
        }
        for path, tensor in rows
    )


def _storage_ids(value: object) -> set[tuple[str, int, int]]:
    return {
        (
            str(tensor.device),
            int(tensor.untyped_storage().data_ptr()),
            int(tensor.untyped_storage().nbytes()),
        )
        for _, tensor in _walk_tensors(value, path="payload", seen=set())
    }


def _dataset_neutral_envelope(
    dataset: OOFEvaluationDataset,
) -> dict[str, object]:
    """Encode evaluator input using primitives and detached CPU tensors."""

    if type(dataset) is not OOFEvaluationDataset:
        raise TypeError("OOF dataset codec requires exact OOFEvaluationDataset")
    dataset.verify_unchanged()
    rows = []
    for row in dataset.rows:
        row.verify_unchanged()
        rows.append({
            "sample_id": row.sample_id,
            "root_source_id": row.root_source_id,
            "anchor_miss_ids": list(row.anchor_miss_ids),
            "reachable_anchor_miss_ids": list(
                row.reachable_anchor_miss_ids
            ),
            "content_fingerprint": row.content_fingerprint,
            "tensors": {
                "base_probability": (
                    row.base_probability.detach().cpu().clone().contiguous()
                ),
                "feature": row.feature.detach().cpu().clone().contiguous(),
                "gt_mask": row.gt_mask.detach().cpu().clone().contiguous(),
                "valid_mask": (
                    row.valid_mask.detach().cpu().clone().contiguous()
                ),
            },
        })
    return {
        "schema_version": OOF_EVALUATION_DATASET_CODEC_SCHEMA,
        "dataset_schema_version": OOF_EVALUATION_DATASET_SCHEMA,
        "fold_id": dataset.fold_id,
        "partition": dataset.partition,
        "closure_fingerprint": dataset.closure_fingerprint,
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "canonical_dataset_payload_json": canonical_json(
            dataset.canonical_payload()
        ),
        "rows": rows,
    }


def _decode_dataset_neutral_envelope(
    value: object,
    *,
    expected_semantic_fingerprint: str,
) -> OOFEvaluationDataset:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "dataset_schema_version",
        "fold_id",
        "partition",
        "closure_fingerprint",
        "dataset_fingerprint",
        "canonical_dataset_payload_json",
        "rows",
    }:
        raise ValueError("OOF evaluation dataset neutral schema changed")
    if (
        value.get("schema_version")
        != OOF_EVALUATION_DATASET_CODEC_SCHEMA
        or value.get("dataset_schema_version")
        != OOF_EVALUATION_DATASET_SCHEMA
        or value.get("dataset_fingerprint")
        != expected_semantic_fingerprint
    ):
        raise PermissionError("OOF evaluation dataset identity changed")
    fold_id = value.get("fold_id")
    partition = value.get("partition")
    closure_fingerprint = value.get("closure_fingerprint")
    canonical_payload_json = value.get("canonical_dataset_payload_json")
    raw_rows = value.get("rows")
    if (
        isinstance(fold_id, bool)
        or not isinstance(fold_id, int)
        or fold_id not in range(4)
        or partition not in {"train", "holdout"}
        or not isinstance(closure_fingerprint, str)
        or len(closure_fingerprint) != 64
        or not isinstance(canonical_payload_json, str)
        or not isinstance(raw_rows, list)
        or not raw_rows
    ):
        raise ValueError("OOF evaluation dataset neutral identity is invalid")
    rows = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping) or set(raw) != {
            "sample_id",
            "root_source_id",
            "anchor_miss_ids",
            "reachable_anchor_miss_ids",
            "content_fingerprint",
            "tensors",
        }:
            raise ValueError(f"OOF evaluation row {index} schema changed")
        tensors = raw.get("tensors")
        if not isinstance(tensors, Mapping) or set(tensors) != {
            "base_probability",
            "feature",
            "gt_mask",
            "valid_mask",
        }:
            raise ValueError(f"OOF evaluation row {index} tensors changed")
        if any(type(tensors[name]) is not Tensor for name in tensors):
            raise TypeError(f"OOF evaluation row {index} tensor type changed")
        base_tensor = tensors["base_probability"]
        feature_tensor = tensors["feature"]
        gt_tensor = tensors["gt_mask"]
        valid_tensor = tensors["valid_mask"]
        if (
            base_tensor.dtype != torch.float32
            or base_tensor.ndim != 4
            or tuple(base_tensor.shape[:2]) != (1, 1)
            or feature_tensor.dtype != torch.float32
            or feature_tensor.ndim != 4
            or tuple(feature_tensor.shape[:2]) != (1, 64)
            or tuple(base_tensor.shape[-2:])
            != tuple(4 * value for value in feature_tensor.shape[-2:])
            or gt_tensor.dtype != torch.bool
            or valid_tensor.dtype != torch.bool
            or gt_tensor.shape != base_tensor.shape
            or valid_tensor.shape != base_tensor.shape
        ):
            raise ValueError(
                f"OOF evaluation row {index} fixed tensor contract changed"
            )
        anchor_miss_ids = raw.get("anchor_miss_ids")
        reachable_anchor_miss_ids = raw.get(
            "reachable_anchor_miss_ids"
        )
        if (
            not isinstance(raw.get("sample_id"), str)
            or not raw["sample_id"]
            or not isinstance(raw.get("root_source_id"), str)
            or not raw["root_source_id"]
            or not isinstance(anchor_miss_ids, list)
            or not isinstance(reachable_anchor_miss_ids, list)
            or any(
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 1
                for item in [
                    *anchor_miss_ids,
                    *reachable_anchor_miss_ids,
                ]
            )
        ):
            raise ValueError(f"OOF evaluation row {index} identity changed")
        row = seal_oof_evaluation_sample(
            sample_id=str(raw["sample_id"]),
            root_source_id=str(raw["root_source_id"]),
            base_probability=base_tensor,
            feature=feature_tensor,
            gt_mask=gt_tensor,
            valid_mask=valid_tensor,
            anchor_miss_ids=anchor_miss_ids,
            reachable_anchor_miss_ids=reachable_anchor_miss_ids,
        )
        if row.content_fingerprint != raw.get("content_fingerprint"):
            raise PermissionError(
                f"OOF evaluation row {index} fingerprint changed"
            )
        rows.append(row)
    dataset = seal_oof_evaluation_dataset(
        fold_id=fold_id,
        partition=str(partition),
        closure_fingerprint=closure_fingerprint,
        rows=rows,
    )
    if (
        dataset.dataset_fingerprint != expected_semantic_fingerprint
        or canonical_json(dataset.canonical_payload())
        != canonical_payload_json
    ):
        raise PermissionError("rebuilt OOF evaluation dataset changed")
    return dataset


def _encode_payload(
    payload: object,
    *,
    partition: str,
    arm: str,
) -> tuple[str, dict[str, object]]:
    expected_kind = _PAYLOAD_KIND_BY_SLOT[(partition, arm)]
    if expected_kind == _PAYLOAD_KIND_SCALAR_CACHE:
        if type(payload) is not CoverageStateScalarCache:
            raise TypeError(
                f"OOF {partition}/{arm} requires CoverageStateScalarCache"
            )
        return expected_kind, build_formal_cache_neutral_envelope(payload)
    if type(payload) is not OOFEvaluationDataset:
        raise TypeError(
            f"OOF {partition}/{arm} requires OOFEvaluationDataset"
        )
    return expected_kind, _dataset_neutral_envelope(payload)


def _decode_payload(
    encoded: object,
    *,
    payload_kind: object,
    partition: str,
    arm: str,
    expected_semantic_fingerprint: str,
) -> object:
    expected_kind = _PAYLOAD_KIND_BY_SLOT[(partition, arm)]
    if payload_kind != expected_kind or not isinstance(encoded, Mapping):
        raise PermissionError("OOF cache slot payload type changed")
    if expected_kind == _PAYLOAD_KIND_SCALAR_CACHE:
        return rebuild_formal_scalar_cache_from_neutral_envelope(
            encoded,
            expected_semantic_cache_fingerprint=(
                expected_semantic_fingerprint
            ),
        )
    return _decode_dataset_neutral_envelope(
        encoded,
        expected_semantic_fingerprint=expected_semantic_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class VerifiedOOFTerminalSeal:
    fold_id: int
    closure_fingerprint: str
    terminal_artifact_fingerprints: tuple[tuple[str, str], ...]
    completed_400_capability_fingerprints: tuple[tuple[str, str], ...]
    run_start_marker_fingerprint: str
    shared_initial_parameter_fingerprint: str
    event_index: int
    seal_fingerprint: str
    _issuer: object


@dataclass(frozen=True, slots=True)
class VerifiedOOFCacheArtifact:
    payload_json: str
    fold_id: int
    partition: str
    arm: str
    closure_fingerprint: str
    terminal_seal_fingerprint: str | None
    semantic_payload_fingerprint: str
    cache_id: str
    path: str
    size_bytes: int
    file_sha256: str
    device: int
    inode: int
    hardlink_count: int
    root_source_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    creation_phase: str
    creation_event: int
    reader_allowlist: tuple[str, ...]
    tensor_ledger_fingerprint: str
    artifact_fingerprint: str
    _issuer: object

    @property
    def payload(self) -> dict[str, object]:
        value = json.loads(self.payload_json)
        if not isinstance(value, dict):
            raise AssertionError("OOF cache token payload changed")
        return value


@dataclass(frozen=True, slots=True)
class VerifiedOOFCacheReader:
    fold_id: int
    partition: str
    arm: str
    closure_fingerprint: str
    semantic_payload_fingerprint: str
    cache_artifact_fingerprint: str
    reader_id: str
    authorization_event: int
    holdout_terminal_seal_fingerprint: str | None
    authorization_fingerprint: str
    _cache_token: VerifiedOOFCacheArtifact = field(repr=False)
    _issuer: object = field(repr=False)


@dataclass(frozen=True, slots=True)
class VerifiedOOFSixCacheSet:
    fold_id: int
    cache_artifact_fingerprints: tuple[str, ...]
    entries_json: str
    set_fingerprint: str
    _issuer: object

    @property
    def protocol_entries(self) -> tuple[dict[str, object], ...]:
        value = json.loads(self.entries_json)
        if not isinstance(value, list):
            raise AssertionError("OOF cache-set entries changed")
        return tuple(dict(row) for row in value)


def require_verified_oof_cache_artifact(
    value: object,
) -> VerifiedOOFCacheArtifact:
    result = _require(
        value,
        VerifiedOOFCacheArtifact,
        name="cache_artifact",
    )
    assert isinstance(result, VerifiedOOFCacheArtifact)
    return result


def require_verified_oof_cache_reader(
    value: object,
) -> VerifiedOOFCacheReader:
    result = _require(
        value,
        VerifiedOOFCacheReader,
        name="reader_authorization",
    )
    assert isinstance(result, VerifiedOOFCacheReader)
    cache = require_verified_oof_cache_artifact(result._cache_token)
    body = {
        "fold_id": cache.fold_id,
        "partition": cache.partition,
        "arm": cache.arm,
        "closure_fingerprint": cache.closure_fingerprint,
        "semantic_payload_fingerprint": (
            cache.semantic_payload_fingerprint
        ),
        "cache_artifact_fingerprint": cache.artifact_fingerprint,
        "reader_id": result.reader_id,
        "authorization_event": result.authorization_event,
        "holdout_terminal_seal_fingerprint": (
            result.holdout_terminal_seal_fingerprint
        ),
    }
    if (
        result.fold_id != cache.fold_id
        or result.partition != cache.partition
        or result.arm != cache.arm
        or result.closure_fingerprint != cache.closure_fingerprint
        or result.semantic_payload_fingerprint
        != cache.semantic_payload_fingerprint
        or result.cache_artifact_fingerprint
        != cache.artifact_fingerprint
        or result.reader_id not in cache.reader_allowlist
        or result.authorization_event
        != (
            OOF_EVENT_HOLDOUT_FIRST_OPEN
            if cache.partition == "holdout"
            else OOF_EVENT_TRAIN_CACHE_CREATED
        )
        or result.authorization_fingerprint != stable_fingerprint(body)
    ):
        raise PermissionError("OOF cache reader binding changed")
    _reverify(cache)
    if cache.partition == "train":
        _require_holdout_cache_paths_absent(cache)
    return result


def require_verified_oof_terminal_seal(
    value: object,
) -> VerifiedOOFTerminalSeal:
    result = _require(value, VerifiedOOFTerminalSeal, name="terminal_seal")
    assert isinstance(result, VerifiedOOFTerminalSeal)
    return result


def _semantic_payload_fingerprint(
    payload: object,
    *,
    tensor_ledger_fingerprint: str,
) -> str:
    """Bind the semantic identity exposed by a supported cache payload."""

    for attribute in (
        "cache_fingerprint",
        "dataset_fingerprint",
        "content_fingerprint",
    ):
        value = getattr(payload, attribute, None)
        if (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        ):
            verifier = getattr(payload, "verify_unchanged", None)
            if callable(verifier):
                verifier()
            return value
    return stable_fingerprint(
        {
            "schema_version": (
                "cure-lite-v24-oof-generated-payload-semantics-v1"
            ),
            "payload_fqcn": (
                f"{type(payload).__module__}.{type(payload).__qualname__}"
            ),
            "tensor_ledger_fingerprint": tensor_ledger_fingerprint,
        }
    )


def _verify_payload_partition(
    payload: object,
    *,
    expected_sample_ids: tuple[str, ...],
) -> None:
    """Reject a full-population payload hidden behind a fold cache token."""

    if type(payload) is CoverageStateScalarCache:
        payload.verify_unchanged()
        observed = {
            value.record.sample_id for value in payload.natural_records
        } | {
            value.record.sample_id for value in payload.pair_records
        }
    else:
        raw = getattr(payload, "sample_ids", None)
        if raw is None:
            # Generated-only tensor fixtures have no source identity.  They
            # exercise physical isolation but are never valid training input.
            return
        observed = set(raw)
    if observed != set(expected_sample_ids):
        raise PermissionError(
            "OOF cache payload sample population differs from the exact "
            "fold partition"
        )


def seal_oof_training_terminals(
    fold_closure: VerifiedOOFFoldClosure,
    *,
    completed_400_capabilities: Mapping[str, object],
) -> VerifiedOOFTerminalSeal:
    from .oof_training import (
        OOF_CANDIDATE_ARM,
        OOF_CONTROL_ARM,
        require_verified_oof_completed_400_capability,
    )

    closure = require_verified_oof_fold_closure(fold_closure)
    if (
        set(completed_400_capabilities)
        != {OOF_CONTROL_ARM, OOF_CANDIDATE_ARM}
    ):
        raise ValueError("OOF terminal seal identity is invalid")
    verified = {
        arm: require_verified_oof_completed_400_capability(
            completed_400_capabilities[arm],
            fold_closure=closure,
            arm=arm,
        )
        for arm in (OOF_CONTROL_ARM, OOF_CANDIDATE_ARM)
    }
    control = verified[OOF_CONTROL_ARM]
    candidate = verified[OOF_CANDIDATE_ARM]
    if (
        control.completed_updates != 400
        or candidate.completed_updates != 400
        or control.run_start_marker_fingerprint
        != candidate.run_start_marker_fingerprint
        or control.schedule_fingerprint != candidate.schedule_fingerprint
        or control.batch_sequence_fingerprint
        != candidate.batch_sequence_fingerprint
        or control.semantic_cache_fingerprint
        != candidate.semantic_cache_fingerprint
        or control.optimizer_config_fingerprint
        != candidate.optimizer_config_fingerprint
        or control.objective_policy_fingerprint
        != candidate.objective_policy_fingerprint
        or control.shared_initial_parameter_fingerprint
        != candidate.shared_initial_parameter_fingerprint
        or control.initial_parameters != candidate.initial_parameters
        or stable_fingerprint(list(control.initial_parameters))
        != control.shared_initial_parameter_fingerprint
        or control.source_fingerprint != candidate.source_fingerprint
        or control.module_instance_id == candidate.module_instance_id
        or control.optimizer_instance_id
        == candidate.optimizer_instance_id
        or {
            str(row["storage_identity_fingerprint"])
            for row in control.parameter_storage_ledger
        }
        & {
            str(row["storage_identity_fingerprint"])
            for row in candidate.parameter_storage_ledger
        }
        or (
            control.payload["terminal_artifact"]["device"],
            control.payload["terminal_artifact"]["inode"],
        )
        == (
            candidate.payload["terminal_artifact"]["device"],
            candidate.payload["terminal_artifact"]["inode"],
        )
    ):
        raise PermissionError(
            "OOF terminal capabilities do not prove one exact paired "
            "completed-400 run"
        )
    terminal_rows = tuple(sorted(
        (
            arm,
            capability.terminal_artifact_fingerprint,
        )
        for arm, capability in verified.items()
    ))
    capability_rows = tuple(sorted(
        (
            arm,
            capability.capability_fingerprint,
        )
        for arm, capability in verified.items()
    ))
    body = {
        "schema_version": OOF_TERMINAL_SEAL_SCHEMA,
        "fold_id": closure.fold_id,
        "closure_fingerprint": closure.closure_fingerprint,
        "terminal_artifact_fingerprints": dict(terminal_rows),
        "completed_400_capability_fingerprints": dict(capability_rows),
        "run_start_marker_fingerprint": (
            control.run_start_marker_fingerprint
        ),
        "shared_initial_parameter_fingerprint": (
            control.shared_initial_parameter_fingerprint
        ),
        "initial_parameters": list(control.initial_parameters),
        "schedule_fingerprint": control.schedule_fingerprint,
        "batch_sequence_fingerprint": (
            control.batch_sequence_fingerprint
        ),
        "semantic_cache_fingerprint": (
            control.semantic_cache_fingerprint
        ),
        "optimizer_config_fingerprint": (
            control.optimizer_config_fingerprint
        ),
        "objective_policy_fingerprint": (
            control.objective_policy_fingerprint
        ),
        "event_index": OOF_EVENT_TERMINALS_SEALED,
    }
    fingerprint = stable_fingerprint(body)
    return _register(VerifiedOOFTerminalSeal(
        fold_id=closure.fold_id,
        closure_fingerprint=closure.closure_fingerprint,
        terminal_artifact_fingerprints=terminal_rows,
        completed_400_capability_fingerprints=capability_rows,
        run_start_marker_fingerprint=(
            control.run_start_marker_fingerprint
        ),
        shared_initial_parameter_fingerprint=(
            control.shared_initial_parameter_fingerprint
        ),
        event_index=OOF_EVENT_TERMINALS_SEALED,
        seal_fingerprint=fingerprint,
        _issuer=_TOKEN_ISSUER,
    ))


def _expected_path(
    destination: Path,
    *,
    fold_id: int,
    partition: str,
    arm: str,
) -> Path:
    if not destination.is_absolute() or destination.name != "cache.pt":
        raise ValueError("OOF cache path must be absolute and end in cache.pt")
    expected_suffix = (
        f"fold_{fold_id}",
        partition,
        _DIRECTORY_BY_ARM[arm],
        "cache.pt",
    )
    if tuple(destination.parts[-4:]) != expected_suffix:
        raise ValueError(
            "OOF cache path does not match fold/partition/arm layout"
        )
    parent = destination.parent.resolve(strict=True)
    if parent != destination.parent:
        raise RuntimeError("OOF cache parent is not canonical")
    return destination


def save_oof_cache_artifact_new(
    payload: object,
    destination: str | Path,
    *,
    fold_closure: VerifiedOOFFoldClosure,
    partition: str,
    arm: str,
    creation_event: int,
    terminal_seal: VerifiedOOFTerminalSeal | None = None,
) -> VerifiedOOFCacheArtifact:
    """Create one train or post-terminal holdout cache without replacement."""

    closure = require_verified_oof_fold_closure(fold_closure)
    if (partition, arm) not in _READERS:
        raise ValueError("unknown OOF cache partition/arm")
    if isinstance(creation_event, bool) or not isinstance(creation_event, int):
        raise TypeError("creation_event must be an integer")
    if partition == "train":
        if creation_event != OOF_EVENT_TRAIN_CACHE_CREATED:
            raise PermissionError(
                "train cache creation event must be the frozen event 1"
            )
        if terminal_seal is not None:
            raise PermissionError("train cache must predate terminal sealing")
        samples = closure.train_sample_ids
        roots = closure.train_root_source_ids
        phase = "pre_training_train_only"
        terminal_seal_fingerprint = None
    else:
        if creation_event != OOF_EVENT_HOLDOUT_CACHE_CREATED:
            raise PermissionError(
                "holdout cache creation event must be the frozen event 4"
            )
        seal = require_verified_oof_terminal_seal(terminal_seal)
        if (
            seal.fold_id != closure.fold_id
            or seal.closure_fingerprint != closure.closure_fingerprint
            or creation_event <= seal.event_index
        ):
            raise PermissionError(
                "holdout cache requires an earlier exact terminal seal"
            )
        samples = closure.held_out_sample_ids
        roots = closure.held_out_root_source_ids
        phase = "post_terminal_seal_holdout_only"
        terminal_seal_fingerprint = seal.seal_fingerprint
    _verify_payload_partition(
        payload,
        expected_sample_ids=samples,
    )
    payload_kind, encoded_payload = _encode_payload(
        payload,
        partition=partition,
        arm=arm,
    )
    if payload_kind == _PAYLOAD_KIND_SCALAR_CACHE:
        neutral_payload = encoded_payload.get("payload")
        if (
            not isinstance(neutral_payload, Mapping)
            or not isinstance(neutral_payload.get("tensor_ledger"), list)
        ):
            raise RuntimeError("OOF scalar neutral tensor ledger is absent")
        tensor_ledger = tuple(
            dict(row) for row in neutral_payload["tensor_ledger"]
        )
    else:
        tensor_ledger = _tensor_ledger(payload)
    tensor_ledger_fp = stable_fingerprint(list(tensor_ledger))
    semantic_payload_fp = _semantic_payload_fingerprint(
        payload,
        tensor_ledger_fingerprint=tensor_ledger_fp,
    )
    path = _expected_path(
        Path(destination),
        fold_id=closure.fold_id,
        partition=partition,
        arm=arm,
    )
    envelope = {
        "schema_version": OOF_CACHE_ENVELOPE_SCHEMA,
        "identity": {
            "fold_id": closure.fold_id,
            "partition": partition,
            "arm": arm,
            "closure_fingerprint": closure.closure_fingerprint,
            "terminal_seal_fingerprint": terminal_seal_fingerprint,
            "semantic_payload_fingerprint": semantic_payload_fp,
            "root_source_ids": list(roots),
            "sample_ids": list(samples),
            "creation_phase": phase,
            "creation_event": creation_event,
            "reader_allowlist": list(_READERS[(partition, arm)]),
        },
        "tensor_ledger": list(tensor_ledger),
        "tensor_ledger_fingerprint": tensor_ledger_fp,
        "semantic_payload_fingerprint": semantic_payload_fp,
        "payload_kind": payload_kind,
        "payload": encoded_payload,
    }
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    failed = True
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            torch.save(envelope, handle)
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        failed = False
    finally:
        if failed:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
    canonical_path, stat_result = _regular_file(path)
    fiemap_flags = _fiemap_flags(canonical_path)
    cache_id = f"oof4-fold-{closure.fold_id}-{partition}-{arm}"
    body = {
        "schema_version": OOF_CACHE_ARTIFACT_SCHEMA,
        "cache_id": cache_id,
        "fold_id": closure.fold_id,
        "partition": partition,
        "arm": arm,
        "closure_fingerprint": closure.closure_fingerprint,
        "terminal_seal_fingerprint": terminal_seal_fingerprint,
        "semantic_payload_fingerprint": semantic_payload_fp,
        "root_source_ids": list(roots),
        "sample_ids": list(samples),
        "realpath": str(canonical_path),
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "size_bytes": stat_result.st_size,
        "file_sha256": _file_sha256(canonical_path),
        "creation_phase": phase,
        "creation_event": creation_event,
        "reader_allowlist": list(_READERS[(partition, arm)]),
        "tensor_ledger_fingerprint": tensor_ledger_fp,
        "fiemap_extent_flags": list(fiemap_flags),
        "loader": dict(_FIXED_LOADER_POLICY),
    }
    artifact_fp = stable_fingerprint(body)
    token_payload = {**body, "artifact_fingerprint": artifact_fp}
    return _register(VerifiedOOFCacheArtifact(
        payload_json=canonical_json(token_payload),
        fold_id=closure.fold_id,
        partition=partition,
        arm=arm,
        closure_fingerprint=closure.closure_fingerprint,
        terminal_seal_fingerprint=terminal_seal_fingerprint,
        semantic_payload_fingerprint=semantic_payload_fp,
        cache_id=cache_id,
        path=str(canonical_path),
        size_bytes=stat_result.st_size,
        file_sha256=str(body["file_sha256"]),
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
        hardlink_count=stat_result.st_nlink,
        root_source_ids=roots,
        sample_ids=samples,
        creation_phase=phase,
        creation_event=creation_event,
        reader_allowlist=_READERS[(partition, arm)],
        tensor_ledger_fingerprint=tensor_ledger_fp,
        artifact_fingerprint=artifact_fp,
        _issuer=_TOKEN_ISSUER,
    ))


def _reverify(value: VerifiedOOFCacheArtifact) -> Path:
    path, stat_result = _regular_file(value.path)
    if (
        stat_result.st_size != value.size_bytes
        or stat_result.st_dev != value.device
        or stat_result.st_ino != value.inode
        or stat_result.st_nlink != value.hardlink_count
        or _file_sha256(path) != value.file_sha256
    ):
        raise RuntimeError("OOF cache file changed")
    if list(_fiemap_flags(path)) != value.payload.get(
        "fiemap_extent_flags"
    ):
        raise RuntimeError("OOF cache physical extent ledger changed")
    return path


def _load_envelope(value: VerifiedOOFCacheArtifact) -> tuple[object, object]:
    path = _reverify(value)
    try:
        envelope = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=False,
        )
    except Exception as error:
        raise RuntimeError("OOF cache fixed non-mmap load failed") from error
    if not isinstance(envelope, Mapping) or set(envelope) != {
        "schema_version",
        "identity",
        "tensor_ledger",
        "tensor_ledger_fingerprint",
        "semantic_payload_fingerprint",
        "payload_kind",
        "payload",
    }:
        raise ValueError("OOF cache envelope schema changed")
    identity = envelope.get("identity")
    if not isinstance(identity, Mapping) or dict(identity) != {
        "fold_id": value.fold_id,
        "partition": value.partition,
        "arm": value.arm,
        "closure_fingerprint": value.closure_fingerprint,
        "terminal_seal_fingerprint": (
            value.terminal_seal_fingerprint
        ),
        "semantic_payload_fingerprint": (
            value.semantic_payload_fingerprint
        ),
        "root_source_ids": list(value.root_source_ids),
        "sample_ids": list(value.sample_ids),
        "creation_phase": value.creation_phase,
        "creation_event": value.creation_event,
        "reader_allowlist": list(value.reader_allowlist),
    }:
        raise PermissionError("OOF cache envelope identity changed")
    payload = _decode_payload(
        envelope["payload"],
        payload_kind=envelope.get("payload_kind"),
        partition=value.partition,
        arm=value.arm,
        expected_semantic_fingerprint=(
            value.semantic_payload_fingerprint
        ),
    )
    _verify_payload_partition(
        payload,
        expected_sample_ids=value.sample_ids,
    )
    if envelope.get("payload_kind") == _PAYLOAD_KIND_SCALAR_CACHE:
        encoded_payload = envelope.get("payload")
        neutral_payload = (
            encoded_payload.get("payload")
            if isinstance(encoded_payload, Mapping)
            else None
        )
        raw_ledger = (
            neutral_payload.get("tensor_ledger")
            if isinstance(neutral_payload, Mapping)
            else None
        )
        if not isinstance(raw_ledger, list):
            raise ValueError("OOF scalar neutral tensor ledger is absent")
        ledger = tuple(dict(row) for row in raw_ledger)
    else:
        ledger = _tensor_ledger(payload)
    if (
        envelope.get("schema_version") != OOF_CACHE_ENVELOPE_SCHEMA
        or envelope.get("tensor_ledger") != list(ledger)
        or stable_fingerprint(list(ledger))
        != value.tensor_ledger_fingerprint
        or envelope.get("tensor_ledger_fingerprint")
        != value.tensor_ledger_fingerprint
        or envelope.get("semantic_payload_fingerprint")
        != value.semantic_payload_fingerprint
        or _semantic_payload_fingerprint(
            payload,
            tensor_ledger_fingerprint=value.tensor_ledger_fingerprint,
        )
        != value.semantic_payload_fingerprint
    ):
        raise ValueError("OOF cache tensor ledger changed")
    return envelope, payload


def issue_oof_cache_reader(
    cache_artifact: VerifiedOOFCacheArtifact,
    *,
    reader_id: str,
    terminal_seal: VerifiedOOFTerminalSeal | None = None,
) -> VerifiedOOFCacheReader:
    cache = require_verified_oof_cache_artifact(cache_artifact)
    if reader_id not in cache.reader_allowlist:
        raise PermissionError("reader is not on this exact cache allowlist")
    terminal_fp: str | None = None
    authorization_event = OOF_EVENT_TRAIN_CACHE_CREATED
    if cache.partition == "holdout":
        seal = require_verified_oof_terminal_seal(terminal_seal)
        if (
            seal.fold_id != cache.fold_id
            or seal.closure_fingerprint != cache.closure_fingerprint
            or seal.event_index >= cache.creation_event
            or cache.terminal_seal_fingerprint
            != seal.seal_fingerprint
        ):
            raise PermissionError("holdout reader lacks the preceding seal")
        terminal_fp = seal.seal_fingerprint
        authorization_event = OOF_EVENT_HOLDOUT_FIRST_OPEN
    elif terminal_seal is not None:
        raise PermissionError("train reader cannot receive a holdout seal")
    else:
        _require_holdout_cache_paths_absent(cache)
    body = {
        "fold_id": cache.fold_id,
        "partition": cache.partition,
        "arm": cache.arm,
        "closure_fingerprint": cache.closure_fingerprint,
        "semantic_payload_fingerprint": (
            cache.semantic_payload_fingerprint
        ),
        "cache_artifact_fingerprint": cache.artifact_fingerprint,
        "reader_id": reader_id,
        "authorization_event": authorization_event,
        "holdout_terminal_seal_fingerprint": terminal_fp,
    }
    fingerprint = stable_fingerprint(body)
    return _register(VerifiedOOFCacheReader(
        fold_id=cache.fold_id,
        partition=cache.partition,
        arm=cache.arm,
        closure_fingerprint=cache.closure_fingerprint,
        semantic_payload_fingerprint=cache.semantic_payload_fingerprint,
        cache_artifact_fingerprint=cache.artifact_fingerprint,
        reader_id=reader_id,
        authorization_event=authorization_event,
        holdout_terminal_seal_fingerprint=terminal_fp,
        authorization_fingerprint=fingerprint,
        _cache_token=cache,
        _issuer=_TOKEN_ISSUER,
    ))


def load_oof_cache_payload(
    reader: VerifiedOOFCacheReader,
) -> object:
    authorization = require_verified_oof_cache_reader(reader)
    cache = require_verified_oof_cache_artifact(
        authorization._cache_token
    )
    if (
        authorization.cache_artifact_fingerprint
        != cache.artifact_fingerprint
        or authorization.reader_id not in cache.reader_allowlist
    ):
        raise PermissionError("OOF cache reader binding changed")
    _, payload = _load_envelope(cache)
    return payload


def load_persisted_oof_cache_payload(
    entry: Mapping[str, object],
    *,
    runtime_root: str | Path,
    fold_id: int,
) -> object:
    """Mechanically rebuild one cache named by a validated fold receipt.

    The file path is derived from the authorization runtime root and slot;
    the caller-supplied ``realpath`` is accepted only when it equals that
    derivation exactly.  This function intentionally issues no reader token:
    it is solely for the third-process evidence verifier after all terminals
    and holdout ledgers already exist.
    """

    if not isinstance(entry, Mapping):
        raise TypeError("OOF persisted cache entry must be a mapping")
    partition = entry.get("partition")
    arm = entry.get("arm")
    if (
        isinstance(fold_id, bool)
        or fold_id not in range(4)
        or partition not in {"train", "holdout"}
        or arm not in OOF_ARMS
    ):
        raise ValueError("OOF persisted cache slot is invalid")
    root = Path(runtime_root)
    if not root.is_absolute():
        raise ValueError("OOF runtime root must be absolute")
    path = (
        root
        / f"fold_{fold_id}"
        / str(partition)
        / _DIRECTORY_BY_ARM[str(arm)]
        / "cache.pt"
    )
    if entry.get("realpath") != str(path):
        raise PermissionError("OOF cache receipt path is not the fixed slot")
    required = {
        "cache_id",
        "artifact_fingerprint",
        "tensor_ledger_fingerprint",
        "partition",
        "arm",
        "closure_fingerprint",
        "terminal_seal_fingerprint",
        "semantic_payload_fingerprint",
        "root_source_ids",
        "sample_ids",
        "realpath",
        "device",
        "inode",
        "size_bytes",
        "file_sha256",
        "creation_phase",
        "creation_event",
        "reader_allowlist",
        "hardlink_count",
        "fiemap_extent_flags",
    }
    if not required.issubset(entry):
        raise ValueError("OOF persisted cache receipt fields are incomplete")
    roots = entry.get("root_source_ids")
    samples = entry.get("sample_ids")
    readers = entry.get("reader_allowlist")
    if (
        not isinstance(roots, list)
        or any(not isinstance(item, str) for item in roots)
        or not isinstance(samples, list)
        or any(not isinstance(item, str) for item in samples)
        or not isinstance(readers, list)
        or any(not isinstance(item, str) for item in readers)
    ):
        raise TypeError("OOF persisted cache receipt lists changed")
    token_payload = {
        **dict(entry),
        "fiemap_extent_flags": list(entry["fiemap_extent_flags"]),
    }
    token = VerifiedOOFCacheArtifact(
        payload_json=canonical_json(token_payload),
        fold_id=fold_id,
        partition=str(partition),
        arm=str(arm),
        closure_fingerprint=str(entry["closure_fingerprint"]),
        terminal_seal_fingerprint=(
            None
            if entry["terminal_seal_fingerprint"] is None
            else str(entry["terminal_seal_fingerprint"])
        ),
        semantic_payload_fingerprint=str(
            entry["semantic_payload_fingerprint"]
        ),
        cache_id=str(entry["cache_id"]),
        path=str(path),
        size_bytes=int(entry["size_bytes"]),
        file_sha256=str(entry["file_sha256"]),
        device=int(entry["device"]),
        inode=int(entry["inode"]),
        hardlink_count=int(entry["hardlink_count"]),
        root_source_ids=tuple(roots),
        sample_ids=tuple(samples),
        creation_phase=str(entry["creation_phase"]),
        creation_event=int(entry["creation_event"]),
        reader_allowlist=tuple(readers),
        tensor_ledger_fingerprint=str(
            entry["tensor_ledger_fingerprint"]
        ),
        artifact_fingerprint=str(entry["artifact_fingerprint"]),
        _issuer=_TOKEN_ISSUER,
    )
    _, payload = _load_envelope(token)
    return payload


def _require_holdout_cache_paths_absent(
    cache: VerifiedOOFCacheArtifact,
) -> None:
    path = Path(cache.path)
    fold_directory = path.parents[2]
    expected_fold = f"fold_{cache.fold_id}"
    if fold_directory.name != expected_fold:
        raise RuntimeError("OOF cache fold path changed")
    holdout = fold_directory / "holdout"
    for arm_directory in _DIRECTORY_BY_ARM.values():
        candidate = holdout / arm_directory / "cache.pt"
        if candidate.exists() or candidate.is_symlink():
            raise PermissionError(
                "train cache reader requires every holdout cache path "
                "to remain absent"
            )


def verify_oof_six_cache_independence(
    cache_artifacts: Sequence[VerifiedOOFCacheArtifact],
) -> VerifiedOOFSixCacheSet:
    if len(cache_artifacts) != 6:
        raise ValueError("OOF fold requires exactly six cache artifacts")
    tokens = tuple(
        require_verified_oof_cache_artifact(value)
        for value in cache_artifacts
    )
    fold_ids = {value.fold_id for value in tokens}
    inventory = {(value.partition, value.arm) for value in tokens}
    expected = {
        (partition, arm)
        for partition in ("train", "holdout")
        for arm in OOF_ARMS
    }
    if len(fold_ids) != 1 or inventory != expected:
        raise ValueError("OOF six-cache fold/inventory changed")
    physical = {(value.device, value.inode) for value in tokens}
    if len(physical) != 6 or len({value.path for value in tokens}) != 6:
        raise PermissionError("OOF caches reuse physical files or paths")
    loaded = [_load_envelope(value) for value in tokens]
    storages = [_storage_ids(payload) for _, payload in loaded]
    if any(not values for values in storages):
        raise ValueError("OOF cache contains no tensor storage")
    for left in range(len(storages)):
        for right in range(left + 1, len(storages)):
            if storages[left] & storages[right]:
                raise PermissionError(
                    "OOF caches share actual loaded tensor storage"
                )
    entries = [
        {
            "cache_id": value.cache_id,
            "artifact_fingerprint": value.artifact_fingerprint,
            "tensor_ledger_fingerprint": (
                value.tensor_ledger_fingerprint
            ),
            "partition": value.partition,
            "arm": value.arm,
            "closure_fingerprint": value.closure_fingerprint,
            "terminal_seal_fingerprint": (
                value.terminal_seal_fingerprint
            ),
            "semantic_payload_fingerprint": (
                value.semantic_payload_fingerprint
            ),
            "root_source_ids": list(value.root_source_ids),
            "sample_ids": list(value.sample_ids),
            "realpath": value.path,
            "device": value.device,
            "inode": value.inode,
            "size_bytes": value.size_bytes,
            "file_sha256": value.file_sha256,
            "creation_phase": value.creation_phase,
            "creation_event": value.creation_event,
            "reader_allowlist": list(value.reader_allowlist),
            "is_symlink": False,
            "hardlink_count": value.hardlink_count,
            "fiemap_extent_flags": value.payload[
                "fiemap_extent_flags"
            ],
            "is_reflink": False,
            "shared_tensor_storage": False,
            "mmap_reused": False,
            "process_cache_reused": False,
        }
        for value in sorted(
            tokens,
            key=lambda item: (item.partition, item.arm),
        )
    ]
    body = {
        "schema_version": OOF_CACHE_SET_SCHEMA,
        "fold_id": next(iter(fold_ids)),
        "cache_artifact_fingerprints": sorted(
            value.artifact_fingerprint for value in tokens
        ),
        "entries": entries,
    }
    fingerprint = stable_fingerprint(body)
    return _register(VerifiedOOFSixCacheSet(
        fold_id=next(iter(fold_ids)),
        cache_artifact_fingerprints=tuple(
            sorted(value.artifact_fingerprint for value in tokens)
        ),
        entries_json=canonical_json(entries),
        set_fingerprint=fingerprint,
        _issuer=_TOKEN_ISSUER,
    ))


__all__ = [
    "OOF_ARMS",
    "OOF_CACHE_ARTIFACT_SCHEMA",
    "OOF_CACHE_ENVELOPE_SCHEMA",
    "OOF_CACHE_SET_SCHEMA",
    "OOF_EVENT_HOLDOUT_CACHE_CREATED",
    "OOF_EVENT_HOLDOUT_FIRST_OPEN",
    "OOF_EVENT_TERMINALS_SEALED",
    "OOF_EVENT_TRAINING_RUN_START",
    "OOF_EVENT_TRAIN_CACHE_CREATED",
    "OOF_TERMINAL_SEAL_SCHEMA",
    "VerifiedOOFCacheArtifact",
    "VerifiedOOFCacheReader",
    "VerifiedOOFSixCacheSet",
    "VerifiedOOFTerminalSeal",
    "issue_oof_cache_reader",
    "load_oof_cache_payload",
    "load_persisted_oof_cache_payload",
    "require_verified_oof_cache_artifact",
    "require_verified_oof_cache_reader",
    "require_verified_oof_terminal_seal",
    "save_oof_cache_artifact_new",
    "seal_oof_training_terminals",
    "verify_oof_six_cache_independence",
]
