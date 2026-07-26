"""Deterministic D_R-only publication of a tensor-free pair catalog.

This module is deliberately narrower than an experiment runner.  It accepts
an already constructed :class:`PairCatalog`, writes only JSON manifests and
receipts, and publishes ``COMPLETE.json`` last.  It never loads D_V/D_T,
trains a decoder, or serializes feature tensors.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from ..cache.schema import file_sha256, stable_fingerprint
from ..paired_types import PAIR_KINDS, PairCatalog


PAIR_PREFLIGHT_MANIFEST_SCHEMA = "cure-lite-pair-preflight-manifest-v1"
PAIR_PREFLIGHT_RECEIPT_SCHEMA = "cure-lite-pair-preflight-receipt-v1"
PAIR_PREFLIGHT_COMPLETE_SCHEMA = "cure-lite-pair-preflight-complete-v1"
_MANIFEST_NAME = "pair_catalog_manifest.json"
_RECEIPT_NAME = "preflight_receipt.json"
_COMPLETE_NAME = "COMPLETE.json"
_INCOMPLETE_NAME = ".incomplete"


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_new_json(path: Path, payload: object) -> None:
    """Atomically create one deterministic JSON file without replacement."""

    encoded = _json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite paired preflight artifact {path}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")

    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{name} contains non-finite number {value}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _canonical_catalog(catalog: PairCatalog) -> dict[str, object]:
    if not isinstance(catalog, PairCatalog):
        raise TypeError("catalog must be a PairCatalog")
    if catalog.split != "D_R":
        raise ValueError("paired preflight permits only D_R")
    payload = catalog.canonical_payload()
    if stable_fingerprint(payload) != catalog.catalog_fingerprint:
        raise RuntimeError("PairCatalog fingerprint does not reproduce")
    # Serialization is also the explicit tensor-free boundary: torch tensors
    # and all other non-JSON objects are rejected here before output creation.
    _json_bytes(payload)
    return payload


def _source_accounting(catalog: PairCatalog) -> dict[str, object]:
    included = (
        *catalog.clean_positive,
        *catalog.component_null,
        *catalog.identity_null,
    )
    group_by_sample: dict[str, str] = {}
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    feature_by_sample: dict[str, str] = {}
    for row in included:
        prior_group = group_by_sample.setdefault(row.sample_id, row.group_id)
        if prior_group != row.group_id:
            raise RuntimeError("one source maps to multiple group IDs")
        prior_feature = feature_by_sample.setdefault(
            row.sample_id,
            row.feature_fingerprint,
        )
        if prior_feature != row.feature_fingerprint:
            raise RuntimeError("one source maps to multiple frozen features")
        counts[row.sample_id][row.pair_kind] += 1
    for row in catalog.exclusions:
        prior_group = group_by_sample.setdefault(row.sample_id, row.group_id)
        if prior_group != row.group_id:
            raise RuntimeError("one excluded source maps to multiple group IDs")
        counts[row.sample_id]["exclusions"] += 1

    identity_sources = [row.sample_id for row in catalog.identity_null]
    if len(identity_sources) != len(set(identity_sources)):
        raise RuntimeError("identity-null rows must be one-per-source")
    if set(identity_sources) != set(group_by_sample):
        raise RuntimeError(
            "identity-null rows must account for every prepared D_R source"
        )
    rows = []
    for sample_id in sorted(group_by_sample):
        row_counts = counts[sample_id]
        rows.append(
            {
                "sample_id": sample_id,
                "group_id": group_by_sample[sample_id],
                "clean_positive": row_counts["clean_positive"],
                "component_null": row_counts["component_null"],
                "identity_null": row_counts["identity_null"],
                "exclusions": row_counts["exclusions"],
            }
        )
    return {
        "prepared_source_images": len(identity_sources),
        "prepared_source_groups": len(set(group_by_sample.values())),
        "trainable_source_images": len(
            {row.sample_id for row in catalog.clean_positive}
        ),
        "control_source_images": len(
            {
                row.sample_id
                for row in (*catalog.component_null, *catalog.identity_null)
            }
        ),
        "excluded_candidate_source_images": len(
            {row.sample_id for row in catalog.exclusions}
        ),
        "unique_feature_fingerprints": len(set(feature_by_sample.values())),
        "feature_fingerprint_references": len(included),
        "per_source": rows,
    }


def build_pair_preflight_manifest(catalog: PairCatalog) -> dict[str, object]:
    """Return the complete tensor-free canonical pair manifest."""

    canonical = _canonical_catalog(catalog)
    core: dict[str, object] = {
        "schema_version": PAIR_PREFLIGHT_MANIFEST_SCHEMA,
        "split": "D_R",
        "pair_catalog_fingerprint": catalog.catalog_fingerprint,
        "canonical_pair_catalog": canonical,
        "storage_contract": {
            "json_only": True,
            "raw_tensor_payloads_written": False,
            "feature_tensor_files_written": 0,
            "feature_storage": "fingerprint_only_no_tensor_duplication",
        },
    }
    return {
        **core,
        "manifest_fingerprint": stable_fingerprint(core),
    }


def build_pair_preflight_receipt(
    catalog: PairCatalog,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    """Summarize bindings, populations, and the non-authorizing gate."""

    canonical = _canonical_catalog(catalog)
    if not isinstance(manifest, Mapping):
        raise TypeError("manifest must be a mapping")
    manifest_payload = dict(manifest)
    manifest_core = dict(manifest_payload)
    manifest_fingerprint = manifest_core.pop("manifest_fingerprint", None)
    if (
        stable_fingerprint(manifest_core) != manifest_fingerprint
        or manifest_core.get("canonical_pair_catalog") != canonical
    ):
        raise ValueError("preflight manifest does not bind this PairCatalog")

    all_pairs = (
        *catalog.clean_positive,
        *catalog.component_null,
        *catalog.identity_null,
    )
    reason_counts = Counter(
        reason
        for row in catalog.exclusions
        for reason in row.reason_codes
    )
    exclusion_kind_counts = Counter(
        row.pair_kind for row in catalog.exclusions
    )
    source_accounting = _source_accounting(catalog)
    trainable_nonempty = bool(catalog.clean_positive)
    two_sources = source_accounting["trainable_source_images"] >= 2
    preflight_passed = trainable_nonempty and two_sources
    core: dict[str, object] = {
        "schema_version": PAIR_PREFLIGHT_RECEIPT_SCHEMA,
        "execution_status": "completed",
        "split": "D_R",
        "input_bindings": {
            "paired_protocol_fingerprint": (
                catalog.paired_protocol_fingerprint
            ),
            "geometry_catalog_fingerprint": (
                catalog.geometry_catalog_fingerprint
            ),
            "prepared_analysis_population_fingerprint": (
                catalog.source_catalog_fingerprint
            ),
            "split_manifest_fingerprint": catalog.manifest_fingerprint,
            "pair_catalog_fingerprint": catalog.catalog_fingerprint,
            "pair_manifest_fingerprint": manifest_fingerprint,
            "pair_manifest_file_sha256": _bytes_sha256(
                _json_bytes(manifest_payload)
            ),
        },
        "counts": {
            "clean_positive": len(catalog.clean_positive),
            "component_null": len(catalog.component_null),
            "identity_null": len(catalog.identity_null),
            "included_pairs": len(all_pairs),
            "trainable_pairs": len(catalog.trainable_pairs),
            "control_pairs": (
                len(catalog.component_null) + len(catalog.identity_null)
            ),
            "exclusions": len(catalog.exclusions),
        },
        "source_accounting": source_accounting,
        "exclusion_accounting": {
            "by_pair_kind": {
                kind: exclusion_kind_counts[kind] for kind in PAIR_KINDS
            },
            "by_reason": {
                reason: reason_counts[reason]
                for reason in sorted(reason_counts)
            },
        },
        "integrity_gates": {
            "d_r_only": True,
            "catalog_fingerprint_verified": True,
            "tensor_free_manifest_verified": True,
            "all_trainable_pairs_projection_visible": all(
                row.projection_visible for row in catalog.trainable_pairs
            ),
            "trainable_population_nonempty": trainable_nonempty,
            "at_least_two_trainable_sources": two_sources,
            "preflight_passed": preflight_passed,
        },
        "execution_policy": {
            "allowed_runtime_splits": ["D_R"],
            "training_performed": False,
            "training_authorized_by_this_artifact": False,
            "d_v_accessed": False,
            "d_t_accessed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "feature_tensors_persisted": False,
        },
        "runner_boundary": {
            "in_memory_catalog_artifact_writer_implemented": True,
            "verified_real_cache_loader_entrypoint_implemented": True,
            "entrypoint": "tools/run_paired_preflight.py",
            "remaining_gap": None,
        },
    }
    return {**core, "receipt_fingerprint": stable_fingerprint(core)}


@dataclass(frozen=True)
class PublishedPairPreflight:
    root: Path
    pair_catalog_fingerprint: str
    manifest_fingerprint: str
    receipt_fingerprint: str
    complete_fingerprint: str

    def verify_unchanged(self) -> None:
        verified = load_pair_preflight_artifact(self.root)
        if (
            verified.pair_catalog_fingerprint
            != self.pair_catalog_fingerprint
            or verified.manifest_fingerprint != self.manifest_fingerprint
            or verified.receipt_fingerprint != self.receipt_fingerprint
            or verified.complete_fingerprint != self.complete_fingerprint
        ):
            raise RuntimeError("published paired preflight identity changed")


def write_pair_preflight_artifact(
    catalog: PairCatalog,
    output_dir: str | Path,
) -> PublishedPairPreflight:
    """Create and seal one deterministic, JSON-only preflight directory."""

    manifest = build_pair_preflight_manifest(catalog)
    receipt = build_pair_preflight_receipt(catalog, manifest)
    requested = Path(output_dir).expanduser()
    if requested.is_symlink() or requested.exists():
        raise FileExistsError(
            f"refusing to overwrite paired preflight output {requested}"
        )
    root = requested.resolve(strict=False)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    incomplete = root / _INCOMPLETE_NAME
    incomplete.open("xb").close()
    _write_new_json(root / _MANIFEST_NAME, manifest)
    _write_new_json(root / _RECEIPT_NAME, receipt)
    complete_core: dict[str, object] = {
        "schema_version": PAIR_PREFLIGHT_COMPLETE_SCHEMA,
        "execution_status": "completed",
        "split": "D_R",
        "pair_catalog_fingerprint": catalog.catalog_fingerprint,
        "manifest_fingerprint": manifest["manifest_fingerprint"],
        "receipt_fingerprint": receipt["receipt_fingerprint"],
        "artifact_files": {
            _MANIFEST_NAME: file_sha256(root / _MANIFEST_NAME),
            _RECEIPT_NAME: file_sha256(root / _RECEIPT_NAME),
        },
        "artifact_file_count": 2,
        "raw_tensor_artifact_file_count": 0,
    }
    complete = {
        **complete_core,
        "complete_fingerprint": stable_fingerprint(complete_core),
    }
    _write_new_json(root / _COMPLETE_NAME, complete)
    incomplete.unlink()
    return load_pair_preflight_artifact(root)


def load_pair_preflight_artifact(
    output_dir: str | Path,
) -> PublishedPairPreflight:
    """Load and fully verify one completed preflight publication."""

    root = Path(output_dir).expanduser().resolve(strict=False)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("paired preflight root must be a regular directory")
    if (root / _INCOMPLETE_NAME).exists():
        raise RuntimeError("paired preflight publication is incomplete")
    expected_names = {_MANIFEST_NAME, _RECEIPT_NAME, _COMPLETE_NAME}
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != expected_names:
        raise RuntimeError("paired preflight artifact inventory changed")
    manifest = _strict_json(root / _MANIFEST_NAME, name="pair manifest")
    receipt = _strict_json(root / _RECEIPT_NAME, name="preflight receipt")
    complete = _strict_json(root / _COMPLETE_NAME, name="COMPLETE receipt")

    manifest_core = dict(manifest)
    manifest_fingerprint = manifest_core.pop("manifest_fingerprint", None)
    if stable_fingerprint(manifest_core) != manifest_fingerprint:
        raise RuntimeError("pair manifest fingerprint mismatch")
    receipt_core = dict(receipt)
    receipt_fingerprint = receipt_core.pop("receipt_fingerprint", None)
    if stable_fingerprint(receipt_core) != receipt_fingerprint:
        raise RuntimeError("preflight receipt fingerprint mismatch")
    complete_core = dict(complete)
    complete_fingerprint = complete_core.pop("complete_fingerprint", None)
    if stable_fingerprint(complete_core) != complete_fingerprint:
        raise RuntimeError("COMPLETE fingerprint mismatch")
    files = complete.get("artifact_files")
    if not isinstance(files, dict) or files != {
        _MANIFEST_NAME: file_sha256(root / _MANIFEST_NAME),
        _RECEIPT_NAME: file_sha256(root / _RECEIPT_NAME),
    }:
        raise RuntimeError("paired preflight file hashes changed")
    pair_catalog_fingerprint = manifest.get("pair_catalog_fingerprint")
    bindings = receipt.get("input_bindings")
    if not isinstance(bindings, dict):
        raise RuntimeError("preflight receipt bindings are malformed")
    if not (
        complete.get("split") == receipt.get("split") == manifest.get("split") == "D_R"
        and complete.get("pair_catalog_fingerprint")
        == bindings.get("pair_catalog_fingerprint")
        == pair_catalog_fingerprint
        and complete.get("manifest_fingerprint") == manifest_fingerprint
        and complete.get("receipt_fingerprint") == receipt_fingerprint
        and bindings.get("pair_manifest_fingerprint") == manifest_fingerprint
        and bindings.get("pair_manifest_file_sha256")
        == file_sha256(root / _MANIFEST_NAME)
    ):
        raise RuntimeError("paired preflight cross-file bindings disagree")
    if receipt.get("execution_policy", {}).get(
        "feature_tensors_persisted"
    ) is not False:
        raise RuntimeError("paired preflight violates tensor-free storage")
    return PublishedPairPreflight(
        root=root,
        pair_catalog_fingerprint=str(pair_catalog_fingerprint),
        manifest_fingerprint=str(manifest_fingerprint),
        receipt_fingerprint=str(receipt_fingerprint),
        complete_fingerprint=str(complete_fingerprint),
    )


__all__ = [
    "PAIR_PREFLIGHT_COMPLETE_SCHEMA",
    "PAIR_PREFLIGHT_MANIFEST_SCHEMA",
    "PAIR_PREFLIGHT_RECEIPT_SCHEMA",
    "PublishedPairPreflight",
    "build_pair_preflight_manifest",
    "build_pair_preflight_receipt",
    "load_pair_preflight_artifact",
    "write_pair_preflight_artifact",
]
