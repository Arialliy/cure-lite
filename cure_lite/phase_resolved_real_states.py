"""Lineage-safe real-state population and deterministic PFCR epoch pools."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .cache.schema import file_sha256, stable_fingerprint
from .experiment.training_pipeline import PreparedTrainingCatalog
from .phase_resolved_real_cache import PFCRRealCacheAdapter
from .sampling import (
    choose_uniform_factual_gt_id,
    choose_uniform_legal_identity,
)
from .train.pools import BranchPools


PFCR_LINEAGE_ALLOWLIST_SCHEMA = (
    "cure-lite-pfcr-lineage-allowlist-v1"
)
PFCR_REAL_STATE_CATALOG_SCHEMA = (
    "cure-lite-pfcr-real-state-catalog-v1"
)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _identity(
    value: object,
    *,
    role: str,
) -> tuple[str, int, int | None]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{role} identity must contain three fields")
    sample_id = value[0]
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"{role} sample_id must be non-empty")
    gt_id = _positive_int(value[1], name=f"{role} gt_id")
    pred = value[2]
    if role == "factual":
        if pred is not None:
            raise ValueError("factual identity pred_id must be null")
        return sample_id, gt_id, None
    pred_id = _positive_int(pred, name="legal pred_id")
    return sample_id, gt_id, pred_id


@dataclass(frozen=True)
class PFCRLineageAllowlist:
    """Frozen factual/legal identities from a passed P0-A1 receipt."""

    source_path: Path
    source_file_sha256: str
    source_receipt_fingerprint: str
    protocol_fingerprint: str
    source_catalog_fingerprint: str
    geometry_catalog_fingerprint: str
    eligible_catalog_fingerprint: str
    factual_identities: tuple[tuple[str, int, None], ...]
    legal_identities: tuple[tuple[str, int, int], ...]
    excluded_legal_identities: tuple[tuple[str, int, int], ...]
    allowlist_fingerprint: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": PFCR_LINEAGE_ALLOWLIST_SCHEMA,
            "source_path": str(self.source_path),
            "source_file_sha256": self.source_file_sha256,
            "source_receipt_fingerprint": (
                self.source_receipt_fingerprint
            ),
            "protocol_fingerprint": self.protocol_fingerprint,
            "source_catalog_fingerprint": (
                self.source_catalog_fingerprint
            ),
            "geometry_catalog_fingerprint": (
                self.geometry_catalog_fingerprint
            ),
            "eligible_catalog_fingerprint": (
                self.eligible_catalog_fingerprint
            ),
            "factual_identities": [
                list(identity) for identity in self.factual_identities
            ],
            "legal_identities": [
                list(identity) for identity in self.legal_identities
            ],
            "excluded_legal_identities": [
                list(identity)
                for identity in self.excluded_legal_identities
            ],
        }


def _strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("P0-A1 receipt may not be a symlink")
    source = path.resolve(strict=True)
    if not source.is_file():
        raise ValueError("P0-A1 receipt must be a regular file")
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("P0-A1 receipt must contain one JSON object")
    return value


def load_pfcr_lineage_allowlist(
    path: str | Path,
) -> PFCRLineageAllowlist:
    """Load the passed P0-A1 identity population without recomputing it."""

    raw_source = Path(path).expanduser()
    if raw_source.is_symlink():
        raise ValueError("P0-A1 receipt may not be a symlink")
    source = raw_source.resolve(strict=True)
    payload = _strict_json(source)
    receipt_fingerprint = payload.get("receipt_fingerprint")
    if not isinstance(receipt_fingerprint, str):
        raise ValueError("P0-A1 receipt fingerprint is missing")
    unsigned = dict(payload)
    unsigned.pop("receipt_fingerprint")
    if stable_fingerprint(unsigned) != receipt_fingerprint:
        raise ValueError("P0-A1 receipt fingerprint does not match")
    if (
        payload.get("schema_version")
        != "cure-lite-p0-a1-population-eligibility-v2"
        or payload.get("split") != "D_R"
        or payload.get("execution_status") != "completed"
        or payload.get("formal_status") != "pass"
        or payload.get("p0_a1_pass") is not True
    ):
        raise ValueError("PFCR requires one completed, passed P0-A1 receipt")
    accounting = payload.get("accounting")
    rules = payload.get("rule_outcomes")
    if (
        not isinstance(accounting, dict)
        or accounting.get("all_candidates_classified_exactly_once")
        is not True
        or accounting.get("duplicate_candidate_identities") != 0
        or accounting.get("duplicate_eligible_identities") != 0
        or accounting.get("invalid_retained_targets") != 0
        or accounting.get("unaccounted_targets") != 0
    ):
        raise ValueError("P0-A1 candidate accounting is not exact")
    required_rules = (
        "all_reachable_factual_geometry_eligible",
        "all_retained_bidirectional_one_to_one_lineage",
        "all_retained_exact_component_projection",
        "all_retained_area_ratio_within_gate",
        "all_retained_centroid_shift_within_gate",
    )
    if not isinstance(rules, dict) or any(
        rules.get(name) is not True for name in required_rules
    ):
        raise ValueError("P0-A1 retained population violates geometry rules")

    identities = payload.get("eligible_target_identities")
    if not isinstance(identities, dict) or set(identities) != {
        "factual",
        "legal",
    }:
        raise ValueError("P0-A1 eligible identities are invalid")
    raw_factual = identities["factual"]
    raw_legal = identities["legal"]
    if not isinstance(raw_factual, list) or not isinstance(raw_legal, list):
        raise ValueError("P0-A1 identity catalogs must be lists")
    factual = tuple(
        _identity(value, role="factual") for value in raw_factual
    )
    legal_general = tuple(
        _identity(value, role="legal") for value in raw_legal
    )
    legal = tuple(
        (sample_id, gt_id, int(pred_id))
        for sample_id, gt_id, pred_id in legal_general
    )
    if factual != tuple(sorted(set(factual))):
        raise ValueError("P0-A1 factual identities are not sorted unique")
    if legal != tuple(sorted(set(legal))):
        raise ValueError("P0-A1 legal identities are not sorted unique")

    excluded_rows = payload.get("excluded_targets")
    if not isinstance(excluded_rows, list):
        raise ValueError("P0-A1 excluded target ledger is invalid")
    excluded: list[tuple[str, int, int]] = []
    for row in excluded_rows:
        if (
            not isinstance(row, dict)
            or row.get("role") != "legal"
        ):
            raise ValueError("PFCR expects only legal P0-A1 exclusions")
        value = _identity(row.get("identity"), role="legal")
        excluded.append((value[0], value[1], int(value[2])))
    excluded_tuple = tuple(sorted(excluded))
    if (
        excluded_tuple != tuple(sorted(set(excluded_tuple)))
        or set(excluded_tuple) & set(legal)
    ):
        raise ValueError("P0-A1 excluded identities are inconsistent")

    counts = payload.get("counts")
    if (
        not isinstance(counts, dict)
        or counts.get("factual_eligible") != len(factual)
        or counts.get("legal_eligible") != len(legal)
        or counts.get("legal_geometry_excluded") != len(excluded_tuple)
        or counts.get("legal_candidates")
        != len(legal) + len(excluded_tuple)
    ):
        raise ValueError("P0-A1 identity counts do not close")
    names = (
        "protocol_fingerprint",
        "source_catalog_fingerprint",
        "geometry_catalog_fingerprint",
        "eligible_catalog_fingerprint",
    )
    if any(
        not isinstance(payload.get(name), str)
        or len(payload[name]) != 64
        for name in names
    ):
        raise ValueError("P0-A1 provenance fingerprint is invalid")
    base_payload = {
        "schema_version": PFCR_LINEAGE_ALLOWLIST_SCHEMA,
        "source_path": str(source),
        "source_file_sha256": file_sha256(source),
        "source_receipt_fingerprint": receipt_fingerprint,
        "protocol_fingerprint": payload["protocol_fingerprint"],
        "source_catalog_fingerprint": (
            payload["source_catalog_fingerprint"]
        ),
        "geometry_catalog_fingerprint": (
            payload["geometry_catalog_fingerprint"]
        ),
        "eligible_catalog_fingerprint": (
            payload["eligible_catalog_fingerprint"]
        ),
        "factual_identities": [list(value) for value in factual],
        "legal_identities": [list(value) for value in legal],
        "excluded_legal_identities": [
            list(value) for value in excluded_tuple
        ],
    }
    result = PFCRLineageAllowlist(
        source_path=source,
        source_file_sha256=base_payload["source_file_sha256"],
        source_receipt_fingerprint=receipt_fingerprint,
        protocol_fingerprint=payload["protocol_fingerprint"],
        source_catalog_fingerprint=(
            payload["source_catalog_fingerprint"]
        ),
        geometry_catalog_fingerprint=(
            payload["geometry_catalog_fingerprint"]
        ),
        eligible_catalog_fingerprint=(
            payload["eligible_catalog_fingerprint"]
        ),
        factual_identities=factual,
        legal_identities=legal,
        excluded_legal_identities=excluded_tuple,
        allowlist_fingerprint=stable_fingerprint(base_payload),
    )
    if result.canonical_payload() != base_payload:
        raise AssertionError("PFCR allowlist payload drifted")
    return result


@dataclass(frozen=True, eq=False)
class PFCRRealStateCatalog:
    """Zero-copy prepared states filtered by the frozen legal allowlist."""

    prepared: PreparedTrainingCatalog
    allowlist: PFCRLineageAllowlist
    selected_legal_indices: tuple[tuple[int, ...], ...]
    cache_contract_fingerprint: str
    factual_target_count: int
    factual_source_count: int
    factual_no_miss_source_count: int
    legal_target_count: int
    legal_source_count: int
    catalog_fingerprint: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": PFCR_REAL_STATE_CATALOG_SCHEMA,
            "cache_contract_fingerprint": (
                self.cache_contract_fingerprint
            ),
            "allowlist_fingerprint": (
                self.allowlist.allowlist_fingerprint
            ),
            "source_catalog_fingerprint": (
                self.allowlist.source_catalog_fingerprint
            ),
            "geometry_catalog_fingerprint": (
                self.allowlist.geometry_catalog_fingerprint
            ),
            "eligible_catalog_fingerprint": (
                self.allowlist.eligible_catalog_fingerprint
            ),
            "source_ids": list(self.prepared.source_ids),
            "factual_target_count": self.factual_target_count,
            "factual_source_count": self.factual_source_count,
            "factual_no_miss_source_count": (
                self.factual_no_miss_source_count
            ),
            "legal_target_count": self.legal_target_count,
            "legal_source_count": self.legal_source_count,
            "selected_legal_identities": [
                [
                    entry.sample_id,
                    entry.decoder_visible_legal_candidates[index].gt_id,
                    entry.decoder_visible_legal_candidates[index].pred_id,
                ]
                for entry, indices in zip(
                    self.prepared.entries,
                    self.selected_legal_indices,
                    strict=True,
                )
                for index in indices
            ],
            "excluded_legal_identities": [
                list(value)
                for value in self.allowlist.excluded_legal_identities
            ],
            "state_tensors_reused": True,
            "detector_metadata_is_model_input": False,
        }

    def verify_unchanged(self) -> None:
        for entry, indices in zip(
            self.prepared.entries,
            self.selected_legal_indices,
            strict=True,
        ):
            for index in indices:
                example = entry.synthetic_examples[index]
                if example.feature is not entry.source.feature:
                    raise RuntimeError(
                        "PFCR legal state no longer reuses its cached feature"
                    )


def build_pfcr_real_state_catalog(
    cache: PFCRRealCacheAdapter,
    allowlist: PFCRLineageAllowlist,
) -> PFCRRealStateCatalog:
    """Bind all factual states and exactly the passed legal population."""

    if not isinstance(cache, PFCRRealCacheAdapter):
        raise TypeError("cache must be PFCRRealCacheAdapter")
    if not isinstance(allowlist, PFCRLineageAllowlist):
        raise TypeError("allowlist must be PFCRLineageAllowlist")
    cache.verify_unchanged()
    prepared = cache.prepared_catalog
    factual = tuple(
        (entry.sample_id, gt_id, None)
        for entry in prepared.entries
        for gt_id in entry.reachable_gt_ids
    )
    if factual != allowlist.factual_identities:
        raise RuntimeError(
            "prepared factual population differs from frozen P0-A1"
        )
    raw_legal = {
        (
            entry.sample_id,
            candidate.gt_id,
            candidate.pred_id,
        )
        for entry in prepared.entries
        for candidate in entry.decoder_visible_legal_candidates
    }
    eligible = set(allowlist.legal_identities)
    excluded = set(allowlist.excluded_legal_identities)
    if raw_legal != eligible | excluded or eligible & excluded:
        raise RuntimeError(
            "prepared legal population does not close against P0-A1"
        )

    selected: list[tuple[int, ...]] = []
    selected_identities: list[tuple[str, int, int]] = []
    legal_sources: set[str] = set()
    for entry in prepared.entries:
        indices = tuple(
            index
            for index, candidate in enumerate(
                entry.decoder_visible_legal_candidates
            )
            if (
                entry.sample_id,
                candidate.gt_id,
                candidate.pred_id,
            )
            in eligible
        )
        for index in indices:
            candidate = entry.decoder_visible_legal_candidates[index]
            example = entry.synthetic_examples[index]
            if example.feature is not entry.source.feature:
                raise RuntimeError(
                    "PFCR state construction copied a cached feature"
                )
            identity = (
                entry.sample_id,
                candidate.gt_id,
                candidate.pred_id,
            )
            selected_identities.append(identity)
            legal_sources.add(entry.sample_id)
        selected.append(indices)
    selected_tuple = tuple(selected)
    if tuple(selected_identities) != allowlist.legal_identities:
        raise RuntimeError(
            "PFCR did not select the exact ordered P0-A1 legal population"
        )

    base_payload = {
        "schema_version": PFCR_REAL_STATE_CATALOG_SCHEMA,
        "cache_contract_fingerprint": (
            cache.contract.contract_fingerprint
        ),
        "allowlist_fingerprint": allowlist.allowlist_fingerprint,
        "source_catalog_fingerprint": (
            allowlist.source_catalog_fingerprint
        ),
        "geometry_catalog_fingerprint": (
            allowlist.geometry_catalog_fingerprint
        ),
        "eligible_catalog_fingerprint": (
            allowlist.eligible_catalog_fingerprint
        ),
        "source_ids": list(prepared.source_ids),
        "factual_target_count": len(factual),
        "factual_source_count": sum(
            bool(entry.reachable_gt_ids) for entry in prepared.entries
        ),
        "factual_no_miss_source_count": sum(
            entry.factual_no_miss_example is not None
            for entry in prepared.entries
        ),
        "legal_target_count": len(selected_identities),
        "legal_source_count": len(legal_sources),
        "selected_legal_identities": [
            list(value) for value in selected_identities
        ],
        "excluded_legal_identities": [
            list(value)
            for value in allowlist.excluded_legal_identities
        ],
        "state_tensors_reused": True,
        "detector_metadata_is_model_input": False,
    }
    result = PFCRRealStateCatalog(
        prepared=prepared,
        allowlist=allowlist,
        selected_legal_indices=selected_tuple,
        cache_contract_fingerprint=(
            cache.contract.contract_fingerprint
        ),
        factual_target_count=len(factual),
        factual_source_count=base_payload["factual_source_count"],
        factual_no_miss_source_count=base_payload[
            "factual_no_miss_source_count"
        ],
        legal_target_count=len(selected_identities),
        legal_source_count=len(legal_sources),
        catalog_fingerprint=stable_fingerprint(base_payload),
    )
    if result.canonical_payload() != base_payload:
        raise AssertionError("PFCR real state catalog payload drifted")
    result.verify_unchanged()
    cache.verify_unchanged()
    return result


def build_pfcr_epoch_pools(
    catalog: PFCRRealStateCatalog,
    *,
    epoch: int,
    global_seed: int,
) -> BranchPools:
    """Select one factual and one legal state per eligible source cycle."""

    if not isinstance(catalog, PFCRRealStateCatalog):
        raise TypeError("catalog must be PFCRRealStateCatalog")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise TypeError("global_seed must be an integer")

    factual_miss = []
    factual_no_miss = []
    synthetic = []
    for entry, indices in zip(
        catalog.prepared.entries,
        catalog.selected_legal_indices,
        strict=True,
    ):
        selected_gt_id = choose_uniform_factual_gt_id(
            entry.reachable_gt_ids,
            sample_id=entry.sample_id,
            epoch=epoch,
            global_seed=global_seed,
        )
        if selected_gt_id is not None:
            factual_miss.append(
                entry.factual_examples[
                    entry.reachable_gt_ids.index(selected_gt_id)
                ]
            )
        elif entry.factual_no_miss_example is not None:
            factual_no_miss.append(entry.factual_no_miss_example)

        legal_identities = tuple(
            entry.decoder_visible_legal_candidates[index].identity
            for index in indices
        )
        selected_legal = choose_uniform_legal_identity(
            legal_identities,
            sample_id=entry.sample_id,
            epoch=epoch,
            global_seed=global_seed,
        )
        if selected_legal is not None:
            selected_index = next(
                index
                for index in indices
                if entry.decoder_visible_legal_candidates[
                    index
                ].identity
                == selected_legal
            )
            synthetic.append(entry.synthetic_examples[selected_index])

    pools = BranchPools(
        factual_miss=tuple(factual_miss),
        factual_no_miss=tuple(factual_no_miss),
        synthetic=tuple(synthetic),
    )
    if (
        len(pools.factual_miss) != catalog.factual_source_count
        or len(pools.factual_no_miss)
        != catalog.factual_no_miss_source_count
        or len(pools.synthetic) != catalog.legal_source_count
    ):
        raise RuntimeError("PFCR epoch pool support unexpectedly changed")
    return pools


__all__ = [
    "PFCR_LINEAGE_ALLOWLIST_SCHEMA",
    "PFCR_REAL_STATE_CATALOG_SCHEMA",
    "PFCRLineageAllowlist",
    "PFCRRealStateCatalog",
    "build_pfcr_epoch_pools",
    "build_pfcr_real_state_catalog",
    "load_pfcr_lineage_allowlist",
]
