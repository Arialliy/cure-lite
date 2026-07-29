"""Two-phase, allowlist-only real-D_R input construction for OOF-4.

Unlike ``build_coverage_state_real_dr_inputs``, this module never constructs
the complete 160-row tensor graph.  It validates the complete signed
manifest/index metadata, then delegates to the restricted D_R loader, which
opens/hashes/deserializes payload files only for the exact fold partition.
The holdout entry point additionally requires the exact terminal seal.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.coverage_state_observability import (
    CoverageStateObservabilityDecision,
    audit_population_observability,
)
from cure_lite.coverage_state_precomputed_cache import (
    CoverageStateScalarCache,
    prepare_scalar_coverage_state_cache,
)
from cure_lite.coverage_state_sobolev import CoverageStateSobolevConfig
from cure_lite.data import ManifestImageDataset, PreprocessConfig
from cure_lite.experiment.cache_pipeline import (
    LoadedDRCacheBundle,
    load_d_r_cache_bundle,
)
from cure_lite.experiment.coverage_state_observability_protocol import (
    CoverageStateObservabilityProtocol,
)
from cure_lite.experiment.coverage_state_raw_catalog import (
    build_coverage_state_raw_catalog,
)
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    CoverageStateRealDRSourceBinding,
)
from cure_lite.experiment.geometry_catalog_protocol import (
    GeometryCatalogProtocol,
)
from cure_lite.experiment.geometry_safe_catalog import (
    GeometrySafeCatalog,
    build_geometry_safe_catalog,
)
from cure_lite.experiment.training_pipeline import (
    CachedTrainingSource,
    prepare_training_catalog,
)
from cure_lite.splits import SplitManifest, load_and_validate_manifest

from .oof_cache import (
    VerifiedOOFTerminalSeal,
    require_verified_oof_terminal_seal,
)
from .oof_evaluation import (
    OOFEvaluationDataset,
    seal_oof_evaluation_dataset,
    seal_oof_evaluation_sample,
)
from .oof_split import (
    VerifiedOOFFoldClosure,
    require_verified_oof_fold_closure,
)


OOF_TRAIN_INPUTS_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-oof-train-restricted-inputs-v1"
)
OOF_HOLDOUT_INPUTS_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-oof-holdout-restricted-inputs-v1"
)


def _strict_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("geometry receipt must contain an object")
    return value


def _partition_bundle(
    *,
    source_binding: CoverageStateRealDRSourceBinding,
    protocol: CoverageStateObservabilityProtocol,
    preprocess: PreprocessConfig,
    sample_ids: tuple[str, ...],
) -> tuple[SplitManifest, LoadedDRCacheBundle]:
    source_binding.verify_unchanged()
    if (
        type(protocol) is not CoverageStateObservabilityProtocol
        or type(preprocess) is not PreprocessConfig
        or not sample_ids
    ):
        raise TypeError("OOF restricted input metadata has a wrong type")
    manifest = load_and_validate_manifest(source_binding.manifest_path)
    if (
        manifest.dataset != protocol.dataset
        or protocol.split != "D_R"
        or set(sample_ids)
        - {record.sample_id for record in manifest.records_for("D_R")}
    ):
        raise PermissionError("OOF partition is outside frozen D_R")
    dataset = ManifestImageDataset(
        manifest,
        "D_R",
        preprocess,
        manifest_path=source_binding.manifest_path,
    )
    bundle = load_d_r_cache_bundle(
        source_binding.state_index_path,
        dataset,
        expected_base_fingerprint=protocol.input_binding.base_fingerprint,
        allowed_sample_ids=sample_ids,
    )
    if tuple(row.sample_id for row in bundle.rows) != tuple(
        sorted(sample_ids)
    ):
        raise PermissionError("restricted loader returned the wrong partition")
    return manifest, bundle


def _evaluation_dataset(
    bundle: LoadedDRCacheBundle,
    closure: VerifiedOOFFoldClosure,
    *,
    partition: str,
) -> OOFEvaluationDataset:
    rows = []
    for source in bundle.rows:
        state = source.state.normalized()
        rows.append(seal_oof_evaluation_sample(
            sample_id=source.sample_id,
            root_source_id=closure.root_by_sample[source.sample_id],
            base_probability=source.base_output.probability,
            feature=source.base_output.feature,
            gt_mask=state.gt_labels > 0,
            valid_mask=state.image_valid_mask,
            anchor_miss_ids=(
                int(value) for value in state.real_miss_ids.tolist()
            ),
            reachable_anchor_miss_ids=(
                int(value) for value in state.reachable_miss_ids.tolist()
            ),
        ))
    return seal_oof_evaluation_dataset(
        fold_id=closure.fold_id,
        partition=partition,
        closure_fingerprint=closure.closure_fingerprint,
        rows=rows,
    )


def _audit_geometry_subset_against_frozen_receipt(
    geometry: GeometrySafeCatalog,
    receipt_path: Path,
    *,
    sample_ids: tuple[str, ...],
) -> None:
    receipt = _strict_json(receipt_path)
    if receipt.get("receipt_fingerprint") is None:
        raise ValueError("frozen geometry receipt is not sealed")
    allowed = set(sample_ids)
    actual = geometry.canonical_payload()
    for field in (
        "sample_audits",
        "factual_records",
        "legal_records",
        "outside_population_records",
    ):
        upstream = receipt.get(field)
        current = actual.get(field)
        if not isinstance(upstream, list) or not isinstance(current, list):
            raise ValueError("frozen geometry receipt fields changed")
        expected = [
            row
            for row in upstream
            if isinstance(row, dict) and row.get("sample_id") in allowed
        ]
        if current != expected:
            raise RuntimeError(
                f"OOF {field} differs from frozen full-D_R geometry"
            )
    if (
        actual.get("schema_version") != receipt.get("schema_version")
        or actual.get("protocol_fingerprint")
        != receipt.get("protocol_fingerprint")
    ):
        raise RuntimeError("OOF geometry protocol binding changed")


@dataclass(frozen=True, eq=False)
class OOFRestrictedTrainInputs:
    fold_id: int
    closure_fingerprint: str
    scalar_cache: CoverageStateScalarCache
    evaluation_dataset: OOFEvaluationDataset
    source_binding_fingerprint: str
    partition_fingerprint: str

    def verify_unchanged(self) -> None:
        self.scalar_cache.verify_unchanged()
        self.evaluation_dataset.verify_unchanged()
        body = {
            "schema_version": OOF_TRAIN_INPUTS_SCHEMA,
            "fold_id": self.fold_id,
            "closure_fingerprint": self.closure_fingerprint,
            "source_binding_fingerprint": self.source_binding_fingerprint,
            "sample_ids": list(self.evaluation_dataset.sample_ids),
            "scalar_cache_fingerprint": self.scalar_cache.cache_fingerprint,
            "evaluation_dataset_fingerprint": (
                self.evaluation_dataset.dataset_fingerprint
            ),
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
        if stable_fingerprint(body) != self.partition_fingerprint:
            raise RuntimeError("OOF train restricted inputs changed")


@dataclass(frozen=True, eq=False)
class OOFRestrictedHoldoutInputs:
    fold_id: int
    closure_fingerprint: str
    terminal_seal_fingerprint: str
    evaluation_dataset: OOFEvaluationDataset
    source_binding_fingerprint: str
    partition_fingerprint: str

    def verify_unchanged(self) -> None:
        self.evaluation_dataset.verify_unchanged()
        body = {
            "schema_version": OOF_HOLDOUT_INPUTS_SCHEMA,
            "fold_id": self.fold_id,
            "closure_fingerprint": self.closure_fingerprint,
            "terminal_seal_fingerprint": self.terminal_seal_fingerprint,
            "source_binding_fingerprint": self.source_binding_fingerprint,
            "sample_ids": list(self.evaluation_dataset.sample_ids),
            "evaluation_dataset_fingerprint": (
                self.evaluation_dataset.dataset_fingerprint
            ),
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }
        if stable_fingerprint(body) != self.partition_fingerprint:
            raise RuntimeError("OOF holdout restricted inputs changed")


def build_oof_restricted_train_inputs(
    *,
    source_binding: CoverageStateRealDRSourceBinding,
    protocol: CoverageStateObservabilityProtocol,
    geometry_protocol: GeometryCatalogProtocol,
    preprocess: PreprocessConfig,
    fold_closure: VerifiedOOFFoldClosure,
) -> OOFRestrictedTrainInputs:
    """Open only the train allowlist and build its isolated scalar cache."""

    closure = require_verified_oof_fold_closure(fold_closure)
    if type(geometry_protocol) is not GeometryCatalogProtocol:
        raise TypeError("geometry_protocol has a wrong type")
    manifest, bundle = _partition_bundle(
        source_binding=source_binding,
        protocol=protocol,
        preprocess=preprocess,
        sample_ids=closure.train_sample_ids,
    )
    sources = tuple(
        CachedTrainingSource(
            row.sample_id,
            row.base_output.feature,
            row.base_output.probability,
            row.state,
        )
        for row in bundle.rows
    )
    legacy = prepare_training_catalog(
        sources,
        occupancy_config=bundle.occupancy_config,
        match_config=bundle.match_config,
        intervention_config=bundle.intervention_config,
    )
    geometry = build_geometry_safe_catalog(
        bundle,
        legacy,
        manifest,
        geometry_protocol,
    )
    _audit_geometry_subset_against_frozen_receipt(
        geometry,
        source_binding.geometry_receipt_path,
        sample_ids=closure.train_sample_ids,
    )
    raw = build_coverage_state_raw_catalog(bundle, manifest, geometry)
    observability = audit_population_observability(raw)
    if (
        observability.decision
        is not CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF
        or not observability.scalar_authorized
        or observability.pp_authorized
    ):
        raise PermissionError("OOF train fold is not scalar-authorized")
    scalar = prepare_scalar_coverage_state_cache(
        raw,
        observability,
        CoverageStateSobolevConfig(
            truncation_radius=raw.feature_stride
        ),
    )
    evaluation = _evaluation_dataset(bundle, closure, partition="train")
    body = {
        "schema_version": OOF_TRAIN_INPUTS_SCHEMA,
        "fold_id": closure.fold_id,
        "closure_fingerprint": closure.closure_fingerprint,
        "source_binding_fingerprint": source_binding.binding_fingerprint,
        "sample_ids": list(evaluation.sample_ids),
        "scalar_cache_fingerprint": scalar.cache_fingerprint,
        "evaluation_dataset_fingerprint": evaluation.dataset_fingerprint,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    result = OOFRestrictedTrainInputs(
        fold_id=closure.fold_id,
        closure_fingerprint=closure.closure_fingerprint,
        scalar_cache=scalar,
        evaluation_dataset=evaluation,
        source_binding_fingerprint=source_binding.binding_fingerprint,
        partition_fingerprint=stable_fingerprint(body),
    )
    result.verify_unchanged()
    return result


def build_oof_restricted_holdout_inputs(
    *,
    source_binding: CoverageStateRealDRSourceBinding,
    protocol: CoverageStateObservabilityProtocol,
    preprocess: PreprocessConfig,
    fold_closure: VerifiedOOFFoldClosure,
    terminal_seal: VerifiedOOFTerminalSeal,
) -> OOFRestrictedHoldoutInputs:
    """Open holdout payloads only after the exact completed-400 seal."""

    closure = require_verified_oof_fold_closure(fold_closure)
    seal = require_verified_oof_terminal_seal(terminal_seal)
    if (
        seal.fold_id != closure.fold_id
        or seal.closure_fingerprint != closure.closure_fingerprint
    ):
        raise PermissionError("terminal seal belongs to another fold")
    _, bundle = _partition_bundle(
        source_binding=source_binding,
        protocol=protocol,
        preprocess=preprocess,
        sample_ids=closure.held_out_sample_ids,
    )
    evaluation = _evaluation_dataset(bundle, closure, partition="holdout")
    body = {
        "schema_version": OOF_HOLDOUT_INPUTS_SCHEMA,
        "fold_id": closure.fold_id,
        "closure_fingerprint": closure.closure_fingerprint,
        "terminal_seal_fingerprint": seal.seal_fingerprint,
        "source_binding_fingerprint": source_binding.binding_fingerprint,
        "sample_ids": list(evaluation.sample_ids),
        "evaluation_dataset_fingerprint": evaluation.dataset_fingerprint,
        "D_V_payload_accessed": False,
        "D_T_payload_accessed": False,
    }
    result = OOFRestrictedHoldoutInputs(
        fold_id=closure.fold_id,
        closure_fingerprint=closure.closure_fingerprint,
        terminal_seal_fingerprint=seal.seal_fingerprint,
        evaluation_dataset=evaluation,
        source_binding_fingerprint=source_binding.binding_fingerprint,
        partition_fingerprint=stable_fingerprint(body),
    )
    result.verify_unchanged()
    return result


__all__ = [
    "OOFRestrictedHoldoutInputs",
    "OOFRestrictedTrainInputs",
    "OOF_HOLDOUT_INPUTS_SCHEMA",
    "OOF_TRAIN_INPUTS_SCHEMA",
    "build_oof_restricted_holdout_inputs",
    "build_oof_restricted_train_inputs",
]
