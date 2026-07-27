"""Strict, reusable construction of the real ``D_R`` scalar CSLF inputs.

This module turns the create-only construction path used by
``tools/audit_coverage_state_observability.py`` into a package API.  It
deliberately has no training, calibration, inference, ``D_V``, or ``D_T``
entry point.  Every source file is bound to the frozen
``coverage_state_observability_v1`` protocol before any cached tensor is
loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Mapping

from ..cache.schema import file_sha256, stable_fingerprint
from ..coverage_state_observability import (
    CoverageStateObservabilityDecision,
    CoverageStatePopulationObservabilityReceipt,
    audit_population_observability,
)
from ..coverage_state_precomputed_cache import (
    CoverageStateScalarCache,
    prepare_scalar_coverage_state_cache,
)
from ..coverage_state_raw_catalog import CoverageStateRawCatalog
from ..coverage_state_sobolev import CoverageStateSobolevConfig
from ..data import ManifestImageDataset, PreprocessConfig
from ..splits import SplitManifest, load_and_validate_manifest
from .cache_pipeline import LoadedDRCacheBundle, load_d_r_cache_bundle
from .coverage_state_observability_protocol import (
    CoverageStateObservabilityProtocol,
    load_coverage_state_observability_protocol,
)
from .coverage_state_raw_catalog import build_coverage_state_raw_catalog
from .geometry_catalog_protocol import (
    GeometryCatalogProtocol,
    load_geometry_catalog_protocol,
)
from .geometry_safe_catalog import (
    GeometrySafeCatalog,
    build_geometry_safe_catalog,
)
from .training_pipeline import (
    CachedTrainingSource,
    prepare_training_catalog,
)


COVERAGE_STATE_REAL_DR_SOURCE_BINDING_SCHEMA = (
    "cure-lite-coverage-state-real-dr-source-binding-v1"
)
COVERAGE_STATE_REAL_DR_INPUTS_SCHEMA = (
    "cure-lite-coverage-state-real-dr-inputs-v1"
)
COVERAGE_STATE_OBSERVABILITY_CONFIG_FILE_SHA256 = (
    "60d42e657f1daed3cb01c7ee93c8f3fe17417542931d853756ccbbeda1f95713"
)


def _canonical_regular_file(path: str | Path, *, name: str) -> Path:
    candidate = Path(path).expanduser()
    absolute = Path(os.path.abspath(candidate))
    if candidate.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if (
        resolved != absolute
        or not resolved.is_file()
        or resolved.is_symlink()
    ):
        raise ValueError(f"{name} must be a canonical regular file")
    return resolved


def _strict_json(path: Path, *, name: str) -> dict[str, object]:
    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite value {item}")
            ),
        )
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain a JSON object")
    return dict(value)


def _fingerprinted(payload: Mapping[str, object]) -> dict[str, object]:
    result = dict(payload)
    if "receipt_fingerprint" in result:
        raise ValueError("payload already contains receipt_fingerprint")
    result["receipt_fingerprint"] = stable_fingerprint(result)
    return result


def _require_equal(
    actual: object,
    expected: object,
    *,
    name: str,
) -> None:
    if actual != expected:
        raise RuntimeError(f"{name} differs from frozen input binding")


@dataclass(frozen=True)
class CoverageStateRealDRSourceBinding:
    """Canonical files and identities authorized for one real ``D_R`` build."""

    manifest_path: Path
    state_index_path: Path
    geometry_config_path: Path
    geometry_receipt_path: Path
    observability_config_path: Path
    manifest_file_sha256: str
    state_index_file_sha256: str
    geometry_config_file_sha256: str
    geometry_receipt_file_sha256: str
    observability_config_file_sha256: str
    observability_config_fingerprint: str
    geometry_protocol_config_fingerprint: str
    geometry_catalog_fingerprint: str
    state_index_fingerprint: str
    base_fingerprint: str
    base_state_fingerprint: str
    state_fingerprint: str
    gt_fingerprint: str
    dataset: str
    split: str
    binding_fingerprint: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": COVERAGE_STATE_REAL_DR_SOURCE_BINDING_SCHEMA,
            "dataset": self.dataset,
            "split": self.split,
            "runtime_splits": ["D_R"],
            "paths": {
                "manifest": str(self.manifest_path),
                "state_index": str(self.state_index_path),
                "geometry_config": str(self.geometry_config_path),
                "geometry_receipt": str(self.geometry_receipt_path),
                "observability_config": str(
                    self.observability_config_path
                ),
            },
            "file_sha256": {
                "manifest": self.manifest_file_sha256,
                "state_index": self.state_index_file_sha256,
                "geometry_config": self.geometry_config_file_sha256,
                "geometry_receipt": self.geometry_receipt_file_sha256,
                "observability_config": (
                    self.observability_config_file_sha256
                ),
            },
            "fingerprints": {
                "observability_config": (
                    self.observability_config_fingerprint
                ),
                "geometry_protocol_config": (
                    self.geometry_protocol_config_fingerprint
                ),
                "geometry_catalog": self.geometry_catalog_fingerprint,
                "state_index": self.state_index_fingerprint,
                "base": self.base_fingerprint,
                "base_state": self.base_state_fingerprint,
                "state": self.state_fingerprint,
                "gt": self.gt_fingerprint,
            },
            "execution_policy": {
                "create_only": True,
                "training_performed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            },
        }

    def verify_unchanged(self) -> None:
        """Recheck every source file and every frozen semantic fingerprint."""

        for path, expected, name in (
            (
                self.manifest_path,
                self.manifest_file_sha256,
                "manifest",
            ),
            (
                self.state_index_path,
                self.state_index_file_sha256,
                "D_R state index",
            ),
            (
                self.geometry_config_path,
                self.geometry_config_file_sha256,
                "geometry config",
            ),
            (
                self.geometry_receipt_path,
                self.geometry_receipt_file_sha256,
                "geometry receipt",
            ),
            (
                self.observability_config_path,
                self.observability_config_file_sha256,
                "observability config",
            ),
        ):
            current = _canonical_regular_file(path, name=name)
            if current != path or file_sha256(current) != expected:
                raise RuntimeError(f"bound {name} file changed")
        protocol = load_coverage_state_observability_protocol(
            self.observability_config_path
        )
        geometry_protocol = load_geometry_catalog_protocol(
            self.geometry_config_path
        )
        state_index = _strict_json(
            self.state_index_path,
            name="D_R state index",
        )
        geometry_receipt = _strict_json(
            self.geometry_receipt_path,
            name="geometry receipt",
        )
        checks = (
            (
                protocol.fingerprint,
                self.observability_config_fingerprint,
                "observability config fingerprint",
            ),
            (
                geometry_protocol.fingerprint,
                self.geometry_protocol_config_fingerprint,
                "geometry protocol fingerprint",
            ),
            (
                geometry_receipt.get("receipt_fingerprint"),
                self.geometry_catalog_fingerprint,
                "geometry catalog fingerprint",
            ),
            (
                state_index.get("index_fingerprint"),
                self.state_index_fingerprint,
                "D_R state index fingerprint",
            ),
            (
                state_index.get("base_fingerprint"),
                self.base_fingerprint,
                "base fingerprint",
            ),
            (
                state_index.get("base_state_fingerprint"),
                self.base_state_fingerprint,
                "base state fingerprint",
            ),
            (
                state_index.get("state_fingerprint"),
                self.state_fingerprint,
                "state fingerprint",
            ),
            (
                state_index.get("gt_fingerprint"),
                self.gt_fingerprint,
                "GT fingerprint",
            ),
            (protocol.dataset, self.dataset, "dataset"),
            (protocol.split, "D_R", "protocol split"),
            (state_index.get("split"), "D_R", "state index split"),
        )
        for actual, expected, name in checks:
            if actual != expected:
                raise RuntimeError(f"bound {name} changed")
        if stable_fingerprint(self.canonical_payload()) != (
            self.binding_fingerprint
        ):
            raise RuntimeError("real D_R source binding changed in memory")


def bind_coverage_state_real_dr_sources(
    *,
    manifest_path: str | Path,
    state_index_path: str | Path,
    geometry_config_path: str | Path,
    geometry_receipt_path: str | Path,
    observability_config_path: str | Path,
) -> tuple[
    CoverageStateRealDRSourceBinding,
    CoverageStateObservabilityProtocol,
    GeometryCatalogProtocol,
    PreprocessConfig,
]:
    """Validate the five frozen source files without loading cached tensors."""

    manifest = _canonical_regular_file(manifest_path, name="manifest")
    state_index_file = _canonical_regular_file(
        state_index_path,
        name="D_R state index",
    )
    geometry_config = _canonical_regular_file(
        geometry_config_path,
        name="geometry config",
    )
    geometry_receipt = _canonical_regular_file(
        geometry_receipt_path,
        name="geometry receipt",
    )
    observability_config = _canonical_regular_file(
        observability_config_path,
        name="observability config",
    )
    config_sha256 = file_sha256(observability_config)
    if config_sha256 != COVERAGE_STATE_OBSERVABILITY_CONFIG_FILE_SHA256:
        raise RuntimeError("observability config is not the frozen v1 file")
    protocol = load_coverage_state_observability_protocol(
        observability_config
    )
    if protocol.split != "D_R":
        raise RuntimeError("real CSLF input construction permits only D_R")
    input_binding = protocol.input_binding
    actual_files = {
        "manifest_file_sha256": file_sha256(manifest),
        "state_index_sha256": file_sha256(state_index_file),
        "geometry_protocol_config_file_sha256": file_sha256(
            geometry_config
        ),
        "geometry_catalog_receipt_file_sha256": file_sha256(
            geometry_receipt
        ),
    }
    for name, actual in actual_files.items():
        _require_equal(
            actual,
            getattr(input_binding, name),
            name=name,
        )
    geometry_protocol = load_geometry_catalog_protocol(geometry_config)
    _require_equal(
        geometry_protocol.fingerprint,
        input_binding.geometry_protocol_config_fingerprint,
        name="geometry protocol fingerprint",
    )
    state_index = _strict_json(
        state_index_file,
        name="D_R state index",
    )
    expected_state = {
        "index_fingerprint": input_binding.state_index_fingerprint,
        "base_fingerprint": input_binding.base_fingerprint,
        "base_state_fingerprint": input_binding.base_state_fingerprint,
        "state_fingerprint": input_binding.state_fingerprint,
        "gt_fingerprint": input_binding.gt_fingerprint,
        "dataset": protocol.dataset,
        "split": "D_R",
    }
    for name, expected in expected_state.items():
        _require_equal(
            state_index.get(name),
            expected,
            name=f"D_R state index {name}",
        )
    receipt = _strict_json(
        geometry_receipt,
        name="geometry receipt",
    )
    _require_equal(
        receipt.get("receipt_fingerprint"),
        input_binding.geometry_catalog_fingerprint,
        name="geometry catalog receipt fingerprint",
    )
    preprocess = PreprocessConfig.from_fingerprint_payload(
        state_index.get("preprocessing")
    )
    kwargs = {
        "manifest_path": manifest,
        "state_index_path": state_index_file,
        "geometry_config_path": geometry_config,
        "geometry_receipt_path": geometry_receipt,
        "observability_config_path": observability_config,
        "manifest_file_sha256": actual_files["manifest_file_sha256"],
        "state_index_file_sha256": actual_files["state_index_sha256"],
        "geometry_config_file_sha256": actual_files[
            "geometry_protocol_config_file_sha256"
        ],
        "geometry_receipt_file_sha256": actual_files[
            "geometry_catalog_receipt_file_sha256"
        ],
        "observability_config_file_sha256": config_sha256,
        "observability_config_fingerprint": protocol.fingerprint,
        "geometry_protocol_config_fingerprint": (
            geometry_protocol.fingerprint
        ),
        "geometry_catalog_fingerprint": (
            input_binding.geometry_catalog_fingerprint
        ),
        "state_index_fingerprint": (
            input_binding.state_index_fingerprint
        ),
        "base_fingerprint": input_binding.base_fingerprint,
        "base_state_fingerprint": input_binding.base_state_fingerprint,
        "state_fingerprint": input_binding.state_fingerprint,
        "gt_fingerprint": input_binding.gt_fingerprint,
        "dataset": protocol.dataset,
        "split": "D_R",
    }
    provisional = CoverageStateRealDRSourceBinding(
        **kwargs,
        binding_fingerprint="0" * 64,
    )
    binding = CoverageStateRealDRSourceBinding(
        **kwargs,
        binding_fingerprint=stable_fingerprint(
            provisional.canonical_payload()
        ),
    )
    binding.verify_unchanged()
    return binding, protocol, geometry_protocol, preprocess


@dataclass(frozen=True, eq=False)
class CoverageStateRealDRInputs:
    """Complete create-only real input graph for scalar CSLF training design."""

    source_binding: CoverageStateRealDRSourceBinding
    manifest: SplitManifest
    bundle: LoadedDRCacheBundle
    geometry_catalog: GeometrySafeCatalog
    raw_catalog: CoverageStateRawCatalog
    observability: CoverageStatePopulationObservabilityReceipt
    scalar_cache: CoverageStateScalarCache
    geometry_receipt_fingerprint: str
    raw_catalog_fingerprint: str
    observability_receipt_fingerprint: str
    scalar_cache_fingerprint: str
    sobolev_config_fingerprint: str
    build_fingerprint: str

    def canonical_payload(self) -> dict[str, object]:
        cache_counts = self.scalar_cache.canonical_payload()["counts"]
        return {
            "schema_version": COVERAGE_STATE_REAL_DR_INPUTS_SCHEMA,
            "dataset": self.source_binding.dataset,
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "source_binding_fingerprint": (
                self.source_binding.binding_fingerprint
            ),
            "fingerprints": {
                "manifest": self.manifest.fingerprint,
                "D_R_bundle_state_index": (
                    self.bundle.state_index_fingerprint
                ),
                "D_R_bundle_state": self.bundle.state_fingerprint,
                "D_R_bundle_gt": self.bundle.gt_fingerprint,
                "legacy_analysis_population": (
                    self.geometry_catalog.source_catalog_fingerprint
                ),
                "geometry_receipt": self.geometry_receipt_fingerprint,
                "raw_catalog": self.raw_catalog_fingerprint,
                "observability": (
                    self.observability_receipt_fingerprint
                ),
                "sobolev_config": self.sobolev_config_fingerprint,
                "scalar_cache": self.scalar_cache_fingerprint,
            },
            "counts": {
                "D_R_rows": len(self.bundle.rows),
                "natural_records": len(
                    self.raw_catalog.natural_records
                ),
                "pair_records": len(self.raw_catalog.pair_records),
                "raw_exclusions": len(self.raw_catalog.exclusions),
                "clean_positive_optimization_eligible": cache_counts[
                    "clean_positive_optimization_eligible"
                ],
                "component_null_optimization_eligible": cache_counts[
                    "component_null_optimization_eligible"
                ],
                "component_null_diagnostic_only": cache_counts[
                    "component_null_diagnostic_only"
                ],
                "identity_null_diagnostic": cache_counts[
                    "identity_null_diagnostic"
                ],
            },
            "representation": "scalar_max",
            "observability_decision": self.observability.decision.value,
            "sobolev_truncation_policy": "equal_feature_stride",
            "feature_stride": self.raw_catalog.feature_stride,
            "execution_policy": {
                "create_only": True,
                "training_performed": False,
                "calibration_performed": False,
                "inference_performed": False,
                "D_V_accessed": False,
                "D_T_accessed": False,
            },
        }

    @property
    def current_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify_unchanged(self) -> None:
        self.source_binding.verify_unchanged()
        if (
            self.source_binding.split != "D_R"
            or self.bundle.split != "D_R"
            or self.raw_catalog.split != "D_R"
            or self.scalar_cache.raw_catalog.split != "D_R"
        ):
            raise RuntimeError("real scalar input graph left exact D_R")
        if self.manifest.dataset != self.source_binding.dataset:
            raise RuntimeError("manifest dataset changed")
        if self.manifest.fingerprint != (
            self.bundle.split_manifest_fingerprint
        ):
            raise RuntimeError("manifest and D_R bundle changed")
        self.bundle.verify_unchanged()
        reconstructed_geometry = _fingerprinted(
            self.geometry_catalog.canonical_payload()
        )
        upstream_geometry = _strict_json(
            self.source_binding.geometry_receipt_path,
            name="geometry receipt",
        )
        if (
            reconstructed_geometry != upstream_geometry
            or reconstructed_geometry["receipt_fingerprint"]
            != self.geometry_receipt_fingerprint
        ):
            raise RuntimeError("reconstructed geometry receipt changed")
        if self.raw_catalog.catalog_fingerprint != (
            self.raw_catalog_fingerprint
        ):
            raise RuntimeError("raw catalog changed")
        if (
            self.observability.raw_catalog_fingerprint
            != self.raw_catalog_fingerprint
            or self.observability.receipt_fingerprint
            != self.observability_receipt_fingerprint
            or self.observability.decision
            is not CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF
            or not self.observability.scalar_authorized
            or self.observability.pp_authorized
        ):
            raise RuntimeError("scalar observability authorization changed")
        if self.scalar_cache.raw_catalog is not self.raw_catalog:
            raise RuntimeError("scalar cache raw-catalog identity changed")
        if self.scalar_cache.observability is not self.observability:
            raise RuntimeError("scalar cache observability identity changed")
        if self.scalar_cache.sobolev_config.truncation_radius != (
            self.raw_catalog.feature_stride
        ):
            raise RuntimeError("Sobolev truncation no longer equals stride")
        self.scalar_cache.verify_unchanged()
        if (
            stable_fingerprint(self.scalar_cache.canonical_payload())
            != self.scalar_cache_fingerprint
            or self.scalar_cache.sobolev_config_fingerprint
            != self.sobolev_config_fingerprint
            or self.current_fingerprint != self.build_fingerprint
        ):
            raise RuntimeError("real scalar input graph changed")


def build_coverage_state_real_dr_inputs(
    *,
    manifest_path: str | Path,
    state_index_path: str | Path,
    geometry_config_path: str | Path,
    geometry_receipt_path: str | Path,
    observability_config_path: str | Path,
) -> CoverageStateRealDRInputs:
    """Build the frozen real scalar cache without any model execution."""

    (
        source_binding,
        protocol,
        geometry_protocol,
        preprocess,
    ) = bind_coverage_state_real_dr_sources(
        manifest_path=manifest_path,
        state_index_path=state_index_path,
        geometry_config_path=geometry_config_path,
        geometry_receipt_path=geometry_receipt_path,
        observability_config_path=observability_config_path,
    )
    manifest = load_and_validate_manifest(source_binding.manifest_path)
    if manifest.dataset != protocol.dataset:
        raise RuntimeError("manifest dataset differs from frozen protocol")
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
    reconstructed_geometry = _fingerprinted(
        geometry.canonical_payload()
    )
    upstream_geometry = _strict_json(
        source_binding.geometry_receipt_path,
        name="geometry receipt",
    )
    if reconstructed_geometry != upstream_geometry:
        raise RuntimeError(
            "reconstructed geometry catalog differs from frozen receipt"
        )
    raw = build_coverage_state_raw_catalog(bundle, manifest, geometry)
    observability = audit_population_observability(raw)
    if (
        observability.decision
        is not CoverageStateObservabilityDecision.AUTHORIZE_SCALAR_CSLF
        or not observability.scalar_authorized
        or observability.pp_authorized
    ):
        raise PermissionError(
            "real scalar cache requires AUTHORIZE_SCALAR_CSLF"
        )
    sobolev = CoverageStateSobolevConfig(
        truncation_radius=raw.feature_stride
    )
    scalar = prepare_scalar_coverage_state_cache(
        raw,
        observability,
        sobolev,
    )
    kwargs = {
        "source_binding": source_binding,
        "manifest": manifest,
        "bundle": bundle,
        "geometry_catalog": geometry,
        "raw_catalog": raw,
        "observability": observability,
        "scalar_cache": scalar,
        "geometry_receipt_fingerprint": geometry.catalog_fingerprint,
        "raw_catalog_fingerprint": raw.catalog_fingerprint,
        "observability_receipt_fingerprint": (
            observability.receipt_fingerprint
        ),
        "scalar_cache_fingerprint": scalar.cache_fingerprint,
        "sobolev_config_fingerprint": (
            scalar.sobolev_config_fingerprint
        ),
    }
    provisional = CoverageStateRealDRInputs(
        **kwargs,
        build_fingerprint="0" * 64,
    )
    result = CoverageStateRealDRInputs(
        **kwargs,
        build_fingerprint=stable_fingerprint(
            provisional.canonical_payload()
        ),
    )
    result.verify_unchanged()
    return result


__all__ = [
    "COVERAGE_STATE_OBSERVABILITY_CONFIG_FILE_SHA256",
    "COVERAGE_STATE_REAL_DR_INPUTS_SCHEMA",
    "COVERAGE_STATE_REAL_DR_SOURCE_BINDING_SCHEMA",
    "CoverageStateRealDRInputs",
    "CoverageStateRealDRSourceBinding",
    "bind_coverage_state_real_dr_sources",
    "build_coverage_state_real_dr_inputs",
]
