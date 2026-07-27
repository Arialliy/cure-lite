"""Detector-independent real-cache adapter for the PFCR decoder.

This module does not import or execute a detector.  It accepts only a
strictly loaded Stage-A bundle containing detached ``(p_b, F_b)`` tensors and
the corresponding deterministic state cache.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .cache.schema import stable_fingerprint
from .experiment.cache_pipeline import LoadedDRCacheBundle
from .experiment.training_pipeline import (
    CachedTrainingSource,
    PreparedTrainingCatalog,
    prepare_training_catalog,
)
from .phase_resolved_relation_decoder import (
    PhaseResolvedRelationDecoderConfig,
)
from .splits import load_and_validate_manifest


PFCR_REAL_CACHE_ADAPTER_SCHEMA = (
    "cure-lite-pfcr-real-cache-adapter-v1"
)


@dataclass(frozen=True)
class PFCRRealCacheContract:
    """Immutable shape and provenance contract inferred from a D_R cache."""

    dataset: str
    sample_count: int
    feature_channels: int
    feature_shape: tuple[int, int]
    output_shape: tuple[int, int]
    feature_stride: int
    occupancy_threshold: float
    split_manifest_fingerprint: str
    preprocessing_fingerprint: str
    base_fingerprint: str
    base_state_fingerprint: str
    state_fingerprint: str
    gt_fingerprint: str
    base_index_fingerprint: str
    state_index_fingerprint: str
    ordered_sample_ids: tuple[str, ...]
    ordered_row_fingerprints: tuple[str, ...]
    contract_fingerprint: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": PFCR_REAL_CACHE_ADAPTER_SCHEMA,
            "dataset": self.dataset,
            "split": "D_R",
            "sample_count": self.sample_count,
            "feature_channels": self.feature_channels,
            "feature_shape": list(self.feature_shape),
            "output_shape": list(self.output_shape),
            "feature_stride": self.feature_stride,
            "occupancy_threshold": self.occupancy_threshold,
            "split_manifest_fingerprint": (
                self.split_manifest_fingerprint
            ),
            "preprocessing_fingerprint": (
                self.preprocessing_fingerprint
            ),
            "base_fingerprint": self.base_fingerprint,
            "base_state_fingerprint": self.base_state_fingerprint,
            "state_fingerprint": self.state_fingerprint,
            "gt_fingerprint": self.gt_fingerprint,
            "base_index_fingerprint": self.base_index_fingerprint,
            "state_index_fingerprint": self.state_index_fingerprint,
            "ordered_sample_ids": list(self.ordered_sample_ids),
            "ordered_row_fingerprints": list(
                self.ordered_row_fingerprints
            ),
            "detector_code_executed": False,
            "base_forward_executed": False,
            "cache_only_inputs": ["p_b", "F_b", "state"],
        }

    def decoder_config(
        self,
        *,
        relation_dim: int = 8,
    ) -> PhaseResolvedRelationDecoderConfig:
        """Create a decoder config from cache-derived dimensions."""

        return PhaseResolvedRelationDecoderConfig(
            feature_channels=self.feature_channels,
            feature_stride=self.feature_stride,
            relation_dim=relation_dim,
        )


@dataclass(frozen=True, eq=False)
class PFCRRealCacheAdapter:
    """Verified zero-copy bridge from Stage-A cache to PFCR training."""

    bundle: LoadedDRCacheBundle
    contract: PFCRRealCacheContract
    sources: tuple[CachedTrainingSource, ...]
    prepared_catalog: PreparedTrainingCatalog

    def verify_unchanged(self) -> None:
        self.bundle.verify_unchanged()
        if self.prepared_catalog.sources != self.sources:
            raise RuntimeError(
                "PFCR prepared catalog no longer binds the cache sources"
            )
        for row, source in zip(
            self.bundle.rows,
            self.sources,
            strict=True,
        ):
            if (
                source.sample_id != row.sample_id
                or source.feature is not row.base_output.feature
                or source.probability is not row.base_output.probability
            ):
                raise RuntimeError(
                    "PFCR cache adapter reconstructed or replaced a "
                    "bound feature/probability object"
                )
            for name in (
                "occupancy",
                "pred_labels",
                "gt_labels",
                "base_match_pairs",
                "real_miss_ids",
                "reachable_miss_ids",
                "legal_pairs",
                "image_valid_mask",
            ):
                if not torch.equal(
                    getattr(source.state, name),
                    getattr(row.state, name),
                ):
                    raise RuntimeError(
                        "PFCR normalized state differs from its bound cache"
                    )


def _infer_common_grid(
    bundle: LoadedDRCacheBundle,
) -> tuple[int, tuple[int, int], tuple[int, int], int]:
    first = bundle.rows[0]
    feature = first.base_output.feature
    probability = first.base_output.probability
    feature_channels = int(feature.shape[1])
    feature_shape = tuple(int(value) for value in feature.shape[-2:])
    output_shape = tuple(
        int(value) for value in probability.shape[-2:]
    )
    if (
        output_shape[0] % feature_shape[0]
        or output_shape[1] % feature_shape[1]
    ):
        raise ValueError(
            "PFCR requires an exact integer feature-to-output stride"
        )
    stride_h = output_shape[0] // feature_shape[0]
    stride_w = output_shape[1] // feature_shape[1]
    if stride_h != stride_w or stride_h < 1:
        raise ValueError(
            "PFCR requires one equal positive spatial stride"
        )

    expected_feature = (1, feature_channels, *feature_shape)
    expected_probability = (1, 1, *output_shape)
    for row in bundle.rows:
        current_feature = row.base_output.feature
        current_probability = row.base_output.probability
        if tuple(current_feature.shape) != expected_feature:
            raise ValueError(
                "all PFCR D_R cached features must share one shape"
            )
        if tuple(current_probability.shape) != expected_probability:
            raise ValueError(
                "all PFCR D_R probabilities must share one output grid"
            )
        if (
            current_feature.dtype != torch.float32
            or current_feature.device.type != "cpu"
            or current_feature.requires_grad
        ):
            raise TypeError(
                "PFCR requires detached CPU FP32 cached features"
            )
        if (
            current_probability.dtype != torch.float32
            or current_probability.device.type != "cpu"
            or current_probability.requires_grad
        ):
            raise TypeError(
                "PFCR requires detached CPU FP32 cached probabilities"
            )
        expected_occupancy = (
            current_probability[0, 0]
            >= bundle.occupancy_config.threshold
        )
        if not torch.equal(expected_occupancy, row.state.occupancy):
            raise ValueError(
                "cached PFCR occupancy differs from frozen p_b threshold"
            )
        if tuple(row.state.image_valid_mask.shape) != output_shape:
            raise ValueError(
                "PFCR image-valid mask and output grid differ"
            )
    return feature_channels, feature_shape, output_shape, stride_h


def adapt_pfcr_d_r_cache(
    bundle: LoadedDRCacheBundle,
) -> PFCRRealCacheAdapter:
    """Validate and prepare a real D_R cache exactly once.

    Component extraction, matching and legal-state preparation occur here,
    before any optimizer loop.  The cached feature/probability objects are
    reused rather than copied or recomputed.
    """

    if not isinstance(bundle, LoadedDRCacheBundle):
        raise TypeError(
            "bundle must be a strictly loaded LoadedDRCacheBundle"
        )
    bundle.verify_unchanged()
    manifest = load_and_validate_manifest(bundle.manifest_path)
    if manifest.fingerprint != bundle.split_manifest_fingerprint:
        raise RuntimeError("D_R manifest identity changed")
    expected_ids = tuple(
        sorted(
            record.sample_id
            for record in manifest.records_for("D_R")
        )
    )
    actual_ids = tuple(row.sample_id for row in bundle.rows)
    if actual_ids != expected_ids:
        raise RuntimeError("D_R cache is not the exact manifest split")

    channels, feature_shape, output_shape, stride = (
        _infer_common_grid(bundle)
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
    prepared = prepare_training_catalog(
        sources,
        occupancy_config=bundle.occupancy_config,
        match_config=bundle.match_config,
        intervention_config=bundle.intervention_config,
    )
    payload = {
        "schema_version": PFCR_REAL_CACHE_ADAPTER_SCHEMA,
        "dataset": manifest.dataset,
        "split": "D_R",
        "sample_count": len(bundle.rows),
        "feature_channels": channels,
        "feature_shape": list(feature_shape),
        "output_shape": list(output_shape),
        "feature_stride": stride,
        "occupancy_threshold": bundle.occupancy_config.threshold,
        "split_manifest_fingerprint": (
            bundle.split_manifest_fingerprint
        ),
        "preprocessing_fingerprint": (
            bundle.preprocessing_fingerprint
        ),
        "base_fingerprint": bundle.base_fingerprint,
        "base_state_fingerprint": bundle.base_state_fingerprint,
        "state_fingerprint": bundle.state_fingerprint,
        "gt_fingerprint": bundle.gt_fingerprint,
        "base_index_fingerprint": bundle.base_index_fingerprint,
        "state_index_fingerprint": bundle.state_index_fingerprint,
        "ordered_sample_ids": list(actual_ids),
        "ordered_row_fingerprints": [
            row.content_fingerprint for row in bundle.rows
        ],
        "detector_code_executed": False,
        "base_forward_executed": False,
        "cache_only_inputs": ["p_b", "F_b", "state"],
    }
    contract = PFCRRealCacheContract(
        dataset=manifest.dataset,
        sample_count=len(bundle.rows),
        feature_channels=channels,
        feature_shape=feature_shape,
        output_shape=output_shape,
        feature_stride=stride,
        occupancy_threshold=bundle.occupancy_config.threshold,
        split_manifest_fingerprint=(
            bundle.split_manifest_fingerprint
        ),
        preprocessing_fingerprint=(
            bundle.preprocessing_fingerprint
        ),
        base_fingerprint=bundle.base_fingerprint,
        base_state_fingerprint=bundle.base_state_fingerprint,
        state_fingerprint=bundle.state_fingerprint,
        gt_fingerprint=bundle.gt_fingerprint,
        base_index_fingerprint=bundle.base_index_fingerprint,
        state_index_fingerprint=bundle.state_index_fingerprint,
        ordered_sample_ids=actual_ids,
        ordered_row_fingerprints=tuple(
            row.content_fingerprint for row in bundle.rows
        ),
        contract_fingerprint=stable_fingerprint(payload),
    )
    if contract.canonical_payload() != payload:
        raise AssertionError("PFCR cache contract payload drifted")
    result = PFCRRealCacheAdapter(
        bundle=bundle,
        contract=contract,
        sources=sources,
        prepared_catalog=prepared,
    )
    result.verify_unchanged()
    return result


__all__ = [
    "PFCR_REAL_CACHE_ADAPTER_SCHEMA",
    "PFCRRealCacheAdapter",
    "PFCRRealCacheContract",
    "adapt_pfcr_d_r_cache",
]
