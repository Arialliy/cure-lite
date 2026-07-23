"""Provenance-bound decoder training for the Gate-2 CURE-Lite pilot.

The lower-level training helpers intentionally remain useful for unit tests and
mechanism studies.  This module is the formal experiment entry point: it only
accepts a fully verified ``D_R`` cache bundle, creates the decoder, loss, and
optimizer internally, and forces Factual-only, exposure-matched Factual-only,
and Uniform-Legal to start from the exact same decoder bytes.

The v0.2 extension deliberately trains only ``miss_aligned_legal`` when an
immutable completed F/Fx/U trio is supplied.  This preserves the paired
initialization contract without repeating three 800-epoch reference runs whose
training semantics did not change.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Callable, Literal

import torch

from ..cache.schema import stable_fingerprint
from ..config import (
    DecoderConfig,
    LossConfig,
    MissAlignmentConfig,
    TrainingConfig,
)
from ..decoder import CURELiteDecoder
from ..losses import CURELiteLoss
from .artifacts import (
    DECODER_ARTIFACT_SCHEMA_V2,
    DECODER_ARTIFACT_SCHEMA_V3,
    DecoderRunConfig,
    LoadedDecoderArtifact,
    _save_decoder_artifact,
    decoder_state_fingerprint,
)
from .cache_pipeline import LoadedDRCacheBundle
from .training_pipeline import (
    CachedTrainingSource,
    FixedEpochTrainingLog,
    FixedTrainingLog,
    PreparedTrainingCatalog,
    TrainingSupportSummary,
    build_epoch_branch_pools_from_catalog,
    prepare_training_catalog,
    require_training_branch_support,
    run_fixed_training,
    summarize_training_support,
)


@dataclass(frozen=True)
class PairedGate2TrainingConfig:
    """Every shared choice for the paired Factual-only/Uniform-Legal run."""

    decoder_config: DecoderConfig
    loss_config: LossConfig = LossConfig()
    training_config: TrainingConfig = TrainingConfig()
    optimizer: Literal["adam", "sgd"] = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 1
    steps_per_epoch: int = 1
    factual_miss_batch: int = 1
    factual_no_miss_batch: int = 1
    synthetic_batch: int = 1
    global_seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.decoder_config, DecoderConfig):
            raise TypeError("decoder_config must be DecoderConfig")
        if not isinstance(self.loss_config, LossConfig):
            raise TypeError("loss_config must be LossConfig")
        if not isinstance(self.training_config, TrainingConfig):
            raise TypeError("training_config must be TrainingConfig")
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError("optimizer must be 'adam' or 'sgd'")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            normalized = float(value)
            if not isfinite(normalized) or normalized < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            if name == "learning_rate" and normalized == 0.0:
                raise ValueError("learning_rate must be positive")
            object.__setattr__(self, name, normalized)
        for name in (
            "epochs",
            "steps_per_epoch",
            "factual_miss_batch",
            "factual_no_miss_batch",
            "synthetic_batch",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.global_seed, bool) or not isinstance(self.global_seed, int):
            raise TypeError("global_seed must be an integer")

    @property
    def branch_batch_sizes(self) -> dict[str, int]:
        return {
            "factual_miss": self.factual_miss_batch,
            "factual_no_miss": self.factual_no_miss_batch,
            "synthetic": self.synthetic_batch,
        }


@dataclass(frozen=True)
class MissAlignedGate2TrainingConfig(PairedGate2TrainingConfig):
    """Frozen v0.2 choices for one M-only extension of completed F/Fx/U."""

    miss_alignment_config: MissAlignmentConfig = MissAlignmentConfig()

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.miss_alignment_config, MissAlignmentConfig):
            raise TypeError("miss_alignment_config must be MissAlignmentConfig")


@dataclass(frozen=True, slots=True)
class _PreparedGate2TrainingSeal:
    bundle: LoadedDRCacheBundle
    sources: tuple[CachedTrainingSource, ...]
    catalog: PreparedTrainingCatalog


@dataclass(frozen=True)
class PreparedGate2Training:
    """One process-local semantic catalog bound to one verified D_R bundle.

    This is an internal orchestration value rather than a persisted artifact.
    Object-identity seals prevent a catalog prepared from one loaded bundle
    from being supplied with another otherwise similar bundle.
    """

    bundle: LoadedDRCacheBundle
    sources: tuple[CachedTrainingSource, ...]
    catalog: PreparedTrainingCatalog
    _verification_token: object

    def _verify_source_seal(self) -> None:
        seal = self._verification_token
        if type(seal) is not _PreparedGate2TrainingSeal:
            raise TypeError(
                "PreparedGate2Training must come from prepare_gate2_training"
            )
        if (
            seal.bundle is not self.bundle
            or seal.sources is not self.sources
            or seal.catalog is not self.catalog
        ):
            raise TypeError("prepared Gate-2 training fields were replaced")

    def __post_init__(self) -> None:
        self._verify_source_seal()
        if not isinstance(self.bundle, LoadedDRCacheBundle):
            raise TypeError("prepared Gate-2 bundle has an invalid type")
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("prepared Gate-2 sources must be a nonempty tuple")
        if any(
            not isinstance(source, CachedTrainingSource) for source in self.sources
        ):
            raise TypeError("prepared Gate-2 sources contain an invalid value")
        if not isinstance(self.catalog, PreparedTrainingCatalog):
            raise TypeError("prepared Gate-2 catalog has an invalid type")
        self._verify_catalog_binding()

    def _verify_catalog_binding(self) -> None:
        self.catalog.require_compatible(
            self.sources,
            occupancy_config=self.bundle.occupancy_config,
            match_config=self.bundle.match_config,
            intervention_config=self.bundle.intervention_config,
        )

    def verify_binding(self) -> None:
        """Check process-local identities/configs without rereading cache files."""

        self._verify_source_seal()
        self._verify_catalog_binding()

    def verify_unchanged(self) -> None:
        """Strictly rehash the bound files and in-memory source tensors."""

        self.verify_binding()
        self.bundle.verify_unchanged()


def prepare_gate2_training(
    bundle: LoadedDRCacheBundle,
) -> PreparedGate2Training:
    """Prepare invariant semantics exactly once for one loaded D_R bundle."""

    if not isinstance(bundle, LoadedDRCacheBundle):
        raise TypeError("bundle must be a LoadedDRCacheBundle")
    bundle.verify_unchanged()
    sources = _training_sources(bundle)
    catalog = prepare_training_catalog(
        sources,
        occupancy_config=bundle.occupancy_config,
        match_config=bundle.match_config,
        intervention_config=bundle.intervention_config,
    )
    seal = _PreparedGate2TrainingSeal(
        bundle=bundle,
        sources=sources,
        catalog=catalog,
    )
    prepared = PreparedGate2Training(
        bundle=bundle,
        sources=sources,
        catalog=catalog,
        _verification_token=seal,
    )
    prepared.verify_unchanged()
    return prepared


def _resolve_prepared_gate2_training(
    bundle: LoadedDRCacheBundle,
    prepared: PreparedGate2Training | None,
) -> PreparedGate2Training:
    if prepared is None:
        return prepare_gate2_training(bundle)
    if not isinstance(prepared, PreparedGate2Training):
        raise TypeError("prepared must be PreparedGate2Training or None")
    prepared.verify_binding()
    if prepared.bundle is not bundle:
        raise ValueError("prepared Gate-2 catalog belongs to another D_R bundle")
    return prepared


@dataclass(frozen=True, slots=True)
class _CompletedRunSeal:
    decoder: CURELiteDecoder
    config: DecoderRunConfig
    training_log: FixedTrainingLog
    final_decoder_fingerprint: str


@dataclass(frozen=True)
class CompletedDecoderRun:
    """One completed, self-consistent formal training result in memory."""

    decoder: CURELiteDecoder
    config: DecoderRunConfig
    training_log: FixedTrainingLog
    final_decoder_fingerprint: str
    _verification_token: object

    def __post_init__(self) -> None:
        self.verify_unchanged()

    def verify_unchanged(self) -> None:
        seal = self._verification_token
        if type(seal) is not _CompletedRunSeal:
            raise TypeError("CompletedDecoderRun must come from the formal trainer")
        if (
            seal.decoder is not self.decoder
            or seal.config is not self.config
            or seal.training_log is not self.training_log
            or seal.final_decoder_fingerprint != self.final_decoder_fingerprint
        ):
            raise TypeError("completed decoder run fields were replaced")
        if not isinstance(self.decoder, CURELiteDecoder):
            raise TypeError("completed run decoder has an invalid type")
        if not isinstance(self.config, DecoderRunConfig):
            raise TypeError("completed run config has an invalid type")
        if not isinstance(self.training_log, FixedTrainingLog):
            raise TypeError("completed run log has an invalid type")
        if self.training_log.variant != self.config.variant:
            raise RuntimeError("training log variant differs from run config")
        if self.training_log.epochs != self.config.trained_epochs:
            raise RuntimeError("training log epochs differ from run config")
        if self.training_log.steps_per_epoch != self.config.steps_per_epoch:
            raise RuntimeError("training log steps differ from run config")
        if decoder_state_fingerprint(self.decoder) != self.final_decoder_fingerprint:
            raise RuntimeError("completed decoder changed after formal training")


@dataclass(frozen=True, slots=True)
class _CompletedPairSeal:
    factual_only: CompletedDecoderRun
    factual_exposure_matched: CompletedDecoderRun
    uniform_legal: CompletedDecoderRun
    initial_decoder_fingerprint: str


@dataclass(frozen=True)
class CompletedPairedGate2Run:
    """The paired F/Fx/U outputs sharing one proven initialization."""

    factual_only: CompletedDecoderRun
    factual_exposure_matched: CompletedDecoderRun
    uniform_legal: CompletedDecoderRun
    initial_decoder_fingerprint: str
    _verification_token: object

    def _verify_source_seal(self) -> None:
        seal = self._verification_token
        if type(seal) is not _CompletedPairSeal:
            raise TypeError(
                "CompletedPairedGate2Run must come from the formal paired trainer"
            )
        if (
            seal.factual_only is not self.factual_only
            or seal.factual_exposure_matched is not self.factual_exposure_matched
            or seal.uniform_legal is not self.uniform_legal
            or seal.initial_decoder_fingerprint
            != self.initial_decoder_fingerprint
        ):
            raise TypeError("completed paired-run fields were replaced")

    def __post_init__(self) -> None:
        self._verify_source_seal()
        if self.factual_only.config.variant != "factual_only":
            raise ValueError("factual_only result has the wrong variant")
        if (
            self.factual_exposure_matched.config.variant
            != "factual_exposure_matched"
        ):
            raise ValueError("factual_exposure_matched result has the wrong variant")
        if self.uniform_legal.config.variant != "uniform_legal":
            raise ValueError("uniform_legal result has the wrong variant")
        if not (
            self.factual_only.config.initial_decoder_fingerprint
            == self.factual_exposure_matched.config.initial_decoder_fingerprint
            == self.uniform_legal.config.initial_decoder_fingerprint
            == self.initial_decoder_fingerprint
        ):
            raise ValueError("paired F/Fx/U runs do not share one initialization")
        common_payloads = []
        for run in (
            self.factual_only,
            self.factual_exposure_matched,
            self.uniform_legal,
        ):
            payload = run.config.canonical_payload()
            payload.pop("variant")
            payload.pop("variant_contract")
            common_payloads.append(payload)
        if any(payload != common_payloads[0] for payload in common_payloads[1:]):
            raise ValueError("paired F/Fx/U run configs differ outside variant")
        for factual_log, exposure_log, uniform_log in zip(
            self.factual_only.training_log.epoch_logs,
            self.factual_exposure_matched.training_log.epoch_logs,
            self.uniform_legal.training_log.epoch_logs,
            strict=True,
        ):
            factual_pools = dict(factual_log.pool_sizes)
            exposure_pools = dict(exposure_log.pool_sizes)
            uniform_pools = dict(uniform_log.pool_sizes)
            for branch in ("factual_miss", "factual_no_miss"):
                if not (
                    factual_pools[branch]
                    == exposure_pools[branch]
                    == uniform_pools[branch]
                ):
                    raise ValueError(
                        f"F/F×/U factual pool size differs for {branch}"
                    )
            factual_metrics = dict(factual_log.metrics)
            exposure_metrics = dict(exposure_log.metrics)
            uniform_metrics = dict(uniform_log.metrics)
            for branch in ("factual_miss", "factual_no_miss"):
                for suffix in (
                    "active",
                    "active_min",
                    "active_max",
                    "states",
                    "states_min",
                    "states_max",
                ):
                    key = f"{branch}/{suffix}"
                    if not (
                        factual_metrics.get(key)
                        == exposure_metrics.get(key)
                        == uniform_metrics.get(key)
                    ):
                        raise ValueError(
                            "F/F×/U factual exposure differs for " f"{key}"
                        )
            for suffix in (
                "active",
                "active_min",
                "active_max",
                "states",
                "states_min",
                "states_max",
            ):
                key = f"synthetic/{suffix}"
                if exposure_metrics.get(key) != uniform_metrics.get(key):
                    raise ValueError(f"F×/U third-slot exposure differs for {key}")
            if exposure_pools["synthetic"] != 0:
                raise ValueError("F× may not contain deletion-synthetic states")
            if uniform_pools["synthetic"] < 1:
                raise ValueError("U must contain deletion-synthetic states")
        self.verify_unchanged()

    def verify_unchanged(self) -> None:
        self._verify_source_seal()
        self.factual_only.verify_unchanged()
        self.factual_exposure_matched.verify_unchanged()
        self.uniform_legal.verify_unchanged()


def _verified_v01_reference_training_fingerprint(
    factual_only: LoadedDecoderArtifact,
    factual_exposure_matched: LoadedDecoderArtifact,
    uniform_legal: LoadedDecoderArtifact,
) -> str:
    """Verify and fingerprint one immutable completed v0.1 F/Fx/U trio."""

    references = (
        factual_only,
        factual_exposure_matched,
        uniform_legal,
    )
    if any(not isinstance(item, LoadedDecoderArtifact) for item in references):
        raise TypeError("F/Fx/U references must be LoadedDecoderArtifact values")
    for item in references:
        item.verify_unchanged()
    expected_variants = (
        "factual_only",
        "factual_exposure_matched",
        "uniform_legal",
    )
    if tuple(item.config.variant for item in references) != expected_variants:
        raise ValueError("reference artifacts must be ordered as F/Fx/U")
    if any(
        item.config.schema_version != DECODER_ARTIFACT_SCHEMA_V2
        for item in references
    ):
        raise ValueError("M-only extension requires completed v0.1 artifact v2 references")
    initial_fingerprints = {
        item.config.initial_decoder_fingerprint for item in references
    }
    if len(initial_fingerprints) != 1:
        raise ValueError("reference F/Fx/U artifacts do not share one initialization")

    common_payloads: list[dict[str, object]] = []
    for item in references:
        payload = item.config.canonical_payload()
        payload.pop("variant")
        payload.pop("variant_contract")
        common_payloads.append(payload)
    if any(payload != common_payloads[0] for payload in common_payloads[1:]):
        raise ValueError("reference F/Fx/U configs differ outside variant")
    return stable_fingerprint(
        {
            "schema_version": "cure-lite-v01-reference-training-trio-v1",
            "common_run_config": common_payloads[0],
            "artifacts": {
                label: {
                    "artifact_fingerprint": item.artifact_fingerprint,
                    "receipt_sha256": item.receipt_sha256,
                    "decoder_state_fingerprint": (
                        item.decoder_state_fingerprint
                    ),
                    "train_log_fingerprint": item.train_log_fingerprint,
                }
                for label, item in zip(
                    ("F", "F×", "U"),
                    references,
                    strict=True,
                )
            },
        }
    )


@dataclass(frozen=True, slots=True)
class _CompletedMissAlignedExtensionSeal:
    factual_only_reference: LoadedDecoderArtifact
    factual_exposure_matched_reference: LoadedDecoderArtifact
    uniform_legal_reference: LoadedDecoderArtifact
    miss_aligned_legal: CompletedDecoderRun
    reference_training_fingerprint: str
    alignment_catalog_fingerprint: str


@dataclass(frozen=True)
class CompletedMissAlignedGate2Extension:
    """One M run paired to immutable completed v0.1 F/Fx/U references."""

    factual_only_reference: LoadedDecoderArtifact
    factual_exposure_matched_reference: LoadedDecoderArtifact
    uniform_legal_reference: LoadedDecoderArtifact
    miss_aligned_legal: CompletedDecoderRun
    reference_training_fingerprint: str
    alignment_catalog_fingerprint: str
    _verification_token: object

    def _verify_source_seal(self) -> None:
        seal = self._verification_token
        if type(seal) is not _CompletedMissAlignedExtensionSeal:
            raise TypeError(
                "CompletedMissAlignedGate2Extension must come from the "
                "formal M-only trainer"
            )
        if (
            seal.factual_only_reference is not self.factual_only_reference
            or seal.factual_exposure_matched_reference
            is not self.factual_exposure_matched_reference
            or seal.uniform_legal_reference is not self.uniform_legal_reference
            or seal.miss_aligned_legal is not self.miss_aligned_legal
            or seal.reference_training_fingerprint
            != self.reference_training_fingerprint
            or seal.alignment_catalog_fingerprint
            != self.alignment_catalog_fingerprint
        ):
            raise TypeError("completed M-only extension fields were replaced")

    def __post_init__(self) -> None:
        self._verify_source_seal()
        self.verify_unchanged()

    @property
    def extension_fingerprint(self) -> str:
        return stable_fingerprint(
            {
                "schema_version": "cure-lite-miss-aligned-gate2-extension-v1",
                "reference_training_fingerprint": (
                    self.reference_training_fingerprint
                ),
                "alignment_catalog_fingerprint": (
                    self.alignment_catalog_fingerprint
                ),
                "miss_aligned_config": (
                    self.miss_aligned_legal.config.canonical_payload()
                ),
                "miss_aligned_final_decoder_fingerprint": (
                    self.miss_aligned_legal.final_decoder_fingerprint
                ),
                "miss_aligned_training_log": (
                    self.miss_aligned_legal.training_log.canonical_epoch_logs()
                ),
            }
        )

    def verify_unchanged(self) -> None:
        self._verify_source_seal()
        reference_fingerprint = _verified_v01_reference_training_fingerprint(
            self.factual_only_reference,
            self.factual_exposure_matched_reference,
            self.uniform_legal_reference,
        )
        if reference_fingerprint != self.reference_training_fingerprint:
            raise RuntimeError("completed F/Fx/U references changed")
        self.miss_aligned_legal.verify_unchanged()
        config = self.miss_aligned_legal.config
        if (
            config.schema_version != DECODER_ARTIFACT_SCHEMA_V3
            or config.variant != "miss_aligned_legal"
        ):
            raise ValueError("extension output is not a v3 miss_aligned_legal run")
        if config.alignment_catalog_fingerprint != (
            self.alignment_catalog_fingerprint
        ):
            raise ValueError("M config and extension bind different alignment catalogs")
        if config.initial_decoder_fingerprint != (
            self.factual_only_reference.config.initial_decoder_fingerprint
        ):
            raise ValueError("M and reference F/Fx/U do not share initialization")
        m_logs = self.miss_aligned_legal.training_log.canonical_epoch_logs()
        reference_logs = (
            self.factual_only_reference.epoch_logs,
            self.factual_exposure_matched_reference.epoch_logs,
            self.uniform_legal_reference.epoch_logs,
        )
        for m_log, f_log, fx_log, u_log in zip(
            m_logs,
            *reference_logs,
            strict=True,
        ):
            m_pools = m_log["pool_sizes"]
            reference_pools = tuple(
                item["pool_sizes"] for item in (f_log, fx_log, u_log)
            )
            for branch in ("factual_miss", "factual_no_miss"):
                if any(
                    pools[branch] != m_pools[branch]
                    for pools in reference_pools
                ):
                    raise ValueError(
                        f"M and reference factual pool sizes differ for {branch}"
                    )
            if m_pools["synthetic"] != m_pools["factual_miss"]:
                raise ValueError(
                    "M synthetic pool must match its factual-miss pool size"
                )
            m_metrics = m_log["metrics"]
            reference_metrics = tuple(
                item["metrics"] for item in (f_log, fx_log, u_log)
            )
            for branch in ("factual_miss", "factual_no_miss"):
                for suffix in (
                    "active",
                    "active_min",
                    "active_max",
                    "states",
                    "states_min",
                    "states_max",
                ):
                    key = f"{branch}/{suffix}"
                    if any(
                        metrics.get(key) != m_metrics.get(key)
                        for metrics in reference_metrics
                    ):
                        raise ValueError(
                            f"M and reference factual exposure differ for {key}"
                        )
            for suffix in (
                "active",
                "active_min",
                "active_max",
                "states",
                "states_min",
                "states_max",
            ):
                key = f"synthetic/{suffix}"
                if (
                    fx_log["metrics"].get(key) != m_metrics.get(key)
                    or u_log["metrics"].get(key) != m_metrics.get(key)
                ):
                    raise ValueError(
                        f"M and reference third-slot exposure differ for {key}"
                    )


def _training_sources(bundle: LoadedDRCacheBundle) -> tuple[CachedTrainingSource, ...]:
    return tuple(
        CachedTrainingSource(
            row.sample_id,
            row.base_output.feature,
            row.base_output.probability,
            row.state,
        )
        for row in bundle.rows
    )


def _optimizer(
    decoder: CURELiteDecoder,
    config: PairedGate2TrainingConfig,
) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(
            decoder.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    return torch.optim.SGD(
        decoder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _run_config(
    bundle: LoadedDRCacheBundle,
    config: PairedGate2TrainingConfig,
    *,
    variant: Literal[
        "factual_only",
        "factual_exposure_matched",
        "uniform_legal",
    ],
    initial_decoder_fingerprint: str,
) -> DecoderRunConfig:
    return DecoderRunConfig(
        variant=variant,
        manifest_fingerprint=bundle.split_manifest_fingerprint,
        manifest_file_sha256=bundle.split_manifest_file_sha256,
        preprocessing_fingerprint=bundle.preprocessing_fingerprint,
        base_fingerprint=bundle.base_fingerprint,
        state_fingerprint=bundle.state_fingerprint,
        gt_fingerprint=bundle.gt_fingerprint,
        base_index_fingerprint=bundle.base_index_fingerprint,
        base_index_sha256=bundle.base_index_sha256,
        state_index_fingerprint=bundle.state_index_fingerprint,
        state_index_sha256=bundle.state_index_sha256,
        initial_decoder_fingerprint=initial_decoder_fingerprint,
        occupancy_config=bundle.occupancy_config,
        match_config=bundle.match_config,
        intervention_config=bundle.intervention_config,
        global_seed=config.global_seed,
        trained_epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        decoder_config=config.decoder_config,
        loss_config=config.loss_config,
        training_config=config.training_config,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        factual_miss_batch=config.factual_miss_batch,
        factual_no_miss_batch=config.factual_no_miss_batch,
        synthetic_batch=config.synthetic_batch,
    )


def _require_v01_references_match_extension(
    bundle: LoadedDRCacheBundle,
    config: MissAlignedGate2TrainingConfig,
    factual_only: LoadedDecoderArtifact,
    factual_exposure_matched: LoadedDecoderArtifact,
    uniform_legal: LoadedDecoderArtifact,
) -> str:
    reference_fingerprint = _verified_v01_reference_training_fingerprint(
        factual_only,
        factual_exposure_matched,
        uniform_legal,
    )
    reference = factual_only.config
    bundle_bindings = {
        "manifest_fingerprint": bundle.split_manifest_fingerprint,
        "manifest_file_sha256": bundle.split_manifest_file_sha256,
        "preprocessing_fingerprint": bundle.preprocessing_fingerprint,
        "base_fingerprint": bundle.base_fingerprint,
        "state_fingerprint": bundle.state_fingerprint,
        "gt_fingerprint": bundle.gt_fingerprint,
        "base_index_fingerprint": bundle.base_index_fingerprint,
        "base_index_sha256": bundle.base_index_sha256,
        "state_index_fingerprint": bundle.state_index_fingerprint,
        "state_index_sha256": bundle.state_index_sha256,
    }
    for name, expected in bundle_bindings.items():
        if getattr(reference, name) != expected:
            raise ValueError(f"reference F/Fx/U {name} differs from D_R bundle")
    if (
        reference.occupancy_config != bundle.occupancy_config
        or reference.match_config != bundle.match_config
        or reference.intervention_config != bundle.intervention_config
    ):
        raise ValueError("reference F/Fx/U semantic configs differ from D_R bundle")
    requested_bindings = {
        "decoder_config": config.decoder_config,
        "loss_config": config.loss_config,
        "training_config": config.training_config,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "trained_epochs": config.epochs,
        "steps_per_epoch": config.steps_per_epoch,
        "factual_miss_batch": config.factual_miss_batch,
        "factual_no_miss_batch": config.factual_no_miss_batch,
        "synthetic_batch": config.synthetic_batch,
        "global_seed": config.global_seed,
    }
    for name, expected in requested_bindings.items():
        if getattr(reference, name) != expected:
            raise ValueError(
                f"M training request differs from completed F/Fx/U at {name}"
            )
    return reference_fingerprint


def _miss_aligned_run_config(
    bundle: LoadedDRCacheBundle,
    config: MissAlignedGate2TrainingConfig,
    *,
    initial_decoder_fingerprint: str,
    alignment_catalog_fingerprint: str,
) -> DecoderRunConfig:
    return DecoderRunConfig(
        variant="miss_aligned_legal",
        manifest_fingerprint=bundle.split_manifest_fingerprint,
        manifest_file_sha256=bundle.split_manifest_file_sha256,
        preprocessing_fingerprint=bundle.preprocessing_fingerprint,
        base_fingerprint=bundle.base_fingerprint,
        state_fingerprint=bundle.state_fingerprint,
        gt_fingerprint=bundle.gt_fingerprint,
        base_index_fingerprint=bundle.base_index_fingerprint,
        base_index_sha256=bundle.base_index_sha256,
        state_index_fingerprint=bundle.state_index_fingerprint,
        state_index_sha256=bundle.state_index_sha256,
        initial_decoder_fingerprint=initial_decoder_fingerprint,
        occupancy_config=bundle.occupancy_config,
        match_config=bundle.match_config,
        intervention_config=bundle.intervention_config,
        global_seed=config.global_seed,
        trained_epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        decoder_config=config.decoder_config,
        loss_config=config.loss_config,
        training_config=config.training_config,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        factual_miss_batch=config.factual_miss_batch,
        factual_no_miss_batch=config.factual_no_miss_batch,
        synthetic_batch=config.synthetic_batch,
        schema_version=DECODER_ARTIFACT_SCHEMA_V3,
        miss_alignment_config=config.miss_alignment_config,
        alignment_catalog_fingerprint=alignment_catalog_fingerprint,
    )


def _fresh_single_decoder(
    config: DecoderConfig,
    *,
    global_seed: int,
    expected_initial_fingerprint: str,
    device: torch.device,
) -> tuple[CURELiteDecoder, str]:
    if device.type == "meta":
        raise ValueError("formal training cannot run on the meta device")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(global_seed)
        decoder = CURELiteDecoder(config)
    initial_fingerprint = decoder_state_fingerprint(decoder)
    if initial_fingerprint != expected_initial_fingerprint:
        raise RuntimeError(
            "deterministic M initialization differs from completed F/Fx/U"
        )
    decoder.to(device)
    if decoder_state_fingerprint(decoder) != initial_fingerprint:
        raise RuntimeError("moving the M decoder changed its initialization")
    return decoder, initial_fingerprint


def _fresh_paired_decoders(
    config: DecoderConfig,
    *,
    global_seed: int,
    device: torch.device,
) -> tuple[CURELiteDecoder, CURELiteDecoder, CURELiteDecoder, str]:
    if device.type == "meta":
        raise ValueError("formal training cannot run on the meta device")
    # Initialization happens on CPU inside an isolated RNG context.  Loading
    # that one state into both variants makes equality independent of run order.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(global_seed)
        template = CURELiteDecoder(config)
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in template.state_dict().items()
    }
    initial_fingerprint = decoder_state_fingerprint(template)
    factual = CURELiteDecoder(config)
    exposure_matched = CURELiteDecoder(config)
    uniform = CURELiteDecoder(config)
    factual.load_state_dict(initial_state, strict=True)
    exposure_matched.load_state_dict(initial_state, strict=True)
    uniform.load_state_dict(initial_state, strict=True)
    factual.to(device)
    exposure_matched.to(device)
    uniform.to(device)
    if not (
        decoder_state_fingerprint(factual)
        == decoder_state_fingerprint(exposure_matched)
        == decoder_state_fingerprint(uniform)
        == initial_fingerprint
    ):
        raise RuntimeError("failed to create byte-identical paired decoders")
    return factual, exposure_matched, uniform, initial_fingerprint


def run_paired_gate2_training(
    bundle: LoadedDRCacheBundle,
    config: PairedGate2TrainingConfig,
    *,
    device: torch.device | str = "cpu",
    prepared: PreparedGate2Training | None = None,
) -> CompletedPairedGate2Run:
    """Train F, Fx, and U with one D_R bundle and identical initialization."""

    if not isinstance(bundle, LoadedDRCacheBundle):
        raise TypeError("bundle must be a LoadedDRCacheBundle")
    if not isinstance(config, PairedGate2TrainingConfig):
        raise TypeError("config must be a PairedGate2TrainingConfig")
    resolved_device = torch.device(device)
    resolved_prepared = _resolve_prepared_gate2_training(bundle, prepared)
    # One strict check immediately before any optimizer update and one after
    # the complete paired run are sufficient to prove immutability. Rehashing
    # the same D_R files after each variant only repeated I/O without changing
    # the experimental contract.
    resolved_prepared.verify_unchanged()
    feature_channels = {int(row.base_output.feature.shape[1]) for row in bundle.rows}
    if feature_channels != {config.decoder_config.feature_channels}:
        raise ValueError("decoder feature channels differ from the D_R cache bundle")
    sources = resolved_prepared.sources
    support_pools = build_epoch_branch_pools_from_catalog(
        resolved_prepared.catalog,
        variant="uniform_legal",
        epoch=0,
        global_seed=config.global_seed,
    )
    require_training_branch_support(support_pools, variant="uniform_legal")
    (
        factual_decoder,
        exposure_matched_decoder,
        uniform_decoder,
        initial_fingerprint,
    ) = _fresh_paired_decoders(
        config.decoder_config,
        global_seed=config.global_seed,
        device=resolved_device,
    )

    outputs: dict[str, CompletedDecoderRun] = {}
    for variant, decoder in (
        ("factual_only", factual_decoder),
        ("factual_exposure_matched", exposure_matched_decoder),
        ("uniform_legal", uniform_decoder),
    ):
        run_config = _run_config(
            bundle,
            config,
            variant=variant,
            initial_decoder_fingerprint=initial_fingerprint,
        )
        log = run_fixed_training(
            decoder,
            CURELiteLoss(config.loss_config).to(resolved_device),
            _optimizer(decoder, config),
            sources,
            variant=variant,
            epochs=config.epochs,
            steps_per_epoch=config.steps_per_epoch,
            branch_batch_sizes=config.branch_batch_sizes,
            global_seed=config.global_seed,
            device=resolved_device,
            occupancy_config=bundle.occupancy_config,
            match_config=bundle.match_config,
            intervention_config=bundle.intervention_config,
            training_config=config.training_config,
            prepared_catalog=resolved_prepared.catalog,
        )
        final_fingerprint = decoder_state_fingerprint(decoder)
        seal = _CompletedRunSeal(
            decoder=decoder,
            config=run_config,
            training_log=log,
            final_decoder_fingerprint=final_fingerprint,
        )
        outputs[variant] = CompletedDecoderRun(
            decoder=decoder,
            config=run_config,
            training_log=log,
            final_decoder_fingerprint=final_fingerprint,
            _verification_token=seal,
        )
        outputs[variant].verify_unchanged()

    factual_output = outputs["factual_only"]
    exposure_matched_output = outputs["factual_exposure_matched"]
    uniform_output = outputs["uniform_legal"]
    pair_seal = _CompletedPairSeal(
        factual_only=factual_output,
        factual_exposure_matched=exposure_matched_output,
        uniform_legal=uniform_output,
        initial_decoder_fingerprint=initial_fingerprint,
    )
    result = CompletedPairedGate2Run(
        factual_only=factual_output,
        factual_exposure_matched=exposure_matched_output,
        uniform_legal=uniform_output,
        initial_decoder_fingerprint=initial_fingerprint,
        _verification_token=pair_seal,
    )
    resolved_prepared.verify_unchanged()
    return result


def run_miss_aligned_gate2_extension(
    bundle: LoadedDRCacheBundle,
    config: MissAlignedGate2TrainingConfig,
    *,
    factual_only_reference: LoadedDecoderArtifact,
    factual_exposure_matched_reference: LoadedDecoderArtifact,
    uniform_legal_reference: LoadedDecoderArtifact,
    device: torch.device | str = "cpu",
    prepared: PreparedGate2Training | None = None,
    training_progress: Callable[[FixedEpochTrainingLog], None] | None = None,
) -> CompletedMissAlignedGate2Extension:
    """Train only M while binding it to completed immutable F/Fx/U references."""

    if not isinstance(bundle, LoadedDRCacheBundle):
        raise TypeError("bundle must be a LoadedDRCacheBundle")
    if not isinstance(config, MissAlignedGate2TrainingConfig):
        raise TypeError("config must be a MissAlignedGate2TrainingConfig")
    if training_progress is not None and not callable(training_progress):
        raise TypeError("training_progress must be callable or None")
    resolved_device = torch.device(device)
    resolved_prepared = _resolve_prepared_gate2_training(bundle, prepared)
    resolved_prepared.catalog.require_compatible(
        resolved_prepared.sources,
        occupancy_config=bundle.occupancy_config,
        match_config=bundle.match_config,
        intervention_config=bundle.intervention_config,
        miss_alignment_config=config.miss_alignment_config,
    )
    resolved_prepared.verify_unchanged()
    reference_fingerprint = _require_v01_references_match_extension(
        bundle,
        config,
        factual_only_reference,
        factual_exposure_matched_reference,
        uniform_legal_reference,
    )
    feature_channels = {int(row.base_output.feature.shape[1]) for row in bundle.rows}
    if feature_channels != {config.decoder_config.feature_channels}:
        raise ValueError("decoder feature channels differ from the D_R cache bundle")
    support_pools = build_epoch_branch_pools_from_catalog(
        resolved_prepared.catalog,
        variant="miss_aligned_legal",
        epoch=0,
        global_seed=config.global_seed,
    )
    require_training_branch_support(
        support_pools,
        variant="miss_aligned_legal",
    )
    alignment_fingerprint = (
        resolved_prepared.catalog.miss_alignment_fingerprint
    )
    decoder, initial_fingerprint = _fresh_single_decoder(
        config.decoder_config,
        global_seed=config.global_seed,
        expected_initial_fingerprint=(
            factual_only_reference.config.initial_decoder_fingerprint
        ),
        device=resolved_device,
    )
    run_config = _miss_aligned_run_config(
        bundle,
        config,
        initial_decoder_fingerprint=initial_fingerprint,
        alignment_catalog_fingerprint=alignment_fingerprint,
    )
    log = run_fixed_training(
        decoder,
        CURELiteLoss(config.loss_config).to(resolved_device),
        _optimizer(decoder, config),
        resolved_prepared.sources,
        variant="miss_aligned_legal",
        epochs=config.epochs,
        steps_per_epoch=config.steps_per_epoch,
        branch_batch_sizes=config.branch_batch_sizes,
        global_seed=config.global_seed,
        device=resolved_device,
        occupancy_config=bundle.occupancy_config,
        match_config=bundle.match_config,
        intervention_config=bundle.intervention_config,
        miss_alignment_config=config.miss_alignment_config,
        training_config=config.training_config,
        prepared_catalog=resolved_prepared.catalog,
        progress=training_progress,
    )
    final_fingerprint = decoder_state_fingerprint(decoder)
    run_seal = _CompletedRunSeal(
        decoder=decoder,
        config=run_config,
        training_log=log,
        final_decoder_fingerprint=final_fingerprint,
    )
    completed_m = CompletedDecoderRun(
        decoder=decoder,
        config=run_config,
        training_log=log,
        final_decoder_fingerprint=final_fingerprint,
        _verification_token=run_seal,
    )
    extension_seal = _CompletedMissAlignedExtensionSeal(
        factual_only_reference=factual_only_reference,
        factual_exposure_matched_reference=(
            factual_exposure_matched_reference
        ),
        uniform_legal_reference=uniform_legal_reference,
        miss_aligned_legal=completed_m,
        reference_training_fingerprint=reference_fingerprint,
        alignment_catalog_fingerprint=alignment_fingerprint,
    )
    result = CompletedMissAlignedGate2Extension(
        factual_only_reference=factual_only_reference,
        factual_exposure_matched_reference=(
            factual_exposure_matched_reference
        ),
        uniform_legal_reference=uniform_legal_reference,
        miss_aligned_legal=completed_m,
        reference_training_fingerprint=reference_fingerprint,
        alignment_catalog_fingerprint=alignment_fingerprint,
        _verification_token=extension_seal,
    )
    resolved_prepared.verify_unchanged()
    return result


def summarize_gate2_training_support(
    bundle: LoadedDRCacheBundle,
    *,
    prepared: PreparedGate2Training | None = None,
) -> TrainingSupportSummary:
    """Return the revalidated D_R support used by formal paired training."""

    if not isinstance(bundle, LoadedDRCacheBundle):
        raise TypeError("bundle must be a LoadedDRCacheBundle")
    resolved_prepared = _resolve_prepared_gate2_training(bundle, prepared)
    summary = summarize_training_support(
        resolved_prepared.sources,
        occupancy_config=bundle.occupancy_config,
        match_config=bundle.match_config,
        intervention_config=bundle.intervention_config,
        prepared_catalog=resolved_prepared.catalog,
    )
    resolved_prepared.verify_binding()
    return summary


def save_completed_decoder_run(
    directory: str | Path,
    run: CompletedDecoderRun,
) -> str:
    """Persist only a verified completed run through the strict artifact API."""

    if not isinstance(run, CompletedDecoderRun):
        raise TypeError("run must be a CompletedDecoderRun")
    run.verify_unchanged()
    fingerprint = _save_decoder_artifact(
        directory,
        run.decoder,
        run.config,
        run.training_log.canonical_epoch_logs(),
    )
    run.verify_unchanged()
    return fingerprint


__all__ = [
    "CompletedMissAlignedGate2Extension",
    "MissAlignedGate2TrainingConfig",
    "PairedGate2TrainingConfig",
    "run_miss_aligned_gate2_extension",
    "run_paired_gate2_training",
    "save_completed_decoder_run",
    "summarize_gate2_training_support",
]
