"""Provenance-bound paired F/U training for the Gate-2 CURE-Lite pilot.

The lower-level training helpers intentionally remain useful for unit tests and
mechanism studies.  This module is the formal experiment entry point: it only
accepts a fully verified ``D_R`` cache bundle, creates the decoder, loss, and
optimizer internally, and forces Factual-only and Uniform-Legal to start from
the exact same decoder bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

import torch

from ..config import DecoderConfig, LossConfig, TrainingConfig
from ..decoder import CURELiteDecoder
from ..losses import CURELiteLoss
from .artifacts import (
    DecoderRunConfig,
    _save_decoder_artifact,
    decoder_state_fingerprint,
)
from .cache_pipeline import LoadedDRCacheBundle
from .training_pipeline import (
    CachedTrainingSource,
    FixedTrainingLog,
    TrainingSupportSummary,
    build_epoch_branch_pools,
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
    uniform_legal: CompletedDecoderRun
    initial_decoder_fingerprint: str


@dataclass(frozen=True)
class CompletedPairedGate2Run:
    """The paired F/U outputs sharing one proven initialization."""

    factual_only: CompletedDecoderRun
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
            or seal.uniform_legal is not self.uniform_legal
            or seal.initial_decoder_fingerprint
            != self.initial_decoder_fingerprint
        ):
            raise TypeError("completed paired-run fields were replaced")

    def __post_init__(self) -> None:
        self._verify_source_seal()
        if self.factual_only.config.variant != "factual_only":
            raise ValueError("factual_only result has the wrong variant")
        if self.uniform_legal.config.variant != "uniform_legal":
            raise ValueError("uniform_legal result has the wrong variant")
        if not (
            self.factual_only.config.initial_decoder_fingerprint
            == self.uniform_legal.config.initial_decoder_fingerprint
            == self.initial_decoder_fingerprint
        ):
            raise ValueError("paired F/U runs do not share one initialization")
        factual_payload = self.factual_only.config.canonical_payload()
        uniform_payload = self.uniform_legal.config.canonical_payload()
        factual_payload.pop("variant")
        uniform_payload.pop("variant")
        if factual_payload != uniform_payload:
            raise ValueError("paired F/U run configs differ outside variant")
        self.verify_unchanged()

    def verify_unchanged(self) -> None:
        self._verify_source_seal()
        self.factual_only.verify_unchanged()
        self.uniform_legal.verify_unchanged()


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
    variant: Literal["factual_only", "uniform_legal"],
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


def _fresh_paired_decoders(
    config: DecoderConfig,
    *,
    global_seed: int,
    device: torch.device,
) -> tuple[CURELiteDecoder, CURELiteDecoder, str]:
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
    uniform = CURELiteDecoder(config)
    factual.load_state_dict(initial_state, strict=True)
    uniform.load_state_dict(initial_state, strict=True)
    factual.to(device)
    uniform.to(device)
    if not (
        decoder_state_fingerprint(factual)
        == decoder_state_fingerprint(uniform)
        == initial_fingerprint
    ):
        raise RuntimeError("failed to create byte-identical paired decoders")
    return factual, uniform, initial_fingerprint


def run_paired_gate2_training(
    bundle: LoadedDRCacheBundle,
    config: PairedGate2TrainingConfig,
    *,
    device: torch.device | str = "cpu",
) -> CompletedPairedGate2Run:
    """Train F and U with one D_R bundle and a byte-identical initialization."""

    if not isinstance(bundle, LoadedDRCacheBundle):
        raise TypeError("bundle must be a LoadedDRCacheBundle")
    if not isinstance(config, PairedGate2TrainingConfig):
        raise TypeError("config must be a PairedGate2TrainingConfig")
    resolved_device = torch.device(device)
    bundle.verify_unchanged()
    feature_channels = {int(row.base_output.feature.shape[1]) for row in bundle.rows}
    if feature_channels != {config.decoder_config.feature_channels}:
        raise ValueError("decoder feature channels differ from the D_R cache bundle")
    sources = _training_sources(bundle)
    support_pools = build_epoch_branch_pools(
        sources,
        variant="uniform_legal",
        epoch=0,
        global_seed=config.global_seed,
        occupancy_config=bundle.occupancy_config,
        match_config=bundle.match_config,
        intervention_config=bundle.intervention_config,
    )
    require_training_branch_support(support_pools, variant="uniform_legal")
    factual_decoder, uniform_decoder, initial_fingerprint = _fresh_paired_decoders(
        config.decoder_config,
        global_seed=config.global_seed,
        device=resolved_device,
    )

    outputs: dict[str, CompletedDecoderRun] = {}
    for variant, decoder in (
        ("factual_only", factual_decoder),
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
        bundle.verify_unchanged()

    factual_output = outputs["factual_only"]
    uniform_output = outputs["uniform_legal"]
    pair_seal = _CompletedPairSeal(
        factual_only=factual_output,
        uniform_legal=uniform_output,
        initial_decoder_fingerprint=initial_fingerprint,
    )
    result = CompletedPairedGate2Run(
        factual_only=factual_output,
        uniform_legal=uniform_output,
        initial_decoder_fingerprint=initial_fingerprint,
        _verification_token=pair_seal,
    )
    bundle.verify_unchanged()
    return result


def summarize_gate2_training_support(
    bundle: LoadedDRCacheBundle,
) -> TrainingSupportSummary:
    """Return the revalidated D_R support used by formal paired training."""

    if not isinstance(bundle, LoadedDRCacheBundle):
        raise TypeError("bundle must be a LoadedDRCacheBundle")
    bundle.verify_unchanged()
    summary = summarize_training_support(
        _training_sources(bundle),
        occupancy_config=bundle.occupancy_config,
        match_config=bundle.match_config,
        intervention_config=bundle.intervention_config,
    )
    bundle.verify_unchanged()
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
    "PairedGate2TrainingConfig",
    "run_paired_gate2_training",
    "save_completed_decoder_run",
    "summarize_gate2_training_support",
]
