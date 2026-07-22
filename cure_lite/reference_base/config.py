"""Configuration for the project-owned Stage-A reference detector.

The reference detector is an experiment input provider, not a component of
CURE-Lite.  Its configuration is intentionally small and detector-specific
choices never enter the CURE-Lite core namespace.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

from ..data import PreprocessConfig


REFERENCE_BASE_CONFIG_SCHEMA = "cure-lite-reference-base-config-v1"


@dataclass(frozen=True)
class ReferenceBaseModelConfig:
    """Fixed compact U-Net dimensions used by the reference provider."""

    in_channels: int = 1
    stem_channels: int = 24
    half_channels: int = 40
    feature_channels: int = 64
    eighth_channels: int = 96
    bottleneck_channels: int = 128
    norm_groups: int = 8

    def __post_init__(self) -> None:
        values = (
            self.in_channels,
            self.stem_channels,
            self.half_channels,
            self.feature_channels,
            self.eighth_channels,
            self.bottleneck_channels,
            self.norm_groups,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in values
        ):
            raise ValueError("reference-base channel counts must be positive integers")
        if self.in_channels != 1:
            raise ValueError("the reference base uses one grayscale input channel")
        for channels in values[1:-1]:
            if channels % self.norm_groups:
                raise ValueError("every reference-base width must divide norm_groups")

    def canonical_payload(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class ReferenceBaseTrainingConfig:
    """Complete training and D_B checkpoint-selection recipe."""

    dataset: str = "IRSTD-1K"
    epochs: int = 800
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    bce_weight: float = 0.2
    soft_iou_weight: float = 1.0
    positive_weight: float = 20.0
    selection_fraction: float = 0.2
    training_seed: int = 42
    selection_seed: int = 42
    device: str = "cuda:1"
    model: ReferenceBaseModelConfig = ReferenceBaseModelConfig()
    preprocess: PreprocessConfig = PreprocessConfig(
        height=256,
        width=256,
        color_mode="L",
        mean=(0.5,),
        std=(0.5,),
    )

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("dataset must be non-empty")
        for name in ("epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("training_seed", "selection_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "learning_rate",
            "weight_decay",
            "bce_weight",
            "soft_iou_weight",
            "positive_weight",
            "selection_fraction",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.bce_weight < 0 or self.soft_iou_weight < 0:
            raise ValueError("loss weights must be nonnegative")
        if self.bce_weight + self.soft_iou_weight <= 0:
            raise ValueError("at least one reference-base loss weight must be positive")
        if self.positive_weight <= 0:
            raise ValueError("positive_weight must be positive")
        if not 0.0 < self.selection_fraction < 1.0:
            raise ValueError("selection_fraction must lie strictly between zero and one")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be non-empty")
        if not isinstance(self.model, ReferenceBaseModelConfig):
            raise TypeError("model must be ReferenceBaseModelConfig")
        if not isinstance(self.preprocess, PreprocessConfig):
            raise TypeError("preprocess must be PreprocessConfig")
        if self.preprocess.color_mode != "L" or len(self.preprocess.mean) != 1:
            raise ValueError("reference-base preprocessing must be grayscale")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": REFERENCE_BASE_CONFIG_SCHEMA,
            "dataset": self.dataset,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "bce_weight": float(self.bce_weight),
            "soft_iou_weight": float(self.soft_iou_weight),
            "positive_weight": float(self.positive_weight),
            "selection_fraction": float(self.selection_fraction),
            "training_seed": self.training_seed,
            "selection_seed": self.selection_seed,
            "device": self.device,
            "model": self.model.canonical_payload(),
            "preprocessing": self.preprocess.fingerprint_payload(),
            "optimizer": "adamw",
            "schedule": "cosine-to-zero",
            "augmentation": "deterministic-dihedral-8",
            "selection_metric": "D_B-select/global-binary-mIoU@0.5",
            "selection_tie_break": "lower-select-loss-then-earlier-epoch",
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ReferenceBaseTrainingConfig":
        expected = {
            "schema_version",
            "dataset",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "bce_weight",
            "soft_iou_weight",
            "positive_weight",
            "selection_fraction",
            "training_seed",
            "selection_seed",
            "device",
            "model",
            "preprocessing",
            "optimizer",
            "schedule",
            "augmentation",
            "selection_metric",
            "selection_tie_break",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("reference-base config fields are not canonical")
        if value["schema_version"] != REFERENCE_BASE_CONFIG_SCHEMA:
            raise ValueError("unsupported reference-base config schema")
        fixed = {
            "optimizer": "adamw",
            "schedule": "cosine-to-zero",
            "augmentation": "deterministic-dihedral-8",
            "selection_metric": "D_B-select/global-binary-mIoU@0.5",
            "selection_tie_break": "lower-select-loss-then-earlier-epoch",
        }
        if any(value[name] != expected_value for name, expected_value in fixed.items()):
            raise ValueError("reference-base fixed recipe fields differ")
        model = value["model"]
        if not isinstance(model, Mapping):
            raise TypeError("reference-base model config must be a mapping")
        result = cls(
            dataset=value["dataset"],  # type: ignore[arg-type]
            epochs=value["epochs"],  # type: ignore[arg-type]
            batch_size=value["batch_size"],  # type: ignore[arg-type]
            learning_rate=value["learning_rate"],  # type: ignore[arg-type]
            weight_decay=value["weight_decay"],  # type: ignore[arg-type]
            bce_weight=value["bce_weight"],  # type: ignore[arg-type]
            soft_iou_weight=value["soft_iou_weight"],  # type: ignore[arg-type]
            positive_weight=value["positive_weight"],  # type: ignore[arg-type]
            selection_fraction=value["selection_fraction"],  # type: ignore[arg-type]
            training_seed=value["training_seed"],  # type: ignore[arg-type]
            selection_seed=value["selection_seed"],  # type: ignore[arg-type]
            device=value["device"],  # type: ignore[arg-type]
            model=ReferenceBaseModelConfig(**dict(model)),
            preprocess=PreprocessConfig.from_fingerprint_payload(
                value["preprocessing"]
            ),
        )
        if result.canonical_payload() != dict(value):
            raise ValueError("reference-base config is not canonical")
        return result


__all__ = [
    "REFERENCE_BASE_CONFIG_SCHEMA",
    "ReferenceBaseModelConfig",
    "ReferenceBaseTrainingConfig",
]
