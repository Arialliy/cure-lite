"""Compact artifact-bound inference for a calibrated CURE-Lite decoder.

The training-time :class:`~cure_lite.model.CURELiteModel` accepts a mutable
decoder and a runtime threshold.  This module creates the inference boundary:
one verified Stage-A artifact is copied into a private decoder, paired with its
frozen Base identity and thresholds, and summarized by a compact receipt.  The
completed Stage-A caches and the two unused control decoders are not retained.

The generic adapter contract supplies the declared Base identity while the
cache receipt carries a library-computed digest of the exact registered model
state.  Construction independently hashes the online Base and requires it to
match that historical cache-time state.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

import torch
from torch import Tensor, nn

from ..cache.schema import stable_fingerprint
from ..config import DecoderConfig, OccupancyConfig, config_to_dict
from ..data import PreprocessConfig
from ..decoder import CURELiteDecoder
from ..frozen_base import FrozenBaseAdapter, frozen_base_state_fingerprint
from ..model import CURELiteModel, CURELiteOutput
from ..types import FrozenBaseOutput
from .artifacts import LoadedDecoderArtifact, decoder_state_fingerprint
from .formal_evaluation import FormalDVThresholdReceipt
from .stage_a_runner import LoadedStageARun


CalibratedMethod = Literal["F", "F×", "U"]
DEPLOYMENT_RECEIPT_SCHEMA = "cure-lite-calibrated-deployment-v2"
_METHOD_VARIANTS = {
    "F": "factual_only",
    "F×": "factual_exposure_matched",
    "U": "uniform_legal",
}


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _shape3(value: object, *, name: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in value
        )
    ):
        raise ValueError(f"{name} must contain three positive integers")
    return value


def _runtime_tensor_signature(module: nn.Module) -> tuple[tuple[object, ...], ...]:
    """Detect replacement, versioned mutation, movement, or unfreezing cheaply."""

    rows: list[tuple[object, ...]] = []
    for kind, values in (
        ("parameter", module.named_parameters()),
        ("buffer", module.named_buffers()),
    ):
        for name, tensor in values:
            rows.append(
                (
                    kind,
                    name,
                    id(tensor),
                    tensor._version,
                    str(tensor.dtype),
                    str(tensor.device),
                    tuple(tensor.shape),
                    bool(tensor.requires_grad),
                )
            )
    return tuple(rows)


def _module_device(module: nn.Module) -> torch.device:
    devices = {
        tensor.device
        for tensor in (*tuple(module.parameters()), *tuple(module.buffers()))
    }
    if not devices:
        return torch.device("cpu")
    if len(devices) != 1:
        raise RuntimeError("frozen Base parameters and buffers span multiple devices")
    return next(iter(devices))


def _selected_sources(
    run: LoadedStageARun,
    method: CalibratedMethod,
) -> tuple[LoadedDecoderArtifact, FormalDVThresholdReceipt]:
    if method == "F":
        return run.factual_artifact, run.calibration.factual_only
    if method == "F×":
        return (
            run.factual_exposure_matched_artifact,
            run.calibration.factual_exposure_matched,
        )
    if method == "U":
        return run.uniform_artifact, run.calibration.uniform_legal
    raise ValueError("method must be one of 'F', 'F×', or 'U'")


@dataclass(frozen=True, slots=True)
class CalibratedDeploymentReceipt:
    method: CalibratedMethod
    decoder_variant: str
    base_fingerprint: str
    base_state_fingerprint: str
    preprocessing: PreprocessConfig
    preprocessing_fingerprint: str
    base_probability_shape: tuple[int, int, int]
    base_feature_shape: tuple[int, int, int]
    occupancy_config: OccupancyConfig
    residual_threshold: float | None
    decoder_config: DecoderConfig
    decoder_artifact_fingerprint: str
    decoder_state_fingerprint: str
    decoder_receipt_sha256: str
    threshold_protocol_fingerprint: str
    stage_a_complete_fingerprint: str

    def __post_init__(self) -> None:
        if self.method not in _METHOD_VARIANTS:
            raise ValueError("unsupported calibrated method")
        if self.decoder_variant != _METHOD_VARIANTS[self.method]:
            raise ValueError("method and decoder variant differ")
        for name in (
            "base_fingerprint",
            "base_state_fingerprint",
            "preprocessing_fingerprint",
            "decoder_artifact_fingerprint",
            "decoder_state_fingerprint",
            "decoder_receipt_sha256",
            "threshold_protocol_fingerprint",
            "stage_a_complete_fingerprint",
        ):
            _digest(getattr(self, name), name=name)
        if not isinstance(self.preprocessing, PreprocessConfig):
            raise TypeError("preprocessing must be a PreprocessConfig")
        if stable_fingerprint(self.preprocessing.fingerprint_payload()) != (
            self.preprocessing_fingerprint
        ):
            raise ValueError("preprocessing fingerprint mismatch")
        if not isinstance(self.occupancy_config, OccupancyConfig):
            raise TypeError("occupancy_config must be an OccupancyConfig")
        if not isinstance(self.decoder_config, DecoderConfig):
            raise TypeError("decoder_config must be a DecoderConfig")
        probability_shape = _shape3(
            self.base_probability_shape,
            name="base_probability_shape",
        )
        feature_shape = _shape3(
            self.base_feature_shape,
            name="base_feature_shape",
        )
        if probability_shape[0] != 1:
            raise ValueError("base probability shape must have one channel")
        if feature_shape[0] != self.decoder_config.feature_channels:
            raise ValueError("base feature shape differs from decoder channels")
        expected_grid = (self.preprocessing.height, self.preprocessing.width)
        if probability_shape[-2:] != expected_grid:
            raise ValueError("base probability grid differs from preprocessing")
        if any(
            feature > source
            for feature, source in zip(
                feature_shape[-2:], probability_shape[-2:], strict=True
            )
        ):
            raise ValueError("base feature grid may not exceed probability grid")
        if self.residual_threshold is not None:
            if isinstance(self.residual_threshold, bool) or not isinstance(
                self.residual_threshold,
                (int, float),
            ):
                raise TypeError("residual_threshold must be numeric or None")
            value = float(self.residual_threshold)
            if not isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("residual_threshold must lie in [0,1]")
            object.__setattr__(self, "residual_threshold", value)

    def canonical_payload(self) -> dict[str, object]:
        core: dict[str, object] = {
            "schema_version": DEPLOYMENT_RECEIPT_SCHEMA,
            "method": self.method,
            "decoder_variant": self.decoder_variant,
            "base_fingerprint": self.base_fingerprint,
            "base_state_fingerprint": self.base_state_fingerprint,
            "preprocessing": self.preprocessing.fingerprint_payload(),
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "base_probability_shape": list(self.base_probability_shape),
            "base_feature_shape": list(self.base_feature_shape),
            "occupancy_config": config_to_dict(self.occupancy_config),
            "residual_threshold": self.residual_threshold,
            "decoder_config": config_to_dict(self.decoder_config),
            "decoder_artifact_fingerprint": self.decoder_artifact_fingerprint,
            "decoder_state_fingerprint": self.decoder_state_fingerprint,
            "decoder_receipt_sha256": self.decoder_receipt_sha256,
            "threshold_protocol_fingerprint": self.threshold_protocol_fingerprint,
            "stage_a_complete_fingerprint": self.stage_a_complete_fingerprint,
            "input_contract": "canonical_preprocessed_float32_nchw",
            "inference_composition": "occupancy_hard_union_residual",
        }
        return {**core, "receipt_fingerprint": stable_fingerprint(core)}

    @property
    def receipt_fingerprint(self) -> str:
        return self.canonical_payload()["receipt_fingerprint"]  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class _CalibratedModelSeal:
    receipt: CalibratedDeploymentReceipt
    receipt_fingerprint: str
    core: CURELiteModel
    base: FrozenBaseAdapter
    decoder: CURELiteDecoder
    base_runtime_signature: tuple[tuple[object, ...], ...]
    decoder_runtime_signature: tuple[tuple[object, ...], ...]
    device: torch.device


class CalibratedCURELiteModel(nn.Module):
    """Inference-only model carrying one compact calibrated receipt.

    Inputs must already follow :attr:`preprocessing`; shape, dtype and device
    are checked.  Tensor values cannot reveal whether normalization was applied,
    so raw-image preprocessing remains the caller/adapter's explicit duty.
    Each forward verifies the small private decoder exactly and checks a cheap
    live-Base signature; :meth:`verify_unchanged` performs the full Base-state
    hash when an explicit integrity check is required.
    """

    def __init__(
        self,
        stage_a_run: LoadedStageARun,
        base: FrozenBaseAdapter,
        *,
        method: CalibratedMethod = "U",
    ) -> None:
        super().__init__()
        self._sealed = False
        if not isinstance(stage_a_run, LoadedStageARun):
            raise TypeError("stage_a_run must be a LoadedStageARun")
        if not isinstance(base, FrozenBaseAdapter):
            raise TypeError("base must implement FrozenBaseAdapter")
        if method not in _METHOD_VARIANTS:
            raise ValueError("method must be one of 'F', 'F×', or 'U'")

        stage_a_run.verify_published_receipts()
        stage_binding = stage_a_run._verify_binding()
        stage_a_run.d_r_bundle._verify_source_seal()
        stage_a_run.d_v_bundle._verify_source_seal()
        artifact, threshold_receipt = _selected_sources(stage_a_run, method)
        threshold_receipt._verify_source_seal()
        artifact.verify_unchanged()

        source_preprocessing = stage_binding.d_v_dataset.preprocess
        if source_preprocessing != stage_binding.d_r_dataset.preprocess:
            raise RuntimeError("D_R and D_V preprocessing contracts differ")
        if not isinstance(source_preprocessing, PreprocessConfig):
            raise TypeError("Stage-A preprocessing must be a PreprocessConfig")
        preprocessing = PreprocessConfig.from_fingerprint_payload(
            source_preprocessing.fingerprint_payload()
        )
        base.validate_preprocessing(preprocessing)
        preprocessing_fingerprint = stable_fingerprint(
            preprocessing.fingerprint_payload()
        )

        config = artifact.config
        protocol = threshold_receipt.protocol
        expected_variant = _METHOD_VARIANTS[method]
        if config.variant != expected_variant:
            raise RuntimeError("selected decoder variant differs from method")
        if threshold_receipt.mode != "residual" or protocol.variant != "residual":
            raise RuntimeError("calibrated CURE-Lite requires a residual receipt")
        if threshold_receipt.decoder_variant != config.variant:
            raise RuntimeError("threshold receipt binds another decoder variant")
        if (
            threshold_receipt.decoder_artifact_fingerprint
            != artifact.artifact_fingerprint
            or threshold_receipt.decoder_receipt_sha256 != artifact.receipt_sha256
            or threshold_receipt.decoder_state_fingerprint
            != artifact.decoder_state_fingerprint
        ):
            raise RuntimeError("threshold receipt binds another decoder artifact")
        if (
            base.fingerprint != config.base_fingerprint
            or threshold_receipt.base_fingerprint != config.base_fingerprint
            or stage_a_run.d_v_bundle.base_fingerprint != config.base_fingerprint
        ):
            raise RuntimeError("Base identity differs from decoder/calibration")
        if (
            preprocessing_fingerprint != config.preprocessing_fingerprint
            or threshold_receipt.preprocessing_fingerprint
            != config.preprocessing_fingerprint
            or stage_a_run.d_v_bundle.preprocessing_fingerprint
            != config.preprocessing_fingerprint
        ):
            raise RuntimeError("preprocessing differs from decoder/calibration")
        if protocol.occupancy_config != config.occupancy_config:
            raise RuntimeError("calibrated occupancy differs from decoder training")
        if base.feature_channels != config.decoder_config.feature_channels:
            raise RuntimeError("Base feature channels differ from decoder training")

        probability_shapes = {
            tuple(int(value) for value in row.base_output.probability.shape[1:])
            for row in stage_a_run.d_v_bundle.rows
        }
        feature_shapes = {
            tuple(int(value) for value in row.base_output.feature.shape[1:])
            for row in stage_a_run.d_v_bundle.rows
        }
        if len(probability_shapes) != 1 or len(feature_shapes) != 1:
            raise RuntimeError("D_V Base outputs do not share one tensor-shape contract")
        base_probability_shape = _shape3(
            next(iter(probability_shapes)),
            name="base_probability_shape",
        )
        base_feature_shape = _shape3(
            next(iter(feature_shapes)),
            name="base_feature_shape",
        )

        device = _module_device(base)
        with torch.random.fork_rng(devices=[]):
            runtime_decoder = CURELiteDecoder(config.decoder_config)
        runtime_decoder.load_state_dict(
            {
                name: value.detach().clone()
                for name, value in artifact.decoder.state_dict().items()
            },
            strict=True,
        )
        runtime_decoder.to(device=device, dtype=torch.float32)
        runtime_decoder.requires_grad_(False).eval()
        if decoder_state_fingerprint(runtime_decoder) != (
            artifact.decoder_state_fingerprint
        ):
            raise RuntimeError("private runtime decoder differs from artifact")

        occupancy = OccupancyConfig(**config_to_dict(config.occupancy_config))
        decoder_config = DecoderConfig(**config_to_dict(config.decoder_config))
        base_state_fingerprint = stage_a_run.d_v_bundle.base_state_fingerprint
        if stage_a_run.d_r_bundle.base_state_fingerprint != base_state_fingerprint:
            raise RuntimeError("D_R and D_V caches bind different Base states")
        if frozen_base_state_fingerprint(base) != base_state_fingerprint:
            raise RuntimeError("online Base state differs from Stage-A caches")
        receipt = CalibratedDeploymentReceipt(
            method=method,
            decoder_variant=config.variant,
            base_fingerprint=config.base_fingerprint,
            base_state_fingerprint=base_state_fingerprint,
            preprocessing=preprocessing,
            preprocessing_fingerprint=preprocessing_fingerprint,
            base_probability_shape=base_probability_shape,
            base_feature_shape=base_feature_shape,
            occupancy_config=occupancy,
            residual_threshold=protocol.selected_threshold,
            decoder_config=decoder_config,
            decoder_artifact_fingerprint=artifact.artifact_fingerprint,
            decoder_state_fingerprint=artifact.decoder_state_fingerprint,
            decoder_receipt_sha256=artifact.receipt_sha256,
            threshold_protocol_fingerprint=protocol.receipt_fingerprint,
            stage_a_complete_fingerprint=stage_a_run.complete_fingerprint,
        )
        core = CURELiteModel(base, runtime_decoder, occupancy_config=occupancy)
        core.eval()
        core.requires_grad_(False)
        self._receipt = receipt
        self._core = core
        seal = _CalibratedModelSeal(
            receipt=receipt,
            receipt_fingerprint=receipt.receipt_fingerprint,
            core=core,
            base=base,
            decoder=runtime_decoder,
            base_runtime_signature=_runtime_tensor_signature(base),
            decoder_runtime_signature=_runtime_tensor_signature(runtime_decoder),
            device=device,
        )
        self._verification_token = seal
        self._sealed = True
        self.train(False)

    @property
    def receipt(self) -> CalibratedDeploymentReceipt:
        return self._receipt

    @property
    def method(self) -> CalibratedMethod:
        return self._receipt.method

    @property
    def preprocessing(self) -> PreprocessConfig:
        return self._receipt.preprocessing

    @property
    def occupancy_threshold(self) -> float:
        return self._receipt.occupancy_config.threshold

    @property
    def residual_threshold(self) -> float | None:
        return self._receipt.residual_threshold

    @property
    def base_fingerprint(self) -> str:
        return self._receipt.base_fingerprint

    @property
    def decoder_artifact_fingerprint(self) -> str:
        return self._receipt.decoder_artifact_fingerprint

    @property
    def stage_a_complete_fingerprint(self) -> str:
        return self._receipt.stage_a_complete_fingerprint

    def _seal(self) -> _CalibratedModelSeal:
        seal = self._verification_token
        if type(seal) is not _CalibratedModelSeal:
            raise TypeError(
                "CalibratedCURELiteModel must come from its strict constructor"
            )
        if (
            seal.receipt is not self._receipt
            or seal.core is not self._core
            or seal.base is not self._core.base
            or seal.decoder is not self._core.decoder
        ):
            raise TypeError("calibrated model bound objects were replaced")
        if self._receipt.receipt_fingerprint != seal.receipt_fingerprint:
            raise RuntimeError("calibrated deployment receipt changed")
        return seal

    def _verify_runtime(self) -> _CalibratedModelSeal:
        seal = self._seal()
        if self._core.base.fingerprint != self._receipt.base_fingerprint:
            raise RuntimeError("online Base identity changed")
        if config_to_dict(self._core.occupancy_config) != config_to_dict(
            self._receipt.occupancy_config
        ):
            raise RuntimeError("online occupancy config changed")
        if self.training or any(module.training for module in self._core.modules()):
            raise RuntimeError("calibrated model left inference mode")
        if _runtime_tensor_signature(self._core.base) != seal.base_runtime_signature:
            raise RuntimeError("online Base tensors changed")
        if (
            _runtime_tensor_signature(self._core.decoder)
            != seal.decoder_runtime_signature
        ):
            raise RuntimeError("online decoder tensors changed")
        if decoder_state_fingerprint(self._core.decoder) != (
            self._receipt.decoder_state_fingerprint
        ):
            raise RuntimeError("online decoder differs from artifact state")
        return seal

    def verify_unchanged(self) -> None:
        """Deeply recheck the compact receipt and both live module states."""

        seal = self._verify_runtime()
        self._core.base.validate_preprocessing(self._receipt.preprocessing)
        if frozen_base_state_fingerprint(self._core.base) != (
            self._receipt.base_state_fingerprint
        ):
            raise RuntimeError("online frozen Base state changed")
        if decoder_state_fingerprint(self._core.decoder) != (
            self._receipt.decoder_state_fingerprint
        ):
            raise RuntimeError("online decoder differs from artifact state")

    def _validate_images(self, images: Tensor, device: torch.device) -> None:
        preprocessing = self._receipt.preprocessing
        channels = 1 if preprocessing.color_mode == "L" else 3
        expected = (
            channels,
            preprocessing.height,
            preprocessing.width,
        )
        if not isinstance(images, Tensor) or images.ndim != 4:
            raise ValueError("images must have shape [B,C,H,W]")
        if images.shape[0] < 1 or tuple(images.shape[1:]) != expected:
            raise ValueError(
                "images differ from the calibrated preprocessing grid/channels"
            )
        if images.dtype != torch.float32:
            raise TypeError("calibrated images must be float32")
        if images.device != device:
            raise ValueError("images and the frozen Base must use the same device")
        if not torch.isfinite(images).all():
            raise ValueError("calibrated images must be finite")

    def train(self, mode: bool = True) -> "CalibratedCURELiteModel":
        if not isinstance(mode, bool):
            raise TypeError("mode must be bool")
        if mode:
            raise RuntimeError("CalibratedCURELiteModel is inference-only")
        super().train(False)
        self._core.eval()
        self._core.requires_grad_(False)
        return self

    def requires_grad_(self, requires_grad: bool = True) -> "CalibratedCURELiteModel":
        if not isinstance(requires_grad, bool):
            raise TypeError("requires_grad must be bool")
        if getattr(self, "_sealed", False) and requires_grad:
            raise RuntimeError("CalibratedCURELiteModel is inference-only")
        super().requires_grad_(requires_grad)
        return self

    def _apply(self, fn):  # type: ignore[no-untyped-def]
        if getattr(self, "_sealed", False):
            raise RuntimeError(
                "move the frozen Base before constructing the calibrated model"
            )
        return super()._apply(fn)

    def load_state_dict(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if getattr(self, "_sealed", False):
            raise RuntimeError("calibrated model state is artifact-bound")
        return super().load_state_dict(*args, **kwargs)

    def forward(self, images: Tensor) -> CURELiteOutput:
        seal = self._verify_runtime()
        self._validate_images(images, seal.device)
        with torch.no_grad():
            base_output = self._core.base.extract(images)
        if not isinstance(base_output, FrozenBaseOutput):
            raise TypeError("extract() must return FrozenBaseOutput")
        if tuple(base_output.probability.shape[1:]) != (
            self._receipt.base_probability_shape
        ):
            raise RuntimeError("online Base probability shape differs from Stage-A")
        if tuple(base_output.feature.shape[1:]) != self._receipt.base_feature_shape:
            raise RuntimeError("online Base feature shape differs from Stage-A")
        output = self._core._compose_from_base_output(
            images,
            base_output,
            residual_threshold=self._receipt.residual_threshold,
        )
        self._verify_runtime()
        return output

    def infer(self, images: Tensor) -> CURELiteOutput:
        return self.forward(images)


def build_calibrated_cure_lite_model(
    stage_a_run: LoadedStageARun,
    base: FrozenBaseAdapter,
    *,
    method: CalibratedMethod = "U",
) -> CalibratedCURELiteModel:
    """Construct one compact, strict calibrated inference model."""

    return CalibratedCURELiteModel(stage_a_run, base, method=method)


__all__ = [
    "CalibratedCURELiteModel",
    "CalibratedDeploymentReceipt",
    "CalibratedMethod",
    "build_calibrated_cure_lite_model",
]
