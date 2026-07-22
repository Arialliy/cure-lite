"""Strict, non-invasive adapter for one pinned official MSHNet revision."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn

from ..cache.schema import (
    BASE_CACHE_SCHEMA,
    build_base_fingerprint,
    file_sha256,
    stable_fingerprint,
)
from ..data import PreprocessConfig
from ..frozen_base import FrozenBaseAdapter
from ..types import FrozenBaseOutput


def _git_value(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"unable to inspect pinned MSHNet repository: {result.stderr.strip()}"
        )
    value = result.stdout.strip()
    if not value:
        raise RuntimeError("git returned an empty MSHNet identity")
    return value


def _load_mshnet_type(source: Path, source_sha256: str) -> type[nn.Module]:
    module_name = f"_cure_lite_pinned_mshnet_{source_sha256[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load pinned model/MSHNet.py")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    model_type = getattr(module, "MSHNet", None)
    if not isinstance(model_type, type) or not issubclass(model_type, nn.Module):
        raise RuntimeError("pinned model/MSHNet.py does not expose nn.Module MSHNet")
    return model_type


class MSHNetAdapter(FrozenBaseAdapter):
    """Own and expose a strictly loaded, frozen MSHNet checkpoint.

    The adapter verifies the Git revision, tree, source bytes, checkpoint bytes,
    strict state-dict compatibility, and preprocessing contract before it
    registers the feature hook. It never edits or imports from the upstream
    package namespace.
    """

    FEATURE_MODULE_NAME = "decoder_0"
    FEATURE_CHANNELS = 16
    FEATURE_STRIDE = 1
    INPUT_CHANNELS = 3
    ADAPTER_VERSION = "cure-lite-mshnet-adapter-v0.2"
    PINNED_UPSTREAM_COMMIT = "46cdfd46802629da51f70124662af7335be74b56"
    PINNED_UPSTREAM_TREE = "f3f53b5135c8ed402109d18ffca1f21eb261a418"
    PINNED_MODEL_SOURCE_SHA256 = (
        "2cb87bbc2c8cd6d7053df9ffb4c0ea7f01acf65c4d1750dd93ac639a94e44c0e"
    )
    FORWARD_KWARGS = {"warm_flag": True}
    OUTPUT_SELECTOR = "strict_tuple[1]"
    NATIVE_MEAN = (0.485, 0.456, 0.406)
    NATIVE_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        repository: str | Path,
        checkpoint_path: str | Path,
        *,
        expected_checkpoint_sha256: str,
        base_training_provenance_fingerprint: str,
        base_training_final_receipt_sha256: str,
        preprocessing: PreprocessConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        repository_path = Path(repository).expanduser().resolve(strict=True)
        checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
        if not repository_path.is_dir() or not checkpoint.is_file():
            raise ValueError("repository must be a directory and checkpoint a file")
        if not isinstance(preprocessing, PreprocessConfig):
            raise TypeError("preprocessing must be PreprocessConfig")
        if preprocessing.color_mode != "RGB":
            raise ValueError("pinned MSHNet requires RGB preprocessing")
        if (
            preprocessing.mean != self.NATIVE_MEAN
            or preprocessing.std != self.NATIVE_STD
        ):
            raise ValueError("pinned MSHNet requires its native ImageNet normalization")
        if preprocessing.height != preprocessing.width:
            raise ValueError("pinned MSHNet evaluation resize must be square")
        if preprocessing.height % 16 != 0:
            raise ValueError("pinned MSHNet evaluation grid must be divisible by 16")
        resolved_device = torch.device(device)

        upstream_commit = _git_value(repository_path, "rev-parse", "HEAD")
        upstream_tree = _git_value(repository_path, "rev-parse", "HEAD^{tree}")
        if upstream_commit != self.PINNED_UPSTREAM_COMMIT:
            raise RuntimeError(
                "MSHNet commit mismatch: "
                f"expected {self.PINNED_UPSTREAM_COMMIT}, got {upstream_commit}"
            )
        if upstream_tree != self.PINNED_UPSTREAM_TREE:
            raise RuntimeError(
                "MSHNet tree mismatch: "
                f"expected {self.PINNED_UPSTREAM_TREE}, got {upstream_tree}"
            )
        source = (repository_path / "model" / "MSHNet.py").resolve(strict=True)
        model_source_sha256 = file_sha256(source)
        if model_source_sha256 != self.PINNED_MODEL_SOURCE_SHA256:
            raise RuntimeError("pinned model/MSHNet.py content SHA256 mismatch")

        checkpoint_sha256 = file_sha256(checkpoint)
        if checkpoint_sha256 != str(expected_checkpoint_sha256).lower():
            raise RuntimeError("checkpoint SHA256 differs from the expected digest")
        model_type = _load_mshnet_type(source, model_source_sha256)
        model = model_type(self.INPUT_CHANNELS)
        try:
            checkpoint_object = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as error:
            raise RuntimeError("unable to load the MSHNet checkpoint safely") from error
        if not isinstance(checkpoint_object, Mapping) or not checkpoint_object:
            raise TypeError("MSHNet weight file must be a non-empty raw state_dict")
        if any(
            not isinstance(name, str) or not isinstance(value, Tensor)
            for name, value in checkpoint_object.items()
        ):
            raise TypeError("MSHNet weight file must contain only tensor state entries")
        try:
            incompatible = model.load_state_dict(checkpoint_object, strict=True)
        except RuntimeError as error:
            raise RuntimeError("MSHNet checkpoint is not strictly compatible") from error
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("strict MSHNet checkpoint load returned incompatible keys")
        if file_sha256(checkpoint) != checkpoint_sha256:
            raise RuntimeError("MSHNet checkpoint changed while it was being loaded")
        if file_sha256(source) != model_source_sha256:
            raise RuntimeError("model/MSHNet.py changed while constructing the adapter")
        if _git_value(repository_path, "rev-parse", "HEAD") != upstream_commit:
            raise RuntimeError("MSHNet commit changed while constructing the adapter")
        if _git_value(repository_path, "rev-parse", "HEAD^{tree}") != upstream_tree:
            raise RuntimeError("MSHNet tree changed while constructing the adapter")

        model.to(resolved_device)
        first_tensor = next(model.parameters(), None)
        if first_tensor is None:
            first_tensor = next(model.buffers(), None)
        if first_tensor is None:
            raise RuntimeError("pinned MSHNet unexpectedly has no parameters or buffers")
        actual_device = first_tensor.device
        super().__init__(model)
        preprocessing_payload = preprocessing.fingerprint_payload()
        self.repository = repository_path
        self.checkpoint_path = checkpoint
        self.checkpoint_sha256 = checkpoint_sha256
        self.upstream_commit = upstream_commit
        self.upstream_tree = upstream_tree
        self.model_source_sha256 = model_source_sha256
        self.base_training_provenance_fingerprint = (
            base_training_provenance_fingerprint
        )
        self.base_training_final_receipt_sha256 = (
            base_training_final_receipt_sha256
        )
        self.preprocessing = preprocessing
        # Bare devices such as ``cuda`` resolve to a concrete index after
        # ``Module.to``; CPU tensors similarly canonicalize ``cpu:0`` to
        # ``cpu``.  Validate inputs against the model's actual device rather
        # than the caller's spelling of the request.
        self.device = actual_device
        self._fingerprint = build_base_fingerprint(
            schema_version=BASE_CACHE_SCHEMA,
            checkpoint_sha256=checkpoint_sha256,
            adapter_version=self.ADAPTER_VERSION,
            upstream_commit=upstream_commit,
            upstream_tree=upstream_tree,
            model_source_sha256=model_source_sha256,
            base_training_provenance_fingerprint=(
                base_training_provenance_fingerprint
            ),
            base_training_final_receipt_sha256=(
                base_training_final_receipt_sha256
            ),
            preprocessing=preprocessing_payload,
            preprocessing_fingerprint=stable_fingerprint(preprocessing_payload),
            feature_module_name=self.FEATURE_MODULE_NAME,
            feature_channels=self.FEATURE_CHANNELS,
            feature_stride=self.FEATURE_STRIDE,
            forward_kwargs=self.FORWARD_KWARGS,
            output_selector=self.OUTPUT_SELECTOR,
        )

        module = dict(self.base.named_modules()).get(self.FEATURE_MODULE_NAME)
        if module is None:
            raise KeyError("pinned MSHNet has no decoder_0 module")
        self._feature: Tensor | None = None
        self._feature_capture_count = 0
        self._extracting = False
        self._closed = False
        self._base_signature = self._capture_base_signature()
        self._base_content_sha256 = self._base_content_digest()
        self._hook_handle = module.register_forward_hook(self._capture_decoder_0)

    @property
    def feature_channels(self) -> int:
        return self.FEATURE_CHANNELS

    @property
    def feature_stride(self) -> int:
        return self.FEATURE_STRIDE

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def validate_preprocessing(self, preprocessing: object) -> None:
        if not isinstance(preprocessing, PreprocessConfig):
            raise TypeError("MSHNet preprocessing must be a PreprocessConfig")
        if preprocessing != self.preprocessing:
            raise ValueError(
                "dataset preprocessing differs from the fingerprinted MSHNet profile"
            )

    def _capture_base_signature(self) -> dict[str, tuple[object, ...]]:
        tensors = {
            **dict(self.base.named_parameters()),
            **dict(self.base.named_buffers()),
        }
        return {
            name: (
                id(value),
                value._version,
                tuple(value.shape),
                value.dtype,
                value.device,
            )
            for name, value in tensors.items()
        }

    def _validate_base_signature(self) -> None:
        if self._capture_base_signature() != self._base_signature:
            raise RuntimeError("frozen MSHNet parameters or buffers changed")

    def _base_content_digest(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.base.state_dict().items()):
            tensor = value.detach().to(device="cpu").contiguous()
            encoded_name = name.encode("utf-8")
            encoded_dtype = str(tensor.dtype).encode("ascii")
            digest.update(len(encoded_name).to_bytes(4, "big"))
            digest.update(encoded_name)
            digest.update(len(encoded_dtype).to_bytes(2, "big"))
            digest.update(encoded_dtype)
            digest.update(len(tensor.shape).to_bytes(2, "big"))
            for dimension in tensor.shape:
                digest.update(int(dimension).to_bytes(8, "big", signed=False))
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
        return digest.hexdigest()

    def assert_base_unchanged(self, *, deep: bool = False) -> None:
        """Reject any mutation of the frozen model.

        The inexpensive identity/version check runs around every extraction.
        ``deep=True`` additionally hashes all state values and catches direct
        ``Tensor.data`` writes that bypass PyTorch's version counter.
        """

        self._validate_base_signature()
        if deep and self._base_content_digest() != self._base_content_sha256:
            raise RuntimeError("frozen MSHNet parameter or buffer values changed")

    def _capture_decoder_0(
        self,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> None:
        del module, inputs
        if not self._extracting:
            raise RuntimeError("MSHNet feature hook fired outside adapter.extract()")
        if not isinstance(output, Tensor):
            raise TypeError("decoder_0 hook output must be a Tensor")
        if output.ndim != 4 or output.shape[1] != self.FEATURE_CHANNELS:
            raise ValueError("decoder_0 must output [B,16,H,W]")
        self._feature_capture_count += 1
        if self._feature_capture_count != 1:
            raise RuntimeError("decoder_0 feature hook fired more than once")
        self._feature = output.detach()
        return None

    @staticmethod
    def _validate_pinned_output(raw_output: Any, batch_size: int) -> Tensor:
        if not isinstance(raw_output, tuple) or len(raw_output) != 2:
            raise TypeError("pinned MSHNet must return (auxiliary_masks, final_logits)")
        auxiliary_masks, logits = raw_output
        if not isinstance(auxiliary_masks, list) or len(auxiliary_masks) != 4:
            raise TypeError("warm_flag=True must return four auxiliary mask logits")
        for item in auxiliary_masks:
            if not isinstance(item, Tensor) or item.ndim != 4:
                raise TypeError("MSHNet auxiliary outputs must be NCHW tensors")
            if item.shape[0] != batch_size or item.shape[1] != 1:
                raise ValueError("MSHNet auxiliary outputs must be [B,1,h,w]")
            if not item.is_floating_point() or not torch.isfinite(item).all():
                raise ValueError("MSHNet auxiliary outputs must be finite floating tensors")
        if not isinstance(logits, Tensor):
            raise TypeError("pinned MSHNet final logits must be a Tensor")
        if logits.ndim != 4 or logits.shape[:2] != (batch_size, 1):
            raise ValueError("MSHNet final logits must be [B,1,H,W]")
        if not logits.is_floating_point() or not torch.isfinite(logits).all():
            raise ValueError("MSHNet final logits must be finite floating point")
        return logits

    def preprocess(self, image: str | Path | Image.Image) -> Tensor:
        """Apply the exact fingerprinted MSHNet image preprocessing."""

        if isinstance(image, (str, Path)):
            with Image.open(Path(image).expanduser().resolve(strict=True)) as loaded:
                source = loaded.convert(self.preprocessing.color_mode)
        elif isinstance(image, Image.Image):
            source = image.convert(self.preprocessing.color_mode)
        else:
            raise TypeError("image must be a path or PIL.Image.Image")
        resized = source.resize(
            (self.preprocessing.width, self.preprocessing.height),
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(resized, dtype=np.float32)
        if array.ndim == 2:
            array = array[..., None]
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1) / 255.0
        mean = torch.tensor(self.preprocessing.mean, dtype=torch.float32)[:, None, None]
        std = torch.tensor(self.preprocessing.std, dtype=torch.float32)[:, None, None]
        return ((tensor - mean) / std).unsqueeze(0).to(self.device).contiguous()

    def _validate_images(self, images: Tensor) -> None:
        if not isinstance(images, Tensor) or images.ndim != 4:
            raise ValueError("images must be [B,C,H,W]")
        expected = (
            self.INPUT_CHANNELS,
            self.preprocessing.height,
            self.preprocessing.width,
        )
        if tuple(images.shape[1:]) != expected or images.shape[0] < 1:
            raise ValueError(
                "images do not match fingerprinted MSHNet channels/evaluation grid"
            )
        if images.dtype != torch.float32:
            raise TypeError("pinned MSHNet adapter currently requires FP32 images")
        if images.device != self.device:
            raise ValueError("images and pinned MSHNet must share a device")
        if not torch.isfinite(images).all():
            raise ValueError("preprocessed MSHNet images must be finite")

    def extract(self, images: Tensor) -> FrozenBaseOutput:
        if self._closed:
            raise RuntimeError("MSHNetAdapter is closed")
        if self._extracting:
            raise RuntimeError("MSHNetAdapter does not support re-entrant extraction")
        self._validate_images(images)
        self._validate_base_signature()
        self.base.requires_grad_(False)
        self.base.eval()
        self._feature = None
        self._feature_capture_count = 0
        self._extracting = True
        try:
            with torch.no_grad():
                raw_output = self.base(images, warm_flag=True)
                logits = self._validate_pinned_output(raw_output, images.shape[0])
        finally:
            self._extracting = False
        feature = self._feature
        if feature is None or self._feature_capture_count != 1:
            raise RuntimeError("decoder_0 hook did not fire exactly once")
        if logits.shape[-2:] != images.shape[-2:]:
            raise ValueError("MSHNet output grid mismatch")
        if feature.shape[0] != images.shape[0]:
            raise ValueError("decoder_0 feature batch does not match input batch")
        if feature.shape[-2:] != images.shape[-2:]:
            raise ValueError("decoder_0 feature must be full resolution")
        output = FrozenBaseOutput(
            probability=torch.sigmoid(logits.float()).detach(),
            feature=feature.detach(),
        )
        self.validate_output(output, images)
        self._validate_base_signature()
        return output

    def train(self, mode: bool = True) -> "MSHNetAdapter":
        del mode
        nn.Module.train(self, False)
        self.requires_grad_(False)
        self.base.eval()
        return self

    def close(self) -> None:
        """Audit final frozen state and idempotently remove the feature hook."""

        if not self._closed:
            try:
                self.assert_base_unchanged(deep=True)
            finally:
                self._hook_handle.remove()
                self._closed = True
                self._feature = None

    def __enter__(self) -> "MSHNetAdapter":
        if self._closed:
            raise RuntimeError("cannot re-enter a closed MSHNetAdapter")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()


__all__ = ["MSHNetAdapter"]
