"""Canonical, content-addressed frozen-source records for full CURE.

The eligible intervention universe must be derived from an actual execution of
the adapter frozen into :class:`CUREProtocol`.  Accepting already materialized
feature/probability tensors at the catalog boundary would only attest to their
post-hoc integrity; it would not attest to where they came from.  This module
closes that gap by making adapter execution the only issuance route.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import torch
from torch import Tensor

from ..frozen_base import FrozenBaseAdapter
from ..splits import SplitManifest
from ..types import Instance, InstanceMap
from .protocol import (
    CUREProtocol,
    frozen_output_fingerprint,
    module_state_fingerprint,
    tensor_content_fingerprint,
)


_FROZEN_SOURCE_RECORD_SEAL = object()


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _clone_instance_map(value: InstanceMap) -> InstanceMap:
    """Deep-copy every tensor that can affect target geometry."""

    return InstanceMap(
        labels=value.labels.detach().to(device="cpu", dtype=torch.int64).clone(),
        instances=tuple(
            Instance(
                instance_id=item.instance_id,
                mask=item.mask.detach().to(device="cpu", dtype=torch.bool).clone(),
                area=item.area,
                bbox=item.bbox,
                centroid=item.centroid,
            )
            for item in value.instances
        ),
    )


@dataclass(frozen=True)
class FrozenSourceRecord:
    """One exact D_R input, GT map, and frozen-adapter execution receipt.

    All tensors are detached CPU snapshots.  ``intensity`` is deliberately a
    view of input channel zero rather than a caller-provided map, so descriptor
    construction cannot silently switch intensity preprocessing.  The seal is
    recomputed from tensor contents at every consumption boundary and therefore
    detects in-place mutation despite PyTorch tensors themselves being mutable.
    """

    sample_id: str
    group_id: str
    base_fingerprint: str
    adapter_fingerprint: str
    adapter_state_fingerprint: str
    preprocessing_fingerprint: str
    protocol_fingerprint: str
    manifest_fingerprint: str
    feature_channels: int
    image: Tensor
    feature: Tensor
    probability: Tensor
    gt: InstanceMap
    input_fingerprint: str
    frozen_output_fingerprint: str
    gt_fingerprint: str
    split_role: str = "D_R"
    schema_version: str = "cure-frozen-source-v1"
    _seal: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        issuing = self._seal is _FROZEN_SOURCE_RECORD_SEAL
        if not issuing and not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _FROZEN_SOURCE_RECORD_SEAL
        ):
            raise ValueError(
                "frozen source record was not issued by canonical adapter execution"
            )
        self._validate_content()
        if issuing:
            object.__setattr__(
                self,
                "_seal",
                (_FROZEN_SOURCE_RECORD_SEAL, self.fingerprint),
            )
        elif self._seal[1] != self.fingerprint:
            raise ValueError("frozen source record content differs from its receipt")

    @property
    def intensity(self) -> Tensor:
        """The exact preprocessed input channel used by source descriptors."""

        return self.image[0, 0]

    def _validate_content(self) -> None:
        for name in (
            "sample_id",
            "group_id",
            "base_fingerprint",
            "adapter_fingerprint",
            "preprocessing_fingerprint",
        ):
            _nonempty(name, getattr(self, name))
        for name in (
            "protocol_fingerprint",
            "manifest_fingerprint",
            "adapter_state_fingerprint",
            "input_fingerprint",
            "frozen_output_fingerprint",
            "gt_fingerprint",
        ):
            _digest(name, getattr(self, name))
        if self.split_role != "D_R":
            raise ValueError("frozen source records are source-only D_R artifacts")
        if self.schema_version != "cure-frozen-source-v1":
            raise ValueError(f"unsupported frozen-source schema {self.schema_version!r}")
        if (
            isinstance(self.feature_channels, bool)
            or not isinstance(self.feature_channels, int)
            or self.feature_channels < 1
        ):
            raise ValueError("feature_channels must be a positive integer")
        if (
            not isinstance(self.image, Tensor)
            or self.image.ndim != 4
            or self.image.shape[0] != 1
            or self.image.shape[1] < 1
        ):
            raise ValueError("image must have canonical shape [1,C,H,W]")
        if (
            self.image.device.type != "cpu"
            or self.image.dtype != torch.float32
            or self.image.requires_grad
        ):
            raise TypeError("canonical source image must be detached CPU float32")
        if (
            not isinstance(self.feature, Tensor)
            or self.feature.ndim != 4
            or self.feature.shape[:2] != (1, self.feature_channels)
        ):
            raise ValueError(
                "feature must have shape [1,feature_channels,h,w]"
            )
        if (
            self.feature.device.type != "cpu"
            or self.feature.dtype != torch.float32
            or self.feature.requires_grad
        ):
            raise TypeError("canonical source feature must be detached CPU float32")
        if (
            not isinstance(self.probability, Tensor)
            or self.probability.ndim != 4
            or self.probability.shape[:2] != (1, 1)
        ):
            raise ValueError("probability must have canonical shape [1,1,H,W]")
        if (
            self.probability.device.type != "cpu"
            or self.probability.dtype != torch.float32
            or self.probability.requires_grad
        ):
            raise TypeError("canonical source probability must be detached CPU float32")
        if not torch.isfinite(self.image).all():
            raise ValueError("source image contains non-finite values")
        if not torch.isfinite(self.feature).all():
            raise ValueError("source feature contains non-finite values")
        if not torch.isfinite(self.probability).all() or torch.any(
            (self.probability < 0.0) | (self.probability > 1.0)
        ):
            raise ValueError("source probability must be finite and lie in [0,1]")
        if not isinstance(self.gt, InstanceMap):
            raise TypeError("gt must be an InstanceMap")
        if self.probability.shape[-2:] != self.image.shape[-2:]:
            raise ValueError("source probability must use the exact input grid")
        if tuple(self.probability.shape[-2:]) != self.gt.shape:
            raise ValueError("source probability and GT must share a grid")
        if self.input_fingerprint != tensor_content_fingerprint(
            "input_image", self.image
        ):
            raise ValueError("source image differs from its input fingerprint")
        if self.gt_fingerprint != tensor_content_fingerprint(
            "gt_labels", self.gt.labels
        ):
            raise ValueError("source GT differs from its GT fingerprint")

    @property
    def fingerprint(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(
            repr(
                (
                    self.schema_version,
                    self.sample_id,
                    self.group_id,
                    self.split_role,
                    self.base_fingerprint,
                    self.adapter_fingerprint,
                    self.adapter_state_fingerprint,
                    self.preprocessing_fingerprint,
                    self.protocol_fingerprint,
                    self.manifest_fingerprint,
                    self.feature_channels,
                    self.input_fingerprint,
                    self.frozen_output_fingerprint,
                    self.gt_fingerprint,
                )
            ).encode("utf-8")
        )
        for name, value in (
            ("input_image", self.image),
            ("feature", self.feature),
            ("probability", self.probability),
            ("gt_labels", self.gt.labels),
        ):
            hasher.update(tensor_content_fingerprint(name, value).encode("ascii"))
        for item in self.gt.instances:
            hasher.update(
                repr(
                    (
                        item.instance_id,
                        item.area,
                        item.bbox,
                        item.centroid,
                    )
                ).encode("utf-8")
            )
            hasher.update(
                tensor_content_fingerprint(
                    f"gt_instance_{item.instance_id}", item.mask
                ).encode("ascii")
            )
        return hasher.hexdigest()

    def validate_receipt(self) -> None:
        self._validate_content()
        if not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _FROZEN_SOURCE_RECORD_SEAL
            and self._seal[1] == self.fingerprint
        ):
            raise ValueError("frozen source record content differs from its receipt")

    def validate_against(
        self,
        protocol: CUREProtocol,
        manifest: SplitManifest,
    ) -> None:
        """Validate the complete identity chain at a consuming boundary."""

        self.validate_receipt()
        if not isinstance(protocol, CUREProtocol):
            raise TypeError("protocol must be CUREProtocol")
        protocol.validate_manifest(manifest)
        protocol.assert_sample(
            self.sample_id,
            split="D_R",
            group_id=self.group_id,
        )
        expected = (
            protocol.base_fingerprint,
            protocol.adapter_fingerprint,
            protocol.base_state_fingerprint,
            protocol.preprocessing_fingerprint,
            protocol.fingerprint,
            protocol.manifest_fingerprint,
            protocol.residual_config.feature_channels,
        )
        actual = (
            self.base_fingerprint,
            self.adapter_fingerprint,
            self.adapter_state_fingerprint,
            self.preprocessing_fingerprint,
            self.protocol_fingerprint,
            self.manifest_fingerprint,
            self.feature_channels,
        )
        if actual != expected:
            raise ValueError("frozen source record differs from the CURE protocol")
        expected_output = frozen_output_fingerprint(
            protocol,
            self.sample_id,
            self.feature,
            self.probability,
        )
        if self.frozen_output_fingerprint != expected_output:
            raise ValueError("frozen adapter output differs from its output fingerprint")


def extract_frozen_source_record(
    *,
    base: FrozenBaseAdapter,
    images: Tensor,
    gt: InstanceMap,
    sample_id: str,
    group_id: str,
    protocol: CUREProtocol,
    manifest: SplitManifest,
) -> FrozenSourceRecord:
    """Run the protocol-bound adapter and issue one immutable source receipt.

    The batch size is deliberately fixed to one so sample/group/GT identity is
    unambiguous.  The adapter is run on a private exact-value snapshot; mutation
    of either the caller's tensor or the snapshot during adapter execution
    cannot silently alter the issued record.
    """

    if not isinstance(base, FrozenBaseAdapter):
        raise TypeError("base must implement FrozenBaseAdapter")
    if not isinstance(protocol, CUREProtocol):
        raise TypeError("protocol must be CUREProtocol")
    if not isinstance(gt, InstanceMap):
        raise TypeError("gt must be InstanceMap")
    protocol.validate_manifest(manifest)
    protocol.assert_sample(sample_id, split="D_R", group_id=group_id)
    if base.fingerprint != protocol.adapter_fingerprint:
        raise ValueError("base adapter differs from the frozen CURE protocol")
    if base.feature_channels != protocol.residual_config.feature_channels:
        raise ValueError("base feature channels differ from the CURE protocol")
    if (
        not isinstance(images, Tensor)
        or images.ndim != 4
        or images.shape[0] != 1
        or images.shape[1] < 1
    ):
        raise ValueError("images must have shape [1,C,H,W] with C>=1")
    if images.dtype != torch.float32 or images.requires_grad:
        raise TypeError("canonical source images must be detached float32")
    if not torch.isfinite(images).all():
        raise ValueError("source images contain non-finite values")
    if tuple(images.shape[-2:]) != gt.shape:
        raise ValueError("source image and GT must share a grid")

    adapter_state_before = module_state_fingerprint(base)
    semantic_fingerprint_before = base.fingerprint
    if adapter_state_before != protocol.base_state_fingerprint:
        raise ValueError(
            "base adapter parameter/buffer state differs from the frozen CURE protocol"
        )
    exact_input = images.detach().clone()
    input_before = exact_input.clone()
    output = base(exact_input)
    if not torch.equal(exact_input, input_before):
        raise RuntimeError("frozen adapter modified its input in place")
    if (
        module_state_fingerprint(base) != adapter_state_before
        or base.fingerprint != semantic_fingerprint_before
    ):
        raise RuntimeError("frozen adapter identity or module state changed during extraction")

    image_snapshot = input_before.to(device="cpu").contiguous().clone()
    feature_snapshot = (
        output.feature.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
    )
    probability_snapshot = (
        output.probability.detach()
        .to(device="cpu", dtype=torch.float32)
        .contiguous()
        .clone()
    )
    gt_snapshot = _clone_instance_map(gt)
    input_fingerprint = tensor_content_fingerprint("input_image", image_snapshot)
    gt_fingerprint = tensor_content_fingerprint("gt_labels", gt_snapshot.labels)
    output_fingerprint = frozen_output_fingerprint(
        protocol,
        sample_id,
        feature_snapshot,
        probability_snapshot,
    )
    return FrozenSourceRecord(
        sample_id=sample_id,
        group_id=group_id,
        base_fingerprint=protocol.base_fingerprint,
        adapter_fingerprint=semantic_fingerprint_before,
        adapter_state_fingerprint=adapter_state_before,
        preprocessing_fingerprint=protocol.preprocessing_fingerprint,
        protocol_fingerprint=protocol.fingerprint,
        manifest_fingerprint=manifest.fingerprint,
        feature_channels=base.feature_channels,
        image=image_snapshot,
        feature=feature_snapshot,
        probability=probability_snapshot,
        gt=gt_snapshot,
        input_fingerprint=input_fingerprint,
        frozen_output_fingerprint=output_fingerprint,
        gt_fingerprint=gt_fingerprint,
        _seal=_FROZEN_SOURCE_RECORD_SEAL,
    )


__all__ = ["FrozenSourceRecord", "extract_frozen_source_record"]
