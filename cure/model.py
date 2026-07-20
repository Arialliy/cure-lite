"""Full CURE inference: frozen base, one residual decoder, noisy-OR."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..frozen_base import FrozenBaseAdapter
from .decoder import CUREResidualDecoder
from .descriptors import dilate_mask
from .protocol import CUREProtocol, module_state_fingerprint


def _module_version_signature(
    module: nn.Module,
) -> tuple[tuple[str, str, int, tuple[int, ...], str, str, int], ...]:
    """Cheap mutation guard; full content hashing is reserved for formal gates."""

    entries: list[tuple[str, str, int, tuple[int, ...], str, str, int]] = []
    for kind, tensors in (
        ("parameter", module.named_parameters()),
        ("buffer", module.named_buffers()),
    ):
        for name, value in tensors:
            entries.append(
                (
                    kind,
                    name,
                    id(value),
                    tuple(value.shape),
                    str(value.dtype),
                    str(value.device),
                    int(value._version),
                )
            )
    return tuple(sorted(entries))


def noisy_or(base_probability: Tensor, residual_probability: Tensor) -> Tensor:
    """Fuse two Bernoulli evidence maps without a second learned scorer."""

    if not isinstance(base_probability, Tensor) or not isinstance(residual_probability, Tensor):
        raise TypeError("noisy_or inputs must be tensors")
    if base_probability.shape != residual_probability.shape:
        raise ValueError("noisy_or inputs must have identical shapes")
    if not base_probability.is_floating_point() or base_probability.dtype != residual_probability.dtype:
        raise TypeError("noisy_or inputs must share a floating dtype")
    if base_probability.device != residual_probability.device:
        raise ValueError("noisy_or inputs must share a device")
    if not torch.isfinite(base_probability).all() or not torch.isfinite(residual_probability).all():
        raise ValueError("noisy_or inputs must be finite")
    if torch.any((base_probability < 0.0) | (base_probability > 1.0)) or torch.any(
        (residual_probability < 0.0) | (residual_probability > 1.0)
    ):
        raise ValueError("noisy_or inputs must lie in [0,1]")
    return torch.clamp(
        base_probability + residual_probability - base_probability * residual_probability,
        min=0.0,
        max=1.0,
    )


@dataclass(frozen=True)
class CUREOutput(Mapping[str, Tensor]):
    base_probability: Tensor
    occupancy: Tensor
    exclusion_mask: Tensor
    residual_logits: Tensor
    residual_probability: Tensor
    final_probability: Tensor

    _KEYS = (
        "base_probability",
        "occupancy",
        "exclusion_mask",
        "residual_logits",
        "residual_probability",
        "final_probability",
    )

    def __getitem__(self, key: str) -> Tensor:
        if key not in self._KEYS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)


class CUREModel(nn.Module):
    """Backbone-agnostic counterfactual-uncensoring inference model."""

    def __init__(
        self,
        base: FrozenBaseAdapter,
        decoder: CUREResidualDecoder,
        protocol: CUREProtocol,
    ) -> None:
        super().__init__()
        if not isinstance(base, FrozenBaseAdapter):
            raise TypeError("base must implement FrozenBaseAdapter")
        if not isinstance(decoder, CUREResidualDecoder):
            raise TypeError("decoder must be CUREResidualDecoder")
        if not isinstance(protocol, CUREProtocol):
            raise TypeError("protocol must be CUREProtocol")
        protocol.validate_receipt()
        if base.feature_channels != decoder.feature_channels:
            raise ValueError("base and residual feature channel counts differ")
        if decoder.config != protocol.residual_config:
            raise ValueError("decoder and frozen CURE protocol configurations differ")
        if base.fingerprint != protocol.adapter_fingerprint:
            raise ValueError("base adapter differs from the frozen CURE protocol")
        self.base = base
        self.decoder = decoder
        self.protocol = protocol
        self._freeze_base()
        # ``FrozenBaseAdapter.fingerprint`` is a semantic identity supplied by
        # the adapter.  It cannot reveal an in-place parameter/buffer mutation
        # when a buggy adapter keeps returning the same identity string.  Keep
        # an independent content receipt and check it at every output consumer
        # boundary.
        self._base_adapter_identity = id(base)
        self._base_adapter_fingerprint = base.fingerprint
        self._base_state_fingerprint = module_state_fingerprint(base)
        if self._base_state_fingerprint != protocol.base_state_fingerprint:
            raise ValueError(
                "frozen base contents differ from the frozen CURE protocol"
            )
        self._base_version_signature = _module_version_signature(base)

    def _freeze_base(self) -> None:
        self.base.requires_grad_(False)
        self.base.eval()

    def _assert_base_immutable(self, *, full_content: bool = False) -> None:
        if id(self.base) != self._base_adapter_identity:
            raise RuntimeError(
                "frozen base adapter was replaced after CUREModel construction"
            )
        if self.base.fingerprint != self._base_adapter_fingerprint:
            raise RuntimeError(
                "frozen base semantic fingerprint changed after CUREModel construction"
            )
        if _module_version_signature(self.base) != self._base_version_signature:
            raise RuntimeError(
                "frozen base parameters or buffers changed after CUREModel construction"
            )
        if (
            full_content
            and module_state_fingerprint(self.base) != self._base_state_fingerprint
        ):
            raise RuntimeError(
                "frozen base contents changed after CUREModel construction"
            )

    @property
    def base_state_fingerprint(self) -> str:
        """Exact construction-time parameter/buffer receipt for the base."""

        self._assert_base_immutable(full_content=True)
        return self._base_state_fingerprint

    def train(self, mode: bool = True) -> "CUREModel":
        super().train(mode)
        self._freeze_base()
        self.decoder.train(mode)
        return self

    def trainable_parameters(self):
        return self.decoder.parameters()

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load decoder state while refusing any changed frozen-base tensor."""

        if not isinstance(state_dict, Mapping):
            raise TypeError("state_dict must be a mapping")
        self._assert_base_immutable()
        if any(name.startswith("base.") for name in state_dict):
            raise RuntimeError(
                "CUREModel.load_state_dict may not target the frozen base; "
                "load a decoder checkpoint through model.decoder"
            )
        result = super().load_state_dict(
            state_dict, strict=strict, assign=assign
        )
        self._freeze_base()
        self._assert_base_immutable()
        return result

    def forward(
        self,
        images: Tensor,
    ) -> CUREOutput:
        self.protocol.validate_receipt()
        self._assert_base_immutable()
        if not isinstance(images, Tensor) or images.ndim != 4:
            raise ValueError("images must have shape [B,C,H,W]")
        if images.shape[0] < 1 or not images.is_floating_point():
            raise ValueError("images must be a non-empty floating batch")
        self._freeze_base()
        with torch.no_grad():
            base_output = self.base.extract(images)
        # A custom adapter may mutate a running buffer inside ``extract`` even
        # in eval/no-grad mode.  Reject that output before it reaches CURE.
        self._assert_base_immutable()
        self.base.validate_output(base_output, images)
        base_probability = base_output.probability.detach()
        feature = base_output.feature.detach()
        occupancy = base_probability >= self.decoder.config.occupancy_threshold
        residual_logits = self.decoder(feature, base_probability, occupancy)
        exclusion = dilate_mask(occupancy, self.decoder.config.suppression_radius)
        residual_probability = torch.sigmoid(residual_logits).masked_fill(exclusion, 0.0)
        final_probability = noisy_or(base_probability, residual_probability)
        return CUREOutput(
            base_probability=base_probability,
            occupancy=occupancy,
            exclusion_mask=exclusion,
            residual_logits=residual_logits,
            residual_probability=residual_probability,
            final_probability=final_probability,
        )


__all__ = ["CUREModel", "CUREOutput", "noisy_or"]
