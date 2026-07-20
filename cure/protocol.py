"""One canonical protocol receipt for every full-CURE artifact.

The method is defined by an intervention distribution, not by whichever
combination of low-level helpers a caller happens to invoke.  This module binds
the data manifest, frozen predictor semantics, coverage rules, descriptor
schema, and OOF estimator configuration into one content-addressed receipt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping

from torch import Tensor
from torch import nn

from ..config import InterventionConfig, MatchConfig
from ..provenance import BaseCheckpointSelection
from ..splits import ALL_SPLITS, SplitManifest
from .config import (
    CURELossConfig,
    CUREResidualConfig,
    DescriptorConfig,
    PropensityConfig,
)


_CURE_PROTOCOL_SEAL = object()


def _nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _sha256_digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CUREProtocol:
    """Self-contained, versioned definition of one full-CURE experiment.

    ``manifest_membership`` deliberately records only identities and grouping,
    while ``manifest_fingerprint`` binds the complete manifest payload.  This
    keeps state validation cheap without weakening the provenance receipt.
    """

    base_fingerprint: str
    adapter_fingerprint: str
    base_state_fingerprint: str
    preprocessing_fingerprint: str
    manifest_fingerprint: str
    manifest_membership: tuple[tuple[str, str, str], ...]
    base_checkpoint_selection: BaseCheckpointSelection
    residual_config: CUREResidualConfig
    loss_config: CURELossConfig = CURELossConfig()
    match_config: MatchConfig = MatchConfig()
    intervention_config: InterventionConfig = InterventionConfig()
    descriptor_config: DescriptorConfig = DescriptorConfig()
    propensity_config: PropensityConfig = PropensityConfig()
    probability_semantics: str = "foreground-probability-v1"
    feature_semantics: str = "adapter-feature-v1"
    schema_version: str = "cure-protocol-v2"
    _seal: object = field(
        default=None,
        repr=False,
        compare=False,
    )
    _cached_fingerprint: str = field(
        default="",
        init=False,
        repr=False,
        compare=False,
    )
    _sample_index: Mapping[str, tuple[str, str]] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        issuing = self._seal is _CURE_PROTOCOL_SEAL
        if not issuing and not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _CURE_PROTOCOL_SEAL
        ):
            raise ValueError("CURE protocol was not issued by from_manifest")
        for name in (
            "base_fingerprint",
            "adapter_fingerprint",
            "preprocessing_fingerprint",
            "manifest_fingerprint",
            "probability_semantics",
            "feature_semantics",
        ):
            _nonempty(name, getattr(self, name))
        _sha256_digest("base_state_fingerprint", self.base_state_fingerprint)
        if self.schema_version != "cure-protocol-v2":
            raise ValueError(f"unsupported CURE protocol {self.schema_version!r}")
        expected_types = (
            ("residual_config", self.residual_config, CUREResidualConfig),
            ("loss_config", self.loss_config, CURELossConfig),
            ("match_config", self.match_config, MatchConfig),
            ("intervention_config", self.intervention_config, InterventionConfig),
            ("descriptor_config", self.descriptor_config, DescriptorConfig),
            ("propensity_config", self.propensity_config, PropensityConfig),
        )
        for name, value, expected in expected_types:
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        rows = self.manifest_membership
        if not rows or rows != tuple(sorted(set(rows))):
            raise ValueError("manifest_membership must be non-empty, unique, and sorted")
        sample_ids: set[str] = set()
        for sample_id, split, group_id in rows:
            _nonempty("sample_id", sample_id)
            _nonempty("group_id", group_id)
            if split not in ALL_SPLITS:
                raise ValueError(f"invalid split role {split!r}")
            if sample_id in sample_ids:
                raise ValueError(f"duplicate protocol sample_id {sample_id!r}")
            sample_ids.add(sample_id)
        present = {split for _, split, _ in rows}
        if present != set(ALL_SPLITS):
            raise ValueError("formal CURE protocol must contain all four split roles")
        if not isinstance(self.base_checkpoint_selection, BaseCheckpointSelection):
            raise TypeError(
                "base_checkpoint_selection must be BaseCheckpointSelection"
            )
        if (
            self.base_checkpoint_selection.split_manifest_fingerprint
            != self.manifest_fingerprint
        ):
            raise ValueError("base checkpoint selection uses a different manifest")
        d_b_ids = {sample_id for sample_id, split, _ in rows if split == "D_B"}
        selected_ids = set(self.base_checkpoint_selection.fit_sample_ids) | set(
            self.base_checkpoint_selection.select_sample_ids
        )
        if selected_ids != d_b_ids:
            raise ValueError(
                "base checkpoint selection must partition protocol D_B samples"
            )
        # The protocol payload is recursively immutable (frozen configs plus
        # tuples), so its digest and sample lookup can be computed once.  These
        # caches keep the per-state integrity path independent of manifest size.
        computed_fingerprint = self._compute_fingerprint()
        object.__setattr__(self, "_cached_fingerprint", computed_fingerprint)
        object.__setattr__(
            self,
            "_sample_index",
            MappingProxyType(
                {
                    sample_id: (split, group_id)
                    for sample_id, split, group_id in rows
                }
            ),
        )
        if issuing:
            object.__setattr__(
                self,
                "_seal",
                (_CURE_PROTOCOL_SEAL, computed_fingerprint),
            )
        elif self._seal[1] != computed_fingerprint:
            raise ValueError("CURE protocol content differs from its receipt")

    @classmethod
    def from_manifest(
        cls,
        manifest: SplitManifest,
        *,
        base_fingerprint: str,
        adapter_fingerprint: str,
        base_state_fingerprint: str,
        preprocessing_fingerprint: str,
        residual_config: CUREResidualConfig,
        base_checkpoint_selection: BaseCheckpointSelection,
        match_config: MatchConfig = MatchConfig(),
        intervention_config: InterventionConfig = InterventionConfig(),
        descriptor_config: DescriptorConfig = DescriptorConfig(),
        propensity_config: PropensityConfig = PropensityConfig(),
        loss_config: CURELossConfig = CURELossConfig(),
        probability_semantics: str = "foreground-probability-v1",
        feature_semantics: str = "adapter-feature-v1",
    ) -> "CUREProtocol":
        if not isinstance(manifest, SplitManifest):
            raise TypeError("manifest must be SplitManifest")
        manifest.validate()
        if not isinstance(base_checkpoint_selection, BaseCheckpointSelection):
            raise TypeError(
                "base_checkpoint_selection must be BaseCheckpointSelection"
            )
        base_checkpoint_selection.validate_against(manifest)
        return cls(
            base_fingerprint=base_fingerprint,
            adapter_fingerprint=adapter_fingerprint,
            base_state_fingerprint=base_state_fingerprint,
            preprocessing_fingerprint=preprocessing_fingerprint,
            manifest_fingerprint=manifest.fingerprint,
            manifest_membership=tuple(
                sorted(
                    (record.sample_id, record.split, record.group_id)
                    for record in manifest.records
                )
            ),
            base_checkpoint_selection=base_checkpoint_selection,
            residual_config=residual_config,
            loss_config=loss_config,
            match_config=match_config,
            intervention_config=intervention_config,
            descriptor_config=descriptor_config,
            propensity_config=propensity_config,
            probability_semantics=probability_semantics,
            feature_semantics=feature_semantics,
            _seal=_CURE_PROTOCOL_SEAL,
        )

    def validate_manifest(self, manifest: SplitManifest) -> None:
        self.validate_receipt()
        if not isinstance(manifest, SplitManifest):
            raise TypeError("manifest must be SplitManifest")
        manifest.validate()
        membership = tuple(
            sorted(
                (record.sample_id, record.split, record.group_id)
                for record in manifest.records
            )
        )
        if (
            manifest.fingerprint != self.manifest_fingerprint
            or membership != self.manifest_membership
        ):
            raise ValueError("manifest differs from the frozen CURE protocol")
        self.base_checkpoint_selection.validate_against(manifest)

    def assert_sample(
        self,
        sample_id: str,
        *,
        split: str | None = None,
        group_id: str | None = None,
    ) -> None:
        # ``__post_init__`` validated and sealed this immutable lookup.  Avoid a
        # linear manifest scan and JSON re-serialization for every selected
        # training state.
        self.validate_receipt()
        try:
            actual_split, actual_group = self._sample_index[sample_id]
        except KeyError as error:
            raise ValueError(
                f"sample {sample_id!r} is absent from the frozen manifest"
            ) from error

        if split is not None and actual_split != split:
            raise ValueError(
                f"sample {sample_id!r} has split {actual_split}, expected {split}"
            )
        if group_id is not None and actual_group != group_id:
            raise ValueError(
                f"sample {sample_id!r} has group {actual_group!r}, "
                f"expected {group_id!r}"
            )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "base_fingerprint": self.base_fingerprint,
            "adapter_fingerprint": self.adapter_fingerprint,
            "base_state_fingerprint": self.base_state_fingerprint,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "manifest_membership": self.manifest_membership,
            "base_checkpoint_selection": self.base_checkpoint_selection.canonical_payload(),
            "base_checkpoint_selection_fingerprint": self.base_checkpoint_selection.fingerprint,
            "probability_semantics": self.probability_semantics,
            "feature_semantics": self.feature_semantics,
            "occupancy_rule": {
                "comparison": "probability>=occupancy_threshold",
                "connectivity": 8,
                "min_component_area": 1,
            },
            "descriptor_schema": (
                "target-score-min,mean,max;local-scr;log-area;"
                "background-score-max;normalized-boundary-distance-v1"
            ),
            "residual_config": asdict(self.residual_config),
            "loss_config": asdict(self.loss_config),
            "match_config": asdict(self.match_config),
            "intervention_config": asdict(self.intervention_config),
            "descriptor_config": asdict(self.descriptor_config),
            "propensity_config": asdict(self.propensity_config),
        }

    def _compute_fingerprint(self) -> str:
        payload = json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def fingerprint(self) -> str:
        # During ``__post_init__`` the cache is not populated yet.
        return self._cached_fingerprint or self._compute_fingerprint()

    def validate_receipt(self) -> None:
        if not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _CURE_PROTOCOL_SEAL
            and self._seal[1] == self._cached_fingerprint
            and bool(self._cached_fingerprint)
        ):
            raise ValueError("CURE protocol content differs from its receipt")


def _update_tensor_hash(hasher: "hashlib._Hash", name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    tensor = value.detach().to(device="cpu").contiguous()
    hasher.update(name.encode("utf-8"))
    hasher.update(str(tensor.dtype).encode("ascii"))
    hasher.update(repr(tuple(tensor.shape)).encode("ascii"))
    hasher.update(tensor.numpy().tobytes(order="C"))


def tensor_content_fingerprint(name: str, value: Tensor) -> str:
    """Return a dtype/shape/content-sensitive SHA-256 tensor digest."""

    _nonempty("tensor name", name)
    hasher = hashlib.sha256()
    _update_tensor_hash(hasher, name, value)
    return hasher.hexdigest()


def frozen_output_fingerprint(
    protocol: CUREProtocol,
    sample_id: str,
    feature: Tensor,
    probability: Tensor,
) -> str:
    """Hash the exact frozen feature/probability pair used by a CURE state."""

    if not isinstance(protocol, CUREProtocol):
        raise TypeError("protocol must be CUREProtocol")
    protocol.assert_sample(sample_id, split="D_R")
    hasher = hashlib.sha256()
    hasher.update(protocol.fingerprint.encode("ascii"))
    hasher.update(sample_id.encode("utf-8"))
    _update_tensor_hash(hasher, "feature", feature)
    _update_tensor_hash(hasher, "probability", probability)
    return hasher.hexdigest()


def catalog_source_fingerprint(
    protocol: CUREProtocol,
    sample_id: str,
    feature: Tensor,
    probability: Tensor,
    intensity: Tensor,
    gt_labels: Tensor,
) -> str:
    """Hash every sample-level source that determines an eligible catalog."""

    hasher = hashlib.sha256()
    hasher.update(
        frozen_output_fingerprint(protocol, sample_id, feature, probability).encode(
            "ascii"
        )
    )
    _update_tensor_hash(hasher, "intensity", intensity)
    _update_tensor_hash(hasher, "gt_labels", gt_labels)
    return hasher.hexdigest()


def module_state_fingerprint(module: nn.Module) -> str:
    """Hash the exact parameter/buffer state of a PyTorch module.

    This complements a semantic adapter fingerprint, which identifies code,
    preprocessing and extraction choices.  The content digest detects a weight
    or running-buffer mutation even when that semantic identity string remains
    unchanged.
    """

    if not isinstance(module, nn.Module):
        raise TypeError("module must be torch.nn.Module")
    hasher = hashlib.sha256()
    hasher.update(
        f"{type(module).__module__}.{type(module).__qualname__}".encode("utf-8")
    )
    state = module.state_dict()
    for name in sorted(state):
        _update_tensor_hash(hasher, name, state[name])
    return hasher.hexdigest()


def decoder_state_fingerprint(decoder: nn.Module, protocol: CUREProtocol) -> str:
    """Bind an exact canonical residual-decoder checkpoint to its protocol."""

    if not isinstance(protocol, CUREProtocol):
        raise TypeError("protocol must be CUREProtocol")
    protocol.validate_receipt()
    # Local import avoids a module cycle: the decoder itself depends on config.
    from .decoder import CUREResidualDecoder

    if type(decoder) is not CUREResidualDecoder:
        raise TypeError("decoder must be the exact CUREResidualDecoder carrier")
    if decoder.config != protocol.residual_config:
        raise ValueError("decoder configuration differs from the CURE protocol")
    hasher = hashlib.sha256()
    hasher.update(protocol.fingerprint.encode("ascii"))
    hasher.update(
        f"{type(decoder).__module__}.{type(decoder).__qualname__}".encode("utf-8")
    )
    state = decoder.state_dict()
    for name in sorted(state):
        _update_tensor_hash(hasher, name, state[name])
    return hasher.hexdigest()


__all__ = [
    "CUREProtocol",
    "catalog_source_fingerprint",
    "decoder_state_fingerprint",
    "frozen_output_fingerprint",
    "module_state_fingerprint",
    "tensor_content_fingerprint",
]
