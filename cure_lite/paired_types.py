"""Immutable paired-state value objects for the CURE-Lite paired route.

The objects in this module are additive.  They do not extend or alter the
legacy ``BranchSupervision``/``BranchBatch`` path: a pair stores the frozen
feature once and carries two occupancy endpoints for one coupled update.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import torch
from torch import Tensor


PAIR_KINDS = ("clean_positive", "component_null", "identity_null")
_HEX_DIGITS = frozenset("0123456789abcdef")


def _require_nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_fingerprint(value: object, *, name: str) -> str:
    text = _require_nonempty_string(value, name=name)
    if len(text) != 64 or any(character not in _HEX_DIGITS for character in text):
        raise ValueError(f"{name} must be a lowercase SHA256 fingerprint")
    return text


def tensor_content_fingerprint(tensor: Tensor) -> str:
    """Return a device-independent fingerprint over dtype, shape, and bytes."""

    if not isinstance(tensor, Tensor):
        raise TypeError("tensor must be a torch.Tensor")
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
    )
    if value.numel():
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _as_single_bool_mask(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.device.type != "cpu" or value.dtype != torch.bool:
        raise TypeError(f"{name} must be a CPU bool tensor")
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(f"{name} must have shape [1,H,W]")
    if min(value.shape[-2:]) < 1:
        raise ValueError(f"{name} spatial dimensions must be non-empty")
    return value.detach().clone().contiguous()


def _as_single_binary_target(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.device.type != "cpu" or value.dtype != torch.float32:
        raise TypeError(f"{name} must be a CPU float32 tensor")
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(f"{name} must have shape [1,H,W]")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    if torch.any((value != 0.0) & (value != 1.0)):
        raise ValueError(f"{name} must be binary")
    return value.detach().clone().contiguous()


def _optional_positive_id(value: object, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be null or a positive integer")
    return value


@dataclass(frozen=True, eq=False)
class PairExample:
    """One same-source before/after pair plus its exact completion truth."""

    pair_id: str
    pair_kind: str
    sample_id: str
    group_id: str
    feature: Tensor
    occupancy_plus: Tensor
    occupancy_minus: Tensor
    removed_component: Tensor
    image_valid_mask: Tensor
    completion_plus: Tensor
    completion_minus: Tensor
    label_increment: Tensor
    clean_increment: Tensor
    evaluation_gt_id: int | None
    native_gt_id: int | None
    pred_id: int | None
    feature_fingerprint: str
    before_match_fingerprint: str
    after_match_fingerprint: str
    projected_occupancy_plus_fingerprint: str
    projected_occupancy_minus_fingerprint: str
    projection_visible: bool
    geometry_safe_bijective_lineage: bool | None
    selected_gt_is_only_new_unmatched: bool | None
    other_match_identities_unchanged: bool | None
    preexisting_unmatched_gt_noninterference: bool | None

    def __post_init__(self) -> None:
        _require_fingerprint(self.pair_id, name="pair_id")
        if self.pair_kind not in PAIR_KINDS:
            raise ValueError(f"unknown pair_kind {self.pair_kind!r}")
        _require_nonempty_string(self.sample_id, name="sample_id")
        _require_nonempty_string(self.group_id, name="group_id")
        if (
            not isinstance(self.feature, Tensor)
            or self.feature.device.type != "cpu"
            or self.feature.dtype != torch.float32
            or self.feature.ndim != 4
            or self.feature.shape[0] != 1
        ):
            raise TypeError("feature must be a CPU float32 tensor [1,C,h,w]")
        if self.feature.shape[1] < 1 or min(self.feature.shape[-2:]) < 1:
            raise ValueError("feature dimensions must be non-empty")
        if self.feature.requires_grad or not torch.isfinite(self.feature).all():
            raise ValueError("feature must be finite and detached")
        for name in (
            "feature_fingerprint",
            "before_match_fingerprint",
            "after_match_fingerprint",
            "projected_occupancy_plus_fingerprint",
            "projected_occupancy_minus_fingerprint",
        ):
            _require_fingerprint(getattr(self, name), name=name)
        if tensor_content_fingerprint(self.feature) != self.feature_fingerprint:
            raise ValueError("feature_fingerprint does not match feature")

        bool_fields = (
            "occupancy_plus",
            "occupancy_minus",
            "removed_component",
            "image_valid_mask",
            "completion_plus",
            "completion_minus",
            "clean_increment",
        )
        normalized_bool = {
            name: _as_single_bool_mask(getattr(self, name), name=name)
            for name in bool_fields
        }
        target = _as_single_binary_target(
            self.label_increment,
            name="label_increment",
        )
        shapes = {
            tuple(value.shape)
            for value in (*normalized_bool.values(), target)
        }
        if len(shapes) != 1:
            raise ValueError("all paired evaluation-grid tensors must share a shape")
        for name, value in normalized_bool.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "label_increment", target)
        object.__setattr__(self, "feature", self.feature.contiguous())

        plus = normalized_bool["occupancy_plus"]
        minus = normalized_bool["occupancy_minus"]
        removed = normalized_bool["removed_component"]
        valid = normalized_bool["image_valid_mask"]
        r_plus = normalized_bool["completion_plus"]
        r_minus = normalized_bool["completion_minus"]
        clean = normalized_bool["clean_increment"]
        increment = target.to(torch.bool)
        if not torch.any(valid):
            raise ValueError("image_valid_mask cannot be empty")
        if torch.any(plus & ~valid) or torch.any(minus & ~valid):
            raise ValueError("occupancy endpoints must remain inside image_valid_mask")
        if torch.any(minus & ~plus):
            raise ValueError("occupancy_minus must be a subset of occupancy_plus")
        if not torch.equal(removed, plus & ~minus):
            raise ValueError(
                "removed_component must equal occupancy_plus minus occupancy_minus"
            )
        if torch.any(r_plus & (~valid | plus)):
            raise ValueError("completion_plus must be valid and writable under O_plus")
        if torch.any(r_minus & (~valid | minus)):
            raise ValueError("completion_minus must be valid and writable under O_minus")
        actual_increment = r_minus & ~r_plus
        if not torch.equal(increment, actual_increment):
            raise ValueError(
                "label_increment must equal completion_minus setminus completion_plus"
            )
        if torch.any(increment & ~valid) or torch.any(clean & ~valid):
            raise ValueError("increment truth must remain inside image_valid_mask")

        evaluation_gt_id = _optional_positive_id(
            self.evaluation_gt_id,
            name="evaluation_gt_id",
        )
        native_gt_id = _optional_positive_id(
            self.native_gt_id,
            name="native_gt_id",
        )
        pred_id = _optional_positive_id(self.pred_id, name="pred_id")
        if not isinstance(self.projection_visible, bool):
            raise TypeError("projection_visible must be bool")
        clean_checks = (
            self.geometry_safe_bijective_lineage,
            self.selected_gt_is_only_new_unmatched,
            self.other_match_identities_unchanged,
            self.preexisting_unmatched_gt_noninterference,
        )
        if any(
            value is not None and not isinstance(value, bool)
            for value in clean_checks
        ):
            raise TypeError("clean-pair checks must be bool or null")
        projected_equal = (
            self.projected_occupancy_plus_fingerprint
            == self.projected_occupancy_minus_fingerprint
        )
        if self.projection_visible == projected_equal:
            raise ValueError(
                "projection_visible disagrees with projected occupancy fingerprints"
            )

        if self.pair_kind == "clean_positive":
            if None in (evaluation_gt_id, native_gt_id, pred_id):
                raise ValueError("clean_positive requires GT and prediction identities")
            if not torch.any(removed) or not self.projection_visible:
                raise ValueError(
                    "clean_positive requires a visible strict component deletion"
                )
            if not torch.any(increment) or not torch.equal(increment, clean):
                raise ValueError(
                    "clean_positive requires non-empty D equal to clean_increment A"
                )
            if not torch.any(valid & ~increment):
                raise ValueError(
                    "clean_positive requires a non-empty zero-response domain"
                )
            if clean_checks != (True, True, True, True):
                raise ValueError(
                    "clean_positive requires all lineage/matching checks to pass"
                )
        elif self.pair_kind == "component_null":
            if pred_id is None:
                raise ValueError("component_null requires pred_id")
            if evaluation_gt_id is not None or native_gt_id is not None:
                raise ValueError("component_null cannot carry a selected GT identity")
            if not torch.any(removed) or not self.projection_visible:
                raise ValueError(
                    "component_null requires a visible strict component deletion"
                )
            if torch.any(increment) or torch.any(clean):
                raise ValueError("component_null requires empty D and A")
            if clean_checks != (None, None, None, None):
                raise ValueError("component_null cannot carry clean-positive checks")
        else:
            if any(
                value is not None
                for value in (evaluation_gt_id, native_gt_id, pred_id)
            ):
                raise ValueError("identity_null cannot carry target/component IDs")
            if not torch.equal(plus, minus) or torch.any(removed):
                raise ValueError("identity_null requires identical occupancy endpoints")
            if torch.any(increment) or torch.any(clean):
                raise ValueError("identity_null requires empty D and A")
            if self.projection_visible:
                raise ValueError("identity_null cannot be projection-visible")
            if not torch.equal(r_plus, r_minus):
                raise ValueError("identity_null requires identical completion endpoints")
            if clean_checks != (None, None, None, None):
                raise ValueError("identity_null cannot carry clean-positive checks")

    @property
    def response_pixels(self) -> int:
        return int(torch.count_nonzero(self.label_increment))

    @property
    def zero_response_pixels(self) -> int:
        return int(
            torch.count_nonzero(
                self.image_valid_mask & ~self.label_increment.to(torch.bool)
            )
        )

    def canonical_payload(self) -> dict[str, object]:
        """Return the tensor-free manifest row used for catalog fingerprinting."""

        increment = self.label_increment.to(torch.bool)
        removed = self.removed_component
        clean = self.clean_increment
        return {
            "pair_id": self.pair_id,
            "pair_kind": self.pair_kind,
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "evaluation_gt_id": self.evaluation_gt_id,
            "native_gt_id": self.native_gt_id,
            "pred_id": self.pred_id,
            "fingerprints": {
                "feature": self.feature_fingerprint,
                "occupancy_plus": tensor_content_fingerprint(
                    self.occupancy_plus
                ),
                "occupancy_minus": tensor_content_fingerprint(
                    self.occupancy_minus
                ),
                "removed_component": tensor_content_fingerprint(removed),
                "image_valid_mask": tensor_content_fingerprint(
                    self.image_valid_mask
                ),
                "completion_plus": tensor_content_fingerprint(
                    self.completion_plus
                ),
                "completion_minus": tensor_content_fingerprint(
                    self.completion_minus
                ),
                "label_increment": tensor_content_fingerprint(
                    self.label_increment
                ),
                "clean_increment": tensor_content_fingerprint(clean),
                "projected_occupancy_plus": (
                    self.projected_occupancy_plus_fingerprint
                ),
                "projected_occupancy_minus": (
                    self.projected_occupancy_minus_fingerprint
                ),
                "before_match": self.before_match_fingerprint,
                "after_match": self.after_match_fingerprint,
            },
            "checks": {
                "projection_visible": self.projection_visible,
                "feature_stored_once_and_shared_by_endpoints": True,
                "single_complete_component_removed": (
                    self.pair_kind != "identity_null"
                ),
                "geometry_safe_bijective_lineage": (
                    self.geometry_safe_bijective_lineage
                ),
                "selected_gt_is_only_new_unmatched": (
                    self.selected_gt_is_only_new_unmatched
                ),
                "other_match_identities_unchanged": (
                    self.other_match_identities_unchanged
                ),
                "preexisting_unmatched_gt_noninterference": (
                    self.preexisting_unmatched_gt_noninterference
                ),
                "actual_increment_equals_clean_increment": torch.equal(
                    increment,
                    clean,
                ),
            },
            "pixel_accounting": {
                "valid": int(torch.count_nonzero(self.image_valid_mask)),
                "removed_component": int(torch.count_nonzero(removed)),
                "label_increment": int(torch.count_nonzero(increment)),
                "zero_response": self.zero_response_pixels,
                "A_intersect_C": int(torch.count_nonzero(clean & removed)),
                "A_setminus_C": int(torch.count_nonzero(clean & ~removed)),
                # On a clean pair A=V∩G∩¬O- and C⊆V∩¬O-, so C\\A=C\\G.
                "C_setminus_G": (
                    int(torch.count_nonzero(removed & ~clean))
                    if self.pair_kind == "clean_positive"
                    else None
                ),
            },
        }


@dataclass(frozen=True)
class PairCatalogExclusion:
    """One explicitly rejected positive/null candidate."""

    pair_kind: str
    sample_id: str
    group_id: str
    evaluation_gt_id: int | None
    native_gt_id: int | None
    pred_id: int | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.pair_kind not in PAIR_KINDS:
            raise ValueError(f"unknown pair_kind {self.pair_kind!r}")
        _require_nonempty_string(self.sample_id, name="sample_id")
        _require_nonempty_string(self.group_id, name="group_id")
        _optional_positive_id(self.evaluation_gt_id, name="evaluation_gt_id")
        _optional_positive_id(self.native_gt_id, name="native_gt_id")
        _optional_positive_id(self.pred_id, name="pred_id")
        if (
            not isinstance(self.reason_codes, tuple)
            or not self.reason_codes
            or self.reason_codes != tuple(sorted(set(self.reason_codes)))
            or any(not isinstance(item, str) or not item for item in self.reason_codes)
        ):
            raise ValueError("reason_codes must be a sorted unique non-empty tuple")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "pair_kind": self.pair_kind,
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "evaluation_gt_id": self.evaluation_gt_id,
            "native_gt_id": self.native_gt_id,
            "pred_id": self.pred_id,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class PairCatalog:
    """Complete D_R paired population with explicit inclusion/exclusion rows."""

    dataset: str
    split: str
    paired_protocol_fingerprint: str
    geometry_catalog_fingerprint: str
    source_catalog_fingerprint: str
    manifest_fingerprint: str
    clean_positive: tuple[PairExample, ...]
    component_null: tuple[PairExample, ...]
    identity_null: tuple[PairExample, ...]
    exclusions: tuple[PairCatalogExclusion, ...]
    catalog_fingerprint: str

    def __post_init__(self) -> None:
        _require_nonempty_string(self.dataset, name="dataset")
        if self.split != "D_R":
            raise ValueError("paired catalog permits only D_R")
        for name in (
            "paired_protocol_fingerprint",
            "geometry_catalog_fingerprint",
            "source_catalog_fingerprint",
            "manifest_fingerprint",
        ):
            _require_fingerprint(getattr(self, name), name=name)
        if self.catalog_fingerprint:
            _require_fingerprint(
                self.catalog_fingerprint,
                name="catalog_fingerprint",
            )
        for name, expected_kind in (
            ("clean_positive", "clean_positive"),
            ("component_null", "component_null"),
            ("identity_null", "identity_null"),
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(value, PairExample)
                or value.pair_kind != expected_kind
                for value in values
            ):
                raise TypeError(f"{name} must contain {expected_kind} PairExample values")
            identities = tuple(
                (
                    value.sample_id,
                    -1
                    if value.evaluation_gt_id is None
                    else value.evaluation_gt_id,
                    -1 if value.pred_id is None else value.pred_id,
                    value.pair_id,
                )
                for value in values
            )
            if identities != tuple(sorted(set(identities))):
                raise ValueError(f"{name} must be canonically ordered and unique")
        all_ids = tuple(
            value.pair_id
            for value in (
                *self.clean_positive,
                *self.component_null,
                *self.identity_null,
            )
        )
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("pair IDs must be globally unique")
        if not isinstance(self.exclusions, tuple) or any(
            not isinstance(value, PairCatalogExclusion)
            for value in self.exclusions
        ):
            raise TypeError("exclusions must contain PairCatalogExclusion values")
        exclusion_keys = tuple(
            (
                row.pair_kind,
                row.sample_id,
                -1 if row.evaluation_gt_id is None else row.evaluation_gt_id,
                -1 if row.pred_id is None else row.pred_id,
            )
            for row in self.exclusions
        )
        if exclusion_keys != tuple(sorted(set(exclusion_keys))):
            raise ValueError("exclusions must be canonically ordered and unique")

    @property
    def trainable_pairs(self) -> tuple[PairExample, ...]:
        return self.clean_positive

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": "cure-lite-pair-catalog-v1",
            "dataset": self.dataset,
            "split": self.split,
            "paired_protocol_fingerprint": self.paired_protocol_fingerprint,
            "geometry_catalog_fingerprint": self.geometry_catalog_fingerprint,
            "source_catalog_fingerprint": self.source_catalog_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
            "counts": {
                "clean_positive": len(self.clean_positive),
                "component_null": len(self.component_null),
                "identity_null": len(self.identity_null),
                "exclusions": len(self.exclusions),
            },
            "clean_positive": [
                value.canonical_payload() for value in self.clean_positive
            ],
            "component_null": [
                value.canonical_payload() for value in self.component_null
            ],
            "identity_null": [
                value.canonical_payload() for value in self.identity_null
            ],
            "exclusions": [
                value.canonical_payload() for value in self.exclusions
            ],
        }


@dataclass(frozen=True)
class PairBatch:
    """Device-ready batch with one feature tensor per pair, not per endpoint."""

    feature: Tensor
    occupancy_plus: Tensor
    occupancy_minus: Tensor
    label_increment: Tensor
    image_valid_mask: Tensor
    pair_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    pair_kinds: tuple[str, ...]
    projection_visible: tuple[bool, ...]

    def validate(self) -> None:
        tensor_fields = (
            self.feature,
            self.occupancy_plus,
            self.occupancy_minus,
            self.label_increment,
            self.image_valid_mask,
        )
        if any(not isinstance(value, Tensor) for value in tensor_fields):
            raise TypeError("all PairBatch tensor fields must be tensors")
        if self.feature.ndim != 4 or self.feature.shape[0] < 1:
            raise ValueError("feature must have shape [B,C,h,w] with B >= 1")
        if any(
            value.ndim != 4 or value.shape[1] != 1
            for value in (
                self.occupancy_plus,
                self.occupancy_minus,
                self.label_increment,
                self.image_valid_mask,
            )
        ):
            raise ValueError("paired evaluation tensors must have shape [B,1,H,W]")
        if not (
            self.occupancy_plus.shape
            == self.occupancy_minus.shape
            == self.label_increment.shape
            == self.image_valid_mask.shape
        ):
            raise ValueError("paired evaluation tensor shapes must match")
        batch_size = self.feature.shape[0]
        if self.occupancy_plus.shape[0] != batch_size:
            raise ValueError("feature and occupancy batch sizes differ")
        if not self.feature.is_floating_point():
            raise TypeError("feature must be floating point")
        if self.occupancy_plus.dtype != torch.bool:
            raise TypeError("occupancy_plus must be bool")
        if self.occupancy_minus.dtype != torch.bool:
            raise TypeError("occupancy_minus must be bool")
        if self.image_valid_mask.dtype != torch.bool:
            raise TypeError("image_valid_mask must be bool")
        if self.label_increment.dtype != torch.float32:
            raise TypeError("label_increment must be float32")
        devices = {value.device for value in tensor_fields}
        if len(devices) != 1:
            raise ValueError("all PairBatch tensors must share a device")
        if not torch.isfinite(self.feature).all() or not torch.isfinite(
            self.label_increment
        ).all():
            raise ValueError("PairBatch floating tensors must be finite")
        if torch.any(
            (self.label_increment != 0.0) & (self.label_increment != 1.0)
        ):
            raise ValueError("label_increment must be binary")
        if torch.any(self.occupancy_minus & ~self.occupancy_plus):
            raise ValueError("occupancy_minus must be a subset of occupancy_plus")
        if torch.any(self.occupancy_plus & ~self.image_valid_mask):
            raise ValueError("occupancy_plus extends outside image_valid_mask")
        if torch.any(self.label_increment.to(torch.bool) & ~self.image_valid_mask):
            raise ValueError("label_increment extends outside image_valid_mask")
        metadata = (
            self.pair_ids,
            self.sample_ids,
            self.group_ids,
            self.pair_kinds,
            self.projection_visible,
        )
        if any(not isinstance(values, tuple) for values in metadata):
            raise TypeError("PairBatch metadata fields must be tuples")
        if any(len(values) != batch_size for values in metadata):
            raise ValueError("PairBatch metadata lengths must equal batch size")
        for pair_id in self.pair_ids:
            _require_fingerprint(pair_id, name="pair_id")
        for name, values in (
            ("sample_ids", self.sample_ids),
            ("group_ids", self.group_ids),
        ):
            for value in values:
                _require_nonempty_string(value, name=name)
        if any(value not in PAIR_KINDS for value in self.pair_kinds):
            raise ValueError("PairBatch contains an unknown pair kind")
        if any(not isinstance(value, bool) for value in self.projection_visible):
            raise TypeError("projection_visible must contain bool values")
        positive = self.label_increment.to(torch.bool).flatten(1).any(dim=1)
        valid = self.image_valid_mask.flatten(1).any(dim=1)
        zero_domain = (
            self.image_valid_mask
            & ~self.label_increment.to(torch.bool)
        ).flatten(1).any(dim=1)
        if not torch.all(valid):
            raise ValueError("every pair requires a non-empty valid domain")
        for index, kind in enumerate(self.pair_kinds):
            endpoints_equal = torch.equal(
                self.occupancy_plus[index],
                self.occupancy_minus[index],
            )
            if kind == "clean_positive":
                if not bool(positive[index]) or not bool(zero_domain[index]):
                    raise ValueError(
                        "every clean_positive requires positive and zero domains"
                    )
                if endpoints_equal:
                    raise ValueError("clean_positive endpoints must differ")
                if not self.projection_visible[index]:
                    raise ValueError(
                        "clean_positive requires a preverified visible projection"
                    )
            elif kind == "component_null":
                if bool(positive[index]) or endpoints_equal:
                    raise ValueError(
                        "component_null requires empty D and different endpoints"
                    )
                if not self.projection_visible[index]:
                    raise ValueError(
                        "component_null requires a preverified visible projection"
                    )
            else:
                if bool(positive[index]) or not endpoints_equal:
                    raise ValueError(
                        "identity_null requires empty D and identical endpoints"
                    )
                if self.projection_visible[index]:
                    raise ValueError(
                        "identity_null cannot be projection-visible"
                    )


def stack_pair_examples(
    examples: Iterable[PairExample],
    *,
    device: torch.device | str,
) -> PairBatch:
    """Stack compatible pair examples without duplicating endpoint features."""

    values = tuple(examples)
    if not values:
        raise ValueError("cannot stack an empty pair selection")
    if any(not isinstance(value, PairExample) for value in values):
        raise TypeError("examples must contain only PairExample values")
    feature_shapes = {tuple(value.feature.shape[1:]) for value in values}
    evaluation_shapes = {tuple(value.occupancy_plus.shape) for value in values}
    if len(feature_shapes) != 1 or len(evaluation_shapes) != 1:
        raise ValueError("selected pairs must have compatible tensor grids")
    batch = PairBatch(
        feature=torch.cat([value.feature for value in values], dim=0).to(device),
        occupancy_plus=torch.stack(
            [value.occupancy_plus for value in values],
            dim=0,
        ).to(device),
        occupancy_minus=torch.stack(
            [value.occupancy_minus for value in values],
            dim=0,
        ).to(device),
        label_increment=torch.stack(
            [value.label_increment for value in values],
            dim=0,
        ).to(device),
        image_valid_mask=torch.stack(
            [value.image_valid_mask for value in values],
            dim=0,
        ).to(device),
        pair_ids=tuple(value.pair_id for value in values),
        sample_ids=tuple(value.sample_id for value in values),
        group_ids=tuple(value.group_id for value in values),
        pair_kinds=tuple(value.pair_kind for value in values),
        projection_visible=tuple(value.projection_visible for value in values),
    )
    batch.validate()
    return batch


__all__ = [
    "PAIR_KINDS",
    "PairBatch",
    "PairCatalog",
    "PairCatalogExclusion",
    "PairExample",
    "stack_pair_examples",
    "tensor_content_fingerprint",
]
