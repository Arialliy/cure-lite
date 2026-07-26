"""Frozen D_R-only inputs for the anchored paired-transition objective.

This module binds an existing clean :class:`PairCatalog` to its exact
``PreparedTrainingCatalog`` sources.  It owns no schedule, optimizer, decoder
forward, calibration, or evaluation-split access.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from ..paired_transition_types import (
    AnchoredPairBatch,
    stack_anchored_pair_examples,
)
from ..paired_types import (
    PairCatalog,
    PairExample,
    tensor_content_fingerprint,
)
from .paired_formal_schedule import prepared_training_catalog_fingerprint
from .training_pipeline import PreparedTrainingCatalog


PAIRED_TRANSITION_INPUT_SCHEMA = "cure-lite-paired-transition-inputs-v1"
PAIRED_TRANSITION_PAIR_BINDING_SCHEMA = (
    "cure-lite-paired-transition-pair-binding-v1"
)
PAIRED_TRANSITION_GT_BINDING_SCHEMA = (
    "cure-lite-paired-transition-gt-binding-v1"
)

_HEX = frozenset("0123456789abcdef")


def _require_fingerprint(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 fingerprint")
    return value


def _pair_binding(
    pair: PairExample,
    *,
    gt_union: Tensor,
) -> dict[str, object]:
    """Return a tensor-free binding for every APTO runtime input."""

    return {
        "schema_version": PAIRED_TRANSITION_PAIR_BINDING_SCHEMA,
        "pair_id": pair.pair_id,
        "pair_kind": pair.pair_kind,
        "sample_id": pair.sample_id,
        "group_id": pair.group_id,
        "feature_fingerprint": tensor_content_fingerprint(pair.feature),
        "declared_feature_fingerprint": pair.feature_fingerprint,
        "occupancy_plus_fingerprint": tensor_content_fingerprint(
            pair.occupancy_plus
        ),
        "occupancy_minus_fingerprint": tensor_content_fingerprint(
            pair.occupancy_minus
        ),
        "image_valid_mask_fingerprint": tensor_content_fingerprint(
            pair.image_valid_mask
        ),
        "completion_plus_fingerprint": tensor_content_fingerprint(
            pair.completion_plus
        ),
        "completion_minus_fingerprint": tensor_content_fingerprint(
            pair.completion_minus
        ),
        "label_increment_fingerprint": tensor_content_fingerprint(
            pair.label_increment
        ),
        "gt_union_fingerprint": tensor_content_fingerprint(gt_union),
        "pair_manifest_row_fingerprint": stable_fingerprint(
            pair.canonical_payload()
        ),
    }


def _gt_bindings(
    gt_union_by_sample: Mapping[str, Tensor],
) -> tuple[list[dict[str, object]], str]:
    rows = [
        {
            "sample_id": sample_id,
            "shape": list(gt_union_by_sample[sample_id].shape),
            "dtype": str(gt_union_by_sample[sample_id].dtype),
            "device": gt_union_by_sample[sample_id].device.type,
            "tensor_fingerprint": tensor_content_fingerprint(
                gt_union_by_sample[sample_id]
            ),
        }
        for sample_id in sorted(gt_union_by_sample)
    ]
    fingerprint = stable_fingerprint(
        {
            "schema_version": PAIRED_TRANSITION_GT_BINDING_SCHEMA,
            "sources": rows,
        }
    )
    return rows, fingerprint


def _materializer_payload(
    *,
    dataset: str,
    pair_catalog_fingerprint: str,
    prepared_catalog_fingerprint: str,
    prepared_source_ids: tuple[str, ...],
    pair_by_id: Mapping[str, PairExample],
    gt_union_by_sample: Mapping[str, Tensor],
    feature_shape: tuple[int, int, int, int],
    evaluation_shape: tuple[int, int, int],
) -> dict[str, object]:
    gt_rows, gt_fingerprint = _gt_bindings(gt_union_by_sample)
    pair_rows = [
        _pair_binding(
            pair_by_id[pair_id],
            gt_union=gt_union_by_sample[pair_by_id[pair_id].sample_id],
        )
        for pair_id in sorted(pair_by_id)
    ]
    clean_pair_sources = sorted(
        {pair.sample_id for pair in pair_by_id.values()}
    )
    return {
        "schema_version": PAIRED_TRANSITION_INPUT_SCHEMA,
        "dataset": dataset,
        "split": "D_R",
        "pair_catalog_fingerprint": pair_catalog_fingerprint,
        "prepared_catalog_fingerprint": prepared_catalog_fingerprint,
        "counts": {
            "clean_pairs": len(pair_rows),
            "clean_pair_sources": len(clean_pair_sources),
            "prepared_sources": len(prepared_source_ids),
        },
        "prepared_source_ids": list(prepared_source_ids),
        "clean_pair_source_ids": clean_pair_sources,
        "signatures": {
            "feature_shape": list(feature_shape),
            "feature_dtype": "torch.float32",
            "evaluation_shape": list(evaluation_shape),
            "state_dtype": "torch.bool",
            "label_increment_dtype": "torch.float32",
            "stored_device": "cpu",
        },
        "gt_unions": gt_rows,
        "gt_union_population_fingerprint": gt_fingerprint,
        "all_clean_pair_inputs": pair_rows,
        "all_clean_pair_input_fingerprint": stable_fingerprint(
            {
                "schema_version": PAIRED_TRANSITION_PAIR_BINDING_SCHEMA,
                "pairs": pair_rows,
            }
        ),
        "materialization_contract": {
            "selection_key": "pair_id",
            "selection_order_preserved": True,
            "all_clean_pairs_bound": True,
            "stack_function": "stack_anchored_pair_examples",
            "raw_tensor_payloads_written": False,
            "population_integrity_verification": (
                "explicit_once_before_and_once_after_training_not_per_update"
            ),
        },
        "execution_policy": {
            "runtime_split_access": ["D_R"],
            "D_V_accessed": False,
            "D_T_accessed": False,
            "training_performed": False,
            "calibration_performed": False,
            "inference_performed": False,
        },
    }


@dataclass(frozen=True, eq=False)
class PairedTransitionInputMaterializer:
    """Sealed full-population APTO input materializer."""

    dataset: str
    pair_catalog_fingerprint: str
    prepared_catalog_fingerprint: str
    prepared_source_ids: tuple[str, ...]
    pair_by_id: Mapping[str, PairExample]
    gt_union_by_sample: Mapping[str, Tensor]
    feature_shape: tuple[int, int, int, int]
    evaluation_shape: tuple[int, int, int]
    materializer_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("dataset must be a non-empty string")
        for name in (
            "pair_catalog_fingerprint",
            "prepared_catalog_fingerprint",
            "materializer_fingerprint",
        ):
            _require_fingerprint(getattr(self, name), name=name)
        if (
            not isinstance(self.prepared_source_ids, tuple)
            or self.prepared_source_ids
            != tuple(sorted(set(self.prepared_source_ids)))
            or any(
                not isinstance(sample_id, str) or not sample_id
                for sample_id in self.prepared_source_ids
            )
        ):
            raise ValueError(
                "prepared_source_ids must be sorted, unique, and non-empty"
            )
        if not self.prepared_source_ids:
            raise ValueError("prepared_source_ids cannot be empty")
        if not isinstance(self.pair_by_id, Mapping):
            raise TypeError("pair_by_id must be a mapping")
        if not isinstance(self.gt_union_by_sample, Mapping):
            raise TypeError("gt_union_by_sample must be a mapping")

        normalized_pairs = {
            pair_id: self.pair_by_id[pair_id]
            for pair_id in sorted(self.pair_by_id)
        }
        if not normalized_pairs:
            raise ValueError("pair_by_id requires the clean-positive population")
        if any(
            not isinstance(pair_id, str)
            or not isinstance(pair, PairExample)
            or pair.pair_id != pair_id
            or pair.pair_kind != "clean_positive"
            for pair_id, pair in normalized_pairs.items()
        ):
            raise ValueError(
                "pair_by_id must bind every key to its clean-positive PairExample"
            )

        if set(self.gt_union_by_sample) != set(self.prepared_source_ids):
            raise ValueError(
                "gt_union_by_sample must cover exactly the prepared sources"
            )
        normalized_gt: dict[str, Tensor] = {}
        for sample_id in self.prepared_source_ids:
            value = self.gt_union_by_sample[sample_id]
            if (
                not isinstance(value, Tensor)
                or value.device.type != "cpu"
                or value.dtype != torch.bool
                or tuple(value.shape) != self.evaluation_shape
            ):
                raise TypeError(
                    "every GT union must be a CPU bool tensor with the "
                    "frozen evaluation shape"
                )
            if not torch.any(value):
                raise ValueError("every prepared source requires a non-empty GT union")
            normalized_gt[sample_id] = value.detach().clone().contiguous()

        if any(
            pair.sample_id not in normalized_gt
            for pair in normalized_pairs.values()
        ):
            raise ValueError("GT population omits a clean-pair source")
        if (
            not isinstance(self.feature_shape, tuple)
            or len(self.feature_shape) != 4
            or self.feature_shape[0] != 1
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in self.feature_shape
            )
        ):
            raise ValueError("feature_shape must be [1,C,h,w]")
        if (
            not isinstance(self.evaluation_shape, tuple)
            or len(self.evaluation_shape) != 3
            or self.evaluation_shape[0] != 1
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in self.evaluation_shape
            )
        ):
            raise ValueError("evaluation_shape must be [1,H,W]")
        if any(
            tuple(pair.feature.shape) != self.feature_shape
            or tuple(pair.image_valid_mask.shape) != self.evaluation_shape
            for pair in normalized_pairs.values()
        ):
            raise ValueError("clean-pair tensors differ from frozen signatures")

        object.__setattr__(
            self,
            "pair_by_id",
            MappingProxyType(normalized_pairs),
        )
        object.__setattr__(
            self,
            "gt_union_by_sample",
            MappingProxyType(normalized_gt),
        )
        if self._canonical_payload_fingerprint() != self.materializer_fingerprint:
            raise ValueError("materializer_fingerprint does not reproduce")

    @property
    def canonical_pair_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.pair_by_id))

    def _canonical_payload(self) -> dict[str, object]:
        return _materializer_payload(
            dataset=self.dataset,
            pair_catalog_fingerprint=self.pair_catalog_fingerprint,
            prepared_catalog_fingerprint=self.prepared_catalog_fingerprint,
            prepared_source_ids=self.prepared_source_ids,
            pair_by_id=self.pair_by_id,
            gt_union_by_sample=self.gt_union_by_sample,
            feature_shape=self.feature_shape,
            evaluation_shape=self.evaluation_shape,
        )

    def _canonical_payload_fingerprint(self) -> str:
        return stable_fingerprint(self._canonical_payload())

    def canonical_receipt(self) -> dict[str, object]:
        """Return a detached receipt containing no raw tensor payload."""

        payload = self._canonical_payload()
        if stable_fingerprint(payload) != self.materializer_fingerprint:
            raise RuntimeError("paired-transition materializer inputs changed")
        receipt = deepcopy(payload)
        receipt["materializer_fingerprint"] = self.materializer_fingerprint
        return receipt

    def verify_unchanged(self) -> None:
        """Re-hash every frozen pair and GT tensor binding."""

        if self._canonical_payload_fingerprint() != self.materializer_fingerprint:
            raise RuntimeError("paired-transition materializer inputs changed")

    def materialize(
        self,
        pair_ids: tuple[str, ...],
        *,
        device: torch.device | str,
    ) -> AnchoredPairBatch:
        """Materialize one ordered clean-pair batch exclusively by pair IDs."""

        if (
            not isinstance(pair_ids, tuple)
            or not pair_ids
            or any(not isinstance(pair_id, str) or not pair_id for pair_id in pair_ids)
        ):
            raise ValueError("pair_ids must be a non-empty tuple of strings")
        if len(set(pair_ids)) != len(pair_ids):
            raise ValueError("pair_ids must be unique within one batch")
        missing = tuple(
            pair_id for pair_id in pair_ids if pair_id not in self.pair_by_id
        )
        if missing:
            raise KeyError(f"unknown clean pair IDs: {missing}")
        try:
            resolved_device = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise ValueError("device must be a valid torch device") from error

        examples = tuple(self.pair_by_id[pair_id] for pair_id in pair_ids)
        batch = stack_anchored_pair_examples(
            examples,
            gt_union_by_sample=self.gt_union_by_sample,
            device=resolved_device,
        )
        if (
            batch.pair_ids != pair_ids
            or batch.sample_ids
            != tuple(example.sample_id for example in examples)
        ):
            raise RuntimeError("anchored materialization changed pair identity order")
        return batch


def build_paired_transition_input_materializer(
    pair_catalog: PairCatalog,
    prepared_catalog: PreparedTrainingCatalog,
) -> PairedTransitionInputMaterializer:
    """Build and seal all APTO runtime inputs from an existing D_R catalog."""

    if not isinstance(pair_catalog, PairCatalog):
        raise TypeError("pair_catalog must be PairCatalog")
    if not isinstance(prepared_catalog, PreparedTrainingCatalog):
        raise TypeError("prepared_catalog must be PreparedTrainingCatalog")
    if pair_catalog.split != "D_R":
        raise ValueError("paired-transition inputs permit only D_R")
    if not pair_catalog.clean_positive:
        raise ValueError("paired-transition inputs require clean-positive pairs")
    if stable_fingerprint(pair_catalog.canonical_payload()) != (
        pair_catalog.catalog_fingerprint
    ):
        raise RuntimeError("pair catalog fingerprint does not reproduce")

    entries = {entry.sample_id: entry for entry in prepared_catalog.entries}
    if (
        set(entries) != set(prepared_catalog.source_ids)
        or tuple(sorted(entries)) != prepared_catalog.source_ids
    ):
        raise RuntimeError("prepared source identities are inconsistent")
    if any(
        pair.sample_id not in entries
        for pair in pair_catalog.clean_positive
    ):
        raise ValueError("prepared catalog is missing a clean-pair source")

    gt_union_by_sample: dict[str, Tensor] = {}
    for sample_id in prepared_catalog.source_ids:
        entry = entries[sample_id]
        value = entry.gt.occupancy.unsqueeze(0).to(
            device="cpu",
            dtype=torch.bool,
        ).contiguous()
        source_valid = entry.source.state.image_valid_mask.unsqueeze(0)
        source_gt = (entry.source.state.gt_labels > 0).unsqueeze(0)
        if (
            value.ndim != 3
            or value.shape[0] != 1
            or not torch.any(value)
            or tuple(value.shape) != tuple(source_valid.shape)
            or not torch.equal(value, source_gt)
            or torch.any(value & ~source_valid)
        ):
            raise ValueError("prepared GT union is invalid or inconsistently bound")
        gt_union_by_sample[sample_id] = value.detach().clone()

    feature_signatures = {
        (
            tuple(pair.feature.shape),
            pair.feature.dtype,
            pair.feature.device.type,
        )
        for pair in pair_catalog.clean_positive
    }
    evaluation_signatures = {
        (
            tuple(pair.image_valid_mask.shape),
            pair.image_valid_mask.dtype,
            pair.image_valid_mask.device.type,
        )
        for pair in pair_catalog.clean_positive
    }
    if len(feature_signatures) != 1 or len(evaluation_signatures) != 1:
        raise RuntimeError("clean-pair tensor signatures are not population-uniform")
    feature_shape, feature_dtype, feature_device = next(
        iter(feature_signatures)
    )
    evaluation_shape, evaluation_dtype, evaluation_device = next(
        iter(evaluation_signatures)
    )
    if (
        feature_shape[0] != 1
        or feature_dtype != torch.float32
        or feature_device != "cpu"
        or evaluation_shape[0] != 1
        or evaluation_dtype != torch.bool
        or evaluation_device != "cpu"
    ):
        raise TypeError("paired-transition source tensors must retain CPU cache dtypes")

    pair_by_id: dict[str, PairExample] = {}
    for pair in pair_catalog.clean_positive:
        if pair.pair_id in pair_by_id:
            raise ValueError("clean pair IDs are not unique")
        entry = entries[pair.sample_id]
        gt_union = gt_union_by_sample[pair.sample_id]
        source_valid = entry.source.state.image_valid_mask.unsqueeze(0)
        increment = pair.label_increment.to(dtype=torch.bool)
        if (
            tensor_content_fingerprint(pair.feature)
            != pair.feature_fingerprint
            or tensor_content_fingerprint(entry.source.feature)
            != pair.feature_fingerprint
            or not torch.equal(pair.feature, entry.source.feature)
            or not torch.equal(pair.image_valid_mask, source_valid)
            or not torch.equal(
                pair.completion_minus,
                pair.completion_plus | increment,
            )
            or not torch.equal(
                pair.completion_minus & ~pair.completion_plus,
                increment,
            )
            or torch.any(pair.completion_plus & ~gt_union)
            or torch.any(pair.completion_minus & ~gt_union)
            or torch.any(
                pair.completion_plus
                & (~pair.image_valid_mask | pair.occupancy_plus)
            )
            or torch.any(
                pair.completion_minus
                & (~pair.image_valid_mask | pair.occupancy_minus)
            )
        ):
            raise RuntimeError(
                "clean pair and prepared feature/valid/completion/GT "
                "bindings differ"
            )
        pair_by_id[pair.pair_id] = pair

    frozen_feature_shape = tuple(int(value) for value in feature_shape)
    frozen_evaluation_shape = tuple(int(value) for value in evaluation_shape)
    prepared_fingerprint = prepared_training_catalog_fingerprint(
        prepared_catalog
    )
    frozen_source_ids = tuple(prepared_catalog.source_ids)
    payload = _materializer_payload(
        dataset=pair_catalog.dataset,
        pair_catalog_fingerprint=pair_catalog.catalog_fingerprint,
        prepared_catalog_fingerprint=prepared_fingerprint,
        prepared_source_ids=frozen_source_ids,
        pair_by_id=pair_by_id,
        gt_union_by_sample=gt_union_by_sample,
        feature_shape=frozen_feature_shape,
        evaluation_shape=frozen_evaluation_shape,
    )
    materializer = PairedTransitionInputMaterializer(
        dataset=pair_catalog.dataset,
        pair_catalog_fingerprint=pair_catalog.catalog_fingerprint,
        prepared_catalog_fingerprint=prepared_fingerprint,
        prepared_source_ids=frozen_source_ids,
        pair_by_id=pair_by_id,
        gt_union_by_sample=gt_union_by_sample,
        feature_shape=frozen_feature_shape,
        evaluation_shape=frozen_evaluation_shape,
        materializer_fingerprint=stable_fingerprint(payload),
    )
    materializer.verify_unchanged()
    return materializer


__all__ = [
    "PAIRED_TRANSITION_GT_BINDING_SCHEMA",
    "PAIRED_TRANSITION_INPUT_SCHEMA",
    "PAIRED_TRANSITION_PAIR_BINDING_SCHEMA",
    "PairedTransitionInputMaterializer",
    "build_paired_transition_input_materializer",
]
