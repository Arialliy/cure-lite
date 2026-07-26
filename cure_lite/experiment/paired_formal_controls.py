"""Frozen full-population inputs for formal paired matched controls.

This module closes the remaining runtime-input boundary between the static
matched-control preflight and :mod:`paired_formal_training`.  It does not own
or select a training schedule, run a decoder, access an evaluation split, or
modify any loss.  Every returned tensor is selected exclusively by the two
``PairExample.pair_id`` values supplied by the already-frozen formal schedule.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from ..paired_control_inputs import (
    DCTCoordinateBasis,
    build_dct_coordinate_basis,
    build_target_permutation,
    materialize_permuted_label_increments,
    target_permutation_compatible,
)
from ..paired_control_losses import (
    build_after_only_synthetic_supervision,
    build_geometry_matched_endpoint_supervision,
)
from ..paired_types import (
    PairBatch,
    PairCatalog,
    PairExample,
    tensor_content_fingerprint,
)
from ..train.paired_control_step import CONTROL_KINDS
from ..train.paired_pools import PAIRED_EPOCHS, PAIRED_STEPS_PER_EPOCH
from .paired_formal_schedule import prepared_training_catalog_fingerprint
from .paired_control_preflight import (
    _strict_json,
    load_control_preflight_artifact,
)
from .training_pipeline import PreparedTrainingCatalog


PAIRED_FORMAL_CONTROL_PREFLIGHT_BINDING_SCHEMA = (
    "cure-lite-paired-formal-control-preflight-binding-v1"
)
PAIRED_FORMAL_CONTROL_PROVIDER_SCHEMA = (
    "cure-lite-paired-formal-control-provider-v1"
)
PAIRED_FORMAL_CONTROL_PAIR_BINDING_SCHEMA = (
    "cure-lite-paired-formal-control-pair-binding-v1"
)
PAIRED_FORMAL_CONTROL_GT_BINDING_SCHEMA = (
    "cure-lite-paired-formal-control-gt-binding-v1"
)
PAIRED_FORMAL_CONTROL_PERMUTATION_BINDING_SCHEMA = (
    "cure-lite-paired-formal-control-permutation-binding-v1"
)

_HEX = frozenset("0123456789abcdef")
_CONTROL_KEYS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "independent_endpoint": frozenset(
            {"gt_union", "completion_plus", "completion_minus"}
        ),
        "after_only": frozenset({"gt_union"}),
        "zero_feature": frozenset(),
        "coordinate_basis": frozenset({"coordinate_basis"}),
        "feature_only": frozenset(),
        "target_permutation": frozenset({"permuted_label_increment"}),
        "plus_detach": frozenset(),
        "minus_detach": frozenset(),
    }
)


def _require_fingerprint(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 fingerprint")
    return value


@dataclass(frozen=True)
class FrozenControlPreflightFingerprints:
    """Exact identities copied from one completed static control preflight."""

    complete_fingerprint: str
    pair_catalog_fingerprint: str
    dct_basis_fingerprint: str
    target_permutation_plan_fingerprint: str
    target_assignment_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "complete_fingerprint",
            "pair_catalog_fingerprint",
            "dct_basis_fingerprint",
            "target_permutation_plan_fingerprint",
            "target_assignment_fingerprint",
        ):
            _require_fingerprint(getattr(self, name), name=name)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                PAIRED_FORMAL_CONTROL_PREFLIGHT_BINDING_SCHEMA
            ),
            "complete_fingerprint": self.complete_fingerprint,
            "pair_catalog_fingerprint": self.pair_catalog_fingerprint,
            "dct_basis_fingerprint": self.dct_basis_fingerprint,
            "target_permutation_plan_fingerprint": (
                self.target_permutation_plan_fingerprint
            ),
            "target_assignment_fingerprint": (
                self.target_assignment_fingerprint
            ),
        }


def load_frozen_control_preflight_fingerprints(
    root: str | Path,
    pair_catalog: PairCatalog,
) -> FrozenControlPreflightFingerprints:
    """Strictly load the five fingerprints needed by a formal provider.

    The authoritative loader first verifies the complete artifact inventory,
    every receipt hash, all cross-file bindings, and READY permutation
    postconditions.  This helper then exposes the already-verified assignment
    fingerprint that the older ``PublishedControlPreflight`` value object does
    not carry.
    """

    if not isinstance(pair_catalog, PairCatalog):
        raise TypeError("pair_catalog must be PairCatalog")
    published = load_control_preflight_artifact(
        root,
        expected_catalog_fingerprint=pair_catalog.catalog_fingerprint,
        expected_protocol_fingerprint=(
            pair_catalog.paired_protocol_fingerprint
        ),
        expected_clean_pair_count=len(pair_catalog.clean_positive),
    )
    if published.status != "READY":
        raise RuntimeError("formal controls require a READY static preflight")
    permutation = _strict_json(
        published.root / "receipts" / "target_permutation.json",
        name="formal control target permutation receipt",
    )
    assignment_fingerprint = _require_fingerprint(
        permutation.get("assignment_fingerprint"),
        name="target_assignment_fingerprint",
    )
    assignments = permutation.get("assignments")
    if (
        not isinstance(assignments, list)
        or stable_fingerprint(assignments) != assignment_fingerprint
        or permutation.get("plan_fingerprint")
        != published.permutation_fingerprint
        or len(assignments) != len(pair_catalog.clean_positive)
    ):
        raise RuntimeError(
            "strict control preflight assignment binding changed"
        )
    return FrozenControlPreflightFingerprints(
        complete_fingerprint=published.complete_fingerprint,
        pair_catalog_fingerprint=published.catalog_fingerprint,
        dct_basis_fingerprint=published.dct_basis_fingerprint,
        target_permutation_plan_fingerprint=(
            published.permutation_fingerprint
        ),
        target_assignment_fingerprint=assignment_fingerprint,
    )


def _pair_tensor_binding(pair: PairExample) -> dict[str, object]:
    return {
        "schema_version": PAIRED_FORMAL_CONTROL_PAIR_BINDING_SCHEMA,
        "pair_id": pair.pair_id,
        "sample_id": pair.sample_id,
        "group_id": pair.group_id,
        "feature_fingerprint": pair.feature_fingerprint,
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
    }


def _gt_binding(
    gt_union_by_sample: Mapping[str, Tensor],
) -> tuple[list[dict[str, object]], str]:
    rows = [
        {
            "sample_id": sample_id,
            "shape": list(gt_union_by_sample[sample_id].shape),
            "dtype": str(gt_union_by_sample[sample_id].dtype),
            "tensor_fingerprint": tensor_content_fingerprint(
                gt_union_by_sample[sample_id]
            ),
        }
        for sample_id in sorted(gt_union_by_sample)
    ]
    return rows, stable_fingerprint(
        {
            "schema_version": PAIRED_FORMAL_CONTROL_GT_BINDING_SCHEMA,
            "sources": rows,
        }
    )


def _assignment_binding(
    *,
    assignment_payloads: tuple[dict[str, str], ...],
    target_by_recipient: Mapping[str, Tensor],
) -> tuple[list[dict[str, str]], str]:
    rows: list[dict[str, str]] = []
    for assignment in assignment_payloads:
        recipient = assignment["recipient_pair_id"]
        rows.append(
            {
                **assignment,
                "runtime_target_fingerprint": tensor_content_fingerprint(
                    target_by_recipient[recipient]
                ),
            }
        )
    return rows, stable_fingerprint(
        {
            "schema_version": (
                PAIRED_FORMAL_CONTROL_PERMUTATION_BINDING_SCHEMA
            ),
            "assignments": rows,
        }
    )


def _provider_payload(
    *,
    dataset: str,
    pair_catalog_fingerprint: str,
    prepared_catalog_fingerprint: str,
    preflight: FrozenControlPreflightFingerprints,
    coordinate_basis: DCTCoordinateBasis,
    pair_by_id: Mapping[str, PairExample],
    gt_union_by_sample: Mapping[str, Tensor],
    completion_plus_by_pair: Mapping[str, Tensor],
    completion_minus_by_pair: Mapping[str, Tensor],
    permuted_target_by_recipient: Mapping[str, Tensor],
    assignment_by_recipient: Mapping[str, Mapping[str, str]],
    feature_shape: tuple[int, int, int, int],
    evaluation_shape: tuple[int, int, int],
) -> dict[str, object]:
    gt_rows, gt_fingerprint = _gt_binding(gt_union_by_sample)
    pair_rows: list[dict[str, object]] = []
    for pair_id in sorted(pair_by_id):
        row = _pair_tensor_binding(pair_by_id[pair_id])
        row["provider_completion_plus_fingerprint"] = (
            tensor_content_fingerprint(completion_plus_by_pair[pair_id])
        )
        row["provider_completion_minus_fingerprint"] = (
            tensor_content_fingerprint(completion_minus_by_pair[pair_id])
        )
        pair_rows.append(row)
    assignment_rows: list[dict[str, str]] = []
    for pair_id in sorted(assignment_by_recipient):
        row = dict(assignment_by_recipient[pair_id])
        row["runtime_target_fingerprint"] = tensor_content_fingerprint(
            permuted_target_by_recipient[pair_id]
        )
        assignment_rows.append(row)
    return {
        "schema_version": PAIRED_FORMAL_CONTROL_PROVIDER_SCHEMA,
        "dataset": dataset,
        "split": "D_R",
        "pair_catalog_fingerprint": pair_catalog_fingerprint,
        "prepared_catalog_fingerprint": prepared_catalog_fingerprint,
        "control_preflight": preflight.canonical_payload(),
        "counts": {
            "clean_pairs": len(pair_by_id),
            "prepared_sources": len(gt_union_by_sample),
            "permutation_assignments": len(assignment_by_recipient),
        },
        "signatures": {
            "feature_shape": list(feature_shape),
            "feature_dtype": "torch.float32",
            "evaluation_shape": list(evaluation_shape),
            "occupancy_dtype": "torch.bool",
            "target_dtype": "torch.float32",
            "stored_device": "cpu",
        },
        "gt_unions": gt_rows,
        "gt_union_population_fingerprint": gt_fingerprint,
        "all_pair_inputs": pair_rows,
        "all_pair_input_fingerprint": stable_fingerprint(
            {
                "schema_version": (
                    PAIRED_FORMAL_CONTROL_PAIR_BINDING_SCHEMA
                ),
                "pairs": pair_rows,
            }
        ),
        "coordinate_basis": {
            "basis_fingerprint": coordinate_basis.basis_fingerprint,
            "tensor_fingerprint": tensor_content_fingerprint(
                coordinate_basis.tensor
            ),
            "shape": list(coordinate_basis.tensor.shape),
            "dtype": str(coordinate_basis.tensor.dtype),
        },
        "target_permutation": {
            "plan_fingerprint": (
                preflight.target_permutation_plan_fingerprint
            ),
            "static_assignment_fingerprint": (
                preflight.target_assignment_fingerprint
            ),
            "runtime_assignments": assignment_rows,
            "runtime_binding_fingerprint": stable_fingerprint(
                {
                    "schema_version": (
                        PAIRED_FORMAL_CONTROL_PERMUTATION_BINDING_SCHEMA
                    ),
                    "assignments": assignment_rows,
                }
            ),
            "source_disjoint": True,
            "fixed_point_free": True,
            "full_recipient_and_donor_marginals": True,
        },
        "control_kwarg_keys": {
            name: sorted(keys) for name, keys in _CONTROL_KEYS.items()
        },
        "selection_contract": {
            "selected_only_by_pair_ids": True,
            "epoch_used_for_selection": False,
            "step_used_for_selection": False,
            "method_changes_schedule": False,
            "seed_specific_data_owned_by_provider": False,
            "raw_tensor_payloads_written": False,
        },
        "runtime_split_access": ["D_R"],
        "D_V_accessed": False,
        "D_T_accessed": False,
        "training_performed": False,
    }


@dataclass(frozen=True, eq=False)
class PairedFormalControlInputProvider:
    """Callable full-population provider for ``FormalControlKwargsProvider``."""

    dataset: str
    pair_catalog_fingerprint: str
    prepared_catalog_fingerprint: str
    preflight: FrozenControlPreflightFingerprints
    coordinate_basis: DCTCoordinateBasis
    pair_by_id: Mapping[str, PairExample]
    gt_union_by_sample: Mapping[str, Tensor]
    completion_plus_by_pair: Mapping[str, Tensor]
    completion_minus_by_pair: Mapping[str, Tensor]
    permuted_target_by_recipient: Mapping[str, Tensor]
    assignment_by_recipient: Mapping[str, Mapping[str, str]]
    feature_shape: tuple[int, int, int, int]
    evaluation_shape: tuple[int, int, int]
    provider_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("dataset must be a non-empty string")
        _require_fingerprint(
            self.pair_catalog_fingerprint,
            name="pair_catalog_fingerprint",
        )
        _require_fingerprint(
            self.prepared_catalog_fingerprint,
            name="prepared_catalog_fingerprint",
        )
        _require_fingerprint(
            self.provider_fingerprint,
            name="provider_fingerprint",
        )
        if not isinstance(self.preflight, FrozenControlPreflightFingerprints):
            raise TypeError("preflight must be FrozenControlPreflightFingerprints")
        if not isinstance(self.coordinate_basis, DCTCoordinateBasis):
            raise TypeError("coordinate_basis must be DCTCoordinateBasis")
        if (
            tuple(self.coordinate_basis.tensor.shape) != self.feature_shape
            or self.coordinate_basis.tensor.dtype != torch.float32
            or self.coordinate_basis.tensor.device.type != "cpu"
        ):
            raise ValueError("coordinate basis differs from the provider signature")
        pair_ids = set(self.pair_by_id)
        if not pair_ids or pair_ids != set(self.completion_plus_by_pair):
            raise ValueError("completion-plus population is incomplete")
        if pair_ids != set(self.completion_minus_by_pair):
            raise ValueError("completion-minus population is incomplete")
        if pair_ids != set(self.permuted_target_by_recipient):
            raise ValueError("permuted-target population is incomplete")
        if pair_ids != set(self.assignment_by_recipient):
            raise ValueError("permutation-assignment population is incomplete")
        if any(
            pair.sample_id not in self.gt_union_by_sample
            for pair in self.pair_by_id.values()
        ):
            raise ValueError("GT population omits a clean-pair source")
        if self._canonical_payload_fingerprint() != self.provider_fingerprint:
            raise ValueError("provider_fingerprint does not reproduce")

    def _canonical_payload(self) -> dict[str, object]:
        return _provider_payload(
            dataset=self.dataset,
            pair_catalog_fingerprint=self.pair_catalog_fingerprint,
            prepared_catalog_fingerprint=(
                self.prepared_catalog_fingerprint
            ),
            preflight=self.preflight,
            coordinate_basis=self.coordinate_basis,
            pair_by_id=self.pair_by_id,
            gt_union_by_sample=self.gt_union_by_sample,
            completion_plus_by_pair=self.completion_plus_by_pair,
            completion_minus_by_pair=self.completion_minus_by_pair,
            permuted_target_by_recipient=(
                self.permuted_target_by_recipient
            ),
            assignment_by_recipient=self.assignment_by_recipient,
            feature_shape=self.feature_shape,
            evaluation_shape=self.evaluation_shape,
        )

    def _canonical_payload_fingerprint(self) -> str:
        return stable_fingerprint(self._canonical_payload())

    def canonical_receipt(self) -> dict[str, object]:
        """Return a detached tensor-free receipt with its reproducible hash."""

        payload = self._canonical_payload()
        if stable_fingerprint(payload) != self.provider_fingerprint:
            raise RuntimeError("formal control provider inputs changed")
        receipt = deepcopy(payload)
        receipt["provider_fingerprint"] = self.provider_fingerprint
        return receipt

    def verify_unchanged(self) -> None:
        """Re-hash all frozen CPU inputs once before a formal run starts."""

        if self._canonical_payload_fingerprint() != self.provider_fingerprint:
            raise RuntimeError("formal control provider inputs changed")

    def _validate_call(
        self,
        *,
        control_kind: str,
        pairs: tuple[PairExample, PairExample],
        pair_batch: PairBatch,
        epoch: int,
        step: int,
        device: torch.device,
    ) -> tuple[str, str]:
        if control_kind not in CONTROL_KINDS:
            raise ValueError(f"unknown formal matched control {control_kind!r}")
        if (
            not isinstance(pairs, tuple)
            or len(pairs) != 2
            or any(not isinstance(pair, PairExample) for pair in pairs)
        ):
            raise TypeError("pairs must be a two-PairExample tuple")
        if not isinstance(pair_batch, PairBatch):
            raise TypeError("pair_batch must be PairBatch")
        # ``formal_batches_for_update`` already validates while stacking and
        # ``paired_control_train_step`` performs the full semantic preflight.
        # Keep this provider check structural to avoid a third full GPU tensor
        # scan on every one of the 32,000 updates.
        runtime_tensors = (
            pair_batch.feature,
            pair_batch.occupancy_plus,
            pair_batch.occupancy_minus,
            pair_batch.label_increment,
            pair_batch.image_valid_mask,
        )
        if any(not isinstance(tensor, Tensor) for tensor in runtime_tensors):
            raise TypeError("PairBatch fields must be tensors")
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError("epoch must be an integer")
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("step must be an integer")
        if epoch < 0 or epoch >= PAIRED_EPOCHS:
            raise ValueError("epoch must lie in [0, 800)")
        if step < 0 or step >= PAIRED_STEPS_PER_EPOCH:
            raise ValueError("step must lie in [0, 40)")
        if not isinstance(device, torch.device):
            raise TypeError("device must be torch.device")
        if any(
            tensor.device != device
            for tensor in runtime_tensors
        ):
            raise ValueError("requested device and PairBatch device differ")

        pair_ids = tuple(pair.pair_id for pair in pairs)
        sample_ids = tuple(pair.sample_id for pair in pairs)
        if len(set(pair_ids)) != 2 or len(set(sample_ids)) != 2:
            raise ValueError("formal pair batch must contain two source-disjoint pairs")
        if pair_ids != pair_batch.pair_ids:
            raise ValueError("pair arguments and PairBatch identities differ")
        if sample_ids != pair_batch.sample_ids:
            raise ValueError("pair arguments and PairBatch source identities differ")
        for index, pair_id in enumerate(pair_ids):
            bound = self.pair_by_id.get(pair_id)
            if bound is None:
                raise KeyError(f"pair_id {pair_id!r} is outside the frozen catalog")
            if (
                bound.sample_id != sample_ids[index]
                or bound.group_id != pair_batch.group_ids[index]
                or pair_batch.pair_kinds[index] != "clean_positive"
                or pair_batch.projection_visible[index] is not True
            ):
                raise ValueError(
                    "runtime pair metadata differs from the frozen catalog"
                )

        if tuple(pair_batch.feature.shape) != (
            2,
            *self.feature_shape[1:],
        ):
            raise ValueError("PairBatch feature shape differs from the freeze")
        if pair_batch.feature.dtype != torch.float32:
            raise TypeError("PairBatch feature dtype differs from the freeze")
        expected_eval = (2, *self.evaluation_shape)
        for name, tensor, dtype in (
            ("occupancy_plus", pair_batch.occupancy_plus, torch.bool),
            ("occupancy_minus", pair_batch.occupancy_minus, torch.bool),
            ("label_increment", pair_batch.label_increment, torch.float32),
            ("image_valid_mask", pair_batch.image_valid_mask, torch.bool),
        ):
            if tuple(tensor.shape) != expected_eval:
                raise ValueError(f"PairBatch {name} shape differs from the freeze")
            if tensor.dtype != dtype:
                raise TypeError(f"PairBatch {name} dtype differs from the freeze")
        return pair_ids

    def __call__(
        self,
        *,
        control_kind: str,
        pairs: tuple[PairExample, PairExample],
        pair_batch: PairBatch,
        epoch: int,
        step: int,
        device: torch.device,
    ) -> Mapping[str, object]:
        pair_ids = self._validate_call(
            control_kind=control_kind,
            pairs=pairs,
            pair_batch=pair_batch,
            epoch=epoch,
            step=step,
            device=device,
        )
        sample_ids = tuple(self.pair_by_id[pair_id].sample_id for pair_id in pair_ids)
        if control_kind == "independent_endpoint":
            return {
                "gt_union": torch.stack(
                    tuple(
                        self.gt_union_by_sample[sample_id]
                        for sample_id in sample_ids
                    ),
                    dim=0,
                ).to(device=device),
                "completion_plus": torch.stack(
                    tuple(
                        self.completion_plus_by_pair[pair_id]
                        for pair_id in pair_ids
                    ),
                    dim=0,
                ).to(device=device),
                "completion_minus": torch.stack(
                    tuple(
                        self.completion_minus_by_pair[pair_id]
                        for pair_id in pair_ids
                    ),
                    dim=0,
                ).to(device=device),
            }
        if control_kind == "after_only":
            return {
                "gt_union": torch.stack(
                    tuple(
                        self.gt_union_by_sample[sample_id]
                        for sample_id in sample_ids
                    ),
                    dim=0,
                ).to(device=device)
            }
        if control_kind == "coordinate_basis":
            return {"coordinate_basis": self.coordinate_basis}
        if control_kind == "target_permutation":
            return {
                "permuted_label_increment": torch.stack(
                    tuple(
                        self.permuted_target_by_recipient[pair_id]
                        for pair_id in pair_ids
                    ),
                    dim=0,
                ).to(device=device)
            }
        return {}


def build_paired_formal_control_provider(
    pair_catalog: PairCatalog,
    prepared_catalog: PreparedTrainingCatalog,
    preflight: FrozenControlPreflightFingerprints,
) -> PairedFormalControlInputProvider:
    """Build and seal all formal matched-control inputs from ``D_R`` only."""

    if not isinstance(pair_catalog, PairCatalog):
        raise TypeError("pair_catalog must be PairCatalog")
    if not isinstance(prepared_catalog, PreparedTrainingCatalog):
        raise TypeError("prepared_catalog must be PreparedTrainingCatalog")
    if not isinstance(preflight, FrozenControlPreflightFingerprints):
        raise TypeError("preflight must be FrozenControlPreflightFingerprints")
    if pair_catalog.split != "D_R":
        raise ValueError("formal control inputs permit only D_R")
    if not pair_catalog.clean_positive:
        raise ValueError("formal control inputs require clean-positive pairs")
    if pair_catalog.catalog_fingerprint != preflight.pair_catalog_fingerprint:
        raise RuntimeError("pair catalog differs from the control preflight")
    if stable_fingerprint(pair_catalog.canonical_payload()) != (
        pair_catalog.catalog_fingerprint
    ):
        raise RuntimeError("pair catalog fingerprint does not reproduce")

    feature_signatures = {
        (tuple(pair.feature.shape), pair.feature.dtype, pair.feature.device.type)
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
    feature_shape, feature_dtype, feature_device = next(iter(feature_signatures))
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
        raise TypeError("formal control source tensors must retain CPU cache dtypes")

    entries = {entry.sample_id: entry for entry in prepared_catalog.entries}
    if set(entries) != set(prepared_catalog.source_ids):
        raise RuntimeError("prepared source identities are inconsistent")
    if any(pair.sample_id not in entries for pair in pair_catalog.clean_positive):
        raise ValueError("prepared catalog is missing a clean-pair source")

    gt_union: dict[str, Tensor] = {}
    for sample_id in prepared_catalog.source_ids:
        entry = entries[sample_id]
        value = entry.gt.occupancy.unsqueeze(0).to(
            device="cpu",
            dtype=torch.bool,
        ).contiguous()
        if value.ndim != 3 or value.shape[0] != 1 or not torch.any(value):
            raise ValueError("every prepared source requires a non-empty GT union")
        if tuple(value.shape) != tuple(
            entry.source.state.image_valid_mask.unsqueeze(0).shape
        ):
            raise ValueError("prepared GT and image-valid grids differ")
        if torch.any(value & ~entry.source.state.image_valid_mask.unsqueeze(0)):
            raise ValueError("prepared GT union extends outside image_valid_mask")
        gt_union[sample_id] = value.detach().clone()

    pair_by_id: dict[str, PairExample] = {}
    completion_plus: dict[str, Tensor] = {}
    completion_minus: dict[str, Tensor] = {}
    for pair in pair_catalog.clean_positive:
        if pair.pair_id in pair_by_id:
            raise ValueError("clean pair IDs are not unique")
        entry = entries[pair.sample_id]
        union = gt_union[pair.sample_id]
        if (
            tuple(pair.image_valid_mask.shape) != tuple(union.shape)
            or tensor_content_fingerprint(entry.source.feature)
            != pair.feature_fingerprint
            or not torch.equal(
                pair.image_valid_mask,
                entry.source.state.image_valid_mask.unsqueeze(0),
            )
        ):
            raise RuntimeError("clean pair and prepared source bindings differ")
        batch_union = union.unsqueeze(0)
        batch_valid = pair.image_valid_mask.unsqueeze(0)
        build_geometry_matched_endpoint_supervision(
            pair.completion_plus.unsqueeze(0),
            pair.occupancy_plus.unsqueeze(0),
            batch_union,
            batch_valid,
        )
        build_geometry_matched_endpoint_supervision(
            pair.completion_minus.unsqueeze(0),
            pair.occupancy_minus.unsqueeze(0),
            batch_union,
            batch_valid,
        )
        build_after_only_synthetic_supervision(
            pair.label_increment.to(torch.bool).unsqueeze(0),
            pair.occupancy_minus.unsqueeze(0),
            batch_union,
            batch_valid,
        )
        pair_by_id[pair.pair_id] = pair
        completion_plus[pair.pair_id] = (
            pair.completion_plus.detach().clone().contiguous()
        )
        completion_minus[pair.pair_id] = (
            pair.completion_minus.detach().clone().contiguous()
        )

    basis = build_dct_coordinate_basis(
        channels=int(feature_shape[1]),
        height=int(feature_shape[2]),
        width=int(feature_shape[3]),
        dtype=feature_dtype,
    )
    if basis.basis_fingerprint != preflight.dct_basis_fingerprint:
        raise RuntimeError("DCT basis differs from the control preflight")

    plan = build_target_permutation(pair_catalog.clean_positive)
    if (
        not plan.ready
        or plan.plan_fingerprint
        != preflight.target_permutation_plan_fingerprint
    ):
        raise RuntimeError(
            "target permutation is not the frozen complete READY plan"
        )
    assignment_payloads = tuple(
        assignment.canonical_payload() for assignment in plan.assignments
    )
    if stable_fingerprint(list(assignment_payloads)) != (
        preflight.target_assignment_fingerprint
    ):
        raise RuntimeError(
            "target permutation assignments differ from the control preflight"
        )
    materialized = materialize_permuted_label_increments(
        pair_catalog.clean_positive,
        plan,
    )
    target_by_recipient = {
        pair_id: target.detach().clone().contiguous()
        for pair_id, target in zip(
            plan.canonical_pair_ids,
            materialized,
            strict=True,
        )
    }
    assignment_rows, _ = _assignment_binding(
        assignment_payloads=assignment_payloads,
        target_by_recipient=target_by_recipient,
    )
    assignment_by_recipient: dict[str, Mapping[str, str]] = {}
    donor_ids: set[str] = set()
    for assignment, row in zip(
        plan.assignments,
        assignment_rows,
        strict=True,
    ):
        recipient = pair_by_id[assignment.recipient_pair_id]
        donor = pair_by_id[assignment.donor_pair_id]
        target = target_by_recipient[recipient.pair_id]
        if (
            not target_permutation_compatible(recipient, donor)
            or recipient.sample_id == donor.sample_id
            or recipient.pair_id == donor.pair_id
            or tensor_content_fingerprint(donor.clean_increment)
            != assignment.donor_target_fingerprint
            or not torch.equal(
                target,
                donor.clean_increment.to(torch.float32),
            )
            or tuple(target.shape) != tuple(recipient.label_increment.shape)
            or target.dtype != torch.float32
            or target.device.type != "cpu"
            or torch.any(target.to(torch.bool) & ~recipient.image_valid_mask)
            or not torch.any(target)
            or not torch.any(
                recipient.image_valid_mask & ~target.to(torch.bool)
            )
        ):
            raise RuntimeError("target-permutation runtime closure failed")
        donor_ids.add(donor.pair_id)
        assignment_by_recipient[recipient.pair_id] = MappingProxyType(row)
    if (
        set(assignment_by_recipient) != set(pair_by_id)
        or donor_ids != set(pair_by_id)
    ):
        raise RuntimeError("target permutation does not close both marginals")

    prepared_fingerprint = prepared_training_catalog_fingerprint(
        prepared_catalog
    )
    frozen_pair_by_id = MappingProxyType(pair_by_id)
    frozen_gt_union = MappingProxyType(gt_union)
    frozen_completion_plus = MappingProxyType(completion_plus)
    frozen_completion_minus = MappingProxyType(completion_minus)
    frozen_targets = MappingProxyType(target_by_recipient)
    frozen_assignments = MappingProxyType(assignment_by_recipient)
    frozen_feature_shape = tuple(int(value) for value in feature_shape)
    frozen_evaluation_shape = tuple(int(value) for value in evaluation_shape)
    fingerprint = stable_fingerprint(
        _provider_payload(
            dataset=pair_catalog.dataset,
            pair_catalog_fingerprint=pair_catalog.catalog_fingerprint,
            prepared_catalog_fingerprint=prepared_fingerprint,
            preflight=preflight,
            coordinate_basis=basis,
            pair_by_id=frozen_pair_by_id,
            gt_union_by_sample=frozen_gt_union,
            completion_plus_by_pair=frozen_completion_plus,
            completion_minus_by_pair=frozen_completion_minus,
            permuted_target_by_recipient=frozen_targets,
            assignment_by_recipient=frozen_assignments,
            feature_shape=frozen_feature_shape,
            evaluation_shape=frozen_evaluation_shape,
        )
    )
    provider = PairedFormalControlInputProvider(
        dataset=pair_catalog.dataset,
        pair_catalog_fingerprint=pair_catalog.catalog_fingerprint,
        prepared_catalog_fingerprint=prepared_fingerprint,
        preflight=preflight,
        coordinate_basis=basis,
        pair_by_id=frozen_pair_by_id,
        gt_union_by_sample=frozen_gt_union,
        completion_plus_by_pair=frozen_completion_plus,
        completion_minus_by_pair=frozen_completion_minus,
        permuted_target_by_recipient=frozen_targets,
        assignment_by_recipient=frozen_assignments,
        feature_shape=frozen_feature_shape,
        evaluation_shape=frozen_evaluation_shape,
        provider_fingerprint=fingerprint,
    )
    provider.verify_unchanged()
    return provider


__all__ = [
    "PAIRED_FORMAL_CONTROL_GT_BINDING_SCHEMA",
    "PAIRED_FORMAL_CONTROL_PAIR_BINDING_SCHEMA",
    "PAIRED_FORMAL_CONTROL_PERMUTATION_BINDING_SCHEMA",
    "PAIRED_FORMAL_CONTROL_PREFLIGHT_BINDING_SCHEMA",
    "PAIRED_FORMAL_CONTROL_PROVIDER_SCHEMA",
    "FrozenControlPreflightFingerprints",
    "PairedFormalControlInputProvider",
    "build_paired_formal_control_provider",
    "load_frozen_control_preflight_fingerprints",
]
