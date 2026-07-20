"""Validation-frozen calibration for full CURE and its Base@B control."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from math import isfinite
from numbers import Real
import os
from pathlib import Path
import re
from typing import Iterable, Sequence

import torch
from torch import Tensor

from ..calibration import FalseAlarmBudget, ThresholdSelection, anchor_miss_ids
from ..config import OccupancyConfig
from ..instances import instances_from_binary_mask, union_instance_masks
from ..matching import match_components
from ..metrics import (
    AggregateEvaluation,
    aggregate_evaluations,
    evaluate_binary_prediction,
    full_pipeline_reachable_anchor_miss_ids,
)
from .descriptors import dilate_mask
from .decoder import CUREResidualDecoder
from .model import CUREModel, noisy_or
from .protocol import (
    CUREProtocol,
    decoder_state_fingerprint,
    tensor_content_fingerprint,
)


_UNSET = object()
_CALIBRATION_SAMPLE_SEAL = object()
_FROZEN_BASE_SEAL = object()
_FROZEN_CURE_SEAL = object()
_CALIBRATION_MODEL_EXECUTION = "cure-model-execution-v1"
_CALIBRATION_TEST_FIXTURE = "private-test-map-fixture-v1"
_TEST_LEDGER_SCHEMA = "cure-dt-one-shot-ledger-v1"


def _map_2d(value: Tensor, *, name: str, binary: bool = False) -> Tensor:
    tensor = torch.as_tensor(value, device="cpu")
    if tensor.ndim == 4 and tensor.shape[:2] == (1, 1):
        tensor = tensor[0, 0]
    elif tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2:
        raise ValueError(f"{name} must contain one 2D map")
    if binary:
        if torch.any((tensor != 0) & (tensor != 1)):
            raise ValueError(f"{name} must be binary")
        return tensor.to(torch.bool).contiguous()
    tensor = tensor.to(torch.float32)
    if not torch.isfinite(tensor).all() or torch.any(
        (tensor < 0.0) | (tensor > 1.0)
    ):
        raise ValueError(f"{name} must be finite and lie in [0,1]")
    return tensor.contiguous()


def _digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CURECalibrationSample:
    """One content-bound D_V or D_T output issued by an exact CUREModel run."""

    sample_id: str
    base_probability: Tensor
    effective_residual_probability: Tensor
    gt_mask: Tensor
    split_role: str
    protocol_fingerprint: str
    base_fingerprint: str
    base_state_fingerprint: str
    decoder_fingerprint: str
    input_fingerprint: str
    provenance: str
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        issuing = self._seal is _CALIBRATION_SAMPLE_SEAL
        if not issuing and not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _CALIBRATION_SAMPLE_SEAL
        ):
            raise ValueError("calibration sample was not issued by the canonical binder")
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if self.split_role not in {"D_V", "D_T"}:
            raise ValueError("calibration sample must be tagged D_V or D_T")
        _digest("protocol_fingerprint", self.protocol_fingerprint)
        if not isinstance(self.base_fingerprint, str) or not self.base_fingerprint:
            raise ValueError("base_fingerprint must be non-empty")
        _digest("base_state_fingerprint", self.base_state_fingerprint)
        _digest("decoder_fingerprint", self.decoder_fingerprint)
        _digest("input_fingerprint", self.input_fingerprint)
        if self.provenance not in {
            _CALIBRATION_MODEL_EXECUTION,
            _CALIBRATION_TEST_FIXTURE,
        }:
            raise ValueError("unknown calibration-sample provenance")
        # Validate the local map contract before sealing its content.
        base = _map_2d(self.base_probability, name="base_probability")
        residual = _map_2d(
            self.effective_residual_probability,
            name="effective_residual_probability",
        )
        gt = _map_2d(self.gt_mask, name="gt_mask", binary=True)
        if base.shape != residual.shape or base.shape != gt.shape:
            raise ValueError("calibration maps must share a grid")
        if issuing:
            object.__setattr__(
                self, "_seal", (_CALIBRATION_SAMPLE_SEAL, self.fingerprint)
            )
        elif self._seal[1] != self.fingerprint:
            raise ValueError("calibration sample content differs from its receipt")

    @property
    def fingerprint(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(
            repr(
                (
                    self.sample_id,
                    self.split_role,
                    self.protocol_fingerprint,
                    self.base_fingerprint,
                    self.base_state_fingerprint,
                    self.decoder_fingerprint,
                    self.input_fingerprint,
                    self.provenance,
                )
            ).encode("utf-8")
        )
        for name, value in (
            ("base_probability", self.base_probability),
            ("effective_residual_probability", self.effective_residual_probability),
            ("gt_mask", self.gt_mask),
        ):
            hasher.update(tensor_content_fingerprint(name, value).encode("ascii"))
        return hasher.hexdigest()

    def validate_receipt(self) -> None:
        if not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _CALIBRATION_SAMPLE_SEAL
            and self._seal[1] == self.fingerprint
        ):
            raise ValueError("calibration sample content differs from its receipt")

    @classmethod
    def from_model(
        cls,
        sample_id: str,
        images: Tensor,
        gt_mask: Tensor,
        *,
        split_role: str,
        model: CUREModel,
    ) -> "CURECalibrationSample":
        """Run the exact frozen-base/CURE decoder and issue one formal sample."""

        if type(model) is not CUREModel:
            raise TypeError("model must be the exact CUREModel carrier")
        protocol = model.protocol
        protocol.validate_receipt()
        protocol.assert_sample(sample_id, split=split_role)
        if not isinstance(images, Tensor) or images.ndim != 4 or images.shape[0] != 1:
            raise ValueError("formal calibration images must have shape [1,C,H,W]")
        if not images.is_floating_point() or not torch.isfinite(images).all():
            raise ValueError("formal calibration images must be finite floating tensors")
        normalized_gt = _map_2d(gt_mask, name="gt_mask", binary=True)
        if tuple(normalized_gt.shape) != tuple(images.shape[-2:]):
            raise ValueError("gt_mask and calibration image grids differ")

        base_state = model.base_state_fingerprint
        decoder_fingerprint = decoder_state_fingerprint(model.decoder, protocol)
        input_fingerprint = tensor_content_fingerprint("calibration_input", images)
        was_training = model.training
        decoder_was_training = model.decoder.training
        model.eval()
        try:
            with torch.no_grad():
                output = model(images)
        finally:
            model.train(was_training)
            model.decoder.train(decoder_was_training)
        if model.base_state_fingerprint != base_state:
            raise RuntimeError("frozen base changed during calibration inference")
        if decoder_state_fingerprint(model.decoder, protocol) != decoder_fingerprint:
            raise RuntimeError("residual decoder changed during calibration inference")

        result = cls(
            sample_id=sample_id,
            base_probability=output.base_probability[0, 0].detach().cpu().clone(),
            effective_residual_probability=(
                output.residual_probability[0, 0].detach().cpu().clone()
            ),
            gt_mask=normalized_gt.detach().cpu().clone(),
            split_role=split_role,
            protocol_fingerprint=protocol.fingerprint,
            base_fingerprint=protocol.base_fingerprint,
            base_state_fingerprint=base_state,
            decoder_fingerprint=decoder_fingerprint,
            input_fingerprint=input_fingerprint,
            provenance=_CALIBRATION_MODEL_EXECUTION,
            _seal=_CALIBRATION_SAMPLE_SEAL,
        )
        result._normalized(
            protocol,
            expected_split=split_role,
            decoder_fingerprint=decoder_fingerprint,
            base_state_fingerprint=base_state,
            require_model_execution=True,
        )
        return result

    @classmethod
    def _bind_maps_for_test_only(
        cls,
        sample_id: str,
        base_probability: Tensor,
        effective_residual_probability: Tensor,
        gt_mask: Tensor,
        *,
        split_role: str,
        protocol: CUREProtocol,
        decoder: CUREResidualDecoder,
    ) -> "CURECalibrationSample":
        """Issue a synthetic map fixture; formal D_V/D_T APIs reject it."""

        if not isinstance(protocol, CUREProtocol):
            raise TypeError("protocol must be CUREProtocol")
        protocol.assert_sample(sample_id, split=split_role)
        decoder_fingerprint = decoder_state_fingerprint(decoder, protocol)
        fixture_fingerprint = hashlib.sha256(
            f"test-only:{protocol.base_fingerprint}".encode("utf-8")
        ).hexdigest()
        result = cls(
            sample_id=sample_id,
            base_probability=base_probability,
            effective_residual_probability=effective_residual_probability,
            gt_mask=gt_mask,
            split_role=split_role,
            protocol_fingerprint=protocol.fingerprint,
            base_fingerprint=protocol.base_fingerprint,
            base_state_fingerprint=fixture_fingerprint,
            decoder_fingerprint=decoder_fingerprint,
            input_fingerprint=fixture_fingerprint,
            provenance=_CALIBRATION_TEST_FIXTURE,
            _seal=_CALIBRATION_SAMPLE_SEAL,
        )
        result._normalized(
            protocol,
            expected_split=split_role,
            decoder_fingerprint=decoder_fingerprint,
            base_state_fingerprint=None,
            require_model_execution=False,
        )
        return result

    def _normalized(
        self,
        protocol: CUREProtocol,
        *,
        expected_split: str,
        decoder_fingerprint: str | None,
        base_state_fingerprint: str | None,
        require_model_execution: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.validate_receipt()
        if require_model_execution and self.provenance != _CALIBRATION_MODEL_EXECUTION:
            raise ValueError(
                "formal calibration requires a sample issued by "
                "CURECalibrationSample.from_model"
            )
        protocol.assert_sample(self.sample_id, split=expected_split)
        if self.split_role != expected_split:
            raise ValueError("calibration sample split tag differs from its use")
        if self.protocol_fingerprint != protocol.fingerprint:
            raise ValueError("calibration sample uses a different CURE protocol")
        if self.base_fingerprint != protocol.base_fingerprint:
            raise ValueError("calibration sample uses a different frozen base")
        if (
            base_state_fingerprint is not None
            and self.base_state_fingerprint != base_state_fingerprint
        ):
            raise ValueError("calibration sample uses different frozen-base contents")
        _digest("decoder_fingerprint", self.decoder_fingerprint)
        if (
            decoder_fingerprint is not None
            and self.decoder_fingerprint != decoder_fingerprint
        ):
            raise ValueError("calibration sample uses a different residual decoder")
        base = _map_2d(self.base_probability, name="base_probability")
        residual = _map_2d(
            self.effective_residual_probability,
            name="effective_residual_probability",
        )
        gt = _map_2d(self.gt_mask, name="gt_mask", binary=True)
        if base.shape != residual.shape or base.shape != gt.shape:
            raise ValueError("calibration maps must share a grid")
        occupancy = base >= protocol.residual_config.occupancy_threshold
        exclusion = dilate_mask(
            occupancy, protocol.residual_config.suppression_radius
        )
        if torch.any(residual[exclusion] != 0.0):
            raise ValueError(
                "effective residual must be zero on the protocol exclusion mask"
            )
        return base, residual, gt


def _validate_sample_set(
    samples: Sequence[CURECalibrationSample],
    protocol: CUREProtocol,
    *,
    expected_split: str,
    decoder_fingerprint: str | None,
    base_state_fingerprint: str,
) -> tuple[tuple[CURECalibrationSample, Tensor, Tensor, Tensor], ...]:
    if not isinstance(protocol, CUREProtocol):
        raise TypeError("protocol must be CUREProtocol")
    items = tuple(samples)
    if not items or any(not isinstance(item, CURECalibrationSample) for item in items):
        raise TypeError("samples must be a non-empty CURECalibrationSample sequence")
    expected_ids = tuple(
        sorted(
            sample_id
            for sample_id, split, _ in protocol.manifest_membership
            if split == expected_split
        )
    )
    observed_ids = tuple(sorted(item.sample_id for item in items))
    if observed_ids != expected_ids:
        raise ValueError(
            f"formal {expected_split} evaluation requires exactly its manifest samples"
        )
    normalized = []
    for sample in sorted(items, key=lambda item: item.sample_id):
        base, residual, gt = sample._normalized(
            protocol,
            expected_split=expected_split,
            decoder_fingerprint=decoder_fingerprint,
            base_state_fingerprint=base_state_fingerprint,
            require_model_execution=True,
        )
        normalized.append((sample, base, residual, gt))
    return tuple(normalized)


def _evaluate_threshold(
    samples: Sequence[CURECalibrationSample],
    threshold: float | None,
    protocol: CUREProtocol,
    *,
    expected_split: str,
    decoder_fingerprint: str | None,
    base_state_fingerprint: str,
    residual_enabled: bool,
) -> AggregateEvaluation:
    tau_o = protocol.residual_config.occupancy_threshold
    if threshold is not None:
        threshold = float(threshold)
        if not isfinite(threshold) or not 0.0 <= threshold <= tau_o:
            raise ValueError("final threshold must lie in [0, occupancy_threshold]")
    normalized = _validate_sample_set(
        samples,
        protocol,
        expected_split=expected_split,
        decoder_fingerprint=decoder_fingerprint if residual_enabled else None,
        base_state_fingerprint=base_state_fingerprint,
    )
    return _evaluate_normalized_threshold(
        normalized,
        threshold,
        protocol,
        residual_enabled=residual_enabled,
    )


def _evaluate_normalized_threshold(
    normalized: Sequence[tuple[CURECalibrationSample, Tensor, Tensor, Tensor]],
    threshold: float | None,
    protocol: CUREProtocol,
    *,
    residual_enabled: bool,
) -> AggregateEvaluation:
    """Evaluate already authenticated maps without rereading a D_T sample set."""

    tau_o = protocol.residual_config.occupancy_threshold
    if threshold is not None:
        threshold = float(threshold)
        if not isfinite(threshold) or not 0.0 <= threshold <= tau_o:
            raise ValueError("final threshold must lie in [0, occupancy_threshold]")
    occupancy_config = OccupancyConfig(threshold=tau_o)
    records = []
    for _, base, residual, gt in normalized:
        occupancy = base >= tau_o
        anchors = anchor_miss_ids(
            base, gt, occupancy_config, protocol.match_config
        )
        reachable = full_pipeline_reachable_anchor_miss_ids(
            occupancy, gt, protocol.match_config
        )
        if threshold is None:
            prediction = occupancy
            residual_mask = torch.zeros_like(occupancy)
        else:
            effective_residual = residual if residual_enabled else torch.zeros_like(residual)
            fused = noisy_or(base, effective_residual)
            prediction = fused >= threshold
            residual_mask = torch.zeros_like(prediction)
            if residual_enabled:
                # Attribute a recovered anchor target to CURE only when it is
                # matched after fusion but remains unmatched in the exact
                # base-only counterfactual at the same tau_f.  Merely adding a
                # residual pixel inside an already base-recovered GT is not
                # residual-caused recovery.
                gt_instances = instances_from_binary_mask(
                    gt, connectivity=8, min_area=1
                )
                base_instances = instances_from_binary_mask(
                    base >= threshold, connectivity=8, min_area=1
                )
                fused_instances = instances_from_binary_mask(
                    prediction, connectivity=8, min_area=1
                )
                base_match = match_components(
                    base_instances, gt_instances, protocol.match_config
                )
                fused_match = match_components(
                    fused_instances, gt_instances, protocol.match_config
                )
                necessary_ids = (
                    set(anchors)
                    & set(fused_match.matched_gt_ids)
                    - set(base_match.matched_gt_ids)
                )
                residual_mask = union_instance_masks(
                    gt_instances, sorted(necessary_ids)
                )
        records.append(
            evaluate_binary_prediction(
                prediction,
                gt,
                protocol.match_config,
                anchor_miss_ids=anchors,
                reachable_anchor_miss_ids=reachable,
                residual_mask=residual_mask,
            )
        )
    return aggregate_evaluations(records)


def evaluate_cure_threshold(
    samples: Sequence[CURECalibrationSample],
    threshold: float | None,
    protocol: CUREProtocol,
    *,
    model: CUREModel,
    split_role: str = "D_V",
) -> AggregateEvaluation:
    """Evaluate one tagged point; formal threshold search remains D_V-only."""

    if split_role != "D_V":
        raise RuntimeError(
            "direct threshold evaluation is D_V-only; use a frozen protocol on D_T"
        )
    if type(model) is not CUREModel:
        raise TypeError("model must be the exact CUREModel carrier")
    if model.protocol.fingerprint != protocol.fingerprint:
        raise ValueError("model uses a different CURE protocol")
    fingerprint = decoder_state_fingerprint(model.decoder, protocol)
    base_state = model.base_state_fingerprint
    return _evaluate_threshold(
        samples,
        threshold,
        protocol,
        expected_split=split_role,
        decoder_fingerprint=fingerprint,
        base_state_fingerprint=base_state,
        residual_enabled=True,
    )


def _sample_set_fingerprint(
    samples: Sequence[CURECalibrationSample],
) -> str:
    items = tuple(samples)
    if not items or any(not isinstance(item, CURECalibrationSample) for item in items):
        raise TypeError("samples must be a non-empty CURECalibrationSample sequence")
    hasher = hashlib.sha256()
    for sample in sorted(items, key=lambda item: item.sample_id):
        sample.validate_receipt()
        hasher.update(sample.fingerprint.encode("ascii"))
    return hasher.hexdigest()


@dataclass(frozen=True)
class CURETestEvaluationLedger:
    """Atomic, process-persistent one-shot ledger for formal D_T metrics.

    A marker is keyed by the frozen operating-point receipt and evaluation arm,
    not by ``experiment_id``.  Changing the experiment label therefore cannot
    reopen the same frozen D_T evaluation.  There is intentionally no reset or
    recovery API: a failed process after reservation conservatively consumes
    that formal pass.
    """

    root: Path

    def __post_init__(self) -> None:
        root = Path(self.root)
        if not root.is_absolute():
            raise ValueError("D_T ledger root must be an absolute path")
        object.__setattr__(self, "root", root)

    def _consume(
        self,
        *,
        experiment_id: str,
        frozen_protocol_fingerprint: str,
        arm: str,
        sample_set_fingerprint: str,
    ) -> None:
        if not isinstance(experiment_id, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", experiment_id
        ) is None:
            raise ValueError(
                "experiment_id must be 1-128 safe alphanumeric/._- characters"
            )
        _digest("frozen_protocol_fingerprint", frozen_protocol_fingerprint)
        _digest("sample_set_fingerprint", sample_set_fingerprint)
        if arm != "paired_cure_base_at_budget":
            raise ValueError("unknown D_T evaluation arm")
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ValueError("D_T ledger root is not a directory")
        marker_key = hashlib.sha256(
            repr(
                (_TEST_LEDGER_SCHEMA, frozen_protocol_fingerprint, arm)
            ).encode("utf-8")
        ).hexdigest()
        marker = self.root / f"{marker_key}.json"
        payload = json.dumps(
            {
                "schema": _TEST_LEDGER_SCHEMA,
                "experiment_id": experiment_id,
                "frozen_protocol_fingerprint": frozen_protocol_fingerprint,
                "arm": arm,
                "sample_set_fingerprint": sample_set_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(marker, flags, 0o600)
        except FileExistsError as error:
            raise RuntimeError(
                "this frozen D_T evaluation arm was already consumed"
            ) from error
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written < 1:
                    raise OSError("failed to persist D_T ledger marker")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


@dataclass(frozen=True)
class FrozenCUREBaseThresholdProtocol:
    """Paired Base@B operating point under the identical anchor and budget."""

    protocol: CUREProtocol
    base_state_fingerprint: str
    base_threshold: float
    budget: FalseAlarmBudget
    validation_metrics: AggregateEvaluation
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        issuing = self._seal is _FROZEN_BASE_SEAL
        if not issuing and not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _FROZEN_BASE_SEAL
        ):
            raise ValueError("Base@B protocol was not issued by D_V selection")
        if not isinstance(self.protocol, CUREProtocol):
            raise TypeError("protocol must be CUREProtocol")
        _digest("base_state_fingerprint", self.base_state_fingerprint)
        if not isinstance(self.budget, FalseAlarmBudget):
            raise TypeError("budget must be FalseAlarmBudget")
        threshold = float(self.base_threshold)
        if (
            not isfinite(threshold)
            or not 0.0 <= threshold <= self.protocol.residual_config.occupancy_threshold
        ):
            raise ValueError("Base@B threshold must preserve the frozen anchor")
        object.__setattr__(self, "base_threshold", threshold)
        if issuing:
            object.__setattr__(self, "_seal", (_FROZEN_BASE_SEAL, self.fingerprint))
        elif self._seal[1] != self.fingerprint:
            raise ValueError("Base@B frozen protocol content differs from its receipt")

    @property
    def fingerprint(self) -> str:
        payload = repr(
            (
                self.protocol.fingerprint,
                self.base_state_fingerprint,
                self.base_threshold,
                self.budget,
                self.validation_metrics,
            )
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_receipt(self) -> None:
        if not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _FROZEN_BASE_SEAL
            and self._seal[1] == self.fingerprint
        ):
            raise ValueError("Base@B frozen protocol content differs from its receipt")

    def evaluate_test(
        self,
        samples: Sequence[CURECalibrationSample],
        **_: object,
    ) -> None:
        raise RuntimeError(
            "standalone Base@B D_T evaluation is forbidden; use the parent "
            "FrozenCUREThresholdProtocol.evaluate_test paired entrypoint"
        )


@dataclass(frozen=True)
class CURETestEvaluationResult:
    """Paired CURE/Base@B metrics from one atomically reserved D_T pass."""

    cure: AggregateEvaluation
    base_at_budget: AggregateEvaluation
    experiment_id: str
    frozen_protocol_fingerprint: str
    sample_set_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.cure, AggregateEvaluation) or not isinstance(
            self.base_at_budget, AggregateEvaluation
        ):
            raise TypeError("paired D_T results must contain aggregate metrics")
        if not isinstance(self.experiment_id, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.experiment_id
        ) is None:
            raise ValueError("invalid paired D_T experiment_id")
        _digest("frozen_protocol_fingerprint", self.frozen_protocol_fingerprint)
        _digest("sample_set_fingerprint", self.sample_set_fingerprint)


@dataclass(frozen=True)
class FrozenCUREThresholdProtocol:
    """D_V-frozen CURE operating point for exactly one untouched D_T pass."""

    protocol: CUREProtocol
    base_state_fingerprint: str
    decoder_fingerprint: str
    final_threshold: float | None
    budget: FalseAlarmBudget
    validation_metrics: AggregateEvaluation
    base_at_budget: FrozenCUREBaseThresholdProtocol
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        issuing = self._seal is _FROZEN_CURE_SEAL
        if not issuing and not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _FROZEN_CURE_SEAL
        ):
            raise ValueError("CURE protocol was not issued by D_V selection")
        if not isinstance(self.protocol, CUREProtocol):
            raise TypeError("protocol must be CUREProtocol")
        _digest("base_state_fingerprint", self.base_state_fingerprint)
        _digest("decoder_fingerprint", self.decoder_fingerprint)
        if not isinstance(self.budget, FalseAlarmBudget):
            raise TypeError("budget must be FalseAlarmBudget")
        if self.final_threshold is not None:
            threshold = float(self.final_threshold)
            if (
                not isfinite(threshold)
                or not 0.0
                <= threshold
                <= self.protocol.residual_config.occupancy_threshold
            ):
                raise ValueError("final threshold must preserve the frozen anchor")
            object.__setattr__(self, "final_threshold", threshold)
        if self.base_at_budget.protocol != self.protocol:
            raise ValueError("CURE and Base@B protocols differ")
        if self.base_at_budget.base_state_fingerprint != self.base_state_fingerprint:
            raise ValueError("CURE and Base@B frozen-base contents differ")
        if self.base_at_budget.budget != self.budget:
            raise ValueError("CURE and Base@B budgets differ")
        self.base_at_budget.validate_receipt()
        if issuing:
            object.__setattr__(self, "_seal", (_FROZEN_CURE_SEAL, self.fingerprint))
        elif self._seal[1] != self.fingerprint:
            raise ValueError("CURE frozen protocol content differs from its receipt")

    @property
    def fingerprint(self) -> str:
        payload = repr(
            (
                self.protocol.fingerprint,
                self.base_state_fingerprint,
                self.decoder_fingerprint,
                self.final_threshold,
                self.budget,
                self.validation_metrics,
                self.base_at_budget.fingerprint,
            )
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_receipt(self) -> None:
        self.base_at_budget.validate_receipt()
        if not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _FROZEN_CURE_SEAL
            and self._seal[1] == self.fingerprint
        ):
            raise ValueError("CURE frozen protocol content differs from its receipt")

    def evaluate_test(
        self,
        samples: Sequence[CURECalibrationSample],
        *,
        ledger: CURETestEvaluationLedger,
        experiment_id: str,
        proposed_final_threshold: object = _UNSET,
        proposed_occupancy_threshold: float | None = None,
        proposed_decoder_fingerprint: str | None = None,
        proposed_base_threshold: float | None = None,
    ) -> CURETestEvaluationResult:
        self.validate_receipt()
        if proposed_final_threshold is not _UNSET:
            proposed = (
                None
                if proposed_final_threshold is None
                else float(proposed_final_threshold)
            )
            if proposed != self.final_threshold:
                raise RuntimeError("D_T final-threshold retuning is forbidden")
        if (
            proposed_occupancy_threshold is not None
            and float(proposed_occupancy_threshold)
            != self.protocol.residual_config.occupancy_threshold
        ):
            raise RuntimeError("D_T occupancy-threshold retuning is forbidden")
        if (
            proposed_decoder_fingerprint is not None
            and proposed_decoder_fingerprint != self.decoder_fingerprint
        ):
            raise RuntimeError("D_T decoder substitution is forbidden")
        if (
            proposed_base_threshold is not None
            and float(proposed_base_threshold)
            != self.base_at_budget.base_threshold
        ):
            raise RuntimeError("D_T Base@B threshold retuning is forbidden")
        if not isinstance(ledger, CURETestEvaluationLedger):
            raise TypeError("ledger must be CURETestEvaluationLedger")
        sample_fingerprint = _sample_set_fingerprint(samples)
        ledger._consume(
            experiment_id=experiment_id,
            frozen_protocol_fingerprint=self.fingerprint,
            arm="paired_cure_base_at_budget",
            sample_set_fingerprint=sample_fingerprint,
        )
        normalized = _validate_sample_set(
            samples,
            self.protocol,
            expected_split="D_T",
            decoder_fingerprint=self.decoder_fingerprint,
            base_state_fingerprint=self.base_state_fingerprint,
        )
        cure_metrics = _evaluate_normalized_threshold(
            normalized,
            self.final_threshold,
            self.protocol,
            residual_enabled=True,
        )
        base_metrics = _evaluate_normalized_threshold(
            normalized,
            self.base_at_budget.base_threshold,
            self.protocol,
            residual_enabled=False,
        )
        cure_metrics = replace(
            cure_metrics,
            budget_violation=not self.budget.accepts(cure_metrics),
        )
        base_metrics = replace(
            base_metrics,
            budget_violation=not self.budget.accepts(base_metrics),
        )
        return CURETestEvaluationResult(
            cure=cure_metrics,
            base_at_budget=base_metrics,
            experiment_id=experiment_id,
            frozen_protocol_fingerprint=self.fingerprint,
            sample_set_fingerprint=sample_fingerprint,
        )


@dataclass(frozen=True)
class CUREThresholdSelection:
    """Threshold search result plus the only protocol allowed on D_T."""

    selection: ThresholdSelection
    frozen_protocol: FrozenCUREThresholdProtocol | None

    def __post_init__(self) -> None:
        if not isinstance(self.selection, ThresholdSelection):
            raise TypeError("selection must be ThresholdSelection")
        if self.selection.feasible != (self.frozen_protocol is not None):
            raise ValueError("feasibility and frozen protocol disagree")
        if self.frozen_protocol is not None:
            self.frozen_protocol.validate_receipt()

    @property
    def threshold(self) -> float | None:
        return self.selection.threshold

    @property
    def metrics(self) -> AggregateEvaluation | None:
        return self.selection.metrics

    @property
    def feasible(self) -> bool:
        return self.selection.feasible

    @property
    def reason(self) -> str | None:
        return self.selection.reason


def _threshold_candidates(
    thresholds: Iterable[float], tau_o: float
) -> tuple[float, ...]:
    candidates: set[float] = set()
    for value in thresholds:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("threshold candidates must be real numbers")
        candidate = float(value)
        if not isfinite(candidate):
            raise ValueError("threshold candidates must be finite")
        if not 0.0 <= candidate <= tau_o:
            raise ValueError(
                "threshold candidates must lie in [0, occupancy_threshold]"
            )
        candidates.add(candidate)
    return tuple(sorted(candidates))


def select_cure_threshold(
    samples: Sequence[CURECalibrationSample],
    thresholds: Iterable[float],
    protocol: CUREProtocol,
    budget: FalseAlarmBudget,
    *,
    model: CUREModel,
) -> CUREThresholdSelection:
    """Freeze CURE and paired Base@B thresholds using D_V exactly once."""

    if not isinstance(protocol, CUREProtocol):
        raise TypeError("protocol must be CUREProtocol")
    if not isinstance(budget, FalseAlarmBudget):
        raise TypeError("budget must be FalseAlarmBudget")
    if type(model) is not CUREModel:
        raise TypeError("model must be the exact CUREModel carrier")
    if model.protocol.fingerprint != protocol.fingerprint:
        raise ValueError("model uses a different CURE protocol")
    decoder_fingerprint = decoder_state_fingerprint(model.decoder, protocol)
    base_state_fingerprint = model.base_state_fingerprint
    tau_o = protocol.residual_config.occupancy_threshold
    candidates = _threshold_candidates(thresholds, tau_o)

    anchor_metrics = _evaluate_threshold(
        samples,
        None,
        protocol,
        expected_split="D_V",
        decoder_fingerprint=None,
        base_state_fingerprint=base_state_fingerprint,
        residual_enabled=False,
    )
    if not budget.accepts(anchor_metrics):
        raise ValueError("FA budget cannot be below the frozen base anchor")

    base_feasible: list[tuple[float, AggregateEvaluation]] = []
    for threshold in tuple(sorted(set((*candidates, tau_o)))):
        metrics = _evaluate_threshold(
            samples,
            threshold,
            protocol,
            expected_split="D_V",
            decoder_fingerprint=None,
            base_state_fingerprint=base_state_fingerprint,
            residual_enabled=False,
        )
        if budget.accepts(metrics):
            base_feasible.append((threshold, metrics))
    base_threshold, base_metrics = max(
        base_feasible,
        key=lambda item: (item[1].pd, item[1].net_rmr, item[0]),
    )
    frozen_base = FrozenCUREBaseThresholdProtocol(
        protocol=protocol,
        base_state_fingerprint=base_state_fingerprint,
        base_threshold=base_threshold,
        budget=budget,
        validation_metrics=base_metrics,
        _seal=_FROZEN_BASE_SEAL,
    )

    feasible: list[tuple[float | None, AggregateEvaluation]] = []
    for threshold in (None, *candidates):
        metrics = _evaluate_threshold(
            samples,
            threshold,
            protocol,
            expected_split="D_V",
            decoder_fingerprint=decoder_fingerprint,
            base_state_fingerprint=base_state_fingerprint,
            residual_enabled=True,
        )
        if budget.accepts(metrics):
            feasible.append((threshold, metrics))
    if not feasible:
        selection = ThresholdSelection(
            threshold=None,
            metrics=None,
            feasible=False,
            reason="no CURE threshold satisfies the preregistered FA budget",
        )
        return CUREThresholdSelection(selection, None)
    threshold, metrics = max(
        feasible,
        key=lambda item: (
            item[1].net_rmr,
            item[1].pd,
            item[1].miou,
            float("inf") if item[0] is None else item[0],
        ),
    )
    selection = ThresholdSelection(threshold, metrics, True)
    frozen = FrozenCUREThresholdProtocol(
        protocol=protocol,
        base_state_fingerprint=base_state_fingerprint,
        decoder_fingerprint=decoder_fingerprint,
        final_threshold=threshold,
        budget=budget,
        validation_metrics=metrics,
        base_at_budget=frozen_base,
        _seal=_FROZEN_CURE_SEAL,
    )
    return CUREThresholdSelection(selection, frozen)


__all__ = [
    "CURECalibrationSample",
    "CURETestEvaluationLedger",
    "CURETestEvaluationResult",
    "CUREThresholdSelection",
    "FrozenCUREBaseThresholdProtocol",
    "FrozenCUREThresholdProtocol",
    "evaluate_cure_threshold",
    "select_cure_threshold",
]
