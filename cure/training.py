"""Minimal full-CURE state batching and optimizer step."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from bisect import bisect_right
import hashlib
import math

import torch
from torch import Tensor

from .decoder import CUREResidualDecoder
from .descriptors import dilate_mask
from .experiment import (
    CURETrainingPolicy,
    CounterfactualSamplingPolicy,
)
from .losses import CUREUncensoringLoss
from ..sampling import stable_hash
from ..instances import instances_from_binary_mask
from ..intervention import enumerate_legal_deletions
from ..matching import match_components
from ..types import InstanceMap, MatchResult
from .protocol import CUREProtocol, frozen_output_fingerprint, tensor_content_fingerprint
from .supervision import (
    build_counterfactual_residual_set_supervision,
    build_factual_residual_set_supervision,
)
from .types import CUREInterventionCatalog, ResidualSetSupervision
from .types import WeightedCounterfactualCandidate


_SAMPLING_RECEIPT_SEAL = object()
_STATE_POOL_SEAL = object()


def _digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _state_receipt_fingerprint(
    sample_id: str,
    protocol: CUREProtocol,
    feature: Tensor,
    base_probability: Tensor,
    supervision: ResidualSetSupervision,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(protocol.fingerprint.encode("ascii"))
    hasher.update(sample_id.encode("utf-8"))
    hasher.update(
        frozen_output_fingerprint(
            protocol, sample_id, feature, base_probability
        ).encode("ascii")
    )
    hasher.update(
        repr(
            (
                supervision.positive_gt_ids,
                supervision.uneditable_gt_ids,
                supervision.ignored_gt_ids,
                supervision.branch,
            )
        ).encode("utf-8")
    )
    for name in (
        "occupancy",
        "editable_mask",
        "target",
        "background_mask",
        "object_masks",
    ):
        hasher.update(
            tensor_content_fingerprint(name, getattr(supervision, name)).encode(
                "ascii"
            )
        )
    return hasher.hexdigest()


def _target_score_mean(probability: Tensor, mask: Tensor) -> float:
    values = probability.detach().to(device="cpu", dtype=torch.float64)[0, 0][
        mask.detach().to(device="cpu", dtype=torch.bool)
    ]
    if values.numel() < 1:
        raise ValueError("target-score control received an empty GT mask")
    result = float(values.mean().item())
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("target-score control must lie in [0,1]")
    return result


def _factual_quantile_boundaries(
    factual_scores: tuple[tuple[tuple[str, int], float], ...],
    strata_count: int,
) -> tuple[float, ...]:
    """Freeze up to K empirical-quantile strata from U_M only.

    Boundaries are midpoints between distinct adjacent factual scores, so tied
    U_M scores are never split by a bookkeeping key.  Every resulting stratum
    therefore has positive factual mass.
    """

    values = sorted(score for _, score in factual_scores)
    if not values:
        raise ValueError("Score-Strata requires at least one factual U_M target")
    boundaries: list[float] = []
    count = len(values)
    for stratum in range(1, strata_count):
        split = (stratum * count + strata_count - 1) // strata_count
        if split <= 0 or split >= count:
            continue
        lower = values[split - 1]
        upper = values[split]
        if lower < upper:
            boundary = (lower + upper) / 2.0
            if not boundaries or boundary > boundaries[-1]:
                boundaries.append(boundary)
    return tuple(boundaries)


def _score_strata_weights(
    *,
    boundaries: tuple[float, ...],
    factual_values: tuple[float, ...],
    legal_values: tuple[float, ...],
    weight_floor: float,
) -> tuple[tuple[float, ...], float, float]:
    """Match legal strata to factual mass while retaining legal-only strata.

    ``weight_floor`` is protocol-bound by :class:`CURETrainingPolicy`.  It is
    applied only when the exact U_M histogram gives a legal stratum zero mass;
    this preserves the shared legal support without disguising the overlap
    diagnostic.
    """

    factual_bins = [0] * (len(boundaries) + 1)
    legal_bins = [0] * (len(boundaries) + 1)
    for score in factual_values:
        factual_bins[bisect_right(boundaries, score)] += 1
    legal_bin_indices = tuple(
        bisect_right(boundaries, score) for score in legal_values
    )
    for bin_index in legal_bin_indices:
        legal_bins[bin_index] += 1
    factual_total = len(factual_values)
    if factual_total < 1:
        raise ValueError("Score-Strata requires at least one factual U_M target")
    transported_mass = sum(
        factual_bins[index]
        for index, legal_count in enumerate(legal_bins)
        if legal_count > 0
    ) / factual_total
    unmatched_mass = 1.0 - transported_mass
    if transported_mass <= 0.0:
        raise ValueError("Score-Strata has zero U_M/U_L score-stratum overlap")
    weights = tuple(
        max(
            (factual_bins[bin_index] / factual_total) / legal_bins[bin_index],
            weight_floor,
        )
        for bin_index in legal_bin_indices
    )
    return weights, transported_mass, unmatched_mass


def _sampling_weights_for_policy(
    policy: CURETrainingPolicy,
    candidates: tuple[WeightedCounterfactualCandidate, ...],
    candidate_scores: tuple[tuple[tuple[str, int, int], float], ...],
    factual_scores: tuple[tuple[tuple[str, int], float], ...],
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    float | None,
    float | None,
]:
    """Return positive weights on exactly the catalog legal support."""

    if tuple(key for key, _ in candidate_scores) != tuple(
        candidate.key for candidate in candidates
    ):
        raise ValueError("candidate-score support differs from legal support")
    scores = tuple(score for _, score in candidate_scores)
    sampler = policy.sampling_policy
    boundaries: tuple[float, ...] = ()
    transported_mass: float | None = None
    unmatched_mass: float | None = None
    if sampler is CounterfactualSamplingPolicy.ODDS:
        weights = tuple(candidate.weight for candidate in candidates)
    elif sampler is CounterfactualSamplingPolicy.UNIFORM:
        weights = tuple(1.0 for _ in candidates)
    elif sampler is CounterfactualSamplingPolicy.SCORE_HARD:
        # Dense reverse ranks keep ties equal and retain every legal candidate.
        distinct = tuple(sorted(set(scores)))
        hardness = {
            value: float(len(distinct) - index)
            for index, value in enumerate(distinct)
        }
        weights = tuple(hardness[value] for value in scores)
    elif sampler is CounterfactualSamplingPolicy.SCORE_STRATA:
        # Match U_L to the empirical U_M mass along one frozen scalar score.
        # This is deliberately not a uniform quantile sampler on U_L.
        boundaries = _factual_quantile_boundaries(
            factual_scores, policy.score_strata_count
        )
        weights, transported_mass, unmatched_mass = _score_strata_weights(
            boundaries=boundaries,
            factual_values=tuple(score for _, score in factual_scores),
            legal_values=scores,
            weight_floor=policy.control_weight_floor,
        )
    elif sampler is CounterfactualSamplingPolicy.ODDS_PLACEBO:
        odds = tuple(candidate.weight for candidate in candidates)
        permutation = tuple(
            sorted(
                range(len(candidates)),
                key=lambda index: (
                    stable_hash(
                        "cure-odds-placebo-v1",
                        policy.protocol_fingerprint,
                        policy.placebo_seed,
                        candidates[index].key,
                    ),
                    candidates[index].key,
                ),
            )
        )
        if len(permutation) > 1 and permutation == tuple(range(len(permutation))):
            permutation = (*permutation[1:], permutation[0])
        weights = tuple(odds[source_index] for source_index in permutation)
        # This exact permutation preserves sum, ESS, and the whole histogram.
        if sorted(weights) != sorted(odds):
            raise AssertionError("Odds-Placebo did not preserve the odds multiset")
    else:  # pragma: no cover - exhaustive enum guard
        raise AssertionError(f"unhandled sampler {sampler!r}")
    if len(weights) != len(candidates) or any(
        not math.isfinite(weight) or weight <= 0.0 for weight in weights
    ):
        raise ValueError("sampling policy must retain positive mass on all legal support")
    return weights, boundaries, transported_mass, unmatched_mass


def _state_universe_fingerprint(
    protocol_fingerprint: str,
    catalog_fingerprint: str,
    factual_receipts: tuple[str, ...],
    counterfactual_receipts: tuple[tuple[tuple[str, int, int], str], ...],
) -> str:
    payload = repr(
        (
            protocol_fingerprint,
            catalog_fingerprint,
            factual_receipts,
            counterfactual_receipts,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cumulative_positive_weights(weights: tuple[float, ...]) -> tuple[float, ...]:
    """Build a searchable CDF with an order-independent exact final total."""

    if not weights:
        raise ValueError("sampling weights cannot be empty")
    cumulative: list[float] = []
    running = 0.0
    for weight in weights[:-1]:
        running = math.fsum((running, weight))
        cumulative.append(running)
    total = math.fsum(weights)
    if cumulative and total <= cumulative[-1]:
        raise ValueError("positive sampling weights lost numerical support")
    cumulative.append(total)
    return tuple(cumulative)


@dataclass(frozen=True)
class CUREStateExample:
    """One factual or coverage-counterfactual residual state.

    A counterfactual example intentionally keeps the factual frozen feature
    and base probability while replacing only ``supervision.occupancy`` with a
    legal deletion state.  That is the proposed coverage intervention, not a
    claim that the tuple is a natural frozen-detector forward state.
    """

    sample_id: str
    feature: Tensor
    base_probability: Tensor
    supervision: ResidualSetSupervision
    protocol_fingerprint: str
    frozen_output_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        _digest("protocol_fingerprint", self.protocol_fingerprint)
        _digest("frozen_output_fingerprint", self.frozen_output_fingerprint)
        if not isinstance(self.feature, Tensor) or self.feature.ndim != 4 or self.feature.shape[0] != 1:
            raise ValueError("feature must have shape [1,C,h,w]")
        if (
            not isinstance(self.base_probability, Tensor)
            or self.base_probability.ndim != 4
            or self.base_probability.shape[:2] != (1, 1)
        ):
            raise ValueError("base_probability must have shape [1,1,H,W]")
        if not isinstance(self.supervision, ResidualSetSupervision):
            raise TypeError("supervision must be ResidualSetSupervision")
        if self.feature.requires_grad or self.base_probability.requires_grad:
            raise ValueError("cached frozen-base tensors must be detached")
        if self.feature.dtype != torch.float32 or self.base_probability.dtype != torch.float32:
            raise TypeError("frozen-base tensors must be float32")
        if self.feature.device != self.base_probability.device:
            raise ValueError("frozen-base tensors must share a device")
        if not torch.isfinite(self.feature).all() or not torch.isfinite(
            self.base_probability
        ).all():
            raise ValueError("frozen-base tensors must be finite")
        if tuple(self.base_probability.shape[-2:]) != tuple(self.supervision.target.shape[-2:]):
            raise ValueError("base probability and supervision grids differ")
        if torch.any((self.base_probability < 0.0) | (self.base_probability > 1.0)):
            raise ValueError("base_probability must lie in [0,1]")

    @classmethod
    def bind(
        cls,
        sample_id: str,
        feature: Tensor,
        base_probability: Tensor,
        supervision: ResidualSetSupervision,
        protocol: CUREProtocol,
    ) -> "CUREStateExample":
        if not isinstance(protocol, CUREProtocol):
            raise TypeError("protocol must be CUREProtocol")
        return cls(
            sample_id=sample_id,
            feature=feature,
            base_probability=base_probability,
            supervision=supervision,
            protocol_fingerprint=protocol.fingerprint,
            frozen_output_fingerprint=frozen_output_fingerprint(
                protocol, sample_id, feature, base_probability
            ),
        )


@dataclass(frozen=True)
class CUREStatePool:
    """Once-validated D_R states used by the per-step sampler.

    Expensive GT/matching/legal-deletion replay happens exactly once in
    :func:`build_cure_state_pool`.  A training draw then hashes only the states
    it selected, rather than rescanning the full source pool every step.
    """

    intervention_catalog: CUREInterventionCatalog
    catalog_fingerprint: str
    training_policy: CURETrainingPolicy
    policy_fingerprint: str
    state_universe_fingerprint: str
    factual_states: tuple[CUREStateExample, ...]
    counterfactual_states: tuple[
        tuple[tuple[str, int, int], CUREStateExample], ...
    ]
    factual_state_fingerprints: tuple[str, ...]
    counterfactual_state_fingerprints: tuple[
        tuple[tuple[str, int, int], str], ...
    ]
    factual_target_scores: tuple[tuple[tuple[str, int], float], ...]
    candidate_target_scores: tuple[
        tuple[tuple[str, int, int], float], ...
    ]
    score_strata_boundaries: tuple[float, ...]
    score_strata_transported_mass_fraction: float | None
    score_strata_unmatched_mass_fraction: float | None
    candidate_sampling_weights: tuple[float, ...]
    cumulative_candidate_weights: tuple[float, ...]
    total_candidate_weight: float
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        issuing = self._seal is _STATE_POOL_SEAL
        if not issuing and not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _STATE_POOL_SEAL
        ):
            raise ValueError("state pool was not issued by the canonical binder")
        if not isinstance(self.intervention_catalog, CUREInterventionCatalog):
            raise TypeError("intervention_catalog must be CUREInterventionCatalog")
        self.intervention_catalog.validate_receipt()
        _digest("catalog_fingerprint", self.catalog_fingerprint)
        _digest("policy_fingerprint", self.policy_fingerprint)
        _digest("state_universe_fingerprint", self.state_universe_fingerprint)
        if not isinstance(self.training_policy, CURETrainingPolicy):
            raise TypeError("training_policy must be CURETrainingPolicy")
        self.training_policy.validate_for_protocol(self.intervention_catalog.protocol)
        if self.policy_fingerprint != self.training_policy.fingerprint:
            raise ValueError("state pool policy fingerprint differs")
        if self.catalog_fingerprint != self.intervention_catalog.fingerprint:
            raise ValueError("state pool catalog fingerprint differs")
        expected_ids = tuple(
            sample_id
            for sample_id, _ in self.intervention_catalog.frozen_output_fingerprints
        )
        if tuple(item.sample_id for item in self.factual_states) != expected_ids:
            raise ValueError("state pool factual universe differs from D_R catalogs")
        if len(self.factual_state_fingerprints) != len(self.factual_states):
            raise ValueError("factual state receipt count differs")
        counter_keys = tuple(key for key, _ in self.counterfactual_states)
        expected_counter_keys = tuple(
            item.key for item in self.intervention_catalog.candidates
        )
        if counter_keys != expected_counter_keys:
            raise ValueError("state pool counterfactual support differs from catalog")
        if tuple(key for key, _ in self.counterfactual_state_fingerprints) != counter_keys:
            raise ValueError("counterfactual state receipts differ from pool support")
        expected_universe = _state_universe_fingerprint(
            self.protocol.fingerprint,
            self.catalog_fingerprint,
            self.factual_state_fingerprints,
            self.counterfactual_state_fingerprints,
        )
        if self.state_universe_fingerprint != expected_universe:
            raise ValueError("state-universe fingerprint differs from pool content")
        factual_score_keys = tuple(key for key, _ in self.factual_target_scores)
        expected_factual_keys = tuple(
            sorted(
                (sample_id, gt_id)
                for sample_id, gt_id, role in self.intervention_catalog.eligible_keys
                if role == "factual_miss"
            )
        )
        if factual_score_keys != expected_factual_keys:
            raise ValueError("factual target scores differ from eligible U_M")
        if tuple(key for key, _ in self.candidate_target_scores) != counter_keys:
            raise ValueError("candidate target scores differ from legal U_L support")
        for _, value in (*self.factual_target_scores, *self.candidate_target_scores):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("target scores must be finite values in [0,1]")
        (
            expected_weights,
            expected_boundaries,
            expected_transported_mass,
            expected_unmatched_mass,
        ) = _sampling_weights_for_policy(
            self.training_policy,
            self.intervention_catalog.candidates,
            self.candidate_target_scores,
            self.factual_target_scores,
        )
        if self.candidate_sampling_weights != expected_weights:
            raise ValueError("candidate weights differ from the bound sampling policy")
        if self.score_strata_boundaries != expected_boundaries:
            raise ValueError("score strata differ from frozen U_M quantiles")
        if (
            self.score_strata_transported_mass_fraction
            != expected_transported_mass
            or self.score_strata_unmatched_mass_fraction
            != expected_unmatched_mass
        ):
            raise ValueError("Score-Strata overlap diagnostics differ")
        if len(self.cumulative_candidate_weights) != len(expected_counter_keys):
            raise ValueError("cumulative candidate weights differ from support")
        if len(self.candidate_sampling_weights) != len(expected_counter_keys):
            raise ValueError("candidate sampling weights differ from support")
        if (
            not math.isfinite(self.total_candidate_weight)
            or self.total_candidate_weight <= 0.0
        ):
            raise ValueError("total candidate weight must be finite and positive")
        expected_cumulative = _cumulative_positive_weights(
            self.candidate_sampling_weights
        )
        for weight, value in zip(
            self.candidate_sampling_weights, expected_cumulative, strict=True
        ):
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError("candidate sampling weights must be positive")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("cumulative candidate weights must strictly increase")
        if self.cumulative_candidate_weights != expected_cumulative:
            raise ValueError("cumulative weights do not sum bound policy weights")
        if not math.isclose(
            self.cumulative_candidate_weights[-1],
            self.total_candidate_weight,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("cumulative and total candidate weights differ")
        protocol = self.intervention_catalog.protocol
        observed_factual = tuple(
            _state_receipt_fingerprint(
                item.sample_id,
                protocol,
                item.feature,
                item.base_probability,
                item.supervision,
            )
            for item in self.factual_states
        )
        observed_counter = tuple(
            (
                key,
                _state_receipt_fingerprint(
                    item.sample_id,
                    protocol,
                    item.feature,
                    item.base_probability,
                    item.supervision,
                ),
            )
            for key, item in self.counterfactual_states
        )
        if observed_factual != self.factual_state_fingerprints:
            raise ValueError("factual state content differs from pool receipts")
        if observed_counter != self.counterfactual_state_fingerprints:
            raise ValueError("counterfactual state content differs from pool receipts")
        if issuing:
            object.__setattr__(self, "_seal", (_STATE_POOL_SEAL, self.fingerprint))
        elif self._seal[1] != self.fingerprint:
            raise ValueError("state pool content differs from its receipt")

    @property
    def protocol(self) -> CUREProtocol:
        return self.intervention_catalog.protocol

    @property
    def fingerprint(self) -> str:
        payload = repr(
            (
                self.protocol.fingerprint,
                self.catalog_fingerprint,
                self.policy_fingerprint,
                self.state_universe_fingerprint,
                self.factual_state_fingerprints,
                self.counterfactual_state_fingerprints,
                self.factual_target_scores,
                self.candidate_target_scores,
                self.score_strata_boundaries,
                self.score_strata_transported_mass_fraction,
                self.score_strata_unmatched_mass_fraction,
                self.candidate_sampling_weights,
                self.cumulative_candidate_weights,
                self.total_candidate_weight,
            )
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_receipt(self) -> None:
        if not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _STATE_POOL_SEAL
            and self._seal[1] == self.fingerprint
        ):
            raise ValueError("state pool content differs from its receipt")
        self.training_policy.validate_for_protocol(self.protocol)

    @property
    def receipt_fingerprint(self) -> str:
        self.validate_receipt()
        return self._seal[1]

    @property
    def candidate_effective_sample_size(self) -> float:
        total = math.fsum(self.candidate_sampling_weights)
        squared = math.fsum(
            value * value for value in self.candidate_sampling_weights
        )
        return total * total / squared


@dataclass(frozen=True)
class CURESamplingReceipt:
    """Module-issued proof that a batch came from the canonical global draw."""

    protocol_fingerprint: str
    catalog_fingerprint: str
    state_pool_fingerprint: str
    state_universe_fingerprint: str
    policy_fingerprint: str
    policy_schema_version: str
    sampler_implementation: str
    sampling_policy: str
    target_policy: str
    background_policy: str
    schedule_name: str
    control_weight_floor: float
    score_strata_transported_mass_fraction: float | None
    score_strata_unmatched_mass_fraction: float | None
    epoch: int
    step: int
    global_seed: int
    factual_count: int
    counterfactual_count: int
    selected_factual_sample_ids: tuple[str, ...]
    selected_candidate_keys: tuple[tuple[str, int, int], ...]
    selected_state_fingerprints: tuple[str, ...]
    _seal: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        issuing = self._seal is _SAMPLING_RECEIPT_SEAL
        if not issuing and not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _SAMPLING_RECEIPT_SEAL
        ):
            raise ValueError("sampling receipt was not issued by the canonical draw")
        _digest("protocol_fingerprint", self.protocol_fingerprint)
        _digest("catalog_fingerprint", self.catalog_fingerprint)
        _digest("state_pool_fingerprint", self.state_pool_fingerprint)
        _digest("state_universe_fingerprint", self.state_universe_fingerprint)
        _digest("policy_fingerprint", self.policy_fingerprint)
        constants = (
            ("policy_schema_version", self.policy_schema_version, "cure-training-policy-v1"),
            (
                "sampler_implementation",
                self.sampler_implementation,
                "global-with-replacement-v1",
            ),
            ("schedule_name", self.schedule_name, "fixed-per-step-v1"),
        )
        for name, value, expected in constants:
            if value != expected:
                raise ValueError(f"unsupported receipt {name} {value!r}")
        try:
            sampler = CounterfactualSamplingPolicy(self.sampling_policy)
        except (TypeError, ValueError) as error:
            raise ValueError("receipt has an invalid sampling_policy") from error
        if self.target_policy not in {
            "selected-deleted-target-only",
            "all-uncovered-targets",
        }:
            raise ValueError("receipt has an invalid target_policy")
        if self.background_policy not in {
            "empty",
            "counterfactual-background-bce",
        }:
            raise ValueError("receipt has an invalid background_policy")
        if (
            not math.isfinite(self.control_weight_floor)
            or not 0.0 < self.control_weight_floor <= 1.0
        ):
            raise ValueError("receipt control_weight_floor must lie in (0,1]")
        diagnostics = (
            self.score_strata_transported_mass_fraction,
            self.score_strata_unmatched_mass_fraction,
        )
        if sampler is CounterfactualSamplingPolicy.SCORE_STRATA:
            if any(
                value is None
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
                for value in diagnostics
            ):
                raise ValueError("Score-Strata receipt requires overlap diagnostics")
            if self.score_strata_transported_mass_fraction <= 0.0:
                raise ValueError("Score-Strata transported mass must be positive")
            if not math.isclose(
                sum(diagnostics), 1.0, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError("Score-Strata overlap masses must sum to one")
        elif diagnostics != (None, None):
            raise ValueError("non-Score-Strata receipt may not carry strata diagnostics")
        for name, value in (
            ("epoch", self.epoch),
            ("step", self.step),
            ("factual_count", self.factual_count),
            ("counterfactual_count", self.counterfactual_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if isinstance(self.global_seed, bool) or not isinstance(self.global_seed, int):
            raise TypeError("global_seed must be an integer")
        if len(self.selected_factual_sample_ids) != self.factual_count:
            raise ValueError("factual draw receipt count differs")
        if len(self.selected_candidate_keys) != self.counterfactual_count:
            raise ValueError("counterfactual draw receipt count differs")
        if len(self.selected_state_fingerprints) != (
            self.factual_count + self.counterfactual_count
        ):
            raise ValueError("state receipt count differs")
        for value in self.selected_state_fingerprints:
            _digest("selected_state_fingerprint", value)
        if issuing:
            object.__setattr__(
                self, "_seal", (_SAMPLING_RECEIPT_SEAL, self.fingerprint)
            )
        elif self._seal[1] != self.fingerprint:
            raise ValueError("sampling receipt content was modified after issuance")

    @property
    def fingerprint(self) -> str:
        payload = repr(
            (
                self.protocol_fingerprint,
                self.catalog_fingerprint,
                self.state_pool_fingerprint,
                self.state_universe_fingerprint,
                self.policy_fingerprint,
                self.policy_schema_version,
                self.sampler_implementation,
                self.sampling_policy,
                self.target_policy,
                self.background_policy,
                self.schedule_name,
                self.control_weight_floor,
                self.score_strata_transported_mass_fraction,
                self.score_strata_unmatched_mass_fraction,
                self.epoch,
                self.step,
                self.global_seed,
                self.factual_count,
                self.counterfactual_count,
                self.selected_factual_sample_ids,
                self.selected_candidate_keys,
                self.selected_state_fingerprints,
            )
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _is_canonical_sampling_receipt(value: object) -> bool:
    return (
        isinstance(value, CURESamplingReceipt)
        and isinstance(value._seal, tuple)
        and len(value._seal) == 2
        and value._seal[0] is _SAMPLING_RECEIPT_SEAL
        and value._seal[1] == value.fingerprint
    )


@dataclass(frozen=True)
class CUREBatch:
    sample_ids: tuple[str, ...]
    feature: Tensor
    base_probability: Tensor
    occupancy: Tensor
    supervisions: tuple[ResidualSetSupervision, ...]
    protocol: CUREProtocol
    training_policy: CURETrainingPolicy
    sampling_receipt: CURESamplingReceipt

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, Tensor)
            for value in (self.feature, self.base_probability, self.occupancy)
        ):
            raise TypeError("batch feature, probability, and occupancy must be tensors")
        if (
            not isinstance(self.sample_ids, tuple)
            or len(self.sample_ids) != self.feature.shape[0]
            or any(not isinstance(value, str) or not value for value in self.sample_ids)
        ):
            raise ValueError("batch requires one non-empty sample_id per state")
        if not isinstance(self.protocol, CUREProtocol):
            raise TypeError("batch protocol must be CUREProtocol")
        if not isinstance(self.training_policy, CURETrainingPolicy):
            raise TypeError("batch training_policy must be CURETrainingPolicy")
        self.training_policy.validate_for_protocol(self.protocol)
        if not isinstance(self.sampling_receipt, CURESamplingReceipt):
            raise TypeError("batch requires a canonical sampling receipt")
        if not _is_canonical_sampling_receipt(self.sampling_receipt):
            raise ValueError("batch sampling receipt is not canonical")
        if self.sampling_receipt.protocol_fingerprint != self.protocol.fingerprint:
            raise ValueError("batch receipt and protocol differ")
        if (
            self.sampling_receipt.policy_fingerprint
            != self.training_policy.fingerprint
        ):
            raise ValueError("batch receipt and training policy differ")
        expected_policy_fields = (
            (
                self.sampling_receipt.policy_schema_version,
                self.training_policy.schema_version,
            ),
            (
                self.sampling_receipt.sampler_implementation,
                self.training_policy.sampler_implementation,
            ),
            (
                self.sampling_receipt.sampling_policy,
                self.training_policy.sampling_policy.value,
            ),
            (
                self.sampling_receipt.target_policy,
                self.training_policy.target_policy.value,
            ),
            (
                self.sampling_receipt.background_policy,
                self.training_policy.background_policy.value,
            ),
            (
                self.sampling_receipt.schedule_name,
                self.training_policy.schedule_name,
            ),
            (
                self.sampling_receipt.control_weight_floor,
                self.training_policy.control_weight_floor,
            ),
            (
                self.sampling_receipt.factual_count,
                self.training_policy.factual_count,
            ),
            (
                self.sampling_receipt.counterfactual_count,
                self.training_policy.counterfactual_count,
            ),
            (
                self.sampling_receipt.global_seed,
                self.training_policy.global_seed,
            ),
        )
        if any(observed != expected for observed, expected in expected_policy_fields):
            raise ValueError("batch receipt fields differ from the training policy")
        if self.feature.ndim != 4 or self.feature.shape[0] < 1:
            raise ValueError("feature must have shape [B,C,h,w]")
        if self.base_probability.ndim != 4 or self.base_probability.shape[1] != 1:
            raise ValueError("base_probability must have shape [B,1,H,W]")
        if self.occupancy.shape != self.base_probability.shape or self.occupancy.dtype != torch.bool:
            raise ValueError("occupancy must be bool with the probability shape")
        if self.feature.shape[0] != self.base_probability.shape[0]:
            raise ValueError("batch tensor sizes differ")
        if self.feature.shape[1] != self.protocol.residual_config.feature_channels:
            raise ValueError("batch feature channels differ from the protocol")
        if self.feature.dtype != torch.float32 or self.base_probability.dtype != torch.float32:
            raise TypeError("batch feature and probability must be float32")
        if not (
            self.feature.device == self.base_probability.device == self.occupancy.device
        ):
            raise ValueError("batch tensors must share a device")
        if not torch.isfinite(self.feature).all() or not torch.isfinite(
            self.base_probability
        ).all():
            raise ValueError("batch tensors must be finite")
        if torch.any((self.base_probability < 0.0) | (self.base_probability > 1.0)):
            raise ValueError("base_probability must lie in [0,1]")
        if not isinstance(self.supervisions, tuple) or len(self.supervisions) != self.feature.shape[0]:
            raise ValueError("one supervision is required per batch item")
        if self.feature.shape[0] != (
            self.sampling_receipt.factual_count
            + self.sampling_receipt.counterfactual_count
        ):
            raise ValueError("batch size differs from the sampling receipt")
        for index, supervision in enumerate(self.supervisions):
            if not isinstance(supervision, ResidualSetSupervision):
                raise TypeError("supervisions must contain ResidualSetSupervision")
            if supervision.target.device != self.feature.device:
                raise ValueError("batch tensors and supervisions must share a device")
            if not torch.equal(self.occupancy[index], supervision.occupancy):
                raise ValueError("batch occupancy differs from supervision state")
            observed_receipt = _state_receipt_fingerprint(
                self.sample_ids[index],
                self.protocol,
                self.feature[index : index + 1],
                self.base_probability[index : index + 1],
                supervision,
            )
            if (
                observed_receipt
                != self.sampling_receipt.selected_state_fingerprints[index]
            ):
                raise ValueError("batch state differs from its sampling receipt")

    def validate_integrity(self) -> None:
        """Recheck mutable tensor content immediately before optimization."""

        self.__post_init__()


def _move_supervision(
    supervision: ResidualSetSupervision,
    device: torch.device | str,
) -> ResidualSetSupervision:
    return ResidualSetSupervision(
        occupancy=supervision.occupancy.to(device),
        editable_mask=supervision.editable_mask.to(device),
        target=supervision.target.to(device),
        background_mask=supervision.background_mask.to(device),
        object_masks=supervision.object_masks.to(device),
        positive_gt_ids=supervision.positive_gt_ids,
        uneditable_gt_ids=supervision.uneditable_gt_ids,
        ignored_gt_ids=supervision.ignored_gt_ids,
        branch=supervision.branch,
    )


def collate_cure_states(
    examples: tuple[CUREStateExample, ...],
    *,
    protocol: CUREProtocol,
    training_policy: CURETrainingPolicy,
    sampling_receipt: CURESamplingReceipt,
    device: torch.device | str,
) -> CUREBatch:
    if not isinstance(examples, tuple) or not examples or any(
        not isinstance(item, CUREStateExample) for item in examples
    ):
        raise TypeError("examples must be a non-empty CUREStateExample tuple")
    if not isinstance(protocol, CUREProtocol):
        raise TypeError("protocol must be CUREProtocol")
    if not isinstance(training_policy, CURETrainingPolicy):
        raise TypeError("training_policy must be CURETrainingPolicy")
    training_policy.validate_for_protocol(protocol)
    if not isinstance(sampling_receipt, CURESamplingReceipt):
        raise TypeError("sampling_receipt must be CURESamplingReceipt")
    if not _is_canonical_sampling_receipt(sampling_receipt):
        raise ValueError("sampling receipt was not issued by the canonical draw")
    for item in examples:
        protocol.assert_sample(item.sample_id, split="D_R")
        if item.protocol_fingerprint != protocol.fingerprint:
            raise ValueError("state and batch protocols differ")
        expected_fingerprint = frozen_output_fingerprint(
            protocol, item.sample_id, item.feature, item.base_probability
        )
        if item.frozen_output_fingerprint != expected_fingerprint:
            raise ValueError("state frozen-output receipt differs from its tensors")
    observed_receipts = tuple(
        _state_receipt_fingerprint(
            item.sample_id,
            protocol,
            item.feature,
            item.base_probability,
            item.supervision,
        )
        for item in examples
    )
    if observed_receipts != sampling_receipt.selected_state_fingerprints:
        raise ValueError("collated states differ from the canonical draw receipt")
    feature_shapes = {tuple(item.feature.shape[1:]) for item in examples}
    probability_shapes = {tuple(item.base_probability.shape[1:]) for item in examples}
    if len(feature_shapes) != 1 or len(probability_shapes) != 1:
        raise ValueError("CURE states must have compatible feature and output grids")
    supervisions = tuple(_move_supervision(item.supervision, device) for item in examples)
    return CUREBatch(
        sample_ids=tuple(item.sample_id for item in examples),
        feature=torch.cat([item.feature for item in examples], dim=0).to(device),
        base_probability=torch.cat(
            [item.base_probability for item in examples], dim=0
        ).to(device),
        occupancy=torch.stack(
            [item.supervision.occupancy for item in examples], dim=0
        ).to(device),
        supervisions=supervisions,
        protocol=protocol,
        training_policy=training_policy,
        sampling_receipt=sampling_receipt,
    )


def _supervision_equal(
    observed: ResidualSetSupervision,
    expected: ResidualSetSupervision,
) -> bool:
    tensor_names = (
        "occupancy",
        "editable_mask",
        "target",
        "background_mask",
        "object_masks",
    )
    tensors_equal = all(
        torch.equal(
            getattr(observed, name).detach().to(device="cpu"),
            getattr(expected, name).detach().to(device="cpu"),
        )
        for name in tensor_names
    )
    metadata_equal = all(
        getattr(observed, name) == getattr(expected, name)
        for name in (
            "positive_gt_ids",
            "uneditable_gt_ids",
            "ignored_gt_ids",
            "branch",
        )
    )
    return tensors_equal and metadata_equal


def _validate_frozen_state(
    state: CUREStateExample,
    catalog: CUREInterventionCatalog,
    gt: InstanceMap,
) -> tuple[Tensor, MatchResult]:
    protocol = catalog.protocol
    protocol.assert_sample(state.sample_id, split="D_R")
    if state.protocol_fingerprint != protocol.fingerprint:
        raise ValueError("training state uses a different CURE protocol")
    expected_outputs = dict(catalog.frozen_output_fingerprints)
    try:
        expected_output = expected_outputs[state.sample_id]
    except KeyError as error:
        raise ValueError("training state has no eligible-catalog source receipt") from error
    actual_output = frozen_output_fingerprint(
        protocol, state.sample_id, state.feature, state.base_probability
    )
    if (
        state.frozen_output_fingerprint != actual_output
        or actual_output != expected_output
    ):
        raise ValueError("training state feature/probability differs from its catalog")
    expected_gt = dict(catalog.gt_fingerprints).get(state.sample_id)
    actual_gt = tensor_content_fingerprint("gt_labels", gt.labels)
    if expected_gt is None or actual_gt != expected_gt:
        raise ValueError("training GT differs from its eligible catalog")
    base_occupancy = (
        state.base_probability.detach().to(device="cpu")[0, 0]
        >= protocol.residual_config.occupancy_threshold
    )
    prediction = instances_from_binary_mask(
        base_occupancy, connectivity=8, min_area=1
    )
    before = match_components(prediction, gt, protocol.match_config)
    editable = ~dilate_mask(
        state.supervision.occupancy.detach().to(device="cpu")[0],
        protocol.residual_config.suppression_radius,
    )
    if not torch.equal(
        state.supervision.editable_mask.detach().to(device="cpu")[0], editable
    ):
        raise ValueError("training state editable mask differs from the protocol")
    return base_occupancy, before


def _validate_factual_state(
    state: CUREStateExample,
    catalog: CUREInterventionCatalog,
    gt: InstanceMap,
) -> tuple[Tensor, MatchResult]:
    if state.supervision.branch != "factual":
        raise ValueError("factual pool contains a non-factual state")
    occupancy, before = _validate_frozen_state(state, catalog, gt)
    if not torch.equal(
        state.supervision.occupancy.detach().to(device="cpu")[0], occupancy
    ):
        raise ValueError("factual occupancy must equal base_probability >= tau_o")
    expected = build_factual_residual_set_supervision(
        occupancy,
        gt,
        before,
        catalog.protocol.match_config,
        suppression_radius=catalog.suppression_radius,
    )
    if not _supervision_equal(state.supervision, expected):
        raise ValueError("factual supervision differs from the canonical builder")
    return occupancy, before


def _validate_counterfactual_state(
    state: CUREStateExample,
    candidate: WeightedCounterfactualCandidate,
    catalog: CUREInterventionCatalog,
    gt: InstanceMap,
    training_policy: CURETrainingPolicy,
) -> None:
    if state.supervision.branch != "counterfactual":
        raise ValueError("weighted candidate maps to a non-counterfactual state")
    if state.sample_id != candidate.sample_id:
        raise ValueError("weighted candidate maps to a different sample_id")
    occupancy, before = _validate_frozen_state(state, catalog, gt)
    prediction = instances_from_binary_mask(occupancy, connectivity=8, min_area=1)
    canonical = {
        (item.gt_id, item.pred_id): item
        for item in enumerate_legal_deletions(
            prediction,
            gt,
            before,
            occupancy,
            match_config=catalog.protocol.match_config,
            intervention_config=catalog.protocol.intervention_config,
        )
    }.get((candidate.deletion.gt_id, candidate.deletion.pred_id))
    if canonical is None or not torch.equal(
        canonical.occupancy_after, candidate.deletion.occupancy_after
    ):
        raise ValueError("candidate is not a canonical legal deletion of this state")
    expected = build_counterfactual_residual_set_supervision(
        canonical,
        gt,
        before,
        occupancy,
        catalog.protocol.match_config,
        suppression_radius=catalog.suppression_radius,
        target_policy=training_policy.target_policy,
        background_policy=training_policy.background_policy,
    )
    if not _supervision_equal(state.supervision, expected):
        raise ValueError(
            "counterfactual supervision differs from the bound policy builder"
        )


def build_cure_state_pool(
    factual_states: tuple[CUREStateExample, ...],
    counterfactual_states: Mapping[tuple[str, int, int], CUREStateExample],
    intervention_catalog: CUREInterventionCatalog,
    gt_by_sample: Mapping[str, InstanceMap],
    *,
    training_policy: CURETrainingPolicy,
) -> CUREStatePool:
    """Replay every expensive state invariant once before optimization."""

    if not isinstance(intervention_catalog, CUREInterventionCatalog):
        raise TypeError("intervention_catalog must be CUREInterventionCatalog")
    intervention_catalog.validate_receipt()
    if not isinstance(training_policy, CURETrainingPolicy):
        raise TypeError("training_policy must be CURETrainingPolicy")
    training_policy.validate_for_protocol(intervention_catalog.protocol)
    if not isinstance(gt_by_sample, Mapping):
        raise TypeError("gt_by_sample must be a mapping")
    expected_sample_ids = tuple(
        sample_id for sample_id, _ in intervention_catalog.frozen_output_fingerprints
    )
    factual_ids = tuple(item.sample_id for item in factual_states)
    if factual_ids != expected_sample_ids:
        raise ValueError(
            "factual pool must contain one canonical sorted state per D_R sample"
        )
    if set(counterfactual_states) != {
        item.key for item in intervention_catalog.candidates
    }:
        raise ValueError(
            "counterfactual state pool must equal the catalog candidate support"
        )
    for state in factual_states:
        try:
            gt = gt_by_sample[state.sample_id]
        except KeyError as error:
            raise ValueError(f"missing GT for sample {state.sample_id!r}") from error
        if not isinstance(gt, InstanceMap):
            raise TypeError("gt_by_sample values must be InstanceMap")
        _validate_factual_state(state, intervention_catalog, gt)
    candidates_by_key = {
        item.key: item for item in intervention_catalog.candidates
    }
    for key, state in counterfactual_states.items():
        try:
            gt = gt_by_sample[state.sample_id]
        except KeyError as error:
            raise ValueError(f"missing GT for sample {state.sample_id!r}") from error
        if not isinstance(gt, InstanceMap):
            raise TypeError("gt_by_sample values must be InstanceMap")
        _validate_counterfactual_state(
            state,
            candidates_by_key[key],
            intervention_catalog,
            gt,
            training_policy,
        )

    factual_receipts = tuple(
        _state_receipt_fingerprint(
            item.sample_id,
            intervention_catalog.protocol,
            item.feature,
            item.base_probability,
            item.supervision,
        )
        for item in factual_states
    )
    ordered_counterfactual = tuple(
        (key, counterfactual_states[key])
        for key in sorted(counterfactual_states)
    )
    counterfactual_receipts = tuple(
        (
            key,
            _state_receipt_fingerprint(
                item.sample_id,
                intervention_catalog.protocol,
                item.feature,
                item.base_probability,
                item.supervision,
            ),
        )
        for key, item in ordered_counterfactual
    )
    state_universe = _state_universe_fingerprint(
        intervention_catalog.protocol.fingerprint,
        intervention_catalog.fingerprint,
        factual_receipts,
        counterfactual_receipts,
    )
    factual_eligible_keys = tuple(
        sorted(
            (sample_id, gt_id)
            for sample_id, gt_id, role in intervention_catalog.eligible_keys
            if role == "factual_miss"
        )
    )
    factual_by_id = {item.sample_id: item for item in factual_states}
    factual_target_scores = tuple(
        (
            key,
            _target_score_mean(
                factual_by_id[key[0]].base_probability,
                gt_by_sample[key[0]].by_id(key[1]).mask,
            ),
        )
        for key in factual_eligible_keys
    )
    candidate_target_scores = tuple(
        (
            candidate.key,
            _target_score_mean(
                counterfactual_states[candidate.key].base_probability,
                gt_by_sample[candidate.sample_id].by_id(
                    candidate.deletion.gt_id
                ).mask,
            ),
        )
        for candidate in intervention_catalog.candidates
    )
    (
        sampling_weights,
        strata_boundaries,
        transported_mass,
        unmatched_mass,
    ) = _sampling_weights_for_policy(
        training_policy,
        intervention_catalog.candidates,
        candidate_target_scores,
        factual_target_scores,
    )
    cumulative_weights = _cumulative_positive_weights(sampling_weights)
    running_weight = cumulative_weights[-1]
    return CUREStatePool(
        intervention_catalog=intervention_catalog,
        catalog_fingerprint=intervention_catalog.fingerprint,
        training_policy=training_policy,
        policy_fingerprint=training_policy.fingerprint,
        state_universe_fingerprint=state_universe,
        factual_states=factual_states,
        counterfactual_states=ordered_counterfactual,
        factual_state_fingerprints=factual_receipts,
        counterfactual_state_fingerprints=counterfactual_receipts,
        factual_target_scores=factual_target_scores,
        candidate_target_scores=candidate_target_scores,
        score_strata_boundaries=strata_boundaries,
        score_strata_transported_mass_fraction=transported_mass,
        score_strata_unmatched_mass_fraction=unmatched_mass,
        candidate_sampling_weights=sampling_weights,
        cumulative_candidate_weights=cumulative_weights,
        total_candidate_weight=running_weight,
        _seal=_STATE_POOL_SEAL,
    )


def draw_fixed_exposure_batch(
    state_pool: CUREStatePool,
    *,
    epoch: int,
    step: int,
    device: torch.device | str,
    factual_count: int | None = None,
    counterfactual_count: int | None = None,
    global_seed: int | None = None,
) -> CUREBatch:
    """Draw a batch in O(batch size) from one validated state pool.

    Propensity is used exactly once, by the global counterfactual sampler.  The
    loss receives no importance weight, preventing accidental squared odds.
    """

    for name, value in (("epoch", epoch), ("step", step)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if not isinstance(state_pool, CUREStatePool):
        raise TypeError("state_pool must be CUREStatePool")
    state_pool.validate_receipt()
    policy = state_pool.training_policy
    # Compatibility arguments are assertions only.  They can never override
    # the receipt-bound exposure or seed.
    for name, observed, expected in (
        ("factual_count", factual_count, policy.factual_count),
        (
            "counterfactual_count",
            counterfactual_count,
            policy.counterfactual_count,
        ),
        ("global_seed", global_seed, policy.global_seed),
    ):
        if observed is not None and observed != expected:
            raise ValueError(f"{name} cannot override the bound training policy")
    factual_count = policy.factual_count
    counterfactual_count = policy.counterfactual_count
    global_seed = policy.global_seed
    intervention_catalog = state_pool.intervention_catalog
    factual_states = state_pool.factual_states

    selected: list[CUREStateExample] = []
    selected_factual_ids: list[str] = []
    selected_candidate_keys: list[tuple[str, int, int]] = []
    for draw_index in range(factual_count):
        index = stable_hash(
            "cure-factual-draw", epoch, step, draw_index, global_seed
        ) % len(factual_states)
        state = factual_states[index]
        observed = _state_receipt_fingerprint(
            state.sample_id,
            state_pool.protocol,
            state.feature,
            state.base_probability,
            state.supervision,
        )
        if observed != state_pool.factual_state_fingerprints[index]:
            raise ValueError("selected factual state differs from its pool receipt")
        selected.append(state)
        selected_factual_ids.append(state.sample_id)
    for draw_index in range(counterfactual_count):
        random_bits = stable_hash(
            "cure-counterfactual-draw-v1", epoch, step, draw_index, global_seed
        )
        position = (
            random_bits / float(1 << 64)
        ) * state_pool.total_candidate_weight
        candidate_index = min(
            bisect_right(state_pool.cumulative_candidate_weights, position),
            len(intervention_catalog.candidates) - 1,
        )
        candidate = intervention_catalog.candidates[candidate_index]
        state_key, state = state_pool.counterfactual_states[candidate_index]
        receipt_key, expected_receipt = (
            state_pool.counterfactual_state_fingerprints[candidate_index]
        )
        if state_key != candidate.key or receipt_key != candidate.key:
            raise ValueError("state-pool candidate ordering changed after binding")
        observed = _state_receipt_fingerprint(
            state.sample_id,
            state_pool.protocol,
            state.feature,
            state.base_probability,
            state.supervision,
        )
        if observed != expected_receipt:
            raise ValueError(
                "selected counterfactual state differs from its pool receipt"
            )
        if not torch.equal(
            state.supervision.occupancy.detach().to(device="cpu")[0],
            candidate.deletion.occupancy_after,
        ):
            raise ValueError("selected candidate deletion changed after pool binding")
        selected.append(state)
        selected_candidate_keys.append(candidate.key)
    receipt = CURESamplingReceipt(
        protocol_fingerprint=intervention_catalog.protocol.fingerprint,
        catalog_fingerprint=state_pool.catalog_fingerprint,
        state_pool_fingerprint=state_pool.receipt_fingerprint,
        state_universe_fingerprint=state_pool.state_universe_fingerprint,
        policy_fingerprint=policy.fingerprint,
        policy_schema_version=policy.schema_version,
        sampler_implementation=policy.sampler_implementation,
        sampling_policy=policy.sampling_policy.value,
        target_policy=policy.target_policy.value,
        background_policy=policy.background_policy.value,
        schedule_name=policy.schedule_name,
        control_weight_floor=policy.control_weight_floor,
        score_strata_transported_mass_fraction=(
            state_pool.score_strata_transported_mass_fraction
        ),
        score_strata_unmatched_mass_fraction=(
            state_pool.score_strata_unmatched_mass_fraction
        ),
        epoch=epoch,
        step=step,
        global_seed=global_seed,
        factual_count=factual_count,
        counterfactual_count=counterfactual_count,
        selected_factual_sample_ids=tuple(selected_factual_ids),
        selected_candidate_keys=tuple(selected_candidate_keys),
        selected_state_fingerprints=tuple(
            _state_receipt_fingerprint(
                item.sample_id,
                intervention_catalog.protocol,
                item.feature,
                item.base_probability,
                item.supervision,
            )
            for item in selected
        ),
        _seal=_SAMPLING_RECEIPT_SEAL,
    )
    return collate_cure_states(
        tuple(selected),
        protocol=intervention_catalog.protocol,
        training_policy=policy,
        sampling_receipt=receipt,
        device=device,
    )


def _optimizer_parameter_ids(optimizer: torch.optim.Optimizer) -> set[int]:
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    identifiers = [id(parameter) for parameter in parameters]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("optimizer contains a parameter more than once")
    if any(not parameter.requires_grad for parameter in parameters):
        raise ValueError("optimizer contains a frozen parameter")
    return set(identifiers)


def train_cure_step(
    decoder: CUREResidualDecoder,
    criterion: CUREUncensoringLoss,
    optimizer: torch.optim.Optimizer,
    batch: CUREBatch,
) -> dict[str, Tensor]:
    """Run exactly one update of the residual decoder and no base parameters."""

    if type(decoder) is not CUREResidualDecoder:
        raise TypeError("decoder must be CUREResidualDecoder")
    if type(criterion) is not CUREUncensoringLoss:
        raise TypeError("criterion must be CUREUncensoringLoss")
    if not isinstance(batch, CUREBatch):
        raise TypeError("batch must be CUREBatch")
    batch.validate_integrity()
    if decoder.config != batch.protocol.residual_config:
        raise ValueError("decoder configuration differs from the batch protocol")
    if criterion.config != batch.protocol.loss_config:
        raise ValueError("loss configuration differs from the batch protocol")
    if not _is_canonical_sampling_receipt(batch.sampling_receipt):
        raise ValueError("batch lacks a canonical global-sampling receipt")
    factual = sum(item.branch == "factual" for item in batch.supervisions)
    counterfactual = sum(
        item.branch == "counterfactual" for item in batch.supervisions
    )
    if (
        factual != batch.sampling_receipt.factual_count
        or counterfactual != batch.sampling_receipt.counterfactual_count
    ):
        raise ValueError("batch branch exposure differs from its sampling receipt")
    expected = {id(parameter) for parameter in decoder.parameters()}
    if _optimizer_parameter_ids(optimizer) != expected:
        raise ValueError("optimizer parameters must equal decoder parameters exactly")
    decoder.train()
    optimizer.zero_grad(set_to_none=True)
    logits = decoder(batch.feature, batch.base_probability, batch.occupancy)
    losses = criterion(logits, batch.supervisions)
    losses["total"].backward()
    optimizer.step()
    return losses


__all__ = [
    "CUREBatch",
    "CUREStateExample",
    "CUREStatePool",
    "build_cure_state_pool",
    "draw_fixed_exposure_batch",
    "train_cure_step",
]
