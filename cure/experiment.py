"""Sealed experiment policies for fair full-CURE controls.

The sampling rule is an experimental factor, not an unrecorded dataloader
choice.  This module therefore binds it, the fixed branch exposure, the random
seed, and the counterfactual supervision ablations to one CURE protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import math
from numbers import Real

from .protocol import CUREProtocol


class CounterfactualSamplingPolicy(str, Enum):
    """Public counterfactual-sampling controls on one legal support."""

    ODDS = "odds"
    UNIFORM = "uniform"
    SCORE_HARD = "score-hard"
    SCORE_STRATA = "score-strata"
    ODDS_PLACEBO = "odds-placebo"


class CounterfactualTargetPolicy(str, Enum):
    """Which newly uncovered targets receive positive CF supervision."""

    SELECTED_DELETED_TARGET_ONLY = "selected-deleted-target-only"
    ALL_UNCOVERED_TARGETS = "all-uncovered-targets"


class CounterfactualBackgroundPolicy(str, Enum):
    """Whether a counterfactual state also carries host-background BCE."""

    EMPTY = "empty"
    BCE = "counterfactual-background-bce"


_TRAINING_POLICY_SEAL = object()


def _non_negative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _digest(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class CURETrainingPolicy:
    """Protocol-bound training and ablation policy.

    Only :meth:`bind` and the controlled ``with_*`` methods can issue an
    instance.  In particular, counts and seeds cannot be changed at draw time.
    The five sampling controls vary only ``sampling_policy`` when created by
    :func:`build_fair_sampling_policy_family`.
    """

    protocol_fingerprint: str
    sampling_policy: CounterfactualSamplingPolicy
    factual_count: int
    counterfactual_count: int
    global_seed: int
    target_policy: CounterfactualTargetPolicy = (
        CounterfactualTargetPolicy.SELECTED_DELETED_TARGET_ONLY
    )
    background_policy: CounterfactualBackgroundPolicy = (
        CounterfactualBackgroundPolicy.EMPTY
    )
    score_strata_count: int = 4
    control_weight_floor: float = 1e-6
    placebo_seed: int = 0
    schedule_name: str = "fixed-per-step-v1"
    sampler_implementation: str = "global-with-replacement-v1"
    score_field: str = "full-gt-target-score-mean-v1"
    schema_version: str = "cure-training-policy-v1"
    _seal: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        issuing = self._seal is _TRAINING_POLICY_SEAL
        if not issuing and not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _TRAINING_POLICY_SEAL
        ):
            raise ValueError("training policy was not issued by the canonical binder")
        _digest("protocol_fingerprint", self.protocol_fingerprint)
        if not isinstance(self.sampling_policy, CounterfactualSamplingPolicy):
            raise TypeError("sampling_policy must be CounterfactualSamplingPolicy")
        if not isinstance(self.target_policy, CounterfactualTargetPolicy):
            raise TypeError("target_policy must be CounterfactualTargetPolicy")
        if not isinstance(self.background_policy, CounterfactualBackgroundPolicy):
            raise TypeError(
                "background_policy must be CounterfactualBackgroundPolicy"
            )
        factual = _non_negative_integer("factual_count", self.factual_count)
        counterfactual = _non_negative_integer(
            "counterfactual_count", self.counterfactual_count
        )
        if factual + counterfactual < 1:
            raise ValueError("fixed exposure must draw at least one state per step")
        if isinstance(self.global_seed, bool) or not isinstance(self.global_seed, int):
            raise TypeError("global_seed must be an integer")
        if (
            isinstance(self.score_strata_count, bool)
            or not isinstance(self.score_strata_count, int)
            or self.score_strata_count < 2
        ):
            raise ValueError("score_strata_count must be an integer of at least two")
        if (
            isinstance(self.control_weight_floor, bool)
            or not isinstance(self.control_weight_floor, Real)
            or not math.isfinite(float(self.control_weight_floor))
            or not 0.0 < float(self.control_weight_floor) <= 1.0
        ):
            raise ValueError("control_weight_floor must lie in (0,1]")
        object.__setattr__(
            self, "control_weight_floor", float(self.control_weight_floor)
        )
        if isinstance(self.placebo_seed, bool) or not isinstance(
            self.placebo_seed, int
        ):
            raise TypeError("placebo_seed must be an integer")
        constants = (
            ("schedule_name", self.schedule_name, "fixed-per-step-v1"),
            (
                "sampler_implementation",
                self.sampler_implementation,
                "global-with-replacement-v1",
            ),
            ("score_field", self.score_field, "full-gt-target-score-mean-v1"),
            ("schema_version", self.schema_version, "cure-training-policy-v1"),
        )
        for name, value, expected in constants:
            if value != expected:
                raise ValueError(f"unsupported {name} {value!r}")
        if issuing:
            object.__setattr__(self, "_seal", (_TRAINING_POLICY_SEAL, self.fingerprint))
        elif self._seal[1] != self.fingerprint:
            raise ValueError("training policy content differs from its receipt")

    @classmethod
    def bind(
        cls,
        protocol: CUREProtocol,
        *,
        sampling_policy: CounterfactualSamplingPolicy = (
            CounterfactualSamplingPolicy.ODDS
        ),
        factual_count: int,
        counterfactual_count: int,
        global_seed: int,
        target_policy: CounterfactualTargetPolicy = (
            CounterfactualTargetPolicy.SELECTED_DELETED_TARGET_ONLY
        ),
        background_policy: CounterfactualBackgroundPolicy = (
            CounterfactualBackgroundPolicy.EMPTY
        ),
        score_strata_count: int = 4,
        control_weight_floor: float = 1e-6,
        placebo_seed: int = 0,
    ) -> "CURETrainingPolicy":
        if not isinstance(protocol, CUREProtocol):
            raise TypeError("protocol must be CUREProtocol")
        protocol.validate_receipt()
        return cls(
            protocol_fingerprint=protocol.fingerprint,
            sampling_policy=sampling_policy,
            factual_count=factual_count,
            counterfactual_count=counterfactual_count,
            global_seed=global_seed,
            target_policy=target_policy,
            background_policy=background_policy,
            score_strata_count=score_strata_count,
            control_weight_floor=control_weight_floor,
            placebo_seed=placebo_seed,
            _seal=_TRAINING_POLICY_SEAL,
        )

    @property
    def fingerprint(self) -> str:
        payload = repr(
            (
                self.protocol_fingerprint,
                self.sampling_policy.value,
                self.factual_count,
                self.counterfactual_count,
                self.global_seed,
                self.target_policy.value,
                self.background_policy.value,
                self.score_strata_count,
                self.control_weight_floor,
                self.placebo_seed,
                self.schedule_name,
                self.sampler_implementation,
                self.score_field,
                self.schema_version,
            )
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_receipt(self) -> None:
        if not (
            isinstance(self._seal, tuple)
            and len(self._seal) == 2
            and self._seal[0] is _TRAINING_POLICY_SEAL
            and self._seal[1] == self.fingerprint
        ):
            raise ValueError("training policy content differs from its receipt")

    def validate_for_protocol(self, protocol: CUREProtocol) -> None:
        self.validate_receipt()
        if not isinstance(protocol, CUREProtocol):
            raise TypeError("protocol must be CUREProtocol")
        protocol.validate_receipt()
        if self.protocol_fingerprint != protocol.fingerprint:
            raise ValueError("training policy is bound to a different CURE protocol")

    def with_sampling_policy(
        self, sampling_policy: CounterfactualSamplingPolicy
    ) -> "CURETrainingPolicy":
        """Vary only the sampling axis and preserve all exposure/supervision."""

        self.validate_receipt()
        return CURETrainingPolicy(
            protocol_fingerprint=self.protocol_fingerprint,
            sampling_policy=sampling_policy,
            factual_count=self.factual_count,
            counterfactual_count=self.counterfactual_count,
            global_seed=self.global_seed,
            target_policy=self.target_policy,
            background_policy=self.background_policy,
            score_strata_count=self.score_strata_count,
            control_weight_floor=self.control_weight_floor,
            placebo_seed=self.placebo_seed,
            _seal=_TRAINING_POLICY_SEAL,
        )

    def with_supervision_policy(
        self,
        *,
        target_policy: CounterfactualTargetPolicy | None = None,
        background_policy: CounterfactualBackgroundPolicy | None = None,
    ) -> "CURETrainingPolicy":
        """Vary only supervision while preserving sampler and fixed exposure."""

        self.validate_receipt()
        return CURETrainingPolicy(
            protocol_fingerprint=self.protocol_fingerprint,
            sampling_policy=self.sampling_policy,
            factual_count=self.factual_count,
            counterfactual_count=self.counterfactual_count,
            global_seed=self.global_seed,
            target_policy=(self.target_policy if target_policy is None else target_policy),
            background_policy=(
                self.background_policy
                if background_policy is None
                else background_policy
            ),
            score_strata_count=self.score_strata_count,
            control_weight_floor=self.control_weight_floor,
            placebo_seed=self.placebo_seed,
            _seal=_TRAINING_POLICY_SEAL,
        )


def build_fair_sampling_policy_family(
    reference: CURETrainingPolicy,
) -> tuple[CURETrainingPolicy, ...]:
    """Issue the five main-table controls with only the sampler varied."""

    if not isinstance(reference, CURETrainingPolicy):
        raise TypeError("reference must be CURETrainingPolicy")
    reference.validate_receipt()
    return tuple(
        reference.with_sampling_policy(policy)
        for policy in CounterfactualSamplingPolicy
    )


__all__ = [
    "CURETrainingPolicy",
    "CounterfactualBackgroundPolicy",
    "CounterfactualSamplingPolicy",
    "CounterfactualTargetPolicy",
    "build_fair_sampling_policy_family",
]
