"""Sealed post-Formal800 structural-retention evaluation for PAET-BFA v21.

This is deliberately not a general ``model + cache`` evaluator.  A caller can
only replay the frozen bounded-400 structural policy with the final model
contained in an authenticated Formal800 result and the exact bounded view
derived from that result's full ``D_R`` scalar cache.  The generic population
diagnostic remains a diagnostic; no performance claim is evaluated here.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch

from ..cache.schema import stable_fingerprint
from ..coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
)
from ..frozen_base import module_state_fingerprint
from .coverage_state_bounded_protocol import CoverageStateBoundedPopulation
from .coverage_state_paet_decision import (
    COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
    CoverageStatePAETBoundedDecision,
    decide_coverage_state_paet_bounded,
)
from .coverage_state_paet_formal_training import (
    COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT,
    COVERAGE_STATE_PAET_FORMAL_RUN_ID,
    CoverageStatePAETFormal800RunResult,
)
from .coverage_state_training import coverage_state_model_fingerprint
from .coverage_state_zero_level_evaluation import (
    COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
    CoverageStateZeroLevelEvaluationConfig,
    CoverageStateZeroLevelEvaluationResult,
    evaluate_coverage_state_zero_level_checkpoint,
)


COVERAGE_STATE_PAET_FORMAL_STRUCTURAL_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal800-structural-retention-v2"
)
COVERAGE_STATE_PAET_FORMAL_STRUCTURAL_PERFORMANCE_STATUS = "NOT_EVALUATED"

# These are coordinates in the completed v21 bounded-400 source receipt.  Do
# not infer them from a caller-provided population: doing so would let a
# different bounded selection become a post-Formal800 receipt.
COVERAGE_STATE_PAET_V21_BOUNDED_CACHE_FINGERPRINT = (
    "c1627d7e838ff57e27f4753e689bd4075d2b8a8f4d2ca00754c206092aaf66d8"
)
COVERAGE_STATE_PAET_V21_BOUNDED_POPULATION_FINGERPRINT = (
    "1a53467d57bea595afcc1edd3330708d1dda39e0e2d606325e552e8993e7841c"
)
COVERAGE_STATE_PAET_V21_SOURCE_RECEIPT_FINGERPRINT = (
    "9689ac7dc4cd95bd0e9bcf79e12e83bc1c8606a96e99ca27945dc07baf4fc74d"
)
COVERAGE_STATE_PAET_V21_DR_GATE_EVIDENCE_FINGERPRINT = (
    "156292da062f7dc45829dcbcf358cfe196a2d6e605c0d8628a02f07f574caeeb"
)


@dataclass(frozen=True)
class _FormalStructuralRetentionSeal:
    """Private provenance issued only after the public evaluator verifies it."""

    issuer: object
    formal_result: CoverageStatePAETFormal800RunResult
    bounded_population: CoverageStateBoundedPopulation
    final_model: CURELitePhaseAlignedEvidenceTransportLevelSet


_FORMAL_STRUCTURAL_SEAL_ISSUER = object()


@dataclass(frozen=True)
class CoverageStatePAETFormalStructuralRetention:
    """One sealed fixed-policy replay of one completed Formal800 result.

    Instances are issued by the private factory below.  In particular, the
    public dataclass constructor cannot manufacture a receipt by supplying
    matching-looking fingerprints.
    """

    run_id: str
    formal_result_fingerprint: str
    formal_authorization_fingerprint: str
    final_model_fingerprint: str
    bounded_cache_fingerprint: str
    bounded_population_fingerprint: str
    source_receipt_fingerprint: str
    diagnostic: CoverageStateZeroLevelEvaluationResult
    frozen_policy_decision: CoverageStatePAETBoundedDecision
    evaluation_invocations: int
    _seal: _FormalStructuralRetentionSeal

    def __post_init__(self) -> None:
        seal = self._seal
        if (
            not isinstance(seal, _FormalStructuralRetentionSeal)
            or seal.issuer is not _FORMAL_STRUCTURAL_SEAL_ISSUER
            or self.run_id != COVERAGE_STATE_PAET_FORMAL_RUN_ID
            or self.evaluation_invocations != 1
            or not isinstance(
                self.diagnostic, CoverageStateZeroLevelEvaluationResult
            )
            or not isinstance(
                self.frozen_policy_decision,
                CoverageStatePAETBoundedDecision,
            )
        ):
            raise PermissionError("formal structural receipt was not sealed")
        formal_result = seal.formal_result
        population = seal.bounded_population
        if (
            self.formal_result_fingerprint != formal_result.result_fingerprint
            or self.formal_authorization_fingerprint
            != formal_result.authorization.authorization_fingerprint
            or self.final_model_fingerprint
            != module_state_fingerprint(seal.final_model)
            or self.bounded_cache_fingerprint
            != population.cache.cache_fingerprint
            or self.bounded_population_fingerprint
            != population.population_fingerprint
            or self.source_receipt_fingerprint
            != formal_result.authorization.real_inputs.source_binding.binding_fingerprint
            or self.diagnostic.cache_fingerprint
            != self.bounded_cache_fingerprint
            or self.diagnostic.checkpoint_fingerprint
            != self.final_model_fingerprint
            or self.frozen_policy_decision.run_id
            != COVERAGE_STATE_PAET_BOUNDED_RUN_ID
            or self.frozen_policy_decision.diagnostic is not self.diagnostic
        ):
            raise ValueError("formal structural receipt bindings changed")

    @property
    def bounded400_structural_advancement_passed(self) -> bool:
        return self._seal.formal_result.authorization.structural_advancement_passed

    @property
    def post_formal_structural_retention_passed(self) -> bool:
        return self.frozen_policy_decision.bounded_gate_passed

    @property
    def generic_population_gate_passed(self) -> bool:
        return self.diagnostic.bounded_gate_passed

    @property
    def performance_status(self) -> str:
        return COVERAGE_STATE_PAET_FORMAL_STRUCTURAL_PERFORMANCE_STATUS

    def verify_unchanged(self) -> None:
        """Revalidate the complete in-memory provenance chain.

        The receipt intentionally retains object identity links to the
        authenticated Formal800 result and bounded population.  Rechecking
        them here prevents a once-valid receipt from remaining usable after
        its model, cache, authorization, or population has changed.
        """

        seal = self._seal
        if (
            not isinstance(seal, _FormalStructuralRetentionSeal)
            or seal.issuer is not _FORMAL_STRUCTURAL_SEAL_ISSUER
        ):
            raise PermissionError("formal structural receipt was not sealed")
        formal_result = seal.formal_result
        population = seal.bounded_population
        formal_result.verify_unchanged()
        population.verify_unchanged()
        if (
            seal.final_model is not formal_result.final_model
            or population.source_cache
            is not formal_result.authorization.real_inputs.scalar_cache
            or self.formal_result_fingerprint
            != formal_result.result_fingerprint
            or self.formal_authorization_fingerprint
            != formal_result.authorization.authorization_fingerprint
            or self.final_model_fingerprint
            != module_state_fingerprint(seal.final_model)
            or self.bounded_cache_fingerprint
            != population.cache.cache_fingerprint
            or self.bounded_population_fingerprint
            != population.population_fingerprint
            or self.source_receipt_fingerprint
            != formal_result.authorization.real_inputs.source_binding.binding_fingerprint
            or self.diagnostic.cache_fingerprint
            != self.bounded_cache_fingerprint
            or self.diagnostic.checkpoint_fingerprint
            != self.final_model_fingerprint
            or self.frozen_policy_decision.diagnostic is not self.diagnostic
        ):
            raise RuntimeError("formal structural receipt changed")

    def canonical_payload(self) -> dict[str, object]:
        self.verify_unchanged()
        return {
            "schema_version": COVERAGE_STATE_PAET_FORMAL_STRUCTURAL_SCHEMA,
            "run_id": self.run_id,
            "runtime_splits": ["D_R"],
            "formal_result_fingerprint": self.formal_result_fingerprint,
            "formal_authorization_fingerprint": (
                self.formal_authorization_fingerprint
            ),
            "final_model_fingerprint": self.final_model_fingerprint,
            "bounded_cache_fingerprint": self.bounded_cache_fingerprint,
            "bounded_population_fingerprint": (
                self.bounded_population_fingerprint
            ),
            "source_receipt_fingerprint": self.source_receipt_fingerprint,
            "input_representation": COVERAGE_STATE_PHASE_INPUT_REPRESENTATION,
            "field_threshold_hex": 0.0.hex(),
            "threshold_search_performed": False,
            "diagnostic": self.diagnostic.canonical_payload(),
            "diagnostic_result_fingerprint": self.diagnostic.result_fingerprint,
            "frozen_structural_policy": (
                self.frozen_policy_decision.canonical_payload()
            ),
            "frozen_structural_policy_fingerprint": (
                self.frozen_policy_decision.decision_fingerprint
            ),
            "policy_origin_run_id": COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
            "policy_reused_without_change": True,
            "bounded400_structural_advancement_passed": (
                self.bounded400_structural_advancement_passed
            ),
            "post_formal_structural_retention_passed": (
                self.post_formal_structural_retention_passed
            ),
            "generic_population_gate_passed": (
                self.generic_population_gate_passed
            ),
            "performance_status": self.performance_status,
            "performance_gate_passed": None,
            "evaluation_invocations": self.evaluation_invocations,
            "training_performed_by_this_layer": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
        }

    @property
    def result_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def _verify_post_formal_structural_inputs(
    formal_result: CoverageStatePAETFormal800RunResult,
    bounded_population: CoverageStateBoundedPopulation,
) -> CURELitePhaseAlignedEvidenceTransportLevelSet:
    """Verify the non-substitutable Formal800/model/population chain."""

    if type(formal_result) is not CoverageStatePAETFormal800RunResult:
        raise TypeError(
            "formal_result must be the exact CoverageStatePAETFormal800RunResult"
        )
    if type(bounded_population) is not CoverageStateBoundedPopulation:
        raise TypeError(
            "bounded_population must be the exact CoverageStateBoundedPopulation"
        )
    formal_result.verify_unchanged()
    bounded_population.verify_unchanged()
    authorization = formal_result.authorization
    final_model = formal_result.final_model
    row = formal_result.training.results[0]
    authorization.verify_model_config(final_model.config)
    if (
        not formal_result.training_complete
        or type(final_model) is not CURELitePhaseAlignedEvidenceTransportLevelSet
        or coverage_state_model_fingerprint(final_model)
        != row.final_model_fingerprint
        or bounded_population.source_cache
        is not authorization.real_inputs.scalar_cache
        or bounded_population.source_cache_fingerprint
        != COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        or bounded_population.cache.cache_fingerprint
        != COVERAGE_STATE_PAET_V21_BOUNDED_CACHE_FINGERPRINT
        or bounded_population.bounded_cache_fingerprint
        != COVERAGE_STATE_PAET_V21_BOUNDED_CACHE_FINGERPRINT
        or bounded_population.population_fingerprint
        != COVERAGE_STATE_PAET_V21_BOUNDED_POPULATION_FINGERPRINT
        or bounded_population.cache.raw_catalog.split != "D_R"
        or authorization.real_inputs.source_binding.binding_fingerprint
        != COVERAGE_STATE_PAET_V21_SOURCE_RECEIPT_FINGERPRINT
        or authorization.bounded_artifact_seal.payload.get(
            "source_binding_fingerprint"
        )
        != COVERAGE_STATE_PAET_V21_SOURCE_RECEIPT_FINGERPRINT
        or authorization.bounded_artifact_seal.payload.get(
            "D_R_gate_evidence_fingerprint"
        )
        != COVERAGE_STATE_PAET_V21_DR_GATE_EVIDENCE_FINGERPRINT
    ):
        raise PermissionError(
            "post-Formal800 structural replay rejects substituted coordinates"
        )
    return final_model


def _issue_formal_structural_retention(
    *,
    formal_result: CoverageStatePAETFormal800RunResult,
    bounded_population: CoverageStateBoundedPopulation,
    final_model: CURELitePhaseAlignedEvidenceTransportLevelSet,
    diagnostic: CoverageStateZeroLevelEvaluationResult,
    decision: CoverageStatePAETBoundedDecision,
) -> CoverageStatePAETFormalStructuralRetention:
    """Private factory: the only place allowed to issue the output seal."""

    return CoverageStatePAETFormalStructuralRetention(
        run_id=COVERAGE_STATE_PAET_FORMAL_RUN_ID,
        formal_result_fingerprint=formal_result.result_fingerprint,
        formal_authorization_fingerprint=(
            formal_result.authorization.authorization_fingerprint
        ),
        final_model_fingerprint=module_state_fingerprint(final_model),
        bounded_cache_fingerprint=bounded_population.cache.cache_fingerprint,
        bounded_population_fingerprint=(
            bounded_population.population_fingerprint
        ),
        source_receipt_fingerprint=(
            formal_result.authorization.real_inputs.source_binding.binding_fingerprint
        ),
        diagnostic=diagnostic,
        frozen_policy_decision=decision,
        evaluation_invocations=1,
        _seal=_FormalStructuralRetentionSeal(
            issuer=_FORMAL_STRUCTURAL_SEAL_ISSUER,
            formal_result=formal_result,
            bounded_population=bounded_population,
            final_model=final_model,
        ),
    )


def evaluate_coverage_state_paet_formal_structural_retention(
    formal_result: CoverageStatePAETFormal800RunResult,
    bounded_population: CoverageStateBoundedPopulation,
    *,
    device: torch.device | str,
) -> CoverageStatePAETFormalStructuralRetention:
    """Replay v21's bounded policy only for an authenticated Formal800 result."""

    final_model = _verify_post_formal_structural_inputs(
        formal_result,
        bounded_population,
    )
    final_model_fingerprint = module_state_fingerprint(final_model)
    was_training = final_model.training
    try:
        final_model.eval()
        diagnostic = evaluate_coverage_state_zero_level_checkpoint(
            final_model,
            bounded_population.cache,
            device=device,
            config=CoverageStateZeroLevelEvaluationConfig(
                input_representation=COVERAGE_STATE_PHASE_INPUT_REPRESENTATION
            ),
        )
    finally:
        final_model.train(was_training)
    decision = decide_coverage_state_paet_bounded(
        diagnostic,
        run_id=COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
    )
    bounded_population.verify_unchanged()
    formal_result.verify_unchanged()
    if module_state_fingerprint(final_model) != final_model_fingerprint:
        raise RuntimeError("Formal800 final model changed during structural replay")
    return _issue_formal_structural_retention(
        formal_result=formal_result,
        bounded_population=bounded_population,
        final_model=final_model,
        diagnostic=diagnostic,
        decision=decision,
    )


__all__ = [
    "COVERAGE_STATE_PAET_FORMAL_STRUCTURAL_PERFORMANCE_STATUS",
    "COVERAGE_STATE_PAET_FORMAL_STRUCTURAL_SCHEMA",
    "COVERAGE_STATE_PAET_V21_BOUNDED_CACHE_FINGERPRINT",
    "COVERAGE_STATE_PAET_V21_BOUNDED_POPULATION_FINGERPRINT",
    "COVERAGE_STATE_PAET_V21_DR_GATE_EVIDENCE_FINGERPRINT",
    "COVERAGE_STATE_PAET_V21_SOURCE_RECEIPT_FINGERPRINT",
    "CoverageStatePAETFormalStructuralRetention",
    "evaluate_coverage_state_paet_formal_structural_retention",
]
