from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.experiment.coverage_state_cmif_p0 import (
    CoverageStateCMIFP0ReplayCandidate,
    CoverageStateCMIFP0SingleRunReceipt,
    CoverageStateCMIFPopulationAudit,
    audit_coverage_state_cmif_population,
)
from tests_v15.coverage_state_training_test_helpers import (
    make_bounded_training_scalar_cache,
)
from tools import audit_coverage_state_cmif_v17 as cli


@pytest.fixture(scope="module")
def toy_cache():
    return make_bounded_training_scalar_cache()


@pytest.fixture(scope="module")
def toy_audit(toy_cache) -> CoverageStateCMIFPopulationAudit:
    return audit_coverage_state_cmif_population(toy_cache)


def _canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _shift_zero_row_observations(
    rows: tuple[tuple[str, str, int, int], ...],
    *,
    source: tuple[str, str],
    destination: tuple[str, str],
    amount: int,
) -> tuple[tuple[str, str, int, int], ...]:
    shifted = []
    for role, stratum, observations, zeros in rows:
        if (role, stratum) == source:
            observations -= amount
        if (role, stratum) == destination:
            observations += amount
        shifted.append((role, stratum, observations, zeros))
    return tuple(shifted)


def test_toy_population_role_filtering_and_diagnostic_exclusion(
    toy_cache,
    toy_audit: CoverageStateCMIFPopulationAudit,
) -> None:
    expected_clean = {
        value.record.pair_id
        for value in toy_cache.pair_records
        if value.optimizer_role == "clean_positive"
    }
    expected_component = {
        value.record.pair_id
        for value in toy_cache.pair_records
        if value.optimizer_role == "component_null"
    }
    expected_identity = {
        value.record.pair_id
        for value in toy_cache.pair_records
        if value.optimizer_role == "identity_diagnostic"
    }
    expected_diagnostic_component = {
        value.record.pair_id
        for value in toy_cache.pair_records
        if value.optimizer_role == "diagnostic_only"
        and value.record.pair_kind == "component_null"
    }
    excluded = expected_identity | expected_diagnostic_component

    assert len(toy_audit.factual_miss_record_ids) == 16
    assert len(toy_audit.factual_no_miss_record_ids) == 16
    assert set(toy_audit.clean_pair_ids) == expected_clean
    assert set(toy_audit.component_pair_ids) == expected_component
    assert len(toy_audit.clean_pair_ids) == 16
    assert len(toy_audit.component_pair_ids) == 16
    assert set(toy_audit.identity_pair_ids) == expected_identity
    assert (
        set(toy_audit.diagnostic_component_pair_ids)
        == expected_diagnostic_component
    )
    assert len(toy_audit.identity_pair_ids) == 16
    assert len(toy_audit.diagnostic_component_pair_ids) == 1
    assert excluded
    assert excluded.isdisjoint(toy_audit.clean_pair_ids)
    assert excluded.isdisjoint(toy_audit.component_pair_ids)


def test_toy_endpoint_strata_and_background_accounting(
    toy_audit: CoverageStateCMIFPopulationAudit,
) -> None:
    rows = {
        (role, stratum): count
        for role, stratum, count in toy_audit.endpoint_stratum_counts
    }

    assert toy_audit.endpoint_state_count == 96
    assert toy_audit.endpoint_domain_observation_count == 6112
    assert toy_audit.endpoint_active_observation_count == 288
    assert toy_audit.endpoint_background_observation_count == 5824
    assert (
        toy_audit.endpoint_background_scanned_count
        == toy_audit.endpoint_background_observation_count
    )
    assert toy_audit.endpoint_background_exact_key_match_count == 0
    assert (
        toy_audit.endpoint_grouped_observation_count
        == toy_audit.endpoint_active_observation_count
    )
    assert rows[("clean_minus", "target")] == 16
    assert rows[("clean_minus", "response_ring")] == 128
    assert rows[("factual_miss", "target")] == 16
    assert rows[("factual_miss", "response_ring")] == 128
    assert sum(
        count
        for (role, stratum), count in rows.items()
        if stratum == "background"
    ) == toy_audit.endpoint_background_observation_count


def test_toy_transition_clean_zero_and_exact_key_accounting(
    toy_audit: CoverageStateCMIFPopulationAudit,
) -> None:
    rows = {
        (role, stratum): count
        for role, stratum, count in toy_audit.transition_role_counts
    }

    assert rows == {
        ("clean_nonzero", "response_core"): 16,
        ("clean_nonzero", "response_ring"): 128,
        ("clean_zero", "zero_response"): 880,
        ("component_zero", "zero_response"): 1024,
    }
    assert toy_audit.transition_observation_count == 2048
    assert toy_audit.transition_exact_key_count == 2048
    assert toy_audit.transition_singleton_key_count == 2048
    assert toy_audit.transition_repeated_key_count == 0
    assert toy_audit.component_nonzero_response_pixel_count == 0


def test_toy_zero_feature_witnesses_match_each_role_and_stratum(
    toy_audit: CoverageStateCMIFPopulationAudit,
) -> None:
    endpoint_counts = {
        (role, stratum): count
        for role, stratum, count in toy_audit.endpoint_stratum_counts
        if stratum in {"target", "response_ring"}
    }
    transition_counts = {
        (role, stratum): count
        for role, stratum, count in toy_audit.transition_role_counts
        if stratum in {"response_core", "response_ring"}
    }

    assert {
        (role, stratum): observations
        for role, stratum, observations, _
        in toy_audit.endpoint_zero_feature_rows
    } == endpoint_counts
    assert {
        (role, stratum): observations
        for role, stratum, observations, _
        in toy_audit.transition_response_zero_feature_rows
    } == transition_counts
    assert all(
        zeros == 0
        for _, _, _, zeros in toy_audit.endpoint_zero_feature_rows
    )
    assert all(
        zeros == 0
        for _, _, _, zeros
        in toy_audit.transition_response_zero_feature_rows
    )
    assert toy_audit.endpoint_zero_feature_count == 0
    assert toy_audit.transition_zero_feature_count == 0
    assert toy_audit.necessary_conditions_passed


def test_population_audit_json_and_fingerprint_replay_exactly(
    toy_audit: CoverageStateCMIFPopulationAudit,
) -> None:
    replay = audit_coverage_state_cmif_population(
        make_bounded_training_scalar_cache()
    )
    first_payload = toy_audit.canonical_payload()
    second_payload = replay.canonical_payload()

    assert _canonical_json_bytes(first_payload) == _canonical_json_bytes(
        second_payload
    )
    assert stable_fingerprint(first_payload) == stable_fingerprint(
        second_payload
    )
    assert isinstance(
        first_payload["reachability"]["distance_histogram"],
        list,
    )
    json.loads(_canonical_json_bytes(first_payload))


def test_population_audit_rejects_derived_accounting_tampering(
    toy_audit: CoverageStateCMIFPopulationAudit,
) -> None:
    endpoint_shift = _shift_zero_row_observations(
        toy_audit.endpoint_zero_feature_rows,
        source=("clean_minus", "response_ring"),
        destination=("clean_minus", "target"),
        amount=16,
    )
    transition_shift = _shift_zero_row_observations(
        toy_audit.transition_response_zero_feature_rows,
        source=("clean_nonzero", "response_ring"),
        destination=("clean_nonzero", "response_core"),
        amount=16,
    )

    mutations = (
        {"formal_source_bound": 1},
        {
            "endpoint_exact_key_count":
                toy_audit.endpoint_exact_key_count + 1
        },
        {
            "transition_exact_key_count":
                toy_audit.transition_exact_key_count + 1
        },
        {"response_distance_histogram": ((0, 1),)},
        {"endpoint_zero_feature_rows": endpoint_shift},
        {"transition_response_zero_feature_rows": transition_shift},
        {"endpoint_lookup_linf_lower_bound_hex": (-1.0).hex()},
    )
    for mutation in mutations:
        with pytest.raises((TypeError, ValueError)):
            replace(toy_audit, **mutation)


def test_single_run_receipt_can_never_authorize_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        CoverageStateCMIFP0SingleRunReceipt,
        "verify_unchanged",
        lambda self: None,
    )
    receipt = object.__new__(CoverageStateCMIFP0SingleRunReceipt)

    assert receipt.training_authorized is False


def test_in_memory_replay_candidate_can_never_authorize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        CoverageStateCMIFP0ReplayCandidate,
        "verify_unchanged",
        lambda self: None,
    )
    candidate = object.__new__(CoverageStateCMIFP0ReplayCandidate)
    object.__setattr__(
        candidate,
        "first",
        SimpleNamespace(receipt_fingerprint="first"),
    )
    object.__setattr__(
        candidate,
        "second",
        SimpleNamespace(receipt_fingerprint="second"),
    )
    object.__setattr__(candidate, "first_canonical_sha256", "a" * 64)
    object.__setattr__(candidate, "second_canonical_sha256", "a" * 64)
    object.__setattr__(
        candidate,
        "checks",
        (("in_memory_consistency", True),),
    )
    object.__setattr__(candidate, "evidence_fingerprint", "b" * 64)

    payload = candidate.canonical_payload()
    assert candidate.replay_consistency_passed
    assert candidate.training_authorized is False
    assert payload["in_memory_replay_consistent"] is True
    assert payload["persisted_independent_r1_r2_required"] is True
    assert payload["training_authorized"] is False
    assert payload["bounded_400_authorized"] is False


def test_cli_validate_is_static_and_does_not_claim_outputs() -> None:
    output_paths = tuple(
        Path(cli._ROOT / relative)
        for relative in cli.REPLICATE_OUTPUT_REPO_PATHS.values()
    )
    before = tuple(
        (path.exists(), path.is_symlink())
        for path in output_paths
    )

    result = cli.validate_create_only()

    after = tuple(
        (path.exists(), path.is_symlink())
        for path in output_paths
    )
    assert after == before
    assert result["static_contract_valid"] is True
    assert result["run_once_implemented"] is True
    assert result["output_claimed"] is False
    assert result["D_R_cached_tensor_payload_accessed"] is False
    assert result["D_V_accessed"] is False
    assert result["D_T_accessed"] is False
    assert result["training_performed"] is False
    assert result["bounded_400_authorized"] is False
    assert result["not_a_P0_result"] is True


def test_cli_decision_requires_persisted_r1_r2_identity() -> None:
    pending = cli._decision_payload(
        replicate="r1",
        single_eligible=True,
        single_receipt_fingerprint="a" * 64,
        persisted_replay_payload=None,
        persisted_r1_byte_identity=False,
    )
    replay = cli._fingerprinted(
        {
            "checks": {
                "r1_COMPLETE_verified": True,
                "r1_single_eligible_for_replay": True,
                "r2_single_eligible_for_replay": True,
                "p0_core_canonical_bytes_identical": True,
                "p0_core_file_sha256_identical": True,
                "p0_core_receipt_fingerprint_identical": True,
            },
            "persisted_replay_passed": True,
        }
    )
    accepted = cli._decision_payload(
        replicate="r2",
        single_eligible=True,
        single_receipt_fingerprint="a" * 64,
        persisted_replay_payload=replay,
        persisted_r1_byte_identity=True,
    )
    rejected = cli._decision_payload(
        replicate="r2",
        single_eligible=True,
        single_receipt_fingerprint="a" * 64,
        persisted_replay_payload=replay,
        persisted_r1_byte_identity=False,
    )

    assert pending["status"] == "CMIF_V17_P0_REPLAY_PENDING"
    assert pending["bounded_400_authorized"] is False
    assert accepted["status"] == "CMIF_V17_P0_PASS"
    assert accepted["bounded_400_authorized"] is True
    assert rejected["status"] == "CMIF_V17_P0_FAIL"
    assert rejected["bounded_400_authorized"] is False
