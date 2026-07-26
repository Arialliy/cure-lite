from __future__ import annotations

import pytest

from cure_lite.cache.schema import stable_fingerprint
from tools.evaluate_crossing_factorized_toy_gate import (
    CASES,
    EXPECTED_CONFIG_FINGERPRINT,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_PARAMETER_COUNT,
    EXPECTED_PARAMETER_TENSORS,
    EXPECTED_PROPOSAL_FINGERPRINT,
    EXPECTED_PROPOSAL_SHA256,
    FROZEN_LEARNING_RATE,
    FROZEN_SEED,
    FROZEN_UPDATES,
    evaluate,
)


@pytest.fixture(scope="module")
def toy_result() -> dict[str, object]:
    return evaluate()


def test_cr_lvec_v7_passes_all_six_frozen_toy_cases(
    toy_result: dict[str, object],
) -> None:
    result = toy_result

    assert result["decision"] == "CR_LVEC_V7_TOY_GATE_PASS"
    assert result["all_pass"] is True
    assert result["passed_case_count"] == 6
    assert result["failed_case_count"] == 0
    assert result["passed_family_count"] == 2
    assert result["passed_cases"] == [case[1] for case in CASES]
    assert result["failed_cases"] == []
    assert result["case_level_outside_count_check_vacuous_count"] == 6
    assert all(
        family["passed"] is True
        for family in result["case_families"].values()
    )
    locality = result["nonvacuous_locality_audit"]
    assert locality["outside_count_support_pixel_count"] > 0
    assert locality["outside_count_support_nonempty"] is True
    assert locality["outside_count_change_exact"] is True
    assert locality["outside_count_change_probability_exact"] is True
    assert locality["deletion_monotone"] is True
    assert locality["all_fields_finite"] is True
    assert locality["passed"] is True

    for row in result["cases"]:
        assert row["passed"] is True
        assert row["failed_checks"] == []
        assert all(row["checks"].values())
        endpoint = row["endpoint_gradient_contract"]
        assert endpoint["plus_finite"] is True
        assert endpoint["minus_finite"] is True
        assert endpoint["plus_nonzero"] is True
        assert endpoint["minus_nonzero"] is True
        assert endpoint["plus_l2_norm"] > 0.0
        assert endpoint["minus_l2_norm"] > 0.0

        gradients = row["parameter_gradient_contract"]
        assert (
            gradients["observed_parameter_tensor_count"]
            == EXPECTED_PARAMETER_TENSORS
        )
        assert (
            gradients["observed_parameter_count"]
            == EXPECTED_PARAMETER_COUNT
        )
        assert (
            gradients[
                "training_step_enforces_finite_gradient_each_update"
            ]
            is True
        )
        assert (
            gradients["all_six_finite_at_first_and_last_update"]
            is True
        )
        assert (
            gradients["all_six_nonzero_at_first_and_last_update"]
            is True
        )
        assert gradients["snapshot_updates"] == [1, FROZEN_UPDATES]
        assert len(gradients["parameters"]) == EXPECTED_PARAMETER_TENSORS
        for parameter in gradients["parameters"].values():
            first = parameter["first_update"]
            last = parameter["last_update"]
            assert first["update"] == 1
            assert last["update"] == FROZEN_UPDATES
            assert first["finite"] is True
            assert last["finite"] is True
            assert first["nonzero"] is True
            assert last["nonzero"] is True
            assert first["l2_norm"] > 0.0
            assert last["l2_norm"] > 0.0

        margin = row["margin_contract"]
        assert margin["snapshot_count"] == 2
        assert (
            margin["snapshot_scope"]
            == "initial_and_final_training_states"
        )
        assert (
            margin["maximum_observed_scope"]
            == "all_decoder_forward_fields_calls_during_case_evaluation"
        )
        assert margin["observed_forward_fields_calls"] == 970
        assert margin["initial"]["all_fields_finite"] is True
        assert margin["final"]["all_fields_finite"] is True
        assert margin["initial"]["ratio_identity_allclose"] is True
        assert margin["final"]["ratio_identity_allclose"] is True
        assert margin["initial"]["forward_crossing_exact"] is True
        assert margin["final"]["forward_crossing_exact"] is True
        assert (
            margin["maximum_observed_absolute_margin"]
            >= margin["initial"]["max_abs_margin"]
        )
        assert (
            margin["maximum_observed_absolute_margin"]
            >= margin["final"]["max_abs_margin"]
        )
        operator = row["ratio_operator_fields"]
        assert operator["identity_exact"] is True
        assert operator["identity_max_abs_logit_delta"] == 0.0
        assert operator["outside_count_change_exact"] is True
        assert operator["outside_count_change_check_vacuous"] is True
        assert (
            operator["clean_response_inside_count_change_support"]
            is True
        )
        if row["family_id"] == (
            "response_outside_removed_component_inside_count_support"
        ):
            assert (
                operator[
                    "clean_response_outside_direct_projected_xor_lift"
                ]
                is True
            )

    numerical = result["numerical_contract_audit"]
    assert numerical["negative_probe_pass"] is True
    assert numerical["negative_probe_gradient"] > 0.0
    assert numerical["first_zero_recovery_probe_failed_fast"] is True
    assert numerical["largest_finite_probe_pass"] is True
    assert numerical["first_nonfinite_probe_failed_fast"] is True
    assert numerical["silent_clamp_observed"] is False
    assert numerical["passed"] is True


def test_support_mismatch_cases_use_single_pixel_deletions(
    toy_result: dict[str, object],
) -> None:
    result = toy_result
    rows = {
        row["case_id"]: row
        for row in result["cases"]
        if row["family_id"]
        == "response_outside_removed_component_inside_count_support"
    }
    assert set(rows) == {
        "support_one_pixel",
        "support_two_pixels",
        "support_three_pixels",
    }
    for row in rows.values():
        geometry = row["geometry_contract"]
        assert geometry["clean_removed_component_pixels"] == [[0, 0]]
        assert geometry["clean_removed_component_pixel_count"] == 1
        assert (
            geometry["component_null_removed_component_pixel_count"]
            == 1
        )
        assert geometry["response_outside_removed_component"] is True
        assert (
            geometry["response_outside_direct_projected_xor_lift"]
            is True
        )
        assert geometry["response_inside_count_change_support"] is True
        assert geometry["passed"] is True


def test_toy_result_exactly_binds_frozen_protocol_and_has_no_real_access(
    toy_result: dict[str, object],
) -> None:
    result = toy_result
    binding = result["protocol_binding"]

    assert binding["toy_config_file_sha256"] == EXPECTED_CONFIG_SHA256
    assert (
        binding["toy_config_fingerprint"]
        == EXPECTED_CONFIG_FINGERPRINT
    )
    assert binding["proposal_file_sha256"] == EXPECTED_PROPOSAL_SHA256
    assert (
        binding["proposal_fingerprint"]
        == EXPECTED_PROPOSAL_FINGERPRINT
    )
    assert result["contract"]["seed"] == FROZEN_SEED
    assert result["contract"]["updates"] == FROZEN_UPDATES
    assert (
        result["contract"]["learning_rate"]
        == FROZEN_LEARNING_RATE
    )
    assert result["bounded_code_creation_authorized"] is False
    assert (
        result["bounded_code_creation_eligible_after_replay"] is True
    )
    assert result["real_D_R_bounded_authorized"] is False
    assert result["real_D_R_status"] == "NOT_RUN_TOY_PHASE"
    assert result["D_R_accessed"] is False
    assert result["D_V_accessed"] is False
    assert result["D_T_accessed"] is False
    assert result["detection_performance_evaluated"] is False
    assert result["formal_800_authorized"] is False
    assert result["automatic_retry_performed"] is False

    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)
