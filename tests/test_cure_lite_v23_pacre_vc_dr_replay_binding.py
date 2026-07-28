from __future__ import annotations

from types import SimpleNamespace

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.experiment.coverage_state_bounded_protocol import (
    COVERAGE_STATE_BOUNDED_SEED,
)
from cure_lite_v23.dataset_free import (
    PACRE_FORMAL_FEATURE_CHANNELS,
    PACRE_FORMAL_FEATURE_STRIDE,
    PACRE_FORMAL_PARAMETER_COUNT,
    PACRE_FORMAL_WIDTH,
)
from cure_lite_v23.dr_gate import (
    PACRE_VC_DR_CHECK_NAMES,
    PACRE_VC_DR_CONTEXT_STATE_COUNT,
    PACRE_VC_DR_TARGET_STATE_COUNT,
    recompute_pacre_vc_dr_checks,
)
from cure_lite_v23.factory import PACRE_VC_PARAMETER_NAMES
from cure_lite_v23.pacre_vc import (
    PACRE_VC_FIELDS_FQCN,
    PACRE_VC_VERIFIER_POLICY,
)


def _inputs():
    source_cache = SimpleNamespace(
        raw_catalog=SimpleNamespace(split="D_R"),
        cache_fingerprint="c" * 64,
    )
    real_inputs = SimpleNamespace(
        source_binding=SimpleNamespace(split="D_R"),
        scalar_cache=source_cache,
    )
    population = SimpleNamespace(
        seed=COVERAGE_STATE_BOUNDED_SEED,
        source_cache=source_cache,
        source_cache_fingerprint=source_cache.cache_fingerprint,
    )
    return real_inputs, population


def _probe() -> dict[str, object]:
    model_fqcn = (
        "cure_lite_v23.pacre_vc."
        "CURELitePACREVerifierCorrectedLevelSet"
    )
    config_fqcn = (
        "cure_lite_v23.pacre_vc."
        "CoverageStatePACREVerifierCorrectedConfig"
    )
    model_contract = {
        "model_class": model_fqcn,
        "config_class": config_fqcn,
        "parameter_count": PACRE_FORMAL_PARAMETER_COUNT,
        "config": {
            "feature_channels": PACRE_FORMAL_FEATURE_CHANNELS,
            "feature_stride": PACRE_FORMAL_FEATURE_STRIDE,
            "width": PACRE_FORMAL_WIDTH,
            "verifier_policy": PACRE_VC_VERIFIER_POLICY,
        },
        "parameter_shapes": {
            name: [1] for name in PACRE_VC_PARAMETER_NAMES
        },
    }
    ordered_target = "t" * 64
    target = {
        "observed_state_count": PACRE_VC_DR_TARGET_STATE_COUNT,
        "expected_state_count": PACRE_VC_DR_TARGET_STATE_COUNT,
        "state_ids_unique": True,
        "source_state_ids_unique": True,
        "legacy_replay_complete": True,
        "all_gate_eligible_integrity_passed": True,
        "ordered_source_state_ids_fingerprint": ordered_target,
    }
    context = {
        "observed_state_count": PACRE_VC_DR_CONTEXT_STATE_COUNT,
        "expected_state_count": PACRE_VC_DR_CONTEXT_STATE_COUNT,
        "state_ids_unique": True,
        "source_state_ids_unique": True,
        "legacy_replay_complete": True,
        "all_gate_eligible_integrity_passed": True,
    }
    initial = "i" * 64
    return {
        "model_fqcn": model_fqcn,
        "config_fqcn": config_fqcn,
        "fields_fqcn": PACRE_VC_FIELDS_FQCN,
        "model_contract": model_contract,
        "model_contract_fingerprint": stable_fingerprint(model_contract),
        "v22_v23_initial_raw_state_parity": True,
        "v22_initial_model_fingerprint": initial,
        "initial_model_fingerprint": initial,
        "representation": {"all_fields_exact_pacre": True},
        "algebra_ledger": {
            "target_summary": target,
            "context_summary": context,
            "target_context_scoped_ids_disjoint": True,
            "union_unique_state_call_count": (
                PACRE_VC_DR_TARGET_STATE_COUNT
                + PACRE_VC_DR_CONTEXT_STATE_COUNT
            ),
            "expected_union_state_call_count": (
                PACRE_VC_DR_TARGET_STATE_COUNT
                + PACRE_VC_DR_CONTEXT_STATE_COUNT
            ),
        },
        "gradient_path": {},
        "field_direction": {},
        "sealed_v22_replay_binding": {
            "expected_initial_model_fingerprint": initial,
            "observed_initial_model_fingerprint": initial,
            "initial_model_fingerprint_matches": True,
            "expected_ordered_target_state_ids_fingerprint": (
                ordered_target
            ),
            "observed_ordered_target_state_ids_fingerprint": (
                ordered_target
            ),
            "ordered_target_state_ids_match": True,
        },
    }


def _checks(probe: dict[str, object]) -> dict[str, bool]:
    real_inputs, population = _inputs()
    return dict(
        recompute_pacre_vc_dr_checks(
            dataset_free_receipt_fingerprint="d" * 64,
            real_inputs=real_inputs,
            bounded_population=population,
            probe=probe,
        )
    )


def test_sealed_replay_recomputes_initial_fingerprint_equality() -> None:
    probe = _probe()
    assert _checks(probe)[PACRE_VC_DR_CHECK_NAMES[2]]

    replay = dict(probe["sealed_v22_replay_binding"])
    replay["expected_initial_model_fingerprint"] = "x" * 64
    replay["initial_model_fingerprint_matches"] = True
    probe["sealed_v22_replay_binding"] = replay

    assert not _checks(probe)[PACRE_VC_DR_CHECK_NAMES[2]]


def test_sealed_replay_recomputes_ordered_target_equality() -> None:
    probe = _probe()
    assert _checks(probe)[PACRE_VC_DR_CHECK_NAMES[3]]

    replay = dict(probe["sealed_v22_replay_binding"])
    replay["expected_ordered_target_state_ids_fingerprint"] = "x" * 64
    replay["ordered_target_state_ids_match"] = True
    probe["sealed_v22_replay_binding"] = replay

    assert not _checks(probe)[PACRE_VC_DR_CHECK_NAMES[3]]
