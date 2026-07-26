from __future__ import annotations

import json
from pathlib import Path

from cure_lite.cache.schema import file_sha256, stable_fingerprint


ROOT = Path(__file__).resolve().parents[1]
V8 = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "coverage_conserving_subpixel_evidence_allocation_v8"
)
V7 = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "continuously_recoverable_log_vacancy_evidence_crossing_v7"
)
PROPOSAL = V8 / "bounded_implementation_proposal_receipt.json"
CONFIG = V8 / "bounded_config.json"
DRY_CLOSURE = V8 / "bounded_dry_run_closure_receipt.json"
V7_CONFIG = V7 / "bounded_config.json"

PROPOSAL_SHA256 = (
    "c9e06e4ae488b2b3b5e93e2c794cc4bbb55ad0bd5d2558ed6ff5b09a0787054d"
)
PROPOSAL_FINGERPRINT = (
    "8f3008e707d03f23be9760a200a9af219fc0eb9f3f284a3635d066370f3da754"
)
CONFIG_SHA256 = (
    "19ebde5b42643e65177084cb52d456e065e7ee9349852e1c68f4f6778a6c9b47"
)
CONFIG_FINGERPRINT = (
    "baf120fdd7886877e70df3c2186035ab78df9c417aebe549250a142652b417ba"
)
DRY_CLOSURE_SHA256 = (
    "811e5582b9dc99b860fc866faa350b538ad06094a997195da6152cd6013fc935"
)
DRY_CLOSURE_FINGERPRINT = (
    "020485ba9e1feb37a64b1e17272113d23596842cfb76eefb94b9e3f2b3c036c6"
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _verify_fingerprint(
    payload: dict[str, object],
    *,
    field: str,
    expected: str,
) -> None:
    observed = payload[field]
    assert observed == expected
    unsigned = dict(payload)
    del unsigned[field]
    assert stable_fingerprint(unsigned) == expected


def test_v8_bounded_protocol_is_canonical_and_metadata_only() -> None:
    proposal = _load(PROPOSAL)
    config = _load(CONFIG)
    dry_closure = _load(DRY_CLOSURE)

    assert file_sha256(PROPOSAL) == PROPOSAL_SHA256
    assert file_sha256(CONFIG) == CONFIG_SHA256
    assert file_sha256(DRY_CLOSURE) == DRY_CLOSURE_SHA256
    _verify_fingerprint(
        proposal,
        field="proposal_fingerprint",
        expected=PROPOSAL_FINGERPRINT,
    )
    _verify_fingerprint(
        config,
        field="config_fingerprint",
        expected=CONFIG_FINGERPRINT,
    )
    _verify_fingerprint(
        dry_closure,
        field="receipt_fingerprint",
        expected=DRY_CLOSURE_FINGERPRINT,
    )

    assert proposal["method_id"] == config["method_id"] == "cc_sea_v8"
    assert config["dataset"] == "IRSTD-1K"
    assert config["split"] == "D_R"
    assert proposal["current_boundary"] == {
        "D_R_protocol_metadata_may_be_read": True,
        "D_R_dataset_or_cached_tensor_payload_access_allowed": False,
        "D_V_access_allowed": False,
        "D_T_access_allowed": False,
        "real_D_R_bounded_execution_authorized": False,
        "calibration_allowed": False,
        "Pd_or_FA_evaluation_allowed": False,
        "formal_800_allowed": False,
        "full_CURE_allowed": False,
        "other_detector_integration_allowed": False,
    }
    scope = dry_closure["authorization_scope"]
    assert isinstance(scope, dict)
    assert scope["real_D_R_bounded_code_creation_authorized"] is True
    assert scope["real_D_R_payload_access_authorized"] is False
    assert scope["real_D_R_bounded_execution_authorized"] is False

    proposal_binding = config[
        "bounded_implementation_proposal_binding"
    ]
    assert isinstance(proposal_binding, dict)
    assert proposal_binding["file_sha256"] == PROPOSAL_SHA256
    assert proposal_binding["fingerprint"] == PROPOSAL_FINGERPRINT
    dry_binding = config["dry_run_closure_binding"]
    assert isinstance(dry_binding, dict)
    assert dry_binding["file_sha256"] == DRY_CLOSURE_SHA256
    assert dry_binding["fingerprint"] == DRY_CLOSURE_FINGERPRINT


def test_v8_changes_only_the_operator_under_the_v7_bounded_contract() -> None:
    v8 = _load(CONFIG)
    v7 = _load(V7_CONFIG)

    for field in ("anchor_population", "outcome_population", "bounded_gates"):
        assert v8[field] == v7[field]
    v8_budget = v8["budget"]
    v7_budget = v7["budget"]
    assert isinstance(v8_budget, dict)
    assert isinstance(v7_budget, dict)
    assert {
        name: v8_budget[name] for name in v7_budget
    } == v7_budget
    assert v8_budget["backward_calls_per_update"] == 1
    assert v8_budget["optimizer_steps_per_update"] == 1

    v8_opt = v8["optimization"]
    v7_opt = v7["optimization"]
    assert isinstance(v8_opt, dict)
    assert isinstance(v7_opt, dict)
    for field in (
        "loss",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "seed",
    ):
        assert v8_opt[field] == v7_opt[field]

    reconstruction = v8["source_reconstruction"]
    assert isinstance(reconstruction, dict)
    expected_fingerprints = {
        "required_pair_catalog_fingerprint": (
            "4886e52d2cfb3392d0f4fdda376159d6e7f694fd449dc809cf8874793febde76"
        ),
        "required_prepared_catalog_fingerprint": (
            "4955e5b4f1749b5f267db0ac1f031335a16cc48a470d6446ca6c99d04a5e85ed"
        ),
        "required_anchor_population_fingerprint": (
            "d251ed9061dd373aa0bf0e4ceeebbafc7ca32a4bab72c2f24601a20868d6d1cd"
        ),
        "required_materializer_fingerprint": (
            "8cc4eac43ad708265d8639c4b577b37bd81be8ccde73e79993ba18c65dca10ff"
        ),
        "required_all_pair_inputs_fingerprint": (
            "f3573b469464015865870440427deed341b7e2cddd8e866bdede2ee44c509b6c"
        ),
        "required_gt_union_population_fingerprint": (
            "afa80e88581fa5ee5f832dc70624d9ee54ca9541d1b33b7f5957ef0ed08e3ae5"
        ),
        "required_factual_schedule_fingerprint": (
            "57264042879d9850aa538e01563496a8d3de7b82556d2b5ef15ca7f32b66fac3"
        ),
        "required_outcome_schedule_fingerprint": (
            "747123867c88fd1444a514bf70e51013b739f39df2857e5ed021239e4847ec93"
        ),
        "required_outcome_sequence_fingerprint": (
            "6f4c45d51cfa8364d97a620af1bad1ea565f9ce4fc72c4d638d141fb056cffd0"
        ),
    }
    for name, expected in expected_fingerprints.items():
        assert reconstruction[name] == expected

    contract = _load(PROPOSAL)["single_change_contract"]
    assert isinstance(contract, dict)
    assert contract["only_allowed_model_change"] == (
        "independent_per_phase_crossing_to_one_cell_budget_plus_"
        "conserved_subpixel_allocation"
    )
    assert contract["parameter_count"] == 4385
    assert contract["parameter_tensor_count"] == 6
    assert contract["loss_unchanged"] is True
    assert contract["population_unchanged"] is True
    assert contract["schedule_unchanged"] is True
    assert contract["conservation_loss_or_auxiliary_module_allowed"] is False


def test_v8_real_execution_remains_separately_blocked() -> None:
    proposal = _load(PROPOSAL)
    config = _load(CONFIG)

    future = proposal["future_real_execution_control"]
    assert isinstance(future, dict)
    assert future["authorization_receipt_not_created"] is True
    assert future["exact_real_D_R_run_count_if_later_authorized"] == 1
    assert future["device_if_later_authorized"] == "cuda:0"
    assert future["pause_temperature_celsius"] == 82
    assert future["resume_temperature_celsius"] == 75
    assert future["resume_allowed"] is False
    assert future["automatic_retry_allowed"] is False
    assert future["formal_800_allowed"] is False

    execution = config["execution_policy"]
    assert isinstance(execution, dict)
    assert execution["same_version_real_bounded_runs_max"] == 1
    assert execution[
        "D_R_payload_access_before_separate_authorization_allowed"
    ] is False
    assert execution["D_V_access_allowed"] is False
    assert execution["D_T_access_allowed"] is False
    assert execution["performance_evaluation_allowed"] is False
    assert execution["formal_800_training_allowed_by_this_config"] is False

    closure = V8 / "bounded_implementation_closure_receipt.json"
    authorization = V8 / "bounded_run_authorization_receipt.json"
    if closure.exists():
        closure_payload = _load(closure)
        closure_fingerprint = closure_payload["receipt_fingerprint"]
        assert isinstance(closure_fingerprint, str)
        _verify_fingerprint(
            closure_payload,
            field="receipt_fingerprint",
            expected=closure_fingerprint,
        )
        assert closure_payload["boundary"][
            "real_D_R_bounded_execution_authorized"
        ] is False
        assert closure_payload["authorization_eligibility"][
            "directly_authorizes_real_D_R_run"
        ] is False
        assert closure_payload["authorization_eligibility"][
            "single_real_D_R_run_eligible"
        ] is True

    if authorization.exists():
        assert closure.exists()
        authorization_payload = _load(authorization)
        authorization_fingerprint = authorization_payload[
            "receipt_fingerprint"
        ]
        assert isinstance(authorization_fingerprint, str)
        _verify_fingerprint(
            authorization_payload,
            field="receipt_fingerprint",
            expected=authorization_fingerprint,
        )
        authorized = authorization_payload["authorization"]
        assert authorized["real_D_R_bounded_execution"] is True
        assert authorized["exact_run_count"] == 1
        assert authorized["device"] == "cuda:0"
        assert authorized["resume_allowed"] is False
        assert authorized["automatic_retry_allowed"] is False
        assert authorized["D_V_access_allowed"] is False
        assert authorized["D_T_access_allowed"] is False
        assert authorized["formal_800_allowed"] is False
        assert authorization_payload["implementation_closure_binding"][
            "file_sha256"
        ] == file_sha256(closure)
