from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.decoder import CURELiteDecoder
from cure_lite.experiment.paired_bounded_learnability import (
    BOUNDED_MICRO_POPULATION_SCHEMA,
    BOUNDED_MICRO_SCHEDULE_SCHEMA,
    BoundedMicroPopulation,
    _ForwardLedger,
    _deterministic_torch_runtime,
    build_bounded_micro_schedule,
)
from cure_lite.losses import CURELiteLoss
from cure_lite.paired_losses import PairedDifferenceLoss
from cure_lite.paired_types import PairBatch
from cure_lite.train.paired_step import paired_train_step
from cure_lite.train.step import BranchBatch
from tools import run_paired_bounded_learnability as runner


_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "paired_bounded_learnability_v1"
    / "config.json"
)


def _config() -> dict[str, object]:
    payload = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _fake_population() -> BoundedMicroPopulation:
    population = object.__new__(BoundedMicroPopulation)
    object.__setattr__(
        population,
        "clean_pairs",
        tuple(
            SimpleNamespace(sample_id=f"source-{index:02d}")
            for index in range(16)
        ),
    )
    object.__setattr__(
        population,
        "factual_miss",
        tuple(object() for _ in range(16)),
    )
    object.__setattr__(
        population,
        "factual_no_miss",
        tuple(object() for _ in range(16)),
    )
    return population


def _counts(rows: list[list[int]], size: int) -> list[int]:
    ledger = Counter(index for row in rows for index in row)
    return [ledger[index] for index in range(size)]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _sealed_failure_artifact(root: Path) -> None:
    config = _config()
    contract = config["control_preflight_contract"]
    assert isinstance(contract, dict)
    pair_catalog_fingerprint = str(
        config["input_binding"]["real_pair_catalog_fingerprint"]
    )

    clean_pairs = [
        {
            "pair_id": stable_fingerprint({"kind": "clean", "index": index}),
            "sample_id": f"source-{index:02d}",
        }
        for index in range(16)
    ]
    factual_miss = [
        {
            "anchor_id": stable_fingerprint(
                {"branch": "factual_miss", "index": index}
            ),
            "sample_id": f"miss-{index:02d}",
            "positive_gt_ids": [index + 1],
        }
        for index in range(16)
    ]
    factual_no_miss = [
        {
            "anchor_id": stable_fingerprint(
                {"branch": "factual_no_miss", "index": index}
            ),
            "sample_id": f"no-miss-{index:02d}",
            "positive_gt_ids": [],
        }
        for index in range(16)
    ]
    component_null = [
        {
            "pair_id": stable_fingerprint(
                {"kind": "component_null", "index": index}
            ),
            "sample_id": f"component-{index:02d}",
        }
        for index in range(16)
    ]
    identity_null = [
        {
            "pair_id": stable_fingerprint(
                {"kind": "identity_null", "index": index}
            ),
            "sample_id": f"identity-{index:02d}",
        }
        for index in range(16)
    ]
    micro_core = {
        "schema_version": BOUNDED_MICRO_POPULATION_SCHEMA,
        "seed": 42,
        "pair_catalog_fingerprint": pair_catalog_fingerprint,
        "prepared_catalog_fingerprint": "9" * 64,
        "selection_rule": (
            "stable-hash-over-identities-source-first-without-"
            "feature-loss-or-result-access-v1"
        ),
        "clean_pairs": clean_pairs,
        "factual_miss": factual_miss,
        "factual_no_miss": factual_no_miss,
        "component_null": component_null,
        "identity_null": identity_null,
    }
    micro = runner._fingerprinted(
        {
            **micro_core,
            "population_fingerprint": stable_fingerprint(micro_core),
        }
    )

    pair_indices = [
        [(2 * update) % 16, (2 * update + 1) % 16]
        for update in range(400)
    ]
    miss_indices = [
        [(4 * update + draw) % 16 for draw in range(4)]
        for update in range(400)
    ]
    no_miss_indices = [
        [(4 * update + draw) % 16 for draw in range(4)]
        for update in range(400)
    ]
    schedule_core = {
        "schema_version": BOUNDED_MICRO_SCHEDULE_SCHEMA,
        "optimizer_updates": 400,
        "steps_per_epoch": 40,
        "pair_indices": pair_indices,
        "factual_miss_indices": miss_indices,
        "factual_no_miss_indices": no_miss_indices,
        "pair_counts": _counts(pair_indices, 16),
        "factual_miss_counts": _counts(miss_indices, 16),
        "factual_no_miss_counts": _counts(no_miss_indices, 16),
    }
    schedule = runner._fingerprinted(
        {
            **schedule_core,
            "schedule_fingerprint": stable_fingerprint(schedule_core),
            "exposure": {
                "pair_counts": schedule_core["pair_counts"],
                "factual_miss_counts": schedule_core[
                    "factual_miss_counts"
                ],
                "factual_no_miss_counts": schedule_core[
                    "factual_no_miss_counts"
                ],
            },
        }
    )
    config_binding = runner._fingerprinted(
        {
            "schema_version": (
                "cure-lite-paired-bounded-config-binding-v1"
            ),
            "config": config,
            "config_file_sha256": runner.BOUNDED_CONFIG_FILE_SHA256,
            "control_preflight_complete_file_sha256": contract[
                "authority_complete_file_sha256"
            ],
            "control_preflight_complete_fingerprint": contract[
                "authority_complete_fingerprint"
            ],
            "control_preflight_run_receipt_file_sha256": contract[
                "run_receipt_file_sha256"
            ],
            "control_preflight_run_receipt_fingerprint": contract[
                "run_receipt_fingerprint"
            ],
            "control_preflight_byte_identical_replay_verified": True,
            "control_preflight_artifact_files": {},
            "implementation_files": {},
            "runtime": {
                "device": "cpu",
                "allowed_split": "D_R",
                "cublas_workspace_config": ":4096:8",
            },
        }
    )
    failure_core = {
        "schema_version": (
            "cure-lite-paired-bounded-execution-failure-v1"
        ),
        "exception_type": "RuntimeError",
        "message": "sealed-test-failure",
        "structural_execution_pass": False,
        "computational_learnability_pass": False,
        "threshold_or_budget_changed": False,
    }
    failure = runner._fingerprinted(failure_core)
    decision = runner._fingerprinted(
        {
            "schema_version": runner.BOUNDED_DECISION_SCHEMA,
            "status": "STRUCTURAL_EXECUTION_ERROR",
            "structural_execution_pass": False,
            "computational_learnability_pass": False,
            "not_performance_evidence": True,
            "authorizes_formal_800": False,
            "authorizes_D_V_or_D_T": False,
            "threshold_or_budget_changed_after_result": False,
            "evidence_kind": "failure",
            "evidence_receipt_fingerprint": failure[
                "receipt_fingerprint"
            ],
            "failure": failure_core,
            "next_route": (
                "review_bounded_evidence_without_threshold_or_budget_change"
            ),
        }
    )

    receipts = root / "receipts"
    _write_json(receipts / "config_binding.json", config_binding)
    _write_json(receipts / "micro_population.json", micro)
    _write_json(receipts / "schedule.json", schedule)
    _write_json(receipts / "failure.json", failure)
    _write_json(receipts / "decision.json", decision)
    artifact_files = runner._artifact_hashes(root)
    complete = runner._fingerprinted(
        {
            "schema_version": runner.BOUNDED_RUN_SCHEMA,
            "execution_status": "complete",
            "decision": "STRUCTURAL_EXECUTION_ERROR",
            "structural_execution_pass": False,
            "computational_learnability_pass": False,
            "not_performance_evidence": True,
            "authorizes_formal_800": False,
            "split": "D_R",
            "pair_catalog_fingerprint": pair_catalog_fingerprint,
            "config_fingerprint": config["config_fingerprint"],
            "config_binding_fingerprint": config_binding[
                "receipt_fingerprint"
            ],
            "control_preflight_complete_fingerprint": contract[
                "authority_complete_fingerprint"
            ],
            "micro_population_fingerprint": micro[
                "population_fingerprint"
            ],
            "micro_population_receipt_fingerprint": micro[
                "receipt_fingerprint"
            ],
            "schedule_fingerprint": schedule["schedule_fingerprint"],
            "schedule_receipt_fingerprint": schedule[
                "receipt_fingerprint"
            ],
            "evidence_kind": "failure",
            "evidence_receipt_fingerprint": failure[
                "receipt_fingerprint"
            ],
            "decision_fingerprint": decision["receipt_fingerprint"],
            "artifact_files": artifact_files,
            "artifact_file_count": len(artifact_files),
            "training_scope": "fresh_decoder_only_bounded_400_updates",
            "formal_training_performed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "performance_evaluation_performed": False,
            "deterministic_runtime_contract_required": True,
            "exact_replay_required_under_same_frozen_environment": True,
        },
        field="complete_fingerprint",
    )
    _write_json(root / "COMPLETE.json", complete)


def test_final_control_r1_r2_binding_is_strict_and_ready() -> None:
    config = _config()
    contract = config["control_preflight_contract"]
    assert isinstance(contract, dict)
    authority = _ROOT / str(contract["authority_complete_path"])
    payload = runner._verify_control_preflight(authority, contract)
    assert payload["execution_status"] == "complete"
    assert payload["status"] == "complete"
    assert payload["target_permutation_status"] == "READY"
    assert payload["matched_controls_static_preflight_pass"] is True
    assert payload["byte_identical_replay_verified"] is True


def test_cli_has_no_seed_budget_or_evaluation_split_override() -> None:
    destinations = {
        action.dest for action in runner.build_parser()._actions
    }
    assert destinations == {
        "help",
        "config",
        "control_preflight_complete",
        "device",
        "output",
    }
    base = [
        "--config",
        str(_CONFIG_PATH),
        "--control-preflight-complete",
        "control.json",
        "--device",
        "cpu",
        "--output",
        "out",
    ]
    for forbidden in ("--seed", "--budget", "--D_V", "--D_T"):
        with pytest.raises(SystemExit):
            runner.parse_args([*base, forbidden, "1"])


def test_cpu_deterministic_policy_restores_process_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specification = _config()["determinism"]
    assert isinstance(specification, dict)
    previous = {
        "algorithms": torch.are_deterministic_algorithms_enabled(),
        "warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "caller-value")
    with _deterministic_torch_runtime(
        torch.device("cpu"),
        specification,
    ) as evidence:
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
        assert evidence["contract_satisfied"] is True
    assert evidence["flags_restored_after_execution"] is True
    assert torch.are_deterministic_algorithms_enabled() is previous[
        "algorithms"
    ]
    assert (
        torch.is_deterministic_algorithms_warn_only_enabled()
        is previous["warn_only"]
    )
    assert (
        torch.backends.cudnn.deterministic
        is previous["cudnn_deterministic"]
    )
    assert torch.backends.cudnn.benchmark is previous["cudnn_benchmark"]
    assert runner.os.environ["CUBLAS_WORKSPACE_CONFIG"] == "caller-value"


def test_frozen_schedule_has_exact_400_update_uniform_ledgers() -> None:
    budget = _config()["budget"]
    assert isinstance(budget, dict)
    schedule = build_bounded_micro_schedule(
        _fake_population(),
        budget,
    )
    assert schedule.optimizer_updates == 400
    assert schedule.steps_per_epoch == 40
    assert len(schedule.pair_indices) == 400
    assert len(schedule.factual_miss_indices) == 400
    assert len(schedule.factual_no_miss_indices) == 400
    assert set(schedule.pair_counts) == {50}
    assert set(schedule.factual_miss_counts) == {100}
    assert set(schedule.factual_no_miss_counts) == {100}
    assert sum(schedule.pair_counts) == 800
    assert sum(schedule.factual_miss_counts) == 1600
    assert sum(schedule.factual_no_miss_counts) == 1600


def test_one_toy_update_has_exact_forward_and_state_budget() -> None:
    feature = torch.zeros(2, 2, 4, 4)
    occupancy_plus = torch.zeros(2, 1, 4, 4, dtype=torch.bool)
    occupancy_minus = torch.zeros_like(occupancy_plus)
    increment = torch.zeros(2, 1, 4, 4)
    for index, location in enumerate(((1, 1), (2, 2))):
        row, column = location
        feature[index, 0, row, column] = 2.0
        occupancy_plus[index, 0, row, column] = True
        increment[index, 0, row, column] = 1.0
    pair_batch = PairBatch(
        feature=feature,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        label_increment=increment,
        image_valid_mask=torch.ones_like(occupancy_plus),
        pair_ids=(
            stable_fingerprint({"pair": 0}),
            stable_fingerprint({"pair": 1}),
        ),
        sample_ids=("toy-source-0", "toy-source-1"),
        group_ids=("toy-group-0", "toy-group-1"),
        pair_kinds=("clean_positive", "clean_positive"),
        projection_visible=(True, True),
    )
    factual_occupancy = torch.zeros(4, 1, 4, 4, dtype=torch.bool)
    factual_valid = torch.ones_like(factual_occupancy)
    factual_batches = {
        "factual_miss": BranchBatch(
            feature=torch.stack(
                [feature[index % 2] for index in range(4)]
            ),
            occupancy=factual_occupancy,
            target=torch.stack(
                [increment[index % 2] for index in range(4)]
            ),
            valid_mask=factual_valid,
        ),
        "factual_no_miss": BranchBatch(
            feature=torch.zeros(4, 2, 4, 4),
            occupancy=factual_occupancy.clone(),
            target=torch.zeros(4, 1, 4, 4),
            valid_mask=factual_valid.clone(),
        ),
    }
    decoder = CURELiteDecoder(feature_channels=2)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    ledger = _ForwardLedger(decoder)
    try:
        logs = paired_train_step(
            decoder,
            CURELiteLoss(),
            PairedDifferenceLoss(),
            optimizer,
            factual_batches,
            pair_batch,
        )
        calls, states = ledger.snapshot()
    finally:
        ledger.close()
    assert logs["optimizer_steps"] == 1
    assert calls == 3
    assert states == 12


def test_runner_structural_gate_keys_match_the_freeze_exactly() -> None:
    config = _config()
    gates = config["gates"]["structural_execution"]
    assert isinstance(gates, dict)
    runner_only = {
        "all_input_fingerprints_verified",
        "control_preflight_verified",
    }
    core_keys = {
        "deterministic_runtime_contract_satisfied",
        "micro_population_counts_exact",
        "clean_pair_sources_distinct",
        "all_optimizer_updates_completed",
        "all_gradients_finite",
        "every_update_total_gradient_norm_positive",
        "decoder_parameters_changed",
        "training_forward_budget_exact",
        "evaluation_forward_budget_exact",
        "total_forward_budget_exact",
        "all_exposure_ledgers_complete",
    }
    assert set(gates) == core_keys | runner_only
    result = {
        "structural_checks": {
            name: True for name in sorted(core_keys)
        },
        "computational_gates": {"all_pass": True},
        "structural_execution_pass": True,
        "computational_learnability_pass": True,
    }
    bound = runner._bind_runner_structural_gates(result, config)
    assert set(bound["structural_checks"]) == set(gates)
    assert bound["structural_execution_pass"] is True
    assert bound["computational_learnability_pass"] is True
    bad = dict(result)
    bad["structural_checks"] = {
        **result["structural_checks"],
        "unexpected": True,
    }
    with pytest.raises(RuntimeError, match="differs from the freeze"):
        runner._bind_runner_structural_gates(bad, config)


def test_sealed_failure_artifact_loads_and_tampering_is_rejected(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bounded-failure"
    _sealed_failure_artifact(output)
    published = runner.load_bounded_learnability_artifact(output)
    assert published.decision == "STRUCTURAL_EXECUTION_ERROR"
    assert published.structural_execution_pass is False
    assert published.computational_learnability_pass is False

    failure_path = output / "receipts" / "failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    failure["message"] = "tampered"
    _write_json(failure_path, failure)
    with pytest.raises(RuntimeError):
        runner.load_bounded_learnability_artifact(output)
