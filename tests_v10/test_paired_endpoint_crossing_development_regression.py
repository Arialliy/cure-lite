from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest
import torch

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.experiment.conservative_toy_inputs import (
    CONSERVATIVE_TOY_CASES,
)
from tools import evaluate_paired_endpoint_crossing_development_regression as dev


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "paired_endpoint_crossing_objective_v10"
)
CONFIG = PROTOCOL / "development_regression_config.json"
PROPOSAL = PROTOCOL / "proposal_receipt.json"
EVALUATOR = (
    ROOT
    / "tools"
    / "evaluate_paired_endpoint_crossing_development_regression.py"
)
CLOSURE = (
    PROTOCOL / "development_regression_r2_implementation_closure.json"
)
R1_INVALIDATION = (
    PROTOCOL
    / "development_regression_r1_verifier_invalidation_receipt.json"
)
PECO_LOSS = ROOT / "cure_lite" / "paired_endpoint_crossing_losses.py"
FROZEN_DECODER = ROOT / "cure_lite" / "conservative_factorized_decoder.py"
PARENT_LOSS = ROOT / "cure_lite" / "paired_outcome_losses.py"
FROZEN_STEP = ROOT / "cure_lite" / "train" / "paired_outcome_step.py"
TOY_INPUTS = (
    ROOT / "cure_lite" / "experiment" / "conservative_toy_inputs.py"
)
ABSOLUTE_LOSS = ROOT / "cure_lite" / "losses.py"
LOSS_CONFIG = ROOT / "cure_lite" / "config.py"

CONFIG_SHA256 = (
    "9b08bc4a89e29414cdcc7f4a100fac29e0eb524ae9613e7985be52d1f486ae4b"
)
CONFIG_FINGERPRINT = (
    "3f6f9b4559cb6c813d966486c798583f076b210d5527f409d8a35571ac97f98a"
)
PROPOSAL_SHA256 = (
    "74eb7196944135fa8c620dca8c6593460fc7b7086d08ce6104071dad9d88e47a"
)
PROPOSAL_FINGERPRINT = (
    "377d3b5e5cdf7fdb2b903bd423897b9aea436ee943e00206cd26865b95599365"
)
PECO_LOSS_SHA256 = (
    "dd8c83f00cd26dcfb55116998bcd53541471f3b17ff9d7e6a7be69e784883205"
)
FROZEN_DECODER_SHA256 = (
    "fb7b4aeb16934218d5add300a3be2350d6c77615064486cb92cf93399ab05528"
)
PARENT_LOSS_SHA256 = (
    "c873b23afe76038f72a93ed99ef9023c090a7fda321c6ac5f725938d774b5c0e"
)
FROZEN_STEP_SHA256 = (
    "479cc663779a48ff7eee447e9582850d8431ccc633e970452ad4c35f526a2265"
)
TOY_INPUTS_SHA256 = (
    "f86418a06adfc6866a900e697992c0a4959e936e2ed439d6d5f13b94aeca40b0"
)
ABSOLUTE_LOSS_SHA256 = (
    "fa47592c89462a694e1fe19f87f223486a4c07f3e5d48fbf8fce6262c0ff25e9"
)
LOSS_CONFIG_SHA256 = (
    "63399343ba36c5d2a06cc26b8fbfc575f056eca1c9280eb417d57434e1bcf471"
)


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_development_config_is_frozen_and_strictly_source_bound() -> None:
    config = _load(CONFIG)
    binding = dev._load_protocol_binding()

    assert file_sha256(CONFIG) == CONFIG_SHA256
    assert file_sha256(PROPOSAL) == PROPOSAL_SHA256
    assert file_sha256(PECO_LOSS) == PECO_LOSS_SHA256
    assert file_sha256(FROZEN_DECODER) == FROZEN_DECODER_SHA256
    assert file_sha256(PARENT_LOSS) == PARENT_LOSS_SHA256
    assert file_sha256(FROZEN_STEP) == FROZEN_STEP_SHA256
    assert file_sha256(TOY_INPUTS) == TOY_INPUTS_SHA256
    assert file_sha256(ABSOLUTE_LOSS) == ABSOLUTE_LOSS_SHA256
    assert file_sha256(LOSS_CONFIG) == LOSS_CONFIG_SHA256

    unsigned = dict(config)
    assert unsigned.pop("config_fingerprint") == CONFIG_FINGERPRINT
    assert stable_fingerprint(unsigned) == CONFIG_FINGERPRINT
    assert binding == {
        "development_config_repo_path": (
            "protocols/IRSTD-1K/"
            "paired_endpoint_crossing_objective_v10/"
            "development_regression_config.json"
        ),
        "development_config_file_sha256": CONFIG_SHA256,
        "development_config_fingerprint": CONFIG_FINGERPRINT,
        "proposal_repo_path": (
            "protocols/IRSTD-1K/"
            "paired_endpoint_crossing_objective_v10/"
            "proposal_receipt.json"
        ),
        "proposal_file_sha256": PROPOSAL_SHA256,
        "proposal_fingerprint": PROPOSAL_FINGERPRINT,
        "source_file_sha256": {
            "decoder": FROZEN_DECODER_SHA256,
            "peco_loss": PECO_LOSS_SHA256,
            "predecessor_loss": PARENT_LOSS_SHA256,
            "training_step": FROZEN_STEP_SHA256,
            "toy_inputs": TOY_INPUTS_SHA256,
            "absolute_loss": ABSOLUTE_LOSS_SHA256,
            "loss_config": LOSS_CONFIG_SHA256,
        },
    }


def test_config_reuses_exact_six_cases_and_labels_evidence_scope() -> None:
    config = _load(CONFIG)

    assert config["method_id"] == "peco_v10"
    assert config["stage_id"] == "dataset_free_development_regression"
    assert config["status"] == (
        "FROZEN_BEFORE_DEVELOPMENT_REGRESSION_EXECUTION"
    )
    assert config["evidentiary_status"] == (
        "candidate_selection_development_regression_"
        "not_independent_confirmation"
    )
    assert config["cases"] == [
        {
            "family_id": family_id,
            "case_id": case_id,
            "clean_pixels": [list(pixel) for pixel in clean_pixels],
        }
        for family_id, case_id, clean_pixels in CONSERVATIVE_TOY_CASES
    ]
    assert config["optimization"]["updates_per_case"] == 320
    assert config["optimization"]["automatic_retry_allowed"] is False
    assert config["decision_rule"]["pass_decision"] == (
        "PECO_V10_DEVELOPMENT_REGRESSION_PASS"
    )
    assert config["decision_rule"]["pass_scope"] == (
        "implementation_regression_only_not_independent_confirmation"
    )
    assert all(
        value is False
        for value in config["execution_boundary"].values()
    )


def test_objective_contract_audit_covers_formula_and_saturations() -> None:
    audit = dev._objective_contract_audit()

    assert audit["all_pass"] is True
    assert all(audit["checks"].values())
    assert audit["formula_max_abs_error"] <= 1.0e-15
    assert all(value > 0.0 for value in audit["plus_gradient"])
    assert all(value < 0.0 for value in audit["minus_gradient"])


def test_mass_conservation_uses_frozen_v8_scale_aware_roundoff_rule() -> None:
    budget = torch.tensor([[[[128.0]]]], dtype=torch.float32)
    allocated = torch.tensor(
        [[[[64.0]], [[64.00001525878906]]]],
        dtype=torch.float32,
    )

    absolute, relative = dev._mass_conservation_errors(
        allocated,
        budget,
    )

    assert absolute > 1.0e-5
    assert relative <= 1.0e-6


def test_evaluator_is_local_additive_and_defaults_to_frozen_320_steps() -> None:
    modules: set[str] = set()
    tree = ast.parse(EVALUATOR.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)

    assert not any(
        module == "tests" or module.startswith("tests.")
        for module in modules
    )
    assert not any(
        "evaluate_conservative_factorized_toy_gate" in module
        for module in modules
    )
    assert not any("stage_a" in module for module in modules)
    assert not any("datasets" in module for module in modules)
    assert (
        "cure_lite.paired_endpoint_crossing_losses" in modules
    )
    assert "cure_lite.train.paired_outcome_step" in modules
    assert inspect.signature(dev._case).parameters[
        "updates"
    ].default == 320


def test_one_update_smoke_preserves_2b_gradients_detach_and_operator() -> None:
    family_id, case_id, clean_pixels = CONSERVATIVE_TOY_CASES[0]
    result = dev._case(
        family_id,
        case_id,
        clean_pixels,
        updates=1,
    )

    assert result["updates_executed"] == 1
    assert result["endpoint_gradient"] == {
        "plus_finite_nonzero": True,
        "minus_finite_nonzero": True,
    }
    assert result["gradient_contract"]["parameter_tensors"] == 6
    assert result["gradient_contract"]["parameters"] == 2593
    assert result["gradient_contract"]["updates_checked"] == 1
    assert result["gradient_contract"]["failure_count"] == 0
    assert result["gradient_contract"]["minimum_l2_norm"] > 0.0
    assert result["feature_detach_contract"] == {
        "pair_feature_requires_grad": True,
        "pair_feature_gradient_is_none": True,
        "factual_feature_gradients_are_none": True,
        "passed": True,
    }
    assert result["forward_contract"] == {
        "initial_paired_call_batch_sizes": [4],
        "paired_batch_size": 2,
        "endpoint_state_count": 4,
        "uses_one_2B_endpoint_forward": True,
        "training_step_decoder_calls": 3,
        "training_step_decoder_states": 12,
        "training_call_count": 3,
        "expected_training_call_count": 3,
        "per_update_batch_sizes_expected": [4, 4, 4],
        "first_update_batch_sizes": [4, 4, 4],
        "last_update_batch_sizes": [4, 4, 4],
        "all_updates_exact_three_4_state_calls": True,
    }
    assert result["initial_operator_audit"]["all_pass"] is True
    assert result["operator_audit"]["all_pass"] is True
    assert result["optimizer_contract"] == {
        "class": "torch.optim.Adam",
        "lr": 0.004,
        "betas": [0.9, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": False,
    }


def test_decision_aggregation_never_claims_independent_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_case(
        family_id: str,
        case_id: str,
        clean_pixels: tuple[tuple[int, int], ...],
    ) -> dict[str, object]:
        return {
            "family_id": family_id,
            "case_id": case_id,
            "clean_pixels": [list(pixel) for pixel in clean_pixels],
            "optimizer_contract": {
                "class": "torch.optim.Adam",
                "lr": 0.004,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
                "maximize": False,
                "foreach": None,
                "capturable": False,
                "differentiable": False,
                "fused": None,
                "decoupled_weight_decay": False,
            },
            "all_pass": True,
        }

    monkeypatch.setattr(dev, "_case", fake_case)
    result = dev.evaluate()

    assert result["decision"] == (
        "PECO_V10_DEVELOPMENT_REGRESSION_PASS"
    )
    assert result["passed_case_count"] == 6
    assert result["passed_family_count"] == 2
    assert result["evidentiary_status"] == (
        "candidate_selection_development_regression_"
        "not_independent_confirmation"
    )
    assert result["interpretation"] == (
        "candidate_selection_development_regression_"
        "not_independent_confirmation_or_detection_performance"
    )
    assert all(
        value is False
        for value in result["execution_boundary"].values()
    )
    unsigned = dict(result)
    observed = unsigned.pop("result_fingerprint")
    assert stable_fingerprint(unsigned) == observed


def test_result_writer_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "development-regression.json"
    dev._write_result(output, {"sentinel": True})
    with pytest.raises(FileExistsError):
        dev._write_result(output, {"sentinel": False})
    assert _load(output) == {"sentinel": True}


def test_signed_implementation_closure_binds_evaluator_and_transitive_sources(
) -> None:
    protocol_binding = dev._load_protocol_binding()
    binding = dev._load_implementation_closure(protocol_binding)
    closure = _load(CLOSURE)
    unsigned = dict(closure)
    fingerprint = unsigned.pop("receipt_fingerprint")

    assert stable_fingerprint(unsigned) == fingerprint
    assert binding == {
        "repo_path": (
            "protocols/IRSTD-1K/"
            "paired_endpoint_crossing_objective_v10/"
            "development_regression_r2_implementation_closure.json"
        ),
        "file_sha256": file_sha256(CLOSURE),
        "receipt_fingerprint": fingerprint,
        "r1_invalidation": {
            "repo_path": (
                "protocols/IRSTD-1K/"
                "paired_endpoint_crossing_objective_v10/"
                "development_regression_r1_verifier_"
                "invalidation_receipt.json"
            ),
            "file_sha256": file_sha256(R1_INVALIDATION),
            "receipt_fingerprint": _load(R1_INVALIDATION)[
                "receipt_fingerprint"
            ],
        },
    }
    sources = {
        row["repo_path"]: row["file_sha256"]
        for row in closure["source_bindings"]
    }
    assert sources[
        "tools/evaluate_paired_endpoint_crossing_development_regression.py"
    ] == file_sha256(EVALUATOR)
    assert sources[
        "tests_v10/test_paired_endpoint_crossing_development_regression.py"
    ] == file_sha256(Path(__file__))
    assert closure["semantic_scope"][
        "component_null_complete_pair_loss"
    ] == "unchanged_plus_anchor_plus_zero_transition"
    assert closure["semantic_scope"]["crossing_boundary"] == (
        "training_residual_logit_zero_not_frozen_calibration_threshold"
    )
    assert closure["semantic_scope"]["correction_scope"] == (
        "restore_frozen_v8_relative_roundoff_audit_only"
    )


def test_implementation_closure_rejects_any_bound_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_binding = dev._load_protocol_binding()
    original = dev.file_sha256
    target = ROOT / "cure_lite" / "conservative_factorized_decoder.py"

    def changed(path: Path) -> str:
        if Path(path) == target:
            return "0" * 64
        return original(path)

    monkeypatch.setattr(dev, "file_sha256", changed)
    with pytest.raises(
        RuntimeError,
        match="closure source differs",
    ):
        dev._load_implementation_closure(protocol_binding)


def test_main_fails_before_evaluation_when_output_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "already-exists.json"
    output.write_text("{}\n", encoding="utf-8")
    called = False

    def forbidden_evaluate() -> dict[str, object]:
        nonlocal called
        called = True
        raise AssertionError("evaluate must not run")

    monkeypatch.setattr(dev, "evaluate", forbidden_evaluate)
    with pytest.raises(FileExistsError, match="output already exists"):
        dev.main(["--output", str(output)])
    assert called is False
