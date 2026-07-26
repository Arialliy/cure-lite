#!/usr/bin/env python3
"""Eight-step dataset-free single-process dry run for CC-SEA v8."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import torch
from torch import Tensor

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import (  # noqa: E402
    canonical_json,
    file_sha256,
    stable_fingerprint,
)
from cure_lite.config import LossConfig  # noqa: E402
from cure_lite.conservative_factorized_config import (  # noqa: E402
    ConservativeFactorizedDecoderConfig,
)
from cure_lite.conservative_factorized_decoder import (  # noqa: E402
    CURELiteConservativeFactorizedDecoder,
    coverage_conserving_phase_evidence,
)
from cure_lite.experiment.conservative_toy_inputs import (  # noqa: E402
    CONSERVATIVE_TOY_CASES,
    build_conservative_toy_case,
)
from cure_lite.factorized_config import FactorizedDecoderConfig  # noqa: E402
from cure_lite.factorized_decoder import (  # noqa: E402
    CURELiteFactorizedDecoder,
)
from cure_lite.losses import CURELiteLoss  # noqa: E402
from cure_lite.paired_outcome_losses import (  # noqa: E402
    OutcomeCompleteTransitionLoss,
)
from cure_lite.train.paired_outcome_step import (  # noqa: E402
    outcome_complete_train_step,
)
from cure_lite.train.paired_step import (  # noqa: E402
    _paired_endpoint_logits,
)


SCHEMA_VERSION = "cure-lite-cc-sea-v8-bounded-dry-run-result-v3"
METHOD_ID = "cc_sea_v8"
FROZEN_SEED = 7817
FROZEN_UPDATES = 8
FROZEN_LEARNING_RATE = 0.001
UPDATE_CASE_INDICES = (0, 1, 2, 3, 4, 5, 0, 3)
_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "coverage_conserving_subpixel_evidence_allocation_v8"
)
_CONFIG_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "coverage_conserving_subpixel_evidence_allocation_v8/"
    "bounded_dry_run_config_v3.json"
)
_PROPOSAL_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "coverage_conserving_subpixel_evidence_allocation_v8/"
    "bounded_dry_run_proposal_receipt_v3.json"
)
_TOY_CLOSURE_REPO_PATH = (
    "protocols/IRSTD-1K/"
    "coverage_conserving_subpixel_evidence_allocation_v8/"
    "toy_gate_closure_receipt.json"
)
_CONFIG = _ROOT / _CONFIG_REPO_PATH
_DISCLOSED_PACKAGE_INITIALIZATION_MODULES = (
    "cure_lite.cache.base_cache",
    "cure_lite.cache.state_cache",
    "cure_lite.data",
    "cure_lite.experiment.cache_pipeline",
    "cure_lite.experiment.training_pipeline",
)


def _regular_bound_file(repo_path: str, *, name: str) -> Path:
    relative = Path(repo_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"{name} must be a repository-relative path")
    path = _ROOT / relative
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{name} must be a regular non-symlink file")
    resolved_root = _ROOT.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if resolved.parent != resolved_root and resolved_root not in resolved.parents:
        raise RuntimeError(f"{name} escapes the repository root")
    return path


def _object(path: Path, *, name: str) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _verified_fingerprint(
    value: dict[str, object],
    *,
    field: str,
    name: str,
) -> str:
    unsigned = dict(value)
    observed = unsigned.pop(field, None)
    if not isinstance(observed, str):
        raise TypeError(f"{name}.{field} must be a string")
    if stable_fingerprint(unsigned) != observed:
        raise RuntimeError(f"{name} fingerprint differs")
    return observed


def _load_protocol_binding() -> dict[str, object]:
    if _CONFIG != _regular_bound_file(
        _CONFIG_REPO_PATH,
        name="v8 dry-run config",
    ):
        raise RuntimeError("v8 dry-run config path differs")
    config = _object(_CONFIG, name="v8 dry-run config")
    config_fingerprint = _verified_fingerprint(
        config,
        field="config_fingerprint",
        name="v8 dry-run config",
    )
    if config.get("schema_version") != (
        "cure-lite-cc-sea-v8-bounded-dry-run-config-v3"
    ):
        raise RuntimeError("v8 dry-run config schema differs")
    if config.get("method_id") != METHOD_ID:
        raise RuntimeError("v8 dry-run method differs")
    expected_cases = [
        {
            "update": update,
            "family_id": CONSERVATIVE_TOY_CASES[index][0],
            "case_id": CONSERVATIVE_TOY_CASES[index][1],
            "clean_pixels": [
                list(pixel) for pixel in CONSERVATIVE_TOY_CASES[index][2]
            ],
        }
        for update, index in enumerate(UPDATE_CASE_INDICES)
    ]
    if config.get("update_cases") != expected_cases:
        raise RuntimeError("v8 dry-run update cases differ")
    if config.get("decoder") != {
        "feature_channels": 8,
        "feature_stride": 4,
        "expected_parameter_tensors": 6,
        "expected_parameter_count": 2593,
        "reference_C64_stride4_parameter_count": 4385,
    }:
        raise RuntimeError("v8 dry-run decoder contract differs")
    if config.get("runtime") != {
        "device": "cpu",
        "deterministic_algorithms": True,
        "torch_threads": 1,
        "seed": FROZEN_SEED,
        "optimizer": "adam",
        "learning_rate": FROZEN_LEARNING_RATE,
        "weight_decay": 0.0,
        "epochs": 2,
        "steps_per_epoch": 4,
        "optimizer_updates": FROZEN_UPDATES,
        "decoder_calls_per_update": 3,
        "decoder_states_per_update": 12,
        "paired_endpoint_policy": "one_cat_2B_forward",
        "fixture_policy": "six_v8_native_toy_cases_fixed_cycle",
    }:
        raise RuntimeError("v8 dry-run runtime contract differs")
    boundary = config.get("execution_boundary")
    if boundary != {
        "D_R_payload_access_allowed": False,
        "D_V_access_allowed": False,
        "D_T_access_allowed": False,
        "dataset_or_cache_payload_access_allowed": False,
        "real_loader_call_allowed": False,
        "package_initialization_module_import_allowed": True,
        "detection_performance_allowed": False,
        "real_D_R_bounded_execution_authorized": False,
        "formal_800_authorized": False,
        "full_CURE_authorized": False,
        "cross_detector_authorized": False,
    }:
        raise RuntimeError("v8 dry-run execution boundary differs")

    proposal_binding = config.get("proposal_binding")
    if not isinstance(proposal_binding, dict):
        raise TypeError("v8 dry proposal binding must be an object")
    if proposal_binding.get("repo_path") != _PROPOSAL_REPO_PATH:
        raise RuntimeError("v8 dry proposal repo path differs")
    proposal_path = _regular_bound_file(
        _PROPOSAL_REPO_PATH,
        name="v8 dry proposal",
    )
    if file_sha256(proposal_path) != proposal_binding.get("file_sha256"):
        raise RuntimeError("v8 dry proposal file hash differs")
    proposal = _object(proposal_path, name="v8 dry proposal")
    proposal_fingerprint = _verified_fingerprint(
        proposal,
        field="proposal_fingerprint",
        name="v8 dry proposal",
    )
    if proposal_fingerprint != proposal_binding.get(
        "proposal_fingerprint"
    ):
        raise RuntimeError("v8 dry proposal fingerprint differs")
    for field in (
        "single_process_required_evidence",
        "closure_required_evidence",
    ):
        if proposal.get(field) != config.get(field):
            raise RuntimeError(f"v8 dry {field} differs")

    closure_binding = config.get("toy_closure_binding")
    if closure_binding != proposal.get("toy_closure_binding"):
        raise RuntimeError("v8 dry toy closure bindings differ")
    if not isinstance(closure_binding, dict):
        raise TypeError("v8 dry toy closure binding must be an object")
    if closure_binding.get("repo_path") != _TOY_CLOSURE_REPO_PATH:
        raise RuntimeError("v8 toy closure repo path differs")
    closure_path = _regular_bound_file(
        _TOY_CLOSURE_REPO_PATH,
        name="v8 toy closure",
    )
    if file_sha256(closure_path) != closure_binding.get("file_sha256"):
        raise RuntimeError("v8 toy closure file hash differs")
    closure = _object(closure_path, name="v8 toy closure")
    closure_fingerprint = _verified_fingerprint(
        closure,
        field="receipt_fingerprint",
        name="v8 toy closure",
    )
    if (
        closure_fingerprint
        != closure_binding.get("receipt_fingerprint")
        or closure.get("decision")
        != "CC_SEA_V8_TOY_GATE_PASS_AND_DRY_RUN_CODE_AUTHORIZED"
    ):
        raise RuntimeError("v8 toy closure decision/binding differs")
    bound_file_count = 0
    for binding_name in ("software_bindings", "test_bindings"):
        bindings = closure.get(binding_name)
        if not isinstance(bindings, dict) or not bindings:
            raise TypeError(f"v8 toy closure {binding_name} must be an object")
        for repo_path, expected_sha256 in bindings.items():
            if not isinstance(repo_path, str) or not isinstance(
                expected_sha256,
                str,
            ):
                raise TypeError(f"v8 toy closure {binding_name} is malformed")
            bound_path = _regular_bound_file(
                repo_path,
                name=f"v8 toy closure bound file {repo_path}",
            )
            if file_sha256(bound_path) != expected_sha256:
                raise RuntimeError(
                    f"v8 toy closure bound file differs: {repo_path}"
                )
            bound_file_count += 1
    return {
        "config_repo_path": str(_CONFIG.relative_to(_ROOT)),
        "config_file_sha256": file_sha256(_CONFIG),
        "config_fingerprint": config_fingerprint,
        "proposal_repo_path": str(proposal_path.relative_to(_ROOT)),
        "proposal_file_sha256": file_sha256(proposal_path),
        "proposal_fingerprint": proposal_fingerprint,
        "toy_closure_repo_path": str(closure_path.relative_to(_ROOT)),
        "toy_closure_file_sha256": file_sha256(closure_path),
        "toy_closure_fingerprint": closure_fingerprint,
        "toy_closure_bound_files_verified": bound_file_count,
    }


def _tensor_digest(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _training_example_fingerprint(
    *,
    family_id: str,
    case_id: str,
    clean_pixels: tuple[tuple[int, int], ...],
    outcome: object,
    factual: dict[str, object],
) -> str:
    pair_batch = outcome.pair_batch
    miss = factual["factual_miss"]
    no_miss = factual["factual_no_miss"]
    return stable_fingerprint(
        {
            "family_id": family_id,
            "case_id": case_id,
            "clean_pixels": [list(pixel) for pixel in clean_pixels],
            "factual_miss": {
                "feature": _tensor_digest(miss.feature),
                "occupancy": _tensor_digest(miss.occupancy),
                "target": _tensor_digest(miss.target),
                "valid_mask": _tensor_digest(miss.valid_mask),
            },
            "factual_no_miss": {
                "feature": _tensor_digest(no_miss.feature),
                "occupancy": _tensor_digest(no_miss.occupancy),
                "target": _tensor_digest(no_miss.target),
                "valid_mask": _tensor_digest(no_miss.valid_mask),
            },
            "pair": {
                "feature": _tensor_digest(pair_batch.feature),
                "occupancy_plus": _tensor_digest(
                    pair_batch.occupancy_plus
                ),
                "occupancy_minus": _tensor_digest(
                    pair_batch.occupancy_minus
                ),
                "label_increment": _tensor_digest(
                    pair_batch.label_increment
                ),
                "image_valid_mask": _tensor_digest(
                    pair_batch.image_valid_mask
                ),
                "pair_ids": list(pair_batch.pair_ids),
                "sample_ids": list(pair_batch.sample_ids),
                "group_ids": list(pair_batch.group_ids),
                "pair_kinds": list(pair_batch.pair_kinds),
                "projection_visible": list(
                    pair_batch.projection_visible
                ),
            },
            "outcome": {
                "completion_plus": _tensor_digest(
                    outcome.completion_plus
                ),
                "completion_minus": _tensor_digest(
                    outcome.completion_minus
                ),
                "gt_union": _tensor_digest(outcome.gt_union),
                "intervention_footprint": _tensor_digest(
                    outcome.intervention_footprint
                ),
            },
        }
    )


def _state_fingerprint(decoder: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in decoder.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_digest(value).encode("ascii"))
    return digest.hexdigest()


def _topology_audit() -> dict[str, object]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8321)
        v4 = CURELiteFactorizedDecoder(FactorizedDecoderConfig(64, 4))
        torch.manual_seed(8321)
        v8 = CURELiteConservativeFactorizedDecoder(
            ConservativeFactorizedDecoderConfig(64, 4)
        )
    checks = {
        "state_keys_exact": tuple(v4.state_dict()) == tuple(v8.state_dict()),
        "initial_state_values_exact": all(
            torch.equal(v4.state_dict()[name], value)
            for name, value in v8.state_dict().items()
        ),
        "child_module_types_exact": (
            tuple(type(module) for module in tuple(v4.modules())[1:])
            == tuple(type(module) for module in tuple(v8.modules())[1:])
        ),
        "reference_parameter_tensors_exact": len(tuple(v8.parameters())) == 6,
        "reference_parameter_count_exact": (
            sum(parameter.numel() for parameter in v8.parameters()) == 4385
        ),
    }
    return {"checks": checks, "all_pass": all(checks.values())}


def _coordinate_gradient_audit() -> dict[str, object]:
    raw_budget = torch.tensor(
        [[[[1.0]], [[0.5]], [[0.0]], [[-0.5]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    burden = torch.zeros((1, 1, 1, 1), dtype=torch.float64)
    budget_fields = coverage_conserving_phase_evidence(raw_budget, burden)
    budget_fields[2].sum().backward()
    budget_gradient = raw_budget.grad.detach().clone()

    raw_selection = raw_budget.detach().clone().requires_grad_(True)
    selection_fields = coverage_conserving_phase_evidence(
        raw_selection,
        burden,
    )
    selection_fields[4][:, 0].sum().backward()
    selection_gradient = raw_selection.grad.detach().clone()
    selection_common = selection_gradient.mean(dim=1, keepdim=True)
    selection_contrast = selection_gradient - selection_common
    checks = {
        "budget_gradient_finite_nonzero": (
            bool(torch.isfinite(budget_gradient).all())
            and float(budget_gradient.norm()) > 0.0
        ),
        "budget_gradient_is_common_mode": float(
            (budget_gradient - budget_gradient.mean(dim=1, keepdim=True))
            .abs()
            .max()
        )
        <= 1.0e-12,
        "selection_gradient_finite_nonzero": (
            bool(torch.isfinite(selection_gradient).all())
            and float(selection_gradient.norm()) > 0.0
        ),
        "selection_has_nonzero_contrast_gradient": float(
            selection_contrast.norm()
        )
        > 0.0,
    }
    return {
        "checks": checks,
        "budget_gradient_l2": float(budget_gradient.norm()),
        "selection_gradient_l2": float(selection_gradient.norm()),
        "selection_contrast_gradient_l2": float(selection_contrast.norm()),
        "all_pass": all(checks.values()),
    }


def _paired_equivalence_audit(
    decoder: CURELiteConservativeFactorizedDecoder,
) -> dict[str, object]:
    family, _, pixels = CONSERVATIVE_TOY_CASES[0]
    outcome, _ = build_conservative_toy_case(family, pixels)
    batch = outcome.pair_batch
    decoder.eval()
    with torch.no_grad():
        batched_plus, batched_minus = _paired_endpoint_logits(
            decoder,
            feature=batch.feature,
            occupancy_plus=batch.occupancy_plus,
            occupancy_minus=batch.occupancy_minus,
        )
        separate_plus = decoder(batch.feature, batch.occupancy_plus)
        separate_minus = decoder(batch.feature, batch.occupancy_minus)
    checks = {
        "plus_bit_exact": torch.equal(batched_plus, separate_plus),
        "minus_bit_exact": torch.equal(batched_minus, separate_minus),
    }
    return {
        "checks": checks,
        "plus_max_abs_error": float(
            (batched_plus - separate_plus).abs().max()
        ),
        "minus_max_abs_error": float(
            (batched_minus - separate_minus).abs().max()
        ),
        "all_pass": all(checks.values()),
    }


def _dual_endpoint_gradient_audit(
    decoder: CURELiteConservativeFactorizedDecoder,
    criterion: OutcomeCompleteTransitionLoss,
) -> dict[str, object]:
    """Require both endpoint scores of both pair roles to affect the loss."""

    family, _, pixels = CONSERVATIVE_TOY_CASES[0]
    outcome, _ = build_conservative_toy_case(family, pixels)
    batch = outcome.pair_batch
    decoder.eval()
    with torch.no_grad():
        logits_plus, logits_minus = _paired_endpoint_logits(
            decoder,
            feature=batch.feature,
            occupancy_plus=batch.occupancy_plus,
            occupancy_minus=batch.occupancy_minus,
        )
    plus_leaf = logits_plus.detach().clone().requires_grad_(True)
    minus_leaf = logits_minus.detach().clone().requires_grad_(True)
    result = criterion(
        plus_leaf,
        minus_leaf,
        outcome.completion_plus,
        batch.occupancy_plus,
        outcome.gt_union,
        batch.label_increment,
        batch.image_valid_mask,
        outcome.intervention_footprint,
    )
    result["total"].backward()
    records: list[dict[str, object]] = []
    for index, pair_kind in enumerate(batch.pair_kinds):
        plus_gradient = plus_leaf.grad[index]
        minus_gradient = minus_leaf.grad[index]
        records.append(
            {
                "pair_index": index,
                "pair_kind": pair_kind,
                "plus_gradient_finite": bool(
                    torch.isfinite(plus_gradient).all()
                ),
                "plus_gradient_nonzero_count": int(
                    torch.count_nonzero(plus_gradient)
                ),
                "plus_gradient_l2": float(
                    plus_gradient.double().norm()
                ),
                "minus_gradient_finite": bool(
                    torch.isfinite(minus_gradient).all()
                ),
                "minus_gradient_nonzero_count": int(
                    torch.count_nonzero(minus_gradient)
                ),
                "minus_gradient_l2": float(
                    minus_gradient.double().norm()
                ),
            }
        )
    checks = {
        "clean_and_component_roles_present": tuple(
            record["pair_kind"] for record in records
        )
        == ("clean_positive", "component_null"),
        "both_endpoints_finite_nonzero_per_pair": all(
            record["plus_gradient_finite"] is True
            and record["plus_gradient_nonzero_count"] > 0
            and record["plus_gradient_l2"] > 0.0
            and record["minus_gradient_finite"] is True
            and record["minus_gradient_nonzero_count"] > 0
            and record["minus_gradient_l2"] > 0.0
            for record in records
        ),
    }
    return {
        "checks": checks,
        "records": records,
        "all_pass": all(checks.values()),
    }


def _state_equation_audit(
    decoder: CURELiteConservativeFactorizedDecoder,
) -> dict[str, object]:
    """Use a controlled positive-budget 5x5 operator locality probe."""

    feature = torch.linspace(
        -1.0,
        1.0,
        steps=8 * 5 * 5,
        dtype=torch.float32,
    ).reshape(1, 8, 5, 5)
    occupancy_plus = torch.zeros((1, 1, 20, 20), dtype=torch.bool)
    occupancy_plus[0, 0, 10, 10] = True
    occupancy_minus = torch.zeros_like(occupancy_plus)
    phase_channels = decoder.feature_stride**2
    phase_contrast = torch.linspace(
        -0.3,
        0.3,
        steps=phase_channels,
        dtype=torch.float32,
    )
    phase_contrast = phase_contrast - phase_contrast.mean()
    raw_phase_evidence = (
        2.0
        + phase_contrast.reshape(1, phase_channels, 1, 1)
    ).expand(1, phase_channels, 5, 5).contiguous()
    decoder.eval()
    with torch.no_grad():
        identity = decoder(feature, occupancy_plus)
        identity_again = decoder(feature, occupancy_plus.clone())
        actual_plus = decoder(feature, occupancy_plus)
        actual_minus = decoder(feature, occupancy_minus)
        _, plus_count, plus_burden = decoder.native_burden_field(
            occupancy_plus,
            feature_size=(5, 5),
        )
        _, minus_count, minus_burden = decoder.native_burden_field(
            occupancy_minus,
            feature_size=(5, 5),
        )
        plus_fields = coverage_conserving_phase_evidence(
            raw_phase_evidence,
            plus_burden,
        )
        minus_fields = coverage_conserving_phase_evidence(
            raw_phase_evidence,
            minus_burden,
        )
        count_release = plus_count - minus_count
        support = torch.nn.functional.pixel_shuffle(
            (count_release > 0.0)
            .expand(-1, phase_channels, -1, -1)
            .to(torch.float32),
            decoder.feature_stride,
        ).to(torch.bool)
        plus_evidence = torch.nn.functional.pixel_shuffle(
            plus_fields[4],
            decoder.feature_stride,
        )
        minus_evidence = torch.nn.functional.pixel_shuffle(
            minus_fields[4],
            decoder.feature_stride,
        )
        delta = minus_evidence - plus_evidence
        actual_delta = actual_minus - actual_plus
        support_pixels = int(torch.count_nonzero(support))
        outside_pixels = int(torch.count_nonzero(~support))
        conservation_error = max(
            float(
                (
                    fields[4].sum(dim=1, keepdim=True)
                    - fields[2]
                )
                .abs()
                .div(fields[2].abs().clamp_min(1.0))
                .max()
            )
            for fields in (plus_fields, minus_fields)
        )
    checks = {
        "identity_exact": torch.equal(identity, identity_again),
        "support_nonempty": support_pixels > 0,
        "outside_support_nonempty": outside_pixels > 0,
        "allocation_occupancy_invariant": torch.equal(
            plus_fields[3],
            minus_fields[3],
        ),
        "budget_deletion_monotone": bool(
            torch.all(minus_fields[2] >= plus_fields[2])
        ),
        "allocated_deletion_monotone": bool(
            torch.all(minus_fields[4] >= plus_fields[4])
        ),
        "outside_count_support_exact": torch.equal(
            delta[~support],
            torch.zeros_like(delta[~support]),
        ),
        "inside_count_support_response_nonzero": (
            support_pixels > 0
            and int(torch.count_nonzero(delta[support])) > 0
        ),
        "actual_decoder_outside_count_support_exact": torch.equal(
            actual_delta[~support],
            torch.zeros_like(actual_delta[~support]),
        ),
        "controlled_common_mode_exact": torch.equal(
            minus_fields[0],
            torch.full_like(minus_fields[0], 2.0),
        ),
        "mass_conservation_relative_error": conservation_error <= 1.0e-6,
    }
    return {
        "checks": checks,
        "probe_kind": "controlled_positive_budget_operator_probe",
        "controlled_common_mode": 2.0,
        "controlled_phase_contrast_min": float(phase_contrast.min()),
        "controlled_phase_contrast_max": float(phase_contrast.max()),
        "feature_grid": [5, 5],
        "evaluation_grid": [20, 20],
        "support_pixel_count": support_pixels,
        "outside_support_pixel_count": outside_pixels,
        "support_max_abs_delta": float(delta[support].abs().max()),
        "mass_conservation_max_relative_error": conservation_error,
        "outside_count_support_max_abs_delta": float(
            delta[~support].abs().max()
        ),
        "actual_decoder_outside_count_support_max_abs_delta": float(
            actual_delta[~support].abs().max()
        ),
        "all_pass": all(checks.values()),
    }


def evaluate() -> dict[str, object]:
    """Execute the exact deterministic eight-update in-memory dry run."""

    protocol_binding = _load_protocol_binding()
    topology = _topology_audit()
    coordinate_gradient = _coordinate_gradient_audit()
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    flags_restored = False
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(FROZEN_SEED)
            decoder = CURELiteConservativeFactorizedDecoder(
                feature_channels=8,
                feature_stride=4,
            )
            state_equation = _state_equation_audit(decoder)
            initial_decoder_fingerprint = _state_fingerprint(decoder)
            optimizer = torch.optim.Adam(
                decoder.parameters(),
                lr=FROZEN_LEARNING_RATE,
                weight_decay=0.0,
            )
            absolute = CURELiteLoss()
            criterion = OutcomeCompleteTransitionLoss(LossConfig())
            named_parameters = tuple(decoder.named_parameters())
            if len(named_parameters) != 6:
                raise AssertionError("v8 dry parameter tensor count differs")
            if sum(value.numel() for _, value in named_parameters) != 2593:
                raise AssertionError("v8 dry parameter count differs")

            trace: list[dict[str, object]] = []
            total_calls = 0
            total_states = 0
            observed_device_types = {
                parameter.device.type for parameter in decoder.parameters()
            }
            for update, case_index in enumerate(UPDATE_CASE_INDICES):
                family, case_id, pixels = CONSERVATIVE_TOY_CASES[case_index]
                outcome, factual = build_conservative_toy_case(family, pixels)
                training_example_fingerprint = _training_example_fingerprint(
                    family_id=family,
                    case_id=case_id,
                    clean_pixels=pixels,
                    outcome=outcome,
                    factual=factual,
                )
                source_features = (
                    factual["factual_miss"].feature,
                    factual["factual_no_miss"].feature,
                    outcome.pair_batch.feature,
                )
                for source_feature in source_features:
                    source_feature.requires_grad_(True)
                    observed_device_types.add(source_feature.device.type)
                observed_calls: list[tuple[Tensor, Tensor]] = []

                def observe(
                    _module: object,
                    args: tuple[object, ...],
                ) -> None:
                    observed_device_types.add(args[0].device.type)
                    observed_device_types.add(args[1].device.type)
                    observed_calls.append(
                        (
                            args[0].detach().cpu().clone(),
                            args[1].detach().cpu().clone(),
                        )
                    )

                before = {
                    name: parameter.detach().clone()
                    for name, parameter in named_parameters
                }
                handle = decoder.register_forward_pre_hook(observe)
                try:
                    logs = outcome_complete_train_step(
                        decoder,
                        absolute,
                        criterion,
                        optimizer,
                        factual,
                        outcome,
                    )
                finally:
                    handle.remove()

                expected_pair_feature = torch.cat(
                    (
                        outcome.pair_batch.feature,
                        outcome.pair_batch.feature,
                    ),
                    dim=0,
                )
                expected_pair_occupancy = torch.cat(
                    (
                        outcome.pair_batch.occupancy_plus,
                        outcome.pair_batch.occupancy_minus,
                    ),
                    dim=0,
                )
                call_checks = {
                    "exactly_three_calls": len(observed_calls) == 3,
                    "factual_miss_input_exact": (
                        len(observed_calls) == 3
                        and torch.equal(
                            observed_calls[0][0],
                            factual["factual_miss"].feature,
                        )
                        and torch.equal(
                            observed_calls[0][1],
                            factual["factual_miss"].occupancy,
                        )
                    ),
                    "factual_no_miss_input_exact": (
                        len(observed_calls) == 3
                        and torch.equal(
                            observed_calls[1][0],
                            factual["factual_no_miss"].feature,
                        )
                        and torch.equal(
                            observed_calls[1][1],
                            factual["factual_no_miss"].occupancy,
                        )
                    ),
                    "paired_2B_feature_exact": (
                        len(observed_calls) == 3
                        and torch.equal(
                            observed_calls[2][0],
                            expected_pair_feature,
                        )
                    ),
                    "paired_2B_occupancy_exact": (
                        len(observed_calls) == 3
                        and torch.equal(
                            observed_calls[2][1],
                            expected_pair_occupancy,
                        )
                    ),
                }
                execution_checks = {
                    "exact_three_calls_in_log": (
                        logs.get("decoder_forward_calls_per_update") == 3
                    ),
                    "exact_twelve_states_in_log": (
                        logs.get("decoder_states_per_update") == 12
                    ),
                    "exact_one_backward_in_log": (
                        logs.get("backward_calls") == 1
                    ),
                    "exact_one_optimizer_step_in_log": (
                        logs.get("optimizer_steps") == 1
                    ),
                    "source_features_required_grad_for_probe": all(
                        feature.requires_grad for feature in source_features
                    ),
                    "source_feature_gradients_remain_none": all(
                        feature.grad is None for feature in source_features
                    ),
                    "decoder_received_detached_features": all(
                        not feature.requires_grad
                        for feature, _ in observed_calls
                    ),
                }
                parameter_records = []
                for name, parameter in named_parameters:
                    gradient = parameter.grad
                    if gradient is None:
                        raise RuntimeError("v8 dry gradient is missing")
                    delta = parameter.detach() - before[name]
                    parameter_records.append(
                        {
                            "name": name,
                            "shape": list(parameter.shape),
                            "numel": parameter.numel(),
                            "gradient_present": True,
                            "gradient_finite": bool(
                                torch.isfinite(gradient).all()
                            ),
                            "gradient_nonzero_count": int(
                                torch.count_nonzero(gradient)
                            ),
                            "gradient_l2": float(gradient.double().norm()),
                            "gradient_max_abs": float(gradient.abs().max()),
                            "parameter_delta_nonzero_count": int(
                                torch.count_nonzero(delta)
                            ),
                            "parameter_delta_l2": float(delta.double().norm()),
                        }
                    )
                parameter_checks = {
                    "all_six_gradients_present_finite_nonzero": (
                        len(parameter_records) == 6
                        and all(
                            record["gradient_present"] is True
                            and record["gradient_finite"] is True
                            and record["gradient_nonzero_count"] > 0
                            and record["gradient_l2"] > 0.0
                            for record in parameter_records
                        )
                    ),
                    "all_six_parameters_updated": (
                        len(parameter_records) == 6
                        and all(
                            record["parameter_delta_nonzero_count"] > 0
                            and record["parameter_delta_l2"] > 0.0
                            for record in parameter_records
                        )
                    ),
                }
                total_calls += len(observed_calls)
                total_states += sum(
                    int(feature.shape[0]) for feature, _ in observed_calls
                )
                trace.append(
                    {
                        "update": update,
                        "epoch": update // 4,
                        "step": update % 4,
                        "family_id": family,
                        "case_id": case_id,
                        "clean_pixels": [list(pixel) for pixel in pixels],
                        "decoder_input_fingerprint": stable_fingerprint(
                            {
                                "factual_miss_feature": _tensor_digest(
                                    factual["factual_miss"].feature
                                ),
                                "factual_no_miss_feature": _tensor_digest(
                                    factual["factual_no_miss"].feature
                                ),
                                "pair_feature": _tensor_digest(
                                    outcome.pair_batch.feature
                                ),
                                "occupancy_plus": _tensor_digest(
                                    outcome.pair_batch.occupancy_plus
                                ),
                                "occupancy_minus": _tensor_digest(
                                    outcome.pair_batch.occupancy_minus
                                ),
                            }
                        ),
                        "training_example_fingerprint": (
                            training_example_fingerprint
                        ),
                        "call_checks": call_checks,
                        "execution_checks": execution_checks,
                        "parameter_checks": parameter_checks,
                        "parameter_records": parameter_records,
                        "losses": logs,
                    }
                )

            paired_equivalence = _paired_equivalence_audit(decoder)
            dual_endpoint_gradient = _dual_endpoint_gradient_audit(
                decoder,
                criterion,
            )
            final_decoder_fingerprint = _state_fingerprint(decoder)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)
        flags_restored = (
            torch.are_deterministic_algorithms_enabled()
            == previous_deterministic
            and torch.get_num_threads() == previous_threads
        )

    disclosed_package_modules = [
        name
        for name in _DISCLOSED_PACKAGE_INITIALIZATION_MODULES
        if name in sys.modules
    ]
    structural_checks = {
        "protocol_binding_valid": True,
        "topology_unchanged": topology["all_pass"] is True,
        "coordinate_gradients_nonzero": coordinate_gradient["all_pass"] is True,
        "exact_eight_updates": len(trace) == FROZEN_UPDATES,
        "exact_24_training_forward_calls": total_calls == 24,
        "exact_96_training_states": total_states == 96,
        "every_update_input_binding_exact": all(
            all(row["call_checks"].values()) for row in trace
        ),
        "every_update_execution_budget_and_feature_detachment": all(
            all(row["execution_checks"].values()) for row in trace
        ),
        "every_update_gradients_and_parameters_change": all(
            all(row["parameter_checks"].values()) for row in trace
        ),
        "complete_training_example_fingerprints": (
            len(
                {
                    row["training_example_fingerprint"]
                    for row in trace[:6]
                }
            )
            == 6
            and trace[0]["training_example_fingerprint"]
            == trace[6]["training_example_fingerprint"]
            and trace[3]["training_example_fingerprint"]
            == trace[7]["training_example_fingerprint"]
        ),
        "paired_2B_equivalence": paired_equivalence["all_pass"] is True,
        "dual_endpoint_gradients": (
            dual_endpoint_gradient["all_pass"] is True
        ),
        "state_equation_contract": state_equation["all_pass"] is True,
        "decoder_changed_after_training": (
            final_decoder_fingerprint != initial_decoder_fingerprint
        ),
        "deterministic_flags_restored": flags_restored,
        "cpu_only": observed_device_types == {"cpu"},
        "package_initialization_imports_disclosed": (
            disclosed_package_modules
            == list(_DISCLOSED_PACKAGE_INITIALIZATION_MODULES)
        ),
    }
    all_pass = all(structural_checks.values())
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "decision": (
            "CC_SEA_V8_DRY_RUN_SINGLE_PROCESS_PASS"
            if all_pass
            else "CC_SEA_V8_DRY_RUN_SINGLE_PROCESS_FAIL"
        ),
        "all_pass": all_pass,
        "single_process_gate_pass": all_pass,
        "closure_eligible_after_replay": all_pass,
        "closure_required_evidence_status": (
            "NOT_EVALUATED_BY_SINGLE_PROCESS"
        ),
        "protocol_binding": protocol_binding,
        "runtime": {
            "device": "cpu",
            "seed": FROZEN_SEED,
            "optimizer_updates": FROZEN_UPDATES,
            "learning_rate": FROZEN_LEARNING_RATE,
            "decoder_calls": total_calls,
            "decoder_states": total_states,
            "observed_device_types": sorted(observed_device_types),
            "base_detector_instances": 0,
            "real_loader_calls": 0,
            "dataset_or_cache_payload_accesses": 0,
            "package_initialization_loader_modules_observed": (
                disclosed_package_modules
            ),
            "package_initialization_imports_disclosed": True,
            "entrypoint_direct_import_audit": (
                "PENDING_CLOSURE_STATIC_TEST"
            ),
        },
        "topology_audit": topology,
        "coordinate_gradient_audit": coordinate_gradient,
        "paired_equivalence_audit": paired_equivalence,
        "dual_endpoint_gradient_audit": dual_endpoint_gradient,
        "state_equation_audit": state_equation,
        "state_equation_audit_phase": "pre_training_frozen_probe",
        "initial_decoder_fingerprint": initial_decoder_fingerprint,
        "final_decoder_fingerprint": final_decoder_fingerprint,
        "structural_checks": structural_checks,
        "trace": trace,
        "execution_boundary": {
            "D_R_accessed": False,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "detection_performance_evaluated": False,
            "real_D_R_bounded_execution_authorized": False,
            "formal_800_authorized": False,
            "full_CURE_authorized": False,
            "cross_detector_authorized": False,
        },
        "interpretation": "in_memory_execution_connectivity_not_performance",
        "real_D_R_bounded_code_creation_authorized": False,
        "automatic_retry_performed": False,
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


def _write_new(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate()
    _write_new(args.output, result)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "result_fingerprint": result["result_fingerprint"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0 if result["all_pass"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
