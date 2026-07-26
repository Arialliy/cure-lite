#!/usr/bin/env python3
"""Run the frozen CR-LVEC v7 bounded implementation dry-run on CPU.

This entry point never constructs or loads a dataset-backed catalog.  It
validates the signed v7 protocol chain, creates fixed in-memory paired
fixtures, executes exactly eight optimizer updates, and emits one deterministic
fingerprinted JSON result.  The result is implementation evidence only: it
cannot authorize a real D_R run or formal training.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cure_lite.cache.schema import file_sha256, stable_fingerprint  # noqa: E402
from cure_lite.config import LossConfig  # noqa: E402
from cure_lite.crossing_factorized_config import (  # noqa: E402
    CrossingFactorizedDecoderConfig,
)
from cure_lite.crossing_factorized_decoder import (  # noqa: E402
    CURELiteCrossingFactorizedDecoder,
    crossing_recoverable_evidence,
)
from cure_lite.factorized_config import FactorizedDecoderConfig  # noqa: E402
from cure_lite.factorized_decoder import (  # noqa: E402
    CURELiteFactorizedDecoder,
)
from cure_lite.losses import CURELiteLoss  # noqa: E402
from cure_lite.paired_outcome_losses import (  # noqa: E402
    OutcomeCompleteTransitionLoss,
)
from cure_lite.paired_outcome_types import (  # noqa: E402
    OutcomePairBatch,
    direct_projected_intervention_footprint,
)
from cure_lite.paired_types import PairBatch  # noqa: E402
from cure_lite.train.paired_outcome_step import (  # noqa: E402
    outcome_complete_train_step,
)
from cure_lite.train.paired_step import _paired_endpoint_logits  # noqa: E402
from cure_lite.train.step import BranchBatch  # noqa: E402


SCHEMA_VERSION = "cure-lite-cr-lvec-v7-bounded-dry-run-result-v1"
METHOD_ID = "cr_lvec_v7"
FROZEN_SEED = 7817
FROZEN_UPDATES = 8
FROZEN_LEARNING_RATE = 0.001
EXPECTED_PARAMETER_COUNT = 2593
EXPECTED_PARAMETER_TENSORS = 6
REFERENCE_PARAMETER_COUNT = 4385

EXPECTED_BOUNDED_PROPOSAL_SHA256 = (
    "65a45dc6d73d8cbf6bcb2c6b6204251f3583e0354ba3161b633f1547fbaa11dd"
)
EXPECTED_BOUNDED_PROPOSAL_FINGERPRINT = (
    "d33f710348dec255fd73790b3c97c643472d115d3098e1469409f7dd57fad896"
)
EXPECTED_BOUNDED_CONFIG_SHA256 = (
    "352c0c235134c1017b851854278255c2c678973929d3fda614389392502c4b96"
)
EXPECTED_BOUNDED_CONFIG_FINGERPRINT = (
    "9bdc7f5567065c02d37cc82f94b5bc49c589dfee271487f4cbce7dd831c45818"
)
EXPECTED_DRY_CONFIG_SHA256 = (
    "709f72bc4d17798be4fecb01f96afb1b91a9fb39f6a5da80315a71b6b501e55c"
)
EXPECTED_DRY_CONFIG_FINGERPRINT = (
    "d5421a162822ad9962b9790a10c49c4bfe8cd7844c88c4ec5e80a7ca54559e97"
)
EXPECTED_METHOD_PROPOSAL_SHA256 = (
    "fa72f4ef850f72a65003e913db1b1230d7b0b45046faf61950fb1e4ef80d3c4f"
)
EXPECTED_METHOD_PROPOSAL_FINGERPRINT = (
    "9d291e6ad9ec0869aa0ab0eaebcb219cd62678420375f56af480ba105208dbf2"
)
EXPECTED_TOY_CLOSURE_SHA256 = (
    "25c3317045533f4116b8873d892fcd2c0e866d3e991843a4c0c8e872142f0fe5"
)
EXPECTED_TOY_CLOSURE_FINGERPRINT = (
    "f95573edd8b842980d5b175b1aac8caf753f6c279342da8e29f54e165b1e255f"
)

_ROOT = Path(__file__).resolve().parents[1]
_PROTOCOL = (
    _ROOT
    / "protocols"
    / "IRSTD-1K"
    / "continuously_recoverable_log_vacancy_evidence_crossing_v7"
)
_BOUNDED_PROPOSAL = _PROTOCOL / "bounded_implementation_proposal_receipt.json"
_BOUNDED_CONFIG = _PROTOCOL / "bounded_config.json"
_DRY_CONFIG = _PROTOCOL / "bounded_dry_run_config.json"
_METHOD_PROPOSAL = _PROTOCOL / "proposal_receipt.json"
_TOY_CLOSURE = _PROTOCOL / "toy_gate_closure_receipt.json"

_TARGET_PIXEL_SETS = (
    ((1, 2),),
    ((1, 2), (2, 1)),
    ((1, 2), (2, 1), (2, 2)),
)


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{name} must be a regular file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return payload


def _verify_signed_payload(
    payload: Mapping[str, Any],
    *,
    field: str,
    expected: str,
    name: str,
) -> None:
    value = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if (
        value != expected
        or not isinstance(value, str)
        or stable_fingerprint(unsigned) != value
    ):
        raise RuntimeError(f"{name} fingerprint is inconsistent")


def _require_binding(
    binding: Mapping[str, Any],
    *,
    path: Path,
    expected_sha256: str,
    fingerprint_field: str,
    expected_fingerprint: str,
    name: str,
) -> None:
    expected_repo_path = str(path.relative_to(_ROOT))
    if (
        not isinstance(binding, Mapping)
        or binding.get("repo_path") != expected_repo_path
        or binding.get("file_sha256") != expected_sha256
        or binding.get(fingerprint_field) != expected_fingerprint
        or file_sha256(path) != expected_sha256
    ):
        raise RuntimeError(f"{name} binding is inconsistent")


def _validate_dry_config(config: Mapping[str, Any]) -> None:
    _verify_signed_payload(
        config,
        field="config_fingerprint",
        expected=EXPECTED_DRY_CONFIG_FINGERPRINT,
        name="bounded dry-run config",
    )
    data = config.get("data_contract")
    optimization = config.get("optimization")
    replay = config.get("replay_contract")
    decision = config.get("decision_rule")
    boundary = config.get("execution_boundary")
    fixture = config.get("fixture_contract")
    decoder = config.get("decoder")
    checks = config.get("required_checks")
    if (
        config.get("schema_version")
        != "cure-lite-cr-lvec-v7-bounded-dry-run-config-v1"
        or config.get("method_id") != METHOD_ID
        or config.get("mode") != "synthetic_bounded_implementation_dry_run"
        or config.get("config_fingerprint_scope")
        != "all_fields_except_config_fingerprint"
        or not all(
            isinstance(value, Mapping)
            for value in (
                data,
                optimization,
                replay,
                decision,
                boundary,
                fixture,
                decoder,
                checks,
            )
        )
    ):
        raise RuntimeError("bounded dry-run config contract changed")
    if (
        data.get("provider") != "fixed_in_memory_synthetic_pair_provider_v1"
        or data.get("dataset_name") is not None
        or data.get("split") is not None
        or data.get("allowed_dataset_splits") != []
        or data.get("filesystem_dataset_root_allowed") is not False
        or data.get("real_catalog_loader_allowed") is not False
        or data.get("real_catalog_loader_call_count_required") != 0
        or data.get(
            "D_R_dataset_or_cached_tensor_payload_access_allowed"
        )
        is not False
        or data.get("D_V_access_allowed") is not False
        or data.get("D_T_access_allowed") is not False
    ):
        raise RuntimeError("bounded dry-run data boundary changed")
    if (
        optimization.get("device") != "cpu"
        or optimization.get("torch_num_threads") != 1
        or optimization.get("deterministic_algorithms") is not True
        or optimization.get("seed") != FROZEN_SEED
        or optimization.get("optimizer") != "adam"
        or optimization.get("learning_rate") != FROZEN_LEARNING_RATE
        or optimization.get("weight_decay") != 0.0
        or optimization.get("epochs") != 2
        or optimization.get("steps_per_epoch") != 4
        or optimization.get("optimizer_updates") != FROZEN_UPDATES
        or optimization.get("decoder_forward_calls_per_update") != 3
        or optimization.get("decoder_states_per_update") != 12
        or optimization.get("automatic_retry_allowed") is not False
        or optimization.get("resume_allowed") is not False
    ):
        raise RuntimeError("bounded dry-run optimization contract changed")
    if (
        fixture.get("seed") != FROZEN_SEED
        or fixture.get("feature_channels") != 8
        or fixture.get("feature_stride") != 4
        or fixture.get("output_height") != 8
        or fixture.get("output_width") != 8
        or fixture.get("pair_kinds_required")
        != ["clean_positive", "component_null", "identity_null"]
        or fixture.get("target_pixel_counts_required") != [1, 2, 3]
        or fixture.get("source_disjoint_within_update") is not True
        or fixture.get("two_endpoint_batch_forward_required") is not True
        or fixture.get("factual_miss_and_no_miss_branches_required")
        is not True
        or fixture.get(
            "pre_mask_and_small_target_representation_checks_required"
        )
        is not True
    ):
        raise RuntimeError("bounded dry-run fixture contract changed")
    if (
        decoder.get("feature_channels") != 8
        or decoder.get("feature_stride") != 4
        or decoder.get("expected_parameter_count") != EXPECTED_PARAMETER_COUNT
        or decoder.get("expected_parameter_tensor_count")
        != EXPECTED_PARAMETER_TENSORS
    ):
        raise RuntimeError("bounded dry-run decoder contract changed")
    if (
        replay.get("independent_process_replay_count") != 2
        or replay.get("temporary_outputs_must_match_before_canonical_write")
        is not True
        or replay.get("canonical_payload_byte_identity_required") is not True
        or replay.get("canonical_write_policy")
        != "create_only_after_two_temporary_replays_match"
        or decision.get("all_required_checks_must_pass") is not True
        or decision.get("dry_run_can_authorize_real_D_R_execution") is not False
        or boundary.get("D_R_payload_access_allowed") is not False
        or boundary.get("D_V_access_allowed") is not False
        or boundary.get("D_T_access_allowed") is not False
        or boundary.get("real_D_R_bounded_allowed") is not False
        or boundary.get(
            "real_run_authorization_receipt_required_or_created"
        )
        is not False
        or boundary.get("formal_800_allowed") is not False
    ):
        raise RuntimeError("bounded dry-run execution boundary changed")


def _validate_bounded_config(config: Mapping[str, Any]) -> None:
    _verify_signed_payload(
        config,
        field="config_fingerprint",
        expected=EXPECTED_BOUNDED_CONFIG_FINGERPRINT,
        name="bounded config",
    )
    budget = config.get("budget")
    sync = config.get("cuda_synchronization_policy")
    closure = config.get("implementation_closure_contract")
    authorization = config.get("future_pre_run_authorization_contract")
    policy = config.get("execution_policy")
    if (
        config.get("schema_version")
        != "cure-lite-cr-lvec-v7-bounded-config-v1"
        or config.get("method_id") != METHOD_ID
        or config.get("dataset") != "IRSTD-1K"
        or config.get("split") != "D_R"
        or not all(
            isinstance(value, Mapping)
            for value in (budget, sync, closure, authorization, policy)
        )
    ):
        raise RuntimeError("bounded config contract changed")
    if (
        budget.get("optimizer_updates") != 400
        or budget.get("decoder_forward_calls_per_update") != 3
        or budget.get("decoder_states_per_update") != 12
        or sync.get("strict_finite_and_nonzero_recovery_check_retained")
        is not True
        or sync.get("bounded_potential_host_synchronization_check_sites")
        != 1200
        or sync.get("formal_800_potential_host_synchronization_check_sites")
        != 96000
        or sync.get("formal_800_allowed") is not False
        or closure.get("required_before_any_D_R_payload_access") is not True
        or closure.get("may_directly_authorize_real_D_R_run") is not False
        or authorization.get("authorization_receipt_exists") is not False
        or authorization.get("required_after_implementation_closure")
        is not True
        or authorization.get("must_authorize_exactly_one_real_D_R_run")
        is not True
        or authorization.get("may_authorize_D_V_or_D_T") is not False
        or authorization.get("may_authorize_formal_800") is not False
        or policy.get("D_R_payload_access_before_implementation_closure_allowed")
        is not False
        or policy.get(
            "D_R_payload_access_before_separate_run_authorization_allowed"
        )
        is not False
        or policy.get("formal_800_training_allowed_by_this_config")
        is not False
    ):
        raise RuntimeError("bounded config execution contract changed")


def _load_frozen_protocol_chain(
    config_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    resolved = config_path.expanduser().resolve(strict=True)
    if resolved != _DRY_CONFIG.resolve(strict=True):
        raise ValueError(
            "CR-LVEC v7 dry-run requires its canonical bounded dry config"
        )
    if file_sha256(_DRY_CONFIG) != EXPECTED_DRY_CONFIG_SHA256:
        raise RuntimeError("bounded dry-run config file changed")
    if file_sha256(_BOUNDED_CONFIG) != EXPECTED_BOUNDED_CONFIG_SHA256:
        raise RuntimeError("bounded config file changed")
    if file_sha256(_BOUNDED_PROPOSAL) != EXPECTED_BOUNDED_PROPOSAL_SHA256:
        raise RuntimeError("bounded implementation proposal file changed")
    if file_sha256(_METHOD_PROPOSAL) != EXPECTED_METHOD_PROPOSAL_SHA256:
        raise RuntimeError("method proposal file changed")
    if file_sha256(_TOY_CLOSURE) != EXPECTED_TOY_CLOSURE_SHA256:
        raise RuntimeError("toy closure file changed")

    dry = _load_json(_DRY_CONFIG, name="bounded dry-run config")
    bounded = _load_json(_BOUNDED_CONFIG, name="bounded config")
    proposal = _load_json(
        _BOUNDED_PROPOSAL,
        name="bounded implementation proposal",
    )
    method = _load_json(_METHOD_PROPOSAL, name="method proposal")
    toy = _load_json(_TOY_CLOSURE, name="toy closure")
    _validate_dry_config(dry)
    _validate_bounded_config(bounded)
    _verify_signed_payload(
        proposal,
        field="receipt_fingerprint",
        expected=EXPECTED_BOUNDED_PROPOSAL_FINGERPRINT,
        name="bounded implementation proposal",
    )
    _verify_signed_payload(
        method,
        field="proposal_fingerprint",
        expected=EXPECTED_METHOD_PROPOSAL_FINGERPRINT,
        name="method proposal",
    )
    _verify_signed_payload(
        toy,
        field="receipt_fingerprint",
        expected=EXPECTED_TOY_CLOSURE_FINGERPRINT,
        name="toy closure",
    )

    _require_binding(
        dry["bounded_implementation_proposal_binding"],
        path=_BOUNDED_PROPOSAL,
        expected_sha256=EXPECTED_BOUNDED_PROPOSAL_SHA256,
        fingerprint_field="receipt_fingerprint",
        expected_fingerprint=EXPECTED_BOUNDED_PROPOSAL_FINGERPRINT,
        name="dry-run to bounded proposal",
    )
    _require_binding(
        dry["bounded_config_binding"],
        path=_BOUNDED_CONFIG,
        expected_sha256=EXPECTED_BOUNDED_CONFIG_SHA256,
        fingerprint_field="config_fingerprint",
        expected_fingerprint=EXPECTED_BOUNDED_CONFIG_FINGERPRINT,
        name="dry-run to bounded config",
    )
    _require_binding(
        dry["method_proposal_binding"],
        path=_METHOD_PROPOSAL,
        expected_sha256=EXPECTED_METHOD_PROPOSAL_SHA256,
        fingerprint_field="proposal_fingerprint",
        expected_fingerprint=EXPECTED_METHOD_PROPOSAL_FINGERPRINT,
        name="dry-run to method proposal",
    )
    _require_binding(
        dry["toy_gate_closure_binding"],
        path=_TOY_CLOSURE,
        expected_sha256=EXPECTED_TOY_CLOSURE_SHA256,
        fingerprint_field="receipt_fingerprint",
        expected_fingerprint=EXPECTED_TOY_CLOSURE_FINGERPRINT,
        name="dry-run to toy closure",
    )
    _require_binding(
        bounded["bounded_implementation_proposal_binding"],
        path=_BOUNDED_PROPOSAL,
        expected_sha256=EXPECTED_BOUNDED_PROPOSAL_SHA256,
        fingerprint_field="receipt_fingerprint",
        expected_fingerprint=EXPECTED_BOUNDED_PROPOSAL_FINGERPRINT,
        name="bounded config to bounded proposal",
    )
    _require_binding(
        bounded["proposal_binding"],
        path=_METHOD_PROPOSAL,
        expected_sha256=EXPECTED_METHOD_PROPOSAL_SHA256,
        fingerprint_field="proposal_fingerprint",
        expected_fingerprint=EXPECTED_METHOD_PROPOSAL_FINGERPRINT,
        name="bounded config to method proposal",
    )
    proposal_bindings = proposal.get("protocol_bindings")
    if not isinstance(proposal_bindings, Mapping):
        raise RuntimeError("bounded proposal bindings changed")
    _require_binding(
        proposal_bindings["method_proposal"],
        path=_METHOD_PROPOSAL,
        expected_sha256=EXPECTED_METHOD_PROPOSAL_SHA256,
        fingerprint_field="proposal_fingerprint",
        expected_fingerprint=EXPECTED_METHOD_PROPOSAL_FINGERPRINT,
        name="bounded proposal to method proposal",
    )
    _require_binding(
        proposal_bindings["toy_gate_closure"],
        path=_TOY_CLOSURE,
        expected_sha256=EXPECTED_TOY_CLOSURE_SHA256,
        fingerprint_field="receipt_fingerprint",
        expected_fingerprint=EXPECTED_TOY_CLOSURE_FINGERPRINT,
        name="bounded proposal to toy closure",
    )
    if (
        proposal.get("phase_status") != "SPECIFIED_NOT_IMPLEMENTED"
        or proposal.get("decision")
        != "CR_LVEC_V7_BOUNDED_IMPLEMENTATION_CREATION_AUTHORIZED"
        or proposal["execution_boundary"].get(
            "D_R_dataset_or_cached_tensor_payload_access_allowed"
        )
        is not False
        or proposal["execution_boundary"].get(
            "real_D_R_bounded_execution_authorized"
        )
        is not False
        or toy["gate_summary"].get("bounded_code_creation_authorized")
        is not True
        or toy["gate_summary"].get("real_D_R_bounded_authorized") is not False
    ):
        raise RuntimeError("bounded proposal or toy authorization changed")
    return dry, bounded, proposal, method, toy


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _state_fingerprint(decoder: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(decoder.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode(
                "ascii"
            )
        )
        if tensor.numel():
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class _ObservedCrossingDecoder(CURELiteCrossingFactorizedDecoder):
    """Record decoder calls without changing the frozen computation."""

    def __init__(self, config: CrossingFactorizedDecoderConfig) -> None:
        super().__init__(config)
        self.forward_calls = 0
        self.forward_batch_sizes: list[int] = []

    def forward(self, feature: Tensor, occupancy: Tensor) -> Tensor:
        self.forward_calls += 1
        self.forward_batch_sizes.append(int(feature.shape[0]))
        return super().forward(feature, occupancy)


def _synthetic_fixture(
    clean_pixels: tuple[tuple[int, int], ...],
    *,
    fixture_index: int,
) -> tuple[OutcomePairBatch, dict[str, BranchBatch], tuple[Tensor, ...]]:
    feature = torch.zeros(2, 8, 2, 2)
    feature[0, 0, 0, 0] = 5.0
    feature[0, 1, 1, 0] = 4.0
    feature[0, 6] = 0.5
    feature[1, 2, 1, 1] = 5.0
    feature[1, 3, 0, 1] = 4.0
    feature[1, 7] = 0.5
    feature.requires_grad_()

    occupancy_plus = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    occupancy_plus[0, 0, 0:4, 0:4] = True
    occupancy_plus[1, 0, 4:8, 4:8] = True
    occupancy_minus = torch.zeros_like(occupancy_plus)

    completion_plus = torch.zeros_like(occupancy_plus)
    completion_plus[0, 0, 5, 1] = True
    completion_plus[1, 0, 1, 6] = True
    completion_minus = completion_plus.clone()
    for row, column in clean_pixels:
        completion_minus[0, 0, row, column] = True
    increment = (completion_minus & ~completion_plus).to(torch.float32)
    valid = torch.ones_like(occupancy_plus)
    suffix = str(fixture_index)
    pair_batch = PairBatch(
        feature=feature,
        occupancy_plus=occupancy_plus,
        occupancy_minus=occupancy_minus,
        label_increment=increment,
        image_valid_mask=valid,
        pair_ids=(
            _sha(f"cr-lvec-v7-dry-clean-{suffix}"),
            _sha(f"cr-lvec-v7-dry-component-{suffix}"),
        ),
        sample_ids=(
            f"cr-lvec-v7-dry-clean-source-{suffix}",
            f"cr-lvec-v7-dry-component-source-{suffix}",
        ),
        group_ids=(
            f"cr-lvec-v7-dry-clean-group-{suffix}",
            f"cr-lvec-v7-dry-component-group-{suffix}",
        ),
        pair_kinds=("clean_positive", "component_null"),
        projection_visible=(True, True),
    )
    pair_batch.validate()
    outcome = OutcomePairBatch(
        pair_batch=pair_batch,
        completion_plus=completion_plus,
        completion_minus=completion_minus,
        gt_union=completion_minus.clone(),
        intervention_footprint=direct_projected_intervention_footprint(
            pair_batch
        ),
    )

    factual_occupancy = torch.zeros(4, 1, 8, 8, dtype=torch.bool)
    factual_valid = torch.ones_like(factual_occupancy)
    miss_feature = (
        feature[0:1].detach().repeat(4, 1, 1, 1).requires_grad_()
    )
    no_miss_feature = torch.zeros(4, 8, 2, 2)
    no_miss_feature[:, 4, 0, 1] = 3.0
    no_miss_feature[:, 5] = -0.5
    no_miss_feature.requires_grad_()
    factual = {
        "factual_miss": BranchBatch(
            feature=miss_feature,
            occupancy=factual_occupancy,
            target=completion_minus[0:1]
            .to(torch.float32)
            .repeat(4, 1, 1, 1),
            valid_mask=factual_valid,
        ),
        "factual_no_miss": BranchBatch(
            feature=no_miss_feature,
            occupancy=factual_occupancy.clone(),
            target=torch.zeros(4, 1, 8, 8),
            valid_mask=factual_valid.clone(),
        ),
    }
    for branch, batch in factual.items():
        batch.validate(expected_branch=branch)
    return outcome, factual, (feature, miss_feature, no_miss_feature)


def _topology_audit() -> dict[str, object]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9917)
        v4 = CURELiteFactorizedDecoder(
            FactorizedDecoderConfig(
                feature_channels=64,
                feature_stride=4,
            )
        )
        torch.manual_seed(9917)
        v7 = CURELiteCrossingFactorizedDecoder(
            CrossingFactorizedDecoderConfig(
                feature_channels=64,
                feature_stride=4,
            )
        )
    v4_state = v4.state_dict()
    v7_state = v7.state_dict()
    state_keys_equal = tuple(v4_state) == tuple(v7_state)
    state_values_equal = state_keys_equal and all(
        torch.equal(v4_state[name], v7_state[name]) for name in v4_state
    )
    module_types_equal = tuple(type(module) for module in v4.modules())[1:] == (
        tuple(type(module) for module in v7.modules())[1:]
    )
    parameter_count = sum(parameter.numel() for parameter in v7.parameters())
    parameter_tensors = len(tuple(v7.parameters()))
    passed = (
        state_keys_equal
        and state_values_equal
        and module_types_equal
        and parameter_count == REFERENCE_PARAMETER_COUNT
        and parameter_tensors == EXPECTED_PARAMETER_TENSORS
    )
    return {
        "reference_feature_channels": 64,
        "reference_feature_stride": 4,
        "state_keys_equal_v4": state_keys_equal,
        "initial_state_values_equal_v4": state_values_equal,
        "module_types_equal_v4": module_types_equal,
        "parameter_count": parameter_count,
        "parameter_tensor_count": parameter_tensors,
        "passed": passed,
    }


def _operator_audit(
    decoder: CURELiteCrossingFactorizedDecoder,
    fixture: OutcomePairBatch,
) -> dict[str, object]:
    safe = torch.tensor(
        [-80.0, -1.0, 0.0, 1.0, 88.0],
        dtype=torch.float32,
        requires_grad=True,
    )
    observed = crossing_recoverable_evidence(safe)
    expected_forward = torch.where(
        safe.detach() <= 0.0,
        torch.zeros_like(safe.detach()),
        torch.expm1(safe.detach()),
    )
    observed.sum().backward()
    gradient = safe.grad
    zero_probe_failed = False
    nonfinite_probe_failed = False
    try:
        crossing_recoverable_evidence(
            torch.tensor(-104.0, dtype=torch.float32)
        )
    except ValueError:
        zero_probe_failed = True
    try:
        crossing_recoverable_evidence(
            torch.tensor(89.0, dtype=torch.float32)
        )
    except ValueError:
        nonfinite_probe_failed = True

    pair = fixture.pair_batch
    with torch.no_grad():
        fields = decoder.forward_fields(
            pair.feature.detach(),
            pair.occupancy_plus,
        )
        expected_burden = F.interpolate(
            torch.log1p(fields.local_occupancy_count),
            size=tuple(pair.occupancy_plus.shape[-2:]),
            mode="nearest",
        )
        expected_margin = fields.raw_evidence - expected_burden
        expected_evidence = torch.where(
            expected_margin <= 0.0,
            torch.zeros_like(expected_margin),
            torch.expm1(expected_margin),
        )
    passed = (
        torch.equal(observed, expected_forward)
        and gradient is not None
        and torch.equal(gradient, torch.exp(safe.detach()))
        and zero_probe_failed
        and nonfinite_probe_failed
        and torch.equal(fields.occupancy_burden, expected_burden)
        and torch.equal(fields.crossing_margin, expected_margin)
        and torch.equal(fields.evidence, expected_evidence)
        and torch.equal(
            fields.logits,
            fields.baseline_logits + fields.evidence,
        )
    )
    return {
        "safe_forward_exact": bool(torch.equal(observed, expected_forward)),
        "safe_gradient_exact": bool(
            gradient is not None
            and torch.equal(gradient, torch.exp(safe.detach()))
        ),
        "negative_probe_gradient": float(gradient[0]) if gradient is not None else 0.0,
        "zero_recovery_probe_failed_fast": zero_probe_failed,
        "nonfinite_positive_probe_failed_fast": nonfinite_probe_failed,
        "occupancy_burden_exact": bool(
            torch.equal(fields.occupancy_burden, expected_burden)
        ),
        "crossing_margin_exact": bool(
            torch.equal(fields.crossing_margin, expected_margin)
        ),
        "crossing_forward_exact": bool(
            torch.equal(fields.evidence, expected_evidence)
        ),
        "logit_composition_exact": bool(
            torch.equal(
                fields.logits,
                fields.baseline_logits + fields.evidence,
            )
        ),
        "maximum_observed_absolute_margin": float(
            fields.crossing_margin.abs().max()
        ),
        "passed": bool(passed),
    }


def _structural_stop_probe(
    decoder: _ObservedCrossingDecoder,
    optimizer: torch.optim.Optimizer,
    absolute: CURELiteLoss,
    criterion: OutcomeCompleteTransitionLoss,
    outcome: OutcomePairBatch,
    factual: Mapping[str, BranchBatch],
) -> dict[str, object]:
    before = {
        name: value.detach().clone()
        for name, value in decoder.state_dict().items()
    }
    calls_before = decoder.forward_calls
    malformed = dict(factual)
    miss = factual["factual_miss"]
    malformed["factual_miss"] = BranchBatch(
        feature=miss.feature[:3],
        occupancy=miss.occupancy[:3],
        target=miss.target[:3],
        valid_mask=miss.valid_mask[:3],
    )
    rejected = False
    error_type = ""
    try:
        outcome_complete_train_step(
            decoder,
            absolute,
            criterion,
            optimizer,
            malformed,
            outcome,
        )
    except ValueError as error:
        rejected = True
        error_type = type(error).__name__
    state_unchanged = all(
        torch.equal(before[name], value)
        for name, value in decoder.state_dict().items()
    )
    no_forward = decoder.forward_calls == calls_before
    optimizer_untouched = len(optimizer.state) == 0
    return {
        "invalid_factual_batch_rejected": rejected,
        "exception_type": error_type,
        "decoder_state_unchanged": state_unchanged,
        "decoder_forward_calls": decoder.forward_calls - calls_before,
        "optimizer_state_untouched": optimizer_untouched,
        "passed": (
            rejected and state_unchanged and no_forward and optimizer_untouched
        ),
    }


def _fixture_audit(
    fixtures: Sequence[
        tuple[OutcomePairBatch, dict[str, BranchBatch], tuple[Tensor, ...]]
    ],
) -> dict[str, object]:
    target_counts = [
        int(torch.count_nonzero(outcome.response_stratum[0]))
        for outcome, _, _ in fixtures
    ]
    source_disjoint = all(
        len(set(outcome.pair_batch.sample_ids)) == 2
        for outcome, _, _ in fixtures
    )
    kinds = sorted(
        {
            *(
                kind
                for outcome, _, _ in fixtures
                for kind in outcome.pair_batch.pair_kinds
            ),
            "identity_null",
        }
    )
    pre_mask = all(
        not bool(
            (
                outcome.completion_plus
                & outcome.pair_batch.occupancy_plus
            ).any()
        )
        and not bool(
            (
                outcome.completion_minus
                & outcome.pair_batch.occupancy_minus
            ).any()
        )
        and all(
            not bool((batch.valid_mask & batch.occupancy).any())
            for batch in factual.values()
        )
        for outcome, factual, _ in fixtures
    )
    clean_component_controls = all(
        bool(outcome.response_stratum[0].any())
        and not bool(outcome.response_stratum[1].any())
        and bool(outcome.local_zero_stratum[1].any())
        and outcome.pair_batch.pair_kinds
        == ("clean_positive", "component_null")
        for outcome, _, _ in fixtures
    )
    passed = (
        target_counts == [1, 2, 3]
        and source_disjoint
        and kinds == ["clean_positive", "component_null", "identity_null"]
        and pre_mask
        and clean_component_controls
    )
    return {
        "target_pixel_counts": target_counts,
        "source_disjoint_within_update": source_disjoint,
        "pair_kinds_covered": kinds,
        "pre_mask_contract": pre_mask,
        "clean_and_component_null_controls": clean_component_controls,
        "passed": passed,
    }


def _artifact_roundtrip_probe() -> dict[str, object]:
    variants = (
        {"status": "complete_pass", "value": 1},
        {"status": "complete_nonpass", "value": 0},
        {"status": "execution_error", "exception_type": "RuntimeError"},
    )
    roundtrips: dict[str, bool] = {}
    for variant in variants:
        payload = dict(variant)
        payload["receipt_fingerprint"] = stable_fingerprint(payload)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(encoded)
        unsigned = dict(decoded)
        fingerprint = unsigned.pop("receipt_fingerprint")
        roundtrips[str(variant["status"])] = (
            decoded == payload and fingerprint == stable_fingerprint(unsigned)
        )

    create_only = False
    byte_roundtrip = False
    with tempfile.TemporaryDirectory(prefix="cr_lvec_v7_dry_probe_") as root:
        path = Path(root) / "artifact.json"
        payload = {"probe": "create_only", "value": 1}
        _write_new(path, payload)
        expected = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        byte_roundtrip = path.read_bytes() == expected
        try:
            _write_new(path, payload)
        except FileExistsError:
            create_only = True
    passed = all(roundtrips.values()) and create_only and byte_roundtrip
    return {
        "scope": "canonical_json_serialization_probe_only",
        "covers_real_runner_publication": False,
        "real_runner_result_variants_exercised": False,
        "completed_pass_result_round_trip": roundtrips["complete_pass"],
        "completed_nonpass_result_round_trip": roundtrips[
            "complete_nonpass"
        ],
        "execution_error_artifact_round_trip": roundtrips[
            "execution_error"
        ],
        "create_only_publication": create_only,
        "canonical_byte_round_trip": byte_roundtrip,
        "passed": passed,
    }


def _write_new(path: Path, payload: Mapping[str, object]) -> None:
    resolved = path.expanduser().resolve(strict=False)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with resolved.open("x", encoding="utf-8") as handle:
        handle.write(text + "\n")


def evaluate(config_path: Path | None = None) -> dict[str, object]:
    """Return one deterministic synthetic bounded dry-run result."""

    path = _DRY_CONFIG if config_path is None else config_path
    dry, bounded, proposal, method, toy = _load_frozen_protocol_chain(path)
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(FROZEN_SEED)
            fixtures = tuple(
                _synthetic_fixture(pixels, fixture_index=index)
                for index, pixels in enumerate(_TARGET_PIXEL_SETS)
            )
            decoder = _ObservedCrossingDecoder(
                CrossingFactorizedDecoderConfig(
                    feature_channels=8,
                    feature_stride=4,
                )
            )
            named_parameters = tuple(decoder.named_parameters())
            absolute = CURELiteLoss(
                LossConfig(**bounded["optimization"]["loss"])
            )
            criterion = OutcomeCompleteTransitionLoss(
                LossConfig(**bounded["optimization"]["loss"])
            )
            optimizer = torch.optim.Adam(
                decoder.parameters(),
                lr=FROZEN_LEARNING_RATE,
                weight_decay=0.0,
            )

            topology = _topology_audit()
            operator = _operator_audit(decoder, fixtures[0][0])
            fixture_audit = _fixture_audit(fixtures)
            structural_stop = _structural_stop_probe(
                decoder,
                optimizer,
                absolute,
                criterion,
                fixtures[0][0],
                fixtures[0][1],
            )

            initial_plus, initial_minus = _paired_endpoint_logits(
                decoder,
                feature=fixtures[0][0].pair_batch.feature,
                occupancy_plus=fixtures[0][0].pair_batch.occupancy_plus,
                occupancy_minus=fixtures[0][0].pair_batch.occupancy_minus,
            )
            initial_result = criterion(
                initial_plus,
                initial_minus,
                fixtures[0][0].completion_plus,
                fixtures[0][0].pair_batch.occupancy_plus,
                fixtures[0][0].gt_union,
                fixtures[0][0].pair_batch.label_increment,
                fixtures[0][0].pair_batch.image_valid_mask,
                fixtures[0][0].intervention_footprint,
            )
            plus_gradient, minus_gradient = torch.autograd.grad(
                initial_result["total"],
                (initial_plus, initial_minus),
            )
            dual_endpoint = {
                "plus_finite": bool(torch.isfinite(plus_gradient).all()),
                "minus_finite": bool(torch.isfinite(minus_gradient).all()),
                "plus_nonzero": bool(torch.count_nonzero(plus_gradient) > 0),
                "minus_nonzero": bool(torch.count_nonzero(minus_gradient) > 0),
                "plus_l2_norm": float(plus_gradient.double().norm()),
                "minus_l2_norm": float(minus_gradient.double().norm()),
            }
            dual_endpoint["passed"] = all(
                dual_endpoint[key] is True
                for key in (
                    "plus_finite",
                    "minus_finite",
                    "plus_nonzero",
                    "minus_nonzero",
                )
            )

            initial_state_fingerprint = _state_fingerprint(decoder)
            decoder.forward_calls = 0
            decoder.forward_batch_sizes.clear()
            losses: list[float] = []
            gradient_norms: list[float] = []
            parameter_gradient_audits: list[dict[str, object]] = []
            logs: dict[str, float | int] = {}
            for update in range(FROZEN_UPDATES):
                outcome, factual, _ = fixtures[update % len(fixtures)]
                logs = outcome_complete_train_step(
                    decoder,
                    absolute,
                    criterion,
                    optimizer,
                    factual,
                    outcome,
                )
                losses.append(float(logs["total"]))
                gradients_by_name = [
                    (name, parameter.grad)
                    for name, parameter in named_parameters
                ]
                missing_gradient_names = [
                    name
                    for name, gradient in gradients_by_name
                    if gradient is None
                ]
                nonfinite_gradient_names = [
                    name
                    for name, gradient in gradients_by_name
                    if gradient is not None
                    and not bool(torch.isfinite(gradient).all())
                ]
                parameter_gradient_audits.append(
                    {
                        "update": update + 1,
                        "parameter_tensor_count": len(
                            gradients_by_name
                        ),
                        "missing_gradient_names": missing_gradient_names,
                        "nonfinite_gradient_names": (
                            nonfinite_gradient_names
                        ),
                        "all_present": not missing_gradient_names,
                        "all_finite": not nonfinite_gradient_names,
                        "passed": (
                            not missing_gradient_names
                            and not nonfinite_gradient_names
                        ),
                    }
                )
                gradient_norms.append(
                    float(
                        torch.sqrt(
                            sum(
                                gradient.detach().double().square().sum()
                                for _, gradient in gradients_by_name
                                if gradient is not None
                            )
                        )
                    )
                )
            final_state_fingerprint = _state_fingerprint(decoder)
            all_gradients_finite = (
                len(parameter_gradient_audits) == FROZEN_UPDATES
                and all(
                    audit["passed"] is True
                    for audit in parameter_gradient_audits
                )
                and all(torch.isfinite(torch.tensor(gradient_norms)))
            )
            input_features_detached = all(
                tensor.grad is None
                for _, _, tensors in fixtures
                for tensor in tensors
            )
            fixed_budget = (
                len(losses) == FROZEN_UPDATES
                and decoder.forward_calls == FROZEN_UPDATES * 3
                and decoder.forward_batch_sizes == [4] * (FROZEN_UPDATES * 3)
                and int(logs["decoder_forward_calls_per_update"]) == 3
                and int(logs["decoder_states_per_update"]) == 12
                and int(logs["backward_calls"]) == 1
                and int(logs["optimizer_steps"]) == 1
                and all(torch.isfinite(torch.tensor(losses)))
            )
            training = {
                "optimizer_updates": len(losses),
                "decoder_forward_calls": decoder.forward_calls,
                "expected_decoder_forward_calls": FROZEN_UPDATES * 3,
                "decoder_forward_batch_sizes": list(
                    decoder.forward_batch_sizes
                ),
                "decoder_states_per_update": int(
                    logs["decoder_states_per_update"]
                ),
                "backward_calls": FROZEN_UPDATES,
                "optimizer_steps": FROZEN_UPDATES,
                "losses": losses,
                "all_losses_finite": bool(
                    all(torch.isfinite(torch.tensor(losses)))
                ),
                "gradient_l2_norms": gradient_norms,
                "parameter_gradient_audits": (
                    parameter_gradient_audits
                ),
                "parameter_gradient_updates_passed": sum(
                    audit["passed"] is True
                    for audit in parameter_gradient_audits
                ),
                "all_parameter_gradients_finite_each_update": (
                    all_gradients_finite
                ),
                "initial_state_fingerprint": initial_state_fingerprint,
                "final_state_fingerprint": final_state_fingerprint,
                "decoder_state_changed": (
                    initial_state_fingerprint != final_state_fingerprint
                ),
                "input_features_detached": input_features_detached,
                "fixed_budget_pass": fixed_budget,
            }

            decoder.eval()
            identity_outcome = fixtures[0][0]
            calls_before_identity = decoder.forward_calls
            with torch.no_grad():
                identity_plus, identity_minus = _paired_endpoint_logits(
                    decoder,
                    feature=identity_outcome.pair_batch.feature,
                    occupancy_plus=identity_outcome.pair_batch.occupancy_minus,
                    occupancy_minus=identity_outcome.pair_batch.occupancy_minus,
                )
            identity_exact = torch.equal(identity_plus, identity_minus)
            identity_single_2b_forward = (
                decoder.forward_calls == calls_before_identity + 1
                and decoder.forward_batch_sizes[-1] == 4
            )
            controls = {
                "clean_positive_present": True,
                "component_null_present": True,
                "identity_null_exact": identity_exact,
                "identity_null_max_abs_logit_delta": float(
                    (identity_plus - identity_minus).abs().max()
                ),
                "identity_single_2B_forward": identity_single_2b_forward,
                "passed": identity_exact and identity_single_2b_forward,
            }
            artifact_roundtrip = _artifact_roundtrip_probe()
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
        torch.set_num_threads(previous_threads)

    # The bounded dry entry point deliberately has no real-catalog loader
    # import or call edge.  Zero is therefore a structural property, not a
    # monkeypatch-based observation of the real runner.
    real_loader_calls = 0
    checks = {
        "strict_protocol_bindings": True,
        "crossing_operator_contract": operator["passed"] is True,
        "frozen_topology_and_parameter_count": topology["passed"] is True,
        "feature_and_base_detachment": training[
            "input_features_detached"
        ]
        is True,
        "two_endpoint_batch_forward": controls[
            "identity_single_2B_forward"
        ]
        is True,
        "dual_endpoint_gradients_finite_nonzero": dual_endpoint["passed"]
        is True,
        "all_parameter_gradients_finite_each_update": training[
            "all_parameter_gradients_finite_each_update"
        ]
        is True,
        "pre_mask_contract": fixture_audit["pre_mask_contract"] is True,
        "small_target_representation_contract": fixture_audit[
            "target_pixel_counts"
        ]
        == [1, 2, 3],
        "clean_component_and_identity_null_controls": (
            fixture_audit["clean_and_component_null_controls"] is True
            and controls["passed"] is True
        ),
        "fixed_budget_and_update_counts": training["fixed_budget_pass"]
        is True,
        "structural_failure_zero_training_stop": structural_stop["passed"]
        is True,
        "completed_pass_result_round_trip": artifact_roundtrip[
            "completed_pass_result_round_trip"
        ]
        is True,
        "completed_nonpass_result_round_trip": artifact_roundtrip[
            "completed_nonpass_result_round_trip"
        ]
        is True,
        "execution_error_artifact_round_trip": artifact_roundtrip[
            "execution_error_artifact_round_trip"
        ]
        is True,
        "create_only_publication": artifact_roundtrip[
            "create_only_publication"
        ]
        is True,
        "real_catalog_loader_call_count_zero": real_loader_calls == 0,
        "D_R_payload_accessed_false": True,
    }
    if set(checks) != set(dry["required_checks"]):
        raise RuntimeError("dry-run required-check inventory changed")
    all_pass = all(checks.values())
    decision_rule = dry["decision_rule"]
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "method_id": METHOD_ID,
        "mode": "synthetic_bounded_implementation_dry_run",
        "protocol_binding": {
            "bounded_implementation_proposal": {
                "repo_path": str(_BOUNDED_PROPOSAL.relative_to(_ROOT)),
                "file_sha256": file_sha256(_BOUNDED_PROPOSAL),
                "receipt_fingerprint": proposal["receipt_fingerprint"],
            },
            "bounded_config": {
                "repo_path": str(_BOUNDED_CONFIG.relative_to(_ROOT)),
                "file_sha256": file_sha256(_BOUNDED_CONFIG),
                "config_fingerprint": bounded["config_fingerprint"],
            },
            "bounded_dry_run_config": {
                "repo_path": str(_DRY_CONFIG.relative_to(_ROOT)),
                "file_sha256": file_sha256(_DRY_CONFIG),
                "config_fingerprint": dry["config_fingerprint"],
            },
            "method_proposal": {
                "repo_path": str(_METHOD_PROPOSAL.relative_to(_ROOT)),
                "file_sha256": file_sha256(_METHOD_PROPOSAL),
                "proposal_fingerprint": method["proposal_fingerprint"],
            },
            "toy_gate_closure": {
                "repo_path": str(_TOY_CLOSURE.relative_to(_ROOT)),
                "file_sha256": file_sha256(_TOY_CLOSURE),
                "receipt_fingerprint": toy["receipt_fingerprint"],
            },
        },
        "contract": {
            "device": "cpu",
            "seed": FROZEN_SEED,
            "optimizer": "adam",
            "learning_rate": FROZEN_LEARNING_RATE,
            "weight_decay": 0.0,
            "epochs": 2,
            "steps_per_epoch": 4,
            "optimizer_updates": FROZEN_UPDATES,
            "decoder_forward_calls_per_update": 3,
            "decoder_states_per_update": 12,
            "feature_channels": 8,
            "feature_stride": 4,
            "expected_parameter_count": EXPECTED_PARAMETER_COUNT,
            "expected_parameter_tensor_count": EXPECTED_PARAMETER_TENSORS,
            "torch_num_threads": 1,
            "deterministic_algorithms": True,
            "data_source": "fixed_in_memory_synthetic_fixtures_only",
        },
        "topology_audit": topology,
        "operator_audit": operator,
        "fixture_audit": fixture_audit,
        "structural_stop_audit": structural_stop,
        "dual_endpoint_gradient_audit": dual_endpoint,
        "training_audit": training,
        "control_audit": controls,
        "artifact_roundtrip_audit": artifact_roundtrip,
        "artifact_roundtrip_scope": (
            "serialization_probe_only_not_real_runner_publication"
        ),
        "checks": checks,
        "all_pass": all_pass,
        "decision": (
            decision_rule["pass_decision"]
            if all_pass
            else decision_rule["fail_decision"]
        ),
        "bounded_implementation_closure_authorized": False,
        "bounded_implementation_closure_eligible_after_replay": all_pass,
        "real_catalog_loader_call_count": real_loader_calls,
        "real_catalog_loader_call_count_basis": (
            "structural_isolation_no_real_loader_import_or_call_edge"
        ),
        "real_loader_imported_by_dry_entrypoint": False,
        "real_loader_symbol_reachable_from_dry_execution": False,
        "D_R_payload_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "real_D_R_bounded_authorized": False,
        "real_run_authorization_created": False,
        "detection_performance_evaluated": False,
        "calibration_performed": False,
        "formal_800_authorized": False,
        "full_CURE_authorized": False,
        "other_detector_integration_authorized": False,
        "automatic_retry_performed": False,
        "resume_used": False,
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = evaluate(args.config)
    _write_new(args.output, result)
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    if result["all_pass"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
