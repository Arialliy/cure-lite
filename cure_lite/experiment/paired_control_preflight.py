"""Deterministic, tensor-free preflight for CURE-Lite matched controls.

The preflight consumes an already reconstructed ``D_R`` :class:`PairCatalog`.
It verifies the zero-feature and feature-only input contracts, builds the
source-independent DCT basis at the real feature shape, and constructs the
complete source-disjoint target permutation.  It never runs a decoder,
optimizer, calibration, evaluation, or any non-``D_R`` split.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from ..cache.schema import file_sha256, stable_fingerprint
from ..paired_control_inputs import (
    TARGET_PERMUTATION_INCONCLUSIVE,
    TARGET_PERMUTATION_READY,
    build_dct_coordinate_basis,
    build_target_permutation,
    feature_only_zero_occupancy,
    nominal_zero_feature_like,
)
from ..paired_types import PairCatalog, tensor_content_fingerprint


CONTROL_PREFLIGHT_PROTOCOL_FINGERPRINT = (
    "5a2f357911fb5f1dc1a946b3dbad429d256c390677d238b2f395fe90ce91fac8"
)
CONTROL_PREFLIGHT_CATALOG_FINGERPRINT = (
    "4886e52d2cfb3392d0f4fdda376159d6e7f694fd449dc809cf8874793febde76"
)
CONTROL_PREFLIGHT_CLEAN_PAIR_COUNT = 206

CONTROL_PREFLIGHT_COMPLETE_SCHEMA = (
    "cure-lite-paired-control-preflight-complete-v1"
)
CONTROL_PREFLIGHT_RUN_SCHEMA = "cure-lite-paired-control-preflight-run-v1"
CONTROL_CONTRACTS_SCHEMA = "cure-lite-paired-control-contracts-v1"
CONTROL_DCT_SCHEMA = "cure-lite-paired-control-dct-basis-v1"
CONTROL_PERMUTATION_SCHEMA = "cure-lite-paired-control-permutation-v1"

REQUIRED_CONTROL_SOURCE_PATHS = (
    "cure_lite/decoder.py",
    "cure_lite/experiment/paired_control_preflight.py",
    "cure_lite/experiment/paired_catalog.py",
    "cure_lite/paired_control_inputs.py",
    "cure_lite/paired_control_losses.py",
    "cure_lite/paired_losses.py",
    "cure_lite/paired_types.py",
    "cure_lite/train/paired_control_step.py",
    "cure_lite/train/paired_step.py",
    "tools/run_paired_control_preflight.py",
    "tools/run_paired_preflight.py",
)

_INCOMPLETE_NAME = ".incomplete"
_COMPLETE_NAME = "COMPLETE.json"
_RECEIPTS_DIR = "receipts"
_CONTRACTS_NAME = "control_contracts.json"
_DCT_NAME = "dct_basis.json"
_PERMUTATION_NAME = "target_permutation.json"
_RUN_NAME = "run_receipt.json"
_RECEIPT_NAMES = frozenset(
    (_CONTRACTS_NAME, _DCT_NAME, _PERMUTATION_NAME, _RUN_NAME)
)
_HEX_DIGITS = frozenset("0123456789abcdef")


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{name} contains non-finite number {value}")

    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} root must be a JSON object")
    return dict(payload)


def _fingerprinted(
    payload: Mapping[str, object],
    *,
    field: str = "receipt_fingerprint",
) -> dict[str, object]:
    result = dict(payload)
    if field in result:
        raise ValueError(f"payload already contains {field}")
    result[field] = stable_fingerprint(result)
    return result


def _verify_fingerprinted(
    payload: Mapping[str, Any],
    *,
    name: str,
    field: str = "receipt_fingerprint",
) -> None:
    fingerprint = payload.get(field)
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if (
        not isinstance(fingerprint, str)
        or stable_fingerprint(unsigned) != fingerprint
    ):
        raise RuntimeError(f"{name} fingerprint is inconsistent")


def _prepare_output(path: Path) -> Path:
    candidate = path.expanduser()
    absolute = Path(os.path.abspath(candidate))
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(
            f"paired control preflight output already exists: {absolute}"
        )
    for parent in (absolute.parent, *absolute.parents):
        if parent.exists() and parent.is_symlink():
            raise ValueError(
                "paired control output may not traverse a symbolic link"
            )
    return absolute


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted((root / _RECEIPTS_DIR).iterdir())
        if path.is_file()
    }


def _validate_catalog(
    catalog: PairCatalog,
    *,
    expected_catalog_fingerprint: str,
    expected_protocol_fingerprint: str,
    expected_clean_pair_count: int,
) -> None:
    if not isinstance(catalog, PairCatalog):
        raise TypeError("catalog must be a PairCatalog")
    if catalog.split != "D_R":
        raise ValueError("paired control preflight permits only D_R")
    if catalog.catalog_fingerprint != expected_catalog_fingerprint:
        raise RuntimeError("pair catalog fingerprint differs from the freeze")
    if catalog.paired_protocol_fingerprint != expected_protocol_fingerprint:
        raise RuntimeError("paired protocol fingerprint differs from the freeze")
    if stable_fingerprint(catalog.canonical_payload()) != (
        catalog.catalog_fingerprint
    ):
        raise RuntimeError("pair catalog fingerprint does not reproduce")
    if (
        isinstance(expected_clean_pair_count, bool)
        or not isinstance(expected_clean_pair_count, int)
        or expected_clean_pair_count < 1
    ):
        raise ValueError("expected_clean_pair_count must be positive")
    if len(catalog.clean_positive) != expected_clean_pair_count:
        raise RuntimeError("clean-positive population differs from the freeze")


def _validate_bindings(
    catalog: PairCatalog,
    input_bindings: Mapping[str, object],
    control_source_hashes: Mapping[str, str],
) -> None:
    expected = {
        "dataset": catalog.dataset,
        "split": "D_R",
        "runtime_splits": ["D_R"],
        "paired_protocol_fingerprint": catalog.paired_protocol_fingerprint,
        "pair_catalog_fingerprint": catalog.catalog_fingerprint,
    }
    for field, value in expected.items():
        if input_bindings.get(field) != value:
            raise RuntimeError(f"control input binding mismatch: {field}")
    for field in (
        "upstream_paired_preflight_complete_fingerprint",
        "upstream_paired_preflight_complete_sha256",
    ):
        _require_sha256(input_bindings.get(field), name=f"input_bindings.{field}")
    if tuple(sorted(control_source_hashes)) != tuple(
        sorted(REQUIRED_CONTROL_SOURCE_PATHS)
    ):
        raise RuntimeError("control source inventory differs from the freeze")
    for path, digest in control_source_hashes.items():
        _require_sha256(digest, name=f"control_source_hashes[{path!r}]")


def _population_signatures(
    catalog: PairCatalog,
) -> tuple[tuple[int, ...], str, tuple[int, ...], str]:
    feature_signatures = {
        (tuple(pair.feature.shape), str(pair.feature.dtype))
        for pair in catalog.clean_positive
    }
    occupancy_signatures = {
        (tuple(pair.occupancy_plus.shape), str(pair.occupancy_plus.dtype))
        for pair in catalog.clean_positive
    } | {
        (tuple(pair.occupancy_minus.shape), str(pair.occupancy_minus.dtype))
        for pair in catalog.clean_positive
    }
    if len(feature_signatures) != 1:
        raise RuntimeError("clean pairs do not share one real feature signature")
    if len(occupancy_signatures) != 1:
        raise RuntimeError("clean pairs do not share one occupancy signature")
    feature_shape, feature_dtype = next(iter(feature_signatures))
    occupancy_shape, occupancy_dtype = next(iter(occupancy_signatures))
    return feature_shape, feature_dtype, occupancy_shape, occupancy_dtype


def build_control_contracts_receipt(
    catalog: PairCatalog,
) -> dict[str, object]:
    """Build and verify the two matched-control input contracts."""

    (
        feature_shape,
        feature_dtype,
        occupancy_shape,
        occupancy_dtype,
    ) = _population_signatures(catalog)
    zero_feature_fingerprints: set[str] = set()
    zero_occupancy_fingerprints: set[str] = set()
    shared_zero_occupancy_object = True
    for pair in catalog.clean_positive:
        zero_feature = nominal_zero_feature_like(pair.feature)
        if (
            tuple(zero_feature.shape) != tuple(pair.feature.shape)
            or zero_feature.dtype != pair.feature.dtype
            or zero_feature.device != pair.feature.device
            or zero_feature.requires_grad
            or torch.count_nonzero(zero_feature).item() != 0
        ):
            raise RuntimeError("nominal zero-feature contract failed")
        zero_feature_fingerprints.add(
            tensor_content_fingerprint(zero_feature)
        )

        zero_plus, zero_minus = feature_only_zero_occupancy(
            pair.occupancy_plus.unsqueeze(0),
            pair.occupancy_minus.unsqueeze(0),
        )
        if (
            zero_plus is not zero_minus
            or tuple(zero_plus.shape)
            != (1, *tuple(pair.occupancy_plus.shape))
            or zero_plus.dtype != torch.bool
            or zero_plus.device != pair.occupancy_plus.device
            or torch.count_nonzero(zero_plus).item() != 0
        ):
            raise RuntimeError("feature-only zero-occupancy contract failed")
        shared_zero_occupancy_object = (
            shared_zero_occupancy_object and zero_plus is zero_minus
        )
        zero_occupancy_fingerprints.add(
            tensor_content_fingerprint(zero_plus)
        )

    if len(zero_feature_fingerprints) != 1:
        raise RuntimeError("zero-feature construction is not population-invariant")
    if len(zero_occupancy_fingerprints) != 1:
        raise RuntimeError(
            "zero-occupancy construction is not population-invariant"
        )
    return _fingerprinted(
        {
            "schema_version": CONTROL_CONTRACTS_SCHEMA,
            "dataset": catalog.dataset,
            "split": "D_R",
            "paired_protocol_fingerprint": (
                catalog.paired_protocol_fingerprint
            ),
            "pair_catalog_fingerprint": catalog.catalog_fingerprint,
            "clean_pair_count": len(catalog.clean_positive),
            "real_feature_signature": {
                "shape": list(feature_shape),
                "dtype": feature_dtype,
                "device": "cpu",
                "uniform_over_clean_population": True,
            },
            "real_occupancy_signature": {
                "shape": list(occupancy_shape),
                "dtype": occupancy_dtype,
                "device": "cpu",
                "uniform_over_clean_population": True,
            },
            "nominal_zero_feature": {
                "construction": "zeros_from_shape_dtype_device_only",
                "retains_shape": True,
                "retains_dtype": True,
                "retains_device": True,
                "all_elements_zero": True,
                "source_values_read": False,
                "requires_grad": False,
                "tensor_fingerprint": next(
                    iter(zero_feature_fingerprints)
                ),
            },
            "feature_only_zero_occupancy": {
                "construction": "one_fixed_zero_for_both_endpoints",
                "retains_shape": True,
                "retains_bool_dtype": True,
                "retains_device": True,
                "all_elements_zero": True,
                "plus_minus_equal": True,
                "shared_tensor_object": shared_zero_occupancy_object,
                "tensor_fingerprint": next(
                    iter(zero_occupancy_fingerprints)
                ),
            },
            "contracts_passed": True,
            "raw_tensor_payloads_written": False,
        }
    )


def build_dct_basis_receipt(
    catalog: PairCatalog,
) -> dict[str, object]:
    """Build the source-independent DCT basis at the real feature shape."""

    feature_shape, feature_dtype, _, _ = _population_signatures(catalog)
    if len(feature_shape) != 4 or feature_shape[0] != 1:
        raise RuntimeError("real feature signature must be [1,C,h,w]")
    dtype = getattr(torch, feature_dtype.removeprefix("torch."), None)
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError("real feature dtype cannot be reconstructed")
    basis = build_dct_coordinate_basis(
        channels=feature_shape[1],
        height=feature_shape[2],
        width=feature_shape[3],
        dtype=dtype,
    )
    return _fingerprinted(
        {
            "schema_version": CONTROL_DCT_SCHEMA,
            "dataset": catalog.dataset,
            "split": "D_R",
            "paired_protocol_fingerprint": (
                catalog.paired_protocol_fingerprint
            ),
            "pair_catalog_fingerprint": catalog.catalog_fingerprint,
            "clean_pair_count": len(catalog.clean_positive),
            "real_feature_shape": list(feature_shape),
            "real_feature_dtype": feature_dtype,
            "basis": basis.canonical_payload,
            "basis_fingerprint": basis.basis_fingerprint,
            "all_clean_pairs_share_basis": True,
            "raw_tensor_payloads_written": False,
        }
    )


def build_target_permutation_receipt(
    catalog: PairCatalog,
) -> dict[str, object]:
    """Build the complete target permutation or an explicit inconclusive result."""

    plan = build_target_permutation(catalog.clean_positive)
    assignments = [
        assignment.canonical_payload() for assignment in plan.assignments
    ]
    donor_counts = {
        pair_id: 0 for pair_id in plan.canonical_pair_ids
    }
    for assignment in plan.assignments:
        donor_counts[assignment.donor_pair_id] += 1
    donor_marginal = [
        {"donor_pair_id": pair_id, "assignment_count": donor_counts[pair_id]}
        for pair_id in plan.canonical_pair_ids
    ]
    full_donor_marginal = (
        plan.ready
        and len(donor_counts) == len(plan.canonical_pair_ids)
        and all(count == 1 for count in donor_counts.values())
    )
    source_disjoint = plan.ready and all(
        assignment.recipient_sample_id != assignment.donor_sample_id
        for assignment in plan.assignments
    )
    fixed_point_free = plan.ready and all(
        assignment.recipient_pair_id != assignment.donor_pair_id
        for assignment in plan.assignments
    )
    status = (
        TARGET_PERMUTATION_READY
        if (
            plan.ready
            and full_donor_marginal
            and source_disjoint
            and fixed_point_free
        )
        else TARGET_PERMUTATION_INCONCLUSIVE
    )
    if status != plan.status:
        raise RuntimeError("target permutation postconditions disagree")
    return _fingerprinted(
        {
            "schema_version": CONTROL_PERMUTATION_SCHEMA,
            "dataset": catalog.dataset,
            "split": "D_R",
            "paired_protocol_fingerprint": (
                catalog.paired_protocol_fingerprint
            ),
            "pair_catalog_fingerprint": catalog.catalog_fingerprint,
            "clean_pair_count": len(catalog.clean_positive),
            "status": plan.status,
            "reason_code": plan.reason_code,
            "compatible_edges": plan.compatible_edges,
            "canonical_pair_ids": list(plan.canonical_pair_ids),
            "assignments": assignments,
            "assignment_fingerprint": stable_fingerprint(assignments),
            "plan_fingerprint": plan.plan_fingerprint,
            "source_disjoint": source_disjoint,
            "fixed_point_free": fixed_point_free,
            "full_donor_marginal": full_donor_marginal,
            "donor_marginal": donor_marginal,
            "donor_marginal_fingerprint": stable_fingerprint(
                donor_marginal
            ),
            "assignment_count": len(assignments),
            "training_materialization_contract": {
                "recipient_batch_pair_ids_must_bind_exact_assignments": True,
                "donor_pair_id_must_resolve_in_frozen_catalog": True,
                "donor_target_must_be_clean_increment": True,
                "donor_target_fingerprint_must_match_assignment": True,
                "materializer": (
                    "cure_lite.paired_control_inputs."
                    "materialize_permuted_label_increments"
                ),
                "unbound_external_permuted_labels_forbidden": True,
                "runtime_training_binding_implemented_by_preflight": False,
                "runtime_training_binding_required_before_control_training": (
                    True
                ),
            },
            "training_performed": False,
        }
    )


@dataclass(frozen=True)
class PublishedControlPreflight:
    root: Path
    status: str
    catalog_fingerprint: str
    protocol_fingerprint: str
    contracts_fingerprint: str
    dct_basis_fingerprint: str
    permutation_fingerprint: str
    run_receipt_fingerprint: str
    complete_fingerprint: str

    def verify_unchanged(
        self,
        *,
        expected_catalog_fingerprint: str = (
            CONTROL_PREFLIGHT_CATALOG_FINGERPRINT
        ),
        expected_protocol_fingerprint: str = (
            CONTROL_PREFLIGHT_PROTOCOL_FINGERPRINT
        ),
        expected_clean_pair_count: int = CONTROL_PREFLIGHT_CLEAN_PAIR_COUNT,
    ) -> None:
        verified = load_control_preflight_artifact(
            self.root,
            expected_catalog_fingerprint=expected_catalog_fingerprint,
            expected_protocol_fingerprint=expected_protocol_fingerprint,
            expected_clean_pair_count=expected_clean_pair_count,
        )
        if verified != self:
            raise RuntimeError("published control preflight identity changed")


def write_control_preflight_artifact(
    catalog: PairCatalog,
    output_dir: str | Path,
    *,
    input_bindings: Mapping[str, object],
    control_source_hashes: Mapping[str, str],
    expected_catalog_fingerprint: str = (
        CONTROL_PREFLIGHT_CATALOG_FINGERPRINT
    ),
    expected_protocol_fingerprint: str = (
        CONTROL_PREFLIGHT_PROTOCOL_FINGERPRINT
    ),
    expected_clean_pair_count: int = CONTROL_PREFLIGHT_CLEAN_PAIR_COUNT,
) -> PublishedControlPreflight:
    """Create one immutable, tensor-free matched-control preflight artifact."""

    _validate_catalog(
        catalog,
        expected_catalog_fingerprint=expected_catalog_fingerprint,
        expected_protocol_fingerprint=expected_protocol_fingerprint,
        expected_clean_pair_count=expected_clean_pair_count,
    )
    _validate_bindings(catalog, input_bindings, control_source_hashes)
    root = _prepare_output(Path(output_dir))
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    incomplete = root / _INCOMPLETE_NAME
    incomplete.open("xb").close()
    receipts_dir = root / _RECEIPTS_DIR
    receipts_dir.mkdir(exist_ok=False)

    contracts = build_control_contracts_receipt(catalog)
    dct = build_dct_basis_receipt(catalog)
    permutation = build_target_permutation_receipt(catalog)
    _write_new_json(receipts_dir / _CONTRACTS_NAME, contracts)
    _write_new_json(receipts_dir / _DCT_NAME, dct)
    _write_new_json(receipts_dir / _PERMUTATION_NAME, permutation)

    ready = permutation["status"] == TARGET_PERMUTATION_READY
    source_hashes = dict(sorted(control_source_hashes.items()))
    source_hashes_fingerprint = stable_fingerprint(source_hashes)
    run_receipt = _fingerprinted(
        {
            "schema_version": CONTROL_PREFLIGHT_RUN_SCHEMA,
            "execution_status": "completed",
            "dataset": catalog.dataset,
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "paired_protocol_fingerprint": (
                catalog.paired_protocol_fingerprint
            ),
            "pair_catalog_fingerprint": catalog.catalog_fingerprint,
            "clean_pair_count": len(catalog.clean_positive),
            "input_bindings": dict(input_bindings),
            "control_source_hashes": source_hashes,
            "control_source_hashes_fingerprint": (
                source_hashes_fingerprint
            ),
            "receipts": {
                "control_contracts_fingerprint": contracts[
                    "receipt_fingerprint"
                ],
                "dct_basis_receipt_fingerprint": dct[
                    "receipt_fingerprint"
                ],
                "dct_basis_fingerprint": dct["basis_fingerprint"],
                "target_permutation_receipt_fingerprint": permutation[
                    "receipt_fingerprint"
                ],
                "target_permutation_plan_fingerprint": permutation[
                    "plan_fingerprint"
                ],
                "target_assignment_fingerprint": permutation[
                    "assignment_fingerprint"
                ],
            },
            "gates": {
                "zero_feature_contract_passed": contracts[
                    "contracts_passed"
                ],
                "feature_only_contract_passed": contracts[
                    "contracts_passed"
                ],
                "dct_basis_constructed_at_real_shape": True,
                "target_permutation_status": permutation["status"],
                "full_donor_marginal": permutation[
                    "full_donor_marginal"
                ],
                "matched_controls_ready": ready,
                "matched_controls_static_preflight_pass": ready,
            },
            "next_route": (
                "run_bounded_d_r_only_matched_control_learnability"
                if ready
                else "report_target_permutation_computationally_inconclusive"
            ),
            "execution_policy": {
                "read_only_splits": ["D_R"],
                "training_performed": False,
                "formal_800_epoch_training_authorized": False,
                "d_v_accessed": False,
                "d_t_accessed": False,
                "model_modified": False,
                "decoder_forward_performed": False,
                "optimizer_step_performed": False,
                "calibration_performed": False,
                "inference_performed": False,
                "full_cure_started": False,
                "backbone_integration_performed": False,
            },
            "publication_contract": {
                "create_only": True,
                "complete_written_last": True,
                "raw_tensor_payloads_written": False,
                "timestamps_or_output_paths_recorded": False,
            },
        }
    )
    _write_new_json(receipts_dir / _RUN_NAME, run_receipt)

    artifact_files = _artifact_hashes(root)
    complete = _fingerprinted(
        {
            "schema_version": CONTROL_PREFLIGHT_COMPLETE_SCHEMA,
            "execution_status": "complete",
            "dataset": catalog.dataset,
            "split": "D_R",
            "runtime_splits": ["D_R"],
            "paired_protocol_fingerprint": (
                catalog.paired_protocol_fingerprint
            ),
            "pair_catalog_fingerprint": catalog.catalog_fingerprint,
            "clean_pair_count": len(catalog.clean_positive),
            "status": "complete",
            "target_permutation_status": permutation["status"],
            "matched_controls_static_preflight_pass": ready,
            "control_source_hashes_fingerprint": (
                source_hashes_fingerprint
            ),
            "contracts_fingerprint": contracts["receipt_fingerprint"],
            "dct_basis_fingerprint": dct["basis_fingerprint"],
            "permutation_fingerprint": permutation["plan_fingerprint"],
            "run_receipt_fingerprint": run_receipt[
                "receipt_fingerprint"
            ],
            "artifact_files": artifact_files,
            "artifact_file_count": len(artifact_files),
            "training_performed": False,
            "d_v_accessed": False,
            "d_t_accessed": False,
            "model_modified": False,
            "next_route": run_receipt["next_route"],
        },
        field="complete_fingerprint",
    )
    # COMPLETE is deliberately the final artifact file created.
    _write_new_json(root / _COMPLETE_NAME, complete)
    incomplete.unlink()
    return load_control_preflight_artifact(
        root,
        expected_catalog_fingerprint=expected_catalog_fingerprint,
        expected_protocol_fingerprint=expected_protocol_fingerprint,
        expected_clean_pair_count=expected_clean_pair_count,
    )


def load_control_preflight_artifact(
    output_dir: str | Path,
    *,
    expected_catalog_fingerprint: str = (
        CONTROL_PREFLIGHT_CATALOG_FINGERPRINT
    ),
    expected_protocol_fingerprint: str = (
        CONTROL_PREFLIGHT_PROTOCOL_FINGERPRINT
    ),
    expected_clean_pair_count: int = CONTROL_PREFLIGHT_CLEAN_PAIR_COUNT,
) -> PublishedControlPreflight:
    """Load and fully check one completed matched-control preflight."""

    root = Path(output_dir).expanduser().resolve(strict=False)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("control preflight root must be a regular directory")
    if (root / _INCOMPLETE_NAME).exists():
        raise RuntimeError("control preflight publication is incomplete")
    if {path.name for path in root.iterdir()} != {
        _RECEIPTS_DIR,
        _COMPLETE_NAME,
    }:
        raise RuntimeError("control preflight top-level inventory changed")
    receipts_dir = root / _RECEIPTS_DIR
    if receipts_dir.is_symlink() or not receipts_dir.is_dir():
        raise ValueError("control receipt directory must be regular")
    if {path.name for path in receipts_dir.iterdir()} != _RECEIPT_NAMES:
        raise RuntimeError("control preflight receipt inventory changed")

    complete = _strict_json(root / _COMPLETE_NAME, name="control COMPLETE")
    contracts = _strict_json(
        receipts_dir / _CONTRACTS_NAME,
        name="control contracts receipt",
    )
    dct = _strict_json(receipts_dir / _DCT_NAME, name="DCT receipt")
    permutation = _strict_json(
        receipts_dir / _PERMUTATION_NAME,
        name="target permutation receipt",
    )
    run_receipt = _strict_json(
        receipts_dir / _RUN_NAME,
        name="control run receipt",
    )
    _verify_fingerprinted(
        complete,
        name="control COMPLETE",
        field="complete_fingerprint",
    )
    for payload, name in (
        (contracts, "control contracts receipt"),
        (dct, "DCT receipt"),
        (permutation, "target permutation receipt"),
        (run_receipt, "control run receipt"),
    ):
        _verify_fingerprinted(payload, name=name)

    if complete.get("artifact_files") != _artifact_hashes(root):
        raise RuntimeError("control preflight artifact hashes changed")
    if complete.get("artifact_file_count") != len(_RECEIPT_NAMES):
        raise RuntimeError("control preflight artifact count changed")
    for payload in (complete, contracts, dct, permutation, run_receipt):
        if payload.get("split") != "D_R":
            raise RuntimeError("control preflight split binding changed")
        if (
            payload.get("paired_protocol_fingerprint")
            != expected_protocol_fingerprint
            or payload.get("pair_catalog_fingerprint")
            != expected_catalog_fingerprint
            or payload.get("clean_pair_count")
            != expected_clean_pair_count
        ):
            raise RuntimeError("control preflight frozen binding changed")

    source_hashes = run_receipt.get("control_source_hashes")
    if not isinstance(source_hashes, Mapping):
        raise RuntimeError("control source hashes are malformed")
    if tuple(sorted(source_hashes)) != tuple(
        sorted(REQUIRED_CONTROL_SOURCE_PATHS)
    ):
        raise RuntimeError("control source inventory changed")
    for path, digest in source_hashes.items():
        _require_sha256(digest, name=f"control_source_hashes[{path!r}]")
    source_hashes_fingerprint = stable_fingerprint(dict(source_hashes))
    if not (
        complete.get("control_source_hashes_fingerprint")
        == run_receipt.get("control_source_hashes_fingerprint")
        == source_hashes_fingerprint
        and complete.get("contracts_fingerprint")
        == contracts.get("receipt_fingerprint")
        == run_receipt.get("receipts", {}).get(
            "control_contracts_fingerprint"
        )
        and complete.get("dct_basis_fingerprint")
        == dct.get("basis_fingerprint")
        == run_receipt.get("receipts", {}).get("dct_basis_fingerprint")
        and dct.get("receipt_fingerprint")
        == run_receipt.get("receipts", {}).get(
            "dct_basis_receipt_fingerprint"
        )
        and complete.get("permutation_fingerprint")
        == permutation.get("plan_fingerprint")
        == run_receipt.get("receipts", {}).get(
            "target_permutation_plan_fingerprint"
        )
        and permutation.get("receipt_fingerprint")
        == run_receipt.get("receipts", {}).get(
            "target_permutation_receipt_fingerprint"
        )
        and permutation.get("assignment_fingerprint")
        == run_receipt.get("receipts", {}).get(
            "target_assignment_fingerprint"
        )
        and complete.get("run_receipt_fingerprint")
        == run_receipt.get("receipt_fingerprint")
        and complete.get("target_permutation_status")
        == permutation.get("status")
    ):
        raise RuntimeError("control preflight cross-file bindings disagree")
    expected_static_gate = (
        permutation.get("status") == TARGET_PERMUTATION_READY
    )
    if (
        complete.get("status") != "complete"
        or complete.get("matched_controls_static_preflight_pass")
        is not expected_static_gate
    ):
        raise RuntimeError("control preflight completion/gate status changed")

    basis_payload = dct.get("basis")
    if (
        not isinstance(basis_payload, Mapping)
        or stable_fingerprint(dict(basis_payload))
        != dct.get("basis_fingerprint")
    ):
        raise RuntimeError("DCT basis fingerprint changed")

    status = permutation.get("status")
    if status not in (
        TARGET_PERMUTATION_READY,
        TARGET_PERMUTATION_INCONCLUSIVE,
    ):
        raise RuntimeError("unknown control preflight status")
    assignments = permutation.get("assignments")
    donor_marginal = permutation.get("donor_marginal")
    if not isinstance(assignments, list) or not isinstance(
        donor_marginal, list
    ):
        raise RuntimeError("permutation assignment accounting is malformed")
    if any(not isinstance(row, Mapping) for row in donor_marginal):
        raise RuntimeError("donor marginal rows must be JSON objects")
    if stable_fingerprint(assignments) != permutation.get(
        "assignment_fingerprint"
    ):
        raise RuntimeError("target assignment fingerprint changed")
    if stable_fingerprint(donor_marginal) != permutation.get(
        "donor_marginal_fingerprint"
    ):
        raise RuntimeError("donor marginal fingerprint changed")
    if status == TARGET_PERMUTATION_READY:
        if not (
            permutation.get("source_disjoint") is True
            and permutation.get("fixed_point_free") is True
            and permutation.get("full_donor_marginal") is True
            and len(assignments) == expected_clean_pair_count
            and len(donor_marginal) == expected_clean_pair_count
            and all(
                row.get("assignment_count") == 1
                for row in donor_marginal
            )
        ):
            raise RuntimeError("ready target permutation postconditions changed")
    elif assignments or permutation.get("reason_code") is None:
        raise RuntimeError("inconclusive target permutation is malformed")
    materialization = permutation.get("training_materialization_contract")
    if not isinstance(materialization, Mapping) or any(
        materialization.get(field) is not True
        for field in (
            "recipient_batch_pair_ids_must_bind_exact_assignments",
            "donor_pair_id_must_resolve_in_frozen_catalog",
            "donor_target_must_be_clean_increment",
            "donor_target_fingerprint_must_match_assignment",
            "unbound_external_permuted_labels_forbidden",
            "runtime_training_binding_required_before_control_training",
        )
    ) or materialization.get(
        "runtime_training_binding_implemented_by_preflight"
    ) is not False:
        raise RuntimeError("target-permutation runtime binding contract changed")

    policy = run_receipt.get("execution_policy")
    if not isinstance(policy, Mapping) or any(
        policy.get(field) is not False
        for field in (
            "training_performed",
            "formal_800_epoch_training_authorized",
            "d_v_accessed",
            "d_t_accessed",
            "model_modified",
            "decoder_forward_performed",
            "optimizer_step_performed",
            "calibration_performed",
            "inference_performed",
            "full_cure_started",
            "backbone_integration_performed",
        )
    ):
        raise RuntimeError("control preflight execution boundary changed")
    return PublishedControlPreflight(
        root=root,
        status=str(status),
        catalog_fingerprint=str(expected_catalog_fingerprint),
        protocol_fingerprint=str(expected_protocol_fingerprint),
        contracts_fingerprint=str(contracts["receipt_fingerprint"]),
        dct_basis_fingerprint=str(dct["basis_fingerprint"]),
        permutation_fingerprint=str(permutation["plan_fingerprint"]),
        run_receipt_fingerprint=str(run_receipt["receipt_fingerprint"]),
        complete_fingerprint=str(complete["complete_fingerprint"]),
    )
