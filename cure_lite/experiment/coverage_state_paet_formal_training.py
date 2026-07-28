"""D_R-only Formal800 training core for the v21 PAET-BFA candidate.

This module deliberately stops at a trained in-memory candidate.  It does not
load ``D_V`` or ``D_T``, calibrate a threshold, run inference, or write a
checkpoint.  A later artifact layer may persist only the final model after
this result has passed its compute ledger.

Formal training is protected by a nominal authorization that binds:

* the exact completed v21 bounded-400 artifact set and source closure;
* the dataset-free and real-``D_R`` prerequisite receipts in that set;
* the complete, unbounded ``real_inputs.scalar_cache``;
* a deterministic seed-42 ``800 x 40`` schedule and its exposure gate;
* the exact PAET-BFA model configuration and current formal implementation.

The bounded result is interpreted only as a structural advancement.  Its
generic full-population zero-level gate is recorded separately and is not
misrepresented as a performance result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import cached_property
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from threading import Lock
import tarfile
from typing import Callable, Mapping

import torch

from ..cache.schema import (
    canonical_json,
    file_sha256,
    stable_fingerprint,
)
from ..coverage_state_phase_aligned_evidence_transport import (
    CSLF_PAET_EQUATION_POLICY,
    CSLF_PAET_FIELD_POLICY,
    CSLF_PAET_FLIP_POLICY,
    CSLF_PAET_TRANSPORT_POLICY,
    CURELitePhaseAlignedEvidenceTransportLevelSet,
    CoverageStatePhaseAlignedEvidenceTransportConfig,
)
from ..coverage_state_precomputed_cache import CoverageStateScalarCache
from ..coverage_state_schedule import (
    COVERAGE_STATE_FORMAL_EPOCHS,
    COVERAGE_STATE_FORMAL_STEPS_PER_EPOCH,
    CoverageStateScheduleConfig,
    CoverageStateTrainingSchedule,
    build_coverage_state_training_schedule,
    coverage_state_formal_exposure_gate,
)
from ..coverage_state_sobolev import CSLF_PMOPE_POLICY
from .coverage_state_bounded_runner import _deterministic_execution
from .coverage_state_paet_dataset_free import (
    COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS,
    COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE,
    COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT,
    COVERAGE_STATE_PAET_FORMAL_WIDTH,
    COVERAGE_STATE_PAET_MARGIN,
)
from .coverage_state_real_dr_inputs import CoverageStateRealDRInputs
from .coverage_state_training import (
    COVERAGE_STATE_FORMAL_SCOPE,
    CoverageStateMatchedTrainingConfig,
    CoverageStateMatchedTrainingResult,
    CoverageStateRunAuthorization,
    coverage_state_model_fingerprint,
    train_matched_coverage_state_paet_bfa_pmope_objectives,
)


COVERAGE_STATE_PAET_FORMAL_SEED = 42
COVERAGE_STATE_PAET_FORMAL_UPDATES = (
    COVERAGE_STATE_FORMAL_EPOCHS
    * COVERAGE_STATE_FORMAL_STEPS_PER_EPOCH
)
COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT = (
    "569b0fb97d819cf1281ca1d148227bc1c5e229b8301065cb536656b5e578e645"
)
COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT = (
    "abc1625c93dc9521b1e824ed4b2e685e867d755d8be5e7b1af3a4a5638240431"
)
COVERAGE_STATE_PAET_FORMAL_EXPOSURE_GATE_FINGERPRINT = (
    "c942578b53fd1ba9524cfcb28d504e9ea205f34af758bda6e9d3b466e5ce2c63"
)
COVERAGE_STATE_PAET_FORMAL_INITIAL_MODEL_FINGERPRINT = (
    "a4086bcffba4035984a8c334b3fa194910bcb7376a573f7f96ef8d36e097240d"
)
COVERAGE_STATE_PAET_FORMAL_RUN_ID = (
    "cure_lite_paet_bfa_v21_pmope_formal_800_seed42_r1"
)
COVERAGE_STATE_PAET_FORMAL_AUTHORIZATION_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal800-authorization-v1"
)
COVERAGE_STATE_PAET_FORMAL_RESULT_SCHEMA = (
    "cure-lite-paet-bfa-v21-formal800-training-result-v1"
)
COVERAGE_STATE_PAET_BOUNDED_SEAL_SCHEMA = (
    "cure-lite-paet-bfa-v21-bounded-artifact-seal-v1"
)

COVERAGE_STATE_PAET_BOUNDED_RUN_ID = (
    "cure_lite_paet_bfa_v21_pmope_bounded_400_r1"
)
COVERAGE_STATE_PAET_BOUNDED_RUN_REPO_PATH = (
    "runs/irstd1k_stage_a_seed42/"
    "cure_lite_paet_bfa_v21_pmope_bounded_400_r1"
)
COVERAGE_STATE_PAET_BOUNDED_COMPLETE_FINGERPRINT = (
    "ffc2e3c1cedb63931657f98323f16eee09d34735664da53b52ff40343b5290ef"
)
COVERAGE_STATE_PAET_BOUNDED_COMPLETE_FILE_SHA256 = (
    "8636e66096799be3766e13d7b135495127d4eeeb07c40daa4ae6a53b544eedb3"
)
COVERAGE_STATE_PAET_SOURCE_CLOSURE_MANIFEST_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_paet_bfa_v21_pmope_bounded_400_ffc2e3c1cedb.json"
)
COVERAGE_STATE_PAET_SOURCE_CLOSURE_MANIFEST_SHA256 = (
    "7669966b02202c9460a819b54c3fd01ad47fc9b1152c8435244a5f3ac2cfc5bc"
)
COVERAGE_STATE_PAET_SOURCE_CLOSURE_ARCHIVE_REPO_PATH = (
    "artifacts/source_closures/"
    "cure_lite_paet_bfa_v21_pmope_bounded_400_ffc2e3c1cedb.tar"
)
COVERAGE_STATE_PAET_SOURCE_CLOSURE_ARCHIVE_SHA256 = (
    "2acbf01c363467c2b7a94c0ee3ec1123c0a1dc3e380a7c8cc19b04ec6edb8776"
)

COVERAGE_STATE_PAET_FORMAL_IMPLEMENTATION_PATHS = (
    "cure_lite/coverage_state_level_set.py",
    "cure_lite/coverage_state_binary_flip_antisymmetric.py",
    "cure_lite/coverage_state_phase_aligned_evidence_transport.py",
    "cure_lite/coverage_state_phase_preserving.py",
    "cure_lite/coverage_state_device_cache.py",
    "cure_lite/coverage_state_schedule.py",
    "cure_lite/experiment/coverage_state_training.py",
    "cure_lite/train/coverage_state_fused_step.py",
    "cure_lite/experiment/coverage_state_paet_formal_training.py",
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _strict_json(path: Path, *, name: str) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{name} is missing or is not a regular file")

    def reject_constant(value: str) -> object:
        raise ValueError(f"{name} contains non-finite JSON value {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} is not strict JSON") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{name} must contain a JSON object")
    return payload


def _require_mapping(
    value: object,
    *,
    name: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def _safe_artifact_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError("bounded artifact path must be nonempty text")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise RuntimeError("bounded artifact path is not canonical")
    return value


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _current_formal_implementation_binding(
) -> tuple[tuple[str, str], ...]:
    root = _repository_root()
    result: list[tuple[str, str]] = []
    for relative in COVERAGE_STATE_PAET_FORMAL_IMPLEMENTATION_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(
                f"formal implementation path is invalid: {relative}"
            )
        result.append((relative, file_sha256(path)))
    return tuple(result)


def _formal_model_config_payload(
    config: CoverageStatePhaseAlignedEvidenceTransportConfig,
) -> dict[str, object]:
    if type(config) is not CoverageStatePhaseAlignedEvidenceTransportConfig:
        raise TypeError("formal v21 requires the exact PAET-BFA config")
    return {
        "model_class": (
            "CURELitePhaseAlignedEvidenceTransportLevelSet"
        ),
        "config": asdict(config),
        "expected_parameter_count": config.expected_parameter_count,
    }


def _scalar_cache_counts(
    cache: CoverageStateScalarCache,
) -> dict[str, int]:
    """Return population counts without serializing every cached tensor."""

    return {
        "natural_total": len(cache.natural_records),
        "pair_total": len(cache.pair_records),
        "clean_positive_optimization_eligible": len(
            cache.clean_positive_records
        ),
        "component_null_total": sum(
            value.record.pair_kind == "component_null"
            for value in cache.pair_records
        ),
        "component_null_optimization_eligible": len(
            cache.component_null_records
        ),
        "component_null_diagnostic_only": sum(
            value.record.pair_kind == "component_null"
            and not value.optimization_eligible
            for value in cache.pair_records
        ),
        "identity_null_diagnostic": sum(
            value.optimizer_role == "identity_diagnostic"
            for value in cache.pair_records
        ),
    }


class _VerifiedScalarCacheView(CoverageStateScalarCache):
    """Identity-bound view whose source was fully verified by its caller."""

    def __init__(self, source: CoverageStateScalarCache) -> None:
        if not isinstance(source, CoverageStateScalarCache):
            raise TypeError("source must be CoverageStateScalarCache")
        super().__init__(
            raw_catalog=source.raw_catalog,
            observability=source.observability,
            sobolev_config=source.sobolev_config,
            natural_records=source.natural_records,
            pair_records=source.pair_records,
            raw_catalog_fingerprint=source.raw_catalog_fingerprint,
            observability_receipt_fingerprint=(
                source.observability_receipt_fingerprint
            ),
            sobolev_config_fingerprint=(
                source.sobolev_config_fingerprint
            ),
        )
        object.__setattr__(self, "_verified_source", source)
        object.__setattr__(
            self,
            "_verified_cache_fingerprint",
            source.cache_fingerprint,
        )

    @property
    def cache_fingerprint(self) -> str:
        return self._verified_cache_fingerprint

    def verify_unchanged(self) -> None:
        source = self._verified_source
        if (
            self.raw_catalog is not source.raw_catalog
            or self.observability is not source.observability
            or self.sobolev_config is not source.sobolev_config
            or self.natural_records is not source.natural_records
            or self.pair_records is not source.pair_records
            or self.raw_catalog_fingerprint
            != source.raw_catalog_fingerprint
            or self.observability_receipt_fingerprint
            != source.observability_receipt_fingerprint
            or self.sobolev_config_fingerprint
            != source.sobolev_config_fingerprint
            or self.cache_fingerprint != source.cache_fingerprint
        ):
            raise RuntimeError("verified scalar-cache view changed")


def expected_coverage_state_paet_formal_config(
    real_inputs: CoverageStateRealDRInputs,
) -> CoverageStatePhaseAlignedEvidenceTransportConfig:
    """Return the only Formal800 PAET-BFA model configuration."""

    if not isinstance(real_inputs, CoverageStateRealDRInputs):
        raise TypeError("real_inputs must be CoverageStateRealDRInputs")
    real_inputs.verify_unchanged()
    return _formal_config_from_verified_real_inputs(real_inputs)


def _formal_config_from_verified_real_inputs(
    real_inputs: CoverageStateRealDRInputs,
) -> CoverageStatePhaseAlignedEvidenceTransportConfig:
    """Resolve the model after the caller has verified ``real_inputs``."""

    cache = real_inputs.scalar_cache
    if (
        real_inputs.source_binding.dataset != "IRSTD-1K"
        or real_inputs.source_binding.split != "D_R"
        or cache.raw_catalog.split != "D_R"
        or cache.sobolev_config.truncation_radius
        != COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE
        or cache.raw_catalog.feature_stride
        != COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE
        or cache.raw_catalog.natural_records[0].feature.shape[1]
        != COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS
    ):
        raise PermissionError(
            "formal PAET model requires the exact full IRSTD-1K D_R contract"
        )
    return CoverageStatePhaseAlignedEvidenceTransportConfig(
        feature_channels=COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS,
        feature_stride=COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE,
        width=COVERAGE_STATE_PAET_FORMAL_WIDTH,
    )


def _audit_source_closure(
    *,
    root: Path,
    closure: Mapping[str, object],
    sealed_implementation: Mapping[str, object],
) -> dict[str, object]:
    manifest_path = (
        root / COVERAGE_STATE_PAET_SOURCE_CLOSURE_MANIFEST_REPO_PATH
    )
    archive_path = (
        root / COVERAGE_STATE_PAET_SOURCE_CLOSURE_ARCHIVE_REPO_PATH
    )
    if (
        file_sha256(manifest_path)
        != COVERAGE_STATE_PAET_SOURCE_CLOSURE_MANIFEST_SHA256
        or file_sha256(archive_path)
        != COVERAGE_STATE_PAET_SOURCE_CLOSURE_ARCHIVE_SHA256
    ):
        raise RuntimeError("v21 source closure file digest changed")
    if (
        closure.get("schema_version")
        != "cure-lite-paet-bfa-v21-source-closure-v1"
        or closure.get("run_repo_path")
        != COVERAGE_STATE_PAET_BOUNDED_RUN_REPO_PATH
        or closure.get("complete_fingerprint")
        != COVERAGE_STATE_PAET_BOUNDED_COMPLETE_FINGERPRINT
        or closure.get("complete_file_sha256")
        != COVERAGE_STATE_PAET_BOUNDED_COMPLETE_FILE_SHA256
        or closure.get("archive_repo_path")
        != COVERAGE_STATE_PAET_SOURCE_CLOSURE_ARCHIVE_REPO_PATH
        or closure.get("archive_sha256")
        != COVERAGE_STATE_PAET_SOURCE_CLOSURE_ARCHIVE_SHA256
        or closure.get("source_file_count") != len(sealed_implementation)
        or closure.get("terminal_decision")
        != "PAET_BFA_V21_BOUNDED_400_GATE_PASS"
    ):
        raise RuntimeError("v21 source closure metadata changed")
    expected = {
        str(name): str(digest)
        for name, digest in sealed_implementation.items()
    }
    actual_archive: dict[str, str] = {}
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            members = archive.getmembers()
            for member in members:
                name = _safe_artifact_relative_path(member.name)
                if (
                    not member.isfile()
                    or name in actual_archive
                    or name not in expected
                ):
                    raise RuntimeError(
                        "v21 source closure member set changed"
                    )
                handle = archive.extractfile(member)
                if handle is None:
                    raise RuntimeError(
                        "v21 source closure member is unreadable"
                    )
                actual_archive[name] = _sha256_bytes(handle.read())
    except (OSError, tarfile.TarError) as error:
        raise RuntimeError("v21 source closure archive is invalid") from error
    if actual_archive != expected:
        raise RuntimeError("v21 source closure contents changed")
    current = {
        name: file_sha256(root / name) for name in sorted(expected)
    }
    if current != expected:
        raise RuntimeError(
            "current v21 inherited implementation differs from its closure"
        )
    return {
        "manifest_repo_path": (
            COVERAGE_STATE_PAET_SOURCE_CLOSURE_MANIFEST_REPO_PATH
        ),
        "manifest_sha256": (
            COVERAGE_STATE_PAET_SOURCE_CLOSURE_MANIFEST_SHA256
        ),
        "archive_repo_path": (
            COVERAGE_STATE_PAET_SOURCE_CLOSURE_ARCHIVE_REPO_PATH
        ),
        "archive_sha256": (
            COVERAGE_STATE_PAET_SOURCE_CLOSURE_ARCHIVE_SHA256
        ),
        "source_file_count": len(expected),
        "source_content_fingerprint": stable_fingerprint(expected),
        "current_inherited_implementation_matches_closure": True,
    }


def _audit_repository_bounded_evidence() -> dict[str, object]:
    """Rehash and semantically audit the completed v21 D_R evidence."""

    root = _repository_root()
    run_root = root / COVERAGE_STATE_PAET_BOUNDED_RUN_REPO_PATH
    complete_path = run_root / "COMPLETE.json"
    if (
        file_sha256(complete_path)
        != COVERAGE_STATE_PAET_BOUNDED_COMPLETE_FILE_SHA256
    ):
        raise RuntimeError("v21 bounded COMPLETE file digest changed")
    complete = _strict_json(complete_path, name="v21 bounded COMPLETE")
    artifacts = _require_mapping(
        complete.get("artifact_files"),
        name="v21 bounded artifact ledger",
    )
    if (
        complete.get("schema_version")
        != "cure-lite-paet-bfa-v21-pmope-bounded-400-run-v1"
        or complete.get("run_id") != COVERAGE_STATE_PAET_BOUNDED_RUN_ID
        or complete.get("status") != "complete"
        or complete.get("complete_fingerprint")
        != COVERAGE_STATE_PAET_BOUNDED_COMPLETE_FINGERPRINT
        or complete.get("artifact_file_count") != len(artifacts)
        or len(artifacts) != 17
        or complete.get("runtime_splits") != ["D_R"]
        or complete.get("D_V_accessed") is not False
        or complete.get("D_T_accessed") is not False
        or complete.get("performance_claim_supported") is not False
        or complete.get("single_attempt") is not True
        or complete.get("resume_allowed") is not False
        or complete.get("automatic_retry_allowed") is not False
    ):
        raise RuntimeError("v21 bounded COMPLETE contract changed")
    artifact_binding: dict[str, str] = {}
    for raw_relative, raw_digest in artifacts.items():
        relative = _safe_artifact_relative_path(raw_relative)
        if (
            not isinstance(raw_digest, str)
            or len(raw_digest) != 64
            or file_sha256(run_root / relative) != raw_digest
        ):
            raise RuntimeError(
                f"v21 bounded artifact changed: {relative}"
            )
        artifact_binding[relative] = raw_digest

    config = _strict_json(
        run_root / "receipts/config.json",
        name="v21 bounded config",
    )
    bounded = _strict_json(
        run_root / "receipts/bounded_result.json",
        name="v21 bounded result",
    )
    decision = _strict_json(
        run_root / "receipts/decision.json",
        name="v21 bounded decision",
    )
    dataset_free = _strict_json(
        run_root / "receipts/dataset_free.json",
        name="v21 dataset-free receipt",
    )
    dr_gate = _strict_json(
        run_root / "receipts/dr_gate.json",
        name="v21 D_R receipt",
    )
    inputs = _strict_json(
        run_root / "receipts/inputs.json",
        name="v21 input receipt",
    )
    bounded_result = _require_mapping(
        bounded.get("result"),
        name="v21 bounded result payload",
    )
    result_checks = _require_mapping(
        bounded_result.get("checks"),
        name="v21 bounded checks",
    )
    candidate = _require_mapping(
        bounded_result.get("candidate_diagnostic"),
        name="v21 bounded candidate diagnostic",
    )
    candidate_gates = _require_mapping(
        candidate.get("gates"),
        name="v21 generic population gates",
    )
    dataset_free_payload = _require_mapping(
        dataset_free.get("dataset_free"),
        name="v21 dataset-free payload",
    )
    dr_payload = _require_mapping(
        dr_gate.get("D_R_gate"),
        name="v21 D_R gate payload",
    )
    dr_checks = _require_mapping(
        dr_payload.get("checks"),
        name="v21 D_R checks",
    )
    real_inputs = _require_mapping(
        inputs.get("real_D_R_inputs"),
        name="v21 real D_R inputs",
    )
    real_fingerprints = _require_mapping(
        real_inputs.get("fingerprints"),
        name="v21 real D_R fingerprints",
    )
    real_counts = _require_mapping(
        real_inputs.get("counts"),
        name="v21 real D_R counts",
    )
    bounded_population = _require_mapping(
        inputs.get("bounded_population"),
        name="v21 bounded population",
    )
    bounded_population_fingerprint = inputs.get(
        "population_fingerprint"
    )
    model = _require_mapping(
        config.get("model"),
        name="v21 bounded model config",
    )
    implementation = _require_mapping(
        config.get("implementation"),
        name="v21 bounded implementation",
    )
    sealed_files = _require_mapping(
        implementation.get("files"),
        name="v21 bounded implementation files",
    )

    structural_advancement_passed = (
        complete.get("bounded_gate_passed") is True
        and complete.get("formal800_eligible") is True
        and decision.get("bounded_gate_passed") is True
        and decision.get("formal800_eligible") is True
        and result_checks.get(
            "predeclared_structural_advancement_gate"
        )
        is True
        and bool(result_checks)
        and all(value is True for value in result_checks.values())
    )
    generic_population_gate_passed = candidate_gates.get(
        "bounded_gate_passed"
    )
    dataset_free_passed = (
        dataset_free.get("all_pass") is True
        and dataset_free_payload.get("all_pass") is True
        and dataset_free.get("D_V_accessed") is False
        and dataset_free.get("D_T_accessed") is False
    )
    dr_passed = (
        dr_gate.get("all_pass") is True
        and dr_payload.get("all_pass") is True
        and bool(dr_checks)
        and all(value is True for value in dr_checks.values())
        and dr_payload.get("decision")
        == "PAET_D_R_IDENTIFIABILITY_PASS"
        and dr_gate.get("D_V_accessed") is False
        and dr_gate.get("D_T_accessed") is False
        and dr_gate.get("training_performed") is False
    )
    if (
        not structural_advancement_passed
        or generic_population_gate_passed is not False
        or not dataset_free_passed
        or not dr_passed
        or dataset_free.get("dataset_free_receipt_fingerprint")
        != dr_payload.get("dataset_free_receipt_fingerprint")
        or bounded.get("result_fingerprint")
        != decision.get("result_fingerprint")
        or bounded.get("result_fingerprint")
        != (
            "3b918ec82399ebf4235e2e5732ebd1613c996acb447c7c5ca"
            "0cad3a528a49aa6"
        )
        or real_inputs.get("dataset") != "IRSTD-1K"
        or real_inputs.get("split") != "D_R"
        or real_inputs.get("runtime_splits") != ["D_R"]
        or real_inputs.get("representation") != "scalar_max"
        or real_inputs.get("feature_stride")
        != COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE
        or real_fingerprints.get("scalar_cache")
        != COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        or stable_fingerprint(bounded_population)
        != bounded_population_fingerprint
        or bounded_population_fingerprint
        != (
            "1a53467d57bea595afcc1edd3330708d1dda39e0e2d606325"
            "e552e8993e7841c"
        )
        or bounded_population.get("bounded_cache_fingerprint")
        != (
            "c1627d7e838ff57e27f4753e689bd4075d2b8a8f4d2ca0075"
            "4c206092aaf66d8"
        )
        or bounded_population.get("source_cache_fingerprint")
        != COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        or bounded_population.get("split") != "D_R"
        or bounded_population.get("seed")
        != COVERAGE_STATE_PAET_FORMAL_SEED
        or bounded_population.get("role_count") != 16
        or bounded_population.get("D_V_accessed") is not False
        or bounded_population.get("D_T_accessed") is not False
        or inputs.get("D_V_accessed") is not False
        or inputs.get("D_T_accessed") is not False
        or dr_payload.get("real_inputs_build_fingerprint")
        is None
        or dr_payload.get("source_binding_fingerprint")
        != real_inputs.get("source_binding_fingerprint")
        or model.get("candidate") != "PAET-BFA"
        or model.get("candidate_objective") != "pmope_joint"
        or model.get("candidate_objective_policy") != CSLF_PMOPE_POLICY
        or model.get("feature_channels")
        != COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS
        or model.get("feature_stride")
        != COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE
        or model.get("width") != COVERAGE_STATE_PAET_FORMAL_WIDTH
        or model.get("parameter_count")
        != COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
        or model.get("field_threshold") != 0.0
        or model.get("threshold_search_performed") is not False
        or model.get("field_policy") != CSLF_PAET_FIELD_POLICY
        or model.get("equation_policy") != CSLF_PAET_EQUATION_POLICY
        or model.get("flip_policy") != CSLF_PAET_FLIP_POLICY
        or model.get("transport_policy") != CSLF_PAET_TRANSPORT_POLICY
        or _require_mapping(
            bounded_result.get("training"),
            name="v21 bounded training",
        ).get("common_initial_model_fingerprint")
        != COVERAGE_STATE_PAET_FORMAL_INITIAL_MODEL_FINGERPRINT
    ):
        raise RuntimeError("v21 bounded semantic evidence changed")

    closure_path = (
        root / COVERAGE_STATE_PAET_SOURCE_CLOSURE_MANIFEST_REPO_PATH
    )
    closure = _strict_json(
        closure_path,
        name="v21 source closure manifest",
    )
    source_closure = _audit_source_closure(
        root=root,
        closure=closure,
        sealed_implementation=sealed_files,
    )
    expected_scalar_counts = {
        "natural_total": real_counts.get("natural_records"),
        "pair_total": real_counts.get("pair_records"),
        "clean_positive_optimization_eligible": real_counts.get(
            "clean_positive_optimization_eligible"
        ),
        "component_null_total": (
            int(real_counts["component_null_optimization_eligible"])
            + int(real_counts["component_null_diagnostic_only"])
        ),
        "component_null_optimization_eligible": real_counts.get(
            "component_null_optimization_eligible"
        ),
        "component_null_diagnostic_only": real_counts.get(
            "component_null_diagnostic_only"
        ),
        "identity_null_diagnostic": real_counts.get(
            "identity_null_diagnostic"
        ),
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        for value in expected_scalar_counts.values()
    ):
        raise RuntimeError("v21 full D_R population counts are malformed")
    return {
        "schema_version": COVERAGE_STATE_PAET_BOUNDED_SEAL_SCHEMA,
        "run_id": COVERAGE_STATE_PAET_BOUNDED_RUN_ID,
        "run_repo_path": COVERAGE_STATE_PAET_BOUNDED_RUN_REPO_PATH,
        "complete_fingerprint": (
            COVERAGE_STATE_PAET_BOUNDED_COMPLETE_FINGERPRINT
        ),
        "complete_file_sha256": (
            COVERAGE_STATE_PAET_BOUNDED_COMPLETE_FILE_SHA256
        ),
        "artifact_binding": dict(sorted(artifact_binding.items())),
        "artifact_binding_fingerprint": stable_fingerprint(
            artifact_binding
        ),
        "bounded_result_fingerprint": bounded.get(
            "result_fingerprint"
        ),
        "structural_advancement_passed": True,
        "generic_population_gate_passed": False,
        "bounded_evidence_is_performance": False,
        "dataset_free_gate_passed": True,
        "dataset_free_receipt_fingerprint": dataset_free.get(
            "dataset_free_receipt_fingerprint"
        ),
        "D_R_gate_passed": True,
        "D_R_gate_evidence_fingerprint": dr_gate.get(
            "D_R_gate_evidence_fingerprint"
        ),
        "real_inputs_build_fingerprint": dr_payload.get(
            "real_inputs_build_fingerprint"
        ),
        "source_binding_fingerprint": real_inputs.get(
            "source_binding_fingerprint"
        ),
        "full_D_R_scalar_cache_fingerprint": real_fingerprints.get(
            "scalar_cache"
        ),
        "full_D_R_scalar_cache_counts": expected_scalar_counts,
        "bounded_population": bounded_population,
        "bounded_population_fingerprint": (
            bounded_population_fingerprint
        ),
        "model": {
            "feature_channels": model.get("feature_channels"),
            "feature_stride": model.get("feature_stride"),
            "width": model.get("width"),
            "parameter_count": model.get("parameter_count"),
            "field_policy": model.get("field_policy"),
            "equation_policy": model.get("equation_policy"),
            "flip_policy": model.get("flip_policy"),
            "transport_policy": model.get("transport_policy"),
            "candidate_objective": model.get("candidate_objective"),
            "candidate_objective_policy": model.get(
                "candidate_objective_policy"
            ),
            "field_threshold_hex": float(
                model.get("field_threshold")
            ).hex(),
        },
        "formal_frozen_coordinates": {
            "seed": COVERAGE_STATE_PAET_FORMAL_SEED,
            "epochs": COVERAGE_STATE_FORMAL_EPOCHS,
            "steps_per_epoch": (
                COVERAGE_STATE_FORMAL_STEPS_PER_EPOCH
            ),
            "updates": COVERAGE_STATE_PAET_FORMAL_UPDATES,
            "full_D_R_scalar_cache_fingerprint": (
                COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
            ),
            "schedule_fingerprint": (
                COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT
            ),
            "exposure_gate_fingerprint": (
                COVERAGE_STATE_PAET_FORMAL_EXPOSURE_GATE_FINGERPRINT
            ),
            "initial_model_fingerprint": (
                COVERAGE_STATE_PAET_FORMAL_INITIAL_MODEL_FINGERPRINT
            ),
            "coordinates_require_runtime_reconstruction": True,
        },
        "source_closure": source_closure,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "performance_claim_supported": False,
    }


@dataclass(frozen=True)
class CoverageStatePAETBoundedArtifactSeal:
    """Revalidatable read-only binding to the completed v21 evidence."""

    audit_canonical_json: str
    audit_fingerprint: str

    def __post_init__(self) -> None:
        try:
            payload = json.loads(self.audit_canonical_json)
        except json.JSONDecodeError as error:
            raise ValueError("bounded artifact seal is not canonical JSON") from error
        if (
            not isinstance(payload, dict)
            or canonical_json(payload) != self.audit_canonical_json
            or stable_fingerprint(payload) != self.audit_fingerprint
        ):
            raise ValueError("bounded artifact seal fingerprint changed")

    @property
    def payload(self) -> dict[str, object]:
        value = json.loads(self.audit_canonical_json)
        if not isinstance(value, dict):
            raise AssertionError("validated seal payload changed type")
        return value

    @property
    def structural_advancement_passed(self) -> bool:
        return self.payload["structural_advancement_passed"] is True

    @property
    def generic_population_gate_passed(self) -> bool:
        return self.payload["generic_population_gate_passed"] is True

    def verify_unchanged(self) -> None:
        current = _audit_repository_bounded_evidence()
        if (
            canonical_json(current) != self.audit_canonical_json
            or stable_fingerprint(current) != self.audit_fingerprint
        ):
            raise RuntimeError("bounded artifact evidence changed")


def load_repository_coverage_state_paet_bounded_artifact_seal(
) -> CoverageStatePAETBoundedArtifactSeal:
    """Load and rehash the exact v21 bounded artifacts without other splits."""

    payload = _audit_repository_bounded_evidence()
    result = CoverageStatePAETBoundedArtifactSeal(
        audit_canonical_json=canonical_json(payload),
        audit_fingerprint=stable_fingerprint(payload),
    )
    return result


class _FormalRunOnceSeal:
    """Process-local claim plus one engine authorization fast path."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._claimed = False
        self._engine_fast_path: (
            tuple[object, object, str] | None
        ) = None

    @property
    def claimed(self) -> bool:
        with self._lock:
            return self._claimed

    def claim(self) -> None:
        with self._lock:
            if self._claimed:
                raise PermissionError(
                    "formal PAET authorization was already consumed"
                )
            self._claimed = True

    def claim_and_arm_engine(
        self,
        *,
        cache: CoverageStateScalarCache,
        schedule: CoverageStateTrainingSchedule,
        scope: str,
    ) -> None:
        with self._lock:
            if self._claimed:
                raise PermissionError(
                    "formal PAET authorization was already consumed"
                )
            self._claimed = True
            self._engine_fast_path = (cache, schedule, scope)

    def consume_engine_fast_path(
        self,
        *,
        cache: CoverageStateScalarCache,
        schedule: CoverageStateTrainingSchedule,
        scope: str,
    ) -> bool:
        with self._lock:
            expected = self._engine_fast_path
            if expected is None:
                return False
            if (
                expected[0] is not cache
                or expected[1] is not schedule
                or expected[2] != scope
            ):
                return False
            self._engine_fast_path = None
            return True


@dataclass(frozen=True)
class _FormalPreparationSeal:
    """Identity seal issued only after the preparation-time full audit."""

    real_inputs: CoverageStateRealDRInputs
    scalar_cache: CoverageStateScalarCache
    bounded_artifact_seal: CoverageStatePAETBoundedArtifactSeal
    schedule: CoverageStateTrainingSchedule
    static_binding_fingerprint: str


def _formal_initial_model_fingerprint(
    config: CoverageStatePhaseAlignedEvidenceTransportConfig,
) -> str:
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(
            COVERAGE_STATE_PAET_FORMAL_SEED
        )
        model = CURELitePhaseAlignedEvidenceTransportLevelSet(config)
        return coverage_state_model_fingerprint(model)


def _formal_authorization_static_binding_payload(
    *,
    run_id: str,
    real_inputs: CoverageStateRealDRInputs,
    bounded_artifact_seal: CoverageStatePAETBoundedArtifactSeal,
    schedule: CoverageStateTrainingSchedule,
    exposure_gate_fingerprint: str,
    exposure_gate_checks: tuple[tuple[str, bool], ...],
    model_config_fingerprint: str,
    expected_parameter_count: int,
    expected_initial_model_fingerprint: str,
    formal_implementation_binding: tuple[tuple[str, str], ...],
    formal_implementation_fingerprint: str,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "real_inputs_build_fingerprint": real_inputs.build_fingerprint,
        "source_binding_fingerprint": (
            real_inputs.source_binding.binding_fingerprint
        ),
        "cache_fingerprint": real_inputs.scalar_cache.cache_fingerprint,
        "bounded_artifact_seal_fingerprint": (
            bounded_artifact_seal.audit_fingerprint
        ),
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "exposure_gate_fingerprint": exposure_gate_fingerprint,
        "exposure_gate_checks": dict(exposure_gate_checks),
        "model_config_fingerprint": model_config_fingerprint,
        "expected_parameter_count": expected_parameter_count,
        "expected_initial_model_fingerprint": (
            expected_initial_model_fingerprint
        ),
        "formal_implementation_binding": dict(
            formal_implementation_binding
        ),
        "formal_implementation_fingerprint": (
            formal_implementation_fingerprint
        ),
    }


@dataclass(frozen=True, eq=False)
class CoverageStatePAETFormal800Authorization(
    CoverageStateRunAuthorization,
):
    """Exact full-D_R authorization for one seed-42 Formal800 attempt."""

    run_id: str
    real_inputs: CoverageStateRealDRInputs
    bounded_artifact_seal: CoverageStatePAETBoundedArtifactSeal
    schedule: CoverageStateTrainingSchedule
    exposure_gate_fingerprint: str
    exposure_gate_checks: tuple[tuple[str, bool], ...]
    model_config_fingerprint: str
    expected_parameter_count: int
    expected_initial_model_fingerprint: str
    formal_implementation_binding: tuple[tuple[str, str], ...]
    formal_implementation_fingerprint: str
    _preparation_seal: _FormalPreparationSeal = field(
        repr=False,
        compare=False,
    )
    _run_once_seal: _FormalRunOnceSeal = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self._validate_lightweight_bindings()
        _ = self.authorization_fingerprint

    def _static_binding_payload(self) -> dict[str, object]:
        return _formal_authorization_static_binding_payload(
            run_id=self.run_id,
            real_inputs=self.real_inputs,
            bounded_artifact_seal=self.bounded_artifact_seal,
            schedule=self.schedule,
            exposure_gate_fingerprint=(
                self.exposure_gate_fingerprint
            ),
            exposure_gate_checks=self.exposure_gate_checks,
            model_config_fingerprint=self.model_config_fingerprint,
            expected_parameter_count=self.expected_parameter_count,
            expected_initial_model_fingerprint=(
                self.expected_initial_model_fingerprint
            ),
            formal_implementation_binding=(
                self.formal_implementation_binding
            ),
            formal_implementation_fingerprint=(
                self.formal_implementation_fingerprint
            ),
        )

    def _validate_lightweight_bindings(self) -> None:
        if self.run_id != COVERAGE_STATE_PAET_FORMAL_RUN_ID:
            raise PermissionError("formal PAET run_id is not frozen")
        if not isinstance(self.real_inputs, CoverageStateRealDRInputs):
            raise TypeError("real_inputs must be CoverageStateRealDRInputs")
        if not isinstance(
            self.bounded_artifact_seal,
            CoverageStatePAETBoundedArtifactSeal,
        ):
            raise TypeError("bounded_artifact_seal has the wrong type")
        if not isinstance(self.schedule, CoverageStateTrainingSchedule):
            raise TypeError("schedule must be CoverageStateTrainingSchedule")
        if not isinstance(
            self._preparation_seal,
            _FormalPreparationSeal,
        ):
            raise TypeError("formal preparation seal is invalid")
        if not isinstance(self._run_once_seal, _FormalRunOnceSeal):
            raise TypeError("formal run-once seal is invalid")
        if (
            self._preparation_seal.real_inputs is not self.real_inputs
            or self._preparation_seal.scalar_cache
            is not self.real_inputs.scalar_cache
            or self._preparation_seal.bounded_artifact_seal
            is not self.bounded_artifact_seal
            or self._preparation_seal.schedule is not self.schedule
            or self._preparation_seal.static_binding_fingerprint
            != stable_fingerprint(self._static_binding_payload())
        ):
            raise PermissionError(
                "formal PAET lightweight preparation seal changed"
            )
        payload = self.bounded_artifact_seal.payload
        frozen_config = (
            CoverageStatePhaseAlignedEvidenceTransportConfig(
                feature_channels=(
                    COVERAGE_STATE_PAET_FORMAL_FEATURE_CHANNELS
                ),
                feature_stride=(
                    COVERAGE_STATE_PAET_FORMAL_FEATURE_STRIDE
                ),
                width=COVERAGE_STATE_PAET_FORMAL_WIDTH,
            )
        )
        if (
            payload.get("structural_advancement_passed") is not True
            or payload.get("generic_population_gate_passed") is not False
            or payload.get("dataset_free_gate_passed") is not True
            or payload.get("D_R_gate_passed") is not True
            or self.real_inputs.scalar_cache.cache_fingerprint
            != COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
            or self.schedule.config
            != CoverageStateScheduleConfig.formal(
                seed=COVERAGE_STATE_PAET_FORMAL_SEED
            )
            or self.schedule.schedule_fingerprint
            != COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT
            or self.exposure_gate_fingerprint
            != COVERAGE_STATE_PAET_FORMAL_EXPOSURE_GATE_FINGERPRINT
            or not self.exposure_gate_checks
            or not all(
                value for _, value in self.exposure_gate_checks
            )
            or self.model_config_fingerprint
            != stable_fingerprint(
                _formal_model_config_payload(frozen_config)
            )
            or self.expected_parameter_count
            != COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
            or self.expected_initial_model_fingerprint
            != COVERAGE_STATE_PAET_FORMAL_INITIAL_MODEL_FINGERPRINT
            or self.formal_implementation_fingerprint
            != stable_fingerprint(
                dict(self.formal_implementation_binding)
            )
        ):
            raise PermissionError(
                "formal PAET lightweight frozen coordinates changed"
            )

    def _validate_bindings(self) -> None:
        self._validate_lightweight_bindings()
        self.real_inputs.verify_unchanged()
        verified_cache = _VerifiedScalarCacheView(
            self.real_inputs.scalar_cache
        )
        self.bounded_artifact_seal.verify_unchanged()
        payload = self.bounded_artifact_seal.payload
        config = _formal_config_from_verified_real_inputs(
            self.real_inputs
        )
        exposure = coverage_state_formal_exposure_gate(
            verified_cache,
            self.schedule,
        )
        counts = _scalar_cache_counts(
            self.real_inputs.scalar_cache
        )
        if (
            payload.get("structural_advancement_passed") is not True
            or payload.get("generic_population_gate_passed") is not False
            or payload.get("bounded_evidence_is_performance") is not False
            or payload.get("dataset_free_gate_passed") is not True
            or payload.get("D_R_gate_passed") is not True
            or payload.get("D_V_accessed") is not False
            or payload.get("D_T_accessed") is not False
            or payload.get("performance_claim_supported") is not False
            or self.real_inputs.build_fingerprint
            != payload.get("real_inputs_build_fingerprint")
            or self.real_inputs.source_binding.binding_fingerprint
            != payload.get("source_binding_fingerprint")
            or self.real_inputs.scalar_cache.cache_fingerprint
            != payload.get("full_D_R_scalar_cache_fingerprint")
            or self.real_inputs.scalar_cache.cache_fingerprint
            != COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
            or counts != payload.get("full_D_R_scalar_cache_counts")
            or self.schedule.cache_fingerprint
            != self.real_inputs.scalar_cache.cache_fingerprint
            or self.schedule.config
            != CoverageStateScheduleConfig.formal(
                seed=COVERAGE_STATE_PAET_FORMAL_SEED
            )
            or self.schedule.config.updates
            != COVERAGE_STATE_PAET_FORMAL_UPDATES
            or self.schedule.schedule_fingerprint
            != COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT
            or exposure.get("all_pass") is not True
            or tuple(sorted(exposure["checks"].items()))
            != self.exposure_gate_checks
            or exposure.get("gate_fingerprint")
            != self.exposure_gate_fingerprint
            or self.exposure_gate_fingerprint
            != COVERAGE_STATE_PAET_FORMAL_EXPOSURE_GATE_FINGERPRINT
            or self.model_config_fingerprint
            != stable_fingerprint(_formal_model_config_payload(config))
            or self.expected_parameter_count
            != COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
            or self.expected_parameter_count
            != config.expected_parameter_count
            or self.expected_initial_model_fingerprint
            != _formal_initial_model_fingerprint(config)
            or self.formal_implementation_binding
            != _current_formal_implementation_binding()
            or self.formal_implementation_fingerprint
            != stable_fingerprint(
                dict(self.formal_implementation_binding)
            )
        ):
            diagnostic_checks = {
                "structural_advancement": (
                    payload.get("structural_advancement_passed") is True
                ),
                "generic_population_status": (
                    payload.get("generic_population_gate_passed") is False
                ),
                "bounded_not_performance": (
                    payload.get("bounded_evidence_is_performance")
                    is False
                ),
                "dataset_free": (
                    payload.get("dataset_free_gate_passed") is True
                ),
                "D_R_gate": payload.get("D_R_gate_passed") is True,
                "split_scope": (
                    payload.get("D_V_accessed") is False
                    and payload.get("D_T_accessed") is False
                    and payload.get("performance_claim_supported")
                    is False
                ),
                "real_inputs_build": (
                    self.real_inputs.build_fingerprint
                    == payload.get("real_inputs_build_fingerprint")
                ),
                "source_binding": (
                    self.real_inputs.source_binding.binding_fingerprint
                    == payload.get("source_binding_fingerprint")
                ),
                "full_cache": (
                    self.real_inputs.scalar_cache.cache_fingerprint
                    == payload.get(
                        "full_D_R_scalar_cache_fingerprint"
                    )
                    == COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
                ),
                "full_cache_counts": (
                    counts
                    == payload.get("full_D_R_scalar_cache_counts")
                ),
                "schedule_cache": (
                    self.schedule.cache_fingerprint
                    == self.real_inputs.scalar_cache.cache_fingerprint
                ),
                "formal_budget": (
                    self.schedule.config
                    == CoverageStateScheduleConfig.formal(
                        seed=COVERAGE_STATE_PAET_FORMAL_SEED
                    )
                    and self.schedule.config.updates
                    == COVERAGE_STATE_PAET_FORMAL_UPDATES
                ),
                "formal_schedule": (
                    self.schedule.schedule_fingerprint
                    == COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT
                ),
                "exposure_pass": exposure.get("all_pass") is True,
                "exposure_checks": (
                    tuple(sorted(exposure["checks"].items()))
                    == self.exposure_gate_checks
                ),
                "exposure_fingerprint": (
                    exposure.get("gate_fingerprint")
                    == self.exposure_gate_fingerprint
                    == (
                        COVERAGE_STATE_PAET_FORMAL_EXPOSURE_GATE_FINGERPRINT
                    )
                ),
                "model_config": (
                    self.model_config_fingerprint
                    == stable_fingerprint(
                        _formal_model_config_payload(config)
                    )
                ),
                "parameter_count": (
                    self.expected_parameter_count
                    == COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
                    == config.expected_parameter_count
                ),
                "initial_model": (
                    self.expected_initial_model_fingerprint
                    == _formal_initial_model_fingerprint(config)
                ),
                "formal_implementation": (
                    self.formal_implementation_binding
                    == _current_formal_implementation_binding()
                    and self.formal_implementation_fingerprint
                    == stable_fingerprint(
                        dict(self.formal_implementation_binding)
                    )
                ),
            }
            failed = sorted(
                name
                for name, passed in diagnostic_checks.items()
                if not passed
            )
            raise PermissionError(
                "formal PAET authorization binding changed: "
                + ", ".join(failed)
            )

    @property
    def structural_advancement_passed(self) -> bool:
        return (
            self.bounded_artifact_seal
            .structural_advancement_passed
        )

    @property
    def generic_population_gate_passed(self) -> bool:
        return (
            self.bounded_artifact_seal
            .generic_population_gate_passed
        )

    @property
    def formal_training_authorized(self) -> bool:
        return (
            self.structural_advancement_passed
            and bool(self.exposure_gate_checks)
            and all(value for _, value in self.exposure_gate_checks)
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": (
                COVERAGE_STATE_PAET_FORMAL_AUTHORIZATION_SCHEMA
            ),
            "run_id": self.run_id,
            "scope": COVERAGE_STATE_FORMAL_SCOPE,
            "runtime_splits": ["D_R"],
            "bounded_artifact_seal_fingerprint": (
                self.bounded_artifact_seal.audit_fingerprint
            ),
            "bounded_evidence_interpretation": (
                "structural_advancement_only_not_performance"
            ),
            "structural_advancement_passed": (
                self.structural_advancement_passed
            ),
            "generic_population_gate_passed": (
                self.generic_population_gate_passed
            ),
            "dataset_free_gate_passed": True,
            "D_R_identifiability_gate_passed": True,
            "real_inputs_build_fingerprint": (
                self.real_inputs.build_fingerprint
            ),
            "source_binding_fingerprint": (
                self.real_inputs.source_binding.binding_fingerprint
            ),
            "full_D_R_scalar_cache_fingerprint": (
                self.real_inputs.scalar_cache.cache_fingerprint
            ),
            "full_D_R_scalar_cache_counts": (
                _scalar_cache_counts(self.real_inputs.scalar_cache)
            ),
            "schedule_fingerprint": self.schedule.schedule_fingerprint,
            "exposure_gate_fingerprint": (
                self.exposure_gate_fingerprint
            ),
            "exposure_gate_checks": dict(
                self.exposure_gate_checks
            ),
            "budget": {
                "seed": COVERAGE_STATE_PAET_FORMAL_SEED,
                "epochs": COVERAGE_STATE_FORMAL_EPOCHS,
                "steps_per_epoch": (
                    COVERAGE_STATE_FORMAL_STEPS_PER_EPOCH
                ),
                "updates": COVERAGE_STATE_PAET_FORMAL_UPDATES,
                "objectives": 1,
            },
            "model_config_fingerprint": (
                self.model_config_fingerprint
            ),
            "model_class": (
                "CURELitePhaseAlignedEvidenceTransportLevelSet"
            ),
            "expected_parameter_count": self.expected_parameter_count,
            "expected_initial_model_fingerprint": (
                self.expected_initial_model_fingerprint
            ),
            "candidate_objective": "pmope_joint",
            "candidate_objective_policy": CSLF_PMOPE_POLICY,
            "field_threshold_hex": 0.0.hex(),
            "threshold_search_performed": False,
            "formal_implementation_binding": dict(
                self.formal_implementation_binding
            ),
            "formal_implementation_fingerprint": (
                self.formal_implementation_fingerprint
            ),
            "training_contract": {
                "from_scratch": True,
                "process_local_single_attempt_claim": True,
                "cross_process_output_claim_required": True,
                "resume_allowed": False,
                "automatic_retry_allowed": False,
                "continuation_checkpoint_consumed": False,
                "checkpoint_policy": "final_model_only",
                "intermediate_checkpoint_saved": False,
                "optimizer_state_saved": False,
            },
            "formal_D_R_training_authorized": (
                self.formal_training_authorized
            ),
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
        }

    @cached_property
    def authorization_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify_unchanged(self) -> None:
        self._validate_bindings()
        if stable_fingerprint(
            self.canonical_payload()
        ) != self.authorization_fingerprint:
            raise RuntimeError("formal PAET authorization changed")

    def verify_model_config(
        self,
        model_config: CoverageStatePhaseAlignedEvidenceTransportConfig,
    ) -> None:
        if (
            type(model_config)
            is not CoverageStatePhaseAlignedEvidenceTransportConfig
            or stable_fingerprint(
                _formal_model_config_payload(model_config)
            )
            != self.model_config_fingerprint
            or model_config.expected_parameter_count
            != self.expected_parameter_count
        ):
            raise PermissionError(
                "formal PAET authorization rejects this model config"
            )

    def verify_for_run(
        self,
        *,
        cache: CoverageStateScalarCache,
        schedule: CoverageStateTrainingSchedule,
        scope: str,
    ) -> None:
        self._validate_lightweight_bindings()
        fast_path = self._run_once_seal.consume_engine_fast_path(
            cache=cache,
            schedule=schedule,
            scope=scope,
        )
        if not fast_path:
            if self._run_once_seal.claimed:
                raise PermissionError(
                    "formal PAET authorization was already consumed"
                )
            self.verify_unchanged()
        self._verify_run_coordinates(
            cache=cache,
            schedule=schedule,
            scope=scope,
        )
        if not fast_path:
            self._run_once_seal.claim()

    def _verify_run_coordinates(
        self,
        *,
        cache: CoverageStateScalarCache,
        schedule: CoverageStateTrainingSchedule,
        scope: str,
    ) -> None:
        if (
            scope != COVERAGE_STATE_FORMAL_SCOPE
            or cache is not self.real_inputs.scalar_cache
            or schedule is not self.schedule
            or cache.cache_fingerprint
            != self.real_inputs.scalar_cache.cache_fingerprint
            or schedule.schedule_fingerprint
            != self.schedule.schedule_fingerprint
            or not self.formal_training_authorized
        ):
            raise PermissionError(
                "formal PAET authorization rejects this run scope"
            )

    def claim_once(self) -> None:
        self.verify_unchanged()
        if not self.formal_training_authorized:
            raise PermissionError("formal PAET training is not authorized")
        self._run_once_seal.claim()

    def _claim_runner_once(
        self,
        *,
        cache: CoverageStateScalarCache,
        schedule: CoverageStateTrainingSchedule,
        scope: str,
    ) -> None:
        """Full-audit once, then arm one exact training-engine verification."""

        if self._run_once_seal.claimed:
            raise PermissionError(
                "formal PAET authorization was already consumed"
            )
        self.verify_unchanged()
        self._verify_run_coordinates(
            cache=cache,
            schedule=schedule,
            scope=scope,
        )
        self._run_once_seal.claim_and_arm_engine(
            cache=cache,
            schedule=schedule,
            scope=scope,
        )


def prepare_coverage_state_paet_formal_800_authorization(
    real_inputs: CoverageStateRealDRInputs,
    model_config: CoverageStatePhaseAlignedEvidenceTransportConfig,
    *,
    bounded_artifact_seal: (
        CoverageStatePAETBoundedArtifactSeal | None
    ) = None,
    run_id: str = COVERAGE_STATE_PAET_FORMAL_RUN_ID,
) -> CoverageStatePAETFormal800Authorization:
    """Build the full-D_R schedule and issue one exact Formal800 approval."""

    if not isinstance(real_inputs, CoverageStateRealDRInputs):
        raise TypeError("real_inputs must be CoverageStateRealDRInputs")
    if run_id != COVERAGE_STATE_PAET_FORMAL_RUN_ID:
        raise PermissionError("formal PAET run_id is not frozen")
    real_inputs.verify_unchanged()
    verified_cache = _VerifiedScalarCacheView(
        real_inputs.scalar_cache
    )
    if bounded_artifact_seal is None:
        seal = (
            load_repository_coverage_state_paet_bounded_artifact_seal()
        )
    else:
        seal = bounded_artifact_seal
        if not isinstance(
            seal,
            CoverageStatePAETBoundedArtifactSeal,
        ):
            raise TypeError("bounded_artifact_seal has the wrong type")
        seal.verify_unchanged()
    if not isinstance(seal, CoverageStatePAETBoundedArtifactSeal):
        raise TypeError("bounded_artifact_seal has the wrong type")
    expected_config = _formal_config_from_verified_real_inputs(
        real_inputs
    )
    if (
        type(model_config)
        is not CoverageStatePhaseAlignedEvidenceTransportConfig
        or model_config != expected_config
    ):
        raise PermissionError("formal PAET model config is not frozen")
    schedule = build_coverage_state_training_schedule(
        verified_cache,
        CoverageStateScheduleConfig.formal(
            seed=COVERAGE_STATE_PAET_FORMAL_SEED
        ),
    )
    exposure = coverage_state_formal_exposure_gate(
        verified_cache,
        schedule,
    )
    if exposure.get("all_pass") is not True:
        raise PermissionError("formal full-D_R exposure gate did not pass")
    implementation = _current_formal_implementation_binding()
    exposure_fingerprint = str(exposure["gate_fingerprint"])
    exposure_checks = tuple(sorted(exposure["checks"].items()))
    model_config_fingerprint = stable_fingerprint(
        _formal_model_config_payload(model_config)
    )
    initial_model_fingerprint = _formal_initial_model_fingerprint(
        model_config
    )
    implementation_fingerprint = stable_fingerprint(
        dict(implementation)
    )
    seal_payload = seal.payload
    counts = _scalar_cache_counts(real_inputs.scalar_cache)
    if (
        seal_payload.get("structural_advancement_passed") is not True
        or seal_payload.get("generic_population_gate_passed") is not False
        or seal_payload.get("dataset_free_gate_passed") is not True
        or seal_payload.get("D_R_gate_passed") is not True
        or real_inputs.build_fingerprint
        != seal_payload.get("real_inputs_build_fingerprint")
        or real_inputs.source_binding.binding_fingerprint
        != seal_payload.get("source_binding_fingerprint")
        or real_inputs.scalar_cache.cache_fingerprint
        != seal_payload.get("full_D_R_scalar_cache_fingerprint")
        or real_inputs.scalar_cache.cache_fingerprint
        != COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT
        or counts
        != seal_payload.get("full_D_R_scalar_cache_counts")
        or schedule.schedule_fingerprint
        != COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT
        or exposure_fingerprint
        != COVERAGE_STATE_PAET_FORMAL_EXPOSURE_GATE_FINGERPRINT
        or not exposure_checks
        or not all(value for _, value in exposure_checks)
        or model_config.expected_parameter_count
        != COVERAGE_STATE_PAET_FORMAL_PARAMETER_COUNT
    ):
        raise PermissionError(
            "formal PAET preparation coordinates did not pass"
        )
    static_payload = _formal_authorization_static_binding_payload(
        run_id=run_id,
        real_inputs=real_inputs,
        bounded_artifact_seal=seal,
        schedule=schedule,
        exposure_gate_fingerprint=exposure_fingerprint,
        exposure_gate_checks=exposure_checks,
        model_config_fingerprint=model_config_fingerprint,
        expected_parameter_count=model_config.expected_parameter_count,
        expected_initial_model_fingerprint=initial_model_fingerprint,
        formal_implementation_binding=implementation,
        formal_implementation_fingerprint=implementation_fingerprint,
    )
    preparation_seal = _FormalPreparationSeal(
        real_inputs=real_inputs,
        scalar_cache=real_inputs.scalar_cache,
        bounded_artifact_seal=seal,
        schedule=schedule,
        static_binding_fingerprint=stable_fingerprint(static_payload),
    )
    result = CoverageStatePAETFormal800Authorization(
        run_id=run_id,
        real_inputs=real_inputs,
        bounded_artifact_seal=seal,
        schedule=schedule,
        exposure_gate_fingerprint=exposure_fingerprint,
        exposure_gate_checks=exposure_checks,
        model_config_fingerprint=model_config_fingerprint,
        expected_parameter_count=model_config.expected_parameter_count,
        expected_initial_model_fingerprint=initial_model_fingerprint,
        formal_implementation_binding=implementation,
        formal_implementation_fingerprint=implementation_fingerprint,
        _preparation_seal=preparation_seal,
        _run_once_seal=_FormalRunOnceSeal(),
    )
    result.verify_model_config(model_config)
    return result


def _formal_result_checks(
    authorization: CoverageStatePAETFormal800Authorization,
    training: CoverageStateMatchedTrainingResult,
    *,
    training_invocations: int,
) -> tuple[tuple[str, bool], ...]:
    row = training.results[0] if len(training.results) == 1 else None
    entry = training.models[0] if len(training.models) == 1 else None
    model = entry[1] if entry is not None else None
    checks = {
        "authorization_consumed_exactly_once": (
            authorization._run_once_seal.claimed
            and training_invocations == 1
        ),
        "full_D_R_cache_and_formal_schedule": (
            row is not None
            and training.cache_fingerprint
            == authorization.real_inputs.scalar_cache.cache_fingerprint
            and row.cache_fingerprint == training.cache_fingerprint
            and training.schedule_fingerprint
            == authorization.schedule.schedule_fingerprint
            and row.schedule_fingerprint
            == training.schedule_fingerprint
        ),
        "fixed_seed42_800x40_compute_budget": (
            row is not None
            and row.seed == COVERAGE_STATE_PAET_FORMAL_SEED
            and row.epochs == COVERAGE_STATE_FORMAL_EPOCHS
            and row.steps_per_epoch
            == COVERAGE_STATE_FORMAL_STEPS_PER_EPOCH
            and row.completed_updates
            == COVERAGE_STATE_PAET_FORMAL_UPDATES
            and row.forward_calls == COVERAGE_STATE_PAET_FORMAL_UPDATES
            and row.backward_calls == COVERAGE_STATE_PAET_FORMAL_UPDATES
            and row.optimizer_steps
            == COVERAGE_STATE_PAET_FORMAL_UPDATES
            and row.logical_state_evaluations
            == 12 * COVERAGE_STATE_PAET_FORMAL_UPDATES
            and row.finite_state_audits
            == COVERAGE_STATE_PAET_FORMAL_UPDATES + 1
        ),
        "singleton_paet_bfa_pmope": (
            row is not None
            and entry is not None
            and row.objective == "pmope_joint"
            and row.objective_policy == CSLF_PMOPE_POLICY
            and entry[0] == "pmope_joint"
            and type(model)
            is CURELitePhaseAlignedEvidenceTransportLevelSet
            and stable_fingerprint(
                _formal_model_config_payload(model.config)
            )
            == authorization.model_config_fingerprint
            and sum(
                parameter.numel() for parameter in model.parameters()
            )
            == authorization.expected_parameter_count
        ),
        "from_scratch_initial_state": (
            row is not None
            and training.common_initial_model_fingerprint
            == authorization.expected_initial_model_fingerprint
            and row.initial_model_fingerprint
            == authorization.expected_initial_model_fingerprint
            and row.final_model_fingerprint
            != row.initial_model_fingerprint
        ),
        "single_final_model_output": (
            row is not None
            and entry is not None
            and len(training.results) == 1
            and len(training.models) == 1
            and tuple(value.objective for value in training.results)
            == ("pmope_joint",)
            and tuple(name for name, _ in training.models)
            == ("pmope_joint",)
        ),
        "fixed_level_set_decode_contract": (
            model is not None
            and type(model)
            is CURELitePhaseAlignedEvidenceTransportLevelSet
            and CSLF_PAET_FIELD_POLICY
            == "phase_aligned_evidence_transport_binary_flip_field_v1"
            and CSLF_PAET_EQUATION_POLICY
            == "bilinear_phase_aligned_shared_silu_energy_binary_odd_projection_v1"
            and CSLF_PAET_FLIP_POLICY
            == "exact_binary_current_center_phase_involution_v1"
            and CSLF_PAET_TRANSPORT_POLICY
            == "align_corners_false_bilinear_then_row_major_phase_pack_v1"
        ),
        "D_R_only_training_inputs": (
            authorization.real_inputs.source_binding.dataset == "IRSTD-1K"
            and authorization.real_inputs.source_binding.split == "D_R"
            and authorization.real_inputs.scalar_cache.raw_catalog.split
            == "D_R"
            and row is not None
            and row.cache_fingerprint
            == authorization.real_inputs.scalar_cache.cache_fingerprint
        ),
        "status_semantics_kept_separate": (
            authorization.structural_advancement_passed
            and not authorization.generic_population_gate_passed
        ),
    }
    return tuple(sorted(checks.items()))


@dataclass(frozen=True, eq=False)
class CoverageStatePAETFormal800RunResult:
    """One completed from-scratch Formal800 D_R training invocation."""

    authorization: CoverageStatePAETFormal800Authorization
    training: CoverageStateMatchedTrainingResult
    training_invocations: int
    checks: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(
                self.authorization,
                CoverageStatePAETFormal800Authorization,
            )
            or not isinstance(
                self.training,
                CoverageStateMatchedTrainingResult,
            )
            or self.training_invocations != 1
            or self.checks != tuple(sorted(self.checks))
            or len({name for name, _ in self.checks})
            != len(self.checks)
        ):
            raise ValueError("formal PAET training result is incomplete")
        self._validate_lightweight()

    @property
    def training_complete(self) -> bool:
        return bool(self.checks) and all(value for _, value in self.checks)

    @property
    def final_model(
        self,
    ) -> CURELitePhaseAlignedEvidenceTransportLevelSet:
        model = self.training.models[0][1]
        if type(model) is not CURELitePhaseAlignedEvidenceTransportLevelSet:
            raise RuntimeError("formal result model type changed")
        return model

    def _validate_lightweight(self) -> None:
        self.authorization._validate_lightweight_bindings()
        self.training.verify_unchanged()
        expected = _formal_result_checks(
            self.authorization,
            self.training,
            training_invocations=self.training_invocations,
        )
        if self.checks != expected:
            raise RuntimeError("formal PAET training result changed")

    def verify_unchanged(self) -> None:
        self.authorization.verify_unchanged()
        self.training.verify_unchanged()
        self._validate_lightweight()

    def canonical_payload(self) -> dict[str, object]:
        self._validate_lightweight()
        return {
            "schema_version": COVERAGE_STATE_PAET_FORMAL_RESULT_SCHEMA,
            "run_id": self.authorization.run_id,
            "runtime_splits": ["D_R"],
            "authorization_fingerprint": (
                self.authorization.authorization_fingerprint
            ),
            "structural_advancement_passed": (
                self.authorization.structural_advancement_passed
            ),
            "generic_population_gate_passed": (
                self.authorization.generic_population_gate_passed
            ),
            "bounded_evidence_interpretation": (
                "structural_advancement_only_not_performance"
            ),
            "training": self.training.canonical_payload(),
            "training_invocations": self.training_invocations,
            "checks": dict(self.checks),
            "failed_checks": [
                name for name, passed in self.checks if not passed
            ],
            "training_complete": self.training_complete,
            "field_threshold_hex": 0.0.hex(),
            "threshold_search_performed": False,
            "training_contract": {
                "from_scratch": True,
                "process_local_single_attempt_claim": True,
                "cross_process_output_claim_required": True,
                "resume_allowed": False,
                "automatic_retry_allowed": False,
                "checkpoint_policy": "final_model_only",
                "intermediate_checkpoint_saved": False,
                "optimizer_state_saved": False,
            },
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "performance_evaluation_performed": False,
            "performance_claim_supported": False,
            "full_CURE_authorized": False,
            "cross_backbone_authorized": False,
        }

    @property
    def result_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())


def run_coverage_state_paet_bfa_pmope_formal_800(
    authorization: CoverageStatePAETFormal800Authorization,
    model_config: CoverageStatePhaseAlignedEvidenceTransportConfig,
    *,
    device: torch.device | str,
    epoch_callback: (
        Callable[[str, Mapping[str, object]], None] | None
    ) = None,
) -> CoverageStatePAETFormal800RunResult:
    """Train once from scratch on full D_R; perform no evaluation."""

    if not isinstance(
        authorization,
        CoverageStatePAETFormal800Authorization,
    ):
        raise TypeError(
            "authorization must be CoverageStatePAETFormal800Authorization"
        )
    resolved_device = torch.device(device)
    if resolved_device != torch.device("cuda:0"):
        raise PermissionError("formal PAET training is frozen to cuda:0")
    if epoch_callback is not None and not callable(epoch_callback):
        raise TypeError("epoch_callback must be callable or None")
    authorization.verify_model_config(model_config)
    authorization._claim_runner_once(
        cache=authorization.real_inputs.scalar_cache,
        schedule=authorization.schedule,
        scope=COVERAGE_STATE_FORMAL_SCOPE,
    )
    training_invocations = 0
    with _deterministic_execution(device):
        training_invocations += 1
        training = (
            train_matched_coverage_state_paet_bfa_pmope_objectives(
                model_config,
                authorization.real_inputs.scalar_cache,
                authorization.schedule,
                config=CoverageStateMatchedTrainingConfig(
                    seed=COVERAGE_STATE_PAET_FORMAL_SEED
                ),
                device=resolved_device,
                authorization=authorization,
                epoch_callback=epoch_callback,
            )
        )
    if not isinstance(training, CoverageStateMatchedTrainingResult):
        raise RuntimeError("formal PAET trainer returned the wrong type")
    result = CoverageStatePAETFormal800RunResult(
        authorization=authorization,
        training=training,
        training_invocations=training_invocations,
        checks=_formal_result_checks(
            authorization,
            training,
            training_invocations=training_invocations,
        ),
    )
    if not result.training_complete:
        raise RuntimeError(
            "formal PAET training completed with an invalid ledger"
        )
    return result


__all__ = [
    "COVERAGE_STATE_PAET_FORMAL_AUTHORIZATION_SCHEMA",
    "COVERAGE_STATE_PAET_FORMAL_EXPOSURE_GATE_FINGERPRINT",
    "COVERAGE_STATE_PAET_FORMAL_FULL_CACHE_FINGERPRINT",
    "COVERAGE_STATE_PAET_FORMAL_INITIAL_MODEL_FINGERPRINT",
    "COVERAGE_STATE_PAET_FORMAL_RESULT_SCHEMA",
    "COVERAGE_STATE_PAET_FORMAL_RUN_ID",
    "COVERAGE_STATE_PAET_FORMAL_SCHEDULE_FINGERPRINT",
    "COVERAGE_STATE_PAET_FORMAL_SEED",
    "COVERAGE_STATE_PAET_FORMAL_UPDATES",
    "CoverageStatePAETBoundedArtifactSeal",
    "CoverageStatePAETFormal800Authorization",
    "CoverageStatePAETFormal800RunResult",
    "expected_coverage_state_paet_formal_config",
    "load_repository_coverage_state_paet_bounded_artifact_seal",
    "prepare_coverage_state_paet_formal_800_authorization",
    "run_coverage_state_paet_bfa_pmope_formal_800",
]
