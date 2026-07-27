"""One-shot formal D_V reveal for the real PFCR CURE-Lite model.

The runner is intentionally closed:

* it accepts exactly the two strictly published PFCR attempts for seeds
  42 and 43, never a raw decoder artifact;
* it reuses the already frozen common D_V comparison protocol and all twelve
  comparator evidence rows from the published paired Wave-A reveal;
* it materializes the D_V bundle exactly once and evaluates the two PFCR
  attempts sequentially on the frozen seed-to-device mapping;
* it has no D_T, threshold, budget, seed, resume, or overwrite interface.

Every non-D_V input is authenticated before the D_V paths are opened.  A
create-only staging directory is claimed before D_V materialization.
``COMPLETE.json`` is written last and the finished directory is published by
an atomic no-replace rename.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import json
import os
from pathlib import Path
from typing import Any, Mapping

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch

from ..cache.schema import file_sha256, stable_fingerprint
from ..data import ManifestImageDataset, PreprocessConfig
from ..phase_resolved_real_cache import (
    PFCRRealCacheAdapter,
    adapt_pfcr_d_r_cache,
)
from ..splits import load_and_validate_manifest
from .cache_pipeline import (
    LoadedDVCacheBundle,
    load_d_r_cache_bundle,
    load_d_v_cache_bundle,
)
from .paired_formal_decision import FormalMethodEvidence
from .paired_formal_evaluation import (
    FrozenComparisonProtocol,
    load_frozen_comparison_protocol,
)
from .paired_formal_wave_reveal import (
    PublishedWaveAReveal,
    load_published_wave_a_reveal,
)
from .phase_resolved_real_formal_decision import (
    PFCR_FORMAL_COMPARATORS,
    PFCR_FORMAL_SEEDS,
    assess_pfcr_formal_d_v_gate,
)
from .phase_resolved_real_formal_evaluation import (
    PFCRFormalDVResult,
    load_pfcr_formal_d_v_result,
    select_and_evaluate_pfcr_formal_method,
)
from .phase_resolved_real_formal_runner import (
    PublishedPFCRRealFormalAttempt,
    load_pfcr_real_formal_attempt,
)


PFCR_REAL_DV_REVEAL_CONFIG_SCHEMA = (
    "cure-lite-pfcr-real-formal-d-v-reveal-config-v1"
)
PFCR_REAL_DV_REVEAL_RECEIPT_SCHEMA = (
    "cure-lite-pfcr-real-formal-d-v-reveal-receipt-v1"
)
PFCR_REAL_DV_REVEAL_COMPLETE_SCHEMA = (
    "cure-lite-pfcr-real-formal-d-v-reveal-complete-v1"
)
PFCR_REAL_DV_REVEAL_PROTOCOL_ID = (
    "irstd1k-pfcr-real-formal-d-v-reveal-v1"
)
PFCR_REAL_DEVICE_MAP = {42: "cuda:0", 43: "cuda:2"}
_EXPECTED_BACKEND_POLICY = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "established_before_cuda_initialization": True,
    "torch_deterministic_algorithms": True,
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}
_EXPECTED_DECISION_GATE = {
    "schema_version": "cure-lite-pfcr-formal-d-v-gate-contract-v1",
    "development_seeds": [42, 43],
    "proposed_method": "PFCR",
    "comparators": [
        "Base@B",
        "F",
        "F×",
        "U",
        "paired_difference",
        "independent_endpoint",
    ],
    "population": {
        "images": 120,
        "total_targets": 170,
        "anchor_covered": 147,
        "anchor_misses": 23,
    },
    "per_seed_not_mean": True,
    "both_seeds_must_pass": True,
    "minimum_true_target_margin": 2,
    "minimum_recovered_anchor_miss_margin": 2,
    "all_methods_require_budget_pass": True,
    "all_methods_minimum_retention": 0.99,
    "pfcr_retention_must_equal": 1.0,
    "pixel_fa_maximum": 1.0e-4,
    "raw_background_fa_maximum": 1.0e-4,
    "fp_components_per_mp_maximum": 100.0,
    "pass_authorization": "FROZEN_CONFIRMATION_ONLY",
    "failure_action": "STOP_AND_PRESERVE_EVIDENCE",
    "authorizes_full_cure": False,
    "authorizes_cross_backbone": False,
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_IMPLEMENTATION_PATHS = (
    "cure_lite/cache/base_cache.py",
    "cure_lite/cache/schema.py",
    "cure_lite/cache/state_cache.py",
    "cure_lite/calibration.py",
    "cure_lite/calibration_ledger.py",
    "cure_lite/config.py",
    "cure_lite/data.py",
    "cure_lite/experiment/cache_pipeline.py",
    "cure_lite/experiment/evaluation_pipeline.py",
    "cure_lite/experiment/formal_evaluation.py",
    "cure_lite/experiment/paired_formal_decision.py",
    "cure_lite/experiment/paired_formal_evaluation.py",
    "cure_lite/experiment/paired_formal_wave_reveal.py",
    "cure_lite/experiment/phase_resolved_real_artifacts.py",
    "cure_lite/experiment/phase_resolved_real_evaluation.py",
    "cure_lite/experiment/phase_resolved_real_formal_decision.py",
    "cure_lite/experiment/phase_resolved_real_formal_evaluation.py",
    "cure_lite/experiment/phase_resolved_real_formal_runner.py",
    "cure_lite/experiment/phase_resolved_real_d_v_reveal.py",
    "cure_lite/instances.py",
    "cure_lite/matching.py",
    "cure_lite/metrics.py",
    "cure_lite/occupancy.py",
    "cure_lite/phase_resolved_real_cache.py",
    "cure_lite/phase_resolved_relation_decoder.py",
    "cure_lite/splits.py",
    "cure_lite/types.py",
    "tools/run_phase_resolved_relation_real_d_v_reveal.py",
)
_HEX = frozenset("0123456789abcdef")
_RESULT_DIR = "results"
_CONFIG_SNAPSHOT_NAME = "protocol_config.json"
_COMPARISON_SNAPSHOT_NAME = "comparison_protocol.json"
_COMPARATOR_NAME = "frozen_comparator_evidence.json"
_DECISION_NAME = "decision.json"
_RECEIPT_NAME = "reveal_receipt.json"
_COMPLETE_NAME = "COMPLETE.json"
_INCOMPLETE_NAME = ".INCOMPLETE.json"

# Strict publication loading must always use the frozen decision function,
# even when a unit test replaces the single authoritative invocation below.
_recompute_pfcr_decision = assess_pfcr_formal_d_v_gate


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{name} contains non-finite number {value}")

    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=reject_duplicates,
                parse_constant=reject_nonfinite,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read strict {name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite reveal artifact {path}")
    with path.open("xb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fingerprinted(
    payload: Mapping[str, object],
    *,
    field: str,
) -> dict[str, object]:
    core = dict(payload)
    return {**core, field: stable_fingerprint(core)}


def _verify_fingerprint(
    value: Mapping[str, object],
    *,
    field: str,
    name: str,
) -> str:
    if field not in value:
        raise ValueError(f"{name} lacks {field}")
    core = dict(value)
    fingerprint = _digest(core.pop(field), name=f"{name}.{field}")
    if stable_fingerprint(core) != fingerprint:
        raise ValueError(f"{name} fingerprint mismatch")
    return fingerprint


def _repo_path(
    value: object,
    *,
    name: str,
    kind: str,
    must_exist: bool = True,
) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or any(part in {"", ".", ".."} for part in Path(value).parts)
    ):
        raise ValueError(f"{name} must be a canonical repository-relative path")
    candidate = (_REPO_ROOT / value).absolute()
    if candidate.is_symlink():
        raise ValueError(f"{name} may not be a symlink")
    if not must_exist:
        try:
            candidate.relative_to(_REPO_ROOT)
        except ValueError as error:
            raise ValueError(f"{name} escapes the repository") from error
        parent = candidate.parent
        if parent.is_symlink():
            raise ValueError(f"{name} parent may not be a symlink")
        resolved_parent = parent.resolve(strict=True)
        if (
            not resolved_parent.is_dir()
            or resolved_parent.is_symlink()
            or resolved_parent != parent
        ):
            raise ValueError(f"{name} parent is not canonical")
        return candidate
    resolved = candidate.resolve(strict=True)
    if resolved != candidate or resolved.is_symlink():
        raise ValueError(f"{name} is not a canonical repository path")
    if kind == "file" and not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    if kind == "directory" and not resolved.is_dir():
        raise ValueError(f"{name} must be a regular directory")
    return resolved


def _file_binding(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "repo_path",
        "file_sha256",
    }:
        raise ValueError(f"{name} fields are not canonical")
    path = value["repo_path"]
    if not isinstance(path, str):
        raise TypeError(f"{name}.repo_path must be a string")
    return {
        "repo_path": path,
        "file_sha256": _digest(
            value["file_sha256"],
            name=f"{name}.file_sha256",
        ),
    }


def _verify_file(path: Path, expected_sha: object, *, name: str) -> None:
    digest = _digest(expected_sha, name=f"{name} SHA256")
    if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
        raise RuntimeError(f"{name} differs from the frozen binding")


def _atomic_rename_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory rename is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            "final reveal output appeared before atomic publication"
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        f"{source} -> {target}",
    )


def _require_atomic_rename_noreplace() -> None:
    if getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None) is None:
        raise RuntimeError("atomic no-replace directory rename is unavailable")


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _validate_attempt_binding(value: object) -> dict[str, object]:
    fields = {
        "seed",
        "device",
        "repo_path",
        "complete_file_sha256",
        "complete_fingerprint",
        "run_receipt_fingerprint",
        "artifact_fingerprint",
        "artifact_receipt_sha256",
        "decoder_state_fingerprint",
        "cache_contract_fingerprint",
        "state_catalog_fingerprint",
        "lineage_allowlist_fingerprint",
        "formal_schedule_fingerprint",
        "preflight_result_fingerprint",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("PFCR formal attempt binding fields are not canonical")
    result = dict(value)
    seed = result["seed"]
    if seed not in PFCR_FORMAL_SEEDS:
        raise ValueError("PFCR formal attempt seed must be 42 or 43")
    if result["device"] != PFCR_REAL_DEVICE_MAP[seed]:
        raise ValueError("PFCR seed-to-device mapping changed")
    if not isinstance(result["repo_path"], str):
        raise TypeError("PFCR formal attempt repo_path must be a string")
    for field in fields - {"seed", "device", "repo_path"}:
        _digest(result[field], name=f"PFCR formal attempt {field}")
    return result


def _validate_comparator_binding(value: object) -> dict[str, object]:
    fields = {
        "repo_path",
        "complete_file_sha256",
        "decision_file_sha256",
        "complete_fingerprint",
        "decision_fingerprint",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("comparator Wave-A binding fields are not canonical")
    result = dict(value)
    if not isinstance(result["repo_path"], str):
        raise TypeError("comparator Wave-A repo_path must be a string")
    for field in fields - {"repo_path"}:
        _digest(result[field], name=f"comparator Wave-A {field}")
    return result


def _validate_cache_bundle_binding(
    value: object,
    *,
    split: str,
) -> dict[str, object]:
    expected = {
        "manifest",
        "index",
        "expected_base_fingerprint",
        "preprocessing",
        "preprocessing_fingerprint",
    }
    if split == "D_R":
        expected = {*expected, "expected_cache_contract_fingerprint"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{split} bundle binding fields are not canonical")
    result = dict(value)
    _file_binding(result["manifest"], name=f"{split} manifest")
    _file_binding(result["index"], name=f"{split} index")
    _digest(
        result["expected_base_fingerprint"],
        name=f"{split} expected_base_fingerprint",
    )
    if split == "D_R":
        _digest(
            result["expected_cache_contract_fingerprint"],
            name="D_R expected_cache_contract_fingerprint",
        )
    preprocessing = PreprocessConfig.from_fingerprint_payload(
        result["preprocessing"]
    )
    fingerprint = _digest(
        result["preprocessing_fingerprint"],
        name=f"{split} preprocessing_fingerprint",
    )
    if stable_fingerprint(preprocessing.fingerprint_payload()) != fingerprint:
        raise ValueError(f"{split} preprocessing fingerprint mismatch")
    return result


def validate_pfcr_real_d_v_reveal_config(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate the complete reveal contract without opening configured paths."""

    if not isinstance(value, Mapping):
        raise TypeError("PFCR D_V reveal config must be a mapping")
    config = dict(value)
    fields = {
        "schema_version",
        "protocol_id",
        "dataset",
        "model",
        "comparison_protocol",
        "attempts",
        "comparator_wave_a",
        "decision_gate",
        "backend_policy",
        "d_r_bundle",
        "d_v_bundle",
        "output_repo_path",
        "implementation_binding",
        "execution_policy",
        "config_fingerprint_scope",
        "config_fingerprint",
    }
    if set(config) != fields:
        raise ValueError("PFCR D_V reveal config fields are not canonical")
    unsigned = dict(config)
    fingerprint = unsigned.pop("config_fingerprint")
    _digest(fingerprint, name="config_fingerprint")
    if stable_fingerprint(unsigned) != fingerprint:
        raise ValueError("PFCR D_V reveal config fingerprint mismatch")
    if (
        config["schema_version"] != PFCR_REAL_DV_REVEAL_CONFIG_SCHEMA
        or config["protocol_id"] != PFCR_REAL_DV_REVEAL_PROTOCOL_ID
        or config["dataset"] != "IRSTD-1K"
        or config["model"] != "CURE-Lite"
        or config["config_fingerprint_scope"]
        != "all-fields-except-config_fingerprint"
    ):
        raise ValueError("PFCR D_V reveal protocol identity changed")
    raw_comparison = config["comparison_protocol"]
    if not isinstance(raw_comparison, Mapping) or set(raw_comparison) != {
        "repo_path",
        "file_sha256",
        "comparison_protocol_fingerprint",
    }:
        raise ValueError("comparison protocol binding fields are not canonical")
    _file_binding(
        {
            "repo_path": raw_comparison["repo_path"],
            "file_sha256": raw_comparison["file_sha256"],
        },
        name="comparison protocol",
    )
    _digest(
        raw_comparison["comparison_protocol_fingerprint"],
        name="comparison_protocol_fingerprint",
    )

    attempts = config["attempts"]
    if not isinstance(attempts, list):
        raise TypeError("attempts must be a list")
    normalized_attempts = tuple(
        _validate_attempt_binding(row) for row in attempts
    )
    if tuple(row["seed"] for row in normalized_attempts) != PFCR_FORMAL_SEEDS:
        raise ValueError("attempts must contain exact ordered seeds 42 and 43")
    _validate_comparator_binding(config["comparator_wave_a"])
    if (
        not isinstance(config["decision_gate"], Mapping)
        or dict(config["decision_gate"]) != _EXPECTED_DECISION_GATE
    ):
        raise ValueError("PFCR formal decision gate changed")
    if (
        not isinstance(config["backend_policy"], Mapping)
        or dict(config["backend_policy"]) != _EXPECTED_BACKEND_POLICY
    ):
        raise ValueError("PFCR deterministic backend policy changed")
    _validate_cache_bundle_binding(config["d_r_bundle"], split="D_R")
    _validate_cache_bundle_binding(config["d_v_bundle"], split="D_V")

    output = config["output_repo_path"]
    if (
        not isinstance(output, str)
        or not output
        or Path(output).is_absolute()
        or any(part in {"", ".", ".."} for part in Path(output).parts)
    ):
        raise ValueError("output_repo_path must be repository-relative")
    implementation = config["implementation_binding"]
    if (
        not isinstance(implementation, Mapping)
        or tuple(sorted(implementation))
        != tuple(sorted(_IMPLEMENTATION_PATHS))
    ):
        raise ValueError("PFCR reveal implementation inventory changed")
    for relative, digest in implementation.items():
        if not isinstance(relative, str):
            raise TypeError("implementation path must be a string")
        _digest(digest, name=f"implementation SHA for {relative}")

    expected_policy = {
        "create_only_output": True,
        "failed_staging_reuse": False,
        "resume": False,
        "overwrite": False,
        "runtime_split": "D_V",
        "allow_D_T": False,
        "d_r_bundle_materializations": 1,
        "d_v_bundle_materializations": 1,
        "pfcr_method_evaluations": 2,
        "frozen_comparator_evidence_rows": 12,
        "formal_decision_evidence_rows": 14,
        "formal_decisions": 1,
        "canonical_bundle_materializer_only": True,
        "canonical_evaluator_only": True,
        "seed_order": [42, 43],
        "seed_device_map": {"42": "cuda:0", "43": "cuda:2"},
        "sequential_evaluation": True,
        "shared_in_memory_d_v_bundle": True,
        "all_non_d_v_inputs_verified_before_d_v": True,
        "all_frozen_inputs_reverified_before_publication": True,
        "complete_written_last": True,
        "atomic_final_rename": True,
        "stdout_only_after_success": True,
    }
    if (
        not isinstance(config["execution_policy"], Mapping)
        or dict(config["execution_policy"]) != expected_policy
    ):
        raise ValueError("PFCR D_V reveal execution policy changed")
    return config


@dataclass(frozen=True, slots=True)
class _RevealConfigSeal:
    source_path: Path
    source_sha256: str
    payload: dict[str, object]


@dataclass(frozen=True)
class LoadedPFCRRealDVRevealConfig:
    source_path: Path
    source_sha256: str
    payload: Mapping[str, object]
    config_fingerprint: str
    _verification_token: object

    def _seal(self) -> _RevealConfigSeal:
        seal = self._verification_token
        if type(seal) is not _RevealConfigSeal:
            raise TypeError("PFCR reveal config must come from strict loader")
        if (
            seal.source_path != self.source_path
            or seal.source_sha256 != self.source_sha256
            or seal.payload is not self.payload
        ):
            raise TypeError("loaded PFCR reveal config fields were replaced")
        return seal

    def verify_unchanged(self) -> None:
        seal = self._seal()
        if (
            self.source_path.is_symlink()
            or file_sha256(self.source_path) != self.source_sha256
            or _strict_json(
                self.source_path,
                name="PFCR D_V reveal config",
            )
            != seal.payload
        ):
            raise RuntimeError("PFCR D_V reveal config changed on disk")
        validate_pfcr_real_d_v_reveal_config(self.payload)

    def __post_init__(self) -> None:
        self.verify_unchanged()
        if self.config_fingerprint != self.payload["config_fingerprint"]:
            raise ValueError("loaded PFCR reveal config fingerprint changed")


def load_pfcr_real_d_v_reveal_config(
    path: str | Path,
) -> LoadedPFCRRealDVRevealConfig:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError("PFCR D_V reveal config may not be a symlink")
    source = candidate.resolve(strict=True)
    payload = validate_pfcr_real_d_v_reveal_config(
        _strict_json(source, name="PFCR D_V reveal config")
    )
    seal = _RevealConfigSeal(source, file_sha256(source), payload)
    return LoadedPFCRRealDVRevealConfig(
        source_path=source,
        source_sha256=seal.source_sha256,
        payload=payload,
        config_fingerprint=str(payload["config_fingerprint"]),
        _verification_token=seal,
    )


@dataclass(frozen=True)
class _VerifiedInputs:
    protocol: FrozenComparisonProtocol
    comparison_protocol_payload: Mapping[str, object]
    attempts: tuple[PublishedPFCRRealFormalAttempt, ...]
    devices: tuple[torch.device, ...]
    comparator_reveal: PublishedWaveAReveal
    comparator_evidence: tuple[FormalMethodEvidence, ...]
    d_r_cache: PFCRRealCacheAdapter
    d_v_manifest_path: Path
    d_v_base_index_path: Path
    d_v_preprocessing: PreprocessConfig
    d_v_expected_base_fingerprint: str
    output: Path
    staging: Path


def _verify_implementation(
    config: LoadedPFCRRealDVRevealConfig,
) -> None:
    implementation = config.payload["implementation_binding"]
    assert isinstance(implementation, Mapping)
    for relative in _IMPLEMENTATION_PATHS:
        path = _repo_path(
            relative,
            name=f"implementation {relative}",
            kind="file",
        )
        _verify_file(
            path,
            implementation[relative],
            name=f"implementation {relative}",
        )


def _verify_runtime_devices() -> tuple[torch.device, ...]:
    """Resolve the frozen device map before any D_V path is opened."""

    if not torch.cuda.is_available():
        raise RuntimeError("PFCR formal D_V reveal requires CUDA")
    devices = tuple(
        torch.device(PFCR_REAL_DEVICE_MAP[seed])
        for seed in PFCR_FORMAL_SEEDS
    )
    if any(
        device.index is None
        or device.index >= torch.cuda.device_count()
        for device in devices
    ):
        raise RuntimeError("a frozen PFCR reveal CUDA device is unavailable")
    identities = []
    for device in devices:
        properties = torch.cuda.get_device_properties(device)
        identities.append(properties.name)
    if any("3090" not in name for name in identities):
        raise RuntimeError(
            "PFCR formal D_V reveal requires the frozen RTX 3090 devices"
        )
    return devices


def _configure_and_verify_backend_policy(
    *,
    require_uninitialized_cuda: bool,
) -> None:
    """Apply the frozen deterministic CUDA policy before any D_V access."""

    if require_uninitialized_cuda and torch.cuda.is_initialized():
        raise RuntimeError(
            "PFCR reveal requires backend policy before CUDA initialization"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG differs from protocol")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    actual = {
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
        "established_before_cuda_initialization": True,
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": (
            torch.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    if actual != _EXPECTED_BACKEND_POLICY:
        raise RuntimeError("unable to establish deterministic backend policy")


def _attempt_identity(
    attempt: PublishedPFCRRealFormalAttempt,
) -> dict[str, object]:
    if not isinstance(attempt, PublishedPFCRRealFormalAttempt):
        raise TypeError(
            "PFCR evaluation accepts only PublishedPFCRRealFormalAttempt"
        )
    artifact = attempt.artifact
    artifact.verify_unchanged()
    return {
        "seed": attempt.seed,
        "root": attempt.root,
        "run_receipt_fingerprint": attempt.run_receipt_fingerprint,
        "complete_fingerprint": attempt.complete_fingerprint,
        "artifact_fingerprint": artifact.artifact_fingerprint,
        "artifact_receipt_sha256": artifact.receipt_sha256,
        "decoder_state_fingerprint": artifact.decoder_state_fingerprint,
        "cache_contract_fingerprint": (
            artifact.config.cache_contract_fingerprint
        ),
        "state_catalog_fingerprint": (
            artifact.config.state_catalog_fingerprint
        ),
        "lineage_allowlist_fingerprint": (
            artifact.config.lineage_allowlist_fingerprint
        ),
        "formal_schedule_fingerprint": (
            artifact.config.formal_schedule_fingerprint
        ),
        "preflight_result_fingerprint": (
            artifact.config.preflight_result_fingerprint
        ),
    }


def _verify_attempt_binding(
    attempt: PublishedPFCRRealFormalAttempt,
    binding: Mapping[str, object],
) -> None:
    identity = _attempt_identity(attempt)
    expected = {
        "seed": binding["seed"],
        "root": attempt.root,
        "run_receipt_fingerprint": binding[
            "run_receipt_fingerprint"
        ],
        "complete_fingerprint": binding["complete_fingerprint"],
        "artifact_fingerprint": binding["artifact_fingerprint"],
        "artifact_receipt_sha256": binding[
            "artifact_receipt_sha256"
        ],
        "decoder_state_fingerprint": binding[
            "decoder_state_fingerprint"
        ],
        "cache_contract_fingerprint": binding[
            "cache_contract_fingerprint"
        ],
        "state_catalog_fingerprint": binding[
            "state_catalog_fingerprint"
        ],
        "lineage_allowlist_fingerprint": binding[
            "lineage_allowlist_fingerprint"
        ],
        "formal_schedule_fingerprint": binding[
            "formal_schedule_fingerprint"
        ],
        "preflight_result_fingerprint": binding[
            "preflight_result_fingerprint"
        ],
    }
    if identity != expected:
        raise RuntimeError(
            "published PFCR formal attempt differs from reveal binding"
        )


def _extract_comparator_evidence(
    reveal: PublishedWaveAReveal,
    *,
    comparison_protocol_fingerprint: str,
) -> tuple[FormalMethodEvidence, ...]:
    if not isinstance(reveal, PublishedWaveAReveal):
        raise TypeError("comparator source must be PublishedWaveAReveal")
    raw = reveal.decision.get("evidence")
    if not isinstance(raw, list) or any(
        not isinstance(row, Mapping) for row in raw
    ):
        raise RuntimeError("published Wave-A evidence is malformed")
    parsed = tuple(FormalMethodEvidence(**dict(row)) for row in raw)
    by_key = {(row.seed, row.method): row for row in parsed}
    expected_keys = tuple(
        (seed, method)
        for seed in PFCR_FORMAL_SEEDS
        for method in PFCR_FORMAL_COMPARATORS
    )
    if (
        len(parsed) != 12
        or len(by_key) != 12
        or set(by_key) != set(expected_keys)
        or {
            row.comparison_protocol_fingerprint for row in parsed
        }
        != {comparison_protocol_fingerprint}
    ):
        raise RuntimeError(
            "published Wave-A does not contain the exact twelve frozen "
            "PFCR comparator rows"
        )
    return tuple(by_key[key] for key in expected_keys)


def _resolve_file_binding(
    value: object,
    *,
    name: str,
) -> Path:
    binding = _file_binding(value, name=name)
    path = _repo_path(
        binding["repo_path"],
        name=name,
        kind="file",
    )
    _verify_file(path, binding["file_sha256"], name=name)
    return path


def _verify_snapshot_sources(
    snapshot: Mapping[str, object],
    comparator_rows: tuple[FormalMethodEvidence, ...],
) -> None:
    """Reload every frozen evidence source named by the config snapshot."""

    implementation = snapshot["implementation_binding"]
    assert isinstance(implementation, Mapping)
    for relative in _IMPLEMENTATION_PATHS:
        path = _repo_path(
            relative,
            name=f"snapshot implementation {relative}",
            kind="file",
        )
        _verify_file(
            path,
            implementation[relative],
            name=f"snapshot implementation {relative}",
        )

    comparison_binding = snapshot["comparison_protocol"]
    assert isinstance(comparison_binding, Mapping)
    comparison_path = _repo_path(
        comparison_binding["repo_path"],
        name="snapshot comparison protocol",
        kind="file",
    )
    _verify_file(
        comparison_path,
        comparison_binding["file_sha256"],
        name="snapshot comparison protocol",
    )
    protocol = load_frozen_comparison_protocol(comparison_path)
    if (
        protocol.comparison_protocol_fingerprint
        != comparison_binding["comparison_protocol_fingerprint"]
    ):
        raise RuntimeError("snapshot comparison protocol changed")

    attempts = snapshot["attempts"]
    assert isinstance(attempts, list)
    for binding in attempts:
        assert isinstance(binding, Mapping)
        root = _repo_path(
            binding["repo_path"],
            name="snapshot PFCR formal attempt",
            kind="directory",
        )
        _verify_file(
            root / _COMPLETE_NAME,
            binding["complete_file_sha256"],
            name="snapshot PFCR attempt COMPLETE",
        )
        _verify_file(
            root / "decoder_artifact" / "receipt.json",
            binding["artifact_receipt_sha256"],
            name="snapshot PFCR artifact receipt",
        )
        attempt = load_pfcr_real_formal_attempt(root)
        _verify_attempt_binding(attempt, binding)

    wave_binding = snapshot["comparator_wave_a"]
    assert isinstance(wave_binding, Mapping)
    wave_root = _repo_path(
        wave_binding["repo_path"],
        name="snapshot comparator Wave-A",
        kind="directory",
    )
    _verify_file(
        wave_root / _COMPLETE_NAME,
        wave_binding["complete_file_sha256"],
        name="snapshot comparator COMPLETE",
    )
    _verify_file(
        wave_root / _DECISION_NAME,
        wave_binding["decision_file_sha256"],
        name="snapshot comparator decision",
    )
    reveal = load_published_wave_a_reveal(wave_root)
    source_rows = _extract_comparator_evidence(
        reveal,
        comparison_protocol_fingerprint=(
            protocol.comparison_protocol_fingerprint
        ),
    )
    if (
        reveal.complete_fingerprint
        != wave_binding["complete_fingerprint"]
        or reveal.decision.get("decision_fingerprint")
        != wave_binding["decision_fingerprint"]
        or source_rows != comparator_rows
    ):
        raise RuntimeError("snapshot comparator Wave-A evidence changed")

    for split, key in (("D_R", "d_r_bundle"), ("D_V", "d_v_bundle")):
        binding = snapshot[key]
        assert isinstance(binding, Mapping)
        _resolve_file_binding(
            binding["manifest"],
            name=f"snapshot {split} manifest",
        )
        _resolve_file_binding(
            binding["index"],
            name=f"snapshot {split} index",
        )


def _load_d_r_cache(
    binding: Mapping[str, object],
) -> PFCRRealCacheAdapter:
    manifest_path = _resolve_file_binding(
        binding["manifest"],
        name="D_R manifest",
    )
    state_index_path = _resolve_file_binding(
        binding["index"],
        name="D_R state index",
    )
    preprocessing = PreprocessConfig.from_fingerprint_payload(
        binding["preprocessing"]
    )
    state_index = _strict_json(
        state_index_path,
        name="D_R state index",
    )
    if (
        state_index.get("preprocessing")
        != preprocessing.fingerprint_payload()
    ):
        raise RuntimeError(
            "D_R state-index preprocessing differs from reveal binding"
        )
    manifest = load_and_validate_manifest(manifest_path)
    dataset = ManifestImageDataset(
        manifest,
        "D_R",
        preprocessing,
        manifest_path=manifest_path,
    )
    bundle = load_d_r_cache_bundle(
        state_index_path,
        dataset,
        expected_base_fingerprint=str(
            binding["expected_base_fingerprint"]
        ),
    )
    cache = adapt_pfcr_d_r_cache(bundle)
    if (
        cache.contract.contract_fingerprint
        != binding["expected_cache_contract_fingerprint"]
    ):
        raise RuntimeError("D_R PFCR cache contract differs from reveal binding")
    return cache


def _verify_non_d_v_inputs(
    config: LoadedPFCRRealDVRevealConfig,
) -> _VerifiedInputs:
    """Authenticate all non-D_V sources before opening D_V paths."""

    config.verify_unchanged()
    _verify_implementation(config)
    _require_atomic_rename_noreplace()
    _configure_and_verify_backend_policy(
        require_uninitialized_cuda=True
    )
    devices = _verify_runtime_devices()

    comparison_binding = config.payload["comparison_protocol"]
    assert isinstance(comparison_binding, Mapping)
    comparison_path = _repo_path(
        comparison_binding["repo_path"],
        name="comparison protocol",
        kind="file",
    )
    _verify_file(
        comparison_path,
        comparison_binding["file_sha256"],
        name="comparison protocol",
    )
    protocol = load_frozen_comparison_protocol(comparison_path)
    if (
        protocol.comparison_protocol_fingerprint
        != comparison_binding["comparison_protocol_fingerprint"]
    ):
        raise RuntimeError("comparison protocol fingerprint differs")
    comparison_protocol_payload = _strict_json(
        comparison_path,
        name="comparison protocol",
    )

    raw_attempts = config.payload["attempts"]
    assert isinstance(raw_attempts, list)
    attempts: list[PublishedPFCRRealFormalAttempt] = []
    for binding in raw_attempts:
        assert isinstance(binding, Mapping)
        root = _repo_path(
            binding["repo_path"],
            name="published PFCR formal attempt",
            kind="directory",
        )
        _verify_file(
            root / _COMPLETE_NAME,
            binding["complete_file_sha256"],
            name="PFCR formal attempt COMPLETE",
        )
        _verify_file(
            root / "decoder_artifact" / "receipt.json",
            binding["artifact_receipt_sha256"],
            name="PFCR formal artifact receipt",
        )
        attempt = load_pfcr_real_formal_attempt(root)
        _verify_attempt_binding(attempt, binding)
        attempts.append(attempt)
    if tuple(attempt.seed for attempt in attempts) != PFCR_FORMAL_SEEDS:
        raise RuntimeError("loaded PFCR attempts are not ordered seeds 42/43")

    comparator_binding = config.payload["comparator_wave_a"]
    assert isinstance(comparator_binding, Mapping)
    comparator_root = _repo_path(
        comparator_binding["repo_path"],
        name="published comparator Wave-A reveal",
        kind="directory",
    )
    _verify_file(
        comparator_root / _COMPLETE_NAME,
        comparator_binding["complete_file_sha256"],
        name="comparator Wave-A COMPLETE",
    )
    _verify_file(
        comparator_root / _DECISION_NAME,
        comparator_binding["decision_file_sha256"],
        name="comparator Wave-A decision",
    )
    comparator_reveal = load_published_wave_a_reveal(comparator_root)
    if (
        comparator_reveal.complete_fingerprint
        != comparator_binding["complete_fingerprint"]
        or comparator_reveal.decision.get("decision_fingerprint")
        != comparator_binding["decision_fingerprint"]
        or comparator_reveal.decision.get(
            "comparison_protocol_fingerprint"
        )
        != protocol.comparison_protocol_fingerprint
    ):
        raise RuntimeError(
            "published comparator Wave-A differs from reveal binding"
        )
    comparator_evidence = _extract_comparator_evidence(
        comparator_reveal,
        comparison_protocol_fingerprint=(
            protocol.comparison_protocol_fingerprint
        ),
    )

    d_r_binding = config.payload["d_r_bundle"]
    assert isinstance(d_r_binding, Mapping)
    d_r_cache = _load_d_r_cache(d_r_binding)
    if {
        attempt.artifact.config.cache_contract_fingerprint
        for attempt in attempts
    } != {d_r_cache.contract.contract_fingerprint}:
        raise RuntimeError(
            "PFCR attempts and frozen D_R cache contract differ"
        )

    output = _repo_path(
        config.payload["output_repo_path"],
        name="PFCR D_V reveal output",
        kind="directory",
        must_exist=False,
    )
    staging = output.with_name(
        f".{output.name}.staging-{config.config_fingerprint}"
    )
    for path, name in ((output, "final output"), (staging, "staging output")):
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"{name} already exists; reveal has no resume or reuse"
            )
    protected = {
        comparison_path,
        comparator_root,
        *(attempt.root for attempt in attempts),
        d_r_cache.bundle.manifest_path,
        d_r_cache.bundle.base_index_path,
        d_r_cache.bundle.state_index_path,
    }
    if any(
        _paths_overlap(target, source)
        for target in (output, staging)
        for source in protected
    ):
        raise ValueError("PFCR reveal output overlaps a frozen input")

    # D_V paths are deliberately resolved only after all objects above pass.
    d_v_binding = config.payload["d_v_bundle"]
    assert isinstance(d_v_binding, Mapping)
    d_v_manifest_path = _resolve_file_binding(
        d_v_binding["manifest"],
        name="D_V manifest",
    )
    d_v_base_index_path = _resolve_file_binding(
        d_v_binding["index"],
        name="D_V base index",
    )
    if any(
        _paths_overlap(target, source)
        for target in (output, staging)
        for source in (d_v_manifest_path, d_v_base_index_path)
    ):
        raise ValueError("PFCR reveal output overlaps a D_V input")
    d_v_preprocessing = PreprocessConfig.from_fingerprint_payload(
        d_v_binding["preprocessing"]
    )
    return _VerifiedInputs(
        protocol=protocol,
        comparison_protocol_payload=comparison_protocol_payload,
        attempts=tuple(attempts),
        devices=devices,
        comparator_reveal=comparator_reveal,
        comparator_evidence=comparator_evidence,
        d_r_cache=d_r_cache,
        d_v_manifest_path=d_v_manifest_path,
        d_v_base_index_path=d_v_base_index_path,
        d_v_preprocessing=d_v_preprocessing,
        d_v_expected_base_fingerprint=str(
            d_v_binding["expected_base_fingerprint"]
        ),
        output=output,
        staging=staging,
    )


def _materialize_one_d_v_bundle(
    inputs: _VerifiedInputs,
) -> LoadedDVCacheBundle:
    manifest = load_and_validate_manifest(inputs.d_v_manifest_path)
    dataset = ManifestImageDataset(
        manifest,
        "D_V",
        inputs.d_v_preprocessing,
        manifest_path=inputs.d_v_manifest_path,
    )
    return load_d_v_cache_bundle(
        inputs.d_v_base_index_path,
        dataset,
        expected_base_fingerprint=(
            inputs.d_v_expected_base_fingerprint
        ),
    )


def _result_filename(result: PFCRFormalDVResult) -> str:
    return f"PFCR_seed{result.seed}.json"


def _claim_staging(
    config: LoadedPFCRRealDVRevealConfig,
    inputs: _VerifiedInputs,
) -> None:
    """Exclusively claim this one-shot reveal before D_V materialization."""

    inputs.staging.mkdir(parents=False, exist_ok=False)
    claim = _fingerprinted(
        {
            "schema_version": (
                "cure-lite-pfcr-real-formal-d-v-reveal-incomplete-v1"
            ),
            "execution_status": "claimed_before_D_V_materialization",
            "config_fingerprint": config.config_fingerprint,
            "comparison_protocol_fingerprint": (
                inputs.protocol.comparison_protocol_fingerprint
            ),
            "decision_gate_fingerprint": stable_fingerprint(
                config.payload["decision_gate"]
            ),
            "backend_policy": dict(config.payload["backend_policy"]),
            "seed_order": [42, 43],
            "seed_device_map": {"42": "cuda:0", "43": "cuda:2"},
            "resume_allowed": False,
            "directory_reuse_allowed": False,
            "results_authoritative": False,
            "D_T_accessed": False,
        },
        field="claim_fingerprint",
    )
    _write_new_json(inputs.staging / _INCOMPLETE_NAME, claim)
    _fsync_directory(inputs.staging)
    _fsync_directory(inputs.staging.parent)


def _verify_frozen_sources_unchanged(
    config: LoadedPFCRRealDVRevealConfig,
    inputs: _VerifiedInputs,
    bundle: LoadedDVCacheBundle,
) -> None:
    """Re-authenticate all frozen sources without rebuilding either cache."""

    config.verify_unchanged()
    _verify_implementation(config)
    _configure_and_verify_backend_policy(
        require_uninitialized_cuda=False
    )
    comparison_binding = config.payload["comparison_protocol"]
    assert isinstance(comparison_binding, Mapping)
    comparison_path = _repo_path(
        comparison_binding["repo_path"],
        name="comparison protocol",
        kind="file",
    )
    _verify_file(
        comparison_path,
        comparison_binding["file_sha256"],
        name="comparison protocol",
    )
    reloaded_protocol = load_frozen_comparison_protocol(comparison_path)
    if (
        reloaded_protocol.comparison_protocol_fingerprint
        != inputs.protocol.comparison_protocol_fingerprint
    ):
        raise RuntimeError("comparison protocol changed during PFCR reveal")

    raw_attempts = config.payload["attempts"]
    assert isinstance(raw_attempts, list)
    for original, binding in zip(
        inputs.attempts,
        raw_attempts,
        strict=True,
    ):
        assert isinstance(binding, Mapping)
        original.artifact.verify_unchanged()
        reloaded = load_pfcr_real_formal_attempt(original.root)
        _verify_attempt_binding(reloaded, binding)
        if _attempt_identity(reloaded) != _attempt_identity(original):
            raise RuntimeError("PFCR formal attempt changed during reveal")

    comparator_binding = config.payload["comparator_wave_a"]
    assert isinstance(comparator_binding, Mapping)
    _verify_file(
        inputs.comparator_reveal.root / _COMPLETE_NAME,
        comparator_binding["complete_file_sha256"],
        name="comparator Wave-A COMPLETE",
    )
    _verify_file(
        inputs.comparator_reveal.root / _DECISION_NAME,
        comparator_binding["decision_file_sha256"],
        name="comparator Wave-A decision",
    )
    reloaded_comparator = load_published_wave_a_reveal(
        inputs.comparator_reveal.root
    )
    reloaded_evidence = _extract_comparator_evidence(
        reloaded_comparator,
        comparison_protocol_fingerprint=(
            inputs.protocol.comparison_protocol_fingerprint
        ),
    )
    if (
        reloaded_comparator.complete_fingerprint
        != inputs.comparator_reveal.complete_fingerprint
        or reloaded_comparator.decision
        != inputs.comparator_reveal.decision
        or reloaded_evidence != inputs.comparator_evidence
    ):
        raise RuntimeError("comparator Wave-A changed during PFCR reveal")

    inputs.d_r_cache.verify_unchanged()
    bundle.verify_unchanged()
    inputs.protocol.verify_bundle(bundle)


@dataclass(frozen=True)
class PublishedPFCRRealDVReveal:
    """Strictly loaded, fully published two-seed PFCR reveal."""

    root: Path
    results: tuple[PFCRFormalDVResult, ...]
    comparator_evidence: tuple[FormalMethodEvidence, ...]
    decision: Mapping[str, object]
    complete_fingerprint: str

    def success_summary(self) -> dict[str, object]:
        return {
            "status": self.decision["status"],
            "model": "CURE-Lite",
            "output": str(self.root),
            "all_seeds_pass": self.decision["all_seeds_pass"],
            "seed_device_map": {"42": "cuda:0", "43": "cuda:2"},
            "decision_fingerprint": self.decision[
                "decision_fingerprint"
            ],
            "complete_fingerprint": self.complete_fingerprint,
            "D_T_accessed": False,
            "authorizes_full_cure": False,
        }


def _publish(
    config: LoadedPFCRRealDVRevealConfig,
    inputs: _VerifiedInputs,
    results: tuple[PFCRFormalDVResult, ...],
    decision: Mapping[str, object],
) -> PublishedPFCRRealDVReveal:
    """Write a complete in-memory result and atomically publish it."""

    if (
        not inputs.staging.is_dir()
        or inputs.staging.is_symlink()
        or {path.name for path in inputs.staging.iterdir()}
        != {_INCOMPLETE_NAME}
    ):
        raise RuntimeError("PFCR reveal staging claim changed")
    claim = _strict_json(
        inputs.staging / _INCOMPLETE_NAME,
        name="PFCR reveal staging claim",
    )
    _verify_fingerprint(
        claim,
        field="claim_fingerprint",
        name="PFCR reveal staging claim",
    )
    if (
        claim.get("schema_version")
        != "cure-lite-pfcr-real-formal-d-v-reveal-incomplete-v1"
        or claim.get("execution_status")
        != "claimed_before_D_V_materialization"
        or claim.get("config_fingerprint") != config.config_fingerprint
        or claim.get("comparison_protocol_fingerprint")
        != inputs.protocol.comparison_protocol_fingerprint
        or claim.get("decision_gate_fingerprint")
        != stable_fingerprint(config.payload["decision_gate"])
        or claim.get("backend_policy") != _EXPECTED_BACKEND_POLICY
        or claim.get("seed_order") != [42, 43]
        or claim.get("seed_device_map")
        != {"42": "cuda:0", "43": "cuda:2"}
        or claim.get("resume_allowed") is not False
        or claim.get("directory_reuse_allowed") is not False
        or claim.get("results_authoritative") is not False
        or claim.get("D_T_accessed") is not False
    ):
        raise RuntimeError("PFCR reveal staging claim semantics changed")

    _write_new_json(
        inputs.staging / _CONFIG_SNAPSHOT_NAME,
        config.payload,
    )
    _write_new_json(
        inputs.staging / _COMPARISON_SNAPSHOT_NAME,
        inputs.comparison_protocol_payload,
    )
    result_dir = inputs.staging / _RESULT_DIR
    result_dir.mkdir()
    for result in results:
        _write_new_json(
            result_dir / _result_filename(result),
            result.canonical_payload(),
        )
    comparator_binding = config.payload["comparator_wave_a"]
    assert isinstance(comparator_binding, Mapping)
    comparator_payload = _fingerprinted(
        {
            "schema_version": (
                "cure-lite-pfcr-frozen-comparator-evidence-v1"
            ),
            "comparison_protocol_fingerprint": (
                inputs.protocol.comparison_protocol_fingerprint
            ),
            "decision_gate_fingerprint": stable_fingerprint(
                config.payload["decision_gate"]
            ),
            "backend_policy": dict(config.payload["backend_policy"]),
            "source_wave": "A",
            "source_complete_file_sha256": comparator_binding[
                "complete_file_sha256"
            ],
            "source_decision_file_sha256": comparator_binding[
                "decision_file_sha256"
            ],
            "source_complete_fingerprint": (
                inputs.comparator_reveal.complete_fingerprint
            ),
            "source_decision_fingerprint": (
                inputs.comparator_reveal.decision[
                    "decision_fingerprint"
                ]
            ),
            "evidence": [
                row.canonical_payload()
                for row in inputs.comparator_evidence
            ],
            "evidence_row_count": 12,
            "D_T_accessed": False,
        },
        field="receipt_fingerprint",
    )
    _write_new_json(
        inputs.staging / _COMPARATOR_NAME,
        comparator_payload,
    )
    _write_new_json(inputs.staging / _DECISION_NAME, decision)

    artifact_files = {
        path.relative_to(inputs.staging).as_posix(): file_sha256(path)
        for path in sorted(inputs.staging.rglob("*"))
        if path.is_file() and path.name != _INCOMPLETE_NAME
    }
    receipt = _fingerprinted(
        {
            "schema_version": PFCR_REAL_DV_REVEAL_RECEIPT_SCHEMA,
            "execution_status": "complete_in_memory",
            "model": "CURE-Lite",
            "config_fingerprint": config.config_fingerprint,
            "config_file_sha256": config.source_sha256,
            "config_snapshot_file": _CONFIG_SNAPSHOT_NAME,
            "config_snapshot_sha256": file_sha256(
                inputs.staging / _CONFIG_SNAPSHOT_NAME
            ),
            "comparison_protocol_snapshot_file": (
                _COMPARISON_SNAPSHOT_NAME
            ),
            "comparison_protocol_snapshot_sha256": file_sha256(
                inputs.staging / _COMPARISON_SNAPSHOT_NAME
            ),
            "comparison_protocol_fingerprint": (
                inputs.protocol.comparison_protocol_fingerprint
            ),
            "decision_gate_fingerprint": stable_fingerprint(
                config.payload["decision_gate"]
            ),
            "backend_policy": dict(config.payload["backend_policy"]),
            "seed_order": [42, 43],
            "seed_device_map": {"42": "cuda:0", "43": "cuda:2"},
            "formal_attempt_complete_fingerprints": {
                str(attempt.seed): attempt.complete_fingerprint
                for attempt in inputs.attempts
            },
            "pfcr_result_receipt_fingerprints": {
                str(result.seed): result.receipt_fingerprint
                for result in results
            },
            "comparator_result_fingerprints": {
                f"{row.seed}:{row.method}": row.result_fingerprint
                for row in inputs.comparator_evidence
            },
            "comparator_evidence_receipt_fingerprint": (
                comparator_payload["receipt_fingerprint"]
            ),
            "comparator_source_complete_file_sha256": (
                comparator_binding["complete_file_sha256"]
            ),
            "comparator_source_decision_file_sha256": (
                comparator_binding["decision_file_sha256"]
            ),
            "comparator_source_complete_fingerprint": (
                comparator_binding["complete_fingerprint"]
            ),
            "comparator_source_decision_fingerprint": (
                comparator_binding["decision_fingerprint"]
            ),
            "decision_fingerprint": decision["decision_fingerprint"],
            "d_r_bundle_materialization_count": 1,
            "d_v_bundle_materialization_count": 1,
            "pfcr_method_evaluation_count": 2,
            "frozen_comparator_evidence_count": 12,
            "formal_decision_evidence_count": 14,
            "formal_decision_count": 1,
            "canonical_bundle_materializer_used": True,
            "canonical_evaluator_used": True,
            "live_sources_verified_before_d_v": True,
            "live_sources_reverified_after_d_v": True,
            "sequential_evaluation": True,
            "shared_in_memory_d_v_bundle": True,
            "D_T_accessed": False,
            "artifact_files_before_receipt": artifact_files,
        },
        field="receipt_fingerprint",
    )
    _write_new_json(inputs.staging / _RECEIPT_NAME, receipt)
    (inputs.staging / _INCOMPLETE_NAME).unlink()
    _fsync_directory(inputs.staging)

    published_files = {
        path.relative_to(inputs.staging).as_posix(): file_sha256(path)
        for path in sorted(inputs.staging.rglob("*"))
        if path.is_file()
    }
    complete = _fingerprinted(
        {
            "schema_version": PFCR_REAL_DV_REVEAL_COMPLETE_SCHEMA,
            "execution_status": "complete",
            "model": "CURE-Lite",
            "config_fingerprint": config.config_fingerprint,
            "comparison_protocol_fingerprint": (
                inputs.protocol.comparison_protocol_fingerprint
            ),
            "decision_gate_fingerprint": stable_fingerprint(
                config.payload["decision_gate"]
            ),
            "backend_policy": dict(config.payload["backend_policy"]),
            "reveal_receipt_fingerprint": (
                receipt["receipt_fingerprint"]
            ),
            "decision_fingerprint": decision["decision_fingerprint"],
            "artifact_files": published_files,
            "artifact_file_count": len(published_files),
            "complete_written_last": True,
            "atomic_final_rename": True,
            "canonical_bundle_materializer_used": True,
            "canonical_evaluator_used": True,
            "live_sources_verified_before_and_after_d_v": True,
            "resume_used": False,
            "overwrite_used": False,
            "D_T_accessed": False,
        },
        field="complete_fingerprint",
    )
    _write_new_json(inputs.staging / _COMPLETE_NAME, complete)
    _fsync_directory(result_dir)
    _fsync_directory(inputs.staging)
    validated = load_published_pfcr_real_d_v_reveal(inputs.staging)
    config.verify_unchanged()
    if inputs.output.exists() or inputs.output.is_symlink():
        raise FileExistsError("final PFCR reveal output appeared")
    _atomic_rename_noreplace(inputs.staging, inputs.output)
    _fsync_directory(inputs.output.parent)
    return PublishedPFCRRealDVReveal(
        root=inputs.output,
        results=validated.results,
        comparator_evidence=validated.comparator_evidence,
        decision=validated.decision,
        complete_fingerprint=validated.complete_fingerprint,
    )


def load_published_pfcr_real_d_v_reveal(
    output_dir: str | Path,
) -> PublishedPFCRRealDVReveal:
    """Strictly load and recompute a fully published PFCR reveal."""

    requested = Path(output_dir).expanduser()
    if requested.is_symlink():
        raise ValueError("published PFCR reveal may not be a symlink")
    root = requested.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("published PFCR reveal must be a regular directory")
    expected_top_level = {
        _RESULT_DIR,
        _CONFIG_SNAPSHOT_NAME,
        _COMPARISON_SNAPSHOT_NAME,
        _COMPARATOR_NAME,
        _DECISION_NAME,
        _RECEIPT_NAME,
        _COMPLETE_NAME,
    }
    top_level = {path.name: path for path in root.iterdir()}
    if set(top_level) != expected_top_level:
        raise RuntimeError("published PFCR reveal top-level inventory changed")
    if (
        top_level[_RESULT_DIR].is_symlink()
        or not top_level[_RESULT_DIR].is_dir()
        or any(
            top_level[name].is_symlink()
            or not top_level[name].is_file()
            for name in expected_top_level - {_RESULT_DIR}
        )
    ):
        raise RuntimeError("published PFCR reveal member types changed")
    expected_results = {
        f"{_RESULT_DIR}/PFCR_seed{seed}.json"
        for seed in PFCR_FORMAL_SEEDS
    }
    expected = {
        *expected_results,
        _CONFIG_SNAPSHOT_NAME,
        _COMPARISON_SNAPSHOT_NAME,
        _COMPARATOR_NAME,
        _DECISION_NAME,
        _RECEIPT_NAME,
        _COMPLETE_NAME,
    }
    members = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    directories = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir()
    }
    if (
        members != expected
        or directories != {_RESULT_DIR}
        or any(path.is_symlink() for path in root.rglob("*"))
        or {
            path.name for path in top_level[_RESULT_DIR].iterdir()
        }
        != {f"PFCR_seed{seed}.json" for seed in PFCR_FORMAL_SEEDS}
    ):
        raise RuntimeError("published PFCR reveal inventory changed")

    complete = _strict_json(
        root / _COMPLETE_NAME,
        name="PFCR reveal COMPLETE",
    )
    complete_fingerprint = _verify_fingerprint(
        complete,
        field="complete_fingerprint",
        name="PFCR reveal COMPLETE",
    )
    actual_files = {
        relative: file_sha256(root / relative)
        for relative in sorted(expected - {_COMPLETE_NAME})
    }
    if (
        complete.get("schema_version")
        != PFCR_REAL_DV_REVEAL_COMPLETE_SCHEMA
        or complete.get("execution_status") != "complete"
        or complete.get("model") != "CURE-Lite"
        or complete.get("decision_gate_fingerprint")
        != stable_fingerprint(_EXPECTED_DECISION_GATE)
        or complete.get("backend_policy") != _EXPECTED_BACKEND_POLICY
        or complete.get("artifact_files") != actual_files
        or complete.get("artifact_file_count") != len(actual_files)
        or complete.get("complete_written_last") is not True
        or complete.get("atomic_final_rename") is not True
        or complete.get("canonical_bundle_materializer_used") is not True
        or complete.get("canonical_evaluator_used") is not True
        or complete.get(
            "live_sources_verified_before_and_after_d_v"
        )
        is not True
        or complete.get("resume_used") is not False
        or complete.get("overwrite_used") is not False
        or complete.get("D_T_accessed") is not False
    ):
        raise RuntimeError("published PFCR reveal COMPLETE semantics changed")

    receipt = _strict_json(
        root / _RECEIPT_NAME,
        name="PFCR reveal receipt",
    )
    receipt_fingerprint = _verify_fingerprint(
        receipt,
        field="receipt_fingerprint",
        name="PFCR reveal receipt",
    )
    snapshot_path = root / _CONFIG_SNAPSHOT_NAME
    snapshot = validate_pfcr_real_d_v_reveal_config(
        _strict_json(
            snapshot_path,
            name="PFCR reveal protocol config snapshot",
        )
    )
    comparison_snapshot_path = root / _COMPARISON_SNAPSHOT_NAME
    comparison_protocol = load_frozen_comparison_protocol(
        comparison_snapshot_path
    )
    snapshot_d_v = snapshot["d_v_bundle"]
    snapshot_d_r = snapshot["d_r_bundle"]
    assert isinstance(snapshot_d_v, Mapping)
    assert isinstance(snapshot_d_r, Mapping)
    d_v_manifest_binding = snapshot_d_v["manifest"]
    d_v_index_binding = snapshot_d_v["index"]
    d_r_manifest_binding = snapshot_d_r["manifest"]
    assert isinstance(d_v_manifest_binding, Mapping)
    assert isinstance(d_v_index_binding, Mapping)
    assert isinstance(d_r_manifest_binding, Mapping)
    if (
        snapshot_d_v["expected_base_fingerprint"]
        != comparison_protocol.base_fingerprint
        or snapshot_d_r["expected_base_fingerprint"]
        != comparison_protocol.base_fingerprint
        or snapshot_d_v["preprocessing_fingerprint"]
        != comparison_protocol.preprocessing_fingerprint
        or snapshot_d_r["preprocessing_fingerprint"]
        != comparison_protocol.preprocessing_fingerprint
        or d_v_manifest_binding["file_sha256"]
        != comparison_protocol.manifest_file_sha256
        or d_r_manifest_binding["file_sha256"]
        != comparison_protocol.manifest_file_sha256
        or d_v_index_binding["file_sha256"]
        != comparison_protocol.d_v_base_index_sha256
    ):
        raise RuntimeError(
            "reveal config cache bindings differ from comparison snapshot"
        )
    expected_receipt_constants = {
        "schema_version": PFCR_REAL_DV_REVEAL_RECEIPT_SCHEMA,
        "execution_status": "complete_in_memory",
        "model": "CURE-Lite",
        "decision_gate_fingerprint": stable_fingerprint(
            _EXPECTED_DECISION_GATE
        ),
        "backend_policy": _EXPECTED_BACKEND_POLICY,
        "config_snapshot_file": _CONFIG_SNAPSHOT_NAME,
        "comparison_protocol_snapshot_file": (
            _COMPARISON_SNAPSHOT_NAME
        ),
        "seed_order": [42, 43],
        "seed_device_map": {"42": "cuda:0", "43": "cuda:2"},
        "d_r_bundle_materialization_count": 1,
        "d_v_bundle_materialization_count": 1,
        "pfcr_method_evaluation_count": 2,
        "frozen_comparator_evidence_count": 12,
        "formal_decision_evidence_count": 14,
        "formal_decision_count": 1,
        "canonical_bundle_materializer_used": True,
        "canonical_evaluator_used": True,
        "live_sources_verified_before_d_v": True,
        "live_sources_reverified_after_d_v": True,
        "sequential_evaluation": True,
        "shared_in_memory_d_v_bundle": True,
        "D_T_accessed": False,
    }
    if (
        any(
            receipt.get(name) != value
            for name, value in expected_receipt_constants.items()
        )
        or complete.get("reveal_receipt_fingerprint")
        != receipt_fingerprint
        or receipt.get("config_snapshot_sha256")
        != file_sha256(snapshot_path)
        or receipt.get("comparison_protocol_snapshot_sha256")
        != file_sha256(comparison_snapshot_path)
        or comparison_protocol.comparison_protocol_fingerprint
        != receipt.get("comparison_protocol_fingerprint")
        or comparison_protocol.comparison_protocol_fingerprint
        != snapshot["comparison_protocol"][
            "comparison_protocol_fingerprint"
        ]
        or receipt.get("config_fingerprint")
        != snapshot.get("config_fingerprint")
        or complete.get("config_fingerprint")
        != snapshot.get("config_fingerprint")
        or complete.get("config_fingerprint")
        != receipt.get("config_fingerprint")
        or complete.get("comparison_protocol_fingerprint")
        != receipt.get("comparison_protocol_fingerprint")
        or complete.get("decision_gate_fingerprint")
        != receipt.get("decision_gate_fingerprint")
        or complete.get("backend_policy") != receipt.get("backend_policy")
        or receipt.get("artifact_files_before_receipt")
        != {
            relative: actual_files[relative]
            for relative in sorted(
                expected - {_RECEIPT_NAME, _COMPLETE_NAME}
            )
        }
    ):
        raise RuntimeError("published PFCR reveal receipt semantics changed")

    results = tuple(
        load_pfcr_formal_d_v_result(
            root / _RESULT_DIR / f"PFCR_seed{seed}.json"
        )
        for seed in PFCR_FORMAL_SEEDS
    )
    snapshot_attempts = snapshot["attempts"]
    assert isinstance(snapshot_attempts, list)
    attempt_by_seed = {
        int(binding["seed"]): binding
        for binding in snapshot_attempts
        if isinstance(binding, Mapping)
    }
    expected_budget = comparison_protocol.budget
    threshold_grid = comparison_protocol.residual_thresholds
    if tuple(result.seed for result in results) != PFCR_FORMAL_SEEDS:
        raise RuntimeError("published PFCR result seed order changed")
    for result in results:
        binding = attempt_by_seed.get(result.seed)
        if (
            not isinstance(binding, Mapping)
            or result.execution_device
            != PFCR_REAL_DEVICE_MAP[result.seed]
            or result.comparison_protocol_fingerprint
            != receipt.get("comparison_protocol_fingerprint")
            or result.comparison_protocol_fingerprint
            != snapshot["comparison_protocol"][
                "comparison_protocol_fingerprint"
            ]
            or result.formal_attempt_complete_fingerprint
            != binding["complete_fingerprint"]
            or result.formal_attempt_run_receipt_fingerprint
            != binding["run_receipt_fingerprint"]
            or result.decoder_artifact_fingerprint
            != binding["artifact_fingerprint"]
            or result.decoder_receipt_sha256
            != binding["artifact_receipt_sha256"]
            or result.decoder_state_fingerprint
            != binding["decoder_state_fingerprint"]
            or result.cache_contract_fingerprint
            != binding["cache_contract_fingerprint"]
            or result.formal_schedule_fingerprint
            != binding["formal_schedule_fingerprint"]
            or result.state_catalog_fingerprint
            != binding["state_catalog_fingerprint"]
            or result.lineage_allowlist_fingerprint
            != binding["lineage_allowlist_fingerprint"]
            or result.preflight_result_fingerprint
            != binding["preflight_result_fingerprint"]
            or result.manifest_fingerprint
            != comparison_protocol.manifest_fingerprint
            or result.manifest_file_sha256
            != comparison_protocol.manifest_file_sha256
            or result.preprocessing_fingerprint
            != comparison_protocol.preprocessing_fingerprint
            or result.base_fingerprint
            != comparison_protocol.base_fingerprint
            or result.d_v_base_index_fingerprint
            != comparison_protocol.d_v_base_index_fingerprint
            or result.d_v_base_index_sha256
            != comparison_protocol.d_v_base_index_sha256
            or result.d_v_image_fingerprint
            != comparison_protocol.d_v_image_fingerprint
            or result.d_v_gt_fingerprint
            != comparison_protocol.d_v_gt_fingerprint
            or result.base_samples_fingerprint
            != comparison_protocol.base_samples_fingerprint
            or result.budget != expected_budget
            or (
                result.selected_threshold is not None
                and result.selected_threshold not in threshold_grid
            )
        ):
            raise RuntimeError(
                "published PFCR result differs from frozen config/protocol"
            )
    common_d_v_bindings = {
        (
            result.manifest_fingerprint,
            result.manifest_file_sha256,
            result.preprocessing_fingerprint,
            result.base_fingerprint,
            result.d_v_base_index_fingerprint,
            result.d_v_base_index_sha256,
            result.d_v_image_fingerprint,
            result.d_v_gt_fingerprint,
            result.base_samples_fingerprint,
        )
        for result in results
    }
    if (
        len(common_d_v_bindings) != 1
        or receipt.get("formal_attempt_complete_fingerprints")
        != {
            str(seed): attempt_by_seed[seed]["complete_fingerprint"]
            for seed in PFCR_FORMAL_SEEDS
        }
        or receipt.get("pfcr_result_receipt_fingerprints")
        != {
            str(result.seed): result.receipt_fingerprint
            for result in results
        }
    ):
        raise RuntimeError("published PFCR result bindings changed")

    comparator = _strict_json(
        root / _COMPARATOR_NAME,
        name="PFCR frozen comparator evidence",
    )
    comparator_fingerprint = _verify_fingerprint(
        comparator,
        field="receipt_fingerprint",
        name="PFCR frozen comparator evidence",
    )
    raw_comparators = comparator.get("evidence")
    if not isinstance(raw_comparators, list) or any(
        not isinstance(row, Mapping) for row in raw_comparators
    ):
        raise RuntimeError("published PFCR comparator evidence is malformed")
    comparator_rows = tuple(
        FormalMethodEvidence(**dict(row)) for row in raw_comparators
    )
    expected_keys = tuple(
        (seed, method)
        for seed in PFCR_FORMAL_SEEDS
        for method in PFCR_FORMAL_COMPARATORS
    )
    if (
        comparator.get("schema_version")
        != "cure-lite-pfcr-frozen-comparator-evidence-v1"
        or comparator.get("decision_gate_fingerprint")
        != receipt.get("decision_gate_fingerprint")
        or comparator.get("backend_policy") != receipt.get("backend_policy")
        or comparator.get("source_wave") != "A"
        or comparator.get("evidence_row_count") != 12
        or comparator.get("D_T_accessed") is not False
        or tuple((row.seed, row.method) for row in comparator_rows)
        != expected_keys
        or {
            row.comparison_protocol_fingerprint
            for row in comparator_rows
        }
        != {receipt.get("comparison_protocol_fingerprint")}
        or receipt.get("comparator_evidence_receipt_fingerprint")
        != comparator_fingerprint
        or receipt.get("comparator_source_complete_file_sha256")
        != comparator.get("source_complete_file_sha256")
        or receipt.get("comparator_source_decision_file_sha256")
        != comparator.get("source_decision_file_sha256")
        or receipt.get("comparator_source_complete_fingerprint")
        != comparator.get("source_complete_fingerprint")
        or receipt.get("comparator_source_decision_fingerprint")
        != comparator.get("source_decision_fingerprint")
        or receipt.get("comparator_result_fingerprints")
        != {
            f"{row.seed}:{row.method}": row.result_fingerprint
            for row in comparator_rows
        }
    ):
        raise RuntimeError("published PFCR comparator bindings changed")
    decision = _strict_json(
        root / _DECISION_NAME,
        name="PFCR formal D_V decision",
    )
    decision_fingerprint = _verify_fingerprint(
        decision,
        field="decision_fingerprint",
        name="PFCR formal D_V decision",
    )
    if (
        decision.get("schema_version")
        != "cure-lite-pfcr-formal-d-v-decision-v1"
        or decision.get("D_T_accessed") is not False
        or decision.get("protocol_fingerprint")
        != receipt.get("config_fingerprint")
        or decision.get("comparison_protocol_fingerprint")
        != receipt.get("comparison_protocol_fingerprint")
        or receipt.get("decision_fingerprint") != decision_fingerprint
        or complete.get("decision_fingerprint") != decision_fingerprint
    ):
        raise RuntimeError("published PFCR decision bindings changed")
    recomputed = _recompute_pfcr_decision(
        (
            *(result.to_formal_method_evidence() for result in results),
            *comparator_rows,
        ),
        protocol_fingerprint=str(receipt["config_fingerprint"]),
        comparison_protocol_fingerprint=str(
            receipt["comparison_protocol_fingerprint"]
        ),
    )
    if decision != recomputed:
        raise RuntimeError(
            "published PFCR decision differs from its fourteen evidence rows"
        )
    return PublishedPFCRRealDVReveal(
        root=root,
        results=results,
        comparator_evidence=comparator_rows,
        decision=decision,
        complete_fingerprint=complete_fingerprint,
    )


def run_pfcr_real_d_v_reveal(
    config_path: str | Path,
) -> PublishedPFCRRealDVReveal:
    """Execute the exact two-seed one-shot PFCR formal reveal."""

    config = load_pfcr_real_d_v_reveal_config(config_path)
    inputs = _verify_non_d_v_inputs(config)
    config.verify_unchanged()
    _claim_staging(config, inputs)
    bundle = _materialize_one_d_v_bundle(inputs)
    if not isinstance(bundle, LoadedDVCacheBundle):
        raise TypeError("bundle materializer must return LoadedDVCacheBundle")
    inputs.protocol.verify_bundle(bundle)

    results: list[PFCRFormalDVResult] = []
    for attempt, device in zip(
        inputs.attempts,
        inputs.devices,
        strict=True,
    ):
        result = select_and_evaluate_pfcr_formal_method(
            bundle,
            inputs.d_r_cache,
            attempt,
            comparison_protocol=inputs.protocol,
            device=device,
        )
        if not isinstance(result, PFCRFormalDVResult):
            raise TypeError("evaluator must return PFCRFormalDVResult")
        result.verify_unchanged()
        artifact = attempt.artifact
        if (
            result.seed != attempt.seed
            or result.execution_device != str(device)
            or result.comparison_protocol_fingerprint
            != inputs.protocol.comparison_protocol_fingerprint
            or result.formal_attempt_complete_fingerprint
            != attempt.complete_fingerprint
            or result.formal_attempt_run_receipt_fingerprint
            != attempt.run_receipt_fingerprint
            or result.decoder_artifact_fingerprint
            != artifact.artifact_fingerprint
            or result.decoder_receipt_sha256 != artifact.receipt_sha256
            or result.decoder_state_fingerprint
            != artifact.decoder_state_fingerprint
            or result.cache_contract_fingerprint
            != inputs.d_r_cache.contract.contract_fingerprint
        ):
            raise RuntimeError(
                "evaluated PFCR result differs from its frozen attempt"
            )
        results.append(result)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    if tuple(result.seed for result in results) != PFCR_FORMAL_SEEDS:
        raise RuntimeError("PFCR result pair is not ordered seeds 42/43")

    evidence = (
        *(result.to_formal_method_evidence() for result in results),
        *inputs.comparator_evidence,
    )
    decision = assess_pfcr_formal_d_v_gate(
        evidence,
        protocol_fingerprint=config.config_fingerprint,
        comparison_protocol_fingerprint=(
            inputs.protocol.comparison_protocol_fingerprint
        ),
    )
    _verify_frozen_sources_unchanged(config, inputs, bundle)
    return _publish(config, inputs, tuple(results), decision)


__all__ = [
    "PFCR_REAL_DEVICE_MAP",
    "PFCR_REAL_DV_REVEAL_COMPLETE_SCHEMA",
    "PFCR_REAL_DV_REVEAL_CONFIG_SCHEMA",
    "PFCR_REAL_DV_REVEAL_PROTOCOL_ID",
    "PFCR_REAL_DV_REVEAL_RECEIPT_SCHEMA",
    "LoadedPFCRRealDVRevealConfig",
    "PublishedPFCRRealDVReveal",
    "load_pfcr_real_d_v_reveal_config",
    "load_published_pfcr_real_d_v_reveal",
    "run_pfcr_real_d_v_reveal",
    "validate_pfcr_real_d_v_reveal_config",
]
