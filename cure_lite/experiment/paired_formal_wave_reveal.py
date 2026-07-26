"""One-shot, create-only Wave-A reveal for formal paired CURE-Lite.

The runner has one deliberately narrow job: authenticate four completed
training attempts and two frozen historical receipt sources before opening the
single frozen D_V bundle, evaluate the four new method/seed artifacts against
that same in-memory bundle, adapt the eight historical evidence rows, and make
exactly one frozen Wave-A decision.

There is no resume, overwrite, partial reveal, threshold override, split
selector, or D_T interface.  A failed run leaves only its deterministic staging
directory.  The final directory becomes visible through one rename only after
all results and the decision have been constructed in memory and COMPLETE.json
has been written last.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from ..cache.schema import file_sha256, stable_fingerprint
from ..data import ManifestImageDataset, PreprocessConfig
from ..splits import load_and_validate_manifest
from .cache_pipeline import LoadedDVCacheBundle, load_d_v_cache_bundle
from .paired_artifacts import (
    LoadedPairedDecoderArtifact,
    load_paired_decoder_artifact,
)
from .paired_formal_decision import (
    FORMAL_SEEDS,
    FormalMethodEvidence,
    assess_formal_wave,
)
from .paired_formal_evaluation import (
    FrozenComparisonProtocol,
    PairedFormalDVResult,
    load_frozen_comparison_protocol,
    load_paired_formal_d_v_result,
    select_and_evaluate_paired_formal_method,
)
from .paired_formal_runner import (
    PublishedPairedFormalAttempt,
    load_paired_formal_attempt,
)
from .paired_historical_evidence import (
    FrozenHistoricalFXV3Sources,
    load_frozen_historical_fx_v3_sources,
)


PAIRED_FORMAL_WAVE_A_REVEAL_CONFIG_SCHEMA = (
    "cure-lite-paired-formal-wave-a-reveal-config-v1"
)
PAIRED_FORMAL_WAVE_A_REVEAL_RECEIPT_SCHEMA = (
    "cure-lite-paired-formal-wave-a-reveal-receipt-v1"
)
PAIRED_FORMAL_WAVE_A_REVEAL_COMPLETE_SCHEMA = (
    "cure-lite-paired-formal-wave-a-reveal-complete-v1"
)
WAVE_A_ATTEMPTS = (
    (42, "paired_difference"),
    (43, "paired_difference"),
    (42, "independent_endpoint"),
    (43, "independent_endpoint"),
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_IMPLEMENTATION_PATHS = (
    "cure_lite/cache/base_cache.py",
    "cure_lite/cache/schema.py",
    "cure_lite/cache/state_cache.py",
    "cure_lite/calibration.py",
    "cure_lite/calibration_ledger.py",
    "cure_lite/config.py",
    "cure_lite/data.py",
    "cure_lite/decoder.py",
    "cure_lite/experiment/artifacts.py",
    "cure_lite/experiment/cache_pipeline.py",
    "cure_lite/experiment/evaluation_pipeline.py",
    "cure_lite/experiment/formal_anchor.py",
    "cure_lite/experiment/formal_evaluation.py",
    "cure_lite/experiment/paired_artifacts.py",
    "cure_lite/experiment/paired_formal_decision.py",
    "cure_lite/experiment/paired_formal_evaluation.py",
    "cure_lite/experiment/paired_formal_runner.py",
    "cure_lite/experiment/paired_formal_wave_reveal.py",
    "cure_lite/experiment/paired_historical_evidence.py",
    "cure_lite/instances.py",
    "cure_lite/matching.py",
    "cure_lite/metrics.py",
    "cure_lite/occupancy.py",
    "cure_lite/splits.py",
    "cure_lite/types.py",
    "tools/run_paired_formal_wave_a_reveal.py",
)
_HEX = frozenset("0123456789abcdef")
_RESULT_DIR = "results"
_HISTORICAL_NAME = "historical_evidence.json"
_DECISION_NAME = "decision.json"
_RECEIPT_NAME = "reveal_receipt.json"
_COMPLETE_NAME = "COMPLETE.json"
_INCOMPLETE_NAME = ".INCOMPLETE.json"

# Keep verification recomputation distinct from the single authoritative
# decision invocation in ``run_wave_a_reveal``.  Tests may replace the latter
# through its dependency seam; the strict loader always uses the frozen
# implementation captured here.
_recompute_wave_decision = assess_formal_wave


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
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
        try:
            resolved_parent = parent.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"{name} parent does not exist") from error
        if (
            not resolved_parent.is_dir()
            or resolved_parent.is_symlink()
            or resolved_parent != parent
        ):
            raise ValueError(f"{name} parent is not a canonical directory")
        return candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} does not exist") from error
    if resolved != candidate or resolved.is_symlink():
        raise ValueError(f"{name} is not a canonical repository path")
    if kind == "file" and not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    if kind == "directory" and not resolved.is_dir():
        raise ValueError(f"{name} must be a regular directory")
    return resolved


def _file_binding(
    value: object,
    *,
    name: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "repo_path",
        "file_sha256",
    }:
        raise ValueError(f"{name} fields are not canonical")
    path = value["repo_path"]
    digest = _digest(value["file_sha256"], name=f"{name}.file_sha256")
    if not isinstance(path, str):
        raise TypeError(f"{name}.repo_path must be a string")
    return {"repo_path": path, "file_sha256": digest}


def _validate_attempt_binding(value: object) -> dict[str, object]:
    fields = {
        "seed",
        "method",
        "repo_path",
        "complete_file_sha256",
        "complete_fingerprint",
        "run_receipt_fingerprint",
        "paired_artifact_fingerprint",
        "artifact_receipt_sha256",
        "decoder_state_fingerprint",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("formal attempt binding fields are not canonical")
    result = dict(value)
    if (result["seed"], result["method"]) not in WAVE_A_ATTEMPTS:
        raise ValueError("formal attempt is not a frozen Wave-A method/seed")
    if not isinstance(result["repo_path"], str):
        raise TypeError("formal attempt repo_path must be a string")
    for field in fields - {"seed", "method", "repo_path"}:
        _digest(result[field], name=f"formal attempt {field}")
    return result


def _validate_historical_binding(value: object) -> dict[str, object]:
    fields = {
        "seed",
        "run_repo_path",
        "complete_file_sha256",
        "complete_fingerprint",
        "protocol_repo_path",
        "protocol_freeze_sha256",
        "stage_a_config_sha256",
        "decision_rule_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("historical source binding fields are not canonical")
    result = dict(value)
    if result["seed"] not in FORMAL_SEEDS:
        raise ValueError("historical source seed must be 42 or 43")
    if not isinstance(result["run_repo_path"], str) or not isinstance(
        result["protocol_repo_path"], str
    ):
        raise TypeError("historical source paths must be strings")
    for field in fields - {"seed", "run_repo_path", "protocol_repo_path"}:
        _digest(result[field], name=f"historical source {field}")
    return result


def validate_wave_a_reveal_config(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate the reveal protocol without opening any configured path."""

    if not isinstance(value, Mapping):
        raise TypeError("Wave-A reveal config must be a mapping")
    config = dict(value)
    fields = {
        "schema_version",
        "protocol_id",
        "dataset",
        "wave",
        "formal_runner_config_fingerprint",
        "comparison_protocol",
        "attempts",
        "historical_sources",
        "d_v_bundle",
        "output_repo_path",
        "implementation_binding",
        "execution_policy",
        "config_fingerprint_scope",
        "config_fingerprint",
    }
    if set(config) != fields:
        raise ValueError("Wave-A reveal config fields are not canonical")
    unsigned = dict(config)
    fingerprint = unsigned.pop("config_fingerprint")
    _digest(fingerprint, name="config_fingerprint")
    if stable_fingerprint(unsigned) != fingerprint:
        raise ValueError("Wave-A reveal config fingerprint mismatch")
    if (
        config["schema_version"]
        != PAIRED_FORMAL_WAVE_A_REVEAL_CONFIG_SCHEMA
        or config["protocol_id"] != "irstd1k-paired-formal-wave-a-reveal-v1"
        or config["dataset"] != "IRSTD-1K"
        or config["wave"] != "A"
        or config["config_fingerprint_scope"]
        != "all-fields-except-config_fingerprint"
    ):
        raise ValueError("Wave-A reveal protocol identity changed")
    _digest(
        config["formal_runner_config_fingerprint"],
        name="formal_runner_config_fingerprint",
    )
    _file_binding(config["comparison_protocol"], name="comparison protocol")

    attempts = config["attempts"]
    if not isinstance(attempts, list):
        raise TypeError("attempts must be a list")
    normalized_attempts = tuple(_validate_attempt_binding(row) for row in attempts)
    if tuple((row["seed"], row["method"]) for row in normalized_attempts) != (
        WAVE_A_ATTEMPTS
    ):
        raise ValueError("attempts must contain the exact ordered Wave-A quartet")

    historical = config["historical_sources"]
    if not isinstance(historical, list):
        raise TypeError("historical_sources must be a list")
    normalized_historical = tuple(
        _validate_historical_binding(row) for row in historical
    )
    if tuple(row["seed"] for row in normalized_historical) != FORMAL_SEEDS:
        raise ValueError(
            "historical_sources must contain exactly ordered seeds 42 and 43"
        )

    bundle = config["d_v_bundle"]
    if not isinstance(bundle, Mapping) or set(bundle) != {
        "manifest",
        "base_index",
        "expected_base_fingerprint",
        "preprocessing",
        "preprocessing_fingerprint",
    }:
        raise ValueError("D_V bundle binding fields are not canonical")
    _file_binding(bundle["manifest"], name="D_V manifest")
    _file_binding(bundle["base_index"], name="D_V base index")
    _digest(
        bundle["expected_base_fingerprint"],
        name="D_V expected_base_fingerprint",
    )
    preprocessing = PreprocessConfig.from_fingerprint_payload(
        bundle["preprocessing"]
    )
    expected_preprocessing = _digest(
        bundle["preprocessing_fingerprint"],
        name="D_V preprocessing_fingerprint",
    )
    if (
        stable_fingerprint(preprocessing.fingerprint_payload())
        != expected_preprocessing
    ):
        raise ValueError("D_V preprocessing fingerprint mismatch")

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
        or tuple(sorted(implementation)) != tuple(sorted(_IMPLEMENTATION_PATHS))
    ):
        raise ValueError("reveal implementation binding inventory changed")
    for relative, digest in implementation.items():
        if not isinstance(relative, str):
            raise TypeError("implementation path must be a string")
        _digest(digest, name=f"implementation SHA for {relative}")
    policy = config["execution_policy"]
    expected_policy = {
        "create_only_output": True,
        "failed_staging_reuse": False,
        "resume": False,
        "overwrite": False,
        "runtime_split": "D_V",
        "allow_D_T": False,
        "bundle_materializations": 1,
        "new_method_evaluations": 4,
        "historical_receipt_adaptations": 8,
        "wave_decisions": 1,
        "all_non_d_v_inputs_verified_before_d_v": True,
        "complete_written_last": True,
        "atomic_final_rename": True,
        "stdout_only_after_success": True,
    }
    if not isinstance(policy, Mapping) or dict(policy) != expected_policy:
        raise ValueError("Wave-A reveal execution policy changed")
    return config


@dataclass(frozen=True, slots=True)
class _RevealConfigSeal:
    source_path: Path
    source_sha256: str
    payload: dict[str, object]


@dataclass(frozen=True)
class LoadedWaveARevealConfig:
    source_path: Path
    source_sha256: str
    payload: Mapping[str, object]
    config_fingerprint: str
    _verification_token: object

    def _seal(self) -> _RevealConfigSeal:
        seal = self._verification_token
        if type(seal) is not _RevealConfigSeal:
            raise TypeError("reveal config must come from the strict loader")
        if (
            seal.source_path != self.source_path
            or seal.source_sha256 != self.source_sha256
            or seal.payload is not self.payload
        ):
            raise TypeError("loaded reveal config fields were replaced")
        return seal

    def verify_unchanged(self) -> None:
        seal = self._seal()
        if (
            self.source_path.is_symlink()
            or file_sha256(self.source_path) != self.source_sha256
            or _strict_json(self.source_path, name="Wave-A reveal config")
            != seal.payload
        ):
            raise RuntimeError("Wave-A reveal config changed on disk")
        validate_wave_a_reveal_config(self.payload)

    def __post_init__(self) -> None:
        self.verify_unchanged()
        if self.config_fingerprint != self.payload["config_fingerprint"]:
            raise ValueError("loaded reveal config fingerprint changed")


def load_wave_a_reveal_config(
    path: str | Path,
) -> LoadedWaveARevealConfig:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError("Wave-A reveal config may not be a symlink")
    source = candidate.resolve(strict=True)
    payload = validate_wave_a_reveal_config(
        _strict_json(source, name="Wave-A reveal config")
    )
    seal = _RevealConfigSeal(source, file_sha256(source), payload)
    return LoadedWaveARevealConfig(
        source_path=source,
        source_sha256=seal.source_sha256,
        payload=payload,
        config_fingerprint=str(payload["config_fingerprint"]),
        _verification_token=seal,
    )


@dataclass(frozen=True)
class _VerifiedInputs:
    protocol: FrozenComparisonProtocol
    attempts: tuple[PublishedPairedFormalAttempt, ...]
    artifacts: tuple[LoadedPairedDecoderArtifact, ...]
    historical_sources: FrozenHistoricalFXV3Sources
    historical_evidence: tuple[FormalMethodEvidence, ...]
    manifest_path: Path
    base_index_path: Path
    preprocessing: PreprocessConfig
    expected_base_fingerprint: str
    output: Path
    staging: Path


def _verify_file(path: Path, expected_sha: object, *, name: str) -> None:
    digest = _digest(expected_sha, name=f"{name} SHA256")
    if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
        raise RuntimeError(f"{name} differs from the frozen binding")


def _atomic_rename_noreplace(source: Path, target: Path) -> None:
    """Publish one directory atomically without permitting replacement."""

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
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(target),
        rename_noreplace,
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
    libc = ctypes.CDLL(None, use_errno=True)
    if getattr(libc, "renameat2", None) is None:
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


def _verify_implementation(config: LoadedWaveARevealConfig) -> None:
    implementation = config.payload["implementation_binding"]
    assert isinstance(implementation, Mapping)
    for relative in _IMPLEMENTATION_PATHS:
        path = _repo_path(relative, name=f"implementation {relative}", kind="file")
        _verify_file(
            path,
            implementation[relative],
            name=f"implementation {relative}",
        )


def _verify_non_d_v_inputs(
    config: LoadedWaveARevealConfig,
) -> _VerifiedInputs:
    """Authenticate every non-D_V input before the D_V paths are opened."""

    config.verify_unchanged()
    _verify_implementation(config)
    _require_atomic_rename_noreplace()

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

    loaded_attempts: list[PublishedPairedFormalAttempt] = []
    loaded_artifacts: list[LoadedPairedDecoderArtifact] = []
    raw_attempts = config.payload["attempts"]
    assert isinstance(raw_attempts, list)
    for binding in raw_attempts:
        assert isinstance(binding, Mapping)
        root = _repo_path(
            binding["repo_path"],
            name="formal attempt",
            kind="directory",
        )
        _verify_file(
            root / _COMPLETE_NAME,
            binding["complete_file_sha256"],
            name="formal attempt COMPLETE",
        )
        _verify_file(
            root / "decoder_artifact" / "receipt.json",
            binding["artifact_receipt_sha256"],
            name="formal attempt artifact receipt",
        )
        attempt = load_paired_formal_attempt(root)
        artifact = load_paired_decoder_artifact(
            root / "decoder_artifact"
        )
        if (
            attempt.seed != binding["seed"]
            or attempt.method != binding["method"]
            or attempt.complete_fingerprint != binding["complete_fingerprint"]
            or attempt.run_receipt_fingerprint
            != binding["run_receipt_fingerprint"]
            or attempt.paired_artifact_fingerprint
            != binding["paired_artifact_fingerprint"]
            or artifact.artifact_fingerprint
            != binding["paired_artifact_fingerprint"]
            or artifact.decoder_state_fingerprint
            != binding["decoder_state_fingerprint"]
            or artifact.config.method != binding["method"]
            or artifact.config.seed != binding["seed"]
            or attempt.config_fingerprint
            != config.payload["formal_runner_config_fingerprint"]
        ):
            raise RuntimeError("formal attempt differs from its reveal binding")
        attempt.verify_unchanged()
        artifact.verify_unchanged()
        loaded_attempts.append(attempt)
        loaded_artifacts.append(artifact)

    raw_historical = config.payload["historical_sources"]
    assert isinstance(raw_historical, list)
    historical_paths: dict[int, tuple[Path, Path]] = {}
    for binding in raw_historical:
        assert isinstance(binding, Mapping)
        run_root = _repo_path(
            binding["run_repo_path"],
            name="historical run root",
            kind="directory",
        )
        protocol_root = _repo_path(
            binding["protocol_repo_path"],
            name="historical protocol root",
            kind="directory",
        )
        _verify_file(
            run_root / _COMPLETE_NAME,
            binding["complete_file_sha256"],
            name="historical COMPLETE",
        )
        _verify_file(
            protocol_root / "protocol_freeze.json",
            binding["protocol_freeze_sha256"],
            name="historical protocol freeze",
        )
        _verify_file(
            protocol_root / "stage_a_config.json",
            binding["stage_a_config_sha256"],
            name="historical Stage-A config",
        )
        _verify_file(
            protocol_root / "stage_a_decision_rule.json",
            binding["decision_rule_sha256"],
            name="historical decision rule",
        )
        historical_paths[int(binding["seed"])] = (run_root, protocol_root)
    sources = load_frozen_historical_fx_v3_sources(
        seed42_run_root=historical_paths[42][0],
        seed42_protocol_root=historical_paths[42][1],
        seed43_run_root=historical_paths[43][0],
        seed43_protocol_root=historical_paths[43][1],
        comparison_protocol=protocol,
        repository_root=_REPO_ROOT,
    )
    for binding in raw_historical:
        assert isinstance(binding, Mapping)
        source = sources.source_for_seed(int(binding["seed"]))
        complete_fingerprints = {
            method.source_complete_fingerprint for method in source.methods
        }
        if (
            complete_fingerprints != {binding["complete_fingerprint"]}
            or source.stage_a_config_sha256
            != binding["stage_a_config_sha256"]
            or source.decision_rule_sha256
            != binding["decision_rule_sha256"]
        ):
            raise RuntimeError("historical source differs from reveal binding")
    historical_evidence = sources.adapted_evidence(protocol)
    if (
        len(historical_evidence) != 8
        or {
            (row.seed, row.method) for row in historical_evidence
        }
        != {
            (seed, method)
            for seed in FORMAL_SEEDS
            for method in ("Base@B", "F", "F×", "U")
        }
    ):
        raise RuntimeError("historical evidence population is not canonical")

    output = _repo_path(
        config.payload["output_repo_path"],
        name="Wave-A output",
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
    protected_paths = {
        comparison_path,
        *(attempt.root for attempt in loaded_attempts),
        *(
            path
            for roots in historical_paths.values()
            for path in roots
        ),
    }
    if any(
        _paths_overlap(target, protected)
        for target in (output, staging)
        for protected in protected_paths
    ):
        raise ValueError("Wave-A output overlaps a frozen input path")
    # D_V paths and hashes are intentionally resolved only after every
    # non-D_V object above has passed strict validation.
    bundle_binding = config.payload["d_v_bundle"]
    assert isinstance(bundle_binding, Mapping)
    manifest_binding = bundle_binding["manifest"]
    index_binding = bundle_binding["base_index"]
    assert isinstance(manifest_binding, Mapping)
    assert isinstance(index_binding, Mapping)
    manifest_path = _repo_path(
        manifest_binding["repo_path"],
        name="D_V manifest",
        kind="file",
    )
    base_index_path = _repo_path(
        index_binding["repo_path"],
        name="D_V base index",
        kind="file",
    )
    _verify_file(
        manifest_path,
        manifest_binding["file_sha256"],
        name="D_V manifest",
    )
    _verify_file(
        base_index_path,
        index_binding["file_sha256"],
        name="D_V base index",
    )
    if any(
        _paths_overlap(target, protected)
        for target in (output, staging)
        for protected in (manifest_path, base_index_path)
    ):
        raise ValueError("Wave-A output overlaps a D_V input path")
    preprocessing = PreprocessConfig.from_fingerprint_payload(
        bundle_binding["preprocessing"]
    )
    return _VerifiedInputs(
        protocol=protocol,
        attempts=tuple(loaded_attempts),
        artifacts=tuple(loaded_artifacts),
        historical_sources=sources,
        historical_evidence=historical_evidence,
        manifest_path=manifest_path,
        base_index_path=base_index_path,
        preprocessing=preprocessing,
        expected_base_fingerprint=str(
            bundle_binding["expected_base_fingerprint"]
        ),
        output=output,
        staging=staging,
    )


def _materialize_one_bundle(inputs: _VerifiedInputs) -> LoadedDVCacheBundle:
    manifest = load_and_validate_manifest(inputs.manifest_path)
    dataset = ManifestImageDataset(
        manifest,
        "D_V",
        inputs.preprocessing,
        manifest_path=inputs.manifest_path,
    )
    return load_d_v_cache_bundle(
        inputs.base_index_path,
        dataset,
        expected_base_fingerprint=inputs.expected_base_fingerprint,
    )


def _result_filename(result: PairedFormalDVResult) -> str:
    return f"{result.method}_seed{result.seed}.json"


def _claim_staging(
    config: LoadedWaveARevealConfig,
    inputs: _VerifiedInputs,
) -> None:
    """Exclusively claim this reveal before the D_V bundle is materialized."""

    inputs.staging.mkdir(parents=False, exist_ok=False)
    claim = _fingerprinted(
        {
            "schema_version": (
                "cure-lite-paired-formal-wave-a-reveal-incomplete-v1"
            ),
            "execution_status": "claimed_before_D_V_materialization",
            "config_fingerprint": config.config_fingerprint,
            "comparison_protocol_fingerprint": (
                inputs.protocol.comparison_protocol_fingerprint
            ),
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
    config: LoadedWaveARevealConfig,
    inputs: _VerifiedInputs,
    bundle: LoadedDVCacheBundle,
) -> None:
    """Re-authenticate every frozen source without rebuilding the D_V bundle."""

    config.verify_unchanged()
    _verify_implementation(config)
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
        raise RuntimeError("comparison protocol changed during reveal")

    for attempt, artifact in zip(
        inputs.attempts,
        inputs.artifacts,
        strict=True,
    ):
        attempt.verify_unchanged()
        artifact.verify_unchanged()

    seed42 = inputs.historical_sources.source_for_seed(42)
    seed43 = inputs.historical_sources.source_for_seed(43)
    reloaded_historical = load_frozen_historical_fx_v3_sources(
        seed42_run_root=seed42.run_root,
        seed42_protocol_root=seed42.protocol_root,
        seed43_run_root=seed43.run_root,
        seed43_protocol_root=seed43.protocol_root,
        comparison_protocol=reloaded_protocol,
        repository_root=_REPO_ROOT,
    )
    if reloaded_historical != inputs.historical_sources:
        raise RuntimeError("historical sources changed during reveal")

    bundle.verify_unchanged()
    inputs.protocol.verify_bundle(bundle)


def _publish(
    config: LoadedWaveARevealConfig,
    inputs: _VerifiedInputs,
    results: tuple[PairedFormalDVResult, ...],
    decision: Mapping[str, object],
) -> "PublishedWaveAReveal":
    """Write the already complete in-memory reveal and atomically publish it."""

    if (
        not inputs.staging.is_dir()
        or inputs.staging.is_symlink()
        or {path.name for path in inputs.staging.iterdir()}
        != {_INCOMPLETE_NAME}
    ):
        raise RuntimeError("Wave-A staging claim changed before publication")
    claim = _strict_json(
        inputs.staging / _INCOMPLETE_NAME,
        name="Wave-A staging claim",
    )
    _verify_fingerprint(
        claim,
        field="claim_fingerprint",
        name="Wave-A staging claim",
    )
    if (
        claim.get("schema_version")
        != "cure-lite-paired-formal-wave-a-reveal-incomplete-v1"
        or claim.get("execution_status")
        != "claimed_before_D_V_materialization"
        or claim.get("config_fingerprint") != config.config_fingerprint
        or claim.get("comparison_protocol_fingerprint")
        != inputs.protocol.comparison_protocol_fingerprint
        or claim.get("resume_allowed") is not False
        or claim.get("directory_reuse_allowed") is not False
        or claim.get("results_authoritative") is not False
        or claim.get("D_T_accessed") is not False
    ):
        raise RuntimeError("Wave-A staging claim semantics changed")
    result_dir = inputs.staging / _RESULT_DIR
    result_dir.mkdir()
    for result in results:
        _write_new_json(
            result_dir / _result_filename(result),
            result.canonical_payload(),
        )
    historical_payload = _fingerprinted(
        {
            "schema_version": (
                "cure-lite-paired-formal-wave-a-historical-evidence-v1"
            ),
            "comparison_protocol_fingerprint": (
                inputs.protocol.comparison_protocol_fingerprint
            ),
            "evidence": [
                row.canonical_payload() for row in inputs.historical_evidence
            ],
        },
        field="receipt_fingerprint",
    )
    _write_new_json(inputs.staging / _HISTORICAL_NAME, historical_payload)
    _write_new_json(inputs.staging / _DECISION_NAME, decision)

    artifact_files = {
        path.relative_to(inputs.staging).as_posix(): file_sha256(path)
        for path in sorted(inputs.staging.rglob("*"))
        if path.is_file() and path.name != _INCOMPLETE_NAME
    }
    receipt = _fingerprinted(
        {
            "schema_version": PAIRED_FORMAL_WAVE_A_REVEAL_RECEIPT_SCHEMA,
            "execution_status": "complete_in_memory",
            "wave": "A",
            "config_fingerprint": config.config_fingerprint,
            "config_file_sha256": config.source_sha256,
            "formal_runner_config_fingerprint": config.payload[
                "formal_runner_config_fingerprint"
            ],
            "comparison_protocol_fingerprint": (
                inputs.protocol.comparison_protocol_fingerprint
            ),
            "new_result_receipt_fingerprints": {
                f"{result.seed}:{result.method}": result.receipt_fingerprint
                for result in results
            },
            "historical_result_fingerprints": {
                f"{row.seed}:{row.method}": row.result_fingerprint
                for row in inputs.historical_evidence
            },
            "decision_fingerprint": decision["decision_fingerprint"],
            "bundle_materialization_count": 1,
            "new_method_evaluation_count": 4,
            "historical_adaptation_count": 8,
            "wave_decision_count": 1,
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
            "schema_version": PAIRED_FORMAL_WAVE_A_REVEAL_COMPLETE_SCHEMA,
            "execution_status": "complete",
            "wave": "A",
            "config_fingerprint": config.config_fingerprint,
            "comparison_protocol_fingerprint": (
                inputs.protocol.comparison_protocol_fingerprint
            ),
            "reveal_receipt_fingerprint": receipt["receipt_fingerprint"],
            "decision_fingerprint": decision["decision_fingerprint"],
            "artifact_files": published_files,
            "artifact_file_count": len(published_files),
            "complete_written_last": True,
            "atomic_final_rename": True,
            "resume_used": False,
            "overwrite_used": False,
            "D_T_accessed": False,
        },
        field="complete_fingerprint",
    )
    _write_new_json(inputs.staging / _COMPLETE_NAME, complete)
    _fsync_directory(result_dir)
    _fsync_directory(inputs.staging)
    validated = load_published_wave_a_reveal(inputs.staging)
    config.verify_unchanged()
    if inputs.output.exists() or inputs.output.is_symlink():
        raise FileExistsError("final reveal output appeared before publication")
    _atomic_rename_noreplace(inputs.staging, inputs.output)
    _fsync_directory(inputs.output.parent)
    return PublishedWaveAReveal(
        root=inputs.output,
        decision=validated.decision,
        result_evidence=validated.result_evidence,
        complete_fingerprint=validated.complete_fingerprint,
    )


@dataclass(frozen=True)
class PublishedWaveAReveal:
    root: Path
    decision: Mapping[str, object]
    result_evidence: tuple[FormalMethodEvidence, ...]
    complete_fingerprint: str

    def success_summary(self) -> dict[str, object]:
        """Return the only payload the CLI is allowed to print."""

        return {
            "status": self.decision["status"],
            "wave": "A",
            "output": str(self.root),
            "all_seeds_pass": self.decision["all_seeds_pass"],
            "decision_fingerprint": self.decision["decision_fingerprint"],
            "complete_fingerprint": self.complete_fingerprint,
        }


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


def load_published_wave_a_reveal(
    output_dir: str | Path,
) -> PublishedWaveAReveal:
    """Strictly load one fully published Wave-A reveal."""

    requested = Path(output_dir).expanduser()
    if requested.is_symlink():
        raise ValueError("published Wave-A reveal may not be a symlink")
    root = requested.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("published Wave-A reveal must be a regular directory")
    members = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    expected_results = {
        f"{_RESULT_DIR}/{method}_seed{seed}.json"
        for seed, method in WAVE_A_ATTEMPTS
    }
    expected = {
        *expected_results,
        _HISTORICAL_NAME,
        _DECISION_NAME,
        _RECEIPT_NAME,
        _COMPLETE_NAME,
    }
    if members != expected or any(
        path.is_symlink()
        for path in root.rglob("*")
    ):
        raise RuntimeError("published Wave-A reveal inventory changed")

    complete = _strict_json(root / _COMPLETE_NAME, name="Wave-A COMPLETE")
    complete_fingerprint = _verify_fingerprint(
        complete,
        field="complete_fingerprint",
        name="Wave-A COMPLETE",
    )
    artifact_files = complete.get("artifact_files")
    actual_files = {
        relative: file_sha256(root / relative)
        for relative in sorted(expected - {_COMPLETE_NAME})
    }
    if (
        complete.get("schema_version")
        != PAIRED_FORMAL_WAVE_A_REVEAL_COMPLETE_SCHEMA
        or complete.get("execution_status") != "complete"
        or complete.get("wave") != "A"
        or complete.get("artifact_files") != actual_files
        or complete.get("artifact_file_count") != len(actual_files)
        or complete.get("complete_written_last") is not True
        or complete.get("atomic_final_rename") is not True
        or complete.get("resume_used") is not False
        or complete.get("overwrite_used") is not False
        or complete.get("D_T_accessed") is not False
    ):
        raise RuntimeError("published Wave-A COMPLETE semantics changed")

    receipt = _strict_json(
        root / _RECEIPT_NAME,
        name="Wave-A reveal receipt",
    )
    receipt_fingerprint = _verify_fingerprint(
        receipt,
        field="receipt_fingerprint",
        name="Wave-A reveal receipt",
    )
    if (
        receipt.get("schema_version")
        != PAIRED_FORMAL_WAVE_A_REVEAL_RECEIPT_SCHEMA
        or receipt.get("execution_status") != "complete_in_memory"
        or receipt.get("wave") != "A"
        or receipt.get("bundle_materialization_count") != 1
        or receipt.get("new_method_evaluation_count") != 4
        or receipt.get("historical_adaptation_count") != 8
        or receipt.get("wave_decision_count") != 1
        or receipt.get("D_T_accessed") is not False
        or complete.get("reveal_receipt_fingerprint") != receipt_fingerprint
        or complete.get("config_fingerprint")
        != receipt.get("config_fingerprint")
        or complete.get("comparison_protocol_fingerprint")
        != receipt.get("comparison_protocol_fingerprint")
        or receipt.get("artifact_files_before_receipt")
        != {
            relative: actual_files[relative]
            for relative in sorted(
                expected
                - {
                    _RECEIPT_NAME,
                    _COMPLETE_NAME,
                }
            )
        }
    ):
        raise RuntimeError("published Wave-A reveal receipt semantics changed")

    results = tuple(
        load_paired_formal_d_v_result(
            root / _RESULT_DIR / f"{method}_seed{seed}.json"
        )
        for seed, method in WAVE_A_ATTEMPTS
    )
    result_receipts = {
        f"{result.seed}:{result.method}": result.receipt_fingerprint
        for result in results
    }
    if receipt.get("new_result_receipt_fingerprints") != result_receipts:
        raise RuntimeError("published Wave-A new-result bindings changed")

    historical = _strict_json(
        root / _HISTORICAL_NAME,
        name="Wave-A historical evidence",
    )
    _verify_fingerprint(
        historical,
        field="receipt_fingerprint",
        name="Wave-A historical evidence",
    )
    raw_historical = historical.get("evidence")
    if not isinstance(raw_historical, list) or len(raw_historical) != 8:
        raise RuntimeError("published historical evidence count changed")
    if any(not isinstance(row, dict) for row in raw_historical):
        raise TypeError("historical evidence rows must be mappings")
    historical_evidence = tuple(
        FormalMethodEvidence(**row) for row in raw_historical
    )
    historical_fingerprints = {
        f"{row.seed}:{row.method}": row.result_fingerprint
        for row in historical_evidence
    }
    if (
        receipt.get("historical_result_fingerprints")
        != historical_fingerprints
        or historical.get("comparison_protocol_fingerprint")
        != receipt.get("comparison_protocol_fingerprint")
    ):
        raise RuntimeError("published historical evidence bindings changed")

    decision = _strict_json(root / _DECISION_NAME, name="Wave-A decision")
    decision_fingerprint = _verify_fingerprint(
        decision,
        field="decision_fingerprint",
        name="Wave-A decision",
    )
    if (
        decision.get("schema_version")
        != "cure-lite-paired-formal-wave-decision-v1"
        or decision.get("wave") != "A"
        or decision.get("D_T_accessed") is not False
        or decision.get("comparison_protocol_fingerprint")
        != receipt.get("comparison_protocol_fingerprint")
        or decision.get("protocol_fingerprint")
        != receipt.get("config_fingerprint")
        or receipt.get("decision_fingerprint") != decision_fingerprint
        or complete.get("decision_fingerprint") != decision_fingerprint
    ):
        raise RuntimeError("published Wave-A decision bindings changed")
    result_evidence = tuple(
        result.to_formal_method_evidence() for result in results
    )
    recomputed = _recompute_wave_decision(
        (*result_evidence, *historical_evidence),
        wave="A",
        protocol_fingerprint=str(receipt["config_fingerprint"]),
        comparison_protocol_fingerprint=str(
            receipt["comparison_protocol_fingerprint"]
        ),
    )
    if decision != recomputed:
        raise RuntimeError(
            "published Wave-A decision differs from its twelve evidence rows"
        )
    return PublishedWaveAReveal(
        root=root,
        decision=decision,
        result_evidence=result_evidence,
        complete_fingerprint=complete_fingerprint,
    )


def run_wave_a_reveal(
    config_path: str | Path,
    *,
    bundle_materializer: Callable[
        [_VerifiedInputs], LoadedDVCacheBundle
    ] = _materialize_one_bundle,
    evaluator: Callable[
        [LoadedDVCacheBundle, LoadedPairedDecoderArtifact],
        PairedFormalDVResult,
    ]
    | None = None,
) -> PublishedWaveAReveal:
    """Execute the exact one-shot Wave-A reveal.

    ``bundle_materializer`` and ``evaluator`` are dependency seams for tests;
    the command-line entry point never exposes them.
    """

    config = load_wave_a_reveal_config(config_path)
    inputs = _verify_non_d_v_inputs(config)
    config.verify_unchanged()
    _claim_staging(config, inputs)
    bundle = bundle_materializer(inputs)
    if not isinstance(bundle, LoadedDVCacheBundle):
        raise TypeError("bundle materializer must return LoadedDVCacheBundle")
    inputs.protocol.verify_bundle(bundle)

    results: list[PairedFormalDVResult] = []
    for artifact in inputs.artifacts:
        result = (
            select_and_evaluate_paired_formal_method(
                bundle,
                artifact,
                comparison_protocol=inputs.protocol,
            )
            if evaluator is None
            else evaluator(bundle, artifact)
        )
        if not isinstance(result, PairedFormalDVResult):
            raise TypeError("evaluator must return PairedFormalDVResult")
        result.verify_unchanged()
        if (
            result.method != artifact.config.method
            or result.seed != artifact.config.seed
            or result.comparison_protocol_fingerprint
            != inputs.protocol.comparison_protocol_fingerprint
        ):
            raise RuntimeError("evaluated result differs from its frozen input")
        results.append(result)
    if tuple((row.seed, row.method) for row in results) != WAVE_A_ATTEMPTS:
        raise RuntimeError("new Wave-A result quartet is not canonical")

    evidence = (
        *(row.to_formal_method_evidence() for row in results),
        *inputs.historical_evidence,
    )
    decision = assess_formal_wave(
        evidence,
        wave="A",
        protocol_fingerprint=config.config_fingerprint,
        comparison_protocol_fingerprint=(
            inputs.protocol.comparison_protocol_fingerprint
        ),
    )

    # Verify all sources again before any result byte is written.  The D_V
    # bundle is not rebuilt; its in-memory and source bindings are rechecked.
    _verify_frozen_sources_unchanged(config, inputs, bundle)
    return _publish(config, inputs, tuple(results), decision)


__all__ = [
    "PAIRED_FORMAL_WAVE_A_REVEAL_COMPLETE_SCHEMA",
    "PAIRED_FORMAL_WAVE_A_REVEAL_CONFIG_SCHEMA",
    "PAIRED_FORMAL_WAVE_A_REVEAL_RECEIPT_SCHEMA",
    "WAVE_A_ATTEMPTS",
    "LoadedWaveARevealConfig",
    "PublishedWaveAReveal",
    "load_published_wave_a_reveal",
    "load_wave_a_reveal_config",
    "run_wave_a_reveal",
    "validate_wave_a_reveal_config",
]
