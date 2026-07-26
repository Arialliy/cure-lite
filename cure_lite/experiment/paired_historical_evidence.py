"""Read-only loader for the two frozen historical fx_v3 receipt sources.

This module deliberately loads receipts, not Stage-A runs.  In particular it
never opens the historical ``d_v`` cache, decoder artifacts, or datasets.  The
small receipt set is authenticated through the frozen comparison binding, the
Stage-A COMPLETE inventory, and the protocol-freeze config/decision-rule
hashes.  The resulting objects contain exactly the inputs required by
``adapt_frozen_historical_method_evidence``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from ..cache.schema import stable_fingerprint
from ..metrics import FORMAL_STAGE_A_METRIC_FIELDS
from .paired_formal_decision import (
    FORMAL_SEEDS,
    HISTORICAL_COMPARATORS,
    FormalMethodEvidence,
)
from .paired_formal_evaluation import (
    FrozenComparisonProtocol,
    HistoricalFXV3Binding,
    adapt_frozen_historical_method_evidence,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_METHOD_ORDER = ("A", "Base@B", "F", "F×", "U")
_RUN_TOP_LEVEL = {
    "COMPLETE.json": "file",
    "d_r": "directory",
    "d_v": "directory",
    "decoders": "directory",
    "receipts": "directory",
}
_RECEIPT_FILES = {
    "anchor.json": "file",
    "calibration.json": "file",
    "config.json": "file",
    "efficiency.json": "file",
    "finalization.json": "file",
    "results.json": "file",
    "support.json": "file",
}
_PROTOCOL_FILES = {
    "protocol_freeze.json": "file",
    "stage_a_config.json": "file",
    "stage_a_decision_rule.json": "file",
}
_HEX = frozenset("0123456789abcdef")


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _regular_root(path: str | Path, *, name: str) -> Path:
    requested = Path(path).expanduser()
    absolute = requested if requested.is_absolute() else _REPO_ROOT / requested
    absolute = absolute.absolute()
    if absolute.is_symlink():
        raise ValueError(f"{name} may not be a symlink")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{name} does not exist") from error
    if resolved != absolute or not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink directory")
    return resolved


def _inventory(directory: Path, *, name: str) -> dict[str, str]:
    members: dict[str, str] = {}
    for path in directory.iterdir():
        if path.is_symlink():
            raise ValueError(f"{name} contains symlink {path.name!r}")
        if path.is_file():
            kind = "file"
        elif path.is_dir():
            kind = "directory"
        else:
            raise ValueError(f"{name} contains non-regular member {path.name!r}")
        members[path.name] = kind
    return members


def _require_inventory(
    directory: Path,
    *,
    expected: Mapping[str, str],
    name: str,
) -> None:
    actual = _inventory(directory, name=name)
    if actual != expected:
        raise ValueError(f"{name} inventory is not canonical")


def _finite_metric(
    value: object,
    *,
    field: str,
    name: str,
) -> float | bool:
    if field == "budget_violation":
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be bool")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _selected_metrics(value: object) -> tuple[tuple[str, float | bool], ...]:
    if not isinstance(value, Mapping) or set(value) != set(
        FORMAL_STAGE_A_METRIC_FIELDS
    ):
        raise ValueError("historical selected metrics fields are not canonical")
    return tuple(
        (
            name,
            _finite_metric(
                value[name],
                field=name,
                name=f"historical selected_metrics.{name}",
            ),
        )
        for name in FORMAL_STAGE_A_METRIC_FIELDS
    )


@dataclass(frozen=True, slots=True)
class FrozenHistoricalMethodSource:
    """Immutable adapter inputs for one frozen historical method."""

    method: str
    seed: int
    selected_metrics: tuple[tuple[str, float | bool], ...]
    source_results_fingerprint: str
    source_protocol_fingerprint: str
    source_run_config_fingerprint: str
    source_config_file_sha256: str
    source_calibration_receipt_fingerprint: str
    source_complete_fingerprint: str

    def canonical_selected_metrics(self) -> dict[str, float | bool]:
        return dict(self.selected_metrics)

    def canonical_adapter_payload(self) -> dict[str, object]:
        return {
            "method": self.method,
            "seed": self.seed,
            "selected_metrics": self.canonical_selected_metrics(),
            "source_results_fingerprint": self.source_results_fingerprint,
            "source_protocol_fingerprint": self.source_protocol_fingerprint,
            "source_run_config_fingerprint": (
                self.source_run_config_fingerprint
            ),
            "source_config_file_sha256": self.source_config_file_sha256,
            "source_calibration_receipt_fingerprint": (
                self.source_calibration_receipt_fingerprint
            ),
            "source_complete_fingerprint": self.source_complete_fingerprint,
        }

    def adapt(
        self,
        comparison_protocol: FrozenComparisonProtocol,
    ) -> FormalMethodEvidence:
        return adapt_frozen_historical_method_evidence(
            comparison_protocol=comparison_protocol,
            **self.canonical_adapter_payload(),
        )


@dataclass(frozen=True, slots=True)
class FrozenHistoricalFXV3Source:
    """Verified receipt-only source for one fx_v3 seed."""

    seed: int
    run_root: Path
    protocol_root: Path
    methods: tuple[FrozenHistoricalMethodSource, ...]
    stage_a_config_sha256: str
    decision_rule_sha256: str

    def method_source(self, method: str) -> FrozenHistoricalMethodSource:
        try:
            return next(item for item in self.methods if item.method == method)
        except StopIteration as error:
            raise ValueError(
                f"historical method must be one of {HISTORICAL_COMPARATORS}"
            ) from error

    def adapted_evidence(
        self,
        comparison_protocol: FrozenComparisonProtocol,
    ) -> tuple[FormalMethodEvidence, ...]:
        return tuple(item.adapt(comparison_protocol) for item in self.methods)


@dataclass(frozen=True, slots=True)
class FrozenHistoricalFXV3Sources:
    """The complete seed-42/43 frozen historical source pair."""

    sources: tuple[FrozenHistoricalFXV3Source, ...]

    def source_for_seed(self, seed: int) -> FrozenHistoricalFXV3Source:
        try:
            return next(item for item in self.sources if item.seed == seed)
        except StopIteration as error:
            raise ValueError(f"seed must be one of {FORMAL_SEEDS}") from error

    def adapted_evidence(
        self,
        comparison_protocol: FrozenComparisonProtocol,
    ) -> tuple[FormalMethodEvidence, ...]:
        return tuple(
            evidence
            for source in self.sources
            for evidence in source.adapted_evidence(comparison_protocol)
        )


def _verify_protocol_freeze(
    protocol_root: Path,
    run_root: Path,
    *,
    repository_root: Path,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    _require_inventory(
        protocol_root,
        expected=_PROTOCOL_FILES,
        name="historical protocol root",
    )
    freeze = _strict_json(
        protocol_root / "protocol_freeze.json",
        name="historical protocol freeze",
    )
    expected_freeze_fields = {
        "schema_version",
        "frozen_at_utc",
        "runtime_splits",
        "unused_split",
        "manifest_fingerprint",
        "manifest_file_sha256",
        "base_training_config_sha256",
        "reference_base_output",
        "reference_service_invocation_id",
        "reference_base_source_sha256",
        "cache_output",
        "build_record_sha256",
        "stage_a_config_sha256",
        "decision_rule_sha256",
        "run_tool_sha256",
        "assessment_tool_sha256",
        "method_source_tree_digest",
        "stage_a_output",
        "stage_a_service_invocation_id",
        "assessment_output",
        "assessment_service_invocation_id",
        "runtime_python_environment",
        "software_test_result",
    }
    if set(freeze) != expected_freeze_fields:
        raise ValueError("historical protocol-freeze fields are not canonical")
    if (
        freeze["schema_version"] != "cure-lite-stage-a-protocol-freeze-v2"
        or freeze["runtime_splits"] != ["D_B", "D_R", "D_V"]
        or freeze["unused_split"] != "D_T"
    ):
        raise ValueError("historical protocol-freeze semantics are invalid")
    output = freeze["stage_a_output"]
    if not isinstance(output, str) or not output or Path(output).is_absolute():
        raise ValueError("historical frozen Stage-A output path is invalid")
    for field in (
        "reference_base_output",
        "cache_output",
        "stage_a_output",
        "assessment_output",
    ):
        value = freeze[field]
        if (
            not isinstance(value, str)
            or not value
            or Path(value).is_absolute()
            or ".." in Path(value).parts
        ):
            raise ValueError(
                f"historical frozen path {field} is not a safe relative path"
            )
    for field in (
        "manifest_fingerprint",
        "manifest_file_sha256",
        "base_training_config_sha256",
        "reference_base_source_sha256",
        "build_record_sha256",
        "run_tool_sha256",
        "assessment_tool_sha256",
        "method_source_tree_digest",
    ):
        _digest(freeze[field], name=f"historical protocol-freeze {field}")
    expected_output = (repository_root / output).resolve(strict=False)
    if expected_output != run_root:
        raise RuntimeError("historical frozen Stage-A output path changed")

    config_sha = _digest(
        freeze["stage_a_config_sha256"],
        name="historical stage_a_config_sha256",
    )
    decision_sha = _digest(
        freeze["decision_rule_sha256"],
        name="historical decision_rule_sha256",
    )
    config_path = protocol_root / "stage_a_config.json"
    decision_path = protocol_root / "stage_a_decision_rule.json"
    if _sha256(config_path) != config_sha:
        raise RuntimeError("historical frozen Stage-A config hash mismatch")
    if _sha256(decision_path) != decision_sha:
        raise RuntimeError("historical frozen decision-rule hash mismatch")
    config = _strict_json(config_path, name="historical frozen Stage-A config")
    decision = _strict_json(
        decision_path,
        name="historical frozen Stage-A decision rule",
    )
    if (
        decision.get("schema_version")
        != "cure-lite-stage-a-decision-rule-v3"
        or decision.get("dataset") != "IRSTD-1K"
        or decision.get("evaluation_split") != "D_V"
        or decision.get("seed") != seed
        or decision.get("stage_a_config_sha256") != config_sha
        or decision.get("method_order") != list(_METHOD_ORDER)
    ):
        raise ValueError("historical frozen decision-rule semantics are invalid")
    return freeze, config, config_sha, decision_sha


def _verify_complete_and_receipts(
    run_root: Path,
    binding: HistoricalFXV3Binding,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _require_inventory(
        run_root,
        expected=_RUN_TOP_LEVEL,
        name="historical run root",
    )
    receipts = run_root / "receipts"
    if receipts.is_symlink() or not receipts.is_dir():
        raise ValueError("historical receipts must be a non-symlink directory")
    _require_inventory(
        receipts,
        expected=_RECEIPT_FILES,
        name="historical receipt root",
    )
    complete = _strict_json(
        run_root / "COMPLETE.json",
        name="historical COMPLETE receipt",
    )
    complete_core = dict(complete)
    actual_complete_fingerprint = _digest(
        complete_core.pop("complete_fingerprint", None),
        name="historical complete_fingerprint",
    )
    if (
        stable_fingerprint(complete_core) != actual_complete_fingerprint
        or actual_complete_fingerprint != binding.complete_fingerprint
    ):
        raise RuntimeError("historical COMPLETE fingerprint mismatch")
    if (
        complete.get("schema_version") != "cure-lite-stage-a-run-v7"
        or complete.get("status") != "complete"
        or complete.get("method") != "CURE-Lite"
        or complete.get("stage") != "Stage-A"
        or complete.get("method_order") != list(_METHOD_ORDER)
        or complete.get("runtime_splits") != ["D_R", "D_V"]
        or complete.get("unused_split") != "D_T"
    ):
        raise ValueError("historical COMPLETE semantics are invalid")
    artifact_files = complete.get("artifact_files")
    if not isinstance(artifact_files, Mapping):
        raise TypeError("historical COMPLETE artifact_files must be a mapping")
    for filename in sorted(_RECEIPT_FILES):
        relative = f"receipts/{filename}"
        expected_sha = _digest(
            artifact_files.get(relative),
            name=f"historical artifact_files.{relative}",
        )
        if _sha256(receipts / filename) != expected_sha:
            raise RuntimeError(f"historical receipt hash mismatch for {filename}")

    config = _strict_json(
        receipts / "config.json",
        name="historical config receipt",
    )
    calibration = _strict_json(
        receipts / "calibration.json",
        name="historical calibration receipt",
    )
    results = _strict_json(
        receipts / "results.json",
        name="historical results receipt",
    )
    if _sha256(receipts / "config.json") != binding.config_file_sha256:
        raise RuntimeError("historical config receipt hash mismatch")
    if (
        complete.get("run_config_fingerprint")
        != binding.run_config_fingerprint
        or complete.get("calibration_receipt_fingerprint")
        != binding.calibration_receipt_fingerprint
        or complete.get("results_fingerprint") != binding.results_fingerprint
    ):
        raise RuntimeError("historical COMPLETE differs from frozen binding")
    return config, calibration, results


def load_frozen_historical_fx_v3_source(
    run_root: str | Path,
    protocol_root: str | Path,
    *,
    comparison_protocol: FrozenComparisonProtocol,
    seed: int,
    repository_root: str | Path = _REPO_ROOT,
) -> FrozenHistoricalFXV3Source:
    """Load one historical seed through receipts only.

    The loader performs no threshold selection, dataset access, cache loading,
    model construction, or evaluation.
    """

    if not isinstance(comparison_protocol, FrozenComparisonProtocol):
        raise TypeError("comparison_protocol must be FrozenComparisonProtocol")
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"seed must be one of {FORMAL_SEEDS}")
    repository = _regular_root(repository_root, name="repository root")
    run = _regular_root(run_root, name="historical run root")
    protocol = _regular_root(protocol_root, name="historical protocol root")
    try:
        run.relative_to(repository)
        protocol.relative_to(repository)
    except ValueError as error:
        raise ValueError(
            "historical roots must be contained by repository_root"
        ) from error

    binding = comparison_protocol.historical_binding(seed)
    freeze, frozen_config, config_sha, decision_sha = _verify_protocol_freeze(
        protocol,
        run,
        repository_root=repository,
        seed=seed,
    )
    config, calibration, results = _verify_complete_and_receipts(run, binding)
    complete = _strict_json(
        run / "COMPLETE.json",
        name="historical COMPLETE receipt",
    )
    if (
        freeze["manifest_fingerprint"]
        != complete.get("manifest_fingerprint")
        or freeze["manifest_file_sha256"]
        != complete.get("manifest_file_sha256")
        or freeze["method_source_tree_digest"]
        != complete.get("source_tree_digest")
        or freeze["reference_base_source_sha256"]
        != complete.get("base_run_identity", {}).get("source_fingerprint")
    ):
        raise RuntimeError(
            "historical protocol-freeze provenance differs from COMPLETE"
        )

    expected_config_fields = {
        "schema_version",
        "method",
        "stage",
        "runtime_splits",
        "unused_split",
        "source_tree_digest",
        "run_config",
        "run_config_fingerprint",
    }
    if set(config) != expected_config_fields:
        raise ValueError("historical config receipt fields are not canonical")
    run_config = config.get("run_config")
    if (
        config.get("schema_version") != "cure-lite-stage-a-config-receipt-v1"
        or config.get("method") != "CURE-Lite"
        or config.get("stage") != "Stage-A"
        or config.get("runtime_splits") != ["D_R", "D_V"]
        or config.get("unused_split") != "D_T"
        or not isinstance(run_config, Mapping)
        or run_config != frozen_config
        or stable_fingerprint(run_config) != binding.run_config_fingerprint
        or config.get("run_config_fingerprint")
        != binding.run_config_fingerprint
    ):
        raise RuntimeError("historical config semantics differ from freeze")

    expected_calibration_fields = {
        "schema_version",
        "method_order",
        "methods",
        "common_training_fingerprint",
        "receipt_fingerprint",
    }
    if set(calibration) != expected_calibration_fields:
        raise ValueError(
            "historical calibration receipt fields are not canonical"
        )
    calibration_methods = calibration.get("methods")
    if (
        calibration.get("schema_version")
        != "cure-lite-stage-a-calibration-receipt-v4"
        or calibration.get("method_order") != list(_METHOD_ORDER)
        or not isinstance(calibration_methods, Mapping)
        or set(calibration_methods) != set(_METHOD_ORDER)
        or calibration.get("receipt_fingerprint")
        != binding.calibration_receipt_fingerprint
    ):
        raise RuntimeError("historical calibration semantics differ from freeze")

    expected_results_fields = {
        "schema_version",
        "method_order",
        "methods",
        "calibration_receipt_fingerprint",
        "results_fingerprint",
    }
    if set(results) != expected_results_fields:
        raise ValueError("historical results receipt fields are not canonical")
    result_methods = results.get("methods")
    results_core = dict(results)
    results_fingerprint = _digest(
        results_core.pop("results_fingerprint", None),
        name="historical results_fingerprint",
    )
    if (
        results.get("schema_version")
        != "cure-lite-stage-a-results-receipt-v3"
        or results.get("method_order") != list(_METHOD_ORDER)
        or not isinstance(result_methods, Mapping)
        or set(result_methods) != set(_METHOD_ORDER)
        or results.get("calibration_receipt_fingerprint")
        != binding.calibration_receipt_fingerprint
        or stable_fingerprint(results_core) != results_fingerprint
        or results_fingerprint != binding.results_fingerprint
    ):
        raise RuntimeError("historical results semantics differ from freeze")

    method_sources: list[FrozenHistoricalMethodSource] = []
    for method in HISTORICAL_COMPARATORS:
        calibration_method = calibration_methods[method]
        if not isinstance(calibration_method, Mapping):
            raise TypeError(f"historical calibration method {method} is invalid")
        source_protocol_fingerprint = _digest(
            calibration_method.get("receipt_fingerprint"),
            name=f"historical calibration {method} receipt_fingerprint",
        )
        if source_protocol_fingerprint != binding.protocol_fingerprint_for(
            method
        ):
            raise RuntimeError(
                f"historical {method} protocol differs from frozen binding"
            )
        selected = result_methods[method]
        selected_tuple = _selected_metrics(selected)
        normalized = dict(selected_tuple)
        if stable_fingerprint(normalized) != binding.metrics_fingerprint_for(
            method
        ):
            raise RuntimeError(
                f"historical {method} metrics differ from frozen binding"
            )
        protocol_metrics = calibration_method.get("protocol")
        if isinstance(protocol_metrics, Mapping):
            protocol_metrics = protocol_metrics.get("selected_metrics")
        else:
            protocol_metrics = calibration_method.get("selected_metrics")
        if protocol_metrics != selected:
            raise RuntimeError(
                f"historical {method} calibration/results metrics differ"
            )
        source = FrozenHistoricalMethodSource(
            method=method,
            seed=seed,
            selected_metrics=selected_tuple,
            source_results_fingerprint=binding.results_fingerprint,
            source_protocol_fingerprint=source_protocol_fingerprint,
            source_run_config_fingerprint=binding.run_config_fingerprint,
            source_config_file_sha256=binding.config_file_sha256,
            source_calibration_receipt_fingerprint=(
                binding.calibration_receipt_fingerprint
            ),
            source_complete_fingerprint=binding.complete_fingerprint,
        )
        # This is a receipt-only consistency check; the adapter performs no
        # evaluation or D_V access.
        source.adapt(comparison_protocol)
        method_sources.append(source)

    return FrozenHistoricalFXV3Source(
        seed=seed,
        run_root=run,
        protocol_root=protocol,
        methods=tuple(method_sources),
        stage_a_config_sha256=config_sha,
        decision_rule_sha256=decision_sha,
    )


def load_frozen_historical_fx_v3_sources(
    *,
    seed42_run_root: str | Path,
    seed42_protocol_root: str | Path,
    seed43_run_root: str | Path,
    seed43_protocol_root: str | Path,
    comparison_protocol: FrozenComparisonProtocol,
    repository_root: str | Path = _REPO_ROOT,
) -> FrozenHistoricalFXV3Sources:
    """Load exactly the authoritative seed-42 and seed-43 receipt pair."""

    sources = tuple(
        load_frozen_historical_fx_v3_source(
            run_root,
            protocol_root,
            comparison_protocol=comparison_protocol,
            seed=seed,
            repository_root=repository_root,
        )
        for seed, run_root, protocol_root in (
            (42, seed42_run_root, seed42_protocol_root),
            (43, seed43_run_root, seed43_protocol_root),
        )
    )
    return FrozenHistoricalFXV3Sources(sources=sources)


__all__ = [
    "FrozenHistoricalFXV3Source",
    "FrozenHistoricalFXV3Sources",
    "FrozenHistoricalMethodSource",
    "load_frozen_historical_fx_v3_source",
    "load_frozen_historical_fx_v3_sources",
]
