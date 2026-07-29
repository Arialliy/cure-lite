"""Frozen repository-source closure for v24 bounded and Formal execution.

The closure is deliberately shared by both execution stages.  It covers the
repository-owned training, evaluation, serialization, cache, metric, model,
and protocol-verifier implementation reachable from either runner.  External
package versions are recorded by the execution receipts; this module binds
the bytes that are under this repository's control.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Final

from cure_lite.cache.schema import file_sha256, stable_fingerprint

from .dr_gate import GCR_PACRE_DR_IMPLEMENTATION_PATHS


GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-unified-source-closure-v1"
)

# Keep this inventory explicit and reviewable.  A newly introduced
# repository-owned runtime dependency must be added here before an
# authorization can be issued.
GCR_PACRE_V24_SOURCE_CLOSURE_PATHS: Final = tuple(
    sorted(
        {
            *GCR_PACRE_DR_IMPLEMENTATION_PATHS,
            "cure_lite/experiment/coverage_state_zero_level_evaluation.py",
            "cure_lite_v23/bounded_runner.py",
            "cure_lite_v23/formal_evaluation.py",
            "cure_lite_v23/formal_training.py",
            "cure_lite_v23/training.py",
            "cure_lite_v24/artifact_io.py",
            "cure_lite_v24/bounded_run_start.py",
            "cure_lite_v24/bounded_runner.py",
            "cure_lite_v24/fixed_dr_evaluator.py",
            "cure_lite_v24/formal_artifacts.py",
            "cure_lite_v24/formal_cache_artifacts.py",
            "cure_lite_v24/formal_run_start.py",
            "cure_lite_v24/formal_training.py",
            "cure_lite_v24/oof_cache.py",
            "cure_lite_v24/oof_evaluation.py",
            "cure_lite_v24/oof_inputs.py",
            "cure_lite_v24/oof_run_start.py",
            "cure_lite_v24/oof_runner.py",
            "cure_lite_v24/oof_split.py",
            "cure_lite_v24/oof_training.py",
            "cure_lite_v24/real_input_factory.py",
            "cure_lite_v24/source_closure.py",
            "cure_lite_v24/terminal_evidence.py",
            "cure_lite_v24/training.py",
            "cure_lite_v24/training_trace.py",
            "tools/prepare_cure_lite_v24_gcr_pacre_training_chain.py",
            "tools/run_cure_lite_v24_gcr_pacre_bounded_400.py",
            "tools/run_cure_lite_v24_gcr_pacre_formal_800.py",
            "tools/run_cure_lite_v24_gcr_pacre_oof4.py",
        }
    )
)


def _repository_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("v24 repository root must be a canonical directory")
    return root


def gcr_pacre_v24_loaded_runtime_source_paths() -> tuple[str, ...]:
    """Return repo-owned Python modules loaded by a v24 runtime process.

    This is intentionally based on ``sys.modules[*].__file__`` rather than
    import-string guesses.  Isolated bounded/Formal subprocess audits can
    therefore prove that every actually loaded repository module is present
    in the frozen inventory.  Test modules and unrelated ``tools`` modules
    are outside the runtime namespace filter.
    """

    root = _repository_root()
    exact_tool_modules = {
        "tools.gcr_pacre_v24_protocol",
        "tools.prepare_cure_lite_v24_gcr_pacre_training_chain",
        "tools.run_cure_lite_v24_gcr_pacre_bounded_400",
        "tools.run_cure_lite_v24_gcr_pacre_formal_800",
        "tools.run_cure_lite_v24_gcr_pacre_oof4",
    }
    paths: set[str] = set()
    for module_name, module in tuple(sys.modules.items()):
        if not (
            module_name == "cure_lite"
            or module_name.startswith("cure_lite.")
            or module_name == "cure_lite_v23"
            or module_name.startswith("cure_lite_v23.")
            or module_name == "cure_lite_v24"
            or module_name.startswith("cure_lite_v24.")
            or module_name in exact_tool_modules
        ):
            continue
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str) or not raw_path:
            continue
        try:
            relative = Path(raw_path).resolve(strict=True).relative_to(root)
        except (FileNotFoundError, ValueError):
            continue
        if relative.suffix == ".py":
            paths.add(relative.as_posix())
    return tuple(sorted(paths))


def audit_gcr_pacre_v24_loaded_source_closure() -> dict[str, object]:
    """Audit the current process's concrete loaded-module closure."""

    loaded = gcr_pacre_v24_loaded_runtime_source_paths()
    inventory = GCR_PACRE_V24_SOURCE_CLOSURE_PATHS
    missing = tuple(sorted(set(loaded) - set(inventory)))
    return {
        "schema_version": (
            "cure-lite-v24-gcr-pacre-loaded-source-closure-audit-v1"
        ),
        "loaded_source_paths": list(loaded),
        "frozen_inventory_paths": list(inventory),
        "missing_from_frozen_inventory": list(missing),
        "missing_count": len(missing),
        "passed": not missing,
    }


def assert_gcr_pacre_v24_loaded_source_closure_complete() -> None:
    """Fail closed if any loaded repo runtime module is outside inventory."""

    audit = audit_gcr_pacre_v24_loaded_source_closure()
    if audit["missing_count"] != 0 or audit["passed"] is not True:
        raise RuntimeError(
            "v24 loaded repository source closure is incomplete: "
            + ", ".join(audit["missing_from_frozen_inventory"])
        )


def gcr_pacre_v24_source_closure_hashes(
) -> tuple[tuple[str, str], ...]:
    """Hash every exact regular file in the unified v24 source closure."""

    assert_gcr_pacre_v24_loaded_source_closure_complete()
    if (
        tuple(sorted(GCR_PACRE_V24_SOURCE_CLOSURE_PATHS))
        != GCR_PACRE_V24_SOURCE_CLOSURE_PATHS
        or len(set(GCR_PACRE_V24_SOURCE_CLOSURE_PATHS))
        != len(GCR_PACRE_V24_SOURCE_CLOSURE_PATHS)
    ):
        raise AssertionError(
            "v24 source closure paths must be sorted and unique"
        )
    root = _repository_root()
    rows: list[tuple[str, str]] = []
    for relative in GCR_PACRE_V24_SOURCE_CLOSURE_PATHS:
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.resolve(strict=True) != path
        ):
            raise RuntimeError(
                f"v24 source closure entry is not regular: {relative}"
            )
        rows.append((relative, file_sha256(path)))
    return tuple(rows)


def gcr_pacre_v24_source_closure_fingerprint(
    rows: tuple[tuple[str, str], ...] | None = None,
) -> str:
    """Return the schema-bound fingerprint of an exact closure snapshot."""

    resolved = (
        gcr_pacre_v24_source_closure_hashes()
        if rows is None
        else rows
    )
    if tuple(name for name, _ in resolved) != (
        GCR_PACRE_V24_SOURCE_CLOSURE_PATHS
    ):
        raise ValueError("v24 source closure inventory changed")
    return stable_fingerprint(
        {
            "schema_version": GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA,
            "source_hashes": dict(resolved),
        }
    )


__all__ = [
    "GCR_PACRE_V24_SOURCE_CLOSURE_PATHS",
    "GCR_PACRE_V24_SOURCE_CLOSURE_SCHEMA",
    "assert_gcr_pacre_v24_loaded_source_closure_complete",
    "audit_gcr_pacre_v24_loaded_source_closure",
    "gcr_pacre_v24_loaded_runtime_source_paths",
    "gcr_pacre_v24_source_closure_fingerprint",
    "gcr_pacre_v24_source_closure_hashes",
]
