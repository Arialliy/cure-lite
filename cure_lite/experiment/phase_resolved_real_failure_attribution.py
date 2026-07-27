"""D_R-only failure attribution for completed real PFCR attempts.

The formal D_V reveal is immutable.  This module therefore consumes only the
strictly loaded D_R state catalog and the two completed seed-42/43 decoder
attempts.  It asks three implementation-level questions:

* does the local feature--coverage relation participate at positive targets;
* does the residual field rank target pixels above valid background;
* do the two independently initialized decoders fail on the same targets.

The output is diagnostic evidence.  It never authorizes D_V/D_T access,
another formal run, Full CURE, or a particular replacement equation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from ..cache.schema import file_sha256, stable_fingerprint
from ..decoder import project_occupancy_to_feature_grid
from ..phase_resolved_feature_coverage_relation import (
    directional_occupancy_basis,
)
from ..phase_resolved_real_cache import PFCRRealCacheAdapter
from ..phase_resolved_real_states import PFCRRealStateCatalog
from ..train.pools import StateExample
from .phase_resolved_real_formal_runner import (
    PublishedPFCRRealFormalAttempt,
    load_pfcr_real_formal_attempt,
)


PFCR_REAL_FAILURE_ATTRIBUTION_SCHEMA = (
    "cure-lite-pfcr-real-d-r-failure-attribution-v1"
)
PFCR_REAL_FAILURE_ATTRIBUTION_COMPLETE_SCHEMA = (
    "cure-lite-pfcr-real-d-r-failure-attribution-complete-v1"
)
_BRANCHES = ("factual_miss", "factual_no_miss", "synthetic")
_RESULT_NAME = "result.json"
_COMPLETE_NAME = "COMPLETE.json"
_HEX = frozenset("0123456789abcdef")
_INTERPRETATION_BOUNDARY = {
    "diagnostic_only": True,
    "causal_failure_attribution_established": False,
    "replacement_equation_authorized": False,
    "new_formal_training_authorized": False,
    "D_V_read": False,
    "D_T_read": False,
    "full_CURE_authorized": False,
    "cross_backbone_authorized": False,
}


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


@dataclass(frozen=True, eq=False)
class PFCRAttributionState:
    """One immutable D_R state with a stable target identity."""

    branch: str
    identity: tuple[str, int | None, int | None]
    example: StateExample

    def __post_init__(self) -> None:
        if self.branch not in _BRANCHES:
            raise ValueError("unknown PFCR attribution branch")
        sample_id, gt_id, pred_id = self.identity
        valid_gt = gt_id is None or (
            not isinstance(gt_id, bool)
            and isinstance(gt_id, int)
            and gt_id >= 1
        )
        valid_pred = pred_id is None or (
            not isinstance(pred_id, bool)
            and isinstance(pred_id, int)
            and pred_id >= 1
        )
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or not valid_gt
            or not valid_pred
            or not isinstance(self.example, StateExample)
            or self.example.sample_id != sample_id
            or self.example.supervision.branch != self.branch
        ):
            raise ValueError("invalid PFCR attribution state")
        if self.branch == "factual_no_miss":
            if gt_id is not None or pred_id is not None:
                raise ValueError("no-miss attribution identity must be target-free")
        elif gt_id is None:
            raise ValueError("positive attribution state requires a GT identity")
        if self.branch == "factual_miss" and pred_id is not None:
            raise ValueError("factual attribution identity has no prediction ID")
        if self.branch == "synthetic" and pred_id is None:
            raise ValueError("synthetic attribution identity requires prediction ID")


def build_pfcr_attribution_states(
    catalog: PFCRRealStateCatalog,
) -> tuple[PFCRAttributionState, ...]:
    """Enumerate the exact factual/no-miss/lineage-safe synthetic population."""

    if not isinstance(catalog, PFCRRealStateCatalog):
        raise TypeError("catalog must be PFCRRealStateCatalog")
    catalog.verify_unchanged()
    rows: list[PFCRAttributionState] = []
    for entry, selected_indices in zip(
        catalog.prepared.entries,
        catalog.selected_legal_indices,
        strict=True,
    ):
        for gt_id, example in zip(
            entry.reachable_gt_ids,
            entry.factual_examples,
            strict=True,
        ):
            rows.append(
                PFCRAttributionState(
                    branch="factual_miss",
                    identity=(entry.sample_id, gt_id, None),
                    example=example,
                )
            )
        if entry.factual_no_miss_example is not None:
            rows.append(
                PFCRAttributionState(
                    branch="factual_no_miss",
                    identity=(entry.sample_id, None, None),
                    example=entry.factual_no_miss_example,
                )
            )
        for index in selected_indices:
            candidate = entry.decoder_visible_legal_candidates[index]
            rows.append(
                PFCRAttributionState(
                    branch="synthetic",
                    identity=(
                        entry.sample_id,
                        candidate.gt_id,
                        candidate.pred_id,
                    ),
                    example=entry.synthetic_examples[index],
                )
            )
    expected = (
        catalog.factual_target_count
        + catalog.factual_no_miss_source_count
        + catalog.legal_target_count
    )
    if len(rows) != expected:
        raise RuntimeError("PFCR attribution population does not close")
    identities = tuple((row.branch, *row.identity) for row in rows)
    if len(identities) != len(set(identities)):
        raise RuntimeError("PFCR attribution identities are not unique")
    catalog.verify_unchanged()
    return tuple(rows)


def _positive_feature_cells(
    target: Tensor,
    *,
    feature_shape: tuple[int, int],
    output_shape: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    if (
        target.dtype != torch.bool
        or target.ndim != 2
        or tuple(target.shape) != output_shape
        or output_shape[0] % feature_shape[0]
        or output_shape[1] % feature_shape[1]
    ):
        raise ValueError("target/output/feature grids are incompatible")
    stride_h = output_shape[0] // feature_shape[0]
    stride_w = output_shape[1] // feature_shape[1]
    coordinates = torch.nonzero(target, as_tuple=False)
    return tuple(
        sorted(
            {
                (int(y) // stride_h, int(x) // stride_w)
                for y, x in coordinates.tolist()
            }
        )
    )


def local_occupancy_support_row(
    state: PFCRAttributionState,
    *,
    feature_shape: tuple[int, int],
) -> dict[str, object]:
    """Measure local occupancy support at a positive target.

    This is a geometry/input diagnostic.  It does not claim that a learned
    affinity, burden, or release value is non-zero.
    """

    if not isinstance(state, PFCRAttributionState):
        raise TypeError("state must be PFCRAttributionState")
    supervision = state.example.supervision
    target = supervision.target[0].to(dtype=torch.bool, device="cpu")
    if not bool(target.any()):
        raise ValueError("local coverage participation requires a positive state")
    occupancy = supervision.occupancy.unsqueeze(0).to(device="cpu")
    projected = project_occupancy_to_feature_grid(
        occupancy,
        feature_shape,
    )
    basis = directional_occupancy_basis(projected)
    local_active = basis.any(dim=1)[0]
    cells = _positive_feature_cells(
        target,
        feature_shape=feature_shape,
        output_shape=tuple(int(value) for value in target.shape),
    )
    flags = tuple(bool(local_active[y, x]) for y, x in cells)
    if not flags:
        raise RuntimeError("positive attribution state has no feature cells")
    return {
        "branch": state.branch,
        "identity": list(state.identity),
        "positive_pixel_count": int(target.sum()),
        "positive_feature_cell_count": len(flags),
        "active_positive_feature_cell_count": sum(flags),
        "active_positive_feature_cell_fraction": sum(flags) / len(flags),
        "any_local_occupancy_support": any(flags),
        "all_positive_cells_local_occupancy_support": all(flags),
    }


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    if (
        not values
        or isinstance(probability, bool)
        or not isinstance(probability, float)
        or not 0.0 <= probability <= 1.0
    ):
        raise ValueError("quantile inputs are invalid")
    ordered = sorted(float(value) for value in values)
    if any(not isfinite(value) for value in ordered):
        raise ValueError("quantile values must be finite")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    normalized = tuple(float(value) for value in values)
    if not normalized or any(not isfinite(value) for value in normalized):
        raise ValueError("summary requires finite values")
    return {
        "count": len(normalized),
        "minimum": min(normalized),
        "q25": _linear_quantile(normalized, 0.25),
        "median": _linear_quantile(normalized, 0.5),
        "mean": sum(normalized) / len(normalized),
        "q75": _linear_quantile(normalized, 0.75),
        "maximum": max(normalized),
    }


def _local_support_summaries(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for branch in ("factual_miss", "synthetic"):
        branch_rows = tuple(
            row for row in rows if row.get("branch") == branch
        )
        if not branch_rows:
            raise ValueError(
                f"local occupancy support has no {branch} rows"
            )
        positive_cells = sum(
            int(row["positive_feature_cell_count"])
            for row in branch_rows
        )
        active_cells = sum(
            int(row["active_positive_feature_cell_count"])
            for row in branch_rows
        )
        if positive_cells <= 0:
            raise ValueError("local occupancy support has no target cells")
        result[branch] = {
            "target_count": len(branch_rows),
            "any_local_occupancy_support_count": sum(
                bool(row["any_local_occupancy_support"])
                for row in branch_rows
            ),
            "zero_local_occupancy_support_count": sum(
                not bool(row["any_local_occupancy_support"])
                for row in branch_rows
            ),
            "positive_feature_cell_count": positive_cells,
            "active_positive_feature_cell_count": active_cells,
            "macro_mean_active_positive_feature_cell_fraction": (
                sum(
                    float(
                        row["active_positive_feature_cell_fraction"]
                    )
                    for row in branch_rows
                )
                / len(branch_rows)
            ),
            "active_positive_feature_cell_fraction": (
                active_cells / positive_cells
            ),
        }
    return result


def residual_score_row(
    probability: Tensor,
    state: PFCRAttributionState,
) -> dict[str, object]:
    """Return target/background endpoint ordering for one residual field."""

    if (
        not isinstance(probability, Tensor)
        or not probability.is_floating_point()
        or probability.ndim != 4
        or tuple(probability.shape[:2]) != (1, 1)
        or not bool(torch.isfinite(probability).all())
        or bool((probability < 0.0).any())
        or bool((probability > 1.0).any())
    ):
        raise ValueError(
            "probability must be finite [0,1] floating [1,1,H,W]"
        )
    supervision = state.example.supervision
    output_shape = tuple(int(value) for value in probability.shape[-2:])
    if tuple(supervision.target.shape[-2:]) != output_shape:
        raise ValueError("probability and attribution state grids differ")
    target = supervision.target.unsqueeze(0).to(
        device=probability.device,
        dtype=torch.bool,
    )
    valid = supervision.valid_mask.unsqueeze(0).to(
        device=probability.device,
    )
    occupancy = supervision.occupancy.unsqueeze(0).to(
        device=probability.device,
    )
    writable = valid & ~occupancy
    positive = target & writable
    negative = ~target & writable
    if not bool(negative.any()):
        raise ValueError("attribution state has no valid background")
    background = probability[negative]
    result: dict[str, object] = {
        "branch": state.branch,
        "identity": list(state.identity),
        "background_max": float(background.max().cpu()),
        "background_q999": float(
            torch.quantile(background.float(), 0.999).cpu()
        ),
    }
    if bool(positive.any()):
        target_scores = probability[positive]
        target_min = float(target_scores.min().cpu())
        target_max = float(target_scores.max().cpu())
        result.update(
            {
                "target_min": target_min,
                "target_mean": float(target_scores.mean().cpu()),
                "target_max": target_max,
                "margin_target_min_vs_background_max": (
                    target_min - result["background_max"]
                ),
                "margin_target_max_vs_background_q999": (
                    target_max - result["background_q999"]
                ),
            }
        )
    return result


def _average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized or any(not isfinite(value) for value in normalized):
        raise ValueError("ranks require finite values")
    order = sorted(range(len(normalized)), key=lambda index: normalized[index])
    ranks = [0.0] * len(normalized)
    start = 0
    while start < len(order):
        end = start + 1
        while (
            end < len(order)
            and normalized[order[end]] == normalized[order[start]]
        ):
            end += 1
        average = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return tuple(ranks)


def spearman_correlation(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    """Exact average-rank Spearman correlation; constant input is undefined."""

    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must align and contain two values")
    x = _average_ranks(left)
    y = _average_ranks(right)
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    centered_x = tuple(value - mean_x for value in x)
    centered_y = tuple(value - mean_y for value in y)
    norm_x = sum(value * value for value in centered_x) ** 0.5
    norm_y = sum(value * value for value in centered_y) ** 0.5
    if norm_x == 0.0 or norm_y == 0.0:
        return None
    return sum(
        a * b for a, b in zip(centered_x, centered_y, strict=True)
    ) / (norm_x * norm_y)


def _branch_summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("branch summary cannot be empty")
    result: dict[str, object] = {
        "state_count": len(rows),
        "background_max": _summary(
            [float(row["background_max"]) for row in rows]
        ),
        "background_q999": _summary(
            [float(row["background_q999"]) for row in rows]
        ),
    }
    positive = [row for row in rows if "target_min" in row]
    if positive:
        for name in (
            "target_min",
            "target_mean",
            "target_max",
            "margin_target_min_vs_background_max",
            "margin_target_max_vs_background_q999",
        ):
            result[name] = _summary(
                [float(row[name]) for row in positive]
            )
        result["target_min_above_background_max_count"] = sum(
            float(row["margin_target_min_vs_background_max"]) > 0.0
            for row in positive
        )
        result["target_max_above_background_q999_count"] = sum(
            float(row["margin_target_max_vs_background_q999"]) > 0.0
            for row in positive
        )
    return result


def _evaluate_attempt(
    attempt: PublishedPFCRRealFormalAttempt,
    states: Sequence[PFCRAttributionState],
    *,
    device: torch.device,
) -> dict[str, object]:
    if not isinstance(attempt, PublishedPFCRRealFormalAttempt):
        raise TypeError("attempt must be a strictly loaded PFCR attempt")
    attempt.artifact.verify_unchanged()
    decoder = attempt.artifact.decoder.to(device)
    decoder.eval()
    rows: list[dict[str, object]] = []
    try:
        with torch.no_grad():
            for state in states:
                supervision = state.example.supervision
                fields = decoder._forward_fields(
                    state.example.feature.to(device),
                    supervision.occupancy.unsqueeze(0).to(device),
                    audit=False,
                )
                rows.append(
                    residual_score_row(
                        fields.completion_probability,
                        state,
                    )
                )
    finally:
        decoder.to("cpu")
    attempt.artifact.verify_unchanged()
    grouped = {
        branch: tuple(row for row in rows if row["branch"] == branch)
        for branch in _BRANCHES
    }
    return {
        "seed": attempt.seed,
        "attempt_complete_fingerprint": attempt.complete_fingerprint,
        "decoder_artifact_fingerprint": (
            attempt.artifact.artifact_fingerprint
        ),
        "decoder_state_fingerprint": (
            attempt.artifact.decoder_state_fingerprint
        ),
        "rows": rows,
        "branch_summaries": {
            branch: _branch_summary(grouped[branch])
            for branch in _BRANCHES
        },
    }


def _seed_consistency(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> dict[str, object]:
    def factual_rows(payload: Mapping[str, object]) -> list[Mapping[str, object]]:
        rows = payload["rows"]
        assert isinstance(rows, list)
        return [
            row
            for row in rows
            if isinstance(row, Mapping)
            and row.get("branch") == "factual_miss"
        ]

    left = factual_rows(first)
    right = factual_rows(second)
    left_ids = [row["identity"] for row in left]
    right_ids = [row["identity"] for row in right]
    if left_ids != right_ids or not left:
        raise RuntimeError("seed factual target populations differ")
    margin_names = (
        "margin_target_min_vs_background_max",
        "margin_target_max_vs_background_q999",
    )
    result: dict[str, object] = {
        "factual_target_count": len(left),
    }
    for name in margin_names:
        left_values = [float(row[name]) for row in left]
        right_values = [float(row[name]) for row in right]
        left_labels = [value > 0.0 for value in left_values]
        right_labels = [value > 0.0 for value in right_values]
        result[name] = {
            "spearman": spearman_correlation(left_values, right_values),
            "sign_agreement_count": sum(
                a == b
                for a, b in zip(left_labels, right_labels, strict=True)
            ),
            "both_fail_identities": [
                left[index]["identity"]
                for index, (a, b) in enumerate(
                    zip(left_labels, right_labels, strict=True)
                )
                if not a and not b
            ],
        }
    return result


def _implementation_binding() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        root / "cure_lite/phase_resolved_feature_coverage_relation.py",
        root / "cure_lite/phase_resolved_relation_decoder.py",
        root / "cure_lite/phase_resolved_real_cache.py",
        root / "cure_lite/phase_resolved_real_states.py",
        root
        / "cure_lite/experiment/phase_resolved_real_formal_runner.py",
    )
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in paths
    }


def _execution_binding(
    device_by_seed: Mapping[int, str | torch.device],
) -> tuple[dict[int, torch.device], dict[str, object]]:
    if set(device_by_seed) != {42, 43}:
        raise ValueError("attribution requires exact seed devices 42/43")
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            "PFCR attribution requires deterministic algorithms"
        )
    if (
        torch.backends.cudnn.benchmark
        or not torch.backends.cudnn.deterministic
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
    ):
        raise RuntimeError(
            "PFCR attribution deterministic backend policy changed"
        )
    normalized: dict[int, torch.device] = {}
    device_payload: dict[str, str] = {}
    device_names: dict[str, str] = {}
    for seed in (42, 43):
        device = torch.device(device_by_seed[seed])
        if device.type == "cuda":
            if (
                device.index is None
                or not torch.cuda.is_available()
                or device.index >= torch.cuda.device_count()
            ):
                raise RuntimeError(
                    f"seed {seed} requires an available explicit CUDA device"
                )
            device_names[str(seed)] = torch.cuda.get_device_name(
                device.index
            )
        elif device.type == "cpu" and device.index is None:
            device_names[str(seed)] = "cpu"
        else:
            raise ValueError("attribution device must be cpu or cuda:<index>")
        normalized[seed] = device
        device_payload[str(seed)] = str(device)
    return normalized, {
        "seed_order": [42, 43],
        "device_by_seed": device_payload,
        "device_name_by_seed": device_names,
        "torch_version": str(torch.__version__),
        "cuda_version": (
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),
    }


def run_pfcr_real_d_r_failure_attribution(
    cache: PFCRRealCacheAdapter,
    catalog: PFCRRealStateCatalog,
    attempts: Sequence[PublishedPFCRRealFormalAttempt],
    *,
    device_by_seed: Mapping[int, str | torch.device],
) -> dict[str, object]:
    """Run the exact D_R-only two-seed PFCR diagnostic in memory."""

    if not isinstance(cache, PFCRRealCacheAdapter):
        raise TypeError("cache must be PFCRRealCacheAdapter")
    if not isinstance(catalog, PFCRRealStateCatalog):
        raise TypeError("catalog must be PFCRRealStateCatalog")
    cache.verify_unchanged()
    catalog.verify_unchanged()
    if (
        catalog.prepared is not cache.prepared_catalog
        or catalog.cache_contract_fingerprint
        != cache.contract.contract_fingerprint
        or stable_fingerprint(catalog.canonical_payload())
        != catalog.catalog_fingerprint
    ):
        raise RuntimeError(
            "PFCR attribution catalog is not bound to its D_R cache"
        )
    attempt_tuple = tuple(attempts)
    if (
        len(attempt_tuple) != 2
        or tuple(attempt.seed for attempt in attempt_tuple) != (42, 43)
    ):
        raise ValueError("attribution requires ordered seeds 42/43 and devices")
    reloaded_attempts: list[PublishedPFCRRealFormalAttempt] = []
    for attempt in attempt_tuple:
        if not isinstance(attempt, PublishedPFCRRealFormalAttempt):
            raise TypeError("attempt must be a strictly loaded PFCR attempt")
        attempt.artifact.verify_unchanged()
        reloaded = load_pfcr_real_formal_attempt(attempt.root)
        if (
            reloaded.seed != attempt.seed
            or reloaded.complete_fingerprint
            != attempt.complete_fingerprint
            or reloaded.run_receipt_fingerprint
            != attempt.run_receipt_fingerprint
            or reloaded.artifact.artifact_fingerprint
            != attempt.artifact.artifact_fingerprint
            or reloaded.artifact.decoder_state_fingerprint
            != attempt.artifact.decoder_state_fingerprint
        ):
            raise RuntimeError(
                "PFCR attribution attempt is not its strict disk load"
            )
        run_config = reloaded.artifact.config
        if (
            reloaded.seed != run_config.seed
            or reloaded.seed != run_config.training_config.seed
            or run_config.cache_contract_fingerprint
            != catalog.cache_contract_fingerprint
            or run_config.state_catalog_fingerprint
            != catalog.catalog_fingerprint
            or run_config.lineage_allowlist_fingerprint
            != catalog.allowlist.allowlist_fingerprint
        ):
            raise RuntimeError(
                "PFCR attribution catalog differs from a formal attempt"
            )
        reloaded_attempts.append(reloaded)
    attempt_tuple = tuple(reloaded_attempts)
    if (
        attempt_tuple[0].complete_fingerprint
        == attempt_tuple[1].complete_fingerprint
        or attempt_tuple[0].artifact.decoder_state_fingerprint
        == attempt_tuple[1].artifact.decoder_state_fingerprint
    ):
        raise RuntimeError("PFCR attribution attempts are not independent")
    devices, execution_binding = _execution_binding(device_by_seed)
    states = build_pfcr_attribution_states(catalog)
    feature_shape = tuple(
        int(value)
        for value in catalog.prepared.sources[0].feature.shape[-2:]
    )
    positive_states = tuple(
        state for state in states if state.branch != "factual_no_miss"
    )
    local_rows = tuple(
        local_occupancy_support_row(
            state,
            feature_shape=feature_shape,
        )
        for state in positive_states
    )
    local_summary = _local_support_summaries(local_rows)

    seed_results = tuple(
        _evaluate_attempt(
            attempt,
            states,
            device=devices[attempt.seed],
        )
        for attempt in attempt_tuple
    )
    core = {
        "schema_version": PFCR_REAL_FAILURE_ATTRIBUTION_SCHEMA,
        "model": "CURE-Lite",
        "method": "PFCR",
        "runtime_split": "D_R",
        "cache_contract_fingerprint": (
            catalog.cache_contract_fingerprint
        ),
        "state_catalog_fingerprint": catalog.catalog_fingerprint,
        "lineage_allowlist_fingerprint": (
            catalog.allowlist.allowlist_fingerprint
        ),
        "population": {
            "factual_miss_targets": catalog.factual_target_count,
            "factual_no_miss_states": (
                catalog.factual_no_miss_source_count
            ),
            "synthetic_targets": catalog.legal_target_count,
        },
        "local_occupancy_support": {
            "rows": list(local_rows),
            "summaries": local_summary,
            "mathematical_boundary": (
                "this reports only geometry/input occupancy support; when "
                "the basis is zero the implemented coverage burden is zero, "
                "but non-zero support alone does not prove learned burden"
            ),
        },
        "seed_results": list(seed_results),
        "seed_consistency": _seed_consistency(
            seed_results[0],
            seed_results[1],
        ),
        "execution_binding": execution_binding,
        "implementation_binding": _implementation_binding(),
        "interpretation_boundary": dict(_INTERPRETATION_BOUNDARY),
    }
    return {
        **core,
        "result_fingerprint": stable_fingerprint(core),
    }


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


def _strict_json(path: Path, *, name: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")

    def reject_duplicates(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{name} contains non-finite value {value}")

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one object")
    return value


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _validate_identity(
    value: object,
    *,
    branch: str,
) -> tuple[str, int | None, int | None]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("attribution identity must be a three-item list")
    sample_id, gt_id, pred_id = value
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("attribution sample identity is invalid")
    for name, item in (("gt_id", gt_id), ("pred_id", pred_id)):
        if item is not None:
            _exact_int(item, name=name, minimum=1)
    if branch == "factual_miss":
        if gt_id is None or pred_id is not None:
            raise ValueError("factual miss identity is invalid")
    elif branch == "factual_no_miss":
        if gt_id is not None or pred_id is not None:
            raise ValueError("factual no-miss identity is invalid")
    elif branch == "synthetic":
        if gt_id is None or pred_id is None:
            raise ValueError("synthetic identity is invalid")
    else:
        raise ValueError("attribution branch is invalid")
    return sample_id, gt_id, pred_id


def _validate_local_row(row: object) -> tuple[object, ...]:
    if not isinstance(row, Mapping) or set(row) != {
        "branch",
        "identity",
        "positive_pixel_count",
        "positive_feature_cell_count",
        "active_positive_feature_cell_count",
        "active_positive_feature_cell_fraction",
        "any_local_occupancy_support",
        "all_positive_cells_local_occupancy_support",
    }:
        raise ValueError("local occupancy support row fields changed")
    branch = row["branch"]
    if branch not in {"factual_miss", "synthetic"}:
        raise ValueError("local occupancy support branch changed")
    identity = _validate_identity(row["identity"], branch=str(branch))
    pixels = _exact_int(
        row["positive_pixel_count"],
        name="positive_pixel_count",
        minimum=1,
    )
    cells = _exact_int(
        row["positive_feature_cell_count"],
        name="positive_feature_cell_count",
        minimum=1,
    )
    active = _exact_int(
        row["active_positive_feature_cell_count"],
        name="active_positive_feature_cell_count",
    )
    fraction = _finite_float(
        row["active_positive_feature_cell_fraction"],
        name="active_positive_feature_cell_fraction",
    )
    if (
        active > cells
        or fraction != active / cells
        or row["any_local_occupancy_support"] is not (active > 0)
        or row["all_positive_cells_local_occupancy_support"]
        is not (active == cells)
    ):
        raise ValueError("local occupancy support row does not close")
    return (branch, *identity)


def _validate_residual_row(row: object) -> tuple[object, ...]:
    if not isinstance(row, Mapping):
        raise ValueError("residual attribution row must be a mapping")
    branch = row.get("branch")
    positive = branch in {"factual_miss", "synthetic"}
    expected = {
        "branch",
        "identity",
        "background_max",
        "background_q999",
    }
    if positive:
        expected |= {
            "target_min",
            "target_mean",
            "target_max",
            "margin_target_min_vs_background_max",
            "margin_target_max_vs_background_q999",
        }
    if set(row) != expected or branch not in _BRANCHES:
        raise ValueError("residual attribution row fields changed")
    identity = _validate_identity(row["identity"], branch=str(branch))
    background_max = _finite_float(
        row["background_max"],
        name="background_max",
    )
    background_q999 = _finite_float(
        row["background_q999"],
        name="background_q999",
    )
    if (
        not 0.0 <= background_q999 <= background_max <= 1.0
    ):
        raise ValueError("residual background endpoints are invalid")
    if positive:
        target_min = _finite_float(row["target_min"], name="target_min")
        target_mean = _finite_float(
            row["target_mean"],
            name="target_mean",
        )
        target_max = _finite_float(row["target_max"], name="target_max")
        margin_min = _finite_float(
            row["margin_target_min_vs_background_max"],
            name="margin_target_min_vs_background_max",
        )
        margin_max = _finite_float(
            row["margin_target_max_vs_background_q999"],
            name="margin_target_max_vs_background_q999",
        )
        if (
            not 0.0 <= target_min <= target_mean <= target_max <= 1.0
            or margin_min != target_min - background_max
            or margin_max != target_max - background_q999
        ):
            raise ValueError("residual target endpoints do not close")
    return (branch, *identity)


def _validate_execution_binding(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "seed_order",
        "device_by_seed",
        "device_name_by_seed",
        "torch_version",
        "cuda_version",
        "deterministic_algorithms",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "CUBLAS_WORKSPACE_CONFIG",
    }:
        raise ValueError("PFCR attribution execution binding changed")
    if (
        value["seed_order"] != [42, 43]
        or value["deterministic_algorithms"] is not True
        or value["cudnn_deterministic"] is not True
        or value["cudnn_benchmark"] is not False
        or value["cuda_matmul_allow_tf32"] is not False
        or value["cudnn_allow_tf32"] is not False
        or value["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}
        or not isinstance(value["torch_version"], str)
        or not value["torch_version"]
        or (
            value["cuda_version"] is not None
            and not isinstance(value["cuda_version"], str)
        )
    ):
        raise ValueError("PFCR attribution backend policy changed")
    for field in ("device_by_seed", "device_name_by_seed"):
        payload = value[field]
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"42", "43"}
            or any(
                not isinstance(item, str) or not item
                for item in payload.values()
            )
        ):
            raise ValueError(f"PFCR attribution {field} changed")


def _verify_result(result: Mapping[str, object]) -> str:
    if not isinstance(result, Mapping):
        raise TypeError("PFCR attribution result must be a mapping")
    expected_top = {
        "schema_version",
        "model",
        "method",
        "runtime_split",
        "cache_contract_fingerprint",
        "state_catalog_fingerprint",
        "lineage_allowlist_fingerprint",
        "population",
        "local_occupancy_support",
        "seed_results",
        "seed_consistency",
        "execution_binding",
        "implementation_binding",
        "interpretation_boundary",
        "result_fingerprint",
    }
    if set(result) != expected_top:
        raise ValueError("PFCR attribution result fields changed")
    core = dict(result)
    fingerprint = _digest(
        core.pop("result_fingerprint"),
        name="result_fingerprint",
    )
    if (
        stable_fingerprint(core) != fingerprint
        or core["schema_version"]
        != PFCR_REAL_FAILURE_ATTRIBUTION_SCHEMA
        or core["model"] != "CURE-Lite"
        or core["method"] != "PFCR"
        or core["runtime_split"] != "D_R"
        or core["interpretation_boundary"]
        != _INTERPRETATION_BOUNDARY
    ):
        raise ValueError("PFCR attribution result fingerprint/identity changed")
    for name in (
        "cache_contract_fingerprint",
        "state_catalog_fingerprint",
        "lineage_allowlist_fingerprint",
    ):
        _digest(core[name], name=name)

    population = core["population"]
    if not isinstance(population, Mapping) or set(population) != {
        "factual_miss_targets",
        "factual_no_miss_states",
        "synthetic_targets",
    }:
        raise ValueError("PFCR attribution population fields changed")
    population_counts = {
        name: _exact_int(population[name], name=name, minimum=1)
        for name in population
    }

    local = core["local_occupancy_support"]
    if not isinstance(local, Mapping) or set(local) != {
        "rows",
        "summaries",
        "mathematical_boundary",
    }:
        raise ValueError("local occupancy support payload changed")
    local_rows = local["rows"]
    if not isinstance(local_rows, list):
        raise ValueError("local occupancy support rows must be a list")
    local_identities = tuple(_validate_local_row(row) for row in local_rows)
    if (
        len(local_identities) != len(set(local_identities))
        or sum(
            row[0] == "factual_miss" for row in local_identities
        )
        != population_counts["factual_miss_targets"]
        or sum(row[0] == "synthetic" for row in local_identities)
        != population_counts["synthetic_targets"]
        or local["summaries"] != _local_support_summaries(local_rows)
        or not isinstance(local["mathematical_boundary"], str)
        or not local["mathematical_boundary"]
    ):
        raise ValueError("local occupancy support payload does not close")

    seed_results = core["seed_results"]
    if not isinstance(seed_results, list) or len(seed_results) != 2:
        raise ValueError("PFCR attribution requires two seed results")
    seed_identities: list[tuple[tuple[object, ...], ...]] = []
    for expected_seed, seed_result in zip(
        (42, 43),
        seed_results,
        strict=True,
    ):
        if not isinstance(seed_result, Mapping) or set(seed_result) != {
            "seed",
            "attempt_complete_fingerprint",
            "decoder_artifact_fingerprint",
            "decoder_state_fingerprint",
            "rows",
            "branch_summaries",
        }:
            raise ValueError("PFCR seed result fields changed")
        if seed_result["seed"] != expected_seed:
            raise ValueError("PFCR seed result order changed")
        for name in (
            "attempt_complete_fingerprint",
            "decoder_artifact_fingerprint",
            "decoder_state_fingerprint",
        ):
            _digest(seed_result[name], name=name)
        rows = seed_result["rows"]
        if not isinstance(rows, list):
            raise ValueError("PFCR residual rows must be a list")
        identities = tuple(_validate_residual_row(row) for row in rows)
        if (
            len(identities) != len(set(identities))
            or sum(row[0] == "factual_miss" for row in identities)
            != population_counts["factual_miss_targets"]
            or sum(row[0] == "factual_no_miss" for row in identities)
            != population_counts["factual_no_miss_states"]
            or sum(row[0] == "synthetic" for row in identities)
            != population_counts["synthetic_targets"]
        ):
            raise ValueError("PFCR residual population does not close")
        grouped = {
            branch: tuple(
                row for row in rows if row["branch"] == branch
            )
            for branch in _BRANCHES
        }
        expected_summaries = {
            branch: _branch_summary(grouped[branch])
            for branch in _BRANCHES
        }
        if seed_result["branch_summaries"] != expected_summaries:
            raise ValueError("PFCR branch summaries do not close")
        seed_identities.append(identities)
    if (
        seed_identities[0] != seed_identities[1]
        or seed_results[0]["attempt_complete_fingerprint"]
        == seed_results[1]["attempt_complete_fingerprint"]
        or seed_results[0]["decoder_state_fingerprint"]
        == seed_results[1]["decoder_state_fingerprint"]
        or core["seed_consistency"]
        != _seed_consistency(seed_results[0], seed_results[1])
    ):
        raise ValueError("PFCR two-seed evidence does not close")

    _validate_execution_binding(core["execution_binding"])
    implementation = core["implementation_binding"]
    if (
        not isinstance(implementation, Mapping)
        or not implementation
        or any(
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or _digest(digest, name=f"implementation[{path!r}]")
            != digest
            for path, digest in implementation.items()
        )
    ):
        raise ValueError("PFCR implementation binding changed")
    return fingerprint


@dataclass(frozen=True, slots=True)
class _PublishedPFCRAttributionSeal:
    root: Path
    result: Mapping[str, object]
    result_fingerprint: str
    complete_fingerprint: str
    result_file_sha256: str
    complete_file_sha256: str


@dataclass(frozen=True)
class PublishedPFCRRealFailureAttribution:
    """Strictly loaded, complete D_R-only diagnostic evidence."""

    root: Path
    result: Mapping[str, object]
    result_fingerprint: str
    complete_fingerprint: str
    result_file_sha256: str
    complete_file_sha256: str
    _verification_token: object

    def _seal(self) -> _PublishedPFCRAttributionSeal:
        seal = self._verification_token
        if type(seal) is not _PublishedPFCRAttributionSeal:
            raise TypeError(
                "PFCR attribution must come from its strict loader"
            )
        if seal.result is not self.result:
            raise TypeError("PFCR attribution result object was replaced")
        return seal

    def __post_init__(self) -> None:
        self.verify_unchanged()

    def verify_unchanged(self) -> None:
        seal = self._seal()
        for name in (
            "root",
            "result_fingerprint",
            "complete_fingerprint",
            "result_file_sha256",
            "complete_file_sha256",
        ):
            if getattr(seal, name) != getattr(self, name):
                raise RuntimeError(
                    "published PFCR attribution binding changed"
                )
        if _verify_result(self.result) != self.result_fingerprint:
            raise RuntimeError("PFCR attribution changed in memory")
        root = self.root
        if (
            root.is_symlink()
            or root.resolve(strict=True) != root
            or not root.is_dir()
        ):
            raise RuntimeError(
                "published PFCR attribution directory changed"
            )
        members = {path.name: path for path in root.iterdir()}
        if (
            set(members) != {_RESULT_NAME, _COMPLETE_NAME}
            or any(
                path.is_symlink() or not path.is_file()
                for path in members.values()
            )
        ):
            raise RuntimeError(
                "published PFCR attribution inventory changed"
            )
        if (
            file_sha256(members[_RESULT_NAME])
            != self.result_file_sha256
            or file_sha256(members[_COMPLETE_NAME])
            != self.complete_file_sha256
        ):
            raise RuntimeError(
                "published PFCR attribution files changed"
            )


def publish_pfcr_real_d_r_failure_attribution(
    output_dir: str | Path,
    result: Mapping[str, object],
) -> PublishedPFCRRealFailureAttribution:
    """Create one diagnostic directory without overwrite or continuation."""

    result_fingerprint = _verify_result(result)
    raw = Path(output_dir).expanduser()
    if raw.exists() or raw.is_symlink():
        raise FileExistsError("refusing to reuse PFCR attribution output")
    parent = raw.parent.resolve(strict=True)
    target = parent / raw.name
    target.mkdir(exist_ok=False)
    result_path = target / _RESULT_NAME
    try:
        with result_path.open("xb") as handle:
            handle.write(_json_bytes(result))
            handle.flush()
            os.fsync(handle.fileno())
        complete_core = {
            "schema_version": (
                PFCR_REAL_FAILURE_ATTRIBUTION_COMPLETE_SCHEMA
            ),
            "execution_status": "complete",
            "runtime_split": "D_R",
            "result_file": _RESULT_NAME,
            "result_file_sha256": file_sha256(result_path),
            "result_fingerprint": result_fingerprint,
            "complete_written_last": True,
            "continuation_supported": False,
            "directory_reuse_allowed": False,
            "D_V_read": False,
            "D_T_read": False,
        }
        complete = {
            **complete_core,
            "complete_fingerprint": stable_fingerprint(complete_core),
        }
        complete_path = target / _COMPLETE_NAME
        with complete_path.open("xb") as handle:
            handle.write(_json_bytes(complete))
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        parent_descriptor = os.open(
            parent,
            os.O_RDONLY | os.O_DIRECTORY,
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except Exception:
        # A partial create-only directory remains intentionally unusable and
        # may not be resumed or reused.
        raise
    return load_published_pfcr_real_d_r_failure_attribution(target)


def load_published_pfcr_real_d_r_failure_attribution(
    output_dir: str | Path,
) -> PublishedPFCRRealFailureAttribution:
    """Strictly load an exact COMPLETE-last PFCR D_R diagnostic."""

    raw = Path(output_dir).expanduser()
    if raw.is_symlink():
        raise ValueError("PFCR attribution directory may not be a symlink")
    root = raw.resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("PFCR attribution output must be a regular directory")
    members = {path.name: path for path in root.iterdir()}
    if set(members) != {_RESULT_NAME, _COMPLETE_NAME} or any(
        path.is_symlink() or not path.is_file()
        for path in members.values()
    ):
        raise ValueError("PFCR attribution output inventory changed")
    result = _strict_json(
        members[_RESULT_NAME],
        name="PFCR attribution result",
    )
    result_fingerprint = _verify_result(result)
    complete = _strict_json(
        members[_COMPLETE_NAME],
        name="PFCR attribution COMPLETE",
    )
    complete_core = dict(complete)
    complete_fingerprint = complete_core.pop(
        "complete_fingerprint",
        None,
    )
    if (
        not isinstance(complete_fingerprint, str)
        or stable_fingerprint(complete_core) != complete_fingerprint
        or complete_core
        != {
            "schema_version": (
                PFCR_REAL_FAILURE_ATTRIBUTION_COMPLETE_SCHEMA
            ),
            "execution_status": "complete",
            "runtime_split": "D_R",
            "result_file": _RESULT_NAME,
            "result_file_sha256": file_sha256(members[_RESULT_NAME]),
            "result_fingerprint": result_fingerprint,
            "complete_written_last": True,
            "continuation_supported": False,
            "directory_reuse_allowed": False,
            "D_V_read": False,
            "D_T_read": False,
        }
    ):
        raise ValueError("PFCR attribution COMPLETE changed")
    result_file_sha256 = file_sha256(members[_RESULT_NAME])
    complete_file_sha256 = file_sha256(members[_COMPLETE_NAME])
    seal = _PublishedPFCRAttributionSeal(
        root=root,
        result=result,
        result_fingerprint=result_fingerprint,
        complete_fingerprint=complete_fingerprint,
        result_file_sha256=result_file_sha256,
        complete_file_sha256=complete_file_sha256,
    )
    return PublishedPFCRRealFailureAttribution(
        root=root,
        result=result,
        result_fingerprint=result_fingerprint,
        complete_fingerprint=complete_fingerprint,
        result_file_sha256=result_file_sha256,
        complete_file_sha256=complete_file_sha256,
        _verification_token=seal,
    )


__all__ = [
    "PFCRAttributionState",
    "PFCR_REAL_FAILURE_ATTRIBUTION_COMPLETE_SCHEMA",
    "PFCR_REAL_FAILURE_ATTRIBUTION_SCHEMA",
    "PublishedPFCRRealFailureAttribution",
    "build_pfcr_attribution_states",
    "load_published_pfcr_real_d_r_failure_attribution",
    "local_occupancy_support_row",
    "publish_pfcr_real_d_r_failure_attribution",
    "residual_score_row",
    "run_pfcr_real_d_r_failure_attribution",
    "spearman_correlation",
]
