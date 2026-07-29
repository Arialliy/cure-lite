"""Concrete, selection-free evaluator for the v24 D_R OOF-4 gate.

The evaluator uses the same occupancy, matching, connected-component, IoU,
retention, and false-alarm primitives as Formal evaluation.  It emits
per-image additive sufficient statistics so four holdout folds can be pooled
without averaging fold metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.config import MatchConfig, OccupancyConfig
from cure_lite.experiment.coverage_state_real_dr_inputs import (
    CoverageStateRealDRInputs,
)
from cure_lite.experiment.coverage_state_training import (
    coverage_state_model_fingerprint,
)
from cure_lite.instances import instances_from_binary_mask
from cure_lite.metrics import (
    ImageEvaluation,
    evaluate_binary_prediction_from_instances,
)
from cure_lite.paired_types import tensor_content_fingerprint
from cure_lite_v23.pacre_vc import (
    CURELitePACREVerifierCorrectedLevelSet,
)
from tools.gcr_pacre_v24_protocol import (
    BASE_A_THRESHOLD,
    BASE_B_THRESHOLD_GRID,
    FactualSufficientStatistics,
    OOF_ARMS,
    select_base_b_train_fold_threshold,
)

from .gcr_pacre import CURELiteGatedCommonResidualPACRELevelSet
from .oof_split import (
    VerifiedOOFFoldClosure,
    require_verified_oof_fold_closure,
)
from .source_closure import GCR_PACRE_V24_SOURCE_CLOSURE_PATHS


OOF_EVALUATION_DATASET_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-oof-evaluation-dataset-v1"
)
OOF_EVALUATION_LEDGER_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-oof-factual-evaluation-ledger-v1"
)
OOF_EVALUATOR_SCHEMA: Final = (
    "cure-lite-v24-gcr-pacre-fixed-oof-evaluator-v1"
)
OOF_BASE_A_ARM: Final = "BaseA"
OOF_BASE_B_ARM: Final = "BaseB_train_fold_selected"
OOF_V23_ARM: Final = "PACRE_VC_v23_control"
OOF_V24_ARM: Final = "GCR_PACRE_v24"
OOF_G1_ARM: Final = "GCR_PACRE_v24_forced_G1"

_OCCUPANCY = OccupancyConfig(
    threshold=BASE_A_THRESHOLD,
    connectivity=8,
    min_component_area=1,
)
_MATCH = MatchConfig(
    max_distance=3.0,
    distance_quantization=1_000_000,
    iou_quantization=1_000_000,
)
_EVALUATOR_SOURCE_PATHS = tuple(
    sorted(
        {
            *GCR_PACRE_V24_SOURCE_CLOSURE_PATHS,
            "cure_lite/instances.py",
            "cure_lite/matching.py",
            "cure_lite/metrics.py",
            "cure_lite_v23/pacre_vc.py",
            "cure_lite_v24/gcr_pacre.py",
            "cure_lite_v24/oof_evaluation.py",
            "tools/gcr_pacre_v24_protocol.py",
        }
    )
)


def _cpu_clone(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.numel() < 1:
        raise TypeError(f"{name} must be a nonempty tensor")
    result = value.detach().to("cpu").clone().contiguous()
    if result.is_floating_point() and not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def oof_evaluator_source_hashes() -> tuple[tuple[str, str], ...]:
    root = _root()
    rows: list[tuple[str, str]] = []
    for relative in _EVALUATOR_SOURCE_PATHS:
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.resolve(strict=True) != path
        ):
            raise RuntimeError(f"OOF evaluator source is invalid: {relative}")
        rows.append((relative, file_sha256(path)))
    return tuple(rows)


@dataclass(frozen=True, eq=False)
class OOFEvaluationSample:
    sample_id: str
    root_source_id: str
    base_probability: Tensor
    feature: Tensor
    gt_mask: Tensor
    valid_mask: Tensor
    anchor_miss_ids: tuple[int, ...]
    reachable_anchor_miss_ids: tuple[int, ...]
    content_fingerprint: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "root_source_id": self.root_source_id,
            "tensor_fingerprints": {
                "base_probability": tensor_content_fingerprint(
                    self.base_probability
                ),
                "feature": tensor_content_fingerprint(self.feature),
                "gt_mask": tensor_content_fingerprint(self.gt_mask),
                "valid_mask": tensor_content_fingerprint(self.valid_mask),
            },
            "anchor_miss_ids": list(self.anchor_miss_ids),
            "reachable_anchor_miss_ids": list(
                self.reachable_anchor_miss_ids
            ),
        }

    def verify_unchanged(self) -> None:
        if (
            not self.sample_id
            or not self.root_source_id
            or self.base_probability.shape[:2] != (1, 1)
            or self.feature.ndim != 4
            or self.feature.shape[0] != 1
            or self.gt_mask.shape != self.base_probability.shape
            or self.valid_mask.shape != self.base_probability.shape
            or self.gt_mask.dtype != torch.bool
            or self.valid_mask.dtype != torch.bool
            or bool(torch.any(self.gt_mask & ~self.valid_mask))
            or tuple(sorted(set(self.anchor_miss_ids)))
            != self.anchor_miss_ids
            or tuple(sorted(set(self.reachable_anchor_miss_ids)))
            != self.reachable_anchor_miss_ids
            or not set(self.reachable_anchor_miss_ids)
            <= set(self.anchor_miss_ids)
            or stable_fingerprint(self.canonical_payload())
            != self.content_fingerprint
        ):
            raise RuntimeError("OOF evaluation sample changed")


def seal_oof_evaluation_sample(
    *,
    sample_id: str,
    root_source_id: str,
    base_probability: Tensor,
    feature: Tensor,
    gt_mask: Tensor,
    valid_mask: Tensor,
    anchor_miss_ids: Iterable[int],
    reachable_anchor_miss_ids: Iterable[int],
) -> OOFEvaluationSample:
    """Clone and content-seal one generated or real factual row."""

    provisional = OOFEvaluationSample(
        sample_id=sample_id,
        root_source_id=root_source_id,
        base_probability=_cpu_clone(
            base_probability,
            name="base_probability",
        ),
        feature=_cpu_clone(feature, name="feature"),
        gt_mask=_cpu_clone(gt_mask, name="gt_mask").to(torch.bool),
        valid_mask=_cpu_clone(valid_mask, name="valid_mask").to(torch.bool),
        anchor_miss_ids=tuple(sorted(set(anchor_miss_ids))),
        reachable_anchor_miss_ids=tuple(
            sorted(set(reachable_anchor_miss_ids))
        ),
        content_fingerprint="0" * 64,
    )
    result = OOFEvaluationSample(
        sample_id=provisional.sample_id,
        root_source_id=provisional.root_source_id,
        base_probability=provisional.base_probability,
        feature=provisional.feature,
        gt_mask=provisional.gt_mask,
        valid_mask=provisional.valid_mask,
        anchor_miss_ids=provisional.anchor_miss_ids,
        reachable_anchor_miss_ids=provisional.reachable_anchor_miss_ids,
        content_fingerprint=stable_fingerprint(
            provisional.canonical_payload()
        ),
    )
    result.verify_unchanged()
    return result


@dataclass(frozen=True, eq=False)
class OOFEvaluationDataset:
    fold_id: int
    partition: str
    closure_fingerprint: str
    rows: tuple[OOFEvaluationSample, ...]
    dataset_fingerprint: str

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(row.sample_id for row in self.rows)

    @property
    def root_source_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row.root_source_id for row in self.rows}))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": OOF_EVALUATION_DATASET_SCHEMA,
            "fold_id": self.fold_id,
            "partition": self.partition,
            "closure_fingerprint": self.closure_fingerprint,
            "sample_ids": list(self.sample_ids),
            "root_source_ids": list(self.root_source_ids),
            "rows": [row.canonical_payload() for row in self.rows],
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }

    def verify_unchanged(self) -> None:
        if (
            self.partition not in {"train", "holdout"}
            or not self.rows
            or len(self.sample_ids) != len(set(self.sample_ids))
            or tuple(sorted(self.sample_ids)) != self.sample_ids
        ):
            raise ValueError("OOF evaluation dataset identity is invalid")
        for row in self.rows:
            row.verify_unchanged()
        if stable_fingerprint(self.canonical_payload()) != (
            self.dataset_fingerprint
        ):
            raise RuntimeError("OOF evaluation dataset changed")


def seal_oof_evaluation_dataset(
    *,
    fold_id: int,
    partition: str,
    closure_fingerprint: str,
    rows: Iterable[OOFEvaluationSample],
) -> OOFEvaluationDataset:
    """Seal cloned rows into one immutable evaluator input."""

    values = tuple(sorted(rows, key=lambda row: row.sample_id))
    provisional = OOFEvaluationDataset(
        fold_id=fold_id,
        partition=partition,
        closure_fingerprint=closure_fingerprint,
        rows=values,
        dataset_fingerprint="0" * 64,
    )
    result = OOFEvaluationDataset(
        fold_id=fold_id,
        partition=partition,
        closure_fingerprint=closure_fingerprint,
        rows=values,
        dataset_fingerprint=stable_fingerprint(
            provisional.canonical_payload()
        ),
    )
    result.verify_unchanged()
    return result


def build_oof_evaluation_dataset(
    real_inputs: CoverageStateRealDRInputs,
    fold_closure: VerifiedOOFFoldClosure,
    *,
    partition: str,
) -> OOFEvaluationDataset:
    """Clone only one fold partition from the trusted complete D_R graph."""

    if type(real_inputs) is not CoverageStateRealDRInputs:
        raise TypeError("real_inputs must be exact CoverageStateRealDRInputs")
    real_inputs.verify_unchanged()
    closure = require_verified_oof_fold_closure(fold_closure)
    if partition == "train":
        expected = closure.train_sample_ids
    elif partition == "holdout":
        expected = closure.held_out_sample_ids
    else:
        raise ValueError("partition must be train or holdout")
    rows_by_id = {row.sample_id: row for row in real_inputs.bundle.rows}
    if set(rows_by_id) != set(closure.root_by_sample):
        raise PermissionError(
            "real D_R bundle differs from the frozen split sample universe"
        )
    rows: list[OOFEvaluationSample] = []
    for sample_id in sorted(expected):
        source = rows_by_id[sample_id]
        state = source.state.normalized()
        rows.append(seal_oof_evaluation_sample(
            sample_id=sample_id,
            root_source_id=closure.root_by_sample[sample_id],
            base_probability=source.base_output.probability,
            feature=source.base_output.feature,
            gt_mask=state.gt_labels > 0,
            valid_mask=state.image_valid_mask,
            anchor_miss_ids=(
                int(value) for value in state.real_miss_ids.tolist()
            ),
            reachable_anchor_miss_ids=(
                int(value) for value in state.reachable_miss_ids.tolist()
            ),
        ))
    return seal_oof_evaluation_dataset(
        fold_id=closure.fold_id,
        partition=partition,
        closure_fingerprint=closure.closure_fingerprint,
        rows=tuple(rows),
    )


def mechanically_replay_oof_fold_evidence(
    fold_receipt: Mapping[str, object],
    *,
    runtime_root: str | Path,
) -> str:
    """Rebuild schedule, terminals and every factual ledger from artifacts.

    The protocol validator calls this only after it has verified the receipt
    schema, split closure and file receipts.  This function supplies the
    missing semantic boundary: all cache files cross a weights-only neutral
    decoder, both models are rebuilt by their frozen factories from actual
    safetensors, and all 51 BaseB train evaluations plus five holdout ledgers
    are rerun on CPU.  Caller-authored metric JSON is never trusted.
    """

    from cure_lite.coverage_state_precomputed_cache import (
        CoverageStateScalarCache,
    )
    from cure_lite.coverage_state_schedule import (
        CoverageStateScheduleConfig,
        build_coverage_state_training_schedule,
    )

    from .artifact_io import read_canonical_json
    from .oof_cache import load_persisted_oof_cache_payload
    from .oof_run_start import (
        OOF_EPOCHS,
        OOF_SEED,
        OOF_STEPS_PER_EPOCH,
    )
    from .oof_training import (
        OOF_CANDIDATE_ARM,
        OOF_CONTROL_ARM,
        load_oof_terminal_model_strict,
    )

    if not isinstance(fold_receipt, Mapping):
        raise TypeError("OOF fold replay requires a receipt mapping")
    fold_id = fold_receipt.get("fold_id")
    if (
        isinstance(fold_id, bool)
        or not isinstance(fold_id, int)
        or fold_id not in range(4)
    ):
        raise ValueError("OOF fold replay identity is invalid")
    root = Path(runtime_root)
    if not root.is_absolute():
        raise ValueError("OOF fold replay runtime root must be absolute")
    fold_directory = root / f"fold_{fold_id}"
    raw_entries = fold_receipt.get("cache_entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != 6:
        raise ValueError("OOF fold replay requires six cache entries")
    entries: dict[tuple[str, str], Mapping[str, object]] = {}
    caches: dict[tuple[str, str], object] = {}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise TypeError("OOF cache entry must be a mapping")
        key = (str(raw_entry.get("partition")), str(raw_entry.get("arm")))
        if key in entries:
            raise ValueError("duplicate OOF persisted cache slot")
        entries[key] = raw_entry
        caches[key] = load_persisted_oof_cache_payload(
            raw_entry,
            runtime_root=root,
            fold_id=fold_id,
        )
    expected_slots = {
        (partition, arm)
        for partition in ("train", "holdout")
        for arm in ("base_eval", OOF_CONTROL_ARM, OOF_CANDIDATE_ARM)
    }
    if set(caches) != expected_slots:
        raise ValueError("OOF persisted cache slot matrix changed")
    train_base = caches[("train", "base_eval")]
    train_control = caches[("train", OOF_CONTROL_ARM)]
    train_candidate = caches[("train", OOF_CANDIDATE_ARM)]
    holdout_base = caches[("holdout", "base_eval")]
    holdout_control = caches[("holdout", OOF_CONTROL_ARM)]
    holdout_candidate = caches[("holdout", OOF_CANDIDATE_ARM)]
    if (
        type(train_base) is not OOFEvaluationDataset
        or type(train_control) is not CoverageStateScalarCache
        or type(train_candidate) is not CoverageStateScalarCache
        or type(holdout_base) is not OOFEvaluationDataset
        or type(holdout_control) is not OOFEvaluationDataset
        or type(holdout_candidate) is not OOFEvaluationDataset
        or train_control.canonical_payload()
        != train_candidate.canonical_payload()
        or holdout_base.canonical_payload()
        != holdout_control.canonical_payload()
        or holdout_base.canonical_payload()
        != holdout_candidate.canonical_payload()
    ):
        raise PermissionError("OOF decoded cache populations/types differ")

    schedule = build_coverage_state_training_schedule(
        train_control,
        CoverageStateScheduleConfig(
            seed=OOF_SEED,
            epochs=OOF_EPOCHS,
            steps_per_epoch=OOF_STEPS_PER_EPOCH,
        ),
    )
    candidate_schedule = build_coverage_state_training_schedule(
        train_candidate,
        CoverageStateScheduleConfig(
            seed=OOF_SEED,
            epochs=OOF_EPOCHS,
            steps_per_epoch=OOF_STEPS_PER_EPOCH,
        ),
    )
    schedule_path = fold_directory / "schedule.json"
    persisted_schedule = read_canonical_json(schedule_path)
    batch_sequence_fingerprint = stable_fingerprint(
        [
            selection.selection_fingerprint
            for selection in schedule.selections
        ]
    )
    run_start_artifact = fold_receipt.get("run_start_artifact")
    if not isinstance(run_start_artifact, Mapping):
        raise TypeError("OOF run-start artifact is invalid")
    run_start = run_start_artifact.get("payload")
    if (
        not isinstance(run_start, Mapping)
        or schedule_path.stat().st_mode & 0o777 != 0o444
        or persisted_schedule != schedule.canonical_payload()
        or candidate_schedule.canonical_payload()
        != schedule.canonical_payload()
        or run_start.get("schedule_fingerprint")
        != schedule.schedule_fingerprint
        or run_start.get("batch_sequence_fingerprint")
        != batch_sequence_fingerprint
        or run_start.get("training_population_fingerprint")
        != train_control.cache_fingerprint
    ):
        raise PermissionError("OOF persisted 10x40 schedule is not factual")

    evaluator = OOFConcreteEvaluator.fixed()
    selected_threshold, raw_base_rows = (
        evaluator.select_base_b_train_only(train_base)
    )
    selection = fold_receipt.get("BaseB_train_fold_selection")
    if not isinstance(selection, Mapping):
        raise TypeError("OOF BaseB selection receipt is invalid")
    normalized_base_rows = [
        {
            **row,
            "train_sample_ids": list(train_base.sample_ids),
            "train_root_source_ids": list(train_base.root_source_ids),
            "input_train_cache_fingerprint": entries[
                ("train", "base_eval")
            ]["file_sha256"],
            "access_audit_receipt_fingerprint": fold_receipt[
                "access_audit_receipt_fingerprint"
            ],
        }
        for row in raw_base_rows
    ]
    selector_policy = (
        "maximize_pd",
        "maximize_retention",
        "minimize_pixel_fa",
        "minimize_raw_background_fa",
        "minimize_fp_components_per_mp",
        "maximize_threshold",
    )
    if (
        selection.get("threshold_grid") != list(BASE_B_THRESHOLD_GRID)
        or selection.get("candidate_rows") != normalized_base_rows
        or selection.get("candidate_ledger_fingerprint")
        != stable_fingerprint(normalized_base_rows)
        or selection.get("selector_policy") != list(selector_policy)
        or selection.get("selector_policy_fingerprint")
        != stable_fingerprint(list(selector_policy))
        or selection.get("selected_threshold") != selected_threshold
    ):
        raise PermissionError("OOF BaseB train-only selection is not factual")

    training_arms = fold_receipt.get("training_arms")
    if not isinstance(training_arms, Mapping):
        raise TypeError("OOF training arm receipts are invalid")
    terminal_by_arm: dict[str, Mapping[str, object]] = {}
    models: dict[str, nn.Module] = {}
    terminal_names = {
        OOF_CONTROL_ARM: "v23_control_terminal.safetensors",
        OOF_CANDIDATE_ARM: "candidate_terminal.safetensors",
    }
    for arm in (OOF_CONTROL_ARM, OOF_CANDIDATE_ARM):
        arm_receipt = training_arms.get(arm)
        if not isinstance(arm_receipt, Mapping):
            raise TypeError(f"OOF training arm {arm} is invalid")
        terminal = arm_receipt.get("terminal_artifact")
        if not isinstance(terminal, Mapping):
            raise TypeError(f"OOF terminal {arm} is invalid")
        terminal_by_arm[arm] = terminal
        models[arm] = load_oof_terminal_model_strict(
            terminal,
            arm=arm,
            expected_path=(
                fold_directory / "terminal" / terminal_names[arm]
            ),
        )

    ledgers = {
        OOF_BASE_A_ARM: evaluator.evaluate_base(
            holdout_base,
            threshold=BASE_A_THRESHOLD,
            arm=OOF_BASE_A_ARM,
        ),
        OOF_BASE_B_ARM: evaluator.evaluate_base(
            holdout_base,
            threshold=selected_threshold,
            arm=OOF_BASE_B_ARM,
        ),
        OOF_V23_ARM: evaluator.evaluate_model(
            holdout_control,
            models[OOF_CONTROL_ARM],
            arm=OOF_V23_ARM,
            device="cpu",
        ),
        OOF_V24_ARM: evaluator.evaluate_model(
            holdout_candidate,
            models[OOF_CANDIDATE_ARM],
            arm=OOF_V24_ARM,
            device="cpu",
        ),
        OOF_G1_ARM: evaluator.evaluate_model(
            holdout_candidate,
            models[OOF_CANDIDATE_ARM],
            arm=OOF_G1_ARM,
            forced_unit_gate=True,
            device="cpu",
        ),
    }
    for arm, ledger in ledgers.items():
        expected_payload = {
            **ledger.canonical_payload(),
            "ledger_fingerprint": ledger.ledger_fingerprint,
        }
        persisted = read_canonical_json(
            fold_directory / "evaluation" / f"{arm}.json"
        )
        if persisted != expected_payload:
            raise PermissionError(
                f"OOF persisted {arm} ledger differs from mechanical replay"
            )

    evaluation_fingerprints = fold_receipt.get(
        "evaluation_artifact_fingerprints"
    )
    if not isinstance(evaluation_fingerprints, Mapping):
        raise TypeError("OOF evaluation fingerprints are invalid")
    sample_by_id = {row.sample_id: row for row in holdout_base.rows}
    factual_rows = []
    for arm in OOF_ARMS:
        for row in ledgers[arm].per_sample_rows:
            sample = sample_by_id[str(row["sample_id"])]
            valid = sample.valid_mask
            anchor = (sample.base_probability >= BASE_A_THRESHOLD) & valid
            factual_rows.append({
                "split": "D_R",
                "evidence_role": "factual_only",
                "fold_id": fold_id,
                "arm": arm,
                "sample_id": sample.sample_id,
                "root_source_id": sample.root_source_id,
                "gt_fingerprint": tensor_content_fingerprint(
                    sample.gt_mask & valid
                ),
                "anchor_state_fingerprint": tensor_content_fingerprint(
                    anchor
                ),
                "evaluation_contract_fingerprint": (
                    evaluator.evaluator_fingerprint
                ),
                "terminal_artifact_fingerprint": (
                    evaluation_fingerprints[arm]
                ),
                "sufficient_statistics": dict(row["statistics"]),
            })
    factual_artifact = read_canonical_json(
        fold_directory / "factual_rows.json"
    )
    if factual_artifact.get("rows") != factual_rows:
        raise PermissionError(
            "OOF factual rows differ from mechanical evaluator output"
        )
    replay_body = {
        "fold_id": fold_id,
        "schedule_fingerprint": schedule.schedule_fingerprint,
        "batch_sequence_fingerprint": batch_sequence_fingerprint,
        "terminal_model_fingerprints": {
            arm: coverage_state_model_fingerprint(models[arm])
            for arm in (OOF_CONTROL_ARM, OOF_CANDIDATE_ARM)
        },
        "evaluation_ledger_fingerprints": {
            arm: ledgers[arm].ledger_fingerprint for arm in OOF_ARMS
        },
        "factual_rows_fingerprint": stable_fingerprint(factual_rows),
    }
    return stable_fingerprint(replay_body)


def _statistics(
    value: ImageEvaluation,
    *,
    valid_pixel_count: int,
) -> FactualSufficientStatistics:
    if (
        isinstance(valid_pixel_count, bool)
        or not isinstance(valid_pixel_count, int)
        or valid_pixel_count < 1
    ):
        raise ValueError("OOF valid domain must contain at least one pixel")
    return FactualSufficientStatistics(
        images=1,
        matched_gt=value.matched_gt,
        total_gt=value.total_gt,
        recovered_anchor_misses=value.recovered_anchor_misses,
        overlap_supported_recovered_anchor_misses=(
            value.overlap_supported_recovered_anchor_misses
        ),
        total_anchor_misses=value.total_anchor_misses,
        retained_anchor_covered=value.retained_anchor_covered,
        total_anchor_covered=value.total_anchor_covered,
        recovered_reachable_anchor_misses=(
            value.recovered_reachable_anchor_misses
        ),
        total_reachable_anchor_misses=value.total_reachable_anchor_misses,
        unmatched_pred_pixels=value.unmatched_pred_pixels,
        unmatched_pred_components=value.unmatched_pred_components,
        raw_background_fp=value.raw_background_fp,
        # ``ImageEvaluation`` reports the rectangular tensor area.  OOF
        # examples may be padded, so false-alarm rates must instead use the
        # exact factual valid domain after masking.
        total_pixels=valid_pixel_count,
        intersection=value.intersection,
        union=value.union,
        sum_image_iou=value.iou,
    )


def _valid_domain_field_fingerprint(
    field: Tensor,
    valid_mask: Tensor,
) -> str:
    if (
        not isinstance(field, Tensor)
        or not isinstance(valid_mask, Tensor)
        or field.shape != valid_mask.shape
        or valid_mask.dtype != torch.bool
        or not bool(torch.any(valid_mask))
    ):
        raise ValueError("field fingerprint requires a nonempty valid domain")
    field_cpu = field.detach().to("cpu").contiguous()
    valid_cpu = valid_mask.detach().to("cpu").contiguous()
    masked = field_cpu.masked_fill(~valid_cpu, 0).contiguous()
    return stable_fingerprint(
        {
            "schema_version": (
                "cure-lite-v24-oof-valid-domain-field-fingerprint-v1"
            ),
            "valid_mask_fingerprint": tensor_content_fingerprint(valid_cpu),
            "masked_field_fingerprint": tensor_content_fingerprint(masked),
        }
    )


def _pool(
    values: Sequence[FactualSufficientStatistics],
) -> FactualSufficientStatistics:
    if not values:
        raise ValueError("cannot pool an empty OOF evaluation")
    result = values[0]
    for value in values[1:]:
        result = result.plus(value)
    return result


def _role_summary(
    energy: Tensor,
    gate: Tensor,
    residual: Tensor,
    mask: Tensor,
) -> dict[str, object]:
    selected_e = energy[mask]
    selected_g = gate[mask]
    selected_d = residual[mask]
    if selected_e.numel() == 0:
        return {"count": 0}

    def summary(value: Tensor) -> dict[str, float]:
        flat = value.detach().to("cpu", dtype=torch.float64).flatten()
        quantiles = torch.quantile(
            flat,
            torch.tensor(
                [0.01, 0.10, 0.50, 0.90, 0.99],
                dtype=torch.float64,
            ),
        )
        return {
            "min": float(flat.min()),
            "max": float(flat.max()),
            "mean": float(flat.mean()),
            "q01": float(quantiles[0]),
            "q10": float(quantiles[1]),
            "q50": float(quantiles[2]),
            "q90": float(quantiles[3]),
            "q99": float(quantiles[4]),
        }

    return {
        "count": int(selected_e.numel()),
        "E": summary(selected_e),
        "G": summary(selected_g),
        "D": {
            **summary(selected_d.abs()),
            "negative_count": int(torch.count_nonzero(selected_d < 0)),
            "zero_count": int(torch.count_nonzero(selected_d == 0)),
            "positive_count": int(torch.count_nonzero(selected_d > 0)),
        },
        "G_saturation": {
            "equal_zero": int(torch.count_nonzero(selected_g == 0)),
            "equal_two": int(torch.count_nonzero(selected_g == 2)),
            "strict_interior": int(
                torch.count_nonzero((selected_g > 0) & (selected_g < 2))
            ),
        },
        "G_D_sign_contingency": {
            "G_lt_1_D_negative": int(
                torch.count_nonzero((selected_g < 1) & (selected_d < 0))
            ),
            "G_lt_1_D_nonnegative": int(
                torch.count_nonzero((selected_g < 1) & (selected_d >= 0))
            ),
            "G_ge_1_D_negative": int(
                torch.count_nonzero((selected_g >= 1) & (selected_d < 0))
            ),
            "G_ge_1_D_nonnegative": int(
                torch.count_nonzero((selected_g >= 1) & (selected_d >= 0))
            ),
        },
    }


@dataclass(frozen=True)
class OOFEvaluationLedger:
    fold_id: int
    partition: str
    arm: str
    operating_point: float | None
    dataset_fingerprint: str
    model_fingerprint: str | None
    per_sample_rows: tuple[dict[str, object], ...]
    pooled_statistics: FactualSufficientStatistics
    field_ledger_fingerprint: str
    prediction_ledger_fingerprint: str
    role_ledger_fingerprint: str
    evaluator_fingerprint: str
    ledger_fingerprint: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": OOF_EVALUATION_LEDGER_SCHEMA,
            "fold_id": self.fold_id,
            "partition": self.partition,
            "arm": self.arm,
            "operating_point": self.operating_point,
            "dataset_fingerprint": self.dataset_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "per_sample_rows": list(self.per_sample_rows),
            "pooled_statistics": asdict(self.pooled_statistics),
            "field_ledger_fingerprint": self.field_ledger_fingerprint,
            "prediction_ledger_fingerprint": (
                self.prediction_ledger_fingerprint
            ),
            "role_ledger_fingerprint": self.role_ledger_fingerprint,
            "evaluator_fingerprint": self.evaluator_fingerprint,
            "D_V_payload_accessed": False,
            "D_T_payload_accessed": False,
        }

    def verify_unchanged(self) -> None:
        if (
            len(self.per_sample_rows) != self.pooled_statistics.images
            or stable_fingerprint(
                [
                    row["field_fingerprint"]
                    for row in self.per_sample_rows
                ]
            )
            != self.field_ledger_fingerprint
            or stable_fingerprint(
                [
                    row["prediction_fingerprint"]
                    for row in self.per_sample_rows
                ]
            )
            != self.prediction_ledger_fingerprint
            or stable_fingerprint(
                [row["role_statistics"] for row in self.per_sample_rows]
            )
            != self.role_ledger_fingerprint
            or stable_fingerprint(self.canonical_payload())
            != self.ledger_fingerprint
        ):
            raise RuntimeError("OOF evaluation ledger changed")


@dataclass(frozen=True)
class OOFConcreteEvaluator:
    evaluator_fingerprint: str
    source_hashes: tuple[tuple[str, str], ...]

    @classmethod
    def fixed(cls) -> "OOFConcreteEvaluator":
        sources = oof_evaluator_source_hashes()
        fingerprint = stable_fingerprint(
            {
                "schema_version": OOF_EVALUATOR_SCHEMA,
                "occupancy": {
                    "threshold": BASE_A_THRESHOLD,
                    "connectivity": 8,
                    "min_component_area": 1,
                },
                "matching": {
                    "max_distance": 3.0,
                    "distance_quantization": 1_000_000,
                    "iou_quantization": 1_000_000,
                },
                "base_grid": list(BASE_B_THRESHOLD_GRID),
                "completion": "(field<0)&~occupancy",
                "pooling": "additive_sufficient_statistics",
                "source_hashes": dict(sources),
            }
        )
        return cls(fingerprint, sources)

    def verify_unchanged(self) -> None:
        if self.source_hashes != oof_evaluator_source_hashes():
            raise RuntimeError("OOF evaluator source closure changed")
        if self != type(self).fixed():
            raise RuntimeError("OOF evaluator policy changed")

    def _seal(
        self,
        dataset: OOFEvaluationDataset,
        *,
        arm: str,
        operating_point: float | None,
        model_fingerprint: str | None,
        rows: list[dict[str, object]],
        statistics: list[FactualSufficientStatistics],
    ) -> OOFEvaluationLedger:
        pooled = _pool(statistics)
        field_fp = stable_fingerprint(
            [row["field_fingerprint"] for row in rows]
        )
        prediction_fp = stable_fingerprint(
            [row["prediction_fingerprint"] for row in rows]
        )
        role_fp = stable_fingerprint(
            [row["role_statistics"] for row in rows]
        )
        provisional = OOFEvaluationLedger(
            fold_id=dataset.fold_id,
            partition=dataset.partition,
            arm=arm,
            operating_point=operating_point,
            dataset_fingerprint=dataset.dataset_fingerprint,
            model_fingerprint=model_fingerprint,
            per_sample_rows=tuple(rows),
            pooled_statistics=pooled,
            field_ledger_fingerprint=field_fp,
            prediction_ledger_fingerprint=prediction_fp,
            role_ledger_fingerprint=role_fp,
            evaluator_fingerprint=self.evaluator_fingerprint,
            ledger_fingerprint="0" * 64,
        )
        result = OOFEvaluationLedger(
            **{
                **provisional.__dict__,
                "ledger_fingerprint": stable_fingerprint(
                    provisional.canonical_payload()
                ),
            }
        )
        result.verify_unchanged()
        return result

    def evaluate_base(
        self,
        dataset: OOFEvaluationDataset,
        *,
        threshold: float,
        arm: str,
    ) -> OOFEvaluationLedger:
        self.verify_unchanged()
        dataset.verify_unchanged()
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not isfinite(float(threshold))
            or not 0.0 <= float(threshold) <= 1.0
            or arm not in {OOF_BASE_A_ARM, OOF_BASE_B_ARM}
        ):
            raise ValueError("invalid fixed Base operating point")
        rows: list[dict[str, object]] = []
        statistics: list[FactualSufficientStatistics] = []
        for sample in dataset.rows:
            base = sample.base_probability
            valid = sample.valid_mask
            anchor = (base >= BASE_A_THRESHOLD) & valid
            prediction = (base >= float(threshold)) & valid
            gt = sample.gt_mask & valid
            pred_instances = instances_from_binary_mask(
                prediction[0, 0],
                connectivity=8,
                min_area=1,
            )
            gt_instances = instances_from_binary_mask(
                gt[0, 0],
                connectivity=8,
                min_area=1,
            )
            residual = prediction & ~anchor
            image = evaluate_binary_prediction_from_instances(
                prediction,
                gt,
                pred_instances,
                gt_instances,
                _MATCH,
                anchor_miss_ids=frozenset(sample.anchor_miss_ids),
                reachable_anchor_miss_ids=frozenset(
                    sample.reachable_anchor_miss_ids
                ),
                residual_mask=residual,
            )
            stat = _statistics(
                image,
                valid_pixel_count=int(torch.count_nonzero(valid)),
            )
            statistics.append(stat)
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "root_source_id": sample.root_source_id,
                    "statistics": asdict(stat),
                    "field_fingerprint": _valid_domain_field_fingerprint(
                        base,
                        valid,
                    ),
                    "prediction_fingerprint": tensor_content_fingerprint(
                        prediction
                    ),
                    "role_statistics": {},
                }
            )
        return self._seal(
            dataset,
            arm=arm,
            operating_point=float(threshold),
            model_fingerprint=None,
            rows=rows,
            statistics=statistics,
        )

    def select_base_b_train_only(
        self,
        dataset: OOFEvaluationDataset,
    ) -> tuple[float, tuple[dict[str, object], ...]]:
        if dataset.partition != "train":
            raise PermissionError("BaseB selection requires train partition")
        ledgers = tuple(
            self.evaluate_base(
                dataset,
                threshold=threshold,
                arm=OOF_BASE_B_ARM,
            )
            for threshold in BASE_B_THRESHOLD_GRID
        )
        rows = tuple(
            {
                "threshold": threshold,
                "selection_split_role": "OOF_train_fold",
                "metrics": {
                    key: ledger.pooled_statistics.metrics()[key]
                    for key in (
                        "pd",
                        "retention",
                        "pixel_fa",
                        "raw_background_fa",
                        "fp_components_per_mp",
                        "budget_violation",
                    )
                },
                "D_V_payload_accessed": False,
                "D_T_payload_accessed": False,
            }
            for threshold, ledger in zip(
                BASE_B_THRESHOLD_GRID,
                ledgers,
                strict=True,
            )
        )
        return select_base_b_train_fold_threshold(rows), rows

    def evaluate_model(
        self,
        dataset: OOFEvaluationDataset,
        model: nn.Module,
        *,
        arm: str,
        forced_unit_gate: bool = False,
        device: torch.device | str = "cpu",
    ) -> OOFEvaluationLedger:
        self.verify_unchanged()
        dataset.verify_unchanged()
        if forced_unit_gate:
            if (
                type(model) is not CURELiteGatedCommonResidualPACRELevelSet
                or arm != OOF_G1_ARM
            ):
                raise TypeError("forced G1 requires the exact v24 model/arm")
        elif arm == OOF_V24_ARM:
            if type(model) is not CURELiteGatedCommonResidualPACRELevelSet:
                raise TypeError("v24 arm requires exact GCR-PACRE model")
        elif arm == OOF_V23_ARM:
            if type(model) is not CURELitePACREVerifierCorrectedLevelSet:
                raise TypeError("v23 arm requires exact PACRE-VC model")
        else:
            raise ValueError("unknown OOF model arm")
        resolved = torch.device(device)
        before = coverage_state_model_fingerprint(model)
        was_training = model.training
        model.eval()
        rows: list[dict[str, object]] = []
        statistics: list[FactualSufficientStatistics] = []
        try:
            with torch.no_grad():
                for sample in dataset.rows:
                    feature = sample.feature.to(resolved)
                    base = sample.base_probability.to(resolved)
                    valid = sample.valid_mask.to(resolved)
                    occupancy = (base >= BASE_A_THRESHOLD) & valid
                    role_statistics: Mapping[str, object] = {}
                    if type(model) is CURELiteGatedCommonResidualPACRELevelSet:
                        fields = model.forward_fields(feature, occupancy)
                        if forced_unit_gate:
                            field = model.forward_forced_unit_gate(
                                feature,
                                occupancy,
                            )
                            gate = torch.ones_like(fields.common_gate)
                        else:
                            field = fields.field
                            gate = fields.common_gate
                        energy = F.pixel_shuffle(
                            fields.common_even_energy,
                            model.config.feature_stride,
                        )
                        gate_fine = F.pixel_shuffle(
                            gate,
                            model.config.feature_stride,
                        )
                        residual_fine = F.pixel_shuffle(
                            fields.residual_odd_interaction,
                            model.config.feature_stride,
                        )
                        target = sample.gt_mask.to(resolved) & valid
                        role_statistics = {
                            "target": _role_summary(
                                energy,
                                gate_fine,
                                residual_fine,
                                target,
                            ),
                            "background": _role_summary(
                                energy,
                                gate_fine,
                                residual_fine,
                                valid & ~target,
                            ),
                        }
                    else:
                        field = model(feature, occupancy)
                    completion = (field < 0) & ~occupancy & valid
                    prediction = (occupancy | completion) & valid
                    prediction_cpu = prediction.to("cpu")
                    gt = sample.gt_mask & sample.valid_mask
                    pred_instances = instances_from_binary_mask(
                        prediction_cpu[0, 0],
                        connectivity=8,
                        min_area=1,
                    )
                    gt_instances = instances_from_binary_mask(
                        gt[0, 0],
                        connectivity=8,
                        min_area=1,
                    )
                    image = evaluate_binary_prediction_from_instances(
                        prediction_cpu,
                        gt,
                        pred_instances,
                        gt_instances,
                        _MATCH,
                        anchor_miss_ids=frozenset(sample.anchor_miss_ids),
                        reachable_anchor_miss_ids=frozenset(
                            sample.reachable_anchor_miss_ids
                        ),
                        residual_mask=completion.to("cpu"),
                    )
                    stat = _statistics(
                        image,
                        valid_pixel_count=int(
                            torch.count_nonzero(sample.valid_mask)
                        ),
                    )
                    statistics.append(stat)
                    rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "root_source_id": sample.root_source_id,
                            "statistics": asdict(stat),
                            "field_fingerprint": (
                                _valid_domain_field_fingerprint(
                                    field,
                                    valid,
                                )
                            ),
                            "prediction_fingerprint": (
                                tensor_content_fingerprint(prediction_cpu)
                            ),
                            "role_statistics": dict(role_statistics),
                        }
                    )
        finally:
            model.train(was_training)
        if coverage_state_model_fingerprint(model) != before:
            raise RuntimeError("OOF evaluator mutated the terminal model")
        return self._seal(
            dataset,
            arm=arm,
            operating_point=BASE_A_THRESHOLD,
            model_fingerprint=before,
            rows=rows,
            statistics=statistics,
        )


__all__ = [
    "OOFConcreteEvaluator",
    "OOFEvaluationDataset",
    "OOFEvaluationLedger",
    "OOFEvaluationSample",
    "OOF_BASE_A_ARM",
    "OOF_BASE_B_ARM",
    "OOF_EVALUATION_DATASET_SCHEMA",
    "OOF_EVALUATION_LEDGER_SCHEMA",
    "OOF_G1_ARM",
    "OOF_V23_ARM",
    "OOF_V24_ARM",
    "build_oof_evaluation_dataset",
    "mechanically_replay_oof_fold_evidence",
    "oof_evaluator_source_hashes",
    "seal_oof_evaluation_dataset",
    "seal_oof_evaluation_sample",
]
