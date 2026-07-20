from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from cure_lite.calibration import FalseAlarmBudget
from cure_lite.cure import (
    CURECalibrationSample,
    CUREModel,
    CUREProtocol,
    CUREResidualConfig,
    CUREResidualDecoder,
    evaluate_cure_threshold,
    select_cure_threshold,
)
from cure_lite.cure.calibration import CURETestEvaluationLedger
from cure_lite.cure.protocol import module_state_fingerprint
from cure_lite.provenance import BaseCheckpointSelection
from cure_lite.splits import SplitManifest, SplitRecord
from cure_lite.toy import ToyFrozenBaseAdapter


def _protocol(
    base: ToyFrozenBaseAdapter,
    *,
    suppression_radius: int = 0,
    initial_residual_probability: float = 0.2,
) -> CUREProtocol:
    manifest = SplitManifest(
        dataset="toy",
        records=(
            SplitRecord("base-fit", "D_B", "base-fit-group", "base-fit.png"),
            SplitRecord(
                "base-select", "D_B", "base-select-group", "base-select.png"
            ),
            SplitRecord("source", "D_R", "source-group", "source.png"),
            SplitRecord("image", "D_V", "validation-group", "validation.png"),
            SplitRecord("test", "D_T", "test-group", "test.png"),
        ),
    )
    return CUREProtocol.from_manifest(
        manifest,
        base_fingerprint="toy-base-checkpoint",
        base_state_fingerprint=module_state_fingerprint(base),
        adapter_fingerprint=base.fingerprint,
        preprocessing_fingerprint="toy-preprocessing",
        residual_config=CUREResidualConfig(
            feature_channels=3,
            width=8,
            groups=4,
            occupancy_threshold=0.5,
            suppression_radius=suppression_radius,
            initial_residual_probability=initial_residual_probability,
        ),
        base_checkpoint_selection=BaseCheckpointSelection.from_manifest(
            manifest,
            fit_sample_ids=("base-fit",),
            select_sample_ids=("base-select",),
        ),
    )


def _model(
    *,
    suppression_radius: int = 0,
    initial_residual_probability: float = 0.2,
) -> CUREModel:
    base = ToyFrozenBaseAdapter()
    protocol = _protocol(
        base,
        suppression_radius=suppression_radius,
        initial_residual_probability=initial_residual_probability,
    )
    decoder = CUREResidualDecoder(protocol.residual_config)
    return CUREModel(base, decoder, protocol)


def _image_and_gt() -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.zeros((1, 1, 7, 7), dtype=torch.float32)
    image[0, 0, 1, 1] = 1.0  # frozen-base probability 0.9
    image[0, 0, 5, 5] = 0.375  # frozen-base probability 0.4
    gt = torch.zeros((7, 7), dtype=torch.bool)
    gt[1, 1] = True
    gt[5, 5] = True
    return image, gt


def _sample(model: CUREModel, *, split_role: str = "D_V") -> CURECalibrationSample:
    image, gt = _image_and_gt()
    return CURECalibrationSample.from_model(
        "image" if split_role == "D_V" else "test",
        image,
        gt,
        split_role=split_role,
        model=model,
    )


def _budget() -> FalseAlarmBudget:
    return FalseAlarmBudget(
        pixel_fa_budget=1.0,
        component_fa_per_mp_budget=float("inf"),
        raw_background_fa_budget=1.0,
        minimum_retention=1.0,
    )


def _selection(model: CUREModel):
    return select_cure_threshold(
        (_sample(model),),
        (0.45, 0.5),
        model.protocol,
        _budget(),
        model=model,
    )


def test_canonical_factory_runs_exact_model_and_binds_content() -> None:
    model = _model()
    model.train()
    sample = _sample(model)
    assert model.training
    assert sample.provenance == "cure-model-execution-v1"
    assert sample.base_state_fingerprint == model.base_state_fingerprint
    output = model(_image_and_gt()[0])
    torch.testing.assert_close(sample.base_probability, output.base_probability[0, 0])
    torch.testing.assert_close(
        sample.effective_residual_probability,
        output.residual_probability[0, 0],
    )


def test_noisy_or_calibration_recovers_anchor_miss() -> None:
    model = _model()
    metrics = evaluate_cure_threshold(
        (_sample(model),),
        0.5,
        model.protocol,
        model=model,
    )
    assert metrics.total_anchor_misses == 1
    assert metrics.recovered_anchor_misses == 1
    assert metrics.retention == 1.0


def test_selection_freezes_cure_and_returns_paired_test_metrics_once(
    tmp_path: Path,
) -> None:
    model = _model()
    selection = _selection(model)
    assert selection.feasible
    assert selection.metrics is not None
    assert selection.metrics.recovered_anchor_misses == 1
    frozen = selection.frozen_protocol
    assert frozen is not None
    assert frozen.base_at_budget.budget == _budget()

    ledger = CURETestEvaluationLedger(tmp_path / "dt-ledger")
    result = frozen.evaluate_test(
        (_sample(model, split_role="D_T"),),
        ledger=ledger,
        experiment_id="toy-exp-001",
    )
    assert result.cure.recovered_anchor_misses == 1
    assert result.base_at_budget.recovered_anchor_misses == 0
    assert len(tuple((tmp_path / "dt-ledger").glob("*.json"))) == 1

    # Re-instantiating the ledger and changing the experiment label cannot
    # reopen the same frozen protocol.
    with pytest.raises(RuntimeError, match="already consumed"):
        frozen.evaluate_test(
            (_sample(model, split_role="D_T"),),
            ledger=CURETestEvaluationLedger(tmp_path / "dt-ledger"),
            experiment_id="toy-exp-002",
        )
    with pytest.raises(RuntimeError, match="standalone Base@B"):
        frozen.base_at_budget.evaluate_test((_sample(model, split_role="D_T"),))


def test_dt_ledger_marker_is_process_persistent(tmp_path: Path) -> None:
    ledger_root = tmp_path / "cross-process-ledger"
    ledger = CURETestEvaluationLedger(ledger_root)
    frozen_fingerprint = "a" * 64
    sample_fingerprint = "b" * 64
    ledger._consume(
        experiment_id="parent-process",
        frozen_protocol_fingerprint=frozen_fingerprint,
        arm="paired_cure_base_at_budget",
        sample_set_fingerprint=sample_fingerprint,
    )
    code = f"""
from pathlib import Path
from cure_lite.cure.calibration import CURETestEvaluationLedger
ledger = CURETestEvaluationLedger(Path({str(ledger_root)!r}))
try:
    ledger._consume(
        experiment_id='child-process',
        frozen_protocol_fingerprint={'a' * 64!r},
        arm='paired_cure_base_at_budget',
        sample_set_fingerprint={'b' * 64!r},
    )
except RuntimeError:
    raise SystemExit(0)
raise SystemExit(1)
"""
    environment = os.environ.copy()
    package_parent = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (package_parent, environment.get("PYTHONPATH", ""))
        if part
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=package_parent,
        env=environment,
        check=True,
    )


def test_threshold_search_rejects_test_samples(tmp_path: Path) -> None:
    model = _model()
    with pytest.raises(ValueError, match="D_V"):
        select_cure_threshold(
            (_sample(model, split_role="D_T"),),
            (0.5,),
            model.protocol,
            _budget(),
            model=model,
        )
    with pytest.raises(RuntimeError, match="frozen protocol"):
        evaluate_cure_threshold(
            (_sample(model, split_role="D_T"),),
            0.5,
            model.protocol,
            model=model,
            split_role="D_T",
        )
    with pytest.raises(ValueError, match="absolute"):
        CURETestEvaluationLedger(Path("relative-ledger"))


def test_final_threshold_cannot_drop_base_occupancy() -> None:
    model = _model()
    with pytest.raises(ValueError, match="occupancy_threshold"):
        evaluate_cure_threshold(
            (_sample(model),),
            0.6,
            model.protocol,
            model=model,
        )


def test_lower_threshold_base_only_recovery_is_not_residual_supported() -> None:
    model = _model()
    metrics = evaluate_cure_threshold(
        (_sample(model),),
        0.4,
        model.protocol,
        model=model,
    )
    assert metrics.recovered_anchor_misses == 1
    assert metrics.overlap_supported_rmr == 0.0


def test_residual_pixels_do_not_claim_an_already_base_recovered_target() -> None:
    model = _model()
    image = torch.zeros((1, 1, 7, 7), dtype=torch.float32)
    image[0, 0, 1, 1] = 1.0
    image[0, 0, 5, 5] = 0.375  # p_b=0.4: base recovers at tau_f
    image[0, 0, 5, 6] = 0.25  # p_b=0.3; residual adds this GT pixel
    gt = torch.zeros((7, 7), dtype=torch.bool)
    gt[1, 1] = True
    gt[5, 5:7] = True
    sample = CURECalibrationSample.from_model(
        "image", image, gt, split_role="D_V", model=model
    )
    metrics = evaluate_cure_threshold(
        (sample,),
        0.4,
        model.protocol,
        model=model,
    )
    assert metrics.recovered_anchor_misses == 1
    assert metrics.overlap_supported_rmr == 0.0


def test_raw_map_fixture_is_private_and_formal_evaluation_rejects_it() -> None:
    model = _model(suppression_radius=1)
    actual = _sample(model)
    invalid_residual = actual.effective_residual_probability.clone()
    invalid_residual[1, 2] = 0.1
    with pytest.raises(ValueError, match="exclusion mask"):
        CURECalibrationSample._bind_maps_for_test_only(
            actual.sample_id,
            actual.base_probability,
            invalid_residual,
            actual.gt_mask,
            split_role="D_V",
            protocol=model.protocol,
            decoder=model.decoder,
        )

    fixture = CURECalibrationSample._bind_maps_for_test_only(
        actual.sample_id,
        actual.base_probability,
        actual.effective_residual_probability,
        actual.gt_mask,
        split_role="D_V",
        protocol=model.protocol,
        decoder=model.decoder,
    )
    with pytest.raises(ValueError, match="from_model"):
        evaluate_cure_threshold(
            (fixture,), 0.5, model.protocol, model=model
        )


def test_frozen_protocol_and_calibration_receipts_reject_bypass(
    tmp_path: Path,
) -> None:
    model = _model()
    frozen = _selection(model).frozen_protocol
    assert frozen is not None
    with pytest.raises(ValueError, match="content differs"):
        replace(frozen, final_threshold=0.3)

    tampered = _sample(model, split_role="D_T")
    tampered.effective_residual_probability[0, 0] = 0.3
    with pytest.raises(ValueError, match="content differs"):
        frozen.evaluate_test(
            (tampered,),
            ledger=CURETestEvaluationLedger(tmp_path / "tampered-ledger"),
            experiment_id="tampered",
        )

    other_decoder = CUREResidualDecoder(model.protocol.residual_config)
    other_model = CUREModel(
        ToyFrozenBaseAdapter(), other_decoder, model.protocol
    )
    other_sample = _sample(other_model, split_role="D_T")
    with pytest.raises(ValueError, match="different residual decoder"):
        frozen.evaluate_test(
            (other_sample,),
            ledger=CURETestEvaluationLedger(tmp_path / "other-ledger"),
            experiment_id="decoder-substitution",
        )


def test_dt_retuning_is_rejected_before_ledger_is_consumed(tmp_path: Path) -> None:
    model = _model()
    frozen = _selection(model).frozen_protocol
    assert frozen is not None
    ledger_root = tmp_path / "retuning-ledger"
    ledger = CURETestEvaluationLedger(ledger_root)
    test_sample = _sample(model, split_role="D_T")
    with pytest.raises(RuntimeError, match="retuning"):
        frozen.evaluate_test(
            (test_sample,),
            ledger=ledger,
            experiment_id="retuned",
            proposed_final_threshold=0.2,
        )
    assert not ledger_root.exists()
    result = frozen.evaluate_test(
        (test_sample,),
        ledger=ledger,
        experiment_id="frozen",
    )
    assert result.cure.images == 1
