from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from cure_lite.decoder import CURELiteDecoder
from cure_lite.experiment.efficiency_evidence import (
    EfficiencyBinding,
    StageAEfficiencyReceipt,
    measure_stage_a_efficiency,
    replay_static_efficiency,
)


def _binding() -> EfficiencyBinding:
    return EfficiencyBinding(
        decoder_artifact_fingerprint="a" * 64,
        decoder_state_fingerprint="b" * 64,
        decoder_receipt_sha256="c" * 64,
        base_index_fingerprint="d" * 64,
        preprocessing_fingerprint="e" * 64,
    )


def _cpu_receipt(decoder: CURELiteDecoder) -> StageAEfficiencyReceipt:
    return measure_stage_a_efficiency(
        decoder,
        decoder_variant="uniform_legal",
        binding=_binding(),
        feature_shape=(1, 3, 4, 4),
        occupancy_shape=(1, 1, 8, 8),
        device="cpu",
        warmup=1,
        repetitions=3,
    )


def test_stage_a_efficiency_is_u_only_zero_input_and_round_trips() -> None:
    decoder = CURELiteDecoder(feature_channels=3)
    decoder.train()
    state_before = {
        name: tensor.detach().clone() for name, tensor in decoder.state_dict().items()
    }

    receipt = _cpu_receipt(decoder)
    payload = receipt.canonical_payload()

    # Exact architecture/shape facts for the fixed decoder at a 4x4 feature grid.
    assert receipt.parameter_count == 19_137
    assert receipt.conv2d_macs == 301_568
    assert receipt.conv2d_flops == 603_136
    assert payload["schema_version"] == (
        "cure-lite-stage-a-efficiency-receipt-v1"
    )
    assert payload["deployed_method"] == "U"
    assert payload["efficiency_is_scientific_gate_metric"] is False
    assert payload["static_evidence"]["decoder_variant"] == "uniform_legal"
    assert payload["static_evidence"]["input_contract"] == {
        "feature_shape": [1, 3, 4, 4],
        "occupancy_shape": [1, 1, 8, 8],
        "feature_dtype": "float32",
        "occupancy_dtype": "bool",
        "content_recipe": "all_zero_synthetic_shape_probe",
        "reads_image_or_gt_content": False,
    }
    execution = payload["execution_measurement"]
    assert execution["role"] == "environment_dependent_execution_measurement"
    assert execution["scientific_gate_metric"] is False
    assert execution["exact_cross_environment_numeric_replay_expected"] is False
    assert execution["environment"]["device_type"] == "cpu"
    assert execution["peak_allocated_bytes"] is None
    assert execution["peak_incremental_allocated_bytes"] is None
    assert execution["median_latency_ms"] >= 0.0
    assert execution["p95_latency_ms"] >= execution["median_latency_ms"]

    restored = StageAEfficiencyReceipt.from_mapping(payload)
    assert restored == receipt
    replay_static_efficiency(restored, decoder)
    assert decoder.training is True
    for name, tensor in decoder.state_dict().items():
        assert torch.equal(tensor, state_before[name])


def test_stage_a_efficiency_rejects_controls_bad_shapes_and_tampering() -> None:
    decoder = CURELiteDecoder(feature_channels=3)
    common = {
        "decoder": decoder,
        "binding": _binding(),
        "occupancy_shape": (1, 1, 8, 8),
        "device": "cpu",
        "warmup": 0,
        "repetitions": 1,
    }
    with pytest.raises(ValueError, match="only the deployed U"):
        measure_stage_a_efficiency(
            decoder_variant="factual_only",
            feature_shape=(1, 3, 4, 4),
            **common,
        )
    with pytest.raises(ValueError, match="channel contract"):
        measure_stage_a_efficiency(
            decoder_variant="uniform_legal",
            feature_shape=(1, 4, 4, 4),
            **common,
        )
    with pytest.raises(ValueError, match="warmup"):
        measure_stage_a_efficiency(
            decoder_variant="uniform_legal",
            feature_shape=(1, 3, 4, 4),
            **{**common, "warmup": False},
        )

    receipt = _cpu_receipt(decoder)
    tampered = copy.deepcopy(receipt.canonical_payload())
    tampered["static_evidence"]["conv2d_macs"] += 1
    with pytest.raises(ValueError, match="FLOPs|not canonical"):
        StageAEfficiencyReceipt.from_mapping(tampered)

    inconsistent = replace(
        receipt,
        conv2d_macs=receipt.conv2d_macs + 1,
        conv2d_flops=receipt.conv2d_flops + 2,
    )
    with pytest.raises(RuntimeError, match="does not reproduce"):
        replay_static_efficiency(inconsistent, decoder)
