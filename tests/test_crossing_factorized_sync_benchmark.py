from __future__ import annotations

import json
from math import isfinite
from pathlib import Path

import pytest
import torch

import cure_lite.crossing_factorized_decoder as crossing_module
from cure_lite.cache.schema import stable_fingerprint
from tools import benchmark_crossing_factorized_sync as benchmark


def test_operator_variants_are_exact_on_supported_domain_and_keep_boundaries() -> None:
    result = benchmark.audit_operator_boundaries()

    assert result["device"] == "cpu"
    assert result["dtype"] == "float32"
    assert result["safe_values"] == [-80.0, -1.0, 0.0, 1.0e-7, 1.0, 88.0]
    assert all(result["forward_bit_exact_to_current"].values())
    assert all(result["gradient_bit_exact_to_current"].values())
    assert all(result["gradient_equals_exp_exact"].values())
    assert result["current_contract_preserved"] is True
    assert result["async_contract_preserved_on_cpu"] is True
    assert result["unchecked_is_diagnostic_only"] is True
    assert result["invalid_async_cuda_probe_executed"] is False

    rejected = result["invalid_rejected"]
    assert all(rejected["current"].values())
    assert all(rejected["async"].values())
    assert not any(rejected["unchecked"].values())
    assert set(result["invalid_values"]) == {
        "zero_recovery",
        "nonfinite_positive",
        "nan",
        "positive_inf",
    }


def test_fixed_cpu_benchmark_has_auditable_schema_without_latency_gate() -> None:
    production = crossing_module.crossing_recoverable_evidence
    previous = {
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }
    result = benchmark.run_benchmark(
        device="cpu",
        warmup=0,
        iterations=1,
        repeats=1,
    )

    assert crossing_module.crossing_recoverable_evidence is production
    assert torch.are_deterministic_algorithms_enabled() is (
        previous["deterministic_algorithms"]
    )
    assert torch.backends.cudnn.benchmark is previous["cudnn_benchmark"]
    assert torch.backends.cudnn.deterministic is previous[
        "cudnn_deterministic"
    ]
    assert torch.backends.cuda.matmul.allow_tf32 is previous[
        "matmul_allow_tf32"
    ]
    assert torch.backends.cudnn.allow_tf32 is previous[
        "cudnn_allow_tf32"
    ]
    assert result["schema_version"] == benchmark.SCHEMA_VERSION
    unsigned = dict(result)
    fingerprint = unsigned.pop("result_fingerprint")
    assert fingerprint == stable_fingerprint(unsigned)
    assert result["scope"] == {
        "synthetic_tensors_only": True,
        "dataset_or_cache_loaded": False,
        "D_R_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "detection_performance_evaluated": False,
        "latency_is_not_a_scientific_gate": True,
        "production_decoder_modified": False,
    }
    contract = result["contract"]
    assert contract["batch_size"] == 4
    assert contract["feature_channels"] == 64
    assert contract["feature_grid"] == [64, 64]
    assert contract["output_grid"] == [256, 256]
    assert contract["feature_stride"] == 4
    assert contract["decoder_calls_per_update"] == 3
    assert contract["formal_updates_for_projection"] == 32_000
    assert contract["formal_decoder_calls_for_projection"] == 96_000
    assert contract["variant_order"] == ["current", "unchecked", "async"]
    assert contract["hard_latency_threshold"] is None
    environment = result["environment"]
    assert environment["deterministic_algorithms"] is True
    assert environment["cudnn_benchmark"] is False
    assert environment["cudnn_deterministic"] is True
    assert environment["matmul_allow_tf32"] is False
    assert environment["cudnn_allow_tf32"] is False

    equivalence = result["full_decoder_equivalence"]
    assert all(equivalence["decoder_forward_bit_exact_to_current"].values())
    assert all(
        equivalence[
            "decoder_parameter_gradient_bit_exact_to_current"
        ].values()
    )
    assert equivalence["parameter_tensor_count"] == 6
    assert equivalence["parameter_count"] == 4_385

    assert set(result["variants"]) == {"current", "unchecked", "async"}
    for variant in result["variants"].values():
        assert set(variant) == {"forward", "forward_backward"}
        for measurement in variant.values():
            assert len(measurement["samples_ms_per_update"]) == 1
            for name in (
                "median_ms_per_update",
                "minimum_ms_per_update",
                "maximum_ms_per_update",
                "median_ms_per_decoder_call",
            ):
                assert isfinite(measurement[name])
                assert measurement[name] > 0.0

    projection = result["projections"]
    assert projection["interpretation"] == (
        "environment_measurement_only_no_authorization_or_gate"
    )
    assert json.loads(
        json.dumps(result, allow_nan=False)
    ) == result


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"device": "cuda:1"}, ValueError),
        ({"warmup": -1}, ValueError),
        ({"iterations": 0}, ValueError),
        ({"repeats": 0}, ValueError),
        ({"iterations": True}, TypeError),
    ],
)
def test_benchmark_rejects_scope_or_count_changes(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        benchmark.run_benchmark(**kwargs)


def test_cli_outputs_json_without_writing_or_hard_timing_gate(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = {
        "schema_version": benchmark.SCHEMA_VERSION,
        "scope": {"synthetic_tensors_only": True},
        "contract": {"hard_latency_threshold": None},
    }
    monkeypatch.setattr(
        benchmark,
        "run_benchmark",
        lambda **kwargs: expected,
    )

    assert benchmark.main(["--device", "cpu"]) == 0
    observed = json.loads(capsys.readouterr().out)
    assert observed == expected


def test_cli_create_only_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "sync.json"
    expected = {
        "schema_version": benchmark.SCHEMA_VERSION,
        "result_fingerprint": "0" * 64,
    }
    monkeypatch.setattr(
        benchmark,
        "run_benchmark",
        lambda **kwargs: expected,
    )

    assert benchmark.main(["--device", "cpu", "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == expected
    with pytest.raises(FileExistsError):
        benchmark.main(["--device", "cpu", "--output", str(output)])
