from __future__ import annotations

import json
from pathlib import Path

import pytest

from cure_lite.cache.schema import stable_fingerprint
from tools import benchmark_conservative_factorized_sync as benchmark


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_RESULT = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "coverage_conserving_subpixel_evidence_allocation_v8"
    / "sync_benchmark_result.json"
)
CANONICAL_SHA256 = (
    "52caa08511aebf26e5e7e746cd1d59017e14d5e8ea86a841f36623901a36f152"
)
CANONICAL_FINGERPRINT = (
    "d2304d3428daadebc40fc9047c9aca20b4c9492d95ccc6c554ae5d541471a0c0"
)


def test_canonical_gpu0_sync_evidence_is_bound() -> None:
    import hashlib

    payload = json.loads(CANONICAL_RESULT.read_text(encoding="utf-8"))
    assert hashlib.sha256(CANONICAL_RESULT.read_bytes()).hexdigest() == (
        CANONICAL_SHA256
    )
    unsigned = dict(payload)
    assert unsigned.pop("result_fingerprint") == CANONICAL_FINGERPRINT
    assert stable_fingerprint(unsigned) == CANONICAL_FINGERPRINT
    assert payload["environment"]["device"] == "cuda:0"
    assert payload["environment"]["device_name"] == (
        "NVIDIA GeForce RTX 3090"
    )
    assert payload["local_scalar_dense_calls_per_decoder_forward"] == {
        "production": 9,
        "unchecked_diagnostic": 0,
    }
    assert payload["full_decoder_equivalence"][
        "decoder_output_bit_exact_to_production"
    ]["unchecked_diagnostic"] is True
    assert payload["full_decoder_equivalence"][
        "parameter_gradient_bit_exact_to_production"
    ]["unchecked_diagnostic"] is True
    assert payload["projections"][
        "production_minus_unchecked_seconds_bounded_1200_calls"
    ] < 2.0
    assert payload["scope"]["D_R_accessed"] is False


def test_sync_audit_is_equivalent_and_synthetic_only() -> None:
    result = benchmark.run_benchmark(
        device="cpu",
        warmup=0,
        iterations=1,
        repeats=1,
    )
    scope = result["scope"]
    assert isinstance(scope, dict)
    assert scope["synthetic_tensors_only"] is True
    assert scope["dataset_or_cache_payload_loaded"] is False
    assert scope["D_R_accessed"] is False
    assert scope["D_V_accessed"] is False
    assert scope["D_T_accessed"] is False
    assert scope["detection_performance_evaluated"] is False
    assert scope["production_decoder_modified"] is False

    boundary = result["numerical_boundary_audit"]
    assert isinstance(boundary, dict)
    assert boundary["safe_forward_bit_exact"] is True
    assert boundary["all_required_invalid_rejected"] is True
    assert all(boundary["production_invalid_rejected"].values())

    equivalence = result["full_decoder_equivalence"]
    assert isinstance(equivalence, dict)
    assert equivalence["parameter_count"] == 4385
    assert equivalence["parameter_tensor_count"] == 6
    assert all(
        equivalence[
            "decoder_output_bit_exact_to_production"
        ].values()
    )
    assert all(
        equivalence[
            "parameter_gradient_bit_exact_to_production"
        ].values()
    )

    counts = result["local_scalar_dense_calls_per_decoder_forward"]
    assert counts == {
        "production": 9,
        "unchecked_diagnostic": 0,
    }
    unsigned = dict(result)
    observed = unsigned.pop("result_fingerprint")
    assert stable_fingerprint(unsigned) == observed


def test_sync_cli_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "sync.json"
    assert benchmark.main(
        [
            "--device",
            "cpu",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--repeats",
            "1",
            "--output",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == (
        "cure-lite-cc-sea-v8-sync-benchmark-v1"
    )
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        benchmark.main(
            [
                "--device",
                "cpu",
                "--warmup",
                "0",
                "--iterations",
                "1",
                "--repeats",
                "1",
                "--output",
                str(output),
            ]
        )
    assert output.read_bytes() == before


def test_sync_benchmark_rejects_nonzero_gpu_index() -> None:
    with pytest.raises(ValueError, match="cpu or cuda:0"):
        benchmark.run_benchmark(
            device="cuda:1",
            warmup=0,
            iterations=1,
            repeats=1,
        )
