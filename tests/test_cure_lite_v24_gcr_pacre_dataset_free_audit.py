from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from cure_lite.cache.schema import stable_fingerprint
from cure_lite_v24.dataset_free import (
    GCR_PACRE_DATASET_FREE_CHECK_NAMES,
    GCR_PACRE_DATASET_FREE_SCHEMA,
    run_gcr_pacre_dataset_free_audit,
    verify_gcr_pacre_dataset_free_receipt,
)
from cure_lite_v24.factory import (
    GCR_PACRE_FORMAL_FEATURE_CHANNELS,
    GCR_PACRE_FORMAL_FEATURE_STRIDE,
    GCR_PACRE_FORMAL_PARAMETER_COUNT,
    GCR_PACRE_FORMAL_WIDTH,
)


_ROOT = Path(__file__).resolve().parents[1]
_CLI_PATH = (
    _ROOT
    / "tools"
    / "audit_cure_lite_v24_gcr_pacre_dataset_free.py"
)


@pytest.fixture(scope="module")
def receipt() -> dict[str, object]:
    # This is deliberately the real public audit: the fixture performs the
    # generated PMOPE/Adam update and formal efficiency measurements.
    return run_gcr_pacre_dataset_free_audit(device="cpu")


def _load_cli() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "audit_cure_lite_v24_gcr_pacre_dataset_free",
        _CLI_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load dataset-free CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reseal_section(section: dict[str, object]) -> None:
    section.pop("section_fingerprint")
    section["section_fingerprint"] = stable_fingerprint(section)


def _reseal_receipt(value: dict[str, object]) -> None:
    value.pop("receipt_fingerprint")
    value["receipt_fingerprint"] = stable_fingerprint(value)


def test_real_generated_audit_passes_all_frozen_checks(
    receipt: dict[str, object],
) -> None:
    assert receipt["schema_version"] == GCR_PACRE_DATASET_FREE_SCHEMA
    assert tuple(receipt["checks"]) == GCR_PACRE_DATASET_FREE_CHECK_NAMES
    assert len(receipt["checks"]) == 30
    assert all(receipt["checks"].values())
    assert receipt["decision"]["gate_passed"] is True
    assert receipt["decision"]["failed_checks"] == []
    assert (
        verify_gcr_pacre_dataset_free_receipt(receipt)
        == receipt["receipt_fingerprint"]
    )


def test_receipt_proves_one_real_pmope_update_and_both_paths(
    receipt: dict[str, object],
) -> None:
    gradients = receipt["evidence"]["gradients"]
    warmup = gradients["warmup"]
    post = gradients["post_warmup"]

    assert warmup["objective"] == "pmope_joint"
    assert warmup["model_forward_calls"] == 1
    assert warmup["backward_calls"] == 1
    assert warmup["optimizer_steps"] == 1
    assert warmup["logical_states"] == 12
    assert warmup["parameter_state_changed"] is True
    assert warmup["all_logged_losses_finite"] is True
    assert (
        warmup["pmope_parameter_gradients"]["scalar_energy_weight"][
            "nonzero_count"
        ]
        > 0
    )
    for path in (
        "residual_path_parameter_gradients",
        "gate_path_parameter_gradients",
    ):
        assert set(post[path]) == {
            "joint_state_weight",
            "joint_hidden_bias",
            "scalar_energy_weight",
        }
        assert all(row["finite"] is True for row in post[path].values())
        assert all(row["nonzero_count"] > 0 for row in post[path].values())


def test_receipt_records_reference_selectivity_and_gate_endpoints(
    receipt: dict[str, object],
) -> None:
    algebra = receipt["evidence"]["algebra"]
    selectivity = receipt["evidence"]["selectivity"]

    assert algebra["fp64_envelope"]["required_components"] == ["field"]
    assert (
        algebra["fp64_envelope"]["components"]["field"]["passed"] is True
    )
    assert algebra["reference_parity"] == {
        "residual_odd_antisymmetric": True,
        "common_even_symmetric": True,
        "gate_symmetric": True,
        "gated_interaction_antisymmetric": True,
    }
    assert (
        algebra["endpoint_witnesses"]["upper_statistics"]["two_count"] > 0
    )
    assert (
        algebra["endpoint_witnesses"]["lower_statistics"]["zero_count"] > 0
    )
    assert selectivity["target_like"]["gcr_minus_pacre"] < 0.0
    assert selectivity["background_like"]["gcr_minus_pacre"] > 0.0
    assert selectivity["common_only"]["completion_count"] == 0


def test_efficiency_receipt_is_formal_same_condition_and_threshold_free(
    receipt: dict[str, object],
) -> None:
    efficiency = receipt["evidence"]["efficiency"]
    conditions = efficiency["common_conditions"]
    expected_config = {
        "feature_channels": GCR_PACRE_FORMAL_FEATURE_CHANNELS,
        "feature_stride": GCR_PACRE_FORMAL_FEATURE_STRIDE,
        "width": GCR_PACRE_FORMAL_WIDTH,
        "parameter_count": GCR_PACRE_FORMAL_PARAMETER_COUNT,
    }

    assert conditions["formal_model_config"] == expected_config
    assert conditions["input_shape"] == [12, 64, 1, 1]
    assert conditions["occupancy_shape"] == [12, 1, 4, 4]
    assert conditions["threshold_or_ratio_gate"] is None
    assert (
        efficiency["interpretation"]
        == "measurement_only_no_post_hoc_lite_overhead_threshold"
    )
    fingerprints = set()
    for arm in efficiency["arms"].values():
        assert arm["model_config"] == {
            key: expected_config[key]
            for key in ("feature_channels", "feature_stride", "width")
        }
        assert arm["parameter_count"] == GCR_PACRE_FORMAL_PARAMETER_COUNT
        assert arm["parameter_bytes"] == GCR_PACRE_FORMAL_PARAMETER_COUNT * 4
        assert arm["checkpoint_bytes"] > 0
        assert arm["forward_flops"] > 0
        assert arm["forward_latency"]["median_ns"] > 0.0
        assert arm["forward_latency"]["p95_ns"] > 0.0
        assert arm["train_step_latency"]["median_ns"] > 0.0
        assert arm["train_step_latency"]["p95_ns"] > 0.0
        assert arm["output_shape"] == [12, 1, 4, 4]
        assert arm["field_tensor_bytes"] == 768
        assert arm["memory"] == {
            "supported": False,
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
        }
        assert arm["oom"] is False
        assert arm["nonfinite"] is False
        fingerprints.add(arm["initial_parameter_fingerprint"])
    assert len(fingerprints) == 1


def test_verifier_rejects_unsealed_nested_tampering(
    receipt: dict[str, object],
) -> None:
    tampered = deepcopy(receipt)
    tampered["evidence"]["selectivity"]["target_like"][
        "gcr_minus_pacre"
    ] = 1.0
    with pytest.raises(ValueError, match="receipt_fingerprint"):
        verify_gcr_pacre_dataset_free_receipt(tampered)


def test_verifier_rederives_checks_after_fully_resealed_tampering(
    receipt: dict[str, object],
) -> None:
    tampered = deepcopy(receipt)
    selectivity = tampered["evidence"]["selectivity"]
    selectivity["target_like"]["gcr_minus_pacre"] = 1.0
    _reseal_section(selectivity)
    _reseal_receipt(tampered)

    with pytest.raises(ValueError, match="checks differ from evidence"):
        verify_gcr_pacre_dataset_free_receipt(tampered)


def test_cli_generated_mode_atomically_publishes_a_real_receipt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _load_cli()
    output = tmp_path / "dataset-free.json"

    assert cli.main(["--device", "cpu", "--output", str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert summary["mode"] == "generated-only"
    assert summary["gate_passed"] is True
    assert summary["output"] == str(output)
    assert (
        verify_gcr_pacre_dataset_free_receipt(written)
        == summary["receipt_fingerprint"]
    )

    with pytest.raises(FileExistsError):
        cli._atomic_write_new_json(output, written)


def test_cli_verify_only_never_reruns_the_audit(
    receipt: dict[str, object],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _load_cli()
    output = tmp_path / "sealed.json"
    cli._atomic_write_new_json(output, receipt)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("verify-only reran generated computation")

    monkeypatch.setattr(
        cli,
        "run_gcr_pacre_dataset_free_audit",
        forbidden,
    )
    assert cli.main(["--verify-only", str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "efficiency_device": "cpu",
        "gate_passed": True,
        "mode": "verify-only",
        "receipt_fingerprint": receipt["receipt_fingerprint"],
    }
