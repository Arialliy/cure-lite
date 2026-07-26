from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from cure_lite.cache.schema import file_sha256, stable_fingerprint
import cure_lite.experiment.conservative_factorized_outcome_bounded
from tools import run_conservative_factorized_outcome_bounded as runner


ROOT = Path(__file__).resolve().parents[1]
CLOSURE = (
    ROOT
    / "protocols"
    / "IRSTD-1K"
    / "coverage_conserving_subpixel_evidence_allocation_v8"
    / "bounded_implementation_closure_receipt.json"
)
THIS_TEST = (
    "tests_v8/"
    "test_conservative_factorized_bounded_implementation_closure.py"
)


def test_signed_v8_bounded_implementation_closure(
    monkeypatch,
) -> None:
    loader_calls = 0

    def forbidden_real_loader(*args, **kwargs):
        nonlocal loader_calls
        del args, kwargs
        loader_calls += 1
        raise AssertionError("closure self-test must not load D_R")

    monkeypatch.setattr(
        runner,
        "_load_frozen_real_inputs",
        forbidden_real_loader,
    )

    payload = json.loads(CLOSURE.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    observed = payload["receipt_fingerprint"]
    unsigned = dict(payload)
    del unsigned["receipt_fingerprint"]
    assert stable_fingerprint(unsigned) == observed

    config = runner._load_config(ROOT / runner.CONFIG_REPO_PATH)
    proposal, _ = runner._load_implementation_proposal(config)
    assert proposal["proposal_fingerprint"] == (
        runner.IMPLEMENTATION_PROPOSAL_FINGERPRINT
    )
    dry = runner._load_frozen_dry_evidence()
    implementation = runner._implementation_binding()
    runner._verify_implementation_files(implementation)
    loaded, path, runtime_signed = runner._load_implementation_closure(
        config,
        implementation,
        dry,
    )
    assert loaded == payload
    assert path == CLOSURE.resolve()
    assert runtime_signed == runner._fingerprinted(implementation)
    assert loader_calls == 0

    assert payload["boundary"] == {
        "D_R_payload_accessed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
        "real_D_R_bounded_execution_authorized": False,
        "formal_800_authorized": False,
        "calibration_performed": False,
        "detection_performance_evaluated": False,
        "full_CURE_authorized": False,
        "other_detector_integration_authorized": False,
    }
    assert payload["authorization_eligibility"] == {
        "single_real_D_R_run_eligible": True,
        "directly_authorizes_real_D_R_run": False,
        "formal_800_authorized": False,
    }
    sync = payload["sync_benchmark_binding"]
    assert sync["result_file_sha256"] == (
        runner.SYNC_BENCHMARK_FILE_SHA256
    )
    assert sync["result_fingerprint"] == (
        runner.SYNC_BENCHMARK_FINGERPRINT
    )
    assert sync["production_local_scalar_calls_per_forward"] == 9
    assert sync["unchecked_local_scalar_calls_per_forward"] == 0

    test_bindings = payload["test_bindings"]
    assert test_bindings["files"][THIS_TEST] == file_sha256(
        ROOT / THIS_TEST
    )
    assert test_bindings["closure_self_test_excluded_pre_signing"] is True
    assert test_bindings["closure_self_test_required_post_signing"] is True

    fresh_audit = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
from pathlib import Path
import sys
import cure_lite.experiment.conservative_factorized_outcome_bounded
from tools import run_conservative_factorized_outcome_bounded as runner

root = Path.cwd().resolve()
implementation = runner._implementation_binding()
runner._verify_implementation_files(implementation)
bound = set(implementation["all_runtime_files"])
loaded = set()
for module in tuple(sys.modules.values()):
    path_text = getattr(module, "__file__", None)
    if not path_text:
        continue
    try:
        relative = (
            Path(path_text).resolve().relative_to(root).as_posix()
        )
    except (OSError, ValueError):
        continue
    if (
        relative.endswith(".py")
        and (
            relative.startswith("cure_lite/")
            or relative.startswith("tools/")
        )
    ):
        loaded.add(relative)
print(
    json.dumps(
        {
            "bound_count": len(bound),
            "loaded_count": len(loaded),
            "unbound_loaded": sorted(loaded - bound),
            "D_R_payload_accessed": False,
        },
        sort_keys=True,
    )
)
""",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    fresh = json.loads(fresh_audit.stdout)
    assert fresh["bound_count"] == 83
    assert fresh["loaded_count"] == 80
    assert fresh["unbound_loaded"] == []
    assert fresh["D_R_payload_accessed"] is False
    assert loader_calls == 0
