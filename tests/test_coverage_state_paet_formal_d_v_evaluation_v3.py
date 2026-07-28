from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import torch

from cure_lite.cache.schema import file_sha256, stable_fingerprint
from cure_lite.coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
    CoverageStatePhaseAlignedEvidenceTransportConfig,
)
from cure_lite.experiment import (
    coverage_state_paet_formal_evaluation as original_evaluation,
)
from cure_lite.experiment.cache_pipeline import LoadedDVCacheBundle
from cure_lite.experiment.coverage_state_paet_formal_attempt import (
    LoadedCoverageStatePAETFormalAttempt,
)
from cure_lite.experiment.paired_formal_evaluation import (
    FrozenComparisonProtocol,
)
from cure_lite.types import FrozenBaseOutput
from cure_lite_eval_v3 import evaluation_source_closure as closure
from cure_lite_eval_v3 import evaluation_v3_amendment as amendment
from cure_lite_eval_v3 import fixed_sample_builder_v3 as builder
from cure_lite_eval_v3 import formal_d_v_runner_v3 as runner


def test_amendment_binds_v2_failure_and_full_parent_chain() -> None:
    result = amendment.verify_evaluation_v3_amendment()

    assert result["verified"] is True
    assert result["prior_D_V_accessed"] is True
    assert result["prior_D_T_accessed"] is False
    assert result["prior_model_forward_calls"] == 0
    assert result["failed_staging_preserved"] is True
    assert result["new_run_id"] == amendment.NEW_RUN_ID
    assert result["new_run_id"] != amendment.FAILED_RUN_ID
    assert result["metric_or_gate_changed"] is False
    assert result["data_or_model_changed"] is False


def test_builder_is_exactly_one_access_path_correction() -> None:
    result = builder.verify_fixed_sample_builder_correction()

    assert result["verified"] is True
    assert result["sole_change"] == {
        "from": "sources.artifact.model",
        "to": "sources.attempt.artifact.model",
    }
    assert result["byte_equivalent_after_inverse_substitution"] is True
    assert result["ast_equivalent_after_inverse_substitution"] is True


def _synthetic_real_seal_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    original_evaluation.PAETFormalArtifactBinding,
    CURELitePhaseAlignedEvidenceTransportLevelSet,
]:
    """Construct the actual private seal shape without opening D_V."""

    monkeypatch.setattr(
        original_evaluation.PAETFormalArtifactBinding,
        "verify_unchanged",
        lambda self: None,
    )
    monkeypatch.setattr(
        original_evaluation.PAETFormalArtifactBinding,
        "verify_cache_and_protocol",
        lambda self, bundle, comparison_protocol: None,
    )
    monkeypatch.setattr(
        original_evaluation.PAETFormalArtifactBinding,
        "verify_model",
        lambda self, model: None,
    )
    monkeypatch.setattr(
        LoadedDVCacheBundle,
        "verify_unchanged",
        lambda self: None,
    )
    monkeypatch.setattr(
        FrozenComparisonProtocol,
        "comparison_protocol_fingerprint",
        property(lambda self: "4" * 64),
    )

    torch.manual_seed(7)
    model = CURELitePhaseAlignedEvidenceTransportLevelSet(
        CoverageStatePhaseAlignedEvidenceTransportConfig(
            feature_channels=2,
            feature_stride=2,
            width=2,
        )
    )
    probability = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    probability[0, 0, 0, 0] = 0.9
    feature = torch.randn((1, 2, 2, 2), dtype=torch.float32)
    gt = torch.zeros((1, 4, 4), dtype=torch.bool)
    gt[0, 3, 3] = True
    row = SimpleNamespace(
        sample_id="synthetic-real-seal",
        base_output=FrozenBaseOutput(probability, feature),
        gt_mask=gt,
    )

    bundle = object.__new__(LoadedDVCacheBundle)
    object.__setattr__(bundle, "rows", (row,))
    object.__setattr__(bundle, "base_index_fingerprint", "1" * 64)
    object.__setattr__(bundle, "d_v_image_fingerprint", "2" * 64)
    object.__setattr__(bundle, "d_v_gt_fingerprint", "3" * 64)

    protocol = object.__new__(FrozenComparisonProtocol)
    attempt = object.__new__(LoadedCoverageStatePAETFormalAttempt)
    object.__setattr__(
        attempt,
        "artifact",
        SimpleNamespace(model=model),
    )
    seal = original_evaluation._PAETFormalArtifactBindingSeal(
        issuer=(
            original_evaluation._PAET_FORMAL_ARTIFACT_BINDING_ISSUER
        ),
        attempt=attempt,
        comparison_protocol=protocol,
        bundle=bundle,
    )
    digest = "a" * 64
    binding = original_evaluation.PAETFormalArtifactBinding(
        seed=42,
        epochs=800,
        steps_per_epoch=40,
        completed_updates=32_000,
        trained_from_scratch=True,
        resumed=False,
        runtime_splits=("D_R",),
        artifact_fingerprint=digest,
        artifact_receipt_sha256=digest,
        model_state_fingerprint=digest,
        model_config_fingerprint=digest,
        formal_training_protocol_fingerprint=digest,
        formal_schedule_fingerprint=digest,
        formal_training_result_fingerprint=digest,
        source_closure_fingerprint=digest,
        source_closure_manifest_sha256=digest,
        source_closure_archive_sha256=digest,
        source_closure_file_count=223,
        structural_source_receipt_fingerprint=digest,
        formal_attempt_complete_fingerprint=digest,
        manifest_fingerprint=digest,
        manifest_file_sha256=digest,
        preprocessing_fingerprint=digest,
        base_fingerprint=digest,
        base_state_fingerprint=digest,
        stage_a_config_sha256=(
            original_evaluation.PAET_FORMAL_STAGE_A_CONFIG_SHA256
        ),
        comparison_protocol_fingerprint=digest,
        _seal=seal,
    )
    return binding, model


def test_real_seal_path_old_builder_fails_and_v3_reaches_first_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding, model = _synthetic_real_seal_binding(monkeypatch)
    forwards: list[tuple[int, ...]] = []
    handle = model.register_forward_hook(
        lambda module, inputs, output: forwards.append(
            tuple(output.shape)
        )
    )
    try:
        with pytest.raises(
            AttributeError,
            match="has no attribute 'artifact'",
        ):
            original_evaluation.build_paet_fixed_d_v_samples(
                binding,
                batch_size=1,
            )
        assert forwards == []

        samples = builder.build_paet_fixed_d_v_samples(
            binding,
            batch_size=1,
        )
    finally:
        handle.remove()

    assert forwards == [(1, 1, 4, 4)]
    assert samples.ordered_sample_ids == ("synthetic-real-seal",)
    assert len(samples.base_samples) == len(samples.cure_samples) == 1


def test_runner_patch_is_exact_and_restores_after_exception() -> None:
    before = runner._actual_runner_globals()

    with pytest.raises(ArithmeticError, match="synthetic"):
        with runner._exact_evaluation_v3_runner_patch():
            during = runner._actual_runner_globals()
            assert during["PAET_FORMAL_DV_RUN_ID"] == (
                amendment.NEW_RUN_ID
            )
            assert during["PAET_FORMAL_DV_OUTPUT_PATH"] == (
                runner.NEW_OUTPUT_PATH
            )
            assert during["PAET_FORMAL_DV_STAGING_PATH"] == (
                runner.NEW_STAGING_PATH
            )
            assert (
                during["build_paet_fixed_d_v_samples"]
                is builder.build_paet_fixed_d_v_samples
            )
            raise ArithmeticError("synthetic")

    after = runner._actual_runner_globals()
    assert runner._runner_globals_are(before, after)
    assert runner._runner_globals_are(
        after,
        runner._expected_original_runner_globals(),
    )


def test_validate_create_only_does_not_access_d_v_or_d_t(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_closure = {
        "manifest_repo_path": "artifacts/v3.json",
        "manifest_sha256": "1" * 64,
        "archive_repo_path": "artifacts/v3.tar",
        "archive_sha256": "2" * 64,
        "content_fingerprint": "3" * 64,
        "file_count": 8,
    }
    amendment_result = {
        "failure_receipt_sha256": "4" * 64,
        "failure_fingerprint": "5" * 64,
        "amendment_repo_path": "protocols/amendment.json",
        "amendment_sha256": "6" * 64,
        "amendment_fingerprint": "7" * 64,
    }
    loaded = SimpleNamespace(
        complete_fingerprint="8" * 64,
        artifact=SimpleNamespace(
            artifact_fingerprint="9" * 64,
            module_state_fingerprint="a" * 64,
        ),
        source_closure_manifest_sha256="b" * 64,
        source_closure_archive_sha256="c" * 64,
        source_closure_content_fingerprint="d" * 64,
    )
    observed: list[dict[str, object]] = []

    def fixed_create_only() -> dict[str, object]:
        observed.append(runner._actual_runner_globals())
        return {
            "run_id": amendment.NEW_RUN_ID,
            "output_repo_path": amendment.NEW_OUTPUT_REPO_PATH,
            "staging_repo_path": amendment.NEW_STAGING_REPO_PATH,
            "create_only": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "output_created": False,
        }

    def forbidden() -> None:
        raise AssertionError("create-only opened fixed D_V inputs")

    monkeypatch.setattr(
        runner,
        "_validate_chain",
        lambda: {
            "amendment": amendment_result,
            "evaluation_v3_source_closure": source_closure,
        },
    )
    monkeypatch.setattr(
        runner._v2,
        "_load_original_attempt_with_aliases",
        lambda: loaded,
    )
    monkeypatch.setattr(
        runner._d_v,
        "validate_paet_formal_d_v_create_only",
        fixed_create_only,
    )
    monkeypatch.setattr(
        runner._d_v,
        "_load_fixed_d_v_inputs",
        forbidden,
    )

    result = runner.validate_paet_formal_d_v_create_only_v3()

    assert len(observed) == 1
    assert observed[0]["PAET_FORMAL_DV_RUN_ID"] == amendment.NEW_RUN_ID
    assert result["D_V_accessed"] is False
    assert result["D_T_accessed"] is False
    assert result["output_created"] is False
    assert result["prior_failed_attempt"]["D_V_accessed"] is True
    assert result["metric_or_gate_changed"] is False
    body = dict(result)
    fingerprint = body.pop("validation_fingerprint")
    assert fingerprint == stable_fingerprint(body)


def _fake_parent_bindings() -> dict[str, object]:
    return {
        "original_formal_source_closure": {
            "manifest_sha256": "1" * 64,
            "archive_sha256": "2" * 64,
            "content_fingerprint": "3" * 64,
            "file_count": 223,
        },
        "formal800_schema_erratum": {
            "repo_path": "protocols/v2.json",
            "sha256": "4" * 64,
            "erratum_fingerprint": "5" * 64,
        },
        "evaluation_v2_source_closure": {
            "manifest_sha256": "6" * 64,
            "archive_sha256": "7" * 64,
            "content_fingerprint": "8" * 64,
            "file_count": 7,
        },
        "evaluation_v2_failure": {
            "repo_path": "runs/failure.json",
            "sha256": "9" * 64,
            "failure_fingerprint": "a" * 64,
            "D_V_accessed": True,
            "D_T_accessed": False,
            "model_forward_calls": 0,
        },
        "evaluation_v3_amendment": {
            "repo_path": "protocols/v3.json",
            "sha256": "b" * 64,
            "amendment_fingerprint": "c" * 64,
        },
    }


def test_v3_source_closure_is_independent_and_create_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    for repo_path in closure._SOURCE_PATHS:
        source = repository / repo_path
        target = tmp_path / repo_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    monkeypatch.setattr(
        closure,
        "_parent_bindings",
        lambda repository_root=None: _fake_parent_bindings(),
    )

    first = closure.build_evaluation_v3_source_closure(tmp_path)
    second = closure.verify_evaluation_v3_source_closure(tmp_path)

    assert first == second
    assert first["sealed"] is True
    assert first["file_count"] == 8
    paths = closure.evaluation_v3_source_closure_paths(tmp_path)
    assert not any(path.startswith("cure_lite/") for path in paths)
    assert not any(
        path.startswith("cure_lite_eval_v2/") for path in paths
    )
    with pytest.raises(FileExistsError):
        closure.build_evaluation_v3_source_closure(tmp_path)

    changed = tmp_path / "cure_lite_eval_v3/__init__.py"
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(
        closure.EvaluationV3SourceClosureError,
        match="live evaluation-v3 source changed",
    ):
        closure.verify_evaluation_v3_source_closure(tmp_path)


def test_external_binding_carries_all_evidence_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "runs/new_d_v"
    output.mkdir(parents=True)
    payloads = {
        "COMPLETE.json": {"complete_fingerprint": "1" * 64},
        "receipt.json": {
            "receipt_fingerprint": "2" * 64,
            "evaluation_result_fingerprint": "3" * 64,
        },
        "decision.json": {"decision_fingerprint": "4" * 64},
    }
    for name, payload in payloads.items():
        (output / name).write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
    monkeypatch.setattr(runner, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(runner, "NEW_OUTPUT_PATH", output)
    monkeypatch.setattr(
        runner._d_v,
        "_validate_published_output",
        lambda path: {
            "gate_passed": True,
            "authorizes_D_T": True,
        },
    )
    amendment_result = {
        "formal_complete_sha256": "5" * 64,
        "formal_complete_fingerprint": "6" * 64,
        "original_source_closure_manifest_sha256": "7" * 64,
        "original_source_closure_archive_sha256": "8" * 64,
        "original_source_closure_content_fingerprint": "9" * 64,
        "original_source_closure_file_count": 223,
        "schema_erratum_repo_path": "protocols/v2.json",
        "schema_erratum_sha256": "a" * 64,
        "schema_erratum_fingerprint": "b" * 64,
        "authorized_alias_count": 2,
        "evaluation_v2_closure_manifest_sha256": "c" * 64,
        "evaluation_v2_closure_archive_sha256": "d" * 64,
        "evaluation_v2_closure_content_fingerprint": "e" * 64,
        "evaluation_v2_closure_file_count": 7,
        "failure_receipt_repo_path": "runs/failure.json",
        "failure_receipt_sha256": "f" * 64,
        "failure_fingerprint": "0" * 64,
        "amendment_repo_path": "protocols/v3.json",
        "amendment_sha256": "1" * 64,
        "amendment_fingerprint": "2" * 64,
    }
    source_closure = {
        "manifest_repo_path": "artifacts/v3.json",
        "manifest_sha256": "3" * 64,
        "archive_repo_path": "artifacts/v3.tar",
        "archive_sha256": "4" * 64,
        "content_fingerprint": "5" * 64,
        "file_count": 8,
    }
    with runner._exact_evaluation_v3_runner_patch():
        binding = runner._build_external_evidence_binding(
            {
                "amendment": amendment_result,
                "evaluation_v3_source_closure": source_closure,
            }
        )

    assert set(binding["D_V"]["files"]) == set(
        runner._D_V_MEMBER_NAMES
    )
    for name, row in binding["D_V"]["files"].items():
        assert row["sha256"] == file_sha256(output / name)
    assert binding["Formal800"]["source_closure_file_count"] == 223
    assert binding["evaluation_v2_source_closure"]["file_count"] == 7
    assert binding["failed_evaluation_v2_attempt"]["D_V_accessed"] is True
    assert (
        binding["failed_evaluation_v2_attempt"][
            "failed_attempt_resumed_or_reused"
        ]
        is False
    )
    assert binding["evaluation_v3_source_closure"]["file_count"] == 8
    assert binding["correction_scope"]["metric_or_gate_modified"] is False
    body = dict(binding)
    fingerprint = body.pop("binding_fingerprint")
    assert fingerprint == stable_fingerprint(body)
