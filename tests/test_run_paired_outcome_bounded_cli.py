from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.experiment.paired_outcome_bounded import (
    PAIRED_OUTCOME_BOUNDED_SCHEMA,
)
from tools import run_paired_outcome_bounded as runner


PAIR_CATALOG_FINGERPRINT = (
    "4886e52d2cfb3392d0f4fdda376159d6e7f694fd449dc809cf8874793febde76"
)


def _internally_fingerprinted(
    payload: dict[str, object],
    *,
    field: str,
) -> dict[str, object]:
    result = dict(payload)
    result[field] = stable_fingerprint(payload)
    return result


class _UnchangedBundle:
    def __init__(self) -> None:
        self.verify_calls = 0

    def verify_unchanged(self) -> None:
        self.verify_calls += 1


def _fake_catalog() -> SimpleNamespace:
    return SimpleNamespace(
        catalog_fingerprint=PAIR_CATALOG_FINGERPRINT,
        split="D_R",
        clean_positive=tuple(range(206)),
        component_null=tuple(range(16)),
    )


def _fake_population() -> SimpleNamespace:
    receipt = _internally_fingerprinted(
        {
            "schema_version": "test-anchor-population",
            "pair_catalog_fingerprint": PAIR_CATALOG_FINGERPRINT,
            "factual_miss": [],
            "factual_no_miss": [],
            "identity_null": [],
        },
        field="population_fingerprint",
    )
    return SimpleNamespace(
        pair_catalog_fingerprint=PAIR_CATALOG_FINGERPRINT,
        population_fingerprint=receipt["population_fingerprint"],
        canonical_receipt=lambda: dict(receipt),
    )


def _fake_factual_schedule(population: SimpleNamespace) -> SimpleNamespace:
    receipt = _internally_fingerprinted(
        {
            "schema_version": "test-factual-schedule",
            "population_fingerprint": population.population_fingerprint,
            "optimizer_updates": 400,
        },
        field="schedule_fingerprint",
    )
    return SimpleNamespace(
        schedule_fingerprint=receipt["schedule_fingerprint"],
        canonical_receipt=lambda: dict(receipt),
    )


def _fake_materializer() -> SimpleNamespace:
    receipt = _internally_fingerprinted(
        {
            "schema_version": "test-outcome-inputs",
            "pair_catalog_fingerprint": PAIR_CATALOG_FINGERPRINT,
            "outcome_pairs": 222,
        },
        field="materializer_fingerprint",
    )
    return SimpleNamespace(
        pair_catalog_fingerprint=PAIR_CATALOG_FINGERPRINT,
        materializer_fingerprint=receipt["materializer_fingerprint"],
        canonical_receipt=lambda: dict(receipt),
    )


def _fake_outcome_schedule() -> SimpleNamespace:
    receipt = _internally_fingerprinted(
        {
            "schema_version": "test-outcome-schedule",
            "catalog_fingerprint": PAIR_CATALOG_FINGERPRINT,
            "optimizer_updates": 400,
        },
        field="schedule_fingerprint",
    )
    return SimpleNamespace(
        catalog_fingerprint=PAIR_CATALOG_FINGERPRINT,
        schedule_fingerprint=receipt["schedule_fingerprint"],
        canonical_receipt=lambda: dict(receipt),
    )


def _fake_core_result() -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": PAIRED_OUTCOME_BOUNDED_SCHEMA,
        "execution_status": "completed",
        "decision": "BOUNDED_MODEL_CODE_GATE_PASS",
        "structural_execution_pass": True,
        "computational_model_code_gate_pass": True,
        "interpretation": {
            "not_detection_performance_evidence": True,
            "D_V_accessed": False,
            "D_T_accessed": False,
            "calibration_performed": False,
            "inference_performed": False,
            "base_or_backbone_updated": False,
        },
    }
    result["result_fingerprint"] = stable_fingerprint(result)
    return result


def test_frozen_config_and_proposal_bindings_validate() -> None:
    config_path = (Path.cwd() / runner.CONFIG_REPO_PATH).resolve()
    config = runner._load_config(config_path)
    proposal, proposal_path, design_path = runner._load_proposal(config)

    assert config["config_fingerprint"] == runner.CONFIG_FINGERPRINT
    assert proposal["proposal_fingerprint"] == runner.PROPOSAL_FINGERPRINT
    assert proposal_path == (Path.cwd() / runner.PROPOSAL_REPO_PATH).resolve()
    assert design_path.is_file()


def test_create_only_output_rejects_an_existing_path(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        runner._prepare_output(existing)


def test_mocked_success_run_publishes_and_round_trips_without_dv_or_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    catalog = _fake_catalog()
    prepared = object()
    bundle = _UnchangedBundle()
    population = _fake_population()
    factual_schedule = _fake_factual_schedule(population)
    materializer = _fake_materializer()
    outcome_schedule = _fake_outcome_schedule()

    monkeypatch.setattr(
        runner.legacy_runner,
        "_load_real_catalog",
        lambda config: (catalog, prepared, bundle, {}),
    )
    monkeypatch.setattr(
        runner,
        "build_outcome_bounded_anchor_population",
        lambda pair_catalog, prepared_catalog, specification: population,
    )
    monkeypatch.setattr(
        runner,
        "build_outcome_factual_anchor_schedule",
        lambda selected, **kwargs: factual_schedule,
    )
    monkeypatch.setattr(
        runner,
        "build_paired_outcome_input_materializer",
        lambda pair_catalog, prepared_catalog: materializer,
    )
    monkeypatch.setattr(
        runner,
        "build_outcome_pair_schedule",
        lambda pair_catalog, **kwargs: outcome_schedule,
    )
    monkeypatch.setattr(
        runner,
        "execute_paired_outcome_bounded",
        lambda *args, **kwargs: _fake_core_result(),
    )

    output = tmp_path / "published"
    result = runner.run(
        argparse.Namespace(
            config=(Path.cwd() / runner.CONFIG_REPO_PATH).resolve(),
            device="cpu",
            output=output,
        )
    )
    published = runner.load_paired_outcome_bounded_artifact(output)

    assert bundle.verify_calls == 1
    assert result["decision"] == "BOUNDED_MODEL_CODE_GATE_PASS"
    assert result["bounded_model_code_gate_pass"] is True
    assert result["directly_authorizes_formal_800"] is False
    assert result["D_V_accessed"] is False
    assert result["D_T_accessed"] is False
    assert published.bounded_model_code_gate_pass is True
    assert published.pair_catalog_fingerprint == PAIR_CATALOG_FINGERPRINT
    assert not (output / ".incomplete").exists()

    complete = runner._strict_json(
        output / "COMPLETE.json",
        name="test COMPLETE",
    )
    assert complete["resume_used"] is False
    assert complete["formal_800_training_performed"] is False
    assert complete["performance_evaluation_performed"] is False
    assert complete["D_V_accessed"] is False
    assert complete["D_T_accessed"] is False
