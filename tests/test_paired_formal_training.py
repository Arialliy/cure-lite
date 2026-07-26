from __future__ import annotations

from dataclasses import replace
import inspect

import pytest
import torch

from cure_lite.cache.schema import stable_fingerprint
from cure_lite.config import (
    DecoderConfig,
    InterventionConfig,
    LossConfig,
    MatchConfig,
    OccupancyConfig,
)
from cure_lite.decoder import CURELiteDecoder
from cure_lite.experiment.artifacts import decoder_state_fingerprint
from cure_lite.experiment.paired_artifacts import (
    PairedDecoderRunConfig,
    method_objective_contract,
)
from cure_lite.experiment.paired_formal_controls import (
    PairedFormalControlInputProvider,
)
from cure_lite.experiment import paired_formal_training as formal_training
from cure_lite.experiment.paired_formal_training import (
    FORMAL_TRAINING_METHODS,
    PAIRED_DIFFERENCE_METHOD,
    execute_paired_formal_training,
    formal_exposure_fingerprints,
)
from cure_lite.losses import CURELiteLoss
from cure_lite.paired_control_inputs import build_dct_coordinate_basis
from cure_lite.paired_losses import PairedDifferenceLoss
from cure_lite.train.paired_control_step import CONTROL_KINDS
from tests.test_paired_formal_schedule import (
    _factual_example,
    _fake_formal_schedule,
)


@pytest.fixture(scope="module")
def formal_schedule():
    factual_miss = tuple(
        _factual_example(
            branch="factual_miss",
            sample_id=f"formal-miss-{index}",
            target_id=index + 1,
        )
        for index in range(5)
    )
    factual_no_miss = tuple(
        _factual_example(
            branch="factual_no_miss",
            sample_id=f"formal-no-miss-{index}",
            target_id=None,
        )
        for index in range(4)
    )
    return _fake_formal_schedule(
        42,
        (factual_miss, factual_no_miss),
    )


def _run_config(
    schedule,
    decoder: CURELiteDecoder,
    *,
    method: str = PAIRED_DIFFERENCE_METHOD,
) -> PairedDecoderRunConfig:
    digests = iter("123456789abcdef0123456789abcdef")
    return PairedDecoderRunConfig(
        method=method,
        seed=42,
        manifest_fingerprint=next(digests) * 64,
        manifest_file_sha256=next(digests) * 64,
        preprocessing_fingerprint=next(digests) * 64,
        base_fingerprint=next(digests) * 64,
        state_fingerprint=next(digests) * 64,
        gt_fingerprint=next(digests) * 64,
        base_index_fingerprint=next(digests) * 64,
        base_index_sha256=next(digests) * 64,
        state_index_fingerprint=next(digests) * 64,
        state_index_sha256=next(digests) * 64,
        formal_protocol_fingerprint=next(digests) * 64,
        paired_objective_fingerprint=next(digests) * 64,
        pair_catalog_fingerprint=(
            schedule.paired_schedule.catalog_fingerprint
        ),
        paired_schedule_fingerprint=(
            schedule.paired_schedule.schedule_fingerprint
        ),
        formal_schedule_fingerprint=schedule.schedule_fingerprint,
        runtime_input_fingerprint=next(digests) * 64,
        control_preflight_fingerprint=next(digests) * 64,
        control_provider_fingerprint=(
            None
            if method == PAIRED_DIFFERENCE_METHOD
            else next(digests) * 64
        ),
        method_contract_fingerprint=stable_fingerprint(
            method_objective_contract(method)
        ),
        initial_decoder_fingerprint=decoder_state_fingerprint(decoder),
        occupancy_config=OccupancyConfig(),
        match_config=MatchConfig(),
        intervention_config=InterventionConfig(),
        decoder_config=decoder.config,
        absolute_loss_config=LossConfig(),
    )


class _FrozenControlProvider:
    def __call__(
        self,
        *,
        control_kind,
        pairs,
        pair_batch,
        epoch,
        step,
        device,
    ):
        del epoch, step
        gt_union = torch.stack(
            tuple(pair.completion_minus for pair in pairs),
            dim=0,
        ).to(device=device)
        if control_kind == "independent_endpoint":
            return {
                "gt_union": gt_union,
                "completion_plus": torch.stack(
                    tuple(pair.completion_plus for pair in pairs),
                    dim=0,
                ).to(device=device),
                "completion_minus": torch.stack(
                    tuple(pair.completion_minus for pair in pairs),
                    dim=0,
                ).to(device=device),
            }
        if control_kind == "after_only":
            return {"gt_union": gt_union}
        if control_kind == "coordinate_basis":
            return {
                "coordinate_basis": build_dct_coordinate_basis(
                    channels=int(pair_batch.feature.shape[1]),
                    height=int(pair_batch.feature.shape[2]),
                    width=int(pair_batch.feature.shape[3]),
                    dtype=pair_batch.feature.dtype,
                )
            }
        if control_kind == "target_permutation":
            return {
                "permuted_label_increment": (
                    pair_batch.label_increment.flip(0)
                )
            }
        return {}


def test_exposure_fingerprints_bind_all_three_sequences_and_ledgers(
    formal_schedule,
) -> None:
    fingerprints = formal_exposure_fingerprints(formal_schedule)
    assert set(fingerprints) == {
        "pair",
        "factual_miss",
        "factual_no_miss",
    }
    assert all(len(value) == 64 for value in fingerprints.values())
    payloads = formal_training._exposure_payloads(formal_schedule)
    assert all(
        payload["formal_schedule_fingerprint"]
        == formal_schedule.schedule_fingerprint
        for payload in payloads.values()
    )
    assert payloads["pair"]["sequence_fingerprint"] == (
        formal_schedule.pair_sequence_fingerprint
    )
    assert payloads["factual_miss"]["sequence_fingerprint"] == (
        formal_schedule.factual_miss_sequence_fingerprint
    )
    assert payloads["factual_no_miss"]["sequence_fingerprint"] == (
        formal_schedule.factual_no_miss_sequence_fingerprint
    )
    assert sum(
        row["count"] for row in payloads["pair"]["identities"]
    ) == 64_000
    assert sum(
        row["count"]
        for row in payloads["factual_miss"]["identities"]
    ) == 128_000
    assert sum(
        row["count"]
        for row in payloads["factual_no_miss"]["identities"]
    ) == 128_000


@pytest.mark.parametrize("method", FORMAL_TRAINING_METHODS)
def test_one_real_update_supports_primary_method_and_all_eight_controls(
    formal_schedule,
    method: str,
) -> None:
    torch.manual_seed(927)
    decoder = CURELiteDecoder(DecoderConfig(feature_channels=2))
    initial = decoder_state_fingerprint(decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    ledger = formal_training._DecoderForwardLedger(decoder)
    try:
        outcome = formal_training._execute_one_update(
            decoder,
            CURELiteLoss(),
            PairedDifferenceLoss(),
            optimizer,
            formal_schedule,
            method=method,
            epoch=0,
            step=0,
            device=torch.device("cpu"),
            control_kwargs_provider=(
                None
                if method == PAIRED_DIFFERENCE_METHOD
                else _FrozenControlProvider()
            ),
            forward_ledger=ledger,
        )
    finally:
        ledger.close()

    assert ledger.snapshot() == (3, 12)
    assert outcome.logs["optimizer_steps"] == 1
    assert torch.isfinite(torch.tensor(outcome.gradient_l2_norm))
    assert decoder_state_fingerprint(decoder) != initial
    if method == PAIRED_DIFFERENCE_METHOD:
        assert "paired/loss" in outcome.logs
    else:
        assert outcome.logs["control_kind"] == method
        assert "control/loss" in outcome.logs


def test_every_method_can_start_from_identical_external_bytes(
    formal_schedule,
) -> None:
    del formal_schedule
    fingerprints = set()
    for _method in FORMAL_TRAINING_METHODS:
        torch.manual_seed(3401)
        decoder = CURELiteDecoder(DecoderConfig(feature_channels=2))
        fingerprints.add(decoder_state_fingerprint(decoder))
    assert len(fingerprints) == 1


def test_zero_gradient_update_is_rejected() -> None:
    decoder = CURELiteDecoder(DecoderConfig(feature_channels=2))
    for parameter in decoder.parameters():
        parameter.grad = torch.zeros_like(parameter)
    with pytest.raises(RuntimeError, match="positive on every update"):
        formal_training._gradient_l2_norm(decoder)


def test_completed_adam_requires_exact_finite_moment_state() -> None:
    decoder = CURELiteDecoder(DecoderConfig(feature_channels=2))
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    for parameter in decoder.parameters():
        optimizer.state[parameter]["step"] = torch.tensor(32_000.0)
    with pytest.raises(RuntimeError, match="exactly step/exp_avg/exp_avg_sq"):
        formal_training._validate_completed_adam(decoder, optimizer)


def test_full_public_loop_builds_800_logs_and_exact_execution_ledger(
    formal_schedule,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(551)
    decoder = CURELiteDecoder(DecoderConfig(feature_channels=2))
    config = _run_config(formal_schedule, decoder)
    optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    first_update = True

    def fake_update(
        decoder,
        absolute_criterion,
        paired_criterion,
        optimizer,
        schedule,
        *,
        method,
        epoch,
        step,
        device,
        control_kwargs_provider,
        forward_ledger,
    ):
        nonlocal first_update
        del (
            absolute_criterion,
            paired_criterion,
            schedule,
            method,
            device,
            control_kwargs_provider,
        )
        forward_ledger.calls += 3
        forward_ledger.states += 12
        if first_update:
            with torch.no_grad():
                next(decoder.parameters()).add_(0.01)
            first_update = False
        if epoch == 799 and step == 39:
            for parameter in decoder.parameters():
                optimizer.state[parameter].update(
                    {
                        "step": torch.tensor(32_000.0),
                        "exp_avg": torch.zeros_like(parameter),
                        "exp_avg_sq": torch.zeros_like(parameter),
                    }
                )
        return formal_training._UpdateOutcome(
            logs={
                "total": 0.6,
                "factual_miss/loss": 0.1,
                "factual_no_miss/loss": 0.2,
                "paired/loss": 0.3,
                "optimizer_steps": 1,
            },
            gradient_l2_norm=1.25,
        )

    monkeypatch.setattr(
        formal_training,
        "_execute_one_update",
        fake_update,
    )
    result = execute_paired_formal_training(
        decoder,
        CURELiteLoss(config.absolute_loss_config),
        PairedDifferenceLoss(),
        optimizer,
        formal_schedule,
        config,
    )

    assert len(result.epoch_logs) == 800
    assert result.epoch_logs[0] == {
        "epoch": 0,
        "steps": 40,
        "metrics": {
            "mean_total_loss": pytest.approx(0.6),
            "mean_factual_miss_loss": pytest.approx(0.1),
            "mean_factual_no_miss_loss": pytest.approx(0.2),
            "mean_paired_or_control_loss": pytest.approx(0.3),
            "minimum_total_loss": pytest.approx(0.6),
            "maximum_total_loss": pytest.approx(0.6),
        },
    }
    ledger = result.execution_ledger
    exposure = formal_exposure_fingerprints(formal_schedule)
    assert ledger.formal_schedule_fingerprint == (
        formal_schedule.schedule_fingerprint
    )
    assert ledger.pair_exposure_fingerprint == exposure["pair"]
    assert ledger.factual_miss_exposure_fingerprint == (
        exposure["factual_miss"]
    )
    assert ledger.factual_no_miss_exposure_fingerprint == (
        exposure["factual_no_miss"]
    )
    assert ledger.optimizer_updates == 32_000
    assert ledger.completed_epochs == 800
    assert ledger.decoder_forward_calls == 96_000
    assert ledger.decoder_state_evaluations == 384_000
    assert ledger.backward_calls == 32_000
    assert ledger.optimizer_steps == 32_000
    assert ledger.minimum_gradient_l2_norm == pytest.approx(1.25)
    assert ledger.maximum_gradient_l2_norm == pytest.approx(1.25)
    assert ledger.final_decoder_fingerprint == (
        decoder_state_fingerprint(decoder)
    )
    assert ledger.final_decoder_fingerprint != (
        ledger.initial_decoder_fingerprint
    )


def test_entry_rejects_recovery_schedule_drift_and_control_injection(
    formal_schedule,
) -> None:
    torch.manual_seed(89)
    decoder = CURELiteDecoder(DecoderConfig(feature_channels=2))
    config = _run_config(formal_schedule, decoder)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    parameter = next(decoder.parameters())
    optimizer.state[parameter]["stale"] = torch.tensor(1)
    with pytest.raises(ValueError, match="resume is forbidden"):
        execute_paired_formal_training(
            decoder,
            CURELiteLoss(),
            PairedDifferenceLoss(),
            optimizer,
            formal_schedule,
            config,
        )

    fresh = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="bind the supplied schedule"):
        execute_paired_formal_training(
            decoder,
            CURELiteLoss(),
            PairedDifferenceLoss(),
            fresh,
            formal_schedule,
            replace(config, formal_schedule_fingerprint="f" * 64),
        )
    with pytest.raises(ValueError, match="cannot receive control inputs"):
        execute_paired_formal_training(
            decoder,
            CURELiteLoss(),
            PairedDifferenceLoss(),
            fresh,
            formal_schedule,
            config,
            control_kwargs_provider=_FrozenControlProvider(),
        )


def test_control_entry_requires_exact_frozen_provider_and_fingerprint(
    formal_schedule,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(891)
    decoder = CURELiteDecoder(DecoderConfig(feature_channels=2))
    config = _run_config(
        formal_schedule,
        decoder,
        method="after_only",
    )
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    with pytest.raises(TypeError, match="PairedFormalControlInputProvider"):
        formal_training._validate_formal_inputs(
            decoder,
            CURELiteLoss(),
            PairedDifferenceLoss(),
            optimizer,
            formal_schedule,
            config,
            _FrozenControlProvider(),
        )

    provider = object.__new__(PairedFormalControlInputProvider)
    object.__setattr__(provider, "provider_fingerprint", "0" * 64)
    monkeypatch.setattr(
        PairedFormalControlInputProvider,
        "verify_unchanged",
        lambda self: None,
    )
    with pytest.raises(ValueError, match="fingerprint"):
        formal_training._validate_formal_inputs(
            decoder,
            CURELiteLoss(),
            PairedDifferenceLoss(),
            optimizer,
            formal_schedule,
            config,
            provider,
        )


def test_public_entry_has_no_horizon_checkpoint_resume_or_split_override() -> None:
    assert FORMAL_TRAINING_METHODS == (
        PAIRED_DIFFERENCE_METHOD,
        *CONTROL_KINDS,
    )
    assert tuple(
        inspect.signature(execute_paired_formal_training).parameters
    ) == (
        "decoder",
        "absolute_criterion",
        "paired_criterion",
        "optimizer",
        "schedule",
        "config",
        "control_kwargs_provider",
    )
    forbidden = {
        "epochs",
        "steps",
        "batch_size",
        "objective_coefficients",
        "checkpoint",
        "resume",
        "split",
        "D_V",
        "D_T",
    }
    assert not forbidden & set(
        inspect.signature(execute_paired_formal_training).parameters
    )
