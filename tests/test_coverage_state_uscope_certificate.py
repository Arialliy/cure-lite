from __future__ import annotations

from dataclasses import replace
from types import MethodType

import pytest
import torch

from cure_lite.coverage_state_centered_mixed_interaction import (
    CURELiteCenteredMixedInteractionLevelSet,
    CoverageStateCenteredMixedInteractionConfig,
)
from cure_lite.experiment.coverage_state_uscope_certificate import (
    COVERAGE_STATE_USCOPE_CERTIFICATE_PAIR_COUNT,
    COVERAGE_STATE_USCOPE_CERTIFICATE_ROLE_COUNT,
    audit_coverage_state_uscope_pair_certificate,
)
from cure_lite.paired_types import tensor_content_fingerprint
from tests_v15.coverage_state_test_helpers import TOY_STRIDE
from tests_v15.coverage_state_training_test_helpers import (
    make_bounded_training_scalar_cache,
    make_training_scalar_cache,
)


@pytest.fixture(scope="module")
def bounded_cache():
    return make_bounded_training_scalar_cache()


def _model() -> CURELiteCenteredMixedInteractionLevelSet:
    return CURELiteCenteredMixedInteractionLevelSet(
        CoverageStateCenteredMixedInteractionConfig(
            feature_channels=2,
            feature_stride=TOY_STRIDE,
            width=3,
        )
    )


def _install_target_sign_forward(
    model: CURELiteCenteredMixedInteractionLevelSet,
    cache,
) -> dict[str, int]:
    fields: dict[tuple[str, str], torch.Tensor] = {}
    for cached in (
        *cache.clean_positive_records,
        *cache.component_null_records,
    ):
        feature_key = tensor_content_fingerprint(cached.record.feature)
        for endpoint in ("plus", "minus"):
            occupancy = getattr(
                cached.record,
                f"occupancy_{endpoint}",
            )
            target = getattr(
                cached.joint_targets,
                f"target_field_{endpoint}",
            )
            fields[
                (
                    feature_key,
                    tensor_content_fingerprint(occupancy),
                )
            ] = torch.sign(target) * 0.5
    calls = {"count": 0}

    def forward(self, feature, occupancy):
        calls["count"] += 1
        rows = []
        for index in range(feature.shape[0]):
            key = (
                tensor_content_fingerprint(
                    feature[index : index + 1]
                ),
                tensor_content_fingerprint(
                    occupancy[index : index + 1]
                ),
            )
            rows.append(
                fields[key].to(
                    device=feature.device,
                    dtype=torch.float32,
                )
            )
        return torch.cat(rows, dim=0)

    model.forward = MethodType(forward, model)
    return calls


def test_certificate_is_read_only_chunked_and_excludes_response_gate(
    bounded_cache,
) -> None:
    model = _model().train()
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(
            parameter,
            float(index + 1),
        )
    calls = _install_target_sign_forward(model, bounded_cache)
    cpu_rng_before = torch.random.get_rng_state().clone()

    receipt = audit_coverage_state_uscope_pair_certificate(
        model,
        bounded_cache,
        device="cpu",
        pair_batch_size=4,
    )

    assert receipt.all_pass is True
    assert receipt.gate_passed is True
    assert receipt.model_forward_invocations == 8
    assert calls["count"] == 8
    assert receipt.clean_positive_count == (
        COVERAGE_STATE_USCOPE_CERTIFICATE_ROLE_COUNT
    )
    assert receipt.component_null_count == (
        COVERAGE_STATE_USCOPE_CERTIFICATE_ROLE_COUNT
    )
    assert len(receipt.pair_certificates) == (
        COVERAGE_STATE_USCOPE_CERTIFICATE_PAIR_COUNT
    )
    assert all(
        float.fromhex(value.gamma_hex) == 0.0
        and value.raw_sign_error_pixels == 0
        and value.pair_certificate_passed
        for value in receipt.pair_certificates
    )
    assert receipt.model_fingerprint_before == (
        receipt.model_fingerprint_after
    )
    assert receipt.model_gradient_fingerprint_before == (
        receipt.model_gradient_fingerprint_after
    )
    assert receipt.model_training_mode_before is True
    assert receipt.model_training_mode_after is True
    assert model.training is True
    assert torch.equal(cpu_rng_before, torch.random.get_rng_state())
    assert receipt.same_sign_response_evaluated is False
    assert receipt.same_sign_response_is_gate is False
    assert "same_sign_response_excluded_from_gate" in dict(
        receipt.checks
    )
    assert not any(
        "response" in name
        and name != "same_sign_response_excluded_from_gate"
        for name, _ in receipt.checks
    )
    assert receipt.optimizer_constructed is False
    assert receipt.backward_performed is False
    assert receipt.training_performed is False
    assert receipt.external_data_accessed is False
    receipt.verify()


def test_initial_cmif_fails_with_explicit_worst_target_witness(
    bounded_cache,
) -> None:
    model = _model()
    cpu_rng_before = torch.random.get_rng_state().clone()

    receipt = audit_coverage_state_uscope_pair_certificate(
        model,
        bounded_cache,
        device=torch.device("cpu"),
        pair_batch_size=32,
    )

    assert receipt.all_pass is False
    assert receipt.model_forward_invocations == 1
    failed = tuple(
        value
        for value in receipt.pair_certificates
        if not value.pair_certificate_passed
    )
    assert len(failed) == COVERAGE_STATE_USCOPE_CERTIFICATE_ROLE_COUNT
    assert all(
        value.optimizer_role == "clean_positive"
        and value.raw_sign_error_pixels > 0
        and value.worst_endpoint == "minus"
        and value.worst_target_sign == -1
        and value.worst_target_kind == "target"
        and float.fromhex(value.gamma_hex)
        >= float.fromhex(receipt.margin_hex)
        for value in failed
    )
    mutation_checks = dict(receipt.checks)
    assert mutation_checks["model_state_preserved"] is True
    assert mutation_checks["model_gradient_buffers_preserved"] is True
    assert mutation_checks["model_training_mode_preserved"] is True
    assert mutation_checks["scalar_cache_preserved"] is True
    assert mutation_checks["global_cpu_rng_preserved"] is True
    assert mutation_checks["selected_device_rng_preserved"] is True
    assert torch.equal(cpu_rng_before, torch.random.get_rng_state())
    receipt.verify()


def test_certificate_requires_the_frozen_16_plus_16_pair_universe() -> None:
    cache = make_training_scalar_cache()
    with pytest.raises(ValueError, match="exactly 16"):
        audit_coverage_state_uscope_pair_certificate(
            _model(),
            cache,
            device="cpu",
        )


def test_receipt_rejects_an_inconsistent_pair_claim(
    bounded_cache,
) -> None:
    model = _model()
    _install_target_sign_forward(model, bounded_cache)
    receipt = audit_coverage_state_uscope_pair_certificate(
        model,
        bounded_cache,
        device="cpu",
        pair_batch_size=8,
    )
    first = replace(
        receipt.pair_certificates[0],
        pair_certificate_passed=False,
    )
    forged = replace(
        receipt,
        pair_certificates=(
            first,
            *receipt.pair_certificates[1:],
        ),
    )
    with pytest.raises(ValueError, match="inconsistent"):
        forged.verify()

