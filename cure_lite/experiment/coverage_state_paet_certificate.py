"""Read-only post-training pair diagnostics for PAET-BFA.

The certificate reports the fixed-margin two-endpoint violation and raw
zero-level sign errors for all 32 optimizer pairs.  These are mandatory
diagnostics, not the v21 bounded performance gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
from math import ceil, isfinite

import torch
from torch import Tensor

from ..cache.schema import stable_fingerprint
from ..coverage_state_phase_aligned_evidence_transport import (
    CURELitePhaseAlignedEvidenceTransportLevelSet,
)
from ..coverage_state_precomputed_cache import (
    CoverageStateCachedPair,
    CoverageStateScalarCache,
)
from ..paired_types import tensor_content_fingerprint
from .coverage_state_bfa_certificate import (
    _fixed_optimizer_pairs,
    _stack_target_geometry,
)


COVERAGE_STATE_PAET_CERTIFICATE_SCHEMA = (
    "cure-lite-paet-bfa-v21-post-training-pair-certificate-v1"
)
COVERAGE_STATE_PAET_CERTIFICATE_ROLE_COUNT = 16
COVERAGE_STATE_PAET_CERTIFICATE_PAIR_COUNT = 32
COVERAGE_STATE_PAET_CERTIFICATE_DEFAULT_PAIR_BATCH_SIZE = 4
COVERAGE_STATE_PAET_CERTIFICATE_POLICY = (
    "fixed-32-pair-full-valid-margin-and-raw-zero-level-diagnostic-v1"
)
COVERAGE_STATE_PAET_CERTIFICATE_ROLES = (
    "clean_positive",
    "component_null",
)


def _finite_hex(value: float, *, name: str) -> str:
    result = float(value)
    if not isfinite(result):
        raise FloatingPointError(f"{name} is non-finite")
    return result.hex()


def _model_state_fingerprint(
    model: CURELitePhaseAlignedEvidenceTransportLevelSet,
) -> str:
    return stable_fingerprint(
        {
            "class": (
                f"{type(model).__module__}.{type(model).__qualname__}"
            ),
            "config": asdict(model.config),
            "state": {
                name: tensor_content_fingerprint(value)
                for name, value in sorted(model.state_dict().items())
            },
        }
    )


def _model_gradient_fingerprint(
    model: CURELitePhaseAlignedEvidenceTransportLevelSet,
) -> str:
    return stable_fingerprint(
        {
            name: (
                None
                if parameter.grad is None
                else tensor_content_fingerprint(parameter.grad)
            )
            for name, parameter in sorted(model.named_parameters())
        }
    )


def _resolve_device(device: torch.device | str) -> torch.device:
    result = torch.device(device)
    if result.type not in {"cpu", "cuda"}:
        raise ValueError("PAET certificate device must be CPU or CUDA")
    if result.type == "cuda":
        if result.index is None:
            raise ValueError(
                "PAET certificate CUDA device needs an explicit index"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if result.index >= torch.cuda.device_count():
            raise ValueError("PAET certificate CUDA device is unavailable")
    return result


def _validate_model_device(
    model: CURELitePhaseAlignedEvidenceTransportLevelSet,
    device: torch.device,
) -> None:
    if type(model) is not CURELitePhaseAlignedEvidenceTransportLevelSet:
        raise TypeError(
            "model must be an exact "
            "CURELitePhaseAlignedEvidenceTransportLevelSet"
        )
    tensors = tuple(model.parameters()) + tuple(model.buffers())
    if not tensors:
        raise ValueError("PAET-BFA model has no state tensors")
    if any(value.device != device for value in tensors):
        raise ValueError(
            "PAET-BFA model must already be on the requested device"
        )
    if any(
        value.is_floating_point() and value.dtype != torch.float32
        for value in tensors
    ):
        raise ValueError("PAET-BFA certificate fixes FP32")


@dataclass(frozen=True)
class CoverageStatePAETPairCertificate:
    """One fixed PAET two-endpoint diagnostic."""

    pair_id: str
    sample_id: str
    optimizer_role: str
    gamma_hex: str
    gamma_plus_hex: str
    gamma_minus_hex: str
    raw_sign_error_pixels: int
    raw_sign_error_pixels_plus: int
    raw_sign_error_pixels_minus: int
    worst_endpoint: str
    worst_coordinate: tuple[int, int]
    worst_target_sign: int
    worst_target_kind: str
    worst_field_hex: str
    gamma_strictly_below_margin: bool
    raw_sign_gate_passed: bool
    pair_certificate_passed: bool

    def canonical_payload(self) -> dict[str, object]:
        return {
            "pair_id": self.pair_id,
            "sample_id": self.sample_id,
            "optimizer_role": self.optimizer_role,
            "gamma_hex": self.gamma_hex,
            "gamma_plus_hex": self.gamma_plus_hex,
            "gamma_minus_hex": self.gamma_minus_hex,
            "raw_sign_error_pixels": self.raw_sign_error_pixels,
            "raw_sign_error_pixels_plus": (
                self.raw_sign_error_pixels_plus
            ),
            "raw_sign_error_pixels_minus": (
                self.raw_sign_error_pixels_minus
            ),
            "worst_endpoint": self.worst_endpoint,
            "worst_coordinate": list(self.worst_coordinate),
            "worst_target_sign": self.worst_target_sign,
            "worst_target_kind": self.worst_target_kind,
            "worst_field_hex": self.worst_field_hex,
            "gamma_strictly_below_margin": (
                self.gamma_strictly_below_margin
            ),
            "raw_sign_gate_passed": self.raw_sign_gate_passed,
            "pair_certificate_passed": self.pair_certificate_passed,
        }


def _certificate_integrity_checks(
    *,
    pair_certificates: tuple[CoverageStatePAETPairCertificate, ...],
    model_fingerprint_before: str,
    model_fingerprint_after: str,
    model_gradient_fingerprint_before: str,
    model_gradient_fingerprint_after: str,
    model_training_mode_before: bool,
    model_training_mode_after: bool,
    cpu_rng_preserved: bool,
    device_rng_preserved: bool,
    cache_preserved: bool,
) -> tuple[tuple[str, bool], ...]:
    clean = tuple(
        value
        for value in pair_certificates
        if value.optimizer_role == "clean_positive"
    )
    component = tuple(
        value
        for value in pair_certificates
        if value.optimizer_role == "component_null"
    )
    return tuple(
        sorted(
            {
                "fixed_16_clean_positive_pairs": (
                    len(clean)
                    == COVERAGE_STATE_PAET_CERTIFICATE_ROLE_COUNT
                ),
                "fixed_16_component_null_pairs": (
                    len(component)
                    == COVERAGE_STATE_PAET_CERTIFICATE_ROLE_COUNT
                ),
                "fixed_32_unique_optimizer_pairs": (
                    len(pair_certificates)
                    == COVERAGE_STATE_PAET_CERTIFICATE_PAIR_COUNT
                    and len(
                        {value.pair_id for value in pair_certificates}
                    )
                    == len(pair_certificates)
                ),
                "model_state_preserved": (
                    model_fingerprint_before == model_fingerprint_after
                ),
                "model_gradient_buffers_preserved": (
                    model_gradient_fingerprint_before
                    == model_gradient_fingerprint_after
                ),
                "model_training_mode_preserved": (
                    model_training_mode_before
                    == model_training_mode_after
                ),
                "scalar_cache_preserved": bool(cache_preserved),
                "global_cpu_rng_preserved": bool(cpu_rng_preserved),
                "selected_device_rng_preserved": (
                    bool(device_rng_preserved)
                ),
                "pair_values_reported_not_bounded_gate": True,
                "optimizer_not_constructed": True,
                "backward_not_performed": True,
                "training_not_performed": True,
                "external_data_not_accessed": True,
            }.items()
        )
    )


@dataclass(frozen=True)
class CoverageStatePAETCertificateReceipt:
    """Immutable read-only PAET certificate and diagnostic summary."""

    schema_version: str
    certificate_policy: str
    cache_fingerprint: str
    model_fingerprint_before: str
    model_fingerprint_after: str
    model_gradient_fingerprint_before: str
    model_gradient_fingerprint_after: str
    model_training_mode_before: bool
    model_training_mode_after: bool
    device: str
    pair_batch_size: int
    model_forward_invocations: int
    margin_hex: str
    clean_positive_count: int
    component_null_count: int
    pair_certificates: tuple[CoverageStatePAETPairCertificate, ...]
    integrity_checks: tuple[tuple[str, bool], ...]
    optimizer_constructed: bool
    backward_performed: bool
    training_performed: bool
    external_data_accessed: bool

    @property
    def integrity_passed(self) -> bool:
        return bool(self.integrity_checks) and all(
            value for _, value in self.integrity_checks
        )

    @property
    def all_pairs_passed(self) -> bool:
        return bool(self.pair_certificates) and all(
            value.pair_certificate_passed
            for value in self.pair_certificates
        )

    @property
    def total_raw_sign_error_pixels(self) -> int:
        return sum(
            value.raw_sign_error_pixels
            for value in self.pair_certificates
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "certificate_policy": self.certificate_policy,
            "cache_fingerprint": self.cache_fingerprint,
            "model_fingerprint_before": self.model_fingerprint_before,
            "model_fingerprint_after": self.model_fingerprint_after,
            "model_gradient_fingerprint_before": (
                self.model_gradient_fingerprint_before
            ),
            "model_gradient_fingerprint_after": (
                self.model_gradient_fingerprint_after
            ),
            "model_training_mode_before": (
                self.model_training_mode_before
            ),
            "model_training_mode_after": self.model_training_mode_after,
            "device": self.device,
            "pair_batch_size": self.pair_batch_size,
            "model_forward_invocations": (
                self.model_forward_invocations
            ),
            "margin_hex": self.margin_hex,
            "clean_positive_count": self.clean_positive_count,
            "component_null_count": self.component_null_count,
            "pair_certificates": [
                value.canonical_payload()
                for value in self.pair_certificates
            ],
            "diagnostic_summary": {
                "all_pairs_passed": self.all_pairs_passed,
                "total_raw_sign_error_pixels": (
                    self.total_raw_sign_error_pixels
                ),
                "pair_result_is_bounded_gate": False,
            },
            "integrity_checks": dict(self.integrity_checks),
            "integrity_passed": self.integrity_passed,
            "optimizer_constructed": self.optimizer_constructed,
            "backward_performed": self.backward_performed,
            "training_performed": self.training_performed,
            "external_data_accessed": self.external_data_accessed,
        }

    @cached_property
    def receipt_fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def verify(self) -> None:
        if (
            self.schema_version != COVERAGE_STATE_PAET_CERTIFICATE_SCHEMA
            or self.certificate_policy
            != COVERAGE_STATE_PAET_CERTIFICATE_POLICY
            or self.clean_positive_count
            != COVERAGE_STATE_PAET_CERTIFICATE_ROLE_COUNT
            or self.component_null_count
            != COVERAGE_STATE_PAET_CERTIFICATE_ROLE_COUNT
            or len(self.pair_certificates)
            != COVERAGE_STATE_PAET_CERTIFICATE_PAIR_COUNT
            or self.model_forward_invocations
            != ceil(
                COVERAGE_STATE_PAET_CERTIFICATE_PAIR_COUNT
                / self.pair_batch_size
            )
            or self.optimizer_constructed
            or self.backward_performed
            or self.training_performed
            or self.external_data_accessed
        ):
            raise ValueError("PAET certificate receipt contract changed")
        expected_order = tuple(
            sorted(
                self.pair_certificates,
                key=lambda value: (
                    value.optimizer_role,
                    value.pair_id,
                ),
            )
        )
        if (
            expected_order != self.pair_certificates
            or len({value.pair_id for value in self.pair_certificates})
            != len(self.pair_certificates)
        ):
            raise ValueError("PAET pair certificates are not canonical")
        margin = float.fromhex(self.margin_hex)
        if not isfinite(margin) or margin <= 0.0:
            raise ValueError("PAET certificate margin is invalid")
        for value in self.pair_certificates:
            gamma = float.fromhex(value.gamma_hex)
            gamma_plus = float.fromhex(value.gamma_plus_hex)
            gamma_minus = float.fromhex(value.gamma_minus_hex)
            if (
                value.optimizer_role
                not in COVERAGE_STATE_PAET_CERTIFICATE_ROLES
                or gamma != max(gamma_plus, gamma_minus)
                or value.raw_sign_error_pixels
                != (
                    value.raw_sign_error_pixels_plus
                    + value.raw_sign_error_pixels_minus
                )
                or value.worst_endpoint not in {"plus", "minus"}
                or value.worst_target_sign not in {-1, 1}
                or value.worst_target_kind
                != (
                    "target"
                    if value.worst_target_sign < 0
                    else "background"
                )
                or value.gamma_strictly_below_margin
                is not (gamma < margin)
                or value.raw_sign_gate_passed
                is not (value.raw_sign_error_pixels == 0)
                or value.pair_certificate_passed
                is not (
                    value.gamma_strictly_below_margin
                    and value.raw_sign_gate_passed
                )
            ):
                raise ValueError("PAET pair certificate is inconsistent")
        expected_checks = _certificate_integrity_checks(
            pair_certificates=self.pair_certificates,
            model_fingerprint_before=self.model_fingerprint_before,
            model_fingerprint_after=self.model_fingerprint_after,
            model_gradient_fingerprint_before=(
                self.model_gradient_fingerprint_before
            ),
            model_gradient_fingerprint_after=(
                self.model_gradient_fingerprint_after
            ),
            model_training_mode_before=self.model_training_mode_before,
            model_training_mode_after=self.model_training_mode_after,
            cpu_rng_preserved=dict(self.integrity_checks).get(
                "global_cpu_rng_preserved",
                False,
            ),
            device_rng_preserved=dict(self.integrity_checks).get(
                "selected_device_rng_preserved",
                False,
            ),
            cache_preserved=dict(self.integrity_checks).get(
                "scalar_cache_preserved",
                False,
            ),
        )
        if self.integrity_checks != expected_checks:
            raise ValueError("PAET certificate integrity checks changed")


def _endpoint_statistics(
    *,
    field: Tensor,
    target_field: Tensor,
    valid_mask: Tensor,
    margin: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    target_sign = torch.sign(target_field)
    violation = (
        valid_mask.to(dtype=field.dtype)
        * torch.relu(margin - target_sign * field)
    )
    sign_error = valid_mask & torch.where(
        target_sign < 0.0,
        field >= 0.0,
        field < 0.0,
    )
    selectable = violation.masked_fill(~valid_mask, float("-inf"))
    return (
        selectable.flatten(1).amax(dim=1),
        sign_error.flatten(1).sum(dim=1),
        violation,
    )


def _pair_certificate(
    *,
    cached: CoverageStateCachedPair,
    field_plus: Tensor,
    field_minus: Tensor,
    target_plus: Tensor,
    target_minus: Tensor,
    valid_mask: Tensor,
    margin: Tensor,
) -> CoverageStatePAETPairCertificate:
    gamma_plus, error_plus, violation_plus = _endpoint_statistics(
        field=field_plus,
        target_field=target_plus,
        valid_mask=valid_mask,
        margin=margin,
    )
    gamma_minus, error_minus, violation_minus = _endpoint_statistics(
        field=field_minus,
        target_field=target_minus,
        valid_mask=valid_mask,
        margin=margin,
    )
    plus = float(gamma_plus.item())
    minus = float(gamma_minus.item())
    if plus >= minus:
        endpoint = "plus"
        violation = violation_plus
        target = target_plus
        field = field_plus
    else:
        endpoint = "minus"
        violation = violation_minus
        target = target_minus
        field = field_minus
    selectable = violation.masked_fill(~valid_mask, float("-inf"))
    flat_index = int(selectable.flatten().argmax().item())
    width = int(selectable.shape[-1])
    row = (flat_index % (selectable.shape[-2] * width)) // width
    column = flat_index % width
    target_sign = int(
        torch.sign(target[0, 0, row, column]).item()
    )
    if target_sign not in {-1, 1}:
        raise ValueError("PAET worst coordinate has no target sign")
    raw_plus = int(error_plus.item())
    raw_minus = int(error_minus.item())
    gamma = max(plus, minus)
    strict = gamma < float(margin.item())
    raw_pass = raw_plus + raw_minus == 0
    return CoverageStatePAETPairCertificate(
        pair_id=cached.record.pair_id,
        sample_id=cached.record.sample_id,
        optimizer_role=cached.optimizer_role,
        gamma_hex=_finite_hex(gamma, name="PAET gamma"),
        gamma_plus_hex=_finite_hex(plus, name="PAET plus gamma"),
        gamma_minus_hex=_finite_hex(minus, name="PAET minus gamma"),
        raw_sign_error_pixels=raw_plus + raw_minus,
        raw_sign_error_pixels_plus=raw_plus,
        raw_sign_error_pixels_minus=raw_minus,
        worst_endpoint=endpoint,
        worst_coordinate=(row, column),
        worst_target_sign=target_sign,
        worst_target_kind=(
            "target" if target_sign < 0 else "background"
        ),
        worst_field_hex=_finite_hex(
            float(field[0, 0, row, column].item()),
            name="PAET worst field",
        ),
        gamma_strictly_below_margin=strict,
        raw_sign_gate_passed=raw_pass,
        pair_certificate_passed=strict and raw_pass,
    )


def audit_coverage_state_paet_pair_certificate(
    model: CURELitePhaseAlignedEvidenceTransportLevelSet,
    cache: CoverageStateScalarCache,
    *,
    device: torch.device | str,
    pair_batch_size: int = (
        COVERAGE_STATE_PAET_CERTIFICATE_DEFAULT_PAIR_BATCH_SIZE
    ),
) -> CoverageStatePAETCertificateReceipt:
    """Report all fixed pair values without modifying model or cache."""

    resolved_device = _resolve_device(device)
    _validate_model_device(model, resolved_device)
    if (
        isinstance(pair_batch_size, bool)
        or not isinstance(pair_batch_size, int)
        or not 1
        <= pair_batch_size
        <= COVERAGE_STATE_PAET_CERTIFICATE_PAIR_COUNT
    ):
        raise ValueError("pair_batch_size must be an integer in [1, 32]")
    pairs = _fixed_optimizer_pairs(cache)
    if (
        model.config.feature_stride != cache.raw_catalog.feature_stride
        or model.config.feature_channels
        != int(pairs[0].record.feature.shape[1])
    ):
        raise ValueError("PAET model and scalar-cache geometry differ")

    cache_fingerprint = cache.cache_fingerprint
    model_before = _model_state_fingerprint(model)
    gradients_before = _model_gradient_fingerprint(model)
    training_before = bool(model.training)
    cpu_rng_before = torch.random.get_rng_state().clone()
    device_rng_before = (
        None
        if resolved_device.type != "cuda"
        else torch.cuda.get_rng_state(resolved_device).clone()
    )
    margin_value = (
        cache.sobolev_config.field_amplitude
        / float(cache.sobolev_config.truncation_radius)
    )
    certificates: list[CoverageStatePAETPairCertificate] = []
    forward_invocations = 0

    with torch.inference_mode():
        for start in range(0, len(pairs), pair_batch_size):
            chunk = pairs[start : start + pair_batch_size]
            feature_once = torch.cat(
                tuple(value.record.feature for value in chunk),
                dim=0,
            ).to(device=resolved_device, dtype=torch.float32)
            occupancy_plus = torch.cat(
                tuple(value.record.occupancy_plus for value in chunk),
                dim=0,
            ).to(device=resolved_device)
            occupancy_minus = torch.cat(
                tuple(value.record.occupancy_minus for value in chunk),
                dim=0,
            ).to(device=resolved_device)
            count = len(chunk)
            field = model(
                torch.cat((feature_once, feature_once), dim=0),
                torch.cat((occupancy_plus, occupancy_minus), dim=0),
            )
            forward_invocations += 1
            if (
                field.dtype != torch.float32
                or not bool(torch.isfinite(field).all())
            ):
                raise FloatingPointError(
                    "PAET certificate fields must be finite FP32"
                )
            field_plus, field_minus = field.split(count, dim=0)
            target_plus, valid_plus = _stack_target_geometry(
                chunk,
                endpoint="plus",
                device=resolved_device,
            )
            target_minus, valid_minus = _stack_target_geometry(
                chunk,
                endpoint="minus",
                device=resolved_device,
            )
            if not torch.equal(valid_plus, valid_minus):
                raise ValueError(
                    "PAET pair endpoints have different valid domains"
                )
            margin = torch.full(
                (),
                margin_value,
                dtype=torch.float32,
                device=resolved_device,
            )
            for index, cached in enumerate(chunk):
                certificates.append(
                    _pair_certificate(
                        cached=cached,
                        field_plus=field_plus[index : index + 1],
                        field_minus=field_minus[index : index + 1],
                        target_plus=target_plus[index : index + 1],
                        target_minus=target_minus[index : index + 1],
                        valid_mask=valid_plus[index : index + 1],
                        margin=margin,
                    )
                )

    cache.verify_unchanged()
    cache_preserved = cache.cache_fingerprint == cache_fingerprint
    model_after = _model_state_fingerprint(model)
    gradients_after = _model_gradient_fingerprint(model)
    training_after = bool(model.training)
    cpu_rng_preserved = torch.equal(
        cpu_rng_before,
        torch.random.get_rng_state(),
    )
    device_rng_preserved = (
        device_rng_before is None
        or torch.equal(
            device_rng_before,
            torch.cuda.get_rng_state(resolved_device),
        )
    )
    ordered = tuple(certificates)
    checks = _certificate_integrity_checks(
        pair_certificates=ordered,
        model_fingerprint_before=model_before,
        model_fingerprint_after=model_after,
        model_gradient_fingerprint_before=gradients_before,
        model_gradient_fingerprint_after=gradients_after,
        model_training_mode_before=training_before,
        model_training_mode_after=training_after,
        cpu_rng_preserved=cpu_rng_preserved,
        device_rng_preserved=device_rng_preserved,
        cache_preserved=cache_preserved,
    )
    result = CoverageStatePAETCertificateReceipt(
        schema_version=COVERAGE_STATE_PAET_CERTIFICATE_SCHEMA,
        certificate_policy=COVERAGE_STATE_PAET_CERTIFICATE_POLICY,
        cache_fingerprint=cache_fingerprint,
        model_fingerprint_before=model_before,
        model_fingerprint_after=model_after,
        model_gradient_fingerprint_before=gradients_before,
        model_gradient_fingerprint_after=gradients_after,
        model_training_mode_before=training_before,
        model_training_mode_after=training_after,
        device=str(resolved_device),
        pair_batch_size=pair_batch_size,
        model_forward_invocations=forward_invocations,
        margin_hex=_finite_hex(
            margin_value,
            name="PAET certificate margin",
        ),
        clean_positive_count=sum(
            value.optimizer_role == "clean_positive"
            for value in ordered
        ),
        component_null_count=sum(
            value.optimizer_role == "component_null"
            for value in ordered
        ),
        pair_certificates=ordered,
        integrity_checks=checks,
        optimizer_constructed=False,
        backward_performed=False,
        training_performed=False,
        external_data_accessed=False,
    )
    result.verify()
    if not result.integrity_passed:
        raise RuntimeError("PAET certificate integrity did not pass")
    return result


__all__ = [
    "COVERAGE_STATE_PAET_CERTIFICATE_DEFAULT_PAIR_BATCH_SIZE",
    "COVERAGE_STATE_PAET_CERTIFICATE_PAIR_COUNT",
    "COVERAGE_STATE_PAET_CERTIFICATE_POLICY",
    "COVERAGE_STATE_PAET_CERTIFICATE_ROLE_COUNT",
    "COVERAGE_STATE_PAET_CERTIFICATE_SCHEMA",
    "CoverageStatePAETCertificateReceipt",
    "CoverageStatePAETPairCertificate",
    "audit_coverage_state_paet_pair_certificate",
]
