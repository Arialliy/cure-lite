"""Exact fixed-sample builder correction for PAET evaluation-v3.

The function below is copied from the sealed original evaluator.  Its sole
logic change is:

``sources.artifact.model`` -> ``sources.attempt.artifact.model``.

The binding seal stores the formal artifact under ``attempt``.  No model,
data, metric, gate, threshold, batching, decoding, or inference behavior is
changed.
"""

from __future__ import annotations

import ast
from hashlib import sha256
import inspect
import textwrap

import torch

from cure_lite.calibration import CalibrationSample
from cure_lite.experiment import (
    coverage_state_paet_formal_evaluation as _original,
)
from cure_lite.experiment.coverage_state_paet_formal_evaluation import (
    PAET_FORMAL_BASE_THRESHOLD,
    PAETFormalArtifactBinding,
    PAETFixedDVSamples,
    _PAETFixedDVSamplesSeal,
    _PAET_FIXED_DV_SAMPLES_ISSUER,
    fixed_paet_completion,
)
from cure_lite.experiment.evaluation_pipeline import (
    calibration_samples_fingerprint,
)
from cure_lite.frozen_base import module_state_fingerprint


ORIGINAL_MODEL_ACCESS = "sources.artifact.model"
CORRECTED_MODEL_ACCESS = "sources.attempt.artifact.model"
ORIGINAL_BUILDER_SOURCE_SHA256 = (
    "31ac969d1c0455c5f0f0e8e5434398f5c0f5901c87b0d179e6da8295e64f5492"
)


def build_paet_fixed_d_v_samples(
    artifact_binding: PAETFormalArtifactBinding,
    *,
    batch_size: int = 8,
) -> PAETFixedDVSamples:
    """Run the exact formal PAET artifact over one verified D_V base cache."""

    if type(artifact_binding) is not PAETFormalArtifactBinding:
        raise TypeError(
            "artifact_binding must be PAETFormalArtifactBinding"
        )
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    artifact_binding.verify_unchanged()
    sources = artifact_binding._sealed_inputs()
    bundle = sources.bundle
    comparison_protocol = sources.comparison_protocol
    model = sources.attempt.artifact.model
    artifact_binding.verify_cache_and_protocol(bundle, comparison_protocol)
    artifact_binding.verify_model(model)
    bundle.verify_unchanged()
    feature_channels = {
        int(row.base_output.feature.shape[1]) for row in bundle.rows
    }
    if feature_channels != {model.feature_channels}:
        raise RuntimeError(
            "D_V feature channels differ from the PAET artifact"
        )
    parameters = tuple(model.parameters())
    if not parameters:
        raise RuntimeError("PAET model unexpectedly has no parameters")
    devices = {parameter.device for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    if len(devices) != 1 or dtypes != {torch.float32}:
        raise RuntimeError(
            "PAET D_V evaluation requires one-device FP32 parameters"
        )
    device = next(iter(devices))

    base_samples: list[CalibrationSample] = []
    cure_samples: list[CalibrationSample] = []
    zero_pixels = 0
    negative_pixels = 0
    completion_pixels = 0
    initial_state = module_state_fingerprint(model)
    was_training = model.training
    try:
        model.eval()
        for start in range(0, len(bundle.rows), batch_size):
            rows = bundle.rows[start : start + batch_size]
            features = torch.cat(
                [row.base_output.feature for row in rows],
                dim=0,
            ).to(device=device, dtype=torch.float32)
            probabilities_cpu = torch.cat(
                [row.base_output.probability for row in rows],
                dim=0,
            ).detach().to(device="cpu", dtype=torch.float32)
            occupancies = (
                probabilities_cpu >= PAET_FORMAL_BASE_THRESHOLD
            ).to(device=device)
            with torch.no_grad():
                field = model(features, occupancies)
                if (
                    tuple(field.shape) != tuple(occupancies.shape)
                    or field.dtype != torch.float32
                    or not bool(torch.isfinite(field).all())
                ):
                    raise RuntimeError(
                        "PAET field violates its formal output contract"
                    )
                completion = fixed_paet_completion(
                    field,
                    occupancies,
                )
            zero_pixels += int(torch.count_nonzero(field == 0).item())
            negative_pixels += int(torch.count_nonzero(field < 0).item())
            completion_pixels += int(
                torch.count_nonzero(completion).item()
            )
            completion_cpu = completion.to(device="cpu")
            for index, row in enumerate(rows):
                base = row.base_output.probability.detach().to(
                    device="cpu",
                    dtype=torch.float32,
                )
                gt = row.gt_mask.detach().to(device="cpu")
                base_candidate = CalibrationSample(
                    row.sample_id,
                    base,
                    torch.zeros_like(base),
                    gt,
                )
                cure_candidate = CalibrationSample(
                    row.sample_id,
                    base,
                    completion_cpu[index : index + 1].to(
                        dtype=torch.float32
                    ),
                    gt,
                )
                normalized_base, zero, normalized_gt = (
                    base_candidate.normalized()
                )
                cure_base, fixed_completion, cure_gt = (
                    cure_candidate.normalized()
                )
                base_samples.append(
                    CalibrationSample(
                        row.sample_id,
                        normalized_base,
                        zero,
                        normalized_gt,
                    )
                )
                cure_samples.append(
                    CalibrationSample(
                        row.sample_id,
                        cure_base,
                        fixed_completion,
                        cure_gt,
                    )
                )
    finally:
        model.train(was_training)

    if module_state_fingerprint(model) != initial_state:
        raise RuntimeError("PAET model changed during D_V evaluation")
    artifact_binding.verify_model(model)
    bundle.verify_unchanged()
    base_tuple = tuple(base_samples)
    cure_tuple = tuple(cure_samples)
    ordered_ids = tuple(row.sample_id for row in bundle.rows)
    result = PAETFixedDVSamples(
        base_samples=base_tuple,
        cure_samples=cure_tuple,
        ordered_sample_ids=ordered_ids,
        base_samples_fingerprint=calibration_samples_fingerprint(
            base_tuple
        ),
        cure_samples_fingerprint=calibration_samples_fingerprint(
            cure_tuple
        ),
        exact_zero_field_pixels=zero_pixels,
        negative_field_pixels=negative_pixels,
        completion_pixels=completion_pixels,
        artifact_binding_fingerprint=(
            artifact_binding.binding_fingerprint
        ),
        comparison_protocol_fingerprint=(
            comparison_protocol.comparison_protocol_fingerprint
        ),
        d_v_base_index_fingerprint=bundle.base_index_fingerprint,
        d_v_image_fingerprint=bundle.d_v_image_fingerprint,
        d_v_gt_fingerprint=bundle.d_v_gt_fingerprint,
        _seal=_PAETFixedDVSamplesSeal(
            issuer=_PAET_FIXED_DV_SAMPLES_ISSUER,
            binding=artifact_binding,
        ),
    )
    return result


def verify_fixed_sample_builder_correction() -> dict[str, object]:
    """Prove the copied function differs by exactly one attribute access."""

    original_source = inspect.getsource(
        _original.build_paet_fixed_d_v_samples
    )
    corrected_source = inspect.getsource(build_paet_fixed_d_v_samples)
    original_sha256 = sha256(original_source.encode("utf-8")).hexdigest()
    corrected_sha256 = sha256(corrected_source.encode("utf-8")).hexdigest()
    if original_sha256 != ORIGINAL_BUILDER_SOURCE_SHA256:
        raise RuntimeError("sealed original fixed-sample builder changed")
    if original_source.count(ORIGINAL_MODEL_ACCESS) != 1:
        raise RuntimeError(
            "original builder no longer has the single recorded bad access"
        )
    if CORRECTED_MODEL_ACCESS in original_source:
        raise RuntimeError("original builder unexpectedly contains correction")
    if corrected_source.count(CORRECTED_MODEL_ACCESS) != 1:
        raise RuntimeError(
            "corrected builder must contain one corrected access"
        )
    if ORIGINAL_MODEL_ACCESS in corrected_source:
        raise RuntimeError("corrected builder retains the bad access")
    byte_normalized = corrected_source.replace(
        CORRECTED_MODEL_ACCESS,
        ORIGINAL_MODEL_ACCESS,
        1,
    )
    if byte_normalized != original_source:
        raise RuntimeError(
            "fixed-sample builder differs beyond the one access path"
        )
    original_ast = ast.dump(
        ast.parse(textwrap.dedent(original_source)),
        include_attributes=False,
    )
    normalized_ast = ast.dump(
        ast.parse(textwrap.dedent(byte_normalized)),
        include_attributes=False,
    )
    if normalized_ast != original_ast:
        raise RuntimeError(
            "fixed-sample builder AST differs beyond the one access path"
        )
    return {
        "verified": True,
        "original_function": (
            "cure_lite.experiment.coverage_state_paet_formal_evaluation."
            "build_paet_fixed_d_v_samples"
        ),
        "corrected_function": (
            "cure_lite_eval_v3.fixed_sample_builder_v3."
            "build_paet_fixed_d_v_samples"
        ),
        "original_source_sha256": original_sha256,
        "corrected_source_sha256": corrected_sha256,
        "sole_change": {
            "from": ORIGINAL_MODEL_ACCESS,
            "to": CORRECTED_MODEL_ACCESS,
        },
        "byte_equivalent_after_inverse_substitution": True,
        "ast_equivalent_after_inverse_substitution": True,
    }


__all__ = [
    "CORRECTED_MODEL_ACCESS",
    "ORIGINAL_BUILDER_SOURCE_SHA256",
    "ORIGINAL_MODEL_ACCESS",
    "build_paet_fixed_d_v_samples",
    "verify_fixed_sample_builder_correction",
]
