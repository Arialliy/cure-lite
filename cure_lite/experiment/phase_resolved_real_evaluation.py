"""Generic D_V residual-sample construction for the PFCR decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..calibration import CalibrationSample
from ..cache.schema import stable_fingerprint
from ..config import OccupancyConfig
from ..phase_resolved_real_cache import PFCRRealCacheContract
from ..phase_resolved_relation_decoder import (
    CURELitePhaseResolvedRelationDecoder,
)
from .cache_pipeline import LoadedDVCacheBundle
from .evaluation_pipeline import calibration_samples_fingerprint


PFCR_DV_SAMPLE_ADAPTER_SCHEMA = (
    "cure-lite-pfcr-dv-sample-adapter-v1"
)


@dataclass(frozen=True)
class PFCRDVSamples:
    """Canonical CPU samples ready for the existing D_V calibrator."""

    samples: tuple[CalibrationSample, ...]
    ordered_sample_ids: tuple[str, ...]
    sample_tensor_fingerprint: str
    cache_contract_fingerprint: str
    d_v_base_index_fingerprint: str
    d_v_image_fingerprint: str
    d_v_gt_fingerprint: str
    adapter_fingerprint: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": PFCR_DV_SAMPLE_ADAPTER_SCHEMA,
            "split": "D_V",
            "sample_count": len(self.samples),
            "ordered_sample_ids": list(self.ordered_sample_ids),
            "sample_tensor_fingerprint": (
                self.sample_tensor_fingerprint
            ),
            "cache_contract_fingerprint": (
                self.cache_contract_fingerprint
            ),
            "d_v_base_index_fingerprint": (
                self.d_v_base_index_fingerprint
            ),
            "d_v_image_fingerprint": self.d_v_image_fingerprint,
            "d_v_gt_fingerprint": self.d_v_gt_fingerprint,
            "detector_code_executed": False,
            "base_forward_executed": False,
            "threshold_selected": False,
        }


def _build_pfcr_d_v_samples(
    bundle: LoadedDVCacheBundle,
    d_r_contract: PFCRRealCacheContract,
    decoder: CURELitePhaseResolvedRelationDecoder,
    occupancy_config: OccupancyConfig,
    *,
    batch_size: int = 8,
) -> PFCRDVSamples:
    """Run only PFCR over a strictly bound D_V base cache."""

    if not isinstance(bundle, LoadedDVCacheBundle):
        raise TypeError("bundle must be LoadedDVCacheBundle")
    if not isinstance(d_r_contract, PFCRRealCacheContract):
        raise TypeError("d_r_contract must be PFCRRealCacheContract")
    if not isinstance(
        decoder,
        CURELitePhaseResolvedRelationDecoder,
    ):
        raise TypeError("decoder must be the PFCR decoder")
    if not isinstance(occupancy_config, OccupancyConfig):
        raise TypeError("occupancy_config must be OccupancyConfig")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    bundle.verify_unchanged()
    bindings = {
        "split_manifest_fingerprint": (
            bundle.split_manifest_fingerprint,
            d_r_contract.split_manifest_fingerprint,
        ),
        "preprocessing_fingerprint": (
            bundle.preprocessing_fingerprint,
            d_r_contract.preprocessing_fingerprint,
        ),
        "base_fingerprint": (
            bundle.base_fingerprint,
            d_r_contract.base_fingerprint,
        ),
        "base_state_fingerprint": (
            bundle.base_state_fingerprint,
            d_r_contract.base_state_fingerprint,
        ),
    }
    for name, (actual, expected) in bindings.items():
        if actual != expected:
            raise RuntimeError(f"D_R/D_V PFCR mismatch for {name}")
    if (
        decoder.feature_channels != d_r_contract.feature_channels
        or decoder.feature_stride != d_r_contract.feature_stride
    ):
        raise ValueError("PFCR decoder and D_R cache contract differ")
    parameters = tuple(decoder.parameters())
    devices = {parameter.device for parameter in parameters}
    dtypes = {parameter.dtype for parameter in parameters}
    if len(devices) != 1 or dtypes != {torch.float32}:
        raise RuntimeError(
            "PFCR D_V evaluation requires one-device FP32 parameters"
        )
    device = next(iter(devices))

    samples: list[CalibrationSample] = []
    was_training = decoder.training
    try:
        decoder.eval()
        for start in range(0, len(bundle.rows), batch_size):
            rows = bundle.rows[start : start + batch_size]
            feature = torch.cat(
                [row.base_output.feature for row in rows],
                dim=0,
            ).to(device=device, dtype=torch.float32)
            probability = torch.cat(
                [row.base_output.probability for row in rows],
                dim=0,
            )
            if (
                tuple(feature.shape[1:])
                != (
                    d_r_contract.feature_channels,
                    *d_r_contract.feature_shape,
                )
                or tuple(probability.shape[1:])
                != (1, *d_r_contract.output_shape)
            ):
                raise ValueError(
                    "D_V tensor shapes differ from the D_R PFCR contract"
                )
            occupancy_cpu = (
                probability >= occupancy_config.threshold
            )
            occupancy = occupancy_cpu.to(device=device)
            with torch.no_grad():
                logits = decoder(feature, occupancy)
                if (
                    tuple(logits.shape) != tuple(occupancy.shape)
                    or not bool(torch.isfinite(logits).all())
                ):
                    raise RuntimeError(
                        "PFCR D_V logits violate the native output contract"
                    )
                residual = torch.sigmoid(logits).masked_fill(
                    occupancy,
                    0.0,
                ).cpu()
            for index, row in enumerate(rows):
                candidate = CalibrationSample(
                    sample_id=row.sample_id,
                    base_probability=row.base_output.probability,
                    residual_probability=residual[index : index + 1],
                    gt_mask=row.gt_mask,
                )
                base, residual_probability, gt = candidate.normalized()
                samples.append(
                    CalibrationSample(
                        row.sample_id,
                        base,
                        residual_probability,
                        gt,
                    )
                )
    finally:
        decoder.train(was_training)
    bundle.verify_unchanged()

    sample_tuple = tuple(samples)
    ordered_ids = tuple(sample.sample_id for sample in sample_tuple)
    expected_ids = tuple(row.sample_id for row in bundle.rows)
    if ordered_ids != expected_ids:
        raise RuntimeError("PFCR D_V sample ordering changed")
    tensor_fingerprint = calibration_samples_fingerprint(sample_tuple)
    payload = {
        "schema_version": PFCR_DV_SAMPLE_ADAPTER_SCHEMA,
        "split": "D_V",
        "sample_count": len(sample_tuple),
        "ordered_sample_ids": list(ordered_ids),
        "sample_tensor_fingerprint": tensor_fingerprint,
        "cache_contract_fingerprint": (
            d_r_contract.contract_fingerprint
        ),
        "d_v_base_index_fingerprint": (
            bundle.base_index_fingerprint
        ),
        "d_v_image_fingerprint": bundle.d_v_image_fingerprint,
        "d_v_gt_fingerprint": bundle.d_v_gt_fingerprint,
        "detector_code_executed": False,
        "base_forward_executed": False,
        "threshold_selected": False,
    }
    result = PFCRDVSamples(
        samples=sample_tuple,
        ordered_sample_ids=ordered_ids,
        sample_tensor_fingerprint=tensor_fingerprint,
        cache_contract_fingerprint=(
            d_r_contract.contract_fingerprint
        ),
        d_v_base_index_fingerprint=bundle.base_index_fingerprint,
        d_v_image_fingerprint=bundle.d_v_image_fingerprint,
        d_v_gt_fingerprint=bundle.d_v_gt_fingerprint,
        adapter_fingerprint=stable_fingerprint(payload),
    )
    if result.canonical_payload() != payload:
        raise AssertionError("PFCR D_V adapter payload drifted")
    return result


__all__ = [
    "PFCR_DV_SAMPLE_ADAPTER_SCHEMA",
    "PFCRDVSamples",
]
