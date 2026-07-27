"""Create-only artifacts for a completed PFCR CURE-Lite formal model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isclose, isfinite
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import torch

from ..cache.schema import canonical_json, file_sha256, stable_fingerprint
from ..phase_resolved_relation_decoder import (
    CURELitePhaseResolvedRelationDecoder,
    PhaseResolvedRelationDecoderConfig,
)
from .phase_resolved_real_training import (
    PFCRRealFormalExecutionLedger,
    PFCRRealFormalTrainingConfig,
    pfcr_model_state_fingerprint,
)
from ..phase_resolved_relation_training import PFCR_TRAIN_RELATION_DIM


PFCR_REAL_DECODER_ARTIFACT_SCHEMA = (
    "cure-lite-pfcr-real-decoder-artifact-v1"
)
_WEIGHTS_NAME = "decoder.safetensors"
_LOG_NAME = "train_log.json"
_LEDGER_NAME = "execution_ledger.json"
_RECEIPT_NAME = "receipt.json"
_HEX = frozenset("0123456789abcdef")
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "run_config",
        "weights_file",
        "weights_sha256",
        "decoder_state_fingerprint",
        "train_log_file",
        "train_log_sha256",
        "execution_ledger_file",
        "execution_ledger_sha256",
        "artifact_fingerprint",
    }
)


def _digest(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _decoder_config_payload(
    config: PhaseResolvedRelationDecoderConfig,
) -> dict[str, object]:
    if not isinstance(config, PhaseResolvedRelationDecoderConfig):
        raise TypeError("decoder_config has the wrong type")
    return asdict(config)


@dataclass(frozen=True)
class PFCRRealDecoderRunConfig:
    """Complete frozen identity of one real PFCR formal training run."""

    seed: int
    cache_contract_fingerprint: str
    state_catalog_fingerprint: str
    lineage_allowlist_fingerprint: str
    formal_schedule_fingerprint: str
    preflight_result_fingerprint: str
    initial_model_fingerprint: str
    decoder_config: PhaseResolvedRelationDecoderConfig
    training_config: PFCRRealFormalTrainingConfig
    schema_version: str = PFCR_REAL_DECODER_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PFCR_REAL_DECODER_ARTIFACT_SCHEMA:
            raise ValueError("unsupported PFCR real decoder artifact schema")
        if self.seed not in {42, 43}:
            raise ValueError("PFCR real artifact seed must be 42 or 43")
        if not isinstance(
            self.decoder_config,
            PhaseResolvedRelationDecoderConfig,
        ):
            raise TypeError("decoder_config has the wrong type")
        if not isinstance(
            self.training_config,
            PFCRRealFormalTrainingConfig,
        ):
            raise TypeError("training_config has the wrong type")
        if self.training_config.seed != self.seed:
            raise ValueError("PFCR artifact and training seeds differ")
        if self.decoder_config.relation_dim != PFCR_TRAIN_RELATION_DIM:
            raise ValueError("PFCR artifact relation_dim changed")
        for name in (
            "cache_contract_fingerprint",
            "state_catalog_fingerprint",
            "lineage_allowlist_fingerprint",
            "formal_schedule_fingerprint",
            "preflight_result_fingerprint",
            "initial_model_fingerprint",
        ):
            _digest(getattr(self, name), name=name)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "cache_contract_fingerprint": (
                self.cache_contract_fingerprint
            ),
            "state_catalog_fingerprint": (
                self.state_catalog_fingerprint
            ),
            "lineage_allowlist_fingerprint": (
                self.lineage_allowlist_fingerprint
            ),
            "formal_schedule_fingerprint": (
                self.formal_schedule_fingerprint
            ),
            "preflight_result_fingerprint": (
                self.preflight_result_fingerprint
            ),
            "initial_model_fingerprint": (
                self.initial_model_fingerprint
            ),
            "decoder_config": _decoder_config_payload(
                self.decoder_config
            ),
            "training_config": self.training_config.canonical_payload(),
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "PFCRRealDecoderRunConfig":
        if not isinstance(value, Mapping):
            raise TypeError("PFCR run config must be a mapping")
        expected = {
            "schema_version",
            "seed",
            "cache_contract_fingerprint",
            "state_catalog_fingerprint",
            "lineage_allowlist_fingerprint",
            "formal_schedule_fingerprint",
            "preflight_result_fingerprint",
            "initial_model_fingerprint",
            "decoder_config",
            "training_config",
        }
        if set(value) != expected:
            raise ValueError("PFCR run config fields are not canonical")
        raw_decoder = value["decoder_config"]
        raw_training = value["training_config"]
        if not isinstance(raw_decoder, Mapping):
            raise TypeError("PFCR decoder config payload must be a mapping")
        if not isinstance(raw_training, Mapping):
            raise TypeError("PFCR training config payload must be a mapping")
        decoder_fields = set(
            PhaseResolvedRelationDecoderConfig.__dataclass_fields__
        )
        if set(raw_decoder) != decoder_fields:
            raise ValueError("PFCR decoder config fields are not canonical")
        training = PFCRRealFormalTrainingConfig(
            seed=raw_training.get("seed"),
            epochs=raw_training.get("epochs"),
            steps_per_epoch=raw_training.get("steps_per_epoch"),
            learning_rate=raw_training.get("learning_rate"),
            weight_decay=raw_training.get("weight_decay"),
            factual_miss_batch=raw_training.get(
                "branch_batch_sizes",
                {},
            ).get("factual_miss"),
            factual_no_miss_batch=raw_training.get(
                "branch_batch_sizes",
                {},
            ).get("factual_no_miss"),
            synthetic_batch=raw_training.get(
                "branch_batch_sizes",
                {},
            ).get("synthetic"),
        )
        if training.canonical_payload() != dict(raw_training):
            raise ValueError("PFCR training config payload is not canonical")
        result = cls(
            schema_version=value["schema_version"],
            seed=value["seed"],
            cache_contract_fingerprint=(
                value["cache_contract_fingerprint"]
            ),
            state_catalog_fingerprint=(
                value["state_catalog_fingerprint"]
            ),
            lineage_allowlist_fingerprint=(
                value["lineage_allowlist_fingerprint"]
            ),
            formal_schedule_fingerprint=(
                value["formal_schedule_fingerprint"]
            ),
            preflight_result_fingerprint=(
                value["preflight_result_fingerprint"]
            ),
            initial_model_fingerprint=(
                value["initial_model_fingerprint"]
            ),
            decoder_config=PhaseResolvedRelationDecoderConfig(
                **dict(raw_decoder)
            ),
            training_config=training,
        )
        if result.canonical_payload() != dict(value):
            raise ValueError("PFCR run config payload is not canonical")
        return result


def _strict_json(path: Path, *, name: str) -> Any:
    if path.is_symlink():
        raise ValueError(f"{name} may not be a symbolic link")
    source = path.resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file")

    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"{name} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{name} contains non-finite number {value}")

    with source.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _normalized_logs(
    logs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, object], ...]:
    if len(logs) != 800:
        raise ValueError("PFCR formal train log must contain 800 epochs")
    normalized = json.loads(canonical_json(tuple(logs)))
    if not isinstance(normalized, list):
        raise TypeError("PFCR formal train log must be a list")
    metric_fields = {
        "mean_total_loss",
        "mean_factual_miss_loss",
        "mean_factual_no_miss_loss",
        "mean_synthetic_loss",
        "minimum_total_loss",
        "maximum_total_loss",
        "minimum_gradient_l2_norm",
        "maximum_gradient_l2_norm",
    }
    output: list[dict[str, object]] = []
    for epoch, row in enumerate(normalized):
        if not isinstance(row, dict) or set(row) != {
            "epoch",
            "steps",
            "optimizer_updates_completed",
            "decoder_forward_calls",
            "decoder_state_evaluations",
            "metrics",
        }:
            raise ValueError("PFCR epoch log fields are not canonical")
        if (
            row["epoch"] != epoch
            or row["steps"] != 40
            or row["optimizer_updates_completed"] != (epoch + 1) * 40
            or row["decoder_forward_calls"] != 40
            or row["decoder_state_evaluations"] != 480
        ):
            raise ValueError("PFCR epoch execution accounting changed")
        metrics = row["metrics"]
        if not isinstance(metrics, dict) or set(metrics) != metric_fields:
            raise ValueError("PFCR epoch metric fields are not canonical")
        for name, value in metrics.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
            ):
                raise ValueError(
                    f"PFCR epoch metric {name} must be finite"
                )
        if (
            metrics["minimum_total_loss"]
            > metrics["maximum_total_loss"]
            or metrics["minimum_gradient_l2_norm"]
            > metrics["maximum_gradient_l2_norm"]
            or metrics["minimum_gradient_l2_norm"] <= 0.0
        ):
            raise ValueError("PFCR epoch metric bounds are inconsistent")
        branch_sum = (
            metrics["mean_factual_miss_loss"]
            + metrics["mean_factual_no_miss_loss"]
            + metrics["mean_synthetic_loss"]
        )
        if not isclose(
            metrics["mean_total_loss"],
            branch_sum,
            rel_tol=1.0e-7,
            abs_tol=1.0e-7,
        ):
            raise ValueError(
                "PFCR mean total loss differs from the three branches"
            )
        output.append(row)
    return tuple(output)


def _decoder_tensors(
    decoder: CURELitePhaseResolvedRelationDecoder,
) -> dict[str, torch.Tensor]:
    if not isinstance(
        decoder,
        CURELitePhaseResolvedRelationDecoder,
    ):
        raise TypeError("decoder must be the PFCR decoder")
    result: dict[str, torch.Tensor] = {}
    for name, tensor in decoder.state_dict().items():
        value = tensor.detach().to("cpu").contiguous()
        if (
            not value.is_floating_point()
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(
                f"PFCR decoder tensor {name!r} must be finite floating point"
            )
        result[name] = value
    if not result:
        raise ValueError("PFCR decoder state is empty")
    return result


@dataclass(frozen=True, slots=True)
class _LoadedPFCRArtifactSeal:
    decoder: CURELitePhaseResolvedRelationDecoder
    config: PFCRRealDecoderRunConfig
    epoch_logs: tuple[Mapping[str, object], ...]
    execution_ledger: PFCRRealFormalExecutionLedger
    source_directory: Path
    decoder_state_fingerprint: str
    artifact_fingerprint: str
    receipt_sha256: str
    weights_sha256: str
    train_log_sha256: str
    execution_ledger_sha256: str
    epoch_logs_fingerprint: str


@dataclass(frozen=True)
class LoadedPFCRRealDecoderArtifact:
    """A fully verified and frozen PFCR formal decoder artifact."""

    decoder: CURELitePhaseResolvedRelationDecoder
    config: PFCRRealDecoderRunConfig
    epoch_logs: tuple[Mapping[str, object], ...]
    execution_ledger: PFCRRealFormalExecutionLedger
    decoder_state_fingerprint: str
    artifact_fingerprint: str
    receipt_sha256: str
    source_directory: Path
    weights_sha256: str
    train_log_sha256: str
    execution_ledger_sha256: str
    _verification_token: object

    def _seal(self) -> _LoadedPFCRArtifactSeal:
        seal = self._verification_token
        if type(seal) is not _LoadedPFCRArtifactSeal:
            raise TypeError("PFCR artifact must come from its strict loader")
        if (
            seal.decoder is not self.decoder
            or seal.config is not self.config
            or seal.epoch_logs is not self.epoch_logs
            or seal.execution_ledger is not self.execution_ledger
        ):
            raise TypeError("loaded PFCR artifact objects were replaced")
        return seal

    def __post_init__(self) -> None:
        self.verify_unchanged()

    def verify_unchanged(self) -> None:
        seal = self._seal()
        for name in (
            "source_directory",
            "decoder_state_fingerprint",
            "artifact_fingerprint",
            "receipt_sha256",
            "weights_sha256",
            "train_log_sha256",
            "execution_ledger_sha256",
        ):
            if getattr(seal, name) != getattr(self, name):
                raise RuntimeError("loaded PFCR artifact binding changed")
        if any(module.training for module in self.decoder.modules()):
            raise RuntimeError("loaded PFCR decoder is not in evaluation mode")
        if any(
            parameter.requires_grad
            for parameter in self.decoder.parameters()
        ):
            raise RuntimeError("loaded PFCR decoder parameters are not frozen")
        if (
            self.decoder.config != self.config.decoder_config
            or pfcr_model_state_fingerprint(self.decoder)
            != self.decoder_state_fingerprint
            or stable_fingerprint(self.epoch_logs)
            != seal.epoch_logs_fingerprint
        ):
            raise RuntimeError("loaded PFCR artifact changed in memory")
        source = self.source_directory
        if source.is_symlink() or source.resolve(strict=True) != source:
            raise RuntimeError("PFCR artifact directory identity changed")
        members = {path.name: path for path in source.iterdir()}
        expected = {
            _WEIGHTS_NAME,
            _LOG_NAME,
            _LEDGER_NAME,
            _RECEIPT_NAME,
        }
        if set(members) != expected or any(
            path.is_symlink() or not path.is_file()
            for path in members.values()
        ):
            raise RuntimeError("PFCR artifact file inventory changed")
        hashes = {
            _WEIGHTS_NAME: self.weights_sha256,
            _LOG_NAME: self.train_log_sha256,
            _LEDGER_NAME: self.execution_ledger_sha256,
            _RECEIPT_NAME: self.receipt_sha256,
        }
        if any(
            file_sha256(members[name]) != digest
            for name, digest in hashes.items()
        ):
            raise RuntimeError("PFCR artifact member changed on disk")


def save_pfcr_real_decoder_artifact(
    directory: str | Path,
    decoder: CURELitePhaseResolvedRelationDecoder,
    config: PFCRRealDecoderRunConfig,
    epoch_logs: Sequence[Mapping[str, Any]],
    execution_ledger: PFCRRealFormalExecutionLedger,
) -> str:
    """Publish a model only after all 32,000 updates have completed."""

    if not isinstance(config, PFCRRealDecoderRunConfig):
        raise TypeError("config must be PFCRRealDecoderRunConfig")
    if not isinstance(
        execution_ledger,
        PFCRRealFormalExecutionLedger,
    ):
        raise TypeError("execution_ledger has the wrong type")
    if decoder.config != config.decoder_config:
        raise ValueError("PFCR decoder topology differs from the run config")
    if (
        decoder.config.expected_parameter_count
        != execution_ledger.trainable_parameter_count
    ):
        raise ValueError(
            "PFCR decoder and execution-ledger parameter counts differ"
        )
    final_fingerprint = pfcr_model_state_fingerprint(decoder)
    if (
        execution_ledger.seed != config.seed
        or execution_ledger.cache_contract_fingerprint
        != config.cache_contract_fingerprint
        or execution_ledger.state_catalog_fingerprint
        != config.state_catalog_fingerprint
        or execution_ledger.lineage_allowlist_fingerprint
        != config.lineage_allowlist_fingerprint
        or execution_ledger.formal_schedule_fingerprint
        != config.formal_schedule_fingerprint
        or execution_ledger.initial_model_fingerprint
        != config.initial_model_fingerprint
        or execution_ledger.final_model_fingerprint
        != final_fingerprint
    ):
        raise ValueError("PFCR execution ledger and run config differ")
    logs = _normalized_logs(epoch_logs)
    target = Path(directory).expanduser().resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(
            f"refusing to overwrite PFCR artifact {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            dir=target.parent,
        )
    )
    try:
        from safetensors.torch import save_file

        weights = staging / _WEIGHTS_NAME
        train_log = staging / _LOG_NAME
        ledger_path = staging / _LEDGER_NAME
        receipt_path = staging / _RECEIPT_NAME
        save_file(_decoder_tensors(decoder), str(weights))
        train_log.write_bytes(_json_bytes(list(logs)))
        ledger_path.write_bytes(
            _json_bytes(execution_ledger.canonical_payload())
        )
        receipt_core = {
            "schema_version": PFCR_REAL_DECODER_ARTIFACT_SCHEMA,
            "artifact_type": "cure_lite_pfcr_real_decoder",
            "run_config": config.canonical_payload(),
            "weights_file": _WEIGHTS_NAME,
            "weights_sha256": file_sha256(weights),
            "decoder_state_fingerprint": final_fingerprint,
            "train_log_file": _LOG_NAME,
            "train_log_sha256": file_sha256(train_log),
            "execution_ledger_file": _LEDGER_NAME,
            "execution_ledger_sha256": file_sha256(ledger_path),
        }
        receipt = {
            **receipt_core,
            "artifact_fingerprint": stable_fingerprint(receipt_core),
        }
        receipt_path.write_bytes(_json_bytes(receipt))
        os.rename(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return load_pfcr_real_decoder_artifact(target).artifact_fingerprint


def load_pfcr_real_decoder_artifact(
    directory: str | Path,
    *,
    expected_config: PFCRRealDecoderRunConfig | None = None,
) -> LoadedPFCRRealDecoderArtifact:
    """Load only a complete, canonical PFCR decoder artifact."""

    raw = Path(directory).expanduser()
    if raw.is_symlink():
        raise ValueError("PFCR artifact directory may not be a symlink")
    source = raw.resolve(strict=True)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("PFCR artifact path must be a regular directory")
    members = {path.name: path for path in source.iterdir()}
    expected = {
        _WEIGHTS_NAME,
        _LOG_NAME,
        _LEDGER_NAME,
        _RECEIPT_NAME,
    }
    if set(members) != expected or any(
        path.is_symlink() or not path.is_file()
        for path in members.values()
    ):
        raise ValueError("PFCR artifact file inventory is not canonical")
    receipt = _strict_json(
        members[_RECEIPT_NAME],
        name="PFCR artifact receipt",
    )
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("PFCR artifact receipt fields are not canonical")
    if (
        receipt["schema_version"]
        != PFCR_REAL_DECODER_ARTIFACT_SCHEMA
        or receipt["artifact_type"]
        != "cure_lite_pfcr_real_decoder"
        or receipt["weights_file"] != _WEIGHTS_NAME
        or receipt["train_log_file"] != _LOG_NAME
        or receipt["execution_ledger_file"] != _LEDGER_NAME
    ):
        raise ValueError("PFCR artifact receipt identity changed")
    expected_hashes = {
        _WEIGHTS_NAME: _digest(
            receipt["weights_sha256"],
            name="weights_sha256",
        ),
        _LOG_NAME: _digest(
            receipt["train_log_sha256"],
            name="train_log_sha256",
        ),
        _LEDGER_NAME: _digest(
            receipt["execution_ledger_sha256"],
            name="execution_ledger_sha256",
        ),
    }
    if any(
        file_sha256(members[name]) != digest
        for name, digest in expected_hashes.items()
    ):
        raise ValueError("PFCR artifact member SHA256 mismatch")
    receipt_core = dict(receipt)
    artifact_fingerprint = receipt_core.pop("artifact_fingerprint")
    if stable_fingerprint(receipt_core) != artifact_fingerprint:
        raise ValueError("PFCR artifact fingerprint mismatch")
    config = PFCRRealDecoderRunConfig.from_mapping(
        receipt["run_config"]
    )
    if expected_config is not None:
        if not isinstance(expected_config, PFCRRealDecoderRunConfig):
            raise TypeError("expected_config has the wrong type")
        if config != expected_config:
            raise ValueError("PFCR artifact config differs from expected")
    logs_raw = _strict_json(
        members[_LOG_NAME],
        name="PFCR train log",
    )
    if not isinstance(logs_raw, list):
        raise ValueError("PFCR train log must contain one list")
    logs = _normalized_logs(logs_raw)
    ledger_raw = _strict_json(
        members[_LEDGER_NAME],
        name="PFCR execution ledger",
    )
    ledger = PFCRRealFormalExecutionLedger.from_mapping(ledger_raw)
    if (
        ledger.seed != config.seed
        or ledger.cache_contract_fingerprint
        != config.cache_contract_fingerprint
        or ledger.state_catalog_fingerprint
        != config.state_catalog_fingerprint
        or ledger.lineage_allowlist_fingerprint
        != config.lineage_allowlist_fingerprint
        or ledger.formal_schedule_fingerprint
        != config.formal_schedule_fingerprint
        or ledger.initial_model_fingerprint
        != config.initial_model_fingerprint
        or config.decoder_config.expected_parameter_count
        != ledger.trainable_parameter_count
    ):
        raise ValueError("PFCR ledger and run config bindings differ")
    try:
        from safetensors.torch import load_file

        tensors = load_file(
            str(members[_WEIGHTS_NAME]),
            device="cpu",
        )
    except Exception as error:
        raise ValueError(
            "PFCR decoder weights are not valid safetensors"
        ) from error
    decoder = CURELitePhaseResolvedRelationDecoder(
        config.decoder_config
    )
    expected_state = decoder.state_dict()
    if set(tensors) != set(expected_state):
        raise ValueError("PFCR decoder weight keys changed")
    for name, expected_tensor in expected_state.items():
        actual = tensors[name]
        if (
            actual.shape != expected_tensor.shape
            or actual.dtype != expected_tensor.dtype
            or not bool(torch.isfinite(actual).all())
        ):
            raise ValueError(
                f"PFCR decoder tensor {name!r} violates its contract"
            )
    decoder.load_state_dict(tensors, strict=True)
    decoder.eval()
    decoder.requires_grad_(False)
    decoder_fingerprint = pfcr_model_state_fingerprint(decoder)
    if (
        decoder_fingerprint
        != receipt["decoder_state_fingerprint"]
        or decoder_fingerprint != ledger.final_model_fingerprint
    ):
        raise ValueError("PFCR loaded decoder fingerprint mismatch")
    receipt_sha = file_sha256(members[_RECEIPT_NAME])
    seal = _LoadedPFCRArtifactSeal(
        decoder=decoder,
        config=config,
        epoch_logs=logs,
        execution_ledger=ledger,
        source_directory=source,
        decoder_state_fingerprint=decoder_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
        receipt_sha256=receipt_sha,
        weights_sha256=expected_hashes[_WEIGHTS_NAME],
        train_log_sha256=expected_hashes[_LOG_NAME],
        execution_ledger_sha256=expected_hashes[_LEDGER_NAME],
        epoch_logs_fingerprint=stable_fingerprint(logs),
    )
    return LoadedPFCRRealDecoderArtifact(
        decoder=decoder,
        config=config,
        epoch_logs=logs,
        execution_ledger=ledger,
        decoder_state_fingerprint=decoder_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
        receipt_sha256=receipt_sha,
        source_directory=source,
        weights_sha256=expected_hashes[_WEIGHTS_NAME],
        train_log_sha256=expected_hashes[_LOG_NAME],
        execution_ledger_sha256=expected_hashes[_LEDGER_NAME],
        _verification_token=seal,
    )


__all__ = [
    "PFCR_REAL_DECODER_ARTIFACT_SCHEMA",
    "LoadedPFCRRealDecoderArtifact",
    "PFCRRealDecoderRunConfig",
    "load_pfcr_real_decoder_artifact",
    "save_pfcr_real_decoder_artifact",
]
