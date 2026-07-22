"""Strict, non-pickle decoder artifacts for formal CURE-Lite runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import torch

from ..cache.schema import canonical_json, file_sha256, stable_fingerprint
from ..config import (
    DecoderConfig,
    InterventionConfig,
    LossConfig,
    MatchConfig,
    OccupancyConfig,
    TrainingConfig,
    config_to_dict,
)
from ..decoder import CURELiteDecoder


DECODER_ARTIFACT_SCHEMA = "cure-lite-decoder-artifact-v2"
DECODER_VARIANTS = frozenset(
    {"factual_only", "factual_exposure_matched", "uniform_legal"}
)
_WEIGHTS_NAME = "decoder.safetensors"
_LOG_NAME = "train_log.json"
_RECEIPT_NAME = "receipt.json"
_SHA256 = frozenset("0123456789abcdef")
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
        "artifact_fingerprint",
    }
)


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in _SHA256 for character in normalized):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA256")
    return normalized


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result == 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


@dataclass(frozen=True)
class DecoderRunConfig:
    """All choices that determine one F or U decoder training run."""

    variant: str
    manifest_fingerprint: str
    manifest_file_sha256: str
    preprocessing_fingerprint: str
    base_fingerprint: str
    state_fingerprint: str
    gt_fingerprint: str
    base_index_fingerprint: str
    base_index_sha256: str
    state_index_fingerprint: str
    state_index_sha256: str
    initial_decoder_fingerprint: str
    occupancy_config: OccupancyConfig
    match_config: MatchConfig
    intervention_config: InterventionConfig
    global_seed: int
    trained_epochs: int
    steps_per_epoch: int
    decoder_config: DecoderConfig
    loss_config: LossConfig = LossConfig()
    training_config: TrainingConfig = TrainingConfig()
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    factual_miss_batch: int = 4
    factual_no_miss_batch: int = 4
    synthetic_batch: int = 4
    schema_version: str = DECODER_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != DECODER_ARTIFACT_SCHEMA:
            raise ValueError(f"unsupported decoder artifact schema {self.schema_version!r}")
        if self.variant not in DECODER_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(DECODER_VARIANTS)}")
        for name in (
            "manifest_fingerprint",
            "manifest_file_sha256",
            "preprocessing_fingerprint",
            "base_fingerprint",
            "state_fingerprint",
            "gt_fingerprint",
            "base_index_fingerprint",
            "base_index_sha256",
            "state_index_fingerprint",
            "state_index_sha256",
            "initial_decoder_fingerprint",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name=name))
        if isinstance(self.global_seed, bool) or not isinstance(self.global_seed, int):
            raise TypeError("global_seed must be an integer")
        for name in (
            "trained_epochs",
            "steps_per_epoch",
            "factual_miss_batch",
            "factual_no_miss_batch",
            "synthetic_batch",
        ):
            _positive_integer(getattr(self, name), name=name)
        if not isinstance(self.decoder_config, DecoderConfig):
            raise TypeError("decoder_config must be DecoderConfig")
        if not isinstance(self.occupancy_config, OccupancyConfig):
            raise TypeError("occupancy_config must be OccupancyConfig")
        if not isinstance(self.match_config, MatchConfig):
            raise TypeError("match_config must be MatchConfig")
        if not isinstance(self.intervention_config, InterventionConfig):
            raise TypeError("intervention_config must be InterventionConfig")
        if not isinstance(self.loss_config, LossConfig):
            raise TypeError("loss_config must be LossConfig")
        if not isinstance(self.training_config, TrainingConfig):
            raise TypeError("training_config must be TrainingConfig")
        if self.optimizer not in {"adam", "sgd"}:
            raise ValueError("optimizer must be 'adam' or 'sgd'")
        object.__setattr__(
            self,
            "learning_rate",
            _finite_nonnegative(
                self.learning_rate, name="learning_rate", positive=True
            ),
        )
        object.__setattr__(
            self,
            "weight_decay",
            _finite_nonnegative(self.weight_decay, name="weight_decay"),
        )

    @property
    def branch_batch_sizes(self) -> dict[str, int]:
        return {
            "factual_miss": self.factual_miss_batch,
            "factual_no_miss": self.factual_no_miss_batch,
            "synthetic": self.synthetic_batch,
        }

    @property
    def variant_contract(self) -> dict[str, Any]:
        """Describe the third loss slot that distinguishes F, Fx, and U."""

        third_slot_sources = {
            "factual_only": "absent",
            "factual_exposure_matched": "independent_factual_positive_replacement",
            "uniform_legal": "uniform_legal_deletion",
        }
        active = self.variant != "factual_only"
        return {
            "third_loss_slot_source": third_slot_sources[self.variant],
            "third_loss_slot_batch": self.synthetic_batch if active else 0,
            "third_loss_slot_coefficient": (
                "training_config.lambda_synthetic" if active else "none"
            ),
            "matched_reference_variant": (
                "uniform_legal"
                if self.variant == "factual_exposure_matched"
                else None
            ),
            "deletion_intervention_used": self.variant == "uniform_legal",
        }

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "variant": self.variant,
            "manifest_fingerprint": self.manifest_fingerprint,
            "manifest_file_sha256": self.manifest_file_sha256,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "base_fingerprint": self.base_fingerprint,
            "state_fingerprint": self.state_fingerprint,
            "gt_fingerprint": self.gt_fingerprint,
            "base_index_fingerprint": self.base_index_fingerprint,
            "base_index_sha256": self.base_index_sha256,
            "state_index_fingerprint": self.state_index_fingerprint,
            "state_index_sha256": self.state_index_sha256,
            "initial_decoder_fingerprint": self.initial_decoder_fingerprint,
            "occupancy_config": config_to_dict(self.occupancy_config),
            "matching_config": config_to_dict(self.match_config),
            "intervention_config": config_to_dict(self.intervention_config),
            "global_seed": self.global_seed,
            "trained_epochs": self.trained_epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "decoder_config": config_to_dict(self.decoder_config),
            "loss_config": config_to_dict(self.loss_config),
            "training_config": config_to_dict(self.training_config),
            "optimization_config": {
                "optimizer": self.optimizer,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
            },
            "branch_batch_sizes": self.branch_batch_sizes,
            "variant_contract": self.variant_contract,
            "fixed_stopping_rule": {
                "epochs": self.trained_epochs,
                "steps_per_epoch": self.steps_per_epoch,
            },
            "data_augmentation": "none_frozen_base_cache",
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DecoderRunConfig":
        expected = {
            "schema_version",
            "variant",
            "manifest_fingerprint",
            "manifest_file_sha256",
            "preprocessing_fingerprint",
            "base_fingerprint",
            "state_fingerprint",
            "gt_fingerprint",
            "base_index_fingerprint",
            "base_index_sha256",
            "state_index_fingerprint",
            "state_index_sha256",
            "initial_decoder_fingerprint",
            "occupancy_config",
            "matching_config",
            "intervention_config",
            "global_seed",
            "trained_epochs",
            "steps_per_epoch",
            "decoder_config",
            "loss_config",
            "training_config",
            "optimization_config",
            "branch_batch_sizes",
            "variant_contract",
            "fixed_stopping_rule",
            "data_augmentation",
        }
        if set(value) != expected:
            raise ValueError("decoder run config fields are not canonical")
        decoder = value["decoder_config"]
        occupancy = value["occupancy_config"]
        matching = value["matching_config"]
        intervention = value["intervention_config"]
        loss = value["loss_config"]
        training = value["training_config"]
        optimization = value["optimization_config"]
        batches = value["branch_batch_sizes"]
        stopping = value["fixed_stopping_rule"]
        variant_contract = value["variant_contract"]
        if not all(
            isinstance(item, Mapping)
            for item in (
                decoder,
                occupancy,
                matching,
                intervention,
                loss,
                training,
                optimization,
                batches,
                stopping,
                variant_contract,
            )
        ):
            raise TypeError("decoder artifact configuration sections must be mappings")
        if value["data_augmentation"] != "none_frozen_base_cache":
            raise ValueError("decoder artifact data_augmentation is not canonical")
        if set(optimization) != {"optimizer", "learning_rate", "weight_decay"}:
            raise ValueError("optimization_config fields are not canonical")
        if set(batches) != {"factual_miss", "factual_no_miss", "synthetic"}:
            raise ValueError("branch_batch_sizes fields are not canonical")
        if stopping != {
            "epochs": value["trained_epochs"],
            "steps_per_epoch": value["steps_per_epoch"],
        }:
            raise ValueError("fixed_stopping_rule differs from trained epochs/steps")
        config = cls(
            schema_version=value["schema_version"],
            variant=value["variant"],
            manifest_fingerprint=value["manifest_fingerprint"],
            manifest_file_sha256=value["manifest_file_sha256"],
            preprocessing_fingerprint=value["preprocessing_fingerprint"],
            base_fingerprint=value["base_fingerprint"],
            state_fingerprint=value["state_fingerprint"],
            gt_fingerprint=value["gt_fingerprint"],
            base_index_fingerprint=value["base_index_fingerprint"],
            base_index_sha256=value["base_index_sha256"],
            state_index_fingerprint=value["state_index_fingerprint"],
            state_index_sha256=value["state_index_sha256"],
            initial_decoder_fingerprint=value["initial_decoder_fingerprint"],
            occupancy_config=OccupancyConfig(**dict(occupancy)),
            match_config=MatchConfig(**dict(matching)),
            intervention_config=InterventionConfig(**dict(intervention)),
            global_seed=value["global_seed"],
            trained_epochs=value["trained_epochs"],
            steps_per_epoch=value["steps_per_epoch"],
            decoder_config=DecoderConfig(**dict(decoder)),
            loss_config=LossConfig(**dict(loss)),
            training_config=TrainingConfig(**dict(training)),
            optimizer=optimization["optimizer"],
            learning_rate=optimization["learning_rate"],
            weight_decay=optimization["weight_decay"],
            factual_miss_batch=batches["factual_miss"],
            factual_no_miss_batch=batches["factual_no_miss"],
            synthetic_batch=batches["synthetic"],
        )
        if config.canonical_payload() != dict(value):
            raise ValueError("decoder run config payload is not canonical")
        return config


@dataclass(frozen=True, slots=True)
class _LoadedArtifactSeal:
    """Object-identity seal that cannot be replayed onto replaced fields."""

    decoder: CURELiteDecoder
    config: DecoderRunConfig
    epoch_logs: tuple[Mapping[str, Any], ...]
    source_directory: Path
    decoder_state_fingerprint: str
    artifact_fingerprint: str
    receipt_sha256: str
    weights_sha256: str
    train_log_sha256: str
    train_log_fingerprint: str


@dataclass(frozen=True)
class LoadedDecoderArtifact:
    decoder: CURELiteDecoder
    config: DecoderRunConfig
    epoch_logs: tuple[Mapping[str, Any], ...]
    decoder_state_fingerprint: str
    artifact_fingerprint: str
    receipt_sha256: str
    source_directory: Path
    weights_sha256: str
    train_log_sha256: str
    train_log_fingerprint: str
    _verification_token: object

    def _verify_source_seal(self) -> None:
        seal = self._verification_token
        if type(seal) is not _LoadedArtifactSeal:
            raise TypeError(
                "LoadedDecoderArtifact must be created by the strict loader"
            )
        if (
            seal.decoder is not self.decoder
            or seal.config is not self.config
            or seal.epoch_logs is not self.epoch_logs
        ):
            raise TypeError("loaded decoder artifact source objects were replaced")
        if (
            seal.source_directory != self.source_directory
            or seal.decoder_state_fingerprint != self.decoder_state_fingerprint
            or seal.artifact_fingerprint != self.artifact_fingerprint
            or seal.receipt_sha256 != self.receipt_sha256
            or seal.weights_sha256 != self.weights_sha256
            or seal.train_log_sha256 != self.train_log_sha256
            or seal.train_log_fingerprint != self.train_log_fingerprint
        ):
            raise TypeError("loaded decoder artifact bound fields were replaced")

    def __post_init__(self) -> None:
        self._verify_source_seal()
        if (
            not isinstance(self.source_directory, Path)
            or not self.source_directory.is_absolute()
        ):
            raise ValueError("artifact source_directory must be an absolute Path")
        for name in (
            "decoder_state_fingerprint",
            "artifact_fingerprint",
            "receipt_sha256",
            "weights_sha256",
            "train_log_sha256",
            "train_log_fingerprint",
        ):
            _digest(getattr(self, name), name=name)
        self.verify_unchanged()

    def verify_unchanged(self) -> None:
        """Reject disk, log, or in-memory weight mutation after loading."""

        self._verify_source_seal()
        if type(self.decoder) is not CURELiteDecoder:
            raise RuntimeError("loaded decoder type changed")
        if not isinstance(self.config, DecoderRunConfig):
            raise RuntimeError("loaded decoder run config changed type")
        if (
            self.decoder.config != self.config.decoder_config
            or self.decoder.feature_channels
            != self.config.decoder_config.feature_channels
        ):
            raise RuntimeError("loaded decoder topology differs from its run config")
        with torch.random.fork_rng(devices=[]):
            expected_decoder = CURELiteDecoder(self.config.decoder_config)
        expected_topology = tuple(
            (name, type(module).__module__, type(module).__qualname__)
            for name, module in expected_decoder.named_modules()
        )
        actual_topology = tuple(
            (name, type(module).__module__, type(module).__qualname__)
            for name, module in self.decoder.named_modules()
        )
        if actual_topology != expected_topology:
            raise RuntimeError("loaded decoder module topology changed")
        if any(module.training for module in self.decoder.modules()):
            raise RuntimeError("loaded decoder left frozen evaluation mode")
        if any(parameter.requires_grad for parameter in self.decoder.parameters()):
            raise RuntimeError("loaded decoder parameters are no longer frozen")
        if self.source_directory.is_symlink():
            raise RuntimeError("loaded decoder artifact directory became a symlink")
        try:
            resolved = self.source_directory.resolve(strict=True)
        except OSError as error:
            raise RuntimeError("loaded decoder artifact directory disappeared") from error
        if resolved != self.source_directory or not resolved.is_dir():
            raise RuntimeError("loaded decoder artifact directory changed")
        members = {path.name: path for path in resolved.iterdir()}
        if set(members) != {_WEIGHTS_NAME, _LOG_NAME, _RECEIPT_NAME}:
            raise RuntimeError("loaded decoder artifact file set changed")
        if any(path.is_symlink() or not path.is_file() for path in members.values()):
            raise RuntimeError("loaded decoder artifact member changed file type")
        if file_sha256(members[_WEIGHTS_NAME]) != self.weights_sha256:
            raise RuntimeError("loaded decoder weights changed on disk")
        if file_sha256(members[_LOG_NAME]) != self.train_log_sha256:
            raise RuntimeError("loaded decoder training log changed on disk")
        if file_sha256(members[_RECEIPT_NAME]) != self.receipt_sha256:
            raise RuntimeError("loaded decoder receipt changed on disk")
        receipt = _load_json(members[_RECEIPT_NAME], name="decoder receipt")
        if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_FIELDS:
            raise RuntimeError("loaded decoder receipt fields changed")
        if (
            receipt["schema_version"] != DECODER_ARTIFACT_SCHEMA
            or receipt["artifact_type"] != "cure_lite_decoder"
            or receipt["weights_file"] != _WEIGHTS_NAME
            or receipt["train_log_file"] != _LOG_NAME
        ):
            raise RuntimeError("loaded decoder receipt identity changed")
        if receipt["run_config"] != self.config.canonical_payload():
            raise RuntimeError("loaded decoder config differs from its receipt")
        if (
            receipt["weights_sha256"] != self.weights_sha256
            or receipt["train_log_sha256"] != self.train_log_sha256
            or receipt["decoder_state_fingerprint"]
            != self.decoder_state_fingerprint
            or receipt["artifact_fingerprint"] != self.artifact_fingerprint
        ):
            raise RuntimeError("loaded decoder receipt bindings changed")
        receipt_core = {
            key: value for key, value in receipt.items() if key != "artifact_fingerprint"
        }
        if stable_fingerprint(receipt_core) != self.artifact_fingerprint:
            raise RuntimeError("loaded decoder artifact fingerprint changed")
        if stable_fingerprint(tuple(self.epoch_logs)) != self.train_log_fingerprint:
            raise RuntimeError("loaded decoder training logs changed in memory")
        if decoder_state_fingerprint(self.decoder) != self.decoder_state_fingerprint:
            raise RuntimeError("loaded decoder changed after artifact verification")


def _normalized_logs(
    logs: Sequence[Mapping[str, Any]],
    *,
    expected_epochs: int,
    expected_steps: int,
    variant: str,
    expected_branch_batches: Mapping[str, int],
) -> tuple[dict[str, Any], ...]:
    if len(logs) != expected_epochs:
        raise ValueError("epoch log count must equal trained_epochs")
    # Round-trip through the canonical encoder to reject tensors, NaN/Inf,
    # non-string keys, and other non-portable values.
    normalized = json.loads(canonical_json(tuple(logs)))
    if not isinstance(normalized, list) or any(
        not isinstance(item, dict) for item in normalized
    ):
        raise TypeError("every epoch log must be a mapping")
    expected_pools = {"factual_miss", "factual_no_miss", "synthetic"}
    for epoch, item in enumerate(normalized):
        if set(item) != {"epoch", "pool_sizes", "metrics"}:
            raise ValueError("epoch log fields are not canonical")
        if item["epoch"] != epoch:
            raise ValueError("epoch logs must be complete and zero-based")
        pools = item["pool_sizes"]
        metrics = item["metrics"]
        if not isinstance(pools, dict) or set(pools) != expected_pools:
            raise ValueError("epoch pool_sizes fields are not canonical")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in pools.values()
        ):
            raise ValueError("epoch pool sizes must be non-negative integers")
        if pools["factual_miss"] < 1 or pools["factual_no_miss"] < 1:
            raise ValueError("decoder logs require both factual training pools")
        if variant in {"factual_only", "factual_exposure_matched"} and pools["synthetic"] != 0:
            raise ValueError(
                f"{variant} logs cannot contain a deletion-synthetic pool"
            )
        if variant == "uniform_legal" and pools["synthetic"] < 1:
            raise ValueError("U logs require a non-empty deletion-synthetic pool")
        if not isinstance(metrics, dict) or not metrics:
            raise TypeError("epoch metrics must be a non-empty mapping")
        if metrics.get("steps") != expected_steps:
            raise ValueError("epoch metrics differ from fixed steps_per_epoch")
        def require_constant_branch(
            branch: str,
            *,
            active: float,
            states: float,
        ) -> None:
            expected = {
                f"{branch}/active": active,
                f"{branch}/active_min": active,
                f"{branch}/active_max": active,
                f"{branch}/states": states,
                f"{branch}/states_min": states,
                f"{branch}/states_max": states,
            }
            if any(metrics.get(name) != value for name, value in expected.items()):
                raise ValueError(
                    f"{variant} {branch} exposure must match on every step"
                )

        for branch in ("factual_miss", "factual_no_miss"):
            require_constant_branch(
                branch,
                active=1.0,
                states=float(expected_branch_batches[branch]),
            )
        if variant == "factual_only":
            try:
                require_constant_branch("synthetic", active=0.0, states=0.0)
            except ValueError as error:
                raise ValueError(
                    "F must leave the third loss slot inactive on every step"
                ) from error
        else:
            try:
                require_constant_branch(
                    "synthetic",
                    active=1.0,
                    states=float(expected_branch_batches["synthetic"]),
                )
            except ValueError as error:
                if variant == "factual_exposure_matched":
                    raise ValueError(
                        "F× third loss slot does not match synthetic_batch exposure "
                        "on every step"
                    ) from error
                raise ValueError(
                    "U third loss slot exposure differs from config on every step"
                ) from error
        for name, value in metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("epoch metrics must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError("epoch metrics must be finite")
    return tuple(normalized)


def _decoder_tensors(decoder: CURELiteDecoder) -> dict[str, torch.Tensor]:
    if not isinstance(decoder, CURELiteDecoder):
        raise TypeError("decoder must be CURELiteDecoder")
    tensors: dict[str, torch.Tensor] = {}
    for name, value in decoder.state_dict().items():
        tensor = value.detach().to(device="cpu").contiguous()
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"decoder state {name!r} must be finite floating point")
        tensors[name] = tensor
    if not tensors:
        raise ValueError("decoder state is empty")
    return tensors


def decoder_state_fingerprint(decoder: CURELiteDecoder) -> str:
    """Hash the exact decoder state independently of artifact serialization."""

    digest = hashlib.sha256()
    digest.update(b"cure-lite-decoder-state-v1")
    for name, tensor in sorted(_decoder_tensors(decoder).items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(canonical_json(list(tensor.shape)).encode("ascii"))
        if tensor.numel():
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _save_decoder_artifact(
    directory: str | Path,
    decoder: CURELiteDecoder,
    config: DecoderRunConfig,
    epoch_logs: Sequence[Mapping[str, Any]],
) -> str:
    """Publish one immutable decoder artifact and return its fingerprint."""

    if not isinstance(config, DecoderRunConfig):
        raise TypeError("config must be DecoderRunConfig")
    if decoder.feature_channels != config.decoder_config.feature_channels:
        raise ValueError("decoder channels differ from decoder_config")
    logs = _normalized_logs(
        epoch_logs,
        expected_epochs=config.trained_epochs,
        expected_steps=config.steps_per_epoch,
        variant=config.variant,
        expected_branch_batches=config.branch_batch_sizes,
    )
    target = Path(directory).expanduser().resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite decoder artifact {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        try:
            from safetensors.torch import save_file
        except ImportError as error:  # pragma: no cover
            raise RuntimeError("safetensors is required for decoder artifacts") from error
        weights = staging / _WEIGHTS_NAME
        log_path = staging / _LOG_NAME
        receipt_path = staging / _RECEIPT_NAME
        save_file(_decoder_tensors(decoder), str(weights))
        log_path.write_bytes(_json_bytes(list(logs)))
        receipt_core = {
            "schema_version": DECODER_ARTIFACT_SCHEMA,
            "artifact_type": "cure_lite_decoder",
            "run_config": config.canonical_payload(),
            "weights_file": _WEIGHTS_NAME,
            "weights_sha256": file_sha256(weights),
            "decoder_state_fingerprint": decoder_state_fingerprint(decoder),
            "train_log_file": _LOG_NAME,
            "train_log_sha256": file_sha256(log_path),
        }
        artifact_fingerprint = stable_fingerprint(receipt_core)
        receipt = dict(receipt_core)
        receipt["artifact_fingerprint"] = artifact_fingerprint
        receipt_path.write_bytes(_json_bytes(receipt))
        os.rename(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    loaded = load_decoder_artifact(target, expected_config=config)
    return loaded.artifact_fingerprint


def _load_json(path: Path, *, name: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error


def load_decoder_artifact(
    directory: str | Path,
    *,
    expected_config: DecoderRunConfig | None = None,
) -> LoadedDecoderArtifact:
    """Load an exact decoder artifact and reject any changed byte or field."""

    requested_source = Path(directory).expanduser()
    if requested_source.is_symlink():
        raise ValueError("decoder artifact must not be addressed through a symlink")
    source = requested_source.resolve(strict=True)
    if not source.is_dir() or source.is_symlink():
        raise ValueError("decoder artifact must be a non-symlink directory")
    expected_names = {_WEIGHTS_NAME, _LOG_NAME, _RECEIPT_NAME}
    actual_names = {path.name for path in source.iterdir()}
    if actual_names != expected_names:
        raise ValueError("decoder artifact file set is not canonical")
    files = {name: source / name for name in expected_names}
    if any(path.is_symlink() or not path.is_file() for path in files.values()):
        raise ValueError("decoder artifact members must be regular non-symlink files")
    receipt = _load_json(files[_RECEIPT_NAME], name="decoder receipt")
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("decoder receipt fields are not canonical")
    if receipt["schema_version"] != DECODER_ARTIFACT_SCHEMA:
        raise ValueError("unsupported decoder receipt schema")
    if receipt["artifact_type"] != "cure_lite_decoder":
        raise ValueError("decoder receipt artifact_type is invalid")
    if receipt["weights_file"] != _WEIGHTS_NAME or receipt["train_log_file"] != _LOG_NAME:
        raise ValueError("decoder receipt uses non-canonical member names")
    weights_sha = _digest(receipt["weights_sha256"], name="weights_sha256")
    state_fingerprint = _digest(
        receipt["decoder_state_fingerprint"],
        name="decoder_state_fingerprint",
    )
    log_sha = _digest(receipt["train_log_sha256"], name="train_log_sha256")
    if file_sha256(files[_WEIGHTS_NAME]) != weights_sha:
        raise ValueError("decoder weights SHA256 mismatch")
    if file_sha256(files[_LOG_NAME]) != log_sha:
        raise ValueError("decoder train log SHA256 mismatch")
    receipt_core = {key: value for key, value in receipt.items() if key != "artifact_fingerprint"}
    artifact_fingerprint = _digest(
        receipt["artifact_fingerprint"], name="artifact_fingerprint"
    )
    if stable_fingerprint(receipt_core) != artifact_fingerprint:
        raise ValueError("decoder artifact fingerprint mismatch")
    run_config_raw = receipt["run_config"]
    if not isinstance(run_config_raw, Mapping):
        raise TypeError("decoder run_config must be a mapping")
    config = DecoderRunConfig.from_mapping(run_config_raw)
    if expected_config is not None and config != expected_config:
        raise ValueError("decoder artifact run config differs from expected_config")
    logs_raw = _load_json(files[_LOG_NAME], name="decoder train log")
    if not isinstance(logs_raw, list) or any(not isinstance(item, dict) for item in logs_raw):
        raise TypeError("decoder train log must be a list of mappings")
    logs = _normalized_logs(
        logs_raw,
        expected_epochs=config.trained_epochs,
        expected_steps=config.steps_per_epoch,
        variant=config.variant,
        expected_branch_batches=config.branch_batch_sizes,
    )
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("safetensors is required for decoder artifacts") from error
    decoder = CURELiteDecoder(
        feature_channels=config.decoder_config.feature_channels,
        width=config.decoder_config.width,
        groups=config.decoder_config.groups,
    )
    state = load_file(str(files[_WEIGHTS_NAME]), device="cpu")
    decoder.load_state_dict(state, strict=True)
    decoder.requires_grad_(False)
    decoder.eval()
    if decoder_state_fingerprint(decoder) != state_fingerprint:
        raise ValueError("decoder state fingerprint mismatch")
    receipt_sha256 = file_sha256(files[_RECEIPT_NAME])
    train_log_fingerprint = stable_fingerprint(tuple(logs))
    seal = _LoadedArtifactSeal(
        decoder=decoder,
        config=config,
        epoch_logs=logs,
        source_directory=source,
        decoder_state_fingerprint=state_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
        receipt_sha256=receipt_sha256,
        weights_sha256=weights_sha,
        train_log_sha256=log_sha,
        train_log_fingerprint=train_log_fingerprint,
    )
    return LoadedDecoderArtifact(
        decoder=decoder,
        config=config,
        epoch_logs=logs,
        decoder_state_fingerprint=state_fingerprint,
        artifact_fingerprint=artifact_fingerprint,
        receipt_sha256=receipt_sha256,
        source_directory=source,
        weights_sha256=weights_sha,
        train_log_sha256=log_sha,
        train_log_fingerprint=train_log_fingerprint,
        _verification_token=seal,
    )


__all__ = [
    "load_decoder_artifact",
]
