"""Strict Stage-A registry contracts used before the first ``D_T`` access.

The single-seed registry is deliberately validated here independently of the
CLI that created it.  A master registry therefore binds immutable, canonical
single-seed artifacts instead of trusting a few convenient fields from each
JSON file.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
import re
from typing import Any, Callable, Mapping, TypeVar

from .calibration import THRESHOLD_SELECTION_RULE
from .config import (
    DecoderConfig,
    InterventionConfig,
    LossConfig,
    MatchConfig,
    OccupancyConfig,
    TrainingConfig,
    config_to_dict,
)
from .data import PreprocessConfig


METHOD_VERSION = "cure-lite-v0.1"
SEED_REGISTRY_SCHEMA_VERSION = "stage-a-frozen-registry-v6"
MASTER_REGISTRY_SCHEMA_VERSION = "stage-a-master-registry-v6"
MINIMUM_FULL_PIPELINE_SEEDS = 5

STAGE_A_METHOD_ORDER = ("A", "Base@B", "F", "F×", "U")
STAGE_A_DECODER_VARIANTS = {
    "F": "factual_only",
    "F×": "factual_exposure_matched",
    "U": "uniform_legal",
}
STAGE_A_VARIANTS: dict[str, str | None] = {
    "A": None,
    "Base@B": None,
    **STAGE_A_DECODER_VARIANTS,
}
STAGE_A_EVALUATION_MODES = {
    "A": "anchor",
    "Base@B": "base_at_budget",
    "F": "residual",
    "F×": "residual",
    "U": "residual",
}

BASE_RUN_IDENTITY_FIELDS = (
    "producer_schema",
    "base_fingerprint",
    "base_state_fingerprint",
    "training_run_fingerprint",
    "completion_receipt_sha256",
    "checkpoint_sha256",
    "selection_fingerprint",
    "source_fingerprint",
)
BASE_RUN_UNIQUE_IDENTITY_FIELDS = (
    "training_run_fingerprint",
    "completion_receipt_sha256",
    "checkpoint_sha256",
    "selection_fingerprint",
)

# This is an exact schema, not merely a list of keys that happen to be compared
# by the current evaluator.  Base/cache identities and grid fields ensure a
# five-seed freeze cannot silently combine calibrations over different D_V
# bytes or different candidate sets.
COMMON_FIELDS = (
    "manifest_fingerprint",
    "base_fingerprint",
    "base_state_fingerprint",
    "base_run_identity",
    "stage_a_complete_fingerprint",
    "d_v_image_fingerprint",
    "d_v_gt_fingerprint",
    "d_v_base_cache_index_fingerprint",
    "d_v_base_cache_index_sha256",
    "d_r_base_cache_index_fingerprint",
    "d_r_base_cache_index_sha256",
    "d_r_state_cache_index_fingerprint",
    "d_r_state_cache_index_sha256",
    "anchor_protocol_sha256",
    "state_fingerprint",
    "tau_o",
    "tau_B",
    "pixel_fa_budget",
    "component_fa_per_mp_budget",
    "raw_background_fa_budget",
    "minimum_retention",
    "null_residual_candidate",
    "threshold_selection_rule",
    "anchor_threshold_grid",
    "anchor_threshold_grid_fingerprint",
    "residual_threshold_grid",
    "residual_threshold_grid_fingerprint",
    "base_threshold_grid",
    "base_threshold_grid_fingerprint",
    "occupancy_config",
    "matching_config",
    "intervention_config",
    "preprocessing",
    "decoder_config",
    "loss_config",
    "training_config",
    "optimization_config",
    "branch_batch_sizes",
    "data_augmentation",
    "fixed_stopping_rule",
    "global_seed",
    "steps_per_epoch",
    "trained_epochs",
    "efficiency_device_type",
    "efficiency_warmup",
    "efficiency_repetitions",
    "efficiency_static_fingerprint",
    "efficiency_receipt_fingerprint",
)

PROTOCOL_ENTRY_FIELDS = (
    "evaluation_mode",
    "decoder_variant",
    "protocol_fingerprint",
    "decoder_artifact_fingerprint",
    "selected_threshold",
)

# These fields define the paired experiment and therefore may not drift across
# full-pipeline seeds.  Seed-derived identities and calibrated thresholds are
# intentionally absent: base/state/anchor/tau_o/tau_B/occupancy/global_seed and
# the per-seed cache identities are allowed to vary.
CROSS_SEED_FIXED_FIELDS = (
    "manifest_fingerprint",
    "d_v_image_fingerprint",
    "d_v_gt_fingerprint",
    "pixel_fa_budget",
    "component_fa_per_mp_budget",
    "raw_background_fa_budget",
    "minimum_retention",
    "null_residual_candidate",
    "threshold_selection_rule",
    "matching_config",
    "intervention_config",
    "preprocessing",
    "decoder_config",
    "loss_config",
    "training_config",
    "optimization_config",
    "branch_batch_sizes",
    "data_augmentation",
    "fixed_stopping_rule",
    "steps_per_epoch",
    "trained_epochs",
    "efficiency_device_type",
    "efficiency_warmup",
    "efficiency_repetitions",
    "anchor_threshold_grid",
    "anchor_threshold_grid_fingerprint",
    "residual_threshold_grid",
    "residual_threshold_grid_fingerprint",
    "base_threshold_grid",
    "base_threshold_grid_fingerprint",
)

CROSS_SEED_VARIABLE_FIELDS = (
    "base_fingerprint",
    "base_state_fingerprint",
    "base_run_identity",
    "stage_a_complete_fingerprint",
    "d_v_base_cache_index_fingerprint",
    "d_v_base_cache_index_sha256",
    "d_r_base_cache_index_fingerprint",
    "d_r_base_cache_index_sha256",
    "d_r_state_cache_index_fingerprint",
    "d_r_state_cache_index_sha256",
    "anchor_protocol_sha256",
    "state_fingerprint",
    "tau_o",
    "tau_B",
    "occupancy_config",
    "global_seed",
    "efficiency_static_fingerprint",
    "efficiency_receipt_fingerprint",
)

# These content identities are seed-derived and therefore not cross-seed fixed,
# but honest distinct base runs cannot produce the same bound artifact digest.
# Calibrated scalar thresholds remain deliberately outside this list because
# different seeds may legitimately select the same candidate.
UNIQUE_SEED_COMMON_IDENTITY_FIELDS = (
    "global_seed",
    "base_fingerprint",
    "base_state_fingerprint",
    "stage_a_complete_fingerprint",
    "d_v_base_cache_index_fingerprint",
    "d_v_base_cache_index_sha256",
    "d_r_base_cache_index_fingerprint",
    "d_r_base_cache_index_sha256",
    "d_r_state_cache_index_fingerprint",
    "d_r_state_cache_index_sha256",
    "anchor_protocol_sha256",
    "state_fingerprint",
    "efficiency_static_fingerprint",
    "efficiency_receipt_fingerprint",
)

if set(CROSS_SEED_FIXED_FIELDS) & set(CROSS_SEED_VARIABLE_FIELDS):
    raise RuntimeError("cross-seed fixed and variable Stage-A fields overlap")
if set(CROSS_SEED_FIXED_FIELDS) | set(CROSS_SEED_VARIABLE_FIELDS) != set(COMMON_FIELDS):
    raise RuntimeError("cross-seed fixed/variable fields do not partition COMMON_FIELDS")
if tuple(STAGE_A_VARIANTS) != STAGE_A_METHOD_ORDER:
    raise RuntimeError("Stage-A method/decoder contracts do not share one order")

_TOP_LEVEL_FIELDS = (
    "schema_version",
    "artifact_type",
    "method_version",
    "stage",
    "split",
    "thresholds_frozen",
    "common_config",
    "protocols",
)
_MASTER_TOP_LEVEL_FIELDS = (
    "schema_version",
    "artifact_type",
    "method_version",
    "stage",
    "split",
    "thresholds_frozen",
    "minimum_full_pipeline_seeds",
    "seed_count",
    "cross_seed_fixed_config",
    "efficiency_protocol",
    "seed_registries",
)
_MASTER_SEED_FIELDS = (
    "registry_sha256",
    "global_seed",
    "base_fingerprint",
    "base_state_fingerprint",
    "base_run_identity",
    "stage_a_complete_fingerprint",
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SEED_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


@dataclass(frozen=True)
class BaseRunIdentity:
    """Detector-neutral identity of one completed Base training run."""

    producer_schema: str
    base_fingerprint: str
    base_state_fingerprint: str
    training_run_fingerprint: str
    completion_receipt_sha256: str
    checkpoint_sha256: str
    selection_fingerprint: str
    source_fingerprint: str

    def to_registry_dict(self) -> dict[str, str]:
        return {
            "producer_schema": self.producer_schema,
            "base_fingerprint": self.base_fingerprint,
            "base_state_fingerprint": self.base_state_fingerprint,
            "training_run_fingerprint": self.training_run_fingerprint,
            "completion_receipt_sha256": self.completion_receipt_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "selection_fingerprint": self.selection_fingerprint,
            "source_fingerprint": self.source_fingerprint,
        }


@dataclass(frozen=True)
class ValidatedSeedRegistry:
    """Canonical, fully validated contents of one full-pipeline seed registry."""

    common_config: dict[str, Any]
    protocols: dict[str, dict[str, Any]]
    base_run_identity: BaseRunIdentity


@dataclass(frozen=True)
class LoadedSeedRegistry:
    """A validated single-seed registry bound to its exact input bytes."""

    path: Path
    sha256: str
    payload: dict[str, Any]
    validated: ValidatedSeedRegistry

    def verify_unchanged(self) -> None:
        try:
            current = self.path.read_bytes()
        except OSError as error:
            raise RuntimeError(
                f"Stage-A seed registry disappeared during use: {self.path}"
            ) from error
        if hashlib.sha256(current).hexdigest() != self.sha256:
            raise RuntimeError(
                f"Stage-A seed registry changed during use: {self.path}"
            )


@dataclass(frozen=True)
class SeedRegistryBinding:
    """The identities frozen for one seed inside a master registry."""

    registry_sha256: str
    global_seed: int
    base_fingerprint: str
    base_state_fingerprint: str
    base_run_identity: BaseRunIdentity
    stage_a_complete_fingerprint: str

    def to_registry_dict(self) -> dict[str, Any]:
        return {
            "registry_sha256": self.registry_sha256,
            "global_seed": self.global_seed,
            "base_fingerprint": self.base_fingerprint,
            "base_state_fingerprint": self.base_state_fingerprint,
            "base_run_identity": self.base_run_identity.to_registry_dict(),
            "stage_a_complete_fingerprint": self.stage_a_complete_fingerprint,
        }


@dataclass(frozen=True)
class ValidatedMasterRegistry:
    """Canonical contents of the >=5-seed pre-D_T master registry."""

    cross_seed_fixed_config: dict[str, Any]
    efficiency_protocol: dict[str, Any]
    seed_registries: dict[str, SeedRegistryBinding]

    def require_seed_binding(
        self,
        seed_id: str,
        *,
        registry_sha256: str,
        global_seed: int,
        base_fingerprint: str,
        base_state_fingerprint: str,
        base_run_identity: BaseRunIdentity | Mapping[str, Any],
        stage_a_complete_fingerprint: str,
    ) -> SeedRegistryBinding:
        """Require an exact SHA/base/receipt binding for one named seed."""

        canonical_id = validate_seed_id(seed_id)
        try:
            frozen = self.seed_registries[canonical_id]
        except KeyError as error:
            raise RuntimeError(
                f"Stage-A master registry does not contain seed {canonical_id!r}"
            ) from error
        expected_identity = _coerce_base_run_identity(
            base_run_identity,
            name=f"expected Base run identity for {canonical_id!r}",
        )
        expected = SeedRegistryBinding(
            registry_sha256=_digest(
                registry_sha256,
                name=f"expected registry SHA256 for {canonical_id!r}",
            ),
            global_seed=_integer(
                global_seed, name=f"expected global_seed for {canonical_id!r}"
            ),
            base_fingerprint=_digest(
                base_fingerprint,
                name=f"expected base_fingerprint for {canonical_id!r}",
            ),
            base_state_fingerprint=_digest(
                base_state_fingerprint,
                name=f"expected base_state_fingerprint for {canonical_id!r}",
            ),
            base_run_identity=expected_identity,
            stage_a_complete_fingerprint=_digest(
                stage_a_complete_fingerprint,
                name=(
                    f"expected stage_a_complete_fingerprint for {canonical_id!r}"
                ),
            ),
        )
        if frozen != expected:
            for field in (
                "registry_sha256",
                "global_seed",
                "base_fingerprint",
                "base_state_fingerprint",
                "base_run_identity",
                "stage_a_complete_fingerprint",
            ):
                if getattr(frozen, field) != getattr(expected, field):
                    raise RuntimeError(
                        f"Stage-A master seed {canonical_id!r} mismatch for {field}"
                    )
            raise RuntimeError(f"Stage-A master seed {canonical_id!r} binding differs")
        return frozen

    def require_seed_registry(
        self,
        seed_id: str,
        seed_registry: LoadedSeedRegistry,
    ) -> SeedRegistryBinding:
        """Bind a loaded single-seed artifact to its named master entry."""

        if not isinstance(seed_registry, LoadedSeedRegistry):
            raise TypeError("seed_registry must be a LoadedSeedRegistry")
        seed_registry.verify_unchanged()
        common = seed_registry.validated.common_config
        for field in CROSS_SEED_FIXED_FIELDS:
            if not _json_equal(
                common[field], self.cross_seed_fixed_config[field]
            ):
                raise RuntimeError(
                    f"Stage-A seed registry {seed_id!r} differs from master for "
                    f"cross-seed fixed field {field}"
                )
        return self.require_seed_binding(
            seed_id,
            registry_sha256=seed_registry.sha256,
            global_seed=common["global_seed"],
            base_fingerprint=common["base_fingerprint"],
            base_state_fingerprint=common["base_state_fingerprint"],
            base_run_identity=seed_registry.validated.base_run_identity,
            stage_a_complete_fingerprint=common["stage_a_complete_fingerprint"],
        )


@dataclass(frozen=True)
class LoadedMasterRegistry:
    """A validated master registry bound to its exact JSON bytes."""

    path: Path
    sha256: str
    payload: dict[str, Any]
    validated: ValidatedMasterRegistry

    def verify_unchanged(self) -> None:
        try:
            current = self.path.read_bytes()
        except OSError as error:
            raise RuntimeError(
                f"Stage-A master registry disappeared during use: {self.path}"
            ) from error
        if hashlib.sha256(current).hexdigest() != self.sha256:
            raise RuntimeError(
                f"Stage-A master registry changed during use: {self.path}"
            )

    def require_seed_registry(
        self,
        seed_id: str,
        seed_registry: LoadedSeedRegistry,
    ) -> SeedRegistryBinding:
        self.verify_unchanged()
        return self.validated.require_seed_registry(seed_id, seed_registry)

    def require_seed_binding(
        self,
        seed_id: str,
        *,
        registry_sha256: str,
        global_seed: int,
        base_fingerprint: str,
        base_state_fingerprint: str,
        base_run_identity: BaseRunIdentity | Mapping[str, Any],
        stage_a_complete_fingerprint: str,
    ) -> SeedRegistryBinding:
        self.verify_unchanged()
        return self.validated.require_seed_binding(
            seed_id,
            registry_sha256=registry_sha256,
            global_seed=global_seed,
            base_fingerprint=base_fingerprint,
            base_state_fingerprint=base_state_fingerprint,
            base_run_identity=base_run_identity,
            stage_a_complete_fingerprint=stage_a_complete_fingerprint,
        )


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: tuple[str, ...] | set[str], *, name: str) -> None:
    expected_set = set(expected)
    actual = set(value)
    if actual != expected_set:
        missing = sorted(expected_set - actual, key=str)
        extra = sorted(actual - expected_set, key=str)
        raise RuntimeError(f"{name} fields differ; missing={missing}, extra={extra}")


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _coerce_base_run_identity(
    value: BaseRunIdentity | Mapping[str, Any],
    *,
    name: str,
) -> BaseRunIdentity:
    if isinstance(value, BaseRunIdentity):
        # The dataclass constructor is public, so validate even an existing
        # instance rather than assuming its strings are well formed.
        mapping: Mapping[str, Any] = value.to_registry_dict()
    else:
        mapping = _mapping(value, name=name)
    _exact_keys(mapping, BASE_RUN_IDENTITY_FIELDS, name=name)
    producer_schema = mapping["producer_schema"]
    if not isinstance(producer_schema, str) or not producer_schema:
        raise ValueError(f"{name}.producer_schema must be non-empty text")
    return BaseRunIdentity(
        producer_schema=producer_schema,
        base_fingerprint=_digest(
            mapping["base_fingerprint"], name=f"{name}.base_fingerprint"
        ),
        base_state_fingerprint=_digest(
            mapping["base_state_fingerprint"],
            name=f"{name}.base_state_fingerprint",
        ),
        training_run_fingerprint=_digest(
            mapping["training_run_fingerprint"],
            name=f"{name}.training_run_fingerprint",
        ),
        completion_receipt_sha256=_digest(
            mapping["completion_receipt_sha256"],
            name=f"{name}.completion_receipt_sha256",
        ),
        checkpoint_sha256=_digest(
            mapping["checkpoint_sha256"], name=f"{name}.checkpoint_sha256"
        ),
        selection_fingerprint=_digest(
            mapping["selection_fingerprint"],
            name=f"{name}.selection_fingerprint",
        ),
        source_fingerprint=_digest(
            mapping["source_fingerprint"], name=f"{name}.source_fingerprint"
        ),
    )


def _float(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    # Requiring the JSON float representation avoids accepting type aliases
    # (for example 1 in one seed and 1.0 in another) in a canonical registry.
    if type(value) is not float:
        raise TypeError(f"{name} must be a canonical JSON float")
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a canonical JSON integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _json_equal(left: object, right: object) -> bool:
    """Type-sensitive equality for canonical JSON values."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(  # type: ignore[arg-type]
            _json_equal(left[key], right[key]) for key in left  # type: ignore[index]
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(  # type: ignore[arg-type]
            _json_equal(a, b) for a, b in zip(left, right, strict=True)  # type: ignore[arg-type]
        )
    return bool(left == right)


_Config = TypeVar("_Config")


def _canonical_config(
    value: object,
    config_type: Callable[..., _Config],
    *,
    name: str,
) -> dict[str, Any]:
    payload = _mapping(value, name=name)
    try:
        canonical = config_to_dict(config_type(**payload))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is invalid: {error}") from error
    if not _json_equal(payload, canonical):
        raise RuntimeError(f"{name} is not canonical")
    return payload


def _canonical_preprocessing(value: object, *, name: str) -> dict[str, Any]:
    """Validate the exact cache-facing ``PreprocessConfig`` representation."""

    payload = _mapping(value, name=name)
    _exact_keys(
        payload,
        {
            "height",
            "width",
            "color_mode",
            "mean",
            "std",
            "image_interpolation",
            "mask_interpolation",
            "range",
        },
        name=name,
    )
    height = _integer(payload["height"], name=f"{name}.height", minimum=1)
    width = _integer(payload["width"], name=f"{name}.width", minimum=1)
    mean_payload = payload["mean"]
    std_payload = payload["std"]
    if not isinstance(mean_payload, list) or not mean_payload:
        raise TypeError(f"{name}.mean must be a non-empty JSON array")
    if not isinstance(std_payload, list) or not std_payload:
        raise TypeError(f"{name}.std must be a non-empty JSON array")
    mean = tuple(
        _float(item, name=f"{name}.mean[{index}]")
        for index, item in enumerate(mean_payload)
    )
    std = tuple(
        _float(item, name=f"{name}.std[{index}]", minimum=0.0)
        for index, item in enumerate(std_payload)
    )
    if any(item == 0.0 for item in std):
        raise ValueError(f"{name}.std values must be positive")
    try:
        config = PreprocessConfig(
            height=height,
            width=width,
            color_mode=payload["color_mode"],
            mean=mean,
            std=std,
            image_interpolation=payload["image_interpolation"],
            mask_interpolation=payload["mask_interpolation"],
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is invalid: {error}") from error
    canonical = config.fingerprint_payload()
    if not _json_equal(payload, canonical):
        raise RuntimeError(f"{name} is not canonical")
    return payload


def threshold_grid_fingerprint(values: list[float] | tuple[float, ...]) -> str:
    """Fingerprint a canonical threshold grid using the calibration encoding."""

    encoded = json.dumps(
        tuple(values), separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _threshold_grid(
    common: Mapping[str, Any],
    key: str,
) -> list[float]:
    value = common.get(key)
    if not isinstance(value, list) or not value:
        raise TypeError(f"common_config.{key} must be a non-empty JSON array")
    grid = [
        _float(item, name=f"common_config.{key}[{index}]", minimum=0.0, maximum=1.0)
        for index, item in enumerate(value)
    ]
    if grid != sorted(set(grid)):
        raise RuntimeError(f"common_config.{key} must be strictly increasing and unique")
    fingerprint_key = f"{key}_fingerprint"
    expected = threshold_grid_fingerprint(grid)
    actual = _digest(common.get(fingerprint_key), name=f"common_config.{fingerprint_key}")
    if actual != expected:
        raise RuntimeError(f"common_config.{fingerprint_key} does not bind {key}")
    return grid


def _validate_common(value: object) -> dict[str, Any]:
    common = _mapping(value, name="common_config")
    _exact_keys(common, COMMON_FIELDS, name="common_config")

    for key in (
        "manifest_fingerprint",
        "base_fingerprint",
        "base_state_fingerprint",
        "stage_a_complete_fingerprint",
        "d_v_image_fingerprint",
        "d_v_gt_fingerprint",
        "d_v_base_cache_index_fingerprint",
        "d_v_base_cache_index_sha256",
        "d_r_base_cache_index_fingerprint",
        "d_r_base_cache_index_sha256",
        "d_r_state_cache_index_fingerprint",
        "d_r_state_cache_index_sha256",
        "anchor_protocol_sha256",
        "state_fingerprint",
        "efficiency_static_fingerprint",
        "efficiency_receipt_fingerprint",
    ):
        _digest(common[key], name=f"common_config.{key}")

    base_run_identity = _coerce_base_run_identity(
        common["base_run_identity"], name="common_config.base_run_identity"
    )
    if base_run_identity.base_fingerprint != common["base_fingerprint"]:
        raise RuntimeError(
            "common_config Base run identity differs from base_fingerprint"
        )
    if base_run_identity.base_state_fingerprint != common["base_state_fingerprint"]:
        raise RuntimeError(
            "common_config Base run identity differs from base_state_fingerprint"
        )

    tau_o = _float(common["tau_o"], name="common_config.tau_o", minimum=0.0, maximum=1.0)
    tau_b = _float(common["tau_B"], name="common_config.tau_B", minimum=0.0, maximum=1.0)
    if tau_b > tau_o:
        raise RuntimeError(
            "common_config.tau_B must not exceed tau_o for the Base@B control"
        )
    _float(common["pixel_fa_budget"], name="common_config.pixel_fa_budget", minimum=0.0)
    component_budget = common["component_fa_per_mp_budget"]
    if component_budget is not None:
        _float(
            component_budget,
            name="common_config.component_fa_per_mp_budget",
            minimum=0.0,
        )
    raw_background_budget = common["raw_background_fa_budget"]
    if raw_background_budget is not None:
        _float(
            raw_background_budget,
            name="common_config.raw_background_fa_budget",
            minimum=0.0,
        )
    _float(
        common["minimum_retention"],
        name="common_config.minimum_retention",
        minimum=0.0,
        maximum=1.0,
    )
    if common["null_residual_candidate"] is not True:
        raise RuntimeError(
            "common_config.null_residual_candidate must be the canonical true flag"
        )
    if common["threshold_selection_rule"] != THRESHOLD_SELECTION_RULE:
        raise RuntimeError(
            "common_config.threshold_selection_rule differs from the "
            "implemented calibration rule"
        )

    anchor_grid = _threshold_grid(common, "anchor_threshold_grid")
    residual_grid = _threshold_grid(common, "residual_threshold_grid")
    base_grid = _threshold_grid(common, "base_threshold_grid")
    if tau_o not in anchor_grid:
        raise RuntimeError("common_config.tau_o is absent from anchor_threshold_grid")
    if tau_o not in base_grid:
        raise RuntimeError("base_threshold_grid must include the anchor threshold tau_o")
    if tau_b not in base_grid:
        raise RuntimeError("common_config.tau_B is absent from base_threshold_grid")

    occupancy = _canonical_config(
        common["occupancy_config"], OccupancyConfig, name="common_config.occupancy_config"
    )
    if occupancy["threshold"] != tau_o:
        raise RuntimeError("common_config occupancy threshold differs from tau_o")
    _canonical_config(
        common["matching_config"], MatchConfig, name="common_config.matching_config"
    )
    _canonical_config(
        common["intervention_config"],
        InterventionConfig,
        name="common_config.intervention_config",
    )
    _canonical_preprocessing(
        common["preprocessing"], name="common_config.preprocessing"
    )
    _canonical_config(
        common["decoder_config"], DecoderConfig, name="common_config.decoder_config"
    )
    _canonical_config(common["loss_config"], LossConfig, name="common_config.loss_config")
    _canonical_config(
        common["training_config"], TrainingConfig, name="common_config.training_config"
    )

    optimization = _mapping(
        common["optimization_config"], name="common_config.optimization_config"
    )
    _exact_keys(
        optimization,
        {"optimizer", "learning_rate", "weight_decay"},
        name="common_config.optimization_config",
    )
    if optimization["optimizer"] not in {"adam", "sgd"}:
        raise ValueError("common_config.optimization_config.optimizer must be adam or sgd")
    _float(
        optimization["learning_rate"],
        name="common_config.optimization_config.learning_rate",
        minimum=0.0,
    )
    if optimization["learning_rate"] == 0.0:
        raise ValueError("common_config optimization learning_rate must be positive")
    _float(
        optimization["weight_decay"],
        name="common_config.optimization_config.weight_decay",
        minimum=0.0,
    )

    branch_sizes = _mapping(
        common["branch_batch_sizes"], name="common_config.branch_batch_sizes"
    )
    _exact_keys(
        branch_sizes,
        {"factual_miss", "factual_no_miss", "synthetic"},
        name="common_config.branch_batch_sizes",
    )
    for key in ("factual_miss", "factual_no_miss", "synthetic"):
        _integer(
            branch_sizes[key],
            name=f"common_config.branch_batch_sizes.{key}",
            minimum=1,
        )
    if common["data_augmentation"] != "none_frozen_base_cache":
        raise RuntimeError("common_config.data_augmentation is not the frozen policy")

    global_seed = _integer(common["global_seed"], name="common_config.global_seed")
    del global_seed
    steps = _integer(
        common["steps_per_epoch"], name="common_config.steps_per_epoch", minimum=1
    )
    epochs = _integer(
        common["trained_epochs"], name="common_config.trained_epochs", minimum=1
    )
    stopping = _mapping(
        common["fixed_stopping_rule"], name="common_config.fixed_stopping_rule"
    )
    _exact_keys(
        stopping,
        {"epochs", "steps_per_epoch"},
        name="common_config.fixed_stopping_rule",
    )
    _integer(stopping["epochs"], name="common_config.fixed_stopping_rule.epochs", minimum=1)
    _integer(
        stopping["steps_per_epoch"],
        name="common_config.fixed_stopping_rule.steps_per_epoch",
        minimum=1,
    )
    if stopping != {"epochs": epochs, "steps_per_epoch": steps}:
        raise RuntimeError("common_config fixed stopping rule differs from completed run")

    if common["efficiency_device_type"] not in {"cpu", "cuda"}:
        raise ValueError("common_config.efficiency_device_type must be cpu or cuda")
    _integer(
        common["efficiency_warmup"],
        name="common_config.efficiency_warmup",
        minimum=0,
    )
    _integer(
        common["efficiency_repetitions"],
        name="common_config.efficiency_repetitions",
        minimum=1,
    )

    return deepcopy(common)


def _validate_protocols(
    value: object,
    *,
    residual_grid: list[float],
    null_residual_candidate: bool,
    tau_o: float,
    tau_b: float,
    anchor_protocol_sha256: str,
) -> dict[str, dict[str, Any]]:
    protocols = _mapping(value, name="protocols")
    _exact_keys(protocols, set(STAGE_A_VARIANTS), name="protocols")
    # ``protocol_fingerprint`` binds the decoder-free anchor receipt for A and
    # the inner BoundDVThresholdProtocol for Base@B/F/F×/U. Base@B's outer
    # receipt is also decoder-free, but the inner protocol fingerprint is the
    # shared registry-level threshold identity for all four calibrated arms.
    protocol_fingerprints: set[str] = set()
    decoder_artifact_fingerprints: set[str] = set()
    result: dict[str, dict[str, Any]] = {}
    for method_id, variant in STAGE_A_VARIANTS.items():
        entry = _mapping(protocols[method_id], name=f"protocols.{method_id}")
        _exact_keys(entry, PROTOCOL_ENTRY_FIELDS, name=f"protocols.{method_id}")
        expected_mode = STAGE_A_EVALUATION_MODES[method_id]
        if entry["evaluation_mode"] != expected_mode:
            raise RuntimeError(
                f"protocols.{method_id}.evaluation_mode must be {expected_mode!r}"
            )
        if entry["decoder_variant"] != variant:
            raise RuntimeError(
                f"protocols.{method_id}.decoder_variant must be {variant!r}"
            )
        protocol_fingerprint = _digest(
            entry["protocol_fingerprint"],
            name=f"protocols.{method_id}.protocol_fingerprint",
        )
        if protocol_fingerprint in protocol_fingerprints:
            raise RuntimeError(
                "protocol fingerprints must be unique across A/Base@B/F/F×/U"
            )
        protocol_fingerprints.add(protocol_fingerprint)

        artifact_fingerprint = entry["decoder_artifact_fingerprint"]
        if variant is None:
            if artifact_fingerprint is not None:
                raise RuntimeError(
                    f"protocols.{method_id} is decoder-free and must not bind "
                    "a decoder artifact"
                )
        else:
            artifact_fingerprint = _digest(
                artifact_fingerprint,
                name=f"protocols.{method_id}.decoder_artifact_fingerprint",
            )
            if artifact_fingerprint in decoder_artifact_fingerprints:
                raise RuntimeError(
                    "decoder artifact fingerprints must be unique across F/F×/U"
                )
            decoder_artifact_fingerprints.add(artifact_fingerprint)

        selected_threshold = entry["selected_threshold"]
        if method_id == "A":
            threshold = _float(
                selected_threshold,
                name="protocols.A.selected_threshold",
                minimum=0.0,
                maximum=1.0,
            )
            if threshold != tau_o:
                raise RuntimeError(
                    "protocols.A.selected_threshold must equal common_config.tau_o"
                )
            if protocol_fingerprint != anchor_protocol_sha256:
                raise RuntimeError(
                    "protocols.A.protocol_fingerprint must equal "
                    "common_config.anchor_protocol_sha256"
                )
        elif method_id == "Base@B":
            threshold = _float(
                selected_threshold,
                name="protocols.Base@B.selected_threshold",
                minimum=0.0,
                maximum=1.0,
            )
            if threshold != tau_b:
                raise RuntimeError(
                    "protocols.Base@B.selected_threshold must equal "
                    "common_config.tau_B"
                )
        elif selected_threshold is None:
            if not null_residual_candidate:
                raise RuntimeError(
                    f"protocols.{method_id}.selected_threshold selects an "
                    "unregistered null residual"
                )
        else:
            tau_r = _float(
                selected_threshold,
                name=f"protocols.{method_id}.selected_threshold",
                minimum=0.0,
                maximum=1.0,
            )
            if tau_r not in residual_grid:
                raise RuntimeError(
                    f"protocols.{method_id}.selected_threshold is absent from "
                    "residual_threshold_grid"
                )
        result[method_id] = deepcopy(entry)
    return result


def validate_seed_registry_mapping(
    payload: object,
    *,
    source: str = "Stage-A seed registry",
) -> ValidatedSeedRegistry:
    """Validate one canonical, frozen A/Base@B/F/F×/U registry.

    Unknown fields, numeric type aliases, non-canonical configuration objects,
    unbound threshold grids, and incomplete run identities are all rejected.
    """

    registry = _mapping(payload, name=source)
    _exact_keys(registry, _TOP_LEVEL_FIELDS, name=source)
    expected = {
        "schema_version": SEED_REGISTRY_SCHEMA_VERSION,
        "artifact_type": "stage_a_frozen_registry",
        "method_version": METHOD_VERSION,
        "stage": "Stage A",
        "split": "D_V",
        "thresholds_frozen": True,
    }
    for key, value in expected.items():
        if not _json_equal(registry[key], value):
            raise RuntimeError(f"{source} mismatch for {key}")

    common = _validate_common(registry["common_config"])
    protocols = _validate_protocols(
        registry["protocols"],
        residual_grid=common["residual_threshold_grid"],
        null_residual_candidate=common["null_residual_candidate"],
        tau_o=common["tau_o"],
        tau_b=common["tau_B"],
        anchor_protocol_sha256=common["anchor_protocol_sha256"],
    )
    identity = _coerce_base_run_identity(
        common["base_run_identity"], name="common_config.base_run_identity"
    )
    return ValidatedSeedRegistry(common, protocols, identity)


def validate_seed_registry(
    payload: object,
    *,
    source: str = "Stage-A seed registry",
) -> ValidatedSeedRegistry:
    """Backward-compatible concise name for ``validate_seed_registry_mapping``."""

    return validate_seed_registry_mapping(payload, source=source)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def load_seed_registry(path: str | Path) -> LoadedSeedRegistry:
    """Read and validate a single-seed registry while preserving its byte SHA."""

    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise FileNotFoundError(f"cannot resolve Stage-A seed registry {candidate}") from error
    if not resolved.is_file():
        raise ValueError(f"Stage-A seed registry is not a regular file: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Stage-A seed registry is not valid UTF-8 JSON: {resolved}") from error
    validated = validate_seed_registry_mapping(
        payload, source=f"Stage-A seed registry {resolved}"
    )
    return LoadedSeedRegistry(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        payload=deepcopy(payload),
        validated=validated,
    )


def _validate_master_fixed_config(value: object) -> dict[str, Any]:
    fixed = _mapping(value, name="cross_seed_fixed_config")
    _exact_keys(fixed, CROSS_SEED_FIXED_FIELDS, name="cross_seed_fixed_config")
    for key in (
        "manifest_fingerprint",
        "d_v_image_fingerprint",
        "d_v_gt_fingerprint",
    ):
        _digest(fixed[key], name=f"cross_seed_fixed_config.{key}")
    _float(
        fixed["pixel_fa_budget"],
        name="cross_seed_fixed_config.pixel_fa_budget",
        minimum=0.0,
    )
    if fixed["component_fa_per_mp_budget"] is not None:
        _float(
            fixed["component_fa_per_mp_budget"],
            name="cross_seed_fixed_config.component_fa_per_mp_budget",
            minimum=0.0,
        )
    if fixed["raw_background_fa_budget"] is not None:
        _float(
            fixed["raw_background_fa_budget"],
            name="cross_seed_fixed_config.raw_background_fa_budget",
            minimum=0.0,
        )
    _float(
        fixed["minimum_retention"],
        name="cross_seed_fixed_config.minimum_retention",
        minimum=0.0,
        maximum=1.0,
    )
    if fixed["null_residual_candidate"] is not True:
        raise RuntimeError(
            "cross_seed_fixed_config.null_residual_candidate must be true"
        )
    if fixed["threshold_selection_rule"] != THRESHOLD_SELECTION_RULE:
        raise RuntimeError(
            "cross_seed_fixed_config.threshold_selection_rule differs from the "
            "implemented calibration rule"
        )

    _canonical_config(
        fixed["matching_config"],
        MatchConfig,
        name="cross_seed_fixed_config.matching_config",
    )
    _canonical_config(
        fixed["intervention_config"],
        InterventionConfig,
        name="cross_seed_fixed_config.intervention_config",
    )
    _canonical_preprocessing(
        fixed["preprocessing"],
        name="cross_seed_fixed_config.preprocessing",
    )
    _canonical_config(
        fixed["decoder_config"],
        DecoderConfig,
        name="cross_seed_fixed_config.decoder_config",
    )
    _canonical_config(
        fixed["loss_config"],
        LossConfig,
        name="cross_seed_fixed_config.loss_config",
    )
    _canonical_config(
        fixed["training_config"],
        TrainingConfig,
        name="cross_seed_fixed_config.training_config",
    )
    optimization = _mapping(
        fixed["optimization_config"],
        name="cross_seed_fixed_config.optimization_config",
    )
    _exact_keys(
        optimization,
        {"optimizer", "learning_rate", "weight_decay"},
        name="cross_seed_fixed_config.optimization_config",
    )
    if optimization["optimizer"] not in {"adam", "sgd"}:
        raise ValueError(
            "cross_seed_fixed_config.optimization_config.optimizer must be adam or sgd"
        )
    learning_rate = _float(
        optimization["learning_rate"],
        name="cross_seed_fixed_config.optimization_config.learning_rate",
        minimum=0.0,
    )
    if learning_rate == 0.0:
        raise ValueError("cross-seed fixed optimization learning_rate must be positive")
    _float(
        optimization["weight_decay"],
        name="cross_seed_fixed_config.optimization_config.weight_decay",
        minimum=0.0,
    )

    branches = _mapping(
        fixed["branch_batch_sizes"],
        name="cross_seed_fixed_config.branch_batch_sizes",
    )
    _exact_keys(
        branches,
        {"factual_miss", "factual_no_miss", "synthetic"},
        name="cross_seed_fixed_config.branch_batch_sizes",
    )
    for key in ("factual_miss", "factual_no_miss", "synthetic"):
        _integer(
            branches[key],
            name=f"cross_seed_fixed_config.branch_batch_sizes.{key}",
            minimum=1,
        )
    if fixed["data_augmentation"] != "none_frozen_base_cache":
        raise RuntimeError(
            "cross_seed_fixed_config.data_augmentation is not the frozen policy"
        )

    steps = _integer(
        fixed["steps_per_epoch"],
        name="cross_seed_fixed_config.steps_per_epoch",
        minimum=1,
    )
    epochs = _integer(
        fixed["trained_epochs"],
        name="cross_seed_fixed_config.trained_epochs",
        minimum=1,
    )
    stopping = _mapping(
        fixed["fixed_stopping_rule"],
        name="cross_seed_fixed_config.fixed_stopping_rule",
    )
    _exact_keys(
        stopping,
        {"epochs", "steps_per_epoch"},
        name="cross_seed_fixed_config.fixed_stopping_rule",
    )
    _integer(
        stopping["epochs"],
        name="cross_seed_fixed_config.fixed_stopping_rule.epochs",
        minimum=1,
    )
    _integer(
        stopping["steps_per_epoch"],
        name="cross_seed_fixed_config.fixed_stopping_rule.steps_per_epoch",
        minimum=1,
    )
    if stopping != {"epochs": epochs, "steps_per_epoch": steps}:
        raise RuntimeError(
            "cross_seed_fixed_config stopping rule differs from epochs/steps"
        )

    if fixed["efficiency_device_type"] not in {"cpu", "cuda"}:
        raise ValueError(
            "cross_seed_fixed_config.efficiency_device_type must be cpu or cuda"
        )
    _integer(
        fixed["efficiency_warmup"],
        name="cross_seed_fixed_config.efficiency_warmup",
        minimum=0,
    )
    _integer(
        fixed["efficiency_repetitions"],
        name="cross_seed_fixed_config.efficiency_repetitions",
        minimum=1,
    )

    _threshold_grid(fixed, "anchor_threshold_grid")
    _threshold_grid(fixed, "residual_threshold_grid")
    _threshold_grid(fixed, "base_threshold_grid")
    return deepcopy(fixed)


def _validate_efficiency_protocol(value: object) -> dict[str, Any]:
    protocol = _mapping(value, name="efficiency_protocol")
    _exact_keys(
        protocol,
        {"device_type", "warmup", "repetitions"},
        name="efficiency_protocol",
    )
    if protocol["device_type"] not in {"cpu", "cuda"}:
        raise ValueError(
            "efficiency_protocol.device_type must be exactly cpu or cuda"
        )
    _integer(protocol["warmup"], name="efficiency_protocol.warmup", minimum=0)
    _integer(
        protocol["repetitions"],
        name="efficiency_protocol.repetitions",
        minimum=1,
    )
    return deepcopy(protocol)


def _validate_master_seed_bindings(value: object) -> dict[str, SeedRegistryBinding]:
    entries = _mapping(value, name="seed_registries")
    if len(entries) < MINIMUM_FULL_PIPELINE_SEEDS:
        raise RuntimeError(
            f"seed_registries must contain at least {MINIMUM_FULL_PIPELINE_SEEDS} seeds"
        )
    result: dict[str, SeedRegistryBinding] = {}
    seen_registry_sha: dict[str, str] = {}
    seen_global_seed: dict[int, str] = {}
    seen_base: dict[str, str] = {}
    seen_base_state: dict[str, str] = {}
    seen_stage_a_complete: dict[str, str] = {}
    shared_base_producer: tuple[str, str] | None = None
    seen_identity: dict[str, dict[str, str]] = {
        field: {} for field in BASE_RUN_UNIQUE_IDENTITY_FIELDS
    }
    seed_ids = [validate_seed_id(raw_seed_id) for raw_seed_id in entries]
    for seed_id in sorted(seed_ids):
        entry = _mapping(entries[seed_id], name=f"seed_registries.{seed_id}")
        _exact_keys(entry, _MASTER_SEED_FIELDS, name=f"seed_registries.{seed_id}")
        registry_sha = _digest(
            entry["registry_sha256"],
            name=f"seed_registries.{seed_id}.registry_sha256",
        )
        global_seed = _integer(
            entry["global_seed"], name=f"seed_registries.{seed_id}.global_seed"
        )
        base_fingerprint = _digest(
            entry["base_fingerprint"],
            name=f"seed_registries.{seed_id}.base_fingerprint",
        )
        base_state_fingerprint = _digest(
            entry["base_state_fingerprint"],
            name=f"seed_registries.{seed_id}.base_state_fingerprint",
        )
        stage_a_complete_fingerprint = _digest(
            entry["stage_a_complete_fingerprint"],
            name=f"seed_registries.{seed_id}.stage_a_complete_fingerprint",
        )
        identity = _coerce_base_run_identity(
            entry["base_run_identity"],
            name=f"seed_registries.{seed_id}.base_run_identity",
        )
        if identity.base_fingerprint != base_fingerprint:
            raise RuntimeError(
                f"seed_registries.{seed_id}.base_run_identity base differs"
            )
        if identity.base_state_fingerprint != base_state_fingerprint:
            raise RuntimeError(
                f"seed_registries.{seed_id}.base_run_identity state differs"
            )
        producer = (identity.producer_schema, identity.source_fingerprint)
        if shared_base_producer is None:
            shared_base_producer = producer
        elif producer != shared_base_producer:
            raise RuntimeError(
                "master seeds must share one Base producer schema and source"
            )
        for label, current, seen in (
            ("registry_sha256", registry_sha, seen_registry_sha),
            ("global_seed", global_seed, seen_global_seed),
            ("base_fingerprint", base_fingerprint, seen_base),
            (
                "base_state_fingerprint",
                base_state_fingerprint,
                seen_base_state,
            ),
            (
                "stage_a_complete_fingerprint",
                stage_a_complete_fingerprint,
                seen_stage_a_complete,
            ),
        ):
            previous = seen.get(current)
            if previous is not None:
                raise RuntimeError(
                    f"master {label} must be unique; {previous!r} and {seed_id!r} reuse it"
                )
            seen[current] = seed_id
        identity_mapping = identity.to_registry_dict()
        for field in BASE_RUN_UNIQUE_IDENTITY_FIELDS:
            current = identity_mapping[field]
            previous = seen_identity[field].get(current)
            if previous is not None:
                raise RuntimeError(
                    f"master {field} must be unique; {previous!r} and {seed_id!r} reuse it"
                )
            seen_identity[field][current] = seed_id
        result[seed_id] = SeedRegistryBinding(
            registry_sha256=registry_sha,
            global_seed=global_seed,
            base_fingerprint=base_fingerprint,
            base_state_fingerprint=base_state_fingerprint,
            base_run_identity=identity,
            stage_a_complete_fingerprint=stage_a_complete_fingerprint,
        )
    return result


def validate_master_registry_mapping(
    payload: object,
    *,
    source: str = "Stage-A master registry",
) -> ValidatedMasterRegistry:
    """Strictly validate a canonical pre-D_T master-registry mapping."""

    master = _mapping(payload, name=source)
    _exact_keys(master, _MASTER_TOP_LEVEL_FIELDS, name=source)
    expected = {
        "schema_version": MASTER_REGISTRY_SCHEMA_VERSION,
        "artifact_type": "stage_a_master_registry",
        "method_version": METHOD_VERSION,
        "stage": "Stage A",
        "split": "D_V",
        "thresholds_frozen": True,
        "minimum_full_pipeline_seeds": MINIMUM_FULL_PIPELINE_SEEDS,
    }
    for key, value in expected.items():
        if not _json_equal(master[key], value):
            raise RuntimeError(f"{source} mismatch for {key}")
    seed_count = _integer(master["seed_count"], name=f"{source}.seed_count", minimum=5)
    fixed = _validate_master_fixed_config(master["cross_seed_fixed_config"])
    efficiency = _validate_efficiency_protocol(master["efficiency_protocol"])
    expected_efficiency = {
        "device_type": fixed["efficiency_device_type"],
        "warmup": fixed["efficiency_warmup"],
        "repetitions": fixed["efficiency_repetitions"],
    }
    if not _json_equal(efficiency, expected_efficiency):
        raise RuntimeError(
            f"{source}.efficiency_protocol differs from seed registry receipts"
        )
    seeds = _validate_master_seed_bindings(master["seed_registries"])
    if seed_count != len(seeds):
        raise RuntimeError(f"{source}.seed_count differs from seed_registries")
    return ValidatedMasterRegistry(fixed, efficiency, seeds)


def validate_master_registry(
    payload: object,
    *,
    source: str = "Stage-A master registry",
) -> ValidatedMasterRegistry:
    """Concise alias for ``validate_master_registry_mapping``."""

    return validate_master_registry_mapping(payload, source=source)


def load_master_registry(path: str | Path) -> LoadedMasterRegistry:
    """Load a strict master registry and bind it to its exact file SHA256."""

    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise FileNotFoundError(f"cannot resolve Stage-A master registry {candidate}") from error
    if not resolved.is_file():
        raise ValueError(f"Stage-A master registry is not a regular file: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Stage-A master registry is not valid UTF-8 JSON: {resolved}") from error
    validated = validate_master_registry_mapping(
        payload, source=f"Stage-A master registry {resolved}"
    )
    return LoadedMasterRegistry(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        payload=deepcopy(payload),
        validated=validated,
    )


def validate_seed_id(seed_id: object) -> str:
    """Return a safe, canonical external full-pipeline seed identifier."""

    if not isinstance(seed_id, str) or _SEED_ID.fullmatch(seed_id) is None:
        raise ValueError(
            "seed ID must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
        )
    return seed_id


def _require_unique(
    entries: Mapping[str, LoadedSeedRegistry],
    *,
    label: str,
    value: Callable[[LoadedSeedRegistry], object],
) -> None:
    seen: dict[object, str] = {}
    for seed_id, loaded in entries.items():
        identity = value(loaded)
        previous = seen.get(identity)
        if previous is not None:
            raise RuntimeError(
                f"{label} must be unique across seeds; {previous!r} and {seed_id!r} reuse it"
            )
        seen[identity] = seed_id


def build_master_registry(
    entries: Mapping[str, LoadedSeedRegistry],
) -> dict[str, Any]:
    """Build a master registry and derive its shared efficiency protocol."""

    if len(entries) < MINIMUM_FULL_PIPELINE_SEEDS:
        raise ValueError(
            f"at least {MINIMUM_FULL_PIPELINE_SEEDS} full-pipeline seed registries are required"
        )
    canonical_entries: dict[str, LoadedSeedRegistry] = {}
    for seed_id, loaded in entries.items():
        canonical_id = validate_seed_id(seed_id)
        if not isinstance(loaded, LoadedSeedRegistry):
            raise TypeError(f"seed {canonical_id!r} is not a loaded seed registry")
        if canonical_id in canonical_entries:
            raise RuntimeError(f"duplicate seed ID {canonical_id!r}")
        canonical_entries[canonical_id] = loaded

    _require_unique(canonical_entries, label="registry SHA256", value=lambda item: item.sha256)
    _require_unique(canonical_entries, label="registry path", value=lambda item: item.path)
    for field in UNIQUE_SEED_COMMON_IDENTITY_FIELDS:
        _require_unique(
            canonical_entries,
            label=field,
            value=lambda item, key=field: item.validated.common_config[key],
        )
    for method_id in STAGE_A_VARIANTS:
        _require_unique(
            canonical_entries,
            label=f"protocols.{method_id}.protocol_fingerprint",
            value=lambda item, mid=method_id: (
                item.validated.protocols[mid]["protocol_fingerprint"]
            ),
        )
    for method_id, variant in STAGE_A_VARIANTS.items():
        if variant is None:
            continue
        _require_unique(
            canonical_entries,
            label=f"protocols.{method_id}.decoder_artifact_fingerprint",
            value=lambda item, mid=method_id: (
                item.validated.protocols[mid]["decoder_artifact_fingerprint"]
            ),
        )

    ordered_ids = sorted(canonical_entries)
    reference_id = ordered_ids[0]
    reference = canonical_entries[reference_id].validated
    for seed_id in ordered_ids[1:]:
        candidate = canonical_entries[seed_id].validated
        for field in CROSS_SEED_FIXED_FIELDS:
            if not _json_equal(
                candidate.common_config[field], reference.common_config[field]
            ):
                raise RuntimeError(
                    f"cross-seed fixed field {field} differs: {reference_id!r} != {seed_id!r}"
                )

    fixed = {
        field: deepcopy(reference.common_config[field])
        for field in CROSS_SEED_FIXED_FIELDS
    }
    seed_map: dict[str, Any] = {}
    for seed_id in ordered_ids:
        loaded = canonical_entries[seed_id]
        common = loaded.validated.common_config
        seed_map[seed_id] = {
            "registry_sha256": loaded.sha256,
            "global_seed": common["global_seed"],
            "base_fingerprint": common["base_fingerprint"],
            "base_state_fingerprint": common["base_state_fingerprint"],
            "base_run_identity": loaded.validated.base_run_identity.to_registry_dict(),
            "stage_a_complete_fingerprint": common[
                "stage_a_complete_fingerprint"
            ],
        }

    master = {
        "schema_version": MASTER_REGISTRY_SCHEMA_VERSION,
        "artifact_type": "stage_a_master_registry",
        "method_version": METHOD_VERSION,
        "stage": "Stage A",
        "split": "D_V",
        "thresholds_frozen": True,
        "minimum_full_pipeline_seeds": MINIMUM_FULL_PIPELINE_SEEDS,
        "seed_count": len(seed_map),
        "cross_seed_fixed_config": fixed,
        "efficiency_protocol": {
            "device_type": reference.common_config["efficiency_device_type"],
            "warmup": reference.common_config["efficiency_warmup"],
            "repetitions": reference.common_config["efficiency_repetitions"],
        },
        "seed_registries": seed_map,
    }
    # Keep the producer and the reusable consumer contract inseparable.
    validate_master_registry_mapping(master)
    return master


__all__ = [
    "BASE_RUN_IDENTITY_FIELDS",
    "BASE_RUN_UNIQUE_IDENTITY_FIELDS",
    "COMMON_FIELDS",
    "CROSS_SEED_FIXED_FIELDS",
    "CROSS_SEED_VARIABLE_FIELDS",
    "MASTER_REGISTRY_SCHEMA_VERSION",
    "METHOD_VERSION",
    "MINIMUM_FULL_PIPELINE_SEEDS",
    "PROTOCOL_ENTRY_FIELDS",
    "SEED_REGISTRY_SCHEMA_VERSION",
    "STAGE_A_DECODER_VARIANTS",
    "STAGE_A_EVALUATION_MODES",
    "STAGE_A_METHOD_ORDER",
    "STAGE_A_VARIANTS",
    "UNIQUE_SEED_COMMON_IDENTITY_FIELDS",
    "BaseRunIdentity",
    "LoadedSeedRegistry",
    "LoadedMasterRegistry",
    "SeedRegistryBinding",
    "ValidatedSeedRegistry",
    "ValidatedMasterRegistry",
    "build_master_registry",
    "load_master_registry",
    "load_seed_registry",
    "threshold_grid_fingerprint",
    "validate_master_registry",
    "validate_master_registry_mapping",
    "validate_seed_id",
    "validate_seed_registry",
    "validate_seed_registry_mapping",
]
