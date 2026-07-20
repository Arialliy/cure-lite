"""Strict provenance contract for a base detector developed only on ``D_B``.

The CURE-Lite evidence protocol is invalid when the frozen base checkpoint has
used ``D_R``, ``D_V``, or ``D_T`` for fitting or checkpoint selection.  This
module records the exact ``D_B`` membership next to the checkpoint and, for the
formal evidence path, binds a group-disjoint ``D_B-fit``/``D_B-select``
partition before downstream caches are trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
import hmac
import json
from pathlib import Path
from typing import Any, Mapping

from .cache.schema import file_sha256, stable_fingerprint
from .splits import SplitManifest


BASE_TRAINING_PROVENANCE_SCHEMA = "cure-lite-base-training-provenance-v1"
BASE_CHECKPOINT_SELECTION_SCHEMA = "cure-lite-base-checkpoint-selection-v1"
CURE_LITE_METHOD_VERSION = "cure-lite-v0.1"


class BaseTrainingProvenanceError(ValueError):
    """Raised when base-training provenance cannot satisfy the formal protocol."""


LEGACY_FORMAL_BASE_PREFLIGHT_SCHEMA = "cure-lite-mshnet-base-preflight-v1"
LEGACY_FORMAL_BASE_FINAL_SCHEMA = "cure-lite-mshnet-base-final-receipt-v1"
FORMAL_BASE_PREFLIGHT_SCHEMA = "cure-lite-mshnet-base-preflight-v2"
FORMAL_BASE_FINAL_SCHEMA = "cure-lite-mshnet-base-final-receipt-v2"


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise BaseTrainingProvenanceError(f"{name} must be a SHA256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise BaseTrainingProvenanceError(
            f"{name} must be a 64-character hexadecimal SHA256 digest"
        )
    return normalized


@dataclass(frozen=True, order=True)
class BaseTrainingSample:
    """One exact sample-to-scene membership declared by base training."""

    sample_id: str
    scene_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise BaseTrainingProvenanceError("D_B sample_id must be non-empty")
        if self.scene_id is not None and (
            not isinstance(self.scene_id, str) or not self.scene_id
        ):
            raise BaseTrainingProvenanceError(
                "D_B scene_id must be null or a non-empty string"
            )

    def canonical_payload(self) -> dict[str, str | None]:
        return {"sample_id": self.sample_id, "scene_id": self.scene_id}


def _manifest_d_b_samples(manifest: SplitManifest) -> tuple[BaseTrainingSample, ...]:
    if not isinstance(manifest, SplitManifest):
        raise TypeError("manifest must be a SplitManifest")
    records = manifest.records_for("D_B")
    manifest.assert_purpose("base_train", records)
    if not records:
        raise BaseTrainingProvenanceError("the split manifest has an empty D_B")
    return tuple(
        sorted(
            BaseTrainingSample(record.sample_id, record.scene_id)
            for record in records
        )
    )


@dataclass(frozen=True)
class BaseCheckpointSelection:
    """Canonical group-disjoint partition used to select a base checkpoint.

    This contract is deliberately separate from :class:`BaseTrainingProvenance`.
    The latter remains a generic declaration that the resulting checkpoint used
    only ``D_B`` samples; the formal launcher additionally has to prove which
    ``D_B`` rows were used for parameter fitting and which were used solely for
    checkpoint selection.
    """

    split_manifest_fingerprint: str
    fit_sample_ids: tuple[str, ...]
    select_sample_ids: tuple[str, ...]
    source_split: str = "D_B"
    fit_role: str = "D_B-fit"
    select_role: str = "D_B-select"
    schema_version: str = BASE_CHECKPOINT_SELECTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != BASE_CHECKPOINT_SELECTION_SCHEMA:
            raise BaseTrainingProvenanceError(
                f"unsupported checkpoint-selection schema {self.schema_version!r}"
            )
        if self.source_split != "D_B":
            raise BaseTrainingProvenanceError(
                "checkpoint-selection source_split must be exactly 'D_B'"
            )
        if self.fit_role != "D_B-fit" or self.select_role != "D_B-select":
            raise BaseTrainingProvenanceError(
                "checkpoint-selection roles must be D_B-fit and D_B-select"
            )
        object.__setattr__(
            self,
            "split_manifest_fingerprint",
            _sha256(
                self.split_manifest_fingerprint,
                name="checkpoint_selection.split_manifest_fingerprint",
            ),
        )
        for name, values in (
            ("fit_sample_ids", self.fit_sample_ids),
            ("select_sample_ids", self.select_sample_ids),
        ):
            if not isinstance(values, tuple):
                raise BaseTrainingProvenanceError(f"{name} must be a tuple")
            if not values:
                raise BaseTrainingProvenanceError(f"{name} cannot be empty")
            if any(not isinstance(value, str) or not value for value in values):
                raise BaseTrainingProvenanceError(
                    f"{name} must contain non-empty sample IDs"
                )
            if values != tuple(sorted(set(values))):
                raise BaseTrainingProvenanceError(
                    f"{name} must be sorted and contain unique sample IDs"
                )
        if set(self.fit_sample_ids) & set(self.select_sample_ids):
            raise BaseTrainingProvenanceError(
                "D_B-fit and D_B-select sample IDs must be disjoint"
            )

    @classmethod
    def from_manifest(
        cls,
        manifest: SplitManifest,
        *,
        fit_sample_ids: tuple[str, ...] | list[str],
        select_sample_ids: tuple[str, ...] | list[str],
    ) -> "BaseCheckpointSelection":
        """Build and validate a complete partition of manifest ``D_B``."""

        selection = cls(
            split_manifest_fingerprint=manifest.fingerprint,
            fit_sample_ids=tuple(sorted(fit_sample_ids)),
            select_sample_ids=tuple(sorted(select_sample_ids)),
        )
        selection.validate_against(manifest)
        return selection

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BaseCheckpointSelection":
        if not isinstance(value, Mapping):
            raise TypeError("checkpoint selection must be a mapping")
        expected_keys = {
            "schema_version",
            "split_manifest_fingerprint",
            "source_split",
            "fit_role",
            "select_role",
            "fit_sample_ids",
            "select_sample_ids",
            "selection_fingerprint",
        }
        if set(value) != expected_keys:
            missing = sorted(expected_keys - set(value))
            unknown = sorted(set(value) - expected_keys)
            raise BaseTrainingProvenanceError(
                "invalid checkpoint-selection fields; "
                f"missing={missing}, unknown={unknown}"
            )
        fit_ids = value["fit_sample_ids"]
        select_ids = value["select_sample_ids"]
        if not isinstance(fit_ids, list) or not isinstance(select_ids, list):
            raise BaseTrainingProvenanceError(
                "persisted checkpoint-selection sample IDs must be lists"
            )
        selection = cls(
            split_manifest_fingerprint=value["split_manifest_fingerprint"],
            fit_sample_ids=tuple(fit_ids),
            select_sample_ids=tuple(select_ids),
            source_split=value["source_split"],
            fit_role=value["fit_role"],
            select_role=value["select_role"],
            schema_version=value["schema_version"],
        )
        declared = _sha256(
            value["selection_fingerprint"], name="selection_fingerprint"
        )
        if not hmac.compare_digest(declared, selection.fingerprint):
            raise BaseTrainingProvenanceError(
                "checkpoint-selection fingerprint does not match its contents"
            )
        return selection

    def canonical_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "split_manifest_fingerprint": self.split_manifest_fingerprint,
            "source_split": self.source_split,
            "fit_role": self.fit_role,
            "select_role": self.select_role,
            "fit_sample_ids": list(self.fit_sample_ids),
            "select_sample_ids": list(self.select_sample_ids),
        }

    @property
    def fingerprint(self) -> str:
        return stable_fingerprint(self.canonical_payload())

    def to_mapping(self) -> dict[str, object]:
        result = self.canonical_payload()
        result["selection_fingerprint"] = self.fingerprint
        return result

    def validate_against(self, manifest: SplitManifest) -> None:
        """Require an exhaustive D_B partition with no shared grouping key."""

        if not isinstance(manifest, SplitManifest):
            raise TypeError("manifest must be a SplitManifest")
        if not hmac.compare_digest(
            self.split_manifest_fingerprint,
            _sha256(manifest.fingerprint, name="manifest.fingerprint"),
        ):
            raise BaseTrainingProvenanceError(
                "checkpoint selection uses a different split manifest"
            )
        d_b_records = manifest.records_for("D_B")
        manifest.assert_purpose("base_train", d_b_records)
        by_id = {record.sample_id: record for record in d_b_records}
        declared = set(self.fit_sample_ids) | set(self.select_sample_ids)
        if declared != set(by_id):
            raise BaseTrainingProvenanceError(
                "D_B-fit/D_B-select must form the exact manifest D_B partition"
            )

        grouping_owners: dict[tuple[str, str], str] = {}
        for role, sample_ids in (
            (self.fit_role, self.fit_sample_ids),
            (self.select_role, self.select_sample_ids),
        ):
            for sample_id in sample_ids:
                for grouping_key in by_id[sample_id].grouping_keys():
                    prior = grouping_owners.setdefault(grouping_key, role)
                    if prior != role:
                        kind, value = grouping_key
                        raise BaseTrainingProvenanceError(
                            f"{kind}={value!r} crosses D_B-fit/D_B-select"
                        )


@dataclass(frozen=True)
class BaseTrainingProvenance:
    """Canonical, fingerprinted declaration of one frozen base checkpoint."""

    checkpoint_sha256: str
    split_manifest_fingerprint: str
    training_split: str
    d_b_samples: tuple[BaseTrainingSample, ...]
    schema_version: str = BASE_TRAINING_PROVENANCE_SCHEMA
    method_version: str = CURE_LITE_METHOD_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BASE_TRAINING_PROVENANCE_SCHEMA:
            raise BaseTrainingProvenanceError(
                f"unsupported base-training provenance schema {self.schema_version!r}"
            )
        if self.method_version != CURE_LITE_METHOD_VERSION:
            raise BaseTrainingProvenanceError(
                f"unsupported CURE-Lite method version {self.method_version!r}"
            )
        if self.training_split != "D_B":
            raise BaseTrainingProvenanceError("base training_split must be exactly 'D_B'")

        object.__setattr__(
            self,
            "checkpoint_sha256",
            _sha256(self.checkpoint_sha256, name="checkpoint_sha256"),
        )
        object.__setattr__(
            self,
            "split_manifest_fingerprint",
            _sha256(
                self.split_manifest_fingerprint,
                name="split_manifest_fingerprint",
            ),
        )
        if not isinstance(self.d_b_samples, tuple):
            raise BaseTrainingProvenanceError("d_b_samples must be a tuple")
        if not self.d_b_samples:
            raise BaseTrainingProvenanceError("d_b_samples cannot be empty")
        if any(not isinstance(item, BaseTrainingSample) for item in self.d_b_samples):
            raise BaseTrainingProvenanceError(
                "d_b_samples must contain only BaseTrainingSample records"
            )
        sample_ids = [item.sample_id for item in self.d_b_samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise BaseTrainingProvenanceError("d_b_samples contains duplicate sample IDs")
        object.__setattr__(self, "d_b_samples", tuple(sorted(self.d_b_samples)))

    @classmethod
    def from_manifest(
        cls,
        manifest: SplitManifest,
        checkpoint_path: str | Path,
    ) -> "BaseTrainingProvenance":
        """Record the current checkpoint digest and exact manifest ``D_B`` rows."""

        return cls(
            checkpoint_sha256=file_sha256(Path(checkpoint_path).resolve(strict=True)),
            split_manifest_fingerprint=manifest.fingerprint,
            training_split="D_B",
            d_b_samples=_manifest_d_b_samples(manifest),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BaseTrainingProvenance":
        """Parse a persisted provenance mapping and verify its declared fingerprint."""

        if not isinstance(value, Mapping):
            raise TypeError("base-training provenance must be a mapping")
        expected_keys = {
            "schema_version",
            "method_version",
            "checkpoint_sha256",
            "split_manifest_fingerprint",
            "training_split",
            "d_b_samples",
            "provenance_fingerprint",
        }
        if set(value) != expected_keys:
            missing = sorted(expected_keys - set(value))
            unknown = sorted(set(value) - expected_keys)
            raise BaseTrainingProvenanceError(
                f"invalid provenance fields; missing={missing}, unknown={unknown}"
            )
        raw_samples = value["d_b_samples"]
        if not isinstance(raw_samples, list):
            raise BaseTrainingProvenanceError("d_b_samples must be a list in persisted data")
        samples: list[BaseTrainingSample] = []
        for raw in raw_samples:
            if not isinstance(raw, Mapping) or set(raw) != {"sample_id", "scene_id"}:
                raise BaseTrainingProvenanceError(
                    "each persisted D_B sample must contain only sample_id and scene_id"
                )
            samples.append(BaseTrainingSample(raw["sample_id"], raw["scene_id"]))

        provenance = cls(
            checkpoint_sha256=value["checkpoint_sha256"],
            split_manifest_fingerprint=value["split_manifest_fingerprint"],
            training_split=value["training_split"],
            d_b_samples=tuple(samples),
            schema_version=value["schema_version"],
            method_version=value["method_version"],
        )
        declared = _sha256(
            value["provenance_fingerprint"], name="provenance_fingerprint"
        )
        if not hmac.compare_digest(declared, provenance.fingerprint):
            raise BaseTrainingProvenanceError(
                "base-training provenance fingerprint does not match its contents"
            )
        return provenance

    def canonical_payload(self) -> dict[str, object]:
        """Return the exact payload covered by :attr:`fingerprint`."""

        return {
            "schema_version": self.schema_version,
            "method_version": self.method_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "split_manifest_fingerprint": self.split_manifest_fingerprint,
            "training_split": self.training_split,
            "d_b_samples": [item.canonical_payload() for item in self.d_b_samples],
        }

    @property
    def fingerprint(self) -> str:
        """Canonical provenance SHA256, independent of input record ordering."""

        return stable_fingerprint(self.canonical_payload())

    def to_mapping(self) -> dict[str, object]:
        """Return a strict persisted representation including its self-check hash."""

        result = self.canonical_payload()
        result["provenance_fingerprint"] = self.fingerprint
        return result

    def validate_against(
        self,
        manifest: SplitManifest,
        checkpoint_path: str | Path,
    ) -> None:
        """Hard-fail unless checkpoint, manifest, split, samples, and scenes agree."""

        expected_manifest = _sha256(
            manifest.fingerprint, name="manifest.fingerprint"
        )
        if not hmac.compare_digest(
            self.split_manifest_fingerprint, expected_manifest
        ):
            raise BaseTrainingProvenanceError(
                "base-training provenance uses a different split manifest"
            )
        expected_samples = _manifest_d_b_samples(manifest)
        if self.d_b_samples != expected_samples:
            raise BaseTrainingProvenanceError(
                "base-training D_B sample IDs or scene IDs differ from the manifest"
            )
        actual_checkpoint = file_sha256(Path(checkpoint_path).resolve(strict=True))
        if not hmac.compare_digest(self.checkpoint_sha256, actual_checkpoint):
            raise BaseTrainingProvenanceError(
                "base-training checkpoint SHA256 does not match the checkpoint file"
            )


def validate_base_training_provenance(
    provenance: BaseTrainingProvenance | Mapping[str, Any],
    manifest: SplitManifest,
    checkpoint_path: str | Path,
) -> str:
    """Validate a provenance object/mapping and return its canonical fingerprint."""

    resolved = (
        provenance
        if isinstance(provenance, BaseTrainingProvenance)
        else BaseTrainingProvenance.from_mapping(provenance)
    )
    resolved.validate_against(manifest, checkpoint_path)
    return resolved.fingerprint


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise BaseTrainingProvenanceError(f"{name} must contain a JSON object")
    return value


def _bound_child(root: Path, raw: object, *, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise BaseTrainingProvenanceError(f"{name} must be a relative path")
    relative = Path(raw)
    if relative.is_absolute() or not relative.parts:
        raise BaseTrainingProvenanceError(f"{name} must be relative to the run")
    candidate = root
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise BaseTrainingProvenanceError(
                f"{name} must be a normalized child path"
            )
        candidate = candidate / part
        if candidate.is_symlink():
            raise BaseTrainingProvenanceError(
                f"{name} may not traverse a symlink"
            )
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise BaseTrainingProvenanceError(f"{name} escapes the training run") from error
    if not resolved.is_file():
        raise BaseTrainingProvenanceError(f"{name} must be a regular file")
    return resolved


def _resolve_manifest_asset(manifest: SplitManifest, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        if manifest.manifest_directory is None:
            raise BaseTrainingProvenanceError(
                "relative manifest assets require a loaded manifest path"
            )
        path = manifest.manifest_directory / path
    return path.resolve(strict=True)


@dataclass(frozen=True)
class FormalBaseTrainingIdentity:
    """Validated identities emitted by the controlled, fresh D_B launcher."""

    provenance_fingerprint: str
    provenance_sha256: str
    final_receipt_sha256: str
    preflight_receipt_sha256: str
    checkpoint_sha256: str
    checkpoint_selection_fingerprint: str
    upstream_commit: str
    upstream_tree: str
    model_source_sha256: str


def validate_formal_base_training_run(
    final_receipt_path: str | Path,
    provenance_path: str | Path,
    manifest: SplitManifest,
    checkpoint_path: str | Path,
) -> FormalBaseTrainingIdentity:
    """Validate the preflight→fresh trainer→checkpoint receipt chain.

    A standalone post-hoc D_B declaration is intentionally insufficient.  The
    final receipt must bind a preflight artifact created before the native child,
    an isolated view containing group-disjoint D_B-fit/D_B-select partitions,
    the exact upstream tree/recipe, and the newly produced checkpoint.  The
    external D_V split is never exposed to base fitting or checkpoint selection.
    """

    from .cache.schema import file_sha256

    final_path = Path(final_receipt_path).resolve(strict=True)
    run_root = final_path.parent
    final = _load_json_object(final_path, name="formal base final receipt")
    if final.get("schema_version") == LEGACY_FORMAL_BASE_FINAL_SCHEMA:
        raise BaseTrainingProvenanceError(
            "legacy formal base receipt is invalid because it used D_V for "
            "base checkpoint selection; regenerate a v2 D_B-fit/D_B-select run"
        )
    required_final = {
        "schema_version": FORMAL_BASE_FINAL_SCHEMA,
        "method_version": CURE_LITE_METHOD_VERSION,
        "status": "completed",
        "split_manifest_fingerprint": manifest.fingerprint,
        "native_exit_code": 0,
    }
    for key, expected in required_final.items():
        if final.get(key) != expected:
            raise BaseTrainingProvenanceError(
                f"formal base final receipt mismatch for {key}"
            )

    preflight_path = run_root / "preflight_receipt.json"
    preflight_sha256 = file_sha256(preflight_path.resolve(strict=True))
    if final.get("preflight_receipt_sha256") != preflight_sha256:
        raise BaseTrainingProvenanceError("preflight receipt SHA256 mismatch")
    preflight = _load_json_object(preflight_path, name="formal base preflight receipt")
    if preflight.get("schema_version") == LEGACY_FORMAL_BASE_PREFLIGHT_SCHEMA:
        raise BaseTrainingProvenanceError(
            "legacy formal base preflight is invalid because it assigned D_V "
            "to base checkpoint selection"
        )
    d_b_view_roles = {"train": "D_B-fit", "validation": "D_B-select"}
    required_preflight = {
        "schema_version": FORMAL_BASE_PREFLIGHT_SCHEMA,
        "method_version": CURE_LITE_METHOD_VERSION,
        "status": "ready_for_fresh_native_training",
        "split_manifest_fingerprint": manifest.fingerprint,
        "dataset_view_roles": d_b_view_roles,
        "fresh_output_policy": "new_output_no_resume_no_checkpoint_fallback",
    }
    for key, expected in required_preflight.items():
        if preflight.get(key) != expected:
            raise BaseTrainingProvenanceError(
                f"formal base preflight mismatch for {key}"
            )
    checkpoint_selection_raw = preflight.get("checkpoint_selection")
    if not isinstance(checkpoint_selection_raw, Mapping):
        raise BaseTrainingProvenanceError(
            "formal base preflight checkpoint-selection contract is missing"
        )
    checkpoint_selection = BaseCheckpointSelection.from_mapping(
        checkpoint_selection_raw
    )
    checkpoint_selection.validate_against(manifest)
    if final.get("checkpoint_selection_fingerprint") != checkpoint_selection.fingerprint:
        raise BaseTrainingProvenanceError(
            "formal base final receipt does not bind checkpoint selection"
        )
    if final.get("dataset_view_fingerprint") != preflight.get(
        "dataset_view_fingerprint"
    ) or final.get("recipe_fingerprint") != preflight.get("recipe_fingerprint"):
        raise BaseTrainingProvenanceError(
            "preflight/final dataset-view or recipe fingerprints differ"
        )
    recipe = preflight.get("recipe")
    if not isinstance(recipe, Mapping):
        raise BaseTrainingProvenanceError("formal base preflight recipe is missing")
    expected_recipe = {
        "training_split": "D_B-fit",
        "validation_split": "D_B-select",
        "checkpoint_selection_split": "D_B-select",
        "external_validation_split": None,
        "resume": False,
        "resume_path": None,
        "input_checkpoint": None,
    }
    for key, expected in expected_recipe.items():
        if recipe.get(key) != expected:
            raise BaseTrainingProvenanceError(
                f"formal base recipe mismatch for {key}"
            )
    if preflight.get("recipe_fingerprint") != stable_fingerprint(recipe):
        raise BaseTrainingProvenanceError(
            "formal base recipe fingerprint does not match the recipe"
        )

    trainer_contract = preflight.get("native_trainer_contract")
    if not isinstance(trainer_contract, Mapping):
        raise BaseTrainingProvenanceError(
            "formal base native-trainer contract is missing"
        )
    required_trainer = {
        "contract": "external-wrapper-native-train-cli-v1",
        "entrypoint": "run_pinned_mshnet_train.py",
        "native_module": "main.py",
        "fresh_output_option": "--save-dir",
        "resume_disabled_argv": ["--if-checkpoint", "false"],
        "boolean_parser": "str2bool",
    }
    for key, expected in required_trainer.items():
        if trainer_contract.get(key) != expected:
            raise BaseTrainingProvenanceError(
                f"formal base native-trainer mismatch for {key}"
            )
    _sha256(
        trainer_contract.get("wrapper_sha256"),
        name="native_trainer_contract.wrapper_sha256",
    )
    _sha256(
        trainer_contract.get("main_source_sha256"),
        name="native_trainer_contract.main_source_sha256",
    )
    view_root_raw = preflight.get("native_child_dataset_root")
    if not isinstance(view_root_raw, str) or not view_root_raw:
        raise BaseTrainingProvenanceError("native child dataset root is missing")
    view_candidate = Path(view_root_raw)
    if view_candidate.is_symlink():
        raise BaseTrainingProvenanceError(
            "native child dataset root may not be a symlink"
        )
    view_root = view_candidate.resolve(strict=True)
    try:
        view_root.relative_to(run_root.resolve(strict=True))
    except ValueError as error:
        raise BaseTrainingProvenanceError(
            "native child dataset root escapes the formal run"
        ) from error
    if not view_root.is_dir():
        raise BaseTrainingProvenanceError(
            "native child dataset root must be a regular directory"
        )
    command = preflight.get("native_command")
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
        raise BaseTrainingProvenanceError("native command binding is invalid")
    if len(command) < 2 or Path(command[1]).name != trainer_contract["entrypoint"]:
        raise BaseTrainingProvenanceError(
            "native command does not use the controlled trainer entrypoint"
        )
    for option, expected in (("--mode", "train"), ("--if-checkpoint", "false")):
        option_positions = [
            index for index, value in enumerate(command) if value == option
        ]
        if (
            len(option_positions) != 1
            or option_positions[0] + 1 >= len(command)
            or command[option_positions[0] + 1] != expected
        ):
            raise BaseTrainingProvenanceError(
                f"native command must bind {option} {expected} exactly once"
            )
    if "--resume-path" in command or "--weight-path" in command:
        raise BaseTrainingProvenanceError(
            "native command may not expose resume or input-weight paths"
        )
    positions = [index for index, value in enumerate(command) if value == "--dataset-dir"]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise BaseTrainingProvenanceError(
            "native command must contain exactly one explicit --dataset-dir"
        )
    if Path(command[positions[0] + 1]).resolve() != view_root:
        raise BaseTrainingProvenanceError(
            "native command does not use the isolated dataset view"
        )
    save_positions = [
        index for index, value in enumerate(command) if value == "--save-dir"
    ]
    if len(save_positions) != 1 or save_positions[0] + 1 >= len(command):
        raise BaseTrainingProvenanceError(
            "native command must contain exactly one explicit --save-dir"
        )
    native_output = Path(command[save_positions[0] + 1]).resolve(strict=True)
    try:
        native_output.relative_to(run_root.resolve(strict=True))
    except ValueError as error:
        raise BaseTrainingProvenanceError(
            "native output escapes the formal run"
        ) from error
    index_path = view_root / "index.json"
    if file_sha256(index_path.resolve(strict=True)) != preflight.get(
        "dataset_view_index_sha256"
    ):
        raise BaseTrainingProvenanceError("isolated dataset-view index changed")
    index = _load_json_object(index_path, name="isolated dataset-view index")
    index_without_fingerprint = dict(index)
    declared_view_fingerprint = index_without_fingerprint.pop(
        "view_fingerprint", None
    )
    if (
        declared_view_fingerprint != preflight.get("dataset_view_fingerprint")
        or declared_view_fingerprint
        != stable_fingerprint(index_without_fingerprint)
        or index.get("records") != preflight.get("dataset_view_records")
    ):
        raise BaseTrainingProvenanceError(
            "isolated dataset-view fingerprint or records are inconsistent"
        )
    if index.get("split_manifest_fingerprint") != manifest.fingerprint or index.get(
        "roles"
    ) != preflight.get("dataset_view_roles"):
        raise BaseTrainingProvenanceError(
            "isolated dataset-view manifest or D_B partition roles are invalid"
        )

    rows = preflight.get("dataset_view_records")
    if not isinstance(rows, list):
        raise BaseTrainingProvenanceError("formal base preflight has no dataset rows")
    manifest_rows = {
        record.sample_id: record
        for record in manifest.records_for("D_B")
    }
    fit_ids = set(checkpoint_selection.fit_sample_ids)
    seen: set[str] = set()
    seen_view_ids: set[str] = set()
    native_ids: dict[str, list[str]] = {"train": [], "validation": []}
    for row in rows:
        if not isinstance(row, Mapping):
            raise BaseTrainingProvenanceError("formal base dataset row is invalid")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in manifest_rows:
            raise BaseTrainingProvenanceError("formal base dataset row is not in D_B")
        if sample_id in seen:
            raise BaseTrainingProvenanceError("formal base dataset rows contain duplicates")
        seen.add(sample_id)
        record = manifest_rows[sample_id]
        expected_role = "train" if sample_id in fit_ids else "validation"
        if row.get("split") != "D_B" or row.get("role") != expected_role:
            raise BaseTrainingProvenanceError("formal base dataset role mismatch")
        view_id = row.get("view_id")
        if (
            not isinstance(view_id, str)
            or not view_id
            or view_id in seen_view_ids
        ):
            raise BaseTrainingProvenanceError(
                "formal base dataset view_id is invalid or duplicated"
            )
        seen_view_ids.add(view_id)
        native_ids[expected_role].append(view_id)
        if record.mask is None:
            raise BaseTrainingProvenanceError("formal base D_B mask is missing")
        source_image_sha256 = file_sha256(
            _resolve_manifest_asset(manifest, record.image)
        )
        source_mask_sha256 = file_sha256(
            _resolve_manifest_asset(manifest, record.mask)
        )
        declared_image_sha256 = row.get("image_sha256")
        declared_mask_sha256 = row.get("mask_sha256")
        if (
            declared_image_sha256 != source_image_sha256
            or declared_mask_sha256 != source_mask_sha256
        ):
            raise BaseTrainingProvenanceError(
                "formal base dataset content differs from the frozen manifest"
            )
        view_image = _bound_child(
            view_root,
            row.get("image"),
            name=f"dataset_view_records[{sample_id!r}].image",
        )
        view_mask = _bound_child(
            view_root,
            row.get("mask"),
            name=f"dataset_view_records[{sample_id!r}].mask",
        )
        if (
            file_sha256(view_image) != declared_image_sha256
            or file_sha256(view_mask) != declared_mask_sha256
        ):
            raise BaseTrainingProvenanceError(
                "isolated dataset-view asset differs from its frozen D_B source"
            )
    if seen != set(manifest_rows):
        raise BaseTrainingProvenanceError(
            "formal base dataset view is not the exact D_B set"
        )
    native_split_files = index.get("native_split_files_sha256")
    expected_split_files = {
        "trainval.txt": "train",
        "test.txt": "validation",
    }
    if not isinstance(native_split_files, Mapping) or set(
        native_split_files
    ) != set(expected_split_files):
        raise BaseTrainingProvenanceError(
            "isolated dataset-view native split-file bindings are invalid"
        )
    for filename, role in expected_split_files.items():
        split_file = _bound_child(view_root, filename, name=filename)
        if native_split_files.get(filename) != file_sha256(split_file):
            raise BaseTrainingProvenanceError(
                f"isolated dataset-view {filename} SHA256 mismatch"
            )
        if split_file.read_text(encoding="utf-8").splitlines() != native_ids[role]:
            raise BaseTrainingProvenanceError(
                f"isolated dataset-view {filename} does not encode {role} rows"
            )

    checkpoint_block = final.get("checkpoint")
    provenance_block = final.get("base_training_provenance")
    if not isinstance(checkpoint_block, Mapping) or not isinstance(
        provenance_block, Mapping
    ):
        raise BaseTrainingProvenanceError("formal base final receipt bindings are missing")
    bound_checkpoint = _bound_child(
        run_root, checkpoint_block.get("path"), name="checkpoint.path"
    )
    supplied_checkpoint = Path(checkpoint_path).resolve(strict=True)
    if bound_checkpoint != supplied_checkpoint:
        raise BaseTrainingProvenanceError("checkpoint is not the launcher's fresh output")
    checkpoint_sha256 = file_sha256(bound_checkpoint)
    if checkpoint_block.get("sha256") != checkpoint_sha256:
        raise BaseTrainingProvenanceError("formal base checkpoint SHA256 mismatch")

    bound_provenance = _bound_child(
        run_root,
        provenance_block.get("path"),
        name="base_training_provenance.path",
    )
    supplied_provenance = Path(provenance_path).resolve(strict=True)
    if bound_provenance != supplied_provenance:
        raise BaseTrainingProvenanceError(
            "base-training provenance is not the launcher's bound artifact"
        )
    provenance_sha256 = file_sha256(bound_provenance)
    if provenance_block.get("sha256") != provenance_sha256:
        raise BaseTrainingProvenanceError("base-training provenance SHA256 mismatch")
    provenance = BaseTrainingProvenance.from_mapping(
        _load_json_object(bound_provenance, name="base-training provenance")
    )
    provenance.validate_against(manifest, bound_checkpoint)
    if provenance_block.get("fingerprint") != provenance.fingerprint:
        raise BaseTrainingProvenanceError("base-training provenance fingerprint mismatch")

    upstream = preflight.get("upstream")
    if not isinstance(upstream, Mapping):
        raise BaseTrainingProvenanceError("formal base upstream binding is missing")
    sources = upstream.get("tracked_python_sources_sha256")
    if not isinstance(sources, Mapping) or not isinstance(
        sources.get("model/MSHNet.py"), str
    ):
        raise BaseTrainingProvenanceError("formal base MSHNet source binding is missing")
    if final.get("upstream_commit") != upstream.get("commit") or final.get(
        "upstream_tree"
    ) != upstream.get("tree"):
        raise BaseTrainingProvenanceError("preflight/final upstream identities differ")
    if final.get("launcher_sha256") != preflight.get("launcher_sha256"):
        raise BaseTrainingProvenanceError(
            "preflight/final launcher identities differ"
        )

    logs = final.get("logs")
    if not isinstance(logs, Mapping):
        raise BaseTrainingProvenanceError("formal base final logs are missing")
    for stream in ("stdout", "stderr"):
        log_path = _bound_child(run_root, logs.get(stream), name=f"logs.{stream}")
        if logs.get(f"{stream}_sha256") != file_sha256(log_path):
            raise BaseTrainingProvenanceError(
                f"formal base {stream} log SHA256 mismatch"
            )

    return FormalBaseTrainingIdentity(
        provenance_fingerprint=provenance.fingerprint,
        provenance_sha256=provenance_sha256,
        final_receipt_sha256=file_sha256(final_path),
        preflight_receipt_sha256=preflight_sha256,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_selection_fingerprint=checkpoint_selection.fingerprint,
        upstream_commit=str(upstream["commit"]),
        upstream_tree=str(upstream["tree"]),
        model_source_sha256=str(sources["model/MSHNet.py"]),
    )


__all__ = [
    "BASE_CHECKPOINT_SELECTION_SCHEMA",
    "BASE_TRAINING_PROVENANCE_SCHEMA",
    "CURE_LITE_METHOD_VERSION",
    "BaseCheckpointSelection",
    "BaseTrainingProvenance",
    "BaseTrainingProvenanceError",
    "BaseTrainingSample",
    "FormalBaseTrainingIdentity",
    "FORMAL_BASE_FINAL_SCHEMA",
    "FORMAL_BASE_PREFLIGHT_SCHEMA",
    "LEGACY_FORMAL_BASE_FINAL_SCHEMA",
    "LEGACY_FORMAL_BASE_PREFLIGHT_SCHEMA",
    "validate_base_training_provenance",
    "validate_formal_base_training_run",
]
