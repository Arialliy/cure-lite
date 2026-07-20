"""Strict provenance contract for a base detector trained only on ``D_B``.

The CURE-Lite evidence protocol is invalid when the frozen base checkpoint has
seen residual-training, validation, or test samples.  This module records the
exact ``D_B`` sample/scene membership next to the checkpoint and validates it
against the frozen split manifest before downstream caches are trusted.
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
CURE_LITE_METHOD_VERSION = "cure-lite-v0.1"


class BaseTrainingProvenanceError(ValueError):
    """Raised when base-training provenance cannot satisfy the formal protocol."""


FORMAL_BASE_PREFLIGHT_SCHEMA = "cure-lite-mshnet-base-preflight-v1"
FORMAL_BASE_FINAL_SCHEMA = "cure-lite-mshnet-base-final-receipt-v1"


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
    an isolated view containing only D_B/D_V, the exact upstream tree/recipe,
    and the newly produced checkpoint.
    """

    from .cache.schema import file_sha256

    final_path = Path(final_receipt_path).resolve(strict=True)
    run_root = final_path.parent
    final = _load_json_object(final_path, name="formal base final receipt")
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
    required_preflight = {
        "schema_version": FORMAL_BASE_PREFLIGHT_SCHEMA,
        "method_version": CURE_LITE_METHOD_VERSION,
        "status": "ready_for_fresh_native_training",
        "split_manifest_fingerprint": manifest.fingerprint,
        "dataset_view_roles": {"train": "D_B", "validation": "D_V"},
        "fresh_output_policy": "new_output_no_resume_no_checkpoint_fallback",
    }
    for key, expected in required_preflight.items():
        if preflight.get(key) != expected:
            raise BaseTrainingProvenanceError(
                f"formal base preflight mismatch for {key}"
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
        "training_split": "D_B",
        "validation_split": "D_V",
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

    rows = preflight.get("dataset_view_records")
    if not isinstance(rows, list):
        raise BaseTrainingProvenanceError("formal base preflight has no dataset rows")
    manifest_rows = {
        record.sample_id: record
        for split in ("D_B", "D_V")
        for record in manifest.records_for(split)
    }
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise BaseTrainingProvenanceError("formal base dataset row is invalid")
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in manifest_rows:
            raise BaseTrainingProvenanceError("formal base dataset row is not D_B/D_V")
        if sample_id in seen:
            raise BaseTrainingProvenanceError("formal base dataset rows contain duplicates")
        seen.add(sample_id)
        record = manifest_rows[sample_id]
        expected_role = "train" if record.split == "D_B" else "validation"
        if row.get("split") != record.split or row.get("role") != expected_role:
            raise BaseTrainingProvenanceError("formal base dataset role mismatch")
        if record.mask is None:
            raise BaseTrainingProvenanceError("formal base D_B/D_V mask is missing")
        if row.get("image_sha256") != file_sha256(
            _resolve_manifest_asset(manifest, record.image)
        ) or row.get("mask_sha256") != file_sha256(
            _resolve_manifest_asset(manifest, record.mask)
        ):
            raise BaseTrainingProvenanceError(
                "formal base dataset content differs from the frozen manifest"
            )
    if seen != set(manifest_rows):
        raise BaseTrainingProvenanceError(
            "formal base dataset view is not the exact D_B/D_V set"
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
        upstream_commit=str(upstream["commit"]),
        upstream_tree=str(upstream["tree"]),
        model_source_sha256=str(sources["model/MSHNet.py"]),
    )


__all__ = [
    "BASE_TRAINING_PROVENANCE_SCHEMA",
    "CURE_LITE_METHOD_VERSION",
    "BaseTrainingProvenance",
    "BaseTrainingProvenanceError",
    "BaseTrainingSample",
    "FormalBaseTrainingIdentity",
    "FORMAL_BASE_FINAL_SCHEMA",
    "FORMAL_BASE_PREFLIGHT_SCHEMA",
    "validate_base_training_provenance",
    "validate_formal_base_training_run",
]
