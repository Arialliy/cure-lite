"""Create-only, D_R-only preflight for the paired formal training schedule.

The preflight serializes the two complete 800 x 40 identity schedules built
by :mod:`cure_lite.experiment.paired_formal_schedule`.  It performs no model
forward, optimization, calibration, or evaluation.  The artifact contains
only identities, exposure counts, fingerprints, and method-to-schedule
bindings; feature tensors are never persisted.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from ..cache.schema import file_sha256, stable_fingerprint
from ..train.paired_pools import (
    PAIRED_EPOCHS,
    PAIRED_EXPOSURES,
    PAIRED_OPTIMIZER_UPDATES,
    PAIRED_STEPS_PER_EPOCH,
    PAIRS_PER_UPDATE,
)
from ..train.paired_step import (
    DECODER_STATES_PER_UPDATE,
    FACTUAL_ANCHOR_BATCH_SIZE,
)
from .paired_formal_schedule import (
    DECODER_FORWARDS_PER_UPDATE,
    FORMAL_METHOD_KINDS,
    PAIRED_FORMAL_BINDING_SCHEMA,
    PairedFormalSchedule,
    bind_paired_formal_schedule,
)


PAIRED_FORMAL_PREFLIGHT_CONFIG_SCHEMA = (
    "cure-lite-paired-formal-preflight-config-v1"
)
PAIRED_FORMAL_PREFLIGHT_SCHEDULE_SCHEMA = (
    "cure-lite-paired-formal-preflight-schedule-v1"
)
PAIRED_FORMAL_PREFLIGHT_METHOD_SCHEMA = (
    "cure-lite-paired-formal-preflight-method-bindings-v1"
)
PAIRED_FORMAL_PREFLIGHT_RECEIPT_SCHEMA = (
    "cure-lite-paired-formal-preflight-receipt-v1"
)
PAIRED_FORMAL_PREFLIGHT_COMPLETE_SCHEMA = (
    "cure-lite-paired-formal-preflight-complete-v1"
)

FORMAL_PREFLIGHT_SEEDS = (42, 43)
_CONFIG_NAME = "frozen_config.json"
_SEED_FILE_NAMES = {
    42: "seed42_schedule.json",
    43: "seed43_schedule.json",
}
_METHOD_NAME = "method_bindings.json"
_RECEIPT_NAME = "preflight_receipt.json"
_COMPLETE_NAME = "COMPLETE.json"
_INCOMPLETE_NAME = ".incomplete"
_HEX = frozenset("0123456789abcdef")
_RECEIPT_GATES = {
    "d_r_only": True,
    "authoritative_catalogs_shared_across_seeds": True,
    "complete_seed42_schedule": True,
    "complete_seed43_schedule": True,
    "exact_4_4_2_composition": True,
    "exact_12_states_and_3_forwards_per_update": True,
    "all_pair_and_anchor_exposures_nonzero": True,
    "all_nine_methods_bound_per_seed": True,
    "method_label_does_not_affect_schedule": True,
    "implementation_sha_binding_verified": True,
}
_RECEIPT_EXECUTION_POLICY = {
    "schedule_only": True,
    "training_performed": False,
    "model_forward_performed": False,
    "optimizer_constructed": False,
    "D_V_accessed": False,
    "D_T_accessed": False,
    "calibration_performed": False,
    "inference_performed": False,
    "formal_training_authorized_by_this_artifact": False,
    "scientific_override_applied": False,
}


def _json_bytes(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new_json(path: Path, payload: object) -> None:
    encoded = _json_bytes(payload)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite formal preflight artifact {path}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _strict_json(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")

    def reject_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{name} contains non-finite number {value}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _fingerprinted(
    payload: Mapping[str, object],
    *,
    field: str,
) -> dict[str, object]:
    core = dict(payload)
    if field in core:
        raise ValueError(f"{field} must not already be present")
    return {**core, field: stable_fingerprint(core)}


def _verify_fingerprint(
    payload: Mapping[str, object],
    *,
    field: str,
    name: str,
) -> str:
    core = dict(payload)
    fingerprint = core.pop(field, None)
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in _HEX for character in fingerprint)
        or stable_fingerprint(core) != fingerprint
    ):
        raise RuntimeError(f"{name} fingerprint mismatch")
    return fingerprint


def validate_paired_formal_preflight_config(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Validate the immutable preflight config and return a plain copy."""

    if not isinstance(payload, Mapping):
        raise TypeError("formal preflight config must be a mapping")
    config = dict(payload)
    fingerprint = config.pop("config_fingerprint", None)
    if stable_fingerprint(config) != fingerprint:
        raise ValueError("formal preflight config fingerprint mismatch")
    if config.get("schema_version") != PAIRED_FORMAL_PREFLIGHT_CONFIG_SCHEMA:
        raise ValueError("unsupported formal preflight config schema")
    if config.get("dataset") != "IRSTD-1K" or config.get("split") != "D_R":
        raise ValueError("formal preflight is frozen to IRSTD-1K D_R")
    if config.get("seeds") != list(FORMAL_PREFLIGHT_SEEDS):
        raise ValueError("formal preflight seeds must remain [42, 43]")
    if config.get("methods") != list(FORMAL_METHOD_KINDS):
        raise ValueError("formal preflight method inventory changed")
    budget = config.get("budget")
    expected_budget = {
        "epochs": PAIRED_EPOCHS,
        "steps_per_epoch": PAIRED_STEPS_PER_EPOCH,
        "optimizer_updates_per_seed": PAIRED_OPTIMIZER_UPDATES,
        "factual_miss_states_per_update": FACTUAL_ANCHOR_BATCH_SIZE,
        "factual_no_miss_states_per_update": FACTUAL_ANCHOR_BATCH_SIZE,
        "clean_pairs_per_update": PAIRS_PER_UPDATE,
        "paired_endpoint_states_per_update": 2 * PAIRS_PER_UPDATE,
        "decoder_states_per_update": DECODER_STATES_PER_UPDATE,
        "decoder_forwards_per_update": DECODER_FORWARDS_PER_UPDATE,
    }
    if budget != expected_budget:
        raise ValueError("formal preflight budget changed")
    policy = config.get("execution_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("formal preflight execution policy is missing")
    exact_policy = {
        "create_only_output": True,
        "allowed_runtime_splits": ["D_R"],
        "schedule_only": True,
        "training_performed": False,
        "allow_D_V": False,
        "allow_D_T": False,
        "allow_calibration": False,
        "allow_inference": False,
        "allow_scientific_overrides": False,
        "resume": False,
        "overwrite": False,
    }
    if dict(policy) != exact_policy:
        raise ValueError("formal preflight execution policy changed")
    if not isinstance(config.get("input_binding"), Mapping):
        raise ValueError("formal preflight input binding is missing")
    implementation = config.get("implementation_binding")
    if not isinstance(implementation, Mapping) or not implementation:
        raise ValueError("formal preflight code binding is missing")
    for relative, digest in implementation.items():
        if not isinstance(relative, str) or not relative:
            raise ValueError("formal preflight code path is invalid")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in _HEX for character in digest)
        ):
            raise ValueError("formal preflight code SHA256 is invalid")
    return {**config, "config_fingerprint": fingerprint}


def _ledger_fingerprint(
    *,
    seed: int,
    formal_schedule_fingerprint: str,
    branch: str,
    identities: list[dict[str, object]],
) -> str:
    return stable_fingerprint(
        {
            "schema_version": (
                "cure-lite-paired-formal-preflight-exposure-ledger-v1"
            ),
            "seed": seed,
            "formal_schedule_fingerprint": formal_schedule_fingerprint,
            "branch": branch,
            "identities": identities,
        }
    )


def build_formal_schedule_receipt(
    schedule: PairedFormalSchedule,
) -> dict[str, object]:
    """Build one tensor-free receipt for a complete seed schedule."""

    if not isinstance(schedule, PairedFormalSchedule):
        raise TypeError("schedule must be a PairedFormalSchedule")
    if schedule.seed not in FORMAL_PREFLIGHT_SEEDS:
        raise ValueError("formal preflight accepts only seeds 42 and 43")
    pair_rows = [
        {
            "pair_id": pair.pair_id,
            "sample_id": pair.sample_id,
            "count": schedule.pair_exposure_counts[index],
        }
        for index, pair in enumerate(schedule.paired_schedule.pairs)
    ]
    miss_rows = [
        {
            **anchor.canonical_payload(),
            "count": schedule.factual_miss_exposure_counts[index],
        }
        for index, anchor in enumerate(schedule.factual_miss_anchors)
    ]
    no_miss_rows = [
        {
            **anchor.canonical_payload(),
            "count": schedule.factual_no_miss_exposure_counts[index],
        }
        for index, anchor in enumerate(schedule.factual_no_miss_anchors)
    ]
    source_rows = [
        row.canonical_payload() for row in schedule.source_exposure_ledger
    ]
    ledgers = {
        "pair": {
            "identities": pair_rows,
            "total": sum(row["count"] for row in pair_rows),
            "zero_exposure": sum(row["count"] == 0 for row in pair_rows),
            "ledger_fingerprint": _ledger_fingerprint(
                seed=schedule.seed,
                formal_schedule_fingerprint=schedule.schedule_fingerprint,
                branch="clean_pair",
                identities=pair_rows,
            ),
        },
        "factual_miss": {
            "identities": miss_rows,
            "total": sum(row["count"] for row in miss_rows),
            "zero_exposure": sum(row["count"] == 0 for row in miss_rows),
            "ledger_fingerprint": _ledger_fingerprint(
                seed=schedule.seed,
                formal_schedule_fingerprint=schedule.schedule_fingerprint,
                branch="factual_miss",
                identities=miss_rows,
            ),
        },
        "factual_no_miss": {
            "identities": no_miss_rows,
            "total": sum(row["count"] for row in no_miss_rows),
            "zero_exposure": sum(
                row["count"] == 0 for row in no_miss_rows
            ),
            "ledger_fingerprint": _ledger_fingerprint(
                seed=schedule.seed,
                formal_schedule_fingerprint=schedule.schedule_fingerprint,
                branch="factual_no_miss",
                identities=no_miss_rows,
            ),
        },
        "source": {
            "identities": source_rows,
            "total": sum(row["total_exposures"] for row in source_rows),
            "zero_exposure": sum(
                row["total_exposures"] == 0 for row in source_rows
            ),
            "ledger_fingerprint": _ledger_fingerprint(
                seed=schedule.seed,
                formal_schedule_fingerprint=schedule.schedule_fingerprint,
                branch="source",
                identities=source_rows,
            ),
        },
    }
    core: dict[str, object] = {
        "schema_version": PAIRED_FORMAL_PREFLIGHT_SCHEDULE_SCHEMA,
        "dataset": "IRSTD-1K",
        "split": "D_R",
        "seed": schedule.seed,
        "prepared_catalog_fingerprint": (
            schedule.prepared_catalog_fingerprint
        ),
        "pair_catalog_fingerprint": (
            schedule.paired_schedule.catalog_fingerprint
        ),
        "paired_schedule_fingerprint": (
            schedule.paired_schedule.schedule_fingerprint
        ),
        "formal_schedule_fingerprint": schedule.schedule_fingerprint,
        "sequence_fingerprints": {
            "pair": schedule.pair_sequence_fingerprint,
            "factual_miss": schedule.factual_miss_sequence_fingerprint,
            "factual_no_miss": (
                schedule.factual_no_miss_sequence_fingerprint
            ),
            "combined": schedule.combined_sequence_fingerprint,
        },
        "budget": {
            "epochs": PAIRED_EPOCHS,
            "steps_per_epoch": PAIRED_STEPS_PER_EPOCH,
            "optimizer_updates": schedule.optimizer_updates,
            "factual_miss_states_per_update": (
                FACTUAL_ANCHOR_BATCH_SIZE
            ),
            "factual_no_miss_states_per_update": (
                FACTUAL_ANCHOR_BATCH_SIZE
            ),
            "clean_pairs_per_update": PAIRS_PER_UPDATE,
            "paired_endpoint_states_per_update": 2 * PAIRS_PER_UPDATE,
            "decoder_states_per_update": DECODER_STATES_PER_UPDATE,
            "decoder_forwards_per_update": DECODER_FORWARDS_PER_UPDATE,
            "decoder_state_evaluations": (
                schedule.decoder_state_evaluations
            ),
            "decoder_forward_calls": schedule.decoder_forward_calls,
        },
        "exposure_ledgers": ledgers,
        "gates": {
            "complete_800_by_40_schedule": (
                schedule.optimizer_updates == PAIRED_OPTIMIZER_UPDATES
            ),
            "exact_4_4_2_composition": True,
            "exact_12_states_per_update": (
                DECODER_STATES_PER_UPDATE == 12
            ),
            "exact_3_forwards_per_update": (
                DECODER_FORWARDS_PER_UPDATE == 3
            ),
            "all_pairs_nonzero_exposure": (
                ledgers["pair"]["zero_exposure"] == 0
            ),
            "all_factual_miss_anchors_nonzero_exposure": (
                ledgers["factual_miss"]["zero_exposure"] == 0
            ),
            "all_factual_no_miss_anchors_nonzero_exposure": (
                ledgers["factual_no_miss"]["zero_exposure"] == 0
            ),
        },
    }
    return _fingerprinted(core, field="schedule_receipt_fingerprint")


def build_formal_method_bindings(
    schedules: Mapping[int, PairedFormalSchedule],
) -> dict[str, object]:
    """Bind all nine method labels to the same schedule within each seed."""

    if set(schedules) != set(FORMAL_PREFLIGHT_SEEDS):
        raise ValueError("method bindings require exact seeds 42 and 43")
    seed_rows = []
    for seed in FORMAL_PREFLIGHT_SEEDS:
        schedule = schedules[seed]
        rows = []
        for method in FORMAL_METHOD_KINDS:
            binding = bind_paired_formal_schedule(
                schedule,
                method_kind=method,
            )
            if binding.schedule is not schedule:
                raise RuntimeError("method binding rebuilt the schedule")
            rows.append(
                {
                    "method": method,
                    "shared_formal_schedule_fingerprint": (
                        binding.shared_schedule_fingerprint
                    ),
                    "binding_fingerprint": (
                        binding.binding_fingerprint
                    ),
                    "method_label_affects_schedule": False,
                }
            )
        if {row["shared_formal_schedule_fingerprint"] for row in rows} != {
            schedule.schedule_fingerprint
        }:
            raise RuntimeError("method labels changed the shared schedule")
        seed_rows.append(
            {
                "seed": seed,
                "formal_schedule_fingerprint": (
                    schedule.schedule_fingerprint
                ),
                "methods": rows,
            }
        )
    core: dict[str, object] = {
        "schema_version": PAIRED_FORMAL_PREFLIGHT_METHOD_SCHEMA,
        "method_inventory": list(FORMAL_METHOD_KINDS),
        "seeds": seed_rows,
        "method_label_affects_schedule": False,
        "all_methods_share_one_schedule_per_seed": True,
    }
    return _fingerprinted(core, field="method_bindings_fingerprint")


def _validate_schedule_receipt(payload: Mapping[str, object]) -> str:
    fingerprint = _verify_fingerprint(
        payload,
        field="schedule_receipt_fingerprint",
        name="formal schedule receipt",
    )
    if payload.get("schema_version") != PAIRED_FORMAL_PREFLIGHT_SCHEDULE_SCHEMA:
        raise RuntimeError("formal schedule receipt schema changed")
    if payload.get("dataset") != "IRSTD-1K" or payload.get("split") != "D_R":
        raise RuntimeError("formal schedule receipt left IRSTD-1K D_R")
    seed = payload.get("seed")
    if seed not in FORMAL_PREFLIGHT_SEEDS:
        raise RuntimeError("formal schedule receipt seed changed")
    budget = payload.get("budget")
    if not isinstance(budget, Mapping) or dict(budget) != {
        "epochs": 800,
        "steps_per_epoch": 40,
        "optimizer_updates": 32_000,
        "factual_miss_states_per_update": 4,
        "factual_no_miss_states_per_update": 4,
        "clean_pairs_per_update": 2,
        "paired_endpoint_states_per_update": 4,
        "decoder_states_per_update": 12,
        "decoder_forwards_per_update": 3,
        "decoder_state_evaluations": 384_000,
        "decoder_forward_calls": 96_000,
    }:
        raise RuntimeError("formal schedule receipt budget changed")
    sequences = payload.get("sequence_fingerprints")
    if not isinstance(sequences, Mapping) or set(sequences) != {
        "pair",
        "factual_miss",
        "factual_no_miss",
        "combined",
    }:
        raise RuntimeError("formal sequence fingerprint inventory changed")
    for digest in (
        payload.get("prepared_catalog_fingerprint"),
        payload.get("pair_catalog_fingerprint"),
        payload.get("paired_schedule_fingerprint"),
        payload.get("formal_schedule_fingerprint"),
        *sequences.values(),
    ):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in _HEX for character in digest)
        ):
            raise RuntimeError("formal schedule contains an invalid digest")
    ledgers = payload.get("exposure_ledgers")
    if not isinstance(ledgers, Mapping) or set(ledgers) != {
        "pair",
        "factual_miss",
        "factual_no_miss",
        "source",
    }:
        raise RuntimeError("formal exposure ledger inventory changed")
    expected_totals = {
        "pair": PAIRED_EXPOSURES,
        "factual_miss": 128_000,
        "factual_no_miss": 128_000,
        "source": PAIRED_EXPOSURES + 256_000,
    }
    branch_names = {
        "pair": "clean_pair",
        "factual_miss": "factual_miss",
        "factual_no_miss": "factual_no_miss",
        "source": "source",
    }
    for branch, expected_total in expected_totals.items():
        ledger = ledgers[branch]
        if not isinstance(ledger, Mapping):
            raise RuntimeError(f"{branch} exposure ledger is malformed")
        identities = ledger.get("identities")
        if not isinstance(identities, list) or not identities:
            raise RuntimeError(f"{branch} exposure ledger is empty")
        count_field = "total_exposures" if branch == "source" else "count"
        total = 0
        for row in identities:
            if (
                not isinstance(row, Mapping)
                or isinstance(row.get(count_field), bool)
                or not isinstance(row.get(count_field), int)
                or row[count_field] < 1
            ):
                raise RuntimeError(f"{branch} exposure row is invalid")
            total += row[count_field]
        expected_ledger_fingerprint = _ledger_fingerprint(
            seed=int(seed),
            formal_schedule_fingerprint=str(
                payload["formal_schedule_fingerprint"]
            ),
            branch=branch_names[branch],
            identities=[dict(row) for row in identities],
        )
        if (
            total != expected_total
            or ledger.get("total") != expected_total
            or ledger.get("zero_exposure") != 0
            or ledger.get("ledger_fingerprint")
            != expected_ledger_fingerprint
        ):
            raise RuntimeError(f"{branch} exposure ledger changed")
    gates = payload.get("gates")
    if not isinstance(gates, Mapping) or not gates or not all(
        value is True for value in gates.values()
    ):
        raise RuntimeError("formal schedule gates are not all true")
    return fingerprint


def _validate_method_bindings(
    payload: Mapping[str, object],
    schedules: Mapping[int, Mapping[str, object]],
) -> str:
    fingerprint = _verify_fingerprint(
        payload,
        field="method_bindings_fingerprint",
        name="formal method bindings",
    )
    if payload.get("schema_version") != PAIRED_FORMAL_PREFLIGHT_METHOD_SCHEMA:
        raise RuntimeError("formal method binding schema changed")
    if payload.get("method_inventory") != list(FORMAL_METHOD_KINDS):
        raise RuntimeError("formal method inventory changed")
    if (
        payload.get("method_label_affects_schedule") is not False
        or payload.get("all_methods_share_one_schedule_per_seed") is not True
    ):
        raise RuntimeError("formal method-sharing contract changed")
    seed_rows = payload.get("seeds")
    if not isinstance(seed_rows, list) or [
        row.get("seed") if isinstance(row, Mapping) else None
        for row in seed_rows
    ] != list(FORMAL_PREFLIGHT_SEEDS):
        raise RuntimeError("formal method seed inventory changed")
    for row in seed_rows:
        seed = row["seed"]
        expected_schedule = schedules[seed]["formal_schedule_fingerprint"]
        if row.get("formal_schedule_fingerprint") != expected_schedule:
            raise RuntimeError("formal method schedule binding changed")
        methods = row.get("methods")
        if (
            not isinstance(methods, list)
            or [method.get("method") for method in methods]
            != list(FORMAL_METHOD_KINDS)
            or any(
                method.get("shared_formal_schedule_fingerprint")
                != expected_schedule
                or method.get("method_label_affects_schedule") is not False
                for method in methods
            )
        ):
            raise RuntimeError("formal method label changed the schedule")
        for method in methods:
            expected_binding = stable_fingerprint(
                {
                    "schema_version": PAIRED_FORMAL_BINDING_SCHEMA,
                    "method_kind": method["method"],
                    "shared_schedule_fingerprint": expected_schedule,
                    "method_label_affects_schedule": False,
                }
            )
            if method.get("binding_fingerprint") != expected_binding:
                raise RuntimeError("formal method binding fingerprint changed")
    return fingerprint


@dataclass(frozen=True)
class PublishedPairedFormalPreflight:
    root: Path
    config_fingerprint: str
    prepared_catalog_fingerprint: str
    pair_catalog_fingerprint: str
    seed42_formal_schedule_fingerprint: str
    seed43_formal_schedule_fingerprint: str
    method_bindings_fingerprint: str
    receipt_fingerprint: str
    complete_fingerprint: str

    def verify_unchanged(self) -> None:
        verified = load_paired_formal_preflight_artifact(self.root)
        if verified != self:
            raise RuntimeError("paired formal preflight identity changed")


def write_paired_formal_preflight_artifact(
    schedules: Mapping[int, PairedFormalSchedule],
    *,
    config: Mapping[str, object],
    config_file_sha256: str,
    input_file_sha256: Mapping[str, str],
    implementation_file_sha256: Mapping[str, str],
    output_dir: str | Path,
) -> PublishedPairedFormalPreflight:
    """Publish one deterministic, tensor-free, create-only preflight."""

    config_payload = validate_paired_formal_preflight_config(config)
    if set(schedules) != set(FORMAL_PREFLIGHT_SEEDS):
        raise ValueError("formal preflight requires exact seeds 42 and 43")
    for seed in FORMAL_PREFLIGHT_SEEDS:
        if (
            not isinstance(schedules[seed], PairedFormalSchedule)
            or schedules[seed].seed != seed
        ):
            raise ValueError("formal preflight schedule seed mismatch")
    prepared = {
        schedules[seed].prepared_catalog_fingerprint
        for seed in FORMAL_PREFLIGHT_SEEDS
    }
    pair_catalogs = {
        schedules[seed].paired_schedule.catalog_fingerprint
        for seed in FORMAL_PREFLIGHT_SEEDS
    }
    if len(prepared) != 1 or len(pair_catalogs) != 1:
        raise RuntimeError("seed schedules do not share authoritative catalogs")
    input_binding = config_payload["input_binding"]
    if (
        not isinstance(input_binding, Mapping)
        or input_binding.get("real_pair_catalog_fingerprint")
        != next(iter(pair_catalogs))
    ):
        raise RuntimeError("formal preflight pair catalog freeze changed")
    pair_identity_sets = {
        frozenset(
            pair.pair_id
            for pair in schedules[seed].paired_schedule.pairs
        )
        for seed in FORMAL_PREFLIGHT_SEEDS
    }
    factual_miss_identity_sets = {
        frozenset(
            anchor.anchor_id
            for anchor in schedules[seed].factual_miss_anchors
        )
        for seed in FORMAL_PREFLIGHT_SEEDS
    }
    factual_no_miss_identity_sets = {
        frozenset(
            anchor.anchor_id
            for anchor in schedules[seed].factual_no_miss_anchors
        )
        for seed in FORMAL_PREFLIGHT_SEEDS
    }
    if (
        len(pair_identity_sets) != 1
        or len(factual_miss_identity_sets) != 1
        or len(factual_no_miss_identity_sets) != 1
    ):
        raise RuntimeError("seed schedules do not share identity populations")
    configured_implementation = config_payload["implementation_binding"]
    if dict(implementation_file_sha256) != configured_implementation:
        raise RuntimeError("formal preflight implementation SHA binding changed")
    for values, name in (
        (input_file_sha256, "input"),
        (implementation_file_sha256, "implementation"),
    ):
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"{name} SHA binding must be non-empty")
        for path, digest in values.items():
            if not isinstance(path, str) or not isinstance(digest, str):
                raise ValueError(f"{name} SHA binding is malformed")
            if len(digest) != 64 or any(c not in _HEX for c in digest):
                raise ValueError(f"{name} SHA binding digest is invalid")
    schedule_receipts = {
        seed: build_formal_schedule_receipt(schedules[seed])
        for seed in FORMAL_PREFLIGHT_SEEDS
    }
    method_bindings = build_formal_method_bindings(schedules)
    if (
        not isinstance(config_file_sha256, str)
        or len(config_file_sha256) != 64
        or any(character not in _HEX for character in config_file_sha256)
    ):
        raise ValueError("formal preflight source config SHA256 is invalid")
    receipt_core: dict[str, object] = {
        "schema_version": PAIRED_FORMAL_PREFLIGHT_RECEIPT_SCHEMA,
        "execution_status": "completed",
        "dataset": "IRSTD-1K",
        "split": "D_R",
        "config_fingerprint": config_payload["config_fingerprint"],
        "source_config_file_sha256": config_file_sha256,
        "prepared_catalog_fingerprint": next(iter(prepared)),
        "pair_catalog_fingerprint": next(iter(pair_catalogs)),
        "seed_schedule_bindings": {
            str(seed): {
                "paired_schedule_fingerprint": (
                    schedules[seed].paired_schedule.schedule_fingerprint
                ),
                "formal_schedule_fingerprint": (
                    schedules[seed].schedule_fingerprint
                ),
                "pair_sequence_fingerprint": (
                    schedules[seed].pair_sequence_fingerprint
                ),
                "factual_miss_sequence_fingerprint": (
                    schedules[seed].factual_miss_sequence_fingerprint
                ),
                "factual_no_miss_sequence_fingerprint": (
                    schedules[seed].factual_no_miss_sequence_fingerprint
                ),
                "combined_sequence_fingerprint": (
                    schedules[seed].combined_sequence_fingerprint
                ),
                "schedule_receipt_fingerprint": (
                    schedule_receipts[seed][
                        "schedule_receipt_fingerprint"
                    ]
                ),
            }
            for seed in FORMAL_PREFLIGHT_SEEDS
        },
        "method_bindings_fingerprint": (
            method_bindings["method_bindings_fingerprint"]
        ),
        "input_file_sha256": dict(sorted(input_file_sha256.items())),
        "implementation_file_sha256": dict(
            sorted(implementation_file_sha256.items())
        ),
        "gates": dict(_RECEIPT_GATES),
        "execution_policy": dict(_RECEIPT_EXECUTION_POLICY),
    }
    receipt = _fingerprinted(receipt_core, field="receipt_fingerprint")

    requested = Path(output_dir).expanduser()
    if requested.is_symlink() or requested.exists():
        raise FileExistsError(
            f"refusing to overwrite formal preflight output {requested}"
        )
    root = requested.resolve(strict=False)
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(exist_ok=False)
    incomplete = root / _INCOMPLETE_NAME
    incomplete.open("xb").close()
    _write_new_json(root / _CONFIG_NAME, config_payload)
    for seed, filename in _SEED_FILE_NAMES.items():
        _write_new_json(root / filename, schedule_receipts[seed])
    _write_new_json(root / _METHOD_NAME, method_bindings)
    _write_new_json(root / _RECEIPT_NAME, receipt)
    artifact_names = (
        _CONFIG_NAME,
        *_SEED_FILE_NAMES.values(),
        _METHOD_NAME,
        _RECEIPT_NAME,
    )
    complete_core: dict[str, object] = {
        "schema_version": PAIRED_FORMAL_PREFLIGHT_COMPLETE_SCHEMA,
        "execution_status": "completed",
        "dataset": "IRSTD-1K",
        "split": "D_R",
        "config_fingerprint": config_payload["config_fingerprint"],
        "prepared_catalog_fingerprint": next(iter(prepared)),
        "pair_catalog_fingerprint": next(iter(pair_catalogs)),
        "seed42_formal_schedule_fingerprint": (
            schedules[42].schedule_fingerprint
        ),
        "seed43_formal_schedule_fingerprint": (
            schedules[43].schedule_fingerprint
        ),
        "method_bindings_fingerprint": (
            method_bindings["method_bindings_fingerprint"]
        ),
        "receipt_fingerprint": receipt["receipt_fingerprint"],
        "artifact_files": {
            name: file_sha256(root / name) for name in artifact_names
        },
        "artifact_file_count": len(artifact_names),
        "raw_tensor_artifact_file_count": 0,
        "training_performed": False,
        "D_V_accessed": False,
        "D_T_accessed": False,
    }
    complete = _fingerprinted(
        complete_core,
        field="complete_fingerprint",
    )
    _write_new_json(root / _COMPLETE_NAME, complete)
    incomplete.unlink()
    return load_paired_formal_preflight_artifact(root)


def load_paired_formal_preflight_artifact(
    output_dir: str | Path,
) -> PublishedPairedFormalPreflight:
    """Strictly load a completed formal schedule preflight artifact."""

    root = Path(output_dir).expanduser().resolve(strict=False)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("formal preflight root must be a regular directory")
    if (root / _INCOMPLETE_NAME).exists():
        raise RuntimeError("formal preflight artifact is incomplete")
    expected_names = {
        _CONFIG_NAME,
        *_SEED_FILE_NAMES.values(),
        _METHOD_NAME,
        _RECEIPT_NAME,
        _COMPLETE_NAME,
    }
    if {path.name for path in root.iterdir()} != expected_names:
        raise RuntimeError("formal preflight artifact inventory changed")
    config = _strict_json(root / _CONFIG_NAME, name="frozen config")
    validate_paired_formal_preflight_config(config)
    schedules = {
        seed: _strict_json(root / filename, name=f"seed-{seed} schedule")
        for seed, filename in _SEED_FILE_NAMES.items()
    }
    schedule_receipt_fingerprints = {
        seed: _validate_schedule_receipt(payload)
        for seed, payload in schedules.items()
    }
    if len({
        schedules[seed]["prepared_catalog_fingerprint"]
        for seed in FORMAL_PREFLIGHT_SEEDS
    }) != 1:
        raise RuntimeError("formal seed prepared catalog bindings disagree")
    if len({
        schedules[seed]["pair_catalog_fingerprint"]
        for seed in FORMAL_PREFLIGHT_SEEDS
    }) != 1:
        raise RuntimeError("formal seed pair catalog bindings disagree")
    methods = _strict_json(
        root / _METHOD_NAME,
        name="formal method bindings",
    )
    method_fingerprint = _validate_method_bindings(methods, schedules)
    receipt = _strict_json(root / _RECEIPT_NAME, name="preflight receipt")
    receipt_fingerprint = _verify_fingerprint(
        receipt,
        field="receipt_fingerprint",
        name="formal preflight receipt",
    )
    complete = _strict_json(root / _COMPLETE_NAME, name="COMPLETE receipt")
    complete_fingerprint = _verify_fingerprint(
        complete,
        field="complete_fingerprint",
        name="formal preflight COMPLETE",
    )
    artifact_names = (
        _CONFIG_NAME,
        *_SEED_FILE_NAMES.values(),
        _METHOD_NAME,
        _RECEIPT_NAME,
    )
    expected_hashes = {
        name: file_sha256(root / name) for name in artifact_names
    }
    if (
        complete.get("artifact_files") != expected_hashes
        or complete.get("artifact_file_count") != len(artifact_names)
        or complete.get("raw_tensor_artifact_file_count") != 0
    ):
        raise RuntimeError("formal preflight artifact file hashes changed")
    config_fingerprint = config["config_fingerprint"]
    prepared = schedules[42]["prepared_catalog_fingerprint"]
    pair_catalog = schedules[42]["pair_catalog_fingerprint"]
    seed_bindings = receipt.get("seed_schedule_bindings")
    if not isinstance(seed_bindings, Mapping):
        raise RuntimeError("formal preflight seed bindings are malformed")
    for seed in FORMAL_PREFLIGHT_SEEDS:
        binding = seed_bindings.get(str(seed))
        schedule = schedules[seed]
        if (
            not isinstance(binding, Mapping)
            or binding.get("formal_schedule_fingerprint")
            != schedule["formal_schedule_fingerprint"]
            or binding.get("paired_schedule_fingerprint")
            != schedule["paired_schedule_fingerprint"]
            or binding.get("schedule_receipt_fingerprint")
            != schedule_receipt_fingerprints[seed]
            or binding.get("pair_sequence_fingerprint")
            != schedule["sequence_fingerprints"]["pair"]
            or binding.get("factual_miss_sequence_fingerprint")
            != schedule["sequence_fingerprints"]["factual_miss"]
            or binding.get("factual_no_miss_sequence_fingerprint")
            != schedule["sequence_fingerprints"]["factual_no_miss"]
            or binding.get("combined_sequence_fingerprint")
            != schedule["sequence_fingerprints"]["combined"]
        ):
            raise RuntimeError("formal preflight seed binding changed")
    if not (
        receipt.get("dataset")
        == complete.get("dataset")
        == "IRSTD-1K"
    ):
        raise RuntimeError("formal preflight dataset binding changed")
    if not (
        receipt.get("split") == complete.get("split") == "D_R"
        and receipt.get("config_fingerprint")
        == complete.get("config_fingerprint")
        == config_fingerprint
        and receipt.get("prepared_catalog_fingerprint")
        == complete.get("prepared_catalog_fingerprint")
        == prepared
        and receipt.get("pair_catalog_fingerprint")
        == complete.get("pair_catalog_fingerprint")
        == pair_catalog
        and receipt.get("method_bindings_fingerprint")
        == complete.get("method_bindings_fingerprint")
        == method_fingerprint
        and complete.get("seed42_formal_schedule_fingerprint")
        == schedules[42]["formal_schedule_fingerprint"]
        and complete.get("seed43_formal_schedule_fingerprint")
        == schedules[43]["formal_schedule_fingerprint"]
        and complete.get("receipt_fingerprint") == receipt_fingerprint
        and complete.get("training_performed") is False
        and complete.get("D_V_accessed") is False
        and complete.get("D_T_accessed") is False
    ):
        raise RuntimeError("formal preflight cross-file bindings disagree")
    receipt_gates = receipt.get("gates")
    receipt_policy = receipt.get("execution_policy")
    source_config_sha = receipt.get("source_config_file_sha256")
    implementation_files = receipt.get("implementation_file_sha256")
    input_files = receipt.get("input_file_sha256")
    if (
        not isinstance(receipt_gates, Mapping)
        or dict(receipt_gates) != _RECEIPT_GATES
        or not isinstance(receipt_policy, Mapping)
        or dict(receipt_policy) != _RECEIPT_EXECUTION_POLICY
        or not isinstance(source_config_sha, str)
        or len(source_config_sha) != 64
        or any(character not in _HEX for character in source_config_sha)
        or implementation_files != config.get("implementation_binding")
        or not isinstance(input_files, Mapping)
        or not input_files
    ):
        raise RuntimeError("formal preflight gate or policy changed")
    for file_map, name in (
        (implementation_files, "implementation"),
        (input_files, "input"),
    ):
        for path, digest in file_map.items():
            if (
                not isinstance(path, str)
                or not path
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in _HEX for character in digest)
            ):
                raise RuntimeError(
                    f"formal preflight {name} SHA binding is malformed"
                )
    return PublishedPairedFormalPreflight(
        root=root,
        config_fingerprint=str(config_fingerprint),
        prepared_catalog_fingerprint=str(prepared),
        pair_catalog_fingerprint=str(pair_catalog),
        seed42_formal_schedule_fingerprint=str(
            schedules[42]["formal_schedule_fingerprint"]
        ),
        seed43_formal_schedule_fingerprint=str(
            schedules[43]["formal_schedule_fingerprint"]
        ),
        method_bindings_fingerprint=str(method_fingerprint),
        receipt_fingerprint=str(receipt_fingerprint),
        complete_fingerprint=str(complete_fingerprint),
    )


__all__ = [
    "FORMAL_PREFLIGHT_SEEDS",
    "PAIRED_FORMAL_PREFLIGHT_COMPLETE_SCHEMA",
    "PAIRED_FORMAL_PREFLIGHT_CONFIG_SCHEMA",
    "PAIRED_FORMAL_PREFLIGHT_METHOD_SCHEMA",
    "PAIRED_FORMAL_PREFLIGHT_RECEIPT_SCHEMA",
    "PAIRED_FORMAL_PREFLIGHT_SCHEDULE_SCHEMA",
    "PublishedPairedFormalPreflight",
    "build_formal_method_bindings",
    "build_formal_schedule_receipt",
    "load_paired_formal_preflight_artifact",
    "validate_paired_formal_preflight_config",
    "write_paired_formal_preflight_artifact",
]
