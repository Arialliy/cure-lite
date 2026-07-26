"""Frozen reachability-aware development inputs for NLCC-v12."""

from __future__ import annotations

from .nlcc_dataset_free_inputs import (
    NLCCInputProfile,
    build_factual_batches,
    build_factual_population,
    build_outcome_batch,
    build_pair_specs,
    build_schedule,
    build_strata,
    catalog_fingerprint,
    catalog_manifest,
    factual_indices_for_update,
    factual_population_fingerprint,
    factual_population_manifest,
    factual_schedule_fingerprint,
    factual_schedule_manifest,
    input_fingerprint,
    input_manifest,
    pair_tensor_manifest,
    reachability_audit,
    schedule_fingerprint,
    schedule_manifest,
)


DEVELOPMENT_PROFILE = NLCCInputProfile(
    profile_id="nlcc_v12_development",
    design_seed=2550254881,
    update_count=320,
    group_dyad_counts=(2, 2, 2, 2, 2, 2, 2, 2),
    group_low_exposures=(24, 24, 24, 24, 24, 24, 6, 6),
    group_high_exposures=(25, 25, 25, 25, 25, 25, 6, 6),
    group_high_quotas=(2, 2, 1, 1, 1, 1, 0, 0),
)


def build_nlcc_development_pair_specs():
    return build_pair_specs(DEVELOPMENT_PROFILE)


def build_nlcc_development_schedule(specs=None):
    return build_schedule(DEVELOPMENT_PROFILE, specs)


def build_nlcc_development_outcome_batch(specs, *, device="cpu"):
    return build_outcome_batch(DEVELOPMENT_PROFILE, specs, device=device)


def build_nlcc_development_factual_population(*, device="cpu"):
    return build_factual_population(DEVELOPMENT_PROFILE, device=device)


def build_nlcc_development_factual_batches(*, update_index, device="cpu"):
    return build_factual_batches(
        DEVELOPMENT_PROFILE,
        update_index=update_index,
        device=device,
    )


def nlcc_development_reachability_audit(specs=None):
    return reachability_audit(DEVELOPMENT_PROFILE, specs)


def nlcc_development_manifest(specs=None):
    return input_manifest(DEVELOPMENT_PROFILE, specs)


def nlcc_development_fingerprint(specs=None):
    return input_fingerprint(DEVELOPMENT_PROFILE, specs)


__all__ = [
    "DEVELOPMENT_PROFILE",
    "build_nlcc_development_factual_batches",
    "build_nlcc_development_factual_population",
    "build_nlcc_development_outcome_batch",
    "build_nlcc_development_pair_specs",
    "build_nlcc_development_schedule",
    "build_strata",
    "catalog_fingerprint",
    "catalog_manifest",
    "factual_indices_for_update",
    "factual_population_fingerprint",
    "factual_population_manifest",
    "factual_schedule_fingerprint",
    "factual_schedule_manifest",
    "nlcc_development_fingerprint",
    "nlcc_development_manifest",
    "nlcc_development_reachability_audit",
    "pair_tensor_manifest",
    "schedule_fingerprint",
    "schedule_manifest",
]
