"""Frozen unique exposure-holdout inputs for NLCC-v12."""

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


HOLDOUT_PROFILE = NLCCInputProfile(
    profile_id="nlcc_v12_exposure_holdout",
    design_seed=1788878112,
    update_count=400,
    group_dyad_counts=(18, 17, 17, 17, 17, 17, 4, 4),
    group_low_exposures=(3, 3, 3, 3, 3, 3, 3, 3),
    group_high_exposures=(4, 4, 4, 4, 4, 4, 4, 4),
    group_high_quotas=(11, 10, 10, 10, 10, 10, 3, 3),
)


def build_nlcc_holdout_pair_specs():
    return build_pair_specs(HOLDOUT_PROFILE)


def build_nlcc_holdout_schedule(specs=None):
    return build_schedule(HOLDOUT_PROFILE, specs)


def build_nlcc_holdout_outcome_batch(specs, *, device="cpu"):
    return build_outcome_batch(HOLDOUT_PROFILE, specs, device=device)


def build_nlcc_holdout_factual_population(*, device="cpu"):
    return build_factual_population(HOLDOUT_PROFILE, device=device)


def build_nlcc_holdout_factual_batches(*, update_index, device="cpu"):
    return build_factual_batches(
        HOLDOUT_PROFILE,
        update_index=update_index,
        device=device,
    )


def nlcc_holdout_reachability_audit(specs=None):
    return reachability_audit(HOLDOUT_PROFILE, specs)


def nlcc_holdout_manifest(specs=None):
    return input_manifest(HOLDOUT_PROFILE, specs)


def nlcc_holdout_fingerprint(specs=None):
    return input_fingerprint(HOLDOUT_PROFILE, specs)


__all__ = [
    "HOLDOUT_PROFILE",
    "build_nlcc_holdout_factual_batches",
    "build_nlcc_holdout_factual_population",
    "build_nlcc_holdout_outcome_batch",
    "build_nlcc_holdout_pair_specs",
    "build_nlcc_holdout_schedule",
    "build_strata",
    "catalog_fingerprint",
    "catalog_manifest",
    "factual_indices_for_update",
    "factual_population_fingerprint",
    "factual_population_manifest",
    "factual_schedule_fingerprint",
    "factual_schedule_manifest",
    "nlcc_holdout_fingerprint",
    "nlcc_holdout_manifest",
    "nlcc_holdout_reachability_audit",
    "pair_tensor_manifest",
    "schedule_fingerprint",
    "schedule_manifest",
]
