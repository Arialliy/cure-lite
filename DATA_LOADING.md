# CURE data and generic Base-cache contract

## Dataset identity is independent of repository location

The project uses the existing IRSTD datasets **NUAA, NUDT, and IRSTD-1K**. Their
names are not changed by CURE-Lite. The files may physically live under a
directory such as `/home/md0/ly/MSHNet/datasets`; that path is storage only and
does not make MSHNet the CURE-Lite Base, loader, dependency, or method host.

CURE owns experiment membership through one explicit manifest. It does not use
an upstream detector's train/test loader to decide which samples belong to:

- `D_B`: reference-Base fitting and checkpoint selection;
- `D_R`: CURE-Lite residual training;
- `D_V`: development calibration and mechanism screening;
- `D_T`: later official evaluation, not read during Stage-A.

For the current IRSTD-1K Stage-A manifest, official-training samples are split
into group-disjoint `D_B / D_R / D_V`; official-test membership is recorded as
`D_T` without opening its images or labels. `D_B` is further divided into
group-disjoint `D_B-fit / D_B-select` for Base training and checkpoint choice.

## Current reference-Base data path

The project-owned reference U-Net is the only Base condition for the current
real Stage-A. Its active configuration at
`protocols/IRSTD-1K/stage_a_seed42/base_training_config.json` specifies 800
epochs using only `D_B-fit`, with model selection only on `D_B-select`.

This U-Net is an experimental provider. Its architecture and training loss are
not part of the CURE-Lite method. Once its checkpoint is fixed, the provider
reads only the manifest's `D_R` and `D_V` records and exports two cache indexes.
The formal Stage-A runner then starts from those indexes and no longer depends
on the provider implementation.

```text
manifest D_B -> reference U-Net training/selection
fixed reference U-Net + manifest D_R/D_V
  -> generic cache indexes
  -> formal CURE-Lite Stage-A
```

## Generic `(p_b, F_b)` cache boundary

Each Base-cache record contains the detector evidence required by CURE-Lite:

- sample identity and split membership;
- frozen probability map `p_b` on the declared evaluation grid;
- detached spatial feature `F_b` with a fixed channel/grid contract;
- preprocessing and Base fingerprints;
- file hashes and index metadata used to reproduce the exact cache bundle.

`tools/run_stage_a.py` accepts only:

```text
--manifest
--d-r-base-index
--d-v-base-index
--reference-base-run
--config
--output
--calibration-workers
```

The required arguments bind the v4 scientific inputs, the provider-verified
Base training run, and the output. The optional
`--calibration-workers` argument changes only candidate-evaluation parallelism;
it must not change candidate metrics, selection, or receipts. The command has
no detector-name, detector-repository, checkpoint-selection, or `D_T` argument.
Consequently, the active Stage-A core cannot switch behavior based on MSHNet or
any other architecture, although a compatible detector-specific cache producer
is still required upstream.

At decoder time, the source-grid occupancy is projected to the feature grid by
adaptive max pooling. Cache preprocessing and grid metadata must therefore be
consistent with the probability and feature tensors. Synthetic training uses
only legal deletions that remain visible after this projection. Stage-A v4
checks the amount of such support before training and records it in
`receipts/support.json`.

## Development-only use of `D_V`

`D_V` is used to select the Base anchor, calibrate `Base@B` and residual
thresholds, and report the current Stage-A development scores. Those values
must be called development selection scores. They cannot be relabelled as
official test performance. Current Stage-A code does not read `D_T` images or
labels.

## Future detector adapters

After CURE-Lite supports the minimal mechanism and Full CURE is designed, thin
adapters will be written for DNANet, UIUNet, MSHNet, and SCTransNet. Each
adapter has only four responsibilities:

1. reproduce the detector's native deterministic evaluation preprocessing;
2. expose its probability output as `p_b`;
3. expose one declared spatial feature as `F_b`; and
4. describe the valid image/feature grids needed by the common cache contract.

Detector training remains in the detector's own project. CURE consumes the
resulting generic caches, so adding a detector must not add detector-specific
logic to the CURE core. Differences in color mode, normalization, resize,
padding, or feature resolution belong to the adapter and cache metadata, not
to dataset naming or Stage-A membership.

Cross-backbone validation is a later CURE experiment. It does not begin before
the current reference-Base CURE-Lite gate has produced an interpretable result.
Until the adapters and those experiments exist, this cache contract establishes
a detector-neutral software boundary, not arbitrary-IRSTD plug-and-play
support.
