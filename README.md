# CURE project: CURE-Lite Stage-A

This repository is developing **CURE**, a detector-independent correction
framework for infrared small-target detection (IRSTD). **CURE-Lite is the first
stage of that project, not the final CURE method.** Its job is to test the
smallest mechanistic hypothesis before adding any mechanism to Full CURE:

> Given a frozen detector, can coverage-state intervention teach a positive
> residual predictor to add factual-miss targets without exceeding the same
> total false-alarm and covered-target-retention constraints?

Full CURE is intentionally not implemented yet. If the real CURE-Lite gate
succeeds, its measured residual errors will determine the additional mechanism
designed for CURE. This order prevents the final method from becoming a list of
unmotivated modules.

## Current status

- CURE-Lite semantic, gradient, cache, training, calibration, and evaluation
  contracts are implemented.
- The latest complete software suite reports `160 passed`. This verifies the
  implementation path, not real-data performance.
- A project-owned reference U-Net is the current Stage-A Base. Its run is
  configured for 800 epochs using only `D_B-fit`, with selection only on
  `D_B-select`. It is an **experimental condition**, not a CURE component or
  novelty claim.
- The formal Stage-A core and CLI consume generic frozen-Base caches containing
  `(p_b, F_b)`; they do not import, instantiate, train, or depend on MSHNet.
- The original frozen single-seed `A / Base@B / F / U` development pilot is
  still allowed to finish in its unchanged checkout. The next formal protocol
  is the five-way `A / Base@B / F / F× / U` comparison; old receipts are not
  rewritten or interpreted as if they contained `F×`.
- Full CURE and cross-backbone experiments have not started.

Older MSHNet-specific integration utilities are historical engineering work.
They are not part of the active Stage-A route and their outputs are not current
CURE-Lite evidence.

## CURE-Lite mechanism

For each image, a frozen Base supplies a probability map `p_b` and one detached
spatial feature `F_b`. CURE-Lite then:

1. builds a fixed-threshold base occupancy state;
2. matches covered and missed targets deterministically;
3. creates factual miss/no-miss supervision from `D_R`;
4. traverses a seed-specific, without-replacement cycle over legal covered
   components for the synthetic branch;
5. trains only a lightweight positive residual decoder; and
6. combines Base and residual predictions with a monotone hard union.

The occupancy map is projected to the feature grid with adaptive max pooling,
so a feature cell is occupied if any source-grid pixel in that cell is
occupied. A legal deletion enters synthetic training only when it also changes
this projected occupancy. This makes the intervention visible to the decoder
instead of treating a source-grid-only change as usable supervision.

The central comparison is not “add a decoder and compare with the Base.” It is
the controlled set `A / Base@B / F / F× / U`:

- `A`: frozen Base at the selected anchor threshold;
- `Base@B`: threshold-relaxed Base under the same total constraints;
- `F`: factual-only residual decoder;
- `F×`: factual-only positive replacement in the same third loss slot,
  matching `U` in initialization, updates, state-forwards, positive exposure,
  batch size, and loss weight without performing a deletion;
- `U`: the same residual design plus uniform projected-visible legal deletion.

`F×` makes extra training exposure an explicit control rather than an informal
caveat. A positive intervention signal requires `U` to beat `Base@B`, `F`, and
`F×` under the same configured constraints.

## Model-independent Stage-A boundary

The active data path is:

```text
project-owned reference U-Net (D_B only; experimental condition)
  -> generic D_R/D_V caches: probability p_b + feature F_b
  -> CURE-Lite Stage-A core
  -> adaptive-max occupancy and projected-visible support check
  -> A / Base@B / F / F× / U training, calibration, and D_V development scores
```

The formal runner starts at the cache boundary. Its command accepts only a
manifest, the `D_R` and `D_V` cache indexes, the Stage-A configuration, and a
new output path:

```bash
python tools/run_stage_a.py \
  --manifest protocols/IRSTD-1K/stage_a_seed42/manifest.json \
  --d-r-base-index /path/to/base_cache/D_R/index.json \
  --d-v-base-index /path/to/base_cache/D_V/index.json \
  --config protocols/IRSTD-1K/stage_a_seed42_fx_v1/stage_a_config.json \
  --output /path/to/new_stage_a_run \
  --calibration-workers 24
```

Calibration now prepares the fixed GT/anchor/matching/reachability context once
and evaluates the threshold ledger in isolated worker processes. Worker count
is execution-only: candidate metrics, tie-breaking, receipts, and full replay
are exactly the same as the scalar reference path. Candidate progress is
reported on stderr. A new five-way run requires a new output directory,
decision rule, and protocol freeze.

The five-way config and decision rule live under
`protocols/IRSTD-1K/stage_a_seed42_fx_v1/`. Its final `protocol_freeze.json`
must be generated only after the implementation commit and output paths are
fixed. Freeze v2 binds both package sources and the run/assessment tool hashes;
the historical four-way freeze remains untouched.

The five-way v3 configuration includes `method_contract` and
`support_requirements`. Before decoder training,
the runner checks that the fixed anchor already satisfies the shared
false-alarm/retention constraints and that `D_R` contains enough factual-miss,
no-miss, reachable-target, synthetic, and projected-visible legal support. The
measured counts and requirements are written to `receipts/support.json`.

All scores produced from `D_V` are **development selection scores**. They may
decide whether to continue or redesign CURE-Lite, but they are not test-set
results and do not establish an ICLR contribution by themselves. No current
Stage-A entry reads `D_T` images or labels.

## Reference Base and cache preparation

The reference Base is maintained inside this repository solely to create a
neutral, reproducible frozen-Base condition:

```bash
python tools/train_reference_base.py \
  --manifest protocols/IRSTD-1K/stage_a_seed42/manifest.json \
  --config protocols/IRSTD-1K/stage_a_seed42/base_training_config.json \
  --output /path/to/new_reference_base_run

python tools/cache_reference_base.py \
  --manifest protocols/IRSTD-1K/stage_a_seed42/manifest.json \
  --reference-base-run /path/to/reference_base_run \
  --output /path/to/new_base_cache \
  --device cuda:0
```

The cache producer is outside the CURE-Lite method core. Replacing this
reference provider later changes only cache production, not Stage-A semantics.

## Project progression

The intended progression is:

```text
CURE-Lite real mechanism gate
  -> if unsupported: revise the minimal mechanism and repeat
  -> if supported: design Full CURE from the measured mismatch
  -> verify Full CURE through thin adapters on DNANet, UIUNet,
     MSHNet, and SCTransNet while keeping the CURE core unchanged
```

Each future adapter will map a detector's native preprocessing, probability
output, feature tensor, and grid metadata into the common `(p_b, F_b)`
contract. The multi-backbone study is what can support the claim that CURE is a
general post-detector correction framework rather than a head tied to one
network.

## Install and public API

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

```python
from cure_lite import (
    CURELiteDecoder,
    CURELiteLoss,
    CURELiteModel,
    FrozenBaseAdapter,
    FrozenBaseOutput,
)
```

See [STAGE.md](STAGE.md) for the project-stage boundary,
[VALIDATION.md](VALIDATION.md) for the evidence gates, and
[DATA_LOADING.md](DATA_LOADING.md) for dataset and cache contracts.
