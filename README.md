# CURE project: CURE-Lite Stage-A

This repository aims to develop **CURE** as a detector-independent correction
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
- The latest recorded active software suite reports `268 passed` with the
  historical MSHNet-adapter test excluded. This verifies an implementation
  path, not real-data performance or the current five-way protocol outcome.
- A project-owned reference U-Net is the current Stage-A Base. Its run is
  configured for 800 epochs using only `D_B-fit`, with selection only on
  `D_B-select`. It is an **experimental condition**, not a CURE component or
  novelty claim.
- The current five-way config trains each CURE-Lite decoder for 20 epochs with
  40 steps per epoch. The Base's 800 epochs and the decoders' `20 x 40` update
  schedule are different training stages and must not be conflated.
- The formal Stage-A core and CLI consume generic frozen-Base caches containing
  `(p_b, F_b)`; they do not import, instantiate, train, or depend on MSHNet.
  This is a detector-neutral core boundary, not evidence that arbitrary IRSTD
  detectors are already plug-and-play: each detector still needs a compatible
  cache producer or thin adapter and a declared feature contract.
- The original single-seed `A / Base@B / F / U` run was terminated and remains
  incomplete. It must not be resumed, upgraded, or used as evidence for the
  current source tree.
- The intended formal protocol is now the five-way `A / Base@B / F / F× / U`
  comparison. Its config, decision rule, and protocol-freeze v2 exist, but it
  has not run to completion, so there is no current formal CURE-Lite result.
- Full CURE and cross-backbone experiments have not started.
- A completed Stage-A run records `receipts/efficiency.json` for the deployed
  `U` decoder. It separates replayable incremental parameter/MAC/FLOP counts
  from device-dependent latency and memory observations. These values describe
  cost; they do not select the method or determine mechanism success.
- CURE-Lite computes the registered Base topology/weight-state fingerprint
  itself. The same value is bound by the `D_R/D_V` cache indexes, Stage-A
  completion record, verified Base-run identity, and calibrated deployment
  record; an adapter declaration alone is not accepted as proof of state
  equality.
- A completed single-seed run can be converted directly into the detector-
  neutral Stage-A seed registry v6 with
  `build_seed_registry_from_stage_a_run`. The builder reads the completed run,
  cache records, selected protocols, verified Base-run identity, and measured
  efficiency protocol instead of accepting a second hand-written experiment
  description.
- Registry v6 is an evidence index, not a place to add thresholds after seeing
  `D_V`. The single-seed decision is fixed before execution by config v4's
  five-way method contract, decision-rule v3, and protocol-freeze v2, and the
  CLI executes that exact strict-Pd rule. Multi-seed confidence intervals are
  a later analysis and are not claimed by the single-seed registry.

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
5. trains one matched lightweight positive residual decoder for each of
   `F`, `F×`, and `U`; and
6. combines Base and residual predictions with a monotone hard union.

The occupancy map is projected to the feature grid with adaptive max pooling,
so a feature cell is occupied if any source-grid pixel in that cell is
occupied. A legal deletion enters synthetic training only when it also changes
this projected occupancy. This makes the intervention visible to the decoder
instead of treating a source-grid-only change as usable supervision.

The central comparison is not “add a decoder and compare with the Base.” It is
the controlled set `A / Base@B / F / F× / U`:

- `A`: decoder-free frozen Base at the selected anchor threshold;
- `Base@B`: decoder-free threshold-relaxed Base under the same total
  constraints;
- `F`: factual-only residual decoder;
- `F×`: factual-only positive replacement in the same third loss slot,
  matching `U` in initialization, updates, active state-forwards, batch size,
  and loss weight without performing a deletion;
- `U`: the same residual design plus uniform projected-visible legal deletion.

`F×` makes extra training exposure an explicit control rather than an informal
caveat. It matches update/state exposure, not target-area, positive-pixel,
difficulty, or feature distributions. A positive intervention signal requires
`U` to beat `Base@B`, `F`, and `F×` under the same configured constraints.
Thus `A` and `Base@B` provide decoder-free references for the Base anchor and
threshold relaxation. `F/F×/U` use the same decoder topology; the comparisons
among them isolate factual learning, matched exposure, and coverage
intervention respectively.

## Detector-neutral Stage-A cache boundary

The active data path is:

```text
project-owned reference U-Net (D_B only; experimental condition)
  -> generic D_R/D_V caches: probability p_b + feature F_b
  -> CURE-Lite Stage-A core
  -> adaptive-max occupancy and projected-visible support check
  -> F / F× / U training
  -> A / Base@B / F / F× / U calibration and D_V development scores
```

The formal runner starts at the cache boundary. Its command also requires the
completed reference-Base run whose provider-verified identity must match both
cache indexes before a Stage-A output directory is created:

```bash
python tools/run_with_gpu_temperature_control.py \
  --gpu 0 \
  --pause-temp 82 \
  --resume-temp 75 \
  --poll-seconds 1 \
  -- \
  /home/md0/ly/MSHNet/.venv/bin/python tools/run_stage_a.py \
    --manifest protocols/IRSTD-1K/stage_a_seed42/manifest.json \
    --d-r-base-index /path/to/base_cache/D_R/index.json \
    --d-v-base-index /path/to/base_cache/D_V/index.json \
    --reference-base-run /path/to/reference_base_run \
    --config protocols/IRSTD-1K/stage_a_seed42_fx_v2/stage_a_config.json \
    --output /path/to/new_stage_a_run \
    --calibration-workers 24
```

The launcher exposes only physical GPU0 to the child command. At 82 degrees
Celsius it pauses the complete child process group, and it continues only
after a valid GPU0 reading reaches 75 degrees Celsius or lower. A failed
temperature probe also pauses active work. These thresholds control execution;
they do not alter the Stage-A method contract or select a result.

Calibration now prepares the fixed GT/anchor/matching/reachability context once
and evaluates the threshold ledger in isolated worker processes. Worker count
is execution-only: candidate metrics, tie-breaking, receipts, and full replay
are exactly the same as the scalar reference path. Candidate progress is
reported on stderr. `--calibration-workers` is therefore an execution-only
parameter rather than part of the v4 scientific method contract. A new five-way
run requires a new output directory, decision rule, and protocol freeze.

The active five-way config and decision rule live under
`protocols/IRSTD-1K/stage_a_seed42_fx_v2/`. Its final `protocol_freeze.json`
was generated after the catalog optimization and output paths were fixed.
Freeze v2
binds both package sources and the run/assessment tool hashes, as well as the
new cache, Stage-A, and assessment paths. The historical four-way freeze
and the aborted `fx_v1` freeze remain untouched and do not authorize resuming
either terminated run. The active five-way protocol is frozen but has no
formal result yet.

The five-way v4 configuration includes `method_contract` and
`support_requirements`. Before decoder training,
the runner checks that the fixed anchor already satisfies the shared
false-alarm/retention constraints and that `D_R` contains enough factual-miss,
no-miss, reachable-target, synthetic, and projected-visible legal support. The
measured counts and requirements are written to `receipts/support.json`.
After calibration, the runner also measures only the deployed `U` decoder on
all-zero synthetic tensors with the verified `D_V` shapes and writes
`receipts/efficiency.json`. This path accepts neither images nor labels and
excludes Base inference, data loading, thresholding, union, and metrics.
Here `v4` names the scientific configuration schema. The completed-run
envelope is `cure-lite-stage-a-run-v7`; it binds the efficiency receipt, the
library-computed Base state fingerprint, and the verified Base-run identity.
The downstream evidence schemas are seed/master registry v6, efficiency
receipt v1, calibrated
deployment v2, CLI summary v4, and replayed assessment v3. These version
numbers describe different artifacts and are not interchangeable.

The CLI summary and replayed assessment expose only the formal development
fields: Pd, mIoU, nIoU, pixel FA, FP components/MP, RawBG-FA, retention,
constraint status, training support, and the efficiency receipt. Additional
internal diagnostics are omitted from this public record and do not enter the
Stage-A success claim.

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

python tools/run_with_gpu_temperature_control.py \
  --gpu 0 \
  --pause-temp 82 \
  --resume-temp 75 \
  --poll-seconds 1 \
  -- \
  /home/md0/ly/MSHNet/.venv/bin/python tools/cache_reference_base.py \
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
contract. Until those adapters and experiments exist, the repository supports
only a detector-neutral core interface, not an arbitrary-IRSTD plug-and-play
claim. A later multi-backbone study is required to test whether Full CURE can
act as a general post-detector correction framework rather than a head tied to
one network.

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
