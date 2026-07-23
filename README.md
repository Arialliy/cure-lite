# CURE project: CURE-Lite Stage-A

This repository develops **CURE**, a detector-independent correction method for
infrared small-target detection (IRSTD). **CURE-Lite is the first design and
verification stage of CURE, not the final CURE method.**

The current order is fixed:

```text
CURE-Lite mechanism design and development verification
  -> Full CURE design, only if the Lite mechanism is supported
  -> cross-backbone and multi-dataset verification, only after Full CURE exists
```

Full CURE has not been designed or implemented. Integrations with DNANet,
UIUNet, MSHNet, and SCTransNet have not started. The project-owned reference
U-Net is only the frozen Base used to produce the current Stage-A caches; it is
not a CURE component or an innovation claim.

## Current status

### v0.1: completed diagnostic stage

The two formal `fx_v3` development runs are complete:

- seed 42: `Pd(F) = 0.905882`, `Pd(U) = 0.888235`;
- seed 43: `Pd(F) = Pd(U) = 0.894118`.

The predeclared v0.1 rule required uniform legal selection `U` to be strictly
better than `Base@B`, `F`, and `F×` while every method remained within the same
false-alarm and covered-target-retention constraints. That strict rule was not
met in either seed.

This is not a failure of the whole CURE idea. It is evidence that **uniformly
selecting a legal synthetic target is not sufficiently aligned with the
factual-miss regime**. The completed v0.1 artifacts remain unchanged and are
used as the reference for the next single-mechanism test.

### v0.2: M completed; the mechanism signal is negative

The v0.2 variant is `miss_aligned_legal`, abbreviated **M**. For every reachable
factual miss, it:

1. computes `log1p(feature RMS)` over that target's positive region using the
   detached frozen-Base feature;
2. computes the same descriptor for every decoder-visible legal target in the
   `D_R` catalog;
3. selects the globally nearest legal target by absolute descriptor distance
   quantized at `1e6`; and
4. resolves exact ties by the canonical target identity. Candidate reuse is
   allowed.

M changes **only synthetic-target selection**. It does not change the frozen
Base, legal-target definition, supervision, residual decoder, loss, branch
weights, optimizer, training horizon, or inference rule. Therefore v0.2 is one
mechanistic correction to the unsupported v0.1 selection assumption, not a
stack of new modules.

The real `D_R` catalog currently gives:

- reachable factual misses mapped: `32 / 32`;
- unique selected legal targets: `16`;
- unique selected legal sources: `16`;
- maximum reuse of one legal target: `7`;
- mean quantized descriptor distance: `31342.1875`;
- maximum quantized descriptor distance: `148544`;
- alignment-catalog fingerprint:
  `f53b65e9962642e2705a6641861d039245c507e6703f8846242376bb43411f73`.

Both independent 800-epoch development runs are complete:

| seed | Pd(F) | Pd(U) | Pd(M) | mIoU(M) | nIoU(M) | recovered misses U / M | v0.2 gate |
|---:|---:|---:|---:|---:|---:|:---:|:---|
| 42 | 0.905882 | 0.888235 | 0.882353 | 0.592637 | 0.549957 | 4 / 3 of 23 | not met |
| 43 | 0.894118 | 0.894118 | 0.876471 | 0.605239 | 0.550010 | 5 / 2 of 23 | not met |

All compared methods satisfy the configured false-alarm and covered-target
retention constraints. Nevertheless, M has lower Pd and recovers fewer anchor
misses than U in both seeds. The formal `mechanism_signal` is therefore false
for both runs.

The run itself did not fail: both runs contain 800 consecutive epochs, 40
steps per epoch, valid completion records, and exact artifact bindings. The
historical A/Base@B/F/F×/U results were reused without retraining or mutation.
Independent replay checks reproduced the selected M threshold and metrics.

The mapping audit also exposes an important limitation of v0.2. Its 32 factual
miss mappings collapse onto only 16 of 209 legal targets, with reuse counts
`[7, 4, 4, 3, 2, 2, 1 x 10]`. The corresponding full-mapping Kish effective
sample size is `1024 / 108 = 9.48`. This is evidence that the one-dimensional
hard nearest-neighbour rule concentrates the synthetic training distribution;
it is not evidence against the broader CURE problem formulation.

The current full test suite reports `359 passed`. This validates the software
path; the two formal runs above provide the separate performance result.

## Mechanism comparison

The v0.1 reference contains five conditions:

- `A`: frozen Base at its selected anchor threshold, without a decoder;
- `Base@B`: threshold-relaxed frozen Base under the same constraints;
- `F`: factual-only residual learning;
- `F×`: factual-positive exposure control in the same third loss slot;
- `U`: factual learning plus uniformly selected decoder-visible legal targets.

The v0.2 extension adds:

- `M`: the same residual learning design as `U`, but with miss-aligned legal
  target selection.

This comparison isolates the selection rule. M is not allowed to gain a
different decoder, loss, supervision rule, update budget, or inference path.

## Formal v0.2 runner

The create-only M runner extends one completed v0.1 run:

- it reads the historical `D_R` and `D_V` caches;
- it reuses the completed `F`, `F×`, and `U` artifacts without retraining them;
- it trains only M for `800` epochs and `40` steps per epoch;
- it calibrates only M on the already frozen residual-threshold grid;
- it evaluates U once at its historical selected threshold to recover the
  directly comparable miss-recovery counts; and
- it writes a new sibling output directory and never modifies the historical
  Stage-A directory.

Example:

```bash
python tools/run_stage_a_m_extension.py \
  --reference-stage-a runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3 \
  --manifest protocols/IRSTD-1K/stage_a_seed42/manifest.json \
  --output runs/irstd1k_stage_a_seed42/cure_lite_stage_a_m_v02_seed42 \
  --device cuda:0 \
  --calibration-workers 24
```

The output path must not already exist. If an execution is interrupted, a
later execution uses another new output path; the runner does not continue an
old partial output.

## v0.2 development rule and decision

M is supported at the current single-seed development gate only if all of the
following hold:

1. every compared method satisfies the configured false-alarm and retention
   constraints;
2. `Pd(M) > Pd(Base@B)`;
3. `Pd(M) > Pd(F)`;
4. `Pd(M) > Pd(F×)`;
5. `Pd(M) > Pd(U)`; and
6. M recovers more anchor misses than U at U's historical selected threshold,
   with equal recovery denominators.

Improving only over U would support the target-selection change, but it would
not yet establish successful CURE-Lite under the full rule. mIoU and nIoU are
secondary checks and cannot replace the Pd and recovery requirements.

The observed result is negative under this predeclared rule in both seeds.
Consequently, Full CURE design and cross-backbone integration remain blocked by
the Lite-stage evidence gate. The next work stays in CURE-Lite: diagnose
factual/legal distribution overlap on `D_R`, then test one support-preserving
distribution correction while keeping the decoder, objective, schedule, and
inference fixed. The current scalar hard-matching selector must not be promoted
as CURE.

All `D_V` values are development-selection results. Stage-A does not read
`D_T`. No statement about final generalization, Full CURE, or cross-backbone
effectiveness may be made from this gate alone.

## Cache boundary

The active data path is:

```text
project-owned frozen reference Base
  -> generic D_R/D_V caches: probability p_b + detached feature F_b
  -> factual state and decoder-visible legal-target catalogs
  -> fixed residual decoder training
  -> A / Base@B / F / F× / U / M development evaluation
```

The CURE-Lite core consumes the generic `(p_b, F_b)` contract and does not
import a detector architecture. This is a clean software boundary, not yet
evidence of cross-backbone transfer. Such evidence belongs to the later Full
CURE stage.

## Artifact versions

Historical identifiers remain unchanged:

- Stage-A scientific config v4;
- completed Stage-A run `cure-lite-stage-a-run-v7`;
- protocol-freeze v2;
- seed/master registry v6;
- efficiency receipt v1;
- calibrated deployment v2;
- CLI summary v4;
- replayed assessment v3; and
- historical decoder artifact `cure-lite-decoder-artifact-v2`.

The new M artifact uses `cure-lite-decoder-artifact-v3` so that it can bind the
alignment catalog. The M extension has its own v1 run/config/reference/
alignment/calibration/results receipts and does not reinterpret a historical
schema.

## Install and public API

Python 3.10 or newer is required.

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

See [STAGE.md](STAGE.md) for the stage boundary,
[VALIDATION.md](VALIDATION.md) for the exact evidence gates, and
[DATA_LOADING.md](DATA_LOADING.md) for dataset and cache contracts.
