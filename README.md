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

Full CURE has not been designed or implemented. No CURE-Lite candidate has
earned cross-detector authorization, and no formal
cross-backbone experiment has started. The repository retains a historical
MSHNet adapter, but it is outside the paired core and is not current integration
evidence. The project-owned reference U-Net is only the frozen Base used to
produce the current Stage-A caches; it is not a CURE component or an innovation
claim.

## Current status

### Current candidate: v24 GCR-PACRE; formal `D_R` remains undecided

v24 introduces **Gated Common-Residual PACRE (GCR-PACRE)** as one frozen
mechanistic change to v23: the PACRE phase-residual interaction is retained,
while its amplitude is gated by parameter-free, flip-even common evidence.
The model definition, independent oracle, dataset-free checks, training and
verification chain, and downstream protocol scaffolding are implemented.

The current evidence boundary is:

```text
v24 core implementation                 COMPLETE
dataset-free / efficiency               PASS (sealed CUDA receipt)
D_R structural r1                       OBSERVABILITY LOST; NO DECISION
D_R structural r2 scientific preaccess  AUTHORIZED
r2 unit realization                     RECOVERED AND SEALED
r2 preaccess compatibility              BLOCKED BEFORE MATERIALIZATION
D_R payload access in r2                NONE
D_R OOF, bounded-400, Formal800         BLOCKED
D_V / D_T                               BLOCKED; payloads not authorized
```

The r1 structural attempt crossed its execution marker, but no authentic
terminal receipt, process identity, or recoverable tool session remained when
execution observability was restored. It is therefore neither a structural
PASS nor a scientific FAIL. Generated-data success, supervisor scaffolding, and
process exit state do not substitute for decision-bearing real-data evidence.

Fresh r2 now has a sealed scientific preaccess authorization and a recovered,
sealed actual-unit realization. Its execution bridge has nevertheless stopped
at successive preaccess compatibility checks. The recorded failures occurred
before result materialization, runtime launch, or scientific-attempt
consumption; no `D_R`, `D_V`, or `D_T` payload was accessed. Consequently r2
has not produced a structural decision and must not be described as either
PASS or FAIL. OOF, bounded training, Formal800, `D_V`, and `D_T` remain blocked
until the explicitly authorized compatibility chain completes and one authentic
r2 structural receipt passes.

See
[the v24 design and execution record](CURE_Lite_PACRE_VC_v23_DV_FAIL归因与v24_GCR_PACRE修改方案.md)
,
[the r2 execution-closure record](CURE_Lite_v24_r2执行阻断归因与user_systemd环境闭合方案.md),
and
[the v24 protocol directory](protocols/IRSTD-1K/gcr_pacre_v24/).

### Historical node: v23 PACRE-VC completed Formal800 and failed `D_V`

v23 adopts PACRE-VC, a verifier-corrected residual/value-centering formulation:

```text
r = p - p_base
field = r - stopgrad(r_target)
completion = (field < 0) and not occupancy
```

Its dataset-free checks and unique real `D_R` gate passed `13 / 13`. The
authorized seed-42 Formal800 run then completed from scratch with no resume,
retry, or intermediate checkpoint:

```text
800 epochs x 40 steps = 32,000 updates
D_V/D_T training-time access = none
```

The single adaptive `D_V` evaluation used the frozen 51-point Base threshold
grid and the fixed PACRE-VC field threshold `0.0`:

| metric | Base@A | best Base@B (0.14) | Base@A + CURE |
|---|---:|---:|---:|
| true targets / 170 | 147 | **150** | 149 |
| recovered anchor misses / 23 | 0 | **3** | 2 |
| Pd | 0.864706 | **0.882353** | 0.876471 |
| mIoU | **0.609559** | 0.607629 | 0.602141 |
| nIoU | **0.565328** | 0.564014 | 0.560866 |

Retention was `1.0`, and all three frozen false-alarm constraints passed.
However, the gate compared CURE against the endpoint-wise best valid Base.
CURE was `-1` on both true targets and recovered misses, and its mIoU and nIoU
were also lower. The terminal decision is therefore:

```text
PACRE_V23_FORMAL_D_V_ADAPTIVE_FAIL
  -> preserve the completed D_R, Formal800, and D_V evidence
  -> do not tune, retry, or reopen D_V
  -> do not access D_T
  -> do not authorize Full CURE or cross-backbone evaluation
```

An append-only verifier corrigendum corrected an exact-key whitelist omission
without changing the model, thresholds, performance gate, or four frozen D_V
terminal files. Independent verification confirmed the same failing decision.

See
[the v23 formal result](CURE_Lite_v23_PACRE_VC_正式结果.md),
[the v23 implementation plan](CURE_Lite_v22_PACRE_D_R失败修订分析与v23_PACRE_VC实施方案.md),
[the verifier corrigendum](audits/pacre_v23_dv_verifier_corrigendum_v1/preregistration.md),
and
[the consolidated result ledger](CURE_Lite_全部结果与当前研究结论.md).

### Historical node: paired CURE-Lite core

The marginal-selection route studied by v0.1/v0.2 and P0 is frozen as
historical negative evidence. The later paired CURE-Lite candidate kept the same
frozen Base, residual decoder, and one-pass inference graph, but changed the
learning object to a same-source coverage response:

```text
Q_theta(F, Pi(O))
  -> same frozen feature with O+ / O-
  -> Delta_g Q = Q_minus - Q_plus
  -> factual-miss anchor + factual-no-miss anchor + paired difference loss
```

The additive pair catalog, `PairExample`/`PairBatch`, paired loss, paired train
step, deterministic `800 x 40` schedule, and toy learnability checks are
implemented. The sealed real `D_R` catalog contains 206 clean-positive,
16 component-null, and 160 identity-null pairs. The real paired preflight,
matched-control static preflight, proposed 400-update bounded learnability,
spatial-tail companion, and 8-control bounded engineering execution all closed
with byte-identical successful replays. The full `800 x 40` factual/pair
schedule, complete runtime-input binding, matched-control provider, common
120-image `D_V` comparison protocol, separate create-only paired artifact
schema, and no-resume formal runner are implemented. Strict `D_R`-only dry
validation and the full `723 passed` regression suite completed.

Wave A is now complete. All four frozen `800 x 40` runs
(`paired_difference` and `independent_endpoint`, seeds 42 and 43) finished
from fresh initialization through the create-only runner. No run restored or
continued a checkpoint. After all four training artifacts were complete, the
single predeclared `D_V` reveal was performed under the common protocol.

The formal decision is `PERFORMANCE_FAIL`:

| seed | paired true targets | paired recovered fixed misses | best cumulative comparator | Wave-A result |
|---:|---:|---:|---:|:---|
| 42 | 147 / 170 | 0 / 23 | 154 / 170 and 7 / 23 | fail |
| 43 | 152 / 170 | 5 / 23 | 152 / 170 and 5 / 23 | fail (tie is not a strict improvement) |

Every compared result passed the frozen false-alarm, covered-target-retention,
and budget constraints. The failure is therefore the predeclared performance
gate, not an execution or constraint failure. Seed 42 is worse than the best
available comparator on both required quantities; seed 43 only ties it.
Per-seed strict improvement was required, so averaging cannot compensate for
either result.

The resulting action is fixed:

```text
PERFORMANCE_FAIL
  -> STOP_AND_PRESERVE_EVIDENCE
  -> freeze this paired CURE-Lite version as a complete negative result
  -> do not authorize Wave B or Wave C
  -> do not authorize Full CURE or cross-backbone integration
```

`D_T` remains unread. The one-time `D_V` reveal does not alter the earlier
meaning of the bounded evidence: the proposed bounded pass established only
deterministic learnability on one fixed `D_R` micro-population, and the control
pass established only executable matched controls under the frozen budget.

The overall research mainline remains unchanged:

```text
CURE-Lite mechanism design and development verification
  -> frozen confirmation only after a Lite mechanism passes
  -> Full CURE design
  -> cross-backbone and multi-dataset verification
```

What is frozen as negative evidence is this paired CURE-Lite version, not the
overall CURE research direction. Any subsequent work must remain at the
CURE-Lite mechanism-design stage until a newly specified mechanism earns a
fresh development gate; the current result cannot be used to skip directly to
Full CURE, Wave B/C, DNANet, UIUNet, MSHNet, SCTransNet, or other backbones.

### Historical v0.1: completed diagnostic stage

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

### Historical v0.2: M completed; the mechanism signal is negative

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

These historical runs remain valid negative evidence. The current `699 passed`
suite additionally covers the later paired implementation; tests do not replace
the formal performance result.

### Historical D_R-only P0: geometry gate failed before S

The next step was not another training run. A frozen, create-only P0 protocol
first tested whether the existing synthetic population was suitable for a
support-preserving marginal distribution correction. It accepts only the
bound `D_R` manifest/cache and has no `D_V`, training, calibration, inference,
or backbone argument.

P0-A traces every native 512-grid GT component through the frozen 256-grid
nearest-neighbour resize. The current catalog gives:

| P0-A item | result |
|---|---:|
| native / evaluation targets | 244 / 242 |
| native disappearances | 2 |
| evaluation merges | 1 |
| native splits | 1 |
| legal targets without one-to-one lineage | 2 |
| legal area-ratio failures | 1 |
| legal centroid-shift failures | 0 |

P0-A therefore fails. In the sequential protocol this makes P0-B and P0-C
secondary diagnostics only, and leaves P0-D's candidate S unevaluated. Their
formal values are `null`, no candidate marginal distribution is constructed,
and no new model is trained.

The `D_R` diagnostic population is 32 reachable factual misses in 24 source
groups, plus one separately recorded unreachable miss. This is intentionally
different from the 23 fixed `D_V` anchor misses used by the historical
performance comparison.

Two independent create-only executions produced byte-identical receipts and
completion records:

- `runs/irstd1k_stage_a_seed42/cure_lite_p0_v1_r3`;
- `runs/irstd1k_stage_a_seed42/cure_lite_p0_v1_r4`.

The exact frozen entry point is:

```bash
python tools/run_p0_diagnostics.py \
  --manifest protocols/IRSTD-1K/stage_a_seed42/manifest.json \
  --state-index runs/irstd1k_stage_a_seed42/cure_lite_stage_a_fx_v3/d_r/state_cache/index.json \
  --config protocols/IRSTD-1K/p0_v1/p0_config.json \
  --output /path/to/new_create_only_p0_run
```

The decision at that protocol stage was `rebuild_synthetic_target_extraction`.
Later failure attribution and hypothesis review stopped the entire marginal
reweighting/S route and redefined the core learning object as same-source
\(\Delta_gQ\). The current next step is therefore paired `D_R` preflight and
bounded learnability, not another P0/S run. This still does not authorize
formal paired training, `D_V` evaluation, Full CURE, or cross-backbone work.

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
the Lite-stage evidence gate. The scalar hard-matching selector must not be
promoted as CURE. The later P0/failure-attribution sequence stopped marginal
distribution correction; the current paired route instead tests a coupled
same-source coverage response without reinterpreting the old M result.

All `D_V` values are development-selection results. Stage-A does not read
`D_T`. No statement about final generalization, Full CURE, or cross-backbone
effectiveness may be made from this gate alone.

## Cache boundary

The current paired data path is:

```text
project-owned frozen reference Base
  -> generic D_R cache: probability p_b + detached feature F_b
  -> factual anchors + same-source clean/null pair catalog
  -> the same fixed residual decoder
  -> L_F+ + L_F0 + L_Delta
  -> one-pass hard-union inference remains unchanged
```

The historical `D_R/D_V -> A/Base@B/F/F×/U/M` path and artifacts remain
unchanged for comparison.

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

### Repository guide

- `cure_lite/`: core models, decoders, training components, cache contracts,
  reference Base, and experiment runners;
- `tools/`: create-only protocol, training, evaluation, audit, and replay
  entry points;
- `protocols/IRSTD-1K/`: frozen configurations, decision rules, receipts, and
  sealed result records;
- `tests/` and `tests_v8/` through `tests_v12/`: shared regression tests and
  mechanism-version-specific verification;
- `datasets/`: local benchmark data used by the frozen manifests; and
- top-level `CURE_Lite_*.md` files: model designs, formal results, audits, and
  next-step research records.

Generated runs, build products, caches, and temporary replay directories are
excluded from version control. Protocol receipts and formal result records are
tracked because they define the evidence trail summarized above.

See [STAGE.md](STAGE.md) for the stage boundary,
[VALIDATION.md](VALIDATION.md) for the exact evidence gates, and
[DATA_LOADING.md](DATA_LOADING.md) for dataset and cache contracts.
