# CURE-Lite validation gates for CURE

## Claim boundary

CURE-Lite is the first mechanism gate for CURE. The v0.1 uniform-selection and
v0.2 M experiments are complete; neither mechanism met its predeclared
development rule. Full CURE and cross-backbone validation have not started.

Four evidence types must remain separate:

1. software tests show that the implementation follows its contracts;
2. the mapping audit shows that M can construct all required training choices;
3. `D_V` scores test the current development hypothesis; and
4. later independent evaluation is required for a generalization claim.

The first two do not imply the third, and none of the first three alone implies
the fourth.

## Gate 0: software consistency

The full test suite currently reports:

```text
376 passed
```

Tests cover the frozen-Base boundary, CC8 occupancy, deterministic matching,
decoder-visible legal targets, supervision, decoder-only optimization,
monotone union, cache identity, v2/v3 decoder artifacts, historical reference
loading, M-only training, single-method calibration, fixed-point U evaluation,
create-only output publication, D_R-only P0 protocol validation, zero-MAD
support preservation, matched group-MMD, and exact exposure-audit guards.

Passing Gate 0 is required before an experiment starts, but it is not an M
performance result.

Historical artifact identifiers remain distinct:

- Stage-A scientific config v4;
- completed Stage-A run v7;
- protocol-freeze v2;
- seed/master registry v6;
- efficiency receipt v1;
- calibrated deployment v2;
- CLI summary v4;
- replayed assessment v3; and
- decoder artifact v2.

M uses decoder artifact v3 and M-extension v1 receipts. Loaders validate the
schema belonging to the artifact they consume; the historical identifiers are
not changed.

## Gate 1: v0.1 diagnostic evidence

The completed v0.1 `fx_v3` results are:

| seed | Pd(F) | Pd(U) | interpretation |
|---:|---:|---:|:---|
| 42 | 0.905882 | 0.888235 | U did not exceed F |
| 43 | 0.894118 | 0.894118 | U tied F |

v0.1 required U to strictly exceed `Base@B`, `F`, and `F×`, with every method
inside the same configured constraints. That rule was not met. The appropriate
conclusion is limited: uniform legal-target selection is unsupported in these
development runs. It is not a claim that the complete CURE direction failed.

## Gate 2: M mapping audit

M is defined by the following deterministic rule:

```text
reachable factual miss
  -> positive-region log1p(frozen-feature RMS)
  -> compare with every decoder-visible legal target in D_R
  -> select the globally nearest quantized descriptor
  -> canonical identity tie-break
```

Distance is the absolute descriptor difference quantized at `1e6`. Candidate
reuse is permitted. M changes only synthetic-target selection; legal-target
definition, supervision, decoder, loss, branch weights, optimizer, training
schedule, and inference remain fixed.

The current real-catalog audit must reproduce exactly:

| check | expected value |
|---|---:|
| mapped reachable factual misses | 32 / 32 |
| unique selected legal targets | 16 |
| unique selected legal sources | 16 |
| maximum target reuse | 7 |
| mean quantized distance | 31342.1875 |
| maximum quantized distance | 148544 |

The expected alignment-catalog fingerprint is
`f53b65e9962642e2705a6641861d039245c507e6703f8846242376bb43411f73`.

Passing Gate 2 proves mapping completeness, determinism, and reproducibility.
It does not establish that M improves any evaluation metric.

## Gate 3: formal M extension

### Frozen comparison

The method order is:

```text
A / Base@B / F / F× / U / M
```

The M runner must use the same manifest, frozen Base, `D_R/D_V` caches, anchor,
matching rules, training budget, threshold grid, and evaluation constraints as
the completed v0.1 reference.

The runner is intentionally incremental:

- historical F/F×/U artifacts are loaded and verified, not retrained;
- only M is trained for `800` epochs and `40` steps per epoch;
- only M is calibrated over the frozen residual-threshold grid;
- historical A/Base@B/F/F×/U formal metrics are reused;
- U is evaluated once at its already selected historical threshold only to
  obtain directly comparable recovery counts; and
- a new sibling output is published without changing the reference run.

The command is:

```bash
python tools/run_stage_a_m_extension.py \
  --reference-stage-a /path/to/completed_fx_v3 \
  --manifest protocols/IRSTD-1K/stage_a_seed42/manifest.json \
  --output /path/to/new_m_extension \
  --device cuda:0 \
  --calibration-workers 24
```

The output directory must not already exist. A partial output is not continued;
the next execution uses a different new directory.

### Formal outputs

For every method, retain the formal development fields:

- total object-level Pd;
- mIoU and nIoU;
- pixel FA;
- FP components per megapixel;
- raw-background FA;
- covered-target retention; and
- constraint status.

For U at its historical threshold and for M at its selected threshold, also
retain:

- recovered anchor misses;
- total anchor misses;
- recovered reachable anchor misses;
- total reachable anchor misses;
- retained and total anchor-covered targets; and
- gross, net, reachable, and overlap-supported recovery diagnostics.

The alignment receipt, M decoder artifact, M calibration receipt, result
receipt, gate receipt, historical reference snapshot, and completion record
must bind the same catalog, caches, Base state, manifest, and source version.

### Success rule

The M development signal is true only if:

```text
all compared methods satisfy the configured constraints
and Pd(M) > Pd(Base@B)
and Pd(M) > Pd(F)
and Pd(M) > Pd(F×)
and Pd(M) > Pd(U)
and recovered_anchor_misses(M) > recovered_anchor_misses(U@historical)
and the M/U recovery denominators are equal
```

mIoU and nIoU are secondary non-degradation checks. They cannot replace the Pd
or recovery requirements.

Interpret results conservatively:

- `M <= U`: miss-aligned selection did not improve on uniform selection;
- `M > U` but `M <= F` or `M <= F×`: the selector may be better than U, but
  the full CURE-Lite mechanism is not supported;
- `M > F/F×/U` but `M <= Base@B`: threshold relaxation remains a sufficient
  explanation;
- every strict comparison and recovery rule holds: record a positive
  single-seed development signal, then verify repeatability before designing
  Full CURE.

### Observed v0.2 result

| seed | selected threshold | Pd(M) | mIoU(M) | nIoU(M) | misses recovered U / M | mechanism signal |
|---:|---:|---:|---:|---:|:---:|:---|
| 42 | 1.0 | 0.882353 | 0.592637 | 0.549957 | 4 / 3 of 23 | false |
| 43 | 1.0 | 0.876471 | 0.605239 | 0.550010 | 5 / 2 of 23 | false |

All compared methods satisfy the configured constraints. For seed 42, M has
the same Pd as Base@B and lower Pd than F and U. For seed 43, M has lower Pd
than Base@B, F, and U. M also recovers fewer anchor misses than U in both
seeds. Secondary IoU or false-alarm improvements cannot override those failed
primary comparisons.

Both runs are complete and reproducible rather than failed executions:

- 800 epochs and 40 steps per epoch are present for each seed;
- the create-only completion records and all recorded file hashes match;
- the historical five-method snapshots are exact;
- independent calibration replay selects threshold 1.0 again; and
- independent fixed-point evaluation reproduces the U/M recovery counts.

The correct Stage 1b decision is therefore: reject the current v0.2 scalar
hard-matching selector, retain the CURE problem hypothesis as unresolved, and
remain in CURE-Lite for the next single-mechanism test.

## Gate 4: D_R-only P0 prerequisite

P0 is a sequential prerequisite, not another performance experiment:

1. P0-A checks native-512 to evaluation-256 target identity and distortion.
2. P0-B checks grouped kNN support in the handcrafted and decoder-joint
   representations, but becomes formal only if P0-A passes.
3. P0-C checks grouped OOF distinguishability and matched group-level MMD,
   but becomes formal only if P0-A passes.
4. P0-D replays the full `800 x 40 x 4` deterministic U/M exposure schedule.
   Candidate S can be constructed only if P0-A/B/C all pass.

The exact `D_R` population is 32 reachable factual misses in 24 groups, one
separately recorded unreachable miss, and 209 decoder-visible legal targets in
146 groups. A 90% support gate therefore requires `ceil(0.9 x 32) = 29`
covered factual targets. The historical 23-anchor denominator belongs to
`D_V` and must not be substituted into P0.

The current P0-A result is false:

| check | observed |
|---|---:|
| native targets -> evaluation targets | 244 -> 242 |
| native disappearances | 2 |
| evaluation merges | 1 |
| native splits | 1 |
| legal targets without one-to-one lineage | 2 |
| legal area-ratio failures | 1 |
| legal centroid-shift failures | 0 |

Consequently, P0-B and P0-C have `p0_*_pass = null` and
`formal_status = not_evaluated_due_to_p0_a_failure`. Their numerical outputs
are secondary diagnostics and cannot be reported as formal gate outcomes.
P0-D has `p0_d_pass = null`; S is `not_evaluated` and is not integrated into
training. The decision route is `rebuild_synthetic_target_extraction`.

The formal P0 decision does not authorize S training, `D_V` evaluation, Full
CURE, or cross-backbone integration.

## Gate 5: later stages

Every Stage-A value is selected or evaluated on `D_V` and must be labelled a
development result. Stage-A does not read `D_T`.

Only after CURE-Lite is supported should Full CURE be designed from the
remaining measured errors. The immediate work is now narrower: repair the
synthetic target extraction/resize protocol and rerun P0-A. Distribution
correction remains blocked until the repaired population passes the sequential
P0 gates. Only after Full CURE exists should thin adapters and unchanged-core
experiments be conducted with DNANet, UIUNet, MSHNet, and SCTransNet. Those
later experiments, not the current interface alone, are needed to evaluate
cross-backbone behavior.
