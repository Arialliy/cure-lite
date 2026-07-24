# CURE project stages

## One project, ordered design stages

The research target is **CURE**, a detector-independent correction method for
IRSTD. **CURE-Lite is the first stage used to establish the smallest viable
mechanism.** It is neither a separate final model nor a reduced substitute for
Full CURE.

```text
Stage 1: CURE-Lite
  v0.1 uniform legal selection (completed diagnostic)
  v0.2 miss-aligned legal selection M (completed; signal negative)
  P0 D_R-only prerequisite audit (completed; P0-A geometry failed)
       |
       | only after the Lite gate is supported
       v
Stage 2: Full CURE design
       |
       | only after Full CURE is defined
       v
Stage 3: cross-backbone and multi-dataset verification
```

Full CURE and the later DNANet/UIUNet/MSHNet/SCTransNet integrations have not
started.

## Stage 1a: what v0.1 established

v0.1 compared the frozen-Base references `A` and `Base@B` with three matched
residual decoders:

- `F`: factual-only learning;
- `F×`: factual-positive exposure control;
- `U`: factual learning plus uniform decoder-visible legal-target selection.

Both formal 800-epoch, 40-step-per-epoch `fx_v3` runs completed:

| seed | Pd(F) | Pd(U) | v0.1 strict U rule |
|---:|---:|---:|:---|
| 42 | 0.905882 | 0.888235 | not met |
| 43 | 0.894118 | 0.894118 | not met |

The strict rule required `Pd(U)` to be greater than the Pd of `Base@B`, `F`,
and `F×` while all constraints held. The observed result does not support
uniform selection. It does not reject the complete CURE direction; it
identifies target selection as the next hypothesis to correct.

The v0.1 run roots, caches, decoder artifacts, calibration receipts, and result
receipts are historical references. They remain unchanged.

## Stage 1b: the v0.2 M mechanism

For each reachable factual miss, M calculates a scalar descriptor from the
frozen feature:

```text
z(target) = log1p(positive-region feature RMS)
```

The source-grid target mask is projected by adaptive average weighting onto the
feature grid. The RMS is normalized by channel count. M compares the factual
descriptor with the descriptors of all decoder-visible legal targets in the
complete `D_R` catalog, uses a `1e6` fixed-point quantization for absolute
distance, and selects the globally nearest target with deterministic identity
tie-breaking. Reusing a legal target for several factual misses is intentional.

Only the synthetic-target selector changes from U to M. These parts stay fixed:

- the frozen Base and cache contract;
- occupancy, matching, and reachability;
- the definition of a decoder-visible legal target;
- factual, no-miss, and synthetic supervision;
- residual decoder architecture and initialization contract;
- loss, branch weights, optimizer, batch sizes, epochs, and updates; and
- monotone-union inference and evaluation constraints.

This creates a direct test of one explanation for v0.1: uniform legal examples
may be valid but distributed differently from factual misses. No second decoder
branch, attention block, auxiliary loss, or changed inference module is added.

## Mapping audit before training

The real `D_R` mapping is:

| item | value |
|---|---:|
| reachable factual misses mapped | 32 / 32 |
| unique selected legal targets | 16 |
| unique selected legal sources | 16 |
| maximum target reuse | 7 |
| mean quantized distance | 31342.1875 |
| maximum quantized distance | 148544 |

The bound catalog fingerprint is
`f53b65e9962642e2705a6641861d039245c507e6703f8846242376bb43411f73`.

This audit proves that the rule is deterministic and executable for the real
catalog. The completed performance experiment is reported separately below.

## Stage 1b execution contract

`tools/run_stage_a_m_extension.py` receives a completed v0.1 Stage-A directory,
the original manifest, a new output directory, and a device. It:

1. verifies and snapshots the completed v0.1 reference;
2. loads the exact historical `D_R/D_V` cache identities;
3. reconstructs and binds the M alignment catalog;
4. reuses historical F/F×/U without retraining;
5. trains only M for the inherited `800 × 40` schedule;
6. calibrates only M on the inherited frozen threshold grid;
7. evaluates U only once at its historical selected threshold;
8. compares M with the copied historical formal results and the fixed-point U
   recovery counts; and
9. publishes a new create-only extension.

The M runner never writes into the v0.1 root and does not accept `D_T`. The
output directory must be new.

## Stage 1b decision

The single-seed development signal is positive only when:

- all methods satisfy the configured constraints;
- `Pd(M)` is strictly greater than `Pd(Base@B)`, `Pd(F)`, `Pd(F×)`, and
  `Pd(U)`;
- M recovers more anchor misses than U at its historical threshold; and
- the M/U recovery denominators match.

The comparison with U asks whether miss-aligned selection improves on uniform
selection. The comparisons with `Base@B`, `F`, and `F×` prevent that narrower
gain from being presented as complete CURE-Lite success.

The two completed runs give:

| seed | Pd(Base@B) | Pd(F) | Pd(F×) | Pd(U) | Pd(M) | recovered U / M | decision |
|---:|---:|---:|---:|---:|---:|:---:|:---|
| 42 | 0.882353 | 0.905882 | 0.876471 | 0.888235 | 0.882353 | 4 / 3 of 23 | negative |
| 43 | 0.882353 | 0.894118 | 0.864706 | 0.894118 | 0.876471 | 5 / 2 of 23 | negative |

Every method satisfies the configured constraints, but M does not strictly
exceed Base@B, F, and U and recovers fewer anchor misses than U in both seeds.
Both formal gate receipts therefore record `mechanism_signal = false`.

This is a valid negative mechanism result, not an interrupted or invalid run.
Each M decoder completed `800 x 40 = 32,000` updates, and independent checks
reproduced the artifact inventory, completion record, historical references,
selected threshold, fixed-point U recovery, and M metrics.

The cross-seed mapping diagnosis explains why the next revision must remain in
CURE-Lite. Thirty-two mappings select only 16 of 209 legal targets; the largest
reuse is 7 and the full-mapping Kish effective sample size is 9.48. Because the
factual and synthetic losses are summed independently, M's pairing affects
training only through the marginal frequency of selected legal targets. A hard
one-dimensional nearest selector therefore concentrates support without
creating a direct pairwise learning signal.

All reported values are `D_V` development results. `D_T` remains unused until
a CURE-Lite mechanism is frozen.

## P0 prerequisite audit after v0.2

The frozen P0 command accepts only the bound `D_R` manifest, state cache,
configuration, and a new output path. It does not expose a `D_V`, training,
calibration, inference, device, or backbone option.

P0 is sequential:

```text
P0-A native 512 -> evaluation 256 geometry
  -> P0-B common support, only as a formal gate if A passes
  -> P0-C grouped distinguishability, only as a formal gate if A passes
  -> P0-D full 800 x 40 x 4 exposure replay
  -> candidate S may be constructed only if A/B/C pass
```

The current P0-A result is:

- 244 native targets become 242 evaluation targets;
- two native targets disappear;
- one evaluation target merges two native targets;
- one native target splits;
- two decoder-visible legal targets lack one-to-one lineage;
- one legal target violates the frozen area-ratio gate; and
- no legal target violates the centroid-shift gate.

Therefore P0-A is false. P0-B and P0-C have `null` formal decisions and retain
their values only as secondary diagnostics. P0-D replays historical U/M
exposure, but candidate S remains `not_evaluated`; no distribution, decoder,
or model is trained. The only authorized route is
`rebuild_synthetic_target_extraction`.

The P0 population is `D_R`: 32 reachable factual misses in 24 source groups
and one separately recorded unreachable miss. The historical value 23 is a
`D_V` anchor-miss denominator and is not used by P0.

## Transition to Full CURE

Full CURE design begins only after CURE-Lite passes its mechanism gate with
adequate repeatability. M is unsupported in both completed runs, and P0-A has
shown that the present synthetic target geometry must be repaired before any
marginal distribution correction is eligible for evaluation. After the
repaired population passes P0-A, P0-B/P0-C must be evaluated formally; only
then may P0-D construct and audit candidate S.

Only after Full CURE is defined should thin adapters be built for DNANet,
UIUNet, MSHNet, and SCTransNet. Those experiments test whether the unchanged
CURE core transfers across detector families; they are not part of the current
Stage 1 claim.

## Software and artifact status

The current full suite reports `376 passed`. Tests establish implementation
consistency, not M performance.

Historical identifiers remain as recorded: Stage-A config v4, completed run
v7, protocol-freeze v2, seed/master registry v6, efficiency receipt v1,
calibrated deployment v2, CLI summary v4, replayed assessment v3, and decoder
artifact v2. M uses decoder artifact v3 and separate M-extension v1 receipts;
the historical identifiers are not renamed.
