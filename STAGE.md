# CURE project stage

## Active stage

The active method in this repository is **CURE-Lite v0.1**. It is a
backbone-independent reference scaffold for testing one minimal hypothesis:

> Can uniform legal coverage-state deletion train a lightweight positive
> residual decoder to recover held-out factual misses under the same total
> false-alarm budget?

The repository does **not** currently contain an active Full CURE method.

## Active code boundary

Only the root modules and `train/` define the current method, including:

- `frozen_base.py`, `occupancy.py`, `matching.py`, and `intervention.py`;
- `supervision.py`, `decoder.py`, `losses.py`, and `model.py`;
- `train/`, calibration, metrics, caches, provenance, and the toy provider.

The active package must not import a Full CURE namespace, propensity model,
miss-odds sampler, or Full-CURE experiment policy.

## Gate before any Full CURE design

1. Freeze the CURE-Lite method and protocol.
2. Pass all software and toy-pipeline tests.
3. Connect one frozen real IRSTD detector through a thin adapter.
4. Run the minimum Stage-A comparison: Base, Factual-only residual, and
   Uniform-Legal CURE-Lite under the same total FA budget.
5. Confirm that gains occur on held-out factual misses and are not explained
   only by lowering the base threshold or by synthetic-state shortcuts.

The next decision is evidence-dependent:

- If CURE-Lite fails, stop or redesign the core mechanism.
- If Uniform-Legal is sufficient, keep the final method simple.
- Only if CURE-Lite works but a measured synthetic/factual distribution gap
  limits transfer may a separate Full CURE project be designed.

## Archived future candidate

An early Full CURE v2 candidate was previously implemented, but it was created
before the CURE-Lite gate. It is not part of the active package. Its exact code
and tests remain recoverable from Git commit `94759d2` and must not be restored
to the active branch before the gate above passes.
