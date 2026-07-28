# PACRE-VC v23 adaptive \(D_V\) relative-performance preregistration

Frozen before any PACRE-VC v23 generated-only receipt, real-\(D_R\) run,
Formal800 run, or v23 \(D_V\) tensor access.  Project-level \(D_V\) was
already revealed by the sealed v21 evaluation, so this is explicitly an
adaptive development evaluation and not independent confirmation.

## Entry condition and identity

- Run ID:
  `cure_lite_pacre_v23_vc_formal_d_v_seed42_r1`.
- Output:
  `runs/irstd1k_stage_a_seed42/cure_lite_pacre_v23_vc_formal_d_v_seed42_r1`.
- The exact v23 Formal800 terminal graph must independently verify:
  seed 42, from scratch, 800 epochs, 40 steps per epoch, 32,000 updates,
  final-only safetensors, no resume/retry, and no \(D_V/D_T\) payload access
  during training.
- The evaluated object is that one exact final PACRE-VC model.  No
  intermediate or historical checkpoint is eligible.
- The D_V model-binding API accepts only the exact loaded artifact carried by
  a process-local terminal identity seal issued after the independent
  Formal800 terminal verifier succeeds.  A raw loaded artifact, a generic
  training ledger, or a caller-assembled receipt is not evaluation-eligible.
- Evaluation is fixed to `cuda:0` with exactly one visible CUDA device and
  the already frozen selected-device numerical-runtime lock.  The runner
  must reject CPU execution, a different visible-device topology, or a
  changed runtime lock.
- Inference batch size is fixed to 8 and is recorded in the result contract.
- Evaluation performs no training, optimizer step, model update, calibration
  of the PACRE field, or checkpoint selection.

## Frozen data and controls

- Dataset/split: the existing 120-sample IRSTD-1K \(D_V\) bundle already
  frozen for the v21 comparison.
- Preprocessing, base checkpoint, matching, connected-component convention,
  metrics, and false-alarm accounting are inherited unchanged from the
  frozen v21 common comparison protocol.
- `Base@A`: base probability threshold 0.72 with the existing CC8 rule.
- `Base@B`: the existing base-only 51-point grid
  \(\{0,0.02,\ldots,1.0\}\), selected under the existing base false-addition
  budget.  All 51 values, including values above the 0.72 Base@A anchor,
  must actually be evaluated and persisted in order; an implementation that
  silently truncates the grid at 0.72 is invalid.  The terminal verifier
  rebuilds the 51-row ledger and independently repeats the selection.  PACRE
  output is not involved in this selection.
- Candidate output:

  \[
  O=\mathbf 1[p_b\ge0.72],\qquad
  C=\mathbf 1[\phi<0]\land\neg O,\qquad
  Y=O\lor C.
  \]

  `field == 0` is not completion.  No sigmoid and no PACRE threshold search
  are permitted.

For each endpoint, `best Base` is the maximum valid value across the frozen
Base controls using that endpoint's declared direction.  The comparator set
cannot be changed after candidate results are visible.

## Success contract: strict relative improvement, no arbitrary effect margin

The v23 adaptive \(D_V\) gate passes only if all checks below pass:

1. CURE `true_targets` is strictly greater than best Base;
2. CURE `recovered_anchor_misses` is strictly greater than best Base;
3. CURE mIoU is at least best Base mIoU;
4. CURE nIoU is at least best Base nIoU;
5. retention is exactly 1.0;
6. pixel false-alarm rate is at most \(10^{-4}\);
7. raw-background false-alarm rate is at most \(10^{-4}\);
8. false-positive components per megapixel are at most 100;
9. `budget_violation` is false;
10. \(D_T\) was not accessed.

There is no minimum \(+2\) target or recovered-miss margin.  A reproducible
\(+1\) satisfies the strict-improvement items.  The non-regression and
false-alarm checks prevent a nominal count gain obtained by degrading
segmentation or flooding the background; these are operating-safety
constraints, not arbitrary uplift margins.

The sealed v21 numbers motivated removal of the unsupported \(+2\) rule but
are not v23 predictions or results.  This file contains no v23 metric value.

## Evidence and stop rules

- The official runner claims the fixed output before materializing \(D_V\).
- Normal completion writes exact create-only artifacts and `COMPLETE.json`
  last.  Execution failure keeps `.incomplete`, writes `FAILURE.json`, and is
  not resumed or retried.
- The independent verifier recomputes the decision from persisted raw
  aggregate metrics and exact bindings; stored PASS booleans are not trusted.
- A \(D_V\) PASS is adaptive evidence only.  It may make the frozen final
  model eligible for a future, separately preregistered one-shot \(D_T\)
  confirmation, but this runner does not authorize or access \(D_T\).
- A \(D_V\) FAIL is preserved as the result.  No threshold, comparator,
  checkpoint, metric, or safety budget is changed in response.
