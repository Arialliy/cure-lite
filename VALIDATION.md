# CURE-Lite validation gates for the CURE project

## Claim boundary

CURE-Lite is the first empirical gate for CURE. It currently tests one
mechanism: uniform deletion of one legal covered component from a frozen Base
occupancy, with the Base feature unchanged, followed by positive residual
learning. It does not yet contain the additional mechanism of Full CURE.

Software correctness, toy closure, and `D_V` development scores are different
kinds of evidence and must not be presented as interchangeable.

## Gate 0: semantics and software

The test suite must cover frozen-Base behavior, CC8 occupancy, deterministic
matching, deletion legality, supervision masks, decoder-only optimization,
monotone union, cache identity, calibration, evaluation, and the formal runner.

Two feature-grid details are part of the method contract:

- full-resolution occupancy is projected with adaptive max pooling;
- a legal source-grid deletion is usable for synthetic learning only if it
  changes the projected occupancy consumed by the decoder.

The latest complete suite reports `149 passed`. This is software evidence only.

## Gate 1: deterministic toy mechanism

`tests/test_cure_lite_toy_mechanism_gate.py` verifies the intended mechanism path
and a negative case where the synthetic support does not overlap the factual
miss regime:

```text
frozen toy Base
  -> occupancy and matching
  -> projected-visible legal deletion
  -> factual/no-miss and synthetic supervision
  -> validation-only threshold choice
  -> held-out toy evaluation
```

Passing the positive fixture shows that the code can express the mechanism.
The negative fixture shows that low synthetic training loss alone is not
evidence of factual-miss transfer. Neither result establishes IRSTD accuracy or
conference-level novelty.

## Gate 2: real single-seed CURE-Lite Stage-A

### Frozen-Base condition

The current Base is the project-owned reference U-Net. Its active configuration
specifies 800 epochs on `D_B-fit`, selection on `D_B-select`, and then fixed
cache export. Its sole role is to produce generic probability/feature caches.
It is not a component of CURE-Lite and does not support a cross-backbone claim.

The formal runner consumes only the `D_R` and `D_V` cache indexes plus the
manifest and Stage-A configuration. It does not import the reference U-Net or
MSHNet. The same runner can later consume caches produced by any detector whose
adapter satisfies the common contract.

### Pre-training checks in Stage-A v2

Before training either decoder, the runner must:

1. select the Base anchor using `D_V` and verify that it already meets all
   configured false-alarm and retention constraints;
2. reconstruct the `D_R` training support;
3. count factual-miss images, factual no-miss images, reachable missed targets,
   synthetic images, projected-visible legal candidates, and their fraction;
4. compare those values with the config's `support_requirements`; and
5. write the measured values and requirements to `receipts/support.json`.

If either the anchor or support requirements do not hold, decoder training does
not start. This prevents an uninformative data condition from being interpreted
as a method result.

### Minimum comparison

The real pilot compares:

1. `A`: frozen Base anchor;
2. `Base@B`: relaxed Base threshold under the same total constraints;
3. `F`: factual-only residual learning;
4. `U`: factual learning plus uniform projected-visible legal deletion.

All four use the same manifest, frozen Base, anchor, matching rule, threshold
grids, and evaluation constraints. `F` and `U` share the factual schedule and
decoder initialization. However, `U` receives additional synthetic examples,
so equal factual exposure is not equal total exposure. Before attributing a
gain specifically to intervention in a submission, add `F×`, an
exposure-matched factual control with the same overall updates/examples as `U`.

### Required development outputs

Report NetRMR, total Pd, GrossRMR, retention, pixel FA, RawBG-FA, FP
components/MP, mIoU, nIoU, legal-candidate availability, projected-visible
legal availability, full-GT closure rate, incremental parameters, and
latency. Also retain the Base-cache indexes, decoder artifacts, calibration
record, support receipt, and completion record needed to reproduce the run.

Every number selected or evaluated on `D_V` must be labelled a **development
selection score**. It is used to decide whether to revise CURE-Lite or proceed;
it is not an official-test result and cannot alone justify an ICLR claim.
`D_T` images and labels are not read in Stage-A.

### Gate decision

The predeclared primary metric is total object-level Pd. For one fixed anchor,
Pd and NetRMR have the same ordering, but Pd is used for the decision because
it is a standard IRSTD metric; NetRMR remains diagnostic. A positive Stage-A
signal requires every method to satisfy the configured constraints and both
strict inequalities `Pd(U) > Pd(Base@B)` and `Pd(U) > Pd(F)`. The CLI records
the two deltas and this Boolean decision. mIoU/nIoU changes are reported as a
secondary quality screen and cannot replace the Pd rule.

- If `Pd(U) <= Pd(Base@B)`, the evidence does not distinguish CURE-Lite from
  Base-threshold relaxation.
- If `Pd(U) <= Pd(F)`, the current pilot does not show added value from the
  legal intervention.
- If `U` beats `F` but not exposure-matched `F×`, the apparent benefit can be
  explained by additional training exposure.
- If `U` beats `Base@B`, `F`, and `F×` by Pd while satisfying every configured
  constraint, repeat with paired seeds before making a stable mechanism claim.

## Gate 3: design Full CURE

Full CURE is the next project stage after CURE-Lite succeeds; it is not an
independent side project. The Lite experiments must first quantify the
remaining mismatch, such as inadequate synthetic/factual overlap or variation
across Base detectors. Only a mechanism that directly addresses that measured
mismatch enters Full CURE.

After Full CURE is defined, validate it as a post-detector plugin through thin
adapters for DNANet, UIUNet, MSHNet, and SCTransNet. The core algorithm and
evaluation rules remain shared. This multi-backbone evidence, together with
multi-dataset and multi-seed results, is required to support generality; it is
not part of the current Lite gate.
