# CURE-Lite validation gates for the CURE project

## Claim boundary

CURE-Lite is the first empirical gate for CURE. It currently tests one
mechanism: uniform deletion of one legal covered component from a frozen Base
occupancy, with the Base feature unchanged, followed by positive residual
learning. It does not yet contain the additional mechanism of Full CURE.

Software correctness, toy closure, and `D_V` development scores are different
kinds of evidence and must not be presented as interchangeable.

## Gate 0: semantics and software

The test suite must cover frozen-Base behavior, 8-connected-component (CC8)
occupancy, deterministic matching, deletion legality, supervision masks,
decoder-only optimization, monotone union, cache identity, calibration,
evaluation, and the formal runner.

Two feature-grid details are part of the method contract:

- full-resolution occupancy is projected with adaptive max pooling;
- a legal source-grid deletion is usable for synthetic learning only if it
  changes the projected occupancy consumed by the decoder.

The latest recorded active suite reports `268 passed` with the historical
MSHNet-adapter test excluded. This is software evidence only; it is not a
completed result for the current five-way protocol.

The active artifact versions are deliberately separate: Stage-A scientific
config v4, completed run v7, seed/master registry v6, efficiency receipt v1,
calibrated deployment v2, CLI summary v4, and replayed assessment v3. A test or
loader must validate the version belonging to the artifact it consumes.

Registry v6 contains no post-run, caller-selected success thresholds. The
single-seed rule that is actually executed is bound before launch by config
v4, decision-rule v3, and protocol-freeze v2. Bootstrap replicates and
confidence intervals enter only after the required paired multi-seed runs
exist; the pilot does not fabricate or imply them.

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

The five-way config, decision rule, and protocol-freeze v2 are prepared, but
the protocol has no completed formal result. The terminated incomplete
four-way run must not be resumed or used as evidence for this gate.

### Frozen-Base condition

The current Base is the project-owned reference U-Net. Its active configuration
specifies 800 epochs on `D_B-fit`, selection on `D_B-select`, and then fixed
cache export. Its sole role is to produce generic probability/feature caches.
It is not a component of CURE-Lite and does not support a cross-backbone claim.
The current CURE-Lite F/F×/U decoder schedule is separately configured as 20
epochs with 40 steps per epoch; it is not an 800-epoch decoder schedule.

The formal runner consumes the `D_R` and `D_V` cache indexes, manifest,
Stage-A configuration, and a provider-verified Base-run identity. The current
CLI obtains that identity from the project reference-Base loader; the
detector-neutral runner itself receives only the generic identity contract and
does not import the reference U-Net or MSHNet. It can later consume another
detector's caches only after that detector has a provider loader or adapter
shown to satisfy the same preprocessing, identity, probability, feature, and
grid contracts. This interface is not yet evidence of arbitrary-detector
plug-and-play behavior.

### Pre-training checks in five-way Stage-A v4

Before training the three F/F×/U decoders, the runner must:

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

The formal mechanism gate compares:

1. `A`: decoder-free frozen Base anchor;
2. `Base@B`: decoder-free relaxed Base threshold under the same total
   constraints;
3. `F`: factual-only residual learning;
4. `F×`: exposure-matched factual-positive replacement in the third loss slot;
5. `U`: factual learning plus uniform projected-visible legal deletion.

All five use the same manifest, frozen Base, anchor, matching rule, declared
threshold-grid protocol, and evaluation constraints. A/Base@B do not own
decoder artifacts; F/F×/U share the residual-threshold grid, one decoder
initialization, and common optimizer/training configuration. F× matches U's
third-slot activation,
state-forwards, batch size, and loss weight, while its deletion-synthetic pool
is required to remain empty. This is update/state-exposure matching; it does
not guarantee matching of positive-pixel mass, target size, difficulty, or
feature distributions.

Threshold calibration must use the shared exact candidate ledger. Fixed GT
instances, anchor instances, anchor misses, and reachability are constructed
once per image; every optimized candidate and the final deterministic
selection must compare exactly equal to the scalar reference implementation.
Worker count is execution-only and is not part of the scientific protocol.
Selection maximizes total Pd under the frozen constraints; exact Pd ties prefer
higher retention, then lower pixel/raw-background/component false alarms, and
finally the more conservative higher threshold. No auxiliary diagnostic ratio
participates in this selection.

### Required development outputs

The headline development outputs are total Pd, retention, pixel FA, RawBG-FA,
FP components/MP, mIoU, nIoU, legal-candidate availability,
projected-visible legal availability, full-GT closure rate, incremental
parameters, Conv2d MACs/FLOPs, and latency. Other internal diagnostic fields
are excluded from the outward score record; they are not headline success
metrics or substitutes for the predeclared Pd gate.
Also retain the Base-cache indexes, decoder artifacts, calibration record,
support receipt, efficiency receipt, and completion record needed to reproduce
the run. The efficiency receipt measures only the deployed `U` decoder on
all-zero synthetic tensors. Static counts are replayable; latency and memory
remain environment-dependent observations and do not participate in method
selection. The fixed measurement uses batch size 1, 10 warm-up forwards and 50
measured forwards, reporting median/p95 latency. Static counts cover Conv2d
only; CPU memory is unavailable and CUDA memory is allocator-specific.

The Stage-A seed registry v6 must be generated from a loaded completed run via
`build_seed_registry_from_stage_a_run`. It binds the detector-neutral Base run
identity, library-computed Base state fingerprint, exact D_R/D_V cache-index
fingerprints and file hashes, Stage-A completion fingerprint, five selected
protocols, and the efficiency receipt. A project reference-Base identity is
accepted only after `load_verified_reference_base_run_identity` has replayed
its completion, selection, checkpoint, and source records. The master registry
derives device type, warm-up count, and repetition count from its seed
registries; these values are not supplied through a second configuration path.
The calibrated deployment v2 constructor independently hashes the online Base
and requires equality with the state captured at cache time; the provider's
declared identity cannot substitute for that check.

Every number selected or evaluated on `D_V` must be labelled a **development
selection score**. It is used to decide whether to revise CURE-Lite or proceed;
it is not an official-test result and cannot alone justify an ICLR claim.
`D_T` images and labels are not read in Stage-A.

### Gate decision

The predeclared primary metric is total object-level Pd. Any auxiliary fields
are internal diagnostics only. A positive Stage-A signal
requires every method to satisfy the configured constraints and all three
strict inequalities `Pd(U) > Pd(Base@B)`, `Pd(U) > Pd(F)`, and
`Pd(U) > Pd(F×)`. The CLI records all three deltas and this Boolean decision.
mIoU/nIoU changes are reported as a secondary quality screen and cannot replace
the Pd rule.

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

After Full CURE is defined, evaluate adapter-mediated integration with DNANet,
UIUNet, MSHNet, and SCTransNet. The intended core algorithm and evaluation
rules should remain shared. This future multi-backbone evidence, together with
multi-dataset and multi-seed results, is required before describing CURE as a
general post-detector plugin; it is not part of the current Lite gate.
