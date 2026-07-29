# GCR-PACRE v24 protocol infrastructure

This directory freezes protocol rules only. It does not authorize or execute
OOF4, bounded-400, Formal800, \(D_V\), or \(D_T\).

Three identities are intentionally separate. The scientific mainline is
Base-anchored completion through common--residual compatibility under one
zero-level-set output and relative Pareto/safety evaluation. A sealed candidate
identity freezes its concrete gate, normalization, objective, sampling, and
optimizer configuration. Non-mainline components may be changed to improve
performance only as a newly preregistered candidate with a new source closure
and adequate ablations. An execution identity such as fresh r2 may not contain
any such scientific change: r2 is only a same-candidate recovery of the
unresolved v24-r1 execution. Before r2 authorization, v24 may instead be retired
in favor of a new candidate, but that candidate cannot inherit the r2 identity.

The executable metadata checks are in
`tools/gcr_pacre_v24_protocol.py`. They provide:

- transitive `root_source_id` derivation and propagation closure;
- verifier-issued immutable evidence tokens (decision functions reject raw
  mappings and caller-asserted unread booleans);
- deterministic root-disjoint OOF4 assignment recomputed directly from the
  checksum-bound manifest and state index;
- real-path/stat/inode/link-count/SHA cache checks, strict creation/open event
  ordering, and exact reader allowlists;
- per-sample factual-only pooling with exact sample-to-root, GT, anchor,
  evaluation-contract, and denominator equality across all arms;
- a complete 51-row train-fold-only BaseB ledger and recomputed selector;
- pooled OOF, paired bounded-400, Formal800, and exact relative gate checks;
- exact source/schedule/finite/artifact/prerequisite validation; and
- exact baseline loading from a checksum-bound sealed aggregate ledger.

Formal validation fixes the production model to 64 feature channels, stride
4, width 32, and exactly 64,064 parameters, all at 800×40. A separate
verifier derives seed-42/seed-43 independence and proves that seed 43 has no
selection, replacement, \(D_V\), or \(D_T\) role.

Both Formal seeds are bound to the same semantic full-\(D_R\) cache and the
same recomputed neutral metadata-and-tensor content fingerprint carried by
the verified bounded materialization. Their physical cache artifacts must
instead have different canonical paths and device/inode identities, one hard
link each, FIEMAP evidence with no shared/unknown/delalloc/encoded extent,
fixed `torch.load(weights_only=True, mmap=False)` receipts, and disjoint
actual tensor storages under a simultaneous pair load. The two schedules use
the same policy and configuration except for seed, so their schedule
fingerprints must differ while the seed-excluded policy fingerprint matches.

`D_T_preregistration.json` freezes the future decision rule before any v24
\(D_V\) authorization, while explicitly leaving \(D_T\) unauthorized. Its
Base controls are fixed before \(D_T\): Base@A uses 0.72 and Base@B uses the
checksum-bound 0.14 selected on the sealed \(D_V\) ledger. \(D_T\) may evaluate
those two operating points once, but may not search or select a threshold.
`D_T_seed42_model_binding.template.json` is intentionally an incomplete
template, not a valid receipt. A completed
`D_T_seed42_model_binding.json` must be issued only after a verified seed-42
Formal800 terminal and must be sealed before any v24 \(D_V\) authorization.

The frozen design closure contains no data-dependent v24 result. Generated
dataset-free audit receipts and explicit invalidation records may coexist in
this directory, but they are execution evidence rather than preregistration
artifacts and do not authorize \(D_R\), \(D_V\), or \(D_T\).
The current runtime evidence is `dataset_free_receipt_r2.json`
(`receipt_fingerprint`
`efd55afcb8709ed20fb67aad8696c348bda166323a548018c7a7f282f1adad0a`);
it passed independent verify-only validation, supersedes invalidated r1, and
still grants no split-execution authorization.

The first authorized D_R structural execution identity, r1, lost execution
observability after its create-once marker and has no authenticated gate
decision. Its historical v1 evidence remains byte-preserved; the authoritative
interpretation is `D_R_structural_attempt_r1_interruption_receipt_v2.json`
(`receipt_fingerprint`
`d599df00a201d65f8cbd4b9390a4ef784a2a504200923b9894e12cd26d249792`).
That v2 distinguishes authorization, observation in available records, and
actual runtime facts that are not independently audited without an OS/filesystem
event ledger. It authorizes neither a fresh r2 attempt nor any downstream stage.

The outer r2 supervision scaffold now consists of
`tools/cure_lite_v24_runtime_supervisor.py` and the non-installable template
`deploy/systemd/cure-lite-v24-gcr-pacre-dr-r2.service.template`. This is
dataset-free infrastructure only: no actual r2 runtime spec, authorization,
attempt commit, materialization claim, installed unit, or r2 execution exists.
The scaffold uses a create-once attempt commit before the single start request,
a separate create-once `ExecCondition` materialization claim bound to the
systemd invocation, numbered hash-chain heartbeats, and per-invocation terminal
sidecars. A future authorization must bind the exact supervisor, realized unit,
runtime spec, and new source closure before any payload access.
The systemd terminal sidecar also revalidates the current supervisor, child
entrypoint, and unit-fragment closure; a successful manager exit cannot yield
`audit_valid=true` after those bytes drift. The supervisor/template/source-
closure target matrix currently passes 35 generated/static/fault-injection
tests. This is not a real user-systemd integration test and carries no
scientific performance meaning.

The release tests must preserve the same process-isolation assumption as the
real bounded/Formal CLIs. In particular, run the source-closure-sensitive
runner matrix in its own Python process, then run the remainder separately:

```text
/home/md0/ly/MSHNet/.venv/bin/python -m pytest -q \
  tests_v24/test_gcr_pacre_v24_training_runners.py
/home/md0/ly/MSHNet/.venv/bin/python -m pytest -q \
  tests_v24 \
  --ignore=tests_v24/test_gcr_pacre_v24_training_runners.py \
  tests/test_cure_lite_v24_gcr_pacre_dataset_free_audit.py \
  tests/test_cure_lite_v24_gcr_pacre_training.py
```

The current isolated matrix is `16 PASS` plus `263 PASS, 1 SKIP`, or
`279 PASS, 1 SKIP` in total. A monolithic pytest process intentionally trips
the fail-closed source audit after unrelated tests import `cure_lite.toy`;
those test-only modules must not be added to the scientific source closure.
Actual r2 remains launch-blocked: the user manager is degraded and an enabled
`Restart=on-failure` service is still auto-restarting an `--device cuda:0`
workload, which conflicts with the r1-frozen r2 device. A momentarily idle GPU
snapshot is not exclusivity evidence; stopping or disabling that unrelated unit
requires separate user authority and a subsequent preflight audit.

`preregistration.json` is self-fingerprinted. The split preregistration,
\(D_T\) preregistration, and exact-baseline binding each anchor that main
fingerprint and carry their own self-fingerprint. `protocol.schema.json`
rejects unknown keys at every decision-bearing object.
`artifact_manifest.json` closes the frozen-design chain by binding the bytes
and declared self-fingerprint of the preregistration, split, \(D_T\),
baseline-binding, schema, and model-binding-template artifacts. Runtime audit
receipts/invalidation records are intentionally outside that design manifest.

The compact D_R split preregistration freezes the 160-sample, 156-root OOF4
assignment from metadata only. Per-fold factual/clean/null/target/context role
counts and their union/intersection proof remain explicitly pending until an
authorized D_R materialization; no global role count has been divided across
folds or otherwise inferred.
