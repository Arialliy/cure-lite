"""Append-only evaluation-v3 correction for the sealed PAET Formal800 run.

This package is deliberately separate from both :mod:`cure_lite` and
:mod:`cure_lite_eval_v2`.  It preserves the failed evaluation-v2 attempt and
introduces one new run identity whose only executable correction is the
sealed-input access path used by the fixed-sample builder.
"""

# Keep imports lightweight.  Runtime entrypoints are imported explicitly from
# ``formal_d_v_runner_v3`` so closure tooling does not import torch as a side
# effect.

__all__: list[str] = []
