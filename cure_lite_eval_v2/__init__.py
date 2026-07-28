"""Append-only evaluation recovery for the sealed PAET Formal800 run.

This package is deliberately outside :mod:`cure_lite`.  The original
Formal800 source closure therefore remains byte-for-byte unchanged.  The
package only repairs two frozen producer/consumer schema-name mismatches in
memory while delegating every scientific and artifact check to the original
strict loader and D_V runner.
"""

# Keep package import lightweight. Runtime entrypoints are imported explicitly
# from ``formal_d_v_runner_v2`` so closure and erratum tooling do not import
# torch as a side effect.

__all__: list[str] = []
