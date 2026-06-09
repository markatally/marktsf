"""Reproduction targets + tolerance gate.

The reproduction policy (per user requirement: ≤2%, use original repo code):

- We run the ORIGINAL authors' code (thuml/Time-Series-Library, vendored under
  external/TSLib) with the official per-model scripts. That is the ground truth.
- ``REPRO_MSE`` holds the number the ORIGINAL code yields on this setup; the gate
  checks our wrapper reproduces it within ``DEFAULT_TOLERANCE`` (2%). On a clean
  re-run this should be ~0% — it guards against environment / wiring drift.
- ``PAPER_MSE`` holds the published paper-table value for CONTEXT only. Original
  code often differs from the paper table by a couple percent (different seeds /
  best-of-runs); that gap is expected and is NOT a wiring bug.

Populate ``REPRO_MSE`` from measured TSLib runs (the runner prints them); start
empty and fill as cells are validated so the gate never fires on a guess.
"""

from __future__ import annotations

__all__ = ["REPRO_MSE", "PAPER_MSE", "DEFAULT_TOLERANCE", "gate_status"]

# Strict gate: our wrapper must reproduce the vendored original code within 2%.
DEFAULT_TOLERANCE = 0.02

# (dataset, model, horizon) -> MSE produced by the ORIGINAL TSLib code here.
# Filled in as each cell is run & verified (see spectre/reproduce/official.py).
REPRO_MSE: dict[tuple[str, str, int], float] = {
    # Verified 2026-06-09 on CPU, num_workers=0, official DLinear_ETTh1 args.
    ("ETTh1", "DLinear", 96): 0.3962,
}

# (dataset, model, horizon) -> published paper-table MSE (context, not gated).
PAPER_MSE: dict[tuple[str, str, int], float] = {
    ("ETTh1", "DLinear", 96): 0.386,
    ("ETTh1", "DLinear", 192): 0.437,
    ("ETTh1", "DLinear", 336): 0.481,
    ("ETTh1", "DLinear", 720): 0.519,
    ("ETTh1", "iTransformer", 96): 0.386,
    ("ETTh1", "PatchTST", 96): 0.414,
    ("ETTh1", "TimeXer", 96): 0.382,
}


def gate_status(our: float, ref: float | None, tol: float = DEFAULT_TOLERANCE) -> str:
    """Classify a reproduced number against its original-code reference.

    "PASS" within tol, "WARN" within 2×tol, "FAIL" beyond, "N/A" if no reference.
    """
    if ref is None:
        return "N/A"
    rel = abs(our - ref) / ref
    if rel <= tol:
        return "PASS"
    if rel <= 2 * tol:
        return "WARN"
    return "FAIL"
