# Oracle Drift Study - First Artifact

Generated from existing TSLib prediction artifacts under `external/TSLib/results`.
This is the first M1 deliverable: it proves the analysis path from frozen
baseline predictions to per-window oracle trajectories. It is not yet the full
finance-MISO gate from `docs/PROPOSAL.md`.

## Scope

- Dataset: `ETTh1`
- Lookback: `96`
- Horizons: `96`, `192`, `336`, `720`
- Models: `DLinear`, `PatchTST`, `TiDE`, `TimeXer`
- Scoring: last channel only (`--target-channel -1`), matching the ETT
  target-column/MISO convention.

## Outputs per horizon

Each `ETTh1_L96_H*_target_last/` directory contains:

- `window_losses.csv`
- `best_architecture_trajectory.csv`
- `summary.json`
- `best_architecture_trajectory.png`

## Summary

| Setting | Best single | Best single MSE | Oracle MSE | Oracle gap | Switch rate | Win counts |
|---|---:|---:|---:|---:|---:|---|
| ETTh1 L96 H96 | PatchTST | 0.055438 | 0.041406 | 25.31% | 12.28% | DLinear 892, PatchTST 534, TiDE 635, TimeXer 724 |
| ETTh1 L96 H192 | TimeXer | 0.069576 | 0.053147 | 23.61% | 9.15% | DLinear 569, PatchTST 352, TiDE 559, TimeXer 1209 |
| ETTh1 L96 H336 | TimeXer | 0.083118 | 0.064141 | 22.83% | 9.98% | DLinear 376, PatchTST 364, TiDE 577, TimeXer 1228 |
| ETTh1 L96 H720 | TimeXer | 0.088817 | 0.082859 | 6.71% | 10.83% | DLinear 0, PatchTST 274, TiDE 555, TimeXer 1332 |

## Reading

Even on ETTh1, which PRISM treats as a comparatively stable contrast dataset,
the per-window oracle is nontrivial: different architectures win different test
windows, and the hindsight oracle has measurable headroom over the best aggregate
single model. This is only an analysis-pipeline result; the actual greenlight
gate still requires finance-MISO runs, seed/noise bands, and a formal nonconstant
switch-rate test.

## Next M1 step

Add the finance-MISO prediction producers for the same artifact contract:
`pred.npy` and `true.npy` with shape `[windows, horizon, channels]`, then rerun
`experiments.PRISM.oracle_drift` on Crypto/CN-Future/FI2010 and compare the
finance switch rates/oracle gaps against this ETTh1 contrast.
