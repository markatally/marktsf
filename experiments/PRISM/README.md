# PRISM M1: Oracle Drift Study

This directory holds the first M1 artifact for PRISM: an oracle analysis over
already-trained baseline predictions. It answers the gate question from
`docs/PROPOSAL.md`: does the per-window best architecture change over time, and
how much headroom does a hindsight oracle have over the best single model?

## Current entrypoint

```bash
python -m experiments.PRISM.oracle_drift \
  --results-root external/TSLib/results \
  --output-dir experiments/PRISM/oracle_drift/ETTh1_L96_H96_target_last \
  --dataset ETTh1 \
  --lookback 96 \
  --horizon 96 \
  --models DLinear PatchTST TiDE TimeXer \
  --target-channel -1
```

Outputs:

- `window_losses.csv` — one row per test window, one MSE column per model.
- `best_architecture_trajectory.csv` — oracle best model, runner-up, margin,
  and gain vs the best single model for each window.
- `summary.json` — aggregate model MSE, oracle MSE, oracle gap, switch count,
  switch rate, and win counts.
- `best_architecture_trajectory.png` — compact visualization of oracle picks
  and rolling model losses.

`--target-channel -1` scores the last channel only, matching the MISO convention
used for ETT-style target-column experiments. Use `--target-channel all` for the
classic symmetric multivariate score.
