# PRISM M1: Oracle Drift Study

This directory holds the first M1 artifact for PRISM: an oracle analysis over
already-trained baseline predictions. It answers the gate question from
`experiments/PRISM/docs/PROPOSAL.md`: does the per-window best architecture change over time, and
how much headroom does a hindsight oracle have over the best single model?

## Step 1 — Produce finance-MISO predictions (Crypto)

Train one baseline model on Crypto 1h close log-returns (14 assets, BTCUSDT
as target, read from `input/Crypto/`; nothing written to the repo):

```bash
python -m experiments.PRISM.produce_predictions \
    --model DLinear --lookback 96 --horizon 96
```

Supported models: `DLinear`, `PatchTST`, `iTransformer`, `TimesNet`.

Run all 4 × 4 (horizons 24 / 48 / 96 / 168):

```bash
for model in DLinear PatchTST iTransformer TimesNet; do
  for h in 24 48 96 168; do
    python -m experiments.PRISM.produce_predictions \
        --model $model --lookback 96 --horizon $h
  done
done
```

Results land in `external/TSLib/results/` (gitignored) named:
`long_term_forecast_Crypto_{L}_{H}_{model}_Crypto_{suffix}`

## Step 2 — Oracle Drift Study

### Crypto (finance primary)

```bash
python -m experiments.PRISM.oracle_drift \
  --results-root external/TSLib/results \
  --output-dir experiments/PRISM/oracle_drift/Crypto_L96_H96_target_last \
  --dataset Crypto \
  --lookback 96 --horizon 96 \
  --models DLinear PatchTST iTransformer TimesNet \
  --target-channel -1
```

### ETTh1 (contrast — quasi-stationary benchmark)

```bash
python -m experiments.PRISM.oracle_drift \
  --results-root external/TSLib/results \
  --output-dir experiments/PRISM/oracle_drift/ETTh1_L96_H96_target_last \
  --dataset ETTh1 \
  --lookback 96 --horizon 96 \
  --models DLinear PatchTST TiDE TimeXer \
  --target-channel -1
```

## Outputs (per run)

- `window_losses.csv` — one row per test window, one MSE column per model.
- `best_architecture_trajectory.csv` — oracle best model, runner-up, margin,
  and gain vs the best single model for each window.
- `summary.json` — aggregate model MSE, oracle MSE, oracle gap, switch count,
  switch rate, and win counts.
- `best_architecture_trajectory.png` — visualization of oracle picks and
  rolling model losses.

`--target-channel -1` scores the last channel only (BTCUSDT for Crypto, OT
for ETT), matching the MISO target-column convention.
