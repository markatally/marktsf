# PRISM Main-Track Hardening Artifact Set

This directory contains the PRISM evidence package after M17 main-track
hardening. The original learned-router/SOTA claim is retired, but the current
artifact now clears a **scoped selective main-track route**:
practical-effect horizon-wise affine calibration for non-financial
sensor/infrastructure forecasting.

## Current Claim Set

1. Optimal-bias drift is broad in the ETT/Weather lightweight screen.
2. Strong validation-tuned Fixed-Share over frozen heterogeneous experts is the
   robust causal tracker.
3. Dynamic beta/drift-loop improves stressed loss on 4/6 datasets after
   delayed-feedback correction, but no full-vs-plain or
   full-vs-validation-single comparison survives block/FDR.
4. The learned router, dynamic beta, and drift-triggered share-rate loop are
   negative or insufficient results in their current form and must not be
   presented as headline method wins.
5. M17 practical-effect selective horizon-wise affine calibration activates only
   when the past split shows >=5% and p<=0.05 improvement against both
   validation-single and delayed Fixed-Share; active cells pass BH/FDR against
   validation-single, delayed Fixed-Share, and descriptor ridge.

## Main-Track Status

`main_track_audit/main_track_audit.json` currently returns
`ALLOW_SCOPED_MAIN_TRACK_SUBMISSION`. The admissible positive claim is scoped:

- scope: 8 non-financial sensor/infrastructure datasets x 2 horizons;
- active cells: Electricity H96, Traffic H96, AQWan H96, AQWan H192;
- inactive cells abstain exactly to validation-single;
- active-cell FDR pass counts are 4/4 against validation-single, delayed
  Fixed-Share, and descriptor ridge.

The old delayed contextual router, champion-risk safe-switch, broad calibrated
stacking, finance, and full SOTA claims remain negative or historical.

## Reproduction

Run from the repository root:

```bash
PY=${PY:-python3}
$PY -m experiments.PRISM.produce_m1c_predictions
for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange; do
  $PY -m experiments.PRISM.oracle_drift \
    --results-root external/TSLib/results \
    --output-dir experiments/PRISM/oracle_drift/M1C_${ds}_L96_H96_target_last \
    --dataset M1C_${ds} --lookback 96 --horizon 96 \
    --models RidgeCov TargetRidge Trend Seasonal EWM \
    --target-channel -1 --include-anchors
  $PY -m experiments.PRISM.online_learning \
    --losses-csv experiments/PRISM/oracle_drift/M1C_${ds}_L96_H96_target_last/window_losses.csv \
    --output-dir experiments/PRISM/oracle_drift/M1C_${ds}_L96_H96_target_last
  $PY -m experiments.PRISM.descriptor_probe \
    --oracle-dir experiments/PRISM/oracle_drift/M1C_${ds}_L96_H96_target_last \
    --output-dir experiments/PRISM/oracle_drift/M1C_${ds}_L96_H96_target_last \
    --dataset M1C_${ds}
done
$PY -m experiments.PRISM.router_viability
$PY -m experiments.PRISM.drift_beta_loop
$PY -m experiments.PRISM.ablations_significance
$PY -m experiments.PRISM.paper_ready
$PY -m experiments.PRISM.main_track_audit
```

The same command sequence is recorded in `paper_ready/REPRODUCE.md`.

## Key Artifacts

- `docs/REPORT.md` — milestone evidence and final gate adjudication.
- `docs/PROPOSAL.md` — historical preregistration, now annotated with M17 status.
- `docs/INTEGRITY_AUDIT.md` — failure-mode audit and corrections made during the
  top-journal review pass.
- `oracle_drift/m1c_summary.json` — consolidated M1c breadth/probe summary.
- `router_viability/router_viability_summary.json` — M2 learned-router kill test.
- `router_viability_h192/router_viability_summary.json` — M7 H=192
  multi-horizon router pilot.
- `router_viability_expanded_h96/router_viability_summary.json` and
  `router_viability_expanded_h192/router_viability_summary.json` — M8 expanded
  expert-pool router pilots.
- `practical_selective_horizon_affine/practical_selective_horizon_affine_summary.json`
  — M17 scoped selective route gate.
- `drift_beta_loop/drift_beta_summary.json` — M3 dynamic beta and drift-stress
  evaluation.
- `ablations_significance/ablations_significance_summary.json` — M4 block-robust
  ablations, FDR, and synthetic identifiability.
- `paper_ready/paper_ready_summary.json` — empirical artifact manifest.
- `main_track_audit/main_track_audit.json` — M6 strong main-track readiness
  audit and blocking criteria.

## Historical Context

M1a/M1b started as an oracle drift study over ETTh1, Crypto/CryptoMISO,
CryptoVol, and FI2010. The finance gate failed under the strict preregistered
condition, so finance artifacts are retained as negative evidence rather than
headline claims.
