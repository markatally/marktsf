# PRISM Empirical Artifact and Main-Track Audit Reproduction

Run from the repository root with the bundled/scientific Python environment.

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
$PY -m experiments.PRISM.champion_risk_gate
$PY -m experiments.PRISM.calibrated_stack_gate
$PY -m experiments.PRISM.drift_beta_loop
$PY -m experiments.PRISM.ablations_significance
$PY -m experiments.PRISM.paper_ready
$PY -m experiments.PRISM.main_track_audit
$PY -m experiments.PRISM.produce_m1c_predictions --horizon 192 --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange
for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange; do
  $PY -m experiments.PRISM.oracle_drift \
    --results-root external/TSLib/results \
    --output-dir experiments/PRISM/oracle_drift/M1C_${ds}_L96_H192_target_last \
    --dataset M1C_${ds} --lookback 96 --horizon 192 \
    --models RidgeCov TargetRidge Trend Seasonal EWM \
    --target-channel -1 --include-anchors
  $PY -m experiments.PRISM.descriptor_probe \
    --oracle-dir experiments/PRISM/oracle_drift/M1C_${ds}_L96_H192_target_last \
    --output-dir experiments/PRISM/oracle_drift/M1C_${ds}_L96_H192_target_last \
    --dataset M1C_${ds} --horizon 192
done
$PY -m experiments.PRISM.router_viability --horizon 192 --output-dir experiments/PRISM/router_viability_h192
$PY -m experiments.PRISM.champion_risk_gate --horizon 192 --output-dir experiments/PRISM/champion_risk_gate_h192
$PY -m experiments.PRISM.calibrated_stack_gate --horizon 192 --output-dir experiments/PRISM/calibrated_stack_gate_h192
$PY -m experiments.PRISM.produce_m1c_predictions --pool expanded --results-root external/TSLib/results_prism_expanded --horizon 96 --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange
$PY -m experiments.PRISM.produce_m1c_predictions --pool expanded --results-root external/TSLib/results_prism_expanded --horizon 192 --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange
for h in 96 192; do
  for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange; do
    $PY -m experiments.PRISM.oracle_drift \
      --results-root external/TSLib/results_prism_expanded \
      --output-dir experiments/PRISM/oracle_drift_expanded/M1C_${ds}_L96_H${h}_target_last \
      --dataset M1C_${ds} --lookback 96 --horizon ${h} \
      --models RidgeCov TargetRidge Trend Seasonal EWM EWMFast EWMSlow SeasonalOffset SeasonalDrift DampedTrend MeanRevert MeanRevertSlow SeasonalEWM SeasonalTrend EWMTrend \
      --target-channel -1 --include-anchors
    $PY -m experiments.PRISM.descriptor_probe \
      --results-root external/TSLib/results_prism_expanded \
      --oracle-dir experiments/PRISM/oracle_drift_expanded/M1C_${ds}_L96_H${h}_target_last \
      --output-dir experiments/PRISM/oracle_drift_expanded/M1C_${ds}_L96_H${h}_target_last \
      --dataset M1C_${ds} --horizon ${h}
  done
done
$PY -m experiments.PRISM.router_viability --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --output-dir experiments/PRISM/router_viability_expanded_h96
$PY -m experiments.PRISM.router_viability --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --horizon 192 --output-dir experiments/PRISM/router_viability_expanded_h192
$PY -m experiments.PRISM.champion_risk_gate --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --output-dir experiments/PRISM/champion_risk_gate_expanded_h96
$PY -m experiments.PRISM.champion_risk_gate --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --horizon 192 --output-dir experiments/PRISM/champion_risk_gate_expanded_h192
$PY -m experiments.PRISM.calibrated_stack_gate --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --output-dir experiments/PRISM/calibrated_stack_gate_expanded_h96
$PY -m experiments.PRISM.calibrated_stack_gate --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --horizon 192 --output-dir experiments/PRISM/calibrated_stack_gate_expanded_h192
$PY -m experiments.PRISM.produce_m1c_predictions --datasets Electricity Traffic --pool expanded --results-root external/TSLib/results_prism_nonfinancial --horizon 96 --max-covariates 64
$PY -m experiments.PRISM.produce_m1c_predictions --datasets Electricity Traffic --pool expanded --results-root external/TSLib/results_prism_nonfinancial --horizon 192 --max-covariates 64
for h in 96 192; do
  for ds in Electricity Traffic; do
    $PY -m experiments.PRISM.oracle_drift \
      --results-root external/TSLib/results_prism_nonfinancial \
      --output-dir experiments/PRISM/oracle_drift_nonfinancial/M1C_${ds}_L96_H${h}_target_last \
      --dataset M1C_${ds} --lookback 96 --horizon ${h} \
      --models RidgeCov TargetRidge Trend Seasonal EWM EWMFast EWMSlow SeasonalOffset SeasonalDrift DampedTrend MeanRevert MeanRevertSlow SeasonalEWM SeasonalTrend EWMTrend \
      --target-channel -1 --include-anchors
    $PY -m experiments.PRISM.descriptor_probe \
      --results-root external/TSLib/results_prism_nonfinancial \
      --oracle-dir experiments/PRISM/oracle_drift_nonfinancial/M1C_${ds}_L96_H${h}_target_last \
      --output-dir experiments/PRISM/oracle_drift_nonfinancial/M1C_${ds}_L96_H${h}_target_last \
      --dataset M1C_${ds} --horizon ${h}
  done
done
$PY -m experiments.PRISM.calibrated_stack_gate --oracle-root experiments/PRISM/oracle_drift_nonfinancial --results-root external/TSLib/results_prism_nonfinancial --output-dir experiments/PRISM/calibrated_stack_gate_nonfinancial_h96 --datasets Electricity Traffic --horizon 96
$PY -m experiments.PRISM.calibrated_stack_gate --oracle-root experiments/PRISM/oracle_drift_nonfinancial --results-root external/TSLib/results_prism_nonfinancial --output-dir experiments/PRISM/calibrated_stack_gate_nonfinancial_h192 --datasets Electricity Traffic --horizon 192
$PY -m experiments.PRISM.nonfinancial_stack_audit
$PY -m experiments.PRISM.calibrated_stack_significance
$PY -m experiments.PRISM.produce_m1c_predictions --datasets PEMS04 PEMS08 --pool expanded --results-root external/TSLib/results_prism_sensor --horizon 96 --max-covariates 64 --shared-context
$PY -m experiments.PRISM.produce_m1c_predictions --datasets PEMS04 PEMS08 --pool expanded --results-root external/TSLib/results_prism_sensor --horizon 192 --max-covariates 64 --shared-context
for h in 96 192; do
  for ds in PEMS04 PEMS08; do
    $PY -m experiments.PRISM.oracle_drift \
      --results-root external/TSLib/results_prism_sensor \
      --output-dir experiments/PRISM/oracle_drift_sensor/M1C_${ds}_L96_H${h}_target_last \
      --dataset M1C_${ds} --lookback 96 --horizon ${h} \
      --models RidgeCov TargetRidge Trend Seasonal EWM EWMFast EWMSlow SeasonalOffset SeasonalDrift DampedTrend MeanRevert MeanRevertSlow SeasonalEWM SeasonalTrend EWMTrend \
      --target-channel -1 --include-anchors
    $PY -m experiments.PRISM.descriptor_probe \
      --results-root external/TSLib/results_prism_sensor \
      --oracle-dir experiments/PRISM/oracle_drift_sensor/M1C_${ds}_L96_H${h}_target_last \
      --output-dir experiments/PRISM/oracle_drift_sensor/M1C_${ds}_L96_H${h}_target_last \
      --dataset M1C_${ds} --horizon ${h}
  done
done
$PY -m experiments.PRISM.calibrated_stack_gate --oracle-root experiments/PRISM/oracle_drift_sensor --results-root external/TSLib/results_prism_sensor --output-dir experiments/PRISM/calibrated_stack_gate_sensor_h96 --datasets PEMS04 PEMS08 --horizon 96
$PY -m experiments.PRISM.calibrated_stack_gate --oracle-root experiments/PRISM/oracle_drift_sensor --results-root external/TSLib/results_prism_sensor --output-dir experiments/PRISM/calibrated_stack_gate_sensor_h192 --datasets PEMS04 PEMS08 --horizon 192
$PY -m experiments.PRISM.sensor_stack_significance
$PY -m experiments.PRISM.online_stack_portfolio
$PY -m experiments.PRISM.sensor_horizon_affine_significance
$PY -m experiments.PRISM.selective_horizon_affine_gate
$PY -m experiments.PRISM.produce_m1c_predictions --datasets Wind AQShunyi AQWan METRLA --pool expanded --results-root external/TSLib/results_prism_sensor_ext --horizon 96 --max-covariates 64 --shared-context
$PY -m experiments.PRISM.produce_m1c_predictions --datasets Wind AQShunyi AQWan METRLA --pool expanded --results-root external/TSLib/results_prism_sensor_ext --horizon 192 --max-covariates 64 --shared-context
for h in 96 192; do
  for ds in Wind AQShunyi AQWan METRLA; do
    $PY -m experiments.PRISM.oracle_drift \
      --results-root external/TSLib/results_prism_sensor_ext \
      --output-dir experiments/PRISM/oracle_drift_sensor_ext/M1C_${ds}_L96_H${h}_target_last \
      --dataset M1C_${ds} --lookback 96 --horizon ${h} \
      --models RidgeCov TargetRidge Trend Seasonal EWM EWMFast EWMSlow SeasonalOffset SeasonalDrift DampedTrend MeanRevert MeanRevertSlow SeasonalEWM SeasonalTrend EWMTrend \
      --target-channel -1
    $PY -m experiments.PRISM.descriptor_probe \
      --results-root external/TSLib/results_prism_sensor_ext \
      --oracle-dir experiments/PRISM/oracle_drift_sensor_ext/M1C_${ds}_L96_H${h}_target_last \
      --output-dir experiments/PRISM/oracle_drift_sensor_ext/M1C_${ds}_L96_H${h}_target_last \
      --dataset M1C_${ds} --horizon ${h}
  done
done
$PY -m experiments.PRISM.practical_selective_horizon_affine_gate
$PY -m experiments.PRISM.practical_selective_horizon_affine_gate --min-effect-pct 0 --output-dir experiments/PRISM/practical_selective_horizon_affine_sensitivity_0
$PY -m experiments.PRISM.practical_selective_horizon_affine_gate --min-effect-pct 2.5 --output-dir experiments/PRISM/practical_selective_horizon_affine_sensitivity_2p5
$PY -m experiments.PRISM.practical_selective_horizon_affine_gate --min-effect-pct 10 --output-dir experiments/PRISM/practical_selective_horizon_affine_sensitivity_10
$PY -m experiments.PRISM.submission_render
$PY -m experiments.PRISM.submission_hardening
$PY -m experiments.PRISM.paper_ready
$PY -m experiments.PRISM.main_track_audit
```

## Final Route

The original PRISM learned-router method remains rejected after strengthened
baseline hardening.  The current positive route is a scoped practical-effect
selective horizon-wise affine calibration method, not the original routing-level
SOTA claim:

- M1b finance gate failed under the strict preregistered condition.
- M2 delayed contextual router fails once a validation-selected single-expert baseline is added.
- M3 dynamic beta/drift loop improves stressed loss on 4/6 datasets.
- M4 full-vs-plain and full-vs-validation-single do not clear the strengthened block/FDR gate.
- M7 H=192 router pilot passes only 1/6 datasets.
- M8 expanded expert pool improves validation-single baselines but current router passes 0/6 at H=96 and 1/6 at H=192.
- M9 champion-risk safe-switch does not rescue the main method: base H96/H192 pass 0/6, expanded H96 passes 0/6, expanded H192 passes 1/6.
- M10 calibrated forecast stacking is a strong near miss: base H96/H192 pass 5/6, expanded H96 passes 4/6, expanded H192 passes 5/6; Exchange remains the repeated blocker.
- M11 narrowed non-financial calibrated-stacking route passes 7/7 datasets at both H=96 and H=192 after adding Electricity and Traffic.
- M12 block/FDR significance remains incomplete after horizon-wise affine stacking: stack vs Fixed-Share passes 14/14, but stack vs validation-single passes 7/14 and stack vs descriptor ridge passes 10/14.
- M13 high-dimensional sensor route adds PEMS04/PEMS08 and remains close but incomplete: stack vs descriptor ridge passes 8/8, Fixed-Share 7/8, and validation-single 6/8.
- M14 delayed online stacker portfolio improves the sensor route to Fixed-Share 8/8 and descriptor ridge 8/8, but validation-single remains 6/8.
- M15 fixed horizon-wise affine stacking is the cleanest current candidate: Fixed-Share 8/8, descriptor ridge 8/8, validation-single 7/8. Electricity H192 remains the single strict blocker.
- M16 selective horizon-wise affine no-harm gate is honest but under-covered: active cells pass 2/2 against all baselines, inactive cells exactly abstain to validation-single, but active coverage is only 2/8.
- M17 practical-effect selective horizon-wise affine activation passes the scoped main-route gate: active cells pass against validation-single, delayed Fixed-Share, and descriptor ridge; inactive cells abstain to validation-single. The active cells are Electricity H96, Traffic H96, AQWan H96, and AQWan H192.
- Drift-triggered share-rate adaptation is rejected in the current form.

## M2 Router Viability

| Dataset | Validation Single | Fixed-Share | Descriptor Ridge | PRISM Router | Gate |
| --- | --- | --- | --- | --- | --- |
| ETTh1 | 0.0675922 (HAR_EWM) | 0.0753692 | 0.061286 | 0.0611503 | PASS |
| ETTh2 | 0.174059 (Seasonal) | 0.210375 | 0.166817 | 0.169313 | FAIL |
| ETTm1 | 0.0281378 (Persistence) | 0.0350865 | 0.0314722 | 0.0313326 | FAIL |
| ETTm2 | 0.103208 (Seasonal) | 0.121959 | 0.0998626 | 0.0995334 | PASS |
| Weather | 0.00100581 (EWM) | 0.00131444 | 0.00100759 | 0.00100431 | PASS |
| Exchange | 0.08638 (TargetRidge) | 0.13091 | 0.202186 | 0.157699 | FAIL |

## M3 Dynamic Beta / Drift Stress

| Dataset | Plain Stress | Loop Stress | Improvement | Beta IQR |
| --- | --- | --- | --- | --- |
| ETTh1 | 0.0768871 | 0.0771981 | -0.404% | 0.262 |
| ETTh2 | 0.204747 | 0.204165 | 0.284% | 0.438 |
| ETTm1 | 0.036547 | 0.0343414 | 6.03% | 0.224 |
| ETTm2 | 0.155006 | 0.15475 | 0.165% | 0.334 |
| Weather | 0.00137443 | 0.00137519 | -0.0556% | 0.271 |
| Exchange | 0.135462 | 0.13529 | 0.127% | 0.235 |

## M4 FDR Ablations

| Dataset | Validation Single | Plain FS | Full | Full vs Plain | Plain FDR | Validation FDR |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 0.0675922 (HAR_EWM) | 0.0753692 | 0.075596 | -0.301% | FAIL | FAIL |
| ETTh2 | 0.174059 (Seasonal) | 0.210375 | 0.209708 | 0.317% | FAIL | FAIL |
| ETTm1 | 0.0281378 (Persistence) | 0.0350865 | 0.0330965 | 5.67% | FAIL | FAIL |
| ETTm2 | 0.103208 (Seasonal) | 0.143851 | 0.143668 | 0.128% | FAIL | FAIL |
| Weather | 0.00100581 (EWM) | 0.00131444 | 0.0013077 | 0.513% | FAIL | FAIL |
| Exchange | 0.08638 (TargetRidge) | 0.13091 | 0.130795 | 0.0879% | FAIL | FAIL |

## Synthetic Identifiability

- State recovery accuracy: 0.966
- Best single loss: 0.6791
- Oracle loss: 0.4372
- Descriptor router loss: 0.4550
