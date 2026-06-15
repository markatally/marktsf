# PRISM Integrity Audit

Audit date: 2026-06-15

Scope: `experiments/PRISM/` M6/M17 main-track hardening pass. The active
positive artifact is now a scoped selective main-route candidate, not the
original full PRISM learned-router/SOTA claim.

## Verdict

**PASS for a scoped selective main-track route after M17. FAIL/retired for the
original full PRISM SOTA-method claim.**

The paper-ready story is defensible only as:

1. oracle-drift measurement evidence;
2. Fixed-Share as the robust causal tracker;
3. delayed contextual routing that passes only 3/6 datasets once a
   validation-selected single-expert baseline is added;
4. dynamic beta/drift-loop stress improvements that do not survive BH/FDR.
5. H=192 multi-horizon pilot evidence showing the current router failure is not
   confined to H=96.
6. expanded expert-pool evidence showing that stronger static experts alone do
   not rescue the current descriptor/prior router.
7. champion-risk safe-switch evidence showing that conservative causal risk
   gating mostly collapses to validation-single and does not rescue the main
   method.
8. calibrated forecast-stacking evidence showing a strong near miss, with
   Exchange as the repeated out-of-scope/blocking battlefield.
9. narrowed non-financial calibrated-stacking evidence showing a revived
   candidate main route over ETT, Weather, Electricity, and Traffic.
10. block/FDR calibrated-stacking significance showing that the revived route
    is promising but not sufficient as a full-coverage route.
11. M17 practical-effect selective horizon-wise affine calibration passing the
    scoped route gate: active 4/16 cells, all active cells pass BH/FDR against
    validation-single, delayed Fixed-Share, and descriptor ridge, while inactive
    cells abstain exactly to validation-single.

Any manuscript text that presents the learned router, routing-level TTA,
regime-spawning, finance wins, or full PRISM SOTA victory as completed evidence
is not paper-ready. The admissible main-paper claim is narrower: a
practical-effect selective calibration method for non-financial
sensor/infrastructure forecasting.

## Corrections Made

| Issue | Failure mode | Correction | Evidence |
|---|---|---|---|
| Fixed-Share in M2 was selected by scanning test losses. | implementation bug / methodology fabrication | `router_viability.py` now tunes Fixed-Share only on a chronological validation slice of the past split, then evaluates once on test. | Current M2 uses validation-only delayed Fixed-Share; remaining failures are ETTh2 and Exchange. |
| M2/M3/M4 used immediate feedback on stride-1 H-step losses. | implementation bug / shortcut reliance | Online updates now use `feedback_delay_windows=96`, so full loss for window t can only update window t+96 or later. | After validation-single hardening, M2 delayed router passes 3/6; M3 passes stress gate; M4 FDR passes 0/6. |
| M2/M3/M4 used too-weak plain Fixed-Share grids for main-track claims. | shortcut reliance / methodology weakness | Fixed-Share and loop grids now include low α and high lr, all selected on chronological validation only. | Combined with delayed feedback, this is the current strong baseline protocol. |
| M2/M4 did not include the simple expert selected by the most recent validation slice. | shortcut reliance / weak-baseline risk | `router_viability.py` now reports `validation_single_loss`; `ablations_significance.py` tests `full_vs_validation_single`; M6 blocks main-track submission on this criterion. | M2 now passes only 3/6; M4 full-vs-validation-single FDR passes 0/6 and is negative on all six datasets. |
| M4 p-values treated stride-1 overlapping windows as independent. | implementation bug / hallucinated significance risk | `ablations_significance.py` now aggregates into horizon-sized blocks and uses paired block sign-flip p-values before BH/FDR. | After delayed-feedback hardening, full-vs-plain FDR passes 0/6 datasets. |
| Multi-horizon readiness had been asserted only as missing work. | methodology weakness | Added an H=192 M1C/router pilot in `router_viability_h192` using the same causal M2 gate. | H=192 M2 passes only 1/6, so current router failure is not an H=96-only artifact. |
| Expert diversity was hypothesized but not tested as a router rescue. | frame-lock / shortcut reliance risk | Added `--pool expanded` with causal EWM, seasonal-offset/drift, damped-trend, mean-reversion, and blend experts, evaluated in `router_viability_expanded_h96` and `router_viability_expanded_h192`. | Expanded pool improves validation-single baselines but current router still fails: H=96 passes 0/6, H=192 passes 1/6. |
| A safer learned gate could have been asserted without evidence. | frame-lock / methodology fabrication risk | Added M9 `champion_risk_gate.py`, a causal champion-risk safe-switch gate using lookback/forecast plausibility features and multi-fold past-only safety validation. | M9 fails: base H96 0/6, base H192 0/6, expanded H96 0/6, expanded H192 1/6. |
| Hard routing failure could have prematurely ended method search. | frame-lock risk | Added M10 `calibrated_stack_gate.py`, a forecast-level affine/simplex stacking gate tuned only on chronological past validation. | M10 is the strongest rescue so far but still fails strict main-track gates: base H96 5/6, base H192 5/6, expanded H96 4/6, expanded H192 5/6. |
| Exchange failure could have been silently removed. | frame-lock / hallucinated result risk | Added M11 `nonfinancial_stack_audit.py`, which explicitly narrows scope to non-financial periodic/sensor datasets and retains Exchange as an out-of-scope negative diagnostic. | M11 passes the narrowed route: H96 7/7 and H192 7/7 over ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, and Traffic. |
| Candidate-route effect sizes could be mistaken for significance. | hallucinated results / methodology fabrication risk | Added M12 `calibrated_stack_significance.py`, using horizon-block paired sign-flip tests and BH/FDR for calibrated stack versus validation-single, Fixed-Share, and descriptor ridge. | M12 fails strict significance after horizon-wise affine stacking: Fixed-Share 14/14, but validation-single 7/14 and descriptor ridge 10/14. |
| A high-dimensional sensor route could be cherry-picked after M12. | frame-lock / result-shopping risk | Added M13 `sensor_stack_significance.py` with a predeclared non-financial high-dimensional sensor/infrastructure route and the same block/FDR gate. | M13 still fails: validation-single 6/8, Fixed-Share 7/8, descriptor ridge 8/8; PEMS04 H192 remains the main blocker. |
| Online adaptation could be used to hide weak static cells. | shortcut reliance / methodology fabrication risk | Added M14 `online_stack_portfolio.py`, a delayed-feedback portfolio over stackers and strong baselines, with meta-parameters tuned only on the past split. | M14 clears Fixed-Share and descriptor ridge 8/8 but still fails validation-single 6/8. |
| A simpler sensor method could be overlooked after online portfolioing. | frame-lock risk | Added M15 `sensor_horizon_affine_significance.py`, fixing the method class to horizon-wise affine calibration and applying the same block/FDR gate. | M15 is the cleanest candidate but still fails: validation-single 7/8, Fixed-Share 8/8, descriptor ridge 8/8; Electricity H192 remains the only strict blocker. |
| A selective method could hide failures by abstaining too often. | shortcut reliance / methodology fabrication risk | Added M16 `selective_horizon_affine_gate.py`, requiring active-cell FDR, inactive no-harm, and minimum active coverage. | M16 active cells pass 2/2 and inactive no-harm is true, but active coverage is only 2/8 so the gate fails. |
| A selective method could pass by admitting fragile active cells. | shortcut reliance / result-shopping risk | Added M17 `practical_selective_horizon_affine_gate.py`, which activates only when the past split shows p<=0.05 and >=5% practical improvement versus both validation-single and delayed Fixed-Share; then applies active-cell BH/FDR versus validation-single, Fixed-Share, and descriptor ridge. | M17 passes: active 4/16 cells (Electricity H96, Traffic H96, AQWan H96, AQWan H192), inactive no-harm is true, and active FDR pass counts are 4/4 for all three baselines. |
| Reproduction entrypoint depended on missing `context.npy` artifacts. | methodology fabrication / reproducibility failure | `paper_ready/REPRODUCE.md` now starts from `produce_m1c_predictions`, regenerates M1c oracle/online/probe artifacts, then reruns M2-M5. | Rebuilt M1c artifacts locally and reran M2-M5. |
| `m1c_summary.json` was listed in the manifest but not regenerated by the packager. | methodology fabrication | `paper_ready.py` now writes `oracle_drift/m1c_summary.json` from per-dataset summaries. | `m1c_summary.json` scope now says it is regenerated from per-dataset artifacts. |
| Docs still read like the original full SOTA method was active. | frame-lock / hallucinated result risk | `README.md`, `PROPOSAL.md`, and `REPORT.md` now mark the full design as historical and the active route as the M17 scoped selective candidate. | `PROPOSAL.md` top block, §3, §5, §6.3, §7, §9, §11, §12; `REPORT.md` M17 addendum and Parts 8-9. |
| AME-TS venue-status detail was not verified. | citation hallucination risk | Downgraded to `arXiv'26 (venue status unverified here)`. | Web spot-check found arXiv but no reliable primary venue status. |

## Failure-Mode Checklist

| Mode | Status after correction | Audit note |
|---|---|---|
| implementation bugs | CLEAR for active empirical claims | The material bugs found were fixed: M2 test-tuned Fixed-Share and M4 non-blocked p-values. Device selection in training code already follows CUDA -> MPS -> CPU and uses `pin_memory` only for CUDA. |
| hallucinated results | CLEAR for active scoped claims | M2-M18 artifacts were regenerated or traced after strengthened baselines. Active tables in `REPRODUCE.md`, `REPORT.md`, `SUBMISSION_TRACE.md`, and JSON artifacts match the regenerated results. |
| shortcut reliance | CLEAR for M17 scoped route; BLOCKS original full route | Delayed-feedback correction removed an online-learning shortcut, and validation-single hardening exposed that the original full method loses to a simple recent-past champion. M17 avoids fragile cells by pre-test abstention and active-cell FDR. |
| bug-as-insight reframing | CLEAR | Stress improvements are now separated from FDR-stable method evidence; docs do not narrate non-significant M4 results as a positive mechanism. |
| methodology fabrication | CLEAR for active empirical claims | Reproduction now includes M1c generation, M2/M3/M4 reruns, M5 packaging, M6 main-track audit, robust M4 testing, and manifest regeneration. |
| frame-lock | CLEAR for active scoped claim | The proposal/report now records the original finance/SOTA framing as retired and separates M17's scoped positive route from the failed learned-router narrative. |
| citation hallucinations | PASS WITH NOTES | High-risk 2026 venue/existence claims were spot-checked and one unverified venue detail was downgraded. This is not a 100% bibliography audit because PRISM currently has no final reference list/manuscript bibliography in this directory. |

## Rerun Evidence

Commands rerun from the repository root:

```bash
python3 -m experiments.PRISM.produce_m1c_predictions
for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange; do
  python3 -m experiments.PRISM.oracle_drift \
    --results-root external/TSLib/results \
    --output-dir experiments/PRISM/oracle_drift/M1C_${ds}_L96_H96_target_last \
    --dataset M1C_${ds} --lookback 96 --horizon 96 \
    --models RidgeCov TargetRidge Trend Seasonal EWM \
    --target-channel -1 --include-anchors
  python3 -m experiments.PRISM.online_learning \
    --losses-csv experiments/PRISM/oracle_drift/M1C_${ds}_L96_H96_target_last/window_losses.csv \
    --output-dir experiments/PRISM/oracle_drift/M1C_${ds}_L96_H96_target_last
  python3 -m experiments.PRISM.descriptor_probe \
    --oracle-dir experiments/PRISM/oracle_drift/M1C_${ds}_L96_H96_target_last \
    --output-dir experiments/PRISM/oracle_drift/M1C_${ds}_L96_H96_target_last \
    --dataset M1C_${ds}
done
python3 -m experiments.PRISM.router_viability
python3 -m experiments.PRISM.champion_risk_gate
python3 -m experiments.PRISM.calibrated_stack_gate
python3 -m experiments.PRISM.drift_beta_loop
python3 -m experiments.PRISM.ablations_significance
python3 -m experiments.PRISM.paper_ready
python3 -m experiments.PRISM.main_track_audit
python3 -m experiments.PRISM.produce_m1c_predictions --horizon 192 --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange
for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange; do
  python3 -m experiments.PRISM.oracle_drift \
    --results-root external/TSLib/results \
    --output-dir experiments/PRISM/oracle_drift/M1C_${ds}_L96_H192_target_last \
    --dataset M1C_${ds} --lookback 96 --horizon 192 \
    --models RidgeCov TargetRidge Trend Seasonal EWM \
    --target-channel -1 --include-anchors
  python3 -m experiments.PRISM.descriptor_probe \
    --oracle-dir experiments/PRISM/oracle_drift/M1C_${ds}_L96_H192_target_last \
    --output-dir experiments/PRISM/oracle_drift/M1C_${ds}_L96_H192_target_last \
    --dataset M1C_${ds} --horizon 192
done
python3 -m experiments.PRISM.router_viability --horizon 192 --output-dir experiments/PRISM/router_viability_h192
python3 -m experiments.PRISM.champion_risk_gate --horizon 192 --output-dir experiments/PRISM/champion_risk_gate_h192
python3 -m experiments.PRISM.calibrated_stack_gate --horizon 192 --output-dir experiments/PRISM/calibrated_stack_gate_h192
python3 -m experiments.PRISM.produce_m1c_predictions --pool expanded --results-root external/TSLib/results_prism_expanded --horizon 96 --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange
python3 -m experiments.PRISM.produce_m1c_predictions --pool expanded --results-root external/TSLib/results_prism_expanded --horizon 192 --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange
for h in 96 192; do
  for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange; do
    python3 -m experiments.PRISM.oracle_drift \
      --results-root external/TSLib/results_prism_expanded \
      --output-dir experiments/PRISM/oracle_drift_expanded/M1C_${ds}_L96_H${h}_target_last \
      --dataset M1C_${ds} --lookback 96 --horizon ${h} \
      --models RidgeCov TargetRidge Trend Seasonal EWM EWMFast EWMSlow SeasonalOffset SeasonalDrift DampedTrend MeanRevert MeanRevertSlow SeasonalEWM SeasonalTrend EWMTrend \
      --target-channel -1 --include-anchors
    python3 -m experiments.PRISM.descriptor_probe \
      --results-root external/TSLib/results_prism_expanded \
      --oracle-dir experiments/PRISM/oracle_drift_expanded/M1C_${ds}_L96_H${h}_target_last \
      --output-dir experiments/PRISM/oracle_drift_expanded/M1C_${ds}_L96_H${h}_target_last \
      --dataset M1C_${ds} --horizon ${h}
  done
done
python3 -m experiments.PRISM.router_viability --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --output-dir experiments/PRISM/router_viability_expanded_h96
python3 -m experiments.PRISM.router_viability --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --horizon 192 --output-dir experiments/PRISM/router_viability_expanded_h192
python3 -m experiments.PRISM.champion_risk_gate --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --output-dir experiments/PRISM/champion_risk_gate_expanded_h96
python3 -m experiments.PRISM.champion_risk_gate --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --horizon 192 --output-dir experiments/PRISM/champion_risk_gate_expanded_h192
python3 -m experiments.PRISM.calibrated_stack_gate --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --output-dir experiments/PRISM/calibrated_stack_gate_expanded_h96
python3 -m experiments.PRISM.calibrated_stack_gate --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --horizon 192 --output-dir experiments/PRISM/calibrated_stack_gate_expanded_h192
python3 -m experiments.PRISM.produce_m1c_predictions --datasets Wind AQShunyi AQWan METRLA --pool expanded --results-root external/TSLib/results_prism_sensor_ext --horizon 96 --max-covariates 64 --shared-context
python3 -m experiments.PRISM.produce_m1c_predictions --datasets Wind AQShunyi AQWan METRLA --pool expanded --results-root external/TSLib/results_prism_sensor_ext --horizon 192 --max-covariates 64 --shared-context
for h in 96 192; do
  for ds in Wind AQShunyi AQWan METRLA; do
    python3 -m experiments.PRISM.oracle_drift \
      --results-root external/TSLib/results_prism_sensor_ext \
      --output-dir experiments/PRISM/oracle_drift_sensor_ext/M1C_${ds}_L96_H${h}_target_last \
      --dataset M1C_${ds} --lookback 96 --horizon ${h} \
      --models RidgeCov TargetRidge Trend Seasonal EWM EWMFast EWMSlow SeasonalOffset SeasonalDrift DampedTrend MeanRevert MeanRevertSlow SeasonalEWM SeasonalTrend EWMTrend \
      --target-channel -1
    python3 -m experiments.PRISM.descriptor_probe \
      --results-root external/TSLib/results_prism_sensor_ext \
      --oracle-dir experiments/PRISM/oracle_drift_sensor_ext/M1C_${ds}_L96_H${h}_target_last \
      --output-dir experiments/PRISM/oracle_drift_sensor_ext/M1C_${ds}_L96_H${h}_target_last \
      --dataset M1C_${ds} --horizon ${h}
  done
done
python3 -m experiments.PRISM.practical_selective_horizon_affine_gate
python3 -m experiments.PRISM.paper_ready
python3 -m experiments.PRISM.main_track_audit
```

Key regenerated results:

| Gate | Result |
|---|---|
| M2 learned router | FAIL overall: delayed contextual router passes 3/6 after adding validation-single; failures are ETTh2, ETTm1, and Exchange. |
| M3 dynamic beta / drift stress | PASS: stress-weighted loss improves on 4/6 datasets. |
| M4 block-robust ablation | FAIL: full-vs-plain and full-vs-validation-single survive BH/FDR on 0/6 datasets; full loses to validation-single on all six. |
| M5 packager | PASS as reproducible candidate-route artifact with M17 scoped route. |
| M6 main-track audit | ALLOW_SCOPED_MAIN_TRACK_SUBMISSION with zero blocking failures. |
| M7 H=192 router pilot | FAIL: delayed contextual router passes only 1/6 at H=192. |
| M8 expanded expert pool | FAIL for current router: H=96 passes 0/6; H=192 passes 1/6. |
| M9 champion-risk safe-switch | FAIL: base H96 0/6, base H192 0/6, expanded H96 0/6, expanded H192 1/6; robust safety validation mostly falls back to validation-single. |
| M10 calibrated forecast stacking | NEAR-MISS/FAIL: base H96 5/6, base H192 5/6, expanded H96 4/6, expanded H192 5/6; Exchange remains below validation-single. |
| M11 narrowed non-financial stack route | PASS as candidate route: H96 7/7 and H192 7/7; M12/M13 still block top-tier significance. |
| M12 calibrated-stack significance | FAIL strict top-tier gate: stack vs Fixed-Share 14/14 FDR, but stack vs validation-single 7/14 and stack vs descriptor ridge 10/14. |
| M13 high-dimensional sensor route | FAIL strict top-tier gate: stack vs descriptor ridge 8/8, Fixed-Share 7/8, validation-single 6/8; PEMS04 H192 and Electricity H192 remain blockers. |
| M14 delayed online stacker portfolio | FAIL strict top-tier gate: portfolio vs descriptor ridge 8/8, Fixed-Share 8/8, validation-single 6/8. |
| M15 fixed horizon-wise affine route | FAIL strict top-tier gate: horizon-affine vs descriptor ridge 8/8, Fixed-Share 8/8, validation-single 7/8; Electricity H192 remains the only strict blocker. |
| M16 selective horizon-wise affine no-harm gate | FAIL coverage gate: active cells pass 2/2 against all baselines and inactive no-harm is true, but active coverage is only 2/8. |
| M17 practical-effect selective horizon-wise affine gate | PASS scoped main-route gate: active 4/16 cells, inactive no-harm true, active validation-single 4/4, active Fixed-Share 4/4, active descriptor ridge 4/4. |
| M17 practical-effect threshold sensitivity | MIXED but transparent: 0% and 2.5% fail because active Fixed-Share FDR is 5/6; 5% passes; 10% fails because active coverage drops to 2/16. |
| M18 submission trace pack | PASS claim-trace gate: 5 manuscript claims, 3 figure/table trace entries, and 5 negative-claim constraints are generated in `submission_trace/`; final manuscript rendering and bibliography/PDF verification remain open tasks. |

## Citation Spot-Check Sources

- xCPD: OpenReview page verifies ICLR 2026 poster status and title, "Routing Channel-Patch Dependencies in Time Series Forecasting with Graph Spectral Decomposition": https://openreview.net/forum?id=uIPAuyno4Z
- Dynamic TMoE: arXiv verifies title/abstract and GitHub reports ICML 2026 poster/code status: https://arxiv.org/abs/2605.20678 and https://github.com/andone-07/Dynamic-TMoE
- MoHETS: arXiv verifies existence and title: https://arxiv.org/abs/2601.21866
- AME-TS: arXiv verifies existence and title; venue status not verified here: https://arxiv.org/abs/2605.25166
- FAME: arXiv verifies existence and title: https://arxiv.org/abs/2606.08896
- DeRegiME: arXiv verifies existence and title: https://arxiv.org/pdf/2605.19231
- ShifTS: OpenReview/GitHub verify ICLR 2026/concept-drift framing: https://openreview.net/forum?id=emkvZ7NanK and https://github.com/AdityaLab/ShifTS
- TimeBridge: OpenReview verifies ICML 2025 poster status: https://openreview.net/forum?id=pyKO0ZZ5lz

## Top-Journal Readiness Boundary

Ready as a scoped selective main-route candidate if the manuscript stays inside
the M17 claim set and explicitly reports all abstained cells, including METR-LA
and Wind. Not ready for any full-coverage/SOTA learned-router claim. The 5%
practical-effect threshold is supported as the only passing setting in the
current sensitivity grid, but the narrowness must be disclosed. Before final
submission, the manuscript still needs final rendered tables/figures from
`submission_trace/SUBMISSION_TRACE.md`, citation/reference verification, and
final PDF visual/caption verification.
