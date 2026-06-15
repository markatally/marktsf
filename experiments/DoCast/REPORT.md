# DoCast -- Experiment Report

**DoCast: Orthogonalized Scenario Forecasting with Controllable Future Covariates**
Date: 2026-06-15
Status: revised after ARS reviewer rejection; split hygiene, fair-control, and statistical-reporting fixes complete
Readiness: revised main-track candidate after adding a third strict-pass deep backbone; TimeXer remains a reported stability caveat

## Executive Summary

DoCast studies scenario forecasting when future covariates are controllable actions rather than purely exogenous known inputs. The revised package no longer uses internal gate labels as paper evidence. The scientific claim is narrower:

- Observational MISO forecasters can fail scenario queries under action-policy confounding.
- DoCast trains a structural response head with an orthogonalized R-learner objective.
- Claims about orthogonalization are now based on fair-control comparisons where D0/D1/D2 receive the same static controls and D1/D2 share item-specific response capacity.
- Real-data evidence is alignment with matched ATT proxies, not randomized causal ground truth.

## Main Results

### Semi-Synthetic Stress Test

At `gamma=0.5`, averaged over seeds 2021-2023:

| Setting | Metric | D0 | D1 | D2 |
|---|---:|---:|---:|---:|
| Hidden-confounder stress | Response RMSE | 0.6226 | 0.5477 | 0.2151 |
| Hidden-confounder stress | Sign-error rate | 100% | 100% | 0% |
| Hidden-confounder stress | WMAPE | 0.5284 | 0.5270 | 0.5310 |
| Shared static controls | Response RMSE | 0.1927 | 0.5103 | 0.2151 |

Interpretation: the original D2-vs-D0 gap is a hidden-confounder stress test. Under shared controls, D2 reduces response RMSE by 58.1% versus the fair D1 structural head; the controlled linear D0 is strong and is not claimed as an orthogonal-loss win.

### Real Matched-ATT Proxy Validation

| Metric | Favorita promotion | M5 markdown |
|---|---:|---:|
| Rows / units | 55,063 / 1,200 | 1,392,000 / 800 |
| Matched ATT proxy | 0.4518 [0.4268, 0.4776] | 0.0841 [0.0581, 0.1095] |
| D0 NEE | 0.3088 | 0.0775 |
| D2 NEE | 0.1051 | 0.0264 |
| NEE reduction | 66.0% | 65.9% |
| Mean paired NEE delta | 0.0496 [0.0400, 0.0596] | 0.0129 [0.0096, 0.0163] |
| Wilcoxon p | 8.78e-15 | 1.29e-06 |

SNAP remains a `c`-type exogenous non-degradation check, not evidence for controllable-action deconfounding.

### Fair-Control Backbone Protocol

| Backbone | D0 RMSE | D1 RMSE | D2 RMSE | D2 vs D1 | WMAPE change |
|---|---:|---:|---:|---:|---:|
| DLinear | 0.3788 | 0.3863 | 0.0986 | 74.4% | -1.04% |
| PatchTST | 0.3993 | 0.3990 | 0.3299 | 17.3% | +2.23% |
| TiDE | 0.3996 | 0.4005 | 0.2514 | 37.0% | -3.08% |
| Transformer | 0.6049 | 0.5082 | 0.2174 | 57.1% | -23.95% |
| TimeXer | 0.3952 | 0.3977 | 0.2892 | 27.2% | +2.14% |

DLinear, PatchTST, TiDE, and Transformer pass the strict seed-level protocol. TimeXer is a mean-pass boundary case: it improves mean response RMSE and mean WMAPE stays within tolerance, but only 2/3 seeds pass because one seed has +6.41% WMAPE change.

## Reviewer-Issue Coverage

| Reviewer concern | Current status |
|---|---|
| Missing identification assumptions | Addressed in `paper/main.tex` with consistency, ignorability, overlap, no-interference, and stable-measurement assumptions. |
| Natural experiment underspecified | Expanded with unit inclusion, matching target, controls, CIs, paired deltas, and robustness-grid summary. |
| D2 had extra controls | Addressed by M2 fair-control diagnostic and M6 fair-control backbone protocol. |
| Rounded-zero p-value reporting | Replaced with bounded/scientific p-value strings and confidence intervals. |
| Missing close prior work | Added causal forecasting, causal pricing, CRN, dynamic DML, TimeXer, and corrected TiDE metadata in `paper/references.bib`. |
| Overuse of readiness labels | Removed from manuscript claims; retained only as local audit metadata. |
| Backbone overclaiming | Corrected old all-backbone pass claim; M6 now has four strict-pass backbones and TimeXer is explicitly reported as a stability caveat. |

## Reproduce

```bash
conda run -n markquant python experiments/DoCast/m0_prior_art.py
conda run -n markquant python experiments/DoCast/m1_audit.py
conda run -n markquant python experiments/DoCast/m2_docast.py
conda run -n markquant python experiments/DoCast/m3_real_data.py
conda run -n markquant python experiments/DoCast/m6_backbone_sweep.py
conda run -n markquant python experiments/DoCast/m4_paper_ready.py
conda run -n markquant python experiments/DoCast/m5_main_track_audit.py
```

Primary outputs:

- `paper/main.tex`
- `paper/references.bib`
- `PAPER.md`
- `m2_docast/docast_summary.json`
- `m3_real_data/real_data_summary.json`
- `m6_backbone_sweep/backbone_sweep_summary.json`
- `m4_paper_ready/paper_ready_summary.json`
- `m5_main_track_audit/main_track_audit.json`

## Claim Boundary

DoCast supports intervention-oriented scenario forecasting under stated assumptions and overlap diagnostics. It does not establish unrestricted intervention validity, randomized causal effects on real retail data, HE-specific empirical validation, TimeXer strict seed-level stability, or leaderboard SOTA forecasting accuracy.
