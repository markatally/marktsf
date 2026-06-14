# DoCast -- Experiment Report

**DoCast: Orthogonalized Scenario Forecasting with Controllable Future Covariates**
Date: 2026-06-15
Status: revised after ARS reviewer rejection; fair-control and statistical-reporting fixes complete
Readiness: revised main-track candidate, not a guaranteed acceptance claim

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
| Hidden-confounder stress | Response RMSE | 0.6271 | 0.5485 | 0.2154 |
| Hidden-confounder stress | Sign-error rate | 100% | 100% | 0% |
| Hidden-confounder stress | WMAPE | 0.5609 | 0.5583 | 0.5542 |
| Shared static controls | Response RMSE | 0.1939 | 0.5105 | 0.2154 |

Interpretation: the original D2-vs-D0 gap is a hidden-confounder stress test. Under shared controls, D2 reduces response RMSE by 58.0% versus the fair D1 structural head; the controlled linear D0 is strong and is not claimed as an orthogonal-loss win.

### Real Matched-ATT Proxy Validation

| Metric | Favorita promotion | M5 markdown |
|---|---:|---:|
| Rows / units | 55,063 / 1,200 | 1,392,000 / 800 |
| Matched ATT proxy | 0.4518 [0.4268, 0.4776] | 0.0841 [0.0581, 0.1095] |
| D0 NEE | 0.3088 | 0.0775 |
| D2 NEE | 0.1051 | 0.0264 |
| NEE reduction | 66.0% | 65.9% |
| Mean paired NEE delta | 0.0496 [0.0400, 0.0596] | 0.0129 [0.0096, 0.0163] |
| Wilcoxon p | 1.36e-24 | 3.22e-16 |

SNAP remains a `c`-type exogenous non-degradation check, not evidence for controllable-action deconfounding.

### Fair-Control Backbone Protocol

| Backbone | D0 RMSE | D1 RMSE | D2 RMSE | D2 vs D1 | WMAPE change |
|---|---:|---:|---:|---:|---:|
| DLinear | 0.3908 | 0.4009 | 0.0785 | 80.4% | -1.20% |
| PatchTST | 0.4102 | 0.4132 | 0.1946 | 52.9% | -2.04% |
| TiDE | 0.4124 | 0.4146 | 0.1986 | 52.1% | -1.05% |
| TimeXer | 0.4068 | 0.4133 | 0.1832 | 55.7% | -1.54% |

## Reviewer-Issue Coverage

| Reviewer concern | Current status |
|---|---|
| Missing identification assumptions | Addressed in `paper/main.tex` with consistency, ignorability, overlap, no-interference, and stable-measurement assumptions. |
| Natural experiment underspecified | Expanded with unit inclusion, matching target, controls, CIs, paired deltas, and robustness-grid summary. |
| D2 had extra controls | Addressed by M2 fair-control diagnostic and M6 fair-control backbone protocol. |
| Rounded-zero p-value reporting | Replaced with bounded/scientific p-value strings and confidence intervals. |
| Missing close prior work | Added causal forecasting, causal pricing, CRN, dynamic DML, TimeXer, and corrected TiDE metadata in `paper/references.bib`. |
| Overuse of readiness labels | Removed from manuscript claims; retained only as local audit metadata. |

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
- `paper/main.pdf`
- `paper/references.bib`
- `PAPER.md`
- `m2_docast/docast_summary.json`
- `m3_real_data/real_data_summary.json`
- `m6_backbone_sweep/backbone_sweep_summary.json`
- `m4_paper_ready/paper_ready_summary.json`
- `m5_main_track_audit/main_track_audit.json`

## Claim Boundary

DoCast supports intervention-oriented scenario forecasting under stated assumptions and overlap diagnostics. It does not establish unrestricted intervention validity, randomized causal effects on real retail data, HE-specific empirical validation, or leaderboard SOTA forecasting accuracy.
