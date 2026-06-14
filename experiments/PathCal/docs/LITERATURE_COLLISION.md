# PathCal Literature Collision Log

## Search Queries

- "time series forecasting conformal prediction distribution shift conditional coverage arxiv OpenReview"
- "multi horizon time series conformal path coverage cumulative maximum"
- "flow conformal prediction multi-dimensional time series"
- "conformal risk control time series forecasting decision risk"
- "pathwise prediction sets time series forecasting"

## Closest Works and Claim-Level Overlap

| Work | Link | Claim overlap | Collision severity | Resolution |
|---|---|---:|---:|---|
| Conformal TS Forecasting | https://proceedings.neurips.cc/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html | TS prediction intervals | Medium | Baseline; mostly horizon/scalar coverage |
| EnbPI | https://arxiv.org/abs/2010.09107 | Dynamic model-agnostic TS intervals | Medium | Baseline |
| Adaptive CP for TS | https://proceedings.mlr.press/v162/zaffran22a.html | Adaptive conformal for TS | Medium | Baseline |
| SPCI | https://proceedings.mlr.press/v202/xu23r.html | Sequential nonconformity forecasting | Medium-high | Strong conformal baseline |
| HopCPT | https://openreview.net/forum?id=KTRwpWCMsC | Similarity-weighted conformal TS | Medium | Baseline |
| Conformal PID | https://arxiv.org/abs/2307.16895 | Online control of coverage | High | Strong adaptive baseline |
| Conformal Risk Control | https://arxiv.org/abs/2208.02814 | Risk-bounded prediction sets | High | General framework; PathCal specializes TS path functionals |
| ProbTS | https://openreview.net/forum?id=lk7SW0bH4x | Distributional benchmark | Medium | Evaluation baseline |
| Flow-based conformal multidimensional TS | https://openreview.net/forum?id=Uv3efQiPBZ | Multidimensional TS conformal sets | High | PathCal must target named path functionals, not generic multidimensional sets |
| Decision-calibrated prediction sets | https://arxiv.org/abs/2606.02081 | Downstream decision calibration | Medium-high | Avoid power-system-specific decision sets; use path-function validity |

## Novelty Boundary

PathCal survives if it focuses on **path-functionals as the estimand**: sum, max, ramp, threshold crossing, first-passage, drawdown. It should not claim a general replacement for multivariate conformal prediction.

## Uncertainty

Search found generic multidimensional conformal work but not a mature TSF benchmark centered on operational path functionals. Need a deeper citation pass around simultaneous prediction bands and functional data conformal methods.

## Novelty Confidence

**Medium-high.** The concept is defensible if the event-library and downstream decision mapping are pre-registered and advanced conformal baselines are included.
