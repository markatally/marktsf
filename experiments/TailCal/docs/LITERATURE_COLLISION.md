# TailCal Literature Collision Log

## Search Queries

- "tail risk extreme event time series forecasting deep learning probabilistic EVT arxiv"
- "rare event time series forecasting EVT transformer calibration"
- "extreme value loss time series forecasting KDD 2019"
- "tail conditional calibration probabilistic time series forecasting"
- "evidential tail aware transformer rare event time series EVEREST"

## Closest Works and Claim-Level Overlap

| Work | Link | Claim overlap | Collision severity | Resolution |
|---|---|---:|---:|---|
| Modeling Extreme Events in Time Series Prediction | https://www.kdd.org/kdd2019/accepted-papers/view/modeling-extreme-events-in-time-series-prediction | EVT-inspired loss for extremes | High | Foundational baseline; TailCal is calibration/evaluation-focused |
| DeepAR | https://arxiv.org/abs/1704.04110 | Probabilistic forecasting | Medium | Baseline |
| TFT | https://arxiv.org/abs/1912.09363 | Quantile forecasting | Medium | Baseline |
| Conformal PID | https://arxiv.org/abs/2307.16895 | Adaptive coverage | Medium | Baseline |
| ProbTS | https://openreview.net/forum?id=lk7SW0bH4x | Distributional benchmark | Medium | Evaluation context |
| Meta reweighting extremes | https://arxiv.org/html/2409.14232v1 | Learned tail weighting | Medium-high | Training baseline |
| AI for Extreme Events | https://arxiv.org/html/2406.20080v1 | Broad survey | Low | Context |
| EVEREST | https://arxiv.org/abs/2601.19022 | Evidential tail-aware Transformer + EVT head | High | Direct architecture collision; TailCal must be model-agnostic |
| Deep extreme mixture model | search result / 2026 preprint | Heavy-tail distributional model | Medium-high | Distributional baseline if accessible |
| Classical EVT/GARCH | classical | Tail modeling | Medium | Statistical baselines |

## Novelty Boundary

TailCal should not claim "first tail-aware TSF." It claims that **tail-conditional calibration and diagnostics are missing from standard TSF evaluation**, and that post-hoc EVT/conformal correction can repair fixed forecasters.

## Uncertainty

Some 2026 tail papers may be unpublished or difficult to access. Venue status and code availability for EVEREST and the deep extreme mixture model require re-checking before baseline finalization.

## Novelty Confidence

**Medium.** Scientific value is high, but method novelty is crowded. The proposal should proceed only if the diagnostic audit shows aggregate metrics seriously misrank tail safety.
