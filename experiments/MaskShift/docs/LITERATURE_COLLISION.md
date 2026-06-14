# MaskShift Literature Collision Log

## Search Queries

- "time series forecasting missing not at random sensor blackout imputation forecasting arxiv 2025"
- "missingness mechanism shift time series forecasting MNAR MCAR blackout"
- "irregular multivariate time series forecasting missing blocks sensor malfunction"
- "ChannelTokenFormer asynchronous missing blocks time series forecasting ICLR 2026"
- "forecasting with missing values imputation then prediction benchmark"

## Closest Works and Claim-Level Overlap

| Work | Link | Claim overlap | Collision severity | Resolution |
|---|---|---:|---:|---|
| GRU-D | https://www.nature.com/articles/s41598-018-24271-9 | Missingness masks and gaps are informative | Medium | Baseline; not mechanism-shift audit |
| BRITS | https://papers.neurips.cc/paper/7911-brits-bidirectional-recurrent-imputation-for-time-series.pdf | End-to-end imputation for prediction | Medium | Baseline; imputation focus |
| Latent ODE | https://proceedings.neurips.cc/paper/8773-latent-ordinary-differential-equations-for-irregularly-sampled-time-series.pdf | Continuous-time irregular modeling | Low | Handles irregularity, not mechanism change |
| Neural CDE | https://proceedings.neurips.cc/paper_files/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html | General irregular MTS representation | Low | Baseline only |
| CSDI | https://arxiv.org/abs/2107.03502 | Conditional diffusion imputation | Medium | Strong imputer baseline |
| Raindrop | https://openreview.net/forum?id=Kwm8I7dU-l5 | Sensor malfunction/dropout | Medium-high | Must compare leave-sensor-out and graph variants |
| GraFITi | https://ojs.aaai.org/index.php/AAAI/article/view/29560 | Graph irregular forecasting | Medium | Baseline; no mask-cause thesis |
| S4M | https://openreview.net/forum?id=BkftcwIVmR | Missing-aware S4 forecasting | Medium-high | Architecture baseline |
| SADI | https://arxiv.org/html/2503.01737v1 | Partial blackout imputation | Medium-high | Covers blackout imputation; MaskShift targets forecast ranking under mechanism shift |
| ChannelTokenFormer | https://openreview.net/forum?id=r4ZamwBE8P | Unified dependency/asynchrony/missing blocks | High | Avoid new unified architecture claims; use as baseline if available |
| Modeling Information Blackouts in MNAR Time Series | https://arxiv.org/html/2601.01480v2 | MNAR blackouts and post-blackout forecasting | High | Strongest collision; MaskShift must broaden to mechanism-typed forecasting benchmark and rank instability |

## Novelty Boundary

MaskShift survives only if framed as: **matched missing rate is not a sufficient experimental control; missingness mechanism is a deployment factor that changes model rankings and forecast validity.** The correction must remain minimal and benchmark-oriented.

## Uncertainty

The MNAR blackout preprint is recent and close. Its final venue and full benchmark scope need re-check before M1. ChannelTokenFormer status was found as an OpenReview submission; acceptance and code availability are uncertain.

## Novelty Confidence

**Medium-high.** High for the mechanism-shift evaluation thesis; medium for any model component. Kill the proposal if the blackout paper already contains a broad mechanism-typed rank-instability benchmark across modern TSF baselines.
