# SupportCast Literature Collision Log

## Search Queries

- "time series foundation model in-context fine-tuning related examples support selection"
- "in-context time series forecasting foundation model support leakage"
- "benchmark leakage time series foundation models pretraining overlap arxiv"
- "negative transfer global models time series forecasting cross series pooling arxiv"
- "Chronos Moirai TimesFM in-context time series prompts"

## Closest Works and Claim-Level Overlap

| Work | Link | Claim overlap | Collision severity | Resolution |
|---|---|---:|---:|---|
| Chronos | https://arxiv.org/abs/2403.07815 | Pretrained probabilistic TSF | Medium | Zero-shot baseline |
| Moirai / Uni2TS | https://arxiv.org/abs/2402.02592 | Universal TSFM | Medium-high | Baseline and support API context |
| TimesFM | https://arxiv.org/abs/2310.10688 | Decoder-only TSFM | Medium | Baseline |
| GIFT-Eval | https://arxiv.org/abs/2410.10393 | TSFM benchmark | Medium | Benchmark context |
| Moirai-MoE | https://arxiv.org/html/2410.10469v1 | Sparse expert TSFM | Medium | Baseline if available |
| In-context fine-tuning for TSFMs | https://openreview.net/forum?id=uxzgGLWPj2 | Related-example prompting/adaptation | High | Direct method baseline; SupportCast audits selection legality |
| Lightweight online adaptation | https://arxiv.org/html/2502.12920v1 | Online TSFM adaptation | Medium | Adaptation baseline |
| Rethinking TSFM evaluation | https://arxiv.org/abs/2510.13654 | Leakage/overlap in TSFM evaluation | High | Broad critique; SupportCast isolates support retrieval |
| TIME benchmark | https://arxiv.org/html/2602.12147v2 | Fresh leakage-free TSFM benchmark | Medium-high | Benchmark context |
| Predictive heterogeneity | https://arxiv.org/html/2604.13748v1 | Adaptive pooling and negative transfer | Medium-high | Theoretical/diagnostic neighbor |
| DELPHYNE negative transfer | https://arxiv.org/html/2506.06288v1 | Cross-domain negative transfer in pretraining | Medium | Context for support harm |

## Novelty Boundary

SupportCast is viable only if it asks: **how much in-context support gain survives legal pre-origin support selection?** It should not claim a new TSFM or generic adapter.

## Uncertainty

Support APIs differ across TSFMs. Reproducibility depends on open weights and deterministic inference. ICML 2025 in-context fine-tuning paper must be read in full before implementing.

## Novelty Confidence

**Medium-high.** Direct collision is real, but the legality audit and negative-transfer abstention angle appears separable.
