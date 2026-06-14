# AvailAudit Literature Collision Log

## Search Queries

- "time series forecasting benchmark leakage evaluation invalid splits arxiv 2025"
- "time series foundation models benchmarking challenges leakage overlap"
- "forecast origin data availability benchmark covariate leakage time series"
- "GIFT-Eval non leaking pretraining dataset time series"
- "TIME benchmark task centric time series foundation model leakage"

## Closest Works and Claim-Level Overlap

| Work | Link | Claim overlap | Collision severity | Resolution |
|---|---|---:|---:|---|
| TFB | https://arxiv.org/abs/2403.20150 | Fair TSF benchmark framework | Medium-high | AvailAudit is a legality checker, not a benchmark suite |
| GIFT-Eval | https://arxiv.org/abs/2410.10393 | General TSFM benchmark and non-leaking pretraining | High | Use as benchmark context; avoid zero-shot dataset novelty claims |
| This Time is Different | local paper index / NeurIPS 2025 | TSFM observability critique | Medium | Related critique; AvailAudit operationalizes manifests |
| Rethinking TSFM evaluation | https://arxiv.org/abs/2510.13654 | TSFM leakage, overlap, benchmark risks | High | Strong collision; AvailAudit must contribute executable per-origin checks |
| TIME benchmark | https://arxiv.org/html/2602.12147v2 | Fresh task-centric leakage-free benchmark | High | Avoid "new benchmark"; audit existing tasks |
| High-fidelity multimodal benchmark | https://arxiv.org/html/2509.24789v1 | Causal and description leakage in multimodal TSF | Medium | Multimodal-specific; cite for leakage principles |
| No Champions in LTSF | https://openreview.net/forum?id=yO1JuBpTBB | Unreliable LTSF benchmarking | Medium | Supports need for benchmark rigor |
| Fast and Slow Streams OTSF | https://openreview.net/forum?id=I0n3EyogMi | Online TSF leakage from update/evaluation order | Medium | Related online leakage; AvailAudit broader availability |
| LastingBench | https://arxiv.org/html/2506.21614v1 | Benchmark leakage defense in LLMs | Low | Conceptual inspiration only |
| M5 competition docs | https://www.kaggle.com/c/m5-forecasting-accuracy | Strong availability-aware competition setup | Low | Positive benchmark example |

## Novelty Boundary

AvailAudit is viable only as an **executable forecast-origin availability protocol**: manifests plus loader checks plus mutation tests. A broad "benchmarks are leaky" paper is already occupied.

## Uncertainty

Some benchmark-leakage work is moving fast in 2026. Re-check TIME, Rethinking TSFM Evaluation, fev-bench, and any ICLR workshop follow-ups before writing a submission.

## Novelty Confidence

**Medium.** High practical value, but high external collision. The proposal passes only if corrected loaders materially change rankings or the checker catches widely used loader mistakes.
