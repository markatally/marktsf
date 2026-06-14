# LagCast Literature Collision Log

## Search Queries

- "time series forecasting stale covariates latency asynchronous exogenous variables forecasting arxiv"
- "forecasting with delayed covariates release time leakage"
- "exogenous variables time series forecasting future covariates availability"
- "covariate informed transformer time series foundation model release lag"
- "time series forecasting benchmark covariate leakage release time"

## Closest Works and Claim-Level Overlap

| Work | Link | Claim overlap | Collision severity | Resolution |
|---|---|---:|---:|---|
| TFT | https://arxiv.org/abs/1912.09363 | Known-future and observed covariate interfaces | Medium | Baseline; does not validate release times |
| NBEATSx | https://arxiv.org/abs/2104.00473 | Exogenous-variable forecasting | Medium | Baseline |
| TimeXer | https://arxiv.org/html/2402.19072v1 | Exogenous-variable Transformer | Medium-high | Primary synchronized-covariate baseline |
| TiDE | https://arxiv.org/abs/2304.08424 | Dense covariate-aware forecaster | Medium | Baseline |
| Adapting pretrained TS models with exogenous variables | https://arxiv.org/abs/2503.12107 | Covariate adapters for pretrained models | Medium | Adapter baseline; no release-time contract |
| CITRAS | https://arxiv.org/pdf/2503.24007 | Covariate-informed Transformer | Medium-high | Covariate-fusion competitor |
| DAG exogenous forecasting | https://arxiv.org/abs/2509.14933 | Correlation modules for exogenous variables | Medium | Architecture baseline |
| CITRAS-FM | https://arxiv.org/html/2606.10798v1 | Covariate-informed foundation model | Medium-high | TSFM baseline |
| Rethinking TSFM evaluation | https://arxiv.org/abs/2510.13654 | Benchmark leakage and overlap | Medium | Broad leakage critique; LagCast isolates covariate release time |
| What-if TSF | https://openreview.net/forum?id=Zbt44sC4tE | Conditional/scenario future events | Low-medium | Scenario benchmark, not stale covariates |

## Novelty Boundary

LagCast must avoid "better exogenous attention." It claims that covariate values need a **release-time legality contract**, and that freshness is a predictive variable distinct from the covariate value.

## Uncertainty

Search did not surface a top-venue paper dedicated to delayed-release covariate legality for neural TSF. Some econometric nowcasting and vintage-data literature may be relevant and should be added before implementation.

## Novelty Confidence

**High for TSF benchmark/contract; medium for method.** The gate is dataset feasibility: at least two datasets need real or defensibly calibrated release-time metadata.
