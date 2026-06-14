# FlowRes Literature Collision Log

## Search Queries

- "multi resolution time series forecasting sampling frequency equivariant arxiv NeurIPS ICML"
- "sampling-rate equivariant time-series forecasting FlowState"
- "time series foundation models cross frequency forecasting LOTSA Moirai"
- "high frequency data gap time series foundation models millisecond resolution"
- "frequency domain forecasting aliasing downsampling time series"

## Closest Works and Claim-Level Overlap

| Work | Link | Claim overlap | Collision severity | Resolution |
|---|---|---:|---:|---|
| FEDformer | https://arxiv.org/abs/2201.12740 | Frequency-domain forecasting | Medium | Same-rate baseline |
| FreTS | local paper index | Frequency-domain MLP | Medium | Baseline |
| FITS | local paper index | Compact frequency linear model | Medium | Baseline |
| FreDF | local paper index | Frequency-domain loss | Medium | Baseline |
| SparseTSF | local paper index | Cross-period sparse modeling | Medium | Periodicity baseline |
| Multi-resolution diffusion TSF | https://iclr.cc/media/iclr-2024/Slides/17883_mrXtGgm.pdf | Multi-resolution generative forecasting | Medium-high | Architecture baseline; not same-process resolution audit |
| Moirai / Uni2TS | https://arxiv.org/abs/2402.02592 | Cross-frequency universal TSFM | High | TSFM baseline |
| GIFT-Eval | https://arxiv.org/abs/2410.10393 | Multi-frequency evaluation | Medium-high | Cross-dataset benchmark, not same-process transformation |
| TimeFound | https://arxiv.org/html/2503.04118v1 | Multi-resolution patching TSFM | Medium-high | Baseline if available |
| FlowState | https://research.ibm.com/publications/flowstate-sampling-rate-equivariant-time-series-forecasting | Sampling-rate equivariant TSFM | High | Direct architecture collision; FlowRes must be evaluation/protocol plus lightweight consistency |
| High-frequency data gap | https://arxiv.org/html/2603.16497v1 | New high-frequency dataset | Medium | Dataset context |

## Novelty Boundary

FlowRes survives by isolating **same-process sampling-rate shift and aliasing diagnostics**, not by claiming the first rate-equivariant model. FlowState is a direct collision for architecture.

## Uncertainty

FlowState publication details and code availability need re-checking. There may also be signal-processing literature on anti-alias forecasting consistency that should be cited.

## Novelty Confidence

**Medium-high for evaluation; medium for method.** Proceed if cross-resolution rank reversals are large and anti-alias consistency beats simple augmentation.
