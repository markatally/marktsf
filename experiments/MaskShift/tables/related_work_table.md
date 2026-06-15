# Related work comparison table

| Work | Venue/year | Occupied claim | MaskShift distinction |
| --- | --- | --- | --- |
| GRU-D | Scientific Reports 2018 | Mask/time-gap-aware RNN prediction | No mechanism-shift benchmark or rank-reversal audit. |
| BRITS | NeurIPS 2018 | Bidirectional imputation/prediction | Optimizes reconstruction/imputation rather than deployment mechanism shift. |
| SADI | AAAI 2025 | Diffusion imputation for partial blackouts | Blackout imputation competitor; not matched-rate multi-mechanism model-selection audit. |
| S4M | ICLR 2025 | Missing-aware S4 forecasting architecture | Architecture baseline; MaskShift is benchmark/theory and tests mask mechanisms as experimental factors. |
| ChannelTokenFormer | ICLR 2026 | Dependency/asynchrony/missingness architecture | Closest architecture collision; MaskShift avoids unified-architecture claims. |
| CRIB/MTSF-M | arXiv 2025 | Revisits MTSF with missing values | Motivates direct forecasting; does not isolate matched-rate mechanism shift/rank reversal. |
| Information blackouts | arXiv 2026 | MNAR traffic blackout state-space model | Closest blackout collision; MaskShift broadens to multiple mechanisms and ranking stability. |
| Robust prediction under missingness shifts | arXiv 2024 | Statistical missingness-shift theory | Non-TSF anchor; MaskShift operationalizes for forecasting benchmarks. |
