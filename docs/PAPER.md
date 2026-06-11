# Paper Index

> 58 papers (52 local PDFs + 6 index-only: Enhancer, no open-access PDF; and the five 2026 MoE/regime additions #54–58, arXiv PDFs **not yet downloaded** — see Notes), covering NeurIPS / ICML / ICLR / AAAI / IJCAI / KDD / CIKM / ICASSP / TMLR / IJF / CSUR and arXiv preprints (2019–2026). Plus 4 classical foundation papers (see "Classical Foundations" section at the end, not counted in the 58-paper MECE index).
> **Strict MECE design**: single discriminating axis + three-level hierarchy + orthogonal attributes flattened to columns.
> - **Level 1 = Research role**: A Task-specific methods / B General foundation models & representations / C Analysis, surveys & resources.
> - **Level 2 = Task · Paradigm · Type**: A by task, B by paradigm, C by type (mutually exclusive and exhaustive within each).
> - **Level 3 = Backbone** (forecasting only, sub-divided due to volume): Transformer / Linear·MLP / State-space·Other / Model-agnostic enhancement.
> - **Focus attributes (columns, not classification layers)**: frequency-domain · channel · non-stationary · irregular sampling · multimodal · data-centric · feature engineering …, orthogonal and multi-valued, avoiding overlapping categories like "frequency Transformer".
> - **Application domain (column)**: general / finance.
> **Published date** = first arXiv submission (N/A if no preprint); **Accepted date** = first day of top venue (N/A for arXiv-only); **Size** = local PDF.

## Taxonomy Overview (Three Levels · Strict MECE)

| Level 1 | Level 2 | Level 3 | No. | Count |
|---------|---------|---------|-----|-------|
| **A Task-Specific Methods** | A1 Forecasting | a Backbone · Transformer | 1–12, 54 | 13 |
| | | b Backbone · Linear / MLP | 13–19 | 7 |
| | | c Backbone · State-space / Other | 20–22, 55–57 | 6 |
| | | d Model-agnostic Enhancement | 23–31 | 9 |
| | A2 Classification | — | 32 | 1 |
| | A3 Anomaly Detection | — | 33–34 | 2 |
| | A4 Imputation & Missing Values | — | 35–36 | 2 |
| | A5 Generation & Synthesis | — | 37 | 1 |
| | A6 Decision & Control | — | 38–39 | 2 |
| **B General Foundation Models & Representations** | B1 Foundation Models / Pre-training | — | 40–45, 58 | 7 |
| | B2 Self-supervised Representation | — | 46–47 | 2 |
| **C Analysis, Surveys & Resources** | C1 Survey | — | 48–50 | 3 |
| | C2 Empirical Analysis | — | 51–52 | 2 |
| | C3 Benchmarks & Datasets | — | 53 | 1 |

> **Decision axis**: ① Is it a "general / pre-trained / multi-task" model? Yes → B (split by paradigm into B1/B2). ② Otherwise, does it propose a method for a specific task? Yes → A (split by task). ③ Otherwise (survey / empirical / dataset) → C.
> **Domain distribution**: 49 general, 9 finance (#9 / #10 / #26 / #38 / #39 / #42 / #43 / #45 / #49). New entries #54–58 are numbered append-only (54+) to avoid renumbering cross-references; their MECE slots are shown in the taxonomy table above.

---

## A1 Forecasting · Transformer Backbone

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 1 | Informer - Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (AAAI 2021).pdf | 140 KB | AAAI | General | — | ProbSparse efficient attention | 2020-12-11 | 2021-02-02 | [2012.07436](https://arxiv.org/pdf/2012.07436) |
| 2 | Autoformer - Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting (NeurIPS 2021).pdf | 173 KB | NeurIPS | General | — | auto-correlation + series decomposition | 2021-06-24 | 2021-12-07 | [2106.13008](https://arxiv.org/pdf/2106.13008) |
| 3 | FEDformer - Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting (ICML 2022).pdf | 540 KB | ICML | General | freq-domain | frequency-enhanced decomposition | 2022-01-31 | 2022-07-19 | [2201.12740](https://arxiv.org/pdf/2201.12740) |
| 4 | Non-stationary Transformers - Exploring the Stationarity in Time Series Forecasting (NeurIPS 2022).pdf | 415 KB | NeurIPS | General | non-stationary | de-stationary attention | 2022-05-29 | 2022-11-29 | [2205.14415](https://arxiv.org/pdf/2205.14415) |
| 5 | Crossformer - Transformer Utilizing Cross-Dimension Dependency for Multivariate TSF (ICLR 2023).pdf | 766 KB | ICLR | General | channel | cross-dimension two-stage attention | 2021-08-02 | 2023-05-01 | [2108.00154](https://arxiv.org/pdf/2108.00154) |
| 6 | PatchTST - A Time Series is Worth 64 Words Long-term Forecasting with Transformers (ICLR 2023).pdf | 336 KB | ICLR | General | channel | patch + channel-independent | 2022-11-27 | 2023-05-01 | [2211.14730](https://arxiv.org/pdf/2211.14730) |
| 7 | iTransformer - Inverted Transformers Are Effective for Time Series Forecasting (ICLR 2024).pdf | 543 KB | ICLR | General | channel | inverted-dimension attention | 2023-10-10 | 2024-05-07 | [2310.06625](https://arxiv.org/pdf/2310.06625) |
| 8 | TFT - Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting (IJF 2021).pdf | 2.5 MB | IJF | General | — | gating + variable selection + multi-horizon attention (MISO-native) | 2019-12-19 | 2021-01-01 | [1912.09363](https://arxiv.org/pdf/1912.09363) |
| 9 | TimeXer - Empowering Transformers for Time Series Forecasting with Exogenous Variables (NeurIPS 2024).pdf | 11 MB | NeurIPS | General | channel | endogenous patch token + exogenous variate token cross-attention (MISO-native) | 2024-02-29 | 2024-12-10 | [2402.19072](https://arxiv.org/pdf/2402.19072) |
| 10 | TimeBridge - Non-Stationarity Matters for Long-term Time Series Forecasting (ICML 2025).pdf | 7.1 MB | ICML | General | non-stationary | non-stationarity dependency bridging | 2024-10-06 | 2025-07-15 | [2410.04442](https://arxiv.org/pdf/2410.04442) |
| 11 | Multi-period Learning for Financial Time Series Forecasting (KDD 2025).pdf | 5.8 MB | KDD | Finance | multi-period | multi-period learning framework (IRF/LWI/MAP) | 2025-11-07 | 2025-08-03 | [2511.08622](https://arxiv.org/pdf/2511.08622) |
| 12 | CAMEF - Causal-Augmented Multi-Modality Event-Driven Financial Forecasting (KDD 2025).pdf | 3.5 MB | KDD | Finance | multimodal | causal-augmented multimodal event-driven | 2025-02-07 | 2025-08-03 | [2502.04592](https://arxiv.org/pdf/2502.04592) |
| 54 | MoHETS - Long-term Time Series Forecasting with Mixture-of-Heterogeneous-Experts (Arxiv 2026).pdf | N/A ⬇ | arXiv | General | freq-domain · channel | sparse heterogeneous MoE in encoder-only Transformer (shared depthwise-conv + routed Fourier experts), covariate cross-attention, per-patch memoryless routing | 2026-01-29 | N/A (under review) | [2601.21866](https://arxiv.org/pdf/2601.21866) |

## A1 Forecasting · Linear / MLP Backbone

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 13 | LTSF-Linear - Are Transformers Effective for Time Series Forecasting (AAAI 2023).pdf | 1.5 MB | AAAI | General | — | single-layer linear (DLinear / NLinear) | 2022-05-26 | 2023-02-07 | [2205.13504](https://arxiv.org/pdf/2205.13504) |
| 14 | RLinear - Revisiting Long-term Time Series Forecasting An Investigation on Linear Mapping (Arxiv 2023).pdf | 62 KB | arXiv | General | non-stationary | linear mapping + RevIN | 2023-05-18 | N/A | [2305.10721](https://arxiv.org/pdf/2305.10721) |
| 15 | TiDE - Long-term Forecasting with TiDE Time-series Dense Encoder (TMLR 2024).pdf | 480 KB | TMLR | General | — | MLP dense encoder | 2023-04-17 | 2023-08-11 | [2304.08424](https://arxiv.org/pdf/2304.08424) |
| 16 | FreTS - Frequency-domain MLPs are More Effective Learners in TSF (NeurIPS 2023).pdf | 3.0 MB | NeurIPS | General | freq-domain | frequency-domain MLP learner | 2023-11-10 | 2023-12-12 | [2311.06184](https://arxiv.org/pdf/2311.06184) |
| 17 | FITS - Modeling Time Series with 10k Parameters (ICLR 2024).pdf | 1.9 MB | ICLR | General | freq-domain | complex frequency-domain linear interpolation | 2023-07-07 | 2024-05-07 | [2307.03756](https://arxiv.org/pdf/2307.03756) |
| 18 | SparseTSF - Modeling Long-term Time Series Forecasting with 1k Parameters (ICML 2024).pdf | 778 KB | ICML | General | — | cross-period sparse modeling | 2024-05-02 | 2024-07-23 | [2405.00946](https://arxiv.org/pdf/2405.00946) |
| 19 | NBEATSx - Neural Basis Expansion Analysis with Exogenous Variables for Forecasting (EnergyAI 2023).pdf | 1.3 MB | Energy&AI | General | — | N-BEATS basis expansion + exogenous variable fusion (MISO-native) | 2021-04-01 | 2023-01-01 | [2104.00473](https://arxiv.org/pdf/2104.00473) |

## A1 Forecasting · State-space / Other Backbone

> No new backbone proposed; operates on existing models' training objectives / test-time adaptation / data and feature processing.

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 20 | Time-SSM - Simplifying and Unifying State Space Models for Time Series (ICML 2025).pdf | 3.7 MB | ICML | General | — | simplified unified SSM | 2024-05-25 | 2025-07-15 | [2405.16312](https://arxiv.org/pdf/2405.16312) |
| 21 | Routing Channel-Patch Dependencies with Graph Spectral Decomposition (Arxiv 2026).pdf | 1.3 MB | arXiv | General | channel | graph spectral decomposition routing | 2026-03-14 | N/A | [2603.13702](https://arxiv.org/pdf/2603.13702) |
| 22 | ReIMTS - Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting (ICLR 2026).pdf | 3.9 MB | ICLR | General | irregular sampling | recursive multi-scale (resampling-free) | 2026-02-25 | 2026-04-23 | [2602.21498](https://arxiv.org/pdf/2602.21498) |
| 55 | DeRegiME - Deep Regime Mixtures for Probabilistic Forecasting under Distribution Shift (Arxiv 2026).pdf | N/A ⬇ | arXiv | General | non-stationary | sparse variational GP with nonstationary regime-mixing kernel + Student-t likelihood over **residual uncertainty** (not architecture routing) | 2026-05-19 | N/A | [2605.19231](https://arxiv.org/pdf/2605.19231) |
| 56 | Dynamic TMoE - A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting (ICML 2026).pdf | N/A ⬇ | ICML | General | non-stationary | MMD drift detection + dynamic heterogeneous expert spawning/pruning + recurrent temporal-memory router (training-time only, no TTA); code: [andone-07/Dynamic-TMoE](https://github.com/andone-07/Dynamic-TMoE) | 2026-05-20 | 2026 (ICML'26, dates TBA) | [2605.20678](https://arxiv.org/pdf/2605.20678) |
| 57 | FAME - Forecastability-Aware Mixture of Experts for Heterogeneous Time Series Forecasting (Arxiv 2026).pdf | N/A ⬇ | arXiv | General | data-centric | per-series forecastability fingerprint → cost-aware sparse routing over heterogeneous pool (incl. LightGBM); suitability mined from validation performance | 2026-06-08 | N/A | [2606.08896](https://arxiv.org/pdf/2606.08896) |

## A1 Forecasting · Model-agnostic Enhancement

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 23 | FreDF - Learning to Forecast in the Frequency Domain (ICLR 2025).pdf | 10 MB | ICLR | General | freq-domain | frequency-domain forecasting loss | 2024-02-04 | 2025-04-24 | [2402.02399](https://arxiv.org/pdf/2402.02399) |
| 24 | DynaTTA - Shift-Aware Test Time Adaptation and Benchmarking for TSF (ICML 2025).pdf | 2.5 MB | ICML | General | non-stationary | shift-aware test-time adaptation | 2025-06-10 | 2025-07-15 | [OpenReview](https://openreview.net/pdf/ffb1f549a5bd198c6e20071241d29260bbbf997a.pdf) |
| 25 | Proceed - Proactive Model Adaptation Against Concept Drift for Online TSF (KDD 2025).pdf | 2.1 MB | KDD | General | non-stationary | proactive model adaptation (concept drift) | 2024-12-11 | 2025-08-03 | [2412.08435](https://arxiv.org/pdf/2412.08435) |
| 26 | Enhancer - A Distribution-Aware Framework with Temporal-Relational Meta-Learning for Stock Prediction (KDD 2025).pdf | N/A | KDD | Finance | non-stationary | temporal-relational meta-learning | N/A | 2025-08-03 | [ACM DL](https://dl.acm.org/doi/10.1145/3711896.3736934) |
| 27 | DCATS - Empowering Time Series Forecasting with LLM-Agents (Arxiv 2025).pdf | 1.4 MB | arXiv | General | data-centric | LLM data-centric agent (cleaning / selection) | 2025-08-06 | N/A | [2508.04231](https://arxiv.org/pdf/2508.04231) |
| 28 | ELATE - Evolutionary Language Model for Automated Time-series Engineering (Arxiv 2025).pdf | 666 KB | arXiv | General | feature engineering | evolutionary LLM automated feature engineering | 2025-08-20 | N/A | [2508.14667](https://arxiv.org/pdf/2508.14667) |
| 29 | Tackling Time Series Forecasting Generalization via Mitigating Concept Drift (Arxiv 2026).pdf | 3.1 MB | arXiv | General | non-stationary | concept drift mitigation for improved generalization | 2025-10-16 | N/A | [2510.14814](https://arxiv.org/pdf/2510.14814) |
| 30 | DTAF - Towards Non-Stationary Time Series Forecasting with Temporal Stabilization (Arxiv 2025).pdf | 4.8 MB | arXiv | General | non-stationary | temporal stabilization | 2025-11-11 | N/A | [2511.08229](https://arxiv.org/pdf/2511.08229) |
| 31 | Partial Channel Dependence with Channel Masks for TSFM (ICASSP 2026).pdf | 3.2 MB | ICASSP | General | channel | channel masking (partial channel dependence) | 2024-10-30 | 2026-05-05 | [2410.23222](https://arxiv.org/pdf/2410.23222) |

## A2 Classification

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 32 | Evo-TFS - Evolutionary Time-Frequency Synthetic Minority Oversampling for Imbalanced Time Series Classification (Arxiv 2026).pdf | 5.1 MB | arXiv | General | resampling · oversampling | time-frequency evolutionary SMOTE | 2026-01-03 | N/A | [2601.01150](https://arxiv.org/pdf/2601.01150) |

## A3 Anomaly Detection

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 33 | General TSAD - Towards a General Time Series Anomaly Detector with Adaptive Bottlenecks and Dual Adversarial Decoders (ICLR 2025).pdf | 2.2 MB | ICLR | General | — | adaptive bottlenecks + dual adversarial decoding | 2024-05-24 | 2025-04-24 | [2405.15273](https://arxiv.org/pdf/2405.15273) |
| 34 | CATCH - Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching (ICLR 2025).pdf | 2.4 MB | ICLR | General | freq-domain · channel | channel-aware frequency patching | 2024-10-16 | 2025-04-24 | [2410.12261](https://arxiv.org/pdf/2410.12261) |

## A4 Imputation & Missing Values

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 35 | SADI - Self-attention-based Diffusion Model for Time-series Imputation in Partial Blackout (AAAI 2025).pdf | 834 KB | AAAI | General | — | self-attention diffusion (partial blackout) | 2025-03-03 | 2025-02-25 | [2503.01737](https://arxiv.org/pdf/2503.01737) |
| 36 | ImputeINR - Time Series Imputation via Implicit Neural Representations (IJCAI 2025).pdf | 3.5 MB | IJCAI | General | — | INR continuous-function imputation (high missing rate) | 2025-05-16 | 2025-08-16 | [2505.10856](https://arxiv.org/pdf/2505.10856) |

## A5 Generation & Synthesis

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 37 | Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking (NeurIPS 2025).pdf | 1.5 MB | NeurIPS | General | irregular sampling | irregular→regular completion-masking diffusion | 2025-10-08 | 2025-12-03 | [2510.06699](https://arxiv.org/pdf/2510.06699) |

## A6 Decision & Control

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 38 | AlphaQCM - Alpha Discovery in Finance with Distributional Reinforcement Learning (ICML 2025).pdf | 576 KB | ICML | Finance | — | distributional RL (quantile conditional moments QCM) | N/A | 2025-07-15 | [OpenReview](https://openreview.net/pdf?id=3sXMHlhBSs) |
| 39 | OPHR - Mastering Volatility Trading with Multi-Agent Deep Reinforcement Learning (NeurIPS 2025).pdf | 1.3 MB | NeurIPS | Finance | — | multi-agent (OP-Agent + HR-Agent) | N/A | 2025-12-03 | [OpenReview](https://openreview.net/pdf?id=2p4AtivyZz) |

## B1 Foundation Models / Pre-training

> General / pre-trained / multi-task models — inherently cross-task, placed in track B to keep track A task-exclusive.

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 40 | Mantis - Lightweight Calibrated Foundation Model for Time Series Classification (Arxiv 2025).pdf | 812 KB | arXiv | General | — | lightweight calibrated foundation model (classification-oriented) | 2025-02-21 | N/A | [2502.15637](https://arxiv.org/pdf/2502.15637) |
| 41 | Time Tracker - MoE-Enhanced Foundation Time Series Forecasting Model (Arxiv 2025).pdf | 1.2 MB | arXiv | General | — | MoE-enhanced foundation model | 2025-05-21 | N/A | [2505.15151](https://arxiv.org/pdf/2505.15151) |
| 42 | Pre-training Time Series Models with Stock Data Customization (KDD 2025).pdf | 1.0 MB | KDD | Finance | — | stock-customized pre-training tasks (SSPT) | 2025-06-20 | 2025-08-03 | [2506.16746](https://arxiv.org/pdf/2506.16746) |
| 43 | FinCast - A Foundation Model for Financial Time-Series Forecasting (CIKM 2025).pdf | 3.4 MB | CIKM | Finance | — | MoE decoder + PQ-Loss | 2025-08-27 | 2025-11-10 | [2508.19609](https://arxiv.org/pdf/2508.19609) |
| 44 | SEMPO - Lightweight Foundation Models for Time Series Forecasting (NeurIPS 2025).pdf | 1.8 MB | NeurIPS | General | — | lightweight spectral + prompt | 2025-10-22 | 2025-12-03 | [2510.19710](https://arxiv.org/pdf/2510.19710) |
| 45 | Kronos - A Foundation Model for the Language of Financial Markets (AAAI 2026).pdf | 11 MB | AAAI | Finance | — | candlestick tokenization + autoregressive pre-training | 2025-08-02 | 2026-01-20 | [2508.02739](https://arxiv.org/pdf/2508.02739) |
| 58 | AME-TS - Anchored Mixture-of-Experts for Time Series Forecasting (Arxiv 2026).pdf | N/A ⬇ | arXiv | General | — | series-level structural descriptors (forecastability/seasonality/trend/sparsity) → soft expert prior guiding token-level routing in a sparse TS foundation model | 2026-05-24 | N/A | [2605.25166](https://arxiv.org/pdf/2605.25166) |

## B2 Self-supervised Representation

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 46 | TimeCHEAT - A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis (AAAI 2025).pdf | 1.3 MB | AAAI | General | irregular sampling | channel harmony (multi-task ISMTS representation) | 2024-12-17 | 2025-02-25 | [2412.12886](https://arxiv.org/pdf/2412.12886) |
| 47 | FEI - Frequency-Masked Embedding Inference for Time Series Representation Learning (AAAI 2025).pdf | 1.3 MB | AAAI | General | freq-domain | frequency-masked embedding inference (non-contrastive) | 2024-12-30 | 2025-02-25 | [2412.20790](https://arxiv.org/pdf/2412.20790) |

## C1 Survey

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 48 | Channel Strategy Survey for Multivariate Time Series Forecasting (Arxiv 2025).pdf | 2.7 MB | arXiv | General | channel | systematic survey of channel strategies | 2025-02-15 | N/A | [2502.10721](https://arxiv.org/pdf/2502.10721) |
| 49 | A Survey of Explainable AI in Financial Time Series Forecasting (CSUR 2025).pdf | 1.0 MB | CSUR | Finance | explainability | financial time-series XAI taxonomy | 2024-07-22 | 2025-05-07 | [2407.15909](https://arxiv.org/pdf/2407.15909) |
| 50 | Deep Learning for Multivariate Time Series Imputation - A Survey (IJCAI 2025).pdf | 618 KB | IJCAI | General | — | multivariate time-series imputation survey | 2024-02-06 | 2025-08-16 | [2402.04059](https://arxiv.org/pdf/2402.04059) |

## C2 Empirical Analysis

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 51 | How Biased is Time Series Forecasting - Channel Dependence and Lookback Windows (Arxiv 2025).pdf | 8.9 MB | arXiv | General | channel · lookback window | channel dependence × lookback window bias | 2025-02-13 | N/A | [2502.09683](https://arxiv.org/pdf/2502.09683) |
| 52 | This Time is Different - Observability Perspective on TSFM (NeurIPS 2025).pdf | 3.6 MB | NeurIPS | General | — | TSFM observability-perspective analysis | 2025-05-20 | 2025-12-03 | [2505.14766](https://arxiv.org/pdf/2505.14766) |

## C3 Benchmarks & Datasets

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 53 | Time-IMM - A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series (NeurIPS 2025).pdf | 2.0 MB | NeurIPS | General | irregular sampling · multimodal | irregular multimodal MTS dataset / benchmark | 2025-06-12 | 2025-12-03 | [2506.10412](https://arxiv.org/pdf/2506.10412) |

---

## Classical Foundations

> The following 4 papers are foundational references cited in PRISM §4 (Theory), **not counted in the 53-paper MECE index above** (all pre-open-access era, 1989–2000).
> PDF column: ✅ downloaded; ❌ paywalled — requires institutional access or manual download via DOI link, then place in `paper/`.

| Abbrev. | Title | Author(s) | Journal | Year | DOI | PDF |
|---------|-------|-----------|---------|------|-----|-----|
| **Hamilton-MS** | A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle | Hamilton | *Econometrica* 57(2):357–384 | 1989 | [10.2307/1912559](https://doi.org/10.2307/1912559) | ❌ paywalled (Wiley / Econometrica, institutional access required) |
| **Ghahramani-SSSM** | Variational Learning for Switching State-Space Models | Ghahramani & Hinton | *Neural Computation* 12(4):831–864 | 2000 | [10.1162/089976600300015619](https://doi.org/10.1162/089976600300015619) | ❌ paywalled (MIT Press, institutional access required) |
| **Herbster-FShare** | Tracking the Best Expert | Herbster & Warmuth | *Machine Learning* 32(2):151–178 | 1998 | [10.1023/A:1007488714892](https://doi.org/10.1023/A:1007488714892) | ❌ paywalled (Springer, institutional access required) |
| **Diebold-DM** | Comparing Predictive Accuracy | Diebold & Mariano | *JBES* 13(3):253–263 | 1995 | [10.1080/07350015.1995.10524599](https://doi.org/10.1080/07350015.1995.10524599) | ✅ `Diebold & Mariano 1995 - Comparing Predictive Accuracy (NBER WP4390, JBES 1995).pdf` (2.1 MB) |

> Once manually downloaded, name the files as follows and place them in `paper/`, then change the corresponding ❌ to ✅:
> - `Hamilton 1989 - A New Approach Nonstationary Time Series Business Cycle (Econometrica).pdf`
> - `Ghahramani & Hinton 2000 - Variational Learning Switching State-Space Models (Neural Computation).pdf`
> - `Herbster & Warmuth 1998 - Tracking the Best Expert (Machine Learning).pdf`

---

## Notes

### MECE Design & Classification Rules
- **Single logic per level**: Level 1 = research role (A task methods / B general models / C meta-research); Level 2 = task (A) / paradigm (B) / type (C); Level 3 = backbone (forecasting only). Values within the same level are mutually exclusive and exhaustive.
- **Orthogonal attributes flattened to columns**: frequency-domain / channel / non-stationary / irregular sampling / multimodal etc. are "stackable focus points"; treating them as classification layers used to cause overlaps like "FEDformer = architecture ∩ frequency-domain"; now unified as "Focus" columns, eliminating overlap.
- **Multi-task paper placement (key trade-off)**: foundation models and general representations (Kronos / Mantis / TimeCHEAT / SEMPO …) are inherently cross-task and cannot be forced into a single task slot → dedicated track B accommodates them, keeping every A-track paper to one task. This is the cleanest partition achievable under "strict MECE with multi-task reality".
- **A/B boundary rule**: general / pre-trained / zero-shot / multi-task → B; task-specific methods → A. Examples: SEMPO, Kronos, FinCast (forecasting foundation models), Mantis (classification foundation model) all go to B1; TimeCHEAT (multi-task ISMTS representation) to B2; while Multi-period and Enhancer (single-task forecasting models) go to A1.
- **Level-3 a/b/c/d are mutually exclusive**: each forecasting paper either contributes a backbone family (a Transformer / b Linear·MLP / c State-space·Other) or proposes no new backbone and instead provides model-agnostic enhancement (d objective / adaptation / data & features).
- **MISO-native annotation**: TFT (#8), TimeXer (#9), NBEATSx (#19) are all MISO-native (inputs include exogenous / known-future covariates), annotated in the Key Mechanism column for quick identification of PRISM's direct competitors.
- **Within-group ordering**: ascending by accepted date (arXiv-only papers ordered by published date).

### Acceptance & Source Notes
- **#24 DynaTTA**: no standalone arXiv; PDF from ICML 2025 OpenReview; published date is OpenReview release date.
- **#26 Enhancer**: ACM DL paid-access only, no open preprint, no local PDF — index entry only (source column points to ACM DL).
- **#38 AlphaQCM / #39 OPHR**: no arXiv; PDFs from OpenReview; published date is N/A.
- **#15 TiDE**: TMLR rolling journal; accepted date taken from OpenReview Published date (2023-08-11).
- **#49 XAI Survey**: CSUR rolling journal; accepted date taken from Crossref online publication date (2025-05-07).
- **#31 Partial Channel**: workshop version published at NeurIPS 2024 TSALM Workshop; full version accepted at ICASSP 2026.
- **#45 Kronos**: AAAI 2026 runs 2026-01-20 to 27; accepted date is conference first day.
- **#22 ReIMTS**: ICLR 2026 main track runs 2026-04-23 to 25; accepted date is main track first day.
- **#36 ImputeINR / #50 Imputation Survey**: IJCAI 2025 runs 2025-08-16 to 22 (Montreal); accepted date is conference first day.
- **#35 SADI**: AAAI 2025 main track; arXiv v1 (2025-03-03) is later than the conference first day, so accepted date is still the conference first day (2025-02-25).
- **#46 TimeCHEAT**: accepted at AAAI 2025 (2025-02-25), slightly before the "past year" window, retained for topic relevance and top-venue status.
- **#8 TFT**: IJF (International Journal of Forecasting) rolling journal; accepted date is 2021-01-01 (vol 37 online first).
- **#19 NBEATSx**: Energy and AI (Elsevier) rolling journal; accepted date is 2023-01-01 (vol 10 online first).
- **#54–58 (2026 MoE/regime wave, added 2026-06-11)**: PRISM competitors per PROPOSAL §6.1–6.2; index-only for now — "N/A ⬇" in Size = arXiv PDF **not yet downloaded** (action item: download via the Source PDF links, name per Filename column, place in `paper/`, then replace "N/A ⬇" with the size). Numbered append-only (54+) to avoid renumbering the cross-referenced #1–53.
- **#54 MoHETS**: arXiv comment "Under review" (v2 2026-03-13); no venue found on OpenReview as of 2026-06-11.
- **#55 DeRegiME**: arXiv-only (Wood, Zohren, Roberts — Oxford); no venue traces as of 2026-06-11.
- **#56 Dynamic TMoE**: **accepted to ICML 2026** per arXiv v1 comment ("Accepted to ICML 2026"); conference dates not yet announced → accepted date provisional ("2026, dates TBA"). Public code verified live 2026-06-11. **Closest PRISM competitor; implemented baseline (PROPOSAL §7.3).**
- **#57 FAME**: arXiv-only as of 2026-06-11; industrial dataset (SNBC vending machines) likely not releasable.
- **#58 AME-TS**: official reviews visible at ICML 2026 FMSD workshop (OpenReview Submission 56) but no public acceptance decision → cited as arXiv preprint; re-check next cycle. Goes to B1 (foundation model) per the A/B boundary rule.
- **Accepted date rule**: first day of the main venue track (tutorials / workshops excluded); arXiv preprints use N/A.
- Total size: approx. **140 MB** (52 local PDFs; Enhancer and #54–58 excluded).
