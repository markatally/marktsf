# Paper Index

> 66 papers (61 local PDFs + 5 index-only: Enhancer, no open-access PDF; and four arXiv-only MoE/regime papers not yet downloaded — see Notes), covering NeurIPS / ICML / ICLR / AAAI / IJCAI / KDD / CIKM / ICASSP / TMLR / IJF / CSUR and arXiv preprints (2019–2026). Plus 4 classical foundation papers (see "Classical Foundations" section at the end, not counted in the 66-paper MECE index).
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
| **A Task-Specific Methods** | A1 Forecasting | a Backbone · Transformer | 1–13 | 13 |
| | | b Backbone · Linear / MLP | 14–21 | 8 |
| | | c Backbone · State-space / Other | 22–29 | 8 |
| | | d Model-agnostic Enhancement | 30–42 | 13 |
| | A2 Classification | — | 43 | 1 |
| | A3 Anomaly Detection | — | 44–45 | 2 |
| | A4 Imputation & Missing Values | — | 46–47 | 2 |
| | A5 Generation & Synthesis | — | 48 | 1 |
| | A6 Decision & Control | — | 49–50 | 2 |
| **B General Foundation Models & Representations** | B1 Foundation Models / Pre-training | — | 51–58 | 8 |
| | B2 Self-supervised Representation | — | 59–60 | 2 |
| **C Analysis, Surveys & Resources** | C1 Survey | — | 61–63 | 3 |
| | C2 Empirical Analysis | — | 64–65 | 2 |
| | C3 Benchmarks & Datasets | — | 66 | 1 |

> **Decision axis**: ① Is it a "general / pre-trained / multi-task" model? Yes → B (split by paradigm into B1/B2). ② Otherwise, does it propose a method for a specific task? Yes → A (split by task). ③ Otherwise (survey / empirical / dataset) → C.
> **Domain distribution**: 57 general, 9 finance (#9 / #10 / #33 / #49 / #50 / #53 / #54 / #56 / #62). All entries are numbered sequentially in MECE section order.

## 2026 ML Forecasting Top 10 (curated 2026-09-05)

> Scope: formally accepted 2026 main-conference papers whose central contribution advances time-series forecasting through adaptation, feature/pattern mining, or model architecture. Ranking balances novelty, technical depth, empirical breadth, reproducibility, and complementarity. NeurIPS 2026 is excluded because no official accepted-paper list was available by the cutoff date.

| Rank | Paper | Venue | Why selected |
|------|-------|-------|--------------|
| 1 | Dynamic TMoE | ICML 2026 | Drift-triggered expert growth/pruning plus a temporal-memory router directly addresses structural non-stationarity. |
| 2 | PULSE | ICML 2026 | Turns phase evolution into a generative inductive bias and couples it with distribution-aware simulation for non-stationary forecasting. |
| 3 | DropoutTS | ICML 2026 | A model-agnostic, sample-wise capacity controller converts spectral noise estimates into adaptive regularization. |
| 4 | TATO | ICLR 2026 | Adapts data to a frozen foundation model through fast, optimized transformation pipelines instead of repeated fine-tuning. |
| 5 | CoRA | ICLR 2026 | Restores multivariate correlation modeling to channel-independent TSFMs with a compact time-varying/time-invariant adapter. |
| 6 | Zeus | ICML 2026 | A tuning-free multi-task TSFM combines point-wise fidelity, multi-scale efficiency, and unified temporal masking. |
| 7 | GTR | ICLR 2026 | Retrieves global-cycle evidence beyond the local lookback window and plugs into heterogeneous forecasting backbones. |
| 8 | ReIMTS | ICLR 2026 | Models informative irregular sampling without destructive resampling through recursive multi-scale representations. |
| 9 | XLinear | AAAI 2026 | A lightweight MLP explicitly separates endogenous targets from exogenous drivers and supports practical MISO forecasting. |
| 10 | SARAF | KDD 2026 | Stationarity-controlled relevance/diversity retrieval avoids redundant or misleading historical evidence under regime shifts. |

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
| 13 | MoHETS - Long-term Time Series Forecasting with Mixture-of-Heterogeneous-Experts (Arxiv 2026).pdf | N/A ⬇ | arXiv | General | freq-domain · channel | sparse heterogeneous MoE in encoder-only Transformer (shared depthwise-conv + routed Fourier experts), covariate cross-attention, per-patch memoryless routing | 2026-01-29 | N/A (under review) | [2601.21866](https://arxiv.org/pdf/2601.21866) |

## A1 Forecasting · Linear / MLP Backbone

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 14 | LTSF-Linear - Are Transformers Effective for Time Series Forecasting (AAAI 2023).pdf | 1.5 MB | AAAI | General | — | single-layer linear (DLinear / NLinear) | 2022-05-26 | 2023-02-07 | [2205.13504](https://arxiv.org/pdf/2205.13504) |
| 15 | RLinear - Revisiting Long-term Time Series Forecasting An Investigation on Linear Mapping (Arxiv 2023).pdf | 62 KB | arXiv | General | non-stationary | linear mapping + RevIN | 2023-05-18 | N/A | [2305.10721](https://arxiv.org/pdf/2305.10721) |
| 16 | TiDE - Long-term Forecasting with TiDE Time-series Dense Encoder (TMLR 2024).pdf | 480 KB | TMLR | General | — | MLP dense encoder | 2023-04-17 | 2023-08-11 | [2304.08424](https://arxiv.org/pdf/2304.08424) |
| 17 | FreTS - Frequency-domain MLPs are More Effective Learners in TSF (NeurIPS 2023).pdf | 3.0 MB | NeurIPS | General | freq-domain | frequency-domain MLP learner | 2023-11-10 | 2023-12-12 | [2311.06184](https://arxiv.org/pdf/2311.06184) |
| 18 | FITS - Modeling Time Series with 10k Parameters (ICLR 2024).pdf | 1.9 MB | ICLR | General | freq-domain | complex frequency-domain linear interpolation | 2023-07-07 | 2024-05-07 | [2307.03756](https://arxiv.org/pdf/2307.03756) |
| 19 | SparseTSF - Modeling Long-term Time Series Forecasting with 1k Parameters (ICML 2024).pdf | 778 KB | ICML | General | — | cross-period sparse modeling | 2024-05-02 | 2024-07-23 | [2405.00946](https://arxiv.org/pdf/2405.00946) |
| 20 | NBEATSx - Neural Basis Expansion Analysis with Exogenous Variables for Forecasting (EnergyAI 2023).pdf | 1.3 MB | Energy&AI | General | — | N-BEATS basis expansion + exogenous variable fusion (MISO-native) | 2021-04-01 | 2023-01-01 | [2104.00473](https://arxiv.org/pdf/2104.00473) |
| 21 | XLinear - A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs (AAAI 2026).pdf | 834 KB | AAAI | General | exogenous variables · channel | endogenous global token as interaction hub + temporal/variate MLPs (MISO-native) | 2026-01-14 | 2026-01-22 | [2601.09237](https://arxiv.org/pdf/2601.09237) |

## A1 Forecasting · State-space / Other Backbone

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 22 | Time-SSM - Simplifying and Unifying State Space Models for Time Series (ICML 2025).pdf | 3.7 MB | ICML | General | — | simplified unified SSM | 2024-05-25 | 2025-07-15 | [2405.16312](https://arxiv.org/pdf/2405.16312) |
| 23 | Routing Channel-Patch Dependencies with Graph Spectral Decomposition (Arxiv 2026).pdf | 1.3 MB | arXiv | General | channel | graph spectral decomposition routing | 2026-03-14 | N/A | [2603.13702](https://arxiv.org/pdf/2603.13702) |
| 24 | ReIMTS - Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting (ICLR 2026).pdf | 3.9 MB | ICLR | General | irregular sampling | recursive multi-scale representation learning without destructive resampling | 2026-02-25 | 2026-04-23 | [2602.21498](https://arxiv.org/pdf/2602.21498) |
| 25 | DeRegiME - Deep Regime Mixtures for Probabilistic Forecasting under Distribution Shift (Arxiv 2026).pdf | N/A ⬇ | arXiv | General | non-stationary | sparse variational GP with nonstationary regime-mixing kernel + Student-t likelihood over **residual uncertainty** (not architecture routing) | 2026-05-19 | N/A | [2605.19231](https://arxiv.org/pdf/2605.19231) |
| 26 | FAME - Forecastability-Aware Mixture of Experts for Heterogeneous Time Series Forecasting (Arxiv 2026).pdf | N/A ⬇ | arXiv | General | data-centric | per-series forecastability fingerprint → cost-aware sparse routing over heterogeneous pool (incl. LightGBM); suitability mined from validation performance | 2026-06-08 | N/A | [2606.08896](https://arxiv.org/pdf/2606.08896) |
| 27 | PULSE - Generative Phase Evolution for Non-Stationary Time Series Forecasting (ICML 2026).pdf | 1.7 MB | ICML | General | non-stationary · distribution shift | phase-anchored disentanglement + generative Phase Router + statistic-aware mixup | 2026-05-16 | 2026-07-07 | [2605.16793](https://arxiv.org/pdf/2605.16793) |
| 28 | Dynamic TMoE - A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting (ICML 2026).pdf | 4.3 MB | ICML | General | non-stationary · adaptation | MMD drift detection + dynamic heterogeneous expert spawning/pruning + temporal-memory router | 2026-05-20 | 2026-07-07 | [2605.20678](https://arxiv.org/pdf/2605.20678) |
| 29 | SARAF - Stationarity-Aware Retrieval-Augmented Time Series Forecasting (KDD 2026).pdf | 1.0 MB | KDD | General | non-stationary · retrieval | stationarity-adaptive relevance/diversity selection + aggregation across historical regimes | 2026-06-02 | 2026-08-11 | [2606.04135](https://arxiv.org/pdf/2606.04135) |

## A1 Forecasting · Model-agnostic Enhancement

> No new backbone proposed; operates on existing models' training objectives / test-time adaptation / data and feature processing.

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 30 | FreDF - Learning to Forecast in the Frequency Domain (ICLR 2025).pdf | 10 MB | ICLR | General | freq-domain | frequency-domain forecasting loss | 2024-02-04 | 2025-04-24 | [2402.02399](https://arxiv.org/pdf/2402.02399) |
| 31 | DynaTTA - Shift-Aware Test Time Adaptation and Benchmarking for TSF (ICML 2025).pdf | 2.5 MB | ICML | General | non-stationary | shift-aware test-time adaptation | 2025-06-10 | 2025-07-15 | [OpenReview](https://openreview.net/pdf/ffb1f549a5bd198c6e20071241d29260bbbf997a.pdf) |
| 32 | Proceed - Proactive Model Adaptation Against Concept Drift for Online TSF (KDD 2025).pdf | 2.1 MB | KDD | General | non-stationary | proactive model adaptation (concept drift) | 2024-12-11 | 2025-08-03 | [2412.08435](https://arxiv.org/pdf/2412.08435) |
| 33 | Enhancer - A Distribution-Aware Framework with Temporal-Relational Meta-Learning for Stock Prediction (KDD 2025).pdf | N/A | KDD | Finance | non-stationary | temporal-relational meta-learning | N/A | 2025-08-03 | [ACM DL](https://dl.acm.org/doi/10.1145/3711896.3736934) |
| 34 | DCATS - Empowering Time Series Forecasting with LLM-Agents (Arxiv 2025).pdf | 1.4 MB | arXiv | General | data-centric | LLM data-centric agent (cleaning / selection) | 2025-08-06 | N/A | [2508.04231](https://arxiv.org/pdf/2508.04231) |
| 35 | ELATE - Evolutionary Language Model for Automated Time-series Engineering (Arxiv 2025).pdf | 666 KB | arXiv | General | feature engineering | evolutionary LLM automated feature engineering | 2025-08-20 | N/A | [2508.14667](https://arxiv.org/pdf/2508.14667) |
| 36 | Tackling Time Series Forecasting Generalization via Mitigating Concept Drift (Arxiv 2026).pdf | 3.1 MB | arXiv | General | non-stationary | concept drift mitigation for improved generalization | 2025-10-16 | N/A | [2510.14814](https://arxiv.org/pdf/2510.14814) |
| 37 | DTAF - Towards Non-Stationary Time Series Forecasting with Temporal Stabilization (Arxiv 2025).pdf | 4.8 MB | arXiv | General | non-stationary | temporal stabilization | 2025-11-11 | N/A | [2511.08229](https://arxiv.org/pdf/2511.08229) |
| 38 | Partial Channel Dependence with Channel Masks for TSFM (ICASSP 2026).pdf | 3.2 MB | ICASSP | General | channel | channel masking (partial channel dependence) | 2024-10-30 | 2026-05-05 | [2410.23222](https://arxiv.org/pdf/2410.23222) |
| 39 | GTR - Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval (ICLR 2026).pdf | 3.6 MB | ICLR | General | periodicity · retrieval | global-cycle retrieval + local/global alignment + lightweight 2D convolution | 2026-02-11 | 2026-04-23 | [2602.10847](https://arxiv.org/pdf/2602.10847) |
| 40 | TATO - Adapt Data to Model Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models (ICLR 2026).pdf | 1.2 MB | ICLR | General | data-centric · non-stationary · adaptation | optimized context slicing, normalization, and outlier-correction pipeline for frozen TSFMs | 2026-02-28 | 2026-04-23 | [2603.00629](https://arxiv.org/pdf/2603.00629) |
| 41 | CoRA - Boosting Time Series Foundation Models for Multivariate Forecasting through Correlation-aware Adapter (ICLR 2026).pdf | 2.5 MB | ICLR | General | channel · adaptation | low-rank time-varying/time-invariant correlation decomposition + heterogeneous-partial contrastive adapter | 2026-03-23 | 2026-04-23 | [2603.21828](https://arxiv.org/pdf/2603.21828) |
| 42 | DropoutTS - Sample-Adaptive Dropout for Robust Time Series Forecasting (ICML 2026).pdf | 2.4 MB | ICML | General | freq-domain · noise robustness · adaptation | spectral-residual noise scoring maps each sample to an adaptive dropout rate | 2026-01-29 | 2026-07-07 | [2601.21726](https://arxiv.org/pdf/2601.21726) |

## A2 Classification

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 43 | Evo-TFS - Evolutionary Time-Frequency Synthetic Minority Oversampling for Imbalanced Time Series Classification (Arxiv 2026).pdf | 5.1 MB | arXiv | General | resampling · oversampling | time-frequency evolutionary SMOTE | 2026-01-03 | N/A | [2601.01150](https://arxiv.org/pdf/2601.01150) |

## A3 Anomaly Detection

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 44 | General TSAD - Towards a General Time Series Anomaly Detector with Adaptive Bottlenecks and Dual Adversarial Decoders (ICLR 2025).pdf | 2.2 MB | ICLR | General | — | adaptive bottlenecks + dual adversarial decoding | 2024-05-24 | 2025-04-24 | [2405.15273](https://arxiv.org/pdf/2405.15273) |
| 45 | CATCH - Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching (ICLR 2025).pdf | 2.4 MB | ICLR | General | freq-domain · channel | channel-aware frequency patching | 2024-10-16 | 2025-04-24 | [2410.12261](https://arxiv.org/pdf/2410.12261) |

## A4 Imputation & Missing Values

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 46 | SADI - Self-attention-based Diffusion Model for Time-series Imputation in Partial Blackout (AAAI 2025).pdf | 834 KB | AAAI | General | — | self-attention diffusion (partial blackout) | 2025-03-03 | 2025-02-25 | [2503.01737](https://arxiv.org/pdf/2503.01737) |
| 47 | ImputeINR - Time Series Imputation via Implicit Neural Representations (IJCAI 2025).pdf | 3.5 MB | IJCAI | General | — | INR continuous-function imputation (high missing rate) | 2025-05-16 | 2025-08-16 | [2505.10856](https://arxiv.org/pdf/2505.10856) |

## A5 Generation & Synthesis

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 48 | Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking (NeurIPS 2025).pdf | 1.5 MB | NeurIPS | General | irregular sampling | irregular→regular completion-masking diffusion | 2025-10-08 | 2025-12-03 | [2510.06699](https://arxiv.org/pdf/2510.06699) |

## A6 Decision & Control

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 49 | AlphaQCM - Alpha Discovery in Finance with Distributional Reinforcement Learning (ICML 2025).pdf | 576 KB | ICML | Finance | — | distributional RL (quantile conditional moments QCM) | N/A | 2025-07-15 | [OpenReview](https://openreview.net/pdf?id=3sXMHlhBSs) |
| 50 | OPHR - Mastering Volatility Trading with Multi-Agent Deep Reinforcement Learning (NeurIPS 2025).pdf | 1.3 MB | NeurIPS | Finance | — | multi-agent (OP-Agent + HR-Agent) | N/A | 2025-12-03 | [OpenReview](https://openreview.net/pdf?id=2p4AtivyZz) |

## B1 Foundation Models / Pre-training

> General / pre-trained / multi-task models — inherently cross-task, placed in track B to keep track A task-exclusive.

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 51 | Mantis - Lightweight Calibrated Foundation Model for Time Series Classification (Arxiv 2025).pdf | 812 KB | arXiv | General | — | lightweight calibrated foundation model (classification-oriented) | 2025-02-21 | N/A | [2502.15637](https://arxiv.org/pdf/2502.15637) |
| 52 | Time Tracker - MoE-Enhanced Foundation Time Series Forecasting Model (Arxiv 2025).pdf | 1.2 MB | arXiv | General | — | MoE-enhanced foundation model | 2025-05-21 | N/A | [2505.15151](https://arxiv.org/pdf/2505.15151) |
| 53 | Pre-training Time Series Models with Stock Data Customization (KDD 2025).pdf | 1.0 MB | KDD | Finance | — | stock-customized pre-training tasks (SSPT) | 2025-06-20 | 2025-08-03 | [2506.16746](https://arxiv.org/pdf/2506.16746) |
| 54 | FinCast - A Foundation Model for Financial Time-Series Forecasting (CIKM 2025).pdf | 3.4 MB | CIKM | Finance | — | MoE decoder + PQ-Loss | 2025-08-27 | 2025-11-10 | [2508.19609](https://arxiv.org/pdf/2508.19609) |
| 55 | SEMPO - Lightweight Foundation Models for Time Series Forecasting (NeurIPS 2025).pdf | 1.8 MB | NeurIPS | General | — | lightweight spectral + prompt | 2025-10-22 | 2025-12-03 | [2510.19710](https://arxiv.org/pdf/2510.19710) |
| 56 | Kronos - A Foundation Model for the Language of Financial Markets (AAAI 2026).pdf | 11 MB | AAAI | Finance | — | candlestick tokenization + autoregressive pre-training | 2025-08-02 | 2026-01-22 | [2508.02739](https://arxiv.org/pdf/2508.02739) |
| 57 | AME-TS - Anchored Mixture-of-Experts for Time Series Forecasting (Arxiv 2026).pdf | N/A ⬇ | arXiv | General | — | series-level structural descriptors (forecastability/seasonality/trend/sparsity) → soft expert prior guiding token-level routing in a sparse TS foundation model | 2026-05-24 | N/A | [2605.25166](https://arxiv.org/pdf/2605.25166) |
| 58 | Zeus - Towards Tuning-Free Foundation Model for Time Series Analysis (ICML 2026).pdf | 3.9 MB | ICML | General | multi-task · zero-shot | point-wise multi-scale U-shaped Transformer + multi-objective temporal masking | 2026-07-02 | 2026-07-07 | [2607.01918](https://arxiv.org/pdf/2607.01918) |

## B2 Self-supervised Representation

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 59 | TimeCHEAT - A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis (AAAI 2025).pdf | 1.3 MB | AAAI | General | irregular sampling | channel harmony (multi-task ISMTS representation) | 2024-12-17 | 2025-02-25 | [2412.12886](https://arxiv.org/pdf/2412.12886) |
| 60 | FEI - Frequency-Masked Embedding Inference for Time Series Representation Learning (AAAI 2025).pdf | 1.3 MB | AAAI | General | freq-domain | frequency-masked embedding inference (non-contrastive) | 2024-12-30 | 2025-02-25 | [2412.20790](https://arxiv.org/pdf/2412.20790) |

## C1 Survey

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 61 | Channel Strategy Survey for Multivariate Time Series Forecasting (Arxiv 2025).pdf | 2.7 MB | arXiv | General | channel | systematic survey of channel strategies | 2025-02-15 | N/A | [2502.10721](https://arxiv.org/pdf/2502.10721) |
| 62 | A Survey of Explainable AI in Financial Time Series Forecasting (CSUR 2025).pdf | 1.0 MB | CSUR | Finance | explainability | financial time-series XAI taxonomy | 2024-07-22 | 2025-05-07 | [2407.15909](https://arxiv.org/pdf/2407.15909) |
| 63 | Deep Learning for Multivariate Time Series Imputation - A Survey (IJCAI 2025).pdf | 618 KB | IJCAI | General | — | multivariate time-series imputation survey | 2024-02-06 | 2025-08-16 | [2402.04059](https://arxiv.org/pdf/2402.04059) |

## C2 Empirical Analysis

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 64 | How Biased is Time Series Forecasting - Channel Dependence and Lookback Windows (Arxiv 2025).pdf | 8.9 MB | arXiv | General | channel · lookback window | channel dependence × lookback window bias | 2025-02-13 | N/A | [2502.09683](https://arxiv.org/pdf/2502.09683) |
| 65 | This Time is Different - Observability Perspective on TSFM (NeurIPS 2025).pdf | 3.6 MB | NeurIPS | General | — | TSFM observability-perspective analysis | 2025-05-20 | 2025-12-03 | [2505.14766](https://arxiv.org/pdf/2505.14766) |

## C3 Benchmarks & Datasets

| # | Filename | Size | Venue | Domain | Focus | Key Mechanism | Published | Accepted | Source PDF |
|---|----------|------|-------|--------|-------|---------------|-----------|----------|------------|
| 66 | Time-IMM - A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series (NeurIPS 2025).pdf | 2.0 MB | NeurIPS | General | irregular sampling · multimodal | irregular multimodal MTS dataset / benchmark | 2025-06-12 | 2025-12-03 | [2506.10412](https://arxiv.org/pdf/2506.10412) |

---

## Classical Foundations

> The following 4 papers are foundational references cited in PRISM §4 (Theory), **not counted in the 66-paper MECE index above** (all pre-open-access era, 1989–2000).
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
- **MISO-native annotation**: TFT (#8), TimeXer (#9), NBEATSx (#20), and XLinear (#21) are MISO-native (inputs include exogenous / known-future covariates), annotated in the Key Mechanism column for quick identification of PRISM's direct competitors.
- **Within-group ordering**: ascending by accepted date (arXiv-only papers ordered by published date).

### Acceptance & Source Notes
- **#31 DynaTTA**: no standalone arXiv; PDF from ICML 2025 OpenReview; published date is OpenReview release date.
- **#33 Enhancer**: ACM DL paid-access only, no open preprint, no local PDF — index entry only (source column points to ACM DL).
- **#49 AlphaQCM / #50 OPHR**: no arXiv; PDFs from OpenReview; published date is N/A.
- **#16 TiDE**: TMLR rolling journal; accepted date taken from OpenReview Published date (2023-08-11).
- **#62 XAI Survey**: CSUR rolling journal; accepted date taken from Crossref online publication date (2025-05-07).
- **#38 Partial Channel**: workshop version published at NeurIPS 2024 TSALM Workshop; full version accepted at ICASSP 2026.
- **#56 Kronos / #21 XLinear**: AAAI 2026 conference runs 2026-01-20 to 27; the main technical program begins 2026-01-22, used as the accepted date.
- **#24 ReIMTS / #39 GTR / #40 TATO / #41 CoRA**: ICLR 2026 begins 2026-04-23; this is used as the main-conference accepted date.
- **#27 PULSE / #28 Dynamic TMoE / #42 DropoutTS / #58 Zeus**: ICML 2026 tutorials are on 2026-07-06 and the main conference begins 2026-07-07; 2026-07-07 is used as the accepted date.
- **#29 SARAF**: accepted to the KDD 2026 Research Track; workshops/tutorials run 2026-08-09 to 10 and the main conference begins 2026-08-11, used as the accepted date.
- **#47 ImputeINR / #63 Imputation Survey**: IJCAI 2025 runs 2025-08-16 to 22 (Montreal); accepted date is conference first day.
- **#46 SADI**: AAAI 2025 main track; arXiv v1 (2025-03-03) is later than the conference first day, so accepted date is still the conference first day (2025-02-25).
- **#59 TimeCHEAT**: accepted at AAAI 2025 (2025-02-25), slightly before the "past year" window, retained for topic relevance and top-venue status.
- **#8 TFT**: IJF (International Journal of Forecasting) rolling journal; accepted date is 2021-01-01 (vol 37 online first).
- **#20 NBEATSx**: Energy and AI (Elsevier) rolling journal; accepted date is 2023-01-01 (vol 10 online first).
- **2026 MoE/regime wave (added 2026-06-11)**: #13 MoHETS, #25 DeRegiME, #26 FAME, #28 Dynamic TMoE, and #57 AME-TS are PRISM competitors per PROPOSAL §6.1–6.2. "N/A ⬇" means the arXiv PDF is not yet downloaded; Dynamic TMoE is now downloaded and conference-verified.
- **#13 MoHETS**: arXiv comment "Under review" (v2 2026-03-13); no venue found on OpenReview as of 2026-09-05.
- **#25 DeRegiME**: arXiv-only (Wood, Zohren, Roberts — Oxford); no venue traces as of 2026-09-05.
- **#28 Dynamic TMoE**: accepted as an ICML 2026 poster; public code and conference metadata re-verified 2026-09-05. **Closest PRISM competitor; implemented baseline (PROPOSAL §7.3).**
- **#26 FAME**: arXiv-only as of 2026-09-05; industrial dataset (SNBC vending machines) likely not releasable.
- **#57 AME-TS**: official reviews visible at the ICML 2026 FMSD workshop but no main-conference acceptance; retained as arXiv-only and classified in B1 under the A/B boundary rule.
- **Accepted date rule**: first day of the main venue track (tutorials / workshops excluded); arXiv preprints use N/A.
- Total size: approx. **195 MB** (61 local PDFs; #13 / #25 / #26 / #33 / #57 excluded).
