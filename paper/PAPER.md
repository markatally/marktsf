# SPECTRE 提案参考论文索引

> 共 50 篇（49 篇本地 PDF + Enhancer 仅索引，无开放获取 PDF），覆盖 NeurIPS / ICML / ICLR / AAAI / IJCAI / KDD / CIKM / ICASSP / TMLR / CSUR 及 arXiv 预印本（2020–2026）。  
> **严格 MECE 设计**：单一可判主轴 + 三级层次 + 正交属性降为列。  
> - **一级 = 研究角色**：A 任务专用方法 / B 通用基础模型与表示 / C 分析、综述与资源。  
> - **二级 = 任务·范式·类型**：A 按任务，B 按范式，C 按类型（各自互斥穷尽）。  
> - **三级 = 骨干**（仅「预测」因体量大再分）：Transformer / 线性·MLP / 状态空间·其他 / 模型无关增强。  
> - **关注属性（列，非分类层）**：频域 · 通道 · 非平稳 · 不规则采样 · 多模态 · 数据中心 · 特征工程 …，正交可多值，避免「频域 Transformer」式交叠。  
> - **应用域（列）**：通用 / 金融。  
> **发表日期** = arXiv 首次提交（无预印本填 N/A）；**收录日期** = 顶会/顶刊首日（arXiv 填 N/A）；**大小** = 本地 PDF。

## 分类总览（三级 · 严格 MECE）

| 一级 | 二级 | 三级 | 序号 | 篇数 |
|------|------|------|------|------|
| **A 任务专用方法** | A1 预测 | a 骨干 · Transformer | 1–10 | 10 |
| | | b 骨干 · 线性 / MLP | 11–16 | 6 |
| | | c 骨干 · 状态空间 / 其他 | 17–19 | 3 |
| | | d 模型无关增强 | 20–28 | 9 |
| | A2 分类 | — | 29 | 1 |
| | A3 异常检测 | — | 30–31 | 2 |
| | A4 插补与缺失值 | — | 32–33 | 2 |
| | A5 生成与合成 | — | 34 | 1 |
| | A6 决策与控制 | — | 35–36 | 2 |
| **B 通用基础模型与表示** | B1 基础模型 / 预训练 | — | 37–42 | 6 |
| | B2 自监督表示 | — | 43–44 | 2 |
| **C 分析、综述与资源** | C1 综述 | — | 45–47 | 3 |
| | C2 实证分析 | — | 48–49 | 2 |
| | C3 基准与数据集 | — | 50 | 1 |

> **判定主轴**：① 是否「通用 / 预训练 / 多任务」模型？是 → B（按范式分 B1/B2）。② 否则是否「提出方法解决某一任务」？是 → A（按任务分）。③ 否（综述 / 实证 / 数据集）→ C。  
> **应用域分布**：通用 41 篇，金融 9 篇（#9 / #10 / #23 / #35 / #36 / #39 / #40 / #42 / #46）。

---

## A1 预测 · Transformer 骨干

| # | 文件名 | 大小 | 顶会 / 顶刊 | 应用域 | 关注属性 | 关键机制 | 发表日期 | 收录日期 | 来源 PDF |
|---|--------|------|-------------|--------|----------|----------|----------|----------|----------|
| 1 | Informer - Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (AAAI 2021).pdf | 140 KB | AAAI | 通用 | — | ProbSparse 高效注意力 | 2020-12-11 | 2021-02-02 | [2012.07436](https://arxiv.org/pdf/2012.07436) |
| 2 | Autoformer - Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting (NeurIPS 2021).pdf | 173 KB | NeurIPS | 通用 | — | 自相关 + 序列分解 | 2021-06-24 | 2021-12-07 | [2106.13008](https://arxiv.org/pdf/2106.13008) |
| 3 | FEDformer - Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting (ICML 2022).pdf | 540 KB | ICML | 通用 | 频域 | 频域增强分解 | 2022-01-31 | 2022-07-19 | [2201.12740](https://arxiv.org/pdf/2201.12740) |
| 4 | Non-stationary Transformers - Exploring the Stationarity in Time Series Forecasting (NeurIPS 2022).pdf | 415 KB | NeurIPS | 通用 | 非平稳 | 去平稳注意力 | 2022-05-29 | 2022-11-29 | [2205.14415](https://arxiv.org/pdf/2205.14415) |
| 5 | Crossformer - Transformer Utilizing Cross-Dimension Dependency for Multivariate TSF (ICLR 2023).pdf | 766 KB | ICLR | 通用 | 通道 | 跨维度两阶段注意力 | 2021-08-02 | 2023-05-01 | [2108.00154](https://arxiv.org/pdf/2108.00154) |
| 6 | PatchTST - A Time Series is Worth 64 Words Long-term Forecasting with Transformers (ICLR 2023).pdf | 336 KB | ICLR | 通用 | 通道 | 分块 + 通道独立 | 2022-11-27 | 2023-05-01 | [2211.14730](https://arxiv.org/pdf/2211.14730) |
| 7 | iTransformer - Inverted Transformers Are Effective for Time Series Forecasting (ICLR 2024).pdf | 543 KB | ICLR | 通用 | 通道 | 倒置维度注意力 | 2023-10-10 | 2024-05-07 | [2310.06625](https://arxiv.org/pdf/2310.06625) |
| 8 | TimeBridge - Non-Stationarity Matters for Long-term Time Series Forecasting (ICML 2025).pdf | 7.1 MB | ICML | 通用 | 非平稳 | 非平稳依赖桥接 | 2024-10-06 | 2025-07-15 | [2410.04442](https://arxiv.org/pdf/2410.04442) |
| 9 | Multi-period Learning for Financial Time Series Forecasting (KDD 2025).pdf | 5.8 MB | KDD | 金融 | 多周期 | 多周期学习框架 (IRF/LWI/MAP) | 2025-11-07 | 2025-08-03 | [2511.08622](https://arxiv.org/pdf/2511.08622) |
| 10 | CAMEF - Causal-Augmented Multi-Modality Event-Driven Financial Forecasting (KDD 2025).pdf | 3.5 MB | KDD | 金融 | 多模态 | 因果增强多模态事件驱动 | 2025-02-07 | 2025-08-03 | [2502.04592](https://arxiv.org/pdf/2502.04592) |

## A1 预测 · 线性 / MLP 骨干

| # | 文件名 | 大小 | 顶会 / 顶刊 | 应用域 | 关注属性 | 关键机制 | 发表日期 | 收录日期 | 来源 PDF |
|---|--------|------|-------------|--------|----------|----------|----------|----------|----------|
| 11 | LTSF-Linear - Are Transformers Effective for Time Series Forecasting (AAAI 2023).pdf | 1.5 MB | AAAI | 通用 | — | 单层线性 (DLinear / NLinear) | 2022-05-26 | 2023-02-07 | [2205.13504](https://arxiv.org/pdf/2205.13504) |
| 12 | RLinear - Revisiting Long-term Time Series Forecasting An Investigation on Linear Mapping (Arxiv 2023).pdf | 62 KB | arXiv | 通用 | 非平稳 | 线性映射 + RevIN | 2023-05-18 | N/A | [2305.10721](https://arxiv.org/pdf/2305.10721) |
| 13 | TiDE - Long-term Forecasting with TiDE Time-series Dense Encoder (TMLR 2024).pdf | 480 KB | TMLR | 通用 | — | MLP 稠密编码器 | 2023-04-17 | 2023-08-11 | [2304.08424](https://arxiv.org/pdf/2304.08424) |
| 14 | FreTS - Frequency-domain MLPs are More Effective Learners in TSF (NeurIPS 2023).pdf | 3.0 MB | NeurIPS | 通用 | 频域 | 频域 MLP 学习器 | 2023-11-10 | 2023-12-12 | [2311.06184](https://arxiv.org/pdf/2311.06184) |
| 15 | FITS - Modeling Time Series with 10k Parameters (ICLR 2024).pdf | 1.9 MB | ICLR | 通用 | 频域 | 复频域线性插值 | 2023-07-07 | 2024-05-07 | [2307.03756](https://arxiv.org/pdf/2307.03756) |
| 16 | SparseTSF - Modeling Long-term Time Series Forecasting with 1k Parameters (ICML 2024).pdf | 778 KB | ICML | 通用 | — | 跨周期稀疏建模 | 2024-05-02 | 2024-07-23 | [2405.00946](https://arxiv.org/pdf/2405.00946) |

## A1 预测 · 状态空间 / 其他骨干

| # | 文件名 | 大小 | 顶会 / 顶刊 | 应用域 | 关注属性 | 关键机制 | 发表日期 | 收录日期 | 来源 PDF |
|---|--------|------|-------------|--------|----------|----------|----------|----------|----------|
| 17 | Time-SSM - Simplifying and Unifying State Space Models for Time Series (ICML 2025).pdf | 3.7 MB | ICML | 通用 | — | 简化统一 SSM | 2024-05-25 | 2025-07-15 | [2405.16312](https://arxiv.org/pdf/2405.16312) |
| 18 | Routing Channel-Patch Dependencies with Graph Spectral Decomposition (Arxiv 2026).pdf | 1.3 MB | arXiv | 通用 | 通道 | 图谱分解路由 | 2026-03-14 | N/A | [2603.13702](https://arxiv.org/pdf/2603.13702) |
| 19 | ReIMTS - Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting (ICLR 2026).pdf | 3.9 MB | ICLR | 通用 | 不规则采样 | 递归多尺度（免重采样） | 2026-02-25 | 2026-04-23 | [2602.21498](https://arxiv.org/pdf/2602.21498) |

## A1 预测 · 模型无关增强

> 不提出新骨干，而是作用于已有模型的训练目标 / 测试期适应 / 数据与特征处理。

| # | 文件名 | 大小 | 顶会 / 顶刊 | 应用域 | 关注属性 | 关键机制 | 发表日期 | 收录日期 | 来源 PDF |
|---|--------|------|-------------|--------|----------|----------|----------|----------|----------|
| 20 | FreDF - Learning to Forecast in the Frequency Domain (ICLR 2025).pdf | 10 MB | ICLR | 通用 | 频域 | 频域预测损失 | 2024-02-04 | 2025-04-24 | [2402.02399](https://arxiv.org/pdf/2402.02399) |
| 21 | DynaTTA - Shift-Aware Test Time Adaptation and Benchmarking for TSF (ICML 2025).pdf | 2.5 MB | ICML | 通用 | 非平稳 | Shift-aware 测试期适应 | 2025-06-10 | 2025-07-15 | [OpenReview](https://openreview.net/pdf/ffb1f549a5bd198c6e20071241d29260bbbf997a.pdf) |
| 22 | Proceed - Proactive Model Adaptation Against Concept Drift for Online TSF (KDD 2025).pdf | 2.1 MB | KDD | 通用 | 非平稳 | 主动模型适应（概念漂移） | 2024-12-11 | 2025-08-03 | [2412.08435](https://arxiv.org/pdf/2412.08435) |
| 23 | Enhancer - A Distribution-Aware Framework with Temporal-Relational Meta-Learning for Stock Prediction (KDD 2025).pdf | N/A | KDD | 金融 | 非平稳 | 时序-关系元学习 | N/A | 2025-08-03 | [ACM DL](https://dl.acm.org/doi/10.1145/3711896.3736934) |
| 24 | DCATS - Empowering Time Series Forecasting with LLM-Agents (Arxiv 2025).pdf | 1.4 MB | arXiv | 通用 | 数据中心 | LLM 数据中心代理（清洗 / 选择） | 2025-08-06 | N/A | [2508.04231](https://arxiv.org/pdf/2508.04231) |
| 25 | ELATE - Evolutionary Language Model for Automated Time-series Engineering (Arxiv 2025).pdf | 666 KB | arXiv | 通用 | 特征工程 | 进化 LLM 自动特征工程 | 2025-08-20 | N/A | [2508.14667](https://arxiv.org/pdf/2508.14667) |
| 26 | Tackling Time Series Forecasting Generalization via Mitigating Concept Drift (Arxiv 2026).pdf | 3.1 MB | arXiv | 通用 | 非平稳 | 缓解概念漂移提升泛化 | 2025-10-16 | N/A | [2510.14814](https://arxiv.org/pdf/2510.14814) |
| 27 | DTAF - Towards Non-Stationary Time Series Forecasting with Temporal Stabilization (Arxiv 2025).pdf | 4.8 MB | arXiv | 通用 | 非平稳 | 时间稳定化 | 2025-11-11 | N/A | [2511.08229](https://arxiv.org/pdf/2511.08229) |
| 28 | Partial Channel Dependence with Channel Masks for TSFM (ICASSP 2026).pdf | 3.2 MB | ICASSP | 通用 | 通道 | 通道掩码（部分通道依赖） | 2024-10-30 | 2026-05-05 | [2410.23222](https://arxiv.org/pdf/2410.23222) |

## A2 分类

| # | 文件名 | 大小 | 顶会 / 顶刊 | 应用域 | 关注属性 | 关键机制 | 发表日期 | 收录日期 | 来源 PDF |
|---|--------|------|-------------|--------|----------|----------|----------|----------|----------|
| 29 | Evo-TFS - Evolutionary Time-Frequency Synthetic Minority Oversampling for Imbalanced Time Series Classification (Arxiv 2026).pdf | 5.1 MB | arXiv | 通用 | 重采样 · 过采样 | 时频域进化 SMOTE | 2026-01-03 | N/A | [2601.01150](https://arxiv.org/pdf/2601.01150) |

## A3 异常检测

| # | 文件名 | 大小 | 顶会 / 顶刊 | 应用域 | 关注属性 | 关键机制 | 发表日期 | 收录日期 | 来源 PDF |
|---|--------|------|-------------|--------|----------|----------|----------|----------|----------|
| 30 | General TSAD - Towards a General Time Series Anomaly Detector with Adaptive Bottlenecks and Dual Adversarial Decoders (ICLR 2025).pdf | 2.2 MB | ICLR | 通用 | — | 自适应瓶颈 + 双对抗解码 | 2024-05-24 | 2025-04-24 | [2405.15273](https://arxiv.org/pdf/2405.15273) |
| 31 | CATCH - Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching (ICLR 2025).pdf | 2.4 MB | ICLR | 通用 | 频域 · 通道 | 通道感知频域分块 | 2024-10-16 | 2025-04-24 | [2410.12261](https://arxiv.org/pdf/2410.12261) |

## A4 插补与缺失值

| # | 文件名 | 大小 | 顶会 / 顶刊 | 应用域 | 关注属性 | 关键机制 | 发表日期 | 收录日期 | 来源 PDF |
|---|--------|------|-------------|--------|----------|----------|----------|----------|----------|
| 32 | SADI - Self-attention-based Diffusion Model for Time-series Imputation in Partial Blackout (AAAI 2025).pdf | 834 KB | AAAI | 通用 | — | 自注意力扩散（部分停电场景） | 2025-03-03 | 2025-02-25 | [2503.01737](https://arxiv.org/pdf/2503.01737) |
| 33 | ImputeINR - Time Series Imputation via Implicit Neural Representations (IJCAI 2025).pdf | 3.5 MB | IJCAI | 通用 | — | INR 连续函数插补（高缺失率） | 2025-05-16 | 2025-08-16 | [2505.10856](https://arxiv.org/pdf/2505.10856) |

## A5 生成与合成

| # | 文件名 | 大小 | 顶会 / 顶刊 | 应用域 | 关注属性 | 关键机制 | 发表日期 | 收录日期 | 来源 PDF |
|---|--------|------|-------------|--------|----------|----------|----------|----------|----------|
| 34 | Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking (NeurIPS 2025).pdf | 1.5 MB | NeurIPS | 通用 | 不规则采样 | 不规则→规则补全掩码扩散 | 2025-10-08 | 2025-12-03 | [2510.06699](https://arxiv.org/pdf/2510.06699) |

## A6 决策与控制

| # | 文件名 | 大小 | 顶会 / 顶刊 | 应用域 | 关注属性 | 关键机制 | 发表日期 | 收录日期 | 来源 PDF |
|---|--------|------|-------------|--------|----------|----------|----------|----------|----------|
| 35 | AlphaQCM - Alpha Discovery in Finance with Distributional Reinforcement Learning (ICML 2025).pdf | 576 KB | ICML | 金融 | — | 分布式 RL（量化条件矩 QCM） | N/A | 2025-07-15 | [OpenReview](https://openreview.net/pdf?id=3sXMHlhBSs) |
| 36 | OPHR - Mastering Volatility Trading with Multi-Agent Deep Reinforcement Learning (NeurIPS 2025).pdf | 1.3 MB | NeurIPS | 金融 | — | 多智能体（OP-Agent + HR-Agent） | N/A | 2025-12-03 | [OpenReview](https://openreview.net/pdf?id=2p4AtivyZz) |

## B1 基础模型 / 预训练

> 通用 / 预训练 / 多任务模型——天然跨任务，单列 B 轨以保 A 轨任务互斥。

| # | 文件名 | 大小 | 顶会 / 顶刊 | 应用域 | 关注属性 | 关键机制 | 发表日期 | 收录日期 | 来源 PDF |
|---|--------|------|-------------|--------|----------|----------|----------|----------|----------|
| 37 | Mantis - Lightweight Calibrated Foundation Model for Time Series Classification (Arxiv 2025).pdf | 812 KB | arXiv | 通用 | — | 轻量校准基础模型（面向分类） | 2025-02-21 | N/A | [2502.15637](https://arxiv.org/pdf/2502.15637) |
| 38 | Time Tracker - MoE-Enhanced Foundation Time Series Forecasting Model (Arxiv 2025).pdf | 1.2 MB | arXiv | 通用 | — | MoE 增强基础模型 | 2025-05-21 | N/A | [2505.15151](https://arxiv.org/pdf/2505.15151) |
| 39 | Pre-training Time Series Models with Stock Data Customization (KDD 2025).pdf | 1.0 MB | KDD | 金融 | — | 股票定制预训练任务 (SSPT) | 2025-06-20 | 2025-08-03 | [2506.16746](https://arxiv.org/pdf/2506.16746) |
| 40 | FinCast - A Foundation Model for Financial Time-Series Forecasting (CIKM 2025).pdf | 3.4 MB | CIKM | 金融 | — | MoE 解码器 + PQ-Loss | 2025-08-27 | 2025-11-10 | [2508.19609](https://arxiv.org/pdf/2508.19609) |
| 41 | SEMPO - Lightweight Foundation Models for Time Series Forecasting (NeurIPS 2025).pdf | 1.8 MB | NeurIPS | 通用 | — | 轻量谱 + 提示 | 2025-10-22 | 2025-12-03 | [2510.19710](https://arxiv.org/pdf/2510.19710) |
| 42 | Kronos - A Foundation Model for the Language of Financial Markets (AAAI 2026).pdf | 11 MB | AAAI | 金融 | — | K 线分词 + 自回归预训练 | 2025-08-02 | 2026-01-20 | [2508.02739](https://arxiv.org/pdf/2508.02739) |

## B2 自监督表示

| # | 文件名 | 大小 | 顶会 / 顶刊 | 应用域 | 关注属性 | 关键机制 | 发表日期 | 收录日期 | 来源 PDF |
|---|--------|------|-------------|--------|----------|----------|----------|----------|----------|
| 43 | TimeCHEAT - A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis (AAAI 2025).pdf | 1.3 MB | AAAI | 通用 | 不规则采样 | 通道和谐（多任务 ISMTS 表示） | 2024-12-17 | 2025-02-25 | [2412.12886](https://arxiv.org/pdf/2412.12886) |
| 44 | FEI - Frequency-Masked Embedding Inference for Time Series Representation Learning (AAAI 2025).pdf | 1.3 MB | AAAI | 通用 | 频域 | 频域掩码嵌入推断（非对比） | 2024-12-30 | 2025-02-25 | [2412.20790](https://arxiv.org/pdf/2412.20790) |

## C1 综述

| # | 文件名 | 大小 | 顶会 / 顶刊 | 应用域 | 关注属性 | 关键机制 | 发表日期 | 收录日期 | 来源 PDF |
|---|--------|------|-------------|--------|----------|----------|----------|----------|----------|
| 45 | Channel Strategy Survey for Multivariate Time Series Forecasting (Arxiv 2025).pdf | 2.7 MB | arXiv | 通用 | 通道 | 通道策略系统综述 | 2025-02-15 | N/A | [2502.10721](https://arxiv.org/pdf/2502.10721) |
| 46 | A Survey of Explainable AI in Financial Time Series Forecasting (CSUR 2025).pdf | 1.0 MB | CSUR | 金融 | 可解释性 | 金融时序 XAI 分类体系 | 2024-07-22 | 2025-05-07 | [2407.15909](https://arxiv.org/pdf/2407.15909) |
| 47 | Deep Learning for Multivariate Time Series Imputation - A Survey (IJCAI 2025).pdf | 618 KB | IJCAI | 通用 | — | 多元时序插补方法综述 | 2024-02-06 | 2025-08-16 | [2402.04059](https://arxiv.org/pdf/2402.04059) |

## C2 实证分析

| # | 文件名 | 大小 | 顶会 / 顶刊 | 应用域 | 关注属性 | 关键机制 | 发表日期 | 收录日期 | 来源 PDF |
|---|--------|------|-------------|--------|----------|----------|----------|----------|----------|
| 48 | How Biased is Time Series Forecasting - Channel Dependence and Lookback Windows (Arxiv 2025).pdf | 8.9 MB | arXiv | 通用 | 通道 · 回看窗口 | 通道依赖 × 回看窗口偏差 | 2025-02-13 | N/A | [2502.09683](https://arxiv.org/pdf/2502.09683) |
| 49 | This Time is Different - Observability Perspective on TSFM (NeurIPS 2025).pdf | 3.6 MB | NeurIPS | 通用 | — | TSFM 可观测性视角分析 | 2025-05-20 | 2025-12-03 | [2505.14766](https://arxiv.org/pdf/2505.14766) |

## C3 基准与数据集

| # | 文件名 | 大小 | 顶会 / 顶刊 | 应用域 | 关注属性 | 关键机制 | 发表日期 | 收录日期 | 来源 PDF |
|---|--------|------|-------------|--------|----------|----------|----------|----------|----------|
| 50 | Time-IMM - A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series (NeurIPS 2025).pdf | 2.0 MB | NeurIPS | 通用 | 不规则采样 · 多模态 | 不规则多模态 MTS 数据集 / 基准 | 2025-06-12 | 2025-12-03 | [2506.10412](https://arxiv.org/pdf/2506.10412) |

---

## 备注

### MECE 设计与判定
- **每级单一逻辑**：一级=研究角色（A 任务方法 / B 通用模型 / C 元研究）；二级=任务（A）/ 范式（B）/ 类型（C）；三级=骨干（仅预测）。同级取值互斥且穷尽。
- **正交属性降为列**：频域 / 通道 / 非平稳 / 不规则采样 / 多模态 等是「可叠加的关注点」，过去当分类层会导致「FEDformer = 架构 ∩ 频域」式交叠；现统一作「关注属性」列，根除重叠。
- **多任务论文的归宿（关键取舍）**：基础模型与通用表示（Kronos / Mantis / TimeCHEAT / SEMPO …）天然跨任务，无法塞进单一任务格 → 单设 **B 轨**容纳，从而 **A 轨每篇任务唯一**。这是「严格 MECE 在多任务现实下」能达到的最干净划分。
- **A/B 边界规则**：通用 / 预训练 / 零样本 / 多任务 → B；针对单一任务的专用方法 → A。例：SEMPO、Kronos、FinCast（预测向基础模型）、Mantis（分类向基础模型）均归 B1；TimeCHEAT（多任务 ISMTS 表示）归 B2；而 Multi-period、Enhancer（单任务专用预测模型）归 A1。
- **三级 a/b/c/d 互斥**：每篇预测论文要么贡献某一骨干家族（a Transformer / b 线性·MLP / c 状态空间·其他），要么不提新骨干而作模型无关增强（d 目标 / 适应 / 数据与特征）。
- **组内排序**：按收录日期升序（arXiv 论文以发表日期定位）。

### 收录与来源
- **#21 DynaTTA**：无独立 arXiv，PDF 取自 ICML 2025 OpenReview，发表日期为 OpenReview 发布日。
- **#23 Enhancer**：仅 ACM DL 付费版，无开放预印本，无本地 PDF，仅索引（来源列指向 ACM DL）。
- **#35 AlphaQCM / #36 OPHR**：无 arXiv，PDF 取自 OpenReview，发表日期填 N/A。
- **#13 TiDE**：TMLR 滚动期刊，收录日期取 OpenReview Published 日（2023-08-11）。
- **#46 XAI Survey**：CSUR 滚动期刊，收录日期取 Crossref 在线发表日（2025-05-07）。
- **#28 Partial Channel**：Workshop 版发表于 NeurIPS 2024 TSALM Workshop，完整版接收于 ICASSP 2026。
- **#42 Kronos**：AAAI 2026 会期 2026-01-20 至 27，收录日期取会议首日。
- **#19 ReIMTS**：ICLR 2026 主会期 2026-04-23 至 25，收录日期取主会首日。
- **#33 ImputeINR / #47 插补综述**：IJCAI 2025 会期 2025-08-16 至 22（Montreal），收录日期取会议首日。
- **#32 SADI**：AAAI 2025 主轨；arXiv v1（2025-03-03）晚于会议首日，收录日期仍取会议首日（2025-02-25）。
- **#43 TimeCHEAT**：收录 AAAI 2025（2025-02-25），略早于「近一年」窗口，因主题契合且属顶会而保留。
- **收录日期说明**：取各顶会主会场首日（tutorials / workshops 不计）；arXiv 预印本填 N/A。
- 总文件大小：约 **125 MB**（49 篇本地 PDF；Enhancer 未计）。
