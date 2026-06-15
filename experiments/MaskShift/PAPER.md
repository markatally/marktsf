# MaskShift: Forecasting Under Missingness-Mechanism Shift

## Abstract

Forecasting papers usually stress-test missing inputs by changing how many observations are removed. Deployment failures often change a different quantity: why observations disappear. A weather station can stop reporting during storms, traffic detectors can go dark during congestion, telemetry can vanish under overload, and retired sensors can disappear permanently. These mechanisms can have the same missing rate while inducing different conditional mask laws. We study this failure mode as **missingness-mechanism shift**. MaskShift is not a new forecasting backbone. It is a benchmark and theory paper showing that matched missing rate is not a sufficient robustness certificate for deployment missingness. We define a controlled encoder-mask protocol, generate matched-rate masks for MCAR, block, value-triggered, volatility-triggered, blackout, and retirement mechanisms, and evaluate forecast risk and model-ranking stability. On Weather and Electricity, mechanism identity explains large degradation variance and can reverse model rankings; Traffic and AirConvection are mixed and are reported as limits, not hidden. With official TSLib PatchTST and TimeXer model classes under a custom MaskShift encoder-mask loop, the maximum degradation over operational mechanisms is 135.8% on Weather and 128.6% on Electricity over three seeds, with worst Kendall tau -1.0 in both datasets. We additionally adapt official ChannelTokenFormer_missing and S4M. ChannelTokenFormer_missing gives mixed missing-aware evidence, while S4M is a useful negative/contrastive baseline under the reduced MaskShift protocol. A typed-head correction does not pass our claim gate, so we report it as a negative diagnostic. The paper's central contribution is a reproducible audit protocol and evidence that missingness topology and conditional mask mechanism should be reported as first-class benchmark factors.

## 1. Introduction

Time-series forecasting systems are usually evaluated on clean histories or on histories corrupted by simple missingness processes. This is convenient but brittle. In deployed systems, missing values are rarely neutral. A sensor missing at random, a sensor missing during high values, a cluster missing during a blackout, and a channel missing after retirement all give the forecaster a different piece of information. The observed value tensor can be matched in size and the overall missing rate can be matched, yet the mask itself has changed meaning.

This paper studies a simple question: does robustness under MCAR or uniform block missingness certify robustness under deployment missingness? We argue and empirically show that the answer is no. The problem is not only that fewer values are observed. The problem is that the conditional law of the mask has shifted. In notation, the deployment distribution changes from a training mask mechanism \(p_A(M \mid X)\) to a deployment mechanism \(p_B(M \mid X)\), while the missing rate \(E[M]\) may remain fixed. A forecaster trained or selected under A can therefore be evaluated under a different conditional predictor at test time.

MaskShift is deliberately scoped as a benchmark/theory paper. It does not propose a new forecasting architecture, does not claim state-of-the-art clean forecasting accuracy, and does not claim that a lightweight typed correction solves missing-data forecasting. Instead, it contributes a failure-mode audit: controlled mask generators, rank-reversal metrics, degradation statistics, and a claim-to-evidence discipline that separates supported claims from tempting but unsupported method claims.

The core results are:

1. **Mechanism matters beyond rate.** In M1/M3, Weather and Electricity pass the mechanism-shift gate under matched missing rate. Across three seed offsets in M10, Weather passes in 2/3 seeds and Electricity in 3/3 seeds.
2. **Model selection can reverse.** In M9/M10, official TSLib PatchTST and TimeXer model classes, evaluated under the MaskShift encoder-mask protocol, show worst Kendall tau -1.0 on Weather and Electricity over three seeds.
3. **Missing-aware architecture coverage is now included.** M11 imports the official ChannelTokenFormer_missing class and M12/M14 import official S4M. CTF shows strong Weather sensitivity but weaker/non-significant Electricity sensitivity; S4M is a negative/contrastive result under both reduced and larger-reduced local protocols.
4. **The result is not only retirement.** M8 shows that Weather and Electricity pass even after excluding the retirement mechanism; value-triggered and blackout mechanisms remain material.
5. **The typed head is a negative result.** H3 fails: the typed/topology head improves Weather but does not produce robust, FDR-backed gains across datasets. It is therefore an ablation and limitation, not a contribution.
6. **Relative degradation needs care.** M7 originally exposed denominator-driven ratio spikes. M10 adds absolute delta, log ratio, and symmetric relative delta, and the paper reports these corrected metrics.

## 2. Formal Setup

Let \(X_{1:T} \in \mathbb{R}^{T \times C}\) be a multivariate time series and let \(M_{1:T} \in \{0,1\}^{T \times C}\) denote the input mask, where \(M_{t,c}=1\) means channel \(c\) is missing at time \(t\). A forecasting example is built from a lookback window \(X_{t-L:t-1}\) and target \(Y_t = X_{t+h-1,c^\star}\), or the full horizon \(X_{t:t+h-1}\) for multi-output neural backbones. The observed encoder input is \(X^{obs} = (1-M) \odot X\) plus a fill rule or mask features.

Most missingness robustness tests vary a scalar rate
\[
\rho = \frac{1}{LC}\sum_{i=t-L}^{t-1}\sum_{c=1}^C M_{i,c}.
\]
MaskShift varies the conditional mechanism. We compare mechanisms \(A\) and \(B\) such that \(\rho_A \approx \rho_B\), while \(p_A(M \mid X)\) and \(p_B(M \mid X)\) differ. The mechanisms in the current benchmark are:

- **MCAR**: independent random deletion.
- **Block**: contiguous channel-level blocks.
- **Value-triggered**: high normalized values are more likely to be missing.
- **Volatility-triggered**: large local changes increase missingness.
- **Blackout**: time blocks remove many channels simultaneously.
- **Retirement**: selected channels disappear late in the sequence.

All masks corrupt only encoder inputs. Forecast targets remain clean. This is important: the audit asks whether the forecaster's input robustness transfers, not whether the benchmark target is observable.

### Risk Under Mask Shift

Let \(Z=(X^{obs},M)\) and let \(\mu_A(Z)=\mathbb{E}_A[Y\mid Z]\) and \(\mu_B(Z)=\mathbb{E}_B[Y\mid Z]\) be squared-loss Bayes predictors under mechanisms A and B. Evaluating the Bayes predictor learned for A under the deployment law B gives
\[
R_B(\mu_A)-R_B(\mu_B)=\mathbb{E}_B[(\mu_A(Z)-\mu_B(Z))^2].
\]
The identity is elementary, but it pinpoints the benchmark failure. Matching the marginal missing rate is a robustness certificate only if it leaves the conditional Bayes predictor invariant on the deployment support.

### Rank-Reversal Proposition

Consider two forecasters \(f_1,f_2\) and two mechanisms A and B with matched missing rate. Let
\[
\Delta_A = R_A(f_1)-R_A(f_2), \qquad \Delta_B = R_B(f_1)-R_B(f_2).
\]
If \(\Delta_A < 0\) and \(\Delta_B > 0\), selection under A reverses under B. Such a reversal is possible whenever the mechanism-specific risk shift is model-dependent:
\[
[R_B(f_1)-R_A(f_1)] - [R_B(f_2)-R_A(f_2)] > -\Delta_A.
\]
This condition can hold even when \(\rho_A=\rho_B\). It is enough that one model uses information encoded in \(M\) or its topology differently from the other. This motivates reporting not only degradation magnitude but also Kendall rank correlation between the MCAR model order and each operational-mechanism model order.

### Role and Limit of Topology Statistics

Let \(S(M)\) be a statistic of the mask topology, such as gap age, block length, channel coverage, neighbor outage density, or a known mechanism label. If
\[
\mathbb{E}_A[Y \mid X^{obs}, S(M)] = \mathbb{E}_B[Y \mid X^{obs}, S(M)]
\]
on the deployment support, then conditioning on \(S\) removes the mechanism-shift term for squared-loss prediction. If the operational cause is not identifiable from observed topology, no small topology head can fully remove the shift. This is exactly what we see empirically: a typed/topology head helps Weather but does not pass as a general method claim.

## 3. Benchmark Protocol

### Datasets

The current submission draft uses Weather, Electricity, Traffic, and AirConvection. Weather and Electricity provide the strongest positive evidence. Traffic and AirConvection are retained because they prevent overgeneralization: both show mixed or weak mechanism effects under the current configuration.

The standard configuration is lookback 48, horizon 12, stride 8, target missing rate 0.35, and chronological train/test splits. The M9 official-architecture adaptation uses reduced train/test samples for deadline feasibility, which is reported explicitly in the checklist.

### Models

M1 evaluates lightweight non-neural variants trained under MCAR and tested under operational mechanisms: zero fill, forward fill, mask features, and topology features. M6 adds lite neural proxies: DLinearLite, PatchTSTLite, and GRU-DLite. M9 imports official TSLib PatchTST and TimeXer model classes from the pinned TSLib revision `4e938a1`, but the training and evaluation loop is MaskShift-specific. It masks encoder inputs only, keeps forecast targets clean, and does not claim to reproduce the full official TSLib benchmark protocol.

M11 imports `ChannelTokenFormer_missing` from the official ChannelTokenFormer repository at revision `b1c100e`. M12 imports the official S4M model class from revision `a718823`; a local device-port patch replaces one hard-coded `.cuda()` memory fetch with `.to(Q.device)` so the official architecture can run on CUDA, MPS, or CPU. These close the largest previous missing-aware baseline gap, but remain official-architecture adaptations rather than the full ChannelTokenFormer practical/irregular benchmark pipeline or the full S4M benchmark protocol.

### Metrics

The primary metrics are:

- MSE, MAE, and sMAPE for forecast error.
- Mechanism eta-squared for degradation variance attributed to mechanism identity.
- Kendall tau between MCAR model ranking and operational-mechanism ranking.
- Gate pass counts over seed offsets.
- Corrected severity metrics: absolute delta versus MCAR, log ratio versus MCAR, and symmetric relative delta.

The original relative degradation ratio is retained as a diagnostic only. It can explode when the MCAR denominator is small, which is why M10 reports corrected metrics for the paper tables.

### Statistical Reporting

M3 uses Benjamini-Hochberg FDR across claim families. M10 adds seed-level descriptive 95% confidence intervals for the core positive M1 datasets and for M9. M13 adds a hierarchy-aware nonparametric bootstrap over lightweight variants and test windows for the M1 aggregate claims. These intervals should be read as sprint-time robustness summaries, not as final large-sample inference. For bounded quantities such as eta-squared and Kendall tau, displayed intervals are clipped to their natural ranges.

## 4. Main Results

### 4.1 Mechanism Identity Explains Risk Shifts

M1 trains under MCAR and tests under matched-rate operational mechanisms. Weather and Electricity pass the mechanism-shift gate. In the original four-dataset audit, Weather has eta-squared 0.614 and Electricity 0.777; Traffic and AirConvection do not pass the full gate. M10 repeats the core positive datasets over three seed offsets:

| Dataset | eta^2 mean [95% CI] | Max degradation mean [95% CI] | Worst tau mean [95% CI] | Gate seeds |
| --- | --- | --- | --- | --- |
| Weather | 0.495 [0.000, 1.000] | 702.0% [341.8, 1062.2] | 0.11 [-1.00, 1.00] | 2/3 |
| Electricity | 0.572 [0.129, 1.000] | 175.1% [129.3, 220.9] | -0.22 [-1.00, 1.00] | 3/3 |

The wide Weather CI is important. It means the paper should not claim a universal stable effect across all seeds and datasets. The conservative claim is that the mechanism shift is material on two canonical TSF datasets and is strong enough to reverse selection in modern architectures.

M13 further checks the M1 aggregate claim with a bootstrap over lightweight variants and test windows. The loss-shift result is stable on Weather and Electricity: max absolute delta has 95% bootstrap intervals of 1.680 [0.514, 3.659] and 1.446 [0.862, 1.943], respectively. The same bootstrap is less decisive for rank instability (rank-instability probabilities 0.46 and 0.53), so the paper anchors the rank-reversal claim primarily in the M9/M10 official-architecture result rather than in lightweight M1 ranks alone.

### 4.2 Non-Retirement Mechanisms Are Sufficient

An easy reviewer objection is that retirement is too obvious: if a sensor disappears forever, degradation is unsurprising. M8 removes this escape hatch. Weather and Electricity still pass when considering only value-triggered, volatility-triggered, and blackout mechanisms.

| Dataset | Strongest non-retirement mechanism | Max non-ret degradation | Worst non-ret tau | Gate |
| --- | --- | --- | --- | --- |
| Weather | value_high | 132.8% | 0.333 | PASS |
| Electricity | value_high | 182.7% | -0.333 | PASS |
| Traffic | volatility | -2.9% | 0.000 | FAIL |
| AirConvection | value_high | 108.0% | 0.667 | FAIL |

This result justifies keeping the broader mechanism-shift framing. Retirement remains a strong mechanism, but it is not the only source of positive evidence.

### 4.3 Official-Architecture Adaptation Shows Rank Reversal

M9/M10 is the most important model-facing result. PatchTST and TimeXer are imported from official TSLib model files and evaluated under MaskShift's custom encoder-mask protocol. This is an official-architecture adaptation, not a full official benchmark-protocol reproduction.

| Dataset | Official architecture classes | Max degradation mean [95% CI] | Worst tau mean [95% CI] | Gate seeds |
| --- | --- | --- | --- | --- |
| Weather | PatchTST_official, TimeXer_official | 135.8% [8.6, 262.9] | -1.00 [-1.00, -1.00] | 3/3 |
| Electricity | PatchTST_official, TimeXer_official | 128.6% [98.1, 159.1] | -1.00 [-1.00, -1.00] | 3/3 |

The consistent tau of -1.0 is the cleanest evidence for the model-selection thesis: the model order under MCAR does not transfer to operational mechanisms in these settings.

M16 extends the same official PatchTST/TimeXer coverage to the two mixed datasets instead of leaving them as lightweight-model-only evidence. Traffic has 35.4% maximum degradation with worst tau 1.0 and ANOVA p=0.0854; AirConvection has 46.1% maximum degradation with worst tau 1.0 and p=0.000122. Both are mixed/negative for the rank-reversal gate. This is useful boundary evidence: the benchmark factor is real, but the paper should not claim that all datasets show official-architecture rank reversal.

### 4.4 Missing-Aware Official Baselines

M11 evaluates the official `ChannelTokenFormer_missing` model class under the same MaskShift encoder-mask protocol. The result is mixed and useful: CTF_missing is not simply broken by MaskShift, but neither does it eliminate mechanism sensitivity.

| Dataset | Backbone | Max degradation mean [95% CI] | Max abs delta [95% CI] | Strongest mechanism | Gate seeds |
| --- | --- | --- | --- | --- | --- |
| Weather | ChannelTokenFormer_missing_official | 96.0% [39.4, 152.6] | 0.264 [0.003, 0.524] | volatility | 1/3 |
| Electricity | ChannelTokenFormer_missing_official | 32.3% [-10.4, 75.0] | 0.584 [-0.130, 1.297] | value_high | 0/3 |

Together with M12, this table changes the submission posture. The paper no longer says that missing-aware official baselines are absent. Instead, it reports two official missing-aware architecture adaptations with architecture-dependent outcomes: mechanism shift is visible for CTF_missing on Weather, while S4M is comparatively robust across three reduced local seed offsets.

M12 evaluates official S4M under the same reduced local protocol over three seed offsets. This is a negative/contrastive result rather than a positive mechanism-shift result: S4M does not pass the mechanism-shift gate in any seed.

| Dataset | Backbone | Max degradation mean [95% CI] | Max abs delta [95% CI] | Strongest mechanism mode | Kruskal p mean [95% CI] | Gate seeds |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Weather | S4M_official | 5.7% [-44.9, 56.3] | 0.071 [-0.620, 0.762] | mixed | 0.509 [0.081, 0.938] | 0/3 |
| Electricity | S4M_official | 6.2% [-0.1, 12.5] | 0.312 [0.093, 0.530] | mixed | 0.933 [0.727, 1.000] | 0/3 |

This negative result improves the benchmark's credibility. MaskShift should not claim that all missing-aware architectures are mechanism-sensitive; instead, it shows that mechanism identity must be reported because sensitivity is architecture- and dataset-dependent.

M14 checks the main scale objection to M12 by doubling channels and train/test windows while keeping the same official S4M class and MaskShift protocol. The S4M conclusion does not flip: Weather has 6.8% max degradation [-8.8, 22.4], 0.109 max absolute delta [-0.134, 0.352], and Kruskal p=0.673 [0.000, 1.000] with 0/3 gate seeds; Electricity has 2.5% [-17.9, 22.8], 0.154 [-1.382, 1.690], and p=0.989 [0.963, 1.000] with 0/3 gate seeds. This remains a reduced-protocol adaptation, and not a full S4M benchmark reproduction, but it weakens the critique that the S4M contrast is only an eight-channel artifact.

### 4.5 Corrected Severity Curves

M7 originally reported a 20% missing-rate relative degradation spike above 7000%. M10 shows why this should not be the headline metric. Relative ratios can be dominated by denominator scale. The submission therefore reports absolute delta, log ratio, and symmetric relative delta.

| Missing rate | eta^2 mean [95% CI] | Max abs delta [95% CI] | Max log ratio [95% CI] | Max symmetric delta [95% CI] | Denom unstable |
| --- | --- | --- | --- | --- | --- |
| 0.10 | 0.194 [0.000, 0.456] | 0.540 [0.225, 0.855] | 0.66 [0.01, 1.30] | 0.62 [0.06, 1.17] | 0/4 |
| 0.20 | 0.286 [0.000, 0.657] | 10.713 [-22.257, 43.683] | 1.33 [-1.55, 4.22] | 0.80 [-0.45, 2.05] | 0/4 |
| 0.35 | 0.423 [0.000, 0.881] | 2.577 [-3.801, 8.954] | 1.27 [-0.40, 2.94] | 0.98 [-0.16, 2.11] | 0/4 |
| 0.50 | 0.472 [0.060, 0.884] | 0.854 [0.452, 1.256] | 0.92 [0.01, 1.83] | 0.82 [0.13, 1.51] | 0/4 |

The corrected table supports the qualitative severity trend without relying on an unstable ratio.

### 4.6 Typed Head Negative Result

The typed/topology head was tested as a minimal correction. It does not pass the paper's method gate. Weather improves by 22.9%, but Electricity improves by only 3.3%, Traffic worsens by 5.6%, and AirConvection improves by 7.5%. The overall typed improvement test has \(p=0.214\). The correct interpretation is not "MaskShift proposes a new typed head"; it is "mechanism labels and topology statistics are plausible diagnostics, but the current lightweight correction is not a robust solution."

## 5. Related Work

GRU-D established that missingness masks and time gaps can carry predictive information in recurrent models. BRITS and later diffusion approaches such as CSDI and SADI focus on imputation or joint reconstruction. SADI is especially relevant because it studies partial blackout scenarios, but its target is imputation quality under blackout-like patterns rather than model selection under matched-rate mechanism shift.

S4M and ChannelTokenFormer are closer architecture-level competitors. S4M designs missing-aware state-space forecasting, while ChannelTokenFormer targets dependency, asynchrony, and missingness in a unified framework. MaskShift is intentionally orthogonal: it does not claim a better architecture, and it would be useful even if a missing-aware architecture wins. The benchmark asks whether the evaluation protocol itself controls the right factor.

Irregular-sampling forecasting methods such as GraFITi address a different data-acquisition axis from matched-rate mechanism shift. CRIB/MTSF-M argues against naive imputation-then-prediction and motivates direct forecasting from partially observed series. Robust prediction under missingness shifts gives a statistical anchor outside TSF. The information-blackout preprint is the closest task collision because it models MNAR traffic blackouts, but MaskShift broadens the question from a single blackout model to multiple mechanisms and model-rank stability.

| Work | Venue/year | Occupied claim | MaskShift distinction |
| --- | --- | --- | --- |
| GRU-D | Scientific Reports 2018 | Mask/time-gap-aware RNN prediction | No mechanism-shift benchmark or rank-reversal audit. |
| BRITS | NeurIPS 2018 | Bidirectional imputation/prediction | Optimizes reconstruction/imputation rather than deployment mechanism shift. |
| SADI | AAAI 2025 | Diffusion imputation for partial blackouts | Blackout imputation competitor; not matched-rate multi-mechanism model-selection audit. |
| S4M | ICLR 2025 | Missing-aware S4 forecasting architecture | Architecture baseline; MaskShift is benchmark/theory and tests mask mechanisms as experimental factors. |
| ChannelTokenFormer | ICLR 2026 | Dependency/asynchrony/missingness architecture | Closest architecture collision; MaskShift avoids unified-architecture claims. |
| GraFITi | AAAI 2024 | Irregularly sampled time-series forecasting | Targets irregular sampling rather than matched-rate conditional mechanism shift. |
| CRIB/MTSF-M | arXiv 2025 | Revisits MTSF with missing values | Motivates direct forecasting; does not isolate matched-rate mechanism shift/rank reversal. |
| Information blackouts | arXiv 2026 | MNAR traffic blackout state-space model | Closest blackout collision; MaskShift broadens to multiple mechanisms and ranking stability. |
| Robust prediction under missingness shifts | arXiv 2024 | Statistical missingness-shift theory | Non-TSF anchor; MaskShift operationalizes for forecasting benchmarks. |

## 6. Claim-to-Evidence Discipline

| ID | Claim | Evidence | Limit | Required wording |
| --- | --- | --- | --- | --- |
| C1 | Matched missing rate is not a robustness certificate. | M1, M3, M7, M9 | Supported on Weather/Electricity; Traffic/AirConvection mixed. | State as evidence-backed benchmark finding, not universal theorem. |
| C2 | Mechanism shift can reverse model rankings. | M1 ranks, M9 official-architecture adaptation | M9 Weather/Electricity worst tau=-1 over three seeds. | Do not claim every dataset/model reverses. |
| C3 | The result is not only sensor retirement. | M8 non-retirement decomposition | Weather/Electricity pass without retirement. | Retirement remains a strong and obvious mechanism; keep decomposition visible. |
| C4 | Typed head is not a new method contribution. | M2/H3 | H3 fails; overall typed p=0.214. | Present as negative/diagnostic ablation. |
| C5 | Official modern architectures are affected. | M9/M10 | PatchTST/TimeXer official classes under custom MaskShift protocol. | Call it official-architecture adaptation, not full official benchmark reproduction. |
| C6 | Missing-aware architecture coverage is included but not decisive. | M11, M12, M14 | CTF_missing shows strong Weather sensitivity but weaker/non-significant Electricity sensitivity; S4M is negative/contrastive with 0/3 gate seeds under both reduced and larger-reduced protocols. | Report as architecture-dependent evidence, not a win/loss claim. |

## 7. Limitations

First, the strongest evidence is concentrated on Weather and Electricity. Traffic and AirConvection are included precisely because they weaken the universal claim. A stronger final submission should add more datasets or explain why the mechanism effect is domain-dependent.

Second, the M9 experiment imports official TSLib model classes, but it does not reproduce the full official TSLib benchmark protocol. The MaskShift protocol uses custom windows, custom masks, reduced training samples for multi-seed feasibility, and an encoder-input-only corruption design.

Third, M11 integrates official ChannelTokenFormer_missing and M12/M14 integrate official S4M, but both are adaptations under MaskShift windows, masks, reduced samples, and local compute limits. M12 uses three seed offsets and M14 doubles S4M channels and train/test windows, but neither reproduces the full S4M benchmark protocol or full-channel setting. The correct claim is coverage of two official missing-aware architecture adaptations, not exhaustive reproduction of their original benchmark protocols.

Fourth, M10/M11/M12 confidence intervals are seed-level descriptive intervals with only three seed offsets, and M13 bootstraps variants/windows rather than a full hierarchy over series, horizons, seeds, and datasets. They help avoid single-aggregate overclaiming but do not replace a full mixed-effects analysis.

Fifth, the typed-head correction fails the method gate. This is not a weakness of the benchmark thesis, but it rules out positioning MaskShift as a new model or repair method.

## 8. Conclusion

MaskShift identifies a simple but under-controlled failure mode in missing-value forecasting evaluation. Matching the missing rate does not match the conditional mask mechanism. When the mechanism shifts, forecast risk and model rankings can shift as well. The current evidence supports a benchmark/theory submission: Weather and Electricity show strong mechanism sensitivity, official PatchTST and TimeXer architecture classes show rank reversal under the MaskShift protocol, ChannelTokenFormer_missing adds mixed missing-aware evidence, S4M adds a useful negative/contrastive missing-aware baseline, and non-retirement mechanisms are sufficient to produce material degradation. The honest submission scope is narrow but useful: report missingness mechanism and topology as first-class benchmark factors, do not certify deployment missingness robustness from MCAR/block tests alone, and treat the failed typed head as a diagnostic negative result rather than a method contribution.

## Reproducibility Notes

Primary scripts:

External repositories for official-architecture runs are expected under ignored `external/` paths:

- `git clone --depth 1 https://github.com/thuml/Time-Series-Library external/TSLib`
- `git clone --depth 1 https://github.com/jinkwan1115/ChannelTokenFormer external/ChannelTokenFormer`
- `git clone --depth 1 https://github.com/WINTERWEEL/S4M.git external/S4M`

Audited revisions are TSLib `4e938a1`, ChannelTokenFormer `b1c100e`, and S4M `a718823`.

- `python3 -m experiments.MaskShift.m0_mask_suite`
- `python3 -m experiments.MaskShift.m1_mechanism_audit`
- `python3 -m experiments.MaskShift.m2_typed_head`
- `python3 -m experiments.MaskShift.m3_statistical_tests`
- `python3 -m experiments.MaskShift.m6_deep_backbone_sweep`
- `python3 -m experiments.MaskShift.m7_severity_curves`
- `python3 -m experiments.MaskShift.m8_mechanism_decomposition`
- `python3 -m experiments.MaskShift.m9_official_tslib_reproduction`
- `python3 -m experiments.MaskShift.m10_submission_hardening`
- `python3 -m experiments.MaskShift.m11_official_ctf_missing_baseline`
- `python3 -m experiments.MaskShift.m12_official_s4m_baseline`
- `python3 -m experiments.MaskShift.m13_hierarchical_bootstrap`
- `python3 -m experiments.MaskShift.m14_s4m_scale_validation`
- `python3 -m experiments.MaskShift.m16_official_tslib_full_coverage`
- `python3 -m experiments.MaskShift.m17_submission_supplement`
- `python3 -m experiments.MaskShift.m18_submission_policy_pack`
- `python3 -m experiments.MaskShift.m19_aaai27_target_readiness`
- `python3 -m experiments.MaskShift.m20_aaai27_preflight_conversion`
- `python3 -m experiments.MaskShift.m21_aaai27_reproducibility_checklist`
- `cd experiments/MaskShift/paper && tectonic --print main.tex && tectonic --print supplement.tex && tectonic --print submission_statements.tex && tectonic --print aaai27_readiness.tex && tectonic --print aaai27_preflight.tex && pdflatex -interaction=nonstopmode aaai27_official.tex && bibtex aaai27_official && pdflatex -interaction=nonstopmode aaai27_official.tex && pdflatex -interaction=nonstopmode aaai27_official.tex && cd ../../..`
- `python3 -m experiments.MaskShift.m17_submission_supplement`
- `python3 -m experiments.MaskShift.m18_submission_policy_pack`
- `python3 -m experiments.MaskShift.m19_aaai27_target_readiness`
- `python3 -m experiments.MaskShift.m20_aaai27_preflight_conversion`
- `cd experiments/MaskShift/paper && tectonic --print submission_statements.tex && cd ../../..`
- `python3 -m experiments.MaskShift.m18_submission_policy_pack`
- `cd experiments/MaskShift/paper && tectonic --print aaai27_readiness.tex && tectonic --print aaai27_preflight.tex && pdflatex -interaction=nonstopmode aaai27_official.tex && bibtex aaai27_official && pdflatex -interaction=nonstopmode aaai27_official.tex && pdflatex -interaction=nonstopmode aaai27_official.tex && cd ../../..`
- `python3 -m experiments.MaskShift.m19_aaai27_target_readiness`
- `python3 -m experiments.MaskShift.m20_aaai27_preflight_conversion`
- `python3 -m experiments.MaskShift.m21_aaai27_reproducibility_checklist`
- `python3 -m experiments.MaskShift.m15_final_integrity_audit`
- `python3 -m experiments.MaskShift.m5_main_track_audit`
- `python3 -m experiments.MaskShift.m18_submission_policy_pack`
- `cd experiments/MaskShift/paper && tectonic --print submission_statements.tex && cd ../../..`
- `python3 -m experiments.MaskShift.m18_submission_policy_pack`
- `python3 -m experiments.MaskShift.m19_aaai27_target_readiness`
- `python3 -m experiments.MaskShift.m20_aaai27_preflight_conversion`
- `python3 -m experiments.MaskShift.m21_aaai27_reproducibility_checklist`
- `cd experiments/MaskShift/paper && tectonic --print aaai27_readiness.tex && tectonic --print aaai27_preflight.tex && pdflatex -interaction=nonstopmode aaai27_official.tex && bibtex aaai27_official && pdflatex -interaction=nonstopmode aaai27_official.tex && pdflatex -interaction=nonstopmode aaai27_official.tex && cd ../../..`
- `python3 -m experiments.MaskShift.m19_aaai27_target_readiness`
- `python3 -m experiments.MaskShift.m20_aaai27_preflight_conversion`
- `python3 -m experiments.MaskShift.m21_aaai27_reproducibility_checklist`
- `python3 -m experiments.MaskShift.m15_final_integrity_audit`
- `python3 -m experiments.MaskShift.m5_main_track_audit`

`PAPER.md` is now the prose source of truth for the submission narrative. `paper/main.tex` is the synchronized generic LaTeX draft, `paper/supplement.tex`/`paper/supplement.pdf` are the reviewer-facing supplement package generated by M17, `paper/submission_statements.tex`/`paper/submission_statements.pdf` are the target-agnostic policy/disclosure package generated by M18, `paper/aaai27_readiness.tex`/`paper/aaai27_readiness.pdf` are the target-specific AAAI-27 dossier generated by M19, `paper/aaai27_preflight.tex`/`paper/aaai27_preflight.pdf` are the M20 anonymous two-column page-pressure preflight, `paper/aaai27_official.tex`/`paper/aaai27_official.pdf` are the official `aaai2027` anonymous submission-template build, and `paper/aaai27_reproducibility_checklist.tex`/`paper/aaai27_reproducibility_checklist.pdf` are the M21 filled official checklist build. The earlier `m4_paper_ready.py` packager is useful for regenerating scaffold figures and summaries, but it should not be rerun as the final manuscript generator unless its template is updated to match this draft.
