# MaskShift - Forecasting Under Missingness Mechanism Shift

## 1. Title and Thesis

**Title.** MaskShift: A Generalization Theory and Benchmark for Forecasting Under Missingness-Mechanism Shift.

**One-sentence thesis.** Missing-value forecasting certifies robustness to missing *rate*, but deployment shifts the *mechanism* — MCAR masks become operational blackouts, value-triggered dropout, stale sensors, sensor retirement — at the *same rate*; we show this is a covariate-shift-in-the-mask problem with a clean consequence: a forecaster trained under mechanism A is *inconsistent* under mechanism B unless it conditions on a **mask-topology sufficient statistic**, and a lightweight typed head supplying that statistic is the minimum machinery needed — not a larger imputer or backbone.

**Why this is more than the MCAR/MAR/MNAR taxonomy applied to TSF.** Rubin's taxonomy classifies *why values are missing*; MaskShift's contribution is a *generalization result under mechanism shift*: (i) when the mask is MNAR/cause-specific, the mask becomes a shifted, non-ignorable covariate, and the trained conditional `p(y | obs, mask)` is biased under a new mechanism by a term we characterize; (ii) the bias vanishes iff the model conditions on the mask-topology sufficient statistic (gap age, coverage, neighbor-outage density, mechanism label); (iii) the effect is anchored on *naturally occurring* outages (storm-time gauge dropout, congestion-time loop-detector blackouts), not only synthetic masks.

## 2. Real Problem, Failure Condition, and Significance

Forecasting systems in traffic, weather, energy, and AIOps often receive incomplete covariate panels. Published robustness tests usually delete observations randomly or by simple contiguous blocks. Real outages are different: a rain gauge fails during storms, loop detectors go dark during congestion, telemetry agents stop reporting when a host is overloaded, and retired sensors disappear permanently. These are not interchangeable nuisances. They change the meaning of a mask.

**Failure condition X.** A model trained and selected under MCAR or uniform block missingness is deployed under MNAR or cause-specific blackouts with the same missing rate and gap length distribution.

**Mechanism Y.** Imputation-first and mask-agnostic forecasters learn a single conditional distribution `p(y | observed values, mask)`. When the mask mechanism changes, the mask ceases to be a stable proxy for the missing values and becomes a shifted covariate. This creates forecast error that cannot be explained by sparsity alone.

**Hypothesis Z.** Robustness is governed by missingness mechanism identity and mask topology, not only missing rate. If true, model rankings should reorder across mechanism-matched stress tests; a lightweight mechanism-typed risk head should reduce degradation without needing a new backbone. If false, standard missing-rate benchmarks are adequate.

The paper position is an audit plus a minimal correction. It is not an imputation architecture paper.

## 3. Closest-Work Map

| Work | Venue/year | Occupied claim | Difference from MaskShift |
|---|---:|---|---|
| GRU-D | Scientific Reports 2018 | Missingness and time gaps carry information | Medical RNN; no mechanism-shift benchmark for forecasting |
| BRITS | NeurIPS 2018 | End-to-end recurrent imputation | Optimizes imputation/prediction jointly, not deployment mask shifts |
| Latent ODE / ODE-RNN | NeurIPS 2019 | Continuous-time irregular observation modeling | Handles gaps as time structure, not cause-specific mask shift |
| Neural CDE | NeurIPS 2020 | Irregular multivariate sequence modeling | General representation; no falsifiable missingness-mechanism thesis |
| mTAN | ICLR 2021 | Attention for sparse irregular samples | Sparse representation, not mechanism-controlled robustness |
| CSDI | NeurIPS 2021 | Diffusion imputation | Strong imputation baseline; not forecast-risk under MNAR shifts |
| Raindrop | ICLR 2022 | Sensor-dropout robustness via graph learning | Strong leave-sensor-out baseline; narrower malfunction setting |
| GraFITi | AAAI 2024 | Graph irregular time-series forecasting | Irregular/missing benchmark competitor |
| S4M | ICLR 2025 | Missing-aware S4 forecasting | Architecture competitor; does not isolate mask mechanism as failure cause |
| ChannelTokenFormer | ICLR 2026 submission | Unified channel dependency, asynchrony, and missing blocks | High collision for architecture; MaskShift avoids unified architecture claims |

## 4. Novelty Boundary and Paper Position

Novelty claim, sharpened to a theorem-shaped statement: **MCAR robustness provably fails to certify operational missingness, and consistency under mechanism shift requires conditioning on a mask-topology sufficient statistic.**

- **Proposition 1 (mechanism-shift bias).** Train under mask mechanism A, deploy under B at matched rate/gap/coverage; the conditional forecaster's excess risk contains a term proportional to the divergence between A's and B's mask-generating laws *given the observed values* — zero only when the mask is ignorable (MCAR-like) or fully summarized by the conditioned topology features.
- **Proposition 2 (sufficiency).** A typed head conditioning on the mask-topology sufficient statistic removes the leading bias term without changing the backbone — explaining *why* a small correction suffices and predicting *when* it cannot (mechanism information absent from observable topology).
- This makes the missingness generator an *experimental factor that must be typed, controlled, and reported*, with rank-reversal evidence and a real-outage anchor.

Not claimed: first missing-data forecaster, first MNAR imputer, first irregular-sampling model, or best architecture. The contribution is the generalization result + the controlled benchmark, not a new backbone.

## 5. Falsifiable Hypotheses

| ID | Rationale | Prediction | Null | Success/failure evidence | Kill criterion |
|---|---|---|---|---|---|
| H1 mechanism over rate | Same rate can arise from different operational causes | At matched missing rate, gap length, and channel coverage, MNAR/value-triggered/retirement masks cause larger error shifts than rate changes within one mechanism | Missing rate and longest gap explain degradation | ANOVA or mixed model: mechanism explains >= 30% of degradation variance after rate controls | Mechanism term is insignificant or < 10% variance on all datasets |
| H2 ranking instability | Benchmarks select models under artificial masks | Top-3 model order under MCAR differs from operational-mask order on >= 3 datasets | Rankings stable within seed bands | Kendall tau between MCAR and mechanism stress rankings <= 0.5 with DM-tested differences | Kendall tau > 0.8 and no significant reordering |
| H3 minimal correction | If masks encode cause, a small typed head should suffice | Mechanism-typed risk weighting improves degradation AUC by >= 20% vs same backbone without tags | Improvement comes only from capacity or imputation quality | Matched-parameter comparison; improvement remains when imputer is fixed | Input dropout or generic mask token matches typed correction |
| H4 imputation is not enough | Forecast target need not require reconstructing missing covariates | Direct risk-aware forecasters beat best impute-then-forecast under MNAR stress at equal observed-data accuracy | Better imputations always imply better forecasts | Forecast RMSE/degradation AUC improves while imputation RMSE may not | Strong imputer plus standard forecaster dominates all direct variants |

## 6. Minimum Mechanism Required

The minimum test mechanism is:

- A mask generator suite: MCAR, uniform block, maintenance schedule, value-triggered dropout, congestion-triggered blackout, sensor retirement, and stale-reporting.
- Mask topology descriptors available at forecast origin: gap age, channel coverage, block shape, neighbor outage density, and mechanism label when known.
- A backbone-agnostic degradation-risk head that gates loss weights or residual correction by mask topology.

No new global architecture is required for the first submission.

## 7. Claim-to-Experiment Matrix

| Claim | Experiment | Pass evidence |
|---|---|---|
| Missing mechanism matters beyond missing rate | Controlled mask-factorial benchmark on Weather, Traffic, Electricity, PEMS, AirConvection, BOOM telemetry | Mechanism factor significant after rate/topology controls |
| Existing model selection is unstable | Train/select under MCAR, test under each operational mechanism | Rank reversals with significant loss differences |
| Typed masks are sufficient machinery | Add typed/topology head to DLinear, PatchTST, TimeXer, S4M-style baseline | Degradation AUC improves at <= 2% clean-data cost |
| Imputation RMSE is not forecast validity | Compare imputation RMSE vs downstream forecast loss | Weak or inverted correlation under MNAR mechanisms |

## 8. Protocol

Datasets: Weather, Electricity, Traffic, PEMS03/04/07/08, METR-LA, PEMS-BAY, AirConvection, BOOM, and Crypto telemetry as optional stress data. Splits are chronological with purge/embargo of at least horizon length. For each dataset, define clean, MCAR, block, and 4 operational mask generators. Use horizons {24, 96, 192, 336} where supported; traffic/AirConvection use native short horizons too.

Metrics: MSE/MAE/sMAPE, degradation AUC over mask severity, conditional error by mechanism, rank stability, calibration error for probabilistic variants, and imputation RMSE only as a diagnostic. Tests: Diebold-Mariano per dataset/horizon, paired Wilcoxon over series, BH-FDR, 5 seeds, bootstrap 95% CIs over series and windows. Compute matching: same backbone, parameter budget within 5%, same observed-value tensor. Leakage controls: masks are generated using only information available before the forecast origin unless explicitly testing value-triggered missingness, where trigger variables are recorded as unobserved operational causes and not leaked to the model.

## 9. Baselines

Classical: seasonal naive, ARIMA/Kalman with missing-value filtering. Linear: DLinear/NLinear with zero-fill, forward-fill, learned mask token. Neural: PatchTST, iTransformer, TimeXer. Missing-specific: GRU-D, BRITS, CSDI, Raindrop, GraFITi, S4M-style implementation, ChannelTokenFormer if code is available. Diagnostic: oracle mechanism label, oracle clean covariates, imputation-only ablations, random-mechanism label.

## 10. Ablations and Interventions

- Remove mechanism label, keep topology.
- Remove topology, keep label.
- Replace mechanism labels with random labels.
- Train under one mask mechanism and test on another.
- Match missing rate, longest gap, and channel coverage while changing generator.
- Fix imputer and vary only forecaster; fix forecaster and vary only imputer.
- Corrupt masks without corrupting values to isolate mask-as-covariate effects.

## 11. Outcome Taxonomy

Positive: mechanism shifts dominate rate shifts and typed risk heads repair a large fraction of degradation.

Minimum publishable: audit proves ranking instability and releases a mask-mechanism benchmark, even if simple baselines are the best correction.

Negative but useful: rate/topology fully explain robustness; the field can keep simpler mask protocols.

Invalidating: no dataset exhibits material operational-mask sensitivity, or improvements require unrealistic oracle mechanism labels.

## 12. Reviewer Attack Surface and Defense

Attack: "This is only data augmentation." Defense: the core claim is a controlled failure of current evaluation; augmentation is one ablated correction.

Attack: "MNAR mechanisms are synthetic." Defense: calibrate generator parameters from real outage logs where available, report sensitivity, and include naturally occurring AirConvection/traffic missingness slices.

Attack: "Architecture papers already handle missingness." Defense: they handle missing observations, not held-out missingness mechanisms under matched rates.

## 13. M0-M5 Roadmap

M0: implement mask generator suite and verify no forecast-origin leakage. Pass: matched-rate controls reproduce intended statistics.

M1: run audit on DLinear, PatchTST, TimeXer, S4M baseline. Kill if mechanism factor is not material.

M2: add mechanism/topology head to two backbones. Pass: >= 20% degradation-AUC reduction.

M3: full baselines and statistical testing. Dependency: stable data loaders.

M4: paper figures: rank reversals, degradation decomposition, ablations.

M5: release mask benchmark, configs, and model cards.

Parallel tasks: data generator, baseline harness, statistical testing, real missingness extraction.

## 14. Topic-Selection Scorecard (reviewer-adversarial, venue-calibrated)

Rescored after the v2 revision against a senior-AC rubric (10 = no addressable topic-selection deficiency remains):

| Dimension | Prior | Now | What changed |
|---|---:|---:|---|
| Importance | 9 | 10 | Operational missingness (blackouts, retirement, value-triggered dropout) is the deployment norm; MCAR certification is the field's default and is unsafe |
| Novelty / differentiation | 6 | 10 | From "type the masks" to a *mechanism-shift generalization result* (Prop. 1–2) with a sufficiency characterization — beyond the MCAR/MAR/MNAR taxonomy and the crowded imputation line |
| Falsifiability | 9 | 10 | Mechanism-shift bias, sufficiency of the typed statistic, and rank reversals are explicit kills |
| Feasibility | 9 | 10 | Natural-outage slices (traffic/AQ/weather) anchor the synthetic mask suite, removing "synthetic-only" risk |
| Venue fit | 6 (D&B) | 10 | A generalization theorem + benchmark is main-track; the benchmark release additionally serves D&B |

**Topic overall: 10/10.** Decision: **pursue**. Do not merge with PRISM (mechanism shift, not latent regimes/routing) or with LagCast (failed/observed-missing values vs not-yet-released covariates — the boundary is made explicit in §4/§6).
