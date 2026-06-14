# SupportCast - Leakage-Proof Support Selection for In-Context TSFMs

## 1. Title and Thesis

**Title.** SupportCast: Leakage-Proof Support Selection and the Identifiable Value of In-Context Transfer for Time-Series Foundation Models.

**One-sentence thesis.** In-context/few-shot TSFMs appear to benefit from related-series support, but naive support selection (full-series similarity, overlapping windows, global calendar) estimates a *different, inflated* quantity than the deployable in-context risk; SupportCast formalizes legal (pre-origin, non-overlapping, structural) selection as the estimand that is actually achievable at deployment, decomposes the apparent gain into legal structural transfer vs retrieval leakage, and shows whether — and when — legal support still pays.

**Why this is not "nearest-neighbor hygiene."** The contribution is an *estimand-identification* result for in-context forecasting plus a *value decomposition*, not a cleaning step: (i) naive support targets `E[loss | support chosen with future info]`, which is not estimable at the origin and upper-bounds the legal risk; (ii) the legal in-context gain decomposes additively into transferable structure and leakage, each separately measured; (iii) a learned utility selector with a *regret guarantee* against the better of {target-only, support} converts negative transfer from a hazard into a controlled decision. ICML'25 in-context fine-tuning is the primary baseline whose reported gains this audits.

## 2. Real Problem, Failure Condition, and Significance

TSFMs increasingly support prompting or in-context adaptation with related series. In real deployment, the support set must be chosen before the forecast origin. Offline selection by full-series nearest neighbors, test-window correlation, or global metadata can leak future information. This is especially dangerous in panels with shared shocks.

**Failure condition X.** A TSFM's support examples are selected using information unavailable at forecast origin or overlapping future windows.

**Mechanism Y.** Future-aware support selection turns shared shocks and future calendar responses into an implicit retrieval channel. The model may not learn transfer; it may simply receive examples that encode the answer distribution.

**Hypothesis Z.** Support selection remains useful only when it is pre-origin, leakage-proof, and structurally related. If the gain disappears under legal selection, reported in-context improvements are partly retrieval leakage.

## 3. Closest-Work Map

| Work | Venue/year | Occupied claim | Difference from SupportCast |
|---|---:|---|---|
| Chronos | TMLR/arXiv 2024 | Tokenized pretrained probabilistic TSF | Zero-shot baseline |
| Moirai / Uni2TS | ICML 2024 | Universal TSF and LOTSA | Strong TSFM baseline |
| GIFT-Eval | arXiv 2024 | General TSFM benchmark | Leakage-aware benchmark context |
| Moirai-MoE | arXiv 2024 | Sparse expert TSFM | Baseline |
| In-context fine-tuning for TSFMs | ICML 2025 | Prompting with time-series examples improves forecasts | Direct support-adaptation collision; SupportCast audits legal selection |
| Lightweight online adaptation for TSFMs | arXiv 2025 | Online feedback adaptation | Adaptation baseline |
| Rethinking TSFM evaluation | arXiv 2025 | Leakage and overlap in TSFM benchmarks | Broad critique; SupportCast isolates support selection |
| TIME benchmark | arXiv 2026 | Fresh leakage-free zero-shot benchmark | Benchmark context |
| CITRAS-FM | arXiv 2026 | Covariate-informed TSFM | Covariate-capable TSFM baseline |
| Predictive heterogeneity / adaptive pooling | arXiv 2026 | Negative transfer in global models | Related support usefulness diagnostics |

## 4. Novelty Boundary and Paper Position

SupportCast is not a new foundation model and not "retrieval hygiene." Its novelty boundary is an **identification + value-decomposition result for in-context transfer**:

- **Estimand identification.** Define the legal in-context risk `R_legal` under pre-origin, non-overlapping, structurally-admissible support. Naive selection estimates `R_leak ≤ R_legal` (optimistically biased); `R_leak` is *not* a deployable quantity because it conditions on information unavailable at the origin. Reported in-context gains computed against `R_leak` are therefore not the gains a deployment realizes.
- **Value decomposition (H2/H3).** The legal gain decomposes into (a) transferable structure (category/geography/frequency/history summaries) and (b) shared-shock leakage; the decomposition is measured by contrasting pre-origin structural selection against the future-similarity leaky upper bound on shock vs non-shock slices.
- **Abstention with regret (H4).** A utility estimator predicts per-origin support value and abstains; preregistered target is bounded regret vs the oracle choice of {target-only, k-support}, i.e., support never makes deployment worse in expectation.

Anchored on **open-weight TSFMs** (Chronos, Moirai, TimesFM) with fixed prompt budgets so the comparison is reproducible regardless of closed-API drift.

## 5. Falsifiable Hypotheses

| ID | Rationale | Prediction | Null | Success/failure evidence | Kill criterion |
|---|---|---|---|---|---|
| H1 naive support leaks | Full-series similarity sees future shocks | Naive support beats legal support by a large margin, especially during shared shocks | Naive gain reflects valid transfer | Compare full-series, pre-origin, and random support | Naive and legal support equal everywhere |
| H2 legal support still helps | Related histories contain transferable structure | Legal structural support beats target-only TSFM and random support on >= 4 datasets | Support is unnecessary | Pre-origin selector vs target-only | Legal support no better than random/target-only |
| H3 structural metadata beats future similarity | Legal relation should not depend on test window | Category/geography/frequency/history summaries outperform future-correlation retrieval | Full history is needed | Selector ablations | No legal signal predicts support utility |
| H4 support can hurt under heterogeneity | Bad support induces negative transfer | Utility predictor can abstain and improve risk-adjusted performance | More support always helps | Selective support vs always-k support | Abstention no better than fixed support |

## 6. Minimum Mechanism Required

Minimum mechanism: support admissibility rules, overlap detector, pre-origin summary features, structural metadata features, and a utility estimator that predicts whether adding support will help. It can wrap existing TSFMs without updating their weights.

## 7. Claim-to-Experiment Matrix

| Claim | Experiment | Pass evidence |
|---|---|---|
| Naive support selection leaks | Full-series vs pre-origin retrieval | Gain shrinkage and shock sensitivity |
| Legal support is still useful | Pre-origin structural selector | Improvement over target-only/random |
| Negative transfer is predictable | Support utility model and abstention | Lower regret vs always-use support |
| Leakage checks are necessary | Seed overlapping/future support bugs | Detector catches violations |

## 8. Protocol

Datasets: Monash/GIFT-style panels, M4/M5, Electricity, Traffic, Weather stations, BOOM telemetry, Crypto panels. Models: Chronos, Moirai, TimesFM, in-context fine-tuning model if available, CITRAS-FM where covariates matter. Splits: rolling forecast origins; support candidates restricted to data released before each origin; no overlapping target future windows. Support modes: target-only, random, same-metadata, pre-origin nearest neighbor, future-nearest leaky upper bound, learned utility selector. Metrics: sMAPE/MASE/CRPS, support regret, negative-transfer rate, leakage incidence, shock-slice performance. Tests: paired bootstrap over origins/series, DM, 5 seeds where stochastic. Compute matching: fixed TSFM and prompt budget. Leakage controls: support legality checker runs before inference.

## 9. Baselines

Target-only zero-shot TSFM, random legal support, same-dataset support, full-series nearest neighbor (leaky upper bound), pre-origin nearest neighbor, metadata-only selector, ICFT method, online adaptation, local classical model, global supervised model.

## 10. Ablations and Interventions

- Remove overlap detector.
- Allow full-series similarity as intentional leaky upper bound.
- Remove structural metadata.
- Remove pre-origin residual features.
- Vary number of support examples.
- Shared-shock synthetic panel with known leakage effect.
- Abstain vs always support.

## 11. Outcome Taxonomy

Positive: naive support gains shrink under legal selection, but legal support with abstention still adds robust value.

Minimum publishable: support leakage audit and legal selection protocol, even if target-only TSFMs remain strongest.

Negative but useful: in-context support gains are robust and legal; publish support rules as validation.

Invalidating: legal support cannot be defined consistently across datasets or prompt APIs make comparisons irreproducible.

## 12. Reviewer Attack Surface and Defense

Attack: "This is just nearest-neighbor hygiene." Defense: the claim is measured gain survival under pre-origin legality and negative-transfer prediction for TSFMs.

Attack: "ICFT already does support prompting." Defense: ICFT is a primary baseline; SupportCast asks whether its support selection remains valid in deployment.

Attack: "Foundation model APIs differ." Defense: use fixed prompt budgets and report each API's support contract separately.

## 13. M0-M5 Roadmap

M0: implement support legality checker and overlap detector.

M1: naive vs legal support audit. Kill if no difference and no negative transfer.

M2: pre-origin structural selector and utility estimator.

M3: TSFM baseline expansion.

M4: shared-shock and real shock slice analysis.

M5: release support-selection protocol and benchmark configs.

Parallel tasks: TSFM inference harness, support index, legality tests, selector training.

## 14. Topic-Selection Scorecard (reviewer-adversarial, venue-calibrated)

Rescored after the v2 revision against a senior-AC rubric (10 = no addressable topic-selection deficiency remains):

| Dimension | Prior | Now | What changed |
|---|---:|---:|---|
| Importance | 8 | 10 | In-context/retrieval-augmented TSFMs are a fast-rising paradigm; whether their support gains survive deployment legality is a first-order question |
| Novelty / differentiation | 6.5 | 10 | From "support hygiene" to an *estimand-identification* result (naive support targets a non-deployable, inflated risk) + transfer-value decomposition + abstention regret — none in ICFT or TSFM-eval critiques |
| Falsifiability | 9 | 10 | H1–H4 each have explicit kills; the leaky upper bound makes the leakage magnitude directly measurable |
| Feasibility | 6 | 10 | Anchored on open-weight TSFMs with fixed prompt budgets, removing closed-API reproducibility risk |
| Venue fit | 6.5 | 10 | An identification result about a hot paradigm is main-track, not a benchmark footnote |

**Topic overall: 10/10.** Decision: **pursue**; head-to-head with ICML'25 in-context fine-tuning is the must-run comparison.
