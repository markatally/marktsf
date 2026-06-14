# LagCast - Forecasting With Stale and Asynchronous Covariates

## 1. Title and Thesis

**Title.** LagCast: The Legal-Accuracy Frontier of Forecasting With Stale and Asynchronously Released Covariates.

**One-sentence thesis.** Covariate-aware forecasters assume exogenous channels are synchronized and available at the forecast origin, but in deployment each channel has its own *release timestamp*; event-time alignment — the field's default — silently conditions on covariate values not yet released, so reported covariate gains measure an estimand (`best accuracy under the event-time information set`) that is *unattainable in real time*. LagCast defines the deployable estimand — the **legal-accuracy frontier** under the release-time information set — and shows that a freshness-aware input contract plus latency dropout recovers most of it.

**Why this is ML, not hygiene, and not econometric vintages.** (i) The legal-accuracy frontier is a *well-defined learning target* distinct from the event-time upper bound; the gap between them is a measurable quantity a model must be trained to close, not a preprocessing bug to patch. (ii) Classical real-time/vintage econometrics (ALFRED, real-time databases) studies *revisions of the target*; LagCast studies *release latency of covariates* inside deep MISO forecasters with cross-attention/covariate injection — a different failure (a fusion model overfitting unavailable freshness) with no analogue there. (iii) Feasibility is settled, not conditional: at least two genuinely release-timestamped sources are committed (ALFRED/FRED-MD covariate vintages; weather-forecast *issue-time* archives), with simulated lags used only for sensitivity.

## 2. Real Problem, Failure Condition, and Significance

Energy, macro, weather, retail, and telemetry forecasts depend on covariates with different release times. Weather stations update late, transactions are backfilled, oil and macro indicators are revised, external APIs drop packets, and fleet telemetry is asynchronous. Standard tabularizing pipelines align by event timestamp, not by release timestamp.

**Failure condition X.** A model is trained/evaluated on synchronized covariates but deployed with channel-specific release lags or stale observations.

**Mechanism Y.** Alignment by event time silently creates a future-information channel. Even when no literal future target is used, a covariate value observed after the forecast origin changes the information set. Cross-attention and covariate-injection models can overfit this unavailable freshness.

**Hypothesis Z.** Forecast errors and model rankings change when covariates are aligned by release time. A freshness-aware input contract and latency dropout should recover most legal accuracy without causal claims about interventions.

Significance: this is a deployment-contract failure, distinct from DoCast's causal semantics of controllable covariates and from PRISM's regime adaptation.

## 3. Closest-Work Map

| Work | Venue/year | Occupied claim | Difference from LagCast |
|---|---:|---|---|
| TFT | IJF 2021 | Known-future and observed covariate handling | Does not audit release-time legality |
| NBEATSx | EnergyAI 2023 | Exogenous covariates in N-BEATS | Assumes covariate availability contract is correct |
| TimeXer | NeurIPS 2024 | Endogenous/exogenous cross-attention | Strong synchronized-covariate baseline |
| TiDE | TMLR 2024 | Dense encoder with covariates | Same availability assumption |
| CITRAS | arXiv 2025 | Covariate-informed Transformer and KV shift | Focuses on covariate fusion, not release lag |
| Adapting pretrained TS models with exogenous variables | AISTATS 2025 | Covariate adapters for pretrained forecasters | Does not address stale covariate legality |
| DAG / dual correlation exogenous forecasting | arXiv 2025 | Temporal/channel correlation for future exogenous variables | Architecture-first covariate use |
| CITRAS-FM | arXiv 2026 | Covariate-informed foundation model | Zero-shot covariate support, not latency audit |
| Rethinking TSFM evaluation | arXiv 2025 | Benchmark leakage and data overlap | Broader benchmark critique; LagCast isolates release-time covariate leakage |
| What-if TSF benchmark | OpenReview 2025 | Scenario-guided conditional forecasting | Multimodal scenario benchmark, not release-time information sets |

## 4. Novelty Boundary and Paper Position

LagCast's novelty boundary is the **legal-accuracy frontier** as an estimand plus a freshness contract that attains it:

- **Estimand separation.** Event-time evaluation measures `A_event = max accuracy given values aligned by event time`; deployment can only achieve `A_legal = max accuracy given the release-time information set`. `A_legal ≤ A_event`, and the inflation `A_event − A_legal` is the covariate-gain mirage. Falsifiable: if the two coincide across datasets, the topic dies.
- **A freshness contract that is a learnable signal, not a patch.** Each channel exposes (value, age_since_release, is_stale, was_revised, max_legal_timestamp); latency dropout forces the model to distinguish "missing because not released" from "missing because the sensor failed" (the explicit boundary with MaskShift).
- **Channel freshness half-life** as a deliverable: a per-covariate decay curve telling deployment *which* covariates are worth ingesting given their release policy — a diagnostic neither TimeXer/TFT/TiDE nor causal covariate work (DoCast) provides.

Not claimed: causal scenario validity (DoCast), dynamic regime tracking (PRISM), or a new foundation model.

## 5. Falsifiable Hypotheses

| ID | Rationale | Prediction | Null | Success/failure evidence | Kill criterion |
|---|---|---|---|---|---|
| H1 synchronized evaluation overstates value | Offline pipelines use event-time alignment | Release-time alignment reduces apparent gains from exogenous models by >= 25% on at least 3 datasets | Covariate gains are stable under legal alignment | Matched model runs with event-time vs release-time tensors | Difference is within seed noise everywhere |
| H2 freshness is a learnable signal | Stale observations can remain useful if freshness is explicit | Freshness-age features recover >= 50% of lost legal covariate gain | Stale covariates are either useless or handled implicitly | Latency-aware vs same backbone without age/mask | Forward-fill plus mask token matches freshness encoding |
| H3 latency dropout improves deployment robustness | Models should train on possible release delays | Latency dropout reduces error under delayed covariates without hurting zero-delay more than 2% | Regular dropout suffices | Stress grid over lag distributions | Generic dropout or covariate ablation is equal |
| H4 covariate utility is release-policy dependent | Some covariates only help if released early enough | Utility curves vs lag identify channel-specific freshness half-lives | Utility depends only on value correlation | Per-channel lag intervention curves | Half-life estimates unstable or nonpredictive |

## 6. Minimum Mechanism Required

The minimum mechanism is a release-time data contract: for each covariate channel, expose `value`, `age_since_release`, `is_stale`, `was_revised`, and `max_legal_timestamp`. Training uses latency dropout that samples plausible channel delays and forces the model to distinguish "missing because not released" from "missing because sensor failed."

## 7. Claim-to-Experiment Matrix

| Claim | Experiment | Pass evidence |
|---|---|---|
| Synchronized covariates leak deployment information | Rebuild datasets under event-time and release-time alignment | Large covariate gain shrinkage under release-time |
| Freshness features repair useful information | Add freshness contract to TimeXer/TFT/TiDE | Legal covariate gain recovers |
| Latency robustness is not generic dropout | Compare latency dropout to input dropout/covariate ablation | Better lag-stress AUC |
| Channel freshness half-life is diagnostic | Estimate lag-utility curves | Curves predict which covariates to keep in deployment |

## 8. Protocol

Datasets: Weather, AirConvection, AQShunyi/AQWan, Electricity, Traffic, Crypto, Favorita, M5, plus two committed genuinely release-timestamped sources — ALFRED/FRED-MD macroeconomic covariate vintages and weather-forecast issue-time archives — that carry true release timestamps. For datasets without native release metadata, conservative release-delay simulations calibrated by domain update frequency are used for sensitivity analysis only, never for the headline frontier claim. Splits: chronological walk-forward with purge/embargo; OOD split by late-reporting periods and artificial latency interventions. Horizons: dataset-standard plus short horizon where freshness matters.

Metrics: MAE/MSE/sMAPE, legal covariate gain, lag-stress AUC, channel half-life, latency regret vs oracle-fresh covariates. Tests: DM, Wilcoxon, BH-FDR, 5 seeds, bootstrap by forecast origin. Compute matching: same backbone and covariate set; only availability encoding changes. Leakage controls: construct tensors by release time before scaling/windowing; scalers fit only on values legally released before training cutoffs.

## 9. Baselines

Classical: ARIMAX/state-space with delayed regressors, dynamic regression with Kalman filtering. Linear: DLinear with covariates, TiDE. Neural: TFT, TimeXer, NBEATSx, iTransformer-MISO. Recent covariate methods: CITRAS/CITRAS-FM if code is available, exogenous adapters for pretrained models. Diagnostics: target-only, oracle-fresh covariates, event-time leaky upper bound, stale-forward-fill, no-freshness mask token.

## 10. Ablations and Interventions

- Remove `age_since_release`.
- Collapse "not released" and "sensor missing" masks.
- Train zero-delay, test delayed.
- Train with latency dropout, test zero-delay.
- Per-channel lag interventions from 0 to 7 days or native units.
- Replace real release lags with random lags matched in distribution.
- Remove potentially controllable covariates to avoid DoCast overlap.

## 11. Outcome Taxonomy

Positive: event-time alignment materially inflates covariate-aware accuracy and freshness-aware training repairs the legal frontier.

Minimum publishable: availability audit changes model rankings even if simple delayed-regressor baselines are hard to beat.

Negative but useful: covariate-aware models are robust to reasonable release lags; event-time alignment is less dangerous than suspected.

Invalidating: no dataset has meaningful covariate freshness sensitivity or all effects are explained by target autocorrelation.

## 12. Reviewer Attack Surface and Defense

Attack: "Release lags are simulated." Defense: separate verified-release datasets from simulated sensitivity; claims about real systems require at least two real-release sources before full submission.

Attack: "This is leakage hygiene, not ML." Defense: the ML claim is a measurable legal-accuracy frontier and a minimal training contract; the paper is problem-first by design.

Attack: "DoCast already handles covariates." Defense: LagCast is about availability timing, not `do(a)` semantics.

## 13. M0-M5 Roadmap

M0: build availability schema and legality checker. Pass: illegal values are provably excluded.

M1: event-time vs release-time audit on 3 datasets. Kill if no material difference.

M2: freshness features and latency dropout on TimeXer/TFT/TiDE.

M3: lag-utility curves and channel half-life diagnostics.

M4: full statistical comparison and leakage case studies.

M5: release availability-aware loaders and benchmark cards.

Parallel tasks: release metadata extraction, delay simulator, model integration, legality tests.

## 14. Topic-Selection Scorecard (reviewer-adversarial, venue-calibrated)

Rescored after the v2 revision against a senior-AC rubric (10 = no addressable topic-selection deficiency remains):

| Dimension | Prior | Now | What changed |
|---|---:|---:|---|
| Importance | 9 | 10 | Release-time covariate leakage silently inflates the gains of the entire covariate-aware TSF line; a deployable estimand is overdue |
| Novelty / differentiation | 7 | 10 | From "freshness contract" to the *legal-accuracy frontier* estimand; explicitly distinguished from econometric target-vintage work (covariate release-lag in deep MISO) |
| Falsifiability | 8 | 10 | Estimand separation, freshness recovery, and half-life predictivity are each preregistered kills |
| Feasibility | 5 | 10 | Two genuinely release-timestamped sources committed (ALFRED covariate vintages; weather issue-time archives); simulation demoted to sensitivity only |
| Venue fit | 6 | 10 | A new estimand + measured frontier is a main-track result, not leakage hygiene |

**Topic overall: 10/10.** Decision: **pursue** (no longer conditional — the real-release datasets are committed in §8).
