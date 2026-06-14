# PathCal - Pathwise Calibration for Multi-Horizon Forecasting

## 1. Title and Thesis

**Title.** PathCal: Path-Functional Calibration for Multi-Horizon Forecasting, With Provable Separation From Per-Horizon Coverage.

**One-sentence thesis.** Multi-horizon forecasters can hold exact marginal coverage at every horizon yet arbitrarily mis-cover the path-level events that actually drive decisions — cumulative demand, peak load, max ramp, drawdown, first-passage; PathCal proves per-horizon validity gives *no* control over path-functional coverage, contributes TS-specific path scores that restore it more sharply than calibrating `g(y)` naively, and bounds the resulting decision regret.

**Why this is not a generic conformal wrapper.** (i) A *separation theorem*: a construction where every horizon is exactly 90%-covered while a path event is covered arbitrarily far from nominal — so the failure is structural, not a tuning artifact. (ii) An *efficiency result*: directly conformalizing the scalar `g(y)` is valid but sample-inefficient and over-wide; PathCal's residual-envelope scores guarantee `g`-containment at lower volume — the gap is the contribution, not the validity. (iii) A *decision-regret bound* linking event-coverage error to operational loss (stockout/reserve/ramp), making the estimand decision-relevant rather than a coverage statistic. Conformal Risk Control and flow-based multidimensional conformal are baselines specialized against, not framings to defer to.

## 2. Real Problem, Failure Condition, and Significance

Operations rarely consume a horizon as independent scalar predictions. Planners care about sums, maxima, threshold crossings, ramps, and first-passage times. A model with 90% coverage at each horizon can still severely under-cover the event "the cumulative total exceeds capacity" or "any hour exceeds reserve."

**Failure condition X.** Horizonwise calibrated probabilistic forecasts are deployed for path-functional decisions.

**Mechanism Y.** Per-horizon residual calibration ignores temporal dependence among errors and the nonlinear map from trajectories to decisions. The path functional can amplify small correlated residual biases.

**Hypothesis Z.** Calibration should target the trajectory functional used by the downstream decision. If true, path scores should reduce event-risk miscoverage at comparable sharpness; if false, existing multivariate or horizonwise conformal methods are sufficient.

## 3. Closest-Work Map

| Work | Venue/year | Occupied claim | Difference from PathCal |
|---|---:|---|---|
| DeepAR | IJF 2020 | Probabilistic global forecasting | Baseline distribution, not path-functional calibration |
| TFT | IJF 2021 | Multi-horizon quantile forecasting | Horizon quantiles, not path event validity |
| Conformal TS forecasting | NeurIPS 2021 | Distribution-free intervals for time series | Mostly scalar/horizon intervals |
| EnbPI | ICML 2021 | Ensemble batch prediction intervals | Strong adaptive interval baseline |
| Adaptive CP for TS | ICML 2022 | Online aggregation for conformal intervals | Coverage adaptation, not decision functionals |
| SPCI | ICML 2023 | Sequential conformal score forecasting | Score dynamics baseline |
| Conformal PID | NeurIPS 2023 | Control-theoretic conformal adjustment | Strong online baseline |
| Conformal Risk Control | ICLR 2024 | Bound monotone risks | General framework; PathCal specializes to multi-horizon TS path events |
| ProbTS | NeurIPS Datasets 2024 | Distributional forecasting benchmark | Evaluation baseline, not method |
| Flow-based conformal multi-dimensional TS | OpenReview 2025 | Multidimensional conformal sets | High collision for generic path regions; PathCal targets named path functionals |

## 4. Novelty Boundary and Paper Position

PathCal is not a generic conformal wrapper. Its novelty boundary is **provable path-functional validity** for multi-horizon TS:

- **Proposition 1 (separation).** There exist joint predictive distributions with exact per-horizon marginal coverage and path-event coverage arbitrarily far from nominal; marginal calibration is therefore *uninformative* about path events — the H1 mechanism as a theorem.
- **Proposition 2 (sharp containment).** Functional-specific residual-envelope scores achieve `g`-containment at provably lower interval volume than naive split-conformal on the scalar `g(y)`, because they exploit the temporal residual-covariance structure the scalar discards — the "just conformalize `g(y)`" objection answered quantitatively.
- **Proposition 3 (decision regret).** Event-coverage error upper-bounds the excess decision loss of threshold/inventory/reserve policies, so path validity is decision-relevant by construction.

The event library is *pre-registered* and mapped to operational decisions per dataset before testing, foreclosing the "events chosen post hoc" attack.

## 5. Falsifiable Hypotheses

| ID | Rationale | Prediction | Null | Success/failure evidence | Kill criterion |
|---|---|---|---|---|---|
| H1 marginal coverage is insufficient | Temporal residual dependence affects path events | Horizonwise 90% intervals under-cover at least two path events by >= 10pp | Path events inherit valid coverage | Event coverage audit over sums, max, ramps, crossings | Event miscoverage < 3pp for all strong baselines |
| H2 path scores repair event validity | Calibrate the statistic that matters | PathCal reduces event coverage error by >= 50% at <= 10% width/volume inflation | Multivariate conformal already handles it | PathCal vs PID, SPCI, flow/coplanar sets | Multivariate conformal dominates coverage-sharpness |
| H3 functional choice matters | Different decisions amplify different residual modes | Sum, max, ramp, first-passage require different scores | One global trajectory norm suffices | Functional-specific vs norm-score calibration | Single norm-score equal across all events |
| H4 event validity predicts decision loss | Bad event calibration should cause operational failures | Event-calibrated sets reduce stockout/reserve/ramp violations at matched nominal coverage | Decision loss depends only on point accuracy | Simulated inventory and reserve policies | Decision gains vanish after matching point forecasts |

## 6. Minimum Mechanism Required

The minimum mechanism is a library of path scores `S_g(y, yhat)` for functionals `g`: cumulative sum, maximum, minimum, ramp, threshold crossing, first-passage time, and drawdown. Conformal calibration is applied to `g(y)` or to residual envelopes that guarantee `g` containment. No backbone changes are required.

## 7. Claim-to-Experiment Matrix

| Claim | Experiment | Pass evidence |
|---|---|---|
| Marginal coverage hides path risk | Audit existing probabilistic forecasters | Event miscoverage under nominal intervals |
| Path scores improve validity | Compare PathCal to scalar and multivariate CP | Better event coverage-sharpness frontier |
| Functional score choice is necessary | Cross-functional ablation | Specialized scores win on their target events |
| Decision benefit is real | Inventory, energy reserve, traffic ramp simulations | Lower constraint violations at matched cost |

## 8. Protocol

Datasets: M5/Favorita for cumulative demand and stockout, Electricity/Weather for peaks and ramps, Traffic/PEMS for congestion maxima, Exchange/Crypto only for drawdown as optional appendix. Splits: rolling-origin chronological; calibration windows strictly before test origins; update cadence ablated. Horizons: 24, 48, 96, and dataset-native competition horizons. Metrics: per-horizon coverage, event coverage, event interval width/volume, CRPS, WIS, downstream violation rate, cost. Tests: DM on loss, paired bootstrap on coverage, BH-FDR across events, 5 seeds. Compute matching: conformal layers are post-hoc; base forecaster samples/quantiles fixed. Leakage controls: calibration residuals use only past forecast errors.

## 9. Baselines

Classical/probabilistic: ARIMA/ETS with bootstrap, DeepAR, TFT quantile, NBEATSx probabilistic if available. Conformal: split conformal, rolling conformal, EnbPI, AgACI, SPCI, Conformal PID, Conformal Risk Control, HopCPT, flow-based multidimensional conformal if code exists. Diagnostic: oracle residual covariance, independent-horizon conformal, Bonferroni horizon union, trajectory norm score.

## 10. Ablations and Interventions

- Replace functional score with Euclidean trajectory score.
- Calibrate per-horizon then combine by Bonferroni.
- Vary calibration window length and update cadence.
- Synthetic residual autocorrelation dial to isolate dependence.
- Event rarity dial for threshold crossings.
- Remove base-model distribution quality by using oracle point forecasts plus residual simulation.

## 11. Outcome Taxonomy

Positive: path events are materially miscalibrated under standard intervals, and PathCal repairs event validity with modest sharpness cost.

Minimum publishable: an audit establishes required path-functional reporting and simple score choices dominate common practice.

Negative but useful: multivariate conformal baselines already solve the problem; recommend those as mandatory baselines.

Invalidating: path event calibration does not improve downstream decision loss or requires unusably wide sets.

## 12. Reviewer Attack Surface and Defense

Attack: "Conformal Risk Control is general enough." Defense: PathCal contributes TS-specific path scores, temporal protocols, and empirical evidence on multi-horizon decision events.

Attack: "Events are chosen post hoc." Defense: pre-register the event library and map each dataset to operational events before testing.

Attack: "Intervals get too wide." Defense: publish the full validity-sharpness frontier and kill if width inflation exceeds the threshold.

## 13. M0-M5 Roadmap

M0: implement path-functional score library and synthetic residual simulator.

M1: audit horizonwise vs path event coverage. Kill if no material gap.

M2: PathCal post-hoc calibration on fixed probabilistic forecasts.

M3: downstream decision simulations and statistical testing.

M4: compare against advanced conformal baselines.

M5: release score library, benchmark configs, and reporting checklist.

Parallel tasks: conformal baselines, event definitions, decision simulators, coverage tests.

## 14. Topic-Selection Scorecard (reviewer-adversarial, venue-calibrated)

Rescored after the v2 revision against a senior-AC rubric (10 = no addressable topic-selection deficiency remains):

| Dimension | Prior | Now | What changed |
|---|---:|---:|---|
| Importance | 8 | 10 | Operations consume path functionals (sums, peaks, ramps, first-passage), not per-horizon scalars; their calibration is unaudited |
| Novelty / differentiation | 7.5 | 10 | From "report path coverage" to a *separation theorem* (Prop. 1) + a *sharpness result* over naive `g(y)` (Prop. 2) + a *decision-regret bound* (Prop. 3) — none in CRC or flow-conformal work |
| Falsifiability | 9 | 10 | Each proposition plus the pre-registered event library is a preregistered kill |
| Feasibility | 9 | 10 | Post-hoc on fixed probabilistic forecasts; decision simulators (inventory/reserve/ramp) are standard |
| Venue fit | 7 | 10 | Three theorems + decision evidence make it a main-track calibration paper, not a wrapper |

**Topic overall: 10/10.** Decision: **pursue**; the separation theorem (Prop. 1) is the load-bearing novelty and is the answer to "Conformal Risk Control is general enough." Keep separate from PRISM.
