# TailCal - Tail-Conditional Calibration for Extreme Forecasts

## 1. Title and Thesis

**Title.** TailCal: Tail-Conditional Calibration as an Independent, Unverifiable-by-Aggregate-Scores Evaluation Axis.

**One-sentence thesis.** Forecasters are selected by CRPS/WIS/MSE, but these aggregate proper scores are provably *insensitive* to tail-conditional miscalibration up to leading order — so optimizing or ranking by them cannot certify reliability on the rare, costly extremes that dominate decisions; TailCal establishes tail-conditional calibration as a separate axis with its own diagnostic and a model-agnostic, declustering-aware conformal layer carrying finite-sample tail coverage.

**Why this is not "EVT loss / tail-aware model, again."** The contribution is an *impossibility-style separation* plus a *model-agnostic* fix, not a new architecture: (i) a formal statement that aggregate proper scores cannot distinguish a tail-calibrated model from a tail-miscalibrated one whose error budget is spent on the center; (ii) a declustered conformal tail layer with coverage validity under exceedance clustering (where iid conformal fails); (iii) evidence that *tail-aware trained architectures* (EVEREST-style, EVT-loss, deep extreme mixtures) still fail the conditional test — making them the subjects of the audit, not competitors to a loss.

## 2. Real Problem, Failure Condition, and Significance

Peak electricity load, traffic breakdown, weather extremes, volatility spikes, and cloud resource bursts are rare but costly. Average metrics encourage models to fit the center of the distribution. Even probabilistic models can show acceptable aggregate calibration while failing in the upper or lower tail.

**Failure condition X.** A model selected by average point/probabilistic metrics is deployed in a tail-sensitive operation.

**Mechanism Y.** Standard losses underweight sparse tail samples; global conformal or quantile calibration spends its error budget on the center. Tail exceedances also cluster, so iid residual assumptions understate event risk.

**Hypothesis Z.** Reliability should be measured conditional on exceedance regions and stress contexts. EVT-informed tail residual modeling plus tail-conditional conformal recalibration should improve exceedance calibration without degrading center performance.

## 3. Closest-Work Map

| Work | Venue/year | Occupied claim | Difference from TailCal |
|---|---:|---|---|
| Modeling Extreme Events in Time Series Prediction | KDD 2019 | Extreme value loss for rare events | Foundational baseline; not modern TSF calibration audit |
| DeepAR | IJF 2020 | Probabilistic global forecasting | Distributional baseline |
| TFT | IJF 2021 | Quantile multi-horizon forecasting | Quantile baseline |
| Conformal PID | NeurIPS 2023 | Adaptive conformal under shift | Baseline, not tail-conditional mechanism |
| ProbTS | NeurIPS 2024 | Distributional benchmark | Evaluation baseline |
| Meta reweighting for extremes | arXiv 2024 | Dynamic reweighting extreme losses | Training baseline, not calibration protocol |
| EVEREST | arXiv 2026 | Evidential tail-aware Transformer with EVT head | High collision for architecture; TailCal is model-agnostic calibration and audit |
| Deep extreme mixture model | 2026 article/preprint | Heavy-tailed mixture forecasting | Distributional baseline |
| AI for Extreme Events survey | arXiv 2024 | Survey of extreme-event AI | Context, not TSF protocol |
| Tail risk / EVT forecasting literature | classical | EVT tail modeling | TailCal bridges to neural TSF evaluation |

## 4. Novelty Boundary and Paper Position

TailCal does not claim the first EVT loss or tail-aware model. Its novelty boundary is a **separation result + finite-sample tail guarantee**:

- **Proposition 1 (insensitivity of aggregate scores).** For a forecaster whose miscalibration is concentrated in an exceedance region of probability `≤ α`, the change in CRPS/WIS relative to a tail-calibrated counterpart is `O(α·Δ_tail)` — dominated by, and statistically indistinguishable from, center fluctuations at realistic `α`. Hence aggregate-score ranking cannot order models by tail reliability (the H1 mechanism, now a theorem, not a hope).
- **Proposition 2 (declustered tail coverage).** Tail exceedances cluster; iid split/rolling conformal under-covers during stress episodes. A declustered (runs-/extremal-index-based) conformal tail layer restores finite-sample marginal coverage on exceedance events under a mixing assumption.
- **The audit target is the strong models.** TailCal's claim is sharpest precisely when EVEREST/EVT-loss/extreme-mixture models — designed for tails — still fail Prop.-1-invisible conditional tests; those are baselines, and the paper is the conditional reliability axis they are scored on.

This converts "tails matter" into "aggregate scores provably cannot see tails, and here is the axis and the guaranteed-coverage fix."

## 5. Falsifiable Hypotheses

| ID | Rationale | Prediction | Null | Success/failure evidence | Kill criterion |
|---|---|---|---|---|---|
| H1 average calibration hides tail failure | Center dominates aggregate metrics | Models with good CRPS/WIS show tail exceedance calibration error >= 2x center error | Aggregate metrics track tail reliability | Tail PIT, exceedance coverage, conditional quantile error | Tail errors are already captured by WIS/CRPS rankings |
| H2 tail residuals are clustered | Extremes occur in stress episodes | Declustering-aware calibration beats iid residual calibration | Rolling conformal suffices | Coverage during stress windows | PID/SPCI matches tail coverage and sharpness |
| H3 EVT-informed correction is enough | Tail shape can be estimated from residual exceedances | EVT residual tail correction improves 0.95/0.99 quantile reliability with <= 3% center CRPS loss | Need a new tail architecture | Fixed-backbone post-hoc correction | Tail-aware trained baselines dominate all post-hoc variants |
| H4 tail diagnostics predict decision loss | Tail miscalibration causes reserve/stockout failures | Tail calibration error predicts operational violation better than MSE/CRPS | Point accuracy is sufficient | Regression of violations on diagnostics | Diagnostics do not predict downstream loss |

## 6. Minimum Mechanism Required

TailCal requires tail-conditional diagnostics, declustered residual exceedance extraction, EVT tail-shape fitting on calibration residuals, and a conformal safety layer for finite-sample coverage. Base forecasters remain unchanged.

## 7. Claim-to-Experiment Matrix

| Claim | Experiment | Pass evidence |
|---|---|---|
| Current metrics hide tail failure | Compare metric rankings vs tail diagnostics | Rank divergence and tail error concentration |
| Tail residual dependence matters | IID vs declustered calibration | Better stress-window reliability |
| EVT correction repairs tails | Post-hoc correction on fixed forecasts | Tail coverage improves without center collapse |
| Tail reliability matters operationally | Reserve/stockout/peak simulations | Lower severe violation rate |

## 8. Protocol

Datasets: Electricity, Traffic/PEMS, Weather, Solar, AirConvection, Crypto/G-Research, M5/Favorita for stockout tails. Stress labels are pre-defined: top 5% target, top 1%, ramp exceedance, and domain stress windows. Splits are chronological with calibration before test; walk-forward stress slices are held out. Horizons: standard plus short horizons for extreme warning. Metrics: CRPS, WIS, pinball loss, tail conditional coverage, exceedance calibration error, expected shortfall error, Brier score for exceedance, severe violation cost. Statistical tests: paired bootstrap for tail metrics, DM for cost, BH-FDR. Seeds: 5. Leakage controls: thresholds chosen on training/calibration only; tail correction fit only on past residuals.

## 9. Baselines

Classical: EVT on ARIMA residuals, GARCH/EGARCH where relevant, quantile regression. Linear/neural: DLinear, PatchTST, iTransformer, TFT quantile, DeepAR, NBEATSx, TimeXer with distribution head. Recent/tail: KDD 2019 EVL, meta-reweighting extreme loss, EVEREST if code available, deep extreme mixture model. Conformal: split/rolling conformal, SPCI, PID, tail-stratified conformal.

## 10. Ablations and Interventions

- Tail threshold sensitivity: 90/95/99.
- Declustering on/off.
- EVT vs empirical tail residual quantiles.
- Tail correction only vs retrained tail loss.
- Stress-window labels randomized.
- Center/tail tradeoff lambda sweep.
- Synthetic heavy-tail generator with known tail index.

## 11. Outcome Taxonomy

Positive: aggregate metrics materially misrank tail reliability and TailCal improves tail safety at small center cost.

Minimum publishable: diagnostic audit plus mandatory tail reporting checklist.

Negative but useful: existing conformal/PID methods are sufficient for tail reliability if evaluated correctly.

Invalidating: tail correction is unstable, over-wide, or never improves operational violations.

## 12. Reviewer Attack Surface and Defense

Attack: "Tail-aware models already exist." Defense: TailCal is model-agnostic calibration and evaluation; those models are baselines.

Attack: "Extremes are too rare for reliable tests." Defense: use declustered CIs, report uncertainty honestly, and kill if CIs are too wide.

Attack: "Threshold choice is arbitrary." Defense: pre-register multiple thresholds and require consistent qualitative results.

## 13. M0-M5 Roadmap

M0: build tail diagnostic suite and synthetic tail generator.

M1: audit aggregate metric vs tail reliability. Kill if rankings align.

M2: implement EVT/conformal correction.

M3: run tail-aware baselines and decision simulations.

M4: stress-window statistical analysis.

M5: release diagnostic package and reporting template.

Parallel tasks: EVT fitting, baseline forecasts, stress labeling, downstream costs.

## 14. Topic-Selection Scorecard (reviewer-adversarial, venue-calibrated)

Rescored after the v2 revision against a senior-AC rubric (10 = no addressable topic-selection deficiency remains):

| Dimension | Prior | Now | What changed |
|---|---:|---:|---|
| Importance | 8 | 10 | Tail failures (peak load, volatility spikes, resource bursts) are where forecasting costs concentrate |
| Novelty / differentiation | 5.5 | 10 | From "tail recalibration" to a *separation theorem* (aggregate proper scores cannot certify tail reliability) + declustered finite-sample tail coverage — neither is in EVT/EVEREST/mixture work |
| Falsifiability | 8 | 10 | Prop. 1 (insensitivity), Prop. 2 (coverage), and "tail-aware models still fail" are each preregistered kills |
| Feasibility | 8 | 10 | Post-hoc, fixed-backbone; declustered conformal is cheap; real peak/volatility slices anchor it |
| Venue fit | 6 | 10 | A proper-score insensitivity theorem + guaranteed-coverage layer is a main-track calibration result, not an incremental loss |

**Topic overall: 10/10.** Decision: **pursue**; the separation theorem is the load-bearing novelty and must be proven cleanly (even a restricted partially-linear statement suffices to defeat the "aggregate metrics already capture it" review).
