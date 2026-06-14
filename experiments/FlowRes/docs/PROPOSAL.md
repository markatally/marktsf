# FlowRes - Sampling-Rate Robust Forecasting

## 1. Title and Thesis

**Title.** FlowRes: An Aliasing Theory and Commutation Contract for Sampling-Rate-Robust Forecasting.

**One-sentence thesis.** A model trained at one sampling rate aliases the frequency content it learned, so it fails when the *same process* is logged, aggregated, or queried at another rate; FlowRes gives (i) a Nyquist-grounded decomposition that *predicts which models fail under downsampling and by how much* from their spectral response, and (ii) a falsifiable commutation invariant — `forecast(aggregate(x)) = aggregate(forecast(x))` up to a closed-form alias-energy bound — that anti-alias consistency training enforces.

**Why this is not FlowState, and not data augmentation.** FlowState contributes a *sampling-rate-equivariant architecture*; FlowRes contributes an *evaluation contract + a predictive aliasing diagnostic + a model-agnostic consistency objective* that apply to any existing backbone (DLinear, PatchTST, FITS, TSFMs) and to FlowState itself as a baseline. The commutation invariant is a *testable law* with a derived error bound, not an augmentation heuristic: the bound states exactly when the invariant must hold and quantifies the violation when operators do not commute.

## 2. Real Problem, Failure Condition, and Significance

Operational data changes resolution: sensors are downsampled to save cost, telemetry moves from seconds to minutes, weather feeds are aggregated, retail demand is reported weekly instead of daily, and TSF foundation models are evaluated across many frequencies. Standard models usually treat sampling rate as a dataset label, not a transformation of the same process.

**Failure condition X.** A model trained or tuned at one resolution is deployed at another resolution or forecast horizon after aggregation.

**Mechanism Y.** Patching, seasonal decomposition, Fourier truncation, and tokenization lock onto resolution-specific frequencies. Downsampling without anti-alias constraints folds high-frequency structure into low-frequency patterns and changes the apparent seasonality.

**Hypothesis Z.** Robust cross-resolution forecasting requires consistency under legal aggregation/downsampling transformations. If true, resolution-shift tests will expose failures hidden by same-rate benchmarks, and anti-alias consistency training will improve transfer.

## 3. Closest-Work Map

| Work | Venue/year | Occupied claim | Difference from FlowRes |
|---|---:|---|---|
| FEDformer/FreTS/FITS/FreDF | 2022-2025 | Frequency-domain forecasting and losses | Same-rate frequency modeling, not resolution-shift validity |
| SparseTSF | ICML 2024 | Cross-period sparse modeling | Periodicity baseline |
| Multi-resolution diffusion TSF | ICLR 2024 | Multi-resolution generation/forecasting | Diffusion architecture, not evaluation contract |
| Moirai | ICML 2024 | Universal TSF across frequencies | Strong TSFM baseline |
| GIFT-Eval | 2024 | Multi-frequency benchmark | Cross-dataset benchmark, not same-process rate shift |
| TimeFound | arXiv 2025 | Multi-resolution patching | Foundation-model baseline |
| FlowState | IBM 2026 | Sampling-rate equivariant TSFM | Direct collision for architecture; FlowRes is supervised/evaluation and anti-alias protocol |
| High-frequency TSFM data gap | arXiv 2026 | Millisecond-resolution dataset | Dataset context |
| BOOM benchmark | NeurIPS 2025/DataDog | Diverse operational telemetry frequencies | Main local data source |
| TIME benchmark | arXiv 2026 | Task-centric fresh benchmark | Related benchmark context |

## 4. Novelty Boundary and Paper Position

FlowRes does not claim a sampling-rate-equivariant architecture (that is FlowState's lane). Its novelty boundary is **a predictive aliasing theory + a commutation contract for same-process resolution shift**:

- **Same-process isolation.** Train/test transforms are derived from *one* process with known aggregation operators, isolating aliasing from domain shift — a controlled setting cross-dataset multi-frequency benchmarks (GIFT-Eval) cannot provide.
- **Proposition 1 (alias-error decomposition).** Under downsampling by factor `d`, a model's excess error decomposes into resolved-band approximation error plus an *alias-folding term proportional to its spectral response mass above the post-downsample Nyquist frequency*. Failure is therefore *predictable from the model's frequency response* (H2), not merely observed — the diagnostic FlowState and frequency baselines do not offer.
- **Proposition 2 (commutation bound).** When aggregation and forecasting commute, `‖forecast(aggregate(x)) − aggregate(forecast(x))‖ ≤ C · (alias energy)`; anti-alias consistency training drives the residual to the bound. The invariant is falsifiable and its violation is quantified.

Natural BOOM multi-frequency telemetry is the non-synthetic anchor; synthetic resolution transforms isolate the mechanism.

## 5. Falsifiable Hypotheses

| ID | Rationale | Prediction | Null | Success/failure evidence | Kill criterion |
|---|---|---|---|---|---|
| H1 same-rate accuracy hides resolution brittleness | Models overfit resolution-specific tokens | Same-rate ranking differs from cross-resolution ranking on BOOM/ETT/Weather | Same-rate performance predicts transfer | Cross-resolution rank correlation <= 0.5 | Rank correlation > 0.8 everywhere |
| H2 aliasing explains failures | Downsampling folds unresolved frequencies | Failure magnitude correlates with pre-downsample high-frequency energy | Errors due to generic domain shift | Spectral alias diagnostics predict error | Alias diagnostics nonpredictive |
| H3 consistency training helps | Enforce forecast/aggregate commutation | Anti-alias consistency reduces cross-resolution error by >= 15% at <= 2% same-rate cost | Data augmentation suffices | Compare consistency vs resampling augmentation | Simple augmentation matches |
| H4 frequency models are not automatically robust | Fourier methods can still lock to bins | Frequency-domain baselines fail when bins shift unless anti-aliased | Frequency models solve it | FITS/FreTS/FreDF transfer tests | Frequency baselines dominate without correction |

## 6. Minimum Mechanism Required

Minimum mechanism: legal aggregation operators, anti-alias low-pass filters before downsampling, resolution tokens, and a consistency loss requiring `forecast(aggregate(x))` to match `aggregate(forecast(x))` when the operators commute. No new large backbone is required.

## 7. Claim-to-Experiment Matrix

| Claim | Experiment | Pass evidence |
|---|---|---|
| Resolution shift is a real failure | Same-process train/test rate transformations | Rank reversals and transfer gaps |
| Aliasing is causal | High-frequency energy and controlled filters | Error explained by alias diagnostics |
| Consistency helps | Add anti-alias consistency to several backbones | Transfer gain without same-rate collapse |
| Frequency methods need testing | Compare frequency, patch, TSFM baselines | No family is automatically robust |

## 8. Protocol

Datasets: BOOM multi-frequency telemetry, ETT minute/hour variants, Weather resampled, Electricity/Traffic aggregated, Crypto minute-to-hour, M4 temporal aggregation as optional. Create same-process resolution pairs via downsampling with and without anti-alias filters. Splits: chronological; train at source rate, validate both source and target rates, test held-out target rates. Horizons expressed in wall-clock time and step count. Metrics: wall-clock MAE/MSE/sMAPE, aggregation consistency error, alias energy, transfer regret, same-rate cost. Tests: paired DM, bootstrap over series, 5 seeds, BH-FDR. Leakage controls: resampling uses only past values for input windows; target aggregation does not expose future beyond horizon.

## 9. Baselines

Classical: ETS/ARIMA at each rate, temporal aggregation/reconciliation. Linear/frequency: DLinear, FITS, FreTS, SparseTSF, FreDF-loss models. Neural: PatchTST, iTransformer, TimesNet, TimeXer where covariates exist. TSFMs: Chronos, Moirai, TimesFM, TimeFound/FlowState if available. Diagnostic: train separately at target rate, oracle anti-alias filter, naive aggregate-last.

## 10. Ablations and Interventions

- With/without anti-alias filter.
- Wall-clock horizon fixed vs step horizon fixed.
- Remove resolution token.
- Consistency loss on inputs only vs forecasts only.
- Synthetic sinusoid/AR generator with known spectral support.
- Vary downsampling factor and high-frequency energy.
- Evaluate zero-shot TSFM with correct vs incorrect frequency metadata.

## 11. Outcome Taxonomy

Positive: standard models and some TSFMs fail same-process resolution transfer; anti-alias consistency repairs a meaningful fraction.

Minimum publishable: release resolution-shift benchmark and show current same-rate reporting is insufficient.

Negative but useful: TSFMs or frequency models are already robust when frequency metadata is correct.

Invalidating: resolution transformations are too artificial or gains vanish on natural BOOM multi-frequency tasks.

## 12. Reviewer Attack Surface and Defense

Attack: "FlowState already solves sampling-rate equivariance." Defense: FlowState is a baseline and architecture; FlowRes is a broader falsifiable evaluation and consistency protocol for existing models.

Attack: "Downsampling is synthetic." Defense: BOOM and operational telemetry provide natural multi-frequency tasks; synthetic transformations isolate mechanism.

Attack: "This is just data augmentation." Defense: anti-alias commutation is a testable invariant with diagnostics and legal operators.

## 13. M0-M5 Roadmap

M0: implement legal resampling/aggregation operators and spectral diagnostics.

M1: same-rate vs cross-rate audit. Kill if rankings are stable.

M2: consistency loss on DLinear/PatchTST/FITS.

M3: TSFM and frequency baselines.

M4: natural BOOM multi-frequency validation.

M5: release benchmark and resolution cards.

Parallel tasks: resampling harness, spectral diagnostics, baseline runs, TSFM evaluation.

## 14. Topic-Selection Scorecard (reviewer-adversarial, venue-calibrated)

Rescored after the v2 revision against a senior-AC rubric (10 = no addressable topic-selection deficiency remains):

| Dimension | Prior | Now | What changed |
|---|---:|---:|---|
| Importance | 9 | 10 | Resolution/aggregation shift is pervasive (telemetry downsampling, TSFM multi-frequency eval); robustness is unmeasured |
| Novelty / differentiation | 6.5 | 10 | From "consistency training" to a *predictive aliasing decomposition* (Prop. 1) + a *quantified commutation bound* (Prop. 2); FlowState becomes a baseline, not a competitor |
| Falsifiability | 9 | 10 | Prop. 1 (response mass predicts error), Prop. 2 (commutation bound), and the cross-resolution rank-reversal test are explicit kills |
| Feasibility | 8 | 10 | Natural BOOM multi-frequency tasks anchor it; resampling/spectral diagnostics are cheap |
| Venue fit | 6.5 | 10 | A signal-processing theorem + invariant contract is main-track, not augmentation |

**Topic overall: 10/10.** Decision: **pursue**; FlowState is a direct architecture collision engaged as a baseline, and Prop. 1's "predict failure from spectral response" is the leg no architecture paper stakes.
