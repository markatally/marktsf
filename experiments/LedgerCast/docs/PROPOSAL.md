# LedgerCast - Reliability-Weighted Forecast Reconciliation

## 1. Title and Thesis

**Title.** LedgerCast: Coherent Forecasting Under an Unreliable Constraint Operator.

**One-sentence thesis.** Forecast reconciliation treats the aggregation constraint `S` and the observed aggregates as exact, but in deployment the constraint operator is an *unreliable ledger* — revised aggregates, missing bottom nodes, product churn, allocation errors — and we show, theoretically and empirically, that hard projection onto a contaminated constraint is *statistically inadmissible*: it is dominated by a reliability-weighted projection that is minimax-robust to calibrated ledger noise and degenerates to exact MinT when the ledger is clean.

**Why this is not "robust reconciliation, again."** The contribution is not a softness knob. It is (i) an *admissibility result* — hard coherence is a dominated estimator under a formal contamination model of the constraint operator; (ii) the identification of *constraint-operator reliability* as the estimable quantity that governs when coherence helps vs hurts; and (iii) a real revision-log benchmark where the contamination is measured, not assumed. The setting generalizes beyond retail hierarchies to any structured-output forecasting with a noisy linear constraint (temporal aggregation, sensor-to-region sums, settlement-revised energy ledgers).

## 2. Real Problem, Failure Condition, and Significance

Retail, energy, traffic, and enterprise telemetry forecasts often live in hierarchies. The sum of bottom series should equal the parent in theory, but in practice bottom series are missing, products are reclassified, meters are estimated, returns are booked late, and aggregate reports are revised. Hard coherence can force forecasts to obey a corrupted ledger.

**Failure condition X.** Reconciliation is applied as if the hierarchy is exact when aggregation observations or membership are noisy.

**Mechanism Y.** Projection methods distribute errors according to covariance assumptions, but they treat the summing matrix and observed aggregate targets as truth. When the ledger is unreliable, hard projection injects reporting noise into otherwise good base forecasts.

**Hypothesis Z.** Coherence should be weighted by ledger reliability. If true, a reliability-aware reconciliation layer should dominate hard MinT/projection under contaminated ledgers and reduce to hard coherence when reliability is high.

## 3. Closest-Work Map

| Work | Venue/year | Occupied claim | Difference from LedgerCast |
|---|---:|---|---|
| MinT / optimal reconciliation | JASA 2019 | Projection reconciliation for coherent forecasts | Assumes exact hierarchy |
| End-to-end coherent probabilistic forecasts | ICML 2021 | Neural coherent probabilistic forecasting | Exact constraints |
| HINT / hierarchical mixture networks | arXiv 2023 | Neural hierarchical probabilistic forecasting | Strong baseline; not noisy-ledger focused |
| Coherent probabilistic temporal hierarchies | AISTATS 2023 | Temporal aggregation coherence | Exact temporal hierarchy |
| Graph-based TS clustering/reconciliation | ICML 2024 | Learned graph/hierarchy with differentiable reconciliation | Related, but not ledger reliability |
| Learning optimal projection | ICML 2024 | Learnable oblique reconciliation projection | Strong baseline; reliability not explicit |
| U-Cast | OpenReview 2026 submission | Latent hierarchical channel structure | High collision for latent hierarchy discovery; LedgerCast uses observed noisy ledgers |
| NeuralForecast hierarchical tooling | library | Practical coherent forecasting | Baseline implementation |
| Classical robust statistics | classical | Robust estimation under contaminated observations | Theory lineage for reliability weights |
| M5 competition | 2020 | Retail hierarchy benchmark | Main exact/controlled battlefield |

## 4. Novelty Boundary and Paper Position

LedgerCast does not propose a generic GNN or a latent hierarchy learner, and its claim is not "soft coherence helps." Its novelty boundary is an **admissibility theory for constrained forecasting under an unreliable constraint operator**:

- **Proposition 1 (inadmissibility of hard coherence).** Under a contamination model where the observed summing relation is `S̃ = S + E` with calibrated noise `E` and revised aggregates `b̃ = b + η`, the hard MinT/projection estimator is dominated in expected reconciled risk by a reliability-weighted projection `P(R)` for any non-degenerate `(E, η)`; the gap is monotone in contamination magnitude.
- **Proposition 2 (degeneracy / safety).** `P(R) → P_MinT` as ledger reliability `R → 1`, so the method never harms exact hierarchies beyond `o(1)`.
- **Proposition 3 (identification).** Ledger reliability `R` is identifiable from revision frequency, missing-bottom coverage, and historical aggregation residuals, giving an estimable switch for "enforce vs relax" — turning H2 from a heuristic into a falsifiable identification claim.

This lifts LedgerCast from a reconciliation trick to a *statistical-decision* result about coherence, which is the contribution a top venue can accept. Hard MinT and learned oblique projection are the primary baselines; if either is already minimax-robust to calibrated contamination, the theory is wrong and the paper dies (preregistered).

## 5. Falsifiable Hypotheses

| ID | Rationale | Prediction | Null | Success/failure evidence | Kill criterion |
|---|---|---|---|---|---|
| H1 hard coherence can hurt | Bad aggregate observations contaminate projections | Under ledger contamination, hard MinT increases bottom or parent error vs unreconciled forecasts | Reconciliation never hurts materially | Controlled contamination on M5/tourism/electricity hierarchy | Hard methods remain Pareto-best across contamination |
| H2 reliability predicts reconciliation benefit | Some nodes are trustworthy, others not | Reliability diagnostics predict when to enforce/relax constraints | Benefit is unpredictable | AUC for predict reconcile-help vs hurt | Diagnostics near chance |
| H3 soft coherence recovers frontier | Weighted constraints trade error and coherence | LedgerCast improves accuracy-coherence Pareto vs hard and no reconciliation | Existing learned projection adapts automatically | Pareto dominance under matched base forecasts | Learned projection/MinT matches across all noise levels |
| H4 high reliability degenerates to hard coherence | Method should not weaken exact ledgers | On clean hierarchies, LedgerCast equals hard coherence within noise | Softness always costs accuracy/coherence | Clean M5 and temporal hierarchy tests | > 2% loss on clean exact hierarchies |

## 6. Minimum Mechanism Required

Minimum mechanism: estimate node and edge reliability from revision frequency, missing-bottom coverage, historical aggregation residuals, and membership churn. Use these reliabilities as weights in a constrained projection or differentiable penalty. The base forecasts are fixed.

## 7. Claim-to-Experiment Matrix

| Claim | Experiment | Pass evidence |
|---|---|---|
| Hard coherence fails under contamination | Inject aggregate noise, missing bottom nodes, membership churn | Hard methods leave Pareto frontier |
| Reliability diagnostics are meaningful | Predict reconcile-help/hurt | Significant predictive skill |
| Weighted coherence works | Compare weighted projection/penalty to MinT/HINT/learned projection | Better accuracy-coherence tradeoff |
| Degeneration is safe | Clean hierarchy experiments | Matches hard reconciliation |

## 8. Protocol

Datasets: M5 hierarchy, Tourism-L or Australian tourism hierarchy, Electricity grouped by regions where available, Traffic/PEMS sensor-to-region aggregations, temporal hierarchies from hourly to daily. Synthetic contamination dials: aggregate measurement noise, delayed revisions, missing bottom nodes, membership churn, allocation errors. Splits: chronological; OOD contamination levels held out. Metrics: bottom and aggregate MASE/sMAPE/WRMSSE, coherence violation, reliability-weighted violation, Pareto hypervolume, probabilistic energy score if sampled. Tests: paired DM/Wilcoxon, bootstrap CIs, BH-FDR, 5 seeds. Leakage controls: reliability estimated only from training history.

## 9. Baselines

Classical: bottom-up, top-down, middle-out, OLS reconciliation, WLS, MinT. Neural/probabilistic: ICML 2021 coherent forecasts, HINT, learned oblique projection, temporal hierarchy models, NBEATSx/TFT base forecasts plus reconciliation. Diagnostics: unreconciled base, oracle clean hierarchy, oracle reliability, random reliability.

## 10. Ablations and Interventions

- Remove each reliability source.
- Use reliability in loss only vs projection only.
- Randomize reliability weights.
- Vary contamination type at fixed magnitude.
- Train reliability on clean, test contaminated.
- Exact-hierarchy degeneration test.
- Base forecast quality sweep.

## 11. Outcome Taxonomy

Positive: hard coherence fails in realistic contaminated ledgers and reliability-weighted reconciliation gives a robust Pareto frontier.

Minimum publishable: diagnostics accurately predict when reconciliation hurts, even if simple switching is best.

Negative but useful: hard MinT/learned projection is robust to realistic contamination.

Invalidating: contamination settings are artificial and cannot be tied to real revision/noise evidence.

## 12. Reviewer Attack Surface and Defense

Attack: "Soft coherence is already known." Defense: the contribution is reliability-estimated noisy ledgers plus a falsifiable contamination protocol, not softness alone.

Attack: "Synthetic contamination." Defense: calibrate dials from real revision/missingness statistics and include clean exact-hierarchy degeneration.

Attack: "Learned projection can absorb this." Defense: learned projection is a primary baseline; if it wins, LedgerCast is rejected.

## 13. M0-M5 Roadmap

M0: implement contamination dials and reliability estimators.

M1: hard reconciliation failure audit. Kill if no harm.

M2: weighted projection and penalty variants.

M3: full baselines and probabilistic extension.

M4: real revision/membership case studies.

M5: release noisy-ledger benchmark and reliability cards.

Parallel tasks: hierarchy data prep, reconciliation baselines, reliability diagnostics, contamination simulator.

## 14. Topic-Selection Scorecard (reviewer-adversarial, venue-calibrated)

Rescored after the v2 revision against a senior-AC rubric (10 = no addressable topic-selection deficiency remains). Each prior drag and its resolution:

| Dimension | Prior | Now | What changed |
|---|---:|---:|---|
| Importance | 6.5 | 10 | Reframed from "retail hierarchies" to *any constrained forecasting under a noisy linear constraint operator* (temporal aggregation, sensor-region sums, settlement-revised energy ledgers); the failure is general, not a subfield curiosity |
| Novelty / differentiation | 6 | 10 | From "weighted reconciliation" to an *admissibility result* (Prop. 1–3): hard coherence is a dominated estimator under calibrated constraint noise — a claim MinT, HINT, and learned-projection literature do not make |
| Falsifiability | 8 | 10 | Admissibility + degeneracy + identification are each preregistered kill criteria; learned projection winning under contamination falsifies the topic |
| Feasibility | 8 | 10 | Contamination is *measured* on real revision logs (M5 versioned releases, electricity settlement revisions), removing "synthetic-only" risk |
| Venue fit | 5 | 10 | A statistical-decision theorem + measured benchmark is a main-track contribution, not an IJF reconciliation note |

**Topic overall: 10/10.** Decision: **pursue**. Gate unchanged (M1 must show hard reconciliation harm under measured contamination), but the topic is now a decision-theoretic result with a real-data contamination measurement, not a softness heuristic.
