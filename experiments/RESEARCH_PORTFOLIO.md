# Research Portfolio - New Time-Series Forecasting Proposals

Generated: 2026-06-14. Scope: `experiments/` only. Existing projects `DoCast` and `PRISM` were audited as occupied research space and were not modified.

## 1. Repository Audit

| Project | Problem | Thesis | Hypotheses | Mechanism | Datasets | Evidence status | Scope | Risks |
|---|---|---|---|---|---|---|---|---|
| DoCast | Scenario forecasting with controllable known-future covariates | MISO forecasters answer observational queries while users need interventional `do(a)` scenarios | SOTA scenario bias under confounding; orthogonal structural head improves validity | Covariate typing, structural response head, sequential orthogonalization | M5, Favorita, semi-synthetic confounding dials | Proposal-stage; strong gate design | Offline single-model causal semantics | Prior-art collision with causal demand forecasting; identification strength |
| PRISM | Non-stationary MISO forecasting where best inductive bias changes over time | Architecture choice is a latent regime state, not a static dataset property | Oracle drift, heterogeneous expert routing, dynamic beta, routing-level adaptation | SSM regime filter, heterogeneous expert library, sparse routing, CI/CD gate | ETT primary after finance gates; finance appendix | M1b gate results already recorded | Regime tracking and expert routing | 2026 heterogeneous-MoE wave; finance leg failure |

## 2. Research-Space Map

Occupied axes:

- **DoCast owns** causal/interventional validity of controllable covariates, orthogonal learning, pricing/promotion scenario validity, retail intervention evaluation.
- **PRISM owns** latent regimes, dynamic architecture selection, heterogeneous MoE, routing-level adaptation, CI/CD coupling under drift, oracle drift studies.

Open axes selected here:

- Missingness mechanism shift and sensor fault semantics.
- Release-time covariate availability and stale inputs.
- Pathwise and tail-conditional uncertainty validity.
- Forecast-origin benchmark legality.
- Noisy hierarchy reconciliation.
- Sampling-rate and aggregation shift.
- Leakage-proof support selection for in-context TSFMs.

Rejected crowded axes:

- Generic conformal TS wrappers.
- Change-point conformal state tracking.
- Unified missing/asynchronous architectures.
- New TSF foundation models.
- Generic hierarchical GNNs.
- Broad benchmark suites without executable legality checks.
- New MoE/routing systems.

## 3. Literature Audit Summary

Primary local sources: `docs/PAPER.md`, `docs/DATASET_AUDIT.md`, local PDFs in `paper/`, plus web/OpenReview/arXiv searches. The audit found strong collision constraints:

- Missing/irregular TSF is crowded: GRU-D, BRITS, Latent ODE, Neural CDE, mTAN, CSDI, Raindrop, GraFITi, S4M, SADI, ChannelTokenFormer, and recent MNAR blackout work.
- Probabilistic/conformal TSF is crowded: NeurIPS 2021 conformal TS, EnbPI, AgACI, SPCI, HopCPT, Conformal PID, Conformal Risk Control, ProbTS, CPTC, CoRel, decision-calibrated sets, ConformalNaive+.
- Hierarchical reconciliation is mature: MinT, ICML 2021 coherent probabilistic forecasts, HINT, temporal hierarchies, graph clustering, and learned oblique projection.
- TSFM evaluation is active: Chronos, Moirai, TimesFM, GIFT-Eval, Rethinking TSFM Evaluation, TIME, in-context fine-tuning, FlowState.
- Tail and extreme forecasting has prior art: KDD 2019 extreme event forecasting, EVT losses, EVEREST, and heavy-tail mixture models.

The retained proposals therefore avoid "first method" claims and stake narrower, falsifiable positions about failure conditions and experimental contracts.

## 4. Candidate Pool and Decisions

Scoring uses 0-10 for importance, authenticity, falsifiability, novelty, position clarity, mechanism coherence, claim-experiment alignment, scientific insight, reviewer defensibility, evaluation validity, feasibility, and differentiation. Retention threshold: no score below 8 and average >= 9.

| # | Candidate | Core failure condition | Decision | Avg | Main reason |
|---:|---|---|---|---:|---|
| 1 | MaskShift | Same missing rate, different missing mechanism | Retain | 9.33 | Strong audit and minimal correction; clear kill |
| 2 | LagCast | Exogenous covariates are stale/delayed | Retain | 9.17 | High deployment authenticity; distinct from causal covariates |
| 3 | PathCal | Horizon coverage fails path events | Retain | 9.25 | Clear uncertainty-validity gap |
| 4 | TailCal | Average metrics hide tail miscalibration | Retain | 9.08 | Important but crowded; method novelty constrained |
| 5 | AvailAudit | Inputs unavailable at forecast origin | Retain | 9.00 | High value; must be executable, not a survey |
| 6 | LedgerCast | Hard coherence under noisy ledgers | Retain/hold at gate | 9.00 | Near threshold; passes only with M1 evidence |
| 7 | FlowRes | Sampling-rate/aggregation shift | Retain | 9.00 | Strong mechanism via aliasing; FlowState is direct baseline |
| 8 | SupportCast | In-context support chosen with future info | Retain/gate carefully | 9.00 | Directly relevant to TSFMs; collision manageable |
| 9 | Generic conformal wrapper | Non-exchangeable TS | Reject | 7.1 | EnbPI/SPCI/PID/CPTC occupy it |
| 10 | Change-point conformal forecaster | Regime shifts with CP | Reject | 7.3 | CPTC and PRISM collision |
| 11 | Graph conformal intervals | Cross-series correlated residuals | Reject | 7.4 | CoRel/HopCPT too close |
| 12 | Decision-calibrated power sets | Downstream robust operation | Reject | 7.8 | Conformal Risk Control and 2026 decision-calibrated work |
| 13 | Unified missing/asynchronous architecture | Robust real-world MTSF | Reject | 7.2 | ChannelTokenFormer direct collision |
| 14 | Imputation harms forecasting | Imputation-first under missingness | Reject/merge | 7.7 | Recent MTSF-M and blackout work too close |
| 15 | Sensor-fault graceful degradation | Corrupted covariates | Hold | 8.4 | Could merge into MaskShift |
| 16 | Gap-topology OOD benchmark | Unseen observation topology | Hold | 8.3 | High collision with GraFITi/ProFITi/CTF |
| 17 | Mask-shift conformal | Missingness shift coverage | Hold/merge | 8.6 | Good but narrower than MaskShift/PathCal |
| 18 | TSFM calibration audit | Foundation model calibration | Reject | 7.6 | Adler/ProbTS/crowded |
| 19 | Selective uncertainty action gate | Escalate when forecast unreliable | Hold | 8.5 | Needs sharper domain and baseline defense |
| 20 | Conformal naive selector | Learned models vs naive floor | Reject | 7.5 | ConformalNaive+ direct collision |
| 21 | Constraint-aware frozen TSFM adapter | Coherent frozen TSFM | Hold | 8.4 | Obvious composition unless strong theorem |
| 22 | Generic GNN hierarchy forecaster | Learn hierarchy structure | Reject | 7.0 | Cini/U-Cast collision |
| 23 | Reconciliation helps-vs-hurts diagnostics | Predict reconciliation benefit | Hold/merge | 8.6 | Could become LedgerCast diagnostic |
| 24 | Latent dynamic hierarchy | Changing membership panels | Reject | 7.8 | U-Cast and PRISM-adjacent |
| 25 | Multi-resolution decision loss | Coherent temporal decisions | Hold | 8.2 | Needs more authentic use case |
| 26 | Covariate availability protocol only | Non-causal leakage | Merge | 8.8 | Folded into AvailAudit/LagCast |
| 27 | TSFM from scratch for irregular/missing | Pretrain robust FM | Reject | 6.9 | Resource-heavy and architecture-first |
| 28 | Negative-transfer global pooling | Predictive heterogeneity | Reject | 7.9 | 2026 predictive heterogeneity direct collision |
| 29 | Intermittent demand global/local | Sparse demand distributions | Reject | 7.8 | Recent intermittent global-vs-local paper; too narrow |
| 30 | Real-time vintage forecasting | Finalized vs first-release labels | Hold | 8.3 | Authentic but data acquisition risk high |
| 31 | Multimodal text leakage audit | Text descriptions leak future | Reject | 7.7 | High-fidelity multimodal benchmark/WIT overlap |
| 32 | Foundation model prompt abstention | When not to use prompts | Merge | 8.5 | Folded into SupportCast utility abstention |
| 33 | Cross-domain pretraining curriculum | Avoid negative transfer | Reject | 7.5 | Foundation-model scale and DELPHYNE overlap |
| 34 | Causal missingness in controllable covariates | Missing promo/price interventions | Reject | 7.0 | DoCast collision |

## 5. Retained Proposal Ranking

| Rank | Slug | Title | Decision | Gate |
|---:|---|---|---|---|
| 1 | `MaskShift` | Forecasting Under Missingness Mechanism Shift | Pursue | Mechanism explains degradation beyond rate |
| 2 | `PathCal` | Pathwise Calibration for Multi-Horizon Forecasting | Pursue | Path events miscalibrated by standard intervals |
| 3 | `LagCast` | Forecasting With Stale and Asynchronous Covariates | Pursue | Release-time alignment changes covariate gains |
| 4 | `TailCal` | Tail-Conditional Calibration for Extreme Forecasts | Pursue | Tail reliability diverges from aggregate metrics |
| 5 | `FlowRes` | Sampling-Rate Robust Forecasting | Pursue | Cross-resolution rank reversals exist |
| 6 | `SupportCast` | Leakage-Proof Support Selection for In-Context TSFMs | Gate carefully | Naive support gain shrinks under legal selection |
| 7 | `AvailAudit` | Forecast-Origin Availability Auditing | Pursue as benchmark validity | Corrected loaders change conclusions or checks catch real bugs |
| 8 | `LedgerCast` | Reliability-Weighted Forecast Reconciliation | Hold at M1 | Hard reconciliation hurts under calibrated ledger contamination |

Shortfall: only 8 proposals pass the stated threshold. Two additional ideas could be made plausible, but forcing them would violate the rejection rules.

## 6. Score Table

| Slug | Imp | Auth | Fals | Nov | Pos | Mech | C-E | Insight | Def | Eval | Feas | Diff | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MaskShift | 10 | 10 | 10 | 9 | 9 | 9 | 10 | 9 | 9 | 9 | 9 | 9 | 9.33 |
| PathCal | 9 | 9 | 10 | 9 | 9 | 9 | 10 | 9 | 8 | 10 | 10 | 9 | 9.25 |
| LagCast | 10 | 10 | 10 | 9 | 9 | 9 | 10 | 9 | 8 | 9 | 8 | 9 | 9.17 |
| TailCal | 10 | 10 | 10 | 8 | 9 | 9 | 10 | 9 | 8 | 9 | 9 | 8 | 9.08 |
| AvailAudit | 10 | 10 | 9 | 8 | 9 | 9 | 9 | 9 | 8 | 10 | 9 | 8 | 9.00 |
| FlowRes | 9 | 9 | 10 | 8 | 9 | 10 | 10 | 9 | 8 | 9 | 9 | 8 | 9.00 |
| SupportCast | 9 | 9 | 10 | 8 | 9 | 9 | 10 | 9 | 8 | 10 | 8 | 9 | 9.00 |
| LedgerCast | 9 | 9 | 10 | 8 | 9 | 9 | 10 | 9 | 8 | 9 | 9 | 9 | 9.00 |

## 7. Pairwise Overlap Matrix

Scale: 0 none, 1 low shared infrastructure, 2 adjacent evaluation axis, 3 material overlap requiring coordination.

| Pair | DoCast | PRISM | MaskShift | LagCast | PathCal | TailCal | AvailAudit | LedgerCast | FlowRes | SupportCast |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DoCast | - | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 |
| PRISM | 0 | - | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 |
| MaskShift | 0 | 1 | - | 2 | 1 | 0 | 1 | 0 | 0 | 0 |
| LagCast | 1 | 0 | 2 | - | 0 | 0 | 3 | 0 | 0 | 1 |
| PathCal | 0 | 0 | 1 | 0 | - | 2 | 0 | 0 | 0 | 0 |
| TailCal | 0 | 1 | 0 | 0 | 2 | - | 0 | 0 | 0 | 0 |
| AvailAudit | 1 | 1 | 1 | 3 | 0 | 0 | - | 1 | 1 | 3 |
| LedgerCast | 0 | 0 | 0 | 0 | 0 | 0 | 1 | - | 1 | 0 |
| FlowRes | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | - | 1 |
| SupportCast | 0 | 0 | 0 | 1 | 0 | 0 | 3 | 0 | 1 | - |

Coordination notes:

- `AvailAudit` overlaps strongly with `LagCast` and `SupportCast`; it should provide shared legality schemas, while those projects provide focused ML papers.
- `PathCal` and `TailCal` share conformal/probabilistic infrastructure but target different estimands.
- `MaskShift` and `LagCast` both concern input availability but differ: missing/corrupt observations vs delayed covariate release.
- No retained proposal has high overlap with DoCast or PRISM.

## 8. Execution Priority

1. **MaskShift**: highest chance of decisive gate within two weeks using existing datasets and synthetic masks.
2. **PathCal**: post-hoc on existing probabilistic outputs; low implementation risk.
3. **LagCast**: high impact but depends on release-time metadata.
4. **FlowRes**: strong if BOOM/ETT resolution pairs are easy to construct.
5. **TailCal**: important; run after PathCal because it shares calibration infrastructure.
6. **SupportCast**: depends on TSFM inference harness and support APIs.
7. **AvailAudit**: run as shared infrastructure and paper only if corrections materially change rankings.
8. **LedgerCast**: run M1 gate cheaply; stop if hard reconciliation remains robust.

## 9. Shared Infrastructure

- Chronological split, purge, and embargo utilities.
- Seed runner with paired DM, Wilcoxon, bootstrap CIs, and BH-FDR.
- Baseline harness: seasonal naive, ARIMA/ETS where feasible, DLinear, PatchTST, iTransformer, TimeXer, TFT, NBEATSx.
- Probabilistic/conformal harness for PathCal and TailCal.
- Availability manifest schema shared by AvailAudit, LagCast, and SupportCast.
- Stress-test generators: masks for MaskShift, latency for LagCast, tail stress for TailCal, resolution transforms for FlowRes, ledger contamination for LedgerCast.
- Model cards reporting compute, parameter count, legal input set, split protocol, and leakage controls.

## 10. File Outputs

Created proposal folders:

- `experiments/MaskShift/docs/PROPOSAL.md`
- `experiments/MaskShift/docs/LITERATURE_COLLISION.md`
- `experiments/LagCast/docs/PROPOSAL.md`
- `experiments/LagCast/docs/LITERATURE_COLLISION.md`
- `experiments/PathCal/docs/PROPOSAL.md`
- `experiments/PathCal/docs/LITERATURE_COLLISION.md`
- `experiments/TailCal/docs/PROPOSAL.md`
- `experiments/TailCal/docs/LITERATURE_COLLISION.md`
- `experiments/AvailAudit/docs/PROPOSAL.md`
- `experiments/AvailAudit/docs/LITERATURE_COLLISION.md`
- `experiments/LedgerCast/docs/PROPOSAL.md`
- `experiments/LedgerCast/docs/LITERATURE_COLLISION.md`
- `experiments/FlowRes/docs/PROPOSAL.md`
- `experiments/FlowRes/docs/LITERATURE_COLLISION.md`
- `experiments/SupportCast/docs/PROPOSAL.md`
- `experiments/SupportCast/docs/LITERATURE_COLLISION.md`

## 11. Next Gate Experiments

| Proposal | Cheapest decisive test | Stop if |
|---|---|---|
| MaskShift | Matched-rate MCAR vs operational masks on DLinear/PatchTST/TimeXer | Mechanism has no significant effect |
| LagCast | Event-time vs simulated/recovered release-time alignment | Covariate gains do not shrink |
| PathCal | Audit path event coverage on existing probabilistic forecasts | Path events are already calibrated |
| TailCal | Compare CRPS/WIS ranking to tail exceedance ranking | Aggregate metrics already predict tail safety |
| AvailAudit | Mutation tests and one corrected loader | Checker cannot catch seeded violations |
| LedgerCast | Inject calibrated ledger contamination into M5 hierarchy | Hard MinT never hurts |
| FlowRes | Train same-rate, test downsampled/upsampled same process | Ranking is stable |
| SupportCast | Full-series vs pre-origin support selection | Naive support gain does not shrink |
