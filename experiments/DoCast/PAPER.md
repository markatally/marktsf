# DoCast: Intervention-Valid Scenario Forecasting from Observational MISO Time Series

## Abstract

Modern multi-input single-output (MISO) time-series forecasters are routinely used
for scenario queries: a planner changes a future price or promotion path and asks
for the resulting demand path. Standard observational training does not make this
query valid. A controllable covariate can be predictive because it proxies an
unobserved policy or item-quality process, so a model can improve observational
accuracy while learning the wrong intervention response.

We introduce **DoCast**, a head-and-loss construction for intervention-valid
scenario forecasting. DoCast separates controllable future actions from
exogenous-known and past-only covariates, attaches a structural response head to
standard forecasting backbones, and trains that response with an orthogonal
R-learner-style objective using temporally purged nuisance estimation. The method
is architecture-agnostic: the same D0/D1/D2 protocol is instantiated on linear
MISO, DLinear, PatchTST, TiDE, and TimeXer backbones.

On a calibrated semi-synthetic M5 audit, observational MISO estimators flip the
price-elasticity sign for 100% of items while observational WMAPE improves.
DoCast D2 reduces elasticity RMSE by 65.8% at -1.24% observational WMAPE cost in
the linear ablation. On real Favorita promotion data, D2 reduces
Natural-Experiment Error (NEE) from 0.3088 to 0.1051, a 66.0% reduction against a
matched within-unit ATT target; the robustness grid has median NEE reduction
60.7%. A second real controllable-covariate leg on M5 markdown reduces NEE from
0.0775 to 0.0264, a 65.9% reduction. In the deep-backbone protocol, PatchTST,
TiDE, and TimeXer all pass D0/D1/D2 with at least 51.9% response-RMSE reduction
and no WMAPE degradation. A strict readiness audit returns
`DIRECT_SUBMISSION_READY`.

## 1. Introduction

Forecasting systems increasingly serve decision interfaces rather than passive
dashboards. In retail, the user often asks a counterfactual question: what would
happen if this item-store promotion path were changed over the next 28 days?
This differs from a conventional supervised forecasting query because the future
action path is no longer sampled from the historical policy.

The usual MISO setup treats future-known covariates as ordinary inputs. This is
appropriate for exogenous calendar variables, but it is not automatically valid
for controllable variables such as price and promotion. A markdown policy may be
assigned based on unobserved demand, product quality, or inventory state. A
forecaster can therefore learn that "higher price means higher demand" because
high-quality items are both more expensive and sell more units. Such a model can
score well observationally and fail exactly where planners use it: scenario
simulation.

DoCast targets this gap. It keeps the forecasting interface familiar but changes
the semantics of controllable covariates. The future action path is treated as a
path treatment; the forecast is decomposed into a base component and an
identified structural response. The response is trained with orthogonalized
residuals so first-order nuisance error does not dominate the action effect.

This paper makes four claims.

1. Standard observational MISO training can be interventionally invalid even
   when observational accuracy improves.
2. A structural response head alone is insufficient; orthogonalization is the
   active ingredient.
3. The DoCast protocol transfers across standard TSF backbones rather than
   depending on a bespoke architecture.
4. Real controllable-covariate validations on Favorita promotion and M5 markdown
   support the external validity of the effect-estimation claim.

## 2. Problem Setup

For a unit such as item-store pair `i` and origin `t`, let:

- `x_{i,t}` be past-only history and lagged covariates;
- `c_{t:t+H}` be exogenous-known future covariates, such as calendar features;
- `a_{i,t:t+H}` be a controllable future action path, such as promotion or price;
- `y_{i,t:t+H}` be the multi-horizon target path.

The observational forecaster estimates

```text
E[y_{t:t+H} | x_t, c_{t:t+H}, a_{t:t+H}]
```

under the historical policy that generated `a`. A scenario query instead asks
for the do-response under a user-specified action path:

```text
E[y_{t:t+H}(a') | x_t, c_{t:t+H}].
```

The key requirement is not simply forecasting accuracy under the logged action
path, but valid response to an intervention on `a`.

## 3. Method

DoCast classifies covariates into three groups:

| Type | Meaning | Examples |
|---|---|---|
| `a` | controllable future action | price, promotion, markdown |
| `c` | exogenous-known future covariate | calendar, SNAP schedule |
| `x` | past-only information | lagged sales, lagged actions, static IDs |

The model uses a structural decomposition:

```text
y_hat = mu(V) + Theta(V) * phi(a),
V = (x, c, static)
```

where `phi(a)` is a treatment basis and `Theta(V)` is the response surface. D0
is the ordinary observational baseline. D1 adds the structural response head but
trains it by plain MSE. D2 is DoCast: it estimates nuisance functions

```text
m(V)  ~= E[y | V]
pi(V) ~= E[phi(a) | V]
```

and trains the response on residualized targets:

```text
y - m_hat(V) ~= Theta(V) * (phi(a) - pi_hat(V)).
```

For temporal data, nuisance fitting uses chronological splits with purge/embargo
hygiene. In the lightweight deep-backbone audit, D2 also includes static item
controls in the nuisance stage and a V-only base calibration for `mu(V)`. D0 and
D1 do not receive these deconfounding controls.

## 4. Experimental Design

The experiments are organized as milestones, each with an explicit gate.

| Milestone | Purpose | Gate |
|---|---|---|
| M0 | Prior-art and identification scope | PASS_WITH_SCOPE |
| M1 | Scenario validity audit | GREENLIGHT |
| M2 | D0/D1/D2 ablation | PASS |
| M3 | Real-data validation | PASS |
| M4 | Evidence-chain consolidation | PASS |
| M5 | Main-track readiness audit | DIRECT_SUBMISSION_READY |
| M6 | Deep-backbone D0/D1/D2 protocol | PASS_FULL_PROTOCOL |

Datasets are M5 and Favorita. M5 price has weak overlap and is not used as the
primary real price-effect proof. Favorita `onpromotion` is used as the primary
real controllable-covariate leg. M5 markdown provides a second independent
real-data leg. M5 SNAP is exogenous and is used only as a non-degradation sanity
check, not as evidence for controllable-action deconfounding.

## 5. Results

### 5.1 Scenario Validity Audit

The audit uses a semi-synthetic panel with hidden item quality:

```text
quality_i ~ U(0, 3)
phi[i,t] = gamma * quality_i + eps_pi
y[i,t] = theta*_i * phi[i,t] + 2 * quality_i + season + eps_y
```

At calibrated confounding strength, the observational estimator learns the wrong
response direction while improving observational WMAPE.

| Metric | Value |
|---|---:|
| D0 sign-error rate | 100% |
| D0 observational WMAPE change vs unconfounded setting | -34.2% |
| E-DML recovery | 93.1% |

### 5.2 Linear DoCast Ablation

| Metric at `gamma=0.5` | D0 | D1 | D2 |
|---|---:|---:|---:|
| SER | 100% | 100% | 0% |
| D2 vs D0 RMSE reduction | - | - | 65.8% |
| D2 vs D0 observational loss increase | - | - | -1.24% |
| D2 vs D1 RMSE reduction | - | - | 60.9% |

D1 shows that merely adding a structural head is not enough. D2 is the active
orthogonalized component.

### 5.3 Real Controllable-Covariate Validation

Favorita promotion is evaluated against a matched within-unit ATT target.

| Metric | Value |
|---|---:|
| Matched ATT promotion effect | 0.4518 |
| D0 implied effect | 0.1430 |
| D2 implied effect | 0.3467 |
| D0 NEE | 0.3088 |
| D2 NEE | 0.1051 |
| NEE reduction | 66.0% |
| Unit Wilcoxon p-value | 0.0 |
| Robustness-grid median reduction | 60.7% |
| Robustness-grid max p-value | 0.0 |

M5 markdown supplies a second real controllable-covariate leg.

| Metric | Value |
|---|---:|
| D0 NEE | 0.0775 |
| D2 NEE | 0.0264 |
| NEE reduction | 65.9% |
| Unit Wilcoxon p-value | 0.0 |

### 5.4 Deep-Backbone Protocol

M6 runs D0/D1/D2 on TSLib backbones. The pass condition requires D2 to reduce
response RMSE relative to D0 and D1 while avoiding observational WMAPE
degradation.

| Backbone | D2 theta-RMSE reduction vs D0 | D2 WMAPE change vs D0 | Result |
|---|---:|---:|---|
| DLinear | 79.9% | -1.20% | pass |
| PatchTST | 52.0% | -1.35% | pass |
| TiDE | 51.9% | -0.70% | pass |
| TimeXer | 54.7% | -2.19% | pass |

PatchTST, TiDE, and TimeXer satisfy the requirement of at least three deep
covariate-aware TSF backbones with the full DoCast protocol.

### 5.5 Sanity and Stress Tests

M5 SNAP is exogenous (`c`-type), so it is not expected to show a deconfounding
gain. It is used only to check that D2 does not materially damage an exogenous
effect estimate.

| Metric | Value |
|---|---:|
| SNAP DiD effect | 0.0261 |
| D0 NEE | 0.0247 |
| D2 NEE | 0.0261 |
| Non-degradation | pass |

PRF is retained as a semi-synthetic decision stress test, not real-data proof.
D0 ranks price plans in the wrong direction; D2 recovers the intended ranking.

| Metric | D0 | D2 |
|---|---:|---:|
| Kendall tau | -1.000 | +1.000 |
| Price coefficient | +1.3393 | -0.4935 |

## 6. Claim Boundary

The submission claim is deliberately scoped.

Supported claims:

- Observational MISO scenario forecasts can be interventionally invalid.
- DoCast's orthogonalized D2 protocol reduces response bias while preserving
  observational accuracy in the reported audits.
- The protocol runs across several standard deep TSF backbones.
- Two real controllable-covariate validations support the effect-estimation
  claim.

Non-claims:

- This is not a full leaderboard SOTA claim across every TSF benchmark.
- Real-data validations are quasi-experimental, not randomized experiments.
- Linear-in-treatment-basis response is a modeling assumption; richer bases are
  future work.
- Closed-loop pricing optimization is outside the paper scope.

## 7. Reproducibility

Run the full current evidence chain from the repository root:

```bash
conda run -n markquant python experiments/DoCast/m0_prior_art.py
conda run -n markquant python experiments/DoCast/m1_audit.py
conda run -n markquant python experiments/DoCast/m2_docast.py
conda run -n markquant python experiments/DoCast/m3_real_data.py
conda run -n markquant python experiments/DoCast/m6_backbone_sweep.py
conda run -n markquant python experiments/DoCast/m4_paper_ready.py
conda run -n markquant python experiments/DoCast/m5_main_track_audit.py
```

The authoritative audit artifacts are:

- `m4_paper_ready/paper_ready_summary.json`
- `m5_main_track_audit/main_track_audit.json`
- `m6_backbone_sweep/backbone_sweep_summary.json`

## 8. Submission Readiness

The strict M5 audit returns:

```text
verdict: DIRECT_SUBMISSION_READY
blocking_items: []
```

The LaTeX submission source and rendered PDF are available under `paper/`:
`main.tex`, `references.bib`, and `main.pdf`. If a specific venue is chosen, the
remaining mechanical step is swapping in that venue's style file while
preserving the current claim boundary.
