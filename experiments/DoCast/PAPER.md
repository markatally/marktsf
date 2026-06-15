# DoCast: Orthogonalized Scenario Forecasting with Controllable Future Covariates

## Abstract

Forecasting models are often used as decision simulators: a planner changes a future promotion, price, outreach, or intervention path and asks for the resulting outcome path. Standard MISO forecasters are not trained for this query. A controllable future covariate can be predictive because it encodes the historical decision policy, so logged-policy accuracy can improve while the learned intervention response is wrong. DoCast is an architecture-agnostic response head trained by an orthogonal R-learner objective with temporally purged nuisance estimation. The revised paper states the estimand, assumptions, overlap-based failure policy, and limits of matched observational validation. In semi-synthetic M5, DoCast reduces response RMSE by 65.7% in a hidden-confounder stress test. Under shared controls, it reduces RMSE by 58.1% versus a fair structural D1 head. On Favorita promotion and M5 markdown data, it reduces Natural-Experiment Error by 66.0% and 65.9% against matched within-unit ATT proxies. In the corrected fair-control backbone audit, DLinear, PatchTST, TiDE, and Transformer pass the strict seed-level protocol; TimeXer improves mean response RMSE by 27.2% versus fair D1 but is reported as a stability caveat because one seed exceeds the 5% WMAPE-degradation tolerance.

# Introduction

Forecasting systems increasingly sit behind planning interfaces. Retail planners ask what demand would be under a proposed promotion path; inventory teams ask how markdowns affect future sales; quality-assurance offices in higher education face analogous scenario questions about outreach intensity, advising interventions, scholarship offers, or capacity policies under fixed cohort and calendar contexts. In all of these cases, the user is not merely asking for the outcome under the logged future covariates. The user is setting part of the future covariate path.

The usual MISO training objective does not distinguish exogenous known-future covariates from controllable actions. This is harmless for variables such as calendar features. It is problematic for variables such as price, promotion, or policy intensity, because those variables are often assigned by an adaptive historical policy. A model can learn that higher prices imply higher demand if high-quality items are both priced higher and sell more. The model may then look better under observational WMAPE while giving the wrong answer to the planner’s scenario query.

DoCast addresses this specific failure mode. It keeps the forecasting interface and backbone architecture intact, but changes how controllable future covariates are interpreted and trained. The future action path is treated as a path treatment. The forecast is decomposed into a base term and a structural response term, and the response is trained with residualized outcome and action signals so first-order nuisance error does not dominate the action effect.

The contribution is deliberately scoped. We do not claim that causal forecasting is a new field, nor that matched observational targets are randomized ground truth. Instead, this paper contributes:

1.  a formal estimand and assumption set for MISO scenario forecasting with controllable future covariates;

2.  an architecture-agnostic DoCast head/loss protocol for path-treatment response estimation;

3.  a fair-control experimental design that separates hidden-confounder stress tests from orthogonal-loss ablations; and

4.  real-data matched-ATT proxy validations on two public retail controllable-covariate legs, with explicit limitations.

The empirical benchmarks are retail datasets because they provide public logged actions and outcomes. The higher-education quality-assurance motivation is a deployment class for scenario forecasting, not an empirical claim about private higher-education intervention logs.

# Related Work

#### Forecasting with known covariates.

Temporal Fusion Transformers, vanilla Transformers, DLinear, PatchTST, TiDE, and TimeXer are representative backbones for multi-horizon forecasting with known or exogenous covariates (Lim et al., 2021; Vaswani et al., 2017; Zeng et al., 2023; Nie et al., 2023; Das et al., 2023; Wang et al., 2024). These models are designed for predictive accuracy under the observed data distribution. DoCast is not a competing backbone. It is a response-estimation layer for the subset of known-future inputs that a planner can set.

#### Causal and counterfactual time-series learning.

Prior work has already shown that observational forecast risk can differ from causal forecast risk (Vankadara et al., 2022), that sequential counterfactual prediction requires treatment-assignment bias control (Bica et al., 2020), and that pricing forecasts can be cast as causal forecasting problems (Schultz et al., 2023). Recent work by Crasson et al. (2024) is especially close: it also combines orthogonal statistical learning with time-series forecasting and evaluates with quasi-experimental treatment-effect targets. DoCast differs by focusing on a MISO forecasting interface with future path treatments, by making the controllable/exogenous covariate taxonomy operational, and by testing the same head/loss protocol across several TSF backbones.

#### Orthogonal learning.

DoCast draws directly from double/debiased machine learning (Chernozhukov et al., 2018), R-learning for heterogeneous treatment effects (Nie and Wager, 2021), orthogonal statistical learning (Foster and Syrgkanis, 2023), and dynamic DML (Lewis and Syrgkanis, 2021). The novelty is not Neyman orthogonality itself. The contribution is the mapping of that machinery onto MISO scenario forecasting with multi-horizon controllable future covariates and an observational-accuracy constraint.

# Estimand and Identification

For unit $`i`$ and forecast origin $`t`$, let $`X_{i,t}`$ denote past outcomes and lagged covariates, $`C_{i,t:t+H}`$ known exogenous future covariates, and $`A_{i,t:t+H}`$ a controllable future action path. Let $`V_{i,t}=(X_{i,t},C_{i,t:t+H},S_i)`$ collect the observed non-action context, including allowed static controls $`S_i`$. The logged MISO predictor estimates
``` math
\begin{equation}
  \mathbb{E}[Y_{i,t:t+H}\mid V_{i,t}, A_{i,t:t+H}],
\end{equation}
```
where $`A`$ is sampled from the historical policy. A scenario query asks instead for
``` math
\begin{equation}
  \psi_h(v,a) =
  \mathbb{E}\!\left[Y_{i,t+h}(a)\mid V_{i,t}=v\right],
  \quad h=1,\ldots,H,
  \label{eq:estimand}
\end{equation}
```
where $`Y_{i,t+h}(a)`$ is the potential outcome under the planner-specified path $`a`$.

The estimand in Eq. <a href="#eq:estimand" data-reference-type="ref" data-reference="eq:estimand">[eq:estimand]</a> is identified only under standard but strong conditions:

1.  **Consistency and timing.** The observed outcome equals the potential outcome under the action path actually taken, and the action path is assigned before the affected outcomes.

2.  **Sequential ignorability.** Conditional on $`V_{i,t}`$, the logged action path is independent of the relevant potential outcome path.

3.  **Overlap.** The requested scenario path lies inside the historical conditional support of actions for comparable $`V`$.

4.  **No interference at the modeled unit level.** One unit’s action path does not change another unit’s outcome unless such spillovers are explicitly modeled in $`V`$.

5.  **Stable measurement.** Outcome, action, and covariate definitions do not change between the logged policy and scenario use.

These assumptions are not guaranteed by DoCast. They are preconditions for the scenario query. A deployment should refuse, downweight, or label a scenario as unsupported when overlap diagnostics fail, for example when the residualized action variance is near zero or the proposed action path falls outside the empirical action range for similar contexts.

# Method

DoCast classifies future covariates into controllable actions $`A`$ and exogenous-known covariates $`C`$. Given a treatment basis $`\phi(A)`$, the forecast is represented as
``` math
\begin{equation}
  \hat Y_{t:t+H} = \mu(V_t) + \Theta(V_t)\phi(A_{t:t+H}),
  \label{eq:structural}
\end{equation}
```
where $`\mu`$ is a base forecast and $`\Theta`$ is the response surface. The same forecasting backbone can be used inside $`\mu`$, inside nuisance models, or as a feature extractor for $`\Theta`$.

#### Baselines and ablations.

D0 is the observational MISO baseline: the action path is an ordinary input or ordinary treatment head under MSE training. D1 uses the same structural response parameterization as D2 but trains it by plain MSE. D2 is DoCast. It estimates nuisance functions
``` math
\begin{equation}
  m(V)\approx\mathbb{E}[Y\mid V],
  \qquad
  \pi(V)\approx\mathbb{E}[\phi(A)\mid V],
\end{equation}
```
using chronological folds with purge/embargo gaps, then trains the response by
``` math
\begin{equation}
  \min_{\Theta}
  \sum
  \left[
    \{Y-\hat m(V)\}
    -
    \Theta(V)\{\phi(A)-\hat \pi(V)\}
  \right]^2 .
  \label{eq:orthogonal}
\end{equation}
```
At test time the forecast is reported in the equivalent form $`\hat \mu(V)+\hat\Theta(V)\phi(A)`$.

#### Fair-control rule.

Reviewer-facing comparisons must distinguish two settings. In the hidden-confounder stress test, D0/D1 deliberately lack the static proxy that D2 uses in its nuisance models; this demonstrates how scenario failure can occur when the policy driver is unobserved by the forecaster. In the fair-control diagnostic and deep-backbone protocol, D0, D1, and D2 all receive the same item static controls, and D1/D2 share item-specific response capacity. Claims about orthogonalization are made against this fair D1 comparison.

# Experimental Design

We use M5 (Makridakis et al., 2022) and Favorita (Corporacion Favorita, 2018). The experiments answer three questions.

#### Q1: Can observational accuracy hide the wrong scenario response?

The semi-synthetic panel uses real M5 demand as a baseline and injects hidden item quality $`q_i`$:
``` math
\begin{align}
  q_i &\sim U(0,3),\\
  \phi_{i,t} &= \gamma q_i + \epsilon^\pi_{i,t},\\
  y_{i,t} &= y^{\mathrm{M5}}_{i,t}
    + \theta_i^\star\phi_{i,t}
    + 2q_i + \epsilon^y_{i,t}.
\end{align}
```
Here $`\theta_i^\star<0`$ is the true price response. Positive $`\gamma`$ makes the logged price proxy item quality.

#### Q2: Does DoCast align better with real matched effect proxies?

The real-data target is a matched within-unit ATT proxy, not a randomized ground truth. For Favorita promotion, promoted rows are matched to the same unit’s non-promoted rows on the same weekday where possible, with all non-promoted rows for the unit as fallback. Eligible units must satisfy minimum within-unit treated and control counts. D0 is a pooled observational treatment coefficient with lag, holiday, weekday, and trend controls. D2 residualizes promotion and outcome on unit, date, weekday, holiday, and lag controls before estimating the response. M5 markdown uses the same design with a binary discount-depth treatment defined by log discount greater than 0.05.

#### Q3: Is the head/loss protocol backbone-specific?

The D0/D1/D2 protocol is run on DLinear, PatchTST, TiDE, Transformer, and TimeXer. This is a lightweight protocol audit, not a leaderboard SOTA experiment. The strict pass criterion requires lower response RMSE than the fair D1 structural head and no WMAPE degradation greater than 5% relative to D0 for every completed seed. Backbones that satisfy the seed mean but not the strict seed-level rule are reported as boundary cases, not as full passes.

# Results

## Semi-Synthetic Stress Test and Fair-Control Diagnostic

At calibrated confounding strength $`\gamma=0.5`$, D0 and D1 learn the wrong response sign in the hidden-confounder stress test, while D2 recovers the sign and reduces response RMSE. Under shared item controls, the controlled D0 becomes strong, which is expected in this correctly specified linear setting. The orthogonalization claim is therefore made against the fair structural D1 head: D2 reduces RMSE by 58.1% relative to D1, but it is not claimed to beat the controlled linear D0 in this diagnostic.

| Setting                  | Metric                |     D0 |     D1 |     D2 |
|:-------------------------|:----------------------|-------:|-------:|-------:|
| Hidden-confounder stress | Response RMSE         | 0.6226 | 0.5477 | 0.2151 |
|                          | Sign-error rate       |   100% |   100% |     0% |
|                          | WMAPE                 | 0.5284 | 0.5270 | 0.5310 |
| Shared static controls   | Response RMSE         | 0.1927 | 0.5103 | 0.2151 |
|                          | D2 reduction vs D1    |      – |      – |  58.1% |
|                          | D2 change vs D0 WMAPE |      – |      – | -0.06% |

Semi-synthetic results at $`\gamma=0.5`$, averaged over seeds 2021–2023. The first block is a hidden-confounder stress test; the second block gives all arms the same item controls. Lower RMSE and WMAPE are better. {#tab:stress}

## Matched Real-Data Effect Proxies

Table <a href="#tab:real" data-reference-type="ref" data-reference="tab:real">2</a> reports two independent controllable-covariate legs. The Favorita promotion analysis uses 55,063 rows and 1,200 matched units from 2017-06-13 to 2017-07-30. M5 markdown uses 1,392,000 rows and 800 units. In both settings, D2 is closer to the matched within-unit ATT proxy. The paired unit-level deltas are positive with bootstrap confidence intervals excluding zero. These results should be read as evidence of alignment with quasi- experimental proxies, not as proof of randomized causal validity.

| Metric | Favorita promotion | M5 markdown |
|:---|---:|---:|
| Rows / units | 55,063 / 1,200 | 1,392,000 / 800 |
| Treated rate | 0.7338 | 0.1669 |
| Matched ATT proxy | 0.4518 \[0.4268, 0.4776\] | 0.0841 \[0.0581, 0.1095\] |
| D0 NEE | 0.3088 | 0.0775 |
| D2 NEE | 0.1051 | 0.0264 |
| NEE reduction | 66.0% | 65.9% |
| Mean paired NEE delta | 0.0496 \[0.0400, 0.0596\] | 0.0129 \[0.0096, 0.0163\] |
| Units where D2 is closer | 62.1% | 62.4% |
| Wilcoxon $`p`$ | $`8.78\times10^{-15}`$ | $`1.29\times10^{-6}`$ |

Real controllable-covariate validation. ATT and paired-delta intervals are bootstrap 95% intervals over units. NEE is absolute error to the matched ATT proxy. {#tab:real}

The Favorita robustness grid completed four configurations. D2 was closer than D0 in all configurations, with median NEE reduction 60.7%. The strict-overlap configuration still reduced NEE by 63.7%. SNAP is treated as exogenous rather than controllable; it is therefore used only as a non-degradation sanity check. D2 did not materially improve SNAP NEE, as expected for a $`C`$-type covariate.

## Backbone Protocol

The fair-control D0/D1/D2 protocol passes at strict seed level for DLinear, PatchTST, TiDE, and Transformer (Table <a href="#tab:backbones" data-reference-type="ref" data-reference="tab:backbones">3</a>). TimeXer is retained as a boundary case: it improves mean response RMSE versus D1 and satisfies the mean WMAPE criterion, but only 2 of 3 seeds satisfy the strict rule because one seed has a 6.41% WMAPE increase. D0/D1/D2 share item static controls, and D1/D2 share item-specific scalar response capacity. The remaining difference is the orthogonalized residual objective and nuisance training.

| Backbone | D0 RMSE | D1 RMSE | D2 RMSE | D2 vs D1 | Mean WMAPE change | Strict status |
|:---------|--------:|--------:|--------:|---------:|------------------:|:--------------|
| DLinear  |  0.3788 |  0.3863 |  0.0986 |    74.4% |            -1.04% | pass |
| PatchTST |  0.3993 |  0.3990 |  0.3299 |    17.3% |            +2.23% | pass |
| TiDE     |  0.3996 |  0.4005 |  0.2514 |    37.0% |            -3.08% | pass |
| Transformer | 0.6049 | 0.5082 | 0.2174 |    57.1% |           -23.95% | pass |
| TimeXer  |  0.3952 |  0.3977 |  0.2892 |    27.2% |            +2.14% | mean-pass caveat |

Fair-control backbone protocol. Response RMSE is against the semi-synthetic ground-truth response. WMAPE change is D2 relative to D0. The TimeXer caveat reflects a seed-level stability failure, not a missing run. {#tab:backbones}

# Operational Covariate Typing and Refusal Policy

DoCast requires a covariate registry before deployment. A variable is typed as $`A`$ only if the user can set it before the forecast horizon and if historical variation contains comparable action support. Calendar variables and public holiday schedules are $`C`$-type. Lagged outcomes, lagged actions, and static unit metadata enter $`V`$. Variables affected by the proposed action, such as inventory after a promotion, should not be placed in $`C`$ without an explicit state-transition model.

The model should refuse or qualify scenario outputs when support is weak. In the current implementation this is recorded through overlap diagnostics such as treatment frequency, within-unit treated/control counts, and residualized action variance. In an applied higher-education quality-assurance deployment, the same rule would apply to intervention variables such as outreach intensity or scholarship offers: if a proposed policy path has not occurred for similar cohorts or institutions, the forecast should be labeled unsupported rather than reported as an intervention effect.

# Limitations

The real-data targets are observational matched ATT proxies. Promotion timing, stockouts, holiday campaigns, bundled marketing, item selection, or cross-item interference can still bias those proxies. The paired evidence shows alignment with the proxy, not ground-truth causal effects. The semi-synthetic stress test uses a linear treatment basis and injected hidden quality; it is useful for controlled sign and RMSE evaluation but cannot cover all deployment policies. The deep-backbone protocol is intentionally lightweight and is not a leaderboard comparison; TimeXer is a mean-pass boundary case rather than a strict seed-level pass. Finally, scenario validity depends on assumptions A1–A5 and on the quality of the covariate registry. DoCast should be treated as a guarded response-estimation protocol, not as an automatic causal guarantee.

# Conclusion

Scenario forecasting with controllable future covariates is a causal query embedded in a forecasting interface. DoCast makes that distinction explicit by separating controllable actions from exogenous-known covariates and training the action response with an orthogonalized objective. The revised evidence shows where the method helps, where a fair controlled baseline is already strong, and where real-data validation remains quasi-experimental. This narrower claim is the appropriate one: DoCast supports intervention-oriented scenario forecasting under stated assumptions and overlap diagnostics, rather than unrestricted “intervention-valid” forecasting.

# Data and Code Availability

The experiments use public M5 and Favorita data. The local reproduction entry point is `experiments/DoCast/m4_paper_ready/REPRODUCE.md`. The artifact manifest includes the scripts `m0_prior_art.py`, `m1_audit.py`, `m2_docast.py`, `m3_real_data.py`, `m6_backbone_sweep.py`, `m4_paper_ready.py`, and `m5_main_track_audit.py`, together with their JSON summaries.

# Ethics Statement

The reported experiments use public retail benchmark datasets and synthetic stress tests. No human-subject data from higher education institutions are used. For deployment in higher-education quality assurance, scenario forecasts should be reviewed for fairness, privacy, and policy accountability before being used for student-facing decisions.

# Author Contributions

Anonymous authors designed the study, implemented the experiments, interpreted the results, and wrote the manuscript. A detailed CRediT assignment will be provided in the non-anonymous version.

# Conflict of Interest and Funding

The authors declare no competing interests. No external funding is reported for this anonymous submission.

# AI Tool Usage Disclosure

AI assistance was used for code inspection, manuscript editing, and consistency checks across experiment artifacts. The authors remain responsible for the experiments, claims, citations, and final text.
