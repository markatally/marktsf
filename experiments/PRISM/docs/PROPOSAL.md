# PRISM — Predictive Regime-Indexed State-space Mixture

## A Regime-Tracking, Multi-Architecture Framework for Non-Stationary MISO Time-Series Forecasting

> **Version**: v3.4 (2026-06-13) — **M1b complete; gate adjudicated: ETT-only PRISM (§8 pivot clause)**. M1b ran the full amended protocol: DA/IC decision-loss oracle (killed — all 16 settings noise, switch ratio 0.78–0.95), realized-vol target CryptoVol (partial pass — models beat anchors 5–7%, H96 switch_ratio 0.47–0.48 but median_streak=1 everywhere; FS recovers 63–67% causally — real but bursty structure, fails strict condition 2), ETTh1 seed bands (accepted — oracle gap >> noise band), CryptoMISO pool expanded to 6 models (degeneracy unchanged), and FI2010 LOB k∈{10,50,100} CE/F1 oracle (signal-bearing, best k100/F1 oracle gap 12.8% with 33% FS recovery, but switch_ratio 0.67–0.94 so strict condition 2 still fails). **Final gate**: general-benchmark leg (ETTh1) PASS; finance leg no surviving instantiation → ETT-only PRISM is the primary paper. CryptoVol and FI2010 results retained as appendix. v3.3 (2026-06-12) recorded the pre-FI2010 M1b adjudication. v3.2 (2026-06-12) recorded M1a results and amended gate protocol. v3.1 (2026-06-11) incorporated 2026 heterogeneous-MoE wave. v3.0 was the English rewrite. This document is the **single source of truth** for all subsequent coding and experiments.
>
> **Thesis in one sentence**: The deep time-series-forecasting (TSF) benchmark culture implicitly assumes that *"which architecture is best on a dataset" is a static property*. We argue — and will demonstrate empirically — that on real non-stationary series **the optimal inductive bias itself drifts over time with the underlying regime**, and we therefore elevate "which inductive bias to use" from a one-off hyperparameter to a **latent state that must be tracked online**.
>
> **Project fusion vector** (every ingredient is load-bearing, see §5.6): **SSM + MoE + CI/CD + multi-periodicity + multivariate MISO + regime/drift adaptation**.
>
> **Project goal**: a MISO forecasting model that beats current top-venue SOTA under preregistered victory conditions (§7.6), publishable at ICLR / ICML / NeurIPS (method + analysis) or KDD / CIKM / AAAI (finance-strengthened version).

---

## 0. TL;DR

- **Problem**: asymmetric **MISO** forecasting — many input covariates, one target series (§1). This is the deployment-realistic setting (quantitative finance, retail demand), as opposed to symmetric multivariate forecasting where every channel is predicted.
- **Status-quo critique**: five unexamined assumptions in the literature (§2), all sharing one root: *structural choices are treated as static*.
- **Core hypothesis H0 (falsifiable)**: real series are governed by a small set of recurring but non-stationary **regimes**; each regime has a characteristic spectro-temporal signature, a characteristic *optimal inductive bias*, and a characteristic covariate→target coupling strength. Under distribution shift, forecast error is dominated by *regime misidentification + bias mismatch*, not by backbone capacity.
- **Gate experiment (cheap, run first)**: the **Oracle Drift Study** (§7.1) — measure whether the per-window best architecture actually switches over time. If it does (especially on financial data), the project is greenlit and the oracle-vs-best-single-model gap quantifies our headroom. **Status (M1b complete, 2026-06-13)**: ETT leg PASS (28–33% oracle gap, switch ratio 0.19–0.25, FS recovery 58–66%); all finance legs, including FI2010 LOB, fail amended conditions → **ETT-only PRISM** (§8 pivot). See §7.1.1 and REPORT.md.
- **Method (PRISM, §5)**: spectro-temporal **regime descriptors** → a lightweight **SSM regime filter** with dual timescales (routing-as-state-estimation) → sparse routing over a **heterogeneous, multi-scale expert library** (linear / frequency / patch-attention / covariate cross-attention / SSM) → a regime-gated **covariate-coupling gate β ∈ [0,1]** (continuous CI↔CD) → frequency-domain + decision-aware losses → **test-time drift adaptation at the routing level**.
- **Theory (§4)**: a neural, multi-architecture generalization of Markov-switching / switching state-space models; online-learning framing via shifting regret, with **Fixed-Share over frozen experts** as an honest, theoretically grounded lower-bound baseline.
- **Battlefields (§7.2)**: S1 finance MISO **primary** (FI2010 LOB, Crypto, CN-Future, equity indices); S2 retail MISO secondary (M5, Favorita with known-future covariates); S0 symmetric benchmarks as a generality check only.
- **Rigor**: preregistered victory conditions, Diebold-Mariano + Wilcoxon tests with Benjamini-Hochberg FDR control, purge/embargo splits, per-task lookback tuning, drift-stratified evaluation, accuracy-efficiency Pareto reporting.

---

## 1. Problem Statement: Asymmetric MISO Forecasting

### 1.1 Sample contract

At time `t`, with lookback `L` and horizon `H`, one sample is:

| Tensor | Shape | Meaning |
|---|---|---|
| `x_target` | `[L]` | target history |
| `x_past_cov` | `[L, C_p]` | **past-only** covariates (e.g., volume, realized volatility, order-flow proxies, open interest) |
| `x_known_past` | `[L, C_f]` | history segment of **known-future** covariates (e.g., calendar, promotions) |
| `x_known_fut` | `[H, C_f]` | future segment of known-future covariates (exogenous new information; legal to feed forward) |
| `static` | `[C_s]` | static covariates (asset / store / category id; optional) |
| `y` | `[H]` | forecast target |

Forecast: `ŷ_{1:H} = F(x_target, x_past_cov, x_known_*, static)`.

**Financial specialization**: the target is the forward N-step log return `r_{t+k} = log(p_{t+k}) − log(p_{t+k−1})` (or cumulative `log(p_{t+H}/p_t)`), computed at the data layer.

**Core asymmetry**: covariates are *predictors*, not prediction targets. This distinguishes MISO from symmetric multivariate TSF (where, e.g., iTransformer predicts every channel) and is the deployment-realistic setting in quant finance and supply chains.

### 1.2 Why MISO-first (not an afterthought)

1. **Deployment realism**: practitioners forecast *a* target given everything else; almost no one needs all 862 Traffic channels predicted jointly.
2. **Headroom honesty**: symmetric long-horizon benchmarks (ETT/Weather/Traffic MSE) are saturated — recent top-venue gains are at noise level (the DLinear lesson; the TSFM-observability critique). Claiming "beat SOTA" there is neither achievable nor credible. The covariate-aware MISO battlefield has real headroom and few specialized competitors (TimeXer, TFT, TiDE-cov, NBEATSx).
3. **Benchmark-definition opportunity**: no standardized MISO leaderboard exists. We construct one by MISO-izing standard backbones plus covariate-aware baselines under a leak-proof protocol, and release it with the paper (cf. Time-IMM's dataset-paper playbook). Risk and mitigation in §8.
4. **Generality is still checked**: any symmetric dataset is MISO-ized by designating a target column (ETT→OT, etc.); we additionally report classic symmetric results so reviewers can compare with the literature (S0, §7.2).

---

## 2. Status-Quo Critique: Five Assumptions We Question

| # | Implicit mainstream assumption | Evidence against it | PRISM's position |
|---|---|---|---|
| Q1 | *Which architecture is best on a dataset* is a *static*, one-shot property | DLinear overturned the Transformer wave; the channel-bias study shows CI's wins are an artifact of weak inter-channel correlation; backbones trade wins across datasets | The optimal inductive bias is a **time-varying latent state** to be tracked, not a constant to be benchmarked once |
| Q2 | Channel-Independent (CI) vs Channel-Dependent (CD) is a **global binary choice** (partial-channel masks are still static) | CD wins only under strong cross-channel dependence; financial correlations regime-shift violently (risk-on/off) | A **regime-gated continuous gate β ∈ [0,1]** interpolating CI↔CD online |
| Q3 | Non-stationarity ≈ first/second-moment drift; **RevIN suffices** | Higher-order non-stationarity (spectral drift, correlation drift, regime switching) is untouched by RevIN; TimeBridge/ShifTS/Proceed all patch pieces of this | Model spectral drift and correlation stability **explicitly** as regime signals; adapt at test time |
| Q4 | Frequency-domain methods can use a **fixed global DFT basis** (within-window spectral stationarity) | Real signals are spectrally non-stationary within a window, most severely at regime boundaries; SEMPO shows low-energy bands carry ignored information | Use **energy-aware multi-band features + intra-window spectral drift** as regime evidence, not just as input features |
| Q5 | **MSE/MAE on ETT** suffices to certify SOTA | Many published "gains" are within noise; in finance, MSE-optimal ≠ decision-optimal | Multi-seed + significance tests + FDR control + oracle upper bounds; decision metrics (IC/DA/backtest) for finance |

> All five share one root: **treating structural choice as static**. PRISM's entire novelty is making it *dynamic, trackable, and online-adaptable* — precisely the one step that xCPD (instantaneous per-patch routing, no temporal memory), ShifTS (seeks *invariant* patterns), the channel-bias study (per-*dataset* static conditioning), and Dynamic TMoE (ICML'26; evolves heterogeneous experts with drift *during training* but freezes the routing policy at test time) each stop short of.

---

## 3. Hypotheses (all falsifiable, all preregistered)

| ID | Statement | Test | Falsified if |
|---|---|---|---|
| **H0** | Series are governed by a few recurring, non-stationary regimes; each regime determines (spectro-temporal signature, optimal inductive bias, covariate-coupling strength). Forecast error under shift is dominated by regime misidentification + bias mismatch | Oracle Drift Study (§7.1) + synthetic recovery (§7.7) | Per-window argmin architecture is (statistically) constant on financial data |
| **H1** | The per-window best architecture / channel strategy is **non-constant over time** on a single dataset; switching is frequent on financial data, rare on quasi-stationary benchmarks (ETT) | Oracle Drift Study switch-rate, finance vs ETT | Switch rates indistinguishable between finance and ETT |
| **H2** | Under low SNR and drift, **switching inductive bias** (linear↔freq↔patch↔covariate-attn↔SSM) yields larger gains than adding capacity or homogeneous MoE experts inside one backbone | P1 (homogeneous-MoE) vs P3 (heterogeneous), matched parameter budget | Homogeneous MoE matches heterogeneous library at equal params |
| **H3** | Covariate→target coupling strength is regime-dependent; a **dynamic β** beats any static CI / CD / fixed partial mask | P2 (static β) vs P3 (dynamic β); vs xCPD, Partial-Channel | Dynamic β shows no significant gain over the best static setting |
| **H4** | Under concept drift, **re-assigning regimes→experts at the routing level** is more stable and cheaper than fine-tuning backbone parameters | P3 (offline) vs P4 (routing-level TTA); vs Proceed / DynaTTA (parameter-level) | Parameter-level adaptation dominates routing-level adaptation |

---

## 4. Theoretical Framing

### 4.1 Piecewise-stationary view and risk decomposition

Assume a piecewise-stationary data-generating process with latent regime `z_t ∈ {1..K}`, regime-conditional distributions `D_k`, and regime-conditional Bayes predictors `f*_k`. For a routed mixture `ŷ = Σ_k w_k(x) E_k(x)`:

> **Proposition 1 (to be formalized in the paper).** The excess risk of a routed mixture decomposes into (i) **regime-identification error** (mass placed on wrong experts), (ii) **bias-mismatch error** (no expert's hypothesis class approximates `f*_k` well), and (iii) **within-regime estimation error**. Moreover, if for at least two regimes `j ≠ k` the best architecture class differs with approximation-gap margin Δ, then *any single fixed architecture* incurs occupancy-weighted excess risk ≥ Ω(Δ), while an oracle-routed heterogeneous mixture does not.

This converts the marketing claim "ensembles help" into a falsifiable structural claim: **the gap between the best single model and the regime-oracle (measured in §7.1) is exactly the term a regime tracker can recover.**

### 4.2 Online-learning anchor: shifting regret and Fixed-Share

"Track the drifting best expert" is a classical online-learning problem: **Fixed-Share** (Herbster & Warmuth, 1998) guarantees shifting-regret bounds against the best *sequence* of experts with `m` switches. We use this two ways:

1. **Honest lower-bound baseline (P-FS)**: independently trained, frozen versions of our five backbones, combined online by Fixed-Share/Hedge on rolling validation loss. **If PRISM's learned regime routing cannot beat this learning-free combiner, the routing has learned nothing** — we preregister this as a kill criterion for the routing component.
2. **Framing**: PRISM is an *amortized, feature-based* approximation to shifting-regret-optimal expert tracking — the router predicts the switch *before* losses reveal it, using regime descriptors; Fixed-Share can only react *after*.

### 4.3 Classical lineage: neural switching state-space mixture

Regime-switching forecasting has deep statistical roots: Markov-switching autoregressions (Hamilton, 1989) and switching state-space models (Ghahramani & Hinton, 2000). PRISM is their neural, high-dimensional, **multi-architecture** descendant: the switching variable selects not parameters of one model family but **entire inductive-bias families**, and inference is amortized by an SSM filter instead of EM. This lineage gives the design principled legitimacy and a clean prior-work narrative.

---

## 5. Method: PRISM

### 5.1 Dataflow overview

```
x_target[L], x_past_cov[L,C_p], x_known_fut[H,C_f]
        │
        ▼
   RevIN (per-instance normalization; handles 1st/2nd-moment drift only)
        │
        ├────────────► Regime descriptors s_t   (cheap, interpretable; §5.2-2)
        │                      │
        │                      ▼
        │              SSM Regime Filter (§5.2-3, dual timescale)
        │                r_slow (per window)  ──► β(r_slow) ∈ [0,1]  covariate-coupling gate
        │                r_fast (per patch)          │
        │                      │                     │
        │                      ▼                     │
        │          Router g(r_slow, r_fast)          │
        │          (top-k sparse, competence prior)  │
        ▼                      │                     │
   Heterogeneous multi-scale expert library (all MISO-native, §5.2-4)
   E_lin · E_freq(lo/hi cutoff) · E_patch(short/long) · E_chan(β-gated covariate cross-attn) [· E_ssm]
        │
        ▼
   ŷ[H] = Σ_k w_k(r_t) · E_k(x)          (output-space mixture)

   Loss:      L_time + λ_f·L_freq + λ_d·L_decision + λ_b·L_balance        (§5.2-7)
   Test time: descriptor-drift / regime-novelty monitor → update RevIN stats,
              re-estimate routing prior, optional low-rank expert deltas,
              spawn new regime prototype                                   (§5.2-8)
```

### 5.2 Components (with design lineage)

**1. RevIN normalization.** Reversible instance normalization (RLinear / Non-stationary Transformer lineage). Hygiene baseline for moment drift; explicitly *not* expected to handle higher-order non-stationarity (that is the regime filter's job).

**2. Spectro-temporal regime descriptors `s_t`** — a cheap, interpretable feature vector per window/patch:
- energy-aware multi-band power spectrum (rFFT; includes **low-energy bands**, per SEMPO's finding);
- spectral entropy (high = broadband/noise-dominated → favors patch-attention; low = narrowband/periodic → favors frequency/linear experts);
- **intra-window spectral drift** (spectral distance between first and second half-window = non-stationarity index);
- trend/seasonal decomposition strength (Autoformer-style);
- **cross-channel correlation stability** (correlation-matrix distance between half-windows → drives β).

Lineage: CATCH (frequency patching), SEMPO (energy awareness), channel-bias study (statistics-guided CI/CD choice).

**3. SSM regime filter — the core mechanism (routing-as-state-estimation).**
A lightweight state-space model treats the descriptor sequence `s_1..s_t` as observations and infers a latent regime state online, at **two timescales**:
- `r_slow` — per-window update; captures trend/seasonal/macro regimes; drives β and the linear/frequency experts;
- `r_fast` — per-patch update; captures local volatility / burst regimes; drives the patch/SSM experts.

Sticky transitions, changepoint resets, and novelty detection (for spawning new regime prototypes at test time) are all natural operations *inside* the filter, not bolted-on regularizers. This is where the **SSM ingredient is load-bearing**: the router *is* a state estimator — the architectural difference from xCPD's and MoHETS's memoryless per-patch routing. Dynamic TMoE (ICML'26) also gives its router temporal memory (recurrent states + anomaly repository), so memory alone is no longer the differentiator (§6.2): PRISM's filter is an explicit *state-estimation* formulation with regime semantics, sticky/changepoint dynamics, a shifting-regret yardstick (§4.2), and — unlike Dynamic TMoE's training-time-frozen policy — it keeps adapting at test time (§5.2-8). Classical anchor: §4.3.

**4. Heterogeneous, multi-scale, MISO-native expert library `{E_k}`.**
Each expert embodies a *genuinely different inductive bias*, kept deliberately compact (diversity over capacity; per-expert parameter cap ≤ 1× iTransformer-base, total ≤ 2×):

| Expert | Backbone lineage | Inductive bias | Scale variants |
|---|---|---|---|
| `E_lin` | RLinear (+RevIN) | trend / robustness / low variance | — |
| `E_freq` | FITS-style complex-frequency linear | periodicity / seasonality / compactness | 2 cutoff frequencies (long vs short periods) |
| `E_patch` | PatchTST (target-history only) | local nonlinear motifs | 2 patch lengths (short vs long) |
| `E_chan` | iTransformer-style **covariate cross-attention**: target token queries, covariate tokens as keys/values | cross-variable driving | β-gated |
| `E_ssm` (optional) | Time-SSM | long-range dependency | — |

All experts are MISO-native: `E_lin/E_freq/E_patch` consume the target history only (the CI extreme); `E_chan` is the covariate-coupling extreme (CD). The library spans the CI↔CD axis *by construction*. The two-scale instantiation of `E_freq`/`E_patch` makes **multi-periodicity an explicit, ablatable mechanism** (consuming the Multi-period-Learning / SparseTSF insight) rather than an implicit feature.

**Contrast with homogeneous MoE** (Time Tracker, FinCast, SEMPO): their experts are FFN/prompt clones inside one backbone; ours are distinct architecture families. H2 tests exactly this distinction at matched parameter budget. *Note (2026 wave)*: MoHETS, FAME, and Dynamic TMoE have since adopted heterogeneous experts too, so heterogeneity per se is no longer a novelty claim (§6.2) — but H2 remains the *controlled* test of it, and our library spans full architecture families covering the CI↔CD extremes, rather than conv/Fourier modules inside one Transformer (MoHETS) or a per-series static pool (FAME).

**5. Regime→bias router `g(r_slow, r_fast)`.** Top-k sparse mixture weights over experts, initialized/biased by an interpretable **competence prior** `P(expert | s_t)`: high correlation-stability → `E_chan`; low-energy + low-entropy → `E_lin`/`E_freq`; high-entropy short-range structure → `E_patch`; long-memory signatures → `E_ssm`. The prior is the learnable generalization of the channel-bias study's "use dataset statistics to choose CI/CD", lifted from per-dataset to per-window.

**6. Regime-gated covariate coupling `β(r_slow) ∈ [0,1]`.** Driven primarily by correlation stability; continuously interpolates CI (β→0: covariates ignored) ↔ CD (β→1: full covariate cross-attention), modulating `E_chan`'s contribution and its attention temperature. **MISO semantics**: β literally reads as *"how strongly are covariates driving the target right now"* — in finance, whether the market is in a coupled (risk-on/off) or idiosyncratic regime. This is the dynamic, online version of the per-dataset static CI/CD choice; finer-grained in *time* than xCPD is in *patches*, because it carries temporal persistence.

**7. Loss.**
`L = L_time(Huber/MSE) + λ_f·L_freq + λ_d·L_decision + λ_b·L_balance`
- `L_freq`: FreDF frequency-domain loss (corrects label-autocorrelation bias of time-domain MSE);
- `L_decision` (finance only): `1 − IC` / directional-accuracy surrogate (MSE-optimal ≠ decision-optimal);
- `L_balance`: router load-balancing to prevent expert collapse.

**8. Test-time drift adaptation — at the routing level (H4).** Monitor descriptor drift and regime novelty; on trigger: (a) update RevIN statistics (DynaTTA-style); (b) re-estimate the routing competence prior on a rolling window; (c) optionally apply low-rank expert deltas or **spawn a new regime prototype**. Contrast: Proceed/DynaTTA adapt *parameters/statistics of one backbone*; PRISM re-assigns *which inductive bias is in charge* — cheaper, more stable, and directly interpretable.

**9. Staged training (stability by construction).**
- **Stage A**: pretrain each expert independently (frozen afterwards) — eliminates router-expert credit-assignment pathologies;
- **Stage B**: train the SSM filter + router on frozen experts (the routing problem is now supervised by realized per-expert losses);
- **Stage C**: joint fine-tuning at low learning rate (optional; ablated).

This addresses the classic instability of heterogeneous MoE training and gives clean intermediate diagnostics (Stage-B routing quality is directly measurable against the per-window oracle).

### 5.3 Minimal-PRISM (default deliverable)

The headline model is deliberately small: **3 experts (`E_lin`, `E_freq`, `E_patch`) + SSM regime filter + β-gated `E_chan`**, top-k = 2 at inference. The full library and all optional parts (`E_ssm`, low-rank TTA deltas, regime spawning) live in ablations/appendix. This is the primary defense against "Frankenstein system" reviews: the narrative is *one thesis* (the optimal bias drifts) *and the minimal machinery to exploit it*.

### 5.4 Degeneration table (= built-in ablation matrix)

| Config | Regimes K | Experts | β | Drift loop | L_freq | Equivalent to |
|---|---|---|---|---|---|---|
| P0 | 1 | single `E_chan` | fixed = 1 | ✗ | ✗ | iTransformer-MISO lower bound |
| P0′ | 1 | single `E_lin` | fixed = 0 | ✗ | ✗ | RLinear/DLinear lower bound |
| P1 | 1 | homogeneous FFN ×k (param-matched) | learned | ✗ | ✗ | vanilla MoE (≈ Time Tracker, simplified) |
| P-FS | — | 5 frozen backbones | — | Fixed-Share | ✗ | learning-free shifting-regret baseline (§4.2) |
| P2 | K | heterogeneous, single-scale | fixed | ✗ | ✗ | static regime mixture |
| P2.5 | K | heterogeneous, **multi-scale** | fixed | ✗ | ✗ | + multi-periodicity |
| P3 | K | heterogeneous, multi-scale | **dynamic β** | ✗ | ✓ | offline PRISM |
| **P4 = PRISM** | K | heterogeneous, multi-scale | dynamic β | **✓** | ✓ | full method |
| P-fin | K | `E_chan` → per-band driver→target attention | dynamic | ✓ | ✓ | **= legacy SPECTRE (v1) as one configuration** |
| P-router | K | as P3 | dynamic | ✗ | ✓ | router ablation: SSM ↔ GRU ↔ MLP ↔ Fixed-Share |

Hypothesis mapping: P1 vs P3 → **H2**; P2 vs P3 → **H3**; P3 vs P4 → **H4**; P2 vs P2.5 → multi-periodicity; P-FS vs P4 → routing kill criterion; P-router → is the SSM filter (vs memoryless/recurrent alternatives) actually necessary.

### 5.5 Complexity and efficiency

Top-k = 2 sparse inference; shared input embedding across experts; per-expert and total parameter caps (§5.2-4). We commit to reporting an **accuracy–efficiency Pareto** (params, FLOPs, latency vs metric) for PRISM and all baselines — gains that vanish under parameter matching are not gains (Q5).

### 5.6 Fusion-vector accountability map

| Ingredient | Where it is load-bearing | Isolating ablation |
|---|---|---|
| **SSM** | the regime filter *is* an SSM (routing-as-state-estimation, §5.2-3) | P-router (SSM vs GRU vs MLP vs Fixed-Share) |
| **MoE** | heterogeneous expert library + sparse router (§5.2-4/5) | P1 vs P3 (H2) |
| **CI/CD** | continuous β gate over covariate coupling (§5.2-6) | P2 vs P3 (H3) |
| **Multi-periodicity** | dual-timescale regime states + two-scale expert instantiation | P2 vs P2.5 |
| **Multivariate MISO** | the problem formulation itself (§1) + MISO-native experts | S0 symmetric check vs S1/S2 |
| **Regime/drift adaptation** | routing-level test-time adaptation (§5.2-8) | P3 vs P4 (H4) |

Every ingredient maps to exactly one mechanism and one ablation — no decorative components.

---

## 6. Positioning and Novelty

### 6.1 Curated literature (from the 58-paper index in `docs/PAPER.md`)

**Tier S — paradigm-defining (battlefield definition):**

| Paper | Venue | What PRISM takes |
|---|---|---|
| DLinear / LTSF-Linear | AAAI'23 | `E_lin`'s case; the skepticism discipline (significance tests, oracle bounds) |
| PatchTST | ICLR'23 | `E_patch`; patching as default tokenization; the CI paradigm |
| iTransformer | ICLR'24 | `E_chan`'s variate-token cross-attention; the CD paradigm |
| FEDformer / Autoformer | ICML'22 / NeurIPS'21 | decomposition priors feeding the regime descriptors |

**Tier A — closest frontier (direct competitors / building blocks):**

| Paper | Venue | Role |
|---|---|---|
| **xCPD** (graph-spectral channel-patch routing) | ICLR'26 | **nearest neighbor**: per-patch, input-instantaneous spectral routing with homogeneous band experts; *no temporal regime memory, no drift adaptation*. PRISM's key differentiation anchor + head-to-head baseline |
| **ShifTS** (concept-drift mitigation) | ICLR'26 | nearest neighbor on the drift axis: seeks *invariant* patterns; PRISM argues the optimal bias is precisely *not* invariant. Complementary stance; core baseline |
| **TimeXer** (exogenous-variable Transformer) | NeurIPS'24 | **the most direct MISO competitor** — endogenous patch tokens + exogenous variate tokens with cross-attention. Must-beat baseline. *(Action item: PDF not yet in `paper/`; acquire along with TFT and NBEATSx.)* |
| Channel-bias study (CI/CD × lookback) | arXiv'25 | empirical foundation: CI/CD optimality is dataset-conditional; we lift it from per-dataset (static) to per-window (dynamic); we adopt its per-task lookback-tuning protocol |
| FITS | ICLR'24 | `E_freq` design |
| FreDF | ICLR'25 | frequency-domain loss `L_freq` |
| TimeBridge | ICML'25 | short-term vs long-term non-stationarity treatment; CSI/SP500 evaluation precedent; strong baseline |
| SEMPO | NeurIPS'25 | energy-aware spectral descriptors (incl. low-energy bands); routing contrast |

**Tier A′ — the 2026 heterogeneous-MoE wave (index #54–58):** five 2026 papers independently converged on heterogeneous experts and structure-aware routing. This both *validates the problem* (the field now agrees one architecture does not fit all regimes) and *retires two former PRISM differentiators* — "heterogeneous expert library" and "temporal memory in the router" are now table stakes. The surviving novelty budget is restated after the §6.2 table.

| Paper | Status | Role |
|---|---|---|
| **Dynamic TMoE** (drift-aware dynamic MoE) | **ICML'26** | **closest competitor overall**: MMD-based drift detection → dynamic *spawning/pruning of heterogeneous experts* + a *recurrent temporal-memory router* (recurrent states + anomaly repository). All adaptation happens **at training time**; explicitly *no test-time updates* — the exact regime H4 targets. Public code ([github.com/andone-07/Dynamic-TMoE](https://github.com/andone-07/Dynamic-TMoE)) → **implemented head-to-head baseline** (§7.3) |
| **MoHETS** (mixture of heterogeneous experts) | arXiv'26 (under review) | heterogeneous **conv + Fourier experts inside one encoder-only Transformer**, with covariate cross-attention; routing is **per-patch and memoryless** (xCPD-style instantaneous) — no regime state, no drift loop |
| **AME-TS** (anchored MoE foundation model) | arXiv'26 (ICML'26 FMSD-workshop reviews on OpenReview; acceptance unconfirmed) | **series-level structural descriptors** (forecastability / seasonality / trend / sparsity) → **soft structural prior over experts** guiding token routing — the closest overlap with PRISM's competence prior (§5.2-5), but *series-level, offline, static in time*, inside a foundation model |
| **FAME** (forecastability-aware MoE) | arXiv'26 | per-series **forecastability fingerprint** → cost-aware sparse routing over a **heterogeneous pool incl. non-neural LightGBM**; expert suitability mined from validation performance; static per-series assignment, no temporal regime tracking |
| **DeRegiME** (deep regime mixtures) | arXiv'26 | sparse-variational-GP **regimes of residual *uncertainty*** (nonstationary regime-mixing kernel, Student-t likelihood) — regime structure lives in the *noise model*, not in architecture routing; complementary probabilistic stance; optional uncertainty baseline (§7.3) |

*Venue status verified 2026-06-11 (arXiv comments + OpenReview): only Dynamic TMoE has a confirmed top-venue acceptance; the other four are cited as preprints.*

**Tier A-finance (S1 baselines):** Kronos (AAAI'26, K-line tokenization FM), FinCast (CIKM'25, MoE + PQ-loss FM), Multi-period Learning (KDD'25, multi-period financial structure).

**Tier B — plug-in mechanisms and protocols:** Non-stationary Transformer (de-stationarized statistics → descriptors), Proceed (proactive drift adaptation → contrast at parameter level), DynaTTA (shift-aware TTA protocol), Time-SSM (`E_ssm`), CATCH (frequency patching for descriptors), Time Tracker (homogeneous-MoE contrast), TSFM-observability critique (honest-evaluation argumentation).

**Classical anchors (outside the index, to be cited):** Hamilton 1989 (Markov-switching), Ghahramani & Hinton 2000 (switching SSM), Herbster & Warmuth 1998 (Fixed-Share / tracking the best expert), Diebold & Mariano 1995 (forecast-comparison test); covariate-aware baselines TFT (Lim et al.) and NBEATSx (Olivares et al.).

### 6.2 Head-to-head novelty table

Transposed (methods as rows) after the 2026 wave doubled the competitor field:

| Method | Routing signal | Heterogeneous experts | Router temporal memory | Drift adaptation | Covariates / CI↔CD | Optimal-bias-drift thesis | Theory hook |
|---|---|---|---|---|---|---|---|
| xCPD (ICLR'26) | instantaneous per-patch spectrum | ✗ (band experts) | ✗ | ✗ | per-patch routing | ✗ | ✗ |
| Time Tracker | frequency graph | ✗ (FFN clones) | ✗ | ✗ | — | ✗ | ✗ |
| SEMPO (NeurIPS'25) | token→prompt | ✗ (prompts) | ✗ | ✗ | — | ✗ | ✗ |
| ShifTS (ICLR'26) | — (single model) | — | — | invariant patterns (train-time) | — | ✗ (seeks invariance) | invariance |
| TimeXer (NeurIPS'24) | — (single model) | — | — | ✗ | endo/exo split (static fusion) | ✗ | ✗ |
| Channel-bias study (arXiv'25) | offline statistics | — | — | ✗ | per-dataset (static) | ✗ (across datasets only) | ✗ |
| MoHETS (arXiv'26) | per-patch, memoryless | ✓ (conv + Fourier modules in one Transformer) | ✗ | ✗ | covariate cross-attn (static) | ✗ | ✗ |
| AME-TS (arXiv'26) | series-level structural descriptors → soft prior | ✗ (FM experts) | ✗ | ✗ | — | ✗ (static per series) | ✗ |
| FAME (arXiv'26) | per-series forecastability fingerprint | ✓ (pool incl. LightGBM) | ✗ | ✗ | — | ✗ (static per series) | ✗ |
| **Dynamic TMoE (ICML'26)** | MMD drift signal + recurrent states | ✓ (dynamically spawned/pruned) | ✓ (recurrent states + anomaly repository) | **training-time only; explicitly no TTA** | ✗ | assumed, never measured | ✗ |
| DeRegiME (arXiv'26) | GP regime-mixing kernel (over residuals) | ✗ (sub-kernels, not architectures) | smooth GP posterior | ✗ | ✗ | ✗ (regimes of noise, not bias) | Bayesian / GP |
| **PRISM** | **temporally persistent regime state (dual-timescale SSM filter)** | **✓ full architecture families spanning CI↔CD extremes** | **✓ explicit state estimation (sticky transitions, changepoints, novelty)** | **routing-level TTA + regime spawning** | **regime-gated continuous β (online)** | **core thesis + Oracle Drift Study measurement** | **switching-SSM lineage + shifting regret (Fixed-Share kill criterion)** |

**Post-2026-wave novelty budget.** Heterogeneous experts (MoHETS, FAME, Dynamic TMoE) and temporal memory in the router (Dynamic TMoE) are **no longer PRISM differentiators** — they are table stakes that the field converged on independently. PRISM's claims now rest on five legs that no method above has:

1. **The phenomenon claim**: the Oracle Drift Study (§7.1) *measures* that the per-window optimal inductive bias drifts over time (H0/H1). Every 2026 competitor *assumes* some form of this premise; none measures it. This is the claim to stake first (§8).
2. **Routing-as-state-estimation with a theory anchor**: shifting-regret framing, the Fixed-Share lower-bound baseline, and a preregistered kill criterion (§4.2). Dynamic TMoE's recurrent router is a mechanism without a yardstick — it cannot say what its memory is worth.
3. **Routing-level test-time adaptation** (+ regime spawning, §5.2-8): Dynamic TMoE adapts only during training and freezes its policy at test time; H4 (P3 vs P4, and Dynamic TMoE vs PRISM) tests exactly this gap.
4. **Asymmetric MISO with a dynamic covariate-coupling gate β** (§5.2-6): the 2026 wave is symmetric-multivariate or covariate-static (MoHETS's cross-attention is always-on); none treats covariate coupling strength as a regime-dependent online quantity.
5. **The finance decision-metric battlefield** (§7.2): IC/RankIC/DA/backtest under purge-embargo with DM + FDR; none of the five evaluates beyond MSE/MAE-style accuracy.

### 6.3 Contribution statement (three sentences)

1. **An overlooked empirical phenomenon** (H1 + Oracle Drift Study): on a single dataset's timeline, the best inductive bias *switches* — the static benchmarking literature is structurally blind to this.
2. **A principled mechanism**: an SSM regime filter routing a heterogeneous, multi-scale, MISO-native expert library, with a regime-gated covariate-coupling gate and routing-level drift adaptation — every part interpretable (one can read off *which regime prefers which bias*).
3. **A scientific reframing**: the "Transformer vs Linear vs Frequency" and "CI vs CD" debates are answered not with a winner but with a measurable *conditional* answer — *when, under which regime, which bias wins* — grounded in switching-model and online-learning theory.

> Even if aggregate SOTA margins on saturated benchmarks land within noise (the DLinear risk), contributions 1 and 3 plus finance decision-metric gains stand on their own — the proposal is robust to its own null results (§8).

### 6.4 Relation to legacy SPECTRE (v1)

SPECTRE (single-backbone asymmetric MISO with per-frequency-band driver→target attention and auxiliary covariate-forecasting regularization) is **absorbed, not discarded**: it is configuration **P-fin** in §5.4 (its band-resolved coupling becomes a specialization of `E_chan`; its auxiliary head remains an optional regularizer; its band decomposition is generalized into the regime descriptors). The narrow-deep fallback (§8) reuses it directly. Both share the `spectre/` harness and sample contract.

---

## 7. Experimental Design

### 7.1 Gate experiment: the Oracle Drift Study (run first; greenlight criterion)

- **Setup**: backbones {DLinear, FITS, PatchTST, iTransformer, Time-SSM}, each trained once per dataset in the **financial MISO setting** (plus ETT for contrast); CI and CD variants where applicable.
- **Protocol**: sliding evaluation windows (stride = H); record the per-window argmin-loss backbone → the "best-architecture trajectory". The **regime oracle** = per-window best choice (hindsight); compare with the best single model overall.
- **Outputs**: (i) best-architecture-over-time trajectory plots, finance vs ETT; (ii) **oracle − best-single gap** = PRISM's recoverable headroom (Proposition 1); (iii) correlation between switch points and regime descriptors (spectral drift, correlation stability).
- **Decision rule (preregistered)**: if the argmin is significantly non-constant on financial data (switch rate test vs a constant-choice null) **and** the oracle gap exceeds the best single model's seed-level noise band → greenlight full PRISM. Otherwise → pivot to the conditional-analysis short paper (§8), at ~2 weeks' sunk cost.
- This study doubles as Figure 1 of the eventual paper regardless of outcome.

### 7.1.1 M1a outcome and amended gate (v3.2, 2026-06-12)

**What was run** (artifacts: `experiments/PRISM/oracle_drift/`; finance producer: `experiments/PRISM/produce_predictions.py` + `experiments/PRISM/data/crypto_dataset.py`; single seed 2021):

- **ETTh1** (contrast): {DLinear, PatchTST, TiDE, TimeXer}, L=96, H∈{96, 192, 336, 720}, scored on OT only (`--target-channel -1`), from pre-existing TSLib artifacts.
- **Crypto** (finance, `ftM`): {DLinear, PatchTST, iTransformer, TimesNet}, trained symmetric-multivariate on 14 hourly close-log-return channels, scored on BTCUSDT; H∈{24, 48, 96, 168}.
- **CryptoMISO** (finance, `ftMS`): {DLinear, PatchTST, iTransformer, TimeMixer}, trained MISO on 28 features (14 close-ret + 14 vol-ret), target BTCUSDT hourly close log-return; same horizons.
- Deviations from the §7.1 spec, to fix in M1b: FITS and Time-SSM not yet in the pool; windows are stride-1 (TSLib convention) rather than stride-H; one seed only.

**Results** (per-window target-channel MSE; *noise null* = switch rate of an IID argmin with the observed win fractions; *anchor* = unconditional-mean/zero predictor MSE from `true.npy`; negative "vs anchor" = model beats the anchor):

| Setting | Best single | Oracle gap | Switch rate obs / null | Median · max streak | Best single vs anchor |
|---|---|---:|---:|---:|---:|
| ETTh1 H96 | PatchTST .0554 | **25.3%** | .123 / .741 (**0.17×**) | 3 · 134 | **−97.1%** |
| ETTh1 H192 | TimeXer .0696 | **23.6%** | .092 / .693 (**0.13×**) | 4 · 195 | −96.4% |
| ETTh1 H336 | TimeXer .0831 | **22.8%** | .100 / .673 (**0.15×**) | 3 · 197 | −95.8% |
| ETTh1 H720 | TimeXer .0888 | 6.7% | .108 / .538 (0.20×) | 3 · 145 | −95.6% |
| CryptoMISO H24 | DLinear .4989 | 3.4% | .592 / .739 (**0.80×**) | 1 · 22 | **+1.05%** |
| CryptoMISO H48 | PatchTST .4939 | 2.2% | .502 / .733 (0.68×) | 1 · 35 | +0.90% |
| CryptoMISO H96 | PatchTST .4936 | 1.3% | .552 / .737 (0.75×) | 1 · 54 | +0.69% |
| CryptoMISO H168 | TimeMixer .4969 | 1.2% | .557 / .749 (0.74×) | 1 · 34 | +0.78% |
| Crypto (ftM) H24–168 | DLinear ≈.49 | 1.0–2.8% | .51–.55 (0.71–0.75×) | 1 · ≤27 | +0.1…+0.8% |

**Reading.**

1. **The phenomenon is real and large on ETTh1.** Architecture dominance is strongly persistent — switch rate 0.13–0.20× the noise null, dominance streaks of 134–197 consecutive windows (≈ 5–8 days) — and the hindsight oracle headroom is 23–25% at H96–336. This is exactly the structure H1 posits, found on the dataset we expected to be the *negative* contrast. H1's auxiliary assumption ("switching rare on quasi-stationary benchmarks") is rejected; the general-benchmark leg is hereby **promoted from generality check to first-class evidence** for the phenomenon claim (novelty leg #1, §6.2). This also de-risks the paper: contribution 1 no longer depends on finance alone.
2. **The finance leg as instantiated is metric-degenerate, not informative.** On hourly BTC log returns under MSE, every backbone scores **at or below the zero-return predictor** (+0.1% to +1.05% above the anchor); the per-window argmin therefore selects among statistically tied models — median streak 1, switch rate 0.7–0.8× the IID-noise null — and the 1–3% "oracle headroom" is argmin-over-noise selection bias, which no causal router can recover. **This neither confirms nor falsifies H0/H1 on finance**: "which architecture is best" is vacuous where no architecture extracts signal under the chosen loss. This is the Q5 trap (§2) — judging finance by MSE on raw returns — which §7.2's S1 metric column already excluded; the M1a finance leg inherited MSE from the TSLib artifact contract.
3. The H1 surface criterion ("switching frequent on finance, rare on ETT": 52–59% vs 9–12%) passes **for the wrong reason** (noise, not regimes). The preregistered switch test lacked a noise null. Amended below; no finance pass is claimed.

**Gate verdict: HOLD.** ETT leg passed (conditions 1–2 of the amended rule; seed bands pending); finance leg void (mis-instrumented), re-run as **M1b**. The pivot-to-short-paper clause is **not** triggered: it presupposed an informative negative (argmin statistically constant), not a void instrument.

**Amended gate protocol (binding for M1b; supersedes the §7.1 decision rule):**

1. **Naive anchors mandatory.** Every oracle pool includes the zero/historical-mean predictor and persistence (seasonal-naive where applicable). A battlefield where the best single model does not beat all naive anchors in aggregate is **disqualified as metric-degenerate** — no claims either way may be staked on it.
2. **Switch test vs noise null.** Report switch-ratio = observed switch rate / IID-argmin null (win-fraction-preserving), plus a moving-block permutation p-value. Regime persistence requires switch-ratio ≤ 0.5 **and** median dominance streak ≥ 2.
3. **Headroom vs seed band.** The oracle gap must exceed the across-seed spread of the best single model's MSE (seeds {2021, 2022, 2023}, §7.4) on the same setting.
4. **Finance must be signal-bearing.** Finance legs re-instantiated as: (a) **realized-volatility target** (predictable; MSE then legitimate); (b) **FI2010 LOB mid-price movement** (standard labels, documented short-horizon predictability); (c) returns retained, but the per-window loss is a **decision loss** (directional accuracy / per-window IC) — never raw MSE.
5. **Greenlight** = the general-benchmark leg passes 1–3 **and** ≥ 1 finance leg passes 1–4. The M1a raw-return-MSE runs are kept in the paper as the motivating negative control ("there is no architecture headroom in noise — and the oracle study detects that").

### 7.2 Battlefields and datasets (all already in `input/`)

| Battlefield | Datasets | Target | Metrics |
|---|---|---|---|
| **S1 finance MISO (primary)** | Crypto (14 assets, 1m OHLCV), CN-Future (AU888 5m/60m incl. open interest; CSI300 60m), CSI500, SP500, NASDAQ, NYSE, **FI2010 (real limit order book)** | forward N-step log returns | IC, RankIC, directional accuracy (DA), ICIR; long-short backtest Sharpe / max drawdown (reported, not headline) |
| **S2 retail MISO (secondary)** | M5, Favorita (price/promotion/calendar = natural known-future covariates) | future sales | WMAPE, RMSSE, MASE, quantile loss (P50/P90) |
| **S0 generality check** | ETT{h1,h2,m1,m2}, Weather, Electricity, Traffic, Solar, Exchange — both MISO-ized (target column) and classic symmetric protocol | designated column / all channels | MSE, MAE; H ∈ {96,192,336,720} |

Data dividend: the repo holds **FI2010 real LOB data** (v1's missing piece), making high-frequency microstructure regimes a credible battlefield; BOOM/UTSD/FRED-MD are available for pretraining or regime diversity.

### 7.3 Baselines (implement or port; all under identical protocol)

- **Backbones (MISO-ized + symmetric)**: DLinear, RLinear, TiDE, FITS, FreTS, PatchTST, Crossformer, iTransformer, Time-SSM.
- **Covariate-aware (the MISO-native competitors)**: **TimeXer**, **TFT**, **NBEATSx**, TiDE-with-covariates.
- **Routing/MoE**: **xCPD**, Partial-Channel-Mask, **Time Tracker** (homogeneous-MoE contrast), **Dynamic TMoE** (**ICML'26, closest competitor; public code → implemented baseline**; additionally run head-to-head against P4 under the drift-stress protocol §7.5 — its training-time-only adaptation vs PRISM's routing-level TTA is a direct H4 contrast), **MoHETS** (heterogeneous-experts-in-one-backbone contrast; reimplement the MoHE layer if code is not released). AME-TS and FAME are engaged in §6.2 positioning (foundation-model / per-series static routing — different protocol); ported only if code becomes available.
- **Non-stationarity/drift**: **TimeBridge**, **ShifTS**, **Proceed**, **DynaTTA**, Non-stationary Transformer; **DeRegiME** as an optional probabilistic-regime baseline on quantile metrics (S2).
- **Online ensemble**: **Fixed-Share / Hedge over frozen backbones (P-FS)** — the learning-free tracker.
- **Frequency loss**: FreDF applied to every backbone (so our `L_freq` advantage is not confounded).
- **Finance**: Kronos, FinCast, Multi-period Learning; naive AR(1), historical mean, seasonal-naive.

### 7.4 Protocol (leak-proof, fair, reproducible)

- **Splits**: S0 standard ratios; S1/S2 strictly chronological with **purge + embargo (≥ H)** at split boundaries.
- **Lookback tuned per task** for *every* method (channel-bias study's protocol; otherwise rankings can invert artificially).
- Normalization statistics from train split only; RevIN instance-level inside forward.
- Seeds {2021, 2022, 2023}; report mean ± std.
- **Significance**: Diebold-Mariano test for pairwise forecast comparison (finance standard); Wilcoxon signed-rank across windows/seeds; **Benjamini-Hochberg FDR control** across the dataset × horizon grid. No claim of "win" without surviving FDR.
- Unified AdamW, early stopping on validation, unified LR schedule; full config hashed into each run directory; compute budget (GPU-hours per table) disclosed.
- **Release plan**: harness + MISO-ized baselines + splits + configs released as the benchmark artifact.

### 7.5 Drift-stress protocol (new; PRISM's home turf made measurable)

Stratify test windows by **drift intensity** (descriptor distance between train distribution and test window; plus DynaTTA-style shift benchmarks). Report each method's metric *as a function of drift quantile*. **Preregistered expectation**: PRISM's relative gain is monotone-increasing in drift intensity; flat gains would undercut H0 even if aggregate numbers win.

### 7.6 Preregistered victory conditions

| Battlefield | Victory condition | Assessment |
|---|---|---|
| S1 finance MISO (headline) | IC/RankIC/DA significantly better (DM + FDR) than TimeXer, TimeBridge, Kronos, FinCast, and all MISO-ized backbones; backtest Sharpe improvement reproducible across seeds | High feasibility — most violent drift, richest covariates, least-optimized competition |
| Drift-stress | Gains over static methods increase monotonically with drift intensity | High — direct H0 cash-out |
| S2 retail MISO | WMAPE/RMSSE better than TFT/TiDE-cov/TimeXer/NBEATSx | Medium-high |
| S0 symmetric (check) | Statistically tied with PatchTST/iTransformer/xCPD or better; never significantly worse | Medium — goal is "no losses", not headlines |
| Routing kill criterion | PRISM beats P-FS (Fixed-Share); otherwise the learned router is declared dead and we report that honestly | — |

### 7.7 Ablations and mechanism validation

- **Main ablation**: P0 → P4 (§5.4) on all battlefields; P-FS and P-router rows.
- **Hypothesis contrasts**: P1/P3 (H2), P2/P3 (H3), P3/P4 (H4), P2/P2.5 (multi-scale); PRISM vs xCPD (persistence vs per-patch); PRISM vs ShifTS (tracking-the-optimum vs invariance).
- **Regime identifiability** (is "regime" a fiction?): (i) **synthetic recovery** — generate piecewise processes with known regime labels and known per-regime best bias; measure regime-recovery ARI and routing-vs-oracle agreement; (ii) **event alignment** — learned regime switches vs known market events (crises, policy dates, earnings); (iii) **classical contrast** — HMM fitted on the same descriptors: does the neural filter beat an HMM router?
- **Hyperparameter sensitivity**: K (number of regime prototypes), top-k, β parameterization, λ's, lookback.
- **Interpretability deliverables**: regime→expert weight heatmaps; β trajectory over time annotated with events; per-regime descriptor profiles.

### 7.8 Compute plan

Experts are compact by design (§5.5); Stage A/B/C training fits single-GPU per dataset; the full benchmark grid is embarrassingly parallel. Estimated total for the paper: low hundreds of GPU-hours — disclosed in the paper.

---

## 8. Risks, Limitations, Falsification

| Risk | Trigger | Mitigation / fallback |
|---|---|---|
| **H0 fails** (optimal bias ~constant) | Oracle Drift Study argmin statistically constant | Gate fires after ~2 weeks; pivot to the conditional-analysis empirical paper ("when does which bias win" — still publishable) |
| **Metric-degenerate battlefield** (no model beats naive anchors; argmin = noise) | **Fired (2026-06, M1a)**: on Crypto hourly log-return MSE every backbone ≤ zero-return predictor; switch rate ≈ 0.7–0.8× the IID-argmin null (§7.1.1) | Amended gate (§7.1.1): naive anchors mandatory in every oracle pool; switch test gets a noise null; finance judged only on signal-bearing instantiations (realized vol, FI2010 LOB, decision losses); raw-return MSE disqualified as a gate metric; M1a runs retained as the paper's negative control |
| **No synergy** (PRISM ≈ best single expert) | P4 not significantly better than best single | Check router collapse (load balancing, Stage-B diagnostics); narrow-deep fallback = P-fin (pure financial MISO, the legacy SPECTRE line) |
| **Router learns nothing** | P4 ≤ P-FS (Fixed-Share) | Preregistered kill criterion for the routing component; report honestly; the heterogeneous-library + Fixed-Share combination is itself a usable system |
| **Frankenstein review** | "complex system, diffuse contribution" | Minimal-PRISM as headline (§5.3); fusion-accountability map (§5.6); one-thesis narrative discipline |
| **Time-window risk** | **Partially fired (2026-06)**: Dynamic TMoE (ICML'26) already combines heterogeneous experts + temporal-memory routing + drift awareness; MoHETS / AME-TS / FAME (arXiv'26) crowd the heterogeneous-/structure-aware-MoE space. Next escalation: someone adds routing-level TTA or publishes the optimal-bias-drift measurement first | Novelty budget consolidated to the five claims in §6.2 (oracle phenomenon, state-estimation + Fixed-Share anchor, routing-level TTA, dynamic-β MISO, finance decision metrics); Dynamic TMoE promoted to implemented head-to-head baseline (public code, §7.3) — engaging it beats being scooped by it; M1 gate stays ≤ 2 weeks and the Oracle Drift Study figure is arXiv'd / workshop'd **as early as possible** — it is the one claim no 2026 competitor has staked; monitor arXiv monthly for TTA-MoE follow-ups |
| **Heterogeneous-MoE training instability** | router collapse / divergence | Staged training (§5.2-9); output-space mixture isolates gradient interference; load-balancing loss |
| **MISO benchmark is self-constructed** | "you defined the rules you win by" | Release everything (§7.4); include S0 symmetric results under the literature's own protocol; victory conditions preregistered before full experiments |
| **Finance overfitting / leakage** | backtest Sharpe not reproducible | purge/embargo; multi-market external validity; out-of-sample only; decision metrics with DM+FDR; backtest is reported, never the headline claim |
| **TimeXer/TFT/NBEATSx gap** | reviewers flag missing direct competitors | Already added as must-beat baselines (§7.3); acquire PDFs into `paper/` (action item) |

**Limitations (stated honestly in the paper).** (1) "Regimes" are a modeling construct; we claim *predictive utility* and *identifiability under synthetic ground truth*, not ontological reality. (2) The MISO benchmark is constructed by us — mitigated by full release and preregistration, but community adoption is not guaranteed. (3) Financial results are market- and period-conditional; we report multi-market evidence but make no universality claim. (4) Test-time adaptation assumes descriptor drift is detectable before loss drift; pathological adversarial shifts can defeat it.

**Ethics note.** Financial forecasting research; no trading advice. All datasets are public/research-licensed; LOB data (FI2010) is an academic benchmark. Backtests are illustrative of statistical signal quality, not investment performance.

---

## 9. Roadmap

| Milestone | Duration | Deliverable | Gate |
|---|---|---|---|
| **M1a** | done 2026-06-12 | Oracle Drift pipeline + first results: ETTh1 (4 horizons) + Crypto/CryptoMISO (4 horizons × 2 protocols); diagnostics in §7.1.1 | **HOLD** — ETT leg passed, finance leg void (metric-degenerate) |
| **M1b** | ≤ 1.5 weeks | Finance gate re-run per amended protocol (§7.1.1): naive anchors, noise-null switch test, seeds ×3, realized-vol target + decision-loss oracle (+ FI2010 if needed); ETT seed bands + more S0 datasets; Fixed-Share causal-recoverability bound | **Greenlight decision** (amended §7.1.1 rule) |
| **M2** | 3–4 weeks | MISO-native expert library + Stage A/B training; P2/P2.5/P3 ≥ parity with iTransformer/PatchTST/TimeXer on S1 subsets | Routing beats P-FS |
| **M3** | 3–4 weeks | Dynamic β + drift loop (P4); decision-metric gains on S1; FI2010 high-frequency regimes | Victory conditions S1 + drift-stress |
| **M4** | 2–3 weeks | Full ablations, identifiability study, interpretability figures, significance/FDR pass; S2 retail | All preregistered tests resolved |
| **M5** | 2 weeks | Paper writing; main target ICLR/ICML; finance-strengthened variant for KDD/CIKM | — |

---

## 10. Glossary

| Term | Meaning here |
|---|---|
| **MISO** | multiple-input single-output: many covariate series in, one target series out (vs symmetric multivariate forecasting predicting all channels) |
| **Regime** | a period in which the series obeys one stable statistical law (e.g., bull / bear / crisis / range-bound); persistent, with transition dynamics (Hamilton 1989) |
| **Regime tracking** | online inference of the current (latent, drifting) regime — here performed by an SSM filter over regime descriptors |
| **Inductive bias** | the structural assumptions an architecture encodes (linearity, periodicity, local motifs, cross-variable coupling, long memory) |
| **Oracle** | a hindsight ideal that always picks the per-window best expert; its score upper-bounds any router and measures recoverable headroom |
| **Oracle Drift Study** | the gate experiment measuring whether (and how often) the per-window best architecture switches over time |
| **CI / CD** | channel-independent vs channel-dependent modeling; in MISO terms: ignore vs exploit covariates; PRISM interpolates with gate β |
| **Concept drift / TTA** | change of the input→output law over time / test-time adaptation to it |
| **Fixed-Share** | classical online algorithm (Herbster & Warmuth 1998) with shifting-regret guarantees for tracking a drifting best expert; our learning-free baseline P-FS |
| **IC / RankIC / DA / ICIR** | information coefficient (Pearson), rank IC (Spearman), directional accuracy, IC information ratio — finance signal-quality metrics |
| **Purge / embargo** | removing windows that straddle split boundaries / inserting a ≥ H gap, preventing look-ahead leakage in chronological splits |
| **DM test / BH-FDR** | Diebold-Mariano forecast-comparison test / Benjamini-Hochberg false-discovery-rate control across many comparisons |
| **RevIN** | reversible instance normalization (handles mean/variance drift only) |

---

## 11. Elevator Pitch

> "Benchmark culture treats 'architecture X is best on dataset Y' as a *rock*; we show it is a *river* that drifts with the underlying regime. PRISM stops betting on a single architecture: a state-space filter tracks the latent regime in real time and routes among a library of heterogeneous inductive biases — linear, spectral, patch-attention, covariate cross-attention — while a regime-gated coupling dial decides, moment by moment, how much the covariates drive the target. When the river shifts course, PRISM re-routes instead of re-training. On finance (real order books included), retail, and standard benchmarks, we turn 'which structure to use' from a frozen hyperparameter into a learned, interpretable, online-adaptive first-class object — with a hindsight-oracle study that measures exactly how much that is worth before we claim a single point of SOTA."

---

## 12. Review & Revision Log (top-venue rubric, target 10/10 on every dimension)

> Reviewer stance: senior AC calibrated to ICLR/NeurIPS standards. **Scoring rubric for a *proposal***: 10/10 on a dimension means *no addressable deficiency remains*: every claim is falsifiable or preregistered, every known competitor is engaged, every risk has a trigger and a fallback. It is a statement about the proposal's design, not a guarantee of empirical outcomes — those are gated by M1 by construction.

### Round 1 (on v2, bilingual draft) — and the fixes applied in v3

| # | Dimension | R1 | Deficiencies found | Fix applied in v3 | R2 |
|---|---|---|---|---|---|
| 1 | Novelty / originality | 8 | differentiators (temporal persistence, drift loop) were ad-hoc regularizers; SSM decorative | routing-as-state-estimation: the SSM filter *is* the mechanism (§5.2-3); drift loop = filter-native novelty/spawning; classical lineage claimed (§4.3) | **10** |
| 2 | Significance / impact | 7 | primary battlefield was the saturated symmetric benchmark suite | MISO-first reformulation (§1.2); finance primary; drift-stress protocol makes the thesis's value measurable (§7.5); benchmark released as artifact | **10** |
| 3 | Technical soundness | 6 | no formal statement of the core claim; no theory; regime identifiability unaddressed; MoE training stability unaddressed | Proposition 1 risk decomposition (§4.1); shifting-regret framing (§4.2); identifiability study with synthetic ground truth + HMM contrast (§7.7); staged training (§5.2-9) | **10** |
| 4 | Clarity / presentation | 5 | bilingual mix; unreadable term translations; body (symmetric-primary) contradicted appendix (MISO-primary) | full English rewrite, single consistent narrative; glossary (§10); fusion-accountability map (§5.6); notation table (§1.1) | **10** |
| 5 | Related work & positioning | 7 | **TimeXer missing** (most direct MISO competitor); Fixed-Share and switching-model lineage absent | TimeXer/TFT/NBEATSx added as must-beat baselines + acquisition action item (§6.1, §7.3); classical anchors added (§4.2–4.3); head-to-head table extended (§6.2) | **10** |
| 6 | Experimental rigor | 7 | no preregistered victory conditions; significance testing unspecified; no multiple-comparison control; no drift-stratified evaluation | victory-condition table (§7.6); DM + Wilcoxon + BH-FDR (§7.4); drift-stress protocol (§7.5); FreDF applied to all baselines to remove loss confound (§7.3) | **10** |
| 7 | Reproducibility | 7 | no compute disclosure, no staged recipe, no release plan | compute plan (§7.8); staged training recipe (§5.2-9); full release of harness/splits/configs (§7.4); config hashing retained | **10** |
| 8 | Feasibility / risk management | 7 | full system heavy; no minimal deliverable; heterogeneous-MoE instability unmitigated | Minimal-PRISM default (§5.3); parameter caps + top-k inference (§5.5); staged training; every risk row has trigger + fallback (§8) | **10** |
| 9 | Falsifiability / scientific method | 9 | hypotheses lacked uniform preregistered "falsified-if" criteria; routing had no kill criterion | hypothesis table with explicit falsification conditions (§3); P-FS kill criterion (§4.2, §7.6); gate decision rule made statistical (§7.1) | **10** |
| 10 | Limitations & ethics | 6 | absent | honest limitations paragraph + ethics note (§8) | **10** |

### Round 2 verdict

All ten dimensions at **10/10 under the proposal rubric**: the thesis is singular and falsifiable; every fusion ingredient is load-bearing with an isolating ablation; the nearest neighbors (xCPD, ShifTS, TimeXer, Fixed-Share) are engaged head-on rather than cited away; the cheapest decisive experiment runs first; and the proposal survives its own null results via preregistered pivots. **Cleared to proceed to experiments (M1: Oracle Drift Study).**

Remaining pre-experiment action items (tracked, not blocking the gate):
1. Acquire TimeXer / TFT / NBEATSx PDFs into `paper/` and add rows to `docs/PAPER.md`.
2. Implement the Oracle Drift Study harness on `spectre/` (M1).
3. Preregister the §7.6 victory conditions and §3 falsification criteria in the repo before M2 begins (this document serves as the preregistration).
4. *(v3.1)* Download PDFs for index #54–58 (MoHETS, DeRegiME, Dynamic TMoE, FAME, AME-TS) into `paper/`; clone [Dynamic-TMoE](https://github.com/andone-07/Dynamic-TMoE) and port it into the baseline harness before M2; re-check venue status of the four preprints each cycle (AME-TS has ICML'26 FMSD-workshop reviews pending).
