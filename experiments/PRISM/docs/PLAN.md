# PLAN — PRISM after M1a (2026-06-12)

**Verdict: continue in the original direction.** The core thesis — *the optimal
inductive bias drifts over time and must be tracked* — is intact and now has its
first strong empirical support (ETTh1: 23–25% hindsight-oracle headroom with
multi-day architecture-dominance streaks). What changed is the *instrumentation*
of the finance gate: raw hourly log-return MSE turned out to be metric-degenerate
(no backbone beats the zero predictor, so "which architecture is best" is
unanswerable there). The proposal is amended accordingly
([PROPOSAL.md §7.1.1](experiments/PRISM/docs/PROPOSAL.md)); method (§5),
positioning (§6), and hypotheses (§3) are unchanged.

Evidence behind this plan: [docs/REPORT.md](experiments/PRISM/docs/REPORT.md).

---

## Phase 0 — Repo hygiene (½ day)

1. Commit the proposal relocation (`docs/PROPOSAL.md` → `experiments/PRISM/docs/PROPOSAL.md`),
   the v3.2 amendment, the M1a artifacts (`experiments/PRISM/oracle_drift/*`,
   `produce_predictions.py`, `data/`), the updated REPORT.md, and this PLAN.md.
2. Tag the commit `m1a-oracle-drift` so the gate evidence is citable/reproducible.

## Phase M1b — Re-instrument and finish the finance gate (≈ 1 week)

Status update (2026-06-13): complete, including the FI2010 LOB leg. Final
adjudication remains **ETT-only PRISM**: ETTh1 passes; all finance legs fail the
amended strict switch condition. See [REPORT.md](experiments/PRISM/docs/REPORT.md)
Part 2 for the completed tables.

Goal: a valid greenlight/kill decision per the amended gate (§7.1.1). Order
chosen so the cheapest, no-retraining steps come first.

3. **Add naive anchors + noise diagnostics to `oracle_drift.py`** (no retraining):
   - Synthetic pool members computed from `true.npy` / the input panel:
     `ZeroPred` (predict 0 / unconditional mean) and `Persistence` (repeat last
     observed value).
   - Extend `summary.json` with: `anchor_mse`, `best_vs_anchor_pct`,
     `switch_null` (IID-argmin, win-fraction-preserving), `switch_ratio`,
     `median_streak`, `max_streak`, and a moving-block permutation p-value for
     the switch test.
   - Re-emit summaries for all 12 existing runs (pure post-processing).
4. **Decision-loss oracle on the existing return forecasts** (no retraining):
   - Add `--window-loss {mse,da,ic}` to `oracle_drift.py`: per-window
     directional accuracy and per-window IC (corr(pred path, true path)).
   - Run on the 8 Crypto/CryptoMISO settings. If DA/IC also shows models ≈
     anchors, that *kills returns as an M1 target* (vol/LOB take over); if DA
     shows persistent per-window winners, returns stay in via decision loss.
5. **Realized-volatility target** (small code change, retraining needed):
   - In `crypto_dataset.py`: `target="vol"` → hourly log realized volatility of
     BTC built from the 1-minute closes (`log RV` is approximately Gaussian and
     strongly autocorrelated → predictable, so MSE is legitimate).
   - In `produce_predictions.py`: `--target vol --dataset-tag CryptoVol`.
   - Run 4 models × H∈{24, 96} first (seed 2021), oracle-drift them with
     anchors (incl. a HAR-style persistence anchor). Expand to all horizons
     only if the anchor check passes.
6. **Seed bands** (gate condition 3): repeat the surviving finance settings and
   ETTh1-equivalent producers with seeds 2022, 2023; report
   best-single across-seed spread vs oracle gap. (ETTh1 seed bands: rerun the 4
   TSLib baselines with the standard scripts, 3 seeds, or accept published
   variance if rerunning is disproportionate — decide when sizing compute.)
7. **Pool completion**: add `FITS` (the `E_freq` proxy — cheap) to the finance
   pool; add `TimeXer` to CryptoMISO (covariate-aware competitor, already in
   TSLib). Optional: Time-SSM if port cost is low.
8. **FI2010 LOB leg** (only if vol + DA/IC legs don't already resolve the gate):
   producer for mid-price movement at k∈{10, 50, 100} ticks from `input/FI2010`,
   per-window cross-entropy/F1 oracle study. This is the largest M1b item —
   gate on it last.
9. **Adjudicate the gate** per amended §7.1.1 rule 5; record the decision +
   tables in REPORT.md and the proposal status line. Greenlight ⇒ M2.
   Finance fails everywhere but ETT-side holds ⇒ rescope S1/S0 priority
   (decide then; the phenomenon paper remains viable). Both fail ⇒ §8 pivot.

## Phase M1c — Strengthen the phenomenon claim (parallel with M1b, ≈ 3–4 days)

Status update (2026-06-14): complete. M1c ran a lightweight MISO breadth
screen on ETTh1/ETTh2/ETTm1/ETTm2/Weather/Exchange at L=96,H=96 using a
closed-form heterogeneous predictor pool (`RidgeCov`, `TargetRidge`, `Trend`,
`Seasonal`, `EWM`) plus anchors. Fixed-Share and descriptor→winner probes were
run for all six settings. See REPORT.md Part 4.

The ETT result is the paper's Figure 1; make it unassailable and test
*causal* recoverability (the bridge from oracle to router).

10. **Breadth**: run the oracle study (with anchors + diagnostics) on ETTh2,
    ETTm1, ETTm2, Weather, Exchange (MISO-ized, target column), using existing
    TSLib artifacts where available. Deliverable: switch-ratio / oracle-gap
    table across datasets — "where does the phenomenon live?"
11. **Fixed-Share causal bound (P-FS-lite)**: run Fixed-Share/Hedge over the
    frozen per-window losses (pure post-processing of `window_losses.csv`) —
    a *causal* tracker, no hindsight. Report recovered fraction of the oracle
    gap on ETTh1 vs Crypto. This quantifies M2's realistic headroom and gives
    the §4.2 baseline its first numbers. Expect: meaningful recovery on ETTh1,
    ≈ 0 on Crypto-MSE (confirming the noise diagnosis causally).
12. **Descriptor → winner probe (routability)**: compute the §5.2-2 regime
    descriptors per window (spectral bands, entropy, intra-window drift,
    correlation stability); fit a causal (train-on-past, predict-forward)
    multinomial probe of the per-window winner. Probe accuracy ≫ marginal
    win-rate on ETTh1 = the router has signal to learn *before* any expert is
    trained. This is the cheapest possible de-risk of M2's Stage B.

## Phase M2 — Expert library + routing (starts only after step 9 greenlight)

13. As per proposal §9/§5.2-9, scoped to the battlefields that passed the gate:
    Stage A expert pretraining (E_lin, E_freq, E_patch, β-gated E_chan),
    Stage B SSM-filter routing on frozen experts, judged first against the
    per-window oracle and the P-FS bound from step 11 (the preregistered kill
    criterion: routing must beat Fixed-Share).

---

## Decision gates recap

| After | Question | Pass → | Fail → |
|---|---|---|---|
| Step 4 | Do return forecasts carry per-window decision-loss structure? | returns stay (decision loss) | returns dropped from M1 targets |
| Step 5 | Does any model beat anchors on log-RV? | vol becomes a finance gate leg | escalate to FI2010 (step 8) |
| Step 9 | Amended gate rule 5 | **M2 greenlight** | rescope per §8 fallbacks |
| Step 11 | Does Fixed-Share recover a meaningful share of the ETTh1 oracle gap causally? | M2 headroom confirmed | router thesis at risk — revisit before M2 spend |

## Compute notes

All M1b/M1c runs are MPS-friendly (minutes per model × horizon at these sizes,
per `produce_predictions.py`); the full M1b grid (≈ 4 models × 2–4 horizons ×
3 seeds × 2 targets) is an overnight batch on the Mac, embarrassingly parallel
if a GPU box is available. Device selection must follow CUDA → MPS → CPU
(AGENTS.md).
