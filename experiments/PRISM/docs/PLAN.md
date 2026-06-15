# PLAN — PRISM after M2 (2026-06-14)

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

Status update (2026-06-15): complete; **gate failed after delayed-feedback and
validation-single hardening**. To keep M2 killable, this milestone used the M1c
frozen expert pool rather than training a large neural system.
`router_viability.py` evaluates ETTh1, ETTh2, ETTm1, ETTm2, Weather, and
Exchange with a chronological 60/40 split: train-selected single, validation
single, oracle, delayed Fixed-Share, descriptor ridge loss probe, and a minimal
PRISM router (descriptor ridge + delayed online loss prior + sticky penalty).

13. **M2 goal**: prove that a learned/descriptor-conditioned router can beat
    Fixed-Share, the descriptor ridge probe, and a validation-selected single
    expert on the ETT/Weather/Exchange battlefields.
14. **M2 gate**: PRISM router loss < Fixed-Share loss, < descriptor ridge loss,
    and < validation-single loss on all six datasets.
15. **M2 deliverables**: `router_viability.py`,
    `router_viability/router_viability_summary.json`, updated docs, commit, tag.
16. **M2 result**: ETTh1, ETTm2, and Weather PASS; ETTh2, ETTm1, and Exchange
    FAIL. ETTh2 fails against descriptor ridge; ETTm1 and Exchange fail against
    validation-selected single experts. Per the strengthened kill criterion,
    the learned router is not viable as the headline system.

## Phase M3 — Dynamic β + drift loop (pivoted)

Status update (2026-06-14): complete; **gate passed narrowly**. The stress
loop improved stress-weighted loss on ETTh1, ETTm2, and Weather, and β had
nontrivial IQR on all three. However, the tuned drift gain was 0.0 throughout:
the gain came from dynamic β weighting, not from drift-triggered share-rate
adaptation. This keeps β as a useful diagnostic/weighting signal but weakens
the claim that the current drift detector adds value.

Entry condition after M2: the standalone learned router failed to beat
Fixed-Share on ETTm2, so M3 proceeds on the **pivot system**: frozen
heterogeneous experts + Fixed-Share as the robust causal tracker, with PRISM
contributions narrowed to dynamic β diagnostics and drift-loop adaptation.
M6 update: delayed-feedback plus validation-single hardening expands this to
six M1c datasets; the delayed contextual router passes only 3/6 and fails
ETTh2, ETTm1, and Exchange.

Goal: test whether descriptor-drift monitoring and a dynamic covariate-coupling
proxy β improve or at least explain the pivot system under controlled drift
stress. Gate: drift-loop Fixed-Share must improve the stressed loss versus plain
Fixed-Share on at least two battlefields, and β must show nontrivial
variation/alignment with channel-correlation descriptors. M6 update: this stress
gate passes on 4/6 versus plain Fixed-Share, but M4 block/FDR significance
still fails on 0/6 and the full loop loses to validation-single on all six.

Deliverables: `drift_beta_loop.py`,
`drift_beta_loop/drift_beta_summary.json`, stress tables in REPORT.md,
PLAN/PROPOSAL updates, commit/tag `m3-dynamic-beta-drift-loop`.

## Phase M4 — Ablations, significance, identifiability

Status update (2026-06-15): complete; **gate failed after strengthened
block/FDR and validation-single hardening**. Full-vs-plain survives BH/FDR on
0/6 datasets; full-vs-validation-single also survives on 0/6 and is negative on
all six datasets. Synthetic identifiability sanity check still recovers the
known regime label with 96.6% accuracy, but this does not rescue the real-data
method claim.

Entry condition after M3: M3 produces fixed per-window method losses for the
pivot system and variants. M4 runs ablations, paired significance tests with
Benjamini-Hochberg FDR, descriptor-feature interpretability, and synthetic
identifiability checks. Gate: if no pivot method is significant after FDR, the
paper becomes an empirical oracle/negative-router study rather than a method
paper.

## Phase M5 — Paper-ready artifacts

Status update (2026-06-14): complete. M5 freezes the final route as an
ETT-only empirical/pivot paper, writes `paper_ready/REPRODUCE.md`, and records a
machine-readable `paper_ready/paper_ready_summary.json`.

Entry condition after M4: final method status is known. M5 freezes artifacts,
tables, reproducibility commands, and final docs; it is a packaging milestone,
not a new algorithmic claim.

## Phase M6/M7/M8 — Main-track audit, multi-horizon pilot, expanded pool

Status update (2026-06-15): complete; **main-track submission remains
blocked**. M6 adds strong readiness criteria over M2/M3/M4 and blocks the
method-paper claim. M7 adds a cheap H=192 M1C router pilot to test whether the
M2 failure is specific to H=96. M8 adds an expanded causal expert pool to test
whether the failure is merely due to insufficient expert diversity.

M7 result: delayed contextual router passes only 1/6 datasets at H=192. M8
result: expanded expert-pool pilots pass 0/6 at H=96 and 1/6 at H=192. The
expanded pool improves static validation champions, but the current
descriptor/prior router cannot exploit that diversity. The next algorithmic
milestone must introduce a new causal gate that predicts when the validation
champion is unsafe; more threshold tuning around the existing descriptor
ridge/prior router is unlikely to produce a strong main-track method result.

Next admissible main-track attempt:

17. **New router signal**: add a genuinely causal predictor of when
    validation-single is unsafe, validated by walk-forward folds before outer
    test.
18. **New expert diversity**: expanded causal experts alone are insufficient;
    any new pool must come with a gate that beats validation-single rather than
    merely lowering the validation-single number.
19. **Reopen M2/M4 only if** H=96 and H=192 both satisfy non-inferiority versus
    validation-single on all six datasets and FDR-stable gains on at least two.

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
