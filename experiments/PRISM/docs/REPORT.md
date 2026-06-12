# Oracle Drift Study — M1a Report (2026-06-12)

Generated from TSLib prediction artifacts under `external/TSLib/results`
(ETTh1: pre-existing benchmark runs; Crypto/CryptoMISO: trained by
`experiments.PRISM.produce_predictions`). This is the complete M1a artifact set
behind the gate decision recorded in `experiments/PRISM/docs/PROPOSAL.md` §7.1.1.

## Scope

| Leg | Dataset | Models | Horizons | Protocol |
|---|---|---|---|---|
| Contrast | `ETTh1` | DLinear, PatchTST, TiDE, TimeXer | 96 / 192 / 336 / 720 | ftM, scored on OT only |
| Finance | `Crypto` | DLinear, PatchTST, iTransformer, TimesNet | 24 / 48 / 96 / 168 | ftM (14 close-ret channels), scored on BTCUSDT |
| Finance MISO | `CryptoMISO` | DLinear, PatchTST, iTransformer, TimeMixer | 24 / 48 / 96 / 168 | ftMS (28 features incl. volume), target BTCUSDT |

All runs: L = 96, seed 2021, stride-1 test windows, per-window MSE on the
target channel (`--target-channel -1`).

## Summary

*Noise null* = expected switch rate of an IID argmin with the observed win
fractions. *Anchor* = unconditional-mean / zero-return predictor MSE computed
from `true.npy`; negative "vs anchor" means the model beats the anchor.

| Setting | Best single | Best MSE | Oracle gap | Switch obs / null | Med · max streak | Best vs anchor |
|---|---|---:|---:|---:|---:|---:|
| ETTh1 H96 | PatchTST | 0.0554 | **25.3%** | .123 / .741 (0.17×) | 3 · 134 | **−97.1%** |
| ETTh1 H192 | TimeXer | 0.0696 | **23.6%** | .092 / .693 (0.13×) | 4 · 195 | −96.4% |
| ETTh1 H336 | TimeXer | 0.0831 | **22.8%** | .100 / .673 (0.15×) | 3 · 197 | −95.8% |
| ETTh1 H720 | TimeXer | 0.0888 | 6.7% | .108 / .538 (0.20×) | 3 · 145 | −95.6% |
| Crypto H24 | DLinear | 0.4944 | 2.8% | .549 / .733 (0.75×) | 1 · 23 | +0.14% |
| Crypto H48 | DLinear | 0.4906 | 1.9% | .568 / .736 (0.77×) | 1 | +0.24% |
| Crypto H96 | DLinear | 0.4919 | 1.3% | .526 / .737 (0.71×) | 1 · 27 | +0.36% |
| Crypto H168 | DLinear | 0.4957 | 1.0% | .511 / .737 (0.69×) | 1 | +0.54% |
| CryptoMISO H24 | DLinear | 0.4989 | 3.4% | .592 / .739 (0.80×) | 1 · 22 | **+1.05%** |
| CryptoMISO H48 | PatchTST | 0.4939 | 2.2% | .502 / .733 (0.68×) | 1 · 35 | +0.90% |
| CryptoMISO H96 | PatchTST | 0.4936 | 1.3% | .552 / .737 (0.75×) | 1 · 54 | +0.69% |
| CryptoMISO H168 | TimeMixer | 0.4969 | 1.2% | .557 / .749 (0.74×) | 1 · 34 | +0.78% |

## Reading

1. **ETTh1: the phenomenon PRISM bets on is real and large.** Per-window
   architecture dominance is strongly persistent (switch rate 0.13–0.20× the
   noise null; dominance streaks up to 197 consecutive windows ≈ 8 days), all
   four models win substantial stretches, and the hindsight oracle holds 23–25%
   headroom over the best single model at H96–336. The models genuinely model
   the series (97% below the naive anchor), so this headroom is signal, not
   selection noise.
2. **Crypto raw-return MSE: metric-degenerate.** Every backbone scores at or
   *below* the zero-return predictor at every horizon, in both ftM and ftMS
   protocols. The per-window argmin is then a choice among statistically tied
   models: median streak 1, switch rate 0.7–0.8× the IID-noise null, and the
   1–3% oracle gap is argmin-over-noise selection bias. These runs say nothing
   about H0/H1 — they show the chosen loss/target carries no architecture
   signal at all. They are retained as the negative control: the oracle study
   correctly distinguishes regime structure (ETTh1) from noise (Crypto-MSE).

## Gate status

**HOLD** (see `experiments/PRISM/docs/PROPOSAL.md` §7.1.1 for the amended, binding protocol):
the ETT leg passes the amended persistence + headroom conditions (seed bands
pending); the finance leg is void as instrumented and is re-run as **M1b** on
signal-bearing instantiations — realized-volatility target, FI2010 LOB, and
decision-loss (DA/IC) oracle analysis of the existing return forecasts — with
naive anchors in every pool, the noise-null switch test, and seeds ×3.

Step-by-step continuation: see `experiments/PRISM/docs/PLAN.md`.
