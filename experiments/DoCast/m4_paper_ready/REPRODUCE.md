# DoCast — Reproduction Manifest

Version: M4 (evidence-chain consolidated)

## Quick Start

```bash
# Environment: markquant conda env (numpy, pandas, scipy, sklearn)
# From repo root:
conda run -n markquant python experiments/DoCast/m0_prior_art.py
conda run -n markquant python experiments/DoCast/m1_audit.py
conda run -n markquant python experiments/DoCast/m2_docast.py
conda run -n markquant python experiments/DoCast/m3_real_data.py
conda run -n markquant python experiments/DoCast/m6_backbone_sweep.py
conda run -n markquant python experiments/DoCast/m4_paper_ready.py
conda run -n markquant python experiments/DoCast/m5_main_track_audit.py
```

## Artifact Manifest

| File | Content |
|---|---|
| `PAPER.md` | Main-track manuscript draft |
| `INTEGRITY_AUDIT.md` | Failure-mode audit and remaining risks |
| `paper/main.tex` | Anonymous LaTeX submission source |
| `paper/references.bib` | Bibliography for LaTeX submission source |
| `paper/README.md` | Build instructions for the submission source |
| `docs/PROPOSAL.md` | Primary specification (v1.0, pre-G0) |
| `docs/COVTYPE.md` | Covariate typing for M5 + Favorita (M0 deliverable) |
| `m0_prior_art/m0_summary.json` | G0 prior-art sweep + identification diagnostics |
| `m1_audit/audit_summary.json` | G1 Scenario Validity Audit; greenlight decision |
| `m2_docast/docast_summary.json` | D0/D1/D2 ablation; H3 gate verdict |
| `m3_real_data/real_data_summary.json` | M5 markdown NEE; Favorita promo NEE; PRF; BH-FDR |
| `m4_paper_ready/paper_ready_summary.json` | Master gate table; scoped headline claims |
| `m4_paper_ready/REPRODUCE.md` | This file |
| `m5_main_track_audit/main_track_audit.json` | Strict main-track readiness audit |
| `m6_backbone_sweep/backbone_sweep_summary.json` | TSLib deep-backbone full D0/D1/D2 protocol |

## Data Requirements

- `input/M5/m5/datasets/` — M5 competition files (calendar, sales, prices, weights)
- `input/Favorita/` — Favorita files (chunks/, holidays_events.csv, oil.csv, etc.)

## Seeds & Reproducibility

The semi-synthetic M1/M2/M6 experiments use seeds `[2021, 2022, 2023]`.
Real-data matched-proxy analyses are deterministic given the input files; bootstrap
intervals use fixed seeds documented in the scripts. The semi-synthetic generator
is parameterized by `gamma` (confounding) and `delta` (V2 feedback).

## Milestone Status

- **M0**: PASS_WITH_SCOPE: novelty confirmed; M5 price overlap weak, use Favorita promotion for real a-type validation
- **M1**: GREENLIGHT
- **M2**: PASS
- **M3**: PASS
- **M4**: PASS
- **M6**: PASS_FAIR_CONTROL_PROTOCOL

## Claim Scope

Current evidence is internally consistent and includes two real a-type validation
legs (Favorita promotion and M5 markdown). M6 completes the D0/D1/D2 DoCast
protocol on DLinear, PatchTST, TiDE, Transformer, and TimeXer with shared item
static controls and matched D1/D2 response capacity in the lightweight
semi-synthetic backbone audit. DLinear, PatchTST, TiDE, and Transformer pass
the strict seed-level protocol; TimeXer is a mean-pass boundary case because
one seed exceeds the 5% WMAPE-degradation tolerance. The claim is scoped to
intervention-oriented scenario forecasting under stated assumptions, not a full
leaderboard SOTA claim across every TSF benchmark.

Run `m5_main_track_audit.py` for the stricter local readiness gate.

## Target Venue

Revised main-track candidate after fair-control and reporting fixes
