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
| `paper/main.tex` | Anonymous LaTeX submission source |
| `paper/main.pdf` | Rendered anonymous submission PDF |
| `paper/references.bib` | Bibliography for LaTeX submission source |
| `paper/README.md` | Build instructions for the submission source |
| `docs/PROPOSAL.md` | Primary specification (v1.0, pre-G0) |
| `docs/COVTYPE.md` | Covariate typing for M5 + Favorita (M0 deliverable) |
| `m0_prior_art/m0_summary.json` | G0 prior-art sweep + identification diagnostics |
| `m1_audit/audit_summary.json` | G1 Scenario Validity Audit; greenlight decision |
| `m2_docast/docast_summary.json` | D0/D1/D2 ablation; H3 gate verdict |
| `m3_real_data/real_data_summary.json` | M5-SNAP NEE; Favorita promo NEE; PRF; BH-FDR |
| `m4_paper_ready/paper_ready_summary.json` | Master gate table; scoped headline claims |
| `m4_paper_ready/REPRODUCE.md` | This file |
| `m5_main_track_audit/main_track_audit.json` | Strict main-track readiness audit |
| `m6_backbone_sweep/backbone_sweep_summary.json` | TSLib deep-backbone full D0/D1/D2 protocol |

## Data Requirements

- `input/M5/m5/datasets/` — M5 competition files (calendar, sales, prices, weights)
- `input/Favorita/` — Favorita files (chunks/, holidays_events.csv, oil.csv, etc.)

## Seeds & Reproducibility

All experiments use seeds `[2021, 2022, 2023]`. Results are deterministic given these seeds.
The semi-synthetic generator is parameterized by `gamma` (confounding) and `delta` (V2 feedback).

## Milestone Status

- **M0**: PASS_WITH_SCOPE: novelty confirmed; M5 price overlap weak, use Favorita promotion for real a-type validation
- **M1**: GREENLIGHT
- **M2**: PASS
- **M3**: PASS
- **M4**: PASS
- **M6**: PASS_FULL_PROTOCOL

## Claim Scope

Current evidence is internally consistent and includes two real a-type validation
legs (Favorita promotion and M5 markdown). M6 completes the full D0/D1/D2
DoCast protocol on DLinear, PatchTST, TiDE, and TimeXer in the lightweight
semi-synthetic backbone audit. The claim is direct-submission scoped: it supports
intervention-valid scenario forecasting evidence, not a full leaderboard SOTA
claim across every TSF benchmark.

Run `m5_main_track_audit.py` for the stricter direct-submission gate.

## Target Venue

Direct main-track submission candidate; M5 audit should be green
