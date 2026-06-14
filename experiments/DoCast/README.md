# DoCast — Do-Operator Forecasting

Orthogonalized scenario forecasting for MISO time series with controllable
future covariates such as prices, promotions, markdowns, or policy
interventions.

## Current Status

M0-M6 are implemented and reproducible in the local `markquant` conda environment.
The package has been revised after an ARS reviewer rejection. The current
version is a revised main-track candidate, not a guaranteed acceptance claim.
The evidence chain is now scoped and internally consistent:

- **M0**: prior-art and covariate typing complete; M5 price overlap is weak, so M5 price is not used as the real-data effect gate.
- **M1**: semi-synthetic scenario-validity audit passes.
- **M2**: DoCast D2 ablation now reports both the hidden-confounder stress test
  and a fair-control diagnostic where all arms receive item controls.
- **M3**: real a-type Favorita promotion validation passes, including a 4-configuration robustness grid; M5 SNAP is only a c-type non-degradation check; PRF is semi-synthetic only.
- **M4**: consolidated summary reports a revised main-track candidate after
  fair-control and reporting fixes.
- **M5**: strict local readiness audit returns `REVISED_MAIN_TRACK_CANDIDATE`.
- **M6**: fair-control D0/D1/D2 protocol passes on DLinear, PatchTST, TiDE, and
  TimeXer.

## Claim Scope

This directory supports a scoped claim for intervention-oriented scenario
forecasting under stated assumptions and overlap diagnostics. It includes two
real controllable-covariate matched-ATT proxy validation legs (Favorita
promotion and M5 markdown) and a lightweight fair-control deep-backbone audit.
It does not claim unrestricted intervention validity, randomized real-data
causal ground truth, HE-specific empirical validation, or full leaderboard SOTA
across every TSF benchmark.

## Reproduce

```bash
conda run -n markquant python experiments/DoCast/m0_prior_art.py
conda run -n markquant python experiments/DoCast/m1_audit.py
conda run -n markquant python experiments/DoCast/m2_docast.py
conda run -n markquant python experiments/DoCast/m3_real_data.py
conda run -n markquant python experiments/DoCast/m6_backbone_sweep.py
conda run -n markquant python experiments/DoCast/m4_paper_ready.py
conda run -n markquant python experiments/DoCast/m5_main_track_audit.py
```

Primary summaries:

- `PAPER.md`
- `paper/main.tex`
- `paper/main.pdf`
- `paper/references.bib`
- `REPORT.md`
- `m3_real_data/real_data_summary.json`
- `m4_paper_ready/paper_ready_summary.json`
- `m4_paper_ready/REPRODUCE.md`
- `m5_main_track_audit/main_track_audit.json`
- `m6_backbone_sweep/backbone_sweep_summary.json`
