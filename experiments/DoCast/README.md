# DoCast — Do-Operator Forecasting

Intervention-valid scenario forecasting for MISO time series with controllable covariates
such as prices and promotions.

## Current Status

M0-M6 are implemented and reproducible in the local `markquant` conda environment.
The evidence chain is now scoped and internally consistent:

- **M0**: prior-art and covariate typing complete; M5 price overlap is weak, so M5 price is not used as the real-data effect gate.
- **M1**: semi-synthetic scenario-validity audit passes.
- **M2**: DoCast D2 ablation passes in the current linear MISO setting.
- **M3**: real a-type Favorita promotion validation passes, including a 4-configuration robustness grid; M5 SNAP is only a c-type non-degradation check; PRF is semi-synthetic only.
- **M4**: consolidated summary reports a direct main-track submission candidate.
- **M5**: strict main-track readiness audit is green: `DIRECT_SUBMISSION_READY`.
- **M6**: full D0/D1/D2 protocol passes on DLinear, PatchTST, TiDE, and TimeXer.

## Claim Scope

This directory supports a direct-submission scoped claim for intervention-valid
scenario forecasting. It includes two real controllable-covariate validation legs
(Favorita promotion and M5 markdown) and a lightweight full-protocol deep-backbone
audit. It still does not claim full leaderboard SOTA across every TSF benchmark.

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
