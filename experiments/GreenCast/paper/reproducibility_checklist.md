# GreenCast Paper Delivery Reproducibility Checklist

## Scope
This checklist verifies that the project is delivered as a **backbone architecture** contribution and not a benchmark/theory-only manuscript.

- [ ] Paper scope is architecture-only and mechanism-first.
- [ ] No section claims benchmark leaderboard ranking without backbone attribution mapping.
- [ ] No section claims theoretical proof as the main contribution.

## Pre-registration and Lock Files
- [ ] `configs/prereg.yaml` exists and is immutable before M2.
- [ ] `configs/*.yaml` include full experiment metadata.
- [ ] Manifest schema includes: dataset, seeds, horizons, beta settings, git hash, device, pin_memory.

## Data and Split Protocol
- [ ] Chronological split + purge/embargo applied.
- [ ] Target rotation and seed list fixed and logged.

## Metrics and Stats
- [ ] Forecast + mechanism metrics are both reported.
- [ ] DM test, Wilcoxon, BH-FDR values are computed and logged.
- [ ] Effect sizes and CIs are included for primary claims.

## Reporting and Failure Gates
- [ ] 10/10 gate matrix in review file is updated with executed evidence.
- [ ] If any fail gate is present, the claim branch is updated consistently (reduced claims / downgrade narrative).

## Deliverables
- [ ] `paper/paper.tex` exists.
- [ ] `paper/tables/RESULTS_main.csv` is appended by run.
- [ ] `paper/tables/Table1.tex` / `Table2.tex` generated from scripts.
- [ ] Figures include architecture and mechanism plots.

