# DoCast Submission Bundle

This directory contains a conference-style anonymous LaTeX source package for
the revised DoCast evidence chain.

## Files

- `main.tex` — anonymous manuscript source.
- `references.bib` — bibliography entries used by the manuscript.

## Build

On a machine with a LaTeX distribution:

```bash
cd experiments/DoCast/paper
tectonic main.tex
```

This package was rendered locally with Tectonic. The generated PDF is
`main.pdf`.

## Evidence Sources

The manuscript numbers are copied from these authoritative artifacts:

- `../m2_docast/docast_summary.json`
- `../m3_real_data/real_data_summary.json`
- `../m6_backbone_sweep/backbone_sweep_summary.json`
- `../m4_paper_ready/paper_ready_summary.json`
- `../m5_main_track_audit/main_track_audit.json`

The strict local readiness audit currently returns
`REVISED_MAIN_TRACK_CANDIDATE`. This is repository metadata, not a paper claim.
