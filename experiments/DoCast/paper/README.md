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
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`main.tex` is the authoritative source. `main.pdf` was rebuilt from this source
with Homebrew TeX Live 20260301 / pdfTeX 1.40.29. The local Tectonic 0.16.9
binary still crashes in its macOS network/runtime initialization, so the
reproducible build path is the `pdflatex`/`bibtex` sequence above.

## Evidence Sources

The manuscript numbers are copied from these authoritative artifacts:

- `../m2_docast/docast_summary.json`
- `../m3_real_data/real_data_summary.json`
- `../m6_backbone_sweep/backbone_sweep_summary.json`
- `../m4_paper_ready/paper_ready_summary.json`
- `../m5_main_track_audit/main_track_audit.json`

The strict local readiness audit currently returns
`REVISED_MAIN_TRACK_CANDIDATE`. Transformer provides the third strict-pass deep
backbone; TimeXer remains a reported stability caveat rather than a strict
seed-level pass. This is repository metadata, not a paper claim.
