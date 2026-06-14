# DoCast Submission Bundle

This directory contains a conference-style anonymous LaTeX source package for
the verified DoCast evidence chain.

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

- `../m4_paper_ready/paper_ready_summary.json`
- `../m5_main_track_audit/main_track_audit.json`
- `../m6_backbone_sweep/backbone_sweep_summary.json`

The strict M5 audit currently returns `DIRECT_SUBMISSION_READY` with no blocking
items.
