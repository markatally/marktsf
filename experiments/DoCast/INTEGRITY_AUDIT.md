# DoCast Integrity Audit

Date: 2026-06-15

Scope: AI-research failure-mode audit for `experiments/DoCast/`, including
experiment code, generated JSON summaries, manuscript drafts, report files, and
the LaTeX submission source.

## Verdict

DoCast is no longer represented as an unconditional all-backbone success. The
corrected status is:

- `M4`: revised main-track candidate with a TimeXer stability caveat.
- `M5`: `REVISED_MAIN_TRACK_CANDIDATE`; all strict local readiness gates pass.
- `M6`: DLinear, PatchTST, TiDE, and Transformer pass the strict seed-level
  fair-control protocol. TimeXer is a mean-pass boundary case: 27.2% mean
  theta-RMSE reduction vs fair D1 and +2.14% mean WMAPE change, but only 2/3
  seeds pass because one seed has +6.41% WMAPE change.

## Failure Modes Found And Fixed

| Failure mode | Finding | Fix |
|---|---|---|
| Implementation bugs | M2 and M6 allowed train/test or nuisance-fold horizon overlap risk. | Added H-origin embargo for held-out tests and origin-level purged folds for cross-fitted nuisances. |
| Implementation bugs | M6 originally had dependency-incomplete TimeXer/PatchTST runs and single-seed/partially failed evidence. | Installed missing `reformer-pytorch`, reran PatchTST and TimeXer across seeds 2021/2022/2023, and added resume plus `--summarize-only`. |
| Implementation bugs | M3 Favorita parsing had pandas dtype fragility after warning cleanup. | Added explicit date format, string-safe promotion parsing, and post-concat numeric coercion for `unit_sales`. |
| Hallucinated results | Old paper text claimed 52.1-80.4% backbone reductions across all four backbones and no WMAPE degradation. Corrected M6 shows strict-pass reductions of 74.4%, 17.3%, 37.0%, and 57.1% on DLinear/PatchTST/TiDE/Transformer, with TimeXer at 27.2% as a mean-pass caveat. | Updated `PAPER.md`, `paper/main.tex`, `REPORT.md`, `README.md`, M4/M5 summaries, and M6 JSON. |
| Hallucinated results | Old p-value strings reported rounded-zero or stale values (`1.36e-24`, `3.22e-16`, `<1e-300`). | Preserved raw small p-values in JSON, report Favorita `8.78e-15`, M5 markdown `1.29e-06`, robustness max `0.0049`, and conservative `<1e-16` for true numerical underflow. |
| Shortcut reliance | Real-data validation can still exploit matched-proxy assumptions rather than randomized causal ground truth. | Manuscript/report now explicitly call Favorita and M5 markdown matched within-unit ATT proxies, not randomized causal validation. |
| Bug-as-insight reframing | TimeXer instability could have been framed as backbone success. | It is now a caveat/boundary result, while Transformer supplies the third strict-pass deep backbone. |
| Methodology fabrication | M4/M5 previously described full fair-control backbone completion as if all deep backbones passed. | M6 now stores strict-vs-mean pass criteria, seed pass counts, and max seed WMAPE change; M5 requires at least three strict-pass deep backbones. |
| Frame-lock | The prior narrative continued to rely on TimeXer despite failed strict evidence. | The strict evidence set was expanded with Transformer, while TimeXer stayed explicitly caveated. |
| Citation hallucinations | `vankadara2022causalforecasting` had the wrong author list; dynamic-DML title was incomplete; TimeXer carried an unverified proceedings venue. | `paper/references.bib` now matches the PMLR/official metadata for Vankadara et al., the Lewis-Syrgkanis title, and verified arXiv metadata for TimeXer. |

## Verification Run

Commands completed:

```bash
python -m py_compile experiments/DoCast/m0_prior_art.py experiments/DoCast/m1_audit.py experiments/DoCast/m2_docast.py experiments/DoCast/m3_real_data.py experiments/DoCast/m4_paper_ready.py experiments/DoCast/m5_main_track_audit.py experiments/DoCast/m6_backbone_sweep.py
conda run -n markquant python experiments/DoCast/m2_docast.py
conda run -n markquant python experiments/DoCast/m3_real_data.py
conda run -n markquant python experiments/DoCast/m6_backbone_sweep.py --backbones PatchTST --seeds 2021
conda run -n markquant python experiments/DoCast/m6_backbone_sweep.py --backbones PatchTST --seeds 2022,2023
conda run -n markquant python experiments/DoCast/m6_backbone_sweep.py --backbones Transformer --seeds 2021
conda run -n markquant python experiments/DoCast/m6_backbone_sweep.py --backbones Transformer --seeds 2022,2023
conda run -n markquant python experiments/DoCast/m6_backbone_sweep.py --backbones TimeXer --seeds 2021
conda run -n markquant python experiments/DoCast/m6_backbone_sweep.py --backbones TimeXer --seeds 2022,2023
conda run -n markquant python experiments/DoCast/m6_backbone_sweep.py --summarize-only
conda run -n markquant python experiments/DoCast/m4_paper_ready.py
conda run -n markquant python experiments/DoCast/m5_main_track_audit.py
cd experiments/DoCast/paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdfinfo main.pdf
```

Additional consistency checks assert the corrected M2/M3/M4/M5/M6 JSON values
and scan for stale headline strings. The final LaTeX run completed with no
undefined citation/reference warnings; `pdfinfo` reports a 9-page, unencrypted
PDF generated by pdfTeX 1.40.29.

## Remaining Risks

- TimeXer is not a strict seed-level pass. It must remain a caveat unless future
  work improves its seed-level stability.
- Real-data results are matched observational proxies, not randomized treatment
  effects.
- `paper/main.pdf` was rebuilt from `paper/main.tex` with Homebrew TeX Live
  20260301 / pdfTeX 1.40.29 after the local Tectonic 0.16.9 binary crashed in
  its macOS runtime initialization. The verified build path is the
  `pdflatex`/`bibtex` sequence in `paper/README.md`.
