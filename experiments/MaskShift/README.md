# MaskShift — Forecasting Under Missingness-Mechanism Shift

MaskShift studies whether MCAR/block missing-value robustness certifies real
deployment outages. The core claim is a benchmark/theory claim: at matched
missing rate, different missingness mechanisms can change forecast risk and
model rankings.

## Reproduce

Run from the repository root:

External repositories required for M9/M11/M12 are intentionally under ignored
`external/` paths. Prepare them before running official-architecture milestones:

```bash
git clone --depth 1 https://github.com/thuml/Time-Series-Library external/TSLib
git clone --depth 1 https://github.com/jinkwan1115/ChannelTokenFormer external/ChannelTokenFormer
git clone --depth 1 https://github.com/WINTERWEEL/S4M.git external/S4M
```

The current audited revisions are TSLib `4e938a1` and ChannelTokenFormer
`b1c100e`, plus S4M `a718823`.

```bash
python3 -m experiments.MaskShift.m0_mask_suite
python3 -m experiments.MaskShift.m1_mechanism_audit
python3 -m experiments.MaskShift.m2_typed_head
python3 -m experiments.MaskShift.m3_statistical_tests
python3 -m experiments.MaskShift.m6_deep_backbone_sweep
python3 -m experiments.MaskShift.m7_severity_curves
python3 -m experiments.MaskShift.m8_mechanism_decomposition
python3 -m experiments.MaskShift.m9_official_tslib_reproduction
python3 -m experiments.MaskShift.m10_submission_hardening
python3 -m experiments.MaskShift.m11_official_ctf_missing_baseline
python3 -m experiments.MaskShift.m12_official_s4m_baseline
python3 -m experiments.MaskShift.m13_hierarchical_bootstrap
python3 -m experiments.MaskShift.m14_s4m_scale_validation
python3 -m experiments.MaskShift.m16_official_tslib_full_coverage
python3 -m experiments.MaskShift.m17_submission_supplement
python3 -m experiments.MaskShift.m18_submission_policy_pack
python3 -m experiments.MaskShift.m19_aaai27_target_readiness
python3 -m experiments.MaskShift.m20_aaai27_preflight_conversion
python3 -m experiments.MaskShift.m21_aaai27_reproducibility_checklist
cd experiments/MaskShift/paper && tectonic --print main.tex && tectonic --print supplement.tex && tectonic --print submission_statements.tex && tectonic --print aaai27_readiness.tex && tectonic --print aaai27_preflight.tex && pdflatex -interaction=nonstopmode aaai27_official.tex && bibtex aaai27_official && pdflatex -interaction=nonstopmode aaai27_official.tex && pdflatex -interaction=nonstopmode aaai27_official.tex && cd ../../..
python3 -m experiments.MaskShift.m17_submission_supplement
python3 -m experiments.MaskShift.m18_submission_policy_pack
python3 -m experiments.MaskShift.m19_aaai27_target_readiness
python3 -m experiments.MaskShift.m20_aaai27_preflight_conversion
cd experiments/MaskShift/paper && tectonic --print submission_statements.tex && cd ../../..
python3 -m experiments.MaskShift.m18_submission_policy_pack
cd experiments/MaskShift/paper && tectonic --print aaai27_readiness.tex && tectonic --print aaai27_preflight.tex && pdflatex -interaction=nonstopmode aaai27_official.tex && bibtex aaai27_official && pdflatex -interaction=nonstopmode aaai27_official.tex && pdflatex -interaction=nonstopmode aaai27_official.tex && cd ../../..
python3 -m experiments.MaskShift.m19_aaai27_target_readiness
python3 -m experiments.MaskShift.m20_aaai27_preflight_conversion
python3 -m experiments.MaskShift.m21_aaai27_reproducibility_checklist
python3 -m experiments.MaskShift.m15_final_integrity_audit
python3 -m experiments.MaskShift.m5_main_track_audit
python3 -m experiments.MaskShift.m18_submission_policy_pack
cd experiments/MaskShift/paper && tectonic --print submission_statements.tex && cd ../../..
python3 -m experiments.MaskShift.m18_submission_policy_pack
python3 -m experiments.MaskShift.m19_aaai27_target_readiness
python3 -m experiments.MaskShift.m20_aaai27_preflight_conversion
python3 -m experiments.MaskShift.m21_aaai27_reproducibility_checklist
cd experiments/MaskShift/paper && tectonic --print aaai27_readiness.tex && tectonic --print aaai27_preflight.tex && pdflatex -interaction=nonstopmode aaai27_official.tex && bibtex aaai27_official && pdflatex -interaction=nonstopmode aaai27_official.tex && pdflatex -interaction=nonstopmode aaai27_official.tex && cd ../../..
python3 -m experiments.MaskShift.m19_aaai27_target_readiness
python3 -m experiments.MaskShift.m20_aaai27_preflight_conversion
python3 -m experiments.MaskShift.m21_aaai27_reproducibility_checklist
python3 -m experiments.MaskShift.m15_final_integrity_audit
python3 -m experiments.MaskShift.m5_main_track_audit
```

`PAPER.md` is the current prose source of truth and `paper/main.tex` is the
synchronized LaTeX submission draft. `m4_paper_ready.py` is an earlier scaffold
packager; do not rerun it as the final manuscript generator unless its template
is updated first.

## Milestones

- M0: mask generator suite, matched-rate audit, research brief, source corpus.
- M1: MCAR-trained forecast-risk and model-rank audit under operational masks.
- M2: minimal topology/mechanism-typed correction audit.
- M3: claim-family statistical tests with BH-FDR.
- M4: paper-ready artifact package.
- M5: strict strong-conference readiness audit.
- M6: lite neural proxy sweep with CUDA/MPS/CPU device handling.
- M7: multi-rate severity curves.
- M8: mechanism decomposition excluding retirement-dominated effects.
- M9: official TSLib PatchTST/TimeXer architecture reproduction.
- M10: submission hardening with seed CIs, corrected severity metrics, tables,
  and missing-aware-baseline audit.
- M11: official ChannelTokenFormer_missing adaptation under the MaskShift
  encoder-mask protocol. Requires `external/ChannelTokenFormer`.
- M12: official S4M adaptation under the MaskShift encoder-mask protocol.
  Requires `external/S4M`; current run is a reduced three-seed contrastive
  baseline and includes a device-port patch for MPS/CPU compatibility.
- M13: hierarchy-aware bootstrap over lightweight variants and test windows.
  This strengthens uncertainty reporting for aggregate M1 claims and keeps
  rank-instability evidence anchored to M9/M10.
- M14: larger reduced S4M scale validation with 16 channels, 64 train windows,
  48 test windows, and three seed offsets. It checks that the M12
  negative/contrastive S4M result is not only an eight-channel artifact.
- M15: final local integrity audit. It blocks on citation graph hygiene,
  BibTeX hygiene, numerical traceability, artifact presence, code compilation,
  device-selection rules, and ARS AI-research failure-mode coverage.
- M16: official TSLib PatchTST/TimeXer full-dataset coverage. It adds
  Traffic and AirConvection coverage under the MaskShift protocol so mixed
  datasets are not only supported by lightweight-model evidence.
- M17: submission supplement and rebuttal-ready package. It generates
  `paper/supplement.tex`, `paper/supplement.pdf`,
  `docs/DATASET_MECHANISM_CARDS.md`, `docs/REBUTTAL_PLAYBOOK.md`, and
  `tables/m17_reviewer_response_matrix.md` so reviewer concerns are mapped to
  audited evidence before submission.
- M18: submission policy and venue-readiness package. It generates
  `docs/SUBMISSION_STATEMENTS.md`, `docs/VENUE_READINESS_AUDIT.md`,
  `tables/m18_policy_readiness_table.md`, and
  `paper/submission_statements.tex`/`paper/submission_statements.pdf` so data,
  code, reproducibility, ethics, AI-assistance, anonymity, and target-template
  caveats are audited before upload.
- M19: AAAI-27 target-readiness package. It generates
  `docs/AAAI27_TARGET_READINESS.md`,
  `docs/AAAI27_REPRODUCIBILITY_CHECKLIST_DRAFT.md`,
  `docs/AAAI27_PHASE_REVIEW_RESPONSE_PLAN.md`,
  `tables/m19_aaai27_gap_table.md`, and
  `paper/aaai27_readiness.tex`/`paper/aaai27_readiness.pdf`. It records that
  the science package and official-template PDF are ready, while the
  OpenReview submission-system/form workflow remains the upload boundary.
- M20: AAAI-27 preflight and official-template conversion. It generates
  `docs/AAAI27_PREFLIGHT_FORMAT_AUDIT.md`,
  `tables/m20_aaai27_preflight_table.md`, and
  `paper/aaai27_preflight.tex`/`paper/aaai27_preflight.pdf`, plus
  `paper/aaai27_official.tex`/`paper/aaai27_official.pdf` using the official
  `aaai2027` kit. Both builds are 5 pages.
- M21: official AAAI-27 reproducibility checklist fill. It generates
  `docs/AAAI27_REPRODUCIBILITY_CHECKLIST_FILLED.md`,
  `paper/aaai27_reproducibility_checklist.tex`, and
  `paper/aaai27_reproducibility_checklist.pdf` from the official kit with 31
  filled answers and zero remaining question placeholders.

## Claim Scope

This package now supports a benchmark/theory route: H1/H2 are the main claims,
while the typed-head result is a diagnostic ablation because H3 is mixed. A
strong main-track submission should preserve this narrow scope: no SOTA claim,
no new-method claim, no full official benchmark-protocol claim, and no
exhaustive missing-aware baseline claim beyond the current reduced CTF/S4M
MaskShift-protocol adaptations plus M14 scale validation and M16 full-dataset
official TSLib coverage. M5 now requires M15, M16, M17, M18, M19, M20, and M21 to pass before
reporting `STRONG_CONFERENCE_READY`.
