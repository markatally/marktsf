# MaskShift Submission Checklist

## Current Verdict

MaskShift is strong-conference submission-ready as a benchmark/theory paper, not as a method paper. The supported thesis is: matched missing rate does not certify forecasting robustness under deployment missingness mechanisms, and model rankings can reverse when the conditional mask mechanism shifts.

M18 adds a target-agnostic submission policy pack. The scientific package is ready for review, and the final upload workflow now mainly requires mapping the policy statements into the selected venue's submission fields.

M19 selects AAAI-27 as the current strong-conference target for the user's near-term submission window. AAAI-27 submission opens on June 24, 2026, with abstracts due July 21 and full papers due July 28. The M19 dossier is complete; the remaining upload blocker is operational rather than scientific or formatting-related: wait for the paper submission site and complete the official reproducibility checklist fields.

M20 adds both a conservative AAAI-style preflight and an official `aaai2027` anonymous submission-template build. Both compile to 5 pages with wide result tables promoted to double-column floats.

M21 fills the official AAAI-27 reproducibility checklist locally from the author kit: 31 answers are present, no question placeholders remain, and the checklist PDF compiles. The OpenReview checklist fields still need to be completed in the submission system when it opens.

## Must-Keep Scope

- Claim MaskShift as an audit benchmark plus theory note.
- Say "official-architecture adaptation" for PatchTST/TimeXer, not "full official benchmark reproduction."
- Say "official missing-aware architecture adaptation" for ChannelTokenFormer_missing, not "full CTF practical benchmark reproduction."
- Say "official S4M architecture adaptation" for M12, not "full S4M benchmark reproduction."
- Present the typed/topology head as a negative diagnostic because H3 fails.
- Report Traffic and AirConvection as mixed evidence, not hidden failures.
- State that ChannelTokenFormer_missing is integrated in M11 and S4M is integrated in M12; both are reduced MaskShift-protocol adaptations.

## Evidence Package

| Claim | Evidence | Status |
| --- | --- | --- |
| Matched rate is insufficient | M1/M3 plus M10 multi-seed Weather/Electricity table | Supported with limits |
| Model ranking can reverse | M9/M10 PatchTST/TimeXer worst tau -1.0 on Weather/Electricity over three seed offsets | Supported for official-architecture adaptation |
| Official full-dataset coverage | M16 PatchTST/TimeXer coverage on Traffic and AirConvection | Integrated; mixed/negative rank evidence reported rather than hidden |
| Missing-aware architecture coverage | M11 official ChannelTokenFormer_missing adaptation over three seed offsets | Integrated; mixed evidence |
| Missing-aware contrastive baseline | M12 official S4M adaptation on Weather/Electricity over three seed offsets | Integrated; negative/robust contrast under reduced protocol |
| S4M scale sensitivity | M14 larger reduced S4M run with 16 channels and doubled train/test windows | Integrated; still 0/3 gate seeds on Weather/Electricity |
| Final integrity | M15 local final integrity audit | PASS with zero blocking issues; now required by M5 |
| Supplement/rebuttal readiness | M17 supplement package, dataset/mechanism cards, rebuttal playbook, and reviewer response matrix | Integrated; now required by M5 and M15 |
| Submission policy/readiness | M18 submission statements, venue-readiness audit, policy table, and compiled policy PDF | Integrated; now required by M5 and M15 |
| AAAI-27 target readiness | M19 target-readiness audit, reproducibility checklist draft, phase-review response plan, gap table, and compiled target dossier | Integrated; now required by M5 and M15; upload blockers explicitly tracked |
| AAAI-27 preflight and official conversion | M20 two-column US-letter anonymous preflight, official `aaai2027` build, page-count audit, and compiled PDFs | Integrated; 5-page preflight and 5-page official-template pass |
| AAAI-27 official reproducibility checklist | M21 filled official checklist TeX/PDF and answer audit | Integrated; 31 answers, zero question placeholders, compiled PDF |
| Not retirement-only | M8 non-retirement decomposition | Supported on Weather/Electricity |
| Relative degradation ratios need correction | M7 plus M10 absolute/log/symmetric metrics | Supported |
| Aggregate-only statistical concern | M13 hierarchical bootstrap over variants/windows | Protocol complete; supports loss-delta uncertainty, rank instability remains mainly M9/M10 evidence |
| Typed head is not a method contribution | M2/H3 p=0.214 and mixed per-dataset effects | Negative result |

## Main Artifacts

- `PAPER.md`: prose source of truth.
- `paper/main.tex`: synchronized LaTeX submission draft.
- `paper/references.bib`: bibliography for the LaTeX draft.
- `tables/main_result_table.md`: M1 multi-seed table.
- `tables/m9_official_architecture_table.md`: M9/M10 official-architecture table.
- `tables/m11_ctf_missing_table.md`: M11 official missing-aware baseline table.
- `tables/m12_s4m_table.md`: M12 official S4M contrastive baseline table.
- `tables/m13_hierarchical_bootstrap_table.md`: M13 hierarchy-aware uncertainty table.
- `tables/m14_s4m_scale_table.md`: M14 larger reduced S4M scale-validation table.
- `tables/m15_integrity_table.md`: M15 final local integrity summary table.
- `tables/m16_official_tslib_full_coverage_table.md`: M16 official TSLib all-dataset coverage table.
- `tables/m17_reviewer_response_matrix.md`: M17 likely-reviewer-concern response matrix.
- `tables/m18_policy_readiness_table.md`: M18 submission policy readiness table.
- `tables/m19_aaai27_gap_table.md`: M19 AAAI-27 target-readiness gap table.
- `tables/m20_aaai27_preflight_table.md`: M20 AAAI-27 preflight page-pressure table.
- `tables/m7_corrected_robustness_table.md`: corrected severity table.
- `tables/claim_evidence_table.md`: claim wording guardrail.
- `docs/MISSING_AWARE_BASELINE_ATTEMPT.md`: audit trail updated after M11 CTF integration.
- `docs/FINAL_INTEGRITY_REPORT.md`: M15 final integrity report.
- `docs/DATASET_MECHANISM_CARDS.md`: M17 reviewer-facing dataset and mechanism cards.
- `docs/REBUTTAL_PLAYBOOK.md`: M17 scope-safe rebuttal playbook.
- `docs/SUBMISSION_STATEMENTS.md`: M18 data/code/reproducibility/ethics/AI/anonymity statements.
- `docs/VENUE_READINESS_AUDIT.md`: M18 target-agnostic venue-readiness audit.
- `docs/AAAI27_TARGET_READINESS.md`: M19 target-specific AAAI-27 readiness audit.
- `docs/AAAI27_REPRODUCIBILITY_CHECKLIST_DRAFT.md`: M19 reproducibility checklist draft for the AAAI form.
- `docs/AAAI27_REPRODUCIBILITY_CHECKLIST_FILLED.md`: M21 filled local official checklist answers.
- `docs/AAAI27_PHASE_REVIEW_RESPONSE_PLAN.md`: M19 response plan for two-phase review and AI-generated first-stage review.
- `docs/AAAI27_PREFLIGHT_FORMAT_AUDIT.md`: M20 preflight format and page-pressure audit.
- `paper/supplement.tex` and `paper/supplement.pdf`: M17 supplement package.
- `paper/submission_statements.tex` and `paper/submission_statements.pdf`: M18 policy/disclosure package.
- `paper/aaai27_readiness.tex` and `paper/aaai27_readiness.pdf`: M19 target-readiness dossier.
- `paper/aaai27_preflight.tex` and `paper/aaai27_preflight.pdf`: M20 anonymous two-column preflight.
- `paper/aaai27_official.tex` and `paper/aaai27_official.pdf`: M20 official `aaai2027` anonymous submission-template build.
- `paper/aaai27_reproducibility_checklist.tex` and `paper/aaai27_reproducibility_checklist.pdf`: M21 local official checklist build.
- `paper/aaai2027.sty`, `paper/aaai2027.bst`, and `paper/AuthorKit27.zip`: official AAAI-27 author kit files.

## Pre-Submission P0

- Increase M10 seed count beyond three for Weather/Electricity if compute allows.
- Add the full S4M benchmark protocol if compute allows; current S4M evidence includes reduced three-seed M12 plus larger reduced M14, but not full original-protocol reproduction.
- Upload `paper/aaai27_official.pdf` after the AAAI paper submission site opens on 2026-06-24.
- Complete the official AAAI reproducibility checklist fields using `docs/AAAI27_REPRODUCIBILITY_CHECKLIST_FILLED.md`.
- Map `docs/SUBMISSION_STATEMENTS.md` into the target venue's required data/code/reproducibility/ethics/AI-disclosure fields.

## Pre-Submission P1

- Expand M9/M16 beyond PatchTST/TimeXer on Traffic/AirConvection if runtime permits, even if results are mixed.
- Expand M13 with more bootstrap strata or mixed-effects analysis if time permits; current variant/window bootstrap is complete.
- Cross-check the OpenReview checklist fields against `docs/AAAI27_REPRODUCIBILITY_CHECKLIST_FILLED.md` immediately before upload.

## Reproduction Commands

Run from the repository root:

External repositories are ignored by git and must exist locally for M9/M11/M12:

```bash
git clone --depth 1 https://github.com/thuml/Time-Series-Library external/TSLib
git clone --depth 1 https://github.com/jinkwan1115/ChannelTokenFormer external/ChannelTokenFormer
git clone --depth 1 https://github.com/WINTERWEEL/S4M.git external/S4M
```

Audited revisions: TSLib `4e938a1`, ChannelTokenFormer `b1c100e`, S4M `a718823`.

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

`m4_paper_ready.py` predates the final narrative and should not be used as the final manuscript generator unless its template is updated.
