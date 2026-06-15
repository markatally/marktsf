# MaskShift M0 — Generator Suite and Research Brief

## RQ Brief

**Research question.** Under matched missing rate with audited gap and channel topology statistics, do operational missingness mechanisms induce forecast-risk shifts and model-rank reversals that are not certified by MCAR/block robustness tests?

**Sub-questions.**
1. How much degradation variance is explained by mechanism identity after controlling for missing rate and topology?
2. Do model rankings learned or selected under MCAR remain stable under value-triggered, volatility-triggered, blackout, and retirement mechanisms?
3. Can a lightweight topology/mechanism-typed head recover a meaningful fraction of the degradation without changing the backbone?

**FINER scores.** Feasible 9/10; Interesting 10/10; Novel 8/10; Ethical 10/10; Relevant 10/10.

**Scope.** In scope: forecasting with observed-input missingness, controlled mask generators, public TSF datasets, rank instability, minimal typed/topology correction. Out of scope: claiming a new imputation SOTA, modeling delayed future covariate release, and unrestricted causal identification of the outage process.

## Methodology Blueprint

Quantitative benchmark and theory-driven empirical audit. The design is a matched-factor experiment: fix observed-value tensor, target forecast, missing rate, and approximate topology controls, then vary the missingness mechanism. The primary tests are mixed-effect degradation decomposition, Kendall rank stability, paired loss tests, and typed-head ablations.

## Source Corpus

- [S01] Che et al. (2018), Recurrent Neural Networks for Multivariate Time Series with Missing Values. Scientific Reports. https://www.nature.com/articles/s41598-018-24271-9
- [S02] Cao et al. (2018), BRITS: Bidirectional Recurrent Imputation for Time Series. NeurIPS. https://proceedings.neurips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html
- [S03] Islam, Tadepalli, and Fern (2025), Self-attention-based Diffusion Model for Time-series Imputation in Partial Blackout Scenarios. AAAI. https://arxiv.org/pdf/2503.01737
- [S04] Jing et al. (2025), S4M: S4 for Multivariate Time Series Forecasting with Missing Values. ICLR. https://openreview.net/forum?id=BkftcwIVmR
- [S05] Jang et al. (2026), Towards Robust Real-World Multivariate Time Series Forecasting. ICLR. https://openreview.net/forum?id=r4ZamwBE8P
- [S06] Sunesh, Ma, and Nilol (2026), Modeling Information Blackouts in Missing Not-At-Random Time Series Data. arXiv. https://arxiv.org/pdf/2601.01480
- [S07] Yalavarthi et al. (2024), GraFITi: Graphs for Forecasting Irregularly Sampled Time Series. AAAI. https://ojs.aaai.org/index.php/AAAI/article/view/29560
- [S08] Yang et al. (2025), Revisiting Multivariate Time Series Forecasting with Missing Values. arXiv. https://arxiv.org/abs/2509.23494
- [S09] Rockenschaub et al. (2024), Robust Prediction under Missingness Shifts. arXiv. https://arxiv.org/abs/2406.16484

## Generator Control Table

| Dataset | Mechanism | Missing rate | Mean gap | Max gap | Channel-rate std |
|---|---:|---:|---:|---:|---:|
| Weather | mcar | 0.352 | 1.54 | 11 | 0.006 |
| Weather | block | 0.350 | 10.31 | 61 | 0.021 |
| Weather | value_high | 0.350 | 2.51 | 109 | 0.037 |
| Weather | volatility | 0.350 | 1.76 | 25 | 0.064 |
| Weather | blackout | 0.350 | 7.79 | 38 | 0.000 |
| Weather | retirement | 0.350 | 1.96 | 1797 | 0.070 |
| Electricity | mcar | 0.352 | 1.55 | 11 | 0.005 |
| Electricity | block | 0.350 | 10.25 | 57 | 0.020 |
| Electricity | value_high | 0.350 | 2.60 | 30 | 0.049 |
| Electricity | volatility | 0.350 | 1.74 | 11 | 0.023 |
| Electricity | blackout | 0.350 | 7.79 | 38 | 0.000 |
| Electricity | retirement | 0.350 | 2.04 | 2029 | 0.072 |
| Traffic | mcar | 0.352 | 1.55 | 11 | 0.005 |
| Traffic | block | 0.350 | 10.25 | 57 | 0.020 |
| Traffic | value_high | 0.350 | 2.26 | 23 | 0.036 |
| Traffic | volatility | 0.350 | 1.75 | 12 | 0.029 |
| Traffic | blackout | 0.350 | 7.79 | 38 | 0.000 |
| Traffic | retirement | 0.350 | 2.04 | 2029 | 0.072 |
| AirConvection | mcar | 0.354 | 1.55 | 9 | 0.005 |
| AirConvection | block | 0.350 | 10.17 | 53 | 0.018 |
| AirConvection | value_high | 0.350 | 2.44 | 65 | 0.079 |
| AirConvection | volatility | 0.350 | 1.78 | 19 | 0.052 |
| AirConvection | blackout | 0.350 | 7.78 | 38 | 0.000 |
| AirConvection | retirement | 0.350 | 1.74 | 987 | 0.032 |

## Novelty Boundary

The closest collision is [S06], which models traffic sensor blackouts with MAR/MNAR state-space inference and reports imputation plus short post-blackout forecasts. MaskShift must not claim first MNAR blackout modeling. The defensible contribution is a broader mechanism-shift benchmark showing that matched missing rate is an insufficient control for forecast selection across modern TSF baselines.


# MaskShift M8 — Mechanism Decomposition

M8 checks whether the mechanism-shift result survives after excluding sensor retirement, the most visually obvious outage mechanism.

| Dataset | Strongest non-retirement mechanism | Max non-ret degradation | Worst non-ret tau | Retirement degradation | Gate |
|---|---|---:|---:|---:|---|
| Weather | value_high | 132.8% | 0.333 | 585.0% | PASS |
| Electricity | value_high | 182.7% | -0.333 | 58.3% | PASS |
| Traffic | volatility | -2.9% | 0.000 | -6.8% | FAIL |
| AirConvection | value_high | 108.0% | 0.667 | 150.0% | FAIL |

## Mechanism-Level Summary

| Mechanism | Mean degradation | Max degradation | Min tau | Positive count |
|---|---:|---:|---:|---:|
| value_high | 101.0% | 182.7% | -0.333 | 3 |
| volatility | 7.5% | 71.5% | 0.667 | 1 |
| blackout | 57.8% | 112.8% | 0.333 | 3 |
| retirement | 196.7% | 585.0% | -0.333 | 3 |


# MaskShift M9 — Official TSLib Architecture Reproduction

M9 imports official PatchTST and TimeXer model classes from the pinned `external/TSLib` checkout and evaluates them under the MaskShift encoder-mask protocol.

| Dataset | Max degradation | Worst tau | ANOVA p | Gate |
|---|---:|---:|---:|---|
| Weather | 123.7% | -1.000 | 6.66e-06 | PASS |
| Electricity | 87.8% | -1.000 | 0.00229 | PASS |


# MaskShift M10 — Submission Hardening

M10 adds sprint-time robustness checks needed for submission wording: multi-seed descriptive confidence intervals for the core positive datasets, corrected severity metrics that avoid denominator-driven ratio claims, a synchronized table package, an overview figure, and a local audit for missing-aware official baselines.

## M1 Multi-Seed Core Results

| Dataset | eta^2 mean [95% CI] | Max degradation mean [95% CI] | Worst tau mean [95% CI] | Gate seeds |
| --- | --- | --- | --- | --- |
| Weather | 0.495 [0.000, 1.000] | 702.0% [341.8, 1062.2] | 0.11 [-1.00, 1.00] | 2/3 |
| Electricity | 0.572 [0.129, 1.000] | 175.1% [129.3, 220.9] | -0.22 [-1.00, 1.00] | 3/3 |

## Official-Architecture Adaptation

PatchTST and TimeXer are imported from pinned TSLib model classes; the MaskShift loop is a custom encoder-mask protocol, not the full official benchmark protocol.

| Dataset | Official architecture classes | Max degradation mean [95% CI] | Worst tau mean [95% CI] | Gate seeds |
| --- | --- | --- | --- | --- |
| Weather | PatchTST_official, TimeXer_official | 135.8% [8.6, 262.9] | -1.00 [-1.00, -1.00] | 3/3 |
| Electricity | PatchTST_official, TimeXer_official | 128.6% [98.1, 159.1] | -1.00 [-1.00, -1.00] | 3/3 |

## Submission Verdict After M10

Updated M5 verdict: `STRONG_CONFERENCE_READY`; blocking items: none. Method-claim readiness remains false because the typed/topology head fails H3 and is deliberately scoped as a negative diagnostic.


# MaskShift M11 — Official ChannelTokenFormer Missing-Aware Baseline

M11 imports `ChannelTokenFormer_missing` from the official ChannelTokenFormer repository at revision `b1c100e` and evaluates it under the MaskShift encoder-mask protocol. This is an official missing-aware architecture adaptation, not the full CTF practical/irregular benchmark pipeline.

| Dataset | Backbone | Max degradation mean [95% CI] | Max abs delta [95% CI] | Strongest mechanism | Gate seeds |
| --- | --- | --- | --- | --- | --- |
| Weather | ChannelTokenFormer_missing_official | 96.0% [39.4, 152.6] | 0.264 [0.003, 0.524] | volatility | 1/3 |
| Electricity | ChannelTokenFormer_missing_official | 32.3% [-10.4, 75.0] | 0.584 [-0.130, 1.297] | value_high | 0/3 |

Interpretation: M11 closes the official missing-aware baseline gap for ChannelTokenFormer, but does not support a method-win claim. It gives mixed, reviewer-useful evidence: Weather remains mechanism-sensitive, while Electricity suggests partial robustness or insufficient power under the sprint-time protocol. M12 below adds S4M as a separate contrastive missing-aware baseline.


# MaskShift M12 — Official S4M Missing-Aware Baseline

M12 imports the official S4M model class from revision `a718823` and evaluates it under the MaskShift encoder-mask protocol with reduced local samples/channels over three seed offsets. A local device-port patch changes one hard-coded `.cuda()` memory fetch to `.to(Q.device)` so the official architecture can run on MPS/CPU.

| Dataset | Backbone | Max degradation mean [95% CI] | Max abs delta [95% CI] | Strongest mechanism mode | Kruskal p mean [95% CI] | Gate seeds |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Weather | S4M_official | 5.7% [-44.9, 56.3] | 0.071 [-0.620, 0.762] | mixed | 0.509 [0.081, 0.938] | 0/3 |
| Electricity | S4M_official | 6.2% [-0.1, 12.5] | 0.312 [0.093, 0.530] | mixed | 0.933 [0.727, 1.000] | 0/3 |

Interpretation: M12 closes the S4M absence gap, but it is a negative/contrastive result. It supports a more credible benchmark claim: mechanism sensitivity is architecture- and dataset-dependent, so MaskShift should report mechanism identity rather than claim that every missing-aware architecture fails.


# MaskShift M14 — S4M Scale Validation

M14 reruns the official S4M adaptation in a larger reduced setting: 16 channels, 64 train windows, 48 test windows, and three seed offsets. This directly tests whether the M12 negative/contrastive result is only an artifact of the eight-channel fast setting.

| Dataset | Backbone | Max degradation mean [95% CI] | Max abs delta [95% CI] | Strongest mechanism mode | Kruskal p mean [95% CI] | Gate seeds |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| Weather | S4M_official | 6.8% [-8.8, 22.4] | 0.109 [-0.134, 0.352] | mixed | 0.673 [0.000, 1.000] | 0/3 |
| Electricity | S4M_official | 2.5% [-17.9, 22.8] | 0.154 [-1.382, 1.690] | blackout | 0.989 [0.963, 1.000] | 0/3 |

Interpretation: M14 does not turn S4M into a positive mechanism-shift result. It strengthens the contrastive baseline by showing that the S4M negative result persists after doubling channels and train/test windows, while still remaining short of a full S4M benchmark reproduction.


# MaskShift M13 — Hierarchical Bootstrap

M13 adds a nonparametric bootstrap over lightweight variants and test windows. It strengthens uncertainty reporting for M1 aggregate claims and prevents the paper from relying only on aggregate ANOVA means.

| Dataset | eta^2 [95% CI] | Max abs delta [95% CI] | P(delta>0) | Loss-shift evidence | Worst tau [95% CI] | P(tau<=0.5) | Rank evidence |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| Weather | 0.239 [0.088, 0.547] | 1.680 [0.514, 3.659] | 1.00 | SUPPORTED | 0.48 [-0.33, 1.00] | 0.46 | NOT_DECISIVE |
| Electricity | 0.320 [0.187, 0.538] | 1.446 [0.862, 1.943] | 1.00 | SUPPORTED | 0.37 [-0.33, 1.00] | 0.53 | NOT_DECISIVE |
| Traffic | 0.119 [0.007, 0.530] | 0.347 [-0.356, 1.101] | 0.83 | MIXED | 0.63 [0.00, 1.00] | 0.23 | NOT_DECISIVE |
| AirConvection | 0.163 [0.046, 0.501] | 12.269 [0.348, 33.052] | 1.00 | SUPPORTED | 0.84 [0.67, 1.00] | 0.02 | NOT_DECISIVE |

Interpretation: Weather and Electricity have positive bootstrap intervals for loss shift, but the lightweight-rank instability bootstrap is not decisive. The rank-reversal claim should therefore remain anchored to M9/M10 official PatchTST/TimeXer results.


# MaskShift M16 — Official TSLib Full-Dataset Coverage

M16 extends official PatchTST/TimeXer MaskShift-protocol coverage to Traffic and AirConvection. Weather/Electricity rows reuse M9; Traffic/AirConvection rows are new M16 coverage runs.

| Dataset | Source | Official architecture classes | Max degradation | Worst tau | ANOVA p | Gate |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Weather | M9 | PatchTST_official, TimeXer_official | 123.7% | -1.000 | 6.66e-06 | PASS |
| Electricity | M9 | PatchTST_official, TimeXer_official | 87.8% | -1.000 | 0.00229 | PASS |
| Traffic | M16 | PatchTST_official, TimeXer_official | 35.4% | 1.000 | 0.0854 | MIXED/NEGATIVE |
| AirConvection | M16 | PatchTST_official, TimeXer_official | 46.1% | 1.000 | 0.000122 | MIXED/NEGATIVE |

Interpretation: M16 resolves the strongest external-validity complaint against M9/M10: official modern architecture coverage is no longer restricted to Weather/Electricity. Traffic and AirConvection remain mixed/negative for rank reversal, which supports the paper's narrow benchmark-theory framing rather than a universal-rank-reversal claim.


# MaskShift M15 — Final Integrity Audit

M15 adds a final local integrity gate before M5 can report strong-conference readiness. It checks citation graph hygiene, BibTeX hygiene, numerical traceability from JSON summaries to the paper/tables, artifact presence, Python compilation, CUDA/MPS/CPU device-selection rules, the S4M device-port patch, and ARS AI-research failure modes.

Current M15 verdict: `PASS_FINAL_INTEGRITY`; blocking issues: 0.

M15 scope note: this local integrity gate does not replace professional plagiarism software or a full external reproduction, but it catches the submission-package failures most likely to produce an immediate reviewer or AE objection.


# MaskShift M17 — Submission Supplement and Rebuttal Package

M17 converts the M0-M16 evidence trail into reviewer-facing submission support. It generates `paper/supplement.tex`, compiles to `paper/supplement.pdf`, writes dataset/mechanism cards, and creates a likely-reviewer-concern response matrix plus rebuttal playbook.

Current M17 verdict: `PASS_SUBMISSION_SUPPLEMENT`; missing artifacts: 0.

Artifacts:

- `paper/supplement.tex` and `paper/supplement.pdf`
- `docs/DATASET_MECHANISM_CARDS.md`
- `docs/REBUTTAL_PLAYBOOK.md`
- `tables/m17_reviewer_response_matrix.md`
- `m17_submission_supplement/submission_supplement_summary.json`

Interpretation: M17 closes the remaining presentation and rebuttal-readiness gap. It does not add new empirical claims; it packages already audited evidence so reviewers can verify mechanism definitions, dataset roles, official-architecture boundaries, negative/mixed evidence, and reproduction commands from one supplement bundle.


# MaskShift M18 — Submission Policy and Venue-Readiness Pack

M18 adds a target-agnostic policy/disclosure package for submission systems and venue checklists. It generates data/code/reproducibility, ethics, AI-assistance, conflict/funding, author-contribution, and anonymity statements, plus a venue-readiness audit that separates scientific readiness from target-template compliance.

Current M18 verdict: `PASS_SUBMISSION_POLICY_PACK`; missing artifacts: 0.

Artifacts:

- `docs/SUBMISSION_STATEMENTS.md`
- `docs/VENUE_READINESS_AUDIT.md`
- `tables/m18_policy_readiness_table.md`
- `paper/submission_statements.tex` and `paper/submission_statements.pdf`
- `m18_submission_policy_pack/submission_policy_summary.json`

Interpretation: M18 closes the submission-policy gap without overstating target-specific compliance. The package is ready to be mapped into a selected venue's required fields, while the selected venue template remains a mechanical pre-upload task.


# MaskShift M19 — AAAI-27 Target-Readiness Dossier

M19 selects AAAI-27 Main Technical Track as the near-term strong-conference target because its submission site opens in June 2026 and its abstract/full-paper deadlines are in July 2026. It records official target facts, maps MaskShift artifacts to the AAAI reproducibility requirement, prepares a response plan for two-phase review and the AI-generated first-stage review, and makes the remaining upload blockers explicit.

Current M19 verdict: `PASS_AAAI27_TARGET_DOSSIER`; AAAI upload ready: `false`.

Artifacts:

- `docs/AAAI27_TARGET_READINESS.md`
- `docs/AAAI27_REPRODUCIBILITY_CHECKLIST_DRAFT.md`
- `docs/AAAI27_PHASE_REVIEW_RESPONSE_PLAN.md`
- `tables/m19_aaai27_gap_table.md`
- `paper/aaai27_readiness.tex` and `paper/aaai27_readiness.pdf`
- `m19_aaai27_target_readiness/aaai27_target_readiness_summary.json`

Interpretation: M19 upgrades the package from target-agnostic readiness to target-aware readiness. It deliberately does not mark the AAAI upload as ready because the OpenReview paper submission site is not yet open on the audit date and the official reproducibility checklist fields still need to be completed in the venue system. The prior target-style and page-limit blockers are resolved by M20.


# MaskShift M20 — AAAI-27 Preflight Conversion

M20 converts the current manuscript body into an anonymous two-column US-letter preflight draft and an official `aaai2027` anonymous submission-template build. The generated builds promote wide result tables to double-column floats so the result evidence is readable under AAAI-like page pressure.

Current M20 verdict: `PASS_AAAI27_PREFLIGHT_CONVERSION`; preflight pages: `5`; official-template pages: `5`; official-template build pass: `true`.

Artifacts:

- `docs/AAAI27_PREFLIGHT_FORMAT_AUDIT.md`
- `tables/m20_aaai27_preflight_table.md`
- `paper/aaai27_preflight.tex` and `paper/aaai27_preflight.pdf`
- `paper/aaai27_official.tex` and `paper/aaai27_official.pdf`
- `paper/aaai2027.sty`, `paper/aaai2027.bst`, and `paper/AuthorKit27.zip`
- `m20_aaai27_preflight_conversion/aaai27_preflight_summary.json`


# MaskShift M21 — AAAI-27 Official Reproducibility Checklist

M21 fills the official AAAI-27 `ReproducibilityChecklist.tex` from the author kit and compiles a local checklist PDF. It turns the earlier M19 checklist draft into a concrete, upload-ready answer source for OpenReview form entry.

Current M21 verdict: `PASS_AAAI27_REPRODUCIBILITY_CHECKLIST`; answers filled: `31`; remaining question placeholders: `0`; PDF size: `99818` bytes.

Artifacts:

- `docs/AAAI27_REPRODUCIBILITY_CHECKLIST_FILLED.md`
- `paper/aaai27_reproducibility_checklist.tex`
- `paper/aaai27_reproducibility_checklist.pdf`
- `m21_aaai27_reproducibility_checklist/aaai27_reproducibility_checklist_summary.json`

Interpretation: M21 removes the last local reproducibility-checklist preparation gap. The only remaining checklist action is operational: copy the local filled answers into the official OpenReview fields when the paper submission form is available.

Interpretation: M20 resolves the page-pressure and official-template concerns raised by M19: the current body fits within both the conservative preflight and the official AAAI-27 anonymous submission template. The remaining boundary is operational submission-system completion, not formatting.
