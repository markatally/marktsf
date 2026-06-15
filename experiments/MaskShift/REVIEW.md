# MaskShift ARS Reviewer Pass v1

## Field Analysis

| Dimension | Assessment |
|---|---|
| Primary discipline | Machine learning for time-series forecasting |
| Secondary disciplines | Missing-data statistics; robust evaluation; sensor/telemetry systems |
| Research paradigm | Quantitative benchmark + theory sketch |
| Methodology type | Controlled benchmark, statistical audit, lightweight model ablation |
| Target tier | Strong ML/forecasting venue candidate only after official baselines |
| Maturity | Historical early-stage assessment; superseded by later M15/M16-ready draft |

Recommended reviewer configuration:
- EIC: senior area chair for ML time-series benchmarking, focused on novelty, benchmark value, and claim scope.
- R1 methodology: statistical ML reviewer specializing in benchmark validity, paired testing, FDR, and leakage controls.
- R2 domain: missing-data and irregular-time-series forecasting reviewer, focused on collision with S4M, ChannelTokenFormer, SADI, CRIB, and information-blackout work.
- R3 perspective: applied sensor/telemetry systems reviewer, focused on operational outage semantics and deployment realism.
- Devil's Advocate: adversarial reviewer focused on whether the current evidence is just a mask-augmentation stress test.

## EIC Review

**Recommendation: Major Revision.**

The paper has a timely and defensible benchmark/theory thesis: matched missing rate is not a sufficient robustness certificate for operational missingness. The H1/H2 evidence is promising, and the paper correctly avoids claiming a new missing-data architecture. However, a strong-conference submission cannot rely on Ridge-style lightweight models plus "lite" neural proxies. The current M6 explicitly states that it is not an official PatchTST/TimeXer/S4M/ChannelTokenFormer reproduction. That is a blocking fit issue for a top ML venue because the closest papers are architecture-heavy and would be reviewed by authors/readers expecting modern TSF baselines.

Strengths:
1. The novelty boundary is clear: mechanism shift and model selection instability, not first missing-value forecaster.
2. The M5 audit is unusually honest and prevents overclaiming the failed typed-head result.
3. H1/H2 are backed by FDR and rank-instability evidence on multiple datasets.

Major weaknesses:
1. Official baseline reproduction is absent. This must be resolved before submission.
2. The theory is currently a sketch, not a theorem/proof that can carry a main-track paper.
3. The abstract still lists typed correction as a contribution even though H3 failed; this should be reframed as an ablation/negative result.

## Methodology Review

**Recommendation: Major Revision.**

The benchmark structure is coherent: chronological splits, pre-origin masks, matched missing rates, BH-FDR, and rank-based stability all align with the research question. The biggest problem is external validity of the model set. A benchmark paper must show the phenomenon is not an artifact of the local feature construction. M6-lite helps but does not satisfy field norms for modern TSF comparison.

Required fixes:
1. Add an official-code or faithful reimplementation sweep for at least PatchTST and one missing-specific competitor (S4M, ChannelTokenFormer if runnable, or CRIB/MTSF-M if official code is available).
2. Add severity curves at 10%, 20%, 35%, and optionally 50% missingness; report degradation AUC rather than a single stress point.
3. Replace ANOVA on aggregate MSE means with a mixed-effects or hierarchical bootstrap analysis where feasible; keep current aggregate eta as a descriptive effect size.
4. Add confidence intervals for eta, rank tau, degradation, and M6 neural results.

## Domain Review

**Recommendation: Major Revision.**

The related-work positioning is mostly correct, but too compressed for a strong venue. The paper needs a sharper distinction from SADI, S4M, ChannelTokenFormer, CRIB/MTSF-M, and the 2026 information-blackout preprint. It should also cite robust prediction under missingness shifts as a statistical anchor and explicitly state how MaskShift differs from missing-data imputation sensitivity analysis.

Required fixes:
1. Add CRIB/MTSF-M and robust prediction under missingness shifts to the source corpus.
2. Expand the related-work table into claim-level differences: task, missingness mechanism, evaluation target, and whether model ranking under matched-rate mechanism shift is tested.
3. Stop calling the typed head a "contribution" unless H3 is repaired; use "diagnostic ablation" instead.

## Perspective Review

**Recommendation: Minor-to-Major Revision.**

From an applied systems perspective, the strongest part is the operational semantics of masks. The weakness is that mechanism labels may not be available in deployed systems. The paper should emphasize topology-observable signals and clearly distinguish known-cause outages from inferred-cause outages.

Required fixes:
1. Add a deployment taxonomy: known mechanism label, inferred mechanism label, topology-only, and unknown mechanism.
2. Explain how an engineer would instantiate value-triggered, volatility-triggered, blackout, and retirement masks from logs.
3. Add a benchmark card for each dataset describing why each synthetic mechanism is plausible for that domain.

## Devil's Advocate Stress Test

Critical challenge: the current paper can be attacked as "just a mask stress-test suite with insufficient baselines." The best counterargument is that benchmark papers are publishable when they expose a systematic evaluation blind spot, but only if they test the models the community actually uses.

Other vulnerabilities:
1. The paper says "matched gap length and channel coverage," but M0 currently guarantees matched rate more strongly than matched topology. The text must not overstate topology matching.
2. Retirement mechanisms dominate some degradation effects; reviewers may argue the result is obvious. The paper needs mechanism-specific decomposition excluding retirement.
3. H3 failed. Any sufficiency theorem should not imply the implemented typed head works empirically.

## Editorial Synthesis

**Decision: Major Revision.**

The paper should proceed as a benchmark/theory main-track candidate, not as a method paper. The revision must either (a) add official modern baselines and severity curves, then submit with typed correction as a secondary negative/partial finding, or (b) narrow to a workshop/benchmark track.

Priority roadmap:
1. **P0 Official baselines**: replace or supplement M6-lite with official PatchTST plus one missing-specific/current competitor.
2. **P0 Claim scope**: rewrite abstract/contributions so H3 is not a headline contribution.
3. **P1 Severity curves**: add multi-rate degradation AUC and mechanism-specific decomposition excluding retirement.
4. **P1 Related work**: add CRIB/MTSF-M, robust prediction under missingness shifts, and claim-level comparison table.
5. **P2 Theory**: turn the theory sketch into at least a formal proposition with assumptions and proof outline.

---

# MaskShift ARS Reviewer Pass v2

## Revision Delta

| Prior concern | Revision response | Status |
|---|---|---|
| Official baselines absent | Added M9 importing official PatchTST and TimeXer model classes from pinned TSLib revision `4e938a1`; both Weather and Electricity pass mechanism-shift gates. | Resolved for submission draft |
| H3 overclaimed | Rewrote abstract/contributions: typed head is a diagnostic partial/negative result, not a method contribution. | Resolved |
| Severity curves missing | Added M7 multi-rate audit at 10%, 20%, 35%, and 50%. | Resolved |
| Retirement dominance | Added M8 non-retirement decomposition; Weather/Electricity pass without relying on retirement. | Resolved |
| Related-work gaps | Added CRIB/MTSF-M and robust prediction under missingness shifts to source corpus and paper text. | Resolved |
| Theory only sketched | Added Proposition 1 giving the squared-loss mechanism-shift excess-risk identity and tying H1/H2/H3 to the term. | Resolved for submission draft |

## EIC Re-Review

**Recommendation: Weak Accept / submission-ready benchmark-theory paper.**

The revised artifact now has a defensible strong-conference route. The key reason is that the thesis no longer depends on the failed typed-head method claim. The main claim is now narrower and stronger: matched missing rate does not certify forecast risk or model selection under operational missingness mechanisms. M1/M3 establish the claim statistically, M7 shows severity dependence, M8 rules out a retirement-only explanation, and M9 shows the effect on official TSLib PatchTST/TimeXer architectures.

Remaining camera-ready risks:
1. Add more seeds and confidence intervals before final submission if compute permits.
2. Add one missing-specific official architecture, preferably S4M or ChannelTokenFormer-compatible, if runnable before the deadline.
3. Expand the related-work comparison table into a full page if the target venue favors benchmark papers.

## Methodology Re-Review

The revised M5 audit is acceptable for a benchmark/theory submission:

- M0 matched-rate controls: pass.
- H1 mechanism factor under FDR: pass.
- H2 rank instability: pass.
- H3 typed correction: fail, correctly scoped as diagnostic.
- M6 lite neural proxy sweep: pass.
- M7 severity curves: pass.
- M8 non-retirement decomposition: pass.
- M9 official TSLib reproduction: pass.

The paper should not claim SOTA forecasting accuracy or a universal correction method. With that boundary, the empirical design is coherent.

## Final Editorial Synthesis

**Decision: Strong-conference submission-ready, benchmark/theory route.**

M5 verdict: `STRONG_CONFERENCE_READY`; blocking items: none.

The final manuscript should be submitted as a benchmark/theory paper about missingness-mechanism shift. The typed-head experiment should remain in the main paper as a useful negative/partial diagnostic because it increases credibility and prevents overclaiming.

---

# MaskShift ARS Reviewer Pass v3

## M10 Re-Review Delta

| Remaining concern | M10 response | Status |
|---|---|---|
| Single-seed core tables | Added three-seed descriptive CIs for Weather/Electricity M1 and M9 official-architecture adaptation. | Resolved for submission draft |
| Relative degradation ratio instability | Added absolute delta, log ratio, and symmetric relative delta tables. | Resolved |
| Missing-aware official baseline ambiguity | Added M11 with official ChannelTokenFormer_missing at revision `b1c100e`; S4M coverage was still open at this pass. | Resolved for CTF; S4M later resolved in v5 |
| Manuscript still scaffold-like | Expanded `PAPER.md` and synchronized `paper/main.tex` into a complete benchmark/theory submission draft. | Resolved |

## Final Recommendation

**Weak Accept / Submit, with narrow claim scope.**

The current package is suitable for a strong-conference submission if positioned as a benchmark/theory paper. It is not suitable as a new forecasting-method paper. The decisive evidence is the M9/M10 rank reversal under official TSLib PatchTST and TimeXer architecture classes, the M11 official ChannelTokenFormer_missing adaptation, the M8 non-retirement decomposition, and the M10 corrected severity reporting.

Residual risks:
1. At this pass, S4M coverage was still open; ChannelTokenFormer_missing was covered but only as a MaskShift protocol adaptation. Resolved in v5.
2. M10 CIs are descriptive three-seed intervals, not a full hierarchical inference analysis.
3. The official-architecture adaptation uses a custom MaskShift protocol and should never be described as a full official benchmark reproduction.

Updated M5 verdict after M10: `STRONG_CONFERENCE_READY`; blocking items: none; `method_claim_ready=false`.

---

# MaskShift ARS Reviewer Pass v4

## M11 Re-Review Delta

| Prior residual risk | M11 response | Status |
|---|---|---|
| Missing-aware official architecture absent | Cloned official ChannelTokenFormer and ran `ChannelTokenFormer_missing` under MaskShift over Weather/Electricity with three seed offsets. | Resolved for CTF |
| Reviewer might argue phenomenon is mask-agnostic model artifact | CTF_missing shows Weather sensitivity (96.0% mean max degradation) but weaker Electricity sensitivity. | Strengthened, mixed |
| Overclaim risk around CTF | Manuscript now says official missing-aware architecture adaptation, not full CTF benchmark reproduction. | Resolved |

## Final Main-Conference Assessment

**Decision: Weak Accept / Main-track submit, benchmark-theory route.**

M11 materially improves the package because the paper no longer relies only on standard TSF backbones plus a lite GRU-D proxy. The strongest claim is still not "MaskShift defeats missing-aware models." The stronger and more defensible claim is that even after adding an official missing-aware architecture, mechanism identity remains a benchmark factor that must be reported, and the evidence is model- and dataset-dependent.

Updated final risks:
1. At this pass, S4M coverage remained open. Resolved in v5.
2. All official-architecture runs are MaskShift-protocol adaptations with reduced samples.
3. The CTF result is mixed, so the paper should frame it as coverage and nuance, not as a headline positive effect.

Updated M5 verdict after M11: `STRONG_CONFERENCE_READY`; blocking items: none; `method_claim_ready=false`.

---

# MaskShift ARS Reviewer Pass v5

## M12 Re-Review Delta

| Prior residual risk | M12 response | Status |
|---|---|---|
| S4M absent | Cloned official S4M (`a718823`) and ran the official S4M model class under the MaskShift encoder-mask protocol on Weather/Electricity over three seed offsets. | Resolved as official S4M adaptation |
| Missing-aware evidence could look cherry-picked | S4M is a negative/contrastive result: Weather max degradation 5.7% [-44.9, 56.3], Electricity max degradation 6.2% [-0.1, 12.5], both 0/3 gate seeds. | Strengthened credibility |
| Device-specific official code failure | Patched one hard-coded `.cuda()` memory fetch in official S4M `Bank.py` to `.to(Q.device)` for CUDA/MPS/CPU compatibility; architecture and forward equations otherwise unchanged. | Resolved and disclosed |

## Final Main-Conference Assessment After M12

**Decision: Stronger Weak Accept / Main-track submit, benchmark-theory route.**

M12 removes the largest remaining baseline-completeness objection. The paper should not turn this into a universal positive claim. The more credible final position is architecture-dependent: PatchTST/TimeXer show rank reversal, CTF_missing shows mixed mechanism sensitivity, and S4M is comparatively robust across three reduced local seed offsets. This makes MaskShift look less like a cherry-picked stress test and more like an honest benchmark factor audit.

Updated final risks:
1. M12 uses reduced channels/samples and the MaskShift loop, so it is coverage, not a full S4M reproduction.
2. M10/M11/M12 still use sprint-time three-seed counts; full mixed-effects inference remains desirable.
3. The paper must preserve the benchmark/theory scope and avoid any claim that missing-aware architectures broadly fail.

Updated M5 verdict after M12: `STRONG_CONFERENCE_READY`; blocking items: none; `method_claim_ready=false`.

---

# MaskShift ARS Reviewer Pass v6

## M13 Re-Review Delta

| Prior residual risk | M13 response | Status |
|---|---|---|
| Aggregate-only statistical evidence | Added a hierarchy-aware bootstrap over lightweight variants and test windows. | Resolved as uncertainty-reporting protocol |
| Risk-shift uncertainty | Weather and Electricity have positive max absolute-delta bootstrap intervals: 1.680 [0.514, 3.659] and 1.446 [0.862, 1.943]. | Strengthened |
| Lightweight rank-instability overclaim | Bootstrap rank-instability probabilities are only 0.46 and 0.53 on Weather/Electricity. | Important limitation; rank reversal should rely on M9/M10 |

## Final Main-Conference Assessment After M13

**Decision: Stronger Weak Accept / Acceptable main-track submission if claim scope remains narrow.**

M13 improves methodological credibility because it reports the hierarchy-aware uncertainty rather than only the favorable aggregate result. The result is nuanced: loss-shift evidence survives the bootstrap on Weather/Electricity, but lightweight-rank instability is not decisive. This is not a rejection-level problem because M9/M10 provide the rank-reversal evidence using official PatchTST/TimeXer architecture classes.

Updated M5 verdict after M13: `STRONG_CONFERENCE_READY`; blocking items: none; `method_claim_ready=false`.

---

# MaskShift ARS Reviewer Pass v7

## M14 Re-Review Delta

| Prior residual risk | M14 response | Status |
|---|---|---|
| S4M negative result might be an eight-channel artifact | Reran official S4M with 16 channels, 64 train windows, 48 test windows, and three seed offsets. | Strengthened |
| Larger reduced S4M might flip to positive mechanism-shift evidence | Weather max degradation 6.8% [-8.8, 22.4], Electricity 2.5% [-17.9, 22.8], both 0/3 gate seeds. | No flip; remains contrastive |
| Full S4M benchmark reproduction still absent | Manuscript now says M12/M14 are MaskShift-protocol adaptations and not full S4M benchmark reproduction. | Limitation preserved |

## Final Main-Conference Assessment After M14

**Decision: Acceptable main-track submission, benchmark-theory route, if venue formatting and final integrity checks remain clean.**

M14 improves the credibility of the S4M contrast because it removes the easiest scale objection against M12. The final position is now stronger and more nuanced: PatchTST/TimeXer demonstrate rank reversal under official architecture classes, CTF_missing shows mixed mechanism sensitivity, and S4M stays comparatively robust under both fast and larger reduced MaskShift settings. This is the right posture for a benchmark paper: report mechanism as a factor, not as a universal failure mode of every missing-aware architecture.

Updated final risks:
1. M12/M14 are still MaskShift-protocol adaptations, not full original S4M benchmark reproduction.
2. M10/M11/M12/M14 remain sprint-time three-seed summaries; full mixed-effects inference would be camera-ready strengthening.
3. The paper should keep the benchmark/theory framing and avoid a method/SOTA claim.

Updated M5 verdict after M14: `STRONG_CONFERENCE_READY`; blocking items: none; `method_claim_ready=false`.

---

# MaskShift ARS Reviewer Pass v8

## M15 Re-Review Delta

| Prior residual risk | M15 response | Status |
|---|---|---|
| Final integrity was advisory rather than gated | Added M15 final local integrity audit and wired it into M5. | Resolved |
| Orphan references and BibTeX warnings could survive to submission | M15 blocks dangling/orphan citations, missing DOI/URL fields, missing external audit-source records, and plain-BibTeX volume/number warnings. | Resolved |
| Main-text numbers could drift from generated JSON/table artifacts | M15 checks core M9/M11/M12/M13/M14 numeric snippets in `main.tex` and M14 table snippets against generated summaries. | Resolved |
| Device rules could regress under future edits | M15 compiles MaskShift scripts and checks torch files for CUDA -> MPS -> CPU selection and bare pin-memory misuse. | Resolved |

## Final Main-Conference Assessment After M15

**Decision: Main-track submission-ready benchmark/theory paper, with the correct target framing and final local integrity gate passed.**

M15 changes the status from "acceptable if final integrity checks remain clean" to "final local integrity checks are clean." The paper remains a benchmark/theory submission, not a method paper. Its strongest route is now: matched-rate mechanism shift is a real evaluation factor; official PatchTST/TimeXer show rank reversal under the MaskShift protocol; CTF_missing is mixed; S4M is contrastively robust under reduced and larger-reduced local settings; and the typed head is a negative diagnostic.

Updated M5 verdict after M15: `STRONG_CONFERENCE_READY`; blocking items: none; `method_claim_ready=false`; `final_integrity=PASS_FINAL_INTEGRITY`.

---

# MaskShift ARS Reviewer Pass v9

## M16 Re-Review Delta

| Prior residual risk | M16 response | Status |
|---|---|---|
| Official PatchTST/TimeXer coverage existed only on Weather/Electricity | Added M16 coverage for Traffic and AirConvection under the same MaskShift encoder-mask protocol. | Resolved |
| Reviewer could argue official-architecture evidence cherry-picked positive datasets | Traffic and AirConvection are reported as mixed/negative for rank reversal: Traffic 35.4% max degradation, tau 1.0; AirConvection 46.1%, tau 1.0. | Resolved |
| Scope could drift toward universal rank reversal | M16 reinforces the narrow claim: mechanism effects are dataset- and architecture-dependent. | Resolved |

## Final Main-Conference Assessment After M16

**Decision: Stronger main-track submission-ready benchmark/theory paper.**

M16 improves reviewer robustness because it removes a common sampling objection: official modern backbones were not evaluated only where MaskShift is strongest. The result is appropriately mixed. Weather/Electricity remain the positive official-architecture rank-reversal evidence, while Traffic/AirConvection show degradation without rank reversal. This is the correct acceptance posture for a benchmark paper: the benchmark factor matters, but its empirical strength varies by domain.

Updated M5 verdict after M16: `STRONG_CONFERENCE_READY`; blocking items: none; `official_full_dataset_coverage=PASS_OFFICIAL_TSLIB_FULL_COVERAGE`.

---

# MaskShift ARS Reviewer Pass v10

## M17 Re-Review Delta

| Prior residual risk | M17 response | Status |
|---|---|---|
| Evidence was distributed across many milestone files | Added a single supplement package with mechanism cards, dataset cards, official coverage table, reviewer concern matrix, and reproduction commands. | Resolved |
| Reviewer could miss that Traffic/AirConvection are boundary evidence | Added `docs/DATASET_MECHANISM_CARDS.md` and supplement dataset cards labeling Weather/Electricity as positive core evidence and Traffic/AirConvection as boundary/mixed evidence. | Resolved |
| Rebuttal preparation could drift into overclaiming | Added `docs/REBUTTAL_PLAYBOOK.md` and `tables/m17_reviewer_response_matrix.md` with scope-safe answers to likely strong-conference objections. | Resolved |
| Final gate did not require a supplement package | M5 now requires M17, and M15 checks the supplement artifacts and core phrases. | Resolved |

## Final Main-Conference Assessment After M17

**Decision: Strong main-track submission-ready benchmark/theory paper, with supplement-ready evidence.**

M17 improves the acceptance posture because it turns the project from a strong local experiment package into a submission package. The paper now has a main manuscript, compiled supplement, dataset/mechanism cards, response matrix, rebuttal playbook, final integrity audit, and final readiness gate. The scientific claim remains narrow: matched missing rate does not certify deployment missingness robustness, and official-architecture model ranking can reverse on Weather/Electricity while Traffic/AirConvection and S4M provide necessary boundary evidence.

Updated M5 verdict after M17: `STRONG_CONFERENCE_READY`; blocking items: none; `submission_supplement=PASS_SUBMISSION_SUPPLEMENT`.

---

# MaskShift ARS Reviewer Pass v11

## M18 Re-Review Delta

| Prior residual risk | M18 response | Status |
|---|---|---|
| Submission policy statements were implicit | Added data, code, reproducibility, ethics, AI-assistance, conflict/funding, author-contribution, and anonymity statements. | Resolved |
| Venue readiness could be overstated before target selection | Added `docs/VENUE_READINESS_AUDIT.md` and labels target venue style as `pending-target-selection`. | Resolved with scope guard |
| Final gates did not require policy/disclosure artifacts | M5 now requires M18, and M15 checks the policy artifacts, PDF size, core phrases, and blocking policy gaps. | Resolved |
| Authors might upload local paths or generic template as-is | M18 explicitly flags anonymous artifact-path scrubbing and target-template replacement before upload. | Resolved as pre-upload action |

## Final Main-Conference Assessment After M18

**Decision: Submit to a strong conference as a benchmark/theory paper after target-template conversion.**

M18 improves the acceptance posture by making the submission operationally complete: the project now has a main manuscript, compiled supplement, compiled policy/disclosure statement PDF, dataset/mechanism cards, reviewer response matrix, rebuttal playbook, final integrity audit, and final readiness gate. The target-specific venue style remains pending by design, so the honest conclusion is not "accepted" or "camera-ready"; it is "scientifically and locally submission-ready for a strong-conference benchmark/theory route, with mechanical venue-template mapping still required before upload."

Updated M5 verdict after M18 integration: `STRONG_CONFERENCE_READY`; blocking items: none; `submission_policy_pack=PASS_SUBMISSION_POLICY_PACK`; `method_claim_ready=false`.

---

# MaskShift ARS Reviewer Pass v12

## M19 Re-Review Delta

| Prior residual risk | M19 response | Status |
|---|---|---|
| Target venue remained abstract | Selected AAAI-27 Main Technical Track as the near-term strong-conference target and recorded official deadlines, page policy, review process, reproducibility, and AI-assistance policy. | Resolved |
| Target appendix/hyperparameter/reproducibility material was still a future P1 | Added `docs/AAAI27_REPRODUCIBILITY_CHECKLIST_DRAFT.md` and mapped datasets, splits, seeds, baselines, code/data, and limitations to checklist-style evidence. | Resolved as draft |
| AAAI two-phase review and AI-generated review could create scope-misread risk | Added `docs/AAAI27_PHASE_REVIEW_RESPONSE_PLAN.md` with evidence-grounded responses to likely human and AI-review objections. | Resolved |
| Generic article-class PDF might be mistaken as upload-ready | M19 records `aaai27_upload_ready=false` and separates scientific/style readiness from submission-system/form completion. | Resolved by explicit blocker |

## Final Main-Conference Assessment After M19

**Decision: Scientifically strong enough for AAAI-27 main-track submission; target-specific operational upload still waits on the venue system.**

M19 strengthens the acceptance posture because the package is now aligned to a concrete venue rather than a generic "strong conference." It also prevents a dangerous false-positive: the current generic `main.pdf` should not be uploaded as the AAAI submission. After M20, target style and page-limit risk are resolved; the remaining action is to use the official PDF in OpenReview once the paper submission site opens and complete the official reproducibility checklist from the M19 draft.

Updated M5 verdict after M19 integration: `STRONG_CONFERENCE_READY`; blocking items: none; `aaai27_target_dossier=PASS_AAAI27_TARGET_DOSSIER`; `aaai27_upload_ready=false`.

---

# MaskShift ARS Reviewer Pass v13

## M20 Re-Review Delta

| Prior residual risk | M20 response | Status |
|---|---|---|
| AAAI page-limit risk was only stated, not preflight-tested | Added an anonymous two-column US-letter preflight build and compiled `paper/aaai27_preflight.pdf` to 5 pages. | Resolved |
| Wide result tables could overflow in two-column format | M20 promotes result tables to double-column `table*` floats in generated preflight and official builds. | Resolved |
| Preflight might be confused with official AAAI compliance | M20 now also downloads/extracts the official author kit and compiles `paper/aaai27_official.pdf` with `aaai2027`. | Resolved |
| Final gates did not require page-pressure or official-template evidence | M5 and M15 now require M20 artifacts, official style files, and PASS status. | Resolved |

## Final Main-Conference Assessment After M20

**Decision: Scientifically strong enough for AAAI-27 main-track submission and official-template ready; operational upload still waits on the site/form workflow.**

M20 materially improves the submission posture: the current body fits within a conservative AAAI-like two-column layout and within the official `aaai2027` anonymous submission template, both at 5 pages. The remaining blocker is now purely operational: when paper submission opens, upload `paper/aaai27_official.pdf` and complete the official reproducibility checklist fields. This is not a scientific or formatting rejection blocker.

Updated M5 verdict after M20 integration: `STRONG_CONFERENCE_READY`; blocking items: none; `aaai27_preflight_conversion=PASS_AAAI27_PREFLIGHT_CONVERSION`; `aaai27_upload_ready=false`.

---

# MaskShift ARS Reviewer Pass v14

## M21 Re-Review Delta

| Prior residual risk | M21 response | Status |
|---|---|---|
| AAAI reproducibility checklist existed only as a target-readiness draft | Filled the official AAAI-27 `ReproducibilityChecklist.tex` from the author kit and compiled `paper/aaai27_reproducibility_checklist.pdf`. | Resolved locally |
| OpenReview checklist entry could drift from the manuscript evidence | Added `docs/AAAI27_REPRODUCIBILITY_CHECKLIST_FILLED.md` with 31 checklist answers and zero remaining question placeholders. | Resolved as upload source |
| Final gates did not require the official checklist artifact | M5 and M15 now require M21 artifacts and PASS status. | Resolved |

## Final Main-Conference Assessment After M21

**Decision: AAAI-27 main-track submission-ready as a benchmark/theory paper; official upload still waits on the site/form workflow.**

M21 closes the local reproducibility-checklist gap. The current package now has a main manuscript, supplement, policy/disclosure PDF, AAAI target dossier, official `aaai2027` anonymous PDF, and filled official checklist PDF. The remaining blocker is not a paper-quality blocker: when the AAAI paper submission site opens, upload `paper/aaai27_official.pdf` and copy the filled checklist answers into OpenReview.

Updated M5 verdict after M21 integration: `STRONG_CONFERENCE_READY`; blocking items: none; `aaai27_reproducibility_checklist=PASS_AAAI27_REPRODUCIBILITY_CHECKLIST`; `aaai27_upload_ready=false`.
