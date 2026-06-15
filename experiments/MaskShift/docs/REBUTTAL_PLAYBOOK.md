# MaskShift Rebuttal Playbook

Use this file after reviews arrive. It is not a response letter; it is a constraint system that prevents overclaiming while answering predictable objections.

## Non-Negotiable Scope Lines

- MaskShift is a benchmark/theory paper, not a new forecasting backbone.
- PatchTST, TimeXer, ChannelTokenFormer_missing, and S4M results are official-architecture adaptations under the MaskShift protocol.
- The typed/topology head is a negative diagnostic; it is not a method contribution.
- Traffic and AirConvection are boundary evidence and must remain visible.
- S4M is a contrastive robust baseline under reduced local settings, not a failure case.

## Reviewer-Concern Playbook

### R1. Is this just another missing-value model paper?

- Answer: No. MaskShift is framed as benchmark/theory and explicitly excludes SOTA or new-backbone claims.
- Cite: PAPER.md Sections 1, 3, 6; M5 claim_scope_safe.
- Safe wording: Resolved by scope discipline.

### R2. Were modern forecasting backbones tested?

- Answer: Yes. M9/M10 import official TSLib PatchTST and TimeXer classes; M16 extends coverage to all four datasets.
- Cite: m9_official_tslib_reproduction_summary.json; m16_official_tslib_full_coverage_summary.json.
- Safe wording: Resolved as official-architecture adaptation.

### R3. Were missing-aware architectures tested?

- Answer: Yes. M11 adapts official ChannelTokenFormer_missing and M12/M14 adapt official S4M.
- Cite: M11, M12, M14 summaries and tables.
- Safe wording: Resolved with mixed/negative evidence disclosed.

### R4. Are Weather/Electricity cherry-picked?

- Answer: M16 reports Traffic and AirConvection official PatchTST/TimeXer coverage as mixed/negative for rank reversal.
- Cite: tables/m16_official_tslib_full_coverage_table.md.
- Safe wording: Resolved by visible boundary evidence.

### R5. Is the result only sensor retirement?

- Answer: No. M8 excludes retirement and still finds positive non-retirement evidence on Weather and Electricity.
- Cite: m8_mechanism_decomposition_summary.json.
- Safe wording: Resolved for the core positive datasets.

### R6. Are relative degradation ratios unstable?

- Answer: M10 adds absolute delta, log ratio, and symmetric relative delta; the paper does not rely on denominator spikes.
- Cite: tables/m7_corrected_robustness_table.md.
- Safe wording: Resolved by corrected reporting.

### R7. Does the typed head work as a method?

- Answer: No. H3 fails and the paper reports the typed/topology head as a negative diagnostic only.
- Cite: m2_summary.json; m3_summary.json; PAPER.md Section 4.6.
- Safe wording: Resolved by de-scoping.

### R8. Is the statistical evidence only aggregate means?

- Answer: M3 uses BH-FDR, M10 adds three-seed CIs, and M13 adds a hierarchy-aware variant/window bootstrap.
- Cite: m3_summary.json; m10_submission_hardening_summary.json; m13_hierarchical_bootstrap_summary.json.
- Safe wording: Resolved for submission, with limitations retained.

### R9. Is the official-code claim over-stated?

- Answer: The paper consistently says official-architecture adaptation under MaskShift, not full official benchmark reproduction.
- Cite: PAPER.md limitations; SUBMISSION_CHECKLIST.md must-keep scope.
- Safe wording: Resolved by wording guardrails.

### R10. Can a reviewer reproduce the package?

- Answer: The README and supplement list M0-M17 commands and external repository revisions.
- Cite: README.md; paper/README.md; paper/supplement.tex.
- Safe wording: Resolved as local reproduction package.

## Dataset-Specific Response Map

### Weather

- Evidence role: positive core evidence
- M1 eta^2: 0.614
- M16 max official PatchTST/TimeXer degradation: 123.7%
- M16 worst tau: -1.000
- Response stance: Use as core positive evidence.

### Electricity

- Evidence role: positive core evidence
- M1 eta^2: 0.777
- M16 max official PatchTST/TimeXer degradation: 87.8%
- M16 worst tau: -1.000
- Response stance: Use as core positive evidence.

### Traffic

- Evidence role: boundary/mixed evidence
- M1 eta^2: 0.012
- M16 max official PatchTST/TimeXer degradation: 35.4%
- M16 worst tau: 1.000
- Response stance: Use as boundary evidence; do not force into a universal claim.

### AirConvection

- Evidence role: boundary/mixed evidence
- M1 eta^2: 0.120
- M16 max official PatchTST/TimeXer degradation: 46.1%
- M16 worst tau: 1.000
- Response stance: Use as boundary evidence; do not force into a universal claim.

## If Asked For More Experiments

Prioritize in this order: more seed offsets for M9/M10, full original-protocol S4M reproduction, larger ChannelTokenFormer_missing adaptation, then a mixed-effects model over seeds/datasets/horizons. Do not add an unverified experiment to the rebuttal.
