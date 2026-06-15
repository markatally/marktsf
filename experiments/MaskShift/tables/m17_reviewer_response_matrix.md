# M17 reviewer response matrix

This table is a pre-submission rebuttal map. It does not fabricate reviewer comments; it maps likely strong-conference concerns to already generated evidence.

| Likely reviewer concern | Short answer | Evidence artifact | Status |
| --- | --- | --- | --- |
| Is this just another missing-value model paper? | No. MaskShift is framed as benchmark/theory and explicitly excludes SOTA or new-backbone claims. | PAPER.md Sections 1, 3, 6; M5 claim_scope_safe. | Resolved by scope discipline. |
| Were modern forecasting backbones tested? | Yes. M9/M10 import official TSLib PatchTST and TimeXer classes; M16 extends coverage to all four datasets. | m9_official_tslib_reproduction_summary.json; m16_official_tslib_full_coverage_summary.json. | Resolved as official-architecture adaptation. |
| Were missing-aware architectures tested? | Yes. M11 adapts official ChannelTokenFormer_missing and M12/M14 adapt official S4M. | M11, M12, M14 summaries and tables. | Resolved with mixed/negative evidence disclosed. |
| Are Weather/Electricity cherry-picked? | M16 reports Traffic and AirConvection official PatchTST/TimeXer coverage as mixed/negative for rank reversal. | tables/m16_official_tslib_full_coverage_table.md. | Resolved by visible boundary evidence. |
| Is the result only sensor retirement? | No. M8 excludes retirement and still finds positive non-retirement evidence on Weather and Electricity. | m8_mechanism_decomposition_summary.json. | Resolved for the core positive datasets. |
| Are relative degradation ratios unstable? | M10 adds absolute delta, log ratio, and symmetric relative delta; the paper does not rely on denominator spikes. | tables/m7_corrected_robustness_table.md. | Resolved by corrected reporting. |
| Does the typed head work as a method? | No. H3 fails and the paper reports the typed/topology head as a negative diagnostic only. | m2_summary.json; m3_summary.json; PAPER.md Section 4.6. | Resolved by de-scoping. |
| Is the statistical evidence only aggregate means? | M3 uses BH-FDR, M10 adds three-seed CIs, and M13 adds a hierarchy-aware variant/window bootstrap. | m3_summary.json; m10_submission_hardening_summary.json; m13_hierarchical_bootstrap_summary.json. | Resolved for submission, with limitations retained. |
| Is the official-code claim over-stated? | The paper consistently says official-architecture adaptation under MaskShift, not full official benchmark reproduction. | PAPER.md limitations; SUBMISSION_CHECKLIST.md must-keep scope. | Resolved by wording guardrails. |
| Can a reviewer reproduce the package? | The README and supplement list M0-M17 commands and external repository revisions. | README.md; paper/README.md; paper/supplement.tex. | Resolved as local reproduction package. |
