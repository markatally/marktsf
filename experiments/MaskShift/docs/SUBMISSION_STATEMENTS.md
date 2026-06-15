# MaskShift Submission Statements

This document is a target-agnostic submission policy pack. It should be copied into the target venue's required fields or appendix once the venue is fixed.

## Gate Context

- M5 readiness: `STRONG_CONFERENCE_READY`; blocking items: `0`
- M15 integrity: `PASS_FINAL_INTEGRITY`; blocking issues: `0`
- M17 supplement: `PASS_SUBMISSION_SUPPLEMENT`; reviewer concerns covered: `10`

## Statement Index

| Statement | Status | Residual venue-specific risk |
| --- | --- | --- |
| Data Availability Statement | review-ready | Raw-dataset license/source metadata should be checked against the target venue's artifact policy before public release. |
| Code Availability Statement | review-ready | Before final submission, package the ignored external/ revisions or provide exact clone commands in the anonymous artifact instructions. |
| Reproducibility Statement | review-ready | Full original-protocol S4M reproduction remains outside the current evidence boundary and must not be implied. |
| Ethics Statement | review-ready | If a target venue requires broader-impact forms, reuse this statement but add venue-specific checkboxes. |
| AI Assistance Disclosure | review-ready | Some venues require exact AI tool names or prompts; add those details only if the target policy requests them. |
| Conflict of Interest Statement | anonymous-review placeholder | Author-specific conflicts cannot be completed without the author list and venue system fields. |
| Funding Statement | anonymous-review placeholder | Funding details require author confirmation. |
| Author Contributions | camera-ready placeholder | Contribution allocation requires the final author list. |
| Anonymity Statement | review-ready | Local absolute paths in JSON artifacts are useful for audit but should be scrubbed or mapped in any public anonymous artifact bundle. |

## Data Availability Statement

Status: `review-ready`

The experiments use time-series benchmark files expected under the repository's input/ directory. The MaskShift package records the exact local paths, selected rows/channels, natural missing rates, and derived windows in M0/M16/M17 artifacts. No private human-subject data is introduced by MaskShift.

Residual venue-specific risk: Raw-dataset license/source metadata should be checked against the target venue's artifact policy before public release.

## Code Availability Statement

Status: `review-ready`

All MaskShift milestone scripts M0-M18, generated JSON summaries, tables, figures, manuscript TeX, and supplement TeX/PDF are included under experiments/MaskShift. External official-architecture repositories are pinned to TSLib 4e938a1, ChannelTokenFormer b1c100e, and S4M a718823.

Residual venue-specific risk: Before final submission, package the ignored external/ revisions or provide exact clone commands in the anonymous artifact instructions.

## Reproducibility Statement

Status: `review-ready`

All reported MaskShift runs use deterministic seeds where applicable, chronological splits, encoder-input-only masks, clean targets, and generated JSON summaries. M15 checks numerical traceability from JSON summaries to paper/tables and M5 requires all scientific gates.

Residual venue-specific risk: Full original-protocol S4M reproduction remains outside the current evidence boundary and must not be implied.

## Ethics Statement

Status: `review-ready`

The work is a benchmark/theory study over time-series datasets and synthetic missingness mechanisms. It does not involve human-subject intervention, user profiling, or sensitive personal data collection. The principal risk is over-trusting MCAR/block robustness in operational systems; the paper mitigates this by arguing for mechanism reporting rather than claiming a universal repair.

Residual venue-specific risk: If a target venue requires broader-impact forms, reuse this statement but add venue-specific checkboxes.

## AI Assistance Disclosure

Status: `review-ready`

AI assistance was used to help organize the research pipeline, draft/edit text, and generate local audit scripts. The authors remain responsible for all claims. Numerical statements are traced to local JSON artifacts by M15, and citations are checked for dangling/orphan references and external audit-source records.

Residual venue-specific risk: Some venues require exact AI tool names or prompts; add those details only if the target policy requests them.

## Conflict of Interest Statement

Status: `anonymous-review placeholder`

For double-blind review, author-identifying conflict details should be supplied through the submission system rather than the anonymous manuscript. The camera-ready version should include the final conflict declaration.

Residual venue-specific risk: Author-specific conflicts cannot be completed without the author list and venue system fields.

## Funding Statement

Status: `anonymous-review placeholder`

For double-blind review, funding acknowledgments should be omitted from the anonymous manuscript when they identify the authors. The camera-ready version should include final funding information or state that no external funding was received.

Residual venue-specific risk: Funding details require author confirmation.

## Author Contributions

Status: `camera-ready placeholder`

Author contributions should be reported in the camera-ready version using the target venue's preferred format, such as CRediT roles, after the anonymous review phase.

Residual venue-specific risk: Contribution allocation requires the final author list.

## Anonymity Statement

Status: `review-ready`

The manuscript and supplement use anonymous authors. Repository paths in reproduction notes should be replaced by anonymous artifact URLs or relative paths before upload if required by the target venue.

Residual venue-specific risk: Local absolute paths in JSON artifacts are useful for audit but should be scrubbed or mapped in any public anonymous artifact bundle.

## Target-Specific Action

Replace the generic article class with the selected venue style and map these statements into the target submission system. Do not claim target-specific compliance until the venue template and policy fields have been checked.
