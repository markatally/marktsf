# PRISM Submission Trace Pack

Generated from current M17 artifacts. This file is a manuscript-facing
traceability aid: every positive claim below must remain within its evidence and
limitations.

## Claim Registry

| Claim ID | Status | Manuscript Locator | Evidence Files | Limitations |
| --- | --- | --- | --- | --- |
| C-M17-SCOPED-ROUTE | SUPPORTED | Abstract / Contributions / Main Results | experiments/PRISM/practical_selective_horizon_affine/practical_selective_horizon_affine_summary.json<br>experiments/PRISM/main_track_audit/main_track_audit.json<br>experiments/PRISM/paper_ready/paper_ready_summary.json | Claim must be scoped to non-financial sensor/infrastructure data.<br>Do not claim full PRISM learned-router success or SOTA dominance. |
| C-M17-ACTIVE-FDR | SUPPORTED | Results Table 1 / Main Results paragraph | experiments/PRISM/practical_selective_horizon_affine/practical_selective_horizon_affine_summary.json | Only active cells support superiority claims.<br>Inactive cells support no-harm abstention, not superiority. |
| C-M17-NO-HARM | SUPPORTED | Method / Selective activation; Results Table 1 | experiments/PRISM/practical_selective_horizon_affine/practical_selective_horizon_affine_summary.json | No-harm is relative to validation-single, not to delayed Fixed-Share or descriptor ridge. |
| C-M17-THRESHOLD-SENSITIVITY | SUPPORTED_WITH_LIMITATION | Sensitivity / Limitations | experiments/PRISM/practical_selective_horizon_affine_sensitivity_0/practical_selective_horizon_affine_summary.json<br>experiments/PRISM/practical_selective_horizon_affine_sensitivity_2p5/practical_selective_horizon_affine_summary.json<br>experiments/PRISM/practical_selective_horizon_affine/practical_selective_horizon_affine_summary.json<br>experiments/PRISM/practical_selective_horizon_affine_sensitivity_10/practical_selective_horizon_affine_summary.json | The paper must disclose that the 5% threshold is the only passing point in the tested grid.<br>Treat threshold selection as a design limitation, not as a universal constant. |
| C-RETIRED-ROUTER | SUPPORTED_NEGATIVE | Ablations / Negative results / Limitations | experiments/PRISM/router_viability/router_viability_summary.json<br>experiments/PRISM/router_viability_h192/router_viability_summary.json<br>experiments/PRISM/champion_risk_gate/champion_risk_gate_summary.json<br>experiments/PRISM/calibrated_stack_significance/calibrated_stack_significance_summary.json<br>experiments/PRISM/sensor_horizon_affine_significance/sensor_horizon_affine_significance_summary.json<br>experiments/PRISM/selective_horizon_affine/selective_horizon_affine_summary.json | These negative results do not falsify future redesigned routers.<br>Do not use M17 to retroactively claim the original router passed. |

## Figure/Table Trace

| Artifact | Caption Claim | Source Data | Rendered Artifacts | Supported Claims | Limitations |
| --- | --- | --- | --- | --- | --- |
| Table 1 | M17 scoped selective route results at the 5% practical-effect activation threshold. | experiments/PRISM/practical_selective_horizon_affine/practical_selective_horizon_affine_summary.json | experiments/PRISM/submission_render/table1_m17_active_cells.csv<br>experiments/PRISM/submission_render/table1_m17_active_cells.md | C-M17-ACTIVE-FDR, C-M17-NO-HARM | Superiority claims apply only to active cells.<br>Inactive no-harm is only relative to validation-single. |
| Table 2 | Practical-effect threshold sensitivity for M17 activation. | experiments/PRISM/practical_selective_horizon_affine_sensitivity_0/practical_selective_horizon_affine_summary.json<br>experiments/PRISM/practical_selective_horizon_affine_sensitivity_2p5/practical_selective_horizon_affine_summary.json<br>experiments/PRISM/practical_selective_horizon_affine/practical_selective_horizon_affine_summary.json<br>experiments/PRISM/practical_selective_horizon_affine_sensitivity_10/practical_selective_horizon_affine_summary.json | experiments/PRISM/submission_render/table2_threshold_sensitivity.csv<br>experiments/PRISM/submission_render/table2_threshold_sensitivity.md | C-M17-THRESHOLD-SENSITIVITY | This is a finite sensitivity grid, not a proof of optimality.<br>The 5% threshold must be presented as a preregistered/design choice for this route, not a universal threshold. |
| Figure 1 | Route evolution and retired claims from M2 through M17. | experiments/PRISM/router_viability/router_viability_summary.json<br>experiments/PRISM/calibrated_stack_significance/calibrated_stack_significance_summary.json<br>experiments/PRISM/sensor_horizon_affine_significance/sensor_horizon_affine_significance_summary.json<br>experiments/PRISM/selective_horizon_affine/selective_horizon_affine_summary.json<br>experiments/PRISM/practical_selective_horizon_affine/practical_selective_horizon_affine_summary.json | experiments/PRISM/submission_render/figure1_route_hardening.png | C-M17-SCOPED-ROUTE, C-RETIRED-ROUTER | Timeline figure is explanatory; primary numerical evidence remains Tables 1-2 and JSON artifacts. |

## Negative Claim Constraints

| Constraint | Rule |
| --- | --- |
| NC-001 | Do not claim that the original delayed contextual PRISM router passes the strengthened main-track gate. |
| NC-002 | Do not claim full-coverage or all-cell superiority; M17 superiority applies only to active cells. |
| NC-003 | Do not claim finance or Exchange wins; these remain negative or out-of-scope diagnostics. |
| NC-004 | Do not omit threshold sensitivity; 5% is the only passing tested practical-effect threshold. |
| NC-005 | Do not claim inactive cells beat Fixed-Share or descriptor ridge; inactive cells only abstain to validation-single. |

## Submission Checklist

- [x] M17 summary exists and passes gate.
- [x] Main-track audit returns `ALLOW_SCOPED_MAIN_TRACK_SUBMISSION`.
- [x] Positive claims are scoped to active M17 cells and non-financial sensor/infrastructure data.
- [x] Threshold sensitivity is generated for 0%, 2.5%, 5%, and 10%.
- [x] Final manuscript tables and figures have been rendered from these trace entries.
- [ ] Final bibliography has been fully verified against primary sources.
- [ ] Final PDF has passed visual/table-caption verification.
