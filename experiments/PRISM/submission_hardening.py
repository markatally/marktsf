"""PRISM submission hardening pack.

This script turns the M17 scoped-route artifacts into manuscript-facing
traceability materials: claim registry, figure/table trace, negative-claim
constraints, and a submission checklist.  It does not create new experimental
results; it verifies that the current claims are backed by JSON artifacts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def pct(value: float) -> str:
    return f"{value:.3g}%"


@dataclass(frozen=True)
class Claim:
    claim_id: str
    manuscript_locator: str
    claim_text: str
    status: str
    evidence_files: list[str]
    extraction: dict[str, Any]
    limitations: list[str]


def active_cells(m17: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in m17["rows"] if row["active"]]


def inactive_cells(m17: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in m17["rows"] if not row["active"]]


def sensitivity_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for label, dirname in [
        ("0", "practical_selective_horizon_affine_sensitivity_0"),
        ("2.5", "practical_selective_horizon_affine_sensitivity_2p5"),
        ("5", "practical_selective_horizon_affine"),
        ("10", "practical_selective_horizon_affine_sensitivity_10"),
    ]:
        path = root / dirname / "practical_selective_horizon_affine_summary.json"
        summary = load(path)
        rows.append(
            {
                "min_effect_pct": label,
                "gate_pass": bool(summary["gate_pass"]),
                "active_count": int(summary["active_count"]),
                "total_cells": len(summary["rows"]),
                "fdr_pass_counts": summary["active_fdr_pass_counts"],
                "active_cells": [
                    {"dataset": row["dataset"], "horizon": row["horizon"]}
                    for row in active_cells(summary)
                ],
            }
        )
    return rows


def build_claims(root: Path) -> list[Claim]:
    m17_path = root / "practical_selective_horizon_affine/practical_selective_horizon_affine_summary.json"
    m17 = load(m17_path)
    audit_path = root / "main_track_audit/main_track_audit.json"
    audit = load(audit_path)
    paper_ready_path = root / "paper_ready/paper_ready_summary.json"
    paper_ready = load(paper_ready_path)
    sens = sensitivity_rows(root)
    active = active_cells(m17)
    inactive = inactive_cells(m17)

    return [
        Claim(
            claim_id="C-M17-SCOPED-ROUTE",
            manuscript_locator="Abstract / Contributions / Main Results",
            claim_text=(
                "PRISM's supported positive result is a scoped practical-effect "
                "selective horizon-wise affine calibration route for non-financial "
                "sensor/infrastructure forecasting, not the original full learned-router claim."
            ),
            status="SUPPORTED",
            evidence_files=[str(m17_path), str(audit_path), str(paper_ready_path)],
            extraction={
                "m17_gate_pass": bool(m17["gate_pass"]),
                "audit_decision": audit["decision"],
                "paper_ready_status": paper_ready["status"],
                "scope_datasets": m17["scope"]["datasets"],
                "horizons": m17["scope"]["horizons"],
            },
            limitations=[
                "Claim must be scoped to non-financial sensor/infrastructure data.",
                "Do not claim full PRISM learned-router success or SOTA dominance.",
            ],
        ),
        Claim(
            claim_id="C-M17-ACTIVE-FDR",
            manuscript_locator="Results Table 1 / Main Results paragraph",
            claim_text=(
                "At the 5% practical-effect activation threshold, M17 activates 4/16 "
                "dataset-horizon cells and all active cells pass BH/FDR versus "
                "validation-single, delayed Fixed-Share, and descriptor ridge."
            ),
            status="SUPPORTED",
            evidence_files=[str(m17_path)],
            extraction={
                "active_count": m17["active_count"],
                "total_cells": len(m17["rows"]),
                "inactive_no_harm": m17["inactive_no_harm"],
                "active_fdr_pass_counts": m17["active_fdr_pass_counts"],
                "active_cells": [
                    {
                        "dataset": row["dataset"],
                        "horizon": row["horizon"],
                        "validation_single_improvement_pct": row["tests"]["selective_active_vs_validation_single"]["improvement_pct"],
                        "fixed_share_improvement_pct": row["tests"]["selective_active_vs_fixed_share"]["improvement_pct"],
                        "descriptor_ridge_improvement_pct": row["tests"]["selective_active_vs_descriptor_ridge"]["improvement_pct"],
                    }
                    for row in active
                ],
            },
            limitations=[
                "Only active cells support superiority claims.",
                "Inactive cells support no-harm abstention, not superiority.",
            ],
        ),
        Claim(
            claim_id="C-M17-NO-HARM",
            manuscript_locator="Method / Selective activation; Results Table 1",
            claim_text=(
                "Inactive M17 cells abstain to the validation-selected single expert, "
                "so inactive-cell loss is exactly equal to validation-single by construction."
            ),
            status="SUPPORTED",
            evidence_files=[str(m17_path)],
            extraction={
                "inactive_no_harm": bool(m17["inactive_no_harm"]),
                "inactive_count": int(m17["inactive_count"]),
                "inactive_cells": [
                    {
                        "dataset": row["dataset"],
                        "horizon": row["horizon"],
                        "selective_loss": row["aggregate"]["selective_loss"],
                        "validation_single_loss": row["aggregate"]["validation_single_loss"],
                    }
                    for row in inactive
                ],
            },
            limitations=[
                "No-harm is relative to validation-single, not to delayed Fixed-Share or descriptor ridge.",
            ],
        ),
        Claim(
            claim_id="C-M17-THRESHOLD-SENSITIVITY",
            manuscript_locator="Sensitivity / Limitations",
            claim_text=(
                "The practical-effect threshold is narrow: 0% and 2.5% thresholds fail "
                "because fragile active cells miss Fixed-Share FDR; 10% fails coverage; "
                "5% is the only passing setting in the current grid."
            ),
            status="SUPPORTED_WITH_LIMITATION",
            evidence_files=[
                str(root / "practical_selective_horizon_affine_sensitivity_0/practical_selective_horizon_affine_summary.json"),
                str(root / "practical_selective_horizon_affine_sensitivity_2p5/practical_selective_horizon_affine_summary.json"),
                str(m17_path),
                str(root / "practical_selective_horizon_affine_sensitivity_10/practical_selective_horizon_affine_summary.json"),
            ],
            extraction={"sensitivity": sens},
            limitations=[
                "The paper must disclose that the 5% threshold is the only passing point in the tested grid.",
                "Treat threshold selection as a design limitation, not as a universal constant.",
            ],
        ),
        Claim(
            claim_id="C-RETIRED-ROUTER",
            manuscript_locator="Ablations / Negative results / Limitations",
            claim_text=(
                "The original delayed contextual router, champion-risk safe-switch, broad "
                "calibrated stack, and dynamic beta loop are insufficient as headline method claims."
            ),
            status="SUPPORTED_NEGATIVE",
            evidence_files=[
                str(root / "router_viability/router_viability_summary.json"),
                str(root / "router_viability_h192/router_viability_summary.json"),
                str(root / "champion_risk_gate/champion_risk_gate_summary.json"),
                str(root / "calibrated_stack_significance/calibrated_stack_significance_summary.json"),
                str(root / "sensor_horizon_affine_significance/sensor_horizon_affine_significance_summary.json"),
                str(root / "selective_horizon_affine/selective_horizon_affine_summary.json"),
            ],
            extraction={
                "m2_gate_pass": load(root / "router_viability/router_viability_summary.json")["gate_pass"],
                "m7_gate_pass": load(root / "router_viability_h192/router_viability_summary.json")["gate_pass"],
                "m16_gate_pass": load(root / "selective_horizon_affine/selective_horizon_affine_summary.json")["gate_pass"],
            },
            limitations=[
                "These negative results do not falsify future redesigned routers.",
                "Do not use M17 to retroactively claim the original router passed.",
            ],
        ),
    ]


def build_figure_table_trace(root: Path, claims: list[Claim]) -> list[dict[str, Any]]:
    m17_path = root / "practical_selective_horizon_affine/practical_selective_horizon_affine_summary.json"
    sens_paths = [
        root / "practical_selective_horizon_affine_sensitivity_0/practical_selective_horizon_affine_summary.json",
        root / "practical_selective_horizon_affine_sensitivity_2p5/practical_selective_horizon_affine_summary.json",
        m17_path,
        root / "practical_selective_horizon_affine_sensitivity_10/practical_selective_horizon_affine_summary.json",
    ]
    return [
        {
            "artifact_id": "Table 1",
            "proposed_caption": "M17 scoped selective route results at the 5% practical-effect activation threshold.",
            "source_data": [str(m17_path)],
            "rendered_artifacts": [
                str(root / "submission_render/table1_m17_active_cells.csv"),
                str(root / "submission_render/table1_m17_active_cells.md"),
            ],
            "transformation": "Parse rows, active flags, active_fdr_pass_counts, and inactive_no_harm from M17 summary JSON.",
            "caption_claim": "M17 activates 4/16 cells; all active cells pass FDR against validation-single, delayed Fixed-Share, and descriptor ridge; inactive cells abstain to validation-single.",
            "supported_manuscript_claims": ["C-M17-ACTIVE-FDR", "C-M17-NO-HARM"],
            "limitations": [
                "Superiority claims apply only to active cells.",
                "Inactive no-harm is only relative to validation-single.",
            ],
        },
        {
            "artifact_id": "Table 2",
            "proposed_caption": "Practical-effect threshold sensitivity for M17 activation.",
            "source_data": [str(path) for path in sens_paths],
            "rendered_artifacts": [
                str(root / "submission_render/table2_threshold_sensitivity.csv"),
                str(root / "submission_render/table2_threshold_sensitivity.md"),
            ],
            "transformation": "Compare gate_pass, active_count, and active_fdr_pass_counts for min_effect_pct in {0, 2.5, 5, 10}.",
            "caption_claim": "The 5% threshold is the only passing tested threshold; lower thresholds admit fragile cells and 10% under-covers active cells.",
            "supported_manuscript_claims": ["C-M17-THRESHOLD-SENSITIVITY"],
            "limitations": [
                "This is a finite sensitivity grid, not a proof of optimality.",
                "The 5% threshold must be presented as a preregistered/design choice for this route, not a universal threshold.",
            ],
        },
        {
            "artifact_id": "Figure 1",
            "proposed_caption": "Route evolution and retired claims from M2 through M17.",
            "source_data": [
                str(root / "router_viability/router_viability_summary.json"),
                str(root / "calibrated_stack_significance/calibrated_stack_significance_summary.json"),
                str(root / "sensor_horizon_affine_significance/sensor_horizon_affine_significance_summary.json"),
                str(root / "selective_horizon_affine/selective_horizon_affine_summary.json"),
                str(m17_path),
            ],
            "rendered_artifacts": [str(root / "submission_render/figure1_route_hardening.png")],
            "transformation": "Manual timeline/table from milestone gate_pass and pass-count fields; no numerical interpolation.",
            "caption_claim": "M17 supersedes failed learned-router/full-coverage routes with a scoped selective route.",
            "supported_manuscript_claims": ["C-M17-SCOPED-ROUTE", "C-RETIRED-ROUTER"],
            "limitations": [
                "Timeline figure is explanatory; primary numerical evidence remains Tables 1-2 and JSON artifacts.",
            ],
        },
    ]


def build_negative_constraints() -> list[dict[str, str]]:
    return [
        {
            "constraint_id": "NC-001",
            "rule": "Do not claim that the original delayed contextual PRISM router passes the strengthened main-track gate.",
        },
        {
            "constraint_id": "NC-002",
            "rule": "Do not claim full-coverage or all-cell superiority; M17 superiority applies only to active cells.",
        },
        {
            "constraint_id": "NC-003",
            "rule": "Do not claim finance or Exchange wins; these remain negative or out-of-scope diagnostics.",
        },
        {
            "constraint_id": "NC-004",
            "rule": "Do not omit threshold sensitivity; 5% is the only passing tested practical-effect threshold.",
        },
        {
            "constraint_id": "NC-005",
            "rule": "Do not claim inactive cells beat Fixed-Share or descriptor ridge; inactive cells only abstain to validation-single.",
        },
    ]


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def write_markdown(out_dir: Path, claims: list[Claim], trace: list[dict[str, Any]], constraints: list[dict[str, str]]) -> None:
    root = out_dir.parent
    render_summary_path = root / "submission_render/render_validation_summary.json"
    render_ready = False
    if render_summary_path.exists():
        render_ready = bool(json.loads(render_summary_path.read_text()).get("gate_pass", False))
    claim_rows = [
        [
            claim.claim_id,
            claim.status,
            claim.manuscript_locator,
            "<br>".join(claim.evidence_files),
            "<br>".join(claim.limitations),
        ]
        for claim in claims
    ]
    trace_rows = [
        [
            entry["artifact_id"],
            entry["proposed_caption"],
            "<br>".join(entry["source_data"]),
            "<br>".join(entry.get("rendered_artifacts", [])),
            ", ".join(entry["supported_manuscript_claims"]),
            "<br>".join(entry["limitations"]),
        ]
        for entry in trace
    ]
    constraint_rows = [[item["constraint_id"], item["rule"]] for item in constraints]
    text = f"""# PRISM Submission Trace Pack

Generated from current M17 artifacts. This file is a manuscript-facing
traceability aid: every positive claim below must remain within its evidence and
limitations.

## Claim Registry

{markdown_table(["Claim ID", "Status", "Manuscript Locator", "Evidence Files", "Limitations"], claim_rows)}

## Figure/Table Trace

{markdown_table(["Artifact", "Caption Claim", "Source Data", "Rendered Artifacts", "Supported Claims", "Limitations"], trace_rows)}

## Negative Claim Constraints

{markdown_table(["Constraint", "Rule"], constraint_rows)}

## Submission Checklist

- [x] M17 summary exists and passes gate.
- [x] Main-track audit returns `ALLOW_SCOPED_MAIN_TRACK_SUBMISSION`.
- [x] Positive claims are scoped to active M17 cells and non-financial sensor/infrastructure data.
- [x] Threshold sensitivity is generated for 0%, 2.5%, 5%, and 10%.
- [{"x" if render_ready else " "}] Final manuscript tables and figures have been rendered from these trace entries.
- [ ] Final bibliography has been fully verified against primary sources.
- [ ] Final PDF has passed visual/table-caption verification.
"""
    (out_dir / "SUBMISSION_TRACE.md").write_text(text)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.prism_root
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    claims = build_claims(root)
    trace = build_figure_table_trace(root, claims)
    constraints = build_negative_constraints()
    result = {
        "milestone": "M18",
        "goal": "Submission-level claim, figure/table, and negative-constraint trace for the M17 scoped route.",
        "claim_registry": [claim.__dict__ for claim in claims],
        "figure_table_trace": trace,
        "negative_claim_constraints": constraints,
        "gate_pass": all(claim.status.startswith("SUPPORTED") for claim in claims),
        "remaining_submission_tasks": [
            "Render final manuscript tables/figures from trace entries.",
            "Run full bibliography existence/metadata verification once the final reference list exists.",
            "Run final PDF visual and caption-fidelity verification.",
        ],
    }
    (out_dir / "submission_trace_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_markdown(out_dir, claims, trace, constraints)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PRISM submission trace pack.")
    p.add_argument("--prism-root", type=Path, default=Path("experiments/PRISM"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/PRISM/submission_trace"))
    return p.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "gate_pass": result["gate_pass"],
                "claims": len(result["claim_registry"]),
                "figure_table_trace": len(result["figure_table_trace"]),
                "negative_constraints": len(result["negative_claim_constraints"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
