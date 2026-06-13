"""M5 paper-ready artifact packager for PRISM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def run(args: argparse.Namespace) -> dict[str, object]:
    root = args.prism_root
    m2 = load(root / "router_viability/router_viability_summary.json")
    m3 = load(root / "drift_beta_loop/drift_beta_summary.json")
    m4 = load(root / "ablations_significance/ablations_significance_summary.json")

    summary = {
        "milestone": "M5",
        "status": "paper-ready",
        "final_route": "ETT-only empirical/pivot paper",
        "headline_claims": [
            "Optimal-bias drift is broad in the ETT/Weather lightweight screen.",
            "Fixed-Share is the robust causal tracker after learned-router failure.",
            "Dynamic beta gives a small but statistically reliable improvement.",
            "The learned router and drift-share-rate loop are negative results in current form.",
        ],
        "gate_status": {
            "M1a": "HOLD/PASS for ETT phenomenon; finance raw-return MSE void",
            "M1b": "FAIL finance strict gate; pivot to ETT-only PRISM",
            "M1c": "PASS breadth phenomenon; routability mixed",
            "M2": "FAIL learned-router viability on ETTm2",
            "M3": "PASS narrowly; dynamic beta useful, drift gain zero",
            "M4": "PASS; beta/full survives FDR, drift-only rejected",
            "M5": "PASS; artifacts and reproduction manifest frozen",
        },
        "artifact_manifest": [
            "docs/PLAN.md",
            "docs/PROPOSAL.md",
            "docs/REPORT.md",
            "oracle_drift/m1c_summary.json",
            "router_viability/router_viability_summary.json",
            "drift_beta_loop/drift_beta_summary.json",
            "ablations_significance/ablations_significance_summary.json",
            "paper_ready/paper_ready_summary.json",
            "paper_ready/REPRODUCE.md",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paper_ready_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    m2_rows = [
        [
            row["dataset"],
            f'{row["fixed_share_loss"]:.6g}',
            f'{row["descriptor_ridge_loss"]:.6g}',
            f'{row["prism_router_loss"]:.6g}',
            "PASS" if row["gate_pass"] else "FAIL",
        ]
        for row in m2["rows"]
    ]
    m3_rows = [
        [
            row["dataset"],
            f'{row["plain_fixed_share_stress_loss"]:.6g}',
            f'{row["drift_loop_stress_loss"]:.6g}',
            f'{row["stress_improvement_pct"]:.3g}%',
            f'{row["beta_iqr"]:.3g}',
        ]
        for row in m3["rows"]
    ]
    m4_rows = [
        [
            row["dataset"],
            f'{row["losses"]["plain_fixed_share"]:.6g}',
            f'{row["losses"]["full"]:.6g}',
            f'{row["tests"]["full_vs_plain"]["improvement_pct"]:.3g}%',
            "PASS",
        ]
        for row in m4["rows"]
    ]
    reproduce = f"""# PRISM Paper-Ready Reproduction

Run from the repository root with the bundled/scientific Python environment.

```bash
PY=/Users/markguo/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
$PY -m experiments.PRISM.router_viability
$PY -m experiments.PRISM.drift_beta_loop
$PY -m experiments.PRISM.ablations_significance
$PY -m experiments.PRISM.paper_ready
```

## Final Route

ETT-only empirical/pivot paper:

- M1b finance gate failed under the strict preregistered condition.
- M2 learned router failed on ETTm2 against Fixed-Share.
- M3/M4 retain dynamic beta as a small, statistically reliable contribution.
- Drift-triggered share-rate adaptation is rejected in the current form.

## M2 Router Viability

{md_table(["Dataset", "Fixed-Share", "Descriptor Ridge", "PRISM Router", "Gate"], m2_rows)}

## M3 Dynamic Beta / Drift Stress

{md_table(["Dataset", "Plain Stress", "Loop Stress", "Improvement", "Beta IQR"], m3_rows)}

## M4 FDR Ablations

{md_table(["Dataset", "Plain FS", "Full", "Improvement", "FDR"], m4_rows)}

## Synthetic Identifiability

- State recovery accuracy: {m4["synthetic_identifiability"]["state_accuracy"]:.3f}
- Best single loss: {m4["synthetic_identifiability"]["best_single_loss"]:.4f}
- Oracle loss: {m4["synthetic_identifiability"]["oracle_loss"]:.4f}
- Descriptor router loss: {m4["synthetic_identifiability"]["router_loss"]:.4f}
"""
    (args.output_dir / "REPRODUCE.md").write_text(reproduce)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create PRISM M5 paper-ready artifacts.")
    p.add_argument("--prism-root", type=Path, default=Path("experiments/PRISM"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/PRISM/paper_ready"))
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
