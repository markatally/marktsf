"""Render PRISM M17 manuscript tables and figure from traceable artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def fmt(value: float) -> str:
    return f"{value:.2f}"


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out) + "\n"


def table1(root: Path, out_dir: Path) -> list[dict[str, Any]]:
    summary = load(root / "practical_selective_horizon_affine/practical_selective_horizon_affine_summary.json")
    rows = []
    for row in summary["rows"]:
        vs = row["tests"]["selective_active_vs_validation_single"]
        fs = row["tests"]["selective_active_vs_fixed_share"]
        dr = row["tests"]["selective_active_vs_descriptor_ridge"]
        rows.append(
            {
                "dataset": row["dataset"],
                "horizon": row["horizon"],
                "active": "yes" if row["active"] else "no",
                "validation_single_model": row["validation_single_model"],
                "vs_validation_single_improvement_pct": fmt(vs["improvement_pct"]),
                "vs_validation_single_p": f"{vs['pvalue']:.4g}",
                "vs_fixed_share_improvement_pct": fmt(fs["improvement_pct"]),
                "vs_fixed_share_p": f"{fs['pvalue']:.4g}",
                "vs_descriptor_ridge_improvement_pct": fmt(dr["improvement_pct"]),
                "vs_descriptor_ridge_p": f"{dr['pvalue']:.4g}",
            }
        )
    headers = list(rows[0])
    write_csv(out_dir / "table1_m17_active_cells.csv", rows, headers)
    md_rows = [[row[h] for h in headers] for row in rows]
    (out_dir / "table1_m17_active_cells.md").write_text(markdown_table(headers, md_rows))
    return rows


def table2(root: Path, out_dir: Path) -> list[dict[str, Any]]:
    specs = [
        ("0", "practical_selective_horizon_affine_sensitivity_0"),
        ("2.5", "practical_selective_horizon_affine_sensitivity_2p5"),
        ("5", "practical_selective_horizon_affine"),
        ("10", "practical_selective_horizon_affine_sensitivity_10"),
    ]
    rows = []
    for threshold, dirname in specs:
        summary = load(root / dirname / "practical_selective_horizon_affine_summary.json")
        active_cells = ", ".join(f"{row['dataset']} H{row['horizon']}" for row in summary["rows"] if row["active"])
        counts = summary["active_fdr_pass_counts"]
        rows.append(
            {
                "min_effect_pct": threshold,
                "gate_pass": "yes" if summary["gate_pass"] else "no",
                "active_cells": f"{summary['active_count']}/{len(summary['rows'])}",
                "vs_validation_single_fdr": f"{counts['selective_active_vs_validation_single']}/{summary['active_count']}",
                "vs_fixed_share_fdr": f"{counts['selective_active_vs_fixed_share']}/{summary['active_count']}",
                "vs_descriptor_ridge_fdr": f"{counts['selective_active_vs_descriptor_ridge']}/{summary['active_count']}",
                "active_cell_list": active_cells,
            }
        )
    headers = list(rows[0])
    write_csv(out_dir / "table2_threshold_sensitivity.csv", rows, headers)
    md_rows = [[row[h] for h in headers] for row in rows]
    (out_dir / "table2_threshold_sensitivity.md").write_text(markdown_table(headers, md_rows))
    return rows


def figure1(root: Path, out_dir: Path) -> dict[str, Any]:
    milestones = [
        ("M2", load(root / "router_viability/router_viability_summary.json")["gate_pass"], "router"),
        ("M7", load(root / "router_viability_h192/router_viability_summary.json")["gate_pass"], "router H192"),
        ("M12", load(root / "calibrated_stack_significance/calibrated_stack_significance_summary.json")["gate_pass"], "stack FDR"),
        ("M15", load(root / "sensor_horizon_affine_significance/sensor_horizon_affine_significance_summary.json")["gate_pass"], "full sensor affine"),
        ("M16", load(root / "selective_horizon_affine/selective_horizon_affine_summary.json")["gate_pass"], "selective no-harm"),
        ("M17", load(root / "practical_selective_horizon_affine/practical_selective_horizon_affine_summary.json")["gate_pass"], "practical selective"),
    ]
    colors = ["#2e7d32" if passed else "#b3261e" for _, passed, _ in milestones]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    xs = list(range(len(milestones)))
    ax.bar(xs, [1] * len(xs), color=colors, edgecolor="#222222", linewidth=0.8)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([])
    ax.set_xticks(xs)
    ax.set_xticklabels([name for name, _, _ in milestones], fontsize=10)
    ax.set_title("PRISM route hardening: failed broad routes to M17 scoped selective pass", fontsize=11)
    for x, (name, passed, label) in zip(xs, milestones):
        ax.text(x, 0.52, "PASS" if passed else "FAIL", ha="center", va="center", color="white", weight="bold", fontsize=9)
        ax.text(x, 1.08, label, ha="center", va="bottom", fontsize=8, rotation=18)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_color("#444444")
    fig.tight_layout()
    png = out_dir / "figure1_route_hardening.png"
    fig.savefig(png, dpi=220)
    plt.close(fig)
    return {
        "path": str(png),
        "milestones": [
            {"milestone": name, "gate_pass": bool(passed), "label": label}
            for name, passed, label in milestones
        ],
    }


def validate_outputs(root: Path, out_dir: Path, table1_rows: list[dict[str, Any]], table2_rows: list[dict[str, Any]], fig_summary: dict[str, Any]) -> dict[str, Any]:
    m17 = load(root / "practical_selective_horizon_affine/practical_selective_horizon_affine_summary.json")
    active_rows = [row for row in table1_rows if row["active"] == "yes"]
    threshold_5 = next(row for row in table2_rows if row["min_effect_pct"] == "5")
    checks = {
        "table1_row_count_matches_m17": len(table1_rows) == len(m17["rows"]),
        "table1_active_count_matches_m17": len(active_rows) == m17["active_count"],
        "table2_threshold_5_gate_pass": threshold_5["gate_pass"] == "yes",
        "table2_threshold_5_active_count_matches_m17": threshold_5["active_cells"] == f"{m17['active_count']}/{len(m17['rows'])}",
        "figure1_file_exists": Path(fig_summary["path"]).exists(),
        "figure1_m17_pass": any(item["milestone"] == "M17" and item["gate_pass"] for item in fig_summary["milestones"]),
    }
    return {
        "gate_pass": all(checks.values()),
        "checks": checks,
        "table1": "table1_m17_active_cells.csv",
        "table2": "table2_threshold_sensitivity.csv",
        "figure1": fig_summary["path"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.prism_root
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t1 = table1(root, out_dir)
    t2 = table2(root, out_dir)
    fig = figure1(root, out_dir)
    result = validate_outputs(root, out_dir, t1, t2, fig)
    (out_dir / "render_validation_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render PRISM submission tables and route-hardening figure.")
    p.add_argument("--prism-root", type=Path, default=Path("experiments/PRISM"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/PRISM/submission_render"))
    return p.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
