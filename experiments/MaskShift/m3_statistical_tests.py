"""M3 — Consolidated statistical tests and claim-family FDR."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from .maskshift_core import OPERATIONAL_MECHANISMS, benjamini_hochberg, ensure_dir, write_json


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m3_statistical_tests")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    m1 = load_json(EXP_DIR / "m1_mechanism_audit" / "m1_summary.json")
    m2 = load_json(EXP_DIR / "m2_typed_head" / "m2_summary.json")

    m1_p = [ds["anova"]["p_value"] for ds in m1["datasets"]]
    m1_q = benjamini_hochberg(m1_p)
    m1_rows = []
    for ds, q in zip(m1["datasets"], m1_q):
        worst_tau = min(row["kendall_tau_vs_mcar_rank"] for row in ds["mechanism_effect_rows"])
        max_degrade = max(row["relative_degradation_vs_mcar"] for row in ds["mechanism_effect_rows"])
        m1_rows.append(
            {
                "dataset": ds["dataset"],
                "eta_squared": ds["anova"]["eta_squared"],
                "p_value": ds["anova"]["p_value"],
                "q_value": q,
                "worst_rank_tau": worst_tau,
                "max_relative_degradation": max_degrade,
                "gate_pass": ds["gate_pass"],
                "fdr_pass": bool(q <= 0.05),
            }
        )

    # Dataset-level paired test: topology-only AUC vs typed AUC over operational
    # mechanisms, with randomized-label ablation reported separately.
    m2_rows = []
    typed_improvements = []
    for ds in m2["datasets"]:
        rows = ds["rows"]
        topo = {
            r["mechanism"]: r["mse"]
            for r in rows
            if r["variant"] == "topology" and r["mechanism"] in OPERATIONAL_MECHANISMS
        }
        typed = {
            r["mechanism"]: r["mse"]
            for r in rows
            if r["variant"] == "typed" and r["mechanism"] in OPERATIONAL_MECHANISMS
        }
        random_label = {
            r["mechanism"]: r["mse"]
            for r in rows
            if r["variant"] == "typed_random_label" and r["mechanism"] in OPERATIONAL_MECHANISMS
        }
        mechanisms = [m for m in OPERATIONAL_MECHANISMS if m in topo and m in typed]
        diff = np.asarray([topo[m] - typed[m] for m in mechanisms], dtype=float)
        random_diff = np.asarray([topo[m] - random_label[m] for m in mechanisms], dtype=float)
        if len(diff) > 1:
            t_stat, p_val = stats.ttest_1samp(diff, popmean=0.0, alternative="greater")
            try:
                w_stat, w_p = stats.wilcoxon(diff, alternative="greater")
            except ValueError:
                w_stat, w_p = float("nan"), 1.0
        else:
            t_stat, p_val, w_stat, w_p = float("nan"), 1.0, float("nan"), 1.0
        typed_improvements.extend(diff.tolist())
        m2_rows.append(
            {
                "dataset": ds["dataset"],
                "mean_topology_minus_typed_mse": float(diff.mean()),
                "mean_topology_minus_random_label_mse": float(random_diff.mean()),
                "typed_reduction": ds["typed_vs_topology_reduction"],
                "clean_mcar_cost": ds["clean_mcar_cost"],
                "paired_t_p": float(p_val),
                "wilcoxon_p": float(w_p),
                "gate_pass": ds["gate_pass"],
            }
        )
    m2_q = benjamini_hochberg([r["paired_t_p"] for r in m2_rows])
    for row, q in zip(m2_rows, m2_q):
        row["q_value"] = q
        row["fdr_pass"] = bool(q <= 0.05)

    if typed_improvements:
        overall_t, overall_p = stats.ttest_1samp(np.asarray(typed_improvements), popmean=0.0, alternative="greater")
    else:
        overall_t, overall_p = float("nan"), 1.0

    h1_pass = sum(r["fdr_pass"] and r["eta_squared"] >= 0.30 for r in m1_rows) >= 2
    h2_pass = sum(r["worst_rank_tau"] <= 0.5 for r in m1_rows) >= 2
    h3_pass = sum(r["fdr_pass"] and r["typed_reduction"] >= 0.20 and r["clean_mcar_cost"] <= 0.02 for r in m2_rows) >= 2
    summary = {
        "milestone": "M3",
        "status": "PASS_STRONG_EVIDENCE" if (h1_pass and h2_pass and h3_pass) else "MIXED_EVIDENCE",
        "claim_tests": {
            "H1_mechanism_over_rate": h1_pass,
            "H2_rank_instability": h2_pass,
            "H3_typed_minimal_correction": h3_pass,
            "overall_typed_improvement_p": float(overall_p),
            "overall_typed_improvement_t": float(overall_t),
        },
        "m1_rows": m1_rows,
        "m2_rows": m2_rows,
        "fdr_method": "Benjamini-Hochberg within claim family",
    }
    write_json(OUT_DIR / "m3_summary.json", summary)

    lines = ["# MaskShift M3 — Statistical Consolidation", ""]
    lines.append("| Claim | Pass | Evidence |")
    lines.append("|---|---|---|")
    lines.append(f"| H1 mechanism over rate | {h1_pass} | {sum(r['fdr_pass'] and r['eta_squared'] >= 0.30 for r in m1_rows)} datasets with q<=0.05 and eta^2>=0.30 |")
    lines.append(f"| H2 rank instability | {h2_pass} | {sum(r['worst_rank_tau'] <= 0.5 for r in m1_rows)} datasets with worst Kendall tau<=0.5 vs MCAR rank |")
    lines.append(f"| H3 typed correction | {h3_pass} | {sum(r['fdr_pass'] and r['typed_reduction'] >= 0.20 and r['clean_mcar_cost'] <= 0.02 for r in m2_rows)} datasets with FDR-backed >=20% typed reduction |")
    (EXP_DIR / "REPORT.md").write_text((EXP_DIR / "REPORT.md").read_text() + "\n\n" + "\n".join(lines) + "\n")
    print(json.dumps({"milestone": "M3", "status": summary["status"], "out": str(OUT_DIR / "m3_summary.json")}, indent=2))


if __name__ == "__main__":
    main()

