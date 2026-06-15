"""M8 — Mechanism decomposition excluding retirement-dominated effects."""

from __future__ import annotations

import json
from pathlib import Path

from .maskshift_core import ensure_dir, write_json


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m8_mechanism_decomposition")
NON_RETIREMENT_MECHANISMS = ["value_high", "volatility", "blackout"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    m1 = load_json(EXP_DIR / "m1_mechanism_audit" / "m1_summary.json")
    dataset_rows = []
    mechanism_rows = []
    for ds in m1["datasets"]:
        rows = {row["mechanism"]: row for row in ds["mechanism_effect_rows"]}
        non_ret = [rows[m] for m in NON_RETIREMENT_MECHANISMS if m in rows]
        max_non_ret = max(row["relative_degradation_vs_mcar"] for row in non_ret)
        worst_non_ret_tau = min(row["kendall_tau_vs_mcar_rank"] for row in non_ret)
        strongest_non_ret = max(non_ret, key=lambda row: row["relative_degradation_vs_mcar"])
        retirement = rows.get("retirement")
        retirement_degradation = (
            retirement["relative_degradation_vs_mcar"] if retirement is not None else None
        )
        non_ret_gate = max_non_ret > 0.05 and worst_non_ret_tau <= 0.5
        dataset_rows.append(
            {
                "dataset": ds["dataset"],
                "max_non_retirement_degradation": max_non_ret,
                "worst_non_retirement_tau": worst_non_ret_tau,
                "strongest_non_retirement_mechanism": strongest_non_ret["mechanism"],
                "retirement_degradation": retirement_degradation,
                "non_retirement_gate": bool(non_ret_gate),
            }
        )
        for mech in NON_RETIREMENT_MECHANISMS + ["retirement"]:
            if mech in rows:
                mechanism_rows.append(
                    {
                        "dataset": ds["dataset"],
                        "mechanism": mech,
                        "relative_degradation_vs_mcar": rows[mech]["relative_degradation_vs_mcar"],
                        "kendall_tau_vs_mcar_rank": rows[mech]["kendall_tau_vs_mcar_rank"],
                    }
                )

    by_mechanism = {}
    for mech in NON_RETIREMENT_MECHANISMS + ["retirement"]:
        subset = [row for row in mechanism_rows if row["mechanism"] == mech]
        by_mechanism[mech] = {
            "mean_relative_degradation": sum(row["relative_degradation_vs_mcar"] for row in subset) / len(subset),
            "max_relative_degradation": max(row["relative_degradation_vs_mcar"] for row in subset),
            "min_rank_tau": min(row["kendall_tau_vs_mcar_rank"] for row in subset),
            "positive_degradation_count": sum(row["relative_degradation_vs_mcar"] > 0.05 for row in subset),
        }

    summary = {
        "milestone": "M8",
        "status": "PASS_NON_RETIREMENT_DECOMPOSITION"
        if sum(row["non_retirement_gate"] for row in dataset_rows) >= 2
        else "HOLD_RETIREMENT_DOMINANCE",
        "non_retirement_mechanisms": NON_RETIREMENT_MECHANISMS,
        "dataset_rows": dataset_rows,
        "mechanism_rows": mechanism_rows,
        "by_mechanism": by_mechanism,
        "gate_rule": ">=2 datasets with >5% non-retirement degradation and Kendall tau<=0.5",
    }
    write_json(OUT_DIR / "mechanism_decomposition_summary.json", summary)

    lines = ["# MaskShift M8 — Mechanism Decomposition", ""]
    lines.append(
        "M8 checks whether the mechanism-shift result survives after excluding sensor retirement, the most visually obvious outage mechanism."
    )
    lines.append("")
    lines.append("| Dataset | Strongest non-retirement mechanism | Max non-ret degradation | Worst non-ret tau | Retirement degradation | Gate |")
    lines.append("|---|---|---:|---:|---:|---|")
    for row in dataset_rows:
        retirement = row["retirement_degradation"]
        retirement_text = "N/A" if retirement is None else f"{retirement:.1%}"
        lines.append(
            "| {dataset} | {strongest_non_retirement_mechanism} | {max_non_retirement_degradation:.1%} | {worst_non_retirement_tau:.3f} | {retirement_text} | {gate} |".format(
                retirement_text=retirement_text,
                gate="PASS" if row["non_retirement_gate"] else "FAIL",
                **row,
            )
        )
    lines.append("")
    lines.append("## Mechanism-Level Summary")
    lines.append("")
    lines.append("| Mechanism | Mean degradation | Max degradation | Min tau | Positive count |")
    lines.append("|---|---:|---:|---:|---:|")
    for mech, vals in by_mechanism.items():
        lines.append(
            f"| {mech} | {vals['mean_relative_degradation']:.1%} | {vals['max_relative_degradation']:.1%} | {vals['min_rank_tau']:.3f} | {vals['positive_degradation_count']} |"
        )
    (EXP_DIR / "REPORT.md").write_text((EXP_DIR / "REPORT.md").read_text() + "\n\n" + "\n".join(lines) + "\n")
    print(json.dumps({"milestone": "M8", "status": summary["status"]}, indent=2))


if __name__ == "__main__":
    main()
