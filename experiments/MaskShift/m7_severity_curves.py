"""M7 — Multi-rate severity curves for the MaskShift audit."""

from __future__ import annotations

import json
from pathlib import Path

from .m1_mechanism_audit import run_dataset
from .maskshift_core import ExperimentConfig, ensure_dir, write_json


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m7_severity_curves")
DATASETS = ["Weather", "Electricity", "Traffic", "AirConvection"]
RATES = [0.10, 0.20, 0.35, 0.50]


def main() -> None:
    rows = []
    for rate in RATES:
        cfg = ExperimentConfig(target_rate=rate)
        for i, dataset in enumerate(DATASETS):
            result = run_dataset(dataset, cfg, seed_offset=i * 1000 + int(rate * 1000))
            worst_tau = min(row["kendall_tau_vs_mcar_rank"] for row in result["mechanism_effect_rows"])
            max_degradation = max(row["relative_degradation_vs_mcar"] for row in result["mechanism_effect_rows"])
            rows.append(
                {
                    "rate": rate,
                    "dataset": dataset,
                    "aggregate_eta_squared": result["anova"]["aggregate_eta_squared"],
                    "window_eta_squared": result["anova"]["window_eta_squared"],
                    "p_value": result["anova"]["p_value"],
                    "worst_rank_tau": worst_tau,
                    "max_relative_degradation": max_degradation,
                    "gate_pass": result["gate_pass"],
                }
            )
    by_rate = {}
    for rate in RATES:
        subset = [r for r in rows if r["rate"] == rate]
        by_rate[str(rate)] = {
            "n_gate_pass": sum(r["gate_pass"] for r in subset),
            "mean_eta": sum(r["aggregate_eta_squared"] for r in subset) / len(subset),
            "min_rank_tau": min(r["worst_rank_tau"] for r in subset),
            "max_relative_degradation": max(r["max_relative_degradation"] for r in subset),
        }
    summary = {
        "milestone": "M7",
        "status": "PASS_SEVERITY_CURVES" if by_rate["0.35"]["n_gate_pass"] >= 2 and by_rate["0.5"]["n_gate_pass"] >= 2 else "HOLD_SEVERITY_CURVES",
        "rates": RATES,
        "datasets": DATASETS,
        "rows": rows,
        "by_rate": by_rate,
    }
    write_json(OUT_DIR / "severity_curves_summary.json", summary)
    print(json.dumps({"milestone": "M7", "status": summary["status"], "by_rate": by_rate}, indent=2))


if __name__ == "__main__":
    main()

