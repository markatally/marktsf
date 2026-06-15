"""M13 block/FDR significance for the high-dimensional sensor route.

The route is predeclared as non-financial, high-dimensional sensor or
infrastructure datasets with train-only covariate selection:

* Electricity and Traffic from M11;
* PEMS04 and PEMS08 added as independent traffic-sensor confirmations.

It uses the same paired horizon-block sign-flip tests and BH/FDR correction as
M12, but it does not inherit ETT/Weather cells that are outside this narrowed
high-dimensional sensor claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.PRISM.calibrated_stack_significance import analyze_one, bh_fdr
from experiments.PRISM.champion_risk_gate import json_safe
from experiments.PRISM.router_viability import default_specs


INFRASTRUCTURE_DATASETS = ("Electricity", "Traffic")
TRAFFIC_SENSOR_DATASETS = ("PEMS04", "PEMS08")
COMPARISONS = ("stack_vs_validation_single", "stack_vs_fixed_share", "stack_vs_descriptor_ridge")


def run(args: argparse.Namespace) -> dict[str, object]:
    route_specs = []
    for horizon in (96, 192):
        route_specs.extend(
            (
                horizon,
                spec,
                args.infrastructure_results_root,
            )
            for spec in default_specs(args.infrastructure_oracle_root, horizon=horizon, datasets=INFRASTRUCTURE_DATASETS)
        )
        route_specs.extend(
            (
                horizon,
                spec,
                args.sensor_results_root,
            )
            for spec in default_specs(args.sensor_oracle_root, horizon=horizon, datasets=TRAFFIC_SENSOR_DATASETS)
        )

    rows = [
        analyze_one(
            spec,
            results_root=results_root,
            lookback=args.lookback,
            horizon=horizon,
            train_frac=args.train_frac,
            seed=args.seed + idx * 17,
            n_perm=args.n_perm,
        )
        for idx, (horizon, spec, results_root) in enumerate(route_specs)
    ]

    fdr_hypotheses = []
    for comparison in COMPARISONS:
        pvals = [row["tests"][comparison]["pvalue"] for row in rows]
        passes = bh_fdr(pvals, args.fdr_alpha)
        for row, passed in zip(rows, passes):
            fdr_hypotheses.append(
                {
                    "dataset": row["dataset"],
                    "horizon": row["horizon"],
                    "comparison": comparison,
                    "pvalue": row["tests"][comparison]["pvalue"],
                    "fdr_pass": bool(passed),
                    "improvement_pct": row["tests"][comparison]["improvement_pct"],
                }
            )
    pass_counts = {
        comparison: sum(
            1 for item in fdr_hypotheses if item["comparison"] == comparison and item["fdr_pass"]
        )
        for comparison in COMPARISONS
    }
    result = {
        "milestone": "M13",
        "goal": "High-dimensional non-financial sensor/infrastructure calibrated-stack significance.",
        "scope": {
            "datasets": list(INFRASTRUCTURE_DATASETS + TRAFFIC_SENSOR_DATASETS),
            "horizons": [96, 192],
            "route_rule": "non-financial high-dimensional sensor/infrastructure datasets with train-only covariate selection",
        },
        "rows": rows,
        "fdr_alpha": args.fdr_alpha,
        "n_perm": args.n_perm,
        "fdr_hypotheses": fdr_hypotheses,
        "fdr_pass_counts": pass_counts,
        "gate_pass": all(count == len(rows) for count in pass_counts.values()),
        "gate": "All 8 dataset-horizon cells must pass BH/FDR against validation-single, delayed Fixed-Share, and descriptor ridge.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "sensor_stack_significance_summary.json").write_text(
        json.dumps(json_safe(result), allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M13 high-dimensional sensor calibrated-stack significance.")
    p.add_argument("--infrastructure-oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift_nonfinancial"))
    p.add_argument("--infrastructure-results-root", type=Path, default=Path("external/TSLib/results_prism_nonfinancial"))
    p.add_argument("--sensor-oracle-root", type=Path, default=Path("experiments/PRISM/oracle_drift_sensor"))
    p.add_argument("--sensor-results-root", type=Path, default=Path("external/TSLib/results_prism_sensor"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/PRISM/sensor_stack_significance"))
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--n-perm", type=int, default=9999)
    p.add_argument("--seed", type=int, default=20260615)
    return p.parse_args()


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "gate_pass": result["gate_pass"],
                "fdr_pass_counts": result["fdr_pass_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
