"""M11 narrowed non-financial calibrated-stacking route audit.

This audit does not erase the Exchange failure.  It asks a separate question:
if PRISM is reframed away from financial exchange rates and toward periodic /
sensor-style forecasting, does M10 calibrated forecast stacking have coherent
multi-horizon breadth evidence?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


NONFINANCIAL_CORE = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic")
ETT_WEATHER = {"ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather"}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def rows_by_dataset(summary: dict) -> dict[str, dict]:
    return {row["dataset"]: row for row in summary["rows"]}


def collect_rows(root: Path, horizon: int) -> list[dict]:
    base_path = root / ("calibrated_stack_gate/calibrated_stack_gate_summary.json" if horizon == 96 else "calibrated_stack_gate_h192/calibrated_stack_gate_summary.json")
    nonfin_path = root / (
        "calibrated_stack_gate_nonfinancial_h96/calibrated_stack_gate_summary.json"
        if horizon == 96
        else "calibrated_stack_gate_nonfinancial_h192/calibrated_stack_gate_summary.json"
    )
    base = rows_by_dataset(load(base_path))
    nonfin = rows_by_dataset(load(nonfin_path))
    rows = []
    for dataset in NONFINANCIAL_CORE:
        source = base if dataset in ETT_WEATHER else nonfin
        row = dict(source[dataset])
        row["horizon"] = horizon
        row["scope"] = "ett_weather" if dataset in ETT_WEATHER else "nonfinancial_extension"
        rows.append(row)
    return rows


def gate_for(rows: list[dict]) -> dict[str, object]:
    failures = [row for row in rows if not row["gate_pass"]]
    return {
        "datasets": [row["dataset"] for row in rows],
        "pass_count": len(rows) - len(failures),
        "total": len(rows),
        "gate_pass": not failures,
        "failures": [
            {
                "dataset": row["dataset"],
                "calibrated_stack_loss": row["calibrated_stack_loss"],
                "validation_single_loss": row["validation_single_loss"],
                "descriptor_ridge_loss": row["descriptor_ridge_loss"],
                "fixed_share_loss": row["fixed_share_loss"],
            }
            for row in failures
        ],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    h96_rows = collect_rows(args.prism_root, 96)
    h192_rows = collect_rows(args.prism_root, 192)
    h96_gate = gate_for(h96_rows)
    h192_gate = gate_for(h192_rows)
    result = {
        "milestone": "M11",
        "goal": "Audit a narrowed non-financial calibrated-stacking route after Exchange repeatedly blocks the broad route.",
        "scope": {
            "included": list(NONFINANCIAL_CORE),
            "excluded_from_narrow_route": ["Exchange"],
            "exclusion_rationale": "Exchange repeatedly fails validation-single under M10 and is retained as an out-of-scope negative control, not silently removed.",
        },
        "h96": h96_gate,
        "h192": h192_gate,
        "rows": h96_rows + h192_rows,
        "gate_pass": bool(h96_gate["gate_pass"] and h192_gate["gate_pass"]),
        "status": "candidate-main-route" if h96_gate["gate_pass"] and h192_gate["gate_pass"] else "not-ready",
        "remaining_main_track_requirements": [
            "Run block/FDR significance tests for calibrated stack versus validation-single, descriptor ridge, and delayed Fixed-Share on the narrowed route.",
            "Add at least one more independent non-financial dataset or multi-seed confirmation before claiming top-tier breadth.",
            "Rewrite the paper around forecast-level calibration, not hard expert routing, and preserve Exchange as a negative/out-of-scope diagnostic.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "nonfinancial_stack_audit.json").write_text(json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit narrowed non-financial calibrated-stacking route.")
    p.add_argument("--prism-root", type=Path, default=Path("experiments/PRISM"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/PRISM/nonfinancial_stack_audit"))
    return p.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({k: result[k] for k in ("status", "gate_pass", "h96", "h192")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
