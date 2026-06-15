"""M16 — Official TSLib architecture coverage on all MaskShift datasets.

M9/M10 carry the main official-architecture rank-reversal evidence on Weather
and Electricity. M16 addresses a different reviewer concern: whether official
PatchTST/TimeXer coverage was limited to the two positive datasets. It keeps the
M9 MaskShift encoder-mask protocol and adds Traffic and AirConvection as
external-validity coverage, reporting mixed/negative outcomes rather than
requiring every dataset to pass the mechanism-shift gate.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import m9_official_tslib_reproduction as m9
from .maskshift_core import ExperimentConfig, ensure_dir, write_json


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m16_official_tslib_full_coverage")
TABLE_DIR = ensure_dir(EXP_DIR / "tables")

ALL_DATASETS = ["Weather", "Electricity", "Traffic", "AirConvection"]
NEW_DATASETS = ["Traffic", "AirConvection"]
SEED_OFFSETS = {
    "Weather": 0,
    "Electricity": 1000,
    "Traffic": 2000,
    "AirConvection": 3000,
}


def load_m9_datasets() -> list[dict]:
    path = EXP_DIR / "m9_official_tslib_reproduction" / "official_tslib_reproduction_summary.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data.get("datasets", [])


def dataset_ok(row: dict) -> bool:
    backbones = {item.get("backbone") for item in row.get("rows", [])}
    mechanisms = {item.get("test_mechanism") for item in row.get("rows", [])}
    return backbones == set(m9.BACKBONES) and mechanisms == set(m9.MECHANISMS)


def write_table(summary: dict) -> None:
    lines = [
        "# M16 official TSLib full-coverage table",
        "",
        "PatchTST and TimeXer are imported from pinned TSLib model classes. Weather/Electricity rows reuse M9; Traffic/AirConvection rows are new M16 coverage runs under the same MaskShift encoder-mask protocol.",
        "",
        "| Dataset | Source | Official architecture classes | Max degradation | Worst tau | ANOVA p | Gate |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary["datasets"]:
        lines.append(
            "| {dataset} | {source} | {backbones} | {degradation:.1%} | {tau:.3f} | {p:.3g} | {gate} |".format(
                dataset=row["dataset"],
                source=row["source"],
                backbones=", ".join(m9.BACKBONES),
                degradation=row["max_relative_degradation"],
                tau=row["worst_rank_tau"],
                p=row["anova_p"],
                gate="PASS" if row["gate_pass"] else "MIXED/NEGATIVE",
            )
        )
    (TABLE_DIR / "m16_official_tslib_full_coverage_table.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    cfg = ExperimentConfig(max_train_samples=360, max_test_samples=160)
    m9_rows_by_dataset = {row["dataset"]: row for row in load_m9_datasets()}
    datasets = []
    errors = []

    for dataset in ALL_DATASETS:
        if dataset in m9_rows_by_dataset:
            row = dict(m9_rows_by_dataset[dataset])
            row["source"] = "M9"
            datasets.append(row)
            continue
        try:
            row = m9.run_dataset(dataset, cfg, seed_offset=SEED_OFFSETS[dataset])
            row["source"] = "M16"
            datasets.append(row)
        except Exception as exc:
            errors.append({"dataset": dataset, "error": repr(exc)})

    dataset_names = {row["dataset"] for row in datasets}
    coverage_complete = (
        dataset_names == set(ALL_DATASETS)
        and all(dataset_ok(row) for row in datasets)
        and not errors
    )
    gate_pass_count = sum(bool(row.get("gate_pass")) for row in datasets)
    summary = {
        "milestone": "M16",
        "status": "PASS_OFFICIAL_TSLIB_FULL_COVERAGE" if coverage_complete else "HOLD_OFFICIAL_TSLIB_FULL_COVERAGE",
        "device": str(m9.DEVICE),
        "tslib_revision": m9.git_revision(m9.TSLIB_DIR),
        "backbones": m9.BACKBONES,
        "config": cfg.__dict__,
        "datasets": datasets,
        "new_datasets": NEW_DATASETS,
        "gate_pass_count": gate_pass_count,
        "coverage_complete": bool(coverage_complete),
        "errors": errors,
        "protocol_note": "Extends official PatchTST/TimeXer MaskShift-protocol coverage to Traffic and AirConvection. Mixed/negative results are expected evidence, not a failure of the coverage milestone.",
    }
    write_json(OUT_DIR / "official_tslib_full_coverage_summary.json", summary)
    if datasets:
        write_table(summary)
    print(
        json.dumps(
            {
                "milestone": "M16",
                "status": summary["status"],
                "device": str(m9.DEVICE),
                "gate_pass_count": gate_pass_count,
                "errors": errors,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
