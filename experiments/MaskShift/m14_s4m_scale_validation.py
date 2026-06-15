"""M14 — S4M scale validation under a larger reduced MaskShift protocol.

M12 closes the official S4M coverage gap with a fast three-seed adaptation.
This milestone probes the most obvious reviewer concern about that run: whether
the negative/contrastive conclusion is an artifact of using only eight channels
and very few windows. The run keeps the same official S4M model class and
MaskShift encoder-mask protocol, but doubles channels and train/test windows.
It is still not the full S4M benchmark protocol.
"""

from __future__ import annotations

import json

from . import m12_official_s4m_baseline as m12
from .maskshift_core import MECHANISMS, ExperimentConfig, ensure_dir, write_json


EXP_DIR = m12.EXP_DIR
OUT_DIR = ensure_dir(EXP_DIR / "m14_s4m_scale_validation")
TABLE_DIR = ensure_dir(EXP_DIR / "tables")

DATASETS = ["Weather", "Electricity"]
SEED_OFFSETS = [0, 10_000, 20_000]
MAX_ROWS = 5000
MAX_CHANNELS = 16
TRAIN_SAMPLES = 64
TEST_SAMPLES = 48


def write_table(summary: dict) -> None:
    lines = [
        "# M14 S4M scale-validation table",
        "",
        "S4M is evaluated with 16 channels, 64 train windows, 48 test windows, and three seed offsets. This remains a MaskShift-protocol adaptation, not the full S4M benchmark protocol.",
        "",
        "| Dataset | Backbone | Max degradation mean [95% CI] | Max abs delta [95% CI] | Strongest mechanism mode | Kruskal p mean [95% CI] | Gate seeds |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in summary["datasets"]:
        lines.append(
            "| {dataset} | {backbone} | {rel} | {abs_delta} | {strongest} | {p_value} | {gates}/{n} |".format(
                dataset=row["dataset"],
                backbone=m12.BACKBONE,
                rel=m12.fmt_ci(row["max_relative_degradation_ci"], pct=True),
                abs_delta=m12.fmt_ci(row["max_absolute_delta_ci"]),
                strongest=row["strongest_mechanism"],
                p_value=m12.fmt_ci(row["kruskal_p_ci"], lower=0.0, upper=1.0),
                gates=row["gate_pass_count"],
                n=row["n_seeds"],
            )
        )
    (TABLE_DIR / "m14_s4m_scale_table.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    old_rows = m12.MAX_ROWS
    old_channels = m12.MAX_CHANNELS
    m12.MAX_ROWS = MAX_ROWS
    m12.MAX_CHANNELS = MAX_CHANNELS
    cfg = ExperimentConfig(max_train_samples=TRAIN_SAMPLES, max_test_samples=TEST_SAMPLES)
    datasets = []
    seed_runs = []
    errors = []
    try:
        for dataset_index, dataset in enumerate(DATASETS):
            dataset_seed_runs = []
            for seed_index, offset in enumerate(SEED_OFFSETS):
                seed_offset = dataset_index * 1000 + offset
                try:
                    run = m12.run_dataset(dataset, cfg, seed_offset=seed_offset)
                    run["seed_index"] = seed_index
                    dataset_seed_runs.append(run)
                    seed_runs.append(run)
                except Exception as exc:
                    errors.append({"dataset": dataset, "seed_offset": seed_offset, "error": repr(exc)})
            if dataset_seed_runs:
                datasets.append(m12.aggregate_dataset(dataset, dataset_seed_runs))
    finally:
        m12.MAX_ROWS = old_rows
        m12.MAX_CHANNELS = old_channels

    expected_runs = len(DATASETS) * len(SEED_OFFSETS)
    protocol_complete = (
        len(seed_runs) == expected_runs
        and len(datasets) == len(DATASETS)
        and all(len(row["rows"]) == len(MECHANISMS) for row in seed_runs)
    )
    mechanism_shift_gate = any(row["gate_pass"] for row in datasets)
    summary = {
        "milestone": "M14",
        "status": "PASS_S4M_SCALE_VALIDATION" if protocol_complete else "HOLD_S4M_SCALE_VALIDATION",
        "device": str(m12.DEVICE),
        "s4m_revision": m12.git_revision(m12.S4M_DIR),
        "s4m_local_diff_stat": m12.git_diff_stat(m12.S4M_DIR),
        "device_port_patch": "external/S4M/s4m/model/Bank.py replaces a hard-coded .cuda() memory fetch with .to(Q.device); architecture and forward equations otherwise unchanged.",
        "config": {
            **cfg.__dict__,
            "max_rows": MAX_ROWS,
            "max_channels": MAX_CHANNELS,
            "epochs": 1,
            "batch_size": 8,
            "scale_factor_vs_m12": {
                "channels": MAX_CHANNELS / 8,
                "train_samples": TRAIN_SAMPLES / 32,
                "test_samples": TEST_SAMPLES / 24,
            },
        },
        "seed_offsets": SEED_OFFSETS,
        "backbone": m12.BACKBONE,
        "datasets": datasets,
        "seed_runs": seed_runs,
        "errors": errors,
        "m14_gate": bool(protocol_complete),
        "mechanism_shift_gate": bool(mechanism_shift_gate),
        "protocol_note": "Imports official S4M model class; MaskShift loop masks encoder inputs only and keeps forecast targets clean; larger reduced setting uses 16 channels, 64 train windows, 48 test windows, and three seed offsets.",
    }
    write_json(OUT_DIR / "s4m_scale_validation_summary.json", summary)
    if datasets:
        write_table(summary)
    print(
        json.dumps(
            {
                "milestone": "M14",
                "status": summary["status"],
                "device": str(m12.DEVICE),
                "errors": errors,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
