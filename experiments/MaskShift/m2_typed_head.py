"""M2 — Minimal typed/topology correction audit."""

from __future__ import annotations

import json
from pathlib import Path

from .maskshift_core import (
    DEFAULT_DATASETS,
    MECHANISMS,
    OPERATIONAL_MECHANISMS,
    ExperimentConfig,
    degradation_auc,
    ensure_dir,
    make_dataset_splits,
    train_mixed_variant,
    write_json,
)


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m2_typed_head")


def run_dataset(dataset: str, cfg: ExperimentConfig, seed_offset: int = 0) -> dict:
    ds = make_dataset_splits(dataset, cfg)
    rows = []
    for variant, random_labels in [
        ("topology", False),
        ("typed", False),
        ("typed", True),
    ]:
        variant_name = "typed_random_label" if random_labels else variant
        variant_rows = train_mixed_variant(
            ds["values"],
            ds["split_idx"],
            ds["train_origins"],
            ds["test_origins"],
            cfg,
            MECHANISMS,
            variant=variant,
            seed=cfg.seed + seed_offset,
            randomize_labels=random_labels,
        )
        for row in variant_rows:
            row["dataset"] = dataset
            row["variant"] = variant_name
        rows.extend(variant_rows)

    by_variant = {}
    for row in rows:
        by_variant.setdefault(row["variant"], []).append(row)
    topo_auc = degradation_auc(by_variant["topology"])
    typed_auc = degradation_auc(by_variant["typed"])
    random_auc = degradation_auc(by_variant["typed_random_label"])
    reduction = (topo_auc - typed_auc) / max(topo_auc, 1e-9)
    random_reduction = (topo_auc - random_auc) / max(topo_auc, 1e-9)
    clean_cost = (
        next(r["mse"] for r in by_variant["typed"] if r["mechanism"] == "mcar")
        - next(r["mse"] for r in by_variant["topology"] if r["mechanism"] == "mcar")
    ) / max(next(r["mse"] for r in by_variant["topology"] if r["mechanism"] == "mcar"), 1e-9)
    gate_pass = reduction >= 0.20 and clean_cost <= 0.02 and reduction > random_reduction + 0.05
    return {
        "dataset": dataset,
        "rows": rows,
        "operational_degradation_auc": {
            "topology": topo_auc,
            "typed": typed_auc,
            "typed_random_label": random_auc,
        },
        "typed_vs_topology_reduction": float(reduction),
        "typed_random_label_reduction": float(random_reduction),
        "clean_mcar_cost": float(clean_cost),
        "gate_pass": bool(gate_pass),
        "gate_rule": "typed reduces operational MSE AUC by >=20%, clean MCAR cost <=2%, and beats randomized labels by >=5pp",
    }


def write_report(summary: dict) -> None:
    lines = ["# MaskShift M2 — Typed/Topology Head Audit", ""]
    lines.append("M2 trains mixed-mechanism forecasters and tests whether mechanism labels add value beyond topology-only features and randomized labels.")
    lines.append("")
    lines.append("| Dataset | AUC topology | AUC typed | Reduction | Clean cost | Random-label reduction | Gate |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for ds in summary["datasets"]:
        auc = ds["operational_degradation_auc"]
        lines.append(
            f"| {ds['dataset']} | {auc['topology']:.4f} | {auc['typed']:.4f} | {ds['typed_vs_topology_reduction']:.1%} | {ds['clean_mcar_cost']:.1%} | {ds['typed_random_label_reduction']:.1%} | {'PASS' if ds['gate_pass'] else 'FAIL'} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    if summary["m2_gate"]:
        lines.append("The minimal typed/topology correction passes on a majority of datasets, supporting the sufficiency claim for a small correction rather than a new backbone.")
    else:
        lines.append("The typed correction is not yet robust enough for a sufficiency headline. The benchmark/audit claim can still survive, but the model contribution must be weakened or expanded.")
    (EXP_DIR / "REPORT.md").write_text((EXP_DIR / "REPORT.md").read_text() + "\n\n" + "\n".join(lines) + "\n")


def main() -> None:
    cfg = ExperimentConfig()
    datasets = []
    for i, dataset in enumerate(DEFAULT_DATASETS):
        datasets.append(run_dataset(dataset, cfg, seed_offset=i * 1000))
    m2_gate = sum(ds["gate_pass"] for ds in datasets) >= 2
    summary = {
        "milestone": "M2",
        "status": "PASS_TYPED_MINIMAL_CORRECTION" if m2_gate else "HOLD_TYPED_CORRECTION",
        "config": cfg.__dict__,
        "mechanisms": MECHANISMS,
        "operational_mechanisms": OPERATIONAL_MECHANISMS,
        "datasets": datasets,
        "m2_gate": bool(m2_gate),
    }
    write_json(OUT_DIR / "m2_summary.json", summary)
    write_report(summary)
    print(json.dumps({"milestone": "M2", "status": summary["status"], "out": str(OUT_DIR / "m2_summary.json")}, indent=2))


if __name__ == "__main__":
    main()

