"""M1 — Forecast-risk audit under missingness-mechanism shift."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from .maskshift_core import (
    DEFAULT_DATASETS,
    MECHANISMS,
    OPERATIONAL_MECHANISMS,
    ExperimentConfig,
    ensure_dir,
    generate_mask,
    kendall_tau_between,
    make_dataset_splits,
    mask_stats,
    rank_models,
    train_predict_ridge,
    write_json,
)


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m1_mechanism_audit")
VARIANTS = ["zero", "ffill", "mask", "topology"]


def run_dataset(dataset: str, cfg: ExperimentConfig, seed_offset: int = 0) -> dict:
    ds = make_dataset_splits(dataset, cfg)
    values = ds["values"]
    split_idx = ds["split_idx"]
    train_origins = ds["train_origins"]
    test_origins = ds["test_origins"]
    train_mask = generate_mask(values, "mcar", cfg.target_rate, cfg.seed + seed_offset, split_idx=split_idx)

    rows = []
    pred_store = {}
    for j, test_mech in enumerate(MECHANISMS):
        test_mask = generate_mask(values, test_mech, cfg.target_rate, cfg.seed + seed_offset + 200 + j * 17, split_idx=split_idx)
        for variant in VARIANTS:
            y, pred, metrics = train_predict_ridge(
                values,
                values,
                train_mask,
                test_mask,
                train_origins,
                test_origins,
                cfg,
                train_mechanism="mcar",
                test_mechanism=test_mech,
                variant=variant,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "train_mechanism": "mcar",
                    "test_mechanism": test_mech,
                    "variant": variant,
                    **metrics,
                    "mask_stats": mask_stats(test_mask),
                }
            )
            pred_store[(test_mech, variant)] = (y, pred)

    ranks = rank_models(rows)
    best_mcar_variant = ranks["mcar"][0]
    rank_taus = {
        mech: kendall_tau_between(ranks["mcar"], ranks[mech])
        for mech in OPERATIONAL_MECHANISMS
    }
    mcar_mse = np.mean([r["mse"] for r in rows if r["test_mechanism"] == "mcar"])
    mech_effect_rows = []
    for mech in OPERATIONAL_MECHANISMS:
        mech_mse = np.mean([r["mse"] for r in rows if r["test_mechanism"] == mech])
        mech_effect_rows.append(
            {
                "mechanism": mech,
                "mean_mse": float(mech_mse),
                "relative_degradation_vs_mcar": float((mech_mse - mcar_mse) / max(mcar_mse, 1e-9)),
                "kendall_tau_vs_mcar_rank": rank_taus[mech],
            }
        )

    aggregate_groups = [
        [r["mse"] for r in rows if r["test_mechanism"] == mech]
        for mech in MECHANISMS
    ]
    aggregate_all = np.asarray([r["mse"] for r in rows], dtype=float)
    aggregate_ss_between = sum(
        len(g) * (np.mean(g) - np.mean(aggregate_all)) ** 2 for g in aggregate_groups
    )
    aggregate_ss_total = float(np.sum((aggregate_all - np.mean(aggregate_all)) ** 2))
    aggregate_eta2 = float(aggregate_ss_between / aggregate_ss_total) if aggregate_ss_total > 0 else 0.0

    # Mechanism-factor significance test on per-window squared losses for the MCAR-selected
    # variant.  Using sample losses rather than four aggregate model means gives
    # the test enough power while keeping the selection protocol honest.
    groups = []
    for mech in MECHANISMS:
        y, pred = pred_store[(mech, best_mcar_variant)]
        groups.append(((y - pred) ** 2).tolist())
    f_stat, p_val = stats.f_oneway(*groups)
    all_losses = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    mean_by_mech = {mech: float(np.mean(g)) for mech, g in zip(MECHANISMS, groups)}
    ss_between = sum(len(g) * (np.mean(g) - np.mean(all_losses)) ** 2 for g in groups)
    ss_total = float(np.sum((all_losses - np.mean(all_losses)) ** 2))
    eta2 = float(ss_between / ss_total) if ss_total > 0 else 0.0

    gate_pass = (
        aggregate_eta2 >= 0.30
        and p_val <= 0.05
        and any(x["kendall_tau_vs_mcar_rank"] <= 0.5 for x in mech_effect_rows)
        and any(x["relative_degradation_vs_mcar"] > 0.05 for x in mech_effect_rows)
    )
    return {
        "dataset": dataset,
        "dataset_meta": ds["meta"],
        "rows": rows,
        "ranks": ranks,
        "mechanism_effect_rows": mech_effect_rows,
        "anova": {
            "selected_variant": best_mcar_variant,
            "f_stat": float(f_stat),
            "p_value": float(p_val),
            "window_eta_squared": eta2,
            "aggregate_eta_squared": aggregate_eta2,
            "eta_squared": aggregate_eta2,
            "mean_by_mechanism": mean_by_mech,
        },
        "gate_pass": bool(gate_pass),
    }


def write_report(summary: dict) -> None:
    lines = ["# MaskShift M1 — Mechanism-Shift Forecast Audit", ""]
    lines.append("M1 trains all lightweight variants under MCAR masks and tests them under matched-rate operational mechanisms. It asks whether MCAR robustness certifies deployment masks.")
    lines.append("")
    lines.append("| Dataset | Mechanism eta^2 | ANOVA p | Worst rank tau vs MCAR | Gate |")
    lines.append("|---|---:|---:|---:|---|")
    for ds in summary["datasets"]:
        worst_tau = min(row["kendall_tau_vs_mcar_rank"] for row in ds["mechanism_effect_rows"])
        lines.append(
            f"| {ds['dataset']} | {ds['anova']['eta_squared']:.3f} | {ds['anova']['p_value']:.3g} | {worst_tau:.3f} | {'PASS' if ds['gate_pass'] else 'FAIL'} |"
        )
    lines.append("")
    lines.append("## Claim Interpretation")
    if summary["m1_gate"]:
        lines.append("M1 supports the audit thesis: mechanism identity changes forecast risk and can reorder model rankings even when the missing rate is matched.")
    else:
        lines.append("M1 is not yet strong enough for a headline claim. Treat any positive rows as pilot evidence and expand datasets/backbones before submission.")
    (EXP_DIR / "REPORT.md").write_text((EXP_DIR / "REPORT.md").read_text() + "\n\n" + "\n".join(lines) + "\n")


def main() -> None:
    cfg = ExperimentConfig()
    datasets = []
    for i, dataset in enumerate(DEFAULT_DATASETS):
        datasets.append(run_dataset(dataset, cfg, seed_offset=i * 1000))
    m1_gate = sum(ds["gate_pass"] for ds in datasets) >= 2
    summary = {
        "milestone": "M1",
        "status": "PASS_MECHANISM_MATERIAL" if m1_gate else "HOLD_EXPAND_EVIDENCE",
        "config": cfg.__dict__,
        "variants": VARIANTS,
        "datasets": datasets,
        "m1_gate": bool(m1_gate),
        "gate_rule": ">=2 datasets with eta^2>=0.30, rank tau<=0.5 for an operational mechanism, and >5% relative degradation vs MCAR",
    }
    write_json(OUT_DIR / "m1_summary.json", summary)
    write_report(summary)
    print(json.dumps({"milestone": "M1", "status": summary["status"], "out": str(OUT_DIR / "m1_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
