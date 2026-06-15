"""M13 — hierarchy-aware bootstrap for MaskShift aggregate claims.

M1 already includes per-window ANOVA for the MCAR-selected lightweight variant,
while M10 adds seed-level descriptive CIs. This milestone adds a nonparametric
hierarchical bootstrap over lightweight variants and test windows so the paper
does not rely only on aggregate model means.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .m1_mechanism_audit import VARIANTS
from .maskshift_core import (
    MECHANISMS,
    OPERATIONAL_MECHANISMS,
    ExperimentConfig,
    build_supervised,
    ensure_dir,
    generate_mask,
    kendall_tau_between,
    make_dataset_splits,
    make_model,
    write_json,
)


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m13_hierarchical_bootstrap")
TABLE_DIR = ensure_dir(EXP_DIR / "tables")
DATASETS = ["Weather", "Electricity", "Traffic", "AirConvection"]
N_BOOT = 400


SETTINGS = {
    "zero": {"fill": "zero", "include_mask": False, "include_topology": False, "include_mechanism": False},
    "ffill": {"fill": "ffill", "include_mask": False, "include_topology": False, "include_mechanism": False},
    "mask": {"fill": "zero", "include_mask": True, "include_topology": False, "include_mechanism": False},
    "topology": {"fill": "zero", "include_mask": True, "include_topology": True, "include_mechanism": False},
}


def ci(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "n": int(values.size),
    }


def fmt_ci(summary: dict, pct: bool = False, digits: int = 3) -> str:
    if pct:
        return f"{summary['mean'] * 100:.1f}% [{summary['ci_low'] * 100:.1f}, {summary['ci_high'] * 100:.1f}]"
    return f"{summary['mean']:.{digits}f} [{summary['ci_low']:.{digits}f}, {summary['ci_high']:.{digits}f}]"


def collect_losses(dataset: str, cfg: ExperimentConfig, seed_offset: int) -> dict:
    ds = make_dataset_splits(dataset, cfg)
    values = ds["values"]
    split_idx = ds["split_idx"]
    train_origins = ds["train_origins"]
    test_origins = ds["test_origins"]
    train_mask = generate_mask(values, "mcar", cfg.target_rate, cfg.seed + seed_offset, split_idx=split_idx)

    fitted = {}
    for variant in VARIANTS:
        x_train, y_train, _ = build_supervised(
            values,
            train_mask,
            train_origins,
            cfg.lookback,
            cfg.horizon,
            mechanism="mcar",
            **SETTINGS[variant],
        )
        model = make_model()
        model.fit(x_train, y_train)
        fitted[variant] = model

    losses: dict[str, dict[str, list[float]]] = {mechanism: {} for mechanism in MECHANISMS}
    means: dict[str, dict[str, float]] = {mechanism: {} for mechanism in MECHANISMS}
    for j, mechanism in enumerate(MECHANISMS):
        test_mask = generate_mask(values, mechanism, cfg.target_rate, cfg.seed + seed_offset + 4200 + j * 17, split_idx=split_idx)
        for variant in VARIANTS:
            x_test, y_test, _ = build_supervised(
                values,
                test_mask,
                test_origins,
                cfg.lookback,
                cfg.horizon,
                mechanism=mechanism,
                **SETTINGS[variant],
            )
            pred = fitted[variant].predict(x_test)
            per_window = (y_test - pred) ** 2
            losses[mechanism][variant] = per_window.astype(float).tolist()
            means[mechanism][variant] = float(np.mean(per_window))
    return {"dataset": dataset, "meta": ds["meta"], "losses": losses, "means": means}


def bootstrap_dataset(collected: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    losses = collected["losses"]
    metrics = {
        "eta_squared": [],
        "max_absolute_delta": [],
        "max_relative_delta": [],
        "worst_rank_tau": [],
    }
    variants = list(VARIANTS)
    for _ in range(N_BOOT):
        sampled_variants = list(rng.choice(variants, size=len(variants), replace=True))
        means = {}
        by_mech_variant = {}
        for mechanism in MECHANISMS:
            variant_means = []
            by_mech_variant[mechanism] = {}
            for variant in sampled_variants:
                arr = np.asarray(losses[mechanism][variant], dtype=float)
                sample = rng.choice(arr, size=arr.size, replace=True)
                mean = float(np.mean(sample))
                variant_means.append(mean)
                by_mech_variant[mechanism].setdefault(variant, []).append(mean)
            means[mechanism] = float(np.mean(variant_means))
        all_variant_means = np.asarray(
            [
                float(np.mean(vals))
                for mechanism in MECHANISMS
                for vals in by_mech_variant[mechanism].values()
            ],
            dtype=float,
        )
        mech_mean_values = np.asarray([means[m] for m in MECHANISMS], dtype=float)
        grand = float(np.mean(all_variant_means))
        total = float(np.sum((all_variant_means - grand) ** 2))
        between = float(np.sum((mech_mean_values - grand) ** 2))
        eta2 = min(1.0, max(0.0, between / total)) if total > 0 else 0.0
        mcar = means["mcar"]
        op_deltas = {mechanism: means[mechanism] - mcar for mechanism in OPERATIONAL_MECHANISMS}
        max_abs = max(op_deltas.values())
        max_rel = max(delta / max(abs(mcar), 1e-9) for delta in op_deltas.values())
        ranks = {}
        for mechanism in MECHANISMS:
            variant_scores = {
                variant: float(np.mean(by_mech_variant[mechanism].get(variant, [np.inf])))
                for variant in variants
            }
            ranks[mechanism] = [name for name, _ in sorted(variant_scores.items(), key=lambda item: item[1])]
        worst_tau = min(kendall_tau_between(ranks["mcar"], ranks[mechanism]) for mechanism in OPERATIONAL_MECHANISMS)
        metrics["eta_squared"].append(eta2)
        metrics["max_absolute_delta"].append(max_abs)
        metrics["max_relative_delta"].append(max_rel)
        metrics["worst_rank_tau"].append(worst_tau)
    summary = {name: ci(np.asarray(values, dtype=float)) for name, values in metrics.items()}
    summary["positive_delta_probability"] = float(np.mean(np.asarray(metrics["max_absolute_delta"]) > 0))
    summary["rank_instability_probability"] = float(np.mean(np.asarray(metrics["worst_rank_tau"]) <= 0.5))
    summary["loss_shift_supported"] = bool(summary["max_absolute_delta"]["ci_low"] > 0)
    summary["rank_instability_supported"] = bool(summary["rank_instability_probability"] >= 0.80)
    summary["primary_gate"] = bool(
        summary["loss_shift_supported"]
        and summary["rank_instability_supported"]
    )
    return summary


def write_table(summary: dict) -> None:
    lines = [
        "# M13 hierarchical bootstrap table",
        "",
        "Bootstrap resamples lightweight variants and test windows. It is a hierarchy-aware uncertainty check for M1 aggregate claims, not a new model result.",
        "",
        "| Dataset | eta^2 [95% CI] | Max abs delta [95% CI] | P(delta>0) | Loss-shift evidence | Worst tau [95% CI] | P(tau<=0.5) | Rank evidence |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for row in summary["dataset_summaries"]:
        lines.append(
            "| {dataset} | {eta} | {abs_delta} | {p_delta:.2f} | {loss_gate} | {tau} | {p_tau:.2f} | {rank_gate} |".format(
                dataset=row["dataset"],
                eta=fmt_ci(row["eta_squared"], digits=3),
                abs_delta=fmt_ci(row["max_absolute_delta"], digits=3),
                tau=fmt_ci(row["worst_rank_tau"], digits=2),
                p_delta=row["positive_delta_probability"],
                p_tau=row["rank_instability_probability"],
                loss_gate="SUPPORTED" if row["loss_shift_supported"] else "MIXED",
                rank_gate="SUPPORTED" if row["rank_instability_supported"] else "NOT_DECISIVE",
            )
        )
    (TABLE_DIR / "m13_hierarchical_bootstrap_table.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    cfg = ExperimentConfig(max_train_samples=650, max_test_samples=300)
    collected = []
    dataset_summaries = []
    for dataset_index, dataset in enumerate(DATASETS):
        data = collect_losses(dataset, cfg, seed_offset=dataset_index * 1000)
        collected.append(data)
        boot = bootstrap_dataset(data, seed=cfg.seed + dataset_index * 1000 + 13)
        boot["dataset"] = dataset
        dataset_summaries.append(boot)
    protocol_complete = len(dataset_summaries) == len(DATASETS)
    primary_gate_count = sum(row["primary_gate"] for row in dataset_summaries if row["dataset"] in {"Weather", "Electricity"})
    summary = {
        "milestone": "M13",
        "status": "PASS_HIERARCHICAL_BOOTSTRAP" if protocol_complete else "HOLD_HIERARCHICAL_BOOTSTRAP",
        "config": {**cfg.__dict__, "n_bootstrap": N_BOOT, "resample_levels": ["variant", "test_window"]},
        "datasets": collected,
        "dataset_summaries": dataset_summaries,
        "m13_gate": bool(protocol_complete),
        "primary_effect_gate_count": int(primary_gate_count),
        "protocol_note": "Nonparametric bootstrap over lightweight variants and test windows; strengthens uncertainty reporting for the M1 mechanism-shift claim.",
    }
    write_json(OUT_DIR / "hierarchical_bootstrap_summary.json", summary)
    write_table(summary)
    print(json.dumps({"milestone": "M13", "status": summary["status"], "primary_effect_gate_count": primary_gate_count}, indent=2))


if __name__ == "__main__":
    main()
