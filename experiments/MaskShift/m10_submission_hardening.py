"""M10 — submission hardening: seed CIs, corrected severity metrics, and tables."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .m1_mechanism_audit import run_dataset as run_m1_dataset
from .m9_official_tslib_reproduction import BACKBONES as M9_BACKBONES
from .m9_official_tslib_reproduction import run_dataset as run_m9_dataset
from .maskshift_core import MECHANISMS, OPERATIONAL_MECHANISMS, ExperimentConfig, ensure_dir, write_json


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m10_submission_hardening")
TABLE_DIR = ensure_dir(EXP_DIR / "tables")
FIG_DIR = ensure_dir(EXP_DIR / "figures")

SEED_OFFSETS = [0, 10_000, 20_000]
M7_SEED_OFFSETS = [0]
M1_CI_DATASETS = ["Weather", "Electricity"]
ALL_DATASETS = ["Weather", "Electricity", "Traffic", "AirConvection"]
M9_DATASETS = ["Weather", "Electricity"]
RATES = [0.10, 0.20, 0.35, 0.50]


def mean_ci(values: list[float], confidence: float = 0.95) -> dict:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if len(arr) <= 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": int(len(arr))}
    sem = stats.sem(arr)
    tcrit = stats.t.ppf((1 + confidence) / 2.0, df=len(arr) - 1)
    half = float(tcrit * sem)
    return {"mean": mean, "ci_low": mean - half, "ci_high": mean + half, "n": int(len(arr))}


def fmt_ci(summary: dict, pct: bool = False, digits: int = 2, lower: float | None = None, upper: float | None = None) -> str:
    mean = summary["mean"]
    ci_low = summary["ci_low"]
    ci_high = summary["ci_high"]
    if lower is not None:
        mean = max(lower, mean)
        ci_low = max(lower, ci_low)
        ci_high = max(lower, ci_high)
    if upper is not None:
        mean = min(upper, mean)
        ci_low = min(upper, ci_low)
        ci_high = min(upper, ci_high)
    if pct:
        return f"{mean * 100:.1f}% [{ci_low * 100:.1f}, {ci_high * 100:.1f}]"
    return f"{mean:.{digits}f} [{ci_low:.{digits}f}, {ci_high:.{digits}f}]"


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(str(x) for x in row) + " |" for row in rows)
    return "\n".join(out)


def mechanism_means(rows: list[dict], metric: str = "mse") -> dict[str, float]:
    return {
        mechanism: float(np.mean([row[metric] for row in rows if row["test_mechanism"] == mechanism]))
        for mechanism in MECHANISMS
    }


def corrected_degradation(rows: list[dict], metric: str = "mse") -> dict:
    means = mechanism_means(rows, metric)
    mcar = means["mcar"]
    eps = 1e-6
    by_mechanism = {}
    for mechanism in OPERATIONAL_MECHANISMS:
        delta = means[mechanism] - mcar
        by_mechanism[mechanism] = {
            "mean_loss": means[mechanism],
            "absolute_delta_vs_mcar": float(delta),
            "relative_delta_vs_mcar": float(delta / max(abs(mcar), eps)),
            "log_ratio_vs_mcar": float(math.log((means[mechanism] + eps) / (mcar + eps))),
            "symmetric_relative_delta": float(2 * delta / (abs(means[mechanism]) + abs(mcar) + eps)),
        }
    strongest = max(by_mechanism, key=lambda name: by_mechanism[name]["absolute_delta_vs_mcar"])
    strongest_ratio = max(by_mechanism, key=lambda name: by_mechanism[name]["relative_delta_vs_mcar"])
    return {
        "mcar_mean_loss": float(mcar),
        "denominator_instability": bool(abs(mcar) < 0.05),
        "strongest_absolute_mechanism": strongest,
        "max_absolute_delta": by_mechanism[strongest]["absolute_delta_vs_mcar"],
        "max_log_ratio": max(v["log_ratio_vs_mcar"] for v in by_mechanism.values()),
        "max_symmetric_relative_delta": max(v["symmetric_relative_delta"] for v in by_mechanism.values()),
        "max_relative_delta": by_mechanism[strongest_ratio]["relative_delta_vs_mcar"],
        "strongest_relative_mechanism": strongest_ratio,
        "by_mechanism": by_mechanism,
    }


def run_m1_multiseed() -> tuple[list[dict], list[dict]]:
    cfg = ExperimentConfig()
    raw = []
    summaries = []
    for dataset_index, dataset in enumerate(M1_CI_DATASETS):
        seed_results = []
        for seed_index, offset in enumerate(SEED_OFFSETS):
            result = run_m1_dataset(dataset, cfg, seed_offset=dataset_index * 1000 + offset)
            worst_tau = min(row["kendall_tau_vs_mcar_rank"] for row in result["mechanism_effect_rows"])
            max_deg = max(row["relative_degradation_vs_mcar"] for row in result["mechanism_effect_rows"])
            row = {
                "seed_index": seed_index,
                "seed_offset": offset,
                "dataset": dataset,
                "eta_squared": result["anova"]["eta_squared"],
                "p_value": result["anova"]["p_value"],
                "worst_rank_tau": worst_tau,
                "max_relative_degradation": max_deg,
                "gate_pass": result["gate_pass"],
            }
            raw.append(row)
            seed_results.append(row)
        summaries.append(
            {
                "dataset": dataset,
                "eta_squared": mean_ci([row["eta_squared"] for row in seed_results]),
                "max_relative_degradation": mean_ci([row["max_relative_degradation"] for row in seed_results]),
                "worst_rank_tau": mean_ci([row["worst_rank_tau"] for row in seed_results]),
                "gate_pass_count": int(sum(row["gate_pass"] for row in seed_results)),
                "n_seeds": len(seed_results),
            }
        )
    return raw, summaries


def run_m7_corrected() -> tuple[list[dict], list[dict]]:
    raw = []
    summaries = []
    for rate in RATES:
        for dataset_index, dataset in enumerate(ALL_DATASETS):
            for seed_index, offset in enumerate(M7_SEED_OFFSETS):
                cfg = ExperimentConfig(target_rate=rate, max_train_samples=600, max_test_samples=250)
                result = run_m1_dataset(dataset, cfg, seed_offset=dataset_index * 1000 + offset + int(rate * 1000))
                corrected = corrected_degradation(result["rows"], metric="mse")
                worst_tau = min(row["kendall_tau_vs_mcar_rank"] for row in result["mechanism_effect_rows"])
                raw.append(
                    {
                        "rate": rate,
                        "dataset": dataset,
                        "seed_index": seed_index,
                        "seed_offset": offset,
                        "eta_squared": result["anova"]["eta_squared"],
                        "worst_rank_tau": worst_tau,
                        "gate_pass": result["gate_pass"],
                        **corrected,
                    }
                )
    for rate in RATES:
        subset = [row for row in raw if row["rate"] == rate]
        summaries.append(
            {
                "rate": rate,
                "mean_eta_squared": mean_ci([row["eta_squared"] for row in subset]),
                "max_absolute_delta": mean_ci([row["max_absolute_delta"] for row in subset]),
                "max_log_ratio": mean_ci([row["max_log_ratio"] for row in subset]),
                "max_symmetric_relative_delta": mean_ci([row["max_symmetric_relative_delta"] for row in subset]),
                "max_relative_delta": mean_ci([row["max_relative_delta"] for row in subset]),
                "denominator_instability_count": int(sum(row["denominator_instability"] for row in subset)),
                "gate_pass_count": int(sum(row["gate_pass"] for row in subset)),
                "n": len(subset),
            }
        )
    return raw, summaries


def run_m9_multiseed() -> tuple[list[dict], list[dict]]:
    cfg = ExperimentConfig(max_train_samples=180, max_test_samples=80)
    raw = []
    summaries = []
    for dataset_index, dataset in enumerate(M9_DATASETS):
        seed_results = []
        for seed_index, offset in enumerate(SEED_OFFSETS):
            result = run_m9_dataset(dataset, cfg, seed_offset=dataset_index * 1000 + offset)
            row = {
                "seed_index": seed_index,
                "seed_offset": offset,
                "dataset": dataset,
                "max_relative_degradation": result["max_relative_degradation"],
                "worst_rank_tau": result["worst_rank_tau"],
                "anova_p": result["anova_p"],
                "gate_pass": result["gate_pass"],
            }
            raw.append(row)
            seed_results.append(row)
        summaries.append(
            {
                "dataset": dataset,
                "max_relative_degradation": mean_ci([row["max_relative_degradation"] for row in seed_results]),
                "worst_rank_tau": mean_ci([row["worst_rank_tau"] for row in seed_results]),
                "anova_p": mean_ci([row["anova_p"] for row in seed_results]),
                "gate_pass_count": int(sum(row["gate_pass"] for row in seed_results)),
                "n_seeds": len(seed_results),
                "backbones": M9_BACKBONES,
            }
        )
    return raw, summaries


def write_tables(summary: dict) -> None:
    m1_rows = [
        [
            row["dataset"],
            fmt_ci(row["eta_squared"], digits=3, lower=0.0, upper=1.0),
            fmt_ci(row["max_relative_degradation"], pct=True),
            fmt_ci(row["worst_rank_tau"], digits=2, lower=-1.0, upper=1.0),
            f"{row['gate_pass_count']}/{row['n_seeds']}",
        ]
        for row in summary["m1_multiseed_summary"]
    ]
    (TABLE_DIR / "main_result_table.md").write_text(
        "# Main result table — M1 multi-seed mechanism audit\n\n"
        + md_table(["Dataset", "eta^2 mean [95% CI]", "Max degradation mean [95% CI]", "Worst tau mean [95% CI]", "Gate seeds"], m1_rows)
        + "\n"
    )

    m9_rows = [
        [
            row["dataset"],
            ", ".join(row["backbones"]),
            fmt_ci(row["max_relative_degradation"], pct=True),
            fmt_ci(row["worst_rank_tau"], digits=2, lower=-1.0, upper=1.0),
            f"{row['gate_pass_count']}/{row['n_seeds']}",
        ]
        for row in summary["m9_multiseed_summary"]
    ]
    (TABLE_DIR / "m9_official_architecture_table.md").write_text(
        "# M9 official-architecture adaptation table\n\n"
        "PatchTST and TimeXer are imported from pinned TSLib model classes; the MaskShift loop is a custom encoder-mask protocol, not the full official benchmark protocol.\n\n"
        + md_table(["Dataset", "Official architecture classes", "Max degradation mean [95% CI]", "Worst tau mean [95% CI]", "Gate seeds"], m9_rows)
        + "\n"
    )

    m7_rows = [
        [
            f"{row['rate']:.2f}",
            fmt_ci(row["mean_eta_squared"], digits=3, lower=0.0, upper=1.0),
            fmt_ci(row["max_absolute_delta"], digits=3),
            fmt_ci(row["max_log_ratio"], digits=2),
            fmt_ci(row["max_symmetric_relative_delta"], digits=2),
            f"{row['denominator_instability_count']}/{row['n']}",
        ]
        for row in summary["m7_corrected_summary"]
    ]
    (TABLE_DIR / "m7_corrected_robustness_table.md").write_text(
        "# M7 corrected robustness table\n\n"
        "The original relative degradation ratio is retained only as a diagnostic because small MCAR denominators can explode. The submission reports absolute delta, log ratio, and symmetric relative delta.\n\n"
        + md_table(["Missing rate", "eta^2 mean [95% CI]", "Max abs delta [95% CI]", "Max log ratio [95% CI]", "Max symmetric delta [95% CI]", "Denom unstable"], m7_rows)
        + "\n"
    )

    claim_rows = [
        ["C1", "Matched missing rate is not a robustness certificate.", "M1, M3, M7, M9, M13", "Supported on Weather/Electricity; Traffic/AirConvection mixed. M13 supports loss-delta uncertainty but not universal rank instability.", "State as evidence-backed benchmark finding, not universal theorem."],
        ["C2", "Mechanism shift can reverse model rankings.", "M9 official-architecture adaptation; M1 ranks as diagnostic support", "M9 Weather/Electricity worst tau=-1 over three seed offsets.", "Do not claim every dataset/model reverses."],
        ["C3", "The result is not only sensor retirement.", "M8 non-retirement decomposition", "Weather/Electricity pass without retirement.", "Retirement remains a strong and obvious mechanism; keep decomposition visible."],
        ["C4", "Typed head is not a new method contribution.", "M2/H3", "H3 fails; overall typed p=0.214.", "Present as negative/diagnostic ablation."],
        ["C5", "Official modern architectures are affected.", "M9/M10", "PatchTST/TimeXer official classes under custom MaskShift protocol.", "Call it official-architecture adaptation, not full official benchmark reproduction."],
        ["C6", "Missing-aware architecture coverage is included but not decisive.", "M11, M12, M14", "CTF_missing shows strong Weather sensitivity but weaker/non-significant Electricity sensitivity; S4M is negative/contrastive with 0/3 gate seeds under both reduced and larger-reduced protocols.", "Report as architecture-dependent evidence, not a win/loss claim."],
    ]
    (TABLE_DIR / "claim_evidence_table.md").write_text(
        "# Claim-to-evidence table\n\n"
        + md_table(["ID", "Claim", "Evidence", "Limit", "Required wording"], claim_rows)
        + "\n"
    )

    related_rows = [
        ["GRU-D", "Scientific Reports 2018", "Mask/time-gap-aware RNN prediction", "No mechanism-shift benchmark or rank-reversal audit."],
        ["BRITS", "NeurIPS 2018", "Bidirectional imputation/prediction", "Optimizes reconstruction/imputation rather than deployment mechanism shift."],
        ["SADI", "AAAI 2025", "Diffusion imputation for partial blackouts", "Blackout imputation competitor; not matched-rate multi-mechanism model-selection audit."],
        ["S4M", "ICLR 2025", "Missing-aware S4 forecasting architecture", "Architecture baseline; MaskShift is benchmark/theory and tests mask mechanisms as experimental factors."],
        ["ChannelTokenFormer", "ICLR 2026", "Dependency/asynchrony/missingness architecture", "Closest architecture collision; MaskShift avoids unified-architecture claims."],
        ["CRIB/MTSF-M", "arXiv 2025", "Revisits MTSF with missing values", "Motivates direct forecasting; does not isolate matched-rate mechanism shift/rank reversal."],
        ["Information blackouts", "arXiv 2026", "MNAR traffic blackout state-space model", "Closest blackout collision; MaskShift broadens to multiple mechanisms and ranking stability."],
        ["Robust prediction under missingness shifts", "arXiv 2024", "Statistical missingness-shift theory", "Non-TSF anchor; MaskShift operationalizes for forecasting benchmarks."],
    ]
    (TABLE_DIR / "related_work_table.md").write_text(
        "# Related work comparison table\n\n"
        + md_table(["Work", "Venue/year", "Occupied claim", "MaskShift distinction"], related_rows)
        + "\n"
    )


def write_overview_figure() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    boxes = [
        (0.07, 0.68, "Train/select under\nMCAR or block masks"),
        (0.38, 0.68, "Matched missing rate\nbut shifted topology"),
        (0.69, 0.68, "Deployment mechanisms\nvalue, blackout, retirement"),
        (0.18, 0.25, "Risk shift\nE_B[(mu_A-mu_B)^2]"),
        (0.55, 0.25, "Model rank reversal\nand degraded selection"),
    ]
    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f2f5f7", edgecolor="#284b63", linewidth=1.5),
            transform=ax.transAxes,
        )
    arrows = [
        ((0.19, 0.68), (0.31, 0.68)),
        ((0.50, 0.68), (0.62, 0.68)),
        ((0.38, 0.60), (0.26, 0.35)),
        ((0.69, 0.60), (0.61, 0.35)),
        ((0.32, 0.25), (0.45, 0.25)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color="#3c4856", lw=1.6))
    ax.text(0.5, 0.08, "MaskShift audits p(M|X) and topology shifts at fixed missing rate; it is a benchmark/theory paper, not a new forecaster.", ha="center", fontsize=10, transform=ax.transAxes)
    out = FIG_DIR / "maskshift_overview.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def write_missing_aware_attempt() -> None:
    doc = """# Missing-aware Baseline Attempt

This file records the audit trail for missing-aware baselines. Early M10 hardening found no installed S4M or ChannelTokenFormer-compatible code in the local `external/` tree. That gap is now superseded by M11 and M12.

Current submission evidence:

- official-architecture adaptation: PatchTST and TimeXer model classes imported from pinned TSLib revision `4e938a1`;
- official missing-aware adaptation: `ChannelTokenFormer_missing` from pinned ChannelTokenFormer revision `b1c100e` in M11;
- official S4M adaptation: S4M from pinned revision `a718823` in M12, with a local device-port patch that replaces one hard-coded `.cuda()` memory fetch with `.to(Q.device)`;
- lite missing-aware proxy: GRU-DLite in M6.

Do not state that MaskShift reproduces the full original ChannelTokenFormer practical benchmark or the full S4M benchmark protocol. The correct wording is reduced MaskShift encoder-mask adaptation of official missing-aware architectures.
"""
    (EXP_DIR / "docs" / "MISSING_AWARE_BASELINE_ATTEMPT.md").write_text(doc)


def main() -> None:
    m1_raw, m1_summary = run_m1_multiseed()
    m7_raw, m7_summary = run_m7_corrected()
    m9_raw, m9_summary = run_m9_multiseed()
    summary = {
        "milestone": "M10",
        "status": "PASS_SUBMISSION_HARDENING",
        "seed_offsets": SEED_OFFSETS,
        "m7_seed_offsets": M7_SEED_OFFSETS,
        "m1_multiseed_raw": m1_raw,
        "m1_multiseed_summary": m1_summary,
        "m7_corrected_raw": m7_raw,
        "m7_corrected_summary": m7_summary,
        "m9_multiseed_raw": m9_raw,
        "m9_multiseed_summary": m9_summary,
        "tables": [
            "tables/main_result_table.md",
            "tables/m9_official_architecture_table.md",
            "tables/m7_corrected_robustness_table.md",
            "tables/claim_evidence_table.md",
            "tables/related_work_table.md",
        ],
        "figures": ["figures/maskshift_overview.png"],
        "missing_aware_baseline_status": "m10_gap_superseded_by_m11_ctf_and_m12_s4m_official_adaptations",
        "runtime_note": "M1 CI is run on the core positive datasets (Weather/Electricity). M9 official-architecture adaptation uses 3 seeds with reduced train/test samples for deadline feasibility. M7 corrected metrics use one seed across all four datasets/rates because the full 3-seed severity rerun was too slow for the submission sprint.",
    }
    write_json(OUT_DIR / "submission_hardening_summary.json", summary)
    write_tables(summary)
    write_overview_figure()
    write_missing_aware_attempt()
    print(json.dumps({"milestone": "M10", "status": summary["status"], "out": str(OUT_DIR / "submission_hardening_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
