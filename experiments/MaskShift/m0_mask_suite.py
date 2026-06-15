"""M0 — Mask generator suite, matched-control audit, and research brief."""

from __future__ import annotations

import json
from pathlib import Path

from .maskshift_core import (
    DEFAULT_DATASETS,
    MECHANISMS,
    ExperimentConfig,
    ensure_dir,
    generate_mask,
    load_dataset,
    mask_stats,
    write_json,
)


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m0_mask_suite")


SOURCES = [
    {
        "id": "S01",
        "title": "Recurrent Neural Networks for Multivariate Time Series with Missing Values",
        "authors": "Che et al.",
        "year": 2018,
        "venue": "Scientific Reports",
        "role": "missingness mask and time-gap baseline",
        "source": "https://www.nature.com/articles/s41598-018-24271-9",
        "verified": True,
    },
    {
        "id": "S02",
        "title": "BRITS: Bidirectional Recurrent Imputation for Time Series",
        "authors": "Cao et al.",
        "year": 2018,
        "venue": "NeurIPS",
        "role": "end-to-end imputation baseline",
        "source": "https://proceedings.neurips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html",
        "verified": True,
    },
    {
        "id": "S03",
        "title": "Self-attention-based Diffusion Model for Time-series Imputation in Partial Blackout Scenarios",
        "authors": "Islam, Tadepalli, and Fern",
        "year": 2025,
        "venue": "AAAI",
        "role": "partial-blackout imputation competitor",
        "source": "https://arxiv.org/pdf/2503.01737",
        "verified": True,
    },
    {
        "id": "S04",
        "title": "S4M: S4 for Multivariate Time Series Forecasting with Missing Values",
        "authors": "Jing et al.",
        "year": 2025,
        "venue": "ICLR",
        "role": "missing-aware forecasting architecture competitor",
        "source": "https://openreview.net/forum?id=BkftcwIVmR",
        "verified": True,
    },
    {
        "id": "S05",
        "title": "Towards Robust Real-World Multivariate Time Series Forecasting",
        "authors": "Jang et al.",
        "year": 2026,
        "venue": "ICLR",
        "role": "dependency/asynchrony/missing-block architecture collision",
        "source": "https://openreview.net/forum?id=r4ZamwBE8P",
        "verified": True,
    },
    {
        "id": "S06",
        "title": "Modeling Information Blackouts in Missing Not-At-Random Time Series Data",
        "authors": "Sunesh, Ma, and Nilol",
        "year": 2026,
        "venue": "arXiv",
        "role": "closest MNAR blackout collision",
        "source": "https://arxiv.org/pdf/2601.01480",
        "verified": True,
    },
    {
        "id": "S07",
        "title": "GraFITi: Graphs for Forecasting Irregularly Sampled Time Series",
        "authors": "Yalavarthi et al.",
        "year": 2024,
        "venue": "AAAI",
        "role": "irregular forecasting baseline",
        "source": "https://ojs.aaai.org/index.php/AAAI/article/view/29560",
        "verified": True,
    },
    {
        "id": "S08",
        "title": "Revisiting Multivariate Time Series Forecasting with Missing Values",
        "authors": "Yang et al.",
        "year": 2025,
        "venue": "arXiv",
        "role": "imputation-then-prediction critique and direct forecasting baseline",
        "source": "https://arxiv.org/abs/2509.23494",
        "verified": True,
    },
    {
        "id": "S09",
        "title": "Robust Prediction under Missingness Shifts",
        "authors": "Rockenschaub et al.",
        "year": 2024,
        "venue": "arXiv",
        "role": "statistical anchor for prediction under missingness shift",
        "source": "https://arxiv.org/abs/2406.16484",
        "verified": True,
    },
]


def write_report(summary: dict) -> None:
    rows = []
    for row in summary["generator_rows"]:
        rows.append(
            "| {dataset} | {mechanism} | {missing_rate:.3f} | {mean_gap:.2f} | {max_gap} | {channel_rate_std:.3f} |".format(
                **row
            )
        )
    report = f"""# MaskShift M0 — Generator Suite and Research Brief

## RQ Brief

**Research question.** Under matched missing rate with audited gap and channel topology statistics, do operational missingness mechanisms induce forecast-risk shifts and model-rank reversals that are not certified by MCAR/block robustness tests?

**Sub-questions.**
1. How much degradation variance is explained by mechanism identity after controlling for missing rate and topology?
2. Do model rankings learned or selected under MCAR remain stable under value-triggered, volatility-triggered, blackout, and retirement mechanisms?
3. Can a lightweight topology/mechanism-typed head recover a meaningful fraction of the degradation without changing the backbone?

**FINER scores.** Feasible 9/10; Interesting 10/10; Novel 8/10; Ethical 10/10; Relevant 10/10.

**Scope.** In scope: forecasting with observed-input missingness, controlled mask generators, public TSF datasets, rank instability, minimal typed/topology correction. Out of scope: claiming a new imputation SOTA, modeling delayed future covariate release, and unrestricted causal identification of the outage process.

## Methodology Blueprint

Quantitative benchmark and theory-driven empirical audit. The design is a matched-factor experiment: fix observed-value tensor, target forecast, missing rate, and approximate topology controls, then vary the missingness mechanism. The primary tests are mixed-effect degradation decomposition, Kendall rank stability, paired loss tests, and typed-head ablations.

## Source Corpus

{chr(10).join(f'- [{s["id"]}] {s["authors"]} ({s["year"]}), {s["title"]}. {s["venue"]}. {s["source"]}' for s in SOURCES)}

## Generator Control Table

| Dataset | Mechanism | Missing rate | Mean gap | Max gap | Channel-rate std |
|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Novelty Boundary

The closest collision is [S06], which models traffic sensor blackouts with MAR/MNAR state-space inference and reports imputation plus short post-blackout forecasts. MaskShift must not claim first MNAR blackout modeling. The defensible contribution is a broader mechanism-shift benchmark showing that matched missing rate is an insufficient control for forecast selection across modern TSF baselines.
"""
    (EXP_DIR / "REPORT.md").write_text(report)


def main() -> None:
    cfg = ExperimentConfig()
    rows = []
    dataset_meta = []
    for dataset in DEFAULT_DATASETS:
        values, meta = load_dataset(dataset)
        split_idx = int(values.shape[0] * 0.7)
        dataset_meta.append(meta)
        for i, mechanism in enumerate(MECHANISMS):
            mask = generate_mask(values, mechanism, cfg.target_rate, cfg.seed + i * 13, split_idx=split_idx)
            stats = mask_stats(mask)
            rows.append({"dataset": dataset, "mechanism": mechanism, **stats})

    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    matched_gate = {}
    for dataset, ds_rows in by_dataset.items():
        rates = [r["missing_rate"] for r in ds_rows]
        matched_gate[dataset] = {
            "max_rate_error": max(abs(r - cfg.target_rate) for r in rates),
            "pass": max(abs(r - cfg.target_rate) for r in rates) <= 0.005,
        }

    summary = {
        "milestone": "M0",
        "status": "PASS" if all(v["pass"] for v in matched_gate.values()) else "WARN",
        "config": cfg.__dict__,
        "datasets": dataset_meta,
        "mechanisms": MECHANISMS,
        "generator_rows": rows,
        "matched_rate_gate": matched_gate,
        "sources": SOURCES,
        "research_question": "Under matched missing rate with audited gap and channel topology statistics, do operational missingness mechanisms induce forecast-risk shifts and model-rank reversals that MCAR/block robustness tests fail to certify?",
        "methodology": "matched-factor benchmark plus typed/topology residual correction audit",
    }
    write_json(OUT_DIR / "m0_summary.json", summary)
    write_report(summary)
    print(json.dumps({"milestone": "M0", "status": summary["status"], "out": str(OUT_DIR / "m0_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
