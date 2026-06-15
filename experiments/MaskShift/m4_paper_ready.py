"""M4 — Paper-ready artifact packager for MaskShift."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from .maskshift_core import DEFAULT_DATASETS, ensure_dir, write_json


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m4_paper_ready")
PAPER_DIR = ensure_dir(EXP_DIR / "paper")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def make_figures(m1: dict, m2: dict) -> list[str]:
    fig_dir = ensure_dir(EXP_DIR / "figures")
    datasets = [ds["dataset"] for ds in m1["datasets"]]
    eta = [ds["anova"]["eta_squared"] for ds in m1["datasets"]]
    worst_tau = [min(row["kendall_tau_vs_mcar_rank"] for row in ds["mechanism_effect_rows"]) for ds in m1["datasets"]]
    fig, ax1 = plt.subplots(figsize=(7, 3.2))
    x = range(len(datasets))
    ax1.bar([i - 0.18 for i in x], eta, width=0.36, label="Mechanism eta^2", color="#3b6ea8")
    ax1.axhline(0.30, color="#3b6ea8", linestyle="--", linewidth=1)
    ax1.set_ylabel("eta^2")
    ax1.set_ylim(0, max(1.0, max(eta) * 1.2))
    ax2 = ax1.twinx()
    ax2.bar([i + 0.18 for i in x], worst_tau, width=0.36, label="Worst Kendall tau", color="#c4512c")
    ax2.axhline(0.50, color="#c4512c", linestyle="--", linewidth=1)
    ax2.set_ylabel("rank tau")
    ax2.set_ylim(-1, 1)
    ax1.set_xticks(list(x), datasets, rotation=20, ha="right")
    ax1.set_title("Mechanism effect size and rank instability")
    fig.tight_layout()
    f1 = fig_dir / "m1_mechanism_rank.png"
    fig.savefig(f1, dpi=180)
    plt.close(fig)

    datasets2 = [ds["dataset"] for ds in m2["datasets"]]
    reductions = [ds["typed_vs_topology_reduction"] for ds in m2["datasets"]]
    clean = [ds["clean_mcar_cost"] for ds in m2["datasets"]]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(datasets2, reductions, label="Operational AUC reduction", color="#2f7f5f")
    ax.plot(datasets2, clean, marker="o", color="#8f3f71", label="Clean MCAR cost")
    ax.axhline(0.20, color="#2f7f5f", linestyle="--", linewidth=1)
    ax.axhline(0.02, color="#8f3f71", linestyle=":", linewidth=1)
    ax.set_ylabel("fraction")
    ax.set_title("Typed/topology diagnostic")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    f2 = fig_dir / "m2_typed_correction.png"
    fig.savefig(f2, dpi=180)
    plt.close(fig)
    return [str(f1), str(f2)]


BIB = r"""
@article{che2018grud,
  title={Recurrent neural networks for multivariate time series with missing values},
  author={Che, Zhengping and Purushotham, Sanjay and Cho, Kyunghyun and Sontag, David and Liu, Yan},
  journal={Scientific Reports},
  volume={8},
  number={1},
  pages={6085},
  year={2018}
}

@inproceedings{cao2018brits,
  title={BRITS: Bidirectional recurrent imputation for time series},
  author={Cao, Wei and Wang, Dong and Li, Jian and Zhou, Hao and Li, Lei and Li, Yitan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2018}
}

@inproceedings{islam2025sadi,
  title={Self-attention-based diffusion model for time-series imputation in partial blackout scenarios},
  author={Islam, Mohammad Rafid Ul and Tadepalli, Prasad and Fern, Alan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}

@inproceedings{jing2025s4m,
  title={S4M: S4 for multivariate time series forecasting with missing values},
  author={Jing, Peng and Yang, Meiqi and Zhang, Qiong and Li, Xiaoxiao},
  booktitle={International Conference on Learning Representations},
  year={2025}
}

@inproceedings{jang2026channeltokenformer,
  title={Towards robust real-world multivariate time series forecasting: A unified framework for dependency, asynchrony, and missingness},
  author={Jang, Jinkwan and Park, Hyungjin and Choi, Jinmyeong and Kim, Taesup},
  booktitle={International Conference on Learning Representations},
  year={2026}
}

@article{sunesh2026blackouts,
  title={Modeling information blackouts in missing not-at-random time series data},
  author={Sunesh, Aman and Ma, Allan and Nilol, Siddarth},
  journal={arXiv preprint arXiv:2601.01480},
  year={2026}
}

@inproceedings{yalavarthi2024grafiti,
  title={GraFITi: Graphs for forecasting irregularly sampled time series},
  author={Yalavarthi, Vikram Kumar and others},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}

@inproceedings{zeng2023dlinear,
  title={Are transformers effective for time series forecasting?},
  author={Zeng, Ailing and Chen, Muxi and Zhang, Lei and Xu, Qiang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023}
}

@inproceedings{nie2023patchtst,
  title={A time series is worth 64 words: Long-term forecasting with transformers},
  author={Nie, Yuqi and Nguyen, Nam H. and Sinthong, Phanwadee and Kalagnanam, Jayant},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@inproceedings{diebold1995dm,
  title={Comparing predictive accuracy},
  author={Diebold, Francis X. and Mariano, Roberto S.},
  journal={Journal of Business and Economic Statistics},
  year={1995}
}

@article{yang2025crib,
  title={Revisiting multivariate time series forecasting with missing values},
  author={Yang, Rui and others},
  journal={arXiv preprint arXiv:2509.23494},
  year={2025}
}

@article{rockenschaub2024missingness,
  title={Robust prediction under missingness shifts},
  author={Rockenschaub, Patrick and Xian, Zhicong and Zamanian, Alireza and Piperno, Marta and Ciora, Octavia-Andreea and Pachl, Elisabeth and Ahmidi, Narges},
  journal={arXiv preprint arXiv:2406.16484},
  year={2024}
}
"""


def write_paper(
    m0: dict,
    m1: dict,
    m2: dict,
    m3: dict,
    m6: dict | None,
    m7: dict | None,
    m8: dict | None,
    m9: dict | None,
    figs: list[str],
    summary: dict,
) -> None:
    m1_rows = [
        [
            ds["dataset"],
            f"{ds['anova']['eta_squared']:.3f}",
            f"{ds['anova']['p_value']:.2g}",
            f"{min(row['kendall_tau_vs_mcar_rank'] for row in ds['mechanism_effect_rows']):.3f}",
            "PASS" if ds["gate_pass"] else "FAIL",
        ]
        for ds in m1["datasets"]
    ]
    m2_rows = [
        [
            ds["dataset"],
            f"{ds['operational_degradation_auc']['topology']:.4f}",
            f"{ds['operational_degradation_auc']['typed']:.4f}",
            f"{ds['typed_vs_topology_reduction']:.1%}",
            f"{ds['clean_mcar_cost']:.1%}",
            "PASS" if ds["gate_pass"] else "FAIL",
        ]
        for ds in m2["datasets"]
    ]
    m6_rows = []
    if m6:
        m6_rows = [
            [
                ds["dataset"],
                f"{ds['max_relative_degradation']:.1%}",
                f"{ds['anova_p']:.3g}",
                f"{min(ds['rank_taus'].values()):.3f}",
                "PASS" if ds["gate_pass"] else "FAIL",
            ]
            for ds in m6["datasets"]
        ]
    m7_rows = []
    if m7:
        m7_rows = [
            [
                rate,
                vals["n_gate_pass"],
                f"{vals['mean_eta']:.3f}",
                f"{vals['min_rank_tau']:.3f}",
                f"{vals['max_relative_degradation']:.1%}",
            ]
            for rate, vals in m7["by_rate"].items()
        ]
    m8_rows = []
    if m8:
        m8_rows = [
            [
                row["dataset"],
                row["strongest_non_retirement_mechanism"],
                f"{row['max_non_retirement_degradation']:.1%}",
                f"{row['worst_non_retirement_tau']:.3f}",
                "PASS" if row["non_retirement_gate"] else "FAIL",
            ]
            for row in m8["dataset_rows"]
        ]
    m9_rows = []
    if m9:
        m9_rows = [
            [
                ds["dataset"],
                f"{ds['max_relative_degradation']:.1%}",
                f"{ds['worst_rank_tau']:.3f}",
                f"{ds['anova_p']:.3g}",
                "PASS" if ds["gate_pass"] else "FAIL",
            ]
            for ds in m9["datasets"]
        ]
    paper = f"""# MaskShift: Forecasting Under Missingness-Mechanism Shift

## Abstract

Missing-value forecasting benchmarks usually vary how many observations are removed, but deployment often changes why observations disappear: random deletion becomes congestion-linked sensor blackout, value-triggered dropout, volatility-linked telemetry loss, or sensor retirement. We formalize this as missingness-mechanism shift: at matched missing rate, the mask becomes a shifted covariate rather than a neutral nuisance. MaskShift contributes (i) a matched-rate mask-generator suite, (ii) an audit protocol for forecast-risk and rank instability under mechanism shift, and (iii) a diagnostic topology/mechanism-typed ablation that tests whether lightweight mask summaries can reduce the shift. On the current evidence package, the claim scope is: **{summary['paper_route']}**. The strongest empirical result is H1/H2 audit evidence; H3 typed-head sufficiency fails as a universal method claim and is reported as a partial/negative result.

## 1. Introduction

Real forecasting systems rarely fail under independent MCAR deletion. Traffic detectors fail during congestion, weather stations disappear during storms, telemetry agents stop reporting under overload, and retired sensors never return. These outages do not merely reduce sample size. They change the semantic meaning of the observation mask.

This paper asks whether MCAR or uniform-block robustness certifies deployment-time missingness. The answer should be no if the mask-generating law is a shifted, non-ignorable covariate. The practical consequence is severe: model selection under artificial masks can choose the wrong forecaster for operational outages.

## 2. Research Question and Contributions

**RQ.** {m0['research_question']}

Contributions:
1. A mechanism-shift formulation for forecasting with missing inputs: train under mechanism A, deploy under mechanism B, with matched missing rate and shared observed-value tensor.
2. A public mask-generator suite covering MCAR, block, value-triggered, volatility-triggered, blackout, and retirement mechanisms.
3. A benchmark audit measuring degradation variance, rank instability, and typed/topology diagnostics under controlled masks.
4. A diagnostic typed-head study showing where lightweight topology/mechanism summaries help and where they fail, so the paper does not overclaim a new architecture.

## 3. Related Work

GRU-D showed that masks and time gaps can be predictive inputs, but it was designed for medical RNN prediction rather than deployment mechanism shift. BRITS and SADI optimize imputation and reconstruction; SADI is particularly relevant for partial blackout imputation. S4M, CRIB/MTSF-M, and ChannelTokenFormer are stronger architecture-level baselines that handle missingness directly. Robust prediction under missingness shifts gives the closest statistical framing outside TSF. The closest task collision is the 2026 information-blackouts preprint, which models MNAR traffic blackouts with state-space inference. MaskShift differs by making missingness mechanism an experimental factor across mechanisms and forecaster rankings rather than focusing on a single blackout model.

## 4. Theory Sketch

Let M denote the input mask and X_obs the observed input values. A forecaster trained under mechanism A minimizes risk under p_A(Y | X_obs, M). Deployment under mechanism B changes p(M | X) and therefore changes the induced conditional unless the mask is ignorable or the model conditions on a sufficient statistic S(M) that captures the mechanism-relevant topology. The leading excess-risk term is controlled by the divergence between the mechanism-conditioned mask laws p_A(M | X_obs, S) and p_B(M | X_obs, S). This yields two falsifiable predictions: matched missing rate is insufficient, and typed topology statistics may reduce the shift term only when the mechanism is identifiable from observed mask topology.

**Proposition 1 (mechanism-shift excess risk).** Let Z=(X_obs, M) and let mu_A(Z)=E_A[Y | Z] and mu_B(Z)=E_B[Y | Z] be the squared-loss Bayes predictors under two missingness mechanisms A and B. Evaluating the predictor learned for A under the deployment law B gives

`R_B(mu_A) - R_B(mu_B) = E_B[(mu_A(Z) - mu_B(Z))^2]`.

Therefore matched missing rate is a sufficient robustness certificate only when the mechanism shift leaves the conditional Bayes predictor invariant on the deployment support. If a mask statistic S makes `E_A[Y | X_obs, S] = E_B[Y | X_obs, S]`, then conditioning on S removes this mechanism-shift term; if not, topology or mechanism labels can help only partially. The empirical H1/H2 tests estimate whether this term is material, while H3 tests one deliberately small candidate statistic.

## 5. Benchmark Protocol

Datasets: {', '.join(DEFAULT_DATASETS.keys())}. The current run uses lookback {m0['config']['lookback']}, horizon {m0['config']['horizon']}, target missing rate {m0['config']['target_rate']}, and chronological train/test splits. All masks corrupt only pre-origin inputs; forecast targets are never masked.

Mechanisms: {', '.join(m0['mechanisms'])}. Metrics: MSE, MAE, sMAPE, mechanism eta^2, Kendall rank tau versus MCAR ranking, typed-head degradation AUC, and BH-FDR adjusted claim-family tests.

## 6. Results

### M1 Mechanism Audit

{table(['Dataset', 'eta^2', 'p', 'worst tau', 'Gate'], m1_rows)}

### M2 Typed-Head Diagnostic

{table(['Dataset', 'AUC topology', 'AUC typed', 'Reduction', 'Clean cost', 'Gate'], m2_rows)}

### M3 Claim-Family Tests

- H1 mechanism over rate: {m3['claim_tests']['H1_mechanism_over_rate']}
- H2 rank instability: {m3['claim_tests']['H2_rank_instability']}
- H3 typed minimal correction: {m3['claim_tests']['H3_typed_minimal_correction']}
- Overall typed improvement paired test p: {m3['claim_tests']['overall_typed_improvement_p']:.3g}

### M6 Deep-Backbone Lite Sweep

{table(['Dataset', 'max degradation', 'ANOVA p', 'worst tau', 'Gate'], m6_rows) if m6_rows else 'M6 not yet run.'}

Scope note: the M6 sweep uses DLinearLite, PatchTSTLite, and GRU-DLite as fast neural proxies. It supports the mechanism-shift thesis but does not replace an official PatchTST/TimeXer/S4M reproduction for final submission.

### M7 Severity Curves

{table(['Missing rate', '# dataset gates', 'mean eta^2', 'min tau', 'max degradation'], m7_rows) if m7_rows else 'M7 not yet run.'}

### M8 Non-Retirement Decomposition

{table(['Dataset', 'Strongest non-retirement mechanism', 'Max non-ret degradation', 'Worst non-ret tau', 'Gate'], m8_rows) if m8_rows else 'M8 not yet run.'}

### M9 Official TSLib Architecture Reproduction

{table(['Dataset', 'Max degradation', 'Worst tau', 'ANOVA p', 'Gate'], m9_rows) if m9_rows else 'M9 not yet run.'}

M9 imports official PatchTST and TimeXer model classes from the pinned TSLib checkout (`{m9['tslib_revision'] if m9 else 'not_run'}`) while preserving MaskShift's encoder-only masking protocol.

![Mechanism effect and rank instability]({Path(figs[0]).relative_to(EXP_DIR)})

![Typed-head diagnostic]({Path(figs[1]).relative_to(EXP_DIR)})

## 7. Limitations

The current package is an honest benchmark/theory artifact with official TSLib PatchTST/TimeXer architecture reproduction. A broader camera-ready package should still add more seeds, confidence intervals, and one missing-specific architecture such as S4M or ChannelTokenFormer-compatible settings. Mechanism labels may be unavailable in deployment; therefore topology-only variants, randomized-label ablations, and the failed universal typed-head claim must remain visible in the main paper.

## 8. Submission Readiness

Route: **{summary['paper_route']}**.

Blocking items are reported by `m5_main_track_audit.py`. A strong top-conference submission requires at minimum: H1/H2 FDR pass on multiple datasets, official modern baselines, severity curves, non-retirement mechanism decomposition, clearly scoped benchmark-only contribution if H3 remains failed, and release-quality mask generator documentation.

## References

See `paper/references.bib`.
"""
    (EXP_DIR / "PAPER.md").write_text(paper)

    tex = rf"""\documentclass[10pt]{{article}}
\usepackage{{times}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{hyperref}}
\usepackage{{amsmath,amssymb}}
\usepackage[margin=1in]{{geometry}}
\title{{MaskShift: Forecasting Under Missingness-Mechanism Shift}}
\author{{Anonymous Authors}}
\date{{}}
\begin{{document}}
\maketitle
\begin{{abstract}}
Missing-value forecasting benchmarks usually vary how many observations are removed, but deployment often changes why observations disappear: random deletion becomes congestion-linked sensor blackout, value-triggered dropout, volatility-linked telemetry loss, or sensor retirement. We formalize this as missingness-mechanism shift and present a matched-rate benchmark plus a diagnostic topology/mechanism-typed ablation. Current evidence route: {summary['paper_route']}.
\end{{abstract}}
\section{{Introduction}}
Real forecasting systems rarely fail under independent MCAR deletion. Traffic detectors fail during congestion, weather stations disappear during storms, telemetry agents stop reporting under overload, and retired sensors never return. These outages change the semantic meaning of the observation mask.
\section{{Theory}}
Let $Z=(X_{{obs}}, M)$ and let $\mu_A(Z)=\mathbb{{E}}_A[Y\mid Z]$ and $\mu_B(Z)=\mathbb{{E}}_B[Y\mid Z]$ be squared-loss Bayes predictors under mechanisms $A$ and $B$. Then $R_B(\mu_A)-R_B(\mu_B)=\mathbb{{E}}_B[(\mu_A(Z)-\mu_B(Z))^2]$. Matched missing rate certifies robustness only when the mechanism shift leaves this conditional predictor invariant on the deployment support.
\section{{Method}}
We fix the observed-value tensor and vary only the mask-generating mechanism. Masks are generated before the forecast origin and targets are never masked. The mechanisms are MCAR, block, value-triggered, volatility-triggered, blackout, and retirement.
\section{{Results}}
\begin{{figure}}[t]
\centering
\includegraphics[width=.95\linewidth]{{../figures/m1_mechanism_rank.png}}
\caption{{Mechanism effect size and rank instability under matched-rate masks.}}
\end{{figure}}
\begin{{figure}}[t]
\centering
\includegraphics[width=.95\linewidth]{{../figures/m2_typed_correction.png}}
\caption{{Typed/topology diagnostic versus topology-only baseline.}}
\end{{figure}}
M3 claim tests: H1={m3['claim_tests']['H1_mechanism_over_rate']}, H2={m3['claim_tests']['H2_rank_instability']}, H3={m3['claim_tests']['H3_typed_minimal_correction']}.
\section{{Limitations}}
The current artifact is a benchmark/theory package with official TSLib PatchTST/TimeXer architecture reproduction. A stronger camera-ready version should add more seeds and missing-specific baselines while keeping benchmark-only scope because the typed correction does not pass robustly.
\bibliographystyle{{plain}}
\bibliography{{references}}
\end{{document}}
"""
    (PAPER_DIR / "main.tex").write_text(tex)
    (PAPER_DIR / "references.bib").write_text(BIB.strip() + "\n")
    (PAPER_DIR / "README.md").write_text(
        "# MaskShift Paper Build\n\n"
        "From `experiments/MaskShift/paper`, run `pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex`.\n"
        "The figures are generated by `python -m experiments.MaskShift.m4_paper_ready`.\n"
    )


def main() -> None:
    m0 = load_json(EXP_DIR / "m0_mask_suite" / "m0_summary.json")
    m1 = load_json(EXP_DIR / "m1_mechanism_audit" / "m1_summary.json")
    m2 = load_json(EXP_DIR / "m2_typed_head" / "m2_summary.json")
    m3 = load_json(EXP_DIR / "m3_statistical_tests" / "m3_summary.json")
    m6_path = EXP_DIR / "m6_deep_backbone_sweep" / "deep_backbone_sweep_summary.json"
    m6 = load_json(m6_path) if m6_path.exists() else None
    m7_path = EXP_DIR / "m7_severity_curves" / "severity_curves_summary.json"
    m7 = load_json(m7_path) if m7_path.exists() else None
    m8_path = EXP_DIR / "m8_mechanism_decomposition" / "mechanism_decomposition_summary.json"
    m8 = load_json(m8_path) if m8_path.exists() else None
    m9_path = EXP_DIR / "m9_official_tslib_reproduction" / "official_tslib_reproduction_summary.json"
    m9 = load_json(m9_path) if m9_path.exists() else None
    figs = make_figures(m1, m2)
    strong = (
        m1.get("m1_gate")
        and m2.get("m2_gate")
        and m3["claim_tests"]["H1_mechanism_over_rate"]
        and m3["claim_tests"]["H2_rank_instability"]
    )
    if strong and m3["claim_tests"]["H3_typed_minimal_correction"]:
        paper_route = "strong main-track candidate after deep-backbone expansion"
    elif m3["claim_tests"]["H1_mechanism_over_rate"] and m3["claim_tests"]["H2_rank_instability"]:
        paper_route = "benchmark/theory main-track candidate; typed correction secondary"
    else:
        paper_route = "pilot evidence only; not submission-ready"
    summary = {
        "milestone": "M4",
        "status": "PAPER_READY_PACKAGE_WRITTEN",
        "paper_route": paper_route,
        "artifact_manifest": [
            "PAPER.md",
            "REPORT.md",
            "paper/main.tex",
            "paper/references.bib",
            "paper/README.md",
            "figures/m1_mechanism_rank.png",
            "figures/m2_typed_correction.png",
            "m0_mask_suite/m0_summary.json",
            "m1_mechanism_audit/m1_summary.json",
            "m2_typed_head/m2_summary.json",
            "m3_statistical_tests/m3_summary.json",
            "m4_paper_ready/paper_ready_summary.json",
            "m4_paper_ready/REPRODUCE.md",
            "m6_deep_backbone_sweep/deep_backbone_sweep_summary.json",
            "m7_severity_curves/severity_curves_summary.json",
            "m8_mechanism_decomposition/mechanism_decomposition_summary.json",
            "m9_official_tslib_reproduction/official_tslib_reproduction_summary.json",
        ],
        "m6_lite_status": m6.get("status") if m6 else "not_run",
        "m7_status": m7.get("status") if m7 else "not_run",
        "m8_status": m8.get("status") if m8 else "not_run",
        "m9_status": m9.get("status") if m9 else "not_run",
    }
    write_paper(m0, m1, m2, m3, m6, m7, m8, m9, figs, summary)
    reproduce = """# MaskShift Reproduction

Run from repository root:

```bash
python3 -m experiments.MaskShift.m0_mask_suite
python3 -m experiments.MaskShift.m1_mechanism_audit
python3 -m experiments.MaskShift.m2_typed_head
python3 -m experiments.MaskShift.m3_statistical_tests
python3 -m experiments.MaskShift.m6_deep_backbone_sweep
python3 -m experiments.MaskShift.m7_severity_curves
python3 -m experiments.MaskShift.m8_mechanism_decomposition
python3 -m experiments.MaskShift.m9_official_tslib_reproduction
python3 -m experiments.MaskShift.m4_paper_ready
python3 -m experiments.MaskShift.m5_main_track_audit
```

Outputs are written under `experiments/MaskShift/`.
"""
    (OUT_DIR / "REPRODUCE.md").write_text(reproduce)
    write_json(OUT_DIR / "paper_ready_summary.json", summary)
    print(json.dumps({"milestone": "M4", "status": summary["status"], "route": paper_route}, indent=2))


if __name__ == "__main__":
    main()
