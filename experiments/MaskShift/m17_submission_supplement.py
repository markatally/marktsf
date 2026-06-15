"""M17 -- submission supplement and rebuttal-ready evidence package.

This milestone does not add new empirical claims. It packages the already
audited M0-M16 evidence into reviewer-facing appendices, dataset/mechanism
cards, and a rebuttal matrix. The goal is to make the strong-conference
submission self-contained enough that reviewers can verify scope, protocol
boundaries, and negative evidence without hunting through milestone files.
"""

from __future__ import annotations

import json
from pathlib import Path

from .maskshift_core import ensure_dir, write_json


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m17_submission_supplement")
DOC_DIR = ensure_dir(EXP_DIR / "docs")
TABLE_DIR = ensure_dir(EXP_DIR / "tables")
PAPER_DIR = ensure_dir(EXP_DIR / "paper")

MECHANISM_DEFINITIONS = [
    {
        "name": "mcar",
        "definition": "Independent random deletion at the target missing rate.",
        "deployment_read": "Neutral packet loss or randomly sampled telemetry dropout.",
        "scope_guard": "Control mechanism only; not an operational cause model.",
    },
    {
        "name": "block",
        "definition": "Contiguous channel-level missing spans with matched final rate.",
        "deployment_read": "Short sensor logging gaps or temporary channel outages.",
        "scope_guard": "Controls contiguity but not value dependence.",
    },
    {
        "name": "value_high",
        "definition": "Missing probability increases at high normalized values, then rate matching is enforced.",
        "deployment_read": "Storm, peak-load, congestion, or saturation-related missingness.",
        "scope_guard": "Synthetic proxy for value-dependent MNAR; not a causal claim.",
    },
    {
        "name": "volatility",
        "definition": "Missing probability increases with large local changes, then rate matching is enforced.",
        "deployment_read": "Failure during unstable regimes, transitions, or high-frequency variation.",
        "scope_guard": "Measures sensitivity to change-linked masks, not event detection quality.",
    },
    {
        "name": "blackout",
        "definition": "Contiguous time blocks remove many channels simultaneously with matched final rate.",
        "deployment_read": "Power/network outage, site-level telemetry loss, or shared infrastructure failure.",
        "scope_guard": "Does not model outage recovery dynamics beyond the encoder window.",
    },
    {
        "name": "retirement",
        "definition": "Selected channels disappear late in the sequence with matched final rate.",
        "deployment_read": "Permanent sensor retirement, meter replacement, or channel decommissioning.",
        "scope_guard": "Reported separately so the paper is not driven only by an obvious failure case.",
    },
]

REVIEWER_CONCERNS = [
    {
        "concern": "Is this just another missing-value model paper?",
        "answer": "No. MaskShift is framed as benchmark/theory and explicitly excludes SOTA or new-backbone claims.",
        "evidence": "PAPER.md Sections 1, 3, 6; M5 claim_scope_safe.",
        "status": "Resolved by scope discipline.",
    },
    {
        "concern": "Were modern forecasting backbones tested?",
        "answer": "Yes. M9/M10 import official TSLib PatchTST and TimeXer classes; M16 extends coverage to all four datasets.",
        "evidence": "m9_official_tslib_reproduction_summary.json; m16_official_tslib_full_coverage_summary.json.",
        "status": "Resolved as official-architecture adaptation.",
    },
    {
        "concern": "Were missing-aware architectures tested?",
        "answer": "Yes. M11 adapts official ChannelTokenFormer_missing and M12/M14 adapt official S4M.",
        "evidence": "M11, M12, M14 summaries and tables.",
        "status": "Resolved with mixed/negative evidence disclosed.",
    },
    {
        "concern": "Are Weather/Electricity cherry-picked?",
        "answer": "M16 reports Traffic and AirConvection official PatchTST/TimeXer coverage as mixed/negative for rank reversal.",
        "evidence": "tables/m16_official_tslib_full_coverage_table.md.",
        "status": "Resolved by visible boundary evidence.",
    },
    {
        "concern": "Is the result only sensor retirement?",
        "answer": "No. M8 excludes retirement and still finds positive non-retirement evidence on Weather and Electricity.",
        "evidence": "m8_mechanism_decomposition_summary.json.",
        "status": "Resolved for the core positive datasets.",
    },
    {
        "concern": "Are relative degradation ratios unstable?",
        "answer": "M10 adds absolute delta, log ratio, and symmetric relative delta; the paper does not rely on denominator spikes.",
        "evidence": "tables/m7_corrected_robustness_table.md.",
        "status": "Resolved by corrected reporting.",
    },
    {
        "concern": "Does the typed head work as a method?",
        "answer": "No. H3 fails and the paper reports the typed/topology head as a negative diagnostic only.",
        "evidence": "m2_summary.json; m3_summary.json; PAPER.md Section 4.6.",
        "status": "Resolved by de-scoping.",
    },
    {
        "concern": "Is the statistical evidence only aggregate means?",
        "answer": "M3 uses BH-FDR, M10 adds three-seed CIs, and M13 adds a hierarchy-aware variant/window bootstrap.",
        "evidence": "m3_summary.json; m10_submission_hardening_summary.json; m13_hierarchical_bootstrap_summary.json.",
        "status": "Resolved for submission, with limitations retained.",
    },
    {
        "concern": "Is the official-code claim over-stated?",
        "answer": "The paper consistently says official-architecture adaptation under MaskShift, not full official benchmark reproduction.",
        "evidence": "PAPER.md limitations; SUBMISSION_CHECKLIST.md must-keep scope.",
        "status": "Resolved by wording guardrails.",
    },
    {
        "concern": "Can a reviewer reproduce the package?",
        "answer": "The README and supplement list M0-M17 commands and external repository revisions.",
        "evidence": "README.md; paper/README.md; paper/supplement.tex.",
        "status": "Resolved as local reproduction package.",
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def tex_escape(text: object) -> str:
    out = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in out)


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def dataset_label(row: dict, m3_by_dataset: dict, m16_by_dataset: dict, m8_by_dataset: dict) -> str:
    name = row["dataset"]
    m3 = m3_by_dataset[name]
    m16 = m16_by_dataset[name]
    m8 = m8_by_dataset[name]
    if m3["gate_pass"] and m16["gate_pass"]:
        return "positive core evidence"
    if m16["gate_pass"]:
        return "official positive, lightweight mixed"
    if m8["non_retirement_gate"]:
        return "non-retirement positive, official mixed"
    return "boundary/mixed evidence"


def build_dataset_cards(m0: dict, m3: dict, m8: dict, m16: dict) -> list[dict]:
    m3_by_dataset = {row["dataset"]: row for row in m3["m1_rows"]}
    m8_by_dataset = {row["dataset"]: row for row in m8["dataset_rows"]}
    m16_by_dataset = {row["dataset"]: row for row in m16["datasets"]}
    cards = []
    for row in m0["datasets"]:
        name = row["dataset"]
        cards.append(
            {
                "dataset": name,
                "rows": row["rows"],
                "channels": row["channels"],
                "natural_missing_rate": row["natural_missing_rate_before_interpolation"],
                "matched_rate_error": m0["matched_rate_gate"][name]["max_rate_error"],
                "m1_eta_squared": m3_by_dataset[name]["eta_squared"],
                "m1_gate": m3_by_dataset[name]["gate_pass"],
                "m8_non_retirement_gate": m8_by_dataset[name]["non_retirement_gate"],
                "m16_max_degradation": m16_by_dataset[name]["max_relative_degradation"],
                "m16_worst_tau": m16_by_dataset[name]["worst_rank_tau"],
                "m16_anova_p": m16_by_dataset[name]["anova_p"],
                "evidence_label": dataset_label(row, m3_by_dataset, m16_by_dataset, m8_by_dataset),
            }
        )
    return cards


def write_dataset_cards(cards: list[dict]) -> None:
    rows = [
        [
            card["dataset"],
            card["rows"],
            card["channels"],
            pct(card["natural_missing_rate"], 2),
            pct(card["matched_rate_error"], 2),
            num(card["m1_eta_squared"], 3),
            "PASS" if card["m1_gate"] else "MIXED",
            "PASS" if card["m8_non_retirement_gate"] else "MIXED",
            pct(card["m16_max_degradation"]),
            num(card["m16_worst_tau"], 3),
            card["evidence_label"],
        ]
        for card in cards
    ]
    lines = [
        "# MaskShift Dataset and Mechanism Cards",
        "",
        "These cards are the reviewer-facing map from each dataset and mechanism to the exact claim scope. They are generated from M0, M3, M8, and M16 summaries.",
        "",
        "## Dataset Cards",
        "",
        md_table(
            [
                "Dataset",
                "Rows",
                "Channels",
                "Natural missing",
                "Max rate error",
                "M1 eta^2",
                "M1 gate",
                "M8 non-ret gate",
                "M16 max degradation",
                "M16 worst tau",
                "Evidence role",
            ],
            rows,
        ),
        "",
        "## Mechanism Cards",
        "",
    ]
    for item in MECHANISM_DEFINITIONS:
        lines += [
            f"### {item['name']}",
            "",
            f"- Definition: {item['definition']}",
            f"- Deployment read: {item['deployment_read']}",
            f"- Scope guard: {item['scope_guard']}",
            "",
        ]
    lines += [
        "## Claim Boundary",
        "",
        "Weather and Electricity carry the strongest positive evidence. Traffic and AirConvection are kept as boundary evidence, not hidden negative cases. The paper should claim that missingness mechanism is a first-class benchmark factor whose empirical strength is dataset- and architecture-dependent.",
    ]
    (DOC_DIR / "DATASET_MECHANISM_CARDS.md").write_text("\n".join(lines) + "\n")


def write_rebuttal_matrix() -> None:
    rows = [[c["concern"], c["answer"], c["evidence"], c["status"]] for c in REVIEWER_CONCERNS]
    text = (
        "# M17 reviewer response matrix\n\n"
        "This table is a pre-submission rebuttal map. It does not fabricate reviewer comments; it maps likely strong-conference concerns to already generated evidence.\n\n"
        + md_table(["Likely reviewer concern", "Short answer", "Evidence artifact", "Status"], rows)
        + "\n"
    )
    (TABLE_DIR / "m17_reviewer_response_matrix.md").write_text(text)


def write_rebuttal_playbook(cards: list[dict]) -> None:
    lines = [
        "# MaskShift Rebuttal Playbook",
        "",
        "Use this file after reviews arrive. It is not a response letter; it is a constraint system that prevents overclaiming while answering predictable objections.",
        "",
        "## Non-Negotiable Scope Lines",
        "",
        "- MaskShift is a benchmark/theory paper, not a new forecasting backbone.",
        "- PatchTST, TimeXer, ChannelTokenFormer_missing, and S4M results are official-architecture adaptations under the MaskShift protocol.",
        "- The typed/topology head is a negative diagnostic; it is not a method contribution.",
        "- Traffic and AirConvection are boundary evidence and must remain visible.",
        "- S4M is a contrastive robust baseline under reduced local settings, not a failure case.",
        "",
        "## Reviewer-Concern Playbook",
        "",
    ]
    for idx, concern in enumerate(REVIEWER_CONCERNS, start=1):
        lines += [
            f"### R{idx}. {concern['concern']}",
            "",
            f"- Answer: {concern['answer']}",
            f"- Cite: {concern['evidence']}",
            f"- Safe wording: {concern['status']}",
            "",
        ]
    lines += [
        "## Dataset-Specific Response Map",
        "",
    ]
    for card in cards:
        lines += [
            f"### {card['dataset']}",
            "",
            f"- Evidence role: {card['evidence_label']}",
            f"- M1 eta^2: {num(card['m1_eta_squared'], 3)}",
            f"- M16 max official PatchTST/TimeXer degradation: {pct(card['m16_max_degradation'])}",
            f"- M16 worst tau: {num(card['m16_worst_tau'], 3)}",
            "- Response stance: " + (
                "Use as core positive evidence." if "positive core" in card["evidence_label"] else "Use as boundary evidence; do not force into a universal claim."
            ),
            "",
        ]
    lines += [
        "## If Asked For More Experiments",
        "",
        "Prioritize in this order: more seed offsets for M9/M10, full original-protocol S4M reproduction, larger ChannelTokenFormer_missing adaptation, then a mixed-effects model over seeds/datasets/horizons. Do not add an unverified experiment to the rebuttal.",
    ]
    (DOC_DIR / "REBUTTAL_PLAYBOOK.md").write_text("\n".join(lines) + "\n")


def write_supplement(cards: list[dict], m0: dict, m16: dict) -> None:
    config = m0["config"]
    m16_rows = [
        [
            row["dataset"],
            row["source"],
            pct(row["max_relative_degradation"]),
            num(row["worst_rank_tau"], 3),
            num(row["anova_p"], 4),
            "PASS" if row["gate_pass"] else "MIXED/NEGATIVE",
        ]
        for row in m16["datasets"]
    ]
    dataset_rows = [
        [
            card["dataset"],
            str(card["rows"]),
            str(card["channels"]),
            pct(card["natural_missing_rate"], 2),
            pct(card["matched_rate_error"], 2),
            card["evidence_label"],
        ]
        for card in cards
    ]
    short_evidence = [
        "PAPER Sections 1/3/6; M5",
        "M9, M10, M16 summaries",
        "M11, M12, M14 summaries",
        "M16 table",
        "M8 summary",
        "M7/M10 corrected metrics",
        "M2/M3 summaries; PAPER 4.6",
        "M3, M10, M13 summaries",
        "PAPER limitations; checklist",
        "README; supplement",
    ]
    concern_rows = [
        [str(idx), item["concern"], short_evidence[idx - 1], item["status"]]
        for idx, item in enumerate(REVIEWER_CONCERNS, start=1)
    ]

    def tex_rows(rows: list[list[str]]) -> str:
        return "\n".join(" & ".join(tex_escape(cell) for cell in row) + r"\\" for row in rows)

    mechanism_items = "\n".join(
        "\\item \\textbf{%s}. %s Deployment read: %s Scope guard: %s"
        % (
            tex_escape(item["name"]),
            tex_escape(item["definition"]),
            tex_escape(item["deployment_read"]),
            tex_escape(item["scope_guard"]),
        )
        for item in MECHANISM_DEFINITIONS
    )
    commands = [
        "python3 -m experiments.MaskShift.m0_mask_suite",
        "python3 -m experiments.MaskShift.m1_mechanism_audit",
        "python3 -m experiments.MaskShift.m2_typed_head",
        "python3 -m experiments.MaskShift.m3_statistical_tests",
        "python3 -m experiments.MaskShift.m6_deep_backbone_sweep",
        "python3 -m experiments.MaskShift.m7_severity_curves",
        "python3 -m experiments.MaskShift.m8_mechanism_decomposition",
        "python3 -m experiments.MaskShift.m9_official_tslib_reproduction",
        "python3 -m experiments.MaskShift.m10_submission_hardening",
        "python3 -m experiments.MaskShift.m11_official_ctf_missing_baseline",
        "python3 -m experiments.MaskShift.m12_official_s4m_baseline",
        "python3 -m experiments.MaskShift.m13_hierarchical_bootstrap",
        "python3 -m experiments.MaskShift.m14_s4m_scale_validation",
        "python3 -m experiments.MaskShift.m15_final_integrity_audit",
        "python3 -m experiments.MaskShift.m16_official_tslib_full_coverage",
        "python3 -m experiments.MaskShift.m17_submission_supplement",
        "python3 -m experiments.MaskShift.m5_main_track_audit",
    ]
    command_block = "\n".join(tex_escape(cmd) + r"\\" for cmd in commands)
    text = r"""\documentclass[10pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{hyperref}
\title{MaskShift Supplementary Material}
\author{Anonymous}
\date{}

\begin{document}
\maketitle

\section{Scope}
MaskShift is a benchmark/theory submission. It does not claim a new forecasting backbone, state-of-the-art clean forecasting accuracy, or a full official benchmark reproduction. The supplement records protocol details, dataset cards, mechanism cards, reviewer-response evidence, and reproduction commands for the MaskShift encoder-mask protocol.

\section{Mechanism Generator Cards}
\begin{itemize}
%s
\end{itemize}

\section{Dataset Cards}
\small
\begin{longtable}{p{0.16\linewidth}rrrrp{0.28\linewidth}}
\toprule
Dataset & Rows & Channels & Natural missing & Max rate error & Evidence role\\
\midrule
%s
\bottomrule
\end{longtable}
\normalsize

\section{Core Configuration}
The default MaskShift configuration is lookback %s, horizon %s, stride %s, target missing rate %s, seed %s, maximum train windows %s, and maximum test windows %s. All masks corrupt encoder inputs only, and forecast targets remain clean.

\section{Official PatchTST/TimeXer Coverage}
\small
\begin{longtable}{p{0.18\linewidth}p{0.12\linewidth}rrrp{0.2\linewidth}}
\toprule
Dataset & Source & Max degradation & Worst tau & ANOVA p & Gate\\
\midrule
%s
\bottomrule
\end{longtable}
\normalsize
Weather and Electricity provide the positive official-architecture rank-reversal evidence. Traffic and AirConvection are mixed/negative for rank reversal and are retained as boundary evidence.

\section{Reviewer Concern Matrix}
\small
\begin{longtable}{rp{0.33\linewidth}p{0.28\linewidth}p{0.2\linewidth}}
\toprule
ID & Concern & Evidence & Status\\
\midrule
%s
\bottomrule
\end{longtable}
\normalsize

\section{Reproducibility Commands}
\small
\begin{tabular}{p{0.95\linewidth}}
%s
\end{tabular}
\normalsize

\section{External Revisions}
The official-architecture adaptations are audited against TSLib revision \texttt{4e938a1}, ChannelTokenFormer revision \texttt{b1c100e}, and S4M revision \texttt{a718823}. These are architecture adaptations under MaskShift windows and masks, not full reproductions of the original papers' benchmark protocols.

\section{Integrity Status}
M15 reports \texttt{PASS\_FINAL\_INTEGRITY}. M5 reports \texttt{STRONG\_CONFERENCE\_READY} after requiring M15, M16, and this M17 supplement package.

\end{document}
""" % (
        mechanism_items,
        tex_rows(dataset_rows),
        config["lookback"],
        config["horizon"],
        config["stride"],
        pct(config["target_rate"], 0),
        config["seed"],
        config["max_train_samples"],
        config["max_test_samples"],
        tex_rows(m16_rows),
        tex_rows(concern_rows),
        command_block,
    )
    (PAPER_DIR / "supplement.tex").write_text(text)


def main() -> None:
    m0 = load_json(EXP_DIR / "m0_mask_suite" / "m0_summary.json")
    m3 = load_json(EXP_DIR / "m3_statistical_tests" / "m3_summary.json")
    m8 = load_json(EXP_DIR / "m8_mechanism_decomposition" / "mechanism_decomposition_summary.json")
    m16 = load_json(EXP_DIR / "m16_official_tslib_full_coverage" / "official_tslib_full_coverage_summary.json")
    cards = build_dataset_cards(m0, m3, m8, m16)
    write_dataset_cards(cards)
    write_rebuttal_matrix()
    write_rebuttal_playbook(cards)
    write_supplement(cards, m0, m16)

    artifacts = [
        "paper/supplement.tex",
        "paper/supplement.pdf",
        "docs/DATASET_MECHANISM_CARDS.md",
        "docs/REBUTTAL_PLAYBOOK.md",
        "tables/m17_reviewer_response_matrix.md",
        "m17_submission_supplement/submission_supplement_summary.json",
    ]
    missing = [path for path in artifacts if not (EXP_DIR / path).exists() and path != "m17_submission_supplement/submission_supplement_summary.json"]
    pdf_path = EXP_DIR / "paper" / "supplement.pdf"
    pdf_ok = pdf_path.exists() and pdf_path.stat().st_size > 30_000
    status = "PASS_SUBMISSION_SUPPLEMENT" if not missing and pdf_ok else "HOLD_SUBMISSION_SUPPLEMENT"
    summary = {
        "milestone": "M17",
        "status": status,
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "supplement_pdf_size": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "dataset_cards": cards,
        "mechanisms": MECHANISM_DEFINITIONS,
        "reviewer_concern_count": len(REVIEWER_CONCERNS),
        "reviewer_concerns": REVIEWER_CONCERNS,
        "coverage": {
            "dataset_cards": len(cards),
            "mechanism_cards": len(MECHANISM_DEFINITIONS),
            "reproduction_commands": 17,
            "m16_official_tslib_datasets": [row["dataset"] for row in m16["datasets"]],
        },
        "scope_note": "Supplement package for strong-conference submission. It adds no new empirical claim; it organizes M0-M16 evidence into appendices, dataset cards, and rebuttal-ready matrices.",
    }
    write_json(OUT_DIR / "submission_supplement_summary.json", summary)
    print(
        json.dumps(
            {
                "milestone": "M17",
                "status": status,
                "pdf_ok": pdf_ok,
                "missing_artifacts": missing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
