"""M18 -- submission policy, disclosure, and venue-readiness package.

Strong ML venues increasingly require statements beyond the scientific result:
data/code availability, ethics, AI-assistance disclosure, conflicts/funding,
author contribution handling, anonymity, and reproducibility. This milestone
generates a target-agnostic policy pack without pretending to satisfy a
specific venue template before the target venue is fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

from .maskshift_core import ensure_dir, write_json


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m18_submission_policy_pack")
DOC_DIR = ensure_dir(EXP_DIR / "docs")
TABLE_DIR = ensure_dir(EXP_DIR / "tables")
PAPER_DIR = ensure_dir(EXP_DIR / "paper")

STATEMENTS = [
    {
        "name": "Data Availability Statement",
        "status": "review-ready",
        "text": "The experiments use time-series benchmark files expected under the repository's input/ directory. The MaskShift package records the exact local paths, selected rows/channels, natural missing rates, and derived windows in M0/M16/M17 artifacts. No private human-subject data is introduced by MaskShift.",
        "reviewer_risk": "Raw-dataset license/source metadata should be checked against the target venue's artifact policy before public release.",
    },
    {
        "name": "Code Availability Statement",
        "status": "review-ready",
        "text": "All MaskShift milestone scripts M0-M18, generated JSON summaries, tables, figures, manuscript TeX, and supplement TeX/PDF are included under experiments/MaskShift. External official-architecture repositories are pinned to TSLib 4e938a1, ChannelTokenFormer b1c100e, and S4M a718823.",
        "reviewer_risk": "Before final submission, package the ignored external/ revisions or provide exact clone commands in the anonymous artifact instructions.",
    },
    {
        "name": "Reproducibility Statement",
        "status": "review-ready",
        "text": "All reported MaskShift runs use deterministic seeds where applicable, chronological splits, encoder-input-only masks, clean targets, and generated JSON summaries. M15 checks numerical traceability from JSON summaries to paper/tables and M5 requires all scientific gates.",
        "reviewer_risk": "Full original-protocol S4M reproduction remains outside the current evidence boundary and must not be implied.",
    },
    {
        "name": "Ethics Statement",
        "status": "review-ready",
        "text": "The work is a benchmark/theory study over time-series datasets and synthetic missingness mechanisms. It does not involve human-subject intervention, user profiling, or sensitive personal data collection. The principal risk is over-trusting MCAR/block robustness in operational systems; the paper mitigates this by arguing for mechanism reporting rather than claiming a universal repair.",
        "reviewer_risk": "If a target venue requires broader-impact forms, reuse this statement but add venue-specific checkboxes.",
    },
    {
        "name": "AI Assistance Disclosure",
        "status": "review-ready",
        "text": "AI assistance was used to help organize the research pipeline, draft/edit text, and generate local audit scripts. The authors remain responsible for all claims. Numerical statements are traced to local JSON artifacts by M15, and citations are checked for dangling/orphan references and external audit-source records.",
        "reviewer_risk": "Some venues require exact AI tool names or prompts; add those details only if the target policy requests them.",
    },
    {
        "name": "Conflict of Interest Statement",
        "status": "anonymous-review placeholder",
        "text": "For double-blind review, author-identifying conflict details should be supplied through the submission system rather than the anonymous manuscript. The camera-ready version should include the final conflict declaration.",
        "reviewer_risk": "Author-specific conflicts cannot be completed without the author list and venue system fields.",
    },
    {
        "name": "Funding Statement",
        "status": "anonymous-review placeholder",
        "text": "For double-blind review, funding acknowledgments should be omitted from the anonymous manuscript when they identify the authors. The camera-ready version should include final funding information or state that no external funding was received.",
        "reviewer_risk": "Funding details require author confirmation.",
    },
    {
        "name": "Author Contributions",
        "status": "camera-ready placeholder",
        "text": "Author contributions should be reported in the camera-ready version using the target venue's preferred format, such as CRediT roles, after the anonymous review phase.",
        "reviewer_risk": "Contribution allocation requires the final author list.",
    },
    {
        "name": "Anonymity Statement",
        "status": "review-ready",
        "text": "The manuscript and supplement use anonymous authors. Repository paths in reproduction notes should be replaced by anonymous artifact URLs or relative paths before upload if required by the target venue.",
        "reviewer_risk": "Local absolute paths in JSON artifacts are useful for audit but should be scrubbed or mapped in any public anonymous artifact bundle.",
    },
]

VENUE_CHECKS = [
    {
        "item": "Double-blind manuscript",
        "status": "pass-with-artifact-pack-note",
        "evidence": "paper/main.tex uses Anonymous authors; M18 flags local paths for artifact bundle scrubbing.",
    },
    {
        "item": "Page-limit readiness",
        "status": "target-dependent",
        "evidence": "main.pdf is 8 pages under generic article class; final target template is still pending.",
    },
    {
        "item": "Supplement readiness",
        "status": "pass",
        "evidence": "M17 supplement.pdf, dataset/mechanism cards, rebuttal playbook, and response matrix are generated.",
    },
    {
        "item": "Reproducibility checklist",
        "status": "pass",
        "evidence": "M0-M18 commands, seeds, external revisions, and generated JSON summaries are recorded.",
    },
    {
        "item": "Ethics and broader-impact statement",
        "status": "pass",
        "evidence": "M18 submission statements include ethics, risks, and mitigation scope.",
    },
    {
        "item": "AI disclosure",
        "status": "pass",
        "evidence": "M18 includes AI-assistance disclosure and links numerical/citation integrity to M15.",
    },
    {
        "item": "Data/code availability",
        "status": "pass-with-license-note",
        "evidence": "M18 records local dataset/code availability and target-policy caveat for raw-data license metadata.",
    },
    {
        "item": "Target venue style",
        "status": "pending-target-selection",
        "evidence": "The current paper uses a generic article class; replace with the selected venue template before upload.",
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_optional_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def write_text_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text() == text:
        return
    path.write_text(text)


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


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


def write_statements_md(m5: dict, m15: dict, m17: dict) -> None:
    rows = [[s["name"], s["status"], s["reviewer_risk"]] for s in STATEMENTS]
    lines = [
        "# MaskShift Submission Statements",
        "",
        "This document is a target-agnostic submission policy pack. It should be copied into the target venue's required fields or appendix once the venue is fixed.",
        "",
        "## Gate Context",
        "",
        f"- M5 readiness: `{m5.get('verdict')}`; blocking items: `{len(m5.get('blocking_items', []))}`",
        f"- M15 integrity: `{m15.get('verdict')}`; blocking issues: `{len(m15.get('blocking_issues', []))}`",
        f"- M17 supplement: `{m17.get('status')}`; reviewer concerns covered: `{m17.get('reviewer_concern_count', 0)}`",
        "",
        "## Statement Index",
        "",
        md_table(["Statement", "Status", "Residual venue-specific risk"], rows),
        "",
    ]
    for item in STATEMENTS:
        lines += [
            f"## {item['name']}",
            "",
            f"Status: `{item['status']}`",
            "",
            item["text"],
            "",
            f"Residual venue-specific risk: {item['reviewer_risk']}",
            "",
        ]
    lines += [
        "## Target-Specific Action",
        "",
        "Replace the generic article class with the selected venue style and map these statements into the target submission system. Do not claim target-specific compliance until the venue template and policy fields have been checked.",
    ]
    write_text_if_changed(DOC_DIR / "SUBMISSION_STATEMENTS.md", "\n".join(lines) + "\n")


def write_venue_audit_md() -> None:
    rows = [[row["item"], row["status"], row["evidence"]] for row in VENUE_CHECKS]
    lines = [
        "# MaskShift Venue-Readiness Audit",
        "",
        "This audit separates target-agnostic readiness from target-specific formatting. It is intended to prevent accidental claims that the generic article manuscript already satisfies a named venue template.",
        "",
        md_table(["Item", "Status", "Evidence"], rows),
        "",
        "## Interpretation",
        "",
        "The scientific and policy-support package is ready for strong-conference review. The remaining venue-specific task is mechanical: select the target venue, replace the generic article class with that venue's template, and map statements into the required fields.",
    ]
    write_text_if_changed(DOC_DIR / "VENUE_READINESS_AUDIT.md", "\n".join(lines) + "\n")


def write_policy_table() -> None:
    rows = [[s["name"], s["status"], s["reviewer_risk"]] for s in STATEMENTS]
    text = (
        "# M18 submission policy readiness table\n\n"
        + md_table(["Statement", "Status", "Residual venue-specific risk"], rows)
        + "\n"
    )
    write_text_if_changed(TABLE_DIR / "m18_policy_readiness_table.md", text)


def write_statements_tex(m5: dict, m15: dict, m17: dict) -> None:
    statement_items = "\n".join(
        "\\subsection{%s}\n\\textbf{Status:} \\texttt{%s}. %s\n\n\\textbf{Residual venue-specific risk:} %s\n"
        % (
            tex_escape(item["name"]),
            tex_escape(item["status"]),
            tex_escape(item["text"]),
            tex_escape(item["reviewer_risk"]),
        )
        for item in STATEMENTS
    )
    venue_rows = "\n".join(
        "%s & %s & %s\\\\"
        % (tex_escape(row["item"]), tex_escape(row["status"]), tex_escape(row["evidence"]))
        for row in VENUE_CHECKS
    )
    text = r"""\documentclass[10pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\title{MaskShift Submission Policy Statements}
\author{Anonymous}
\date{}

\begin{document}
\maketitle

\section{Gate Context}
M5 readiness: \texttt{%s}, blocking items: %s. M15 integrity: \texttt{%s}, blocking issues: %s. M17 supplement: \texttt{%s}, reviewer concerns covered: %s.

\section{Submission Statements}
%s

\section{Venue-Readiness Audit}
\small
\begin{longtable}{p{0.22\linewidth}p{0.22\linewidth}p{0.45\linewidth}}
\toprule
Item & Status & Evidence\\
\midrule
%s
\bottomrule
\end{longtable}
\normalsize

\section{Target-Specific Action}
This policy pack is target-agnostic. Replace the generic article class with the selected venue template and map these statements into the target submission system before upload. Do not claim target-specific compliance until the venue policy fields have been checked.

\end{document}
""" % (
        tex_escape(m5.get("verdict")),
        len(m5.get("blocking_items", [])),
        tex_escape(m15.get("verdict")),
        len(m15.get("blocking_issues", [])),
        tex_escape(m17.get("status")),
        m17.get("reviewer_concern_count", 0),
        statement_items,
        venue_rows,
    )
    write_text_if_changed(PAPER_DIR / "submission_statements.tex", text)


def main() -> None:
    m5 = load_optional_json(EXP_DIR / "m5_main_track_audit" / "main_track_audit.json")
    m15 = load_optional_json(EXP_DIR / "m15_final_integrity_audit" / "final_integrity_summary.json")
    m17 = load_optional_json(EXP_DIR / "m17_submission_supplement" / "submission_supplement_summary.json")
    write_statements_md(m5, m15, m17)
    write_venue_audit_md()
    write_policy_table()
    write_statements_tex(m5, m15, m17)

    artifacts = [
        "docs/SUBMISSION_STATEMENTS.md",
        "docs/VENUE_READINESS_AUDIT.md",
        "tables/m18_policy_readiness_table.md",
        "paper/submission_statements.tex",
        "paper/submission_statements.pdf",
        "m18_submission_policy_pack/submission_policy_summary.json",
    ]
    missing = [
        path
        for path in artifacts
        if path != "m18_submission_policy_pack/submission_policy_summary.json" and not (EXP_DIR / path).exists()
    ]
    pdf_path = EXP_DIR / "paper" / "submission_statements.pdf"
    tex_path = EXP_DIR / "paper" / "submission_statements.tex"
    pdf_fresh = pdf_path.exists() and tex_path.exists() and pdf_path.stat().st_mtime >= tex_path.stat().st_mtime
    pdf_ok = pdf_path.exists() and pdf_path.stat().st_size > 20_000 and pdf_fresh
    required_sections = [item["name"] for item in STATEMENTS]
    statements_text = (DOC_DIR / "SUBMISSION_STATEMENTS.md").read_text()
    section_misses = [section for section in required_sections if section not in statements_text]
    blocking_policy_gaps = [
        row for row in VENUE_CHECKS if row["status"] in {"fail", "missing"}
    ]
    status = (
        "PASS_SUBMISSION_POLICY_PACK"
        if not missing and pdf_ok and not section_misses and not blocking_policy_gaps
        else "HOLD_SUBMISSION_POLICY_PACK"
    )
    summary = {
        "milestone": "M18",
        "status": status,
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "submission_statements_pdf_size": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "submission_statements_pdf_fresh": pdf_fresh,
        "statement_count": len(STATEMENTS),
        "statements": STATEMENTS,
        "venue_checks": VENUE_CHECKS,
        "section_misses": section_misses,
        "blocking_policy_gaps": blocking_policy_gaps,
        "target_specific_template_status": "pending-target-selection",
        "scope_note": "Target-agnostic policy and disclosure pack. It does not claim compliance with a named venue template before target selection.",
    }
    write_json(OUT_DIR / "submission_policy_summary.json", summary)
    print(
        json.dumps(
            {
                "milestone": "M18",
                "status": status,
                "pdf_ok": pdf_ok,
                "pdf_fresh": pdf_fresh,
                "missing_artifacts": missing,
                "section_misses": section_misses,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
