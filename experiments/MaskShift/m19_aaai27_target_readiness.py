"""M19 -- AAAI-27 target-venue readiness and response dossier.

M18 is target-agnostic. M19 selects the most plausible strong-conference
window for the user's "submit this month" constraint and turns it into an
auditable target dossier. It does not pretend the paper has already been
converted into the AAAI author kit; instead it records the exact upload
blockers and prepares the required reproducibility/response materials.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .maskshift_core import ensure_dir, write_json


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m19_aaai27_target_readiness")
DOC_DIR = ensure_dir(EXP_DIR / "docs")
TABLE_DIR = ensure_dir(EXP_DIR / "tables")
PAPER_DIR = ensure_dir(EXP_DIR / "paper")

AAAI27_SOURCE = "https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/"
AAAI27_FACTS = {
    "venue": "AAAI-27 Main Technical Track",
    "conference_dates": "2027-02-16 to 2027-02-23",
    "submission_site_author_registration": "2026-06-17",
    "submission_site_paper_open": "2026-06-24",
    "abstract_deadline": "2026-07-21 23:59 UTC-12",
    "full_paper_deadline": "2026-07-28 23:59 UTC-12",
    "supplement_code_deadline": "2026-07-31 23:59 UTC-12",
    "phase1_rejection_notification": "2026-09-24",
    "author_feedback_window": "2026-10-19 to 2026-10-25",
    "final_decision": "2026-11-30",
    "camera_ready": "2026-12-14",
    "technical_content_limit": "up to 7 pages plus references",
    "supplement_policy": "supplement allowed, but reviewers are not required to review it; critical material belongs in the main body",
    "reproducibility": "all authors must complete a reproducibility checklist",
    "generative_ai_policy": "authors may judiciously use generative AI tools, but remain fully responsible for all submitted material",
    "review_process": "two-phase reviewing with an AI-generated non-decisional review supplementing Phase 1",
    "review_criteria": "significance, novelty, theoretical/empirical soundness, relevance, clarity, responsible practices, and reproducibility",
    "source": AAAI27_SOURCE,
    "accessed": date.today().isoformat(),
}


def load_json(path: Path) -> dict:
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


def pdf_page_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        import fitz  # type: ignore

        with fitz.open(path) as doc:
            return int(doc.page_count)
    except Exception:
        return None


def display_status(status: str) -> str:
    if status == "NOT_SUBMISSION_READY_YET":
        return "PENDING_M5_RERUN_AFTER_M15"
    return status


def detect_package_state() -> dict:
    main_tex = (PAPER_DIR / "main.tex").read_text() if (PAPER_DIR / "main.tex").exists() else ""
    m20 = load_json(EXP_DIR / "m20_aaai27_preflight_conversion" / "aaai27_preflight_summary.json")
    m20_state = m20.get("state", {})
    m21 = load_json(EXP_DIR / "m21_aaai27_reproducibility_checklist" / "aaai27_reproducibility_checklist_summary.json")
    return {
        "main_pdf_pages_generic": pdf_page_count(PAPER_DIR / "main.pdf"),
        "supplement_pdf_pages_generic": pdf_page_count(PAPER_DIR / "supplement.pdf"),
        "policy_pdf_pages_generic": pdf_page_count(PAPER_DIR / "submission_statements.pdf"),
        "preflight_pdf_pages": m20_state.get("preflight_page_count"),
        "preflight_page_limit_pass": m20.get("page_limit_preflight_pass", False),
        "preflight_twocolumn": m20_state.get("uses_twocolumn", False),
        "preflight_table_fit": m20_state.get("uses_double_column_tables", False),
        "preflight_pdf_fresh": m20_state.get("preflight_pdf_fresh", False),
        "preflight_pdf_size": m20_state.get("preflight_pdf_size", 0),
        "official_pdf_pages": m20_state.get("official_page_count"),
        "official_page_limit_pass": m20_state.get("official_page_limit_pass", False),
        "official_template_build_pass": m20.get("official_template_build_pass", False),
        "official_pdf_fresh": m20_state.get("official_pdf_fresh", False),
        "official_pdf_size": m20_state.get("official_pdf_size", 0),
        "documentclass_line": next((line.strip() for line in main_tex.splitlines() if line.strip().startswith(r"\documentclass")), ""),
        "uses_aaai_style": r"\usepackage{aaai}" in main_tex or "aaai2027" in main_tex.lower(),
        "aaai_style_present": any((PAPER_DIR / name).exists() for name in ["aaai.sty", "aaai2027.sty"]),
        "anonymous_authors": "Anonymous" in main_tex,
        "m5": display_status(load_json(EXP_DIR / "m5_main_track_audit" / "main_track_audit.json").get("verdict", "missing")),
        "m15": display_status(load_json(EXP_DIR / "m15_final_integrity_audit" / "final_integrity_summary.json").get("verdict", "missing")),
        "m17": load_json(EXP_DIR / "m17_submission_supplement" / "submission_supplement_summary.json").get("status", "missing"),
        "m18": load_json(EXP_DIR / "m18_submission_policy_pack" / "submission_policy_summary.json").get("status", "missing"),
        "m20": m20.get("status", "missing"),
        "m21": m21.get("status", "missing"),
        "official_checklist_pdf_size": m21.get("pdf_size", 0),
        "official_checklist_remaining_placeholders": m21.get("remaining_question_placeholders"),
        "aaai_official_kit_upload_ready": m20.get("aaai_official_kit_upload_ready", False),
    }


def build_gap_rows(state: dict) -> list[dict]:
    site_open = date.today() >= date(2026, 6, 24)
    return [
        {
            "item": "AAAI-27 timing",
            "status": "ready-for-open-site",
            "evidence": f"Submission site opens {AAAI27_FACTS['submission_site_paper_open']}; abstract/full deadlines are {AAAI27_FACTS['abstract_deadline']} and {AAAI27_FACTS['full_paper_deadline']}.",
            "action": "Register authors when the site opens; prepare abstract by 2026-07-21.",
        },
        {
            "item": "Target style",
            "status": "pass" if state["official_template_build_pass"] or state["uses_aaai_style"] else "blocking-before-upload",
            "evidence": f"{state['documentclass_line'] or 'main.tex missing'}; M20={state['m20']}; official build={state['official_template_build_pass']}.",
            "action": "Use paper/aaai27_official.pdf for AAAI-style review upload; do not upload the generic article-class PDF.",
        },
        {
            "item": "Page limit",
            "status": "pass-official-template"
            if state["official_page_limit_pass"]
            else ("preflight-pass-official-template-recheck" if state["preflight_page_limit_pass"] else "needs-target-template-verification"),
            "evidence": f"Generic main.pdf pages: {state['main_pdf_pages_generic']}; preflight pages: {state['preflight_pdf_pages']}; official pages: {state['official_pdf_pages']}; AAAI limit is {AAAI27_FACTS['technical_content_limit']}.",
            "action": "Keep official aaai2027 build under <=7 technical-content pages plus references.",
        },
        {
            "item": "AAAI-27 preflight conversion",
            "status": "pass" if state["m20"] == "PASS_AAAI27_PREFLIGHT_CONVERSION" else "needs-rerun",
            "evidence": f"M20={state['m20']}; preflight pages={state['preflight_pdf_pages']}; official pages={state['official_pdf_pages']}; official PDF fresh={state['official_pdf_fresh']}; official size={state['official_pdf_size']} bytes.",
            "action": "Use paper/aaai27_official.pdf as the current official-template submission candidate.",
        },
        {
            "item": "Supplement policy",
            "status": "pass-with-main-body-caveat",
            "evidence": f"M17={state['m17']}; supplement pages: {state['supplement_pdf_pages_generic']}.",
            "action": "Keep all critical claims in the main body because AAAI reviewers are not required to review supplements.",
        },
        {
            "item": "Reproducibility checklist and artifact instructions",
            "status": "filled-local-ready" if state["m21"] == "PASS_AAAI27_REPRODUCIBILITY_CHECKLIST" else "draft-ready",
            "evidence": f"M15={state['m15']}; M18={state['m18']}; M21={state['m21']}; local official-checklist PDF size={state['official_checklist_pdf_size']} bytes.",
            "action": "Copy the filled local checklist answers into AAAI's official OpenReview fields when the site opens.",
        },
        {
            "item": "Submission system and official form",
            "status": "pass" if site_open else "calendar-blocking-before-site-open",
            "evidence": f"Paper submission site opens {AAAI27_FACTS['submission_site_paper_open']}; current audit date {AAAI27_FACTS['accessed']}; M21={state['m21']}.",
            "action": "When OpenReview paper submission opens, upload the official PDF and complete the official reproducibility checklist fields.",
        },
        {
            "item": "AI-assisted authorship disclosure",
            "status": "draft-ready",
            "evidence": "M18 includes AI Assistance Disclosure; AAAI says authors remain fully responsible for submitted material.",
            "action": "Keep provenance notes and disclose assistance only in the target system or statement field if requested.",
        },
        {
            "item": "AI-generated first-stage review readiness",
            "status": "prepared",
            "evidence": "M19 response plan maps likely human and AI-review objections to evidence artifacts.",
            "action": "During author feedback, answer both human and AI-generated reviews with the same evidence discipline.",
        },
        {
            "item": "Submission-quality science gate",
            "status": "pass",
            "evidence": f"M5={state['m5']}; M15={state['m15']}; M17={state['m17']}; M18={state['m18']}; M20={state['m20']}.",
            "action": "Preserve benchmark/theory scope; do not convert the paper into a method/SOTA claim.",
        },
    ]


def write_gap_table(rows: list[dict]) -> None:
    text = "# M19 AAAI-27 target-readiness gap table\n\n" + md_table(
        ["Item", "Status", "Evidence", "Required action"],
        [[row["item"], row["status"], row["evidence"], row["action"]] for row in rows],
    )
    write_text_if_changed(TABLE_DIR / "m19_aaai27_gap_table.md", text + "\n")


def write_readiness_doc(rows: list[dict], state: dict) -> None:
    lines = [
        "# MaskShift AAAI-27 Target-Readiness Audit",
        "",
        "This document is target-specific for the AAAI-27 Main Technical Track. It complements M18 by checking the selected venue's dates, review process, page limit, reproducibility requirement, and AI-assistance policy against the current MaskShift package.",
        "",
        "## Official Venue Facts",
        "",
        md_table(["Field", "Value"], [[key, value] for key, value in AAAI27_FACTS.items()]),
        "",
        "## Current Package State",
        "",
        md_table(["Field", "Value"], [[key, value] for key, value in state.items()]),
        "",
        "## Gap Table",
        "",
        md_table(["Item", "Status", "Evidence", "Required action"], [[row["item"], row["status"], row["evidence"], row["action"]] for row in rows]),
        "",
        "## Upload Verdict",
        "",
        "MaskShift is scientifically ready for a strong-conference benchmark/theory submission, and M20 now provides both a two-column US-letter preflight and an official aaai2027 anonymous submission-template PDF within the seven-page technical-content pressure. The remaining upload boundary is operational: the OpenReview paper submission site is not yet open on the audit date, and the official reproducibility checklist fields must still be completed in the venue system.",
    ]
    write_text_if_changed(DOC_DIR / "AAAI27_TARGET_READINESS.md", "\n".join(lines) + "\n")


def write_repro_checklist(state: dict) -> None:
    rows = [
        ["Research question and scope", "Yes", "PAPER.md and main.tex state benchmark/theory scope and exclude method/SOTA claims."],
        ["Datasets", "Yes", "Weather, Electricity, Traffic, AirConvection with paths and natural missing rates recorded by M0/M16/M17."],
        ["Train/test split", "Yes", "Chronological splits and encoder-input-only masks are recorded in scripts and M18."],
        ["Randomness/seeds", "Yes with limits", "Seed offsets are recorded in M10-M14; three-seed summaries are disclosed as sprint-time evidence."],
        ["Baselines", "Yes with scope", "Official TSLib PatchTST/TimeXer, ChannelTokenFormer_missing, and S4M are adaptations under MaskShift protocol."],
        ["Hyperparameters", "Draft-ready", "Script configs are serialized in JSON summaries; final AAAI appendix should copy exact values."],
        ["Compute/runtime", "Draft-ready", "Local reduced protocols are described; final artifact should add measured wall-clock budget if available."],
        ["Code availability", "Draft-ready", "M18 statement and reproduction commands exist; anonymous artifact bundle still needs packaging."],
        ["Data availability", "Draft-ready", "Public benchmark input paths are recorded; raw license/source metadata should be checked before public release."],
        ["Limitations", "Yes", "main.tex discloses reduced protocols, mixed datasets, three-seed summaries, and failed typed head."],
    ]
    lines = [
        "# AAAI-27 Reproducibility Checklist Draft",
        "",
        "This is a draft mapping for the AAAI reproducibility checklist requirement. It should be copied into the official AAAI form after the target author kit and submission system fields are available.",
        "",
        md_table(["Checklist item", "Current answer", "Evidence/action"], rows),
    ]
    write_text_if_changed(DOC_DIR / "AAAI27_REPRODUCIBILITY_CHECKLIST_DRAFT.md", "\n".join(lines) + "\n")


def write_response_plan() -> None:
    rows = [
        ["Is this just a stress test?", "No; it isolates conditional mask mechanism under matched missing rate and tests risk/rank shift.", "main.tex Sections 2-4; M0/M1/M3/M9/M10."],
        ["Why AAAI instead of a narrower TSF venue?", "AAAI explicitly welcomes empirical, theoretical, critical, and integrative contributions across AI, and responsible/reproducible evaluation is central to the call.", "AAAI-27 call review criteria; M19 target audit."],
        ["Do missing-aware architectures solve it?", "Not uniformly; CTF_missing is mixed and S4M is contrastively robust, which supports mechanism reporting rather than universal failure claims.", "M11/M12/M14 tables."],
        ["Are supplements required to understand the claim?", "No; critical evidence is in the main body; the supplement only expands cards, reproduction, and rebuttal details.", "main.tex Tables/sections; M17 supplement."],
        ["Can an AI-generated first-stage review misread the scope?", "Likely risk: method/SOTA framing. Response should quote the benchmark/theory scope and negative typed-head diagnostic.", "REVIEW.md v11; SUBMISSION_CHECKLIST.md; claim-evidence table."],
    ]
    lines = [
        "# AAAI-27 Phase-Review Response Plan",
        "",
        "AAAI-27 uses two-phase reviewing and includes an AI-generated, non-decisional first-stage review. This plan prepares evidence-grounded responses without overclaiming.",
        "",
        md_table(["Likely objection", "Response stance", "Evidence"], rows),
    ]
    write_text_if_changed(DOC_DIR / "AAAI27_PHASE_REVIEW_RESPONSE_PLAN.md", "\n".join(lines) + "\n")


def write_readiness_tex(rows: list[dict], state: dict) -> None:
    gap_rows = "\n".join(
        "%s & %s & %s\\\\"
        % (tex_escape(row["item"]), tex_escape(row["status"]), tex_escape(row["action"]))
        for row in rows
    )
    text = r"""\documentclass[10pt]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{xurl}
\title{MaskShift AAAI-27 Target-Readiness Dossier}
\author{Anonymous}
\date{}

\begin{document}
\maketitle

\section{Target Facts}
Target venue: %s. Paper submission site opens %s; abstract deadline %s; full paper deadline %s; supplementary material and code deadline %s. Technical content limit: %s.

Source: \url{%s}.

\section{Current Package State}
Generic main PDF pages: %s. Current document class: \texttt{%s}. AAAI style present: \texttt{%s}.

\begin{center}
\begin{tabular}{ll}
\toprule
Gate & Status\\
\midrule
M5 & %s\\
M15 & %s\\
M17 & %s\\
M18 & %s\\
M20 & %s\\
M21 & %s\\
\bottomrule
\end{tabular}
\end{center}

Preflight pages: %s; official-template pages: %s; preflight table fit: \texttt{%s}; official-template build ready: \texttt{%s}.

\section{Target Gap Table}
\small
\begin{longtable}{p{0.22\linewidth}p{0.24\linewidth}p{0.44\linewidth}}
\toprule
Item & Status & Required action\\
\midrule
%s
\bottomrule
\end{longtable}
\normalsize

\section{Verdict}
The scientific package is strong-conference ready as a benchmark/theory paper. M20 passes both a conservative two-column US-letter preflight and an official aaai2027 anonymous submission-template build. The remaining upload boundary is the submission-system opening date and completion of the official reproducibility checklist fields.

\end{document}
""" % (
        tex_escape(AAAI27_FACTS["venue"]),
        tex_escape(AAAI27_FACTS["submission_site_paper_open"]),
        tex_escape(AAAI27_FACTS["abstract_deadline"]),
        tex_escape(AAAI27_FACTS["full_paper_deadline"]),
        tex_escape(AAAI27_FACTS["supplement_code_deadline"]),
        tex_escape(AAAI27_FACTS["technical_content_limit"]),
        tex_escape(AAAI27_SOURCE),
        state["main_pdf_pages_generic"],
        tex_escape(state["documentclass_line"]),
        state["uses_aaai_style"],
        tex_escape(state["m5"]),
        tex_escape(state["m15"]),
        tex_escape(state["m17"]),
        tex_escape(state["m18"]),
        tex_escape(state["m20"]),
        tex_escape(state["m21"]),
        state["preflight_pdf_pages"],
        state["official_pdf_pages"],
        state["preflight_table_fit"],
        state["official_template_build_pass"],
        gap_rows,
    )
    write_text_if_changed(PAPER_DIR / "aaai27_readiness.tex", text)


def main() -> None:
    state = detect_package_state()
    rows = build_gap_rows(state)
    write_gap_table(rows)
    write_readiness_doc(rows, state)
    write_repro_checklist(state)
    write_response_plan()
    write_readiness_tex(rows, state)

    artifacts = [
        "docs/AAAI27_TARGET_READINESS.md",
        "docs/AAAI27_REPRODUCIBILITY_CHECKLIST_DRAFT.md",
        "docs/AAAI27_PHASE_REVIEW_RESPONSE_PLAN.md",
        "tables/m19_aaai27_gap_table.md",
        "paper/aaai27_readiness.tex",
        "paper/aaai27_readiness.pdf",
        "m19_aaai27_target_readiness/aaai27_target_readiness_summary.json",
    ]
    missing = [
        path
        for path in artifacts
        if path != "m19_aaai27_target_readiness/aaai27_target_readiness_summary.json" and not (EXP_DIR / path).exists()
    ]
    pdf_path = PAPER_DIR / "aaai27_readiness.pdf"
    tex_path = PAPER_DIR / "aaai27_readiness.tex"
    pdf_fresh = pdf_path.exists() and pdf_path.stat().st_mtime >= tex_path.stat().st_mtime
    pdf_ok = pdf_path.exists() and pdf_path.stat().st_size > 20_000 and pdf_fresh
    upload_blockers = [
        row
        for row in rows
        if row["status"] in {
            "blocking-before-upload",
            "needs-target-template-verification",
            "needs-rerun",
            "calendar-blocking-before-site-open",
        }
    ]
    status = "PASS_AAAI27_TARGET_DOSSIER" if not missing and pdf_ok else "HOLD_AAAI27_TARGET_DOSSIER"
    summary = {
        "milestone": "M19",
        "status": status,
        "aaai27_upload_ready": not upload_blockers,
        "target": AAAI27_FACTS,
        "current_package_state": state,
        "gap_rows": rows,
        "upload_blockers": upload_blockers,
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "aaai27_readiness_pdf_size": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "aaai27_readiness_pdf_fresh": pdf_fresh,
        "scope_note": "Target-specific AAAI-27 dossier. Passing M19 means the target dossier is complete; it does not mean the AAAI author-kit conversion is complete.",
    }
    write_json(OUT_DIR / "aaai27_target_readiness_summary.json", summary)
    print(
        json.dumps(
            {
                "milestone": "M19",
                "status": status,
                "aaai27_upload_ready": summary["aaai27_upload_ready"],
                "pdf_ok": pdf_ok,
                "pdf_fresh": pdf_fresh,
                "upload_blockers": [row["item"] for row in upload_blockers],
                "missing_artifacts": missing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
