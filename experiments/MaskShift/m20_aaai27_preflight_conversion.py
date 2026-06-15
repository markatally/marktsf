"""M20 -- AAAI-27 two-column preflight conversion.

The official AAAI-27 page says the author kit exists, but the local workspace
does not yet contain the official aaai2027.sty/aaai2027.bst files. This
milestone therefore performs a conservative preflight conversion: two-column,
US-letter, anonymous LaTeX with existing content and bibliography, then records
page-count evidence. It is not a substitute for the official author kit.
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

from .maskshift_core import ensure_dir, write_json


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m20_aaai27_preflight_conversion")
DOC_DIR = ensure_dir(EXP_DIR / "docs")
TABLE_DIR = ensure_dir(EXP_DIR / "tables")
PAPER_DIR = ensure_dir(EXP_DIR / "paper")


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def write_text_if_changed(path: Path, text: str) -> None:
    if path.exists() and path.read_text() == text:
        return
    path.write_text(text)


def pdf_page_count(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        import fitz  # type: ignore

        with fitz.open(path) as doc:
            return int(doc.page_count)
    except Exception:
        return None


def extract_body(main_tex: str) -> str:
    match = re.search(r"\\begin\{document\}(.*)\\end\{document\}", main_tex, re.S)
    if not match:
        raise ValueError("Cannot find document body in paper/main.tex")
    body = match.group(1).strip()
    body = body.replace(r"\maketitle", "").strip()
    return body


def adapt_body_for_twocolumn_preflight(body: str) -> str:
    """Let manuscript result tables use the full text width in preflight mode."""
    body = body.replace(r"\begin{table}[t]", r"\begin{table*}[t]")
    body = body.replace(r"\end{table}", r"\end{table*}")
    return body


def ensure_official_style_files() -> dict[str, bool]:
    kit_dir = PAPER_DIR / "AuthorKit27"
    zip_path = PAPER_DIR / "AuthorKit27.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            for member in [
                "AuthorKit27/aaai2027.sty",
                "AuthorKit27/aaai2027.bst",
                "AuthorKit27/AnonymousSubmission2027.tex",
                "AuthorKit27/ReproducibilityChecklist.tex",
            ]:
                target = PAPER_DIR / member
                if not target.exists():
                    archive.extract(member, PAPER_DIR)
    copied = {}
    for name in ["aaai2027.sty", "aaai2027.bst"]:
        source = kit_dir / name
        target = PAPER_DIR / name
        if source.exists() and (not target.exists() or target.read_bytes() != source.read_bytes()):
            shutil.copyfile(source, target)
            copied[name] = True
        else:
            copied[name] = False
    return copied


def build_preflight_tex() -> str:
    main_tex = (PAPER_DIR / "main.tex").read_text()
    body = adapt_body_for_twocolumn_preflight(extract_body(main_tex))
    # The preflight keeps existing figure/table content, but uses a compact
    # two-column layout approximating AAAI's page pressure. Official kit
    # conversion should replace this wrapper, not the scientific body.
    return (
        r"""\documentclass[letterpaper,twocolumn,10pt]{article}
\usepackage{times}
\usepackage{helvet}
\usepackage{courier}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{amsmath,amssymb}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage[letterpaper,margin=0.75in]{geometry}
\frenchspacing
\setlength{\pdfpagewidth}{8.5in}
\setlength{\pdfpageheight}{11in}
\setcounter{secnumdepth}{0}
\setlength{\textfloatsep}{7pt plus 1pt minus 2pt}
\setlength{\floatsep}{6pt plus 1pt minus 2pt}
\setlength{\intextsep}{6pt plus 1pt minus 2pt}
\setlength{\abovecaptionskip}{3pt}
\setlength{\belowcaptionskip}{0pt}
\title{MaskShift: Forecasting Under Missingness-Mechanism Shift}
\author{Anonymous Submission}
\date{}

\begin{document}
\maketitle

"""
        + body
        + "\n\\end{document}\n"
    )


def build_official_tex() -> str:
    main_tex = (PAPER_DIR / "main.tex").read_text()
    body = adapt_body_for_twocolumn_preflight(extract_body(main_tex))
    body = body.replace("\\bibliographystyle{plain}\n", "")
    body = body.replace("\\bibliographystyle{plain}", "")
    return (
        r"""\documentclass[letterpaper]{article}
\usepackage[submission]{aaai2027}
\usepackage[hyphens]{url}
\usepackage{graphicx}
\urlstyle{rm}
\def\UrlFont{\rm}
\usepackage{natbib}
\usepackage{caption}
\frenchspacing
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\pdfinfo{
/TemplateVersion (2027.1)
}
\setcounter{secnumdepth}{0}
\title{MaskShift: Forecasting Under Missingness-Mechanism Shift}
\author{Anonymous Submission}
\affiliations{}

\begin{document}
\maketitle

"""
        + body
        + "\n\\end{document}\n"
    )


def detect_state() -> dict:
    tex_path = PAPER_DIR / "aaai27_preflight.tex"
    pdf_path = PAPER_DIR / "aaai27_preflight.pdf"
    official_tex_path = PAPER_DIR / "aaai27_official.tex"
    official_pdf_path = PAPER_DIR / "aaai27_official.pdf"
    tex = tex_path.read_text() if tex_path.exists() else ""
    official_tex = official_tex_path.read_text() if official_tex_path.exists() else ""
    official_files = {
        "aaai2027.sty": (PAPER_DIR / "aaai2027.sty").exists(),
        "aaai2027.bst": (PAPER_DIR / "aaai2027.bst").exists(),
        "aaai.sty": (PAPER_DIR / "aaai.sty").exists(),
    }
    pdf_fresh = pdf_path.exists() and tex_path.exists() and pdf_path.stat().st_mtime >= tex_path.stat().st_mtime
    official_pdf_fresh = (
        official_pdf_path.exists()
        and official_tex_path.exists()
        and official_pdf_path.stat().st_mtime >= official_tex_path.stat().st_mtime
    )
    official_page_count = pdf_page_count(official_pdf_path)
    return {
        "preflight_tex_exists": tex_path.exists(),
        "preflight_pdf_exists": pdf_path.exists(),
        "preflight_pdf_fresh": pdf_fresh,
        "preflight_pdf_size": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "preflight_page_count": pdf_page_count(pdf_path),
        "official_tex_exists": official_tex_path.exists(),
        "official_pdf_exists": official_pdf_path.exists(),
        "official_pdf_fresh": official_pdf_fresh,
        "official_pdf_size": official_pdf_path.stat().st_size if official_pdf_path.exists() else 0,
        "official_page_count": official_page_count,
        "official_page_limit_pass": official_page_count is not None and official_page_count <= 7,
        "official_uses_aaai2027_submission": r"\usepackage[submission]{aaai2027}" in official_tex,
        "official_uses_aaai_bst": r"\bibliographystyle{plain}" not in official_tex and r"\bibliography{references}" in official_tex,
        "uses_twocolumn": "twocolumn" in tex,
        "uses_letterpaper": "letterpaper" in tex,
        "uses_double_column_tables": r"\begin{table*}" in tex and r"\end{table*}" in tex,
        "official_uses_double_column_tables": r"\begin{table*}" in official_tex and r"\end{table*}" in official_tex,
        "anonymous_submission": "Anonymous Submission" in tex and "Anonymous Authors" not in tex,
        "official_anonymous_submission": "Anonymous Submission" in official_tex and "Anonymous Authors" not in official_tex,
        "official_style_files_present": official_files,
        "official_aaai_author_kit_ready": official_files["aaai2027.sty"] and official_files["aaai2027.bst"],
        "author_kit_zip_present": (PAPER_DIR / "AuthorKit27.zip").exists(),
    }


def write_preflight_doc(state: dict) -> None:
    rows = [
        ["Two-column preflight", state["uses_twocolumn"], "Generated from paper/main.tex into paper/aaai27_preflight.tex."],
        ["US Letter preflight", state["uses_letterpaper"], "Preflight wrapper uses letterpaper."],
        ["Two-column table fit", state["uses_double_column_tables"], "Result tables are promoted to table* floats to avoid single-column overflow."],
        ["Anonymous title block", state["anonymous_submission"], "Author line is Anonymous Submission."],
        ["Page-count preflight", state["preflight_page_count"], "Must be <=7 pages under the preflight wrapper."],
        ["Official style files", state["official_style_files_present"], "Official aaai2027.sty/aaai2027.bst are present when both required files are true."],
        ["Official AAAI build", state["official_pdf_exists"], f"paper/aaai27_official.pdf pages={state['official_page_count']}, fresh={state['official_pdf_fresh']}."],
    ]
    lines = [
        "# AAAI-27 Preflight Conversion Audit",
        "",
        "M20 converts the current generic article manuscript into both a two-column, US-letter, anonymous preflight draft and an official `aaai2027` anonymous submission-template build from the official AAAI author kit. The preflight remains a page-pressure audit; `paper/aaai27_official.pdf` is the current official-template submission candidate.",
        "",
        md_table(["Check", "Value", "Evidence"], rows),
        "",
        "## Upload Boundary",
        "",
        "Passing M20 means the manuscript body fits a conservative AAAI-style preflight and the official `aaai2027` anonymous submission-template build passes. The remaining upload boundary is the OpenReview submission workflow and official reproducibility checklist fields.",
    ]
    write_text_if_changed(DOC_DIR / "AAAI27_PREFLIGHT_FORMAT_AUDIT.md", "\n".join(lines) + "\n")


def write_preflight_table(state: dict) -> None:
    rows = [
        ["Preflight PDF", state["preflight_pdf_exists"], f"{state['preflight_pdf_size']} bytes"],
        ["PDF fresh", state["preflight_pdf_fresh"], "PDF timestamp is newer than TeX"],
        ["Page count", state["preflight_page_count"], "<=7 required for preflight pass"],
        ["Two-column", state["uses_twocolumn"], "twocolumn option in generated TeX"],
        ["Table fit", state["uses_double_column_tables"], "Wide result tables use table* in the two-column preflight"],
        ["Official kit ready", state["official_aaai_author_kit_ready"], "Requires aaai2027.sty and aaai2027.bst"],
        ["Official PDF", state["official_pdf_exists"], f"{state['official_pdf_size']} bytes"],
        ["Official page count", state["official_page_count"], "<=7 required under official template"],
    ]
    text = "# M20 AAAI-27 preflight format table\n\n" + md_table(["Item", "Value", "Evidence"], rows) + "\n"
    write_text_if_changed(TABLE_DIR / "m20_aaai27_preflight_table.md", text)


def main() -> None:
    ensure_official_style_files()
    tex = build_preflight_tex()
    write_text_if_changed(PAPER_DIR / "aaai27_preflight.tex", tex)
    official_tex = build_official_tex()
    write_text_if_changed(PAPER_DIR / "aaai27_official.tex", official_tex)
    state = detect_state()
    write_preflight_doc(state)
    write_preflight_table(state)
    artifacts = [
        "docs/AAAI27_PREFLIGHT_FORMAT_AUDIT.md",
        "tables/m20_aaai27_preflight_table.md",
        "paper/aaai27_preflight.tex",
        "paper/aaai27_preflight.pdf",
        "paper/aaai27_official.tex",
        "paper/aaai27_official.pdf",
        "paper/aaai2027.sty",
        "paper/aaai2027.bst",
        "paper/AuthorKit27.zip",
        "m20_aaai27_preflight_conversion/aaai27_preflight_summary.json",
    ]
    missing = [
        path
        for path in artifacts
        if path != "m20_aaai27_preflight_conversion/aaai27_preflight_summary.json" and not (EXP_DIR / path).exists()
    ]
    page_ok = state["preflight_page_count"] is not None and state["preflight_page_count"] <= 7
    preflight_ok = (
        not missing
        and state["preflight_pdf_fresh"]
        and state["preflight_pdf_size"] > 50_000
        and page_ok
        and state["uses_twocolumn"]
        and state["uses_letterpaper"]
        and state["uses_double_column_tables"]
        and state["anonymous_submission"]
    )
    official_ok = (
        not missing
        and state["official_aaai_author_kit_ready"]
        and state["official_pdf_fresh"]
        and state["official_pdf_size"] > 50_000
        and state["official_page_limit_pass"]
        and state["official_uses_aaai2027_submission"]
        and state["official_uses_aaai_bst"]
        and state["official_uses_double_column_tables"]
        and state["official_anonymous_submission"]
    )
    summary = {
        "milestone": "M20",
        "status": "PASS_AAAI27_PREFLIGHT_CONVERSION" if preflight_ok and official_ok else "HOLD_AAAI27_PREFLIGHT_CONVERSION",
        "state": state,
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "page_limit_preflight_pass": page_ok,
        "official_template_build_pass": official_ok,
        "aaai_official_kit_upload_ready": official_ok,
        "scope_note": "AAAI-style preflight plus official aaai2027 submission-template build. Submission-system upload still requires the venue form/checklist workflow.",
    }
    write_json(OUT_DIR / "aaai27_preflight_summary.json", summary)
    print(
        json.dumps(
            {
                "milestone": "M20",
                "status": summary["status"],
                "pages": state["preflight_page_count"],
                "pdf_fresh": state["preflight_pdf_fresh"],
                "official_kit_ready": state["official_aaai_author_kit_ready"],
                "official_template_build_pass": official_ok,
                "official_pages": state["official_page_count"],
                "missing_artifacts": missing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
