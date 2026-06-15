"""M21 -- Fill the AAAI-27 reproducibility checklist locally.

The official submission system fields still have to be completed when the site
opens. This milestone prepares a source-of-truth local checklist from the
AAAI-27 author kit so those fields can be copied without improvising.
"""

from __future__ import annotations

import json
from pathlib import Path

from .maskshift_core import ensure_dir, write_json


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m21_aaai27_reproducibility_checklist")
DOC_DIR = ensure_dir(EXP_DIR / "docs")
PAPER_DIR = ensure_dir(EXP_DIR / "paper")


ANSWERS = [
    "NA -- MaskShift does not introduce a new AI method; the benchmark protocol and mask generators are described in the main paper and supplement.",
    "yes -- The paper separates benchmark/theory claims, empirical evidence, negative diagnostics, and limitations.",
    "yes -- Related work and dataset/mechanism cards identify the missing-data forecasting and TSF context needed to replicate the study.",
    "yes -- The paper includes a theoretical risk-identity note and rank-reversal condition.",
    "yes -- The matched-rate mask-shift setting, clean-target protocol, and benchmark scope are stated explicitly.",
    "partial -- The formal risk identity and rank-reversal condition are stated; empirical claims are tied to audited tables rather than theorem statements.",
    "partial -- The risk identity is elementary and explained; full formal proof expansion is not needed for the benchmark contribution.",
    "yes -- The paper gives intuition for mechanism-specific Bayes predictors and model-dependent risk shifts.",
    "yes -- Missing-data and forecasting references are cited in the paper and audit materials.",
    "yes -- The theoretical warning is tested empirically through matched-rate mechanism-shift experiments.",
    "yes -- Negative and contrastive experiments are included, including typed-head failure and S4M robustness.",
    "yes -- The study uses Weather, Electricity, Traffic, and AirConvection.",
    "yes -- Weather/Electricity provide positive evidence; Traffic/AirConvection are retained as mixed boundary datasets.",
    "NA -- No new dataset is introduced.",
    "NA -- No new dataset is introduced.",
    "partial -- The datasets are standard public TSF benchmarks; final public-release notes should preserve upstream source/license metadata.",
    "yes -- The datasets are public benchmark datasets used through local benchmark paths.",
    "NA -- No non-public dataset is used.",
    "yes -- The paper is primarily computational.",
    "partial -- Script configs and JSON summaries record the reduced protocols and seed offsets; the final appendix/checklist should carry the exact values.",
    "yes -- Mask generation, preprocessing, and evaluation scripts are included in the MaskShift package.",
    "yes -- Training, evaluation, statistical analysis, and table-generation scripts are included.",
    "yes -- The code can be released with the anonymous artifact bundle and public repository on publication.",
    "NA -- No new forecasting backbone is introduced; benchmark code is documented where non-obvious.",
    "yes -- Seed offsets and multi-seed summaries are recorded in M10-M14 and the paper.",
    "partial -- Software/runtime constraints and local reduced protocols are recorded; final submission fields should add exact hardware inventory if requested.",
    "yes -- MSE, MAE, sMAPE, eta-squared, Kendall tau, corrected severity metrics, and uncertainty summaries are described.",
    "yes -- The paper reports the number of seed offsets and gate-seed counts for each core result.",
    "yes -- The analysis includes CIs, hierarchical bootstrap summaries, FDR/statistical tests, and mixed/negative boundary cases.",
    "yes -- The package uses BH-FDR, ANOVA/Kruskal checks, and bootstrap uncertainty where appropriate.",
    "partial -- Final model/protocol values are serialized in JSON summaries and scripts; the official form can cite those artifacts.",
]


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


def build_filled_checklist() -> tuple[str, int]:
    source_path = PAPER_DIR / "AuthorKit27" / "ReproducibilityChecklist.tex"
    source = source_path.read_text()
    marker = "% The questions start here"
    before, after = source.split(marker, 1)
    for answer in ANSWERS:
        after = after.replace("Type your response here", answer, 1)
    remaining = after.count("Type your response here")
    return before + marker + after, remaining


def main() -> None:
    filled, remaining = build_filled_checklist()
    tex_path = PAPER_DIR / "aaai27_reproducibility_checklist.tex"
    pdf_path = PAPER_DIR / "aaai27_reproducibility_checklist.pdf"
    write_text_if_changed(tex_path, filled)
    rows = [
        ["Answer slots", len(ANSWERS), "Filled after the official checklist question marker."],
        ["Remaining placeholders", remaining, "Must be 0 in the question section."],
        ["Checklist PDF", pdf_path.exists(), f"{pdf_path.stat().st_size if pdf_path.exists() else 0} bytes"],
    ]
    write_text_if_changed(
        DOC_DIR / "AAAI27_REPRODUCIBILITY_CHECKLIST_FILLED.md",
        "# AAAI-27 Filled Reproducibility Checklist\n\n"
        + md_table(["Item", "Value", "Evidence"], rows)
        + "\n\nThis local checklist is prepared from the official AAAI-27 author kit. The submission-system fields still need to be completed when OpenReview opens.\n",
    )
    artifacts = [
        "docs/AAAI27_REPRODUCIBILITY_CHECKLIST_FILLED.md",
        "paper/aaai27_reproducibility_checklist.tex",
        "paper/aaai27_reproducibility_checklist.pdf",
        "m21_aaai27_reproducibility_checklist/aaai27_reproducibility_checklist_summary.json",
    ]
    missing = [
        path
        for path in artifacts
        if path != "m21_aaai27_reproducibility_checklist/aaai27_reproducibility_checklist_summary.json"
        and not (EXP_DIR / path).exists()
    ]
    pdf_fresh = pdf_path.exists() and pdf_path.stat().st_mtime >= tex_path.stat().st_mtime
    pdf_ok = pdf_path.exists() and pdf_path.stat().st_size > 20_000 and pdf_fresh
    summary = {
        "milestone": "M21",
        "status": "PASS_AAAI27_REPRODUCIBILITY_CHECKLIST" if not missing and remaining == 0 and pdf_ok else "HOLD_AAAI27_REPRODUCIBILITY_CHECKLIST",
        "answer_count": len(ANSWERS),
        "remaining_question_placeholders": remaining,
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "pdf_size": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "pdf_fresh": pdf_fresh,
        "scope_note": "Local filled checklist from official AAAI-27 kit; OpenReview fields still require manual submission-system completion.",
    }
    write_json(OUT_DIR / "aaai27_reproducibility_checklist_summary.json", summary)
    print(json.dumps({"milestone": "M21", "status": summary["status"], "remaining": remaining, "pdf_ok": pdf_ok}, indent=2))


if __name__ == "__main__":
    main()
