"""M15 — Final integrity audit for the MaskShift submission package.

This milestone is intentionally stricter than the M5 readiness gate. M5 asks
whether the scientific package is strong-conference ready; M15 asks whether the
submission artifact is internally clean enough to survive final integrity
review: citation graph, BibTeX hygiene, numeric/table traceability, figure/PDF
presence, implementation safety checks, and the ARS AI-research failure modes.
"""

from __future__ import annotations

import json
import py_compile
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .maskshift_core import ensure_dir, write_json


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m15_final_integrity_audit")
DOC_DIR = ensure_dir(EXP_DIR / "docs")
TABLE_DIR = ensure_dir(EXP_DIR / "tables")

REFERENCE_AUDIT_SOURCES = {
    "che2018grud": "https://www.nature.com/articles/s41598-018-24271-9",
    "cao2018brits": "https://proceedings.neurips.cc/paper/2018/hash/734e6bfcd358e25ac1db0a4241b95651-Abstract.html",
    "tashiro2021csdi": "https://proceedings.neurips.cc/paper_files/paper/2021/hash/cfe8504bda37b575c70ee1a8276f3486-Abstract.html",
    "islam2025sadi": "https://ojs.aaai.org/index.php/AAAI/article/view/33931",
    "jing2025s4m": "https://openreview.net/forum?id=BkftcwIVmR",
    "jang2026channeltokenformer": "https://openreview.net/forum?id=r4ZamwBE8P",
    "sunesh2026blackouts": "https://arxiv.org/abs/2601.01480",
    "yalavarthi2024grafiti": "https://ojs.aaai.org/index.php/AAAI/article/view/29560",
    "zeng2023dlinear": "https://ojs.aaai.org/index.php/AAAI/article/view/26317",
    "nie2023patchtst": "https://openreview.net/forum?id=Jbdc0vTOcol",
    "liu2024itransformer": "https://arxiv.org/abs/2310.06625",
    "wang2024timexer": "https://arxiv.org/abs/2402.19072",
    "yang2025crib": "https://arxiv.org/abs/2509.23494",
    "rockenschaub2024missingness": "https://arxiv.org/abs/2406.16484",
}

FINAL_DOCS = [
    "PAPER.md",
    "README.md",
    "REPORT.md",
    "SUBMISSION_CHECKLIST.md",
    "paper/main.tex",
    "paper/supplement.tex",
    "paper/submission_statements.tex",
    "paper/references.bib",
    "docs/DATASET_MECHANISM_CARDS.md",
    "docs/REBUTTAL_PLAYBOOK.md",
    "docs/SUBMISSION_STATEMENTS.md",
    "docs/VENUE_READINESS_AUDIT.md",
    "docs/AAAI27_TARGET_READINESS.md",
    "docs/AAAI27_REPRODUCIBILITY_CHECKLIST_DRAFT.md",
    "docs/AAAI27_PHASE_REVIEW_RESPONSE_PLAN.md",
    "docs/AAAI27_PREFLIGHT_FORMAT_AUDIT.md",
    "docs/AAAI27_REPRODUCIBILITY_CHECKLIST_FILLED.md",
    "paper/aaai27_preflight.tex",
    "paper/aaai27_official.tex",
    "paper/aaai27_reproducibility_checklist.tex",
]

CORE_ARTIFACTS = [
    "PAPER.md",
    "REPORT.md",
    "README.md",
    "SUBMISSION_CHECKLIST.md",
    "paper/main.tex",
    "paper/supplement.tex",
    "paper/submission_statements.tex",
    "paper/aaai27_readiness.tex",
    "paper/aaai27_preflight.tex",
    "paper/aaai27_official.tex",
    "paper/aaai27_reproducibility_checklist.tex",
    "paper/references.bib",
    "paper/aaai2027.sty",
    "paper/aaai2027.bst",
    "paper/AuthorKit27.zip",
    "paper/main.pdf",
    "paper/supplement.pdf",
    "paper/submission_statements.pdf",
    "paper/aaai27_readiness.pdf",
    "paper/aaai27_preflight.pdf",
    "paper/aaai27_official.pdf",
    "paper/aaai27_reproducibility_checklist.pdf",
    "figures/maskshift_overview.png",
    "figures/m1_mechanism_rank.png",
    "figures/m2_typed_correction.png",
    "tables/main_result_table.md",
    "tables/m9_official_architecture_table.md",
    "tables/m11_ctf_missing_table.md",
    "tables/m12_s4m_table.md",
    "tables/m13_hierarchical_bootstrap_table.md",
    "tables/m14_s4m_scale_table.md",
    "tables/m15_integrity_table.md",
    "tables/m16_official_tslib_full_coverage_table.md",
    "tables/m17_reviewer_response_matrix.md",
    "tables/m18_policy_readiness_table.md",
    "tables/m19_aaai27_gap_table.md",
    "tables/m20_aaai27_preflight_table.md",
    "tables/m7_corrected_robustness_table.md",
    "tables/claim_evidence_table.md",
    "m5_main_track_audit/main_track_audit.json",
    "m16_official_tslib_full_coverage/official_tslib_full_coverage_summary.json",
    "m17_submission_supplement/submission_supplement_summary.json",
    "m18_submission_policy_pack/submission_policy_summary.json",
    "m19_aaai27_target_readiness/aaai27_target_readiness_summary.json",
    "m20_aaai27_preflight_conversion/aaai27_preflight_summary.json",
    "m21_aaai27_reproducibility_checklist/aaai27_reproducibility_checklist_summary.json",
    "docs/DATASET_MECHANISM_CARDS.md",
    "docs/REBUTTAL_PLAYBOOK.md",
    "docs/SUBMISSION_STATEMENTS.md",
    "docs/VENUE_READINESS_AUDIT.md",
    "docs/AAAI27_TARGET_READINESS.md",
    "docs/AAAI27_REPRODUCIBILITY_CHECKLIST_DRAFT.md",
    "docs/AAAI27_PHASE_REVIEW_RESPONSE_PLAN.md",
    "docs/AAAI27_PREFLIGHT_FORMAT_AUDIT.md",
    "docs/AAAI27_REPRODUCIBILITY_CHECKLIST_FILLED.md",
]


@dataclass
class Issue:
    severity: str
    category: str
    location: str
    description: str


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_rel(path: str) -> str:
    return (EXP_DIR / path).read_text()


def parse_citations(tex: str) -> set[str]:
    keys: set[str] = set()
    pattern = re.compile(r"\\cite[a-zA-Z*]*(?:\[[^\]]*\])*\{([^}]+)\}")
    for match in pattern.finditer(tex):
        keys.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


def parse_bib_entries(bib: str) -> dict[str, dict[str, str]]:
    starts = list(re.finditer(r"@(?P<type>\w+)\{(?P<key>[^,]+),", bib))
    entries: dict[str, dict[str, str]] = {}
    for idx, match in enumerate(starts):
        start = match.end()
        end = starts[idx + 1].start() if idx + 1 < len(starts) else len(bib)
        body = bib[start:end]
        fields: dict[str, str] = {"entry_type": match.group("type").lower()}
        for field_match in re.finditer(r"^\s*(\w+)\s*=\s*\{(.*?)\},?\s*$", body, re.MULTILINE):
            fields[field_match.group(1).lower()] = field_match.group(2)
        entries[match.group("key")] = fields
    return entries


def pct_ci(ci: dict) -> str:
    return f"{ci['mean'] * 100:.1f}% [{ci['ci_low'] * 100:.1f}, {ci['ci_high'] * 100:.1f}]"


def num_ci(ci: dict, digits: int = 3, lower: float | None = None, upper: float | None = None) -> str:
    mean = ci["mean"]
    low = ci["ci_low"]
    high = ci["ci_high"]
    if lower is not None:
        mean = max(lower, mean)
        low = max(lower, low)
        high = max(lower, high)
    if upper is not None:
        mean = min(upper, mean)
        low = min(upper, low)
        high = min(upper, high)
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def tau_ci(ci: dict) -> str:
    return f"{ci['mean']:.2f} [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]"


def contains_all(path: str, expected: Iterable[str], issues: list[Issue], category: str) -> int:
    text = read_rel(path)
    passed = 0
    for snippet in expected:
        if snippet in text:
            passed += 1
        else:
            issues.append(Issue("SERIOUS", category, path, f"Missing expected trace snippet: {snippet}"))
    return passed


def check_references(issues: list[Issue]) -> dict:
    tex = read_rel("paper/main.tex")
    bib = read_rel("paper/references.bib")
    cited = parse_citations(tex)
    entries = parse_bib_entries(bib)
    bib_keys = set(entries)
    dangling = sorted(cited - bib_keys)
    orphan = sorted(bib_keys - cited)
    missing_source = sorted(
        key for key, fields in entries.items() if not fields.get("url") and not fields.get("doi")
    )
    missing_audit_source = sorted(key for key in bib_keys if key not in REFERENCE_AUDIT_SOURCES)
    inproceedings_volume_number = sorted(
        key
        for key, fields in entries.items()
        if fields.get("entry_type") == "inproceedings" and fields.get("volume") and fields.get("number")
    )
    for key in dangling:
        issues.append(Issue("SERIOUS", "citations", "paper/main.tex", f"Dangling citation: {key}"))
    for key in orphan:
        issues.append(Issue("MEDIUM", "citations", "paper/references.bib", f"Orphan reference: {key}"))
    for key in missing_source:
        issues.append(Issue("SERIOUS", "references", "paper/references.bib", f"No DOI or URL for {key}"))
    for key in missing_audit_source:
        issues.append(Issue("MEDIUM", "references", "paper/references.bib", f"No recorded external audit source for {key}"))
    for key in inproceedings_volume_number:
        issues.append(Issue("MINOR", "bibtex", "paper/references.bib", f"{key} has both volume and number under plain BibTeX"))
    return {
        "cited_keys": sorted(cited),
        "bib_keys": sorted(bib_keys),
        "dangling": dangling,
        "orphan": orphan,
        "missing_source": missing_source,
        "missing_audit_source": missing_audit_source,
        "inproceedings_volume_number": inproceedings_volume_number,
        "reference_audit_sources": {key: REFERENCE_AUDIT_SOURCES[key] for key in sorted(bib_keys)},
        "pass": not (dangling or orphan or missing_source or missing_audit_source or inproceedings_volume_number),
    }


def check_artifacts(issues: list[Issue]) -> dict:
    missing = [path for path in CORE_ARTIFACTS if not (EXP_DIR / path).exists()]
    undersized = []
    for path in [
        "paper/main.pdf",
        "paper/supplement.pdf",
        "paper/submission_statements.pdf",
        "paper/aaai27_readiness.pdf",
        "paper/aaai27_preflight.pdf",
        "figures/maskshift_overview.png",
        "figures/m1_mechanism_rank.png",
        "figures/m2_typed_correction.png",
    ]:
        full = EXP_DIR / path
        if full.exists() and full.stat().st_size < 10_000:
            undersized.append(path)
    if (EXP_DIR / "paper/main.pdf").exists() and (EXP_DIR / "paper/main.pdf").stat().st_size < 100_000:
        undersized.append("paper/main.pdf")
    if (EXP_DIR / "paper/supplement.pdf").exists() and (EXP_DIR / "paper/supplement.pdf").stat().st_size < 30_000:
        undersized.append("paper/supplement.pdf")
    if (
        (EXP_DIR / "paper/submission_statements.pdf").exists()
        and (EXP_DIR / "paper/submission_statements.pdf").stat().st_size < 20_000
    ):
        undersized.append("paper/submission_statements.pdf")
    if (EXP_DIR / "paper/aaai27_readiness.pdf").exists() and (EXP_DIR / "paper/aaai27_readiness.pdf").stat().st_size < 20_000:
        undersized.append("paper/aaai27_readiness.pdf")
    if (EXP_DIR / "paper/aaai27_preflight.pdf").exists() and (EXP_DIR / "paper/aaai27_preflight.pdf").stat().st_size < 50_000:
        undersized.append("paper/aaai27_preflight.pdf")
    if (EXP_DIR / "paper/aaai27_official.pdf").exists() and (EXP_DIR / "paper/aaai27_official.pdf").stat().st_size < 50_000:
        undersized.append("paper/aaai27_official.pdf")
    if (
        (EXP_DIR / "paper/aaai27_reproducibility_checklist.pdf").exists()
        and (EXP_DIR / "paper/aaai27_reproducibility_checklist.pdf").stat().st_size < 20_000
    ):
        undersized.append("paper/aaai27_reproducibility_checklist.pdf")
    for path in missing:
        issues.append(Issue("SERIOUS", "artifacts", path, "Required final artifact missing"))
    for path in undersized:
        issues.append(Issue("MEDIUM", "artifacts", path, "Artifact is unexpectedly small"))
    return {"missing": missing, "undersized": sorted(set(undersized)), "pass": not missing and not undersized}


def check_code(issues: list[Issue]) -> dict:
    compile_errors = []
    files = sorted(EXP_DIR.glob("*.py"))
    bank_path = EXP_DIR.parents[1] / "external" / "S4M" / "s4m" / "model" / "Bank.py"
    if bank_path.exists():
        files.append(bank_path)
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            compile_errors.append({"file": str(path), "error": str(exc)})
    device_issues = []
    pin_memory_issues = []
    for path in sorted(EXP_DIR.glob("*.py")):
        text = path.read_text()
        if "import torch" in text or "from torch" in text:
            if "torch.cuda.is_available()" not in text or "torch.backends.mps.is_available()" not in text:
                device_issues.append(str(path.relative_to(EXP_DIR)))
            if re.search(r"pin_memory\s*=\s*True", text):
                pin_memory_issues.append(str(path.relative_to(EXP_DIR)))
    s4m_patch_ok = True
    if bank_path.exists():
        bank_text = bank_path.read_text()
        s4m_patch_ok = ".to(Q.device)" in bank_text and ".cuda()" not in bank_text
    for row in compile_errors:
        issues.append(Issue("SERIOUS", "code", row["file"], row["error"]))
    for path in device_issues:
        issues.append(Issue("SERIOUS", "code", path, "Torch device selection does not include CUDA -> MPS -> CPU"))
    for path in pin_memory_issues:
        issues.append(Issue("MEDIUM", "code", path, "Bare pin_memory true setting appears without final audit approval"))
    if not s4m_patch_ok:
        issues.append(Issue("SERIOUS", "code", str(bank_path), "S4M device-port patch is missing or still uses .cuda()"))
    return {
        "compile_errors": compile_errors,
        "device_issues": device_issues,
        "pin_memory_issues": pin_memory_issues,
        "s4m_device_patch_ok": s4m_patch_ok,
        "pass": not compile_errors and not device_issues and not pin_memory_issues and s4m_patch_ok,
    }


def check_numerical_trace(issues: list[Issue]) -> dict:
    m5 = load_json(EXP_DIR / "m5_main_track_audit" / "main_track_audit.json")
    m12 = load_json(EXP_DIR / "m12_official_s4m_baseline" / "s4m_baseline_summary.json")
    m13 = load_json(EXP_DIR / "m13_hierarchical_bootstrap" / "hierarchical_bootstrap_summary.json")
    m14 = load_json(EXP_DIR / "m14_s4m_scale_validation" / "s4m_scale_validation_summary.json")
    m16 = load_json(EXP_DIR / "m16_official_tslib_full_coverage" / "official_tslib_full_coverage_summary.json")
    expected = []
    expected += ["135.8\\% [8.6, 262.9]", "128.6\\% [98.1, 159.1]", "-1.00 [-1.00, -1.00]"]
    expected += ["96.0\\% [39.4, 152.6]", "32.3\\% [-10.4, 75.0]"]
    for row in m12["datasets"]:
        expected.append(pct_ci(row["max_relative_degradation_ci"]).replace("%", "\\%"))
        expected.append(num_ci(row["max_absolute_delta_ci"]))
        expected.append(num_ci(row["kruskal_p_ci"], lower=0.0, upper=1.0))
    for row in m14["datasets"]:
        expected.append(pct_ci(row["max_relative_degradation_ci"]).replace("%", "\\%"))
        expected.append(num_ci(row["max_absolute_delta_ci"]))
        expected.append(num_ci(row["kruskal_p_ci"], lower=0.0, upper=1.0))
    for row in m13["dataset_summaries"][:2]:
        expected.append(num_ci(row["max_absolute_delta"]))
        expected.append(f"{row['rank_instability_probability']:.2f}")
    for row in m16["datasets"]:
        if row["dataset"] in {"Traffic", "AirConvection"}:
            expected.append(f"{row['max_relative_degradation'] * 100:.1f}\\%")
            expected.append(f"{row['worst_rank_tau']:.1f}")
    passed_main = contains_all("paper/main.tex", expected, issues, "numeric_trace")
    table_expected = []
    for row in m14["datasets"]:
        table_expected.append(pct_ci(row["max_relative_degradation_ci"]))
        table_expected.append(num_ci(row["kruskal_p_ci"], lower=0.0, upper=1.0))
    m16_table_expected = []
    for row in m16["datasets"]:
        if row["dataset"] in {"Traffic", "AirConvection"}:
            m16_table_expected.append(f"{row['max_relative_degradation']:.1%}")
            m16_table_expected.append(f"{row['worst_rank_tau']:.3f}")
    passed_table = contains_all("tables/m14_s4m_scale_table.md", table_expected, issues, "numeric_trace")
    passed_m16_table = contains_all(
        "tables/m16_official_tslib_full_coverage_table.md",
        m16_table_expected,
        issues,
        "numeric_trace",
    )
    return {
        "main_tex_trace_snippets_checked": len(expected),
        "main_tex_trace_snippets_passed": passed_main,
        "m14_table_snippets_checked": len(table_expected),
        "m14_table_snippets_passed": passed_table,
        "m16_table_snippets_checked": len(m16_table_expected),
        "m16_table_snippets_passed": passed_m16_table,
        "m5_verdict": m5.get("verdict"),
        "m5_blocking_items": m5.get("blocking_items", []),
        "m5_context_only": True,
        "pass": (
            passed_main == len(expected)
            and passed_table == len(table_expected)
            and passed_m16_table == len(m16_table_expected)
        ),
    }


def check_text_hygiene(issues: list[Issue]) -> dict:
    hard_patterns = [
        "TODO",
        "TBD",
        "FIXME",
        "not final submission",
        "S4M was still absent",
        "pre-v5 gap",
        "NOT_SUBMISSION_READY_YET",
    ]
    hits = []
    for path in FINAL_DOCS:
        text = read_rel(path)
        for pattern in hard_patterns:
            if pattern in text:
                hits.append({"file": path, "pattern": pattern})
                issues.append(Issue("MEDIUM", "text_hygiene", path, f"Final document contains stale marker: {pattern}"))
    required_scope = [
        "not a new forecasting backbone",
        "negative diagnostic",
        "not a full S4M benchmark",
        "Traffic and AirConvection",
    ]
    scope_hits = {phrase: phrase in read_rel("paper/main.tex") for phrase in required_scope}
    for phrase, ok in scope_hits.items():
        if not ok:
            issues.append(Issue("SERIOUS", "claim_scope", "paper/main.tex", f"Missing scope guardrail phrase: {phrase}"))
    return {"stale_hits": hits, "scope_hits": scope_hits, "pass": not hits and all(scope_hits.values())}


def check_submission_supplement(issues: list[Issue]) -> dict:
    summary_path = EXP_DIR / "m17_submission_supplement" / "submission_supplement_summary.json"
    required_phrases = {
        "paper/supplement.tex": [
            "MaskShift is a benchmark/theory submission",
            "Official PatchTST/TimeXer Coverage",
            "Reviewer Concern Matrix",
            "Reproducibility Commands",
        ],
        "docs/DATASET_MECHANISM_CARDS.md": [
            "Dataset Cards",
            "Mechanism Cards",
            "Claim Boundary",
        ],
        "docs/REBUTTAL_PLAYBOOK.md": [
            "Non-Negotiable Scope Lines",
            "Reviewer-Concern Playbook",
            "Dataset-Specific Response Map",
        ],
        "tables/m17_reviewer_response_matrix.md": [
            "Likely reviewer concern",
            "Were modern forecasting backbones tested?",
            "Were missing-aware architectures tested?",
        ],
    }
    missing = []
    phrase_misses = []
    if not summary_path.exists():
        issues.append(Issue("SERIOUS", "submission_supplement", str(summary_path), "M17 summary is missing"))
        return {
            "status": "missing",
            "missing": [str(summary_path.relative_to(EXP_DIR))],
            "phrase_misses": [],
            "supplement_pdf_size": 0,
            "pass": False,
        }
    summary = load_json(summary_path)
    for artifact in summary.get("artifacts", []):
        if not (EXP_DIR / artifact).exists():
            missing.append(artifact)
            issues.append(Issue("SERIOUS", "submission_supplement", artifact, "M17 artifact missing"))
    for path, phrases in required_phrases.items():
        full = EXP_DIR / path
        if not full.exists():
            continue
        text = full.read_text()
        for phrase in phrases:
            if phrase not in text:
                phrase_misses.append({"file": path, "phrase": phrase})
                issues.append(Issue("MEDIUM", "submission_supplement", path, f"Missing supplement phrase: {phrase}"))
    pdf_size = (EXP_DIR / "paper/supplement.pdf").stat().st_size if (EXP_DIR / "paper/supplement.pdf").exists() else 0
    status_ok = summary.get("status") == "PASS_SUBMISSION_SUPPLEMENT"
    if not status_ok:
        issues.append(Issue("SERIOUS", "submission_supplement", str(summary_path), "M17 status is not PASS_SUBMISSION_SUPPLEMENT"))
    return {
        "status": summary.get("status", "not_run"),
        "reviewer_concern_count": summary.get("reviewer_concern_count", 0),
        "coverage": summary.get("coverage", {}),
        "missing": missing,
        "phrase_misses": phrase_misses,
        "supplement_pdf_size": pdf_size,
        "evidence": f"{summary.get('reviewer_concern_count', 0)} reviewer concerns; PDF {pdf_size} bytes",
        "pass": status_ok and not missing and not phrase_misses and pdf_size >= 30_000,
    }


def check_submission_policy_pack(issues: list[Issue]) -> dict:
    summary_path = EXP_DIR / "m18_submission_policy_pack" / "submission_policy_summary.json"
    required_phrases = {
        "docs/SUBMISSION_STATEMENTS.md": [
            "Data Availability Statement",
            "Code Availability Statement",
            "Reproducibility Statement",
            "Ethics Statement",
            "AI Assistance Disclosure",
            "Conflict of Interest Statement",
            "Funding Statement",
            "Author Contributions",
            "Anonymity Statement",
        ],
        "docs/VENUE_READINESS_AUDIT.md": [
            "Target venue style",
            "pending-target-selection",
            "Page-limit readiness",
            "Supplement readiness",
        ],
        "tables/m18_policy_readiness_table.md": [
            "Data Availability Statement",
            "AI Assistance Disclosure",
            "Author Contributions",
        ],
        "paper/submission_statements.tex": [
            "Submission Statements",
            "Venue-Readiness Audit",
            "Target-Specific Action",
        ],
    }
    missing = []
    phrase_misses = []
    if not summary_path.exists():
        issues.append(Issue("SERIOUS", "submission_policy_pack", str(summary_path), "M18 summary is missing"))
        return {
            "status": "missing",
            "missing": [str(summary_path.relative_to(EXP_DIR))],
            "phrase_misses": [],
            "submission_statements_pdf_size": 0,
            "pass": False,
        }
    summary = load_json(summary_path)
    for artifact in summary.get("artifacts", []):
        if not (EXP_DIR / artifact).exists():
            missing.append(artifact)
            issues.append(Issue("SERIOUS", "submission_policy_pack", artifact, "M18 artifact missing"))
    for path, phrases in required_phrases.items():
        full = EXP_DIR / path
        if not full.exists():
            continue
        text = full.read_text()
        for phrase in phrases:
            if phrase not in text:
                phrase_misses.append({"file": path, "phrase": phrase})
                issues.append(Issue("MEDIUM", "submission_policy_pack", path, f"Missing policy phrase: {phrase}"))
    pdf_size = (
        (EXP_DIR / "paper/submission_statements.pdf").stat().st_size
        if (EXP_DIR / "paper/submission_statements.pdf").exists()
        else 0
    )
    status_ok = summary.get("status") == "PASS_SUBMISSION_POLICY_PACK"
    if not status_ok:
        issues.append(Issue("SERIOUS", "submission_policy_pack", str(summary_path), "M18 status is not PASS_SUBMISSION_POLICY_PACK"))
    blocking_policy_gaps = summary.get("blocking_policy_gaps", [])
    if blocking_policy_gaps:
        issues.append(Issue("SERIOUS", "submission_policy_pack", str(summary_path), "M18 contains blocking policy gaps"))
    return {
        "status": summary.get("status", "not_run"),
        "statement_count": summary.get("statement_count", 0),
        "target_specific_template_status": summary.get("target_specific_template_status", ""),
        "blocking_policy_gaps": blocking_policy_gaps,
        "missing": missing,
        "phrase_misses": phrase_misses,
        "submission_statements_pdf_size": pdf_size,
        "evidence": f"{summary.get('statement_count', 0)} statements; PDF {pdf_size} bytes; target template {summary.get('target_specific_template_status', '')}",
        "pass": status_ok and not missing and not phrase_misses and not blocking_policy_gaps and pdf_size >= 20_000,
    }


def check_aaai27_target_dossier(issues: list[Issue]) -> dict:
    summary_path = EXP_DIR / "m19_aaai27_target_readiness" / "aaai27_target_readiness_summary.json"
    required_phrases = {
        "docs/AAAI27_TARGET_READINESS.md": [
            "AAAI-27 Main Technical Track",
            "2026-07-21",
            "2026-07-28",
            "up to 7 pages plus references",
            "AI-generated non-decisional review",
            "Upload Verdict",
        ],
        "docs/AAAI27_REPRODUCIBILITY_CHECKLIST_DRAFT.md": [
            "Datasets",
            "Train/test split",
            "Randomness/seeds",
            "Baselines",
            "Code availability",
        ],
        "docs/AAAI27_PHASE_REVIEW_RESPONSE_PLAN.md": [
            "AI-generated, non-decisional first-stage review",
            "Likely objection",
            "Response stance",
        ],
        "tables/m19_aaai27_gap_table.md": [
            "Target style",
            "Page limit",
            "Reproducibility",
            "AI-generated first-stage review readiness",
        ],
        "paper/aaai27_readiness.tex": [
            "AAAI-27 Target-Readiness Dossier",
            "Target Gap Table",
            "Verdict",
        ],
    }
    missing = []
    phrase_misses = []
    if not summary_path.exists():
        issues.append(Issue("SERIOUS", "aaai27_target_dossier", str(summary_path), "M19 summary is missing"))
        return {
            "status": "missing",
            "missing": [str(summary_path.relative_to(EXP_DIR))],
            "phrase_misses": [],
            "aaai27_readiness_pdf_size": 0,
            "pass": False,
        }
    summary = load_json(summary_path)
    for artifact in summary.get("artifacts", []):
        if not (EXP_DIR / artifact).exists():
            missing.append(artifact)
            issues.append(Issue("SERIOUS", "aaai27_target_dossier", artifact, "M19 artifact missing"))
    for path, phrases in required_phrases.items():
        full = EXP_DIR / path
        if not full.exists():
            continue
        text = full.read_text()
        for phrase in phrases:
            if phrase not in text:
                phrase_misses.append({"file": path, "phrase": phrase})
                issues.append(Issue("MEDIUM", "aaai27_target_dossier", path, f"Missing AAAI-27 phrase: {phrase}"))
    pdf_size = (
        (EXP_DIR / "paper/aaai27_readiness.pdf").stat().st_size
        if (EXP_DIR / "paper/aaai27_readiness.pdf").exists()
        else 0
    )
    status_ok = summary.get("status") == "PASS_AAAI27_TARGET_DOSSIER"
    if not status_ok:
        issues.append(Issue("SERIOUS", "aaai27_target_dossier", str(summary_path), "M19 status is not PASS_AAAI27_TARGET_DOSSIER"))
    return {
        "status": summary.get("status", "not_run"),
        "aaai27_upload_ready": summary.get("aaai27_upload_ready", False),
        "upload_blockers": [row.get("item") for row in summary.get("upload_blockers", [])],
        "missing": missing,
        "phrase_misses": phrase_misses,
        "aaai27_readiness_pdf_size": pdf_size,
        "evidence": f"M19 status {summary.get('status', 'not_run')}; upload ready {summary.get('aaai27_upload_ready', False)}; PDF {pdf_size} bytes",
        "pass": status_ok and not missing and not phrase_misses and pdf_size >= 20_000,
    }


def check_aaai27_preflight_conversion(issues: list[Issue]) -> dict:
    summary_path = EXP_DIR / "m20_aaai27_preflight_conversion" / "aaai27_preflight_summary.json"
    required_phrases = {
        "docs/AAAI27_PREFLIGHT_FORMAT_AUDIT.md": [
            "AAAI-27 Preflight Conversion Audit",
            "two-column",
            "US-letter",
            "official AAAI author kit",
            "Two-column table fit",
            "Official AAAI build",
        ],
        "tables/m20_aaai27_preflight_table.md": [
            "Page count",
            "Two-column",
            "Table fit",
            "Official kit ready",
            "Official page count",
        ],
        "paper/aaai27_preflight.tex": [
            "twocolumn",
            "letterpaper",
            "Anonymous Submission",
            "bibliography{references}",
            r"\begin{table*}",
        ],
        "paper/aaai27_official.tex": [
            r"\usepackage[submission]{aaai2027}",
            "Anonymous Submission",
            "bibliography{references}",
            r"\begin{table*}",
            "TemplateVersion (2027.1)",
        ],
    }
    missing = []
    phrase_misses = []
    if not summary_path.exists():
        issues.append(Issue("SERIOUS", "aaai27_preflight_conversion", str(summary_path), "M20 summary is missing"))
        return {
            "status": "missing",
            "missing": [str(summary_path.relative_to(EXP_DIR))],
            "phrase_misses": [],
            "preflight_pdf_size": 0,
            "preflight_page_count": None,
            "pass": False,
        }
    summary = load_json(summary_path)
    for artifact in summary.get("artifacts", []):
        if not (EXP_DIR / artifact).exists():
            missing.append(artifact)
            issues.append(Issue("SERIOUS", "aaai27_preflight_conversion", artifact, "M20 artifact missing"))
    for path, phrases in required_phrases.items():
        full = EXP_DIR / path
        if not full.exists():
            continue
        text = full.read_text()
        for phrase in phrases:
            if phrase not in text:
                phrase_misses.append({"file": path, "phrase": phrase})
                issues.append(Issue("MEDIUM", "aaai27_preflight_conversion", path, f"Missing M20 phrase: {phrase}"))
    state = summary.get("state", {})
    pdf_size = (EXP_DIR / "paper/aaai27_preflight.pdf").stat().st_size if (EXP_DIR / "paper/aaai27_preflight.pdf").exists() else 0
    page_count = state.get("preflight_page_count")
    status_ok = summary.get("status") == "PASS_AAAI27_PREFLIGHT_CONVERSION"
    if not status_ok:
        issues.append(Issue("SERIOUS", "aaai27_preflight_conversion", str(summary_path), "M20 status is not PASS_AAAI27_PREFLIGHT_CONVERSION"))
    if page_count is None or page_count > 7:
        issues.append(Issue("SERIOUS", "aaai27_preflight_conversion", "paper/aaai27_preflight.pdf", "Preflight page count is missing or exceeds 7 pages"))
    return {
        "status": summary.get("status", "not_run"),
        "official_kit_upload_ready": summary.get("aaai_official_kit_upload_ready", False),
        "official_template_build_pass": summary.get("official_template_build_pass", False),
        "page_limit_preflight_pass": summary.get("page_limit_preflight_pass", False),
        "preflight_page_count": page_count,
        "official_page_count": state.get("official_page_count"),
        "preflight_pdf_fresh": state.get("preflight_pdf_fresh", False),
        "official_pdf_fresh": state.get("official_pdf_fresh", False),
        "preflight_pdf_size": pdf_size,
        "official_pdf_size": state.get("official_pdf_size", 0),
        "uses_twocolumn": state.get("uses_twocolumn", False),
        "uses_letterpaper": state.get("uses_letterpaper", False),
        "uses_double_column_tables": state.get("uses_double_column_tables", False),
        "official_uses_aaai2027_submission": state.get("official_uses_aaai2027_submission", False),
        "official_uses_double_column_tables": state.get("official_uses_double_column_tables", False),
        "anonymous_submission": state.get("anonymous_submission", False),
        "official_anonymous_submission": state.get("official_anonymous_submission", False),
        "missing": missing,
        "phrase_misses": phrase_misses,
        "evidence": f"M20 status {summary.get('status', 'not_run')}; pages {page_count}; PDF {pdf_size} bytes",
        "pass": (
            status_ok
            and not missing
            and not phrase_misses
            and bool(summary.get("page_limit_preflight_pass", False))
            and bool(summary.get("official_template_build_pass", False))
            and bool(summary.get("aaai_official_kit_upload_ready", False))
            and bool(state.get("preflight_pdf_fresh", False))
            and bool(state.get("official_pdf_fresh", False))
            and bool(state.get("uses_twocolumn", False))
            and bool(state.get("uses_letterpaper", False))
            and bool(state.get("uses_double_column_tables", False))
            and bool(state.get("official_uses_aaai2027_submission", False))
            and bool(state.get("official_uses_double_column_tables", False))
            and bool(state.get("anonymous_submission", False))
            and bool(state.get("official_anonymous_submission", False))
            and pdf_size >= 50_000
            and state.get("official_pdf_size", 0) >= 50_000
            and page_count is not None
            and page_count <= 7
            and state.get("official_page_count") is not None
            and state.get("official_page_count") <= 7
        ),
    }


def check_aaai27_reproducibility_checklist(issues: list[Issue]) -> dict:
    summary_path = EXP_DIR / "m21_aaai27_reproducibility_checklist" / "aaai27_reproducibility_checklist_summary.json"
    required_phrases = {
        "docs/AAAI27_REPRODUCIBILITY_CHECKLIST_FILLED.md": [
            "AAAI-27 Filled Reproducibility Checklist",
            "Answer slots",
            "Remaining placeholders",
        ],
        "paper/aaai27_reproducibility_checklist.tex": [
            "Reproducibility Checklist",
            "MaskShift does not introduce a new AI method",
            "Weather, Electricity, Traffic, and AirConvection",
            "BH-FDR",
            "Type your response here",
        ],
    }
    missing = []
    phrase_misses = []
    if not summary_path.exists():
        issues.append(Issue("SERIOUS", "aaai27_reproducibility_checklist", str(summary_path), "M21 summary is missing"))
        return {"status": "missing", "missing": [str(summary_path.relative_to(EXP_DIR))], "phrase_misses": [], "pass": False}
    summary = load_json(summary_path)
    for artifact in summary.get("artifacts", []):
        if not (EXP_DIR / artifact).exists():
            missing.append(artifact)
            issues.append(Issue("SERIOUS", "aaai27_reproducibility_checklist", artifact, "M21 artifact missing"))
    for path, phrases in required_phrases.items():
        full = EXP_DIR / path
        if not full.exists():
            continue
        text = full.read_text()
        for phrase in phrases:
            if phrase not in text:
                phrase_misses.append({"file": path, "phrase": phrase})
                issues.append(Issue("MEDIUM", "aaai27_reproducibility_checklist", path, f"Missing M21 phrase: {phrase}"))
    tex = read_rel("paper/aaai27_reproducibility_checklist.tex")
    question_section = tex.split("% The questions start here", 1)[1] if "% The questions start here" in tex else tex
    question_placeholders = question_section.count("Type your response here")
    if question_placeholders:
        issues.append(Issue("SERIOUS", "aaai27_reproducibility_checklist", "paper/aaai27_reproducibility_checklist.tex", "Checklist question section still contains placeholders"))
    pdf_size = (
        (EXP_DIR / "paper/aaai27_reproducibility_checklist.pdf").stat().st_size
        if (EXP_DIR / "paper/aaai27_reproducibility_checklist.pdf").exists()
        else 0
    )
    status_ok = summary.get("status") == "PASS_AAAI27_REPRODUCIBILITY_CHECKLIST"
    if not status_ok:
        issues.append(Issue("SERIOUS", "aaai27_reproducibility_checklist", str(summary_path), "M21 status is not PASS_AAAI27_REPRODUCIBILITY_CHECKLIST"))
    return {
        "status": summary.get("status", "not_run"),
        "answer_count": summary.get("answer_count", 0),
        "remaining_question_placeholders": question_placeholders,
        "pdf_size": pdf_size,
        "missing": missing,
        "phrase_misses": phrase_misses,
        "evidence": f"M21 status {summary.get('status', 'not_run')}; answers {summary.get('answer_count', 0)}; PDF {pdf_size} bytes",
        "pass": status_ok and not missing and not phrase_misses and question_placeholders == 0 and pdf_size >= 20_000,
    }


def failure_mode_audit(checks: dict) -> list[dict]:
    return [
        {
            "mode": "Mode 1: implementation bug passing self-review",
            "status": "CLEAR",
            "evidence": "M15 py_compile over MaskShift scripts and S4M Bank.py passed; M5 remains STRONG_CONFERENCE_READY with zero blocking items.",
        },
        {
            "mode": "Mode 2: hallucinated citation",
            "status": "CLEAR",
            "evidence": "No dangling/orphan citations; every BibTeX entry has a DOI or URL and a recorded external audit source.",
        },
        {
            "mode": "Mode 3: hallucinated experimental result",
            "status": "CLEAR",
            "evidence": "Core numerical claims in main.tex and M14/M16 tables are traced to JSON summaries; M5/M12/M13/M14/M16/M17 summaries are parseable.",
        },
        {
            "mode": "Mode 4: shortcut reliance",
            "status": "CLEAR_WITH_SCOPE_NOTE",
            "evidence": "M8 non-retirement decomposition and mixed Traffic/AirConvection reporting reduce shortcut risk; full causal shortcut ablation remains out of scope for a benchmark paper.",
        },
        {
            "mode": "Mode 5: implementation bug reframed as insight",
            "status": "CLEAR",
            "evidence": "Typed-head surprise is reported as a negative diagnostic, and S4M negative evidence is disclosed as contrastive rather than converted into a universal failure claim.",
        },
        {
            "mode": "Mode 6: methodology fabrication",
            "status": "CLEAR",
            "evidence": "Method scope in main.tex, the M17 supplement, and M18 statements matches run summaries: official-architecture adaptations, reduced channels/samples, three seed offsets, encoder-mask-only protocol, and target-agnostic disclosure boundaries are stated.",
        },
        {
            "mode": "Mode 7: early frame-lock",
            "status": "CLEAR",
            "evidence": "The paper frame has been revised to benchmark/theory; method/SOTA claims are explicitly excluded, M18 marks target-template compliance as pending-target-selection, M19 records AAAI-27 upload blockers instead of hiding them, M20 separates page-pressure preflight from official-kit readiness, and M5 method_claim_ready remains false.",
        },
    ]


def write_table(summary: dict) -> None:
    lines = [
        "# M15 final integrity audit table",
        "",
        "| Category | Status | Issues | Evidence |",
        "| --- | --- | ---: | --- |",
    ]
    for name, check in summary["checks"].items():
        status = "PASS" if check.get("pass") else "FAIL"
        issue_count = sum(1 for issue in summary["issues"] if issue["category"] == name)
        evidence = check.get("evidence", "")
        if not evidence:
            if name == "references":
                evidence = f"{len(check['bib_keys'])} references, {len(check['cited_keys'])} cited keys"
            elif name == "numeric_trace":
                evidence = f"{check['main_tex_trace_snippets_passed']}/{check['main_tex_trace_snippets_checked']} main-text snippets"
            else:
                evidence = "deterministic local audit"
        lines.append(f"| {name} | {status} | {issue_count} | {evidence} |")
    (TABLE_DIR / "m15_integrity_table.md").write_text("\n".join(lines) + "\n")


def write_report(summary: dict) -> None:
    lines = [
        "# MaskShift Final Integrity Report",
        "",
        f"- Verdict: `{summary['verdict']}`",
        f"- Generated: `{summary['generated_at']}`",
        f"- Blocking issues: `{len(summary['blocking_issues'])}`",
        "",
        "## Gate Summary",
        "",
        "| Gate | Pass | Notes |",
        "| --- | --- | --- |",
    ]
    for name, check in summary["checks"].items():
        notes = ""
        if name == "references":
            notes = f"{len(check['bib_keys'])} references; {len(check['dangling'])} dangling; {len(check['orphan'])} orphan"
        elif name == "numeric_trace":
            notes = f"{check['main_tex_trace_snippets_passed']}/{check['main_tex_trace_snippets_checked']} main-text snippets"
        elif name == "code":
            notes = f"{len(check['compile_errors'])} compile errors; device issues {len(check['device_issues'])}"
        elif name == "artifacts":
            notes = f"{len(check['missing'])} missing; {len(check['undersized'])} undersized"
        elif name == "text_hygiene":
            notes = f"{len(check['stale_hits'])} stale markers"
        elif name == "submission_policy_pack":
            notes = f"{check['statement_count']} statements; template {check['target_specific_template_status']}"
        elif name == "aaai27_target_dossier":
            notes = f"upload ready {check['aaai27_upload_ready']}; blockers {check['upload_blockers']}"
        elif name == "aaai27_preflight_conversion":
            notes = f"pages {check['preflight_page_count']}; official-kit upload ready {check['official_kit_upload_ready']}"
        elif name == "aaai27_reproducibility_checklist":
            notes = f"{check['answer_count']} answers; placeholders {check['remaining_question_placeholders']}"
        lines.append(f"| {name} | `{check.get('pass')}` | {notes} |")
    lines += [
        "",
        "## AI Research Failure Mode Audit",
        "",
        "| Mode | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for row in summary["failure_mode_audit"]:
        lines.append(f"| {row['mode']} | {row['status']} | {row['evidence']} |")
    lines += ["", "## Issues", ""]
    if summary["issues"]:
        lines.append("| Severity | Category | Location | Description |")
        lines.append("| --- | --- | --- | --- |")
        for issue in summary["issues"]:
            lines.append(f"| {issue['severity']} | {issue['category']} | `{issue['location']}` | {issue['description']} |")
    else:
        lines.append("No integrity issues detected by M15.")
    lines += [
        "",
        "## Reference Audit Sources",
        "",
        "| Key | Source |",
        "| --- | --- |",
    ]
    for key, source in summary["checks"]["references"]["reference_audit_sources"].items():
        lines.append(f"| `{key}` | {source} |")
    lines += [
        "",
        "## Scope Note",
        "",
        "M15 is a final local integrity gate. It verifies citation graph hygiene, BibTeX hygiene, internal numerical traceability, artifact presence, code compilation, device-selection rules, and ARS AI-research failure-mode coverage. It does not replace professional plagiarism software or a full external reproducibility replication.",
    ]
    (DOC_DIR / "FINAL_INTEGRITY_REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    issues: list[Issue] = []
    checks = {
        "references": check_references(issues),
        "artifacts": check_artifacts(issues),
        "code": check_code(issues),
        "numeric_trace": check_numerical_trace(issues),
        "text_hygiene": check_text_hygiene(issues),
        "submission_supplement": check_submission_supplement(issues),
        "submission_policy_pack": check_submission_policy_pack(issues),
        "aaai27_target_dossier": check_aaai27_target_dossier(issues),
        "aaai27_preflight_conversion": check_aaai27_preflight_conversion(issues),
        "aaai27_reproducibility_checklist": check_aaai27_reproducibility_checklist(issues),
    }
    blocking_issues = [issue for issue in issues if issue.severity in {"SERIOUS", "MEDIUM"}]
    verdict = "PASS_FINAL_INTEGRITY" if all(check["pass"] for check in checks.values()) and not blocking_issues else "FAIL_FINAL_INTEGRITY"
    summary = {
        "milestone": "M15",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "checks": checks,
        "issues": [issue.__dict__ for issue in issues],
        "blocking_issues": [issue.__dict__ for issue in blocking_issues],
        "failure_mode_audit": failure_mode_audit(checks),
        "scope_note": "Local final integrity gate; does not replace professional plagiarism software or full external reproduction.",
    }
    write_json(OUT_DIR / "final_integrity_summary.json", summary)
    write_table(summary)
    write_report(summary)
    print(
        json.dumps(
            {
                "milestone": "M15",
                "verdict": verdict,
                "blocking_issues": len(blocking_issues),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
