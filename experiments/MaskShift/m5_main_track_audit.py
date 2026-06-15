"""M5 — Strong-conference readiness audit for MaskShift."""

from __future__ import annotations

import json
from pathlib import Path

from .maskshift_core import ensure_dir, write_json


EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m5_main_track_audit")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def exists(path: str) -> bool:
    return (EXP_DIR / path).exists()


def main() -> None:
    m0 = load_json(EXP_DIR / "m0_mask_suite" / "m0_summary.json")
    m1 = load_json(EXP_DIR / "m1_mechanism_audit" / "m1_summary.json")
    m2 = load_json(EXP_DIR / "m2_typed_head" / "m2_summary.json")
    m3 = load_json(EXP_DIR / "m3_statistical_tests" / "m3_summary.json")
    m4 = load_json(EXP_DIR / "m4_paper_ready" / "paper_ready_summary.json")
    m6_path = EXP_DIR / "m6_deep_backbone_sweep" / "deep_backbone_sweep_summary.json"
    m6 = load_json(m6_path) if m6_path.exists() else {}
    m7_path = EXP_DIR / "m7_severity_curves" / "severity_curves_summary.json"
    m7 = load_json(m7_path) if m7_path.exists() else {}
    m8_path = EXP_DIR / "m8_mechanism_decomposition" / "mechanism_decomposition_summary.json"
    m8 = load_json(m8_path) if m8_path.exists() else {}
    m9_path = EXP_DIR / "m9_official_tslib_reproduction" / "official_tslib_reproduction_summary.json"
    m9 = load_json(m9_path) if m9_path.exists() else {}
    m10_path = EXP_DIR / "m10_submission_hardening" / "submission_hardening_summary.json"
    m10 = load_json(m10_path) if m10_path.exists() else {}
    m11_path = EXP_DIR / "m11_official_ctf_missing_baseline" / "ctf_missing_baseline_summary.json"
    m11 = load_json(m11_path) if m11_path.exists() else {}
    m12_path = EXP_DIR / "m12_official_s4m_baseline" / "s4m_baseline_summary.json"
    m12 = load_json(m12_path) if m12_path.exists() else {}
    m13_path = EXP_DIR / "m13_hierarchical_bootstrap" / "hierarchical_bootstrap_summary.json"
    m13 = load_json(m13_path) if m13_path.exists() else {}
    m14_path = EXP_DIR / "m14_s4m_scale_validation" / "s4m_scale_validation_summary.json"
    m14 = load_json(m14_path) if m14_path.exists() else {}
    m15_path = EXP_DIR / "m15_final_integrity_audit" / "final_integrity_summary.json"
    m15 = load_json(m15_path) if m15_path.exists() else {}
    m16_path = EXP_DIR / "m16_official_tslib_full_coverage" / "official_tslib_full_coverage_summary.json"
    m16 = load_json(m16_path) if m16_path.exists() else {}
    m17_path = EXP_DIR / "m17_submission_supplement" / "submission_supplement_summary.json"
    m17 = load_json(m17_path) if m17_path.exists() else {}
    m18_path = EXP_DIR / "m18_submission_policy_pack" / "submission_policy_summary.json"
    m18 = load_json(m18_path) if m18_path.exists() else {}
    m19_path = EXP_DIR / "m19_aaai27_target_readiness" / "aaai27_target_readiness_summary.json"
    m19 = load_json(m19_path) if m19_path.exists() else {}
    m20_path = EXP_DIR / "m20_aaai27_preflight_conversion" / "aaai27_preflight_summary.json"
    m20 = load_json(m20_path) if m20_path.exists() else {}
    m21_path = EXP_DIR / "m21_aaai27_reproducibility_checklist" / "aaai27_reproducibility_checklist_summary.json"
    m21 = load_json(m21_path) if m21_path.exists() else {}
    paper_text = (EXP_DIR / "PAPER.md").read_text().lower()

    fdr_h1_count = sum(r["fdr_pass"] and r["eta_squared"] >= 0.30 for r in m3["m1_rows"])
    rank_count = sum(r["worst_rank_tau"] <= 0.5 for r in m3["m1_rows"])
    typed_count = sum(
        r["fdr_pass"] and r["typed_reduction"] >= 0.20 and r["clean_mcar_cost"] <= 0.02
        for r in m3["m2_rows"]
    )
    artifact_paths = [
        "PAPER.md",
        "REPORT.md",
        "paper/main.tex",
        "paper/references.bib",
        "m4_paper_ready/REPRODUCE.md",
        "figures/m1_mechanism_rank.png",
        "figures/m2_typed_correction.png",
        "figures/maskshift_overview.png",
        "tables/main_result_table.md",
        "tables/m9_official_architecture_table.md",
        "tables/m7_corrected_robustness_table.md",
        "tables/claim_evidence_table.md",
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
        "docs/MISSING_AWARE_BASELINE_ATTEMPT.md",
        "docs/FINAL_INTEGRITY_REPORT.md",
        "docs/DATASET_MECHANISM_CARDS.md",
        "docs/REBUTTAL_PLAYBOOK.md",
        "docs/SUBMISSION_STATEMENTS.md",
        "docs/VENUE_READINESS_AUDIT.md",
        "docs/AAAI27_TARGET_READINESS.md",
        "docs/AAAI27_REPRODUCIBILITY_CHECKLIST_DRAFT.md",
        "docs/AAAI27_PHASE_REVIEW_RESPONSE_PLAN.md",
        "docs/AAAI27_PREFLIGHT_FORMAT_AUDIT.md",
        "docs/AAAI27_REPRODUCIBILITY_CHECKLIST_FILLED.md",
        "SUBMISSION_CHECKLIST.md",
        "paper/supplement.tex",
        "paper/supplement.pdf",
        "paper/submission_statements.tex",
        "paper/submission_statements.pdf",
        "paper/aaai27_readiness.tex",
        "paper/aaai27_readiness.pdf",
        "paper/aaai27_preflight.tex",
        "paper/aaai27_preflight.pdf",
        "paper/aaai27_official.tex",
        "paper/aaai27_official.pdf",
        "paper/aaai27_reproducibility_checklist.tex",
        "paper/aaai27_reproducibility_checklist.pdf",
        "paper/aaai2027.sty",
        "paper/aaai2027.bst",
        "paper/AuthorKit27.zip",
        "m8_mechanism_decomposition/mechanism_decomposition_summary.json",
        "m9_official_tslib_reproduction/official_tslib_reproduction_summary.json",
        "m10_submission_hardening/submission_hardening_summary.json",
        "m11_official_ctf_missing_baseline/ctf_missing_baseline_summary.json",
        "m12_official_s4m_baseline/s4m_baseline_summary.json",
        "m13_hierarchical_bootstrap/hierarchical_bootstrap_summary.json",
        "m14_s4m_scale_validation/s4m_scale_validation_summary.json",
        "m15_final_integrity_audit/final_integrity_summary.json",
        "m16_official_tslib_full_coverage/official_tslib_full_coverage_summary.json",
        "m17_submission_supplement/submission_supplement_summary.json",
        "m18_submission_policy_pack/submission_policy_summary.json",
        "m19_aaai27_target_readiness/aaai27_target_readiness_summary.json",
        "m20_aaai27_preflight_conversion/aaai27_preflight_summary.json",
        "m21_aaai27_reproducibility_checklist/aaai27_reproducibility_checklist_summary.json",
    ]
    gates = {
        "m0_matched_rate_controls": {
            "pass": all(v["pass"] for v in m0["matched_rate_gate"].values()),
            "evidence": m0["matched_rate_gate"],
        },
        "h1_mechanism_factor_fdr": {
            "pass": fdr_h1_count >= 2,
            "required": ">=2 datasets with q<=0.05 and eta^2>=0.30",
            "observed": fdr_h1_count,
        },
        "h2_rank_instability": {
            "pass": rank_count >= 2,
            "required": ">=2 datasets with Kendall tau<=0.5 against MCAR ranking",
            "observed": rank_count,
        },
        "h3_typed_correction": {
            "pass": typed_count >= 2,
            "required": ">=2 datasets with FDR-backed typed reduction >=20% and clean cost <=2%",
            "observed": typed_count,
            "note": "If this fails, keep typed head as secondary and submit benchmark/theory route only after deep-backbone expansion.",
        },
        "artifact_package": {
            "pass": all(exists(p) for p in artifact_paths),
            "required": artifact_paths,
            "missing": [p for p in artifact_paths if not exists(p)],
        },
        "claim_scope_safe": {
            "pass": all(
                phrase in paper_text
                for phrase in [
                    "not a new forecasting backbone",
                    "negative diagnostic",
                    "official-architecture adaptation",
                ]
            ),
            "required": "paper explicitly weakens method/SOTA claims and labels M9 protocol scope",
            "paper_route": m4["paper_route"],
        },
        "deep_backbone_sweep": {
            "pass": bool(m6.get("m6_gate", False) and m6.get("deep_lite_protocol_complete", False)),
            "required": ">=3 neural proxy backbones across >=2 datasets with at least one mechanism-shift gate pass",
            "observed": {
                "status": m6.get("status", "not_run"),
                "backbones": m6.get("backbones", []),
                "datasets": [d.get("dataset") for d in m6.get("datasets", [])],
                "scope_note": m6.get("scope_note", ""),
            },
        },
        "severity_curves": {
            "pass": bool(m7.get("status") == "PASS_SEVERITY_CURVES"),
            "required": "multi-rate severity audit with >=2 dataset gates at 35% and 50%",
            "observed": m7.get("by_rate", {}),
        },
        "non_retirement_decomposition": {
            "pass": bool(m8.get("status") == "PASS_NON_RETIREMENT_DECOMPOSITION"),
            "required": "show mechanism-shift evidence is not solely driven by retirement",
            "observed": {
                "status": m8.get("status", "not_run"),
                "gate_rule": m8.get("gate_rule", ""),
                "dataset_rows": m8.get("dataset_rows", []),
            },
        },
        "official_baseline_reproduction": {
            "pass": bool(m9.get("status") == "PASS_OFFICIAL_TSLIB_REPRODUCTION"),
            "required": "official PatchTST/TimeXer/S4M or ChannelTokenFormer-compatible reproduction before camera-ready-strength claims",
            "observed": {
                "status": m9.get("status", "not_run"),
                "backbones": m9.get("backbones", []),
                "datasets": [d.get("dataset") for d in m9.get("datasets", [])],
                "tslib_revision": m9.get("tslib_revision", ""),
                "protocol_note": m9.get("protocol_note", ""),
            },
        },
        "submission_hardening": {
            "pass": bool(m10.get("status") == "PASS_SUBMISSION_HARDENING"),
            "required": "M10 seed CIs, corrected severity metrics, tables, overview figure, and missing-aware-baseline audit",
            "observed": {
                "status": m10.get("status", "not_run"),
                "runtime_note": m10.get("runtime_note", ""),
                "missing_aware_baseline_status": m10.get("missing_aware_baseline_status", ""),
                "final_missing_aware_evidence": "superseded by M11 ChannelTokenFormer_missing and M12 S4M official adaptations",
                "tables": m10.get("tables", []),
            },
        },
        "missing_aware_official_baseline": {
            "pass": bool(m11.get("status") == "PASS_OFFICIAL_CTF_MISSING_ADAPTATION"),
            "required": "official missing-aware architecture adaptation, preferably S4M or ChannelTokenFormer-compatible",
            "observed": {
                "status": m11.get("status", "not_run"),
                "backbone": m11.get("backbone", ""),
                "ctf_revision": m11.get("ctf_revision", ""),
                "mechanism_shift_gate": m11.get("mechanism_shift_gate", False),
                "dataset_summaries": m11.get("dataset_summaries", []),
                "protocol_note": m11.get("protocol_note", ""),
            },
        },
        "s4m_official_baseline": {
            "pass": bool(m12.get("status") == "PASS_OFFICIAL_S4M_ADAPTATION"),
            "required": "official S4M architecture adaptation included as a contrastive missing-aware baseline",
            "observed": {
                "status": m12.get("status", "not_run"),
                "backbone": m12.get("backbone", ""),
                "s4m_revision": m12.get("s4m_revision", ""),
                "mechanism_shift_gate": m12.get("mechanism_shift_gate", False),
                "datasets": [
                    {
                        "dataset": row.get("dataset"),
                        "max_relative_degradation": row.get("max_relative_degradation"),
                        "max_relative_degradation_ci": row.get("max_relative_degradation_ci"),
                        "kruskal_p": row.get("kruskal_p"),
                        "kruskal_p_ci": row.get("kruskal_p_ci"),
                        "gate_pass_count": row.get("gate_pass_count"),
                        "n_seeds": row.get("n_seeds"),
                        "gate_pass": row.get("gate_pass"),
                    }
                    for row in m12.get("datasets", [])
                ],
                "protocol_note": m12.get("protocol_note", ""),
            },
        },
        "hierarchical_bootstrap": {
            "pass": bool(m13.get("status") == "PASS_HIERARCHICAL_BOOTSTRAP"),
            "required": "hierarchy-aware uncertainty check over variants and windows for M1 aggregate claims",
            "observed": {
                "status": m13.get("status", "not_run"),
                "primary_effect_gate_count": m13.get("primary_effect_gate_count", 0),
                "dataset_summaries": [
                    {
                        "dataset": row.get("dataset"),
                        "max_absolute_delta": row.get("max_absolute_delta"),
                        "positive_delta_probability": row.get("positive_delta_probability"),
                        "loss_shift_supported": row.get("loss_shift_supported"),
                        "rank_instability_probability": row.get("rank_instability_probability"),
                        "rank_instability_supported": row.get("rank_instability_supported"),
                        "primary_gate": row.get("primary_gate"),
                    }
                    for row in m13.get("dataset_summaries", [])
                ],
                "protocol_note": m13.get("protocol_note", ""),
            },
        },
        "s4m_scale_validation": {
            "pass": bool(m14.get("status") == "PASS_S4M_SCALE_VALIDATION"),
            "required": "larger reduced S4M run to test whether the M12 negative/contrastive result is only an eight-channel artifact",
            "observed": {
                "status": m14.get("status", "not_run"),
                "backbone": m14.get("backbone", ""),
                "config": m14.get("config", {}),
                "mechanism_shift_gate": m14.get("mechanism_shift_gate", False),
                "datasets": [
                    {
                        "dataset": row.get("dataset"),
                        "max_relative_degradation": row.get("max_relative_degradation"),
                        "max_relative_degradation_ci": row.get("max_relative_degradation_ci"),
                        "kruskal_p": row.get("kruskal_p"),
                        "kruskal_p_ci": row.get("kruskal_p_ci"),
                        "gate_pass_count": row.get("gate_pass_count"),
                        "n_seeds": row.get("n_seeds"),
                        "gate_pass": row.get("gate_pass"),
                    }
                    for row in m14.get("datasets", [])
                ],
                "protocol_note": m14.get("protocol_note", ""),
            },
        },
        "final_integrity": {
            "pass": bool(m15.get("verdict") == "PASS_FINAL_INTEGRITY" and not m15.get("blocking_issues", [])),
            "required": "M15 final local integrity audit must pass with zero blocking issues",
            "observed": {
                "verdict": m15.get("verdict", "not_run"),
                "blocking_issues": m15.get("blocking_issues", []),
                "issue_count": len(m15.get("issues", [])),
                "scope_note": m15.get("scope_note", ""),
            },
        },
        "official_full_dataset_coverage": {
            "pass": bool(m16.get("status") == "PASS_OFFICIAL_TSLIB_FULL_COVERAGE" and m16.get("coverage_complete")),
            "required": "official PatchTST/TimeXer coverage over all four MaskShift datasets, including mixed/negative Traffic and AirConvection",
            "observed": {
                "status": m16.get("status", "not_run"),
                "coverage_complete": m16.get("coverage_complete", False),
                "gate_pass_count": m16.get("gate_pass_count", 0),
                "new_datasets": m16.get("new_datasets", []),
                "protocol_note": m16.get("protocol_note", ""),
            },
        },
        "submission_supplement": {
            "pass": bool(
                m17.get("status") == "PASS_SUBMISSION_SUPPLEMENT"
                and all(exists(path) for path in m17.get("artifacts", []) if not path.endswith("submission_supplement_summary.json"))
            ),
            "required": "M17 reviewer-facing supplement package with dataset cards, rebuttal playbook, response matrix, supplement TeX, and supplement PDF",
            "observed": {
                "status": m17.get("status", "not_run"),
                "reviewer_concern_count": m17.get("reviewer_concern_count", 0),
                "coverage": m17.get("coverage", {}),
                "supplement_pdf_size": m17.get("supplement_pdf_size", 0),
                "scope_note": m17.get("scope_note", ""),
            },
            "missing": [path for path in m17.get("artifacts", []) if not exists(path)],
        },
        "submission_policy_pack": {
            "pass": bool(
                m18.get("status") == "PASS_SUBMISSION_POLICY_PACK"
                and all(exists(path) for path in m18.get("artifacts", []) if not path.endswith("submission_policy_summary.json"))
                and not m18.get("blocking_policy_gaps", [])
            ),
            "required": "M18 target-agnostic submission policy pack with disclosure, ethics, reproducibility, anonymity, venue-readiness audit, TeX, and PDF",
            "observed": {
                "status": m18.get("status", "not_run"),
                "statement_count": m18.get("statement_count", 0),
                "target_specific_template_status": m18.get("target_specific_template_status", ""),
                "submission_statements_pdf_size": m18.get("submission_statements_pdf_size", 0),
                "scope_note": m18.get("scope_note", ""),
            },
            "missing": [path for path in m18.get("artifacts", []) if not exists(path)],
        },
        "aaai27_target_dossier": {
            "pass": bool(
                m19.get("status") == "PASS_AAAI27_TARGET_DOSSIER"
                and all(exists(path) for path in m19.get("artifacts", []) if not path.endswith("aaai27_target_readiness_summary.json"))
            ),
            "required": "M19 AAAI-27 target-specific readiness dossier, reproducibility checklist draft, phase-review response plan, gap table, TeX, and PDF",
            "observed": {
                "status": m19.get("status", "not_run"),
                "aaai27_upload_ready": m19.get("aaai27_upload_ready", False),
                "upload_blockers": [row.get("item") for row in m19.get("upload_blockers", [])],
                "aaai27_readiness_pdf_size": m19.get("aaai27_readiness_pdf_size", 0),
                "scope_note": m19.get("scope_note", ""),
            },
            "missing": [path for path in m19.get("artifacts", []) if not exists(path)],
        },
        "aaai27_preflight_conversion": {
            "pass": bool(
                m20.get("status") == "PASS_AAAI27_PREFLIGHT_CONVERSION"
                and m20.get("page_limit_preflight_pass", False)
                and all(exists(path) for path in m20.get("artifacts", []) if not path.endswith("aaai27_preflight_summary.json"))
            ),
            "required": "M20 two-column US-letter anonymous AAAI-style preflight with <=7 pages, table-fit evidence, TeX, PDF, and explicit official-kit boundary",
            "observed": {
                "status": m20.get("status", "not_run"),
                "page_limit_preflight_pass": m20.get("page_limit_preflight_pass", False),
                "aaai_official_kit_upload_ready": m20.get("aaai_official_kit_upload_ready", False),
                "official_template_build_pass": m20.get("official_template_build_pass", False),
                "state": m20.get("state", {}),
                "scope_note": m20.get("scope_note", ""),
            },
            "missing": [path for path in m20.get("artifacts", []) if not exists(path)],
        },
        "aaai27_reproducibility_checklist": {
            "pass": bool(
                m21.get("status") == "PASS_AAAI27_REPRODUCIBILITY_CHECKLIST"
                and m21.get("remaining_question_placeholders", 1) == 0
                and all(exists(path) for path in m21.get("artifacts", []) if not path.endswith("aaai27_reproducibility_checklist_summary.json"))
            ),
            "required": "M21 filled local official AAAI-27 reproducibility checklist TeX/PDF with zero question placeholders",
            "observed": {
                "status": m21.get("status", "not_run"),
                "answer_count": m21.get("answer_count", 0),
                "remaining_question_placeholders": m21.get("remaining_question_placeholders", None),
                "pdf_size": m21.get("pdf_size", 0),
                "scope_note": m21.get("scope_note", ""),
            },
            "missing": [path for path in m21.get("artifacts", []) if not exists(path)],
        },
    }
    main_track_ready = (
        gates["m0_matched_rate_controls"]["pass"]
        and gates["h1_mechanism_factor_fdr"]["pass"]
        and gates["h2_rank_instability"]["pass"]
        and gates["artifact_package"]["pass"]
        and gates["claim_scope_safe"]["pass"]
        and gates["deep_backbone_sweep"]["pass"]
        and gates["severity_curves"]["pass"]
        and gates["non_retirement_decomposition"]["pass"]
        and gates["official_baseline_reproduction"]["pass"]
        and gates["submission_hardening"]["pass"]
        and gates["missing_aware_official_baseline"]["pass"]
        and gates["s4m_official_baseline"]["pass"]
        and gates["hierarchical_bootstrap"]["pass"]
        and gates["s4m_scale_validation"]["pass"]
        and gates["final_integrity"]["pass"]
        and gates["official_full_dataset_coverage"]["pass"]
        and gates["submission_supplement"]["pass"]
        and gates["submission_policy_pack"]["pass"]
        and gates["aaai27_target_dossier"]["pass"]
        and gates["aaai27_preflight_conversion"]["pass"]
        and gates["aaai27_reproducibility_checklist"]["pass"]
    )
    # The typed correction is not mandatory for a benchmark/theory route, but it
    # is mandatory if the paper claims a method contribution.
    method_claim_ready = main_track_ready and gates["h3_typed_correction"]["pass"]
    blocking = [name for name, gate in gates.items() if not gate["pass"] and name != "h3_typed_correction"]
    route_lower = m4["paper_route"].lower()
    typed_is_primary = "typed correction secondary" not in route_lower and "method" in route_lower
    if typed_is_primary and not gates["h3_typed_correction"]["pass"]:
        blocking.append("h3_typed_correction")

    next_actions = []
    if not gates["deep_backbone_sweep"]["pass"]:
        next_actions.append("Implement M6 deep-backbone sweep with PatchTST plus one channel-dependent Transformer and one missing-specific baseline.")
    if not gates["official_baseline_reproduction"]["pass"]:
        next_actions.append("Run M9 official TSLib PatchTST/TimeXer reproduction or add S4M/ChannelTokenFormer-compatible baselines before final submission.")
    if not gates["severity_curves"]["pass"]:
        next_actions.append("Run M7 multi-rate severity curves and include degradation AUC in the paper.")
    if not gates["non_retirement_decomposition"]["pass"]:
        next_actions.append("Run M8 non-retirement decomposition so reviewers cannot dismiss the result as retirement-only.")
    if not gates["submission_hardening"]["pass"]:
        next_actions.append("Run M10 submission hardening to add seed CIs, corrected severity metrics, and claim-evidence tables.")
    if not gates["missing_aware_official_baseline"]["pass"]:
        next_actions.append("Integrate and run an official missing-aware architecture such as ChannelTokenFormer_missing or S4M.")
    if not gates["s4m_official_baseline"]["pass"]:
        next_actions.append("Integrate S4M as an official missing-aware contrastive baseline, even if the outcome is negative.")
    if not gates["hierarchical_bootstrap"]["pass"]:
        next_actions.append("Run M13 hierarchical bootstrap over variants/windows and report uncertainty limits for aggregate M1 claims.")
    if not gates["s4m_scale_validation"]["pass"]:
        next_actions.append("Run M14 larger reduced S4M validation so the S4M conclusion is not only an eight-channel artifact.")
    if not gates["final_integrity"]["pass"]:
        next_actions.append("Run M15 final integrity audit and resolve all blocking citation, traceability, artifact, and code-integrity issues.")
    if not gates["official_full_dataset_coverage"]["pass"]:
        next_actions.append("Run M16 official TSLib full-dataset coverage so Traffic/AirConvection are not only lightweight-model evidence.")
    if not gates["submission_supplement"]["pass"]:
        next_actions.append("Run M17 submission supplement packaging and compile paper/supplement.pdf before reporting final submission readiness.")
    if not gates["submission_policy_pack"]["pass"]:
        next_actions.append("Run M18 submission policy pack and compile paper/submission_statements.pdf before reporting final submission readiness.")
    if not gates["aaai27_target_dossier"]["pass"]:
        next_actions.append("Run M19 AAAI-27 target-readiness dossier and compile paper/aaai27_readiness.pdf before reporting final target readiness.")
    if not gates["aaai27_preflight_conversion"]["pass"]:
        next_actions.append("Run M20 AAAI-27 two-column preflight conversion and compile paper/aaai27_preflight.pdf before reporting page-pressure readiness.")
    if not gates["aaai27_reproducibility_checklist"]["pass"]:
        next_actions.append("Run M21 AAAI-27 reproducibility checklist fill and compile paper/aaai27_reproducibility_checklist.pdf before reporting venue-form readiness.")
    if not gates["h1_mechanism_factor_fdr"]["pass"] or not gates["h2_rank_instability"]["pass"]:
        next_actions.append("Expand datasets/seeds or adjust mechanisms; do not submit until H1/H2 are FDR-backed.")
    typed_weakened = "negative diagnostic" in paper_text and "not a new forecasting backbone" in paper_text
    if not gates["h3_typed_correction"]["pass"] and not typed_weakened:
        next_actions.append("Either improve typed correction or weaken it to a secondary diagnostic in the paper.")

    summary = {
        "milestone": "M5",
        "verdict": "STRONG_CONFERENCE_READY" if main_track_ready else "NOT_SUBMISSION_READY_YET",
        "method_claim_ready": bool(method_claim_ready),
        "main_track_ready": bool(main_track_ready),
        "aaai27_upload_ready": bool(m19.get("aaai27_upload_ready", False) and m20.get("aaai_official_kit_upload_ready", False)),
        "aaai27_preflight_ready": bool(m20.get("status") == "PASS_AAAI27_PREFLIGHT_CONVERSION"),
        "gates": gates,
        "blocking_items": blocking,
        "next_actions": next_actions,
    }
    write_json(OUT_DIR / "main_track_audit.json", summary)
    print(json.dumps({"milestone": "M5", "verdict": summary["verdict"], "blocking_items": blocking}, indent=2))


if __name__ == "__main__":
    main()
