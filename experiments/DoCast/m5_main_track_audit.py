"""M5 — Main-track submission readiness audit.

This script is intentionally stricter than M4. M4 asks whether the evidence
chain is internally consistent. M5 asks whether the package is directly ready
for a main-track ML/TSF submission.
"""

from __future__ import annotations

import json
from pathlib import Path


EXP_DIR = Path(__file__).parent
OUT_DIR = EXP_DIR / "m5_main_track_audit"
OUT_DIR.mkdir(exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    m3 = load_json(EXP_DIR / "m3_real_data" / "real_data_summary.json")
    m4 = load_json(EXP_DIR / "m4_paper_ready" / "paper_ready_summary.json")
    m6_path = EXP_DIR / "m6_backbone_sweep" / "backbone_sweep_summary.json"
    m6 = load_json(m6_path) if m6_path.exists() else {}

    backbones = m4.get("backbones_evaluated", [])
    if m6.get("rows"):
        backbones = sorted(set(backbones + [
            f"{row['backbone']} (M6 full D0/D1/D2)" if row.get("protocol_pass") else f"{row['backbone']} (M6 compatibility)"
            for row in m6["rows"]
            if row.get("status") == "complete"
        ]))
    required_backbone_count = 3
    deep_backbone_names = ["TimeXer", "TFT", "TiDE", "NBEATSx", "PatchTST"]
    deep_backbones_done = [
        b for b in backbones
        if any(name.lower() in b.lower() for name in deep_backbone_names)
    ]

    real_a_type_legs = []
    fav = m3.get("h5_favorita_promo", {})
    fav_rob = m3.get("favorita_promo_robustness", {})
    if fav.get("h5_fav_pass") and fav_rob.get("robust_pass"):
        real_a_type_legs.append("Favorita promotion matched-ATT")
    markdown = m3.get("h5_m5_markdown", {})
    if markdown.get("h5_markdown_pass"):
        real_a_type_legs.append("M5 markdown matched-ATT")

    gates = {
        "internal_evidence_chain": {
            "pass": bool(m4.get("evidence_chain_greenlit", False)),
            "evidence": "M4 evidence_chain_greenlit",
        },
        "real_a_type_validation_robust": {
            "pass": bool(m3.get("h5_overall_pass", False) and fav_rob.get("robust_pass", False)),
            "evidence": {
                "favorita_d0_nee": fav.get("d0_nee"),
                "favorita_d2_nee": fav.get("d2_nee"),
                "median_robust_reduction": fav_rob.get("median_nee_reduction_frac"),
                "max_robust_p": fav_rob.get("max_unit_wilcoxon_p"),
            },
        },
        "sota_backbone_sweep": {
            "pass": bool(
                len(deep_backbones_done) >= required_backbone_count
                and m6.get("full_docast_protocol_complete", False)
            ),
            "required": f">= {required_backbone_count} deep covariate-aware TSF backbones with fair-control D0/D1/D2 DoCast protocol",
            "observed": backbones,
            "deep_backbones_detected": deep_backbones_done,
            "m6_compatibility_complete": m6.get("n_deep_backbones_complete", 0),
            "full_docast_protocol_complete": m6.get("full_docast_protocol_complete", False),
            "fairness": "D0/D1/D2 share item static controls; D1/D2 share item-specific response capacity",
        },
        "second_independent_real_a_type_leg": {
            "pass": len(real_a_type_legs) >= 2,
            "required": ">= 2 independent real controllable-covariate validation legs",
            "observed": real_a_type_legs,
        },
        "claim_scope_safe": {
            "pass": bool(
                "PRF is semi-synthetic only" in m4.get("claim_scope", "")
                and (
                    (
                        m6.get("full_docast_protocol_complete", False)
                        and "M6 completed the fair-control D0/D1/D2 protocol" in m4.get("claim_scope", "")
                    )
                    or "not a completed SOTA" in m4.get("claim_scope", "")
                )
            ),
            "evidence": m4.get("claim_scope", ""),
        },
    }

    ready = all(g["pass"] for g in gates.values())
    missing = [name for name, gate in gates.items() if not gate["pass"]]

    next_actions = []
    if not gates["sota_backbone_sweep"]["pass"]:
        next_actions.append(
            "Extend the M6 compatibility sweep into a fair-control D0/D1/D2 DoCast protocol for TimeXer, PatchTST, and TiDE."
        )
    if not gates["second_independent_real_a_type_leg"]["pass"]:
        next_actions.append(
            "Add a second independent real a-type validation leg or clearly narrow the venue/claim."
        )
    if not gates["real_a_type_validation_robust"]["pass"]:
        next_actions.append(
            "Strengthen real a-type validation robustness until H5 passes under the claim-family FDR."
        )

    summary = {
        "milestone": "M5",
        "main_track_ready": ready,
        "verdict": "REVISED_MAIN_TRACK_CANDIDATE" if ready else "NOT_DIRECTLY_SUBMITTABLE_YET",
        "gates": gates,
        "blocking_items": missing,
        "next_actions": next_actions,
    }

    out = OUT_DIR / "main_track_audit.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"M5 main-track audit → {out}")
    print(f"Verdict: {summary['verdict']}")
    if missing:
        print("Blocking gates:")
        for item in missing:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
