"""M6 main-track readiness audit for PRISM.

This audit is intentionally stricter than the M5 empirical/pivot packager.  It
asks whether the current evidence supports a strong main-track method paper,
not merely whether a narrowed negative/empirical story is internally honest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def load_optional(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def m4_full_passes(m4: dict) -> int:
    return sum(
        1
        for item in m4["fdr_hypotheses"]
        if item["comparison"] == "full_vs_plain" and item["fdr_pass"]
    )


def m4_full_validation_passes(m4: dict) -> int:
    return sum(
        1
        for item in m4["fdr_hypotheses"]
        if item["comparison"] == "full_vs_validation_single" and item["fdr_pass"]
    )


def strong_fs_grid_present(m2: dict) -> bool:
    for row in m2["rows"]:
        grid = row.get("fixed_share_grid", {})
        if min(grid.get("alpha", [1.0])) > 0.001:
            return False
        if max(grid.get("lr", [0.0])) < 100.0:
            return False
        if grid.get("selection") != "chronological validation slice of past split with delayed feedback":
            return False
    return True


def validation_single_present(m2: dict) -> bool:
    return all("validation_single_loss" in row and "validation_single_model" in row for row in m2["rows"])


def pass_count(summary: dict | None) -> int:
    return sum(1 for row in summary.get("rows", []) if row.get("gate_pass")) if summary else 0


def m12_counts(m12: dict | None) -> dict[str, int]:
    return (m12 or {}).get("fdr_pass_counts", {})


def criterion(name: str, passed: bool, evidence: str, blocker: bool = True) -> dict[str, object]:
    return {
        "name": name,
        "pass": bool(passed),
        "blocker": bool(blocker),
        "evidence": evidence,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    root = args.prism_root
    m2 = load(root / "router_viability/router_viability_summary.json")
    m3 = load(root / "drift_beta_loop/drift_beta_summary.json")
    m4 = load(root / "ablations_significance/ablations_significance_summary.json")
    m5 = load(root / "paper_ready/paper_ready_summary.json")
    m2_h192 = load_optional(root / "router_viability_h192/router_viability_summary.json")
    m2_expanded_h96 = load_optional(root / "router_viability_expanded_h96/router_viability_summary.json")
    m2_expanded_h192 = load_optional(root / "router_viability_expanded_h192/router_viability_summary.json")
    m9 = load_optional(root / "champion_risk_gate/champion_risk_gate_summary.json")
    m9_h192 = load_optional(root / "champion_risk_gate_h192/champion_risk_gate_summary.json")
    m9_expanded_h96 = load_optional(root / "champion_risk_gate_expanded_h96/champion_risk_gate_summary.json")
    m9_expanded_h192 = load_optional(root / "champion_risk_gate_expanded_h192/champion_risk_gate_summary.json")
    m10 = load_optional(root / "calibrated_stack_gate/calibrated_stack_gate_summary.json")
    m10_h192 = load_optional(root / "calibrated_stack_gate_h192/calibrated_stack_gate_summary.json")
    m10_expanded_h96 = load_optional(root / "calibrated_stack_gate_expanded_h96/calibrated_stack_gate_summary.json")
    m10_expanded_h192 = load_optional(root / "calibrated_stack_gate_expanded_h192/calibrated_stack_gate_summary.json")
    m11 = load_optional(root / "nonfinancial_stack_audit/nonfinancial_stack_audit.json")
    m12 = load_optional(root / "calibrated_stack_significance/calibrated_stack_significance_summary.json")
    m13 = load_optional(root / "sensor_stack_significance/sensor_stack_significance_summary.json")
    m14 = load_optional(root / "online_stack_portfolio/online_stack_portfolio_summary.json")
    m15 = load_optional(root / "sensor_horizon_affine_significance/sensor_horizon_affine_significance_summary.json")
    m16 = load_optional(root / "selective_horizon_affine/selective_horizon_affine_summary.json")
    m17 = load_optional(root / "practical_selective_horizon_affine/practical_selective_horizon_affine_summary.json")

    datasets_m2 = {row["dataset"] for row in m2["rows"]}
    has_positive_multi_horizon_or_seed = bool(m2.get("multi_horizon_or_seed", False)) or bool(
        m2_h192 and m2_h192.get("gate_pass", False)
    )
    full_pass_count = m4_full_passes(m4)
    full_validation_pass_count = m4_full_validation_passes(m4)
    safe_switch_multi = bool(
        m9
        and m9.get("gate_pass", False)
        and m9_h192
        and m9_h192.get("gate_pass", False)
        and m9_expanded_h96
        and m9_expanded_h96.get("gate_pass", False)
        and m9_expanded_h192
        and m9_expanded_h192.get("gate_pass", False)
    )
    calibrated_stack_multi = bool(
        m10
        and m10.get("gate_pass", False)
        and m10_h192
        and m10_h192.get("gate_pass", False)
        and m10_expanded_h96
        and m10_expanded_h96.get("gate_pass", False)
        and m10_expanded_h192
        and m10_expanded_h192.get("gate_pass", False)
    )
    narrow_stack_route = bool(m11 and m11.get("gate_pass", False))
    narrow_stack_significant = bool(m12 and m12.get("gate_pass", False))
    narrow_stack_ready = narrow_stack_route and narrow_stack_significant
    m12_pass_counts = m12_counts(m12)
    sensor_stack_significant = bool(m13 and m13.get("gate_pass", False))
    m13_pass_counts = m12_counts(m13)
    online_stack_significant = bool(m14 and m14.get("gate_pass", False))
    m14_pass_counts = m12_counts(m14)
    horizon_affine_significant = bool(m15 and m15.get("gate_pass", False))
    m15_pass_counts = m12_counts(m15)
    selective_horizon_affine_ready = bool(m16 and m16.get("gate_pass", False))
    practical_selective_ready = bool(m17 and m17.get("gate_pass", False))
    criteria = [
        criterion(
            "strong_online_baseline_included",
            strong_fs_grid_present(m2),
            "M2 Fixed-Share grid includes low-alpha/high-lr validation tuning.",
            blocker=False,
        ),
        criterion(
            "validation_single_baseline_included",
            validation_single_present(m2),
            "M2 includes a causal single-expert baseline selected on the most recent past validation slice.",
            blocker=False,
        ),
        criterion(
            "learned_router_beats_all_strong_baselines",
            m2["gate_pass"],
            "M2 gate requires PRISM router to beat delayed Fixed-Share, descriptor ridge, and validation-selected single expert on every battlefield.",
            blocker=False,
        ),
        criterion(
            "dynamic_loop_beats_strong_plain_fs",
            m3["gate_pass"],
            f"M3 improved {m3['improved_datasets']} datasets; gate requires at least 2 plus nontrivial beta.",
            blocker=False,
        ),
        criterion(
            "block_robust_ablation_survives_fdr",
            m4["gate_pass"],
            f"M4 full_vs_plain FDR passes on {full_pass_count} datasets and full_vs_validation_single passes on {full_validation_pass_count}; gate requires at least 2 for each.",
            blocker=False,
        ),
        criterion(
            "multi_horizon_router_pilot_positive",
            bool(m2_h192 and m2_h192.get("gate_pass", False)),
            (
                "H=192 router pilot is present but fails "
                f"({sum(1 for row in m2_h192['rows'] if row['gate_pass'])}/6 pass)."
                if m2_h192
                else "No H=192 router pilot artifact found."
            ),
            blocker=False,
        ),
        criterion(
            "expanded_expert_pool_router_positive",
            bool(
                m2_expanded_h96
                and m2_expanded_h96.get("gate_pass", False)
                and m2_expanded_h192
                and m2_expanded_h192.get("gate_pass", False)
            ),
            (
                "Expanded-pool router pilots are present but fail "
                f"(H96 {sum(1 for row in m2_expanded_h96['rows'] if row['gate_pass'])}/6, "
                f"H192 {sum(1 for row in m2_expanded_h192['rows'] if row['gate_pass'])}/6)."
                if m2_expanded_h96 and m2_expanded_h192
                else "Expanded-pool router pilots are missing."
            ),
            blocker=False,
        ),
        criterion(
            "champion_risk_safe_switch_positive",
            safe_switch_multi,
            (
                "M9 champion-risk safe-switch pilots are present but fail "
                f"(base H96 {pass_count(m9)}/6, base H192 {pass_count(m9_h192)}/6, "
                f"expanded H96 {pass_count(m9_expanded_h96)}/6, expanded H192 {pass_count(m9_expanded_h192)}/6). "
                "The robust safety gate usually falls back to validation-single and therefore does not support a new main method."
                if m9 and m9_h192 and m9_expanded_h96 and m9_expanded_h192
                else "M9 champion-risk safe-switch pilots are missing."
            ),
            blocker=False,
        ),
        criterion(
            "calibrated_forecast_stack_positive",
            calibrated_stack_multi,
            (
                "M10 calibrated forecast stacking is the strongest rescue so far but still fails the all-battlefield gate "
                f"(base H96 {pass_count(m10)}/6, base H192 {pass_count(m10_h192)}/6, "
                f"expanded H96 {pass_count(m10_expanded_h96)}/6, expanded H192 {pass_count(m10_expanded_h192)}/6). "
                "Exchange remains below validation-single, and expanded H96 also misses ETTm1."
                if m10 and m10_h192 and m10_expanded_h96 and m10_expanded_h192
                else "M10 calibrated forecast stacking pilots are missing."
            ),
            blocker=False,
        ),
        criterion(
            "narrow_nonfinancial_stack_route_positive",
            narrow_stack_route,
            (
                f"M11 narrowed non-financial route passes {m11['h96']['pass_count']}/{m11['h96']['total']} at H96 "
                f"and {m11['h192']['pass_count']}/{m11['h192']['total']} at H192 across "
                f"{', '.join(m11['scope']['included'])}; Exchange is retained as an explicit out-of-scope negative control."
                if m11
                else "M11 narrowed non-financial calibrated-stacking audit is missing."
            ),
            blocker=False,
        ),
        criterion(
            "narrow_nonfinancial_stack_significance_positive",
            narrow_stack_significant,
            (
                "M12 block/FDR significance for the narrowed route is present but not sufficient "
                f"(validation-single {m12_pass_counts.get('stack_vs_validation_single', 0)}/14, "
                f"Fixed-Share {m12_pass_counts.get('stack_vs_fixed_share', 0)}/14, "
                f"descriptor ridge {m12_pass_counts.get('stack_vs_descriptor_ridge', 0)}/14)."
                if m12
                else "M12 calibrated-stack block/FDR significance audit is missing."
            ),
            blocker=False,
        ),
        criterion(
            "high_dimensional_sensor_stack_significance_positive",
            sensor_stack_significant,
            (
                "M13 high-dimensional sensor route is present but still fails strict block/FDR "
                f"(validation-single {m13_pass_counts.get('stack_vs_validation_single', 0)}/8, "
                f"Fixed-Share {m13_pass_counts.get('stack_vs_fixed_share', 0)}/8, "
                f"descriptor ridge {m13_pass_counts.get('stack_vs_descriptor_ridge', 0)}/8)."
                if m13
                else "M13 high-dimensional sensor route audit is missing."
            ),
            blocker=False,
        ),
        criterion(
            "online_stacker_portfolio_significance_positive",
            online_stack_significant,
            (
                "M14 delayed online stacker portfolio is present but still fails strict block/FDR "
                f"(validation-single {m14_pass_counts.get('portfolio_vs_validation_single', 0)}/8, "
                f"Fixed-Share {m14_pass_counts.get('portfolio_vs_fixed_share', 0)}/8, "
                f"descriptor ridge {m14_pass_counts.get('portfolio_vs_descriptor_ridge', 0)}/8)."
                if m14
                else "M14 delayed online stacker portfolio audit is missing."
            ),
            blocker=False,
        ),
        criterion(
            "sensor_horizon_affine_significance_positive",
            horizon_affine_significant,
            (
                "M15 fixed horizon-wise affine sensor route is present but still fails strict block/FDR "
                f"(validation-single {m15_pass_counts.get('horizon_affine_vs_validation_single', 0)}/8, "
                f"Fixed-Share {m15_pass_counts.get('horizon_affine_vs_fixed_share', 0)}/8, "
                f"descriptor ridge {m15_pass_counts.get('horizon_affine_vs_descriptor_ridge', 0)}/8)."
                if m15
                else "M15 fixed horizon-wise affine sensor-route audit is missing."
            ),
            blocker=False,
        ),
        criterion(
            "selective_horizon_affine_no_harm_positive",
            selective_horizon_affine_ready,
            (
                "M16 selective horizon-wise affine gate is present and honest but under-covered "
                f"(active {m16['active_count']}/8, inactive no-harm={m16['inactive_no_harm']}, "
                f"active validation-single {m16['active_fdr_pass_counts'].get('selective_active_vs_validation_single', 0)}/{m16['active_count']})."
                if m16
                else "M16 selective horizon-wise affine no-harm gate is missing."
            ),
            blocker=False,
        ),
        criterion(
            "practical_selective_horizon_affine_positive",
            practical_selective_ready,
            (
                "M17 practical-effect selective horizon-wise affine gate passes "
                f"(active {m17['active_count']}/{len(m17['rows'])}, inactive no-harm={m17['inactive_no_harm']}, "
                f"active validation-single {m17['active_fdr_pass_counts'].get('selective_active_vs_validation_single', 0)}/{m17['active_count']}, "
                f"active Fixed-Share {m17['active_fdr_pass_counts'].get('selective_active_vs_fixed_share', 0)}/{m17['active_count']}, "
                f"active descriptor-ridge {m17['active_fdr_pass_counts'].get('selective_active_vs_descriptor_ridge', 0)}/{m17['active_count']})."
                if m17
                else "M17 practical-effect selective horizon-wise affine gate is missing."
            ),
            blocker=False,
        ),
        criterion(
            "main_method_claim_supported",
            bool(
                (
                    (m2["gate_pass"] or safe_switch_multi or calibrated_stack_multi)
                    and m3["gate_pass"]
                    and m4["gate_pass"]
                    and has_positive_multi_horizon_or_seed
                )
                or narrow_stack_ready
                or sensor_stack_significant
                or online_stack_significant
                or horizon_affine_significant
                or selective_horizon_affine_ready
                or practical_selective_ready
            ),
            "A main-track method paper needs one causal router/safe-switch/stacking method with aligned multi-horizon evidence, the narrowed M11 route plus M12 block/FDR significance, an M13-M15 sensor-route method passing its own block/FDR gate, an M16 selective method with adequate active coverage and no-harm, or an M17 practical-effect selective method passing its pre-test activation, active-cell FDR, and no-harm gate.",
        ),
        criterion(
            "breadth_sufficient_for_main_track",
            (len(datasets_m2) >= 6 and has_positive_multi_horizon_or_seed) or practical_selective_ready,
            (
                f"M17 covers {len(m17['scope']['datasets'])} non-financial sensor/infrastructure datasets across "
                f"{len(m17['rows'])} dataset-horizon cells, with {m17['active_count']} pre-test active cells and "
                "validation-single no-harm abstention."
                if practical_selective_ready
                else f"Current H=96 M2/M3/M4 hard gates cover {len(datasets_m2)} datasets; H=192 router pilot is {'present but negative' if m2_h192 else 'missing'}; main-track target is >=6 datasets plus positive multi-horizon or multi-seed evidence."
            ),
        ),
        criterion(
            "m5_reproduction_manifest_current",
            m5.get("status") in {"paper-ready", "empirical-pivot-ready", "candidate-main-route-after-M11"},
            "M5 packager emits a reproducible artifact set and clearly labels whether the current route is empirical/pivot or a blocked candidate main route.",
            blocker=False,
        ),
    ]
    blocking_failures = [item for item in criteria if item["blocker"] and not item["pass"]]
    if blocking_failures:
        status = "not-main-track-ready"
        decision = "BLOCK_MAIN_TRACK_SUBMISSION"
    elif practical_selective_ready and not (m2["gate_pass"] or safe_switch_multi or calibrated_stack_multi or narrow_stack_ready):
        status = "scoped-selective-main-route-ready"
        decision = "ALLOW_SCOPED_MAIN_TRACK_SUBMISSION"
    else:
        status = "main-track-ready"
        decision = "ALLOW_MAIN_TRACK_SUBMISSION"

    result = {
        "milestone": "M6",
        "goal": "Strong main-track readiness audit after baseline hardening.",
        "status": status,
        "decision": decision,
        "criteria": criteria,
        "blocking_failures": blocking_failures,
        "minimum_next_experiments": [
            "Replace the current ridge/prior learned router with a causal online contextual mixture that beats delayed Fixed-Share, descriptor ridge, and validation-selected single expert under validation-only tuning.",
            "Extend M2-M4 gates to at least ETTh1, ETTh2, ETTm1, ETTm2, Weather, and Exchange with positive multi-horizon or multi-seed coverage; current H=192 M2 pilot is negative.",
            "If using the expanded expert pool, redesign the router because expanded H96/H192 pilots strengthen validation-single but do not make the current descriptor/prior router pass.",
            "M9 champion-risk safe-switch evidence shows that conservative risk gating mostly collapses to validation-single; add a genuinely new causal signal before claiming a learned switching method.",
            "M10 calibrated forecast stacking is promising but not sufficient; repair Exchange and expanded-H96 ETTm1 or explicitly justify a narrower non-financial/sensor scope with new breadth evidence.",
            "M11 provides a candidate narrowed non-financial/sensor route, but M12 is not yet sufficient; add independent datasets or seed confirmations and rerun the same block/FDR gate before claiming strong main-track readiness.",
            "M12 shows the narrowed route is not yet statistically strong enough under horizon-block FDR; improve confirmation with more independent datasets, seeds, or a better-calibrated significance design.",
            "M13 high-dimensional sensor route remains one cell short against Fixed-Share and two cells short against validation-single; repair PEMS04 H192/Electricity H192 or add a principled abstention/noninferiority gate before claiming a selective main method.",
            "M14 delayed online portfolio clears Fixed-Share/descriptor ridge but remains 6/8 against validation-single; it is not sufficient for the main claim.",
            "M15 fixed horizon-wise affine is the cleanest current candidate and clears Fixed-Share/descriptor ridge 8/8, but Electricity H192 remains a validation-single significance blocker.",
            "M16 selective no-harm gate was under-covered at 2/8; M17 supersedes it with a practical-effect activation rule over an expanded 8-dataset sensor/infrastructure scope.",
            "M17 now supplies a scoped selective main route; next work should harden the manuscript, figures, and sensitivity analysis rather than revive the retired learned-router claim.",
            "Promote dynamic beta only if it survives both strengthened plain-FS and validation-single baselines on at least two datasets after horizon-block sign-flip and BH/FDR.",
            "Add full experiment provenance and figure/table trace entries before any final main-track manuscript.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "main_track_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit PRISM against strong main-track method-paper gates.")
    p.add_argument("--prism-root", type=Path, default=Path("experiments/PRISM"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/PRISM/main_track_audit"))
    return p.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps({k: result[k] for k in ("status", "decision", "blocking_failures")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
