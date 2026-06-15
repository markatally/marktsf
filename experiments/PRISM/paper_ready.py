"""M5 paper-ready artifact packager for PRISM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def load_optional(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def pass_count(summary: dict | None) -> int:
    return sum(1 for row in summary.get("rows", []) if row.get("gate_pass")) if summary else 0


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def write_m1c_summary(root: Path) -> dict[str, object]:
    rows = []
    for dataset in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Exchange"]:
        run_dir = root / f"oracle_drift/M1C_{dataset}_L96_H96_target_last"
        summary = load(run_dir / "summary.json")
        online = load(run_dir / "online_learning_summary.json")
        probe = load(run_dir / "descriptor_probe_summary.json")
        rows.append(
            {
                "dataset": dataset,
                "best_single": summary["best_single_model"],
                "best_loss": summary["best_single_loss"],
                "oracle_gap_rel": summary["oracle_gap_rel"],
                "switch_ratio": summary["switch_ratio"],
                "median_streak": summary["median_streak"],
                "max_streak": summary["max_streak"],
                "best_vs_anchor_pct": summary["best_vs_anchor_pct"],
                "fs_gap_recovered_frac": online["best_fixed_share"]["gap_recovered_frac"],
                "probe_accuracy": probe["probe_accuracy"],
                "probe_baseline_accuracy": probe["marginal_baseline_accuracy"],
                "probe_accuracy_lift": probe["accuracy_lift"],
            }
        )
    result = {
        "scope": "M1c lightweight breadth oracle and descriptor-probe summary regenerated from per-dataset artifacts.",
        "rows": rows,
    }
    (root / "oracle_drift/m1c_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run(args: argparse.Namespace) -> dict[str, object]:
    root = args.prism_root
    write_m1c_summary(root)
    m2 = load(root / "router_viability/router_viability_summary.json")
    m3 = load(root / "drift_beta_loop/drift_beta_summary.json")
    m4 = load(root / "ablations_significance/ablations_significance_summary.json")
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
    full_fdr = {
        item["dataset"]: item["fdr_pass"]
        for item in m4["fdr_hypotheses"]
        if item["comparison"] == "full_vs_plain"
    }
    full_vs_validation_fdr = {
        item["dataset"]: item["fdr_pass"]
        for item in m4["fdr_hypotheses"]
        if item["comparison"] == "full_vs_validation_single"
    }

    summary = {
        "milestone": "M5",
        "status": "candidate-main-route-after-M17",
        "main_track_status": (
            "scoped selective main route found after M17: practical-effect horizon-wise affine activation passes "
            f"with active {m17['active_count']}/{len(m17['rows'])} cells and inactive no-harm"
            if m17 and m17.get("gate_pass")
            else "candidate narrowed route remains blocked; M17 practical selective gate is missing or failed"
        ),
        "final_route": "Scoped practical-effect selective horizon-wise affine calibration over non-financial sensor/infrastructure forecasting; Exchange retained as explicit negative/out-of-scope diagnostic",
        "headline_claims": [
            "Optimal-bias drift is broad in the ETT/Weather lightweight screen.",
            "Strong validation-tuned Fixed-Share is the robust causal tracker after learned-router failure.",
            "Dynamic beta/drift-loop improves stressed loss on several datasets after delayed-feedback correction, but no full-vs-plain comparison survives block/FDR.",
            "The H=192 router pilot passes only 1/6 datasets, so the learned-router failure is not confined to H=96.",
            "Expanded expert-pool pilots improve validation-single baselines but still fail the current router gates.",
            "Champion-risk safe-switch pilots mostly collapse to validation-single; they do not provide a replacement main method.",
            "Calibrated forecast stacking is the strongest rescue attempt so far but still misses the all-battlefield main-track gate.",
            "A narrowed non-financial M11 route passes 7/7 datasets at both H=96 and H=192 after adding Electricity and Traffic.",
            "Horizon-wise affine stacking improves calibrated-stack effect sizes, but M12 block/FDR significance is still not sufficient versus validation-single and descriptor ridge.",
            "M13 adds PEMS04/PEMS08 as high-dimensional sensor confirmations; the route is close but still fails strict block/FDR.",
            "M14 delayed online stacker portfolio clears Fixed-Share/descriptor-ridge but still fails validation-single FDR.",
            "M15 fixed horizon-wise affine stacking is the cleanest current sensor-route candidate, but Electricity H192 remains a validation-single blocker.",
            "M16 selective horizon-wise affine no-harm gate passes on active cells but is under-covered at 2/8 active cells.",
            (
                "M17 practical-effect selective horizon-wise affine gate passes after adding Wind, AQShunyi, AQWan, and METR-LA: "
                f"active {m17['active_count']}/{len(m17['rows'])}, all active cells pass FDR vs validation-single, delayed Fixed-Share, and descriptor ridge, and inactive cells abstain to validation-single."
                if m17 and m17.get("gate_pass")
                else "M17 practical-effect selective horizon-wise affine gate is not available as a positive claim."
            ),
            "M17 threshold sensitivity is narrow and must be reported: min-effect 0% and 2.5% fail because fragile active cells do not pass Fixed-Share FDR; 10% fails active-coverage; 5% passes.",
            "The learned router, dynamic beta, and drift-share-rate loop are not sufficient for a main-track method claim in current form.",
        ],
        "gate_status": {
            "M1a": "HOLD/PASS for ETT phenomenon; finance raw-return MSE void",
            "M1b": "FAIL finance strict gate; pivot to ETT-only PRISM",
            "M1c": "PASS breadth phenomenon; routability mixed",
            "M2": "FAIL learned-router viability on the expanded delayed-feedback battlefield with validation-single baseline",
            "M3": "PASS" if m3["gate_pass"] else "FAIL under strengthened plain-FS baseline",
            "M4": "PASS" if m4["gate_pass"] else "FAIL under strengthened block/FDR ablation gate",
            "M5": "PASS as reproducible candidate-route artifact; does not clear main-track submission",
            "M7": "FAIL H=192 router pilot passes only 1/6 datasets",
            "M8": "FAIL expanded-pool router pilots pass 0/6 at H=96 and 1/6 at H=192",
            "M9": (
                "FAIL champion-risk safe-switch pilots "
                f"(base H96 {pass_count(m9)}/6, base H192 {pass_count(m9_h192)}/6, "
                f"expanded H96 {pass_count(m9_expanded_h96)}/6, expanded H192 {pass_count(m9_expanded_h192)}/6)"
            ),
            "M10": (
                "NEAR-MISS/FAIL calibrated forecast stacking "
                f"(base H96 {pass_count(m10)}/6, base H192 {pass_count(m10_h192)}/6, "
                f"expanded H96 {pass_count(m10_expanded_h96)}/6, expanded H192 {pass_count(m10_expanded_h192)}/6)"
            ),
            "M11": (
                "PASS candidate narrowed non-financial route "
                f"(H96 {m11['h96']['pass_count']}/{m11['h96']['total']}, H192 {m11['h192']['pass_count']}/{m11['h192']['total']})"
                if m11
                else "MISSING narrowed non-financial route audit"
            ),
            "M12": (
                "FAIL block/FDR calibrated-stack significance "
                f"(validation-single {m12['fdr_pass_counts']['stack_vs_validation_single']}/14, "
                f"Fixed-Share {m12['fdr_pass_counts']['stack_vs_fixed_share']}/14, "
                f"descriptor ridge {m12['fdr_pass_counts']['stack_vs_descriptor_ridge']}/14)"
                if m12
                else "MISSING calibrated-stack block/FDR significance"
            ),
            "M13": (
                "FAIL high-dimensional sensor-stack significance "
                f"(validation-single {m13['fdr_pass_counts']['stack_vs_validation_single']}/8, "
                f"Fixed-Share {m13['fdr_pass_counts']['stack_vs_fixed_share']}/8, "
                f"descriptor ridge {m13['fdr_pass_counts']['stack_vs_descriptor_ridge']}/8)"
                if m13
                else "MISSING high-dimensional sensor-stack significance"
            ),
            "M14": (
                "FAIL delayed online stacker portfolio "
                f"(validation-single {m14['fdr_pass_counts']['portfolio_vs_validation_single']}/8, "
                f"Fixed-Share {m14['fdr_pass_counts']['portfolio_vs_fixed_share']}/8, "
                f"descriptor ridge {m14['fdr_pass_counts']['portfolio_vs_descriptor_ridge']}/8)"
                if m14
                else "MISSING delayed online stacker portfolio"
            ),
            "M15": (
                "FAIL fixed horizon-wise affine sensor route "
                f"(validation-single {m15['fdr_pass_counts']['horizon_affine_vs_validation_single']}/8, "
                f"Fixed-Share {m15['fdr_pass_counts']['horizon_affine_vs_fixed_share']}/8, "
                f"descriptor ridge {m15['fdr_pass_counts']['horizon_affine_vs_descriptor_ridge']}/8)"
                if m15
                else "MISSING fixed horizon-wise affine sensor route"
            ),
            "M16": (
                "FAIL selective horizon-wise affine no-harm gate "
                f"(active {m16['active_count']}/8, inactive no-harm={m16['inactive_no_harm']}, "
                f"active validation-single {m16['active_fdr_pass_counts']['selective_active_vs_validation_single']}/{m16['active_count']})"
                if m16
                else "MISSING selective horizon-wise affine no-harm gate"
            ),
            "M17": (
                "PASS practical-effect selective horizon-wise affine gate "
                f"(active {m17['active_count']}/{len(m17['rows'])}, inactive no-harm={m17['inactive_no_harm']}, "
                f"active validation-single {m17['active_fdr_pass_counts']['selective_active_vs_validation_single']}/{m17['active_count']}, "
                f"active Fixed-Share {m17['active_fdr_pass_counts']['selective_active_vs_fixed_share']}/{m17['active_count']}, "
                f"active descriptor ridge {m17['active_fdr_pass_counts']['selective_active_vs_descriptor_ridge']}/{m17['active_count']})"
                if m17
                else "MISSING practical-effect selective horizon-wise affine gate"
            ),
        },
        "artifact_manifest": [
            "docs/PLAN.md",
            "docs/PROPOSAL.md",
            "docs/REPORT.md",
            "docs/INTEGRITY_AUDIT.md",
            "oracle_drift/m1c_summary.json",
            "router_viability/router_viability_summary.json",
            "router_viability_h192/router_viability_summary.json",
            "router_viability_expanded_h96/router_viability_summary.json",
            "router_viability_expanded_h192/router_viability_summary.json",
            "champion_risk_gate/champion_risk_gate_summary.json",
            "champion_risk_gate_h192/champion_risk_gate_summary.json",
            "champion_risk_gate_expanded_h96/champion_risk_gate_summary.json",
            "champion_risk_gate_expanded_h192/champion_risk_gate_summary.json",
            "calibrated_stack_gate/calibrated_stack_gate_summary.json",
            "calibrated_stack_gate_h192/calibrated_stack_gate_summary.json",
            "calibrated_stack_gate_expanded_h96/calibrated_stack_gate_summary.json",
            "calibrated_stack_gate_expanded_h192/calibrated_stack_gate_summary.json",
            "oracle_drift_nonfinancial/M1C_Electricity_L96_H96_target_last/summary.json",
            "oracle_drift_nonfinancial/M1C_Electricity_L96_H192_target_last/summary.json",
            "oracle_drift_nonfinancial/M1C_Traffic_L96_H96_target_last/summary.json",
            "oracle_drift_nonfinancial/M1C_Traffic_L96_H192_target_last/summary.json",
            "calibrated_stack_gate_nonfinancial_h96/calibrated_stack_gate_summary.json",
            "calibrated_stack_gate_nonfinancial_h192/calibrated_stack_gate_summary.json",
            "nonfinancial_stack_audit/nonfinancial_stack_audit.json",
            "calibrated_stack_significance/calibrated_stack_significance_summary.json",
            "oracle_drift_sensor/M1C_PEMS04_L96_H96_target_last/summary.json",
            "oracle_drift_sensor/M1C_PEMS04_L96_H192_target_last/summary.json",
            "oracle_drift_sensor/M1C_PEMS08_L96_H96_target_last/summary.json",
            "oracle_drift_sensor/M1C_PEMS08_L96_H192_target_last/summary.json",
            "calibrated_stack_gate_sensor_h96/calibrated_stack_gate_summary.json",
            "calibrated_stack_gate_sensor_h192/calibrated_stack_gate_summary.json",
            "sensor_stack_significance/sensor_stack_significance_summary.json",
            "online_stack_portfolio/online_stack_portfolio_summary.json",
            "sensor_horizon_affine_significance/sensor_horizon_affine_significance_summary.json",
            "selective_horizon_affine/selective_horizon_affine_summary.json",
            "oracle_drift_sensor_ext/M1C_Wind_L96_H96_target_last/summary.json",
            "oracle_drift_sensor_ext/M1C_Wind_L96_H192_target_last/summary.json",
            "oracle_drift_sensor_ext/M1C_AQShunyi_L96_H96_target_last/summary.json",
            "oracle_drift_sensor_ext/M1C_AQShunyi_L96_H192_target_last/summary.json",
            "oracle_drift_sensor_ext/M1C_AQWan_L96_H96_target_last/summary.json",
            "oracle_drift_sensor_ext/M1C_AQWan_L96_H192_target_last/summary.json",
            "oracle_drift_sensor_ext/M1C_METRLA_L96_H96_target_last/summary.json",
            "oracle_drift_sensor_ext/M1C_METRLA_L96_H192_target_last/summary.json",
            "practical_selective_horizon_affine/practical_selective_horizon_affine_summary.json",
            "practical_selective_horizon_affine_sensitivity_0/practical_selective_horizon_affine_summary.json",
            "practical_selective_horizon_affine_sensitivity_2p5/practical_selective_horizon_affine_summary.json",
            "practical_selective_horizon_affine_sensitivity_10/practical_selective_horizon_affine_summary.json",
            "submission_trace/submission_trace_summary.json",
            "submission_trace/SUBMISSION_TRACE.md",
            "submission_render/table1_m17_active_cells.csv",
            "submission_render/table1_m17_active_cells.md",
            "submission_render/table2_threshold_sensitivity.csv",
            "submission_render/table2_threshold_sensitivity.md",
            "submission_render/figure1_route_hardening.png",
            "submission_render/render_validation_summary.json",
            "drift_beta_loop/drift_beta_summary.json",
            "ablations_significance/ablations_significance_summary.json",
            "paper_ready/paper_ready_summary.json",
            "paper_ready/REPRODUCE.md",
            "main_track_audit/main_track_audit.json",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paper_ready_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    m2_rows = [
        [
            row["dataset"],
            f'{row["validation_single_loss"]:.6g} ({row["validation_single_model"]})',
            f'{row["fixed_share_loss"]:.6g}',
            f'{row["descriptor_ridge_loss"]:.6g}',
            f'{row["prism_router_loss"]:.6g}',
            "PASS" if row["gate_pass"] else "FAIL",
        ]
        for row in m2["rows"]
    ]
    m3_rows = [
        [
            row["dataset"],
            f'{row["plain_fixed_share_stress_loss"]:.6g}',
            f'{row["drift_loop_stress_loss"]:.6g}',
            f'{row["stress_improvement_pct"]:.3g}%',
            f'{row["beta_iqr"]:.3g}',
        ]
        for row in m3["rows"]
    ]
    m4_rows = [
        [
            row["dataset"],
            f'{row["losses"]["validation_single"]:.6g} ({row["validation_single"]["model"]})',
            f'{row["losses"]["plain_fixed_share"]:.6g}',
            f'{row["losses"]["full"]:.6g}',
            f'{row["tests"]["full_vs_plain"]["improvement_pct"]:.3g}%',
            "PASS" if full_fdr.get(row["dataset"], False) else "FAIL",
            "PASS" if full_vs_validation_fdr.get(row["dataset"], False) else "FAIL",
        ]
        for row in m4["rows"]
    ]
    reproduce = f"""# PRISM Empirical Artifact and Main-Track Audit Reproduction

Run from the repository root with the bundled/scientific Python environment.

```bash
PY=${{PY:-python3}}
$PY -m experiments.PRISM.produce_m1c_predictions
for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange; do
  $PY -m experiments.PRISM.oracle_drift \\
    --results-root external/TSLib/results \\
    --output-dir experiments/PRISM/oracle_drift/M1C_${{ds}}_L96_H96_target_last \\
    --dataset M1C_${{ds}} --lookback 96 --horizon 96 \\
    --models RidgeCov TargetRidge Trend Seasonal EWM \\
    --target-channel -1 --include-anchors
  $PY -m experiments.PRISM.online_learning \\
    --losses-csv experiments/PRISM/oracle_drift/M1C_${{ds}}_L96_H96_target_last/window_losses.csv \\
    --output-dir experiments/PRISM/oracle_drift/M1C_${{ds}}_L96_H96_target_last
  $PY -m experiments.PRISM.descriptor_probe \\
    --oracle-dir experiments/PRISM/oracle_drift/M1C_${{ds}}_L96_H96_target_last \\
    --output-dir experiments/PRISM/oracle_drift/M1C_${{ds}}_L96_H96_target_last \\
    --dataset M1C_${{ds}}
done
$PY -m experiments.PRISM.router_viability
$PY -m experiments.PRISM.champion_risk_gate
$PY -m experiments.PRISM.calibrated_stack_gate
$PY -m experiments.PRISM.drift_beta_loop
$PY -m experiments.PRISM.ablations_significance
$PY -m experiments.PRISM.paper_ready
$PY -m experiments.PRISM.main_track_audit
$PY -m experiments.PRISM.produce_m1c_predictions --horizon 192 --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange
for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange; do
  $PY -m experiments.PRISM.oracle_drift \\
    --results-root external/TSLib/results \\
    --output-dir experiments/PRISM/oracle_drift/M1C_${{ds}}_L96_H192_target_last \\
    --dataset M1C_${{ds}} --lookback 96 --horizon 192 \\
    --models RidgeCov TargetRidge Trend Seasonal EWM \\
    --target-channel -1 --include-anchors
  $PY -m experiments.PRISM.descriptor_probe \\
    --oracle-dir experiments/PRISM/oracle_drift/M1C_${{ds}}_L96_H192_target_last \\
    --output-dir experiments/PRISM/oracle_drift/M1C_${{ds}}_L96_H192_target_last \\
    --dataset M1C_${{ds}} --horizon 192
done
$PY -m experiments.PRISM.router_viability --horizon 192 --output-dir experiments/PRISM/router_viability_h192
$PY -m experiments.PRISM.champion_risk_gate --horizon 192 --output-dir experiments/PRISM/champion_risk_gate_h192
$PY -m experiments.PRISM.calibrated_stack_gate --horizon 192 --output-dir experiments/PRISM/calibrated_stack_gate_h192
$PY -m experiments.PRISM.produce_m1c_predictions --pool expanded --results-root external/TSLib/results_prism_expanded --horizon 96 --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange
$PY -m experiments.PRISM.produce_m1c_predictions --pool expanded --results-root external/TSLib/results_prism_expanded --horizon 192 --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange
for h in 96 192; do
  for ds in ETTh1 ETTh2 ETTm1 ETTm2 Weather Exchange; do
    $PY -m experiments.PRISM.oracle_drift \\
      --results-root external/TSLib/results_prism_expanded \\
      --output-dir experiments/PRISM/oracle_drift_expanded/M1C_${{ds}}_L96_H${{h}}_target_last \\
      --dataset M1C_${{ds}} --lookback 96 --horizon ${{h}} \\
      --models RidgeCov TargetRidge Trend Seasonal EWM EWMFast EWMSlow SeasonalOffset SeasonalDrift DampedTrend MeanRevert MeanRevertSlow SeasonalEWM SeasonalTrend EWMTrend \\
      --target-channel -1 --include-anchors
    $PY -m experiments.PRISM.descriptor_probe \\
      --results-root external/TSLib/results_prism_expanded \\
      --oracle-dir experiments/PRISM/oracle_drift_expanded/M1C_${{ds}}_L96_H${{h}}_target_last \\
      --output-dir experiments/PRISM/oracle_drift_expanded/M1C_${{ds}}_L96_H${{h}}_target_last \\
      --dataset M1C_${{ds}} --horizon ${{h}}
  done
done
$PY -m experiments.PRISM.router_viability --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --output-dir experiments/PRISM/router_viability_expanded_h96
$PY -m experiments.PRISM.router_viability --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --horizon 192 --output-dir experiments/PRISM/router_viability_expanded_h192
$PY -m experiments.PRISM.champion_risk_gate --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --output-dir experiments/PRISM/champion_risk_gate_expanded_h96
$PY -m experiments.PRISM.champion_risk_gate --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --horizon 192 --output-dir experiments/PRISM/champion_risk_gate_expanded_h192
$PY -m experiments.PRISM.calibrated_stack_gate --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --output-dir experiments/PRISM/calibrated_stack_gate_expanded_h96
$PY -m experiments.PRISM.calibrated_stack_gate --oracle-root experiments/PRISM/oracle_drift_expanded --results-root external/TSLib/results_prism_expanded --horizon 192 --output-dir experiments/PRISM/calibrated_stack_gate_expanded_h192
$PY -m experiments.PRISM.produce_m1c_predictions --datasets Electricity Traffic --pool expanded --results-root external/TSLib/results_prism_nonfinancial --horizon 96 --max-covariates 64
$PY -m experiments.PRISM.produce_m1c_predictions --datasets Electricity Traffic --pool expanded --results-root external/TSLib/results_prism_nonfinancial --horizon 192 --max-covariates 64
for h in 96 192; do
  for ds in Electricity Traffic; do
    $PY -m experiments.PRISM.oracle_drift \\
      --results-root external/TSLib/results_prism_nonfinancial \\
      --output-dir experiments/PRISM/oracle_drift_nonfinancial/M1C_${{ds}}_L96_H${{h}}_target_last \\
      --dataset M1C_${{ds}} --lookback 96 --horizon ${{h}} \\
      --models RidgeCov TargetRidge Trend Seasonal EWM EWMFast EWMSlow SeasonalOffset SeasonalDrift DampedTrend MeanRevert MeanRevertSlow SeasonalEWM SeasonalTrend EWMTrend \\
      --target-channel -1 --include-anchors
    $PY -m experiments.PRISM.descriptor_probe \\
      --results-root external/TSLib/results_prism_nonfinancial \\
      --oracle-dir experiments/PRISM/oracle_drift_nonfinancial/M1C_${{ds}}_L96_H${{h}}_target_last \\
      --output-dir experiments/PRISM/oracle_drift_nonfinancial/M1C_${{ds}}_L96_H${{h}}_target_last \\
      --dataset M1C_${{ds}} --horizon ${{h}}
  done
done
$PY -m experiments.PRISM.calibrated_stack_gate --oracle-root experiments/PRISM/oracle_drift_nonfinancial --results-root external/TSLib/results_prism_nonfinancial --output-dir experiments/PRISM/calibrated_stack_gate_nonfinancial_h96 --datasets Electricity Traffic --horizon 96
$PY -m experiments.PRISM.calibrated_stack_gate --oracle-root experiments/PRISM/oracle_drift_nonfinancial --results-root external/TSLib/results_prism_nonfinancial --output-dir experiments/PRISM/calibrated_stack_gate_nonfinancial_h192 --datasets Electricity Traffic --horizon 192
$PY -m experiments.PRISM.nonfinancial_stack_audit
$PY -m experiments.PRISM.calibrated_stack_significance
$PY -m experiments.PRISM.produce_m1c_predictions --datasets PEMS04 PEMS08 --pool expanded --results-root external/TSLib/results_prism_sensor --horizon 96 --max-covariates 64 --shared-context
$PY -m experiments.PRISM.produce_m1c_predictions --datasets PEMS04 PEMS08 --pool expanded --results-root external/TSLib/results_prism_sensor --horizon 192 --max-covariates 64 --shared-context
for h in 96 192; do
  for ds in PEMS04 PEMS08; do
    $PY -m experiments.PRISM.oracle_drift \\
      --results-root external/TSLib/results_prism_sensor \\
      --output-dir experiments/PRISM/oracle_drift_sensor/M1C_${{ds}}_L96_H${{h}}_target_last \\
      --dataset M1C_${{ds}} --lookback 96 --horizon ${{h}} \\
      --models RidgeCov TargetRidge Trend Seasonal EWM EWMFast EWMSlow SeasonalOffset SeasonalDrift DampedTrend MeanRevert MeanRevertSlow SeasonalEWM SeasonalTrend EWMTrend \\
      --target-channel -1 --include-anchors
    $PY -m experiments.PRISM.descriptor_probe \\
      --results-root external/TSLib/results_prism_sensor \\
      --oracle-dir experiments/PRISM/oracle_drift_sensor/M1C_${{ds}}_L96_H${{h}}_target_last \\
      --output-dir experiments/PRISM/oracle_drift_sensor/M1C_${{ds}}_L96_H${{h}}_target_last \\
      --dataset M1C_${{ds}} --horizon ${{h}}
  done
done
$PY -m experiments.PRISM.calibrated_stack_gate --oracle-root experiments/PRISM/oracle_drift_sensor --results-root external/TSLib/results_prism_sensor --output-dir experiments/PRISM/calibrated_stack_gate_sensor_h96 --datasets PEMS04 PEMS08 --horizon 96
$PY -m experiments.PRISM.calibrated_stack_gate --oracle-root experiments/PRISM/oracle_drift_sensor --results-root external/TSLib/results_prism_sensor --output-dir experiments/PRISM/calibrated_stack_gate_sensor_h192 --datasets PEMS04 PEMS08 --horizon 192
$PY -m experiments.PRISM.sensor_stack_significance
$PY -m experiments.PRISM.online_stack_portfolio
$PY -m experiments.PRISM.sensor_horizon_affine_significance
$PY -m experiments.PRISM.selective_horizon_affine_gate
$PY -m experiments.PRISM.produce_m1c_predictions --datasets Wind AQShunyi AQWan METRLA --pool expanded --results-root external/TSLib/results_prism_sensor_ext --horizon 96 --max-covariates 64 --shared-context
$PY -m experiments.PRISM.produce_m1c_predictions --datasets Wind AQShunyi AQWan METRLA --pool expanded --results-root external/TSLib/results_prism_sensor_ext --horizon 192 --max-covariates 64 --shared-context
for h in 96 192; do
  for ds in Wind AQShunyi AQWan METRLA; do
    $PY -m experiments.PRISM.oracle_drift \\
      --results-root external/TSLib/results_prism_sensor_ext \\
      --output-dir experiments/PRISM/oracle_drift_sensor_ext/M1C_${{ds}}_L96_H${{h}}_target_last \\
      --dataset M1C_${{ds}} --lookback 96 --horizon ${{h}} \\
      --models RidgeCov TargetRidge Trend Seasonal EWM EWMFast EWMSlow SeasonalOffset SeasonalDrift DampedTrend MeanRevert MeanRevertSlow SeasonalEWM SeasonalTrend EWMTrend \\
      --target-channel -1
    $PY -m experiments.PRISM.descriptor_probe \\
      --results-root external/TSLib/results_prism_sensor_ext \\
      --oracle-dir experiments/PRISM/oracle_drift_sensor_ext/M1C_${{ds}}_L96_H${{h}}_target_last \\
      --output-dir experiments/PRISM/oracle_drift_sensor_ext/M1C_${{ds}}_L96_H${{h}}_target_last \\
      --dataset M1C_${{ds}} --horizon ${{h}}
  done
done
$PY -m experiments.PRISM.practical_selective_horizon_affine_gate
$PY -m experiments.PRISM.practical_selective_horizon_affine_gate --min-effect-pct 0 --output-dir experiments/PRISM/practical_selective_horizon_affine_sensitivity_0
$PY -m experiments.PRISM.practical_selective_horizon_affine_gate --min-effect-pct 2.5 --output-dir experiments/PRISM/practical_selective_horizon_affine_sensitivity_2p5
$PY -m experiments.PRISM.practical_selective_horizon_affine_gate --min-effect-pct 10 --output-dir experiments/PRISM/practical_selective_horizon_affine_sensitivity_10
$PY -m experiments.PRISM.submission_render
$PY -m experiments.PRISM.submission_hardening
$PY -m experiments.PRISM.paper_ready
$PY -m experiments.PRISM.main_track_audit
```

## Final Route

The original PRISM learned-router method remains rejected after strengthened
baseline hardening.  The current positive route is a scoped practical-effect
selective horizon-wise affine calibration method, not the original routing-level
SOTA claim:

- M1b finance gate failed under the strict preregistered condition.
- M2 delayed contextual router fails once a validation-selected single-expert baseline is added.
- M3 dynamic beta/drift loop improves stressed loss on 4/6 datasets.
- M4 full-vs-plain and full-vs-validation-single do not clear the strengthened block/FDR gate.
- M7 H=192 router pilot passes only 1/6 datasets.
- M8 expanded expert pool improves validation-single baselines but current router passes 0/6 at H=96 and 1/6 at H=192.
- M9 champion-risk safe-switch does not rescue the main method: base H96/H192 pass 0/6, expanded H96 passes 0/6, expanded H192 passes 1/6.
- M10 calibrated forecast stacking is a strong near miss: base H96/H192 pass 5/6, expanded H96 passes 4/6, expanded H192 passes 5/6; Exchange remains the repeated blocker.
- M11 narrowed non-financial calibrated-stacking route passes 7/7 datasets at both H=96 and H=192 after adding Electricity and Traffic.
- M12 block/FDR significance remains incomplete after horizon-wise affine stacking: stack vs Fixed-Share passes 14/14, but stack vs validation-single passes 7/14 and stack vs descriptor ridge passes 10/14.
- M13 high-dimensional sensor route adds PEMS04/PEMS08 and remains close but incomplete: stack vs descriptor ridge passes 8/8, Fixed-Share 7/8, and validation-single 6/8.
- M14 delayed online stacker portfolio improves the sensor route to Fixed-Share 8/8 and descriptor ridge 8/8, but validation-single remains 6/8.
- M15 fixed horizon-wise affine stacking is the cleanest current candidate: Fixed-Share 8/8, descriptor ridge 8/8, validation-single 7/8. Electricity H192 remains the single strict blocker.
- M16 selective horizon-wise affine no-harm gate is honest but under-covered: active cells pass 2/2 against all baselines, inactive cells exactly abstain to validation-single, but active coverage is only 2/8.
- M17 practical-effect selective horizon-wise affine activation passes the scoped main-route gate: active cells pass against validation-single, delayed Fixed-Share, and descriptor ridge; inactive cells abstain to validation-single. The active cells are Electricity H96, Traffic H96, AQWan H96, and AQWan H192.
- Drift-triggered share-rate adaptation is rejected in the current form.

## M2 Router Viability

{md_table(["Dataset", "Validation Single", "Fixed-Share", "Descriptor Ridge", "PRISM Router", "Gate"], m2_rows)}

## M3 Dynamic Beta / Drift Stress

{md_table(["Dataset", "Plain Stress", "Loop Stress", "Improvement", "Beta IQR"], m3_rows)}

## M4 FDR Ablations

{md_table(["Dataset", "Validation Single", "Plain FS", "Full", "Full vs Plain", "Plain FDR", "Validation FDR"], m4_rows)}

## Synthetic Identifiability

- State recovery accuracy: {m4["synthetic_identifiability"]["state_accuracy"]:.3f}
- Best single loss: {m4["synthetic_identifiability"]["best_single_loss"]:.4f}
- Oracle loss: {m4["synthetic_identifiability"]["oracle_loss"]:.4f}
- Descriptor router loss: {m4["synthetic_identifiability"]["router_loss"]:.4f}
"""
    (args.output_dir / "REPRODUCE.md").write_text(reproduce)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create PRISM M5 paper-ready artifacts.")
    p.add_argument("--prism-root", type=Path, default=Path("experiments/PRISM"))
    p.add_argument("--output-dir", type=Path, default=Path("experiments/PRISM/paper_ready"))
    return p.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
