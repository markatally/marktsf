"""Re-score frozen development predictions using BOOM's task aggregation rules.

This is deliberately not an official leaderboard: horizon, selected variables,
origins, and development split remain those of the historical R1 experiment.
The official low-variance split and shifted geometric mean are inspected from
a pinned local upstream file, not guessed from observed performance.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc

from r1.data import boom_development, sha256, calibration_atom
from r1.evaluation import forecast_scores, paired_cluster_interval


def official_low_variance(path):
    for statement in ast.parse(Path(path).read_text()).body:
        if isinstance(statement, ast.Assign) and any(isinstance(t, ast.Name) and t.id=="LOW_VARIANCE_DATASETS" for t in statement.targets):
            return set(ast.literal_eval(statement.value))
    raise ValueError("Upstream LOW_VARIANCE_DATASETS definition missing")


def shifted_gmean(values, epsilon=1e-5):
    values = np.asarray(values, float)
    if not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Invalid shifted geometric mean inputs")
    return float(np.exp(np.log(values+epsilon).mean())-epsilon)


def audit_aggregation(repo_root, output_dir, data_dir, foundation_predictions, toto_predictions,
                      refinement_runs, coupling_run, upstream_leaderboard,
                      runtime_python, horizon=32, max_tasks=48, max_variates=4):
    dest, root = Path(output_dir), Path(data_dir)
    dest.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(root/"cases.parquet")
    arrays = np.load(root/"arrays.npz")
    valid = np.flatnonzero(meta.partition.eq("validation"))
    records, _ = boom_development(repo_root, max_tasks, max_variates)
    lookup = {(s.dataset, s.item): s for s in records}
    cases = {}
    for index in valid:
        row = meta.iloc[index]
        key = (row["dataset"], row["item"], int(row.origin))
        s = lookup[key[:2]]
        if key[2]<s.train_end or key[2]+horizon>=len(s.values):
            raise ValueError("Audit origin outside frozen development validation")
        target = s.values[key[2]+1:key[2]+1+horizon]
        digest = hashlib.sha256(np.ascontiguousarray(target, dtype="<f8").tobytes()).hexdigest()
        cases[key] = dict(target=target, digest=digest, index=index, series=s)
    if len(cases)!=len(valid):
        raise ValueError("Duplicate validation key")
    predictions = {}
    def add(key, arm, seed, q, digest=None):
        if key not in cases or (digest is not None and digest!=cases[key]["digest"]):
            raise ValueError("Prediction target identity mismatch")
        entry = (key, arm, seed)
        if entry in predictions or np.asarray(q).shape!=(horizon, 9):
            raise ValueError("Duplicate or malformed forecast")
        predictions[entry] = np.asarray(q, float)
    for path in sorted(Path(foundation_predictions).glob("*.npz")):
        with np.load(path) as a:
            add((str(a["dataset"]), str(a["item"]), int(a["origin"])), "timesfm3", -1, a["quantiles"], str(a["target_hash"]))
    for path in sorted(Path(toto_predictions).glob("*.npz")):
        with np.load(path) as a:
            for key, q, digest in zip(a["keys"], a["quantiles"], a["target_hashes"], strict=True):
                add((str(key[0]), str(key[1]), int(key[2])), "toto2_313m", -1, q, str(digest))
    for run in refinement_runs:
        run = Path(run)
        seed = json.loads((run/"summary.json").read_text())["seed"]
        for arm in ("direct", "geometric", "semi_markov"):
            with np.load(run/f"{arm}_validation.npz") as a:
                if not np.array_equal(a["indices"], valid):
                    raise ValueError("Refinement case order changed")
                for index, q in zip(valid, a["quantiles"], strict=True):
                    row = meta.iloc[index]
                    add((row["dataset"], row["item"], int(row.origin)), "atom_"+arm, seed,
                        q*arrays["scale"][index]+arrays["atom"][index])
    for path in sorted(Path(coupling_run).glob("*_predictions.npz")):
        arm, seed, suffix = path.stem.rsplit("_", 2)
        with np.load(path) as a:
            if not np.array_equal(a["indices"], valid):
                raise ValueError("Coupling case order changed")
            for index, q in zip(valid, a["quantiles"], strict=True):
                row = meta.iloc[index]
                add((row["dataset"], row["item"], int(row.origin)), "coupling_"+arm, int(seed),
                    q*arrays["scale"][index]+arrays["atom"][index])
    inputs, outputs = dest/"naive_inputs", dest/"naive_predictions"
    inputs.mkdir(exist_ok=True)
    frequencies = {}
    for key, case in cases.items():
        s = case["series"]
        if s.dataset not in frequencies:
            path = sorted(Path(s.source).glob("*.arrow"))[0]
            with path.open("rb") as stream:
                batch = ipc.open_stream(stream).read_next_batch()
                frequencies[s.dataset] = batch.column(batch.schema.get_field_index("freq"))[0].as_py()
        history = s.values[:key[2]+1]
        np.savez_compressed(inputs/f'{case["index"]}.npz', context=history[-2048:], key=np.asarray(key),
                            target_hash=case["digest"], freq=frequencies[s.dataset])
    subprocess.run([runtime_python, str(Path(repo_root)/"src/r1/baseline_worker.py"), "--parameters",
        json.dumps(dict(input_dir=str(inputs), output_dir=str(outputs), horizon=horizon))],
        check=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE":"1"})
    seasons = {}
    for path in sorted(outputs.glob("*.npz")):
        with np.load(path) as a:
            key = (str(a["key"][0]), str(a["key"][1]), int(a["key"][2]))
            add(key, "seasonalnaive", -1, a["quantiles"], str(a["target_hash"]))
            seasons[key] = int(a["season"])
    # Mandatory cheap controls: a long constant validation segment can be won
    # by persistence without any duration model or probabilistic calibration.
    for key, case in cases.items():
        history = case["series"].values[:key[2]+1]
        add(key, "last_value", -1, np.full((horizon, 9), history[-1]))
        atom = calibration_atom(history)
        if atom is None:
            raise ValueError("Historical atom disappeared from the fixed atom pool")
        add(key, "historical_atom", -1, np.full((horizon, 9), atom))
    sizes = {}
    for key, arm, seed in predictions:
        sizes[arm, seed] = sizes.get((arm, seed), 0)+1
    if set(sizes.values())!={len(cases)}:
        raise ValueError("Models do not share the complete frozen case set")
    rows, coverage = [], []
    levels = np.arange(.1, 1, .1)
    for (key, arm, seed), q in predictions.items():
        case = cases[key]
        y = case["target"]
        s = case["series"]
        history = s.values[:key[2]+1]
        lag = seasons[key] if seasons[key]<=len(history) else 1
        seasonal_error = float(np.abs(history[lag:]-history[:-lag]).mean())
        score = forecast_scores(y, q[:, 4], q, levels)
        atom = calibration_atom(history)
        atom_mask = y==atom
        loss = 2*np.maximum((y[:, None]-q)*levels, (y[:, None]-q)*(levels-1)).mean(-1)
        mae = float(np.abs(y-q[:, 4]).mean())
        rows.append(dict(dataset=key[0], item=key[1], origin=key[2], arm=arm, seed=seed,
            numerator=score["quantile_loss_numerator"], denominator=score["absolute_target_denominator"],
            case_wql9=score["wql9"], mae=mae, mse=score["mse"],
            mase=mae/seasonal_error if seasonal_error else np.nan,
            atom_loss=float(loss[atom_mask].sum()), nonatom_loss=float(loss[~atom_mask].sum()),
            atom_points=int(atom_mask.sum()), points=len(y)))
        for i, level in enumerate(levels):
            coverage.append(dict(dataset=key[0], arm=arm, seed=seed, quantile=level,
                                 empirical_cdf=float(np.mean(y<=q[:, i]))))
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["dataset", "arm", "seed"]).agg(numerator=("numerator", "sum"),
        denominator=("denominator", "sum"), mase=("mase", "mean"), mae=("mae", "mean"),
        case_mean=("case_wql9", "mean"), atom_loss=("atom_loss", "sum"), nonatom_loss=("nonatom_loss", "sum"))
    grouped["wql9"] = grouped.numerator/grouped.denominator
    if not np.isfinite(grouped.wql9).all():
        raise ValueError("Undefined task WQL; no imputation during this audit")
    task = grouped.groupby(["dataset", "arm"])[["wql9", "mase", "mae", "case_mean"]].mean()
    naive = task.xs("seasonalnaive", level="arm")
    low = official_low_variance(upstream_leaderboard) | set(naive.index[naive.mase.eq(0)])
    table = task.reset_index()
    table["partition"] = np.where(table.dataset.isin(low), "low_variance_unscaled", "nonzero_scaled")
    table["scaled_wql9"] = table.wql9/table.dataset.map(naive.wql9)
    table["scaled_mase"] = table.mase/table.dataset.map(naive.mase)
    summary = {}
    for (partition, arm), group in table.groupby(["partition", "arm"]):
        key = "scaled_wql9" if partition=="nonzero_scaled" else "wql9"
        if not np.isfinite(group[key]).all():
            raise ValueError("Undefined naive ratio; must report instead of silently imputing")
        summary.setdefault(partition, {})[arm] = dict(tasks=len(group),
            shifted_geometric_wql9=shifted_gmean(group[key]), task_wql9_mean=float(group.wql9.mean()),
            historical_case_wql9_mean=float(group.case_mean.mean()))
    partition_comparisons = {}
    for partition, subtable in table.groupby("partition"):
        metric = "scaled_wql9" if partition=="nonzero_scaled" else "wql9"
        scores = subtable.pivot(index="dataset", columns="arm", values=metric)
        comparisons = {}
        for arm in ("atom_direct", "atom_geometric", "atom_semi_markov", "toto2_313m", "last_value", "historical_atom"):
            logs = np.log((scores[arm]+1e-5)/(scores.timesfm3+1e-5))
            interval = paired_cluster_interval(logs)
            leave_one = np.exp((logs.sum()-logs)/(len(logs)-1)) if len(logs)>1 else np.asarray([np.nan])
            comparisons[arm] = dict(log_ratio_interval=interval,
                shifted_score_ratio=float(np.exp(logs.mean())),
                tasks_won=int((scores[arm]<scores.timesfm3).sum()), tasks=len(logs),
                leave_one_task_out_ratio_min=float(np.min(leave_one)),
                leave_one_task_out_ratio_max=float(np.max(leave_one)),
                largest_log_improvements=logs.nsmallest(min(3,len(logs))).to_dict())
        partition_comparisons[partition] = comparisons
    pivot = task.wql9.unstack("arm")
    # Log ratios align with multiplicative aggregation; resample tasks after seed averaging.
    log_ratio = np.log((pivot.atom_semi_markov+1e-5)/(pivot.timesfm3+1e-5))
    raw = frame[frame.arm.eq("timesfm3")]
    raw_task = raw.groupby("dataset")[["atom_loss", "nonatom_loss"]].sum().sum(axis=1)
    diagnostics = dict(atom_point_fraction=float(raw.atom_points.sum()/raw.points.sum()),
        atom_raw_loss_share=float(raw.atom_loss.sum()/(raw.atom_loss.sum()+raw.nonatom_loss.sum())),
        largest_five_tasks_raw_loss_share=float(raw_task.nlargest(5).sum()/raw_task.sum()),
        empirical_cdf_by_quantile=pd.DataFrame(coverage).query("arm == 'timesfm3'").groupby("quantile").empirical_cdf.mean().to_dict())
    result = dict(stage="official_style_development_aggregation", cases=len(cases), datasets=len(pivot),
        scores=summary, diagnostics=diagnostics, partition_comparisons=partition_comparisons,
        atom_semi_minus_timesfm_task_wql=paired_cluster_interval(pivot.atom_semi_markov-pivot.timesfm3),
        atom_semi_to_timesfm_log_ratio=paired_cluster_interval(log_ratio),
        invalid_mase_cases=int(frame.mase.isna().sum()),
        upstream_leaderboard_sha=sha256(upstream_leaderboard),
        baseline_runtime=json.loads((outputs/"runtime.json").read_text()),
        sota=False, scope="Official-style aggregation of original H32 development cases; not official horizons, complete variables, windows, held-out test or leaderboard. No invalid-value imputation. Seed-average task scores, not seed ensemble predictions.")
    frame.to_parquet(dest/"case_scores.parquet", index=False)
    table.to_parquet(dest/"task_scores.parquet", index=False)
    pd.DataFrame(coverage).to_parquet(dest/"coverage.parquet", index=False)
    (dest/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
