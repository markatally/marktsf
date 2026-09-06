"""Matched causal price controls and cumulative-return diagnostics; no alpha claim."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from r1.data import price_development
from r1.evaluation import forecast_scores, paired_cluster_interval


def evaluate_price_controls(repo_root, output_dir, reference_path, foundation_predictions,
                            reference_predictions, max_assets=24, horizon=32,
                            volatility_window=96, ewma_alpha=.06):
    if volatility_window < 2 or not 0 < ewma_alpha <= 1:
        raise ValueError("Invalid causal volatility configuration")
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    reference = pd.read_parquet(reference_path)
    if not reference.horizon.eq(horizon).all():
        raise ValueError("Financial horizon mismatch")
    keys = reference[["dataset", "item", "origin", "cluster", "target_hash"]].drop_duplicates()
    if keys.duplicated(["dataset", "item", "origin"]).any():
        raise ValueError("Ambiguous reference cases")
    records, _ = price_development(repo_root, max_assets=max_assets)
    lookup = {(s.dataset, s.item): s for s in records}
    forecasts = {}
    for directory, is_foundation in ((foundation_predictions, True), (reference_predictions, False)):
        for path in sorted(Path(directory).glob("*.npz")):
            with np.load(path) as data:
                key = (str(data["dataset"]), str(data["item"]), int(data["origin"]))
                arm = "timesfm3_native" if is_foundation else path.stem.split("_", 2)[2]
                if not is_foundation and arm not in {"geometric_independent", "geometric_coupled", "age_independent", "age_coupled"}:
                    raise ValueError("Unknown stored empirical arm")
                if (key, arm) in forecasts:
                    raise ValueError("Duplicate financial prediction")
                forecasts[key, arm] = (data["quantiles"].copy(), str(data["target_hash"]))
    levels = np.arange(.1, 1, .1)
    normal = norm.ppf(levels)[None, :]
    sqrt_h = np.sqrt(np.arange(1, horizon+1))[:, None]
    rows = []
    for case in keys.itertuples():
        key = (case.dataset, case.item, case.origin)
        series = lookup[key[:2]]
        if case.origin < series.train_end or case.origin+horizon >= len(series.values):
            raise ValueError("Price case outside frozen development validation")
        history = series.values[:case.origin+1]
        target = series.values[case.origin+1:case.origin+horizon+1]
        digest = hashlib.sha256(np.ascontiguousarray(target, dtype="<f8").tobytes()).hexdigest()
        if digest != case.target_hash or not np.isfinite(history).all() or np.any(history <= 0):
            raise ValueError("Invalid or mismatched financial case")
        level = history[-1]
        increments = np.diff(history[-volatility_window-1:])
        returns = np.diff(np.log(history))
        log_sigma = max(float(np.std(returns[-volatility_window:])), 1e-8)
        variance = float(returns[0]**2)
        for r in returns[1:]:
            variance = (1-ewma_alpha)*variance+ewma_alpha*r*r
        predictions = {
            "point_random_walk": np.full((horizon, 9), level),
            "arithmetic_gaussian_rw": level+max(float(np.std(increments)), 1e-8)*sqrt_h*normal,
            # Zero mean log increments; median stays at last observed price.
            "log_gaussian_rw": level*np.exp(log_sigma*sqrt_h*normal),
            "log_gaussian_ewma": level*np.exp(np.sqrt(max(variance, 1e-16))*sqrt_h*normal),
        }
        for arm in ("timesfm3_native", "geometric_independent", "geometric_coupled", "age_independent", "age_coupled"):
            q, stored_hash = forecasts[key, arm]
            if stored_hash != digest:
                raise ValueError("Stored financial target hash mismatch")
            predictions[arm] = q
        for arm, q in predictions.items():
            score = forecast_scores(target, q[:, 4], q, levels)
            # Simple cumulative returns are an affine transform of price at this
            # origin. They remain scoreable even if a model predicts negative price.
            return_target, return_q = target/level-1, q/level-1
            rscore = forecast_scores(return_target, return_q[:, 4], return_q, levels)
            coverage = np.mean((target >= q[:, 0]) & (target <= q[:, -1]))
            rows.append(dict(dataset=case.dataset, item=case.item, origin=case.origin, cluster=case.cluster,
                arm=arm, price_wql9=score["wql9"], price_mse=score["mse"],
                return_wql9=rscore["wql9"], return_mse=rscore["mse"],
                coverage80=float(coverage), interval80_relative_width=float(np.mean(q[:, -1]-q[:, 0])/level),
                nonpositive_quantile_fraction=float(np.mean(q <= 0))))
            np.savez_compressed(dest/f"{len(rows):05d}.npz", dataset=case.dataset, item=case.item,
                origin=case.origin, arm=arm, quantiles=q, target_hash=digest, target=target)
    frame = pd.DataFrame(rows)
    if frame[["price_wql9", "return_wql9"]].isna().any().any():
        raise ValueError("Undefined price/return WQL; do not silently drop cases")
    metrics = ["price_wql9", "price_mse", "return_wql9", "return_mse", "coverage80", "interval80_relative_width", "nonpositive_quantile_fraction"]
    grouped = frame.groupby(["cluster", "arm"])[metrics].mean()
    price = grouped.price_wql9.unstack("arm")
    result = dict(stage="matched_financial_controls", cases=len(keys), assets=keys.groupby(["dataset", "item"]).ngroups,
        calendar_months=len(price), volatility_window=volatility_window, ewma_alpha=ewma_alpha,
        scores={a: grouped.xs(a, level="arm").mean().to_dict() for a in frame.arm.unique()},
        candidate_minus_timesfm=paired_cluster_interval(price.age_coupled-price.timesfm3_native),
        candidate_minus_ewma=paired_cluster_interval(price.age_coupled-price.log_gaussian_ewma),
        sota=False, scope="246 frozen development cases; shared months not independent guarantees; no trading backtest; Gaussian log random walks are classical baselines, not GARCH fits")
    frame.to_parquet(dest/"case_scores.parquet", index=False)
    (dest/"summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
