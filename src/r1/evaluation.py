"""Scores and paired clustered uncertainty; overlapping windows are not iid."""
import numpy as np
import pandas as pd


def paired_cluster_interval(values, seed=2021, repetitions=2000):
    values = np.asarray(values, dtype=float)
    if len(values) < 8:
        return {"mean": float(values.mean()), "ci95": None,
                "clusters": len(values), "reason": "fewer_than_8_clusters"}
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(repetitions, len(values)))].mean(1)
    return {"mean": float(values.mean()),
            "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
            "clusters": len(values)}


def event_scores(valid, predictions, seed=2021):
    out = valid[["dataset", "item", "horizon", "origin", "cluster", "label"]].copy()
    summary = []
    for arm, probabilities in predictions.items():
        p = np.clip(probabilities, 1e-7, 1-1e-7)
        y = out.label.to_numpy()
        out[f"{arm}_brier"] = (p-y)**2
        out[f"{arm}_logloss"] = -y*np.log(p)-(1-y)*np.log1p(-p)
    for horizon, frame in out.groupby("horizon"):
        # Equal weight to clusters, then equal weight to observed rows in cluster.
        grouped = frame.groupby("cluster").mean(numeric_only=True)
        for metric in ("brier", "logloss"):
            base = grouped[f"context_only_{metric}"]
            candidate = grouped[f"context_and_age_{metric}"]
            delta = paired_cluster_interval(candidate-base, seed=seed)
            summary.append({"horizon": int(horizon), "metric": metric,
                "context_only": float(base.mean()), "context_and_age": float(candidate.mean()),
                "relative_improvement_pct": float(100*(base.mean()-candidate.mean())/base.mean()),
                "paired_delta_age_minus_base": delta})
    return summary, out


def forecast_scores(target, point, quantile_values, quantile_levels):
    target, point, quantile_values = map(np.asarray, (target, point, quantile_values))
    if point.shape != target.shape or quantile_values.shape != target.shape+(len(quantile_levels),):
        raise ValueError("Forecast shape does not match targets/quantiles")
    if not all(np.isfinite(a).all() for a in (target, point, quantile_values)):
        raise ValueError("Non-finite forecast or target")
    error = target[..., None]-quantile_values
    loss = 2*np.maximum(np.asarray(quantile_levels)*error,
                         (np.asarray(quantile_levels)-1)*error)
    denom = np.abs(target).sum()
    numerator = loss.sum(axis=tuple(range(target.ndim))).mean()
    return {"mse": float(np.mean((target-point)**2)),
            "mae": float(np.mean(np.abs(target-point))),
            "wql9": float(numerator/denom) if denom else None,
            "quantile_loss_numerator":float(numerator),"absolute_target_denominator":float(denom),
            "crossing_fraction": float(np.mean(np.diff(quantile_values, axis=-1)<0)),
            "note": "Development 9-quantile WQL; not official full BOOM aggregation"}
