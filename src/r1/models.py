"""Strong inexpensive controls before investing in a new distribution operator."""
import time
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_event_probe(frame, seed=2021, max_iter=100, max_leaf_nodes=15, threads=4):
    metadata = {"label", "partition", "dataset", "item", "origin", "cluster"}
    full = [c for c in frame if c not in metadata]
    base = [c for c in full if c not in ("log_age", "age_left_censored")]
    train = frame[frame.partition == "train"]
    valid = frame[frame.partition == "validation"]
    counts = train.groupby("dataset").size()
    weights = train.dataset.map(lambda d: 1.0/counts[d]).to_numpy()
    weights = weights * (len(weights) / weights.sum())
    order = np.random.default_rng(seed).permutation(["context_only", "context_and_age"])
    predictions, models, timing = {}, {}, {}
    for arm in order:
        cols = base if arm == "context_only" else full
        model = HistGradientBoostingClassifier(max_iter=max_iter,
                    max_leaf_nodes=max_leaf_nodes, learning_rate=0.06,
                    l2_regularization=1.0, min_samples_leaf=40,
                    early_stopping=False, random_state=seed)
        tick = time.perf_counter()
        with threadpool_limits(limits=threads):
            model.fit(train[cols], train.label, sample_weight=weights)
            predictions[arm] = model.predict_proba(valid[cols])[:, 1]
        timing[arm] = time.perf_counter() - tick
        models[arm] = (model, cols)
    return valid.copy(), predictions, timing, models


def timesfm3_forecaster(checkpoint_path, batch_size=4):
    from timesfm3 import TimesFM3Evaluator, ModelConfig
    device = select_device()
    return TimesFM3Evaluator(ModelConfig(checkpoint_path=str(checkpoint_path),
                            per_core_batch_size=batch_size, device=str(device)))


def gaussian_random_walk(contexts, horizon, quantiles):
    """Zero-drift arithmetic increments; a diagnostic baseline, not a new model."""
    from scipy.stats import norm
    point, forecasts = [], []
    for y in contexts:
        sigma = max(float(np.std(np.diff(y[-97:]))), 1e-8)
        center = np.full(horizon, y[-1])
        spread = np.sqrt(np.arange(1, horizon+1)) * sigma
        point.append(center)
        forecasts.append(center[:, None] + spread[:, None]*norm.ppf(quantiles))
    return np.asarray(point), np.asarray(forecasts)
