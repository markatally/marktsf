"""FI2010 LOB classification producer for PRISM M1b step 8.

The input files use the standard FI2010 transposed layout:
144 feature rows followed by 5 label rows for k = 10, 20, 30, 50, 100 ticks.
This producer trains lightweight NumPy classifiers, writes class-probability
``pred.npy`` and integer-label ``true.npy`` arrays, and names result directories
so ``experiments.PRISM.oracle_drift`` can consume them directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

LABEL_ROWS = {10: -5, 20: -4, 30: -3, 50: -2, 100: -1}
MODELS = ("ClassPrior", "Centroid", "DiagGaussian", "LinearSoftmax")


@dataclass(frozen=True)
class FI2010Split:
    train_x: np.ndarray
    train_y: dict[int, np.ndarray]
    test_x: np.ndarray
    test_y: dict[int, np.ndarray]


def _load_file(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 149:
        raise ValueError(f"Unexpected FI2010 shape for {path}: {arr.shape}")
    return arr


def load_fi2010(input_dir: Path, ks: list[int]) -> FI2010Split:
    train_path = input_dir / "Train_Dst_NoAuction_DecPre_CF_7.txt"
    test_paths = sorted(input_dir.glob("Test_Dst_NoAuction_DecPre_CF_*.txt"))
    if not train_path.exists() or not test_paths:
        raise FileNotFoundError(f"Missing FI2010 train/test files under {input_dir}")

    train = _load_file(train_path)
    tests = [_load_file(p) for p in test_paths]

    train_x = train[:144].T.astype(np.float32)
    test_x = np.concatenate([t[:144].T.astype(np.float32) for t in tests], axis=0)

    train_y = {k: train[LABEL_ROWS[k]].astype(np.int64) - 1 for k in ks}
    test_y = {k: np.concatenate([t[LABEL_ROWS[k]].astype(np.int64) - 1 for t in tests]) for k in ks}

    if any(y.min() < 0 or y.max() > 2 for y in [*train_y.values(), *test_y.values()]):
        raise ValueError("Expected FI2010 labels in {1,2,3}")

    return FI2010Split(train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((train_x - mean) / std).astype(np.float32), ((test_x - mean) / std).astype(np.float32)


def class_prior(train_y: np.ndarray, n: int) -> np.ndarray:
    counts = np.bincount(train_y, minlength=3).astype(np.float64) + 1.0
    probs = counts / counts.sum()
    return np.repeat(probs[None, :], n, axis=0)


def centroid_probs(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    centroids = np.stack([train_x[train_y == k].mean(axis=0) for k in range(3)], axis=0)
    d2 = ((test_x[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2)
    scale = np.median(d2)
    logits = -d2 / max(scale, 1e-6)
    return softmax(logits)


def diag_gaussian_probs(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    priors = np.bincount(train_y, minlength=3).astype(np.float64) + 1.0
    priors /= priors.sum()
    means = []
    vars_ = []
    for k in range(3):
        xk = train_x[train_y == k]
        means.append(xk.mean(axis=0))
        vars_.append(np.clip(xk.var(axis=0), 1e-3, None))
    means_arr = np.stack(means, axis=0)
    vars_arr = np.stack(vars_, axis=0)
    logp = -0.5 * (((test_x[:, None, :] - means_arr[None, :, :]) ** 2) / vars_arr[None, :, :]).sum(axis=2)
    logp += -0.5 * np.log(vars_arr).sum(axis=1)[None, :] + np.log(priors)[None, :]
    return softmax(logp)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def linear_softmax_probs(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, d = train_x.shape
    w = rng.normal(0.0, 0.01, size=(d, 3))
    b = np.zeros(3, dtype=np.float64)
    y_onehot = np.eye(3, dtype=np.float64)[train_y]
    reg = 1e-4

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        last_loss = 0.0
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = train_x[idx].astype(np.float64)
            yb = y_onehot[idx]
            probs = softmax(xb @ w + b)
            grad = (probs - yb) / len(idx)
            w -= lr * (xb.T @ grad + reg * w)
            b -= lr * grad.sum(axis=0)
            picked = probs[np.arange(len(idx)), train_y[idx]]
            last_loss = float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())
        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            log.info("LinearSoftmax epoch %d/%d train_ce=%.4f", epoch, epochs, last_loss)
    return softmax(test_x.astype(np.float64) @ w + b)


def window_artifacts(probs: np.ndarray, labels: np.ndarray, eval_window: int) -> tuple[np.ndarray, np.ndarray]:
    n_windows = len(labels) // eval_window
    if n_windows == 0:
        raise ValueError(f"eval_window={eval_window} exceeds test samples={len(labels)}")
    end = n_windows * eval_window
    pred = probs[:end].reshape(n_windows, eval_window, 3).astype(np.float32)
    true = labels[:end].reshape(n_windows, eval_window, 1).astype(np.int64)
    return pred, true


def save_artifact(
    results_root: Path,
    dataset: str,
    lookback: int,
    horizon: int,
    model_name: str,
    seed: int,
    pred: np.ndarray,
    true: np.ndarray,
    metadata: dict[str, object],
) -> None:
    suffix = f"ftC_sl{lookback}_pl{horizon}_seed{seed}"
    out = results_root / f"long_term_forecast_{dataset}_{lookback}_{horizon}_{model_name}_{dataset}_{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred.npy", pred)
    np.save(out / "true.npy", true)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    log.info("saved %s pred=%s true=%s", out, pred.shape, true.shape)


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    ks = [int(k) for k in args.ks]
    unknown = sorted(set(ks) - set(LABEL_ROWS))
    if unknown:
        raise ValueError(f"Unsupported k values {unknown}; choices are {sorted(LABEL_ROWS)}")

    model_names = args.models
    unknown_models = sorted(set(model_names) - set(MODELS))
    if unknown_models:
        raise ValueError(f"Unknown models {unknown_models}; choices are {list(MODELS)}")

    split = load_fi2010(Path(args.input_dir), ks)
    train_x, test_x = standardize(split.train_x, split.test_x)
    log.info("FI2010 train=%s test=%s eval_window=%d", train_x.shape, test_x.shape, args.eval_window)

    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    lookback = args.lookback
    horizon = args.eval_window

    for k in ks:
        train_y = split.train_y[k]
        test_y = split.test_y[k]
        dataset = f"{args.dataset_prefix}K{k}"
        log.info("k=%d train_counts=%s test_counts=%s", k, np.bincount(train_y, minlength=3), np.bincount(test_y, minlength=3))

        all_probs: dict[str, np.ndarray] = {}
        if "ClassPrior" in model_names:
            all_probs["ClassPrior"] = class_prior(train_y, len(test_y))
        if "Centroid" in model_names:
            all_probs["Centroid"] = centroid_probs(train_x, train_y, test_x)
        if "DiagGaussian" in model_names:
            all_probs["DiagGaussian"] = diag_gaussian_probs(train_x, train_y, test_x)
        if "LinearSoftmax" in model_names:
            all_probs["LinearSoftmax"] = linear_softmax_probs(
                train_x,
                train_y,
                test_x,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed + k,
            )

        for model_name, probs in all_probs.items():
            pred, true = window_artifacts(probs, test_y, args.eval_window)
            save_artifact(
                results_root,
                dataset,
                lookback,
                horizon,
                model_name,
                args.seed,
                pred,
                true,
                {
                    "dataset": dataset,
                    "source": "FI2010",
                    "label_horizon_ticks": k,
                    "label_mapping": {"0": "down", "1": "stationary", "2": "up"},
                    "model": model_name,
                    "seed": args.seed,
                    "eval_window": args.eval_window,
                    "train_samples": int(len(train_y)),
                    "test_samples": int(len(test_y)),
                },
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Produce FI2010 classification artifacts for PRISM M1b.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", default="input/FI2010")
    p.add_argument("--results-root", default="external/TSLib/results")
    p.add_argument("--dataset-prefix", default="FI2010")
    p.add_argument("--ks", nargs="+", type=int, default=[10, 50, 100])
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--lookback", type=int, default=100)
    p.add_argument("--eval-window", type=int, default=256)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=2021)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
