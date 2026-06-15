"""M12 — official S4M missing-aware baseline adaptation.

This milestone imports the official S4M model class and evaluates it under the
MaskShift encoder-mask protocol. The run is intentionally resource-bounded:
official architecture, reduced channels/samples, clean forecast targets, and a
small number of epochs. It is not a full reproduction of the S4M paper's
benchmark protocol.
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .maskshift_core import (
    MECHANISMS,
    OPERATIONAL_MECHANISMS,
    ROOT,
    ExperimentConfig,
    ensure_dir,
    generate_mask,
    load_dataset,
    sample_origins,
    train_test_normalize,
    write_json,
)


warnings.filterwarnings("ignore", category=UserWarning)

EXP_DIR = Path(__file__).parent
OUT_DIR = ensure_dir(EXP_DIR / "m12_official_s4m_baseline")
TABLE_DIR = ensure_dir(EXP_DIR / "tables")
S4M_DIR = ROOT / "external" / "S4M"
S4M_CODE_DIR = S4M_DIR / "s4m"

DATASETS = ["Weather", "Electricity"]
SEED_OFFSETS = [0, 10_000, 20_000]
BACKBONE = "S4M_official"
MAX_ROWS = 3500
MAX_CHANNELS = 8


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def git_revision(path: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def git_diff_stat(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "diff", "--stat"], text=True).strip()
    except Exception:
        return "unknown"


def load_s4m_model():
    if not S4M_CODE_DIR.exists():
        raise FileNotFoundError(f"S4M not found at {S4M_CODE_DIR}")
    sys.path.insert(0, str(S4M_CODE_DIR))
    sys.path.insert(0, str(S4M_CODE_DIR / "model"))
    from model.S4M import Model  # type: ignore

    return Model


def make_config(channels: int) -> SimpleNamespace:
    return SimpleNamespace(
        dropout=0.05,
        mask=True,
        d_var=channels,
        classification=0,
        plot=0,
        d_model=8,
        en_conv_hidden_size=8,
        e_layers=1,
        d_ff=16,
        short_len=12,
        W=3,
        en_rnn_hidden_sizes=[8, 8],
        input_keep_prob=0.9,
        output_keep_prob=0.9,
        factor=3,
        output_attention=False,
        n_heads=2,
        memory_size=20,
        K=10,
        momentum=0.99,
        is_training=1,
        M=10,
        per_mem_size=5,
        thres1=0.95,
        thres2=0.3,
        topM=20,
        topK=5,
        n=4,
    )


def make_model(channels: int) -> nn.Module:
    model_cls = load_s4m_model()
    return model_cls(make_config(channels))


def make_reduced_dataset_splits(dataset: str, cfg: ExperimentConfig) -> dict:
    values_raw, meta = load_dataset(dataset, max_rows=MAX_ROWS, max_channels=MAX_CHANNELS)
    split_idx = int(values_raw.shape[0] * 0.7)
    values, norm_meta = train_test_normalize(values_raw, split_idx)
    train_origins = sample_origins(
        len(values),
        cfg.lookback,
        cfg.horizon,
        start=cfg.lookback,
        stop=split_idx,
        stride=cfg.stride,
        max_samples=cfg.max_train_samples,
    )
    test_origins = sample_origins(
        len(values),
        cfg.lookback,
        cfg.horizon,
        start=split_idx,
        stop=len(values) - cfg.horizon,
        stride=cfg.stride,
        max_samples=cfg.max_test_samples,
    )
    return {
        "values": values,
        "split_idx": split_idx,
        "train_origins": train_origins,
        "test_origins": test_origins,
        "meta": {**meta, "split_idx": split_idx, "normalization": norm_meta},
    }


def build_window_tensors(
    values: np.ndarray,
    mask: np.ndarray,
    origins: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    ys = []
    obs_flags = []
    max_idxs = []
    min_idxs = []
    max_values = []
    min_values = []
    for origin in origins:
        x = values[origin - cfg.lookback : origin].copy()
        m = mask[origin - cfg.lookback : origin].copy()
        observed = ~m
        x_obs = np.where(observed, x, 0.0)
        idx = np.arange(cfg.lookback)[:, None].repeat(values.shape[1], axis=1)
        max_idx = np.abs(np.argmax(x_obs, axis=0)[None, :] - idx)
        min_idx = np.abs(np.argmin(x_obs, axis=0)[None, :] - idx)
        xs.append(x_obs)
        ys.append(values[origin : origin + cfg.horizon])
        obs_flags.append(observed.astype(np.float32))
        max_idxs.append(max_idx)
        min_idxs.append(min_idx)
        max_values.append(np.max(x_obs, axis=0)[None, :].repeat(cfg.lookback, axis=0))
        min_values.append(np.min(x_obs, axis=0)[None, :].repeat(cfg.lookback, axis=0))
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(ys, dtype=np.float32),
        np.asarray(obs_flags, dtype=np.float32),
        np.asarray(max_idxs, dtype=np.float32),
        np.asarray(min_idxs, dtype=np.float32),
        np.asarray(max_values, dtype=np.float32),
        np.asarray(min_values, dtype=np.float32),
    )


def to_device(batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(item.to(DEVICE) for item in batch)


def train_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    obs: np.ndarray,
    max_idx: np.ndarray,
    min_idx: np.ndarray,
    max_value: np.ndarray,
    min_value: np.ndarray,
    cfg: ExperimentConfig,
    epochs: int = 1,
) -> nn.Module:
    model = model.to(DEVICE)
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y),
        torch.from_numpy(obs),
        torch.from_numpy(max_idx),
        torch.from_numpy(min_idx),
        torch.from_numpy(max_value),
        torch.from_numpy(min_value),
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=(DEVICE.type == "cuda"))
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    model.train()
    first = next(iter(loader))
    xb, _, mb, maxb, minb, maxvb, minvb = to_device(first)
    model.warmup(xb, mb, maxb, minb, maxvb, minvb)
    if hasattr(model, "mem_net"):
        model.mem_net.is_training = 1
    for _ in range(epochs):
        for xb, yb, mb, maxb, minb, maxvb, minvb in loader:
            xb, yb, mb, maxb, minb, maxvb, minvb = to_device((xb, yb, mb, maxb, minb, maxvb, minvb))
            opt.zero_grad(set_to_none=True)
            out = model(xb, mb, maxb, minb, maxvb, minvb)
            loss = loss_fn(out[:, -cfg.horizon :, :], yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    obs: np.ndarray,
    max_idx: np.ndarray,
    min_idx: np.ndarray,
    max_value: np.ndarray,
    min_value: np.ndarray,
    cfg: ExperimentConfig,
) -> dict:
    model.eval()
    if hasattr(model, "mem_net"):
        model.mem_net.is_training = 0
    preds = []
    for start in range(0, len(x), 8):
        xb = torch.from_numpy(x[start : start + 8]).to(DEVICE)
        mb = torch.from_numpy(obs[start : start + 8]).to(DEVICE)
        maxb = torch.from_numpy(max_idx[start : start + 8]).to(DEVICE)
        minb = torch.from_numpy(min_idx[start : start + 8]).to(DEVICE)
        maxvb = torch.from_numpy(max_value[start : start + 8]).to(DEVICE)
        minvb = torch.from_numpy(min_value[start : start + 8]).to(DEVICE)
        pred = model(xb, mb, maxb, minb, maxvb, minvb)[:, -cfg.horizon :, :]
        preds.append(pred.detach().cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    target_pred = pred[:, -1, 0]
    target_true = y[:, -1, 0]
    target_losses = (target_pred - target_true) ** 2
    return {
        "target_mse": float(np.mean(target_losses)),
        "target_mae": float(np.mean(np.abs(target_pred - target_true))),
        "full_horizon_mse": float(np.mean((pred - y) ** 2)),
        "target_losses": target_losses.astype(float).tolist(),
    }


def run_dataset(dataset: str, cfg: ExperimentConfig, seed_offset: int) -> dict:
    torch.manual_seed(cfg.seed + seed_offset)
    np.random.seed(cfg.seed + seed_offset)
    ds = make_reduced_dataset_splits(dataset, cfg)
    values = ds["values"]
    split_idx = ds["split_idx"]
    train_origins = ds["train_origins"]
    test_origins = ds["test_origins"]

    train_mask = generate_mask(values, "mcar", cfg.target_rate, cfg.seed + seed_offset, split_idx=split_idx)
    x_train, y_train, obs_train, max_train, min_train, maxv_train, minv_train = build_window_tensors(
        values, train_mask, train_origins, cfg
    )
    model = make_model(values.shape[1])
    model = train_model(model, x_train, y_train, obs_train, max_train, min_train, maxv_train, minv_train, cfg)

    rows = []
    loss_groups = []
    for i, mechanism in enumerate(MECHANISMS):
        test_mask = generate_mask(values, mechanism, cfg.target_rate, cfg.seed + seed_offset + 9000 + i * 17, split_idx=split_idx)
        x_test, y_test, obs_test, max_test, min_test, maxv_test, minv_test = build_window_tensors(
            values, test_mask, test_origins, cfg
        )
        metrics = evaluate_model(model, x_test, y_test, obs_test, max_test, min_test, maxv_test, minv_test, cfg)
        loss_groups.append(metrics["target_losses"])
        rows.append(
            {
                "dataset": dataset,
                "seed_offset": seed_offset,
                "backbone": BACKBONE,
                "test_mechanism": mechanism,
                "target_mse": metrics["target_mse"],
                "target_mae": metrics["target_mae"],
                "full_horizon_mse": metrics["full_horizon_mse"],
                "n_test_windows": len(metrics["target_losses"]),
            }
        )

    mcar_mse = next(row["target_mse"] for row in rows if row["test_mechanism"] == "mcar")
    mechanism_effect_rows = []
    for mechanism in OPERATIONAL_MECHANISMS:
        mech_mse = next(row["target_mse"] for row in rows if row["test_mechanism"] == mechanism)
        mechanism_effect_rows.append(
            {
                "mechanism": mechanism,
                "mean_target_mse": float(mech_mse),
                "absolute_delta_vs_mcar": float(mech_mse - mcar_mse),
                "relative_degradation_vs_mcar": float((mech_mse - mcar_mse) / max(mcar_mse, 1e-9)),
                "log_ratio_vs_mcar": float(np.log((mech_mse + 1e-6) / (mcar_mse + 1e-6))),
            }
        )
    try:
        h_stat, p_value = stats.kruskal(*loss_groups)
    except ValueError:
        h_stat, p_value = float("nan"), float("nan")
    max_degradation = max(row["relative_degradation_vs_mcar"] for row in mechanism_effect_rows)
    max_abs_delta = max(row["absolute_delta_vs_mcar"] for row in mechanism_effect_rows)
    strongest = max(mechanism_effect_rows, key=lambda row: row["absolute_delta_vs_mcar"])["mechanism"]
    gate_pass = max_degradation > 0.05 and (np.isnan(p_value) or p_value <= 0.05)
    return {
        "dataset": dataset,
        "seed_offset": seed_offset,
        "meta": ds["meta"],
        "rows": rows,
        "mechanism_effect_rows": mechanism_effect_rows,
        "kruskal_h": float(h_stat),
        "kruskal_p": float(p_value),
        "mcar_target_mse": float(mcar_mse),
        "max_relative_degradation": float(max_degradation),
        "max_absolute_delta": float(max_abs_delta),
        "strongest_mechanism": strongest,
        "gate_pass": bool(gate_pass),
    }


def mean_ci(values: list[float], confidence: float = 0.95) -> dict:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if len(arr) <= 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": int(len(arr))}
    sem = stats.sem(arr)
    tcrit = stats.t.ppf((1 + confidence) / 2.0, df=len(arr) - 1)
    half = float(tcrit * sem)
    return {"mean": mean, "ci_low": mean - half, "ci_high": mean + half, "n": int(len(arr))}


def fmt_ci(
    summary: dict,
    pct: bool = False,
    digits: int = 3,
    lower: float | None = None,
    upper: float | None = None,
) -> str:
    mean = summary["mean"]
    ci_low = summary["ci_low"]
    ci_high = summary["ci_high"]
    if lower is not None:
        mean = max(lower, mean)
        ci_low = max(lower, ci_low)
        ci_high = max(lower, ci_high)
    if upper is not None:
        mean = min(upper, mean)
        ci_low = min(upper, ci_low)
        ci_high = min(upper, ci_high)
    if pct:
        return f"{mean * 100:.1f}% [{ci_low * 100:.1f}, {ci_high * 100:.1f}]"
    return f"{mean:.{digits}f} [{ci_low:.{digits}f}, {ci_high:.{digits}f}]"


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def aggregate_dataset(dataset: str, seed_runs: list[dict]) -> dict:
    max_relative = mean_ci([row["max_relative_degradation"] for row in seed_runs])
    max_abs = mean_ci([row["max_absolute_delta"] for row in seed_runs])
    kruskal_p = mean_ci([row["kruskal_p"] for row in seed_runs])
    kruskal_p = {
        **kruskal_p,
        "mean": min(1.0, max(0.0, kruskal_p["mean"])),
        "ci_low": min(1.0, max(0.0, kruskal_p["ci_low"])),
        "ci_high": min(1.0, max(0.0, kruskal_p["ci_high"])),
    }
    gate_pass_count = int(sum(row["gate_pass"] for row in seed_runs))
    strongest_values = [row["strongest_mechanism"] for row in seed_runs]
    strongest_mode = max(sorted(set(strongest_values)), key=strongest_values.count)
    if strongest_values.count(strongest_mode) <= len(strongest_values) / 2:
        strongest_mode = "mixed"
    return {
        "dataset": dataset,
        "backbone": BACKBONE,
        "n_seeds": len(seed_runs),
        "seed_offsets": [row["seed_offset"] for row in seed_runs],
        "gate_pass_count": gate_pass_count,
        "gate_pass": bool(gate_pass_count >= max(1, int(np.ceil(len(seed_runs) / 2)))),
        "max_relative_degradation": max_relative["mean"],
        "max_relative_degradation_ci": max_relative,
        "max_absolute_delta": max_abs["mean"],
        "max_absolute_delta_ci": max_abs,
        "kruskal_p": kruskal_p["mean"],
        "kruskal_p_ci": kruskal_p,
        "strongest_mechanism": strongest_mode,
        "seed_summaries": [
            {
                "seed_offset": row["seed_offset"],
                "max_relative_degradation": row["max_relative_degradation"],
                "max_absolute_delta": row["max_absolute_delta"],
                "kruskal_p": row["kruskal_p"],
                "strongest_mechanism": row["strongest_mechanism"],
                "gate_pass": row["gate_pass"],
            }
            for row in seed_runs
        ],
    }


def write_table(summary: dict) -> None:
    lines = [
        "# M12 official S4M adaptation table",
        "",
        "S4M is imported from the official repository and evaluated under the MaskShift encoder-mask protocol with reduced channels/samples and three seed offsets. This is not the full S4M benchmark reproduction.",
        "",
        "| Dataset | Backbone | Max degradation mean [95% CI] | Max abs delta [95% CI] | Strongest mechanism mode | Kruskal p mean [95% CI] | Gate seeds |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in summary["datasets"]:
        lines.append(
            f"| {row['dataset']} | {BACKBONE} | {fmt_ci(row['max_relative_degradation_ci'], pct=True)} | {fmt_ci(row['max_absolute_delta_ci'])} | {row['strongest_mechanism']} | {fmt_ci(row['kruskal_p_ci'], lower=0.0, upper=1.0)} | {row['gate_pass_count']}/{row['n_seeds']} |"
        )
    (TABLE_DIR / "m12_s4m_table.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    cfg = ExperimentConfig(max_train_samples=32, max_test_samples=24)
    datasets = []
    seed_runs = []
    errors = []
    for dataset_index, dataset in enumerate(DATASETS):
        dataset_seed_runs = []
        for seed_index, offset in enumerate(SEED_OFFSETS):
            seed_offset = dataset_index * 1000 + offset
            try:
                run = run_dataset(dataset, cfg, seed_offset=seed_offset)
                run["seed_index"] = seed_index
                dataset_seed_runs.append(run)
                seed_runs.append(run)
            except Exception as exc:
                errors.append({"dataset": dataset, "seed_offset": seed_offset, "error": repr(exc)})
        if dataset_seed_runs:
            datasets.append(aggregate_dataset(dataset, dataset_seed_runs))
    expected_runs = len(DATASETS) * len(SEED_OFFSETS)
    protocol_complete = (
        len(seed_runs) == expected_runs
        and len(datasets) == len(DATASETS)
        and all(len(row["rows"]) == len(MECHANISMS) for row in seed_runs)
    )
    mechanism_shift_gate = any(row["gate_pass"] for row in datasets)
    summary = {
        "milestone": "M12",
        "status": "PASS_OFFICIAL_S4M_ADAPTATION" if protocol_complete else "HOLD_OFFICIAL_S4M_ADAPTATION",
        "device": str(DEVICE),
        "s4m_dir": str(S4M_DIR),
        "s4m_revision": git_revision(S4M_DIR),
        "s4m_local_diff_stat": git_diff_stat(S4M_DIR),
        "official_model_files": [
            "external/S4M/s4m/model/S4M.py",
            "external/S4M/s4m/model/Bank.py",
            "external/S4M/s4m/model/S4/s4/models/s4/s4.py",
            "external/S4M/s4m/model/S4/s4/models/s4/s4_mask1.py",
        ],
        "device_port_patch": "external/S4M/s4m/model/Bank.py replaces a hard-coded .cuda() memory fetch with .to(Q.device); architecture and forward equations otherwise unchanged.",
        "config": {**cfg.__dict__, "max_rows": MAX_ROWS, "max_channels": MAX_CHANNELS, "epochs": 1, "batch_size": 8},
        "seed_offsets": SEED_OFFSETS,
        "backbone": BACKBONE,
        "datasets": datasets,
        "seed_runs": seed_runs,
        "errors": errors,
        "protocol_note": "Imports official S4M model class; MaskShift loop masks encoder inputs only and keeps forecast targets clean; reduced samples/channels for local MPS feasibility; three seed offsets are used for submission hardening.",
        "m12_gate": bool(protocol_complete),
        "mechanism_shift_gate": bool(mechanism_shift_gate),
    }
    write_json(OUT_DIR / "s4m_baseline_summary.json", summary)
    if datasets:
        write_table(summary)
    print(json.dumps({"milestone": "M12", "status": summary["status"], "device": str(DEVICE), "errors": errors}, indent=2))


if __name__ == "__main__":
    main()
