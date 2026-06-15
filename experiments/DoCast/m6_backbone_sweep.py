"""M6 — Deep-backbone D0/D1/D2 sweep.

This trains lightweight TSLib backbones on the DoCast semi-synthetic sample
shape under three heads. All three arms receive the same static item controls
in the backbone input; D1 and D2 also share item-specific scalar response
capacity. This keeps the M6 comparison focused on the orthogonalized objective
rather than an extra-control advantage.

D0: observational backbone + unconstrained horizon-wise treatment head.
D1: structural item-specific scalar treatment head without orthogonalization.
D2: DoCast-style R-learner using backbone nuisances m(V) and pi(V).

The goal is not to tune SOTA accuracy; it is to verify that the causal protocol
survives non-linear TSF backbones and that the full D0/D1/D2 pathway is runnable.
"""

from __future__ import annotations

import json
import math
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
TSLIB = ROOT / "external" / "TSLib"
sys.path.insert(0, str(TSLIB))

import torch
from torch.utils.data import DataLoader, TensorDataset

from m2_docast import load_m5, generate_synthetic


OUT_DIR = Path(__file__).parent / "m6_backbone_sweep"
OUT_DIR.mkdir(exist_ok=True)

L = 56
H = 28
N_ITEMS = 24
TRAIN_ORIGINS = 220
TEST_ORIGINS = 40
EMBARGO = H
K_FOLDS = 2
GAMMA = 0.5
THETA_ITEM_WEIGHT = 0.8
SEEDS = [2021, 2022, 2023]
BACKBONES = ["DLinear", "PatchTST", "TiDE", "Transformer", "TimeXer"]
DEEP_BACKBONES = {"PatchTST", "TiDE", "Transformer", "TimeXer"}


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def time_features(T: int, month: np.ndarray, wday: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.sin(2 * math.pi * month / 12),
        np.cos(2 * math.pi * month / 12),
        np.sin(2 * math.pi * wday / 7),
    ]).astype(np.float32)


def build_tensor_data(data: dict, gamma: float, seed: int) -> dict:
    base = {
        "Y": data["Y"][:N_ITEMS],
        "P": data["P"][:N_ITEMS],
        "snap": data["snap"],
        "month": data["month"],
        "wday": data["wday"],
        "n": N_ITEMS,
        "T": data["T"],
    }
    syn = generate_synthetic(base, gamma=gamma, seed=seed)
    log_y = np.log(syn["Y_syn"] + 1.0).astype(np.float32)
    phi = syn["log_pr"].astype(np.float32)
    theta_star = syn["theta_star"][:N_ITEMS].astype(np.float32)
    tf = time_features(base["T"], base["month"], base["wday"])

    train_end = min(L + TRAIN_ORIGINS, base["T"] - H - EMBARGO - TEST_ORIGINS)
    test_start = train_end + EMBARGO
    test_end = min(test_start + TEST_ORIGINS, base["T"] - H)

    def make_split(t0: int, t1: int):
        x_list, xm_list, xdm_list, y_list, phi_future_list, item_list, origin_list = [], [], [], [], [], [], []
        for i in range(N_ITEMS):
            for t in range(max(L, t0), t1):
                y_lag = log_y[i, t - L:t]
                phi_lag = phi[i, t - L:t]
                snap_lag = np.repeat(base["snap"][t], L).astype(np.float32)
                x = np.column_stack([y_lag, phi_lag, snap_lag]).astype(np.float32)
                x_mark = tf[t - L:t]
                future_tf = tf[t:t + H]
                future_phi = phi[i, t:t + H]
                # Future mark includes calendar features plus the planned treatment.
                x_dec_mark = np.column_stack([future_tf, future_phi]).astype(np.float32)
                y = log_y[i, t:t + H]
                x_list.append(x)
                xm_list.append(x_mark)
                xdm_list.append(x_dec_mark)
                y_list.append(y)
                phi_future_list.append(future_phi)
                item_list.append(i)
                origin_list.append(t)
        return {
            "x": torch.tensor(np.stack(x_list), dtype=torch.float32),
            "x_mark": torch.tensor(np.stack(xm_list), dtype=torch.float32),
            "x_dec_mark": torch.tensor(np.stack(xdm_list), dtype=torch.float32),
            "y": torch.tensor(np.stack(y_list), dtype=torch.float32),
            "phi_future": torch.tensor(np.stack(phi_future_list), dtype=torch.float32),
            "item_idx": torch.tensor(item_list, dtype=torch.long),
            "origin": torch.tensor(origin_list, dtype=torch.long),
        }

    return {
        "train": make_split(L, train_end),
        "test": make_split(test_start, test_end),
        "theta_star": theta_star,
        "theta_star_mean": float(theta_star.mean()),
        "theta_star_ser_sign": -1,
        "seed": seed,
        "train_end": int(train_end),
        "test_start": int(test_start),
        "test_end": int(test_end),
    }


def model_config(name: str, enc_in: int = 3) -> Namespace:
    return Namespace(
        task_name="long_term_forecast",
        features="M",
        seq_len=L,
        label_len=H,
        pred_len=H,
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=enc_in,
        d_model=16,
        n_heads=2,
        e_layers=1,
        d_layers=1,
        d_ff=32,
        factor=1,
        dropout=0.05,
        embed="timeF",
        freq="d",
        activation="gelu",
        use_norm=1,
        patch_len=8,
        individual=False,
        moving_avg=25,
    )


def load_model(name: str, enc_in: int = 3):
    mod = __import__(f"models.{name}", fromlist=["Model"])
    return mod.Model(model_config(name, enc_in=enc_in))


def forward_model(model, name: str, x, x_mark, x_dec_mark):
    b = x.shape[0]
    x_dec = torch.zeros((b, H, x.shape[-1]), device=x.device)
    if name == "TiDE":
        # TiDE expects future temporal features in batch_y_mark.
        y_mark = x_dec_mark[:, :, :3]
        out = model(x, x_mark, x_dec, y_mark)
    else:
        out = model(x, x_mark, x_dec, x_dec_mark[:, :, :3])
    return out[:, :, 0]


def wmape_log(pred: torch.Tensor, y: torch.Tensor) -> float:
    yh = torch.exp(pred) - 1.0
    yt = torch.exp(y) - 1.0
    return float((torch.abs(yh - yt).sum() / (yt.sum() + 1e-8)).detach().cpu())


def make_loader(split: dict, batch_size: int = 64, device: torch.device | None = None, include_item: bool = False):
    tensors = [split["x"], split["x_mark"], split["x_dec_mark"], split["y"], split["phi_future"]]
    if include_item:
        tensors.append(split["item_idx"])
    ds = TensorDataset(*tensors)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=(device is not None and device.type == "cuda"))


def select_split(split: dict, mask_or_idx) -> dict:
    """Slice a tensor split while preserving all aligned fields."""
    if isinstance(mask_or_idx, np.ndarray):
        idx = torch.as_tensor(mask_or_idx)
    else:
        idx = mask_or_idx
    if idx.dtype == torch.bool:
        idx = torch.where(idx)[0]
    out = {}
    n = split["x"].shape[0]
    for key, value in split.items():
        if torch.is_tensor(value) and value.shape[:1] == (n,):
            out[key] = value[idx]
        else:
            out[key] = value
    return out


def zero_treatment_channel(x: torch.Tensor) -> torch.Tensor:
    x_v = x.clone()
    x_v[:, :, 1] = 0.0
    return x_v


def add_item_static_channels(split: dict, n_items: int = N_ITEMS) -> dict:
    """Append item one-hot channels to the observed V representation."""
    item_oh = torch.nn.functional.one_hot(split["item_idx"], num_classes=n_items).float()
    item_seq = item_oh[:, None, :].repeat(1, split["x"].shape[1], 1)
    out = dict(split)
    out["x"] = torch.cat([split["x"], item_seq], dim=-1)
    return out


def purged_time_folds(split: dict, k_folds: int = K_FOLDS, embargo: int = EMBARGO):
    """Yield train/eval masks over contiguous origin blocks with an origin embargo."""
    origins = split["origin"].numpy()
    unique_origins = np.array(sorted(np.unique(origins)))
    for fold_origins in np.array_split(unique_origins, k_folds):
        if len(fold_origins) == 0:
            continue
        min_t = int(fold_origins.min())
        max_t = int(fold_origins.max())
        eval_mask = np.isin(origins, fold_origins)
        train_mask = (origins < min_t - embargo) | (origins > max_t + embargo)
        if train_mask.sum() == 0 or eval_mask.sum() == 0:
            continue
        yield torch.as_tensor(train_mask), torch.as_tensor(eval_mask)


def theta_rmse(theta_hat, theta_star: np.ndarray) -> float:
    if np.isscalar(theta_hat):
        theta_arr = np.full_like(theta_star, float(theta_hat), dtype=np.float32)
    else:
        theta_arr = np.asarray(theta_hat, dtype=np.float32)
    return float(np.sqrt(np.mean((theta_arr - theta_star) ** 2)))


def train_d0_or_d1(name: str, tensors: dict, device: torch.device, response_mode: str, seed: int) -> dict:
    torch.manual_seed(seed)
    train = tensors["train"]
    test = tensors["test"]
    model = load_model(name, enc_in=train["x"].shape[-1]).to(device)

    if response_mode == "item":
        theta = torch.nn.Parameter(torch.zeros(N_ITEMS, device=device))
        include_item = True
    elif response_mode == "horizon":
        theta = torch.nn.Parameter(torch.zeros(H, device=device))
        include_item = False
    else:
        raise ValueError(f"unknown response_mode={response_mode}")

    loader = make_loader(train, device=device, include_item=include_item)
    opt = torch.optim.AdamW(list(model.parameters()) + [theta], lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    model.train()
    losses = []
    for _epoch in range(3):
        epoch_losses = []
        for batch in loader:
            if include_item:
                xb, xmb, xdmb, yb, phib, itemb = batch
                itemb = itemb.to(device)
            else:
                xb, xmb, xdmb, yb, phib = batch
            xb = xb.to(device)
            xmb = xmb.to(device)
            xdmb = xdmb.to(device)
            yb = yb.to(device)
            phib = phib.to(device)
            opt.zero_grad(set_to_none=True)
            base = forward_model(model, name, xb, xmb, xdmb)
            if response_mode == "item":
                pred = base + phib * theta[itemb][:, None]
            else:
                pred = base + phib * theta
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)))

    model.eval()
    with torch.no_grad():
        x = test["x"].to(device)
        xm = test["x_mark"].to(device)
        xdm = test["x_dec_mark"].to(device)
        y = test["y"].to(device)
        phi = test["phi_future"].to(device)
        if response_mode == "item":
            theta_eval = theta[test["item_idx"].to(device)][:, None]
            pred = forward_model(model, name, x, xm, xdm) + phi * theta_eval
            theta_for_rmse = theta.detach().cpu().numpy()
        else:
            pred = forward_model(model, name, x, xm, xdm) + phi * theta
            theta_for_rmse = float(theta.mean().detach().cpu())
        theta_hat = float(np.mean(theta.detach().cpu().numpy()))
        wmape = wmape_log(pred, y)

    if response_mode == "item":
        ser = float(np.mean(np.sign(theta_for_rmse) != np.sign(tensors["theta_star"])))
    else:
        ser = float(np.sign(theta_hat) != np.sign(tensors["theta_star_mean"]))
    rmse = theta_rmse(theta_for_rmse, tensors["theta_star"])
    return {
        "status": "complete",
        "response_mode": response_mode,
        "train_loss_first": round(losses[0], 5),
        "train_loss_last": round(losses[-1], 5),
        "obs_wmape": round(wmape, 4),
        "theta_hat": round(theta_hat, 4),
        "theta_star_mean": round(tensors["theta_star_mean"], 4),
        "theta_abs_error": round(rmse, 4),
        "ser": ser,
    }


def train_backbone_target(name: str, split: dict, device: torch.device, target: str, seed: int, epochs: int = 5):
    torch.manual_seed(seed)
    model = load_model(name, enc_in=split["x"].shape[-1]).to(device)
    loader = make_loader(split, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _epoch in range(epochs):
        for xb, xmb, xdmb, yb, phib in loader:
            xb = zero_treatment_channel(xb).to(device)
            xmb = xmb.to(device)
            xdmb = xdmb.to(device)
            y_target = (yb if target == "y" else phib).to(device)
            opt.zero_grad(set_to_none=True)
            pred = forward_model(model, name, xb, xmb, xdmb)
            loss = loss_fn(pred, y_target)
            loss.backward()
            opt.step()
    return model


def item_fe_adjust(train_pred: torch.Tensor, test_pred: torch.Tensor, train: dict, test: dict, target_key: str):
    """Calibrate deep nuisance residuals with static item fixed effects."""
    return item_fe_adjust_to_target(train_pred, test_pred, train, test, train[target_key])


def item_fe_adjust_to_target(
    train_pred: torch.Tensor,
    test_pred: torch.Tensor,
    train: dict,
    test: dict,
    train_target: torch.Tensor,
):
    """Calibrate arbitrary deep predictions to a train target using item fixed effects."""
    from sklearn.linear_model import Ridge

    y_train = train_target.numpy()
    train_raw = train_pred.numpy()
    test_raw = test_pred.numpy()
    n_items = int(max(train["item_idx"].max(), test["item_idx"].max()).item()) + 1
    fe_train = torch.nn.functional.one_hot(train["item_idx"], num_classes=n_items).numpy()
    fe_test = torch.nn.functional.one_hot(test["item_idx"], num_classes=n_items).numpy()
    reg = Ridge(alpha=1.0)
    reg.fit(fe_train, y_train - train_raw)
    train_adj = train_raw + reg.predict(fe_train)
    test_adj = test_raw + reg.predict(fe_test)
    raw_mse = float(np.mean((train_raw - y_train) ** 2))
    adj_mse = float(np.mean((train_adj - y_train) ** 2))
    return (
        torch.tensor(train_adj, dtype=torch.float32),
        torch.tensor(test_adj, dtype=torch.float32),
        {"raw_mse": round(raw_mse, 5), "item_fe_adjusted_mse": round(adj_mse, 5)},
    )


def v_feature_matrix(split: dict) -> np.ndarray:
    x = zero_treatment_channel(split["x"])
    mean = x.mean(dim=1)
    std = x.std(dim=1)
    last = x[:, -1, :]
    return torch.cat([mean, std, last], dim=1).numpy()


def ridge_adjust_to_target(
    train_pred: torch.Tensor,
    test_pred: torch.Tensor,
    train: dict,
    test: dict,
    train_target: torch.Tensor,
):
    """Calibrate a nuisance with a lightweight V-only residual model."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y_train = train_target.numpy()
    train_raw = train_pred.numpy()
    test_raw = test_pred.numpy()
    x_train = v_feature_matrix(train)
    x_test = v_feature_matrix(test)
    reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    reg.fit(x_train, y_train - train_raw)
    train_adj = train_raw + reg.predict(x_train)
    test_adj = test_raw + reg.predict(x_test)
    raw_mse = float(np.mean((train_raw - y_train) ** 2))
    adj_mse = float(np.mean((train_adj - y_train) ** 2))
    return (
        torch.tensor(train_adj, dtype=torch.float32),
        torch.tensor(test_adj, dtype=torch.float32),
        {"raw_mse": round(raw_mse, 5), "v_ridge_adjusted_mse": round(adj_mse, 5)},
    )


def batched_predict(model, name: str, split: dict, device: torch.device, v_only: bool) -> torch.Tensor:
    preds = []
    ds = TensorDataset(split["x"], split["x_mark"], split["x_dec_mark"])
    loader = DataLoader(ds, batch_size=128, shuffle=False, pin_memory=(device.type == "cuda"))
    model.eval()
    with torch.no_grad():
        for xb, xmb, xdmb in loader:
            if v_only:
                xb = zero_treatment_channel(xb)
            preds.append(forward_model(model, name, xb.to(device), xmb.to(device), xdmb.to(device)).cpu())
    return torch.cat(preds, dim=0)


def cross_fitted_nuisances(name: str, train: dict, device: torch.device, seed: int):
    m_oof = torch.zeros_like(train["y"])
    pi_oof = torch.zeros_like(train["phi_future"])
    covered = torch.zeros(train["y"].shape[0], dtype=torch.bool)
    fold_diags = []

    for fold_id, (fit_mask, eval_mask) in enumerate(purged_time_folds(train)):
        fit_split = select_split(train, fit_mask)
        eval_split = select_split(train, eval_mask)
        fold_seed = seed + 1009 * (fold_id + 1)

        m_model = train_backbone_target(name, fit_split, device, target="y", seed=fold_seed, epochs=3)
        pi_model = train_backbone_target(name, fit_split, device, target="phi", seed=fold_seed + 17, epochs=3)

        m_fit_raw = batched_predict(m_model, name, fit_split, device, v_only=True)
        m_eval_raw = batched_predict(m_model, name, eval_split, device, v_only=True)
        _, m_eval, m_diag = item_fe_adjust(m_fit_raw, m_eval_raw, fit_split, eval_split, target_key="y")

        pi_fit_raw = batched_predict(pi_model, name, fit_split, device, v_only=True)
        pi_eval_raw = batched_predict(pi_model, name, eval_split, device, v_only=True)
        _, pi_eval, pi_diag = item_fe_adjust(pi_fit_raw, pi_eval_raw, fit_split, eval_split, target_key="phi_future")

        m_oof[eval_mask] = m_eval
        pi_oof[eval_mask] = pi_eval
        covered[eval_mask] = True
        fold_diags.append({
            "fold": fold_id,
            "fit_n": int(fit_mask.sum().item()),
            "eval_n": int(eval_mask.sum().item()),
            "m": m_diag,
            "pi": pi_diag,
        })

    if not bool(torch.all(covered)):
        missing = ~covered
        fallback_model_m = train_backbone_target(name, train, device, target="y", seed=seed + 9001, epochs=3)
        fallback_model_pi = train_backbone_target(name, train, device, target="phi", seed=seed + 9018, epochs=3)
        m_raw = batched_predict(fallback_model_m, name, train, device, v_only=True)
        pi_raw = batched_predict(fallback_model_pi, name, train, device, v_only=True)
        m_adj, _, _ = item_fe_adjust(m_raw, m_raw, train, train, target_key="y")
        pi_adj, _, _ = item_fe_adjust(pi_raw, pi_raw, train, train, target_key="phi_future")
        m_oof[missing] = m_adj[missing]
        pi_oof[missing] = pi_adj[missing]

    return m_oof, pi_oof, {
        "folds": fold_diags,
        "covered_share": round(float(covered.float().mean().item()), 4),
        "oof_m_mse": round(float(torch.mean((m_oof - train["y"]) ** 2).item()), 5),
        "oof_pi_mse": round(float(torch.mean((pi_oof - train["phi_future"]) ** 2).item()), 5),
    }


def train_d2(name: str, tensors: dict, device: torch.device, seed: int) -> dict:
    train = add_item_static_channels(tensors["train"])
    test = add_item_static_channels(tensors["test"])

    m_oof, pi_oof, oof_diag = cross_fitted_nuisances(name, train, device, seed)

    m_model = train_backbone_target(name, train, device, target="y", seed=seed + 20003)
    pi_model = train_backbone_target(name, train, device, target="phi", seed=seed + 20021)
    m_tr_raw = batched_predict(m_model, name, train, device, v_only=True)
    m_te_raw = batched_predict(m_model, name, test, device, v_only=True)
    pi_tr_raw = batched_predict(pi_model, name, train, device, v_only=True)
    pi_te_raw = batched_predict(pi_model, name, test, device, v_only=True)

    m_tr, m_te, m_diag = item_fe_adjust(m_tr_raw, m_te_raw, train, test, target_key="y")
    pi_tr, pi_te, pi_diag = item_fe_adjust(pi_tr_raw, pi_te_raw, train, test, target_key="phi_future")

    y_tr = train["y"]
    phi_tr = train["phi_future"]
    y_res = y_tr - m_oof
    phi_res = phi_tr - pi_oof
    theta_global = float((phi_res * y_res).sum().item() / ((phi_res * phi_res).sum().item() + 1e-8))

    item_theta = np.full(N_ITEMS, theta_global, dtype=np.float32)
    item_idx = train["item_idx"].numpy()
    phi_res_np = phi_res.numpy()
    y_res_np = y_res.numpy()
    for item in range(N_ITEMS):
        mask = item_idx == item
        denom = float(np.sum(phi_res_np[mask] * phi_res_np[mask]))
        if denom > 1e-6:
            item_theta[item] = float(np.sum(phi_res_np[mask] * y_res_np[mask]) / denom)

    # Light shrinkage keeps item-level response estimates stable while still
    # allowing the structural head to represent heterogeneous effects.
    theta = THETA_ITEM_WEIGHT * item_theta + (1.0 - THETA_ITEM_WEIGHT) * theta_global

    # Equivalent structural form to m(V) + theta * (phi - pi(V)), but more
    # stable for finite-sample forecasting because the final base learns
    # mu(V) = E[y - theta * phi | V] directly instead of subtracting two
    # separately estimated nuisances at test time.
    theta_tr = torch.tensor(theta[train["item_idx"].numpy()], dtype=torch.float32)[:, None]
    theta_te = torch.tensor(theta[test["item_idx"].numpy()], dtype=torch.float32)[:, None]
    mu_tr_raw = m_tr_raw - theta_tr * pi_tr_raw
    mu_te_raw = m_te_raw - theta_te * pi_te_raw
    mu_target = train["y"] - theta_tr * train["phi_future"]
    _, mu_te, mu_diag = ridge_adjust_to_target(mu_tr_raw, mu_te_raw, train, test, mu_target)
    pred = mu_te + theta_te * test["phi_future"]
    wmape = wmape_log(pred, test["y"])
    theta_mean = float(np.mean(theta))
    ser = float(np.mean(np.sign(theta) != np.sign(tensors["theta_star"])))
    rmse = theta_rmse(theta, tensors["theta_star"])
    return {
        "status": "complete",
        "obs_wmape": round(wmape, 4),
        "theta_hat": round(theta_mean, 4),
        "theta_star_mean": round(tensors["theta_star_mean"], 4),
        "theta_abs_error": round(rmse, 4),
        "ser": ser,
        "nuisance_diagnostics": {
            "m": m_diag,
            "pi": pi_diag,
            "mu": mu_diag,
            "cross_fit": oof_diag,
            "residual_phi_variance": round(float(torch.var(phi_res).item()), 5),
            "theta_global": round(theta_global, 4),
            "theta_item_weight": THETA_ITEM_WEIGHT,
            "theta_item_min": round(float(np.min(theta)), 4),
            "theta_item_max": round(float(np.max(theta)), 4),
        },
    }


def run_protocol(name: str, tensors: dict, device: torch.device, seed: int) -> dict:
    fair_tensors = {
        **tensors,
        "train": add_item_static_channels(tensors["train"]),
        "test": add_item_static_channels(tensors["test"]),
    }
    d0 = train_d0_or_d1(name, fair_tensors, device, response_mode="horizon", seed=seed)
    d1 = train_d0_or_d1(name, fair_tensors, device, response_mode="item", seed=seed + 101)
    d2 = train_d2(name, tensors, device, seed=seed + 202)
    d2_vs_d0_ser_reduction = (d0["ser"] - d2["ser"]) / (d0["ser"] + 1e-8) if d0["ser"] > 0 else 0.0
    d2_vs_d0_theta_error_reduction = (d0["theta_abs_error"] - d2["theta_abs_error"]) / (d0["theta_abs_error"] + 1e-8)
    d2_vs_d1_theta_error_reduction = (d1["theta_abs_error"] - d2["theta_abs_error"]) / (d1["theta_abs_error"] + 1e-8)
    obs_loss_increase = (d2["obs_wmape"] - d0["obs_wmape"]) / (d0["obs_wmape"] + 1e-8)
    return {
        "backbone": name,
        "seed": seed,
        "train_end": tensors["train_end"],
        "test_start": tensors["test_start"],
        "test_end": tensors["test_end"],
        "status": "complete",
        "fairness": "D0/D1/D2 share item static controls; D1/D2 share item-specific response capacity",
        "d0": d0,
        "d1": d1,
        "d2": d2,
        "d2_vs_d0_ser_reduction": round(float(d2_vs_d0_ser_reduction), 4),
        "d2_vs_d0_theta_error_reduction": round(float(d2_vs_d0_theta_error_reduction), 4),
        "d2_vs_d1_theta_error_reduction": round(float(d2_vs_d1_theta_error_reduction), 4),
        "d2_vs_d0_obs_loss_increase": round(float(obs_loss_increase), 4),
        "protocol_pass": bool(
            d2["theta_abs_error"] <= d1["theta_abs_error"]
            and obs_loss_increase <= 0.05
        ),
    }


def mean_numeric(values: list[float], ndigits: int = 4):
    return round(float(np.mean(values)), ndigits)


def aggregate_metric_dict(dicts: list[dict]) -> dict:
    out = {"status": "complete"}
    for key in dicts[0]:
        values = [d.get(key) for d in dicts]
        if key == "nuisance_diagnostics":
            continue
        if isinstance(values[0], (int, float, np.floating)) and all(isinstance(v, (int, float, np.floating)) for v in values):
            out[key] = mean_numeric(values)
        elif all(v == values[0] for v in values):
            out[key] = values[0]
    if "nuisance_diagnostics" in dicts[0]:
        out["nuisance_diagnostics"] = {
            "note": "seed-level diagnostics are stored in seed_rows; aggregate row reports primary metrics only"
        }
    return out


def aggregate_backbone_rows(name: str, seed_rows: list[dict]) -> dict:
    complete = [r for r in seed_rows if r.get("status") == "complete"]
    if not complete:
        return {"backbone": name, "status": "failed", "n_seeds": 0}
    mean_theta_pass = mean_numeric([r["d1"]["theta_abs_error"] - r["d2"]["theta_abs_error"] for r in complete]) >= 0
    max_obs_loss_increase = max(float(r["d2_vs_d0_obs_loss_increase"]) for r in complete)
    mean_obs_loss_increase = mean_numeric([r["d2_vs_d0_obs_loss_increase"] for r in complete])
    n_seed_protocol_pass = sum(1 for r in complete if r.get("protocol_pass"))
    return {
        "backbone": name,
        "status": "complete",
        "n_seeds": len(complete),
        "seeds": sorted(int(r["seed"]) for r in complete),
        "protocol_criteria": {
            "theta_error": "D2 theta_abs_error <= D1 theta_abs_error",
            "obs_loss": "D2 observational WMAPE increase <= 5% versus D0",
            "strict_pass": "all completed seeds satisfy both criteria",
            "mean_pass": "seed-mean theta and seed-mean WMAPE satisfy both criteria",
        },
        "fairness": complete[0]["fairness"],
        "split": {
            "embargo": EMBARGO,
            "train_end_mean": mean_numeric([r["train_end"] for r in complete], ndigits=1),
            "test_start_mean": mean_numeric([r["test_start"] for r in complete], ndigits=1),
            "test_end_mean": mean_numeric([r["test_end"] for r in complete], ndigits=1),
        },
        "d0": aggregate_metric_dict([r["d0"] for r in complete]),
        "d1": aggregate_metric_dict([r["d1"] for r in complete]),
        "d2": aggregate_metric_dict([r["d2"] for r in complete]),
        "d2_vs_d0_ser_reduction": mean_numeric([r["d2_vs_d0_ser_reduction"] for r in complete]),
        "d2_vs_d0_theta_error_reduction": mean_numeric([r["d2_vs_d0_theta_error_reduction"] for r in complete]),
        "d2_vs_d1_theta_error_reduction": mean_numeric([r["d2_vs_d1_theta_error_reduction"] for r in complete]),
        "d2_vs_d0_obs_loss_increase": mean_obs_loss_increase,
        "max_seed_d2_vs_d0_obs_loss_increase": round(float(max_obs_loss_increase), 4),
        "n_seed_protocol_pass": int(n_seed_protocol_pass),
        "mean_protocol_pass": bool(mean_theta_pass and mean_obs_loss_increase <= 0.05),
        "protocol_pass": bool(all(r.get("protocol_pass") for r in complete)),
    }


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run DoCast M6 corrected backbone protocol.")
    parser.add_argument("--backbones", default=",".join(BACKBONES),
                        help="Comma-separated backbone names to run.")
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS),
                        help="Comma-separated integer seeds to run.")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Refresh the JSON aggregate from existing seed_rows without rerunning models.")
    return parser.parse_args()


def write_summary(selected_backbones: list[str], selected_seeds: list[int], seed_rows: list[dict], device: torch.device) -> None:
    rows = [
        aggregate_backbone_rows(name, [r for r in seed_rows if r.get("backbone") == name])
        for name in selected_backbones
    ]

    complete = [r for r in rows if r.get("status") == "complete"]
    deep_complete = [r for r in complete if r["backbone"] in DEEP_BACKBONES]
    deep_protocol_pass = [r for r in deep_complete if r.get("protocol_pass")]
    summary = {
        "milestone": "M6",
        "status": "complete" if complete else "failed",
        "scope": "lightweight deep-backbone D0/D1/D2 protocol on DoCast semi-synthetic sample",
        "device": str(device),
        "gamma": GAMMA,
        "seeds": selected_seeds,
        "k_folds": K_FOLDS,
        "embargo": EMBARGO,
        "n_backbones_complete": len(complete),
        "n_deep_backbones_complete": len(deep_complete),
        "n_deep_protocol_pass": len(deep_protocol_pass),
        "full_docast_protocol_complete": bool(len(deep_complete) >= 3 and len(deep_protocol_pass) >= 3),
        "rows": rows,
        "seed_rows": seed_rows,
    }
    out = OUT_DIR / "backbone_sweep_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"M6 backbone sweep → {out}")
    for row in rows:
        print(row)


def main() -> None:
    args = parse_args()
    selected_backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    selected_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    device = select_device()
    print(f"Using device: {device}")
    data = load_m5()
    out = OUT_DIR / "backbone_sweep_summary.json"
    if out.exists():
        with open(out) as f:
            existing = json.load(f)
        if args.summarize_only:
            seed_rows = existing.get("seed_rows", [])
        else:
            stale_keys = {(b, s) for b in selected_backbones for s in selected_seeds}
            seed_rows = [
                r for r in existing.get("seed_rows", [])
                if (r.get("backbone"), r.get("seed")) not in stale_keys
            ]
    else:
        seed_rows = []
    if args.summarize_only:
        if not seed_rows:
            raise RuntimeError("No existing M6 seed_rows found for --summarize-only.")
        all_seeds = sorted({int(r["seed"]) for r in seed_rows if "seed" in r})
        write_summary(BACKBONES, all_seeds, seed_rows, device)
        return
    for name in selected_backbones:
        for seed in selected_seeds:
            print(f"Running D0/D1/D2 protocol for {name} seed={seed}...", flush=True)
            tensors = build_tensor_data(data, gamma=GAMMA, seed=seed)
            try:
                seed_rows.append(run_protocol(name, tensors, device, seed=seed))
            except Exception as exc:
                seed_rows.append({"backbone": name, "seed": seed, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            all_seeds = sorted({int(r["seed"]) for r in seed_rows if "seed" in r})
            write_summary(BACKBONES, all_seeds, seed_rows, device)


if __name__ == "__main__":
    main()
