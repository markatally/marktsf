"""
Finance-MISO baseline prediction producer for PRISM M1.

Trains one baseline model on Crypto hourly MISO features (close log-returns +
optional volume log-returns, BTCUSDT as target column -1) and saves
``pred.npy`` / ``true.npy`` in the artifact contract expected by
``experiments.PRISM.oracle_drift``.

Results land in ``external/TSLib/results/`` (gitignored) under a directory
whose name matches oracle_drift's resolve_result_dir prefix pattern:
    long_term_forecast_{dataset_tag}_{L}_{H}_{model}_{dataset_tag}_{suffix}

Usage
-----
# proper MISO run (close + volume, features=MS, full-size models)
python -m experiments.PRISM.produce_predictions \\
    --model DLinear --lookback 96 --horizon 96 \\
    --include-volume --features MS --dataset-tag CryptoMISO

# run all 4 baselines × 4 horizons
for model in DLinear PatchTST iTransformer TimeMixer; do
  for h in 24 48 96 168; do
    python -m experiments.PRISM.produce_predictions \\
        --model $model --horizon $h \\
        --include-volume --features MS --dataset-tag CryptoMISO
  done
done
"""

from __future__ import annotations

import argparse
import importlib
import logging
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Add external/TSLib to sys.path so TSLib model imports work.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TSLIB = _REPO_ROOT / "external" / "TSLib"
if str(_TSLIB) not in sys.path:
    sys.path.insert(0, str(_TSLIB))

from experiments.PRISM.data.crypto_dataset import (  # noqa: E402
    ASSETS,
    CryptoDataset,
    build_crypto_panel,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Model registry.  TimeMixer replaces TimesNet (broken FFT in torch 2.x).
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: dict[str, str] = {
    "DLinear": "models.DLinear",
    "PatchTST": "models.PatchTST",
    "iTransformer": "models.iTransformer",
    "TimeMixer": "models.TimeMixer",
}


def _import_model_class(name: str):
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Choices: {list(_MODEL_REGISTRY)}")
    mod = importlib.import_module(_MODEL_REGISTRY[name])
    return mod.Model


# ---------------------------------------------------------------------------
# Configs namespace — full-size, per-model tuned, all four models covered.
# ---------------------------------------------------------------------------
def _build_configs(
    model_name: str,
    n_features: int,
    seq_len: int,
    pred_len: int,
    label_len: int,
) -> SimpleNamespace:
    """
    Return a configs namespace with paper-level hyperparameters for each model.

    Sizes chosen so every model has ≥ 100 K parameters — large enough for
    inductive biases to manifest in predictions — while staying trainable on
    MPS in a few minutes per run.

    Model       params (28 features, H=96)
    -------     --------
    DLinear     ~18 K   (linear; size doesn't scale with d_model)
    PatchTST    ~547 K  (d_model=128, n_heads=8, e_layers=3)
    iTransformer ~1.1M  (d_model=256, n_heads=4, e_layers=2)
    TimeMixer   ~91 K   (d_model=64, e_layers=2, cross-channel)
    """
    cfg = SimpleNamespace(
        # task
        task_name="long_term_forecast",
        features="M",
        # data shape — c_out stays enc_in; MS slicing is done externally
        enc_in=n_features,
        dec_in=n_features,
        c_out=n_features,
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len,
        # shared defaults
        d_model=128,
        n_heads=8,
        e_layers=3,
        d_layers=1,
        d_ff=256,
        moving_avg=25,
        dropout=0.1,
        activation="gelu",
        factor=1,
        distil=True,
        freq="h",
        embed="timeF",
        use_norm=1,
        # DLinear
        individual=False,
        # iTransformer
        class_strategy="projection",
        output_attention=False,
        # TimeMixer
        channel_independence=0,          # CD: cross-channel mixing for MISO
        decomp_method="moving_avg",
        down_sampling_layers=3,
        down_sampling_window=2,
        down_sampling_method="avg",
        top_k=5,
        # misc
        use_amp=False,
        augmentation_ratio=0,
        expand=2,
        d_conv=4,
        num_kernels=6,
    )
    # Per-model overrides — tuned to paper-level sizes.
    if model_name == "iTransformer":
        cfg.d_model = 256
        cfg.n_heads = 4
        cfg.e_layers = 2
        cfg.d_ff = 512
    elif model_name == "TimeMixer":
        cfg.d_model = 64
        cfg.d_ff = 64
        cfg.e_layers = 2
    return cfg


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _forward(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    batch_x_mark: torch.Tensor,
    batch_y_mark: torch.Tensor,
    pred_len: int,
    label_len: int,
    f_dim: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    One forward pass.  Returns (output, target) both sliced to [:, pred_len, f_dim:].

    f_dim=0  → features='M' (all channels in loss)
    f_dim=-1 → features='MS' (target channel only in loss)
    """
    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :])
    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1)
    out = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    return out[:, -pred_len:, f_dim:], batch_y[:, -pred_len:, f_dim:]


def _val_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    pred_len: int,
    label_len: int,
    f_dim: int,
) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            out, target = _forward(
                model, batch_x, batch_y, batch_x_mark, batch_y_mark,
                pred_len, label_len, f_dim,
            )
            total += criterion(out, target).item() * batch_x.size(0)
            count += batch_x.size(0)
    model.train()
    return total / count if count else float("inf")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    pred_len: int,
    label_len: int,
    f_dim: int = 0,
    lr: float = 1e-3,
    max_epochs: int = 20,
    patience: int = 5,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state: dict | None = None
    wait = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        t0 = time.time()
        last_train_loss = 0.0
        for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            out, target = _forward(
                model, batch_x, batch_y, batch_x_mark, batch_y_mark,
                pred_len, label_len, f_dim,
            )
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last_train_loss = loss.item()

        val_loss = _val_loss(
            model, val_loader, criterion, device, pred_len, label_len, f_dim,
        )
        log.info(
            "epoch %2d | train=%.6f  val=%.6f  [%.1fs]",
            epoch, last_train_loss, val_loss, time.time() - t0,
        )

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log.info("early stop at epoch %d (best val=%.6f)", epoch, best_val)
                break

    if best_state is not None:
        model.load_state_dict(best_state)


def collect_predictions(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    pred_len: int,
    label_len: int,
    f_dim: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (preds, trues) arrays of shape [n_windows, H, C] where C=1 for MS."""
    model.eval()
    preds_list: list[np.ndarray] = []
    trues_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            out, target = _forward(
                model, batch_x, batch_y, batch_x_mark, batch_y_mark,
                pred_len, label_len, f_dim,
            )
            preds_list.append(out.cpu().numpy())
            trues_list.append(target.cpu().numpy())

    return np.concatenate(preds_list, axis=0), np.concatenate(trues_list, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(
        "device: %s | model: %s | L=%d H=%d | features=%s | vol=%s",
        device, args.model, args.lookback, args.horizon,
        args.features, args.include_volume,
    )

    # f_dim=-1 for MS (target channel only), 0 for M (all channels).
    f_dim = -1 if args.features == "MS" else 0

    # Build panel in-memory; nothing written to disk.
    panel = build_crypto_panel(
        Path(args.input_dir), freq=args.freq, include_volume=args.include_volume,
    )
    n_features = panel.shape[1]
    label_len = min(48, args.lookback // 2)

    configs = _build_configs(args.model, n_features, args.lookback, args.horizon, label_len)

    ds_kwargs = dict(seq_len=args.lookback, pred_len=args.horizon, label_len=label_len)
    train_ds = CryptoDataset(panel, "train", **ds_kwargs)
    val_ds   = CryptoDataset(panel, "val",   **ds_kwargs)
    test_ds  = CryptoDataset(panel, "test",  **ds_kwargs)
    log.info(
        "features=%d | windows → train=%d  val=%d  test=%d",
        n_features, len(train_ds), len(val_ds), len(test_ds),
    )

    loader_kwargs = dict(num_workers=0, pin_memory=device.type == "cuda")  # pin_memory only safe on CUDA
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, **loader_kwargs,
    )

    # PatchTST exposes patch_len/stride as constructor kwargs.
    extra: dict = {}
    if args.model == "PatchTST":
        extra = {"patch_len": 16, "stride": 8}

    ModelClass = _import_model_class(args.model)
    model = ModelClass(configs, **extra).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("params: %d", n_params)

    train_model(
        model, train_loader, val_loader, device,
        pred_len=args.horizon, label_len=label_len, f_dim=f_dim,
        lr=args.lr, max_epochs=args.epochs, patience=args.patience,
    )

    preds, trues = collect_predictions(
        model, test_loader, device,
        pred_len=args.horizon, label_len=label_len, f_dim=f_dim,
    )
    log.info("preds %s  trues %s", preds.shape, trues.shape)

    # Directory name matches oracle_drift.resolve_result_dir's prefix pattern:
    #   long_term_forecast_{dataset}_{L}_{H}_{model}_{dataset}_
    tag = args.dataset_tag
    feat_str = f"ft{args.features}"
    suffix = f"{feat_str}_sl{args.lookback}_pl{args.horizon}_seed{args.seed}"
    result_dir = (
        Path(args.results_root)
        / f"long_term_forecast_{tag}_{args.lookback}_{args.horizon}_{args.model}_{tag}_{suffix}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / "pred.npy", preds)
    np.save(result_dir / "true.npy", trues)
    log.info("saved → %s", result_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PRISM M1 — Crypto MISO prediction producer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input-dir", default="input/Crypto",
        help="path to Crypto CSV directory (read-only; nothing written here)",
    )
    p.add_argument(
        "--results-root", default="external/TSLib/results",
        help="root directory for pred/true artifacts (gitignored)",
    )
    p.add_argument("--model", choices=list(_MODEL_REGISTRY), default="DLinear")
    p.add_argument("--lookback", type=int, default=96, help="lookback L (hours)")
    p.add_argument("--horizon", type=int, default=96, help="forecast horizon H (hours)")
    p.add_argument("--freq", default="1h", help="pandas resample frequency")
    p.add_argument(
        "--features", choices=["M", "MS"], default="MS",
        help="M=predict all channels; MS=predict target channel only (MISO)",
    )
    p.add_argument(
        "--include-volume", action="store_true", default=False,
        help="add volume log-returns as MISO covariates (28 features instead of 14)",
    )
    p.add_argument(
        "--dataset-tag", default="CryptoMISO",
        help="tag used in result directory names and oracle_drift --dataset",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5, help="early-stopping patience")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA/MPS available")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
