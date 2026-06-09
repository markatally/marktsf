"""Authoritative baseline reproduction via the ORIGINAL authors' code.

Runs thuml/Time-Series-Library (vendored at external/TSLib) using each model's
official script, so reproduced numbers come from the authors' exact code +
hyperparameters — the only honest way to hit a ≤2% reproduction bar.

What this wrapper does per official script:
  1. parse the ``python -u run.py ...`` invocations (one per horizon),
  2. override the data root to our input/ETT/, force CPU + num_workers=0
     (TSLib's default 10 workers are ~270× slower under macOS spawn),
  3. run it, parse ``mse:.., mae:..`` from stdout,
  4. gate the result against reference.py.

Usage:
    python -m spectre.reproduce.official --models DLinear,PatchTST --datasets ETTh1
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

from ..experiments.reference import (
    PAPER_MSE,
    REPRO_MSE,
    gate_status,
)

__all__ = ["reproduce", "run_official_script"]

# Repo + vendored-repo locations.
_REPO = Path(__file__).resolve().parents[2]
_TSLIB = _REPO / "external" / "TSLib"
_ETT_ROOT = _REPO / "input" / "ETT"
_ETT_SCRIPTS = _TSLIB / "scripts" / "long_term_forecast" / "ETT_script"

# Regex for TSLib's final test line, e.g. "mse:0.3962, mae:0.4108, dtw:...".
_RESULT_RE = re.compile(r"mse:([0-9.]+),\s*mae:([0-9.]+)")


def _script_path(model: str, dataset: str) -> Path | None:
    """Locate the official script for (model, dataset); None if absent."""
    p = _ETT_SCRIPTS / f"{model}_{dataset}.sh"
    return p if p.exists() else None


def _parse_invocations(script_text: str) -> list[list[str]]:
    """Extract each run.py invocation from a script as an arg list.

    Handles ``model_name=X`` variable substitution and backslash line
    continuations; returns args WITHOUT the leading ``python -u run.py``.
    """
    # Capture a `model_name=Foo` assignment for $model_name substitution.
    m = re.search(r"model_name\s*=\s*(\S+)", script_text)
    model_name = m.group(1) if m else ""
    text = script_text.replace("$model_name", model_name).replace(
        "${model_name}", model_name
    )
    # Fold backslash-newline continuations into single logical lines.
    text = text.replace("\\\n", " ")
    invocations: list[list[str]] = []
    for line in text.splitlines():
        if "run.py" not in line:
            continue
        tokens = shlex.split(line)
        # Drop the interpreter prefix tokens up to and including 'run.py'.
        if "run.py" in tokens:
            tokens = tokens[tokens.index("run.py") + 1 :]
        invocations.append(tokens)
    return invocations


def _override_args(args: list[str], root_path: Path) -> list[str]:
    """Force our data root, CPU execution, and 0 dataloader workers."""
    drop_with_value = {"--root_path", "--num_workers", "--gpu_type", "--gpu"}
    drop_flag = {"--use_gpu"}
    out: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in drop_with_value:
            i += 2  # skip flag + its value
            continue
        if tok in drop_flag:
            i += 1  # skip valueless flag
            continue
        out.append(tok)
        i += 1
    # Re-add our overrides (CPU + single-process loading + our data root).
    out += [
        "--root_path", str(root_path),
        "--no_use_gpu",
        "--num_workers", "0",
    ]
    return out


def _arg_value(args: list[str], flag: str, default: str = "") -> str:
    return args[args.index(flag) + 1] if flag in args else default


def run_official_script(
    model: str, dataset: str, horizons: set[int] | None, epochs: int | None = None
) -> list[dict]:
    """Run every (matching-horizon) invocation in a model's official script."""
    sp = _script_path(model, dataset)
    if sp is None:
        print(f"  [skip] no official script: {model}_{dataset}.sh")
        return []
    rows: list[dict] = []
    for args in _parse_invocations(sp.read_text()):
        pred_len = int(_arg_value(args, "--pred_len", "0"))
        if horizons and pred_len not in horizons:
            continue
        cmd_args = _override_args(args, _ETT_ROOT)
        if epochs is not None:  # optional fast smoke override
            cmd_args = _set_arg(cmd_args, "--train_epochs", str(epochs))
        print(f"  [run] {model} {dataset} pred_len={pred_len} ...", flush=True)
        proc = subprocess.run(
            [sys.executable, "run.py", *cmd_args],
            cwd=_TSLIB,
            capture_output=True,
            text=True,
        )
        match = _RESULT_RE.search(proc.stdout)
        if not match:
            print(f"    FAILED to parse result (exit {proc.returncode}). Last stderr:")
            print("    " + "\n    ".join(proc.stderr.strip().splitlines()[-3:]))
            continue
        our = float(match.group(1))
        rows.append(_make_row(dataset, model, pred_len, our))
        _print_row(rows[-1])
    return rows


def _set_arg(args: list[str], flag: str, value: str) -> list[str]:
    out = list(args)
    if flag in out:
        out[out.index(flag) + 1] = value
    else:
        out += [flag, value]
    return out


def _make_row(dataset: str, model: str, pred_len: int, our: float) -> dict:
    key = (dataset, model, pred_len)
    ref = REPRO_MSE.get(key)
    return {
        "dataset": dataset,
        "model": model,
        "H": pred_len,
        "our_mse": our,
        "repro_ref": ref,
        "paper": PAPER_MSE.get(key),
        "status": gate_status(our, ref),
    }


def _print_row(r: dict) -> None:
    ref = "   --  " if r["repro_ref"] is None else f"{r['repro_ref']:.4f}"
    paper = "  -- " if r["paper"] is None else f"{r['paper']:.3f}"
    print(
        f"    {r['dataset']:>6} | {r['model']:>13} | H={r['H']:>3} | "
        f"our={r['our_mse']:.4f} | repro_ref={ref} | paper={paper} | {r['status']}"
    )


def reproduce(
    models: list[str], datasets: list[str], horizons: set[int] | None, epochs: int | None
) -> list[dict]:
    if not _TSLIB.exists():
        raise FileNotFoundError(
            f"vendored TSLib not found at {_TSLIB}. Clone it first:\n"
            "  git clone --depth 1 https://github.com/thuml/Time-Series-Library.git external/TSLib"
        )
    rows: list[dict] = []
    for ds in datasets:
        for model in models:
            rows += run_official_script(model, ds, horizons, epochs)
    _summary(rows)
    return rows


def _summary(rows: list[dict]) -> None:
    from collections import Counter

    c = Counter(r["status"] for r in rows)
    print("=" * 78)
    print(f"  REPRODUCTION GATE (≤2% vs original code)  {dict(c)}")
    print("  RESULT:", "FAIL" if c.get("FAIL") else "OK")
    print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="DLinear,PatchTST,iTransformer,TiDE,TimeXer")
    ap.add_argument("--datasets", default="ETTh1,ETTh2,ETTm1,ETTm2")
    ap.add_argument("--horizons", default="", help="comma list; empty = all in script")
    ap.add_argument("--epochs", type=int, default=None, help="override train_epochs")
    args = ap.parse_args()
    horizons = {int(h) for h in args.horizons.split(",") if h.strip()} or None
    reproduce(
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        datasets=[d.strip() for d in args.datasets.split(",") if d.strip()],
        horizons=horizons,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
