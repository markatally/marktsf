from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.PRISM.oracle_drift import StudyConfig, load_window_mse, run_study


def _write_result(root: Path, dataset: str, lookback: int, horizon: int, model: str, pred: np.ndarray, true: np.ndarray) -> None:
    result_dir = root / f"long_term_forecast_{dataset}_{lookback}_{horizon}_{model}_{dataset}_unit_0"
    result_dir.mkdir(parents=True)
    np.save(result_dir / "pred.npy", pred)
    np.save(result_dir / "true.npy", true)


def test_load_window_mse_scores_selected_target_channel(tmp_path: Path) -> None:
    result_dir = tmp_path / "cell"
    result_dir.mkdir()
    true = np.zeros((2, 3, 2), dtype=float)
    pred = np.zeros_like(true)
    pred[0, :, 1] = [1, 2, 3]
    pred[1, :, 1] = [2, 2, 2]
    pred[:, :, 0] = 100
    np.save(result_dir / "pred.npy", pred)
    np.save(result_dir / "true.npy", true)

    losses = load_window_mse(result_dir, target_channel=-1)

    np.testing.assert_allclose(losses, np.array([(1 + 4 + 9) / 3, 4.0]))


def test_run_study_writes_oracle_artifacts(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    output_dir = tmp_path / "out"
    dataset = "Toy"
    lookback = 4
    horizon = 2
    true = np.zeros((4, horizon, 1), dtype=float)

    # Model A wins windows 0 and 2; model B wins windows 1 and 3.
    pred_a = np.array([[[0], [0]], [[3], [3]], [[1], [1]], [[4], [4]]], dtype=float)
    pred_b = np.array([[[2], [2]], [[0], [0]], [[2], [2]], [[0], [0]]], dtype=float)
    _write_result(results_root, dataset, lookback, horizon, "A", pred_a, true)
    _write_result(results_root, dataset, lookback, horizon, "B", pred_b, true)

    summary = run_study(
        StudyConfig(
            results_root=str(results_root),
            output_dir=str(output_dir),
            dataset=dataset,
            lookback=lookback,
            horizon=horizon,
            models=("A", "B"),
            target_channel=-1,
        )
    )

    assert summary["best_single_model"] == "B"
    assert summary["switch_count"] == 3
    assert summary["switch_rate"] == 1.0
    assert summary["oracle_gap_abs"] > 0
    assert (output_dir / "window_losses.csv").exists()
    assert (output_dir / "best_architecture_trajectory.csv").exists()
    assert (output_dir / "best_architecture_trajectory.png").exists()
    saved = json.loads((output_dir / "summary.json").read_text())
    assert saved["win_counts"] == {"A": 2, "B": 2}
