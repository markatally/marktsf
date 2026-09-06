"""External StatsForecast/GluonTS runtime for the official-style naive control."""
import argparse
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from gluonts.time_feature import get_seasonality
from statsforecast.models import SeasonalNaive


def run(input_dir, output_dir, horizon=32):
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(Path(input_dir).glob("*.npz")):
        with np.load(path) as data:
            season = get_seasonality(str(data["freq"]))
            history = data["context"].astype(np.float32)
            result = SeasonalNaive(season).forecast(y=history, h=horizon, level=[0, 20, 40, 60, 80])
            columns = []
            for i in range(1, 10):
                interval = abs(20*i-100)
                columns.append(result[f'{"lo" if i<=5 else "hi"}-{interval}'])
            quantiles = np.stack(columns, -1)
            if not np.isfinite(quantiles).all():
                raise ValueError(f"Nonfinite official naive forecast for {path.name}; no silent substitution")
            np.savez_compressed(dest/path.name, key=data["key"], target_hash=data["target_hash"],
                                quantiles=quantiles, season=season)
    versions = {k: importlib.metadata.version(k) for k in ("numpy", "pandas", "statsforecast", "gluonts")}
    (dest/"runtime.json").write_text(json.dumps(versions, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", required=True)
    run(**json.loads(parser.parse_args().parameters))
