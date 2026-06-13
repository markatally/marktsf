#!/usr/bin/env bash
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

PYTHON=/opt/homebrew/Caskroom/miniforge/base/bin/python

run() {
  echo "=== $* ==="
  $PYTHON -m experiments.PRISM.produce_predictions \
    --target vol --dataset-tag CryptoVol \
    --lookback 96 --patience 5 --epochs 20 \
    "$@"
}

echo "--- TimeMixer seed2021 ---"
run --model TimeMixer --horizon 24 --seed 2021
run --model TimeMixer --horizon 96 --seed 2021

echo "--- seed2022 missing ---"
run --model PatchTST    --horizon 24 --seed 2022
run --model PatchTST    --horizon 96 --seed 2022
run --model iTransformer --horizon 24 --seed 2022
run --model iTransformer --horizon 96 --seed 2022
run --model TimeMixer   --horizon 24 --seed 2022
run --model TimeMixer   --horizon 96 --seed 2022

echo "--- seed2023 missing ---"
run --model PatchTST    --horizon 96 --seed 2023
run --model iTransformer --horizon 24 --seed 2023
run --model iTransformer --horizon 96 --seed 2023
run --model TimeMixer   --horizon 24 --seed 2023
run --model TimeMixer   --horizon 96 --seed 2023

echo "=== CryptoVol training COMPLETE ==="
