#!/usr/bin/env bash
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

PYTHON=/opt/homebrew/Caskroom/miniforge/base/bin/python

run() {
  echo "=== $* ==="
  $PYTHON -m experiments.PRISM.produce_predictions \
    --dataset-tag CryptoMISO \
    --lookback 96 --patience 5 --epochs 20 \
    "$@"
}

echo "--- FreTS H48/96/168 ---"
run --model FreTS --horizon 48 --seed 2021
run --model FreTS --horizon 96 --seed 2021
run --model FreTS --horizon 168 --seed 2021

echo "--- TimeXer H48/96/168 ---"
run --model TimeXer --horizon 48 --seed 2021
run --model TimeXer --horizon 96 --seed 2021
run --model TimeXer --horizon 168 --seed 2021

echo "=== CryptoMISO pool COMPLETE ==="
