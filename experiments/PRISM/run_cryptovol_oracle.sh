#!/usr/bin/env bash
# Run oracle_drift + online_learning for CryptoVol H24 and H96, all 3 seeds.
# Usage: bash experiments/PRISM/run_cryptovol_oracle.sh
set -euo pipefail
cd /Users/mark/Git/hub/mark-tsf

PYTHON=/opt/homebrew/Caskroom/miniforge/base/bin/python
RESULTS=external/TSLib/results
OUT=experiments/PRISM/oracle_drift

MODELS=(DLinear PatchTST iTransformer TimeMixer)

for H in 24 96; do
  for SEED in 2021 2022 2023; do
    OUTDIR="${OUT}/CryptoVol_L96_H${H}_seed${SEED}_target_last"
    echo "=== CryptoVol H${H} seed${SEED} ==="
    $PYTHON -m experiments.PRISM.oracle_drift \
      --results-root $RESULTS \
      --output-dir "$OUTDIR" \
      --dataset CryptoVol --lookback 96 --horizon $H \
      --models "${MODELS[@]}" \
      --target-channel -1 \
      --include-anchors \
      --seed $SEED \
      2>&1 | tail -5

    # Run online learning on this study's window_losses.csv
    if [ -f "${OUTDIR}/window_losses.csv" ]; then
      $PYTHON -m experiments.PRISM.online_learning \
        --losses-csv "${OUTDIR}/window_losses.csv" \
        --output-dir "$OUTDIR" \
        2>&1 | python3 -c "
import sys, json
data = sys.stdin.read()
for line in data.splitlines():
    if line.strip().startswith('{'):
        try:
            s = json.loads(line)
            bfs = s.get('best_fixed_share', {})
            print(f'  FS gap_rec={bfs.get(\"gap_recovered_frac\",\"?\"):.3f}  lr={bfs.get(\"lr\",\"?\")}  alpha={bfs.get(\"alpha\",\"?\")}')
        except: pass
"
    fi
    echo ""
  done
done
echo "=== CryptoVol oracle COMPLETE ==="
