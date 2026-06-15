"""Training entry for GreenCast experiments.

This module defines the required CLI contract, device policy, and manifest output
for the publication-grade pre-registered experiment pipeline.

The script intentionally keeps a deterministic placeholder training loop: it prepares
all required run artifacts so that the experiment pipeline and review stack are
end-to-end testable before a full model implementation is injected.
"""

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import torch


def _as_list(v: str) -> List[int]:
    if not v:
        return []
    return [int(x.strip()) for x in v.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--dataset', type=str, required=True)
    p.add_argument('--run_id', type=str, required=True)
    p.add_argument('--stage', type=str, default='M0', choices=['M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6'])
    p.add_argument('--seeds', type=str, default='2021,2022,2023,2024,2025')
    p.add_argument('--horizons', type=str, default='24,48,96,192')
    p.add_argument('--domain', type=str, default='general')
    p.add_argument('--results-root', type=str, default='experiments/GreenCast/results')
    p.add_argument('--dataloader-num-workers', type=int, default=4)
    p.add_argument('--pin-memory', action='store_true', help='deprecated: pin-memory is auto-selected')
    p.add_argument('--status', type=str, default='init', choices=['init', 'placeholder', 'failed', 'done'])
    p.add_argument('--notes', type=str, default='')
    return p.parse_args()


def select_device() -> torch.device:
    # Required by AGENTS.md and proposal constraints.
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding='utf-8')


def write_manifest(run_dir: Path, args: argparse.Namespace, extra: Dict[str, Any]) -> None:
    manifest = {
        'run_id': args.run_id,
        'dataset': args.dataset,
        'domain': args.domain,
        'experiment_stage': args.stage,
        'config': args.config,
        'seeds': _as_list(args.seeds),
        'horizons': _as_list(args.horizons),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(),
        'dataloader_num_workers': args.dataloader_num_workers,
    }
    manifest.update(extra)
    write_text(run_dir / 'manifest.json', json.dumps(manifest, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()

    device = select_device()
    pin_memory = device.type == 'cuda'  # only CUDA supports pinned memory
    if args.pin_memory and not pin_memory:
        print('warn: --pin-memory was requested but cannot be enabled outside CUDA; auto-corrected to False')

    run_dir = Path(args.results_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Keep a manifest-first audit trail even before model code is filled.
    extra = {
        'device': str(device),
        'pin_memory': pin_memory,
        'model': 'GreenCast',
        'backbone': 'StableGreenBank',
        'status': args.status,
        'notes': args.notes,
        'train_version': 'v1.4',
        'residual_ratio_cap': 0.25,
        'command': ' '.join(__import__('sys').argv),
    }

    write_manifest(run_dir, args, extra)

    # Save a frozen config snapshot for reproducibility trace.
    write_text(run_dir / 'config_snapshot.yaml', Path(args.config).read_text(encoding='utf-8') if Path(args.config).exists() else '')
    write_text(run_dir / 'train.log', f"run_id={args.run_id}\ndataset={args.dataset}\nstage={args.stage}\nseeds={args.seeds}\nhorizons={args.horizons}\ndevice={device}\npin_memory={pin_memory}\n")

    # Placeholder outputs for compatibility.
    for seed in _as_list(args.seeds):
        (run_dir / f'predictions_{seed}.parquet').write_bytes(b'')

    (run_dir / 'failures.json').write_text('[]', encoding='utf-8')

    print('train placeholder', vars(args))
    print(f'device={device}, pin_memory={pin_memory}')


if __name__ == '__main__':
    main()
