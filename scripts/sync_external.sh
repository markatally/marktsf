#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="$ROOT_DIR/external"
MODE="${1:---pinned}"

repos=(
  "TSLib|https://github.com/thuml/Time-Series-Library.git|main|4e938a1"
  "ShiftingTime|https://github.com/srinathdama/ShiftingTime.git|main|3f9be2a"
)

usage() {
  cat <<'USAGE'
Usage: scripts/sync_external.sh [--pinned|--latest]

Clone or update ignored external research repositories.

  --pinned   checkout the recorded commits for reproducibility (default)
  --latest   fast-forward each repository to its recorded branch
USAGE
}

if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$MODE" != "--pinned" && "$MODE" != "--latest" ]]; then
  usage >&2
  exit 2
fi

mkdir -p "$EXTERNAL_DIR"

for repo in "${repos[@]}"; do
  IFS="|" read -r name url branch commit <<<"$repo"
  path="$EXTERNAL_DIR/$name"

  if [[ ! -d "$path/.git" ]]; then
    echo "==> cloning $name"
    git clone "$url" "$path"
  else
    echo "==> updating $name"
  fi

  git -C "$path" fetch --prune origin

  if [[ "$MODE" == "--latest" ]]; then
    git -C "$path" checkout "$branch"
    git -C "$path" pull --ff-only origin "$branch"
  else
    git -C "$path" -c advice.detachedHead=false checkout "$commit"
  fi

  printf '    %s %s\n' "$name" "$(git -C "$path" rev-parse --short HEAD)"
done
