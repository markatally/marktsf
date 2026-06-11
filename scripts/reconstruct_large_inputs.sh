#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

reconstruct() {
  local out_file="$1"
  local expected_size="$2"
  local chunk_dir
  chunk_dir="$(dirname "$out_file")/chunks"

  if [[ ! -d "$chunk_dir" ]]; then
    echo "missing chunk directory: $chunk_dir" >&2
    return 1
  fi

  shopt -s nullglob
  local parts=("$chunk_dir"/train.csv.part-*)
  shopt -u nullglob

  if [[ "${#parts[@]}" -eq 0 ]]; then
    echo "no chunks found in $chunk_dir" >&2
    return 1
  fi

  cat "${parts[@]}" > "$out_file"

  local actual_size
  actual_size="$(wc -c < "$out_file" | tr -d ' ')"
  if [[ "$actual_size" != "$expected_size" ]]; then
    echo "size mismatch for $out_file: expected $expected_size, got $actual_size" >&2
    return 1
  fi

  echo "reconstructed $out_file ($actual_size bytes)"
}

reconstruct "$ROOT_DIR/input/Favorita/train.csv" 4997452288
reconstruct "$ROOT_DIR/input/G-Research-Crypto/train.csv" 2819286393
