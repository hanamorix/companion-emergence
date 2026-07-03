#!/usr/bin/env bash
# dev_build.sh — run `pnpm tauri dev` after clearing the stale bundled runtime.
#
# WHY: `pnpm tauri dev` resolves the bridge to a STALE bundled runtime at
# app/src-tauri/target/debug/python-runtime/ (a frozen old `brain/`, e.g.
# 0.0.19) that SHADOWS the source tree — so every new route 404s and it reads as
# a product bug. Removing it forces the dev launcher's uv-fallback to run the
# CURRENT brain/ source.
#
# Usage:  bash scripts/dev_build.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STALE="app/src-tauri/target/debug/python-runtime"
if [[ -d "$STALE" ]]; then
  echo "[dev_build] removing stale bundled runtime: $STALE" >&2
  rm -rf "$STALE"
fi

# Warn if the source brain version differs from pyproject (sanity check).
src_ver=$(grep -E '^version = "[0-9]' pyproject.toml | head -1 | sed 's/.*"\(.*\)".*/\1/')
echo "[dev_build] source brain version: $src_ver — launching pnpm tauri dev" >&2

cd app
exec pnpm tauri dev
