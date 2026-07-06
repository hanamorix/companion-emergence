#!/usr/bin/env bash
# One-shot dev bootstrap for a fresh clone (public-sync-reroot P1).
#
# Git hooks are NOT cloned — a fresh clone is unguarded on its first push
# until this runs (criterion C9). CONTRIBUTING documents the one-liner;
# this script is the same thing plus dependency setup.

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

echo "[bootstrap] enabling leak-guard hooks (core.hooksPath=hooks)"
git config core.hooksPath hooks

if [ ! -f hooks/leak-rules.local ]; then
  cat <<'EOF'
[bootstrap] NOTE: hooks/leak-rules.local not found (gitignored, maintainer-only).
            The structural arms (identity/path/binary) still run on every push.
            Maintainer: copy your local rules file in for the content arm.
EOF
fi

if command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] uv sync --extra dev"
  uv sync --extra dev
fi
if command -v pnpm >/dev/null 2>&1 && [ -d app ]; then
  echo "[bootstrap] pnpm install (app/)"
  (cd app && pnpm install)
fi

echo "[bootstrap] done"
