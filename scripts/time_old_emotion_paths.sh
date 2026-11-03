#!/usr/bin/env bash
# Safety tier: 1. Reads no persona or user data file other than the path given (it also runs `git rev-parse HEAD`,
# reads /proc on Linux, imports the app's modules, which may read the app's own settings file) and removes its copy.
#
# Times three old emotion code paths on a copy of a synthetic memories.db. See scripts/time_old_emotion_paths.py
# (or run with -h) for what it prints. The path is optional and defaults to test-data/synthetic-db/memories.db.
#
# Usage:
#   bash scripts/time_old_emotion_paths.sh [path to memories.db]
#
# Interpreter order: $PYTHON, then $REPO_ROOT/.venv/bin/python, then `uv run python`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/time_old_emotion_paths.py"

if [ -n "${PYTHON:-}" ]; then
  exec "$PYTHON" "$SCRIPT" "$@"
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  exec "$REPO_ROOT/.venv/bin/python" "$SCRIPT" "$@"
elif command -v uv >/dev/null 2>&1; then
  cd "$REPO_ROOT"
  exec uv run python "$SCRIPT" "$@"
else
  echo "time_old_emotion_paths: no Python found. Set PYTHON=/path/to/python, create $REPO_ROOT/.venv, or install uv." >&2
  exit 1
fi
