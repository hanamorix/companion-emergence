#!/usr/bin/env bash
# Safety tier: 1. Reads no persona or user data file other than the path given (it also runs `git rev-parse HEAD`,
# reads /proc on Linux, imports the app's modules, which may read the app's own settings file) and removes its copy.
#
# Counts how often each reader condition fires, old fixed level against the engine's `feels_high`, at several floors, on a
# copy of a synthetic memories.db. See scripts/measure_feels_high_thresholds.py (or run with -h) for what it prints.
#
# Usage:
#   bash scripts/measure_feels_high_thresholds.sh <path to memories.db> [--floors 0,0.1,0.25,0.5,1.0,2.0]
#
# Interpreter order: $PYTHON, then $REPO_ROOT/.venv/bin/python, then `uv run python`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/measure_feels_high_thresholds.py"

if [ -n "${PYTHON:-}" ]; then
  exec "$PYTHON" "$SCRIPT" "$@"
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  exec "$REPO_ROOT/.venv/bin/python" "$SCRIPT" "$@"
elif command -v uv >/dev/null 2>&1; then
  exec uv run --project "$REPO_ROOT" python "$SCRIPT" "$@"
else
  echo "measure_feels_high_thresholds: no Python found. Set PYTHON=/path/to/python, create $REPO_ROOT/.venv, or install uv." >&2
  exit 1
fi
