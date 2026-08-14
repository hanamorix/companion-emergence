#!/usr/bin/env bash
# Wrapper for agent_send.py — auto-locates the repo venv python, runs it as a package module from the
# repo root (so `tests.harness` + `brain` import cleanly), forwards all args.
# Usage: LIVE_ENV=<env.json> ./agent_send.sh [--new] "the human message"
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# repo root = .../companion-emergence ; this wrapper is at tests/harness/
REPO="$(cd "$HERE/../.." && pwd)"
PY="$REPO/.venv/bin/python"
if [ ! -x "$PY" ]; then PY="$(command -v python3 || command -v python)"; fi
cd "$REPO"
exec "$PY" -m tests.harness.agent_send "$@"
