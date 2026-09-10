#!/usr/bin/env bash
# One-time setup: give the live-test HARNESS (tests/harness) its own `claude`
# login, in a stable dir it can reuse across sandboxed runs. (#236)
#
# WHY: the sandbox points CLAUDE_CONFIG_DIR at an isolated dir so the `claude`
# CLI never reads your ~/.claude. An explicit CLAUDE_CONFIG_DIR keys its OWN
# credential (per dir path) and never falls back to your default login, so a
# fresh tempdir is always "Not logged in" — and on a Mac the credential lives in
# the Keychain, not a file, so nothing can be copied in. A stable dir logged
# into once keeps its Keychain entry; the sandbox uses it whenever the
# `.harness-authed` marker exists, and the live example skips when it doesn't.
#
# Run it ONCE:  bash scripts/setup_harness_claude_login.sh
# Undo it:      rm -f "<harness dir>/.harness-authed"
# Override dir: CE_HARNESS_CLAUDE_CONFIG_DIR=/path bash scripts/setup_harness_claude_login.sh

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve the dir exactly as tests/harness/sandbox.py::_harness_config_dir does.
HARNESS_CFG="$(uv run python -c 'import importlib; print(importlib.import_module("tests.harness.sandbox")._harness_config_dir())')"
MARKER="$HARNESS_CFG/.harness-authed"

echo "Harness claude config dir: $HARNESS_CFG"
mkdir -p "$HARNESS_CFG"

if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' not found on PATH. Install Claude Code and sign in normally first." >&2
  exit 1
fi

echo
echo "A browser sign-in will open. Log in with the SAME Anthropic account you"
echo "use normally — this just gives the test harness its own copy of the login."
echo

CLAUDE_CONFIG_DIR="$HARNESS_CFG" claude auth login

echo
echo "Verifying..."
STATUS_JSON="$(CLAUDE_CONFIG_DIR="$HARNESS_CFG" claude auth status 2>/dev/null || true)"

if printf '%s' "$STATUS_JSON" | grep -q '"loggedIn": *true'; then
  printf 'ok' > "$MARKER"
  echo "✅ Harness login verified. Marker written: $MARKER"
  echo "   Sandboxed live runs (e.g. tests/harness/examples/test_generic_run.py) now authenticate."
else
  rm -f "$MARKER"
  echo "❌ Login did not verify as logged-in. Marker NOT written; sandboxed live runs" >&2
  echo "   will skip with a reason pointing here. Re-run to try again." >&2
  echo "   auth status was: $STATUS_JSON" >&2
  exit 1
fi
