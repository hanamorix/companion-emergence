#!/usr/bin/env bash
# Wheel/sdist clean-install smoke test.
#
# Builds the wheel, installs it in a fresh venv (via `uv venv`), and
# runs a series of read-only `nell` invocations to confirm the package
# metadata + entry points are honest.
#
# Usage:
#   bash scripts/smoke_test_wheel.sh
#
# Exits non-zero on the first failure. Tears down the temp venv on
# success or failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
TMP_HOME="$(mktemp -d)"
cleanup() {
  rm -rf "$TMP_DIR" "$TMP_HOME"
}
trap cleanup EXIT

echo "[smoke] repo: $REPO_ROOT"
echo "[smoke] tmp venv dir: $TMP_DIR"
echo "[smoke] tmp NELLBRAIN_HOME: $TMP_HOME"

# Build the wheel + sdist into a temp dist/.
echo "[smoke] uv build → wheel + sdist"
cd "$REPO_ROOT"
rm -rf dist/
uv build --wheel --sdist >/dev/null

WHEEL="$(ls dist/*.whl | head -n1)"
SDIST="$(ls dist/*.tar.gz | head -n1)"
[ -n "$WHEEL" ] || { echo "[smoke] FAIL: no wheel produced"; exit 1; }
[ -n "$SDIST" ] || { echo "[smoke] FAIL: no sdist produced"; exit 1; }
echo "[smoke] wheel: $(basename "$WHEEL") ($(du -h "$WHEEL" | cut -f1))"
echo "[smoke] sdist: $(basename "$SDIST") ($(du -h "$SDIST" | cut -f1))"

# Fresh venv. uv venv + uv pip install bypasses the project's lockfile
# so we're testing what an outside installer would see.
echo "[smoke] uv venv → fresh interpreter"
uv venv "$TMP_DIR/venv" >/dev/null
VENV_PY="$TMP_DIR/venv/bin/python"
VENV_NELL="$TMP_DIR/venv/bin/nell"

echo "[smoke] uv pip install <wheel>"
VIRTUAL_ENV="$TMP_DIR/venv" uv pip install --quiet "$WHEEL"

# The nell entry point should land on PATH inside the venv.
[ -x "$VENV_NELL" ] || { echo "[smoke] FAIL: nell script missing at $VENV_NELL"; exit 1; }

# --version + --help — fast read-only confirmations.
echo "[smoke] nell --version"
"$VENV_NELL" --version
echo "[smoke] nell --help (first 5 lines)"
"$VENV_NELL" --help | head -5

# Ensure the brain package is fully importable from the wheel.
echo "[smoke] brain package import"
"$VENV_PY" -c "
import brain
import brain.cli, brain.bridge.server, brain.images, brain.chat.engine
print('  brain.__file__:', brain.__file__)
"

# Smoke a non-interactive nell init against the tmp NELLBRAIN_HOME so
# we exercise persona creation against the installed code path.
echo "[smoke] nell init smoke (against tmp NELLBRAIN_HOME)"
NELLBRAIN_HOME="$TMP_HOME" "$VENV_NELL" init \
  --persona smoke_persona \
  --user-name Smoke \
  --voice-template default \
  --fresh
[ -f "$TMP_HOME/personas/smoke_persona/persona_config.json" ] || {
  echo "[smoke] FAIL: persona_config.json not created"; exit 1;
}
echo "[smoke] persona_config.json:"
cat "$TMP_HOME/personas/smoke_persona/persona_config.json"

# Read-only status confirms the persona is visible to the installed CLI.
echo "[smoke] nell status"
NELLBRAIN_HOME="$TMP_HOME" "$VENV_NELL" status --persona smoke_persona

echo "[smoke] PASS — wheel installs clean, nell runs, persona init works"
