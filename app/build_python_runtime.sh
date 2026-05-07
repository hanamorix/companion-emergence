#!/usr/bin/env bash
# Phase 7 — build a portable Python runtime for the Tauri bundle.
#
# Downloads python-build-standalone for the host architecture,
# extracts it into app/src-tauri/python-runtime/, builds the
# companion-emergence wheel, and installs the brain into the
# extracted runtime's site-packages. The result is a self-contained
# Python tree that the Tauri bundle ships in Resources/.
#
# Result layout:
#   app/src-tauri/python-runtime/bin/python3
#   app/src-tauri/python-runtime/lib/python3.13/site-packages/brain/...
#
# Re-runnable: deletes the previous python-runtime/ tree first.
#
# Usage:
#   bash app/build_python_runtime.sh
#
# Cross-platform note: this script targets macOS arm64 + x86_64. The
# python-build-standalone manifest URL contains the platform triple
# so other platforms work by changing the asset selector.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$REPO_ROOT/app/src-tauri/python-runtime"
PY_VERSION="3.13.1"
# Astral's manifest tag — pinned for reproducibility. Update by
# checking https://github.com/astral-sh/python-build-standalone/releases
PBS_TAG="20250115"

case "$(uname -s)/$(uname -m)" in
  Darwin/arm64)
    PBS_TARGET="aarch64-apple-darwin"
    ;;
  Darwin/x86_64)
    PBS_TARGET="x86_64-apple-darwin"
    ;;
  Linux/x86_64)
    PBS_TARGET="x86_64-unknown-linux-gnu"
    ;;
  *)
    echo "[build] unsupported platform $(uname -s)/$(uname -m)" >&2
    exit 1
    ;;
esac

PBS_ASSET="cpython-${PY_VERSION}+${PBS_TAG}-${PBS_TARGET}-install_only.tar.gz"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${PBS_ASSET}"

echo "[build] target: ${PBS_TARGET}"
echo "[build] python: ${PY_VERSION}+${PBS_TAG}"
echo "[build] runtime dir: ${RUNTIME_DIR}"

# 1. Clean previous runtime
echo "[build] removing previous runtime tree"
rm -rf "$RUNTIME_DIR"
mkdir -p "$RUNTIME_DIR"

# 2. Fetch + extract python-build-standalone
TMP_TAR="$(mktemp -t pbs-XXXX.tar.gz)"
trap 'rm -f "$TMP_TAR"' EXIT

echo "[build] downloading ${PBS_ASSET}"
if ! curl -fsSL -o "$TMP_TAR" "$PBS_URL"; then
  echo "[build] FAIL: download ${PBS_URL}" >&2
  exit 1
fi
echo "[build] extracting"
# python-build-standalone tarballs contain a top-level `python/` dir;
# strip it so our runtime root is bin/, lib/, share/.
tar -xzf "$TMP_TAR" -C "$RUNTIME_DIR" --strip-components=1

PY_BIN="$RUNTIME_DIR/bin/python3"
[ -x "$PY_BIN" ] || { echo "[build] FAIL: extracted python missing at $PY_BIN" >&2; exit 1; }

echo "[build] python sanity check"
"$PY_BIN" --version

# 3. Build the brain wheel from the source tree
echo "[build] uv build → wheel"
cd "$REPO_ROOT"
rm -rf dist/
uv build --wheel >/dev/null
WHEEL="$(ls dist/*.whl | head -n1)"
[ -n "$WHEEL" ] || { echo "[build] FAIL: no wheel produced" >&2; exit 1; }
echo "[build] wheel: $(basename "$WHEEL")"

# 4. Install the wheel into the bundled runtime
echo "[build] installing brain + deps into bundled python"
"$PY_BIN" -m ensurepip --upgrade >/dev/null
"$PY_BIN" -m pip install --quiet --upgrade pip
"$PY_BIN" -m pip install --quiet "$WHEEL"

# 5. Verify the brain entry point + import work in the bundled python
echo "[build] verify brain import + entry point"
"$PY_BIN" -c "import brain; import brain.cli; print('  brain.__file__:', brain.__file__)"
"$RUNTIME_DIR/bin/nell" --version

# 6. Strip pyc cache + tests dirs to shrink the bundle a bit
echo "[build] strip __pycache__ + tests from site-packages"
find "$RUNTIME_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$RUNTIME_DIR/lib" -type d -name "tests" -prune -exec rm -rf {} +

SIZE="$(du -sh "$RUNTIME_DIR" | cut -f1)"
echo "[build] PASS — bundled runtime ready at $RUNTIME_DIR ($SIZE)"
