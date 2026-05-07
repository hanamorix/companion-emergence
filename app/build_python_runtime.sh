#!/usr/bin/env bash
# Phase 7 — build a portable Python runtime for the Tauri bundle.
#
# Downloads python-build-standalone for the host architecture,
# extracts it into app/src-tauri/python-runtime/, builds the
# companion-emergence wheel, and installs the brain into the
# extracted runtime's site-packages. The result is a self-contained
# Python tree that the Tauri bundle ships in Resources/.
#
# Cross-platform layout — python-build-standalone uses different
# directory shapes per OS:
#
#   macOS / Linux:  python-runtime/bin/python3
#                   python-runtime/bin/nell           ← entry point
#                   python-runtime/lib/python3.13/site-packages/brain/
#
#   Windows:        python-runtime/python.exe
#                   python-runtime/Scripts/nell.exe   ← entry point
#                   python-runtime/Lib/site-packages/brain/
#
# The Rust `nell_command(app)` helper resolves the right entry point
# at runtime via cfg!(windows).
#
# Usage:
#   bash app/build_python_runtime.sh
#
# On Windows, run via Git Bash (ships with Git for Windows) so the
# tar/curl/find toolchain works the same.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$REPO_ROOT/app/src-tauri/python-runtime"
PY_VERSION="3.13.1"
# Astral's manifest tag — pinned for reproducibility. Update by
# checking https://github.com/astral-sh/python-build-standalone/releases
PBS_TAG="20250115"

# Host platform → python-build-standalone target triple + paths.
case "$(uname -s)/$(uname -m)" in
  Darwin/arm64)
    PBS_TARGET="aarch64-apple-darwin"
    HOST_OS="unix"
    ;;
  Darwin/x86_64)
    PBS_TARGET="x86_64-apple-darwin"
    HOST_OS="unix"
    ;;
  Linux/x86_64)
    PBS_TARGET="x86_64-unknown-linux-gnu"
    HOST_OS="unix"
    ;;
  Linux/aarch64 | Linux/arm64)
    PBS_TARGET="aarch64-unknown-linux-gnu"
    HOST_OS="unix"
    ;;
  MINGW*/x86_64 | MSYS*/x86_64 | CYGWIN*/x86_64)
    PBS_TARGET="x86_64-pc-windows-msvc-shared"
    HOST_OS="windows"
    ;;
  *)
    echo "[build] unsupported platform $(uname -s)/$(uname -m)" >&2
    echo "[build] supported: macOS arm64, macOS x86_64, Linux x86_64, Linux arm64, Windows x86_64" >&2
    exit 1
    ;;
esac

PBS_ASSET="cpython-${PY_VERSION}+${PBS_TAG}-${PBS_TARGET}-install_only.tar.gz"
PBS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${PBS_ASSET}"

if [ "$HOST_OS" = "windows" ]; then
  PY_BIN="$RUNTIME_DIR/python.exe"
  NELL_BIN="$RUNTIME_DIR/Scripts/nell.exe"
  SITE_PACKAGES="$RUNTIME_DIR/Lib/site-packages"
else
  PY_BIN="$RUNTIME_DIR/bin/python3"
  NELL_BIN="$RUNTIME_DIR/bin/nell"
  SITE_PACKAGES="$RUNTIME_DIR/lib/python3.13/site-packages"
fi

echo "[build] target: ${PBS_TARGET}"
echo "[build] python: ${PY_VERSION}+${PBS_TAG}"
echo "[build] runtime dir: ${RUNTIME_DIR}"
echo "[build] host OS: ${HOST_OS}"

# 1. Clean previous runtime — preserve .gitkeep placeholder so Tauri's
# bundle.resources glob keeps resolving in clean checkouts where the
# build hasn't run yet.
echo "[build] removing previous runtime tree (keeping .gitkeep)"
mkdir -p "$RUNTIME_DIR"
find "$RUNTIME_DIR" -mindepth 1 -not -name ".gitkeep" -prune -exec rm -rf {} + 2>/dev/null || true

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
# strip it so our runtime root is bin/, lib/, share/ (or python.exe +
# Scripts/ + Lib/ on Windows).
tar -xzf "$TMP_TAR" -C "$RUNTIME_DIR" --strip-components=1

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

# 5. Replace the pip-generated nell entry point with a relocatable
# wrapper. pip bakes the absolute path of *this build's* python into
# the script's shebang, which means the bundled nell.app will try to
# exec the build host's python-runtime/python3 on a USER'S
# machine where that path does not exist (or worse, exists from a
# stale build). The wrapper resolves the bundled python by relative
# path so the runtime tree is fully relocatable.
if [ "$HOST_OS" != "windows" ]; then
  echo "[build] writing relocatable nell wrapper (Unix)"
  cat > "$NELL_BIN" <<'NELL_WRAPPER'
#!/bin/sh
# Relocatable nell launcher — runs whichever python3 lives next door.
# Generated by app/build_python_runtime.sh; do not pip-reinstall over.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/python3" -c '
import sys
from brain.cli import main
sys.exit(main())
' "$@"
NELL_WRAPPER
  chmod +x "$NELL_BIN"
else
  # On Windows pip generates Scripts/nell.exe, a small launcher binary
  # that already resolves python via relative path lookup. No wrapper
  # rewrite needed. (If a future Windows setuptools ever bakes an
  # absolute path, mirror the Unix wrapper using a .bat file.)
  echo "[build] Windows: keeping pip-generated Scripts/nell.exe (relative-path launcher)"
fi

# 6a. Verify the brain entry point + import work in the bundled python
echo "[build] verify brain import + entry point"
"$PY_BIN" -c "import brain; import brain.cli; print('  brain:', brain.__file__)"
"$NELL_BIN" --version
echo "[build] verify nell can resolve brain.voice_templates"
"$PY_BIN" -c "
from importlib.resources import files
content = files('brain.voice_templates').joinpath('nell-voice.md').read_text(encoding='utf-8')
print('  nell-voice.md:', len(content), 'bytes')
"

# 6b. Strip pyc cache + tests dirs to shrink the bundle a bit
echo "[build] strip __pycache__ + tests from site-packages"
find "$RUNTIME_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$SITE_PACKAGES" -type d -name "tests" -prune -exec rm -rf {} + 2>/dev/null || true

SIZE="$(du -sh "$RUNTIME_DIR" | cut -f1)"
echo "[build] PASS — bundled runtime ready at $RUNTIME_DIR ($SIZE)"
