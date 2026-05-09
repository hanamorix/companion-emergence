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
  NELL_BIN="$RUNTIME_DIR/Scripts/nell.bat"
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
TMP_SUMS="$(mktemp -t pbs-sums-XXXX)"
TMP_REQ="$(mktemp -t companion-runtime-reqs-XXXX.txt)"
TMP_VERIFY="$(mktemp -d -t companion-runtime-verify-XXXX)"
cleanup() {
  rm -f "$TMP_TAR" "$TMP_SUMS" "$TMP_REQ"
  rm -rf "$TMP_VERIFY"
}
trap cleanup EXIT

echo "[build] downloading ${PBS_ASSET}"
if ! curl -fsSL -o "$TMP_TAR" "$PBS_URL"; then
  echo "[build] FAIL: download ${PBS_URL}" >&2
  exit 1
fi

# Audit 2026-05-07 P3-7: verify the tarball against the
# upstream SHA256SUMS manifest before extraction. Catches transit
# corruption, mirror tampering, accidental upstream replacement.
# Does NOT defend against a coordinated upstream compromise where
# the asset + SHA256SUMS get replaced together — for that level of
# supply-chain rigor, hardcode a known-good SHA256SUMS hash and
# verify the manifest itself. We accept the upstream-trust posture
# here since python-build-standalone is Astral-published.
SUMS_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/SHA256SUMS"
echo "[build] fetching SHA256SUMS for verification"
if ! curl -fsSL -o "$TMP_SUMS" "$SUMS_URL"; then
  echo "[build] FAIL: download ${SUMS_URL}" >&2
  exit 1
fi
EXPECTED_SHA="$(grep "  ${PBS_ASSET}\$" "$TMP_SUMS" | awk '{print $1}' | head -n1)"
if [ -z "$EXPECTED_SHA" ]; then
  echo "[build] FAIL: ${PBS_ASSET} not listed in SHA256SUMS" >&2
  exit 1
fi
# shasum on macOS / typical *nix; sha256sum on Linux / Windows Git Bash.
# Pick whichever the host has so the script runs on every CI runner.
if command -v shasum >/dev/null 2>&1; then
  ACTUAL_SHA="$(shasum -a 256 "$TMP_TAR" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
  ACTUAL_SHA="$(sha256sum "$TMP_TAR" | awk '{print $1}')"
elif command -v certutil >/dev/null 2>&1; then
  # Windows fallback — certutil ships with the OS. Output looks like:
  #   SHA256 hash of ...:
  #   AB CD EF ...   (hex words separated by spaces)
  #   CertUtil: -hashfile command completed successfully.
  ACTUAL_SHA="$(certutil -hashfile "$TMP_TAR" SHA256 \
    | grep -v -i "hash\|certutil" \
    | tr -d ' \r\n' \
    | tr 'A-Z' 'a-z')"
else
  echo "[build] FAIL: no sha256 tool found (shasum / sha256sum / certutil)" >&2
  exit 1
fi
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  echo "[build] FAIL: SHA256 mismatch for ${PBS_ASSET}" >&2
  echo "[build]   expected: $EXPECTED_SHA" >&2
  echo "[build]   actual:   $ACTUAL_SHA" >&2
  exit 1
fi
echo "[build] sha256 verified: ${EXPECTED_SHA:0:16}…"

echo "[build] extracting"
# python-build-standalone tarballs contain a top-level `python/` dir;
# strip it so our runtime root is bin/, lib/, share/ (or python.exe +
# Scripts/ + Lib/ on Windows).
tar -xzf "$TMP_TAR" -C "$RUNTIME_DIR" --strip-components=1

[ -x "$PY_BIN" ] || { echo "[build] FAIL: extracted python missing at $PY_BIN" >&2; exit 1; }

echo "[build] python sanity check"
"$PY_BIN" --version

# 3. Build the brain wheel from the source tree
echo "[build] uv build -> wheel"
cd "$REPO_ROOT"
rm -rf dist/
uv build --wheel >/dev/null
WHEEL="$(ls dist/*.whl | head -n1)"
[ -n "$WHEEL" ] || { echo "[build] FAIL: no wheel produced" >&2; exit 1; }
echo "[build] wheel: $(basename "$WHEEL")"

# 4. Install locked runtime dependencies + the wheel into the bundled runtime.
# The previous pip install resolved broad pyproject ranges at build time, so two
# release builds from the same commit could ship different dependency versions.
echo "[build] exporting locked runtime requirements"
uv export \
  --format requirements.txt \
  --no-dev \
  --no-emit-project \
  --locked \
  --output-file "$TMP_REQ" >/dev/null

echo "[build] installing locked deps into bundled python"
uv pip install --python "$PY_BIN" --require-hashes --requirements "$TMP_REQ" --quiet

echo "[build] installing companion-emergence wheel into bundled python"
uv pip install --python "$PY_BIN" --no-deps --quiet "$WHEEL"

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
  # On Windows ``uv pip install`` produces Scripts/nell.exe as a uv
  # trampoline — a small launcher exe with the *build host's*
  # absolute path to python.exe baked into it. On a user's machine
  # that path doesn't exist, and the trampoline aborts with
  # "uv trampoline failed to canonicalize script path". Replace it
  # with a relocatable nell.bat that resolves python by relative
  # path. Rust's bundled_nell_path() looks for nell.bat first.
  echo "[build] writing relocatable nell.bat wrapper (Windows)"
  NELL_BAT="$RUNTIME_DIR/Scripts/nell.bat"
  cat > "$NELL_BAT" <<'NELL_BAT_WRAPPER'
@echo off
rem Relocatable nell launcher — runs whichever python.exe lives in
rem ../ next to this Scripts dir. Generated by build_python_runtime.sh.
"%~dp0..\python.exe" -c "import sys; from brain.cli import main; sys.exit(main())" %*
NELL_BAT_WRAPPER
  # Remove the broken uv trampoline so nothing's tempted to invoke it.
  rm -f "$NELL_BIN"
fi

# 6a. Verify the brain entry point + import work in the bundled python
echo "[build] verify brain import + entry point"
(
  cd "$TMP_VERIFY"
  "$PY_BIN" -c "
from pathlib import Path
import brain
import brain.cli
brain_file = Path(brain.__file__).resolve()
runtime_root = Path('$RUNTIME_DIR').resolve()
print('  brain:', brain_file)
assert str(brain_file).startswith(str(runtime_root)), brain_file
"
  "$NELL_BIN" --version
  echo "[build] verify nell can resolve brain.voice_templates"
  "$PY_BIN" -c "
from importlib.resources import files
content = files('brain.voice_templates').joinpath('nell-voice.md').read_text(encoding='utf-8')
print('  nell-voice.md:', len(content), 'bytes')
"
)

# 6b. Strip everything we don't need at runtime to shrink the bundle.
# Audit 2026-05-07 P3-9: the previous prune only removed __pycache__
# and site-packages tests; the bundle stayed ~135 MB. This pass also
# removes IDLE/Tk/Tkinter (NellFace doesn't use the stdlib GUI),
# python-config / Makefile / static libs / headers (all dev-only),
# upstream test packages from the stdlib, idle stubs, and pyc files.
echo "[build] strip dev-only and unused stdlib from bundled runtime"
PRUNE_SIZE_BEFORE="$(du -sk "$RUNTIME_DIR" | cut -f1)"

# Universal: pyc caches everywhere
find "$RUNTIME_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$RUNTIME_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true

# Universal: site-packages test directories (third-party dep tests)
find "$SITE_PACKAGES" -type d -name "tests" -prune -exec rm -rf {} + 2>/dev/null || true
find "$SITE_PACKAGES" -type d -name "test" -prune -exec rm -rf {} + 2>/dev/null || true

if [ "$HOST_OS" = "unix" ]; then
  STDLIB_DIR="$RUNTIME_DIR/lib/python3.13"
  # Stdlib test packages — unittest framework stays (used at runtime),
  # but the test corpora are dev-only.
  rm -rf "$STDLIB_DIR/test" 2>/dev/null || true
  # IDLE and turtle demo — interactive stdlib GUI tools, not used.
  rm -rf "$STDLIB_DIR/idlelib" "$STDLIB_DIR/turtledemo" 2>/dev/null || true
  # Tcl/Tk + tkinter — Python's GUI bindings; NellFace UI is React/Tauri.
  rm -rf "$STDLIB_DIR/tkinter" 2>/dev/null || true
  rm -rf "$RUNTIME_DIR/lib/tcl"* "$RUNTIME_DIR/lib/tk"* "$RUNTIME_DIR/lib/itcl"* "$RUNTIME_DIR/lib/thread"* 2>/dev/null || true
  # Dev-only: Python C headers, static libs, Makefiles, pkg-config.
  rm -rf "$RUNTIME_DIR/include" 2>/dev/null || true
  rm -rf "$RUNTIME_DIR/lib/pkgconfig" 2>/dev/null || true
  find "$RUNTIME_DIR/lib" -name "*.a" -delete 2>/dev/null || true
  find "$RUNTIME_DIR/lib" -name "Makefile*" -delete 2>/dev/null || true
  find "$RUNTIME_DIR/lib" -name "config-*" -type d -prune -exec rm -rf {} + 2>/dev/null || true
  # Man pages — interactive doc, not used by the .app.
  rm -rf "$RUNTIME_DIR/share/man" 2>/dev/null || true
  # Runtime should not self-mutate. Remove package managers and interactive/dev
  # helper entry points after the locked install is complete.
  rm -f "$RUNTIME_DIR/bin/pip" "$RUNTIME_DIR/bin/pip3" "$RUNTIME_DIR/bin/pip3.13" 2>/dev/null || true
  rm -f "$RUNTIME_DIR/bin/idle3" "$RUNTIME_DIR/bin/idle3.13" "$RUNTIME_DIR/bin/pydoc3" "$RUNTIME_DIR/bin/pydoc3.13" 2>/dev/null || true
  rm -f "$RUNTIME_DIR/bin/python3-config" "$RUNTIME_DIR/bin/python3.13-config" 2>/dev/null || true
  rm -f "$RUNTIME_DIR/bin/f2py" "$RUNTIME_DIR/bin/numpy-config" 2>/dev/null || true
else
  # Windows layout — stdlib at Lib/, no separate include/lib split for
  # most dev-only artifacts. Same prune set, mapped paths.
  STDLIB_DIR="$RUNTIME_DIR/Lib"
  rm -rf "$STDLIB_DIR/test" 2>/dev/null || true
  rm -rf "$STDLIB_DIR/idlelib" "$STDLIB_DIR/turtledemo" 2>/dev/null || true
  rm -rf "$STDLIB_DIR/tkinter" 2>/dev/null || true
  rm -rf "$RUNTIME_DIR/tcl" 2>/dev/null || true
  rm -rf "$RUNTIME_DIR/Doc" 2>/dev/null || true
  rm -f "$RUNTIME_DIR/Scripts/pip"* "$RUNTIME_DIR/Scripts/idle"* "$RUNTIME_DIR/Scripts/pydoc"* 2>/dev/null || true
  rm -f "$RUNTIME_DIR/Scripts/f2py"* "$RUNTIME_DIR/Scripts/numpy-config"* 2>/dev/null || true
fi

rm -rf "$SITE_PACKAGES"/pip "$SITE_PACKAGES"/pip-*.dist-info 2>/dev/null || true
rm -rf "$SITE_PACKAGES"/setuptools "$SITE_PACKAGES"/setuptools-*.dist-info 2>/dev/null || true
rm -rf "$SITE_PACKAGES"/wheel "$SITE_PACKAGES"/wheel-*.dist-info 2>/dev/null || true

PRUNE_SIZE_AFTER="$(du -sk "$RUNTIME_DIR" | cut -f1)"
PRUNE_SAVED_KB=$((PRUNE_SIZE_BEFORE - PRUNE_SIZE_AFTER))
echo "[build] prune saved: $((PRUNE_SAVED_KB / 1024)) MB (before $((PRUNE_SIZE_BEFORE / 1024)) MB → after $((PRUNE_SIZE_AFTER / 1024)) MB)"

if [ "$HOST_OS" = "unix" ]; then
  echo "[build] runtime binary architecture"
  file "$PY_BIN" || true
fi
"$PY_BIN" -c "import platform, sys; print('  executable:', sys.executable); print('  machine:', platform.machine())"

SIZE_KB="$(du -sk "$RUNTIME_DIR" | cut -f1)"
SIZE="$(du -sh "$RUNTIME_DIR" | cut -f1)"
# Default budget covers all current targets:
#   macOS arm64/x86_64 land at ~125-140 MB after pruning.
#   Linux x86_64 lands at ~390 MB (python-build-standalone ships
#     more shared objects than the macOS tarball).
#   Windows ships an embedded .pyd-heavy tree at ~250 MB.
# 450 MB gives all three headroom while still tripping if a future
# unrelated dep drags hundreds of megabytes of bloat in.
MAX_RUNTIME_MB="${MAX_RUNTIME_MB:-450}"
MAX_RUNTIME_KB=$((MAX_RUNTIME_MB * 1024))
if [ "$SIZE_KB" -gt "$MAX_RUNTIME_KB" ]; then
  echo "[build] FAIL: bundled runtime is $SIZE, budget is ${MAX_RUNTIME_MB}M" >&2
  exit 1
fi
echo "[build] PASS - bundled runtime ready at $RUNTIME_DIR ($SIZE, budget ${MAX_RUNTIME_MB}M)"
