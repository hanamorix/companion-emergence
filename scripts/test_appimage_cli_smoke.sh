#!/usr/bin/env bash
# Drives scripts/appimage_cli_smoke.sh against stub "AppImages" (C28b, #408).
#
# The stubs live in a directory whose name contains a space (the real AppImage name does).
# The POSITIVE stub emulates the CLI mode (cds to $COMPANION_NELL_CWD, prints a version, resolves
# KINDLED_HOME=./rel against its cwd, exits 2 for an unknown key). Each NEGATIVE stub breaks exactly
# one check, and the smoke script must fail for it. This executes the script's quoting, comparisons
# and exit logic; it does NOT exercise AppRun (that is the release-time smoke).
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMOKE="$HERE/appimage_cli_smoke.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
STUBS="$WORK/stub dir with spaces"
mkdir -p "$STUBS"

# $1 = file name; $2 = cwd handling (restore|keep); $3 = version line; $4 = unknown-key exit code
make_stub() {
  local file="$STUBS/$1" restore="$2" version="$3" badkey="$4"
  {
    echo '#!/bin/sh'
    echo '[ "$1" = "--companion-nell-cli" ] || exit 64'
    echo 'shift'
    if [ "$restore" = restore ]; then
      echo 'cd "${COMPANION_NELL_CWD:-.}" || exit 65'
    fi
    echo 'case "$1" in'
    echo "  --version) echo '$version'; exit 0 ;;"
    echo '  paths)'
    echo '    shift'
    echo '    key=""; for a in "$@"; do key="$a"; done'
    echo '    case "$key" in'
    echo '      home) case "${KINDLED_HOME:-}" in ./*) echo "$(pwd -P)/${KINDLED_HOME#./}" ;; *) echo "${KINDLED_HOME:-}" ;; esac; exit 0 ;;'
    echo "      *) exit $badkey ;;"
    echo '    esac ;;'
    echo 'esac'
    echo 'exit 64'
  } > "$file"
  chmod +x "$file"
}

make_stub good.AppImage restore "companion-emergence 0.0.42" 2
make_stub no-cwd.AppImage keep "companion-emergence 0.0.42" 2
make_stub bad-version.AppImage restore "something else entirely" 2
make_stub bad-exit.AppImage restore "companion-emergence 0.0.42" 0

fail=0
if ! bash "$SMOKE" "$STUBS/good.AppImage"; then
  echo "FAIL: the smoke script rejected the good stub" >&2
  fail=1
fi
for bad in no-cwd bad-version bad-exit; do
  if bash "$SMOKE" "$STUBS/$bad.AppImage" >/dev/null 2>&1; then
    echo "FAIL: the smoke script accepted the $bad stub" >&2
    fail=1
  fi
done
if [ "$fail" -eq 0 ]; then
  echo "ok: appimage_cli_smoke.sh passes the good stub and fails the three bad ones"
fi
exit "$fail"
