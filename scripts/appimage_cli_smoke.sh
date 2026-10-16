#!/usr/bin/env bash
# Smoke test of the AppImage's CLI mode (#408). Usage: scripts/appimage_cli_smoke.sh <path-to.AppImage>
#
# The Linux `~/.local/bin/nell` shortcut is a script that runs the AppImage with
# --companion-nell-cli. This script checks what the shortcut relies on, using a real AppImage:
#   1. python starts under AppRun's environment (the hand-off scrubs it): --version prints a
#      version line and exits 0;
#   2. the caller's working directory is restored (AppRun changes it): with COMPANION_NELL_CWD=$W
#      and the process started from ANOTHER directory, a relative KINDLED_HOME resolves under $W;
#   3. the exit status of the bundled nell is passed through: an unknown `paths` key exits non-zero.
# Everything runs in throwaway directories. CI has no FUSE, so extract-and-run mode is used purely
# to run the image; the product never installs a shortcut from such a run.
#
# It is called from the release workflow (a release-time gate) and can be run by hand on any Linux
# machine. scripts/test_appimage_cli_smoke.sh drives it against stub AppImages.
set -u

fail() {
  echo "appimage_cli_smoke: FAIL: $*" >&2
  exit 1
}

APPIMAGE="${1:-}"
[ -n "$APPIMAGE" ] && [ -f "$APPIMAGE" ] || {
  echo "usage: appimage_cli_smoke.sh <path-to.AppImage>" >&2
  exit 2
}
APPIMAGE="$(cd "$(dirname "$APPIMAGE")" && pwd -P)/$(basename "$APPIMAGE")"
chmod +x "$APPIMAGE" 2>/dev/null || true

WORK="$(mktemp -d)"
ELSEWHERE="$(mktemp -d)"
HOME_TMP="$(mktemp -d)"
trap 'rm -rf "$WORK" "$ELSEWHERE" "$HOME_TMP"' EXIT
WORK="$(cd "$WORK" && pwd -P)"
export APPIMAGE_EXTRACT_AND_RUN=1

# 1. version
out="$(cd "$ELSEWHERE" && COMPANION_NELL_CWD="$WORK" KINDLED_HOME="$HOME_TMP/kh" "$APPIMAGE" --companion-nell-cli --version)"
rc=$?
[ "$rc" -eq 0 ] || fail "--version exited $rc (python did not start under the AppImage environment?): $out"
printf '%s\n' "$out" | grep -Eq '^companion-emergence [0-9]' || fail "unexpected --version output: $out"

# 2. working directory restored
out="$(cd "$ELSEWHERE" && COMPANION_NELL_CWD="$WORK" KINDLED_HOME=./rel "$APPIMAGE" --companion-nell-cli paths --persona smoke home)"
rc=$?
[ "$rc" -eq 0 ] || fail "paths exited $rc: $out"
[ "$out" = "$WORK/rel" ] || fail "working directory was not restored: expected '$WORK/rel', got '$out'"

# 3. exit status passed through
(cd "$ELSEWHERE" && COMPANION_NELL_CWD="$WORK" KINDLED_HOME="$HOME_TMP/kh" "$APPIMAGE" --companion-nell-cli paths --persona smoke no_such_key) >/dev/null 2>&1
rc=$?
[ "$rc" -ne 0 ] || fail "an unknown paths key exited 0 (exit status not passed through)"

echo "appimage_cli_smoke: ok"
