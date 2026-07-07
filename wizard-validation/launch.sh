#!/usr/bin/env bash
# Launch the freshly-built "Companion Emergence.app" with an ISOLATED
# KINDLED_HOME so the wizard fires from scratch and your live nell
# persona (~/Library/Application Support/companion-emergence) is never
# touched.
#
# Logs go to ./launch.log so you can inspect what the bridge / brain
# emitted if anything misbehaves.

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$DIR/Companion Emergence.app/Contents/MacOS/nellface"
HOME_DIR="$DIR/nellbrain_home"
LOG="$DIR/launch.log"

mkdir -p "$HOME_DIR"

# Agent/dev shells inject an Anthropic proxy + API overrides that make the
# brain's nested `claude -p` calls 401 (see CLAUDE.md gotchas). Strip them
# so this behaves like a real user launch regardless of which shell runs it.
unset ANTHROPIC_BASE_URL ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_MODEL
export NO_PROXY="*"

echo "[launch] KINDLED_HOME=$HOME_DIR"
echo "[launch] log=$LOG"
echo "[launch] starting Companion Emergence (v$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$DIR/Companion Emergence.app/Contents/Info.plist" 2>/dev/null || echo '?'))…"

# Launch the binary directly (not via `open`) so the env vars are
# inherited by the process. KINDLED_HOME is canonical since v0.0.13;
# NELLBRAIN_HOME kept for belt-and-braces back-compat.
KINDLED_HOME="$HOME_DIR" NELLBRAIN_HOME="$HOME_DIR" "$APP" >"$LOG" 2>&1 &
PID=$!
echo "[launch] pid=$PID"
echo "[launch] tail logs with:  tail -f $LOG"
echo "[launch] kill app with:   kill $PID"
