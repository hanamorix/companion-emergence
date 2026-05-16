#!/usr/bin/env bash
# Launch NellFace.app with a FRESH NELLBRAIN_HOME so the wizard fires
# from scratch instead of reading your existing live persona dir.
#
# Logs go to ./launch.log so you can inspect what the bridge / brain
# emitted if anything misbehaves.

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$DIR/NellFace.app/Contents/MacOS/nellface"
HOME_DIR="$DIR/nellbrain_home"
LOG="$DIR/launch.log"

echo "[launch] NELLBRAIN_HOME=$HOME_DIR"
echo "[launch] log=$LOG"
echo "[launch] starting NellFace…"

# We launch the binary directly (not via `open`) so NELLBRAIN_HOME is
# inherited by the process. `open` doesn't pass env vars by default.
NELLBRAIN_HOME="$HOME_DIR" "$APP" >"$LOG" 2>&1 &
PID=$!
echo "[launch] pid=$PID"
echo "[launch] tail logs with:  tail -f $LOG"
echo "[launch] kill app with:   kill $PID"
