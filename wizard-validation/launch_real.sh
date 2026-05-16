#!/usr/bin/env bash
# Launch NellFace.app against the LIVE nell persona at
# ~/Library/Application Support/companion-emergence/. No
# NELLBRAIN_HOME override so platformdirs picks the default.
# The wizard does NOT fire because app_config.json has
# selected_persona='nell' — the app routes straight to the main UI.

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="$DIR/NellFace.app/Contents/MacOS/nellface"
LOG="$DIR/launch_real.log"

echo "[real] using live persona dir: ~/Library/Application Support/companion-emergence/"
echo "[real] log: $LOG"
"$APP" >"$LOG" 2>&1 &
PID=$!
echo "[real] pid=$PID"
echo "[real] tail logs:  tail -f $LOG"
echo "[real] kill app:   kill $PID"
