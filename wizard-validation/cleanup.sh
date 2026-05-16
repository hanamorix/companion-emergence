#!/usr/bin/env bash
# Wipe the test NELLBRAIN_HOME so the next launch.sh fires the wizard
# from scratch again.

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOME_DIR="$DIR/nellbrain_home"

# Kill any running NellFace from this validation env first.
pkill -f "$DIR/NellFace.app/Contents/MacOS/nellface" 2>/dev/null || true
sleep 1

echo "[cleanup] removing $HOME_DIR contents"
rm -rf "$HOME_DIR"
mkdir -p "$HOME_DIR"
rm -f "$DIR/launch.log"
echo "[cleanup] done — wizard will fire on next launch.sh"
