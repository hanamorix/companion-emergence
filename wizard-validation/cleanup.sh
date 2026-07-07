#!/usr/bin/env bash
# Wipe the test KINDLED_HOME so the next launch.sh fires the wizard
# from scratch again — and undo the two side effects a full wizard run
# leaves on the real machine:
#   1. a per-persona launchd agent (com.companion-emergence.supervisor.<persona>)
#      pointing at this rig's temp home
#   2. ~/.local/bin/nell retargeted at this rig's bundled runtime
# Your live nell persona + its launchd agent are never touched (labels
# are per-persona; we only remove agents for personas found in the rig's
# temp home).

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOME_DIR="$DIR/nellbrain_home"

# Kill any running smoke-test app from this validation env first.
pkill -f "$DIR/Companion Emergence.app/Contents/MacOS/nellface" 2>/dev/null || true
pkill -f "$DIR/NellFace.app/Contents/MacOS/nellface" 2>/dev/null || true
sleep 1

# 1. Remove launchd agents for personas created in the rig's temp home.
if [ -d "$HOME_DIR/personas" ]; then
  for p in "$HOME_DIR/personas"/*/; do
    [ -d "$p" ] || continue
    persona="$(basename "$p")"
    label="com.companion-emergence.supervisor.$persona"
    plist="$HOME/Library/LaunchAgents/$label.plist"
    # APFS is case-insensitive: "$label.plist" for a rig persona named
    # "Nell" resolves to the REAL nell.plist. Only remove a plist whose
    # contents actually point at THIS rig's temp home — never a live
    # agent that belongs to a real install. (Learned the hard way.)
    if [ -f "$plist" ] && grep -q "$HOME_DIR" "$plist"; then
      echo "[cleanup] removing launchd agent $label"
      launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
      rm -f "$plist"
    elif [ -f "$plist" ]; then
      echo "[cleanup] SKIP $label — plist does not reference this rig's home"
    fi
  done
fi

# 2. Restore ~/.local/bin/nell if the wizard retargeted it at this rig.
if [ -f "$DIR/nell-symlink-target.txt" ] && [ -L "$HOME/.local/bin/nell" ]; then
  saved="$(cat "$DIR/nell-symlink-target.txt")"
  current="$(readlink "$HOME/.local/bin/nell")"
  if [ "$current" != "$saved" ]; then
    echo "[cleanup] restoring ~/.local/bin/nell -> $saved"
    ln -sf "$saved" "$HOME/.local/bin/nell"
  fi
fi

echo "[cleanup] removing $HOME_DIR contents"
rm -rf "$HOME_DIR"
mkdir -p "$HOME_DIR"
rm -f "$DIR/launch.log"
echo "[cleanup] done — wizard will fire on next launch.sh"
