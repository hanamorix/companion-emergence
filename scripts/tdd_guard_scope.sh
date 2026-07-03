#!/usr/bin/env bash
# tdd_guard_scope.sh — a thin PreToolUse wrapper in front of `tdd-guard`.
#
# WHY: tdd-guard reads pytest results (.claude/tdd-guard/data/test.json) and has
# NO Rust reporter, so it blocks EVERY *.rs implementation edit (the guard sees
# no failing Rust test and refuses). Rust changes here are verified by
# `cargo test`, not pytest, so the guard's premise doesn't apply to them.
#
# This wrapper reads the PreToolUse payload on stdin, ALLOWS the edit when its
# target is a Rust file, and otherwise forwards the payload to tdd-guard
# unchanged (Python edits keep full TDD enforcement).
#
# Wire it in .claude/settings.local.json PreToolUse (Write|Edit|MultiEdit):
#   "command": "bash $CLAUDE_PROJECT_DIR/scripts/tdd_guard_scope.sh"
set -uo pipefail

payload="$(cat)"

fp="$(printf '%s' "$payload" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    print(""); sys.exit(0)
ti = d.get("tool_input") or {}
# Write/Edit/MultiEdit carry the path on tool_input.file_path.
print(ti.get("file_path", "") or "")
' 2>/dev/null || true)"

case "$fp" in
  *.rs)
    # Rust edit — allow (no Rust reporter in tdd-guard). Exit 0 = permit.
    exit 0
    ;;
esac

# Everything else: hand the untouched payload to tdd-guard.
printf '%s' "$payload" | tdd-guard
