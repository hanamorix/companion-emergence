#!/usr/bin/env python3
"""Read-before-edit gate for plan-ledger files (PreToolUse + PostToolUse hook, one script).

METHODOLOGY.md: "Spec, brief, ledger and handoff files are covered by a read-before-edit hook so
they are never edited from memory after a compaction." Mechanics:

* ``PostToolUse`` on ``Read`` — when the file is a gated one, stamp its path with the current time
  in a small JSON file under ``.claude/hooks/`` (gitignored, per-checkout).
* ``PreToolUse`` on ``Edit`` / ``Write`` / ``MultiEdit`` — when the target is a gated file that
  already exists on disk, deny unless a stamp exists AND is newer than the file's mtime (a read
  that predates the last write does not count). Creating a new gated file is allowed.

Gated basenames: ``*-LEDGER.md``, ``*-spec.md``, ``*-brief.md``, ``DESIGN-INVARIANTS.md`` and the
``plan-ledger.*.md`` config. Everything else passes through untouched. Fails open on any internal
error (a broken hook must never block ordinary editing); a denial is the only non-passthrough
outcome and it states the file and the remedy.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

_GATED = (
    re.compile(r".*-LEDGER\.md$"),
    re.compile(r".*-spec\.md$"),
    re.compile(r".*-brief\.md$"),
    re.compile(r"^DESIGN-INVARIANTS\.md$"),
    re.compile(r"^plan-ledger\..*\.md$"),
)
_EDIT_TOOLS = {"Edit", "Write", "MultiEdit"}
_STAMP_FILE = Path(".claude/hooks/plan-ledger-reads.json")


def _project_root() -> Path:
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent.parent


def _is_gated(path: Path) -> bool:
    name = path.name
    return any(p.match(name) for p in _GATED)


def _stamp_path() -> Path:
    return _project_root() / _STAMP_FILE


def _load_stamps() -> dict[str, float]:
    try:
        return json.loads(_stamp_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_stamps(stamps: dict[str, float]) -> None:
    sp = _stamp_path()
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(stamps, indent=0, sort_keys=True), encoding="utf-8")


def _target(payload: dict) -> Path | None:
    raw = (payload.get("tool_input") or {}).get("file_path")
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = _project_root() / path
    return path.resolve()


def _deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        return 0
    event = payload.get("hook_event_name")
    tool = payload.get("tool_name")
    target = _target(payload)
    if target is None or not _is_gated(target):
        return 0
    key = str(target)

    if event == "PostToolUse" and tool == "Read":
        stamps = _load_stamps()
        stamps[key] = time.time()
        _save_stamps(stamps)
        return 0

    if event == "PreToolUse" and tool in _EDIT_TOOLS:
        if not target.exists():
            return 0  # creating a new ledger / spec is fine
        stamps = _load_stamps()
        read_at = stamps.get(key)
        try:
            mtime = target.stat().st_mtime
        except OSError:
            return 0
        if read_at is None or read_at < mtime:
            _deny(
                f"plan-ledger read gate: {target.name} is a ledger/spec/invariants file and has not "
                "been read since it last changed. Read it in full (Read tool) before editing so it "
                "is never edited from memory after a compaction; then retry the edit."
            )
        return 0
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 — fail open: a hook bug must never block editing
        sys.exit(0)
