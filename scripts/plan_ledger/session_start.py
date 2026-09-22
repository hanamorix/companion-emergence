#!/usr/bin/env python3
"""SessionStart hook: point a fresh, resumed or compacted session at any ACTIVE plan ledger.

The plan-ledger skill's state is the ledger on disk, not the session (METHODOLOGY.md,
"Conventions the skill relies on"). After a compaction the skill text may be gone from context,
so this hook re-injects a one-paragraph reminder naming every ``status: ACTIVE`` ledger under the
project's ledger dir and the cold-start rule (read it in full, quote its Secret token first).

Wired from ``.claude/settings.json`` on the ``startup|resume|compact`` matcher. Read-only.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

LEDGER_DIR = Path("docs/plan-ledger/ledgers")
_ACTIVE = re.compile(r"^status:\s*ACTIVE\b", re.MULTILINE)


def _project_root() -> Path:
    env = os.environ.get("CLAUDE_PROJECT_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent.parent


def _active_ledgers(root: Path) -> list[Path]:
    ledger_dir = root / LEDGER_DIR
    if not ledger_dir.is_dir():
        return []
    found: list[Path] = []
    for path in sorted(ledger_dir.glob("*-LEDGER.md")):
        try:
            head = path.read_text(encoding="utf-8", errors="replace")[:2000]
        except OSError:
            continue
        if _ACTIVE.search(head):
            found.append(path)
    return found


def main() -> int:
    try:
        json.load(
            sys.stdin
        )  # the event payload is not needed; consume it so the pipe closes cleanly
    except (json.JSONDecodeError, OSError):
        pass
    root = _project_root()
    ledgers = _active_ledgers(root)
    if not ledgers:
        return 0
    listed = "\n".join(f"  - {p.relative_to(root).as_posix()}" for p in ledgers)
    context = (
        "plan-ledger: an ACTIVE plan ledger exists for this project. The ledger on disk is the "
        "state; this session is not. Before any planning or spec work, read it IN FULL, quote its "
        "`## Secret:` token in your first message, and continue from `Open forks`. If it is stale "
        "relative to the conversation, reconcile from the transcript by quotation, never from "
        "memory. Config: plan-ledger.companion.md (repo root).\n" + listed
    )
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": context,
                }
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
