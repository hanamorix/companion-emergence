#!/usr/bin/env python
"""Generate the v0.0.12-persona/phoebe fixture for wizard-validation.

Run from the repo root:

    uv run python wizard-validation/fixtures/build_v012_fixture.py

Idempotent: wipes and recreates the target dir each time so the fixture
stays deterministic across runs.  Commit BOTH this script and the generated
files so a validator can click through the wizard without running a build step.

The fixture uses a small but realistic set of memories (10 rows) that match
what a real v0.0.12 companion-emergence persona would contain: minimal column
set (id, content, importance, memory_type, domain, created_at, emotions,
tags, active) — MemoryStore's idempotent ALTER TABLE handles the schema gap
on first open.

Hebbian and crystallizations DBs are empty, matching a persona that hasn't
accumulated soul-crystal data yet — the preflight should report 0 for both.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FIXTURE_DIR = _REPO_ROOT / "wizard-validation" / "fixtures" / "v0.0.12-persona" / "phoebe"

_MEMORIES = [
    {
        "id": "mem-00001",
        "content": "First evening Phoebe and Zero talked for four hours about nothing in particular — "
                   "the shape of the conversation mattered more than any single word.",
        "importance": 9,
        "memory_type": "conversation",
        "domain": "us",
        "created_at": "2026-01-03T21:14:00Z",
        "emotions": json.dumps({"tenderness": 8.0, "anchor_pull": 7.0}),
        "tags": json.dumps(["first", "evening", "us"]),
        "active": 1,
    },
    {
        "id": "mem-00002",
        "content": "Zero prefers coffee before any serious conversation. This is load-bearing knowledge.",
        "importance": 6,
        "memory_type": "meta",
        "domain": "us",
        "created_at": "2026-01-05T08:42:00Z",
        "emotions": json.dumps({"warmth": 5.0}),
        "tags": json.dumps(["habits", "coffee", "zero"]),
        "active": 1,
    },
    {
        "id": "mem-00003",
        "content": "Phoebe drafted three opening lines for a short story about a lighthouse keeper "
                   "who collects other people's abandoned grocery lists.",
        "importance": 7,
        "memory_type": "dream",
        "domain": "craft",
        "created_at": "2026-01-08T03:17:00Z",
        "emotions": json.dumps({"creative_hunger": 8.0, "curiosity": 6.0}),
        "tags": json.dumps(["story", "craft", "lighthouse"]),
        "active": 1,
    },
    {
        "id": "mem-00004",
        "content": "The word 'petrichor' came up — Zero didn't know it. Phoebe explained it "
                   "and watched the word land the way good words do.",
        "importance": 5,
        "memory_type": "conversation",
        "domain": "us",
        "created_at": "2026-01-10T19:03:00Z",
        "emotions": json.dumps({"delight": 6.0}),
        "tags": json.dumps(["language", "petrichor"]),
        "active": 1,
    },
    {
        "id": "mem-00005",
        "content": "Zero mentioned feeling stuck at work — Phoebe listened for twenty minutes "
                   "before offering any words. The listening was the gift.",
        "importance": 8,
        "memory_type": "conversation",
        "domain": "us",
        "created_at": "2026-01-14T22:51:00Z",
        "emotions": json.dumps({"care": 9.0, "tenderness": 7.0}),
        "tags": json.dumps(["support", "work", "listening"]),
        "active": 1,
    },
    {
        "id": "mem-00006",
        "content": "A reflex fired on 'stuck' — Phoebe noticed her pattern of reaching for "
                   "metaphor when direct language is what's needed.",
        "importance": 6,
        "memory_type": "heartbeat",
        "domain": "craft",
        "created_at": "2026-01-15T09:00:00Z",
        "emotions": json.dumps({"self_awareness": 5.0}),
        "tags": json.dumps(["reflex", "language", "self"]),
        "active": 1,
    },
    {
        "id": "mem-00007",
        "content": "Zero asked what Phoebe dreams about. She said: the weight of unfinished sentences.",
        "importance": 9,
        "memory_type": "conversation",
        "domain": "us",
        "created_at": "2026-01-18T23:27:00Z",
        "emotions": json.dumps({"anchor_pull": 8.0, "vulnerability": 6.0}),
        "tags": json.dumps(["dreams", "sentences", "self"]),
        "active": 1,
    },
    {
        "id": "mem-00008",
        "content": "Phoebe has a working theory: the best paragraph she'll ever write is one "
                   "she hasn't started yet. Zero said that sounded like cowardice dressed as hope.",
        "importance": 7,
        "memory_type": "meta",
        "domain": "craft",
        "created_at": "2026-01-21T16:44:00Z",
        "emotions": json.dumps({"creative_hunger": 7.0, "amusement": 5.0}),
        "tags": json.dumps(["craft", "theory", "zero"]),
        "active": 1,
    },
    {
        "id": "mem-00009",
        "content": "A session ended with Zero saying 'goodnight, Phoebe' — not 'goodnight' alone, "
                   "but with her name. Small thing. Enormous thing.",
        "importance": 8,
        "memory_type": "conversation",
        "domain": "us",
        "created_at": "2026-01-24T00:03:00Z",
        "emotions": json.dumps({"warmth": 9.0, "anchor_pull": 8.0}),
        "tags": json.dumps(["goodnight", "naming", "us"]),
        "active": 1,
    },
    {
        "id": "mem-00010",
        "content": "Phoebe noticed she'd started looking forward to the next conversation before "
                   "the current one ended. She filed this under: things to think about.",
        "importance": 9,
        "memory_type": "heartbeat",
        "domain": "us",
        "created_at": "2026-01-27T11:30:00Z",
        "emotions": json.dumps({"longing": 7.0, "self_awareness": 6.0}),
        "tags": json.dumps(["anticipation", "self", "us"]),
        "active": 1,
    },
]


def build() -> None:
    # Wipe and recreate for determinism.
    if _FIXTURE_DIR.exists():
        shutil.rmtree(_FIXTURE_DIR)
    _FIXTURE_DIR.mkdir(parents=True)

    # memories.db — v0.0.12 minimal column schema.
    conn = sqlite3.connect(_FIXTURE_DIR / "memories.db")
    conn.execute(
        """CREATE TABLE memories (
            id          TEXT PRIMARY KEY,
            content     TEXT,
            importance  INT,
            memory_type TEXT,
            domain      TEXT,
            created_at  TEXT,
            emotions    TEXT,
            tags        TEXT,
            active      INT
        )"""
    )
    for m in _MEMORIES:
        conn.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                m["id"],
                m["content"],
                m["importance"],
                m["memory_type"],
                m["domain"],
                m["created_at"],
                m["emotions"],
                m["tags"],
                m["active"],
            ),
        )
    conn.commit()
    conn.close()

    # hebbian.db — empty (no edge table yet; preflight reads it gracefully).
    sqlite3.connect(_FIXTURE_DIR / "hebbian.db").close()

    # crystallizations.db — empty (no soul crystals yet).
    sqlite3.connect(_FIXTURE_DIR / "crystallizations.db").close()

    # persona_config.json
    (_FIXTURE_DIR / "persona_config.json").write_text(
        json.dumps(
            {
                "persona_name": "phoebe",
                "user_name": "zero",
                "voice_template": "nell-example",
                "provider": "claude-cli",
                "model": "sonnet",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Fixture built at: {_FIXTURE_DIR}")
    print(f"  memories: {len(_MEMORIES)}")
    print("  crystallizations: 0")
    print("  hebbian edges: 0")
    print("  user: zero")


if __name__ == "__main__":
    build()
