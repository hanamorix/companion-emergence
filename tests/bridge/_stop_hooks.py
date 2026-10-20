"""Calls the real-process stop tests run inside the bridge (KINDLED_TEST_CALLS, #456 C52 / C59).

Each takes the persona dir and makes one real model call through the real provider; the `claude` it spawns is the fake
one on PATH (tests/bridge/fake_claude), which blocks until the test releases it.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

MODEL = "claude-haiku-4-5"


def _provider():
    from brain.bridge.provider import ClaudeCliProvider

    return ClaudeCliProvider(model=MODEL, timeout_seconds=600)


def generate_call(persona_dir: Path) -> None:
    out = _provider().generate("hello")
    (Path(persona_dir) / "hook_generate_result.txt").write_text(out, encoding="utf-8")


def compaction_call(persona_dir: Path) -> None:
    from brain.chat.compaction import compact_conversation

    compact_conversation(
        persona_dir, "sess_c", older_than=timedelta(0), fold_existing_summary=True, provider=_provider(),
        min_keep_tail=0,
    )


def pass2_call(persona_dir: Path) -> None:
    from brain.chat import pass2_queue

    pass2_queue.drain_all_locked(persona_dir)


def soul_call(persona_dir: Path) -> None:
    from brain.memory.store import MemoryStore
    from brain.soul.review import review_pending_candidates
    from brain.soul.store import SoulStore

    store = MemoryStore(Path(persona_dir) / "memories.db", integrity_check=False)
    soul = SoulStore(str(Path(persona_dir) / "crystallizations.db"))
    try:
        review_pending_candidates(persona_dir, store=store, soul_store=soul, provider=_provider())
    finally:
        store.close()
        soul.close()
