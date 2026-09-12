"""#202: voice-reflection evidence carries content, not just ids; the tone placeholder is gone."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.bridge import supervisor
from brain.memory.store import Memory, MemoryStore
from brain.soul.crystallization import Crystallization
from brain.soul.store import SoulStore


def _cryst(id_: str, moment: str, *, days_ago: float) -> Crystallization:
    return Crystallization(
        id=id_,
        moment=moment,
        love_type="storge",
        why_it_matters="it stayed",
        crystallized_at=datetime.now(UTC) - timedelta(days=days_ago),
    )


def test_read_recent_crystallizations_carries_text_and_drops_stale(tmp_path: Path) -> None:
    store = SoulStore(str(tmp_path / "crystallizations.db"))
    try:
        store.create(_cryst("c_new", "the kitchen light left on for me", days_ago=1))
        store.create(_cryst("c_old", "an old thing", days_ago=30))
    finally:
        store.close()

    out = supervisor._read_recent_crystallizations(tmp_path, days=7)

    assert [c["id"] for c in out] == ["c_new"]
    assert "kitchen light" in out[0]["text"]
    assert "it stayed" in out[0]["text"]


def test_read_recent_dreams_reads_dream_memories_with_text(tmp_path: Path) -> None:
    store = MemoryStore(str(tmp_path / "memories.db"), integrity_check=False)
    try:
        fresh = Memory.create_new(
            content="DREAM: the sailor kept the boat", memory_type="dream", domain="self"
        )
        store.create(fresh)
        stale = Memory.create_new(content="DREAM: long ago", memory_type="dream", domain="self")
        stale.created_at = datetime.now(UTC) - timedelta(days=30)
        store.create(stale)
        store.create(Memory.create_new(content="not a dream", memory_type="conversation", domain="us"))
    finally:
        store.close()

    out = supervisor._read_recent_dreams(tmp_path, days=7)

    assert [d["id"] for d in out] == [fresh.id]
    assert "sailor kept the boat" in out[0]["text"]


def test_message_tone_placeholder_is_retired() -> None:
    assert not hasattr(supervisor, "_read_recent_message_tones")
