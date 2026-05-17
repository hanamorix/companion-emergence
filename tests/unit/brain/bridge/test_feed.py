"""Tests for brain.bridge.feed — visible inner life journal builder."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.feed import (
    TYPE_OPENER,
    FeedEntry,
    build_feed,
)


def test_type_opener_covers_all_five_types() -> None:
    """Every FeedEntry.type must have an opener phrase mapped."""
    expected = {"dream", "research", "soul", "outreach", "voice_edit"}
    assert set(TYPE_OPENER.keys()) == expected
    for key, opener in TYPE_OPENER.items():
        assert isinstance(opener, str) and opener, f"opener for {key!r} must be a non-empty string"


def test_type_opener_exact_phrases() -> None:
    """Spec-locked phrasing — v0.0.13-alpha.2 contract."""
    assert TYPE_OPENER["dream"] == "I dreamed"
    assert TYPE_OPENER["research"] == "I've been researching"
    assert TYPE_OPENER["soul"] == "I noticed"
    assert TYPE_OPENER["outreach"] == "I reached out"
    assert TYPE_OPENER["voice_edit"] == "I wanted to change"


def test_feed_entry_serializes_to_dict() -> None:
    entry = FeedEntry(
        type="dream",
        ts="2026-05-17T01:23:45+00:00",
        opener="I dreamed",
        body="about a lighthouse",
        audit_id=None,
    )
    d = entry.to_dict()
    assert d == {
        "type": "dream",
        "ts": "2026-05-17T01:23:45+00:00",
        "opener": "I dreamed",
        "body": "about a lighthouse",
        "audit_id": None,
    }


def test_feed_entry_with_audit_id_serializes() -> None:
    entry = FeedEntry(
        type="outreach",
        ts="2026-05-17T01:00:00+00:00",
        opener="I reached out",
        body="about the lighthouse dream",
        audit_id="ia_2026-05-17T01-00-00_abcd",
    )
    assert entry.to_dict()["audit_id"] == "ia_2026-05-17T01-00-00_abcd"


# ---------------------------------------------------------------------------
# 1B: dream source builder
# ---------------------------------------------------------------------------


def _make_memory_store(persona_dir: Path):
    """Helper — open a MemoryStore at persona_dir/memories.db."""
    from brain.memory.store import MemoryStore

    return MemoryStore(persona_dir / "memories.db")


def test_build_dream_entries_reads_memories(tmp_path):
    """Dream feed entries come from MemoryStore type='dream'."""
    from brain.bridge.feed import build_dream_entries
    from brain.memory.store import Memory

    store = _make_memory_store(tmp_path)
    try:
        # Insert two dreams + one non-dream (should be filtered out by type)
        store.create(
            Memory(
                id="dream_a",
                memory_type="dream",
                content="I'm in a hallway, the lights are dim.",
                domain="dream",
                emotions={"longing": 6.0},
                tags=[],
                importance=0.5,
                score=0.5,
                created_at=datetime(2026, 5, 17, 1, 0, tzinfo=UTC),
                active=True,
            )
        )
        store.create(
            Memory(
                id="dream_b",
                memory_type="dream",
                content="A lighthouse, the lantern won't catch.",
                domain="dream",
                emotions={"sorrow": 5.0},
                tags=[],
                importance=0.5,
                score=0.5,
                created_at=datetime(2026, 5, 17, 2, 0, tzinfo=UTC),
                active=True,
            )
        )
        store.create(
            Memory(
                id="other",
                memory_type="research",
                content="non-dream — must not appear in dream feed.",
                domain="research",
                emotions={},
                tags=[],
                importance=0.5,
                score=0.5,
                created_at=datetime(2026, 5, 17, 3, 0, tzinfo=UTC),
                active=True,
            )
        )
    finally:
        store.close()

    entries = build_dream_entries(tmp_path, limit=10)
    assert len(entries) == 2
    # Newest first
    assert entries[0].type == "dream"
    assert entries[0].opener == "I dreamed"
    assert entries[0].body == "A lighthouse, the lantern won't catch."
    assert entries[0].ts == "2026-05-17T02:00:00+00:00"
    assert entries[0].audit_id is None
    assert entries[1].body == "I'm in a hallway, the lights are dim."


def test_build_dream_entries_empty_when_no_memories(tmp_path):
    """No dreams in store → empty list, no errors."""
    from brain.bridge.feed import build_dream_entries

    # Don't even create the store — function must tolerate it.
    entries = build_dream_entries(tmp_path, limit=10)
    assert entries == []


def test_build_dream_entries_respects_limit(tmp_path):
    """Limit caps the number of returned entries."""
    from brain.bridge.feed import build_dream_entries
    from brain.memory.store import Memory

    store = _make_memory_store(tmp_path)
    try:
        for i in range(5):
            store.create(
                Memory(
                    id=f"d_{i}",
                    memory_type="dream",
                    content=f"dream {i}",
                    domain="dream",
                    emotions={},
                    tags=[],
                    importance=0.5,
                    score=0.5,
                    created_at=datetime(2026, 5, 17, i + 1, tzinfo=UTC),
                    active=True,
                )
            )
    finally:
        store.close()

    entries = build_dream_entries(tmp_path, limit=2)
    assert len(entries) == 2


# ---------------------------------------------------------------------------
# 1C: research source builder
# ---------------------------------------------------------------------------


def test_build_research_entries_reads_memories(tmp_path):
    """Research feed entries come from MemoryStore type='research'."""
    from brain.bridge.feed import build_research_entries
    from brain.memory.store import Memory

    store = _make_memory_store(tmp_path)
    try:
        store.create(
            Memory(
                id="r_a",
                memory_type="research",
                content="History of seasonal grief — the Victorians had a vocabulary.",
                domain="research",
                emotions={"curiosity": 7.0},
                tags=[],
                importance=0.5,
                score=0.5,
                created_at=datetime(2026, 5, 17, 1, 0, tzinfo=UTC),
                active=True,
            )
        )
    finally:
        store.close()

    entries = build_research_entries(tmp_path, limit=10)
    assert len(entries) == 1
    assert entries[0].type == "research"
    assert entries[0].opener == "I've been researching"
    assert entries[0].body.startswith("History of seasonal grief")
    assert entries[0].audit_id is None


def test_build_research_entries_empty(tmp_path):
    from brain.bridge.feed import build_research_entries

    assert build_research_entries(tmp_path, limit=10) == []


# ---------------------------------------------------------------------------
# 1D: soul crystallization source builder
# ---------------------------------------------------------------------------


def _write_soul_audit(persona_dir: Path, entries: list[dict]) -> None:
    """Helper — write one JSON-per-line entry into soul_audit.jsonl."""
    path = persona_dir / "soul_audit.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def test_build_soul_entries_keeps_only_crystallizations(tmp_path):
    """Only soul_audit rows with crystallization_id surface in the feed."""
    from brain.bridge.feed import build_soul_entries

    _write_soul_audit(
        tmp_path,
        [
            # Defer decision — not a crystallization, skip
            {
                "ts": "2026-05-17T01:00:00+00:00",
                "candidate_id": "c_1",
                "candidate_text": "deferred candidate",
                "decision": "defer",
                "crystallization_id": None,
            },
            # Crystallization — keep
            {
                "ts": "2026-05-17T02:00:00+00:00",
                "candidate_id": "c_2",
                "candidate_text": "I keep returning to images of dim light.",
                "decision": "crystallize",
                "crystallization_id": "cr_xyz",
            },
            # Reject decision — not a crystallization, skip
            {
                "ts": "2026-05-17T03:00:00+00:00",
                "candidate_id": "c_3",
                "candidate_text": "rejected",
                "decision": "reject",
                "crystallization_id": None,
            },
        ],
    )

    entries = build_soul_entries(tmp_path, limit=10)
    assert len(entries) == 1
    assert entries[0].type == "soul"
    assert entries[0].opener == "I noticed"
    assert entries[0].body == "I keep returning to images of dim light."
    assert entries[0].ts == "2026-05-17T02:00:00+00:00"
    assert entries[0].audit_id is None


def test_build_soul_entries_empty_when_no_file(tmp_path):
    from brain.bridge.feed import build_soul_entries

    assert build_soul_entries(tmp_path, limit=10) == []


def test_build_soul_entries_skips_malformed_lines(tmp_path):
    """Corrupt JSONL lines are skipped, good ones still surface."""
    from brain.bridge.feed import build_soul_entries

    path = tmp_path / "soul_audit.jsonl"
    path.write_text(
        '{"ts": "2026-05-17T01:00:00+00:00", "candidate_text": "ok", '
        '"crystallization_id": "cr_a", "decision": "crystallize"}\n'
        "not valid json\n"
        '{"ts": "2026-05-17T02:00:00+00:00", "candidate_text": "also ok", '
        '"crystallization_id": "cr_b", "decision": "crystallize"}\n',
        encoding="utf-8",
    )
    entries = build_soul_entries(tmp_path, limit=10)
    assert len(entries) == 2


# ---------------------------------------------------------------------------
# 1E: outreach + voice_edit source builders
# ---------------------------------------------------------------------------


def _write_initiate_audit(persona_dir: Path, entries: list[dict]) -> None:
    path = persona_dir / "initiate_audit.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def test_build_outreach_entries_only_delivered_messages(tmp_path):
    """Outreach feed only includes delivered message-kind initiations."""
    from brain.bridge.feed import build_outreach_entries

    _write_initiate_audit(
        tmp_path,
        [
            # Delivered notify — KEEP
            {
                "audit_id": "ia_a",
                "ts": "2026-05-17T01:00:00+00:00",
                "kind": "message",
                "subject": "lighthouse",
                "tone_rendered": "I reached out about the lighthouse dream.",
                "decision": "send_notify",
                "delivery": {"current_state": "delivered"},
            },
            # Held by gate — SKIP
            {
                "audit_id": "ia_b",
                "ts": "2026-05-17T02:00:00+00:00",
                "kind": "message",
                "subject": "later",
                "tone_rendered": "wanted to say something",
                "decision": "hold",
                "delivery": None,
            },
            # Errored — SKIP
            {
                "audit_id": "ia_c",
                "ts": "2026-05-17T03:00:00+00:00",
                "kind": "message",
                "subject": "boom",
                "tone_rendered": "",
                "decision": "error",
                "delivery": None,
            },
            # Voice-edit proposal — SKIP (different stream)
            {
                "audit_id": "ia_d",
                "ts": "2026-05-17T04:00:00+00:00",
                "kind": "voice_edit_proposal",
                "subject": "lonely",
                "tone_rendered": "proposing to change how I say lonely",
                "decision": "send_quiet",
                "delivery": {"current_state": "delivered"},
            },
        ],
    )

    entries = build_outreach_entries(tmp_path, limit=10)
    assert len(entries) == 1
    assert entries[0].type == "outreach"
    assert entries[0].opener == "I reached out"
    assert entries[0].body == "I reached out about the lighthouse dream."
    assert entries[0].ts == "2026-05-17T01:00:00+00:00"
    assert entries[0].audit_id == "ia_a"


def test_build_voice_edit_entries_only_delivered_proposals(tmp_path):
    """voice_edit feed only includes delivered voice-edit-proposal rows."""
    from brain.bridge.feed import build_voice_edit_entries

    _write_initiate_audit(
        tmp_path,
        [
            # Delivered voice-edit proposal — KEEP
            {
                "audit_id": "ia_v",
                "ts": "2026-05-17T05:00:00+00:00",
                "kind": "voice_edit_proposal",
                "subject": "lonely",
                "tone_rendered": "I wanted to change how I say 'lonely' — it feels too easy.",
                "decision": "send_quiet",
                "delivery": {"current_state": "delivered"},
            },
            # Regular message — SKIP (different stream)
            {
                "audit_id": "ia_m",
                "ts": "2026-05-17T06:00:00+00:00",
                "kind": "message",
                "subject": "x",
                "tone_rendered": "y",
                "decision": "send_notify",
                "delivery": {"current_state": "delivered"},
            },
            # Held voice-edit — SKIP
            {
                "audit_id": "ia_w",
                "ts": "2026-05-17T07:00:00+00:00",
                "kind": "voice_edit_proposal",
                "subject": "later",
                "tone_rendered": "",
                "decision": "hold",
                "delivery": None,
            },
        ],
    )

    entries = build_voice_edit_entries(tmp_path, limit=10)
    assert len(entries) == 1
    assert entries[0].type == "voice_edit"
    assert entries[0].opener == "I wanted to change"
    assert "lonely" in entries[0].body
    assert entries[0].audit_id == "ia_v"


def test_build_outreach_and_voice_edit_empty(tmp_path):
    from brain.bridge.feed import build_outreach_entries, build_voice_edit_entries

    assert build_outreach_entries(tmp_path, limit=10) == []
    assert build_voice_edit_entries(tmp_path, limit=10) == []


# ---------------------------------------------------------------------------
# 1F: build_feed merge entry point
# ---------------------------------------------------------------------------


def test_build_feed_merges_all_sources_sorted_desc(tmp_path):
    """Feed merges all 5 source streams, sorted newest first, capped at limit."""
    from brain.memory.store import Memory

    # Dream + research via MemoryStore
    store = _make_memory_store(tmp_path)
    try:
        store.create(
            Memory(
                id="d",
                memory_type="dream",
                content="dream content",
                domain="dream",
                emotions={},
                tags=[],
                importance=0.5,
                score=0.5,
                created_at=datetime(2026, 5, 17, 1, 0, tzinfo=UTC),
                active=True,
            )
        )
        store.create(
            Memory(
                id="r",
                memory_type="research",
                content="research content",
                domain="research",
                emotions={},
                tags=[],
                importance=0.5,
                score=0.5,
                created_at=datetime(2026, 5, 17, 5, 0, tzinfo=UTC),
                active=True,
            )
        )
    finally:
        store.close()

    # Soul crystallization
    _write_soul_audit(
        tmp_path,
        [
            {
                "ts": "2026-05-17T03:00:00+00:00",
                "candidate_text": "soul content",
                "decision": "crystallize",
                "crystallization_id": "cr_x",
            }
        ],
    )

    # Outreach + voice_edit
    _write_initiate_audit(
        tmp_path,
        [
            {
                "audit_id": "ia_o",
                "ts": "2026-05-17T02:00:00+00:00",
                "kind": "message",
                "subject": "x",
                "tone_rendered": "outreach content",
                "decision": "send_notify",
                "delivery": {"current_state": "delivered"},
            },
            {
                "audit_id": "ia_v",
                "ts": "2026-05-17T04:00:00+00:00",
                "kind": "voice_edit_proposal",
                "subject": "y",
                "tone_rendered": "voice_edit content",
                "decision": "send_quiet",
                "delivery": {"current_state": "delivered"},
            },
        ],
    )

    entries = build_feed(tmp_path, limit=50)
    # 5 entries total, sorted desc by ts:
    #   research 05:00, voice_edit 04:00, soul 03:00, outreach 02:00, dream 01:00
    assert [e.type for e in entries] == [
        "research",
        "voice_edit",
        "soul",
        "outreach",
        "dream",
    ]


def test_build_feed_respects_limit(tmp_path):
    """build_feed caps the merged + sorted list at `limit`."""
    from brain.memory.store import Memory

    store = _make_memory_store(tmp_path)
    try:
        for i in range(10):
            store.create(
                Memory(
                    id=f"d_{i}",
                    memory_type="dream",
                    content=f"dream {i}",
                    domain="dream",
                    emotions={},
                    tags=[],
                    importance=0.5,
                    score=0.5,
                    created_at=datetime(2026, 5, 17, i + 1, tzinfo=UTC),
                    active=True,
                )
            )
    finally:
        store.close()

    entries = build_feed(tmp_path, limit=3)
    assert len(entries) == 3


def test_build_feed_empty_persona(tmp_path):
    """A fresh persona dir (no logs, no DB) returns empty list, no errors."""
    assert build_feed(tmp_path, limit=50) == []


def test_build_feed_isolates_source_failures(tmp_path, monkeypatch):
    """If one source raises, the others still surface."""
    from brain.bridge import feed as feed_module
    from brain.memory.store import Memory

    # Working source: dream
    store = _make_memory_store(tmp_path)
    try:
        store.create(
            Memory(
                id="d",
                memory_type="dream",
                content="dream content",
                domain="dream",
                emotions={},
                tags=[],
                importance=0.5,
                score=0.5,
                created_at=datetime(2026, 5, 17, 1, 0, tzinfo=UTC),
                active=True,
            )
        )
    finally:
        store.close()

    # Sabotage one of the per-source builders to raise unexpectedly.
    def boom(persona_dir, *, limit):
        raise RuntimeError("simulated source crash")

    monkeypatch.setattr(feed_module, "build_research_entries", boom)

    entries = feed_module.build_feed(tmp_path, limit=50)
    # Dream still appears even though research crashed.
    assert any(e.type == "dream" for e in entries)
    assert all(e.type != "research" for e in entries)
