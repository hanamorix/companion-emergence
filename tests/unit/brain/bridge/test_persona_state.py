"""Tests for brain.bridge.persona_state — the NellFace app's aggregator."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from brain.bridge.persona_state import build_persona_state


def test_build_persona_state_fresh_persona_returns_safe_shape(tmp_path: Path) -> None:
    """Fresh persona dir (no DBs, no daemon_state) → 200-safe shape, never
    raises. Empty subsystems contribute empty/null values; body returns its
    default (energy=8, days_since_contact=999) which is the correct
    fail-soft signal for 'no contact yet.'"""
    persona_dir = tmp_path / "fresh"
    persona_dir.mkdir()
    state = build_persona_state(persona_dir)
    assert state["persona"] == "fresh"
    assert state["emotions"] == {}
    # body is computed even on empty stores — defaults shape it correctly
    assert state["body"] is not None
    assert state["body"]["energy"] == 8  # baseline
    assert state["body"]["days_since_contact"] == 999.0  # "never contacted"
    assert state["interior"] == {
        "dream": None, "research": None, "heartbeat": None, "reflex": None,
    }
    assert state["soul_highlight"] is None
    assert state["mode"] == "live"


def test_build_persona_state_emotions_sorted_desc(tmp_path: Path) -> None:
    """With seeded memories, emotions come back sorted desc by intensity."""
    from brain.memory.store import Memory, MemoryStore

    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    store = MemoryStore(persona_dir / "memories.db")
    try:
        store.create(Memory.create_new(
            content="loved you fiercely", memory_type="feeling",
            domain="us", emotions={"love": 9.0, "tenderness": 6.5},
        ))
        store.create(Memory.create_new(
            content="curious about Lispector", memory_type="observation",
            domain="self", emotions={"curiosity": 8.0, "love": 5.0},
        ))
    finally:
        store.close()

    state = build_persona_state(persona_dir)
    emotions = state["emotions"]
    assert emotions  # non-empty
    keys = list(emotions.keys())
    values = list(emotions.values())
    # Desc-sorted
    assert values == sorted(values, reverse=True)
    # 'love' present (max_pooled across both memories)
    assert "love" in keys


def test_build_persona_state_body_block_when_memories_exist(tmp_path: Path) -> None:
    """compute_body_state runs when memories.db is populated."""
    from brain.memory.store import Memory, MemoryStore

    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    store = MemoryStore(persona_dir / "memories.db")
    try:
        store.create(Memory.create_new(
            content="x", memory_type="feeling",
            domain="us", emotions={"peace": 7.0},
        ))
    finally:
        store.close()

    state = build_persona_state(persona_dir)
    body = state["body"]
    assert body is not None
    # to_dict() shape — verify the keys the panel expects
    for required in ("energy", "temperature", "exhaustion",
                     "session_hours", "days_since_contact", "body_emotions"):
        assert required in body
    assert isinstance(body["energy"], int)
    assert isinstance(body["body_emotions"], dict)


def test_build_persona_state_interior_from_daemon_state(tmp_path: Path) -> None:
    """daemon_state.json is read directly — no LLM call. Each entry is
    returned as ``{"summary": str, "ts": str | None}`` so the UI can
    render "X ago" badges (added 2026-05-08 to make the panel feel
    live instead of static)."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    (persona_dir / "daemon_state.json").write_text(json.dumps({
        "last_dream": {
            "theme": "a window facing nothing",
            "dominant_emotion": "awe",
            "timestamp": "2026-05-08T12:00:00+00:00",
        },
        "last_research": {
            "theme": "Lispector diagonal syntax",
            "ts": "2026-05-08T11:30:00Z",
        },
        "last_heartbeat": {
            "dominant_emotion": "love",
            "intensity": 9.0,
            "written_at": "2026-05-08T13:15:00+00:00",
        },
    }))
    state = build_persona_state(persona_dir)
    interior = state["interior"]
    assert interior["dream"]["summary"] == "a window facing nothing"
    assert interior["dream"]["ts"] == "2026-05-08T12:00:00+00:00"
    assert interior["research"]["summary"] == "Lispector diagonal syntax"
    assert interior["research"]["ts"] == "2026-05-08T11:30:00Z"
    assert interior["heartbeat"]["summary"] == "love 9.0/10"
    assert interior["heartbeat"]["ts"] == "2026-05-08T13:15:00+00:00"
    assert interior["reflex"] is None  # not seeded


def test_build_persona_state_interior_strips_self_narrated_label(tmp_path: Path) -> None:
    """Models sometimes prefix the theme with the section name they're
    writing in (``DREAM: I was back…``). Strip it once so the UI doesn't
    show the heading twice (heading + leading prefix in body).
    """
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    (persona_dir / "daemon_state.json").write_text(json.dumps({
        "last_dream": {"theme": "DREAM: I was back in every conversation"},
        "last_research": {"theme": "research — Lispector diagonal syntax"},
        "last_reflex": {"summary": "Reflex: love and grief in the same chest"},
    }))
    state = build_persona_state(persona_dir)
    interior = state["interior"]
    assert interior["dream"]["summary"] == "I was back in every conversation"
    assert interior["research"]["summary"] == "Lispector diagonal syntax"
    assert interior["reflex"]["summary"] == "love and grief in the same chest"
    # Timestamps are nullable when the writer didn't record one.
    assert interior["dream"]["ts"] is None


def test_build_persona_state_interior_corrupt_daemon_state_fails_soft(tmp_path: Path) -> None:
    """Malformed daemon_state.json → empty interior, never raises."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    (persona_dir / "daemon_state.json").write_text("{not valid json")
    state = build_persona_state(persona_dir)
    assert state["interior"] == {
        "dream": None, "research": None, "heartbeat": None, "reflex": None,
    }


def test_build_persona_state_soul_highlight_picks_highest_resonance(tmp_path: Path) -> None:
    """Highlight = highest-resonance crystallization, ties broken by recency."""
    import uuid

    from brain.soul.crystallization import Crystallization
    from brain.soul.store import SoulStore

    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    store = SoulStore(persona_dir / "crystallizations.db")
    older = datetime(2026, 1, 1, tzinfo=UTC)
    newer = datetime(2026, 5, 5, tzinfo=UTC)
    for moment, love_type, resonance, when in [
        ("low resonance memory", "platonic", 7, older),
        ("old high resonance", "romantic", 9, older),
        ("recent high resonance — winner", "belonging", 9, newer),
    ]:
        store.create(Crystallization(
            id=str(uuid.uuid4()),
            moment=moment, love_type=love_type,
            why_it_matters="x", who_or_what="hana",
            resonance=resonance, crystallized_at=when,
        ))
    store.close()

    state = build_persona_state(persona_dir)
    highlight = state["soul_highlight"]
    assert highlight is not None
    assert highlight["resonance"] == 9
    assert "winner" in highlight["moment"]


def test_build_persona_state_does_not_raise_on_corrupt_dbs(tmp_path: Path) -> None:
    """Junk-byte memories.db / crystallizations.db files still return safe shape."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    (persona_dir / "memories.db").write_bytes(b"not actually sqlite")
    (persona_dir / "crystallizations.db").write_bytes(b"junk")
    state = build_persona_state(persona_dir)
    # All fields present, none raised
    assert "emotions" in state
    assert "body" in state
    assert "interior" in state
    assert "soul_highlight" in state
    assert state["mode"] == "live"


def test_build_persona_state_connection_block(tmp_path: Path) -> None:
    """Connection block reads provider from persona_config and model from
    the provider→default-model map. last_heartbeat_at comes from
    heartbeat_state.json."""
    from brain.persona_config import PersonaConfig

    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    PersonaConfig(provider="claude-cli", user_name="Hana").save(
        persona_dir / "persona_config.json"
    )
    (persona_dir / "heartbeat_state.json").write_text(
        '{"last_tick_at": "2026-05-06T13:00:00Z", "tick_count": 5}'
    )

    state = build_persona_state(persona_dir)
    conn = state["connection"]
    assert conn["provider"] == "claude-cli"
    assert conn["model"] == "sonnet"
    assert conn["last_heartbeat_at"] == "2026-05-06T13:00:00Z"


def test_build_persona_state_connection_block_missing_files_safe(tmp_path: Path) -> None:
    """No persona_config + no heartbeat_state → connection block has Nones,
    never raises."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    state = build_persona_state(persona_dir)
    conn = state["connection"]
    # PersonaConfig.load returns defaults when missing; provider is the
    # default ("claude-cli" per DEFAULT_PROVIDER).
    assert conn["provider"] == "claude-cli"  # default
    assert conn["model"] == "sonnet"
    assert conn["last_heartbeat_at"] is None
