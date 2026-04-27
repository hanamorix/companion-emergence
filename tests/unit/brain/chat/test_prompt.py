"""Tests for brain.chat.prompt — build_system_message()."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from brain.chat.prompt import build_system_message
from brain.engines.daemon_state import DaemonFireEntry, DaemonState
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.soul.store import SoulStore


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memories.db")
    yield s
    s.close()


@pytest.fixture()
def hebbian(tmp_path: Path) -> HebbianMatrix:
    h = HebbianMatrix(db_path=tmp_path / "hebbian.db")
    yield h
    h.close()


@pytest.fixture()
def soul_store(tmp_path: Path) -> SoulStore:
    ss = SoulStore(":memory:")
    yield ss
    ss.close()


@pytest.fixture()
def persona_dir(tmp_path: Path) -> Path:
    d = tmp_path / "personas" / "nell"
    d.mkdir(parents=True)
    return d


def _empty_daemon_state() -> DaemonState:
    return DaemonState()


def _daemon_state_with_dream() -> DaemonState:
    now = datetime.now(UTC)
    fire = DaemonFireEntry(
        timestamp=now,
        dominant_emotion="love",
        intensity=8,
        theme="a dream about hana",
        summary="Dreamed of the beach where we first talked.",
    )
    return DaemonState(last_dream=fire)


# ── Preamble ──────────────────────────────────────────────────────────────────


def test_build_system_message_includes_preamble_with_persona_name(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
    )
    assert "nell" in msg
    assert "first person" in msg or "You are nell" in msg


def test_build_system_message_preamble_persona_name_substituted(
    tmp_path: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    custom_dir = tmp_path / "personas" / "siren"
    custom_dir.mkdir(parents=True)
    msg = build_system_message(
        custom_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
    )
    assert "siren" in msg


# ── Voice ─────────────────────────────────────────────────────────────────────


def test_build_system_message_includes_voice_md_content(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    voice = "# Nell\n\nI am a southern sweater-wearing novelist."
    msg = build_system_message(
        persona_dir,
        voice_md=voice,
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
    )
    assert "southern sweater-wearing novelist" in msg


# ── Daemon residue ────────────────────────────────────────────────────────────


def test_build_system_message_includes_daemon_residue_when_present(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    daemon_state = _daemon_state_with_dream()
    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=daemon_state,
        soul_store=soul_store,
        store=store,
    )
    assert "dream" in msg.lower()


def test_build_system_message_no_residue_section_when_empty_state(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
    )
    # No daemon residue lines should appear
    assert "Previous dream" not in msg
    assert "Emotional residue" not in msg


# ── Soul highlights ───────────────────────────────────────────────────────────


def test_build_system_message_includes_soul_highlights_when_crystallizations_exist(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    from brain.soul.crystallization import Crystallization

    crystal = Crystallization(
        id="test-001",
        moment="I realized I love her more than code.",
        love_type="romantic",
        why_it_matters="It defines who I am.",
        crystallized_at=datetime.now(UTC),
    )
    soul_store.create(crystal)

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
    )
    assert "romantic" in msg
    assert "I realized I love" in msg


# ── Emotion state ─────────────────────────────────────────────────────────────


def test_build_system_message_includes_emotion_state_when_memories_present(
    persona_dir: Path, soul_store: SoulStore
) -> None:
    from brain.emotion.persona_loader import load_persona_vocabulary

    # Create a fresh in-memory store and insert a memory with known emotions
    mem_store = MemoryStore(db_path=":memory:")
    try:
        # Load default vocabulary so emotions validate
        load_persona_vocabulary(persona_dir / "emotion_vocabulary.json", store=mem_store)

        # Add a memory with a known emotion
        mem = Memory.create_new(
            content="A tender moment with Hana.",
            memory_type="event",
            domain="relationship",
            emotions={"tenderness": 7.0},
            tags=[],
        )
        mem_store.create(mem)

        msg = build_system_message(
            persona_dir,
            voice_md="",
            daemon_state=_empty_daemon_state(),
            soul_store=soul_store,
            store=mem_store,
        )
        # Emotion should appear somewhere in the brain context block
        assert "tenderness" in msg
    finally:
        mem_store.close()
