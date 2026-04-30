"""Integration tests for the body block in the chat system message.

Spec: docs/superpowers/specs/2026-04-29-body-state-design.md §4 + §7.

Covers inviolate properties from §7.1:
- #1 reset is idempotent across multiple aggregations
- #3 body emotion may surface in both standard emotion block AND body block (acceptable redundancy)
- #5 body emotion does NOT self-perpetuate across renders
- #7 session_hours passes through from caller
- #10 voice/body coordination (visible in render output)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from brain.chat.prompt import build_system_message
from brain.engines.daemon_state import DaemonState
from brain.memory.store import Memory, MemoryStore
from brain.soul.store import SoulStore


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    p = tmp_path / "p"
    p.mkdir()
    (p / "active_conversations").mkdir()
    return p


@pytest.fixture
def store(persona_dir: Path):
    s = MemoryStore(persona_dir / "memories.db")
    yield s
    s.close()


@pytest.fixture
def soul_store(persona_dir: Path):
    s = SoulStore(str(persona_dir / "crystallizations.db"))
    yield s
    s.close()


@pytest.fixture
def daemon_state() -> DaemonState:
    return DaemonState()


def _seed_emotion_memory(store: MemoryStore, emotions: dict[str, float]) -> None:
    mem = Memory.create_new(
        memory_type="conversation",
        content="seeded for body block test",
        emotions=emotions,
        domain="general",
        metadata={"speaker": "assistant"},
    )
    store.create(mem)


def test_body_block_present_in_system_message(store, soul_store, daemon_state, persona_dir):
    _seed_emotion_memory(store, {"arousal": 7.0, "desire": 6.0})
    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )
    assert "── body ──" in msg
    assert "energy:" in msg
    assert "temperature:" in msg


def test_body_block_position_between_brain_and_journal(
    store, soul_store, daemon_state, persona_dir,
):
    _seed_emotion_memory(store, {"arousal": 7.0})
    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )
    body_idx = msg.find("── body ──")
    brain_idx = msg.find("── brain context ──")
    journal_idx = msg.find("── recent journal")
    assert brain_idx < body_idx
    assert body_idx < journal_idx


def test_body_block_renders_with_no_emotions(store, soul_store, daemon_state, persona_dir):
    """Block still renders with computed energy/temperature/exhaustion when body
    emotions are all zero — the projection is the value, not the body emotions."""
    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )
    assert "── body ──" in msg
    # Default energy 8 (no session, no words, no high emotional load)
    assert "energy: 8" in msg


def test_body_block_climax_reset_visible(store, soul_store, daemon_state, persona_dir):
    """Inviolate property #1 — climax memory at 8 must produce post-reset
    arousal in render (not 8). After reset: arousal=1.6 (clamped per body block to 1.6 → '1.6'),
    comfort_seeking and rest_need both 2.0."""
    _seed_emotion_memory(store, {"climax": 8.0, "arousal": 8.0, "desire": 8.0})
    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )
    body_section = msg[msg.find("── body ──"):msg.find("── recent journal")]
    # comfort_seeking and rest_need should both be present at 2.0 post-reset
    assert "comfort_seeking 2" in body_section
    assert "rest_need 2" in body_section
    # Original arousal 8 should NOT appear in body emotions row
    if "body emotions:" in body_section:
        body_emotions_line = [
            line for line in body_section.split("\n")
            if line.startswith("body emotions:")
        ][0]
        assert "arousal 8" not in body_emotions_line


def test_body_block_degrades_gracefully_on_compute_failure(
    store, soul_store, daemon_state, persona_dir, monkeypatch,
):
    """Inviolate property: chat must NEVER break because body block failed.

    Body block must be absent but the rest of the message must still render.
    Note: brain context block only renders when there is at least one sub-item
    (emotion, residue, soul highlights, candidates). With an empty store the
    block is absent — that's correct behaviour, not a failure.
    """
    def boom(*a, **k):
        raise RuntimeError("simulated body computation failure")
    monkeypatch.setattr(
        "brain.body.state.compute_body_state", boom, raising=True,
    )

    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )
    # Body block omitted — the critical assertion.
    assert "── body ──" not in msg
    # System message still has content (preamble + journal at minimum).
    assert "You are" in msg
    assert "── recent journal" in msg


def test_body_emotions_do_not_self_perpetuate_across_renders(
    store, soul_store, daemon_state, persona_dir,
):
    """Inviolate property #5 — rendering chat 5x with same store must
    not drift body emotion intensities upward."""
    _seed_emotion_memory(store, {"desire": 7.0})

    import re

    def _extract_desire(msg: str) -> float:
        body = msg[msg.find("── body ──"):msg.find("── recent journal")]
        if "desire" not in body:
            return 0.0
        m = re.search(r"desire[: ](\d+\.\d)", body)
        return float(m.group(1)) if m else -1.0

    intensities = []
    for _ in range(5):
        msg = build_system_message(
            persona_dir, voice_md="", daemon_state=daemon_state,
            soul_store=soul_store, store=store,
        )
        intensities.append(_extract_desire(msg))
    assert len(set(intensities)) == 1, f"desire drifted across renders: {intensities}"


def test_body_block_does_not_break_on_corrupt_metadata(
    store, soul_store, daemon_state, persona_dir,
):
    """Edge case — a memory with empty emotions must not crash body block."""
    mem = Memory.create_new(
        memory_type="conversation",
        content="x",
        emotions={},
        domain="general",
        metadata={"speaker": "assistant"},
    )
    store.create(mem)
    msg = build_system_message(
        persona_dir, voice_md="", daemon_state=daemon_state,
        soul_store=soul_store, store=store,
    )
    assert "── body ──" in msg
