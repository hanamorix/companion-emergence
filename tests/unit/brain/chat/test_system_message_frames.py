"""Tests that build_system_message embeds the monologue + reply frames."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    p = tmp_path / "personas" / "nell"
    p.mkdir(parents=True)
    (p / "voice.md").write_text("sweater-wearing novelist")
    return p


def _build(persona_dir: Path, user_input: str = "hi") -> str:
    from brain.chat.prompt import build_system_message
    from brain.engines.daemon_state import DaemonState
    from brain.memory.store import MemoryStore
    from brain.soul.store import SoulStore

    store = MemoryStore(persona_dir / "memories.db")
    soul = SoulStore(persona_dir / "crystallizations.db")
    try:
        return build_system_message(
            persona_dir,
            voice_md="sweater-wearing novelist",
            daemon_state=DaemonState(),
            soul_store=soul,
            store=store,
            user_input=user_input,
        )
    finally:
        store.close()


def test_system_message_contains_monologue_tool_invitation(persona_dir: Path):
    # Assert the frame, not the tool name: `record_monologue` is one of the 27
    # names the generated tool inventory puts in every system message, so the
    # bare name is true of any message at all — including one built with no
    # monologue frame.
    msg = _build(persona_dir)
    assert "── inner monologue" in msg
    assert "record_monologue" in msg.split("── inner monologue")[1]


def test_system_message_contains_reply_frame_label(persona_dir: Path):
    msg = _build(persona_dir)
    assert "── visible reply" in msg


def test_reply_frame_comes_after_monologue_frame(persona_dir: Path):
    """Reply frame must be last so the model treats it as the immediate context.

    Anchored on the frame marker, NOT on `record_monologue`. The tool inventory
    names that tool near the top of every message, so indexing on the name
    measured the inventory's position and would have held with the monologue
    frame moved after the reply, or gone.
    """
    msg = _build(persona_dir)
    assert msg.index("── inner monologue") < msg.index("── visible reply")


def test_monologue_frame_includes_persona_name(persona_dir: Path):
    # Look inside the frame. The inventory is rendered with the companion's
    # name in it, so `"nell" in msg` is true however the frame is built — the
    # same way this test's sibling in test_prompt.py went quietly dead.
    msg = _build(persona_dir)
    frame = msg.split("── inner monologue")[1].split("── visible reply")[0]
    assert "nell" in frame.lower()


def test_existing_voice_block_still_present(persona_dir: Path):
    """Don't regress existing prompt assembly — voice content must survive the new frames."""
    msg = _build(persona_dir)
    # The creative-dna block is always assembled; its header is a reliable marker
    # that the existing prompt infrastructure hasn't been disrupted.
    assert "── creative dna" in msg


@pytest.fixture
def system_message(persona_dir: Path) -> str:
    return _build(persona_dir)


def test_reply_frame_is_last_block(system_message: str):
    """v0.0.33 Track 2b: the address reboot must be the LAST substantive
    content — recency-positioned for models that weight late context."""
    assert system_message.rstrip().endswith("the answer needs room.")
    assert "never 'she'/'her'/'they'" in system_message.split("── visible reply")[-1]


def test_deferred_d2_prompt_size_canary(system_message: str):
    """Canary (D2): if ambient blocks bloat enough to bury the reply-frame
    reboot, revisit the deferred compression/reorder pass. Ledger:
    project_companion_emergence_deferred.md.

    Headroom is mostly spent. The bound of 9_000 was set at ~3x the fixture's
    then-size (2901 chars, v0.0.33). The generated tool inventory added ~3823
    chars ABOVE the reply frame — which is the burial this canary watches for —
    taking the fixture to ~6726, about 1.3x of bound. It still passes, and the
    inventory is a deliberate trade (a complete toolset in the frozen prefix,
    paid once per session as cache-create, then cache-read). But the next
    ambient block of any size trips this. When it does: that is the canary
    working, not a stale bound to raise — take the compression/reorder pass.
    """
    assert len(system_message) < 9_000
    assert system_message.rstrip().endswith("the answer needs room.")
