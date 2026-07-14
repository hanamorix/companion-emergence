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
    """Canary (D2), IMAGE path: bound the combined builder, where the reboot
    really can be buried. Ledger: project_companion_emergence_deferred.md.

    `build_system_message` is one blob — inventory, ambient and the reboot in a
    single string, reboot last — so total length is still a fair proxy for
    burial here, and this is the only path where it is. Text turns went to
    build_static_system_message + build_volatile_context in v0.0.39; their guard
    is test_deferred_d2_volatile_tail_stays_small below.

    9_000 was ~3x the v0.0.33 fixture (2901). It is now ~1.5x. That is real:
    the tool inventory is ~3.4k of it, deliberately, so she cannot be wrong
    about her own faculties. If this trips, take the compression pass — do not
    raise the bound to make it quiet.
    """
    assert len(system_message) < 9_000
    assert system_message.rstrip().endswith("the answer needs room.")


def test_deferred_d2_volatile_tail_stays_small(persona_dir: Path):
    """Canary (D2), TEXT path: bound what actually precedes the reboot.

    D2 was written when there was one builder, so total length was a fair proxy
    for "ambient bloat buries the reboot". v0.0.39's caching split moved the
    volatile blocks to the stdin tail — for cache reasons, but the effect is
    D2's own "ambient block reorder": on this path the reboot now rides at the
    END of a ~1.5k tail, after the history, and the frozen prefix that grew
    (voice, fence, inventory) sits before the history where it cannot bury it.

    So the quantity that matters here is the tail, not the total. Growth in the
    static prefix is a cost question — the prompt-cache pays it once a session.
    Growth in THIS is the recency question, and it is what pushes the reboot up
    away from the model's attention.
    """
    from brain.chat.prompt import build_volatile_context
    from brain.engines.daemon_state import DaemonState
    from brain.memory.store import MemoryStore
    from brain.soul.store import SoulStore

    store = MemoryStore(persona_dir / "memories.db")
    soul = SoulStore(persona_dir / "crystallizations.db")
    try:
        tail = build_volatile_context(
            persona_dir,
            voice_md="sweater-wearing novelist",
            daemon_state=DaemonState(),
            soul_store=soul,
            store=store,
            user_input="hi",
        )
    finally:
        store.close()

    # ~3x the current tail (≈1.5k), the same ratio D2's original bound used.
    assert len(tail) < 4_500
    # The reboot is the last thing in the tail, and the tail is the last thing
    # the model reads. Both halves matter: a bound alone would pass with the
    # reboot moved to the top of the tail.
    assert tail.rstrip().endswith("the answer needs room.")
