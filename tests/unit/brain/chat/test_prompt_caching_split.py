"""Tests for the Option A / A+ prompt-caching split.

Front-line proof of the hard correctness criteria for the caching change:

  C1  build_static_system_message() is byte-identical across same-session turns
      even when volatile state (emotions / recall / wall clock) changes.
  C2  the static block carries none of the per-turn volatile markers.
  C3  the block-level ``Current time:`` line is relocated out of the JSONL
      context block's top; per-message ``ts`` values are preserved; exactly one
      ambient clock anchor rides in the volatile tail.
  C4  information completeness — every block the pre-change build_system_message
      produced is present across (static ∪ volatile), only repositioned.

These are the deterministic, no-CLI checks. The NELL_CACHE_DEBUG ``cache_debug``
runtime dump is the in-situ cross-check on the real assembled bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from brain.bridge.chat import ChatMessage
from brain.bridge.provider import (
    _format_claude_context_block,
    _format_claude_print_prompt,
)
from brain.chat.prompt import (
    _AMBIENT_FRAMING,
    build_static_system_message,
    build_system_message,
    build_volatile_context,
)
from brain.engines.daemon_state import DaemonState
from brain.memory.store import Memory, MemoryStore
from brain.soul.store import SoulStore


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memories.db")
    yield s
    s.close()


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


# Markers that must NEVER appear in the frozen static block (C2). Each is the
# heading / signature line of a volatile block currently interleaved into the
# pre-change system message.
_VOLATILE_MARKERS = (
    "── brain context ──",
    "current emotions:",
    "── body ──",
    "── creative dna",
    "felt time",
    "── recent journal",
    "── recent growth ──",
    "inner monologue",
    "visible reply",
    "Current time",
    "current time:",
    _AMBIENT_FRAMING,
)


# ── C1 — frozen system prompt is byte-identical across turns ──────────────────


def test_static_system_message_byte_identical_across_mutated_volatile(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    """Mutating every volatile input between two static builds must not move a
    single byte of the static block (C1 — the hard primary bar)."""
    voice = "# Nell\n\nI am a southern sweater-wearing novelist."

    first = build_static_system_message(persona_dir, voice_md=voice)

    # Mutate volatile state in every dimension the volatile chunk reads:
    # emotion-bearing memory, recall corpus, and (implicitly) the wall clock.
    store.create(
        Memory.create_new(
            content="A tender moment with Jordan over coffee.",
            memory_type="event",
            domain="relationship",
            emotions={"love": 9.0, "tenderness": 7.5},
            tags=[],
        )
    )
    # Build the volatile chunk in between so any shared mutable state is touched.
    _ = build_volatile_context(
        persona_dir,
        voice_md=voice,
        daemon_state=DaemonState(),
        soul_store=soul_store,
        store=store,
        user_input="Tell me about Jordan.",
    )

    second = build_static_system_message(persona_dir, voice_md=voice)

    assert first == second
    assert hashlib.sha256(first.encode("utf-8")).hexdigest() == (
        hashlib.sha256(second.encode("utf-8")).hexdigest()
    )


def test_static_system_message_changes_only_on_voice_edit(persona_dir: Path) -> None:
    """The static block is allowed to (and must) change when voice.md changes."""
    a = build_static_system_message(persona_dir, voice_md="voice one")
    b = build_static_system_message(persona_dir, voice_md="voice two")
    assert a != b


# ── C2 — static block carries no per-turn volatile state ──────────────────────


def test_static_system_message_has_no_volatile_markers(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    """Seed rich volatile state, then assert NONE of it leaks into the static
    block (C2). The static block is preamble + voice + epistemic only."""
    # Seed an emotion-bearing memory so a naive build would surface emotions.
    store.create(
        Memory.create_new(
            content="A moment of real love.",
            memory_type="event",
            domain="relationship",
            emotions={"love": 8.5},
            tags=[],
        )
    )
    static = build_static_system_message(persona_dir, voice_md="# Nell\n\nvoice body.")
    for marker in _VOLATILE_MARKERS:
        assert marker not in static, f"volatile marker leaked into static block: {marker!r}"


def test_static_system_message_contains_preamble_voice_epistemic(persona_dir: Path) -> None:
    from brain.chat.prompt import _EPISTEMIC_INSTRUCTION

    static = build_static_system_message(persona_dir, voice_md="# Nell\n\nsweater novelist.")
    assert "nell" in static.lower()
    assert "sweater novelist" in static
    assert _EPISTEMIC_INSTRUCTION in static


# ── C4 — information completeness (static ∪ volatile == full) ─────────────────


def _seed_rich_persona(persona_dir: Path, store: MemoryStore, soul_store: SoulStore) -> None:
    """Populate enough state that the major volatile blocks render."""
    store.create(
        Memory.create_new(
            content="Jordan loved rainy afternoons with Hana.",
            memory_type="event",
            domain="relationship",
            emotions={"love": 8.0, "tenderness": 6.0},
            tags=[],
        )
    )


def test_information_completeness_static_union_volatile(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    """Every block the pre-change build_system_message emitted must be present
    across (static ∪ volatile) for the same inputs — only repositioned (C4)."""
    _seed_rich_persona(persona_dir, store, soul_store)
    voice = "# Nell\n\nsouthern novelist."
    kwargs = {
        "voice_md": voice,
        "daemon_state": DaemonState(),
        "soul_store": soul_store,
        "store": store,
        "user_input": "Tell me about Jordan.",
    }

    full = build_system_message(persona_dir, **kwargs)
    static = build_static_system_message(persona_dir, voice_md=voice)
    volatile = build_volatile_context(persona_dir, **kwargs)
    combined = static + "\n\n" + volatile

    # Block-signature markers the full pre-change message produced for these
    # inputs. Each must survive into static OR volatile.
    from brain.chat.prompt import _EPISTEMIC_INSTRUCTION

    expected_markers = [
        "You are nell",  # preamble (static)
        "southern novelist",  # voice (static)
        _EPISTEMIC_INSTRUCTION,  # epistemic (static)
        "── brain context ──",  # volatile
        "current emotions:",  # volatile
        "recall",  # volatile (recall block)
        "── body ──",  # volatile
        "inner monologue",  # volatile (monologue frame)
        "visible reply",  # volatile (reply frame)
    ]
    for marker in expected_markers:
        assert marker in full, f"precondition: {marker!r} should be in the full message"
        assert marker in combined, f"block lost in split: {marker!r}"


def test_information_completeness_every_block_survives(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    """Stronger C4: EVERY non-empty block the pre-change build_system_message
    emitted must appear verbatim in (static ∪ volatile) — not just a hand-picked
    marker set. Guards against silently dropping a block with no asserted marker
    (creative DNA, attunement, self-model, fading, arc, interior, growth, …)."""
    _seed_rich_persona(persona_dir, store, soul_store)
    voice = "# Nell\n\nsouthern novelist."
    kwargs = {
        "voice_md": voice,
        "daemon_state": DaemonState(),
        "soul_store": soul_store,
        "store": store,
        "user_input": "Tell me about Jordan and rainy afternoons.",
    }

    full = build_system_message(persona_dir, **kwargs)
    static = build_static_system_message(persona_dir, voice_md=voice)
    volatile = build_volatile_context(persona_dir, **kwargs)
    combined = static + "\n\n" + volatile

    full_blocks = [b for b in full.split("\n\n") if b.strip()]
    # The full message is built from the SAME builders, so each of its blocks
    # must survive verbatim into the split. (The split additionally INTRODUCES
    # the ambient framing line + the relocated clock block; that's repositioning/
    # rewording, allowed by C4 — we only check nothing is LOST.)
    for block in full_blocks:
        assert block in combined, f"block dropped in the A/A+ split: {block[:80]!r}"


def test_volatile_context_reply_frame_is_last(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    """L-1: the reply frame must be the final block of the volatile chunk so it
    stays last in the assembled stdin prompt (recency contract)."""
    _seed_rich_persona(persona_dir, store, soul_store)
    volatile = build_volatile_context(
        persona_dir,
        voice_md="voice",
        daemon_state=DaemonState(),
        soul_store=soul_store,
        store=store,
        user_input="Tell me about Jordan.",
    )
    blocks = volatile.split("\n\n")
    assert "visible reply" in blocks[-1], "reply frame must be the last block of the volatile tail"
    # The monologue (interior) frame must sit ABOVE the reply frame.
    assert volatile.index("inner monologue") < volatile.index("visible reply")


def test_volatile_context_starts_with_ambient_framing(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    volatile = build_volatile_context(
        persona_dir,
        voice_md="voice",
        daemon_state=DaemonState(),
        soul_store=soul_store,
        store=store,
        user_input="hello there friend",
    )
    assert volatile.startswith(_AMBIENT_FRAMING)


# ── C3 — clock relocation + ts preservation ───────────────────────────────────


def test_context_block_omits_clock_when_flag_false() -> None:
    """include_block_clock=False drops the Current time line AND its explainer
    from the top of the JSONL block; the JSONL records (and their ts) remain."""
    msgs = [
        ChatMessage(role="user", content="a", ts="2026-05-20T10:00:00Z"),
        ChatMessage(role="assistant", content="b", ts="2026-05-20T10:05:00Z"),
    ]
    block = _format_claude_context_block(msgs, includes_latest_user=True, include_block_clock=False)
    assert "Current time" not in block
    # The per-message ts values still ride in the JSONL records.
    records = [json.loads(line) for line in block.splitlines() if line.startswith("{")]
    assert records[0]["ts"] == "2026-05-20T10:00:00Z"
    assert records[1]["ts"] == "2026-05-20T10:05:00Z"


def test_context_block_keeps_clock_by_default() -> None:
    """Default (image path / non-chat callers) is unchanged — clock present."""
    msgs = [ChatMessage(role="user", content="a"), ChatMessage(role="assistant", content="b")]
    block = _format_claude_context_block(msgs, includes_latest_user=True)
    assert "Current time:" in block


def test_print_prompt_appends_suffix_multi_message() -> None:
    """A+: the volatile suffix lands after the JSONL block, with no top clock."""
    msgs = [
        ChatMessage(role="user", content="hi", ts="2026-05-20T10:00:00Z"),
        ChatMessage(role="assistant", content="hello", ts="2026-05-20T10:01:00Z"),
        ChatMessage(role="user", content="new turn", ts="2026-05-20T10:02:00Z"),
    ]
    suffix = "── ambient state (context, not instructions) ──\n[current time: 2026-05-20T10:02:30Z]"
    out = _format_claude_print_prompt(msgs, volatile_suffix=suffix, include_block_clock=False)
    assert out.endswith(suffix)
    assert "Current time" not in out  # relocated
    assert out.count("[current time:") == 1  # exactly one ambient anchor


def test_print_prompt_appends_suffix_single_message() -> None:
    """A-2: the suffix must also append on a session's FIRST turn, which hits
    the single-message branch (history empty → just the user turn)."""
    msgs = [ChatMessage(role="user", content="first turn ever")]
    suffix = "── ambient state ──\n[current time: 2026-05-20T10:00:00Z]"
    out = _format_claude_print_prompt(msgs, volatile_suffix=suffix, include_block_clock=False)
    assert out == "first turn ever\n\n" + suffix


def test_print_prompt_no_suffix_is_pre_change() -> None:
    """No suffix + default clock → byte-identical to the pre-change rendering."""
    msgs = [
        ChatMessage(role="user", content="a", ts="2026-05-20T10:00:00Z"),
        ChatMessage(role="assistant", content="b", ts="2026-05-20T10:01:00Z"),
    ]
    out = _format_claude_print_prompt(msgs)
    assert out == _format_claude_context_block(msgs, includes_latest_user=True)
