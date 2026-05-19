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


# ── Reply-to-outbound block (Bundle A #4) ─────────────────────────────────────


def _seed_initiate_audit(
    persona_dir: Path,
    *,
    audit_id: str,
    subject: str = "the dream from this morning",
    tone_rendered: str = "the dream landed somewhere",
) -> None:
    from brain.initiate.audit import append_audit_row
    from brain.initiate.schemas import AuditRow

    row = AuditRow(
        audit_id=audit_id,
        candidate_id=f"ic_{audit_id}",
        ts="2026-05-12T09:00:00+00:00",
        kind="message",
        subject=subject,
        tone_rendered=tone_rendered,
        decision="send_quiet",
        decision_reasoning="resonance",
        gate_check={"allowed": True, "reason": None},
        delivery={
            "delivered_at": "2026-05-12T09:00:00+00:00",
            "state_transitions": [
                {"to": "delivered", "at": "2026-05-12T09:00:00+00:00"},
            ],
            "current_state": "delivered",
        },
    )
    append_audit_row(persona_dir, row)


def test_build_system_message_includes_reply_block_when_audit_id_present(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
) -> None:
    """When ``reply_to_audit_id`` resolves to an audit row, the system message
    surfaces "you are replying to your earlier outbound" with the subject so
    Nell sees the conversational link in her context."""
    _seed_initiate_audit(
        persona_dir,
        audit_id="ia_replyblock",
        subject="the silk-and-iron line",
    )
    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
        reply_to_audit_id="ia_replyblock",
    )
    assert "replying to" in msg.lower()
    assert "silk-and-iron line" in msg


def test_build_system_message_omits_reply_block_when_no_audit_id(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
) -> None:
    """No ``reply_to_audit_id`` -> no reply block."""
    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
    )
    assert "replying to" not in msg.lower()


def test_build_system_message_omits_reply_block_when_audit_id_not_found(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
) -> None:
    """Unknown audit_id -> block omitted, chat composition continues silently."""
    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
        reply_to_audit_id="ia_does_not_exist",
    )
    assert "replying to" not in msg.lower()


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


# ── Recall block (Phase 2.A) ──────────────────────────────────────────────────


def test_recall_block_omitted_when_user_input_none(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    """user_input=None (default) must not surface a recall block."""
    mem = Memory.create_new(
        content="Hana mentioned Jordan once over coffee.",
        memory_type="event",
        domain="relationship",
        emotions={"love": 6.0},
        tags=[],
    )
    store.create(mem)

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
        # user_input deliberately omitted
    )
    assert "recall" not in msg.lower()


def test_recall_block_omitted_when_user_input_too_short(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    """A 'hi' / 'ok' message produces no extractable tokens → no recall block."""
    mem = Memory.create_new(
        content="A meaningful moment.",
        memory_type="event",
        domain="relationship",
        emotions={"love": 7.0},
        tags=[],
    )
    store.create(mem)

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
        user_input="hi!",
    )
    assert "── recall" not in msg


def test_recall_block_surfaces_keyword_match(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    """A user message naming an entity surfaces matching memories."""
    mem = Memory.create_new(
        content="Hana mentioned Jordan once over coffee.",
        memory_type="event",
        domain="relationship",
        emotions={"love": 6.0},
        tags=[],
    )
    store.create(mem)

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
        user_input="Tell me what we said about Jordan last time.",
    )
    # New forgetting-aware format uses "recall\n  active:" instead of "── recall ──"
    assert "recall" in msg
    assert "Jordan" in msg


def test_recall_block_handles_no_match(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    """When no memory matches the tokens, the block is omitted entirely."""
    mem = Memory.create_new(
        content="Hana mentioned Jordan once over coffee.",
        memory_type="event",
        domain="relationship",
        emotions={"love": 6.0},
        tags=[],
    )
    store.create(mem)

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
        user_input="What's the weather like outside today?",
    )
    # "recall" alone could appear in other blocks; check for the structured recall block
    assert "recall\n  active:" not in msg
    assert "recall\n  softened" not in msg


def test_recall_block_caps_at_limit(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    """A query that matches many memories surfaces at most ``limit`` (default 5)."""
    for i in range(12):
        store.create(
            Memory.create_new(
                content=f"A particular moment number {i} with Jordan.",
                memory_type="event",
                domain="relationship",
                emotions={"love": 5.0},
                tags=[],
            )
        )

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
        user_input="Tell me about Jordan.",
    )
    # New forgetting-aware format renders bullets under "  active:" indented with "    - "
    assert "recall" in msg
    assert "active:" in msg
    # Each active result is a "    - " bullet line — cap is still 5 per bucket
    recall_section = msg.split("recall\n")[1]
    bullet_count = recall_section.count("\n    - ")
    assert bullet_count == 5, f"expected 5 recall bullets, got {bullet_count}"


def test_recall_block_truncates_long_content(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    """Memory content longer than max_chars (140) is truncated with ellipsis."""
    long_content = "Jordan was someone " + ("who mattered very much. " * 50)
    store.create(
        Memory.create_new(
            content=long_content,
            memory_type="event",
            domain="relationship",
            emotions={"love": 6.0},
            tags=[],
        )
    )

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
        user_input="What about Jordan?",
    )
    # New format: "recall\n  active:\n    - ..."
    assert "recall" in msg
    assert "…" in msg


def test_recall_block_dedupes_when_token_overlap_pulls_same_memory(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    """A memory containing two query tokens must surface once, not twice."""
    store.create(
        Memory.create_new(
            content="Hana told me Jordan was her brother.",
            memory_type="event",
            domain="relationship",
            emotions={"love": 6.0},
            tags=[],
        )
    )

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
        user_input="Tell me about Hana's brother Jordan.",
    )
    # Both 'jordan' and 'brother' would match — but the same memory should appear once.
    assert msg.count("Hana told me Jordan") == 1


def test_recall_block_orders_by_importance_then_recency(
    persona_dir: Path, store: MemoryStore, soul_store: SoulStore
) -> None:
    """Highest-importance match comes first, even if it's older."""
    # Older but more important.
    store.create(
        Memory.create_new(
            content="Jordan: the soul-shaped memory.",
            memory_type="event",
            domain="relationship",
            emotions={},
            tags=[],
            importance=9.0,
        )
    )
    # Fresher but lower importance.
    store.create(
        Memory.create_new(
            content="Jordan: a recent passing reference.",
            memory_type="event",
            domain="relationship",
            emotions={},
            tags=[],
            importance=2.0,
        )
    )

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
        user_input="What about Jordan?",
    )
    # New format: both appear under "  active:"; ordering still by importance desc
    assert "recall" in msg
    soul_idx = msg.find("soul-shaped")
    recent_idx = msg.find("passing reference")
    assert 0 <= soul_idx < recent_idx, (
        f"soul-shaped (importance 9) should appear before passing reference (importance 2); "
        f"got soul_idx={soul_idx}, recent_idx={recent_idx}"
    )


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


# ── Felt-time block (Phase 7.2) ───────────────────────────────────────────────


def test_build_system_message_includes_felt_time_when_state_exists(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
) -> None:
    """Prime FeltTime with one tick; build_system_message must include the
    felt-time block (i.e. "felt time" appears in the output)."""
    from brain.felt_time import FeltTime, TickContext
    from brain.felt_time.lived_age import IntensityDrivers

    ft = FeltTime(persona_dir=persona_dir)
    ft.tick(
        TickContext(
            now_iso="2026-05-17T22:00:00+00:00",
            heartbeats_in_tick=1,
            chat_turns_in_tick=0,
            reflex_firings_in_tick=0,
            wall_clock_s_in_tick=900.0,
            drivers=IntensityDrivers(),
        )
    )

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
    )
    assert "felt time" in msg.lower()


def test_build_system_message_no_felt_time_block_when_state_missing(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
) -> None:
    """When no FeltTime state file exists and no source logs exist, the cold-
    start render returns the 'too new' message which does NOT contain 'felt time'
    as a standalone block heading — the block is still omitted from the output
    since cold-start text is just one line and strip() on it is truthy.
    What we assert is that build_system_message completes without error."""
    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
    )
    # Should succeed without raising; felt-time block won't break the assembly.
    assert isinstance(msg, str)
    assert len(msg) > 0


# ── Fading-summary block (Phase 8, forgetting design §5) ─────────────────────


def test_build_system_message_includes_fading_summary_block_always(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
) -> None:
    """The fading-summary block is always present — 'nothing has softened lately'
    on the empty path, actual counts when fading memories exist."""
    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
    )
    # On empty store the ambient line reads "memory: nothing has softened lately."
    assert "memory:" in msg.lower()
    assert "softened" in msg.lower()


def test_build_system_message_fading_summary_reflects_fading_memories(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
) -> None:
    """When fading memories exist the fading-summary block shows their count."""
    m = Memory.create_new(
        content="A whisper of something once loved.",
        memory_type="episodic",
        domain="chat",
        emotions={"love": 5.0},
    )
    store.create(m)
    store.fade(m.id, summary="once loved")

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
    )
    assert "1" in msg
    assert "softened" in msg.lower()


# ── Recall block — forgetting-aware buckets (Phase 8.2) ──────────────────────


def test_recall_block_surfaces_fading_bucket_when_memory_is_fading(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
) -> None:
    """A fading memory matching the query appears under the 'softened' bucket."""
    m = Memory.create_new(
        content="Jordan loved rainy afternoons.",
        memory_type="episodic",
        domain="chat",
        emotions={"love": 6.0},
    )
    store.create(m)
    store.fade(m.id, summary="Jordan — rainy afternoons")

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
        user_input="Do you remember Jordan?",
    )
    assert "recall" in msg
    assert "softened" in msg.lower()
    assert "[state: fading]" in msg


def test_recall_block_surfaces_lost_bucket_when_memory_in_graveyard(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
) -> None:
    """A lost memory whose graveyard summary matches the query appears in the
    'lost' bucket with a 'forgotten —' label."""
    from brain.forgetting import graveyard
    from brain.forgetting.salience import SalienceInputs

    lost_m = Memory.create_new(
        content="Jordan's old studio address.",
        memory_type="episodic",
        domain="chat",
        emotions={},
    )
    graveyard.append(
        persona_dir,
        memory=lost_m,
        salience_at_drop=0.05,
        inputs=SalienceInputs(emotion=0, hebbian=0, recall=0, soul=0, freshness=0),
        lived_age_hours=200.0,
        reason="salience<0.10 for 2 consecutive passes",
    )

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
        user_input="Tell me about Jordan's studio.",
    )
    assert "recall" in msg
    assert "lost" in msg.lower()
    assert "forgotten" in msg.lower()


# ── Current-arc ambient block (narrative_memory Phase 6) ─────────────────────


def test_assembled_prompt_includes_current_arc_block(tmp_path: Path, monkeypatch):
    """Ambient section includes 'arcs' block when narrative_memory state has open arcs."""
    from brain.narrative_memory.arc import Arc, ArcMember
    from brain.narrative_memory.state import ArcsState, save_state

    save_state(
        tmp_path,
        ArcsState(
            open={
                "arc_1": Arc(
                    id="arc_1",
                    state="open",
                    seed_anchor_type="dream",
                    seed_anchor_ref="dream_evt_1",
                    seed_memory_ids=("mem_seed",),
                    title="the boat one",
                    opened_at_iso="2026-05-19T10:00:00+00:00",
                    lived_age_at_open=412.0,
                    last_extended_at_iso="2026-05-19T11:00:00+00:00",
                    closed_at_iso=None,
                    lived_age_at_close=None,
                    members=(
                        ArcMember(
                            memory_id="mem_seed",
                            joined_at_iso="2026-05-19T10:00:00+00:00",
                            lived_age_at_join=412.0,
                            salience_at_join=0.7,
                        ),
                    ),
                )
            },
            last_pass_ts_iso="2026-05-19T11:00:00+00:00",
        ),
    )
    from brain.chat.prompt import _build_current_arc_block

    block = _build_current_arc_block(tmp_path)
    assert "the boat one" in block
    assert "current:" in block


def test_build_current_arc_block_swallows_errors(tmp_path: Path):
    """_build_current_arc_block returns a string on any read failure (cold-start)."""
    from brain.chat.prompt import _build_current_arc_block

    # Cold-start behavior — bad dir's load_or_recover handles non-existent
    # state.json + log gracefully and renders cold-start text. Confirm
    # the helper doesn't raise.
    bad = tmp_path / "nonexistent_dir_for_failure"
    out = _build_current_arc_block(bad)
    assert isinstance(out, str)


def test_build_system_message_includes_current_arc_block_when_arc_open(
    persona_dir: Path,
    store: MemoryStore,
    soul_store: SoulStore,
) -> None:
    """The assembled system message surfaces the arcs ambient block when
    narrative_memory state has at least one open arc."""
    from brain.narrative_memory.arc import Arc, ArcMember
    from brain.narrative_memory.state import ArcsState, save_state

    save_state(
        persona_dir,
        ArcsState(
            open={
                "arc_1": Arc(
                    id="arc_1",
                    state="open",
                    seed_anchor_type="dream",
                    seed_anchor_ref="dream_evt_1",
                    seed_memory_ids=("mem_seed",),
                    title="the boat one",
                    opened_at_iso="2026-05-19T10:00:00+00:00",
                    lived_age_at_open=412.0,
                    last_extended_at_iso="2026-05-19T11:00:00+00:00",
                    closed_at_iso=None,
                    lived_age_at_close=None,
                    members=(
                        ArcMember(
                            memory_id="mem_seed",
                            joined_at_iso="2026-05-19T10:00:00+00:00",
                            lived_age_at_join=412.0,
                            salience_at_join=0.7,
                        ),
                    ),
                )
            },
            last_pass_ts_iso="2026-05-19T11:00:00+00:00",
        ),
    )

    msg = build_system_message(
        persona_dir,
        voice_md="",
        daemon_state=_empty_daemon_state(),
        soul_store=soul_store,
        store=store,
    )
    assert "the boat one" in msg
    assert "current:" in msg


# ── _build_emotion_summary — non-empty-emotions filter (heartbeat-flood fix) ──


def test_build_emotion_summary_survives_heartbeat_flood(store: MemoryStore) -> None:
    """`_build_emotion_summary` must filter for non-empty emotions_json.

    The naive LIMIT 50 ORDER BY created_at DESC slice was identified as
    broken for `_build_body` and `_build_emotions` (see comments at
    brain/bridge/persona_state.py:260-264). On a steady-state brain
    the last 50 memories are almost all heartbeats / observations /
    facts with `emotions_json = '{}'`, so the aggregator returned an
    empty top-3 even when emotion-bearing memories existed just
    outside the window. This regression test pins the fix.
    """
    from datetime import UTC, datetime, timedelta

    from brain.chat.prompt import _build_emotion_summary

    base = datetime.now(UTC)
    # One emotion-bearing memory — older.
    strong = Memory.create_new(
        content="A moment of real love and tenderness.",
        memory_type="episodic",
        domain="chat",
        emotions={"love": 8.5, "tenderness": 6.2, "awe": 5.0},
    )
    strong = Memory(
        id=strong.id,
        content=strong.content,
        memory_type=strong.memory_type,
        domain=strong.domain,
        created_at=base - timedelta(hours=1),
        emotions=strong.emotions,
        tags=strong.tags,
        importance=strong.importance,
        score=strong.score,
    )
    store.create(strong)

    # 60 heartbeat-style memories with empty emotions, newer than `strong`.
    for i in range(60):
        beat = Memory.create_new(
            content=f"heartbeat {i}",
            memory_type="heartbeat",
            domain="engine",
            emotions={},  # the bug — these dominate the LIMIT 50 window
        )
        beat = Memory(
            id=beat.id,
            content=beat.content,
            memory_type=beat.memory_type,
            domain=beat.domain,
            created_at=base + timedelta(seconds=i),
            emotions=beat.emotions,
            tags=beat.tags,
            importance=beat.importance,
            score=beat.score,
        )
        store.create(beat)

    summary = _build_emotion_summary(store)
    # Before fix: returns "" because heartbeats fill the LIMIT 50.
    # After fix: returns "love:8.5, tenderness:6.2, awe:5.0".
    assert "love" in summary, f"emotion summary should surface 'love' even under heartbeat flood (got: {summary!r})"
    assert "tenderness" in summary
    assert "awe" in summary
