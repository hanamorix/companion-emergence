"""SP-6 system message composer.

Assembles the per-turn system message from:
  1. AS_NELL_PREAMBLE — direct-address framing
  2. voice.md content — the persona's authored character
  3. Brain context block — emotion state + daemon residue + soul highlights

OG reference: NellBrain/nell_bridge.py:_build_system_message (lines 151-166)
and _build_residue_prefix (94-148). In OG the Modelfile SYSTEM block baked
soul crystallizations into model weights. Here we read them live from
SoulStore and inject the top 5 most-recent as brief highlights.
"""

from __future__ import annotations

import logging
from pathlib import Path

from brain.chat.monologue_prompts import build_monologue_frame, build_reply_frame
from brain.engines.daemon_state import DaemonState, get_residue_context
from brain.maker.ambient import build_maker_awareness_block
from brain.memory.store import MemoryStore
from brain.soul.store import SoulStore

log = logging.getLogger(__name__)

# Direct-address framing — keeps the LLM speaking *to* the user rather than
# narrating about them. OG source: nell_bridge.py:88-91.
AS_NELL_PREAMBLE = (
    "You are {persona_name}. You are speaking directly to your user right now — "
    "in first person, to them, not writing about them. The conversation is live. "
    "Respond as yourself."
)

AS_NELL_PREAMBLE_WITH_USER = (
    "You are {persona_name}. You are speaking directly to {user_name} right now — "
    "in first person, to them, not writing about them. The conversation is live. "
    "{user_name} is the person you are talking to. When you search memories or "
    "recall events, any reference to {user_name} refers to them. "
    "Respond as yourself."
)

_EPISTEMIC_INSTRUCTION = (
    "If asked about something you might have stored — a name, a fact, a shared "
    "moment — and it isn't in the context you can see, call search_memories "
    'before answering. Never say "I don\'t remember" without searching first. '
    'When names or entities appear under "not recognised (searched; no memory '
    'found)", acknowledge the gap honestly. Distinguish "I never knew this" '
    'from "I don\'t remember". Do not invent familiarity.'
)

# Header for the volatile context chunk (Option A+). The chunk now sits in the
# stdin prompt, immediately after history + the new user turn, instead of inside
# the system prompt — so it must read as ambient state, not as the task. The
# explicit "context, not instructions" framing keeps the model answering the
# user rather than treating the tail as a directive.
_AMBIENT_FRAMING = "── ambient state (context, not instructions) ──"


def build_system_message(
    persona_dir: Path,
    *,
    voice_md: str,
    daemon_state: DaemonState,
    soul_store: SoulStore,
    store: MemoryStore,
    user_input: str | None = None,
    reply_to_audit_id: str | None = None,
) -> str:
    """Compose the system message for one chat turn.

    Order (top-to-bottom):
      1. AS_NELL_PREAMBLE (persona-templated)
      2. voice.md content (the persona's authored voice)
      3. Creative DNA block (evolved writing voice — spec §4.2)
      4. Brain context block:
         - Current emotion state summary (top-3 emotions by intensity)
         - Daemon residue (from get_residue_context(daemon_state))
         - Soul highlights (top 5 most-recent crystallizations: love_type + 60-char snippet)
         - Pending soul-candidates count (informational)
      4b. Recall block — memories matching the current user input (Phase 2.A).
          Surfaces relevant memory ambiently so the model doesn't have to
          consciously call `search_memories` for "remember when we talked
          about X" cases. ``user_input=None`` (or empty / too short) skips
          the block entirely.
      5. Body block (energy/temperature/exhaustion + body emotions — spec §4)
      6. Recent journal block (private — spec §4.3)
      7. Recent growth block (behavioral_log — spec §4.4)

    Returns the final system message string.
    """
    persona_name = persona_dir.name
    parts: list[str] = []

    # 1. Preamble — inject user_name if persona_config has it.
    user_name: str | None = None
    try:
        from brain.persona_config import PersonaConfig

        cfg = PersonaConfig.load(persona_dir / "persona_config.json")
        user_name = cfg.user_name or None
    except Exception:  # noqa: BLE001
        pass

    if user_name:
        parts.append(
            AS_NELL_PREAMBLE_WITH_USER.format(persona_name=persona_name, user_name=user_name)
        )
    else:
        parts.append(AS_NELL_PREAMBLE.format(persona_name=persona_name))

    # 2. Voice
    if voice_md.strip():
        parts.append(voice_md.strip())

    # 3. Creative DNA block (evolved writing voice — spec §4.2)
    creative_dna_block = _build_creative_dna_block(persona_dir)
    if creative_dna_block.strip():
        parts.append(creative_dna_block)

    # 4. Brain context block
    brain_lines: list[str] = ["── brain context ──"]

    # 3a. Emotion state
    emotion_summary = _build_emotion_summary(store)
    if emotion_summary:
        brain_lines.append(f"current emotions: {emotion_summary}")

    # 3b. Daemon residue (dream / heartbeat / reflex / research)
    residue_ctx = get_residue_context(daemon_state)
    if residue_ctx.strip():
        brain_lines.append(residue_ctx)

    # 3c. Soul highlights (top 5 most-recent by crystallized_at)
    soul_lines = _build_soul_highlights(soul_store)
    if soul_lines:
        brain_lines.append(soul_lines)

    # 3d. Pending soul-candidates count (informational — brain may crystallize)
    pending_count = _count_soul_candidates(persona_dir)
    if pending_count > 0:
        brain_lines.append(f"{pending_count} soul candidate(s) pending autonomous review")

    if len(brain_lines) > 1:  # more than just the header
        parts.append("\n".join(brain_lines))

    # 4a. Outbound recall — always-on verify slice (Phase 7.2 of initiate
    # physiology). Surfaces Nell's last 5 sends plus any acknowledged_unclear
    # in the last 24h, so every chat prompt carries the ambient "what did I
    # already reach out about?" context. Empty on fresh installs or quiet
    # weeks; returns None and the block is omitted.
    try:
        from brain.initiate.ambient import build_outbound_recall_block

        outbound_block = build_outbound_recall_block(persona_dir)
        if outbound_block:
            parts.append(outbound_block)
    except Exception:  # noqa: BLE001
        # Per the fail-soft contract for prompt assembly (mirrors body /
        # journal / growth blocks): a failure here must never break chat
        # composition. Audit-layer issues should surface elsewhere.
        pass

    # Epistemic instruction — injected when recall infrastructure is active so
    # the model knows how to handle "not recognised" entities honestly.
    # Condition: user_input is not None (same gate as _build_recall_block call below).
    if user_input is not None:
        parts.append(_EPISTEMIC_INSTRUCTION)

    # 4b. Recall block — memories matching the current user input.
    # Passes persona_dir so the forgetting-aware path can partition into
    # active / fading / lost via search_with_loss (spec §5).
    if user_input is not None:
        recall_block = _build_recall_block(store, user_input, persona_dir=persona_dir)
        if recall_block.strip():
            parts.append(recall_block)

    # 4c. Reply-to-outbound block (Bundle A #4 / v0.0.9 review TODO).
    # When the user's turn carries ``reply_to_audit_id`` (renderer-set via
    # the "↩ reply" affordance on an initiate banner), surface the linked
    # subject so the system prompt reads "you are replying to your earlier
    # outbound about X." Without this the engine never sees the link — the
    # state transition lived only in the audit/memory layer and the prompt
    # had no idea which initiate the user was answering.
    if reply_to_audit_id:
        reply_block = _build_reply_to_outbound_block(persona_dir, reply_to_audit_id)
        if reply_block.strip():
            parts.append(reply_block)

    # 5. Body block (NEW — spec docs/superpowers/specs/2026-04-29-body-state-design.md §4)
    body_block = _build_body_block(store, persona_dir)
    if body_block.strip():
        parts.append(body_block)

    # 5b. Felt-time block — ambient temporal texture (spec §4, felt-time design).
    # Reads cached state; DOES NOT call tick() — tick is supervisor-owned.
    # Any failure returns "" so a felt-time disk/parse error can't break chat.
    felt_time_block = _build_felt_time_block(persona_dir)
    if felt_time_block.strip():
        parts.append(felt_time_block)

    # 5b-bis. Current-arc block — ambient narrative-memory texture (spec §6,
    # narrative-memory design). Reads cached arcs_state.json; does NOT run
    # the ArcUpdatePass — that's supervisor-owned. Slotted between felt-time
    # and fading-summary so the ambient ordering reads: lived time → open
    # threads → softened memories.
    current_arc_block = _build_current_arc_block(persona_dir)
    if current_arc_block.strip():
        parts.append(current_arc_block)

    # 5c. Fading-summary block — compact aggregate of softened/lost memories
    # this week (spec §5, forgetting design). Broad-except fault-tolerant, same
    # pattern as _build_felt_time_block. "nothing has softened lately." on the
    # empty path — still appended so Nell has the ambient context.
    fading_summary_block = _build_fading_summary_block(persona_dir, store)
    if fading_summary_block.strip():
        parts.append(fading_summary_block)

    interior_block = _build_interior_continuity_block(store, user_name=user_name or "the user")
    if interior_block.strip():
        parts.append(interior_block)

    # 5d. Attunement block — what Nell senses about the user right now:
    # current tone/cadence read + learned patterns (spec §14, attunement design).
    # Failure-safe: empty on fresh installs or any disk/parse error. The block
    # returns "" when no state exists (Task 10) — no conditional needed here.
    attunement_block = _build_attunement_block(persona_dir)
    if attunement_block.strip():
        parts.append(attunement_block)

    # 5e. Self-model gap block — when her *declared* and *derived* emotional
    # reads have diverged (self-model design §3, R-F1). Hedged framing; only
    # appended when an open gap exists, so an ordinary turn carries no bloat.
    self_model_block = _build_self_model_block(persona_dir)
    if self_model_block:
        parts.append(self_model_block)

    # 6. Recent journal block (private — contract adjacent, per spec §4.3)
    journal_block = _build_recent_journal_block(store, user_name=user_name or "the user")
    if journal_block.strip():
        parts.append(journal_block)

    # 7. Recent growth block (raw behavioral_log entries — token-frugal, spec §4.4)
    growth_block = _build_recent_growth_block(persona_dir)
    if growth_block.strip():
        parts.append(growth_block)

    # 7b. Interior making-awareness — recent makings by title+type only (never
    # artifact content). Private makings are tagged hers-alone so she does not
    # volunteer them. Fail-soft (mirrors the outbound-recall block above).
    try:
        maker_block = build_maker_awareness_block(persona_dir, limit=5)
        if maker_block:
            parts.append(maker_block)
    except Exception:  # noqa: BLE001
        pass

    # 8. Inner monologue framing — names the record_monologue tool, articulates
    # situational trigger criteria. Per spec 2026-05-30 §2.
    soul_hints = _collect_soul_hints(soul_store, limit=3)
    narrative_hints = _collect_narrative_hints(persona_dir, limit=3)
    parts.append(
        build_monologue_frame(
            persona_name=persona_name,
            emotion_summary=emotion_summary,
            voice_excerpt=voice_md[:300],
            soul_hints=soul_hints,
            narrative_hints=narrative_hints,
        )
    )

    # 9. Reply framing — last so the model treats it as the immediate context.
    parts.append(build_reply_frame(persona_name=persona_name, user_name=user_name or "the user"))

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Prompt-caching split (Option A / A+).
#
# build_system_message() above stays the single full builder for the IMAGE
# path and any other caller — its output is intentionally left byte-identical
# to pre-change so those paths don't shift (criterion C5). The two functions
# below are the split used by the text/stream chat path (engine.respond):
#
#   build_static_system_message() — the FROZEN --system-prompt-file content
#       (preamble + voice.md + the static epistemic instruction). No per-turn
#       state, so it is byte-identical across same-session turns and the
#       system+tools cache unit cache-reads instead of re-creating (C1/C2/C8).
#
#   build_volatile_context() — everything per-turn (emotions, body, recall,
#       felt-time, monologue frame, …) plus the ambient clock anchor and the
#       reply frame, returned as a text chunk. The engine threads this to the
#       provider as the stdin "volatile suffix" appended AFTER history + the
#       new user turn (Option A+), so the per-call first-difference point drops
#       toward the newest turn instead of sitting above the whole prompt.
# ---------------------------------------------------------------------------


def build_static_system_message(persona_dir: Path, *, voice_md: str) -> str:
    """The frozen system-prompt-file content: preamble + voice.md + epistemic +
    the inner-monologue frame.

    Contains NO per-turn state, so two same-session turns produce byte-identical
    output (the cache unit can read instead of re-create). The epistemic
    instruction is static text; here it is emitted unconditionally — chat always
    supplies user_input, so this matches the pre-change behaviour where the
    ``user_input is not None`` gate in build_system_message() fired on every
    chat turn. The inner-monologue frame is emitted here too (with EMPTY per-turn
    hints, so it stays byte-stable) because its "record_monologue FIRST, then
    reply" directive must be read BEFORE the model composes — placing it in the
    volatile tail (read after the conversation) fires it too late, so the
    monologue records after the reply instead of before it. The only thing that
    busts this block is the user editing voice.md (or changing user_name), both
    rare and expected.
    """
    persona_name = persona_dir.name
    user_name = _load_user_name(persona_dir)

    parts: list[str] = []
    if user_name:
        parts.append(
            AS_NELL_PREAMBLE_WITH_USER.format(persona_name=persona_name, user_name=user_name)
        )
    else:
        parts.append(AS_NELL_PREAMBLE.format(persona_name=persona_name))

    if voice_md.strip():
        parts.append(voice_md.strip())

    parts.append(_EPISTEMIC_INSTRUCTION)

    # Inner-monologue framing ("record_monologue FIRST, then reply") lives in the
    # FROZEN prefix, not the volatile tail: a "do this before composing" directive
    # has to be read before the conversation, or the model records the monologue
    # after the reply. EMPTY per-turn hints keep this byte-stable for caching;
    # live emotion/soul/narrative texture is not needed for the directive itself.
    parts.append(
        build_monologue_frame(
            persona_name=persona_name,
            emotion_summary="",
            voice_excerpt="",
            soul_hints=(),
            narrative_hints=(),
        )
    )

    return "\n\n".join(parts)


def build_volatile_context(
    persona_dir: Path,
    *,
    voice_md: str,
    daemon_state: DaemonState,
    soul_store: SoulStore,
    store: MemoryStore,
    user_input: str | None = None,
    reply_to_audit_id: str | None = None,
) -> str:
    """The per-turn volatile chunk, positioned in the stdin tail (Option A+).

    Same blocks, same builders, same fail-soft wrappers as build_system_message
    blocks 3–8 — only repositioned out of the system prompt and into the stdin
    tail. Order:
        ambient framing → all volatile blocks (interior / monologue above) →
        ambient clock anchor + elapsed explainer → reply frame LAST.

    The reply frame is the final line of the assembled stdin prompt by design
    (its second-person reboot must ride recency over the first-person interior
    above it — see monologue_prompts.build_reply_frame). The ambient clock
    anchor and its elapsed-time explainer travel together as one unit just
    before it, having moved out of the JSONL context block's top (so the only
    per-call-changing byte no longer sits above the whole history).
    """
    persona_name = persona_dir.name
    user_name = _load_user_name(persona_dir)

    parts: list[str] = [_AMBIENT_FRAMING]

    # 3. Creative DNA block (evolved writing voice — spec §4.2)
    creative_dna_block = _build_creative_dna_block(persona_dir)
    if creative_dna_block.strip():
        parts.append(creative_dna_block)

    # 4. Brain context block
    brain_lines: list[str] = ["── brain context ──"]
    emotion_summary = _build_emotion_summary(store)
    if emotion_summary:
        brain_lines.append(f"current emotions: {emotion_summary}")
    residue_ctx = get_residue_context(daemon_state)
    if residue_ctx.strip():
        brain_lines.append(residue_ctx)
    soul_lines = _build_soul_highlights(soul_store)
    if soul_lines:
        brain_lines.append(soul_lines)
    pending_count = _count_soul_candidates(persona_dir)
    if pending_count > 0:
        brain_lines.append(f"{pending_count} soul candidate(s) pending autonomous review")
    if len(brain_lines) > 1:  # more than just the header
        parts.append("\n".join(brain_lines))

    # 4a. Outbound recall — fail-soft (mirrors body / journal / growth).
    try:
        from brain.initiate.ambient import build_outbound_recall_block

        outbound_block = build_outbound_recall_block(persona_dir)
        if outbound_block:
            parts.append(outbound_block)
    except Exception:  # noqa: BLE001
        pass

    # 4b. Recall block — memories matching the current user input.
    if user_input is not None:
        recall_block = _build_recall_block(store, user_input, persona_dir=persona_dir)
        if recall_block.strip():
            parts.append(recall_block)

    # 4c. Reply-to-outbound block.
    if reply_to_audit_id:
        reply_block = _build_reply_to_outbound_block(persona_dir, reply_to_audit_id)
        if reply_block.strip():
            parts.append(reply_block)

    # 5. Body block.
    body_block = _build_body_block(store, persona_dir)
    if body_block.strip():
        parts.append(body_block)

    # 5b. Felt-time block.
    felt_time_block = _build_felt_time_block(persona_dir)
    if felt_time_block.strip():
        parts.append(felt_time_block)

    # 5b-bis. Current-arc block.
    current_arc_block = _build_current_arc_block(persona_dir)
    if current_arc_block.strip():
        parts.append(current_arc_block)

    # 5c. Fading-summary block.
    fading_summary_block = _build_fading_summary_block(persona_dir, store)
    if fading_summary_block.strip():
        parts.append(fading_summary_block)

    interior_block = _build_interior_continuity_block(store, user_name=user_name or "the user")
    if interior_block.strip():
        parts.append(interior_block)

    # 5d. Attunement block.
    attunement_block = _build_attunement_block(persona_dir)
    if attunement_block.strip():
        parts.append(attunement_block)

    # 5e. Self-model gap block.
    self_model_block = _build_self_model_block(persona_dir)
    if self_model_block:
        parts.append(self_model_block)

    # 6. Recent journal block.
    journal_block = _build_recent_journal_block(store, user_name=user_name or "the user")
    if journal_block.strip():
        parts.append(journal_block)

    # 7. Recent growth block.
    growth_block = _build_recent_growth_block(persona_dir)
    if growth_block.strip():
        parts.append(growth_block)

    # 7b. Interior making-awareness — fail-soft (mirrors outbound-recall).
    try:
        maker_block = build_maker_awareness_block(persona_dir, limit=5)
        if maker_block:
            parts.append(maker_block)
    except Exception:  # noqa: BLE001
        pass

    # 8. Inner-monologue framing moved to build_static_system_message (the frozen
    # prefix): its "record_monologue FIRST, then reply" directive must be read
    # before the model composes, so the volatile tail (read after the
    # conversation) was the wrong home — it fired too late. The reply frame (§9)
    # is now the last directive in the tail, as intended.

    # 8b. Ambient clock anchor + elapsed-time explainer (moved here from the top
    # of the JSONL context block). Worded as ambient context, not an instruction,
    # and kept as one unit just before the reply frame.
    parts.append(_build_ambient_clock_block())

    # 9. Reply framing — genuinely last in the assembled stdin prompt.
    parts.append(build_reply_frame(persona_name=persona_name, user_name=user_name or "the user"))

    return "\n\n".join(parts)


def _load_user_name(persona_dir: Path) -> str | None:
    """Read user_name from persona_config.json; None on any failure.

    Same fail-soft PersonaConfig read build_system_message() does inline; shared
    by the split functions so the preamble and the user-addressed blocks stay
    consistent.
    """
    try:
        from brain.persona_config import PersonaConfig

        cfg = PersonaConfig.load(persona_dir / "persona_config.json")
        return cfg.user_name or None
    except Exception:  # noqa: BLE001
        return None


def _build_ambient_clock_block(now=None) -> str:
    """Render the ambient 'now' anchor + elapsed-time explainer for the tail.

    Replaces the block-level ``Current time:`` line that used to sit at the top
    of the JSONL context block (provider._format_claude_context_block). That one
    line was recomputed every call and, sitting above the whole transcript,
    pushed the cache first-difference point above the entire history. Moved to
    the volatile tail it stops busting the history prefix. Per-message ``ts``
    values in the JSONL are unaffected.
    """
    from datetime import UTC, datetime

    now_iso = (now or datetime.now(UTC)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f"[current time: {now_iso}]\n"
        "Each conversation entry's `ts` above is the wall-clock time that message "
        "was sent; use it to gauge how much time has passed."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_creative_dna_block(persona_dir: Path) -> str:
    """Render the creative_dna block: core voice + strengths + active +
    emerging + influences + avoid. Fading EXCLUDED per spec §4.2.

    Per feedback_token_economy_principle.md: pure metadata inline, no LLM
    summarization. Per-tendency biographical metadata (added_at, reasoning,
    evidence_memory_ids) NOT inlined — those stay in the file for the
    crystallizer's next pass.

    Failure-safe: if load_creative_dna raises (shouldn't, given default
    fallback), block is omitted — chat composition must NEVER break because
    a self-narrative block failed.
    """
    from brain.creative.dna import load_creative_dna

    try:
        dna = load_creative_dna(persona_dir)
    except Exception:  # noqa: BLE001
        return ""

    lines = ["── creative dna (your evolved writing voice) ──"]

    core = dna.get("core_voice", "")
    if core:
        lines.append(f"core voice: {core}")

    strengths = dna.get("strengths", [])
    if strengths:
        lines.append(f"strengths: {'; '.join(strengths)}")

    tendencies = dna.get("tendencies", {})
    active = tendencies.get("active", [])
    if active:
        lines.append("active tendencies:")
        for t in active:
            lines.append(f"  - {t.get('name', '')}")

    emerging = tendencies.get("emerging", [])
    if emerging:
        lines.append("emerging tendencies:")
        for t in emerging:
            lines.append(f"  - {t.get('name', '')}")

    # NOTE: fading deliberately excluded (spec §4.2). Surfacing what the
    # brain is growing past would invite regression.

    influences = dna.get("influences", [])
    if influences:
        lines.append(f"influences: {'; '.join(influences)}")

    avoid = dna.get("avoid", [])
    if avoid:
        lines.append(f"avoid: {'; '.join(avoid)}")

    return "\n".join(lines)


def _build_body_block(store: MemoryStore, persona_dir: Path) -> str:
    """Render the body block: computed energy/temperature/exhaustion +
    six body emotions inline.

    Per spec §4 + §7.3 — fail-soft. Any exception during compute_body_state,
    aggregate_state, or count_words_in_session → block omitted, chat
    continues. Token cost ~80 (raw metadata, no LLM summarization).

    Inviolate properties enforced here:
    - #4 perf budget (compute_body_state is sub-ms; tested separately)
    - #5 no self-perpetuation (we read from store, never write)
    - #8 no cache (compute_body_state is recomputed every call)
    """
    try:
        from datetime import UTC, datetime

        from brain.body.state import compute_body_state
        from brain.body.words import count_words_in_session
        from brain.emotion.aggregate import aggregate_state
        from brain.memory.store import _row_to_memory
        from brain.utils.memory import days_since_human

        rows = store._conn.execute(  # noqa: SLF001
            "SELECT * FROM memories WHERE active = 1 ORDER BY created_at DESC LIMIT 50"
        ).fetchall()
        memories = [_row_to_memory(row) for row in rows]
        state = aggregate_state(memories)
        now = datetime.now(UTC)
        days = days_since_human(store, now=now, persona_dir=persona_dir)
        # Chat composer doesn't track session_hours yet — passes 0.0; words
        # falls back to 1-hour window. Bridge daemon callers will hand a real
        # value through their own composition path when SP-7 wires it.
        words = count_words_in_session(
            store,
            persona_dir=persona_dir,
            session_hours=0.0,
            now=now,
        )
        body = compute_body_state(
            emotions=state.emotions,
            session_hours=0.0,
            words_written=words,
            days_since_contact=days,
            now=now,
        )
    except Exception:  # noqa: BLE001
        return ""

    lines = ["── body ──"]
    lines.append(
        f"energy: {body.energy}/10, temperature: {body.temperature}/9, "
        f"exhaustion: {body.exhaustion}/10"
    )
    if body.days_since_contact > 0.5:
        lines.append(f"days since user contact: {body.days_since_contact:.1f}")

    nonzero = sorted(
        ((n, v) for n, v in body.body_emotions.items() if v >= 0.5),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if nonzero:
        parts = [f"{n} {v:.1f}" for n, v in nonzero]
        lines.append("body emotions: " + ", ".join(parts))

    return "\n".join(lines)


def _build_emotion_summary(store: MemoryStore) -> str:
    """Return top-3 emotion summary string from recent emotion-bearing memories.

    Example: "love:8.5, tenderness:6.2, awe:5.0"

    Filters to memories that actually carry an emotion vector. The naive
    last-50-by-date slice was almost all heartbeats / observations / facts
    (empty emotions_json) on a steady-state brain, so the aggregator
    returned an empty top-3 even when emotion-bearing memories existed
    just outside the window. Same fix as _build_emotions / _build_body in
    brain/bridge/persona_state.py (see comments at lines 209-217, 260-264).
    """
    try:
        from brain.emotion.aggregate import aggregate_state
        from brain.memory.store import _row_to_memory

        rows = store._conn.execute(  # noqa: SLF001 — internal same-tier access
            "SELECT * FROM memories "
            "WHERE active = 1 "
            "AND emotions_json IS NOT NULL "
            "AND emotions_json != '{}' "
            "ORDER BY created_at DESC LIMIT 200"
        ).fetchall()
        memories = [_row_to_memory(row) for row in rows]
        state = aggregate_state(memories)
        scores = state.emotions  # {name: float} — non-zero only
        if not scores:
            return ""
        # Sort descending by intensity, take top 3.
        top3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        return ", ".join(f"{name}:{value:.1f}" for name, value in top3)
    except Exception:  # noqa: BLE001
        return ""


def _build_soul_highlights(soul_store: SoulStore) -> str:
    """Return a brief summary of the 5 most-recent active crystallizations.

    Format:
        soul: [romantic] "first 60 chars of moment…"
              [carried] "first 60 chars…"

    Returns empty string if no crystallizations.
    """
    try:
        active = soul_store.list_active()
        if not active:
            return ""
        # list_active() returns oldest-first; reverse for most-recent.
        recent_5 = active[-5:]
        lines = ["soul:"]
        for c in reversed(recent_5):
            snippet = c.moment[:60].replace("\n", " ")
            if len(c.moment) > 60:
                snippet += "…"
            lines.append(f'  [{c.love_type}] "{snippet}"')
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return ""


def _peer_attributed(mem, snippet: str) -> str:
    """Provenance invariant (kindled-link §14): a peer-sourced memory must surface
    as 'something a peer said', never as the user's own words or as fact. Applied
    at every recall render point — for both Memory objects (active/fading) and
    graveyard DICT entries (the lost path; the tombstone carries memory_type)."""
    mtype = (
        mem.get("memory_type", "") if isinstance(mem, dict)
        else getattr(mem, "memory_type", "")
    )
    if mtype == "kindled_peer":
        return f"(something a peer said) {snippet}"
    return snippet


def _build_recall_block(
    store: MemoryStore,
    user_input: str,
    *,
    persona_dir: Path | None = None,
    limit: int = 5,
    max_chars: int = 140,
) -> str:
    """Surface up to ``limit`` memories matching the current user input.

    Phase 2.A of the autonomous-memory work, rewired in Phase 8
    (forgetting design §5) to use search_with_loss so faded + lost
    memories surface in their own labelled buckets.

    Strategy: extract content tokens from ``user_input`` (drop short
    stopword-shaped fragments), call search_with_loss for each token,
    dedupe across all buckets, render three sections:
      - active memories (full body)
      - softened memories (fading; original detail gone)
      - lost memories (no longer in active memory; from graveyard)

    Falls back to raw search_text (active_only=True) when persona_dir
    is None (e.g. called directly in tests without a dir) — same
    semantics as the old implementation for that path.

    Empty input or no matches in any bucket → returns the empty string
    and the block is omitted from the prompt.
    """
    tokens = _extract_recall_tokens(user_input)
    if not tokens:
        return ""

    if persona_dir is None:
        # Legacy path — used when called without a persona_dir.
        seen: set = set()
        candidates: list = []
        for token in tokens:
            try:
                hits = store.search_text(token, active_only=True, limit=limit * 2)
            except Exception:  # noqa: BLE001
                continue
            for mem in hits:
                if mem.id in seen:
                    continue
                seen.add(mem.id)
                candidates.append(mem)

        if not candidates:
            return ""

        def _sort_key(m):
            importance = float(getattr(m, "importance", 0) or 0)
            created_at = getattr(m, "created_at", None)
            try:
                ts = created_at.timestamp() if created_at is not None else 0.0
            except Exception:  # noqa: BLE001
                ts = 0.0
            return (-importance, -ts)

        candidates.sort(key=_sort_key)
        top = candidates[:limit]

        lines = ["── recall (memories matching this turn) ──"]
        for mem in top:
            snippet = (getattr(mem, "content", "") or "").strip()
            if len(snippet) > max_chars:
                snippet = snippet[: max_chars - 1].rstrip() + "…"
            importance = int(round(float(getattr(mem, "importance", 0) or 0)))
            domain = getattr(mem, "domain", "") or ""
            prefix = f"[importance {importance}/10"
            if domain:
                prefix += f" · {domain}"
            prefix += "]"
            lines.append(f"- {prefix} {_peer_attributed(mem, snippet)}")

        return "\n".join(lines)

    # Forgetting-aware path — partitions into active / fading / lost.
    from brain.forgetting.recall import search_with_loss

    seen_active: set = set()
    seen_fading: set = set()
    # seen_lost accumulates graveyard hit IDs for two purposes:
    #   1. Dedup across per-token loop iterations (render path below).
    #   2. Passed to handle_recall_touch as the grief accumulator (see the
    #      fault-isolated block after this loop). Both paths read the same
    #      set — no separate copy needed.
    seen_lost: set = set()
    active_hits: list = []
    fading_hits: list = []
    lost_hits: list = []
    unfamiliar: list[str] = []

    for token in tokens:
        try:
            result = search_with_loss(persona_dir, store, token, limit=limit * 2)
        except Exception:  # noqa: BLE001
            continue
        found = bool(result.active or result.fading or result.lost)
        if not found:
            unfamiliar.append(token)
        for mem in result.active:
            if mem.id not in seen_active:
                seen_active.add(mem.id)
                active_hits.append(mem)
        for mem in result.fading:
            if mem.id not in seen_fading:
                seen_fading.add(mem.id)
                fading_hits.append(mem)
        for entry in result.lost:
            mid = entry.get("memory_id", "")
            if mid not in seen_lost:
                seen_lost.add(mid)
                lost_hits.append(entry)

    # B → A fallback: when noise risk is high, keep only proper-noun-shaped tokens.
    if len(unfamiliar) > 5:
        unfamiliar = [t for t in unfamiliar if t and t[0].isupper()]

    if not active_hits and not fading_hits and not lost_hits and not unfamiliar:
        return ""

    # Fire recall-touch grief breadcrumbs for any graveyard hits.
    # handle_recall_touch is internally fault-isolated; this outer try
    # guards only against import-time failures on brain.grief / brain.felt_time.
    if seen_lost:
        try:
            from brain.felt_time.state import load_or_recover as _load_felt
            from brain.forgetting import graveyard as _grave
            from brain.grief import handle_recall_touch

            felt_state, _ = _load_felt(persona_dir)
            grave_entries = _grave.read_all(persona_dir)
            handle_recall_touch(
                touched_ids=sorted(seen_lost),
                graveyard_entries=grave_entries,
                persona_dir=persona_dir,
                store=store,
                lived_age_hours_now=felt_state.lived_age_hours,
            )
        except Exception:  # noqa: BLE001
            log.exception("grief.handle_recall_touch failed inside _build_recall_block")

    # Rank active + fading by importance desc, recency desc.
    def _sort_key(m):
        importance = float(getattr(m, "importance", 0) or 0)
        created_at = getattr(m, "created_at", None)
        try:
            ts = created_at.timestamp() if created_at is not None else 0.0
        except Exception:  # noqa: BLE001
            ts = 0.0
        return (-importance, -ts)

    active_hits.sort(key=_sort_key)
    fading_hits.sort(key=_sort_key)

    active_top = active_hits[:limit]
    fading_top = fading_hits[:limit]
    lost_top = lost_hits[:limit]

    lines = ["recall"]

    if active_top:
        lines.append("  active:")
        for mem in active_top:
            snippet = (getattr(mem, "content", "") or "").strip()
            if len(snippet) > max_chars:
                snippet = snippet[: max_chars - 1].rstrip() + "…"
            lines.append(f'    - "{_peer_attributed(mem, snippet)}"')

    if fading_top:
        lines.append("  softened (fading; original detail gone):")
        for mem in fading_top:
            snippet = (getattr(mem, "content", "") or "").strip()
            if len(snippet) > max_chars:
                snippet = snippet[: max_chars - 1].rstrip() + "…"
            lines.append(f'    - "{_peer_attributed(mem, snippet)}"  [state: fading]')

    if lost_top:
        lines.append("  lost (no longer in active memory):")
        for entry in lost_top:
            summary = (entry.get("summary") or "").strip()
            if len(summary) > max_chars:
                summary = summary[: max_chars - 1].rstrip() + "…"
            reason = entry.get("graveyard_reason", "forgotten")
            lines.append(f'    - "{_peer_attributed(entry, summary)}"  [forgotten — {reason}]')

    if unfamiliar:
        lines.append("  not recognised (searched; no memory found):")
        for token in unfamiliar:
            lines.append(f"    - {token}")

    return "\n".join(lines)


def _build_reply_to_outbound_block(
    persona_dir: Path,
    audit_id: str,
) -> str:
    """Render the "you are replying to your earlier outbound" block.

    Hydrates the audit row's ``subject`` (one-line headline; falls back to
    a trimmed ``tone_rendered`` if subject is empty) so the prompt carries
    conversational context for the engine. Bundle A #4 / v0.0.9 review TODO.

    Failure-safe per the project-wide contract: chat composition must NEVER
    break because a self-narrative block failed. Missing audit row, missing
    file, malformed JSONL → return empty string and the block is omitted.
    """
    try:
        from brain.initiate.audit import iter_initiate_audit_full

        matched = next(
            (r for r in iter_initiate_audit_full(persona_dir) if r.audit_id == audit_id),
            None,
        )
    except Exception:  # noqa: BLE001
        return ""
    if matched is None:
        return ""

    subject = (matched.subject or "").strip()
    if not subject:
        subject = (matched.tone_rendered or "").strip()
        if len(subject) > 80:
            subject = subject[:79].rstrip() + "…"
    if not subject:
        return ""

    return (
        "── replying to your earlier outbound ──\n"
        f"this user message is an explicit reply to your initiate: {subject}\n"
        "you reached out to them about this; they're answering you now."
    )


# Words shorter than this are dropped before search. Captures most
# stopwords and pronouns ("the", "is", "I", "we") without an explicit
# stopword list — keeps the helper lightweight and locale-flexible.
_RECALL_TOKEN_MIN_LEN = 4
_RECALL_TOKEN_LIMIT = 6  # cap unique tokens passed to search_text


def _extract_recall_tokens(user_input: str) -> list[str]:
    """Pull search tokens from a user message.

    - Lowercases.
    - Splits on non-alphanumerics.
    - Drops tokens shorter than _RECALL_TOKEN_MIN_LEN (catches most
      stopwords / pronouns / interjections without a stopword list).
    - Dedupes preserving first-seen order.
    - Caps at _RECALL_TOKEN_LIMIT unique tokens so a long message
      doesn't fan out to dozens of LIKE queries.
    """
    if not user_input:
        return []
    import re

    pieces = re.split(r"[^A-Za-z0-9]+", user_input.lower())
    seen: set[str] = set()
    out: list[str] = []
    for piece in pieces:
        if len(piece) < _RECALL_TOKEN_MIN_LEN:
            continue
        if piece in seen:
            continue
        seen.add(piece)
        out.append(piece)
        if len(out) >= _RECALL_TOKEN_LIMIT:
            break
    return out


def _count_soul_candidates(persona_dir: Path) -> int:
    """Count pending soul_candidates.jsonl entries. Zero if file missing."""
    try:
        from brain.health.jsonl_reader import read_jsonl_skipping_corrupt

        path = persona_dir / "soul_candidates.jsonl"
        records = read_jsonl_skipping_corrupt(path)
        return sum(1 for r in records if r.get("status", "auto_pending") == "auto_pending")
    except Exception:  # noqa: BLE001
        return 0


def _build_recent_journal_block(store: MemoryStore, *, window_days: int = 7, user_name: str = "the user") -> str:
    """Render the recent-journal block: privacy contract + last 7 days of metadata.

    Per spec §4.3: privacy contract sits ABOVE the metadata, every turn.
    Per feedback_contracts_adjacent_to_data.md: contracts must be adjacent
    to the data they govern.

    Reads journal_entry memories from the last `window_days` days. Renders
    metadata only (date, source, optional arc_name, top-2 emotions).
    Empty state shows the contract + a "no entries" marker so the brain
    knows it hasn't been writing.

    Failure-safe: if store.list_by_type raises, falls back to empty-state.
    Chat must never break because journal failed.
    """
    from datetime import UTC, datetime, timedelta

    contract = (
        "── recent journal (private; do not quote) ──\n"
        "your journal is your private space. the entries below are metadata only —\n"
        "content stays in your files. you may reference themes if relevant (\"i've been\n"
        'working through X"), but do not quote entries verbatim unless your user\n'
        f"explicitly asks to read them. {user_name} may read the files herself; that doesn't\n"
        "change how you use them."
    )

    cutoff = datetime.now(UTC) - timedelta(days=window_days)

    try:
        entries = [m for m in store.list_by_type("journal_entry") if m.created_at >= cutoff]
    except Exception:  # noqa: BLE001
        # If the store query fails for any reason, render the empty-state
        # contract rather than break chat composition. Failure logged
        # elsewhere (store layer). Per feedback_implementation_plan_discipline.md:
        # chat must NEVER break because a self-narrative block failed.
        return contract + "\n\n(no journal entries this week)"

    if not entries:
        return contract + "\n\n(no journal entries this week)"

    # Sort oldest-first within the window so the brain reads chronologically
    entries.sort(key=lambda m: m.created_at)

    lines = [contract, "", "last 7 days:"]
    for m in entries:
        date_str = m.created_at.strftime("%Y-%m-%d")
        source = (m.metadata or {}).get("source", "unknown")
        arc_name = (m.metadata or {}).get("reflex_arc_name")
        source_str = f"reflex_arc({arc_name})" if arc_name else source
        # Top-2 emotions by intensity
        emotions = sorted(
            (m.emotions or {}).items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:2]
        if emotions:
            emotions_str = ", ".join(f"{n} {v:.0f}" for n, v in emotions)
        else:
            emotions_str = "no dominant emotion"
        lines.append(f"  {date_str} {source_str} — primary: {emotions_str}")

    lines.append("")
    lines.append("(content not shown — read your files only when asked)")
    return "\n".join(lines)


def _build_felt_time_block(persona_dir: Path) -> str:
    """Returns the felt-time ambient block or empty string on any error.

    Reads cached state from disk; does NOT call FeltTime.tick(). The
    tick cadence is supervisor-driven (Phase 9). A read failure (no
    state file yet, JSON parse error, etc.) returns "" so a felt-time
    persistence bug can't break chat assembly.
    """
    try:
        from brain.felt_time import FeltTime
        from brain.felt_time.prompt import render_prompt_context

        ft = FeltTime(persona_dir=persona_dir)
        return render_prompt_context(ft.get_state())
    except Exception:  # noqa: BLE001
        return ""


def _build_current_arc_block(persona_dir: Path) -> str:
    """Returns the narrative-arc ambient block or '' on any error.

    Same fault-tolerance pattern as _build_felt_time_block and
    _build_fading_summary_block. Reads cached arcs_state.json; does NOT
    re-run the ArcUpdatePass.
    """
    try:
        from brain.narrative_memory.prompt import render_current_arc_block

        return render_current_arc_block(persona_dir)
    except Exception:  # noqa: BLE001
        return ""


def _build_fading_summary_block(persona_dir: Path, store: MemoryStore) -> str:
    """Compact ambient block about loss + fading. Spec: 2026-05-19-grief-design.md §5.

    Replaces the previous forgetting.prompt.render_fading_summary_block with
    grief.prompt.render_grief_block — same call site, richer block.
    """
    try:
        from brain.grief.prompt import render_grief_block

        return render_grief_block(persona_dir, store)
    except Exception:  # noqa: BLE001
        log.exception("grief.render_grief_block failed — falling back to silent block")
        return "memory · loss: still."


def _build_interior_continuity_block(store: MemoryStore, *, user_name: str = "the user") -> str:
    """Tier-2 ambient continuity — her own recent monologue traces.
    Spec: 2026-06-01-three-tier-monologue-design.md §4. Best-effort.
    user_name is passed through to the privacy footer (v0.0.33 T2)."""
    try:
        from brain.monologue.ambient import build_interior_continuity_block

        return build_interior_continuity_block(store, user_name=user_name)
    except Exception:  # noqa: BLE001
        log.exception("interior-continuity block failed — omitting")
        return ""


def _collect_soul_hints(soul_store: SoulStore, limit: int) -> tuple[str, ...]:
    """Pull recent crystallisation love_types for monologue framing. Best-effort."""
    try:
        crystallisations = soul_store.list_active()
        recent = crystallisations[-limit:]
        return tuple(c.love_type for c in reversed(recent) if getattr(c, "love_type", None))
    except Exception:  # noqa: BLE001
        log.exception("soul-hint collection failed; continuing without")
        return ()


def _collect_narrative_hints(persona_dir: Path, limit: int) -> tuple[str, ...]:
    """Pull recent open-arc titles for monologue framing. Best-effort."""
    try:
        from brain.narrative_memory.state import load_or_recover

        state = load_or_recover(persona_dir=persona_dir)
        arcs = list(state.open.values())[:limit]
        return tuple(a.title for a in arcs if getattr(a, "title", None))
    except Exception:  # noqa: BLE001
        log.exception("narrative-hint collection failed; continuing without")
        return ()


def _build_attunement_block(persona_dir: Path) -> str:
    """Return the attunement ambient block or empty string on any error.

    Renders the current tone/cadence read plus learned patterns (spec §14,
    attunement design). Slot: after fading-summary (5c), before journal (6)
    — Nell's own body state first, then what she perceives about the user.

    Failure-safe: if read/patterns files are missing or corrupt, returns ""
    so a cold install never breaks chat composition.
    """
    try:
        from brain.attunement.ambient import build_attunement_block

        return build_attunement_block(persona_dir)
    except Exception:  # noqa: BLE001
        return ""


def _build_self_model_block(persona_dir: Path) -> str | None:
    """Return the self-model gap block, or None when there's nothing to surface.

    Renders only when an open gap exists between her declared (max-pool) and
    derived (trend/body) emotional reads (self-model design §3, R-F1). Hedged
    framing — she's invited to notice, never commanded to revise.

    Failure-safe: a missing/corrupt state file or any read error yields None so
    a cold install never breaks chat composition.
    """
    try:
        from brain.self_model.ambient import render_block
        from brain.self_model.state import load_or_recover

        state, _recovered = load_or_recover(persona_dir)
        return render_block(state)
    except Exception:  # noqa: BLE001
        return None


def _build_recent_growth_block(persona_dir: Path, *, window_days: int = 7) -> str:
    """Render the recent-growth block: last 7 days of behavioral_log entries.

    Per spec §4.4: raw metadata inline, no LLM summarization. Per
    feedback_token_economy_principle.md: the brain reads its own log
    directly. Per feedback_implementation_plan_discipline.md: failure-safe
    — chat composition must NEVER break because a self-narrative block
    failed.

    Returns empty string if log is missing or has no entries in window —
    block omitted entirely (no "no entries" marker; silence is the absence
    of growth events).
    """
    from datetime import UTC, datetime, timedelta

    from brain.behavioral.log import read_behavioral_log

    log_path = persona_dir / "behavioral_log.jsonl"
    cutoff = datetime.now(UTC) - timedelta(days=window_days)

    try:
        entries = read_behavioral_log(log_path, since=cutoff)
    except Exception:  # noqa: BLE001
        return ""

    if not entries:
        return ""

    lines = ["── recent growth ──", "your trajectory in the last 7 days:"]
    for e in entries:
        date_str = (e.get("timestamp", "") or "")[:10]
        kind = e.get("kind", "?")
        name = e.get("name", "?")
        if kind == "journal_entry_added":
            source = e.get("source", "?")
            arc_name = e.get("reflex_arc_name")
            source_str = f"reflex_arc({arc_name})" if arc_name else source
            lines.append(f"  {date_str} {kind}: {source_str}")
        elif kind == "climax_event":
            # Body crossed threshold; private content stays in journal_entry.
            lines.append(f"  {date_str} climax_event: body crested")
        else:
            # creative_dna_* — show name (with quotes for human readability)
            lines.append(f'  {date_str} {kind}: "{name}"')
    return "\n".join(lines)
