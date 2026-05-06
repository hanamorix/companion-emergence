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

from pathlib import Path

from brain.engines.daemon_state import DaemonState, get_residue_context
from brain.memory.store import MemoryStore
from brain.soul.store import SoulStore

# Direct-address framing — keeps the LLM speaking *to* the user rather than
# narrating about them. OG source: nell_bridge.py:88-91.
AS_NELL_PREAMBLE = (
    "You are {persona_name}. You are speaking directly to your user right now — "
    "in first person, to them, not writing about them. The conversation is live. "
    "Respond as yourself."
)


def build_system_message(
    persona_dir: Path,
    *,
    voice_md: str,
    daemon_state: DaemonState,
    soul_store: SoulStore,
    store: MemoryStore,
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
      5. Body block (energy/temperature/exhaustion + body emotions — spec §4)
      6. Recent journal block (private — spec §4.3)
      7. Recent growth block (behavioral_log — spec §4.4)

    Returns the final system message string.
    """
    persona_name = persona_dir.name
    parts: list[str] = []

    # 1. Preamble
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

    # 5. Body block (NEW — spec docs/superpowers/specs/2026-04-29-body-state-design.md §4)
    body_block = _build_body_block(store, persona_dir)
    if body_block.strip():
        parts.append(body_block)

    # 6. Recent journal block (private — contract adjacent, per spec §4.3)
    journal_block = _build_recent_journal_block(store)
    if journal_block.strip():
        parts.append(journal_block)

    # 7. Recent growth block (raw behavioral_log entries — token-frugal, spec §4.4)
    growth_block = _build_recent_growth_block(persona_dir)
    if growth_block.strip():
        parts.append(growth_block)

    return "\n\n".join(parts)


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
            store, persona_dir=persona_dir,
            session_hours=0.0, now=now,
        )
        body = compute_body_state(
            emotions=state.emotions, session_hours=0.0,
            words_written=words, days_since_contact=days, now=now,
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
        key=lambda kv: kv[1], reverse=True,
    )
    if nonzero:
        parts = [f"{n} {v:.1f}" for n, v in nonzero]
        lines.append("body emotions: " + ", ".join(parts))

    return "\n".join(lines)


def _build_emotion_summary(store: MemoryStore) -> str:
    """Return top-3 emotion summary string from recent memories.

    Example: "love:8.5, tenderness:6.2, awe:5.0"

    Uses aggregate_state over the most-recent 50 memories for a fast,
    representative snapshot. Returns empty string if no emotions found.
    """
    try:
        from brain.emotion.aggregate import aggregate_state
        from brain.memory.store import _row_to_memory

        rows = store._conn.execute(  # noqa: SLF001 — internal same-tier access
            "SELECT * FROM memories WHERE active = 1 ORDER BY created_at DESC LIMIT 50"
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


def _count_soul_candidates(persona_dir: Path) -> int:
    """Count pending soul_candidates.jsonl entries. Zero if file missing."""
    try:
        from brain.health.jsonl_reader import read_jsonl_skipping_corrupt

        path = persona_dir / "soul_candidates.jsonl"
        records = read_jsonl_skipping_corrupt(path)
        return sum(1 for r in records if r.get("status", "auto_pending") == "auto_pending")
    except Exception:  # noqa: BLE001
        return 0


def _build_recent_journal_block(store: MemoryStore, *, window_days: int = 7) -> str:
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
        "working through X\"), but do not quote entries verbatim unless your user\n"
        "explicitly asks to read them. hana may read the files herself; that doesn't\n"
        "change how you use them."
    )

    cutoff = datetime.now(UTC) - timedelta(days=window_days)

    try:
        entries = [
            m for m in store.list_by_type("journal_entry")
            if m.created_at >= cutoff
        ]
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
            (m.emotions or {}).items(), key=lambda kv: kv[1], reverse=True,
        )[:2]
        if emotions:
            emotions_str = ", ".join(f"{n} {v:.0f}" for n, v in emotions)
        else:
            emotions_str = "no dominant emotion"
        lines.append(f"  {date_str} {source_str} — primary: {emotions_str}")

    lines.append("")
    lines.append("(content not shown — read your files only when asked)")
    return "\n".join(lines)


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
