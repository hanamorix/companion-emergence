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
      3. Brain context block:
         - Current emotion state summary (top-3 emotions by intensity)
         - Daemon residue (from get_residue_context(daemon_state))
         - Soul highlights (top 5 most-recent crystallizations: love_type + 60-char snippet)
         - Pending soul-candidates count (informational)

    Returns the final system message string.
    """
    persona_name = persona_dir.name
    parts: list[str] = []

    # 1. Preamble
    parts.append(AS_NELL_PREAMBLE.format(persona_name=persona_name))

    # 2. Voice
    if voice_md.strip():
        parts.append(voice_md.strip())

    # 3. Brain context block
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

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
