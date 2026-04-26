"""boot tool implementation — full session-boot composition."""

from __future__ import annotations

from pathlib import Path

from brain.engines.daemon_state import get_residue_context, load_daemon_state
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.tools.impls.get_body_state import get_body_state
from brain.tools.impls.get_emotional_state import get_emotional_state
from brain.tools.impls.get_personality import get_personality
from brain.tools.impls.get_soul import get_soul


def boot(
    *,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    persona_dir: Path,
) -> dict:
    """Full session-boot composition.

    Assembles emotional state, personality, soul, body state, and daemon
    residue into a single dict plus a context_prose paragraph the LLM can
    read at session start to anchor itself.

    Returns
    -------
    dict with keys:
        emotional_state  — get_emotional_state result
        personality      — get_personality result (stub for now)
        soul             — get_soul result (stub for now)
        body_state       — get_body_state result (stub for now)
        daemon_residue   — str from get_residue_context
        context_prose    — 2-4 sentence prose for LLM session anchor
    """
    ctx = {"store": store, "hebbian": hebbian, "persona_dir": persona_dir}

    emotional_state = get_emotional_state(**ctx)
    personality = get_personality(**ctx)
    soul = get_soul(**ctx)
    body_state = get_body_state(**ctx)

    daemon_state, _ = load_daemon_state(persona_dir)
    daemon_residue = get_residue_context(daemon_state)

    # Build context prose
    dominant = emotional_state.get("dominant") or "unknown"
    top_3 = [entry["emotion"] for entry in (emotional_state.get("top_5") or [])[:3]]
    top_3_str = ", ".join(top_3) if top_3 else "none recorded"
    days_away = body_state.get("days_since_contact", 0.0) or 0.0

    residue_summary = ""
    if daemon_residue:
        # Grab just the first line for the prose sentence — full residue is in
        # the daemon_residue field for the LLM to read in detail.
        residue_summary = daemon_residue.splitlines()[0]

    prose_parts = [
        f"I'm in a state of {dominant}, feeling: {top_3_str}.",
    ]
    if residue_summary:
        prose_parts.append(f"The residue from my recent engines still hums: '{residue_summary}'.")
    if days_away > 1:
        prose_parts.append(f"Hana's been away {days_away:.1f} days.")
    prose_parts.append("Reading the room before I speak.")

    context_prose = " ".join(prose_parts)

    return {
        "emotional_state": emotional_state,
        "personality": personality,
        "soul": soul,
        "body_state": body_state,
        "daemon_residue": daemon_residue,
        "context_prose": context_prose,
    }
