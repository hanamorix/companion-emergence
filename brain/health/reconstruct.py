"""Identity reconstruction — rebuild vocabulary from memory references.

When all backups of emotion_vocabulary.json are corrupt and reset would
otherwise fire, the brain re-learns its own vocabulary from how it has
been operating: scan memories.db for distinct emotion names, register
framework baseline + persona-extensions for any unknown names found.
"""

from __future__ import annotations

from brain.emotion import vocabulary as _vocabulary
from brain.memory.store import MemoryStore

PLACEHOLDER_DESCRIPTION = "(reconstructed from memory)"
PLACEHOLDER_DECAY_DAYS = 1.0  # conservative — fast decay until user re-tunes


def reconstruct_vocabulary_from_memories(store: MemoryStore) -> dict:
    """Build emotion_vocabulary.json content: baseline + extensions found in memories."""
    baseline_names: set[str] = {e.name for e in _vocabulary._BASELINE}
    seen_names: set[str] = set()
    for mem in store.search_text("", active_only=True, limit=None):
        for name in mem.emotions:
            seen_names.add(name)

    entries: list[dict] = []
    # Framework baseline always — these are immutable identity.
    for e in _vocabulary._BASELINE:
        entries.append(
            {
                "name": e.name,
                "description": e.description,
                "category": e.category,
                "decay_half_life_days": e.decay_half_life_days,
                "intensity_clamp": e.intensity_clamp,
            }
        )

    # Persona extensions: any emotion name in memories not in baseline.
    for name in sorted(seen_names - baseline_names):
        entries.append(
            {
                "name": name,
                "description": PLACEHOLDER_DESCRIPTION,
                "category": "persona_extension",
                "decay_half_life_days": PLACEHOLDER_DECAY_DAYS,
                "intensity_clamp": 10.0,
            }
        )

    return {"version": 1, "emotions": entries}
