"""Identity reconstruction — rebuild vocabulary from memory references.

When all backups of emotion_vocabulary.json are corrupt and reset would
otherwise fire, the brain re-learns its own vocabulary from how it has
been operating: scan memories.db for distinct emotion names and make a
persona-extension entry for each name that is not a core feeling. Core
feelings are never written to her file (#801 section 8).
"""

from __future__ import annotations

from brain.emotion import vocabulary as _vocabulary
from brain.memory.store import MemoryStore

PLACEHOLDER_DESCRIPTION = "(reconstructed from memory)"
# Persona-typical persistence (14d half-life matches standard persona-extension
# entries).  The original 1.0 fast-decay default caused the stub-flood bug
# (2026-06-10 investigation): reconstructed/healed entries decayed 14x faster
# than proper persona entries and were erased from memories in ~9-13 days.
PLACEHOLDER_DECAY_DAYS = 14.0


def reconstruct_vocabulary_from_memories(store: MemoryStore) -> dict:
    """Build emotion_vocabulary.json content: the persona extensions found in memories (no core feeling)."""
    baseline_names: set[str] = {e.name for e in _vocabulary._BASELINE}
    seen_names: set[str] = set()
    for mem in store.list_active(limit=None):
        # Names only: a stored emotion whose value is not finite still has a name the vocabulary needs.
        for name in MemoryStore.stored_emotion_names(mem):
            tidied = _vocabulary.gate_name(name)  # #174 tidied, #801 and gated: a refused name makes no entry
            if tidied is not None:
                seen_names.add(tidied)

    entries: list[dict] = []
    # A core feeling is never written to her file (#801 section 8): the framework holds its definition.
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
