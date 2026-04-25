"""Vocabulary crystallizer — Phase 2a stub.

Phase 2a returns []. Phase 2b will:
- Cluster memories by emotional configuration vectors
- Detect clusters that recur but don't have a name in current_vocabulary_names
- Detect clusters that align with specific relational dynamics
- Apply quality gates (novelty, evidence threshold, score threshold)
- Apply rate limit (max 1 proposal per tick)
- Use LLM-mediated naming (via brain.bridge.provider.LLMProvider)
"""

from __future__ import annotations

from brain.growth.proposal import EmotionProposal
from brain.memory.store import MemoryStore


def crystallize_vocabulary(
    store: MemoryStore,
    *,
    current_vocabulary_names: set[str],
) -> list[EmotionProposal]:
    """Mine memory + relational dynamics for novel emotional configurations.

    Phase 2a behavior: returns [] always. The signature accepts arguments
    Phase 2b will use; ignored in 2a.
    """
    # Phase 2b will read store + current_vocabulary_names.
    _ = store
    _ = current_vocabulary_names
    return []
