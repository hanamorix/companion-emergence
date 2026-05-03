"""SP-4 ingest types — ExtractedItem + IngestReport dataclasses.

These are the boundary types for the 8-stage conversation ingest pipeline.
ExtractedItem represents one candidate memory pulled from a transcript.
IngestReport summarises what happened when a session was closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VALID_LABELS = frozenset({"observation", "feeling", "decision", "question", "fact", "note"})


@dataclass
class ExtractedItem:
    """One candidate memory extracted from a conversation transcript.

    Attributes:
        text:        The durable memory text, already stripped.
        label:       One of VALID_LABELS. Coerced to "observation" on
                     normalization if unknown.
        importance:  1-10 integer signal from the extraction LLM.
                     Clamped on normalization; defaults to 5.
    """

    text: str
    label: str = "observation"
    importance: int = 5

    def normalize(self) -> ExtractedItem:
        """Coerce label, clamp importance, strip text whitespace.

        Returns self so callers can chain: item.normalize().
        """
        # Boundary validation: coerce unknown labels to the safe default.
        if self.label not in VALID_LABELS:
            self.label = "observation"
        # Clamp importance to [1, 10]; handle non-int values gracefully.
        try:
            self.importance = max(1, min(10, int(self.importance)))
        except (TypeError, ValueError):
            self.importance = 5
        self.text = (self.text or "").strip()
        return self


@dataclass
class IngestReport:
    """Summary of what happened when one conversation session was closed.

    Attributes:
        session_id:      The closed session's identifier.
        extracted:       Total candidate items returned by the LLM.
        committed:       Items that passed dedupe + were written to the store.
        deduped:         Items skipped because they matched an existing memory.
        soul_candidates: Items queued to soul_candidates.jsonl (importance >= threshold).
        errors:          Items that failed the commit step.
        memory_ids:      IDs of newly created memories (committed items only).
    """

    session_id: str
    extracted: int = 0
    committed: int = 0
    deduped: int = 0
    soul_candidates: int = 0
    soul_queue_errors: int = 0
    errors: int = 0
    memory_ids: list[str] = field(default_factory=list)
