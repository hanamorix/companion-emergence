"""breadcrumb.py — intensity formulas + content phrases + write path.

Three intensity formulas (§3):
  - compute_drop_intensity     — hard-deleted memory
  - compute_arc_close_intensity — closed narrative arc
  - compute_touch_intensity     — recall-touch (lives in recall.py)

Content phrases (§4) — all deterministic, zero LLM:
  - drop:         "the memory of {first_6_words_of_summary} is gone"
  - arc-close:    "the arc '{arc_name}' has closed"
  - recall-touch: "reached for {first_6_words_of_summary} — gone"

Spec: docs/superpowers/specs/2026-05-19-grief-design.md §3 + §4
"""

from __future__ import annotations

from typing import Literal

from brain.grief import policy
from brain.memory.store import Memory, MemoryStore


def _clamp(x: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, x))


def compute_drop_intensity(
    *, emotion_at_ingest_max: float
) -> float:
    """Drop-time grief intensity per spec §3.

    Args:
        emotion_at_ingest_max: max emotion value on the dropped memory,
            normalised to [0, 1] (raw 0-10 emotion / 10).

    Returns:
        Grief intensity in [0, 10].

    Note:
        salience_at_drop is NOT a factor — at drop time it is by
        definition near zero (LOST_THRESHOLD = 0.10), and multiplying
        would silently suppress grief. See spec §3.
    """
    raw = emotion_at_ingest_max * policy.DROP_SCALE
    return _clamp(raw)


def compute_arc_close_intensity(*, arc_max_member_emotion: float) -> float:
    """Arc-close grief intensity per spec §3.

    Args:
        arc_max_member_emotion: max emotion value across arc members at
            close time, normalised to [0, 1]. See spec §3 for the
            graveyard-proxy fallback that callers apply for members
            already lost before the arc closed.

    Returns:
        Grief intensity in [0, 10].
    """
    raw = arc_max_member_emotion * policy.ARC_CLOSE_SCALE
    return _clamp(raw)


def first_n_words(text: str, n: int) -> str:
    """First n whitespace-split words, joined by single space. Empty -> ''."""
    if not text:
        return ""
    words = text.split()
    if not words:
        return ""
    return " ".join(words[:n])


def drop_phrase(summary: str, *, lived_days_ago: float | None = None) -> str:
    """Deterministic drop-time content phrase per spec §4.

    Distinct wording from fading ('gone soft') — drop is lost, not softened.
    """
    head = first_n_words(summary, 6)
    if head:
        return f"the memory of {head} is gone"
    if lived_days_ago is None:
        return "a memory is gone"
    return f"a memory from {int(lived_days_ago)} lived-days ago is gone"


def recall_touch_phrase(summary: str) -> str:
    """Deterministic recall-touch content phrase per spec §4."""
    head = first_n_words(summary, 6)
    if not head:
        return "reached for a lost memory — gone"
    return f"reached for {head} — gone"


def arc_close_phrase(arc_name: str) -> str:
    """Deterministic arc-close content phrase per spec §4."""
    return f"the arc '{arc_name}' has closed"


SubtypeLiteral = Literal["drop", "arc_close", "recall_touch"]
ReferentTypeLiteral = Literal["memory", "arc"]


def write_breadcrumb(
    *,
    store: MemoryStore,
    intensity: float,
    subtype: SubtypeLiteral,
    referent_type: ReferentTypeLiteral,
    referent_id: str,
    content: str,
    residue_emotion: tuple[str, float] | None,
    triggering_arc_id: str | None = None,
) -> str:
    """Write a grief breadcrumb memory per spec §4. Returns the new memory id.

    No threshold gating here — callers apply the threshold before calling.
    """
    emotions: dict[str, float] = {"memory_grief": float(intensity)}
    if residue_emotion is not None:
        name, value = residue_emotion
        # Skip if name is the primary grief key (avoid double-counting) or value <= 0.
        if name != "memory_grief" and value > 0.0:
            emotions[name] = float(value) * policy.RESIDUE_FACTOR

    metadata: dict[str, object] = {
        "grief_referent_type": referent_type,
        "grief_referent_id": referent_id,
        "grief_subtype": subtype,
        "triggering_arc_id": triggering_arc_id,
    }
    memory = Memory.create_new(
        content=content,
        memory_type="grief_event",
        domain="grief",
        emotions=emotions,
        metadata=metadata,
    )
    return store.create(memory)
