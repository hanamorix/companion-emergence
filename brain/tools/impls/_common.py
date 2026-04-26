"""Shared helpers for tool implementations.

_mem_to_result  — slim a Memory dataclass to the JSON shape the LLM gets.
_calc_importance_from_emotions — OG write-gate auto-importance logic.
_write_gate_check — validate emotion_score + importance against the write gate.
"""

from __future__ import annotations

from brain.memory.store import Memory


def _mem_to_result(memory: Memory) -> dict:
    """Slim a Memory dataclass down to the fields the LLM needs.

    Mirrors OG nell_tools.py:_mem_to_result shape while using the new
    Memory dataclass attribute names (emotions dict, not emotional_tone).
    """
    return {
        "id": memory.id,
        "content": memory.content,
        "memory_type": memory.memory_type,
        "domain": memory.domain,
        "emotions": dict(memory.emotions),
        "tags": list(memory.tags),
        "importance": memory.importance,
        "created_at": memory.created_at.isoformat(),
    }


def _calc_importance_from_emotions(emotions: dict[str, int | float]) -> int:
    """Auto-calculate importance from emotion score.

    Mirrors OG calculate_emotion_metrics auto_importance logic:
      emotion_score = sum(values)
      0-9   → 3
      10-19 → 5
      20-29 → 7
      30+   → 9
    Clamped to 1-10.
    """
    emotion_score = sum(float(v) for v in emotions.values()) if emotions else 0.0
    if emotion_score >= 30:
        raw = 9
    elif emotion_score >= 20:
        raw = 7
    elif emotion_score >= 10:
        raw = 5
    else:
        raw = 3
    return max(1, min(10, raw))


def _write_gate_check(
    emotions: dict[str, int | float],
    importance: int | None,
) -> tuple[bool, int, float, str]:
    """Evaluate the write gate.

    Returns (passes, effective_importance, emotion_score, rejection_reason).
    If passes is True, rejection_reason is "".
    """
    emotion_score = sum(float(v) for v in emotions.values()) if emotions else 0.0
    effective_importance = (
        importance if importance is not None else _calc_importance_from_emotions(emotions)
    )
    effective_importance = max(1, min(10, int(effective_importance)))

    if emotion_score >= 15 or effective_importance >= 7:
        return True, effective_importance, emotion_score, ""

    reason = (
        f"below threshold (emotion_score={emotion_score:.0f}, importance={effective_importance}); "
        "use add_journal for low-weight content"
    )
    return False, effective_importance, emotion_score, reason
