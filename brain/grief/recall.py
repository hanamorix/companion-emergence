"""recall.py — recall-touch detection + intensity per spec §3, §6.

handle_recall_touch is wired into:
  - chat.prompt._build_recall_block (per user turn)
  - tools.dispatch._dispatch_recall_forgotten (Nell's own tool calls)
  - narrative_memory.__init__ membership refresh (internal arc touch)

Spec: docs/superpowers/specs/2026-05-19-grief-design.md §3 + §6
"""

from __future__ import annotations

from brain.grief import policy


def _clamp(x: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, x))


def compute_touch_intensity(
    *,
    grave_emotion_max: float,
    salience_at_drop: float,
    lived_days_since_loss: float,
) -> float:
    """Recall-touch grief intensity per spec §3.

    intensity = clamp(grave_emotion_max × salience_at_drop × 5.0 × recency_factor, 0, 10)

    recency_factor = 0.5 ** ((d / T) ** 2)

    where T = RECENCY_LIVED_DAYS_HALF_LIFE (default 14). This is a Gaussian
    half-life: at exactly T lived-days since loss, recency_factor = 0.5 and
    intensity is halved. Beyond T the decay accelerates — a 60-day-old loss
    is nearly imperceptible even with high salience inputs.
    """
    d = max(lived_days_since_loss, 0.0)
    half_life = policy.RECENCY_LIVED_DAYS_HALF_LIFE
    recency = 0.5 ** ((d / half_life) ** 2)
    raw = grave_emotion_max * salience_at_drop * policy.RECALL_TOUCH_SCALE * recency
    return _clamp(raw)
