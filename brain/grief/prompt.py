"""prompt.py — ambient grief block per spec §5.

Replaces forgetting.prompt.render_fading_summary_block at the chat
prompt-builder call site. Names up to 2 specific losses (one memory,
one arc), with deterministic lived-time stamps + heavy/medium/light
weight labels.

Spec: docs/superpowers/specs/2026-05-19-grief-design.md §5
"""

from __future__ import annotations

from brain.grief import policy

_LOST_WINDOW_DAYS = 7


def weight_bucket(*, emotion_max_normalised: float) -> str:
    """Bucket emotion_at_ingest_max into heavy / medium / light per spec §5.

    Thresholds use 0-10 scale via WEIGHT_HEAVY (7.0) and WEIGHT_MEDIUM (3.0)
    re-scaled to 0-1 for normalised inputs.
    """
    scaled = emotion_max_normalised * 10.0
    if scaled >= policy.WEIGHT_HEAVY:
        return "heavy"
    if scaled >= policy.WEIGHT_MEDIUM:
        return "medium"
    return "light"
