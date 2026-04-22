"""Temporal decay for emotions.

Each emotion in the vocabulary has a half-life (or None = identity-level,
doesn't decay). apply_decay() walks a state and applies exponential decay
to each emotion based on the elapsed time since it was last touched.

Design per spec Section 10.1 (per-emotion decay curves).
"""

from __future__ import annotations

from brain.emotion.state import EmotionalState
from brain.emotion.vocabulary import get as _get_emotion

# Below this intensity, the emotion is considered noise and removed entirely.
# Prevents residue accumulation from very-old events.
_NOISE_FLOOR: float = 0.01

_SECONDS_PER_DAY: float = 24 * 3600


def apply_decay(state: EmotionalState, elapsed_seconds: float) -> None:
    """Decay every known emotion in the state by its half-life.

    Emotions with half_life=None (identity-level) are untouched.
    Emotions not in the vocabulary are also untouched (stale-data guard —
    see EmotionalState.from_dict's permissive contract).
    Emotions decayed below the noise floor are removed.

    Mutates state in place; recomputes dominant after.
    """
    if elapsed_seconds <= 0:
        return

    to_remove: list[str] = []
    elapsed_days = elapsed_seconds / _SECONDS_PER_DAY

    for name, intensity in state.emotions.items():
        emotion = _get_emotion(name)
        if emotion is None:
            # Stale or persona-specific emotion no longer registered — leave it.
            continue
        if emotion.decay_half_life_days is None:
            # Identity-level — no decay.
            continue

        # Exponential decay: new = old * (1/2)^(elapsed / half_life)
        ratio = 0.5 ** (elapsed_days / emotion.decay_half_life_days)
        new_intensity = intensity * ratio

        if new_intensity < _NOISE_FLOOR:
            to_remove.append(name)
        else:
            state.emotions[name] = new_intensity

    for name in to_remove:
        del state.emotions[name]

    state._recompute_dominant()
