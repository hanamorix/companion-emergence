"""Aggregate a current EmotionalState from a list of memories.

Reflex uses this to evaluate arc triggers: what is the persona's
current emotional state, synthesized across recent memories.

Strategy: max-pool per emotion. The strongest signal across the
input memories wins — matches how OG reflex_engine read peaks,
not averages, for threshold evaluation.

After max-pooling, _apply_climax_reset is called: when aggregated
`climax >= 7` it dampens arousal + desire and raises comfort_seeking
+ rest_need, modeling the body's natural release cycle. Per spec
§2.2 of docs/superpowers/specs/2026-04-29-body-state-design.md.
"""

from __future__ import annotations

from collections.abc import Iterable

from brain.emotion.state import EmotionalState
from brain.emotion.vocabulary import get as _get_emotion
from brain.memory.store import Memory

_CLIMAX_THRESHOLD = 7.0
_AROUSAL_DAMPEN = 0.2
_AROUSAL_FLOOR = 0.5
_DESIRE_DAMPEN = 0.6
_COMFORT_BUMP = 2.0
_REST_BUMP = 2.0
_INTENSITY_CLAMP = 10.0


def aggregate_state(memories: Iterable[Memory]) -> EmotionalState:
    """Return an EmotionalState that is the per-emotion max across inputs.

    Unknown emotions (not in the registered vocabulary) are silently
    skipped — a persona's old memories may contain retired emotion
    names that no longer validate via EmotionalState.set.

    After max-pooling, applies the climax reset hook (§2.2). All
    callers (chat, reflex, body-state) get the post-reset state.
    """
    pooled: dict[str, float] = {}
    for mem in memories:
        for name, intensity in mem.emotions.items():
            try:
                value = float(intensity)
            except (TypeError, ValueError):
                continue
            if value <= 0.0:
                continue
            if _get_emotion(name) is None:
                continue
            if value > pooled.get(name, 0.0):
                pooled[name] = value

    state = EmotionalState()
    for name, value in pooled.items():
        try:
            state.set(name, value)
        except (KeyError, ValueError):
            # clamp violation or validation failure — skip
            continue
    return _apply_climax_reset(state)


def _apply_climax_reset(state: EmotionalState) -> EmotionalState:
    """Apply post-climax reset to the AGGREGATED state.

    Reset is a state-time computation, not storage-time. Memory store
    keeps original memory weights; current felt state reflects the
    body's natural release cycle. Returns a NEW EmotionalState; never
    mutates input.

    Idempotent: applying twice to same state produces same first-
    application result (reset values are absolute factors, not deltas
    on already-reset values).

    Reconciliation 2026-04-30 (Option A): retargeted from spec's
    `physical_arousal` to existing `arousal` baseline emotion.
    """
    climax = state.emotions.get("climax", 0.0)
    if climax < _CLIMAX_THRESHOLD:
        return state.copy()

    out = state.copy()

    if "arousal" in out.emotions:
        new_arousal = max(_AROUSAL_FLOOR, out.emotions["arousal"] * _AROUSAL_DAMPEN)
        out.set("arousal", new_arousal)
    if "desire" in out.emotions:
        out.set("desire", out.emotions["desire"] * _DESIRE_DAMPEN)

    new_comfort = min(_INTENSITY_CLAMP, out.emotions.get("comfort_seeking", 0.0) + _COMFORT_BUMP)
    out.set("comfort_seeking", new_comfort)

    new_rest = min(_INTENSITY_CLAMP, out.emotions.get("rest_need", 0.0) + _REST_BUMP)
    out.set("rest_need", new_rest)

    # Idempotency gate: after firing, climax itself is set to 0 — the release
    # cycle has completed. A second application sees climax=0 < threshold,
    # returns a copy unchanged. Biologically accurate: the crest has passed.
    out.set("climax", 0.0)

    return out
