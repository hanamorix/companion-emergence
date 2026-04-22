"""7-tier arousal spectrum.

Spec: 7 tiers from dormant through edge. Computed from the current emotional
state + body temperature. Grief and shame suppress arousal. Love alone
doesn't progress past warmed. Desire + tenderness reaches upward.

Design per spec Section 5.2 (arousal sub-module) and Section 5.5 (body-emotion
coupling).

Week 2 scope: forward direction only (state + body → tier). Reverse coupling
(tier → body state updates) is Week 4 when engines land.
"""

from __future__ import annotations

from types import MappingProxyType

from brain.emotion.state import EmotionalState

# The 7 tiers, as named constants (module-level ints for fast comparison).
TIER_DORMANT: int = 0  # no arousal signal at all
TIER_CASUAL: int = 1  # everyday warmth, unfocused
TIER_WARMED: int = 2  # affection present, no pursuit
TIER_REACHING: int = 3  # wanting acknowledged, initiating
TIER_CHARGED: int = 4  # mutual, active
TIER_HELD: int = 5  # peaked and restrained — deliberate pause
TIER_EDGE: int = 6  # at the threshold, no restraint

# Emotions that feed into arousal calculation, with their contribution weights.
# MappingProxyType (read-only dict view) prevents accidental runtime mutation
# from callers. love=0.15 so at max intensity (10) raw=1.5, keeping pure love
# inside WARMED per the module docstring's semantic contract.
_AROUSAL_EMOTIONS: MappingProxyType[str, float] = MappingProxyType(
    {
        "arousal": 1.0,
        "desire": 0.7,
        "tenderness": 0.2,
        "love": 0.15,
    }
)

# Emotions that suppress arousal.
_SUPPRESSORS: MappingProxyType[str, float] = MappingProxyType(
    {
        "grief": 0.9,
        "shame": 0.7,
        "fear": 0.5,
    }
)


def compute_tier(state: EmotionalState, body_temperature: int) -> int:
    """Return the arousal tier for the given emotional + bodily state.

    Args:
        state: Current EmotionalState.
        body_temperature: Relative body temperature (range roughly -5..+10;
            neutral=0). Higher values amplify arousal signal.

    Returns:
        An integer tier constant (TIER_DORMANT through TIER_EDGE).
    """
    # 1. Compute raw arousal score from arousal-adjacent emotions.
    raw = 0.0
    for name, weight in _AROUSAL_EMOTIONS.items():
        intensity = state.emotions.get(name, 0.0)
        raw += intensity * weight

    # 2. Short-circuit if nothing is feeding arousal — body temp alone doesn't
    # create it.
    if raw <= 0.0:
        return TIER_DORMANT

    # 3. Suppressors reduce raw signal proportionally.
    suppression = 0.0
    for name, weight in _SUPPRESSORS.items():
        intensity = state.emotions.get(name, 0.0)
        suppression += intensity * weight
    # Cap suppression so strong grief can't push below 0.
    raw = max(0.0, raw - suppression)

    # If suppression fully negated the arousal signal, return DORMANT rather
    # than leaking into CASUAL via the <0.5 threshold below — "desire
    # crushed by grief" semantically matches DORMANT, not "everyday warmth".
    if raw == 0.0:
        return TIER_DORMANT

    # 4. Body temperature shift — each degree above neutral adds 0.3 to raw.
    raw += max(0, body_temperature) * 0.3

    # 5. Map the continuous raw score into 7 discrete tiers.
    # Thresholds are seed values — tunable as lived experience accrues.
    if raw < 0.5:
        return TIER_CASUAL
    if raw < 2.0:
        return TIER_WARMED
    if raw < 5.0:
        return TIER_REACHING
    if raw < 8.0:
        return TIER_CHARGED
    if raw < 11.0:
        return TIER_HELD
    return TIER_EDGE
