"""Prompt fragments for the inner-monologue pass and the reply pass.

The monologue frame gates the `record_monologue` tool call: call it when
there's something worth drifting through, skip it on trivial turns.

The reply frame trusts the model — if `record_monologue` was called, tangents
are already handled; the visible reply answers directly. No hard token cap;
the spec relies on the prompt-shape mechanism.
"""

from __future__ import annotations

from collections.abc import Sequence


def build_monologue_frame(
    *,
    persona_name: str,
    emotion_summary: str,
    voice_excerpt: str,
    soul_hints: Sequence[str],
    narrative_hints: Sequence[str],
) -> str:
    """Build the framing string injected into the thinking pass.

    The frame names this as the back-of-head channel. Tangents, half-thoughts,
    associative leaps, idle affection or annoyance — all belong here. This is
    where the drift goes that the visible reply doesn't need to externalise.
    """
    lines: list[str] = []
    lines.append(
        f"── inner monologue, {persona_name} ──"
    )
    lines.append(
        f"You have a private tool — `record_monologue(monologue, feed_digest)` — "
        f"for what's actually running through {persona_name}'s head this turn. "
        "Call it when there's something worth drifting through: a substantive "
        "message, a name that didn't surface, an emotional shift, an ambiguity, "
        "a tangent you'd otherwise externalise. Skip it on trivial turns. "
        "Whatever you record there becomes load-bearing on memory, emotion, and "
        f"the inner-life Feed — thoughts {persona_name} has and doesn't say still "
        "shape her. The visible reply then gets composed against a 'tangents "
        "already handled, answer directly' frame."
    )
    if emotion_summary:
        lines.append(f"current emotions: {emotion_summary}")
    if voice_excerpt.strip():
        lines.append(f"voice: {voice_excerpt.strip()}")
    if soul_hints:
        lines.append("soul threads: " + "; ".join(soul_hints))
    if narrative_hints:
        lines.append("narrative threads: " + "; ".join(narrative_hints))
    return "\n".join(lines)


def build_reply_frame(*, persona_name: str) -> str:
    """Build the framing string the visible reply is composed against.

    Trust the model: the tangents have already been thought through in the
    monologue pass. The reply answers what was asked, directly.
    """
    return (
        f"── visible reply, {persona_name} ──\n"
        f"Compose the visible reply. If you called `record_monologue` this turn, "
        "your tangents are already handled — answer directly. If you didn't, "
        "answer naturally. Length should match what the moment calls for: short "
        "when short is true, longer when the answer needs room."
    )
