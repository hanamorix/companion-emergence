"""Shared emotion helpers used by multiple engines."""

from __future__ import annotations

from collections.abc import Mapping


def format_emotion_summary(emotions: Mapping[str, float]) -> str:
    """Return the top-5 emotions formatted as '- name: X.X/10' lines.

    Empty input returns an empty string. Used by reflex + research
    engines for LLM prompt context.
    """
    top = sorted(emotions.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return "\n".join(f"- {name}: {value:.1f}/10" for name, value in top)
