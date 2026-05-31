"""Render the attunement block for Nell's ambient prompt context.

alpha.1 form: tone + cadence current read + learned tone/cadence patterns.
The addressability directive ("Don't force it") is reserved for v0.0.28
final.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from brain.attunement.schemas import ADDRESS_COOLDOWN_HOURS
from brain.attunement.store import read_current_read, read_learned_patterns


def _valence_phrase(valence: float, intensity: float) -> str:
    """Map valence + intensity into a short natural-language phrase."""
    if intensity < 0.2:
        return "even"
    if valence > 0.4:
        return "buoyant" if intensity > 0.6 else "soft and bright"
    if valence < -0.4:
        return "heavy" if intensity > 0.6 else "subdued"
    return "raw and bright" if intensity > 0.6 else "steady"


def _is_addressed_recently(last_addressed_at: str | None) -> bool:
    if last_addressed_at is None:
        return False
    try:
        addressed = datetime.fromisoformat(last_addressed_at.replace("Z", "+00:00"))
        if addressed.tzinfo is None:
            addressed = addressed.replace(tzinfo=UTC)
    except ValueError:
        return False
    now = datetime.now(UTC)
    return (now - addressed).total_seconds() < ADDRESS_COOLDOWN_HOURS * 3600


def build_attunement_block(persona_dir: Path) -> str:
    """Return the rendered attunement block, or empty string when no state exists."""
    read = read_current_read(persona_dir)
    patterns = read_learned_patterns(persona_dir)

    lines: list[str] = []

    if read is not None and read.tone_label != "unknown":
        lines.append("# What you sense about her right now")
        lines.append(f"She sounds {read.tone_label} — {read.tone_justification}")
        lines.append(f"Her cadence is {read.cadence_label} — {read.cadence_justification}")
        lines.append(f"Mood feels {_valence_phrase(read.mood_valence, read.mood_intensity)}.")
        if read.predicted_arc_shape:
            lines.append(f"Where this seems to be heading: {read.predicted_arc_shape}")

    surfaceable = [
        p for p in patterns
        if p.maturity in {"forming", "known"}
        and not _is_addressed_recently(p.last_addressed_at)
    ]
    if surfaceable:
        if lines:
            lines.append("")
        lines.append("# What you've come to know about her")
        for p in surfaceable:
            if p.maturity == "known":
                lines.append(f"- {p.description}")
            else:
                lines.append(f"- You seem to {p.description.lower()}")

    return "\n".join(lines)
