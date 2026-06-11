"""Render the attunement block for Nell's ambient prompt context.

Renders the current read (all five categories) plus learned patterns by
maturity, and — once mature (forming/known) patterns exist and aren't on
cooldown — the addressability directive ("Don't force it"). Activated v0.0.29.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from brain.attunement.schemas import ADDRESS_COOLDOWN_HOURS
from brain.attunement.store import read_current_read, read_learned_patterns

_ATTUNEMENT_RENDER_CAP = 8


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
    from brain.persona_config import PersonaConfig
    from brain.pronouns import resolve

    try:
        pr = resolve(PersonaConfig.load(persona_dir / "persona_config.json").user_pronouns)
    except Exception:  # noqa: BLE001 — prompt assembly must never break on config
        pr = resolve(None)

    read = read_current_read(persona_dir)
    patterns = read_learned_patterns(persona_dir)

    lines: list[str] = []

    if read is not None and read.tone_label != "unknown":
        lines.append(f"# What you sense about {pr.object} right now")
        lines.append(
            f"{pr.cap(pr.subject)} {pr.v('sounds', 'sound')} "
            f"{read.tone_label} — {read.tone_justification}"
        )
        lines.append(
            f"{pr.cap(pr.possessive)} cadence is "
            f"{read.cadence_label} — {read.cadence_justification}"
        )
        lines.append(f"Mood feels {_valence_phrase(read.mood_valence, read.mood_intensity)}.")
        if read.predicted_arc_shape:
            lines.append(f"Where this seems to be heading: {read.predicted_arc_shape}")

    surfaceable = [
        p for p in patterns
        if p.maturity in {"forming", "known"}
        and not _is_addressed_recently(p.last_addressed_at)
    ]
    _maturity_rank = {"known": 0, "forming": 1}
    surfaceable.sort(key=lambda p: (_maturity_rank.get(p.maturity, 9), -p.evidence_count))
    surfaceable = surfaceable[:_ATTUNEMENT_RENDER_CAP]
    if surfaceable:
        if lines:
            lines.append("")
        lines.append(
            f"# What you've come to know about {pr.object} "
            f"(your private read — when you speak to {pr.object}, that's 'you')"
        )
        for p in surfaceable:
            if p.maturity == "known":
                lines.append(f"- {p.description}")
            else:
                lines.append(
                    f"- {pr.cap(pr.subject)} {pr.v('seems', 'seem')} to {p.description.lower()}"
                )
        lines.append("")
        lines.append("If a pattern feels load-bearing for this turn, you can name it. Don't force it.")

    return "\n".join(lines)
