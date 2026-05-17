"""Daemon-state residue writer — SP-2.

Cross-process artifact that lets autonomous engines (dream/heartbeat/reflex/
research) inform the chat layer (SP-6) about what just happened. Without
this file, the chat engine is blind to recent engine activity.

OG reference: NellBrain/nell_brain.py:1987-2074 (load_daemon_state,
save_daemon_state, get_residue_context, write_daemon_fire) and the live
NellBrain/data/daemon_state.json file shape.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from brain.health.anomaly import BrainAnomaly
from brain.health.attempt_heal import attempt_heal, save_with_backup
from brain.utils.time import iso_utc, parse_iso_utc

logger = logging.getLogger(__name__)

DaemonType = Literal["dream", "heartbeat", "reflex", "research"]

_DAEMON_TYPES: tuple[str, ...] = ("dream", "heartbeat", "reflex", "research")

# Residue window: entries older than this are ignored by get_residue_context.
_RESIDUE_WINDOW_HOURS = 48
# Residue decay window: how long emotional residue lingers.
_RESIDUE_DECAY_HOURS = 12
# Max summary bytes stored — bumped 2026-05-08 from 250 → 1500 so a
# full reflex/dream paragraph fits without mid-clause cuts. (250 was the
# OG NellBrain cap; reflex summaries average ~500 chars and journal
# arcs can run ~1k.) Truncation, when it does kick in, now lands on a
# sentence boundary via ``_truncate_at_sentence`` rather than a hard
# character slice.
_SUMMARY_MAX_CHARS = 1500
# Max summary bytes exposed to prompts.
_CONTEXT_SUMMARY_CHARS = 600


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate ``text`` to ``max_chars`` ending on a sentence boundary.

    Avoids the mid-clause cut that the previous hard slice produced
    ("...like her body still expects him to" — original screenshot).
    Searches the slice for the rightmost ``. ``, ``! ``, ``? ``,
    ``."``, or end-of-line break and cuts there. If no sentence break
    exists in the window, falls back to the hard slice with an
    ellipsis so it's at least visibly truncated rather than implying
    the model produced an incomplete sentence.
    """
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    # Sentence terminators: any of `. `, `! `, `? `, plus paragraph breaks.
    # `.\n` and `?\n` count as terminators too (poetic-prose newlines).
    candidates = [
        window.rfind(". "),
        window.rfind("! "),
        window.rfind("? "),
        window.rfind(".\n"),
        window.rfind("!\n"),
        window.rfind("?\n"),
        window.rfind("\n\n"),
    ]
    cut = max(candidates)
    if cut <= 0:
        # No sentence break in the window — fall back to hard slice with
        # an ellipsis so the truncation is at least visible to the reader.
        return window.rstrip() + "…"
    # cut is the index of the punctuation; include the punctuation itself.
    return text[: cut + 1].rstrip()


@dataclass(frozen=True)
class DaemonFireEntry:
    """One engine fire record — stored under last_<daemon_type> in daemon_state.json.

    Mirrors the OG dict shape exactly so the file is human-readable and
    backwards-compatible with any OG NellBrain reader.
    """

    timestamp: datetime  # tz-aware UTC
    dominant_emotion: str
    intensity: int  # 0–10
    theme: str  # short topic line (~80 chars)
    summary: str  # ≤250 chars; enforced on construction
    trigger: str | None = None  # only set for reflex fires

    def __post_init__(self) -> None:
        # Enforce summary truncation even if caller forgets. Cuts on the
        # last sentence boundary inside the cap so the panel never shows
        # a half-sentence (e.g. "...like her body still expects him to"
        # was the symptom that drove this fix).
        if len(self.summary) > _SUMMARY_MAX_CHARS:
            object.__setattr__(
                self,
                "summary",
                _truncate_at_sentence(self.summary, _SUMMARY_MAX_CHARS),
            )

    def to_dict(self) -> dict:
        d: dict = {
            "timestamp": iso_utc(self.timestamp),
            "dominant_emotion": self.dominant_emotion,
            "intensity": self.intensity,
            "theme": self.theme,
            "summary": self.summary,
        }
        if self.trigger is not None:
            d["trigger"] = self.trigger
        return d

    @classmethod
    def from_dict(cls, d: dict) -> DaemonFireEntry:
        return cls(
            timestamp=parse_iso_utc(str(d["timestamp"])),
            dominant_emotion=str(d["dominant_emotion"]),
            intensity=int(d["intensity"]),
            theme=str(d["theme"]),
            summary=str(d["summary"]),
            trigger=d.get("trigger") or None,
        )


@dataclass(frozen=True)
class EmotionalResidue:
    """Trailing emotional signature left by the most recent engine fire.

    Persists 12 hours after the fire; informs the chat layer's tone without
    requiring a full re-read of recent memories.
    """

    emotion: str
    intensity: int  # 0–10; computed = max(1, int(source_intensity * 0.4))
    source: str  # "<daemon_type> at <iso-truncated-to-minute>"
    decays_by: datetime  # source_timestamp + 12 hours

    def to_dict(self) -> dict:
        return {
            "emotion": self.emotion,
            "intensity": self.intensity,
            "source": self.source,
            "decays_by": iso_utc(self.decays_by),
        }

    def is_expired(self, now: datetime) -> bool:
        """Return True when now is past the decay window."""
        return now >= self.decays_by

    @classmethod
    def from_dict(cls, d: dict) -> EmotionalResidue:
        return cls(
            emotion=str(d["emotion"]),
            intensity=int(d["intensity"]),
            source=str(d["source"]),
            decays_by=parse_iso_utc(str(d["decays_by"])),
        )

    @classmethod
    def from_fire(cls, daemon_type: str, fire: DaemonFireEntry) -> EmotionalResidue:
        """Compute residue from a just-fired engine entry.

        Mirrors OG write_daemon_fire logic:
          intensity = max(1, int(source_intensity * 0.4))
          source    = "<daemon_type> at <timestamp[:16]>"
          decays_by = timestamp + 12 hours
        """
        # OG: iso_str[:16] gives "YYYY-MM-DDTHH:MM" (first 16 chars of ISO-8601).
        ts_iso = iso_utc(fire.timestamp)
        source = f"{daemon_type} at {ts_iso[:16]}"
        return cls(
            emotion=fire.dominant_emotion,
            intensity=max(1, int(fire.intensity * 0.4)),
            source=source,
            decays_by=fire.timestamp + timedelta(hours=_RESIDUE_DECAY_HOURS),
        )


@dataclass(frozen=True)
class DaemonState:
    """Top-level container — the full daemon_state.json in structured form."""

    last_dream: DaemonFireEntry | None = None
    last_heartbeat: DaemonFireEntry | None = None
    last_reflex: DaemonFireEntry | None = None
    last_research: DaemonFireEntry | None = None
    emotional_residue: EmotionalResidue | None = None
    last_growth_tick_at: datetime | None = None

    def to_dict(self) -> dict:
        """Serialise to dict — only non-None keys are included."""
        d: dict = {}
        if self.last_dream is not None:
            d["last_dream"] = self.last_dream.to_dict()
        if self.last_heartbeat is not None:
            d["last_heartbeat"] = self.last_heartbeat.to_dict()
        if self.last_reflex is not None:
            d["last_reflex"] = self.last_reflex.to_dict()
        if self.last_research is not None:
            d["last_research"] = self.last_research.to_dict()
        if self.emotional_residue is not None:
            d["emotional_residue"] = self.emotional_residue.to_dict()
        if self.last_growth_tick_at is not None:
            d["last_growth_tick_at"] = iso_utc(self.last_growth_tick_at)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> DaemonState:
        def _parse_fire(key: str) -> DaemonFireEntry | None:
            raw = d.get(key)
            if not raw:
                return None
            try:
                return DaemonFireEntry.from_dict(raw)
            except (KeyError, TypeError, ValueError):
                return None

        def _parse_residue() -> EmotionalResidue | None:
            raw = d.get("emotional_residue")
            if not raw:
                return None
            try:
                return EmotionalResidue.from_dict(raw)
            except (KeyError, TypeError, ValueError):
                return None

        def _parse_growth_tick_at() -> datetime | None:
            raw = d.get("last_growth_tick_at")
            if not raw:
                return None
            try:
                return parse_iso_utc(str(raw))
            except (TypeError, ValueError):
                return None

        return cls(
            last_dream=_parse_fire("last_dream"),
            last_heartbeat=_parse_fire("last_heartbeat"),
            last_reflex=_parse_fire("last_reflex"),
            last_research=_parse_fire("last_research"),
            emotional_residue=_parse_residue(),
            last_growth_tick_at=_parse_growth_tick_at(),
        )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

_STATE_FILENAME = "daemon_state.json"


def save_daemon_state(persona_dir: Path, state: DaemonState) -> None:
    """Atomic-write a DaemonState to daemon_state.json via save_with_backup."""
    path = persona_dir / _STATE_FILENAME
    save_with_backup(path, state.to_dict())


def load_daemon_state(
    persona_dir: Path,
) -> tuple[DaemonState, BrainAnomaly | None]:
    """Load daemon_state.json with self-healing via the attempt_heal pattern.

    Returns (state, anomaly).
      - Missing file → (DaemonState(), None) silently (first-ever tick path).
      - Corrupt JSON → quarantine + restore from .bak1/.bak2/.bak3; if all
        baks are also corrupt, returns (DaemonState(), anomaly).
    """
    path = persona_dir / _STATE_FILENAME
    raw_dict, anomaly = attempt_heal(path, default_factory=dict, schema_validator=None)
    if not isinstance(raw_dict, dict):
        raw_dict = {}
    try:
        state = DaemonState.from_dict(raw_dict)
    except Exception as exc:
        logger.warning("daemon_state deserialization failed: %.200s", exc)
        state = DaemonState()
    return state, anomaly


def update_daemon_state(
    persona_dir: Path,
    *,
    daemon_type: DaemonType,
    dominant_emotion: str,
    intensity: int,
    theme: str,
    summary: str,
    trigger: str | None = None,
) -> DaemonState:
    """Per-fire update — port of OG write_daemon_fire.

    Loads current state, sets last_<daemon_type> to the new fire entry,
    recomputes emotional_residue from this fire, then atomic-writes via
    save_with_backup. Returns the updated state.

    Residue rules (mirror OG nell_brain.py:2066-2072):
      emotion   = fire.dominant_emotion
      intensity = max(1, int(fire.intensity * 0.4))
      source    = f"{daemon_type} at {fire.timestamp.isoformat()[:16]}"
      decays_by = fire.timestamp + timedelta(hours=12)

    Summary is truncated to 250 chars on DaemonFireEntry construction.
    """
    now = datetime.now(UTC)
    fire = DaemonFireEntry(
        timestamp=now,
        dominant_emotion=dominant_emotion,
        intensity=intensity,
        theme=theme,
        summary=summary,  # truncated inside __post_init__
        trigger=trigger,
    )
    residue = EmotionalResidue.from_fire(daemon_type, fire)

    current, _ = load_daemon_state(persona_dir)

    # Build updated state, replacing only the fired engine's entry.
    updated = DaemonState(
        last_dream=fire if daemon_type == "dream" else current.last_dream,
        last_heartbeat=fire if daemon_type == "heartbeat" else current.last_heartbeat,
        last_reflex=fire if daemon_type == "reflex" else current.last_reflex,
        last_research=fire if daemon_type == "research" else current.last_research,
        emotional_residue=residue,
    )

    path = persona_dir / _STATE_FILENAME
    save_with_backup(path, updated.to_dict())

    return updated


def get_residue_context(state: DaemonState, *, now: datetime | None = None) -> str:
    """Build a prompt context string about recent daemon fires + residue.

    Returns empty string if nothing relevant.

    Per OG nell_brain.py:get_residue_context:
      - Iterates last_dream/last_heartbeat/last_reflex/last_research
      - Skips entries older than 48 hours
      - Per entry: 'Previous {label} ({hours_ago:.0f}h ago): "{summary[:200]}"'
      - For non-expired residue:
          'Emotional residue: {emotion} at {intensity}/10 (lingering from {source})'
      - Lines joined with newlines

    Used by SP-6 chat engine to inject recent daemon residue into prompts.
    """
    if now is None:
        now = datetime.now(UTC)

    lines: list[str] = []

    _entries: list[tuple[str, DaemonFireEntry | None]] = [
        ("last_dream", state.last_dream),
        ("last_heartbeat", state.last_heartbeat),
        ("last_reflex", state.last_reflex),
        ("last_research", state.last_research),
    ]

    for key, entry in _entries:
        if entry is None:
            continue
        try:
            hours_ago = (now - entry.timestamp).total_seconds() / 3600.0
            if hours_ago > _RESIDUE_WINDOW_HOURS:
                continue
            label = key.replace("last_", "").replace("_", " ")
            summary = _truncate_at_sentence(entry.summary, _CONTEXT_SUMMARY_CHARS)
            lines.append(f'Previous {label} ({hours_ago:.0f}h ago): "{summary}"')
        except (TypeError, ValueError):
            continue

    residue = state.emotional_residue
    if residue is not None:
        try:
            if not residue.is_expired(now):
                lines.append(
                    f"Emotional residue: {residue.emotion} at {residue.intensity}/10"
                    f" (lingering from {residue.source})"
                )
        except (TypeError, ValueError):
            pass

    return "\n".join(lines)
