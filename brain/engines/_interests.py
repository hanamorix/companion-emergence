"""Interest dataclass + InterestSet persistence.

Shared by brain.engines.research AND the interest-ingestion hook in
brain.engines.heartbeat, so it lives in its own module rather than
inside research.py.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from brain.utils.time import iso_utc, parse_iso_utc

if TYPE_CHECKING:
    from brain.health.anomaly import BrainAnomaly

logger = logging.getLogger(__name__)

_VALID_SCOPES = ("internal", "external", "either")
Scope = Literal["internal", "external", "either"]


@dataclass(frozen=True)
class Interest:
    """One persona-level curiosity — topic + pull_score + scope + keywords."""

    id: str
    topic: str
    pull_score: float
    scope: Scope
    related_keywords: tuple[str, ...]
    notes: str
    first_seen: datetime  # tz-aware UTC
    last_fed: datetime  # tz-aware UTC
    last_researched_at: datetime | None
    feed_count: int
    source_types: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict) -> Interest:
        required = (
            "id",
            "topic",
            "pull_score",
            "scope",
            "related_keywords",
            "notes",
            "first_seen",
            "last_fed",
            "feed_count",
            "source_types",
        )
        for key in required:
            if key not in data:
                raise KeyError(f"Interest missing required key: {key!r}")

        scope = data["scope"]
        if scope not in _VALID_SCOPES:
            raise ValueError(
                f"Interest {data.get('topic')!r}: scope must be one of {_VALID_SCOPES}, got {scope!r}"
            )

        last_researched_raw = data.get("last_researched_at")
        last_researched = parse_iso_utc(last_researched_raw) if last_researched_raw else None

        return cls(
            id=str(data["id"]),
            topic=str(data["topic"]),
            pull_score=float(data["pull_score"]),
            scope=scope,
            related_keywords=tuple(str(k) for k in data["related_keywords"]),
            notes=str(data["notes"]),
            first_seen=parse_iso_utc(data["first_seen"]),
            last_fed=parse_iso_utc(data["last_fed"]),
            last_researched_at=last_researched,
            feed_count=int(data["feed_count"]),
            source_types=tuple(str(s) for s in data["source_types"]),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "pull_score": self.pull_score,
            "scope": self.scope,
            "related_keywords": list(self.related_keywords),
            "notes": self.notes,
            "first_seen": iso_utc(self.first_seen),
            "last_fed": iso_utc(self.last_fed),
            "last_researched_at": (
                iso_utc(self.last_researched_at) if self.last_researched_at else None
            ),
            "feed_count": self.feed_count,
            "source_types": list(self.source_types),
        }


@dataclass(frozen=True)
class InterestSet:
    """Loaded set of Interest records, with atomic save + helper queries."""

    interests: tuple[Interest, ...] = field(default_factory=tuple)

    @classmethod
    def load_with_anomaly(
        cls, path: Path, *, default_path: Path
    ) -> tuple[InterestSet, BrainAnomaly | None]:
        """Load with self-healing from .bak rotation if corrupt.

        Returns (instance, anomaly_or_None).
          - Missing file → load from default_path, no anomaly.
          - Corrupt file → quarantine, restore from .bak1/.bak2/.bak3 or write
            default; parse the result with existing per-entry coercion.
        """
        from brain.health.attempt_heal import attempt_heal

        def _default_factory() -> dict:
            return json.loads(default_path.read_text(encoding="utf-8"))

        def _schema_validator(data: object) -> None:
            if not isinstance(data, dict) or not isinstance(data.get("interests"), list):
                raise ValueError("interests schema invalid: missing 'interests' list")

        if not path.exists():
            data = _default_factory()
            anomaly = None
        else:
            data, anomaly = attempt_heal(path, _default_factory, schema_validator=_schema_validator)

        out: list[Interest] = []
        for raw in data.get("interests", []):
            try:
                out.append(Interest.from_dict(raw))
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("interest %r failed to load: %s", raw.get("topic"), exc)
                continue
        return cls(interests=tuple(out)), anomaly

    @classmethod
    def load(cls, path: Path, *, default_path: Path) -> InterestSet:
        """Load interests; on corrupt file, quarantine + heal, log WARNING."""
        source_path = path if path.exists() else default_path
        if source_path != path:
            logger.warning("interests file %s not found, using defaults", path)
            data = json.loads(default_path.read_text(encoding="utf-8"))
            out: list[Interest] = []
            for raw in data.get("interests", []):
                try:
                    out.append(Interest.from_dict(raw))
                except (KeyError, ValueError, TypeError) as exc:
                    logger.warning("interest %r failed to load: %s", raw.get("topic"), exc)
                    continue
            return cls(interests=tuple(out))

        instance, anomaly = cls.load_with_anomaly(path, default_path=default_path)
        if anomaly is not None:
            logger.warning(
                "InterestSet anomaly detected: %s action=%s file=%s",
                anomaly.kind,
                anomaly.action,
                anomaly.file,
            )
        return instance

    def save(self, path: Path) -> None:
        """Atomic save via .bak rotation (save_with_backup)."""
        from brain.health.adaptive import compute_treatment
        from brain.health.attempt_heal import save_with_backup

        payload = {"version": 1, "interests": [i.to_dict() for i in self.interests]}
        treatment = compute_treatment(path.parent, path.name)
        save_with_backup(path, payload, backup_count=treatment.backup_count)

    def find_by_topic(self, topic: str) -> Interest | None:
        lower = topic.lower()
        for i in self.interests:
            if i.topic.lower() == lower:
                return i
        return None

    def bump(self, topic: str, *, amount: float, now: datetime) -> InterestSet:
        """Return a new InterestSet with topic's pull_score nudged.

        Unknown topics return self unchanged (caller decides whether to add).
        """
        out: list[Interest] = []
        matched = False
        lower = topic.lower()
        for i in self.interests:
            if i.topic.lower() == lower:
                matched = True
                out.append(
                    Interest(
                        id=i.id,
                        topic=i.topic,
                        pull_score=i.pull_score + amount,
                        scope=i.scope,
                        related_keywords=i.related_keywords,
                        notes=i.notes,
                        first_seen=i.first_seen,
                        last_fed=now,
                        last_researched_at=i.last_researched_at,
                        feed_count=i.feed_count + 1,
                        source_types=i.source_types,
                    )
                )
            else:
                out.append(i)
        if not matched:
            return self
        return InterestSet(interests=tuple(out))

    def list_eligible(
        self, *, pull_threshold: float, cooldown_hours: float, now: datetime
    ) -> list[Interest]:
        """Return interests past pull_threshold + past cooldown.

        Sorted by pull_score desc, then by last_researched_at ascending
        (never-researched beats ever-researched on equal pull_score).
        """
        out: list[Interest] = []
        for i in self.interests:
            if i.pull_score < pull_threshold:
                continue
            if i.last_researched_at is not None:
                hours_since = (now - i.last_researched_at).total_seconds() / 3600.0
                if hours_since < cooldown_hours:
                    continue
            out.append(i)

        def sort_key(i: Interest) -> tuple[float, float]:
            last = i.last_researched_at
            # never-researched: very old effective timestamp
            ts = last.timestamp() if last is not None else 0.0
            return (-i.pull_score, ts)

        return sorted(out, key=sort_key)

    def upsert(self, interest: Interest) -> InterestSet:
        """Return a new InterestSet with interest added or replaced (by id)."""
        out = [i for i in self.interests if i.id != interest.id]
        out.append(interest)
        return InterestSet(interests=tuple(out))
