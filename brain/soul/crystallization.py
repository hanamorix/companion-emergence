"""Crystallization — a permanent soul moment that cannot be unfelt.

A crystallization is an identity-level claim. Unlike memories (which decay
via Hebbian) and reflexes (which fade), crystallizations are permanent.
The only path out is revoke — which moves to a revoked array but never
deletes. Per OG nell_soul.json live shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True)
class Crystallization:
    """One soul-level moment.

    Attributes:
        id: UUID string.
        moment: The moment in Nell's voice — what happened.
        love_type: One of LOVE_TYPES keys.
        why_it_matters: Short explanation of why this is permanent.
        crystallized_at: tz-aware UTC datetime.
        who_or_what: Optional — hana, jordan, writing, etc.
        resonance: 1-10 emotional intensity at time of crystallization.
        permanent: Always True for active crystallizations.
        revoked_at: Set when revoked; None if active.
        revoked_reason: Explanation if revoked; empty string if active.
    """

    id: str
    moment: str
    love_type: str
    why_it_matters: str
    crystallized_at: datetime
    who_or_what: str = ""
    resonance: int = 8
    permanent: bool = True
    revoked_at: datetime | None = None
    revoked_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dict for JSON or SQLite storage."""
        return {
            "id": self.id,
            "moment": self.moment,
            "love_type": self.love_type,
            "why_it_matters": self.why_it_matters,
            "crystallized_at": self.crystallized_at.isoformat(),
            "who_or_what": self.who_or_what,
            "resonance": self.resonance,
            "permanent": self.permanent,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "revoked_reason": self.revoked_reason,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Crystallization:
        """Restore from a dict produced by to_dict.

        Tz-naive timestamps are coerced to UTC for compatibility with
        OG nell_soul.json entries that lack explicit tz info.
        """
        crystallized_at = _coerce_utc(d["crystallized_at"])
        revoked_at_raw = d.get("revoked_at")
        revoked_at = _coerce_utc(revoked_at_raw) if revoked_at_raw else None

        return cls(
            id=str(d["id"]),
            moment=str(d["moment"]),
            love_type=str(d["love_type"]),
            why_it_matters=str(d.get("why_it_matters", "")),
            crystallized_at=crystallized_at,
            who_or_what=str(d.get("who_or_what") or ""),
            resonance=int(d.get("resonance", 8)),
            permanent=bool(d.get("permanent", True)),
            revoked_at=revoked_at,
            revoked_reason=str(d.get("revoked_reason") or ""),
        )


def _coerce_utc(ts: str) -> datetime:
    """Parse ISO-8601 timestamp; coerce tz-naive values to UTC."""
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
