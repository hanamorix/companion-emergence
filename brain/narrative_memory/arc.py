"""Arc + ArcMember frozen dataclasses.

Per spec §3 — no I/O here. Serialisation lives in state.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArcMember:
    """One memory's membership in an arc.

    salience_at_join is captured at join time via forgetting.salience.score
    and is used for deterministic member-cap eviction (see policy.py).
    """

    memory_id: str
    joined_at_iso: str
    lived_age_at_join: float
    salience_at_join: float


@dataclass(frozen=True)
class Arc:
    """An anchor-seeded narrative arc.

    state transitions are one-way ("open" -> "closed"); a re-emerging
    theme produces a fresh arc seeded by a new anchor.

    seed_anchor_type is one of {"dream", "growth", "soul"} in v1
    (weather_shift anchors are skipped per spec §2).
    """

    id: str
    state: str
    seed_anchor_type: str
    seed_anchor_ref: str
    seed_memory_ids: tuple[str, ...]
    title: str
    opened_at_iso: str
    lived_age_at_open: float
    last_extended_at_iso: str
    closed_at_iso: str | None
    lived_age_at_close: float | None
    members: tuple[ArcMember, ...]
