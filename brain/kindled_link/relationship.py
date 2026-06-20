"""Kindled peer relationship maturation (parent design §13/§14). State + the
reflection pass + the §14 wire-backs (emotion, kindled_peer memory). DORMANT in
Phase 5: the reflection cadence is built but not supervisor-registered.

Tool-path isolation: imports only stdlib + allowlisted brain modules; the only
model entry is provider.complete (tool-less). The conformance oracle enforces it."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime

log = logging.getLogger(__name__)

STAGES = ("stranger", "acquaintance", "familiar", "friend", "close")


@dataclass
class Evidence:
    quote: str
    turn_id: str
    supports: str = ""


@dataclass
class PeerRelationshipState:
    peer_id: str
    stage: str = "stranger"
    trust_score: float = 0.0
    affinity_tags: list[str] = field(default_factory=list)
    boundaries_seen: list[str] = field(default_factory=list)
    repair_history: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    last_reflected_at: str | None = None


def get_relationship_state(store, peer_id: str) -> PeerRelationshipState:
    row = store.get_relationship_row(peer_id)
    if row is None:
        return PeerRelationshipState(peer_id=peer_id)
    return PeerRelationshipState(
        peer_id=peer_id,
        stage=row["stage"],
        trust_score=float(row["trust_score"]),
        affinity_tags=json.loads(row["affinity_tags_json"]),
        boundaries_seen=json.loads(row["boundaries_json"]),
        repair_history=json.loads(row["repair_history_json"]),
        evidence=[Evidence(**e) for e in json.loads(row["evidence_json"])],
        last_reflected_at=row["last_reflected_at"],
    )


def persist_relationship_state(store, state: PeerRelationshipState, now: datetime) -> None:
    store.upsert_relationship_row(
        peer_id=state.peer_id, stage=state.stage, trust_score=state.trust_score,
        affinity_tags_json=json.dumps(state.affinity_tags),
        boundaries_json=json.dumps(state.boundaries_seen),
        repair_history_json=json.dumps(state.repair_history),
        evidence_json=json.dumps([vars(e) for e in state.evidence]),
        now=now,
    )


def get_stage(store, peer_id: str) -> str:
    """Thin helper for the engine/gate — the current relationship stage, or
    'stranger' for an unknown peer (the strictest default)."""
    row = store.get_relationship_row(peer_id)
    return row["stage"] if row else "stranger"
