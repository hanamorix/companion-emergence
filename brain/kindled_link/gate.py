"""Outbound-payload gate seam for Kindled peer sessions. Phase 3 ships the
Protocol + a fail-closed deny-all default; Phase 4 swaps in the real
privacy_gate.py. Nothing crosses to a peer ungated (parent design §5 inv. 127)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

# action ∈ {send, revise, hold, end_or_pause} (parent design §12)
_ACTIONS = frozenset({"send", "revise", "hold", "end_or_pause"})


@dataclass
class OutboundPayload:
    """The full thing reviewed + (if cleared) sent: draft body AND the
    relationship hint. In Phase 3 the hint is minimal/optional (Phase 5
    populates it) but the SHAPE carries it so the gate reviews it from day one."""
    body: str
    relationship_hint: dict | None = None


@dataclass
class GateDecision:
    action: str
    reason: str = ""
    revision_constraints: str | None = None
    texture_score: float = 0.0

    def __post_init__(self) -> None:
        if self.action not in _ACTIONS:
            raise ValueError(f"unknown gate action: {self.action!r}")


@runtime_checkable
class PrivacyGate(Protocol):
    def review(
        self,
        payload: OutboundPayload,
        *,
        peer_id: str,
        stage: str,
        transcript_summary: str,
        reason: str,
        now: datetime,
        today: str,
    ) -> GateDecision: ...


class DenyAllGate:
    """Phase 3 default. No privacy gate exists yet, so every payload is held.
    The engine's full draft→gate→send path is exercised, but nothing leaves."""

    def review(
        self,
        payload: OutboundPayload,
        *,
        peer_id: str,
        stage: str,
        transcript_summary: str,
        reason: str,
        now: datetime,
        today: str,
    ) -> GateDecision:
        return GateDecision(action="hold", reason="phase-3: no privacy gate yet")
