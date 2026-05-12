"""Data structures for the initiate pipeline.

Three core types:

* InitiateCandidate — what gets queued by event emitters
* AuditRow — what gets written by the review tick (mutates as state transitions)
* EmotionalSnapshot / SemanticContext — embedded structures
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal

CandidateKind = Literal["message", "voice_edit_proposal"]
CandidateSource = Literal["dream", "crystallization", "emotion_spike", "voice_reflection"]
Decision = Literal[
    "send_notify", "send_quiet", "hold", "drop", "error", "filtered_pre_compose"
]
StateName = Literal[
    "pending", "delivered", "read",
    "replied_explicit", "acknowledged_unclear", "unanswered",
    "dismissed",
]


def make_candidate_id(now: datetime) -> str:
    """Generate a sortable, unique candidate ID. Format: ic_<iso8601>_<rand>."""
    stamp = now.strftime("%Y-%m-%dT%H-%M-%S")
    return f"ic_{stamp}_{secrets.token_hex(2)}"


def make_audit_id(now: datetime) -> str:
    """Generate a sortable, unique audit ID. Format: ia_<iso8601>_<rand>."""
    stamp = now.strftime("%Y-%m-%dT%H-%M-%S")
    return f"ia_{stamp}_{secrets.token_hex(2)}"


@dataclass
class EmotionalSnapshot:
    vector: dict[str, float]
    rolling_baseline_mean: float
    rolling_baseline_stdev: float
    current_resonance: float
    delta_sigma: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EmotionalSnapshot:
        return cls(**d)


@dataclass
class SemanticContext:
    linked_memory_ids: list[str] = field(default_factory=list)
    topic_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SemanticContext:
        return cls(**d)


@dataclass
class InitiateCandidate:
    """A queued initiate candidate.

    `emotional_snapshot` is Optional because some emitters (notably
    voice-reflection, which looks at the last week of activity) have no
    moment-in-time emotion to capture. Carrying a zero-filled snapshot
    there would be a structurally-valid lie; None is semantically honest.

    Non-heartbeat emitters that DO have an emotional context at emit time
    (dream cycle, crystallizers) populate the `vector` field from that
    context. The rolling_baseline / current_resonance / delta_sigma fields
    are heartbeat-specific — non-periodic emitters leave them at 0.0 with
    a docstring note rather than fabricating values.
    """

    candidate_id: str
    ts: str  # ISO 8601 with tz
    kind: CandidateKind
    source: CandidateSource
    source_id: str
    semantic_context: SemanticContext
    emotional_snapshot: EmotionalSnapshot | None = None
    claimed_at: str | None = None
    # Voice-edit-only payload (None for kind="message").
    proposal: dict[str, Any] | None = None

    def to_jsonl(self) -> str:
        d = {
            "candidate_id": self.candidate_id,
            "ts": self.ts,
            "kind": self.kind,
            "source": self.source,
            "source_id": self.source_id,
            "emotional_snapshot": (
                self.emotional_snapshot.to_dict()
                if self.emotional_snapshot is not None
                else None
            ),
            "semantic_context": self.semantic_context.to_dict(),
            "claimed_at": self.claimed_at,
        }
        if self.proposal is not None:
            d["proposal"] = self.proposal
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> InitiateCandidate:
        d = json.loads(line)
        snap_raw = d.get("emotional_snapshot")
        snap = EmotionalSnapshot.from_dict(snap_raw) if snap_raw is not None else None
        return cls(
            candidate_id=d["candidate_id"],
            ts=d["ts"],
            kind=d["kind"],
            source=d["source"],
            source_id=d["source_id"],
            emotional_snapshot=snap,
            semantic_context=SemanticContext.from_dict(d["semantic_context"]),
            claimed_at=d.get("claimed_at"),
            proposal=d.get("proposal"),
        )


@dataclass
class AuditRow:
    audit_id: str
    candidate_id: str
    ts: str
    kind: CandidateKind
    subject: str
    tone_rendered: str
    decision: Decision
    decision_reasoning: str
    gate_check: dict[str, Any]
    delivery: dict[str, Any] | None = None
    # Voice-edit-only payload (None for kind="message").
    diff: str | None = None
    user_modified: bool = False

    def record_transition(self, to: StateName, at: str) -> None:
        """Append a state transition; create the delivery block lazily."""
        if self.delivery is None:
            self.delivery = {
                "delivered_at": at if to == "delivered" else None,
                "state_transitions": [],
                "current_state": to,
            }
        self.delivery["state_transitions"].append({"to": to, "at": at})
        self.delivery["current_state"] = to
        if to == "delivered" and self.delivery["delivered_at"] is None:
            self.delivery["delivered_at"] = at

    def to_jsonl(self) -> str:
        d = {
            "audit_id": self.audit_id,
            "candidate_id": self.candidate_id,
            "ts": self.ts,
            "kind": self.kind,
            "subject": self.subject,
            "tone_rendered": self.tone_rendered,
            "decision": self.decision,
            "decision_reasoning": self.decision_reasoning,
            "gate_check": self.gate_check,
            "delivery": self.delivery,
        }
        if self.diff is not None:
            d["diff"] = self.diff
            d["user_modified"] = self.user_modified
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> AuditRow:
        d = json.loads(line)
        return cls(
            audit_id=d["audit_id"],
            candidate_id=d["candidate_id"],
            ts=d["ts"],
            kind=d["kind"],
            subject=d["subject"],
            tone_rendered=d["tone_rendered"],
            decision=d["decision"],
            decision_reasoning=d["decision_reasoning"],
            gate_check=d["gate_check"],
            delivery=d.get("delivery"),
            diff=d.get("diff"),
            user_modified=d.get("user_modified", False),
        )
