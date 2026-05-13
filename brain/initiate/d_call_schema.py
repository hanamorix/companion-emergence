"""Per-call audit row for D-reflection ticks.

One row per heartbeat tick where D actually fired (queue non-empty).
The substrate for the stateless-but-observable contract — joins against
initiate_audit + delivery_state later for hit-rate computation.

File: <persona_dir>/initiate_d_calls.jsonl (append-only).
"""
from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

ModelTier = Literal["haiku", "sonnet"]
FailureType = Literal[
    "timeout",
    "provider_error",
    "malformed_json",
    "rate_limit",
    "both_low_confidence",
]


def make_d_call_id(now: datetime) -> str:
    """Generate a sortable, unique D-call ID. Format: dc_<iso8601>_<rand>."""
    stamp = now.strftime("%Y-%m-%dT%H-%M-%S")
    return f"dc_{stamp}_{secrets.token_hex(2)}"


@dataclass
class DCallRow:
    d_call_id: str
    ts: str  # ISO 8601 with tz
    tick_id: str
    model_tier_used: ModelTier
    candidates_in: int
    promoted_out: int
    filtered_out: int
    latency_ms: int
    tokens_input: int
    tokens_output: int
    failure_type: FailureType | None = None
    retry_count: int = 0
    tick_note: str | None = None

    def to_jsonl(self) -> str:
        d: dict[str, Any] = {
            "d_call_id": self.d_call_id,
            "ts": self.ts,
            "tick_id": self.tick_id,
            "model_tier_used": self.model_tier_used,
            "candidates_in": self.candidates_in,
            "promoted_out": self.promoted_out,
            "filtered_out": self.filtered_out,
            "latency_ms": self.latency_ms,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "failure_type": self.failure_type,
            "retry_count": self.retry_count,
            "tick_note": self.tick_note,
        }
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> DCallRow:
        d = json.loads(line)
        return cls(
            d_call_id=d["d_call_id"],
            ts=d["ts"],
            tick_id=d["tick_id"],
            model_tier_used=d["model_tier_used"],
            candidates_in=d["candidates_in"],
            promoted_out=d["promoted_out"],
            filtered_out=d["filtered_out"],
            latency_ms=d["latency_ms"],
            tokens_input=d["tokens_input"],
            tokens_output=d["tokens_output"],
            failure_type=d.get("failure_type"),
            retry_count=d.get("retry_count", 0),
            tick_note=d.get("tick_note"),
        )
