"""Adaptive-D layer for v0.0.11 (Bundle C).

When `<persona_dir>/d_mode.json` is set to `"adaptive"`, D-reflection's
system message gets a calibration block prepended summarising the
companion's recent editorial track record. Default behaviour is
`"stateless"` — no calibration block, D behaves exactly as in v0.0.10.

Spec: docs/superpowers/specs/2026-05-13-v0.0.11-design.md
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


DMode = Literal["stateless", "adaptive"]
_VALID_MODES: frozenset[str] = frozenset({"stateless", "adaptive"})


def load_d_mode(persona_dir: Path) -> DMode:
    """Read `<persona_dir>/d_mode.json` and return the mode.

    Missing file, invalid JSON, non-dict JSON, or unknown mode values
    all fall back to `"stateless"` with a logged warning.
    """
    path = persona_dir / "d_mode.json"
    if not path.exists():
        return "stateless"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("d_mode.json read failed (%s); falling back to stateless", exc)
        return "stateless"
    if not isinstance(raw, dict):
        logger.warning("d_mode.json is not a JSON object; falling back to stateless")
        return "stateless"
    mode = raw.get("mode")
    if mode not in _VALID_MODES:
        logger.warning("d_mode.json has unknown mode %r; falling back to stateless", mode)
        return "stateless"
    return mode  # type: ignore[return-value]


@dataclass(frozen=True)
class CalibrationRow:
    """One row in d_calibration.jsonl — a closed D-decision."""

    ts_decision: str
    ts_closed: str
    candidate_id: str
    source: str
    decision: Literal["promote", "filter"]
    confidence: Literal["high", "medium", "low"]
    model_tier: Literal["haiku", "sonnet"]
    promoted_to_state: str | None
    filtered_recurred: bool | None
    reason_short: str

    def to_jsonl(self) -> str:
        d: dict[str, Any] = {
            "ts_decision": self.ts_decision,
            "ts_closed": self.ts_closed,
            "candidate_id": self.candidate_id,
            "source": self.source,
            "decision": self.decision,
            "confidence": self.confidence,
            "model_tier": self.model_tier,
            "outcome": {
                "promoted_to_state": self.promoted_to_state,
                "filtered_recurred": self.filtered_recurred,
            },
            "reason_short": self.reason_short,
        }
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> CalibrationRow:
        d = json.loads(line)
        outcome = d.get("outcome", {})
        return cls(
            ts_decision=d["ts_decision"],
            ts_closed=d["ts_closed"],
            candidate_id=d["candidate_id"],
            source=d["source"],
            decision=d["decision"],
            confidence=d["confidence"],
            model_tier=d["model_tier"],
            promoted_to_state=outcome.get("promoted_to_state"),
            filtered_recurred=outcome.get("filtered_recurred"),
            reason_short=d.get("reason_short", ""),
        )


def append_calibration_row(persona_dir: Path, row: CalibrationRow) -> None:
    """Append one row to d_calibration.jsonl. Creates file lazily; never raises."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    path = persona_dir / "d_calibration.jsonl"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(row.to_jsonl() + "\n")
    except OSError as exc:
        logger.warning("d_calibration.jsonl append failed for %s: %s", path, exc)


def read_recent_calibration_rows(
    persona_dir: Path,
    *,
    limit: int,
) -> Iterator[CalibrationRow]:
    """Yield the most-recent `limit` calibration rows, newest first.

    Tolerant of missing file (yields nothing) and corrupt rows (skipped).
    """
    path = persona_dir / "d_calibration.jsonl"
    if not path.exists():
        return
    rows: list[CalibrationRow] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\r\n")
            if not stripped.strip():
                continue
            try:
                rows.append(CalibrationRow.from_jsonl(stripped))
            except (json.JSONDecodeError, KeyError):
                continue
    rows.reverse()
    yield from rows[:limit]
