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
import statistics
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


_CALIBRATION_BLOCK_TEMPLATE = """\
=== Your recent editorial track record ===
Last 20 closed decisions:

  PROMOTED:
    • {n_replied} reached replied_explicit  ← {user_name} engaged
    • {n_acknowledged} reached acknowledged_unclear
    • {n_dismissed} reached dismissed       ← {user_name} ↩'d
    • {n_pending} still pending

  FILTERED:
    • {n_stayed_silent} stayed silent in draft (not re-emitted)
    • {n_recurred} re-emitted within 48h (you may have been too cautious)

Use this only as light context. It is who you've been, not who you must be.
"""


def build_calibration_block(persona_dir: Path, *, user_name: str) -> str:
    """Render the calibration block for D's adaptive system message.

    Reads the last 20 closed rows from d_calibration.jsonl. Counts buckets
    by outcome. Returns a formatted block ready to prepend to D's system
    message. With no history, all bucket counts are 0.
    """
    rows = list(read_recent_calibration_rows(persona_dir, limit=20))

    n_replied = sum(
        1 for r in rows if r.decision == "promote" and r.promoted_to_state == "replied_explicit"
    )
    n_acknowledged = sum(
        1 for r in rows if r.decision == "promote" and r.promoted_to_state == "acknowledged_unclear"
    )
    n_dismissed = sum(
        1 for r in rows if r.decision == "promote" and r.promoted_to_state == "dismissed"
    )
    # "Pending" = promoted rows not yet closed. read_recent_calibration_rows
    # only returns closed rows, so this is 0 from the calibration view.
    # The block still shows the bucket for completeness.
    n_pending = 0

    n_stayed_silent = sum(
        1 for r in rows if r.decision == "filter" and r.filtered_recurred is False
    )
    n_recurred = sum(1 for r in rows if r.decision == "filter" and r.filtered_recurred is True)

    return _CALIBRATION_BLOCK_TEMPLATE.format(
        n_replied=n_replied,
        n_acknowledged=n_acknowledged,
        n_dismissed=n_dismissed,
        n_pending=n_pending,
        n_stayed_silent=n_stayed_silent,
        n_recurred=n_recurred,
        user_name=user_name,
    )


@dataclass(frozen=True)
class DriftAlert:
    """Signal that D-reflection's promote-rate has drifted from its
    historical median. Emitted by detect_drift; consumed by the
    supervisor for operator-tier telemetry."""

    current_rate: float
    historical_median: float
    delta_sigma: float


def _read_all_d_calls(persona_dir: Path) -> list:
    """Read every row from initiate_d_calls.jsonl without a time filter."""
    import json as _json

    from brain.initiate.d_call_schema import DCallRow

    path = persona_dir / "initiate_d_calls.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\r\n")
            if not stripped.strip():
                continue
            try:
                rows.append(DCallRow.from_jsonl(stripped))
            except (_json.JSONDecodeError, KeyError):
                continue
    return rows


def detect_drift(persona_dir: Path) -> DriftAlert | None:
    """Compare recent D promote-rate against historical median.

    Returns a DriftAlert when |delta| > 2 * historical_stdev_via_MAD.
    Returns None when history is insufficient or flat.
    """
    rows = _read_all_d_calls(persona_dir)
    if len(rows) < 100:
        return None  # bootstrap floor

    rows.sort(key=lambda r: r.ts)
    current = rows[-30:]
    historical = rows[:-30]
    if len(historical) < 50:
        return None  # still bootstrapping

    def promote_rate(rs: list) -> float:
        total_in = sum(r.candidates_in for r in rs)
        total_promoted = sum(r.promoted_out for r in rs)
        return total_promoted / max(1, total_in)

    current_rate = promote_rate(current)
    historical_rates = [promote_rate([r]) for r in historical if r.candidates_in > 0]
    if len(historical_rates) < 2:
        return None
    historical_median = statistics.median(historical_rates)
    # Use sample stdev; MAD collapses to 0 when > 50 % of per-row rates
    # are identical (common with small candidates_in values).
    historical_stdev = statistics.stdev(historical_rates)

    if historical_stdev < 1e-6:
        return None  # flat history; can't detect drift meaningfully

    delta_sigma = abs(current_rate - historical_median) / historical_stdev
    if delta_sigma > 2.0:
        return DriftAlert(
            current_rate=current_rate,
            historical_median=historical_median,
            delta_sigma=delta_sigma,
        )
    return None
