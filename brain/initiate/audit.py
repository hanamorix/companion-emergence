"""Initiate audit log — append + read + per-row state mutation.

File contract:
- initiate_audit.jsonl (active) — per-row mutations allowed via atomic rewrite
- initiate_audit.YYYY.jsonl.gz (archives) — yearly archive, kept forever

Mirrors brain.soul.audit + iter_audit_full from the v0.0.8 retention work.
Same forever-keep policy: every decision Nell makes about reaching out
must remain accessible.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.health.jsonl_reader import iter_jsonl_streaming
from brain.initiate.d_call_schema import DCallRow
from brain.initiate.schemas import AuditRow, StateName

logger = logging.getLogger(__name__)

_ARCHIVE_PATTERN = re.compile(r"^initiate_audit\.(\d{4})\.jsonl\.gz$")


def append_audit_row(persona_dir: Path, row: AuditRow) -> None:
    """Append one row to initiate_audit.jsonl (creates file lazily)."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    path = persona_dir / "initiate_audit.jsonl"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(row.to_jsonl() + "\n")
    except OSError as exc:
        logger.warning("initiate audit append failed for %s: %s", path, exc)


def update_audit_state(
    persona_dir: Path,
    *,
    audit_id: str,
    new_state: StateName,
    at: str,
) -> None:
    """Mutate one audit row's delivery block to record a state transition.

    Atomic via temp + rename. The audit log row mutates in place — the
    delivery.state_transitions array carries the full timeline, but the
    current_state field reflects the latest.
    """
    path = persona_dir / "initiate_audit.jsonl"
    if not path.exists():
        return
    rows: list[AuditRow] = []
    found = False
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\r\n")
            if not stripped.strip():
                continue
            try:
                row = AuditRow.from_jsonl(stripped)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("skipping corrupt audit row in %s: %s", path, exc)
                continue
            if row.audit_id == audit_id:
                row.record_transition(new_state, at)
                found = True
            rows.append(row)
    if not found:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(r.to_jsonl() + "\n")
        tmp.replace(path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        logger.warning("audit state update failed for %s: %s", path, exc)


def read_recent_audit(
    persona_dir: Path,
    *,
    window_hours: float,
    now: datetime | None = None,
) -> Iterator[AuditRow]:
    """Yield audit rows from the active file whose ts is within `window_hours`.

    Streaming — does not load the full file. Archives are NOT scanned by
    this reader (use iter_initiate_audit_full for that). The window is
    relative to `now` (defaults to datetime.now(UTC)).
    """
    path = persona_dir / "initiate_audit.jsonl"
    if not path.exists():
        return
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(hours=window_hours)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\r\n")
            if not stripped.strip():
                continue
            try:
                row = AuditRow.from_jsonl(stripped)
            except (json.JSONDecodeError, KeyError):
                continue
            try:
                row_ts = datetime.fromisoformat(row.ts)
            except ValueError:
                continue
            if row_ts >= cutoff:
                yield row


def iter_initiate_audit_full(persona_dir: Path) -> Iterator[AuditRow]:
    """Yield every audit row across active + yearly archives, chronologically.

    Mirrors brain.soul.audit.iter_audit_full. Walks
    initiate_audit.YYYY.jsonl.gz archives oldest year -> newest year, then
    the active file. Streaming via iter_jsonl_streaming so memory stays
    bounded.
    """
    if persona_dir.exists():
        archives: list[tuple[int, Path]] = []
        for child in persona_dir.iterdir():
            m = _ARCHIVE_PATTERN.match(child.name)
            if m:
                archives.append((int(m.group(1)), child))
        archives.sort(key=lambda t: t[0])
        for _year, archive_path in archives:
            for raw in iter_jsonl_streaming(archive_path):
                try:
                    yield AuditRow.from_jsonl(json.dumps(raw))
                except (KeyError, TypeError):
                    continue
    active = persona_dir / "initiate_audit.jsonl"
    for raw in iter_jsonl_streaming(active):
        try:
            yield AuditRow.from_jsonl(json.dumps(raw))
        except (KeyError, TypeError):
            continue


def append_d_call_row(persona_dir: Path, row: DCallRow) -> None:
    """Append one row to initiate_d_calls.jsonl (creates file lazily)."""
    persona_dir.mkdir(parents=True, exist_ok=True)
    path = persona_dir / "initiate_d_calls.jsonl"
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(row.to_jsonl() + "\n")
    except OSError as exc:
        logger.warning("initiate_d_calls append failed for %s: %s", path, exc)


def read_recent_d_calls(
    persona_dir: Path,
    *,
    window_hours: float,
    now: datetime | None = None,
) -> Iterator[DCallRow]:
    """Yield D-call rows within `window_hours` of `now` (defaults to datetime.now(UTC))."""
    path = persona_dir / "initiate_d_calls.jsonl"
    if not path.exists():
        return
    now = now or datetime.now(UTC)
    cutoff = now - timedelta(hours=window_hours)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\r\n")
            if not stripped.strip():
                continue
            try:
                row = DCallRow.from_jsonl(stripped)
            except (json.JSONDecodeError, KeyError):
                continue
            try:
                row_ts = datetime.fromisoformat(row.ts)
            except ValueError:
                continue
            if row_ts >= cutoff:
                yield row


# Decision values that mark D-promoted rows.
_D_PROMOTED_DECISIONS: frozenset[str] = frozenset({
    "promoted_by_d",
    "promoted_by_d_malformed_fallback",
    "promoted_by_d_after_3_failures",
})

# Decision values that mark D-filtered rows (used by Pass 2 in T11).
_D_FILTERED_DECISIONS: frozenset[str] = frozenset({
    "filtered_pre_compose",
    "filtered_pre_compose_low_confidence",
    "filtered_d_budget",
})

# Delivery states that count as "terminal" for promoted candidates.
_TERMINAL_STATES: frozenset[str] = frozenset({
    "replied_explicit",
    "acknowledged_unclear",
    "unanswered",
    "dismissed",
})


def run_calibration_closer_tick(
    persona_dir: Path,
    *,
    now: datetime | None = None,
) -> None:
    """Walk recent audit rows, write d_calibration.jsonl rows for D
    decisions that have reached terminal state.

    Pass 1: promoted-by-D rows where delivery.current_state is in
    _TERMINAL_STATES.

    Pass 2 (T11): filtered-by-D rows where 48h has elapsed since the
    decision timestamp.

    Idempotent: dedupes against existing calibration rows via candidate_id.
    """
    from brain.initiate.adaptive import (
        CalibrationRow,
        append_calibration_row,
        read_recent_calibration_rows,
    )

    now = now or datetime.now(UTC)

    # Build dedupe set from existing calibration rows.
    seen_ids = {
        r.candidate_id
        for r in read_recent_calibration_rows(persona_dir, limit=500)
    }

    # Pass 1 — promoted-by-D rows that have reached terminal.
    for audit_row in read_recent_audit(persona_dir, window_hours=24 * 90, now=now):
        if audit_row.decision not in _D_PROMOTED_DECISIONS:
            continue
        if audit_row.candidate_id in seen_ids:
            continue
        delivery = audit_row.delivery or {}
        current_state = delivery.get("current_state")
        if current_state not in _TERMINAL_STATES:
            continue
        # Closure timestamp: the state_transitions entry for current_state.
        ts_closed = now.isoformat()
        for transition in delivery.get("state_transitions", []):
            if transition.get("to") == current_state:
                ts_closed = transition.get("at", ts_closed)

        cal = CalibrationRow(
            ts_decision=audit_row.ts,
            ts_closed=ts_closed,
            candidate_id=audit_row.candidate_id,
            source=_extract_source_from_decision(audit_row),
            decision="promote",
            confidence=_extract_confidence(audit_row),
            model_tier=_extract_model_tier(audit_row),
            promoted_to_state=current_state,
            filtered_recurred=None,
            reason_short=audit_row.decision_reasoning[:80],
        )
        append_calibration_row(persona_dir, cal)
        seen_ids.add(audit_row.candidate_id)

    # Pass 2 — filtered-by-D rows that have aged past 48h.
    forty_eight_hours = timedelta(hours=48)
    # Window padded to 72h to catch some closer-was-down-yesterday cases.
    for audit_row in read_recent_audit(persona_dir, window_hours=72, now=now):
        if audit_row.decision not in _D_FILTERED_DECISIONS:
            continue
        if audit_row.candidate_id in seen_ids:
            continue
        try:
            decision_ts = datetime.fromisoformat(audit_row.ts)
        except ValueError:
            continue
        if (now - decision_ts) < forty_eight_hours:
            continue  # not yet terminal

        # Determine filtered_recurred: did the same (source, source_id)
        # re-emit as a candidate within 48h of the original decision?
        source_id_of_filter = audit_row.gate_check.get("source_id")
        filtered_recurred = False
        if source_id_of_filter:
            filtered_recurred = _source_id_recurred_within(
                persona_dir,
                source_id=source_id_of_filter,
                source=audit_row.gate_check.get("source", ""),
                start_ts=decision_ts,
                window=forty_eight_hours,
            )

        cal = CalibrationRow(
            ts_decision=audit_row.ts,
            ts_closed=(decision_ts + forty_eight_hours).isoformat(),
            candidate_id=audit_row.candidate_id,
            source=_extract_source_from_decision(audit_row),
            decision="filter",
            confidence=_extract_confidence(audit_row),
            model_tier=_extract_model_tier(audit_row),
            promoted_to_state=None,
            filtered_recurred=filtered_recurred,
            reason_short=audit_row.decision_reasoning[:80],
        )
        append_calibration_row(persona_dir, cal)
        seen_ids.add(audit_row.candidate_id)


def _source_id_recurred_within(
    persona_dir: Path,
    *,
    source_id: str,
    source: str,
    start_ts: datetime,
    window: timedelta,
) -> bool:
    """Check whether a candidate with (source, source_id) was emitted into
    the queue within `window` after `start_ts`."""
    from brain.initiate.emit import read_candidates

    end_ts = start_ts + window
    for c in read_candidates(persona_dir):
        if c.source != source or c.source_id != source_id:
            continue
        try:
            c_ts = datetime.fromisoformat(c.ts)
        except ValueError:
            continue
        if start_ts < c_ts <= end_ts:
            return True
    return False


def _extract_source_from_decision(audit_row: AuditRow) -> str:
    """Best-effort extract candidate source from gate_check; 'unknown' fallback.

    v0.0.10 audit rows don't currently carry source directly.
    """
    return audit_row.gate_check.get("source", "unknown")


def _extract_confidence(audit_row: AuditRow) -> str:
    """Best-effort extract D's confidence from gate_check; 'high' fallback."""
    return audit_row.gate_check.get("d_confidence", "high")


def _extract_model_tier(audit_row: AuditRow) -> str:
    """Best-effort extract D's model tier from gate_check; 'haiku' fallback."""
    return audit_row.gate_check.get("d_model_tier", "haiku")
