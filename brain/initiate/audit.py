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
