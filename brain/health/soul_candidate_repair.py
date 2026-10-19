"""One-time startup migration repairing stuck monologue soul candidates.

Before v0.0.34, the monologue crystallisation path queued candidates with
``text=theme`` (a bare 2-word label) and buried the evidence in a placeholder
memory.  The reviewer saw context-free fragments ("Ordinary trust",
"Building the control on purpose") that deferred forever — 20 stuck on the
reference persona.

This migration:
  - Targets: ``status == "auto_pending"`` AND ``session_id == "monologue"``
    (the only writer of that session_id).
  - For each: fetch the ``memory_id`` memory.  Content non-empty →
    rewrite ``text`` to the memory content (or ``old_text — content``
    if the old text isn't already contained), clear ``last_deferred_at``,
    set ``defer_count = 0`` → next review pass decides with real context.
  - Memory missing/empty → mark ``expired`` with
    ``reason="pre-fix fragment with no recoverable context"``.
  - Rewrite goes through the same ``file_lock`` the queue/review paths use.

State file ``<persona_dir>/soul_candidate_repair_state.json`` → idempotent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brain.memory.store import MemoryStore

from brain.health.json_io import (
    JsonFileError,
    Recovery,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from brain.utils.failure_scope import settle_scope
from brain.utils.file_lock import file_lock

logger = logging.getLogger(__name__)

_STATE_FILE = "soul_candidate_repair_state.json"
_CANDIDATES_FILE = "soul_candidates.jsonl"
_TEXT_CAP = 600

# Target: only auto_pending monologue-sourced candidates.
_TARGET_STATUS = "auto_pending"
_TARGET_SESSION_ID = "monologue"


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class SoulCandidateRepairReport:
    repaired: int       # candidates whose text was backfilled
    expired: int        # candidates expired (no recoverable context)
    status: str         # "complete" | "skipped"
    completed_at: str   # ISO timestamp


# ---------------------------------------------------------------------------
# State I/O
# ---------------------------------------------------------------------------


def _state_path(persona_dir: Path) -> Path:
    return persona_dir / _STATE_FILE


def _load_state(persona_dir: Path) -> dict | None:
    # CONTENT: an unreadable file raises (as it always did); a missing or corrupt one is "no state".
    return read_json(
        _state_path(persona_dir), default=lambda: None, recover=Recovery.CONTENT, expect=dict
    )


def _save_state(persona_dir: Path, state: dict) -> None:
    write_json(_state_path(persona_dir), state, indent=2, fsync=False)


def _write_and_return(
    persona_dir: Path, *, repaired: int, expired: int
) -> SoulCandidateRepairReport:
    completed_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    state = {
        "status": "complete",
        "repaired": repaired,
        "expired": expired,
        "completed_at": completed_at,
    }
    _save_state(persona_dir, state)
    return SoulCandidateRepairReport(
        repaired=repaired,
        expired=expired,
        status="complete",
        completed_at=completed_at,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def should_run_soul_candidate_repair(persona_dir: Path) -> bool:
    """Return True iff the repair has not completed AND there are targets.

    Targets: auto_pending candidates with session_id=="monologue".
    """
    existing = _load_state(persona_dir)
    if existing is not None and existing.get("status") == "complete":
        return False

    records = read_jsonl(persona_dir / _CANDIDATES_FILE)

    return any(
        r.get("status") == _TARGET_STATUS and r.get("session_id") == _TARGET_SESSION_ID
        for r in records
    )


def run_soul_candidate_repair(
    persona_dir: Path,
    *,
    store: MemoryStore,
) -> SoulCandidateRepairReport:
    """Run the one-time soul candidate repair.  Idempotent — skips if complete.

    Parameters
    ----------
    persona_dir:
        Path to the persona's data directory.
    store:
        Open MemoryStore for memory content lookup.  Provider-free — no LLM calls.
    """
    existing = _load_state(persona_dir)
    if existing is not None and existing.get("status") == "complete":
        return SoulCandidateRepairReport(
            repaired=existing.get("repaired", 0),
            expired=existing.get("expired", 0),
            status="complete",
            completed_at=existing.get("completed_at", ""),
        )

    candidates_path = persona_dir / _CANDIDATES_FILE
    if not candidates_path.exists():
        return _write_and_return(persona_dir, repaired=0, expired=0)

    with file_lock(candidates_path):
        return _repair_locked(persona_dir, candidates_path, store=store)


# ---------------------------------------------------------------------------
# Internal: locked body
# ---------------------------------------------------------------------------


def _repair_locked(
    persona_dir: Path,
    candidates_path: Path,
    *,
    store: MemoryStore,
) -> SoulCandidateRepairReport:
    """Read-modify-rewrite with the file lock already held by the caller."""
    try:
        # RAISE: the records are rewritten below, so a failed read must stop here, not look like a short file.
        records = read_jsonl(candidates_path, recover=Recovery.RAISE)
    except JsonFileError as exc:
        # A read it could not do is NOT a completed repair: no state file is written, so the next start tries again (#460).
        logger.warning(
            "soul_candidate_repair: cannot read candidates: %s; leaving the repair pending", exc
        )
        return SoulCandidateRepairReport(repaired=0, expired=0, status="skipped", completed_at="")

    repaired = 0
    expired = 0
    targets = 0
    skipped_ids: list[str] = []

    for record in records:
        if not _is_target(record):
            continue
        targets += 1

        old_text = str(record.get("text") or "").strip()
        memory_id = str(record.get("memory_id") or "").strip()

        try:
            content = _fetch_content(store, memory_id)
        except Exception as exc:  # noqa: BLE001 — decided by cause below; never swallowed
            if settle_scope(exc, store) == "environment":
                raise  # locked / closed / I/O: nothing saved, nothing recorded; the next start retries
            # A fault of THIS row (it will fail the same way again): leave the record exactly as it was.
            logger.warning(
                "soul_candidate_repair: cannot read memory %s (%s: %s); leaving its candidate as it was",
                memory_id, type(exc).__name__, exc,
            )
            skipped_ids.append(memory_id)
            continue

        if content:
            new_text = _build_text(old_text, content)
            record["text"] = new_text
            record["defer_count"] = 0
            record.pop("last_deferred_at", None)
            repaired += 1
            logger.info(
                "soul_candidate_repair: repaired candidate %s (text length: %d → %d)",
                memory_id,
                len(old_text),
                len(new_text),
            )
        else:
            record["status"] = "expired"
            record["reason"] = "pre-fix fragment with no recoverable context"
            record["expired_at"] = datetime.now(UTC).isoformat()
            expired += 1
            logger.info(
                "soul_candidate_repair: expired candidate %s (no recoverable content)",
                memory_id,
            )

    if targets and len(skipped_ids) == targets:
        # Item faults are per row; every target failing is systemic, so this is not a finished repair.
        logger.warning(
            "soul_candidate_repair: all %d targets failed to read their memory; not recording the repair as complete", targets
        )
        return SoulCandidateRepairReport(repaired=0, expired=0, status="skipped", completed_at="")

    if repaired > 0 or expired > 0:
        _save_candidates(candidates_path, records)

    return _write_and_return(persona_dir, repaired=repaired, expired=expired)


def _is_target(record: dict) -> bool:
    """True iff this record is a stuck monologue auto_pending candidate."""
    return (
        record.get("status") == _TARGET_STATUS
        and record.get("session_id") == _TARGET_SESSION_ID
    )


def _fetch_content(store: MemoryStore, memory_id: str) -> str | None:
    """Fetch memory content for memory_id.

    Returns None when there is no content to recover: an empty id, a row that does not exist, or empty content. A FAILED
    read raises (it is not "no content"): the caller decides by cause what to do (#460).
    """
    if not memory_id:
        return None
    mem = store.get(memory_id)
    if mem is None:
        return None
    content = (mem.content or "").strip()
    return content if content else None


def _build_text(old_text: str, content: str) -> str:
    """Combine old_text and content into the new candidate text.

    Dedup guard: if the memory content already starts with or contains
    the old text (post-S1 placeholder memories already hold the combined
    string), use the memory content directly.  Otherwise combine them.

    Capped at _TEXT_CAP characters.
    """
    if old_text and old_text in content:
        # Memory content already contains the old label — use it directly.
        return content[:_TEXT_CAP]
    if old_text:
        combined = f"{old_text} — {content}"
    else:
        combined = content
    return combined[:_TEXT_CAP]


def _save_candidates(candidates_path: Path, records: list[dict]) -> None:
    """Atomic write of soul_candidates.jsonl (the caller holds the file lock)."""
    write_jsonl(candidates_path, records, fsync=False, nonfinite="null")
