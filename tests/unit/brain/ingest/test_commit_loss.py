"""P0 regression: commit_item write failures must not silently lose turns.

When store.create raises (e.g. sqlite3.OperationalError: database is locked),
commit_item returns None. The pipeline must treat that as a non-clean ingest
and HOLD the buffer / cursor for retry, not delete it.

Dedupe (is_duplicate → True) is NOT a failure — retry is idempotent, already-
committed items hit is_duplicate on the next pass and cursor still advances.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import patch

from brain.bridge.provider import LLMProvider
from brain.ingest.buffer import ingest_turn, read_cursor, write_backoff
from brain.ingest.extract import ExtractionOutcome
from brain.ingest.pipeline import (
    _BACKOFF_FAILURE_THRESHOLD,
    close_session,
    extract_session_snapshot,
    finalize_stale_sessions,
)
from brain.ingest.types import ExtractedItem
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


class _NoopProvider(LLMProvider):
    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "[]"

    def name(self) -> str:
        return "noop"

    def chat(self, messages, *, tools=None, options=None):
        raise NotImplementedError


def _seed(persona_dir: Path, n: int = 3, sid: str = "s1") -> str:
    for i in range(n):
        ingest_turn(
            persona_dir,
            {
                "speaker": "user",
                "text": f"durable fact {i}",
                "session_id": sid,
                "ts": f"2026-01-01T12:0{i}:00+00:00",
            },
        )
    return sid


def test_close_session_retains_buffer_when_a_commit_fails(tmp_path: Path):
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    sid = _seed(persona_dir, 3)
    items = [
        ExtractedItem(text=f"durable fact {i}", label="fact", importance=5)
        for i in range(3)
    ]
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    real_create = store.create
    calls: dict[str, int] = {"n": 0}

    def flaky_create(mem):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("database is locked")
        return real_create(mem)

    try:
        with patch(
            "brain.ingest.pipeline.extract_items_with_status",
            return_value=ExtractionOutcome(items=items, failed=False, error=None),
        ), patch.object(store, "create", side_effect=flaky_create):
            report = close_session(
                persona_dir,
                sid,
                store=store,
                hebbian=hebbian,
                provider=_NoopProvider(),
            )
    finally:
        store.close()
        hebbian.close()

    assert report.commit_failures >= 1
    assert (
        persona_dir / "active_conversations" / f"{sid}.jsonl"
    ).exists(), "buffer must NOT be deleted when a commit failed"


# ---------------------------------------------------------------------------
# STEP 3a — extract_session_snapshot holds cursor when commit fails
# ---------------------------------------------------------------------------


def test_snapshot_holds_cursor_when_commit_fails(tmp_path: Path):
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    sid = _seed(persona_dir, 3)
    items = [
        ExtractedItem(text=f"durable fact {i}", label="fact", importance=5)
        for i in range(3)
    ]
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    cursor_before = read_cursor(persona_dir, sid)  # None — no cursor yet

    def always_raise(mem):
        raise RuntimeError("database is locked")

    try:
        with patch(
            "brain.ingest.pipeline.extract_items_with_status",
            return_value=ExtractionOutcome(items=items, failed=False, error=None),
        ), patch.object(store, "create", side_effect=always_raise):
            report = extract_session_snapshot(
                persona_dir,
                sid,
                store=store,
                hebbian=hebbian,
                provider=_NoopProvider(),
            )
    finally:
        store.close()
        hebbian.close()

    assert report.commit_failures >= 1
    cursor_after = read_cursor(persona_dir, sid)
    assert cursor_after == cursor_before, (
        "cursor must NOT advance when commit fails (held for retry)"
    )


# ---------------------------------------------------------------------------
# STEP 3b — cursor advances on pure dedupe (not a commit failure)
# ---------------------------------------------------------------------------


def test_snapshot_advances_cursor_on_dedupe(tmp_path: Path):
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    sid = _seed(persona_dir, 3)
    items = [
        ExtractedItem(text=f"durable fact {i}", label="fact", importance=5)
        for i in range(3)
    ]
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    try:
        with patch(
            "brain.ingest.pipeline.extract_items_with_status",
            return_value=ExtractionOutcome(items=items, failed=False, error=None),
        ), patch("brain.ingest.pipeline.is_duplicate", return_value=True):
            report = extract_session_snapshot(
                persona_dir,
                sid,
                store=store,
                hebbian=hebbian,
                provider=_NoopProvider(),
            )
    finally:
        store.close()
        hebbian.close()

    assert report.commit_failures == 0
    assert report.deduped == len(items)
    cursor_after = read_cursor(persona_dir, sid)
    assert cursor_after is not None, (
        "cursor MUST advance when all items are deduped (dedupe is not a failure)"
    )


# ---------------------------------------------------------------------------
# STEP 4a — finalize_stale_sessions holds buffer when commit fails
# ---------------------------------------------------------------------------


def test_finalize_holds_buffer_when_commit_fails(tmp_path: Path):
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    sid = _seed(persona_dir, 3)
    items = [
        ExtractedItem(text=f"durable fact {i}", label="fact", importance=5)
        for i in range(3)
    ]
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    def always_raise(mem):
        raise RuntimeError("database is locked")

    try:
        with patch(
            "brain.ingest.pipeline.extract_items_with_status",
            return_value=ExtractionOutcome(items=items, failed=False, error=None),
        ), patch.object(store, "create", side_effect=always_raise), patch(
            "brain.ingest.pipeline.session_silence_minutes",
            return_value=99999.0,
        ):
            reports = finalize_stale_sessions(
                persona_dir,
                store=store,
                hebbian=hebbian,
                provider=_NoopProvider(),
                finalize_after_hours=0.0,
            )
    finally:
        store.close()
        hebbian.close()

    assert len(reports) == 1
    assert reports[0].commit_failures >= 1
    assert (
        persona_dir / "active_conversations" / f"{sid}.jsonl"
    ).exists(), "buffer must NOT be deleted when finalize has commit failures"


# ---------------------------------------------------------------------------
# STEP 4b — finalize dead-letters after max retry threshold
# ---------------------------------------------------------------------------


def test_finalize_deadletters_after_max_retry(tmp_path: Path):
    persona_dir = tmp_path / "p"
    persona_dir.mkdir()
    sid = _seed(persona_dir, 3)
    items = [
        ExtractedItem(text=f"durable fact {i}", label="fact", importance=5)
        for i in range(3)
    ]
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")

    # Pre-write a backoff sidecar at the failure threshold
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()
    write_backoff(
        persona_dir,
        sid,
        failures=_BACKOFF_FAILURE_THRESHOLD,
        first_failure_at=now_iso,
    )

    def always_raise(mem):
        raise RuntimeError("database is locked")

    try:
        with patch(
            "brain.ingest.pipeline.extract_items_with_status",
            return_value=ExtractionOutcome(items=items, failed=False, error=None),
        ), patch.object(store, "create", side_effect=always_raise), patch(
            "brain.ingest.pipeline.session_silence_minutes",
            return_value=99999.0,
        ):
            reports = finalize_stale_sessions(
                persona_dir,
                store=store,
                hebbian=hebbian,
                provider=_NoopProvider(),
                finalize_after_hours=0.0,
            )
    finally:
        store.close()
        hebbian.close()

    assert len(reports) == 1
    original_buf = persona_dir / "active_conversations" / f"{sid}.jsonl"
    poison_buf = persona_dir / "active_conversations" / "poison" / f"{sid}.jsonl"
    assert not original_buf.exists(), "original buffer must be moved out of active_conversations/"
    assert poison_buf.exists(), "dead-lettered buffer must appear in poison/"
