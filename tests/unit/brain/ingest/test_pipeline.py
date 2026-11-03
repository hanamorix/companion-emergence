"""Tests for brain.ingest.pipeline — close_session, close_stale_sessions,
extract_session_snapshot, snapshot_stale_sessions, finalize_stale_sessions."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import LLMProvider
from brain.engines.consolidation import Decision, run_consolidation
from brain.health.json_io import append_jsonl
from brain.ingest import pipeline
from brain.ingest.buffer import (
    append_archive,
    ingest_turn,
    read_archive,
    read_backoff,
    read_cursor,
    read_session,
    rewrite_session_atomic,
    write_backoff,
    write_cursor,
)
from brain.ingest.pipeline import (
    close_session,
    close_stale_sessions,
    extract_session_snapshot,
    finalize_stale_sessions,
    snapshot_stale_sessions,
)
from brain.ingest.soul_queue import list_soul_candidates
from brain.ingest.types import VALID_LABELS
from brain.memory.hebbian import HebbianMatrix
from brain.memory.pending import PendingQueue
from brain.memory.store import MemoryStore

# memory-consolidation migration: gated ingest labels (fact/observation/…) are
# enqueued as candidates, not written to memories.db. Promote them through the
# consolidation gate (inject a promote-all classifier) so the pre-existing
# store.get() end-state assertions hold — the candidate id is preserved on
# promote, so report.memory_ids still resolves.
_PROMOTE_ALL = lambda _cand, _ctx: Decision("new")  # noqa: E731

# ---------------------------------------------------------------------------
# Fake providers for pipeline tests
# ---------------------------------------------------------------------------


class _CannedProvider(LLMProvider):
    """Returns a canned JSON extraction response."""

    def __init__(self, items: list[dict]) -> None:
        self._payload = json.dumps(items)

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return self._payload

    def name(self) -> str:
        return "fake-canned"

    def chat(self, messages: list[ChatMessage], *, tools=None, options=None) -> ChatResponse:
        return ChatResponse(content=self._payload, tool_calls=())


class _GarbageProvider(LLMProvider):
    """Always returns unparseable garbage."""

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "absolutely not json {]"

    def name(self) -> str:
        return "fake-garbage"

    def chat(self, messages: list[ChatMessage], *, tools=None, options=None) -> ChatResponse:
        return ChatResponse(content="garbage", tool_calls=())


class _TrackingProvider(LLMProvider):
    """Canned provider that also records what was sent.

    Used by snapshot tests that need to assert on which turns ended up in
    the extraction prompt (cursor-driven filtering).
    """

    def __init__(self, items: list[dict] | None = None) -> None:
        self._payload = json.dumps(items if items is not None else [])
        self.call_count = 0
        self.last_transcript: str | None = None

    def reset_calls(self) -> None:
        self.call_count = 0
        self.last_transcript = None

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.call_count += 1
        self.last_transcript = prompt
        return self._payload

    def name(self) -> str:
        return "fake-tracking"

    def chat(self, messages, *, tools=None, options=None):
        self.call_count += 1
        return ChatResponse(content=self._payload, tool_calls=())


@pytest.fixture
def tracking_provider() -> _TrackingProvider:
    return _TrackingProvider()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    # On-disk so store.persona_dir == the persona_dir passed to close_session:
    # gated ingest labels enqueue to <persona_dir>/pending_candidates.jsonl.
    return MemoryStore(tmp_path / "memories.db")


@pytest.fixture
def hebbian() -> HebbianMatrix:
    return HebbianMatrix(":memory:")


@pytest.fixture
def canned_provider() -> _CannedProvider:
    """Provider that returns 2 extraction items."""
    return _CannedProvider(
        [
            {"text": "Hana is a novelist", "label": "fact", "importance": 7},
            {"text": "Nell wears sweaters", "label": "observation", "importance": 5},
        ]
    )


# ---------------------------------------------------------------------------
# close_session tests
# ---------------------------------------------------------------------------


def test_close_session_on_missing_buffer_returns_empty_report(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, canned_provider: _CannedProvider
) -> None:
    """close_session on a nonexistent session returns a zero-count IngestReport."""
    report = close_session(
        tmp_path,
        "no_such_session",
        store=store,
        hebbian=hebbian,
        provider=canned_provider,
    )
    assert report.session_id == "no_such_session"
    assert report.extracted == 0
    assert report.committed == 0
    assert report.deduped == 0
    assert report.errors == 0
    assert report.memory_ids == []


def test_close_session_full_path_correct_counts(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """Full pipeline: 3 turns → extract 2 items → 0 deduped → 2 enqueued (both labels are
    gated, not in GATE_BYPASS_TYPES — #167: 'committed' now means durable-only)."""
    # Ingest 3 turns.
    for speaker, text in [
        ("Hana", "tell me something"),
        ("Nell", "I write dark fiction"),
        ("Hana", "nice"),
    ]:
        ingest_turn(tmp_path, {"session_id": "sess_full", "speaker": speaker, "text": text})

    # Provider returns 2 items.
    provider = _CannedProvider(
        [
            {"text": "Nell writes dark fiction", "label": "fact", "importance": 6},
            {"text": "Hana appreciates honesty", "label": "observation", "importance": 4},
        ]
    )
    report = close_session(
        tmp_path,
        "sess_full",
        store=store,
        hebbian=hebbian,
        provider=provider,
    )

    assert report.extracted == 2
    assert report.committed == 0
    assert report.enqueued == 2
    assert report.deduped == 0
    assert report.errors == 0
    assert len(report.memory_ids) == 2
    # Gated candidates are enqueued; promote them, then both memories exist.
    run_consolidation(store, persona_dir=store.persona_dir, classifier=_PROMOTE_ALL)
    for mid in report.memory_ids:
        assert store.get(mid) is not None


def test_close_session_deletes_buffer_after_run(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, canned_provider: _CannedProvider
) -> None:
    """close_session removes the buffer file after processing."""
    ingest_turn(tmp_path, {"session_id": "sess_del", "speaker": "Hana", "text": "hello"})
    buf_file = tmp_path / "active_conversations" / "sess_del.jsonl"
    assert buf_file.exists()

    close_session(tmp_path, "sess_del", store=store, hebbian=hebbian, provider=canned_provider)
    assert not buf_file.exists()


def test_close_session_soul_candidate_queued_when_importance_above_threshold(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """Items with importance >= 8 are queued as soul candidates."""
    ingest_turn(tmp_path, {"session_id": "sess_soul", "speaker": "Hana", "text": "I love you"})
    provider = _CannedProvider(
        [
            {"text": "Hana loves Nell deeply", "label": "feeling", "importance": 9},
            {"text": "minor note", "label": "note", "importance": 3},
        ]
    )
    report = close_session(
        tmp_path,
        "sess_soul",
        store=store,
        hebbian=hebbian,
        provider=provider,
    )

    assert report.soul_candidates == 1
    candidates = list_soul_candidates(tmp_path)
    assert len(candidates) == 1
    assert candidates[0]["text"] == "Hana loves Nell deeply"
    assert candidates[0]["status"] == "auto_pending"


def test_close_session_with_extraction_failure_counts_error_and_retains_buffer(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """Provider/parse failure is retryable: error counted, source buffer retained."""
    ingest_turn(tmp_path, {"session_id": "sess_fail", "speaker": "Hana", "text": "hi"})
    provider = _GarbageProvider()
    report = close_session(
        tmp_path,
        "sess_fail",
        store=store,
        hebbian=hebbian,
        provider=provider,
        config={"extraction_max_retries": 0},
    )

    assert report.extracted == 0
    assert report.committed == 0
    assert report.errors == 1
    assert (tmp_path / "active_conversations" / "sess_fail.jsonl").exists()
    assert read_archive(tmp_path, "sess_fail") == []  # a held close archives nothing (#555)


def test_close_session_valid_empty_extraction_deletes_buffer_without_error(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """A valid [] extraction is success-empty, not retryable failure."""
    ingest_turn(tmp_path, {"session_id": "sess_empty", "speaker": "Hana", "text": "hi"})
    provider = _CannedProvider([])

    report = close_session(
        tmp_path,
        "sess_empty",
        store=store,
        hebbian=hebbian,
        provider=provider,
        config={"extraction_max_retries": 0},
    )

    assert report.extracted == 0
    assert report.committed == 0
    assert report.errors == 0
    assert not (tmp_path / "active_conversations" / "sess_empty.jsonl").exists()


def test_close_session_memory_and_soul_candidate_both_written(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """High-importance item: enqueued as a candidate (gated label) AND queued as soul
    candidate."""
    ingest_turn(tmp_path, {"session_id": "sess_both", "speaker": "Hana", "text": "this matters"})
    provider = _CannedProvider(
        [{"text": "The most important truth", "label": "observation", "importance": 10}]
    )
    report = close_session(
        tmp_path,
        "sess_both",
        store=store,
        hebbian=hebbian,
        provider=provider,
    )

    assert report.committed == 0
    assert report.enqueued == 1
    assert report.soul_candidates == 1
    assert len(report.memory_ids) == 1

    mem_id = report.memory_ids[0]
    # Gated candidate — promote it, then it is a real memories.db row (id kept).
    run_consolidation(store, persona_dir=store.persona_dir, classifier=_PROMOTE_ALL)
    memory = store.get(mem_id)
    assert memory is not None
    assert memory.content == "The most important truth"

    candidates = list_soul_candidates(tmp_path)
    assert len(candidates) == 1
    assert candidates[0]["memory_id"] == mem_id


def test_close_session_counts_soul_queue_failure_without_claiming_success(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An enqueued (gated) memory is not reported as queued if the soul queue append fails."""
    ingest_turn(
        tmp_path, {"session_id": "sess_soul_fail", "speaker": "Hana", "text": "this matters"}
    )
    provider = _CannedProvider(
        [{"text": "The important truth was remembered", "label": "observation", "importance": 10}]
    )
    monkeypatch.setattr("brain.ingest.pipeline.queue_soul_candidate", lambda *a, **k: False)

    report = close_session(
        tmp_path,
        "sess_soul_fail",
        store=store,
        hebbian=hebbian,
        provider=provider,
    )

    assert report.committed == 0
    assert report.enqueued == 1
    assert report.soul_candidates == 0
    assert report.soul_queue_errors == 1
    assert list_soul_candidates(tmp_path) == []


# ---------------------------------------------------------------------------
# #167 — committed vs. enqueued telemetry
# ---------------------------------------------------------------------------


def test_close_session_gated_type_reports_enqueued_not_committed(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """C1: a gated (non-GATE_BYPASS_TYPES) label increments enqueued, not committed, and
    lands in pending_candidates.jsonl, not memories.db."""
    ingest_turn(tmp_path, {"session_id": "sess_gated", "speaker": "Hana", "text": "hi"})
    provider = _CannedProvider([{"text": "A gated fact", "label": "fact", "importance": 5}])
    report = close_session(
        tmp_path, "sess_gated", store=store, hebbian=hebbian, provider=provider
    )

    assert report.committed == 0
    assert report.enqueued == 1
    mem_id = report.memory_ids[0]
    assert store.get(mem_id) is None
    queued = PendingQueue(store.persona_dir).read_recent("fact", limit=5)
    assert any(m.id == mem_id for m in queued)


def test_close_session_bypass_type_reports_committed_not_enqueued(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C2 (round-3 revision, BT-dissolved — see changes/fix-167-committed-telemetry/decisions.md):
    real conversation-extraction can never natively carry a GATE_BYPASS_TYPES label — VALID_LABELS
    and GATE_BYPASS_TYPES are disjoint by design, and ExtractedItem.normalize() (called
    unconditionally at the SCORE stage) coerces any out-of-VALID_LABELS label to "observation"
    before it reaches the branch under test. This test widens VALID_LABELS for its own duration
    only, so a real "journal_entry"-labeled item survives SCORE intact and reaches the real,
    unmodified classification branch in pipeline.py — proving that branch is correct, not that
    real conversation traffic exercises it (it structurally cannot, today)."""
    monkeypatch.setattr(
        "brain.ingest.types.VALID_LABELS", frozenset({*VALID_LABELS, "journal_entry"})
    )
    ingest_turn(tmp_path, {"session_id": "sess_bypass", "speaker": "Hana", "text": "logging"})
    provider = _CannedProvider(
        [{"text": "A deliberate journal entry", "label": "journal_entry", "importance": 6}]
    )
    report = close_session(
        tmp_path, "sess_bypass", store=store, hebbian=hebbian, provider=provider
    )

    assert report.committed == 1
    assert report.enqueued == 0
    mem_id = report.memory_ids[0]
    memory = store.get(mem_id)
    assert memory is not None
    assert memory.content == "A deliberate journal entry"


def test_snapshot_bypass_type_reports_committed_not_enqueued(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C3: extract_session_snapshot's sibling of test_close_session_bypass_type_reports_
    committed_not_enqueued above — same VALID_LABELS-widening technique, same rationale."""
    monkeypatch.setattr(
        "brain.ingest.types.VALID_LABELS", frozenset({*VALID_LABELS, "journal_entry"})
    )
    ingest_turn(tmp_path, {"session_id": "sess_snap_bypass", "speaker": "Hana", "text": "logging"})
    provider = _CannedProvider(
        [{"text": "A deliberate journal entry (snapshot)", "label": "journal_entry", "importance": 6}]
    )
    report = extract_session_snapshot(
        tmp_path, "sess_snap_bypass", store=store, hebbian=hebbian, provider=provider
    )

    assert report.committed == 1
    assert report.enqueued == 0
    mem_id = report.memory_ids[0]
    memory = store.get(mem_id)
    assert memory is not None
    assert memory.content == "A deliberate journal entry (snapshot)"


def test_close_session_log_line_renders_enqueued_count(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, caplog: pytest.LogCaptureFixture
) -> None:
    """C12: the conversation_ingested log line actually renders enqueued=N, not just that the
    token 'enqueued' appears in the source format string."""
    ingest_turn(tmp_path, {"session_id": "sess_log", "speaker": "Hana", "text": "hi"})
    provider = _CannedProvider([{"text": "A gated fact", "label": "fact", "importance": 5}])
    with caplog.at_level(logging.INFO, logger="brain.ingest.pipeline"):
        close_session(tmp_path, "sess_log", store=store, hebbian=hebbian, provider=provider)

    ingested_records = [r for r in caplog.records if r.message.startswith("conversation_ingested")]
    assert len(ingested_records) == 1
    assert "committed=0" in ingested_records[0].message
    assert "enqueued=1" in ingested_records[0].message


def test_snapshot_log_line_renders_enqueued_count(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, caplog: pytest.LogCaptureFixture
) -> None:
    """C12: extract_session_snapshot's conversation_snapshot log line renders enqueued=N."""
    ingest_turn(tmp_path, {"session_id": "sess_snap_log", "speaker": "Hana", "text": "hi"})
    provider = _CannedProvider([{"text": "A gated fact", "label": "fact", "importance": 5}])
    with caplog.at_level(logging.INFO, logger="brain.ingest.pipeline"):
        extract_session_snapshot(
            tmp_path, "sess_snap_log", store=store, hebbian=hebbian, provider=provider
        )

    snapshot_records = [r for r in caplog.records if r.message.startswith("conversation_snapshot ")]
    assert len(snapshot_records) == 1
    assert "committed=0" in snapshot_records[0].message
    assert "enqueued=1" in snapshot_records[0].message


def test_finalize_log_line_renders_enqueued_count(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, caplog: pytest.LogCaptureFixture
) -> None:
    """C12: finalize_stale_sessions's conversation_finalized log line renders enqueued=N."""
    ingest_turn(tmp_path, {"session_id": "sess_final_log", "speaker": "Hana", "text": "hi"})
    old_ts = (datetime.now(UTC) - timedelta(hours=25)).isoformat(timespec="seconds")
    buf_file = tmp_path / "active_conversations" / "sess_final_log.jsonl"
    buf_file.write_text(
        json.dumps(
            {
                "session_id": "sess_final_log",
                "speaker": "Hana",
                "text": "hi",
                "ts": old_ts,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    provider = _CannedProvider([{"text": "A gated fact", "label": "fact", "importance": 5}])
    with caplog.at_level(logging.INFO, logger="brain.ingest.pipeline"):
        finalize_stale_sessions(
            tmp_path, finalize_after_hours=24.0, store=store, hebbian=hebbian, provider=provider
        )

    finalized_records = [r for r in caplog.records if r.message.startswith("conversation_finalized")]
    assert len(finalized_records) == 1
    assert "committed=0" in finalized_records[0].message
    assert "enqueued=1" in finalized_records[0].message


# ---------------------------------------------------------------------------
# close_stale_sessions tests
# ---------------------------------------------------------------------------


def test_close_stale_sessions_ignores_fresh_sessions(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, canned_provider: _CannedProvider
) -> None:
    """close_stale_sessions does not close sessions whose last turn is recent."""
    # Ingest a turn with a current timestamp (fresh).
    ingest_turn(tmp_path, {"session_id": "sess_fresh", "speaker": "Hana", "text": "just now"})

    reports = close_stale_sessions(
        tmp_path,
        silence_minutes=60.0,  # 1-hour threshold — fresh session won't match.
        store=store,
        hebbian=hebbian,
        provider=canned_provider,
    )

    assert reports == []
    # Buffer should still exist.
    assert (tmp_path / "active_conversations" / "sess_fresh.jsonl").exists()


def test_close_stale_sessions_closes_old_sessions(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """close_stale_sessions closes sessions whose last turn is older than threshold."""
    # Write a buffer file with a past timestamp directly (bypassing ingest_turn's auto-ts).
    buf_dir = tmp_path / "active_conversations"
    buf_dir.mkdir(parents=True)
    old_ts = (datetime.now(UTC) - timedelta(minutes=30)).isoformat(timespec="seconds")
    buf_file = buf_dir / "sess_old.jsonl"
    buf_file.write_text(
        json.dumps({"session_id": "sess_old", "speaker": "Nell", "text": "old turn", "ts": old_ts})
        + "\n",
        encoding="utf-8",
    )

    provider = _CannedProvider(
        [{"text": "Nell thought about something long ago", "label": "observation", "importance": 4}]
    )
    reports = close_stale_sessions(
        tmp_path,
        silence_minutes=5.0,
        store=store,
        hebbian=hebbian,
        provider=provider,
    )

    assert len(reports) == 1
    assert reports[0].session_id == "sess_old"
    assert reports[0].committed >= 0  # May be 0 or 1 depending on extraction.
    # Buffer should be cleaned up.
    assert not buf_file.exists()


# ---- Bug A (audit-3): close_session threads user_name through to extract ----


def test_close_session_passes_named_speakers_to_extract(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """When persona_config.user_name is set, close_session must pass it
    plus the persona name to the extractor so the LLM sees a named
    transcript and the disambiguation prompt header. Bug A regression.

    The persona dir name is the assistant_name (always known); user_name
    comes from persona_config.json — when None, falls through to legacy."""
    from brain.persona_config import PersonaConfig

    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    PersonaConfig(user_name="Hana").save(persona_dir / "persona_config.json")

    # Buffer with both speaker types
    ingest_turn(persona_dir, {"session_id": "sess_named", "speaker": "user", "text": "hi"})
    ingest_turn(
        persona_dir, {"session_id": "sess_named", "speaker": "assistant", "text": "hey love"}
    )

    captured: list[str] = []

    class _CapProvider(LLMProvider):
        def name(self):
            return "cap"

        def healthy(self):
            return True

        def chat(self, *a, **kw):
            raise NotImplementedError

        def generate(self, prompt, *, system=None):
            captured.append(prompt)
            return "[]"  # empty extraction valid

    close_session(persona_dir, "sess_named", store=store, hebbian=hebbian, provider=_CapProvider())

    assert len(captured) == 1
    p = captured[0]
    # Named-prompt disambiguation header present
    assert "Hana is the human user" in p
    assert "nell is the assistant" in p
    # Transcript uses real speaker names instead of generic labels
    assert "Hana: hi" in p
    assert "nell: hey love" in p


def test_close_session_falls_back_to_legacy_when_user_name_unset(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """No persona_config.json (or user_name unset) → legacy prompt path.
    Forkers who haven't set the field don't hit a regression."""
    persona_dir = tmp_path / "nell"
    persona_dir.mkdir()
    # No persona_config.json on disk

    ingest_turn(persona_dir, {"session_id": "sess_legacy", "speaker": "user", "text": "hi"})

    captured: list[str] = []

    class _CapProvider(LLMProvider):
        def name(self):
            return "cap"

        def healthy(self):
            return True

        def chat(self, *a, **kw):
            raise NotImplementedError

        def generate(self, prompt, *, system=None):
            captured.append(prompt)
            return "[]"

    close_session(persona_dir, "sess_legacy", store=store, hebbian=hebbian, provider=_CapProvider())

    assert len(captured) == 1
    p = captured[0]
    # Legacy path — no disambiguation header
    assert "is the human user" not in p
    # Generic 'user:' label preserved
    assert "user: hi" in p


# ---------------------------------------------------------------------------
# extract_session_snapshot tests
# ---------------------------------------------------------------------------


def test_snapshot_preserves_buffer_and_writes_cursor(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    tracking_provider: _TrackingProvider,
) -> None:
    sid = "sess_abc"
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "I love watercolour"})
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "assistant", "text": "noted"})

    report = extract_session_snapshot(
        tmp_path,
        sid,
        store=store,
        hebbian=hebbian,
        provider=tracking_provider,
    )

    buf = tmp_path / "active_conversations" / f"{sid}.jsonl"
    assert buf.exists(), "snapshot must NOT delete the buffer"
    assert report.session_id == sid
    cursor = read_cursor(tmp_path, sid)
    assert cursor is not None, "snapshot must write the cursor"


def test_snapshot_with_cursor_only_extracts_new_turns(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    tracking_provider: _TrackingProvider,
) -> None:
    sid = "sess_abc"
    ingest_turn(
        tmp_path,
        {"session_id": sid, "speaker": "user", "text": "old", "ts": "2026-05-10T20:00:00+00:00"},
    )
    write_cursor(tmp_path, sid, "2026-05-10T20:00:00+00:00")
    ingest_turn(
        tmp_path,
        {"session_id": sid, "speaker": "user", "text": "new", "ts": "2026-05-10T20:05:00+00:00"},
    )

    tracking_provider.reset_calls()
    extract_session_snapshot(
        tmp_path,
        sid,
        store=store,
        hebbian=hebbian,
        provider=tracking_provider,
    )

    assert tracking_provider.last_transcript is not None
    assert "new" in tracking_provider.last_transcript
    assert "old" not in tracking_provider.last_transcript


def test_snapshot_with_no_new_turns_skips_provider_call(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    tracking_provider: _TrackingProvider,
) -> None:
    sid = "sess_abc"
    ingest_turn(
        tmp_path,
        {"session_id": sid, "speaker": "user", "text": "x", "ts": "2026-05-10T20:00:00+00:00"},
    )
    write_cursor(tmp_path, sid, "2026-05-10T20:00:00+00:00")

    tracking_provider.reset_calls()
    report = extract_session_snapshot(
        tmp_path,
        sid,
        store=store,
        hebbian=hebbian,
        provider=tracking_provider,
    )

    assert tracking_provider.call_count == 0
    assert report.extracted == 0


def test_snapshot_malformed_cursor_falls_back_to_full(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    tracking_provider: _TrackingProvider,
) -> None:
    sid = "sess_abc"
    (tmp_path / "active_conversations").mkdir(parents=True, exist_ok=True)
    (tmp_path / "active_conversations" / f"{sid}.cursor").write_text("garbage")
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "hello"})

    tracking_provider.reset_calls()
    extract_session_snapshot(
        tmp_path,
        sid,
        store=store,
        hebbian=hebbian,
        provider=tracking_provider,
    )

    assert tracking_provider.last_transcript is not None
    assert "hello" in tracking_provider.last_transcript


def test_snapshot_on_empty_buffer_returns_empty_report(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    tracking_provider: _TrackingProvider,
) -> None:
    sid = "sess_abc"
    (tmp_path / "active_conversations").mkdir(parents=True, exist_ok=True)
    (tmp_path / "active_conversations" / f"{sid}.jsonl").touch()

    tracking_provider.reset_calls()
    report = extract_session_snapshot(
        tmp_path,
        sid,
        store=store,
        hebbian=hebbian,
        provider=tracking_provider,
    )

    assert report.extracted == 0
    assert tracking_provider.call_count == 0


# ---------------------------------------------------------------------------
# snapshot_stale_sessions tests
# ---------------------------------------------------------------------------


def test_snapshot_stale_sessions_keeps_buffer_alive(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    tracking_provider: _TrackingProvider,
) -> None:
    sid = "sess_abc"
    old_ts = (datetime.now(UTC) - timedelta(minutes=6)).isoformat()
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "earlier", "ts": old_ts})

    reports = snapshot_stale_sessions(
        tmp_path,
        silence_minutes=5.0,
        store=store,
        hebbian=hebbian,
        provider=tracking_provider,
    )

    assert len(reports) == 1
    assert reports[0].session_id == sid
    buf = tmp_path / "active_conversations" / f"{sid}.jsonl"
    assert buf.exists(), "snapshot sweep must NOT delete the buffer"


def test_snapshot_stale_sessions_skips_fresh_sessions(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    tracking_provider: _TrackingProvider,
) -> None:
    sid = "sess_abc"
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "just now"})

    reports = snapshot_stale_sessions(
        tmp_path,
        silence_minutes=5.0,
        store=store,
        hebbian=hebbian,
        provider=tracking_provider,
    )

    assert reports == []


def test_snapshot_stale_sessions_respects_ten_minute_threshold(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    tracking_provider: _TrackingProvider,
) -> None:
    """Issue #132 Part B (C14 test 2, regression guard): a fixture session idle 9
    minutes must NOT be extracted; one idle just over 10 minutes must. This
    exercises snapshot_stale_sessions's pre-existing, unchanged gate-comparison
    logic — it does not itself prove the production default changed (that is
    `test_run_folded_and_build_app_no_longer_take_silence_minutes` in
    tests/unit/brain/bridge/test_supervisor_db_overhead.py); it confirms the
    mechanism the new default now feeds into still works correctly.
    """
    fresh_sid = "sess_nine_min"
    fresh_ts = (datetime.now(UTC) - timedelta(minutes=9)).isoformat()
    ingest_turn(
        tmp_path, {"session_id": fresh_sid, "speaker": "user", "text": "still here", "ts": fresh_ts}
    )

    stale_sid = "sess_ten_min"
    stale_ts = (datetime.now(UTC) - timedelta(minutes=10, seconds=5)).isoformat()
    ingest_turn(
        tmp_path,
        {"session_id": stale_sid, "speaker": "user", "text": "long gone", "ts": stale_ts},
    )

    reports = snapshot_stale_sessions(
        tmp_path,
        silence_minutes=10.0,
        store=store,
        hebbian=hebbian,
        provider=tracking_provider,
    )

    assert [r.session_id for r in reports] == [stale_sid]
    assert tracking_provider.call_count == 1


def test_snapshot_stale_sessions_cleans_ghost_buffer(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    tracking_provider: _TrackingProvider,
) -> None:
    """A buffer file with no readable turns (corrupt / never-written) is
    silently cleaned up without generating a report."""
    sid = "sess_ghost"
    (tmp_path / "active_conversations").mkdir(parents=True, exist_ok=True)
    (tmp_path / "active_conversations" / f"{sid}.jsonl").touch()

    reports = snapshot_stale_sessions(
        tmp_path,
        silence_minutes=5.0,
        store=store,
        hebbian=hebbian,
        provider=tracking_provider,
    )

    assert reports == []
    buf = tmp_path / "active_conversations" / f"{sid}.jsonl"
    assert not buf.exists(), "ghost buffer should be cleaned up"


def test_snapshot_stale_sessions_per_session_error_isolation(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    tracking_provider: _TrackingProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If reading one session's buffer raises (corrupt buffer / FS error),
    the sweep must continue to the next session — not abort the whole loop."""
    sid_bad = "sess_bad"
    sid_good = "sess_good"
    old = (datetime.now(UTC) - timedelta(minutes=6)).isoformat()
    ingest_turn(tmp_path, {"session_id": sid_bad, "speaker": "user", "text": "boom", "ts": old})
    ingest_turn(tmp_path, {"session_id": sid_good, "speaker": "user", "text": "ok", "ts": old})

    import brain.ingest.pipeline as pipeline_mod

    real_read = pipeline_mod.read_session

    def selective_raise(persona_dir, session_id):
        if session_id == sid_bad:
            raise OSError("simulated disk error")
        return real_read(persona_dir, session_id)

    monkeypatch.setattr(pipeline_mod, "read_session", selective_raise)

    reports = snapshot_stale_sessions(
        tmp_path,
        silence_minutes=5.0,
        store=store,
        hebbian=hebbian,
        provider=tracking_provider,
    )

    # The good session still produced a report; the bad one was skipped
    # via the per-session try/except and the loop continued.
    sids_reported = {r.session_id for r in reports}
    assert sid_good in sids_reported, "good session should still be snapshotted"
    assert sid_bad not in sids_reported, "bad session should NOT produce a report"


# ---------------------------------------------------------------------------
# finalize_stale_sessions tests
# ---------------------------------------------------------------------------


def test_finalize_under_threshold_skips(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    tracking_provider: _TrackingProvider,
) -> None:
    sid = "sess_abc"
    recent = (datetime.now(UTC) - timedelta(hours=10)).isoformat()
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "x", "ts": recent})

    reports = finalize_stale_sessions(
        tmp_path,
        finalize_after_hours=24.0,
        store=store,
        hebbian=hebbian,
        provider=tracking_provider,
    )

    assert reports == []
    buf = tmp_path / "active_conversations" / f"{sid}.jsonl"
    assert buf.exists()


def test_finalize_at_threshold_extracts_but_keeps_buffer(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    tracking_provider: _TrackingProvider,
) -> None:
    # Cascade-compaction 1c (stage-3 L2): finalize is now EXTRACTION-ONLY — it runs
    # the final extraction but no longer deletes the buffer/cursor. The rollover path
    # owns buffer deletion (a still-live session's buffer stays until it rolls over).
    sid = "sess_abc"
    old = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "user", "text": "x", "ts": old})
    write_cursor(tmp_path, sid, old)

    reports = finalize_stale_sessions(
        tmp_path,
        finalize_after_hours=24.0,
        store=store,
        hebbian=hebbian,
        provider=tracking_provider,
    )

    assert len(reports) == 1
    assert reports[0].session_id == sid
    buf = tmp_path / "active_conversations" / f"{sid}.jsonl"
    assert buf.exists(), "finalize is extraction-only — it must NOT delete the buffer"


def test_finalize_per_session_error_isolation(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
) -> None:
    """A provider that raises on the first call shouldn't abort the whole sweep —
    both stale sessions must still be reported and cleaned up."""

    class _ExplodingOnceProvider(LLMProvider):
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, prompt: str, *, system: str | None = None) -> str:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("model down")
            return "[]"

        def name(self) -> str:
            return "fake-exploding"

        def chat(self, messages, *, tools=None, options=None):
            return ChatResponse(content="[]", tool_calls=())

    sid_a = "sess_aaa"
    sid_b = "sess_bbb"
    old = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    ingest_turn(tmp_path, {"session_id": sid_a, "speaker": "user", "text": "a", "ts": old})
    ingest_turn(tmp_path, {"session_id": sid_b, "speaker": "user", "text": "b", "ts": old})

    reports = finalize_stale_sessions(
        tmp_path,
        finalize_after_hours=24.0,
        store=store,
        hebbian=hebbian,
        provider=_ExplodingOnceProvider(),
    )

    # Both stale sessions reach finalize regardless of the first one exploding.
    assert len(reports) == 2
    # Extraction-only finalize keeps both buffers (rollover owns deletion); the
    # point here is error isolation — one session raising doesn't abort the sweep.
    for sid in (sid_a, sid_b):
        buf = tmp_path / "active_conversations" / f"{sid}.jsonl"
        assert buf.exists()


# ---------------------------------------------------------------------------
# close_session cursor cleanup tests
# ---------------------------------------------------------------------------


def test_close_session_full_path_also_deletes_cursor(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
) -> None:
    """Explicit close must clean up the cursor sidecar alongside the buffer."""
    sid = "sess_close_cur"
    for speaker, text in [("Hana", "hi"), ("Nell", "hello"), ("Hana", "bye")]:
        ingest_turn(tmp_path, {"session_id": sid, "speaker": speaker, "text": text})
    write_cursor(tmp_path, sid, datetime.now(UTC).isoformat())

    provider = _CannedProvider(
        [
            {"text": "test memory", "label": "fact", "importance": 5},
        ]
    )
    close_session(tmp_path, sid, store=store, hebbian=hebbian, provider=provider)

    buf = tmp_path / "active_conversations" / f"{sid}.jsonl"
    cursor_file = tmp_path / "active_conversations" / f"{sid}.cursor"
    assert not buf.exists()
    assert not cursor_file.exists()


def test_close_session_empty_path_also_deletes_cursor(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    canned_provider: _CannedProvider,
) -> None:
    """Empty-session early-return path must also clean up the cursor."""
    sid = "sess_empty_cur"
    # Empty buffer file + stale cursor.
    (tmp_path / "active_conversations").mkdir(parents=True, exist_ok=True)
    (tmp_path / "active_conversations" / f"{sid}.jsonl").touch()
    write_cursor(tmp_path, sid, datetime.now(UTC).isoformat())

    close_session(tmp_path, sid, store=store, hebbian=hebbian, provider=canned_provider)

    cursor_file = tmp_path / "active_conversations" / f"{sid}.cursor"
    assert not cursor_file.exists()
    archive_dir = tmp_path / "archived_conversations"
    assert not archive_dir.exists() or not any(archive_dir.iterdir())  # an empty buffer writes no archive file (#555)


# ---------------------------------------------------------------------------
# F-011 — extraction backoff
# ---------------------------------------------------------------------------


def test_snapshot_records_backoff_on_extraction_failure(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
) -> None:
    """First failed extraction writes the backoff sidecar with failures=1."""
    sid = "sess_backoff1"
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "Hana", "text": "hi"})

    provider = _GarbageProvider()
    report = extract_session_snapshot(
        tmp_path,
        sid,
        store=store,
        hebbian=hebbian,
        provider=provider,
        config={"extraction_max_retries": 0},
    )

    assert report.errors == 1
    state = read_backoff(tmp_path, sid)
    assert state is not None
    assert state["failures"] == 1
    # first_failure_at must be a parseable ISO ts.
    datetime.fromisoformat(state["first_failure_at"].replace("Z", "+00:00"))


def test_snapshot_backoff_increments_anchor_stays(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
) -> None:
    """A second failure increments failures but does NOT move first_failure_at."""
    sid = "sess_backoff_inc"
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "Hana", "text": "hi"})
    anchor = "2026-05-10T19:00:00+00:00"
    write_backoff(tmp_path, sid, failures=1, first_failure_at=anchor)

    provider = _GarbageProvider()
    extract_session_snapshot(
        tmp_path,
        sid,
        store=store,
        hebbian=hebbian,
        provider=provider,
        config={"extraction_max_retries": 0},
    )

    state = read_backoff(tmp_path, sid)
    assert state is not None
    assert (state["failures"], state["first_failure_at"]) == (2, anchor)
    assert state["failing_end_ts"]  # 524-4: the failing chunk's end rides along with the count


def test_snapshot_backoff_skips_after_threshold_within_window(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    tracking_provider: _TrackingProvider,
) -> None:
    """At failures=3 within the 10-min window, extraction is skipped entirely."""
    sid = "sess_backoff_skip"
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "Hana", "text": "hi"})
    # Anchor inside the window.
    now = datetime.now(UTC).isoformat()
    write_backoff(tmp_path, sid, failures=3, first_failure_at=now)

    report = extract_session_snapshot(
        tmp_path,
        sid,
        store=store,
        hebbian=hebbian,
        provider=tracking_provider,
    )

    # Provider must NOT be called.
    assert tracking_provider.call_count == 0
    assert report.errors == 1
    assert report.extracted == 0
    # Backoff state is preserved (only success clears it).
    state = read_backoff(tmp_path, sid)
    assert state is not None
    assert state["failures"] == 3


def test_snapshot_backoff_resumes_after_window_expires(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    tracking_provider: _TrackingProvider,
) -> None:
    """When the 10-min window has elapsed, extraction runs again."""
    sid = "sess_backoff_resume"
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "Hana", "text": "hi"})
    # Anchor well outside the window.
    stale = (datetime.now(UTC) - timedelta(minutes=30)).isoformat()
    write_backoff(tmp_path, sid, failures=5, first_failure_at=stale)

    extract_session_snapshot(
        tmp_path,
        sid,
        store=store,
        hebbian=hebbian,
        provider=tracking_provider,
    )

    # Provider WAS called this time.
    assert tracking_provider.call_count == 1
    # Success cleared the backoff state.
    assert read_backoff(tmp_path, sid) is None


def test_snapshot_backoff_clears_on_success(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
) -> None:
    """A successful extraction deletes any prior backoff state."""
    sid = "sess_backoff_clear"
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "Hana", "text": "hi"})
    # Pre-existing backoff state under the threshold so guard doesn't fire.
    write_backoff(
        tmp_path,
        sid,
        failures=2,
        first_failure_at=datetime.now(UTC).isoformat(),
    )

    provider = _CannedProvider([{"text": "Hana said hi", "label": "fact", "importance": 4}])
    report = extract_session_snapshot(
        tmp_path,
        sid,
        store=store,
        hebbian=hebbian,
        provider=provider,
    )

    assert report.errors == 0
    assert report.committed == 0
    assert report.enqueued >= 1
    assert read_backoff(tmp_path, sid) is None


def test_close_session_deletes_backoff(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    canned_provider: _CannedProvider,
) -> None:
    """close_session cleans up the backoff sidecar along with buffer + cursor."""
    sid = "sess_close_backoff"
    ingest_turn(tmp_path, {"session_id": sid, "speaker": "Hana", "text": "hi"})
    write_backoff(
        tmp_path,
        sid,
        failures=2,
        first_failure_at=datetime.now(UTC).isoformat(),
    )

    close_session(tmp_path, sid, store=store, hebbian=hebbian, provider=canned_provider)

    assert read_backoff(tmp_path, sid) is None
    backoff_file = tmp_path / "active_conversations" / f"{sid}.backoff"
    assert not backoff_file.exists()


def test_close_session_empty_path_deletes_backoff(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    canned_provider: _CannedProvider,
) -> None:
    """The empty-buffer early-return path also clears backoff state."""
    sid = "sess_close_empty_backoff"
    (tmp_path / "active_conversations").mkdir(parents=True, exist_ok=True)
    (tmp_path / "active_conversations" / f"{sid}.jsonl").touch()
    write_backoff(
        tmp_path,
        sid,
        failures=3,
        first_failure_at=datetime.now(UTC).isoformat(),
    )

    close_session(tmp_path, sid, store=store, hebbian=hebbian, provider=canned_provider)

    assert read_backoff(tmp_path, sid) is None


def test_finalize_stale_sessions_deletes_backoff(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    canned_provider: _CannedProvider,
) -> None:
    """finalize_stale_sessions must scrub the backoff sidecar too."""
    sid = "sess_final_backoff"
    # Ingest with a stale ts so silence threshold trips.
    stale_ts = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
    ingest_turn(
        tmp_path,
        {"session_id": sid, "speaker": "Hana", "text": "hi", "ts": stale_ts},
    )
    write_backoff(
        tmp_path,
        sid,
        failures=2,
        first_failure_at=datetime.now(UTC).isoformat(),
    )

    finalize_stale_sessions(
        tmp_path,
        finalize_after_hours=24.0,
        store=store,
        hebbian=hebbian,
        provider=canned_provider,
    )

    assert read_backoff(tmp_path, sid) is None
    backoff_file = tmp_path / "active_conversations" / f"{sid}.backoff"
    assert not backoff_file.exists()


# ---------------------------------------------------------------------------
# F1 (#259) increment 5 — the evict-on-failed-commit dance is gone. Dedupe no
# longer writes a candidate vector anywhere (it embeds the candidate
# directly, with nothing to write or evict), so a failed commit has nothing
# to roll back. The dead `embeddings` kwarg itself (accepted-but-unused for
# call-site compatibility through increment 5-7) is removed in the F1
# increment 8 code teardown — these tests now just confirm a failed commit
# is counted correctly, with no cache/evict machinery in the picture at all.
# ---------------------------------------------------------------------------


def test_close_session_commit_failure_does_not_evict(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    canned_provider: _CannedProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed commit_item() must still be counted as an error/commit_failure.
    There is no candidate cache row to roll back anymore (no `embeddings`
    kwarg, no evict machinery)."""
    import brain.ingest.pipeline as pipeline_mod

    ingest_turn(tmp_path, {"session_id": "sess_commit_fail", "speaker": "Hana", "text": "hi"})
    monkeypatch.setattr(pipeline_mod, "commit_item", lambda *args, **kwargs: None)

    report = close_session(
        tmp_path,
        "sess_commit_fail",
        store=store,
        hebbian=hebbian,
        provider=canned_provider,
    )

    assert report.errors == 2
    assert report.commit_failures == 2
    assert report.memory_ids == []


def test_snapshot_commit_failure_does_not_evict(
    tmp_path: Path,
    store: MemoryStore,
    hebbian: HebbianMatrix,
    canned_provider: _CannedProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same guarantee for extract_session_snapshot's identical DEDUPE+COMMIT
    loop."""
    import brain.ingest.pipeline as pipeline_mod

    ingest_turn(tmp_path, {"session_id": "sess_snap_commit_fail", "speaker": "Hana", "text": "hi"})
    monkeypatch.setattr(pipeline_mod, "commit_item", lambda *args, **kwargs: None)

    report = extract_session_snapshot(
        tmp_path,
        "sess_snap_commit_fail",
        store=store,
        hebbian=hebbian,
        provider=canned_provider,
    )

    assert report.errors == 2
    assert report.commit_failures == 2
    assert report.memory_ids == []


# ── #364: an unparseable extraction keeps every attempt's reply ───────────────
def _parse_failure_rows(persona_dir: Path) -> list[dict]:
    from brain.health.parse_failures import PARSE_FAILURE_LOG

    path = persona_dir / PARSE_FAILURE_LOG
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def test_close_session_records_each_unparseable_extraction_attempt(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    ingest_turn(tmp_path, {"session_id": "sess_pf", "speaker": "Hana", "text": "hi"})
    close_session(
        tmp_path, "sess_pf", store=store, hebbian=hebbian,
        provider=_GarbageProvider(), config={"extraction_max_retries": 1},
    )
    rows = _parse_failure_rows(tmp_path)
    assert [(r["site"], r["raw"]) for r in rows] == [("ingest_extract", "absolutely not json {]")] * 2


def test_session_snapshot_records_an_unparseable_extraction(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    from brain.ingest.pipeline import extract_session_snapshot

    ingest_turn(tmp_path, {"session_id": "sess_pf2", "speaker": "Hana", "text": "hi"})
    extract_session_snapshot(
        tmp_path, "sess_pf2", store=store, hebbian=hebbian,
        provider=_GarbageProvider(), config={"extraction_max_retries": 0},
    )
    assert [r["site"] for r in _parse_failure_rows(tmp_path)] == ["ingest_extract"]


# ---------------------------------------------------------------------------
# close_session archives the buffer before it removes it (#555)
# ---------------------------------------------------------------------------

_ARCHIVE_SID = "sess_arch"


def _archive_rows(sid: str, count: int, *, start: datetime, with_summary: bool = False) -> list[dict]:
    """``count`` alternating user/assistant rows with fixed timestamps, optionally led by a summary row."""
    rows: list[dict] = []
    if with_summary:
        rows.append(
            {
                "session_id": sid,
                "speaker": "summary",
                "text": "They talked about gardens.",
                "ts": start.isoformat(timespec="seconds"),
                "compaction": {"gen": 1, "folded": False, "covers_until_ts": start.isoformat(timespec="seconds")},
            }
        )
    for i in range(count):
        rows.append(
            {
                "session_id": sid,
                "speaker": "user" if i % 2 == 0 else "assistant",
                "text": f"row {i} of {sid}",
                "ts": (start + timedelta(seconds=10 * (i + 1))).isoformat(timespec="seconds"),
            }
        )
    return rows


def _seed_closable(persona: Path, sid: str, rows: list[dict]) -> list[dict]:
    """Write ``rows`` as the buffer, a cursor at the last non-summary row (extraction is a no-op) and a backoff sidecar."""
    rewrite_session_atomic(persona, sid, rows)
    write_cursor(persona, sid, [r for r in rows if r["speaker"] != "summary"][-1]["ts"])
    write_backoff(persona, sid, failures=1, first_failure_at=datetime.now(UTC).isoformat())
    return read_session(persona, sid)


def _close_noop(persona: Path, sid: str, store: MemoryStore, hebbian: HebbianMatrix):
    return close_session(persona, sid, store=store, hebbian=hebbian, provider=_CannedProvider([]))


def _sidecars(persona: Path, sid: str) -> dict[str, Path]:
    d = persona / "active_conversations"
    return {"buffer": d / f"{sid}.jsonl", "cursor": d / f"{sid}.cursor", "backoff": d / f"{sid}.backoff"}


def test_close_archives_every_buffer_row_before_removing_it(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """C1: user, assistant and summary rows, those at or before the cursor and those after it, reach the archive verbatim."""
    start = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
    rows = _seed_closable(tmp_path, _ARCHIVE_SID, _archive_rows(_ARCHIVE_SID, 6, start=start, with_summary=True))
    # a cursor in the middle: rows after it are extracted by this close, rows at or before it never are
    write_cursor(tmp_path, _ARCHIVE_SID, rows[3]["ts"])

    report = _close_noop(tmp_path, _ARCHIVE_SID, store, hebbian)

    assert report.reached_end
    assert read_archive(tmp_path, _ARCHIVE_SID) == rows
    assert not any(p.exists() for p in _sidecars(tmp_path, _ARCHIVE_SID).values())


def test_close_keeps_an_earlier_archived_batch_first(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """C1: what compaction left in the archive stays intact and first; the close's rows follow it."""
    start = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
    earlier = _archive_rows(_ARCHIVE_SID, 3, start=start - timedelta(hours=2))
    append_archive(tmp_path, _ARCHIVE_SID, earlier)
    rows = _seed_closable(tmp_path, _ARCHIVE_SID, _archive_rows(_ARCHIVE_SID, 4, start=start, with_summary=True))

    assert _close_noop(tmp_path, _ARCHIVE_SID, store, hebbian).reached_end

    assert read_archive(tmp_path, _ARCHIVE_SID) == earlier + rows
    assert not _sidecars(tmp_path, _ARCHIVE_SID)["buffer"].exists()


def test_close_after_a_real_compaction_archives_every_row_once(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """C1: the realistic order. Compaction folds and archives the old rows; the close archives what compaction left (summary + newest 40)."""
    from brain.chat.compaction import compact_conversation

    class _SummaryProvider:
        def generate(self, prompt=None, *, system=None, **kw) -> str:
            return "They talked about gardens for a long while."

    start = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)
    original = _archive_rows(_ARCHIVE_SID, 120, start=start)
    _seed_closable(tmp_path, _ARCHIVE_SID, original)
    result = compact_conversation(
        tmp_path, _ARCHIVE_SID, older_than=timedelta(0), fold_existing_summary=False, provider=_SummaryProvider()
    )
    assert result.compacted
    after_compaction = read_archive(tmp_path, _ARCHIVE_SID)
    left_in_buffer = read_session(tmp_path, _ARCHIVE_SID)
    assert len(after_compaction) == 80 and len(left_in_buffer) == 41  # 80 folded, summary + newest 40 kept

    assert _close_noop(tmp_path, _ARCHIVE_SID, store, hebbian).reached_end

    archive = read_archive(tmp_path, _ARCHIVE_SID)
    assert archive[: len(after_compaction)] == after_compaction  # compaction's batch first, intact
    assert archive[len(after_compaction) :] == left_in_buffer  # the close's batch after it
    assert [r["text"] for r in archive if r["speaker"] != "summary"] == [r["text"] for r in original]  # each turn once, in order


def test_close_has_archived_everything_when_it_removes_the_buffer(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C2: at the instant the buffer is removed the archive already holds all of its rows."""
    rows = _seed_closable(
        tmp_path, _ARCHIVE_SID, _archive_rows(_ARCHIVE_SID, 4, start=datetime(2026, 10, 1, 9, 0, tzinfo=UTC))
    )
    seen_at_delete: list[list[dict]] = []
    real_delete = pipeline.delete_session_buffer

    def _delete(persona_dir, session_id):
        seen_at_delete.append(read_archive(persona_dir, session_id))
        real_delete(persona_dir, session_id)

    monkeypatch.setattr(pipeline, "delete_session_buffer", _delete)

    _close_noop(tmp_path, _ARCHIVE_SID, store, hebbian)

    assert seen_at_delete == [rows]


def test_close_keeps_buffer_cursor_and_backoff_when_the_archive_write_fails(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C3: an archive failure propagates before anything is removed; the three files are byte-identical."""
    _seed_closable(tmp_path, _ARCHIVE_SID, _archive_rows(_ARCHIVE_SID, 4, start=datetime(2026, 10, 1, 9, 0, tzinfo=UTC)))
    paths = _sidecars(tmp_path, _ARCHIVE_SID)
    before = {name: p.read_bytes() for name, p in paths.items()}

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(pipeline, "archive_once", _boom)

    with pytest.raises(OSError, match="disk full"):
        _close_noop(tmp_path, _ARCHIVE_SID, store, hebbian)

    assert {name: p.read_bytes() for name, p in paths.items()} == before


def test_close_keeps_the_buffer_when_the_archive_reports_zero_bytes_written(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An append that reports zero bytes for a non-empty batch is not a durable archive: the close raises and removes nothing."""
    _seed_closable(tmp_path, _ARCHIVE_SID, _archive_rows(_ARCHIVE_SID, 4, start=datetime(2026, 10, 1, 9, 0, tzinfo=UTC)))
    paths = _sidecars(tmp_path, _ARCHIVE_SID)
    before = {name: p.read_bytes() for name, p in paths.items()}
    monkeypatch.setattr(pipeline, "archive_once", lambda *args, **kwargs: 0)

    with pytest.raises(OSError, match="zero bytes"):
        _close_noop(tmp_path, _ARCHIVE_SID, store, hebbian)

    assert {name: p.read_bytes() for name, p in paths.items()} == before


def test_a_crash_between_the_archive_and_the_removal_loses_nothing_and_archives_once(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C4: the buffer survives a failure at the removal, the retry completes, and the archive holds each row exactly once."""
    rows = _seed_closable(
        tmp_path, _ARCHIVE_SID, _archive_rows(_ARCHIVE_SID, 4, start=datetime(2026, 10, 1, 9, 0, tzinfo=UTC))
    )
    real_delete = pipeline.delete_session_buffer
    calls: list[int] = []

    def _delete_once_raising(persona_dir, session_id):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("killed at the removal")
        real_delete(persona_dir, session_id)

    monkeypatch.setattr(pipeline, "delete_session_buffer", _delete_once_raising)

    with pytest.raises(OSError, match="killed at the removal"):
        _close_noop(tmp_path, _ARCHIVE_SID, store, hebbian)
    assert read_archive(tmp_path, _ARCHIVE_SID) == rows
    assert _sidecars(tmp_path, _ARCHIVE_SID)["buffer"].exists()

    assert _close_noop(tmp_path, _ARCHIVE_SID, store, hebbian).reached_end
    assert not _sidecars(tmp_path, _ARCHIVE_SID)["buffer"].exists()
    assert read_archive(tmp_path, _ARCHIVE_SID) == rows  # once, not twice


def test_a_crash_between_the_archive_append_and_its_marker_loses_no_row(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C14: the window inside archive_once. Nothing is removed, and the retry leaves every row in the archive (a repeat is possible here, a loss is not)."""
    from brain.ingest import buffer as buffer_mod

    rows = _seed_closable(
        tmp_path, _ARCHIVE_SID, _archive_rows(_ARCHIVE_SID, 4, start=datetime(2026, 10, 1, 9, 0, tzinfo=UTC))
    )
    paths = _sidecars(tmp_path, _ARCHIVE_SID)
    before = {name: p.read_bytes() for name, p in paths.items()}
    real_marker = buffer_mod.write_archive_marker
    calls: list[int] = []

    def _marker_once_raising(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("killed before the marker")
        return real_marker(*args, **kwargs)

    monkeypatch.setattr(buffer_mod, "write_archive_marker", _marker_once_raising)

    with pytest.raises(OSError, match="killed before the marker"):
        _close_noop(tmp_path, _ARCHIVE_SID, store, hebbian)
    assert {name: p.read_bytes() for name, p in paths.items()} == before

    assert _close_noop(tmp_path, _ARCHIVE_SID, store, hebbian).reached_end
    assert not paths["buffer"].exists()
    archive = read_archive(tmp_path, _ARCHIVE_SID)
    assert all(archive.count(r) >= 1 for r in rows)


def test_a_writer_that_takes_no_append_lock_keeps_the_buffer_and_the_retry_archives_its_row(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C5: compaction's whole-file rewrite takes no append lock; a row it leaves in the buffer during the archive write is never removed unarchived."""
    from brain.ingest import buffer as buffer_mod

    rows = _seed_closable(
        tmp_path, _ARCHIVE_SID, _archive_rows(_ARCHIVE_SID, 4, start=datetime(2026, 10, 1, 9, 0, tzinfo=UTC))
    )
    added = {
        "session_id": _ARCHIVE_SID,
        "speaker": "user",
        "text": "a row the rewrite added",
        "ts": datetime(2026, 10, 1, 9, 30, tzinfo=UTC).isoformat(timespec="seconds"),
    }
    fired: list[int] = []

    def _archive_then_rewrite(persona_dir, session_id, records, **kwargs):
        written = buffer_mod.archive_once(persona_dir, session_id, records, **kwargs)
        if not fired:
            fired.append(1)
            rewrite_session_atomic(persona_dir, session_id, [*records, added])
        return written

    monkeypatch.setattr(pipeline, "archive_once", _archive_then_rewrite)
    paths = _sidecars(tmp_path, _ARCHIVE_SID)

    first = _close_noop(tmp_path, _ARCHIVE_SID, store, hebbian)

    assert first.reached_end is False
    assert read_session(tmp_path, _ARCHIVE_SID) == [*rows, added]
    assert paths["cursor"].exists() and paths["backoff"].exists()

    second = _close_noop(tmp_path, _ARCHIVE_SID, store, hebbian)

    assert second.reached_end
    assert not paths["buffer"].exists()
    archive = read_archive(tmp_path, _ARCHIVE_SID)
    assert added in archive and all(r in archive for r in rows)


def test_the_close_holds_the_append_lock_from_the_late_rows_check_to_the_removal(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C13: a lock-respecting appender is refused at the late-rows read, in the archive write and at the removal; an ingest_turn afterwards is not lost."""
    from brain.ingest import buffer as buffer_mod

    _seed_closable(tmp_path, _ARCHIVE_SID, _archive_rows(_ARCHIVE_SID, 4, start=datetime(2026, 10, 1, 9, 0, tzinfo=UTC)))
    buffer_path = _sidecars(tmp_path, _ARCHIVE_SID)["buffer"]
    probe_row = {"session_id": _ARCHIVE_SID, "speaker": "user", "text": "probe", "ts": "2026-10-01T10:00:00+00:00"}
    at_late_read: list[int | None] = []
    at_archive: list[int | None] = []
    at_delete: list[int | None] = []
    at_backoff: list[int | None] = []
    real_read_after = pipeline.read_session_after
    real_delete = pipeline.delete_session_buffer
    real_delete_backoff = pipeline.delete_backoff
    reads: list[int] = []

    def _read_after(persona_dir, session_id, after_ts):
        out = real_read_after(persona_dir, session_id, after_ts)
        reads.append(1)
        if len(reads) > 1:  # the first read opens the close, before the lock; every later one is the late-rows check
            at_late_read.append(append_jsonl(buffer_path, probe_row, lock="try"))
        return out

    def _archive(persona_dir, session_id, records, **kwargs):
        at_archive.append(append_jsonl(buffer_path, probe_row, lock="try"))
        return buffer_mod.archive_once(persona_dir, session_id, records, **kwargs)

    def _delete(persona_dir, session_id):
        at_delete.append(append_jsonl(buffer_path, probe_row, lock="try"))
        real_delete(persona_dir, session_id)

    def _delete_backoff(persona_dir, session_id):  # the last removal of the close
        at_backoff.append(append_jsonl(buffer_path, probe_row, lock="try"))
        real_delete_backoff(persona_dir, session_id)

    monkeypatch.setattr(pipeline, "read_session_after", _read_after)
    monkeypatch.setattr(pipeline, "archive_once", _archive)
    monkeypatch.setattr(pipeline, "delete_session_buffer", _delete)
    monkeypatch.setattr(pipeline, "delete_backoff", _delete_backoff)

    assert _close_noop(tmp_path, _ARCHIVE_SID, store, hebbian).reached_end

    assert (at_late_read, at_archive, at_delete, at_backoff) == ([None], [None], [None], [None])
    assert not buffer_path.exists()
    # positive control: the probe does write when nobody holds the lock, so a None above means "held", not "the probe cannot write"
    assert append_jsonl(buffer_path, probe_row, lock="try") is not None
    buffer_path.unlink()
    ingest_turn(tmp_path, {"session_id": _ARCHIVE_SID, "speaker": "user", "text": "after the close"})
    assert [r["text"] for r in read_session(tmp_path, _ARCHIVE_SID)] == ["after the close"]


def test_a_torn_line_is_dropped_with_the_file_and_every_parseable_row_is_archived(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    """Policy D5, as compaction and the rollover treat it: the buffer reader skips an unparseable line, so the close archives the parseable rows and the fragment goes with the file."""
    rows = _seed_closable(
        tmp_path, _ARCHIVE_SID, _archive_rows(_ARCHIVE_SID, 4, start=datetime(2026, 10, 1, 9, 0, tzinfo=UTC))
    )
    buffer_path = _sidecars(tmp_path, _ARCHIVE_SID)["buffer"]
    with buffer_path.open("ab") as fh:  # a crash mid-append leaves a torn tail (#434)
        fh.write(b'{"session_id": "sess_arch", "speaker": "user", "te')

    assert _close_noop(tmp_path, _ARCHIVE_SID, store, hebbian).reached_end

    assert read_archive(tmp_path, _ARCHIVE_SID) == rows
    assert not buffer_path.exists()


# ---------------------------------------------------------------------------
# The ghost-file sweeps delete an empty buffer only under its append lock
# ---------------------------------------------------------------------------

_GHOST_SID = "sess_ghost_race"
_GHOST_SWEEPS = [
    pytest.param(snapshot_stale_sessions, id="snapshot_stale_sessions"),
    pytest.param(close_stale_sessions, id="close_stale_sessions"),
]


def _ghost(persona: Path) -> Path:
    buf = persona / "active_conversations" / f"{_GHOST_SID}.jsonl"
    buf.parent.mkdir(parents=True, exist_ok=True)
    buf.touch()
    return buf


@pytest.mark.parametrize("sweep", _GHOST_SWEEPS)
def test_a_turn_landing_after_the_empty_read_is_not_deleted_with_the_ghost(
    sweep, tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sweep's first read finds no rows; a turn appended before the unlink must keep the buffer."""
    _ghost(tmp_path)
    real_read = pipeline.read_session
    reads: list[int] = []

    def _read_then_append(persona_dir, session_id):
        out = real_read(persona_dir, session_id)
        if session_id == _GHOST_SID and not reads:
            reads.append(1)
            ingest_turn(persona_dir, {"session_id": _GHOST_SID, "speaker": "user", "text": "a turn mid-sweep"})
        return out

    monkeypatch.setattr(pipeline, "read_session", _read_then_append)

    sweep(tmp_path, silence_minutes=5.0, store=store, hebbian=hebbian, provider=_CannedProvider([]))

    assert reads == [1]
    assert [r["text"] for r in read_session(tmp_path, _GHOST_SID)] == ["a turn mid-sweep"]


@pytest.mark.parametrize("sweep", _GHOST_SWEEPS)
def test_the_ghost_unlink_runs_under_the_append_lock(
    sweep, tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock-respecting appender is refused while the sweep unlinks the ghost; the ghost is still removed."""
    buf = _ghost(tmp_path)
    probe = {"session_id": _GHOST_SID, "speaker": "user", "text": "probe", "ts": "2026-10-01T10:00:00+00:00"}
    at_delete: list[int | None] = []
    real_delete = pipeline.delete_session_buffer

    def _delete(persona_dir, session_id):
        at_delete.append(append_jsonl(buf, probe, lock="try"))
        if at_delete[-1] is None:  # held: the unlink goes ahead on the still-empty file
            real_delete(persona_dir, session_id)

    monkeypatch.setattr(pipeline, "delete_session_buffer", _delete)

    sweep(tmp_path, silence_minutes=5.0, store=store, hebbian=hebbian, provider=_CannedProvider([]))

    assert at_delete == [None]
    assert not buf.exists()
    # positive control: with nobody holding the lock the probe writes, so None above means "held"
    buf.touch()
    assert append_jsonl(buf, probe, lock="try") is not None
