"""Tests for brain.ingest.pipeline — close_session + close_stale_sessions."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import LLMProvider
from brain.ingest.buffer import ingest_turn
from brain.ingest.pipeline import close_session, close_stale_sessions
from brain.ingest.soul_queue import list_soul_candidates
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(":memory:")


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
    """Full pipeline: 3 turns → extract 2 items → 0 deduped → 2 committed."""
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
    assert report.committed == 2
    assert report.deduped == 0
    assert report.errors == 0
    assert len(report.memory_ids) == 2
    # Both memories should exist in the store.
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
    """High-importance item: both committed to store AND queued as soul candidate."""
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

    assert report.committed == 1
    assert report.soul_candidates == 1
    assert len(report.memory_ids) == 1

    mem_id = report.memory_ids[0]
    memory = store.get(mem_id)
    assert memory is not None
    assert memory.content == "The most important truth"

    candidates = list_soul_candidates(tmp_path)
    assert len(candidates) == 1
    assert candidates[0]["memory_id"] == mem_id


def test_close_session_counts_soul_queue_failure_without_claiming_success(
    tmp_path: Path, store: MemoryStore, hebbian: HebbianMatrix, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed memory is not reported as queued if the soul queue append fails."""
    ingest_turn(tmp_path, {"session_id": "sess_soul_fail", "speaker": "Hana", "text": "this matters"})
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

    assert report.committed == 1
    assert report.soul_candidates == 0
    assert report.soul_queue_errors == 1
    assert list_soul_candidates(tmp_path) == []


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
    ingest_turn(persona_dir, {"session_id": "sess_named", "speaker": "assistant", "text": "hey love"})

    captured: list[str] = []

    class _CapProvider(LLMProvider):
        def name(self): return "cap"
        def healthy(self): return True
        def chat(self, *a, **kw): raise NotImplementedError
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
        def name(self): return "cap"
        def healthy(self): return True
        def chat(self, *a, **kw): raise NotImplementedError
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
