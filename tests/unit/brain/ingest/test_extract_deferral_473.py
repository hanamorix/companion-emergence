"""#473 (ingest caller): a deferred model call is "not now", not a failed extraction.

`extract_items_with_status` caught every exception from `provider.generate` as one failed attempt, so a `ThrottleDeferred` (an expired
login, a bridge that is stopping) was retried straight away and then reported as `failed=True`. The pipeline recorded that in the
session's backoff sidecar, and after three recorded failures `finalize_stale_sessions` moved the buffer to `poison/` and deleted its
cursor: a conversation that was only waiting for the login to come back was quarantined without being extracted. A deferral now
returns a `deferred` outcome after ONE call; the pipeline leaves the cursor, the buffer and the backoff sidecar untouched.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.cli_throttle import ThrottleDeferred
from brain.bridge.provider import LLMProvider
from brain.bridge.provider_auth import ProviderAuthDeferred
from brain.ingest.buffer import ingest_turn, read_backoff, read_cursor
from brain.ingest.extract import extract_items_with_status
from brain.ingest.pipeline import (
    close_session,
    extract_session_snapshot,
    finalize_stale_sessions,
)
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


class _Deferring(LLMProvider):
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    def generate(self, prompt, *, system=None):
        self.calls += 1
        raise self.exc

    def name(self):
        return "deferring"

    def chat(self, messages: list[ChatMessage], *, tools=None, options=None) -> ChatResponse:
        self.calls += 1
        raise self.exc


class _Broken(_Deferring):
    pass


@pytest.fixture
def parts(tmp_path):
    return tmp_path, MemoryStore(tmp_path / "memories.db"), HebbianMatrix(":memory:")


@pytest.mark.parametrize("exc", [ThrottleDeferred("not now"), ProviderAuthDeferred("login expired")])
def test_a_deferred_call_is_one_call_and_a_deferred_outcome(exc):
    provider = _Deferring(exc)
    outcome = extract_items_with_status("Hana: hello there", provider=provider, max_retries=3)
    assert provider.calls == 1  # no retry burn
    assert outcome.deferred is True
    assert outcome.failed is False
    assert outcome.items == []


def test_an_ordinary_provider_failure_is_still_a_failed_extraction():
    provider = _Broken(RuntimeError("boom"))
    outcome = extract_items_with_status("Hana: hello there", provider=provider, max_retries=1)
    assert provider.calls == 2
    assert outcome.failed is True and outcome.deferred is False


def test_a_snapshot_deferred_leaves_cursor_backoff_and_buffer_alone(parts, caplog):
    persona, store, hebbian = parts
    ingest_turn(persona, {"session_id": "s1", "speaker": "Hana", "text": "hello", "ts": "2026-06-15T10:00:00+00:00"})
    provider = _Deferring(ThrottleDeferred("not now"))
    for _ in range(5):
        report = extract_session_snapshot(persona, "s1", store=store, hebbian=hebbian, provider=provider)
        assert report.errors == 0 and report.commit_failures == 0
        assert report.deferred is True
    assert read_backoff(persona, "s1") is None
    assert read_cursor(persona, "s1") is None
    assert (persona / "active_conversations" / "s1.jsonl").exists()


def test_close_session_deferred_retains_the_buffer_without_an_error(parts):
    persona, store, hebbian = parts
    ingest_turn(persona, {"session_id": "s1", "speaker": "Hana", "text": "hello", "ts": "2026-06-15T10:00:00+00:00"})
    report = close_session(persona, "s1", store=store, hebbian=hebbian, provider=_Deferring(ThrottleDeferred("x")))
    assert report.errors == 0 and report.deferred is True
    assert (persona / "active_conversations" / "s1.jsonl").exists()
    assert read_backoff(persona, "s1") is None


def test_a_session_waiting_for_the_login_is_never_moved_to_poison(parts):
    persona, store, hebbian = parts
    old = (datetime.now(UTC) - timedelta(days=3)).isoformat()
    ingest_turn(persona, {"session_id": "s1", "speaker": "Hana", "text": "hello", "ts": old})
    provider = _Deferring(ProviderAuthDeferred("login expired"))
    for _ in range(6):  # the poison threshold is 3 recorded failures
        finalize_stale_sessions(persona, store=store, hebbian=hebbian, provider=provider)
    assert (persona / "active_conversations" / "s1.jsonl").exists()
    assert not (persona / "active_conversations" / "poison").exists()
    assert read_backoff(persona, "s1") is None


def test_a_real_failure_still_reaches_the_poison_queue(parts):
    persona, store, hebbian = parts
    old = (datetime.now(UTC) - timedelta(days=3)).isoformat()
    ingest_turn(persona, {"session_id": "s1", "speaker": "Hana", "text": "hello", "ts": old})
    provider = _Broken(RuntimeError("boom"))
    for _ in range(6):
        finalize_stale_sessions(persona, store=store, hebbian=hebbian, provider=provider)
    # #523: finalize sets aside the failing chunk (here the whole one-row span) in poison/<sid>.span.jsonl; the buffer stays.
    assert (persona / "active_conversations" / "poison" / "s1.span.jsonl").exists()
