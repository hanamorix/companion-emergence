"""Review follow-ups on #470 (the other maintainer's review of the PR head 584ec8a1): each test names the defect it pins.

Written first and observed failing on 584ec8a1 (the classifier recursion test passes there and guards the fix: the moved scan
must keep catching RecursionError). Synthetic names only; every persona is a tmp dir.
"""

from __future__ import annotations

import pytest

from brain.engines.consolidation import _make_haiku_classifier
from brain.memory.store import Memory


class _Reply:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, *_a, **_k) -> str:
        return self.reply


def _cand() -> Memory:
    return Memory.create_new(content="a synthetic candidate", memory_type="dream", domain="us", emotions={})


# --- item 1: the classifier reads a verdict out of prose with braces ---------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        '{"verdict": "new"}\nNote: the candidate quotes {braces}.',
        'Verdict: {"verdict": "new", "names": []} -- I considered {x} first',
        'Here is {the} verdict: {"verdict": "new"}',
        '{"note": "context"} then the answer {"verdict": "new"}',
    ],
)
def test_classifier_reads_a_verdict_with_braces_in_the_surrounding_prose(reply):
    # Fails on 584ec8a1: the greedy `\{.*\}` spans the stray braces, json.loads fails, UnusableClassifierReply -> the
    # candidate is quarantined for good although the model gave a valid verdict.
    decision = _make_haiku_classifier(_Reply(reply))(_cand(), [])
    assert decision.verdict == "new"


def test_classifier_treats_a_reply_nested_past_the_recursion_limit_as_unusable():
    # The scan must catch RecursionError as `json.loads` callers do (#406's tuple): escaping the classifier it would read as
    # a FAILED CALL (ServiceCallFailed -> requeue and end the pass) on every tick, not one quarantine with evidence.
    from brain.engines.consolidation import UnusableClassifierReply

    reply = '{"verdict": "new", "x": ' + "[" * 200_000 + "]" * 200_000 + "}"
    with pytest.raises(UnusableClassifierReply):
        _make_haiku_classifier(_Reply(reply))(_cand(), [])


def test_find_json_object_block_prefers_the_given_key_and_catches_recursion():
    from brain.utils.llm_output import find_json_object_block

    assert find_json_object_block('{"a": 1} {"verdict": "new"}', prefer_key="verdict") == '{"verdict": "new"}'
    assert find_json_object_block('{"a": 1} {"verdict": "new"}') == '{"a": 1}'
    assert find_json_object_block("no braces at all") is None
    deep = '{"x": ' + "[" * 200_000 + "]" * 200_000 + "}"
    assert find_json_object_block(deep, prefer_key="verdict") == deep  # the balanced fallback; the caller's loads reports it


# --- item 1b: soul review's parse_decision always returns a Decision ----------------------------------------------------


def test_soul_parse_decision_returns_a_decision_for_a_reply_nested_past_the_recursion_limit():
    # On 584ec8a1 the #478 scan catches only ValueError, so RecursionError escapes `parse_decision` (documented as "Always
    # returns a Decision") and ends the whole soul-review pass at its unguarded call.
    from brain.soul.review import parse_decision

    raw = '{"decision": "accept", "x": ' + "[" * 200_000 + "]" * 200_000 + "}"
    decision = parse_decision(raw, "cand-1")
    assert decision.decision == "defer"
    assert decision.parse_error


# --- item 2: one extraction per session (#477): close and finalize -----------------------------------------------------


class _CountingProvider:
    def __init__(self) -> None:
        self.calls = 0

    def name(self):
        return "stub"

    def healthy(self):
        return True

    def generate(self, prompt=None, *, system=None, **kw):
        self.calls += 1
        return "[]"


@pytest.fixture
def world(tmp_path):
    import json

    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    persona = tmp_path / "persona"
    (persona / "active_conversations").mkdir(parents=True)
    (persona / "persona_config.json").write_text(json.dumps({"provider": "fake", "searcher": "fake"}), encoding="utf-8")
    store = MemoryStore(persona / "memories.db")
    hebbian = HebbianMatrix(persona / "hebbian.db")
    yield persona, store, hebbian
    store.close()
    hebbian.close()


def _seed(persona, sid):
    from datetime import UTC, datetime, timedelta

    from brain.ingest.buffer import ingest_turn

    old = datetime(2026, 1, 1, tzinfo=UTC)
    for i in range(4):
        ingest_turn(
            persona,
            {
                "session_id": sid,
                "speaker": "user" if i % 2 == 0 else "assistant",
                "text": f"turn {i}",
                "ts": (old + timedelta(minutes=i)).isoformat(),
            },
        )
    return persona / "active_conversations" / f"{sid}.jsonl"


def test_close_session_skips_while_another_extraction_holds_the_session(world):
    # Fails on 584ec8a1: close_session takes no extraction lock, so it extracts and DELETES the buffer while a snapshot pass
    # (the stale sweep, a rollover, /sessions/snapshot) is mid-extraction of the same turns.
    from brain.ingest.buffer import extract_lock_path
    from brain.ingest.pipeline import close_session
    from brain.utils.file_lock import file_lock

    persona, store, hebbian = world
    sid = "sess_close_busy"
    buf = _seed(persona, sid)
    provider = _CountingProvider()
    with file_lock(extract_lock_path(persona, sid), blocking=False) as held:
        assert held
        report = close_session(persona, sid, store=store, hebbian=hebbian, provider=provider)
    assert report.busy is True
    assert provider.calls == 0
    assert buf.exists()


def test_close_session_still_closes_when_nothing_holds_the_session(world):
    # PIN (passes on 584ec8a1): the lock is only taken, never left held, and a free session closes as before.
    from brain.ingest.pipeline import close_session

    persona, store, hebbian = world
    sid = "sess_close_free"
    buf = _seed(persona, sid)
    provider = _CountingProvider()
    report = close_session(persona, sid, store=store, hebbian=hebbian, provider=provider)
    assert report.busy is False
    assert provider.calls >= 1
    assert not buf.exists()


def test_finalize_does_not_poison_a_session_another_extraction_holds(world, caplog):
    # Fails on 584ec8a1: the snapshot reports busy, but finalize still reads the backoff sidecar and renames the buffer into
    # poison/ (and deletes its cursor) under the pass that holds the session.
    import logging
    from datetime import UTC, datetime, timedelta

    from brain.ingest.buffer import extract_lock_path, read_backoff, write_backoff
    from brain.ingest.pipeline import finalize_stale_sessions
    from brain.utils.file_lock import file_lock

    persona, store, hebbian = world
    sid = "sess_finalize_busy"
    buf = _seed(persona, sid)
    write_backoff(persona, sid, failures=3, first_failure_at=(datetime.now(UTC) - timedelta(hours=1)).isoformat())
    with caplog.at_level(logging.INFO), file_lock(extract_lock_path(persona, sid), blocking=False) as held:
        assert held
        reports = finalize_stale_sessions(persona, store=store, hebbian=hebbian, provider=_CountingProvider())
    assert [r.busy for r in reports] == [True]
    assert buf.exists()
    assert not (persona / "active_conversations" / "poison" / f"{sid}.jsonl").exists()
    assert "conversation_finalize_deadletter_busy" in caplog.text
    assert read_backoff(persona, sid) is not None  # left for the next tick: the failure record is kept too


def test_finalize_still_poisons_a_session_nothing_holds(world):
    # PIN (passes on 584ec8a1): with the lock free the dead-letter happens. #523: only the failing chunk
    # (here the whole short span) is set aside in poison/<sid>.span.jsonl; the buffer stays.
    from datetime import UTC, datetime

    from brain.ingest.buffer import write_backoff
    from brain.ingest.pipeline import finalize_stale_sessions

    persona, store, hebbian = world
    sid = "sess_finalize_free"
    buf = _seed(persona, sid)
    write_backoff(persona, sid, failures=3, first_failure_at=datetime.now(UTC).isoformat())  # inside the backoff window: the snapshot is skipped
    finalize_stale_sessions(persona, store=store, hebbian=hebbian, provider=_CountingProvider())
    assert buf.exists()
    assert (persona / "active_conversations" / "poison" / f"{sid}.span.jsonl").exists()


# --- item 3: the environment requeue keeps the ORIGINAL entry -----------------------------------------------------------


def test_environment_requeue_keeps_the_original_entry_not_the_mutated_candidate(tmp_path, monkeypatch):
    # Fails on 584ec8a1: `_dispatch` sets `correction_of` on the in-memory candidate before `store.create`; when the create
    # hits a locked database the gate requeues that MUTATED Memory.
    import sqlite3

    import brain.engines.consolidation as cons
    from brain.engines.consolidation import Decision, run_consolidation
    from brain.health.json_io import read_jsonl
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.pending import PendingQueue
    from brain.memory.store import MemoryStore

    store = MemoryStore(tmp_path / "memories.db")
    heb = HebbianMatrix(tmp_path / "hebbian.db")
    try:
        target = Memory.create_new(content="an existing memory", memory_type="dream", domain="us", emotions={})
        store.create(target)
        queue = PendingQueue(tmp_path)
        queue.enqueue(_cand(), source="dream")
        original = read_jsonl(queue.path)[0]
        real_create = store.create

        def locked_create(mem):
            if mem.id == original["id"]:
                raise sqlite3.OperationalError("database is locked")
            return real_create(mem)

        monkeypatch.setattr(store, "create", locked_create)
        monkeypatch.setattr(cons, "_embed_candidate", lambda _c: None)
        result = run_consolidation(
            store,
            persona_dir=tmp_path,
            hebbian=heb,
            classifier=lambda _c, _ctx: Decision("correction", target_id=target.id),
        )
        assert result.failed == 1
        requeued = [r for r in read_jsonl(queue.path) if r.get("id") == original["id"]]
        assert len(requeued) == 1
        assert "correction_of" not in (requeued[0].get("metadata") or {})
    finally:
        store.close()
        heb.close()


# --- item 8: rollover's dead `errors == 0` clause -----------------------------------------------------------------------


def test_a_report_that_did_not_complete_with_no_errors_is_always_busy_or_deferred():
    # PIN (passes before and after; the clause removed from rollover.py changes no behaviour): once `completed` is False,
    # `errors == 0` can only hold when busy or deferred is already set, and rollover tests those first.
    import itertools

    from brain.ingest.types import IngestReport

    # #523: a pass can now also stop before the end of its span with no error (a chunk limit); `reached_end` says so, and the
    # rollover's extraction is never bounded, so for it this case cannot arise.
    for busy, deferred, backoff_skipped, errors, reached_end in itertools.product(
        [False, True], [False, True], [False, True], [0, 1], [False, True]
    ):
        report = IngestReport(
            session_id="s", busy=busy, deferred=deferred, backoff_skipped=backoff_skipped, errors=errors, reached_end=reached_end
        )
        if not report.completed and report.errors == 0:
            assert report.busy or report.deferred or not report.reached_end


# --- finalize's dead-letter move when the buffer is already gone (a rollover reaped it) --------------------------------


def _at_threshold(persona, sid):
    from datetime import UTC, datetime

    from brain.ingest.buffer import write_backoff

    write_backoff(persona, sid, failures=3, first_failure_at=datetime.now(UTC).isoformat())  # inside the window: no new attempt


def test_finalize_skips_the_dead_letter_move_when_the_buffer_is_already_gone(world, monkeypatch, caplog):
    # Fails on the head before this change: the backoff is read before the lock is taken; a successful rollover that reaped the
    # buffer in between leaves `src.rename` raising FileNotFoundError, which escapes the per-session handling and aborts the
    # tick's sweep (the next session is never finalized).
    import logging

    import brain.ingest.pipeline as pipeline
    from brain.ingest.buffer import read_backoff
    from brain.ingest.types import IngestReport

    persona, store, hebbian = world
    gone = _seed(persona, "sess_a_gone")
    free = _seed(persona, "sess_b_free")
    _at_threshold(persona, "sess_a_gone")
    _at_threshold(persona, "sess_b_free")
    real = pipeline.extract_session_snapshot

    def snapshot(persona_dir, sid, **kw):
        if sid == "sess_a_gone":  # the rollover finished and reaped this buffer after finalize listed it
            gone.unlink()
            return IngestReport(session_id=sid)
        return real(persona_dir, sid, **kw)

    monkeypatch.setattr(pipeline, "extract_session_snapshot", snapshot)
    with caplog.at_level(logging.INFO):
        reports = pipeline.finalize_stale_sessions(persona, store=store, hebbian=hebbian, provider=_CountingProvider())

    assert sorted(r.session_id for r in reports) == ["sess_a_gone", "sess_b_free"]  # the sweep went on
    poison = persona / "active_conversations" / "poison"
    assert not (poison / "sess_a_gone.span.jsonl").exists()  # nothing to poison
    assert read_backoff(persona, "sess_a_gone") is None  # the stale failure record is cleared
    assert "conversation_finalize_deadletter_gone" in caplog.text
    # The other session's failing chunk was still set aside (#523: only the chunk; its buffer stays).
    assert (poison / "sess_b_free.span.jsonl").exists() and free.exists()


def test_finalize_survives_a_fault_while_setting_a_chunk_aside(world, monkeypatch):
    # #523: finalize no longer renames the buffer; it sets the failing chunk aside through
    # poison_unextracted_span. A fault there for one session (here a buffer that vanished mid-way) must not escape and abort the
    # sweep for the others.
    import brain.ingest.pipeline as pipeline
    from brain.ingest.buffer import read_backoff

    persona, store, hebbian = world
    _seed(persona, "sess_a_race")
    _seed(persona, "sess_b_free")
    _at_threshold(persona, "sess_a_race")
    _at_threshold(persona, "sess_b_free")
    real = pipeline.poison_unextracted_span

    def poison(persona_dir, sid):
        if sid == "sess_a_race":
            raise FileNotFoundError(2, "No such file or directory", sid)
        return real(persona_dir, sid)

    monkeypatch.setattr(pipeline, "poison_unextracted_span", poison)
    reports = pipeline.finalize_stale_sessions(persona, store=store, hebbian=hebbian, provider=_CountingProvider())

    assert sorted(r.session_id for r in reports) == ["sess_a_race", "sess_b_free"]
    assert (persona / "active_conversations" / "poison" / "sess_b_free.span.jsonl").exists()
    assert read_backoff(persona, "sess_b_free") is None  # its chunk was set aside: the failure record is cleared
    # The faulted session keeps its failure record, so the next tick tries to set its chunk aside again.
    kept = read_backoff(persona, "sess_a_race")
    assert kept is not None and kept["failures"] >= 3


# --- round-4 S2: find_json_object_block is linear on a reply of many unclosed braces -------------------------------------


def _reference_balanced_block(text: str, start: int) -> str | None:
    """The #478 scan exactly as it was before it was made linear (kept here as the oracle)."""
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _reference_find_json_object_block(text: str, prefer_key: str | None = None) -> str | None:
    """`find_json_object_block` as it was before (quadratic on many unclosed braces): the oracle for "no result changes"."""
    import json
    import re

    object_start = re.compile(r'\{\s*"')
    decoder = json.JSONDecoder()
    first_object = None
    first_balanced = None
    inside_failed_until = 0
    start = text.find("{")
    while start >= 0:
        if start < inside_failed_until:
            start = text.find("{", start + 1)
            continue
        try:
            obj, end = decoder.raw_decode(text, start)
        except (ValueError, RecursionError):
            obj = None
            balanced = _reference_balanced_block(text, start)
            if balanced is not None:
                inside_failed_until = start + len(balanced)
            elif object_start.match(text, start):
                inside_failed_until = len(text)
        else:
            balanced = None
            if isinstance(obj, dict):
                block = text[start:end]
                if prefer_key is not None and prefer_key in obj:
                    return block
                if first_object is None:
                    first_object = block
        if first_balanced is None:
            first_balanced = balanced if balanced is not None else _reference_balanced_block(text, start)
        start = text.find("{", start + 1)
    return first_object if first_object is not None else first_balanced


def test_find_json_object_block_gives_the_same_result_as_before_on_every_short_reply():
    # PIN (passes before and after): every string up to 6 characters over the characters that matter to the scan (braces,
    # quotes, a backslash, a letter, a colon), for no preferred key and for one that can match.
    import itertools

    from brain.utils.llm_output import find_json_object_block

    for length in range(1, 7):
        for chars in itertools.product('{}"\\a:', repeat=length):
            text = "".join(chars)
            for key in (None, "a"):
                assert find_json_object_block(text, key) == _reference_find_json_object_block(text, key), (text, key)


def test_find_json_object_block_gives_the_same_result_as_before_on_longer_random_replies():
    import random

    from brain.utils.llm_output import find_json_object_block

    rng = random.Random(478)
    pieces = ["{", "}", '"', "\\", "\\\\", '\\"', ":", ",", " ", "a", "1", '"a"', '"decision"', '{"a": 1}', '{"decision": "x"}',
              "prose ", "[", "]", "{{", "}}", '{"', '"}', "\n", "\t", "\r", "{}", "{ }"]
    for _ in range(4000):
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 30)))
        for key in (None, "a", "decision"):
            assert find_json_object_block(text, key) == _reference_find_json_object_block(text, key), (text, key)


@pytest.mark.parametrize("unit", ["{", "{ ", 'x {"', "{[", '{"a": {', 'say { "'])
def test_find_json_object_block_is_linear_on_a_reply_of_many_unclosed_braces(unit):
    # S2 of the round-4 review: each `{` that failed to decode re-scanned to the end of the text, so 8000 braces took 5 s (the
    # classifier is exposed to a degenerate repetition reply). A generous ceiling, hundreds of times what a linear pass needs
    # and far under what the quadratic one takes at this size.
    import time

    from brain.utils.llm_output import find_json_object_block

    text = unit * 10_000
    started = time.perf_counter()
    find_json_object_block(text, prefer_key="verdict")
    elapsed = time.perf_counter() - started

    assert elapsed < 1.5, f"{elapsed:.2f}s for {len(text)} chars of {unit!r}"


def test_find_json_object_block_still_finds_the_answer_after_many_unclosed_braces():
    from brain.utils.llm_output import find_json_object_block

    text = "{ " * 5_000 + 'then the answer {"verdict": "new"}'
    assert find_json_object_block(text, prefer_key="verdict") == '{"verdict": "new"}'
