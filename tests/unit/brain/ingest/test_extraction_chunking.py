"""#523: chunked memory extraction. The whole since-cursor span reaches the extractor, oldest first, in chunks no larger
than ``ingest.extraction_chunk_chars``, and the cursor advances per committed chunk.

Synthetic persona only (persona Kestrel, user Bob, made-up words). Criteria: ``1.5-criteria.md`` of the #523 run (C1-C12).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import pytest

from brain.ingest import ingest_utils
from brain.ingest.extract import format_transcript, format_turn_line
from brain.ingest.ingest_utils import TurnChunk, chunk_turns, extraction_chunk_chars

_T0 = datetime(2026, 10, 1, 20, 0, 0, tzinfo=UTC)


def _ts(seconds: float) -> str:
    return (_T0 + timedelta(seconds=seconds)).isoformat(timespec="seconds")


def _row(speaker: str, text: str, seconds: float | None) -> dict:
    row = {"session_id": "sess_kestrel", "speaker": speaker, "text": text}
    if seconds is not None:
        row["ts"] = _ts(seconds)
    return row


# ---------------------------------------------------------------------------
# Oracles (each is shown to fail on a named broken chunker variant below)
# ---------------------------------------------------------------------------


def _boundary_ts_violations(chunks: list[TurnChunk]) -> list[int]:
    """C3(a): indexes k where a row of chunk k shares (or exceeds) a ts of a row in a later chunk."""
    bad = []
    for k in range(len(chunks) - 1):
        before = [ingest_utils.parse_turn_ts(t.get("ts")) for c in chunks[: k + 1] for t in c.turns]
        after = [ingest_utils.parse_turn_ts(t.get("ts")) for c in chunks[k + 1 :] for t in c.turns]
        before = [b for b in before if b is not None]
        after = [a for a in after if a is not None]
        if before and after and max(before) >= min(after):
            bad.append(k)
    return bad


def _reply_openers(chunks: list[TurnChunk]) -> list[int]:
    """C3(b): indexes of chunks after the first that open with an assistant or image row."""
    return [k for k, c in enumerate(chunks) if k > 0 and c.turns[0].get("speaker") in ("assistant", "image")]


def _chunk_text_problems(chunks: list[TurnChunk], max_chars: int) -> list[str]:
    """C2 at the chunker: each chunk's text is exactly its whole rows formatted, within the limit unless oversize."""
    problems = []
    for k, c in enumerate(chunks):
        expected = "\n".join(format_turn_line(t, "Bob", "Kestrel") for t in c.turns)
        if c.text != expected or c.text != format_transcript(list(c.turns), user_name="Bob", assistant_name="Kestrel"):
            problems.append(f"chunk {k}: text is not its whole rows")
        if len(c.text) > max_chars and not c.oversize:
            problems.append(f"chunk {k}: {len(c.text)} > {max_chars} and not oversize")
    return problems


def _same_ts_pairs_history() -> list[dict]:
    """Exchanges where the user row and its reply share one second, and a group of four rows in one second."""
    rows = []
    t = 0
    for i in range(6):
        rows.append(_row("user", f"zorbel question {i} " + "x" * 40, t))
        rows.append(_row("assistant", f"quindle answer {i} " + "y" * 40, t))
        t += 30
    # four rows in one second: no cut may fall inside it even before its second user row
    rows += [
        _row("user", "plimsy one " + "a" * 40, t),
        _row("assistant", "plimsy two " + "b" * 40, t),
        _row("user", "plimsy three " + "c" * 40, t),
        _row("assistant", "plimsy four " + "d" * 40, t),
    ]
    return rows


def _straddling_pairs_history() -> list[dict]:
    """Exchanges whose reply is stamped a second (or minutes) after its user row."""
    rows = []
    t = 0
    for i in range(8):
        rows.append(_row("user", f"vantric ask {i} " + "u" * 50, t))
        rows.append(_row("assistant", f"vantric reply {i} " + "r" * 50, t + 1 + (i % 3) * 60))
        t += 300
    return rows


def _any_boundary(speaker, ts, prefix_max, suffix_min):  # noqa: ANN001 - variant of ingest_utils._cut_allowed
    return True


def _ts_rule_only(speaker, ts, prefix_max, suffix_min):  # noqa: ANN001 - variant: no exchange rule
    return ts is not None and prefix_max is not None and suffix_min is not None and prefix_max < suffix_min


# ---------------------------------------------------------------------------
# The chunker (C2, C3 a/b at the chunker; the pipeline halves are below)
# ---------------------------------------------------------------------------


def test_chunk_text_is_whole_rows_within_the_limit() -> None:
    rows = _straddling_pairs_history() + _same_ts_pairs_history()
    for max_chars in (150, 300, 700, 5000):
        chunks = chunk_turns(rows, max_chars=max_chars, user_name="Bob", assistant_name="Kestrel")
        assert [t for c in chunks for t in c.turns] == rows  # every row once, in order
        assert _chunk_text_problems(chunks, max_chars) == []


def test_same_ts_rows_are_never_split() -> None:
    chunks = chunk_turns(_same_ts_pairs_history(), max_chars=120, user_name="Bob", assistant_name="Kestrel")
    assert len(chunks) > 3
    assert _boundary_ts_violations(chunks) == []
    # the four rows of one second stay in one chunk even though the group holds a second user row
    group = [c for c in chunks if any("plimsy" in t["text"] for t in c.turns)]
    assert len(group) == 1 and len(group[0].turns) == 4 and group[0].oversize


def test_no_chunk_opens_with_a_reply_even_when_the_pair_straddles_a_second() -> None:
    chunks = chunk_turns(_straddling_pairs_history(), max_chars=80, user_name="Bob", assistant_name="Kestrel")
    assert len(chunks) == 8  # each exchange is its own (oversize) chunk: a user row is never cut from its reply
    assert _reply_openers(chunks) == []
    assert all(len(c.turns) == 2 for c in chunks)


@pytest.mark.parametrize(
    ("variant", "history", "oracle"),
    [
        (_any_boundary, _same_ts_pairs_history, _boundary_ts_violations),
        (_any_boundary, _straddling_pairs_history, _reply_openers),
        (_ts_rule_only, _straddling_pairs_history, _reply_openers),
    ],
    ids=["any-boundary-splits-same-ts", "any-boundary-opens-with-reply", "no-exchange-rule-opens-with-reply"],
)
def test_oracles_catch_broken_chunker_variants(monkeypatch, variant, history, oracle) -> None:
    """Fail-first kept executable (C3): the oracles above fail on a chunker that cuts at any row boundary, and on one
    without the exchange rule."""
    monkeypatch.setattr(ingest_utils, "_cut_allowed", variant)
    chunks = chunk_turns(history(), max_chars=80, user_name="Bob", assistant_name="Kestrel")
    assert oracle(chunks) != []


def test_cursor_ts_is_the_newest_row_up_to_the_chunk_end() -> None:
    rows = _straddling_pairs_history()
    chunks = chunk_turns(rows, max_chars=300, user_name="Bob", assistant_name="Kestrel")
    seen: list[dict] = []
    for c in chunks:
        seen += list(c.turns)
        newest = max(seen, key=lambda t: ingest_utils.parse_turn_ts(t["ts"]))
        assert c.cursor_ts == newest["ts"]


def test_a_unit_larger_than_the_limit_is_sent_whole() -> None:
    rows = [_row("user", "short " * 3, 0), _row("assistant", "grumvex " * 400, 1), _row("user", "after", 60)]
    chunks = chunk_turns(rows, max_chars=500, user_name="Bob", assistant_name="Kestrel")
    assert [len(c.turns) for c in chunks] == [2, 1]
    assert chunks[0].oversize and len(chunks[0].text) > 500
    assert "grumvex " * 400 in chunks[0].text  # whole, never cut mid-row
    assert not chunks[1].oversize


def test_rows_without_a_ts_join_the_row_before_and_are_logged(caplog) -> None:
    rows = [_row("user", "a" * 60, 0), _row("assistant", "b" * 60, 1), _row("user", "c" * 60, None), _row("user", "d" * 60, 90)]
    with caplog.at_level(logging.WARNING, logger="brain.ingest.ingest_utils"):
        chunks = chunk_turns(rows, max_chars=70, user_name="Bob", assistant_name="Kestrel")
    assert [[t["text"][0] for t in c.turns] for c in chunks] == [["a", "b", "c"], ["d"]]
    assert any("conversation_extract_ts_missing" in r.message for r in caplog.records)


def test_summary_rows_are_dropped() -> None:
    rows = [_row("summary", "faded recap", 999), _row("user", "hello", 0), _row("assistant", "hi", 1)]
    chunks = chunk_turns(rows, max_chars=1000, user_name="Bob", assistant_name="Kestrel")
    assert len(chunks) == 1 and [t["speaker"] for t in chunks[0].turns] == ["user", "assistant"]
    assert chunks[0].cursor_ts == _ts(1)


# ---------------------------------------------------------------------------
# C12: the named limit, validated
# ---------------------------------------------------------------------------


@pytest.fixture()
def tunables_home(tmp_path, monkeypatch):
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path))
    import brain.tunables as t
    from brain.utils import log_throttle

    t._reset_for_tests()
    log_throttle._last.clear()
    yield tmp_path
    t._reset_for_tests()
    log_throttle._last.clear()


def _write_tunables(home, overrides) -> None:
    (home / "tunables.json").write_text(json.dumps({"defaults": {}, "overrides": overrides}), encoding="utf-8")


def test_chunk_chars_default_is_registered_with_its_value(tunables_home) -> None:
    import brain.tunables as t

    assert extraction_chunk_chars() == 24_000
    assert t._registry["ingest.extraction_chunk_chars"] == 24_000


@pytest.mark.parametrize("bad", [0, -1, "x", True, False])
def test_a_bad_override_is_ignored_with_one_warning(tunables_home, caplog, bad) -> None:
    _write_tunables(tunables_home, {"ingest.extraction_chunk_chars": bad})
    with caplog.at_level(logging.WARNING):
        assert extraction_chunk_chars() == 24_000
        assert extraction_chunk_chars() == 24_000
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "ingest.extraction_chunk_chars" in r.getMessage()]
    assert len(warnings) == 1, [r.getMessage() for r in warnings]


def test_a_positive_override_changes_the_chunk_size(tunables_home) -> None:
    _write_tunables(tunables_home, {"ingest.extraction_chunk_chars": 120})
    assert extraction_chunk_chars() == 120


# ---------------------------------------------------------------------------
# C5: `completed` means the cursor reached the end of the span
# ---------------------------------------------------------------------------


def test_a_bare_report_is_not_completed() -> None:
    """A report from any path that did not say it reached the end is not completed (fail-safe toward keeping the buffer)."""
    from brain.ingest.types import IngestReport

    assert IngestReport(session_id="s").completed is False
    assert IngestReport(session_id="s", chunks=1).completed is False  # stopped after chunk 1 with no error
    assert IngestReport(session_id="s", reached_end=True).completed is True
    assert IngestReport(session_id="s", reached_end=True, busy=True).completed is False
    assert IngestReport(session_id="s", reached_end=True, deferred=True).completed is False
    assert IngestReport(session_id="s", reached_end=True, errors=1).completed is False


# ---------------------------------------------------------------------------
# The pipeline (C1-C11): a real persona dir, real buffer/cursor/pending queue, a recording stub provider
# ---------------------------------------------------------------------------

from brain.bridge.chat import ChatMessage, ChatResponse  # noqa: E402
from brain.bridge.cli_throttle import ThrottleDeferred  # noqa: E402
from brain.bridge.provider import LLMProvider  # noqa: E402
from brain.ingest import pipeline  # noqa: E402
from brain.ingest.buffer import (  # noqa: E402
    ingest_turn,
    list_active_sessions,
    read_archive,
    read_backoff,
    read_cursor,
    read_session,
    write_backoff,
    write_cursor,
)
from brain.memory.hebbian import HebbianMatrix  # noqa: E402
from brain.memory.store import MemoryStore  # noqa: E402
from brain.persona_config import PersonaConfig  # noqa: E402

_CHUNK = 600  # the chunk limit the pipeline tests run with (patched; the tunable path is the canary's job)


class _Stub(LLMProvider):
    """Records every prompt; ``plan(n, prompt)`` may return "defer", "fail", "kbint", or a callable run before answering.

    Answers one distinct fact per call (``items=True``) or an empty array. ``cursor_at_call`` records the session cursor
    each call saw (C4: the cursor after chunk k is read at call k+1)."""

    def __init__(self, persona=None, sid=None, plan=None, items=False, importance=5):
        self.prompts: list[str] = []
        self.cursor_at_call: list[str | None] = []
        self._persona, self._sid, self._plan, self._items, self._importance = persona, sid, plan, items, importance

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append(prompt)
        n = len(self.prompts)
        if self._persona is not None and self._sid is not None:
            self.cursor_at_call.append(read_cursor(self._persona, self._sid))
        action = self._plan(n, prompt) if self._plan else None
        if callable(action):
            action()
        elif action == "defer":
            raise ThrottleDeferred("not now (stub)")
        elif action == "fail":
            raise RuntimeError("stub extraction failure")
        elif action == "kbint":
            raise KeyboardInterrupt
        if not self._items:
            return "[]"
        return json.dumps([{"text": f"fact number {n} about blorvath {n * 7919}", "label": "fact", "importance": self._importance}])

    def name(self) -> str:
        return "stub-recording"

    def chat(self, messages: list[ChatMessage], *, tools=None, options=None) -> ChatResponse:
        return ChatResponse(content="[]", tool_calls=())


class _SummaryStub:
    """Compaction provider stub: any summary text."""

    def generate(self, prompt=None, *, system=None, **kw) -> str:
        return "Bob and Kestrel talked about many made-up things."


@pytest.fixture
def world(tmp_path, monkeypatch):
    persona = tmp_path / "Kestrel"
    persona.mkdir()
    PersonaConfig(user_name="Bob").save(persona / "persona_config.json")
    store = MemoryStore(persona / "memories.db")
    hebbian = HebbianMatrix(persona / "hebbian.db")
    monkeypatch.setattr(pipeline, "extraction_chunk_chars", lambda: _CHUNK, raising=False)
    transcripts: list[str] = []
    real_extract = pipeline.extract_items_with_status

    def _spy(transcript, **kw):
        transcripts.append(transcript)
        return real_extract(transcript, **kw)

    monkeypatch.setattr(pipeline, "extract_items_with_status", _spy)
    yield persona, store, hebbian, transcripts
    store.close()
    hebbian.close()


def _word(sid: str, i: int, who: str) -> str:
    return f"w{sid[-3:]}{who}{i:03d}q"


def _seed(persona, sid: str, exchanges: int, *, start: datetime | None = None, gap_s: int = 120, reply_lag_s: int = 1) -> list[dict]:
    """Seed ``exchanges`` user/assistant pairs, each row ~110 chars with a unique word; replies stamped after their user row."""
    start = start or (datetime.now(UTC) - timedelta(hours=2))
    for i in range(exchanges):
        t = start + timedelta(seconds=i * gap_s)
        ingest_turn(persona, {"session_id": sid, "speaker": "user", "text": f"{_word(sid, i, 'u')} " + "m" * 90, "ts": t.isoformat(timespec="seconds")})
        ingest_turn(
            persona,
            {"session_id": sid, "speaker": "assistant", "text": f"{_word(sid, i, 'a')} " + "n" * 90,
             "ts": (t + timedelta(seconds=reply_lag_s)).isoformat(timespec="seconds")},
        )
    return read_session(persona, sid)


def _expected_chunks(rows: list[dict]) -> list[TurnChunk]:
    return chunk_turns(rows, max_chars=_CHUNK, user_name="Bob", assistant_name="Kestrel")


def _line(row: dict) -> str:
    return format_turn_line(row, "Bob", "Kestrel")


def _snapshot(world_, sid, provider, **kw):
    persona, store, hebbian, _ = world_
    return pipeline.extract_session_snapshot(persona, sid, store=store, hebbian=hebbian, provider=provider, **kw)


def _close(world_, sid, provider):
    persona, store, hebbian, _ = world_
    return pipeline.close_session(persona, sid, store=store, hebbian=hebbian, provider=provider)


def _long_span(persona, sid: str) -> list[dict]:
    """A lull-free span of more than 3x the DEFAULT limit (24,000 chars), so the pre-change tail cut (fixed at that size)
    fires on it too; the pipeline runs with the default limit here."""
    rows = _seed(persona, sid, 340, gap_s=30)
    assert len("\n".join(_line(r) for r in rows)) >= 3 * 24_000
    return rows


def test_c1_the_head_reaches_the_extractor(world, monkeypatch) -> None:
    persona, _, _, _ = world
    monkeypatch.setattr(pipeline, "extraction_chunk_chars", lambda: 24_000, raising=False)
    rows = _long_span(persona, "sess_c01")
    head = _word("sess_c01", 0, "u")
    stub = _Stub()
    report = _snapshot(world, "sess_c01", stub)
    assert head in stub.prompts[0]
    for r in rows:
        assert sum(_line(r) in p for p in stub.prompts) == 1, r["text"][:12]
    assert report.completed and report.reached_end


@pytest.mark.parametrize("limit", [_CHUNK, 24_000])
def test_c2_each_transcript_is_whole_turns_within_the_limit(world, monkeypatch, caplog, limit) -> None:
    persona, _, _, transcripts = world
    monkeypatch.setattr(pipeline, "extraction_chunk_chars", lambda: limit, raising=False)
    if limit == _CHUNK:
        _seed(persona, "sess_c02", 14)
        # one multi-line row longer than the limit: sent whole
        ingest_turn(persona, {"session_id": "sess_c02", "speaker": "user", "text": "zx " + "big line\n" * 120,
                              "ts": datetime.now(UTC).isoformat(timespec="seconds")})
    else:
        _long_span(persona, "sess_c02")
    rows = read_session(persona, "sess_c02")
    with caplog.at_level(logging.DEBUG):
        report = _snapshot(world, "sess_c02", _Stub())
    lines = [_line(r) for r in rows]
    for text in transcripts:
        assert text.startswith(("Bob: ", "Kestrel: "))
        # the transcript is exactly a run of whole formatted turns (a multi-line body counts as one turn)
        k = next(i for i, ln in enumerate(lines) if text.startswith(ln))
        j = k
        rebuilt = lines[k]
        while rebuilt != text:
            j += 1
            assert j < len(lines), "transcript is not a run of whole turns"
            rebuilt += "\n" + lines[j]
        if len(text) > limit:
            assert j == k  # only a single over-long turn may exceed the limit
    assert "\n".join(transcripts) == "\n".join(lines)
    assert report.completed
    if limit == _CHUNK:
        assert any("conversation_extract_oversize_unit" in r.getMessage() for r in caplog.records)


def test_c3c_a_deferral_resumes_at_the_next_chunk_with_nothing_skipped_or_repeated(world) -> None:
    persona, _, _, transcripts = world
    rows = _seed(persona, "sess_c03", 12, reply_lag_s=1)
    chunks = _expected_chunks(rows)
    assert len(chunks) >= 3
    first = _snapshot(world, "sess_c03", _Stub(plan=lambda n, p: "defer" if n == 2 else None))
    assert first.deferred and not first.completed
    transcripts.clear()
    second = _snapshot(world, "sess_c03", _Stub())
    assert second.completed
    assert transcripts[0].startswith(_line(chunks[1].turns[0]))
    assert "\n".join(transcripts) == "\n".join(c.text for c in chunks[1:])


def test_c4_cursor_per_chunk(world) -> None:
    persona, _, _, _ = world
    rows = _seed(persona, "sess_c04", 12)
    chunks = _expected_chunks(rows)
    write_backoff(persona, "sess_c04", failures=1, first_failure_at=datetime.now(UTC).isoformat())
    stub = _Stub(persona, "sess_c04")
    report = _snapshot(world, "sess_c04", stub)
    assert stub.cursor_at_call == [None] + [c.cursor_ts for c in chunks[:-1]]
    assert report.completed and report.chunks == len(chunks) == len(stub.prompts)
    assert read_cursor(persona, "sess_c04") == chunks[-1].cursor_ts
    assert read_backoff(persona, "sess_c04") is None  # a successful chunk deletes the backoff file


@pytest.mark.parametrize("k", [1, 2])
def test_c4_a_deferral_at_chunk_k_counts_nothing(world, k) -> None:
    persona, _, _, _ = world
    rows = _seed(persona, "sess_c4d", 12)
    chunks = _expected_chunks(rows)
    report = _snapshot(world, "sess_c4d", _Stub(plan=lambda n, p: "defer" if n == k else None))
    assert report.deferred and report.errors == 0 and not report.completed
    assert read_backoff(persona, "sess_c4d") is None
    assert read_cursor(persona, "sess_c4d") == (chunks[k - 2].cursor_ts if k > 1 else None)


def test_c4_an_extraction_failure_at_chunk_2_records_one_failure(world) -> None:
    persona, _, _, _ = world
    rows = _seed(persona, "sess_c4f", 12)
    chunks = _expected_chunks(rows)
    marker = _word("sess_c4f", 0, "u")
    # chunk 2 fails on every attempt (the extractor retries once)
    second_head = _line(chunks[1].turns[0])
    report = _snapshot(world, "sess_c4f", _Stub(plan=lambda n, p: "fail" if second_head in p else None))
    assert report.errors == 1 and not report.completed
    assert read_backoff(persona, "sess_c4f")["failures"] == 1
    assert read_cursor(persona, "sess_c4f") == chunks[0].cursor_ts
    assert marker  # chunk 1 committed


def test_c4_a_commit_failure_at_chunk_2_increments_the_backoff(world, monkeypatch) -> None:
    persona, _, _, _ = world
    rows = _seed(persona, "sess_c4c", 12)
    chunks = _expected_chunks(rows)
    write_backoff(persona, "sess_c4c", failures=1, first_failure_at=datetime.now(UTC).isoformat())
    real_commit = pipeline.commit_item
    calls = {"n": 0}

    def _commit(item, **kw):
        calls["n"] += 1
        return None if calls["n"] == 2 else real_commit(item, **kw)

    monkeypatch.setattr(pipeline, "commit_item", _commit)
    report = _snapshot(world, "sess_c4c", _Stub(items=True))
    assert report.commit_failures >= 1 and not report.completed
    # chunk 1's success cleared the old backoff; chunk 2's commit failure recorded one through record_extraction_failure
    assert read_backoff(persona, "sess_c4c")["failures"] == 1
    assert read_cursor(persona, "sess_c4c") == chunks[0].cursor_ts


def test_c5_completed_only_when_the_end_was_reached(world) -> None:
    persona, _, _, _ = world
    _seed(persona, "sess_c05", 12)
    bounded = _snapshot(world, "sess_c05", _Stub(), max_chunks=1)
    assert bounded.chunks == 1 and bounded.errors == 0 and not bounded.reached_end and not bounded.completed
    full = _snapshot(world, "sess_c05", _Stub())
    assert full.completed
    empty = _snapshot(world, "sess_c05", _Stub())
    assert empty.chunks == 0 and empty.completed


def test_c7_close_reads_after_the_cursor_and_chunks_the_rest(world) -> None:
    persona, _, _, transcripts = world
    rows = _seed(persona, "sess_c07", 16)
    write_cursor(persona, "sess_c07", rows[7]["ts"])  # the first 4 exchanges are behind the cursor
    write_backoff(persona, "sess_c07", failures=1, first_failure_at=datetime.now(UTC).isoformat())
    stub = _Stub()
    report = _close(world, "sess_c07", stub)
    sent = "\n".join(transcripts)
    assert all(_line(r) not in sent for r in rows[:8])
    assert all(_line(r) in sent for r in rows[8:])
    assert report.completed and report.chunks == len(stub.prompts) > 1
    assert "sess_c07" not in list_active_sessions(persona)
    assert read_cursor(persona, "sess_c07") is None and read_backoff(persona, "sess_c07") is None


def test_c7_close_failure_at_chunk_2_keeps_the_buffer_and_resumes(world) -> None:
    persona, _, _, transcripts = world
    rows = _seed(persona, "sess_c7f", 12)
    chunks = _expected_chunks(rows)
    second_head = _line(chunks[1].turns[0])
    report = _close(world, "sess_c7f", _Stub(plan=lambda n, p: "fail" if second_head in p else None))
    assert not report.completed and report.errors == 1
    assert "sess_c7f" in list_active_sessions(persona)
    assert read_cursor(persona, "sess_c7f") == chunks[0].cursor_ts
    assert read_backoff(persona, "sess_c7f") is None  # close records no backoff (unchanged rule)
    transcripts.clear()
    again = _close(world, "sess_c7f", _Stub())
    assert again.completed
    assert "\n".join(transcripts) == "\n".join(c.text for c in chunks[1:])


def test_c7_a_row_that_arrives_during_the_last_chunk_is_kept(world) -> None:
    persona, _, _, _ = world
    rows = _seed(persona, "sess_c7l", 12)
    last_line = _line(rows[-1])

    def _late():
        ingest_turn(persona, {"session_id": "sess_c7l", "speaker": "user", "text": "late zibbit row",
                              "ts": (datetime.now(UTC) + timedelta(seconds=5)).isoformat(timespec="seconds")})

    # the row is persisted while the call that carries the span's last row runs
    report = _close(world, "sess_c7l", _Stub(plan=lambda n, p: _late if last_line in p else None))
    assert "sess_c7l" in list_active_sessions(persona)
    assert any(r["text"] == "late zibbit row" for r in read_session(persona, "sess_c7l"))
    assert not report.completed and not report.reached_end


def test_c8_compaction_mid_pass_loses_nothing(world) -> None:
    from brain.chat.compaction import compact_conversation

    persona, _, _, _ = world
    rows = _seed(persona, "sess_c08", 12, start=datetime.now(UTC) - timedelta(days=3))
    archived_unprompted: list[str] = []
    poison_results: list[object] = []
    stub = _Stub()

    def _between():
        compact_conversation(persona, "sess_c08", older_than=timedelta(0), fold_existing_summary=True,
                             provider=_SummaryStub(), min_keep_tail=0)
        poison_results.append(pipeline.poison_unextracted_span(persona, "sess_c08"))
        prompted = "\n".join(stub.prompts)
        archived_unprompted.extend(r["text"][:12] for r in read_archive(persona, "sess_c08")
                                   if r.get("speaker") != "summary" and _line(r) not in prompted)

    stub._plan = lambda n, p: _between if n == 2 else None
    report = _snapshot(world, "sess_c08", stub)
    assert report.completed
    assert poison_results == [None]  # busy: the pass holds the extract lock
    assert archived_unprompted == []
    assert any(r.get("speaker") != "summary" for r in read_archive(persona, "sess_c08"))  # compaction did archive chunk 1
    for r in rows:
        assert sum(_line(r) in p for p in stub.prompts) == 1


def test_c9_poison_sets_aside_only_the_failing_chunk(world) -> None:
    persona, _, _, _ = world
    rows = _seed(persona, "sess_c09", 12)
    chunks = _expected_chunks(rows)
    assert len(chunks) >= 3
    write_cursor(persona, "sess_c09", chunks[0].cursor_ts)  # chunk 1 committed
    write_backoff(persona, "sess_c09", failures=3, first_failure_at=datetime.now(UTC).isoformat())
    assert pipeline.poison_unextracted_span(persona, "sess_c09") == len(chunks[1].turns)
    poisoned = [json.loads(ln) for ln in (persona / "active_conversations" / "poison" / "sess_c09.span.jsonl").read_text().splitlines()]
    assert [r["text"] for r in poisoned] == [r["text"] for r in chunks[1].turns]
    assert read_cursor(persona, "sess_c09") == chunks[1].cursor_ts
    assert read_backoff(persona, "sess_c09") is None
    after = [r for r in read_session(persona, "sess_c09") if r["ts"] > chunks[1].cursor_ts]
    assert [r["text"] for r in after] == [r["text"] for c in chunks[2:] for r in c.turns]


def test_c10_a_chunked_pass_says_so(world, caplog) -> None:
    persona, _, _, _ = world
    _seed(persona, "sess_c10", 12)
    stub = _Stub()
    with caplog.at_level(logging.INFO, logger="brain.ingest.pipeline"):
        report = _snapshot(world, "sess_c10", stub)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("conversation_extract_chunked")]
    assert len(lines) == 1 and f"chunks={len(stub.prompts)}" in lines[0]
    assert report.chunks == len(stub.prompts) > 1
    caplog.clear()
    _seed(persona, "sess_c1x", 1)
    with caplog.at_level(logging.INFO, logger="brain.ingest.pipeline"):
        one = _snapshot(world, "sess_c1x", _Stub())
    assert one.chunks == 1
    assert not any(r.getMessage().startswith("conversation_extract_chunked") for r in caplog.records)


def test_c11a_the_sweep_makes_one_call_per_session_and_keeps_work(world) -> None:
    from brain.bridge.supervisor import _snapshot_has_work

    persona, store, hebbian, _ = world
    rows = _seed(persona, "sess_c11", 6)
    assert len(_expected_chunks(rows)) == 3
    stub = _Stub()
    pipeline.snapshot_stale_sessions(persona, silence_minutes=0.0, store=store, hebbian=hebbian, provider=stub)
    assert len(stub.prompts) == 1
    assert _snapshot_has_work(persona) is True


def test_c11d_finalize_makes_one_call_per_run(world) -> None:
    persona, store, hebbian, _ = world
    rows = _seed(persona, "sess_c1d", 6, start=datetime.now(UTC) - timedelta(days=2))
    assert len(_expected_chunks(rows)) == 3
    stub = _Stub()
    pipeline.finalize_stale_sessions(persona, finalize_after_hours=24.0, store=store, hebbian=hebbian, provider=stub)
    assert len(stub.prompts) == 1


def _unbounded(monkeypatch) -> None:
    """The broken variant for C11: every caller's snapshot runs to the end (``max_chunks`` ignored)."""
    real = pipeline.extract_session_snapshot

    def _ignore_bound(*a, max_chunks=None, **kw):
        return real(*a, **kw)

    monkeypatch.setattr(pipeline, "extract_session_snapshot", _ignore_bound)


def test_c11_oracle_catches_unbounded_sweep_and_finalize(world, monkeypatch) -> None:
    """Fail-first kept executable (C11 a/d): with the bound removed the sweep and finalize make all three calls."""
    persona, store, hebbian, _ = world
    _unbounded(monkeypatch)
    _seed(persona, "sess_c1u", 6)
    stub = _Stub()
    pipeline.snapshot_stale_sessions(persona, silence_minutes=0.0, store=store, hebbian=hebbian, provider=stub)
    assert len(stub.prompts) == 3
    _seed(persona, "sess_c1v", 6, start=datetime.now(UTC) - timedelta(days=2))
    stub2 = _Stub()
    pipeline.finalize_stale_sessions(persona, finalize_after_hours=24.0, store=store, hebbian=hebbian, provider=stub2)
    assert len(stub2.prompts) == 3


def test_c11e_close_makes_every_call(world) -> None:
    persona, _, _, _ = world
    rows = _seed(persona, "sess_c1e", 6)
    assert len(_expected_chunks(rows)) == 3
    stub = _Stub()
    assert _close(world, "sess_c1e", stub).completed
    assert len(stub.prompts) == 3


# ---- fail-first variant kept executable (C8): a cursor written to the span end before the loop ----


def test_c8_oracle_catches_a_cursor_written_before_the_loop(world, monkeypatch) -> None:
    from brain.chat.compaction import compact_conversation

    persona, _, _, _ = world
    rows = _seed(persona, "sess_c8v", 12, start=datetime.now(UTC) - timedelta(days=3))
    real_run_chunks = pipeline._run_chunks

    def _early_cursor(persona_dir, session_id, turns, **kw):
        # the variant: the cursor goes to the span end before the loop, instead of to each committed chunk's end
        write_cursor(persona_dir, session_id, rows[-1]["ts"])
        monkeypatch.setattr(pipeline, "write_cursor", lambda *a, **k: None)
        return real_run_chunks(persona_dir, session_id, turns, **kw)

    monkeypatch.setattr(pipeline, "_run_chunks", _early_cursor)
    stub = _Stub()
    unprompted: list[str] = []

    def _between():
        compact_conversation(persona, "sess_c8v", older_than=timedelta(0), fold_existing_summary=True,
                             provider=_SummaryStub(), min_keep_tail=0)
        prompted = "\n".join(stub.prompts)
        unprompted.extend(r["text"][:12] for r in read_archive(persona, "sess_c8v")
                          if r.get("speaker") != "summary" and _line(r) not in prompted)

    stub._plan = lambda n, p: _between if n == 2 else None
    _snapshot(world, "sess_c8v", stub)
    assert unprompted != []


# ---------------------------------------------------------------------------
# The rollover (C6, C9 second half, C11e weekly): real perform_rollover, compaction provider stubbed
# ---------------------------------------------------------------------------


@pytest.fixture
def rollover_world(world, monkeypatch):
    import brain.chat.rollover as rollover

    monkeypatch.setattr(rollover, "build_compaction_provider", lambda persona_dir: _SummaryStub())
    return world


def _roll(world_, sid, provider, **kw):
    from brain.chat.rollover import perform_rollover

    persona, store, hebbian, _ = world_
    return perform_rollover(persona, sid, "Kestrel", seed_mode="summary_only", provider=provider, store=store, hebbian=hebbian, **kw)


def test_c6_the_rollover_extracts_every_chunk_before_deleting(rollover_world, monkeypatch) -> None:
    persona, _, _, _ = rollover_world
    monkeypatch.setattr(pipeline, "extraction_chunk_chars", lambda: 24_000, raising=False)
    rows = _seed(persona, "sess_c06", 300, gap_s=30)
    assert len(chunk_turns(rows, max_chars=24_000, user_name="Bob", assistant_name="Kestrel")) == 3
    buffer_present: list[bool] = []
    stub = _Stub(plan=lambda n, p: (lambda: buffer_present.append("sess_c06" in list_active_sessions(persona))))
    new_sid = _roll(rollover_world, "sess_c06", stub)
    for r in rows:
        assert sum(_line(r) in p for p in stub.prompts) == 1, r["text"][:12]
    assert new_sid is not None and "sess_c06" not in list_active_sessions(persona)
    assert buffer_present == [True, True, True]


def test_c6_a_deferral_at_chunk_2_keeps_the_old_buffer(rollover_world) -> None:
    persona, _, _, _ = rollover_world
    rows = _seed(persona, "sess_c6d", 6)
    chunks = _expected_chunks(rows)
    assert len(chunks) == 3
    assert _roll(rollover_world, "sess_c6d", _Stub(plan=lambda n, p: "defer" if n == 2 else None)) is None
    assert "sess_c6d" in list_active_sessions(persona)
    assert read_cursor(persona, "sess_c6d") == chunks[0].cursor_ts


@pytest.mark.parametrize("rerun_ok", [True, False])
def test_c9_after_a_poison_the_rollover_extracts_the_rest_or_keeps_the_buffer(rollover_world, rerun_ok) -> None:
    persona, _, _, _ = rollover_world
    rows = _seed(persona, "sess_c9r", 8)
    chunks = _expected_chunks(rows)
    assert len(chunks) == 4
    write_cursor(persona, "sess_c9r", chunks[0].cursor_ts)  # chunk 1 committed earlier
    write_backoff(persona, "sess_c9r", failures=2, first_failure_at=(datetime.now(UTC) - timedelta(minutes=30)).isoformat())
    second = _line(chunks[1].turns[0])
    later = [_line(c.turns[0]) for c in chunks[2:]]

    def plan(n, p):
        if second in p:
            return "fail"  # chunk 2 fails: its third recorded failure
        if not rerun_ok and any(h in p for h in later):
            return "fail"
        return None

    stub = _Stub(plan=plan)
    new_sid = _roll(rollover_world, "sess_c9r", stub)
    poisoned = (persona / "active_conversations" / "poison" / "sess_c9r.span.jsonl").read_text().splitlines()
    assert [json.loads(x)["text"] for x in poisoned] == [r["text"] for r in chunks[1].turns]
    if rerun_ok:
        assert new_sid is not None and "sess_c9r" not in list_active_sessions(persona)
        for c in chunks[2:]:
            assert all(any(_line(r) in p for p in stub.prompts) for r in c.turns)
    else:
        assert new_sid is None and "sess_c9r" in list_active_sessions(persona)
        assert read_cursor(persona, "sess_c9r") == chunks[1].cursor_ts


def test_c11e_the_weekly_rollover_makes_every_call(rollover_world) -> None:
    from brain.chat.rollover import maybe_weekly_rollover

    persona, store, hebbian, _ = rollover_world
    rows = _seed(persona, "sess_c1w", 6, start=datetime.now(UTC) - timedelta(days=8))
    assert len(_expected_chunks(rows)) == 3
    stub = _Stub()
    new_sid = maybe_weekly_rollover(
        persona, "sess_c1w", "Kestrel", weekly_age=timedelta(days=7), quiet_gap=timedelta(minutes=10),
        provider=stub, store=store, hebbian=hebbian,
    )
    assert len(stub.prompts) == 3
    assert new_sid is not None


def test_c11a_three_central_passes_extract_the_three_chunks(world, monkeypatch) -> None:
    """The sweep as the supervisor runs it: the real job table through ``run_central_pass``, one chunk per idle pass."""
    from brain.bridge import central_cadence, supervisor

    persona, _, _, _ = world
    rows = _seed(persona, "sess_c1p", 6)
    chunks = _expected_chunks(rows)
    assert len(chunks) == 3
    stub = _Stub()
    monkeypatch.setattr(supervisor, "build_tier_provider", lambda persona_dir, tier: stub)

    class _Bus:
        def publish(self, event):
            pass

    jobs = supervisor._build_gated_jobs(
        persona_dir=persona, provider=stub, event_bus=_Bus(), is_session_busy=None, finalize_after_hours=24.0,
        finalize_interval_s=None, initiate_review_interval_s=None, maintenance_interval_s=None, self_model_interval_s=None,
        compaction_interval_s=None, calibration_interval_s=None, interest_sweep_interval_s=None,
        judge_selftune_interval_s=None, clustering_interval_s=None, intensity_drivers=lambda: None,
        tick_stats={"pruned_empty_sessions": 0, "closed_sessions": 0},
    )
    for k in range(1, 4):
        central_cadence.run_central_pass(persona, jobs, is_idle=lambda: True, slot_available=lambda: True)
        assert len(stub.prompts) == k
        assert read_cursor(persona, "sess_c1p") == chunks[k - 1].cursor_ts
    assert supervisor._snapshot_has_work(persona) is False
