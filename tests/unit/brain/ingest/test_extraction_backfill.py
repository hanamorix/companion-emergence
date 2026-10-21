"""#523: the extraction backfill (`nell memory backfill-extraction`) re-extracts conversation text the old tail cut never
showed the extractor. Synthetic persona only (persona Kestrel, user Bob, made-up words). Criteria B1-B16 of the #523 run.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta

import pytest

from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.cli_throttle import ThrottleDeferred
from brain.bridge.provider import LLMProvider
from brain.engines.consolidation import Decision, run_consolidation
from brain.ingest import extraction_backfill as eb
from brain.ingest.buffer import append_archive, ingest_turn, write_cursor
from brain.ingest.commit import commit_item
from brain.ingest.extract import format_turn_line
from brain.ingest.ingest_utils import chunk_turns
from brain.ingest.types import ExtractedItem
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore
from brain.persona_config import PersonaConfig
from brain.utils.time import lenient_utc

_PROMOTE_ALL = lambda _cand, _ctx: Decision("new")  # noqa: E731


@pytest.fixture
def stores(tmp_path):
    persona = tmp_path / "Kestrel"
    persona.mkdir()
    store = MemoryStore(persona / "memories.db")
    hebbian = HebbianMatrix(persona / "hebbian.db")
    yield persona, store, hebbian
    store.close()
    hebbian.close()


# ---------------------------------------------------------------------------
# commit_item: the back-date and the marker survive the pending queue and the gate (B9/B15 plumbing)
# ---------------------------------------------------------------------------


def test_commit_item_back_dates_and_marks_through_the_gate(stores) -> None:
    persona, store, hebbian = stores
    when = datetime(2026, 7, 3, 21, 14, 5, tzinfo=UTC)
    marker = {"extraction_backfill": {"from": "2026-07-03T21:00:00+00:00", "to": "2026-07-03T21:14:05+00:00"}}
    item = ExtractedItem(text="Bob keeps a jar of snorvel buttons by the door", label="fact", importance=6)
    mem_id = commit_item(item, session_id="sess_old", store=store, hebbian=hebbian, extra_metadata=marker, created_at=when)
    assert mem_id is not None
    run_consolidation(store, persona_dir=persona, classifier=_PROMOTE_ALL)
    mem = store.get(mem_id)
    assert mem is not None
    assert mem.created_at == when
    assert mem.metadata["extraction_backfill"] == marker["extraction_backfill"]
    assert mem.metadata["source_summary"] == "conversation:sess_old"


def test_commit_item_without_the_new_arguments_dates_now(stores) -> None:
    persona, store, hebbian = stores
    before = datetime.now(UTC)
    item = ExtractedItem(text="Bob named the kettle Plonk", label="fact", importance=6)
    mem_id = commit_item(item, session_id="sess_new", store=store, hebbian=hebbian)
    run_consolidation(store, persona_dir=persona, classifier=_PROMOTE_ALL)
    mem = store.get(mem_id)
    assert mem is not None and mem.created_at >= before.replace(microsecond=0)
    assert "extraction_backfill" not in mem.metadata


# ---------------------------------------------------------------------------
# Fixtures for the backfill runs
# ---------------------------------------------------------------------------

_CHUNK = 600
_USER = "Bob"


class _RunawayLoopError(AssertionError):
    """Raised by the hard bounds below, so a broken stop rule FAILS the test at once instead of looping until the CI job times
    out. The pass bound raises from the wrapper around ``run_central_pass``, outside every handler of the code under test; the
    call bound is a second net (inside the extractor it counts as a failed call)."""


_CALL_LIMIT = 60
_PASS_LIMIT = 60


@pytest.fixture(autouse=True)
def _bounded_central_passes(monkeypatch):
    from brain.bridge import central_cadence

    real = central_cadence.run_central_pass
    passes = {"n": 0}

    def _bounded(*a, **k):
        passes["n"] += 1
        if passes["n"] > _PASS_LIMIT:
            raise _RunawayLoopError(f"more than {_PASS_LIMIT} central passes: a stop rule is broken")
        return real(*a, **k)

    monkeypatch.setattr(central_cadence, "run_central_pass", _bounded)


class _Stub(LLMProvider):
    """Records every extraction prompt; ``plan(n, prompt)`` may return "defer", "fail", "kbint" or a callable to run first.
    Answers one distinct fact per call (importance ``importance``) unless ``items`` is False."""

    def __init__(self, plan=None, *, items=True, importance=5, text=None):
        self.prompts: list[str] = []
        self._plan, self._items, self._importance, self._text = plan, items, importance, text

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.prompts.append(prompt)
        n = len(self.prompts)
        if len(self.prompts) > _CALL_LIMIT:
            raise _RunawayLoopError(f"more than {_CALL_LIMIT} model calls: a stop rule is broken")
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
        text = self._text or f"Bob mentioned the quorble number {n * 7919} on that evening"
        return json.dumps([{"text": text, "label": "fact", "importance": self._importance}])

    def name(self) -> str:
        return "stub-backfill"

    def chat(self, messages: list[ChatMessage], *, tools=None, options=None) -> ChatResponse:
        return ChatResponse(content="[]", tool_calls=())


@pytest.fixture
def kestrel(tmp_path, monkeypatch):
    persona = tmp_path / "Kestrel"
    persona.mkdir()
    PersonaConfig(user_name=_USER).save(persona / "persona_config.json")
    monkeypatch.setattr(eb, "extraction_chunk_chars", lambda: _CHUNK)
    store = MemoryStore(persona / "memories.db")
    hebbian = HebbianMatrix(persona / "hebbian.db")
    yield persona, store, hebbian
    store.close()
    hebbian.close()
    eb._TEST_SEAMS.clear()


def _rows(sid: str, start: datetime, n: int, *, gap_s: int = 60, lag_s: int = 1, size: int = 90, tag: str = "") -> list[dict]:
    """``n`` exchanges; each reply stamped ``lag_s`` after its user row; each row carries a unique made-up word."""
    out = []
    for i in range(n):
        t = start + timedelta(seconds=i * gap_s)
        out.append({"session_id": sid, "speaker": "user", "text": f"zv{tag}{sid[-2:]}u{i:03d} " + "m" * size, "ts": t.isoformat(timespec="seconds")})
        out.append({"session_id": sid, "speaker": "assistant", "text": f"zv{tag}{sid[-2:]}a{i:03d} " + "n" * size,
                    "ts": (t + timedelta(seconds=lag_s)).isoformat(timespec="seconds")})
    return out


def _archive(persona, sid, rows) -> None:
    append_archive(persona, sid, rows)


def _active(persona, sid, rows, *, cursor_at: int | None) -> None:
    for r in rows:
        ingest_turn(persona, dict(r))
    if cursor_at is not None:
        write_cursor(persona, sid, rows[cursor_at]["ts"])


def _line(row: dict) -> str:
    return format_turn_line(row, _USER, "Kestrel")


def _run(persona, store, hebbian, stub, *, now=None) -> eb.Step:
    return eb.run_one_chunk(persona, store=store, hebbian=hebbian, provider=stub, now=now)


def _run_all(persona, store, hebbian, stub, limit: int = 200) -> None:
    for _ in range(limit):
        if not eb.has_work(persona):
            return
        _run(persona, store, hebbian, stub)
    raise AssertionError("the plan did not finish")


def _arm(persona, mode) -> dict:
    _outcome, msg = eb.arm(persona, mode)
    state = eb.load_state(persona)
    assert state is not None and state["mode"] == mode, msg
    return state


_OLD = datetime(2026, 8, 3, 19, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# B1 [H]: the heads planner covers what the lull MODEL says the old cap cut
# ---------------------------------------------------------------------------


def _model_passes(rows: list[dict]) -> list[list[dict]]:
    """Pass times as the old code ran them per the MODEL: a 60 s tick (at whole minutes); a pass at the first tick at least
    300 s (before 2026-09-07T20:51:12Z) or 600 s (after) past the last row; each pass takes every row since the last one."""
    passes: list[list[dict]] = []
    pending: list[dict] = []
    for i, row in enumerate(rows):
        pending.append(row)
        ts = lenient_utc(row["ts"])
        nxt = lenient_utc(rows[i + 1]["ts"]) if i + 1 < len(rows) else None
        tick = ts.replace(second=0) + timedelta(minutes=1)
        while nxt is None or tick < nxt:
            lull = 300 if tick < eb._LULL_BOUNDARY else 600
            if (tick - ts).total_seconds() >= lull:
                passes.append(pending)
                pending = []
                break
            tick += timedelta(minutes=1)
            if nxt is None and (tick - ts).total_seconds() > 3600:
                break
    if pending:
        passes.append(pending)
    return passes


def _never_shown(passes: list[list[dict]], user_label: str, assistant_label: str, cap: int) -> list[dict]:
    """Replay the pre-change cut (``text[-cap:]`` of the formatted pass) and return the rows not shown in full."""
    out = []
    for p in passes:
        lines = [format_turn_line(r, user_label, assistant_label) for r in p]
        text = "\n".join(lines)
        start_of_kept = max(0, len(text) - cap)
        offset = 0
        for r, ln in zip(p, lines, strict=True):
            if offset < start_of_kept:
                out.append(r)
            offset += len(ln) + 1
    return out


def _b1_history(persona) -> tuple[str, list[dict]]:
    """One session, archived. Pre-boundary: a span split by a 330 s gap with no tick in [300, 330) after its last row (no
    pass under the model). Post-boundary: a span long enough to be cut, with a long user name in force."""
    sid = "sess_b1"
    # last row of the first part at ss = 25, so the ticks fall at +35, +95, ..., +275, +335: none in [300, 330)
    a1 = _rows(sid, datetime(2026, 9, 1, 18, 0, 25, tzinfo=UTC) - timedelta(seconds=60 * 13), 14, gap_s=60, lag_s=0, tag="a")
    last_a1 = lenient_utc(a1[-1]["ts"])
    assert last_a1.second == 25
    a2 = _rows(sid, last_a1 + timedelta(seconds=330), 10, gap_s=60, lag_s=0, tag="b")
    b = _rows(sid, datetime(2026, 9, 20, 18, 0, 0, tzinfo=UTC), 24, gap_s=60, tag="c")
    rows = a1 + a2 + b
    _archive(persona, sid, rows)
    return sid, rows


_LONG_USER = "Bob Quillamore-Vantrexley"  # the configured user name in force for the B1 history: longer than "user"
_LONG_PERSONA = "Kestrelwingfeather-Quillamoor-Vantrexhollow"  # the persona dir name for B1: longer than "assistant", so both label halves matter


def _b1_world(persona, monkeypatch) -> tuple[object, str, list[dict], list[dict]]:
    """The B1 fixture (amended B1, main ruling M3): the cut replayed with the labels IN FORCE (the configured user name and the
    persona dir name), and the fixture's own checks that this set is strictly larger than the shortest-label set and than each
    half alone (follow-up: with a persona name shorter than "assistant" the assistant half could not fail)."""
    b1 = persona.parent / _LONG_PERSONA
    b1.mkdir()
    PersonaConfig(user_name=_LONG_USER).save(b1 / "persona_config.json")
    monkeypatch.setattr(eb, "_HISTORICAL_CAP_CHARS", 2_500)
    sid, rows = _b1_history(b1)
    passes = _model_passes(rows)
    in_force = {r["text"] for r in _never_shown(passes, _LONG_USER, _LONG_PERSONA, 2_500)}
    for user_label, assistant_label in (("user", "assistant"), ("user", _LONG_PERSONA), (_LONG_USER, "assistant")):
        weaker = {r["text"] for r in _never_shown(passes, user_label, assistant_label, 2_500)}
        assert weaker < in_force, f"the fixture must make the labels matter: {user_label!r}, {assistant_label!r}"
    return b1, sid, rows, [r for r in rows if r["text"] in in_force]


def _covered(persona, sid, rows_cut) -> list[dict]:
    targets = eb.plan_heads(persona, datetime.now(UTC).isoformat())
    spans = [(lenient_utc(t["from_ts"]), lenient_utc(t["to_ts"])) for t in targets if t["sid"] == sid]
    return [r for r in rows_cut if not any(lo <= lenient_utc(r["ts"]) <= hi for lo, hi in spans)]


def test_b1_the_heads_plan_covers_every_row_the_model_says_was_cut(kestrel, monkeypatch) -> None:
    persona, _, _ = kestrel
    b1, sid, _rows_, in_force = _b1_world(persona, monkeypatch)
    assert _covered(b1, sid, in_force) == []


def test_b1_oracle_catches_a_planner_with_the_300_s_rule(kestrel, monkeypatch) -> None:
    """Fail-first kept executable: with the pre-boundary lull at 300 s the 330 s gap splits the span and its head is missed."""
    persona, _, _ = kestrel
    b1, sid, _rows_, in_force = _b1_world(persona, monkeypatch)
    monkeypatch.setattr(eb, "_PRE_LULL_UPPER_S", 300)
    assert _covered(b1, sid, in_force) != []


@pytest.mark.parametrize(
    "labels",
    [("user", "assistant"), ("user", _LONG_PERSONA), (_LONG_USER, "assistant")],
    ids=["both-shortest", "shortest-user", "shortest-assistant"],
)
def test_b1_oracle_catches_a_planner_with_the_shortest_labels(kestrel, monkeypatch, labels) -> None:
    """Fail-first kept executable (amended B1): a planner measuring with the shortest labels, or the shortest of either half,
    misses rows of the in-force cut."""
    persona, _, _ = kestrel
    b1, sid, _rows_, in_force = _b1_world(persona, monkeypatch)
    monkeypatch.setattr(eb, "_plan_labels", lambda p: labels)
    assert _covered(b1, sid, in_force) != []


# ---------------------------------------------------------------------------
# B2 [A]: the everything planner
# ---------------------------------------------------------------------------


def test_b2_every_row_behind_the_cursor_is_in_exactly_one_target(kestrel) -> None:
    persona, _, _ = kestrel
    old = _rows("sess_b2a", _OLD, 6)
    _archive(persona, "sess_b2a", old + [{"session_id": "sess_b2a", "speaker": "summary", "text": "a recap", "ts": old[-1]["ts"]}])
    live = _rows("sess_b2b", _OLD + timedelta(days=3), 6)
    _active(persona, "sess_b2b", live, cursor_at=7)  # rows 0..7 are behind the cursor
    targets = eb.plan_all(persona, datetime.now(UTC).isoformat())
    assert [t["sid"] for t in targets] == ["sess_b2a", "sess_b2b"]  # oldest first
    expected = old + live[:8]
    found = [r for t in targets for r in eb._target_rows(persona, t, datetime.now(UTC).isoformat())]
    assert [r["text"] for r in found] == [r["text"] for r in expected]
    assert not any(r["text"] == live[8]["text"] for r in found)


# ---------------------------------------------------------------------------
# B3: dry run writes nothing and calls nothing
# ---------------------------------------------------------------------------


def _listing(root) -> list[tuple]:
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*"))


def test_b3_dry_run_writes_nothing_and_calls_nothing(kestrel) -> None:
    persona, _, _ = kestrel
    _active(persona, "sess_b3", _rows("sess_b3", _OLD, 8), cursor_at=13)
    assert not (persona / "archived_conversations").exists()
    stub = _Stub()
    before = _listing(persona)
    text = eb.dry_run_text(persona)
    assert _listing(persona) == before
    assert stub.prompts == []
    assert "mode heads" in text and "mode all" in text and "sess_b3" in text and "calls=" in text and "chars=" in text


def test_b3_oracle_catches_a_reader_that_creates_the_archive_dir(kestrel, monkeypatch) -> None:
    from brain.ingest import buffer

    persona, _, _ = kestrel
    _active(persona, "sess_b3v", _rows("sess_b3v", _OLD, 8), cursor_at=13)
    monkeypatch.setattr(eb, "_archive_files", lambda p, sid: buffer._archive_read_order(p, sid))
    before = _listing(persona)
    eb.dry_run_text(persona)
    assert _listing(persona) != before


# ---------------------------------------------------------------------------
# B4: resume after a kill
# ---------------------------------------------------------------------------


def _foreground(persona, store, hebbian, stub, outputs=None):
    return eb.run_foreground(
        persona, store=store, hebbian=hebbian, provider=stub, bridge_running=lambda: False,
        out=(outputs.append if outputs is not None else (lambda _m: None)),
    )


def test_b4_a_kill_mid_run_loses_nothing_and_repeats_nothing(kestrel) -> None:
    persona, store, hebbian = kestrel
    rows = _rows("sess_b4", _OLD, 12)
    _archive(persona, "sess_b4", rows)
    _arm(persona, "all")
    chunks = chunk_turns(rows, max_chars=_CHUNK, user_name=_USER, assistant_name="Kestrel")
    k = 2
    assert len(chunks) > k + 1
    first = _Stub(plan=lambda n, p: "kbint" if n == k + 1 else None)
    with pytest.raises(KeyboardInterrupt):
        _foreground(persona, store, hebbian, first)
    assert eb.load_state(persona)["targets"][0]["done_through_ts"] == chunks[k - 1].cursor_ts
    second = _Stub()
    assert _foreground(persona, store, hebbian, second) == "complete"
    prompts = first.prompts[:k] + second.prompts
    for r in rows:
        assert sum(_line(r) in p for p in prompts) == 1, r["text"][:12]


def test_b4_oracle_catches_progress_saved_before_the_commits(kestrel, monkeypatch) -> None:
    persona, store, hebbian = kestrel
    rows = _rows("sess_b4v", _OLD, 12)
    _archive(persona, "sess_b4v", rows)
    _arm(persona, "all")
    real = eb.run_chunk_stages

    def _save_first(persona_dir, sid, transcript, **kw):
        last = [r for r in rows if _line(r) in transcript][-1]

        def fn(cur):
            cur["targets"][0]["done_through_ts"] = last["ts"]
            return cur

        eb.update_state(persona_dir, fn)
        return real(persona_dir, sid, transcript, **kw)

    monkeypatch.setattr(eb, "run_chunk_stages", _save_first)
    first = _Stub(plan=lambda n, p: "kbint" if n == 3 else None)
    with pytest.raises(KeyboardInterrupt):
        _foreground(persona, store, hebbian, first)
    second = _Stub()
    _foreground(persona, store, hebbian, second)
    prompts = first.prompts[:2] + second.prompts
    assert any(sum(_line(r) in p for p in prompts) != 1 for r in rows)


# ---------------------------------------------------------------------------
# B5: no lost update on the state file
# ---------------------------------------------------------------------------


def _b5(persona, store, hebbian) -> dict:
    _archive(persona, "sess_b5", _rows("sess_b5", _OLD, 8))
    _arm(persona, "all")
    s_may_go, s_done = threading.Event(), threading.Event()

    def _inside_rmw():
        s_may_go.set()
        s_done.wait(timeout=0.5)  # with the real lock S cannot finish before this write

    eb._TEST_SEAMS["between_read_and_write"] = _inside_rmw

    def _s():
        s_may_go.wait(timeout=5)
        eb.stop(persona)
        s_done.set()

    t = threading.Thread(target=_s)
    t.start()
    step = _run(persona, store, hebbian, _Stub())
    t.join(timeout=10)
    assert step.advanced
    return eb.load_state(persona)


def test_b5_a_stop_during_a_progress_write_is_not_lost(kestrel) -> None:
    persona, store, hebbian = kestrel
    state = _b5(persona, store, hebbian)
    assert state["targets"][0]["done_through_ts"] is not None
    assert state["status"] == "stopped"


def test_b5_oracle_catches_a_missing_state_lock(kestrel, monkeypatch) -> None:
    from contextlib import nullcontext

    persona, store, hebbian = kestrel
    monkeypatch.setattr(eb, "file_lock", lambda *a, **k: nullcontext(True))
    state = _b5(persona, store, hebbian)
    assert state["status"] != "stopped"  # the job's stale read overwrote the stop


# ---------------------------------------------------------------------------
# B6: plan identity
# ---------------------------------------------------------------------------


def test_b6_a_chunk_from_an_old_plan_changes_nothing_but_discarded(kestrel) -> None:
    persona, store, hebbian = kestrel
    _archive(persona, "sess_b6", _rows("sess_b6", _OLD, 8))
    _arm(persona, "all")
    replaced: list[dict] = []

    def _restart_during_the_call():
        replaced.append(eb.update_state(persona, lambda _cur: eb._new_state(persona, "all")))

    step = _run(persona, store, hebbian, _Stub(plan=lambda n, p: _restart_during_the_call if n == 1 else None))
    after = eb.load_state(persona)
    assert step.reason == "plan changed" and not step.advanced
    expected = json.loads(json.dumps(replaced[0]))
    expected["counters"]["discarded"] = 1
    assert after == expected


def test_b6_oracle_catches_a_write_without_the_plan_check(kestrel, monkeypatch) -> None:
    persona, store, hebbian = kestrel
    _archive(persona, "sess_b6v", _rows("sess_b6v", _OLD, 8))
    _arm(persona, "all")
    monkeypatch.setattr(eb, "_same_plan", lambda cur, plan_id: True)
    _run(persona, store, hebbian, _Stub(plan=lambda n, p: (lambda: eb.update_state(persona, lambda _c: eb._new_state(persona, "all")))
                                         if n == 1 else None))
    assert eb.load_state(persona)["targets"][0]["done_through_ts"] is not None  # Q's target marked done by P's chunk


def test_b6_restart_is_refused_while_a_run_holds_the_runner_lock(kestrel) -> None:
    from brain.utils.file_lock import file_lock

    persona, _, _ = kestrel
    _archive(persona, "sess_b6r", _rows("sess_b6r", _OLD, 8))
    _arm(persona, "all")
    before = eb.state_path(persona).read_bytes()
    with file_lock(persona / eb._RUNNER_LOCK):
        outcome, msg = eb.restart(persona, "all")
    assert outcome == "refused" and "in progress" in msg
    assert eb.state_path(persona).read_bytes() == before


# ---------------------------------------------------------------------------
# B7: one runner
# ---------------------------------------------------------------------------


def test_b7_a_held_runner_lock_skips_the_job_and_ends_the_foreground_run(kestrel) -> None:
    from contextlib import nullcontext

    from brain.utils.file_lock import file_lock

    persona, store, hebbian = kestrel
    _archive(persona, "sess_b7", _rows("sess_b7", _OLD, 8))
    _arm(persona, "all")
    stub = _Stub()
    job = eb.backfill_gated_job(persona, store_ctx=lambda: nullcontext(store), hebbian_ctx=lambda: nullcontext(hebbian),
                                provider_factory=lambda: stub)
    outputs: list[str] = []
    with file_lock(persona / eb._RUNNER_LOCK):
        assert job.run() is eb.JobOutcome.SKIPPED
        assert _foreground(persona, store, hebbian, stub, outputs) == "not advanced"
    assert stub.prompts == []
    assert "another backfill run" in outputs[-1] and "Progress is kept" in outputs[-1]


# ---------------------------------------------------------------------------
# B8 (foreground half): through run_central_pass, and the stop rule
# ---------------------------------------------------------------------------


def test_b8_the_foreground_loop_uses_the_central_cadence_function(kestrel, monkeypatch) -> None:
    from brain.bridge import central_cadence

    persona, store, hebbian = kestrel
    _archive(persona, "sess_b8", _rows("sess_b8", _OLD, 8))
    _arm(persona, "all")
    real = central_cadence.run_central_pass
    passes: list[list[str]] = []

    def _spy(persona_dir, jobs, **kw):
        passes.append([j.name for j in jobs])
        return real(persona_dir, jobs, **kw)

    monkeypatch.setattr(central_cadence, "run_central_pass", _spy)
    stub = _Stub()
    assert _foreground(persona, store, hebbian, stub) == "complete"
    assert passes and all(p == ["extraction_backfill"] for p in passes)
    assert len(stub.prompts) == len(passes)


def test_b8_a_recent_saved_user_turn_means_no_call(kestrel) -> None:
    persona, store, hebbian = kestrel
    _archive(persona, "sess_b8n", _rows("sess_b8n", _OLD, 8))
    ingest_turn(persona, {"session_id": "sess_b8live", "speaker": "user", "text": "fresh glimmet", "ts": datetime.now(UTC).isoformat()})
    _arm(persona, "all")
    stub, outputs = _Stub(), []
    assert _foreground(persona, store, hebbian, stub, outputs) == "not advanced"
    assert stub.prompts == []
    assert "chat is not idle" in outputs[-1] and "Progress is kept" in outputs[-1]


def test_b8_a_deferral_ends_the_foreground_run_after_one_call(kestrel) -> None:
    persona, store, hebbian = kestrel
    _archive(persona, "sess_b8d", _rows("sess_b8d", _OLD, 8))
    _arm(persona, "all")
    stub, outputs = _Stub(plan=lambda n, p: "defer"), []  # a broken stop rule hits the pass bound and fails
    assert _foreground(persona, store, hebbian, stub, outputs) == "not advanced"
    assert len(stub.prompts) == 1
    assert "refused for now" in outputs[-1] and "Progress is kept" in outputs[-1]
    assert eb.load_state(persona)["targets"][0]["done_through_ts"] is None


def test_b8_a_plan_stopped_between_passes_ends_the_run(kestrel, monkeypatch) -> None:
    from brain.bridge import central_cadence

    persona, store, hebbian = kestrel
    _archive(persona, "sess_b8s", _rows("sess_b8s", _OLD, 8))
    _arm(persona, "all")
    real = central_cadence.run_central_pass
    calls = {"n": 0}

    def _then_stop(persona_dir, jobs, **kw):
        out = real(persona_dir, jobs, **kw)
        calls["n"] += 1
        if calls["n"] == 1:
            eb.stop(persona_dir)  # as another terminal would
        return out

    monkeypatch.setattr(central_cadence, "run_central_pass", _then_stop)
    stub, outputs = _Stub(), []
    assert _foreground(persona, store, hebbian, stub, outputs) == "stopped"
    assert len(stub.prompts) == 1
    assert "stopped" in outputs[-1] and "Progress is kept" in outputs[-1]


def test_b8_oracle_catches_a_loop_that_stops_only_without_a_lull(kestrel, monkeypatch) -> None:
    persona, store, hebbian = kestrel
    _archive(persona, "sess_b8v", _rows("sess_b8v", _OLD, 8))
    _arm(persona, "all")
    monkeypatch.setattr(eb, "_keep_going", lambda b, a, decisions, o: not any(d.action == "skip-no-lull" for d in decisions))
    stub = _Stub(plan=lambda n, p: "kbint" if n > 3 else "defer")  # the call limit: a broken loop cannot hang the test
    with pytest.raises(KeyboardInterrupt):
        _foreground(persona, store, hebbian, stub)
    assert len(stub.prompts) >= 2


# ---------------------------------------------------------------------------
# B9: dedup and the marker; B10: the live session is left alone
# ---------------------------------------------------------------------------


def test_b9_items_are_deduplicated_and_marked(kestrel, monkeypatch) -> None:
    from brain.bridge import model_tier
    from brain.memory import embeddings as embeddings_mod
    from brain.memory.store import Memory

    persona, store, hebbian = kestrel
    provider = embeddings_mod.FakeEmbeddingProvider(dim=384)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, provider.model_id())
    known = "Bob keeps three snorvel buttons in a blue jar"
    m = Memory.create_new(content=known, memory_type="fact", domain="brain", emotions={})
    store.create(m)
    store.embed_row(m.id, m.content)
    rows = _rows("sess_b9", _OLD, 12)
    _archive(persona, "sess_b9", rows)
    _arm(persona, "all")
    _run(persona, store, hebbian, _Stub(text=known))
    assert eb.load_state(persona)["counters"]["deduped"] == 1
    _run_all(persona, store, hebbian, _Stub())
    run_consolidation(store, persona_dir=persona, classifier=_PROMOTE_ALL)
    made = [x for x in store.list_active(limit=100) if x.id != m.id]
    assert made
    for mem in made:
        assert mem.metadata["source_summary"] == "conversation:sess_b9"
        assert set(mem.metadata["extraction_backfill"]) == {"from", "to"}


def test_b10_the_live_session_files_are_untouched(kestrel) -> None:
    persona, store, hebbian = kestrel
    live = _rows("sess_b10", _OLD, 10)
    _active(persona, "sess_b10", live, cursor_at=15)
    from brain.ingest.buffer import write_backoff

    write_backoff(persona, "sess_b10", failures=1, first_failure_at=datetime.now(UTC).isoformat())
    (persona / "active_conversations" / "sess_b10.extract.lock").write_bytes(b"")
    before = {p.name: p.read_bytes() for p in (persona / "active_conversations").iterdir() if p.is_file()}
    _arm(persona, "all")
    _run_all(persona, store, hebbian, _Stub())
    after = {p.name: p.read_bytes() for p in (persona / "active_conversations").iterdir() if p.is_file()}
    assert after == before


# ---------------------------------------------------------------------------
# B11: rows carried by a rollover; a row that exists nowhere
# ---------------------------------------------------------------------------


def _b11_world(persona, monkeypatch):
    import brain.chat.rollover as rollover

    class _Summary:
        def generate(self, prompt=None, *, system=None, **kw):
            return "Bob and Kestrel talked about many made-up things."

    monkeypatch.setattr(rollover, "build_compaction_provider", lambda p: _Summary())
    rows = _rows("sess_b11", _OLD, 40)  # 80 rows: the tail of 40 is carried into the successor
    _active(persona, "sess_b11", rows, cursor_at=79)
    return rows


def test_b11_rows_carried_by_a_rollover_are_found_through_rolled_to(kestrel, monkeypatch) -> None:
    from brain.chat.rollover import perform_rollover

    persona, store, hebbian = kestrel
    rows = _b11_world(persona, monkeypatch)
    _arm(persona, "all")
    new_sid = perform_rollover(persona, "sess_b11", "Kestrel", seed_mode="tiers_plus_tail")
    assert new_sid is not None
    stub = _Stub()
    _run_all(persona, store, hebbian, stub)
    for r in rows:
        assert sum(_line(r) in p for p in stub.prompts) == 1, r["text"][:12]
    assert eb.load_state(persona)["counters"]["missing"] == 0


def test_b11_oracle_catches_a_reader_of_the_original_sid_only(kestrel, monkeypatch) -> None:
    from brain.chat.rollover import perform_rollover

    persona, store, hebbian = kestrel
    rows = _b11_world(persona, monkeypatch)
    _arm(persona, "all")
    perform_rollover(persona, "sess_b11", "Kestrel", seed_mode="tiers_plus_tail")
    monkeypatch.setattr(eb, "_rolled_to", lambda p, sid: None)
    stub = _Stub()
    _run_all(persona, store, hebbian, stub)
    assert any(not any(_line(r) in p for p in stub.prompts) for r in rows)


def test_b11_a_row_that_exists_nowhere_is_counted_missing(kestrel) -> None:
    from brain.ingest.buffer import rewrite_session_atomic

    persona, store, hebbian = kestrel
    rows = _rows("sess_b11m", _OLD, 6)
    _active(persona, "sess_b11m", rows, cursor_at=11)
    _arm(persona, "all")
    rewrite_session_atomic(persona, "sess_b11m", [r for r in rows if r is not rows[4]])
    _run_all(persona, store, hebbian, _Stub())
    assert eb.load_state(persona)["counters"]["missing"] == 1
    assert "missing=1" in eb.status_text(persona)


# ---------------------------------------------------------------------------
# B12: failures and deferrals
# ---------------------------------------------------------------------------


def test_b12_three_failures_mark_a_target_failed_and_the_rest_run(kestrel) -> None:
    persona, store, hebbian = kestrel
    bad = _rows("sess_b12a", _OLD, 4, tag="x")
    good = _rows("sess_b12b", _OLD + timedelta(days=1), 4, tag="y")
    _archive(persona, "sess_b12a", bad)
    _archive(persona, "sess_b12b", good)
    _arm(persona, "all")
    stub = _Stub(plan=lambda n, p: "fail" if "zvx" in p else None)
    steps = [_run(persona, store, hebbian, stub) for _ in range(3)]
    assert [s.advanced for s in steps] == [False, False, True]
    state = eb.load_state(persona)
    assert state["targets"][0]["status"] == "failed" and state["targets"][1]["status"] == "open"
    _run_all(persona, store, hebbian, stub)
    state = eb.load_state(persona)
    assert state["targets"][1]["status"] == "done" and state["status"] == "complete"


def test_b12_a_deferral_counts_nothing(kestrel) -> None:
    persona, store, hebbian = kestrel
    _archive(persona, "sess_b12d", _rows("sess_b12d", _OLD, 4))
    _arm(persona, "all")
    before = eb.state_path(persona).read_bytes()
    step = _run(persona, store, hebbian, _Stub(plan=lambda n, p: "defer"))
    assert step == eb.Step(False, "deferred")
    assert eb.state_path(persona).read_bytes() == before


# ---------------------------------------------------------------------------
# B13 [A]: gate load is observable
# ---------------------------------------------------------------------------


def test_b13_status_reports_counts_and_enqueued_matches_the_queue(kestrel) -> None:
    persona, store, hebbian = kestrel
    _archive(persona, "sess_b13", _rows("sess_b13", _OLD, 12))
    _arm(persona, "all")
    queue = persona / "pending_candidates.jsonl"
    before = len(queue.read_text().splitlines()) if queue.exists() else 0
    _run_all(persona, store, hebbian, _Stub())
    state = eb.load_state(persona)
    added = len(queue.read_text().splitlines()) - before
    assert state["counters"]["chunks"] > 1
    assert added == state["counters"]["enqueued"] > 0
    text = eb.status_text(persona)
    assert f"extracted={state['counters']['extracted']}" in text and f"enqueued={added}" in text


# ---------------------------------------------------------------------------
# B15: back-dated, and the silence clock never moves past the real newest user turn
# ---------------------------------------------------------------------------


def _b15_history(persona) -> list[dict]:
    """Archived only, so the time since the user spoke is read from memories; every reply is stamped after its user row,
    the last one by minutes."""
    rows = _rows("sess_b15", datetime.now(UTC) - timedelta(days=5), 9, gap_s=60, lag_s=20)
    last_user = lenient_utc(rows[-2]["ts"])
    rows[-1]["ts"] = (last_user + timedelta(minutes=4)).isoformat(timespec="seconds")
    _archive(persona, "sess_b15", rows)
    return rows


def _silence_read_at(store, now) -> datetime:
    from brain.utils.memory import days_since_human

    return now - timedelta(days=days_since_human(store, now))


def test_b15_each_memory_is_dated_at_its_chunks_last_user_turn(kestrel) -> None:
    persona, store, hebbian = kestrel
    rows = _b15_history(persona)
    _arm(persona, "all")
    stub = _Stub()
    created: dict[str, datetime] = {}
    real = eb.run_chunk_stages

    def _spy(persona_dir, sid, transcript, **kw):
        result = real(persona_dir, sid, transcript, **kw)
        users = [r for r in rows if r["speaker"] == "user" and _line(r) in transcript]
        for mid in kw["report"].memory_ids:
            created[mid] = lenient_utc(users[-1]["ts"])
        return result

    eb.run_chunk_stages = _spy
    try:
        _run_all(persona, store, hebbian, stub)
    finally:
        eb.run_chunk_stages = real
    run_consolidation(store, persona_dir=persona, classifier=_PROMOTE_ALL)
    assert created
    for mid, expect in created.items():
        assert store.get(mid).created_at == expect
    newest_user = lenient_utc(rows[-2]["ts"])
    now = datetime.now(UTC)
    assert _silence_read_at(store, now) <= newest_user + timedelta(seconds=1)


def test_b15_the_silence_clock_is_unchanged_when_a_newer_memory_exists(kestrel) -> None:
    from brain.utils.memory import days_since_human

    persona, store, hebbian = kestrel
    _b15_history(persona)
    recent = commit_item(ExtractedItem(text="Bob asked about the plimsy kettle", label="fact"), session_id="sess_new",
                         store=store, hebbian=hebbian, created_at=datetime.now(UTC) - timedelta(days=2))
    run_consolidation(store, persona_dir=persona, classifier=_PROMOTE_ALL)
    assert store.get(recent) is not None
    now = datetime.now(UTC)
    before = days_since_human(store, now)
    _arm(persona, "all")
    _run_all(persona, store, hebbian, _Stub())
    run_consolidation(store, persona_dir=persona, classifier=_PROMOTE_ALL)
    assert days_since_human(store, now) == before


def test_b15_a_chunk_with_no_user_row(kestrel) -> None:
    persona, _, _ = kestrel
    rows = _b15_history(persona)
    now = datetime.now(UTC)
    opener = ({"session_id": "s", "speaker": "assistant", "text": "hello", "ts": rows[-1]["ts"]},)
    assert eb._backdate(persona, opener, rows[-1]["ts"], now) == lenient_utc(rows[-2]["ts"])  # the newest user turn


@pytest.mark.parametrize("variant", ["chunk-end", "dated-now"])
def test_b15_oracle_catches_the_wrong_dates(kestrel, monkeypatch, variant) -> None:
    from brain.utils.memory import days_since_human

    persona, store, hebbian = kestrel
    rows = _b15_history(persona)
    newest_user = lenient_utc(rows[-2]["ts"])
    if variant == "chunk-end":
        monkeypatch.setattr(eb, "_backdate", lambda p, r, cursor_ts, now: min(lenient_utc(cursor_ts), now))
    else:
        monkeypatch.setattr(eb, "_backdate", lambda p, r, cursor_ts, now: None)
    _arm(persona, "all")
    _run_all(persona, store, hebbian, _Stub())
    run_consolidation(store, persona_dir=persona, classifier=_PROMOTE_ALL)
    now = datetime.now(UTC)
    if variant == "chunk-end":
        assert _silence_read_at(store, now) > newest_user + timedelta(seconds=1)
    else:
        assert days_since_human(store, now) < 0.01


def test_b15_the_pipeline_still_dates_at_creation(kestrel) -> None:
    from brain.ingest.pipeline import extract_session_snapshot

    persona, store, hebbian = kestrel
    _active(persona, "sess_b15p", _rows("sess_b15p", datetime.now(UTC) - timedelta(days=4), 2), cursor_at=None)
    before = datetime.now(UTC).replace(microsecond=0)
    report = extract_session_snapshot(persona, "sess_b15p", store=store, hebbian=hebbian, provider=_Stub())
    run_consolidation(store, persona_dir=persona, classifier=_PROMOTE_ALL)
    assert report.memory_ids and all(store.get(m).created_at >= before for m in report.memory_ids)


# ---------------------------------------------------------------------------
# B16: soul candidates by mode
# ---------------------------------------------------------------------------


def _soul_count(persona) -> int:
    path = persona / "soul_candidates.jsonl"
    return len(path.read_text().splitlines()) if path.exists() else 0


@pytest.mark.parametrize(("mode", "expected"), [("heads", 1), ("all", 0)])
def test_b16_soul_candidates_by_mode(kestrel, monkeypatch, mode, expected) -> None:
    persona, store, hebbian = kestrel
    monkeypatch.setattr(eb, "_HISTORICAL_CAP_CHARS", 300)
    _archive(persona, "sess_b16", _rows("sess_b16", _OLD, 2))
    _arm(persona, mode)
    reports = []
    real = eb.run_chunk_stages
    monkeypatch.setattr(eb, "run_chunk_stages", lambda *a, **k: (reports.append(k["report"]), real(*a, **k))[1])
    _run(persona, store, hebbian, _Stub(importance=9))
    assert _soul_count(persona) == expected
    assert reports[0].soul_candidates == expected


def test_b16_the_pipeline_still_queues_soul_candidates(kestrel) -> None:
    from brain.ingest.pipeline import extract_session_snapshot

    persona, store, hebbian = kestrel
    _active(persona, "sess_b16p", _rows("sess_b16p", _OLD, 2), cursor_at=None)
    extract_session_snapshot(persona, "sess_b16p", store=store, hebbian=hebbian, provider=_Stub(importance=9))
    assert _soul_count(persona) == 1


def test_b16_oracle_catches_soul_on_under_all(kestrel, monkeypatch) -> None:
    persona, store, hebbian = kestrel
    _archive(persona, "sess_b16v", _rows("sess_b16v", _OLD, 2))
    _arm(persona, "all")
    real = eb.run_chunk_stages
    monkeypatch.setattr(eb, "run_chunk_stages", lambda *a, **k: real(*a, **{**k, "queue_soul": True}))
    _run(persona, store, hebbian, _Stub(importance=9))
    assert _soul_count(persona) == 1


# ---------------------------------------------------------------------------
# B8 (bridge half): the supervisor's job table and one central pass
# ---------------------------------------------------------------------------


def _table(persona, stub, monkeypatch):
    from brain.bridge import supervisor

    monkeypatch.setattr(supervisor, "build_tier_provider", lambda persona_dir, tier: stub)

    class _Bus:
        def publish(self, event):
            pass

    return supervisor._build_gated_jobs(
        persona_dir=persona, provider=stub, event_bus=_Bus(), is_session_busy=None, finalize_after_hours=24.0,
        finalize_interval_s=None, initiate_review_interval_s=None, maintenance_interval_s=None, self_model_interval_s=None,
        compaction_interval_s=None, calibration_interval_s=None, interest_sweep_interval_s=None,
        judge_selftune_interval_s=None, clustering_interval_s=None, intensity_drivers=lambda: None,
        tick_stats={"pruned_empty_sessions": 0, "closed_sessions": 0},
    )


def _three_chunk_plan(persona) -> None:
    rows = _rows("sess_b8b", _OLD, 6)
    assert len(chunk_turns(rows, max_chars=_CHUNK, user_name=_USER, assistant_name="Kestrel")) == 3
    _archive(persona, "sess_b8b", rows)
    _arm(persona, "all")


def _pass_with_a_later_job(persona, stub, monkeypatch, **kw):
    from brain.bridge import central_cadence

    ran: list[str] = []
    jobs = _table(persona, stub, monkeypatch) + [
        central_cadence.GatedJob("maintenance", run=lambda: ran.append("maintenance"), has_work=lambda: True)
    ]
    decisions = central_cadence.run_central_pass(persona, jobs, **kw)
    return decisions, ran


def test_b8_the_job_sits_right_after_recall_diagnostics_flush(kestrel, monkeypatch) -> None:
    from brain.bridge.central_cadence import GATED_JOB_ORDER

    persona, _, _ = kestrel
    names = [j.name for j in _table(persona, _Stub(), monkeypatch)]
    assert "extraction_backfill" in names
    assert GATED_JOB_ORDER[GATED_JOB_ORDER.index("recall_diagnostics_flush") + 1] == "extraction_backfill"
    assert names[names.index("recall_diagnostics_flush") + 1] == "extraction_backfill"


def test_b8_one_pass_makes_one_call_and_the_later_jobs_still_run(kestrel, monkeypatch) -> None:
    persona, _, _ = kestrel
    _three_chunk_plan(persona)
    stub = _Stub()
    decisions, ran = _pass_with_a_later_job(persona, stub, monkeypatch, is_idle=lambda: True, slot_available=lambda: True)
    assert len(stub.prompts) == 1
    assert ran == ["maintenance"]
    assert ("extraction_backfill", "completed") in [(d.job, d.action) for d in decisions]


def test_b8_no_lull_means_no_call(kestrel, monkeypatch) -> None:
    persona, _, _ = kestrel
    _three_chunk_plan(persona)
    stub = _Stub()
    decisions, _ = _pass_with_a_later_job(persona, stub, monkeypatch, is_idle=lambda: False, slot_available=lambda: True)
    assert stub.prompts == []
    assert decisions[0].action == "skip-no-lull"


def test_b8_a_denied_slot_inside_run_is_skipped(kestrel, monkeypatch) -> None:
    from contextlib import contextmanager

    persona, _, _ = kestrel
    _three_chunk_plan(persona)

    @contextmanager
    def _denied():
        yield False

    monkeypatch.setattr(eb.cli_throttle, "background_slot", _denied)
    stub = _Stub()
    decisions, _ = _pass_with_a_later_job(persona, stub, monkeypatch, is_idle=lambda: True, slot_available=lambda: True)
    assert stub.prompts == []
    assert ("extraction_backfill", "skipped") in [(d.job, d.action) for d in decisions]


def test_b8_oracle_catches_a_job_that_loops(kestrel, monkeypatch) -> None:
    persona, _, _ = kestrel
    _three_chunk_plan(persona)
    real = eb.run_one_chunk

    def _loop(persona_dir, **kw):
        step = real(persona_dir, **kw)
        while eb.has_work(persona_dir):
            step = real(persona_dir, **kw)
        return step

    monkeypatch.setattr(eb, "run_one_chunk", _loop)
    stub = _Stub()
    _pass_with_a_later_job(persona, stub, monkeypatch, is_idle=lambda: True, slot_available=lambda: True)
    assert len(stub.prompts) == 3


# ---------------------------------------------------------------------------
# B17: the plan stops where chunked extraction began (main ruling M4)
# ---------------------------------------------------------------------------


def _b17_world(persona, monkeypatch, marker: datetime) -> tuple[list[dict], list[dict]]:
    monkeypatch.setattr(eb, "_HISTORICAL_CAP_CHARS", 2_500)
    before = _rows("sess_b17a", marker - timedelta(days=3), 20, gap_s=30, tag="p")
    after = _rows("sess_b17b", marker + timedelta(days=2), 20, gap_s=30, tag="q")
    assert len("\n".join(_line(r) for r in after)) > 2_500  # over the (patched) historical cap: a "lost head" by the model
    _archive(persona, "sess_b17a", before)
    _archive(persona, "sess_b17b", after)
    (persona / "extraction_chunked_since.json").write_text(json.dumps({"since": marker.isoformat()}), encoding="utf-8")
    return before, after


@pytest.mark.parametrize("mode", ["heads", "all"])
def test_b17_text_after_the_chunked_extraction_marker_is_not_planned(kestrel, monkeypatch, mode) -> None:
    persona, _, _ = kestrel
    marker = datetime.now(UTC) - timedelta(days=10)
    _b17_world(persona, monkeypatch, marker)
    state = _arm(persona, mode)
    sids = {t["sid"] for t in state["targets"]}
    assert "sess_b17b" not in sids
    assert "sess_b17a" in sids
    assert lenient_utc(state["cutoff_ts"]) == marker
    assert all(lenient_utc(t["to_ts"]) <= marker for t in state["targets"])
    assert "sess_b17b" not in eb.dry_run_text(persona)


def test_b17_without_a_marker_the_cutoff_is_the_plan_time(kestrel, monkeypatch) -> None:
    persona, _, _ = kestrel
    marker = datetime.now(UTC) - timedelta(days=10)
    _b17_world(persona, monkeypatch, marker)
    (persona / "extraction_chunked_since.json").unlink()
    state = _arm(persona, "heads")
    assert lenient_utc(state["cutoff_ts"]) > datetime.now(UTC) - timedelta(minutes=1)
    assert "sess_b17b" in {t["sid"] for t in state["targets"]}
