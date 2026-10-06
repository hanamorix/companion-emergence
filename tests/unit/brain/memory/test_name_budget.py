"""Cost-series increment (name-recall fix S145, S155, S157, S160): the keyword name
query's continuation walk runs under the per-message time budget and learns its
own cost (criteria CS8a-CS8g, CS3a name half).

A "continuation" is the second query `MemoryStore._fts_rows_kept` makes when the
first page (50 rows) came back full and the name filter rejected some of it. The
fixture is a real file database in which 200 short rows hold the modal "will"
written in lower case (the stopword-name "Will" matches only when capitalized,
S151, so every one is rejected) and five long rows hold "Will": by bm25 the 200
come first, so the five hits lie past the first page. Time is scripted: each row
the filter judges advances the budget clock by a fixed cost per character
(dyadic, so every boundary is exact). Synthetic data only.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from brain.memory import cost_series, keyword_budget, relevance
from brain.memory.store import Memory, MemoryStore, _Deadline

B = keyword_budget.recall_budget_seconds()
BIG_ROWS = 40_000
REJECTS = 200
KEEPS = 5
ROW = "i will see you at the harbour {i:04d}"  # a fixed width, lower-case modal: rejected
ROW_CHARS = len(ROW.format(i=0))
PER_CHAR = 2.0**-10  # scripted seconds per character judged
ROW_COST = ROW_CHARS * PER_CHAR
FIRST_PAGE = relevance.CANDIDATE_POOL


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(keyword_budget, "_clock", c)
    return c


@pytest.fixture
def store(tmp_path: Path) -> Iterator[MemoryStore]:
    s = MemoryStore(tmp_path / "memories.db", integrity_check=False)
    for i in range(REJECTS):
        s.create(Memory.create_new(content=ROW.format(i=i), memory_type="conversation", domain="us"))
    for j in range(KEEPS):
        filler = " ".join(f"padding{j}x{k}" for k in range(60))
        s.create(
            Memory.create_new(
                content=f"Will brought the lantern number {j} and {filler}",
                memory_type="conversation",
                domain="us",
            )
        )
    yield s
    s.close()


class Judged:
    """The name filter's own calls, in order, with the scripted time each costs."""

    def __init__(self) -> None:
        self.texts: list[str] = []


@pytest.fixture
def judged(monkeypatch: pytest.MonkeyPatch, clock: Clock) -> Judged:
    log = Judged()
    real = relevance.name_hit_ok

    def spy(text, names, known=None):
        log.texts.append(text)
        clock.t += len(text) * PER_CHAR
        return real(text, names, known)

    monkeypatch.setattr(relevance, "name_hit_ok", spy)
    return log


def _walk(store: MemoryStore, clock: Clock, judged: Judged, *, left: float | None):
    """`rank_name_hits` for "will"; with `left`, the budget leaves exactly that
    many seconds when the continuation starts (the first page's judging time is
    spent before it, scripted, and counts toward the message's budget)."""
    clock.t = 0.0
    judged.texts.clear()
    started = None
    if left is not None:
        began = FIRST_PAGE * ROW_COST  # the clock when the continuation begins
        started = began - (keyword_budget.recall_budget_seconds() - left)
    hits = relevance.rank_name_hits(store, None, ["will"], limit=relevance.CANDIDATE_POOL, started=started)
    return [m.content for m, _ in hits], list(judged.texts)


def _consume_discards() -> None:
    """Spend the process's first-sample discards through the production path."""
    keyword_budget.record_name_walk(MemoryStore(":memory:"), 1, 0.0)


def _fit(store: MemoryStore, overhead: float = 0.0, rate: float = PER_CHAR) -> None:
    """Two sizes on one exact line: the estimate is (overhead, rate)."""
    _consume_discards()
    for rows in (10, 20):
        x = rows * ROW_CHARS
        keyword_budget.record_name_walk(store, x, overhead + rate * x)


def _name_samples() -> list[cost_series._Sample]:  # noqa: SLF001
    with cost_series._pending_lock:  # noqa: SLF001
        return [s for s in cost_series._pending if s.series == "name matching"]  # noqa: SLF001


# ---------------------------------------------------------------------------
# CS8a: the continuation is budgeted; judged rows are a prefix.
# ---------------------------------------------------------------------------


def test_fixture_the_first_page_is_all_rejects_and_the_hits_lie_past_it(
    store: MemoryStore, clock: Clock, judged: Judged
) -> None:
    hits, texts = _walk(store, clock, judged, left=None)
    assert len(hits) == KEEPS and all(h.startswith("Will") for h in hits)
    assert len(texts) == FIRST_PAGE + (REJECTS - FIRST_PAGE) + KEEPS, "every row judged, unbudgeted"
    assert all(t.startswith("i will") for t in texts[:FIRST_PAGE])


def test_cs8a_without_an_estimate_the_clock_alone_stops_the_walk_when_left_is_reached(
    store: MemoryStore, clock: Clock, judged: Judged
) -> None:
    _, full = _walk(store, clock, judged, left=None)
    hits, texts = _walk(store, clock, judged, left=10.5 * ROW_COST)
    continuation = texts[FIRST_PAGE:]
    assert len(continuation) == 11, "10 rows leave 0.5 row: the 11th is judged, then the elapsed time has reached left"
    assert continuation == full[FIRST_PAGE : FIRST_PAGE + 11], "a prefix of the unbudgeted walk's rows"
    assert hits == [], "the rejected prefix holds no hit"


def test_cs8a_with_an_estimate_the_prediction_stops_the_walk_from_the_second_row_on(
    store: MemoryStore, clock: Clock, judged: Judged
) -> None:
    _fit(store)
    _, full = _walk(store, clock, judged, left=None)
    _, texts = _walk(store, clock, judged, left=10.5 * ROW_COST)
    continuation = texts[FIRST_PAGE:]
    # predict(k rows) = k x ROW_COST <= 10.5 x ROW_COST for k <= 10; the 11th would be predicted over left
    assert len(continuation) == 10
    assert continuation == full[FIRST_PAGE : FIRST_PAGE + 10]


@pytest.mark.parametrize("rows", [1, 2, 3, 40])
def test_cs8a_the_prediction_is_exact_at_the_boundary(
    store: MemoryStore, clock: Clock, judged: Judged, rows: int
) -> None:
    """`<=`: exactly `rows` rows predicted at exactly left fit; one more does not."""
    _fit(store)
    _, texts = _walk(store, clock, judged, left=rows * ROW_COST)
    assert len(texts[FIRST_PAGE:]) == max(rows, 1), "the first row is judged whenever the clock allows"


def test_cs8a_the_prediction_counts_the_next_rows_characters_and_the_overhead(
    store: MemoryStore, clock: Clock, judged: Judged
) -> None:
    _fit(store, overhead=3 * ROW_COST)
    _, texts = _walk(store, clock, judged, left=10 * ROW_COST)
    # overhead 3 rows' worth + k rows must be <= 10 rows' worth: k <= 7
    assert len(texts[FIRST_PAGE:]) == 7


def test_cs8a_kept_rows_are_a_prefix_of_the_unbudgeted_kept_rows(
    store: MemoryStore, clock: Clock, judged: Judged
) -> None:
    full_hits, full_texts = _walk(store, clock, judged, left=None)
    enough = (REJECTS - FIRST_PAGE + 3) * ROW_COST + 2 * 600 * PER_CHAR  # reaches into the keeps
    hits, texts = _walk(store, clock, judged, left=enough)
    assert 0 < len(hits) < KEEPS
    assert set(hits) < set(full_hits)
    # in the walk's own (query) order the kept rows are a prefix of the unbudgeted walk's
    kept = [t for t in texts if t.startswith("Will")]
    full_kept = [t for t in full_texts if t.startswith("Will")]
    assert kept == full_kept[: len(kept)] and sorted(kept) == sorted(hits)
    assert len(texts) < FIRST_PAGE + (REJECTS - FIRST_PAGE) + KEEPS


# ---------------------------------------------------------------------------
# CS8b: the continuation query itself is clock-bounded.
# ---------------------------------------------------------------------------


class _Budget:
    """A KeepBudget stub: `left` seconds, always allows, records what it is told."""

    def __init__(self, left: float | None, *, allows=lambda rows, chars: True) -> None:
        self._left = left
        self._allows = allows
        self.finished: list[int] = []

    def begin(self):
        return self._left

    def allows(self, rows_judged, chars):
        return self._allows(rows_judged, chars)

    def finish(self, chars):
        self.finished.append(chars)


@pytest.fixture(scope="module")
def big_store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[MemoryStore]:
    s = MemoryStore(tmp_path_factory.mktemp("big") / "big.db", integrity_check=False)
    s._conn.executemany(  # noqa: SLF001
        "INSERT INTO memories (id, content, memory_type, domain, emotions_json, tags_json, created_at) "
        "VALUES (?, ?, 'conversation', 'us', '{}', '[]', '2026-01-01T00:00:00+00:00')",
        [(f"id{i:06d}", f"i will {i} " + "x" * 40) for i in range(BIG_ROWS)],
    )
    s._conn.commit()  # noqa: SLF001
    yield s
    s.close()


def _kept(store: MemoryStore, budget) -> list:
    sql = (
        "SELECT m.*, bm25(memories_fts) AS _bm25 FROM memories_fts f JOIN memories m ON m.rowid = f.rowid "
        "WHERE memories_fts MATCH ? AND m.active = 1 ORDER BY _bm25 ASC"
    )
    return store._fts_rows_kept(sql, ['"will"'], 10, lambda text: False, budget)  # noqa: SLF001


def test_cs8b_a_deadline_that_expires_during_the_continuation_query_interrupts_it(
    big_store: MemoryStore,
) -> None:
    first_page_kept = []  # the filter rejects everything: nothing to keep from the first page
    budget = _Budget(0.0)
    assert _kept(big_store, budget) == first_page_kept, "returns the first page's kept rows"
    assert budget.finished == [0], "interrupted with no row judged: nothing recorded"
    # the connection is healthy afterwards
    assert big_store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == BIG_ROWS  # noqa: SLF001


def test_cs8b_an_interrupt_the_deadline_did_not_fire_still_propagates(big_store: MemoryStore) -> None:
    def locked(rows, chars):
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError, match="locked"):
        _kept(big_store, _Budget(60.0, allows=locked))


def test_cs8b_a_timer_callback_run_after_disarm_interrupts_nothing(big_store: MemoryStore) -> None:
    deadline = _Deadline(big_store._conn, 3600.0)  # noqa: SLF001
    deadline.arm()
    deadline.disarm()
    deadline._fire()  # noqa: SLF001 - the timer thread running late, after the cursor closed
    assert deadline.fired is False
    assert big_store._conn.execute("SELECT 1").fetchone()[0] == 1  # noqa: SLF001
    # and a full read still completes on the same connection
    assert len(big_store._conn.execute("SELECT id FROM memories").fetchall()) == BIG_ROWS  # noqa: SLF001


def test_cs8b_the_deadline_interrupt_fires_only_while_armed_and_is_recorded(big_store: MemoryStore) -> None:
    deadline = _Deadline(big_store._conn, 0.0)  # noqa: SLF001
    deadline.arm()
    with pytest.raises(sqlite3.OperationalError):
        for _ in range(50):  # long enough for the 0 s timer to fire inside a statement
            big_store._conn.execute("SELECT COUNT(*), SUM(length(content)) FROM memories").fetchall()  # noqa: SLF001
    assert deadline.fired is True
    deadline.disarm()
    assert big_store._conn.execute("SELECT 1").fetchone()[0] == 1  # noqa: SLF001


# ---------------------------------------------------------------------------
# CS8c: the first page always runs; left <= 0 runs no continuation.
# ---------------------------------------------------------------------------


class _Spy:
    """A connection proxy that logs every statement; `interrupt` etc. forwarded."""

    def __init__(self, conn) -> None:
        self._conn = conn
        self.sql: list[str] = []

    def execute(self, sql, *args):
        self.sql.append(sql)
        return self._conn.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_cs8c_with_no_time_left_the_first_pages_kept_rows_come_back_and_no_continuation_runs(
    store: MemoryStore, clock: Clock, judged: Judged, monkeypatch: pytest.MonkeyPatch
) -> None:
    spy = _Spy(store._conn)  # noqa: SLF001
    monkeypatch.setattr(store, "_conn", spy)
    for left in (0.0, -1.0):
        spy.sql.clear()
        hits, texts = _walk(store, clock, judged, left=left)
        assert hits == [], "the first page held no hit"
        assert len(texts) == FIRST_PAGE, "the first page is judged whole, whatever the budget"
        assert not [s for s in spy.sql if "memories_fts" in s and "LIMIT" not in s], "no continuation query"
        assert [s for s in spy.sql if "memories_fts" in s and "LIMIT" in s], "the first page ran"


def test_cs8c_the_first_pages_kept_rows_equal_the_result_when_the_budget_is_spent(
    tmp_path: Path, clock: Clock, judged: Judged
) -> None:
    s = MemoryStore(tmp_path / "kept.db", integrity_check=False)
    try:
        for i in range(10):  # kept rows ON the first page (short, so they rank first), plus rejects after
            s.create(Memory.create_new(content=f"Will {i}", memory_type="conversation", domain="us"))
        for i in range(120):
            s.create(Memory.create_new(content=ROW.format(i=i), memory_type="conversation", domain="us"))
        full, _ = _walk(s, clock, judged, left=None)
        none_left, _ = _walk(s, clock, judged, left=0.0)
        assert none_left == full[: len(none_left)] and len(none_left) == 10
    finally:
        s.close()


# ---------------------------------------------------------------------------
# CS8d: one sample per walk that judged a row; the process's first is discarded.
# ---------------------------------------------------------------------------


def test_cs8d_a_walk_records_one_sample_x_characters_judged_y_its_seconds(
    store: MemoryStore, clock: Clock, judged: Judged
) -> None:
    _consume_discards()
    _, texts = _walk(store, clock, judged, left=10.5 * ROW_COST)
    continuation = texts[FIRST_PAGE:]
    (sample,) = _name_samples()
    assert sample.x == sum(len(t) for t in continuation)
    assert sample.seconds == pytest.approx(sum(len(t) for t in continuation) * PER_CHAR)


def test_cs8d_the_first_walk_of_a_process_is_discarded_the_second_recorded(
    store: MemoryStore, clock: Clock, judged: Judged
) -> None:
    _walk(store, clock, judged, left=10.5 * ROW_COST)
    assert _name_samples() == [], "S160: the process's first name-walk sample is discarded"
    _walk(store, clock, judged, left=10.5 * ROW_COST)
    assert len(_name_samples()) == 1


def test_cs8d_a_walk_that_judged_no_row_records_nothing(
    store: MemoryStore, clock: Clock, judged: Judged
) -> None:
    _consume_discards()
    _walk(store, clock, judged, left=0.0)  # no continuation at all
    assert _name_samples() == []
    # a walk the clock stops before its first row (left already spent when it starts judging)
    budget = keyword_budget.NameWalkBudget(store, started=-keyword_budget.recall_budget_seconds() * 2)
    assert budget.begin() is None
    assert _name_samples() == []


# ---------------------------------------------------------------------------
# CS8e: wiring: `started` comes from the top of each entry point.
# ---------------------------------------------------------------------------


def test_cs8e_passive_recall_passes_the_clock_reading_from_the_top_of_the_block(
    tmp_path: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from brain.chat import prompt as prompt_mod
    from brain.memory import known_names as kn

    persona = tmp_path / "persona"
    persona.mkdir()
    s = MemoryStore(tmp_path / "w.db", integrity_check=False)
    try:
        s.create(Memory.create_new(content="Will came by the harbour", memory_type="conversation", domain="us"))
        assert kn.admit_names(persona, ["Quill"], "tool")
        s.create(Memory.create_new(content="Quill came by the harbour", memory_type="conversation", domain="us"))
        clock.t = 7.0
        seen: list[float | None] = []
        real = prompt_mod.rank_name_hits

        def spy(store, hebbian, names, **kw):
            seen.append(kw.get("started"))
            clock.t = 99.0  # time has moved on since the top of the block
            return real(store, hebbian, names, **kw)

        monkeypatch.setattr(prompt_mod, "rank_name_hits", spy)
        prompt_mod._build_recall_block(s, "Quill is here", persona_dir=persona, limit=8)
        assert seen == [7.0]
    finally:
        s.close()


def test_cs8e_the_search_tool_passes_the_clock_reading_from_the_top_of_its_call(
    tmp_path: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    import brain.tools.impls.search_memories as tool
    from brain.memory import known_names as kn
    from brain.tools.dispatch import dispatch
    from tests.unit.brain.tools.test_search_memories_keyword_merge import _ctx
    from tests.unit.brain.tools.test_search_memories_keyword_merge import _mem as tool_mem

    ctx = _ctx(tmp_path)
    tool_mem(ctx["store"], "Quill came by the harbour")
    assert kn.admit_names(ctx["persona_dir"], ["Quill"], "tool")
    clock.t = 3.0
    seen: list[float | None] = []
    real = tool.rank_name_hits
    real_rank = tool.rank_memories

    def advance(*a, **k):
        clock.t = 50.0
        return real_rank(*a, **k)

    def spy(store, hebbian, names, **kw):
        seen.append(kw.get("started"))
        return real(store, hebbian, names, **kw)

    monkeypatch.setattr(tool, "rank_name_hits", spy)
    monkeypatch.setattr(tool, "rank_memories", advance)
    dispatch("search_memories", {"query": "Quill", "mode": "lexical", "limit": 5}, **ctx)
    assert seen == [3.0]


def test_cs8e_without_started_the_walk_is_unbudgeted(
    store: MemoryStore, clock: Clock, judged: Judged, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs_seen: list[dict] = []
    real = MemoryStore.search_fts_scored

    def spy(self, query, **kw):
        kwargs_seen.append(kw)
        return real(self, query, **kw)

    monkeypatch.setattr(MemoryStore, "search_fts_scored", spy)
    _walk(store, clock, judged, left=None)
    assert kwargs_seen and "keep_budget" not in kwargs_seen[0]
    _walk(store, clock, judged, left=10 * ROW_COST)
    assert kwargs_seen[-1]["keep_budget"] is not None


# ---------------------------------------------------------------------------
# CS8f: ordinary names take no continuation and record nothing.
# ---------------------------------------------------------------------------


def test_cs8f_an_ordinary_name_runs_no_continuation_and_records_nothing(
    tmp_path: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    s = MemoryStore(tmp_path / "ordinary.db", integrity_check=False)
    try:
        for i in range(120):
            s.create(Memory.create_new(content=f"Quill walked the pier {i:03d}", memory_type="conversation", domain="us"))
        begun: list[int] = []
        real_begin = keyword_budget.NameWalkBudget.begin
        monkeypatch.setattr(
            keyword_budget.NameWalkBudget, "begin", lambda self: begun.append(1) or real_begin(self)
        )
        _consume_discards()
        with_budget = relevance.rank_name_hits(s, None, ["quill"], limit=50, started=0.0)
        without = relevance.rank_name_hits(s, None, ["quill"], limit=50)
        assert [m.id for m, _ in with_budget] == [m.id for m, _ in without]
        assert begun == [], "no rejected row on the first page: the budget is never consulted"
        assert _name_samples() == []
    finally:
        s.close()


# ---------------------------------------------------------------------------
# CS8g: an over-high estimate still judges the first row and records it.
# ---------------------------------------------------------------------------


def test_cs8g_an_estimate_predicting_even_the_first_row_over_left_still_judges_it_and_records(
    store: MemoryStore, clock: Clock, judged: Judged
) -> None:
    _fit(store, overhead=100.0)  # an absurdly high saved estimate
    _, texts = _walk(store, clock, judged, left=10.5 * ROW_COST)
    assert len(texts[FIRST_PAGE:]) == 1, "the first row is judged (the clock allows); the prediction stops the second"
    sample = _name_samples()[-1]
    assert sample.x == ROW_CHARS and sample.seconds == pytest.approx(ROW_COST)


# ---------------------------------------------------------------------------
# NameWalkBudget itself.
# ---------------------------------------------------------------------------


def test_the_budget_reads_the_estimate_before_it_starts_its_clock(
    store: MemoryStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    real = keyword_budget._name_cost.estimate  # noqa: SLF001

    def estimate(s, key):
        order.append("read")
        clock.t += B / 8  # a slow read: its time must not count as the walk's
        return real(s, key)

    monkeypatch.setattr(keyword_budget._name_cost, "estimate", estimate)  # noqa: SLF001
    budget = keyword_budget.NameWalkBudget(store, started=0.0)
    left = budget.begin()
    assert order == ["read"]
    assert left == pytest.approx(B - B / 8)
    assert budget.allows(0, 1) is True, "the walk's own elapsed time starts after the read"
