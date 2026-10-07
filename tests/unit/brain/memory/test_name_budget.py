"""Cost-series increment (name-recall fix S145, S155, S157, S160): the keyword name
query's continuation walk runs under the per-message time budget and learns its
own cost (criteria CS8a-CS8g, CS3a name half).

A "continuation" is the walk `MemoryStore._fts_rows_kept` makes past the first page
(50 rows) when it came back full and the name filter rejected some of it. The
fixture is a real file database in which 200 short rows hold the modal "will"
written in lower case (the stopword-name "Will" matches only when capitalized,
S151, so every one is rejected) and five long rows hold "Will": by bm25 the 200
come first, so the five hits lie past the first page. Time is scripted: each row
the filter judges advances the budget clock by a fixed cost per character
(dyadic, so every boundary is exact). Synthetic data only.
"""

from __future__ import annotations

import inspect
import sqlite3
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from brain.memory import cost_series, keyword_budget, relevance
from brain.memory import store as store_mod
from brain.memory.store import Memory, MemoryStore

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


def test_cs8a_without_an_estimate_the_pages_and_the_clock_stop_the_walk_when_left_is_reached(
    store: MemoryStore, clock: Clock, judged: Judged
) -> None:
    _, full = _walk(store, clock, judged, left=None)
    hits, texts = _walk(store, clock, judged, left=10.5 * ROW_COST)
    continuation = texts[FIRST_PAGE:]
    assert len(continuation) == 10, "S170: one page holds the 10 rows that fit; the 0.5 row left is less than a row"
    assert continuation == full[FIRST_PAGE : FIRST_PAGE + 10], "a prefix of the unbudgeted walk's rows"
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
# CS8b (S170): the continuation is read in pages of ONE narrow ordering
# statement; the deadline is checked between pages and a page's size is derived
# from the measured per-row cost. Time is scripted (dyadic numbers, so every
# boundary is exact): the first page's `execute` and the ordering statement
# advance the clock from a sqlite trace callback, each judged row from `keep`.
# ---------------------------------------------------------------------------

SQL = (
    "SELECT m.*, bm25(memories_fts) AS _bm25 FROM memories_fts f JOIN memories m ON m.rowid = f.rowid "
    "WHERE memories_fts MATCH ? AND m.active = 1 ORDER BY _bm25 ASC"
)
FAMILY_SQL = (
    "SELECT m.*, bm25(memories_fts) AS _bm25 FROM memories_fts f JOIN memories m ON m.rowid = f.rowid "
    "WHERE memories_fts MATCH ? AND m.active = 1 ORDER BY (m.memory_type IN (?, ?)) ASC, _bm25 ASC"
)
FIRST = 5  # the first page's LIMIT in these tests
PAGED_ROWS = 40


class _Budget:
    """A KeepBudget on a scripted clock (`clock`: None = a clock that never moves).
    With `clock_bound`, `allows` ends the walk once its elapsed time reaches `left`,
    as `NameWalkBudget.allows` does; `allows=` replaces that."""

    def __init__(self, left: float | None, *, clock: Clock | None = None, clock_bound: bool = False, allows=None) -> None:
        self._left = left
        self._clock = clock
        self._clock_bound = clock_bound
        self._allows = allows
        self.began = 0.0
        self.finished: list[int] = []

    def now(self) -> float:
        return 0.0 if self._clock is None else self._clock.t

    def begin(self):
        self.began = self.now()
        return self._left

    def allows(self, rows_judged, chars):
        if self._allows is not None:
            return self._allows(rows_judged, chars)
        return not (self._clock_bound and self.now() - self.began >= self._left)

    def finish(self, chars):
        self.finished.append(chars)


class _Keep:
    """The name filter: keeps "Will" rows, records every text it is asked about and
    advances the scripted clock by `cost(i)` for the i-th call (0-based)."""

    def __init__(self, clock: Clock, cost=lambda i: 0.0) -> None:
        self.clock = clock
        self.cost = cost
        self.texts: list[str] = []

    def __call__(self, text: str) -> bool:
        i = len(self.texts)
        self.texts.append(text)
        self.clock.t += self.cost(i)
        return text.startswith("Will")


def _make_paged_store(path: Path, *, family: bool = False, long_rows: bool = False) -> MemoryStore:
    s = MemoryStore(path, integrity_check=False)
    for i in range(PAGED_ROWS):  # equal lengths: equal bm25, so ties straddle every page boundary
        kind = "monologue" if family and i % 3 == 0 else "conversation"
        s.create(Memory.create_new(content=ROW.format(i=i), memory_type=kind, domain="us"))
    for j in range(3):
        filler = " ".join(f"padding{j}x{k}" for k in range(20 + 5 * j))
        if long_rows:
            filler *= 40
        s.create(
            Memory.create_new(
                content=f"Will brought the lantern number {j} and {filler}", memory_type="conversation", domain="us"
            )
        )
    return s


@pytest.fixture(autouse=True)
def _page_loop_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """A walk whose page loop never ends (a mutant, a regression) fails instead of hanging."""
    real = store_mod._page_rows  # noqa: SLF001
    asked = {"n": 0}

    def guarded(remaining: float, per_row: float) -> int:
        asked["n"] += 1
        assert asked["n"] <= 20_000, "the page loop did not end"
        return real(remaining, per_row)

    monkeypatch.setattr(store_mod, "_page_rows", guarded)


@pytest.fixture
def paged(tmp_path: Path) -> Iterator[MemoryStore]:
    s = _make_paged_store(tmp_path / "paged.db")
    yield s
    s.close()


def _trace(store: MemoryStore, clock: Clock, *, first: float = 0.0, ordering: float = 0.0) -> list[str]:
    """Log every statement the store's connection starts; the first page's statement
    advances the scripted clock by `first`, the ordering statement by `ordering`."""
    statements: list[str] = []

    def callback(sql: str) -> None:
        statements.append(sql)
        if sql.startswith("SELECT m.*, bm25") and "LIMIT" in sql:
            clock.t += first
        elif sql.startswith("SELECT m.id, "):
            clock.t += ordering

    store._conn.set_trace_callback(callback)  # noqa: SLF001
    return statements


def _walk_rows(store: MemoryStore, budget, keep, *, sql: str = SQL, params=None):
    return store._fts_rows_kept(sql, params or ['"will"'], FIRST, keep, budget)  # noqa: SLF001


def _ids(rows) -> list[tuple[str, float]]:
    return [(r["id"], r["_bm25"]) for r in rows]


def _records(rows) -> list[tuple[tuple[str, ...], tuple]]:
    """Every column of every row (the full record and the score), in order."""
    return [(tuple(r.keys()), tuple(r)) for r in rows]


@pytest.mark.parametrize(
    ("first_cost", "left"),
    [(1.0, 1.0), (0.5, 1.0), (1.0, 7.0), (2.0**-20, 1000.0)],
    ids=["page of 1", "page of 2", "page of 7 then 1", "the whole cursor"],
)
def test_cs8b1_paging_judges_every_row_once_in_the_unbudgeted_order_whatever_the_page_size(
    paged: MemoryStore, clock: Clock, first_cost: float, left: float
) -> None:
    clock.t = 0.0
    full_keep = _Keep(clock)
    expected = _walk_rows(paged, None, full_keep)
    clock.t = 0.0
    keep = _Keep(clock, cost=lambda i: first_cost if i < FIRST else 0.0)
    rows = _walk_rows(paged, _Budget(left, clock=clock), keep)
    assert len(full_keep.texts) == PAGED_ROWS + 3, "the unbudgeted walk judges every match"
    assert keep.texts == full_keep.texts, "each row once, in the unbudgeted order, the first page's ids not again"
    assert _ids(rows) == _ids(expected), "the same full records and scores, in the same order"
    assert _records(rows) == _records(expected) and rows, "every column of every kept row, not just the id"


def test_cs8b1_the_family_types_order_by_pages_the_same_way(tmp_path: Path, clock: Clock) -> None:
    s = _make_paged_store(tmp_path / "family.db", family=True)
    try:
        params = ['"will"', "monologue", "monologue_trace"]
        full_keep = _Keep(clock)
        expected = _walk_rows(s, None, full_keep, sql=FAMILY_SQL, params=params)
        keep = _Keep(clock, cost=lambda i: 1.0 if i < FIRST else 0.0)
        rows = _walk_rows(s, _Budget(5.0, clock=clock), keep, sql=FAMILY_SQL, params=params)
        assert keep.texts == full_keep.texts and _ids(rows) == _ids(expected)
    finally:
        s.close()


def test_cs8b1_an_ordering_that_differs_from_the_first_pages_loses_and_repeats_no_row(
    paged: MemoryStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The order equality of the narrow and the wide statement is an observation, not a
    guarantee: the walk skips by id, so even an ordering that disagrees with the first
    page's (here: reversed) judges every row exactly once."""
    reversed_sql = (
        "SELECT m.id, bm25(memories_fts) AS _bm25 FROM memories_fts f JOIN memories m ON m.rowid = f.rowid "
        "WHERE memories_fts MATCH ? AND m.active = 1 ORDER BY m.id DESC"
    )
    monkeypatch.setattr(store_mod, "_ordering_sql", lambda sql: reversed_sql)
    full_keep = _Keep(clock)
    expected = _walk_rows(paged, None, full_keep)
    keep = _Keep(clock, cost=lambda i: 1.0 if i < FIRST else 0.0)
    rows = _walk_rows(paged, _Budget(5.0, clock=clock), keep)
    assert len(keep.texts) == len(set(keep.texts)) == PAGED_ROWS + 3
    assert sorted(keep.texts) == sorted(full_keep.texts)
    assert {r["id"] for r in rows} == {r["id"] for r in expected}


@pytest.mark.parametrize(
    ("remaining", "per_row", "rows"),
    [(0.75, 0.125, 6), (0.125, 0.125, 1), (0.0625, 0.125, 0), (1.0, 0.0, 1), (0.0, 0.0, 0)],
)
def test_cs8b2_the_page_size_is_the_rows_that_fit_the_time_remaining(remaining: float, per_row: float, rows: int) -> None:
    assert store_mod._page_rows(remaining, per_row) == rows  # noqa: SLF001


def test_cs8b2_a_per_row_cost_too_small_to_divide_is_capped_never_raised() -> None:
    assert store_mod._page_rows(1.0, 5e-324) == sys.maxsize  # noqa: SLF001 - the quotient is infinite
    assert store_mod._page_rows(1.0, 1e-300) == sys.maxsize  # noqa: SLF001 - beyond an int


def test_cs8b2_when_no_row_is_affordable_after_the_first_pages_overhead_no_continuation_runs(
    paged: MemoryStore, clock: Clock
) -> None:
    statements = _trace(paged, clock, first=0.9375)
    keep = _Keep(clock, cost=lambda i: 0.125 if i < FIRST else 0.0)
    budget = _Budget(1.0, clock=clock)
    rows = _walk_rows(paged, budget, keep)
    # remaining 1.0 - overhead 0.9375 = 0.0625 is less than one row (0.125)
    assert rows == [] and len(keep.texts) == FIRST, "the first page's kept rows, nothing else judged"
    assert not [s for s in statements if s.startswith("SELECT m.id, ")], "no ordering statement"
    assert budget.finished == [0], "nothing judged past the first page: nothing recorded"


def test_cs8b2_the_first_page_of_rows_holds_what_fits_after_the_ordering_statement(
    paged: MemoryStore, clock: Clock
) -> None:
    statements = _trace(paged, clock, ordering=0.25)
    keep = _Keep(clock, cost=lambda i: 0.125)
    budget = _Budget(1.0, clock=clock, clock_bound=True)
    _walk_rows(paged, budget, keep)
    ordering = [s for s in statements if s.startswith("SELECT m.id, ")]
    assert len(ordering) == 1, "the ordering statement runs once"
    # remaining after it is 0.75 = six rows at 0.125: the first page of rows holds six
    assert len(keep.texts) == FIRST + 6
    assert budget.finished == [sum(len(t) for t in keep.texts[FIRST:])]


def test_cs8b2_the_ordering_statement_sorts_ids_and_scores_not_rows(paged: MemoryStore, clock: Clock) -> None:
    statements = _trace(paged, clock)
    _walk_rows(paged, _Budget(1000.0, clock=clock), _Keep(clock, cost=lambda i: 0.0))
    (ordering,) = [s for s in statements if s.startswith("SELECT m.id, ")]
    select_list = ordering.split(" FROM ")[0]
    assert "m.*" not in select_list and "content" not in select_list, select_list
    assert "ORDER BY _bm25 ASC" in ordering and "LIMIT" not in ordering


def test_cs8b2_a_per_row_cost_of_zero_gives_pages_of_one_row_and_still_finishes(
    paged: MemoryStore, clock: Clock
) -> None:
    keep = _Keep(clock)
    _walk_rows(paged, _Budget(1.0, clock=clock), keep)
    assert len(keep.texts) == PAGED_ROWS + 3


def _page_sizes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[float, float]]:
    calls: list[tuple[float, float]] = []
    real = store_mod._page_rows  # noqa: SLF001

    def recording(remaining: float, per_row: float) -> int:
        calls.append((remaining, per_row))
        return real(remaining, per_row)

    monkeypatch.setattr(store_mod, "_page_rows", recording)
    return calls


def test_cs8b3_a_page_is_sized_from_the_time_remaining_and_the_worst_per_row_figure_of_the_continuation(
    paged: MemoryStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _page_sizes(monkeypatch)

    def cost(i: int) -> float:
        if i < FIRST:
            return 0.125  # the first page: 0.125 s a row
        j = i - FIRST  # the j-th row judged past the first page
        if j < 16:  # page 1 judges sixteen rows: fifteen cheap and one slow
            return 0.5 if j == 15 else 0.0625
        return 0.03125 if j < 29 else 0.25  # page 2 judges thirteen cheaper rows; page 3 only slow ones

    left = 2.625  # page 1 holds 21 ids: the five first-page ids are skipped, sixteen rows are judged
    _walk_rows(paged, _Budget(left, clock=clock, clock_bound=True), _Keep(clock, cost=cost))
    assert calls[0] == (left, 0.125), "page 1: all the time left, the first page's per-row figure"
    page_one = (15 * 0.0625 + 0.5) / 16
    assert calls[1] == (left - 1.4375, page_one), "page 2: the time remaining, page 1's own figure"
    assert calls[2][1] == page_one, "page 3: a cheaper page 2 does not shrink the figure again (the worst is kept)"
    assert calls[2][0] == pytest.approx(left - 1.4375 - 13 * 0.03125)


def test_cs8b3_a_page_of_only_skipped_ids_does_not_change_the_per_row_figure(
    paged: MemoryStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _page_sizes(monkeypatch)
    # first page 1.0 s a row, left 3.0: page 1 holds three ids, all of them first-page ids (skipped)
    _walk_rows(paged, _Budget(3.0, clock=clock, clock_bound=True), _Keep(clock, cost=lambda i: 1.0 if i < FIRST else 0.0))
    assert calls[0] == (3.0, 1.0)
    assert calls[1] == (3.0, 1.0), "nothing was judged on page 1: the figure stays"


def test_cs8b3_the_walk_ends_when_the_cursor_is_exhausted_or_limit_rows_are_kept(
    paged: MemoryStore, clock: Clock
) -> None:
    keep = _Keep(clock)
    rows = paged._fts_rows_kept(SQL, ['"will"'], 2, keep, _Budget(1.0, clock=clock))  # noqa: SLF001
    # limit 2: the two shortest "Will" rows end the walk before the cursor is read to its end
    assert len(rows) == 2 and len(keep.texts) < PAGED_ROWS + 3


def test_cs8b4_no_interrupt_machinery_is_left_in_the_store() -> None:
    assert not hasattr(store_mod, "_Deadline")
    assert "threading" not in vars(store_mod)
    for obj in (
        store_mod.MemoryStore.search_fts_scored,
        store_mod.MemoryStore._fts_rows_kept,  # noqa: SLF001
        store_mod.KeepBudget,
    ):
        text = inspect.getsource(obj).lower()
        assert "interrupt" not in text and "deadline" not in text, obj


class _Cursor:
    def __init__(self, cursor, log: list[str]) -> None:
        self._cursor = cursor
        self._log = log

    def __iter__(self):
        return iter(self._cursor)

    def close(self) -> None:
        self._log.append("closed")
        self._cursor.close()


class _Conn:
    """A connection proxy that wraps the ordering cursor to see it closed."""

    def __init__(self, conn) -> None:
        self._conn = conn
        self.log: list[str] = []

    def execute(self, sql, *args):
        cursor = self._conn.execute(sql, *args)
        return _Cursor(cursor, self.log) if sql.startswith("SELECT m.id, ") else cursor

    def __getattr__(self, name):
        return getattr(self._conn, name)


def test_cs8b4_an_error_while_reading_a_page_propagates_closes_the_cursor_and_still_finishes(
    paged: MemoryStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    proxy = _Conn(paged._conn)  # noqa: SLF001
    monkeypatch.setattr(paged, "_conn", proxy)
    calls = {"n": 0}

    def locked(rows, chars):
        calls["n"] += 1
        if calls["n"] == 3:
            raise sqlite3.OperationalError("database is locked")
        return True

    budget = _Budget(60.0, clock=clock, allows=locked)
    keep = _Keep(clock, cost=lambda i: 1.0 if i < FIRST else 0.0)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        _walk_rows(paged, budget, keep)
    assert proxy.log == ["closed"], "the ordering cursor is closed on the way out"
    assert budget.finished == [sum(len(t) for t in keep.texts[FIRST:])], "the two rows judged are handed to finish"


def test_cs8b4_an_error_from_a_by_id_read_propagates_closes_the_cursor_and_still_finishes(
    paged: MemoryStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FailingRead(_Conn):
        reads = 0

        def execute(self, sql, *args):
            if "WHERE m.id =" in sql:
                type(self).reads += 1
                if type(self).reads == 3:
                    raise sqlite3.OperationalError("database is locked")
            return super().execute(sql, *args)

    proxy = _FailingRead(paged._conn)  # noqa: SLF001
    monkeypatch.setattr(paged, "_conn", proxy)
    budget = _Budget(60.0, clock=clock)
    keep = _Keep(clock, cost=lambda i: 1.0 if i < FIRST else 0.0)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        _walk_rows(paged, budget, keep)
    assert proxy.log == ["closed"], "the ordering cursor is closed on the way out"
    assert budget.finished == [sum(len(t) for t in keep.texts[FIRST:])], "the rows judged so far are handed to finish"
    assert len(keep.texts) == FIRST + 2


def test_cs8b3_a_short_page_ends_the_walk_and_the_page_loop_is_bounded(
    paged: MemoryStore, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cursor runs dry: the short page ends the walk. The loop guard turns a walk that
    would spin on an exhausted cursor (the mutant) into a failure instead of a hang."""
    real = store_mod._page_rows  # noqa: SLF001
    asked = {"n": 0}

    def guarded(remaining: float, per_row: float) -> int:
        asked["n"] += 1
        assert asked["n"] <= 3 * (PAGED_ROWS + 3), "the page loop did not end on an exhausted cursor"
        return real(remaining, per_row)

    monkeypatch.setattr(store_mod, "_page_rows", guarded)
    keep = _Keep(clock)
    _walk_rows(paged, _Budget(1.0, clock=clock), keep)
    assert len(keep.texts) == PAGED_ROWS + 3 and asked["n"] <= PAGED_ROWS + 3 + 1


def _bounded_walk(paged: MemoryStore, clock: Clock, *, left: float, first_cost: float, later_cost: float,
                  first: float = 0.0, ordering: float = 0.0):
    clock.t = 0.0
    statements = _trace(paged, clock, first=first, ordering=ordering)
    keep = _Keep(clock, cost=lambda i: first_cost if i < FIRST else later_cost)
    budget = _Budget(left, clock=clock, clock_bound=True)
    _walk_rows(paged, budget, keep)
    return statements, keep, clock.t - budget.began


def test_cs8b5_with_exact_figures_the_walk_fits_left(paged: MemoryStore, clock: Clock) -> None:
    _, keep, elapsed = _bounded_walk(paged, clock, left=2.0, first_cost=0.125, later_cost=0.125, ordering=0.25)
    assert elapsed <= 2.0 and len(keep.texts) == FIRST + 14, "0.25 + 14 rows x 0.125 = 2.0"


def test_cs8b5_a_per_row_figure_measured_four_times_too_low_still_ends_within_one_row_of_left(
    paged: MemoryStore, clock: Clock
) -> None:
    _, keep, elapsed = _bounded_walk(paged, clock, left=2.0, first_cost=0.03125, later_cost=0.125, ordering=0.25)
    assert elapsed <= 2.0 + 0.125, "the per-row clock ends the page the estimate over-sized"
    assert len(keep.texts) - FIRST < PAGED_ROWS + 3 - FIRST, "the walk did not reach the end of the cursor"


@pytest.mark.parametrize("ordering_over_first", [1.0, 1.25, 1.5, 2.0])
@pytest.mark.parametrize("left_over_first", [1.0, 1.25, 1.5, 2.0])
def test_cs8b5_the_residual_overshoot_is_the_ordering_statements_excess_over_the_first_pages_execute(
    paged: MemoryStore, clock: Clock, ordering_over_first: float, left_over_first: float
) -> None:
    overhead, row = 0.25, 0.0625
    left = left_over_first * overhead
    statements, _, elapsed = _bounded_walk(
        paged, clock, left=left, first_cost=row, later_cost=row, first=overhead, ordering=ordering_over_first * overhead
    )
    ran = any(s.startswith("SELECT m.id, ") for s in statements)
    assert ran == (left - overhead >= row), "the guard: run the ordering statement exactly when one row is affordable"
    if ran:
        excess = (ordering_over_first - 1.0) * overhead
        assert elapsed <= left + excess + row


def test_cs8b6_long_contents_leave_the_hits_of_an_ample_budget_equal_to_the_unbudgeted_walk(tmp_path: Path) -> None:
    s = _make_paged_store(tmp_path / "long.db", long_rows=True)
    try:
        sink = Clock()
        full = _walk_rows(s, None, _Keep(sink))
        real_clock = _Budget(60.0)
        real_clock.now = time.perf_counter  # type: ignore[method-assign]
        rows = _walk_rows(s, real_clock, _Keep(sink))
        assert _ids(rows) == _ids(full) and len(rows) == 3
        assert max(len(r["content"]) for r in rows) > 3000
    finally:
        s.close()


def test_the_name_walk_budget_reads_the_modules_clock(store: MemoryStore, clock: Clock) -> None:
    clock.t = 3.5
    assert keyword_budget.NameWalkBudget(store, started=0.0).now() == 3.5


def test_cs8a_a_refused_row_stops_the_walk_it_does_not_skip_to_a_shorter_one(store: MemoryStore) -> None:
    """Rows in a fixed order: one short reject and two long rows on the first page
    (limit 3), then three more long rows, then 199 short rejects. The budget
    refuses the third continuation row (too many characters in all); the walk
    STOPS there, though each short row after it would have been allowed on its own."""
    sql = (
        "SELECT m.*, 0.0 AS _bm25 FROM memories m WHERE m.active = 1 "
        "ORDER BY (CASE WHEN m.content LIKE 'i will see you at the harbour 0000' THEN 0 "
        "WHEN m.content LIKE 'Will%' THEN 1 ELSE 2 END), m.id"
    )
    judged: list[str] = []

    def keep(text: str) -> bool:
        judged.append(text)
        return "lantern number 0 " in text  # one long row is kept, the rest rejected

    long_row = len(
        next(r[0] for r in store._conn.execute("SELECT content FROM memories") if r[0].startswith("Will"))  # noqa: SLF001
    )
    budget = _Budget(60.0, allows=lambda rows, chars: chars <= 2 * long_row + 100)
    store._fts_rows_kept(sql, [], 3, keep, budget)  # noqa: SLF001
    assert len(judged) == 3 + 2, "the first page (3 rows) and two continuation rows, then the walk stops"
    assert budget.finished == [2 * long_row]


# ---------------------------------------------------------------------------
# CS8c: the first page always runs; left <= 0 runs no continuation.
# ---------------------------------------------------------------------------


class _Spy:
    """A connection proxy that logs every statement; everything else is forwarded."""

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


def test_cs8e_the_search_tools_semantic_mode_passes_the_clock_reading_too(
    tmp_path: Path, clock: Clock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool's default (semantic) mode merges the keyword search in through a
    second call site; it must pass `started` as well (the lexical call site is
    pinned above)."""
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

    monkeypatch.setattr(tool, "_semantic_top_k", lambda *a, **k: tool._SemanticTopK(grouped=[], guaranteed=[]))
    monkeypatch.setattr(tool, "rank_name_hits", spy)
    monkeypatch.setattr(tool, "rank_memories", advance)
    dispatch("search_memories", {"query": "Quill", "mode": "semantic", "limit": 5}, **ctx)
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
