"""Tests for brain.memory.embedding_backfill — the idle-chipped backfill
(F1 #259 increment 3: row-based backlog, runtime-derived batch size,
skip-and-log cursor-freeze fix).

Every test uses on-disk sqlite files (not ":memory:") for memories.db, since
several tests simulate a "kill mid-backfill" by discarding one MemoryStore
and opening a fresh one against the same file — an in-memory db would not
survive that.

Idle-gating (busy tick skipped / idle tick runs) is NOT exercised here —
`run_embedding_backfill_tick` itself does not gate on cli_throttle; the
supervisor call site does that (see
tests/unit/brain/bridge/test_supervisor.py's idle-gate tests).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.memory import embedding_backfill
from brain.memory.embedding_backfill import (
    DEFAULT_SCAN_CAP,
    MIN_CHARS_TO_EMBED,
    _load_cursor,
    delete_legacy_embeddings_db,
    run_embedding_backfill_tick,
    run_embedding_backfill_to_completion,
)
from brain.memory.embeddings import EmbeddingProvider
from brain.memory.store import Memory, MemoryStore
from brain.paths import cadence_state_path


def _mem(content: str, *, created_at: datetime) -> Memory:
    m = Memory.create_new(content=content, memory_type="conversation", domain="us")
    m.created_at = created_at
    return m


def _open_store(persona_dir: Path) -> MemoryStore:
    return MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)


def _seed(persona_dir: Path, n: int, *, prefix: str = "memory content long enough") -> list[Memory]:
    store = _open_store(persona_dir)
    base = datetime(2020, 1, 1, tzinfo=UTC)
    try:
        made = []
        for i in range(n):
            m = _mem(f"{prefix} number {i}", created_at=base + timedelta(days=i))
            store.create(m)
            made.append(m)
        return made
    finally:
        store.close()


def _row_embedding(store: MemoryStore, memory_id: str) -> bytes | None:
    row = store._conn.execute(
        "SELECT embedding FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()
    return row["embedding"] if row is not None else None


def _is_embedded(store: MemoryStore, memory_id: str) -> bool:
    return _row_embedding(store, memory_id) is not None


# ---------------------------------------------------------------------------
# Basic behaviour — backlog is `embedding IS NULL` on the row itself
# ---------------------------------------------------------------------------


def test_tick_embeds_missing_rows(tmp_path: Path) -> None:
    """A fresh backlog of un-embedded rows gets embedded onto the ROW
    (embedding IS NOT NULL after), up to batch_size."""
    made = _seed(tmp_path, 3)
    store = _open_store(tmp_path)
    try:
        result = run_embedding_backfill_tick(tmp_path, store, batch_size=10, scan_cap=10)
    finally:
        store.close()

    assert result.embedded == 3
    assert result.scanned == 3
    assert result.errors == 0

    store2 = _open_store(tmp_path)
    try:
        for m in made:
            assert _is_embedded(store2, m.id) is True
    finally:
        store2.close()


def test_tick_is_a_near_no_op_once_caught_up(tmp_path: Path) -> None:
    """A second tick with nothing new to embed does no real compute work —
    the backlog query itself (`embedding IS NULL`) excludes rows already
    embedded, so `scanned` drops to 0."""
    _seed(tmp_path, 3)
    store = _open_store(tmp_path)
    try:
        run_embedding_backfill_tick(tmp_path, store, batch_size=10, scan_cap=10)
        result = run_embedding_backfill_tick(tmp_path, store, batch_size=10, scan_cap=10)
    finally:
        store.close()

    assert result.embedded == 0
    assert result.scanned == 0
    assert result.errors == 0


def test_skips_rows_under_min_chars(tmp_path: Path) -> None:
    """Very short content is never embedded — noise-vector guard. As of the
    FIX B (SQL-level short-row exclusion), the short row is excluded by
    `list_unembedded_since`'s own query — it never becomes a `candidate`,
    so it is not `scanned` and does not count as `skipped_short` either
    (that Python-side counter is now only a defense-in-depth backstop that
    should never actually trigger against this method's own query)."""
    store = _open_store(tmp_path)
    short = "x" * (MIN_CHARS_TO_EMBED - 1)
    long_enough = "y" * MIN_CHARS_TO_EMBED
    short_m = _mem(short, created_at=datetime(2020, 1, 1, tzinfo=UTC))
    long_m = _mem(long_enough, created_at=datetime(2020, 1, 2, tzinfo=UTC))
    store.create(short_m)
    store.create(long_m)
    store.close()

    store = _open_store(tmp_path)
    result = run_embedding_backfill_tick(tmp_path, store, batch_size=10, scan_cap=10)
    assert _is_embedded(store, short_m.id) is False
    assert _is_embedded(store, long_m.id) is True
    store.close()

    assert result.scanned == 1  # the short row was excluded at the SQL level
    assert result.skipped_short == 0
    assert result.embedded == 1


def test_steady_state_short_row_backlog_query_is_a_cheap_no_op(tmp_path: Path) -> None:
    """FIX B (F1 #259 increment-3 red-team, quiescence): once every
    EMBEDDABLE row is embedded, a permanently-short row must never
    resurface in the backlog query — `list_unembedded_since` excludes it in
    SQL — so the tick's cursor-reset-on-drain (see the module docstring's
    "Resumable/idempotent" section) is a genuinely cheap no-op: it re-scans
    nothing, not even the short row, on every subsequent tick."""
    store = _open_store(tmp_path)
    short = "x" * (MIN_CHARS_TO_EMBED - 1)
    long_enough = "y" * MIN_CHARS_TO_EMBED
    short_m = _mem(short, created_at=datetime(2020, 1, 1, tzinfo=UTC))
    long_m = _mem(long_enough, created_at=datetime(2020, 1, 2, tzinfo=UTC))
    store.create(short_m)
    store.create(long_m)

    # First tick: embeds the long row, the short row is never a candidate.
    result1 = run_embedding_backfill_tick(tmp_path, store, batch_size=10, scan_cap=10)
    assert result1.embedded == 1
    assert result1.scanned == 1

    # Steady state: the short row alone must not make later ticks re-scan.
    result2 = run_embedding_backfill_tick(tmp_path, store, batch_size=10, scan_cap=10)
    store.close()
    assert result2.scanned == 0
    assert result2.embedded == 0
    assert result2.skipped_short == 0


def test_already_embedded_rows_are_not_in_the_backlog(tmp_path: Path) -> None:
    """A row already embedded (e.g. by embed-on-write at promotion) simply
    never appears in the backlog query — no separate "already cached"
    bookkeeping needed the way the old content-hash cache required."""
    made = _seed(tmp_path, 5)

    store = _open_store(tmp_path)
    for m in made[:2]:
        store.embed_row(m.id, m.content)

    result = run_embedding_backfill_tick(tmp_path, store, batch_size=10, scan_cap=10)
    store.close()

    # Only the 3 genuinely-unembedded rows are even scanned.
    assert result.scanned == 3
    assert result.embedded == 3


# ---------------------------------------------------------------------------
# Bounded per tick
# ---------------------------------------------------------------------------


def test_batch_size_bounds_embeds_per_tick(tmp_path: Path) -> None:
    """A backlog larger than batch_size is only chipped by batch_size per tick."""
    _seed(tmp_path, 10)
    store = _open_store(tmp_path)
    result = run_embedding_backfill_tick(tmp_path, store, batch_size=4, scan_cap=100)
    assert result.embedded == 4
    store.close()

    store2 = _open_store(tmp_path)
    embedded_count = sum(1 for m in store2.list_active() if _is_embedded(store2, m.id))
    store2.close()
    assert embedded_count == 4


def test_backlog_shrinks_across_repeated_ticks(tmp_path: Path) -> None:
    """Repeated bounded ticks eventually clear a backlog larger than one batch."""
    _seed(tmp_path, 10)

    total_embedded = 0
    for _ in range(5):  # 5 ticks * batch_size=3 >= 10 rows
        store = _open_store(tmp_path)
        result = run_embedding_backfill_tick(tmp_path, store, batch_size=3, scan_cap=100)
        store.close()
        total_embedded += result.embedded

    assert total_embedded == 10
    store = _open_store(tmp_path)
    embedded_count = sum(1 for m in store.list_active() if _is_embedded(store, m.id))
    store.close()
    assert embedded_count == 10


def test_cursor_advances_not_resets_when_batch_cap_stops_a_tick_short_of_its_window(
    tmp_path: Path,
) -> None:
    """F1 #259 increment 7 perf fix: when `batch_size < remaining_backlog <
    scan_cap`, a tick that hits its OWN batch-size cap (stops before
    processing every row the SQL query fetched) must persist a forward
    cursor, not reset to None — resetting here (keying only off
    `len(candidates) < scan_cap`, the pre-fix behavior) forced every
    subsequent tick under a no-sleep drain to re-scan the full backlog from
    the top despite real progress being made. Once the remaining backlog
    genuinely fits inside a tick's batch (no cap hit, window smaller than
    scan_cap), the cursor still resets to None — the "catch rows that went
    NULL behind the cursor" self-healing property is preserved for that
    case."""
    from brain.memory import embeddings as embeddings_mod

    made = _seed(tmp_path, 10)
    model_id = embeddings_mod.build_embedding_provider().model_id()

    # Tick 1: batch_size(3) < backlog(10) < scan_cap(100) — hits the batch
    # cap partway through the fetched window (scans 1 row past the 3rd
    # embed to discover the cap, per the tick's own accounting).
    store = _open_store(tmp_path)
    try:
        result1 = run_embedding_backfill_tick(tmp_path, store, batch_size=3, scan_cap=100)
    finally:
        store.close()
    assert result1.embedded == 3
    assert result1.scanned == 4

    # The cursor must have ADVANCED (pinned at the 3rd/last-embedded row),
    # not reset to None, even though len(candidates)=10 < scan_cap=100.
    persisted = _load_cursor(tmp_path, model_id)
    assert persisted is not None
    assert persisted[1] == made[2].id

    # Tick 2 continues from the persisted cursor instead of re-scanning from
    # the top — it must not re-see the 3 already-embedded rows.
    store2 = _open_store(tmp_path)
    try:
        result2 = run_embedding_backfill_tick(tmp_path, store2, batch_size=3, scan_cap=100)
    finally:
        store2.close()
    assert result2.embedded == 3
    assert result2.scanned == 4  # rows 3,4,5 embedded; row 6 discovers the cap

    store3 = _open_store(tmp_path)
    try:
        for m in made[:6]:
            assert _is_embedded(store3, m.id) is True
        for m in made[6:]:
            assert _is_embedded(store3, m.id) is False
    finally:
        store3.close()
    # Still mid-backlog — cursor stays pinned forward, not reset.
    assert _load_cursor(tmp_path, model_id) is not None

    # Tick 3: 4 rows remain, batch_size=3 — hits the cap again.
    store4 = _open_store(tmp_path)
    try:
        result3 = run_embedding_backfill_tick(tmp_path, store4, batch_size=3, scan_cap=100)
    finally:
        store4.close()
    assert result3.embedded == 3
    assert _load_cursor(tmp_path, model_id) is not None

    # Tick 4: exactly 1 row remains — the batch cap is NOT hit (1 < 3), so
    # the tick genuinely exhausts its fetched window. The cursor now resets
    # to None, preserving the self-healing property.
    store5 = _open_store(tmp_path)
    try:
        result4 = run_embedding_backfill_tick(tmp_path, store5, batch_size=3, scan_cap=100)
    finally:
        store5.close()
    assert result4.embedded == 1
    assert result4.scanned == 1
    assert _load_cursor(tmp_path, model_id) is None

    store6 = _open_store(tmp_path)
    try:
        for m in made:
            assert _is_embedded(store6, m.id) is True
    finally:
        store6.close()


def test_scan_cap_bounds_rows_examined(tmp_path: Path) -> None:
    """scan_cap bounds how many rows are even looked at, independent of batch_size."""
    _seed(tmp_path, 50)
    store = _open_store(tmp_path)
    try:
        result = run_embedding_backfill_tick(tmp_path, store, batch_size=100, scan_cap=5)
    finally:
        store.close()

    assert result.scanned == 5
    assert result.embedded == 5


def test_default_scan_cap_is_reasonable() -> None:
    """Sanity: the module constant is a genuine bound, not accidentally unlimited."""
    assert 0 < DEFAULT_SCAN_CAP < 100_000


# ---------------------------------------------------------------------------
# Runtime-derived batch size (F1 #259 increment 3) — replaces the old
# hardcoded DEFAULT_BATCH_SIZE = 25.
# ---------------------------------------------------------------------------


def test_derived_batch_size_reflects_measured_per_embed_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch size a tick actually uses (when not explicitly overridden)
    is `floor(batch_budget_seconds / measured_per_embed_seconds)` — a
    derived figure, never the old literal 25. `scan_cap` is deliberately
    large here (well above the expected derived batch) so the FIX C
    scan_cap upper-clamp (see `test_derived_batch_size_is_clamped_to_scan_cap`
    below) does not interfere with what this test is isolating."""
    _seed(tmp_path, 1)
    store = _open_store(tmp_path)

    monkeypatch.setattr(
        embedding_backfill, "_measure_per_embed_seconds", lambda provider: 0.5  # noqa: ARG005
    )
    try:
        result = run_embedding_backfill_tick(tmp_path, store, scan_cap=1000)
    finally:
        store.close()

    budget = embedding_backfill._batch_budget_seconds()
    expected_batch = math.floor(budget / 0.5)
    assert result.batch_size == expected_batch
    assert result.batch_size != 25, "must be derived, not the old hardcoded literal"


def test_derived_batch_size_is_measured_once_per_process_and_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-embed measurement runs once (cached by model_id), not once
    per tick — a second tick under the same model_id must not re-measure."""
    _seed(tmp_path, 4)
    calls = {"n": 0}

    def _fake_measure(provider):  # noqa: ANN001, ARG001
        calls["n"] += 1
        return 1.0

    monkeypatch.setattr(embedding_backfill, "_measure_per_embed_seconds", _fake_measure)

    store = _open_store(tmp_path)
    # batch_size deliberately NOT overridden on either call — the derived
    # path (and thus the measure-once/cache mechanism) only engages when
    # the caller doesn't pin an explicit batch_size.
    run_embedding_backfill_tick(tmp_path, store, scan_cap=10)
    run_embedding_backfill_tick(tmp_path, store, scan_cap=10)
    store.close()

    assert calls["n"] == 1


def test_derived_batch_size_clamps_to_at_least_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A per-embed time so large that budget/per_embed floors to 0 must
    still make forward progress — clamp to >= 1."""
    _seed(tmp_path, 1)
    store = _open_store(tmp_path)

    huge_per_embed = embedding_backfill._batch_budget_seconds() * 100
    monkeypatch.setattr(
        embedding_backfill, "_measure_per_embed_seconds", lambda provider: huge_per_embed  # noqa: ARG005
    )
    try:
        result = run_embedding_backfill_tick(tmp_path, store, scan_cap=10)
    finally:
        store.close()

    assert result.batch_size == 1
    assert result.embedded == 1


def test_derived_batch_size_is_clamped_to_scan_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FIX C (F1 #259 increment-3 red-team, F2): a fluke-fast per-embed
    measurement must not derive a batch size larger than `scan_cap` — a
    tick can never usefully embed more rows than it even scans
    (`candidates` is itself `LIMIT scan_cap`), so the derived batch is
    clamped to it. Measuring at exactly the floor still derives an
    enormous RAW batch (budget / a-millisecond is tens of thousands); the
    scan_cap clamp is what keeps the reported/cached `batch_size` sane."""
    _seed(tmp_path, 3)
    store = _open_store(tmp_path)

    monkeypatch.setattr(
        embedding_backfill,
        "_measure_per_embed_seconds",
        lambda provider: embedding_backfill._MIN_PLAUSIBLE_PER_EMBED_SECONDS,  # noqa: ARG005
    )
    try:
        result = run_embedding_backfill_tick(tmp_path, store, scan_cap=3)
    finally:
        store.close()

    assert result.batch_size == 3
    assert result.embedded == 3


def test_measured_per_embed_seconds_is_floored_against_near_zero() -> None:
    """FIX C (F1 #259 increment-3 red-team, F2): a provider that returns
    (near-)instantly — a measurement artifact, e.g. a trivial fake or
    clock-resolution noise — must not drive the measured mean per-embed
    time below `_MIN_PLAUSIBLE_PER_EMBED_SECONDS`. This is the floor that
    keeps a fluke-fast measurement from deriving an absurd batch size in
    the first place (the scan_cap clamp above is the second line of
    defense on top of it)."""
    import numpy as np

    from brain.memory.embeddings import EmbeddingProvider

    class _InstantProvider(EmbeddingProvider):
        def embed(self, text: str):  # noqa: ANN201, ARG002
            return np.ones(8, dtype="float32")

        def embedding_dim(self) -> int:
            return 8

        def model_id(self) -> str:
            return "instant-test"

    measured = embedding_backfill._measure_per_embed_seconds(_InstantProvider())
    assert measured >= embedding_backfill._MIN_PLAUSIBLE_PER_EMBED_SECONDS


# ---------------------------------------------------------------------------
# Resumability / idempotency
# ---------------------------------------------------------------------------


def test_kill_and_rerun_resumes_without_dupes(tmp_path: Path) -> None:
    """Simulates a process kill mid-backfill: a fresh MemoryStore opened
    against the same on-disk file must resume, never re-embed a row that's
    already embedded, and the backlog must shrink to zero."""
    _seed(tmp_path, 12)

    store_a = _open_store(tmp_path)
    result_a = run_embedding_backfill_tick(tmp_path, store_a, batch_size=5, scan_cap=100)
    store_a.close()
    assert result_a.embedded == 5

    store_b = _open_store(tmp_path)
    result_b = run_embedding_backfill_tick(tmp_path, store_b, batch_size=5, scan_cap=100)
    store_b.close()
    assert result_b.embedded == 5
    # The backlog query itself already excludes the first 5 (they're no
    # longer NULL) — `scanned` here is 6, not 7: the loop peeks one row past
    # the batch-size cutoff to discover it should stop (pre-existing
    # accounting shape, unrelated to the backlog-definition change), never
    # re-examining the already-resolved first 5.
    assert result_b.scanned == 6

    store_c = _open_store(tmp_path)
    result_c = run_embedding_backfill_tick(tmp_path, store_c, batch_size=5, scan_cap=100)
    embedded_count = sum(1 for m in store_c.list_active() if _is_embedded(store_c, m.id))
    store_c.close()
    assert result_c.embedded == 2
    assert embedded_count == 12  # no dupes: 5 + 5 + 2, not more


# ---------------------------------------------------------------------------
# Keyset pagination / tied created_at
# ---------------------------------------------------------------------------


def test_tied_created_at_row_is_reachable_and_gets_embedded_across_ticks(
    tmp_path: Path,
) -> None:
    """3 memories share an IDENTICAL created_at (the shape a bulk migrator
    import produces). Under a bare-timestamp cursor, a tick that embedded 2
    of the 3 would pin the cursor at their shared created_at and the 3rd
    memory would become permanently unreachable. With the (created_at, id)
    keyset cursor, the 3rd memory must still be embedded."""
    shared_ts = datetime(2020, 1, 1, tzinfo=UTC)
    store = _open_store(tmp_path)
    made = []
    for i, suffix in enumerate(("aaaa", "bbbb", "cccc")):
        m = _mem(f"memory content long enough number {i}", created_at=shared_ts)
        m.id = f"{suffix}0000-0000-0000-0000-000000000000"
        store.create(m)
        made.append(m)
    store.close()

    # Tick 1: batch_size=2 processes the first 2 (by id, since created_at
    # ties) and, since the backlog is larger than scan_cap, pins a forward
    # cursor at the shared timestamp + the 2nd row's id.
    store = _open_store(tmp_path)
    result1 = run_embedding_backfill_tick(tmp_path, store, batch_size=2, scan_cap=2)
    store.close()
    assert result1.embedded == 2

    store_check = _open_store(tmp_path)
    assert _is_embedded(store_check, made[0].id) is True
    assert _is_embedded(store_check, made[1].id) is True
    assert _is_embedded(store_check, made[2].id) is False
    store_check.close()

    # Tick 2: the 3rd memory — sharing the exact created_at with the row the
    # cursor is pinned at — must still be reachable and get embedded.
    store = _open_store(tmp_path)
    result2 = run_embedding_backfill_tick(tmp_path, store, batch_size=2, scan_cap=2)
    store.close()

    assert result2.scanned == 1  # the 3rd row IS reachable, not silently skipped
    assert result2.embedded == 1

    store_final = _open_store(tmp_path)
    assert _is_embedded(store_final, made[2].id) is True
    store_final.close()


# ---------------------------------------------------------------------------
# Fault isolation + cursor-freeze fix (F1 #259 increment 3)
# ---------------------------------------------------------------------------


def test_provider_failure_does_not_starve_later_rows_in_the_same_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising provider is fault-isolated to the failing row: the tick
    keeps scanning past it within the SAME tick — no starvation of later
    rows."""
    from brain.memory import embeddings as embeddings_mod

    _seed(tmp_path, 3)

    class _FlakyProvider(EmbeddingProvider):
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, text: str):  # noqa: ANN201
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated provider failure")
            import numpy as np

            return np.ones(8, dtype="float32")

        def embedding_dim(self) -> int:
            return 8

        def model_id(self) -> str:
            return "flaky-test"

    # ONE shared instance: `store.embed_row` (called per-row) does its OWN
    # `build_embedding_provider()` lookup, separate from the one this tick
    # does for model_id/batch-size — a fresh instance per call would reset
    # `.calls` and never see the 2nd call the whole test depends on
    # (mirrors the real function's per-model_id singleton cache).
    flaky_provider = _FlakyProvider()
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: flaky_provider)

    store = _open_store(tmp_path)
    result = run_embedding_backfill_tick(tmp_path, store, batch_size=10, scan_cap=10)
    store.close()

    assert result.errors == 1
    assert result.scanned == 3  # keeps scanning past the failing (2nd) row
    assert result.embedded == 2  # 1st and 3rd rows succeed despite the 2nd failing


def test_permanently_failing_row_does_not_freeze_the_cursor_across_ticks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression proof for the pre-increment-3 freeze bug: a permanently-
    failing row used to pin `resolved_up_to` before it forever, so a later
    tick re-scanned the same scan_cap window (same failing row leading it)
    and never reached rows past the cap. Skip-and-log lets the cursor
    advance past the failure so a later tick makes genuine forward progress
    to rows beyond the first tick's scan window — this is what "fails on the
    old freeze behavior" means: under the old code, `result2.scanned` below
    would be 3 again (re-scanning "number 0/1/2"), not 2 (reaching "3"/"4")."""
    from brain.memory import embeddings as embeddings_mod

    made = _seed(tmp_path, 5)  # "... number 0" .. "... number 4"

    class _FailsOnFirst(EmbeddingProvider):
        def embed(self, text: str):  # noqa: ANN201
            if text.endswith("number 0"):
                raise RuntimeError("simulated PERMANENT provider failure")
            import numpy as np

            return np.ones(8, dtype="float32")

        def embedding_dim(self) -> int:
            return 8

        def model_id(self) -> str:
            return "fails-on-first-test"

    # Stateless (fails purely on text content), but captured as ONE instance
    # for consistency with the other fault-injection tests in this file —
    # see the shared-instance comment in
    # test_provider_failure_does_not_starve_later_rows_in_the_same_tick.
    fails_on_first_provider = _FailsOnFirst()
    monkeypatch.setattr(
        embeddings_mod, "build_embedding_provider", lambda: fails_on_first_provider
    )

    store = _open_store(tmp_path)
    result1 = run_embedding_backfill_tick(tmp_path, store, batch_size=10, scan_cap=3)
    store.close()
    assert result1.scanned == 3  # "number 0", "1", "2"
    assert result1.errors == 1
    assert result1.embedded == 2  # "1" and "2"; "0" permanently fails

    store2 = _open_store(tmp_path)
    result2 = run_embedding_backfill_tick(tmp_path, store2, batch_size=10, scan_cap=3)
    store2.close()

    # Forward progress past the failing row, not a frozen re-scan of the
    # same window.
    assert result2.scanned == 2  # "number 3" and "number 4"
    assert result2.embedded == 2
    assert result2.errors == 0

    # The permanently-failing row itself stays un-embedded (skip-and-log
    # never pretends it succeeded) — everything after it does not.
    store3 = _open_store(tmp_path)
    assert _is_embedded(store3, made[0].id) is False
    for m in made[1:]:
        assert _is_embedded(store3, m.id) is True
    store3.close()


def test_permanently_failing_row_does_not_starve_rows_after_it_across_many_ticks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row that fails on EVERY attempt must not block rows after it from
    being embedded — neither within one tick nor across many. Only the
    permanently-failing row itself stays stuck."""
    from brain.memory import embeddings as embeddings_mod

    made = _seed(tmp_path, 5)

    class _AlwaysFailsOnOne(EmbeddingProvider):
        def embed(self, text: str):  # noqa: ANN201
            if text.endswith("number 1"):
                raise RuntimeError("simulated PERMANENT provider failure")
            import numpy as np

            return np.ones(8, dtype="float32")

        def embedding_dim(self) -> int:
            return 8

        def model_id(self) -> str:
            return "always-fails-test"

    # Stateless, but ONE shared instance for consistency — see
    # test_provider_failure_does_not_starve_later_rows_in_the_same_tick.
    always_fails_provider = _AlwaysFailsOnOne()
    monkeypatch.setattr(
        embeddings_mod, "build_embedding_provider", lambda: always_fails_provider
    )

    total_embedded = 0
    total_errors = 0
    for _ in range(3):  # several ticks — the permanent failure never clears
        store = _open_store(tmp_path)
        result = run_embedding_backfill_tick(tmp_path, store, batch_size=10, scan_cap=10)
        store.close()
        total_embedded += result.embedded
        total_errors += result.errors

    store_final = _open_store(tmp_path)
    assert _is_embedded(store_final, made[1].id) is False  # "number 1" — permanent failure
    for m in (made[0], made[2], made[3], made[4]):
        assert _is_embedded(store_final, m.id) is True
    store_final.close()
    assert total_embedded == 4
    # Skip-and-log means it's only ever ATTEMPTED once it's freshly in the
    # backlog window each tick — the cursor reset (backlog fits in one scan
    # window here) means the still-NULL row reappears every tick, so it
    # contributes an error each time.
    assert total_errors == 3


# ---------------------------------------------------------------------------
# Old-format (pre-keyset, bare-timestamp) cursor migration
# ---------------------------------------------------------------------------


def test_old_bare_timestamp_cursor_resets_to_none_and_rescans_idempotently(
    tmp_path: Path,
) -> None:
    """A cursor file written by a pre-keyset build of this module stores
    ``cursor`` as a bare ISO-timestamp STRING, not the current
    ``{"created_at", "id"}`` dict. That old shape is deliberately NOT
    half-interpreted — it is treated as unparseable and reset to ``None``,
    reopening the whole backlog for a full rescan."""
    made = _seed(tmp_path, 5)

    from brain.memory import embeddings as embeddings_mod

    model_id = embeddings_mod.build_embedding_provider().model_id()

    old_format_payload = {
        "model_id": model_id,
        "cursor": "2020-01-03T00:00:00+00:00",  # bare string, not a dict
    }
    cursor_path = cadence_state_path(tmp_path, "embedding_backfill_cursor.json")
    cursor_path.parent.mkdir(parents=True, exist_ok=True)
    cursor_path.write_text(json.dumps(old_format_payload), encoding="utf-8")

    # Sanity check: confirm this old shape actually exercises the
    # old-format branch of _load_cursor.
    assert _load_cursor(tmp_path, model_id) is None

    store = _open_store(tmp_path)
    try:
        result = run_embedding_backfill_tick(tmp_path, store, batch_size=10, scan_cap=10)
    finally:
        store.close()

    assert result.scanned == 5  # full rescan from the top, not from "2020-01-03"
    assert result.errors == 0
    assert result.embedded == 5

    store2 = _open_store(tmp_path)
    try:
        for m in made:
            assert _is_embedded(store2, m.id) is True
    finally:
        store2.close()

    store3 = _open_store(tmp_path)
    try:
        result2 = run_embedding_backfill_tick(tmp_path, store3, batch_size=10, scan_cap=10)
    finally:
        store3.close()

    assert result2.embedded == 0
    assert result2.scanned == 0
    assert result2.errors == 0


# ---------------------------------------------------------------------------
# Row + warm-matrix integration (F1 #259 increment 3: writes go through
# MemoryStore.embed_row, which also pushes the vector into the warm matrix).
# ---------------------------------------------------------------------------


def test_backfill_writes_land_in_the_warm_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row the backfill embeds must be immediately visible in the process
    warm matrix (via `store.embed_row`), not just on the row — mirrors
    embed-on-write's own contract."""
    import numpy as np

    from brain.bridge import model_tier
    from brain.memory import embeddings as embeddings_mod
    from brain.memory.embedding_matrix import build_embedding_matrix

    provider = embeddings_mod.FakeEmbeddingProvider(dim=384)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, provider.model_id())

    made = _seed(tmp_path, 1)
    store = _open_store(tmp_path)
    try:
        result = run_embedding_backfill_tick(tmp_path, store, batch_size=10, scan_cap=10)
    finally:
        store.close()
    assert result.embedded == 1

    matrix = build_embedding_matrix(Path(tmp_path) / "memories.db")
    vec = matrix.get(made[0].id)
    assert vec is not None
    np.testing.assert_array_equal(vec, provider.embed(made[0].content).astype(np.float32))


def test_warm_matrix_put_failure_is_not_counted_as_a_backfill_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FIX D (F1 #259 increment-3 red-team, F3): once `store.embed_row`'s
    row UPDATE has committed, the row IS durably embedded — a subsequent
    `EmbeddingMatrix.put` failure must not propagate out of `embed_row`,
    and must therefore not be counted in the tick's `errors` field. Before
    the fix, a put failure here would have been indistinguishable from a
    genuine embed failure, under-reporting a row that is actually fine."""
    from brain.bridge import model_tier
    from brain.memory import embedding_matrix as embedding_matrix_mod
    from brain.memory import embeddings as embeddings_mod

    provider = embeddings_mod.FakeEmbeddingProvider(dim=384)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, provider.model_id())

    def _boom_put(self, memory_id: str, vector) -> None:  # noqa: ANN001, ARG001
        raise RuntimeError("simulated warm-matrix put failure")

    monkeypatch.setattr(embedding_matrix_mod.EmbeddingMatrix, "put", _boom_put)

    made = _seed(tmp_path, 1)
    store = _open_store(tmp_path)
    try:
        result = run_embedding_backfill_tick(tmp_path, store, batch_size=10, scan_cap=10)
    finally:
        store.close()

    assert result.embedded == 1
    assert result.errors == 0

    store2 = _open_store(tmp_path)
    try:
        assert _is_embedded(store2, made[0].id) is True
    finally:
        store2.close()


# ---------------------------------------------------------------------------
# Model-mismatch backlog clause (FIX A, Planning-ruled 2026-09-16, F1 #259
# increment-3 spec-gap): after a MODEL_EMBEDDING swap, old-model rows stay
# non-NULL but are stale — the backlog must re-open them, not just rows
# whose embedding is outright NULL.
# ---------------------------------------------------------------------------


def test_backfill_reembeds_stale_model_rows_after_a_model_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row embedded under an OLD model_id (embedding non-NULL, but
    `embedding_model_id` != the current model) IS in the backlog and gets
    re-embedded under the CURRENT model. A row already embedded under the
    CURRENT model is left alone (not re-embedded, not counted again)."""
    from brain.bridge import model_tier
    from brain.memory import embeddings as embeddings_mod

    # FakeEmbeddingProvider.model_id() is dim-qualified ("fake-<dim>"), not
    # per-instance-unique — two DIFFERENT dims is how this test gets two
    # genuinely different model_ids to simulate a swap (the warm matrix
    # isn't exercised by this test, so the dim mismatch doesn't matter here).
    old_provider = embeddings_mod.FakeEmbeddingProvider(dim=256)

    made = _seed(tmp_path, 2)
    stale_row, current_row = made[0], made[1]

    store = _open_store(tmp_path)
    # Embed both rows under the "old" model first.
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: old_provider)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, old_provider.model_id())
    store.embed_row(stale_row.id, stale_row.content)
    store.embed_row(current_row.id, current_row.content)

    # Now swap to a NEW model (different model_id) and re-embed only the
    # "current_row" under it directly (simulating it was written fresh
    # after the swap), leaving "stale_row" behind under the old model_id.
    new_provider = embeddings_mod.FakeEmbeddingProvider(dim=384)
    assert new_provider.model_id() != old_provider.model_id()
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: new_provider)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, new_provider.model_id())
    store.embed_row(current_row.id, current_row.content)

    result = run_embedding_backfill_tick(tmp_path, store, batch_size=10, scan_cap=10)
    store.close()

    # Only the stale-model row was backlog for this tick.
    assert result.scanned == 1
    assert result.embedded == 1

    store2 = _open_store(tmp_path)
    try:
        stale_after = store2._conn.execute(
            "SELECT embedding_model_id FROM memories WHERE id = ?", (stale_row.id,)
        ).fetchone()
        current_after = store2._conn.execute(
            "SELECT embedding_model_id FROM memories WHERE id = ?", (current_row.id,)
        ).fetchone()
    finally:
        store2.close()

    assert stale_after["embedding_model_id"] == new_provider.model_id()
    assert current_after["embedding_model_id"] == new_provider.model_id()


def test_backfill_is_a_no_op_when_all_rows_match_the_current_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Steady state (no model swap): every row already carries the current
    model's id, so the model-mismatch clause contributes nothing — the
    backlog query returns empty, matching the pre-FIX-A no-op behavior."""
    from brain.bridge import model_tier
    from brain.memory import embeddings as embeddings_mod

    provider = embeddings_mod.FakeEmbeddingProvider(dim=384)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, provider.model_id())

    made = _seed(tmp_path, 3)
    store = _open_store(tmp_path)
    for m in made:
        store.embed_row(m.id, m.content)

    result = run_embedding_backfill_tick(tmp_path, store, batch_size=10, scan_cap=10)
    store.close()

    assert result.scanned == 0
    assert result.embedded == 0


# ---------------------------------------------------------------------------
# Drain-to-completion (F1 #259 increment 6 — the `nell embed backfill` CLI's
# operator escape hatch). Reuses this file's `_mem`/`_open_store`/`_seed`/
# `_is_embedded` helpers.
# ---------------------------------------------------------------------------


def test_drain_embeds_every_row_and_terminates_with_empty_backlog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The common case: no failures. The drain embeds the WHOLE backlog in
    one call and reports an empty backlog afterward (acceptance criterion 8
    / the increment-6 spec: "drains all un-embedded rows to completion")."""
    from brain.bridge import model_tier
    from brain.memory import embeddings as embeddings_mod

    provider = embeddings_mod.FakeEmbeddingProvider(dim=384)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, provider.model_id())

    made = _seed(tmp_path, 11)
    store = _open_store(tmp_path)
    try:
        # Small batch/scan bounds force several ticks so this also proves
        # the LOOP (not just a single generous tick) does the draining.
        result = run_embedding_backfill_to_completion(
            tmp_path, store, batch_size=3, scan_cap=3
        )
    finally:
        store.close()

    assert result.embedded == 11
    assert result.failed == 0
    assert result.stopped_reason == "no_more_candidates"
    assert result.ticks > 1  # genuinely looped, not a one-shot generous tick

    store2 = _open_store(tmp_path)
    try:
        for m in made:
            assert _is_embedded(store2, m.id) is True
        assert store2.count_unembedded(current_model_id=provider.model_id(), min_chars=1) == 0
    finally:
        store2.close()


def test_drain_terminates_when_a_row_permanently_fails_without_looping_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row that fails on EVERY attempt must not spin the drain loop
    forever — this is the exact bug this function exists to close: a small
    backlog that is entirely (or down to its last row) permanently-failing
    resets its tick-level cursor to `None` on every tick (see
    `run_embedding_backfill_tick`'s closing comment), so a naive
    `while backlog: tick()` loop would repeat the identical failing tick
    without end. Proven here by: the call RETURNS (pytest itself is the
    forever-loop timeout backstop), a bounded `ticks` count, the failing row
    left un-embedded, every other row embedded, and `result.failed == 1` —
    NOT 2, proving the failed count is a deduped final backlog count, not a
    naive sum of the two separate tick-level attempts this scenario
    produces (see the function's own docstring)."""
    from brain.bridge import model_tier
    from brain.memory import embeddings as embeddings_mod

    made = _seed(tmp_path, 5)  # "... number 0" .. "... number 4"

    class _AlwaysFailsOnTwo(embeddings_mod.EmbeddingProvider):
        def embed(self, text: str):  # noqa: ANN201
            if text.endswith("number 2"):
                raise RuntimeError("simulated PERMANENT provider failure")
            import numpy as np

            return np.ones(8, dtype="float32")

        def embedding_dim(self) -> int:
            return 8

        def model_id(self) -> str:
            return "always-fails-on-two-test"

    provider = _AlwaysFailsOnTwo()
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, provider.model_id())

    store = _open_store(tmp_path)
    try:
        result = run_embedding_backfill_to_completion(
            tmp_path, store, batch_size=2, scan_cap=2
        )
    finally:
        store.close()

    assert result.embedded == 4
    assert result.failed == 1  # deduped final count, not a raw attempt sum
    assert result.stopped_reason == "stalled_no_progress"
    # Bounded: nowhere near an infinite loop. Generous upper bound so this
    # isn't brittle to the exact tick-accounting shape, while still failing
    # hard if the termination guard regresses into spinning.
    assert result.ticks < 20

    store2 = _open_store(tmp_path)
    try:
        failing = made[2]
        assert _is_embedded(store2, failing.id) is False
        for m in (made[0], made[1], made[3], made[4]):
            assert _is_embedded(store2, m.id) is True
        assert store2.count_unembedded(current_model_id=provider.model_id(), min_chars=1) == 1
    finally:
        store2.close()


def test_drain_scattered_failure_in_a_large_backlog_still_terminates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A permanently-failing row BURIED inside a backlog larger than
    `scan_cap` must not stall the drain either — its containing tick's
    cursor advances PAST the full scan_cap window regardless of the failure
    (see run_embedding_backfill_tick's skip-and-log cursor logic), so later
    ticks keep making genuine progress on rows further along. The drain
    still terminates, and the one failing row is still reflected in the
    final `.failed` count even though the loop's own stop condition here is
    "no_more_candidates", not "stalled_no_progress" (see the function's
    docstring on why a scattered failure can sit behind an already-advanced
    cursor)."""
    from brain.bridge import model_tier
    from brain.memory import embeddings as embeddings_mod

    made = _seed(tmp_path, 20)
    failing = made[10]

    class _FailsOnOneBuried(embeddings_mod.EmbeddingProvider):
        def embed(self, text: str):  # noqa: ANN201
            if text.endswith("number 10"):
                raise RuntimeError("simulated PERMANENT provider failure")
            import numpy as np

            return np.ones(8, dtype="float32")

        def embedding_dim(self) -> int:
            return 8

        def model_id(self) -> str:
            return "fails-on-one-buried-test"

    provider = _FailsOnOneBuried()
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, provider.model_id())

    store = _open_store(tmp_path)
    try:
        result = run_embedding_backfill_to_completion(
            tmp_path, store, batch_size=5, scan_cap=5
        )
    finally:
        store.close()

    assert result.embedded == 19
    assert result.failed == 1
    assert result.ticks < 20

    store2 = _open_store(tmp_path)
    try:
        assert _is_embedded(store2, failing.id) is False
        for m in made:
            if m.id != failing.id:
                assert _is_embedded(store2, m.id) is True
    finally:
        store2.close()


def test_drain_progress_cb_reports_cumulative_running_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`progress_cb` is invoked after every tick with CUMULATIVE
    `(embedded, failed, scanned)` — the CLI ticker's contract (F1 #259
    increment 6). Cumulative embedded must be monotonically non-decreasing
    and the LAST call must match the function's own returned totals."""
    from brain.bridge import model_tier
    from brain.memory import embeddings as embeddings_mod

    provider = embeddings_mod.FakeEmbeddingProvider(dim=384)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, provider.model_id())

    _seed(tmp_path, 7)
    store = _open_store(tmp_path)

    calls: list[tuple[int, int, int]] = []

    def _cb(embedded: int, failed: int, scanned: int) -> None:
        calls.append((embedded, failed, scanned))

    try:
        result = run_embedding_backfill_to_completion(
            tmp_path, store, batch_size=2, scan_cap=2, progress_cb=_cb
        )
    finally:
        store.close()

    assert len(calls) == result.ticks
    assert len(calls) > 1
    embedded_series = [c[0] for c in calls]
    assert embedded_series == sorted(embedded_series)  # monotonically non-decreasing
    assert calls[-1][0] == result.embedded
    assert calls[-1][2] == result.scanned


def test_drain_empty_backlog_returns_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to embed — the drain must not loop at all, just confirm
    there is nothing left."""
    from brain.bridge import model_tier
    from brain.memory import embeddings as embeddings_mod

    provider = embeddings_mod.FakeEmbeddingProvider(dim=384)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, provider.model_id())

    store = _open_store(tmp_path)  # no memories seeded at all
    try:
        result = run_embedding_backfill_to_completion(tmp_path, store, batch_size=10, scan_cap=10)
    finally:
        store.close()

    assert result.embedded == 0
    assert result.failed == 0
    assert result.scanned == 0
    assert result.ticks == 1
    assert result.stopped_reason == "no_more_candidates"


# ---------------------------------------------------------------------------
# delete_legacy_embeddings_db — run-once, fail-safe deletion of the old
# embeddings.db file (F1 #259 increment 9, spec §5 / S9, invariant I9).
# ---------------------------------------------------------------------------


def _write_dummy_embeddings_db(persona_dir: Path, *, with_sidecars: bool = True) -> Path:
    """A stand-in for the legacy embeddings.db file — content is irrelevant
    to the function under test, which never opens it, only checks existence
    and deletes it."""
    db_path = persona_dir / "embeddings.db"
    db_path.write_bytes(b"legacy sqlite content, never actually read")
    if with_sidecars:
        (persona_dir / "embeddings.db-wal").write_bytes(b"wal")
        (persona_dir / "embeddings.db-shm").write_bytes(b"shm")
    return db_path


def test_delete_legacy_embeddings_db_deletes_when_fully_embedded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) embeddings.db exists + every active >=MIN_CHARS row already
    carries a current-model embedding -> the file AND its -wal/-shm
    sidecars are deleted."""
    from brain.memory import embeddings as embeddings_mod

    provider = embeddings_mod.FakeEmbeddingProvider(dim=384)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)

    made = _seed(tmp_path, 2)
    store = _open_store(tmp_path)
    try:
        for m in made:
            store.embed_row(m.id, m.content)

        db_path = _write_dummy_embeddings_db(tmp_path)
        assert store.count_unembedded(current_model_id=provider.model_id(), min_chars=1) == 0

        result = delete_legacy_embeddings_db(tmp_path, store)
    finally:
        store.close()

    assert result is True
    assert not db_path.exists()
    assert not (tmp_path / "embeddings.db-wal").exists()
    assert not (tmp_path / "embeddings.db-shm").exists()


def test_delete_legacy_embeddings_db_defers_when_backlog_nonempty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) embeddings.db exists but some active row is still un-embedded
    under the current model -> NOT deleted, deferred for the next run."""
    from brain.memory import embeddings as embeddings_mod

    provider = embeddings_mod.FakeEmbeddingProvider(dim=384)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)

    made = _seed(tmp_path, 2)
    store = _open_store(tmp_path)
    try:
        # Only embed the first row — the second stays backlog.
        store.embed_row(made[0].id, made[0].content)

        db_path = _write_dummy_embeddings_db(tmp_path)
        assert store.count_unembedded(current_model_id=provider.model_id(), min_chars=1) == 1

        result = delete_legacy_embeddings_db(tmp_path, store)
    finally:
        store.close()

    assert result is False
    assert db_path.exists()
    assert (tmp_path / "embeddings.db-wal").exists()
    assert (tmp_path / "embeddings.db-shm").exists()


def test_delete_legacy_embeddings_db_noop_when_file_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(c) no embeddings.db at all -> no-op, no error, no provider needed
    (the file-existence check short-circuits before any embedding-provider
    lookup, so a broken/unavailable provider must not matter here)."""
    from brain.memory import embeddings as embeddings_mod

    def _boom() -> None:
        raise AssertionError("build_embedding_provider should not be called when the file is absent")

    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", _boom)

    store = _open_store(tmp_path)
    try:
        assert not (tmp_path / "embeddings.db").exists()
        result = delete_legacy_embeddings_db(tmp_path, store)
    finally:
        store.close()

    assert result is False
    assert not (tmp_path / "embeddings.db").exists()


def test_delete_legacy_embeddings_db_idempotent_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(d) a second call after the file is already gone is a clean no-op —
    does not raise, does not re-log a deletion."""
    from brain.memory import embeddings as embeddings_mod

    provider = embeddings_mod.FakeEmbeddingProvider(dim=384)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)

    made = _seed(tmp_path, 1)
    store = _open_store(tmp_path)
    try:
        store.embed_row(made[0].id, made[0].content)
        _write_dummy_embeddings_db(tmp_path)

        first = delete_legacy_embeddings_db(tmp_path, store)
        second = delete_legacy_embeddings_db(tmp_path, store)
    finally:
        store.close()

    assert first is True
    assert second is False
    assert not (tmp_path / "embeddings.db").exists()


def test_delete_legacy_embeddings_db_delete_error_is_caught_and_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """(e) a delete failure (e.g. a permission error) is caught + logged,
    never raised — a stuck stale file must never crash a startup caller."""
    from brain.memory import embeddings as embeddings_mod

    provider = embeddings_mod.FakeEmbeddingProvider(dim=384)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)

    made = _seed(tmp_path, 1)
    store = _open_store(tmp_path)
    try:
        store.embed_row(made[0].id, made[0].content)
        db_path = _write_dummy_embeddings_db(tmp_path, with_sidecars=False)

        def _raise_permission_error(self: Path, *args: object, **kwargs: object) -> None:  # noqa: ANN401
            raise PermissionError(f"simulated permission error deleting {self}")

        monkeypatch.setattr(Path, "unlink", _raise_permission_error)

        with caplog.at_level("WARNING"):
            result = delete_legacy_embeddings_db(tmp_path, store)  # must not raise
    finally:
        store.close()

    assert result is False
    assert db_path.exists()  # left in place, not partially removed
    assert "failed to delete legacy embeddings.db" in caplog.text
