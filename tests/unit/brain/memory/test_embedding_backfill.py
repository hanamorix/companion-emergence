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
    run_embedding_backfill_tick,
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
    """Very short content is never embedded — noise-vector guard."""
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

    assert result.skipped_short == 1
    assert result.embedded == 1


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
    derived figure, never the old literal 25."""
    _seed(tmp_path, 1)
    store = _open_store(tmp_path)

    monkeypatch.setattr(
        embedding_backfill, "_measure_per_embed_seconds", lambda provider: 0.5  # noqa: ARG005
    )
    try:
        result = run_embedding_backfill_tick(tmp_path, store, scan_cap=10)
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
