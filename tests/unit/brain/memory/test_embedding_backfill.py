"""Tests for brain.memory.embedding_backfill — the idle-chipped backfill.

Stage 2 of the local semantic-retrieval build. Every test uses on-disk
sqlite files (not ":memory:") for both memories.db and embeddings.db, since
several tests simulate a "kill mid-backfill" by discarding one MemoryStore/
EmbeddingCache pair and opening a fresh one against the same files — an
in-memory db would not survive that.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from brain.memory.embedding_backfill import (
    DEFAULT_SCAN_CAP,
    MIN_CHARS_TO_EMBED,
    run_embedding_backfill_tick,
)
from brain.memory.embeddings import EmbeddingCache, EmbeddingProvider, FakeEmbeddingProvider
from brain.memory.store import Memory, MemoryStore


def _mem(content: str, *, created_at: datetime) -> Memory:
    m = Memory.create_new(content=content, memory_type="conversation", domain="us")
    m.created_at = created_at
    return m


def _open_store(persona_dir: Path) -> MemoryStore:
    return MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)


def _open_cache(persona_dir: Path, provider: EmbeddingProvider | None = None) -> EmbeddingCache:
    return EmbeddingCache(persona_dir / "embeddings.db", provider or FakeEmbeddingProvider(dim=32))


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


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


def test_tick_embeds_missing_rows(tmp_path: Path) -> None:
    """A fresh backlog of un-embedded rows gets embedded, up to batch_size."""
    _seed(tmp_path, 3)
    store = _open_store(tmp_path)
    cache = _open_cache(tmp_path)
    try:
        result = run_embedding_backfill_tick(tmp_path, store, cache, batch_size=10, scan_cap=10)
    finally:
        store.close()
        cache.close()

    assert result.embedded == 3
    assert result.scanned == 3
    assert result.errors == 0


def test_tick_is_a_near_no_op_once_caught_up(tmp_path: Path) -> None:
    """A second tick with nothing new to embed does no real compute work."""
    _seed(tmp_path, 3)
    store = _open_store(tmp_path)
    cache = _open_cache(tmp_path)
    try:
        run_embedding_backfill_tick(tmp_path, store, cache, batch_size=10, scan_cap=10)
        result = run_embedding_backfill_tick(tmp_path, store, cache, batch_size=10, scan_cap=10)
    finally:
        store.close()
        cache.close()

    assert result.embedded == 0
    assert result.errors == 0


def test_skips_rows_under_min_chars(tmp_path: Path) -> None:
    """Very short content is never embedded — noise-vector guard."""
    store = _open_store(tmp_path)
    short = "x" * (MIN_CHARS_TO_EMBED - 1)
    long_enough = "y" * MIN_CHARS_TO_EMBED
    store.create(_mem(short, created_at=datetime(2020, 1, 1, tzinfo=UTC)))
    store.create(_mem(long_enough, created_at=datetime(2020, 1, 2, tzinfo=UTC)))
    store.close()

    store = _open_store(tmp_path)
    cache = _open_cache(tmp_path)
    result = run_embedding_backfill_tick(tmp_path, store, cache, batch_size=10, scan_cap=10)
    assert cache.has(short) is False
    assert cache.has(long_enough) is True
    store.close()
    cache.close()

    assert result.skipped_short == 1
    assert result.embedded == 1


def test_already_cached_rows_do_not_consume_embed_budget(tmp_path: Path) -> None:
    """A row already embedded (e.g. by the ingest pipeline's own embed-on-write
    side effect) is skipped without eating into batch_size."""
    _seed(tmp_path, 5)

    # Pre-embed 2 of the 5 rows directly, simulating the ingest-pipeline path.
    store = _open_store(tmp_path)
    cache = _open_cache(tmp_path)
    active = store.list_active()
    for m in active[:2]:
        cache.get_or_compute(m.content)

    result = run_embedding_backfill_tick(tmp_path, store, cache, batch_size=10, scan_cap=10)
    store.close()
    cache.close()

    assert result.already_cached == 2
    assert result.embedded == 3  # the remaining 3, not re-embedding the pre-cached ones


# ---------------------------------------------------------------------------
# Bounded per tick
# ---------------------------------------------------------------------------


def test_batch_size_bounds_embeds_per_tick(tmp_path: Path) -> None:
    """A backlog larger than batch_size is only chipped by batch_size per tick."""
    _seed(tmp_path, 10)
    store = _open_store(tmp_path)
    cache = _open_cache(tmp_path)
    result = run_embedding_backfill_tick(tmp_path, store, cache, batch_size=4, scan_cap=100)
    assert result.embedded == 4
    assert cache.count() == 4
    store.close()
    cache.close()


def test_backlog_shrinks_across_repeated_ticks(tmp_path: Path) -> None:
    """Repeated bounded ticks eventually clear a backlog larger than one batch."""
    _seed(tmp_path, 10)

    total_embedded = 0
    for _ in range(5):  # 5 ticks * batch_size=3 >= 10 rows
        store = _open_store(tmp_path)
        cache = _open_cache(tmp_path)
        result = run_embedding_backfill_tick(tmp_path, store, cache, batch_size=3, scan_cap=100)
        store.close()
        cache.close()
        total_embedded += result.embedded

    assert total_embedded == 10
    cache = _open_cache(tmp_path)
    assert cache.count() == 10
    cache.close()


def test_scan_cap_bounds_rows_examined(tmp_path: Path) -> None:
    """scan_cap bounds how many rows are even looked at, independent of batch_size."""
    _seed(tmp_path, 50)
    store = _open_store(tmp_path)
    cache = _open_cache(tmp_path)
    try:
        result = run_embedding_backfill_tick(tmp_path, store, cache, batch_size=100, scan_cap=5)
    finally:
        store.close()
        cache.close()

    assert result.scanned == 5
    assert result.embedded == 5


def test_default_scan_cap_is_reasonable() -> None:
    """Sanity: the module constant is a genuine bound, not accidentally unlimited."""
    assert 0 < DEFAULT_SCAN_CAP < 100_000


# ---------------------------------------------------------------------------
# Resumability / idempotency — the load-bearing acceptance criterion
# ---------------------------------------------------------------------------


def test_kill_and_rerun_resumes_without_dupes(tmp_path: Path) -> None:
    """Simulates a process kill mid-backfill: a fresh MemoryStore/EmbeddingCache
    pair opened against the same on-disk files must resume, never re-embed
    (dupe) an already-cached row, and the backlog must shrink to zero."""
    _seed(tmp_path, 12)

    # Tick 1: process A embeds a partial batch, then "dies" (handles closed,
    # discarded — nothing more happens with them).
    store_a = _open_store(tmp_path)
    cache_a = _open_cache(tmp_path)
    result_a = run_embedding_backfill_tick(tmp_path, store_a, cache_a, batch_size=5, scan_cap=100)
    store_a.close()
    cache_a.close()
    assert result_a.embedded == 5

    count_after_a = _open_cache(tmp_path)
    assert count_after_a.count() == 5
    count_after_a.close()

    # Tick 2: a fresh process resumes — must not re-embed the first 5, and
    # must make forward progress on the remaining 7.
    store_b = _open_store(tmp_path)
    cache_b = _open_cache(tmp_path)
    result_b = run_embedding_backfill_tick(tmp_path, store_b, cache_b, batch_size=5, scan_cap=100)
    store_b.close()
    cache_b.close()
    assert result_b.embedded == 5
    assert result_b.already_cached == 0  # cursor skipped straight past the resolved prefix

    count_after_b = _open_cache(tmp_path)
    assert count_after_b.count() == 10  # no dupes: 5 + 5, not 5 + 10
    count_after_b.close()

    # Tick 3: clears the remaining 2.
    store_c = _open_store(tmp_path)
    cache_c = _open_cache(tmp_path)
    result_c = run_embedding_backfill_tick(tmp_path, store_c, cache_c, batch_size=5, scan_cap=100)
    store_c.close()
    cache_c.close()
    assert result_c.embedded == 2

    final_cache = _open_cache(tmp_path)
    assert final_cache.count() == 12
    final_cache.close()


def test_provider_failure_does_not_lose_progress_or_crash(tmp_path: Path) -> None:
    """A raising provider stops the tick early (fault-isolated) without
    losing already-embedded rows or crashing the caller; the failing row is
    retried on the next tick."""
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

    store = _open_store(tmp_path)
    provider = _FlakyProvider()
    cache = EmbeddingCache(tmp_path / "embeddings.db", provider)
    result = run_embedding_backfill_tick(tmp_path, store, cache, batch_size=10, scan_cap=10)
    store.close()
    cache.close()

    assert result.embedded == 1  # the first row succeeded before the failure
    assert result.errors == 1
    assert result.scanned == 2  # stopped at the failing row, never reached the third
    # Same-model retry semantics (does the failed row get picked up again
    # without re-embedding the one that already succeeded?) are covered by
    # test_same_model_retry_after_failure_reembeds_only_the_failed_row below.


def test_same_model_retry_after_failure_reembeds_only_the_failed_row(tmp_path: Path) -> None:
    """A same-model second tick after a mid-batch failure only needs to
    embed the row(s) still missing — already-embedded ones are cache hits."""
    _seed(tmp_path, 3)

    class _FailsOnce(EmbeddingProvider):
        def __init__(self) -> None:
            self.attempts = 0
            self._model_id = "fails-once-test"

        def embed(self, text: str):  # noqa: ANN201
            self.attempts += 1
            if self.attempts == 2 and text.endswith("number 1"):
                raise RuntimeError("simulated transient failure")
            import numpy as np

            return np.ones(8, dtype="float32")

        def embedding_dim(self) -> int:
            return 8

        def model_id(self) -> str:
            return self._model_id

    provider = _FailsOnce()
    store = _open_store(tmp_path)
    cache = EmbeddingCache(tmp_path / "embeddings.db", provider)
    result1 = run_embedding_backfill_tick(tmp_path, store, cache, batch_size=10, scan_cap=10)
    store.close()
    cache.close()
    assert result1.errors == 1
    assert result1.embedded == 1

    # Reopen with a provider under the SAME model_id that no longer fails.
    class _NowWorks(EmbeddingProvider):
        def embed(self, text: str):  # noqa: ANN201
            import numpy as np

            return np.ones(8, dtype="float32")

        def embedding_dim(self) -> int:
            return 8

        def model_id(self) -> str:
            return "fails-once-test"

    store2 = _open_store(tmp_path)
    cache2 = EmbeddingCache(tmp_path / "embeddings.db", _NowWorks())
    result2 = run_embedding_backfill_tick(tmp_path, store2, cache2, batch_size=10, scan_cap=10)
    store2.close()
    cache2.close()

    assert result2.errors == 0
    assert result2.embedded == 2  # the previously-failed row + the third row

    final_cache = EmbeddingCache(tmp_path / "embeddings.db", _NowWorks())
    assert final_cache.count() == 3
    final_cache.close()


# ---------------------------------------------------------------------------
# Model swap
# ---------------------------------------------------------------------------


def test_model_swap_reopens_the_full_backlog(tmp_path: Path) -> None:
    """A different model_id must re-embed everything, not trust the old
    cursor position (old vectors are a different model's vector space)."""
    _seed(tmp_path, 4)

    store = _open_store(tmp_path)
    cache_old = _open_cache(tmp_path, FakeEmbeddingProvider(dim=16))
    run_embedding_backfill_tick(tmp_path, store, cache_old, batch_size=10, scan_cap=10)
    store.close()
    cache_old.close()

    old_count = EmbeddingCache(tmp_path / "embeddings.db", FakeEmbeddingProvider(dim=16)).count()
    assert old_count == 4

    # New model (different dim -> different model_id).
    store2 = _open_store(tmp_path)
    cache_new = _open_cache(tmp_path, FakeEmbeddingProvider(dim=32))
    result = run_embedding_backfill_tick(tmp_path, store2, cache_new, batch_size=10, scan_cap=10)
    store2.close()
    cache_new.close()

    assert result.embedded == 4  # full re-embed under the new model_id, cursor ignored
