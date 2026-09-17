"""Tests for brain.ingest.dedupe — DEDUPE stage.

F1 (#259) increment 5: `is_duplicate` sources EXISTING vectors from the warm
matrix over `store`'s memories.db (i.e. already row-embedded memories), and
embeds the transient, not-yet-committed CANDIDATE text directly via the
production provider. Neither side of the comparison touches the old
`embeddings.db` / `EmbeddingCache` content-hash cache at all.
"""

from __future__ import annotations

import pytest

from brain.bridge import model_tier
from brain.ingest.dedupe import DEFAULT_DEDUP_THRESHOLD, is_duplicate
from brain.memory import embeddings as embeddings_mod
from brain.memory.store import Memory, MemoryStore

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _use_384_fake_provider(monkeypatch: pytest.MonkeyPatch) -> embeddings_mod.FakeEmbeddingProvider:
    """Align `build_embedding_provider()` AND `model_tier`'s embedding-tier
    model id to one 384-dim `FakeEmbeddingProvider` — mirrors the identical
    helper in tests/unit/brain/memory/test_store.py. The model-id alignment
    is load-bearing: `is_duplicate` embeds the candidate through
    `build_embedding_provider()`, but the warm matrix's lazy build filters
    EXISTING rows by `model_tier.model_for_tier(TIER_EMBEDDING)` — a SEPARATE
    lookup that must resolve to the same model id, or the matrix loads
    nothing. 384 specifically is no longer load-bearing (#259 inc7 red-team
    F1: `EmbeddingMatrix` decodes each row to its own stored byte-length, not
    a hardcoded expected width) — kept only for consistency with the rest of
    this suite's production-shaped vectors."""
    provider = embeddings_mod.FakeEmbeddingProvider(dim=384)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, provider.model_id())
    return provider


class _NamedFakeProvider(embeddings_mod.FakeEmbeddingProvider):
    """A 384-dim FakeEmbeddingProvider with an EXPLICIT model id, so two
    instances can share the (matrix-required) 384 dim while still reporting
    different model ids — `FakeEmbeddingProvider.model_id()` is derived
    purely from `dim`, so two same-dim instances can't otherwise differ."""

    def __init__(self, model_id: str) -> None:
        super().__init__(dim=384)
        self._model_id_override = model_id

    def model_id(self) -> str:
        return self._model_id_override


def _seed_embedded_memory(store: MemoryStore, content: str) -> str:
    """Create + commit a memory AND embed its row (simulating embed-on-write
    at pending-queue -> committed-memory promotion, F1 #259 step 4) so it is
    visible to the warm matrix `is_duplicate` reads from."""
    m = Memory.create_new(content=content, memory_type="episodic", domain="chat", emotions={})
    store.create(m)
    store.embed_row(m.id, m.content)
    return m.id


@pytest.fixture
def store(tmp_path) -> MemoryStore:
    # A REAL file, not MemoryStore(":memory:") — the warm matrix opens its
    # OWN sqlite3 connection straight to store.db_path; an in-memory-only
    # store's writes are invisible to that connection (two independent
    # `:memory:` databases). Mirrors test_store.py's identical fixture note.
    return MemoryStore(tmp_path / "memories.db")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_is_duplicate_returns_false_when_no_embedded_rows_exist(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cold-start / pre-backfill: no row has an embedding yet, so the warm
    matrix snapshot is empty and dedupe simply lets the item through."""
    _use_384_fake_provider(monkeypatch)
    assert is_duplicate("fresh memory", store=store) is False


def test_is_duplicate_returns_true_for_a_near_duplicate(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed, row-embedded memory with the SAME text as the candidate
    is caught as a duplicate: existing vector from the matrix, candidate
    vector from a direct provider embed — same text, deterministic fake
    provider, so cosine similarity is 1.0."""
    _use_384_fake_provider(monkeypatch)
    text = "Nell loves writing and spending time with Hana"
    _seed_embedded_memory(store, text)

    assert is_duplicate(text, store=store, threshold=DEFAULT_DEDUP_THRESHOLD) is True


def test_is_duplicate_returns_false_for_a_non_duplicate(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sufficiently different stored content does not trip the threshold."""
    _use_384_fake_provider(monkeypatch)
    _seed_embedded_memory(store, "the quick brown fox jumps over the lazy dog")

    result = is_duplicate(
        "Nell is a sweater-wearing novelist", store=store, threshold=DEFAULT_DEDUP_THRESHOLD
    )
    assert result is False


def test_is_duplicate_ignores_rows_from_a_different_model_id(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row embedded under a PRIOR/different model id must never enter the
    comparison. Dedupe no longer filters this itself (the old raw
    `WHERE model_id = ?` scan is gone) — it inherits the guarantee from the
    warm matrix's own build query, which is scoped to the CURRENT
    `model_tier` model id, so a stale row is simply invisible to a freshly
    built matrix."""
    from brain.memory import embedding_matrix as embedding_matrix_mod

    text = "Nell loves writing and spending time with Hana"

    old_provider = _NamedFakeProvider(model_id="fake-384-old")
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: old_provider)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, old_provider.model_id())
    _seed_embedded_memory(store, text)

    # Simulate a model swap: a NEW provider/model id, and a fresh matrix
    # build (a real process restart would do this naturally; here we reset
    # the process-wide matrix cache the same way the suite's autouse fixture
    # does between tests).
    embedding_matrix_mod._reset_embedding_matrix_cache()
    new_provider = _NamedFakeProvider(model_id="fake-384-new")
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: new_provider)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, new_provider.model_id())

    result = is_duplicate(text, store=store, threshold=DEFAULT_DEDUP_THRESHOLD)
    assert result is False


def test_is_duplicate_never_constructs_an_embeddingcache(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Structural guard: the new dedupe path must never construct an
    EmbeddingCache (the old embeddings.db content-hash cache) — neither to
    read existing vectors nor to write/cache the transient candidate's."""
    _use_384_fake_provider(monkeypatch)
    text = "Nell loves writing and spending time with Hana"
    _seed_embedded_memory(store, text)

    def _boom(*args, **kwargs):
        raise AssertionError("is_duplicate must never construct an EmbeddingCache")

    monkeypatch.setattr(embeddings_mod.EmbeddingCache, "__init__", _boom)

    # Both the duplicate and non-duplicate paths must complete without ever
    # hitting the patched constructor.
    assert is_duplicate(text, store=store, threshold=DEFAULT_DEDUP_THRESHOLD) is True
    assert is_duplicate("something entirely unrelated here", store=store) is False


def test_is_duplicate_fails_soft_on_matrix_error(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any failure building/reading the matrix (or embedding the candidate)
    is caught and logged; the item is let through rather than crashing the
    pipeline — same fail-soft posture as before the rewrite."""
    from brain.memory.embedding_matrix import EmbeddingMatrix

    _use_384_fake_provider(monkeypatch)

    def _boom(self):
        raise RuntimeError("simulated matrix build failure")

    monkeypatch.setattr(EmbeddingMatrix, "snapshot", _boom)

    assert is_duplicate("anything", store=store) is False
