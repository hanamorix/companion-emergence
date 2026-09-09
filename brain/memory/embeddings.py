"""Embedding provider abstraction + content-hash cache.

Provider interface: EmbeddingProvider ABC. Two concrete providers:
- FakeEmbeddingProvider: deterministic hash-based, zero network, used in tests.
- FastEmbedProvider: real local embeddings via `fastembed` (ONNX, no torch,
  no network at inference — the model file is downloaded once into the
  shared cache dir and used offline after). Production default.

Cache: EmbeddingCache layers a SQLite content-hash cache on top of any
provider. `get_or_compute(content)` returns the vector, hitting cache on
repeat calls. Content hashed via SHA-256; first 32 hex chars used as key.
Cache rows also carry a `model_id` — the id of the model that produced the
vector — so swapping providers (e.g. FakeEmbeddingProvider → a real model,
or one real model → another) is a targeted invalidation instead of silently
serving a vector some other model made. `get_or_compute` only ever considers
rows whose `model_id` matches the cache's own provider.

Design per spec Section 4.1 (brain/memory/embeddings.py) and Section 10.1
(content-hash embedding cache).
"""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

_DEFAULT_DIM = 256


class EmbeddingProvider(ABC):
    """Abstract embedding provider. Subclasses implement `embed`, `embedding_dim`
    and `model_id`."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a 1-D numpy array of dimension `embedding_dim()`."""

    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the output dimension of vectors this provider produces."""

    @abstractmethod
    def model_id(self) -> str:
        """Return a stable identifier for the model producing these vectors.

        Stored alongside every cached vector (`embedding_cache.model_id`) so
        a provider swap is a targeted cache invalidation — a vector made by
        one model/dim is never read back as if it came from another. Two
        providers that produce incompatible vectors MUST return different
        ids (dimension alone is not a safe proxy: two different models can
        share a dimension).
        """


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic pseudo-random embedding provider for tests.

    Uses SHA-256 of the input text to seed a NumPy Generator, then produces
    a unit-norm vector. Same text always produces the same vector; different
    text produces different vectors. No network, no external dependencies.
    """

    def __init__(self, dim: int = _DEFAULT_DIM) -> None:
        self._dim = dim

    def embed(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(seed=seed)
        vec = rng.standard_normal(self._dim)
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            raise ValueError(f"FakeEmbeddingProvider produced a zero-norm vector (dim={self._dim})")
        return vec / norm

    def embedding_dim(self) -> int:
        return self._dim

    def model_id(self) -> str:
        # Dim-qualified so two Fake instances of different dims (seen across
        # the test suite) never collide on the same cache rows.
        return f"fake-{self._dim}"


class FastEmbedProvider(EmbeddingProvider):
    """Real local embedding provider via `fastembed` (ONNX runtime, no torch).

    Production default. Model id comes from `model_tier.py`
    (`model_for_tier(TIER_EMBEDDING)`), never hardcoded here — see that
    module's docstring for why every model selection routes through it.

    The model file is downloaded once (fastembed's own lazy-download-on-first-
    use behavior) into `cache_dir` and used fully offline after — no network
    call happens at embed() time once the file is cached. Construction itself
    does NOT download; the download is deferred to fastembed's own internals
    on first `embed()` call, same as fastembed's default behavior.
    """

    def __init__(self, model_id: str, cache_dir: str | Path, dim: int) -> None:
        # Imported lazily so importing this module never requires fastembed/
        # onnxruntime to be installed unless the real provider is actually
        # constructed (tests exclusively use FakeEmbeddingProvider).
        from fastembed import TextEmbedding

        self._model_id = model_id
        self._dim = dim
        # lazy_load=True: defer the (one-time) model-file load/download to
        # the first embed() call rather than construction time. Every current
        # call site constructs this off the message hot path already, but
        # deferring keeps construction itself cheap and never network-bound.
        self._model = TextEmbedding(model_name=model_id, cache_dir=str(cache_dir), lazy_load=True)

    def embed(self, text: str) -> np.ndarray:
        # TextEmbedding.embed() takes an iterable and yields one vector per
        # input; we pass exactly one string and take the one result.
        (vec,) = self._model.embed([text])
        return np.asarray(vec, dtype=np.float32)

    def embedding_dim(self) -> int:
        return self._dim

    def model_id(self) -> str:
        return self._model_id


class EmbeddingCache:
    """Content-hash cache on top of any EmbeddingProvider.

    Storage: SQLite table with (content_hash TEXT PRIMARY KEY, vector BLOB,
    dim INTEGER, model_id TEXT, created_at TEXT). Hash is SHA-256 hex (first
    32 chars). Vector stored as raw float32 bytes via np.ndarray.tobytes().
    model_id is the producing provider's id (see EmbeddingProvider.model_id);
    every read/write here is scoped to it.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS embedding_cache (
        content_hash TEXT PRIMARY KEY,
        vector BLOB NOT NULL,
        dim INTEGER NOT NULL,
        model_id TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """

    def __init__(self, db_path: str | Path, provider: EmbeddingProvider) -> None:
        self._conn = sqlite3.connect(str(db_path))
        # WAL + 5s busy_timeout — supervisor opens this cache from a
        # background thread; without WAL, any concurrent reader/writer
        # surfaces as `database is locked`. In-memory dbs reject WAL;
        # fallback keeps `:memory:` working in tests.
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.OperationalError:
            pass
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(self._SCHEMA)
        # Idempotent column migration for personas/dbs created before the
        # model_id column existed — CREATE TABLE IF NOT EXISTS above leaves a
        # pre-existing table alone, so check + ALTER, mirroring the
        # recall_count ALTER-guard pattern in store.py.
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(embedding_cache)").fetchall()}
        if "model_id" not in existing:
            self._conn.execute(
                "ALTER TABLE embedding_cache ADD COLUMN model_id TEXT NOT NULL DEFAULT ''"
            )
        self._conn.commit()
        self._provider = provider
        self._model_id = provider.model_id()

    @property
    def model_id(self) -> str:
        """The model id this cache's provider produces — every read/write
        here is scoped to rows carrying this id."""
        return self._model_id

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def get_or_compute(self, content: str) -> np.ndarray:
        """Return the cached embedding for content, computing + storing on miss.

        Cache rows are keyed by (content_hash, model_id) — a row written by a
        DIFFERENT provider (e.g. FakeEmbeddingProvider's 256-dim vectors vs a
        real 384-dim model) is never returned; a miss on model_id mismatch
        recomputes and overwrites the row with this provider's vector (a
        stale row from a prior model is targeted, lazy invalidation, not a
        silent dim mismatch).
        """
        key = self._hash(content)
        row = self._conn.execute(
            "SELECT vector, dim FROM embedding_cache WHERE content_hash = ? AND model_id = ?",
            (key, self._model_id),
        ).fetchone()
        if row is not None:
            return np.frombuffer(row[0], dtype=np.float32).copy().reshape(row[1])

        vec = self._provider.embed(content).astype(np.float32)
        self._conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (content_hash, vector, dim, model_id) "
            "VALUES (?, ?, ?, ?)",
            (key, vec.tobytes(), vec.shape[0], self._model_id),
        )
        self._conn.commit()
        # Return a float32 copy for consistency with cache hits.
        return vec.copy()

    def count(self) -> int:
        """Return the number of cached embeddings."""
        return int(self._conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0])

    def evict(self, content: str) -> None:
        """Remove a cached vector (by content hash).

        Used to undo a dedupe-compute when the memory failed to commit, so
        that a retry pass isn't dropped as a self-duplicate (the vector would
        otherwise sit in the cache and match at cosine 1.0 on the next call
        to is_duplicate, which snapshots existing rows before computing the
        candidate).
        """
        self._conn.execute(
            "DELETE FROM embedding_cache WHERE content_hash = ?", (self._hash(content),)
        )
        self._conn.commit()

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two vectors. Range [-1, 1]."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_embedding_provider() -> EmbeddingProvider:
    """The production embedding provider: FastEmbedProvider pinned to
    `model_tier.TIER_EMBEDDING`'s model id, caching the model file in the
    shared `get_cache_dir()` (one download across every persona on the box,
    per the build-plan recommendation — the model isn't persona-specific
    data, just a local asset).

    The model id/dim come from `model_tier.py`, never hardcoded here — same
    convention as every Claude tier in that module.
    """
    from brain.bridge.model_tier import MODEL_EMBEDDING_DIM, TIER_EMBEDDING, model_for_tier
    from brain.paths import get_cache_dir

    return FastEmbedProvider(
        model_id=model_for_tier(TIER_EMBEDDING),
        cache_dir=get_cache_dir(),
        dim=MODEL_EMBEDDING_DIM,
    )


def build_embedding_cache(persona_dir: str | Path) -> EmbeddingCache:
    """The production EmbeddingCache for a persona: `embeddings.db` under
    `persona_dir`, backed by `build_embedding_provider()`.

    ONE construction helper instead of every call site repeating
    `EmbeddingCache(persona_dir / "embeddings.db", FakeEmbeddingProvider(...))`
    — centralizes the production provider choice so a future model swap (or
    provider change) is a one-function edit, not an N-call-site hunt.
    Tests that need a cache under the fake provider construct EmbeddingCache
    directly with FakeEmbeddingProvider, as before.
    """
    return EmbeddingCache(Path(persona_dir) / "embeddings.db", build_embedding_provider())
