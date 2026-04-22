"""Embedding provider abstraction + content-hash cache.

Provider interface: EmbeddingProvider ABC. Two concrete providers:
- FakeEmbeddingProvider: deterministic hash-based, zero network, used in tests.
- OllamaEmbeddingProvider: calls local Ollama /api/embeddings endpoint
  (will be added in Week 5 when the bridge lands).

Cache: EmbeddingCache layers a SQLite content-hash cache on top of any
provider. `get_or_compute(content)` returns the vector, hitting cache on
repeat calls. Content hashed via SHA-256; first 32 hex chars used as key.

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
    """Abstract embedding provider. Subclasses implement `embed` and `embedding_dim`."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a 1-D numpy array of dimension `embedding_dim()`."""

    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the output dimension of vectors this provider produces."""


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


class EmbeddingCache:
    """Content-hash cache on top of any EmbeddingProvider.

    Storage: SQLite table with (content_hash TEXT PRIMARY KEY, vector BLOB,
    dim INTEGER, created_at TEXT). Hash is SHA-256 hex (first 32 chars).
    Vector stored as raw float32 bytes via np.ndarray.tobytes().
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS embedding_cache (
        content_hash TEXT PRIMARY KEY,
        vector BLOB NOT NULL,
        dim INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """

    def __init__(self, db_path: str | Path, provider: EmbeddingProvider) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.executescript(self._SCHEMA)
        self._conn.commit()
        self._provider = provider

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    def get_or_compute(self, content: str) -> np.ndarray:
        """Return the cached embedding for content, computing + storing on miss."""
        key = self._hash(content)
        row = self._conn.execute(
            "SELECT vector, dim FROM embedding_cache WHERE content_hash = ?", (key,)
        ).fetchone()
        if row is not None:
            return np.frombuffer(row[0], dtype=np.float32).copy().reshape(row[1])

        vec = self._provider.embed(content).astype(np.float32)
        self._conn.execute(
            "INSERT INTO embedding_cache (content_hash, vector, dim) VALUES (?, ?, ?)",
            (key, vec.tobytes(), vec.shape[0]),
        )
        self._conn.commit()
        # Return a float32 copy for consistency with cache hits.
        return vec.copy()

    def count(self) -> int:
        """Return the number of cached embeddings."""
        return int(self._conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0])

    @staticmethod
    def _hash(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two vectors. Range [-1, 1]."""
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
