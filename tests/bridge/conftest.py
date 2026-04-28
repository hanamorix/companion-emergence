"""Shared fixtures for bridge tests."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from brain.memory.embeddings import EmbeddingCache, FakeEmbeddingProvider
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import MemoryStore


@pytest.fixture
def persona_dir(tmp_path: Path) -> Path:
    """A fresh empty persona directory for one test."""
    p = tmp_path / "test-persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    # Minimal persona_config.json so build_app's lifespan can resolve a provider
    # in tests without touching the real Claude CLI.
    (p / "persona_config.json").write_text('{"provider": "fake", "searcher": "fake"}')
    return p


@pytest.fixture(autouse=True)
def _reset_session_registry():
    """Clear the in-memory session registry between bridge tests."""
    from brain.chat.session import reset_registry
    reset_registry()
    yield
    reset_registry()


@pytest.fixture
def store(persona_dir: Path) -> Iterator[MemoryStore]:
    """An open MemoryStore against an in-tmp SQLite db."""
    s = MemoryStore(persona_dir / "memories.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def hebbian(persona_dir: Path) -> Iterator[HebbianMatrix]:
    """An open HebbianMatrix against an in-tmp SQLite db."""
    h = HebbianMatrix(persona_dir / "hebbian.db")
    try:
        yield h
    finally:
        h.close()


@pytest.fixture
def embeddings(persona_dir: Path) -> Iterator[EmbeddingCache]:
    """An open in-memory EmbeddingCache with FakeEmbeddingProvider."""
    provider = FakeEmbeddingProvider(dim=256)
    e = EmbeddingCache(persona_dir / "embeddings.db", provider)
    try:
        yield e
    finally:
        e.close()
