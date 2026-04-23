"""Tests for brain.engines.dream — associative dream cycle."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.provider import FakeProvider
from brain.engines.dream import DreamEngine, NoSeedAvailable
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore(db_path=":memory:")


@pytest.fixture
def hebbian() -> HebbianMatrix:
    return HebbianMatrix(db_path=":memory:")


@pytest.fixture
def engine(store: MemoryStore, hebbian: HebbianMatrix, tmp_path: Path) -> DreamEngine:
    return DreamEngine(
        store=store,
        hebbian=hebbian,
        embeddings=None,
        provider=FakeProvider(),
        log_path=tmp_path / "dreams.log.jsonl",
    )


def _mem(content: str, importance: float = 5.0, **kw: object) -> Memory:
    defaults: dict[str, object] = {"memory_type": "conversation", "domain": "us"}
    defaults.update(kw)
    m = Memory.create_new(content=content, **defaults)  # type: ignore[arg-type]
    m.importance = importance
    return m


def test_run_cycle_raises_if_no_seed_candidates(engine: DreamEngine) -> None:
    with pytest.raises(NoSeedAvailable):
        engine.run_cycle(lookback_hours=24)


def test_run_cycle_picks_highest_importance_seed_in_window(
    engine: DreamEngine, store: MemoryStore
) -> None:
    low = _mem("low", importance=2.0)
    high = _mem("high", importance=8.0)
    store.create(low)
    store.create(high)

    result = engine.run_cycle()
    assert result.seed.id == high.id


def test_run_cycle_explicit_seed_overrides_autoselect(
    engine: DreamEngine, store: MemoryStore
) -> None:
    m1 = _mem("low", importance=2.0)
    m2 = _mem("explicit", importance=1.0)
    store.create(m1)
    store.create(m2)

    result = engine.run_cycle(seed_id=m2.id)
    assert result.seed.id == m2.id


def test_run_cycle_writes_dream_memory_to_store(engine: DreamEngine, store: MemoryStore) -> None:
    seed = _mem("seed content", importance=8.0)
    store.create(seed)

    result = engine.run_cycle()
    assert result.memory is not None
    assert result.memory.memory_type == "dream"

    restored = store.get(result.memory.id)
    assert restored is not None
    assert restored.memory_type == "dream"


def test_run_cycle_dream_content_starts_with_dream_prefix(
    engine: DreamEngine, store: MemoryStore
) -> None:
    seed = _mem("seed", importance=8.0)
    store.create(seed)

    result = engine.run_cycle()
    assert result.memory is not None
    assert result.memory.content.startswith("DREAM:")


def test_run_cycle_metadata_includes_seed_and_activated_ids(
    engine: DreamEngine, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    seed = _mem("seed", importance=8.0)
    neighbour = _mem("neighbour", importance=4.0)
    store.create(seed)
    store.create(neighbour)
    hebbian.strengthen(seed.id, neighbour.id, delta=0.7)

    result = engine.run_cycle()
    assert result.memory is not None
    md = result.memory.metadata
    assert md["seed_id"] == seed.id
    assert neighbour.id in md["activated"]
    assert md["provider"] == "fake"


def test_run_cycle_strengthens_edges_to_each_activated_neighbour(
    engine: DreamEngine, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    seed = _mem("seed", importance=8.0)
    n1 = _mem("n1", importance=4.0)
    n2 = _mem("n2", importance=4.0)
    for m in (seed, n1, n2):
        store.create(m)
    hebbian.strengthen(seed.id, n1.id, delta=0.5)
    hebbian.strengthen(seed.id, n2.id, delta=0.5)

    before_n1 = hebbian.weight(seed.id, n1.id)
    before_n2 = hebbian.weight(seed.id, n2.id)

    result = engine.run_cycle()
    assert result.strengthened_edges == 2

    assert hebbian.weight(seed.id, n1.id) > before_n1
    assert hebbian.weight(seed.id, n2.id) > before_n2


def test_run_cycle_dry_run_returns_result_without_writes(
    engine: DreamEngine, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    seed = _mem("seed", importance=8.0)
    neighbour = _mem("neighbour", importance=4.0)
    store.create(seed)
    store.create(neighbour)
    hebbian.strengthen(seed.id, neighbour.id, delta=0.5)

    count_before = store.count()
    weight_before = hebbian.weight(seed.id, neighbour.id)

    result = engine.run_cycle(dry_run=True)

    assert result.memory is None
    assert result.dream_text is None
    assert result.strengthened_edges == 0
    assert store.count() == count_before
    assert hebbian.weight(seed.id, neighbour.id) == weight_before


def test_run_cycle_dry_run_populates_seed_neighbours_and_prompt(
    engine: DreamEngine, store: MemoryStore
) -> None:
    seed = _mem("seed", importance=8.0)
    store.create(seed)

    result = engine.run_cycle(dry_run=True)
    assert result.seed.id == seed.id
    assert result.prompt != ""
    assert result.system_prompt != ""


def test_run_cycle_respects_lookback_window(engine: DreamEngine, store: MemoryStore) -> None:
    old = _mem("old", importance=9.0)
    old.created_at = datetime.now(UTC) - timedelta(hours=48)
    store.create(old)
    recent = _mem("recent", importance=2.0)
    store.create(recent)

    result = engine.run_cycle(lookback_hours=24)
    assert result.seed.id == recent.id


def test_run_cycle_appends_to_dreams_log(
    engine: DreamEngine, store: MemoryStore, tmp_path: Path
) -> None:
    seed = _mem("seed", importance=8.0)
    store.create(seed)

    engine.run_cycle()

    log_path = tmp_path / "dreams.log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["seed_id"] == seed.id
    assert entry["provider"] == "fake"
    assert "timestamp" in entry
    assert "dream_id" in entry


def test_run_cycle_dry_run_does_not_log(
    engine: DreamEngine, store: MemoryStore, tmp_path: Path
) -> None:
    seed = _mem("seed", importance=8.0)
    store.create(seed)

    engine.run_cycle(dry_run=True)
    log_path = tmp_path / "dreams.log.jsonl"
    assert not log_path.exists() or log_path.read_text() == ""


def test_run_cycle_prompt_contains_seed_and_neighbours(
    engine: DreamEngine, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    seed = _mem("the seed thought", importance=8.0)
    neighbour = _mem("the neighbour thought", importance=4.0)
    store.create(seed)
    store.create(neighbour)
    hebbian.strengthen(seed.id, neighbour.id, delta=0.5)

    result = engine.run_cycle(dry_run=True)
    assert "the seed thought" in result.prompt
    assert "the neighbour thought" in result.prompt


def test_run_cycle_system_prompt_mentions_nell_and_dream_prefix(
    engine: DreamEngine, store: MemoryStore
) -> None:
    seed = _mem("seed", importance=8.0)
    store.create(seed)

    result = engine.run_cycle(dry_run=True)
    assert "Nell" in result.system_prompt
    assert "DREAM:" in result.system_prompt


def test_run_cycle_respects_neighbour_limit(
    engine: DreamEngine, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    seed = _mem("seed", importance=8.0)
    store.create(seed)
    for i in range(10):
        n = _mem(f"n{i}")
        store.create(n)
        hebbian.strengthen(seed.id, n.id, delta=0.5)

    result = engine.run_cycle(dry_run=True, neighbour_limit=3)
    assert len(result.neighbours) <= 3
