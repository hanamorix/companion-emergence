"""Tests for brain.engines.dream — associative dream cycle."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain.bridge.chat import ChatMessage, ChatResponse
from brain.bridge.provider import FakeProvider, LLMProvider
from brain.engines.dream import DreamEngine, NoSeedAvailable
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore


class _PrefixedFakeProvider(LLMProvider):
    """Returns a response already starting with 'DREAM:' — exercises the
    engine's no-double-prefix branch.
    """

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "DREAM: already prefixed response"

    def name(self) -> str:
        return "fake-prefixed"

    def chat(self, messages: list[ChatMessage], *, tools=None, options=None) -> ChatResponse:  # type: ignore[override]
        return ChatResponse(content=self.generate(messages[-1].content), tool_calls=())


class _UnprefixedFakeProvider(LLMProvider):
    """Returns a response WITHOUT the DREAM prefix — exercises the engine's
    auto-prefix branch explicitly.
    """

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        return "just a raw thought with no prefix"

    def name(self) -> str:
        return "fake-unprefixed"

    def chat(self, messages: list[ChatMessage], *, tools=None, options=None) -> ChatResponse:  # type: ignore[override]
        return ChatResponse(content=self.generate(messages[-1].content), tool_calls=())


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
        persona_name="Nell",
        persona_system_prompt=(
            "You are Nell. You just woke from a dream about interconnected memories. "
            "Reflect in first person, 2-3 sentences, starting with 'DREAM: '. "
            "Be honest and specific, not abstract."
        ),
    )


def _mem(content: str, importance: float = 5.0, **kw: object) -> Memory:
    defaults: dict[str, object] = {"memory_type": "conversation", "domain": "us"}
    defaults.update(kw)
    m = Memory.create_new(content=content, **defaults)  # type: ignore[arg-type]
    m.importance = importance
    return m


def test_run_cycle_raises_if_no_seed_candidates(engine: DreamEngine) -> None:
    with pytest.raises(NoSeedAvailable):
        engine.run_cycle()


def test_run_cycle_picks_highest_importance_seed_in_window(
    engine: DreamEngine, store: MemoryStore
) -> None:
    low = _mem("low", importance=2.0)
    high = _mem("high", importance=8.0)
    store.create(low)
    store.create(high)

    result = engine.run_cycle()
    assert result.seed.id == high.id


def test_run_cycle_autoselects_production_ingest_memory_shape(
    engine: DreamEngine, store: MemoryStore
) -> None:
    """Dream seeds include extracted chat memories tagged by production ingest."""
    seed = Memory.create_new(
        content="Hana told me a meaningful thing in chat",
        memory_type="fact",
        domain="brain",
        tags=["auto_ingest", "conversation", "fact"],
        importance=9.0,
        metadata={"source_summary": "conversation:sess-prod"},
    )
    store.create(seed)

    result = engine.run_cycle()
    assert result.seed.id == seed.id


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

    result = engine.run_cycle()
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
    store: MemoryStore, hebbian: HebbianMatrix, tmp_path: Path
) -> None:
    """neighbour_limit is constructor-level calibration; build a fresh engine
    with the override and assert it propagates into the cycle."""
    capped_engine = DreamEngine(
        store=store,
        hebbian=hebbian,
        embeddings=None,
        provider=FakeProvider(),
        log_path=tmp_path / "dreams.log.jsonl",
        persona_name="Nell",
        persona_system_prompt="You are Nell. Reflect in first person.",
        neighbour_limit=3,
    )
    seed = _mem("seed", importance=8.0)
    store.create(seed)
    for i in range(10):
        n = _mem(f"n{i}")
        store.create(n)
        hebbian.strengthen(seed.id, n.id, delta=0.5)

    result = capped_engine.run_cycle(dry_run=True)
    assert len(result.neighbours) <= 3


def test_run_cycle_explicit_seed_not_found_raises(engine: DreamEngine) -> None:
    """When --seed id doesn't exist in the store, NoSeedAvailable surfaces.

    CLI users can pass stale ids; the error must be a clean engine-level
    exception, not a downstream AttributeError.
    """
    with pytest.raises(NoSeedAvailable, match="not found"):
        engine.run_cycle(seed_id="00000000-0000-0000-0000-000000000000")


def test_run_cycle_auto_prefixes_when_llm_omits_dream_prefix(
    store: MemoryStore, hebbian: HebbianMatrix, tmp_path: Path
) -> None:
    """If the LLM response doesn't start with 'DREAM:', the engine prepends it.

    Explicitly exercises the auto-prefix branch that FakeProvider's always-
    prefixed output skips over.
    """
    engine = DreamEngine(
        store=store,
        hebbian=hebbian,
        embeddings=None,
        provider=_UnprefixedFakeProvider(),
        log_path=tmp_path / "dreams.log.jsonl",
        persona_name="Nell",
        persona_system_prompt="You are Nell. Reflect in first person.",
    )
    seed = _mem("seed", importance=8.0)
    store.create(seed)

    result = engine.run_cycle()
    assert result.memory is not None
    assert result.memory.content == "DREAM: just a raw thought with no prefix"


def test_run_cycle_does_not_double_prefix_when_llm_already_prefixed(
    store: MemoryStore, hebbian: HebbianMatrix, tmp_path: Path
) -> None:
    """LLM response that already starts with 'DREAM:' is NOT double-prefixed."""
    engine = DreamEngine(
        store=store,
        hebbian=hebbian,
        embeddings=None,
        provider=_PrefixedFakeProvider(),
        log_path=tmp_path / "dreams.log.jsonl",
        persona_name="Nell",
        persona_system_prompt="You are Nell. Reflect in first person.",
    )
    seed = _mem("seed", importance=8.0)
    store.create(seed)

    result = engine.run_cycle()
    assert result.memory is not None
    assert result.memory.content == "DREAM: already prefixed response"
    assert not result.memory.content.startswith("DREAM: DREAM:")


def test_run_cycle_no_neighbours_branch_surfaces_in_prompt(
    engine: DreamEngine, store: MemoryStore
) -> None:
    """Seed with zero Hebbian edges uses the 'no neighbours' prompt branch
    and reports strengthened_edges=0 (no edges to reinforce).
    """
    seed = _mem("lonely seed", importance=8.0)
    store.create(seed)

    result = engine.run_cycle(dry_run=True)
    assert len(result.neighbours) == 0
    assert "No other memories resonated" in result.prompt


def test_dream_system_prompt_uses_persona_name(tmp_path: Path) -> None:
    """DreamEngine must render the persona name into its system prompt,
    not hardcode 'Nell'. Multi-persona correctness fix.
    """
    from brain.bridge.provider import FakeProvider
    from brain.engines.dream import DreamEngine
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import Memory, MemoryStore

    captured: dict[str, str | None] = {}

    class CapturingProvider(FakeProvider):
        def generate(self, prompt: str, *, system: str | None = None) -> str:
            captured["system"] = system
            return "DREAM: test"

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        store.create(
            Memory.create_new(
                content="seed",
                memory_type="conversation",
                domain="us",
                emotions={"love": 5.0},
            )
        )
        engine = DreamEngine(
            store=store,
            hebbian=hm,
            embeddings=None,
            provider=CapturingProvider(),
            persona_name="Iris",
            persona_system_prompt="You are Iris. Reflect in first person, 2-3 sentences, starting with 'DREAM: '.",
            lookback_hours=100000,
        )
        try:
            engine.run_cycle()
        except Exception:
            pass  # NoSeedAvailable may raise for incomplete setup; voice check still valid

    finally:
        store.close()
        hm.close()

    assert captured.get("system") is not None
    assert "Iris" in captured["system"]
    assert "Nell" not in captured["system"]


def test_dream_completion_emits_initiate_candidate(tmp_path: Path) -> None:
    """After a dream is logged, an initiate candidate is emitted.

    Phase 4.1 of the initiate physiology: the dream engine fires
    `emit_initiate_candidate` with `source="dream"` immediately after
    the log write. Wrapped in try/except inside the engine so an emit
    failure can't break the dream itself.
    """
    from brain.engines.dream import DreamEngine
    from brain.initiate.emit import read_candidates

    persona_dir = tmp_path / "p"
    persona_dir.mkdir()

    store = MemoryStore(db_path=":memory:")
    hm = HebbianMatrix(db_path=":memory:")
    try:
        seed = _mem("seed for initiate emit", importance=8.0)
        store.create(seed)

        engine = DreamEngine(
            store=store,
            hebbian=hm,
            embeddings=None,
            provider=FakeProvider(),
            log_path=persona_dir / "dreams.log.jsonl",
            persona_dir=persona_dir,
            persona_name="Nell",
            persona_system_prompt=(
                "You are Nell. Reflect in first person, 2-3 sentences, "
                "starting with 'DREAM: '."
            ),
        )
        engine.run_cycle()
    finally:
        store.close()
        hm.close()

    candidates = read_candidates(persona_dir)
    assert len(candidates) == 1
    assert candidates[0].source == "dream"
    assert candidates[0].kind == "message"


def test_dream_engine_empty_persona_raises() -> None:
    """DreamEngine must reject empty persona_name / persona_system_prompt
    to force callers to be explicit."""
    import pytest

    from brain.bridge.provider import FakeProvider
    from brain.engines.dream import DreamEngine
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    store = MemoryStore(":memory:")
    hm = HebbianMatrix(":memory:")
    try:
        with pytest.raises(ValueError, match="persona_name"):
            DreamEngine(
                store=store,
                hebbian=hm,
                embeddings=None,
                provider=FakeProvider(),
                # persona_name omitted → should raise
            )
    finally:
        store.close()
        hm.close()
