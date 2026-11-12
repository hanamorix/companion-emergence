"""Pins of today's emotion-writer bugs, for the emotion rework (#240, work item A1).

Each bug has a `test_today_*` that passes now and a `test_intended_*` marked xfail(strict), so the PR that fixes the bug
has to remove the marker and rewrite the pin. "The felt-state read" is `aggregate_state` today; the B1 PRs switch to the
engine's read with the switch on.

The design's sixth row (spelling variants pooling as separate channels) has no pin here: #565 fixed it, and
`test_aggregate.py` covers it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain import dev_constants
from brain.bridge.provider import FakeProvider
from brain.chat.extractor import _apply_emotion_delta
from brain.emotion import engine, engine_files
from brain.emotion.aggregate import aggregate_state
from brain.emotion.engine import EmotionEngine, Reading
from brain.engines.consolidation import Decision, run_consolidation
from brain.engines.dream import DreamEngine
from brain.engines.heartbeat import HeartbeatEngine, HeartbeatState
from brain.kindled_link.relationship import apply_peer_emotion, write_kindled_peer_memory
from brain.kindled_link.store import KindledLinkStore
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.paths import get_persona_dir
from brain.self_model.reconcile import reconcile_self_read

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)
CAP = dev_constants.KINDLED_LINK_PEER_EMOTION_WINDOW_CAP
WINDOW_HOURS = dev_constants.KINDLED_LINK_PEER_EMOTION_WINDOW_HOURS


_PINNED = pytest.mark.no_base_failure(reason="pins a writer bug that is on main until a B1 PR fixes it, so the old code cannot fail it")


def _flipped_by(pr: str) -> Callable[[Callable[..., None]], Callable[..., None]]:
    # raises=AssertionError: a broken setup must fail loudly, not pass as the expected failure.
    xfail = pytest.mark.xfail(strict=True, raises=AssertionError, reason=f"writer bug, fixed by {pr}")
    return lambda test: _PINNED(xfail(test))


@pytest.fixture
def persona_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("KINDLED_HOME", str(tmp_path / "home"))
    path = get_persona_dir("pins")
    path.mkdir(parents=True)
    return path


@pytest.fixture
def store(persona_dir: Path):
    memory_store = MemoryStore(persona_dir / "memories.db")
    yield memory_store
    memory_store.close()


@pytest.fixture
def hebbian():
    matrix = HebbianMatrix(db_path=":memory:")
    yield matrix
    matrix.close()


def _felt(store: MemoryStore) -> dict[str, float]:
    return aggregate_state(store.list_active()).emotions


def _go_live(persona_dir: Path, store: MemoryStore, **seed: float) -> EmotionEngine:
    """A live persona with no state file applies nothing, so seed a memory first, then write the switch."""
    store.create(Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions=seed or {"joy": 5.0}))
    assert engine_files.write_switch(persona_dir, datetime.now(UTC))
    return engine.engine_for(persona_dir)


def _settled(eng: EmotionEngine) -> dict[str, Reading]:
    eng.note_active(datetime.now(UTC))
    return eng.read()


def _seed(store: MemoryStore, emotions: dict[str, float]) -> Memory:
    memory = Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions=emotions)
    store.create(memory)
    return memory


def _one_heartbeat(store: MemoryStore, hebbian: HebbianMatrix, persona_dir: Path) -> None:
    engine = HeartbeatEngine(
        store=store,
        hebbian=hebbian,
        provider=FakeProvider(),
        state_path=persona_dir / "heartbeat_state.json",
        config_path=persona_dir / "heartbeat_config.json",
        dream_log_path=persona_dir / "dreams.log.jsonl",
        heartbeat_log_path=persona_dir / "heartbeats.log.jsonl",
        persona_name="Pins",
        persona_system_prompt="You are Pins.",
    )
    engine._apply_emotion_decay(3600.0, tick_at=NOW, state=HeartbeatState.fresh(trigger="open"), dry_run=False)


def _write_dream_over_ten(
    store: MemoryStore, hebbian: HebbianMatrix, persona_dir: Path, *, eng: EmotionEngine | None
) -> tuple[Memory, Memory]:
    """Dream on a seed and two neighbours, each love=6.0, and promote it past the gate. Returns the dream and the seed."""
    seed = _seed(store, {"love": 6.0})
    neighbours = [(_seed(store, {"love": 6.0}), 1.0), (_seed(store, {"love": 6.0}), 1.0)]
    engine = DreamEngine(
        store=store,
        hebbian=hebbian,
        provider=FakeProvider(),
        log_path=persona_dir / "dreams.log.jsonl",
        persona_name="Pins",
        persona_system_prompt="You are Pins.",
    )
    dream = engine._write_dream_memory(seed, neighbours, "DREAM: a warm room", eng=eng)
    run_consolidation(store, persona_dir=persona_dir, classifier=lambda _candidate, _context: Decision("new"))
    return dream, seed


def _write_negative_peer_memory(store: MemoryStore) -> str:
    """Write a peer memory with one negative value and one ordinary value. Returns its id."""
    write_kindled_peer_memory(
        store,
        peer_id="kid_a",
        session_id="s1",
        speaker="peer",
        stage="acquaintance",
        content="a peer said something",
        emotions={"tenderness": -0.3, "joy": 5.0},
    )
    return next(m.id for m in store.list_active() if m.memory_type == "kindled_peer")


def _peer_changes(persona_dir: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Apply two peer changes an hour apart, the second to a different emotion. Returns what each one applied."""
    link_store = KindledLinkStore(persona_dir / "kindled_link.db")
    try:
        first = apply_peer_emotion(link_store, "kid_a", {"tenderness": 0.5, "joy": 0.5}, NOW)
        second = apply_peer_emotion(link_store, "kid_a", {"love": 0.5}, NOW + timedelta(hours=1))
    finally:
        link_store.close()
    return first, second


@_PINNED
def test_today_extractor_negative_delta_raises_the_feeling(persona_dir: Path, store: MemoryStore) -> None:
    assert "joy" not in _felt(store)
    _apply_emotion_delta({"joy": -0.2}, persona_dir)
    assert _felt(store)["joy"] == pytest.approx(2.0)


@_flipped_by("B1-1")
def test_intended_extractor_negative_delta_lowers_the_feeling(persona_dir: Path, store: MemoryStore) -> None:
    _seed(store, {"joy": 5.0})
    _apply_emotion_delta({"joy": -0.2}, persona_dir)
    assert _felt(store)["joy"] < 5.0


@_PINNED
def test_today_reconcile_negative_delta_raises_the_feeling(persona_dir: Path, store: MemoryStore) -> None:
    assert "joy" not in _felt(store)
    result = reconcile_self_read(persona_dir=persona_dir, action="revise", channel="joy", delta=-0.2)
    assert result["delta_written"] is True
    assert _felt(store)["joy"] == pytest.approx(2.0)


@_flipped_by("B1-1")
def test_intended_reconcile_negative_delta_lowers_the_feeling(persona_dir: Path, store: MemoryStore) -> None:
    _seed(store, {"joy": 5.0})
    reconcile_self_read(persona_dir=persona_dir, action="revise", channel="joy", delta=-0.2)
    assert _felt(store)["joy"] < 5.0


@_PINNED
def test_today_dream_over_ten_is_erased_by_one_heartbeat(
    persona_dir: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    dream, seed = _write_dream_over_ten(store, hebbian, persona_dir, eng=None)
    assert store.get(dream.id).emotions == {"love": pytest.approx(18.0)}
    _one_heartbeat(store, hebbian, persona_dir)
    assert "love" not in store.get(dream.id).emotions
    # The seed goes through the same pass and keeps its love: it is the size that erases, not the time.
    assert "love" in store.get(seed.id).emotions


def test_intended_dream_value_stays_in_range(persona_dir: Path, store: MemoryStore, hebbian: HebbianMatrix) -> None:
    eng = _go_live(persona_dir, store)
    dream, _seed_memory = _write_dream_over_ten(store, hebbian, persona_dir, eng=eng)
    assert store.get(dream.id).emotions["love"] <= 10.0


@_PINNED
def test_today_negative_peer_value_is_erased_by_one_heartbeat(
    persona_dir: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    memory_id = _write_negative_peer_memory(store)
    assert store.get(memory_id).emotions["tenderness"] == pytest.approx(-0.3)
    _one_heartbeat(store, hebbian, persona_dir)
    emotions = store.get(memory_id).emotions
    assert "tenderness" not in emotions
    # The ordinary value in the same row survives: it is the sign that erases, not the time.
    assert "joy" in emotions


@_flipped_by("B1-2")
def test_intended_peer_values_are_never_negative(store: MemoryStore) -> None:
    memory_id = _write_negative_peer_memory(store)
    assert all(value >= 0.0 for value in store.get(memory_id).emotions.values())


@_PINNED
def test_today_peer_bucket_caps_summed_size_across_emotions(persona_dir: Path) -> None:
    first, second = _peer_changes(persona_dir)
    # One change is scaled down to the cap across both emotions.
    assert first == {"tenderness": pytest.approx(CAP / 2), "joy": pytest.approx(CAP / 2)}
    # A different emotion gets no bucket of its own: only what drained in that hour is free.
    assert second == {"love": pytest.approx(CAP / WINDOW_HOURS)}


@_flipped_by("B1-2")
def test_intended_peer_bucket_does_not_scale(persona_dir: Path) -> None:
    first, second = _peer_changes(persona_dir)
    assert first == {"tenderness": 0.5, "joy": 0.5}
    assert second == {"love": 0.5}
