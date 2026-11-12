"""Pins of today's emotion-writer bugs, for the emotion rework (#240, work item A1).

Each bug has a `test_today_*` that passes now and a `test_intended_*` marked xfail(strict), so the PR that fixes the bug
has to remove the marker and rewrite the pin. "The felt-state read" is `aggregate_state` today; the B1 PRs switch to the
engine's read with the switch on.

The design's sixth row (spelling variants pooling as separate channels) has no pin here: #565 fixed it, and
`test_aggregate.py` covers it.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from brain import dev_constants
from brain.bridge.provider import FakeProvider
from brain.chat.extractor import _apply_emotion_delta
from brain.emotion import engine
from brain.emotion.engine import EmotionEngine, Reading
from brain.engines.consolidation import Decision, run_consolidation
from brain.engines.dream import DreamEngine
from brain.engines.heartbeat import HeartbeatConfig, HeartbeatEngine, HeartbeatState
from brain.kindled_link.relationship import (
    apply_peer_emotion,
    run_relationship_reflection,
    write_kindled_peer_memory,
)
from brain.kindled_link.store import KindledLinkStore
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.paths import get_persona_dir
from brain.self_model.reconcile import reconcile_self_read
from tests.unit.brain.emotion._engine_support import make_live

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)
CAP = dev_constants.KINDLED_LINK_PEER_EMOTION_WINDOW_CAP
WINDOW_HOURS = dev_constants.KINDLED_LINK_PEER_EMOTION_WINDOW_HOURS


_PINNED = pytest.mark.no_base_failure(reason="pins a writer bug that is on main until a B1 PR fixes it, so the old code cannot fail it")


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


def _go_live(persona_dir: Path, store: MemoryStore, **seed: float) -> EmotionEngine:
    """A live persona with no state file applies nothing, so seed a memory first, then record the switch-over."""
    store.create(Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions=seed or {"joy": 5.0}))
    make_live(persona_dir, datetime.now(UTC))
    return engine.engine_for(persona_dir)


def _settled(eng: EmotionEngine) -> dict[str, Reading]:
    eng.note_active(datetime.now(UTC))
    return eng.read()


def _seed(store: MemoryStore, emotions: dict[str, float]) -> Memory:
    memory = Memory.create_new(content="seed", memory_type="conversation", domain="us", emotions=emotions)
    store.create(memory)
    return memory


@pytest.fixture(autouse=True)
def _fake_tier_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """A heartbeat tick routes its consolidation gate through the tier accessor; never build a real provider here."""
    monkeypatch.setattr("brain.engines.heartbeat.build_tier_provider", lambda persona_dir, tier: FakeProvider())


def _one_heartbeat(store: MemoryStore, hebbian: HebbianMatrix, persona_dir: Path) -> None:
    """Run one real heartbeat tick an hour after the last one, with reflex, research, growth and the heartbeat memory off."""
    config_path = persona_dir / "heartbeat_config.json"
    HeartbeatConfig(reflex_enabled=False, research_enabled=False, growth_enabled=False, emit_memory="never").save(config_path)
    state_path = persona_dir / "heartbeat_state.json"
    prior = HeartbeatState.fresh(trigger="open")
    prior.last_tick_at = datetime.now(UTC) - timedelta(hours=1)
    prior.save(state_path)
    engine = HeartbeatEngine(
        store=store,
        hebbian=hebbian,
        provider=FakeProvider(),
        state_path=state_path,
        config_path=config_path,
        dream_log_path=persona_dir / "dreams.log.jsonl",
        heartbeat_log_path=persona_dir / "heartbeats.log.jsonl",
        persona_name="Pins",
        persona_system_prompt="You are Pins.",
    )
    result = engine.run_tick(trigger="open", dry_run=False)
    assert not result.initialized and result.skipped_reason is None, "the helper must run a full tick"


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


class _GrantedSlot:
    @contextlib.contextmanager
    def background_slot(self, *, now=None):  # noqa: ANN001, ANN202 - the throttle's shape
        yield True

    def should_yield(self, *, now=None) -> bool:  # noqa: ANN001
        return False


class _Reply:
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, prompt: str) -> str:
        return self.reply


def _live_peer_memories(persona_dir: Path, store: MemoryStore, *emotions_json: str) -> list[dict[str, float]]:
    """Run one live relationship reflection per reply, an hour apart. Returns each peer memory's stored emotions, in order."""
    _go_live(persona_dir, store)
    link_store = KindledLinkStore(persona_dir / "kindled_link.db")
    try:
        for hour, emotion in enumerate(emotions_json):
            at = NOW + timedelta(hours=hour)
            reply = (
                '{"proposed_stage":"stranger","trust_score":0.1,"affinity_tags":[],"boundaries_seen":[],"evidence":[],'
                '"hard_breach":false,"memory_summary":"I talked with a peer.","emotion":' + emotion + "}"
            )
            run_relationship_reflection(
                store=link_store, provider=_Reply(reply), peer_id="kid_a", transcript="peer: hello there, friend",
                now=at, today=at.strftime("%Y-%m-%d"), throttle=_GrantedSlot(), persona_dir=persona_dir,
                mem_store=store, persona_name="Pins", session_id="s1",
            )
    finally:
        link_store.close()
    return [m._stored_emotions for m in sorted(store.list_by_type("kindled_peer"), key=lambda m: m.created_at)]


def _peer_changes(persona_dir: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Apply two peer changes an hour apart, the second to a different emotion. Returns what each one applied."""
    link_store = KindledLinkStore(persona_dir / "kindled_link.db")
    try:
        first = apply_peer_emotion(link_store, "kid_a", {"tenderness": 0.5, "joy": 0.5}, NOW)
        second = apply_peer_emotion(link_store, "kid_a", {"love": 0.5}, NOW + timedelta(hours=1))
    finally:
        link_store.close()
    return first, second


def test_intended_extractor_negative_delta_lowers_the_feeling(persona_dir: Path, store: MemoryStore) -> None:
    eng = _go_live(persona_dir, store, joy=5.0)
    before = _settled(eng)["joy"].short
    _apply_emotion_delta({"joy": -0.2}, persona_dir)
    assert _settled(eng)["joy"].short < before


def test_intended_reconcile_negative_delta_lowers_the_feeling(persona_dir: Path, store: MemoryStore) -> None:
    eng = _go_live(persona_dir, store, joy=5.0)
    before = _settled(eng)["joy"].short
    reconcile_self_read(persona_dir=persona_dir, action="revise", channel="joy", delta=-0.2)
    assert _settled(eng)["joy"].short < before


def test_today_dream_over_ten_is_erased_by_one_heartbeat(
    persona_dir: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    # The id keeps its history: before Engine 6 the heartbeat's per-memory decay pass erased this value. The pass is gone
    # (the engine fades on read), so a heartbeat leaves every stored value as written. The out-of-range write is B1-4's.
    dream, seed = _write_dream_over_ten(store, hebbian, persona_dir, eng=None)
    assert store.get(dream.id)._stored_emotions == {"love": pytest.approx(18.0)}
    _one_heartbeat(store, hebbian, persona_dir)
    assert store.get(dream.id)._stored_emotions == {"love": pytest.approx(18.0)}
    assert store.get(seed.id)._stored_emotions == {"love": pytest.approx(6.0)}


def test_intended_dream_value_stays_in_range(persona_dir: Path, store: MemoryStore, hebbian: HebbianMatrix) -> None:
    eng = _go_live(persona_dir, store)
    dream, _seed_memory = _write_dream_over_ten(store, hebbian, persona_dir, eng=eng)
    assert store.get(dream.id)._stored_emotions["love"] <= 10.0


def test_today_negative_peer_value_is_erased_by_one_heartbeat(
    persona_dir: Path, store: MemoryStore, hebbian: HebbianMatrix
) -> None:
    # The id keeps its history: before Engine 6 the heartbeat's decay pass erased the negative value. The pass is gone, so
    # a heartbeat leaves the row as written. The negative write is B1-2's.
    memory_id = _write_negative_peer_memory(store)
    assert store.get(memory_id)._stored_emotions["tenderness"] == pytest.approx(-0.3)
    _one_heartbeat(store, hebbian, persona_dir)
    emotions = store.get(memory_id)._stored_emotions
    assert emotions["tenderness"] == pytest.approx(-0.3)
    assert emotions["joy"] == pytest.approx(5.0)


def test_intended_peer_values_are_never_negative(persona_dir: Path, store: MemoryStore) -> None:
    [emotions] = _live_peer_memories(persona_dir, store, '{"tenderness": -0.3, "joy": 5.0}')
    assert emotions == {"tenderness": 0.0, "joy": 5.0}


@_PINNED
def test_today_peer_bucket_caps_summed_size_across_emotions(persona_dir: Path) -> None:
    first, second = _peer_changes(persona_dir)
    # One change is scaled down to the cap across both emotions.
    assert first == {"tenderness": pytest.approx(CAP / 2), "joy": pytest.approx(CAP / 2)}
    # A different emotion gets no bucket of its own: only what drained in that hour is free.
    assert second == {"love": pytest.approx(CAP / WINDOW_HOURS)}


def test_intended_peer_bucket_does_not_scale(persona_dir: Path, store: MemoryStore) -> None:
    first, second = _live_peer_memories(persona_dir, store, '{"tenderness": 0.5, "joy": 0.5}', '{"love": 0.5}')
    assert first == {"tenderness": 0.5, "joy": 0.5}
    assert second == {"love": 0.5}


@pytest.mark.no_base_failure(reason="pins the absence of a row; main's live branch already writes none, only its off branch did")
def test_the_extractor_and_reconcile_write_no_emotion_row(persona_dir: Path, store: MemoryStore) -> None:
    """Pass-2 and reconcile deltas fold into the engine only: no `monologue_emotion` or `self_model_reconcile` row."""
    _go_live(persona_dir, store, joy=5.0)
    _apply_emotion_delta({"joy": -0.2}, persona_dir)
    reconcile_self_read(persona_dir=persona_dir, action="revise", channel="joy", delta=-0.2)
    assert store.list_by_type("monologue_emotion", active_only=False) == []
    assert store.list_by_type("self_model_reconcile", active_only=False) == []
