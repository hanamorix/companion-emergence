"""Every reader of how the kindled feels gives the answer the emotion engine gives.

About a dozen readers each turn the engine's read into the numbers they use (the body block, the persona panels, the
tools, the rest gate, the self-model tick, the maker, the soul review, the reflex and research engines, the felt-time
drivers, the heartbeat summary). This file puts each reader beside the one answer the engine gives: the short averages of
its ``read()``, here the ones an ``emotion_engine_stub`` holds.

Two file-backed stores stand behind the readers, with emotions that CONTRADICT the stub (so a reader that went back to
pooling memory rows would give the stored numbers and fail):

* D1: an emotive memory (``nostalgia``), another (``joy``, ``relief``), then 30 emotionless rows.
* D2: only two emotive memories, buried under more than 50 newer emotionless rows.

For each reader the map it derives and uses is captured at the point where it hands it on (the argument the body
projection receives, what a panel or tool returns, the ``emotions`` the reflex engine evaluates, the state the research
engine builds, what the heartbeat summary takes its peak from). A cell then checks, in order, that exactly one capture
was made (1, after the call), that the stub is set up to show the case (2, before the call: every number of its table is
different from every other, and its last fold is a nonzero interval before the instant the reader is called), and only
then asserts the one thing under test (3): the captured map equals the stub's short averages. Checks 1 and 2 use
``pytest.fail`` so no expected failure could hide them; check 3 is the only ``AssertionError``.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain.body import state as body_state_module
from brain.bridge import persona_state, supervisor
from brain.bridge.provider import FakeProvider
from brain.chat import prompt
from brain.emotion import state as emotion_state_module
from brain.emotion.engine import Reading
from brain.engines import reflex as reflex_module
from brain.engines import research as research_module
from brain.engines.heartbeat import HeartbeatEngine
from brain.health.json_io import write_json
from brain.maker import sources as maker_sources
from brain.memory.hebbian import HebbianMatrix
from brain.memory.store import Memory, MemoryStore
from brain.search.base import NoopWebSearcher
from brain.soul import review as soul_review
from brain.tools.impls.get_body_state import get_body_state
from brain.tools.impls.get_emotional_state import get_emotional_state

# Bound at import, before any wrapper is installed.
_REAL_BODY_STATE = body_state_module.body_state_from_short
_REAL_PEAK = HeartbeatEngine._peak_emotion
_REAL_EVALUATE = reflex_module.ReflexEngine._evaluate
_REAL_EMOTIONAL_STATE = emotion_state_module.EmotionalState

_TOOL_BODY_STATE = importlib.import_module("brain.tools.impls.get_body_state")


# ── what the engine reads (the stub's table) ─────────────────────────────────

# name -> (short, long, difference). Every number of the table is different from every other (the stub does not
# need `difference` to equal short minus long), so a reader that took the long average or the difference, or mixed up
# two feelings, gives a different map. All are
# exact at one decimal (the panels round to one decimal) and none equals a number the stores below hold.
_STUB_TABLES: dict[str, dict[str, tuple[float, float, float]]] = {
    "D1": {"nostalgia": (2.5, 6.5, -4.0), "joy": (7.5, 3.5, 4.5), "relief": (5.5, 1.5, 3.0)},
    "D2": {"nostalgia": (4.5, 8.5, 2.0), "joy": (6.5, 0.5, -3.0)},
}


def _short_averages(table: dict[str, tuple[float, float, float]]) -> dict[str, float]:
    """The reference: the short average of each feeling, from the test's own table."""
    return {name: short for name, (short, _long, _difference) in table.items()}


# ── datasets (the stores behind the readers) ─────────────────────────────────


def _row(i: int, base: datetime, memory_type: str, emotions: dict[str, float]) -> Memory:
    mem = Memory.create_new(
        content=f"same-answer row {i}", memory_type=memory_type, domain="us", emotions=emotions
    )
    return dataclasses.replace(mem, created_at=base + timedelta(minutes=i))


def _fill(persona_dir: Path, rows: list[tuple[str, dict[str, float]]]) -> MemoryStore:
    """A file-backed store holding ``rows`` oldest first, one minute apart, starting eight hours ago."""
    base = datetime.now(UTC) - timedelta(hours=8)
    store = MemoryStore(persona_dir / "memories.db")
    for i, (memory_type, emotions) in enumerate(rows):
        store.create(_row(i, base, memory_type, emotions))
    return store


def _d1(persona_dir: Path) -> MemoryStore:
    rows: list[tuple[str, dict[str, float]]] = [
        ("episodic", {"nostalgia": 6.0}),
        ("episodic", {"joy": 2.0, "relief": 4.0}),
    ]
    rows += [("heartbeat", {})] * 30
    return _fill(persona_dir, rows)


def _d2(persona_dir: Path) -> MemoryStore:
    """Two emotive rows, then 60 emotionless ones, so the newest 50 rows hold no emotion at all."""
    rows: list[tuple[str, dict[str, float]]] = [
        ("episodic", {"nostalgia": 6.0, "joy": 5.0}),
        ("episodic", {"joy": 3.0}),
    ]
    rows += [("heartbeat", {})] * 60
    return _fill(persona_dir, rows)


_DATASETS: dict[str, Callable[[Path], MemoryStore]] = {"D1": _d1, "D2": _d2}


# ── surfaces ─────────────────────────────────────────────────────────────────


class _Bus:
    """An event bus that takes whatever is published and drops it."""

    def publish(self, event: object) -> None:
        pass


def _capture_body_state(monkeypatch, captures: list, module: object) -> None:
    """Record the short map each call of `body_state_from_short` receives (the argument is what the site derived)."""

    def wrapper(short, **kwargs):
        captures.append(dict(short))
        return _REAL_BODY_STATE(short, **kwargs)

    monkeypatch.setattr(module, "body_state_from_short", wrapper)


def _install_body_state(monkeypatch, captures: list) -> None:
    # The readers that import the projection at call time find the patched attribute of the module.
    _capture_body_state(monkeypatch, captures, body_state_module)


def _install_body_state_in_tool(monkeypatch, captures: list) -> None:
    # `get_body_state` binds the projection at import, so the tool module's own name is patched.
    _capture_body_state(monkeypatch, captures, _TOOL_BODY_STATE)


def _install_nothing(monkeypatch, captures: list) -> None:
    pass


def _persona_emotions(c: SimpleNamespace) -> None:
    c.captures.append(persona_state._build_emotions(c.persona_dir))


def _persona_body(c: SimpleNamespace) -> None:
    persona_state._build_body(c.persona_dir, now=c.now)


def _prompt_body(c: SimpleNamespace) -> None:
    prompt._build_body_block(c.store, c.persona_dir)


def _get_emotional_state(c: SimpleNamespace) -> None:
    hebbian = HebbianMatrix(":memory:")
    try:
        c.captures.append(get_emotional_state(store=c.store, hebbian=hebbian, persona_dir=c.persona_dir)["all"])
    finally:
        hebbian.close()


def _get_body_state(c: SimpleNamespace) -> None:
    hebbian = HebbianMatrix(":memory:")
    try:
        get_body_state(store=c.store, hebbian=hebbian, persona_dir=c.persona_dir)
    finally:
        hebbian.close()


def _drivers(c: SimpleNamespace) -> None:
    supervisor._derive_intensity_drivers(c.persona_dir, 0, 900.0)


def _selfmodel(c: SimpleNamespace) -> None:
    from brain.bridge import cli_throttle

    cli_throttle.reset()  # a fresh persona's tick is due; the throttle must not hold a chat spell from another test
    supervisor._run_self_model_tick(c.persona_dir, provider=FakeProvider(), event_bus=_Bus())


def _rest(c: SimpleNamespace) -> None:
    supervisor._run_initiate_review_tick(c.persona_dir, FakeProvider(), _Bus())


def _maker(c: SimpleNamespace) -> None:
    c.captures.append({"peak": maker_sources.current_emotional_intensity(c.store)})


def _review(c: SimpleNamespace) -> None:
    c.captures.append(soul_review._current_emotional_summary(c.persona_dir))


def _install_reflex(monkeypatch, captures: list) -> None:
    def wrapper(self, arcs, emotions, *args, **kwargs):
        captures.append(dict(emotions))
        return _REAL_EVALUATE(self, arcs, emotions, *args, **kwargs)

    monkeypatch.setattr(reflex_module.ReflexEngine, "_evaluate", wrapper)


def _reflex(c: SimpleNamespace) -> None:
    arc = {
        "name": "same_answer_arc",
        "description": "a one-arc file so the engine reaches its emotion read",
        "trigger": {"joy": 1},
        "days_since_human_min": 0,
        "cooldown_hours": 1.0,
        "action": "generate_journal",
        "output_memory_type": "reflex_journal",
        "prompt_template": "x",
    }
    arcs_path = c.persona_dir / "reflex_arcs.json"
    write_json(arcs_path, {"version": 1, "arcs": [arc]})
    engine = reflex_module.ReflexEngine(
        store=c.store,
        provider=FakeProvider(),
        persona_name="Test",
        persona_system_prompt="You are Test.",
        arcs_path=arcs_path,
        log_path=c.persona_dir / "reflex_log.json",
        default_arcs_path=Path(reflex_module.__file__).parent / "default_reflex_arcs.json",
    )
    engine.run_tick(dry_run=True)


def _install_research(monkeypatch, captures: list) -> None:
    class Recording(_REAL_EMOTIONAL_STATE):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captures.append(dict(self.emotions))

    monkeypatch.setattr(emotion_state_module, "EmotionalState", Recording)


def _research(c: SimpleNamespace) -> None:
    interest = {
        "id": "i1",
        "topic": "marine bioluminescence",
        "pull_score": 7.0,
        "scope": "either",
        "related_keywords": ["marine", "bioluminescence", "ocean"],
        "notes": "",
        "first_seen": "2026-04-01T10:00:00Z",
        "last_fed": "2026-04-15T10:00:00Z",
        "last_researched_at": None,
        "feed_count": 3,
        "source_types": ["manual"],
    }
    interests_path = c.persona_dir / "interests.json"
    write_json(interests_path, {"version": 1, "interests": [interest]})
    engine = research_module.ResearchEngine(
        store=c.store,
        provider=FakeProvider(),
        searcher=NoopWebSearcher(),
        persona_name="Test",
        persona_system_prompt="You are Test.",
        interests_path=interests_path,
        research_log_path=c.persona_dir / "research_log.json",
        default_interests_path=Path(research_module.__file__).parent / "default_interests.json",
    )
    engine.run_tick(dry_run=True, days_since_human_override=0.0)


def _install_peak(monkeypatch, captures: list) -> None:
    def wrapper(emotions, *, default=("curiosity", 5)):
        captures.append(dict(emotions))
        return _REAL_PEAK(emotions, default=default)

    monkeypatch.setattr(HeartbeatEngine, "_peak_emotion", staticmethod(wrapper))


def _heartbeat(c: SimpleNamespace) -> None:
    interests_path = c.persona_dir / "interests.json"
    write_json(interests_path, {"version": 1, "interests": []})
    hebbian = HebbianMatrix(":memory:")
    try:
        engine = HeartbeatEngine(
            store=c.store,
            hebbian=hebbian,
            provider=FakeProvider(),
            state_path=c.persona_dir / "heartbeat_state.json",
            config_path=c.persona_dir / "heartbeat_config.json",
            dream_log_path=c.persona_dir / "dreams.log.jsonl",
            heartbeat_log_path=c.persona_dir / "heartbeats.log.jsonl",
            interests_path=interests_path,
            research_log_path=c.persona_dir / "research_log.json",
            persona_name="Test",
            persona_system_prompt="You are Test.",
        )
        engine._write_daemon_state_for_heartbeat(
            c.persona_dir, elapsed_seconds=900.0, memories_decayed=0, edges_pruned=0, dream_id=None
        )
    finally:
        hebbian.close()


def _top_three_summary(reference: dict[str, float]) -> dict[str, str]:
    """The string the soul review builds from a map: its top three by value, `name:value` to one decimal."""
    top = sorted(reference.items(), key=lambda item: -item[1])[:3]
    return {"summary": ", ".join(f"{name}:{round(value, 1)}" for name, value in top)}


def _as_is(reference: dict[str, float]) -> dict[str, float]:
    return reference


# surface id -> (install the capture, the call, what the capture must equal given the reference)
_SURFACES: dict[str, tuple[Callable, Callable, Callable]] = {
    "persona_emotions": (_install_nothing, _persona_emotions, _as_is),
    "persona_body": (_install_body_state, _persona_body, _as_is),
    "prompt_body": (_install_body_state, _prompt_body, _as_is),
    "get_emotional_state": (_install_nothing, _get_emotional_state, _as_is),
    "get_body_state": (_install_body_state_in_tool, _get_body_state, _as_is),
    "drivers": (_install_body_state, _drivers, _as_is),
    "selfmodel": (_install_body_state, _selfmodel, _as_is),
    "rest": (_install_body_state, _rest, _as_is),
    "maker": (_install_nothing, _maker, lambda ref: {"peak": max(ref.values())}),
    "review": (_install_nothing, _review, _top_three_summary),
    "reflex": (_install_reflex, _reflex, _as_is),
    "research": (_install_research, _research, _as_is),
    "heartbeat": (_install_peak, _heartbeat, _as_is),
}

_NO_D2 = {"prompt_body", "get_body_state"}


def _cells() -> list:
    cells = []
    for surface in _SURFACES:
        for dataset in _DATASETS:
            if dataset == "D2" and surface in _NO_D2:
                continue
            cells.append(pytest.param(surface, dataset, id=f"{surface}-{dataset}"))
    return cells


CELLS = _cells()


def _check_the_stub_shows_the_case(stub, called_at: datetime, surface: str, dataset: str) -> None:
    """Check 2: the stub's numbers are told apart, and its last fold is a nonzero interval before the call."""
    numbers: list[float] = []
    for name, reading in stub.readings.items():
        if len({reading.short, reading.long, reading.difference}) != 3:
            pytest.fail(f"{surface}-{dataset}: {name} does not have distinct short, long and difference values")
        numbers += [reading.short, reading.long, reading.difference]
    if len(set(numbers)) != len(numbers):
        pytest.fail(
            f"{surface}-{dataset}: two numbers of the stub's table are equal, so a reader that took the wrong "
            "field or mixed up two feelings could go unseen"
        )
    if not stub.last_fold_at < called_at:
        pytest.fail(f"{surface}-{dataset}: the stub's last fold is not an interval before the call")


@pytest.mark.no_base_failure(
    reason=(
        "same-answer test, re-pointed at the engine read; its 24 ids are the safety net's (#581) and are unchanged, so "
        "the base check does not judge them. Each cell is shown able to fail by running the file against the base "
        "source, where the readers still take their old branches (recorded in the run's harness notes)"
    )
)
@pytest.mark.parametrize(("surface", "dataset"), CELLS)
def test_surface_matches_the_store_pool(surface, dataset, tmp_path, monkeypatch, emotion_engine_stub):
    # (The id keeps the safety net's name (#581) so the 24 cells stay the same ids; what they compare against is now
    # the engine's short averages.)
    install, call, expected_from = _SURFACES[surface]
    table = _STUB_TABLES[dataset]
    emotion_engine_stub.readings = {
        name: Reading(short=short, long=long, difference=difference, shift=0)
        for name, (short, long, difference) in table.items()
    }
    emotion_engine_stub.high = set(table)
    captures: list = []
    install(monkeypatch, captures)

    store = _DATASETS[dataset](tmp_path)
    try:
        called_at = datetime.now(UTC)
        _check_the_stub_shows_the_case(emotion_engine_stub, called_at, surface, dataset)
        context = SimpleNamespace(store=store, persona_dir=tmp_path, now=called_at, captures=captures)
        try:
            call(context)
        except AssertionError as exc:
            pytest.fail(f"{surface}: the call itself raised AssertionError: {exc}")
    finally:
        store.close()

    # (1) exactly one capture
    if len(captures) != 1:
        pytest.fail(f"{surface}-{dataset}: expected exactly one capture, got {len(captures)}")

    # (3) the one assertion under test
    reference = _short_averages(table)
    captured = captures[0]
    if surface == "review":
        captured = {"summary": captured}
    assert captured == expected_from(reference)
