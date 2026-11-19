"""Every reader of how the kindled feels should give the answer the store gives.

About a dozen readers each pool memory rows for the emotion strengths their own way (different windows, one
that sums where the rest take a maximum). This file puts each reader beside the one answer the store gives,
``aggregate_state`` over every active row, on two stores:

* D1: an emotive memory (``nostalgia``) older than the newest 200 emotion-bearing rows, then 30 emotionless rows.
* D2: only two emotive memories, buried under more than 50 newer emotionless rows.

For each reader the map it pools is captured by wrapping ``aggregate_state`` at the module that calls it (the
heartbeat summary, which sums by hand, is captured at ``HeartbeatEngine._peak_emotion``). A cell then checks, in
order, that the wrapper fired once (1), that the dataset shows the disagreement it exists to show (2), that the
reader still reads exactly the window it is known to read (3), and only then asserts the one thing under test (4):
the captured map equals the full-store pool. Checks 1 to 3 use ``pytest.fail`` so a strict expected failure can
never hide them; check 4 is the only ``AssertionError``.

Twelve cells are strict expected failures today because those readers disagree with the store. Nothing marks a
persona live: the marks stay through the reader switch-over, which points the same cells at the engine.
"""

from __future__ import annotations

import dataclasses
import importlib
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from brain.bridge import persona_state, supervisor
from brain.bridge.provider import FakeProvider
from brain.chat import prompt
from brain.emotion import aggregate
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
_REAL = aggregate.aggregate_state
_REAL_PEAK = HeartbeatEngine._peak_emotion

_TOOL_EMOTIONAL_STATE = importlib.import_module("brain.tools.impls.get_emotional_state")
_TOOL_BODY_STATE = importlib.import_module("brain.tools.impls.get_body_state")


# ── datasets ─────────────────────────────────────────────────────────────────


def _row(i: int, base: datetime, memory_type: str, emotions: dict[str, float]) -> Memory:
    mem = Memory.create_new(
        content=f"same-answer row {i}", memory_type=memory_type, domain="us", emotions=emotions
    )
    return dataclasses.replace(mem, created_at=base + timedelta(minutes=i))


def _fill(persona_dir: Path, rows: list[tuple[str, dict[str, float]]]) -> tuple[MemoryStore, list[Memory]]:
    """A file-backed store holding ``rows`` oldest first, one minute apart, starting eight hours ago."""
    base = datetime.now(UTC) - timedelta(hours=8)
    store = MemoryStore(persona_dir / "memories.db")
    mems = [_row(i, base, memory_type, emotions) for i, (memory_type, emotions) in enumerate(rows)]
    for mem in mems:
        store.create(mem)
    return store, mems


def _d1(persona_dir: Path) -> tuple[MemoryStore, list[Memory]]:
    """231 rows: the oldest emotive row is rank 201 among emotion-bearing rows, so a 200-row window misses it."""
    rows: list[tuple[str, dict[str, float]]] = [
        ("episodic", {"nostalgia": 6.0}),
        ("episodic", {"joy": 2.0, "relief": 4.0}),
    ]
    rows += [("episodic", {"joy": 2.0})] * 199
    rows += [("heartbeat", {})] * 30
    return _fill(persona_dir, rows)


def _d2(persona_dir: Path) -> tuple[MemoryStore, list[Memory]]:
    """62 rows: two emotive rows, then 60 emotionless ones, so the newest 50 rows hold no emotion at all."""
    rows: list[tuple[str, dict[str, float]]] = [
        ("episodic", {"nostalgia": 6.0, "joy": 5.0}),
        ("episodic", {"joy": 3.0}),
    ]
    rows += [("heartbeat", {})] * 60
    return _fill(persona_dir, rows)


_DATASETS: dict[str, Callable[[Path], tuple[MemoryStore, list[Memory]]]] = {"D1": _d1, "D2": _d2}


def _pool(rows: list[Memory]) -> dict[str, float]:
    return dict(_REAL(rows).emotions)


def _newest_first(mems: list[Memory]) -> list[Memory]:
    return list(reversed(mems))


def _window_pool(window: str, mems: list[Memory]) -> dict[str, float]:
    """The map a reader with this window should pool, computed from the rows the builder made."""
    newest = _newest_first(mems)
    if window == "emotive200":
        return _pool([m for m in newest if m.emotions][:200])
    if window == "all50":
        return _pool(newest[:50])
    if window == "all":
        return _pool(mems)
    if window == "sum":
        names = {name for m in mems for name in m.emotions}
        return {name: sum(m.emotions.get(name, 0.0) for m in mems) for name in names}
    raise ValueError(f"unknown window {window!r}")


# ── surfaces ─────────────────────────────────────────────────────────────────


class _Bus:
    """An event bus that takes whatever is published and drops it."""

    def publish(self, event: object) -> None:
        pass


def _persona_emotions(c: SimpleNamespace) -> None:
    persona_state._build_emotions(c.persona_dir)


def _persona_body(c: SimpleNamespace) -> None:
    persona_state._build_body(c.persona_dir, now=c.now)


def _prompt_body(c: SimpleNamespace) -> None:
    prompt._build_body_block(c.store, c.persona_dir)


def _get_emotional_state(c: SimpleNamespace) -> None:
    hebbian = HebbianMatrix(":memory:")
    try:
        get_emotional_state(store=c.store, hebbian=hebbian, persona_dir=c.persona_dir)
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
    supervisor._self_model_reflect(c.persona_dir, provider=FakeProvider(), event_bus=_Bus(), now=c.now)


def _rest(c: SimpleNamespace) -> None:
    supervisor._run_initiate_review_tick(c.persona_dir, FakeProvider(), _Bus())


def _maker(c: SimpleNamespace) -> None:
    maker_sources.current_emotional_intensity(c.store)


def _review(c: SimpleNamespace) -> None:
    soul_review._current_emotional_summary(c.store)


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
            c.persona_dir, elapsed_seconds=900.0, edges_pruned=0, dream_id=None
        )
    finally:
        hebbian.close()


# surface id -> (module that calls aggregate_state, the call, the window the site is known to read)
_SURFACES: dict[str, tuple[object, Callable[[SimpleNamespace], None], str]] = {
    "persona_emotions": (aggregate, _persona_emotions, "emotive200"),
    "persona_body": (aggregate, _persona_body, "emotive200"),
    "prompt_body": (aggregate, _prompt_body, "emotive200"),
    "get_emotional_state": (_TOOL_EMOTIONAL_STATE, _get_emotional_state, "all"),
    "get_body_state": (_TOOL_BODY_STATE, _get_body_state, "emotive200"),
    "drivers": (aggregate, _drivers, "emotive200"),
    "selfmodel": (aggregate, _selfmodel, "emotive200"),
    "rest": (aggregate, _rest, "emotive200"),
    "maker": (maker_sources, _maker, "emotive200"),
    "review": (aggregate, _review, "all50"),
    "reflex": (reflex_module, _reflex, "all"),
    "research": (aggregate, _research, "all"),
    "heartbeat": (HeartbeatEngine, _heartbeat, "sum"),
}

# ── cells and their marks ────────────────────────────────────────────────────

_WINDOW_200 = "reads the newest 200 emotion-bearing rows and misses the oldest emotive row (`nostalgia`)"
_EXPECTED_FAILURES: dict[tuple[str, str], str] = {
    ("persona_emotions", "D1"): _WINDOW_200,
    ("persona_body", "D1"): _WINDOW_200,
    ("prompt_body", "D1"): _WINDOW_200,
    ("get_body_state", "D1"): _WINDOW_200,
    ("drivers", "D1"): _WINDOW_200,
    ("selfmodel", "D1"): _WINDOW_200,
    ("rest", "D1"): _WINDOW_200,
    ("maker", "D1"): _WINDOW_200,
    (
        "review",
        "D1",
    ): "reads the newest 50 rows with no emotion filter; 30 emotionless rows take part of that window",
    (
        "heartbeat",
        "D1",
    ): "sums each emotion over every memory where the other surfaces take the maximum (joy 400.0 against 2.0)",
    ("review", "D2"): "the newest 50 rows are all emotionless, so it sees nothing",
    (
        "heartbeat",
        "D2",
    ): "sums each emotion over every memory where the other surfaces take the maximum (joy 8.0 against 5.0)",
}
_NO_D2 = {"prompt_body", "get_body_state"}


def _cells() -> list:
    cells = []
    for surface in _SURFACES:
        for dataset in _DATASETS:
            if dataset == "D2" and surface in _NO_D2:
                continue
            reason = _EXPECTED_FAILURES.get((surface, dataset))
            marks = (
                [pytest.mark.xfail(strict=True, raises=AssertionError, reason=reason)] if reason else []
            )
            cells.append(pytest.param(surface, dataset, id=f"{surface}-{dataset}", marks=marks))
    return cells


CELLS = _cells()


def _wrapper(captures: list):
    def wrapper(memories):
        rows = list(memories)
        result = _REAL(rows)
        captures.append(([m.id for m in rows], dict(result.emotions)))
        return result

    return wrapper


def _peak_wrapper(captures: list):
    def wrapper(emotions, *, default=("curiosity", 5)):
        captures.append((None, dict(emotions)))
        return _REAL_PEAK(emotions, default=default)

    return wrapper


@pytest.mark.no_base_failure(
    reason=(
        "same-answer test: its cells pass, or fail as strict expected failures, on the old code by design; "
        "it pins the old readers until phase C re-points it"
    )
)
@pytest.mark.parametrize(("surface", "dataset"), CELLS)
def test_surface_matches_the_store_pool(surface, dataset, tmp_path, monkeypatch):
    module, call, window = _SURFACES[surface]
    captures: list = []
    if surface == "heartbeat":
        monkeypatch.setattr(HeartbeatEngine, "_peak_emotion", staticmethod(_peak_wrapper(captures)))
    else:
        monkeypatch.setattr(module, "aggregate_state", _wrapper(captures))

    store, mems = _DATASETS[dataset](tmp_path)
    try:
        context = SimpleNamespace(store=store, persona_dir=tmp_path, now=datetime.now(UTC))
        try:
            call(context)
        except AssertionError as exc:
            pytest.fail(f"{surface}: the call itself raised AssertionError: {exc}")
    finally:
        store.close()

    full = _pool(mems)

    # (1) the wrapper fired once, on rows (or keys) of this dataset
    if len(captures) != 1:
        pytest.fail(f"{surface}-{dataset}: expected exactly one capture, got {len(captures)}")
    ids, captured = captures[0]
    if ids is not None and not set(ids) <= {m.id for m in mems}:
        pytest.fail(f"{surface}-{dataset}: captured rows that are not in the dataset")
    if ids is None and not set(captured) <= {name for m in mems for name in m.emotions}:
        pytest.fail(f"{surface}-{dataset}: the heartbeat summary used emotion names the dataset does not hold")

    # (2) the dataset shows the disagreement it exists to show
    if dataset == "D1":
        seen_by_a_window = _window_pool("emotive200", mems)
        which = "newest-200 emotion-bearing pool"
    else:
        seen_by_a_window = _pool(_newest_first(mems)[:50])
        which = "newest-50 pool"
    if full == seen_by_a_window:
        pytest.fail(f"{dataset}: the full pool equals the {which}, so the dataset no longer shows the case")

    # (3) the reader still reads exactly the window it is known to read
    expected_for_site = _window_pool(window, mems)
    if captured != expected_for_site:
        pytest.fail(
            f"{surface}-{dataset}: captured {captured} but the {window} window pools {expected_for_site}; "
            "the site no longer reads the window it is known to read"
        )

    # (4) the one assertion under test
    assert captured == full
