"""Name-recall fix S85 revised, S91, S92 (spec §2): the COSINE floor is computed
once per process at process start, off the reply path
(`floor_startup.compute_missing_floors`: the bridge's startup thread and
`nell chat --no-bridge` session start); the RERANK floor on first need, in the
background; a failed bootstrap of either is retried in the background on each
incoming message and at the next lull by the central cadence jobs; never more
than one bootstrap in flight per floor. Recall only ever PEEKS the caches.

Also pins the model ids the job and the lookup key on to the ids the real
providers report.

All offline: stubbed fastembed classes, an in-tmp store, injected activity
markers. Synthetic data only.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from brain.bridge import cli_throttle, model_tier, server
from brain.memory import floor_calibration, floor_startup
from brain.memory import reranker as reranker_mod
from brain.memory.embeddings import build_embedding_provider
from brain.memory.reranker import build_reranker_provider, resolve_reranker_model_id
from brain.memory.semantic_recall import run_semantic_recall
from brain.memory.store import MemoryStore
from tests.unit.brain.memory.test_embeddings import _StubTextEmbedding
from tests.unit.brain.memory.test_monologue_last import (
    _QUERY,
    _cosine_floor,
    _ids,
    _install_reranker,
    _Recording,
    _seed,
)


@pytest.fixture(autouse=True)
def _clean_startup_flag():
    floor_startup._inflight.clear()  # noqa: SLF001
    yield
    floor_startup._inflight.clear()  # noqa: SLF001


@pytest.fixture
def background_on(monkeypatch: pytest.MonkeyPatch):
    """Let the on-need / on-message starters spawn real daemon threads (the
    suite-wide default inhibits them)."""
    monkeypatch.setattr(floor_startup, "_background_inhibited", False)


_COSINE_ID = "startup-cosine-id"
_RERANK_ID = "startup-rerank-id"


def _floor_row(model_id: str, key: str, value: float = 0.5) -> dict:
    return {
        key: model_id,
        "floor": value,
        "raw_fit_floor": value,
        "sample_pairs": 6,
        "is_cold_start": True,
        "updated_at": None,
    }


@pytest.fixture
def computes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Both computing functions replaced by recorders that cache like the real
    ones; the model ids pinned to test ids. Returns the ordered event log of
    ("start"/"end", kind)."""
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(floor_startup, "embedder_model_id", lambda: _COSINE_ID)
    monkeypatch.setattr(floor_startup, "reranker_model_id", lambda: _RERANK_ID)

    def _cosine(model_id: str):
        events.append(("start", "cosine"))
        floor_calibration._cosine_bootstrap_floor_cache[model_id] = _floor_row(  # noqa: SLF001
            model_id, "embedder_model_id"
        )
        events.append(("end", "cosine"))
        return dict(floor_calibration._cosine_bootstrap_floor_cache[model_id])  # noqa: SLF001

    def _rerank(model_id: str):
        events.append(("start", "rerank"))
        floor_calibration._bootstrap_floor_cache[model_id] = _floor_row(  # noqa: SLF001
            model_id, "reranker_model_id"
        )
        events.append(("end", "rerank"))
        return dict(floor_calibration._bootstrap_floor_cache[model_id])  # noqa: SLF001

    monkeypatch.setattr(floor_calibration, "get_cosine_bootstrap_floor", _cosine)
    monkeypatch.setattr(floor_calibration, "get_bootstrap_floor", _rerank)
    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kw: object())
    return events


# ---------------------------------------------------------------------------
# compute_missing_floors: the process-start computation
# ---------------------------------------------------------------------------


def test_startup_computes_only_the_cosine_floor_never_the_rerank_one(
    computes, tmp_path: Path
) -> None:
    """S91: nothing loads the reranker at boot."""
    floor_startup.compute_missing_floors(tmp_path)

    assert computes == [("start", "cosine"), ("end", "cosine")]
    assert floor_calibration.peek_cosine_bootstrap_floor(_COSINE_ID) is not None
    assert floor_calibration.peek_bootstrap_floor(_RERANK_ID) is None


def test_startup_skips_a_floor_that_is_calibrated_or_already_cached(
    computes, tmp_path: Path
) -> None:
    store = MemoryStore(tmp_path / "memories.db")
    store.write_cosine_floor(
        _COSINE_ID, floor=0.5, raw_fit_floor=0.5, sample_pairs=300, is_cold_start=False
    )
    store.close()

    floor_startup.compute_missing_floors(tmp_path)
    assert computes == [], "calibrated cosine row: skipped"

    store = MemoryStore(tmp_path / "memories.db")
    store._conn.execute("DELETE FROM cosine_floor_calibration")  # noqa: SLF001
    store._conn.commit()  # noqa: SLF001
    store.close()
    floor_startup.compute_missing_floors(tmp_path)
    assert computes == [("start", "cosine"), ("end", "cosine")]

    computes.clear()
    floor_startup.compute_missing_floors(tmp_path)
    assert computes == [], "cached: nothing to do"


def test_a_failing_cosine_bootstrap_never_raises_and_is_recorded(
    computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _boom(_id):
        raise RuntimeError("embedder exploded")

    monkeypatch.setattr(floor_calibration, "get_cosine_bootstrap_floor", _boom)

    floor_startup.compute_missing_floors(tmp_path)

    assert floor_calibration.peek_cosine_bootstrap_floor(_COSINE_ID) is None
    assert floor_calibration.bootstrap_failed("cosine", _COSINE_ID)
    assert not floor_startup.inflight("cosine")


def test_a_raising_run_helper_never_raises_and_releases_the_slot(
    computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _boom(**kw):
        raise RuntimeError("cosine helper raised")

    monkeypatch.setattr(floor_startup, "run_cosine_floor", _boom)

    floor_startup.compute_missing_floors(tmp_path)

    assert not floor_startup.startup_compute_active()


def test_startup_never_raises_when_the_store_cannot_be_opened(
    computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _boom(*a, **k):
        raise RuntimeError("cannot open")

    monkeypatch.setattr(MemoryStore, "__init__", _boom)

    floor_startup.compute_missing_floors(tmp_path)

    assert computes == []
    assert not floor_startup.startup_compute_active()


def test_the_store_is_closed_before_any_model_is_loaded(
    computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    order: list[str] = []
    real_close = MemoryStore.close
    monkeypatch.setattr(
        MemoryStore, "close", lambda self: (order.append("close"), real_close(self))[1]
    )
    real = floor_calibration.get_cosine_bootstrap_floor
    monkeypatch.setattr(
        floor_calibration,
        "get_cosine_bootstrap_floor",
        lambda mid: (order.append("compute"), real(mid))[1],
    )

    floor_startup.compute_missing_floors(tmp_path)

    assert order.index("close") < order.index("compute")


def test_the_startup_flag_is_set_while_computing_and_cleared_after_even_on_failure(
    computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[bool] = []
    monkeypatch.setattr(
        floor_calibration,
        "get_cosine_bootstrap_floor",
        lambda mid: seen.append(floor_startup.startup_compute_active()) or None,
    )

    floor_startup.compute_missing_floors(tmp_path)

    assert seen == [True]
    assert not floor_startup.startup_compute_active()


def test_a_startup_failure_is_not_retried_by_the_job_until_chat_has_happened(
    computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(floor_calibration, "get_cosine_bootstrap_floor", lambda _id: None)
    marker = {"m": 5.0}

    floor_startup.compute_missing_floors(tmp_path, activity_marker=lambda: marker["m"])

    store = MemoryStore(tmp_path / "memories.db")
    try:
        assert not floor_startup.cosine_floor_due(store, activity_marker=5.0), "same lull"
        assert floor_startup.cosine_floor_due(store, activity_marker=6.0), "next lull"
    finally:
        store.close()


def test_start_background_runs_the_computation_on_a_daemon_thread(computes, tmp_path: Path) -> None:
    thread = floor_startup.start_background(tmp_path, name="floor-bootstrap-test")

    thread.join(timeout=10)

    assert thread.daemon and not thread.is_alive()
    assert ("end", "cosine") in computes


def test_start_background_forwards_the_activity_marker_to_the_thread(
    computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(floor_calibration, "get_cosine_bootstrap_floor", lambda _id: None)

    floor_startup.start_background(tmp_path, activity_marker=lambda: 7.0).join(timeout=10)

    assert not floor_calibration.cosine_bootstrap_due(_COSINE_ID, activity_marker=7.0)
    assert floor_calibration.cosine_bootstrap_due(_COSINE_ID, activity_marker=8.0)


# ---------------------------------------------------------------------------
# The real activity marker (the retry rule's premise) and the startup order
# ---------------------------------------------------------------------------


def test_the_real_chat_activity_marker_changes_when_chat_happens_and_only_then() -> None:
    cli_throttle.reset()
    try:
        idle_marker = cli_throttle.chat_activity_marker()
        cli_throttle.is_chat_idle()
        assert cli_throttle.chat_activity_marker() == idle_marker, "idle checks do not change it"

        cli_throttle.note_user_message(at=100.0)
        after_message = cli_throttle.chat_activity_marker()
        assert after_message != idle_marker

        cli_throttle.note_reply_end(at=160.0)
        after_reply = cli_throttle.chat_activity_marker()
        assert after_reply != after_message

        cli_throttle.mark_interactive_active(at=400.0)
        assert cli_throttle.chat_activity_marker() != after_reply
        for _ in range(3):
            cli_throttle.is_chat_idle(now=10_000.0)
            cli_throttle.slot_available(now=10_000.0)
        assert cli_throttle.chat_activity_marker() == 400.0, "a lull passing does not change it"
    finally:
        cli_throttle.reset()


def test_the_real_marker_drives_the_retry_rule_end_to_end(
    computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cli_throttle.reset()
    try:
        monkeypatch.setattr(floor_calibration, "get_cosine_bootstrap_floor", lambda _id: None)
        floor_startup.compute_missing_floors(
            tmp_path, activity_marker=cli_throttle.chat_activity_marker
        )
        store = MemoryStore(tmp_path / "memories.db")
        try:
            marker = cli_throttle.chat_activity_marker
            assert not floor_startup.cosine_floor_due(store, activity_marker=marker())

            cli_throttle.note_user_message(at=50.0)
            cli_throttle.note_reply_end(at=60.0)

            assert floor_startup.cosine_floor_due(store, activity_marker=marker())
        finally:
            store.close()
    finally:
        cli_throttle.reset()


def test_the_bridge_seeds_the_idle_anchor_before_it_starts_the_floor_thread(
    bridge_persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    order: list[str] = []
    real_seed = cli_throttle.seed_last_message_from_active_conversations

    def _seed(persona_dir):
        order.append("seed")
        return real_seed(persona_dir)

    def _start(persona_dir, *, activity_marker, name="floor-bootstrap"):
        order.append("floor-thread")
        t = threading.Thread(target=lambda: None, daemon=True)
        t.start()
        return t

    monkeypatch.setattr(cli_throttle, "seed_last_message_from_active_conversations", _seed)
    monkeypatch.setattr(floor_startup, "start_background", _start)
    monkeypatch.setattr("brain.bridge.supervisor.run_folded", lambda **kw: None)
    app = server.build_app(
        persona_dir=bridge_persona, client_origin="tests", background_threads=True
    )

    with TestClient(app):
        pass

    assert order == ["seed", "floor-thread"]


# ---------------------------------------------------------------------------
# Recall never computes the rerank bootstrap (the old per-turn hot-path retry)
# ---------------------------------------------------------------------------


def _reranking_store(monkeypatch, tmp_path):
    store = MemoryStore(tmp_path / "memories.db")
    genuine, _ = _seed(store, monkeypatch, [0.9, 0.8, 0.7, 0.6, 0.5, 0.45], [])
    _cosine_floor(store, 0.4)
    _install_reranker(monkeypatch, _Recording({g.content: 9.0 for g in genuine}))
    return store, genuine


def test_a_reranked_turn_without_a_rerank_floor_never_computes_the_bootstrap_on_the_reply_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        floor_calibration,
        "get_bootstrap_floor",
        lambda _id: pytest.fail("the reply path computed the rerank bootstrap"),
    )
    store, genuine = _reranking_store(monkeypatch, tmp_path)

    for _ in range(3):
        result = run_semantic_recall(store, tmp_path, _QUERY)
        assert result is not None and result.path == "cosine", "no rerank floor yet: cosine path"
        assert set(_ids(genuine)) >= set(_ids([*result.full, *result.snippet]))


def test_the_first_reranked_turn_with_no_floor_takes_the_cosine_path_flags_it_and_bootstraps_in_the_background(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, background_on
) -> None:
    """S91: the turn that would rerank finds no rerank floor: it takes the cosine
    path, raises the need flag, and the bootstrap runs on a background thread
    (not on this thread); the next turn reranks once the floor exists."""
    store, genuine = _reranking_store(monkeypatch, tmp_path)
    monkeypatch.setattr(floor_startup, "reranker_model_id", lambda: "fake-reranker")
    threads_seen: list[str] = []
    gate = threading.Event()

    def _bootstrap(model_id: str):
        threads_seen.append(threading.current_thread().name)
        gate.wait(5)
        floor_calibration._bootstrap_floor_cache[model_id] = _floor_row(  # noqa: SLF001
            model_id, "reranker_model_id", 1.0
        )
        return dict(floor_calibration._bootstrap_floor_cache[model_id])  # noqa: SLF001

    monkeypatch.setattr(floor_calibration, "get_bootstrap_floor", _bootstrap)
    started: list = []
    real_request = floor_startup.request_rerank_bootstrap
    monkeypatch.setattr(
        floor_startup,
        "request_rerank_bootstrap",
        lambda: started.append(real_request()) or started[-1],
    )

    first = run_semantic_recall(store, tmp_path, _QUERY)

    assert first is not None and first.path == "cosine"
    (thread,) = started
    assert thread is not None and thread.daemon
    assert ("rerank", "fake-reranker") in floor_calibration._bootstrap_needed, (  # noqa: SLF001
        "the need is flagged"
    )
    assert floor_startup.inflight("rerank"), "the background bootstrap is in flight"
    # while it is still running, another reranked turn neither blocks nor starts a second one
    second_while_running = run_semantic_recall(store, tmp_path, _QUERY)
    assert second_while_running.path == "cosine"
    assert started[-1] is None, "one at a time"
    gate.set()
    thread.join(timeout=10)
    assert threads_seen == [thread.name] and threading.current_thread().name not in threads_seen
    assert not floor_startup.inflight("rerank")

    third = run_semantic_recall(store, tmp_path, _QUERY)
    assert third is not None and third.path == "reranked", "the next turn reranks"


def test_once_a_rerank_floor_exists_the_reranked_path_is_used_without_any_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store, genuine = _reranking_store(monkeypatch, tmp_path)
    monkeypatch.setattr(
        floor_startup,
        "request_rerank_bootstrap",
        lambda: pytest.fail("a floor exists: nothing to request"),
    )
    floor_calibration._bootstrap_floor_cache["fake-reranker"] = _floor_row(  # noqa: SLF001
        "fake-reranker", "reranker_model_id", 1.0
    )

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == "reranked"


def test_a_session_that_never_reranks_never_loads_the_reranker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, background_on
) -> None:
    """S91: a pool too small to rerank never raises the need, so the rerank
    bootstrap never starts."""
    monkeypatch.setattr(
        floor_startup,
        "request_rerank_bootstrap",
        lambda: pytest.fail("this session never reranks"),
    )
    store = MemoryStore(tmp_path / "memories.db")
    _seed(store, monkeypatch, [0.9, 0.8], [])
    _cosine_floor(store, 0.4)
    _install_reranker(monkeypatch, _Recording({}))

    result = run_semantic_recall(store, tmp_path, _QUERY)

    assert result is not None and result.path == "cosine"


# ---------------------------------------------------------------------------
# S92: a failed bootstrap retries in the background on each incoming message
# ---------------------------------------------------------------------------


def _fail_then_succeed(monkeypatch, kind: str, model_id: str):
    """`get_*_bootstrap_floor` that fails until `state["ok"]` is set; records the
    thread each attempt ran on and can be gated."""
    state = {"ok": False, "calls": [], "gate": None}

    def _compute(mid: str):
        state["calls"].append(threading.current_thread().name)
        if state["gate"] is not None:
            state["gate"].wait(5)
        if not state["ok"]:
            return None
        row = _floor_row(mid, "embedder_model_id" if kind == "cosine" else "reranker_model_id")
        cache = (
            floor_calibration._cosine_bootstrap_floor_cache  # noqa: SLF001
            if kind == "cosine"
            else floor_calibration._bootstrap_floor_cache  # noqa: SLF001
        )
        cache[mid] = row
        return dict(row)

    name = "get_cosine_bootstrap_floor" if kind == "cosine" else "get_bootstrap_floor"
    monkeypatch.setattr(floor_calibration, name, _compute)
    return state


@pytest.mark.parametrize("kind", ["cosine", "rerank"])
def test_a_failed_bootstrap_is_retried_in_the_background_on_each_message_until_it_succeeds(
    kind, computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, background_on
) -> None:
    model_id = _COSINE_ID if kind == "cosine" else _RERANK_ID
    state = _fail_then_succeed(monkeypatch, kind, model_id)
    run = floor_startup.run_cosine_floor if kind == "cosine" else floor_startup.run_rerank_floor
    assert run(activity_marker=1.0) is None, "the first attempt failed"
    assert floor_calibration.bootstrap_failed(kind, model_id)
    state["calls"].clear()
    me = threading.current_thread().name

    for _ in range(3):
        threads = floor_startup.on_incoming_message(tmp_path)
        assert len(threads) == 1
        threads[0].join(timeout=10)
        assert state["calls"][-1] == threads[0].name != me, "off the reply path"
    assert len(state["calls"]) == 3, "one attempt per message while it keeps failing"

    state["ok"] = True
    (thread,) = floor_startup.on_incoming_message(tmp_path)
    thread.join(timeout=10)
    assert not floor_calibration.bootstrap_failed(kind, model_id)

    state["calls"].clear()
    assert floor_startup.on_incoming_message(tmp_path) == []
    assert state["calls"] == [], "succeeded: no more retries"


@pytest.mark.parametrize("kind", ["cosine", "rerank"])
def test_never_more_than_one_bootstrap_in_flight_per_floor(
    kind, computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, background_on
) -> None:
    model_id = _COSINE_ID if kind == "cosine" else _RERANK_ID
    state = _fail_then_succeed(monkeypatch, kind, model_id)
    run = floor_startup.run_cosine_floor if kind == "cosine" else floor_startup.run_rerank_floor
    run(activity_marker=1.0)
    state["calls"].clear()
    state["gate"] = threading.Event()

    first = floor_startup.on_incoming_message(tmp_path)
    assert len(first) == 1
    for _ in range(5):
        assert floor_startup.on_incoming_message(tmp_path) == [], "already in flight"
    assert floor_startup.inflight(kind)
    state["gate"].set()
    first[0].join(timeout=10)

    assert len(state["calls"]) == 1, "no concurrent duplicates"
    assert not floor_startup.inflight(kind)


def test_the_two_floors_retry_independently_each_at_most_one_in_flight(
    computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, background_on
) -> None:
    cosine = _fail_then_succeed(monkeypatch, "cosine", _COSINE_ID)
    rerank_calls: list[str] = []
    monkeypatch.setattr(
        floor_calibration, "get_bootstrap_floor", lambda mid: rerank_calls.append(mid) or None
    )
    floor_startup.run_cosine_floor(activity_marker=1.0)
    floor_startup.run_rerank_floor(activity_marker=1.0)
    cosine["calls"].clear()
    rerank_calls.clear()

    threads = floor_startup.on_incoming_message(tmp_path)
    for t in threads:
        t.join(timeout=10)

    assert len(threads) == 2
    assert len(cosine["calls"]) == 1 and len(rerank_calls) == 1


def test_a_message_retry_skips_a_floor_that_is_calibrated(
    computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, background_on
) -> None:
    state = _fail_then_succeed(monkeypatch, "cosine", _COSINE_ID)
    floor_startup.run_cosine_floor(activity_marker=1.0)
    state["calls"].clear()
    store = MemoryStore(tmp_path / "memories.db")
    store.write_cosine_floor(
        _COSINE_ID, floor=0.5, raw_fit_floor=0.5, sample_pairs=300, is_cold_start=False
    )
    store.close()

    for t in floor_startup.on_incoming_message(tmp_path):
        t.join(timeout=10)

    assert state["calls"] == [], "a calibrated row makes the failed bootstrap moot"


def test_a_message_without_a_failed_bootstrap_starts_nothing(
    computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, background_on
) -> None:
    assert floor_startup.on_incoming_message(tmp_path) == []
    assert computes == []


def test_respond_hooks_the_message_retry_before_the_turn_and_survives_a_failing_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`respond()` calls `on_incoming_message(persona_dir)` once per message,
    after the activity marker moved (so the cadence rule sees the chat), and a
    raising hook never touches the turn."""
    import json

    from brain import paths
    from brain.chat.session import reset_registry
    from brain.cli import main

    d = tmp_path / "personas" / "nell"
    d.mkdir(parents=True)
    (d / "persona_config.json").write_text(json.dumps({"provider": "fake", "searcher": "noop"}))
    (d / "emotion_vocabulary.json").write_text(json.dumps({"version": 1, "emotions": []}))
    monkeypatch.setattr(paths, "get_home", lambda: tmp_path)
    monkeypatch.setattr("brain.cli._start_floor_bootstrap_for_direct_chat", lambda *a, **k: None)
    seen: list = []

    def _hook(persona_dir):
        seen.append((persona_dir, cli_throttle.chat_activity_marker()))
        raise RuntimeError("hook exploded")

    monkeypatch.setattr(floor_startup, "on_incoming_message", _hook)
    cli_throttle.reset()
    reset_registry()
    try:
        with patch("builtins.input", side_effect=["one", "two", "exit"]):
            assert main(["chat", "--persona", "nell", "--no-bridge"]) == 0
    finally:
        reset_registry()
        cli_throttle.reset()

    assert [p for p, _ in seen] == [d, d], "once per message"
    assert all(m != float("-inf") for _, m in seen), "the marker had already moved"


# ---------------------------------------------------------------------------
# Rerank due / run: the next-lull retry rule, never raises
# ---------------------------------------------------------------------------


def test_the_rerank_bootstrap_retries_only_after_chat_activity_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        floor_calibration, "get_bootstrap_floor", lambda mid: calls.append(mid) or None
    )
    assert not floor_calibration.rerank_bootstrap_due("rr", activity_marker=1.0), (
        "S91: not wanted until a turn needed it or an attempt failed"
    )
    floor_calibration.note_bootstrap_needed("rerank", "rr")
    assert floor_calibration.rerank_bootstrap_due("rr", activity_marker=1.0)

    assert floor_calibration.run_rerank_bootstrap("rr", activity_marker=1.0) is None

    assert not floor_calibration.rerank_bootstrap_due("rr", activity_marker=1.0)
    assert floor_calibration.rerank_bootstrap_due("rr", activity_marker=2.0)
    assert len(calls) == 1


def test_run_rerank_bootstrap_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_id):
        raise RuntimeError("reranker exploded")

    monkeypatch.setattr(floor_calibration, "get_bootstrap_floor", _boom)

    assert floor_calibration.run_rerank_bootstrap("rr", activity_marker=1.0) is None
    assert not floor_calibration.rerank_bootstrap_due("rr", activity_marker=1.0)


def test_a_reranker_provider_build_failure_is_a_recorded_failed_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(floor_startup, "reranker_model_id", lambda: "rr")

    def _boom(**kw):
        raise RuntimeError("model file missing")

    monkeypatch.setattr(reranker_mod, "build_reranker_provider", _boom)
    monkeypatch.setattr(
        floor_calibration,
        "get_bootstrap_floor",
        lambda _id: pytest.fail("no provider, no bootstrap"),
    )

    assert floor_startup.run_rerank_floor(activity_marker=3.0) is None
    assert not floor_calibration.rerank_bootstrap_due("rr", activity_marker=3.0)
    assert floor_calibration.rerank_bootstrap_due("rr", activity_marker=4.0)


# ---------------------------------------------------------------------------
# The rerank retry job (central cadence)
# ---------------------------------------------------------------------------


def _bootstrap_job(persona_dir: Path):
    from tests.unit.brain.bridge.test_central_cadence import _real_jobs

    return [j for j in _real_jobs(persona_dir) if j.name == "rerank_floor_bootstrap"]


def _pass(persona_dir, jobs, *, idle: bool):
    from datetime import UTC, datetime

    from brain.bridge import central_cadence

    return central_cadence.run_central_pass(
        persona_dir,
        jobs,
        is_idle=lambda: idle,
        slot_available=lambda: True,
        now_wall=lambda: datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_the_rerank_job_retries_a_failed_bootstrap_once_per_lull(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.bridge import supervisor
    from tests.unit.brain.bridge.test_central_cadence import _granting_slot, _persona

    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _granting_slot)
    marker = {"m": 1.0}
    monkeypatch.setattr(cli_throttle, "chat_activity_marker", lambda: marker["m"])
    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kw: object())
    calls: list[str] = []
    monkeypatch.setattr(
        floor_calibration, "get_bootstrap_floor", lambda mid: calls.append(mid) or None
    )
    persona_dir = _persona(tmp_path)
    jobs = _bootstrap_job(persona_dir)
    assert [j.name for j in jobs] == ["rerank_floor_bootstrap"]

    _pass(persona_dir, jobs, idle=True)
    assert calls == [], "S91: never due until a turn needed the floor or an attempt failed"
    floor_calibration.note_bootstrap_needed("rerank", resolve_reranker_model_id())

    _pass(persona_dir, jobs, idle=False)
    assert calls == [], "never inside a chat"
    _pass(persona_dir, jobs, idle=True)
    assert calls == [resolve_reranker_model_id()]
    for _ in range(4):
        _pass(persona_dir, jobs, idle=True)
    assert len(calls) == 1, "same lull: no retry"

    marker["m"] = 2.0
    _pass(persona_dir, jobs, idle=True)
    assert len(calls) == 2, "one retry at the next lull"


@pytest.mark.parametrize("kind", ["cosine", "rerank"])
def test_a_floor_job_is_not_due_while_a_bootstrap_of_that_floor_is_in_flight(
    kind, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.unit.brain.bridge.test_central_cadence import _persona

    persona_dir = _persona(tmp_path)
    (job,) = _bootstrap_job_named(persona_dir, f"{kind}_floor_bootstrap")
    if kind == "rerank":
        floor_calibration.note_bootstrap_needed("rerank", resolve_reranker_model_id())
    monkeypatch.setattr(cli_throttle, "chat_activity_marker", lambda: 1.0)
    assert job.has_work() is True, "test precondition: due when nothing is in flight"

    assert floor_startup.try_begin(kind)
    try:
        assert job.has_work() is False
    finally:
        floor_startup.end(kind)


def _bootstrap_job_named(persona_dir: Path, name: str):
    from tests.unit.brain.bridge.test_central_cadence import _real_jobs

    return [j for j in _real_jobs(persona_dir) if j.name == name]


def test_the_rerank_job_is_not_due_with_a_calibrated_row_or_a_cached_bootstrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.bridge import supervisor
    from tests.unit.brain.bridge.test_central_cadence import _granting_slot, _persona

    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _granting_slot)
    monkeypatch.setattr(reranker_mod, "build_reranker_provider", lambda **kw: object())
    calls: list[str] = []
    monkeypatch.setattr(
        floor_calibration, "get_bootstrap_floor", lambda mid: calls.append(mid) or None
    )
    persona_dir = _persona(tmp_path)
    store = MemoryStore(persona_dir / "memories.db")
    store.write_reranker_floor(
        resolve_reranker_model_id(),
        floor=1.0,
        raw_fit_floor=1.0,
        sample_pairs=300,
        is_cold_start=False,
    )
    store.close()

    _pass(persona_dir, _bootstrap_job(persona_dir), idle=True)

    assert calls == []


# ---------------------------------------------------------------------------
# Wiring: the bridge startup thread and `nell chat --no-bridge`
# ---------------------------------------------------------------------------


@pytest.fixture
def bridge_persona(tmp_path: Path) -> Path:
    p = tmp_path / "test-persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    (p / "persona_config.json").write_text('{"provider": "fake", "searcher": "fake"}')
    return p


def test_the_bridge_lifespan_starts_the_floor_bootstrap_thread_once(
    bridge_persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[dict] = []
    done = threading.Event()

    def _spy(persona_dir, *, activity_marker, name="floor-bootstrap"):
        started.append({"persona_dir": persona_dir, "marker": activity_marker, "name": name})
        t = threading.Thread(target=done.set, daemon=True)
        t.start()
        return t

    monkeypatch.setattr(floor_startup, "start_background", _spy)
    monkeypatch.setattr("brain.bridge.supervisor.run_folded", lambda **kw: None)
    app = server.build_app(
        persona_dir=bridge_persona, client_origin="tests", background_threads=True
    )

    with TestClient(app):
        assert done.wait(5)
        assert app.state.bridge.floor_bootstrap_thread is not None

    assert len(started) == 1
    assert started[0]["persona_dir"] == bridge_persona
    assert started[0]["marker"] is cli_throttle.chat_activity_marker


def test_the_bridge_lifespan_starts_no_floor_thread_when_background_threads_are_off(
    bridge_persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        floor_startup,
        "start_background",
        lambda *a, **k: pytest.fail("threads are off: no floor bootstrap thread"),
    )
    app = server.build_app(
        persona_dir=bridge_persona, client_origin="tests", background_threads=False
    )

    with TestClient(app):
        pass


def test_the_bridge_survives_a_floor_thread_that_cannot_start(
    bridge_persona: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*a, **k):
        raise RuntimeError("no threads for you")

    monkeypatch.setattr(floor_startup, "start_background", _boom)
    monkeypatch.setattr("brain.bridge.supervisor.run_folded", lambda **kw: None)
    app = server.build_app(
        persona_dir=bridge_persona, client_origin="tests", background_threads=True
    )

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def test_direct_chat_start_computes_synchronously_for_a_one_shot_and_in_a_thread_for_the_repl(
    tmp_path: Path,
) -> None:
    from brain.cli import _start_floor_bootstrap_for_direct_chat

    with (
        patch.object(floor_startup, "compute_missing_floors") as compute,
        patch.object(floor_startup, "start_background") as background,
    ):
        _start_floor_bootstrap_for_direct_chat(tmp_path, blocking=True)
        compute.assert_called_once_with(tmp_path, activity_marker=cli_throttle.chat_activity_marker)
        background.assert_not_called()

    with (
        patch.object(floor_startup, "compute_missing_floors") as compute,
        patch.object(floor_startup, "start_background") as background,
    ):
        _start_floor_bootstrap_for_direct_chat(tmp_path, blocking=False)
        compute.assert_not_called()
        background.assert_called_once()
        assert background.call_args.args == (tmp_path,)


def test_direct_chat_start_never_breaks_the_chat(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from brain.cli import _start_floor_bootstrap_for_direct_chat

    with patch.object(floor_startup, "compute_missing_floors", side_effect=RuntimeError("boom")):
        _start_floor_bootstrap_for_direct_chat(tmp_path, blocking=True)

    assert "floor bootstrap not started" in capsys.readouterr().err


def _direct_chat_persona(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import json

    from brain import paths

    d = tmp_path / "personas" / "nell"
    d.mkdir(parents=True)
    (d / "persona_config.json").write_text(json.dumps({"provider": "fake", "searcher": "noop"}))
    (d / "emotion_vocabulary.json").write_text(json.dumps({"version": 1, "emotions": []}))
    monkeypatch.setattr(paths, "get_home", lambda: tmp_path)
    return d


def test_a_one_shot_no_bridge_chat_computes_the_floors_once_before_its_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from brain.chat.session import reset_registry
    from brain.cli import main

    _direct_chat_persona(tmp_path, monkeypatch)
    reset_registry()
    with patch("brain.cli._start_floor_bootstrap_for_direct_chat") as start:
        assert main(["chat", "--persona", "nell", "--no-bridge", "hello"]) == 0
    reset_registry()

    assert start.call_count == 1
    assert start.call_args.kwargs == {"blocking": True}


def test_a_repl_no_bridge_chat_starts_the_floors_once_per_session_not_per_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from brain.chat.session import reset_registry
    from brain.cli import main

    _direct_chat_persona(tmp_path, monkeypatch)
    reset_registry()
    with (
        patch("brain.cli._start_floor_bootstrap_for_direct_chat") as start,
        patch("builtins.input", side_effect=["first turn", "second turn", "exit"]),
    ):
        assert main(["chat", "--persona", "nell", "--no-bridge"]) == 0
    reset_registry()

    assert start.call_count == 1
    assert start.call_args.kwargs == {"blocking": False}


# ---------------------------------------------------------------------------
# The model-id pins: what the job and the lookups key on IS what the providers report
# ---------------------------------------------------------------------------


def test_the_embedder_id_the_bootstrap_keys_on_is_the_real_providers_model_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """S85 pin: `floor_startup.embedder_model_id()` (the tier's model string,
    what the job and the due predicate use) equals `FastEmbedProvider.model_id()`
    as the REAL `build_embedding_provider()` builds it, so the bootstrap's own
    id check can never fail on a mismatch in production (a mismatch would be
    permanent keyword-only). Built through a stubbed fastembed (no model)."""
    monkeypatch.setattr("fastembed.TextEmbedding", _StubTextEmbedding)
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: tmp_path)

    provider = build_embedding_provider()

    assert provider.model_id() == floor_startup.embedder_model_id()
    assert floor_startup.embedder_model_id() == model_tier.model_for_tier(model_tier.TIER_EMBEDDING)


def test_the_embedder_id_pin_follows_a_repointed_tier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("fastembed.TextEmbedding", _StubTextEmbedding)
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: tmp_path)
    monkeypatch.setitem(model_tier.TIER_MODEL, model_tier.TIER_EMBEDDING, "some/other-embedder")

    assert build_embedding_provider().model_id() == floor_startup.embedder_model_id()
    assert floor_startup.embedder_model_id() == "some/other-embedder"


class _StubCrossEncoder:
    def __init__(self, model_name: str, cache_dir: str, lazy_load: bool = False, **kwargs) -> None:
        self.model_name = model_name


@pytest.mark.parametrize("precision", ["fp16", "fp32"])
def test_the_reranker_id_the_bootstrap_keys_on_is_the_real_providers_model_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, precision: str
) -> None:
    """The rerank counterpart: `resolve_reranker_model_id()` (used by the due
    checks and the startup thread) equals `CrossEncoderProvider.model_id()` as
    the REAL `build_reranker_provider()` builds it, for both precisions."""
    monkeypatch.setattr(
        "fastembed.rerank.cross_encoder.text_cross_encoder.TextCrossEncoder", _StubCrossEncoder
    )
    monkeypatch.setattr(reranker_mod, "_register_fp16_reranker_model", lambda *a, **k: None)
    monkeypatch.setattr("brain.paths.get_cache_dir", lambda: tmp_path)
    real_get = reranker_mod.tunables.get_tunable
    monkeypatch.setattr(
        reranker_mod.tunables,
        "get_tunable",
        lambda key, default=None: (
            precision if key == "reranker.precision" else real_get(key, default)
        ),
    )
    reranker_mod._reset_reranker_provider_cache()  # noqa: SLF001

    provider = build_reranker_provider()

    assert provider.model_id() == resolve_reranker_model_id() == floor_startup.reranker_model_id()
    expected = (
        model_tier.MODEL_RERANKER_FP16
        if precision == "fp16"
        else model_tier.model_for_tier(model_tier.TIER_RERANKER)
    )
    assert provider.model_id() == expected


def test_an_unrecognised_precision_override_warns_once_not_on_every_call(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    real_get = reranker_mod.tunables.get_tunable
    monkeypatch.setattr(
        reranker_mod.tunables,
        "get_tunable",
        lambda key, default=None: "fp8" if key == "reranker.precision" else real_get(key, default),
    )
    reranker_mod._warned_precisions.clear()  # noqa: SLF001

    with caplog.at_level("WARNING", logger="brain.memory.reranker"):
        for _ in range(4):
            assert resolve_reranker_model_id() == model_tier.MODEL_RERANKER_FP16

    warnings = [r for r in caplog.records if "unrecognized reranker.precision" in r.getMessage()]
    assert len(warnings) == 1


def test_start_background_claims_the_cosine_slot_before_the_thread_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cadence job must never win a race against the startup thread: the
    flag is already set when `start_background` returns, even before the
    thread has been scheduled."""
    gate = threading.Event()
    seen: list[bool] = []

    def _slow(persona_dir, *, activity_marker, **kw):
        gate.wait(5)

    monkeypatch.setattr(floor_startup, "compute_missing_floors", _slow)
    thread = floor_startup.start_background(tmp_path)
    try:
        seen.append(floor_startup.startup_compute_active())
    finally:
        gate.set()
        thread.join(timeout=5)

    assert seen == [True]


# ---------------------------------------------------------------------------
# Review round 1: the reply thread never waits behind a running bootstrap, the
# request is not per-rerank, no-start states are recorded, moot records are
# cleared, the hook sits inside `respond`'s try, slots are always released
# ---------------------------------------------------------------------------


def _finishes_within(fn, seconds: float = 5.0) -> bool:
    done = threading.Event()

    def _run():
        try:
            fn()
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True).start()
    return done.wait(seconds)


def test_the_reply_thread_helpers_never_wait_behind_a_bootstrap_holding_the_compute_locks(
    computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, background_on
) -> None:
    """The real compute functions hold their locks across a model download; the
    helpers the reply thread calls (`on_incoming_message`,
    `request_rerank_bootstrap`, the state readers) must not use those locks."""
    floor_calibration._record_attempt("cosine", _COSINE_ID, False, 1.0)  # noqa: SLF001
    floor_calibration._record_attempt("rerank", _RERANK_ID, False, 1.0)  # noqa: SLF001
    # the spawned retries would need the compute locks: keep them from starting
    monkeypatch.setattr(floor_startup, "_background_inhibited", True)
    with (
        floor_calibration._cosine_bootstrap_floor_cache_lock,  # noqa: SLF001
        floor_calibration._bootstrap_floor_cache_lock,  # noqa: SLF001
    ):
        assert _finishes_within(lambda: floor_startup.on_incoming_message(tmp_path))
        assert _finishes_within(floor_startup.request_rerank_bootstrap)
        assert _finishes_within(lambda: floor_calibration.bootstrap_failed("cosine", _COSINE_ID))
        assert _finishes_within(lambda: floor_calibration.note_bootstrap_needed("rerank", "x"))
        assert _finishes_within(
            lambda: floor_calibration.rerank_bootstrap_due(_RERANK_ID, activity_marker=2.0)
        )
        assert _finishes_within(
            lambda: floor_calibration.cosine_bootstrap_due(_COSINE_ID, activity_marker=2.0)
        )


def test_the_first_need_request_is_not_repeated_per_rerank_once_a_bootstrap_failed(
    computes, monkeypatch: pytest.MonkeyPatch, background_on
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        floor_calibration, "get_bootstrap_floor", lambda mid: calls.append(mid) or None
    )

    first = floor_startup.request_rerank_bootstrap()
    assert first is not None
    first.join(timeout=10)
    assert len(calls) == 1 and floor_calibration.bootstrap_failed("rerank", _RERANK_ID)

    for _ in range(4):
        assert floor_startup.request_rerank_bootstrap() is None, "retries belong to the message"
    assert len(calls) == 1


def test_a_startup_that_could_not_even_try_is_recorded_so_the_message_retry_picks_it_up(
    computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, background_on
) -> None:
    real_init = MemoryStore.__init__
    monkeypatch.setattr(
        MemoryStore, "__init__", lambda *a, **k: (_ for _ in ()).throw(OSError("no"))
    )

    floor_startup.compute_missing_floors(tmp_path)

    assert floor_calibration.bootstrap_failed("cosine", _COSINE_ID), "recorded, not silent"
    monkeypatch.setattr(MemoryStore, "__init__", real_init)
    for t in floor_startup.on_incoming_message(tmp_path):
        t.join(timeout=10)
    assert ("end", "cosine") in computes, "the next message computed it"


def test_a_moot_failed_record_is_cleared_once_a_calibrated_row_exists(
    computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, background_on
) -> None:
    floor_calibration._record_attempt("cosine", _COSINE_ID, False, 1.0)  # noqa: SLF001
    store = MemoryStore(tmp_path / "memories.db")
    store.write_cosine_floor(
        _COSINE_ID, floor=0.5, raw_fit_floor=0.5, sample_pairs=300, is_cold_start=False
    )
    store.close()

    (thread,) = floor_startup.on_incoming_message(tmp_path)
    thread.join(timeout=10)

    assert not floor_calibration.bootstrap_failed("cosine", _COSINE_ID)
    assert floor_startup.on_incoming_message(tmp_path) == [], "later messages start nothing"
    assert computes == []


def test_the_retry_body_closes_its_store(
    computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, background_on
) -> None:
    closed: list[bool] = []
    real_close = MemoryStore.close
    monkeypatch.setattr(
        MemoryStore, "close", lambda self: (closed.append(True), real_close(self))[1]
    )
    floor_calibration._record_attempt("cosine", _COSINE_ID, False, 1.0)  # noqa: SLF001

    for t in floor_startup.on_incoming_message(tmp_path):
        t.join(timeout=10)

    assert closed, "the retry's store was closed"


def test_a_background_attempt_records_the_registered_activity_marker(
    computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, background_on
) -> None:
    monkeypatch.setattr(floor_calibration, "get_cosine_bootstrap_floor", lambda _id: None)
    monkeypatch.setattr(floor_calibration, "get_bootstrap_floor", lambda _id: None)
    floor_startup.set_activity_marker_provider(lambda: 9.0)
    floor_calibration._record_attempt("cosine", _COSINE_ID, False, 1.0)  # noqa: SLF001

    for t in floor_startup.on_incoming_message(tmp_path):
        t.join(timeout=10)
    request = floor_startup.request_rerank_bootstrap()
    request.join(timeout=10)

    assert not floor_calibration.cosine_bootstrap_due(_COSINE_ID, activity_marker=9.0)
    assert floor_calibration.cosine_bootstrap_due(_COSINE_ID, activity_marker=10.0)
    assert not floor_calibration.rerank_bootstrap_due(_RERANK_ID, activity_marker=9.0)
    assert floor_calibration.rerank_bootstrap_due(_RERANK_ID, activity_marker=10.0)


def test_start_background_registers_the_marker_provider_and_refuses_a_taken_slot(
    computes, tmp_path: Path
) -> None:
    assert floor_startup.try_begin("cosine")
    try:
        assert floor_startup.start_background(tmp_path, activity_marker=lambda: 3.0) is None
    finally:
        floor_startup.end("cosine")
    assert floor_startup._current_marker() == 3.0  # noqa: SLF001, the provider was registered

    thread = floor_startup.start_background(tmp_path, activity_marker=lambda: 4.0)
    assert thread is not None
    thread.join(timeout=10)
    assert floor_startup._current_marker() == 4.0  # noqa: SLF001


def test_slots_are_always_released_and_a_taken_slot_is_refused(
    computes, monkeypatch: pytest.MonkeyPatch, background_on
) -> None:
    # a spawned body that raises releases its slot
    thread = floor_startup._spawn("cosine", lambda: 1 / 0, name="boom")  # noqa: SLF001
    thread.join(timeout=10)
    assert not floor_startup.inflight("cosine")

    # run_guarded releases on an exception, and refuses when the slot is taken
    with pytest.raises(ZeroDivisionError):
        floor_startup.run_guarded("rerank", lambda **kw: 1 / 0)
    assert not floor_startup.inflight("rerank")
    ran: list[int] = []
    assert floor_startup.try_begin("rerank")
    try:
        assert floor_startup.run_guarded("rerank", lambda **kw: ran.append(1)) is False
        assert ran == []
    finally:
        floor_startup.end("rerank")
    assert floor_startup.run_guarded("rerank", lambda **kw: ran.append(1)) is True

    # a thread that cannot start releases its slot
    monkeypatch.setattr(
        threading.Thread, "start", lambda self: (_ for _ in ()).throw(RuntimeError("no threads"))
    )
    with pytest.raises(RuntimeError):
        floor_startup._spawn("cosine", lambda: None, name="nostart")  # noqa: SLF001
    assert not floor_startup.inflight("cosine")
    with pytest.raises(RuntimeError):
        floor_startup.start_background(Path("."))
    assert not floor_startup.inflight("cosine")


def test_the_cadence_job_runs_through_the_in_flight_slot(
    computes, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from brain.bridge import supervisor
    from tests.unit.brain.bridge.test_central_cadence import _granting_slot, _persona

    monkeypatch.setattr(supervisor.cli_throttle, "background_slot", _granting_slot)
    persona_dir = _persona(tmp_path)
    (job,) = _bootstrap_job_named(persona_dir, "cosine_floor_bootstrap")
    from brain.bridge.central_cadence import JobOutcome

    assert floor_startup.try_begin("cosine")
    try:
        assert job.run() is JobOutcome.SKIPPED, "another cosine bootstrap is in flight"
        assert computes == []
    finally:
        floor_startup.end("cosine")
    assert job.run() is JobOutcome.COMPLETED
    assert ("end", "cosine") in computes


def _engine_persona(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import json

    from brain import paths

    d = tmp_path / "personas" / "nell"
    d.mkdir(parents=True)
    (d / "persona_config.json").write_text(json.dumps({"provider": "fake", "searcher": "noop"}))
    (d / "emotion_vocabulary.json").write_text(json.dumps({"version": 1, "emotions": []}))
    monkeypatch.setattr(paths, "get_home", lambda: tmp_path)
    return d


def test_the_message_hook_runs_before_the_turn_and_inside_the_reply_bookkeeping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook runs before `_respond_inner`, and a BaseException from it still
    lets `note_reply_end()` run (the in-flight counter is never left stuck)."""
    from brain.chat import engine

    d = _engine_persona(tmp_path, monkeypatch)
    order: list[str] = []
    monkeypatch.setattr(floor_startup, "on_incoming_message", lambda pd: order.append("hook"))
    monkeypatch.setattr(engine, "_respond_inner", lambda *a, **k: order.append("turn") or "done")
    cli_throttle.reset()
    try:
        assert engine.respond(d, "hi", store=None, hebbian=None, provider=None) == "done"
        assert order == ["hook", "turn"]
        assert not cli_throttle.reply_in_flight()

        def _interrupt(pd):
            raise KeyboardInterrupt

        monkeypatch.setattr(floor_startup, "on_incoming_message", _interrupt)
        with pytest.raises(KeyboardInterrupt):
            engine.respond(d, "hi", store=None, hebbian=None, provider=None)
        assert not cli_throttle.reply_in_flight(), "note_reply_end ran despite the BaseException"
    finally:
        cli_throttle.reset()
