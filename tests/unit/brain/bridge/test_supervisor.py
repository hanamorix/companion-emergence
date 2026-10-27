"""Tests for brain.bridge.supervisor.run_folded — the bridge supervisor
thread that runs session-cleanup AND autonomous heartbeat cadences."""

from __future__ import annotations

import contextlib
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from brain.bridge import persisted_cadence
from brain.bridge.events import EventBus
from brain.bridge.provider import FakeProvider
from brain.bridge.supervisor import (
    _ROLLING_LOG_POLICIES,
    _run_clustering_tick,
    _run_heartbeat_tick,
    _run_initiate_review_tick,  # noqa: F401 — imported to assert symbol exists
    _run_log_rotation_tick,
    _run_voice_reflection_tick,  # noqa: F401 — imported to assert symbol exists
    run_folded,
)
from brain.engines import interest_sweep


@pytest.fixture(autouse=True)
def _no_leaked_supervisor_threads():
    """Fail the test that leaks a live thread, not the innocent one downstream.

    Many tests in this file start a supervisor thread and none used try/finally.
    An assertion between `t.start()` and `stop.set()` therefore leaves a daemon
    thread running for the whole session; the heartbeat-failure test's thread
    logs ERROR every tick, which is how it broke an unrelated maker test on
    Windows CI (#110). This turns that silent cross-test poisoning into a local
    failure at the point of the leak.
    """
    before = {t.ident for t in threading.enumerate()}
    yield
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        leaked = [t for t in threading.enumerate()
                  if t.ident not in before and t.is_alive()]
        if not leaked:
            return
        time.sleep(0.05)
    assert not leaked, (
        f"test leaked {len(leaked)} live thread(s): {[t.name for t in leaked]} — "
        "stop the supervisor thread in a finally so a failed assertion cannot "
        "leave it running and logging into later tests"
    )


def _wait_until(pred, *, timeout: float = 30.0, interval: float = 0.02) -> bool:
    """Poll until `pred()` holds or the deadline passes; return the final value.

    Replaces fixed `time.sleep(N)` windows before an assertion (#110). A
    constant sleep encodes an assumption about how fast the runner is: the
    Windows CI box is routinely ~25% slower than a local machine, and these
    supervisor loops then had not finished a pass when the assertion ran.
    Polling is also FASTER on a quick runner — it stops as soon as the
    condition holds instead of always burning the full window.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return bool(pred())


# run_folded's pre-loop startup (vocab repair, soul-candidate repair, store opens) can
# pass 5 s on a slow CI runner (#272). A generous budget costs nothing on a fast one:
# every wait below returns as soon as its condition holds.
_STARTUP_BUDGET_S = 30.0


@contextlib.contextmanager
def _running_supervisor(stop: threading.Event, runner):
    """Run `runner` (a closure calling run_folded(stop, ...)) in a daemon thread and
    ALWAYS stop + join it on the way out, so an assertion that fails inside the block
    can't leak a live supervisor into later tests (#110, #272). Yields the thread."""
    t = threading.Thread(target=runner, daemon=True)
    t.start()
    try:
        yield t
    finally:
        stop.set()
        t.join(timeout=_STARTUP_BUDGET_S)


def _ran_a_few_passes(bus: _CapturingBus) -> bool:
    """The loop really ran (each pass publishes supervisor_tick) — so a "never fires
    when disabled" check isn't vacuous on a slow runner whose startup outlasts a sleep."""
    return _wait_until(
        lambda: sum(e.get("type") == "supervisor_tick" for e in bus.events) >= 3,
        timeout=_STARTUP_BUDGET_S,
    )


def test_audit_logs_registered_for_rotation() -> None:
    """The append-only audit logs must be bounded by a rolling policy, not unbounded."""
    registered = {name for name, _ in _ROLLING_LOG_POLICIES}
    assert "chat_usage.jsonl" in registered
    assert "file_access.jsonl" in registered
    assert "attunement_errors.jsonl" in registered


def test_consolidation_archive_is_never_rotated() -> None:
    """#430 C6: the forgetting exemption reads every merge row of consolidation_archive.jsonl
    (``brain.soul.candidates.merge_target_map``). Rotating or archiving it away un-protects merged
    soul moments, so it must stay out of both rotation tables."""
    from brain.bridge.supervisor import _YEARLY_ARCHIVE_LOGS
    from brain.engines.consolidation import _ARCHIVE_FILENAME

    assert _ARCHIVE_FILENAME not in {name for name, _ in _ROLLING_LOG_POLICIES}
    assert _ARCHIVE_FILENAME not in {name for name, _ in _YEARLY_ARCHIVE_LOGS}


def _persona_dir(tmp_path: Path) -> Path:
    """Minimal persona dir for the supervisor to operate against."""
    p = tmp_path / "test-persona"
    p.mkdir()
    (p / "active_conversations").mkdir()
    (p / "persona_config.json").write_text('{"provider": "fake", "searcher": "noop"}')
    return p


def _seed_overdue(persona_dir: Path, filename: str) -> None:
    """ram-spike-fix INC-9 (S22/S34): a MISSING gated-job cadence file is
    created as "last ran now" (the job waits one full interval), so a test
    that wants a gated job to fire on its first pass seeds an OVERDUE file."""
    persisted_cadence.save_cadence(
        persona_dir,
        filename,
        persisted_cadence.CadenceState(next_at=datetime.now(UTC) - timedelta(hours=1)),
    )


def _seed_self_model_overdue(persona_dir: Path) -> None:
    """Same, for self-model articulation's own cadence module (S29)."""
    from brain.self_model import cadence as self_model_cadence

    self_model_cadence.save(
        persona_dir,
        self_model_cadence.SelfModelCadenceState(
            next_reflection_at=datetime.now(UTC) - timedelta(hours=1), consecutive_failures=0
        ),
    )


class _CapturingBus:
    """In-process bus stand-in that records every published event.

    The real EventBus.publish() requires a bound asyncio loop and is a
    no-op otherwise; tests don't run a loop so we substitute a duck-typed
    bus that records every dict the supervisor publishes.
    """

    def __init__(self) -> None:
        self.events: list[dict] = []

    def publish(self, event: dict) -> None:
        self.events.append(event)


def test_run_folded_exits_when_stop_event_is_set(tmp_path: Path) -> None:
    """The supervisor loop honors stop_event and returns cleanly."""
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    stop.set()  # already set — loop should exit immediately
    # Should return without hanging
    run_folded(
        stop,
        persona_dir=persona_dir,
        provider=FakeProvider(),
        event_bus=bus,
        tick_interval_s=0.1,
        heartbeat_interval_s=None,
    )


def test_run_folded_skips_heartbeat_when_disabled(tmp_path: Path) -> None:
    """heartbeat_interval_s=None disables the autonomous cadence."""
    persona_dir = _persona_dir(tmp_path)
    bus = _CapturingBus()
    stop = threading.Event()
    fired: list[int] = []

    def fake_heartbeat(*args, **kwargs):
        fired.append(1)

    def runner():
        with patch("brain.bridge.supervisor._run_heartbeat_tick", side_effect=fake_heartbeat):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=None,
            )

    with _running_supervisor(stop, runner) as t:
        assert _ran_a_few_passes(bus), "supervisor loop never ran"
    assert not t.is_alive()
    assert len(fired) == 0, "heartbeat fired even though disabled"


def test_run_folded_fires_heartbeat_after_interval(tmp_path: Path) -> None:
    """When heartbeat_interval_s elapses, _run_heartbeat_tick is called.

    Drives the loop in a worker thread so we can stop it from the test
    once we've observed the heartbeat fire. tick_interval_s is small
    so the loop's stop_event.wait() exits promptly when we set stop.
    """
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    fired = threading.Event()

    def fake_heartbeat(*args, **kwargs):
        fired.set()

    def runner():
        with patch("brain.bridge.supervisor._run_heartbeat_tick", side_effect=fake_heartbeat):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=0.0,  # fire on first iteration
            )

    with _running_supervisor(stop, runner) as t:
        assert fired.wait(timeout=_STARTUP_BUDGET_S), "heartbeat never fired despite zero interval"
    assert not t.is_alive(), "supervisor loop did not exit after stop_event"


def test_run_folded_retries_heartbeat_soon_after_a_skip_not_a_full_interval(
    tmp_path: Path,
) -> None:
    """Round-2 red-team MINOR (INC-7, self-acknowledged gap in
    8-harness.md): `last_heartbeat_at` must advance ONLY when
    `_heartbeat_and_felt_time` actually does something (returns non-None),
    not on a skip (e.g. a reply in flight, S21) — otherwise a skip would
    reset the full `heartbeat_interval_s` countdown, and a busy chat
    session could defer the heartbeat indefinitely. Proven here by making
    `_heartbeat_and_felt_time` return None for its first two calls, then a
    real (non-None) sentinel, and recording the `last_heartbeat_at` VALUE
    the supervisor hands to each call (its 4th positional arg) rather than
    the wall-clock gap between calls: two skip calls must see the exact
    same `last_heartbeat_at` (proving a skip left it untouched), and the
    call after the first success must see an ADVANCED value relative to
    what the success itself saw (proving a real run does advance it).
    This is deterministic — no real-time margin — because it inspects the
    value the code under test computed, not how long the test took to
    observe it (a wall-clock-gap version of this assertion flaked on a
    loaded macOS CI runner: an outer-loop pass was delayed long enough to
    make a real skip's gap look like a full-interval gap by coincidence)."""
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    seen_last_heartbeat_at: list[float] = []
    call_results: list[bool] = []  # True once a call is allowed to "succeed"

    heartbeat_interval_s = 0.3
    tick_interval_s = 0.05

    def fake_heartbeat_and_felt_time(persona_dir, provider, event_bus, last_heartbeat_at):
        seen_last_heartbeat_at.append(last_heartbeat_at)
        # First 2 calls simulate a reply-in-flight skip (S21): return None,
        # doing nothing. From the 3rd call on, simulate a real pass.
        succeeded = len(seen_last_heartbeat_at) >= 3
        call_results.append(succeeded)
        return object() if succeeded else None

    def runner():
        with patch(
            "brain.bridge.supervisor._heartbeat_and_felt_time",
            side_effect=fake_heartbeat_and_felt_time,
        ):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=tick_interval_s,
                heartbeat_interval_s=heartbeat_interval_s,
            )

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    try:
        deadline = time.monotonic() + _STARTUP_BUDGET_S
        while time.monotonic() < deadline and len(seen_last_heartbeat_at) < 4:
            time.sleep(0.02)
        assert len(seen_last_heartbeat_at) >= 4, "did not observe enough heartbeat attempts"
    finally:
        stop.set()
        t.join(timeout=_STARTUP_BUDGET_S)
    assert not t.is_alive()

    # The two SKIPPED calls (both None) must have been handed the exact
    # SAME `last_heartbeat_at` value — proving the skip left it untouched
    # instead of restarting the full heartbeat_interval_s countdown.
    first_success_idx = call_results.index(True)
    assert first_success_idx >= 2, "expected at least 2 skips before the first success"
    assert seen_last_heartbeat_at[1] == seen_last_heartbeat_at[0], (
        f"a skipped attempt was handed a different last_heartbeat_at "
        f"({seen_last_heartbeat_at[1]!r} vs {seen_last_heartbeat_at[0]!r}) — "
        f"a skip must not advance it, or a busy session could defer the "
        f"heartbeat indefinitely"
    )

    # The call AFTER the first success must be handed an ADVANCED
    # last_heartbeat_at relative to the value the success itself saw —
    # proving a real (non-skipped) run DOES advance it.
    assert first_success_idx + 1 < len(seen_last_heartbeat_at), (
        "no call observed after the first success"
    )
    assert seen_last_heartbeat_at[first_success_idx + 1] > seen_last_heartbeat_at[first_success_idx], (
        "last_heartbeat_at was not advanced after a real (non-skipped) heartbeat run"
    )


def test_heartbeat_failure_does_not_break_supervisor_loop(tmp_path: Path) -> None:
    """A heartbeat exception is fault-isolated; supervisor keeps ticking.

    We let the patched heartbeat raise on every call, observe at least
    two attempts (proving the loop survived the first one), then stop.
    """
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    attempts: list[int] = []
    second_attempt = threading.Event()

    def boom(*args, **kwargs):
        attempts.append(1)
        if len(attempts) >= 2:
            second_attempt.set()
        raise RuntimeError("simulated heartbeat failure")

    def runner():
        with patch("brain.bridge.supervisor._run_heartbeat_tick", side_effect=boom):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=0.0,
            )

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    try:
        # #110: this assertion sits between start and stop. Without the finally,
        # a failure here skips stop.set() and the daemon thread runs for the rest
        # of the session — raising every tick and logging ERROR into whatever
        # test is running later. That is what failed an unrelated maker test
        # eight minutes downstream on Windows CI.
        assert second_attempt.wait(timeout=30.0), (
            "loop did not survive first heartbeat exception"
        )
    finally:
        stop.set()
        t.join(timeout=30.0)
    assert not t.is_alive()


def test_run_heartbeat_tick_publishes_result_event(tmp_path: Path) -> None:
    """_run_heartbeat_tick fires HeartbeatEngine.run_tick and publishes."""
    persona_dir = _persona_dir(tmp_path)
    bus = _CapturingBus()

    # Fake HeartbeatEngine returns a result we can predict
    fake_result = MagicMock(
        memories_decayed=0,
        edges_pruned=0,
        dream_id=None,
        reflex_fired=(),
        research_fired=None,
        growth_emotions_added=0,
        reflex_error=None,
        growth_error=None,
    )
    fake_engine_class = MagicMock(
        return_value=MagicMock(run_tick=MagicMock(return_value=fake_result))
    )
    with (
        patch("brain.engines.heartbeat.HeartbeatEngine", fake_engine_class),
        patch("brain.emotion.persona_loader.load_persona_vocabulary"),
    ):
        _run_heartbeat_tick(persona_dir, FakeProvider(), bus)

    assert any(e.get("type") == "heartbeat_tick" for e in bus.events), (
        "heartbeat_tick event was not published"
    )


# ---------------------------------------------------------------------------
# Snapshot sweep + finalize cadence tests (Phase B sticky sessions)
# ---------------------------------------------------------------------------


def test_supervisor_snapshot_sweep_keeps_session_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a snapshot sweep, the session must remain in _SESSIONS and its
    buffer file on disk."""
    from brain.chat.session import create_session, get_session, reset_registry
    from brain.ingest.buffer import ingest_turn

    # ram-spike-fix INC-9 (S66/S72/S83): the snapshot/prune is a gated job of
    # the central cadence function with NO per-session age threshold — it
    # runs at any idle pass while a buffer has un-extracted turns. Chat is
    # idle by default (conftest's autouse cli_throttle reset).

    reset_registry()
    persona_dir = _persona_dir(tmp_path)
    sess = create_session(persona_dir.name)
    # The registry entry reflects the buffered turn (as the chat path would):
    # it is created before this supervisor starts and no message is seen in
    # the process, so an EMPTY one would be prunable under S84 — this test is
    # about the snapshot not evicting a session that has turns.
    sess.turns = 1
    sid = sess.session_id
    old_ts = (datetime.now(UTC) - timedelta(minutes=6)).isoformat()
    ingest_turn(
        persona_dir, {"session_id": sid, "speaker": "user", "text": "earlier", "ts": old_ts}
    )

    bus = _CapturingBus()

    stop = threading.Event()
    t = threading.Thread(
        target=run_folded,
        args=(stop,),
        kwargs={
            "persona_dir": persona_dir,
            "provider": FakeProvider(),
            "event_bus": bus,
            "tick_interval_s": 0.1,
            "heartbeat_interval_s": None,
            "soul_review_interval_s": None,
            "finalize_interval_s": None,
        },
    )
    t.start()
    _wait_until(lambda: "session_snapshot" in [e.get("type") for e in bus.events])
    stop.set()
    t.join(timeout=30.0)

    buf = persona_dir / "active_conversations" / f"{sid}.jsonl"
    assert buf.exists(), "snapshot sweep must NOT delete the buffer"
    assert get_session(sid) is not None, "snapshot sweep must NOT evict from _SESSIONS"
    types = [e.get("type") for e in bus.events]
    assert "session_snapshot" in types
    assert "session_closed" not in types
    reset_registry()


def test_supervisor_snapshot_sweep_defers_while_chat_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ram-spike-fix INC-6, C4(c) runtime coverage for the session-snapshot/
    prune caller specifically (closes a stage-6 code red-team label-audit gap
    — the prior test for this caller was static text-presence, not a runtime
    "flip not-idle, assert it defers" test): a session old enough to be
    snapshotted, but chat marked recently active, must NOT be swept."""
    from brain.bridge import cli_throttle
    from brain.chat.session import create_session, reset_registry
    from brain.ingest.buffer import ingest_turn

    cli_throttle.mark_interactive_active()

    reset_registry()
    persona_dir = _persona_dir(tmp_path)
    sess = create_session(persona_dir.name)
    sid = sess.session_id
    old_ts = (datetime.now(UTC) - timedelta(minutes=6)).isoformat()
    ingest_turn(
        persona_dir, {"session_id": sid, "speaker": "user", "text": "earlier", "ts": old_ts}
    )

    bus = _CapturingBus()
    stop = threading.Event()
    t = threading.Thread(
        target=run_folded,
        args=(stop,),
        kwargs={
            "persona_dir": persona_dir,
            "provider": FakeProvider(),
            "event_bus": bus,
            "tick_interval_s": 0.1,
            "heartbeat_interval_s": None,
            "soul_review_interval_s": None,
            "finalize_interval_s": None,
        },
    )
    t.start()
    # Give it several ticks to prove the deferral is not just timing luck.
    _wait_until(
        lambda: len([e for e in bus.events if e.get("type") == "supervisor_tick"]) >= 3
    )
    stop.set()
    t.join(timeout=30.0)

    types = [e.get("type") for e in bus.events]
    assert "session_snapshot" not in types, (
        "snapshot must defer while chat is not idle, even for a stale-enough session"
    )
    reset_registry()


def test_supervisor_tick_embeds_backlogged_memory(tmp_path: Path) -> None:
    """Live-path proof for the embedding backfill wiring (F1 #259 increment
    3: writes land on the ROW, not the old embeddings.db cache): a memory
    committed straight via MemoryStore.create() (bypassing the ingest
    pipeline's own embed-on-write, like ~11 real write sites do) gets
    embedded by the supervisor's own idle-gated per-tick block — not a
    mock-call assertion, an actual vector landing in the row's `embedding`
    column. Chat is idle throughout (the suite-wide cli_throttle reset
    leaves it idle by default — see the busy-defers test below for the
    other half of the idle gate)."""
    from brain.memory.store import Memory, MemoryStore

    persona_dir = _persona_dir(tmp_path)
    store = MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
    memory = Memory.create_new(
        content="a memory long enough to clear the embed min-chars floor",
        memory_type="conversation",
        domain="us",
    )
    store.create(memory)
    store.close()

    bus = _CapturingBus()
    stop = threading.Event()
    t = threading.Thread(
        target=run_folded,
        args=(stop,),
        kwargs={
            "persona_dir": persona_dir,
            "provider": FakeProvider(),
            "event_bus": bus,
            "tick_interval_s": 0.1,
            "heartbeat_interval_s": None,
            "soul_review_interval_s": None,
            "finalize_interval_s": None,
        },
    )
    t.start()
    _wait_until(lambda: "supervisor_tick" in [e.get("type") for e in bus.events])
    stop.set()
    t.join(timeout=30.0)

    store2 = MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
    try:
        row = store2._conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", (memory.id,)
        ).fetchone()
        assert row is not None
        assert row["embedding"] is not None
    finally:
        store2.close()


def test_supervisor_embedding_backfill_defers_while_chat_active(tmp_path: Path) -> None:
    """Idle-gate proof (F1 #259 increment 3): before this increment the
    backfill ran on EVERY base tick unconditionally, unlike every other
    background maintenance cadence in this file. With chat marked
    interactive-active throughout, `cli_throttle.background_slot()` must
    deny the backfill's slot on every tick, so a backlogged memory stays
    un-embedded across several ticks — mirrors
    tests/unit/brain/bridge/test_background_yields.py's
    mark_interactive_active() pattern for the other idle-gated engines."""
    from brain.bridge import cli_throttle
    from brain.memory.store import Memory, MemoryStore

    persona_dir = _persona_dir(tmp_path)
    store = MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
    memory = Memory.create_new(
        content="a memory long enough to clear the embed min-chars floor",
        memory_type="conversation",
        domain="us",
    )
    store.create(memory)
    store.close()

    cli_throttle.mark_interactive_active()

    bus = _CapturingBus()
    stop = threading.Event()
    t = threading.Thread(
        target=run_folded,
        args=(stop,),
        kwargs={
            "persona_dir": persona_dir,
            "provider": FakeProvider(),
            "event_bus": bus,
            "tick_interval_s": 0.1,
            "heartbeat_interval_s": None,
            "soul_review_interval_s": None,
            "finalize_interval_s": None,
        },
    )
    t.start()
    _wait_until(
        lambda: len([e for e in bus.events if e.get("type") == "supervisor_tick"]) >= 3
    )
    stop.set()
    t.join(timeout=30.0)

    store2 = MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
    try:
        row = store2._conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", (memory.id,)
        ).fetchone()
        assert row is not None
        assert row["embedding"] is None, (
            "backfill must be idle-gated: deferred on every tick while chat is active"
        )
    finally:
        store2.close()


def test_supervisor_clustering_tick_writes_row_and_centroids_not_embeddings_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live-path proof for F1 #259 increment 4's clustering rewire:
    `_run_clustering_tick` sources vectors from the warm matrix over
    `memories.db` (via `run_clustering_pass`) and writes `cluster_id`/
    `cluster_model_id` onto the row plus the `cluster_centroids` table —
    not the old `MemoryClusterStore`/`embeddings.db` side file this tick
    used to open unconditionally on every firing. Seeds rows directly
    (bypassing the real embed provider, same convention as
    `brain/memory/clustering.py`'s own tests) and aligns the active
    embedding tier to the seeded model_id so the matrix's lazy build
    actually finds them."""
    from brain.bridge import model_tier
    from brain.memory.clustering import MIN_VECTORS_TO_CLUSTER
    from brain.memory.store import Memory, MemoryStore

    test_model_id = "fake-clustering-tick-model"
    monkeypatch.setitem(model_tier.LOCAL_MODEL_IDS, model_tier.TIER_EMBEDDING, test_model_id)

    persona_dir = _persona_dir(tmp_path)
    store = MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
    ids: list[str] = []
    for i in range(MIN_VECTORS_TO_CLUSTER):
        m = Memory.create_new(
            content=f"clustering tick memory number {i}",
            memory_type="conversation",
            domain="us",
        )
        store.create(m)
        vec = np.full(384, float(i), dtype=np.float32)
        store._conn.execute(
            "UPDATE memories SET embedding = ?, embedding_model_id = ? WHERE id = ?",
            (vec.tobytes(), test_model_id, m.id),
        )
        ids.append(m.id)
    store._conn.commit()
    store.close()

    _run_clustering_tick(persona_dir)

    store2 = MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
    try:
        for mid in ids:
            row = store2._conn.execute(
                "SELECT cluster_id, cluster_model_id FROM memories WHERE id = ?", (mid,)
            ).fetchone()
            assert row is not None
            assert row["cluster_id"] is not None
            assert row["cluster_model_id"] == test_model_id
        n_centroids = store2._conn.execute(
            "SELECT COUNT(*) FROM cluster_centroids WHERE model_id = ?", (test_model_id,)
        ).fetchone()[0]
        assert n_centroids > 0
    finally:
        store2.close()

    # The old side file must never have been created — this tick no longer
    # opens embeddings.db at all.
    assert not (persona_dir / "embeddings.db").exists()


def test_supervisor_startup_deletes_legacy_embeddings_db_when_fully_embedded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Live-path proof for the F1 #259 increment 9 startup hook: run_folded's
    one-shot, run-once fail-safe deletion of the legacy embeddings.db file
    (spec §5 / S9, invariant I9) actually runs during startup — even with
    stop_event already set, so the main tick loop body never executes at
    all, proving the hook lives before that loop, not inside it — and
    actually deletes the stale file once every active row already carries a
    current-model embedding."""
    from brain.bridge import model_tier
    from brain.memory import embeddings as embeddings_mod
    from brain.memory.store import Memory, MemoryStore

    provider = embeddings_mod.FakeEmbeddingProvider(dim=384)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)
    # The deletion gate reads the current model id without building a provider.
    monkeypatch.setitem(model_tier.LOCAL_MODEL_IDS, model_tier.TIER_EMBEDDING, provider.model_id())

    persona_dir = _persona_dir(tmp_path)
    store = MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
    memory = Memory.create_new(
        content="a memory long enough to clear the embed min-chars floor",
        memory_type="conversation",
        domain="us",
    )
    store.create(memory)
    store.embed_row(memory.id, memory.content)
    store.close()

    db_path = persona_dir / "embeddings.db"
    db_path.write_bytes(b"legacy sqlite content, never actually read")

    bus = EventBus()
    stop = threading.Event()
    stop.set()  # already set — the main tick loop body must never run

    run_folded(
        stop,
        persona_dir=persona_dir,
        provider=FakeProvider(),
        event_bus=bus,
        tick_interval_s=0.1,
        heartbeat_interval_s=None,
    )

    assert not db_path.exists()


def test_supervisor_startup_defers_legacy_embeddings_db_deletion_with_backlog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same startup hook, other half: a still-un-embedded active row means
    the backlog gate is non-empty, so the legacy file must be left in place
    for the next startup rather than deleted early."""
    from brain.bridge import model_tier
    from brain.memory import embeddings as embeddings_mod
    from brain.memory.store import Memory, MemoryStore

    provider = embeddings_mod.FakeEmbeddingProvider(dim=384)
    monkeypatch.setattr(embeddings_mod, "build_embedding_provider", lambda: provider)
    # The deletion gate reads the current model id without building a provider.
    monkeypatch.setitem(model_tier.LOCAL_MODEL_IDS, model_tier.TIER_EMBEDDING, provider.model_id())

    persona_dir = _persona_dir(tmp_path)
    store = MemoryStore(str(persona_dir / "memories.db"), integrity_check=False)
    memory = Memory.create_new(
        content="a memory long enough to clear the embed min-chars floor, left unembedded",
        memory_type="conversation",
        domain="us",
    )
    store.create(memory)  # deliberately NOT embedded
    store.close()

    db_path = persona_dir / "embeddings.db"
    db_path.write_bytes(b"legacy sqlite content, never actually read")

    bus = EventBus()
    stop = threading.Event()
    stop.set()

    run_folded(
        stop,
        persona_dir=persona_dir,
        provider=FakeProvider(),
        event_bus=bus,
        tick_interval_s=0.1,
        heartbeat_interval_s=None,
    )

    assert db_path.exists()


def test_supervisor_finalize_cadence_drops_old_sessions(tmp_path: Path) -> None:
    """The hourly finalize cadence at 24h silence runs, evicts the session from _SESSIONS and publishes
    session_finalized. Finalize is extraction-only: it never deletes or moves the buffer (the rollover owns that). Here
    the fake provider's reply never parses, so after three recorded failures finalize sets the failing chunk aside in
    poison/ and moves the cursor past it; the buffer, now fully dealt with, stays until a rollover reaps it, which the
    weekly rollover does once its oldest turn is a week old."""
    import json as _json

    from brain.chat.compaction import _ROLLOVER_QUIET_GAP, _WEEKLY_ROLLOVER_AGE
    from brain.chat.rollover import maybe_weekly_rollover
    from brain.chat.session import create_session, get_session, reset_registry
    from brain.ingest.buffer import ingest_turn, read_backoff, read_cursor
    from brain.memory.hebbian import HebbianMatrix
    from brain.memory.store import MemoryStore

    reset_registry()
    persona_dir = _persona_dir(tmp_path)
    sess = create_session(persona_dir.name)
    sid = sess.session_id
    old_ts = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    ingest_turn(
        persona_dir, {"session_id": sid, "speaker": "user", "text": "ancient", "ts": old_ts}
    )

    bus = _CapturingBus()

    stop = threading.Event()
    t = threading.Thread(
        target=run_folded,
        args=(stop,),
        kwargs={
            "persona_dir": persona_dir,
            "provider": FakeProvider(),
            "event_bus": bus,
            "tick_interval_s": 0.1,
            "heartbeat_interval_s": None,
            "soul_review_interval_s": None,
            "finalize_after_hours": 24.0,
            "finalize_interval_s": 0.05,  # fire near-immediately
        },
    )
    t.start()
    buf = persona_dir / "active_conversations" / f"{sid}.jsonl"
    span = persona_dir / "active_conversations" / "poison" / f"{sid}.span.jsonl"
    _wait_until(lambda: span.exists() and read_cursor(persona_dir, sid) is not None)
    stop.set()
    t.join(timeout=30.0)

    set_aside = [_json.loads(line) for line in span.read_text(encoding="utf-8").splitlines()]
    assert [r["text"] for r in set_aside] == ["ancient"], "finalize must set the failing chunk aside"
    assert read_cursor(persona_dir, sid) == old_ts, "the cursor moves past the chunk set aside"
    assert read_backoff(persona_dir, sid) is None, "the failure record is cleared"
    assert buf.exists(), "finalize never deletes or moves the buffer; the rollover reaps it"
    assert get_session(sid) is None, "finalize must remove from _SESSIONS"
    types = [e.get("type") for e in bus.events]
    assert "session_finalized" in types

    # The buffer does not stay forever: once its oldest turn is a week old, the weekly rollover (the daily compaction job)
    # reaps it, with nothing left to extract.
    store = MemoryStore(persona_dir / "memories.db")
    hebbian = HebbianMatrix(persona_dir / "hebbian.db")
    try:
        new_sid = maybe_weekly_rollover(
            persona_dir, sid, persona_dir.name,
            weekly_age=_WEEKLY_ROLLOVER_AGE, quiet_gap=_ROLLOVER_QUIET_GAP,
            now=datetime.now(UTC) + _WEEKLY_ROLLOVER_AGE, provider=FakeProvider(),
            store=store, hebbian=hebbian,
        )
    finally:
        store.close()
        hebbian.close()
    assert new_sid is not None and not buf.exists(), "the weekly rollover reaps the old buffer"
    reset_registry()


# ---------------------------------------------------------------------------
# Log rotation tick (Phase 4 of JSONL retention plan)
# ---------------------------------------------------------------------------


def _write_oversize_log(persona_dir: Path, name: str, size_bytes: int) -> Path:
    """Write a JSONL log file padded past size_bytes for rotation tests."""
    p = persona_dir / name
    big = "x" * 1000
    import json as _json

    with p.open("w", encoding="utf-8") as f:
        line = _json.dumps({"at": "2026-01-01T00:00:00+00:00", "pad": big}) + "\n"
        # Write enough lines to exceed size_bytes.
        per_line = len(line.encode())
        n = (size_bytes // per_line) + 2
        for _ in range(n):
            f.write(line)
    return p


def test_run_log_rotation_tick_rotates_oversize_heartbeat(tmp_path: Path) -> None:
    """An oversize heartbeats.log.jsonl gets rotated to .1.gz."""
    persona_dir = _persona_dir(tmp_path)
    _write_oversize_log(persona_dir, "heartbeats.log.jsonl", size_bytes=200_000)
    bus = _CapturingBus()

    # Force the cap small so the seeded log trips it.
    _run_log_rotation_tick(
        persona_dir,
        bus,
        rolling_size_bytes=10_000,
    )

    assert (persona_dir / "heartbeats.log.jsonl.1.gz").exists()
    types = [e.get("type") for e in bus.events]
    assert "log_rotation" in types


def test_run_log_rotation_tick_archives_old_year_in_soul_audit(
    tmp_path: Path,
) -> None:
    """soul_audit.jsonl with 2024 entries → archived to .2024.jsonl.gz."""
    persona_dir = _persona_dir(tmp_path)
    audit = persona_dir / "soul_audit.jsonl"
    import json as _json

    with audit.open("w", encoding="utf-8") as f:
        f.write(_json.dumps({"ts": "2024-06-15T00:00:00+00:00", "seq": 0}) + "\n")
        f.write(_json.dumps({"ts": "2026-06-15T00:00:00+00:00", "seq": 1}) + "\n")

    bus = _CapturingBus()
    _run_log_rotation_tick(
        persona_dir,
        bus,
        now=datetime(2026, 5, 11, tzinfo=UTC),
    )

    assert (persona_dir / "soul_audit.2024.jsonl.gz").exists()
    # Active retains only 2026 entries.
    remaining = [_json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
    assert len(remaining) == 1
    assert remaining[0]["seq"] == 1


def test_run_log_rotation_tick_archives_old_year_in_initiate_audit(
    tmp_path: Path,
) -> None:
    """initiate_audit.jsonl with 2024 entries → archived to .2024.jsonl.gz."""
    persona_dir = _persona_dir(tmp_path)
    audit = persona_dir / "initiate_audit.jsonl"
    import json as _json

    with audit.open("w", encoding="utf-8") as f:
        f.write(_json.dumps({"ts": "2024-06-15T00:00:00+00:00", "seq": 0}) + "\n")
        f.write(_json.dumps({"ts": "2026-06-15T00:00:00+00:00", "seq": 1}) + "\n")
    bus = _CapturingBus()
    _run_log_rotation_tick(persona_dir, bus, now=datetime(2026, 5, 11, tzinfo=UTC))
    assert (persona_dir / "initiate_audit.2024.jsonl.gz").exists()


def test_run_log_rotation_tick_no_op_when_within_caps(tmp_path: Path) -> None:
    """Small logs → tick runs cleanly, creates nothing."""
    persona_dir = _persona_dir(tmp_path)
    (persona_dir / "heartbeats.log.jsonl").write_text("{}\n", encoding="utf-8")
    bus = _CapturingBus()
    _run_log_rotation_tick(persona_dir, bus)
    # No archive should exist.
    assert list(persona_dir.glob("*.gz")) == []


def test_run_log_rotation_tick_fault_isolated_per_log(tmp_path: Path) -> None:
    """A failure rotating one log doesn't block other logs in the same tick."""
    persona_dir = _persona_dir(tmp_path)
    # Seed an oversize heartbeats AND an oversize dreams. Patch the rolling
    # rotator to raise on heartbeats only; dreams must still rotate.
    _write_oversize_log(persona_dir, "heartbeats.log.jsonl", 200_000)
    _write_oversize_log(persona_dir, "dreams.log.jsonl", 200_000)
    bus = _CapturingBus()

    real_rotate = __import__(
        "brain.health.log_rotation", fromlist=["rotate_rolling_size"]
    ).rotate_rolling_size

    def selective(log_path, **kw):
        if "heartbeats" in log_path.name:
            raise OSError("simulated disk error")
        return real_rotate(log_path, **kw)

    with patch("brain.bridge.supervisor.rotate_rolling_size", side_effect=selective):
        _run_log_rotation_tick(persona_dir, bus, rolling_size_bytes=10_000)

    # heartbeats archive missing (failed), dreams archive present.
    assert not (persona_dir / "heartbeats.log.jsonl.1.gz").exists()
    assert (persona_dir / "dreams.log.jsonl.1.gz").exists()


def test_run_log_rotation_tick_emits_event_per_rotation(tmp_path: Path) -> None:
    """Each rotation publishes a structured log_rotation event."""
    persona_dir = _persona_dir(tmp_path)
    _write_oversize_log(persona_dir, "heartbeats.log.jsonl", 200_000)
    bus = _CapturingBus()
    _run_log_rotation_tick(persona_dir, bus, rolling_size_bytes=10_000)

    rotations = [e for e in bus.events if e.get("type") == "log_rotation"]
    assert len(rotations) >= 1
    e = rotations[0]
    assert e["log"] == "heartbeats.log.jsonl"
    assert e["action"] == "rotated"


def test_run_folded_fires_log_rotation_after_interval(tmp_path: Path) -> None:
    """run_folded wires log rotation into the cadence loop."""
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    fired = threading.Event()

    def fake_rotation(*args, **kwargs):
        fired.set()

    def runner():
        with patch(
            "brain.bridge.supervisor._run_log_rotation_tick",
            side_effect=fake_rotation,
        ):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=None,
                soul_review_interval_s=None,
                finalize_interval_s=None,
                log_rotation_interval_s=0.0,  # fire on first iteration
            )

    with _running_supervisor(stop, runner) as t:
        assert fired.wait(timeout=_STARTUP_BUDGET_S), "log rotation never fired despite zero interval"
    assert not t.is_alive()


def test_run_folded_self_model_fault_isolated(tmp_path: Path) -> None:
    """A self-model tick that raises must not take down the supervisor loop."""
    persona_dir = _persona_dir(tmp_path)
    _seed_self_model_overdue(persona_dir)
    bus = EventBus()
    stop = threading.Event()
    fired = threading.Event()

    def boom(*args, **kwargs):
        fired.set()
        raise RuntimeError("self-model exploded")

    def runner():
        with patch("brain.bridge.supervisor._run_self_model_tick", side_effect=boom):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=None,
                soul_review_interval_s=None,
                finalize_interval_s=None,
                log_rotation_interval_s=None,
                initiate_review_interval_s=None,
                voice_reflection_interval_s=None,
                self_model_interval_s=0.0,
            )

    with _running_supervisor(stop, runner) as t:
        assert fired.wait(timeout=_STARTUP_BUDGET_S), "self-model tick never fired"
        # Loop survives the crash — give it time to keep ticking, then stop.
        time.sleep(0.2)
    assert not t.is_alive(), "supervisor loop died on a self-model crash (not fault-isolated)"


def test_run_folded_skips_log_rotation_when_disabled(tmp_path: Path) -> None:
    """log_rotation_interval_s=None disables the cadence."""
    persona_dir = _persona_dir(tmp_path)
    bus = _CapturingBus()
    stop = threading.Event()
    fired: list[int] = []

    def fake_rotation(*args, **kwargs):
        fired.append(1)

    def runner():
        with patch(
            "brain.bridge.supervisor._run_log_rotation_tick",
            side_effect=fake_rotation,
        ):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=None,
                soul_review_interval_s=None,
                finalize_interval_s=None,
                log_rotation_interval_s=None,
            )

    with _running_supervisor(stop, runner) as t:
        assert _ran_a_few_passes(bus), "supervisor loop never ran"
    assert not t.is_alive()
    assert fired == [], "log rotation fired even though disabled"


def test_run_folded_fires_voice_reflection_after_interval(tmp_path: Path) -> None:
    """run_folded wires voice reflection into the cadence loop."""
    persona_dir = _persona_dir(tmp_path)
    bus = EventBus()
    stop = threading.Event()
    fired = threading.Event()

    def fake_voice(*args, **kwargs):
        fired.set()

    def runner():
        with patch(
            "brain.bridge.supervisor._run_voice_reflection_tick",
            side_effect=fake_voice,
        ):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=None,
                soul_review_interval_s=None,
                finalize_interval_s=None,
                log_rotation_interval_s=None,
                initiate_review_interval_s=None,
                voice_reflection_interval_s=0.0,
            )

    with _running_supervisor(stop, runner) as t:
        assert fired.wait(timeout=_STARTUP_BUDGET_S), "voice reflection never fired"
    assert not t.is_alive()


def test_run_folded_skips_voice_reflection_when_disabled(tmp_path: Path) -> None:
    persona_dir = _persona_dir(tmp_path)
    bus = _CapturingBus()
    stop = threading.Event()
    fired: list[int] = []

    def fake_voice(*args, **kwargs):
        fired.append(1)

    def runner():
        with patch(
            "brain.bridge.supervisor._run_voice_reflection_tick",
            side_effect=fake_voice,
        ):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=None,
                soul_review_interval_s=None,
                finalize_interval_s=None,
                log_rotation_interval_s=None,
                initiate_review_interval_s=None,
                voice_reflection_interval_s=None,
            )

    with _running_supervisor(stop, runner) as t:
        assert _ran_a_few_passes(bus), "supervisor loop never ran"
    assert not t.is_alive()
    assert fired == []


# ---------------------------------------------------------------------------
# The self-model, initiate-review and interest-sweep cadences are gated jobs of the central cadence function: the
# thread tests that ran a real supervisor to see them "fire when due" or "stay off when disabled" became two
# deterministic checks (#540). What each job does when due is tested in test_central_cadence.py
# (test_c15a_all_17_due_run_in_s55_order_real_table, test_c6_completed_interval_job_advances_by_its_interval,
# test_c21_*); here: the job table has the job exactly when its interval is not None, and run_folded hands every
# interval argument to the table builder unchanged.
# ---------------------------------------------------------------------------

_TABLE_INTERVALS = (
    "finalize_interval_s",
    "initiate_review_interval_s",
    "maintenance_interval_s",
    "self_model_interval_s",
    "compaction_interval_s",
    "calibration_interval_s",
    "interest_sweep_interval_s",
    "judge_selftune_interval_s",
    "clustering_interval_s",
)


def _job_names(persona_dir: Path, **intervals: float | None) -> list[str]:
    from brain.bridge import supervisor as sup

    kwargs = dict.fromkeys(_TABLE_INTERVALS)
    kwargs.update(intervals)
    jobs = sup._build_gated_jobs(
        persona_dir=persona_dir,
        provider=FakeProvider(),
        event_bus=EventBus(),
        is_session_busy=None,
        finalize_after_hours=24.0,
        intensity_drivers=lambda: None,
        tick_stats={"closed_sessions": 0, "pruned_empty_sessions": 0},
        **kwargs,
    )
    return [j.name for j in jobs]


@pytest.mark.parametrize(
    "job, knob",
    [
        ("self_model_articulation", "self_model_interval_s"),
        ("initiate_review", "initiate_review_interval_s"),
        ("interest_sweep", "interest_sweep_interval_s"),
    ],
)
@pytest.mark.parametrize("interval", [0.0, 3600.0])
def test_a_gated_job_is_in_the_table_exactly_when_its_interval_is_not_none(
    tmp_path: Path, job: str, knob: str, interval: float
) -> None:
    persona_dir = _persona_dir(tmp_path)
    assert job not in _job_names(persona_dir, **{knob: None}), "interval None must leave the job out"
    assert job in _job_names(persona_dir, **{knob: interval}), "a number (zero included) must put the job in"


def test_run_folded_hands_every_interval_to_the_job_table_unchanged(tmp_path: Path) -> None:
    """The arguments of run_folded reach the table builder as given (None included: no ``x or default``), and the
    defaults leave the calibration and sweep jobs on. The supervisor thread tests this replaces ran with the defaults,
    so a default of None here would have switched deploy recalibration and calibration off in production."""
    import inspect

    from brain.bridge import supervisor as sup

    given = {
        "soul_review_interval_s": 11.0,
        "finalize_interval_s": 12.0,
        "initiate_review_interval_s": 13.0,
        "self_model_interval_s": 14.0,
        "compaction_interval_s": 15.0,
        "calibration_interval_s": 16.0,
        "interest_sweep_interval_s": 17.0,
        "judge_selftune_interval_s": 18.0,
        "clustering_interval_s": 19.0,
    }
    to_table = {"soul_review_interval_s": "maintenance_interval_s"}  # run_folded's name -> the builder's name
    real_build = sup._build_gated_jobs

    def run_with(where: str, **intervals: float | None) -> dict:
        (tmp_path / where).mkdir()
        seen: list[dict] = []
        stop = threading.Event()
        stop.set()  # the table is built before the loop; a stopped supervisor returns right after

        def spy(**kwargs):
            seen.append(kwargs)
            return real_build(**kwargs)

        with patch("brain.bridge.supervisor._build_gated_jobs", side_effect=spy):
            run_folded(
                stop,
                persona_dir=_persona_dir(tmp_path / where),
                provider=FakeProvider(),
                event_bus=EventBus(),
                tick_interval_s=0.05,
                heartbeat_interval_s=None,
                **intervals,
            )
        assert len(seen) == 1, "run_folded must build the job table exactly once"
        return seen[0]

    got = run_with("given", **given)
    for knob, value in given.items():
        assert got[to_table.get(knob, knob)] == value, knob
    off = run_with("none", **dict.fromkeys(given))
    for knob in given:
        assert off[to_table.get(knob, knob)] is None, knob
    defaults = inspect.signature(run_folded).parameters
    assert defaults["calibration_interval_s"].default is not None
    assert defaults["interest_sweep_interval_s"].default is not None
    assert defaults["initiate_review_interval_s"].default is not None


# ---------------------------------------------------------------------------
# B7 — supervisor passes is_rest_state derived from body energy into
#       _run_initiate_review_tick (fail-open: body read error → False)
# ---------------------------------------------------------------------------

def test_supervisor_initiate_review_tick_passes_rest_state_low_energy(tmp_path: Path) -> None:
    """When body energy is low (≤ _REST_ENERGY_THRESHOLD), supervisor must pass
    is_rest_state=True into run_initiate_review_tick."""
    from unittest.mock import MagicMock, patch

    from brain.body.state import BodyState
    from brain.bridge.supervisor import _run_initiate_review_tick  # noqa: F811
    from brain.initiate.review import _REST_ENERGY_THRESHOLD  # type: ignore[attr-defined]

    persona_dir = _persona_dir(tmp_path)
    bus = _CapturingBus()

    captured: dict[str, object] = {}

    def fake_review_tick(persona_dir, *, provider, voice_template, cap_per_tick, user_presence, is_rest_state=False, should_pause=None, paused_out=None):
        captured["is_rest_state"] = is_rest_state

    low_energy_body = BodyState(
        energy=_REST_ENERGY_THRESHOLD,
        temperature=5,
        exhaustion=4,
        session_hours=0.0,
        days_since_contact=1.0,
        body_emotions={},
        computed_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )

    with (
        patch("brain.bridge.supervisor.run_initiate_review_tick", fake_review_tick),
        patch("brain.body.state.compute_body_state", return_value=low_energy_body),
        patch("brain.bridge.supervisor.compute_user_presence", return_value=None),
    ):
        _run_initiate_review_tick(persona_dir, MagicMock(), bus)

    assert captured.get("is_rest_state") is True, f"expected is_rest_state=True for low energy, got: {captured}"


def test_supervisor_initiate_review_tick_passes_rest_state_active_energy(tmp_path: Path) -> None:
    """When body energy is active (> _REST_ENERGY_THRESHOLD), supervisor must pass
    is_rest_state=False into run_initiate_review_tick."""
    from unittest.mock import MagicMock, patch

    from brain.body.state import BodyState
    from brain.bridge.supervisor import _run_initiate_review_tick  # noqa: F811
    from brain.initiate.review import _REST_ENERGY_THRESHOLD  # type: ignore[attr-defined]

    persona_dir = _persona_dir(tmp_path)
    bus = _CapturingBus()

    captured: dict[str, object] = {}

    def fake_review_tick(persona_dir, *, provider, voice_template, cap_per_tick, user_presence, is_rest_state=False, should_pause=None, paused_out=None):
        captured["is_rest_state"] = is_rest_state

    active_energy_body = BodyState(
        energy=_REST_ENERGY_THRESHOLD + 1,
        temperature=5,
        exhaustion=2,
        session_hours=0.0,
        days_since_contact=1.0,
        body_emotions={},
        computed_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
    )

    with (
        patch("brain.bridge.supervisor.run_initiate_review_tick", fake_review_tick),
        patch("brain.body.state.compute_body_state", return_value=active_energy_body),
        patch("brain.bridge.supervisor.compute_user_presence", return_value=None),
    ):
        _run_initiate_review_tick(persona_dir, MagicMock(), bus)

    assert captured.get("is_rest_state") is False, f"expected is_rest_state=False for active energy, got: {captured}"


def test_supervisor_initiate_review_tick_rest_state_fail_open(tmp_path: Path) -> None:
    """When body state computation raises, supervisor must pass is_rest_state=False
    (fail-open: a body bug must never permanently silence recall-resonance)."""
    from unittest.mock import MagicMock, patch


    persona_dir = _persona_dir(tmp_path)
    bus = _CapturingBus()

    captured: dict[str, object] = {}

    def fake_review_tick(persona_dir, *, provider, voice_template, cap_per_tick, user_presence, is_rest_state=False, should_pause=None, paused_out=None):
        captured["is_rest_state"] = is_rest_state

    with (
        patch("brain.bridge.supervisor.run_initiate_review_tick", fake_review_tick),
        patch("brain.body.state.compute_body_state", side_effect=RuntimeError("body read failed")),
        patch("brain.bridge.supervisor.compute_user_presence", return_value=None),
    ):
        _run_initiate_review_tick(persona_dir, MagicMock(), bus)

    assert captured.get("is_rest_state") is False, (
        f"fail-open violated: expected is_rest_state=False on body error, got: {captured}"
    )


# ---------------------------------------------------------------------------
# Interest sweep (Task 10) — weekly persisted wall-clock cadence. Mirrors the
# maintenance block's load-at-startup / is_due-gate / fully-wrapped-body /
# end-of-block-advance+save discipline (defer #21 pattern).
# ---------------------------------------------------------------------------


def test_run_folded_interest_sweep_advances_cadence_even_when_tick_raises(tmp_path: Path) -> None:
    """run_sweep_tick never raises by its own contract, but the supervisor's
    wrap around it must hold anyway: a raised exception must not stop the
    end-of-block advance+save, and must not kill the supervisor loop."""
    persona_dir = _persona_dir(tmp_path)
    _seed_overdue(persona_dir, interest_sweep.SWEEP_CADENCE_FILE)
    bus = EventBus()
    stop = threading.Event()
    fired = threading.Event()

    def fake_sweep(**kwargs):
        fired.set()
        raise RuntimeError("boom")

    def runner():
        with patch("brain.engines.interest_sweep.run_sweep_tick", side_effect=fake_sweep):
            run_folded(
                stop,
                persona_dir=persona_dir,
                provider=FakeProvider(),
                event_bus=bus,
                tick_interval_s=0.05,
                heartbeat_interval_s=None,
                soul_review_interval_s=None,
                finalize_interval_s=None,
                log_rotation_interval_s=None,
                initiate_review_interval_s=None,
                voice_reflection_interval_s=None,
            )

    with _running_supervisor(stop, runner) as t:
        # #210: wait for the tick itself rather than a fixed 0.3 s window (flaked on windows-latest).
        assert fired.wait(timeout=_STARTUP_BUDGET_S), "interest sweep tick never ran"
    assert not t.is_alive(), "supervisor loop must not die from a tick exception"

    state = persisted_cadence.load_cadence(persona_dir, interest_sweep.SWEEP_CADENCE_FILE)
    assert state.next_at is not None
    assert state.next_at > datetime.now(UTC), "cadence must advance even when the tick raised"


